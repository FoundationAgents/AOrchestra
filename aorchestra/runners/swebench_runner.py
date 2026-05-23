"""SWE-bench Runner with MainAgent orchestration for aorchestra."""
from __future__ import annotations

import asyncio
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from base.engine.async_llm import LLMsConfig, create_llm_instance
from base.engine.logs import logger
from benchmark.common.env import BasicInfo, Environment
from benchmark.common.runner import Runner, StepRecord, LevelResult
from benchmark.benchmark import Benchmark, LevelSpec
from benchmark.bench_swebench import SWEBenchConfig, SWEBenchEnvironment
from benchmark.swebench.data_loader import SWEBenchDataLoader, SWEBenchInstance
from aorchestra.main_agent import MainAgent
from aorchestra.prompts.swebench import SWEBenchMainAgentPrompt
from aorchestra.tools.delegate import DelegateTaskTool
from aorchestra.tools.submit import SubmitTool
from aorchestra.config import SWEBenchOrchestraConfig


class SWEBenchSubAgentRunner(Runner):
    """Runner for SubAgent (standard agent-environment loop) for SWE-bench."""
    
    async def run(self, agent, env: Environment) -> LevelResult:
        """Run SubAgent with standard interaction loop."""
        import inspect
        from base.engine.logs import LogLevel
        
        logger.info(f"[SWEBenchSubAgentRunner] Starting SubAgent execution")
        
        try:
            info = env.get_basic_info()
            agent.reset(info)

            reset_result = env.reset()
            obs = await reset_result if inspect.isawaitable(reset_result) else reset_result

            history = []
            total_reward = 0.0
            graded_reward: Optional[float] = None
            max_steps = info.max_steps

            for t in range(max_steps):
                current_step = t + 1
                logger.log_to_file(LogLevel.INFO, f"Environment Observation:{obs}")
                
                try:
                    if self.step_timeout:
                        step_result = await asyncio.wait_for(
                            agent.step(
                                observation=obs,
                                history=history,
                                current_step=current_step,
                                max_steps=max_steps,
                            ),
                            timeout=self.step_timeout,
                        )
                    else:
                        step_result = await agent.step(
                            observation=obs,
                            history=history,
                            current_step=current_step,
                            max_steps=max_steps,
                        )
                except asyncio.TimeoutError:
                    logger.error(f"Agent step timed out after {self.step_timeout} seconds")
                    step_record = StepRecord(
                        observation=obs,
                        action={"error": "step_timeout"},
                        reward=0.0,
                        raw_response="step timeout",
                        done=True,
                        info={"error": "step_timeout"},
                        raw_input=None,
                    )
                    history.append(step_record)
                    break

                if isinstance(step_result, (list, tuple)):
                    if len(step_result) == 3:
                        action, raw_response, raw_input = step_result
                    elif len(step_result) == 2:
                        action, raw_response = step_result
                        raw_input = None
                    else:
                        raise ValueError(f"agent.step returned {len(step_result)} values")
                else:
                    raise TypeError(f"agent.step returned unsupported type: {type(step_result)}")
                
                logger.info(f"\n[SWEBench SubAgent Step {current_step}/{max_steps}] ACTION: {action}")

                obs_next, reward, done, step_info = await env.step(action)

                step_graded = bool(
                    step_info.get("submitted") or step_info.get("max_steps_reached")
                )

                step_record = StepRecord(
                    observation=obs,
                    action=action,
                    reward=reward if step_graded else 0.0,
                    raw_response=raw_response,
                    done=done,
                    info=step_info,
                    raw_input=raw_input,
                )
                history.append(step_record)
                if step_graded:
                    graded_reward = float(reward)
                obs = obs_next

                if done:
                    break

            if graded_reward is not None:
                total_reward = graded_reward

            usage_summary = agent.llm.get_usage_summary()
            result = LevelResult(
                model=usage_summary.get("model", ""),
                total_reward=total_reward,
                steps=len(history),
                done=history[-1].done if history else False,
                trace=history,
                cost=usage_summary.get("total_cost", 0.0),
                input_tokens=usage_summary.get("total_input_tokens", 0),
                output_tokens=usage_summary.get("total_output_tokens", 0),
            )
            reward_repr = (
                f"{result.total_reward:.4f}"
                if graded_reward is not None
                else "n/a (not graded)"
            )
            logger.info(f"[SWEBenchSubAgentRunner] SubAgent completed: steps={result.steps}, reward={reward_repr}")
            return result
            
        except Exception as e:
            logger.error(f"[SWEBenchSubAgentRunner] SubAgent execution failed: {e}", exc_info=True)
            raise


class SWEBenchRunner(Runner):
    """Runner for MainAgent with SubAgent delegation for SWE-bench."""
    
    def __init__(
        self,
        main_model: str,
        sub_models: List[str],
        max_attempts: int = 10,
        prompt_builder=None,
        trajectory_dir: Path | None = None,
        csv_summary_path: Path | None = None,
    ):
        self.main_model = main_model
        self.sub_models = sub_models
        self.max_attempts = max_attempts
        self.prompt_builder = prompt_builder or SWEBenchMainAgentPrompt
        self.trajectory_dir = Path(trajectory_dir) if trajectory_dir else None
        self.csv_summary_path = Path(csv_summary_path) if csv_summary_path else None
        self._csv_lock = asyncio.Lock()
    
    async def run(self, agent, env: Environment) -> LevelResult:
        """Run MainAgent orchestration for SWE-bench."""
        env_info = env.get_basic_info()
        logger.info(f"[SWEBenchOrchestra] Starting task: {env_info.env_id}")
        
        # Create MainAgent info
        main_info = BasicInfo(
            env_id=env_info.env_id,
            instruction=env_info.instruction,
            action_space="",
            max_steps=self.max_attempts,
            meta_data=env_info.meta_data,
        )
        
        # Create MainAgent with tools
        logger.info(f"[SWEBenchOrchestra] Creating MainAgent with model={self.main_model}")
        main_llm = create_llm_instance(LLMsConfig.default().get(self.main_model))
        sub_runner = SWEBenchSubAgentRunner()
        
        delegate = DelegateTaskTool(
            env=env, 
            runner=sub_runner, 
            models=self.sub_models,
            benchmark_type="swebench",
        )
        submit = SubmitTool(env=env)
        
        main_agent = MainAgent(
            llm=main_llm,
            sub_models=self.sub_models,
            tools=[delegate, submit],
            prompt_builder=self.prompt_builder,
            max_attempts=self.max_attempts,
            benchmark_type="swebench",
        )
        main_agent.reset(main_info)
        
        # Orchestration loop
        history = []
        total_reward = 0.0
        total_sub_cost = 0.0
        done = False
        level_result = None
        exception_occurred = None
        
        try:
            for attempt_idx in range(self.max_attempts):
                logger.info(f"[SWEBenchOrchestra] MainAgent attempt {attempt_idx + 1}/{self.max_attempts}")
                
                try:
                    action, resp = await main_agent.step(None, history)
                except Exception as step_error:
                    logger.error(f"[SWEBenchOrchestra] MainAgent.step() FAILED: {step_error}", exc_info=True)
                    raise
                
                action_name = action.get("action")
                result = action.get("result", {})
                step_done = result.get("done", False)
                is_submit = action_name == "submit"
                # Only submit produces a real graded reward; other actions
                # (delegate_task, etc.) have no reward signal — record 0.0 to
                # avoid implying the grader judged a score it never ran.
                reward = float(result.get("reward", 0.0)) if is_submit else 0.0

                if action_name == "delegate_task":
                    sub_cost = result.get("cost", 0.0)
                    total_sub_cost += sub_cost
                    if hasattr(env, "get_current_patch"):
                        try:
                            result["git_diff"] = await env.get_current_patch()
                        except Exception as patch_err:
                            logger.warning(f"[SWEBenchOrchestra] get_current_patch failed: {patch_err}")
                            result["git_diff"] = ""

                history.append(StepRecord(
                    observation={},
                    action=action,
                    reward=reward,
                    raw_response=resp,
                    done=step_done,
                    info=result,
                ))

                if is_submit:
                    total_reward = reward
                
                if step_done and is_submit:
                    done = True
                    break
            
            # Force submit if needed
            if not done:
                logger.info("[SWEBenchOrchestra] Max attempts reached without submit")
                executor = getattr(env, "_executor", None)
                container_ready = hasattr(env, '_container_started') and env._container_started
                
                if container_ready and executor and hasattr(executor, "run_tests"):
                    try:
                        reward, _ = await executor.run_tests()
                        total_reward = float(reward)
                        done = True
                    except Exception as e:
                        logger.error(f"[SWEBenchOrchestra] Forced submit failed: {e}")
                        done = True
                        total_reward = 0.0
            
        except Exception as e:
            logger.error(f"[SWEBenchOrchestra] Exception: {e}", exc_info=True)
            exception_occurred = e
            
        finally:
            # Cleanup
            if hasattr(env, 'close'):
                try:
                    await env.close()
                except Exception as e:
                    logger.error(f"[SWEBenchOrchestra] Cleanup error: {e}")
            
            # Build result
            try:
                usage = main_agent.llm.get_usage_summary() if main_agent else {}
                main_cost = usage.get("total_cost", 0.0)
                total_cost = main_cost + total_sub_cost
                
                level_result = LevelResult(
                    model=usage.get("model", self.main_model),
                    total_reward=total_reward,
                    steps=len(history),
                    done=done,
                    trace=history,
                    cost=total_cost,
                    input_tokens=usage.get("total_input_tokens", 0),
                    output_tokens=usage.get("total_output_tokens", 0),
                )
                
                if self.trajectory_dir:
                    self._save_trajectory(env_info, level_result, main_agent, history)
                
                if self.csv_summary_path:
                    await self._save_csv(env_info.env_id, level_result)
                    
            except Exception as save_error:
                logger.error(f"[SWEBenchOrchestra] Save error: {save_error}")
            
            if exception_occurred:
                raise exception_occurred
            
            return level_result
    
    def _save_trajectory(self, info: BasicInfo, result: LevelResult, 
                         main_agent, history: List[StepRecord]) -> None:
        """Save detailed trajectory."""
        try:
            self.trajectory_dir.mkdir(parents=True, exist_ok=True)
            
            attempts = []
            for i, record in enumerate(history):
                action_data = record.action
                action_name = action_data.get("action")
                result_data = action_data.get("result", {})
                
                attempt = {
                    "attempt": i + 1,
                    "subtask_history": action_data.get("subtask_history", ""),
                    "main_agent": {
                        "action": action_name,
                        "params": action_data.get("params", {}),
                        "raw_response": record.raw_response,
                    },
                }
                
                if action_name == "delegate_task":
                    attempt["sub_agent"] = {
                        "model": result_data.get("model"),
                        "tools_assigned": result_data.get("tools_assigned", []),
                        "steps": result_data.get("steps_taken", 0),
                        "cost": result_data.get("cost", 0.0),
                        "finish_result": result_data.get("finish_result"),
                        "trace_summary": result_data.get("trace_summary", ""),
                        "trace": result_data.get("trace", []),
                        "git_diff": result_data.get("git_diff", ""),
                    }
                elif action_name == "submit":
                    attempt["submit_result"] = {
                        "success": result_data.get("success"),
                        "reward": result_data.get("reward"),
                    }
                
                attempts.append(attempt)
            
            trajectory = {
                "task_id": info.env_id,
                "instruction": info.instruction,
                "metadata": info.meta_data,
                "main_model": self.main_model,
                "sub_models": self.sub_models,
                "total_reward": result.total_reward,
                "success": result.total_reward > 0,
                "done": result.done,
                "total_attempts": len(history),
                "total_cost": result.cost,
                "timestamp": result.timestamp,
                "attempts": attempts,
            }
            
            filename = f"{info.env_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            # Sanitize filename (replace / with _)
            filename = filename.replace("/", "_")
            filepath = self.trajectory_dir / filename
            
            with filepath.open("w", encoding="utf-8") as f:
                json.dump(trajectory, f, indent=2, ensure_ascii=False)
                
            logger.info(f"[SWEBenchOrchestra] Trajectory saved to {filepath}")
        except Exception as e:
            logger.error(f"[SWEBenchOrchestra] Failed to save trajectory: {e}")
    
    async def _save_csv(self, task_id: str, result: LevelResult) -> None:
        """Save summary to CSV."""
        async with self._csv_lock:
            try:
                self.csv_summary_path.parent.mkdir(parents=True, exist_ok=True)
                
                fieldnames = ["task_id", "model", "success", "reward", "attempts", "cost", "timestamp"]
                
                need_header = not self.csv_summary_path.exists() or self.csv_summary_path.stat().st_size == 0
                if need_header:
                    with self.csv_summary_path.open("w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                
                success = result.total_reward > 0
                with self.csv_summary_path.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow({
                        "task_id": task_id,
                        "model": result.model,
                        "success": success,
                        "reward": f"{result.total_reward:.4f}",
                        "attempts": result.steps,
                        "cost": f"{result.cost:.6f}",
                        "timestamp": result.timestamp,
                    })
            except Exception as e:
                logger.error(f"[SWEBenchOrchestra] Failed to save CSV: {e}")


class SWEBenchOrchestra(Benchmark):
    """SWE-bench benchmark with aorchestra (MainAgent + SubAgent)."""
    
    def __init__(self, config: SWEBenchOrchestraConfig):
        self.config = config
        self.main_model = config.main_model
        self.sub_models = config.sub_models
        self.max_attempts = config.max_attempts
        
        # Create data loader
        self._data_loader = SWEBenchDataLoader(
            dataset_name=config.dataset_name,
            split=config.split,
            cache_dir=config.cache_dir,
            subset_seed=config.subset_seed,
            subset_sizes=config.subset_sizes,
            subset_role=config.subset_role,
        )
        
        # Instance cache
        self._instances: Dict[str, SWEBenchInstance] = {}
        
        # Setup paths
        trajectory_dir = config.trajectory_dir or (config.result_folder / "trajectories")
        csv_path = config.csv_summary_path or (config.result_folder / "results.csv")
        
        # Create runner with SWE-bench specific prompt
        self._runner = SWEBenchRunner(
            main_model=config.main_model,
            sub_models=config.sub_models,
            max_attempts=config.max_attempts,
            prompt_builder=SWEBenchMainAgentPrompt,
            trajectory_dir=trajectory_dir,
            csv_summary_path=csv_path,
        )
    
    def _load_selected_ids(self) -> Optional[Set[str]]:
        """Load selected instance IDs from file if configured."""
        if not self.config.selected_ids_file:
            return None
        
        ids_file = self.config.selected_ids_file
        if not ids_file.exists():
            logger.warning(f"Selected IDs file not found: {ids_file}")
            return None
        
        try:
            with ids_file.open("r", encoding="utf-8") as f:
                ids = json.load(f)
            if isinstance(ids, list):
                selected = set(ids)
                logger.info(f"Loaded {len(selected)} selected instance IDs from {ids_file}")
                return selected
            else:
                logger.warning(f"Selected IDs file should contain a JSON array: {ids_file}")
                return None
        except Exception as e:
            logger.error(f"Failed to load selected IDs from {ids_file}: {e}")
            return None
    
    def list_levels(self) -> List[LevelSpec]:
        """List all available SWE-bench instances."""
        instances = self._data_loader.load_instances()
        
        # Load selected IDs filter
        selected_ids = self._load_selected_ids()
        
        levels = []
        for inst in instances:
            # Filter by selected IDs if provided
            if selected_ids is not None and inst.instance_id not in selected_ids:
                continue
            
            self._instances[inst.instance_id] = inst
            levels.append({
                "id": inst.instance_id,
                "_instance": inst,
            })
            
            if self.config.max_tasks and len(levels) >= self.config.max_tasks:
                break
        
        logger.info(f"Loaded {len(levels)} SWE-bench instances for aorchestra")
        if selected_ids is not None:
            logger.info(f"  (filtered from {len(selected_ids)} selected IDs)")
        return levels
    
    def make_env(self, level: LevelSpec) -> SWEBenchEnvironment:
        """Create environment for a specific instance."""
        instance = level.get("_instance") or self._instances.get(level["id"])
        if not instance:
            raise ValueError(f"Instance not found: {level['id']}")
        
        # Create SWEBenchConfig from SWEBenchOrchestraConfig
        swe_config = SWEBenchConfig(
            dataset_name=self.config.dataset_name,
            split=self.config.split,
            subset_seed=self.config.subset_seed,
            subset_sizes=self.config.subset_sizes,
            subset_role=self.config.subset_role,
            max_steps=self.config.max_steps,
            max_tasks=self.config.max_tasks,
            docker_timeout=self.config.docker_timeout,
            model=self.config.main_model,
            result_folder=self.config.result_folder,
            trajectory_dir=self.config.trajectory_dir,
            csv_summary_path=self.config.csv_summary_path,
            timestamp=self.config.timestamp,
            env_init=self.config.env_init,
            cache_dir=self.config.cache_dir,
            window_size=self.config.window_size,
        )
        
        return SWEBenchEnvironment(level, swe_config, instance)
    
    async def run(self, levels: List[LevelSpec], max_concurrency: int = 1):
        """Run benchmark with SWEBenchRunner."""
        semaphore = asyncio.Semaphore(max_concurrency)
        results = {}
        
        async def run_level(level):
            level_id = level.get("id", str(level))
            env = None
            async with semaphore:
                try:
                    logger.info(f"[SWEBenchOrchestra] Creating environment for {level_id}")
                    env = self.make_env(level)
                    
                    # Reset environment to start container
                    logger.info(f"[SWEBenchOrchestra] Resetting environment for {level_id}")
                    await env.reset()
                    
                    logger.info(f"[SWEBenchOrchestra] Starting runner for {level_id}")
                    result = await self._runner.run(None, env)
                    logger.info(f"[SWEBenchOrchestra] Completed {level_id}: reward={result.total_reward}")
                    results[level_id] = result
                except Exception as e:
                    import traceback
                    logger.error(f"[SWEBenchOrchestra] Task {level_id} failed: {e}")
                    logger.error(f"[SWEBenchOrchestra] Traceback:\n{traceback.format_exc()}")
                    results[level_id] = None
                finally:
                    # Cleanup environment
                    if env and hasattr(env, 'close'):
                        try:
                            await env.close()
                        except Exception as cleanup_err:
                            logger.warning(f"[SWEBenchOrchestra] Cleanup error for {level_id}: {cleanup_err}")
        
        # Run tasks with proper exception handling
        tasks = [asyncio.create_task(run_level(level)) for level in levels]
        gather_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check for any unhandled exceptions from gather
        for i, res in enumerate(gather_results):
            if isinstance(res, Exception):
                level_id = levels[i].get("id", str(levels[i]))
                logger.error(f"[SWEBenchOrchestra] Unhandled exception for {level_id}: {res}")
                if level_id not in results:
                    results[level_id] = None
        
        # Print summary
        total = len(results)
        success_count = sum(1 for r in results.values() if r and r.total_reward > 0)
        total_reward = sum(r.total_reward for r in results.values() if r)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"SWE-bench aorchestra Summary:")
        logger.info(f"  Total tasks: {total}")
        logger.info(f"  Successful: {success_count}/{total}")
        logger.info(f"  Total reward: {total_reward:.2f}")
        if total > 0:
            logger.info(f"  Average reward: {total_reward/total:.4f}")
        logger.info(f"{'='*60}")
        
        return results
