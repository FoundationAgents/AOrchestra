"""
GAIA Benchmark Runner with tool-based MainAgent (delegate_task).
"""
from __future__ import annotations

import asyncio
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from base.agent.base_action import BaseAction
from base.engine.async_llm import LLMsConfig, create_llm_instance
from base.engine.logs import logger
from benchmark.common.runner import Runner
from benchmark.gaia.scorer import question_scorer
from benchmark.gaia.llm_scorer import llm_semantic_score
from aorchestra.main_agent import MainAgent
from aorchestra.prompts.gaia import GAIAMainAgentPrompt
from aorchestra.tools.delegate import DelegateTaskTool


class GAIARunner:
    """Run GAIA levels with a tool-based MainAgent (delegate_task)."""

    def __init__(
        self,
        benchmark,
        main_model: str,
        sub_models: List[str],
        max_attempts: int,
        gaia_tools: List[BaseAction],
    ):
        self.benchmark = benchmark
        self.main_model = main_model
        self.sub_models = sub_models
        self.max_attempts = max_attempts
        self.gaia_tools = gaia_tools

    def _prepare_csv(
        self, csv_path: Path | None
    ) -> Tuple[csv.DictWriter | None, asyncio.Lock | None, Any]:
        if not csv_path:
            return None, None, None
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "task_id",
                "level",
                "main_model",
                "sub_models",
                "final_sub_model",
                "success",
                "reward",
                "attempts",
                "sub_cost",
                "main_cost",
                "total_cost",
                "timestamp",
                "start_time",
                "end_time",
                "error",
            ],
        )
        csv_writer.writeheader()
        csv_file.flush()
        return csv_writer, asyncio.Lock(), csv_file

    def _save_trajectory(
        self,
        trajectory_folder: Path,
        level_id: str,
        timestamp: str,
        level: int | None,
        question: str | None,
        expected_answer: str | None,
        file_name: str | None,
        instruction: str | None,
        meta: dict | None,
        attempts_detail: List[Dict[str, Any]],
        success: bool,
        total_reward: float,
        total_cost: float,
        main_cost: float,
        sub_cost: float,
        final_sub_model: str | None,
        error: str | None,
        start_time: str | None = None,
        end_time: str | None = None,
        max_attempts: int | None = None,
    ) -> None:
        if not trajectory_folder:
            return
        trajectory_folder.mkdir(parents=True, exist_ok=True)
        filename = f"{level_id}_{timestamp}.json" if timestamp else f"{level_id}.json"
        trajectory_file = trajectory_folder / filename
        data = {
            "task_id": level_id,
            "timestamp": timestamp,
            "start_time": start_time,
            "end_time": end_time,
            "level": level,
            "question": question,
            "expected_answer": expected_answer,
            "file_name": file_name,
            "main_model": self.main_model,
            "sub_models": self.sub_models,
            "max_attempts": max_attempts or self.max_attempts,
            "success": success,
            "total_reward": total_reward,
            "total_cost": total_cost,
            "main_cost": main_cost,
            "sub_cost": sub_cost,
            "attempts": len(attempts_detail),
            "trajectory": attempts_detail,
            "final_sub_model": final_sub_model,
            "error": error,
            "instruction": instruction,
            "meta": meta or {},
        }
        with trajectory_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"[GAIA] Trajectory saved to {trajectory_file}")

    async def run_levels(
        self,
        levels,
        max_concurrency: int,
        csv_path: Path | None,
        trajectory_folder: Path,
        timestamp: str,
    ) -> Dict[str, Dict[str, Any]]:
        semaphore = asyncio.Semaphore(max(1, max_concurrency))
        csv_writer, csv_lock, csv_file = self._prepare_csv(csv_path)
        results: Dict[str, Dict[str, Any]] = {}

        async def run_single(level_spec):
            level_id = level_spec.get("task_id") or level_spec.get("id") or str(level_spec)
            task_level = level_spec.get("Level")
            question = level_spec.get("Question", "")
            expected_answer = level_spec.get("Final answer", "")
            file_name = level_spec.get("file_name", "")

            async with semaphore:
                env = None
                start_time = datetime.now().isoformat()
                try:
                    logger.info(f"[GAIA] Starting task: {level_id} (Level {task_level})")
                    env = self.benchmark.make_env(level_spec, tools=self.gaia_tools)
                    basic_info = env.get_basic_info()

                    main_llm = create_llm_instance(LLMsConfig.default().get(self.main_model))
                    runner = Runner()
                    
                    # Create model alias mapping
                    model_to_alias = {
                        model: f"model_{i+1}" for i, model in enumerate(self.sub_models)
                    }
                    alias_to_model = {v: k for k, v in model_to_alias.items()}
                    
                    delegate_tool = DelegateTaskTool(
                        env=env,
                        runner=runner,
                        models=self.sub_models,
                        benchmark_type="gaia",
                        alias_to_model=alias_to_model,
                    )
                    
                    # GAIA uses complete action instead of submit
                    from aorchestra.tools.complete import CompleteTool
                    complete_tool = CompleteTool()
                    
                    main_agent = MainAgent(
                        llm=main_llm,
                        sub_models=self.sub_models,
                        tools=[delegate_tool, complete_tool],
                        subagent_tools=self.gaia_tools,
                        prompt_builder=GAIAMainAgentPrompt,
                        max_attempts=self.max_attempts,
                        benchmark_type="gaia",
                        mask_model_names=True,
                    )

                    # Run MainAgent
                    from benchmark.common.env import BasicInfo
                    main_info = BasicInfo(
                        env_id=level_id,
                        instruction=basic_info.instruction,
                        action_space="",
                        max_steps=self.max_attempts,
                        meta_data=basic_info.meta_data,
                    )
                    main_agent.reset(main_info)
                    
                    main_cost_before = main_agent.get_usage_cost()
                    
                    # Orchestration loop
                    attempts_detail = []
                    final_answer = None
                    
                    for attempt_idx in range(self.max_attempts):
                        action, resp = await main_agent.step(None, [])
                        
                        action_name = action.get("action")
                        params = action.get("params", {})
                        result = action.get("result", {})
                        subtask_history = action.get("subtask_history", "")
                        
                        attempts_detail.append({
                            "attempt": attempt_idx + 1,
                            "action": action_name,
                            "params": params,
                            "result": result,
                            "raw_response": resp,
                            "subtask_history": subtask_history,
                        })
                        
                        if action_name == "complete":
                            final_answer = params.get("answer")
                            break

                    # Fallback: if MainAgent never called complete, pick the most common "done" result
                    if final_answer is None and attempts_detail:
                        from collections import Counter
                        done_results = [
                            a.get("result", {}).get("finish_result", {}).get("result", "")
                            for a in attempts_detail
                            if a.get("action") == "delegate_task"
                            and a.get("result", {}).get("finish_result", {}).get("status") == "done"
                            and a.get("result", {}).get("finish_result", {}).get("result")
                            and a.get("result", {}).get("finish_result", {}).get("result") != "-"
                        ]
                        if done_results:
                            final_answer = Counter(done_results).most_common(1)[0][0]
                            logger.warning(
                                f"[GAIA] MainAgent never called complete. "
                                f"Fallback: using most common done result: {final_answer}"
                            )

                    main_cost_after = main_agent.get_usage_cost()
                    main_cost = max(0.0, main_cost_after - main_cost_before)

                    # Scoring
                    total_reward = 0.0
                    success = False
                    if final_answer:
                        logger.info(f"[GAIA] MainAgent complete with answer: {final_answer}")
                        
                        # Step 1: String matching
                        reward = question_scorer(str(final_answer), expected_answer)
                        
                        # Step 2: LLM semantic matching
                        if reward < 0.5:
                            logger.info(f"[GAIA] String match failed, trying LLM semantic scoring...")
                            try:
                                llm_reward = await llm_semantic_score(str(final_answer), expected_answer)
                                if llm_reward > 0.5:
                                    logger.info(f"[GAIA] LLM semantic match SUCCESS!")
                                    reward = llm_reward
                            except Exception as e:
                                logger.warning(f"[GAIA] LLM scoring failed: {e}")
                        
                        total_reward = reward
                        success = reward > 0.5

                    # Calculate sub_cost
                    sub_cost = sum(
                        float(a.get("result", {}).get("cost", 0.0) or 0.0)
                        for a in attempts_detail
                    )
                    total_cost = sub_cost + main_cost
                    final_sub_model = None  # Can be extracted from attempts_detail

                    end_time = datetime.now().isoformat()

                    self._save_trajectory(
                        trajectory_folder,
                        level_id,
                        timestamp,
                        task_level,
                        question,
                        expected_answer,
                        file_name,
                        basic_info.instruction,
                        basic_info.meta_data or {},
                        attempts_detail,
                        success,
                        total_reward,
                        total_cost,
                        main_cost,
                        sub_cost,
                        final_sub_model,
                        None,
                        start_time,
                        end_time,
                    )

                    if csv_writer and csv_lock:
                        async with csv_lock:
                            csv_writer.writerow({
                                "task_id": level_id,
                                "level": task_level,
                                "main_model": self.main_model,
                                "sub_models": ",".join(self.sub_models),
                                "final_sub_model": final_sub_model,
                                "success": success,
                                "reward": f"{total_reward:.4f}",
                                "attempts": len(attempts_detail),
                                "sub_cost": f"{sub_cost:.6f}",
                                "main_cost": f"{main_cost:.6f}",
                                "total_cost": f"{total_cost:.6f}",
                                "timestamp": timestamp,
                                "start_time": start_time,
                                "end_time": end_time,
                                "error": None,
                            })
                            csv_file.flush()

                    results[level_id] = {
                        "success": success,
                        "reward": total_reward,
                        "answer": final_answer,
                        "expected_answer": expected_answer,
                    }
                    logger.info(
                        f"[GAIA] Completed task: {level_id} | success={success} "
                        f"reward={total_reward:.4f} attempts={len(attempts_detail)}"
                    )
                    
                except Exception as e:
                    logger.error(f"[GAIA] Task {level_id} failed: {type(e).__name__}: {e}")
                    end_time = datetime.now().isoformat()
                    
                    self._save_trajectory(
                        trajectory_folder,
                        level_id,
                        timestamp,
                        task_level,
                        question,
                        expected_answer,
                        file_name,
                        None,
                        None,
                        attempts_detail,
                        False,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        None,
                        str(e),
                        start_time,
                        end_time,
                    )
                    
                    if csv_writer and csv_lock:
                        async with csv_lock:
                            csv_writer.writerow({
                                "task_id": level_id,
                                "level": task_level,
                                "main_model": self.main_model,
                                "sub_models": ",".join(self.sub_models),
                                "final_sub_model": None,
                                "success": False,
                                "reward": "0.0000",
                                "attempts": 0,
                                "sub_cost": "0.000000",
                                "main_cost": "0.000000",
                                "total_cost": "0.000000",
                                "timestamp": timestamp,
                                "start_time": start_time,
                                "end_time": end_time,
                                "error": str(e),
                            })
                            csv_file.flush()
                    
                    results[level_id] = {"success": False, "reward": 0.0, "error": str(e)}
                    
                finally:
                    if env and hasattr(env, "close"):
                        try:
                            await env.close()
                        except Exception as e:
                            logger.debug(f"[GAIA] Error closing env for {level_id}: {e}")

        tasks = [asyncio.create_task(run_single(level)) for level in levels]
        try:
            await asyncio.gather(*tasks)
        finally:
            if csv_file:
                csv_file.close()

        return results
