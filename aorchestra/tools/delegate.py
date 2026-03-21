"""DelegateTaskTool - Unified task delegation tool"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List

from pydantic import Field, PrivateAttr

from base.agent.base_action import BaseAction
from base.agent.memory import Memory
from base.engine.async_llm import LLMsConfig, create_llm_instance
from base.engine.logs import logger
from aorchestra.subagents import ReActAgent
from aorchestra.tools.trace_formatter import (
    create_gaia_formatter,
    create_terminalbench_formatter,
    create_swebench_formatter,
)


def _make_serializable(obj: Any) -> Any:
    """Recursively convert an object to JSON-serializable format."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _make_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    # Fallback: convert to string
    return str(obj)


class DelegateTaskTool(BaseAction):
    """Unified task delegation tool supporting GAIA, TerminalBench, and SWE-bench"""
    
    name: str = "delegate_task"
    description: str = "Delegate task to SubAgent that executes commands"
    parameters: Dict[str, Any] = Field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "task_instruction": {"type": "string", "description": "Task for SubAgent"},
            "context": {"type": "string", "description": "Additional context/hints"},
            "model": {"type": "string", "description": "Model to use"},
            "tools": {"type": "array", "items": {"type": "string"}, "description": "Tools for SubAgent (optional)"},
        },
        "required": ["task_instruction", "model"]
    })
    
    # Core dependencies
    env: Any = Field(default=None, exclude=True)
    runner: Any = Field(default=None, exclude=True)
    models: list = Field(default_factory=list)
    
    # Configuration
    benchmark_type: str = Field(default="terminalbench")  # "gaia" | "terminalbench" | "swebench"
    alias_to_model: Dict[str, str] = Field(default_factory=dict)  # Model name masking (optional)
    
    # Internal state
    _trace_formatter: Any = PrivateAttr(default=None)
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(
        self,
        env,
        runner,
        models: list,
        benchmark_type: str = "terminalbench",
        alias_to_model: Dict[str, str] = None,
    ):
        super().__init__()
        self.env = env
        self.runner = runner
        self.models = models
        self.benchmark_type = benchmark_type
        self.alias_to_model = alias_to_model or {}
        
        # Create corresponding trace formatter
        if benchmark_type == "gaia":
            self._trace_formatter = create_gaia_formatter()
        elif benchmark_type == "swebench":
            self._trace_formatter = create_swebench_formatter()
        else:
            self._trace_formatter = create_terminalbench_formatter()
        
        # Set model enum (using alias or real name)
        display_models = list(self.alias_to_model.keys()) if self.alias_to_model else models
        self.parameters = {
            "type": "object",
            "properties": {
                "task_instruction": {"type": "string", "description": "Task for SubAgent"},
                "context": {"type": "string", "description": "Additional context/hints"},
                "model": {
                    "type": "string", 
                    "description": f"Model to use. MUST be one of: {display_models}",
                    "enum": display_models
                },
                "tools": {"type": "array", "items": {"type": "string"}, "description": "Tools for SubAgent (optional)"},
            },
            "required": ["task_instruction", "model"]
        }
    
    async def __call__(
        self, 
        task_instruction: str, 
        model: str, 
        context: str = "", 
        tools: List[str] = None
    ) -> Dict:
        """Execute delegated task
        
        Args:
            task_instruction: Task description for SubAgent to execute
            model: Model to use (can be an alias)
            context: Additional context information
            tools: List of tools allowed for SubAgent
            
        Returns:
            Dictionary containing execution results
        """
        # 1. Parse model name (if alias mapping exists)
        real_model = self.alias_to_model.get(model, model)
        if real_model not in self.models:
            return {"error": f"Invalid model: {model}", "steps_taken": 0, "done": False}

        effective_tools = tools
        if self.benchmark_type == "gaia":
            effective_tools = None
        
        logger.info(f"[DelegateTool] Creating SubAgent with model={real_model}, tools={effective_tools}")
        
        # 2. Get original question
        original_question = getattr(self.env, 'instruction', '') or ''
        
        # 3. Create SubAgent (select different Agent based on benchmark_type)
        llm = create_llm_instance(LLMsConfig.default().get(real_model))
        
        if self.benchmark_type == "swebench":
            # SWE-bench uses dedicated SWEBenchSubAgent (DISCUSSION + COMMAND format)
            from aorchestra.subagents import SWEBenchSubAgent
            sub_agent = SWEBenchSubAgent(
                llm=llm,
                task_instruction=task_instruction,
                context=context,
                original_question=original_question,
                memory=Memory(llm=llm, max_memory=20),
            )
        else:
            # GAIA and TerminalBench use ReActAgent (JSON format)
            sub_agent = ReActAgent(
                llm=llm,
                benchmark_type=self.benchmark_type,
                task_instruction=task_instruction,
                context=context,
                original_question=original_question,
                allowed_tools=effective_tools,
                memory=Memory(llm=llm, max_memory=10),
            )
        
        # 4. Temporarily replace env.instruction
        original_instruction = getattr(self.env, 'instruction', None)
        if hasattr(self.env, 'instruction'):
            self.env.instruction = task_instruction
        
        try:
            # 5. Get env max_steps before execution
            env_max_steps = self.env.get_basic_info().max_steps
            
            # 6. Execute (GAIA uses specialized loop with forced finish)
            if self.benchmark_type == "gaia":
                result = await self._run_gaia_with_forced_finish(
                    sub_agent, self.env, env_max_steps
                )
            else:
                result = await self.runner.run(sub_agent, self.env)
            
            # 7. Extract finish_result
            finish_result = None
            if result.trace:
                last = result.trace[-1]
                if last.info.get("finished") and last.info.get("finish_result"):
                    finish_result = last.info["finish_result"]
            
            # 8. Summarize trace (using fixed gemini-3-flash-preview model)
            trace_summary = await self._summarize_trace(result.trace, task_instruction, finish_result)
            
            # Convert StepRecord objects to dicts for JSON serialization
            trace_serializable = [_make_serializable(step) for step in result.trace] if result.trace else []
            
            return {
                "model": real_model,
                "tools_assigned": effective_tools,
                "steps_taken": result.steps,
                "done": result.done,
                "cost": result.cost,
                "finish_result": finish_result,
                "trace": trace_serializable,
                "trace_summary": trace_summary,
                "statistics": {
                    "total_steps": result.steps, 
                    "max_steps": env_max_steps, 
                    "completed": result.done
                },
            }
            
        except Exception as e:
            logger.error(f"[DelegateTool] Error: {e}")
            return {"error": str(e), "steps_taken": 0, "done": False, "cost": 0.0}
            
        finally:
            # 8. Restore env.instruction
            if original_instruction is not None and hasattr(self.env, 'instruction'):
                self.env.instruction = original_instruction
    
    async def _run_gaia_with_forced_finish(
        self, agent, env, max_steps: int
    ):
        """GAIA-specific execution loop that passes step info and forces finish on last step."""
        import inspect
        from benchmark.common.runner import LevelResult, StepRecord
        from datetime import datetime

        start_time = datetime.now().isoformat()

        info = env.get_basic_info()
        agent.reset(info)

        reset_result = env.reset()
        obs = await reset_result if inspect.isawaitable(reset_result) else reset_result

        history: list[StepRecord] = []
        total_reward = 0.0

        for t in range(max_steps):
            current_step = t + 1

            step_result = await agent.step(
                observation=obs,
                history=history,
                current_step=current_step,
                max_steps=max_steps,
            )

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

            if t == max_steps - 1 and action.get("action") != "finish":
                last_output = ""
                if isinstance(obs, dict):
                    last_output = str(obs.get("output", "") or obs.get("error", ""))
                if not last_output:
                    for prev in reversed(history):
                        prev_obs = prev.observation
                        if isinstance(prev_obs, dict):
                            last_output = str(
                                prev_obs.get("output", "") or prev_obs.get("error", "")
                            )
                            if last_output:
                                break
                if len(last_output) > 300:
                    last_output = last_output[:300]

                logger.info(
                    f"[DelegateTool] Forcing finish at step {current_step}/{max_steps}"
                )
                action = {
                    "action": "finish",
                    "params": {
                        "result": last_output,
                        "status": "partial",
                        "summary": f"Forced finish at step {current_step}/{max_steps}. Last output preserved.",
                    },
                }
                raw_response = "forced_finish_on_last_step"

            obs_next, reward, done, step_info = await env.step(action)

            step_record = StepRecord(
                observation=obs,
                action=action,
                reward=reward,
                raw_response=raw_response,
                done=done,
                info=step_info,
                raw_input=raw_input,
            )
            history.append(step_record)
            total_reward += reward
            obs = obs_next

            if done:
                break

        end_time = datetime.now().isoformat()
        usage_summary = agent.llm.get_usage_summary()
        return LevelResult(
            model=usage_summary.get("model", ""),
            total_reward=total_reward,
            steps=len(history),
            done=history[-1].done if history else False,
            trace=history,
            cost=usage_summary.get("total_cost", 0.0),
            input_tokens=usage_summary.get("total_input_tokens", 0),
            output_tokens=usage_summary.get("total_output_tokens", 0),
            start_time=start_time,
            end_time=end_time,
        )

    async def _summarize_trace(self, trace, task_instruction: str, finish_result: dict = None) -> str:
        """Summarize execution trace (using fixed gemini-3-flash-preview model)"""
        if not trace:
            return "No steps executed"

        trace_text = self._trace_formatter.format_trace(trace)

        # Select different summary prompt based on benchmark_type
        if self.benchmark_type == "gaia":
            original_question = getattr(self.env, 'instruction', '') or task_instruction
            # Include finish_result so the summarizer knows the final output
            finish_section = ""
            if finish_result:
                fr_result = finish_result.get("result", "")
                fr_status = finish_result.get("status", "")
                finish_section = f"""
== SUBAGENT FINAL OUTPUT ==
Status: {fr_status}
Result: {fr_result}
"""
            prompt = f"""You are a neutral trajectory summarizer. Your job is to objectively describe what the SubAgent did and found, NOT to judge whether it succeeded or failed.

== ORIGINAL TASK ==
{original_question}
{finish_section}
== EXECUTION TRACE ==
{trace_text}

== INSTRUCTIONS ==
Summarize the SubAgent's execution in 3-7 neutral bullets:
- What information sources were consulted (URLs, APIs, files, tools used)
- What key data points or intermediate findings were obtained
- What the SubAgent's final answer/conclusion was and how it arrived there
- Any notable obstacles encountered (errors, timeouts, access issues)

IMPORTANT RULES:
- Be factual and objective. Do NOT editorialize about quality or completeness.
- Do NOT contradict the SubAgent's final output shown above. If the SubAgent produced a result, report it as-is.
- Do NOT speculate about whether the result is correct or incorrect.
- Focus on WHAT was done and found, not on what was NOT done.

Output ONLY the bullet-point summary."""
        elif self.benchmark_type == "swebench":
            original_question = getattr(self.env, 'instruction', '') or task_instruction
            prompt = f"""You are a trajectory summarizer for a SWE-bench task (GitHub issue fixing).
Review the SubAgent's execution trace and compare against the original issue.

== ORIGINAL ISSUE ==
{original_question[:1000]}

== EXECUTION TRACE ==
{trace_text}

== OUTPUT ==
Based on the trace, answer:
1. ✅ CODE CHANGES: What code changes were made? Which files were modified?
2. ✅ TESTS: Were tests run? Did they pass?
3. ❌ REMAINING: What is still needed to fully fix the issue?

Summarize in 5-10 bullets: key progress, problems, remaining issues.
Be specific and concise. Output ONLY the sections above."""
        else:  # terminalbench
            original_question = getattr(self.env, 'instruction', '') or task_instruction
            prompt = f"""You are a trajectory summarizer. Review the SubAgent's execution trace.
Compare the execution trace against the original task requirements.

== ORIGINAL TASK ==
{original_question}

== EXECUTION TRACE ==
{trace_text}

== OUTPUT ==
Based on the trace, answer:
1. ✅ COMPLETED: What requirements from the original task were actually done?
2. ❌ REMAINING: What requirements are still missing or not properly tested?

Summarize in 5-10 bullets: key progress, problems, remaining issues.
Be specific and concise. Output ONLY the two sections above."""
        
        try:
            review_llm = create_llm_instance(
                LLMsConfig.default().get("gemini-3-flash-preview")
            )
            return (await review_llm(prompt)).strip()
        except Exception as e:
            logger.warning(f"[DelegateTool] Trace summarization failed: {e}")
            return f"Steps: {len(trace)}"
