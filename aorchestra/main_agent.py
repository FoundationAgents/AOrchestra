"""MainAgent: orchestrates SubAgents via tool calls."""
from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, List, Optional

from pydantic import Field

from base.agent.base_agent import BaseAgent
from base.agent.memory import Memory
from base.engine.async_llm import ModelPricing
from base.engine.logs import logger, LogLevel
from benchmark.common.env import BasicInfo
from aorchestra.common.utils import indent_text
from base.engine.utils import parse_llm_action_response


def build_model_pricing_table(
    sub_models: List[str], 
    model_to_alias: Dict[str, str] = None
) -> str:
    """Generate a pricing table for available sub-models."""
    lines = ["| Model | Input $/1K | Output $/1K |"]
    lines.append("|-------|-----------|------------|")
    
    alias_to_model = {v: k for k, v in model_to_alias.items()} if model_to_alias else {}
    
    for model_display in sub_models:
        real_model = alias_to_model.get(model_display, model_display)
        input_price = ModelPricing.get_price(real_model, "input")
        output_price = ModelPricing.get_price(real_model, "output")
        lines.append(f"| {model_display} | ${input_price:.5f} | ${output_price:.5f} |")
    
    return "\n".join(lines)


class MainAgent(BaseAgent):
    """Orchestrator that delegates tasks to SubAgents."""
    
    name: str = Field(default="MainAgent")
    description: str = Field(default="Multi-agent orchestrator")
    
    sub_models: List[str] = Field(default_factory=list)
    tools: List[Any] = Field(default_factory=list)
    subagent_tools: List[Any] = Field(default_factory=list)  # Tools for SubAgent (used in prompt)
    prompt_builder: Optional[Any] = Field(default=None)
    max_attempts: int = Field(default=10)
    benchmark_type: str = Field(default="terminalbench")  # "gaia" | "terminalbench"
    
    # Model name masking (optional)
    mask_model_names: bool = Field(default=False)
    model_to_alias: Dict[str, str] = Field(default_factory=dict)
    alias_to_model: Dict[str, str] = Field(default_factory=dict)
    masked_sub_models: List[str] = Field(default_factory=list)
    
    # State
    memory: Memory = Field(default=None)
    instruction: str = Field(default="")
    meta: Dict[str, Any] = Field(default_factory=dict)
    attempt: int = Field(default=0)
    context: str = Field(default="")
    history: List[Dict] = Field(default_factory=list)
    task_entries: List[Dict] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        super().__init__(**data)
        # Set up model name masking
        if self.mask_model_names and self.sub_models:
            self.model_to_alias = {
                model: f"model_{i+1}" for i, model in enumerate(self.sub_models)
            }
            self.alias_to_model = {v: k for k, v in self.model_to_alias.items()}
            self.masked_sub_models = list(self.model_to_alias.values())
        else:
            self.model_to_alias = {m: m for m in self.sub_models}
            self.alias_to_model = {m: m for m in self.sub_models}
            self.masked_sub_models = self.sub_models
    
    def reset(self, env_info: BasicInfo) -> None:
        self.memory = Memory(llm=self.llm, max_memory=20)
        self.instruction = env_info.instruction
        self.meta = env_info.meta_data or {}
        self.attempt = 0
        self.context = ""
        self.history = []
        self.task_entries = []
    
    def get_usage_cost(self) -> float:
        return self.llm.get_usage_summary().get("total_cost", 0.0)
    
    
    def _format_subtask_history(self) -> str:
        """Generate subtask history for prompt usage"""
        if not self.task_entries:
            return "No subtasks completed yet."
        
        lines = []
        done_count = 0
        all_completed = []
        all_issues = []
        
        for e in self.task_entries:
            emoji = "✅" if e["status"] == "done" else "⚠️"
            steps_info = f'{e.get("steps_taken", "?")}/{e.get("max_steps", 30)}'
            model_display = e.get("model", "?")
            
            # Model name masking
            if self.mask_model_names and model_display in self.model_to_alias:
                model_display = self.model_to_alias[model_display]
            
            entry_lines = [
                f'[Attempt {e["attempt"]}] {emoji} {e["status"]} | Model: {model_display} | Steps: {steps_info}',
                f'├─ Task: {e.get("instruction", "N/A")}',
            ]
            
            # GAIA format
            if self.benchmark_type == "gaia":
                result_str = f'"{e.get("result", "")}"' if e.get("result") and e.get("result") != "-" else "(no result)"
                entry_lines.append(f'├─ Result: {result_str}')
                if e.get("summary"):
                    entry_lines.append(f'├─ Summary: {e["summary"]}')
                if e.get("status") in ("timeout", "partial") and (not e.get("result") or e.get("result") == "-"):
                    entry_lines.append(f'├─ Note: SubAgent did not complete. Check Trace summary below for intermediate findings.')
            # TerminalBench format
            else:
                if e.get("message"):
                    entry_lines.append(f'├─ Message: {e["message"]}')
                completed = e.get("completed", [])
                if completed:
                    entry_lines.append(f'├─ ✅ Completed: {completed}')
                    all_completed.extend(completed)
                issues = e.get("issues", [])
                if issues:
                    entry_lines.append(f'├─ ❌ Issues: {issues}')
                    all_issues.extend(issues)
            
            # Trace summary
            trace_summary = e.get("trace_summary", "")
            if trace_summary and trace_summary != "N/A":
                entry_lines.append(f'└─ Trace summary:\n{indent_text(trace_summary, "   ")}')
            else:
                entry_lines[-1] = entry_lines[-1].replace('├─', '└─')
            
            lines.append("\n".join(entry_lines))
            
            if e["status"] == "done":
                done_count += 1
        
        # Summary
        summary_lines = [f"---", f"Summary: {done_count}/{len(self.task_entries)} subtasks done"]
        if self.benchmark_type == "terminalbench":
            if all_completed:
                summary_lines.append(f"✅ All completed: {all_completed}")
            if all_issues:
                summary_lines.append(f"❌ All issues: {all_issues}")
        
        lines.append("\n".join(summary_lines))
        
        return "\n\n".join(lines)
    
    async def step(self, observation, history, **kwargs) -> tuple:
        """Execute one orchestration decision."""
        self.attempt += 1
        logger.info(f"[MainAgent] Step {self.attempt}/{self.max_attempts}")
        
        subtask_history = self._format_subtask_history()
        logger.info(f"[MainAgent] Subtask history:\n{subtask_history}")
        
        # Build prompt
        if self.prompt_builder:
            prompt = self.prompt_builder.build_prompt(
                instruction=self.instruction,
                meta=self.meta,
                prior_context=self.context,
                attempt_index=self.attempt,
                max_attempts=self.max_attempts,
                sub_models=self.masked_sub_models,
                subtask_history=subtask_history,
                model_to_alias=self.model_to_alias if self.mask_model_names else None,
                tools=self.subagent_tools,
            )
        else:
            prompt = self._default_prompt()
        
        # Log prompt
        prompt_msg = f"\n{'='*80}\n[MainAgent Attempt {self.attempt}] PROMPT:\n{'='*80}\n{prompt}\n{'='*80}\n"
        logger.warning(prompt_msg)
        logger.log_to_file(LogLevel.INFO, prompt_msg)
        
        # Get LLM decision
        logger.info(f"[MainAgent] Calling LLM...")
        resp = await self.llm(prompt)
        
        # Log response
        response_msg = f"\n{'='*80}\n[MainAgent Attempt {self.attempt}] RAW RESPONSE:\n{'='*80}\n{resp}\n{'='*80}\n"
        logger.warning(response_msg)
        logger.log_to_file(LogLevel.INFO, response_msg)
        
        # Parse decision (graceful fallback — never throws)
        decision = parse_llm_action_response(resp)
        
        # Log parsed decision
        decision_msg = f"\n{'='*80}\n[MainAgent Attempt {self.attempt}] PARSED DECISION:\n{'='*80}\n{json.dumps(decision, indent=2, ensure_ascii=False)}\n{'='*80}\n"
        logger.warning(decision_msg)
        logger.log_to_file(LogLevel.INFO, decision_msg)
        
        action_name = decision.get("action")
        params = decision.get("params", {})
        
        # Execute tool
        tool = next((t for t in self.tools if t.name == action_name), None)
        if not tool:
            return {"action": "error", "error": f"Unknown action: {action_name}"}, resp
        
        result = await tool(**params)
        
        # Update context and task_entries
        self._update_context(action_name, params, result)
        
        # Log updated context
        context_msg = f"\n{'='*80}\n[MainAgent Attempt {self.attempt}] UPDATED CONTEXT:\n{'='*80}\n{self.context}\n{'='*80}\n"
        logger.warning(context_msg)
        logger.log_to_file(LogLevel.INFO, context_msg)
        
        return {
            "action": action_name,
            "params": params,
            "result": result,
            "subtask_history": subtask_history,
        }, resp
    
    def _default_prompt(self) -> str:
        return f"""Task: {self.instruction}

Context:
{self.context or 'First attempt'}

Return JSON: {{"action": "...", "reasoning": "...", "params": {{...}}}}"""
    
    def _update_context(self, action: str, params: Dict, result: Dict) -> None:
        """Update context and task_entries after tool execution."""
        summary = f"[{self.attempt}] {action}\n"
        
        if action == "delegate_task":
            finish = result.get("finish_result", {})
            if finish:
                summary += f"  Status: {finish.get('status')}\n"
                if finish.get('completed'):
                    summary += f"  Done: {finish['completed']}\n"
                if finish.get('issues'):
                    summary += f"  Issues: {finish['issues']}\n"
                if finish.get('message'):
                    summary += f"  Message: {finish.get('message')}\n"
                # GAIA format
                if finish.get('result'):
                    summary += f"  Result: {finish.get('result')}\n"
            else:
                summary += f"  Steps: {result.get('steps_taken', 0)}, Done: {result.get('done', False)}\n"
            
            # Build task_entry
            finish_result = result.get('finish_result', {})
            if finish_result:
                entry_status = finish_result.get('status', 'partial')
                entry_message = finish_result.get('message', '')
                entry_completed = finish_result.get('completed', [])
                entry_issues = finish_result.get('issues', [])
                entry_result = finish_result.get('result', '-')
                entry_summary = finish_result.get('summary', '')
            else:
                entry_status = 'partial'
                entry_message = 'SubAgent did not finish (max steps reached).'
                entry_completed = []
                entry_issues = ['SubAgent timeout - did not call finish']
                entry_result = '-'
                entry_summary = ''
            
            self.task_entries.append({
                "attempt": self.attempt,
                "status": entry_status,
                "instruction": params.get('task_instruction', 'N/A'),
                "model": params.get('model', 'unknown'),
                "steps_taken": result.get('steps_taken', 0),
                "max_steps": result.get('statistics', {}).get('max_steps', 30),
                "cost": result.get('cost', 0),
                "message": entry_message,
                "completed": entry_completed,
                "issues": entry_issues,
                "result": entry_result,
                "summary": entry_summary,
                "trace_summary": result.get('trace_summary', ''),
            })
            
        elif action == "submit":
            summary += f"  Success: {result.get('success')}, Reward: {result.get('reward')}\n"
        
        elif action == "complete":
            summary += f"  Answer: {params.get('answer', 'N/A')}\n"
        
        self.context = summary + "\n" + self.context
        self.history.append({"attempt": self.attempt, "action": action, "result": result})
    
    async def run(self, request: Optional[str] = None) -> str:
        return "Orchestration via Runner"
