"""ReActAgent - ReAct-style SubAgent implementation"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from pydantic import Field

from base.agent.base_agent import BaseAgent
from base.agent.memory import Memory
from base.engine.utils import parse_llm_action_response, parse_llm_output
from base.engine.logs import logger, LogLevel
from benchmark.common.env import BasicInfo, Observation, Action


# GAIA SubAgent Prompt Template
GAIA_PROMPT = """You are a specialized SubAgent. Complete the assigned task efficiently.

==== Progress ====
[Step {current_step}/{max_steps}] Remaining {remaining_steps} steps
{budget_warning}
If you run out of steps without "finish", your work is lost and marked as timeout.

==== Your Task (from MainAgent) ====
{task_instruction}

==== Context ====
{context}

==== Original Question (for reference) ====
{original_question}

==== Available Tools ====
{action_space}

==== Guidelines ====
1. Focus on completing YOUR TASK above
2. Think step by step before outputting an action
3. Write key observations to the "memory" field
4. Use print() in ExecuteCodeAction to see computation results
5. Once done, use 'finish' IMMEDIATELY with your best result
6. You MUST ONLY use tools listed in Available Tools. Do NOT invent new tools.
7. Pay attention to precision, units, and format requirements in the question
8. Prefer PRIMARY SOURCES: download and read the actual document, webpage, subtitle file, or dataset rather than relying on search result snippets or secondary descriptions
9. For calculation, counting, or filtering tasks, use ExecuteCodeAction and print intermediate values — do not estimate mentally
10. For video/audio content questions, prefer extracting subtitles or transcripts over relying on web descriptions
11. For tasks involving provided files (PDF, PPTX, Excel, etc.), inspect and extract content programmatically (e.g. python-pptx, openpyxl, PyPDF2) rather than guessing from descriptions
12. For tasks requiring data from a specific website or database, navigate to the primary site or use its API directly rather than relying on search engine summaries
13. Return the EXACT value from the source — do not paraphrase, change plural/singular forms, or round unless the question explicitly asks for it

⚠️ BUDGET: When remaining_steps <= 5, use 'finish' NOW!

==== Output Format ====
CRITICAL: You MUST reply with ONLY a JSON object. No explanations, no markdown, no other text.

For tool use:
{{"action": "<tool_name>", "params": {{"<param>": "<value>"}}, "memory": "<key observations>"}}

For finish:
{{"action": "finish", "params": {{"result": "<your answer>", "status": "done|partial|blocked", "summary": "<what you did>"}}, "memory": "<final notes>"}}

==== Memory ====
{memory}

==== Current Observation ====
{obs}
"""


# TerminalBench SubAgent Prompt Template
TERMINALBENCH_PROMPT = """
==== Progress ====
[Step {current_step}/{max_steps}] Remaining: {remaining_steps} step(s)
{budget_warning}
If you run out of steps without "finish", your work is lost and marked as timeout.

==== Your Task (from MainAgent) ====
{task_instruction}

==== Context (from previous attempts) ====
{context}
Use this info: repeat what WORKED, avoid what FAILED.

==== Original Question (for reference) ====
{original_question}

==== Action Space ====
{action_space}

==== Memory ====
Recent memory:
{memory}

==== Current Observation ====
{obs}

==== Thinking ====
Think step by step before outputting an action. Write key reasoning in memory for future steps.

==== Action Guidelines ====
You have TWO actions available:

1. **execute** - Run shell commands and observe results
   - Use this to install packages, configure services, verify status, etc.
   - Example: "apt update && apt install -y nginx"

2. **finish** - Report your progress to MainAgent
   - Use when task is COMPLETE (status="done")
   - Use when you made PROGRESS but need more work (status="partial")
   - ⚠️ MUST use before running out of steps! Your work is LOST if you timeout.

**What to report in finish:**
- completed: List SUCCESSFUL steps that WORKED (e.g., ["apt update succeeded", "nginx installed"])
- issues: List FAILED attempts with WHY (e.g., ["nginx -v failed: command not found"])
- message: Brief summary of current state

This info helps the NEXT SubAgent know what to repeat and what to avoid.

==== Output Format ====
⚠️ CRITICAL: You MUST reply with ONLY a JSON object. No explanations, no markdown, no other text.

For execute:
{{"action": "execute", "params": {{"command": "your shell command"}}, "memory": "key findings"}}

For finish:
{{"action": "finish", "params": {{"status": "done|partial", "completed": [...], "issues": [...], "message": "..."}}, "memory": "final notes"}}


"""


class ReActAgent(BaseAgent):
    """ReAct-style SubAgent supporting GAIA and TerminalBench"""
    
    name: str = Field(default="ReActAgent")
    description: str = Field(default="ReAct-style SubAgent for Orchestra framework")
    
    # Core fields
    benchmark_type: str = Field(default="terminalbench")  # "gaia" | "terminalbench"
    task_instruction: str = Field(default="")             # Subtask assigned by MainAgent
    context: str = Field(default="")                      # Context/hints
    original_question: str = Field(default="")            # Original complete question
    allowed_tools: List[str] | None = Field(default=None) # Tool restrictions
    
    # Internal state
    current_env_instruction: str = Field(default="")
    current_action_space: str = Field(default="")
    memory: Memory = Field(default=None)
    
    class Config:
        arbitrary_types_allowed = True
    
    def reset(self, env_info: BasicInfo) -> None:
        """Initialize Agent"""
        if self.memory is None:
            self.memory = Memory(llm=self.llm, max_memory=10)
        else:
            self.memory.clear()
        
        # Save original question
        if not self.original_question:
            self.original_question = env_info.instruction
        
        self.current_env_instruction = env_info.instruction
        
        # Tool filtering (if allowed_tools specified)
        if self.allowed_tools:
            self.current_action_space = self._filter_action_space(
                env_info.action_space, 
                self.allowed_tools
            )
            logger.info(f"[ReActAgent] Filtered to tools: {self.allowed_tools}")
        else:
            self.current_action_space = env_info.action_space
    
    def _normalize_tool_name(self, name: str) -> str:
        """Normalize tool name for fuzzy matching"""
        normalized = name.lower().replace("_", "")
        if normalized.endswith("action"):
            normalized = normalized[:-6]
        return normalized
    
    def _tool_matches(self, tool_name: str, allowed_tools: List[str]) -> bool:
        """Check if tool name matches (supports fuzzy matching)"""
        if tool_name in allowed_tools:
            return True
        
        normalized_tool = self._normalize_tool_name(tool_name)
        for allowed in allowed_tools:
            if self._normalize_tool_name(allowed) == normalized_tool:
                return True
        
        return False
    
    def _filter_action_space(self, action_space: str, allowed_tools: List[str]) -> str:
        """Filter action_space, keeping only allowed tool descriptions"""
        blocks = re.split(r'\n(?=### )', action_space)
        
        filtered_blocks = []
        for block in blocks:
            if block.startswith("Available actions"):
                filtered_blocks.append(block.rstrip())
                continue
            
            match = re.match(r'### (\w+)', block)
            if match:
                tool_name = match.group(1)
                if self._tool_matches(tool_name, allowed_tools):
                    filtered_blocks.append(block.rstrip())
        
        return "\n\n".join(filtered_blocks)
    
    def parse_action(self, resp: str) -> Dict[str, Any]:
        """Parse LLM response to action"""
        return parse_llm_action_response(resp)
    
    def _get_memory(self) -> str:
        """Get memory text"""
        return self.memory.as_text()
    
    def _get_budget_warning(self, remaining_steps: int) -> str:
        """Generate budget warning"""
        if remaining_steps <= 3:
            return f"🚨 CRITICAL: Only {remaining_steps} steps left! Use 'finish' NOW!"
        elif remaining_steps <= 5:
            return f"⚠️ Warning: {remaining_steps} steps remaining. Plan to finish soon."
        return ""
    
    def _build_prompt(
        self,
        observation: Any,
        current_step: int,
        max_steps: int,
        remaining_steps: int,
        budget_warning: str,
    ) -> str:
        """Build prompt based on benchmark_type"""
        if self.benchmark_type == "gaia":
            return GAIA_PROMPT.format(
                task_instruction=self.task_instruction,
                context=self.context or "None",
                original_question=self.original_question,
                action_space=self.current_action_space,
                memory=self._get_memory(),
                obs=observation,
                current_step=current_step,
                max_steps=max_steps,
                remaining_steps=remaining_steps,
                budget_warning=budget_warning,
            )
        else:  # terminalbench
            return TERMINALBENCH_PROMPT.format(
                task_instruction=self.task_instruction,
                context=self.context or "No additional context provided.",
                original_question=self.original_question,
                action_space=self.current_action_space,
                memory=self._get_memory(),
                obs=observation,
                current_step=current_step,
                max_steps=max_steps,
                remaining_steps=remaining_steps,
                budget_warning=budget_warning,
            )
    
    async def step(
        self, 
        observation: Observation, 
        history: Any, 
        current_step: int = 1, 
        max_steps: int = 30
    ) -> tuple[Action, str, str]:
        """Execute one step
        
        Returns:
            tuple: (action, raw_response, raw_input_prompt)
        """
        remaining_steps = max_steps - current_step
        budget_warning = self._get_budget_warning(remaining_steps)
        
        # Build prompt
        prompt = self._build_prompt(
            observation=observation,
            current_step=current_step,
            max_steps=max_steps,
            remaining_steps=remaining_steps,
            budget_warning=budget_warning,
        )
        
        logger.log_to_file(LogLevel.INFO, f"ReActAgent Input:\n{prompt}\n")
        
        try:
            resp = await self.llm(prompt)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            resp = ""
        
        # Parse response
        memory_content = parse_llm_output(resp, "memory")
        thinking = memory_content.get("memory") if isinstance(memory_content, dict) else None
        action = self.parse_action(resp)
        
        logger.agent_action(f"ReActAgent Action: {action}")
        
        # Update memory
        agent_obs = history[-1].info.get("last_action_result") if history else None
        await self.memory.add_memory(obs=agent_obs, action=action, thinking=thinking, raw_response=resp)
        
        return action, resp, prompt
    
    async def run(self, request: str = None) -> str:
        """Standalone run - not used in Orchestra mode"""
        return ""
