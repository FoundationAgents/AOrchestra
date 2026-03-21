"""
GAIA-specific MainAgent prompt.
GAIA tasks are question-answering tasks that require:
- Web search and information retrieval
- File analysis (PDF, images, audio, etc.)
- Code execution for computation
- Final answer extraction
"""
import json
from typing import Any, Dict, List

from aorchestra.main_agent import build_model_pricing_table


def format_tools_description(tools: List[Any]) -> str:
    """Format tools list into description string."""
    if not tools:
        return "No tools available."
    
    descriptions = []
    for tool in tools:
        desc = f"Tool Name: {tool.name}\nDescription: {tool.description}"
        if tool.parameters:
            desc += f"\nParameters: {json.dumps(tool.parameters, indent=2)}"
        descriptions.append(desc)
    
    return "\n\n".join(descriptions)


class GAIAMainAgentPrompt:
    """Generate prompts for GAIA benchmark tasks."""
    
    @staticmethod
    def build_prompt(
        instruction: str,
        meta: Dict[str, Any],
        prior_context: str,
        attempt_index: int,
        max_attempts: int,
        sub_models: List[str],
        subtask_history: str = "",
        model_to_alias: Dict[str, str] = None,
        tools: List[Any] = None,
    ) -> str:
        remaining_attempts = max_attempts - attempt_index
        model_pricing_table = build_model_pricing_table(sub_models, model_to_alias)
        tools_description = format_tools_description(tools or [])
        
        return f"""
You are the MainAgent (Orchestrator). Your task is to solve the given QUESTION by decomposing it into subtasks and delegating each to a sub-agent.

DECISION PROCESS:
1. REVIEW the SUBTASK HISTORY below — check the Result field and trace summary of each attempt.
2. CONVERGENCE CHECK (apply in order):
   a. If ≥2 attempts independently arrived at the SAME answer (using different sources or methods) → You MUST 'complete' with that answer. Do not delegate again.
   b. If only 1 attempt remaining → You MUST 'complete' now with your best available answer.
   c. If you have exactly 1 "done" result and it is a factual answer (number, name, date, list) → delegate ONE verification task with different wording/approach before completing.
   d. Otherwise → delegate the next subtask.
3. TRUST SUBAGENT DATA: SubAgent results come from real-time search and computation. Trust empirical findings over your own prior knowledge. Only override a SubAgent result if another SubAgent provides contradicting evidence from a verifiable source.

BUDGET AWARENESS:
- You have LIMITED attempts (see Progress below)
- Each delegation costs time and resources — choose models wisely based on task complexity

==== MODEL SELECTION GUIDE ====
{model_pricing_table}

Note: Higher-priced models are generally more capable. Price correlates with model strength.

Model Selection Strategy:
- Choose cheaper models for simple tasks
- Choose more capable models for complex reasoning or critical attempts

==== Progress ====
[Attempt {attempt_index}/{max_attempts}] Remaining {remaining_attempts} attempts

==== QUESTION ====
{instruction}

==== SUBTASK HISTORY ====
{subtask_history if subtask_history else "No subtasks completed yet."}

==== OUTPUT ====
BEFORE completing, re-read the QUESTION carefully and verify:
- Units: if the question asks "how many thousand km", answer "5" not "5000". Always match the unit the question specifies.
- Precision: if it says "to 2 decimal places", give "3.14" not "3.1" or "3.141". If it says "rounded to nearest tenth", give "7.3" not "7.28".
- Format: if it asks for "comma-separated list in alphabetical order", sort your items. If it specifies a date format like DD/MM/YYYY, use that exact format.
- Implicit precision: if the question gives an example answer like "so you'd give 12.5", match that level of precision in your own answer.
- Answer-only: provide ONLY the raw value (number, name, date, list). No labels, no units (unless asked), no extra words.
- Numbers: write numbers in plain numeric form (e.g. "100000000" not "100 million", "2" not "two").

Return JSON:

If results are SUFFICIENT:
{{
  "action": "complete",
  "reasoning": "The subtask results show [X], which answers the question. Verified: units=[Y], precision=[Z].",
  "params": {{ "answer": "concise answer" }}
}}

If more work is NEEDED:
{{
  "action": "delegate_task",
  "reasoning": "We have [X] from previous attempts, but still need [Y] to answer the question",
  "params": {{
    "task_instruction": "A SPECIFIC, ACTIONABLE subtask",
    "context": "Verified facts from previous SubAgent results only. Do not include your own guesses or interpretations.",
    "model": "one of {sub_models}"
  }}
}}
""".strip()
