from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.agents.base import Agent
from src.schemas.messages import AgentState, Message
from src.utils.llm_clients import LLMClient


@dataclass
class JudgeResult:
    answer_to_Q1: str
    assumptions: List[str]
    redundant_assumption: Optional[str]
    new_problem: Optional[str]
    solution_for_new_problem: Optional[str]


class JudgeAgent(Agent):
    """Rule-based stand-in for the judge agent."""

    def __init__(
        self,
        prompt_path: str,
        llm_client: LLMClient | None = None,
    ) -> None:
        super().__init__(name="judge")
        self.prompt = Path(prompt_path).read_text(encoding="utf-8")
        self.llm_client = llm_client

    def step(self, state: AgentState) -> Message:
        if self.llm_client:
            self.llm_client.complete(self.prompt, state.task, state.memory)

        structured = self._determine_redundancy(state.task.description)
        content = self._format_handoff(structured)
        return Message(
            speaker=self.name,
            content=content,
            metadata={"structured": structured.__dict__},
        )

    def _determine_redundancy(self, description: str) -> JudgeResult:
        assumptions = self._extract_assumptions(description)
        redundant = next((a for a in assumptions if "redundant" in a.lower()), None)
        has_redundant = redundant is not None

        answer = "Yes" if has_redundant else "No"
        new_problem = None
        solution = None
        if has_redundant:
            base_statement = redundant.split(":", 1)[-1].strip()
            new_problem = f"Show that {base_statement} follows from the remaining assumptions."
            solution = (
                f"Therefore, {base_statement} is derivable without assuming it directly."
            )

        return JudgeResult(
            answer_to_Q1=answer,
            assumptions=assumptions,
            redundant_assumption=redundant,
            new_problem=new_problem,
            solution_for_new_problem=solution,
        )

    @staticmethod
    def _extract_assumptions(description: str) -> List[str]:
        assumptions: List[str] = []
        for line in description.splitlines():
            line = line.strip()
            if line.lower().startswith("assumption"):
                assumptions.append(line)
        return assumptions or ["Assumption 1: (unspecified)"]

    @staticmethod
    def _format_handoff(result: JudgeResult) -> str:
        lines = [f"Answer to Q1: {result.answer_to_Q1}"]
        if result.redundant_assumption:
            lines.append("Assumptions (without redundant one):")
            filtered = [
                a
                for a in result.assumptions
                if a != result.redundant_assumption
            ]
            lines.extend(filtered or ["(none)"])
            lines.append("New_problem:")
            lines.append(result.new_problem or "")
            if result.solution_for_new_problem:
                lines.append("Solution_for_new_problem:")
                lines.append(result.solution_for_new_problem)
        else:
            lines.append("Redundant Assumption: no")
        return "\n".join(lines).strip()


