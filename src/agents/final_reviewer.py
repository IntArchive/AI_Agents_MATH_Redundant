from __future__ import annotations

from typing import Dict, Optional

from src.agents.base import Agent
from src.schemas.messages import AgentState, Message
from src.tools.proof_checker import ProofCheckerTool


class FinalReviewerAgent(Agent):

    def __init__(
        self,
        prompt_path: str,
        proof_checker: ProofCheckerTool,
    ) -> None:
        super().__init__(name="final reviewer", tools=[proof_checker])
        self.proof_checker = proof_checker

    def step(self, state: AgentState) -> Message:
        judge_message = self._find_latest_judge_message(state)
        structured = judge_message.metadata.get("structured") if judge_message else None

        proof_check = self.proof_checker.run(
            {
                "problem": structured.get("new_problem") if structured else None,
                "solution": structured.get("solution_for_new_problem") if structured else None,
            }
        )
        verdict = proof_check.get("ok", False)
        proof_line = f"Proof: {'True' if verdict else 'False'}"

        final_payload = "final: No redundant assumption detected."
        if structured and structured.get("redundant_assumption"):
            final_payload = (
                f"final: Redundant assumption resolved. "
                f"New problem: {structured.get('new_problem')}"
            )

        content = f"{proof_line}\n{final_payload}"
        metadata: Dict[str, Optional[str]] = {
            "proof_ok": str(verdict),
            "reason": proof_check.get("reason"),
        }
        return Message(speaker=self.name, content=content, metadata=metadata)

    @staticmethod
    def _find_latest_judge_message(state: AgentState) -> Message | None:
        for message in reversed(state.memory):
            if message.speaker == "judge":
                return message
        return None


