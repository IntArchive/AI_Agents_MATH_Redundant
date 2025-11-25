from __future__ import annotations

from typing import Any, Dict

from src.tools.base import Tool


class ProofCheckerTool(Tool):
    """Toy proof checker that enforces deterministic heuristics for tests."""

    def __init__(self) -> None:
        super().__init__(name="proof_checker")

    def run(self, query: Dict[str, Any]) -> Dict[str, Any]:
        proof_text = (query.get("solution") or "").strip()
        problem = (query.get("problem") or "").strip()

        if not proof_text:
            return {"ok": False, "reason": "empty proof"}

        indicators = ["therefore", "hence", "thus", "qed"]
        confident = any(token in proof_text.lower() for token in indicators)
        contains_problem_ref = bool(problem) and problem.lower() in proof_text.lower()

        verdict = confident or contains_problem_ref
        return {
            "ok": verdict,
            "reason": "heuristic match" if verdict else "missing justification",
        }


