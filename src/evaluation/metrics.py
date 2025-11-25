from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class EvaluationResult:
    task_id: str
    success: bool
    latency_ms: int | None = None


def success_rate(results: List[EvaluationResult]) -> float:
    if not results:
        return 0.0
    successes = sum(1 for r in results if r.success)
    return successes / len(results)

