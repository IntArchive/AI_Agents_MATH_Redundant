from __future__ import annotations

from src.evaluation.metrics import EvaluationResult, success_rate


def main():
    # Placeholder values until the evaluation harness is wired up.
    dummy = [
        EvaluationResult(task_id="t1", success=True),
        EvaluationResult(task_id="t2", success=False),
    ]
    print(f"Success rate: {success_rate(dummy):.2%}")


if __name__ == "__main__":
    main()

