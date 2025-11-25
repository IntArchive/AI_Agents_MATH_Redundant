from src.evaluation.metrics import EvaluationResult, success_rate


def test_success_rate():
    results = [
        EvaluationResult(task_id="a", success=True),
        EvaluationResult(task_id="b", success=False),
    ]
    assert success_rate(results) == 0.5

