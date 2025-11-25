from src.tools.proof_checker import ProofCheckerTool


def test_proof_checker_requires_signal():
    tool = ProofCheckerTool()
    result = tool.run({"problem": "Show 1=1", "solution": "Thus, 1 = 1. QED."})
    assert result["ok"] is True


def test_proof_checker_rejects_empty():
    tool = ProofCheckerTool()
    result = tool.run({"problem": "P", "solution": ""})
    assert result["ok"] is False


