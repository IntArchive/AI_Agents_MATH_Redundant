from src.agents.final_reviewer import FinalReviewerAgent
from src.agents.judge import JudgeAgent
from src.schemas.messages import Task
from src.tools.proof_checker import ProofCheckerTool
from src.workflows.orchestrator import Orchestrator, Role


def test_orchestrator_stops_on_final(tmp_path):
    judge_prompt = tmp_path / "judge.md"
    reviewer_prompt = tmp_path / "reviewer.md"
    judge_prompt.write_text("judge prompt", encoding="utf-8")
    reviewer_prompt.write_text("reviewer prompt", encoding="utf-8")

    orchestrator = Orchestrator(
        roles=[
            Role("judge", JudgeAgent(prompt_path=str(judge_prompt))),
            Role(
                "final reviewer",
                FinalReviewerAgent(
                    prompt_path=str(reviewer_prompt),
                    proof_checker=ProofCheckerTool(),
                ),
            ),
        ],
        max_rounds=4,
    )
    description = "Assumption 1: redundant angle.\nConclude?"
    outputs = orchestrator.run(Task(id="1", description=description))
    assert "final reviewer" in outputs
    assert "final:" in outputs["final reviewer"]

