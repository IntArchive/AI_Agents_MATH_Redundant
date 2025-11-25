from src.agents.judge import JudgeAgent
from src.schemas.messages import AgentState, Message, Task


def build_state(description: str) -> AgentState:
    task = Task(id="t1", description=description)
    memory = [Message(speaker="user", content=description)]
    return AgentState(task=task, memory=memory, artifacts={}, round_idx=1)


def test_judge_detects_redundant_assumption(tmp_path):
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Judge prompt", encoding="utf-8")
    agent = JudgeAgent(prompt_path=str(prompt))

    description = "Assumption 1: redundant angle.\nAssumption 2: standard."
    message = agent.step(build_state(description))

    assert "New_problem" in message.content or "New_problem:" in message.content
    assert message.metadata["structured"]["redundant_assumption"] is not None


def test_judge_handles_absence(tmp_path):
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Judge prompt", encoding="utf-8")
    agent = JudgeAgent(prompt_path=str(prompt))

    description = "Assumption 1: unique angle."
    message = agent.step(build_state(description))

    assert "Redundant Assumption: no" in message.content
    assert message.metadata["structured"]["redundant_assumption"] is None


