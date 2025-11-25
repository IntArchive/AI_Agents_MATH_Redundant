from src.schemas.messages import Task
from src.workflows.orchestrator import Orchestrator, Role


class DummyAgent:
    def __init__(self, name: str):
        self.name = name

    def step(self, state):
        from src.schemas.messages import Message

        return Message(speaker=self.name, content="final: done")


def test_orchestrator_stops_on_final():
    orchestrator = Orchestrator(roles=[Role("judge", DummyAgent("judge"))], max_rounds=2)
    outputs = orchestrator.run(Task(id="1", description="Test"))
    assert "judge" in outputs

