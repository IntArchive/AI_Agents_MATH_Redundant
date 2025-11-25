from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from src.agents.base import Agent
from src.memory.artifacts import ArtifactsStore
from src.memory.transcript import Transcript
from src.schemas.messages import AgentState, Message, Task


@dataclass
class Role:
    name: str
    agent: Agent


class Orchestrator:
    """Round-robin controller coordinating agents and shared memory."""

    def __init__(
        self,
        roles: Iterable[Role],
        max_rounds: int = 4,
        transcript: Transcript | None = None,
        artifacts: ArtifactsStore | None = None,
    ) -> None:
        self.roles: List[Role] = list(roles)
        self.max_rounds = max_rounds
        self.transcript = transcript or Transcript()
        self.artifacts = artifacts or ArtifactsStore()

    def run(self, task: Task) -> Dict[str, str]:
        outputs: Dict[str, str] = {}
        self.transcript.reset([Message(speaker="user", content=task.description)])
        self.artifacts.reset()

        for round_idx in range(1, self.max_rounds + 1):
            for role in self.roles:
                state = AgentState(
                    task=task,
                    memory=self.transcript.all(),
                    artifacts=self.artifacts.as_dict(),
                    round_idx=round_idx,
                )
                response = role.agent.step(state)
                self.transcript.append(response)
                outputs[role.name] = response.content

                if any(line.strip().startswith("final:") for line in response.content.splitlines()):
                    return outputs

        return outputs

