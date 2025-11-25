from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List

from src.schemas.messages import AgentState, Message
from src.tools.base import Tool


class Agent(ABC):
    """Base contract for every reasoning agent in the system."""

    name: str

    def __init__(self, name: str, tools: Iterable[Tool] | None = None) -> None:
        self.name = name
        self.tools: List[Tool] = list(tools or [])

    @abstractmethod
    def step(self, state: AgentState) -> Message:
        """Produce the next message based on shared state."""

    def register_tool(self, tool: Tool) -> None:
        self.tools.append(tool)

