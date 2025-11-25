from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from src.schemas.messages import Message, Task


class LLMClient(ABC):
    """Lightweight interface so agents can swap between real and stub models."""

    @abstractmethod
    def complete(self, prompt: str, task: Task, memory: Iterable[Message]) -> str:
        """Return a model completion for the given prompt, task, and transcript."""


class EchoLLMClient(LLMClient):
    """Fallback implementation used for local tests without external APIs."""

    def complete(self, prompt: str, task: Task, memory: Iterable[Message]) -> str:
        transcript = "\n".join(f"{m.speaker}: {m.content}" for m in memory)
        return (
            f"{prompt.strip()}\n\n"
            f"Task: {task.description.strip()}\n"
            f"Transcript:\n{transcript.strip()}"
        )


