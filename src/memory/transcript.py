from __future__ import annotations

from typing import Iterable, List

from src.schemas.messages import Message


class Transcript:
    """Append-only conversation log backing the orchestrator."""

    def __init__(self) -> None:
        self._turns: List[Message] = []

    def reset(self, initial: Iterable[Message] | None = None) -> None:
        self._turns = list(initial or [])

    def append(self, message: Message) -> None:
        self._turns.append(message)

    def last(self, k: int = 1) -> List[Message]:
        if k <= 0:
            return []
        return self._turns[-k:]

    def all(self) -> List[Message]:
        return list(self._turns)

