from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class Tool(ABC):
    """Protocol describing a callable capability."""

    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def run(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool logic and return structured output."""

