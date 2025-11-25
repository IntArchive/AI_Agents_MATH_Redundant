from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    """Single turn exchanged between agents, tools, or user."""

    speaker: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """Top-level request the orchestrator must satisfy."""

    id: str
    description: str
    context: Optional[Dict[str, Any]] = None


@dataclass
class AgentState:
    """Shared view passed into each agent step."""

    task: Task
    memory: List[Message]
    artifacts: Dict[str, Any] = field(default_factory=dict)
    round_idx: int = 0

