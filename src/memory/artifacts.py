from __future__ import annotations

from typing import Any, Dict


class ArtifactsStore:
    """Lightweight dictionary-like storage for intermediate objects."""

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def reset(self) -> None:
        self._store.clear()

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._store)

