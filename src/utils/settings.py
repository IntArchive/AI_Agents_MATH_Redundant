from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel


class LLMConfig(BaseModel):
    provider: str
    model: str
    temperature: float


class AgentPromptConfig(BaseModel):
    prompt_path: str


class WorkflowConfig(BaseModel):
    max_rounds: int


class LoggingConfig(BaseModel):
    level: str


class AppConfig(BaseModel):
    llm: LLMConfig
    agents: Dict[str, AgentPromptConfig]
    workflow: WorkflowConfig
    logging: LoggingConfig


def load_config(env: str = "base") -> AppConfig:
    base = _read_yaml(Path("configs/base.yaml"))
    if env != "base":
        override_path = Path(f"configs/{env}.yaml")
        if override_path.exists():
            base = _merge_dicts(base, _read_yaml(override_path))
    return AppConfig(
        llm=LLMConfig(**base["llm"]),
        agents={k: AgentPromptConfig(**v) for k, v in base.get("agents", {}).items()},
        workflow=WorkflowConfig(**base["workflow"]),
        logging=LoggingConfig(**base.get("logging", {"level": "INFO"})),
    )


def _read_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = _merge_dicts(base[key], value)
        else:
            merged[key] = value
    return merged

