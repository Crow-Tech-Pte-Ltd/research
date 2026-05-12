from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .schema import ChoicePair, ContextSpec, ModelSpec, RunConfig

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_models(path: str | Path = ROOT / "configs/models.yaml") -> list[ModelSpec]:
    data = load_yaml(path)
    return [ModelSpec(**item) for item in data["models"]]


def load_choice_pairs(path: str | Path = ROOT / "data/questions.yaml") -> list[ChoicePair]:
    data = load_yaml(path)
    return [ChoicePair(**item) for item in data["choice_pairs"]]


def load_contexts(path: str | Path = ROOT / "data/contexts.yaml") -> list[ContextSpec]:
    data = load_yaml(path)
    return [ContextSpec(**item) for item in data["contexts"]]


def load_run_config(path: str | Path = ROOT / "configs/run_pilot.yaml") -> RunConfig:
    data = load_yaml(path)
    return RunConfig(**data)


def load_local_env() -> None:
    load_dotenv(ROOT / ".env", override=False)
