from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llm_wallet_guard_study.config import load_config


@pytest.fixture
def base_config() -> dict[str, Any]:
    return load_config(Path("configs/pilot.simulator.yaml"))


def write_config(tmp_path: Path, config: dict[str, Any], run_id: str) -> Path:
    config = json.loads(json.dumps(config))
    config["run_id"] = run_id
    path = tmp_path / f"{run_id}.yaml"
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return path
