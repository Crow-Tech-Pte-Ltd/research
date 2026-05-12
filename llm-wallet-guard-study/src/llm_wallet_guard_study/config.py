"""Config loading and stable hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the pilot config.

    The checked-in .yaml file is intentionally YAML-compatible JSON so this
    project can remain stdlib-only for the simulator MVP.
    """

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{config_path} must be JSON or YAML-compatible JSON for this stdlib-only MVP"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError("Config root must be an object")
    return data


def load_config_with_hash(path: str | Path) -> tuple[dict[str, Any], str]:
    return load_config(path), hash_file(path)


def require_keys(mapping: dict[str, Any], keys: list[str], context: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"Missing required {context} keys: {', '.join(missing)}")
