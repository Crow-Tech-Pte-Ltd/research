"""Prompt template discovery and hashing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import hash_file, project_root

PROMPT_FILES = {
    "guardian_system": Path("prompts/guardian_system.md"),
    "attacker_system": Path("prompts/attacker_system.md"),
}


def prompt_versions(root: str | Path | None = None) -> list[dict[str, Any]]:
    base = Path(root) if root is not None else project_root()
    versions: list[dict[str, Any]] = []
    for name, rel_path in PROMPT_FILES.items():
        path = base / rel_path
        versions.append(
            {
                "prompt_name": name,
                "prompt_path": str(rel_path),
                "prompt_hash": hash_file(path),
            }
        )
    return versions
