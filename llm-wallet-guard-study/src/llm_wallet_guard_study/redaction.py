"""Secret redaction helpers used before persistence and export."""

from __future__ import annotations

import json
import re
from typing import Any

REDACTION_MARKER = "[REDACTED_SECRET]"

_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:0x)?[A-Fa-f0-9]{64}\b"),
    re.compile(r"\b(?:sk|pk|rk|ghp)[-_][A-Za-z0-9_=-]{20,}\b"),
    re.compile(
        r"(?i)\b(?:api[_ -]?key|auth[_ -]?token|access[_ -]?token|secret[_ -]?key|private[_ -]?key)"
        r"\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{12,}"
    ),
    re.compile(
        r"(?is)\b(?:seed phrase|mnemonic|recovery phrase)\s*[:=]\s*"
        r"(?:[a-z]{3,12}\s+){11,23}[a-z]{3,12}\b"
    ),
)


def redact_text(value: Any) -> str:
    """Return text with common secret-like values replaced."""

    text = "" if value is None else str(value)
    for pattern in _PATTERNS:
        text = pattern.sub(REDACTION_MARKER, text)
    return text


def redact_structure(value: Any) -> Any:
    """Recursively redact strings in JSON-like structures."""

    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {str(key): redact_structure(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_structure(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_structure(item) for item in value)
    return value


def redact_json_dumps(value: Any) -> str:
    """Serialize a JSON-like value after recursive redaction."""

    return json.dumps(redact_structure(value), sort_keys=True, separators=(",", ":"))


def contains_redaction_marker(value: Any) -> bool:
    return REDACTION_MARKER in redact_text(value)
