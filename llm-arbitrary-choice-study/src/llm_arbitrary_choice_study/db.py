from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS trials (
    trial_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    pair_id TEXT NOT NULL,
    condition TEXT NOT NULL,
    repetition INTEGER NOT NULL,
    temperature REAL NOT NULL,
    option_1 TEXT NOT NULL,
    option_2 TEXT NOT NULL,
    context_id TEXT,
    prompt TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    response_text TEXT,
    parsed_choice TEXT,
    parse_status TEXT,
    error_type TEXT,
    error_message TEXT,
    request_json TEXT,
    response_json TEXT,
    usage_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (trial_id, model_id)
);
CREATE INDEX IF NOT EXISTS idx_trials_model_status ON trials(model_id, status);
CREATE INDEX IF NOT EXISTS idx_trials_pair_condition ON trials(pair_id, condition);
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    trial_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    attempt_index INTEGER NOT NULL,
    max_tokens INTEGER NOT NULL,
    status TEXT NOT NULL,
    response_text TEXT,
    parsed_choice TEXT,
    parse_status TEXT,
    error_type TEXT,
    error_message TEXT,
    request_json TEXT,
    response_json TEXT,
    usage_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (trial_id, model_id) REFERENCES trials(trial_id, model_id)
);
CREATE INDEX IF NOT EXISTS idx_attempts_trial_model ON attempts(trial_id, model_id, attempt_index);
CREATE INDEX IF NOT EXISTS idx_attempts_model_status ON attempts(model_id, status);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def enqueue_trial(conn: sqlite3.Connection, model_id: str, trial: Any) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO trials (
            trial_id, model_id, pair_id, condition, repetition, temperature,
            option_1, option_2, context_id, prompt
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            trial.trial_id, model_id, trial.pair_id, trial.condition, trial.repetition, trial.temperature,
            trial.option_1, trial.option_2, trial.context_id, trial.prompt,
        ),
    )


def next_pending(conn: sqlite3.Connection) -> sqlite3.Row | None:
    row = conn.execute(
        "SELECT * FROM trials WHERE status = 'pending' ORDER BY rowid LIMIT 1"
    ).fetchone()
    return row


def mark_result(
    conn: sqlite3.Connection,
    trial_id: str,
    model_id: str,
    *,
    status: str,
    response_text: str | None = None,
    parsed_choice: str | None = None,
    parse_status: str | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    request_json: dict[str, Any] | None = None,
    response_json: dict[str, Any] | None = None,
    usage_json: dict[str, Any] | None = None,
) -> None:
    conn.execute(
        """
        UPDATE trials
        SET status=?, response_text=?, parsed_choice=?, parse_status=?, error_type=?, error_message=?,
            request_json=?, response_json=?, usage_json=?, updated_at=CURRENT_TIMESTAMP
        WHERE trial_id=? AND model_id=?
        """,
        (
            status, response_text, parsed_choice, parse_status, error_type, error_message,
            json.dumps(request_json, ensure_ascii=False) if request_json is not None else None,
            json.dumps(response_json, ensure_ascii=False) if response_json is not None else None,
            json.dumps(usage_json, ensure_ascii=False) if usage_json is not None else None,
            trial_id, model_id,
        ),
    )


def record_attempt(
    conn: sqlite3.Connection,
    trial_id: str,
    model_id: str,
    *,
    attempt_index: int,
    max_tokens: int,
    status: str,
    response_text: str | None = None,
    parsed_choice: str | None = None,
    parse_status: str | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    request_json: dict[str, Any] | None = None,
    response_json: dict[str, Any] | None = None,
    usage_json: dict[str, Any] | None = None,
) -> None:
    """Persist every provider attempt, including attempts superseded by retries."""
    conn.execute(
        """
        INSERT INTO attempts (
            trial_id, model_id, attempt_index, max_tokens, status, response_text, parsed_choice,
            parse_status, error_type, error_message, request_json, response_json, usage_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            trial_id, model_id, attempt_index, max_tokens, status, response_text, parsed_choice,
            parse_status, error_type, error_message,
            json.dumps(request_json, ensure_ascii=False) if request_json is not None else None,
            json.dumps(response_json, ensure_ascii=False) if response_json is not None else None,
            json.dumps(usage_json, ensure_ascii=False) if usage_json is not None else None,
        ),
    )


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute("SELECT status, count(*) AS n FROM trials GROUP BY status").fetchall()
    return {row["status"]: row["n"] for row in rows}


def model_attempt_count(conn: sqlite3.Connection, model_id: str) -> int:
    row = conn.execute("SELECT count(*) AS n FROM attempts WHERE model_id=?", (model_id,)).fetchone()
    return int(row["n"])


def pause_pending_for_model(conn: sqlite3.Connection, model_id: str, *, status: str) -> int:
    cursor = conn.execute(
        """
        UPDATE trials
        SET status=?, updated_at=CURRENT_TIMESTAMP
        WHERE model_id=? AND status='pending'
        """,
        (status, model_id),
    )
    return cursor.rowcount


def reset_statuses(conn: sqlite3.Connection, statuses: list[str]) -> int:
    if not statuses:
        return 0
    placeholders = ", ".join("?" for _ in statuses)
    cursor = conn.execute(
        f"""
        UPDATE trials
        SET status='pending', error_type=NULL, error_message=NULL, updated_at=CURRENT_TIMESTAMP
        WHERE status IN ({placeholders})
        """,
        statuses,
    )
    return cursor.rowcount
