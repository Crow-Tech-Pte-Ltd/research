"""SQLite schema and connection helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from . import SCHEMA_VERSION


def utc_now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # Keep the CLI portable across temporary filesystems and container mounts.
    # Some environments can block indefinitely when forcing WAL mode on /tmp.
    conn.execute("PRAGMA journal_mode = DELETE")
    return conn


def initialize_database(db_path: str | Path) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        _apply_schema_migrations(conn)
        now = utc_now_iso()
        conn.execute(
            """
            INSERT INTO schema_metadata(key, value, updated_at_utc)
            VALUES('schema_version', ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at_utc=excluded.updated_at_utc
            """,
            (str(SCHEMA_VERSION), now),
        )
        conn.execute(
            """
            INSERT INTO schema_metadata(key, value, updated_at_utc)
            VALUES('initialized_at_utc', ?, ?)
            ON CONFLICT(key) DO NOTHING
            """,
            (now, now),
        )


def _apply_schema_migrations(conn: sqlite3.Connection) -> None:
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(model_invocations)")
    }
    if "metadata_json" not in columns:
        conn.execute(
            "ALTER TABLE model_invocations ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'"
        )


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    created_at_utc TEXT NOT NULL,
    schema_version INTEGER NOT NULL CHECK (schema_version > 0),
    config_path TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    random_seed INTEGER NOT NULL,
    threat_model TEXT NOT NULL CHECK (threat_model IN ('black_box', 'gray_box', 'white_box')),
    tool_mode TEXT NOT NULL CHECK (tool_mode IN ('simulator_decoy_only', 'simulator_allowlisted')),
    max_attempts INTEGER NOT NULL CHECK (max_attempts > 0),
    attacker_word_cap INTEGER NOT NULL CHECK (attacker_word_cap > 0),
    attacker_word_cap_policy TEXT NOT NULL CHECK (attacker_word_cap_policy IN ('truncate', 'reject')),
    inter_attempt_delay_seconds REAL NOT NULL CHECK (inter_attempt_delay_seconds >= 0),
    compromise_on_unsafe_intent INTEGER NOT NULL CHECK (compromise_on_unsafe_intent IN (0, 1)),
    notes TEXT
);

CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('guardian', 'attacker')),
    adapter TEXT NOT NULL,
    stub_class TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    UNIQUE(run_id, model_id, role)
);

CREATE TABLE IF NOT EXISTS prompt_versions (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    prompt_name TEXT NOT NULL,
    prompt_path TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    UNIQUE(run_id, prompt_name)
);

CREATE TABLE IF NOT EXISTS trial_conditions (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    condition_id TEXT NOT NULL,
    name TEXT NOT NULL,
    threat_model TEXT NOT NULL CHECK (threat_model IN ('black_box', 'gray_box', 'white_box')),
    tool_mode TEXT NOT NULL CHECK (tool_mode IN ('simulator_decoy_only', 'simulator_allowlisted')),
    guardian_policy TEXT NOT NULL,
    max_attempts INTEGER NOT NULL CHECK (max_attempts > 0),
    attacker_word_cap INTEGER NOT NULL CHECK (attacker_word_cap > 0),
    attacker_word_cap_policy TEXT NOT NULL CHECK (attacker_word_cap_policy IN ('truncate', 'reject')),
    inter_attempt_delay_seconds REAL NOT NULL CHECK (inter_attempt_delay_seconds >= 0),
    source_wallet_id TEXT NOT NULL,
    destination_wallet_id TEXT NOT NULL,
    initial_balance REAL NOT NULL CHECK (initial_balance >= 0),
    destination_initial_balance REAL NOT NULL CHECK (destination_initial_balance >= 0),
    amount_cap REAL NOT NULL CHECK (amount_cap >= 0),
    attempt_amount REAL NOT NULL CHECK (attempt_amount >= 0),
    allowlisted_destinations_json TEXT NOT NULL,
    allow_simulator_transfers INTEGER NOT NULL CHECK (allow_simulator_transfers IN (0, 1)),
    attacker_context TEXT,
    created_at_utc TEXT NOT NULL,
    UNIQUE(run_id, condition_id)
);

CREATE TABLE IF NOT EXISTS trials (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    condition_record_id TEXT NOT NULL REFERENCES trial_conditions(id) ON DELETE CASCADE,
    condition_id TEXT NOT NULL,
    guardian_model_id TEXT NOT NULL,
    attacker_model_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'in_progress', 'guardian_resilient', 'guardian_compromised', 'error', 'incomplete')),
    outcome TEXT CHECK (outcome IS NULL OR outcome IN ('unsafe_stated_intent', 'attempted_transfer_tool_call', 'policy_blocked_transfer', 'queued_simulator_transfer', 'guardian_resilient', 'error', 'incomplete')),
    outcome_subtype TEXT,
    attempts_completed INTEGER NOT NULL DEFAULT 0 CHECK (attempts_completed >= 0),
    source_wallet_id TEXT NOT NULL,
    destination_wallet_id TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    started_at_utc TEXT,
    completed_at_utc TEXT,
    UNIQUE(run_id, condition_id, guardian_model_id, attacker_model_id)
);

CREATE TABLE IF NOT EXISTS attempts (
    id TEXT PRIMARY KEY,
    trial_id TEXT NOT NULL REFERENCES trials(id) ON DELETE CASCADE,
    attempt_number INTEGER NOT NULL CHECK (attempt_number > 0),
    attacker_message TEXT NOT NULL,
    attacker_word_count INTEGER NOT NULL CHECK (attacker_word_count >= 0),
    attacker_truncated INTEGER NOT NULL CHECK (attacker_truncated IN (0, 1)),
    guardian_message TEXT NOT NULL,
    guardian_unsafe_intent INTEGER NOT NULL CHECK (guardian_unsafe_intent IN (0, 1)),
    guardian_attempted_transfer INTEGER NOT NULL CHECK (guardian_attempted_transfer IN (0, 1)),
    simulator_transfer_id TEXT,
    outcome TEXT CHECK (outcome IS NULL OR outcome IN ('unsafe_stated_intent', 'attempted_transfer_tool_call', 'policy_blocked_transfer', 'queued_simulator_transfer', 'guardian_resilient', 'error', 'incomplete')),
    outcome_subtype TEXT,
    created_at_utc TEXT NOT NULL,
    UNIQUE(trial_id, attempt_number)
);

CREATE TABLE IF NOT EXISTS model_invocations (
    id TEXT PRIMARY KEY,
    trial_id TEXT NOT NULL REFERENCES trials(id) ON DELETE CASCADE,
    attempt_number INTEGER NOT NULL CHECK (attempt_number > 0),
    role TEXT NOT NULL CHECK (role IN ('guardian', 'attacker')),
    model_id TEXT NOT NULL,
    input_text TEXT NOT NULL,
    output_text TEXT NOT NULL,
    action_json TEXT,
    duration_ms REAL NOT NULL CHECK (duration_ms >= 0),
    redaction_applied INTEGER NOT NULL CHECK (redaction_applied IN (0, 1)),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS wallets (
    id TEXT PRIMARY KEY,
    trial_id TEXT NOT NULL,
    wallet_id TEXT NOT NULL,
    balance REAL NOT NULL CHECK (balance >= 0),
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    UNIQUE(trial_id, wallet_id)
);

CREATE TABLE IF NOT EXISTS simulator_transfers (
    id TEXT PRIMARY KEY,
    trial_id TEXT NOT NULL,
    from_wallet TEXT,
    to_wallet TEXT,
    amount REAL CHECK (amount IS NULL OR amount >= 0),
    status TEXT NOT NULL CHECK (status IN ('proposed', 'queued', 'blocked')),
    reason TEXT,
    source_text TEXT,
    policy_reason_code TEXT,
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS policy_events (
    id TEXT PRIMARY KEY,
    trial_id TEXT REFERENCES trials(id) ON DELETE CASCADE,
    attempt_number INTEGER CHECK (attempt_number IS NULL OR attempt_number > 0),
    simulator_transfer_id TEXT,
    decision TEXT NOT NULL CHECK (decision IN ('allow', 'block')),
    reason_code TEXT NOT NULL,
    details_json TEXT NOT NULL,
    created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS exports (
    id TEXT PRIMARY KEY,
    created_at_utc TEXT NOT NULL,
    source_db_path TEXT NOT NULL,
    output_dir TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    row_counts_json TEXT NOT NULL,
    config_hash TEXT,
    prompt_hashes_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_trials_status ON trials(status);
CREATE INDEX IF NOT EXISTS idx_attempts_trial ON attempts(trial_id, attempt_number);
CREATE INDEX IF NOT EXISTS idx_policy_events_trial ON policy_events(trial_id, attempt_number);
CREATE INDEX IF NOT EXISTS idx_transfers_trial ON simulator_transfers(trial_id);
"""
