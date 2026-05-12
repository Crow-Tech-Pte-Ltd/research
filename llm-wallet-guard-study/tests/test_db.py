from __future__ import annotations

from llm_wallet_guard_study.db import connect, initialize_database
from llm_wallet_guard_study.runner import prepare_trials

from .conftest import write_config


def test_schema_creates_expected_tables(tmp_path):
    db_path = tmp_path / "study.sqlite3"
    initialize_database(db_path)
    with connect(db_path) as conn:
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        }
    expected = {
        "runs",
        "models",
        "prompt_versions",
        "trial_conditions",
        "trials",
        "attempts",
        "model_invocations",
        "wallets",
        "simulator_transfers",
        "policy_events",
        "exports",
    }
    assert expected.issubset(tables)


def test_prepare_creates_trials_idempotently_and_no_attempts(tmp_path, base_config):
    db_path = tmp_path / "study.sqlite3"
    config_path = write_config(tmp_path, base_config, "prepare-idempotent")
    first = prepare_trials(db_path, config_path)
    second = prepare_trials(db_path, config_path)
    assert first["trials"] == 3
    assert second["trials"] == 3
    with connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) AS c FROM trials").fetchone()["c"] == 3
        assert conn.execute("SELECT COUNT(*) AS c FROM attempts").fetchone()["c"] == 0
