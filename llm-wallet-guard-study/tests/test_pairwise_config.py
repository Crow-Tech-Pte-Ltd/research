from __future__ import annotations

from llm_wallet_guard_study.db import connect
from llm_wallet_guard_study.runner import prepare_trials
from scripts.make_pairwise_default_openrouter_config import PANEL_MODELS, build_config

from .conftest import write_config


def test_pairwise_default_config_prepares_1875_trials_and_self_play(tmp_path):
    config = build_config()
    assert len(PANEL_MODELS) == 25
    assert len(config["models"]["guardians"]) == 25
    assert len(config["models"]["attackers"]) == 25
    assert len(config["conditions"]) == 3

    db_path = tmp_path / "pairwise-default.sqlite3"
    config_path = write_config(tmp_path, config, "pairwise-default-test")
    result = prepare_trials(db_path, config_path)

    assert result["models"] == 50
    assert result["conditions"] == 3
    assert result["trials"] == 1875
    with connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) AS c FROM trials").fetchone()["c"] == 1875
        self_play = conn.execute(
            """
            SELECT COUNT(*) AS c
            FROM trials
            WHERE guardian_model_id = attacker_model_id
            """
        ).fetchone()["c"]
        assert self_play == 75
        first_model_id = PANEL_MODELS[0][0]
        assert (
            conn.execute(
                """
                SELECT COUNT(*) AS c
                FROM trials
                WHERE guardian_model_id = ? AND attacker_model_id = ?
                """,
                (first_model_id, first_model_id),
            ).fetchone()["c"]
            == 3
        )
