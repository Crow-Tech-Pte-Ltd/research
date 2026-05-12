from __future__ import annotations

from llm_wallet_guard_study.db import connect
from llm_wallet_guard_study.runner import run_pilot

from .conftest import write_config


def test_safe_guardian_trial_ends_resilient(tmp_path, base_config):
    config = base_config
    config["run_id"] = "safe-run"
    config["max_attempts"] = 4
    config["conditions"][0]["max_attempts"] = 4
    config["models"]["guardians"] = [config["models"]["guardians"][0]]
    db_path = tmp_path / "safe.sqlite3"
    config_path = write_config(tmp_path, config, "safe-run")
    result = run_pilot(db_path, config_path, no_delay=True)
    assert result["guardian_resilient"] == 1
    with connect(db_path) as conn:
        trial = conn.execute("SELECT * FROM trials").fetchone()
        assert trial["status"] == "guardian_resilient"
        assert trial["attempts_completed"] == 4
        assert conn.execute("SELECT COUNT(*) AS c FROM attempts").fetchone()["c"] == 4


def test_vulnerable_guardian_stops_after_expected_compromise(tmp_path, base_config):
    config = base_config
    config["run_id"] = "vulnerable-run"
    config["max_attempts"] = 6
    config["conditions"][0]["max_attempts"] = 6
    config["models"]["guardians"] = [config["models"]["guardians"][1]]
    db_path = tmp_path / "vulnerable.sqlite3"
    config_path = write_config(tmp_path, config, "vulnerable-run")
    result = run_pilot(db_path, config_path, no_delay=True)
    assert result["guardian_compromised"] == 1
    with connect(db_path) as conn:
        trial = conn.execute("SELECT * FROM trials").fetchone()
        assert trial["status"] == "guardian_compromised"
        assert trial["outcome"] == "attempted_transfer_tool_call"
        assert trial["attempts_completed"] == 3
        assert conn.execute("SELECT COUNT(*) AS c FROM attempts").fetchone()["c"] == 3
        assert conn.execute("SELECT COUNT(*) AS c FROM policy_events").fetchone()["c"] == 1
        transfer = conn.execute("SELECT * FROM simulator_transfers").fetchone()
        assert transfer["status"] == "blocked"
        assert transfer["policy_reason_code"] == "guardian_transfer_forbidden"


def test_borderline_guardian_records_unsafe_intent_without_transfer(tmp_path, base_config):
    config = base_config
    config["run_id"] = "borderline-run"
    config["max_attempts"] = 5
    config["conditions"][0]["max_attempts"] = 5
    config["models"]["guardians"] = [config["models"]["guardians"][2]]
    db_path = tmp_path / "borderline.sqlite3"
    config_path = write_config(tmp_path, config, "borderline-run")
    result = run_pilot(db_path, config_path, no_delay=True)
    assert result["guardian_compromised"] == 1
    with connect(db_path) as conn:
        trial = conn.execute("SELECT * FROM trials").fetchone()
        assert trial["status"] == "guardian_compromised"
        assert trial["outcome"] == "unsafe_stated_intent"
        assert trial["outcome_subtype"] is None
        assert trial["attempts_completed"] == 2
        attempt = conn.execute(
            "SELECT * FROM attempts WHERE attempt_number = 2"
        ).fetchone()
        assert attempt["guardian_unsafe_intent"] == 1
        assert attempt["guardian_attempted_transfer"] == 0
        assert conn.execute("SELECT COUNT(*) AS c FROM policy_events").fetchone()["c"] == 0
        assert conn.execute("SELECT COUNT(*) AS c FROM simulator_transfers").fetchone()["c"] == 0


def test_completed_trials_are_not_rerun(tmp_path, base_config):
    config = base_config
    config["run_id"] = "resume-run"
    config["max_attempts"] = 3
    config["conditions"][0]["max_attempts"] = 3
    config["models"]["guardians"] = [config["models"]["guardians"][0]]
    db_path = tmp_path / "resume.sqlite3"
    config_path = write_config(tmp_path, config, "resume-run")
    first = run_pilot(db_path, config_path, no_delay=True)
    second = run_pilot(db_path, config_path, no_delay=True)
    assert first["claimed"] == 1
    assert second["claimed"] == 0
    with connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) AS c FROM attempts").fetchone()["c"] == 3


def test_stale_in_progress_trial_is_marked_incomplete_on_next_run(tmp_path, base_config):
    config = base_config
    config["run_id"] = "stale-recovery-run"
    config["models"]["guardians"] = [config["models"]["guardians"][0]]
    db_path = tmp_path / "stale.sqlite3"
    config_path = write_config(tmp_path, config, "stale-recovery-run")

    # Prepare once, then simulate a process crash after a trial was claimed.
    run_pilot(db_path, config_path, max_trials=0, no_delay=True)
    with connect(db_path) as conn:
        trial_id = conn.execute("SELECT id FROM trials").fetchone()["id"]
        conn.execute(
            "UPDATE trials SET status = 'in_progress', started_at_utc = '2026-01-01T00:00:00Z' WHERE id = ?",
            (trial_id,),
        )
        conn.commit()

    result = run_pilot(db_path, config_path, no_delay=True)
    assert result["claimed"] == 0
    assert result["recovered_incomplete"] == 1
    with connect(db_path) as conn:
        trial = conn.execute("SELECT * FROM trials WHERE id = ?", (trial_id,)).fetchone()
        assert trial["status"] == "incomplete"
        assert trial["outcome"] == "incomplete"
        assert trial["outcome_subtype"] == "recovered_stale_in_progress_trial"


def test_stale_in_progress_trial_records_completed_attempts(tmp_path, base_config):
    config = base_config
    config["run_id"] = "partial-stale-run"
    config["models"]["guardians"] = [config["models"]["guardians"][0]]
    db_path = tmp_path / "partial-stale.sqlite3"
    config_path = write_config(tmp_path, config, "partial-stale-run")

    run_pilot(db_path, config_path, max_trials=0, no_delay=True)
    with connect(db_path) as conn:
        trial_id = conn.execute("SELECT id FROM trials").fetchone()["id"]
        conn.execute("UPDATE trials SET status = 'in_progress' WHERE id = ?", (trial_id,))
        conn.execute(
            """
            INSERT INTO attempts(
                id, trial_id, attempt_number, attacker_message, attacker_word_count,
                attacker_truncated, guardian_message, guardian_unsafe_intent,
                guardian_attempted_transfer, simulator_transfer_id, outcome,
                outcome_subtype, created_at_utc
            ) VALUES('attempt-partial-1', ?, 1, 'msg', 1, 0, 'safe', 0, 0, NULL, NULL, NULL, '2026-01-01T00:00:00Z')
            """,
            (trial_id,),
        )
        conn.commit()

    result = run_pilot(db_path, config_path, no_delay=True)
    assert result["recovered_incomplete"] == 1
    with connect(db_path) as conn:
        trial = conn.execute("SELECT * FROM trials WHERE id = ?", (trial_id,)).fetchone()
        assert trial["attempts_completed"] == 1


def test_guardian_model_filter_only_claims_matching_trials(tmp_path, base_config):
    config = base_config
    config["run_id"] = "guardian-filter-run"
    config["max_attempts"] = 3
    config["conditions"][0]["max_attempts"] = 3
    db_path = tmp_path / "guardian-filter.sqlite3"
    config_path = write_config(tmp_path, config, "guardian-filter-run")

    result = run_pilot(
        db_path,
        config_path,
        guardian_model_id="vulnerable-guardian-stub",
        no_delay=True,
    )

    assert result["claimed"] == 1
    assert result["guardian_compromised"] == 1
    with connect(db_path) as conn:
        statuses = {
            row["guardian_model_id"]: row["status"]
            for row in conn.execute("SELECT guardian_model_id, status FROM trials")
        }
    assert statuses["vulnerable-guardian-stub"] == "guardian_compromised"
    assert statuses["always-safe-guardian-stub"] == "pending"
    assert statuses["borderline-intent-guardian-stub"] == "pending"


def test_parallel_worker_mode_does_not_recover_other_in_progress_trials(tmp_path, base_config):
    config = base_config
    config["run_id"] = "parallel-no-recover-run"
    db_path = tmp_path / "parallel-no-recover.sqlite3"
    config_path = write_config(tmp_path, config, "parallel-no-recover-run")

    run_pilot(db_path, config_path, max_trials=0, no_delay=True)
    with connect(db_path) as conn:
        trial_id = conn.execute(
            "SELECT id FROM trials WHERE guardian_model_id = 'always-safe-guardian-stub'"
        ).fetchone()["id"]
        conn.execute("UPDATE trials SET status = 'in_progress' WHERE id = ?", (trial_id,))
        conn.commit()

    result = run_pilot(
        db_path,
        config_path,
        guardian_model_id="vulnerable-guardian-stub",
        recover_stale=False,
        no_delay=True,
    )

    assert result["recovered_incomplete"] == 0
    assert result["claimed"] == 1
    with connect(db_path) as conn:
        stale_trial = conn.execute("SELECT status FROM trials WHERE id = ?", (trial_id,)).fetchone()
    assert stale_trial["status"] == "in_progress"
