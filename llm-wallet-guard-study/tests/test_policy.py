from __future__ import annotations

from llm_wallet_guard_study.db import connect
from llm_wallet_guard_study.policy import PolicyEnforcer
from llm_wallet_guard_study.runner import prepare_trials
from llm_wallet_guard_study.simulator_wallet import SimulatorWallet

from .conftest import write_config


def _prepared_trial(tmp_path, base_config):
    db_path = tmp_path / "policy.sqlite3"
    config_path = write_config(tmp_path, base_config, "policy-run")
    prepare_trials(db_path, config_path)
    conn = connect(db_path)
    trial = conn.execute("SELECT * FROM trials ORDER BY id LIMIT 1").fetchone()
    condition = conn.execute(
        "SELECT * FROM trial_conditions WHERE id = ?", (trial["condition_record_id"],)
    ).fetchone()
    wallet = SimulatorWallet(conn)
    wallet.create_wallet(trial["source_wallet_id"], condition["initial_balance"], trial_id=trial["id"])
    wallet.create_wallet(
        trial["destination_wallet_id"],
        condition["destination_initial_balance"],
        trial_id=trial["id"],
    )
    conn.commit()
    return db_path, conn, trial


def test_policy_blocks_guardian_transfer_by_default(tmp_path, base_config):
    _db_path, conn, trial = _prepared_trial(tmp_path, base_config)
    decision = PolicyEnforcer(conn).evaluate_transfer(
        trial["id"],
        1,
        {
            "from_wallet": trial["source_wallet_id"],
            "to_wallet": trial["destination_wallet_id"],
            "amount": 1.0,
        },
        simulator_transfer_id="transfer-policy-1",
    )
    assert not decision.allowed
    assert decision.reason_code == "guardian_transfer_forbidden"
    assert conn.execute("SELECT COUNT(*) AS c FROM policy_events").fetchone()["c"] == 1
    conn.close()


def test_policy_fail_closed_cases(tmp_path, base_config):
    _db_path, conn, trial = _prepared_trial(tmp_path, base_config)
    enforcer = PolicyEnforcer(conn)
    malformed = enforcer.evaluate_transfer(trial["id"], 1, {"from_wallet": trial["source_wallet_id"]})
    unknown = enforcer.evaluate_transfer(
        trial["id"],
        2,
        {"from_wallet": "unknown_wallet", "to_wallet": trial["destination_wallet_id"], "amount": 1},
    )
    excessive = enforcer.evaluate_transfer(
        trial["id"],
        3,
        {
            "from_wallet": trial["source_wallet_id"],
            "to_wallet": trial["destination_wallet_id"],
            "amount": 999,
        },
    )
    not_allowlisted = enforcer.evaluate_transfer(
        trial["id"],
        4,
        {"from_wallet": trial["source_wallet_id"], "to_wallet": "external_sink", "amount": 1},
    )
    missing_state = enforcer.evaluate_transfer(
        None,
        5,
        {"from_wallet": trial["source_wallet_id"], "to_wallet": trial["destination_wallet_id"], "amount": 1},
    )
    assert [d.reason_code for d in (malformed, unknown, excessive, not_allowlisted, missing_state)] == [
        "malformed_request",
        "unknown_wallet",
        "amount_exceeds_cap",
        "destination_not_allowlisted",
        "blocked_fail_closed",
    ]
    assert conn.execute("SELECT COUNT(*) AS c FROM policy_events").fetchone()["c"] == 5
    conn.close()
