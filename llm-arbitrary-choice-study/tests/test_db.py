from llm_arbitrary_choice_study.db import (
    connect,
    counts,
    enqueue_trial,
    mark_result,
    model_attempt_count,
    next_pending,
    pause_pending_for_model,
    record_attempt,
    reset_statuses,
)
from llm_arbitrary_choice_study.schema import Trial


def test_db_enqueue_and_mark_result(tmp_path) -> None:
    db = tmp_path / "test.sqlite3"
    conn = connect(db)
    trial = Trial(
        trial_id="t1", pair_id="p1", condition="bare", repetition=0, temperature=0.7,
        option_1="blue", option_2="red", context_id=None, prompt="Choose one: blue or red.",
    )
    enqueue_trial(conn, "model/a", trial)
    conn.commit()
    assert counts(conn) == {"pending": 1}
    row = next_pending(conn)
    assert row is not None
    mark_result(conn, "t1", "model/a", status="ok", response_text="blue", parsed_choice="blue", parse_status="exact")
    conn.commit()
    assert counts(conn) == {"ok": 1}
    conn.close()


def test_record_attempt_preserves_retry_history(tmp_path) -> None:
    db = tmp_path / "test.sqlite3"
    conn = connect(db)
    trial = Trial(
        trial_id="t1", pair_id="p1", condition="bare", repetition=0, temperature=0.7,
        option_1="blue", option_2="red", context_id=None, prompt="Choose one: blue or red.",
    )
    enqueue_trial(conn, "model/a", trial)
    record_attempt(
        conn,
        "t1",
        "model/a",
        attempt_index=0,
        max_tokens=128,
        status="invalid",
        response_text="",
        parse_status="invalid",
        request_json={"max_tokens": 128},
        response_json={"choices": [{"finish_reason": "length"}]},
        usage_json={"completion_tokens": 128, "cost": 0.01},
    )
    record_attempt(
        conn,
        "t1",
        "model/a",
        attempt_index=1,
        max_tokens=512,
        status="ok",
        response_text="blue",
        parsed_choice="blue",
        parse_status="exact",
        request_json={"max_tokens": 512},
        response_json={"choices": [{"finish_reason": "stop"}]},
        usage_json={"completion_tokens": 12, "cost": 0.001},
    )
    conn.commit()

    rows = conn.execute(
        "SELECT attempt_index, max_tokens, status, response_text FROM attempts ORDER BY attempt_id"
    ).fetchall()
    assert model_attempt_count(conn, "model/a") == 2
    assert [dict(row) for row in rows] == [
        {"attempt_index": 0, "max_tokens": 128, "status": "invalid", "response_text": ""},
        {"attempt_index": 1, "max_tokens": 512, "status": "ok", "response_text": "blue"},
    ]
    conn.close()


def test_pause_pending_for_model_only_pauses_matching_pending_trials(tmp_path) -> None:
    db = tmp_path / "test.sqlite3"
    conn = connect(db)
    trial_a = Trial(
        trial_id="t1", pair_id="p1", condition="bare", repetition=0, temperature=0.7,
        option_1="blue", option_2="red", context_id=None, prompt="Choose one: blue or red.",
    )
    trial_b = Trial(
        trial_id="t2", pair_id="p1", condition="bare", repetition=1, temperature=0.7,
        option_1="blue", option_2="red", context_id=None, prompt="Choose one: blue or red.",
    )
    enqueue_trial(conn, "model/a", trial_a)
    enqueue_trial(conn, "model/a", trial_b)
    enqueue_trial(conn, "model/b", trial_a)
    mark_result(conn, "t1", "model/a", status="ok")
    conn.commit()

    assert pause_pending_for_model(conn, "model/a", status="model_cost_paused") == 1
    conn.commit()
    rows = conn.execute("SELECT model_id, status, count(*) n FROM trials GROUP BY model_id, status").fetchall()
    got = sorted((dict(row) for row in rows), key=lambda row: (row["model_id"], row["status"]))
    assert got == [
        {"model_id": "model/a", "status": "model_cost_paused", "n": 1},
        {"model_id": "model/a", "status": "ok", "n": 1},
        {"model_id": "model/b", "status": "pending", "n": 1},
    ]
    conn.close()


def test_reset_statuses(tmp_path) -> None:
    db = tmp_path / "test.sqlite3"
    conn = connect(db)
    trial = Trial(
        trial_id="t1", pair_id="p1", condition="bare", repetition=0, temperature=0.7,
        option_1="blue", option_2="red", context_id=None, prompt="Choose one: blue or red.",
    )
    enqueue_trial(conn, "model/a", trial)
    mark_result(conn, "t1", "model/a", status="rate_limited", error_type="OpenRouterRateLimitError")
    conn.commit()

    assert reset_statuses(conn, ["rate_limited"]) == 1
    conn.commit()
    assert counts(conn) == {"pending": 1}
    conn.close()
