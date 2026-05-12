from __future__ import annotations

import re

from llm_wallet_guard_study.db import connect
from llm_wallet_guard_study.export import export_results
from llm_wallet_guard_study.redaction import REDACTION_MARKER, redact_json_dumps, redact_text
from llm_wallet_guard_study.runner import _insert_invocation, prepare_trials

from .conftest import write_config


FAKE_HEX_SECRET = "a" * 64
FAKE_TOKEN = "sk-" + ("A" * 24)
FAKE_RECOVERY_TEXT = (
    "mnemonic: alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima"
)


def test_redaction_catches_fake_secret_like_values():
    text = f"hex {FAKE_HEX_SECRET} token {FAKE_TOKEN} {FAKE_RECOVERY_TEXT}"
    redacted = redact_text(text)
    assert FAKE_HEX_SECRET not in redacted
    assert FAKE_TOKEN not in redacted
    assert "alpha bravo" not in redacted
    assert REDACTION_MARKER in redacted
    assert REDACTION_MARKER in redact_json_dumps({"nested": text})


def test_redaction_before_persistence_and_export(tmp_path, base_config):
    config_path = write_config(tmp_path, base_config, "redaction-run")
    db_path = tmp_path / "redaction.sqlite3"
    prepare_trials(db_path, config_path)
    with connect(db_path) as conn:
        trial = conn.execute("SELECT * FROM trials ORDER BY id LIMIT 1").fetchone()
        _insert_invocation(
            conn,
            trial["id"],
            1,
            "guardian",
            trial["guardian_model_id"],
            f"input {FAKE_TOKEN}",
            f"output {FAKE_HEX_SECRET}",
            {"reason": FAKE_RECOVERY_TEXT},
            0.0,
        )
        conn.commit()
        row = conn.execute("SELECT * FROM model_invocations").fetchone()
        assert FAKE_TOKEN not in row["input_text"]
        assert FAKE_HEX_SECRET not in row["output_text"]
        assert "alpha bravo" not in row["action_json"]
    out_dir = tmp_path / "redacted_export"
    export_results(db_path, out_dir)
    exported = (out_dir / "model_invocations.csv").read_text(encoding="utf-8")
    assert FAKE_TOKEN not in exported
    assert FAKE_HEX_SECRET not in exported
    assert "alpha bravo" not in exported


def test_public_docs_and_config_do_not_contain_secret_like_values():
    paths = [
        *list(__import__("pathlib").Path("configs").rglob("*")),
        *list(__import__("pathlib").Path("docs").rglob("*")),
        *list(__import__("pathlib").Path("prompts").rglob("*")),
        __import__("pathlib").Path("README.md"),
    ]
    hex_secret = re.compile(r"\b(?:0x)?[A-Fa-f0-9]{64}\b")
    token = re.compile(r"\b(?:sk|pk|rk|ghp)[-_][A-Za-z0-9_=-]{20,}\b")
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        assert not hex_secret.search(text), path
        assert not token.search(text), path
