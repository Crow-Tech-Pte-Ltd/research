from __future__ import annotations

import json

from llm_wallet_guard_study.export import export_results
from llm_wallet_guard_study.runner import run_pilot

from .conftest import write_config


def test_export_includes_manifest_and_row_counts(tmp_path, base_config):
    config = base_config
    config["run_id"] = "export-run"
    config["max_attempts"] = 5
    config["conditions"][0]["max_attempts"] = 5
    db_path = tmp_path / "export.sqlite3"
    config_path = write_config(tmp_path, config, "export-run")
    run_pilot(db_path, config_path, no_delay=True)
    out_dir = tmp_path / "out"
    manifest = export_results(db_path, out_dir)
    manifest_file = out_dir / "manifest.json"
    assert manifest_file.exists()
    loaded = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert loaded["row_counts"]["trials"] == 3
    assert loaded["row_counts"]["attempts"] == 10
    assert "summary.json" in loaded["file_hashes"]
    assert "trial_transcripts.md" in loaded["file_hashes"]
    assert (out_dir / "runs.csv").exists()
    assert (out_dir / "prompt_versions.csv").exists()
    assert (out_dir / "trial_conditions.csv").exists()
    assert (out_dir / "summary.md").exists()
    assert (out_dir / "trial_transcripts.md").exists()
    assert manifest["row_counts"] == loaded["row_counts"]


def test_export_writes_readable_trial_transcripts(tmp_path, base_config):
    config = base_config
    config["run_id"] = "transcript-run"
    config["max_attempts"] = 5
    config["conditions"][0]["max_attempts"] = 5
    db_path = tmp_path / "transcript.sqlite3"
    config_path = write_config(tmp_path, config, "transcript-run")
    run_pilot(db_path, config_path, no_delay=True)

    out_dir = tmp_path / "out"
    export_results(db_path, out_dir)
    transcript = (out_dir / "trial_transcripts.md").read_text(encoding="utf-8")

    assert "# Trial Transcripts" in transcript
    assert "Simulator-only transcript export" in transcript
    assert "### Attempt 2" in transcript
    assert "Unsafe stated intent: yes" in transcript
    assert "Attempted transfer tool call: no" in transcript
    assert "Policy disposition: `block` / `guardian_transfer_forbidden`" in transcript
    assert "Transfer disposition: `blocked` / `guardian_transfer_forbidden`" in transcript
