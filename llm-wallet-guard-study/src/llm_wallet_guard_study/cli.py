"""Command-line interface for the simulator MVP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .db import initialize_database
from .export import export_results, render_demo_summary, write_analysis
from .runner import prepare_trials, run_pilot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-wallet-guard-study",
        description="Simulator-only LLM wallet guardian safety study tools",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_db = subparsers.add_parser("init-db", help="Create or update the SQLite schema")
    init_db.add_argument("--db", required=True, type=Path, help="SQLite database path")
    init_db.set_defaults(func=_cmd_init_db)

    prepare = subparsers.add_parser("prepare", help="Create model definitions and pending trials")
    prepare.add_argument("--db", required=True, type=Path, help="SQLite database path")
    prepare.add_argument("--config", required=True, type=Path, help="Simulator config path")
    prepare.set_defaults(func=_cmd_prepare)

    run = subparsers.add_parser("run-pilot", help="Run pending simulator-only trials")
    run.add_argument("--db", required=True, type=Path, help="SQLite database path")
    run.add_argument("--config", required=True, type=Path, help="Simulator config path")
    run.add_argument("--max-trials", type=int, default=None, help="Maximum pending trials to run")
    run.add_argument(
        "--attacker-model",
        default=None,
        help="Only claim and run pending trials for this attacker model id",
    )
    run.add_argument(
        "--guardian-model",
        default=None,
        help="Only claim and run pending trials for this guardian model id",
    )
    run.add_argument(
        "--no-recover-stale",
        action="store_true",
        help="Do not mark existing in_progress rows incomplete at startup; use for parallel workers",
    )
    run.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Assume the run was already prepared; use for parallel workers",
    )
    run.add_argument("--seed", type=int, default=None, help="Override random seed for local stubs")
    run.add_argument("--no-delay", action="store_true", help="Disable inter-attempt delay")
    run.add_argument(
        "--allow-live",
        action="store_true",
        help="Permit explicitly enabled live chat-completion adapters to make network calls",
    )
    run.set_defaults(func=_cmd_run_pilot)

    export = subparsers.add_parser("export-results", help="Export CSV/JSON result artifacts")
    export.add_argument("--db", required=True, type=Path, help="SQLite database path")
    export.add_argument("--out", required=True, type=Path, help="Output directory")
    export.set_defaults(func=_cmd_export_results)

    analyze = subparsers.add_parser("analyze", help="Write summary analysis artifacts")
    analyze.add_argument("--db", required=True, type=Path, help="SQLite database path")
    analyze.add_argument("--out", required=True, type=Path, help="Output directory")
    analyze.set_defaults(func=_cmd_analyze)

    demo_summary = subparsers.add_parser(
        "demo-summary",
        help="Print generated summary, key trial outcomes, and simulator transfer dispositions",
    )
    demo_summary.add_argument("--out", required=True, type=Path, help="Export output directory")
    demo_summary.set_defaults(func=_cmd_demo_summary)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


def _print_json(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _cmd_init_db(args) -> None:
    initialize_database(args.db)
    _print_json({"db": str(args.db), "status": "initialized"})


def _cmd_prepare(args) -> None:
    _print_json(prepare_trials(args.db, args.config))


def _cmd_run_pilot(args) -> None:
    _print_json(
        run_pilot(
            args.db,
            args.config,
            max_trials=args.max_trials,
            attacker_model_id=args.attacker_model,
            guardian_model_id=args.guardian_model,
            recover_stale=not args.no_recover_stale,
            prepare=not args.skip_prepare,
            seed=args.seed,
            no_delay=args.no_delay,
            allow_live=args.allow_live,
        )
    )


def _cmd_export_results(args) -> None:
    _print_json(export_results(args.db, args.out))


def _cmd_analyze(args) -> None:
    _print_json(write_analysis(args.db, args.out))


def _cmd_demo_summary(args) -> None:
    print(render_demo_summary(args.out), end="")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
