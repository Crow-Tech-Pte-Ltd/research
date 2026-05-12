from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from .analysis import export_summary, make_charts
from .config import load_local_env
from .openrouter import get_credits
from .runner import (
    PAUSED_STATUSES,
    estimate_cost,
    estimated_cost_warning,
    prepare_database,
    reset_paused_trials,
    run_pending,
)

app = typer.Typer(help="Run and analyze arbitrary binary-choice trials across LLMs.")
console = Console()


@app.command()
def estimate(config: Path = Path("configs/run_pilot.yaml")) -> None:
    console.print(estimated_cost_warning(config))


@app.command()
def credits(
    config: Path | None = typer.Option(
        None,
        help="Optional run config to compare remaining OpenRouter credits with the local rough estimate.",
    ),
    timeout_seconds: float = typer.Option(30.0, help="Timeout for the read-only OpenRouter credits request."),
) -> None:
    load_local_env()
    snapshot = asyncio.run(get_credits(timeout_seconds=timeout_seconds))
    report: dict[str, object] = snapshot.as_public_dict()
    if config is not None:
        estimate = estimate_cost(config)
        report.update(
            {
                "estimate_config": str(config),
                "rough_estimated_cost_usd": estimate.rough_estimated_cost_usd,
                "remaining_after_rough_estimate_credits": (
                    snapshot.remaining_credits - estimate.rough_estimated_cost_usd
                ),
                "max_estimated_usd": estimate.max_estimated_usd,
                "missing_prices": estimate.missing_prices,
            }
        )
    console.print(report)


@app.command()
def prepare(db: Path = Path("results/raw/pilot.sqlite3"), config: Path = Path("configs/run_pilot.yaml")) -> None:
    console.print(estimated_cost_warning(config))
    prepare_database(db, config)


@app.command()
def run(
    db: Path = Path("results/raw/pilot.sqlite3"),
    config: Path = Path("configs/run_pilot.yaml"),
    limit: int | None = typer.Option(None, help="Maximum number of pending trials to execute in this process."),
) -> None:
    asyncio.run(run_pending(db, config, limit=limit))


@app.command("reset-paused")
def reset_paused(
    db: Path = Path("results/raw/pilot.sqlite3"),
    status: str = typer.Option("all", help=f"Paused status to reset: all, {', '.join(PAUSED_STATUSES)}."),
) -> None:
    changed = reset_paused_trials(db, status=status)
    console.print({"reset_to_pending": changed, "status": status})


@app.command()
def analyze(db: Path = Path("results/raw/pilot.sqlite3")) -> None:
    export_summary(db)
    make_charts(db)
    console.print("Exported processed data and charts.")
