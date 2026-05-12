from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math
import random
from pathlib import Path

from rich.console import Console

from .config import load_choice_pairs, load_contexts, load_local_env, load_models, load_run_config
from .db import (
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
from .openrouter import OpenRouterAuthError, OpenRouterBudgetError, OpenRouterRateLimitError, complete, extract_text
from .prompts import generate_trials, parse_choice
from .schema import ChoicePair, ContextSpec, ModelSpec, RunConfig

console = Console()
PAUSED_STATUSES = ("budget_paused", "rate_limited", "auth_error", "model_cost_paused")


class ModelCostSpikeError(RuntimeError):
    """Raised when early live usage shows that a model is too costly to continue safely."""


@dataclass(frozen=True)
class CostEstimate:
    rough_estimated_cost_usd: float
    trials_per_model: int
    model_count: int
    estimated_prompt_tokens_per_model: int
    estimated_completion_tokens_per_model: int
    missing_prices: list[str]
    max_estimated_usd: float | None


def selected_model_ids(cfg: RunConfig, models: dict[str, ModelSpec]) -> list[str]:
    selected = cfg.models or [m.id for m in models.values() if m.default_active]
    unknown = sorted(set(selected) - set(models))
    if unknown:
        raise ValueError(f"Unknown models in run config: {unknown}")
    return selected


def rough_token_count(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def is_super_cheap_model(cfg: RunConfig, model: ModelSpec) -> bool:
    if cfg.cheap_model_invalid_retry_max_tokens is None:
        return False
    if model.completion_price_per_token is None:
        return False
    price_per_million = model.completion_price_per_token * 1_000_000
    return price_per_million <= cfg.cheap_model_completion_price_per_million_threshold


def token_limits_for_model(cfg: RunConfig, model: ModelSpec) -> list[int]:
    token_limits = [
        cfg.max_tokens,
        *cfg.invalid_retry_max_tokens,
        *cfg.model_invalid_retry_max_tokens.get(model.id, []),
    ]
    if is_super_cheap_model(cfg, model):
        cheap_limit = cfg.cheap_model_invalid_retry_max_tokens
        if cheap_limit is not None:
            token_limits.append(cheap_limit)
    # Preserve configured order while removing duplicates, so retries grow only when needed.
    return list(dict.fromkeys(token_limits))


def estimate_cost_for_components(
    cfg: RunConfig,
    models: dict[str, ModelSpec],
    pairs: list[ChoicePair],
    contexts: list[ContextSpec],
) -> CostEstimate:
    selected = selected_model_ids(cfg, models)
    trials = generate_trials(pairs, contexts, cfg)
    estimated_prompt_tokens = sum(rough_token_count(trial.prompt) for trial in trials)
    max_completion_budget = 0
    total = 0.0
    missing: list[str] = []
    for model_id in selected:
        m = models[model_id]
        if m.prompt_price_per_token is None or m.completion_price_per_token is None:
            missing.append(model_id)
            continue
        token_limits = token_limits_for_model(cfg, m)
        first_attempt_budget = token_limits[0]
        retry_budget = sum(token_limits[1:]) * cfg.invalid_retry_estimated_fraction
        completion_budget = len(trials) * (first_attempt_budget + retry_budget)
        max_completion_budget = max(max_completion_budget, int(math.ceil(completion_budget)))
        total += (
            estimated_prompt_tokens * m.prompt_price_per_token
            + completion_budget * m.completion_price_per_token
        )
    return CostEstimate(
        rough_estimated_cost_usd=total,
        trials_per_model=len(trials),
        model_count=len(selected),
        estimated_prompt_tokens_per_model=estimated_prompt_tokens,
        estimated_completion_tokens_per_model=max_completion_budget,
        missing_prices=missing,
        max_estimated_usd=cfg.max_estimated_usd,
    )


def estimate_cost(run_config_path: str | Path) -> CostEstimate:
    cfg = load_run_config(run_config_path)
    return estimate_cost_for_components(
        cfg,
        {m.id: m for m in load_models()},
        load_choice_pairs(),
        load_contexts(),
    )


def format_cost_estimate(estimate: CostEstimate) -> str:
    return (
        f"rough_estimated_cost_usd={estimate.rough_estimated_cost_usd:.4f}; "
        f"max_estimated_usd={estimate.max_estimated_usd}; "
        f"trials_per_model={estimate.trials_per_model}; "
        f"models={estimate.model_count}; "
        f"estimated_prompt_tokens_per_model={estimate.estimated_prompt_tokens_per_model}; "
        f"estimated_completion_tokens_per_model={estimate.estimated_completion_tokens_per_model}; "
        f"missing_prices={estimate.missing_prices}"
    )


def enforce_estimated_budget(estimate: CostEstimate) -> None:
    if estimate.max_estimated_usd is None:
        return
    if estimate.missing_prices:
        raise ValueError(
            "max_estimated_usd is set, but some selected models are missing price metadata: "
            f"{estimate.missing_prices}"
        )
    if estimate.rough_estimated_cost_usd > estimate.max_estimated_usd:
        raise ValueError(
            "Rough estimated cost exceeds max_estimated_usd: "
            f"{estimate.rough_estimated_cost_usd:.4f} > {estimate.max_estimated_usd:.4f}"
        )


def prepare_database(db_path: str | Path, run_config_path: str | Path) -> None:
    estimate = estimate_cost(run_config_path)
    enforce_estimated_budget(estimate)
    cfg = load_run_config(run_config_path)
    models = {m.id: m for m in load_models()}
    selected = selected_model_ids(cfg, models)

    trials = generate_trials(load_choice_pairs(), load_contexts(), cfg)
    conn = connect(db_path)
    try:
        for model_id in selected:
            for trial in trials:
                enqueue_trial(conn, model_id, trial)
        conn.commit()
        console.print({"queued": len(trials) * len(selected), "counts": counts(conn)})
    finally:
        conn.close()


def should_retry_invalid(text: str, response_json: dict, parse_status: str) -> bool:
    if text.strip():
        return False
    choices = response_json.get("choices") or []
    if not choices:
        return True
    finish_reason = choices[0].get("finish_reason")
    native_finish_reason = choices[0].get("native_finish_reason")
    return finish_reason == "length" or native_finish_reason in {"length", "max_output_tokens"}


def usage_cost(response_json: dict) -> float | None:
    usage = response_json.get("usage") or {}
    cost = usage.get("cost")
    if cost is None:
        return None
    try:
        return float(cost)
    except (TypeError, ValueError):
        return None


def check_attempt_cost_guard(conn, cfg: RunConfig, row, response_json: dict, max_tokens: int) -> None:
    cost = usage_cost(response_json)
    if cfg.max_single_attempt_cost_usd is None or cost is None:
        return
    attempts_seen = model_attempt_count(conn, row["model_id"])
    if attempts_seen > cfg.single_attempt_cost_guard_first_n:
        return
    if cost > cfg.max_single_attempt_cost_usd:
        raise ModelCostSpikeError(
            f"{row['model_id']} single attempt cost ${cost:.6f} exceeded "
            f"${cfg.max_single_attempt_cost_usd:.6f} within first "
            f"{cfg.single_attempt_cost_guard_first_n} attempts; max_tokens={max_tokens}"
        )


async def complete_with_invalid_retries(
    conn,
    row,
    cfg: RunConfig,
    model: ModelSpec,
) -> tuple[dict, dict, str, str | None, str]:
    token_limits = token_limits_for_model(cfg, model)
    last_request_json: dict | None = None
    last_response_json: dict | None = None
    last_text = ""
    last_choice: str | None = None
    last_parse_status = "invalid"
    for attempt_index, max_tokens in enumerate(token_limits):
        request_json, response_json = await complete(
            model=row["model_id"],
            prompt=row["prompt"],
            temperature=row["temperature"],
            max_tokens=max_tokens,
            timeout_seconds=cfg.request_timeout_seconds,
            rate_limit_max_retries=cfg.rate_limit_max_retries,
            rate_limit_base_delay_seconds=cfg.rate_limit_base_delay_seconds,
            rate_limit_max_delay_seconds=cfg.rate_limit_max_delay_seconds,
        )
        text = extract_text(response_json)
        choice, parse_status = parse_choice(text, row["option_1"], row["option_2"])
        status = "ok" if choice is not None else "invalid"
        record_attempt(
            conn,
            row["trial_id"],
            row["model_id"],
            attempt_index=attempt_index,
            max_tokens=max_tokens,
            status=status,
            response_text=text,
            parsed_choice=choice,
            parse_status=parse_status,
            request_json=request_json,
            response_json=response_json,
            usage_json=response_json.get("usage"),
        )
        conn.commit()
        check_attempt_cost_guard(conn, cfg, row, response_json, max_tokens)
        last_request_json = request_json
        last_response_json = response_json
        last_text = text
        last_choice = choice
        last_parse_status = parse_status
        if choice is not None:
            return request_json, response_json, text, choice, parse_status
        if attempt_index == len(token_limits) - 1:
            break
        if not should_retry_invalid(text, response_json, parse_status):
            break
    assert last_request_json is not None and last_response_json is not None
    return last_request_json, last_response_json, last_text, last_choice, last_parse_status


def estimated_cost_warning(run_config_path: str | Path) -> str:
    return format_cost_estimate(estimate_cost(run_config_path))


def reset_paused_trials(db_path: str | Path, status: str = "all") -> int:
    if status == "all":
        statuses = list(PAUSED_STATUSES)
    elif status in PAUSED_STATUSES:
        statuses = [status]
    else:
        raise ValueError(f"status must be one of {(*PAUSED_STATUSES, 'all')}")

    conn = connect(db_path)
    try:
        changed = reset_statuses(conn, statuses)
        conn.commit()
        return changed
    finally:
        conn.close()


async def run_pending(db_path: str | Path, run_config_path: str | Path, limit: int | None = None) -> None:
    enforce_estimated_budget(estimate_cost(run_config_path))
    load_local_env()
    cfg = load_run_config(run_config_path)
    models = {m.id: m for m in load_models()}
    conn = connect(db_path)
    rng = random.Random()
    completed = 0
    try:
        while True:
            if limit is not None and completed >= limit:
                break
            row = next_pending(conn)
            if row is None:
                console.print("No pending trials.")
                break
            delay = rng.uniform(cfg.min_delay_seconds, cfg.max_delay_seconds)
            await asyncio.sleep(delay)
            try:
                model = models[row["model_id"]]
                request_json, response_json, text, choice, parse_status = await complete_with_invalid_retries(
                    conn, row, cfg, model
                )
                status = "ok" if choice is not None else "invalid"
                mark_result(
                    conn, row["trial_id"], row["model_id"],
                    status=status, response_text=text, parsed_choice=choice, parse_status=parse_status,
                    request_json=request_json, response_json=response_json, usage_json=response_json.get("usage"),
                )
            except ModelCostSpikeError as exc:
                paused = pause_pending_for_model(conn, row["model_id"], status="model_cost_paused")
                mark_result(
                    conn, row["trial_id"], row["model_id"],
                    status="model_cost_paused", error_type="ModelCostSpikeError", error_message=str(exc)[:1000],
                )
                conn.commit()
                console.print({"model_cost_paused": row["model_id"], "pending_paused": paused, "reason": str(exc)})
            except OpenRouterBudgetError as exc:
                record_attempt(
                    conn,
                    row["trial_id"],
                    row["model_id"],
                    attempt_index=0,
                    max_tokens=cfg.max_tokens,
                    status="budget_paused",
                    error_type="OpenRouterBudgetError",
                    error_message=str(exc)[:1000],
                )
                mark_result(
                    conn, row["trial_id"], row["model_id"],
                    status="budget_paused", error_type="OpenRouterBudgetError", error_message=str(exc)[:1000],
                )
                conn.commit()
                raise SystemExit("OpenRouter reports insufficient credits/quota. Pausing so more budget can be added.") from exc
            except OpenRouterAuthError as exc:
                record_attempt(
                    conn,
                    row["trial_id"],
                    row["model_id"],
                    attempt_index=0,
                    max_tokens=cfg.max_tokens,
                    status="auth_error",
                    error_type="OpenRouterAuthError",
                    error_message=str(exc)[:1000],
                )
                mark_result(
                    conn, row["trial_id"], row["model_id"],
                    status="auth_error", error_type="OpenRouterAuthError", error_message=str(exc)[:1000],
                )
                conn.commit()
                raise SystemExit("OpenRouter authentication failed. Check OPENROUTER_API_KEY before retrying.") from exc
            except OpenRouterRateLimitError as exc:
                record_attempt(
                    conn,
                    row["trial_id"],
                    row["model_id"],
                    attempt_index=0,
                    max_tokens=cfg.max_tokens,
                    status="rate_limited",
                    error_type="OpenRouterRateLimitError",
                    error_message=str(exc)[:1000],
                )
                mark_result(
                    conn, row["trial_id"], row["model_id"],
                    status="rate_limited", error_type="OpenRouterRateLimitError", error_message=str(exc)[:1000],
                )
                conn.commit()
                raise SystemExit("OpenRouter rate limiting persisted after bounded retries. Pausing the run.") from exc
            except Exception as exc:  # noqa: BLE001 - preserve all provider failures for analysis
                record_attempt(
                    conn,
                    row["trial_id"],
                    row["model_id"],
                    attempt_index=0,
                    max_tokens=cfg.max_tokens,
                    status="error",
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:1000],
                )
                mark_result(
                    conn, row["trial_id"], row["model_id"],
                    status="error", error_type=type(exc).__name__, error_message=str(exc)[:1000],
                )
            conn.commit()
            completed += 1
            if completed % 10 == 0:
                console.print({"completed_this_run": completed, "counts": counts(conn)})
    finally:
        console.print({"final_counts": counts(conn)})
        conn.close()
