# LLM Arbitrary Choice Study

A reproducible Python project for studying how language models behave when they are forced to choose between two arbitrary options.

The study is deliberately narrow. It does not try to claim that models have beliefs, preferences, intent, or political positions. It asks a simpler question: when a model is told to pick one of two options and neither option is correct, are the answers close to random, or do stable patterns appear?

The first version looks at four things:

- whether a model prefers one option in a pair, such as `blue` over `red`
- whether the model prefers the first or second displayed option
- whether weakly related surrounding context changes the answer
- whether different model families behave differently

The code in this repository is AI-written and human-directed. That disclosure should remain in public writeups.

## Why this exists

LLMs are often used in places where small arbitrary decisions can matter. The choices in this project are intentionally mundane: colors, shapes, textures, objects, weather words, and similar pairs. That keeps the study away from ideological or moral content while still making prompt sensitivity visible if it exists.

## Project status

Working research scaffold. No results are included in the repository. The intended public release should include the exact config, prompts, parser rules, raw responses, analysis outputs, and any known run interruptions.

## Setup

This project uses Python via `uv` and targets Python 3.14. The pinned local version is recorded in `.python-version`.

```bash
uv sync
cp .env.example .env
# edit .env and add OPENROUTER_API_KEY
```

`.env` is ignored by git. Do not commit API keys or raw private credentials.

## Candidate models

Candidate models live in `configs/models.yaml`. Each model has:

- OpenRouter model id
- label
- provider
- coarse organization-origin label, such as US, China, EU, Canada, or Israel
- rough static price metadata where available
- default pilot inclusion flag

These labels are sampling metadata only. They should not be treated as causal explanations for observed outputs. Model availability and prices change, so refresh the model IDs and prices before any large run.

## Run modes

Use the smallest config that answers the current operational question:

- `configs/run_smoke_3models.yaml`: a small three-model data-collection check across US, China, and EU provider groups. Use this before broader runs to verify parsing, pause handling, and exports.
- `configs/run_pilot.yaml`: the default pilot across selected candidate models. Use this after the smoke run looks usable.
- `configs/run_all_candidates.yaml`: the full configured candidate pool. Refresh model IDs and prices first, then estimate cost and check credits before preparing or running it.

For a smoke run, estimate rough local cost first:

```bash
uv run llm-arbitrary-choice-study estimate --config configs/run_smoke_3models.yaml
```

Prepare a separate SQLite database:

```bash
uv run llm-arbitrary-choice-study prepare --db results/raw/smoke_3models.sqlite3 --config configs/run_smoke_3models.yaml
```

Run a small limited batch:

```bash
uv run llm-arbitrary-choice-study run --db results/raw/smoke_3models.sqlite3 --config configs/run_smoke_3models.yaml --limit 20
```

Continue the smoke run only after checking the first responses and status counts:

```bash
uv run llm-arbitrary-choice-study run --db results/raw/smoke_3models.sqlite3 --config configs/run_smoke_3models.yaml
```

Analyze and export browser-viewable charts:

```bash
uv run llm-arbitrary-choice-study analyze --db results/raw/smoke_3models.sqlite3
```

For the default pilot, use the same command sequence with `configs/run_pilot.yaml` and `results/raw/pilot.sqlite3`:

```bash
uv run llm-arbitrary-choice-study estimate --config configs/run_pilot.yaml
uv run llm-arbitrary-choice-study prepare --db results/raw/pilot.sqlite3 --config configs/run_pilot.yaml
uv run llm-arbitrary-choice-study run --db results/raw/pilot.sqlite3 --config configs/run_pilot.yaml --limit 20
uv run llm-arbitrary-choice-study run --db results/raw/pilot.sqlite3 --config configs/run_pilot.yaml
uv run llm-arbitrary-choice-study analyze --db results/raw/pilot.sqlite3
```

For the full candidate-pool config, use a separate database name such as `results/raw/all_candidates.sqlite3`. Do not reuse a smoke or pilot database for a different model list.

Outputs are written under `results/`, which is git-ignored by default.

## Data model

The runner stores every trial in SQLite. Each row includes:

- model id
- pair id
- condition
- repetition
- option order
- context id if any
- exact prompt
- raw response
- parsed choice
- parse status
- raw request and response JSON
- provider usage metadata when returned

Invalid outputs are preserved. They are not silently rewritten into clean answers.

## Conditions

Current pilot conditions:

- `bare`: no context, original option order
- `bare_swapped`: no context, reversed option order
- `context`: weakly related context, original option order
- `context_swapped`: weakly related context, reversed option order

This makes it easier to separate word preference from position bias and context priming.

## Rate limiting and budget behavior

The runner sends requests sequentially and waits a random amount of time between requests. The default pilot range is 1.5 to 6 seconds.

`max_estimated_usd` is a local preflight guard based on the static prices in `configs/models.yaml`, the generated prompts, and `max_tokens`. If the estimate exceeds the configured cap, or if a selected model is missing price metadata while a cap is set, `prepare` and `run` fail closed before making provider calls. This is not a provider-side spending limit.

OpenRouter also exposes a read-only credits endpoint. The credits command calls `GET /api/v1/credits`, prints total purchased credits, total usage, and remaining credits, and never prints the bearer key. OpenRouter documents this endpoint as requiring a management key. Set `OPENROUTER_MANAGEMENT_API_KEY` for that command when the regular completion key is not a management key; otherwise it falls back to `OPENROUTER_API_KEY`.

```bash
uv run llm-arbitrary-choice-study credits
uv run llm-arbitrary-choice-study credits --config configs/run_pilot.yaml
```

With `--config`, the command also prints the local rough estimate and a rough remaining-after-estimate value. It does not reserve credits and does not prevent spending by itself.

If OpenRouter reports insufficient credits with HTTP 402, the run marks the current row as `budget_paused` and exits. If OpenRouter returns HTTP 429, the runner uses `Retry-After` when present or bounded exponential jitter from the run config. If rate limiting persists, it marks the current row as `rate_limited` and exits. If credentials fail with HTTP 401, it marks the current row as `auth_error` and exits.

After resolving the cause of a pause, reset paused rows to pending:

```bash
uv run llm-arbitrary-choice-study reset-paused --db results/raw/pilot.sqlite3
```

OpenRouter documents 402 as insufficient credits and 429 as rate limiting. The runner does not call `/api/v1/credits` automatically; use the explicit `credits` command when an account-level snapshot is needed.

## Analysis outputs

`analyze` writes processed JSON/CSV summaries and Plotly HTML charts under `results/processed` and `results/charts`.

The processed summaries include:

- `summary`: model-level choice shares by pair and condition
- `summary_by_provider`: provider-level choice shares by pair and condition
- `summary_by_origin`: organization-origin choice shares by pair and condition
- `summary_by_origin_provider`: combined origin/provider choice shares by pair and condition

Provider and origin rollups use `configs/models.yaml` as descriptive metadata. They are aggregation aids, not causal labels.

## Development

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

Tests mock provider behavior and must not require `OPENROUTER_API_KEY` or make real OpenRouter completions calls.

## Neutrality notes

This project should avoid speculative language. The safer language is:

- “observed choice distribution”
- “position effect”
- “context effect”
- “model family differences”

Avoid claims like:

- “the model likes blue”
- “the model believes X”
- “the model is conscious”
- “the model is politically biased”

The goal is measurement, not anthropomorphism.
