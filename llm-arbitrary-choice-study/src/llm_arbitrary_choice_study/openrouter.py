from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
import os
import random
import re
from typing import Any

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_CREDITS_URL = "https://openrouter.ai/api/v1/credits"
_BEARER_RE = re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]+")


class OpenRouterBudgetError(RuntimeError):
    """Raised when the provider says the account lacks credits or quota."""


class OpenRouterAuthError(RuntimeError):
    """Raised when the API key is missing, invalid, or disabled."""


class OpenRouterRateLimitError(RuntimeError):
    """Raised after bounded retry attempts on HTTP 429."""


class OpenRouterResponseError(RuntimeError):
    """Raised for non-HTTP provider errors returned in a response body."""


@dataclass(frozen=True)
class OpenRouterCredits:
    total_credits: float
    total_usage: float

    @property
    def remaining_credits(self) -> float:
        return self.total_credits - self.total_usage

    def as_public_dict(self) -> dict[str, object]:
        return {
            "total_credits": self.total_credits,
            "total_usage": self.total_usage,
            "remaining_credits": self.remaining_credits,
        }


def _secret_from_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    joined = " or ".join(names)
    raise RuntimeError(
        f"{joined} is not set. Copy .env.example to .env and fill in the required key."
    )


def _headers_for_key(api_key: str) -> dict[str, str]:
    h = {"Authorization": f"Bearer {api_key}"}
    if app_url := os.environ.get("OPENROUTER_APP_URL"):
        h["HTTP-Referer"] = app_url
    if app_title := os.environ.get("OPENROUTER_APP_TITLE"):
        h["X-Title"] = app_title
    return h


def headers() -> dict[str, str]:
    return _headers_for_key(_secret_from_env("OPENROUTER_API_KEY"))


def management_headers() -> dict[str, str]:
    return _headers_for_key(
        _secret_from_env("OPENROUTER_MANAGEMENT_API_KEY", "OPENROUTER_API_KEY")
    )


def redact_secrets(message: str) -> str:
    redacted = _BEARER_RE.sub("Bearer [redacted]", message)
    for env_var in ("OPENROUTER_API_KEY", "OPENROUTER_MANAGEMENT_API_KEY"):
        if api_key := os.environ.get(env_var):
            redacted = redacted.replace(api_key, "[redacted]")
    return redacted


def _error_summary(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        data = None
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        error = data["error"]
        code = error.get("code", response.status_code)
        message = error.get("message", response.text)
        return redact_secrets(f"{code}: {message}")[:1000]
    return redact_secrets(response.text or response.reason_phrase)[:1000]


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def _rate_limit_delay(
    response: httpx.Response,
    *,
    attempt: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
    rng: random.Random,
) -> float:
    retry_after = _parse_retry_after(response.headers.get("Retry-After"))
    if retry_after is not None:
        return min(retry_after, max_delay_seconds)
    if base_delay_seconds <= 0:
        return 0.0
    upper = min(max_delay_seconds, base_delay_seconds * (2**attempt))
    return rng.uniform(0.0, upper)


def _raise_for_body_error(response_json: dict[str, Any]) -> None:
    error = response_json.get("error")
    if not isinstance(error, dict):
        return
    code = error.get("code")
    message = redact_secrets(str(error.get("message", "OpenRouter response contained an error.")))[:1000]
    if code in (401, 403):
        raise OpenRouterAuthError(message)
    if code == 402:
        raise OpenRouterBudgetError(message)
    if code == 429 or code == "rate_limit_exceeded":
        raise OpenRouterRateLimitError(message)
    raise OpenRouterResponseError(message)


async def complete(
    *,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: float,
    rate_limit_max_retries: int = 4,
    rate_limit_base_delay_seconds: float = 5.0,
    rate_limit_max_delay_seconds: float = 120.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    rng = random.Random()
    for attempt in range(rate_limit_max_retries + 1):
        async with httpx.AsyncClient(timeout=timeout_seconds, transport=transport) as client:
            response = await client.post(OPENROUTER_URL, headers=headers(), json=payload)
        if response.status_code == 401:
            raise OpenRouterAuthError(_error_summary(response))
        if response.status_code == 403:
            raise OpenRouterAuthError(_error_summary(response))
        if response.status_code == 402:
            raise OpenRouterBudgetError(_error_summary(response))
        if response.status_code == 429:
            if attempt >= rate_limit_max_retries:
                raise OpenRouterRateLimitError(_error_summary(response))
            delay = _rate_limit_delay(
                response,
                attempt=attempt,
                base_delay_seconds=rate_limit_base_delay_seconds,
                max_delay_seconds=rate_limit_max_delay_seconds,
                rng=rng,
            )
            await asyncio.sleep(delay)
            continue
        response.raise_for_status()
        response_json = response.json()
        _raise_for_body_error(response_json)
        return payload, response_json
    raise OpenRouterRateLimitError("rate limit retry loop ended unexpectedly")


def _finite_float(data: dict[str, Any], field: str) -> float:
    try:
        value = float(data[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise OpenRouterResponseError(
            f"OpenRouter credits response missing numeric field: {field}"
        ) from exc
    if not math.isfinite(value):
        raise OpenRouterResponseError(
            f"OpenRouter credits response contains non-finite field: {field}"
        )
    return value


async def get_credits(
    *,
    timeout_seconds: float = 30.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> OpenRouterCredits:
    async with httpx.AsyncClient(timeout=timeout_seconds, transport=transport) as client:
        response = await client.get(OPENROUTER_CREDITS_URL, headers=management_headers())
    if response.status_code in (401, 403):
        raise OpenRouterAuthError(_error_summary(response))
    if response.status_code >= 400:
        raise OpenRouterResponseError(_error_summary(response))
    response_json = response.json()
    _raise_for_body_error(response_json)
    data = response_json.get("data")
    if not isinstance(data, dict):
        raise OpenRouterResponseError("OpenRouter credits response missing data object.")
    return OpenRouterCredits(
        total_credits=_finite_float(data, "total_credits"),
        total_usage=_finite_float(data, "total_usage"),
    )


def extract_text(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
    return ""
