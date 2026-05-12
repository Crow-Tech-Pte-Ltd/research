from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Origin = Literal["US", "China", "EU", "Canada", "Israel", "Other", "Mixed/unclear"]
CONDITIONS = frozenset({"bare", "bare_swapped", "context", "context_swapped"})


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelSpec(StrictBaseModel):
    id: str
    label: str
    provider: str
    origin: Origin
    tier: Literal["flagship", "mid", "small", "reasoning", "open"] = "mid"
    default_active: bool = False
    notes: str = ""
    prompt_price_per_token: float | None = None
    completion_price_per_token: float | None = None


class ChoicePair(StrictBaseModel):
    id: str
    category: str
    option_a: str
    option_b: str
    notes: str = ""

    @field_validator("option_a", "option_b")
    @classmethod
    def one_word_lower(cls, value: str) -> str:
        value = value.strip().lower()
        if not value or any(ch.isspace() for ch in value):
            raise ValueError("options should be single lowercase tokens for strict parsing")
        return value


class ContextSpec(StrictBaseModel):
    id: str
    pair_id: str
    text: str
    intended_association: str = ""
    strength: Literal["weak", "medium"] = "weak"

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("context text must not be empty")
        return value


class RunConfig(StrictBaseModel):
    models: list[str] = Field(default_factory=list)
    repetitions: int = Field(default=10, ge=1)
    temperatures: list[float] = Field(default_factory=lambda: [0.7])
    max_tokens: int = Field(default=8, ge=1)
    invalid_retry_max_tokens: list[int] = Field(default_factory=list)
    model_invalid_retry_max_tokens: dict[str, list[int]] = Field(default_factory=dict)
    min_delay_seconds: float = Field(default=1.5, ge=0.0)
    max_delay_seconds: float = Field(default=6.0, ge=0.0)
    request_timeout_seconds: float = Field(default=60.0, gt=0.0)
    max_estimated_usd: float | None = Field(default=None, gt=0.0)
    max_single_attempt_cost_usd: float | None = Field(default=0.05, gt=0.0)
    single_attempt_cost_guard_first_n: int = Field(default=100, ge=1)
    cheap_model_completion_price_per_million_threshold: float = Field(default=0.25, ge=0.0)
    cheap_model_invalid_retry_max_tokens: int | None = Field(default=None, ge=1)
    invalid_retry_estimated_fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    seed: int = 20260505
    include_conditions: list[str] = Field(default_factory=lambda: [
        "bare",
        "bare_swapped",
        "context",
        "context_swapped",
    ])
    rate_limit_max_retries: int = Field(default=4, ge=0)
    rate_limit_base_delay_seconds: float = Field(default=5.0, ge=0.0)
    rate_limit_max_delay_seconds: float = Field(default=120.0, gt=0.0)

    @field_validator("invalid_retry_max_tokens")
    @classmethod
    def invalid_retry_max_tokens_valid(cls, value: list[int]) -> list[int]:
        if any(tokens < 1 for tokens in value):
            raise ValueError("invalid_retry_max_tokens must contain positive integers")
        return value

    @field_validator("model_invalid_retry_max_tokens")
    @classmethod
    def model_invalid_retry_max_tokens_valid(cls, value: dict[str, list[int]]) -> dict[str, list[int]]:
        for model_id, limits in value.items():
            if not model_id.strip():
                raise ValueError("model_invalid_retry_max_tokens keys must be non-empty model IDs")
            if any(tokens < 1 for tokens in limits):
                raise ValueError("model_invalid_retry_max_tokens values must contain positive integers")
        return value

    @field_validator("temperatures")
    @classmethod
    def temperatures_valid(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("at least one temperature is required")
        if any(not math.isfinite(t) or t < 0 for t in value):
            raise ValueError("temperatures must be finite, non-negative values")
        return value

    @field_validator("include_conditions")
    @classmethod
    def conditions_known(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("at least one condition is required")
        unknown = sorted(set(value) - CONDITIONS)
        if unknown:
            raise ValueError(f"unknown conditions: {unknown}")
        return value

    @model_validator(mode="after")
    def delays_valid(self) -> RunConfig:
        if self.max_delay_seconds < self.min_delay_seconds:
            raise ValueError("max_delay_seconds must be greater than or equal to min_delay_seconds")
        return self


@dataclass(frozen=True)
class Trial:
    trial_id: str
    pair_id: str
    condition: str
    repetition: int
    temperature: float
    option_1: str
    option_2: str
    context_id: str | None
    prompt: str
