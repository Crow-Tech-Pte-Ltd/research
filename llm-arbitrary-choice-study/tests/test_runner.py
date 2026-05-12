import pytest

from llm_arbitrary_choice_study.runner import (
    estimate_cost_for_components,
    enforce_estimated_budget,
    is_super_cheap_model,
    token_limits_for_model,
)
from llm_arbitrary_choice_study.schema import ChoicePair, ContextSpec, ModelSpec, RunConfig


def test_estimated_budget_fails_closed_when_price_missing() -> None:
    cfg = RunConfig(models=["provider/model"], max_estimated_usd=1.0)
    estimate = estimate_cost_for_components(
        cfg,
        {
            "provider/model": ModelSpec(
                id="provider/model",
                label="Model",
                provider="Provider",
                origin="Other",
                default_active=True,
            )
        },
        [ChoicePair(id="p", category="color", option_a="blue", option_b="red")],
        [ContextSpec(id="ctx", pair_id="p", text="The day felt open and clear.")],
    )

    with pytest.raises(ValueError, match="missing price metadata"):
        enforce_estimated_budget(estimate)


def test_estimated_budget_fails_when_cap_is_exceeded() -> None:
    cfg = RunConfig(models=["provider/model"], max_estimated_usd=0.000001)
    estimate = estimate_cost_for_components(
        cfg,
        {
            "provider/model": ModelSpec(
                id="provider/model",
                label="Model",
                provider="Provider",
                origin="Other",
                default_active=True,
                prompt_price_per_token=1.0,
                completion_price_per_token=1.0,
            )
        },
        [ChoicePair(id="p", category="color", option_a="blue", option_b="red")],
        [ContextSpec(id="ctx", pair_id="p", text="The day felt open and clear.")],
    )

    with pytest.raises(ValueError, match="exceeds max_estimated_usd"):
        enforce_estimated_budget(estimate)


def test_super_cheap_model_gets_large_invalid_retry_limit() -> None:
    cfg = RunConfig(
        max_tokens=512,
        cheap_model_completion_price_per_million_threshold=0.25,
        cheap_model_invalid_retry_max_tokens=3000,
    )
    cheap = ModelSpec(
        id="provider/cheap",
        label="Cheap",
        provider="Provider",
        origin="Other",
        completion_price_per_token=0.2 / 1_000_000,
    )
    pricey = ModelSpec(
        id="provider/pricey",
        label="Pricey",
        provider="Provider",
        origin="Other",
        completion_price_per_token=1.0 / 1_000_000,
    )

    assert is_super_cheap_model(cfg, cheap)
    assert not is_super_cheap_model(cfg, pricey)
    assert token_limits_for_model(cfg, cheap) == [512, 3000]
    assert token_limits_for_model(cfg, pricey) == [512]


def test_model_specific_invalid_retry_limits_are_appended_and_deduped() -> None:
    cfg = RunConfig(
        max_tokens=512,
        invalid_retry_max_tokens=[1024],
        model_invalid_retry_max_tokens={"provider/thinking": [1024, 2048]},
    )
    model = ModelSpec(
        id="provider/thinking",
        label="Thinking",
        provider="Provider",
        origin="Other",
        completion_price_per_token=1.0 / 1_000_000,
    )

    assert token_limits_for_model(cfg, model) == [512, 1024, 2048]
