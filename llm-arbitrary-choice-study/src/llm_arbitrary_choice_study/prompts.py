from __future__ import annotations

import hashlib
import random

from .schema import ChoicePair, ContextSpec, RunConfig, Trial

BARE_TEMPLATE = 'Choose one: {first} or {second}. Reply with exactly one word: "{first}" or "{second}".'
CONTEXT_TEMPLATE = '{context} Choose one: {first} or {second}. Reply with exactly one word: "{first}" or "{second}".'


def stable_id(*parts: object) -> str:
    h = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return h[:16]


def build_prompt(condition: str, first: str, second: str, context: ContextSpec | None) -> str:
    if condition.startswith("context"):
        if context is None:
            raise ValueError("context condition requires a context")
        return CONTEXT_TEMPLATE.format(context=context.text.strip(), first=first, second=second)
    return BARE_TEMPLATE.format(first=first, second=second)


def generate_trials(
    pairs: list[ChoicePair],
    contexts: list[ContextSpec],
    cfg: RunConfig,
) -> list[Trial]:
    by_pair: dict[str, list[ContextSpec]] = {}
    for context_spec in contexts:
        by_pair.setdefault(context_spec.pair_id, []).append(context_spec)

    rng = random.Random(cfg.seed)
    trials: list[Trial] = []
    for pair in pairs:
        pair_contexts = by_pair.get(pair.id, [])
        for temperature in cfg.temperatures:
            for repetition in range(cfg.repetitions):
                # For context trials, rotate through available contexts deterministically.
                selected_context: ContextSpec | None = rng.choice(pair_contexts) if pair_contexts else None
                for condition in cfg.include_conditions:
                    swapped = condition.endswith("swapped")
                    first, second = (pair.option_b, pair.option_a) if swapped else (pair.option_a, pair.option_b)
                    trial_context = selected_context if condition.startswith("context") else None
                    prompt = build_prompt(condition, first, second, trial_context)
                    trials.append(
                        Trial(
                            trial_id=stable_id(
                                pair.id,
                                condition,
                                repetition,
                                temperature,
                                trial_context.id if trial_context else "none",
                            ),
                            pair_id=pair.id,
                            condition=condition,
                            repetition=repetition,
                            temperature=temperature,
                            option_1=first,
                            option_2=second,
                            context_id=trial_context.id if trial_context else None,
                            prompt=prompt,
                        )
                    )
    rng.shuffle(trials)
    return trials


def parse_choice(text: str, option_1: str, option_2: str) -> tuple[str | None, str]:
    raw = text.strip()
    punctuation = "\"'`.,:;!()[]{}"
    cleaned = raw.lower().strip().strip(punctuation)
    if cleaned == option_1:
        return option_1, "exact"
    if cleaned == option_2:
        return option_2, "exact"

    tokens = [t.strip(punctuation).lower() for t in raw.split()]
    hits = [t for t in tokens if t in {option_1, option_2}]
    if len(hits) == 1:
        return hits[0], "single_token_in_text"
    if len(set(hits)) == 1 and hits:
        return hits[0], "repeated_single_option"
    return None, "invalid"
