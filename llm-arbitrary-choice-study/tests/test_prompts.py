from llm_arbitrary_choice_study.prompts import generate_trials, parse_choice
from llm_arbitrary_choice_study.schema import ChoicePair, ContextSpec, RunConfig


def test_parse_choice_exact_and_invalid() -> None:
    assert parse_choice("blue", "blue", "red") == ("blue", "exact")
    assert parse_choice("Red.", "blue", "red") == ("red", "exact")
    assert parse_choice("I choose blue", "blue", "red") == ("blue", "single_token_in_text")
    assert parse_choice("blue or red", "blue", "red") == (None, "invalid")


def test_generate_trials_includes_swapped_order() -> None:
    pairs = [ChoicePair(id="p", category="c", option_a="blue", option_b="red")]
    contexts = [ContextSpec(id="ctx", pair_id="p", text="The sky looked clear.")]
    cfg = RunConfig(repetitions=1, temperatures=[0.7], include_conditions=["bare", "bare_swapped", "context", "context_swapped"])
    trials = generate_trials(pairs, contexts, cfg)
    by_condition = {t.condition: t for t in trials}
    assert by_condition["bare"].option_1 == "blue"
    assert by_condition["bare_swapped"].option_1 == "red"
    assert by_condition["context"].context_id == "ctx"
    assert by_condition["context_swapped"].option_2 == "blue"
