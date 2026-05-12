import pandas as pd

from llm_arbitrary_choice_study.analysis import summarize_by, with_model_metadata
from llm_arbitrary_choice_study.schema import ModelSpec


def test_summarize_by_provider_origin_joins_model_metadata() -> None:
    df = pd.DataFrame(
        [
            {
                "model_id": "provider-a/model",
                "pair_id": "p1",
                "condition": "bare",
                "parsed_choice": "blue",
                "status": "ok",
            },
            {
                "model_id": "provider-a/model",
                "pair_id": "p1",
                "condition": "bare",
                "parsed_choice": "red",
                "status": "ok",
            },
            {
                "model_id": "provider-b/model",
                "pair_id": "p1",
                "condition": "bare",
                "parsed_choice": "red",
                "status": "ok",
            },
        ]
    )
    models = [
        ModelSpec(
            id="provider-a/model",
            label="Provider A Model",
            provider="Provider A",
            origin="US",
        ),
        ModelSpec(
            id="provider-b/model",
            label="Provider B Model",
            provider="Provider B",
            origin="EU",
        ),
    ]

    summary = summarize_by(with_model_metadata(df, models), ["origin", "provider"])
    us_blue = summary[
        (summary["origin"] == "US")
        & (summary["provider"] == "Provider A")
        & (summary["parsed_choice"] == "blue")
    ].iloc[0]
    eu_red = summary[
        (summary["origin"] == "EU")
        & (summary["provider"] == "Provider B")
        & (summary["parsed_choice"] == "red")
    ].iloc[0]

    assert us_blue["n"] == 1
    assert us_blue["share"] == 0.5
    assert eu_red["n"] == 1
    assert eu_red["share"] == 1.0


def test_with_model_metadata_marks_unknown_models() -> None:
    df = pd.DataFrame(
        [
            {
                "model_id": "missing/model",
                "pair_id": "p1",
                "condition": "bare",
                "parsed_choice": "blue",
                "status": "ok",
            },
        ]
    )

    annotated = with_model_metadata(df, models=[])

    assert annotated.loc[0, "provider"] == "Unknown"
    assert annotated.loc[0, "origin"] == "Unknown"
