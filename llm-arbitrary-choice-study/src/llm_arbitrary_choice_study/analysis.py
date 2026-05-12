from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px

from .config import load_models
from .schema import ModelSpec


def load_results(db_path: str | Path) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query("SELECT * FROM trials", conn)
    finally:
        conn.close()


def model_metadata_frame(models: list[ModelSpec] | None = None) -> pd.DataFrame:
    model_specs = models if models is not None else load_models()
    return pd.DataFrame(
        [
            {
                "model_id": model.id,
                "model_label": model.label,
                "provider": model.provider,
                "origin": model.origin,
                "tier": model.tier,
            }
            for model in model_specs
        ],
        columns=["model_id", "model_label", "provider", "origin", "tier"],
    )


def with_model_metadata(df: pd.DataFrame, models: list[ModelSpec] | None = None) -> pd.DataFrame:
    annotated = df.merge(model_metadata_frame(models), on="model_id", how="left")
    for column in ("model_label", "provider", "origin", "tier"):
        annotated[column] = annotated[column].fillna("Unknown")
    return annotated


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["status"].isin(["ok", "invalid"])]
    grouped = (
        valid.groupby(["model_id", "pair_id", "condition", "parsed_choice"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    totals = grouped.groupby(["model_id", "pair_id", "condition"])["n"].transform("sum")
    grouped["share"] = grouped["n"] / totals
    return grouped


def summarize_by(df: pd.DataFrame, dimensions: list[str]) -> pd.DataFrame:
    missing = sorted(set(dimensions) - set(df.columns))
    if missing:
        raise ValueError(f"Cannot summarize by missing columns: {missing}")
    valid = df[df["status"].isin(["ok", "invalid"])]
    grouping = [*dimensions, "pair_id", "condition", "parsed_choice"]
    grouped = valid.groupby(grouping, dropna=False).size().reset_index(name="n")
    total_grouping = [*dimensions, "pair_id", "condition"]
    totals = grouped.groupby(total_grouping)["n"].transform("sum")
    grouped["share"] = grouped["n"] / totals
    return grouped


def write_table(df: pd.DataFrame, out: Path, stem: str) -> None:
    df.to_json(out / f"{stem}.json", orient="records", force_ascii=False, indent=2)
    (out / f"{stem}.csv").write_text(df.to_csv(index=False), encoding="utf-8")


def export_summary(db_path: str | Path, out_dir: str | Path = "results/processed") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_results(db_path)
    annotated = with_model_metadata(df)
    summary = summarize(df)
    df.to_json(out / "trials.json", orient="records", force_ascii=False, indent=2)
    write_table(summary, out, "summary")
    write_table(summarize_by(annotated, ["provider"]), out, "summary_by_provider")
    write_table(summarize_by(annotated, ["origin"]), out, "summary_by_origin")
    write_table(summarize_by(annotated, ["origin", "provider"]), out, "summary_by_origin_provider")


def make_charts(db_path: str | Path, out_dir: str | Path = "results/charts") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(load_results(db_path))
    valid = summary[summary["parsed_choice"].notna()]
    if valid.empty:
        return
    fig = px.bar(
        valid,
        x="pair_id",
        y="share",
        color="parsed_choice",
        facet_row="condition",
        facet_col="model_id",
        title="Choice share by model, pair, and condition",
    )
    fig.write_html(out / "choice_share.html", include_plotlyjs="cdn")
    metadata = {"chart_files": ["choice_share.html"]}
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
