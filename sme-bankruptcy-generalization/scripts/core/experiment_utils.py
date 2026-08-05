from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .io import write_csv
from . import research_pipeline as rp


def select_features(feature_set: str) -> list[str]:
    normalized = feature_set.lower().replace("-", "_")
    if normalized in {"candidate_60", "candidate60"}:
        return rp.candidate_60_columns()
    if normalized in {"all_63", "all63"}:
        return rp.all_feature_columns()
    if normalized in {"survey_20", "survey20"}:
        return rp.survey_20_columns()
    raise ValueError(f"Unknown feature set: {feature_set}")


def validate_target_year(year: int, config: ExperimentConfig) -> None:
    if year not in config.evaluation_years:
        allowed = ", ".join(str(value) for value in config.evaluation_years)
        raise ValueError(f"Unsupported evaluation year {year}; choose one of {allowed}")


def merge_rows(
    path: str | Path,
    new_rows: pd.DataFrame,
    unique_columns: Sequence[str],
    *,
    compression: str | None = None,
) -> pd.DataFrame:
    output = Path(path)
    if output.exists():
        existing = pd.read_csv(output)
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows.copy()
    combined = combined.drop_duplicates(list(unique_columns), keep="last")
    write_csv(output, combined, compression=compression)
    return combined


def sector_metrics(
    data: pd.DataFrame,
    scores: Sequence[float],
    threshold: float,
) -> pd.DataFrame:
    score_array = np.asarray(scores, dtype=float)
    rows: list[dict] = []
    for sector in rp.SECTORS:
        mask = data["sector"].eq(sector).to_numpy()
        metrics = rp.evaluate_scores(data.loc[mask, "target"], score_array[mask], threshold)
        metrics["sector"] = sector
        rows.append(metrics)
    return pd.DataFrame(rows)


def add_macro_metrics(metrics: dict, sector_table: pd.DataFrame) -> dict:
    output = dict(metrics)
    columns = (
        "roc_auc",
        "pr_auc",
        "brier",
        "gmean_selected",
        "recall_at_1pct",
        "recall_at_3pct",
        "recall_at_5pct",
    )
    for column in columns:
        output[f"macro_{column}"] = float(sector_table[column].mean(skipna=True))
        output[f"std_sector_{column}"] = float(sector_table[column].std(ddof=1, skipna=True))
    return output


def external_path_help(kind: str) -> str:
    return (
        f"Path to the external {kind}. Data and generated artifacts are intentionally "
        "kept outside the Git repository."
    )
