from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreparedUpliftData:
    frame: pd.DataFrame
    features: pd.DataFrame
    target: np.ndarray
    treatment: np.ndarray
    strata: np.ndarray
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    excluded_columns: list[str]


def load_uplift_data(path: str | Path) -> pd.DataFrame:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Provide the local CSV path with --input; see data/README.md."
        )
    if data_path.suffix.lower() != ".csv":
        raise ValueError("Orange uplift input must be a CSV file.")
    return pd.read_csv(data_path)


def validate_uplift_frame(frame: pd.DataFrame) -> None:
    missing = {"y", "t"}.difference(frame.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    if not frame["y"].isin([0, 1]).all():
        raise ValueError("y must contain only 0 and 1.")
    if set(frame["y"].dropna().astype(int).unique()) != {0, 1}:
        raise ValueError("y must contain both 0 and 1.")
    if not frame["t"].isin([0, 1]).all():
        raise ValueError("t must contain only 0 and 1.")
    if set(frame["t"].dropna().astype(int).unique()) != {0, 1}:
        raise ValueError("t must contain both 0 and 1.")
    if frame.empty:
        raise ValueError("Dataset must not be empty.")
    if frame.duplicated().any():
        raise ValueError("Exact duplicate rows are not allowed.")
    if not any(column.startswith("PC") for column in frame.columns):
        raise ValueError("At least one PC feature is required.")


def _stratified_sample(frame: pd.DataFrame, rows: int, seed: int) -> pd.DataFrame:
    working = frame.assign(_stratum=2 * frame["t"].astype(int) + frame["y"].astype(int))
    pieces = []
    for _, group in working.groupby("_stratum", sort=False):
        n_group = max(1, round(rows * len(group) / len(working)))
        pieces.append(group.sample(min(n_group, len(group)), random_state=seed))
    sampled = pd.concat(pieces).sample(frac=1.0, random_state=seed).head(rows)
    return sampled.drop(columns="_stratum").reset_index(drop=True)


def prepare_uplift_frame(
    frame: pd.DataFrame,
    near_constant_threshold: float = 0.999,
    max_rows: int | None = None,
    seed: int = 42,
) -> PreparedUpliftData:
    validate_uplift_frame(frame)
    prepared = frame.copy()
    if max_rows is not None and len(prepared) > max_rows:
        prepared = _stratified_sample(prepared, max_rows, seed)
    candidate_columns = [column for column in prepared.columns if column not in {"y", "t"}]
    excluded = []
    for column in candidate_columns:
        counts = prepared[column].value_counts(dropna=False, normalize=True)
        if prepared[column].nunique(dropna=False) <= 1 or (not counts.empty and counts.iloc[0] >= near_constant_threshold):
            excluded.append(column)
    feature_columns = [column for column in candidate_columns if column not in excluded]
    numeric_columns = [column for column in feature_columns if column.startswith("PC")]
    categorical_columns = [column for column in feature_columns if column.startswith("FACTOR")]
    features = prepared[feature_columns].copy()
    target = prepared["y"].astype(int).to_numpy()
    treatment = prepared["t"].astype(int).to_numpy()
    strata = 2 * treatment + target
    return PreparedUpliftData(
        frame=prepared,
        features=features,
        target=target,
        treatment=treatment,
        strata=strata,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        excluded_columns=excluded,
    )
