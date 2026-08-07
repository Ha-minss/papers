"""Partially pooled logistic regression with shrunken sector slope deviations.

The model uses a shared coefficient for every financial ratio and adds
sector-specific slope deviations through effect-coded interactions. Interaction
columns are multiplied by ``interaction_scale`` before ridge logistic fitting;
values below one make a given sector deviation more expensive under the common
L2 penalty and therefore shrink it toward the shared slope.
"""
from __future__ import annotations

from itertools import product
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .research_pipeline import SECTORS, make_preprocessor


def effect_code_sectors(
    sectors: Sequence[str],
    levels: Sequence[str] = SECTORS,
) -> tuple[np.ndarray, list[str]]:
    """Return symmetric effect coding with K-1 columns for K sector levels.

    The final level receives -1 in every column. Each earlier level receives 1
    in its own column and 0 elsewhere. With one observation per level, every
    contrast column sums to zero.
    """
    levels = tuple(levels)
    if len(levels) < 2:
        raise ValueError("At least two sector levels are required")
    index = {level: i for i, level in enumerate(levels)}
    out = np.zeros((len(sectors), len(levels) - 1), dtype=float)
    for row, sector in enumerate(sectors):
        if sector not in index:
            raise ValueError(f"Unknown sector: {sector}")
        sector_index = index[sector]
        if sector_index == len(levels) - 1:
            out[row, :] = -1.0
        else:
            out[row, sector_index] = 1.0
    columns = [f"sector_{level}" for level in levels[:-1]]
    return out, columns


def make_shrinkage_interaction_design(
    X_shared: np.ndarray,
    sectors: Sequence[str],
    interaction_scale: float,
    levels: Sequence[str] = SECTORS,
) -> np.ndarray:
    """Build shared slopes, sector intercept contrasts, and shrunken slopes."""
    X = np.asarray(X_shared, dtype=float)
    if X.ndim != 2:
        raise ValueError("X_shared must be a 2D matrix")
    if len(X) != len(sectors):
        raise ValueError("X_shared and sectors must have the same row count")
    if interaction_scale < 0:
        raise ValueError("interaction_scale must be non-negative")
    contrasts, _ = effect_code_sectors(sectors, levels=levels)
    interactions = np.einsum("ij,ik->ijk", X, contrasts).reshape(len(X), -1)
    interactions *= float(interaction_scale)
    return np.column_stack([X, contrasts, interactions])


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    preprocess_mode: str,
    C: float,
    interaction_scale: float,
    seed: int,
) -> np.ndarray:
    preprocessor = make_preprocessor(preprocess_mode, features, model_family="linear")
    X_train_shared = preprocessor.fit_transform(train.loc[:, features])
    X_test_shared = preprocessor.transform(test.loc[:, features])
    X_train = make_shrinkage_interaction_design(
        X_train_shared, train["sector"].tolist(), interaction_scale,
    )
    X_test = make_shrinkage_interaction_design(
        X_test_shared, test["sector"].tolist(), interaction_scale,
    )
    model = LogisticRegression(
        C=float(C),
        solver="liblinear",
        max_iter=5000,
        random_state=seed,
    )
    model.fit(X_train, train["target"].to_numpy(dtype=int))
    return model.predict_proba(X_test)[:, 1]


def _cross_fitted_predictions(
    frame: pd.DataFrame,
    features: Sequence[str],
    preprocess_mode: str,
    C: float,
    interaction_scale: float,
    seed: int,
    n_splits: int,
) -> np.ndarray:
    frame = frame.reset_index(drop=True)
    y = frame["target"].to_numpy(dtype=int)
    min_class = int(np.bincount(y, minlength=2).min())
    splits = max(2, min(int(n_splits), min_class))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    predictions = np.full(len(frame), np.nan, dtype=float)
    for fold, (train_index, validation_index) in enumerate(cv.split(np.zeros(len(y)), y)):
        predictions[validation_index] = _fit_predict(
            frame.iloc[train_index],
            frame.iloc[validation_index],
            features,
            preprocess_mode,
            C,
            interaction_scale,
            seed + fold,
        )
    if not np.isfinite(predictions).all():
        raise RuntimeError("Incomplete shrinkage-interaction OOF predictions")
    return predictions


def shrinkage_interaction_oof_and_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    preprocess_mode: str = "reference",
    seed: int = 42,
    C_grid: Sequence[float] = (0.01, 0.03, 0.1),
    interaction_scale_grid: Sequence[float] = (0.1, 0.25, 0.5),
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Tune shrinkage on training OOF predictions and score a future set.

    Selection uses training-only out-of-fold average precision. Brier score and
    the stronger-shrinkage setting are deterministic tie breakers. The future
    test set never participates in hyperparameter selection.
    """
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    y = train["target"].to_numpy(dtype=int)
    tuning_rows: list[dict] = []
    candidate_predictions: dict[tuple[float, float], np.ndarray] = {}

    for C, interaction_scale in product(C_grid, interaction_scale_grid):
        C = float(C)
        interaction_scale = float(interaction_scale)
        oof = _cross_fitted_predictions(
            train,
            features,
            preprocess_mode,
            C,
            interaction_scale,
            seed,
            n_splits,
        )
        candidate_predictions[(C, interaction_scale)] = oof
        tuning_rows.append({
            "C": C,
            "interaction_scale": interaction_scale,
            "oof_pr_auc": float(average_precision_score(y, oof)),
            "oof_roc_auc": float(roc_auc_score(y, oof)),
            "oof_brier": float(brier_score_loss(y, oof)),
        })

    # Highest PR-AUC first; then lowest Brier; then stronger interaction
    # shrinkage and lower model flexibility for a deterministic conservative tie.
    selected = sorted(
        tuning_rows,
        key=lambda row: (
            -row["oof_pr_auc"],
            row["oof_brier"],
            row["interaction_scale"],
            row["C"],
        ),
    )[0]
    selected_key = (selected["C"], selected["interaction_scale"])
    selected_oof = candidate_predictions[selected_key]
    future = _fit_predict(
        train,
        test,
        features,
        preprocess_mode,
        selected["C"],
        selected["interaction_scale"],
        seed,
    )
    metadata = {
        "selected_C": selected["C"],
        "selected_interaction_scale": selected["interaction_scale"],
        "selection_metric": "training_oof_pr_auc_then_brier",
        "tuning_results": tuning_rows,
        "design": "shared slopes + effect-coded sector intercepts + ridge-shrunken sector slope deviations",
    }
    return selected_oof, future, metadata
