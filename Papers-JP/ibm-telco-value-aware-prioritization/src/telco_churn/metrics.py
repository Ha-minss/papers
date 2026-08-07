from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def expected_calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.digitize(probability, edges[1:-1], right=True)
    value = 0.0
    for index in range(bins):
        mask = bucket == index
        if mask.any():
            value += mask.mean() * abs(y_true[mask].mean() - probability[mask].mean())
    return float(value)


def top_fraction_metrics(y_true: np.ndarray, probability: np.ndarray, fraction: float) -> dict[str, float]:
    selected_n = math.ceil(len(y_true) * fraction)
    order = np.argsort(probability, kind="mergesort")[::-1][:selected_n]
    precision = float(y_true[order].mean())
    recall = float(y_true[order].sum() / y_true.sum()) if y_true.sum() else 0.0
    prevalence = float(y_true.mean())
    return {
        "precision": precision,
        "recall": recall,
        "lift": precision / prevalence if prevalence else 0.0,
        "selected_n": selected_n,
        "churners": int(y_true[order].sum()),
    }



def dummy_baseline_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    top_fractions: tuple[float, ...] = (0.05, 0.10, 0.20),
) -> dict[str, float]:
    """Evaluate a prior-only model as a no-ranking baseline.

    Fold-specific training prevalences may differ slightly, but the model has no
    customer-level information. Ranking metrics are therefore fixed to their
    no-skill values while proper scoring rules use the actual OOF probabilities.
    """
    probability = np.asarray(probability, dtype=float)
    prevalence = float(np.mean(y_true))
    result: dict[str, float] = {
        "ROC_AUC": 0.5,
        "PR_AUC": prevalence,
        "Brier": float(brier_score_loss(y_true, probability)),
        "LogLoss": float(log_loss(y_true, np.clip(probability, 1e-7, 1 - 1e-7))),
        "ECE10": expected_calibration_error(y_true, probability),
    }
    for fraction in top_fractions:
        percent = int(round(fraction * 100))
        result.update(
            {
                f"Precision@{percent}%": prevalence,
                f"Recall@{percent}%": float(fraction),
                f"Lift@{percent}%": 1.0,
                f"Churners@{percent}%": float(np.sum(y_true) * fraction),
            }
        )
    return result

def classification_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    top_fractions: tuple[float, ...] = (0.05, 0.10, 0.20),
) -> dict[str, float]:
    probability = np.asarray(probability, dtype=float)
    result: dict[str, float] = {
        "ROC_AUC": float(roc_auc_score(y_true, probability)),
        "PR_AUC": float(average_precision_score(y_true, probability)),
        "Brier": float(brier_score_loss(y_true, probability)),
        "LogLoss": float(log_loss(y_true, np.clip(probability, 1e-7, 1 - 1e-7))),
        "ECE10": expected_calibration_error(y_true, probability),
    }
    constant_score = bool(np.allclose(probability, probability[0]))
    for fraction in top_fractions:
        percent = int(round(fraction * 100))
        if constant_score:
            precision = float(y_true.mean())
            recall = float(fraction)
            lift = 1.0
            churners = float(y_true.sum() * fraction)
        else:
            values = top_fraction_metrics(y_true, probability, fraction)
            precision = values["precision"]
            recall = values["recall"]
            lift = values["lift"]
            churners = values["churners"]
        result.update(
            {
                f"Precision@{percent}%": precision,
                f"Recall@{percent}%": recall,
                f"Lift@{percent}%": lift,
                f"Churners@{percent}%": churners,
            }
        )
    return result


def stratified_metric_intervals(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    iterations: int = 1000,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Bootstrap classification metrics while preserving class counts."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    positive = np.flatnonzero(y_true == 1)
    negative = np.flatnonzero(y_true == 0)
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("Both outcome classes are required for stratified bootstrap.")
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"ROC_AUC": [], "PR_AUC": [], "Brier": []}
    for _ in range(iterations):
        index = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        sampled_y = y_true[index]
        sampled_probability = probability[index]
        values["ROC_AUC"].append(float(roc_auc_score(sampled_y, sampled_probability)))
        values["PR_AUC"].append(float(average_precision_score(sampled_y, sampled_probability)))
        values["Brier"].append(float(brier_score_loss(sampled_y, sampled_probability)))
    return {
        f"{name}_CI": [
            float(np.quantile(metric_values, 0.025)),
            float(np.quantile(metric_values, 0.975)),
        ]
        for name, metric_values in values.items()
    }
