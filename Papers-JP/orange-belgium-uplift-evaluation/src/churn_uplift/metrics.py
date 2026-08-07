from __future__ import annotations

import math

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def fold_percentile_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("values must be a non-empty one-dimensional array.")
    return rankdata(values, method="average") / len(values)


def selected_group_effect(y: np.ndarray, treatment: np.ndarray, selected: np.ndarray) -> float:
    selected = np.asarray(selected, dtype=bool)
    control = selected & (treatment == 0)
    treated = selected & (treatment == 1)
    if not control.any() or not treated.any():
        return float("nan")
    return float(y[control].mean() - y[treated].mean())


def hajek_uplift_curve_metrics(
    score: np.ndarray,
    y: np.ndarray,
    treatment: np.ndarray,
    top_fractions: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.50),
) -> dict[str, float]:
    score = np.asarray(score, dtype=float)
    y = np.asarray(y, dtype=int)
    treatment = np.asarray(treatment, dtype=int)
    overall_effect = selected_group_effect(y, treatment, np.ones(len(y), dtype=bool))
    if np.allclose(score, score[0]):
        result = {"Qini": 0.0, "AUUC": overall_effect / 2.0, "ATE_benefit_pp": overall_effect * 100}
        for fraction in top_fractions:
            result[f"Top{int(round(fraction * 100))}_benefit_pp"] = overall_effect * 100
        return result
    order = np.argsort(score, kind="mergesort")[::-1]
    sorted_y = y[order]
    sorted_t = treatment[order]
    control = sorted_t == 0
    treated = sorted_t == 1
    cumulative_control = np.cumsum(control)
    cumulative_treated = np.cumsum(treated)
    cumulative_y_control = np.cumsum(sorted_y * control)
    cumulative_y_treated = np.cumsum(sorted_y * treated)
    effect = np.full(len(y), np.nan, dtype=float)
    valid = (cumulative_control > 0) & (cumulative_treated > 0)
    effect[valid] = (
        cumulative_y_control[valid] / cumulative_control[valid]
        - cumulative_y_treated[valid] / cumulative_treated[valid]
    )
    effect = np.where(np.isnan(effect), 0.0, effect)
    fraction_axis = np.arange(1, len(y) + 1) / len(y)
    gain = effect * fraction_axis
    random_line = fraction_axis * overall_effect
    auuc = float(np.trapezoid(gain, fraction_axis))
    qini = float(np.trapezoid(gain - random_line, fraction_axis))
    result = {"Qini": qini, "AUUC": auuc, "ATE_benefit_pp": overall_effect * 100}
    for fraction in top_fractions:
        index = min(len(y) - 1, math.ceil(len(y) * fraction) - 1)
        result[f"Top{int(round(fraction * 100))}_benefit_pp"] = float(effect[index] * 100)
    return result


def expected_calibration_error(
    y: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    probability = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.digitize(probability, edges[1:-1], right=True)
    value = 0.0
    for index in range(bins):
        mask = bucket == index
        if mask.any():
            value += mask.mean() * abs(y[mask].mean() - probability[mask].mean())
    return float(value)


def risk_classification_metrics(
    y: np.ndarray,
    probability: np.ndarray,
    top_fractions: tuple[float, ...] = (0.05, 0.10, 0.20),
) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    result: dict[str, float] = {
        "ROC_AUC": float(roc_auc_score(y, probability)),
        "PR_AUC": float(average_precision_score(y, probability)),
        "Brier": float(brier_score_loss(y, probability)),
        "LogLoss": float(log_loss(y, np.clip(probability, 1e-7, 1 - 1e-7))),
        "ECE10": expected_calibration_error(y, probability, bins=10),
    }
    prevalence = float(y.mean())
    total_churners = int(y.sum())
    for fraction in top_fractions:
        selected_n = math.ceil(len(y) * fraction)
        selected = np.argsort(probability, kind="mergesort")[::-1][:selected_n]
        churners = int(y[selected].sum())
        precision = float(churners / selected_n)
        recall = float(churners / total_churners) if total_churners else 0.0
        lift = float(precision / prevalence) if prevalence else 0.0
        percent = int(round(fraction * 100))
        result.update(
            {
                f"Precision@{percent}%": precision,
                f"Recall@{percent}%": recall,
                f"Lift@{percent}%": lift,
                f"Churners@{percent}%": churners,
            }
        )
    return result


def stratified_bootstrap_indices(
    treatment: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Resample control and treated rows separately, preserving group sizes."""
    treatment = np.asarray(treatment, dtype=int)
    control = np.flatnonzero(treatment == 0)
    treated = np.flatnonzero(treatment == 1)
    if len(control) == 0 or len(treated) == 0:
        raise ValueError("Both treatment groups are required for bootstrap sampling.")
    return np.concatenate(
        [
            rng.choice(control, size=len(control), replace=True),
            rng.choice(treated, size=len(treated), replace=True),
        ]
    )


def bootstrap_metric_interval(
    score: np.ndarray,
    y: np.ndarray,
    treatment: np.ndarray,
    metric_name: str,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        index = stratified_bootstrap_indices(treatment, rng)
        metric = hajek_uplift_curve_metrics(score[index], y[index], treatment[index]).get(metric_name)
        if metric is not None and np.isfinite(metric):
            values.append(metric)
    if not values:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))
