from __future__ import annotations

import math

import numpy as np
import pandas as pd


def calculate_value_at_risk(probability: np.ndarray, cltv: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=float)
    cltv = np.asarray(cltv, dtype=float)
    if probability.shape != cltv.shape:
        raise ValueError("probability and CLTV must have the same shape.")
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("probability must be between 0 and 1.")
    if (cltv < 0).any():
        raise ValueError("CLTV must be non-negative.")
    return probability * cltv


def evaluate_priority_policy(
    y_true: np.ndarray,
    cltv: np.ndarray,
    score: np.ndarray,
    fraction: float,
    value_at_risk: np.ndarray | None = None,
) -> dict[str, float]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")
    selected_n = math.ceil(len(y_true) * fraction)
    selected = np.argsort(score, kind="mergesort")[::-1][:selected_n]
    selected_y = y_true[selected]
    selected_cltv = cltv[selected]
    common_value_score = np.asarray(score if value_at_risk is None else value_at_risk, dtype=float)
    if common_value_score.shape != np.asarray(score).shape:
        raise ValueError("value_at_risk and score must have the same shape.")
    total_churned_cltv = float(cltv[y_true == 1].sum())
    observed = float(selected_cltv[selected_y == 1].sum())
    prevalence = float(y_true.mean())
    selected_rate = float(selected_y.mean())
    return {
        "selected_n": selected_n,
        "selected_churn_rate": selected_rate,
        "churners_captured": int(selected_y.sum()),
        "churn_recall": float(selected_y.sum() / y_true.sum()) if y_true.sum() else 0.0,
        "churn_lift": selected_rate / prevalence if prevalence else 0.0,
        "observed_churned_CLTV": observed,
        "observed_churned_CLTV_capture": observed / total_churned_cltv if total_churned_cltv else 0.0,
        "value_capture_lift": (observed / total_churned_cltv) / fraction if total_churned_cltv else 0.0,
        "average_CLTV_selected": float(selected_cltv.mean()),
        "expected_value_at_risk_sum": float(common_value_score[selected].sum()),
    }


def compare_priority_policies(
    y_true: np.ndarray,
    cltv: np.ndarray,
    churn_probability: np.ndarray,
    fractions: tuple[float, ...],
) -> pd.DataFrame:
    value_at_risk = calculate_value_at_risk(churn_probability, cltv)
    policies = {
        "CLTV_only": cltv,
        "Churn_risk": churn_probability,
        "Value_at_Risk": value_at_risk,
    }
    rows: list[dict[str, float | str]] = []
    total_churned_cltv = float(cltv[y_true == 1].sum())
    for fraction in fractions:
        rows.append(
            {
                "policy": "Random_expected",
                "contact_fraction": fraction,
                "selected_n": math.ceil(len(y_true) * fraction),
                "selected_churn_rate": float(y_true.mean()),
                "churners_captured": float(y_true.sum() * fraction),
                "churn_recall": fraction,
                "churn_lift": 1.0,
                "observed_churned_CLTV": total_churned_cltv * fraction,
                "observed_churned_CLTV_capture": fraction,
                "value_capture_lift": 1.0,
                "average_CLTV_selected": float(cltv.mean()),
                "expected_value_at_risk_sum": float(value_at_risk.sum() * fraction),
            }
        )
        for name, score in policies.items():
            result = evaluate_priority_policy(
                y_true, cltv, score, fraction, value_at_risk=value_at_risk
            )
            rows.append({"policy": name, "contact_fraction": fraction, **result})
    return pd.DataFrame(rows)
