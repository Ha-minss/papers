from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd


def _selected_statistics(
    y: np.ndarray,
    treatment: np.ndarray,
    selected: np.ndarray,
) -> dict[str, float]:
    selected = np.asarray(selected, dtype=bool)
    selected_y = np.asarray(y, dtype=int)[selected]
    selected_t = np.asarray(treatment, dtype=int)[selected]
    control = selected_t == 0
    treated = selected_t == 1
    n0 = int(control.sum())
    n1 = int(treated.sum())
    p0 = float(selected_y[control].mean()) if n0 else float("nan")
    p1 = float(selected_y[treated].mean()) if n1 else float("nan")
    effect = p0 - p1
    if n0 and n1 and np.isfinite(effect):
        se = math.sqrt(p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1)
    else:
        se = float("nan")
    selected_n = int(selected.sum())
    return {
        "selected_n": selected_n,
        "control_n": n0,
        "treated_n": n1,
        "control_churn_rate": p0,
        "treated_churn_rate": p1,
        "estimated_benefit_pp": effect * 100,
        "benefit_CI_low_pp": (effect - 1.96 * se) * 100,
        "benefit_CI_high_pp": (effect + 1.96 * se) * 100,
        "estimated_prevented_churns": effect * selected_n,
    }


def simulate_campaign_policy(
    score: np.ndarray,
    y: np.ndarray,
    treatment: np.ndarray,
    fraction: float,
    contact_cost: float,
    saved_customer_value: float,
) -> dict[str, float]:
    """Evaluate one fixed-depth ranking under one illustrative economic scenario."""
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1.")
    if contact_cost < 0 or saved_customer_value < 0:
        raise ValueError("economic assumptions must be non-negative.")
    if fraction == 0:
        return {
            "contact_fraction": 0.0,
            "selected_n": 0,
            "control_n": 0,
            "treated_n": 0,
            "control_churn_rate": float("nan"),
            "treated_churn_rate": float("nan"),
            "estimated_benefit_pp": 0.0,
            "benefit_CI_low_pp": 0.0,
            "benefit_CI_high_pp": 0.0,
            "estimated_prevented_churns": 0.0,
            "contact_cost_total": 0.0,
            "gross_value": 0.0,
            "net_value": 0.0,
            "break_even_saved_value": float("inf"),
        }
    selected_n = math.ceil(len(y) * fraction)
    order = np.argsort(np.asarray(score), kind="mergesort")[::-1]
    selected = np.zeros(len(y), dtype=bool)
    selected[order[:selected_n]] = True
    statistics = _selected_statistics(y, treatment, selected)
    prevented = statistics["estimated_prevented_churns"]
    total_cost = selected_n * contact_cost
    gross_value = prevented * saved_customer_value
    return {
        "contact_fraction": fraction,
        **statistics,
        "contact_cost_total": total_cost,
        "gross_value": gross_value,
        "net_value": gross_value - total_cost,
        "break_even_saved_value": total_cost / prevented if prevented > 0 else float("inf"),
    }


def compare_campaign_policies(
    model_scores: dict[str, np.ndarray],
    y: np.ndarray,
    treatment: np.ndarray,
    fractions: tuple[float, ...],
    scenarios: Mapping[str, Mapping[str, float]],
    break_even_contact_cost: float = 30.0,
) -> pd.DataFrame:
    """Evaluate each ranking at each contact depth under named scenarios."""
    rows: list[dict[str, float | str]] = []
    for model_name, score in model_scores.items():
        for fraction in fractions:
            base = simulate_campaign_policy(
                score,
                y,
                treatment,
                fraction,
                contact_cost=break_even_contact_cost,
                saved_customer_value=0.0,
            )
            row: dict[str, float | str] = {
                "model": model_name,
                "contact_fraction": fraction,
                "selected_n": base["selected_n"],
                "control_n": base["control_n"],
                "treated_n": base["treated_n"],
                "control_churn_rate": base["control_churn_rate"],
                "treated_churn_rate": base["treated_churn_rate"],
                "estimated_benefit_pp": base["estimated_benefit_pp"],
                "benefit_CI_low_pp": base["benefit_CI_low_pp"],
                "benefit_CI_high_pp": base["benefit_CI_high_pp"],
                "estimated_prevented_churns": base["estimated_prevented_churns"],
            }
            for scenario_name, assumptions in scenarios.items():
                saved_value = float(assumptions["saved_customer_value"])
                contact_cost = float(assumptions["contact_cost"])
                row[f"{scenario_name}_net_value"] = (
                    base["estimated_prevented_churns"] * saved_value
                    - base["selected_n"] * contact_cost
                )
            prevented = base["estimated_prevented_churns"]
            row[f"break_even_saved_value_at_{break_even_contact_cost:g}_cost"] = (
                base["selected_n"] * break_even_contact_cost / prevented
                if prevented > 0
                else float("inf")
            )
            rows.append(row)
    return pd.DataFrame(rows)
