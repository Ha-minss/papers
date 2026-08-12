from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ALPHA_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)


def classify_alpha_mechanism(
    endpoint_gap_pp: float,
    positive_adjacent_n: int,
    positive_seed_slope_n: int,
) -> str:
    if (
        endpoint_gap_pp >= 1.0
        and positive_adjacent_n == 4
        and positive_seed_slope_n >= 24
    ):
        return "strong"
    if (
        endpoint_gap_pp >= 0.5
        and positive_adjacent_n >= 3
        and positive_seed_slope_n >= 18
    ):
        return "partial"
    return "failure"


def _validated_panel(panel: pd.DataFrame) -> pd.DataFrame:
    clean = panel.copy()
    if len(clean) != 150:
        raise ValueError(f"alpha panel must contain 150 rows, found {len(clean)}")
    if not clean["status"].eq("ok").all():
        raise ValueError("alpha panel contains unsuccessful rows")
    if clean.duplicated(["alpha", "seed"]).any():
        raise ValueError("alpha panel contains duplicate alpha-seed keys")
    if set(clean["alpha"].astype(float)) != set(ALPHA_GRID):
        raise ValueError("alpha panel differs from the preregistered grid")
    counts = clean.groupby("alpha")["seed"].nunique()
    if not counts.eq(30).all():
        raise ValueError("each alpha must contain exactly 30 seeds")
    return clean


def alpha_summary(panel: pd.DataFrame) -> pd.DataFrame:
    clean = _validated_panel(panel)
    rows = []
    for alpha, group in clean.groupby("alpha", sort=True):
        rows.append(
            {
                "alpha": float(alpha),
                "n": int(len(group)),
                "ovd_mean": float(group["ovd_pct"].mean()),
                "ovd_median": float(group["ovd_pct"].median()),
                "ovd_q10": float(group["ovd_pct"].quantile(0.10)),
                "ovd_q90": float(group["ovd_pct"].quantile(0.90)),
                "ovd_positive_n": int((group["ovd_pct"] > 0.0).sum()),
                "joint_cost_mean": float(group["joint_cost"].mean()),
                "ind_cost_mean": float(group["ind_cost"].mean()),
                "joint_max_oper008_util": float(group["joint_max_oper008_util"].max()),
                "ind_max_oper008_util": float(group["ind_max_oper008_util"].max()),
                "mean_action_l1": float(group["mean_action_l1"].mean()),
            }
        )
    result = pd.DataFrame(rows)
    result["adjacent_ovd_increase_pp"] = result["ovd_mean"].diff()
    return result


def seed_slopes(panel: pd.DataFrame) -> pd.DataFrame:
    clean = _validated_panel(panel)
    rows = []
    for seed, group in clean.groupby("seed", sort=True):
        ordered = group.sort_values("alpha")
        slope, intercept = np.polyfit(
            ordered["alpha"].to_numpy(dtype=float),
            ordered["ovd_pct"].to_numpy(dtype=float),
            1,
        )
        rows.append(
            {
                "seed": int(seed),
                "ovd_slope_pp_per_alpha": float(slope),
                "ovd_intercept": float(intercept),
                "positive_slope": bool(slope > 0.0),
            }
        )
    return pd.DataFrame(rows)


def stage2_decision(panel: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    summary = alpha_summary(panel)
    slopes = seed_slopes(panel)
    endpoint_gap = float(summary.iloc[-1]["ovd_mean"] - summary.iloc[0]["ovd_mean"])
    positive_adjacent = int((summary["adjacent_ovd_increase_pp"].dropna() > 0.0).sum())
    positive_slopes = int(slopes["positive_slope"].sum())
    classification = classify_alpha_mechanism(
        endpoint_gap,
        positive_adjacent,
        positive_slopes,
    )
    decision = {
        "classification": classification,
        "predictive_validation_allowed": classification in {"strong", "partial"},
        "endpoint_gap_pp": endpoint_gap,
        "positive_adjacent_n": positive_adjacent,
        "positive_seed_slope_n": positive_slopes,
        "thresholds": {
            "strong_endpoint_gap_pp": 1.0,
            "strong_positive_adjacent_n": 4,
            "strong_positive_seed_slope_n": 24,
            "partial_endpoint_gap_pp": 0.5,
            "partial_positive_adjacent_n": 3,
            "partial_positive_seed_slope_n": 18,
        },
        "interpretation_scope": "computational mechanism evidence; seed variation is not population inference",
    }
    return decision, summary, slopes


def write_stage2_outputs(panel: pd.DataFrame, results_dir: Path) -> dict:
    decision, summary, slopes = stage2_decision(panel)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(results_dir / "stage2_alpha_summary.csv", index=False)
    slopes.to_csv(results_dir / "stage2_seed_slopes.csv", index=False)
    destination = results_dir / "stage2_decision.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the preregistered Stage 2 alpha sweep")
    parser.add_argument("panel_csv", type=Path)
    parser.add_argument("--results-dir", type=Path, default=Path("../ijpe_local/results/stage2"))
    args = parser.parse_args()
    decision = write_stage2_outputs(pd.read_csv(args.panel_csv), args.results_dir)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
