from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


ABLATIONS = ("nonbinding", "dedicated", "no_bom")


def classify_contrast(
    mean_full: float,
    mean_ablation: float,
    same_signed_n: int,
) -> str:
    reduction_pp = mean_full - mean_ablation
    reduction_fraction = reduction_pp / abs(mean_full) if mean_full != 0 else 0.0
    increase_pp = mean_ablation - mean_full
    opposite = increase_pp >= 0.5 or (
        mean_full != 0 and increase_pp / abs(mean_full) >= 0.20
    )
    if opposite:
        return "failure"
    if reduction_fraction >= 0.50 and reduction_pp >= 1.0 and same_signed_n >= 24:
        return "strong"
    if (reduction_fraction >= 0.20 or reduction_pp >= 0.5) and same_signed_n >= 18:
        return "partial"
    return "failure"


def _successful(df: pd.DataFrame) -> pd.DataFrame:
    if "status" not in df:
        raise ValueError("input must include status")
    return df.loc[df["status"] == "ok"].copy()


def summarize_architectures(df: pd.DataFrame) -> pd.DataFrame:
    clean = _successful(df)
    rows = []
    for architecture, group in clean.groupby("architecture", sort=False):
        ovd = group["ovd_pct"]
        rows.append(
            {
                "architecture": architecture,
                "n": int(len(group)),
                "ovd_mean": float(ovd.mean()),
                "ovd_median": float(ovd.median()),
                "ovd_q10": float(ovd.quantile(0.10)),
                "ovd_q25": float(ovd.quantile(0.25)),
                "ovd_q75": float(ovd.quantile(0.75)),
                "ovd_q90": float(ovd.quantile(0.90)),
                "ovd_positive_n": int((ovd > 0.0).sum()),
                "joint_holding_mean": float(group["joint_holding"].mean()),
                "ind_holding_mean": float(group["ind_holding"].mean()),
                "joint_backorder_mean": float(group["joint_backorder"].mean()),
                "ind_backorder_mean": float(group["ind_backorder"].mean()),
                "joint_max_oper004_util": float(group["joint_max_oper004_util"].max()),
                "ind_max_oper004_util": float(group["ind_max_oper004_util"].max()),
                "joint_max_oper008_util": float(group["joint_max_oper008_util"].max()),
                "ind_max_oper008_util": float(group["ind_max_oper008_util"].max()),
                "mean_action_l1": float(group["mean_action_l1"].mean()),
            }
        )
    return pd.DataFrame(rows)


def paired_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    clean = _successful(df)
    baseline = clean.loc[
        clean["architecture"] == "baseline", ["seed", "ovd_pct"]
    ].rename(columns={"ovd_pct": "baseline_ovd_pct"})
    rows = []
    for architecture in ABLATIONS:
        ablation = clean.loc[
            clean["architecture"] == architecture, ["seed", "ovd_pct"]
        ].rename(columns={"ovd_pct": "ablation_ovd_pct"})
        paired = baseline.merge(ablation, on="seed", validate="one_to_one")
        reduction = paired["baseline_ovd_pct"] - paired["ablation_ovd_pct"]
        mean_full = float(paired["baseline_ovd_pct"].mean())
        mean_ablation = float(paired["ablation_ovd_pct"].mean())
        same_signed_n = int((reduction > 0.0).sum())
        rows.append(
            {
                "architecture": architecture,
                "paired_n": int(len(paired)),
                "baseline_ovd_mean": mean_full,
                "ablation_ovd_mean": mean_ablation,
                "reduction_pp": mean_full - mean_ablation,
                "reduction_fraction": (
                    (mean_full - mean_ablation) / abs(mean_full)
                    if mean_full != 0.0
                    else 0.0
                ),
                "same_signed_n": same_signed_n,
                "classification": classify_contrast(
                    mean_full, mean_ablation, same_signed_n
                ),
            }
        )
    return pd.DataFrame(rows)


def write_stage1_outputs(df: pd.DataFrame, results_dir: Path) -> dict:
    summary = summarize_architectures(df)
    contrasts = paired_contrasts(df)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(results_dir / "stage1_architecture_summary.csv", index=False)
    contrasts.to_csv(results_dir / "stage1_paired_contrasts.csv", index=False)
    classifications = dict(zip(contrasts["architecture"], contrasts["classification"]))
    decision = {
        "stage2_allowed": classifications.get("dedicated") in {"strong", "partial"},
        "mechanism_extension_allowed": any(
            value != "failure" for value in classifications.values()
        ),
        "classifications": classifications,
        "thresholds": {
            "strong_reduction_fraction": 0.50,
            "strong_reduction_pp": 1.0,
            "strong_same_signed_n": 24,
            "partial_reduction_fraction": 0.20,
            "partial_reduction_pp": 0.5,
            "partial_same_signed_n": 18,
        },
        "rule": {
            "strong": "reduction_fraction >= 0.50 AND reduction_pp >= 1.0 AND same_signed_n >= 24",
            "partial": "(reduction_fraction >= 0.20 OR reduction_pp >= 0.5) AND same_signed_n >= 18",
            "opposite_failure": "increase >= 0.5 pp OR >= 20%",
        },
    }
    decision_path = results_dir / "stage1_decision.json"
    temporary = decision_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, decision_path)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize POM v4 Stage 1 ablations")
    parser.add_argument("runs_csv", type=Path)
    parser.add_argument("--results-dir", type=Path, default=Path("../ijpe_local/results/stage1"))
    args = parser.parse_args()
    decision = write_stage1_outputs(pd.read_csv(args.runs_csv), args.results_dir)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
