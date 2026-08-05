from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .core import research_pipeline as rp
from .core.config import load_experiment_config
from .core.diagnostic_analysis import (
    annual_event_summary,
    calibration_bins,
    event_count_performance_gap,
    standardized_wasserstein_drift,
)
from .core.experiment_utils import select_features
from .core.io import ensure_work_subdir, load_prepared_frame, write_csv, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build data-drift, calibration, and sector-sparsity diagnostics without plotting."
    )
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "experiment.json"))
    parser.add_argument("--reference-year", type=int, default=2013)
    return parser


def _calibration_outputs(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    bin_rows: list[pd.DataFrame] = []
    grouping = ["model", "imbalance"]
    for keys, group in predictions.groupby(grouping, sort=True):
        model, imbalance = keys
        y = group["target"].to_numpy(dtype=int)
        scores = group["score"].to_numpy(dtype=float)
        actual_rate = float(y.mean())
        mean_risk = float(scores.mean())
        summary_rows.append(
            {
                "model": model,
                "imbalance": imbalance,
                "n_obs": len(group),
                "n_positive": int(y.sum()),
                "actual_rate": actual_rate,
                "mean_predicted_risk": mean_risk,
                "predicted_to_actual_ratio": mean_risk / actual_rate if actual_rate else np.nan,
                "pr_auc": float(average_precision_score(y, scores)),
                "roc_auc": float(roc_auc_score(y, scores)),
                "brier": float(brier_score_loss(y, scores)),
            }
        )
        bins = calibration_bins(y, scores, n_bins=10)
        bins["model"] = model
        bins["imbalance"] = imbalance
        bin_rows.append(bins)
    return pd.DataFrame(summary_rows), pd.concat(bin_rows, ignore_index=True)


def _event_count_correlations(gaps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in gaps.groupby("model", sort=True):
        if len(group) < 3:
            correlation = np.nan
            p_value = np.nan
        else:
            result = spearmanr(group["train_positive"], group["pr_auc_gap"], nan_policy="omit")
            correlation = float(result.statistic)
            p_value = float(result.pvalue)
        rows.append(
            {
                "model": model,
                "n_sector_year_cells": len(group),
                "spearman_rho": correlation,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_experiment_config(args.config)
    frame = load_prepared_frame(args.data_file, args.work_dir)
    features = select_features(config.feature_set)
    tables_dir = ensure_work_subdir(args.work_dir, "diagnostics")
    findings: dict[str, object] = {}

    annual = annual_event_summary(frame)
    write_csv(tables_dir / "annual_data_summary.csv", annual)
    findings["annual_event_rate"] = annual.to_dict("records")

    missingness = (
        frame.groupby("eval_year")[features]
        .apply(lambda group: group.isna().mean())
        .reset_index()
        .melt(id_vars="eval_year", var_name="feature", value_name="missing_rate")
    )
    write_csv(tables_dir / "missingness_by_year.csv", missingness)

    drift = standardized_wasserstein_drift(frame, features, args.reference_year)
    write_csv(tables_dir / "feature_distribution_drift.csv", drift)
    findings["largest_distribution_shifts"] = (
        drift[drift["eval_year"].ne(args.reference_year)]
        .sort_values("standardized_wasserstein", ascending=False)
        .head(10)[["feature", "eval_year", "standardized_wasserstein"]]
        .to_dict("records")
    )

    prediction_path = Path(args.work_dir).expanduser().resolve() / "predictions" / "factorial_predictions.csv.gz"
    if prediction_path.is_file():
        predictions = pd.read_csv(prediction_path)
        predictions = predictions[
            predictions["evaluation"].eq("rolling_oot")
            & predictions["preprocess"].eq("reference")
        ].copy()
        summary, bins = _calibration_outputs(predictions)
        write_csv(tables_dir / "resampling_calibration_summary.csv", summary)
        write_csv(tables_dir / "resampling_calibration_bins.csv", bins)
        findings["resampling_calibration"] = summary.to_dict("records")
    else:
        findings["resampling_calibration"] = "Skipped: factorial_predictions.csv.gz not found"

    sector_path = Path(args.work_dir).expanduser().resolve() / "tables" / "structure_sector_metrics.csv"
    if sector_path.is_file():
        sector_metrics = pd.read_csv(sector_path)
        sector_metrics = sector_metrics[sector_metrics["evaluation"].eq("rolling_oot")]
        gaps = event_count_performance_gap(sector_metrics)
        correlations = _event_count_correlations(gaps)
        write_csv(tables_dir / "sector_event_count_performance_gap.csv", gaps)
        write_csv(tables_dir / "sector_event_count_correlations.csv", correlations)
        findings["sector_event_count_correlations"] = correlations.to_dict("records")
    else:
        findings["sector_event_count_correlations"] = "Skipped: structure_sector_metrics.csv not found"

    write_json(tables_dir / "diagnostic_findings.json", findings)
    print(f"Wrote diagnostic tables to {tables_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
