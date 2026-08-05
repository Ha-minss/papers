from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .core.config import load_experiment_config
from .core.io import ensure_work_subdir, write_csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paired bootstrap comparison for partial-pooling slopes.")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "experiment.json"))
    parser.add_argument("--repetitions", type=int, default=None)
    return parser


def _metrics(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            average_precision_score(y, scores),
            roc_auc_score(y, scores),
            brier_score_loss(y, scores),
        ],
        dtype=float,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_experiment_config(args.config)
    repetitions = args.repetitions or config.bootstrap_repetitions
    predictions_dir = Path(args.work_dir).expanduser().resolve() / "predictions"
    partial_path = predictions_dir / "partial_pooling_predictions.csv.gz"
    if not partial_path.is_file():
        raise FileNotFoundError(f"Missing {partial_path}; run run_partial_pooling first.")
    partial = pd.read_csv(partial_path)

    baseline_frames = []
    for path in sorted(predictions_dir.glob("oot_*_logistic.csv.gz")):
        baseline_frames.append(pd.read_csv(path))
    if not baseline_frames:
        raise FileNotFoundError("No logistic OOT predictions found; run run_primary_predictions first.")
    baseline = pd.concat(baseline_frames, ignore_index=True)
    combined = pd.concat([baseline, partial], ignore_index=True)
    wide = combined.pivot_table(
        index=["row_id", "sector", "eval_year", "target"],
        columns="structure",
        values="score",
        aggfunc="first",
    ).reset_index()
    structures = ("pooled", "partial_pool", "sector_specific", "partial_pool_slopes")
    missing = [name for name in structures if name not in wide]
    if missing:
        raise ValueError(f"Missing structures in prediction data: {missing}")
    wide = wide.dropna(subset=list(structures)).reset_index(drop=True)

    years: dict[int, dict] = {}
    for year, group in wide.groupby("eval_year", sort=True):
        group = group.reset_index(drop=True)
        strata = [
            part.index.to_numpy()
            for _, part in group.groupby(["sector", "target"], sort=True)
        ]
        years[int(year)] = {
            "target": group["target"].to_numpy(dtype=int),
            "scores": {name: group[name].to_numpy(dtype=float) for name in structures},
            "strata": strata,
        }

    def mean_metrics(structure: str, sampled: dict[int, np.ndarray] | None = None) -> np.ndarray:
        values = []
        for year, payload in years.items():
            indices = np.arange(len(payload["target"])) if sampled is None else sampled[year]
            values.append(_metrics(payload["target"][indices], payload["scores"][structure][indices]))
        return np.mean(np.asarray(values), axis=0)

    point = {name: mean_metrics(name) for name in structures}
    rng = np.random.default_rng(config.seed + 7331)
    baselines = structures[:-1]
    boot = {name: np.empty((3, repetitions), dtype=float) for name in baselines}
    for repetition in range(repetitions):
        sampled = {
            year: np.concatenate(
                [rng.choice(indices, size=len(indices), replace=True) for indices in payload["strata"]]
            )
            for year, payload in years.items()
        }
        new_metrics = mean_metrics("partial_pool_slopes", sampled)
        for baseline_name in baselines:
            old_metrics = mean_metrics(baseline_name, sampled)
            boot[baseline_name][:, repetition] = (
                new_metrics[0] - old_metrics[0],
                new_metrics[1] - old_metrics[1],
                old_metrics[2] - new_metrics[2],
            )

    rows: list[dict] = []
    metric_names = ("pr_auc", "roc_auc", "brier_improvement")
    for baseline_name in baselines:
        point_delta = np.asarray(
            [
                point["partial_pool_slopes"][0] - point[baseline_name][0],
                point["partial_pool_slopes"][1] - point[baseline_name][1],
                point[baseline_name][2] - point["partial_pool_slopes"][2],
            ]
        )
        for index, metric in enumerate(metric_names):
            values = boot[baseline_name][index]
            rows.append(
                {
                    "new_structure": "partial_pool_slopes",
                    "baseline_structure": baseline_name,
                    "metric": metric,
                    "point_delta_positive_is_better": float(point_delta[index]),
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                    "bootstrap_probability_improvement": float(np.mean(values > 0)),
                    "bootstrap_two_sided_p": float(
                        min(1.0, 2 * min(np.mean(values <= 0), np.mean(values >= 0)))
                    ),
                    "bootstrap_repetitions": repetitions,
                }
            )

    tables_dir = ensure_work_subdir(args.work_dir, "tables")
    output = tables_dir / "partial_pooling_paired_bootstrap.csv"
    write_csv(output, pd.DataFrame(rows))
    print(f"Wrote {len(rows)} paired comparisons to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
