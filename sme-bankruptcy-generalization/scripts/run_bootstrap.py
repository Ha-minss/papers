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
    parser = argparse.ArgumentParser(description="Compute stratified bootstrap intervals from OOT predictions.")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "experiment.json"))
    parser.add_argument("--repetitions", type=int, default=None)
    return parser


def _recall_at_fraction(y: np.ndarray, scores: np.ndarray, fraction: float) -> float:
    k = max(1, int(np.ceil(len(y) * fraction)))
    top = np.argpartition(-scores, k - 1)[:k]
    return float(y[top].sum() / y.sum()) if y.sum() else np.nan


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_experiment_config(args.config)
    repetitions = args.repetitions or config.bootstrap_repetitions
    predictions_dir = Path(args.work_dir).expanduser().resolve() / "predictions"
    paths = sorted(predictions_dir.glob("oot_*_*.csv.gz"))
    if not paths:
        raise FileNotFoundError(
            f"No OOT prediction files found in {predictions_dir}. Run run_primary_predictions first."
        )

    rng = np.random.default_rng(config.seed + 2026)
    rows: list[dict] = []
    for path in paths:
        data = pd.read_csv(path)
        year = int(data["eval_year"].iloc[0])
        model = str(data["model"].iloc[0])
        for structure, group in data.groupby("structure", sort=True):
            y = group["target"].to_numpy(dtype=int)
            scores = group["score"].to_numpy(dtype=float)
            positives = np.flatnonzero(y == 1)
            negatives = np.flatnonzero(y == 0)
            if len(positives) == 0 or len(negatives) == 0:
                raise ValueError(f"Both classes are required for bootstrap: {path}")
            samples = {
                "pr_auc": [],
                "roc_auc": [],
                "brier": [],
                "recall_at_1pct": [],
                "recall_at_5pct": [],
            }
            for _ in range(repetitions):
                indices = np.concatenate(
                    [
                        rng.choice(positives, len(positives), replace=True),
                        rng.choice(negatives, len(negatives), replace=True),
                    ]
                )
                y_boot = y[indices]
                score_boot = scores[indices]
                samples["pr_auc"].append(average_precision_score(y_boot, score_boot))
                samples["roc_auc"].append(roc_auc_score(y_boot, score_boot))
                samples["brier"].append(brier_score_loss(y_boot, score_boot))
                samples["recall_at_1pct"].append(_recall_at_fraction(y_boot, score_boot, 0.01))
                samples["recall_at_5pct"].append(_recall_at_fraction(y_boot, score_boot, 0.05))

            point = {
                "pr_auc": average_precision_score(y, scores),
                "roc_auc": roc_auc_score(y, scores),
                "brier": brier_score_loss(y, scores),
                "recall_at_1pct": _recall_at_fraction(y, scores, 0.01),
                "recall_at_5pct": _recall_at_fraction(y, scores, 0.05),
            }
            for metric, values in samples.items():
                array = np.asarray(values, dtype=float)
                rows.append(
                    {
                        "target_year": year,
                        "model": model,
                        "structure": structure,
                        "metric": metric,
                        "point": float(point[metric]),
                        "ci_low": float(np.quantile(array, 0.025)),
                        "ci_high": float(np.quantile(array, 0.975)),
                        "bootstrap_repetitions": repetitions,
                    }
                )

    tables_dir = ensure_work_subdir(args.work_dir, "tables")
    output = tables_dir / "bootstrap_primary_ci.csv"
    write_csv(output, pd.DataFrame(rows))
    print(f"Wrote {len(rows)} bootstrap summaries to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
