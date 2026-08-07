from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .core import research_pipeline as rp
from .core.config import load_experiment_config
from .core.experiment_utils import add_macro_metrics, sector_metrics, select_features, validate_target_year
from .core.hierarchical_logistic import shrinkage_interaction_oof_and_test
from .core.io import ensure_work_subdir, load_prepared_frame, write_csv, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit ridge-shrunken sector slope deviations on rolling OOT splits."
    )
    parser.add_argument("--year", type=int, action="append", default=None, help="Repeat to run selected years.")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "experiment.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_experiment_config(args.config)
    years = tuple(args.year) if args.year else config.evaluation_years
    for year in years:
        validate_target_year(year, config)

    frame = load_prepared_frame(args.data_file, args.work_dir)
    features = select_features(config.feature_set)
    metric_rows: list[dict] = []
    sector_rows: list[pd.DataFrame] = []
    tuning_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []

    for year in years:
        started = time.time()
        train = frame[frame["eval_year"].lt(year)].reset_index(drop=True)
        test = frame[frame["eval_year"].eq(year)].reset_index(drop=True)
        oof, future, metadata = shrinkage_interaction_oof_and_test(
            train,
            test,
            features,
            preprocess_mode="reference",
            seed=config.seed,
            C_grid=config.partial_pooling.C_grid,
            interaction_scale_grid=config.partial_pooling.interaction_scale_grid,
            n_splits=config.partial_pooling.cv_folds,
        )
        threshold = rp.choose_gmean_threshold(train["target"], oof)
        metrics = rp.evaluate_scores(test["target"], future, threshold)
        metrics.update(
            {
                "evaluation": "rolling_oot",
                "target_year": year,
                "train_years": ",".join(str(value) for value in sorted(train["eval_year"].unique())),
                "feature_set": config.feature_set,
                "feature_count": len(features),
                "preprocess": "reference",
                "model": "logistic",
                "imbalance": "none",
                "structure": "partial_pool_slopes",
                "selected_C": metadata["selected_C"],
                "selected_interaction_scale": metadata["selected_interaction_scale"],
                "validation_pr_auc": float(average_precision_score(train["target"], oof)),
                "validation_roc_auc": float(roc_auc_score(train["target"], oof)),
                "validation_brier": float(brier_score_loss(train["target"], oof)),
                "elapsed_seconds": time.time() - started,
            }
        )
        by_sector = sector_metrics(test, future, threshold)
        metric_rows.append(add_macro_metrics(metrics, by_sector))
        by_sector["target_year"] = year
        by_sector["evaluation"] = "rolling_oot"
        by_sector["model"] = "logistic"
        by_sector["structure"] = "partial_pool_slopes"
        by_sector["train_positive"] = by_sector["sector"].map(train.groupby("sector")["target"].sum())
        sector_rows.append(by_sector)

        for row in metadata["tuning_results"]:
            tuning_rows.append(
                {
                    "target_year": year,
                    "train_years": metrics["train_years"],
                    "selected": bool(
                        row["C"] == metadata["selected_C"]
                        and row["interaction_scale"] == metadata["selected_interaction_scale"]
                    ),
                    **row,
                }
            )

        prediction = test[["row_id", "sector", "eval_year", "target"]].copy()
        prediction["score"] = future
        prediction["model"] = "logistic"
        prediction["structure"] = "partial_pool_slopes"
        prediction["evaluation"] = "rolling_oot"
        prediction_rows.append(prediction)
        print(
            f"{year}: scale={metadata['selected_interaction_scale']:.2f} "
            f"PR-AUC={metrics['pr_auc']:.4f} Brier={metrics['brier']:.6f}",
            flush=True,
        )

    tables_dir = ensure_work_subdir(args.work_dir, "tables")
    predictions_dir = ensure_work_subdir(args.work_dir, "predictions")
    write_csv(tables_dir / "partial_pooling_metrics.csv", pd.DataFrame(metric_rows))
    write_csv(tables_dir / "partial_pooling_sector_metrics.csv", pd.concat(sector_rows, ignore_index=True))
    write_csv(tables_dir / "partial_pooling_tuning.csv", pd.DataFrame(tuning_rows))
    write_csv(
        predictions_dir / "partial_pooling_predictions.csv.gz",
        pd.concat(prediction_rows, ignore_index=True),
        compression="gzip",
    )
    write_json(
        tables_dir / "partial_pooling_metadata.json",
        {
            "method": "ridge shrinkage-interaction logistic",
            "design": "shared slopes, effect-coded sector intercepts, and shrunken sector slope deviations",
            "selection": "training-only OOF PR-AUC with Brier tie-break",
            "C_grid": list(config.partial_pooling.C_grid),
            "interaction_scale_grid": list(config.partial_pooling.interaction_scale_grid),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
