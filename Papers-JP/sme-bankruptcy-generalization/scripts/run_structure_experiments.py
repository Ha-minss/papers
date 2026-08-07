from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .core import research_pipeline as rp
from .core.config import load_experiment_config
from .core.experiment_utils import (
    add_macro_metrics,
    merge_rows,
    sector_metrics,
    select_features,
    validate_target_year,
)
from .core.io import ensure_work_subdir, load_prepared_frame, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare pooled, sector-specific, and partial-pooling structures.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--evaluation", choices=("random", "oot"), required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--data-file", default=None)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "experiment.json"),
    )
    return parser


def _sector_matrix(scores, sectors) -> np.ndarray:
    base = logit(np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)).reshape(-1, 1)
    dummies = pd.get_dummies(
        pd.Categorical(sectors, categories=rp.SECTORS), dtype=float
    ).reindex(columns=list(rp.SECTORS), fill_value=0.0)
    return np.column_stack([base, dummies.to_numpy()])


def _crossfit_sector_calibration(y, base_scores, sectors, *, seed: int, C: float, folds: int):
    X = _sector_matrix(base_scores, sectors)
    y_array = np.asarray(y, dtype=int)
    n_splits = max(2, min(folds, int(np.bincount(y_array, minlength=2).min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    predictions = np.full(len(y_array), np.nan)
    for fold, (train_index, validation_index) in enumerate(cv.split(X, y_array)):
        model = LogisticRegression(
            C=C,
            solver="liblinear",
            max_iter=3000,
            random_state=seed + fold,
        )
        model.fit(X[train_index], y_array[train_index])
        predictions[validation_index] = model.predict_proba(X[validation_index])[:, 1]
    return predictions


def _fit_sector_calibrator(y, base_scores, sectors, test_scores, test_sectors, *, seed: int, C: float):
    model = LogisticRegression(C=C, solver="liblinear", max_iter=3000, random_state=seed)
    model.fit(_sector_matrix(base_scores, sectors), np.asarray(y, dtype=int))
    return model.predict_proba(_sector_matrix(test_scores, test_sectors))[:, 1]


def _random_predictions(data: pd.DataFrame, features: list[str], model_name: str, seed: int, folds: int, C: float):
    augmented = rp.add_sector_onehot(data, features)
    augmented_features = features + [f"sector_{sector}" for sector in rp.SECTORS]
    pooled = rp.cv_predictions(
        augmented,
        augmented_features,
        "reference",
        model_name,
        "none",
        n_splits=folds,
        seed=seed,
    )
    sector_specific = np.full(len(data), np.nan)
    for index, sector in enumerate(rp.SECTORS):
        mask = data["sector"].eq(sector).to_numpy()
        subset = data.loc[mask].reset_index(drop=True)
        sector_specific[np.where(mask)[0]] = rp.cv_predictions(
            subset,
            features,
            "reference",
            model_name,
            "none",
            n_splits=folds,
            seed=seed + 100 * index,
        )
    partial_pool = _crossfit_sector_calibration(
        data["target"], pooled, data["sector"], seed=seed, C=C, folds=folds
    )
    return {
        "pooled": pooled,
        "sector_specific": sector_specific,
        "partial_pool": partial_pool,
    }, None


def _oot_predictions(train: pd.DataFrame, test: pd.DataFrame, features: list[str], model_name: str, seed: int, folds: int, C: float):
    pooled_oof, pooled_test = rp.pooled_predictions(
        train, test, features, "reference", model_name, "none", seed
    )
    sector_oof, sector_test = rp.sector_specific_oof_and_test(
        train, test, features, "reference", model_name, "none", seed
    )
    partial_oof = _crossfit_sector_calibration(
        train["target"], pooled_oof, train["sector"], seed=seed, C=C, folds=folds
    )
    partial_test = _fit_sector_calibrator(
        train["target"],
        pooled_oof,
        train["sector"],
        pooled_test,
        test["sector"],
        seed=seed,
        C=C,
    )
    return (
        {
            "pooled": pooled_test,
            "sector_specific": sector_test,
            "partial_pool": partial_test,
        },
        {
            "pooled": pooled_oof,
            "sector_specific": sector_oof,
            "partial_pool": partial_oof,
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_experiment_config(args.config)
    validate_target_year(args.year, config)
    frame = load_prepared_frame(args.data_file, args.work_dir)
    features = select_features(config.feature_set)
    tables_dir = ensure_work_subdir(args.work_dir, "tables")
    predictions_dir = ensure_work_subdir(args.work_dir, "predictions")

    test = frame[frame["eval_year"].eq(args.year)].reset_index(drop=True)
    train = frame[frame["eval_year"].lt(args.year)].reset_index(drop=True)
    metrics_rows: list[dict] = []
    sector_rows: list[pd.DataFrame] = []
    prediction_rows: list[pd.DataFrame] = []

    for model_name in config.models:
        started = time.time()
        if args.evaluation == "random":
            test_scores, oof_scores = _random_predictions(
                test,
                features,
                model_name,
                config.seed,
                config.cv_folds,
                config.partial_pooling.C,
            )
            evaluation_label = "random_cv_same_year"
        else:
            test_scores, oof_scores = _oot_predictions(
                train,
                test,
                features,
                model_name,
                config.seed,
                config.cv_folds,
                config.partial_pooling.C,
            )
            evaluation_label = "rolling_oot"

        for structure, scores in test_scores.items():
            if args.evaluation == "random":
                threshold = rp.choose_gmean_threshold(test["target"], scores)
            else:
                assert oof_scores is not None
                threshold = rp.choose_gmean_threshold(train["target"], oof_scores[structure])

            metrics = rp.evaluate_scores(test["target"], scores, threshold)
            metrics.update(
                {
                    "stage": "structure",
                    "evaluation": evaluation_label,
                    "target_year": args.year,
                    "train_years": (
                        str(args.year)
                        if args.evaluation == "random"
                        else ",".join(str(value) for value in sorted(train["eval_year"].unique()))
                    ),
                    "feature_set": config.feature_set,
                    "feature_count": len(features),
                    "preprocess": "reference",
                    "model": model_name,
                    "imbalance": "none",
                    "structure": structure,
                    "elapsed_model_bundle_seconds": time.time() - started,
                }
            )
            if args.evaluation == "oot":
                metrics["validation_pr_auc"] = float(
                    average_precision_score(train["target"], oof_scores[structure])
                )
                metrics["validation_roc_auc"] = float(
                    roc_auc_score(train["target"], oof_scores[structure])
                )

            by_sector = sector_metrics(test, scores, threshold)
            metrics_rows.append(add_macro_metrics(metrics, by_sector))
            by_sector["evaluation"] = evaluation_label
            by_sector["target_year"] = args.year
            by_sector["model"] = model_name
            by_sector["structure"] = structure
            positive_counts = (
                train.groupby("sector")["target"].sum()
                if args.evaluation == "oot"
                else test.groupby("sector")["target"].sum()
            )
            by_sector["train_positive"] = by_sector["sector"].map(positive_counts)
            sector_rows.append(by_sector)

            prediction = test[["row_id", "sector", "eval_year", "target"]].copy()
            prediction["score"] = scores
            prediction["evaluation"] = evaluation_label
            prediction["model"] = model_name
            prediction["structure"] = structure
            prediction_rows.append(prediction)
            print(
                f"{args.year} {evaluation_label} {model_name} {structure}: "
                f"macro PR-AUC={metrics_rows[-1]['macro_pr_auc']:.4f}",
                flush=True,
            )

    merge_rows(
        tables_dir / "structure_metrics.csv",
        pd.DataFrame(metrics_rows),
        ("target_year", "evaluation", "model", "structure"),
    )
    merge_rows(
        tables_dir / "structure_sector_metrics.csv",
        pd.concat(sector_rows, ignore_index=True),
        ("target_year", "evaluation", "model", "structure", "sector"),
    )
    merge_rows(
        predictions_dir / "structure_predictions.csv.gz",
        pd.concat(prediction_rows, ignore_index=True),
        ("row_id", "evaluation", "model", "structure"),
        compression="gzip",
    )
    write_json(
        tables_dir / "structure_experiment_metadata.json",
        {
            "pooled": "one model across sectors with sector indicators",
            "sector_specific": "one independent model per sector",
            "partial_pool": "pooled score plus L2-regularized sector-intercept calibration",
            "primary_metrics": ["macro_pr_auc", "macro_brier", "pr_auc", "brier"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
