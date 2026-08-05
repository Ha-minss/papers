from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .core import research_pipeline as rp
from .core.config import load_experiment_config
from .core.experiment_utils import merge_rows, select_features, validate_target_year
from .core.io import ensure_work_subdir, load_prepared_frame, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
warnings.filterwarnings("ignore", message="X does not have valid feature names")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the controlled preprocessing and imbalance experiment.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--evaluation", choices=("random", "oot"), required=True)
    parser.add_argument("--work-dir", required=True, help="External directory for data and generated results.")
    parser.add_argument("--data-file", default=None, help="Optional prepared CSV.GZ path.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "experiment.json"),
        help="Experiment JSON file.",
    )
    return parser


def _resample(X, y, method: str, seed: int, target_ratio: float):
    y = np.asarray(y, dtype=int)
    if method in {"none", "class_weight"}:
        return X, y
    try:
        from imblearn.combine import SMOTEENN
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "imbalanced-learn is required for resampling experiments. "
            "Install dependencies with `python -m pip install -r requirements.txt`."
        ) from exc

    if method == "random_under":
        return RandomUnderSampler(random_state=seed).fit_resample(X, y)
    k_neighbors = rp.safe_smote_k(y)
    if method == "smote":
        sampler = SMOTE(
            sampling_strategy=target_ratio,
            random_state=seed,
            k_neighbors=k_neighbors,
        )
    elif method == "smoteenn":
        sampler = SMOTEENN(
            sampling_strategy=target_ratio,
            random_state=seed,
            smote=SMOTE(
                sampling_strategy=target_ratio,
                random_state=seed,
                k_neighbors=k_neighbors,
            ),
            enn=EditedNearestNeighbours(n_jobs=-1),
        )
    else:
        raise ValueError(f"Unknown imbalance method: {method}")
    return sampler.fit_resample(X, y)


def _sector_augmented(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    augmented = rp.add_sector_onehot(frame, features)
    return augmented, features + [f"sector_{sector}" for sector in rp.SECTORS]


def _fold_bundle(data: pd.DataFrame, features: list[str], preprocess: str, seed: int, folds: int):
    y = data["target"].to_numpy(dtype=int)
    n_splits = max(2, min(folds, int(np.bincount(y, minlength=2).min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    bundle = []
    for fold, (train_index, validation_index) in enumerate(cv.split(np.zeros(len(y)), y)):
        preprocessor = rp.make_preprocessor(preprocess, features)
        bundle.append(
            {
                "validation_index": validation_index,
                "X_train": preprocessor.fit_transform(data.iloc[train_index][features]),
                "X_validation": preprocessor.transform(data.iloc[validation_index][features]),
                "y_train": y[train_index],
                "y_validation": y[validation_index],
                "fold": fold,
            }
        )
    return bundle


def _oof_predictions(bundle, model_name: str, imbalance: str, seed: int, target_ratio: float, n_rows: int):
    predictions = np.full(n_rows, np.nan)
    fold_rows: list[dict] = []
    for item in bundle:
        X_fit, y_fit = _resample(
            item["X_train"], item["y_train"], imbalance, seed + item["fold"], target_ratio
        )
        model = rp.model_factory(model_name, y_fit, imbalance=imbalance, seed=seed + item["fold"])
        model.fit(X_fit, y_fit)
        scores = model.predict_proba(item["X_validation"])[:, 1]
        predictions[item["validation_index"]] = scores
        fold_rows.append(
            {
                "gmean_0_5": rp.gmean_at_threshold(item["y_validation"], scores, 0.5),
                "roc_auc": roc_auc_score(item["y_validation"], scores),
                "pr_auc": average_precision_score(item["y_validation"], scores),
            }
        )
    return predictions, pd.DataFrame(fold_rows)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_experiment_config(args.config)
    validate_target_year(args.year, config)
    frame = load_prepared_frame(args.data_file, args.work_dir)
    tables_dir = ensure_work_subdir(args.work_dir, "tables")
    predictions_dir = ensure_work_subdir(args.work_dir, "predictions")
    base_features = select_features(config.feature_set)

    if args.evaluation == "random":
        evaluation_data = frame[frame["eval_year"].eq(args.year)].reset_index(drop=True)
        train_data = evaluation_data
        test_data = evaluation_data
    else:
        train_data = frame[frame["eval_year"].lt(args.year)].reset_index(drop=True)
        test_data = frame[frame["eval_year"].eq(args.year)].reset_index(drop=True)
        evaluation_data = train_data

    rows: list[dict] = []
    selected_predictions: list[pd.DataFrame] = []
    configurations: list[tuple[list[str], str, str, str]] = []
    for preprocess in config.preprocessing_modes:
        imbalances = config.imbalance_methods if preprocess == "reference" else ("none",)
        for model in config.models:
            for imbalance in imbalances:
                configurations.append((base_features, preprocess, model, imbalance))

    for features, preprocess, model_name, imbalance in configurations:
        started = time.time()
        train_augmented, augmented_features = _sector_augmented(train_data, features)
        fold_bundle = _fold_bundle(
            train_augmented, augmented_features, preprocess, config.seed, config.cv_folds
        )
        oof, fold_metrics = _oof_predictions(
            fold_bundle,
            model_name,
            imbalance,
            config.seed,
            config.smote_sampling_strategy,
            len(train_augmented),
        )
        threshold = rp.choose_gmean_threshold(train_augmented["target"], oof)

        if args.evaluation == "random":
            scores = oof
            scored_data = train_augmented
            evaluation_label = "random_cv_same_year"
        else:
            test_augmented, _ = _sector_augmented(test_data, features)
            preprocessor = rp.make_preprocessor(preprocess, augmented_features)
            X_train = preprocessor.fit_transform(train_augmented[augmented_features])
            X_test = preprocessor.transform(test_augmented[augmented_features])
            X_fit, y_fit = _resample(
                X_train,
                train_augmented["target"].to_numpy(dtype=int),
                imbalance,
                config.seed,
                config.smote_sampling_strategy,
            )
            fitted = rp.model_factory(model_name, y_fit, imbalance=imbalance, seed=config.seed)
            fitted.fit(X_fit, y_fit)
            scores = fitted.predict_proba(X_test)[:, 1]
            scored_data = test_augmented
            evaluation_label = "rolling_oot"

        metrics = rp.evaluate_scores(scored_data["target"], scores, threshold)
        metrics.update(
            {
                "stage": "factorial",
                "evaluation": evaluation_label,
                "target_year": args.year,
                "train_years": (
                    str(args.year)
                    if args.evaluation == "random"
                    else ",".join(str(value) for value in sorted(train_data["eval_year"].unique()))
                ),
                "feature_set": config.feature_set,
                "feature_count": len(features),
                "preprocess": preprocess,
                "model": model_name,
                "imbalance": imbalance,
                "structure": "pooled",
                "validation_pr_auc": float(average_precision_score(train_augmented["target"], oof)),
                "validation_roc_auc": float(roc_auc_score(train_augmented["target"], oof)),
                "mean_fold_pr_auc": float(fold_metrics["pr_auc"].mean()),
                "mean_fold_roc_auc": float(fold_metrics["roc_auc"].mean()),
                "mean_fold_gmean_0_5": float(fold_metrics["gmean_0_5"].mean()),
                "elapsed_seconds": time.time() - started,
            }
        )
        rows.append(metrics)

        if args.evaluation == "oot" and (
            (preprocess == "reference" and imbalance in {"none", "smote", "smoteenn"})
            or (preprocess == "practical" and imbalance == "none")
        ):
            prediction = scored_data[["row_id", "sector", "eval_year", "target"]].copy()
            prediction["score"] = scores
            prediction["evaluation"] = evaluation_label
            prediction["feature_set"] = config.feature_set
            prediction["preprocess"] = preprocess
            prediction["model"] = model_name
            prediction["imbalance"] = imbalance
            selected_predictions.append(prediction)

        print(
            f"{args.year} {evaluation_label} {preprocess} {model_name} {imbalance}: "
            f"PR-AUC={metrics['pr_auc']:.4f} Brier={metrics['brier']:.6f}",
            flush=True,
        )

    metrics_frame = pd.DataFrame(rows)
    merge_rows(
        tables_dir / "factorial_metrics.csv",
        metrics_frame,
        ("target_year", "evaluation", "feature_set", "preprocess", "model", "imbalance"),
    )
    if selected_predictions:
        merge_rows(
            predictions_dir / "factorial_predictions.csv.gz",
            pd.concat(selected_predictions, ignore_index=True),
            ("row_id", "evaluation", "feature_set", "preprocess", "model", "imbalance"),
            compression="gzip",
        )
    write_json(
        tables_dir / "factorial_experiment_metadata.json",
        {
            "random_cv": "stratified folds within the target year",
            "rolling_oot": "all earlier evaluation years train; target year test",
            "smote_sampling_strategy": config.smote_sampling_strategy,
            "threshold_policy": "selected on training out-of-fold predictions",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
