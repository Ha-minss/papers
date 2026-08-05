from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .core import research_pipeline as rp
from .core.config import load_experiment_config
from .core.experiment_utils import select_features, validate_target_year
from .core.io import ensure_work_subdir, load_prepared_frame, write_csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate OOT prediction files for bootstrap inference.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--model", choices=("logistic", "xgboost", "lightgbm"), required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "experiment.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_experiment_config(args.config)
    validate_target_year(args.year, config)
    if args.model not in config.models:
        raise ValueError(f"Model {args.model} is not enabled in {args.config}")
    frame = load_prepared_frame(args.data_file, args.work_dir)
    features = select_features(config.feature_set)
    train = frame[frame["eval_year"].lt(args.year)].reset_index(drop=True)
    test = frame[frame["eval_year"].eq(args.year)].reset_index(drop=True)

    pooled_oof, pooled_test = rp.pooled_predictions(
        train, test, features, "reference", args.model, "none", config.seed
    )
    sector_oof, sector_test = rp.sector_specific_oof_and_test(
        train, test, features, "reference", args.model, "none", config.seed
    )
    partial_oof, partial_test = rp.partial_pool_oof_and_test(
        train,
        test,
        features,
        "reference",
        args.model,
        "none",
        config.seed,
        config.partial_pooling.C,
    )

    rows: list[pd.DataFrame] = []
    for structure, test_scores, train_scores in (
        ("pooled", pooled_test, pooled_oof),
        ("sector_specific", sector_test, sector_oof),
        ("partial_pool", partial_test, partial_oof),
    ):
        output = test[["row_id", "sector", "eval_year", "target"]].copy()
        output["score"] = test_scores
        output["structure"] = structure
        output["model"] = args.model
        output["threshold_from_train"] = rp.choose_gmean_threshold(train["target"], train_scores)
        rows.append(output)

    predictions = pd.concat(rows, ignore_index=True)
    predictions_dir = ensure_work_subdir(args.work_dir, "predictions")
    path = predictions_dir / f"oot_{args.year}_{args.model}.csv.gz"
    write_csv(path, predictions, compression="gzip")
    print(f"Wrote {len(predictions):,} predictions to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
