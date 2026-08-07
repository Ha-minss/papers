from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .data import load_uplift_data, prepare_uplift_frame
from .pipeline import run_pipeline
from .provenance import file_sha256


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orange-uplift",
        description="Validate data or run the Orange Belgium churn-uplift evaluation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-data", help="Validate the input data contract.")
    validate.add_argument("--input", type=Path, required=True, help="Path to a CSV file.")
    validate.add_argument(
        "--near-constant-threshold", type=float, default=0.999,
        help="Maximum dominant-value share before a feature is excluded.",
    )

    run = subparsers.add_parser("run", help="Run the full analysis pipeline.")
    run.add_argument("--input", type=Path, required=True, help="Path to a CSV file.")
    run.add_argument("--config", type=Path, required=True, help="Path to an experiment YAML file.")
    run.add_argument("--output", type=Path, required=True, help="Directory for generated outputs.")
    run.add_argument("--overwrite", action="store_true", help="Replace a non-empty output directory.")
    return parser


def _validate_data(path: Path, threshold: float) -> dict[str, object]:
    frame = load_uplift_data(path)
    data = prepare_uplift_frame(frame, near_constant_threshold=threshold)
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "rows": len(data.frame),
        "columns": data.frame.shape[1],
        "churners": int(data.target.sum()),
        "treated": int(data.treatment.sum()),
        "features": len(data.feature_columns),
        "excluded_columns": data.excluded_columns,
        "status": "valid",
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "validate-data":
        result = _validate_data(args.input, args.near_constant_threshold)
    else:
        result = run_pipeline(
            args.input,
            args.config,
            args.output,
            overwrite=args.overwrite,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
