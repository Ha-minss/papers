from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .data import build_dataset, load_telco_data
from .pipeline import run_pipeline
from .provenance import file_sha256


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ibm-telco-value",
        description="Validate data or run the IBM Telco value-aware prioritization pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-data", help="Validate the input data contract.")
    validate.add_argument("--input", type=Path, required=True, help="Path to a CSV or XLSX file.")

    run = subparsers.add_parser("run", help="Run the full analysis pipeline.")
    run.add_argument("--input", type=Path, required=True, help="Path to a CSV or XLSX file.")
    run.add_argument("--config", type=Path, required=True, help="Path to an experiment YAML file.")
    run.add_argument("--output", type=Path, required=True, help="Directory for generated outputs.")
    run.add_argument("--overwrite", action="store_true", help="Replace a non-empty output directory.")
    return parser


def _validate_data(path: Path) -> dict[str, object]:
    frame = load_telco_data(path)
    dataset = build_dataset(frame)
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "rows": len(dataset.frame),
        "columns": dataset.frame.shape[1],
        "churners": int(dataset.target.sum()),
        "churn_rate": float(dataset.target.mean()),
        "model_features": dataset.features.shape[1],
        "status": "valid",
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "validate-data":
        result = _validate_data(args.input)
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
