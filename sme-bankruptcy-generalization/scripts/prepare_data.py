from __future__ import annotations

import argparse
from pathlib import Path

from .core.config import load_data_schema
from .core.io import ensure_work_subdir, write_csv, write_json
from .core.research_pipeline import all_feature_columns, feature_dictionary, load_raw_csvs

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the 32 raw Slovak SME CSV files and build one analysis table."
    )
    parser.add_argument("--raw-dir", required=True, help="External directory containing the raw CSV files.")
    parser.add_argument("--work-dir", required=True, help="External directory for generated local artifacts.")
    parser.add_argument(
        "--schema",
        default=str(PROJECT_ROOT / "config" / "data_schema.json"),
        help="Dataset schema JSON file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    schema = load_data_schema(args.schema)
    frame = load_raw_csvs(
        args.raw_dir,
        expected_csv_files=schema.expected_csv_files,
        expected_columns=schema.expected_columns,
        delimiter=schema.delimiter,
        decimal=schema.decimal,
        missing_values=schema.missing_values,
    )
    if tuple(sorted(frame["sector"].unique())) != tuple(sorted(schema.sectors)):
        raise ValueError("Observed sectors do not match config/data_schema.json")

    processed = ensure_work_subdir(args.work_dir, "processed")
    tables = ensure_work_subdir(args.work_dir, "tables")
    data_path = processed / "slovak_sme_all_rows_with_flags.csv.gz"
    write_csv(data_path, frame, compression="gzip")
    write_csv(tables / "feature_dictionary.csv", feature_dictionary())

    summary = {
        "rows": int(len(frame)),
        "positive_cases": int(frame["target"].sum()),
        "event_rate": float(frame["target"].mean()),
        "feature_columns": len(all_feature_columns()),
        "years": sorted(int(value) for value in frame["eval_year"].unique()),
        "sectors": sorted(str(value) for value in frame["sector"].unique()),
        "all_missing_rows": int(frame["is_all_missing"].sum()),
        "duplicate_rows": int(frame["is_exact_duplicate"].sum()),
        "label_conflicts": int(frame["has_label_conflict"].sum()),
    }
    write_json(tables / "data_quality_summary.json", summary)
    print(f"Prepared {len(frame):,} rows at {data_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
