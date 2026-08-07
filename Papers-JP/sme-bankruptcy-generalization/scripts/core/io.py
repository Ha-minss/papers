from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DATA_FILENAME = "slovak_sme_all_rows_with_flags.csv.gz"


def resolve_data_file(data_file: str | Path | None, work_dir: str | Path) -> Path:
    if data_file is not None:
        path = Path(data_file).expanduser().resolve()
    else:
        path = Path(work_dir).expanduser().resolve() / "processed" / DEFAULT_DATA_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Prepared data not found: {path}. Run `python -m scripts.prepare_data` first."
        )
    return path


def load_prepared_frame(data_file: str | Path | None, work_dir: str | Path) -> pd.DataFrame:
    path = resolve_data_file(data_file, work_dir)
    frame = pd.read_csv(path)
    required = {"row_id", "target", "sector", "eval_year"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Prepared data is missing required columns: {missing}")
    return frame


def ensure_work_subdir(work_dir: str | Path, name: str) -> Path:
    path = Path(work_dir).expanduser().resolve() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)


def write_csv(path: str | Path, frame: pd.DataFrame, *, compression: str | None = None) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    frame.to_csv(temporary, index=False, compression=compression)
    temporary.replace(output)
