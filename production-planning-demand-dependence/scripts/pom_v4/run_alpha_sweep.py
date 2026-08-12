from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from ..baseline.ovd_study import InstanceData, load_all

from .controlled_lp_v4 import paired_run_v4
from .data_audit import EXPECTED_XLSX_SHA256
from .run_structural_ablation import REQUIRED_COLUMNS
from .structural_architecture import ArchitectureSpec, calibration_upstream_weights


INTERMEDIATE_ALPHAS = (0.25, 0.50, 0.75)
ALPHA_COLUMNS = REQUIRED_COLUMNS


@dataclass(frozen=True, order=True)
class AlphaTask:
    alpha: float
    seed: int


def intermediate_tasks(seeds: Iterable[int] = range(1, 31)) -> list[AlphaTask]:
    selected = tuple(int(seed) for seed in seeds)
    return [AlphaTask(alpha, seed) for alpha in INTERMEDIATE_ALPHAS for seed in selected]


def _validate_unique_successful(
    frame: pd.DataFrame,
    keys: list[str],
    expected_rows: int,
    label: str,
) -> None:
    if len(frame) != expected_rows:
        raise ValueError(f"{label} must contain {expected_rows} rows, found {len(frame)}")
    if "status" not in frame or not frame["status"].eq("ok").all():
        raise ValueError(f"{label} contains unsuccessful rows")
    if frame.duplicated(keys).any():
        raise ValueError(f"{label} contains duplicate keys")


def combine_with_endpoints(
    stage1: pd.DataFrame,
    intermediate: pd.DataFrame,
    expected_seeds: set[int] | None = None,
) -> pd.DataFrame:
    seeds = expected_seeds or set(range(1, 31))
    endpoints = stage1.loc[stage1["architecture"].isin(["dedicated", "baseline"])].copy()
    _validate_unique_successful(endpoints, ["architecture", "seed"], 2 * len(seeds), "endpoints")
    if set(endpoints["seed"].astype(int)) != seeds:
        raise ValueError("endpoint seeds do not match the frozen seed set")
    endpoints["alpha"] = endpoints["architecture"].map({"dedicated": 0.0, "baseline": 1.0})
    endpoints["architecture"] = "pooling"

    middle = intermediate.copy()
    _validate_unique_successful(middle, ["alpha", "seed"], 3 * len(seeds), "intermediate rows")
    if set(middle["seed"].astype(int)) != seeds:
        raise ValueError("intermediate seeds do not match the frozen seed set")
    if set(middle["alpha"].astype(float)) != set(INTERMEDIATE_ALPHAS):
        raise ValueError("intermediate alpha grid differs from the preregistered grid")
    middle["architecture"] = "pooling"

    combined = pd.concat([endpoints, middle], ignore_index=True, sort=False)
    combined = combined.sort_values(["alpha", "seed"]).reset_index(drop=True)
    _validate_unique_successful(combined, ["alpha", "seed"], 5 * len(seeds), "alpha panel")
    return combined


def _checkpoint_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid checkpoint line {line_number}: {exc}") from exc
    return rows


def pending_tasks(tasks: list[AlphaTask], checkpoint: Path) -> list[AlphaTask]:
    completed = {
        AlphaTask(float(row["alpha"]), int(row["seed"]))
        for row in _checkpoint_rows(checkpoint)
        if row.get("status") == "ok"
    }
    return [task for task in tasks if task not in completed]


def _append_checkpoint(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _rebuild_csv(checkpoint: Path, output: Path) -> None:
    latest = {
        AlphaTask(float(row["alpha"]), int(row["seed"])): row
        for row in _checkpoint_rows(checkpoint)
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ALPHA_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for key in sorted(latest):
            writer.writerow({column: latest[key].get(column, "") for column in ALPHA_COLUMNS})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)


def run_intermediate_tasks(
    inst,
    tasks: list[AlphaTask],
    checkpoint: Path,
    output_csv: Path,
    S: int = 20,
    H: int = 4,
    calibration: int = 20,
) -> list[dict]:
    weights = calibration_upstream_weights(inst, calibration=calibration)
    for task in pending_tasks(tasks, checkpoint):
        started = time.monotonic()
        row = {
            "data_sha256": EXPECTED_XLSX_SHA256,
            "architecture": "pooling",
            "seed": task.seed,
            "S": S,
            "H": H,
            "calibration": calibration,
            "upstream_multiplier": 1.0,
            "alpha": task.alpha,
            "status": "failed",
            "error": "",
        }
        try:
            result = paired_run_v4(
                inst,
                task.seed,
                ArchitectureSpec.pooling(task.alpha),
                weights,
                S=S,
                H=H,
                calibration=calibration,
            )
            row.update(result)
            row["joint_max_oper008_util"] = row.pop("joint_max_upstream_pool_util")
            row["ind_max_oper008_util"] = row.pop("ind_max_upstream_pool_util")
            row["status"] = "ok"
        except Exception as exc:  # preserve every failed task for audit and resume
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["elapsed_sec"] = time.monotonic() - started
        _append_checkpoint(checkpoint, row)
        _rebuild_csv(checkpoint, output_csv)
    return _checkpoint_rows(checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the preregistered Stage 2 alpha sweep")
    parser.add_argument("--stage1", type=Path, default=Path("../ijpe_local/results/stage1/stage1_raw.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("../ijpe_local/results/stage2"))
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--S", type=int, default=20)
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--calibration", type=int, default=20)
    args = parser.parse_args()

    inst = InstanceData(load_all(), "SET4")
    seeds = set(range(1, args.seeds + 1))
    checkpoint = args.results_dir / "stage2_checkpoint.jsonl"
    intermediate_csv = args.results_dir / "stage2_intermediate_raw.csv"
    run_intermediate_tasks(
        inst,
        intermediate_tasks(sorted(seeds)),
        checkpoint,
        intermediate_csv,
        S=args.S,
        H=args.H,
        calibration=args.calibration,
    )
    panel = combine_with_endpoints(
        pd.read_csv(args.stage1),
        pd.read_csv(intermediate_csv),
        expected_seeds=seeds,
    )
    panel.to_csv(args.results_dir / "stage2_alpha_panel.csv", index=False)


if __name__ == "__main__":
    main()
