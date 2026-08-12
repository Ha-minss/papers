from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

from ..baseline.ovd_study import InstanceData, load_all

from .controlled_lp_v4 import paired_run_v4, run_policy_v4
from .data_audit import EXPECTED_XLSX_SHA256
from .structural_architecture import ArchitectureSpec, calibration_upstream_weights


ARCHITECTURES = ("baseline", "nonbinding", "dedicated", "no_bom")
REQUIRED_COLUMNS = [
    "data_sha256",
    "architecture",
    "seed",
    "S",
    "H",
    "calibration",
    "upstream_multiplier",
    "alpha",
    "joint_cost",
    "ind_cost",
    "ovd_pct",
    "joint_holding",
    "ind_holding",
    "joint_backorder",
    "ind_backorder",
    "joint_max_oper004_util",
    "ind_max_oper004_util",
    "joint_max_oper008_util",
    "ind_max_oper008_util",
    "joint_upstream_quantity",
    "ind_upstream_quantity",
    "mean_action_l1",
    "elapsed_sec",
    "status",
    "error",
]


@dataclass(frozen=True, order=True)
class TaskKey:
    architecture: str
    seed: int


def fixed_tasks(seeds: Iterable[int] = range(1, 31)) -> list[TaskKey]:
    selected = tuple(int(seed) for seed in seeds)
    return [TaskKey(architecture, seed) for architecture in ARCHITECTURES for seed in selected]


def _checkpoint_rows(checkpoint: Path) -> list[dict]:
    if not checkpoint.exists():
        return []
    rows = []
    for line_number, line in enumerate(
        checkpoint.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid checkpoint line {line_number}: {exc}") from exc
    return rows


def pending_tasks(
    tasks: list[TaskKey],
    checkpoint: Path,
    accept_failures: bool = False,
) -> list[TaskKey]:
    completed = set()
    for row in _checkpoint_rows(checkpoint):
        status = row.get("status")
        if status == "ok" or (accept_failures and status == "failed"):
            completed.add(TaskKey(row["architecture"], int(row["seed"])))
    return [task for task in tasks if task not in completed]


def select_nonbinding_multiplier(
    max_utilization: Callable[[float], float],
    start: float = 10.0,
    threshold: float = 0.90,
    max_multiplier: float = 10_240.0,
) -> float:
    multiplier = float(start)
    while True:
        utilization = float(max_utilization(multiplier))
        if utilization <= threshold:
            return multiplier
        if multiplier >= max_multiplier:
            raise RuntimeError(
                f"nonbinding preflight did not reach {threshold:.3f} by {multiplier:g}x"
            )
        multiplier = min(multiplier * 2.0, max_multiplier)


def architecture_spec(label: str, nonbinding_multiplier: float) -> ArchitectureSpec:
    if label == "baseline":
        return ArchitectureSpec.baseline()
    if label == "nonbinding":
        return ArchitectureSpec.nonbinding(nonbinding_multiplier)
    if label == "dedicated":
        return ArchitectureSpec.pooling(0.0)
    if label == "no_bom":
        return ArchitectureSpec.no_bom()
    raise ValueError(f"unknown architecture: {label}")


def _append_checkpoint(checkpoint: Path, row: Mapping) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def rebuild_csv(checkpoint: Path, output_csv: Path) -> None:
    rows = _checkpoint_rows(checkpoint)
    latest = {
        TaskKey(row["architecture"], int(row["seed"])): row
        for row in rows
    }
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for key in sorted(latest):
            writer.writerow({column: latest[key].get(column, "") for column in REQUIRED_COLUMNS})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_csv)


def run_nonbinding_preflight(
    inst,
    weights: Mapping[str, float],
    output_path: Path,
    seeds: Iterable[int] = range(1, 31),
    S: int = 20,
    H: int = 4,
    calibration: int = 20,
) -> float:
    evidence: list[dict[str, float]] = []

    def maximum_utilization(multiplier: float) -> float:
        maximum = 0.0
        spec = ArchitectureSpec.nonbinding(multiplier)
        for seed in seeds:
            for treatment in ("joint", "independent"):
                result = run_policy_v4(
                    inst,
                    treatment,
                    int(seed),
                    spec,
                    weights,
                    S=S,
                    H=H,
                    calibration=calibration,
                )
                maximum = max(maximum, float(result["max_upstream_pool_util"]))
        evidence.append({"multiplier": multiplier, "max_oper008_utilization": maximum})
        return maximum

    selected = select_nonbinding_multiplier(maximum_utilization)
    payload = {
        "selection_rule": "first doubled multiplier with max OPER008 utilization <= 0.90",
        "ovd_inspected": False,
        "selected_multiplier": selected,
        "evidence": evidence,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    return selected


def run_tasks(
    inst,
    tasks: list[TaskKey],
    checkpoint: Path,
    output_csv: Path,
    nonbinding_multiplier: float,
    S: int = 20,
    H: int = 4,
    calibration: int = 20,
    accept_failures: bool = False,
) -> list[dict]:
    weights = calibration_upstream_weights(inst, calibration=calibration)
    for task in pending_tasks(tasks, checkpoint, accept_failures=accept_failures):
        started = time.monotonic()
        row = {
            "data_sha256": EXPECTED_XLSX_SHA256,
            "architecture": task.architecture,
            "seed": task.seed,
            "S": S,
            "H": H,
            "calibration": calibration,
            "status": "failed",
            "error": "",
        }
        try:
            spec = architecture_spec(task.architecture, nonbinding_multiplier)
            result = paired_run_v4(
                inst,
                task.seed,
                spec,
                weights,
                S=S,
                H=H,
                calibration=calibration,
            )
            row.update(result)
            row["architecture"] = task.architecture
            row["joint_max_oper008_util"] = row.pop("joint_max_upstream_pool_util")
            row["ind_max_oper008_util"] = row.pop("ind_max_upstream_pool_util")
            row["status"] = "ok"
        except Exception as exc:  # task-level isolation is intentional
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["elapsed_sec"] = time.monotonic() - started
        _append_checkpoint(checkpoint, row)
        rebuild_csv(checkpoint, output_csv)
    return _checkpoint_rows(checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preregistered POM v4 Stage 1 ablations")
    parser.add_argument("--results-dir", type=Path, default=Path("../ijpe_local/results/stage1"))
    parser.add_argument("--accept-failures", action="store_true")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--S", type=int, default=20)
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--calibration", type=int, default=20)
    args = parser.parse_args()

    inst = InstanceData(load_all(), "SET4")
    weights = calibration_upstream_weights(inst, calibration=args.calibration)
    preflight_path = args.results_dir / "nonbinding_preflight.json"
    if preflight_path.exists():
        multiplier = float(json.loads(preflight_path.read_text())["selected_multiplier"])
    else:
        multiplier = run_nonbinding_preflight(
            inst,
            weights,
            preflight_path,
            range(1, args.seeds + 1),
            S=args.S,
            H=args.H,
            calibration=args.calibration,
        )
    run_tasks(
        inst,
        fixed_tasks(range(1, args.seeds + 1)),
        args.results_dir / "stage1_checkpoint.jsonl",
        args.results_dir / "stage1_raw.csv",
        multiplier,
        S=args.S,
        H=args.H,
        calibration=args.calibration,
        accept_failures=args.accept_failures,
    )


if __name__ == "__main__":
    main()
