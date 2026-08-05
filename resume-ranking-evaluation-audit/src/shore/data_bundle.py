from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class BundlePaths:
    output_dir: Path
    users: Path
    jobs: Path
    interactions: Path
    train_pairs: Path
    valid_pairs: Path
    confit_valid: Path
    confit_test: Path
    pairwise_valid: Path
    pairwise_test: Path
    stage_valid: Path
    stage_test: Path

    @classmethod
    def from_output_dir(cls, output_dir: str | Path) -> "BundlePaths":
        root = Path(output_dir)
        return cls(
            output_dir=root,
            users=root / "users_clean.csv.gz",
            jobs=root / "jobs_clean.csv.gz",
            interactions=root / "interactions_dedup.csv.gz",
            train_pairs=root / "train_binary_pairs.csv.gz",
            valid_pairs=root / "valid_binary_pairs.csv.gz",
            confit_valid=root / "eval_confit_valid_100.csv.gz",
            confit_test=root / "eval_confit_test_100.csv.gz",
            pairwise_valid=root / "eval_matched_pairwise_valid_10seeds.csv.gz",
            pairwise_test=root / "eval_matched_pairwise_10seeds.csv.gz",
            stage_valid=root / "eval_stage_specific_valid_1plus2.csv.gz",
            stage_test=root / "eval_stage_specific_1plus2.csv.gz",
        )


@dataclass
class BundleTables:
    users: pd.DataFrame
    jobs: pd.DataFrame
    interactions: pd.DataFrame | None
    train_pairs: pd.DataFrame | None
    valid_pairs: pd.DataFrame | None
    confit_valid: pd.DataFrame
    confit_test: pd.DataFrame
    pairwise_valid: pd.DataFrame
    pairwise_test: pd.DataFrame
    stage_valid: pd.DataFrame
    stage_test: pd.DataFrame


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_bundle(bundle_zip: str | Path, extract_dir: str | Path, force: bool = False) -> BundlePaths:
    bundle_zip, extract_dir = Path(bundle_zip), Path(extract_dir)
    if not bundle_zip.exists():
        raise FileNotFoundError(bundle_zip)
    if force and extract_dir.exists():
        shutil.rmtree(extract_dir)
    output_dir = extract_dir / "output"
    if not output_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(bundle_zip) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise RuntimeError(f"ZIP CRC error: {bad}")
            archive.extractall(extract_dir)
    return BundlePaths.from_output_dir(output_dir)


def verify_manifest(extract_dir: str | Path) -> pd.DataFrame:
    extract_dir = Path(extract_dir)
    manifest_path = extract_dir / "MANIFEST.json"
    if not manifest_path.exists():
        return pd.DataFrame(columns=["file", "bytes", "sha256_ok", "rows_ok"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for item in manifest:
        path = extract_dir / "output" / item["file"]
        if not path.exists():
            raise FileNotFoundError(path)
        hash_ok = sha256_file(path) == item["sha256"]
        if not hash_ok:
            raise ValueError(f"SHA256 mismatch: {item['file']}")
        rows_ok = True
        expected_rows = item.get("rows")
        if expected_rows is not None:
            actual_rows = sum(len(c) for c in pd.read_csv(path, usecols=[0], chunksize=100_000))
            rows_ok = actual_rows == int(expected_rows)
            if not rows_ok:
                raise ValueError(f"row count mismatch: {item['file']} {actual_rows} != {expected_rows}")
        rows.append({"file": item["file"], "bytes": path.stat().st_size, "sha256_ok": hash_ok, "rows_ok": rows_ok})
    return pd.DataFrame(rows)


def _read(path: Path, *, required: bool = True) -> pd.DataFrame | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    dtype = {"user_id": "string", "jd_no": "string"}
    return pd.read_csv(path, dtype=dtype)


def load_bundle_tables(paths: BundlePaths, require_training: bool = True) -> BundleTables:
    return BundleTables(
        users=_read(paths.users),
        jobs=_read(paths.jobs),
        interactions=_read(paths.interactions, required=require_training),
        train_pairs=_read(paths.train_pairs, required=require_training),
        valid_pairs=_read(paths.valid_pairs, required=require_training),
        confit_valid=_read(paths.confit_valid),
        confit_test=_read(paths.confit_test),
        pairwise_valid=_read(paths.pairwise_valid),
        pairwise_test=_read(paths.pairwise_test),
        stage_valid=_read(paths.stage_valid),
        stage_test=_read(paths.stage_test),
    )


def _source_column(frame: pd.DataFrame) -> str:
    if "candidate_source" in frame.columns:
        return "candidate_source"
    if "negative_type" in frame.columns:
        return "negative_type"
    raise ValueError("evaluation frame needs candidate_source or negative_type")


def validate_eval_frames(conventional: pd.DataFrame, pairwise: pd.DataFrame, expected_conventional_size: int = 100) -> None:
    required = {"query_id", "jd_no", "user_id", "label"}
    for name, frame in [("conventional", conventional), ("pairwise", pairwise)]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
    if not conventional.groupby("query_id").size().eq(expected_conventional_size).all():
        raise ValueError("conventional query size mismatch")
    if not conventional.groupby("query_id")["label"].sum().ge(1).all():
        raise ValueError("conventional query without a positive")
    if not pairwise.groupby("query_id").size().eq(2).all():
        raise ValueError("pairwise query size mismatch")
    if not pairwise.groupby("query_id")["label"].sum().eq(1).all():
        raise ValueError("pairwise query must contain one positive")
    _source_column(pairwise)
