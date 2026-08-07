"""Prepare cleaned tables and text embeddings for the paper pipeline."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import re
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

csv.field_size_limit(min(sys.maxsize, 2_147_483_647))

EXPECTED_JOB_COLUMNS = [
    "JobID", "WindowID", "Title", "Description", "Requirements",
    "City", "State", "Country", "Zip5", "StartDate", "EndDate",
]
TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class RunConfig:
    data_dir: str
    output_dir: str
    gte_model: str = "Alibaba-NLP/gte-modernbert-base"
    jobbert_model: str = "TechWolf/JobBERT-v2"
    gte_max_length: int = 256
    jobbert_max_length: int = 64
    job_chunk_size: int = 50_000
    user_chunk_size: int = 25_000
    gte_batch_size: int = 64
    jobbert_batch_size: int = 128
    output_dtype: str = "float16"


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value)
    text = text.replace("\\r", " ").replace("\\n", " ").replace("\xa0", " ")
    text = html.unescape(text)
    text = TAG_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def parse_iso_timestamp(value: str) -> int:
    dt = datetime.fromisoformat(value.strip())
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def required_input_paths(data_dir: Path) -> list[Path]:
    paths = [
        data_dir / "apps.tsv",
        data_dir / "users.tsv",
        data_dir / "user_history.tsv",
        data_dir / "test_users.tsv",
        data_dir / "window_dates.tsv",
    ]
    archives = sorted(data_dir.glob("jobs_part*.zip"))
    if not archives:
        archives = [data_dir / "jobs_part1.zip"]
    return paths + archives


def validate_inputs(data_dir: Path) -> dict[str, object]:
    missing = [str(p) for p in required_input_paths(data_dir) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))
    report = {
        "data_dir": str(data_dir),
        "files": {
            p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in required_input_paths(data_dir)
        },
    }
    return report


def _job_archives(data_dir: Path) -> list[Path]:
    archives = sorted(data_dir.glob("jobs_part*.zip"))
    if not archives:
        raise FileNotFoundError(f"No staged job archives found in {data_dir}")
    return archives


def iter_valid_job_records(data_dir: Path) -> Iterator[dict[str, object]]:
    expected_header = "\t".join(EXPECTED_JOB_COLUMNS).encode("utf-8")
    for archive in _job_archives(data_dir):
        with zipfile.ZipFile(archive) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            if len(names) != 1:
                raise ValueError(f"Expected exactly one TSV in {archive.name}; found {names}")
            with zf.open(names[0], "r") as raw:
                header = raw.readline().rstrip(b"\r\n")
                if header != expected_header:
                    raise ValueError(f"Unexpected header in {archive.name}: {header[:200]!r}")
                for line_no, line in enumerate(raw, start=2):
                    columns = line.rstrip(b"\r\n").split(b"\t")
                    if len(columns) != 11:
                        continue
                    row = [c.decode("utf-8", errors="replace") for c in columns]
                    try:
                        job_id = int(row[0])
                        window_id = int(row[1])
                        start_ts = parse_iso_timestamp(row[9])
                        end_ts = parse_iso_timestamp(row[10])
                    except (TypeError, ValueError):
                        continue
                    title = clean_text(row[2])
                    requirements = clean_text(row[4])
                    yield {
                        "job_id": job_id,
                        "window_id": window_id,
                        "title": title,
                        "requirements": requirements,
                        "job_text": clean_text(f"Job title: {title}. Requirements: {requirements}"),
                        "city": clean_text(row[5]),
                        "state": clean_text(row[6]),
                        "country": clean_text(row[7]),
                        "start_ts": start_ts,
                        "end_ts": end_ts,
                        "source_archive": archive.name,
                        "source_line": line_no,
                    }


def prepare_clean_jobs(data_dir: Path, output_dir: Path, chunk_size: int = 50_000) -> dict[str, object]:
    parquet_dir = output_dir / "prepared" / "jobs"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(parquet_dir.glob("jobs_*.parquet"))
    manifest_path = parquet_dir / "manifest.json"
    if existing and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "complete":
            print(f"[resume] Clean jobs already complete: {manifest['rows']:,} rows")
            return manifest

    for stale in parquet_dir.glob("jobs_*.parquet"):
        stale.unlink()

    records: list[dict[str, object]] = []
    rows = 0
    shard = 0
    job_ids_seen: set[int] = set()
    duplicates = 0
    started = time.time()

    def flush() -> None:
        nonlocal records, shard
        if not records:
            return
        frame = pd.DataFrame.from_records(records)
        path = parquet_dir / f"jobs_{shard:05d}.parquet"
        frame.to_parquet(path, index=False, compression="zstd")
        print(f"  saved {path.name}: {len(frame):,} rows")
        records = []
        shard += 1

    for rec in iter_valid_job_records(data_dir):
        jid = int(rec["job_id"])
        if jid in job_ids_seen:
            duplicates += 1
            continue
        job_ids_seen.add(jid)
        records.append(rec)
        rows += 1
        if len(records) >= chunk_size:
            flush()
    flush()

    manifest = {
        "status": "complete",
        "rows": rows,
        "duplicate_job_ids_skipped": duplicates,
        "shards": shard,
        "columns": list(pd.read_parquet(parquet_dir / "jobs_00000.parquet", engine="pyarrow").columns),
        "seconds": round(time.time() - started, 2),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_user_documents(data_dir: Path, output_dir: Path, max_recent_titles: int = 20) -> dict[str, object]:
    prepared_dir = output_dir / "prepared" / "users"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    output_path = prepared_dir / "user_documents.parquet"
    manifest_path = prepared_dir / "manifest.json"
    if output_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "complete":
            print(f"[resume] User documents already complete: {manifest['rows']:,} rows")
            return manifest

    users = pd.read_csv(data_dir / "users.tsv", sep="\t", low_memory=False)
    history = pd.read_csv(
        data_dir / "user_history.tsv",
        sep="\t",
        usecols=["UserID", "Sequence", "JobTitle"],
        low_memory=False,
    )
    history["Sequence"] = pd.to_numeric(history["Sequence"], errors="coerce")
    history = history.dropna(subset=["UserID", "Sequence", "JobTitle"])
    history = history.sort_values(["UserID", "Sequence"], kind="mergesort")
    history = history.groupby("UserID", sort=False).head(max_recent_titles)
    history["JobTitle"] = history["JobTitle"].map(clean_text)
    joined_titles = history.groupby("UserID", sort=False)["JobTitle"].agg("; ".join)

    users["recent_titles"] = users["UserID"].map(joined_titles).fillna("")

    def safe(row: pd.Series, name: str) -> str:
        value = row.get(name, "")
        return clean_text(value)

    def compose(row: pd.Series) -> str:
        roles = safe(row, "recent_titles") or "not provided"
        major = safe(row, "Major") or "not provided"
        degree = safe(row, "DegreeType") or "not provided"
        experience = safe(row, "TotalYearsExperience") or "not provided"
        location = " ".join(
            x for x in [safe(row, "City"), safe(row, "State"), safe(row, "Country")] if x
        ) or "not provided"
        return clean_text(
            f"Recent roles: {roles}. Major: {major}. Degree: {degree}. "
            f"Years of experience: {experience}. Location: {location}."
        )

    users_out = pd.DataFrame({
        "user_id": pd.to_numeric(users["UserID"], errors="raise").astype("int64"),
        "window_id": pd.to_numeric(users["WindowID"], errors="raise").astype("int16"),
        "user_text": users.apply(compose, axis=1),
    })
    users_out.to_parquet(output_path, index=False, compression="zstd")
    manifest = {
        "status": "complete",
        "rows": len(users_out),
        "users_with_job_titles": int(users_out["user_id"].isin(joined_titles.index).sum()),
        "max_recent_titles": max_recent_titles,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _load_sentence_transformer(model_name: str, max_length: int):
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {"device": device}
    try:
        if device == "cuda":
            model = SentenceTransformer(
                model_name,
                model_kwargs={"torch_dtype": torch.float16},
                **kwargs,
            )
        else:
            model = SentenceTransformer(model_name, **kwargs)
    except TypeError:
        model = SentenceTransformer(model_name, **kwargs)
        if device == "cuda":
            model.half()
    model.max_seq_length = max_length
    return model, device


def _encode_with_retry(model, texts: Sequence[str], batch_size: int, device: str) -> tuple[np.ndarray, int]:
    import torch

    current = batch_size
    while current >= 1:
        try:
            with torch.inference_mode():
                embeddings = model.encode(
                    list(texts),
                    batch_size=current,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=True,
                )
            return np.asarray(embeddings), current
        except RuntimeError as exc:
            message = str(exc).lower()
            if device != "cuda" or ("out of memory" not in message and "cuda" not in message):
                raise
            torch.cuda.empty_cache()
            next_size = current // 2
            if next_size < 1:
                raise
            print(f"[OOM] Reducing batch size from {current} to {next_size}")
            current = next_size
    raise RuntimeError("Could not encode even with batch size 1")


def _embedding_manifest_matches(path: Path, expected: dict[str, object]) -> bool:
    if not path.exists():
        return False
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return all(actual.get(key) == value for key, value in expected.items()) and actual.get("status") == "complete"


def embed_parquet_shards(
    parquet_paths: Sequence[Path],
    output_dir: Path,
    model_name: str,
    text_column: str,
    id_column: str,
    max_length: int,
    batch_size: int,
    output_dtype: str = "float16",
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model, device = _load_sentence_transformer(model_name, max_length)
    dtype = np.float16 if output_dtype == "float16" else np.float32
    total_rows = 0
    effective_batches: list[int] = []
    started = time.time()

    for shard_index, parquet_path in enumerate(parquet_paths):
        ids_path = output_dir / f"shard_{shard_index:05d}_ids.npy"
        emb_path = output_dir / f"shard_{shard_index:05d}_embeddings.npy"
        shard_manifest_path = output_dir / f"shard_{shard_index:05d}.json"
        source_hash = sha256_file(parquet_path)
        expected = {
            "source_file": parquet_path.name,
            "source_sha256": source_hash,
            "model_name": model_name,
            "text_column": text_column,
            "id_column": id_column,
            "max_length": max_length,
            "output_dtype": output_dtype,
        }
        if ids_path.exists() and emb_path.exists() and _embedding_manifest_matches(shard_manifest_path, expected):
            prior = json.loads(shard_manifest_path.read_text(encoding="utf-8"))
            total_rows += int(prior["rows"])
            print(f"[resume] {parquet_path.name} already embedded ({prior['rows']:,} rows)")
            continue

        frame = pd.read_parquet(parquet_path, columns=[id_column, text_column])
        texts = frame[text_column].fillna("").astype(str).tolist()
        embeddings, effective_batch = _encode_with_retry(model, texts, batch_size, device)
        embeddings = embeddings.astype(dtype, copy=False)
        ids = pd.to_numeric(frame[id_column], errors="raise").to_numpy(dtype=np.int64)
        if len(ids) != len(embeddings):
            raise RuntimeError(f"Length mismatch for {parquet_path.name}")
        norms = np.linalg.norm(embeddings.astype(np.float32), axis=1)
        if not np.isfinite(embeddings).all() or not np.isfinite(norms).all():
            raise RuntimeError(f"Non-finite embeddings in {parquet_path.name}")
        if len(norms) and float(np.max(np.abs(norms - 1.0))) > 0.05:
            raise RuntimeError(f"Embeddings are not sufficiently normalized in {parquet_path.name}")
        np.save(ids_path, ids, allow_pickle=False)
        np.save(emb_path, embeddings, allow_pickle=False)
        shard_manifest = {
            **expected,
            "status": "complete",
            "rows": len(ids),
            "embedding_dim": int(embeddings.shape[1]),
            "effective_batch_size": effective_batch,
            "ids_sha256": sha256_file(ids_path),
            "embeddings_sha256": sha256_file(emb_path),
        }
        shard_manifest_path.write_text(json.dumps(shard_manifest, indent=2), encoding="utf-8")
        total_rows += len(ids)
        effective_batches.append(effective_batch)
        print(f"  embedded {parquet_path.name}: {len(ids):,} rows -> {emb_path.name}")

    summary = {
        "status": "complete",
        "model_name": model_name,
        "device": device,
        "text_column": text_column,
        "id_column": id_column,
        "max_length": max_length,
        "requested_batch_size": batch_size,
        "minimum_effective_batch_size": min(effective_batches) if effective_batches else None,
        "output_dtype": output_dtype,
        "rows": total_rows,
        "shards": len(parquet_paths),
        "seconds": round(time.time() - started, 2),
    }
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def prepare_jobbert_title_documents(data_dir: Path, output_dir: Path) -> dict[str, object]:
    target_dir = output_dir / "prepared" / "jobbert_titles"
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / "app_job_titles.parquet"
    manifest_path = target_dir / "manifest.json"
    if output_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "complete":
            print(f"[resume] JobBERT title documents complete: {manifest['rows']:,} rows")
            return manifest

    app_ids = set(
        pd.read_csv(data_dir / "apps.tsv", sep="\t", usecols=["JobID"])["JobID"]
        .astype("int64")
        .unique()
        .tolist()
    )
    rows: list[dict[str, object]] = []
    jobs_dir = output_dir / "prepared" / "jobs"
    for path in sorted(jobs_dir.glob("jobs_*.parquet")):
        frame = pd.read_parquet(path, columns=["job_id", "title"])
        selected = frame[frame["job_id"].isin(app_ids)].copy()
        if not selected.empty:
            selected = selected.rename(columns={"title": "job_title"})
            rows.extend(selected.to_dict("records"))
    result = pd.DataFrame.from_records(rows).drop_duplicates("job_id").sort_values("job_id")
    result.to_parquet(output_path, index=False, compression="zstd")
    manifest = {
        "status": "complete",
        "rows": len(result),
        "unique_application_job_ids": len(app_ids),
        "missing_titles_for_application_jobs": len(app_ids - set(result["job_id"].astype(int).tolist())),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_all(config: RunConfig) -> dict[str, object]:
    data_dir = Path(config.data_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_started = time.time()

    input_report = validate_inputs(data_dir)
    (output_dir / "input_validation.json").write_text(
        json.dumps(input_report, indent=2), encoding="utf-8"
    )
    job_prepare = prepare_clean_jobs(data_dir, output_dir, config.job_chunk_size)
    user_prepare = build_user_documents(data_dir, output_dir)

    jobs_parquets = sorted((output_dir / "prepared" / "jobs").glob("jobs_*.parquet"))
    gte_jobs = embed_parquet_shards(
        jobs_parquets,
        output_dir / "embeddings" / "gte_jobs",
        config.gte_model,
        "job_text",
        "job_id",
        config.gte_max_length,
        config.gte_batch_size,
        config.output_dtype,
    )

    user_parquet = output_dir / "prepared" / "users" / "user_documents.parquet"
    gte_users = embed_parquet_shards(
        [user_parquet],
        output_dir / "embeddings" / "gte_users",
        config.gte_model,
        "user_text",
        "user_id",
        config.gte_max_length,
        config.gte_batch_size,
        config.output_dtype,
    )

    title_prepare = prepare_jobbert_title_documents(data_dir, output_dir)
    title_parquet = output_dir / "prepared" / "jobbert_titles" / "app_job_titles.parquet"
    jobbert = embed_parquet_shards(
        [title_parquet],
        output_dir / "embeddings" / "jobbert_app_titles",
        config.jobbert_model,
        "job_title",
        "job_id",
        config.jobbert_max_length,
        config.jobbert_batch_size,
        config.output_dtype,
    )

    summary = {
        "status": "complete",
        "config": asdict(config),
        "input_validation": input_report,
        "prepared_jobs": job_prepare,
        "prepared_users": user_prepare,
        "prepared_jobbert_titles": title_prepare,
        "gte_jobs": gte_jobs,
        "gte_users": gte_users,
        "jobbert_app_titles": jobbert,
        "total_seconds": round(time.time() - run_started, 2),
    }
    summary_path = output_dir / "embedding_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def verify_outputs(output_dir: Path) -> dict[str, object]:
    checks: dict[str, object] = {"status": "complete", "groups": {}}
    for name in ["gte_jobs", "gte_users", "jobbert_app_titles"]:
        directory = output_dir / "embeddings" / name
        ids_files = sorted(directory.glob("shard_*_ids.npy"))
        emb_files = sorted(directory.glob("shard_*_embeddings.npy"))
        if len(ids_files) != len(emb_files) or not ids_files:
            raise RuntimeError(f"Incomplete output group: {name}")
        total = 0
        dim = None
        for ids_path, emb_path in zip(ids_files, emb_files):
            ids = np.load(ids_path, mmap_mode="r")
            emb = np.load(emb_path, mmap_mode="r")
            if len(ids) != len(emb):
                raise RuntimeError(f"Length mismatch: {ids_path.name}, {emb_path.name}")
            if emb.ndim != 2:
                raise RuntimeError(f"Expected 2-D embeddings: {emb_path.name}")
            total += len(ids)
            dim = emb.shape[1] if dim is None else dim
            if emb.shape[1] != dim:
                raise RuntimeError(f"Dimension mismatch in {name}")
        checks["groups"][name] = {"rows": total, "embedding_dim": dim, "shards": len(ids_files)}
    verify_path = output_dir / "output_verification.json"
    verify_path.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    return checks
