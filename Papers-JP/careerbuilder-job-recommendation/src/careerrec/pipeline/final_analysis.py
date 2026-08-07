from __future__ import annotations

"""Final paper analysis pipeline.

Reuses Phase 1 embeddings, Phase 2 concentration outputs, and Phase 3 BM25/
Item-KNN candidates. It builds location-aware GTE retrieval, selects the best
content model chronologically, compares Item-KNN hybrids/gates for users with
at least two prior applications, and writes paper-ready tables/statistics.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import gc
import json
import math
import shutil
import time
import warnings
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Phase4Config:
    data_dir: str
    output_dir: str
    local_cache_dir: str = ".cache/careerrec/final"
    localized_top_k: int = 1000
    recommendation_k: int = 200
    evaluation_k: int = 10
    query_batch_size: int = 128
    rrf_constant: int = 60
    min_local_pool: int = 100
    bootstrap_repetitions: int = 1000
    random_seed: int = 42


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return _read_json(path).get("status") == "complete"
    except Exception:
        return False


def _norm(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


def _history_segment(n: int) -> str:
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    if n <= 4:
        return "3-4"
    if n <= 9:
        return "5-9"
    return "10+"


def _dense_index(ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) == 0:
        return np.full(1, -1, dtype=np.int64)
    if ids.min() < 0:
        raise ValueError("IDs must be non-negative")
    out = np.full(int(ids.max()) + 1, -1, dtype=np.int64)
    out[ids] = np.arange(len(ids), dtype=np.int64)
    return out


def _candidate_file(directory: Path, window_id: int) -> Path:
    matches = sorted(directory.glob(f"window_{window_id}_top*.npz"))
    if not matches:
        raise FileNotFoundError(f"No candidate file for window {window_id} in {directory}")
    return matches[0]


@lru_cache(maxsize=32)
def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: np.asarray(data[k]) for k in data.files}


def _truth_sets(offsets: np.ndarray, flat: np.ndarray) -> list[set[int]]:
    offsets = np.asarray(offsets, dtype=np.int64)
    flat = np.asarray(flat, dtype=np.int64)
    return [set(map(int, flat[offsets[i]: offsets[i + 1]])) for i in range(len(offsets) - 1)]


def _window_cutoff(windows: pd.DataFrame, window_id: int) -> pd.Timestamp:
    row = windows.loc[windows["Window"].astype(int) == int(window_id)].iloc[0]
    return row["Train Start"] + pd.Timedelta(days=5)


def load_core_tables(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    apps = pd.read_csv(
        data_dir / "apps.tsv",
        sep="\t",
        usecols=["UserID", "WindowID", "ApplicationDate", "JobID"],
        low_memory=False,
    )
    apps["UserID"] = pd.to_numeric(apps["UserID"], errors="raise").astype("int64")
    apps["WindowID"] = pd.to_numeric(apps["WindowID"], errors="raise").astype("int16")
    apps["JobID"] = pd.to_numeric(apps["JobID"], errors="raise").astype("int64")
    apps["ApplicationDate"] = pd.to_datetime(apps["ApplicationDate"], errors="coerce", utc=True)
    apps = apps.dropna(subset=["ApplicationDate"]).sort_values("ApplicationDate", kind="mergesort")

    users = pd.read_csv(data_dir / "users.tsv", sep="\t", low_memory=False)
    users["UserID"] = pd.to_numeric(users["UserID"], errors="raise").astype("int64")
    users["WindowID"] = pd.to_numeric(users["WindowID"], errors="raise").astype("int16")
    for col in ["City", "State", "Country"]:
        if col not in users:
            users[col] = ""
        users[f"_{col.lower()}"] = users[col].map(_norm)

    windows = pd.read_csv(data_dir / "window_dates.tsv", sep="\t")
    windows["Window"] = pd.to_numeric(windows["Window"], errors="raise").astype("int16")
    for col in ["Train Start", "Train End / Test Start", "Test End"]:
        windows[col] = pd.to_datetime(windows[col], errors="raise", utc=True)
    return apps, users, windows


# ---------------------------------------------------------------------------
# Staging and active job cache
# ---------------------------------------------------------------------------


def _copy_if_needed(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    print(f"[stage] {src.name} -> {dst}")
    shutil.copy2(src, dst)


def stage_embedding_inputs(output_dir: Path, cache_dir: Path) -> dict[str, object]:
    job_parquets = sorted((output_dir / "prepared" / "jobs").glob("jobs_*.parquet"))
    job_id_shards = sorted((output_dir / "embeddings" / "gte_jobs").glob("shard_*_ids.npy"))
    job_emb_shards = sorted((output_dir / "embeddings" / "gte_jobs").glob("shard_*_embeddings.npy"))
    if not job_parquets or not (len(job_parquets) == len(job_id_shards) == len(job_emb_shards)):
        raise RuntimeError("Prepared jobs and GTE job shards are incomplete")

    local_parquets: list[Path] = []
    local_ids: list[Path] = []
    local_emb: list[Path] = []
    for src in job_parquets:
        dst = cache_dir / "jobs" / src.name
        _copy_if_needed(src, dst)
        local_parquets.append(dst)
    for src in job_id_shards:
        dst = cache_dir / "gte_jobs" / src.name
        _copy_if_needed(src, dst)
        local_ids.append(dst)
    for src in job_emb_shards:
        dst = cache_dir / "gte_jobs" / src.name
        _copy_if_needed(src, dst)
        local_emb.append(dst)

    user_ids_src = output_dir / "embeddings" / "gte_users" / "shard_00000_ids.npy"
    user_emb_src = output_dir / "embeddings" / "gte_users" / "shard_00000_embeddings.npy"
    user_ids_dst = cache_dir / "gte_users" / user_ids_src.name
    user_emb_dst = cache_dir / "gte_users" / user_emb_src.name
    _copy_if_needed(user_ids_src, user_ids_dst)
    _copy_if_needed(user_emb_src, user_emb_dst)
    return {
        "job_parquets": local_parquets,
        "job_id_shards": local_ids,
        "job_emb_shards": local_emb,
        "user_ids": user_ids_dst,
        "user_embeddings": user_emb_dst,
    }


def build_active_cache(
    staged: Mapping[str, object],
    windows: pd.DataFrame,
    window_id: int,
    cache_dir: Path,
) -> dict[str, Path]:
    ids_path = cache_dir / f"window_{window_id}_ids.npy"
    emb_path = cache_dir / f"window_{window_id}_embeddings.npy"
    meta_path = cache_dir / f"window_{window_id}_meta.parquet"
    done_path = cache_dir / f"window_{window_id}.json"
    if ids_path.exists() and emb_path.exists() and meta_path.exists() and _is_complete(done_path):
        return {"ids": ids_path, "embeddings": emb_path, "meta": meta_path}

    cutoff_ts = int(_window_cutoff(windows, window_id).timestamp())
    id_parts: list[np.ndarray] = []
    emb_parts: list[np.ndarray] = []
    meta_parts: list[pd.DataFrame] = []
    for parquet_path, ids_shard_path, emb_shard_path in zip(
        staged["job_parquets"], staged["job_id_shards"], staged["job_emb_shards"]
    ):
        frame = pd.read_parquet(
            parquet_path,
            columns=["job_id", "window_id", "start_ts", "end_ts", "city", "state", "country"],
        )
        ids = np.load(ids_shard_path, mmap_mode="r")
        emb = np.load(emb_shard_path, mmap_mode="r")
        if len(frame) != len(ids) or len(ids) != len(emb):
            raise RuntimeError(f"Shard length mismatch: {Path(parquet_path).name}")
        if not np.array_equal(frame["job_id"].to_numpy(dtype=np.int64), np.asarray(ids, dtype=np.int64)):
            raise RuntimeError(f"Job order mismatch: {Path(parquet_path).name}")
        mask = (
            (frame["window_id"].to_numpy(dtype=np.int16) == int(window_id))
            & (frame["start_ts"].to_numpy(dtype=np.int64) <= cutoff_ts)
            & (frame["end_ts"].to_numpy(dtype=np.int64) >= cutoff_ts)
        )
        if not np.any(mask):
            continue
        ids_local = frame.loc[mask, "job_id"].to_numpy(dtype=np.int64)
        id_parts.append(ids_local)
        emb_parts.append(np.asarray(emb[mask], dtype=np.float16))
        m = frame.loc[mask, ["job_id", "city", "state", "country"]].copy()
        m["city_key"] = m["city"].map(_norm)
        m["state_key"] = m["state"].map(_norm)
        m["country_key"] = m["country"].map(_norm)
        meta_parts.append(m[["job_id", "city_key", "state_key", "country_key"]])

    if not id_parts:
        raise RuntimeError(f"No active jobs for window {window_id}")
    ids_all = np.concatenate(id_parts)
    emb_all = np.concatenate(emb_parts)
    meta_all = pd.concat(meta_parts, ignore_index=True)
    order = np.argsort(ids_all, kind="mergesort")
    ids_all = ids_all[order]
    emb_all = emb_all[order]
    meta_all = meta_all.iloc[order].reset_index(drop=True)
    if meta_all["job_id"].duplicated().any():
        raise RuntimeError(f"Duplicate active JobID in window {window_id}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(ids_path, ids_all, allow_pickle=False)
    np.save(emb_path, emb_all, allow_pickle=False)
    meta_all.to_parquet(meta_path, index=False, compression="zstd")
    _write_json(done_path, {
        "status": "complete",
        "window_id": int(window_id),
        "rows": int(len(ids_all)),
        "embedding_dim": int(emb_all.shape[1]),
    })
    del emb_all
    gc.collect()
    return {"ids": ids_path, "embeddings": emb_path, "meta": meta_path}


# ---------------------------------------------------------------------------
# Retrieval and fusion
# ---------------------------------------------------------------------------


def filter_seen_topk(
    candidate_ids: Sequence[int],
    candidate_scores: Sequence[float],
    seen: set[int],
    top_k: int,
) -> tuple[list[int], list[float]]:
    ids_out: list[int] = []
    scores_out: list[float] = []
    used: set[int] = set()
    for jid, score in zip(candidate_ids, candidate_scores):
        j = int(jid)
        if j < 0 or j in seen or j in used:
            continue
        ids_out.append(j)
        scores_out.append(float(score))
        used.add(j)
        if len(ids_out) >= top_k:
            break
    return ids_out, scores_out


def multi_rrf(
    rankings: Sequence[Sequence[int]],
    weights: Sequence[float],
    top_k: int,
    constant: int = 60,
) -> list[int]:
    if len(rankings) != len(weights):
        raise ValueError("rankings and weights must have equal lengths")
    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        w = float(weight)
        if w <= 0:
            continue
        for rank, jid in enumerate(ranking, start=1):
            j = int(jid)
            if j < 0:
                continue
            scores[j] = scores.get(j, 0.0) + w / (constant + rank)
    ordered = sorted(scores, key=lambda j: (-scores[j], j))
    return ordered[:top_k]


def fuse_arrays(
    first: np.ndarray,
    second: np.ndarray,
    second_weight: np.ndarray | float,
    top_k: int,
    constant: int,
) -> np.ndarray:
    first = np.asarray(first)
    second = np.asarray(second)
    if np.isscalar(second_weight):
        weights = np.full(len(first), float(second_weight), dtype=np.float32)
    else:
        weights = np.asarray(second_weight, dtype=np.float32)
    if len(first) != len(second) or len(first) != len(weights):
        raise ValueError("Candidate arrays have inconsistent lengths")
    out = np.full((len(first), top_k), -1, dtype=np.int32)
    for i in range(len(first)):
        ranking = multi_rrf(
            [first[i], second[i]],
            [1.0 - float(weights[i]), float(weights[i])],
            top_k,
            constant,
        )
        out[i, : len(ranking)] = np.asarray(ranking, dtype=np.int32)
    return out


def _torch_group_search(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidate_job_ids: np.ndarray,
    top_k: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    candidates = torch.as_tensor(candidate_embeddings, device=device, dtype=dtype)
    k = min(int(top_k), len(candidate_job_ids))
    out_ids = np.full((len(query_embeddings), top_k), -1, dtype=np.int32)
    out_scores = np.full((len(query_embeddings), top_k), -np.inf, dtype=np.float32)
    current_batch = max(1, int(batch_size))
    start = 0
    while start < len(query_embeddings):
        stop = min(start + current_batch, len(query_embeddings))
        try:
            q = torch.as_tensor(query_embeddings[start:stop], device=device, dtype=dtype)
            scores = q @ candidates.T
            values, indices = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)
            idx_np = indices.detach().cpu().numpy()
            val_np = values.detach().cpu().numpy().astype(np.float32)
            out_ids[start:stop, :k] = np.asarray(candidate_job_ids, dtype=np.int64)[idx_np].astype(np.int32)
            out_scores[start:stop, :k] = val_np
            del q, scores, values, indices
            if device.type == "cuda":
                torch.cuda.empty_cache()
            start = stop
        except RuntimeError as exc:
            if device.type != "cuda" or "out of memory" not in str(exc).lower() or current_batch <= 1:
                raise
            torch.cuda.empty_cache()
            current_batch = max(1, current_batch // 2)
            print(f"[OOM] local search batch -> {current_batch}")
    del candidates
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out_ids, out_scores


def _past_sets(apps: pd.DataFrame, user_ids: np.ndarray, cutoff: pd.Timestamp) -> dict[int, set[int]]:
    user_set = set(np.asarray(user_ids, dtype=np.int64).tolist())
    past = apps[(apps["ApplicationDate"] < cutoff) & (apps["UserID"].isin(user_set))]
    return {
        int(uid): set(group["JobID"].astype(np.int64).tolist())
        for uid, group in past.groupby("UserID", sort=False)
    }


def _city_popularity_lists(
    active_meta: pd.DataFrame,
    past_apps: pd.DataFrame,
    users_window: pd.DataFrame,
    top_k: int,
) -> dict[int, list[int]]:
    counts = past_apps["JobID"].value_counts()
    meta = active_meta.copy()
    meta["popularity"] = meta["job_id"].map(counts).fillna(0).astype(np.int64)
    city_rankings: dict[tuple[str, str, str], list[int]] = {}
    for key, group in meta.groupby(["country_key", "state_key", "city_key"], sort=False):
        if not key[2]:
            continue
        ranked = group.sort_values(["popularity", "job_id"], ascending=[False, True])
        city_rankings[key] = ranked["job_id"].head(top_k).astype(int).tolist()
    result: dict[int, list[int]] = {}
    for uid, country, state, city in users_window[["UserID", "_country", "_state", "_city"]].itertuples(index=False, name=None):
        key = (country, state, city)
        result[int(uid)] = city_rankings.get(key, [])
    return result


def build_localized_candidates_for_window(
    config: Phase4Config,
    window_id: int,
    apps: pd.DataFrame,
    users: pd.DataFrame,
    windows: pd.DataFrame,
    staged: Mapping[str, object],
    phase2_dir: Path,
    phase4_dir: Path,
    active_cache_dir: Path,
) -> dict:
    out_dir = phase4_dir / "localized_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / f"window_{window_id}_gte_state_top{config.localized_top_k}.npz"
    city_path = out_dir / f"window_{window_id}_gte_state_city_top{config.localized_top_k}.npz"
    report_path = out_dir / f"window_{window_id}.json"
    if state_path.exists() and city_path.exists() and _is_complete(report_path):
        print(f"[resume] localized GTE window {window_id}")
        return _read_json(report_path)

    global_payload = _load_npz(_candidate_file(phase2_dir / "gte_candidates", window_id))
    user_ids = global_payload["user_ids"].astype(np.int64)
    global_ids = global_payload["candidate_job_ids"].astype(np.int32)
    global_scores = global_payload["candidate_scores"].astype(np.float32)
    cutoff = _window_cutoff(windows, window_id)
    seen_by_user = _past_sets(apps, user_ids, cutoff)

    cache = build_active_cache(staged, windows, window_id, active_cache_dir)
    active_ids = np.load(cache["ids"], mmap_mode="r")
    active_emb = np.load(cache["embeddings"], mmap_mode="r")
    active_meta = pd.read_parquet(cache["meta"])
    active_meta["geo_key"] = list(zip(active_meta["country_key"], active_meta["state_key"]))

    user_ids_all = np.load(staged["user_ids"], mmap_mode="r")
    user_emb_all = np.load(staged["user_embeddings"], mmap_mode="r")
    user_lookup = _dense_index(np.asarray(user_ids_all, dtype=np.int64))
    q_rows = user_lookup[user_ids]
    if np.any(q_rows < 0):
        raise RuntimeError("Missing user embeddings in localized retrieval")
    query_embeddings = np.asarray(user_emb_all[q_rows], dtype=np.float16)

    profiles = users.set_index("UserID").loc[user_ids].reset_index()
    profiles["geo_key"] = list(zip(profiles["_country"], profiles["_state"]))
    state_ids = np.full((len(user_ids), config.localized_top_k), -1, dtype=np.int32)
    state_scores = np.full((len(user_ids), config.localized_top_k), -np.inf, dtype=np.float32)

    # Search by country/state. If the state pool is too small or unavailable, use country.
    active_country_groups = active_meta.groupby("country_key", sort=False).indices
    active_geo_groups = active_meta.groupby("geo_key", sort=False).indices
    user_groups = profiles.groupby("geo_key", sort=False).indices
    for geo_key, user_positions in user_groups.items():
        country, state = geo_key
        candidate_positions = active_geo_groups.get(geo_key)
        if candidate_positions is None or len(candidate_positions) < config.min_local_pool:
            candidate_positions = active_country_groups.get(country)
        if candidate_positions is None or len(candidate_positions) == 0:
            continue
        user_positions = np.asarray(user_positions, dtype=np.int64)
        candidate_positions = np.asarray(candidate_positions, dtype=np.int64)
        raw_ids, raw_scores = _torch_group_search(
            query_embeddings[user_positions],
            np.asarray(active_emb[candidate_positions], dtype=np.float16),
            np.asarray(active_ids[candidate_positions], dtype=np.int64),
            config.localized_top_k + 256,
            config.query_batch_size,
        )
        for local_i, global_pos in enumerate(user_positions):
            kept_ids, kept_scores = filter_seen_topk(
                raw_ids[local_i], raw_scores[local_i],
                seen_by_user.get(int(user_ids[global_pos]), set()),
                config.localized_top_k,
            )
            if kept_ids:
                state_ids[global_pos, : len(kept_ids)] = np.asarray(kept_ids, dtype=np.int32)
                state_scores[global_pos, : len(kept_scores)] = np.asarray(kept_scores, dtype=np.float32)
        print(f"[local GTE] W{window_id} {geo_key}: users={len(user_positions):,}, jobs={len(candidate_positions):,}")

    # Location popularity is built only from past interactions, never future labels.
    past_all = apps[apps["ApplicationDate"] < cutoff]
    users_window = profiles[["UserID", "_country", "_state", "_city"]].copy()
    city_pop = _city_popularity_lists(active_meta, past_all, users_window, top_k=300)

    state_fused = np.full((len(user_ids), config.localized_top_k), -1, dtype=np.int32)
    city_fused = np.full((len(user_ids), config.localized_top_k), -1, dtype=np.int32)
    for i, uid in enumerate(user_ids):
        global_list = [int(x) for x in global_ids[i] if int(x) >= 0]
        state_list = [int(x) for x in state_ids[i] if int(x) >= 0]
        if state_list:
            state_rank = multi_rrf(
                [global_list, state_list], [0.5, 0.5],
                config.localized_top_k, config.rrf_constant,
            )
        else:
            state_rank = global_list[: config.localized_top_k]
        city_list = city_pop.get(int(uid), [])
        if city_list:
            city_rank = multi_rrf(
                [global_list, state_list or global_list, city_list],
                [0.4, 0.4, 0.2], config.localized_top_k, config.rrf_constant,
            )
        else:
            city_rank = state_rank
        state_fused[i, : len(state_rank)] = np.asarray(state_rank, dtype=np.int32)
        city_fused[i, : len(city_rank)] = np.asarray(city_rank, dtype=np.int32)

    common_payload = {
        "user_ids": user_ids,
        "truth_offsets": global_payload["truth_offsets"].astype(np.int64),
        "truth_job_ids": global_payload["truth_job_ids"].astype(np.int32),
        "n_past_apps": global_payload["n_past_apps"].astype(np.int32),
    }
    np.savez_compressed(
        state_path,
        **common_payload,
        candidate_job_ids=state_fused,
    )
    np.savez_compressed(
        city_path,
        **common_payload,
        candidate_job_ids=city_fused,
    )
    truths = _truth_sets(common_payload["truth_offsets"], common_payload["truth_job_ids"])
    report = {
        "status": "complete",
        "window_id": int(window_id),
        "users": int(len(user_ids)),
        "active_jobs": int(len(active_ids)),
        "global_candidate_recall": candidate_recall(global_ids, truths),
        "state_candidate_recall": candidate_recall(state_fused, truths),
        "state_city_candidate_recall": candidate_recall(city_fused, truths),
    }
    _write_json(report_path, report)
    del active_emb, query_embeddings, state_scores
    gc.collect()
    return report


# ---------------------------------------------------------------------------
# Metrics and diagnostics
# ---------------------------------------------------------------------------


def user_metrics(
    window_id: int,
    model: str,
    user_ids: np.ndarray,
    recs: np.ndarray,
    truths: Sequence[set[int]],
    counts: np.ndarray,
    k: int = 10,
) -> pd.DataFrame:
    rows: list[dict] = []
    for i, uid in enumerate(user_ids):
        truth = truths[i]
        top = [int(x) for x in recs[i, :k] if int(x) >= 0]
        hits = [rank for rank, jid in enumerate(top, start=1) if jid in truth]
        dcg = sum(1.0 / math.log2(rank + 1) for rank in hits)
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(truth), k) + 1))
        rows.append({
            "window_id": int(window_id),
            "user_id": int(uid),
            "model": model,
            "n_past_apps": int(counts[i]),
            "history_segment": _history_segment(int(counts[i])),
            "ndcg10": float(dcg / ideal) if ideal > 0 else 0.0,
            "recall10": float(len(set(top) & truth) / len(truth)) if truth else 0.0,
            "mrr": float(1.0 / min(hits)) if hits else 0.0,
        })
    return pd.DataFrame(rows)


def candidate_recall(recs: np.ndarray, truths: Sequence[set[int]]) -> float:
    hits = 0
    total = 0
    for row, truth in zip(recs, truths):
        rec_set = set(int(x) for x in row if int(x) >= 0)
        hits += len(rec_set & truth)
        total += len(truth)
    return float(hits / total) if total else 0.0


def ranking_diagnostics(
    recs: np.ndarray,
    active_job_ids: np.ndarray,
    past_counts: Mapping[int, int],
    k: int = 10,
) -> dict[str, float]:
    top = np.asarray(recs[:, :k], dtype=np.int64)
    valid = top[top >= 0]
    unique = np.unique(valid) if len(valid) else np.array([], dtype=np.int64)
    avg_pop = float(np.mean([past_counts.get(int(j), 0) for j in valid])) if len(valid) else 0.0
    return {
        "catalog_coverage10": float(len(unique) / len(active_job_ids)) if len(active_job_ids) else 0.0,
        "unique_jobs10": int(len(unique)),
        "average_past_popularity10": avg_pop,
    }


def load_model_candidates(
    phase2_dir: Path,
    phase3_dir: Path,
    phase4_dir: Path,
    window_id: int,
    top_k: int,
    rrf_constant: int,
) -> tuple[dict[str, np.ndarray], dict]:
    gte = _load_npz(_candidate_file(phase2_dir / "gte_candidates", window_id))
    bm25 = _load_npz(_candidate_file(phase3_dir / "bm25_candidates", window_id))
    itemknn = _load_npz(_candidate_file(phase3_dir / "itemknn_candidates", window_id))
    state_matches = sorted((phase4_dir / "localized_candidates").glob(f"window_{window_id}_gte_state_top*.npz"))
    city_matches = sorted((phase4_dir / "localized_candidates").glob(f"window_{window_id}_gte_state_city_top*.npz"))
    if not state_matches:
        raise FileNotFoundError(f"Missing state candidates for window {window_id}")
    if not city_matches:
        raise FileNotFoundError(f"Missing state-city candidates for window {window_id}")
    state = _load_npz(state_matches[0])
    city = _load_npz(city_matches[0])
    user_ids = gte["user_ids"].astype(np.int64)
    for name, payload in [("bm25", bm25), ("itemknn", itemknn), ("state", state), ("city", city)]:
        if not np.array_equal(payload["user_ids"].astype(np.int64), user_ids):
            raise RuntimeError(f"User ordering mismatch: {name} W{window_id}")
    candidates = {
        "gte_global": gte["candidate_job_ids"].astype(np.int32),
        "gte_state": state["candidate_job_ids"].astype(np.int32),
        "gte_state_city": city["candidate_job_ids"].astype(np.int32),
        "bm25": bm25["candidate_job_ids"].astype(np.int32),
        "itemknn": itemknn["candidate_job_ids"].astype(np.int32),
    }
    candidates["bm25_gte_local"] = fuse_arrays(
        candidates["gte_state_city"], candidates["bm25"], 0.5,
        top_k, rrf_constant,
    )
    return candidates, gte


def evaluate_base_models(
    config: Phase4Config,
    apps: pd.DataFrame,
    windows: pd.DataFrame,
    staged: Mapping[str, object],
    phase2_dir: Path,
    phase3_dir: Path,
    phase4_dir: Path,
    active_cache_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = phase4_dir / "user_metrics_base.parquet"
    diag_path = phase4_dir / "ranking_diagnostics.csv"
    if metrics_path.exists() and diag_path.exists():
        return pd.read_parquet(metrics_path), pd.read_csv(diag_path)
    metric_parts: list[pd.DataFrame] = []
    diag_rows: list[dict] = []
    for window_id in range(1, 8):
        candidates, payload = load_model_candidates(
            phase2_dir, phase3_dir, phase4_dir, window_id,
            config.localized_top_k, config.rrf_constant,
        )
        truths = _truth_sets(payload["truth_offsets"], payload["truth_job_ids"])
        user_ids = payload["user_ids"].astype(np.int64)
        counts = payload["n_past_apps"].astype(np.int32)
        cache = build_active_cache(staged, windows, window_id, active_cache_dir)
        active_ids = np.load(cache["ids"], mmap_mode="r")
        cutoff = _window_cutoff(windows, window_id)
        past_counts = apps.loc[apps["ApplicationDate"] < cutoff, "JobID"].value_counts().to_dict()
        for model, recs in candidates.items():
            metric_parts.append(user_metrics(
                window_id, model, user_ids, recs, truths, counts, config.evaluation_k
            ))
            diag_rows.append({
                "window_id": int(window_id),
                "model": model,
                "candidate_recall1000": candidate_recall(recs[:, : config.localized_top_k], truths),
                **ranking_diagnostics(recs, active_ids, past_counts, config.evaluation_k),
            })
    metrics = pd.concat(metric_parts, ignore_index=True)
    diagnostics = pd.DataFrame(diag_rows)
    metrics.to_parquet(metrics_path, index=False, compression="zstd")
    diagnostics.to_csv(diag_path, index=False)
    return metrics, diagnostics


# ---------------------------------------------------------------------------
# Chronological Item-KNN gates
# ---------------------------------------------------------------------------


GATE_FEATURES = {
    "count": ["log1p_count"],
    "gte_concentration": [
        "log1p_count", "gte_concentration", "gte_count_x_concentration"
    ],
    "jobbert_concentration": [
        "log1p_count", "jobbert_concentration", "jobbert_count_x_concentration"
    ],
}


def _feature_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    result = frame[list(columns)].copy()
    for col in columns:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0)
    return result.to_numpy(dtype=np.float64)


def _metric_lookup(metrics: pd.DataFrame, model: str) -> pd.DataFrame:
    return metrics.loc[metrics["model"] == model, ["window_id", "user_id", "ndcg10"]].rename(
        columns={"ndcg10": model}
    )


def _mean_ndcg_subset(
    recs: np.ndarray,
    payload: Mapping[str, np.ndarray],
    mask: np.ndarray,
    k: int,
) -> float:
    truths = _truth_sets(payload["truth_offsets"], payload["truth_job_ids"])
    metrics = user_metrics(
        0, "tmp", payload["user_ids"], recs, truths,
        payload["n_past_apps"], k,
    )
    return float(metrics.loc[mask, "ndcg10"].mean())


def run_final_gates(
    config: Phase4Config,
    base_metrics: pd.DataFrame,
    features: pd.DataFrame,
    phase2_dir: Path,
    phase3_dir: Path,
    phase4_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gate_path = phase4_dir / "user_metrics_final_gates.parquet"
    coef_path = phase4_dir / "gate_coefficients.csv"
    tuning_path = phase4_dir / "gate_tuning.csv"
    fold_path = phase4_dir / "fold_content_selection.csv"
    if gate_path.exists() and coef_path.exists() and tuning_path.exists() and fold_path.exists():
        return (
            pd.read_parquet(gate_path), pd.read_csv(coef_path),
            pd.read_csv(tuning_path), pd.read_csv(fold_path),
        )

    content_models = ["gte_global", "gte_state", "gte_state_city", "bm25", "bm25_gte_local"]
    folds = [
        {"train": [1, 2, 3], "tune": 4, "test": 5},
        {"train": [1, 2, 3, 4], "tune": 5, "test": 6},
        {"train": [1, 2, 3, 4, 5], "tune": 6, "test": 7},
    ]
    c_grid = [0.01, 0.1, 1.0, 10.0]
    static_grid = [0.25, 0.5, 0.75]
    metric_parts: list[pd.DataFrame] = []
    coef_rows: list[dict] = []
    tuning_rows: list[dict] = []
    fold_rows: list[dict] = []

    for fold in folds:
        tune_metrics = base_metrics[
            (base_metrics["window_id"] == fold["tune"])
            & (base_metrics["n_past_apps"] >= 2)
            & (base_metrics["model"].isin(content_models))
        ]
        tune_summary = tune_metrics.groupby("model")["ndcg10"].mean().sort_values(ascending=False)
        selected_content = str(tune_summary.index[0])
        fold_rows.append({
            "train_windows": ",".join(map(str, fold["train"])),
            "tune_window": fold["tune"],
            "test_window": fold["test"],
            "selected_content_model": selected_content,
            "tune_ndcg10_2plus": float(tune_summary.iloc[0]),
        })

        content_metric = _metric_lookup(base_metrics, selected_content)
        behavior_metric = _metric_lookup(base_metrics, "itemknn")
        gate_data = features.merge(content_metric, on=["window_id", "user_id"], how="inner")
        gate_data = gate_data.merge(behavior_metric, on=["window_id", "user_id"], how="inner")
        gate_data = gate_data[gate_data["n_past_apps"] >= 2].copy()
        gate_data["behavior_better"] = (
            gate_data["itemknn"] > gate_data[selected_content]
        ).astype(np.int8)

        # Tune static hybrid using the tune window only.
        tune_payload = _load_npz(_candidate_file(phase2_dir / "gte_candidates", fold["tune"]))
        tune_candidates, _ = load_model_candidates(
            phase2_dir, phase3_dir, phase4_dir, fold["tune"],
            config.recommendation_k, config.rrf_constant,
        )
        tune_mask = tune_payload["n_past_apps"].astype(np.int32) >= 2
        best_static = None
        for weight in static_grid:
            recs = fuse_arrays(
                tune_candidates[selected_content], tune_candidates["itemknn"],
                weight, config.recommendation_k, config.rrf_constant,
            )
            score = _mean_ndcg_subset(recs, tune_payload, tune_mask, config.evaluation_k)
            tuning_rows.append({
                "gate": "static_hybrid", "test_window": fold["test"],
                "parameter": "behavior_weight", "value": weight,
                "validation_score": score,
            })
            if best_static is None or score > best_static[0]:
                best_static = (score, weight)
        assert best_static is not None

        test_candidates, test_payload = load_model_candidates(
            phase2_dir, phase3_dir, phase4_dir, fold["test"],
            config.recommendation_k, config.rrf_constant,
        )
        test_user_ids = test_payload["user_ids"].astype(np.int64)
        test_counts = test_payload["n_past_apps"].astype(np.int32)
        test_truths = _truth_sets(test_payload["truth_offsets"], test_payload["truth_job_ids"])
        test_2plus = test_counts >= 2

        static_recs = fuse_arrays(
            test_candidates[selected_content], test_candidates["itemknn"],
            best_static[1], config.recommendation_k, config.rrf_constant,
        )
        static_metrics = user_metrics(
            fold["test"], "static_itemknn_hybrid", test_user_ids,
            static_recs, test_truths, test_counts, config.evaluation_k,
        )
        static_metrics["primary_2plus"] = test_2plus
        static_metrics["selected_content_model"] = selected_content
        static_metrics["gate_weight"] = float(best_static[1])
        metric_parts.append(static_metrics)

        for gate_name, columns in GATE_FEATURES.items():
            train = gate_data[gate_data["window_id"].isin(fold["train"])].copy()
            tune = gate_data[gate_data["window_id"] == fold["tune"]].copy()
            test = gate_data[gate_data["window_id"] == fold["test"]].copy()
            scaler = StandardScaler()
            x_train = scaler.fit_transform(_feature_matrix(train, columns))
            y_train = train["behavior_better"].to_numpy(dtype=np.int8)
            x_tune = scaler.transform(_feature_matrix(tune, columns))
            y_tune = tune["behavior_better"].to_numpy(dtype=np.int8)

            best = None
            for c_value in c_grid:
                if len(np.unique(y_train)) < 2:
                    model = None
                    tune_prob = np.full(len(tune), float(y_train.mean()), dtype=np.float32)
                else:
                    model = LogisticRegression(
                        C=c_value, solver="lbfgs", class_weight="balanced",
                        max_iter=1000, random_state=config.random_seed,
                    )
                    model.fit(x_train, y_train)
                    tune_prob = model.predict_proba(x_tune)[:, 1].astype(np.float32)
                score = -float(log_loss(y_tune, np.clip(tune_prob, 1e-6, 1 - 1e-6), labels=[0, 1]))
                tuning_rows.append({
                    "gate": gate_name, "test_window": fold["test"],
                    "parameter": "C", "value": c_value,
                    "validation_score": score,
                })
                if best is None or score > best[0]:
                    best = (score, c_value, model)
            assert best is not None
            _, best_c, best_model = best

            test_features = features.set_index(["window_id", "user_id"]).loc[
                [(fold["test"], int(uid)) for uid in test_user_ids]
            ].reset_index()
            x_test = scaler.transform(_feature_matrix(test_features, columns))
            if best_model is None:
                weights = np.full(len(test_user_ids), float(y_train.mean()), dtype=np.float32)
                coef_values = np.zeros(len(columns), dtype=float)
                intercept = float(y_train.mean())
            else:
                weights = best_model.predict_proba(x_test)[:, 1].astype(np.float32)
                coef_values = best_model.coef_[0]
                intercept = float(best_model.intercept_[0])
            # Users with fewer than two interactions are outside the gate's domain;
            # they receive the selected content ranking only.
            weights[test_counts < 2] = 0.0
            recs = fuse_arrays(
                test_candidates[selected_content], test_candidates["itemknn"],
                weights, config.recommendation_k, config.rrf_constant,
            )
            metrics = user_metrics(
                fold["test"], f"{gate_name}_itemknn_gate", test_user_ids,
                recs, test_truths, test_counts, config.evaluation_k,
            )
            metrics["primary_2plus"] = test_2plus
            metrics["selected_content_model"] = selected_content
            metrics["gate_weight"] = weights
            metrics["selected_C"] = float(best_c)
            metric_parts.append(metrics)

            for feature, coef in zip(columns, coef_values):
                coef_rows.append({
                    "gate": gate_name, "test_window": fold["test"],
                    "feature": feature, "coefficient_standardized": float(coef),
                    "selected_C": float(best_c),
                })
            coef_rows.append({
                "gate": gate_name, "test_window": fold["test"],
                "feature": "intercept", "coefficient_standardized": intercept,
                "selected_C": float(best_c),
            })

    gate_metrics = pd.concat(metric_parts, ignore_index=True)
    gate_metrics.to_parquet(gate_path, index=False, compression="zstd")
    coefficients = pd.DataFrame(coef_rows)
    tuning = pd.DataFrame(tuning_rows)
    folds_df = pd.DataFrame(fold_rows)
    coefficients.to_csv(coef_path, index=False)
    tuning.to_csv(tuning_path, index=False)
    folds_df.to_csv(fold_path, index=False)
    return gate_metrics, coefficients, tuning, folds_df


# ---------------------------------------------------------------------------
# Paper-ready statistics
# ---------------------------------------------------------------------------


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if len(p) == 0:
        return p
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def paired_bootstrap_ci(values: np.ndarray, repetitions: int, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    for i in range(repetitions):
        means[i] = values[rng.integers(0, len(values), len(values))].mean()
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def run_paper_analysis(
    config: Phase4Config,
    base_metrics: pd.DataFrame,
    gate_metrics: pd.DataFrame,
    features: pd.DataFrame,
    folds_df: pd.DataFrame,
    phase4_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_path = phase4_dir / "paper_model_summary.csv"
    tests_path = phase4_dir / "paper_statistical_tests.csv"
    regression_path = phase4_dir / "paper_regression_results.csv"
    quartile_path = phase4_dir / "paper_concentration_quartiles.csv"
    if all(p.exists() for p in [summary_path, tests_path, regression_path, quartile_path]):
        return (
            pd.read_csv(summary_path), pd.read_csv(tests_path),
            pd.read_csv(regression_path), pd.read_csv(quartile_path),
        )

    # Main paper population: chronological test windows and >=2 prior applications.
    main_base = base_metrics[
        base_metrics["window_id"].isin([5, 6, 7]) & (base_metrics["n_past_apps"] >= 2)
    ].copy()
    main_gate = gate_metrics[gate_metrics["primary_2plus"]].copy()
    all_main = pd.concat([main_base, main_gate], ignore_index=True)
    summary = (
        all_main.groupby("model", as_index=False)
        .agg(
            user_windows=("user_id", "size"),
            ndcg10=("ndcg10", "mean"),
            recall10=("recall10", "mean"),
            mrr=("mrr", "mean"),
        )
        .sort_values("ndcg10", ascending=False)
    )
    summary.to_csv(summary_path, index=False)

    # Paired gate comparisons against count-only.
    count_name = "count_itemknn_gate"
    test_rows: list[dict] = []
    for challenger in ["gte_concentration_itemknn_gate", "jobbert_concentration_itemknn_gate"]:
        pair = main_gate[main_gate["model"].isin([count_name, challenger])].pivot_table(
            index=["window_id", "user_id"], columns="model", values="ndcg10"
        ).dropna()
        diff = pair[challenger].to_numpy() - pair[count_name].to_numpy()
        mean, low, high = paired_bootstrap_ci(
            diff, config.bootstrap_repetitions, config.random_seed
        )
        try:
            p_value = float(wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue)
        except ValueError:
            p_value = 1.0
        test_rows.append({
            "reference": count_name, "challenger": challenger,
            "n_pairs": int(len(diff)), "mean_ndcg_difference": mean,
            "bootstrap_ci_low": low, "bootstrap_ci_high": high,
            "wilcoxon_p": p_value,
        })
    tests = pd.DataFrame(test_rows)
    if len(tests):
        tests["holm_adjusted_p"] = holm_adjust(tests["wilcoxon_p"].to_numpy())
    tests.to_csv(tests_path, index=False)

    # Build delta data using the content model selected for each fold/test window.
    selected_by_test = dict(zip(folds_df["test_window"].astype(int), folds_df["selected_content_model"]))
    delta_parts: list[pd.DataFrame] = []
    for test_window, content_model in selected_by_test.items():
        content = base_metrics[
            (base_metrics["window_id"] == test_window)
            & (base_metrics["model"] == content_model)
            & (base_metrics["n_past_apps"] >= 2)
        ][["window_id", "user_id", "ndcg10"]].rename(columns={"ndcg10": "content_ndcg"})
        item = base_metrics[
            (base_metrics["window_id"] == test_window)
            & (base_metrics["model"] == "itemknn")
            & (base_metrics["n_past_apps"] >= 2)
        ][["window_id", "user_id", "ndcg10"]].rename(columns={"ndcg10": "itemknn_ndcg"})
        part = content.merge(item, on=["window_id", "user_id"], how="inner")
        part["selected_content_model"] = content_model
        delta_parts.append(part)
    delta = pd.concat(delta_parts, ignore_index=True)
    delta = delta.merge(features, on=["window_id", "user_id"], how="inner")
    delta["delta_ndcg"] = delta["itemknn_ndcg"] - delta["content_ndcg"]
    delta["recency_log"] = np.log1p(delta["days_since_last_app"].fillna(3650.0))

    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise ImportError("statsmodels is required for final regression") from exc

    regression_rows: list[pd.DataFrame] = []
    for label, concentration in [
        ("gte", "gte_concentration"),
        ("jobbert", "jobbert_concentration"),
    ]:
        frame = delta.dropna(subset=[concentration]).copy()
        frame["concentration_value"] = frame[concentration].astype(float)
        formula = (
            "delta_ndcg ~ log1p_count + concentration_value + "
            "log1p_count:concentration_value + recency_log + "
            "profile_completeness + C(window_id)"
        )
        fit = smf.ols(formula, data=frame).fit(cov_type="HC3")
        result = pd.DataFrame({
            "representation": label,
            "term": fit.params.index,
            "coefficient": fit.params.values,
            "std_error_hc3": fit.bse.values,
            "p_value": fit.pvalues.values,
            "ci_low": fit.conf_int()[0].values,
            "ci_high": fit.conf_int()[1].values,
            "n_observations": int(fit.nobs),
            "r_squared": float(fit.rsquared),
        })
        regression_rows.append(result)
    regression = pd.concat(regression_rows, ignore_index=True)
    regression.to_csv(regression_path, index=False)

    # Within-count-group concentration quartiles.
    quartile_parts: list[pd.DataFrame] = []
    delta["count_group"] = delta["n_past_apps"].map(lambda n: _history_segment(int(n)))
    delta = delta[delta["count_group"].isin(["2", "3-4", "5-9", "10+"])].copy()
    for representation, concentration in [
        ("gte", "gte_concentration"),
        ("jobbert", "jobbert_concentration"),
    ]:
        frame = delta.dropna(subset=[concentration]).copy()
        # Rank-based qcut avoids duplicate-edge failures.
        group_keys = ["window_id", "count_group"]
        ranks = frame.groupby(group_keys, observed=True)[concentration].rank(method="first")
        sizes = frame.groupby(group_keys, observed=True)[concentration].transform("size")
        q_index = np.floor((ranks - 1) * 4 / sizes).clip(0, 3)
        labels = np.array(["Q1", "Q2", "Q3", "Q4"], dtype=object)
        frame["concentration_quartile"] = labels[q_index.astype(int).to_numpy()]
        frame.loc[sizes < 4, "concentration_quartile"] = pd.NA
        summary_q = (
            frame.dropna(subset=["concentration_quartile"])
            .groupby(["count_group", "concentration_quartile"], observed=True, as_index=False)
            .agg(
                user_windows=("user_id", "size"),
                mean_concentration=(concentration, "mean"),
                delta_ndcg=("delta_ndcg", "mean"),
                itemknn_ndcg=("itemknn_ndcg", "mean"),
                content_ndcg=("content_ndcg", "mean"),
            )
        )
        summary_q.insert(0, "representation", representation)
        quartile_parts.append(summary_q)
    quartiles = pd.concat(quartile_parts, ignore_index=True)
    quartiles.to_csv(quartile_path, index=False)
    delta.to_parquet(phase4_dir / "paper_delta_user_level.parquet", index=False, compression="zstd")
    return summary, tests, regression, quartiles


# ---------------------------------------------------------------------------
# Reporting, verification, orchestration
# ---------------------------------------------------------------------------


def write_report(
    config: Phase4Config,
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    regression: pd.DataFrame,
    quartiles: pd.DataFrame,
    diagnostics: pd.DataFrame,
    folds: pd.DataFrame,
    phase4_dir: Path,
) -> None:
    lines = [
        "# CareerBuilder Final Experiment Results",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(asdict(config), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Chronological content-model selection",
        "",
        folds.to_markdown(index=False),
        "",
        "## Main model performance (W5-W7, users with >=2 prior applications)",
        "",
        summary.to_markdown(index=False),
        "",
        "## Gate comparisons",
        "",
        tests.to_markdown(index=False) if len(tests) else "No comparisons available.",
        "",
        "## Candidate recall, coverage, and popularity diagnostics",
        "",
        diagnostics[diagnostics["window_id"].isin([5, 6, 7])].to_markdown(index=False),
        "",
        "## Regression results",
        "",
        regression.to_markdown(index=False),
        "",
        "## Concentration quartiles within application-count groups",
        "",
        quartiles.to_markdown(index=False),
        "",
        "## Interpretation guardrail",
        "",
        "The concentration analyses are associational, not causal. The primary population is users with at least two prior applications, because semantic concentration is undefined with fewer than two observations.",
    ]
    (phase4_dir / "final_paper_results.md").write_text("\n".join(lines), encoding="utf-8")


def verify_phase4(phase4_dir: Path) -> dict:
    expected = [
        phase4_dir / "user_metrics_base.parquet",
        phase4_dir / "user_metrics_final_gates.parquet",
        phase4_dir / "ranking_diagnostics.csv",
        phase4_dir / "fold_content_selection.csv",
        phase4_dir / "gate_coefficients.csv",
        phase4_dir / "gate_tuning.csv",
        phase4_dir / "paper_model_summary.csv",
        phase4_dir / "paper_statistical_tests.csv",
        phase4_dir / "paper_regression_results.csv",
        phase4_dir / "paper_concentration_quartiles.csv",
        phase4_dir / "paper_delta_user_level.parquet",
        phase4_dir / "final_paper_results.md",
    ]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        raise RuntimeError("Phase 4 outputs are missing:\n" + "\n".join(missing))
    summary = pd.read_csv(phase4_dir / "paper_model_summary.csv")
    gates = pd.read_parquet(phase4_dir / "user_metrics_final_gates.parquet")
    required_gates = {
        "count_itemknn_gate",
        "gte_concentration_itemknn_gate",
        "jobbert_concentration_itemknn_gate",
    }
    if not required_gates.issubset(set(gates["model"])):
        raise RuntimeError("Final gate models are incomplete")
    result = {
        "status": "complete",
        "paper_models": summary["model"].tolist(),
        "gate_models": sorted(required_gates),
        "primary_gate_rows_2plus": int(gates["primary_2plus"].sum()),
        "files": [str(p.relative_to(phase4_dir)) for p in expected],
    }
    _write_json(phase4_dir / "verification.json", result)
    return result


def create_return_bundle(base_dir: Path, phase4_dir: Path) -> Path:
    bundle = base_dir / "paper_results_bundle.zip"
    include = [
        "config.json", "verification.json", "phase4_summary.json",
        "final_paper_results.md", "paper_model_summary.csv",
        "paper_statistical_tests.csv", "paper_regression_results.csv",
        "paper_concentration_quartiles.csv", "fold_content_selection.csv",
        "ranking_diagnostics.csv", "gate_coefficients.csv", "gate_tuning.csv",
    ]
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in include:
            path = phase4_dir / name
            if path.exists():
                zf.write(path, arcname=name)
    return bundle


def run_phase4(config: Phase4Config) -> dict:
    started = time.time()
    data_dir = Path(config.data_dir)
    output_dir = Path(config.output_dir)
    base_dir = output_dir.parent
    cache_dir = Path(config.local_cache_dir)
    phase2_dir = output_dir / "semantic"
    phase3_dir = output_dir / "evaluation"
    phase4_dir = output_dir / "final"
    phase4_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _write_json(phase4_dir / "config.json", asdict(config))

    if not _is_complete(phase2_dir / "verification.json"):
        raise RuntimeError("Phase 2 is not complete")
    if not _is_complete(phase3_dir / "verification.json"):
        raise RuntimeError("Phase 3 is not complete")

    apps, users, windows = load_core_tables(data_dir)
    staged = stage_embedding_inputs(output_dir, cache_dir / "staged")
    active_cache_dir = cache_dir / "active_windows"

    localized_reports: dict[str, dict] = {}
    for window_id in range(1, 8):
        localized_reports[str(window_id)] = build_localized_candidates_for_window(
            config, window_id, apps, users, windows, staged,
            phase2_dir, phase4_dir, active_cache_dir,
        )

    base_metrics, diagnostics = evaluate_base_models(
        config, apps, windows, staged, phase2_dir, phase3_dir,
        phase4_dir, active_cache_dir,
    )
    features_path = phase3_dir / "user_features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(features_path)
    features = pd.read_parquet(features_path)
    gate_metrics, coefficients, tuning, folds = run_final_gates(
        config, base_metrics, features, phase2_dir, phase3_dir, phase4_dir
    )
    summary, tests, regression, quartiles = run_paper_analysis(
        config, base_metrics, gate_metrics, features, folds, phase4_dir
    )
    write_report(config, summary, tests, regression, quartiles, diagnostics, folds, phase4_dir)
    verification = verify_phase4(phase4_dir)
    run_summary = {
        "status": "complete",
        "seconds": round(time.time() - started, 2),
        "phase4_dir": str(phase4_dir),
        "localized_reports": localized_reports,
        "verification": verification,
    }
    _write_json(phase4_dir / "phase4_summary.json", run_summary)
    bundle = create_return_bundle(base_dir, phase4_dir)
    run_summary["return_bundle"] = str(bundle)
    _write_json(phase4_dir / "phase4_summary.json", run_summary)
    return run_summary


__all__ = [
    "Phase4Config", "run_phase4", "multi_rrf", "fuse_arrays",
    "filter_seen_topk", "holm_adjust", "paired_bootstrap_ci",
]
