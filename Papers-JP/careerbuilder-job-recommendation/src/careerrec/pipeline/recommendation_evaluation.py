from __future__ import annotations

"""Recommendation and chronological evaluation pipeline.

Consumes Phase 1 embeddings and Phase 2 GPU candidates/concentration outputs and
runs the CPU-heavy recommendation/evaluation stages:

- BM25 lexical retrieval with bm25s
- Item-KNN and implicit ALS behavior recommenders
- GTE/BM25 and GTE/ALS reciprocal-rank-fusion baselines
- rolling count-only and semantic-concentration gates
- user-level metrics, segment summaries, coverage/popularity diagnostics
- regression and paired statistical comparisons

All outputs are checkpointed under ``artifacts/evaluation`` so rerunning resumes
completed windows.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import gc
import json
import math
import os
import shutil
import time
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase3Config:
    data_dir: str
    output_dir: str
    local_cache_dir: str = ".cache/careerrec/evaluation"
    recommendation_k: int = 200
    evaluation_k: int = 10
    bm25_recommendation_k: int = 1000
    bm25_search_k: int = 1500
    bm25_query_batch_size: int = 256
    itemknn_k: int = 100
    itemknn_search_k: int = 700
    als_factors: int = 64
    als_regularization: float = 0.03
    als_alpha: float = 20.0
    als_iterations: int = 15
    behavior_batch_size: int = 512
    rrf_constant: int = 60
    bootstrap_repetitions: int = 1000
    random_seed: int = 42
    run_bm25: bool = True
    run_itemknn: bool = True
    run_als: bool = True
    run_analysis: bool = True


# ---------------------------------------------------------------------------
# Generic helpers
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


def _copy_if_needed(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    print(f"[stage] {src} -> {dst}")
    shutil.copy2(src, dst)


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


def _normalize_key(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def _finite_or(value: object, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


# ---------------------------------------------------------------------------
# Data loading and temporal views
# ---------------------------------------------------------------------------


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
    apps["ApplicationDate"] = pd.to_datetime(
        apps["ApplicationDate"], errors="coerce", utc=True
    )
    apps = apps.dropna(subset=["ApplicationDate"]).sort_values(
        "ApplicationDate", kind="mergesort"
    )

    users = pd.read_csv(data_dir / "users.tsv", sep="\t", low_memory=False)
    users["UserID"] = pd.to_numeric(users["UserID"], errors="raise").astype("int64")
    users["WindowID"] = pd.to_numeric(users["WindowID"], errors="raise").astype("int16")

    windows = pd.read_csv(data_dir / "window_dates.tsv", sep="\t")
    windows["Window"] = pd.to_numeric(windows["Window"], errors="raise").astype("int16")
    for column in ["Train Start", "Train End / Test Start", "Test End"]:
        windows[column] = pd.to_datetime(windows[column], errors="raise", utc=True)
    return apps, users, windows


def window_cutoff(windows: pd.DataFrame, window_id: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    row = windows.loc[windows["Window"].astype(int) == int(window_id)].iloc[0]
    return row["Train Start"] + pd.Timedelta(days=5), row["Train End / Test Start"]


@lru_cache(maxsize=8)
def load_phase2_window(phase2_dir: Path, window_id: int, top_k: int = 1000) -> dict:
    path = phase2_dir / "gte_candidates" / f"window_{window_id}_top{top_k}.npz"
    if not path.exists():
        matches = sorted((phase2_dir / "gte_candidates").glob(f"window_{window_id}_top*.npz"))
        if not matches:
            raise FileNotFoundError(f"No Phase 2 candidates for window {window_id}")
        path = matches[0]
    with np.load(path, allow_pickle=False) as data:
        return {key: np.array(data[key]) for key in data.files}


def truth_sets_from_ragged(offsets: np.ndarray, flat: np.ndarray) -> list[set[int]]:
    offsets = np.asarray(offsets, dtype=np.int64)
    flat = np.asarray(flat, dtype=np.int64)
    return [set(map(int, flat[offsets[i] : offsets[i + 1]])) for i in range(len(offsets) - 1)]


def build_past_sets(
    apps: pd.DataFrame,
    eval_user_ids: np.ndarray,
    cutoff: pd.Timestamp,
) -> tuple[dict[int, set[int]], dict[int, pd.Timestamp]]:
    eval_set = set(np.asarray(eval_user_ids, dtype=np.int64).tolist())
    past = apps[(apps["ApplicationDate"] < cutoff) & (apps["UserID"].isin(eval_set))]
    seen = {
        int(uid): set(group["JobID"].astype(np.int64).tolist())
        for uid, group in past.groupby("UserID", sort=False)
    }
    last_date = past.groupby("UserID")["ApplicationDate"].max().to_dict()
    return seen, {int(k): v for k, v in last_date.items()}


# ---------------------------------------------------------------------------
# Prepared job/user data staging
# ---------------------------------------------------------------------------


def stage_small_inputs(output_dir: Path, cache_dir: Path) -> dict[str, Path | list[Path]]:
    users_src = output_dir / "prepared" / "users" / "user_documents.parquet"
    users_dst = cache_dir / "prepared" / "users" / users_src.name
    _copy_if_needed(users_src, users_dst)

    job_srcs = sorted((output_dir / "prepared" / "jobs").glob("jobs_*.parquet"))
    if not job_srcs:
        raise RuntimeError("Prepared job parquet shards are missing")
    job_dsts: list[Path] = []
    for src in job_srcs:
        dst = cache_dir / "prepared" / "jobs" / src.name
        _copy_if_needed(src, dst)
        job_dsts.append(dst)
    return {"user_docs": users_dst, "job_parquets": job_dsts}


def load_active_jobs(
    job_parquets: Sequence[Path],
    window_id: int,
    cutoff: pd.Timestamp,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    wanted = list(
        columns
        or [
            "job_id",
            "job_text",
            "city",
            "state",
            "country",
            "window_id",
            "start_ts",
            "end_ts",
        ]
    )
    required = set(wanted) | {"job_id", "window_id", "start_ts", "end_ts"}
    cutoff_ts = int(cutoff.timestamp())
    parts = []
    for path in job_parquets:
        frame = pd.read_parquet(path, columns=sorted(required))
        mask = (
            (frame["window_id"].to_numpy(dtype=np.int16) == int(window_id))
            & (frame["start_ts"].to_numpy(dtype=np.int64) <= cutoff_ts)
            & (frame["end_ts"].to_numpy(dtype=np.int64) >= cutoff_ts)
        )
        if np.any(mask):
            parts.append(frame.loc[mask, wanted])
    if not parts:
        raise RuntimeError(f"No active jobs for window {window_id}")
    result = pd.concat(parts, ignore_index=True)
    result = result.drop_duplicates("job_id", keep="first").sort_values("job_id", kind="mergesort")
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ranking metrics and candidate diagnostics
# ---------------------------------------------------------------------------


def metrics_for_recommendations(
    window_id: int,
    user_ids: np.ndarray,
    recommendations: np.ndarray,
    truth_sets: Sequence[set[int]],
    history_counts: np.ndarray,
    model_name: str,
    k: int = 10,
) -> pd.DataFrame:
    user_ids = np.asarray(user_ids, dtype=np.int64)
    recommendations = np.asarray(recommendations)
    history_counts = np.asarray(history_counts, dtype=np.int32)
    if len(user_ids) != len(recommendations) or len(user_ids) != len(truth_sets):
        raise ValueError("Metric inputs have inconsistent row counts")

    rows = []
    for idx, uid in enumerate(user_ids):
        truth = truth_sets[idx]
        recs = [int(x) for x in recommendations[idx, :k] if int(x) >= 0]
        hit_positions = [rank for rank, jid in enumerate(recs, start=1) if jid in truth]
        hits = len(set(recs) & truth)
        ideal_len = min(len(truth), k)
        ideal = sum(1.0 / math.log2(i + 2) for i in range(ideal_len))
        dcg = sum(1.0 / math.log2(rank + 1) for rank in hit_positions)
        ndcg = dcg / ideal if ideal > 0 else np.nan
        recall = hits / len(truth) if truth else np.nan
        reciprocal_rank = 1.0 / min(hit_positions) if hit_positions else 0.0
        n_hist = int(history_counts[idx])
        rows.append(
            {
                "window_id": int(window_id),
                "user_id": int(uid),
                "model": model_name,
                "ndcg10": float(ndcg),
                "recall10": float(recall),
                "mrr": float(reciprocal_rank),
                "history_count": n_hist,
                "history_segment": _history_segment(n_hist),
                "truth_count": int(len(truth)),
                "recommendation_count": int(len(recs)),
            }
        )
    return pd.DataFrame(rows)


def candidate_recall(
    recommendations: np.ndarray,
    truth_sets: Sequence[set[int]],
    k: int,
) -> dict[str, float | int]:
    total_truth = 0
    total_hits = 0
    user_hits = 0
    for recs, truth in zip(recommendations[:, :k], truth_sets):
        rec_set = {int(x) for x in recs if int(x) >= 0}
        overlap = len(rec_set & truth)
        total_truth += len(truth)
        total_hits += overlap
        user_hits += int(overlap > 0)
    return {
        "truth_jobs": int(total_truth),
        "candidate_hits": int(total_hits),
        "candidate_recall": float(total_hits / total_truth) if total_truth else np.nan,
        "candidate_user_hit_rate": float(user_hits / len(truth_sets)) if truth_sets else np.nan,
    }


def ranking_diagnostics(
    recommendations: np.ndarray,
    active_job_ids: np.ndarray,
    past_popularity: Mapping[int, int],
    k: int = 10,
) -> dict[str, float | int]:
    top = np.asarray(recommendations[:, :k])
    valid = top[top >= 0].astype(np.int64, copy=False)
    unique_recommended = np.unique(valid) if len(valid) else np.array([], dtype=np.int64)
    average_popularity = (
        float(np.mean([past_popularity.get(int(jid), 0) for jid in valid]))
        if len(valid)
        else np.nan
    )
    return {
        "catalog_coverage_at_10": float(len(unique_recommended) / len(active_job_ids))
        if len(active_job_ids)
        else np.nan,
        "unique_jobs_at_10": int(len(unique_recommended)),
        "average_past_popularity_at_10": average_popularity,
        "user_coverage_at_10": float(np.mean(np.any(top >= 0, axis=1))) if len(top) else np.nan,
    }


# ---------------------------------------------------------------------------
# Popularity fallback
# ---------------------------------------------------------------------------


def build_location_popularity(
    active_jobs: pd.DataFrame,
    past_apps: pd.DataFrame,
    pool_size: int = 2000,
) -> dict:
    counts = past_apps["JobID"].value_counts().to_dict()
    frame = active_jobs[["job_id", "city", "state", "country"]].copy()
    frame["popularity"] = frame["job_id"].map(counts).fillna(0).astype(np.int64)
    frame = frame.sort_values(["popularity", "job_id"], ascending=[False, True], kind="mergesort")

    global_pool = frame["job_id"].head(pool_size).astype(np.int64).tolist()
    state_pools = {}
    city_pools = {}
    for key, group in frame.groupby(["country", "state"], dropna=False, sort=False):
        norm_key = tuple(_normalize_key(x) for x in key)
        state_pools[norm_key] = group["job_id"].head(pool_size).astype(np.int64).tolist()
    for key, group in frame.groupby(["country", "state", "city"], dropna=False, sort=False):
        norm_key = tuple(_normalize_key(x) for x in key)
        city_pools[norm_key] = group["job_id"].head(pool_size).astype(np.int64).tolist()
    return {
        "counts": {int(k): int(v) for k, v in counts.items()},
        "global": global_pool,
        "state": state_pools,
        "city": city_pools,
    }


def fill_with_location_popularity(
    existing: Sequence[int],
    seen: set[int],
    profile: Mapping[str, object],
    popularity: Mapping,
    top_k: int,
) -> list[int]:
    out: list[int] = []
    used = set(seen)
    for jid in existing:
        j = int(jid)
        if j < 0 or j in used:
            continue
        out.append(j)
        used.add(j)
        if len(out) >= top_k:
            return out

    country = _normalize_key(profile.get("Country", ""))
    state = _normalize_key(profile.get("State", ""))
    city = _normalize_key(profile.get("City", ""))
    pools = [
        popularity["city"].get((country, state, city), []),
        popularity["state"].get((country, state), []),
        popularity["global"],
    ]
    for pool in pools:
        for jid in pool:
            j = int(jid)
            if j in used:
                continue
            out.append(j)
            used.add(j)
            if len(out) >= top_k:
                return out
    return out


# ---------------------------------------------------------------------------
# BM25 lexical retrieval
# ---------------------------------------------------------------------------


def _bm25_dependencies():
    try:
        import bm25s  # type: ignore
        import Stemmer  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "BM25 dependencies are missing. Run: pip install 'bm25s[core]' PyStemmer"
        ) from exc
    return bm25s, Stemmer


def build_bm25_candidates_for_window(
    config: Phase3Config,
    window_id: int,
    apps: pd.DataFrame,
    users: pd.DataFrame,
    windows: pd.DataFrame,
    staged: Mapping,
    phase2_dir: Path,
    phase3_dir: Path,
) -> dict:
    out_dir = phase3_dir / "bm25_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"window_{window_id}_top{config.bm25_recommendation_k}.npz"
    report_path = out_dir / f"window_{window_id}.json"
    if out_path.exists() and _is_complete(report_path):
        print(f"[resume] BM25 window {window_id}")
        return _read_json(report_path)

    bm25s, Stemmer = _bm25_dependencies()
    phase2 = load_phase2_window(phase2_dir, window_id)
    user_ids = phase2["user_ids"].astype(np.int64)
    truth_offsets = phase2["truth_offsets"].astype(np.int64)
    truth_job_ids = phase2["truth_job_ids"].astype(np.int32)
    history_counts = phase2["n_past_apps"].astype(np.int32)
    truth_sets = truth_sets_from_ragged(truth_offsets, truth_job_ids)

    cutoff, _ = window_cutoff(windows, window_id)
    active = load_active_jobs(staged["job_parquets"], window_id, cutoff)
    active_job_ids = active["job_id"].to_numpy(dtype=np.int64)
    corpus_texts = active["job_text"].fillna("").astype(str).tolist()

    user_docs = pd.read_parquet(staged["user_docs"], columns=["user_id", "user_text"])
    query_map = user_docs.set_index("user_id")["user_text"]
    queries = [str(query_map.get(int(uid), "")) for uid in user_ids]

    eval_profiles = users.set_index("UserID").loc[user_ids]
    seen_by_user, _ = build_past_sets(apps, user_ids, cutoff)
    past_all = apps[apps["ApplicationDate"] < cutoff]
    popularity = build_location_popularity(active, past_all)

    extra_stopwords = {
        "recent",
        "roles",
        "role",
        "major",
        "degree",
        "years",
        "year",
        "experience",
        "location",
        "provided",
        "not",
        "job",
        "title",
        "requirements",
    }
    base_stopwords = set(getattr(bm25s.stopwords, "STOPWORDS_EN", []))
    tokenizer = bm25s.tokenization.Tokenizer(
        stopwords=sorted(base_stopwords | extra_stopwords),
        stemmer=Stemmer.Stemmer("english"),
    )
    print(f"[BM25] window {window_id}: tokenize/index {len(corpus_texts):,} active jobs")
    corpus_tokens = tokenizer.tokenize(corpus_texts, update_vocab=True, show_progress=True)
    retriever = bm25s.BM25(method="lucene", backend="numpy")
    retriever.index(corpus_tokens, show_progress=True)

    n_users = len(user_ids)
    rec_ids = np.full((n_users, config.bm25_recommendation_k), -1, dtype=np.int32)
    rec_scores = np.full((n_users, config.bm25_recommendation_k), -np.inf, dtype=np.float32)
    search_k = min(len(active_job_ids), max(config.bm25_search_k, config.bm25_recommendation_k))

    for start in range(0, n_users, config.bm25_query_batch_size):
        stop = min(start + config.bm25_query_batch_size, n_users)
        query_tokens = tokenizer.tokenize(
            queries[start:stop],
            update_vocab=False,
            show_progress=False,
        )
        result_ids, result_scores = retriever.retrieve(
            query_tokens,
            corpus=active_job_ids,
            k=search_k,
            show_progress=False,
            n_threads=0,
        )
        result_ids = np.asarray(result_ids, dtype=np.int64)
        result_scores = np.asarray(result_scores, dtype=np.float32)
        for local, uid in enumerate(user_ids[start:stop]):
            profile = eval_profiles.loc[int(uid)].to_dict()
            seen = seen_by_user.get(int(uid), set())
            existing = [int(x) for x in result_ids[local] if int(x) not in seen]
            filled = fill_with_location_popularity(
                existing, seen, profile, popularity, config.bm25_recommendation_k
            )
            n = len(filled)
            rec_ids[start + local, :n] = np.asarray(filled, dtype=np.int32)
            # Preserve BM25 score when available; fallback items receive decreasing rank score.
            score_map = {
                int(jid): float(score)
                for jid, score in zip(result_ids[local], result_scores[local])
            }
            rec_scores[start + local, :n] = np.asarray(
                [score_map.get(int(jid), -1e6 - rank) for rank, jid in enumerate(filled)],
                dtype=np.float32,
            )
        print(f"[BM25] window {window_id}: {stop:,}/{n_users:,} users")

    np.savez_compressed(
        out_path,
        user_ids=user_ids,
        candidate_job_ids=rec_ids,
        candidate_scores=rec_scores,
        truth_offsets=truth_offsets,
        truth_job_ids=truth_job_ids,
        n_past_apps=history_counts,
    )
    metrics = metrics_for_recommendations(
        window_id,
        user_ids,
        rec_ids,
        truth_sets,
        history_counts,
        "bm25",
        config.evaluation_k,
    )
    metrics_path = out_dir / f"window_{window_id}_user_metrics.parquet"
    metrics.to_parquet(metrics_path, index=False, compression="zstd")
    report = {
        "status": "complete",
        "window_id": int(window_id),
        "users": int(n_users),
        "active_jobs": int(len(active_job_ids)),
        "recommendation_k": int(config.bm25_recommendation_k),
        **candidate_recall(rec_ids, truth_sets, min(config.bm25_recommendation_k, 1000)),
        **ranking_diagnostics(rec_ids, active_job_ids, popularity["counts"], config.evaluation_k),
    }
    _write_json(report_path, report)
    del retriever, corpus_tokens, tokenizer, corpus_texts, active, user_docs
    gc.collect()
    return report


# ---------------------------------------------------------------------------
# Behavior models: Item-KNN and ALS
# ---------------------------------------------------------------------------


def _implicit_dependencies():
    try:
        import implicit  # type: ignore
    except ImportError as exc:
        raise ImportError("implicit is missing. Run: pip install implicit==0.7.2") from exc
    return implicit


def build_interaction_matrix(
    past_apps: pd.DataFrame,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, dict[int, int], dict[int, int]]:
    user_ids = np.sort(past_apps["UserID"].unique().astype(np.int64))
    job_ids = np.sort(past_apps["JobID"].unique().astype(np.int64))
    user_map = {int(uid): idx for idx, uid in enumerate(user_ids)}
    job_map = {int(jid): idx for idx, jid in enumerate(job_ids)}
    rows = past_apps["UserID"].map(user_map).to_numpy(dtype=np.int32)
    cols = past_apps["JobID"].map(job_map).to_numpy(dtype=np.int32)
    data = np.ones(len(past_apps), dtype=np.float32)
    matrix = sparse.coo_matrix(
        (data, (rows, cols)), shape=(len(user_ids), len(job_ids)), dtype=np.float32
    ).tocsr()
    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    return matrix, user_ids, job_ids, user_map, job_map


def _behavior_recommendations(
    model,
    model_name: str,
    interaction_matrix: sparse.csr_matrix,
    model_user_map: Mapping[int, int],
    model_job_ids: np.ndarray,
    model_job_map: Mapping[int, int],
    eval_user_ids: np.ndarray,
    active_job_ids: np.ndarray,
    seen_by_user: Mapping[int, set[int]],
    eval_profiles: pd.DataFrame,
    popularity: Mapping,
    config: Phase3Config,
) -> tuple[np.ndarray, np.ndarray]:
    n_users = len(eval_user_ids)
    out_ids = np.full((n_users, config.recommendation_k), -1, dtype=np.int32)
    out_scores = np.full((n_users, config.recommendation_k), -np.inf, dtype=np.float32)

    active_set = set(np.asarray(active_job_ids, dtype=np.int64).tolist())
    active_compact = np.asarray(
        [model_job_map[int(jid)] for jid in active_job_ids if int(jid) in model_job_map],
        dtype=np.int32,
    )

    for start in range(0, n_users, config.behavior_batch_size):
        stop = min(start + config.behavior_batch_size, n_users)
        for idx in range(start, stop):
            uid = int(eval_user_ids[idx])
            profile = eval_profiles.loc[uid].to_dict()
            seen = seen_by_user.get(uid, set())
            raw_global: list[int] = []
            raw_scores: list[float] = []
            if uid in model_user_map:
                row_id = int(model_user_map[uid])
                user_row = interaction_matrix[row_id]
                if model_name == "als" and len(active_compact):
                    ids, scores = model.recommend(
                        row_id,
                        user_row,
                        N=min(config.itemknn_search_k, len(active_compact)),
                        filter_already_liked_items=True,
                        items=active_compact,
                    )
                    valid_pairs = [(int(x), float(score)) for x, score in zip(ids, scores) if int(x) >= 0]
                    raw_global = [int(model_job_ids[x]) for x, _ in valid_pairs]
                    raw_scores = [score for _, score in valid_pairs]
                else:
                    ids, scores = model.recommend(
                        row_id,
                        user_row,
                        N=min(config.itemknn_search_k, interaction_matrix.shape[1]),
                        filter_already_liked_items=True,
                    )
                    for compact, score in zip(ids, scores):
                        if int(compact) < 0:
                            continue
                        jid = int(model_job_ids[int(compact)])
                        if jid in active_set:
                            raw_global.append(jid)
                            raw_scores.append(float(score))
                        if len(raw_global) >= config.recommendation_k:
                            break
            filled = fill_with_location_popularity(
                raw_global, seen, profile, popularity, config.recommendation_k
            )
            score_map = {jid: score for jid, score in zip(raw_global, raw_scores)}
            n = len(filled)
            out_ids[idx, :n] = np.asarray(filled, dtype=np.int32)
            out_scores[idx, :n] = np.asarray(
                [score_map.get(int(jid), -1e6 - rank) for rank, jid in enumerate(filled)],
                dtype=np.float32,
            )
        print(f"[{model_name}] {stop:,}/{n_users:,} users")
    return out_ids, out_scores


def build_behavior_candidates_for_window(
    config: Phase3Config,
    window_id: int,
    model_name: str,
    apps: pd.DataFrame,
    users: pd.DataFrame,
    windows: pd.DataFrame,
    staged: Mapping,
    phase2_dir: Path,
    phase3_dir: Path,
) -> dict:
    if model_name not in {"itemknn", "als"}:
        raise ValueError(model_name)
    out_dir = phase3_dir / f"{model_name}_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"window_{window_id}_top{config.recommendation_k}.npz"
    report_path = out_dir / f"window_{window_id}.json"
    if out_path.exists() and _is_complete(report_path):
        print(f"[resume] {model_name} window {window_id}")
        return _read_json(report_path)

    implicit = _implicit_dependencies()
    phase2 = load_phase2_window(phase2_dir, window_id)
    eval_user_ids = phase2["user_ids"].astype(np.int64)
    truth_offsets = phase2["truth_offsets"].astype(np.int64)
    truth_job_ids = phase2["truth_job_ids"].astype(np.int32)
    history_counts = phase2["n_past_apps"].astype(np.int32)
    truth_sets = truth_sets_from_ragged(truth_offsets, truth_job_ids)

    cutoff, _ = window_cutoff(windows, window_id)
    past_all = apps[apps["ApplicationDate"] < cutoff].copy()
    matrix, train_user_ids, train_job_ids, user_map, job_map = build_interaction_matrix(past_all)
    active = load_active_jobs(staged["job_parquets"], window_id, cutoff)
    active_job_ids = active["job_id"].to_numpy(dtype=np.int64)
    popularity = build_location_popularity(active, past_all)
    seen_by_user, _ = build_past_sets(apps, eval_user_ids, cutoff)
    eval_profiles = users.set_index("UserID").loc[eval_user_ids]

    print(
        f"[{model_name}] window {window_id}: fit {matrix.shape[0]:,} users x "
        f"{matrix.shape[1]:,} interacted jobs, nnz={matrix.nnz:,}"
    )
    if model_name == "itemknn":
        model = implicit.nearest_neighbours.CosineRecommender(
            K=config.itemknn_k, num_threads=0
        )
        model.fit(matrix, show_progress=True)
    else:
        model = implicit.als.AlternatingLeastSquares(
            factors=config.als_factors,
            regularization=config.als_regularization,
            alpha=config.als_alpha,
            iterations=config.als_iterations,
            num_threads=0,
            random_state=config.random_seed,
            calculate_training_loss=False,
        )
        model.fit(matrix, show_progress=True)

    rec_ids, rec_scores = _behavior_recommendations(
        model,
        model_name,
        matrix,
        user_map,
        train_job_ids,
        job_map,
        eval_user_ids,
        active_job_ids,
        seen_by_user,
        eval_profiles,
        popularity,
        config,
    )
    np.savez_compressed(
        out_path,
        user_ids=eval_user_ids,
        candidate_job_ids=rec_ids,
        candidate_scores=rec_scores,
        truth_offsets=truth_offsets,
        truth_job_ids=truth_job_ids,
        n_past_apps=history_counts,
    )
    metrics = metrics_for_recommendations(
        window_id,
        eval_user_ids,
        rec_ids,
        truth_sets,
        history_counts,
        model_name,
        config.evaluation_k,
    )
    metrics.to_parquet(
        out_dir / f"window_{window_id}_user_metrics.parquet",
        index=False,
        compression="zstd",
    )
    report = {
        "status": "complete",
        "window_id": int(window_id),
        "users": int(len(eval_user_ids)),
        "train_users": int(matrix.shape[0]),
        "train_jobs": int(matrix.shape[1]),
        "train_interactions": int(matrix.nnz),
        "active_jobs": int(len(active_job_ids)),
        **candidate_recall(rec_ids, truth_sets, min(config.recommendation_k, 1000)),
        **ranking_diagnostics(rec_ids, active_job_ids, popularity["counts"], config.evaluation_k),
    }
    _write_json(report_path, report)
    del model, matrix, active, past_all
    gc.collect()
    return report


# ---------------------------------------------------------------------------
# Rank fusion and model evaluation
# ---------------------------------------------------------------------------


def weighted_rrf(
    first: Sequence[int],
    second: Sequence[int],
    second_weight: float,
    top_k: int,
    constant: int = 60,
) -> list[int]:
    w = float(np.clip(second_weight, 0.0, 1.0))
    scores: dict[int, float] = {}
    for rank, jid in enumerate(first, start=1):
        j = int(jid)
        if j < 0:
            continue
        scores[j] = scores.get(j, 0.0) + (1.0 - w) / (constant + rank)
    for rank, jid in enumerate(second, start=1):
        j = int(jid)
        if j < 0:
            continue
        scores[j] = scores.get(j, 0.0) + w / (constant + rank)
    ordered = sorted(scores, key=lambda jid: (-scores[jid], jid))
    return ordered[:top_k]


def fuse_candidate_arrays(
    first: np.ndarray,
    second: np.ndarray,
    weights: np.ndarray | float,
    top_k: int,
    constant: int,
) -> np.ndarray:
    first = np.asarray(first)
    second = np.asarray(second)
    if np.isscalar(weights):
        weights = np.full(len(first), float(weights), dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if len(first) != len(second) or len(first) != len(weights):
        raise ValueError("RRF inputs have inconsistent row counts")
    out = np.full((len(first), top_k), -1, dtype=np.int32)
    for i in range(len(first)):
        fused = weighted_rrf(first[i], second[i], float(weights[i]), top_k, constant)
        out[i, : len(fused)] = np.asarray(fused, dtype=np.int32)
    return out


@lru_cache(maxsize=16)
def load_candidate_file(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: np.array(data[key]) for key in data.files}


def candidate_path_for_window(directory: Path, window_id: int) -> Path:
    matches = sorted(directory.glob(f"window_{window_id}_top*.npz"))
    if not matches:
        raise FileNotFoundError(f"No candidates in {directory} for window {window_id}")
    return matches[0]


def evaluate_base_and_static_models(
    config: Phase3Config,
    apps: pd.DataFrame,
    users: pd.DataFrame,
    windows: pd.DataFrame,
    staged: Mapping,
    phase2_dir: Path,
    phase3_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = phase3_dir / "user_metrics_base.parquet"
    diagnostics_path = phase3_dir / "ranking_diagnostics_base.csv"
    if metrics_path.exists() and diagnostics_path.exists():
        return pd.read_parquet(metrics_path), pd.read_csv(diagnostics_path)

    metric_parts = []
    diagnostics = []
    for window_id in range(1, 8):
        gte = load_phase2_window(phase2_dir, window_id)
        user_ids = gte["user_ids"].astype(np.int64)
        truth_sets = truth_sets_from_ragged(gte["truth_offsets"], gte["truth_job_ids"])
        history_counts = gte["n_past_apps"].astype(np.int32)
        cutoff, _ = window_cutoff(windows, window_id)
        active = load_active_jobs(staged["job_parquets"], window_id, cutoff)
        past_all = apps[apps["ApplicationDate"] < cutoff]
        past_pop = past_all["JobID"].value_counts().to_dict()
        active_ids = active["job_id"].to_numpy(dtype=np.int64)

        candidates: dict[str, np.ndarray] = {
            "gte": gte["candidate_job_ids"].astype(np.int32)
        }
        for model in ["bm25", "itemknn", "als"]:
            directory = phase3_dir / f"{model}_candidates"
            if directory.exists():
                try:
                    payload = load_candidate_file(candidate_path_for_window(directory, window_id))
                except FileNotFoundError:
                    continue
                if not np.array_equal(payload["user_ids"].astype(np.int64), user_ids):
                    raise RuntimeError(f"User order mismatch for {model} window {window_id}")
                candidates[model] = payload["candidate_job_ids"].astype(np.int32)

        if "bm25" in candidates:
            candidates["bm25_gte_static"] = fuse_candidate_arrays(
                candidates["gte"],
                candidates["bm25"],
                0.5,
                config.recommendation_k,
                config.rrf_constant,
            )
        if "als" in candidates:
            candidates["gte_als_static"] = fuse_candidate_arrays(
                candidates["gte"],
                candidates["als"],
                0.5,
                config.recommendation_k,
                config.rrf_constant,
            )

        for model_name, recs in candidates.items():
            metric_parts.append(
                metrics_for_recommendations(
                    window_id,
                    user_ids,
                    recs,
                    truth_sets,
                    history_counts,
                    model_name,
                    config.evaluation_k,
                )
            )
            diagnostics.append(
                {
                    "window_id": window_id,
                    "model": model_name,
                    **candidate_recall(recs, truth_sets, min(len(recs[0]), 1000)),
                    **ranking_diagnostics(recs, active_ids, past_pop, config.evaluation_k),
                }
            )

    metrics = pd.concat(metric_parts, ignore_index=True)
    metrics.to_parquet(metrics_path, index=False, compression="zstd")
    diag = pd.DataFrame(diagnostics)
    diag.to_csv(diagnostics_path, index=False)
    return metrics, diag


# ---------------------------------------------------------------------------
# Gate features, rolling training, and robustness gates
# ---------------------------------------------------------------------------


def build_user_feature_table(
    apps: pd.DataFrame,
    users: pd.DataFrame,
    windows: pd.DataFrame,
    phase2_dir: Path,
    phase3_dir: Path,
) -> pd.DataFrame:
    path = phase3_dir / "user_features.parquet"
    if path.exists():
        return pd.read_parquet(path)

    gte_conc = pd.read_parquet(phase2_dir / "gte_concentration.parquet")
    jobbert_conc = pd.read_parquet(phase2_dir / "jobbert_concentration.parquet")
    gte_conc = gte_conc.rename(
        columns={
            "centroid_cosine": "gte_centroid_cosine",
            "pairwise_cosine": "gte_pairwise_cosine",
            "n_embedded_apps": "gte_n_embedded_apps",
        }
    )
    jobbert_conc = jobbert_conc.rename(
        columns={
            "centroid_cosine": "jobbert_centroid_cosine",
            "pairwise_cosine": "jobbert_pairwise_cosine",
            "n_embedded_apps": "jobbert_n_embedded_apps",
        }
    )
    keep_jb = [
        "user_id",
        "window_id",
        "jobbert_centroid_cosine",
        "jobbert_pairwise_cosine",
        "jobbert_n_embedded_apps",
    ]
    features = gte_conc.merge(jobbert_conc[keep_jb], on=["user_id", "window_id"], how="left")

    profiles = users.copy()
    profile_columns = [
        column
        for column in [
            "Major",
            "DegreeType",
            "TotalYearsExperience",
            "City",
            "State",
            "Country",
        ]
        if column in profiles.columns
    ]
    completeness = np.zeros(len(profiles), dtype=np.float32)
    for column in profile_columns:
        values = profiles[column]
        completeness += (~values.isna() & values.astype(str).str.strip().ne("")).to_numpy(dtype=np.float32)
    if profile_columns:
        completeness /= len(profile_columns)
    profile_frame = pd.DataFrame(
        {
            "user_id": profiles["UserID"].astype(np.int64),
            "window_id": profiles["WindowID"].astype(np.int16),
            "profile_completeness": completeness,
        }
    )
    features = features.merge(profile_frame, on=["user_id", "window_id"], how="left")

    recency_parts = []
    for window_id in range(1, 8):
        cutoff, _ = window_cutoff(windows, window_id)
        window_users = users.loc[users["WindowID"] == window_id, "UserID"].astype(np.int64)
        past = apps[(apps["ApplicationDate"] < cutoff) & (apps["UserID"].isin(window_users))]
        last_dates = past.groupby("UserID")["ApplicationDate"].max()
        days = (cutoff - window_users.map(last_dates)).dt.total_seconds() / 86400.0
        days = days.fillna(3650.0).clip(lower=0.0, upper=3650.0)
        recency_parts.append(
            pd.DataFrame(
                {
                    "user_id": window_users.to_numpy(dtype=np.int64),
                    "window_id": np.full(len(window_users), window_id, dtype=np.int16),
                    "days_since_last_app": days.to_numpy(dtype=np.float32),
                }
            )
        )
    features = features.merge(
        pd.concat(recency_parts, ignore_index=True), on=["user_id", "window_id"], how="left"
    )
    features["log1p_count"] = np.log1p(features["n_past_apps"].fillna(0).astype(float))
    features["gte_has_concentration"] = (
        features["gte_n_embedded_apps"].fillna(0).astype(int) >= 2
    ).astype(np.int8)
    features["jobbert_has_concentration"] = (
        features["jobbert_n_embedded_apps"].fillna(0).astype(int) >= 2
    ).astype(np.int8)
    features["gte_concentration"] = features["gte_pairwise_cosine"].where(
        features["gte_has_concentration"].eq(1), np.nan
    )
    features["jobbert_concentration"] = features["jobbert_pairwise_cosine"].where(
        features["jobbert_has_concentration"].eq(1), np.nan
    )
    features["gte_count_x_concentration"] = (
        features["log1p_count"] * features["gte_concentration"].fillna(0.0)
    )
    features["jobbert_count_x_concentration"] = (
        features["log1p_count"] * features["jobbert_concentration"].fillna(0.0)
    )
    features.to_parquet(path, index=False, compression="zstd")
    return features


GATE_COLUMNS = {
    "count": ["log1p_count"],
    "gte_concentration": [
        "log1p_count",
        "gte_concentration",
        "gte_count_x_concentration",
        "gte_has_concentration",
        "days_since_last_app",
        "profile_completeness",
    ],
    "jobbert_concentration": [
        "log1p_count",
        "jobbert_concentration",
        "jobbert_count_x_concentration",
        "jobbert_has_concentration",
        "days_since_last_app",
        "profile_completeness",
    ],
}


def _prepare_gate_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    matrix = frame[list(columns)].copy()
    for column in columns:
        if "concentration" in column and "has_" not in column and "count_x" not in column:
            matrix[column] = matrix[column].fillna(0.0)
        elif column == "days_since_last_app":
            matrix[column] = matrix[column].fillna(3650.0)
        else:
            matrix[column] = matrix[column].fillna(0.0)
    return matrix.to_numpy(dtype=np.float64)


def _gate_rankings_for_window(
    config: Phase3Config,
    phase2_dir: Path,
    phase3_dir: Path,
    window_id: int,
    weights: np.ndarray,
) -> tuple[np.ndarray, dict]:
    gte = load_phase2_window(phase2_dir, window_id)
    als = load_candidate_file(candidate_path_for_window(phase3_dir / "als_candidates", window_id))
    if not np.array_equal(gte["user_ids"].astype(np.int64), als["user_ids"].astype(np.int64)):
        raise RuntimeError(f"GTE/ALS user mismatch in window {window_id}")
    counts = gte["n_past_apps"].astype(np.int32)
    weights = np.asarray(weights, dtype=np.float32).copy()
    weights[counts <= 0] = 0.0
    recs = fuse_candidate_arrays(
        gte["candidate_job_ids"],
        als["candidate_job_ids"],
        weights,
        config.recommendation_k,
        config.rrf_constant,
    )
    return recs, gte


def _mean_ndcg_for_recs(window_id: int, recs: np.ndarray, payload: Mapping, k: int) -> float:
    truth = truth_sets_from_ragged(payload["truth_offsets"], payload["truth_job_ids"])
    metrics = metrics_for_recommendations(
        window_id,
        payload["user_ids"],
        recs,
        truth,
        payload["n_past_apps"],
        "temporary",
        k,
    )
    return float(metrics["ndcg10"].mean())


def run_rolling_gates(
    config: Phase3Config,
    base_metrics: pd.DataFrame,
    features: pd.DataFrame,
    phase2_dir: Path,
    phase3_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gate_metrics_path = phase3_dir / "user_metrics_gates.parquet"
    coefficients_path = phase3_dir / "gate_coefficients.csv"
    tuning_path = phase3_dir / "gate_tuning.csv"
    if gate_metrics_path.exists() and coefficients_path.exists() and tuning_path.exists():
        return (
            pd.read_parquet(gate_metrics_path),
            pd.read_csv(coefficients_path),
            pd.read_csv(tuning_path),
        )

    wide = base_metrics[base_metrics["model"].isin(["gte", "als"])].pivot_table(
        index=["window_id", "user_id"], columns="model", values="ndcg10"
    ).reset_index()
    if "gte" not in wide.columns or "als" not in wide.columns:
        raise RuntimeError("GTE and ALS metrics are required for gate training")
    gate_data = features.merge(wide, on=["window_id", "user_id"], how="inner")
    gate_data["behavior_better"] = (gate_data["als"] > gate_data["gte"]).astype(np.int8)

    folds = [
        {"train": [1, 2, 3], "tune": 4, "test": 5},
        {"train": [1, 2, 3, 4], "tune": 5, "test": 6},
        {"train": [1, 2, 3, 4, 5], "tune": 6, "test": 7},
    ]
    c_grid = [0.01, 0.1, 1.0, 10.0]
    metric_parts = []
    coefficient_rows = []
    tuning_rows = []

    for gate_name, columns in GATE_COLUMNS.items():
        for fold in folds:
            train = gate_data[gate_data["window_id"].isin(fold["train"])].copy()
            tune = gate_data[gate_data["window_id"] == fold["tune"]].copy()
            test = gate_data[gate_data["window_id"] == fold["test"]].copy()
            scaler = StandardScaler()
            x_train = scaler.fit_transform(_prepare_gate_matrix(train, columns))
            y_train = train["behavior_better"].to_numpy(dtype=np.int8)
            tune_order = load_phase2_window(phase2_dir, fold["tune"])["user_ids"].astype(np.int64)
            tune_ordered = tune.set_index("user_id").loc[tune_order].reset_index()
            x_tune = scaler.transform(_prepare_gate_matrix(tune_ordered, columns))
            best = None
            for c_value in c_grid:
                if len(np.unique(y_train)) < 2:
                    constant_weight = float(y_train.mean())
                    tune_weights = np.full(len(tune_ordered), constant_weight, dtype=np.float32)
                    model = None
                else:
                    model = LogisticRegression(
                        C=c_value,
                        solver="lbfgs",
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=config.random_seed,
                    )
                    model.fit(x_train, y_train)
                    tune_weights = model.predict_proba(x_tune)[:, 1].astype(np.float32)
                recs, payload = _gate_rankings_for_window(
                    config, phase2_dir, phase3_dir, fold["tune"], tune_weights
                )
                tune_ndcg = _mean_ndcg_for_recs(
                    fold["tune"], recs, payload, config.evaluation_k
                )
                tuning_rows.append(
                    {
                        "gate": gate_name,
                        "train_windows": ",".join(map(str, fold["train"])),
                        "tune_window": fold["tune"],
                        "test_window": fold["test"],
                        "C": c_value,
                        "tune_ndcg10": tune_ndcg,
                    }
                )
                if best is None or tune_ndcg > best[0]:
                    best = (tune_ndcg, c_value, model)
            assert best is not None
            _, best_c, best_model = best
            expected_order = load_phase2_window(phase2_dir, fold["test"])["user_ids"].astype(np.int64)
            test_indexed = test.set_index("user_id").loc[expected_order]
            x_test_ordered = scaler.transform(_prepare_gate_matrix(test_indexed.reset_index(), columns))
            if best_model is None:
                test_weights_ordered = np.full(
                    len(test_indexed), float(y_train.mean()), dtype=np.float32
                )
            else:
                test_weights_ordered = best_model.predict_proba(x_test_ordered)[:, 1].astype(np.float32)
            recs, payload = _gate_rankings_for_window(
                config,
                phase2_dir,
                phase3_dir,
                fold["test"],
                test_weights_ordered,
            )
            truth = truth_sets_from_ragged(payload["truth_offsets"], payload["truth_job_ids"])
            model_name = f"{gate_name}_gate"
            part = metrics_for_recommendations(
                fold["test"],
                payload["user_ids"],
                recs,
                truth,
                payload["n_past_apps"],
                model_name,
                config.evaluation_k,
            )
            part["gate_weight"] = test_weights_ordered
            part["selected_C"] = best_c
            metric_parts.append(part)
            if best_model is None:
                coef_values = np.zeros(len(columns), dtype=float)
                intercept_value = float(y_train.mean())
            else:
                coef_values = best_model.coef_[0]
                intercept_value = float(best_model.intercept_[0])
            for name, coef in zip(columns, coef_values):
                coefficient_rows.append(
                    {
                        "gate": gate_name,
                        "test_window": fold["test"],
                        "feature": name,
                        "coefficient_standardized": float(coef),
                        "selected_C": best_c,
                    }
                )
            coefficient_rows.append(
                {
                    "gate": gate_name,
                    "test_window": fold["test"],
                    "feature": "intercept",
                    "coefficient_standardized": intercept_value,
                    "selected_C": best_c,
                }
            )

    gate_metrics = pd.concat(metric_parts, ignore_index=True)
    gate_metrics.to_parquet(gate_metrics_path, index=False, compression="zstd")
    coefficients = pd.DataFrame(coefficient_rows)
    coefficients.to_csv(coefficients_path, index=False)
    tuning = pd.DataFrame(tuning_rows)
    tuning.to_csv(tuning_path, index=False)
    return gate_metrics, coefficients, tuning


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def paired_bootstrap_ci(
    differences: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[float, float, float]:
    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    for i in range(repetitions):
        indices = rng.integers(0, len(values), size=len(values))
        means[i] = values[indices].mean()
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def run_regression_and_tests(
    config: Phase3Config,
    all_metrics: pd.DataFrame,
    features: pd.DataFrame,
    phase3_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    regression_path = phase3_dir / "regression_results.csv"
    tests_path = phase3_dir / "statistical_tests.csv"
    if regression_path.exists() and tests_path.exists():
        return pd.read_csv(regression_path), pd.read_csv(tests_path)

    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise ImportError("statsmodels is required for regression") from exc

    wide = all_metrics[all_metrics["model"].isin(["gte", "als"])].pivot_table(
        index=["window_id", "user_id"], columns="model", values="ndcg10"
    ).reset_index()
    regression = features.merge(wide, on=["window_id", "user_id"], how="inner")
    regression["delta_ndcg"] = regression["als"] - regression["gte"]
    regression["gte_concentration_filled"] = regression["gte_concentration"].fillna(0.0)
    regression["recency_log"] = np.log1p(regression["days_since_last_app"].fillna(3650.0))
    formula = (
        "delta_ndcg ~ log1p_count + gte_concentration_filled + "
        "log1p_count:gte_concentration_filled + gte_has_concentration + "
        "recency_log + profile_completeness + C(window_id)"
    )
    fit = smf.ols(formula, data=regression).fit(cov_type="HC3")
    regression_result = pd.DataFrame(
        {
            "term": fit.params.index,
            "coefficient": fit.params.values,
            "std_error_hc3": fit.bse.values,
            "p_value": fit.pvalues.values,
            "ci_low": fit.conf_int()[0].values,
            "ci_high": fit.conf_int()[1].values,
            "n_observations": int(fit.nobs),
            "r_squared": float(fit.rsquared),
        }
    )
    regression_result.to_csv(regression_path, index=False)

    gate_models = [
        "count_gate",
        "gte_concentration_gate",
        "jobbert_concentration_gate",
    ]
    comparison_models = [m for m in gate_models if m in set(all_metrics["model"])]
    test_rows = []
    if "count_gate" in comparison_models:
        for challenger in [m for m in comparison_models if m != "count_gate"]:
            for metric in ["ndcg10", "recall10", "mrr"]:
                pair = all_metrics[all_metrics["model"].isin(["count_gate", challenger])].pivot_table(
                    index=["window_id", "user_id"], columns="model", values=metric
                ).dropna()
                diff = pair[challenger].to_numpy() - pair["count_gate"].to_numpy()
                mean_diff, low, high = paired_bootstrap_ci(
                    diff, config.bootstrap_repetitions, config.random_seed
                )
                nonzero = diff[np.abs(diff) > 1e-15]
                if len(nonzero):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        p_value = float(
                            wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided").pvalue
                        )
                else:
                    p_value = 1.0
                test_rows.append(
                    {
                        "reference": "count_gate",
                        "challenger": challenger,
                        "metric": metric,
                        "n_pairs": int(len(diff)),
                        "mean_difference": mean_diff,
                        "bootstrap_ci_low": low,
                        "bootstrap_ci_high": high,
                        "wilcoxon_p": p_value,
                    }
                )
    tests = pd.DataFrame(test_rows)
    if len(tests):
        tests["holm_adjusted_p"] = holm_adjust(tests["wilcoxon_p"].to_numpy())
    tests.to_csv(tests_path, index=False)
    return regression_result, tests


# ---------------------------------------------------------------------------
# Summaries, report, verification
# ---------------------------------------------------------------------------


def build_summaries(
    metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    phase3_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = (
        metrics.groupby(["window_id", "model"], as_index=False)
        .agg(
            users=("user_id", "nunique"),
            ndcg10=("ndcg10", "mean"),
            recall10=("recall10", "mean"),
            mrr=("mrr", "mean"),
        )
        .sort_values(["window_id", "model"])
    )
    overall_available = (
        metrics.groupby("model", as_index=False)
        .agg(
            user_windows=("user_id", "size"),
            windows=("window_id", "nunique"),
            ndcg10=("ndcg10", "mean"),
            recall10=("recall10", "mean"),
            mrr=("mrr", "mean"),
        )
        .sort_values("ndcg10", ascending=False)
    )
    common_metrics = metrics[metrics["window_id"].isin([5, 6, 7])].copy()
    overall_common = (
        common_metrics.groupby("model", as_index=False)
        .agg(
            user_windows=("user_id", "size"),
            windows=("window_id", "nunique"),
            ndcg10=("ndcg10", "mean"),
            recall10=("recall10", "mean"),
            mrr=("mrr", "mean"),
        )
        .sort_values("ndcg10", ascending=False)
    )
    segments_available = (
        metrics.groupby(["model", "history_segment"], as_index=False)
        .agg(
            user_windows=("user_id", "size"),
            ndcg10=("ndcg10", "mean"),
            recall10=("recall10", "mean"),
            mrr=("mrr", "mean"),
        )
        .sort_values(["model", "history_segment"])
    )
    segments_common = (
        common_metrics.groupby(["model", "history_segment"], as_index=False)
        .agg(
            user_windows=("user_id", "size"),
            ndcg10=("ndcg10", "mean"),
            recall10=("recall10", "mean"),
            mrr=("mrr", "mean"),
        )
        .sort_values(["model", "history_segment"])
    )
    summary.to_csv(phase3_dir / "model_summary_by_window.csv", index=False)
    overall_available.to_csv(
        phase3_dir / "model_summary_overall_available_windows.csv", index=False
    )
    overall_common.to_csv(phase3_dir / "model_summary_overall.csv", index=False)
    overall_common.to_csv(
        phase3_dir / "model_summary_common_test_windows.csv", index=False
    )
    segments_available.to_csv(
        phase3_dir / "model_summary_by_history_segment_available_windows.csv", index=False
    )
    segments_common.to_csv(
        phase3_dir / "model_summary_by_history_segment.csv", index=False
    )
    diagnostics.to_csv(phase3_dir / "ranking_diagnostics.csv", index=False)
    return summary, overall_common, segments_common


def write_final_report(
    config: Phase3Config,
    overall: pd.DataFrame,
    segments: pd.DataFrame,
    tests: pd.DataFrame,
    phase3_dir: Path,
) -> None:
    top_models = overall.head(10).copy()
    lines = [
        "# CareerBuilder Phase 3 CPU Results",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(asdict(config), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Model performance on common chronological test windows (W5-W7)",
        "",
        top_models.to_markdown(index=False),
        "",
        "## Performance by application-history segment",
        "",
        segments.to_markdown(index=False),
        "",
        "## Gate statistical comparisons",
        "",
        tests.to_markdown(index=False) if len(tests) else "No gate comparison was available.",
        "",
        "## Interpretation guardrail",
        "",
        "The concentration regression is associational. It does not establish that changing semantic concentration causally changes recommender performance.",
    ]
    (phase3_dir / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


def verify_phase3(config: Phase3Config, phase3_dir: Path) -> dict:
    expected = [
        phase3_dir / "user_metrics_base.parquet",
        phase3_dir / "model_summary_by_window.csv",
        phase3_dir / "model_summary_overall.csv",
        phase3_dir / "model_summary_common_test_windows.csv",
        phase3_dir / "model_summary_overall_available_windows.csv",
        phase3_dir / "model_summary_by_history_segment.csv",
        phase3_dir / "ranking_diagnostics.csv",
        phase3_dir / "final_report.md",
    ]
    if config.run_analysis:
        expected.extend(
            [
                phase3_dir / "user_features.parquet",
                phase3_dir / "user_metrics_gates.parquet",
                phase3_dir / "gate_coefficients.csv",
                phase3_dir / "gate_tuning.csv",
                phase3_dir / "regression_results.csv",
                phase3_dir / "statistical_tests.csv",
            ]
        )
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise RuntimeError("Phase 3 outputs are missing:\n" + "\n".join(missing))
    metrics = pd.read_parquet(phase3_dir / "user_metrics_base.parquet")
    if metrics.empty or not {"gte", "als"}.issubset(set(metrics["model"])):
        raise RuntimeError("Base metrics are incomplete")
    result = {
        "status": "complete",
        "base_metric_rows": int(len(metrics)),
        "base_models": sorted(metrics["model"].unique().tolist()),
        "files": [str(path.relative_to(phase3_dir)) for path in expected],
    }
    if (phase3_dir / "user_metrics_gates.parquet").exists():
        gate = pd.read_parquet(phase3_dir / "user_metrics_gates.parquet")
        result["gate_metric_rows"] = int(len(gate))
        result["gate_models"] = sorted(gate["model"].unique().tolist())
    _write_json(phase3_dir / "verification.json", result)
    return result


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_phase3(config: Phase3Config) -> dict:
    started = time.time()
    data_dir = Path(config.data_dir)
    output_dir = Path(config.output_dir)
    cache_dir = Path(config.local_cache_dir)
    phase2_dir = output_dir / "semantic"
    phase3_dir = output_dir / "evaluation"
    phase3_dir.mkdir(parents=True, exist_ok=True)
    _write_json(phase3_dir / "config.json", asdict(config))

    if not _is_complete(phase2_dir / "verification.json"):
        raise RuntimeError("Phase 2 verification.json is missing or incomplete")

    apps, users, windows = load_core_tables(data_dir)
    staged = stage_small_inputs(output_dir, cache_dir)
    reports: dict[str, dict] = {}

    for window_id in range(1, 8):
        if config.run_bm25:
            reports[f"bm25_w{window_id}"] = build_bm25_candidates_for_window(
                config,
                window_id,
                apps,
                users,
                windows,
                staged,
                phase2_dir,
                phase3_dir,
            )
        if config.run_itemknn:
            reports[f"itemknn_w{window_id}"] = build_behavior_candidates_for_window(
                config,
                window_id,
                "itemknn",
                apps,
                users,
                windows,
                staged,
                phase2_dir,
                phase3_dir,
            )
        if config.run_als:
            reports[f"als_w{window_id}"] = build_behavior_candidates_for_window(
                config,
                window_id,
                "als",
                apps,
                users,
                windows,
                staged,
                phase2_dir,
                phase3_dir,
            )

    base_metrics, diagnostics = evaluate_base_and_static_models(
        config, apps, users, windows, staged, phase2_dir, phase3_dir
    )
    all_metrics = base_metrics.copy()
    gate_metrics = pd.DataFrame()
    regression = pd.DataFrame()
    tests = pd.DataFrame()
    if config.run_analysis:
        if "als" not in set(base_metrics["model"]):
            raise RuntimeError("ALS results are required for gates and delta-NDCG analysis")
        features = build_user_feature_table(apps, users, windows, phase2_dir, phase3_dir)
        gate_metrics, _, _ = run_rolling_gates(
            config, base_metrics, features, phase2_dir, phase3_dir
        )
        all_metrics = pd.concat([base_metrics, gate_metrics], ignore_index=True)
        all_metrics.to_parquet(
            phase3_dir / "user_metrics_all.parquet", index=False, compression="zstd"
        )
        regression, tests = run_regression_and_tests(
            config, all_metrics, features, phase3_dir
        )

    summary, overall, segments = build_summaries(all_metrics, diagnostics, phase3_dir)
    write_final_report(config, overall, segments, tests, phase3_dir)
    verification = verify_phase3(config, phase3_dir)
    run_summary = {
        "status": "complete",
        "seconds": round(time.time() - started, 2),
        "phase3_dir": str(phase3_dir),
        "verification": verification,
        "reports": reports,
    }
    _write_json(phase3_dir / "phase3_summary.json", run_summary)
    return run_summary


__all__ = [
    "Phase3Config",
    "run_phase3",
    "weighted_rrf",
    "fuse_candidate_arrays",
    "metrics_for_recommendations",
    "truth_sets_from_ragged",
    "holm_adjust",
]
