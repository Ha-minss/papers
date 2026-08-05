"""Build semantic candidates and application-history concentration features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable
import json
import shutil
import time

import numpy as np
import pandas as pd


def _dense_index(ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size == 0:
        return np.full(1, -1, dtype=np.int64)
    if ids.min() < 0:
        raise ValueError('IDs must be non-negative')
    out = np.full(int(ids.max()) + 1, -1, dtype=np.int64)
    out[ids] = np.arange(len(ids), dtype=np.int64)
    return out


def temporal_views(
    users: pd.DataFrame,
    apps: pd.DataFrame,
    windows: pd.DataFrame,
    window_id: int,
    active_job_ids: np.ndarray,
) -> dict:
    row = windows.loc[windows['Window'].astype(int) == int(window_id)].iloc[0]
    train_start = pd.Timestamp(row['Train Start'])
    train_end = pd.Timestamp(row['Train End / Test Start'])
    if train_start.tzinfo is None:
        train_start = train_start.tz_localize('UTC')
    else:
        train_start = train_start.tz_convert('UTC')
    if train_end.tzinfo is None:
        train_end = train_end.tz_localize('UTC')
    else:
        train_end = train_end.tz_convert('UTC')
    cutoff = train_start + pd.Timedelta(days=5)

    window_users = users.loc[users['WindowID'].astype(int) == int(window_id), 'UserID'].astype(np.int64)
    user_set = set(window_users.tolist())
    active_set = set(np.asarray(active_job_ids, dtype=np.int64).tolist())

    apps_w_users = apps[apps['UserID'].isin(user_set)]
    future = apps_w_users[
        (apps_w_users['WindowID'].astype(int) == int(window_id))
        & (apps_w_users['ApplicationDate'] >= cutoff)
        & (apps_w_users['ApplicationDate'] < train_end)
        & (apps_w_users['JobID'].isin(active_set))
    ]
    truth_by_user = {
        int(uid): sorted(set(group['JobID'].astype(np.int64).tolist()))
        for uid, group in future.groupby('UserID', sort=True)
    }
    eval_user_ids = np.array(sorted(truth_by_user), dtype=np.int64)

    past = apps_w_users[apps_w_users['ApplicationDate'] < cutoff]
    past_by_user = {
        int(uid): set(group['JobID'].astype(np.int64).tolist())
        for uid, group in past.groupby('UserID', sort=False)
    }

    return {
        'cutoff': cutoff,
        'validation_end': train_end,
        'eval_user_ids': eval_user_ids,
        'truth_by_user': truth_by_user,
        'past_by_user': past_by_user,
    }


def compute_concentration_from_events(
    target_user_ids: np.ndarray,
    event_user_ids: np.ndarray,
    event_embedding_rows: np.ndarray,
    embedding_matrix: np.ndarray,
    chunk_size: int = 50_000,
) -> dict[str, np.ndarray]:
    target_user_ids = np.asarray(target_user_ids, dtype=np.int64)
    event_user_ids = np.asarray(event_user_ids, dtype=np.int64)
    event_embedding_rows = np.asarray(event_embedding_rows, dtype=np.int64)
    if len(event_user_ids) != len(event_embedding_rows):
        raise ValueError('event arrays must have the same length')

    user_index = _dense_index(target_user_ids)
    n_users = len(target_user_ids)
    dim = int(embedding_matrix.shape[1])
    sums = np.zeros((n_users, dim), dtype=np.float32)
    counts = np.zeros(n_users, dtype=np.int32)
    sum_sq_norms = np.zeros(n_users, dtype=np.float32)

    for start in range(0, len(event_user_ids), chunk_size):
        stop = min(start + chunk_size, len(event_user_ids))
        uids = event_user_ids[start:stop]
        rows = event_embedding_rows[start:stop]
        valid = (
            (uids >= 0)
            & (uids < len(user_index))
            & (rows >= 0)
            & (rows < len(embedding_matrix))
        )
        if not np.any(valid):
            continue
        local_users = user_index[uids[valid]]
        valid2 = local_users >= 0
        if not np.any(valid2):
            continue
        local_users = local_users[valid2]
        vectors = np.asarray(embedding_matrix[rows[valid][valid2]], dtype=np.float32)
        np.add.at(sums, local_users, vectors)
        np.add.at(counts, local_users, 1)
        np.add.at(sum_sq_norms, local_users, np.einsum('ij,ij->i', vectors, vectors))

    sum_norms = np.linalg.norm(sums, axis=1)
    centroid = np.full(n_users, np.nan, dtype=np.float32)
    has_any = counts > 0
    centroid[has_any] = sum_norms[has_any] / counts[has_any]

    pairwise = np.full(n_users, np.nan, dtype=np.float32)
    has_pair = counts > 1
    pairwise[has_pair] = (
        np.square(sum_norms[has_pair]) - sum_sq_norms[has_pair]
    ) / (counts[has_pair] * (counts[has_pair] - 1))

    return {
        'n_embedded_apps': counts,
        'centroid_cosine': centroid,
        'pairwise_cosine': pairwise,
    }


def filter_seen_topk(
    candidate_ids: np.ndarray,
    candidate_scores: np.ndarray,
    seen_job_ids: set[int],
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    kept_ids = []
    kept_scores = []
    for jid, score in zip(candidate_ids, candidate_scores):
        jid_int = int(jid)
        if jid_int < 0 or jid_int in seen_job_ids:
            continue
        kept_ids.append(jid_int)
        kept_scores.append(float(score))
        if len(kept_ids) >= top_k:
            break
    return np.asarray(kept_ids, dtype=np.int64), np.asarray(kept_scores, dtype=np.float32)

@dataclass(frozen=True)
class Phase2Config:
    data_dir: str
    output_dir: str
    local_cache_dir: str = '.cache/careerrec/semantic'
    top_k: int = 1000
    query_batch_size: int = 128
    concentration_chunk_size: int = 25_000


def _copy_if_needed(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    print(f'[stage] {src.name} -> {dst}')
    shutil.copy2(src, dst)


def stage_phase2_inputs(output_dir: Path, local_cache_dir: Path) -> dict[str, list[Path] | Path]:
    source_jobs = sorted((output_dir / 'prepared' / 'jobs').glob('jobs_*.parquet'))
    source_gte_ids = sorted((output_dir / 'embeddings' / 'gte_jobs').glob('shard_*_ids.npy'))
    source_gte_emb = sorted((output_dir / 'embeddings' / 'gte_jobs').glob('shard_*_embeddings.npy'))
    if not source_jobs or not (len(source_jobs) == len(source_gte_ids) == len(source_gte_emb)):
        raise RuntimeError('Prepared job shards and GTE shards are incomplete or misaligned')

    local_jobs_dir = local_cache_dir / 'prepared_jobs'
    local_gte_jobs_dir = local_cache_dir / 'gte_jobs'
    local_users_dir = local_cache_dir / 'gte_users'
    local_jobbert_dir = local_cache_dir / 'jobbert_app_titles'

    local_jobs, local_gte_ids, local_gte_emb = [], [], []
    for src in source_jobs:
        dst = local_jobs_dir / src.name
        _copy_if_needed(src, dst)
        local_jobs.append(dst)
    for src in source_gte_ids:
        dst = local_gte_jobs_dir / src.name
        _copy_if_needed(src, dst)
        local_gte_ids.append(dst)
    for src in source_gte_emb:
        dst = local_gte_jobs_dir / src.name
        _copy_if_needed(src, dst)
        local_gte_emb.append(dst)

    user_ids_src = output_dir / 'embeddings' / 'gte_users' / 'shard_00000_ids.npy'
    user_emb_src = output_dir / 'embeddings' / 'gte_users' / 'shard_00000_embeddings.npy'
    jobbert_ids_src = output_dir / 'embeddings' / 'jobbert_app_titles' / 'shard_00000_ids.npy'
    jobbert_emb_src = output_dir / 'embeddings' / 'jobbert_app_titles' / 'shard_00000_embeddings.npy'
    for src in [user_ids_src, user_emb_src, jobbert_ids_src, jobbert_emb_src]:
        if not src.exists():
            raise FileNotFoundError(src)

    user_ids = local_users_dir / user_ids_src.name
    user_emb = local_users_dir / user_emb_src.name
    jobbert_ids = local_jobbert_dir / jobbert_ids_src.name
    jobbert_emb = local_jobbert_dir / jobbert_emb_src.name
    _copy_if_needed(user_ids_src, user_ids)
    _copy_if_needed(user_emb_src, user_emb)
    _copy_if_needed(jobbert_ids_src, jobbert_ids)
    _copy_if_needed(jobbert_emb_src, jobbert_emb)

    return {
        'job_parquets': local_jobs,
        'gte_job_ids': local_gte_ids,
        'gte_job_embeddings': local_gte_emb,
        'gte_user_ids': user_ids,
        'gte_user_embeddings': user_emb,
        'jobbert_ids': jobbert_ids,
        'jobbert_embeddings': jobbert_emb,
    }


def _read_core_tables(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    apps = pd.read_csv(
        data_dir / 'apps.tsv', sep='\t',
        usecols=['UserID', 'WindowID', 'ApplicationDate', 'JobID'],
        low_memory=False,
    )
    apps['UserID'] = pd.to_numeric(apps['UserID'], errors='raise').astype('int64')
    apps['WindowID'] = pd.to_numeric(apps['WindowID'], errors='raise').astype('int16')
    apps['JobID'] = pd.to_numeric(apps['JobID'], errors='raise').astype('int64')
    apps['ApplicationDate'] = pd.to_datetime(apps['ApplicationDate'], errors='coerce', utc=True)
    apps = apps.dropna(subset=['ApplicationDate']).sort_values('ApplicationDate', kind='mergesort')

    users = pd.read_csv(
        data_dir / 'users.tsv', sep='\t',
        usecols=['UserID', 'WindowID', 'City', 'State', 'Country'],
        low_memory=False,
    )
    users['UserID'] = pd.to_numeric(users['UserID'], errors='raise').astype('int64')
    users['WindowID'] = pd.to_numeric(users['WindowID'], errors='raise').astype('int16')

    windows = pd.read_csv(data_dir / 'window_dates.tsv', sep='\t')
    for col in ['Train Start', 'Train End / Test Start', 'Test End']:
        windows[col] = pd.to_datetime(windows[col], errors='raise', utc=True)
    windows['Window'] = pd.to_numeric(windows['Window'], errors='raise').astype('int16')
    return apps, users, windows


def _window_cutoff(windows: pd.DataFrame, window_id: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    row = windows.loc[windows['Window'] == int(window_id)].iloc[0]
    return row['Train Start'] + pd.Timedelta(days=5), row['Train End / Test Start']


def build_active_window_cache(
    window_id: int,
    cutoff: pd.Timestamp,
    staged: dict,
    cache_dir: Path,
) -> tuple[Path, Path]:
    ids_path = cache_dir / f'window_{window_id}_active_job_ids.npy'
    emb_path = cache_dir / f'window_{window_id}_active_gte_embeddings.npy'
    meta_path = cache_dir / f'window_{window_id}_active.json'
    if ids_path.exists() and emb_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get('status') == 'complete':
            return ids_path, emb_path

    cutoff_ts = int(cutoff.timestamp())
    id_chunks, emb_chunks = [], []
    for parquet_path, ids_shard_path, emb_shard_path in zip(
        staged['job_parquets'], staged['gte_job_ids'], staged['gte_job_embeddings']
    ):
        frame = pd.read_parquet(
            parquet_path, columns=['job_id', 'window_id', 'start_ts', 'end_ts']
        )
        ids = np.load(ids_shard_path, mmap_mode='r')
        emb = np.load(emb_shard_path, mmap_mode='r')
        if len(frame) != len(ids) or len(ids) != len(emb):
            raise RuntimeError(f'Shard length mismatch: {parquet_path.name}')
        frame_ids = frame['job_id'].to_numpy(dtype=np.int64)
        if not np.array_equal(frame_ids, np.asarray(ids, dtype=np.int64)):
            raise RuntimeError(f'Job order mismatch: {parquet_path.name}')
        mask = (
            (frame['window_id'].to_numpy(dtype=np.int16) == int(window_id))
            & (frame['start_ts'].to_numpy(dtype=np.int64) <= cutoff_ts)
            & (frame['end_ts'].to_numpy(dtype=np.int64) >= cutoff_ts)
        )
        if np.any(mask):
            id_chunks.append(frame_ids[mask])
            emb_chunks.append(np.asarray(emb[mask], dtype=np.float16))

    if not id_chunks:
        raise RuntimeError(f'No active jobs found for window {window_id}')
    active_ids = np.concatenate(id_chunks).astype(np.int64, copy=False)
    active_emb = np.concatenate(emb_chunks).astype(np.float16, copy=False)
    order = np.argsort(active_ids, kind='mergesort')
    active_ids = active_ids[order]
    active_emb = active_emb[order]
    if len(np.unique(active_ids)) != len(active_ids):
        raise RuntimeError(f'Duplicate active job IDs in window {window_id}')

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(ids_path, active_ids, allow_pickle=False)
    np.save(emb_path, active_emb, allow_pickle=False)
    meta_path.write_text(json.dumps({
        'status': 'complete', 'window_id': int(window_id),
        'cutoff': cutoff.isoformat(), 'rows': int(len(active_ids)),
        'embedding_dim': int(active_emb.shape[1]),
    }, indent=2))
    return ids_path, emb_path


def build_gte_application_cache(
    apps: pd.DataFrame,
    staged: dict,
    cache_dir: Path,
) -> tuple[Path, Path]:
    ids_path = cache_dir / 'gte_application_job_ids.npy'
    emb_path = cache_dir / 'gte_application_job_embeddings.npy'
    meta_path = cache_dir / 'gte_application_jobs.json'
    if ids_path.exists() and emb_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get('status') == 'complete':
            return ids_path, emb_path

    app_ids = np.sort(apps['JobID'].unique().astype(np.int64))
    max_id = int(app_ids.max())
    wanted = np.zeros(max_id + 1, dtype=np.bool_)
    wanted[app_ids] = True
    id_chunks, emb_chunks = [], []
    for ids_shard_path, emb_shard_path in zip(staged['gte_job_ids'], staged['gte_job_embeddings']):
        ids = np.load(ids_shard_path, mmap_mode='r')
        emb = np.load(emb_shard_path, mmap_mode='r')
        valid_range = (ids >= 0) & (ids <= max_id)
        mask = np.zeros(len(ids), dtype=np.bool_)
        mask[valid_range] = wanted[np.asarray(ids[valid_range], dtype=np.int64)]
        if np.any(mask):
            id_chunks.append(np.asarray(ids[mask], dtype=np.int64))
            emb_chunks.append(np.asarray(emb[mask], dtype=np.float16))
    found_ids = np.concatenate(id_chunks)
    found_emb = np.concatenate(emb_chunks)
    order = np.argsort(found_ids, kind='mergesort')
    found_ids, found_emb = found_ids[order], found_emb[order]
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(ids_path, found_ids, allow_pickle=False)
    np.save(emb_path, found_emb, allow_pickle=False)
    meta_path.write_text(json.dumps({
        'status': 'complete', 'rows': int(len(found_ids)),
        'requested_unique_jobs': int(len(app_ids)),
        'missing_jobs': int(len(app_ids) - len(found_ids)),
        'embedding_dim': int(found_emb.shape[1]),
    }, indent=2))
    return ids_path, emb_path


def _embedding_rows_for_job_ids(job_ids: np.ndarray, embedding_ids: np.ndarray) -> np.ndarray:
    lookup = _dense_index(np.asarray(embedding_ids, dtype=np.int64))
    job_ids = np.asarray(job_ids, dtype=np.int64)
    rows = np.full(len(job_ids), -1, dtype=np.int64)
    valid = (job_ids >= 0) & (job_ids < len(lookup))
    rows[valid] = lookup[job_ids[valid]]
    return rows


def _total_past_counts(target_user_ids: np.ndarray, past: pd.DataFrame) -> np.ndarray:
    counts_by_user = past.groupby('UserID').size()
    return pd.Series(target_user_ids).map(counts_by_user).fillna(0).to_numpy(dtype=np.int32)


def build_concentration_table(
    model_name: str,
    embedding_ids_path: Path,
    embedding_path: Path,
    apps: pd.DataFrame,
    users: pd.DataFrame,
    windows: pd.DataFrame,
    output_path: Path,
    chunk_size: int,
) -> dict:
    if output_path.exists():
        frame = pd.read_parquet(output_path, columns=['user_id'])
        if len(frame) == len(users):
            return {'status': 'complete', 'rows': int(len(frame)), 'model': model_name, 'resumed': True}

    embedding_ids = np.load(embedding_ids_path, mmap_mode='r')
    embeddings = np.load(embedding_path, mmap_mode='r')
    all_parts = []
    for window_id in sorted(users['WindowID'].unique().astype(int)):
        cutoff, _ = _window_cutoff(windows, window_id)
        window_users = users.loc[users['WindowID'] == window_id, 'UserID'].to_numpy(dtype=np.int64)
        user_set = set(window_users.tolist())
        past = apps[(apps['UserID'].isin(user_set)) & (apps['ApplicationDate'] < cutoff)]
        rows = _embedding_rows_for_job_ids(past['JobID'].to_numpy(dtype=np.int64), embedding_ids)
        result = compute_concentration_from_events(
            target_user_ids=window_users,
            event_user_ids=past['UserID'].to_numpy(dtype=np.int64),
            event_embedding_rows=rows,
            embedding_matrix=embeddings,
            chunk_size=chunk_size,
        )
        total_counts = _total_past_counts(window_users, past)
        part = pd.DataFrame({
            'user_id': window_users,
            'window_id': np.full(len(window_users), window_id, dtype=np.int16),
            'cutoff': pd.Timestamp(cutoff),
            'n_past_apps': total_counts,
            'n_embedded_apps': result['n_embedded_apps'],
            'centroid_cosine': result['centroid_cosine'],
            'pairwise_cosine': result['pairwise_cosine'],
        })
        all_parts.append(part)
        print(f'[concentration:{model_name}] window {window_id}: {len(part):,} users')
    result_frame = pd.concat(all_parts, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_parquet(output_path, index=False, compression='zstd')
    return {
        'status': 'complete', 'rows': int(len(result_frame)), 'model': model_name,
        'users_with_2plus_embedded_apps': int((result_frame['n_embedded_apps'] >= 2).sum()),
    }


def _load_user_queries(
    eval_user_ids: np.ndarray,
    user_embedding_ids: np.ndarray,
    user_embeddings: np.ndarray,
) -> np.ndarray:
    rows = _embedding_rows_for_job_ids(eval_user_ids, user_embedding_ids)
    if np.any(rows < 0):
        missing = eval_user_ids[rows < 0][:10].tolist()
        raise RuntimeError(f'Missing user embeddings, examples: {missing}')
    return np.asarray(user_embeddings[rows], dtype=np.float16)


def retrieve_topk_torch(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidate_job_ids: np.ndarray,
    query_user_ids: np.ndarray,
    past_by_user: dict[int, set[int]],
    top_k: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor_dtype = torch.float16 if device.type == 'cuda' else torch.float32
    candidate_tensor = torch.as_tensor(candidate_embeddings, device=device, dtype=tensor_dtype)
    candidate_job_ids = np.asarray(candidate_job_ids, dtype=np.int64)
    max_seen = max((len(past_by_user.get(int(uid), set())) for uid in query_user_ids), default=0)
    search_k = min(len(candidate_job_ids), top_k + max(128, min(max_seen, 2000)))
    out_ids = np.full((len(query_user_ids), top_k), -1, dtype=np.int32)
    out_scores = np.full((len(query_user_ids), top_k), -np.inf, dtype=np.float16)

    current_batch = int(batch_size)
    start = 0
    while start < len(query_user_ids):
        stop = min(start + current_batch, len(query_user_ids))
        try:
            q = torch.as_tensor(query_embeddings[start:stop], device=device, dtype=tensor_dtype)
            scores = q @ candidate_tensor.T
            values, indices = torch.topk(scores, k=search_k, dim=1, largest=True, sorted=True)
            ids_np = candidate_job_ids[indices.detach().cpu().numpy()]
            scores_np = values.detach().cpu().numpy().astype(np.float32)
            del q, scores, values, indices
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        except RuntimeError as exc:
            if device.type != 'cuda' or 'out of memory' not in str(exc).lower() or current_batch <= 1:
                raise
            torch.cuda.empty_cache()
            current_batch = max(1, current_batch // 2)
            print(f'[OOM] query batch -> {current_batch}')
            continue

        for local_i, uid in enumerate(query_user_ids[start:stop]):
            kept_ids, kept_scores = filter_seen_topk(
                ids_np[local_i], scores_np[local_i], past_by_user.get(int(uid), set()), top_k
            )
            n = len(kept_ids)
            out_ids[start + local_i, :n] = kept_ids.astype(np.int32)
            out_scores[start + local_i, :n] = kept_scores.astype(np.float16)
        start = stop
        print(f'  candidates: {start:,}/{len(query_user_ids):,}')

    del candidate_tensor
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return out_ids, out_scores, current_batch


def _ragged_truth(eval_user_ids: np.ndarray, truth_by_user: dict[int, list[int]]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(eval_user_ids) + 1, dtype=np.int64)
    flat = []
    for i, uid in enumerate(eval_user_ids):
        values = truth_by_user[int(uid)]
        flat.extend(values)
        offsets[i + 1] = len(flat)
    return offsets, np.asarray(flat, dtype=np.int32)


def _candidate_metrics(
    user_ids: np.ndarray,
    candidate_ids: np.ndarray,
    truth_by_user: dict[int, list[int]],
) -> dict:
    hits = 0
    truths = 0
    users_hit = 0
    for uid, recs in zip(user_ids, candidate_ids):
        truth = set(truth_by_user[int(uid)])
        rec_set = set(int(x) for x in recs if int(x) >= 0)
        overlap = len(truth & rec_set)
        hits += overlap
        truths += len(truth)
        users_hit += int(overlap > 0)
    return {
        'reachable_truth_jobs': int(truths),
        'candidate_hits_at_k': int(hits),
        'candidate_recall_at_k': float(hits / truths) if truths else None,
        'user_hit_rate_at_k': float(users_hit / len(user_ids)) if len(user_ids) else None,
    }


def build_gte_candidates(
    apps: pd.DataFrame,
    users: pd.DataFrame,
    windows: pd.DataFrame,
    staged: dict,
    local_cache_dir: Path,
    output_dir: Path,
    top_k: int,
    batch_size: int,
) -> dict:
    candidates_dir = output_dir / 'gte_candidates'
    candidates_dir.mkdir(parents=True, exist_ok=True)
    user_embedding_ids = np.load(staged['gte_user_ids'], mmap_mode='r')
    user_embeddings = np.load(staged['gte_user_embeddings'], mmap_mode='r')
    window_reports = {}

    for window_id in sorted(users['WindowID'].unique().astype(int)):
        out_path = candidates_dir / f'window_{window_id}_top{top_k}.npz'
        report_path = candidates_dir / f'window_{window_id}.json'
        if out_path.exists() and report_path.exists():
            report = json.loads(report_path.read_text())
            if report.get('status') == 'complete':
                window_reports[str(window_id)] = report
                print(f'[resume] candidates window {window_id}')
                continue

        cutoff, validation_end = _window_cutoff(windows, window_id)
        active_ids_path, active_emb_path = build_active_window_cache(
            window_id, cutoff, staged, local_cache_dir / 'active_windows'
        )
        active_ids = np.load(active_ids_path, mmap_mode='r')
        active_embeddings = np.load(active_emb_path, mmap_mode='r')
        views = temporal_views(users, apps, windows, window_id, active_ids)
        eval_user_ids = views['eval_user_ids']
        if len(eval_user_ids) == 0:
            raise RuntimeError(f'No reachable validation users in window {window_id}')
        queries = _load_user_queries(eval_user_ids, user_embedding_ids, user_embeddings)
        candidate_ids, candidate_scores, effective_batch = retrieve_topk_torch(
            queries, active_embeddings, active_ids, eval_user_ids,
            views['past_by_user'], top_k, batch_size,
        )
        truth_offsets, truth_job_ids = _ragged_truth(eval_user_ids, views['truth_by_user'])
        n_past_apps = np.array(
            [len(views['past_by_user'].get(int(uid), set())) for uid in eval_user_ids],
            dtype=np.int32,
        )
        np.savez_compressed(
            out_path,
            user_ids=eval_user_ids,
            candidate_job_ids=candidate_ids,
            candidate_scores=candidate_scores,
            truth_offsets=truth_offsets,
            truth_job_ids=truth_job_ids,
            n_past_apps=n_past_apps,
        )
        metrics = _candidate_metrics(eval_user_ids, candidate_ids, views['truth_by_user'])
        report = {
            'status': 'complete', 'window_id': int(window_id),
            'cutoff': cutoff.isoformat(), 'validation_end': validation_end.isoformat(),
            'users': int(len(eval_user_ids)), 'active_jobs': int(len(active_ids)),
            'top_k': int(top_k), 'effective_query_batch_size': int(effective_batch),
            **metrics,
        }
        report_path.write_text(json.dumps(report, indent=2))
        window_reports[str(window_id)] = report
        print(f'[done] window {window_id}: recall@{top_k}={metrics["candidate_recall_at_k"]:.4f}')

    manifest = {
        'status': 'complete', 'top_k': int(top_k), 'windows': window_reports,
        'total_users': int(sum(x['users'] for x in window_reports.values())),
    }
    (candidates_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def verify_phase2(output_dir: Path, expected_users: int) -> dict:
    candidates_dir = output_dir / 'gte_candidates'
    candidate_files = sorted(candidates_dir.glob('window_*_top*.npz'))
    gte_path = output_dir / 'gte_concentration.parquet'
    jobbert_path = output_dir / 'jobbert_concentration.parquet'
    if len(candidate_files) != 7:
        raise RuntimeError(f'Expected 7 candidate files, found {len(candidate_files)}')
    if not gte_path.exists() or not jobbert_path.exists():
        raise RuntimeError('Concentration outputs are missing')
    gte = pd.read_parquet(gte_path, columns=['user_id', 'n_embedded_apps', 'centroid_cosine'])
    jobbert = pd.read_parquet(jobbert_path, columns=['user_id', 'n_embedded_apps', 'centroid_cosine'])
    if len(gte) != expected_users or len(jobbert) != expected_users:
        raise RuntimeError('Concentration row count mismatch')
    for name, frame in [('gte', gte), ('jobbert', jobbert)]:
        valid = frame['n_embedded_apps'] > 0
        if not np.isfinite(frame.loc[valid, 'centroid_cosine']).all():
            raise RuntimeError(f'Non-finite {name} concentration for users with history')
    result = {
        'status': 'complete',
        'gte_candidate_windows': len(candidate_files),
        'gte_concentration_rows': int(len(gte)),
        'jobbert_concentration_rows': int(len(jobbert)),
    }
    (output_dir / 'verification.json').write_text(json.dumps(result, indent=2))
    return result


def run_phase2(config: Phase2Config) -> dict:
    started = time.time()
    data_dir = Path(config.data_dir)
    base_output_dir = Path(config.output_dir)
    phase2_output_dir = base_output_dir / 'semantic'
    phase2_output_dir.mkdir(parents=True, exist_ok=True)
    local_cache_dir = Path(config.local_cache_dir)
    local_cache_dir.mkdir(parents=True, exist_ok=True)

    apps, users, windows = _read_core_tables(data_dir)
    staged = stage_phase2_inputs(base_output_dir, local_cache_dir / 'staged')
    gte_app_ids, gte_app_emb = build_gte_application_cache(
        apps, staged, local_cache_dir / 'application_embeddings'
    )

    gte_concentration = build_concentration_table(
        'gte-modernbert-base', gte_app_ids, gte_app_emb,
        apps, users, windows, phase2_output_dir / 'gte_concentration.parquet',
        config.concentration_chunk_size,
    )
    jobbert_concentration = build_concentration_table(
        'JobBERT-v2', staged['jobbert_ids'], staged['jobbert_embeddings'],
        apps, users, windows, phase2_output_dir / 'jobbert_concentration.parquet',
        config.concentration_chunk_size,
    )
    candidates = build_gte_candidates(
        apps, users, windows, staged, local_cache_dir,
        phase2_output_dir, config.top_k, config.query_batch_size,
    )
    verification = verify_phase2(phase2_output_dir, len(users))
    summary = {
        'status': 'complete',
        'config': config.__dict__,
        'gte_concentration': gte_concentration,
        'jobbert_concentration': jobbert_concentration,
        'gte_candidates': candidates,
        'verification': verification,
        'seconds': round(time.time() - started, 2),
    }
    (phase2_output_dir / 'phase2_summary.json').write_text(json.dumps(summary, indent=2))
    return summary
