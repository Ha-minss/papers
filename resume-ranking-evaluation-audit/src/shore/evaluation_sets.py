from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

STAGE_ORDER = {
    "random_unobserved": 0,
    "exposed_unviewed": 1,
    "viewed_unapplied": 2,
    "applied_rejected": 3,
    "applied_satisfied": 4,
}


@dataclass(frozen=True)
class SplitResult:
    train_jobs: tuple[str, ...]
    valid_jobs: tuple[str, ...]
    test_jobs: tuple[str, ...]


def assign_deepest_stage(actions: pd.DataFrame) -> pd.DataFrame:
    required = {"jd_no", "user_id", "stage"}
    missing = required - set(actions.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    x = actions.copy()
    x["stage_rank"] = x["stage"].map(STAGE_ORDER)
    if x["stage_rank"].isna().any():
        bad = sorted(x.loc[x["stage_rank"].isna(), "stage"].astype(str).unique())
        raise ValueError(f"unknown stages: {bad}")
    idx = x.groupby(["jd_no", "user_id"], sort=False)["stage_rank"].idxmax()
    return x.loc[idx, ["jd_no", "user_id", "stage"]].reset_index(drop=True)


def build_job_disjoint_splits(jobs: Iterable[str], seed: int, ratios=(0.7, 0.15, 0.15)) -> SplitResult:
    jobs = np.array(sorted({str(j) for j in jobs}), dtype=object)
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("ratios must sum to 1")
    rng = np.random.default_rng(seed)
    rng.shuffle(jobs)
    n = len(jobs)
    a = int(n * ratios[0])
    b = a + int(n * ratios[1])
    return SplitResult(tuple(jobs[:a]), tuple(jobs[a:b]), tuple(jobs[b:]))


def build_conventional_queries(
    positives: pd.DataFrame,
    candidate_users: Iterable[str],
    candidate_count: int = 100,
    seed: int = 20260730,
) -> pd.DataFrame:
    if candidate_count < 2:
        raise ValueError("candidate_count must be >= 2")
    users = np.array(sorted({str(u) for u in candidate_users}), dtype=object)
    rng = np.random.default_rng(seed)
    rows = []
    for qid, row in positives[["jd_no", "user_id"]].drop_duplicates().reset_index(drop=True).iterrows():
        pos = str(row.user_id)
        pool = users[users != pos]
        if len(pool) < candidate_count - 1:
            raise ValueError("not enough candidate users")
        negs = rng.choice(pool, size=candidate_count - 1, replace=False)
        query_id = f"{row.jd_no}:random:{qid}"
        rows.append((query_id, str(row.jd_no), pos, 1, "applied_satisfied"))
        rows.extend((query_id, str(row.jd_no), str(u), 0, "random_unobserved") for u in negs)
    return pd.DataFrame(rows, columns=["query_id", "jd_no", "user_id", "label", "candidate_type"])


def build_matched_pairwise_queries(stages: pd.DataFrame, seed: int = 20260730) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    negative_types = ["random_unobserved", "exposed_unviewed", "viewed_unapplied", "applied_rejected"]
    for job, g in stages.groupby("jd_no", sort=True):
        positives = g.loc[g.stage.eq("applied_satisfied"), "user_id"].astype(str).unique()
        if len(positives) == 0:
            continue
        pos = str(rng.choice(positives))
        available = {t: g.loc[g.stage.eq(t), "user_id"].astype(str).unique() for t in negative_types}
        if any(len(available[t]) == 0 for t in negative_types):
            continue
        for t in negative_types:
            neg = str(rng.choice(available[t]))
            qid = f"{job}:{t}"
            rows.extend([
                (qid, str(job), pos, 1, t),
                (qid, str(job), neg, 0, t),
            ])
    return pd.DataFrame(rows, columns=["query_id", "jd_no", "user_id", "label", "negative_type"])
