from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def cluster_bootstrap_mean(df: pd.DataFrame, value_col: str, cluster_col: str = "jd_no", iterations: int = 2000, seed: int = 20260802) -> dict:
    clusters = np.array(sorted(df[cluster_col].astype(str).unique()), dtype=object)
    if len(clusters) == 0:
        raise ValueError("no clusters")
    by_cluster = df.groupby(cluster_col)[value_col].mean()
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=float)
    for i in range(iterations):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        draws[i] = float(by_cluster.reindex(sampled).mean())
    return {
        "mean": float(df[value_col].mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "clusters": int(len(clusters)),
    }


def paired_cluster_bootstrap(left: pd.DataFrame, right: pd.DataFrame, value_col: str, cluster_col: str = "jd_no", iterations: int = 2000, seed: int = 20260802) -> dict:
    l = left.groupby(cluster_col)[value_col].mean()
    r = right.groupby(cluster_col)[value_col].mean()
    common = sorted(set(l.index.astype(str)) & set(r.index.astype(str)))
    if not common:
        raise ValueError("no common clusters")
    delta = pd.Series({c: float(l.loc[c] - r.loc[c]) for c in common})
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=float)
    arr = np.array(common, dtype=object)
    for i in range(iterations):
        sampled = rng.choice(arr, size=len(arr), replace=True)
        draws[i] = float(delta.reindex(sampled).mean())
    return {
        "difference": float(delta.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "clusters": int(len(common)),
    }


def model_rank_correlations(table: pd.DataFrame, random_col: str = "random_accuracy", hard_col: str = "recruiter_hard_accuracy") -> dict:
    x = table[[random_col, hard_col]].dropna()
    s = spearmanr(x[random_col], x[hard_col])
    k = kendalltau(x[random_col], x[hard_col])
    return {"spearman_rho": float(s.statistic), "spearman_p": float(s.pvalue), "kendall_tau": float(k.statistic), "kendall_p": float(k.pvalue), "models": int(len(x))}
