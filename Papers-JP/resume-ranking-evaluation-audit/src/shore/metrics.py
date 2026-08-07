from __future__ import annotations

import numpy as np
import pandas as pd


def dcg(labels: np.ndarray, k: int) -> float:
    labels = np.asarray(labels, dtype=float)[:k]
    if labels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, labels.size + 2))
    return float(np.sum((2.0 ** labels - 1.0) * discounts))


def ranked_metrics_per_query(df: pd.DataFrame, score_col: str, k: int = 10) -> pd.DataFrame:
    rows = []
    for query_id, g in df.groupby("query_id", sort=False):
        g = g.sort_values([score_col, "user_id"], ascending=[False, True], kind="mergesort")
        labels = g["label"].to_numpy(dtype=int)
        ideal = np.sort(labels)[::-1]
        ndcg = dcg(labels, k) / max(dcg(ideal, k), 1e-12)
        positive_ranks = np.flatnonzero(labels > 0) + 1
        mrr = 0.0 if len(positive_ranks) == 0 else 1.0 / positive_ranks[0]
        recall = float(labels[:k].sum() / max(labels.sum(), 1))
        rows.append({"query_id": query_id, "jd_no": str(g.jd_no.iloc[0]), "nDCG@10": ndcg, "MRR": mrr, "Recall@10": recall})
    return pd.DataFrame(rows)


def pairwise_metrics_per_query(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows = []
    for query_id, g in df.groupby("query_id", sort=False):
        if len(g) != 2 or set(g.label.astype(int)) != {0, 1}:
            raise ValueError(f"pairwise query {query_id} must contain one positive and one negative")
        pos = float(g.loc[g.label.eq(1), score_col].iloc[0])
        neg = float(g.loc[g.label.eq(0), score_col].iloc[0])
        rows.append({
            "query_id": query_id,
            "jd_no": str(g.jd_no.iloc[0]),
            "negative_type": str(g.negative_type.iloc[0]),
            "margin": pos - neg,
            "accuracy": float(pos > neg) + 0.5 * float(pos == neg),
        })
    return pd.DataFrame(rows)
