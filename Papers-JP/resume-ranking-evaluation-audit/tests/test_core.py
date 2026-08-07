import pytest
pytest.importorskip("torch")
import numpy as np
import pandas as pd
import torch

from shore.adaptation import pairwise_ranknet_loss, preservation_loss
from shore.evaluation_sets import assign_deepest_stage, build_job_disjoint_splits
from shore.metrics import pairwise_metrics_per_query, ranked_metrics_per_query
from shore.reranking import build_unique_pair_table, merge_pair_scores
from shore.retrieval import reciprocal_rank_fusion
from shore.statistics import model_rank_correlations


def test_deepest_stage_and_disjoint_split():
    x = pd.DataFrame({"jd_no": ["j1", "j1"], "user_id": ["u1", "u1"], "stage": ["viewed_unapplied", "applied_rejected"]})
    assert assign_deepest_stage(x).stage.iloc[0] == "applied_rejected"
    s = build_job_disjoint_splits([f"j{i}" for i in range(20)], seed=7)
    assert not (set(s.train_jobs) & set(s.valid_jobs) | set(s.train_jobs) & set(s.test_jobs) | set(s.valid_jobs) & set(s.test_jobs))


def test_metrics_and_statistics():
    ranked = pd.DataFrame({"query_id": ["q"] * 3, "jd_no": ["j"] * 3, "user_id": ["p", "n1", "n2"], "label": [1, 0, 0], "score": [3.0, 2.0, 1.0]})
    out = ranked_metrics_per_query(ranked, "score")
    assert out.loc[0, "nDCG@10"] == 1.0
    pair = pd.DataFrame({"query_id": ["q", "q"], "jd_no": ["j", "j"], "user_id": ["p", "n"], "label": [1, 0], "negative_type": ["applied_rejected"] * 2, "score": [0.2, 0.4]})
    assert pairwise_metrics_per_query(pair, "score").accuracy.iloc[0] == 0.0
    corr = model_rank_correlations(pd.DataFrame({"random_accuracy": [0.9, 0.8, 0.7], "recruiter_hard_accuracy": [0.4, 0.5, 0.6]}))
    assert np.isclose(corr["spearman_rho"], -1.0)


def test_retrieval_reranking_and_losses():
    rrf = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60)
    assert np.isclose(rrf["a"], rrf["b"])
    a = pd.DataFrame({"jd_no": ["j", "j"], "user_id": ["u1", "u2"]})
    pairs = build_unique_pair_table(a, a)
    merged = merge_pair_scores(a, pairs, np.array([0.1, 0.2]))
    assert merged.score.notna().all()
    assert pairwise_ranknet_loss(torch.tensor([1.0]), torch.tensor([0.0])).item() < pairwise_ranknet_loss(torch.tensor([0.0]), torch.tensor([1.0])).item()
    assert preservation_loss(torch.tensor([1.0]), torch.tensor([1.0]), 0.5).item() == 0.0
