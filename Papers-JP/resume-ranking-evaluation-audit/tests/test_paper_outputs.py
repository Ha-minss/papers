from pathlib import Path
import pandas as pd

from shore.paper import reproduce_paper


def test_reproduce_paper_creates_all_core_tables_and_figures(tmp_path):
    artifacts = tmp_path / "artifacts"
    results = artifacts / "results"
    results.mkdir(parents=True)
    pd.DataFrame([
        {"system": "bm25", "negative_pool": "random_unlabeled", "pairwise_accuracy": .86, "ci_low": .8, "ci_high": .9},
        {"system": "bm25", "negative_pool": "applied_rejected", "pairwise_accuracy": .53, "ci_low": .45, "ci_high": .61},
        {"system": "qwen3_reranker_zero", "negative_pool": "random_unlabeled", "pairwise_accuracy": .93, "ci_low": .89, "ci_high": .96},
        {"system": "qwen3_reranker_zero", "negative_pool": "exposed_no_browse", "pairwise_accuracy": .60, "ci_low": .52, "ci_high": .68},
        {"system": "qwen3_reranker_zero", "negative_pool": "browsed_no_apply", "pairwise_accuracy": .51, "ci_low": .44, "ci_high": .58},
        {"system": "qwen3_reranker_zero", "negative_pool": "applied_rejected", "pairwise_accuracy": .48, "ci_low": .39, "ci_high": .57},
    ]).to_csv(results / "model_by_candidate_pool_with_bootstrap_ci.csv", index=False)
    pd.DataFrame([
        {"system": "bm25", "nDCG@10": .40, "MRR": .37, "Recall@10": .60},
        {"system": "qwen3_reranker_zero", "nDCG@10": .65, "MRR": .62, "Recall@10": .85},
    ]).to_csv(results / "retrieval_and_reranker_test_metrics.csv", index=False)
    pd.DataFrame([
        {"system": "qwen3_reranker_zero", "negative_type": "random_unlabeled", "margin": 2.0},
        {"system": "qwen3_reranker_zero", "negative_type": "applied_rejected", "margin": -0.2},
    ]).to_csv(results / "pairwise_query_margins.csv", index=False)
    pd.DataFrame([
        {"system": "bm25", "candidate_count": 10, "nDCG@10_mean": .6, "nDCG@10_sd": .01},
        {"system": "bm25", "candidate_count": 100, "nDCG@10_mean": .4, "nDCG@10_sd": .01},
    ]).to_csv(results / "candidate_pool_size_sensitivity.csv", index=False)
    outputs = reproduce_paper(artifacts, tmp_path / "out")
    names = {p.name for p in outputs.files}
    assert {"headline_model_results.csv", "conventional_ranking.csv", "model_pool_shift.pdf", "qwen_funnel.pdf", "score_margin_ecdf.pdf", "candidate_pool_sensitivity.pdf"}.issubset(names)
