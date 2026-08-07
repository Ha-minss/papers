from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shore.data_bundle import BundlePaths, load_bundle_tables, validate_eval_frames
from shore.retrieval import BM25Retriever, DenseRetriever, score_matrix, rrf_score_matrix
from shore.reranking import QwenCausalReranker, score_unique_pairs
from shore.execution import ExperimentRunner


class FakeEncoder:
    def encode(self, texts, **kwargs):
        out = []
        for text in texts:
            text = str(text)
            out.append([len(text), text.count("python") + text.count("机器") + 1])
        x = np.asarray(out, dtype=np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        return x


class FakePairScorer:
    def predict(self, pairs):
        return np.asarray([len(a) - len(b) for a, b in pairs], dtype=np.float32)


def _write_bundle(root: Path) -> BundlePaths:
    output = root / "output"
    output.mkdir(parents=True)
    users = pd.DataFrame({"user_id": ["u1", "u2", "u3"], "resume_text": ["python", "机器学习", "sales"]})
    jobs = pd.DataFrame({"jd_no": ["j1", "j2"], "job_text": ["python engineer", "机器学习 engineer"], "job_split": ["valid", "test"]})
    confit = pd.DataFrame({
        "query_id": ["q1", "q1"], "jd_no": ["j1", "j1"], "user_id": ["u1", "u2"],
        "label": [1, 0], "candidate_source": ["applied_satisfied", "random_unlabeled"]
    })
    pairwise = pd.DataFrame({
        "query_id": ["p1", "p1"], "jd_no": ["j1", "j1"], "user_id": ["u1", "u2"],
        "label": [1, 0], "candidate_source": ["applied_satisfied", "applied_rejected"]
    })
    users.to_csv(output / "users_clean.csv.gz", index=False)
    jobs.to_csv(output / "jobs_clean.csv.gz", index=False)
    confit.to_csv(output / "eval_confit_valid_100.csv.gz", index=False)
    confit.to_csv(output / "eval_confit_test_100.csv.gz", index=False)
    pairwise.to_csv(output / "eval_matched_pairwise_valid_10seeds.csv.gz", index=False)
    pairwise.to_csv(output / "eval_matched_pairwise_10seeds.csv.gz", index=False)
    pairwise.to_csv(output / "eval_stage_specific_valid_1plus2.csv.gz", index=False)
    pairwise.to_csv(output / "eval_stage_specific_1plus2.csv.gz", index=False)
    return BundlePaths.from_output_dir(output)


def test_bundle_loader_and_validation(tmp_path):
    paths = _write_bundle(tmp_path)
    tables = load_bundle_tables(paths, require_training=False)
    assert list(tables.users.user_id) == ["u1", "u2", "u3"]
    validate_eval_frames(tables.confit_valid, tables.pairwise_valid, expected_conventional_size=2)


def test_bm25_and_dense_produce_aligned_score_matrices():
    jobs = ["python engineer", "机器学习 engineer"]
    resumes = ["python", "机器学习", "sales"]
    bm25 = BM25Retriever(tokenizer=lambda s: s.lower().split()).fit(resumes)
    bm = bm25.score(jobs)
    dense = DenseRetriever("fake", model=FakeEncoder()).score(jobs, resumes)
    assert bm.shape == dense.shape == (2, 3)
    assert bm[0, 0] > bm[0, 2]
    assert np.isfinite(dense).all()
    fused = rrf_score_matrix(bm, dense, k=60)
    assert fused.shape == (2, 3)


def test_unique_pair_scoring_is_restart_safe(tmp_path):
    pairs = pd.DataFrame({"jd_no": ["j1", "j1"], "user_id": ["u1", "u2"]})
    jobs = {"j1": "python engineer"}
    resumes = {"u1": "python", "u2": "sales"}
    out = tmp_path / "scores.npy"
    first = score_unique_pairs(pairs, jobs, resumes, FakePairScorer(), out, batch_size=1)
    second = score_unique_pairs(pairs, jobs, resumes, FakePairScorer(), out, batch_size=1)
    assert np.array_equal(first, second)
    assert out.exists()


def test_qwen_yes_no_probability_uses_token_logits():
    logits = np.array([[1.0, 3.0], [4.0, 2.0]], dtype=np.float32)
    scores = QwenCausalReranker.yes_probability(logits, yes_index=1, no_index=0)
    assert scores[0] > 0.5
    assert scores[1] < 0.5


def test_experiment_runner_executes_real_registered_stage(tmp_path):
    runner = ExperimentRunner(tmp_path)
    runner.register("one", lambda ctx: {"value": ctx["x"] + 1})
    result = runner.run(["one"], context={"x": 2})
    assert result["one"]["value"] == 3
    assert (tmp_path / "manifests" / "one.json").exists()
