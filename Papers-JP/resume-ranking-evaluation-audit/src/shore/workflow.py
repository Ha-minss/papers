from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .adaptation import TrainingConfig, build_same_job_pairs, train_pairwise_cross_encoder, train_pointwise_cross_encoder
from .config import ExperimentConfig
from .data_bundle import BundlePaths, extract_bundle, load_bundle_tables, validate_eval_frames, verify_manifest
from .metrics import pairwise_metrics_per_query, ranked_metrics_per_query
from .paper import reproduce_paper
from .reranking import CrossEncoderReranker, QwenCausalReranker, build_unique_pair_table, merge_pair_scores, score_unique_pairs
from .retrieval import BM25Retriever, DenseRetriever, rrf_score_matrix
from .statistics import cluster_bootstrap_mean, paired_cluster_bootstrap


class FullExperimentWorkflow:
    """Concrete stage implementations for the full paper experiment.

    Every expensive stage writes its output beneath ``artifact_dir/cache``. Methods
    can be called independently, which lets the manifest runner resume after an
    interrupted GPU runtime.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.raw = config.raw
        self.artifact_dir = config.artifact_dir
        self.cache_dir = self.artifact_dir / "cache"
        self.result_dir = self.artifact_dir / "results"
        self.model_dir = self.artifact_dir / "models"
        self.data_dir = self.artifact_dir / "data"
        for path in [self.cache_dir, self.result_dir, self.model_dir, self.data_dir]:
            path.mkdir(parents=True, exist_ok=True)

    @property
    def bundle_zip(self) -> Path:
        return Path(self.raw.get("paths", {}).get("data_bundle", "data/aliyun_zhaopin_preprocessed_bundle_v2.zip"))

    @property
    def bundle_paths(self) -> BundlePaths:
        return BundlePaths.from_output_dir(self.data_dir / "extracted" / "output")

    def prepare_data(self, _: dict) -> dict:
        paths = extract_bundle(self.bundle_zip, self.data_dir / "extracted")
        integrity = verify_manifest(paths.output_dir.parent)
        integrity_path = self.result_dir / "data_integrity.csv"
        integrity.to_csv(integrity_path, index=False)
        return {"output_dir": str(paths.output_dir), "integrity_path": str(integrity_path), "verified_files": int(len(integrity))}

    def _ensure_data(self) -> BundlePaths:
        paths = self.bundle_paths
        if not paths.users.exists():
            self.prepare_data({})
        return paths

    def _eval_tables(self):
        paths = self._ensure_data()
        tables = load_bundle_tables(paths, require_training=False)
        expected = int(self.raw.get("candidate_count", 100))
        validate_eval_frames(tables.confit_valid, tables.pairwise_valid, expected_conventional_size=expected)
        validate_eval_frames(tables.confit_test, tables.pairwise_test, expected_conventional_size=expected)
        return tables

    def _eval_index(self) -> tuple[list[str], list[str], list[str], list[str], dict[str, str], dict[str, str]]:
        tables = self._eval_tables()
        eval_frames = [tables.confit_valid, tables.confit_test, tables.pairwise_valid, tables.pairwise_test, tables.stage_valid, tables.stage_test]
        job_ids = sorted(set().union(*(set(frame.jd_no.astype(str)) for frame in eval_frames)))
        user_ids = tables.users.user_id.astype(str).tolist()
        jobs = tables.jobs.copy()
        jobs["jd_no"] = jobs.jd_no.astype(str)
        selected = jobs[jobs.jd_no.isin(job_ids)].drop_duplicates("jd_no")
        missing = sorted(set(job_ids) - set(selected.jd_no))
        if missing:
            raise KeyError(f"missing job text for {len(missing)} evaluation jobs")
        selected = selected.set_index("jd_no").loc[job_ids].reset_index()
        user_frame = tables.users.copy()
        user_frame["user_id"] = user_frame.user_id.astype(str)
        user_frame = user_frame.set_index("user_id").loc[user_ids].reset_index()
        job_texts = selected.job_text.fillna("").astype(str).tolist()
        resume_texts = user_frame.resume_text.fillna("").astype(str).tolist()
        job_map = dict(zip(job_ids, job_texts))
        resume_map = dict(zip(user_ids, resume_texts))
        index_path = self.cache_dir / "score_index.json"
        index_path.write_text(json.dumps({"job_ids": job_ids, "user_ids": user_ids}, ensure_ascii=False), encoding="utf-8")
        return job_ids, user_ids, job_texts, resume_texts, job_map, resume_map

    def run_bm25(self, _: dict) -> dict:
        job_ids, user_ids, jobs, resumes, _, _ = self._eval_index()
        output = self.cache_dir / "bm25_scores.npy"
        if not output.exists():
            matrix = BM25Retriever().fit(resumes).score(jobs)
            np.save(output, matrix.astype(np.float32))
        matrix = np.load(output, mmap_mode="r")
        return {"score_path": str(output), "shape": list(matrix.shape), "jobs": len(job_ids), "users": len(user_ids)}

    def _run_dense(self, config_key: str, output_name: str, batch_key: str, instruction: str | None = None) -> dict:
        job_ids, user_ids, jobs, resumes, _, _ = self._eval_index()
        output = self.cache_dir / output_name
        if not output.exists():
            models = self.raw.get("models", {})
            batch_sizes = self.raw.get("batch_sizes", {})
            retriever = DenseRetriever(
                models[config_key],
                batch_size=int(batch_sizes.get(batch_key, 32)),
                max_length=int(self.raw.get("max_length", 512)),
                query_instruction=instruction,
            )
            np.save(output, retriever.score(jobs, resumes).astype(np.float32))
        matrix = np.load(output, mmap_mode="r")
        return {"score_path": str(output), "shape": list(matrix.shape), "jobs": len(job_ids), "users": len(user_ids)}

    def run_chinese_dense(self, _: dict) -> dict:
        return self._run_dense("chinese_dense", "ritrieve_scores.npy", "chinese_dense")

    def run_qwen_embedding(self, _: dict) -> dict:
        instruction = self.raw.get("qwen_query_instruction") or (
            "Given a job posting, retrieve resumes of candidates who are suitable for the role "
            "based on job function, skills, work experience, education, industry, location, and salary requirements."
        )
        return self._run_dense("qwen_embedding", "qwen3_scores.npy", "qwen_embedding", instruction)

    def run_hybrid_rrf(self, _: dict) -> dict:
        bm25 = self.cache_dir / "bm25_scores.npy"
        qwen = self.cache_dir / "qwen3_scores.npy"
        if not bm25.exists():
            self.run_bm25({})
        if not qwen.exists():
            self.run_qwen_embedding({})
        output = self.cache_dir / "hybrid_rrf_scores.npy"
        if not output.exists():
            fused = rrf_score_matrix(np.load(bm25), np.load(qwen), k=int(self.raw.get("rrf_k", 60)))
            np.save(output, fused.astype(np.float32))
        return {"score_path": str(output), "shape": list(np.load(output, mmap_mode="r").shape)}

    def _unique_pairs(self) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
        tables = self._eval_tables()
        pair_path = self.cache_dir / "rerank_unique_pairs.csv.gz"
        frames = [tables.confit_valid, tables.confit_test, tables.pairwise_valid, tables.pairwise_test, tables.stage_valid, tables.stage_test]
        if pair_path.exists():
            pairs = pd.read_csv(pair_path, dtype={"jd_no": str, "user_id": str})
        else:
            pairs = build_unique_pair_table(*frames)
            pairs.to_csv(pair_path, index=False)
        _, _, _, _, job_map, resume_map = self._eval_index()
        return pairs, job_map, resume_map

    def run_gte_reranker(self, _: dict) -> dict:
        pairs, jobs, resumes = self._unique_pairs()
        output = self.cache_dir / "gte_zero_pair_scores.npy"
        scorer = CrossEncoderReranker(
            self.raw.get("models", {})["gte_reranker"],
            batch_size=int(self.raw.get("batch_sizes", {}).get("gte_reranker", 32)),
            max_length=int(self.raw.get("max_length", 512)),
        )
        scores = score_unique_pairs(pairs, jobs, resumes, scorer, output, batch_size=scorer.batch_size)
        return {"score_path": str(output), "pair_path": str(self.cache_dir / "rerank_unique_pairs.csv.gz"), "pairs": len(scores)}

    def run_qwen_reranker(self, _: dict) -> dict:
        pairs, jobs, resumes = self._unique_pairs()
        output = self.cache_dir / "qwen3_reranker_zero_pair_scores.npy"
        scorer = QwenCausalReranker(
            self.raw.get("models", {})["qwen_reranker"],
            batch_size=int(self.raw.get("batch_sizes", {}).get("qwen_reranker", 8)),
            max_length=int(self.raw.get("max_length", 512)),
        )
        scores = score_unique_pairs(pairs, jobs, resumes, scorer, output, batch_size=scorer.batch_size)
        return {"score_path": str(output), "pair_path": str(self.cache_dir / "rerank_unique_pairs.csv.gz"), "pairs": len(scores)}

    def _score_matrix_frame(self, frame: pd.DataFrame, score_path: Path) -> pd.DataFrame:
        index = json.loads((self.cache_dir / "score_index.json").read_text(encoding="utf-8"))
        job_to_row = {j: i for i, j in enumerate(index["job_ids"])}
        user_to_col = {u: i for i, u in enumerate(index["user_ids"])}
        matrix = np.load(score_path, mmap_mode="r")
        out = frame.copy()
        jr = out.jd_no.astype(str).map(job_to_row)
        uc = out.user_id.astype(str).map(user_to_col)
        if jr.isna().any() or uc.isna().any():
            raise KeyError("score matrix ID mapping failed")
        out["score"] = matrix[jr.astype(int).to_numpy(), uc.astype(int).to_numpy()]
        return out

    @staticmethod
    def _negative_column(frame: pd.DataFrame) -> str:
        return "candidate_source" if "candidate_source" in frame.columns else "negative_type"

    def _candidate_sensitivity(self, scored: pd.DataFrame, system: str) -> list[dict]:
        sizes = [int(x) for x in self.raw.get("random_pool_sizes", [10, 20, 50, 100])]
        repeats = int(self.raw.get("sensitivity_repeats", 20))
        rows: list[dict] = []
        rng = np.random.default_rng(self.config.seed)
        for size in sizes:
            if size < 2:
                continue
            repeat_values = []
            for _ in range(repeats):
                sampled_parts = []
                for _, group in scored.groupby("query_id", sort=False):
                    positives = group[group.label.astype(int).eq(1)]
                    negatives = group[group.label.astype(int).eq(0)]
                    need = size - len(positives)
                    if need < 0 or len(negatives) < need:
                        continue
                    chosen = negatives.iloc[rng.choice(len(negatives), size=need, replace=False)] if need else negatives.iloc[:0]
                    sampled_parts.append(pd.concat([positives, chosen], ignore_index=True))
                if sampled_parts:
                    metrics = ranked_metrics_per_query(pd.concat(sampled_parts, ignore_index=True), "score")
                    repeat_values.append(float(metrics["nDCG@10"].mean()))
            if repeat_values:
                rows.append({
                    "system": system,
                    "pool_type": "random_unobserved",
                    "candidate_count": size,
                    "nDCG@10_mean": float(np.mean(repeat_values)),
                    "nDCG@10_sd": float(np.std(repeat_values, ddof=1)) if len(repeat_values) > 1 else 0.0,
                    "repeats": len(repeat_values),
                })
        return rows

    def evaluate_zero_shot(self, _: dict) -> dict:
        tables = self._eval_tables()
        pairs, _, _ = self._unique_pairs()
        systems = {
            "bm25": ("matrix", self.cache_dir / "bm25_scores.npy"),
            "ritrieve": ("matrix", self.cache_dir / "ritrieve_scores.npy"),
            "qwen3": ("matrix", self.cache_dir / "qwen3_scores.npy"),
            "hybrid_rrf": ("matrix", self.cache_dir / "hybrid_rrf_scores.npy"),
            "gte_zero": ("pair", self.cache_dir / "gte_zero_pair_scores.npy"),
            "qwen3_reranker_zero": ("pair", self.cache_dir / "qwen3_reranker_zero_pair_scores.npy"),
        }
        ranking_rows: list[dict] = []
        pool_rows: list[dict] = []
        margin_frames: list[pd.DataFrame] = []
        sensitivity_rows: list[dict] = []
        gap_rows: list[dict] = []
        for system, (kind, path) in systems.items():
            if not path.exists():
                continue
            if kind == "matrix":
                conventional = self._score_matrix_frame(tables.confit_test, path)
                pairwise = self._score_matrix_frame(tables.pairwise_test, path)
            else:
                scores = np.load(path)
                conventional = merge_pair_scores(tables.confit_test, pairs, scores)
                pairwise = merge_pair_scores(tables.pairwise_test, pairs, scores)
            ranked = ranked_metrics_per_query(conventional, "score")
            ranking_rows.append({"system": system, **ranked[["nDCG@10", "MRR", "Recall@10"]].mean().to_dict(), "queries": len(ranked)})
            sensitivity_rows.extend(self._candidate_sensitivity(conventional, system))
            source_col = self._negative_column(pairwise)
            pairwise = pairwise.rename(columns={source_col: "negative_type"}) if source_col != "negative_type" else pairwise
            per_query = pairwise_metrics_per_query(pairwise, "score")
            per_query.insert(0, "system", system)
            margin_frames.append(per_query)
            for negative_type, group in per_query.groupby("negative_type"):
                ci = cluster_bootstrap_mean(group, "accuracy", iterations=self.config.bootstrap_iterations, seed=self.config.seed)
                pool_rows.append({"system": system, "negative_pool": negative_type, "pairwise_accuracy": float(group.accuracy.mean()), **ci})
            random_group = per_query[per_query.negative_type.isin(["random_unlabeled", "random_unobserved"])]
            hard_group = per_query[per_query.negative_type.eq("applied_rejected")]
            if len(random_group) and len(hard_group):
                gap = paired_cluster_bootstrap(random_group, hard_group, "accuracy", iterations=self.config.bootstrap_iterations, seed=self.config.seed)
                gap_rows.append({"system": system, "gap": gap["difference"], "ci_low": gap["ci_low"], "ci_high": gap["ci_high"], "jobs": gap["clusters"]})
        ranking_path = self.result_dir / "retrieval_and_reranker_test_metrics.csv"
        pool_path = self.result_dir / "model_by_candidate_pool_with_bootstrap_ci.csv"
        margin_path = self.result_dir / "pairwise_query_margins.csv"
        sensitivity_path = self.result_dir / "candidate_pool_size_sensitivity.csv"
        gap_path = self.result_dir / "paired_pool_gap_bootstrap.csv"
        pd.DataFrame(ranking_rows).to_csv(ranking_path, index=False)
        pd.DataFrame(pool_rows).to_csv(pool_path, index=False)
        (pd.concat(margin_frames, ignore_index=True) if margin_frames else pd.DataFrame()).to_csv(margin_path, index=False)
        pd.DataFrame(sensitivity_rows).to_csv(sensitivity_path, index=False)
        pd.DataFrame(gap_rows).to_csv(gap_path, index=False)
        return {
            "ranking_metrics": str(ranking_path), "pool_metrics": str(pool_path),
            "margins": str(margin_path), "sensitivity": str(sensitivity_path),
            "paired_gaps": str(gap_path), "systems": len(ranking_rows)
        }

    def _evaluate_cross_encoder(self, scorer, split: str) -> dict:
        tables = self._eval_tables()
        conventional = tables.confit_valid if split == "valid" else tables.confit_test
        pairwise = tables.pairwise_valid if split == "valid" else tables.pairwise_test
        eval_pairs = build_unique_pair_table(conventional, pairwise)
        _, _, _, _, jobs, resumes = self._eval_index()
        text_pairs = [(jobs[str(j)], resumes[str(u)]) for j, u in zip(eval_pairs.jd_no, eval_pairs.user_id)]
        scores = np.asarray(scorer.predict(text_pairs), dtype=np.float32)
        conventional_scored = merge_pair_scores(conventional, eval_pairs, scores)
        pairwise_scored = merge_pair_scores(pairwise, eval_pairs, scores)
        source_col = self._negative_column(pairwise_scored)
        if source_col != "negative_type":
            pairwise_scored = pairwise_scored.rename(columns={source_col: "negative_type"})
        ranked = ranked_metrics_per_query(conventional_scored, "score")
        pair = pairwise_metrics_per_query(pairwise_scored, "score")
        hard = pair[pair.negative_type.eq("applied_rejected")]
        random_pool = pair[pair.negative_type.isin(["random_unlabeled", "random_unobserved"])]
        return {
            f"{split}_recruiter_hard_accuracy": float(hard.accuracy.mean()) if len(hard) else float("nan"),
            f"{split}_random_accuracy": float(random_pool.accuracy.mean()) if len(random_pool) else float("nan"),
            f"{split}_confit_ndcg10": float(ranked["nDCG@10"].mean()),
        }

    def _training_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        paths = self._ensure_data()
        tables = load_bundle_tables(paths, require_training=True)
        users = tables.users.copy()
        users["user_id"] = users.user_id.astype(str)
        resume_map = users.set_index("user_id").resume_text.fillna("").astype(str).to_dict()
        needed = set(tables.train_pairs.jd_no.astype(str)) | set(tables.valid_pairs.jd_no.astype(str))
        jobs = tables.jobs.copy()
        jobs["jd_no"] = jobs.jd_no.astype(str)
        job_map = jobs[jobs.jd_no.isin(needed)].drop_duplicates("jd_no").set_index("jd_no").job_text.fillna("").astype(str).to_dict()

        def enrich(frame: pd.DataFrame) -> pd.DataFrame:
            x = frame.copy()
            x["jd_no"] = x.jd_no.astype(str)
            x["user_id"] = x.user_id.astype(str)
            x["job_text"] = x.jd_no.map(job_map)
            x["resume_text"] = x.user_id.map(resume_map)
            label_col = "confit_label" if "confit_label" in x.columns else "label"
            x["label"] = x[label_col].astype(float)
            if x[["job_text", "resume_text", "label"]].isna().any().any():
                raise ValueError("missing training text or label")
            return x
        return enrich(tables.train_pairs), enrich(tables.valid_pairs)

    def _training_config(self, strategy: str) -> TrainingConfig:
        training = self.raw.get("training", {})
        return TrainingConfig(
            epochs=int(training.get(f"{strategy}_epochs", training.get("epochs", 1))),
            learning_rate=float(training.get("learning_rate", 2e-5)),
            batch_size=int(training.get("batch_size", 8)),
            gradient_accumulation=int(training.get("gradient_accumulation", 4)),
            warmup_ratio=float(training.get("warmup_ratio", 0.1)),
            distill_weight=float(training.get("distill_weight", 1.0)),
            max_length=int(self.raw.get("max_length", 512)),
            seed=self.config.seed,
        )

    def _evaluate_saved_pair_scores(self, score_path: Path, split: str) -> dict:
        tables = self._eval_tables()
        pairs, _, _ = self._unique_pairs()
        conventional = tables.confit_valid if split == "valid" else tables.confit_test
        pairwise = tables.pairwise_valid if split == "valid" else tables.pairwise_test
        scores = np.load(score_path)
        conventional_scored = merge_pair_scores(conventional, pairs, scores)
        pairwise_scored = merge_pair_scores(pairwise, pairs, scores)
        source_col = self._negative_column(pairwise_scored)
        if source_col != "negative_type":
            pairwise_scored = pairwise_scored.rename(columns={source_col: "negative_type"})
        ranked = ranked_metrics_per_query(conventional_scored, "score")
        pair = pairwise_metrics_per_query(pairwise_scored, "score")
        hard = pair[pair.negative_type.eq("applied_rejected")]
        random_pool = pair[pair.negative_type.isin(["random_unlabeled", "random_unobserved"])]
        return {
            f"{split}_recruiter_hard_accuracy": float(hard.accuracy.mean()) if len(hard) else float("nan"),
            f"{split}_random_accuracy": float(random_pool.accuracy.mean()) if len(random_pool) else float("nan"),
            f"{split}_confit_ndcg10": float(ranked["nDCG@10"].mean()),
        }

    def run_pointwise_adaptation(self, _: dict) -> dict:
        train, valid = self._training_frames()
        output = self.model_dir / "gte_pointwise"
        result = train_pointwise_cross_encoder(
            self.raw.get("models", {})["gte_reranker"],
            train[["job_text", "resume_text", "label"]],
            valid[["job_text", "resume_text", "label"]],
            output,
            self._training_config("pointwise"),
        )
        pairs, jobs, resumes = self._unique_pairs()
        score_path = self.cache_dir / "gte_finetuned_pair_scores.npy"
        scorer = CrossEncoderReranker(
            result["model_dir"],
            batch_size=int(self.raw.get("batch_sizes", {}).get("gte_reranker", 32)),
            max_length=int(self.raw.get("max_length", 512)),
        )
        score_unique_pairs(pairs, jobs, resumes, scorer, score_path, batch_size=scorer.batch_size)
        metrics = self._evaluate_saved_pair_scores(score_path, "test")
        payload = {**result, **metrics, "score_path": str(score_path), "selected": False}
        result_path = self.result_dir / "gte_pointwise_result.json"
        result_path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
        payload["result_path"] = str(result_path)
        return payload

    def _adaptation_baseline(self) -> dict[str, float]:
        score_path = self.cache_dir / "gte_zero_pair_scores.npy"
        if not score_path.exists():
            self.run_gte_reranker({})
        metrics = self._evaluate_saved_pair_scores(score_path, "valid")
        return {
            "hard_accuracy": metrics["valid_recruiter_hard_accuracy"],
            "random_accuracy": metrics["valid_random_accuracy"],
            "ndcg10": metrics["valid_confit_ndcg10"],
        }

    def _adaptation_callback(self, baseline: dict[str, float]):
        max_drop = float(self.raw.get("training", {}).get("max_general_drop", 0.01))
        from .adaptation import evaluate_checkpoint_gate

        def callback(cross_encoder, epoch: int) -> dict:
            metrics = self._evaluate_cross_encoder(cross_encoder, "valid")
            accepted = evaluate_checkpoint_gate(
                metrics["valid_recruiter_hard_accuracy"],
                metrics["valid_random_accuracy"],
                metrics["valid_confit_ndcg10"],
                baseline,
                max_general_drop=max_drop,
            )
            return {**metrics, "accepted": accepted}
        return callback

    def run_pairwise_adaptation(self, _: dict) -> dict:
        train, _ = self._training_frames()
        pairs = build_same_job_pairs(train, label_col="label")
        baseline = self._adaptation_baseline()
        result = train_pairwise_cross_encoder(
            self.raw.get("models", {})["gte_reranker"],
            pairs,
            self.model_dir / "gte_pairwise_ranknet",
            self._training_config("pairwise"),
            checkpoint_callback=self._adaptation_callback(baseline),
        )
        accepted = [r for r in result["history"] if r.get("accepted")]
        payload = {**result, "baseline": baseline, "selected": accepted[-1]["checkpoint"] if accepted else None}
        result_path = self.result_dir / "gte_pairwise_ranknet_result.json"
        result_path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
        payload["result_path"] = str(result_path)
        return payload

    def run_preservation_adaptation(self, _: dict) -> dict:
        train, _ = self._training_frames()
        pairs = build_same_job_pairs(train, label_col="label")
        teacher = CrossEncoderReranker(
            self.raw.get("models", {})["gte_reranker"],
            batch_size=int(self.raw.get("batch_sizes", {}).get("gte_reranker", 32)),
            max_length=int(self.raw.get("max_length", 512)),
        )
        positive_pairs = list(zip(pairs.job_text.astype(str), pairs.positive_text.astype(str)))
        negative_pairs = list(zip(pairs.job_text.astype(str), pairs.negative_text.astype(str)))
        pairs["teacher_margin"] = teacher.predict(positive_pairs) - teacher.predict(negative_pairs)
        del teacher
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        baseline = self._adaptation_baseline()
        result = train_pairwise_cross_encoder(
            self.raw.get("models", {})["gte_reranker"],
            pairs,
            self.model_dir / "gte_preservation",
            self._training_config("preservation"),
            preservation=True,
            checkpoint_callback=self._adaptation_callback(baseline),
        )
        accepted = [r for r in result["history"] if r.get("accepted")]
        payload = {**result, "baseline": baseline, "selected": accepted[-1]["checkpoint"] if accepted else None}
        result_path = self.result_dir / "gte_headlast_distill_result.json"
        result_path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
        payload["result_path"] = str(result_path)
        return payload

    def run_paper_analysis(self, _: dict) -> dict:
        outputs = reproduce_paper(self.artifact_dir, self.artifact_dir / "paper_reproduction")
        return {"output_dir": str(outputs.output_dir), "files": len(outputs.files)}
