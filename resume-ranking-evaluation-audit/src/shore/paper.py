from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .statistics import model_rank_correlations


@dataclass(frozen=True)
class PaperOutputs:
    output_dir: Path
    files: tuple[Path, ...]


def _find(artifact_dir: Path, names: list[str], required: bool = True) -> Path | None:
    for name in names:
        matches = list(artifact_dir.rglob(name))
        if matches:
            return max(matches, key=lambda p: p.stat().st_mtime)
    if required:
        raise FileNotFoundError(f"none of {names} found under {artifact_dir}")
    return None


def _normalise_pool_table(table: pd.DataFrame) -> pd.DataFrame:
    table = table.rename(columns={"model": "system", "pool": "negative_pool", "accuracy": "pairwise_accuracy"})
    required = {"system", "negative_pool", "pairwise_accuracy"}
    if not required.issubset(table.columns):
        raise ValueError(f"pool table missing columns: {sorted(required - set(table.columns))}")
    return table


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def reproduce_paper(artifact_dir: Path, output_dir: Path) -> PaperOutputs:
    artifact_dir, output_dir = Path(artifact_dir), Path(output_dir)
    table_dir, figure_dir = output_dir / "tables", output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    pool_source = _find(artifact_dir, ["model_by_candidate_pool_with_bootstrap_ci.csv", "full_model_results_table.csv"])
    pool = _normalise_pool_table(pd.read_csv(pool_source))
    pivot = pool.pivot_table(index="system", columns="negative_pool", values="pairwise_accuracy", aggfunc="first")
    random_col = "random_unlabeled" if "random_unlabeled" in pivot.columns else "random_unobserved"
    headline = pd.DataFrame({
        "model": pivot.index,
        "random_accuracy": pivot.get(random_col),
        "recruiter_hard_accuracy": pivot.get("applied_rejected"),
    }).dropna().reset_index(drop=True)
    headline["gap"] = headline.random_accuracy - headline.recruiter_hard_accuracy
    headline.to_csv(table_dir / "headline_model_results.csv", index=False)
    pool.to_csv(table_dir / "candidate_pool_results.csv", index=False)

    ranking_source = _find(artifact_dir, ["retrieval_and_reranker_test_metrics.csv", "full_model_results_table.csv"], required=False)
    if ranking_source is not None:
        ranking = pd.read_csv(ranking_source)
        ranking.to_csv(table_dir / "conventional_ranking.csv", index=False)

    corr = model_rank_correlations(headline) if len(headline) >= 2 else {
        "spearman_rho": np.nan, "spearman_p": np.nan, "kendall_tau": np.nan, "kendall_p": np.nan, "models": len(headline)
    }
    (table_dir / "model_rank_correlations.json").write_text(json.dumps(corr, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for _, row in headline.iterrows():
        ax.plot([0, 1], [100 * row.random_accuracy, 100 * row.recruiter_hard_accuracy], marker="o", label=row.model)
    ax.axhline(50, linestyle="--", linewidth=1)
    ax.set_xticks([0, 1], ["Random unobserved", "Recruiter rejected"])
    ax.set_ylabel("Pairwise accuracy (%)")
    ax.set_ylim(35, 100)
    if len(headline):
        ax.legend(fontsize=7, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    _save(fig, figure_dir / "model_pool_shift.pdf")

    qwen = pool[pool.system.eq("qwen3_reranker_zero")].copy()
    order = ["random_unlabeled", "random_unobserved", "exposed_no_browse", "exposed_unviewed", "browsed_no_apply", "viewed_unapplied", "applied_rejected"]
    canonical = {
        "random_unlabeled": "Random", "random_unobserved": "Random",
        "exposed_no_browse": "Exposed", "exposed_unviewed": "Exposed",
        "browsed_no_apply": "Viewed", "viewed_unapplied": "Viewed",
        "applied_rejected": "Rejected",
    }
    qwen["order"] = qwen.negative_pool.map({name: i for i, name in enumerate(order)})
    qwen["canonical_pool"] = qwen.negative_pool.map(canonical)
    qwen = qwen.sort_values("order").drop_duplicates("canonical_pool", keep="first") if len(qwen) else qwen
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    if len(qwen):
        x = np.arange(len(qwen))
        y = qwen.pairwise_accuracy.to_numpy(float)
        if {"ci_low", "ci_high"}.issubset(qwen.columns):
            err = np.vstack([y - qwen.ci_low.to_numpy(float), qwen.ci_high.to_numpy(float) - y])
            ax.errorbar(x, y, yerr=err, marker="o", capsize=3)
        else:
            ax.plot(x, y, marker="o")
        ax.set_xticks(x, [canonical.get(v, v) for v in qwen.negative_pool], rotation=15)
    ax.axhline(.5, linestyle="--", linewidth=1)
    ax.set_ylim(.35, 1.0)
    ax.set_ylabel("Pairwise accuracy")
    _save(fig, figure_dir / "qwen_funnel.pdf")

    margin_source = _find(artifact_dir, ["pairwise_query_margins.csv", "pairwise_query_margins.parquet"], required=False)
    if margin_source is not None:
        margins = pd.read_parquet(margin_source) if margin_source.suffix == ".parquet" else pd.read_csv(margin_source)
        fig, ax = plt.subplots(figsize=(4.4, 3.1))
        for pool_name, group in margins[margins.system.eq("qwen3_reranker_zero")].groupby("negative_type", sort=False):
            values = np.sort(group.margin.to_numpy(float))
            if len(values):
                ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=canonical.get(pool_name, pool_name))
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_xlabel("Positive score minus comparison score")
        ax.set_ylabel("Empirical cumulative probability")
        ax.legend(frameon=False, fontsize=7)
        _save(fig, figure_dir / "score_margin_ecdf.pdf")

    sensitivity_source = _find(artifact_dir, ["candidate_pool_size_sensitivity.csv"], required=False)
    if sensitivity_source is not None:
        sensitivity = pd.read_csv(sensitivity_source)
        fig, ax = plt.subplots(figsize=(4.4, 3.1))
        for system, group in sensitivity.groupby("system", sort=False):
            group = group.sort_values("candidate_count")
            ax.errorbar(group.candidate_count, group["nDCG@10_mean"], yerr=group.get("nDCG@10_sd"), marker="o", label=system)
        ax.set_xscale("log")
        ax.set_xlabel("Random candidate-pool size")
        ax.set_ylabel("nDCG@10")
        ax.legend(frameon=False, fontsize=7)
        _save(fig, figure_dir / "candidate_pool_sensitivity.pdf")


    # Adaptation results are intentionally split by evaluation split.
    adaptation_rows = []
    pointwise_path = _find(artifact_dir, ["gte_pointwise_result.json"], required=False)
    if pointwise_path is not None:
        data = json.loads(pointwise_path.read_text(encoding="utf-8"))
        adaptation_rows.append({
            "split": "test", "strategy": "Pointwise binary fine-tuning",
            "recruiter_hard_accuracy": data.get("test_recruiter_hard_accuracy"),
            "random_candidate_accuracy": data.get("test_random_accuracy"),
            "nDCG@10": data.get("test_confit_ndcg10"), "selected": bool(data.get("selected")),
        })
    for filename, strategy in [
        ("gte_pairwise_ranknet_result.json", "Same-job pairwise RankNet"),
        ("gte_headlast_distill_result.json", "Preservation-oriented partial tuning"),
    ]:
        path = _find(artifact_dir, [filename], required=False)
        if path is None:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        history = data.get("history", [])
        if history:
            best = max(history, key=lambda r: float(r.get("valid_recruiter_hard_accuracy", float("-inf"))))
            adaptation_rows.append({
                "split": "validation", "strategy": strategy,
                "recruiter_hard_accuracy": best.get("valid_recruiter_hard_accuracy"),
                "random_candidate_accuracy": best.get("valid_random_accuracy"),
                "nDCG@10": best.get("valid_confit_ndcg10"), "selected": data.get("selected") is not None,
            })
    if adaptation_rows:
        adaptation = pd.DataFrame(adaptation_rows)
        adaptation.to_csv(table_dir / "fine_tuning_strategy_comparison.csv", index=False)
        for split, group in adaptation.groupby("split", sort=False):
            group.to_csv(table_dir / f"adaptation_{split}.csv", index=False)
        fig, ax = plt.subplots(figsize=(5.4, 3.5))
        for split, group in adaptation.groupby("split", sort=False):
            ax.scatter(group["nDCG@10"], 100 * group.recruiter_hard_accuracy, label=split)
            for _, row in group.iterrows():
                ax.annotate(row.strategy, (row["nDCG@10"], 100 * row.recruiter_hard_accuracy), xytext=(3, 3), textcoords="offset points", fontsize=6)
        ax.axhline(50, linestyle="--", linewidth=1)
        ax.set_xlabel("nDCG@10 (within the same split)")
        ax.set_ylabel("Recruiter-hard accuracy (%)")
        ax.legend(frameon=False)
        _save(fig, figure_dir / "adaptation_tradeoff.pdf")

    summary = {
        "models": int(len(headline)),
        "largest_gap": float(headline.gap.max()) if len(headline) else None,
        "smallest_gap": float(headline.gap.min()) if len(headline) else None,
        **corr,
        "excluded_analysis": "recruiter-rejected candidate-pool-size sensitivity",
    }
    (output_dir / "reproduction_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    files = tuple(sorted(p for p in output_dir.rglob("*") if p.is_file()))
    return PaperOutputs(output_dir, files)
