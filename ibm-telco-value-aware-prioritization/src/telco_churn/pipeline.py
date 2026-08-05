from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import ExperimentConfig, load_config
from .data import (
    CATEGORICAL_FEATURES,
    LEAKAGE_COLUMNS,
    MODEL_FEATURES,
    POLICY_ONLY_COLUMNS,
    build_dataset,
    data_quality_summary,
    load_telco_data,
)
from .explainability import global_shap_importance
from .metrics import stratified_metric_intervals
from .modeling import (
    choose_final_parameters,
    fit_final_model,
    run_nested_candidates,
    run_oof_baselines,
)
from .prioritization import calculate_value_at_risk, compare_priority_policies
from .paths import prepare_output_dir
from .provenance import build_run_metadata
from .reporting import write_figures, write_json, write_markdown_report


def _segment_summary(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["Tenure Band"] = pd.cut(
        working["Tenure Months"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6", "7-12", "13-24", "25-48", "49-72"],
    )
    grouped = (
        working.groupby(
            ["Contract", "Internet Service", "Tech Support", "Tenure Band"],
            observed=True,
        )["Churn Value"]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(columns={"sum": "churners", "mean": "churn_rate"})
    )
    return grouped[grouped["count"].ge(50)].sort_values(
        ["churn_rate", "count"], ascending=[False, False]
    )


def run_pipeline(
    data_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    config: ExperimentConfig = load_config(config_path)
    output = prepare_output_dir(output_dir, overwrite=overwrite)
    write_json(output / "run_metadata.json", build_run_metadata(data_path, config_path))

    raw_frame = load_telco_data(data_path)
    dataset = build_dataset(raw_frame, max_rows=config.max_rows, seed=config.random_seed)
    frame = dataset.frame

    quality = data_quality_summary(raw_frame)
    quality.to_csv(output / "data_quality.csv", index=False)

    baseline = run_oof_baselines(dataset.features, dataset.target, config)
    nested = run_nested_candidates(dataset.features, dataset.target, config)
    model_metrics = pd.concat([baseline.metrics, nested.metrics], ignore_index=True).sort_values(
        "PR_AUC", ascending=False
    )
    model_metrics.to_csv(output / "model_metrics.csv", index=False)

    selected_nested_row = nested.metrics.iloc[0]
    selected_model = selected_nested_row["model"].removesuffix("_NestedRaw")
    selected_oof = nested.predictions[selected_model]

    write_json(output / "nested_fold_best_params.json", nested.best_params_by_fold)
    final_params, best_inner_score, parameter_source_fold = choose_final_parameters(
        selected_model, nested.best_params_by_fold
    )
    final_model, preprocessor = fit_final_model(
        selected_model,
        dataset.features,
        dataset.target,
        final_params,
        seed=9000,
    )
    intervals = stratified_metric_intervals(
        dataset.target,
        selected_oof,
        iterations=config.bootstrap_iterations,
        seed=config.random_seed,
    )
    selection = {
        "selected_model": f"{selected_model}_NestedRaw",
        "selected_model_family": selected_model,
        "selected_oof_variant": f"{selected_model}_NestedRaw",
        "selection_metric": "OOF PR-AUC",
        "selected_oof_metrics": {
            key: float(selected_nested_row[key])
            for key in ["ROC_AUC", "PR_AUC", "Brier", "LogLoss", "ECE10"]
        },
        "outer_folds": config.cross_validation.outer_folds,
        "inner_folds": config.cross_validation.inner_folds,
        "optuna_trials_per_outer_fold": config.optuna.trials,
        **intervals,
        "final_params": final_params,
        "final_parameter_source_outer_fold": parameter_source_fold,
        "final_parameter_source_inner_PR_AUC": best_inner_score,
        "note": "CatBoost and XGBoost should be treated as practically tied when their PR-AUC difference is negligible.",
    }
    write_json(output / "model_selection.json", selection)

    policy = compare_priority_policies(
        dataset.target,
        dataset.cltv,
        selected_oof,
        config.top_fractions,
    )
    policy.to_csv(output / "policy_comparison.csv", index=False)

    value_at_risk = calculate_value_at_risk(selected_oof, dataset.cltv)
    priority = frame[
        [
            "CustomerID",
            "Churn Value",
            "CLTV",
            "Contract",
            "Internet Service",
            "Tech Support",
            "Tenure Months",
            "Monthly Charges",
        ]
    ].copy()
    priority["OOF Churn Probability"] = selected_oof
    priority["Value at Risk"] = value_at_risk
    priority["Risk Rank"] = pd.Series(selected_oof).rank(method="first", ascending=False).astype(int)
    priority["Value at Risk Rank"] = pd.Series(value_at_risk).rank(
        method="first", ascending=False
    ).astype(int)
    priority = priority.sort_values("Value at Risk Rank")
    priority.head(100).to_csv(output / "priority_sample_top100.csv", index=False)
    if config.output.save_oof_predictions:
        priority.to_csv(output / "customer_priority_oof.csv", index=False)

    shap_importance = global_shap_importance(
        selected_model,
        final_model,
        preprocessor,
        dataset.features,
        sample_size=config.shap_sample_size,
        seed=config.random_seed,
    )
    shap_importance.to_csv(output / "shap_importance.csv", index=False)

    segment_summary = _segment_summary(frame)
    segment_summary.to_csv(output / "segment_summary.csv", index=False)

    dataset_summary = {
        "customers": len(frame),
        "churners": int(dataset.target.sum()),
        "churn_rate": float(dataset.target.mean()),
        "model_features": MODEL_FEATURES,
        "excluded_leakage": sorted(LEAKAGE_COLUMNS),
        "policy_only": sorted(POLICY_ONLY_COLUMNS),
        "sampled_for_quick_mode": len(frame) != len(raw_frame),
    }
    write_json(output / "dataset_summary.json", dataset_summary)

    if config.output.generate_figures:
        write_figures(
            output,
            model_metrics,
            policy,
            shap_importance,
            dataset.target,
            selected_oof,
            dataset.cltv,
        )
    write_markdown_report(
        output,
        dataset_summary,
        model_metrics,
        selection,
        policy,
        shap_importance,
        segment_summary,
    )

    if config.output.save_model:
        artifact = {"model_name": selected_model, "model": final_model, "preprocessor": preprocessor}
        joblib.dump(artifact, output / "model.joblib")

    return {
        "output_dir": str(output),
        "selected_model": selected_model,
        "oof_pr_auc": float(selected_nested_row["PR_AUC"]),
        "customers": len(frame),
    }
