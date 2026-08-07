from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ExperimentConfig, load_config
from .data import load_uplift_data, prepare_uplift_frame
from .learners import (
    run_repeated_uplift_validation,
    run_risk_model_comparison,
    tune_t_xgboost,
)
from .metrics import bootstrap_metric_interval, hajek_uplift_curve_metrics
from .policy import compare_campaign_policies
from .paths import prepare_output_dir
from .provenance import build_run_metadata
from .reporting import write_figures, write_json, write_report


def _select_uplift_model(metrics: pd.DataFrame) -> str | None:
    pure = metrics[~metrics["model"].eq("Risk_XGBoost")].copy()
    eligible = pure[
        pure["mean_fold_Qini"].gt(0)
        & pure["positive_fold_share"].ge(0.70)
        & pure["Qini_CI_low"].gt(0)
        & pure["Top10_CI_low_pp"].gt(0)
    ]
    if eligible.empty:
        return None
    return str(eligible.sort_values("mean_fold_Qini", ascending=False).iloc[0]["model"])


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

    raw = load_uplift_data(data_path)
    data = prepare_uplift_frame(
        raw,
        near_constant_threshold=config.near_constant_threshold,
        max_rows=config.max_rows,
        seed=config.random_seed,
    )
    dataset_summary = {
        "customers": len(data.frame),
        "columns": data.frame.shape[1],
        "churners": int(data.target.sum()),
        "churn_rate": float(data.target.mean()),
        "treated": int(data.treatment.sum()),
        "treatment_share": float(data.treatment.mean()),
        "control_churners": int(data.target[data.treatment == 0].sum()),
        "features": len(data.feature_columns),
        "excluded_columns": data.excluded_columns,
        "sampled_for_quick_mode": len(data.frame) != len(raw),
    }
    write_json(output / "dataset_summary.json", dataset_summary)

    risk = run_risk_model_comparison(data, config)
    risk.metrics.to_csv(output / "risk_model_metrics.csv", index=False)

    t_params, tuning_summary = tune_t_xgboost(data, config)
    write_json(output / "t_xgboost_tuning.json", tuning_summary)

    validation = run_repeated_uplift_validation(
        data,
        config,
        risk_params=risk.best_xgboost_params,
        t_params=t_params,
    )
    uplift_metrics = validation.aggregate_metrics.copy()

    qini_low = []
    qini_high = []
    top10_low = []
    top10_high = []
    top20_low = []
    top20_high = []
    for model_index, model_name in enumerate(uplift_metrics["model"]):
        score = validation.rank_scores[model_name]
        bootstrap_seed = 918 + model_index
        low, high = bootstrap_metric_interval(
            score,
            data.target,
            data.treatment,
            "Qini",
            config.bootstrap_iterations,
            bootstrap_seed,
        )
        low10, high10 = bootstrap_metric_interval(
            score,
            data.target,
            data.treatment,
            "Top10_benefit_pp",
            config.bootstrap_iterations,
            bootstrap_seed,
        )
        low20, high20 = bootstrap_metric_interval(
            score,
            data.target,
            data.treatment,
            "Top20_benefit_pp",
            config.bootstrap_iterations,
            bootstrap_seed,
        )
        qini_low.append(low)
        qini_high.append(high)
        top10_low.append(low10)
        top10_high.append(high10)
        top20_low.append(low20)
        top20_high.append(high20)
    uplift_metrics["Qini_CI_low"] = qini_low
    uplift_metrics["Qini_CI_high"] = qini_high
    uplift_metrics["Top10_CI_low_pp"] = top10_low
    uplift_metrics["Top10_CI_high_pp"] = top10_high
    uplift_metrics["Top20_CI_low_pp"] = top20_low
    uplift_metrics["Top20_CI_high_pp"] = top20_high
    uplift_metrics.to_csv(output / "uplift_metrics.csv", index=False)
    validation.fold_metrics.to_csv(output / "fold_metrics.csv", index=False)

    scenarios = {
        scenario.name: {
            "contact_cost": scenario.contact_cost,
            "saved_customer_value": scenario.saved_customer_value,
        }
        for scenario in config.economics.scenarios
    }
    policy = compare_campaign_policies(
        validation.rank_scores,
        data.target,
        data.treatment,
        config.top_fractions,
        scenarios,
        break_even_contact_cost=config.economics.break_even_contact_cost,
    )
    policy.to_csv(output / "campaign_policy.csv", index=False)

    scenario_decisions: dict[str, dict[str, object]] = {}
    for scenario_name in scenarios:
        column = f"{scenario_name}_net_value"
        best_row = policy.loc[policy[column].idxmax()]
        if float(best_row[column]) > 0:
            scenario_decisions[scenario_name] = {
                "policy": str(best_row["model"]),
                "contact_fraction": float(best_row["contact_fraction"]),
                "net_value": float(best_row[column]),
            }
        else:
            scenario_decisions[scenario_name] = {
                "policy": "No contact",
                "contact_fraction": 0.0,
                "net_value": 0.0,
            }
    write_json(output / "scenario_decisions.json", scenario_decisions)

    scenario_columns = [f"{name}_net_value" for name in scenarios]
    base_column = "Base_net_value" if "Base_net_value" in scenario_columns else scenario_columns[0]
    best_policy_by_model = (
        policy.sort_values(["model", base_column], ascending=[True, False])
        .groupby("model", as_index=False)
        .head(1)
        .sort_values(base_column, ascending=False)
    )
    best_policy_by_model.to_csv(output / "best_policy_by_model.csv", index=False)

    selected = _select_uplift_model(uplift_metrics)
    best_pure = (
        uplift_metrics[~uplift_metrics["model"].eq("Risk_XGBoost")]
        .sort_values("mean_fold_Qini", ascending=False)
        .iloc[0]
    )
    selection = {
        "selected_uplift_model": selected,
        "best_pure_uplift_candidate": str(best_pure["model"]),
        "decision": (
            f"Deploy {selected} only after an independent randomized holdout test."
            if selected
            else "Do not deploy a pure uplift model. Use the risk model only to define a candidate population for a new randomized holdout experiment."
        ),
        "acceptance_rule": {
            "mean_fold_Qini": "> 0",
            "positive_fold_share": ">= 0.70",
            "bootstrap_Qini_lower_bound": "> 0",
            "bootstrap_top10_effect_lower_bound": "> 0",
        },
        "economic_scenario_decisions": scenario_decisions,
    }
    write_json(output / "selection.json", selection)

    if config.output.save_row_level_scores:
        score_frame = pd.DataFrame(
            {"y": data.target, "t": data.treatment, **validation.rank_scores}
        )
        score_frame.to_csv(output / "row_level_scores.csv", index=False)

    if config.output.generate_figures:
        write_figures(output, uplift_metrics, policy)
    write_report(output, dataset_summary, risk.metrics, uplift_metrics, policy, selection)

    return {
        "output_dir": str(output),
        "selected_uplift_model": selected,
        "best_ranking": str(uplift_metrics.iloc[0]["model"]),
        "customers": len(data.frame),
    }
