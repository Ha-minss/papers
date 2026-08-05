from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_figures(
    output_dir: Path,
    uplift_metrics: pd.DataFrame,
    policy: pd.DataFrame,
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    chart = uplift_metrics.sort_values("mean_fold_Qini")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].barh(chart["model"], chart["mean_fold_Qini"], xerr=chart["sd_fold_Qini"])
    axes[0].axvline(0.0, linewidth=1)
    axes[0].set_xlabel("Mean fold Qini")
    axes[0].set_ylabel("Model")
    axes[0].set_title("Repeated validation")

    pooled = uplift_metrics.sort_values("Top10_benefit_pp")
    xerr = [
        pooled["Top10_benefit_pp"] - pooled["Top10_CI_low_pp"],
        pooled["Top10_CI_high_pp"] - pooled["Top10_benefit_pp"],
    ]
    axes[1].errorbar(
        pooled["Top10_benefit_pp"],
        pooled["model"],
        xerr=xerr,
        fmt="o",
        capsize=3,
    )
    axes[1].axvline(0.0, linewidth=1)
    axes[1].set_xlabel("Top-10% estimated churn reduction (pp)")
    axes[1].set_ylabel("")
    axes[1].set_title("Conditional bootstrap interval")
    fig.tight_layout()
    fig.savefig(figures / "model_evidence.png", dpi=180)
    plt.close(fig)

    risk = policy[policy["model"].eq("Risk_XGBoost")].sort_values("contact_fraction")
    if not risk.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        x = risk["contact_fraction"] * 100
        lower = risk["estimated_benefit_pp"] - risk["benefit_CI_low_pp"]
        upper = risk["benefit_CI_high_pp"] - risk["estimated_benefit_pp"]
        axes[0].errorbar(x, risk["estimated_benefit_pp"], yerr=[lower, upper], fmt="o-", capsize=3)
        axes[0].axhline(0.0, linewidth=1)
        axes[0].set_xlabel("Customers contacted (%)")
        axes[0].set_ylabel("Estimated churn reduction (pp)")
        axes[0].set_title("Risk ranking by contact depth")

        if "Base_net_value" in risk.columns:
            axes[1].plot(x, risk["Base_net_value"], marker="o")
            axes[1].axhline(0.0, linewidth=1)
            axes[1].set_xlabel("Customers contacted (%)")
            axes[1].set_ylabel("Illustrative net value")
            axes[1].set_title("Base economic scenario")
        fig.tight_layout()
        fig.savefig(figures / "policy_depth.png", dpi=180)
        plt.close(fig)


def write_report(
    output_dir: Path,
    dataset_summary: dict[str, Any],
    risk_metrics: pd.DataFrame,
    uplift_metrics: pd.DataFrame,
    policy: pd.DataFrame,
    selection: dict[str, Any],
) -> None:
    top = uplift_metrics.iloc[0]
    risk_top10 = policy[
        policy["model"].eq("Risk_XGBoost") & policy["contact_fraction"].eq(0.10)
    ]
    risk_top10_text = "not evaluated"
    if not risk_top10.empty:
        risk_top10_text = f"{risk_top10.iloc[0]['estimated_benefit_pp']:.2f}%p"
    report = f"""# Orange Belgium churn uplift validation

## Technical summary

- Customers: **{dataset_summary['customers']:,}**, churn rate: **{dataset_summary['churn_rate']:.2%}**.
- Best repeated-CV ranking: **{top['model']}**, mean fold Qini **{top['mean_fold_Qini']:.6f}**.
- Accepted pure uplift model: **{selection['selected_uplift_model'] or 'None'}**.
- Risk-XGBoost top-10% point estimate: **{risk_top10_text}**.

## Conventional churn-risk models

{risk_metrics.to_markdown(index=False, floatfmt='.5f')}

## Repeated uplift validation

{uplift_metrics.to_markdown(index=False, floatfmt='.6f')}

## Campaign-policy simulation

{policy.to_markdown(index=False, floatfmt='.4f')}

## Decision

{selection['decision']}

## Limitations

- The outcome is rare and the control group contains relatively few churn events.
- The public file combines campaigns without a campaign identifier.
- PCA and anonymized factors prevent business-driver interpretation.
- Risk ranking is not an individual treatment-effect model.
- Economic assumptions are illustrative, not Orange Belgium or J:COM costs.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
