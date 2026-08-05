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
    model_metrics: pd.DataFrame,
    policy: pd.DataFrame,
    shap_importance: pd.DataFrame,
    y_true,
    churn_probability,
    cltv,
) -> None:
    import math

    import numpy as np
    from sklearn.calibration import calibration_curve

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    y = np.asarray(y_true, dtype=int)
    probability = np.asarray(churn_probability, dtype=float)
    value = np.asarray(cltv, dtype=float)
    if not (len(y) == len(probability) == len(value)):
        raise ValueError("y_true, churn_probability, and CLTV must have the same length.")

    # Fig. 1: out-of-fold probability calibration.
    bins = min(10, max(2, len(y) // 2))
    observed, predicted = calibration_curve(y, probability, n_bins=bins, strategy="quantile")
    plt.figure(figsize=(6.5, 5.2))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect calibration")
    plt.plot(predicted, observed, marker="o", label="OOF model")
    plt.xlabel("Mean predicted churn probability")
    plt.ylabel("Observed churn rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "fig1_calibration.png", dpi=180)
    plt.close()

    # Fig. 2: SHAP-based global importance.
    top = shap_importance.head(15).sort_values("mean_abs_SHAP")
    plt.figure(figsize=(7.2, 5.8))
    plt.barh(top["feature"], top["mean_abs_SHAP"])
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(figure_dir / "fig2_shap_importance.png", dpi=180)
    plt.close()

    # Fig. 3: composition of equal-capacity risk and value-weighted selections.
    value_score = probability * value
    selected_n = math.ceil(len(y) * 0.10)
    risk_selected = np.zeros(len(y), dtype=bool)
    value_selected = np.zeros(len(y), dtype=bool)
    risk_selected[np.argsort(probability, kind="mergesort")[::-1][:selected_n]] = True
    value_selected[np.argsort(value_score, kind="mergesort")[::-1][:selected_n]] = True
    groups = {
        "Other": ~(risk_selected | value_selected),
        "Both": risk_selected & value_selected,
        "Risk only": risk_selected & ~value_selected,
        "Value-weighted only": value_selected & ~risk_selected,
    }
    plt.figure(figsize=(7.2, 5.6))
    for label, mask in groups.items():
        if mask.any():
            plt.scatter(probability[mask], value[mask], s=14, alpha=0.65, label=label)
    plt.xlabel("OOF churn probability")
    plt.ylabel("CLTV")
    plt.legend(markerscale=1.4)
    plt.tight_layout()
    plt.savefig(figure_dir / "fig3_risk_value_scatter.png", dpi=180)
    plt.close()

    # Fig. 4: cumulative observed churned-CLTV capture.
    total_churned_value = float(value[y == 1].sum())
    x = np.arange(1, len(y) + 1) / len(y)
    plt.figure(figsize=(7.0, 5.2))
    for label, score in (("Risk-only", probability), ("Value-weighted", value_score)):
        order = np.argsort(score, kind="mergesort")[::-1]
        captured = np.cumsum(value[order] * (y[order] == 1))
        curve = captured / total_churned_value if total_churned_value else np.zeros(len(y))
        plt.plot(x * 100, curve * 100, label=label)
    plt.plot(x * 100, x * 100, linestyle="--", linewidth=1, label="Random expectation")
    plt.xlabel("Customers reviewed (%)")
    plt.ylabel("Observed churned CLTV captured (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "fig4_value_capture_curve.png", dpi=180)
    plt.close()


def write_markdown_report(
    output_dir: Path,
    dataset_summary: dict[str, Any],
    model_metrics: pd.DataFrame,
    selection: dict[str, Any],
    policy: pd.DataFrame,
    shap_importance: pd.DataFrame,
    segments: pd.DataFrame,
) -> None:
    selected = selection["selected_model"]
    selected_row = model_metrics.loc[model_metrics["model"].eq(selected)].iloc[0]
    top10 = policy[policy["contact_fraction"].eq(0.10)]
    top20 = policy[policy["contact_fraction"].eq(0.20)]
    report = f"""# IBM Telco 고객 해지 위험 및 고객가치 우선순위

## 핵심 결론

- 고객 **{dataset_summary['customers']:,}명**, 해지율 **{dataset_summary['churn_rate']:.2%}**.
- 선택 모델: **{selected}**, OOF PR-AUC **{selected_row['PR_AUC']:.4f}**.
- CLTV는 모델 입력에서 제외하고 `OOF churn probability × CLTV`로 사후 우선순위를 계산했다.

## OOF 모델 비교

{model_metrics.to_markdown(index=False, floatfmt='.4f')}

## 고객 유지 우선순위: 상위 10%

{top10.to_markdown(index=False, floatfmt='.4f')}

## 고객 유지 우선순위: 상위 20%

{top20.to_markdown(index=False, floatfmt='.4f')}

## SHAP 주요 예측 변수

{shap_importance.head(15).to_markdown(index=False, floatfmt='.5f')}

## 고위험 관찰 세그먼트

{segments.head(10).to_markdown(index=False, floatfmt='.4f')}

## 해석 범위

이 분석은 해지 위험과 손실 가능 고객가치를 우선순위화한다. SHAP과 그룹별 해지율은 인과효과가 아니며, 실제 유지 캠페인의 증분효과는 처치·통제 데이터로 별도 검증해야 한다.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
