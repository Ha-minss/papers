import numpy as np
import pandas as pd

from telco_churn.reporting import write_figures


def test_write_figures_creates_the_four_manuscript_figures(tmp_path):
    model_metrics = pd.DataFrame(
        {"model": ["CatBoost_Nested"], "PR_AUC": [0.7]}
    )
    policy = pd.DataFrame(
        {
            "policy": ["Churn_risk", "Value_at_Risk"],
            "contact_fraction": [0.1, 0.1],
            "observed_churned_CLTV_capture": [0.3, 0.35],
        }
    )
    shap = pd.DataFrame(
        {"feature": ["Contract", "Tenure Months"], "mean_abs_SHAP": [0.2, 0.1]}
    )
    y = np.array([1, 0, 1, 0, 1, 0])
    probability = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.1])
    cltv = np.array([100, 1000, 500, 100, 900, 100], dtype=float)

    write_figures(tmp_path, model_metrics, policy, shap, y, probability, cltv)

    expected = {
        "fig1_calibration.png",
        "fig2_shap_importance.png",
        "fig3_risk_value_scatter.png",
        "fig4_value_capture_curve.png",
    }
    assert expected == {path.name for path in (tmp_path / "figures").glob("*.png")}
