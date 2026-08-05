from __future__ import annotations

from pathlib import Path

import numpy as np

from churn_uplift.config import load_config
from churn_uplift.learners import _cross_fitted_nuisance_predictions
from churn_uplift.metrics import risk_classification_metrics


def test_paper_config_loads_frozen_screening_parameters() -> None:
    config = load_config(Path('config/paper.yaml'))
    assert config.tune_t_xgboost is False
    assert config.risk_model_parameters['XGBoost']['max_depth'] == 6
    assert config.risk_model_parameters['LightGBM']['num_leaves'] == 30
    assert config.risk_model_parameters['CatBoost']['border_count'] == 64
    assert config.t_xgboost_parameters['n_estimators'] == 371


def test_cross_fitted_nuisance_predictions_never_score_training_rows(monkeypatch) -> None:
    matrix = np.arange(24, dtype=float).reshape(12, 2)
    y = np.array([0, 1] * 6, dtype=int)
    treatment = np.array([0, 0, 1, 1] * 3, dtype=int)

    class NoLeakClassifier:
        def fit(self, x, target, **kwargs):
            del target, kwargs
            self.seen = set(np.asarray(x)[:, 0].tolist())
            return self

        def predict_proba(self, x):
            values = np.asarray(x)[:, 0]
            assert not self.seen.intersection(values.tolist())
            probability = np.full(len(values), 0.25, dtype=float)
            return np.column_stack([1.0 - probability, probability])

    monkeypatch.setattr(
        'churn_uplift.learners.make_xgb_classifier',
        lambda params, seed: NoLeakClassifier(),
    )
    mu0, mu1 = _cross_fitted_nuisance_predictions(
        matrix,
        y,
        treatment,
        params={'n_estimators': 10},
        cv_seed=17,
        outer_fold=2,
    )
    assert mu0.shape == (12,)
    assert mu1.shape == (12,)
    assert np.all(np.isfinite(mu0))
    assert np.all(np.isfinite(mu1))


def test_risk_metrics_include_calibration_and_top_fraction_results() -> None:
    y = np.array([0, 0, 0, 1, 1], dtype=int)
    probability = np.array([0.05, 0.1, 0.2, 0.8, 0.9], dtype=float)
    result = risk_classification_metrics(y, probability)
    assert 'ECE10' in result
    assert result['Churners@20%'] == 1
    assert result['Recall@20%'] == 0.5
    assert result['Lift@20%'] > 1
