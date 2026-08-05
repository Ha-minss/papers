from __future__ import annotations

from pathlib import Path

from telco_churn.config import load_config
from telco_churn.modeling import choose_final_parameters, make_preprocessor, suggest_params


class RecordingTrial:
    def __init__(self) -> None:
        self.names: list[str] = []

    def suggest_float(self, name, *args, **kwargs):
        self.names.append(name)
        return float(args[0])

    def suggest_int(self, name, *args, **kwargs):
        self.names.append(name)
        return int(args[0])

    def suggest_categorical(self, name, values):
        self.names.append(name)
        return values[0]


def test_paper_config_uses_eight_trials_and_one_thousand_bootstraps() -> None:
    config = load_config(Path('config/paper.yaml'))
    assert config.optuna.trials == 8
    assert config.bootstrap_iterations == 1000


def test_nested_search_keeps_tree_count_fixed() -> None:
    for model_name, forbidden in [
        ('LightGBM', 'n_estimators'),
        ('XGBoost', 'n_estimators'),
        ('CatBoost', 'iterations'),
    ]:
        trial = RecordingTrial()
        suggest_params(trial, model_name)
        assert forbidden not in trial.names


def test_preprocessor_matches_paper_unknown_category_policy() -> None:
    preprocessor = make_preprocessor()
    encoder = preprocessor.transformers[1][1].named_steps['onehot']
    assert encoder.handle_unknown == 'ignore'
    assert encoder.min_frequency == 10


def test_final_parameters_use_best_inner_fold_record_for_selected_family() -> None:
    records = [
        {'model': 'CatBoost', 'outer_fold': 1, 'best_inner_PR_AUC': 0.67, 'best_params': {'depth': 4}},
        {'model': 'XGBoost', 'outer_fold': 1, 'best_inner_PR_AUC': 0.68, 'best_params': {'max_depth': 3}},
        {'model': 'CatBoost', 'outer_fold': 2, 'best_inner_PR_AUC': 0.71, 'best_params': {'depth': 6}},
    ]
    params, score, fold = choose_final_parameters('CatBoost', records)
    assert params == {'depth': 6}
    assert score == 0.71
    assert fold == 2


def test_stratified_bootstrap_returns_ordered_intervals() -> None:
    import numpy as np
    from telco_churn.metrics import stratified_metric_intervals

    y = np.array([0] * 20 + [1] * 10, dtype=int)
    probability = np.linspace(0.01, 0.99, len(y))
    intervals = stratified_metric_intervals(y, probability, iterations=40, seed=42)
    for name in ('ROC_AUC_CI', 'PR_AUC_CI', 'Brier_CI'):
        low, high = intervals[name]
        assert low <= high
