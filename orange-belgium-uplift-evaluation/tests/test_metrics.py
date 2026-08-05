import numpy as np

from churn_uplift.metrics import (
    fold_percentile_rank,
    hajek_uplift_curve_metrics,
    selected_group_effect,
)


def test_positive_uplift_means_churn_reduction():
    y = np.array([1, 0, 0, 0])
    t = np.array([0, 0, 1, 1])
    effect = selected_group_effect(y, t, np.array([True, True, True, True]))
    assert effect == 0.5


def test_fold_percentile_rank_preserves_order():
    rank = fold_percentile_rank(np.array([10.0, 30.0, 20.0]))
    assert list(np.argsort(rank)) == [0, 2, 1]


def test_hajek_qini_is_zero_for_constant_scores_with_balanced_effect():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    t = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    score = np.ones(len(y))
    result = hajek_uplift_curve_metrics(score, y, t, top_fractions=(0.5,))
    assert abs(result["Qini"]) < 1e-12


def test_stratified_bootstrap_preserves_treatment_group_sizes():
    from churn_uplift.metrics import stratified_bootstrap_indices

    treatment = np.array([0, 0, 0, 1, 1])
    sampled = stratified_bootstrap_indices(treatment, np.random.default_rng(7))
    assert len(sampled) == len(treatment)
    assert int((treatment[sampled] == 0).sum()) == 3
    assert int((treatment[sampled] == 1).sum()) == 2
