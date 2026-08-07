import numpy as np

from telco_churn.metrics import classification_metrics


def test_dummy_mean_matches_prevalence_pr_auc():
    y = np.array([0, 0, 1, 1])
    probability = np.full(4, y.mean())
    result = classification_metrics(y, probability, top_fractions=(0.5,))
    assert result["ROC_AUC"] == 0.5
    assert result["PR_AUC"] == y.mean()
    assert result["Lift@50%"] == 1.0


def test_dummy_baseline_has_no_ranking_lift_even_when_fold_priors_differ():
    from telco_churn.metrics import dummy_baseline_metrics

    y = np.array([0, 0, 1, 1])
    fold_prior_probability = np.array([0.49, 0.51, 0.49, 0.51])
    result = dummy_baseline_metrics(y, fold_prior_probability, top_fractions=(0.5,))
    assert result["ROC_AUC"] == 0.5
    assert result["PR_AUC"] == 0.5
    assert result["Lift@50%"] == 1.0
