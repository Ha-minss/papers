import numpy as np

from telco_churn.prioritization import calculate_value_at_risk, evaluate_priority_policy


def test_value_at_risk_multiplies_probability_by_cltv():
    result = calculate_value_at_risk(np.array([0.2, 0.8]), np.array([1000.0, 500.0]))
    np.testing.assert_allclose(result, [200.0, 400.0])


def test_value_at_risk_policy_captures_high_value_churner():
    y = np.array([1, 0, 1, 0])
    cltv = np.array([100.0, 1000.0, 900.0, 100.0])
    score = np.array([10.0, 20.0, 30.0, 0.0])
    result = evaluate_priority_policy(y, cltv, score, fraction=0.25)
    assert result["observed_churned_CLTV"] == 900.0
    assert result["observed_churned_CLTV_capture"] == 0.9


def test_policy_expected_value_at_risk_sum_uses_common_value_score():
    import numpy as np
    from telco_churn.prioritization import compare_priority_policies

    y = np.array([1, 0, 1, 0])
    cltv = np.array([100.0, 1000.0, 500.0, 100.0])
    probability = np.array([0.9, 0.8, 0.7, 0.1])
    result = compare_priority_policies(y, cltv, probability, (0.5,))
    risk_row = result[result['policy'].eq('Churn_risk')].iloc[0]
    # Risk selects rows 0 and 1; their common value-at-risk is 90 + 800.
    assert risk_row['expected_value_at_risk_sum'] == 890.0
