import numpy as np

from churn_uplift.policy import simulate_campaign_policy


def test_no_contact_policy_has_zero_net_value():
    result = simulate_campaign_policy(
        score=np.array([0.2, 0.1]),
        y=np.array([1, 0]),
        treatment=np.array([0, 1]),
        fraction=0.0,
        contact_cost=30.0,
        saved_customer_value=500.0,
    )
    assert result["selected_n"] == 0
    assert result["net_value"] == 0.0


def test_negative_estimated_effect_is_not_clipped_to_zero():
    result = simulate_campaign_policy(
        score=np.array([4.0, 3.0, 2.0, 1.0]),
        y=np.array([0, 1, 0, 1]),
        treatment=np.array([0, 1, 0, 1]),
        fraction=1.0,
        contact_cost=30.0,
        saved_customer_value=500.0,
    )
    assert result["estimated_prevented_churns"] < 0
    assert result["net_value"] < -120.0


def test_policy_statistics_include_wald_interval_and_group_counts():
    result = simulate_campaign_policy(
        score=np.array([4.0, 3.0, 2.0, 1.0]),
        y=np.array([1, 0, 0, 0]),
        treatment=np.array([0, 1, 0, 1]),
        fraction=1.0,
        contact_cost=30.0,
        saved_customer_value=500.0,
    )
    assert result["control_n"] == 2
    assert result["treated_n"] == 2
    assert result["benefit_CI_low_pp"] < result["estimated_benefit_pp"]
    assert result["benefit_CI_high_pp"] > result["estimated_benefit_pp"]


def test_compare_policies_writes_named_scenario_columns():
    from churn_uplift.policy import compare_campaign_policies

    frame = compare_campaign_policies(
        {"risk": np.array([4.0, 3.0, 2.0, 1.0])},
        np.array([1, 0, 0, 0]),
        np.array([0, 1, 0, 1]),
        (0.5,),
        {"Base": {"contact_cost": 30.0, "saved_customer_value": 500.0}},
    )
    assert "Base_net_value" in frame.columns
    assert "break_even_saved_value_at_30_cost" in frame.columns
