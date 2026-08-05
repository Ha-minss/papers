import pandas as pd
import pytest

from churn_uplift.data import prepare_uplift_frame, validate_uplift_frame


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PC1": [0.1, -0.2, 0.3, -0.1],
            "FACTOR1": ["A", "B", "A", "B"],
            "FACTOR3": ["constant"] * 4,
            "FACTOR8": ["x", "x", "x", "y"],
            "y": [0, 1, 0, 1],
            "t": [0, 0, 1, 1],
        }
    )


def test_prepare_excludes_constant_and_near_constant_features():
    prepared = prepare_uplift_frame(sample_frame(), near_constant_threshold=0.74)
    assert "FACTOR3" not in prepared.feature_columns
    assert "FACTOR8" not in prepared.feature_columns


def test_validate_rejects_non_binary_treatment():
    frame = sample_frame()
    frame.loc[0, "t"] = 2
    with pytest.raises(ValueError, match="t must contain only 0 and 1"):
        validate_uplift_frame(frame)


def test_validate_requires_both_treatment_groups():
    frame = pd.DataFrame({"y": [0, 1], "t": [1, 1], "PC1": [0.1, 0.2]})
    with pytest.raises(ValueError, match="both 0 and 1"):
        validate_uplift_frame(frame)


def test_validate_requires_both_outcome_classes():
    frame = pd.DataFrame({"y": [0, 0], "t": [0, 1], "PC1": [0.1, 0.2]})
    with pytest.raises(ValueError, match="both 0 and 1"):
        validate_uplift_frame(frame)
