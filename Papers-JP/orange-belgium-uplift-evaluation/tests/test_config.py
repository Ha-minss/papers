import pytest

from churn_uplift.config import CrossValidationConfig, ExperimentConfig, validate_config


def test_config_rejects_fewer_than_two_folds():
    config = ExperimentConfig(cross_validation=CrossValidationConfig(folds=1))
    with pytest.raises(ValueError, match="folds must be at least 2"):
        validate_config(config)


def test_config_rejects_invalid_contact_fraction():
    config = ExperimentConfig(top_fractions=(0.0, 0.1))
    with pytest.raises(ValueError, match="top_fractions"):
        validate_config(config)
