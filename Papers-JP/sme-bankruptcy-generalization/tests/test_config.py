from pathlib import Path

from scripts.core.config import load_data_schema, load_experiment_config

ROOT = Path(__file__).resolve().parents[1]


def test_experiment_config_has_reproducible_defaults():
    config = load_experiment_config(ROOT / "config" / "experiment.json")
    assert config.seed == 42
    assert config.evaluation_years == (2014, 2015, 2016)
    assert config.models == ("logistic", "xgboost", "lightgbm")
    assert config.feature_set == "candidate_60"
    assert config.partial_pooling.interaction_scale_grid == (0.10, 0.25, 0.50)


def test_data_schema_matches_public_dataset_shape():
    schema = load_data_schema(ROOT / "config" / "data_schema.json")
    assert schema.expected_csv_files == 32
    assert schema.expected_columns == 64
    assert schema.sectors == ("agriculture", "construction", "manufacture", "retail")
