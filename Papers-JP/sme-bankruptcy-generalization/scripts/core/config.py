from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PartialPoolingConfig:
    C: float
    C_grid: tuple[float, ...]
    interaction_scale_grid: tuple[float, ...]
    cv_folds: int


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    evaluation_years: tuple[int, ...]
    feature_set: str
    models: tuple[str, ...]
    preprocessing_modes: tuple[str, ...]
    imbalance_methods: tuple[str, ...]
    smote_sampling_strategy: float
    cv_folds: int
    partial_pooling: PartialPoolingConfig
    bootstrap_repetitions: int


@dataclass(frozen=True)
class DataSchema:
    expected_csv_files: int
    expected_columns: int
    sectors: tuple[str, ...]
    evaluation_years: tuple[int, ...]
    delimiter: str
    decimal: str
    missing_values: tuple[str, ...]


def _read_json(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    return json.loads(file_path.read_text(encoding="utf-8"))


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    raw = _read_json(path)
    partial = raw["partial_pooling"]
    config = ExperimentConfig(
        seed=int(raw["seed"]),
        evaluation_years=tuple(int(v) for v in raw["evaluation_years"]),
        feature_set=str(raw["feature_set"]),
        models=tuple(str(v) for v in raw["models"]),
        preprocessing_modes=tuple(str(v) for v in raw["preprocessing_modes"]),
        imbalance_methods=tuple(str(v) for v in raw["imbalance_methods"]),
        smote_sampling_strategy=float(raw["smote_sampling_strategy"]),
        cv_folds=int(raw["cv_folds"]),
        partial_pooling=PartialPoolingConfig(
            C=float(partial["C"]),
            C_grid=tuple(float(v) for v in partial["C_grid"]),
            interaction_scale_grid=tuple(float(v) for v in partial["interaction_scale_grid"]),
            cv_folds=int(partial["cv_folds"]),
        ),
        bootstrap_repetitions=int(raw["bootstrap_repetitions"]),
    )
    if not 0 < config.smote_sampling_strategy <= 1:
        raise ValueError("smote_sampling_strategy must be in (0, 1]")
    if len(config.evaluation_years) == 0:
        raise ValueError("At least one evaluation year is required")
    return config


def load_data_schema(path: str | Path) -> DataSchema:
    raw = _read_json(path)
    return DataSchema(
        expected_csv_files=int(raw["expected_csv_files"]),
        expected_columns=int(raw["expected_columns"]),
        sectors=tuple(str(v) for v in raw["sectors"]),
        evaluation_years=tuple(int(v) for v in raw["evaluation_years"]),
        delimiter=str(raw["delimiter"]),
        decimal=str(raw["decimal"]),
        missing_values=tuple(str(v) for v in raw["missing_values"]),
    )
