from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CrossValidationConfig:
    outer_folds: int = 5
    inner_folds: int = 3


@dataclass(frozen=True)
class OptunaConfig:
    trials: int = 8
    timeout_seconds: int | None = None


@dataclass(frozen=True)
class OutputConfig:
    save_oof_predictions: bool = False
    save_model: bool = False
    generate_figures: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    random_seed: int = 42
    cross_validation: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    candidate_models: tuple[str, ...] = ("LightGBM", "XGBoost", "CatBoost")
    top_fractions: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)
    shap_sample_size: int = 2500
    bootstrap_iterations: int = 1000
    max_rows: int | None = None


def validate_config(config: ExperimentConfig) -> ExperimentConfig:
    cv = config.cross_validation
    if cv.outer_folds < 2:
        raise ValueError("cross_validation.outer_folds must be at least 2.")
    if cv.inner_folds < 2:
        raise ValueError("cross_validation.inner_folds must be at least 2.")
    if config.optuna.trials < 1:
        raise ValueError("optuna.trials must be at least 1.")
    if config.bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be at least 1.")
    if not config.candidate_models:
        raise ValueError("candidate_models must not be empty.")
    supported = {"LightGBM", "XGBoost", "CatBoost"}
    unknown = set(config.candidate_models).difference(supported)
    if unknown:
        raise ValueError(f"Unsupported candidate_models: {sorted(unknown)}")
    if not config.top_fractions or any(not 0 < value <= 1 for value in config.top_fractions):
        raise ValueError("top_fractions must contain values in (0, 1].")
    if config.shap_sample_size < 1:
        raise ValueError("shap_sample_size must be at least 1.")
    return config


def _tuple(value: Any) -> tuple:
    if value is None:
        return tuple()
    return tuple(value)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cv_raw = raw.get("cross_validation", {})
    optuna_raw = raw.get("optuna", {})
    output_raw = raw.get("output", {})
    config = ExperimentConfig(
        random_seed=int(raw.get("random_seed", 42)),
        cross_validation=CrossValidationConfig(
            outer_folds=int(cv_raw.get("outer_folds", 5)),
            inner_folds=int(cv_raw.get("inner_folds", 3)),
        ),
        optuna=OptunaConfig(
            trials=int(optuna_raw.get("trials", 8)),
            timeout_seconds=optuna_raw.get("timeout_seconds"),
        ),
        output=OutputConfig(
            save_oof_predictions=bool(output_raw.get("save_oof_predictions", False)),
            save_model=bool(output_raw.get("save_model", False)),
            generate_figures=bool(output_raw.get("generate_figures", True)),
        ),
        candidate_models=_tuple(raw.get("candidate_models", ["LightGBM", "XGBoost", "CatBoost"])),
        top_fractions=tuple(float(v) for v in raw.get("top_fractions", [0.05, 0.10, 0.20, 0.30])),
        shap_sample_size=int(raw.get("shap_sample_size", 2500)),
        bootstrap_iterations=int(raw.get("bootstrap_iterations", 1000)),
        max_rows=None if raw.get("max_rows") is None else int(raw["max_rows"]),
    )
    return validate_config(config)
