from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_RISK_MODEL_PARAMETERS: dict[str, dict[str, Any]] = {
    "LightGBM": {
        "learning_rate": 0.021887439425976252,
        "num_leaves": 30,
        "max_depth": 7,
        "min_child_samples": 129,
        "subsample": 0.8261534422933426,
        "colsample_bytree": 0.6841852399022343,
        "reg_alpha": 0.08768125777016172,
        "reg_lambda": 0.046021703974597504,
    },
    "XGBoost": {
        "learning_rate": 0.03513049653867429,
        "max_depth": 6,
        "min_child_weight": 19.978808992275813,
        "subsample": 0.8996646210492591,
        "colsample_bytree": 0.7046065241548528,
        "gamma": 0.3899863008405066,
        "reg_alpha": 0.00017775399007348227,
        "reg_lambda": 4.88310591478166,
    },
    "CatBoost": {
        "learning_rate": 0.0735914047804502,
        "depth": 6,
        "l2_leaf_reg": 5.809498538661968,
        "random_strength": 0.10685033500506824,
        "bagging_temperature": 2.4247746304049858,
        "border_count": 64,
    },
}

DEFAULT_T_XGBOOST_PARAMETERS: dict[str, Any] = {
    "n_estimators": 371,
    "learning_rate": 0.01997819627241509,
    "max_depth": 3,
    "min_child_weight": 79.88264302101751,
    "subsample": 0.8045679118611637,
    "colsample_bytree": 0.8299360666559304,
    "gamma": 4.217075115009047,
    "reg_alpha": 0.15101375536543646,
    "reg_lambda": 0.5645181073420262,
}


@dataclass(frozen=True)
class CrossValidationConfig:
    folds: int = 5
    tuning_folds: int = 3
    repeated_seeds: tuple[int, ...] = (17, 42, 73)


@dataclass(frozen=True)
class OptunaConfig:
    trials: int = 24
    stability_penalty: float = 0.25
    timeout_seconds: int | None = None


@dataclass(frozen=True)
class EconomicScenario:
    name: str
    contact_cost: float
    saved_customer_value: float


@dataclass(frozen=True)
class EconomicsConfig:
    scenarios: tuple[EconomicScenario, ...] = (
        EconomicScenario("Conservative", 20.0, 200.0),
        EconomicScenario("Base", 30.0, 500.0),
        EconomicScenario("Upside", 50.0, 1000.0),
    )
    break_even_contact_cost: float = 30.0


@dataclass(frozen=True)
class OutputConfig:
    save_row_level_scores: bool = False
    generate_figures: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    random_seed: int = 42
    cross_validation: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    economics: EconomicsConfig = field(default_factory=EconomicsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    risk_models: tuple[str, ...] = ("Logistic", "LightGBM", "XGBoost", "CatBoost")
    uplift_models: tuple[str, ...] = (
        "Risk_XGBoost",
        "T_XGBoost",
        "TO_XGBoost",
        "DR_XGBoost",
        "T_RandomForest",
        "TO_RandomForest",
    )
    risk_model_parameters: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {name: dict(values) for name, values in DEFAULT_RISK_MODEL_PARAMETERS.items()}
    )
    t_xgboost_parameters: dict[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_T_XGBOOST_PARAMETERS)
    )
    tune_t_xgboost: bool = False
    top_fractions: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.50)
    near_constant_threshold: float = 0.999
    bootstrap_iterations: int = 1000
    max_rows: int | None = None


def validate_config(config: ExperimentConfig) -> ExperimentConfig:
    cv = config.cross_validation
    if cv.folds < 2:
        raise ValueError("cross_validation.folds must be at least 2.")
    if cv.tuning_folds < 2:
        raise ValueError("cross_validation.tuning_folds must be at least 2.")
    if not cv.repeated_seeds:
        raise ValueError("cross_validation.repeated_seeds must not be empty.")
    if config.optuna.trials < 1:
        raise ValueError("optuna.trials must be at least 1.")
    if config.bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be at least 1.")
    if not config.top_fractions or any(not 0 < value <= 1 for value in config.top_fractions):
        raise ValueError("top_fractions must contain values in (0, 1].")
    if not 0 < config.near_constant_threshold <= 1:
        raise ValueError("near_constant_threshold must be in (0, 1].")
    if not config.economics.scenarios:
        raise ValueError("economics.scenarios must not be empty.")
    if config.economics.break_even_contact_cost < 0:
        raise ValueError("economics.break_even_contact_cost must be non-negative.")
    names = [scenario.name for scenario in config.economics.scenarios]
    if len(names) != len(set(names)):
        raise ValueError("economics scenario names must be unique.")
    if any(
        scenario.contact_cost < 0 or scenario.saved_customer_value < 0
        for scenario in config.economics.scenarios
    ):
        raise ValueError("economic values must be non-negative.")
    missing_risk = {name for name in config.risk_models if name not in {"Logistic"} and name not in config.risk_model_parameters}
    if missing_risk:
        raise ValueError(f"Missing risk model parameters for: {sorted(missing_risk)}")
    required_t = set(DEFAULT_T_XGBOOST_PARAMETERS)
    if not required_t.issubset(config.t_xgboost_parameters):
        raise ValueError("t_xgboost_parameters is missing required XGBoost fields.")
    return config


def _mapping_of_mappings(raw: Any, fallback: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if raw is None:
        return {name: dict(values) for name, values in fallback.items()}
    if not isinstance(raw, dict):
        raise ValueError("risk_model_parameters must be a mapping.")
    return {str(name): dict(values) for name, values in raw.items()}


def load_config(path: str | Path) -> ExperimentConfig:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cv = raw.get("cross_validation", {})
    optuna = raw.get("optuna", {})
    economics = raw.get("economics", {})
    output = raw.get("output", {})
    config = ExperimentConfig(
        random_seed=int(raw.get("random_seed", 42)),
        cross_validation=CrossValidationConfig(
            folds=int(cv.get("folds", 5)),
            tuning_folds=int(cv.get("tuning_folds", 3)),
            repeated_seeds=tuple(int(v) for v in cv.get("repeated_seeds", [17, 42, 73])),
        ),
        optuna=OptunaConfig(
            trials=int(optuna.get("trials", 24)),
            stability_penalty=float(optuna.get("stability_penalty", 0.25)),
            timeout_seconds=optuna.get("timeout_seconds"),
        ),
        economics=EconomicsConfig(
            scenarios=tuple(
                EconomicScenario(
                    str(name),
                    float(values["contact_cost"]),
                    float(values["saved_customer_value"]),
                )
                for name, values in economics.get(
                    "scenarios",
                    {
                        "Conservative": {"contact_cost": 20.0, "saved_customer_value": 200.0},
                        "Base": {"contact_cost": 30.0, "saved_customer_value": 500.0},
                        "Upside": {"contact_cost": 50.0, "saved_customer_value": 1000.0},
                    },
                ).items()
            ),
            break_even_contact_cost=float(economics.get("break_even_contact_cost", 30.0)),
        ),
        output=OutputConfig(
            save_row_level_scores=bool(output.get("save_row_level_scores", False)),
            generate_figures=bool(output.get("generate_figures", True)),
        ),
        risk_models=tuple(raw.get("risk_models", ["Logistic", "LightGBM", "XGBoost", "CatBoost"])),
        uplift_models=tuple(
            raw.get(
                "uplift_models",
                [
                    "Risk_XGBoost",
                    "T_XGBoost",
                    "TO_XGBoost",
                    "DR_XGBoost",
                    "T_RandomForest",
                    "TO_RandomForest",
                ],
            )
        ),
        risk_model_parameters=_mapping_of_mappings(
            raw.get("risk_model_parameters"), DEFAULT_RISK_MODEL_PARAMETERS
        ),
        t_xgboost_parameters=dict(raw.get("t_xgboost_parameters", DEFAULT_T_XGBOOST_PARAMETERS)),
        tune_t_xgboost=bool(raw.get("tune_t_xgboost", False)),
        top_fractions=tuple(float(v) for v in raw.get("top_fractions", [0.05, 0.10, 0.20, 0.30, 0.50])),
        near_constant_threshold=float(raw.get("near_constant_threshold", 0.999)),
        bootstrap_iterations=int(raw.get("bootstrap_iterations", 1000)),
        max_rows=None if raw.get("max_rows") is None else int(raw["max_rows"]),
    )
    return validate_config(config)
