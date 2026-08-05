from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .config import ExperimentConfig
from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from .metrics import classification_metrics, dummy_baseline_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)


@dataclass(frozen=True)
class OOFResult:
    metrics: pd.DataFrame
    predictions: dict[str, np.ndarray]


@dataclass(frozen=True)
class NestedResult:
    metrics: pd.DataFrame
    predictions: dict[str, np.ndarray]
    best_params_by_fold: list[dict[str, Any]]


def make_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=10,
                                sparse_output=True,
                            ),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=1.0,
    )


def prepare_catboost_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for column in CATEGORICAL_FEATURES:
        prepared[column] = prepared[column].fillna("__MISSING__").astype(str)
    return prepared


def default_params(model_name: str) -> dict[str, Any]:
    if model_name == "LightGBM":
        return {
            "learning_rate": 0.035,
            "num_leaves": 15,
            "max_depth": 5,
            "min_child_samples": 50,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.05,
            "reg_lambda": 2.0,
        }
    if model_name == "XGBoost":
        return {
            "learning_rate": 0.035,
            "max_depth": 4,
            "min_child_weight": 5.0,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "gamma": 0.1,
            "reg_alpha": 0.02,
            "reg_lambda": 2.0,
        }
    if model_name == "CatBoost":
        return {
            "learning_rate": 0.035,
            "depth": 5,
            "l2_leaf_reg": 5.0,
            "random_strength": 0.5,
            "bagging_temperature": 0.5,
            "border_count": 128,
        }
    raise ValueError(f"Unsupported model: {model_name}")


def make_model(model_name: str, params: dict[str, Any], seed: int):
    if model_name == "LightGBM":
        model_params = {"n_estimators": 400, **params}
        return LGBMClassifier(
            objective="binary",
            verbosity=-1,
            n_jobs=-1,
            random_state=seed,
            **model_params,
        )
    if model_name == "XGBoost":
        model_params = {"n_estimators": 450, **params}
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
            verbosity=0,
            **model_params,
        )
    if model_name == "CatBoost":
        model_params = {"iterations": 500, **params}
        return CatBoostClassifier(
            loss_function="Logloss",
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
            random_seed=seed,
            cat_features=CATEGORICAL_FEATURES,
            **model_params,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def suggest_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    """Paper search space; tree counts remain fixed by ``make_model``."""
    if model_name == "LightGBM":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.09, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 47),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 25, 100),
            "subsample": trial.suggest_float("subsample", 0.75, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.2, 8.0, log=True),
        }
    if model_name == "XGBoost":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.09, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.75, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.3, 8.0, log=True),
        }
    if model_name == "CatBoost":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.09, log=True),
            "depth": trial.suggest_int("depth", 3, 7),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 12.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
            "border_count": trial.suggest_categorical("border_count", [64, 128]),
        }
    raise ValueError(f"Unsupported model: {model_name}")


def _fit_predict(
    model_name: str,
    params: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    if model_name == "CatBoost":
        model = make_model(model_name, params, seed)
        model.fit(prepare_catboost_frame(x_train), y_train)
        return model.predict_proba(prepare_catboost_frame(x_valid))[:, 1]
    preprocessor = make_preprocessor(scale_numeric=False)
    train_matrix = preprocessor.fit_transform(x_train)
    valid_matrix = preprocessor.transform(x_valid)
    model = make_model(model_name, params, seed)
    model.fit(train_matrix, y_train)
    return model.predict_proba(valid_matrix)[:, 1]


def run_oof_baselines(
    features: pd.DataFrame,
    target: np.ndarray,
    config: ExperimentConfig,
) -> OOFResult:
    model_names = ["DummyMean", "Logistic", "LightGBM", "XGBoost", "CatBoost"]
    folds = list(
        StratifiedKFold(
            n_splits=config.cross_validation.outer_folds,
            shuffle=True,
            random_state=config.random_seed,
        ).split(features, target)
    )
    predictions: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, Any]] = []
    for model_name in model_names:
        oof = np.zeros(len(target), dtype=float)
        for fold_no, (train_idx, valid_idx) in enumerate(folds, 1):
            x_train = features.iloc[train_idx]
            x_valid = features.iloc[valid_idx]
            y_train = target[train_idx]
            if model_name == "DummyMean":
                model = DummyClassifier(strategy="prior")
                model.fit(np.zeros((len(train_idx), 1)), y_train)
                oof[valid_idx] = model.predict_proba(np.zeros((len(valid_idx), 1)))[:, 1]
            elif model_name == "Logistic":
                model = Pipeline(
                    [
                        ("preprocessor", make_preprocessor(scale_numeric=True)),
                        (
                            "model",
                            LogisticRegression(
                                max_iter=3000,
                                solver="liblinear",
                                random_state=config.random_seed + fold_no,
                            ),
                        ),
                    ]
                )
                model.fit(x_train, y_train)
                oof[valid_idx] = model.predict_proba(x_valid)[:, 1]
            else:
                oof[valid_idx] = _fit_predict(
                    model_name,
                    default_params(model_name),
                    x_train,
                    y_train,
                    x_valid,
                    config.random_seed + fold_no,
                )
        predictions[model_name] = oof
        metric_values = (
            dummy_baseline_metrics(target, oof, config.top_fractions)
            if model_name == "DummyMean"
            else classification_metrics(target, oof, config.top_fractions)
        )
        metric_rows.append({"model": model_name, **metric_values})
    return OOFResult(
        pd.DataFrame(metric_rows).sort_values("PR_AUC", ascending=False).reset_index(drop=True),
        predictions,
    )


def _tune_model_for_outer_fold(
    model_name: str,
    features: pd.DataFrame,
    target: np.ndarray,
    config: ExperimentConfig,
    outer_fold: int,
) -> tuple[dict[str, Any], float]:
    inner_cv = list(
        StratifiedKFold(
            n_splits=config.cross_validation.inner_folds,
            shuffle=True,
            random_state=config.random_seed + 100 * outer_fold,
        ).split(features, target)
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model_name)
        fold_scores: list[float] = []
        for inner_no, (train_idx, valid_idx) in enumerate(inner_cv):
            prediction = _fit_predict(
                model_name,
                params,
                features.iloc[train_idx],
                target[train_idx],
                features.iloc[valid_idx],
                config.random_seed + outer_fold * 1000 + inner_no,
            )
            fold_scores.append(classification_metrics(target[valid_idx], prediction)["PR_AUC"])
        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.random_seed + outer_fold),
    )
    study.optimize(
        objective,
        n_trials=config.optuna.trials,
        timeout=config.optuna.timeout_seconds,
        show_progress_bar=False,
    )
    return dict(study.best_params), float(study.best_value)


def run_nested_candidates(
    features: pd.DataFrame,
    target: np.ndarray,
    config: ExperimentConfig,
) -> NestedResult:
    outer_cv = list(
        StratifiedKFold(
            n_splits=config.cross_validation.outer_folds,
            shuffle=True,
            random_state=config.random_seed,
        ).split(features, target)
    )
    predictions = {name: np.zeros(len(target), dtype=float) for name in config.candidate_models}
    parameter_records: list[dict[str, Any]] = []
    for model_name in config.candidate_models:
        for outer_fold, (train_idx, valid_idx) in enumerate(outer_cv, 1):
            x_train = features.iloc[train_idx].reset_index(drop=True)
            y_train = target[train_idx]
            x_valid = features.iloc[valid_idx].reset_index(drop=True)
            best_params, inner_score = _tune_model_for_outer_fold(
                model_name,
                x_train,
                y_train,
                config,
                outer_fold,
            )
            predictions[model_name][valid_idx] = _fit_predict(
                model_name,
                best_params,
                x_train,
                y_train,
                x_valid,
                seed=config.random_seed + outer_fold * 3000,
            )
            parameter_records.append(
                {
                    "model": model_name,
                    "outer_fold": outer_fold,
                    "best_inner_PR_AUC": inner_score,
                    "best_params": best_params,
                }
            )
    rows = [
        {
            "model": f"{model_name}_NestedRaw",
            **classification_metrics(target, probability, config.top_fractions),
        }
        for model_name, probability in predictions.items()
    ]
    return NestedResult(
        pd.DataFrame(rows).sort_values(["PR_AUC", "Brier"], ascending=[False, True]).reset_index(drop=True),
        predictions,
        parameter_records,
    )


def choose_final_parameters(
    model_name: str,
    parameter_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], float, int]:
    candidates = [record for record in parameter_records if record["model"] == model_name]
    if not candidates:
        raise ValueError(f"No nested parameter records found for {model_name}.")
    best = max(candidates, key=lambda record: float(record["best_inner_PR_AUC"]))
    return dict(best["best_params"]), float(best["best_inner_PR_AUC"]), int(best["outer_fold"])


def fit_final_model(
    model_name: str,
    features: pd.DataFrame,
    target: np.ndarray,
    params: dict[str, Any],
    *,
    seed: int = 9000,
) -> tuple[Any, Any | None]:
    """Fit the explanation/deployment model using the best outer-fold parameters."""
    if model_name == "CatBoost":
        model = make_model(model_name, params, seed)
        model.fit(prepare_catboost_frame(features), target)
        return model, None
    preprocessor = make_preprocessor(scale_numeric=False)
    matrix = preprocessor.fit_transform(features)
    model = make_model(model_name, params, seed)
    model.fit(matrix, target)
    return model, preprocessor
