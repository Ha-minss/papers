from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping as lgb_early_stopping, log_evaluation
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from .config import DEFAULT_RISK_MODEL_PARAMETERS, ExperimentConfig
from .data import PreparedUpliftData
from .metrics import fold_percentile_rank, hajek_uplift_curve_metrics, risk_classification_metrics
from .preprocessing import make_preprocessor

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)


@dataclass(frozen=True)
class RiskComparisonResult:
    metrics: pd.DataFrame
    predictions: dict[str, np.ndarray]
    best_xgboost_params: dict[str, Any]


@dataclass(frozen=True)
class UpliftValidationResult:
    fold_metrics: pd.DataFrame
    aggregate_metrics: pd.DataFrame
    rank_scores: dict[str, np.ndarray]
    tuned_t_params: dict[str, Any]
    tuning_summary: dict[str, float | str | dict[str, Any]]


def prepare_catboost_fold(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    categorical_columns: list[str],
    *,
    min_frequency: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply fold-local rare and unknown category handling for CatBoost."""
    prepared_train = train.copy()
    prepared_valid = valid.copy()
    for column in categorical_columns:
        prepared_train[column] = prepared_train[column].fillna("__MISSING__").astype(str)
        prepared_valid[column] = prepared_valid[column].fillna("__MISSING__").astype(str)
        counts = prepared_train[column].value_counts(dropna=False)
        rare = set(counts[counts < min_frequency].index)
        prepared_train[column] = prepared_train[column].where(
            ~prepared_train[column].isin(rare), "__RARE__"
        )
        known = set(prepared_train[column].unique())
        prepared_valid[column] = prepared_valid[column].where(
            prepared_valid[column].isin(known), "__UNKNOWN__"
        )
    return prepared_train, prepared_valid


def default_xgb_params() -> dict[str, Any]:
    """Return the frozen XGBoost screening parameters used by the paper."""
    return dict(DEFAULT_RISK_MODEL_PARAMETERS["XGBoost"])


def make_xgb_classifier(params: dict[str, Any], seed: int) -> XGBClassifier:
    return XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
        verbosity=0,
    )


def make_xgb_regressor(params: dict[str, Any], seed: int) -> XGBRegressor:
    return XGBRegressor(
        **params,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
        verbosity=0,
    )


def run_risk_model_comparison(
    data: PreparedUpliftData,
    config: ExperimentConfig,
) -> RiskComparisonResult:
    """Reproduce the paper's five-fold OOF risk-model comparison.

    Hyperparameters were selected in a separate three-fold screening stage and
    are frozen in the configuration. The final OOF loop uses early stopping for
    the three boosted-tree models exactly as described in the paper workflow.
    """
    folds = StratifiedKFold(
        n_splits=config.cross_validation.folds,
        shuffle=True,
        random_state=config.random_seed,
    )
    predictions = {name: np.zeros(len(data.target), dtype=float) for name in config.risk_models}
    for fold_no, (train_idx, valid_idx) in enumerate(
        folds.split(data.features, data.strata), 1
    ):
        x_train = data.features.iloc[train_idx]
        x_valid = data.features.iloc[valid_idx]
        y_train = data.target[train_idx]
        y_valid = data.target[valid_idx]
        for model_name in config.risk_models:
            if model_name == "Logistic":
                model = Pipeline(
                    [
                        (
                            "preprocessor",
                            make_preprocessor(
                                data.numeric_columns,
                                data.categorical_columns,
                                scale_numeric=True,
                            ),
                        ),
                        (
                            "model",
                            LogisticRegression(
                                max_iter=3000,
                                C=0.2,
                                solver="liblinear",
                                random_state=config.random_seed,
                            ),
                        ),
                    ]
                )
                model.fit(x_train, y_train)
                predictions[model_name][valid_idx] = model.predict_proba(x_valid)[:, 1]
                continue

            if model_name == "CatBoost":
                train_cat, valid_cat = prepare_catboost_fold(
                    x_train, x_valid, data.categorical_columns
                )
                params = dict(config.risk_model_parameters["CatBoost"])
                model = CatBoostClassifier(
                    **params,
                    iterations=900,
                    loss_function="Logloss",
                    eval_metric="PRAUC",
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=-1,
                    random_seed=config.random_seed,
                    cat_features=data.categorical_columns,
                    od_type="Iter",
                    od_wait=70,
                )
                model.fit(train_cat, y_train, eval_set=(valid_cat, y_valid), verbose=False)
                predictions[model_name][valid_idx] = model.predict_proba(valid_cat)[:, 1]
                continue

            preprocessor = make_preprocessor(data.numeric_columns, data.categorical_columns)
            train_matrix = preprocessor.fit_transform(x_train)
            valid_matrix = preprocessor.transform(x_valid)
            if model_name == "LightGBM":
                params = dict(config.risk_model_parameters["LightGBM"])
                model = LGBMClassifier(
                    **params,
                    objective="binary",
                    verbosity=-1,
                    n_jobs=-1,
                    random_state=config.random_seed,
                    n_estimators=1000,
                )
                model.fit(
                    train_matrix,
                    y_train,
                    eval_set=[(valid_matrix, y_valid)],
                    callbacks=[lgb_early_stopping(70, verbose=False), log_evaluation(0)],
                )
            elif model_name == "XGBoost":
                params = {
                    **config.risk_model_parameters["XGBoost"],
                    "n_estimators": 1000,
                    "early_stopping_rounds": 70,
                }
                model = make_xgb_classifier(params, config.random_seed)
                model.fit(train_matrix, y_train, eval_set=[(valid_matrix, y_valid)], verbose=False)
            else:
                raise ValueError(f"Unsupported risk model: {model_name}")
            predictions[model_name][valid_idx] = model.predict_proba(valid_matrix)[:, 1]

    rows = [
        {"model": name, **risk_classification_metrics(data.target, probability)}
        for name, probability in predictions.items()
    ]
    metrics = pd.DataFrame(rows).sort_values("PR_AUC", ascending=False).reset_index(drop=True)
    uplift_risk_params = {
        **config.risk_model_parameters["XGBoost"],
        "n_estimators": 240,
    }
    return RiskComparisonResult(metrics, predictions, uplift_risk_params)


def _suggest_t_xgb_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 420),
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "min_child_weight": trial.suggest_float("min_child_weight", 8.0, 80.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 20.0, log=True),
    }


def tune_t_xgboost(
    data: PreparedUpliftData,
    config: ExperimentConfig,
) -> tuple[dict[str, Any], dict[str, float | str | dict[str, Any]]]:
    """Tune the T-learner screening model or return the paper's frozen result."""
    if not config.tune_t_xgboost:
        params = dict(config.t_xgboost_parameters)
        return params, {
            "source": "frozen_from_paper_screening",
            "trials": config.optuna.trials,
            "params": params,
        }

    folds = list(
        StratifiedKFold(
            n_splits=config.cross_validation.tuning_folds,
            shuffle=True,
            random_state=config.random_seed,
        ).split(data.features, data.strata)
    )

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_t_xgb_params(trial)
        qini_values = []
        for fold_no, (train_idx, valid_idx) in enumerate(folds):
            preprocessor = make_preprocessor(data.numeric_columns, data.categorical_columns)
            train_matrix = preprocessor.fit_transform(data.features.iloc[train_idx])
            valid_matrix = preprocessor.transform(data.features.iloc[valid_idx])
            train_t = data.treatment[train_idx]
            train_y = data.target[train_idx]
            control = train_t == 0
            treated = train_t == 1
            model0 = make_xgb_classifier(params, config.random_seed + fold_no)
            model1 = make_xgb_classifier(params, config.random_seed + 100 + fold_no)
            model0.fit(train_matrix[control], train_y[control])
            model1.fit(train_matrix[treated], train_y[treated])
            score = model0.predict_proba(valid_matrix)[:, 1] - model1.predict_proba(valid_matrix)[:, 1]
            qini_values.append(
                hajek_uplift_curve_metrics(
                    score,
                    data.target[valid_idx],
                    data.treatment[valid_idx],
                    config.top_fractions,
                )["Qini"]
            )
        mean_qini = float(np.mean(qini_values))
        sd_qini = float(np.std(qini_values, ddof=1)) if len(qini_values) > 1 else 0.0
        trial.set_user_attr("mean_qini", mean_qini)
        trial.set_user_attr("sd_qini", sd_qini)
        return mean_qini - config.optuna.stability_penalty * sd_qini

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.random_seed),
    )
    study.optimize(
        objective,
        n_trials=config.optuna.trials,
        timeout=config.optuna.timeout_seconds,
        show_progress_bar=False,
    )
    summary: dict[str, float | str | dict[str, Any]] = {
        "source": "retuned",
        "objective": float(study.best_value),
        "mean_qini": float(study.best_trial.user_attrs["mean_qini"]),
        "sd_qini": float(study.best_trial.user_attrs["sd_qini"]),
        "params": dict(study.best_params),
    }
    return dict(study.best_params), summary


def _cross_fitted_nuisance_predictions(
    train_matrix,
    train_y: np.ndarray,
    train_t: np.ndarray,
    *,
    params: dict[str, Any],
    cv_seed: int,
    outer_fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create two-fold OOF nuisance predictions for the DR pseudo-outcome."""
    inner_strata = 2 * train_t + train_y
    inner_cv = StratifiedKFold(
        n_splits=2,
        shuffle=True,
        random_state=cv_seed + outer_fold,
    )
    mu0_oof = np.zeros(len(train_y), dtype=float)
    mu1_oof = np.zeros(len(train_y), dtype=float)
    nuisance_params = {
        **params,
        "n_estimators": min(120, int(params["n_estimators"])),
    }
    for inner_no, (inner_train, inner_valid) in enumerate(
        inner_cv.split(train_matrix, inner_strata), 1
    ):
        inner_t = train_t[inner_train]
        inner_y = train_y[inner_train]
        inner_control = inner_t == 0
        inner_treated = inner_t == 1
        model0 = make_xgb_classifier(
            nuisance_params, cv_seed * 3000 + outer_fold * 10 + inner_no
        )
        model1 = make_xgb_classifier(
            nuisance_params, cv_seed * 3000 + 100 + outer_fold * 10 + inner_no
        )
        model0.fit(train_matrix[inner_train][inner_control], inner_y[inner_control])
        model1.fit(train_matrix[inner_train][inner_treated], inner_y[inner_treated])
        mu0_oof[inner_valid] = model0.predict_proba(train_matrix[inner_valid])[:, 1]
        mu1_oof[inner_valid] = model1.predict_proba(train_matrix[inner_valid])[:, 1]
    return mu0_oof, mu1_oof


def _fold_scores(
    train_matrix,
    valid_matrix,
    train_y: np.ndarray,
    train_t: np.ndarray,
    model_names: tuple[str, ...],
    risk_params: dict[str, Any],
    t_params: dict[str, Any],
    *,
    cv_seed: int,
    fold_no: int,
) -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {}
    control = train_t == 0
    treated = train_t == 1
    propensity = float(np.clip(train_t.mean(), 1e-6, 1 - 1e-6))

    if "Risk_XGBoost" in model_names:
        model = make_xgb_classifier(risk_params, cv_seed * 100 + fold_no)
        model.fit(train_matrix, train_y)
        scores["Risk_XGBoost"] = model.predict_proba(valid_matrix)[:, 1]

    if "T_XGBoost" in model_names:
        model0 = make_xgb_classifier(t_params, cv_seed * 1000 + fold_no)
        model1 = make_xgb_classifier(t_params, cv_seed * 1000 + 100 + fold_no)
        model0.fit(train_matrix[control], train_y[control])
        model1.fit(train_matrix[treated], train_y[treated])
        scores["T_XGBoost"] = (
            model0.predict_proba(valid_matrix)[:, 1]
            - model1.predict_proba(valid_matrix)[:, 1]
        )

    reg_params = {
        **t_params,
        "n_estimators": min(220, int(t_params["n_estimators"])),
    }
    transformed_outcome = (
        train_y * (1 - train_t) / (1 - propensity)
        - train_y * train_t / propensity
    )
    if "TO_XGBoost" in model_names:
        model = make_xgb_regressor(reg_params, cv_seed * 2000 + fold_no)
        model.fit(train_matrix, transformed_outcome)
        scores["TO_XGBoost"] = np.clip(model.predict(valid_matrix), -1, 1)

    if "DR_XGBoost" in model_names:
        mu0_train, mu1_train = _cross_fitted_nuisance_predictions(
            train_matrix,
            train_y,
            train_t,
            params=t_params,
            cv_seed=cv_seed,
            outer_fold=fold_no,
        )
        dr_target = (
            mu0_train
            - mu1_train
            + (1 - train_t) * (train_y - mu0_train) / (1 - propensity)
            - train_t * (train_y - mu1_train) / propensity
        )
        model = make_xgb_regressor(reg_params, cv_seed * 4000 + fold_no)
        model.fit(train_matrix, dr_target)
        scores["DR_XGBoost"] = np.clip(model.predict(valid_matrix), -1, 1)

    if "T_RandomForest" in model_names:
        model0 = RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_leaf=35,
            max_features="sqrt",
            n_jobs=-1,
            random_state=cv_seed * 5000 + fold_no,
        )
        model1 = RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_leaf=35,
            max_features="sqrt",
            n_jobs=-1,
            random_state=cv_seed * 5000 + 100 + fold_no,
        )
        model0.fit(train_matrix[control], train_y[control])
        model1.fit(train_matrix[treated], train_y[treated])
        scores["T_RandomForest"] = (
            model0.predict_proba(valid_matrix)[:, 1]
            - model1.predict_proba(valid_matrix)[:, 1]
        )

    if "TO_RandomForest" in model_names:
        model = RandomForestRegressor(
            n_estimators=150,
            max_depth=8,
            min_samples_leaf=45,
            max_features="sqrt",
            n_jobs=-1,
            random_state=cv_seed * 6000 + fold_no,
        )
        model.fit(train_matrix, transformed_outcome)
        scores["TO_RandomForest"] = np.clip(model.predict(valid_matrix), -1, 1)
    return scores


def run_repeated_uplift_validation(
    data: PreparedUpliftData,
    config: ExperimentConfig,
    risk_params: dict[str, Any],
    t_params: dict[str, Any],
) -> UpliftValidationResult:
    rank_sums = {name: np.zeros(len(data.target), dtype=float) for name in config.uplift_models}
    counts = {name: np.zeros(len(data.target), dtype=int) for name in config.uplift_models}
    fold_rows: list[dict[str, Any]] = []

    for repeat_seed in config.cross_validation.repeated_seeds:
        cv = StratifiedKFold(
            n_splits=config.cross_validation.folds,
            shuffle=True,
            random_state=repeat_seed,
        )
        for fold_no, (train_idx, valid_idx) in enumerate(
            cv.split(data.features, data.strata), 1
        ):
            preprocessor = make_preprocessor(data.numeric_columns, data.categorical_columns)
            train_matrix = preprocessor.fit_transform(data.features.iloc[train_idx])
            valid_matrix = preprocessor.transform(data.features.iloc[valid_idx])
            scores = _fold_scores(
                train_matrix,
                valid_matrix,
                data.target[train_idx],
                data.treatment[train_idx],
                config.uplift_models,
                risk_params,
                t_params,
                cv_seed=repeat_seed,
                fold_no=fold_no,
            )
            for model_name, raw_score in scores.items():
                metrics = hajek_uplift_curve_metrics(
                    raw_score,
                    data.target[valid_idx],
                    data.treatment[valid_idx],
                    config.top_fractions,
                )
                fold_rows.append(
                    {
                        "model": model_name,
                        "repeat_seed": repeat_seed,
                        "fold": fold_no,
                        **metrics,
                    }
                )
                rank_score = fold_percentile_rank(raw_score)
                rank_sums[model_name][valid_idx] += rank_score
                counts[model_name][valid_idx] += 1

    expected_count = len(config.cross_validation.repeated_seeds)
    for model_name in config.uplift_models:
        if not np.all(counts[model_name] == expected_count):
            raise RuntimeError(f"Incomplete OOF coverage for {model_name}.")
    rank_scores = {
        name: rank_sums[name] / counts[name]
        for name in config.uplift_models
    }
    fold_metrics = pd.DataFrame(fold_rows)
    aggregate_rows = []
    for model_name in config.uplift_models:
        model_folds = fold_metrics[fold_metrics["model"].eq(model_name)]
        pooled = hajek_uplift_curve_metrics(
            rank_scores[model_name],
            data.target,
            data.treatment,
            config.top_fractions,
        )
        row: dict[str, Any] = {
            "model": model_name,
            "mean_fold_Qini": float(model_folds["Qini"].mean()),
            "sd_fold_Qini": float(model_folds["Qini"].std(ddof=1)),
            "median_fold_Qini": float(model_folds["Qini"].median()),
            "positive_folds": int(model_folds["Qini"].gt(0).sum()),
            "positive_fold_share": float(model_folds["Qini"].gt(0).mean()),
            "mean_Top10_benefit_pp": float(model_folds.get("Top10_benefit_pp", pd.Series(dtype=float)).mean()),
            "sd_Top10_benefit_pp": float(model_folds.get("Top10_benefit_pp", pd.Series(dtype=float)).std(ddof=1)),
            "mean_Top20_benefit_pp": float(model_folds.get("Top20_benefit_pp", pd.Series(dtype=float)).mean()),
            "Qini": pooled["Qini"],
            "AUUC": pooled["AUUC"],
            "ATE_benefit_pp": pooled["ATE_benefit_pp"],
        }
        row.update({key: value for key, value in pooled.items() if key.startswith("Top")})
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows).sort_values("mean_fold_Qini", ascending=False).reset_index(drop=True)
    return UpliftValidationResult(
        fold_metrics=fold_metrics,
        aggregate_metrics=aggregate,
        rank_scores=rank_scores,
        tuned_t_params=t_params,
        tuning_summary={},
    )
