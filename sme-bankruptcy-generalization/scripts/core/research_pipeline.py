from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FILENAME_RE = re.compile(
    r"^(?P<status>bankrupt|nonbankrupt)_(?P<sector>agriculture|construction|manufacture|retail)_"
    r"(?P<eval>13|14|15|16)_year_(?P<y1>\d{2})_(?P<y2>\d{2})_(?P<y3>\d{2})\.csv$"
)
SECTORS = ("agriculture", "construction", "manufacture", "retail")
RATIO_NAMES = {
    1: ("total_asset_turnover", "TAT"),
    2: ("asset_turnover_days", "ATD"),
    3: ("days_total_receivables_outstanding", "DTR"),
    4: ("inventory_turnover_days", "ITD"),
    5: ("cash_ratio", "L1"),
    6: ("quick_ratio", "L2"),
    7: ("current_ratio", "L3"),
    8: ("return_on_assets", "ROA"),
    9: ("return_on_equity", "ROE"),
    10: ("return_on_sales", "ROS"),
    11: ("return_on_investment", "ROI"),
    12: ("labor_to_revenue_ratio", "LRR"),
    13: ("wages_to_added_value_ratio", "WAR"),
    14: ("labor_productivity", "LP"),
    15: ("debt_to_assets_ratio", "DA"),
    16: ("debt_to_equity_ratio", "DE"),
    17: ("financial_leverage", "FL"),
    18: ("debt_to_income_ratio", "DIR"),
    19: ("debt_service_coverage_ratio", "DCR"),
    20: ("asset_coverage_ratio", "ACR"),
    21: ("bank_liabilities_to_debt_ratio", "BL"),
}


def parse_filename(filename: str) -> dict:
    m = FILENAME_RE.match(Path(filename).name)
    if not m:
        raise ValueError(f"Unexpected filename: {filename}")
    return {
        "target": 1 if m.group("status") == "bankrupt" else 0,
        "sector": m.group("sector"),
        "eval_year": 2000 + int(m.group("eval")),
        "financial_years": tuple(2000 + int(m.group(k)) for k in ("y1", "y2", "y3")),
    }


def all_feature_columns() -> list[str]:
    return [
        f"ratio_{r:02d}_t_minus_{lag}"
        for lag in (3, 2, 1)
        for r in range(1, 22)
    ]


def candidate_60_columns() -> list[str]:
    """Reconstructed 20-ratio x 3-year candidate set; not author-confirmed."""
    return [
        f"ratio_{r:02d}_t_minus_{lag}"
        for lag in (3, 2, 1)
        for r in range(1, 21)
    ]


def reference_60_columns() -> list[str]:
    """Backward-compatible alias for candidate_60_columns()."""
    return candidate_60_columns()


def survey_20_columns() -> list[str]:
    """Most recent year, 20 ratios, matching the 2025 survey's 20-attribute Slovak cells."""
    return [f"ratio_{r:02d}_t_minus_1" for r in range(1, 21)]


def reference_40_columns() -> list[str]:
    return [
        f"ratio_{r:02d}_t_minus_{lag}"
        for lag in (2, 1)
        for r in range(1, 21)
    ]


def feature_dictionary() -> pd.DataFrame:
    rows = []
    for lag in (3, 2, 1):
        for r in range(1, 22):
            name, abbr = RATIO_NAMES[r]
            rows.append({
                "ratio_index": r,
                "feature_column": f"ratio_{r:02d}_t_minus_{lag}",
                "ratio_name": name,
                "abbreviation": abbr,
                "relative_year": f"t-{lag}",
                "in_reference_20": r <= 20,
                "excluded_from_reference": r == 21,
            })
    return pd.DataFrame(rows)


def load_raw_csvs(
    raw_dir: str | Path,
    *,
    expected_csv_files: int = 32,
    expected_columns: int = 64,
    delimiter: str = ";",
    decimal: str = ",",
    missing_values: Sequence[str] = ("NA",),
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    frames = []
    cols = all_feature_columns()
    paths = sorted(raw_dir.glob("*.csv"))
    if len(paths) != expected_csv_files:
        raise ValueError(f"Expected {expected_csv_files} CSV files, found {len(paths)}")
    for path in paths:
        meta = parse_filename(path.name)
        raw = pd.read_csv(
            path, sep=delimiter, decimal=decimal, na_values=list(missing_values)
        )
        if raw.shape[1] != expected_columns:
            raise ValueError(
                f"Expected {expected_columns} columns in {path.name}; got {raw.shape[1]}"
            )
        source_row = raw.iloc[:, 0].astype("Int64")
        x = raw.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        x.columns = cols
        fy = meta["financial_years"]
        out = x.copy()
        out.insert(0, "source_row", source_row)
        out.insert(0, "source_file", path.name)
        out.insert(0, "financial_year_t_minus_1", fy[2])
        out.insert(0, "financial_year_t_minus_2", fy[1])
        out.insert(0, "financial_year_t_minus_3", fy[0])
        out.insert(0, "eval_year", meta["eval_year"])
        out.insert(0, "sector", meta["sector"])
        out.insert(0, "target", meta["target"])
        out.insert(0, "row_id", [
            hashlib.sha1(f"{path.name}:{int(i)}".encode()).hexdigest()[:16]
            for i in source_row
        ])
        frames.append(out)
    frame = pd.concat(frames, ignore_index=True)
    return add_quality_flags(frame)


def add_quality_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    feats = all_feature_columns()
    out["is_all_missing"] = out[feats].isna().all(axis=1)
    # Exact duplicates within the natural source cell. Keep them and flag them.
    cell_keys = ["sector", "eval_year", "target"]
    hashes = pd.util.hash_pandas_object(out[feats], index=False).astype("uint64")
    out["feature_hash"] = hashes.astype(str)
    out["duplicate_group_id"] = (
        out[cell_keys + ["feature_hash"]]
        .astype(str)
        .agg("|".join, axis=1)
        .map(lambda s: hashlib.sha1(s.encode()).hexdigest()[:16])
    )
    out["duplicate_group_size"] = out.groupby("duplicate_group_id")["row_id"].transform("size")
    out["is_exact_duplicate"] = out["duplicate_group_size"] > 1
    # Same sector/year/features but conflicting target.
    conflict_key = ["sector", "eval_year", "feature_hash"]
    out["has_label_conflict"] = out.groupby(conflict_key)["target"].transform("nunique") > 1
    return out


def make_rolling_splits(frame: pd.DataFrame) -> list[tuple[tuple[int, ...], int]]:
    years = sorted(int(y) for y in frame["eval_year"].unique())
    return [(tuple(years[:i]), years[i]) for i in range(1, len(years))]


class TrainWinsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        arr = pd.DataFrame(X)
        self.lower_ = arr.quantile(self.lower).to_numpy(dtype=float)
        self.upper_ = arr.quantile(self.upper).to_numpy(dtype=float)
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        return np.clip(arr, self.lower_, self.upper_)


def make_preprocessor(
    mode: str,
    numeric_columns: Sequence[str],
    model_family: str = "linear",
) -> Pipeline:
    mode = mode.lower()
    steps = []
    if mode == "reference":
        steps = [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    elif mode == "median_std":
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    elif mode == "median_indicator_std":
        steps = [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    elif mode == "practical":
        steps = [
            ("winsor", TrainWinsorizer(0.01, 0.99)),
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    else:
        raise ValueError(f"Unknown preprocessing mode: {mode}")
    return Pipeline(steps)


def safe_smote_k(y: Sequence[int]) -> int:
    y = np.asarray(y, dtype=int)
    n_minority = int(np.bincount(y, minlength=2).min())
    return max(1, min(5, n_minority - 1))


def model_factory(model_name: str, y_train: Sequence[int], imbalance: str = "none", seed: int = 42):
    model_name = model_name.lower()
    imbalance = imbalance.lower()
    y = np.asarray(y_train, dtype=int)
    n_pos = max(1, int(y.sum()))
    n_neg = max(1, int((1 - y).sum()))
    ratio = n_neg / n_pos
    if model_name == "logistic":
        return LogisticRegression(
            C=0.1,
            solver="liblinear",
            max_iter=4000,
            class_weight="balanced" if imbalance == "class_weight" else None,
            random_state=seed,
        )
    if model_name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=140,
            max_depth=3,
            learning_rate=0.05,
            min_child_weight=3,
            gamma=0.0,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=ratio if imbalance == "class_weight" else 1.0,
            n_jobs=2,
            random_state=seed,
            tree_method="hist",
        )
    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=140,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=5,
            min_child_samples=30,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            class_weight="balanced" if imbalance == "class_weight" else None,
            random_state=seed,
            n_jobs=2,
            verbosity=-1,
        )
    raise ValueError(model_name)


def resample_training(X, y, method: str, seed: int = 42):
    method = method.lower()
    if method in {"none", "class_weight"}:
        return X, np.asarray(y, dtype=int)
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTEENN
    if method == "random_under":
        sampler = RandomUnderSampler(random_state=seed)
    elif method == "smote":
        sampler = SMOTE(random_state=seed, k_neighbors=safe_smote_k(y))
    elif method == "smoteenn":
        sampler = SMOTEENN(
            random_state=seed,
            smote=SMOTE(random_state=seed, k_neighbors=safe_smote_k(y)),
        )
    else:
        raise ValueError(method)
    return sampler.fit_resample(X, y)


def gmean_at_threshold(y_true: Sequence[int], scores: Sequence[float], threshold: float) -> float:
    y = np.asarray(y_true, dtype=int)
    pred = (np.asarray(scores, dtype=float) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return float(math.sqrt(sensitivity * specificity))


def choose_gmean_threshold(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Choose the observed-score threshold maximizing sqrt(TPR * TNR)."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(scores, dtype=float)
    fpr, tpr, thresholds = roc_curve(y, p)
    gm = np.sqrt(tpr * (1.0 - fpr))
    idx = int(np.nanargmax(gm))
    threshold = thresholds[idx]
    if not np.isfinite(threshold):
        finite = thresholds[np.isfinite(thresholds)]
        return float(finite.max()) if len(finite) else 0.5
    return float(threshold)


def top_fraction_metrics(y_true: Sequence[int], scores: Sequence[float], fractions=(0.01, 0.03, 0.05)) -> dict:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    result = {}
    total_pos = int(y.sum())
    for f in fractions:
        n = max(1, int(math.ceil(len(y) * f)))
        top = y[order[:n]]
        captured = int(top.sum())
        key = int(round(f * 100))
        result[f"captured_at_{key}pct"] = captured
        result[f"recall_at_{key}pct"] = captured / total_pos if total_pos else np.nan
        result[f"precision_at_{key}pct"] = captured / n
    return result


def calibration_metrics(y_true: Sequence[int], scores: Sequence[float]) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    x = logit(p).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    m.fit(x, y)
    return float(m.intercept_[0]), float(m.coef_[0, 0])


def evaluate_scores(y_true: Sequence[int], scores: Sequence[float], threshold: float = 0.5) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(scores, dtype=float), 1e-12, 1 - 1e-12)
    intercept, slope = calibration_metrics(y, p)
    out = {
        "n_obs": len(y),
        "n_positive": int(y.sum()),
        "actual_rate": float(y.mean()),
        "mean_predicted_probability": float(p.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, p)) if y.sum() else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "gmean_0_5": gmean_at_threshold(y, p, 0.5),
        "gmean_selected": gmean_at_threshold(y, p, threshold),
        "selected_threshold": float(threshold),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }
    out.update(top_fraction_metrics(y, p))
    return out


def _fit_predict_one(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    preprocess_mode: str,
    model_name: str,
    imbalance: str,
    seed: int,
):
    pre = make_preprocessor(preprocess_mode, features, model_family="linear" if model_name == "logistic" else "tree")
    X_train = pre.fit_transform(train.loc[:, features])
    X_test = pre.transform(test.loc[:, features])
    y_train = train["target"].to_numpy(dtype=int)
    X_fit, y_fit = resample_training(X_train, y_train, imbalance, seed)
    model = model_factory(model_name, y_fit, imbalance=imbalance, seed=seed)
    model.fit(X_fit, y_fit)
    p = model.predict_proba(X_test)[:, 1]
    return p, pre, model


def cv_predictions(
    frame: pd.DataFrame,
    features: Sequence[str],
    preprocess_mode: str,
    model_name: str,
    imbalance: str,
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    y = frame["target"].to_numpy(dtype=int)
    min_class = int(np.bincount(y, minlength=2).min())
    splits = max(2, min(n_splits, min_class))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    pred = np.full(len(frame), np.nan)
    for fold, (tr, va) in enumerate(cv.split(np.zeros(len(y)), y)):
        p, _, _ = _fit_predict_one(
            frame.iloc[tr], frame.iloc[va], features, preprocess_mode,
            model_name, imbalance, seed + fold,
        )
        pred[va] = p
    return pred


def rolling_oot_result(
    frame: pd.DataFrame,
    train_years: Sequence[int],
    test_year: int,
    features: Sequence[str],
    preprocess_mode: str,
    model_name: str,
    imbalance: str,
    seed: int = 42,
) -> tuple[dict, pd.DataFrame]:
    train = frame[frame.eval_year.isin(train_years)].reset_index(drop=True)
    test = frame[frame.eval_year.eq(test_year)].reset_index(drop=True)
    oof = cv_predictions(train, features, preprocess_mode, model_name, imbalance, seed=seed)
    threshold = choose_gmean_threshold(train.target, oof)
    p_test, pre, model = _fit_predict_one(train, test, features, preprocess_mode, model_name, imbalance, seed)
    metrics = evaluate_scores(test.target, p_test, threshold)
    metrics.update({
        "train_years": ",".join(map(str, train_years)),
        "test_year": test_year,
        "preprocess": preprocess_mode,
        "feature_count": len(features),
        "model": model_name,
        "imbalance": imbalance,
        "validation_pr_auc": average_precision_score(train.target, oof),
        "validation_roc_auc": roc_auc_score(train.target, oof),
        "validation_gmean_selected": gmean_at_threshold(train.target, oof, threshold),
    })
    predictions = test[["row_id", "sector", "eval_year", "target"]].copy()
    predictions["score"] = p_test
    return metrics, predictions


def random_cv_result(
    frame: pd.DataFrame,
    features: Sequence[str],
    preprocess_mode: str,
    model_name: str,
    imbalance: str,
    seed: int = 42,
) -> tuple[dict, pd.DataFrame]:
    p = cv_predictions(frame.reset_index(drop=True), features, preprocess_mode, model_name, imbalance, seed=seed)
    threshold = choose_gmean_threshold(frame.target, p)
    metrics = evaluate_scores(frame.target, p, threshold)
    metrics.update({
        "preprocess": preprocess_mode,
        "feature_count": len(features),
        "model": model_name,
        "imbalance": imbalance,
        "evaluation": "random_cv",
    })
    predictions = frame[["row_id", "sector", "eval_year", "target"]].reset_index(drop=True).copy()
    predictions["score"] = p
    return metrics, predictions


def pooled_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    preprocess_mode: str,
    model_name: str,
    imbalance: str = "none",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    train_aug = add_sector_onehot(train, features)
    test_aug = add_sector_onehot(test, features)
    aug_features = list(features) + [f"sector_{s}" for s in SECTORS]
    oof = cv_predictions(train_aug, aug_features, preprocess_mode, model_name, imbalance, seed=seed)
    p_test, _, _ = _fit_predict_one(train_aug, test_aug, aug_features, preprocess_mode, model_name, imbalance, seed)
    return oof, p_test


def add_sector_onehot(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    out = frame.copy()
    for sector in SECTORS:
        out[f"sector_{sector}"] = (out["sector"] == sector).astype(float)
    return out


def sector_specific_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    preprocess_mode: str,
    model_name: str,
    imbalance: str = "none",
    seed: int = 42,
) -> np.ndarray:
    pred = np.full(len(test), np.nan)
    for i, sector in enumerate(SECTORS):
        tr = train[train.sector.eq(sector)].reset_index(drop=True)
        mask = test.sector.eq(sector).to_numpy()
        te = test.loc[mask].reset_index(drop=True)
        if te.empty:
            continue
        p, _, _ = _fit_predict_one(tr, te, features, preprocess_mode, model_name, imbalance, seed + i)
        pred[np.where(mask)[0]] = p
    return pred


def partial_pool_calibration(
    y_train: Sequence[int],
    pooled_oof: Sequence[float],
    train_sector: Sequence[str],
    pooled_test: Sequence[float],
    test_sector: Sequence[str],
    C: float = 0.1,
) -> np.ndarray:
    def matrix(scores, sectors):
        base = logit(np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)).reshape(-1, 1)
        d = pd.get_dummies(pd.Categorical(sectors, categories=SECTORS), dtype=float)
        d = d.reindex(columns=list(SECTORS), fill_value=0.0).to_numpy()
        return np.column_stack([base, d])
    Xtr = matrix(pooled_oof, train_sector)
    Xte = matrix(pooled_test, test_sector)
    cal = LogisticRegression(C=C, solver="liblinear", max_iter=3000)
    cal.fit(Xtr, np.asarray(y_train, dtype=int))
    return cal.predict_proba(Xte)[:, 1]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def foldwise_cv_result(
    frame: pd.DataFrame,
    features: Sequence[str],
    preprocess_mode: str,
    model_name: str,
    imbalance: str,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[dict, pd.DataFrame]:
    """Return fold-level benchmark summaries plus complete out-of-fold scores."""
    frame = frame.reset_index(drop=True)
    y = frame["target"].to_numpy(dtype=int)
    min_class = int(np.bincount(y, minlength=2).min())
    splits = max(2, min(n_splits, min_class))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    pred = np.full(len(frame), np.nan)
    fold_rows = []
    for fold, (tr, va) in enumerate(cv.split(np.zeros(len(y)), y)):
        p, _, _ = _fit_predict_one(
            frame.iloc[tr], frame.iloc[va], features, preprocess_mode,
            model_name, imbalance, seed + fold,
        )
        pred[va] = p
        best_t = choose_gmean_threshold(y[va], p)
        fold_rows.append({
            "fold": fold,
            "gmean_0_5": gmean_at_threshold(y[va], p, 0.5),
            "gmean_posthoc_best": gmean_at_threshold(y[va], p, best_t),
            "posthoc_threshold": best_t,
            "roc_auc": roc_auc_score(y[va], p),
            "pr_auc": average_precision_score(y[va], p),
            "n_positive": int(y[va].sum()),
        })
    folds = pd.DataFrame(fold_rows)
    threshold = choose_gmean_threshold(y, pred)
    result = evaluate_scores(y, pred, threshold)
    result.update({
        "n_folds": splits,
        "mean_fold_gmean_0_5": float(folds.gmean_0_5.mean()),
        "std_fold_gmean_0_5": float(folds.gmean_0_5.std(ddof=1)),
        "mean_fold_gmean_posthoc_best": float(folds.gmean_posthoc_best.mean()),
        "std_fold_gmean_posthoc_best": float(folds.gmean_posthoc_best.std(ddof=1)),
        "mean_fold_roc_auc": float(folds.roc_auc.mean()),
        "std_fold_roc_auc": float(folds.roc_auc.std(ddof=1)),
        "mean_fold_pr_auc": float(folds.pr_auc.mean()),
        "std_fold_pr_auc": float(folds.pr_auc.std(ddof=1)),
    })
    predictions = frame[["row_id", "sector", "eval_year", "target"]].copy()
    predictions["score"] = pred
    return result, predictions


def sector_specific_oof_and_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    preprocess_mode: str,
    model_name: str,
    imbalance: str = "none",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fitted train scores and future scores from one model per sector."""
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    oof = np.full(len(train), np.nan)
    ptest = np.full(len(test), np.nan)
    sectors = sorted(set(train.sector.unique()).union(test.sector.unique()))
    for i, sector in enumerate(sectors):
        tr_mask = train.sector.eq(sector).to_numpy()
        te_mask = test.sector.eq(sector).to_numpy()
        tr = train.loc[tr_mask].reset_index(drop=True)
        te = test.loc[te_mask].reset_index(drop=True)
        if tr.empty:
            raise ValueError(f"No training rows for sector {sector}")
        if tr.target.nunique() < 2:
            raise ValueError(f"Training sector {sector} has only one class")
        sector_oof = cv_predictions(
            tr, features, preprocess_mode, model_name, imbalance, seed=seed + 100 * i,
        )
        oof[np.where(tr_mask)[0]] = sector_oof
        if not te.empty:
            sector_test, _, _ = _fit_predict_one(
                tr, te, features, preprocess_mode, model_name, imbalance, seed + 100 * i,
            )
            ptest[np.where(te_mask)[0]] = sector_test
    if not np.isfinite(oof).all() or not np.isfinite(ptest).all():
        raise RuntimeError("Incomplete sector-specific predictions")
    return oof, ptest


def _partial_pool_matrix(scores: Sequence[float], sectors: Sequence[str]) -> np.ndarray:
    base = logit(np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)).reshape(-1, 1)
    d = pd.get_dummies(pd.Categorical(sectors, categories=SECTORS), dtype=float)
    d = d.reindex(columns=list(SECTORS), fill_value=0.0).to_numpy()
    return np.column_stack([base, d])


def partial_pool_oof_and_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    preprocess_mode: str,
    model_name: str,
    imbalance: str = "none",
    seed: int = 42,
    calibration_C: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Pooled base learner with cross-fitted ridge sector-intercept calibration."""
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    pooled_oof, pooled_test = pooled_predictions(
        train, test, features, preprocess_mode, model_name, imbalance, seed,
    )
    y = train.target.to_numpy(dtype=int)
    Xcal = _partial_pool_matrix(pooled_oof, train.sector)
    Xtest = _partial_pool_matrix(pooled_test, test.sector)
    min_class = int(np.bincount(y, minlength=2).min())
    splits = max(2, min(5, min_class))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed + 991)
    cal_oof = np.full(len(train), np.nan)
    for fold, (tr, va) in enumerate(cv.split(Xcal, y)):
        cal = LogisticRegression(
            C=calibration_C, solver="liblinear", max_iter=3000, random_state=seed + fold,
        )
        cal.fit(Xcal[tr], y[tr])
        cal_oof[va] = cal.predict_proba(Xcal[va])[:, 1]
    final_cal = LogisticRegression(
        C=calibration_C, solver="liblinear", max_iter=3000, random_state=seed,
    )
    final_cal.fit(Xcal, y)
    cal_test = final_cal.predict_proba(Xtest)[:, 1]
    return cal_oof, cal_test
