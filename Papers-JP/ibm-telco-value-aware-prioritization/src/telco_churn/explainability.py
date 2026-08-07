from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from catboost import Pool

from .data import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES
from .modeling import prepare_catboost_frame


def global_shap_importance(
    model_name: str,
    model: Any,
    preprocessor: Any | None,
    features: pd.DataFrame,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    sample_size = min(sample_size, len(features))
    sample_idx = np.random.default_rng(seed).choice(len(features), sample_size, replace=False)
    sample = features.iloc[sample_idx]
    if model_name == "CatBoost":
        values = model.get_feature_importance(
            Pool(prepare_catboost_frame(sample), cat_features=CATEGORICAL_FEATURES),
            type="ShapValues",
        )[:, :-1]
        aggregated = values
        names = MODEL_FEATURES
    else:
        import shap

        matrix = preprocessor.transform(sample)
        raw_values = shap.TreeExplainer(model).shap_values(matrix)
        if isinstance(raw_values, list):
            raw_values = raw_values[-1]
        transformed_names = list(preprocessor.get_feature_names_out())
        names = MODEL_FEATURES
        aggregated = np.zeros((len(sample), len(names)), dtype=float)
        positions = {name: index for index, name in enumerate(names)}
        categorical_sorted = sorted(CATEGORICAL_FEATURES, key=len, reverse=True)
        for column_index, transformed_name in enumerate(transformed_names):
            clean = transformed_name.split("__", 1)[-1]
            original = clean if clean in NUMERIC_FEATURES else next(
                (name for name in categorical_sorted if clean == name or clean.startswith(name + "_")),
                None,
            )
            if original is not None:
                aggregated[:, positions[original]] += raw_values[:, column_index]
    return pd.DataFrame(
        {
            "feature": names,
            "mean_abs_SHAP": np.abs(aggregated).mean(axis=0),
            "mean_SHAP": aggregated.mean(axis=0),
        }
    ).sort_values("mean_abs_SHAP", ascending=False)
