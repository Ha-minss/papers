from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from .research_pipeline import SECTORS


def annual_event_summary(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby('eval_year', as_index=False)['target'].agg(
        n_obs='size', n_positive='sum'
    )
    grouped['event_rate'] = grouped['n_positive'] / grouped['n_obs']
    return grouped.sort_values('eval_year').reset_index(drop=True)


def standardized_wasserstein_drift(
    frame: pd.DataFrame,
    features: Sequence[str],
    reference_year: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    reference = frame[frame['eval_year'].eq(reference_year)]
    years = sorted(int(y) for y in frame['eval_year'].dropna().unique())
    for feature in features:
        ref = pd.to_numeric(reference[feature], errors='coerce').dropna().to_numpy(float)
        if len(ref) == 0:
            scale = np.nan
        else:
            q25, q75 = np.nanpercentile(ref, [25, 75])
            scale = float(q75 - q25)
            if not np.isfinite(scale) or scale <= 0:
                scale = float(np.nanstd(ref))
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
        for year in years:
            cur = pd.to_numeric(
                frame.loc[frame['eval_year'].eq(year), feature], errors='coerce'
            ).dropna().to_numpy(float)
            if len(ref) == 0 or len(cur) == 0:
                value = np.nan
            elif year == reference_year:
                value = 0.0
            else:
                value = float(wasserstein_distance(ref, cur) / scale)
            rows.append({
                'feature': feature,
                'eval_year': year,
                'reference_year': reference_year,
                'standardized_wasserstein': value,
                'reference_nonmissing': len(ref),
                'current_nonmissing': len(cur),
            })
    return pd.DataFrame(rows)



def population_stability_index(
    reference_values: Sequence[float],
    current_values: Sequence[float],
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    reference = pd.Series(np.asarray(reference_values, dtype=float)).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    current = pd.Series(np.asarray(current_values, dtype=float)).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(reference) == 0 or len(current) == 0:
        return np.nan
    quantiles = np.linspace(0, 1, int(n_bins) + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 2:
        return 0.0 if np.allclose(reference, current) else np.nan
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_share = np.clip(ref_counts / ref_counts.sum(), epsilon, None)
    cur_share = np.clip(cur_counts / cur_counts.sum(), epsilon, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))

def calibration_bins(
    y_true: Sequence[int],
    scores: Sequence[float],
    n_bins: int = 10,
) -> pd.DataFrame:
    data = pd.DataFrame({
        'target': np.asarray(y_true, dtype=int),
        'score': np.asarray(scores, dtype=float),
    }).replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return pd.DataFrame(columns=['bin', 'count', 'mean_predicted', 'observed_rate', 'score_min', 'score_max'])
    ranks = data['score'].rank(method='first')
    bins = min(int(n_bins), len(data))
    data['bin'] = pd.qcut(ranks, q=bins, labels=False, duplicates='drop')
    out = data.groupby('bin', as_index=False).agg(
        count=('target', 'size'),
        mean_predicted=('score', 'mean'),
        observed_rate=('target', 'mean'),
        score_min=('score', 'min'),
        score_max=('score', 'max'),
    )
    out['bin'] = out['bin'].astype(int) + 1
    return out


def event_count_performance_gap(metrics: pd.DataFrame) -> pd.DataFrame:
    required = ['target_year', 'model', 'sector', 'train_positive', 'structure', 'pr_auc']
    missing = [c for c in required if c not in metrics]
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    subset = metrics[metrics['structure'].isin(['pooled', 'sector_specific'])].copy()
    wide = subset.pivot_table(
        index=['target_year', 'model', 'sector', 'train_positive'],
        columns='structure',
        values='pr_auc',
        aggfunc='first',
    ).reset_index()
    wide.columns.name = None
    wide = wide.dropna(subset=['pooled', 'sector_specific']).copy()
    wide['pr_auc_gap'] = wide['sector_specific'] - wide['pooled']
    return wide


def effective_sector_slopes(
    shared_coefficients: np.ndarray,
    raw_interaction_coefficients: np.ndarray,
    interaction_scale: float,
    sectors: Sequence[str] = SECTORS,
) -> dict[str, np.ndarray]:
    shared = np.asarray(shared_coefficients, dtype=float)
    interactions = np.asarray(raw_interaction_coefficients, dtype=float)
    expected = (len(sectors) - 1, len(shared))
    if interactions.shape != expected:
        raise ValueError(f'Expected interaction shape {expected}; got {interactions.shape}')
    scaled = float(interaction_scale) * interactions
    out: dict[str, np.ndarray] = {}
    for idx, sector in enumerate(sectors[:-1]):
        out[sector] = shared + scaled[idx]
    out[sectors[-1]] = shared - scaled.sum(axis=0)
    return out
