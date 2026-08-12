from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from ..baseline.ovd_study import make_matched_scenarios


@dataclass(frozen=True)
class DemandPanel:
    products: tuple[str, ...]
    dates: np.ndarray
    demand: np.ndarray


def _sheet_rows(data: dict, sheet: str) -> list[dict]:
    header = data[sheet][0]
    return [
        dict(zip(header, row))
        for row in data[sheet][1:]
        if any(value is not None for value in row)
    ]


def load_full_set4(path: Path) -> DemandPanel:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    rows = [
        row
        for row in _sheet_rows(data, "Demand")
        if row["probleminstanceid"] == "SET4"
    ]
    products = tuple(sorted({str(row["materialid"]) for row in rows}))
    dates = np.array(sorted({float(row["deliverydate"]) for row in rows}), dtype=float)
    keys = [(str(row["materialid"]), float(row["deliverydate"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("SET4 demand contains duplicate product-date keys")
    if any(row["quantity"] is None for row in rows):
        raise ValueError("SET4 demand contains missing quantities")
    mapping = {
        (str(row["materialid"]), float(row["deliverydate"])): float(row["quantity"])
        for row in rows
    }
    expected = {(product, date) for product in products for date in dates}
    if set(mapping) != expected:
        raise ValueError("SET4 demand is not a complete product-week panel")
    demand = np.array(
        [[mapping[product, date] for date in dates] for product in products],
        dtype=float,
    )
    if np.any(demand < 0.0):
        raise ValueError("SET4 demand contains negative values")
    return DemandPanel(products=products, dates=dates, demand=demand)


def _validated_score_inputs(
    ensemble: np.ndarray,
    observation: np.ndarray,
    scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(ensemble, dtype=float)
    observed = np.asarray(observation, dtype=float)
    denominator = np.asarray(scale, dtype=float)
    if samples.ndim != 2 or observed.ndim != 1 or denominator.ndim != 1:
        raise ValueError("ensemble must be 2D and observation/scale must be 1D")
    if samples.shape[1] != observed.size or observed.size != denominator.size:
        raise ValueError("ensemble, observation, and scale dimensions must agree")
    if samples.shape[0] == 0 or np.any(denominator <= 0.0):
        raise ValueError("ensemble must be nonempty and scale must be positive")
    return samples / denominator, observed / denominator


def energy_score(
    ensemble: np.ndarray,
    observation: np.ndarray,
    scale: np.ndarray,
) -> float:
    samples, observed = _validated_score_inputs(ensemble, observation, scale)
    observation_term = np.linalg.norm(samples - observed[None, :], axis=1).mean()
    pair_term = np.linalg.norm(
        samples[:, None, :] - samples[None, :, :], axis=2
    ).mean()
    return float(observation_term - 0.5 * pair_term)


def variogram_score(
    ensemble: np.ndarray,
    observation: np.ndarray,
    scale: np.ndarray,
    power: float = 0.5,
) -> float:
    samples, observed = _validated_score_inputs(ensemble, observation, scale)
    if not 0.0 < power <= 2.0:
        raise ValueError("variogram power must be in (0, 2]")
    terms = []
    for left in range(observed.size):
        for right in range(left + 1, observed.size):
            realized = abs(observed[left] - observed[right]) ** power
            forecast = np.mean(abs(samples[:, left] - samples[:, right]) ** power)
            terms.append((realized - forecast) ** 2)
    return float(np.mean(terms)) if terms else 0.0


def marginal_crps(
    ensemble: np.ndarray,
    observation: np.ndarray,
    scale: np.ndarray,
) -> float:
    samples, observed = _validated_score_inputs(ensemble, observation, scale)
    first = np.abs(samples - observed[None, :]).mean(axis=0)
    second = np.abs(samples[:, None, :] - samples[None, :, :]).mean(axis=(0, 1))
    return float(np.mean(first - 0.5 * second))


def classify_predictive_evidence(
    energy_relative_gain: float,
    variogram_relative_gain: float,
    positive_energy_seed_n: int,
    positive_variogram_seed_n: int,
    max_marginal_crps_diff: float,
) -> str:
    if max_marginal_crps_diff > 1e-10:
        return "failure"
    if (
        energy_relative_gain >= 0.005
        and variogram_relative_gain >= 0.01
        and min(positive_energy_seed_n, positive_variogram_seed_n) >= 24
    ):
        return "strong"
    if (
        energy_relative_gain > 0.0
        and variogram_relative_gain > 0.0
        and min(positive_energy_seed_n, positive_variogram_seed_n) >= 18
    ):
        return "partial"
    return "failure"


def score_origin(
    panel: DemandPanel,
    origin: int,
    seed: int,
    scenario_count: int = 100,
    window: int = 8,
) -> dict:
    if origin < 20 or origin >= panel.demand.shape[1]:
        raise ValueError("origin must be in [20, number of weeks)")
    if scenario_count <= 1:
        raise ValueError("scenario_count must exceed one")
    proxy = SimpleNamespace(demand=panel.demand, finished=panel.products)
    joint = make_matched_scenarios(
        proxy, origin, 1, scenario_count, seed, window, "joint"
    )[:, :, 0]
    independent = make_matched_scenarios(
        proxy, origin, 1, scenario_count, seed, window, "independent"
    )[:, :, 0]
    if not np.allclose(
        np.sort(joint, axis=0),
        np.sort(independent, axis=0),
        atol=0.0,
        rtol=0.0,
    ):
        raise RuntimeError("marginal scenario multisets differ")

    observation = panel.demand[:, origin]
    scale = np.maximum(np.std(panel.demand[:, :origin], axis=1, ddof=1), 1.0)
    joint_energy = energy_score(joint, observation, scale)
    independent_energy = energy_score(independent, observation, scale)
    joint_variogram = variogram_score(joint, observation, scale)
    independent_variogram = variogram_score(independent, observation, scale)
    joint_marginal = marginal_crps(joint, observation, scale)
    independent_marginal = marginal_crps(independent, observation, scale)
    return {
        "seed": int(seed),
        "origin": int(origin),
        "target_excel_date": float(panel.dates[origin]),
        "training_end_index": int(origin - 1),
        "target_index": int(origin),
        "scenario_count": int(scenario_count),
        "window": int(window),
        "joint_energy": joint_energy,
        "ind_energy": independent_energy,
        "energy_gain": independent_energy - joint_energy,
        "joint_variogram": joint_variogram,
        "ind_variogram": independent_variogram,
        "variogram_gain": independent_variogram - joint_variogram,
        "joint_marginal_crps": joint_marginal,
        "ind_marginal_crps": independent_marginal,
        "marginal_crps_diff": independent_marginal - joint_marginal,
    }


def _moving_block_interval(
    values: np.ndarray,
    block_length: int = 8,
    repetitions: int = 2000,
    seed: int = 20260812,
) -> dict[str, float]:
    series = np.asarray(values, dtype=float)
    if series.ndim != 1 or series.size < block_length:
        raise ValueError("block bootstrap requires a one-dimensional sufficiently long series")
    rng = np.random.default_rng(seed)
    starts = np.arange(series.size)
    means = np.empty(repetitions, dtype=float)
    blocks_needed = int(np.ceil(series.size / block_length))
    offsets = np.arange(block_length)
    for repetition in range(repetitions):
        selected_starts = rng.choice(starts, size=blocks_needed, replace=True)
        indices = ((selected_starts[:, None] + offsets[None, :]) % series.size).ravel()
        means[repetition] = series[indices[: series.size]].mean()
    return {
        "mean": float(series.mean()),
        "block_length": int(block_length),
        "repetitions": int(repetitions),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
    }


def run_predictive_scoring(
    panel: DemandPanel,
    seeds: range | list[int] = range(1, 31),
    calibration: int = 20,
    scenario_count: int = 100,
    window: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    rows = [
        score_origin(panel, origin, int(seed), scenario_count, window)
        for seed in seeds
        for origin in range(calibration, panel.demand.shape[1])
    ]
    raw = pd.DataFrame(rows)
    seed_summary = (
        raw.groupby("seed", as_index=False)
        .agg(
            origins=("origin", "size"),
            joint_energy=("joint_energy", "mean"),
            ind_energy=("ind_energy", "mean"),
            energy_gain=("energy_gain", "mean"),
            joint_variogram=("joint_variogram", "mean"),
            ind_variogram=("ind_variogram", "mean"),
            variogram_gain=("variogram_gain", "mean"),
            max_abs_marginal_crps_diff=("marginal_crps_diff", lambda x: float(np.max(np.abs(x)))),
        )
    )
    origin_summary = (
        raw.groupby("origin", as_index=False)
        .agg(
            seeds=("seed", "size"),
            target_excel_date=("target_excel_date", "first"),
            joint_energy=("joint_energy", "mean"),
            ind_energy=("ind_energy", "mean"),
            energy_gain=("energy_gain", "mean"),
            joint_variogram=("joint_variogram", "mean"),
            ind_variogram=("ind_variogram", "mean"),
            variogram_gain=("variogram_gain", "mean"),
            max_abs_marginal_crps_diff=("marginal_crps_diff", lambda x: float(np.max(np.abs(x)))),
        )
        .sort_values("origin")
    )
    energy_relative_gain = float(raw["energy_gain"].mean() / raw["ind_energy"].mean())
    variogram_relative_gain = float(
        raw["variogram_gain"].mean() / raw["ind_variogram"].mean()
    )
    positive_energy = int((seed_summary["energy_gain"] > 0.0).sum())
    positive_variogram = int((seed_summary["variogram_gain"] > 0.0).sum())
    max_marginal_diff = float(np.max(np.abs(raw["marginal_crps_diff"])))
    classification = classify_predictive_evidence(
        energy_relative_gain,
        variogram_relative_gain,
        positive_energy,
        positive_variogram,
        max_marginal_diff,
    )
    decision = {
        "classification": classification,
        "joint_predictive_advantage_supported": classification in {"strong", "partial"},
        "weeks_in_source": int(panel.demand.shape[1]),
        "forecast_origins": int(origin_summary.shape[0]),
        "seed_origin_rows": int(raw.shape[0]),
        "energy_relative_gain": energy_relative_gain,
        "variogram_relative_gain": variogram_relative_gain,
        "positive_energy_seed_n": positive_energy,
        "positive_variogram_seed_n": positive_variogram,
        "max_marginal_crps_diff": max_marginal_diff,
        "energy_gain_block_interval": _moving_block_interval(
            origin_summary["energy_gain"].to_numpy(), seed=20260812
        ),
        "variogram_gain_block_interval": _moving_block_interval(
            origin_summary["variogram_gain"].to_numpy(), seed=20260813
        ),
        "thresholds": {
            "marginal_crps_tolerance": 1e-10,
            "strong_energy_relative_gain": 0.005,
            "strong_variogram_relative_gain": 0.01,
            "strong_positive_seed_n_each": 24,
            "partial_energy_relative_gain": "strictly positive",
            "partial_variogram_relative_gain": "strictly positive",
            "partial_positive_seed_n_each": 18,
        },
    }
    return raw, seed_summary, origin_summary, decision


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def write_predictive_outputs(
    panel: DemandPanel,
    results_dir: Path,
    seeds: range | list[int] = range(1, 31),
    calibration: int = 20,
    scenario_count: int = 100,
    window: int = 8,
) -> dict:
    raw, seed_summary, origin_summary, decision = run_predictive_scoring(
        panel,
        seeds=seeds,
        calibration=calibration,
        scenario_count=scenario_count,
        window=window,
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(raw, results_dir / "stage3_predictive_raw.csv")
    _atomic_csv(seed_summary, results_dir / "stage3_seed_summary.csv")
    _atomic_csv(origin_summary, results_dir / "stage3_origin_summary.csv")
    destination = results_dir / "stage3_decision.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return decision


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run 148-week SET4 joint predictive scoring")
    parser.add_argument("--data-json", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("../ijpe_local/results/stage3"))
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--calibration", type=int, default=20)
    parser.add_argument("--scenarios", type=int, default=100)
    parser.add_argument("--window", type=int, default=8)
    args = parser.parse_args()
    decision = write_predictive_outputs(
        load_full_set4(args.data_json),
        args.results_dir,
        seeds=range(1, args.seeds + 1),
        calibration=args.calibration,
        scenario_count=args.scenarios,
        window=args.window,
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
