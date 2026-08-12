from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Mapping

import numpy as np


class ArchitectureMode(str, Enum):
    BASELINE = "baseline"
    NONBINDING_UPSTREAM = "nonbinding_upstream"
    POOLING = "pooling"
    NO_BOM = "no_bom"


@dataclass(frozen=True)
class ArchitectureSpec:
    mode: ArchitectureMode
    alpha: float = 1.0
    upstream_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.upstream_multiplier <= 0.0:
            raise ValueError("upstream_multiplier must be positive")

    @classmethod
    def baseline(cls) -> "ArchitectureSpec":
        return cls(ArchitectureMode.BASELINE)

    @classmethod
    def pooling(cls, alpha: float) -> "ArchitectureSpec":
        return cls(ArchitectureMode.POOLING, alpha=float(alpha))

    @classmethod
    def nonbinding(cls, multiplier: float) -> "ArchitectureSpec":
        return cls(
            ArchitectureMode.NONBINDING_UPSTREAM,
            upstream_multiplier=float(multiplier),
        )

    @classmethod
    def no_bom(cls) -> "ArchitectureSpec":
        return cls(ArchitectureMode.NO_BOM)


@dataclass(frozen=True)
class CapacityPools:
    shared_capacity: float
    dedicated_capacity: Mapping[str, float]


def gross_requirements(inst, demand: np.ndarray) -> dict[str, np.ndarray]:
    """Propagate finished demand down the supplied acyclic BOM."""

    @lru_cache(None)
    def coefficients(material: str) -> dict[str, float]:
        result = {material: 1.0} if material in inst.finished else {}
        for successor in inst.successors[material]:
            for finished, value in coefficients(successor).items():
                result[finished] = (
                    result.get(finished, 0.0)
                    + inst.ratio[(material, successor)] * value
                )
        return result

    output: dict[str, np.ndarray] = {}
    for material in inst.materials:
        series = np.zeros(demand.shape[1], dtype=float)
        for finished, coefficient in coefficients(material).items():
            series += coefficient * demand[inst.fidx[finished], :]
        output[material] = series
    return output


def calibration_upstream_weights(
    inst,
    calibration: int = 20,
    floor: float = 0.01,
) -> dict[str, float]:
    if calibration <= 0 or calibration > inst.demand.shape[1]:
        raise ValueError("calibration must select a nonempty prefix of demand")
    if floor < 0.0 or floor >= 1.0:
        raise ValueError("floor must be in [0, 1)")

    requirements = gross_requirements(inst, inst.demand[:, :calibration])
    upstream = inst.pm["OPER008"]
    raw = {
        material: float(inst.prodtime[material] * np.mean(requirements[material]))
        for material in upstream
    }
    total = sum(raw.values())
    weights = {
        material: (raw[material] / total if total > 0.0 else 1.0 / len(upstream))
        for material in upstream
    }
    if any(value == 0.0 for value in weights.values()):
        weights = {material: max(value, floor) for material, value in weights.items()}
        scale = sum(weights.values())
        weights = {material: value / scale for material, value in weights.items()}
    return weights


def capacity_pools(
    inst,
    spec: ArchitectureSpec,
    weights: Mapping[str, float],
) -> CapacityPools:
    upstream = inst.pm["OPER008"]
    if set(weights) != set(upstream):
        raise ValueError("weights must cover exactly the OPER008 materials")
    if not np.isclose(sum(weights.values()), 1.0, atol=1e-12):
        raise ValueError("weights must sum to one")

    capacity = float(inst.weekly_cap["OPER008"])
    alpha = spec.alpha if spec.mode is ArchitectureMode.POOLING else 1.0
    return CapacityPools(
        shared_capacity=alpha * capacity,
        dedicated_capacity={
            material: (1.0 - alpha) * float(weights[material]) * capacity
            for material in upstream
        },
    )
