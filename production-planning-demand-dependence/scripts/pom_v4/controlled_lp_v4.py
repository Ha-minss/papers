from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..baseline.controlled_lp import LPB, solve_policy as baseline_solve_policy
from ..baseline.ovd_study import make_matched_scenarios

from .structural_architecture import (
    ArchitectureMode,
    ArchitectureSpec,
    calibration_upstream_weights,
    capacity_pools,
)


@dataclass(frozen=True)
class PolicySolution:
    action: dict[str, float]
    shared_action: dict[str, float]
    dedicated_action: dict[str, float]
    objective: float


def _active_ratio(inst, spec: ArchitectureSpec) -> Mapping[tuple[str, str], float]:
    return {} if spec.mode is ArchitectureMode.NO_BOM else inst.ratio


def _successors(inst, ratio: Mapping[tuple[str, str], float]) -> dict[str, list[str]]:
    return {
        material: [
            successor
            for successor in inst.materials
            if ratio.get((material, successor), 0.0) > 0.0
        ]
        for material in inst.materials
    }


def _uses_baseline_solver(spec: ArchitectureSpec) -> bool:
    return (
        spec.mode is ArchitectureMode.BASELINE
        or (spec.mode is ArchitectureMode.POOLING and spec.alpha == 1.0)
    )


def solve_policy_v4(
    inst,
    scenarios: np.ndarray,
    state: dict,
    spec: ArchitectureSpec,
    weights: Mapping[str, float],
) -> PolicySolution | None:
    if _uses_baseline_solver(spec):
        action = baseline_solve_policy(inst, scenarios, state, cap_scale=1.0)
        if action is None:
            return None
        upstream = set(inst.pm["OPER008"])
        return PolicySolution(
            action=action,
            shared_action={p: (action[p] if p in upstream else 0.0) for p in inst.materials},
            dedicated_action={p: 0.0 for p in inst.materials},
            objective=float("nan"),
        )

    scenario_count, _, horizon = scenarios.shape
    builder = LPB()
    q: dict[tuple[int, str, int], int] = {}
    inventory: dict[tuple[int, str, int], int] = {}
    backlog: dict[tuple[int, str, int], int] = {}

    for scenario in range(scenario_count):
        for step in range(horizon):
            for material in inst.materials:
                q[scenario, material, step] = builder.var()
                inventory[scenario, material, step] = builder.var(
                    cost=inst.invcost[material] / scenario_count
                )
                backlog[scenario, material, step] = builder.var(
                    ub=0.0 if material in inst.intermediate else np.inf,
                    cost=inst.bocost[material] / scenario_count,
                )

    pooling = spec.mode is ArchitectureMode.POOLING and spec.alpha < 1.0
    shared_q: dict[tuple[int, str, int], int] = {}
    dedicated_q: dict[tuple[int, str, int], int] = {}
    upstream = inst.pm["OPER008"]
    pools = capacity_pools(inst, spec, weights) if pooling else None
    if pooling:
        for scenario in range(scenario_count):
            for step in range(horizon):
                for material in upstream:
                    shared_q[scenario, material, step] = builder.var()
                    dedicated_q[scenario, material, step] = builder.var()
                    builder.con(
                        {
                            q[scenario, material, step]: 1.0,
                            shared_q[scenario, material, step]: -1.0,
                            dedicated_q[scenario, material, step]: -1.0,
                        },
                        0.0,
                        0.0,
                    )

    for scenario in range(1, scenario_count):
        for material in inst.materials:
            builder.con(
                {
                    q[scenario, material, 0]: 1.0,
                    q[0, material, 0]: -1.0,
                },
                0.0,
                0.0,
            )
        if pooling:
            for material in upstream:
                builder.con(
                    {
                        shared_q[scenario, material, 0]: 1.0,
                        shared_q[0, material, 0]: -1.0,
                    },
                    0.0,
                    0.0,
                )
                builder.con(
                    {
                        dedicated_q[scenario, material, 0]: 1.0,
                        dedicated_q[0, material, 0]: -1.0,
                    },
                    0.0,
                    0.0,
                )

    ratio = _active_ratio(inst, spec)
    successors = _successors(inst, ratio)
    finished_position = {material: index for index, material in enumerate(inst.finished)}

    for scenario in range(scenario_count):
        for step in range(horizon):
            for material in inst.materials:
                coefficients = {
                    q[scenario, material, step]: 1.0,
                    inventory[scenario, material, step]: -1.0,
                    backlog[scenario, material, step]: 1.0,
                }
                if step > 0:
                    coefficients[inventory[scenario, material, step - 1]] = 1.0
                    coefficients[backlog[scenario, material, step - 1]] = -1.0
                demand = (
                    float(scenarios[scenario, finished_position[material], step])
                    if material in finished_position
                    else 0.0
                )
                for successor in successors[material]:
                    coefficients[q[scenario, successor, step]] = (
                        coefficients.get(q[scenario, successor, step], 0.0)
                        - ratio[material, successor]
                    )
                initial_net = (
                    state["inv"].get(material, 0.0)
                    - state["bo"].get(material, 0.0)
                    if step == 0
                    else 0.0
                )
                rhs = demand - initial_net
                builder.con(coefficients, rhs, rhs)

            packaging = {
                q[scenario, material, step]: inst.prodtime[material]
                for material in inst.pm["OPER004"]
            }
            builder.con(
                packaging,
                -np.inf,
                inst.weekly_cap["OPER004"],
            )

            if pooling:
                builder.con(
                    {
                        shared_q[scenario, material, step]: inst.prodtime[material]
                        for material in upstream
                    },
                    -np.inf,
                    pools.shared_capacity,
                )
                for material in upstream:
                    builder.con(
                        {
                            dedicated_q[scenario, material, step]: inst.prodtime[material]
                        },
                        -np.inf,
                        pools.dedicated_capacity[material],
                    )
            else:
                upstream_multiplier = (
                    spec.upstream_multiplier
                    if spec.mode is ArchitectureMode.NONBINDING_UPSTREAM
                    else 1.0
                )
                builder.con(
                    {
                        q[scenario, material, step]: inst.prodtime[material]
                        for material in upstream
                    },
                    -np.inf,
                    inst.weekly_cap["OPER008"] * upstream_multiplier,
                )

    result = builder.solve()
    if result.x is None:
        return None

    action = {
        material: float(result.x[q[0, material, 0]])
        for material in inst.materials
    }
    shared_action = {material: 0.0 for material in inst.materials}
    dedicated_action = {material: 0.0 for material in inst.materials}
    if pooling:
        for material in upstream:
            shared_action[material] = float(result.x[shared_q[0, material, 0]])
            dedicated_action[material] = float(result.x[dedicated_q[0, material, 0]])
    else:
        for material in upstream:
            shared_action[material] = action[material]

    return PolicySolution(
        action=action,
        shared_action=shared_action,
        dedicated_action=dedicated_action,
        objective=float(result.fun),
    )


def execute_week_v4(
    inst,
    action: Mapping[str, float],
    state: dict,
    actual_demand: Mapping[str, float],
    spec: ArchitectureSpec,
) -> tuple[dict, dict[str, float]]:
    ratio = _active_ratio(inst, spec)
    successors = _successors(inst, ratio)
    new_inventory: dict[str, float] = {}
    new_backlog: dict[str, float] = {}
    for material in inst.materials:
        consumption = sum(
            ratio.get((material, successor), 0.0) * action[successor]
            for successor in successors[material]
        )
        net = (
            state["inv"].get(material, 0.0)
            - state["bo"].get(material, 0.0)
            + action[material]
            - actual_demand.get(material, 0.0)
            - consumption
        )
        if material in inst.intermediate:
            new_inventory[material] = max(net, 0.0)
            new_backlog[material] = 0.0
        else:
            new_inventory[material] = max(net, 0.0)
            new_backlog[material] = max(-net, 0.0)
    holding = sum(
        inst.invcost[material] * new_inventory[material]
        for material in inst.materials
    )
    backorder = sum(
        inst.bocost[material] * new_backlog[material]
        for material in inst.finished
    )
    return (
        {"inv": new_inventory, "bo": new_backlog},
        {"holding": holding, "backorder": backorder, "total": holding + backorder},
    )


def _pool_utilization(
    inst,
    solution: PolicySolution,
    spec: ArchitectureSpec,
    weights: Mapping[str, float],
) -> float:
    upstream = inst.pm["OPER008"]
    if spec.mode is ArchitectureMode.POOLING and spec.alpha < 1.0:
        pools = capacity_pools(inst, spec, weights)
        utilizations: list[float] = []
        shared_load = sum(
            inst.prodtime[p] * solution.shared_action[p] for p in upstream
        )
        if pools.shared_capacity > 0.0:
            utilizations.append(shared_load / pools.shared_capacity)
        for material in upstream:
            cap = pools.dedicated_capacity[material]
            load = inst.prodtime[material] * solution.dedicated_action[material]
            if cap > 0.0:
                utilizations.append(load / cap)
            elif load > 1e-10:
                return float("inf")
        return max(utilizations, default=0.0)

    multiplier = (
        spec.upstream_multiplier
        if spec.mode is ArchitectureMode.NONBINDING_UPSTREAM
        else 1.0
    )
    load = sum(inst.prodtime[p] * solution.action[p] for p in upstream)
    return load / (inst.weekly_cap["OPER008"] * multiplier)


def run_policy_v4(
    inst,
    treatment: str,
    seed: int,
    spec: ArchitectureSpec,
    weights: Mapping[str, float] | None = None,
    S: int = 20,
    H: int = 4,
    window: int = 8,
    calibration: int = 20,
) -> dict:
    if weights is None:
        weights = calibration_upstream_weights(inst, calibration=calibration)
    state = {
        "inv": {material: 0.0 for material in inst.materials},
        "bo": {material: 0.0 for material in inst.materials},
    }
    totals = {"holding": 0.0, "backorder": 0.0, "total": 0.0}
    upstream_quantity = 0.0
    max_oper004 = 0.0
    max_upstream_pool = 0.0
    actions: list[dict[str, float]] = []

    for origin in range(calibration, len(inst.dates)):
        horizon = min(H, len(inst.dates) - origin)
        scenarios = make_matched_scenarios(
            inst,
            origin,
            horizon,
            S,
            seed,
            window,
            treatment,
        )
        solution = solve_policy_v4(inst, scenarios, state, spec, weights)
        if solution is None:
            raise RuntimeError(
                f"infeasible policy: treatment={treatment} seed={seed} origin={origin}"
            )
        actual = {
            material: float(inst.demand[inst.fidx[material], origin])
            for material in inst.finished
        }
        state, costs = execute_week_v4(inst, solution.action, state, actual, spec)
        for key in totals:
            totals[key] += costs[key]
        upstream_quantity += sum(solution.action[p] for p in inst.pm["OPER008"])
        packaging_load = sum(
            inst.prodtime[p] * solution.action[p] for p in inst.pm["OPER004"]
        )
        max_oper004 = max(
            max_oper004,
            packaging_load / inst.weekly_cap["OPER004"],
        )
        max_upstream_pool = max(
            max_upstream_pool,
            _pool_utilization(inst, solution, spec, weights),
        )
        actions.append(solution.action)

    return {
        **totals,
        "upstream_quantity": upstream_quantity,
        "max_oper004_util": max_oper004,
        "max_upstream_pool_util": max_upstream_pool,
        "actions": actions,
    }


def _mean_normalized_l1(
    joint_actions: list[Mapping[str, float]],
    independent_actions: list[Mapping[str, float]],
    materials: list[str],
) -> float:
    values = []
    for joint, independent in zip(joint_actions, independent_actions):
        numerator = sum(abs(joint[p] - independent[p]) for p in materials)
        denominator = max(
            sum(abs(joint[p]) for p in materials),
            sum(abs(independent[p]) for p in materials),
            1e-12,
        )
        values.append(numerator / denominator)
    return float(np.mean(values)) if values else 0.0


def paired_run_v4(
    inst,
    seed: int,
    spec: ArchitectureSpec,
    weights: Mapping[str, float] | None = None,
    **kwargs,
) -> dict:
    if weights is None:
        weights = calibration_upstream_weights(
            inst,
            calibration=int(kwargs.get("calibration", 20)),
        )
    joint = run_policy_v4(inst, "joint", seed, spec, weights, **kwargs)
    independent = run_policy_v4(
        inst,
        "independent",
        seed,
        spec,
        weights,
        **kwargs,
    )
    joint_cost = joint["total"]
    independent_cost = independent["total"]
    return {
        "seed": seed,
        "architecture": spec.mode.value,
        "alpha": spec.alpha,
        "upstream_multiplier": spec.upstream_multiplier,
        "joint_cost": joint_cost,
        "ind_cost": independent_cost,
        "ovd_pct": 100.0 * (independent_cost - joint_cost) / joint_cost if joint_cost else 0.0,
        "joint_holding": joint["holding"],
        "ind_holding": independent["holding"],
        "joint_backorder": joint["backorder"],
        "ind_backorder": independent["backorder"],
        "joint_upstream_quantity": joint["upstream_quantity"],
        "ind_upstream_quantity": independent["upstream_quantity"],
        "joint_max_oper004_util": joint["max_oper004_util"],
        "ind_max_oper004_util": independent["max_oper004_util"],
        "joint_max_upstream_pool_util": joint["max_upstream_pool_util"],
        "ind_max_upstream_pool_util": independent["max_upstream_pool_util"],
        "mean_action_l1": _mean_normalized_l1(
            joint["actions"],
            independent["actions"],
            inst.materials,
        ),
    }
