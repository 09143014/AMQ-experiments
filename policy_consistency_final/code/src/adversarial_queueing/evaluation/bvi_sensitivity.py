"""Sensitivity checks for bounded value iteration truncation levels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adversarial_queueing.algorithms.bvi import run_bounded_value_iteration
from adversarial_queueing.envs.service_rate_control import ServiceRateControlConfig, ServiceRateControlEnv
from adversarial_queueing.evaluation.policy_grid import PolicyGridConfig, bvi_policy_grid
from adversarial_queueing.evaluation.rollout import (
    EvaluationConfig,
    evaluate_policy,
    make_bvi_defender_policy,
    random_attacker_policy,
)


@dataclass(frozen=True)
class BVISensitivityConfig:
    max_queue_lengths: tuple[int, ...]
    tolerance: float = 1e-6
    max_iterations: int = 2000
    boundary_mode: str = "clip"
    eval_state: int = 0


def run_bvi_sensitivity(
    env_config: ServiceRateControlConfig,
    sensitivity_config: BVISensitivityConfig,
    evaluation_config: EvaluationConfig,
    policy_grid_config: PolicyGridConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for bound in sensitivity_config.max_queue_lengths:
        bound_env_config = ServiceRateControlConfig(
            lambda_arrival=env_config.lambda_arrival,
            mu_levels=env_config.mu_levels,
            service_costs=env_config.service_costs,
            gamma=env_config.gamma,
            q_congestion=env_config.q_congestion,
            attack_cost=env_config.attack_cost,
            initial_state=env_config.initial_state,
            uniformization_rate=env_config.uniformization_rate,
            robust_defender_actions=env_config.robust_defender_actions,
            bvi_max_queue_length=bound,
            boundary_mode=sensitivity_config.boundary_mode,
        )
        env = ServiceRateControlEnv(bound_env_config)
        result = run_bounded_value_iteration(
            env,
            max_queue_length=bound,
            tolerance=sensitivity_config.tolerance,
            max_iterations=sensitivity_config.max_iterations,
        )
        evaluation = evaluate_policy(
            env,
            defender_policy=make_bvi_defender_policy(result),
            attacker_policy=random_attacker_policy,
            config=evaluation_config,
        )
        _policy_rows, policy_summary = bvi_policy_grid(env, result, policy_grid_config)
        rows.append(
            {
                "max_queue_length": bound,
                "iterations": result.iterations,
                "residual": result.residual,
                "value_at_eval_state": result.values[sensitivity_config.eval_state],
                "average_cost_mean": evaluation.summary["average_cost_mean"],
                "boundary_hit_fraction_mean": evaluation.summary[
                    "boundary_hit_fraction_mean"
                ],
                "first_state_p_high_at_least_threshold": policy_summary[
                    "first_state_p_high_at_least_threshold"
                ],
            }
        )

    summary = _summarize_sensitivity(rows)
    summary.update(
        {
            "num_bounds": len(rows),
            "bounds": list(sensitivity_config.max_queue_lengths),
            "eval_state": sensitivity_config.eval_state,
        }
    )
    return rows, summary


def _summarize_sensitivity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("BVI sensitivity requires at least one bound")
    values = [float(row["value_at_eval_state"]) for row in rows]
    thresholds = [row["first_state_p_high_at_least_threshold"] for row in rows]
    residuals = [float(row["residual"]) for row in rows]
    boundary_hits = [float(row["boundary_hit_fraction_mean"]) for row in rows]
    return {
        "value_range_at_eval_state": max(values) - min(values),
        "last_value_at_eval_state": values[-1],
        "max_residual": max(residuals),
        "max_boundary_hit_fraction_mean": max(boundary_hits),
        "policy_high_thresholds": thresholds,
        "policy_high_threshold_stable": len(set(thresholds)) == 1,
    }
