"""Policy diagnostics for the service-rate-control benchmark."""

from __future__ import annotations

from typing import Any

import numpy as np

from adversarial_queueing.algorithms.amq import LinearAMQTrainer
from adversarial_queueing.algorithms.bvi import BVIResult
from adversarial_queueing.algorithms.minimax_solver import solve_zero_sum_matrix_game
from adversarial_queueing.algorithms.nnq import NNQTrainer
from adversarial_queueing.envs.service_rate_control import ServiceRateControlEnv


def service_rate_amq_q_diagnostic(
    env: ServiceRateControlEnv,
    trainer: LinearAMQTrainer,
    bvi_result: BVIResult,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare AMQ Q values with Bellman targets and bounded BVI reference Q."""

    rows: list[dict[str, Any]] = []
    for state in sorted(int(state) for state in bvi_result.values):
        for attacker_action in env.attacker_actions(state):
            for defender_action in env.defender_actions(state):
                q_amq = trainer.q_value(state, attacker_action, defender_action)
                amq_target = _amq_bellman_target(
                    env,
                    trainer,
                    state,
                    attacker_action,
                    defender_action,
                )
                q_bvi_reference = _bvi_q_value(
                    env,
                    bvi_result,
                    state,
                    attacker_action,
                    defender_action,
                )
                residual = amq_target - q_amq
                reference_gap = q_amq - q_bvi_reference
                rows.append(
                    {
                        "state": state,
                        "attacker_action": int(attacker_action),
                        "defender_action": int(defender_action),
                        "q_amq": float(q_amq),
                        "amq_bellman_target": float(amq_target),
                        "amq_bellman_residual": float(residual),
                        "amq_bellman_abs_residual": abs(float(residual)),
                        "q_bvi_reference": float(q_bvi_reference),
                        "q_reference_signed_gap": float(reference_gap),
                        "q_reference_abs_gap": abs(float(reference_gap)),
                    }
                )
    return rows, _q_diagnostic_summary(rows, method="amq")


def service_rate_nnq_q_diagnostic(
    env: ServiceRateControlEnv,
    trainer: NNQTrainer,
    bvi_result: BVIResult,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare NNQ Q values with Bellman targets and bounded BVI reference Q."""

    rows: list[dict[str, Any]] = []
    for state in sorted(int(state) for state in bvi_result.values):
        q_matrix = trainer.q_matrix(state)
        for attacker_action in env.attacker_actions(state):
            for defender_action in env.defender_actions(state):
                q_nnq = _nnq_q_value(trainer, state, attacker_action, defender_action)
                nnq_target = _nnq_bellman_target(
                    env,
                    trainer,
                    state,
                    attacker_action,
                    defender_action,
                )
                q_bvi_reference = _bvi_q_value(
                    env,
                    bvi_result,
                    state,
                    attacker_action,
                    defender_action,
                )
                residual = nnq_target - q_nnq
                reference_gap = q_nnq - q_bvi_reference
                rows.append(
                    {
                        "state": state,
                        "attacker_action": int(attacker_action),
                        "defender_action": int(defender_action),
                        "q_nnq": float(q_nnq),
                        "nnq_bellman_target": float(nnq_target),
                        "nnq_bellman_residual": float(residual),
                        "nnq_bellman_abs_residual": abs(float(residual)),
                        "q_bvi_reference": float(q_bvi_reference),
                        "q_reference_signed_gap": float(reference_gap),
                        "q_reference_abs_gap": abs(float(reference_gap)),
                        "q_action_spread": float(q_matrix.max() - q_matrix.min()),
                        "q_state_mean": float(q_matrix.mean()),
                    }
                )
    return rows, _q_diagnostic_summary(rows, method="nnq")


def _bvi_q_value(
    env: ServiceRateControlEnv,
    result: BVIResult,
    state: int,
    attacker_action: int,
    defender_action: int,
) -> float:
    max_queue_length = result.max_queue_length
    if max_queue_length is None:
        max_queue_length = max(int(state) for state in result.values)
    expected_next = 0.0
    for next_state, prob in env.transition_probabilities(
        state, attacker_action, defender_action
    ).items():
        expected_next += prob * result.values[min(int(next_state), max_queue_length)]
    return float(
        env.cost(state, attacker_action, defender_action)
        + env.discount * expected_next
    )


def _amq_bellman_target(
    env: ServiceRateControlEnv,
    trainer: LinearAMQTrainer,
    state: int,
    attacker_action: int,
    defender_action: int,
) -> float:
    expected_next = 0.0
    for next_state, prob in env.transition_probabilities(
        state, attacker_action, defender_action
    ).items():
        expected_next += prob * trainer.value(next_state)
    return float(env.cost(state, attacker_action, defender_action) + env.discount * expected_next)


def _nnq_bellman_target(
    env: ServiceRateControlEnv,
    trainer: NNQTrainer,
    state: int,
    attacker_action: int,
    defender_action: int,
) -> float:
    expected_next = 0.0
    for next_state, prob in env.transition_probabilities(
        state, attacker_action, defender_action
    ).items():
        expected_next += prob * _nnq_value(trainer, int(next_state))
    return float(env.cost(state, attacker_action, defender_action) + env.discount * expected_next)


def _nnq_value(trainer: NNQTrainer, state: int) -> float:
    return float(solve_zero_sum_matrix_game(trainer.q_matrix(state))["value"])


def _nnq_q_value(
    trainer: NNQTrainer,
    state: int,
    attacker_action: int,
    defender_action: int,
) -> float:
    attacker_index = trainer.attacker_actions.index(attacker_action)
    defender_index = trainer.defender_actions.index(defender_action)
    return float(trainer.q_matrix(state)[attacker_index, defender_index])


def _q_diagnostic_summary(
    rows: list[dict[str, Any]],
    method: str,
) -> dict[str, Any]:
    residual_key = f"{method}_bellman_abs_residual"
    q_key = f"q_{method}"
    residuals = np.array([row[residual_key] for row in rows], dtype=float)
    reference_gaps = np.array([row["q_reference_abs_gap"] for row in rows], dtype=float)
    method_q_abs = np.array([abs(row[q_key]) for row in rows], dtype=float)
    bvi_q_abs = np.array([abs(row["q_bvi_reference"]) for row in rows], dtype=float)
    summary = {
        "num_q_entries": len(rows),
        f"{method}_bellman_abs_residual_mean": float(residuals.mean()),
        f"{method}_bellman_abs_residual_max": float(residuals.max()),
        "q_reference_abs_gap_mean": float(reference_gaps.mean()),
        "q_reference_abs_gap_max": float(reference_gaps.max()),
        f"mean_abs_{method}_q": float(method_q_abs.mean()),
        "mean_abs_bvi_reference_q": float(bvi_q_abs.mean()),
        "by_state": _group_q_diagnostic_summary(rows, method),
    }
    if method == "nnq":
        action_spreads = np.array([row["q_action_spread"] for row in rows], dtype=float)
        state_means = np.array([abs(row["q_state_mean"]) for row in rows], dtype=float)
        summary.update(
            {
                "q_action_spread_mean": float(action_spreads.mean()),
                "q_action_spread_max": float(action_spreads.max()),
                "mean_abs_q_state_mean": float(state_means.mean()),
            }
        )
    return summary


def _group_q_diagnostic_summary(
    rows: list[dict[str, Any]],
    method: str,
) -> list[dict[str, Any]]:
    groups = sorted({int(row["state"]) for row in rows})
    residual_key = f"{method}_bellman_abs_residual"
    summaries = []
    for state in groups:
        group_rows = [row for row in rows if int(row["state"]) == state]
        residuals = np.array([row[residual_key] for row in group_rows], dtype=float)
        reference_gaps = np.array(
            [row["q_reference_abs_gap"] for row in group_rows],
            dtype=float,
        )
        summaries.append(
            {
                "state": state,
                "num_q_entries": len(group_rows),
                f"{method}_bellman_abs_residual_mean": float(residuals.mean()),
                "q_reference_abs_gap_mean": float(reference_gaps.mean()),
            }
        )
    return summaries
