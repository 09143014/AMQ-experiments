"""Policy inspection helpers for the routing benchmark."""

from __future__ import annotations

from typing import Any, Hashable

import numpy as np

from adversarial_queueing.algorithms.amq import LinearAMQTrainer
from adversarial_queueing.algorithms.bvi import BVIResult
from adversarial_queueing.algorithms.minimax_solver import solve_zero_sum_matrix_game
from adversarial_queueing.algorithms.nnq import NNQTrainer
from adversarial_queueing.envs.routing import RoutingEnv, State


def bvi_routing_policy_inspection(
    env: RoutingEnv,
    result: BVIResult,
    probability_threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Inspect BVI defender strategies across the bounded routing state space."""

    rows: list[dict[str, Any]] = []
    for state in sorted(result.values, key=_state_sort_key):
        if not isinstance(state, tuple):
            raise ValueError("routing policy inspection requires tuple states")
        game = _bvi_game_at_state(env, result, state)
        rows.append(_policy_row("bvi", env, state, game))
    return rows, _summary(rows, probability_threshold)


def amq_routing_policy_inspection(
    env: RoutingEnv,
    trainer: LinearAMQTrainer,
    max_queue_length: int,
    probability_threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Inspect AMQ defender strategies across a bounded routing state grid."""

    rows: list[dict[str, Any]] = []
    states = sorted(
        _routing_states(env.config.num_queues, max_queue_length),
        key=_state_sort_key,
    )
    for state in states:
        game = solve_zero_sum_matrix_game(trainer.q_matrix(state))
        rows.append(_policy_row("amq", env, state, game))
    return rows, _summary(rows, probability_threshold)


def nnq_routing_policy_inspection(
    env: RoutingEnv,
    trainer: NNQTrainer,
    max_queue_length: int,
    probability_threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Inspect NNQ defender strategies across a bounded routing state grid."""

    rows: list[dict[str, Any]] = []
    states = sorted(
        _routing_states(env.config.num_queues, max_queue_length),
        key=_state_sort_key,
    )
    for state in states:
        game = solve_zero_sum_matrix_game(trainer.q_matrix(state))
        rows.append(_policy_row("nnq", env, state, game))
    return rows, _summary(rows, probability_threshold)


def compare_amq_bvi_routing_policies(
    env: RoutingEnv,
    trainer: LinearAMQTrainer,
    bvi_result: BVIResult,
    probability_threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare AMQ defender policy against a bounded BVI reference policy."""

    rows: list[dict[str, Any]] = []
    for state in sorted(bvi_result.values, key=_state_sort_key):
        if not isinstance(state, tuple):
            raise ValueError("routing policy comparison requires tuple states")
        amq_game = solve_zero_sum_matrix_game(trainer.q_matrix(state))
        bvi_game = _bvi_game_at_state(env, bvi_result, state)
        p_defend_amq = float(amq_game["defender_strategy"][1])
        p_defend_bvi = float(bvi_game["defender_strategy"][1])
        signed_gap = p_defend_amq - p_defend_bvi
        rows.append(
            {
                "state": list(state),
                "total_queue": sum(state),
                "imbalance": max(state) - min(state),
                "p_defend_amq": p_defend_amq,
                "p_defend_bvi_reference": p_defend_bvi,
                "p_defend_signed_gap": signed_gap,
                "p_defend_abs_gap": abs(signed_gap),
                "amq_over_defends": bool(signed_gap >= probability_threshold),
                "amq_under_defends": bool(signed_gap <= -probability_threshold),
            }
        )
    return rows, _comparison_summary(rows, probability_threshold)


def compare_nnq_bvi_routing_policies(
    env: RoutingEnv,
    trainer: NNQTrainer,
    bvi_result: BVIResult,
    probability_threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare NNQ defender policy against a bounded BVI reference policy."""

    rows: list[dict[str, Any]] = []
    for state in sorted(bvi_result.values, key=_state_sort_key):
        if not isinstance(state, tuple):
            raise ValueError("routing policy comparison requires tuple states")
        nnq_game = solve_zero_sum_matrix_game(trainer.q_matrix(state))
        bvi_game = _bvi_game_at_state(env, bvi_result, state)
        p_defend_nnq = float(nnq_game["defender_strategy"][1])
        p_defend_bvi = float(bvi_game["defender_strategy"][1])
        signed_gap = p_defend_nnq - p_defend_bvi
        rows.append(
            {
                "state": list(state),
                "total_queue": sum(state),
                "imbalance": max(state) - min(state),
                "p_defend_nnq": p_defend_nnq,
                "p_defend_bvi_reference": p_defend_bvi,
                "p_defend_signed_gap": signed_gap,
                "p_defend_abs_gap": abs(signed_gap),
                "nnq_over_defends": bool(signed_gap >= probability_threshold),
                "nnq_under_defends": bool(signed_gap <= -probability_threshold),
            }
        )
    return rows, _comparison_summary(rows, probability_threshold, method="nnq")


def routing_amq_q_diagnostic(
    env: RoutingEnv,
    trainer: LinearAMQTrainer,
    bvi_result: BVIResult,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare AMQ Q values with AMQ Bellman targets and bounded BVI reference Q."""

    rows: list[dict[str, Any]] = []
    for state in sorted(bvi_result.values, key=_state_sort_key):
        if not isinstance(state, tuple):
            raise ValueError("routing Q diagnostic requires tuple states")
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
                amq_residual = amq_target - q_amq
                reference_gap = q_amq - q_bvi_reference
                rows.append(
                    {
                        "state": list(state),
                        "total_queue": sum(state),
                        "imbalance": max(state) - min(state),
                        "attacker_action": int(attacker_action),
                        "defender_action": int(defender_action),
                        "q_amq": float(q_amq),
                        "amq_bellman_target": float(amq_target),
                        "amq_bellman_residual": float(amq_residual),
                        "amq_bellman_abs_residual": abs(float(amq_residual)),
                        "q_bvi_reference": float(q_bvi_reference),
                        "q_reference_signed_gap": float(reference_gap),
                        "q_reference_abs_gap": abs(float(reference_gap)),
                    }
                )
    return rows, _q_diagnostic_summary(rows)


def routing_nnq_q_diagnostic(
    env: RoutingEnv,
    trainer: NNQTrainer,
    bvi_result: BVIResult,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare NNQ Q values with NNQ Bellman targets and bounded BVI reference Q."""

    rows: list[dict[str, Any]] = []
    for state in sorted(bvi_result.values, key=_state_sort_key):
        if not isinstance(state, tuple):
            raise ValueError("routing Q diagnostic requires tuple states")
        nnq_matrix = trainer.q_matrix(state)
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
                nnq_residual = nnq_target - q_nnq
                reference_gap = q_nnq - q_bvi_reference
                rows.append(
                    {
                        "state": list(state),
                        "total_queue": sum(state),
                        "imbalance": max(state) - min(state),
                        "attacker_action": int(attacker_action),
                        "defender_action": int(defender_action),
                        "q_nnq": float(q_nnq),
                        "nnq_bellman_target": float(nnq_target),
                        "nnq_bellman_residual": float(nnq_residual),
                        "nnq_bellman_abs_residual": abs(float(nnq_residual)),
                        "q_bvi_reference": float(q_bvi_reference),
                        "q_reference_signed_gap": float(reference_gap),
                        "q_reference_abs_gap": abs(float(reference_gap)),
                        "q_action_spread": float(nnq_matrix.max() - nnq_matrix.min()),
                        "q_state_mean": float(nnq_matrix.mean()),
                    }
                )
    return rows, _q_diagnostic_summary(rows, method="nnq")


def _bvi_game_at_state(env: RoutingEnv, result: BVIResult, state: State) -> dict[str, Any]:
    max_queue_length = result.max_queue_length
    if max_queue_length is None:
        max_queue_length = max(max(value) for value in result.values if isinstance(value, tuple))

    attacker_actions = tuple(env.attacker_actions(state))
    defender_actions = tuple(env.defender_actions(state))
    payoff = np.zeros((len(attacker_actions), len(defender_actions)), dtype=float)
    for ai, attacker_action in enumerate(attacker_actions):
        for bi, defender_action in enumerate(defender_actions):
            expected_next = 0.0
            for next_state, prob in env.transition_probabilities(
                state, attacker_action, defender_action
            ).items():
                expected_next += prob * result.values[
                    _bounded_state(next_state, max_queue_length)
                ]
            payoff[ai, bi] = (
                env.cost(state, attacker_action, defender_action)
                + env.discount * expected_next
            )
    return solve_zero_sum_matrix_game(payoff)


def _bvi_q_value(
    env: RoutingEnv,
    result: BVIResult,
    state: State,
    attacker_action: int,
    defender_action: int,
) -> float:
    max_queue_length = result.max_queue_length
    if max_queue_length is None:
        max_queue_length = max(max(value) for value in result.values if isinstance(value, tuple))

    expected_next = 0.0
    for next_state, prob in env.transition_probabilities(
        state, attacker_action, defender_action
    ).items():
        expected_next += prob * result.values[_bounded_state(next_state, max_queue_length)]
    return float(
        env.cost(state, attacker_action, defender_action)
        + env.discount * expected_next
    )


def _amq_bellman_target(
    env: RoutingEnv,
    trainer: LinearAMQTrainer,
    state: State,
    attacker_action: int,
    defender_action: int,
) -> float:
    expected_next = 0.0
    for next_state, prob in env.transition_probabilities(
        state, attacker_action, defender_action
    ).items():
        expected_next += prob * trainer.value(next_state)
    return float(env.cost(state, attacker_action, defender_action) + env.discount * expected_next)


def _nnq_q_value(
    trainer: NNQTrainer,
    state: State,
    attacker_action: int,
    defender_action: int,
) -> float:
    attacker_index = trainer.attacker_actions.index(attacker_action)
    defender_index = trainer.defender_actions.index(defender_action)
    return float(trainer.q_matrix(state)[attacker_index, defender_index])


def _nnq_bellman_target(
    env: RoutingEnv,
    trainer: NNQTrainer,
    state: State,
    attacker_action: int,
    defender_action: int,
) -> float:
    expected_next = 0.0
    for next_state, prob in env.transition_probabilities(
        state, attacker_action, defender_action
    ).items():
        expected_next += prob * _nnq_value(trainer, next_state)
    return float(env.cost(state, attacker_action, defender_action) + env.discount * expected_next)


def _nnq_value(trainer: NNQTrainer, state: Hashable) -> float:
    return float(solve_zero_sum_matrix_game(trainer.q_matrix(state))["value"])


def _policy_row(
    method: str,
    env: RoutingEnv,
    state: State,
    game: dict[str, Any],
) -> dict[str, Any]:
    defender_strategy = game["defender_strategy"]
    attacker_strategy = game["attacker_strategy"]
    if defender_strategy.shape[0] != 2:
        raise ValueError("routing policy inspection expects two defender actions")
    if attacker_strategy.shape[0] != 2:
        raise ValueError("routing policy inspection expects two attacker actions")
    return {
        "method": method,
        "state": list(state),
        "total_queue": sum(state),
        "imbalance": max(state) - min(state),
        "nominal_targets": list(
            env.routed_arrival_targets(state, attacker_action=0, defender_action=0)
        ),
        "attacked_targets": list(
            env.routed_arrival_targets(state, attacker_action=1, defender_action=0)
        ),
        "p_no_defend": float(defender_strategy[0]),
        "p_defend": float(defender_strategy[1]),
        "p_no_attack": float(attacker_strategy[0]),
        "p_attack": float(attacker_strategy[1]),
        "value": float(game["value"]),
    }


def _q_diagnostic_summary(
    rows: list[dict[str, Any]],
    method: str = "amq",
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
        "by_total_queue": _group_q_diagnostic_summary(rows, "total_queue", method),
        "by_imbalance": _group_q_diagnostic_summary(rows, "imbalance", method),
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


def _comparison_summary(
    rows: list[dict[str, Any]],
    probability_threshold: float,
    method: str = "amq",
) -> dict[str, Any]:
    abs_gaps = np.array([row["p_defend_abs_gap"] for row in rows], dtype=float)
    signed_gaps = np.array([row["p_defend_signed_gap"] for row in rows], dtype=float)
    over_key = f"{method}_over_defends"
    under_key = f"{method}_under_defends"
    over_defend_rows = [row for row in rows if row[over_key]]
    under_defend_rows = [row for row in rows if row[under_key]]
    return {
        "num_compared_states": len(rows),
        "p_defend_abs_gap_mean": float(abs_gaps.mean()),
        "p_defend_abs_gap_max": float(abs_gaps.max()),
        "p_defend_signed_gap_mean": float(signed_gaps.mean()),
        "gap_probability_threshold": probability_threshold,
        f"num_states_{method}_over_defends": len(over_defend_rows),
        f"num_states_{method}_under_defends": len(under_defend_rows),
        f"first_state_{method}_over_defends": (
            None if not over_defend_rows else over_defend_rows[0]["state"]
        ),
        f"first_state_{method}_under_defends": (
            None if not under_defend_rows else under_defend_rows[0]["state"]
        ),
        "by_total_queue": _group_gap_summary(rows, "total_queue"),
        "by_imbalance": _group_gap_summary(rows, "imbalance"),
    }


def _summary(
    rows: list[dict[str, Any]],
    probability_threshold: float,
) -> dict[str, Any]:
    defend_probs = np.array([row["p_defend"] for row in rows], dtype=float)
    defend_rows = [row for row in rows if row["p_defend"] >= probability_threshold]
    return {
        "num_policy_states": len(rows),
        "defend_probability_mean": float(defend_probs.mean()),
        "defend_probability_max": float(defend_probs.max()),
        "defend_probability_threshold": probability_threshold,
        "num_states_p_defend_at_least_threshold": len(defend_rows),
        "first_state_p_defend_at_least_threshold": (
            None if not defend_rows else defend_rows[0]["state"]
        ),
    }


def _group_gap_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups = sorted({int(row[key]) for row in rows})
    summaries = []
    for group in groups:
        group_rows = [row for row in rows if int(row[key]) == group]
        abs_gaps = np.array([row["p_defend_abs_gap"] for row in group_rows], dtype=float)
        signed_gaps = np.array(
            [row["p_defend_signed_gap"] for row in group_rows],
            dtype=float,
        )
        summaries.append(
            {
                key: group,
                "num_states": len(group_rows),
                "p_defend_abs_gap_mean": float(abs_gaps.mean()),
                "p_defend_signed_gap_mean": float(signed_gaps.mean()),
            }
        )
    return summaries


def _group_q_diagnostic_summary(
    rows: list[dict[str, Any]],
    key: str,
    method: str,
) -> list[dict[str, Any]]:
    groups = sorted({int(row[key]) for row in rows})
    summaries = []
    residual_key = f"{method}_bellman_abs_residual"
    for group in groups:
        group_rows = [row for row in rows if int(row[key]) == group]
        residuals = np.array(
            [row[residual_key] for row in group_rows],
            dtype=float,
        )
        reference_gaps = np.array(
            [row["q_reference_abs_gap"] for row in group_rows],
            dtype=float,
        )
        summaries.append(
            {
                key: group,
                "num_q_entries": len(group_rows),
                f"{method}_bellman_abs_residual_mean": float(residuals.mean()),
                "q_reference_abs_gap_mean": float(reference_gaps.mean()),
            }
        )
    return summaries


def _bounded_state(state: Hashable, max_queue_length: int) -> Hashable:
    if isinstance(state, tuple):
        return tuple(min(int(value), max_queue_length) for value in state)
    return min(int(state), max_queue_length)


def _routing_states(num_queues: int, max_queue_length: int) -> list[State]:
    if num_queues <= 0:
        raise ValueError("num_queues must be positive")
    states: list[State] = [()]
    for _ in range(num_queues):
        states = [
            (*prefix, value)
            for prefix in states
            for value in range(max_queue_length + 1)
        ]
    return states


def _state_sort_key(state: Hashable) -> tuple[int, tuple[int, ...]]:
    if not isinstance(state, tuple):
        return (int(state), (int(state),))
    return (sum(state), tuple(int(value) for value in state))
