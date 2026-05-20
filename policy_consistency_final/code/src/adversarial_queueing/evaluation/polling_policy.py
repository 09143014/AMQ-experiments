"""Policy inspection helpers for the polling benchmark."""

from __future__ import annotations

from typing import Any, Hashable

import numpy as np

from adversarial_queueing.algorithms.amq import LinearAMQTrainer
from adversarial_queueing.algorithms.bvi import BVIResult
from adversarial_queueing.algorithms.minimax_solver import solve_zero_sum_matrix_game
from adversarial_queueing.algorithms.nnq import NNQTrainer
from adversarial_queueing.envs.polling import PollingEnv, State


def bvi_polling_policy_inspection(
    env: PollingEnv,
    result: BVIResult,
    probability_threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for state in sorted(result.values, key=_state_sort_key):
        if not isinstance(state, tuple):
            raise ValueError("polling policy inspection requires tuple states")
        game = _bvi_game_at_state(env, result, state)
        rows.append(_policy_row("bvi", env, state, game))
    return rows, _summary(rows, probability_threshold)


def amq_polling_policy_inspection(
    env: PollingEnv,
    trainer: LinearAMQTrainer,
    max_queue_length: int,
    probability_threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for state in sorted(
        _polling_states(env.config.num_queues, max_queue_length),
        key=_state_sort_key,
    ):
        game = solve_zero_sum_matrix_game(trainer.q_matrix(state))
        rows.append(_policy_row("amq", env, state, game))
    return rows, _summary(rows, probability_threshold)


def nnq_polling_policy_inspection(
    env: PollingEnv,
    trainer: NNQTrainer,
    max_queue_length: int,
    probability_threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for state in sorted(
        _polling_states(env.config.num_queues, max_queue_length),
        key=_state_sort_key,
    ):
        game = solve_zero_sum_matrix_game(trainer.q_matrix(state))
        rows.append(_policy_row("nnq", env, state, game))
    return rows, _summary(rows, probability_threshold)


def compare_amq_bvi_polling_policies(
    env: PollingEnv,
    trainer: LinearAMQTrainer,
    bvi_result: BVIResult,
    probability_threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for state in sorted(bvi_result.values, key=_state_sort_key):
        if not isinstance(state, tuple):
            raise ValueError("polling policy comparison requires tuple states")
        amq_game = solve_zero_sum_matrix_game(trainer.q_matrix(state))
        bvi_game = _bvi_game_at_state(env, bvi_result, state)
        p_defend_amq = float(amq_game["defender_strategy"][1])
        p_defend_bvi = float(bvi_game["defender_strategy"][1])
        signed_gap = p_defend_amq - p_defend_bvi
        queues = tuple(int(value) for value in state[:-1])
        rows.append(
            {
                "state": list(state),
                "queues": list(queues),
                "position": int(state[-1]),
                "total_queue": sum(queues),
                "queue_gap": max(queues) - min(queues),
                "p_defend_amq": p_defend_amq,
                "p_defend_bvi_reference": p_defend_bvi,
                "p_defend_signed_gap": signed_gap,
                "p_defend_abs_gap": abs(signed_gap),
                "amq_over_defends": bool(signed_gap >= probability_threshold),
                "amq_under_defends": bool(signed_gap <= -probability_threshold),
            }
        )
    return rows, _comparison_summary(rows, probability_threshold)


def polling_amq_q_diagnostic(
    env: PollingEnv,
    trainer: LinearAMQTrainer,
    bvi_result: BVIResult,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for state in sorted(bvi_result.values, key=_state_sort_key):
        if not isinstance(state, tuple):
            raise ValueError("polling Q diagnostic requires tuple states")
        queues = tuple(int(value) for value in state[:-1])
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
                        "queues": list(queues),
                        "position": int(state[-1]),
                        "total_queue": sum(queues),
                        "queue_gap": max(queues) - min(queues),
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


def polling_nnq_q_diagnostic(
    env: PollingEnv,
    trainer: NNQTrainer,
    bvi_result: BVIResult,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for state in sorted(bvi_result.values, key=_state_sort_key):
        if not isinstance(state, tuple):
            raise ValueError("polling Q diagnostic requires tuple states")
        queues = tuple(int(value) for value in state[:-1])
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
                nnq_residual = nnq_target - q_nnq
                reference_gap = q_nnq - q_bvi_reference
                rows.append(
                    {
                        "state": list(state),
                        "queues": list(queues),
                        "position": int(state[-1]),
                        "total_queue": sum(queues),
                        "queue_gap": max(queues) - min(queues),
                        "attacker_action": int(attacker_action),
                        "defender_action": int(defender_action),
                        "q_nnq": float(q_nnq),
                        "nnq_bellman_target": float(nnq_target),
                        "nnq_bellman_residual": float(nnq_residual),
                        "nnq_bellman_abs_residual": abs(float(nnq_residual)),
                        "q_bvi_reference": float(q_bvi_reference),
                        "q_reference_signed_gap": float(reference_gap),
                        "q_reference_abs_gap": abs(float(reference_gap)),
                        "q_action_spread": float(q_matrix.max() - q_matrix.min()),
                        "q_state_mean": float(q_matrix.mean()),
                    }
                )
    return rows, _q_diagnostic_summary(rows, method="nnq")


def _bvi_game_at_state(env: PollingEnv, result: BVIResult, state: State) -> dict[str, Any]:
    max_queue_length = result.max_queue_length
    if max_queue_length is None:
        max_queue_length = max(
            max(value[:-1]) for value in result.values if isinstance(value, tuple)
        )

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
                    _bounded_polling_state(next_state, max_queue_length)
                ]
            payoff[ai, bi] = (
                env.cost(state, attacker_action, defender_action)
                + env.discount * expected_next
            )
    return solve_zero_sum_matrix_game(payoff)


def _bvi_q_value(
    env: PollingEnv,
    result: BVIResult,
    state: State,
    attacker_action: int,
    defender_action: int,
) -> float:
    max_queue_length = result.max_queue_length
    if max_queue_length is None:
        max_queue_length = max(
            max(value[:-1]) for value in result.values if isinstance(value, tuple)
        )

    expected_next = 0.0
    for next_state, prob in env.transition_probabilities(
        state, attacker_action, defender_action
    ).items():
        expected_next += prob * result.values[
            _bounded_polling_state(next_state, max_queue_length)
        ]
    return float(
        env.cost(state, attacker_action, defender_action)
        + env.discount * expected_next
    )


def _amq_bellman_target(
    env: PollingEnv,
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
    return float(
        env.cost(state, attacker_action, defender_action)
        + env.discount * expected_next
    )


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
    env: PollingEnv,
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
    return float(
        env.cost(state, attacker_action, defender_action)
        + env.discount * expected_next
    )


def _nnq_value(trainer: NNQTrainer, state: Hashable) -> float:
    return float(solve_zero_sum_matrix_game(trainer.q_matrix(state))["value"])


def _policy_row(
    method: str,
    env: PollingEnv,
    state: State,
    game: dict[str, Any],
) -> dict[str, Any]:
    defender_strategy = game["defender_strategy"]
    attacker_strategy = game["attacker_strategy"]
    if defender_strategy.shape[0] != 2:
        raise ValueError("polling policy inspection expects two defender actions")
    if attacker_strategy.shape[0] != 2:
        raise ValueError("polling policy inspection expects two attacker actions")
    queues = tuple(int(value) for value in state[:-1])
    position = int(state[-1])
    gap = max(queues) - min(queues)
    return {
        "method": method,
        "state": list(state),
        "queues": list(queues),
        "position": position,
        "total_queue": sum(queues),
        "queue_gap": gap,
        "nominal_targets": list(
            env.polling_targets(state, attacker_action=0, defender_action=0)
        ),
        "attacked_targets": list(
            env.polling_targets(state, attacker_action=1, defender_action=0)
        ),
        "p_no_defend": float(defender_strategy[0]),
        "p_defend": float(defender_strategy[1]),
        "p_no_attack": float(attacker_strategy[0]),
        "p_attack": float(attacker_strategy[1]),
        "value": float(game["value"]),
    }


def _comparison_summary(
    rows: list[dict[str, Any]],
    probability_threshold: float,
) -> dict[str, Any]:
    abs_gaps = np.array([row["p_defend_abs_gap"] for row in rows], dtype=float)
    signed_gaps = np.array([row["p_defend_signed_gap"] for row in rows], dtype=float)
    over_defend_rows = [row for row in rows if row["amq_over_defends"]]
    under_defend_rows = [row for row in rows if row["amq_under_defends"]]
    return {
        "num_compared_states": len(rows),
        "p_defend_abs_gap_mean": float(abs_gaps.mean()),
        "p_defend_abs_gap_max": float(abs_gaps.max()),
        "p_defend_signed_gap_mean": float(signed_gaps.mean()),
        "gap_probability_threshold": probability_threshold,
        "num_states_amq_over_defends": len(over_defend_rows),
        "num_states_amq_under_defends": len(under_defend_rows),
        "first_state_amq_over_defends": (
            None if not over_defend_rows else over_defend_rows[0]["state"]
        ),
        "first_state_amq_under_defends": (
            None if not under_defend_rows else under_defend_rows[0]["state"]
        ),
        "by_queue_gap": _group_gap_summary(rows, "queue_gap"),
        "by_total_queue": _group_gap_summary(rows, "total_queue"),
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
        "by_queue_gap": _group_q_diagnostic_summary(rows, "queue_gap", method),
        "by_total_queue": _group_q_diagnostic_summary(rows, "total_queue", method),
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


def _summary(
    rows: list[dict[str, Any]],
    probability_threshold: float,
) -> dict[str, Any]:
    defend_probs = np.array([row["p_defend"] for row in rows], dtype=float)
    defend_rows = [row for row in rows if row["p_defend"] >= probability_threshold]
    gap_rows = [row for row in rows if int(row["queue_gap"]) > 0]
    gap_defend_rows = [
        row
        for row in gap_rows
        if row["p_defend"] >= probability_threshold
    ]
    return {
        "num_policy_states": len(rows),
        "defend_probability_mean": float(defend_probs.mean()),
        "defend_probability_max": float(defend_probs.max()),
        "defend_probability_threshold": probability_threshold,
        "num_states_p_defend_at_least_threshold": len(defend_rows),
        "first_state_p_defend_at_least_threshold": (
            None if not defend_rows else defend_rows[0]["state"]
        ),
        "num_gap_states": len(gap_rows),
        "num_gap_states_p_defend_at_least_threshold": len(gap_defend_rows),
        "by_queue_gap": _group_summary(rows, "queue_gap"),
        "by_total_queue": _group_summary(rows, "total_queue"),
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


def _group_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups = sorted({int(row[key]) for row in rows})
    summaries = []
    for group in groups:
        group_rows = [row for row in rows if int(row[key]) == group]
        defend_probs = np.array([row["p_defend"] for row in group_rows], dtype=float)
        summaries.append(
            {
                key: group,
                "num_states": len(group_rows),
                "p_defend_mean": float(defend_probs.mean()),
                "p_defend_max": float(defend_probs.max()),
            }
        )
    return summaries


def _polling_states(num_queues: int, max_queue_length: int) -> list[State]:
    queue_states: list[tuple[int, ...]] = [()]
    for _ in range(num_queues):
        queue_states = [
            (*prefix, value)
            for prefix in queue_states
            for value in range(max_queue_length + 1)
        ]
    return [
        (*queues, position)
        for queues in queue_states
        for position in range(num_queues)
    ]


def _bounded_polling_state(state: Hashable, max_queue_length: int) -> State:
    if not isinstance(state, tuple):
        raise ValueError("polling bounded state requires tuple state")
    queues = tuple(min(int(value), max_queue_length) for value in state[:-1])
    return (*queues, int(state[-1]))


def _state_sort_key(state: Hashable) -> tuple[int, tuple[int, ...]]:
    if not isinstance(state, tuple):
        return (int(state), (int(state),))
    queues = tuple(int(value) for value in state[:-1])
    return (sum(queues), (*queues, int(state[-1])))
