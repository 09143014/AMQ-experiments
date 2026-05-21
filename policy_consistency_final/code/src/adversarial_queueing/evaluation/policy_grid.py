"""Policy-grid inspection for service-rate-control benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adversarial_queueing.algorithms.amq import LinearAMQTrainer
from adversarial_queueing.algorithms.bvi import BVIResult
from adversarial_queueing.algorithms.minimax_solver import solve_zero_sum_matrix_game
from adversarial_queueing.algorithms.nnq import NNQTrainer
from adversarial_queueing.envs.service_rate_control import ServiceRateControlEnv


@dataclass(frozen=True)
class PolicyGridConfig:
    max_state: int = 10
    high_probability_threshold: float = 0.5


def amq_policy_grid(
    env: ServiceRateControlEnv,
    trainer: LinearAMQTrainer,
    config: PolicyGridConfig,
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | None]]:
    rows = []
    for state in range(config.max_state + 1):
        game = solve_zero_sum_matrix_game(trainer.q_matrix(state))
        rows.append(_policy_row("amq", state, game["defender_strategy"], game["attacker_strategy"]))
    return rows, _threshold_summary(rows, config)


def nnq_policy_grid(
    env: ServiceRateControlEnv,
    trainer: NNQTrainer,
    config: PolicyGridConfig,
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | None]]:
    rows = []
    for state in range(config.max_state + 1):
        game = solve_zero_sum_matrix_game(trainer.q_matrix(state))
        rows.append(_policy_row("nnq", state, game["defender_strategy"], game["attacker_strategy"]))
    return rows, _threshold_summary(rows, config)


def bvi_policy_grid(
    env: ServiceRateControlEnv,
    result: BVIResult,
    config: PolicyGridConfig,
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | None]]:
    rows = []
    max_value_state = max(result.values)
    for state in range(config.max_state + 1):
        clipped_state = min(state, max_value_state)
        attacker_actions = tuple(env.attacker_actions(clipped_state))
        defender_actions = tuple(env.defender_actions(clipped_state))
        payoff = np.zeros((len(attacker_actions), len(defender_actions)), dtype=float)
        for ai, attacker_action in enumerate(attacker_actions):
            for bi, defender_action in enumerate(defender_actions):
                expected_next = 0.0
                for next_state, prob in env.transition_probabilities(
                    clipped_state, attacker_action, defender_action
                ).items():
                    expected_next += prob * result.values[min(int(next_state), max_value_state)]
                payoff[ai, bi] = (
                    env.cost(clipped_state, attacker_action, defender_action)
                    + env.discount * expected_next
                )
        game = solve_zero_sum_matrix_game(payoff)
        rows.append(_policy_row("bvi", state, game["defender_strategy"], game["attacker_strategy"]))
    return rows, _threshold_summary(rows, config)


def _policy_row(
    method: str,
    state: int,
    defender_strategy: np.ndarray,
    attacker_strategy: np.ndarray | None = None,
) -> dict[str, float | int | str]:
    if defender_strategy.shape[0] != 2:
        raise ValueError("service-rate v2 policy grid expects two defender actions")
    row = {
        "method": method,
        "state": state,
        "p_no_defend": float(defender_strategy[0]),
        "p_defend": float(defender_strategy[1]),
    }
    if attacker_strategy is not None:
        if attacker_strategy.shape[0] != 2:
            raise ValueError("service-rate policy grid currently expects two attacker actions")
        row["p_no_attack"] = float(attacker_strategy[0])
        row["p_attack"] = float(attacker_strategy[1])
    return row


def _threshold_summary(
    rows: list[dict[str, float | int | str]],
    config: PolicyGridConfig,
) -> dict[str, float | int | None]:
    defend_threshold_state = None
    attack_threshold_state = None
    for row in rows:
        state = int(row["state"])
        if defend_threshold_state is None and row["p_defend"] >= config.high_probability_threshold:
            defend_threshold_state = state
        if (
            attack_threshold_state is None
            and "p_attack" in row
            and row["p_attack"] >= config.high_probability_threshold
        ):
            attack_threshold_state = state
    return {
        "policy_grid_max_state": config.max_state,
        "high_probability_threshold": config.high_probability_threshold,
        "first_state_p_defend_at_least_threshold": defend_threshold_state,
        "first_state_p_attack_at_least_threshold": attack_threshold_state,
    }
