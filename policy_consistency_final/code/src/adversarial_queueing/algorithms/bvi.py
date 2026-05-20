"""Bounded value iteration for finite truncated games."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Hashable, Sequence

import numpy as np

from adversarial_queueing.algorithms.minimax_solver import solve_zero_sum_matrix_game
from adversarial_queueing.envs.base import BaseAdversarialQueueEnv


@dataclass(frozen=True)
class BVIResult:
    values: dict[Hashable, float]
    iterations: int
    residual: float
    max_queue_length: int | None = None


def run_bounded_value_iteration(
    env: BaseAdversarialQueueEnv,
    max_queue_length: int,
    tolerance: float = 1e-8,
    max_iterations: int = 10_000,
    states: Sequence[Hashable] | None = None,
) -> BVIResult:
    """Run value iteration on a bounded finite queueing state space."""

    state_list = list(states) if states is not None else list(range(max_queue_length + 1))
    values = {state: 0.0 for state in state_list}

    for iteration in range(1, max_iterations + 1):
        new_values: dict[Hashable, float] = {}
        residual = 0.0

        for state in state_list:
            payoff = np.zeros(
                (len(env.attacker_actions(state)), len(env.defender_actions(state))),
                dtype=float,
            )
            for ai, attacker_action in enumerate(env.attacker_actions(state)):
                for bi, defender_action in enumerate(env.defender_actions(state)):
                    expected_next = 0.0
                    for next_state, prob in env.transition_probabilities(
                        state, attacker_action, defender_action
                    ).items():
                        bounded_next = bound_state_for_bvi(
                            env,
                            next_state,
                            max_queue_length,
                        )
                        expected_next += prob * values[bounded_next]
                    payoff[ai, bi] = (
                        env.cost(state, attacker_action, defender_action)
                        + env.discount * expected_next
                    )

            value = solve_zero_sum_matrix_game(payoff)["value"]
            new_values[state] = value
            residual = max(residual, abs(value - values[state]))

        values = new_values
        if residual < tolerance:
            return BVIResult(
                values=values,
                iterations=iteration,
                residual=residual,
                max_queue_length=max_queue_length,
            )

    return BVIResult(
        values=values,
        iterations=max_iterations,
        residual=residual,
        max_queue_length=max_queue_length,
    )


def bounded_queue_states(num_queues: int, max_queue_length: int) -> list[Hashable]:
    """Return bounded states for one or more queues."""

    if num_queues <= 0:
        raise ValueError("num_queues must be positive")
    if max_queue_length < 0:
        raise ValueError("max_queue_length must be nonnegative")
    if num_queues == 1:
        return list(range(max_queue_length + 1))
    return list(product(range(max_queue_length + 1), repeat=num_queues))


def bound_state_for_bvi(
    env: BaseAdversarialQueueEnv,
    state: Hashable,
    max_queue_length: int,
) -> Hashable:
    """Clip only queue coordinates when looking up bounded BVI values.

    Routing states are all queue lengths. Polling states append the server
    position after the queue lengths, so clipping every tuple component would
    corrupt valid positions when ``num_queues > max_queue_length + 1``.
    """

    if max_queue_length < 0:
        raise ValueError("max_queue_length must be nonnegative")
    if isinstance(state, tuple):
        config = getattr(env, "config", None)
        num_queues = getattr(config, "num_queues", None)
        if num_queues is not None and len(state) == int(num_queues) + 1:
            queues = tuple(min(int(value), max_queue_length) for value in state[:-1])
            return (*queues, int(state[-1]))
        return tuple(min(int(value), max_queue_length) for value in state)
    return min(int(state), max_queue_length)
