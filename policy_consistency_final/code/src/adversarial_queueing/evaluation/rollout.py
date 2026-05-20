"""Rollout evaluation for benchmark policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Hashable

import numpy as np

from adversarial_queueing.algorithms.amq import LinearAMQTrainer
from adversarial_queueing.algorithms.bvi import BVIResult, bound_state_for_bvi
from adversarial_queueing.algorithms.minimax_solver import solve_zero_sum_matrix_game
from adversarial_queueing.algorithms.nnq import NNQTrainer
from adversarial_queueing.envs.base import BaseAdversarialQueueEnv

AttackerPolicy = Callable[[Hashable, np.random.Generator, BaseAdversarialQueueEnv], int]
DefenderPolicy = Callable[[Hashable, np.random.Generator, BaseAdversarialQueueEnv], int]


@dataclass(frozen=True)
class EvaluationConfig:
    num_episodes: int = 5
    horizon: int = 25
    seed: int = 0
    tail_threshold: int = 8
    boundary_state: int | None = None


@dataclass(frozen=True)
class RolloutResult:
    rows: list[dict[str, Any]]
    summary: dict[str, float | int]


def evaluate_policy(
    env: BaseAdversarialQueueEnv,
    defender_policy: DefenderPolicy,
    attacker_policy: AttackerPolicy,
    config: EvaluationConfig,
) -> RolloutResult:
    """Evaluate policies using the same environment step convention as training."""

    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, Any]] = []

    for episode in range(config.num_episodes):
        state = env.reset(seed=config.seed + episode)
        discounted_cost = 0.0
        total_cost = 0.0
        tail_count = 0
        boundary_hits = 0

        for step in range(config.horizon):
            attacker_action = int(attacker_policy(state, rng, env))
            defender_action = int(defender_policy(state, rng, env))
            next_state, cost, _info = env.step(attacker_action, defender_action)
            total_cost += float(cost)
            discounted_cost += (env.discount**step) * float(cost)
            next_load = _state_load(next_state, env)
            tail_count += int(next_load >= config.tail_threshold)
            if config.boundary_state is not None:
                boundary_hits += int(next_load >= config.boundary_state)
            state = next_state

        rows.append(
            {
                "episode": episode,
                "seed": config.seed + episode,
                "total_cost": total_cost,
                "average_cost": total_cost / config.horizon,
                "discounted_cost": discounted_cost,
                "final_state": state,
                "final_load": _state_load(state, env),
                "tail_fraction": tail_count / config.horizon,
                "boundary_hit_fraction": boundary_hits / config.horizon,
            }
        )

    summary = _summarize_rows(rows)
    summary.update(
        {
            "num_episodes": config.num_episodes,
            "horizon": config.horizon,
            "seed": config.seed,
            "tail_threshold": config.tail_threshold,
        }
    )
    return RolloutResult(rows=rows, summary=summary)


def rollout_state_visitation(
    env: BaseAdversarialQueueEnv,
    defender_policy: DefenderPolicy,
    attacker_policy: AttackerPolicy,
    config: EvaluationConfig,
) -> list[dict[str, Any]]:
    """Return per-state visit counts under a policy pair."""

    rng = np.random.default_rng(config.seed)
    counts: dict[Hashable, int] = {}
    total_visits = 0

    for episode in range(config.num_episodes):
        state = env.reset(seed=config.seed + episode)
        for _step in range(config.horizon):
            counts[state] = counts.get(state, 0) + 1
            total_visits += 1
            attacker_action = int(attacker_policy(state, rng, env))
            defender_action = int(defender_policy(state, rng, env))
            state, _cost, _info = env.step(attacker_action, defender_action)

    rows = []
    for state, count in sorted(counts.items(), key=lambda item: _state_sort_key(item[0])):
        rows.append(
            {
                "state": _json_state(state),
                "visit_count": int(count),
                "visit_fraction": float(count / total_visits)
                if total_visits
                else 0.0,
            }
        )
    return rows


def random_attacker_policy(
    state: Hashable, rng: np.random.Generator, env: BaseAdversarialQueueEnv
) -> int:
    return int(rng.choice(env.attacker_actions(state)))


def always_attacker_policy(
    state: Hashable, rng: np.random.Generator, env: BaseAdversarialQueueEnv
) -> int:
    return int(max(env.attacker_actions(state)))


def make_amq_defender_policy(trainer: LinearAMQTrainer) -> DefenderPolicy:
    def policy(
        state: Hashable, rng: np.random.Generator, env: BaseAdversarialQueueEnv
    ) -> int:
        game = solve_zero_sum_matrix_game(trainer.q_matrix(state))
        defender_actions = tuple(env.defender_actions(state))
        return int(rng.choice(defender_actions, p=game["defender_strategy"]))

    return policy


def make_amq_attacker_policy(trainer: LinearAMQTrainer) -> AttackerPolicy:
    def policy(
        state: Hashable, rng: np.random.Generator, env: BaseAdversarialQueueEnv
    ) -> int:
        game = solve_zero_sum_matrix_game(trainer.q_matrix(state))
        attacker_actions = tuple(env.attacker_actions(state))
        return int(rng.choice(attacker_actions, p=game["attacker_strategy"]))

    return policy


def make_nnq_defender_policy(trainer: NNQTrainer) -> DefenderPolicy:
    def policy(
        state: Hashable, rng: np.random.Generator, env: BaseAdversarialQueueEnv
    ) -> int:
        game = solve_zero_sum_matrix_game(trainer.q_matrix(state))
        defender_actions = tuple(env.defender_actions(state))
        return int(rng.choice(defender_actions, p=game["defender_strategy"]))

    return policy


def make_nnq_attacker_policy(trainer: NNQTrainer) -> AttackerPolicy:
    def policy(
        state: Hashable, rng: np.random.Generator, env: BaseAdversarialQueueEnv
    ) -> int:
        game = solve_zero_sum_matrix_game(trainer.q_matrix(state))
        attacker_actions = tuple(env.attacker_actions(state))
        return int(rng.choice(attacker_actions, p=game["attacker_strategy"]))

    return policy


def make_bvi_defender_policy(result: BVIResult) -> DefenderPolicy:
    max_queue_length = _bvi_bound(result)

    def policy(
        state: Hashable, rng: np.random.Generator, env: BaseAdversarialQueueEnv
    ) -> int:
        clipped_state = bound_state_for_bvi(env, state, max_queue_length)
        attacker_actions = tuple(env.attacker_actions(clipped_state))
        defender_actions = tuple(env.defender_actions(clipped_state))
        payoff = np.zeros((len(attacker_actions), len(defender_actions)), dtype=float)
        for ai, attacker_action in enumerate(attacker_actions):
            for bi, defender_action in enumerate(defender_actions):
                expected_next = 0.0
                for next_state, prob in env.transition_probabilities(
                    clipped_state, attacker_action, defender_action
                ).items():
                    expected_next += prob * result.values[
                        bound_state_for_bvi(env, next_state, max_queue_length)
                    ]
                payoff[ai, bi] = (
                    env.cost(clipped_state, attacker_action, defender_action)
                    + env.discount * expected_next
                )
        game = solve_zero_sum_matrix_game(payoff)
        return int(rng.choice(defender_actions, p=game["defender_strategy"]))

    return policy


def make_bvi_attacker_policy(result: BVIResult) -> AttackerPolicy:
    max_queue_length = _bvi_bound(result)

    def policy(
        state: Hashable, rng: np.random.Generator, env: BaseAdversarialQueueEnv
    ) -> int:
        clipped_state = bound_state_for_bvi(env, state, max_queue_length)
        game = solve_zero_sum_matrix_game(
            _bvi_payoff_matrix(env, result, clipped_state, max_queue_length)
        )
        attacker_actions = tuple(env.attacker_actions(clipped_state))
        return int(rng.choice(attacker_actions, p=game["attacker_strategy"]))

    return policy


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    average_costs = np.array([row["average_cost"] for row in rows], dtype=float)
    discounted_costs = np.array([row["discounted_cost"] for row in rows], dtype=float)
    final_loads = np.array([row["final_load"] for row in rows], dtype=float)
    tail_fractions = np.array([row["tail_fraction"] for row in rows], dtype=float)
    boundary_fractions = np.array([row["boundary_hit_fraction"] for row in rows], dtype=float)
    return {
        "average_cost_mean": float(average_costs.mean()),
        "average_cost_std": float(average_costs.std(ddof=0)),
        "discounted_cost_mean": float(discounted_costs.mean()),
        "final_state_mean": float(final_loads.mean()),
        "final_load_mean": float(final_loads.mean()),
        "tail_fraction_mean": float(tail_fractions.mean()),
        "boundary_hit_fraction_mean": float(boundary_fractions.mean()),
    }


def _state_load(state: Hashable, env: BaseAdversarialQueueEnv | None = None) -> int:
    if env is not None and hasattr(env, "queue_load"):
        return int(env.queue_load(state))
    if isinstance(state, tuple):
        return int(sum(int(value) for value in state))
    return int(state)


def _json_state(state: Hashable) -> int | list[int]:
    if isinstance(state, tuple):
        return [int(value) for value in state]
    return int(state)


def _state_sort_key(state: Hashable) -> tuple:
    if isinstance(state, tuple):
        return (sum(int(value) for value in state), tuple(int(value) for value in state))
    return (int(state),)


def _bvi_bound(result: BVIResult) -> int:
    if result.max_queue_length is not None:
        return result.max_queue_length
    max_value = 0
    for state in result.values:
        if isinstance(state, tuple):
            max_value = max(max_value, *(int(value) for value in state))
        else:
            max_value = max(max_value, int(state))
    return max_value


def _bvi_payoff_matrix(
    env: BaseAdversarialQueueEnv,
    result: BVIResult,
    state: Hashable,
    max_queue_length: int,
) -> np.ndarray:
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
                    bound_state_for_bvi(env, next_state, max_queue_length)
                ]
            payoff[ai, bi] = (
                env.cost(state, attacker_action, defender_action)
                + env.discount * expected_next
            )
    return payoff
