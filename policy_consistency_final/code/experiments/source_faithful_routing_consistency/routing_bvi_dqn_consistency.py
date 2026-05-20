#!/usr/bin/env python3
"""Source-faithful routing BVI vs two-hidden-layer DQN consistency probe.

This isolated experiment follows the routing security game in
``paper source/BVI source.pdf``:

* finite truncated state grid ``{0, ..., B}^n``;
* Algorithm 2 style Shapley/value iteration for the attacker-defender game;
* two actions for each player: attacker ``NA/A`` and defender ``NP/P``;
* DQN-style NNQ with two hidden ReLU layers, replay buffer, target network,
  epsilon-greedy minimax behavior, and sampled TD backup only.
* an optional neural fitted minimax-Q diagnostic that still learns a neural
  2x2 Q function from Bellman targets, but uses exact environment expectations
  to reduce sampling noise.

No BVI labels are used during DQN training.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


State = tuple[int, ...]


@dataclass(frozen=True)
class RoutingSecurityParams:
    arrival_rate: float = 0.15
    service_rate: float = 0.375
    discount_rate: float = 0.10
    attack_cost: float = 0.10
    defend_cost: float = 0.20
    bound: int = 20
    num_queues: int = 3

    @property
    def denominator(self) -> float:
        return self.discount_rate + self.arrival_rate + self.num_queues * self.service_rate

    @property
    def lambda_tilde(self) -> float:
        return self.arrival_rate / self.denominator

    @property
    def mu_tilde(self) -> float:
        return self.service_rate / self.denominator

    @property
    def termination_probability(self) -> float:
        return self.discount_rate / self.denominator


def all_states(bound: int, num_queues: int) -> list[State]:
    return list(itertools.product(range(bound + 1), repeat=num_queues))


def shortest_arrival_state(state: State, bound: int) -> State:
    index = min(range(len(state)), key=lambda idx: (state[idx], idx))
    next_state = list(state)
    next_state[index] = min(next_state[index] + 1, bound)
    return tuple(next_state)


def longest_arrival_state(state: State, bound: int) -> State:
    index = max(range(len(state)), key=lambda idx: (state[idx], -idx))
    next_state = list(state)
    next_state[index] = min(next_state[index] + 1, bound)
    return tuple(next_state)


def service_state(state: State, queue_index: int) -> State:
    next_state = list(state)
    next_state[queue_index] = max(next_state[queue_index] - 1, 0)
    return tuple(next_state)


def next_arrival_state(state: State, attacker_action: int, defender_action: int, bound: int) -> State:
    if attacker_action == 1 and defender_action == 0:
        return longest_arrival_state(state, bound)
    return shortest_arrival_state(state, bound)


def immediate_cost(
    state: State,
    attacker_action: int,
    defender_action: int,
    params: RoutingSecurityParams,
) -> float:
    return (
        float(sum(state))
        + params.defend_cost * float(defender_action)
        - params.attack_cost * float(attacker_action)
    )


def matrix_game(cost_matrix: np.ndarray) -> dict[str, Any]:
    """Solve min_defender max_attacker cost_matrix[a, d]."""

    matrix = np.asarray(cost_matrix, dtype=float)
    if matrix.shape != (2, 2):
        raise ValueError("this source-faithful probe expects 2x2 matrix games")
    a, b = float(matrix[0, 0]), float(matrix[0, 1])
    c, d = float(matrix[1, 0]), float(matrix[1, 1])

    defender_candidates = [0.0, 1.0]
    denom = (a - b) - (c - d)
    if abs(denom) > 1e-12:
        p_col0 = (d - b) / denom
        if 0.0 <= p_col0 <= 1.0:
            defender_candidates.append(float(p_col0))
    best_defender_p = 0.0
    value = float("inf")
    for p_col0 in defender_candidates:
        row0 = b + (a - b) * p_col0
        row1 = d + (c - d) * p_col0
        candidate_value = max(row0, row1)
        if candidate_value < value - 1e-12:
            value = candidate_value
            best_defender_p = p_col0
    defender = np.asarray([best_defender_p, 1.0 - best_defender_p], dtype=float)

    attacker_candidates = [0.0, 1.0]
    denom = (a - c) - (b - d)
    if abs(denom) > 1e-12:
        q_row0 = (d - c) / denom
        if 0.0 <= q_row0 <= 1.0:
            attacker_candidates.append(float(q_row0))
    best_attacker_q = 0.0
    best_lower_value = -float("inf")
    for q_row0 in attacker_candidates:
        col0 = c + (a - c) * q_row0
        col1 = d + (b - d) * q_row0
        candidate_value = min(col0, col1)
        if candidate_value > best_lower_value + 1e-12:
            best_lower_value = candidate_value
            best_attacker_q = q_row0
    attacker = np.asarray([best_attacker_q, 1.0 - best_attacker_q], dtype=float)
    return {
        "value": value,
        "attacker_strategy": attacker,
        "defender_strategy": defender,
    }


def bvi_payoff_matrix(state: State, values: np.ndarray, params: RoutingSecurityParams) -> np.ndarray:
    """Build the source-paper auxiliary matrix M(x, V)."""

    short_state = shortest_arrival_state(state, params.bound)
    long_state = longest_arrival_state(state, params.bound)
    base = (
        float(sum(state))
        + params.mu_tilde
        * sum(values[service_state(state, queue_index)] for queue_index in range(params.num_queues))
        + params.lambda_tilde * values[short_state]
    )
    delta = params.lambda_tilde * (values[long_state] - values[short_state])
    return base + np.array(
        [
            [0.0, params.defend_cost],
            [-params.attack_cost + delta, -params.attack_cost + params.defend_cost],
        ],
        dtype=float,
    )


def source_auxiliary_game(
    state: State,
    values: np.ndarray,
    params: RoutingSecurityParams,
) -> dict[str, Any]:
    """Solve the source-paper auxiliary game using its closed-form regimes.

    For Algorithm 2 in the BVI source, each state only depends on

    ``delta = lambda_tilde * (V(x + e_max) - V(x + e_min))``.

    The Shapley-Snow solution reduces to three cases:

    * low risk: no attack, no defend;
    * medium risk: attack, no defend;
    * high risk: mixed attack and mixed defend.
    """

    short_state = shortest_arrival_state(state, params.bound)
    long_state = longest_arrival_state(state, params.bound)
    base = (
        float(sum(state))
        + params.mu_tilde
        * sum(values[service_state(state, queue_index)] for queue_index in range(params.num_queues))
        + params.lambda_tilde * values[short_state]
    )
    delta = max(
        float(params.lambda_tilde * (values[long_state] - values[short_state])),
        0.0,
    )
    matrix = base + np.array(
        [
            [0.0, params.defend_cost],
            [-params.attack_cost + delta, -params.attack_cost + params.defend_cost],
        ],
        dtype=float,
    )
    if delta <= params.attack_cost + 1e-12:
        attacker = np.asarray([1.0, 0.0], dtype=float)
        defender = np.asarray([1.0, 0.0], dtype=float)
        value = base
    elif delta <= params.defend_cost + 1e-12:
        attacker = np.asarray([0.0, 1.0], dtype=float)
        defender = np.asarray([1.0, 0.0], dtype=float)
        value = base - params.attack_cost + delta
    else:
        p_attack = params.defend_cost / delta
        p_defend = 1.0 - params.attack_cost / delta
        attacker = np.asarray([1.0 - p_attack, p_attack], dtype=float)
        defender = np.asarray([1.0 - p_defend, p_defend], dtype=float)
        value = base + params.defend_cost - (
            params.attack_cost * params.defend_cost / delta
        )
    return {
        "value": float(value),
        "attacker_strategy": attacker,
        "defender_strategy": defender,
        "q_matrix": matrix,
        "delta": delta,
    }


def run_source_bvi(params: RoutingSecurityParams, tolerance: float, max_iterations: int) -> dict[str, Any]:
    values = np.zeros((params.bound + 1,) * params.num_queues, dtype=float)
    states = all_states(params.bound, params.num_queues)
    residual = math.inf
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        old = values.copy()
        new_values = old.copy()
        residual = 0.0
        for state in states:
            game = source_auxiliary_game(state, old, params)
            new_values[state] = game["value"]
            residual = max(residual, abs(new_values[state] - old[state]))
        values = new_values
        if residual < tolerance:
            break

    policy: dict[State, dict[str, Any]] = {}
    for state in states:
        game = source_auxiliary_game(state, values, params)
        policy[state] = {
            "value": float(game["value"]),
            "attacker_strategy": game["attacker_strategy"],
            "defender_strategy": game["defender_strategy"],
            "attacker_action": int(np.argmax(game["attacker_strategy"])),
            "defender_action": int(np.argmax(game["defender_strategy"])),
            "q_matrix": game["q_matrix"],
            "delta": game["delta"],
        }
    return {
        "values": values,
        "policy": policy,
        "iterations": iteration,
        "residual": residual,
    }


class TwoLayerDQN:
    def __init__(
        self,
        rng: np.random.Generator,
        hidden_size: int,
        input_size: int,
        output_size: int = 4,
        architecture: str = "standard",
    ):
        if architecture not in {"standard", "dueling"}:
            raise ValueError("architecture must be 'standard' or 'dueling'")
        self.architecture = architecture
        self.output_size = output_size
        scale1 = math.sqrt(2.0 / input_size)
        scale2 = math.sqrt(2.0 / hidden_size)
        self.w1 = rng.normal(0.0, scale1, size=(input_size, hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.w2 = rng.normal(0.0, scale2, size=(hidden_size, hidden_size))
        self.b2 = np.zeros(hidden_size)
        if architecture == "standard":
            self.w3 = rng.normal(0.0, 0.01, size=(hidden_size, output_size))
            self.b3 = np.zeros(output_size)
        else:
            self.wv = rng.normal(0.0, 0.01, size=(hidden_size, 1))
            self.bv = np.zeros(1)
            self.wa = rng.normal(0.0, 0.01, size=(hidden_size, output_size))
            self.ba = np.zeros(output_size)
        self.m = {name: np.zeros_like(value) for name, value in self.params().items()}
        self.v = {name: np.zeros_like(value) for name, value in self.params().items()}
        self.t = 0

    def params(self) -> dict[str, np.ndarray]:
        shared = {
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
        }
        if self.architecture == "standard":
            shared.update({"w3": self.w3, "b3": self.b3})
        else:
            shared.update({"wv": self.wv, "bv": self.bv, "wa": self.wa, "ba": self.ba})
        return shared

    def copy(self) -> "TwoLayerDQN":
        clone = object.__new__(TwoLayerDQN)
        clone.architecture = self.architecture
        clone.output_size = self.output_size
        for name, value in self.params().items():
            setattr(clone, name, value.copy())
        clone.m = {name: np.zeros_like(value) for name, value in clone.params().items()}
        clone.v = {name: np.zeros_like(value) for name, value in clone.params().items()}
        clone.t = 0
        return clone

    def soft_update_from(self, source: "TwoLayerDQN", tau: float) -> None:
        for name, value in self.params().items():
            setattr(self, name, (1.0 - tau) * value + tau * getattr(source, name))

    def predict_batch(self, states: np.ndarray) -> np.ndarray:
        _x, h1_pre, h1, h2_pre, h2, out = self._forward(states)
        return out

    def predict(self, state_features: np.ndarray) -> np.ndarray:
        return self.predict_batch(state_features.reshape(1, -1))[0]

    def train_batch(
        self,
        states: np.ndarray,
        action_indices: np.ndarray,
        targets: np.ndarray,
        learning_rate: float,
        loss_type: str,
        huber_delta: float,
        grad_clip_norm: float | None,
        sample_weights: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray]:
        x, h1_pre, h1, h2_pre, h2, out = self._forward(states)
        batch_size = states.shape[0]
        chosen = out[np.arange(batch_size), action_indices]
        errors = chosen - targets
        weights = (
            np.ones(batch_size, dtype=float)
            if sample_weights is None
            else np.asarray(sample_weights, dtype=float)
        )
        weights = weights / max(float(weights.mean()), 1e-12)
        if loss_type == "mse":
            loss = float(np.mean(weights * errors * errors))
            chosen_grad = weights * 2.0 * errors / batch_size
        elif loss_type == "huber":
            abs_errors = np.abs(errors)
            quadratic = np.minimum(abs_errors, huber_delta)
            linear = abs_errors - quadratic
            loss = float(np.mean(weights * (0.5 * quadratic * quadratic + huber_delta * linear)))
            chosen_grad = weights * np.where(
                abs_errors <= huber_delta,
                errors,
                huber_delta * np.sign(errors),
            ) / batch_size
        else:
            raise ValueError("loss_type must be 'mse' or 'huber'")

        grad_out = np.zeros_like(out)
        grad_out[np.arange(batch_size), action_indices] = chosen_grad

        if self.architecture == "standard":
            grad_w3 = h2.T @ grad_out
            grad_b3 = grad_out.sum(axis=0)
            grad_h2 = grad_out @ self.w3.T
            output_grads = {"w3": grad_w3, "b3": grad_b3}
        else:
            grad_value = grad_out.sum(axis=1, keepdims=True)
            grad_advantage = grad_out - grad_out.mean(axis=1, keepdims=True)
            grad_wv = h2.T @ grad_value
            grad_bv = grad_value.sum(axis=0)
            grad_wa = h2.T @ grad_advantage
            grad_ba = grad_advantage.sum(axis=0)
            grad_h2 = grad_value @ self.wv.T + grad_advantage @ self.wa.T
            output_grads = {"wv": grad_wv, "bv": grad_bv, "wa": grad_wa, "ba": grad_ba}
        grad_h2_pre = grad_h2 * (h2_pre > 0.0)
        grad_w2 = h1.T @ grad_h2_pre
        grad_b2 = grad_h2_pre.sum(axis=0)
        grad_h1 = grad_h2_pre @ self.w2.T
        grad_h1_pre = grad_h1 * (h1_pre > 0.0)
        grad_w1 = x.T @ grad_h1_pre
        grad_b1 = grad_h1_pre.sum(axis=0)

        grads = {
            "w1": grad_w1,
            "b1": grad_b1,
            "w2": grad_w2,
            "b2": grad_b2,
        }
        grads.update(output_grads)
        if grad_clip_norm is not None:
            total_norm = math.sqrt(sum(float(np.sum(grad * grad)) for grad in grads.values()))
            if total_norm > grad_clip_norm:
                scale = grad_clip_norm / (total_norm + 1e-12)
                grads = {name: grad * scale for name, grad in grads.items()}
        self._adam_step(grads, learning_rate)
        return loss, np.abs(errors)

    def train_full_batch(
        self,
        states: np.ndarray,
        targets: np.ndarray,
        learning_rate: float,
        loss_type: str,
        huber_delta: float,
        grad_clip_norm: float | None,
    ) -> tuple[float, np.ndarray]:
        x, h1_pre, h1, h2_pre, h2, out = self._forward(states)
        errors = out - targets
        batch_size = states.shape[0]
        normalizer = float(batch_size * self.output_size)
        if loss_type == "mse":
            loss = float(np.mean(errors * errors))
            grad_out = 2.0 * errors / normalizer
        elif loss_type == "huber":
            abs_errors = np.abs(errors)
            quadratic = np.minimum(abs_errors, huber_delta)
            linear = abs_errors - quadratic
            loss = float(np.mean(0.5 * quadratic * quadratic + huber_delta * linear))
            grad_out = np.where(
                abs_errors <= huber_delta,
                errors,
                huber_delta * np.sign(errors),
            ) / normalizer
        else:
            raise ValueError("loss_type must be 'mse' or 'huber'")

        if self.architecture == "standard":
            grad_w3 = h2.T @ grad_out
            grad_b3 = grad_out.sum(axis=0)
            grad_h2 = grad_out @ self.w3.T
            output_grads = {"w3": grad_w3, "b3": grad_b3}
        else:
            grad_value = grad_out.sum(axis=1, keepdims=True)
            grad_advantage = grad_out - grad_out.mean(axis=1, keepdims=True)
            grad_wv = h2.T @ grad_value
            grad_bv = grad_value.sum(axis=0)
            grad_wa = h2.T @ grad_advantage
            grad_ba = grad_advantage.sum(axis=0)
            grad_h2 = grad_value @ self.wv.T + grad_advantage @ self.wa.T
            output_grads = {"wv": grad_wv, "bv": grad_bv, "wa": grad_wa, "ba": grad_ba}
        grad_h2_pre = grad_h2 * (h2_pre > 0.0)
        grad_w2 = h1.T @ grad_h2_pre
        grad_b2 = grad_h2_pre.sum(axis=0)
        grad_h1 = grad_h2_pre @ self.w2.T
        grad_h1_pre = grad_h1 * (h1_pre > 0.0)
        grad_w1 = x.T @ grad_h1_pre
        grad_b1 = grad_h1_pre.sum(axis=0)

        grads = {
            "w1": grad_w1,
            "b1": grad_b1,
            "w2": grad_w2,
            "b2": grad_b2,
        }
        grads.update(output_grads)
        if grad_clip_norm is not None:
            total_norm = math.sqrt(sum(float(np.sum(grad * grad)) for grad in grads.values()))
            if total_norm > grad_clip_norm:
                scale = grad_clip_norm / (total_norm + 1e-12)
                grads = {name: grad * scale for name, grad in grads.items()}
        self._adam_step(grads, learning_rate)
        return loss, np.abs(errors)

    def _forward(self, states: np.ndarray) -> tuple[np.ndarray, ...]:
        x = np.asarray(states, dtype=float)
        h1_pre = x @ self.w1 + self.b1
        h1 = np.maximum(h1_pre, 0.0)
        h2_pre = h1 @ self.w2 + self.b2
        h2 = np.maximum(h2_pre, 0.0)
        if self.architecture == "standard":
            out = h2 @ self.w3 + self.b3
        else:
            value = h2 @ self.wv + self.bv
            advantage = h2 @ self.wa + self.ba
            out = value + advantage - advantage.mean(axis=1, keepdims=True)
        return x, h1_pre, h1, h2_pre, h2, out

    def _adam_step(self, grads: dict[str, np.ndarray], learning_rate: float) -> None:
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        self.t += 1
        for name, grad in grads.items():
            self.m[name] = beta1 * self.m[name] + (1.0 - beta1) * grad
            self.v[name] = beta2 * self.v[name] + (1.0 - beta2) * (grad * grad)
            m_hat = self.m[name] / (1.0 - beta1**self.t)
            v_hat = self.v[name] / (1.0 - beta2**self.t)
            param = getattr(self, name)
            setattr(self, name, param - learning_rate * m_hat / (np.sqrt(v_hat) + eps))


def encode_state(state: State, bound: int, feature_set: str = "raw") -> np.ndarray:
    q = np.asarray(state, dtype=float)
    scale = max(float(bound), 1.0)
    raw = q / scale
    if feature_set == "raw":
        return raw
    if feature_set in {"structural", "poly2"}:
        sorted_q = np.sort(raw)
        total = float(raw.sum())
        min_q = float(raw.min())
        max_q = float(raw.max())
        gap = max_q - min_q
        features = [
            *raw,
            total / max(float(len(state)), 1.0),
            min_q,
            max_q,
            gap,
            float(raw.std()),
            *sorted_q,
            *(raw == min_q),
            *(raw == max_q),
        ]
        if feature_set == "poly2":
            base = [*raw, total, min_q, max_q, gap]
            features.extend(value * value for value in base)
            for left in range(len(base)):
                for right in range(left + 1, len(base)):
                    features.append(base[left] * base[right])
        return np.asarray(features, dtype=float)
    raise ValueError("feature_set must be 'raw', 'structural', or 'poly2'")


def dqn_matrix(
    network: TwoLayerDQN,
    state: State,
    bound: int,
    feature_set: str = "raw",
) -> np.ndarray:
    return network.predict(encode_state(state, bound, feature_set)).reshape(2, 2)


def sample_next_state(
    state: State,
    attacker_action: int,
    defender_action: int,
    params: RoutingSecurityParams,
    rng: np.random.Generator,
) -> State | None:
    r = rng.random()
    if r < params.termination_probability:
        return None
    r -= params.termination_probability
    if r < params.lambda_tilde:
        return next_arrival_state(state, attacker_action, defender_action, params.bound)
    r -= params.lambda_tilde
    for queue_index in range(params.num_queues):
        if r < params.mu_tilde:
            return service_state(state, queue_index)
        r -= params.mu_tilde
    return service_state(state, params.num_queues - 1)


def transition_from_uniform(
    state: State,
    attacker_action: int,
    defender_action: int,
    params: RoutingSecurityParams,
    uniform: float,
) -> State | None:
    r = uniform
    if r < params.termination_probability:
        return None
    r -= params.termination_probability
    if r < params.lambda_tilde:
        return next_arrival_state(state, attacker_action, defender_action, params.bound)
    r -= params.lambda_tilde
    for queue_index in range(params.num_queues):
        if r < params.mu_tilde:
            return service_state(state, queue_index)
        r -= params.mu_tilde
    return service_state(state, params.num_queues - 1)


def sample_binary_action(strategy: np.ndarray, uniform: float) -> int:
    return int(uniform >= float(strategy[0]))


def sample_initial_state(
    params: RoutingSecurityParams,
    rng: np.random.Generator,
    reset_state: State,
    initial_state_sampling: str,
) -> State:
    if initial_state_sampling == "zero":
        return reset_state
    if initial_state_sampling == "random_grid":
        return tuple(int(rng.integers(0, params.bound + 1)) for _ in range(params.num_queues))
    raise ValueError("initial_state_sampling must be 'zero' or 'random_grid'")


def train_dqn(
    params: RoutingSecurityParams,
    *,
    seed: int,
    total_steps: int,
    hidden_size: int,
    dqn_architecture: str,
    batch_size: int,
    replay_capacity: int,
    target_update_interval: int,
    epsilon: float,
    epsilon_final: float,
    epsilon_decay_steps: int,
    learning_rate: float,
    reset_state: State,
    reward_scale: float,
    state_sampling: str,
    initial_state_sampling: str,
    episode_horizon: int,
    loss_type: str,
    huber_delta: float,
    grad_clip_norm: float | None,
    target_update_tau: float,
    n_step: int,
    prioritized_replay_alpha: float,
    prioritized_replay_beta: float,
    priority_epsilon: float,
) -> dict[str, Any]:
    if n_step < 1:
        raise ValueError("n_step must be >= 1")
    rng = np.random.default_rng(seed)
    network = TwoLayerDQN(
        rng,
        hidden_size=hidden_size,
        input_size=params.num_queues,
        architecture=dqn_architecture,
    )
    target_network = network.copy()
    replay: list[tuple[State, int, int, float, State | None]] = []
    priorities: list[float] = []
    pending: list[tuple[State, int, int, float, State | None]] = []
    state = sample_initial_state(params, rng, reset_state, initial_state_sampling)
    losses: list[float] = []

    def push_n_step_entry(length: int) -> None:
        first_state, first_attacker, first_defender, _first_cost, _first_next = pending[0]
        total_cost = float(sum(item[3] for item in pending[:length]))
        bootstrap_state = pending[length - 1][4]
        replay.append((first_state, first_attacker, first_defender, total_cost, bootstrap_state))
        priorities.append(max(priorities, default=1.0))
        if len(replay) > replay_capacity:
            replay.pop(0)
            priorities.pop(0)
        pending.pop(0)

    def flush_pending() -> None:
        while pending:
            push_n_step_entry(len(pending))

    for step in range(1, total_steps + 1):
        if epsilon_decay_steps > 0:
            progress = min(1.0, step / float(epsilon_decay_steps))
            current_epsilon = epsilon + progress * (epsilon_final - epsilon)
        else:
            current_epsilon = epsilon
        if state_sampling == "rollout" and (step - 1) % episode_horizon == 0 and pending:
            flush_pending()
        if state_sampling == "uniform_grid":
            if pending:
                flush_pending()
            state = tuple(int(rng.integers(0, params.bound + 1)) for _ in range(params.num_queues))
        elif state_sampling == "rollout" and (step - 1) % episode_horizon == 0:
            state = sample_initial_state(params, rng, reset_state, initial_state_sampling)
        if rng.random() < current_epsilon:
            attacker_action = int(rng.integers(0, 2))
            defender_action = int(rng.integers(0, 2))
        else:
            game = matrix_game(dqn_matrix(network, state, params.bound))
            attacker_action = int(rng.choice((0, 1), p=game["attacker_strategy"]))
            defender_action = int(rng.choice((0, 1), p=game["defender_strategy"]))

        cost = immediate_cost(state, attacker_action, defender_action, params) / reward_scale
        next_state = sample_next_state(state, attacker_action, defender_action, params, rng)
        pending.append((state, attacker_action, defender_action, cost, next_state))
        if len(pending) >= n_step:
            push_n_step_entry(n_step)
        if next_state is None:
            flush_pending()

        if len(replay) >= batch_size:
            if prioritized_replay_alpha > 0.0:
                priority_array = np.asarray(priorities, dtype=float)
                scaled = np.power(priority_array + priority_epsilon, prioritized_replay_alpha)
                probabilities = scaled / scaled.sum()
                indices = rng.choice(
                    len(replay),
                    size=batch_size,
                    replace=False,
                    p=probabilities,
                )
                sample_weights = np.power(
                    len(replay) * probabilities[indices],
                    -prioritized_replay_beta,
                )
                sample_weights = sample_weights / max(float(sample_weights.max()), 1e-12)
            else:
                indices = rng.choice(len(replay), size=batch_size, replace=False)
                sample_weights = None
            batch = [replay[int(index)] for index in indices]
            states = np.vstack([encode_state(item[0], params.bound) for item in batch])
            action_indices = np.asarray([item[1] * 2 + item[2] for item in batch], dtype=int)
            targets = []
            for _s, _a, _d, c, ns in batch:
                if ns is None:
                    targets.append(c)
                else:
                    targets.append(
                        c + matrix_game(dqn_matrix(target_network, ns, params.bound))["value"]
                    )
            loss, abs_errors = network.train_batch(
                states,
                action_indices,
                np.asarray(targets, dtype=float),
                learning_rate,
                loss_type,
                huber_delta,
                grad_clip_norm,
                sample_weights,
            )
            if prioritized_replay_alpha > 0.0:
                for index, error in zip(indices, abs_errors, strict=True):
                    priorities[int(index)] = float(error) + priority_epsilon
            losses.append(loss)

        if target_update_tau < 1.0:
            target_network.soft_update_from(network, target_update_tau)
        elif step % target_update_interval == 0:
            target_network = network.copy()
        if state_sampling == "rollout":
            state = (
                sample_initial_state(params, rng, reset_state, initial_state_sampling)
                if next_state is None
                else next_state
            )
        elif state_sampling != "uniform_grid":
            raise ValueError("state_sampling must be 'rollout' or 'uniform_grid'")

    return {
        "network": network,
        "target_network": target_network,
        "loss_mean_last_100": float(np.mean(losses[-100:])) if losses else 0.0,
        "num_updates": len(losses),
        "final_state": state,
    }


def train_tabular_minimax_q(
    params: RoutingSecurityParams,
    *,
    seed: int,
    total_steps: int,
    epsilon: float,
    learning_rate: float,
    reset_state: State,
    reward_scale: float,
    initial_state_sampling: str,
    episode_horizon: int,
    learning_rate_decay_power: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    q_values = np.zeros((params.bound + 1,) * params.num_queues + (2, 2), dtype=float)
    visit_counts = np.zeros_like(q_values, dtype=np.int64)
    state = sample_initial_state(params, rng, reset_state, initial_state_sampling)
    td_errors: list[float] = []

    for step in range(1, total_steps + 1):
        if (step - 1) % episode_horizon == 0:
            state = sample_initial_state(params, rng, reset_state, initial_state_sampling)
        if rng.random() < epsilon:
            attacker_action = int(rng.integers(0, 2))
            defender_action = int(rng.integers(0, 2))
        else:
            game = matrix_game(q_values[state])
            attacker_action = int(rng.choice((0, 1), p=game["attacker_strategy"]))
            defender_action = int(rng.choice((0, 1), p=game["defender_strategy"]))

        cost = immediate_cost(state, attacker_action, defender_action, params) / reward_scale
        next_state = sample_next_state(state, attacker_action, defender_action, params, rng)
        if next_state is None:
            target = cost
        else:
            target = cost + matrix_game(q_values[next_state])["value"]
        old = q_values[state + (attacker_action, defender_action)]
        error = float(target - old)
        index = state + (attacker_action, defender_action)
        visit_counts[index] += 1
        step_size = learning_rate / (float(visit_counts[index]) ** learning_rate_decay_power)
        q_values[index] = old + step_size * error
        td_errors.append(abs(error))
        state = (
            sample_initial_state(params, rng, reset_state, initial_state_sampling)
            if next_state is None
            else next_state
        )

    return {
        "q_values": q_values,
        "td_error_mean_last_1000": float(np.mean(td_errors[-1000:])) if td_errors else 0.0,
        "min_visit_count": int(visit_counts.min()),
        "median_visit_count": float(np.median(visit_counts)),
        "num_updates": len(td_errors),
        "final_state": state,
    }


def train_model_based_minimax_q(
    params: RoutingSecurityParams,
    *,
    reward_scale: float,
    tolerance: float,
    max_iterations: int,
) -> dict[str, Any]:
    states = all_states(params.bound, params.num_queues)

    def build_q_from_values(values: np.ndarray) -> np.ndarray:
        q_values = np.zeros((params.bound + 1,) * params.num_queues + (2, 2), dtype=float)
        for state in states:
            for attacker_action in (0, 1):
                for defender_action in (0, 1):
                    route_state = next_arrival_state(
                        state,
                        attacker_action,
                        defender_action,
                        params.bound,
                    )
                    expected_future = (
                        params.lambda_tilde * values[route_state]
                        + params.mu_tilde
                        * sum(
                            values[service_state(state, queue_index)]
                            for queue_index in range(params.num_queues)
                        )
                    )
                    q_values[state + (attacker_action, defender_action)] = (
                        immediate_cost(state, attacker_action, defender_action, params)
                        / reward_scale
                        + expected_future
                    )
        return q_values

    q_values = np.zeros((params.bound + 1,) * params.num_queues + (2, 2), dtype=float)
    values = np.zeros((params.bound + 1,) * params.num_queues, dtype=float)
    residual = math.inf
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        old_values = values.copy()
        new_q_values = build_q_from_values(old_values)
        new_values = np.zeros_like(values)
        residual = 0.0
        for state in states:
            new_values[state] = matrix_game(new_q_values[state])["value"]
            residual = max(residual, abs(new_values[state] - old_values[state]))
        q_values = new_q_values
        values = new_values
        if residual < tolerance:
            break
    q_values = build_q_from_values(values)
    return {
        "q_values": q_values,
        "td_error_mean_last_1000": None,
        "min_visit_count": None,
        "median_visit_count": None,
        "iterations": iteration,
        "residual": residual,
        "num_updates": iteration * len(states) * 4,
    }


def train_sample_sync_minimax_q(
    params: RoutingSecurityParams,
    *,
    seed: int,
    reward_scale: float,
    samples_per_backup: int,
    tolerance: float,
    max_iterations: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    states = all_states(params.bound, params.num_queues)
    q_values = np.zeros((params.bound + 1,) * params.num_queues + (2, 2), dtype=float)
    values = np.zeros((params.bound + 1,) * params.num_queues, dtype=float)
    residual = math.inf
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        old_values = values.copy()
        new_q_values = np.zeros_like(q_values)
        new_values = np.zeros_like(values)
        residual = 0.0
        for state in states:
            for attacker_action in (0, 1):
                for defender_action in (0, 1):
                    total = 0.0
                    for _sample in range(samples_per_backup):
                        next_state = sample_next_state(
                            state,
                            attacker_action,
                            defender_action,
                            params,
                            rng,
                        )
                        total += 0.0 if next_state is None else old_values[next_state]
                    new_q_values[state + (attacker_action, defender_action)] = (
                        immediate_cost(state, attacker_action, defender_action, params)
                        / reward_scale
                        + total / samples_per_backup
                    )
            new_values[state] = matrix_game(new_q_values[state])["value"]
            residual = max(residual, abs(new_values[state] - old_values[state]))
        q_values = new_q_values
        values = new_values
        if residual < tolerance:
            break
    return {
        "q_values": q_values,
        "td_error_mean_last_1000": None,
        "min_visit_count": samples_per_backup,
        "median_visit_count": float(samples_per_backup),
        "iterations": iteration,
        "residual": residual,
        "num_updates": iteration * len(states) * 4 * samples_per_backup,
    }


def train_neural_fitted_minimax_q(
    params: RoutingSecurityParams,
    *,
    seed: int,
    hidden_size: int,
    dqn_architecture: str,
    batch_size: int,
    learning_rate: float,
    reward_scale: float,
    feature_set: str,
    fitted_iterations: int,
    fitted_epochs: int,
    loss_type: str,
    huber_delta: float,
    grad_clip_norm: float | None,
) -> dict[str, Any]:
    """Fit a neural Q function to no-BVI-label Bellman minimax targets.

    This is a deliberate relaxation from online sampled DQN: the backup uses
    exact known transition probabilities to lower variance.  The learned object
    remains a neural 2x2 Q network, and the target policy is never read from BVI.
    """

    rng = np.random.default_rng(seed)
    probe_features = encode_state((0,) * params.num_queues, params.bound, feature_set)
    network = TwoLayerDQN(
        rng,
        hidden_size=hidden_size,
        input_size=int(probe_features.size),
        architecture=dqn_architecture,
    )
    states = all_states(params.bound, params.num_queues)
    state_features = np.vstack(
        [encode_state(state, params.bound, feature_set) for state in states]
    )
    losses: list[float] = []
    residuals: list[float] = []
    target_fit_errors: list[float] = []
    num_updates = 0

    def values_from(net: TwoLayerDQN) -> np.ndarray:
        values = np.zeros((params.bound + 1,) * params.num_queues, dtype=float)
        predictions = net.predict_batch(state_features).reshape(len(states), 2, 2)
        for index, state in enumerate(states):
            values[state] = matrix_game(predictions[index])["value"]
        return values

    for _iteration in range(1, fitted_iterations + 1):
        target_network = network.copy()
        old_values = values_from(target_network)
        targets = np.zeros((len(states), 4), dtype=float)
        for row_index, state in enumerate(states):
            for attacker_action in (0, 1):
                for defender_action in (0, 1):
                    route_state = next_arrival_state(
                        state,
                        attacker_action,
                        defender_action,
                        params.bound,
                    )
                    expected_future = (
                        params.lambda_tilde * old_values[route_state]
                        + params.mu_tilde
                        * sum(
                            old_values[service_state(state, queue_index)]
                            for queue_index in range(params.num_queues)
                        )
                    )
                    action_index = attacker_action * 2 + defender_action
                    targets[row_index, action_index] = (
                        immediate_cost(state, attacker_action, defender_action, params)
                        / reward_scale
                        + expected_future
                    )

        before_values = old_values
        for _epoch in range(fitted_epochs):
            order = rng.permutation(len(states))
            for start in range(0, len(states), batch_size):
                indices = order[start : start + batch_size]
                loss, abs_errors = network.train_full_batch(
                    state_features[indices],
                    targets[indices],
                    learning_rate,
                    loss_type,
                    huber_delta,
                    grad_clip_norm,
                )
                losses.append(loss)
                target_fit_errors.append(float(abs_errors.mean()))
                num_updates += 1
        after_values = values_from(network)
        residuals.append(float(np.max(np.abs(after_values - before_values))))

    return {
        "network": network,
        "target_network": network.copy(),
        "loss_mean_last_100": float(np.mean(losses[-100:])) if losses else 0.0,
        "td_error_mean_last_1000": float(np.mean(target_fit_errors[-1000:]))
        if target_fit_errors
        else 0.0,
        "iterations": fitted_iterations,
        "residual": residuals[-1] if residuals else None,
        "num_updates": num_updates,
        "final_state": None,
        "feature_set": feature_set,
    }


def train_neural_fixed_point_q(
    params: RoutingSecurityParams,
    *,
    seed: int,
    hidden_size: int,
    dqn_architecture: str,
    batch_size: int,
    learning_rate: float,
    reward_scale: float,
    feature_set: str,
    fitting_epochs: int,
    loss_type: str,
    huber_delta: float,
    grad_clip_norm: float | None,
    tolerance: float,
    max_iterations: int,
    center_targets: bool,
) -> dict[str, Any]:
    """Fit an MLP to the model-based Bellman-Q fixed point.

    The target Q table is computed from the routing Bellman equation, not from
    BVI policy labels.  Centering removes the per-state constant that has no
    effect on a matrix game's equilibrium and lets the network focus on the
    action-dependent part of Q.
    """

    fixed_point = train_model_based_minimax_q(
        params,
        reward_scale=reward_scale,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    rng = np.random.default_rng(seed)
    probe_features = encode_state((0,) * params.num_queues, params.bound, feature_set)
    network = TwoLayerDQN(
        rng,
        hidden_size=hidden_size,
        input_size=int(probe_features.size),
        architecture=dqn_architecture,
    )
    states = all_states(params.bound, params.num_queues)
    state_features = np.vstack(
        [encode_state(state, params.bound, feature_set) for state in states]
    )
    targets = np.vstack([fixed_point["q_values"][state].reshape(4) for state in states])
    if center_targets:
        targets = targets - targets.mean(axis=1, keepdims=True)

    losses: list[float] = []
    fit_errors: list[float] = []
    num_updates = 0
    for _epoch in range(fitting_epochs):
        order = rng.permutation(len(states))
        for start in range(0, len(states), batch_size):
            indices = order[start : start + batch_size]
            loss, abs_errors = network.train_full_batch(
                state_features[indices],
                targets[indices],
                learning_rate,
                loss_type,
                huber_delta,
                grad_clip_norm,
            )
            losses.append(loss)
            fit_errors.append(float(abs_errors.mean()))
            num_updates += 1

    return {
        "network": network,
        "target_network": network.copy(),
        "loss_mean_last_100": float(np.mean(losses[-100:])) if losses else 0.0,
        "td_error_mean_last_1000": float(np.mean(fit_errors[-1000:]))
        if fit_errors
        else 0.0,
        "iterations": fixed_point["iterations"],
        "residual": fixed_point["residual"],
        "num_updates": num_updates,
        "final_state": None,
        "feature_set": feature_set,
        "center_targets": center_targets,
    }


def q_table_matrix(q_values: np.ndarray, state: State) -> np.ndarray:
    return q_values[state]


def evaluate_consistency(
    bvi_policy: dict[State, dict[str, Any]],
    params: RoutingSecurityParams,
    states: list[State],
    *,
    network: TwoLayerDQN | None = None,
    q_values: np.ndarray | None = None,
    network_feature_set: str = "raw",
) -> dict[str, Any]:
    if (network is None) == (q_values is None):
        raise ValueError("provide exactly one of network or q_values")
    rows = []
    defender_matches = 0
    attacker_matches = 0
    joint_matches = 0
    probability_gaps = []
    for state in states:
        bvi = bvi_policy[state]
        learned_matrix = (
            dqn_matrix(network, state, params.bound, network_feature_set)
            if network is not None
            else q_table_matrix(q_values, state)
        )
        dqn_game = matrix_game(learned_matrix)
        bvi_attacker_action = int(bvi["attacker_action"])
        bvi_defender_action = int(bvi["defender_action"])
        dqn_attacker_action = int(np.argmax(dqn_game["attacker_strategy"]))
        dqn_defender_action = int(np.argmax(dqn_game["defender_strategy"]))
        attacker_match = bvi_attacker_action == dqn_attacker_action
        defender_match = bvi_defender_action == dqn_defender_action
        joint_match = attacker_match and defender_match
        defender_matches += int(defender_match)
        attacker_matches += int(attacker_match)
        joint_matches += int(joint_match)
        gap = 0.25 * (
            float(np.abs(bvi["attacker_strategy"] - dqn_game["attacker_strategy"]).sum())
            + float(np.abs(bvi["defender_strategy"] - dqn_game["defender_strategy"]).sum())
        )
        probability_gaps.append(gap)
        rows.append(
            {
                "state": state,
                "bvi_attacker": bvi_attacker_action,
                "dqn_attacker": dqn_attacker_action,
                "bvi_defender": bvi_defender_action,
                "dqn_defender": dqn_defender_action,
                "attacker_match": attacker_match,
                "defender_match": defender_match,
                "joint_match": joint_match,
                "bvi_attacker_p_attack": float(bvi["attacker_strategy"][1]),
                "dqn_attacker_p_attack": float(dqn_game["attacker_strategy"][1]),
                "bvi_defender_p_defend": float(bvi["defender_strategy"][1]),
                "dqn_defender_p_defend": float(dqn_game["defender_strategy"][1]),
                "probability_gap": gap,
            }
        )
    total = len(states)
    return {
        "total": total,
        "attacker_agreement": attacker_matches / total,
        "defender_agreement": defender_matches / total,
        "joint_agreement": joint_matches / total,
        "mean_probability_similarity": 1.0 - float(np.mean(probability_gaps)),
        "max_probability_gap": float(np.max(probability_gaps)),
        "rows": rows,
    }


def learned_matrix_for_state(
    state: State,
    params: RoutingSecurityParams,
    *,
    network: TwoLayerDQN | None = None,
    q_values: np.ndarray | None = None,
    network_feature_set: str = "raw",
) -> np.ndarray:
    if (network is None) == (q_values is None):
        raise ValueError("provide exactly one of network or q_values")
    return (
        dqn_matrix(network, state, params.bound, network_feature_set)
        if network is not None
        else q_table_matrix(q_values, state)
    )


def evaluate_q_diagnostics(
    bvi_policy: dict[State, dict[str, Any]],
    params: RoutingSecurityParams,
    states: list[State],
    reward_scale: float,
    *,
    network: TwoLayerDQN | None = None,
    q_values: np.ndarray | None = None,
    network_feature_set: str = "raw",
) -> dict[str, Any]:
    errors = []
    bvi_ranges = []
    margin_rows = []
    bins = [
        ("margin_lt_0.05", 0.0, 0.05),
        ("margin_0.05_0.10", 0.05, 0.10),
        ("margin_0.10_0.25", 0.10, 0.25),
        ("margin_ge_0.25", 0.25, math.inf),
    ]
    bin_stats = {
        name: {"total": 0, "joint_matches": 0, "q_mae_sum": 0.0}
        for name, _lo, _hi in bins
    }
    regime_stats = {
        name: {"total": 0, "joint_matches": 0, "q_mae_sum": 0.0}
        for name in ("low_no_attack_no_defend", "medium_attack_no_defend", "high_mixed")
    }

    for state in states:
        bvi = bvi_policy[state]
        target_matrix = np.asarray(bvi["q_matrix"], dtype=float) / reward_scale
        learned_matrix = learned_matrix_for_state(
            state,
            params,
            network=network,
            q_values=q_values,
            network_feature_set=network_feature_set,
        )
        diff = learned_matrix - target_matrix
        abs_diff = np.abs(diff)
        errors.extend(abs_diff.reshape(-1).tolist())
        bvi_ranges.append(float(target_matrix.max() - target_matrix.min()))

        game = matrix_game(learned_matrix)
        attacker_match = int(np.argmax(bvi["attacker_strategy"])) == int(
            np.argmax(game["attacker_strategy"])
        )
        defender_match = int(np.argmax(bvi["defender_strategy"])) == int(
            np.argmax(game["defender_strategy"])
        )
        joint_match = attacker_match and defender_match
        margin = min(
            abs(float(bvi["attacker_strategy"][1]) - 0.5),
            abs(float(bvi["defender_strategy"][1]) - 0.5),
        )
        q_mae = float(abs_diff.mean())
        delta = float(bvi["delta"])
        if delta <= params.attack_cost + 1e-12:
            regime = "low_no_attack_no_defend"
        elif delta <= params.defend_cost + 1e-12:
            regime = "medium_attack_no_defend"
        else:
            regime = "high_mixed"
        margin_rows.append(
            {
                "state": state,
                "regime": regime,
                "margin": margin,
                "q_mae": q_mae,
                "q_max_abs_error": float(abs_diff.max()),
                "bvi_q_range": float(target_matrix.max() - target_matrix.min()),
                "joint_match": joint_match,
                "bvi_p_attack": float(bvi["attacker_strategy"][1]),
                "learned_p_attack": float(game["attacker_strategy"][1]),
                "bvi_p_defend": float(bvi["defender_strategy"][1]),
                "learned_p_defend": float(game["defender_strategy"][1]),
            }
        )
        for name, lo, hi in bins:
            if lo <= margin < hi:
                stat = bin_stats[name]
                stat["total"] += 1
                stat["joint_matches"] += int(joint_match)
                stat["q_mae_sum"] += q_mae
                break
        regime_stat = regime_stats[regime]
        regime_stat["total"] += 1
        regime_stat["joint_matches"] += int(joint_match)
        regime_stat["q_mae_sum"] += q_mae

    error_array = np.asarray(errors, dtype=float)
    q_range_array = np.asarray(bvi_ranges, dtype=float)
    finalized_bins = {}
    for name, stat in bin_stats.items():
        total = stat["total"]
        finalized_bins[name] = {
            "total": total,
            "joint_agreement": stat["joint_matches"] / total if total else None,
            "q_mae": stat["q_mae_sum"] / total if total else None,
        }
    finalized_regimes = {}
    for name, stat in regime_stats.items():
        total = stat["total"]
        finalized_regimes[name] = {
            "total": total,
            "joint_agreement": stat["joint_matches"] / total if total else None,
            "q_mae": stat["q_mae_sum"] / total if total else None,
        }

    worst_q_states = sorted(margin_rows, key=lambda row: row["q_mae"], reverse=True)[:10]
    near_tie_disagreements = [
        row
        for row in sorted(margin_rows, key=lambda item: item["margin"])
        if not row["joint_match"]
    ][:10]
    return {
        "q_mae": float(error_array.mean()),
        "q_rmse": float(np.sqrt(np.mean(error_array * error_array))),
        "q_max_abs_error": float(error_array.max()),
        "bvi_q_range_mean": float(q_range_array.mean()),
        "bvi_q_range_p10": float(np.quantile(q_range_array, 0.10)),
        "bvi_q_range_p50": float(np.quantile(q_range_array, 0.50)),
        "by_policy_margin": finalized_bins,
        "by_bvi_regime": finalized_regimes,
        "worst_q_states": worst_q_states,
        "near_tie_disagreements": near_tie_disagreements,
    }


def default_rollout_initial_states(params: RoutingSecurityParams) -> list[State]:
    candidates = [
        (0,) * params.num_queues,
        tuple(
            min(params.bound, int(round(params.bound * fraction)))
            for fraction in np.linspace(0.25, 0.75, params.num_queues)
        ),
        tuple(
            min(params.bound, value)
            for value in ([params.bound, params.bound // 2, 0] + [0] * params.num_queues)[
                : params.num_queues
            ]
        ),
    ]
    unique = []
    for state in candidates:
        if state not in unique:
            unique.append(state)
    return unique


def evaluate_closed_loop_rollouts(
    bvi_policy: dict[State, dict[str, Any]],
    params: RoutingSecurityParams,
    *,
    horizon: int,
    seed: int,
    network: TwoLayerDQN | None = None,
    q_values: np.ndarray | None = None,
    network_feature_set: str = "raw",
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    cases = []
    for initial_state in default_rollout_initial_states(params):
        bvi_state = initial_state
        learned_state = initial_state
        rows = []
        attacker_matches = 0
        defender_matches = 0
        joint_matches = 0
        greedy_joint_matches = 0
        same_state_steps = 0
        for step in range(horizon):
            bvi = bvi_policy[bvi_state]
            learned_matrix = learned_matrix_for_state(
                learned_state,
                params,
                network=network,
                q_values=q_values,
                network_feature_set=network_feature_set,
            )
            learned_game = matrix_game(learned_matrix)
            attacker_u = float(rng.random())
            defender_u = float(rng.random())
            env_u = float(rng.random())

            bvi_attacker = sample_binary_action(bvi["attacker_strategy"], attacker_u)
            bvi_defender = sample_binary_action(bvi["defender_strategy"], defender_u)
            learned_attacker = sample_binary_action(learned_game["attacker_strategy"], attacker_u)
            learned_defender = sample_binary_action(learned_game["defender_strategy"], defender_u)

            attacker_match = bvi_attacker == learned_attacker
            defender_match = bvi_defender == learned_defender
            joint_match = attacker_match and defender_match
            greedy_joint_match = (
                int(np.argmax(bvi["attacker_strategy"]))
                == int(np.argmax(learned_game["attacker_strategy"]))
                and int(np.argmax(bvi["defender_strategy"]))
                == int(np.argmax(learned_game["defender_strategy"]))
            )
            attacker_matches += int(attacker_match)
            defender_matches += int(defender_match)
            joint_matches += int(joint_match)
            greedy_joint_matches += int(greedy_joint_match)
            same_state_steps += int(bvi_state == learned_state)

            next_bvi_state = transition_from_uniform(
                bvi_state,
                bvi_attacker,
                bvi_defender,
                params,
                env_u,
            )
            next_learned_state = transition_from_uniform(
                learned_state,
                learned_attacker,
                learned_defender,
                params,
                env_u,
            )
            rows.append(
                {
                    "step": step,
                    "bvi_state": bvi_state,
                    "learned_state": learned_state,
                    "same_state": bvi_state == learned_state,
                    "bvi_action": [bvi_attacker, bvi_defender],
                    "learned_action": [learned_attacker, learned_defender],
                    "joint_match": joint_match,
                    "greedy_joint_match": greedy_joint_match,
                    "bvi_p_attack": float(bvi["attacker_strategy"][1]),
                    "learned_p_attack": float(learned_game["attacker_strategy"][1]),
                    "bvi_p_defend": float(bvi["defender_strategy"][1]),
                    "learned_p_defend": float(learned_game["defender_strategy"][1]),
                    "env_uniform": env_u,
                }
            )
            bvi_state = initial_state if next_bvi_state is None else next_bvi_state
            learned_state = initial_state if next_learned_state is None else next_learned_state

        cases.append(
            {
                "initial_state": initial_state,
                "horizon": horizon,
                "attacker_agreement": attacker_matches / horizon,
                "defender_agreement": defender_matches / horizon,
                "sampled_joint_agreement": joint_matches / horizon,
                "greedy_joint_agreement": greedy_joint_matches / horizon,
                "same_state_fraction": same_state_steps / horizon,
                "final_bvi_state": bvi_state,
                "final_learned_state": learned_state,
                "rows": rows,
            }
        )
    return cases


def rollout_states(
    bvi_policy: dict[State, dict[str, Any]],
    params: RoutingSecurityParams,
    *,
    episodes: int,
    horizon: int,
    seed: int,
    initial_state: State,
) -> list[State]:
    rng = np.random.default_rng(seed)
    visited = []
    for _episode in range(episodes):
        state = initial_state
        for _step in range(horizon):
            visited.append(state)
            policy = bvi_policy[state]
            attacker_action = int(rng.choice((0, 1), p=policy["attacker_strategy"]))
            defender_action = int(rng.choice((0, 1), p=policy["defender_strategy"]))
            next_state = sample_next_state(state, attacker_action, defender_action, params, rng)
            state = initial_state if next_state is None else next_state
    return sorted(set(visited))


def write_json(path: Path, data: dict[str, Any]) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, dict):
            return {str(key): convert(val) for key, val in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(convert(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_dqn_parameters(output_path: Path, network: TwoLayerDQN) -> dict[str, str]:
    params = network.params()
    param_path = output_path.with_suffix(".dqn_params.npz")
    summary_path = output_path.with_suffix(".dqn_params_summary.json")
    param_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(param_path, **params)
    summary = {}
    for name, value in params.items():
        flat = value.reshape(-1)
        summary[name] = {
            "shape": list(value.shape),
            "mean": float(value.mean()),
            "std": float(value.std()),
            "min": float(value.min()),
            "max": float(value.max()),
            "first_values": flat[: min(10, flat.size)].tolist(),
        }
    write_json(summary_path, summary)
    return {
        "parameter_file": str(param_path),
        "parameter_summary_file": str(summary_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bound", type=int, default=20)
    parser.add_argument("--num-queues", type=int, default=3)
    parser.add_argument("--bvi-tolerance", type=float, default=1e-7)
    parser.add_argument("--bvi-max-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--learner",
        choices=(
            "dqn",
            "tabular",
            "model_based_q",
            "sample_sync_q",
            "neural_fqi",
            "neural_fixed_point_q",
        ),
        default="dqn",
    )
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dqn-architecture", choices=("standard", "dueling"), default="standard")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=20_000)
    parser.add_argument("--target-update-interval", type=int, default=500)
    parser.add_argument("--episode-horizon", type=int, default=75)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--eval-horizon", type=int, default=75)
    parser.add_argument("--epsilon", type=float, default=0.20)
    parser.add_argument("--epsilon-final", type=float, default=None)
    parser.add_argument("--epsilon-decay-steps", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--loss-type", choices=("mse", "huber"), default="mse")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--prioritized-replay-alpha", type=float, default=0.0)
    parser.add_argument("--prioritized-replay-beta", type=float, default=0.4)
    parser.add_argument("--priority-epsilon", type=float, default=1e-6)
    parser.add_argument(
        "--target-update-tau",
        type=float,
        default=1.0,
        help="Use tau < 1 for soft target updates; tau=1 keeps hard interval updates.",
    )
    parser.add_argument(
        "--tabular-learning-rate-decay-power",
        type=float,
        default=0.0,
        help="For tabular diagnostics, use alpha / visit_count**power.",
    )
    parser.add_argument(
        "--samples-per-backup",
        type=int,
        default=20,
        help="For sample_sync_q diagnostics, samples used for each state-action backup.",
    )
    parser.add_argument(
        "--network-feature-set",
        choices=("raw", "structural", "poly2"),
        default="raw",
        help=(
            "Input features for neural learners. raw is normalized queues only; "
            "structural/poly2 add queue summaries but no one-hot state ids."
        ),
    )
    parser.add_argument(
        "--fitted-iterations",
        type=int,
        default=120,
        help="For neural_fqi, outer fitted Bellman iterations.",
    )
    parser.add_argument(
        "--fitted-epochs",
        type=int,
        default=3,
        help="For neural_fqi, epochs fitting each Bellman target batch.",
    )
    parser.add_argument(
        "--fixed-point-fitting-epochs",
        type=int,
        default=5_000,
        help="For neural_fixed_point_q, epochs fitting the Bellman-Q fixed point.",
    )
    parser.add_argument(
        "--no-center-fixed-point-targets",
        action="store_true",
        help="For neural_fixed_point_q, fit absolute Q values instead of centered action gaps.",
    )
    parser.add_argument(
        "--reward-scale",
        choices=("denominator", "none"),
        default="denominator",
        help=(
            "Use the continuous-time HJB denominator to scale DQN costs. "
            "This is a positive payoff scaling and does not change matrix-game strategies."
        ),
    )
    parser.add_argument(
        "--reward-scale-value",
        type=float,
        default=None,
        help=(
            "Optional positive cost scale used for DQN targets. Positive scaling "
            "does not change the minimax equilibrium policies represented by an exact Q function."
        ),
    )
    parser.add_argument(
        "--state-sampling",
        choices=("rollout", "uniform_grid"),
        default="rollout",
        help=(
            "rollout is ordinary online DQN sampling from the simulator; "
            "uniform_grid is a no-BVI-label generative-state coverage check."
        ),
    )
    parser.add_argument(
        "--initial-state-sampling",
        choices=("zero", "random_grid"),
        default="zero",
        help=(
            "Initial state distribution for rollout episodes. random_grid starts each "
            "episode from a bounded state and then follows the simulator dynamics."
        ),
    )
    parser.add_argument("--output", default="experiments/source_faithful_routing_consistency/results/summary.json")
    args = parser.parse_args()

    params = RoutingSecurityParams(bound=args.bound, num_queues=args.num_queues)
    start = time.perf_counter()
    bvi = run_source_bvi(params, args.bvi_tolerance, args.bvi_max_iterations)
    bvi_seconds = time.perf_counter() - start

    start = time.perf_counter()
    if args.reward_scale_value is not None:
        if args.reward_scale_value <= 0.0:
            raise ValueError("--reward-scale-value must be positive")
        reward_scale = args.reward_scale_value
        reward_scale_label = f"value:{args.reward_scale_value:g}"
    else:
        reward_scale = params.denominator if args.reward_scale == "denominator" else 1.0
        reward_scale_label = args.reward_scale

    if args.learner == "dqn":
        network_feature_set = "raw"
        learner = train_dqn(
            params,
            seed=args.seed,
            total_steps=args.total_steps,
            hidden_size=args.hidden_size,
            dqn_architecture=args.dqn_architecture,
            batch_size=args.batch_size,
            replay_capacity=args.replay_capacity,
            target_update_interval=args.target_update_interval,
            epsilon=args.epsilon,
            epsilon_final=args.epsilon if args.epsilon_final is None else args.epsilon_final,
            epsilon_decay_steps=args.epsilon_decay_steps,
            learning_rate=args.learning_rate,
            reset_state=(0,) * args.num_queues,
            reward_scale=reward_scale,
            state_sampling=args.state_sampling,
            initial_state_sampling=args.initial_state_sampling,
            episode_horizon=args.episode_horizon,
            loss_type=args.loss_type,
            huber_delta=args.huber_delta,
            grad_clip_norm=args.grad_clip_norm,
            target_update_tau=args.target_update_tau,
            n_step=args.n_step,
            prioritized_replay_alpha=args.prioritized_replay_alpha,
            prioritized_replay_beta=args.prioritized_replay_beta,
            priority_epsilon=args.priority_epsilon,
        )
    elif args.learner == "tabular":
        learner = train_tabular_minimax_q(
            params,
            seed=args.seed,
            total_steps=args.total_steps,
            epsilon=args.epsilon,
            learning_rate=args.learning_rate,
            reset_state=(0,) * args.num_queues,
            reward_scale=reward_scale,
            initial_state_sampling=args.initial_state_sampling,
            episode_horizon=args.episode_horizon,
            learning_rate_decay_power=args.tabular_learning_rate_decay_power,
        )
    elif args.learner in {"model_based_q", "sample_sync_q"}:
        network_feature_set = "raw"
        if args.learner == "model_based_q":
            learner = train_model_based_minimax_q(
                params,
                reward_scale=reward_scale,
                tolerance=args.bvi_tolerance,
                max_iterations=args.bvi_max_iterations,
            )
        else:
            learner = train_sample_sync_minimax_q(
                params,
                seed=args.seed,
                reward_scale=reward_scale,
                samples_per_backup=args.samples_per_backup,
                tolerance=args.bvi_tolerance,
                max_iterations=args.bvi_max_iterations,
            )
    elif args.learner == "neural_fqi":
        network_feature_set = args.network_feature_set
        learner = train_neural_fitted_minimax_q(
            params,
            seed=args.seed,
            hidden_size=args.hidden_size,
            dqn_architecture=args.dqn_architecture,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            reward_scale=reward_scale,
            feature_set=args.network_feature_set,
            fitted_iterations=args.fitted_iterations,
            fitted_epochs=args.fitted_epochs,
            loss_type=args.loss_type,
            huber_delta=args.huber_delta,
            grad_clip_norm=args.grad_clip_norm,
        )
    else:
        network_feature_set = args.network_feature_set
        learner = train_neural_fixed_point_q(
            params,
            seed=args.seed,
            hidden_size=args.hidden_size,
            dqn_architecture=args.dqn_architecture,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            reward_scale=reward_scale,
            feature_set=args.network_feature_set,
            fitting_epochs=args.fixed_point_fitting_epochs,
            loss_type=args.loss_type,
            huber_delta=args.huber_delta,
            grad_clip_norm=args.grad_clip_norm,
            tolerance=args.bvi_tolerance,
            max_iterations=args.bvi_max_iterations,
            center_targets=not args.no_center_fixed_point_targets,
        )
    learner_seconds = time.perf_counter() - start

    grid_states = all_states(args.bound, args.num_queues)
    rollout_test_states = rollout_states(
        bvi["policy"],
        params,
        episodes=200,
        horizon=75,
        seed=args.seed + 10_000,
        initial_state=(0,) * args.num_queues,
    )
    if args.learner in {"dqn", "neural_fqi", "neural_fixed_point_q"}:
        grid = evaluate_consistency(
            bvi["policy"],
            params,
            grid_states,
            network=learner["network"],
            network_feature_set=network_feature_set,
        )
        rollout = evaluate_consistency(
            bvi["policy"],
            params,
            rollout_test_states,
            network=learner["network"],
            network_feature_set=network_feature_set,
        )
        q_diagnostics = evaluate_q_diagnostics(
            bvi["policy"],
            params,
            grid_states,
            reward_scale,
            network=learner["network"],
            network_feature_set=network_feature_set,
        )
        closed_loop_rollouts = evaluate_closed_loop_rollouts(
            bvi["policy"],
            params,
            horizon=args.eval_horizon,
            seed=args.seed + 20_000,
            network=learner["network"],
            network_feature_set=network_feature_set,
        )
        parameter_artifacts = save_dqn_parameters(Path(args.output), learner["network"])
    else:
        grid = evaluate_consistency(bvi["policy"], params, grid_states, q_values=learner["q_values"])
        rollout = evaluate_consistency(
            bvi["policy"],
            params,
            rollout_test_states,
            q_values=learner["q_values"],
        )
        q_diagnostics = evaluate_q_diagnostics(
            bvi["policy"],
            params,
            grid_states,
            reward_scale,
            q_values=learner["q_values"],
        )
        closed_loop_rollouts = evaluate_closed_loop_rollouts(
            bvi["policy"],
            params,
            horizon=args.eval_horizon,
            seed=args.seed + 20_000,
            q_values=learner["q_values"],
        )
        parameter_artifacts = {}
    summary = {
        "params": params.__dict__,
        "source": {
            "paper": "paper source/BVI source.pdf, Appendix A.5 Algorithm 2",
            "github_reference": "https://github.com/jiayiWang3/ParallelServer/blob/master/Fig6.ipynb",
            "note": "DQN training uses sampled TD targets only; BVI policies are used only for evaluation.",
        },
        "bvi": {
            "iterations": bvi["iterations"],
            "residual": bvi["residual"],
            "runtime_seconds": bvi_seconds,
        },
        "dqn": {
            "learner": args.learner,
            "architecture": (
                f"2 hidden ReLU layers ({args.dqn_architecture})"
                if args.learner in {"dqn", "neural_fqi", "neural_fixed_point_q"}
                else (
                    "tabular sampled minimax-Q"
                    if args.learner == "tabular"
                    else (
                        "model-based synchronous minimax-Q"
                        if args.learner == "model_based_q"
                        else "sample-based synchronous minimax-Q"
                    )
                )
            ),
            "dqn_architecture": args.dqn_architecture
            if args.learner in {"dqn", "neural_fqi", "neural_fixed_point_q"}
            else None,
            "network_feature_set": network_feature_set
            if args.learner in {"dqn", "neural_fqi", "neural_fixed_point_q"}
            else None,
            "hidden_size": args.hidden_size,
            "total_steps": args.total_steps,
            "batch_size": args.batch_size,
            "replay_capacity": args.replay_capacity,
            "target_update_interval": args.target_update_interval,
            "episode_horizon": args.episode_horizon,
            "n_step": args.n_step if args.learner == "dqn" else None,
            "fitted_iterations": args.fitted_iterations if args.learner == "neural_fqi" else None,
            "fitted_epochs": args.fitted_epochs if args.learner == "neural_fqi" else None,
            "fixed_point_fitting_epochs": (
                args.fixed_point_fitting_epochs
                if args.learner == "neural_fixed_point_q"
                else None
            ),
            "center_fixed_point_targets": (
                not args.no_center_fixed_point_targets
                if args.learner == "neural_fixed_point_q"
                else None
            ),
            "eval_horizon": args.eval_horizon,
            "epsilon": args.epsilon,
            "epsilon_final": args.epsilon if args.epsilon_final is None else args.epsilon_final,
            "epsilon_decay_steps": args.epsilon_decay_steps,
            "learning_rate": args.learning_rate,
            "loss_type": args.loss_type
            if args.learner in {"dqn", "neural_fqi", "neural_fixed_point_q"}
            else None,
            "huber_delta": args.huber_delta
            if args.learner in {"dqn", "neural_fqi", "neural_fixed_point_q"}
            else None,
            "grad_clip_norm": args.grad_clip_norm
            if args.learner in {"dqn", "neural_fqi", "neural_fixed_point_q"}
            else None,
            "target_update_tau": args.target_update_tau if args.learner == "dqn" else None,
            "prioritized_replay_alpha": (
                args.prioritized_replay_alpha if args.learner == "dqn" else None
            ),
            "prioritized_replay_beta": (
                args.prioritized_replay_beta if args.learner == "dqn" else None
            ),
            "priority_epsilon": args.priority_epsilon if args.learner == "dqn" else None,
            "reward_scale": reward_scale_label,
            "state_sampling": args.state_sampling,
            "initial_state_sampling": args.initial_state_sampling,
            "tabular_learning_rate_decay_power": (
                args.tabular_learning_rate_decay_power if args.learner == "tabular" else None
            ),
            "samples_per_backup": args.samples_per_backup if args.learner == "sample_sync_q" else None,
            "loss_mean_last_100": learner.get("loss_mean_last_100"),
            "td_error_mean_last_1000": learner.get("td_error_mean_last_1000"),
            "min_visit_count": learner.get("min_visit_count"),
            "median_visit_count": learner.get("median_visit_count"),
            "model_based_iterations": learner.get("iterations"),
            "model_based_residual": learner.get("residual"),
            "num_updates": learner["num_updates"],
            "runtime_seconds": learner_seconds,
            **parameter_artifacts,
        },
        "consistency": {
            "grid": {key: value for key, value in grid.items() if key != "rows"},
            "bvi_rollout_visited_states": {key: value for key, value in rollout.items() if key != "rows"},
            "closed_loop_rollouts": [
                {key: value for key, value in case.items() if key != "rows"}
                for case in closed_loop_rollouts
            ],
        },
        "q_diagnostics": q_diagnostics,
        "sample_disagreements": {
            "grid": [row for row in grid["rows"] if not row["joint_match"]][:20],
            "bvi_rollout_visited_states": [row for row in rollout["rows"] if not row["joint_match"]][:20],
        },
        "closed_loop_rollout_rows": closed_loop_rollouts,
    }
    write_json(Path(args.output), summary)
    print(json.dumps(summary["consistency"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
