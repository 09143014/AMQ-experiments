"""Parallel-queue routing benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from adversarial_queueing.envs.base import BaseAdversarialQueueEnv


State = tuple[int, ...]


@dataclass(frozen=True)
class RoutingConfig:
    """Configuration for the parallel-queue routing benchmark."""

    lambda_arrival: float
    mu_rates: tuple[float, ...]
    gamma: float = 0.95
    attack_cost: float = 0.5
    defend_cost: float = 0.2
    congestion_cost: str = "sum"
    initial_state: State | None = None
    uniformization_rate: float | None = None
    bvi_max_queue_length: int = 20
    boundary_mode: str = "clip"

    def __post_init__(self) -> None:
        if self.lambda_arrival <= 0:
            raise ValueError("lambda_arrival must be positive")
        if len(self.mu_rates) < 2:
            raise ValueError("routing benchmark requires at least two queues")
        if any(mu <= 0 for mu in self.mu_rates):
            raise ValueError("all mu_rates must be positive")
        if not 0 < self.gamma < 1:
            raise ValueError("gamma must be in (0, 1)")
        if self.congestion_cost not in {"sum", "sum_square"}:
            raise ValueError("congestion_cost must be 'sum' or 'sum_square'")
        if self.boundary_mode != "clip":
            raise ValueError("only boundary_mode='clip' is implemented")
        if self.initial_state is not None and len(self.initial_state) != len(self.mu_rates):
            raise ValueError("initial_state length must match mu_rates")

    @property
    def num_queues(self) -> int:
        return len(self.mu_rates)

    @property
    def initial_state_value(self) -> State:
        if self.initial_state is not None:
            return tuple(int(x) for x in self.initial_state)
        return tuple(0 for _ in self.mu_rates)

    @property
    def uniformization_rate_value(self) -> float:
        if self.uniformization_rate is not None:
            return self.uniformization_rate
        return self.lambda_arrival + sum(self.mu_rates)

    @property
    def beta(self) -> float:
        rate = self.uniformization_rate_value
        return rate * (1.0 / self.gamma - 1.0)


class RoutingEnv(BaseAdversarialQueueEnv):
    """Uniformized CTMC routing Markov game for parallel queues."""

    def __init__(self, config: RoutingConfig):
        self.config = config
        self._rng = np.random.default_rng()
        self._state = config.initial_state_value

    @property
    def discount(self) -> float:
        return self.config.gamma

    @property
    def uniformization_rate(self) -> float:
        return self.config.uniformization_rate_value

    def reset(self, seed: int | None = None) -> State:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = self.config.initial_state_value
        return self._state

    def set_state(self, state) -> State:
        self._state = self._coerce_state(state)
        return self._state

    def reset_to_state(self, state, seed: int | None = None) -> State:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = self._coerce_state(state)
        return self._state

    def attacker_actions(self, state) -> tuple[int, int]:
        return (0, 1)

    def defender_actions(self, state) -> tuple[int, int]:
        return (0, 1)

    def encode_state(self, state) -> list[float]:
        return [float(x) for x in self._coerce_state(state)]

    def routed_arrival_targets(
        self, state, attacker_action: int, defender_action: int
    ) -> tuple[int, ...]:
        x = self._coerce_state(state)
        if attacker_action == 1 and defender_action == 0:
            target_value = max(x)
        else:
            target_value = min(x)
        return tuple(index for index, value in enumerate(x) if value == target_value)

    def instantaneous_cost(self, state, attacker_action: int, defender_action: int) -> float:
        x = self._coerce_state(state)
        if self.config.congestion_cost == "sum":
            congestion = float(sum(x))
        else:
            congestion = float(sum(value * value for value in x))
        return (
            congestion
            - self.config.attack_cost * float(attacker_action)
            + self.config.defend_cost * float(defender_action)
        )

    def cost(self, state, attacker_action: int, defender_action: int, next_state=None) -> float:
        return self.instantaneous_cost(state, attacker_action, defender_action) / (
            self.uniformization_rate + self.config.beta
        )

    def transition_probabilities(
        self, state, attacker_action: int, defender_action: int
    ) -> Mapping[State, float]:
        x = self._coerce_state(state)
        rate = self.uniformization_rate
        probs: dict[State, float] = {}

        arrival_prob = self.config.lambda_arrival / rate
        targets = self.routed_arrival_targets(x, attacker_action, defender_action)
        for target in targets:
            next_state = list(x)
            next_state[target] += 1
            next_tuple = self._clip_state(tuple(next_state))
            probs[next_tuple] = probs.get(next_tuple, 0.0) + arrival_prob / len(targets)

        service_prob_total = 0.0
        for index, mu in enumerate(self.config.mu_rates):
            if x[index] <= 0:
                continue
            service_prob = mu / rate
            service_prob_total += service_prob
            next_state = list(x)
            next_state[index] -= 1
            next_tuple = tuple(next_state)
            probs[next_tuple] = probs.get(next_tuple, 0.0) + service_prob

        self_prob = 1.0 - arrival_prob - service_prob_total
        if self_prob < -1e-12:
            raise ValueError("uniformization_rate is smaller than outgoing rate")
        probs[x] = probs.get(x, 0.0) + max(0.0, self_prob)
        return probs

    def step(self, attacker_action: int, defender_action: int):
        probs = self.transition_probabilities(self._state, attacker_action, defender_action)
        states = list(probs)
        weights = np.array([probs[s] for s in states], dtype=float)
        weights = weights / weights.sum()
        selected = int(self._rng.choice(len(states), p=weights))
        next_state = states[selected]
        one_step_cost = self.cost(self._state, attacker_action, defender_action, next_state)
        info = {
            "arrival_targets": self.routed_arrival_targets(
                self._state, attacker_action, defender_action
            ),
            "instantaneous_cost": self.instantaneous_cost(
                self._state, attacker_action, defender_action
            ),
        }
        self._state = next_state
        return next_state, one_step_cost, info

    def _coerce_state(self, state) -> State:
        x = tuple(int(value) for value in state)
        if len(x) != self.config.num_queues:
            raise ValueError("state length must match number of queues")
        if any(value < 0 for value in x):
            raise ValueError("queue lengths must be nonnegative")
        return x

    def _clip_state(self, state: State) -> State:
        return tuple(min(value, self.config.bvi_max_queue_length) for value in state)
