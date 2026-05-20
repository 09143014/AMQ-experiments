"""Polling benchmark with attackable longest-queue polling decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from adversarial_queueing.envs.base import BaseAdversarialQueueEnv


State = tuple[int, ...]


@dataclass(frozen=True)
class PollingConfig:
    """Configuration for a uniformized polling Markov game."""

    lambda_arrivals: tuple[float, ...]
    mu_service: float
    gamma: float = 0.95
    attack_cost: float = 0.5
    defend_cost: float = 0.2
    switch_cost: float = 0.1
    queue_cost: str = "sum"
    initial_queues: tuple[int, ...] | None = None
    initial_position: int = 0
    uniformization_rate: float | None = None
    bvi_max_queue_length: int = 3
    boundary_mode: str = "clip"

    def __post_init__(self) -> None:
        if len(self.lambda_arrivals) < 2:
            raise ValueError("polling benchmark requires at least two queues")
        if any(rate <= 0.0 for rate in self.lambda_arrivals):
            raise ValueError("lambda_arrivals must be positive")
        if self.mu_service <= 0.0:
            raise ValueError("mu_service must be positive")
        if not 0 < self.gamma < 1:
            raise ValueError("gamma must be in (0, 1)")
        if self.queue_cost not in {"sum", "sum_square", "fairness"}:
            raise ValueError("queue_cost must be 'sum', 'sum_square', or 'fairness'")
        if self.boundary_mode != "clip":
            raise ValueError("only boundary_mode='clip' is implemented")
        if not 0 <= self.initial_position < self.num_queues:
            raise ValueError("initial_position must be a valid queue index")
        if self.initial_queues is not None and len(self.initial_queues) != self.num_queues:
            raise ValueError("initial_queues length must match lambda_arrivals")

    @property
    def num_queues(self) -> int:
        return len(self.lambda_arrivals)

    @property
    def initial_state_value(self) -> State:
        queues = (
            tuple(int(value) for value in self.initial_queues)
            if self.initial_queues is not None
            else tuple(0 for _ in self.lambda_arrivals)
        )
        return (*queues, int(self.initial_position))

    @property
    def uniformization_rate_value(self) -> float:
        if self.uniformization_rate is not None:
            return self.uniformization_rate
        return sum(self.lambda_arrivals) + self.mu_service

    @property
    def beta(self) -> float:
        rate = self.uniformization_rate_value
        return rate * (1.0 / self.gamma - 1.0)


class PollingEnv(BaseAdversarialQueueEnv):
    """Uniformized CTMC polling game.

    State is encoded as ``(x_0, ..., x_{n-1}, p)``, where ``p`` is the current
    server position before the next polling decision.
    """

    def __init__(self, config: PollingConfig):
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
        return [float(value) for value in self._coerce_state(state)]

    def queue_load(self, state) -> int:
        queues, _position = self._split_state(state)
        return int(sum(queues))

    def polling_targets(
        self,
        state,
        attacker_action: int,
        defender_action: int,
    ) -> tuple[int, ...]:
        queues, _position = self._split_state(state)
        if attacker_action == 1 and defender_action == 0:
            target_value = min(queues)
        else:
            target_value = max(queues)
        return tuple(index for index, value in enumerate(queues) if value == target_value)

    def instantaneous_cost(
        self,
        state,
        attacker_action: int,
        defender_action: int,
    ) -> float:
        queues, position = self._split_state(state)
        if self.config.queue_cost == "sum":
            queue_cost = float(sum(queues))
        elif self.config.queue_cost == "sum_square":
            queue_cost = float(sum(value * value for value in queues))
        else:
            mean = float(sum(queues)) / len(queues)
            queue_cost = float(sum((value - mean) ** 2 for value in queues))
        switch_probability = self._expected_switch_probability(
            queues,
            position,
            attacker_action,
            defender_action,
        )
        return (
            queue_cost
            + self.config.switch_cost * switch_probability
            - self.config.attack_cost * float(attacker_action)
            + self.config.defend_cost * float(defender_action)
        )

    def cost(self, state, attacker_action: int, defender_action: int, next_state=None) -> float:
        return self.instantaneous_cost(state, attacker_action, defender_action) / (
            self.uniformization_rate + self.config.beta
        )

    def transition_probabilities(
        self,
        state,
        attacker_action: int,
        defender_action: int,
    ) -> Mapping[State, float]:
        queues, _position = self._split_state(state)
        rate = self.uniformization_rate
        arrival_total = sum(self.config.lambda_arrivals)
        service_rate = self.config.mu_service
        self_prob = 1.0 - (arrival_total + service_rate) / rate
        if self_prob < -1e-12:
            raise ValueError("uniformization_rate is smaller than outgoing rate")

        probs: dict[State, float] = {}
        targets = self.polling_targets(state, attacker_action, defender_action)
        target_weight = 1.0 / len(targets)
        for target in targets:
            base_state = (*queues, target)
            probs[base_state] = probs.get(base_state, 0.0) + target_weight * max(0.0, self_prob)

            for index, arrival_rate in enumerate(self.config.lambda_arrivals):
                next_queues = list(queues)
                next_queues[index] += 1
                next_state = (*self._clip_queues(tuple(next_queues)), target)
                probs[next_state] = (
                    probs.get(next_state, 0.0)
                    + target_weight * arrival_rate / rate
                )

            if queues[target] > 0:
                next_queues = list(queues)
                next_queues[target] -= 1
                next_state = (*tuple(next_queues), target)
            else:
                next_state = (*queues, target)
            probs[next_state] = probs.get(next_state, 0.0) + target_weight * service_rate / rate
        return probs

    def step(self, attacker_action: int, defender_action: int):
        probs = self.transition_probabilities(self._state, attacker_action, defender_action)
        states = list(probs)
        weights = np.array([probs[state] for state in states], dtype=float)
        weights = weights / weights.sum()
        selected = int(self._rng.choice(len(states), p=weights))
        next_state = states[selected]
        one_step_cost = self.cost(self._state, attacker_action, defender_action, next_state)
        info = {
            "polling_targets": self.polling_targets(
                self._state,
                attacker_action,
                defender_action,
            ),
            "instantaneous_cost": self.instantaneous_cost(
                self._state,
                attacker_action,
                defender_action,
            ),
        }
        self._state = next_state
        return next_state, one_step_cost, info

    def _expected_switch_probability(
        self,
        queues: tuple[int, ...],
        position: int,
        attacker_action: int,
        defender_action: int,
    ) -> float:
        targets = self.polling_targets((*queues, position), attacker_action, defender_action)
        return float(sum(1 for target in targets if target != position) / len(targets))

    def _split_state(self, state) -> tuple[tuple[int, ...], int]:
        coerced = self._coerce_state(state)
        return coerced[:-1], int(coerced[-1])

    def _coerce_state(self, state) -> State:
        values = tuple(int(value) for value in state)
        if len(values) != self.config.num_queues + 1:
            raise ValueError("polling state length must equal num_queues + 1")
        queues = values[:-1]
        position = values[-1]
        if any(value < 0 for value in queues):
            raise ValueError("queue lengths must be nonnegative")
        if not 0 <= position < self.config.num_queues:
            raise ValueError("polling position must be a valid queue index")
        return values

    def _clip_queues(self, queues: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(min(value, self.config.bvi_max_queue_length) for value in queues)
