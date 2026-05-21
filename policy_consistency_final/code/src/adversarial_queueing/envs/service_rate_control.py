"""Single-queue service-rate-control benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from adversarial_queueing.envs.base import BaseAdversarialQueueEnv


@dataclass(frozen=True)
class ServiceRateControlConfig:
    """Configuration for the service-rate-control defend game."""

    lambda_arrival: float
    mu_levels: tuple[float, float, float]
    service_costs: tuple[float, float, float]
    gamma: float = 0.95
    q_congestion: float = 1.0
    attack_cost: float = 0.5
    defend_cost: float = 0.2
    initial_state: int = 0
    uniformization_rate: float | None = None
    low_threshold: int = 5
    high_threshold: int = 15
    bvi_max_queue_length: int = 20
    boundary_mode: str = "clip"

    def __post_init__(self) -> None:
        if self.lambda_arrival <= 0:
            raise ValueError("lambda_arrival must be positive")
        if len(self.mu_levels) != 3:
            raise ValueError("service-rate-control benchmark requires exactly three mu_levels")
        if len(self.service_costs) != len(self.mu_levels):
            raise ValueError("service_costs must match mu_levels")
        if not 0 < self.gamma < 1:
            raise ValueError("gamma must be in (0, 1)")
        if self.low_threshold < 0:
            raise ValueError("low_threshold must be nonnegative")
        if self.high_threshold <= self.low_threshold:
            raise ValueError("high_threshold must be larger than low_threshold")
        if self.boundary_mode != "clip":
            raise ValueError("only boundary_mode='clip' is implemented in the baseline")

    @property
    def uniformization_rate_value(self) -> float:
        if self.uniformization_rate is not None:
            return self.uniformization_rate
        return self.lambda_arrival + max(self.mu_levels)

    @property
    def beta(self) -> float:
        rate = self.uniformization_rate_value
        return rate * (1.0 / self.gamma - 1.0)


class ServiceRateControlEnv(BaseAdversarialQueueEnv):
    """Uniformized CTMC service-rate-control Markov game."""

    def __init__(self, config: ServiceRateControlConfig):
        self.config = config
        self._rng = np.random.default_rng()
        self._state = config.initial_state

    @property
    def discount(self) -> float:
        return self.config.gamma

    @property
    def uniformization_rate(self) -> float:
        return self.config.uniformization_rate_value

    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = self.config.initial_state
        return self._state

    def attacker_actions(self, state) -> tuple[int, int]:
        return (0, 1)

    def defender_actions(self, state) -> tuple[int, ...]:
        del state
        return (0, 1)

    def encode_state(self, state) -> list[float]:
        return [float(state)]

    def baseline_service_level(self, state: int) -> int:
        """Return the threshold-policy service level for ``state``.

        Levels are encoded as 0=low, 1=medium, 2=high.
        """

        x = int(state)
        if x < self.config.low_threshold:
            return 0
        if x < self.config.high_threshold:
            return 1
        return 2

    def realized_service_level(
        self,
        state: int,
        attacker_action: int,
        defender_action: int,
    ) -> int:
        """Return the service level applied for one transition.

        A successful attack, i.e. attack without defense, forces high service
        for this step only. Otherwise the server follows the fixed threshold
        baseline policy.
        """

        if attacker_action == 1 and defender_action == 0:
            return 2
        return self.baseline_service_level(state)

    def realized_mu(self, state: int, attacker_action: int, defender_action: int) -> float:
        return self.config.mu_levels[
            self.realized_service_level(state, attacker_action, defender_action)
        ]

    def instantaneous_cost(self, state: int, attacker_action: int, defender_action: int) -> float:
        realized_level = self.realized_service_level(state, attacker_action, defender_action)
        service_cost = self.config.service_costs[realized_level]
        congestion = self.config.q_congestion * float(state * state)
        return (
            congestion
            + service_cost
            + self.config.defend_cost * defender_action
            - self.config.attack_cost * attacker_action
        )

    def cost(self, state, attacker_action: int, defender_action: int, next_state=None) -> float:
        # Exact uniformized conversion for discounted continuous-time cost.
        return self.instantaneous_cost(state, attacker_action, defender_action) / (
            self.uniformization_rate + self.config.beta
        )

    def transition_probabilities(
        self, state, attacker_action: int, defender_action: int
    ) -> Mapping[int, float]:
        x = int(state)
        rate = self.uniformization_rate
        arrival_prob = self.config.lambda_arrival / rate
        service_prob = (
            self.realized_mu(x, attacker_action, defender_action) / rate if x > 0 else 0.0
        )

        probs: dict[int, float] = {}
        next_up = x + 1
        if next_up > self.config.bvi_max_queue_length and self.config.boundary_mode == "clip":
            next_up = self.config.bvi_max_queue_length
        probs[next_up] = probs.get(next_up, 0.0) + arrival_prob

        if x > 0:
            probs[x - 1] = probs.get(x - 1, 0.0) + service_prob

        self_prob = 1.0 - arrival_prob - service_prob
        if self_prob < -1e-12:
            raise ValueError("uniformization_rate is smaller than outgoing rate")
        probs[x] = probs.get(x, 0.0) + max(0.0, self_prob)
        return probs

    def step(self, attacker_action: int, defender_action: int):
        probs = self.transition_probabilities(self._state, attacker_action, defender_action)
        states = list(probs)
        weights = np.array([probs[s] for s in states], dtype=float)
        weights = weights / weights.sum()
        next_state = int(self._rng.choice(states, p=weights))
        one_step_cost = self.cost(self._state, attacker_action, defender_action, next_state)
        info = {
            "baseline_service_level": self.baseline_service_level(self._state),
            "realized_service_level": self.realized_service_level(
                self._state, attacker_action, defender_action
            ),
            "realized_mu": self.realized_mu(self._state, attacker_action, defender_action),
            "instantaneous_cost": self.instantaneous_cost(
                self._state, attacker_action, defender_action
            ),
        }
        self._state = next_state
        return next_state, one_step_cost, info
