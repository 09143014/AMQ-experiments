"""Shared environment interface for adversarial queueing games."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Sequence


class BaseAdversarialQueueEnv(ABC):
    """Minimal interface shared by benchmark environments."""

    @abstractmethod
    def reset(self, seed: int | None = None):
        """Return an initial state."""

    @abstractmethod
    def step(self, attacker_action: int, defender_action: int):
        """Sample one transition and return ``(next_state, cost, info)``."""

    @abstractmethod
    def transition_probabilities(
        self, state, attacker_action: int, defender_action: int
    ) -> Mapping[object, float]:
        """Return uniformized transition probabilities for a state-action pair."""

    @abstractmethod
    def cost(self, state, attacker_action: int, defender_action: int, next_state=None) -> float:
        """Return one-step defender cost under the configured discount convention."""

    @abstractmethod
    def attacker_actions(self, state) -> Sequence[int]:
        """Return available attacker actions."""

    @abstractmethod
    def defender_actions(self, state) -> Sequence[int]:
        """Return available defender actions."""

    @abstractmethod
    def encode_state(self, state) -> list[float]:
        """Return a numeric representation of state."""

