"""Feature maps for the service-rate-control benchmark."""

from __future__ import annotations

import numpy as np


def service_rate_features(
    state: int,
    attacker_action: int,
    defender_action: int,
    feature_set: str = "basic_quadratic",
    num_attacker_actions: int = 2,
    num_defender_actions: int = 3,
) -> np.ndarray:
    """Return ``phi(x, a, b)`` for service-rate-control AMQ.

    ``basic_quadratic`` intentionally matches the initial engineering
    proposal in the experiment spec: ``[1, x, x^2, a, b, ab]``.
    """

    x = float(state)
    a = float(attacker_action)
    b = float(defender_action)
    if feature_set == "basic_quadratic":
        return np.array([1.0, x, x * x, a, b, a * b], dtype=float)

    if feature_set == "action_interaction":
        attacker = _one_hot(attacker_action, num_attacker_actions)
        defender = _one_hot(defender_action, num_defender_actions)
        return np.concatenate(
            [
                np.array([1.0, x, x * x], dtype=float),
                attacker,
                defender,
                x * defender,
                x * x * defender,
                np.outer(attacker, defender).ravel(),
            ]
        )

    raise ValueError(f"unknown service-rate feature_set: {feature_set}")


def service_rate_feature_dim(
    feature_set: str = "basic_quadratic",
    num_attacker_actions: int = 2,
    num_defender_actions: int = 3,
) -> int:
    return int(
        service_rate_features(
            state=0,
            attacker_action=0,
            defender_action=0,
            feature_set=feature_set,
            num_attacker_actions=num_attacker_actions,
            num_defender_actions=num_defender_actions,
        ).shape[0]
    )


def _one_hot(index: int, size: int) -> np.ndarray:
    if not 0 <= index < size:
        raise ValueError(f"index {index} outside one-hot size {size}")
    out = np.zeros(size, dtype=float)
    out[index] = 1.0
    return out

