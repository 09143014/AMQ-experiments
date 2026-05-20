"""Feature maps for the routing benchmark."""

from __future__ import annotations

import numpy as np


def routing_features(
    state: tuple[int, ...],
    attacker_action: int,
    defender_action: int,
    feature_set: str = "basic",
) -> np.ndarray:
    """Return ``phi(x, a, b)`` for routing AMQ prototypes."""

    x = np.array(state, dtype=float)
    a = float(attacker_action)
    b = float(defender_action)

    if feature_set == "basic":
        return np.concatenate(
            [
                np.array(
                    [
                        1.0,
                        float(x.sum()),
                        float(np.dot(x, x)),
                        float(x.max() - x.min()) if x.size else 0.0,
                        a,
                        b,
                        a * b,
                    ],
                    dtype=float,
                ),
                x,
            ]
        )

    if feature_set == "action_interaction":
        attacker = _one_hot(attacker_action, 2)
        defender = _one_hot(defender_action, 2)
        aggregate = np.array(
            [
                1.0,
                float(x.sum()),
                float(np.dot(x, x)),
                float(x.max() - x.min()) if x.size else 0.0,
            ],
            dtype=float,
        )
        return np.concatenate(
            [
                aggregate,
                x,
                attacker,
                defender,
                np.outer(attacker, defender).ravel(),
                aggregate[1:] * defender[1],
                x * defender[1],
                aggregate[1:] * attacker[1],
                x * attacker[1],
            ]
        )

    if feature_set == "full_action_interaction":
        base = _routing_state_basis(x)
        joint_action = attacker_action * 2 + defender_action
        joint = _one_hot(joint_action, 4)
        return np.outer(joint, base).ravel()

    if feature_set == "normalized_full_action_interaction":
        base = _normalized_routing_state_basis(x)
        joint_action = attacker_action * 2 + defender_action
        joint = _one_hot(joint_action, 4)
        return np.outer(joint, base).ravel()

    raise ValueError(f"unknown routing feature_set: {feature_set}")


def routing_feature_dim(num_queues: int, feature_set: str = "basic") -> int:
    return int(
        routing_features(
            state=tuple(0 for _ in range(num_queues)),
            attacker_action=0,
            defender_action=0,
            feature_set=feature_set,
        ).shape[0]
    )


def _routing_state_basis(x: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.array(
                [
                    1.0,
                    float(x.sum()),
                    float(np.dot(x, x)),
                    float(x.max() - x.min()) if x.size else 0.0,
                ],
                dtype=float,
            ),
            x,
        ]
    )


def _normalized_routing_state_basis(x: np.ndarray) -> np.ndarray:
    scale = max(float(x.size), 1.0)
    return np.concatenate(
        [
            np.array(
                [
                    1.0,
                    float(x.sum() / scale),
                    float(np.dot(x, x) / scale),
                    (float(x.max() - x.min()) / scale) if x.size else 0.0,
                ],
                dtype=float,
            ),
            x / scale,
        ]
    )


def _one_hot(index: int, size: int) -> np.ndarray:
    if not 0 <= index < size:
        raise ValueError(f"index {index} outside one-hot size {size}")
    out = np.zeros(size, dtype=float)
    out[index] = 1.0
    return out
