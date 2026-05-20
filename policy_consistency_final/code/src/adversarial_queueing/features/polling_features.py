"""Feature maps for the polling benchmark."""

from __future__ import annotations

import numpy as np


def polling_features(
    state: tuple[int, ...],
    attacker_action: int,
    defender_action: int,
    feature_set: str = "basic",
) -> np.ndarray:
    """Return ``phi(s, a, b)`` for polling AMQ smoke experiments."""

    values = tuple(int(value) for value in state)
    queues = np.array(values[:-1], dtype=float)
    position = int(values[-1])
    a = float(attacker_action)
    b = float(defender_action)
    if feature_set == "basic":
        return np.concatenate(
            [
                np.array(
                    [
                        1.0,
                        float(queues.sum()),
                        float(np.dot(queues, queues)),
                        float(queues.max() - queues.min()) if queues.size else 0.0,
                        float(position),
                        a,
                        b,
                        a * b,
                    ],
                    dtype=float,
                ),
                queues,
            ]
        )
    if feature_set in {
        "action_interaction",
        "augmented_action_interaction",
        "calibrated_action_interaction",
    }:
        if feature_set == "calibrated_action_interaction":
            base = _polling_calibrated_state_basis(queues, position)
        elif feature_set == "augmented_action_interaction":
            base = _polling_augmented_state_basis(queues, position)
        else:
            base = _polling_state_basis(queues, position)
        joint_action = attacker_action * 2 + defender_action
        joint = _one_hot(joint_action, 4)
        return np.outer(joint, base).ravel()
    raise ValueError(f"unknown polling feature_set: {feature_set}")


def polling_feature_dim(num_queues: int, feature_set: str = "basic") -> int:
    state = tuple(0 for _ in range(num_queues)) + (0,)
    return int(
        polling_features(
            state=state,
            attacker_action=0,
            defender_action=0,
            feature_set=feature_set,
        ).shape[0]
    )


def _polling_state_basis(queues: np.ndarray, position: int) -> np.ndarray:
    return np.concatenate(
        [
            np.array(
                [
                    1.0,
                    float(queues.sum()),
                    float(np.dot(queues, queues)),
                    float(queues.max() - queues.min()) if queues.size else 0.0,
                    float(position),
                ],
                dtype=float,
            ),
            queues,
        ]
    )


def _polling_augmented_state_basis(queues: np.ndarray, position: int) -> np.ndarray:
    min_value = float(queues.min()) if queues.size else 0.0
    max_value = float(queues.max()) if queues.size else 0.0
    position_one_hot = np.zeros(queues.size, dtype=float)
    if queues.size:
        position_one_hot[position] = 1.0
    return np.concatenate(
        [
            queues,
            position_one_hot,
            np.array(
                [
                    1.0,
                    float(queues.sum()),
                    min_value,
                    max_value,
                    max_value - min_value,
                    float(queues[position]) if queues.size else 0.0,
                    float(np.dot(queues, queues)),
                ],
                dtype=float,
            ),
            np.array(
                [1.0 if value == min_value else 0.0 for value in queues],
                dtype=float,
            ),
            np.array(
                [1.0 if value == max_value else 0.0 for value in queues],
                dtype=float,
            ),
        ]
    )


def _polling_calibrated_state_basis(queues: np.ndarray, position: int) -> np.ndarray:
    base = _polling_augmented_state_basis(queues, position)
    total = float(queues.sum())
    min_value = float(queues.min()) if queues.size else 0.0
    max_value = float(queues.max()) if queues.size else 0.0
    gap = max_value - min_value
    current_queue = float(queues[position]) if queues.size else 0.0
    return np.concatenate(
        [
            base,
            np.array(
                [
                    1.0 if total == 0.0 else 0.0,
                    1.0 if total <= 1.0 else 0.0,
                    1.0 if gap == 0.0 else 0.0,
                    1.0 if gap == 0.0 and total > 0.0 else 0.0,
                    1.0 if current_queue == 0.0 else 0.0,
                    total * gap,
                    current_queue * gap,
                ],
                dtype=float,
            ),
        ]
    )


def _one_hot(index: int, size: int) -> np.ndarray:
    if not 0 <= index < size:
        raise ValueError(f"index {index} outside one-hot size {size}")
    out = np.zeros(size, dtype=float)
    out[index] = 1.0
    return out
