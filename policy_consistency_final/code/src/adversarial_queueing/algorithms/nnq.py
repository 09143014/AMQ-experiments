"""Neural-network minimax Q-learning smoke baseline implemented with NumPy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable

import numpy as np

from adversarial_queueing.algorithms.minimax_solver import solve_zero_sum_matrix_game
from adversarial_queueing.envs.base import BaseAdversarialQueueEnv
from adversarial_queueing.envs.polling import PollingEnv
from adversarial_queueing.envs.routing import RoutingEnv
from adversarial_queueing.envs.service_rate_control import ServiceRateControlEnv


@dataclass(frozen=True)
class NNQConfig:
    hidden_size: int = 32
    learning_rate: float = 0.001
    total_steps: int = 200
    batch_size: int = 16
    replay_capacity: int = 1000
    target_update_interval: int = 50
    epsilon: float = 0.2
    seed: int = 0
    log_interval: int = 20
    state_scale: float = 10.0
    state_feature_set: str = "env"
    forced_defender_action_probability: float = 0.0
    forced_defender_action: int | None = None
    exploring_starts_probability: float = 0.0
    exploring_starts_max_queue_length: int | None = None
    backup_mode: str = "sampled"


@dataclass(frozen=True)
class NNQResult:
    network: "NumpyQNetwork"
    metrics: list[dict[str, Any]]
    final_state: Hashable


class NumpyQNetwork:
    """One-hidden-layer Q network returning a Q matrix for finite actions."""

    def __init__(
        self,
        hidden_size: int,
        input_size: int,
        output_size: int,
        rng: np.random.Generator,
        state_scale: float,
    ):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.state_scale = state_scale
        self.w1 = rng.normal(0.0, 0.1, size=(input_size, hidden_size))
        self.b1 = np.zeros(hidden_size, dtype=float)
        self.w2 = rng.normal(0.0, 0.1, size=(hidden_size, output_size))
        self.b2 = np.zeros(output_size, dtype=float)
        self._adam_m = self._zeros_like_params()
        self._adam_v = self._zeros_like_params()
        self._adam_t = 0

    def copy(self) -> "NumpyQNetwork":
        clone = object.__new__(NumpyQNetwork)
        clone.hidden_size = self.hidden_size
        clone.input_size = self.input_size
        clone.output_size = self.output_size
        clone.state_scale = self.state_scale
        clone.w1 = self.w1.copy()
        clone.b1 = self.b1.copy()
        clone.w2 = self.w2.copy()
        clone.b2 = self.b2.copy()
        clone._adam_m = clone._zeros_like_params()
        clone._adam_v = clone._zeros_like_params()
        clone._adam_t = 0
        return clone

    def predict(self, encoded_state: np.ndarray) -> np.ndarray:
        x = self._input(encoded_state)
        hidden_pre = x @ self.w1 + self.b1
        hidden = np.maximum(hidden_pre, 0.0)
        return (hidden @ self.w2 + self.b2).ravel()

    def q_matrix(
        self,
        encoded_state: np.ndarray,
        num_attacker: int,
        num_defender: int,
    ) -> np.ndarray:
        return self.predict(encoded_state).reshape(num_attacker, num_defender)

    def train_batch(
        self,
        encoded_states: np.ndarray,
        action_indices: np.ndarray,
        targets: np.ndarray,
        learning_rate: float,
    ) -> float:
        grad_w1 = np.zeros_like(self.w1)
        grad_b1 = np.zeros_like(self.b1)
        grad_w2 = np.zeros_like(self.w2)
        grad_b2 = np.zeros_like(self.b2)
        losses = []

        for encoded_state, action_index, target in zip(
            encoded_states, action_indices, targets
        ):
            x = self._input(encoded_state)
            hidden_pre = x @ self.w1 + self.b1
            hidden = np.maximum(hidden_pre, 0.0)
            output = (hidden @ self.w2 + self.b2).ravel()
            error = output[int(action_index)] - float(target)
            losses.append(error * error)

            grad_output = np.zeros(self.output_size, dtype=float)
            grad_output[int(action_index)] = 2.0 * error / len(encoded_states)
            grad_w2 += np.outer(hidden.ravel(), grad_output)
            grad_b2 += grad_output
            grad_hidden = grad_output @ self.w2.T
            grad_hidden_pre = grad_hidden * (hidden_pre.ravel() > 0.0)
            grad_w1 += np.outer(x.ravel(), grad_hidden_pre)
            grad_b1 += grad_hidden_pre

        self._adam_step(
            {
                "w1": grad_w1,
                "b1": grad_b1,
                "w2": grad_w2,
                "b2": grad_b2,
            },
            learning_rate,
        )
        return float(np.mean(losses))

    def _input(self, encoded_state: np.ndarray) -> np.ndarray:
        x = np.asarray(encoded_state, dtype=float).reshape(1, self.input_size)
        return x / self.state_scale

    def _zeros_like_params(self) -> dict[str, np.ndarray]:
        return {
            "w1": np.zeros_like(self.w1),
            "b1": np.zeros_like(self.b1),
            "w2": np.zeros_like(self.w2),
            "b2": np.zeros_like(self.b2),
        }

    def _adam_step(self, grads: dict[str, np.ndarray], learning_rate: float) -> None:
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        self._adam_t += 1
        for name, grad in grads.items():
            self._adam_m[name] = beta1 * self._adam_m[name] + (1.0 - beta1) * grad
            self._adam_v[name] = beta2 * self._adam_v[name] + (1.0 - beta2) * (grad * grad)
            m_hat = self._adam_m[name] / (1.0 - beta1**self._adam_t)
            v_hat = self._adam_v[name] / (1.0 - beta2**self._adam_t)
            param = getattr(self, name)
            setattr(self, name, param - learning_rate * m_hat / (np.sqrt(v_hat) + eps))


class NNQTrainer:
    """Small NumPy NNQ trainer for finite-action benchmark smoke experiments."""

    def __init__(self, env: BaseAdversarialQueueEnv, config: NNQConfig):
        self.env = env
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        initial_state = env.reset(seed=config.seed)
        self.attacker_actions = tuple(env.attacker_actions(initial_state))
        self.defender_actions = tuple(env.defender_actions(initial_state))
        self._attacker_action_to_index = {
            int(action): index for index, action in enumerate(self.attacker_actions)
        }
        self._defender_action_to_index = {
            int(action): index for index, action in enumerate(self.defender_actions)
        }
        output_size = len(self.attacker_actions) * len(self.defender_actions)
        input_size = len(self._encode_state(initial_state))
        self.network = NumpyQNetwork(
            hidden_size=config.hidden_size,
            input_size=input_size,
            output_size=output_size,
            rng=self.rng,
            state_scale=config.state_scale,
        )
        self.target_network = self.network.copy()
        self.replay: list[tuple[Hashable, int, int, float, Hashable]] = []
        self._model_target_cache: dict[tuple[int, Hashable, int, int], float] = {}

    def train(self) -> NNQResult:
        state = self.env.reset(seed=self.config.seed)
        metrics: list[dict[str, Any]] = []
        last_loss = 0.0

        for step in range(1, self.config.total_steps + 1):
            state = self._maybe_exploring_start(state)
            attacker_action, defender_action = self._behavior_actions(state)
            next_state, cost, _info = self.env.step(attacker_action, defender_action)
            self._append_replay(state, attacker_action, defender_action, float(cost), next_state)

            if len(self.replay) >= self.config.batch_size:
                last_loss = self._train_one_batch()

            if step % self.config.target_update_interval == 0:
                self.target_network = self.network.copy()
                self._model_target_cache.clear()

            if step == 1 or step % self.config.log_interval == 0 or step == self.config.total_steps:
                metrics.append(
                    {
                        "step": step,
                        "state": _json_state(state),
                        "attacker_action": attacker_action,
                        "defender_action": defender_action,
                        "next_state": _json_state(next_state),
                        "cost": float(cost),
                        "loss": float(last_loss),
                        "q_norm": float(
                            np.linalg.norm(self.network.predict(self._encode_state(state)))
                        ),
                        "replay_size": len(self.replay),
                    }
                )
            state = next_state

        return NNQResult(network=self.network.copy(), metrics=metrics, final_state=state)

    def q_matrix(self, state: Hashable) -> np.ndarray:
        return self.network.q_matrix(
            self._encode_state(state),
            len(self.attacker_actions),
            len(self.defender_actions),
        )

    def _behavior_actions(self, state: Hashable) -> tuple[int, int]:
        if self.rng.random() < self.config.epsilon:
            attacker = int(self.rng.choice(self.attacker_actions))
            defender = int(self.rng.choice(self.defender_actions))
        else:
            game = solve_zero_sum_matrix_game(self.q_matrix(state))
            attacker = int(self.rng.choice(self.attacker_actions, p=game["attacker_strategy"]))
            defender = int(self.rng.choice(self.defender_actions, p=game["defender_strategy"]))
        defender = self._maybe_force_defender_action(defender)
        return attacker, defender

    def _maybe_force_defender_action(self, defender_action: int) -> int:
        probability = self.config.forced_defender_action_probability
        if probability <= 0.0:
            return defender_action
        if probability > 1.0:
            raise ValueError("forced_defender_action_probability must be in [0, 1]")
        if self.config.forced_defender_action is None:
            raise ValueError(
                "forced_defender_action is required when defender action forcing is enabled"
            )
        forced_action = int(self.config.forced_defender_action)
        if forced_action not in self.defender_actions:
            raise ValueError("forced_defender_action must be a valid defender action")
        if self.rng.random() < probability:
            return forced_action
        return defender_action

    def _append_replay(
        self,
        state: Hashable,
        attacker_action: int,
        defender_action: int,
        cost: float,
        next_state: Hashable,
    ) -> None:
        self.replay.append((state, attacker_action, defender_action, cost, next_state))
        if len(self.replay) > self.config.replay_capacity:
            self.replay.pop(0)

    def _train_one_batch(self) -> float:
        indices = self.rng.choice(len(self.replay), size=self.config.batch_size, replace=False)
        batch = [self.replay[int(index)] for index in indices]
        if self.config.backup_mode in {"full_action", "full_action_centered"}:
            return self._train_full_action_batch(
                batch,
                center_targets_by_state=self.config.backup_mode == "full_action_centered",
            )
        if self.config.backup_mode != "sampled":
            raise ValueError(
                "NNQ backup_mode must be 'sampled', 'full_action', or "
                "'full_action_centered'"
            )
        encoded_states = np.vstack([self._encode_state(item[0]) for item in batch])
        action_indices = np.array(
            [
                self._attacker_action_to_index[item[1]] * len(self.defender_actions)
                + self._defender_action_to_index[item[2]]
                for item in batch
            ],
            dtype=int,
        )
        targets = np.array([self._target(item[3], item[4]) for item in batch], dtype=float)
        return self.network.train_batch(
            encoded_states,
            action_indices,
            targets,
            self.config.learning_rate,
        )

    def _train_full_action_batch(
        self,
        batch: list[tuple[Hashable, int, int, float, Hashable]],
        center_targets_by_state: bool = False,
    ) -> float:
        encoded_states = []
        action_indices = []
        targets = []
        seen_states = []
        for state, _attacker_action, _defender_action, _cost, _next_state in batch:
            if state in seen_states:
                continue
            seen_states.append(state)
            for attacker_action in self.attacker_actions:
                for defender_action in self.defender_actions:
                    encoded_states.append(self._encode_state(state))
                    action_indices.append(
                        self._attacker_action_to_index[attacker_action]
                        * len(self.defender_actions)
                        + self._defender_action_to_index[defender_action]
                    )
                    targets.append(
                        self._model_target(state, attacker_action, defender_action)
                    )
            if center_targets_by_state:
                state_targets = targets[-len(self.attacker_actions) * len(self.defender_actions) :]
                target_center = float(np.mean(state_targets))
                for offset in range(1, len(state_targets) + 1):
                    targets[-offset] = targets[-offset] - target_center
        return self.network.train_batch(
            np.vstack(encoded_states),
            np.array(action_indices, dtype=int),
            np.array(targets, dtype=float),
            self.config.learning_rate,
        )

    def _target(self, cost: float, next_state: Hashable) -> float:
        next_matrix = self.target_network.q_matrix(
            self._encode_state(next_state),
            len(self.attacker_actions),
            len(self.defender_actions),
        )
        next_value = solve_zero_sum_matrix_game(next_matrix)["value"]
        return float(cost + self.env.discount * next_value)

    def _model_target(
        self,
        state: Hashable,
        attacker_action: int,
        defender_action: int,
    ) -> float:
        cache_key = (id(self.target_network), state, int(attacker_action), int(defender_action))
        if cache_key in self._model_target_cache:
            return self._model_target_cache[cache_key]
        expected_next_value = 0.0
        for next_state, probability in self.env.transition_probabilities(
            state, attacker_action, defender_action
        ).items():
            next_matrix = self.target_network.q_matrix(
                self._encode_state(next_state),
                len(self.attacker_actions),
                len(self.defender_actions),
            )
            expected_next_value += probability * solve_zero_sum_matrix_game(next_matrix)[
                "value"
            ]
        target = float(
            self.env.cost(state, attacker_action, defender_action)
            + self.env.discount * expected_next_value
        )
        self._model_target_cache[cache_key] = target
        return target

    def _encode_state(self, state: Hashable) -> np.ndarray:
        if self.config.state_feature_set == "env":
            return np.asarray(self.env.encode_state(state), dtype=float)
        if self.config.state_feature_set == "routing_augmented":
            if not isinstance(self.env, RoutingEnv):
                raise ValueError("routing_augmented state features require RoutingEnv")
            x = tuple(float(value) for value in self.env.encode_state(state))
            mu = tuple(float(value) for value in self.env.config.mu_rates)
            min_value = min(x)
            max_value = max(x)
            return np.asarray(
                [
                    *x,
                    *mu,
                    *(queue / service_rate for queue, service_rate in zip(x, mu)),
                    sum(x),
                    min_value,
                    max_value,
                    max_value - min_value,
                    *(1.0 if value == min_value else 0.0 for value in x),
                    *(1.0 if value == max_value else 0.0 for value in x),
                ],
                dtype=float,
            )
        if self.config.state_feature_set == "routing_tabular":
            if not isinstance(self.env, RoutingEnv):
                raise ValueError("routing_tabular state features require RoutingEnv")
            bound = int(self.env.config.bvi_max_queue_length)
            base = bound + 1
            values = tuple(
                min(max(int(value), 0), bound)
                for value in self.env.encode_state(state)
            )
            index = 0
            for value in values:
                index = index * base + value
            features = np.zeros(base ** self.env.config.num_queues, dtype=float)
            features[index] = 1.0
            return features
        if self.config.state_feature_set == "polling_augmented":
            if not isinstance(self.env, PollingEnv):
                raise ValueError("polling_augmented state features require PollingEnv")
            values = tuple(float(value) for value in self.env.encode_state(state))
            queues = values[:-1]
            position = int(values[-1])
            min_value = min(queues)
            max_value = max(queues)
            position_one_hot = [
                1.0 if index == position else 0.0
                for index in range(self.env.config.num_queues)
            ]
            return np.asarray(
                [
                    *queues,
                    *position_one_hot,
                    sum(queues),
                    min_value,
                    max_value,
                    max_value - min_value,
                    queues[position],
                    *(1.0 if value == min_value else 0.0 for value in queues),
                    *(1.0 if value == max_value else 0.0 for value in queues),
                ],
                dtype=float,
            )
        if self.config.state_feature_set == "polling_structural":
            if not isinstance(self.env, PollingEnv):
                raise ValueError("polling_structural state features require PollingEnv")
            values = tuple(float(value) for value in self.env.encode_state(state))
            queues = values[:-1]
            position = int(values[-1])
            min_value = min(queues)
            max_value = max(queues)
            gap = max_value - min_value
            position_one_hot = [
                1.0 if index == position else 0.0
                for index in range(self.env.config.num_queues)
            ]
            queue_zero = [1.0 if value == 0.0 else 0.0 for value in queues]
            queue_positive = [1.0 if value > 0.0 else 0.0 for value in queues]
            return np.asarray(
                [
                    *queues,
                    *position_one_hot,
                    sum(queues),
                    min_value,
                    max_value,
                    gap,
                    1.0 if min_value == 0.0 else 0.0,
                    1.0 if max_value == 0.0 else 0.0,
                    1.0 if gap == 0.0 else 0.0,
                    1.0 if min_value == 0.0 and max_value > 0.0 else 0.0,
                    queues[position],
                    *queue_zero,
                    *queue_positive,
                    *(1.0 if value == min_value else 0.0 for value in queues),
                    *(1.0 if value == max_value else 0.0 for value in queues),
                ],
                dtype=float,
            )
        if self.config.state_feature_set == "service_rate_augmented":
            if not isinstance(self.env, ServiceRateControlEnv):
                raise ValueError(
                    "service_rate_augmented state features require ServiceRateControlEnv"
                )
            x = float(self.env.encode_state(state)[0])
            return np.asarray(
                [
                    x,
                    x * x,
                    np.sqrt(x + 1.0),
                    1.0 if x == 0.0 else 0.0,
                    1.0 if x <= 1.0 else 0.0,
                    1.0 if x >= 2.0 else 0.0,
                    1.0 if x >= 5.0 else 0.0,
                    1.0 if x >= 10.0 else 0.0,
                ],
                dtype=float,
            )
        raise ValueError(f"unknown NNQ state_feature_set: {self.config.state_feature_set}")

    def _maybe_exploring_start(self, state: Hashable) -> Hashable:
        probability = self.config.exploring_starts_probability
        if probability <= 0.0:
            return state
        if probability > 1.0:
            raise ValueError("exploring_starts_probability must be in [0, 1]")
        if self.rng.random() >= probability:
            return state
        if self.config.exploring_starts_max_queue_length is None:
            raise ValueError(
                "exploring_starts_max_queue_length is required when exploring starts are enabled"
            )
        if isinstance(self.env, RoutingEnv):
            bound = int(self.config.exploring_starts_max_queue_length)
            if bound < 0:
                raise ValueError("exploring_starts_max_queue_length must be nonnegative")
            sampled = tuple(
                int(self.rng.integers(0, bound + 1))
                for _ in range(self.env.config.num_queues)
            )
            return self.env.set_state(sampled)
        if isinstance(self.env, PollingEnv):
            bound = int(self.config.exploring_starts_max_queue_length)
            if bound < 0:
                raise ValueError("exploring_starts_max_queue_length must be nonnegative")
            sampled_queues = tuple(
                int(self.rng.integers(0, bound + 1))
                for _ in range(self.env.config.num_queues)
            )
            sampled_position = int(self.rng.integers(0, self.env.config.num_queues))
            return self.env.set_state((*sampled_queues, sampled_position))
        raise ValueError("exploring starts are supported only for routing and polling NNQ")


def _json_state(state: Hashable) -> int | list[int]:
    if isinstance(state, tuple):
        return [int(value) for value in state]
    return int(state)
