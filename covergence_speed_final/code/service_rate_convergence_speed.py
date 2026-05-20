#!/usr/bin/env python3
"""Service-rate-control convergence speed for AMQ-extension vs fitted NNQ."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_CODE_ROOT = REPO_ROOT / "policy_consistency_final" / "code"
POLICY_SRC_ROOT = POLICY_CODE_ROOT / "src"
LEGACY_PROJECT_ROOT = Path(__file__).resolve().parents[3] / "minimax_queueing_experiments"
for candidate in (POLICY_CODE_ROOT, POLICY_SRC_ROOT, LEGACY_PROJECT_ROOT):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from adversarial_queueing.algorithms.minimax_solver import solve_zero_sum_matrix_game  # noqa: E402
from adversarial_queueing.algorithms.nnq import NNQConfig, NNQTrainer  # noqa: E402
from adversarial_queueing.envs.service_rate_control import (  # noqa: E402
    ServiceRateControlConfig,
    ServiceRateControlEnv,
)


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "service_rate_smoke"
)


@dataclass(frozen=True)
class ServiceAMQConfig:
    seed: int = 0
    eta0: float = 1e-4
    decay_power: float = 0.6
    initial_state_sampling: str = "random_grid"
    exploring_starts_max_queue_length: int = 20


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("checkpoint list cannot be empty")
    if sorted(set(values)) != values:
        raise argparse.ArgumentTypeError("checkpoints must be sorted unique integers")
    return values


def default_env_config(max_queue_length: int = 20) -> ServiceRateControlConfig:
    return ServiceRateControlConfig(
        gamma=0.95,
        lambda_arrival=2.0,
        mu_levels=(1.0, 3.0, 5.0),
        service_costs=(0.0, 0.2, 3.0),
        attack_cost=0.5,
        q_congestion=0.2,
        initial_state=0,
        uniformization_rate=7.0,
        robust_defender_actions=(2,),
        bvi_max_queue_length=max_queue_length,
    )


def default_nnq_config(seed: int, total_steps: int) -> NNQConfig:
    return NNQConfig(
        hidden_size=32,
        learning_rate=0.0005,
        total_steps=total_steps,
        batch_size=64,
        replay_capacity=10000,
        target_update_interval=250,
        epsilon=0.1,
        seed=seed,
        log_interval=max(1, total_steps),
        state_scale=10.0,
        state_feature_set="service_rate_augmented",
        backup_mode="sampled",
    )


def behavior_probabilities(state: int) -> tuple[np.ndarray, np.ndarray]:
    exp_term = float(np.exp(-float(state) / 2.0))
    attacker = np.asarray([1.0 - exp_term, exp_term], dtype=float)
    if state == 0:
        defender = np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)
    else:
        # Keep all defender actions explored while biasing toward robust high service.
        defender = np.asarray([0.5 * exp_term, 0.5 * exp_term, 1.0 - exp_term], dtype=float)
        defender = defender / defender.sum()
    return attacker, defender


def service_amq_features(state: int, attacker_action: int, defender_action: int, num_defender: int = 3) -> np.ndarray:
    x = float(state)
    attacker = np.zeros(2, dtype=float)
    attacker[int(attacker_action)] = 1.0
    defender = np.zeros(num_defender, dtype=float)
    defender[int(defender_action)] = 1.0
    state_basis = np.asarray([1.0, x, x * x], dtype=float)
    return np.concatenate(
        [
            state_basis,
            attacker,
            defender,
            x * defender,
            x * x * defender,
            np.outer(attacker, defender).ravel(),
        ]
    )


class ServiceAMQLearner:
    def __init__(self, env: ServiceRateControlEnv, config: ServiceAMQConfig):
        self.env = env
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.weights = np.zeros(service_amq_features(0, 0, 0, len(env.defender_actions(0))).size)

    def q_matrix(self, state: int) -> np.ndarray:
        defender_actions = self.env.defender_actions(state)
        matrix = np.zeros((2, len(defender_actions)), dtype=float)
        for attacker_action in (0, 1):
            for defender_action in defender_actions:
                matrix[attacker_action, defender_action] = float(
                    service_amq_features(state, attacker_action, defender_action, len(defender_actions))
                    @ self.weights
                )
        return matrix

    def value(self, state: int) -> float:
        return float(solve_zero_sum_matrix_game(self.q_matrix(state))["value"])

    def train_to_checkpoints(self, checkpoints: Iterable[int]) -> dict[int, dict[str, Any]]:
        checkpoints = tuple(checkpoints)
        checkpoint_set = set(checkpoints)
        max_step = max(checkpoints)
        snapshots: dict[int, dict[str, Any]] = {}
        state = self._sample_initial_state()
        self.env.reset(seed=self.config.seed)
        self.env._state = state
        td_errors: list[float] = []
        started = time.perf_counter()
        for step in range(1, max_step + 1):
            attacker_policy, defender_policy = behavior_probabilities(state)
            attacker_action = int(self.rng.choice((0, 1), p=attacker_policy))
            defender_action = int(self.rng.choice(self.env.defender_actions(state), p=defender_policy))
            next_state, cost, _info = self.env.step(attacker_action, defender_action)
            phi = service_amq_features(state, attacker_action, defender_action, len(self.env.defender_actions(state)))
            current_q = float(phi @ self.weights)
            td_error = float(cost + self.env.discount * self.value(next_state) - current_q)
            eta = float(self.config.eta0 / (step**self.config.decay_power))
            self.weights = self.weights + eta * phi * td_error
            td_errors.append(abs(td_error))
            state = int(next_state)
            if step in checkpoint_set:
                snapshots[step] = {
                    "weights": self.weights.copy(),
                    "elapsed_seconds": time.perf_counter() - started,
                    "mean_abs_td_error_recent": float(np.mean(td_errors[-100:])) if td_errors else 0.0,
                    "weight_norm": float(np.linalg.norm(self.weights)),
                }
        return snapshots

    def _sample_initial_state(self) -> int:
        if self.config.initial_state_sampling == "zero":
            return 0
        if self.config.initial_state_sampling == "random_grid":
            return int(self.rng.integers(0, self.config.exploring_starts_max_queue_length + 1))
        raise ValueError("initial_state_sampling must be 'zero' or 'random_grid'")


def train_nnq_checkpoints(env: ServiceRateControlEnv, config: NNQConfig, checkpoints: Iterable[int]) -> dict[int, dict[str, Any]]:
    checkpoints = tuple(checkpoints)
    checkpoint_set = set(checkpoints)
    trainer = NNQTrainer(env, config)
    state = trainer.env.reset(seed=config.seed)
    snapshots: dict[int, dict[str, Any]] = {}
    losses: list[float] = []
    started = time.perf_counter()
    cumulative_work = 0
    for step in range(1, max(checkpoints) + 1):
        attacker_action, defender_action = trainer._behavior_actions(state)
        next_state, cost, _info = trainer.env.step(attacker_action, defender_action)
        trainer._append_replay(state, attacker_action, defender_action, float(cost), next_state)
        if len(trainer.replay) >= config.batch_size:
            losses.append(float(trainer._train_one_batch()))
            cumulative_work += int(config.batch_size)
        if step % config.target_update_interval == 0:
            trainer.target_network = trainer.network.copy()
            trainer._model_target_cache.clear()
        state = next_state
        if step in checkpoint_set:
            snapshots[step] = {
                "network": trainer.network.copy(),
                "elapsed_seconds": time.perf_counter() - started,
                "loss_mean_recent": float(np.mean(losses[-100:])) if losses else 0.0,
                "replay_size": len(trainer.replay),
                "num_gradient_updates": len(losses),
                "cumulative_work": cumulative_work,
            }
    return snapshots


def policy_from_matrix(matrix: np.ndarray) -> dict[str, Any]:
    game = solve_zero_sum_matrix_game(matrix)
    attacker = np.asarray(game["attacker_strategy"], dtype=float)
    defender = np.asarray(game["defender_strategy"], dtype=float)
    return {
        "p_attack": float(attacker[1]),
        "defender_strategy": [float(x) for x in defender],
        "attacker_action": int(np.argmax(attacker)),
        "defender_action": int(np.argmax(defender)),
        "value": float(game["value"]),
    }


def policy_rows(method: str, env: ServiceRateControlEnv, config: Any, snapshots: dict[int, dict[str, Any]], states: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if method == "amq":
        learner = ServiceAMQLearner(env, config)
        for checkpoint, snapshot in snapshots.items():
            learner.weights = np.asarray(snapshot["weights"], dtype=float).copy()
            for state in states:
                rows.append({"method": method, "checkpoint": checkpoint, "state": state, **policy_from_matrix(learner.q_matrix(state))})
    else:
        trainer = NNQTrainer(env, config)
        for checkpoint, snapshot in snapshots.items():
            trainer.network = snapshot["network"]
            for state in states:
                rows.append({"method": method, "checkpoint": checkpoint, "state": state, **policy_from_matrix(trainer.q_matrix(state))})
    return rows


def summarize(rows: list[dict[str, Any]], checkpoints: list[int], epsilon: float) -> dict[str, Any]:
    final_checkpoint = checkpoints[-1]
    final_by_state = {int(row["state"]): row for row in rows if int(row["checkpoint"]) == final_checkpoint}
    summaries = []
    for checkpoint in checkpoints:
        checkpoint_rows = [row for row in rows if int(row["checkpoint"]) == checkpoint]
        attacker_gaps = []
        defender_gaps = []
        joint_agree = []
        for row in checkpoint_rows:
            final = final_by_state[int(row["state"])]
            attacker_gaps.append(abs(float(row["p_attack"]) - float(final["p_attack"])))
            defender_gaps.append(
                0.5
                * sum(
                    abs(float(left) - float(right))
                    for left, right in zip(row["defender_strategy"], final["defender_strategy"])
                )
            )
            joint_agree.append(
                int(
                    row["attacker_action"] == final["attacker_action"]
                    and row["defender_action"] == final["defender_action"]
                )
            )
        attacker_gap = float(np.mean(attacker_gaps))
        defender_gap = float(np.mean(defender_gaps))
        joint_gap = 0.5 * attacker_gap + 0.5 * defender_gap
        summaries.append(
            {
                "checkpoint": checkpoint,
                "attacker_gap": attacker_gap,
                "defender_gap": defender_gap,
                "joint_gap": joint_gap,
                "policy_similarity_percent": float((1.0 - joint_gap) * 100.0),
                "joint_action_agreement_percent": float(np.mean(joint_agree) * 100.0),
            }
        )
    stable = None
    for index, item in enumerate(summaries):
        if all(later["joint_gap"] <= epsilon for later in summaries[index:]):
            stable = int(item["checkpoint"])
            break
    return {
        "epsilon": epsilon,
        "final_checkpoint": final_checkpoint,
        "stable_checkpoint": stable,
        "stable_before_horizon": stable is not None and stable < final_checkpoint,
        "censored_at_horizon": stable is None or stable == final_checkpoint,
        "not_stabilized_within_horizon": stable is None,
        "checkpoints": summaries,
    }


def strip_snapshots(snapshots: dict[int, dict[str, Any]], method: str, batch_size: int = 64) -> dict[str, Any]:
    out = {}
    for checkpoint, snapshot in snapshots.items():
        row = {key: value for key, value in snapshot.items() if key not in {"weights", "network"}}
        if method == "amq":
            row["parameter_type"] = "linear_weights"
            row["num_parameters"] = int(np.asarray(snapshot["weights"]).size)
            row["weight_norm"] = float(np.linalg.norm(snapshot["weights"]))
            row["work_accounting_unit"] = "online_td_target"
            row["work_to_checkpoint"] = int(checkpoint)
            row["primary_work_to_checkpoint"] = int(checkpoint)
            row["effective_target_updates"] = int(checkpoint)
        else:
            network = snapshot["network"]
            row["parameter_type"] = "neural_network"
            row["num_parameters"] = int(network.w1.size + network.b1.size + network.w2.size + network.b2.size)
            row["parameter_norm"] = float(np.sqrt(np.sum(network.w1 * network.w1) + np.sum(network.b1 * network.b1) + np.sum(network.w2 * network.w2) + np.sum(network.b2 * network.b2)))
            row["work_accounting_unit"] = "sampled_td_target"
            row["work_to_checkpoint"] = int(snapshot.get("cumulative_work", int(snapshot["num_gradient_updates"]) * int(batch_size)))
            row["primary_work_to_checkpoint"] = int(snapshot.get("cumulative_work", int(snapshot["num_gradient_updates"]) * int(batch_size)))
            row["effective_target_updates"] = int(snapshot.get("cumulative_work", int(snapshot["num_gradient_updates"]) * int(batch_size)))
        out[str(checkpoint)] = row
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-queue-length", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--amq-eta0", type=float, default=1e-4)
    parser.add_argument("--amq-checkpoints", type=parse_int_list, default=parse_int_list("100,300,1000,3000,10000"))
    parser.add_argument("--dqn-checkpoints", type=parse_int_list, default=parse_int_list("100,300,1000,3000,10000"))
    args = parser.parse_args()

    env_config = default_env_config(args.max_queue_length)
    states = list(range(args.max_queue_length + 1))
    amq_config = ServiceAMQConfig(seed=args.seed, eta0=args.amq_eta0, exploring_starts_max_queue_length=args.max_queue_length)
    nnq_config = default_nnq_config(args.seed, max(args.dqn_checkpoints))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    amq_snapshots = ServiceAMQLearner(ServiceRateControlEnv(env_config), amq_config).train_to_checkpoints(args.amq_checkpoints)
    dqn_snapshots = train_nnq_checkpoints(ServiceRateControlEnv(env_config), nnq_config, args.dqn_checkpoints)
    amq_rows = policy_rows("amq", ServiceRateControlEnv(env_config), amq_config, amq_snapshots, states)
    dqn_rows = policy_rows("dqn", ServiceRateControlEnv(env_config), nnq_config, dqn_snapshots, states)
    write_jsonl(output_dir / "service_rate_amq_policy_rows.jsonl", amq_rows)
    write_jsonl(output_dir / "service_rate_dqn_policy_rows.jsonl", dqn_rows)
    summary = {
        "benchmark": "service-rate-control",
        "status": "experiment",
        "num_eval_states": len(states),
        "env_config": asdict(env_config),
        "amq_config": asdict(amq_config),
        "dqn_config": asdict(nnq_config),
        "note": "Service-rate-control is an extension benchmark for AMQ; it is not in the original AMQ paper.",
        "amq_snapshots": strip_snapshots(amq_snapshots, "amq"),
        "dqn_snapshots": strip_snapshots(dqn_snapshots, "dqn", nnq_config.batch_size),
        "amq_stabilization": summarize(amq_rows, args.amq_checkpoints, args.epsilon),
        "dqn_stabilization": summarize(dqn_rows, args.dqn_checkpoints, args.epsilon),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
