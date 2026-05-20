#!/usr/bin/env python3
"""Polling convergence-speed experiment for AMQ2 vs fitted/full-action NNQ.

The polling DQN/NNQ side intentionally follows the policy-consistency artifact:
`NNQTrainer` with `backup_mode=full_action`, `polling_augmented` features, and
three queues.  The AMQ side uses the paper AMQ2 feature template adapted to the
polling target semantics of the local benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3] / "minimax_queueing_experiments"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adversarial_queueing.algorithms.minimax_solver import solve_zero_sum_matrix_game  # noqa: E402
from adversarial_queueing.algorithms.nnq import NNQConfig, NNQTrainer  # noqa: E402
from adversarial_queueing.envs.polling import PollingConfig, PollingEnv, State  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    Path("/Users/zheqihu/research/minimax_queueing_results_report")
    / "covergence_speed_final"
    / "results"
    / "polling_smoke"
)


@dataclass(frozen=True)
class PollingAMQConfig:
    seed: int = 0
    feature_set: str = "amq2"
    eta0: float = 1e-7
    decay_power: float = 0.6
    initial_state_sampling: str = "random_grid"
    exploring_starts_max_queue_length: int = 30


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("checkpoint list cannot be empty")
    if sorted(set(values)) != values:
        raise argparse.ArgumentTypeError("checkpoints must be sorted unique integers")
    if values[0] <= 0:
        raise argparse.ArgumentTypeError("checkpoints must be positive")
    return values


def default_polling_config(max_queue_length: int = 30) -> PollingConfig:
    return PollingConfig(
        lambda_arrivals=(1.0, 1.25, 1.5),
        mu_service=2.0,
        gamma=0.95,
        attack_cost=0.5,
        defend_cost=0.2,
        switch_cost=0.3,
        queue_cost="sum",
        initial_queues=(0, 0, 0),
        initial_position=0,
        uniformization_rate=6.0,
        bvi_max_queue_length=max_queue_length,
    )


def default_nnq_config(seed: int, total_steps: int) -> NNQConfig:
    return NNQConfig(
        hidden_size=64,
        learning_rate=0.001,
        total_steps=total_steps,
        batch_size=32,
        replay_capacity=4096,
        target_update_interval=200,
        epsilon=0.15,
        seed=seed,
        log_interval=max(1, total_steps),
        state_scale=30.0,
        state_feature_set="polling_augmented",
        exploring_starts_probability=0.5,
        exploring_starts_max_queue_length=30,
        backup_mode="full_action",
    )


def split_state(state: State) -> tuple[tuple[int, ...], int]:
    return tuple(int(value) for value in state[:-1]), int(state[-1])


def paper_behavior_probabilities(state: State) -> tuple[np.ndarray, np.ndarray]:
    queues, _position = split_state(state)
    norm = float(sum(queues))
    exp_term = float(np.exp(-norm / 2.0))
    attacker = np.asarray([1.0 - exp_term, exp_term], dtype=float)
    if sum(queues) == 0:
        defender = np.asarray([0.5, 0.5], dtype=float)
    else:
        defender = np.asarray([exp_term, 1.0 - exp_term], dtype=float)
    return attacker, defender


def polling_delta_indices(env: PollingEnv, state: State, attacker_action: int, defender_action: int) -> tuple[int, ...]:
    return tuple(int(index) for index in env.polling_targets(state, attacker_action, defender_action))


def polling_amq_features(
    env: PollingEnv,
    state: State,
    attacker_action: int,
    defender_action: int,
    feature_set: str,
) -> np.ndarray:
    queues, position = split_state(state)
    targets = set(polling_delta_indices(env, state, attacker_action, defender_action))
    blocks: list[float] = []
    for index, queue_length in enumerate(queues):
        adjusted = float(queue_length + (1.0 if index in targets else 0.0))
        same_position = 1.0 if index == position else 0.0
        if feature_set == "amq1":
            blocks.extend([1.0, adjusted, float(attacker_action), float(defender_action), same_position])
        elif feature_set == "amq2":
            blocks.extend(
                [
                    1.0,
                    adjusted,
                    adjusted * adjusted,
                    float(attacker_action),
                    float(defender_action),
                    same_position,
                ]
            )
        else:
            raise ValueError("feature_set must be 'amq1' or 'amq2'")
    return np.asarray(blocks, dtype=float)


class PollingAMQLearner:
    def __init__(self, env: PollingEnv, config: PollingAMQConfig):
        self.env = env
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        dim = polling_amq_features(env, env.config.initial_state_value, 0, 0, config.feature_set).size
        self.weights = np.zeros(dim, dtype=float)

    def q_matrix(self, state: State) -> np.ndarray:
        matrix = np.zeros((2, 2), dtype=float)
        for attacker_action in (0, 1):
            for defender_action in (0, 1):
                matrix[attacker_action, defender_action] = float(
                    polling_amq_features(
                        self.env,
                        state,
                        attacker_action,
                        defender_action,
                        self.config.feature_set,
                    )
                    @ self.weights
                )
        return matrix

    def value(self, state: State) -> float:
        return float(solve_zero_sum_matrix_game(self.q_matrix(state))["value"])

    def train_to_checkpoints(self, checkpoints: Iterable[int]) -> dict[int, dict[str, Any]]:
        checkpoints = tuple(checkpoints)
        checkpoint_set = set(checkpoints)
        max_step = max(checkpoints)
        snapshots: dict[int, dict[str, Any]] = {}
        state = self._sample_initial_state()
        self.env.reset_to_state(state, seed=self.config.seed)
        td_errors: list[float] = []
        started = time.perf_counter()
        for step in range(1, max_step + 1):
            attacker_policy, defender_policy = paper_behavior_probabilities(state)
            attacker_action = int(self.rng.choice((0, 1), p=attacker_policy))
            defender_action = int(self.rng.choice((0, 1), p=defender_policy))
            next_state, cost, _info = self.env.step(attacker_action, defender_action)
            phi = polling_amq_features(
                self.env,
                state,
                attacker_action,
                defender_action,
                self.config.feature_set,
            )
            current_q = float(phi @ self.weights)
            next_value = self.value(next_state)
            td_error = float(cost + self.env.discount * next_value - current_q)
            eta = float(self.config.eta0 / (step**self.config.decay_power))
            self.weights = self.weights + eta * phi * td_error
            td_errors.append(abs(td_error))
            state = next_state
            if step in checkpoint_set:
                snapshots[step] = {
                    "weights": self.weights.copy(),
                    "elapsed_seconds": time.perf_counter() - started,
                    "mean_abs_td_error_recent": float(np.mean(td_errors[-100:])) if td_errors else 0.0,
                    "weight_norm": float(np.linalg.norm(self.weights)),
                }
        return snapshots

    def _sample_initial_state(self) -> State:
        if self.config.initial_state_sampling == "zero":
            return self.env.config.initial_state_value
        if self.config.initial_state_sampling == "random_grid":
            bound = int(self.config.exploring_starts_max_queue_length)
            queues = tuple(int(self.rng.integers(0, bound + 1)) for _ in range(self.env.config.num_queues))
            position = int(self.rng.integers(0, self.env.config.num_queues))
            return (*queues, position)
        raise ValueError("initial_state_sampling must be 'zero' or 'random_grid'")


def train_nnq_checkpoints(env: PollingEnv, config: NNQConfig, checkpoints: Iterable[int]) -> dict[int, dict[str, Any]]:
    checkpoints = tuple(checkpoints)
    checkpoint_set = set(checkpoints)
    max_step = max(checkpoints)
    trainer = NNQTrainer(env, config)
    state = trainer.env.reset(seed=config.seed)
    snapshots: dict[int, dict[str, Any]] = {}
    losses: list[float] = []
    started = time.perf_counter()
    cumulative_work = 0
    for step in range(1, max_step + 1):
        state = trainer._maybe_exploring_start(state)
        attacker_action, defender_action = trainer._behavior_actions(state)
        next_state, cost, _info = trainer.env.step(attacker_action, defender_action)
        trainer._append_replay(state, attacker_action, defender_action, float(cost), next_state)
        if len(trainer.replay) >= config.batch_size:
            loss, work_increment = train_one_batch_with_work_counter(trainer)
            losses.append(float(loss))
            cumulative_work += int(work_increment)
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


def train_one_batch_with_work_counter(trainer: NNQTrainer) -> tuple[float, int]:
    """Train one NNQ batch and count exact effective target evaluations.

    For the polling convergence protocol the finalized DQN uses full-action
    backups.  One batch evaluates every attacker/defender action pair for each
    distinct sampled state; duplicate states in the replay batch are intentionally
    counted once, matching NNQTrainer._train_full_action_batch.
    """

    indices = trainer.rng.choice(len(trainer.replay), size=trainer.config.batch_size, replace=False)
    batch = [trainer.replay[int(index)] for index in indices]
    if trainer.config.backup_mode in {"full_action", "full_action_centered"}:
        seen_states = []
        for state, _attacker_action, _defender_action, _cost, _next_state in batch:
            if state in seen_states:
                continue
            seen_states.append(state)
        work_increment = len(seen_states) * len(trainer.attacker_actions) * len(trainer.defender_actions)
        loss = trainer._train_full_action_batch(
            batch,
            center_targets_by_state=trainer.config.backup_mode == "full_action_centered",
        )
        return float(loss), int(work_increment)
    if trainer.config.backup_mode != "sampled":
        raise ValueError(
            "NNQ backup_mode must be 'sampled', 'full_action', or 'full_action_centered'"
        )
    return float(trainer._train_one_batch()), int(trainer.config.batch_size)


def evaluation_states(env: PollingEnv, max_queue_length: int, limit: int | None) -> list[State]:
    states = [
        (*queues, position)
        for queues in np.ndindex((max_queue_length + 1,) * env.config.num_queues)
        for position in range(env.config.num_queues)
    ]
    states = [tuple(int(value) for value in state) for state in states]
    if limit is None or limit >= len(states):
        return states
    rng = np.random.default_rng(20260520)
    anchors = {
        (0, 0, 0, 0),
        (max_queue_length, max_queue_length, max_queue_length, 0),
        (0, max_queue_length // 2, max_queue_length, 1),
        (max_queue_length, 0, max_queue_length // 2, 2),
    }
    anchors = {state for state in anchors if len(state) == env.config.num_queues + 1}
    remaining = [state for state in states if state not in anchors]
    sampled_count = max(0, limit - len(anchors))
    sampled_indices = rng.choice(len(remaining), size=sampled_count, replace=False)
    return sorted(anchors) + [remaining[int(index)] for index in sampled_indices]


def policy_from_matrix(matrix: np.ndarray) -> dict[str, Any]:
    game = solve_zero_sum_matrix_game(matrix)
    attacker = np.asarray(game["attacker_strategy"], dtype=float)
    defender = np.asarray(game["defender_strategy"], dtype=float)
    return {
        "p_attack": float(attacker[1]),
        "p_defend": float(defender[1]),
        "attacker_action": int(np.argmax(attacker)),
        "defender_action": int(np.argmax(defender)),
        "value": float(game["value"]),
    }


def amq_policy_rows(env: PollingEnv, config: PollingAMQConfig, snapshots: dict[int, dict[str, Any]], states: list[State]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    learner = PollingAMQLearner(env, config)
    for checkpoint, snapshot in snapshots.items():
        learner.weights = np.asarray(snapshot["weights"], dtype=float).copy()
        for state in states:
            rows.append(
                {
                    "method": "amq",
                    "checkpoint": checkpoint,
                    "state": list(state),
                    **policy_from_matrix(learner.q_matrix(state)),
                }
            )
    return rows


def nnq_policy_rows(env: PollingEnv, config: NNQConfig, snapshots: dict[int, dict[str, Any]], states: list[State]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    trainer = NNQTrainer(env, config)
    for checkpoint, snapshot in snapshots.items():
        trainer.network = snapshot["network"]
        for state in states:
            rows.append(
                {
                    "method": "dqn",
                    "checkpoint": checkpoint,
                    "state": list(state),
                    **policy_from_matrix(trainer.q_matrix(state)),
                }
            )
    return rows


def summarize_stabilization(rows: list[dict[str, Any]], checkpoints: list[int], epsilon: float) -> dict[str, Any]:
    final_checkpoint = checkpoints[-1]
    by_checkpoint = {checkpoint: [] for checkpoint in checkpoints}
    final_by_state: dict[tuple[int, ...], dict[str, Any]] = {}
    for row in rows:
        state = tuple(int(value) for value in row["state"])
        if int(row["checkpoint"]) == final_checkpoint:
            final_by_state[state] = row
    for row in rows:
        checkpoint = int(row["checkpoint"])
        state = tuple(int(value) for value in row["state"])
        final = final_by_state[state]
        by_checkpoint[checkpoint].append(
            {
                "attacker_gap": abs(float(row["p_attack"]) - float(final["p_attack"])),
                "defender_gap": abs(float(row["p_defend"]) - float(final["p_defend"])),
                "attacker_action_agree": int(row["attacker_action"] == final["attacker_action"]),
                "defender_action_agree": int(row["defender_action"] == final["defender_action"]),
                "joint_action_agree": int(
                    row["attacker_action"] == final["attacker_action"]
                    and row["defender_action"] == final["defender_action"]
                ),
            }
        )
    checkpoint_summaries = []
    for checkpoint in checkpoints:
        entries = by_checkpoint[checkpoint]
        attacker_gap = float(np.mean([entry["attacker_gap"] for entry in entries]))
        defender_gap = float(np.mean([entry["defender_gap"] for entry in entries]))
        joint_gap = 0.5 * attacker_gap + 0.5 * defender_gap
        checkpoint_summaries.append(
            {
                "checkpoint": checkpoint,
                "attacker_gap": attacker_gap,
                "defender_gap": defender_gap,
                "joint_gap": joint_gap,
                "policy_similarity_percent": float((1.0 - joint_gap) * 100.0),
                "attacker_action_agreement_percent": float(
                    np.mean([entry["attacker_action_agree"] for entry in entries]) * 100.0
                ),
                "defender_action_agreement_percent": float(
                    np.mean([entry["defender_action_agree"] for entry in entries]) * 100.0
                ),
                "joint_action_agreement_percent": float(
                    np.mean([entry["joint_action_agree"] for entry in entries]) * 100.0
                ),
            }
        )
    stable_checkpoint = None
    for index, item in enumerate(checkpoint_summaries):
        if all(later["joint_gap"] <= epsilon for later in checkpoint_summaries[index:]):
            stable_checkpoint = int(item["checkpoint"])
            break
    return {
        "epsilon": epsilon,
        "final_checkpoint": final_checkpoint,
        "stable_checkpoint": stable_checkpoint,
        "stable_before_horizon": stable_checkpoint is not None and stable_checkpoint < final_checkpoint,
        "censored_at_horizon": stable_checkpoint is None or stable_checkpoint == final_checkpoint,
        "not_stabilized_within_horizon": stable_checkpoint is None,
        "checkpoints": checkpoint_summaries,
    }


def strip_snapshot_payloads(snapshots: dict[int, dict[str, Any]], method: str, batch_size: int = 32) -> dict[str, Any]:
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
            row["parameter_norm"] = float(
                np.sqrt(
                    np.sum(network.w1 * network.w1)
                    + np.sum(network.b1 * network.b1)
                    + np.sum(network.w2 * network.w2)
                    + np.sum(network.b2 * network.b2)
                )
            )
            row["work_accounting_unit"] = "distinct_state_full_action_model_targets"
            row["work_to_checkpoint"] = int(snapshot.get("cumulative_work", 0))
            row["primary_work_to_checkpoint"] = int(snapshot.get("cumulative_work", 0))
            row["effective_target_updates"] = int(snapshot.get("cumulative_work", 0))
            row["legacy_upper_bound_target_updates"] = int(snapshot["num_gradient_updates"]) * int(batch_size) * 4
        out[str(checkpoint)] = row
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-queue-length", type=int, default=30)
    parser.add_argument("--eval-state-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--amq-eta0", type=float, default=1e-7)
    parser.add_argument("--amq-decay-power", type=float, default=0.6)
    parser.add_argument("--amq-checkpoints", type=parse_int_list, default=parse_int_list("1,2,5,10,20,50,100,200,500,1000,2000,5000,10000"))
    parser.add_argument("--dqn-checkpoints", type=parse_int_list, default=parse_int_list("100,200,500,1000,2000,5000,10000,20000,50000"))
    args = parser.parse_args()

    env_config = default_polling_config(args.max_queue_length)
    amq_env = PollingEnv(env_config)
    dqn_env = PollingEnv(env_config)
    amq_config = PollingAMQConfig(
        seed=args.seed,
        eta0=args.amq_eta0,
        decay_power=args.amq_decay_power,
        exploring_starts_max_queue_length=args.max_queue_length,
    )
    nnq_config = default_nnq_config(args.seed, total_steps=max(args.dqn_checkpoints))
    states = evaluation_states(amq_env, args.max_queue_length, args.eval_state_limit)
    full_grid_size = (args.max_queue_length + 1) ** amq_env.config.num_queues * amq_env.config.num_queues

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    amq_snapshots = PollingAMQLearner(amq_env, amq_config).train_to_checkpoints(args.amq_checkpoints)
    dqn_snapshots = train_nnq_checkpoints(dqn_env, nnq_config, args.dqn_checkpoints)
    amq_rows = amq_policy_rows(PollingEnv(env_config), amq_config, amq_snapshots, states)
    dqn_rows = nnq_policy_rows(PollingEnv(env_config), nnq_config, dqn_snapshots, states)
    amq_summary = summarize_stabilization(amq_rows, args.amq_checkpoints, args.epsilon)
    dqn_summary = summarize_stabilization(dqn_rows, args.dqn_checkpoints, args.epsilon)
    write_jsonl(output_dir / "polling_amq_policy_rows.jsonl", amq_rows)
    write_jsonl(output_dir / "polling_dqn_policy_rows.jsonl", dqn_rows)
    summary = {
        "benchmark": "polling",
        "status": "smoke" if len(states) < full_grid_size else "experiment",
        "num_eval_states": len(states),
        "full_grid_size": full_grid_size,
        "env_config": asdict(env_config),
        "amq_config": asdict(amq_config),
        "dqn_config": asdict(nnq_config),
        "note": (
            "Polling DQN follows the policy-consistency artifact: NNQTrainer, "
            "backup_mode=full_action, polling_augmented features, 50k max steps."
        ),
        "amq_snapshots": strip_snapshot_payloads(amq_snapshots, "amq"),
        "dqn_snapshots": strip_snapshot_payloads(dqn_snapshots, "dqn", batch_size=nnq_config.batch_size),
        "amq_stabilization": amq_summary,
        "dqn_stabilization": dqn_summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
