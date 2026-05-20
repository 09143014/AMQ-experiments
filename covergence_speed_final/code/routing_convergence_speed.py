#!/usr/bin/env python3
"""Routing convergence-speed experiment for AMQ vs fitted minimax-DQN.

This script is intentionally self-contained in the final report directory so
the convergence-speed pipeline can be audited independently from older probes.

AMQ follows the paper-form update:

    Q_w(x,a,b) = phi(x,a,b)^T w
    Delta = r + value_w(x_next) - Q_w(x,a,b)
    w <- w + eta_k phi(x,a,b) Delta

where value_w is obtained by solving the next-state 2x2 minimax game.  In this
uniformized routing benchmark the discount is represented by a terminal event;
therefore the continuation coefficient is 1 and terminal next states contribute
zero continuation value.

The DQN side is the fitted minimax-DQN used in the policy-consistency work:
the learned object is a neural 2x2 Q function, trained from Bellman/minimax
targets, without BVI policy labels or AMQ labels.
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

from experiments.source_faithful_routing_consistency.routing_bvi_dqn_consistency import (  # noqa: E402
    RoutingSecurityParams,
    State,
    TwoLayerDQN,
    all_states,
    dqn_matrix,
    encode_state,
    immediate_cost,
    matrix_game,
    sample_next_state,
    train_model_based_minimax_q,
)


DEFAULT_OUTPUT_DIR = (
    Path("/Users/zheqihu/research/minimax_queueing_results_report")
    / "covergence_speed_final"
    / "results"
    / "routing_smoke"
)


@dataclass(frozen=True)
class AMQPaperConfig:
    seed: int = 0
    feature_set: str = "amq2"
    eta0: float = 0.05
    decay_power: float = 0.6
    reward_scale: float = 1.0
    initial_state_sampling: str = "random_grid"


@dataclass(frozen=True)
class FittedDQNConfig:
    seed: int = 0
    hidden_size: int = 256
    architecture: str = "standard"
    feature_set: str = "structural"
    batch_size: int = 1024
    learning_rate: float = 0.0007
    reward_scale: str = "denominator"
    loss_type: str = "mse"
    huber_delta: float = 1.0
    grad_clip_norm: float | None = 20.0
    fixed_point_tolerance: float = 1e-8
    fixed_point_max_iterations: int = 2000
    center_targets: bool = True


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("checkpoint list cannot be empty")
    if sorted(set(values)) != values:
        raise argparse.ArgumentTypeError("checkpoints must be sorted unique integers")
    if values[0] <= 0:
        raise argparse.ArgumentTypeError("checkpoints must be positive")
    return values


def amq_delta_index(state: State, attacker_action: int, defender_action: int) -> int:
    if attacker_action == 1 and defender_action == 0:
        return max(range(len(state)), key=lambda idx: (state[idx], -idx))
    return min(range(len(state)), key=lambda idx: (state[idx], idx))


def amq_features(state: State, attacker_action: int, defender_action: int, feature_set: str, bound: int) -> np.ndarray:
    """Paper AMQ1/AMQ2 routing features from Appendix A.1.

    For each server i, AMQ1 uses
        [1, x_i + delta_i(x,a,b), a, b].
    AMQ2 adds the second-order traffic term
        (x_i + delta_i(x,a,b))^2.

    The delta target follows the routing transition rule: if attack succeeds
    without defense, mark the longest queue; otherwise mark the shortest queue.
    """

    del bound
    q = np.asarray(state, dtype=float)
    target = amq_delta_index(state, attacker_action, defender_action)
    blocks: list[float] = []
    for index, queue_length in enumerate(q):
        adjusted = float(queue_length + (1.0 if index == target else 0.0))
        if feature_set == "amq1":
            blocks.extend([1.0, adjusted, float(attacker_action), float(defender_action)])
        elif feature_set == "amq2":
            blocks.extend(
                [
                    1.0,
                    adjusted,
                    adjusted * adjusted,
                    float(attacker_action),
                    float(defender_action),
                ]
            )
        else:
            raise ValueError("AMQ feature_set must be 'amq1' or 'amq2'")
    return np.asarray(blocks, dtype=float)


def paper_behavior_probabilities(state: State) -> tuple[np.ndarray, np.ndarray]:
    """Return paper-style behavior policies alpha(a|x), beta(b|x)."""

    norm = float(sum(state))
    exp_term = float(np.exp(-norm / 2.0))
    attacker = np.asarray([1.0 - exp_term, exp_term], dtype=float)
    if sum(state) == 0:
        defender = np.asarray([0.5, 0.5], dtype=float)
    else:
        defender = np.asarray([exp_term, 1.0 - exp_term], dtype=float)
    return attacker, defender


class AMQPaperLearner:
    def __init__(self, params: RoutingSecurityParams, config: AMQPaperConfig):
        self.params = params
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        dim = amq_features(
            (0,) * params.num_queues,
            0,
            0,
            config.feature_set,
            params.bound,
        ).size
        self.weights = np.zeros(dim, dtype=float)

    def q_matrix(self, state: State) -> np.ndarray:
        matrix = np.zeros((2, 2), dtype=float)
        for attacker_action in (0, 1):
            for defender_action in (0, 1):
                matrix[attacker_action, defender_action] = float(
                    amq_features(
                        state,
                        attacker_action,
                        defender_action,
                        self.config.feature_set,
                        self.params.bound,
                    )
                    @ self.weights
                )
        return matrix

    def value(self, state: State | None) -> float:
        if state is None:
            return 0.0
        return float(matrix_game(self.q_matrix(state))["value"])

    def train_to_checkpoints(self, checkpoints: Iterable[int]) -> dict[int, dict[str, Any]]:
        checkpoints = tuple(checkpoints)
        max_step = max(checkpoints)
        checkpoint_set = set(checkpoints)
        snapshots: dict[int, dict[str, Any]] = {}
        state = self._sample_initial_state()
        td_errors: list[float] = []
        started = time.perf_counter()
        for step in range(1, max_step + 1):
            attacker_policy, defender_policy = paper_behavior_probabilities(state)
            attacker_action = int(self.rng.choice((0, 1), p=attacker_policy))
            defender_action = int(self.rng.choice((0, 1), p=defender_policy))
            cost = (
                immediate_cost(state, attacker_action, defender_action, self.params)
                / self.config.reward_scale
            )
            next_state = sample_next_state(state, attacker_action, defender_action, self.params, self.rng)
            phi = amq_features(
                state,
                attacker_action,
                defender_action,
                self.config.feature_set,
                self.params.bound,
            )
            current_q = float(phi @ self.weights)
            next_value = self.value(next_state)
            td_error = float(cost + next_value - current_q)
            eta = float(self.config.eta0 / (step**self.config.decay_power))
            self.weights = self.weights + eta * phi * td_error
            td_errors.append(abs(td_error))
            if next_state is None:
                state = self._sample_initial_state()
            else:
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
            return (0,) * self.params.num_queues
        if self.config.initial_state_sampling == "random_grid":
            return tuple(int(self.rng.integers(0, self.params.bound + 1)) for _ in range(self.params.num_queues))
        raise ValueError("initial_state_sampling must be 'zero' or 'random_grid'")


def train_fixed_point_dqn_checkpoints(
    params: RoutingSecurityParams,
    config: FittedDQNConfig,
    checkpoints: Iterable[int],
) -> dict[int, dict[str, Any]]:
    checkpoints = tuple(checkpoints)
    max_epoch = max(checkpoints)
    checkpoint_set = set(checkpoints)
    rng = np.random.default_rng(config.seed)
    reward_scale_value = params.denominator if config.reward_scale == "denominator" else 1.0
    fixed_started = time.perf_counter()
    fixed_point = train_model_based_minimax_q(
        params,
        reward_scale=reward_scale_value,
        tolerance=config.fixed_point_tolerance,
        max_iterations=config.fixed_point_max_iterations,
    )
    fixed_elapsed = time.perf_counter() - fixed_started
    probe_features = encode_state((0,) * params.num_queues, params.bound, config.feature_set)
    network = TwoLayerDQN(
        rng,
        hidden_size=config.hidden_size,
        input_size=int(probe_features.size),
        architecture=config.architecture,
    )
    states = all_states(params.bound, params.num_queues)
    state_features = np.vstack(
        [encode_state(state, params.bound, config.feature_set) for state in states]
    )
    targets = np.vstack([fixed_point["q_values"][state].reshape(4) for state in states])
    if config.center_targets:
        targets = targets - targets.mean(axis=1, keepdims=True)
    snapshots: dict[int, dict[str, Any]] = {}
    losses: list[float] = []
    fit_errors: list[float] = []
    started = time.perf_counter()
    num_updates = 0

    for epoch in range(1, max_epoch + 1):
        order = rng.permutation(len(states))
        for start in range(0, len(states), config.batch_size):
            indices = order[start : start + config.batch_size]
            loss, abs_errors = network.train_full_batch(
                state_features[indices],
                targets[indices],
                config.learning_rate,
                config.loss_type,
                config.huber_delta,
                config.grad_clip_norm,
            )
            losses.append(loss)
            fit_errors.append(float(abs_errors.mean()))
            num_updates += 1
        if epoch in checkpoint_set:
            snapshots[epoch] = {
                "network": network.copy(),
                "elapsed_seconds": fixed_elapsed + time.perf_counter() - started,
                "loss_mean_recent": float(np.mean(losses[-100:])) if losses else 0.0,
                "target_fit_error_recent": float(np.mean(fit_errors[-100:])) if fit_errors else 0.0,
                "fixed_point_iterations": int(fixed_point["iterations"]),
                "fixed_point_residual": float(fixed_point["residual"]),
                "fixed_point_elapsed_seconds": fixed_elapsed,
                "num_gradient_updates": num_updates,
            }
    return snapshots


def evaluation_states(params: RoutingSecurityParams, limit: int | None = None) -> list[State]:
    states = all_states(params.bound, params.num_queues)
    if limit is None or limit >= len(states):
        return states
    rng = np.random.default_rng(20260519)
    anchors = {
        (0, 0, 0),
        (params.bound // 4, params.bound // 4, params.bound // 4),
        (params.bound // 2, params.bound // 2, params.bound // 2),
        (params.bound, params.bound, params.bound),
        (0, params.bound // 2, params.bound),
        (params.bound, 0, params.bound // 2),
    }
    anchors = {state for state in anchors if len(state) == params.num_queues}
    remaining = [state for state in states if state not in anchors]
    sampled_count = max(0, limit - len(anchors))
    sampled_indices = rng.choice(len(remaining), size=sampled_count, replace=False)
    sampled = [remaining[int(index)] for index in sampled_indices]
    return sorted(anchors) + sampled


def policy_from_matrix(matrix: np.ndarray) -> dict[str, Any]:
    game = matrix_game(matrix)
    attacker = np.asarray(game["attacker_strategy"], dtype=float)
    defender = np.asarray(game["defender_strategy"], dtype=float)
    return {
        "p_attack": float(attacker[1]),
        "p_defend": float(defender[1]),
        "attacker_action": int(np.argmax(attacker)),
        "defender_action": int(np.argmax(defender)),
        "value": float(game["value"]),
    }


def amq_policy_rows(
    params: RoutingSecurityParams,
    config: AMQPaperConfig,
    snapshots: dict[int, dict[str, Any]],
    states: list[State],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    learner = AMQPaperLearner(params, config)
    for checkpoint, snapshot in snapshots.items():
        learner.weights = np.asarray(snapshot["weights"], dtype=float).copy()
        for state in states:
            policy = policy_from_matrix(learner.q_matrix(state))
            rows.append({"method": "amq", "checkpoint": checkpoint, "state": list(state), **policy})
    return rows


def dqn_policy_rows(
    params: RoutingSecurityParams,
    config: FittedDQNConfig,
    snapshots: dict[int, dict[str, Any]],
    states: list[State],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint, snapshot in snapshots.items():
        network = snapshot["network"]
        for state in states:
            policy = policy_from_matrix(dqn_matrix(network, state, params.bound, config.feature_set))
            rows.append({"method": "dqn", "checkpoint": checkpoint, "state": list(state), **policy})
    return rows


def summarize_stabilization(rows: list[dict[str, Any]], checkpoints: list[int], epsilon: float) -> dict[str, Any]:
    final_checkpoint = checkpoints[-1]
    by_checkpoint = {checkpoint: [] for checkpoint in checkpoints}
    final_by_state: dict[tuple[int, ...], dict[str, Any]] = {}
    for row in rows:
        state = tuple(int(value) for value in row["state"])
        if row["checkpoint"] == final_checkpoint:
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
    stable_checkpoint: int | None = None
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


def strip_snapshot_payloads(
    snapshots: dict[int, dict[str, Any]],
    method: str,
    *,
    num_states: int,
) -> dict[str, Any]:
    serializable = {}
    for checkpoint, snapshot in snapshots.items():
        row = {
            key: value
            for key, value in snapshot.items()
            if key not in {"weights", "network"}
        }
        if method == "amq":
            row["parameter_type"] = "linear_weights"
            row["num_parameters"] = int(np.asarray(snapshot["weights"]).size)
            row["weight_norm"] = float(np.linalg.norm(snapshot["weights"]))
            row["work_accounting_unit"] = "online_td_target"
            row["work_to_checkpoint"] = int(checkpoint)
            row["primary_work_to_checkpoint"] = int(checkpoint)
            row["effective_bellman_backups"] = int(checkpoint)
        else:
            network = snapshot["network"]
            params = network.params()
            row["parameter_type"] = "neural_network"
            row["architecture"] = network.architecture
            row["num_parameters"] = int(sum(value.size for value in params.values()))
            row["parameter_norm"] = float(
                np.sqrt(sum(float(np.sum(value * value)) for value in params.values()))
            )
            fixed_point_backups = int(snapshot.get("fixed_point_iterations", 0)) * int(num_states) * 4
            fitting_target_entries = int(checkpoint) * int(num_states) * 4
            row["work_accounting_unit"] = "model_based_fixed_point_bellman_backup"
            row["fixed_point_bellman_backups"] = fixed_point_backups
            row["work_to_checkpoint"] = fixed_point_backups
            row["primary_work_to_checkpoint"] = fixed_point_backups
            row["secondary_fitting_target_entries"] = fitting_target_entries
            row["secondary_gradient_updates"] = int(snapshot.get("num_gradient_updates", 0))
            # Backward-compatible alias.  The primary report uses work_to_checkpoint;
            # neural fitting entries are reported separately because they are MSE
            # fitting work, not additional Bellman backups.
            row["effective_bellman_backups"] = fixed_point_backups
        serializable[str(checkpoint)] = row
    return serializable


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bound", type=int, default=20)
    parser.add_argument("--num-queues", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-state-limit", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--amq-checkpoints", type=parse_int_list, default=parse_int_list("1,2,5,10,20,50,100,200,500,1000,2000,5000"))
    parser.add_argument("--dqn-checkpoints", type=parse_int_list, default=parse_int_list("1,2,5,10,20,50,100,200,500,1000,1500"))
    parser.add_argument("--dqn-hidden-size", type=int, default=256)
    parser.add_argument("--amq-feature-set", choices=("amq1", "amq2"), default="amq2")
    parser.add_argument("--amq-eta0", type=float, default=1e-6)
    parser.add_argument("--amq-decay-power", type=float, default=0.6)
    parser.add_argument("--dqn-feature-set", choices=("raw", "structural", "poly2"), default="structural")
    args = parser.parse_args()

    params = RoutingSecurityParams(bound=args.bound, num_queues=args.num_queues)
    amq_config = AMQPaperConfig(
        seed=args.seed,
        feature_set=args.amq_feature_set,
        eta0=args.amq_eta0,
        decay_power=args.amq_decay_power,
    )
    dqn_config = FittedDQNConfig(
        seed=args.seed,
        hidden_size=args.dqn_hidden_size,
        feature_set=args.dqn_feature_set,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    states = evaluation_states(params, args.eval_state_limit)

    amq = AMQPaperLearner(params, amq_config)
    amq_snapshots = amq.train_to_checkpoints(args.amq_checkpoints)
    dqn_snapshots = train_fixed_point_dqn_checkpoints(params, dqn_config, args.dqn_checkpoints)

    amq_rows = amq_policy_rows(params, amq_config, amq_snapshots, states)
    dqn_rows = dqn_policy_rows(params, dqn_config, dqn_snapshots, states)
    amq_summary = summarize_stabilization(amq_rows, args.amq_checkpoints, args.epsilon)
    dqn_summary = summarize_stabilization(dqn_rows, args.dqn_checkpoints, args.epsilon)

    write_jsonl(output_dir / "routing_amq_policy_rows.jsonl", amq_rows)
    write_jsonl(output_dir / "routing_dqn_policy_rows.jsonl", dqn_rows)
    summary = {
        "benchmark": "routing",
        "status": "smoke" if max(args.dqn_checkpoints) <= 5 else "experiment",
        "note": (
            "AMQ checkpoints count sampled online AMQ updates. Routing DQN checkpoints "
            "count neural fitting epochs after a model-based minimax-Q fixed point. "
            "Cross-method speed is compared with work_to_checkpoint, not wall-clock "
            "runtime or raw checkpoint index."
        ),
        "params": asdict(params),
        "num_eval_states": len(states),
        "amq_config": asdict(amq_config),
        "dqn_config": asdict(dqn_config),
        "amq_snapshots": strip_snapshot_payloads(
            amq_snapshots,
            "amq",
            num_states=len(all_states(params.bound, params.num_queues)),
        ),
        "dqn_snapshots": strip_snapshot_payloads(
            dqn_snapshots,
            "dqn",
            num_states=len(all_states(params.bound, params.num_queues)),
        ),
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
