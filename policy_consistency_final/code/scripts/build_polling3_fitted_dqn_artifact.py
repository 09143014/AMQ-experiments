#!/usr/bin/env python3
"""Fit a polling minimax-DQN to model-based Bellman-Q fixed-point targets."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ROUTING_SOURCE = ROOT / "experiments" / "source_faithful_routing_consistency"
for path in (SRC, ROUTING_SOURCE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from adversarial_queueing.algorithms.minimax_solver import solve_zero_sum_matrix_game
from adversarial_queueing.utils.config import load_config
from routing_bvi_dqn_consistency import TwoLayerDQN


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def summarize_policy(p_defend: np.ndarray, p_attack: np.ndarray) -> dict:
    defend = np.asarray(p_defend, dtype=float)
    attack = np.asarray(p_attack, dtype=float)
    return {
        "num_policy_states": int(defend.size),
        "defend_probability_mean": float(defend.mean()),
        "defend_probability_max": float(defend.max()),
        "attack_probability_mean": float(attack.mean()),
        "attack_probability_max": float(attack.max()),
        "num_states_p_defend_at_least_threshold": int(np.sum(defend >= 0.5)),
    }


def solve_two_by_two_grid(payoff: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized 2x2 ``min_defender max_attacker`` solve.

    Returns value, p_attack, p_defend arrays over the trailing state grid.
    """

    a = payoff[0, 0]
    b = payoff[0, 1]
    c = payoff[1, 0]
    d = payoff[1, 1]

    value_x1 = np.maximum(a, c)
    value_x0 = np.maximum(b, d)
    denom = (a - b) - (c - d)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_int = (d - b) / denom
    valid = (np.abs(denom) > 1e-12) & (x_int >= 0.0) & (x_int <= 1.0)
    row0_int = b + (a - b) * x_int
    row1_int = d + (c - d) * x_int
    value_int = np.where(valid, np.maximum(row0_int, row1_int), np.inf)

    # Tie convention matches minimax_solver.py: prefer no-defend, then defend,
    # then the mixed intersection only if it strictly improves the value.
    p_no_defend = np.ones_like(a, dtype=float)
    value = value_x1.copy()
    use_x0 = value_x0 < value - 1e-12
    p_no_defend = np.where(use_x0, 0.0, p_no_defend)
    value = np.where(use_x0, value_x0, value)
    use_int = value_int < value - 1e-12
    p_no_defend = np.where(use_int, x_int, p_no_defend)
    value = np.where(use_int, value_int, value)

    col_value_y1 = np.minimum(a, b)
    col_value_y0 = np.minimum(c, d)
    denom_attacker = (a - c) - (b - d)
    with np.errstate(divide="ignore", invalid="ignore"):
        y_int = (d - c) / denom_attacker
    valid_y = (
        (np.abs(denom_attacker) > 1e-12)
        & (y_int >= 0.0)
        & (y_int <= 1.0)
    )
    col0_int = c + (a - c) * y_int
    col1_int = d + (b - d) * y_int
    attacker_value_int = np.where(valid_y, np.minimum(col0_int, col1_int), -np.inf)

    p_no_attack = np.ones_like(a, dtype=float)
    attacker_value = col_value_y1.copy()
    use_y0 = col_value_y0 > attacker_value + 1e-12
    p_no_attack = np.where(use_y0, 0.0, p_no_attack)
    attacker_value = np.where(use_y0, col_value_y0, attacker_value)
    use_yint = attacker_value_int > attacker_value + 1e-12
    p_no_attack = np.where(use_yint, y_int, p_no_attack)

    return value, 1.0 - p_no_attack, 1.0 - p_no_defend


def expected_next_values(
    values: np.ndarray,
    queues: list[np.ndarray],
    positions: np.ndarray,
    rates: np.ndarray,
    mu: float,
    rate: float,
    *,
    target_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if target_mode not in {"min", "max"}:
        raise ValueError("target_mode must be 'min' or 'max'")

    max_q = values.shape[0] - 1
    q0, q1, q2 = queues
    q_stack = np.stack(queues, axis=0)
    target_value = q_stack.min(axis=0) if target_mode == "min" else q_stack.max(axis=0)
    target_masks = [queue == target_value for queue in queues]
    target_count = np.maximum(
        np.sum(np.stack(target_masks, axis=0), axis=0).astype(float),
        1.0,
    )
    arrival_total = float(np.sum(rates))
    self_prob = max(0.0, 1.0 - (arrival_total + mu) / rate)

    expected3 = np.zeros_like(q0, dtype=float)
    switch_probability = np.zeros((*q0.shape, 3), dtype=float)

    for target, mask in enumerate(target_masks):
        weight = mask.astype(float) / target_count
        target_expected = self_prob * values[q0, q1, q2, target]

        for index, arrival_rate in enumerate(rates):
            next_queues = [q0, q1, q2]
            next_queues[index] = np.minimum(next_queues[index] + 1, max_q)
            target_expected = target_expected + (arrival_rate / rate) * values[
                next_queues[0],
                next_queues[1],
                next_queues[2],
                target,
            ]

        service_queues = [q0, q1, q2]
        service_queues[target] = np.maximum(service_queues[target] - 1, 0)
        target_expected = target_expected + (mu / rate) * values[
            service_queues[0],
            service_queues[1],
            service_queues[2],
            target,
        ]

        expected3 = expected3 + weight * target_expected
        switch_probability = switch_probability + weight[..., None] * (
            positions != target
        ).astype(float)

    return expected3[..., None] + np.zeros_like(positions, dtype=float), switch_probability


def run_vectorized_bvi(
    *,
    max_q: int,
    rates: np.ndarray,
    mu: float,
    rate: float,
    gamma: float,
    attack_cost: float,
    defend_cost: float,
    switch_cost: float,
    tolerance: float,
    max_iterations: int,
    progress_interval: int,
) -> dict:
    values = np.zeros((max_q + 1, max_q + 1, max_q + 1, 3), dtype=float)
    residual = float("inf")
    p_attack = np.zeros_like(values)
    p_defend = np.zeros_like(values)
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        payoff = q_targets_from_values(
            values,
            max_q=max_q,
            rates=rates,
            mu=mu,
            rate=rate,
            gamma=gamma,
            attack_cost=attack_cost,
            defend_cost=defend_cost,
            switch_cost=switch_cost,
        )
        new_values, p_attack, p_defend = solve_two_by_two_grid(payoff)
        residual = float(np.max(np.abs(new_values - values)))
        values = new_values
        if iteration == 1 or iteration % progress_interval == 0 or residual <= tolerance:
            print(f"bvi_iter={iteration} residual={residual:.6g}", flush=True)
        if residual <= tolerance:
            break
    return {
        "values": values,
        "p_attack": p_attack,
        "p_defend": p_defend,
        "iterations": iteration,
        "residual": residual,
    }


def write_policy_inspection(path: Path, bvi_result: dict, *, max_q: int) -> None:
    rows = []
    values = bvi_result["values"]
    p_attack = bvi_result["p_attack"]
    p_defend = bvi_result["p_defend"]
    for q0 in range(max_q + 1):
        for q1 in range(max_q + 1):
            for q2 in range(max_q + 1):
                queues = (q0, q1, q2)
                for position in range(3):
                    rows.append(
                        {
                            "method": "bvi",
                            "state": [q0, q1, q2, position],
                            "queues": [q0, q1, q2],
                            "position": position,
                            "total_queue": int(q0 + q1 + q2),
                            "queue_gap": int(max(queues) - min(queues)),
                            "nominal_targets": list(targets(queues, "max")),
                            "attacked_targets": list(targets(queues, "min")),
                            "p_no_defend": float(1.0 - p_defend[q0, q1, q2, position]),
                            "p_defend": float(p_defend[q0, q1, q2, position]),
                            "p_no_attack": float(1.0 - p_attack[q0, q1, q2, position]),
                            "p_attack": float(p_attack[q0, q1, q2, position]),
                            "value": float(values[q0, q1, q2, position]),
                        }
                    )
    write_jsonl(path, rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvi-config",
        default=str(ROOT / "configs" / "polling3_bvi_policy_probe_max30.yaml"),
    )
    parser.add_argument(
        "--artifact-root",
        default=str(ROOT / "artifacts" / "polling3"),
    )
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--architecture", choices=["standard", "dueling"], default="standard")
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--center-targets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--feature-set",
        choices=[
            "polling_structural_v1",
            "polling_structural_v2",
            "polling_structural_v3",
            "polling_coordinate_onehot_v1",
            "polling_pairwise_diff_onehot_v1",
            "polling_bucket_onehot_v1",
        ],
        default="polling_structural_v1",
    )
    parser.add_argument("--mixed-state-weight", type=float, default=0.0)
    parser.add_argument("--margin-state-weight", type=float, default=0.0)
    parser.add_argument("--contrast-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--last-layer-refit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After MLP training, refit the final linear layer by least squares "
            "against the Bellman-Q targets. This still fits Q targets, not "
            "BVI policy labels."
        ),
    )
    parser.add_argument("--last-layer-ridge", type=float, default=1e-8)
    parser.add_argument("--progress-interval", type=int, default=100)
    args = parser.parse_args()

    artifact_root = Path(args.artifact_root)
    bvi_dir = artifact_root / "bvi"
    dqn_dir = artifact_root / "dqn"
    artifact_root.mkdir(parents=True, exist_ok=True)
    bvi_dir.mkdir(parents=True, exist_ok=True)
    dqn_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.bvi_config)
    config = load_config(config_path)
    shutil.copy2(config_path, bvi_dir / "config.yaml")
    shutil.copy2(config_path, dqn_dir / "source_env_config.yaml")

    env_cfg = config["env"]
    bvi_cfg = config["bvi"]
    rates = np.asarray(env_cfg["lambda_arrivals"], dtype=float)
    max_q = int(bvi_cfg["max_queue_length"])
    mu = float(env_cfg["mu_service"])
    rate = float(env_cfg.get("uniformization_rate") or (rates.sum() + mu))
    gamma = float(env_cfg.get("gamma", 0.95))
    attack_cost = float(env_cfg.get("attack_cost", 0.5))
    defend_cost = float(env_cfg.get("defend_cost", 0.2))
    switch_cost = float(env_cfg.get("switch_cost", 0.1))

    started = time.time()
    bvi_result = run_vectorized_bvi(
        max_q=max_q,
        rates=rates,
        mu=mu,
        rate=rate,
        gamma=gamma,
        attack_cost=attack_cost,
        defend_cost=defend_cost,
        switch_cost=switch_cost,
        tolerance=float(bvi_cfg.get("tolerance", 1e-6)),
        max_iterations=int(bvi_cfg.get("max_iterations", 1000)),
        progress_interval=args.progress_interval,
    )
    bvi_elapsed = time.time() - started
    write_policy_inspection(bvi_dir / "policy_inspection.jsonl", bvi_result, max_q=max_q)
    write_json(
        bvi_dir / "summary.json",
        {
            "algorithm": "bvi",
            "benchmark": "polling",
            "implementation": "vectorized_polling3_dual_attacker",
            "iterations": int(bvi_result["iterations"]),
            "residual": float(bvi_result["residual"]),
            "elapsed_seconds": bvi_elapsed,
            "max_queue_length": max_q,
            "num_states": int((max_q + 1) ** 3 * 3),
            "policy_inspection": summarize_policy(
                bvi_result["p_defend"],
                bvi_result["p_attack"],
            ),
        },
    )

    states, features = polling_feature_matrix(max_q, feature_set=args.feature_set)
    dqn_fixed_point = run_model_based_minimax_q(
        max_q=max_q,
        rates=rates,
        mu=mu,
        rate=rate,
        gamma=gamma,
        attack_cost=attack_cost,
        defend_cost=defend_cost,
        switch_cost=switch_cost,
        tolerance=float(bvi_cfg.get("tolerance", 1e-6)),
        max_iterations=int(bvi_cfg.get("max_iterations", 1000)),
        progress_interval=args.progress_interval,
    )
    q_targets = dqn_fixed_point["q_targets"]
    flat_targets = q_targets.reshape(4, -1).T
    if args.center_targets:
        flat_targets = flat_targets - flat_targets.mean(axis=1, keepdims=True)
    sample_weights = state_sampling_weights(
        q_targets,
        mixed_scale=args.mixed_state_weight,
        margin_scale=args.margin_state_weight,
    )

    fit_started = time.time()
    network, fit_metrics = fit_network(
        features,
        flat_targets,
        sample_weights=sample_weights,
        hidden_size=args.hidden_size,
        architecture=args.architecture,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        contrast_loss_weight=args.contrast_loss_weight,
        progress_interval=args.progress_interval,
    )
    fit_elapsed = time.time() - fit_started
    refit_metrics = None
    if args.last_layer_refit:
        refit_metrics = refit_last_layer(
            network,
            features,
            flat_targets,
            ridge=args.last_layer_ridge,
        )
    rows, policy_summary = dqn_policy_rows(states, network, features)
    write_jsonl(dqn_dir / "policy_inspection.jsonl", rows)
    save_network(
        dqn_dir / "dqn_params.npz",
        network,
        max_q=max_q,
        feature_set=args.feature_set,
    )
    write_jsonl(dqn_dir / "fit_metrics.jsonl", fit_metrics)
    write_json(
        dqn_dir / "summary.json",
        {
            "algorithm": "fitted_minimax_dqn",
            "benchmark": "polling",
            "implementation": "model_based_q_fixed_point_mlp",
            "target_source": "independent_model_based_minimax_q_fixed_point",
            "fixed_point_iterations": int(dqn_fixed_point["iterations"]),
            "fixed_point_residual": float(dqn_fixed_point["residual"]),
            "notes": (
                "DQN targets are produced by an independent model-based minimax-Q "
                "fixed-point solve. BVI values and BVI policies are not used as "
                "DQN training labels."
            ),
            "hidden_size": args.hidden_size,
            "architecture": args.architecture,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "feature_set": args.feature_set,
            "center_targets": bool(args.center_targets),
            "mixed_state_weight": args.mixed_state_weight,
            "margin_state_weight": args.margin_state_weight,
            "contrast_loss_weight": args.contrast_loss_weight,
            "last_layer_refit": bool(args.last_layer_refit),
            "last_layer_ridge": args.last_layer_ridge,
            "last_layer_refit_metrics": refit_metrics,
            "elapsed_seconds": fit_elapsed,
            "final_loss": fit_metrics[-1]["loss"] if fit_metrics else None,
            "final_abs_error_mean": (
                fit_metrics[-1]["eval_abs_error_mean"] if fit_metrics else None
            ),
            "policy_inspection": policy_summary,
            "params_file": "dqn_params.npz",
        },
    )
    print(f"wrote {artifact_root}")
    return 0


def run_model_based_minimax_q(
    *,
    max_q: int,
    rates: np.ndarray,
    mu: float,
    rate: float,
    gamma: float,
    attack_cost: float,
    defend_cost: float,
    switch_cost: float,
    tolerance: float,
    max_iterations: int,
    progress_interval: int,
) -> dict:
    values = np.zeros((max_q + 1, max_q + 1, max_q + 1, 3), dtype=float)
    q_targets = np.zeros((2, 2, max_q + 1, max_q + 1, max_q + 1, 3), dtype=float)
    residual = float("inf")
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        old_values = values.copy()
        q_targets = q_targets_from_values(
            old_values,
            max_q=max_q,
            rates=rates,
            mu=mu,
            rate=rate,
            gamma=gamma,
            attack_cost=attack_cost,
            defend_cost=defend_cost,
            switch_cost=switch_cost,
        )
        new_values, _p_attack, _p_defend = solve_two_by_two_grid(q_targets)
        residual = float(np.max(np.abs(new_values - old_values)))
        values = new_values
        if iteration == 1 or iteration % progress_interval == 0 or residual <= tolerance:
            print(f"dqn_fixed_point_iter={iteration} residual={residual:.6g}", flush=True)
        if residual <= tolerance:
            break
    q_targets = q_targets_from_values(
        values,
        max_q=max_q,
        rates=rates,
        mu=mu,
        rate=rate,
        gamma=gamma,
        attack_cost=attack_cost,
        defend_cost=defend_cost,
        switch_cost=switch_cost,
    )
    return {
        "q_targets": q_targets,
        "values": values,
        "iterations": iteration,
        "residual": residual,
    }


def q_targets_from_values(
    values: np.ndarray,
    *,
    max_q: int,
    rates: np.ndarray,
    mu: float,
    rate: float,
    gamma: float,
    attack_cost: float,
    defend_cost: float,
    switch_cost: float,
) -> np.ndarray:
    shape = (max_q + 1, max_q + 1, max_q + 1, 3)
    q0, q1, q2 = np.indices(shape[:3])
    queues = [q0, q1, q2]
    positions = np.arange(3, dtype=int)[None, None, None, :]
    beta = rate * (1.0 / gamma - 1.0)
    cost_denom = rate + beta
    payoff = np.empty((2, 2, *shape), dtype=float)
    for attacker_action in (0, 1):
        for defender_action in (0, 1):
            target_mode = "min" if attacker_action == 1 and defender_action == 0 else "max"
            expected_next, switch_probability = expected_next_values(
                values,
                queues,
                positions,
                rates,
                mu,
                rate,
                target_mode=target_mode,
            )
            immediate = (
                q0[..., None]
                + q1[..., None]
                + q2[..., None]
                + switch_cost * switch_probability
                - attack_cost * float(attacker_action)
                + defend_cost * float(defender_action)
            ) / cost_denom
            payoff[attacker_action, defender_action] = immediate + gamma * expected_next
    return payoff


def polling_feature_matrix(
    max_q: int,
    *,
    feature_set: str,
) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    states: list[tuple[int, int, int, int]] = []
    features = []
    scale = max(float(max_q), 1.0)
    for q0 in range(max_q + 1):
        for q1 in range(max_q + 1):
            for q2 in range(max_q + 1):
                for position in range(3):
                    state = (q0, q1, q2, position)
                    states.append(state)
                    q = np.asarray((q0, q1, q2), dtype=float) / scale
                    sorted_q = np.sort(q)
                    min_q = float(q.min())
                    max_value = float(q.max())
                    total = float(q.sum()) / 3.0
                    gap = max_value - min_q
                    min_mask = [1.0 if value == min_q else 0.0 for value in q]
                    max_mask = [1.0 if value == max_value else 0.0 for value in q]
                    position_one_hot = [1.0 if index == position else 0.0 for index in range(3)]
                    queue_at_position = float(q[position])
                    base = [
                        *q,
                        *position_one_hot,
                        total,
                        min_q,
                        max_value,
                        gap,
                        float(q.std()),
                        queue_at_position,
                        *sorted_q,
                        *min_mask,
                        *max_mask,
                        *(1.0 if value == 0.0 else 0.0 for value in q),
                    ]
                    if feature_set in {
                        "polling_structural_v2",
                        "polling_structural_v3",
                        "polling_coordinate_onehot_v1",
                        "polling_pairwise_diff_onehot_v1",
                        "polling_bucket_onehot_v1",
                    }:
                        min_count = float(sum(min_mask)) / 3.0
                        max_count = float(sum(max_mask)) / 3.0
                        pairwise_abs = [
                            abs(float(q[0] - q[1])),
                            abs(float(q[0] - q[2])),
                            abs(float(q[1] - q[2])),
                        ]
                        base.extend(
                            [
                                float(sum(position_one_hot[i] * min_mask[i] for i in range(3))),
                                float(sum(position_one_hot[i] * max_mask[i] for i in range(3))),
                                min_count,
                                max_count,
                                *pairwise_abs,
                            ]
                        )
                    if feature_set in {
                        "polling_structural_v3",
                        "polling_coordinate_onehot_v1",
                        "polling_pairwise_diff_onehot_v1",
                        "polling_bucket_onehot_v1",
                    }:
                        min_count_raw = max(float(sum(min_mask)), 1.0)
                        max_count_raw = max(float(sum(max_mask)), 1.0)
                        position_in_min = float(min_mask[position])
                        position_in_max = float(max_mask[position])
                        min_switch_probability = 1.0 - position_in_min / min_count_raw
                        max_switch_probability = 1.0 - position_in_max / max_count_raw
                        base.extend(
                            [
                                min_switch_probability,
                                max_switch_probability,
                                float(min_count_raw > 1.0),
                                float(max_count_raw > 1.0),
                                float(queue_at_position - min_q),
                                float(max_value - queue_at_position),
                            ]
                        )
                    if feature_set in {
                        "polling_coordinate_onehot_v1",
                        "polling_pairwise_diff_onehot_v1",
                        "polling_bucket_onehot_v1",
                    }:
                        for raw_value in (q0, q1, q2):
                            base.extend(
                                1.0 if raw_value == bucket else 0.0
                                for bucket in range(max_q + 1)
                            )
                        # Low-dimensional coordinate/position interactions. These are still
                        # feature engineering, not a full joint-state lookup table.
                        for index, value in enumerate(q):
                            base.append(value * position_one_hot[index])
                            base.append(value * min_mask[index])
                            base.append(value * max_mask[index])
                    if feature_set in {"polling_pairwise_diff_onehot_v1", "polling_bucket_onehot_v1"}:
                        for left, right in ((q0, q1), (q0, q2), (q1, q2)):
                            diff = int(left - right)
                            base.extend(
                                1.0 if diff == bucket else 0.0
                                for bucket in range(-max_q, max_q + 1)
                            )
                        base.extend(
                            [
                                float(q0 == q1),
                                float(q0 == q2),
                                float(q1 == q2),
                                float(q0 < q1),
                                float(q0 < q2),
                                float(q1 < q2),
                            ]
                        )
                    if feature_set == "polling_bucket_onehot_v1":
                        integer_buckets = [
                            (int(q0 + q1 + q2), range(0, 3 * max_q + 1)),
                            (int(min(q0, q1, q2)), range(0, max_q + 1)),
                            (int(max(q0, q1, q2)), range(0, max_q + 1)),
                            (int(max(q0, q1, q2) - min(q0, q1, q2)), range(0, max_q + 1)),
                            (int((q0, q1, q2)[position]), range(0, max_q + 1)),
                        ]
                        for value, buckets in integer_buckets:
                            base.extend(1.0 if value == bucket else 0.0 for bucket in buckets)
                    elif (
                        feature_set != "polling_structural_v1"
                        and feature_set != "polling_structural_v2"
                        and feature_set != "polling_structural_v3"
                        and feature_set != "polling_coordinate_onehot_v1"
                        and feature_set != "polling_pairwise_diff_onehot_v1"
                    ):
                        raise ValueError(f"unknown polling feature set: {feature_set}")
                    poly_seed = [total, min_q, max_value, gap, queue_at_position, float(q.std())]
                    base.extend(value * value for value in poly_seed)
                    base.extend(q[i] * q[j] for i in range(3) for j in range(i + 1, 3))
                    features.append(base)
    return states, np.asarray(features, dtype=float)


def fit_network(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    sample_weights: np.ndarray | None,
    hidden_size: int,
    architecture: str,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    seed: int,
    contrast_loss_weight: float,
    progress_interval: int,
) -> tuple[TwoLayerDQN, list[dict]]:
    rng = np.random.default_rng(seed)
    network = TwoLayerDQN(
        rng=rng,
        hidden_size=hidden_size,
        input_size=features.shape[1],
        output_size=4,
        architecture=architecture,
    )
    metrics = []
    num_states = features.shape[0]
    all_indices = np.arange(num_states)
    sampling_probabilities = None
    if sample_weights is not None:
        weights = np.asarray(sample_weights, dtype=float)
        sampling_probabilities = weights / weights.sum()
    for epoch in range(1, epochs + 1):
        batch_indices = rng.choice(
            all_indices,
            size=min(batch_size, num_states),
            replace=False,
            p=sampling_probabilities,
        )
        if contrast_loss_weight > 0.0:
            loss, abs_errors = train_matrix_batch_with_contrast(
                network,
                features[batch_indices],
                targets[batch_indices],
                learning_rate=learning_rate,
                contrast_loss_weight=contrast_loss_weight,
                grad_clip_norm=20.0,
            )
        else:
            loss, abs_errors = network.train_full_batch(
                features[batch_indices],
                targets[batch_indices],
                learning_rate=learning_rate,
                loss_type="mse",
                huber_delta=1.0,
                grad_clip_norm=20.0,
            )
        if epoch == 1 or epoch % progress_interval == 0 or epoch == epochs:
            eval_indices = rng.choice(all_indices, size=min(8192, num_states), replace=False)
            predictions = network.predict_batch(features[eval_indices])
            eval_abs = np.abs(predictions - targets[eval_indices])
            metric = {
                "epoch": epoch,
                "loss": float(loss),
                "batch_abs_error_mean": float(abs_errors.mean()),
                "eval_abs_error_mean": float(eval_abs.mean()),
                "eval_abs_error_max": float(eval_abs.max()),
            }
            print(
                "epoch={epoch} loss={loss:.6g} eval_mae={eval_abs_error_mean:.6g}".format(
                    **metric
                ),
                flush=True,
            )
            metrics.append(metric)
    return network, metrics


def refit_last_layer(
    network: TwoLayerDQN,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    ridge: float,
) -> dict:
    """Least-squares refit of the final linear Q head.

    This is a low-noise finishing step for fitted minimax-DQN: the hidden state
    representation remains learned by backpropagation, and the closed-form solve
    only fits Bellman-Q matrix targets. It does not use BVI actions or policy
    probabilities as labels.
    """

    if network.architecture != "standard":
        raise ValueError("last-layer refit currently supports only standard architecture")
    _x, _h1_pre, _h1, _h2_pre, h2, _out = network._forward(features)
    design = np.concatenate([h2, np.ones((h2.shape[0], 1), dtype=float)], axis=1)
    system = design.T @ design + float(ridge) * np.eye(design.shape[1], dtype=float)
    rhs = design.T @ targets
    coefficients = np.linalg.solve(system, rhs)
    network.w3 = coefficients[:-1]
    network.b3 = coefficients[-1]
    predictions = network.predict_batch(features)
    abs_errors = np.abs(predictions - targets)
    return {
        "abs_error_mean": float(abs_errors.mean()),
        "abs_error_max": float(abs_errors.max()),
    }


def train_matrix_batch_with_contrast(
    network: TwoLayerDQN,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    learning_rate: float,
    contrast_loss_weight: float,
    grad_clip_norm: float | None,
) -> tuple[float, np.ndarray]:
    """Train on Q entries plus action-difference contrasts.

    The extra contrast term is still a Bellman-Q fitting loss: it only uses the
    target Q matrix, not BVI policy probabilities or actions. It emphasizes the
    action-dependent quantities that determine a 2x2 matrix-game equilibrium.
    """

    x, h1_pre, h1, h2_pre, h2, out = network._forward(features)
    errors = out - targets
    batch_size = features.shape[0]
    normalizer = float(batch_size * network.output_size)
    grad_out = 2.0 * errors / normalizer
    loss = float(np.mean(errors * errors))

    out_m = out.reshape(-1, 2, 2)
    target_m = targets.reshape(-1, 2, 2)
    contrast_pairs = [
        ((0, 0), (0, 1)),
        ((1, 0), (1, 1)),
        ((0, 0), (1, 0)),
        ((0, 1), (1, 1)),
        ((0, 0), (1, 1)),
        ((1, 0), (0, 1)),
    ]
    grad_contrast = np.zeros_like(out_m)
    contrast_loss = 0.0
    contrast_norm = float(batch_size * len(contrast_pairs))
    for left, right in contrast_pairs:
        diff_error = (
            out_m[:, left[0], left[1]]
            - out_m[:, right[0], right[1]]
            - target_m[:, left[0], left[1]]
            + target_m[:, right[0], right[1]]
        )
        contrast_loss += float(np.mean(diff_error * diff_error))
        grad = 2.0 * contrast_loss_weight * diff_error / contrast_norm
        grad_contrast[:, left[0], left[1]] += grad
        grad_contrast[:, right[0], right[1]] -= grad
    loss += contrast_loss_weight * contrast_loss / len(contrast_pairs)
    grad_out += grad_contrast.reshape(out.shape)

    if network.architecture == "standard":
        grad_w3 = h2.T @ grad_out
        grad_b3 = grad_out.sum(axis=0)
        grad_h2 = grad_out @ network.w3.T
        output_grads = {"w3": grad_w3, "b3": grad_b3}
    else:
        grad_value = grad_out.sum(axis=1, keepdims=True)
        grad_advantage = grad_out - grad_out.mean(axis=1, keepdims=True)
        grad_wv = h2.T @ grad_value
        grad_bv = grad_value.sum(axis=0)
        grad_wa = h2.T @ grad_advantage
        grad_ba = grad_advantage.sum(axis=0)
        grad_h2 = grad_value @ network.wv.T + grad_advantage @ network.wa.T
        output_grads = {"wv": grad_wv, "bv": grad_bv, "wa": grad_wa, "ba": grad_ba}
    grad_h2_pre = grad_h2 * (h2_pre > 0.0)
    grad_w2 = h1.T @ grad_h2_pre
    grad_b2 = grad_h2_pre.sum(axis=0)
    grad_h1 = grad_h2_pre @ network.w2.T
    grad_h1_pre = grad_h1 * (h1_pre > 0.0)
    grad_w1 = x.T @ grad_h1_pre
    grad_b1 = grad_h1_pre.sum(axis=0)

    grads = {
        "w1": grad_w1,
        "b1": grad_b1,
        "w2": grad_w2,
        "b2": grad_b2,
    }
    grads.update(output_grads)
    if grad_clip_norm is not None:
        total_norm = float(np.sqrt(sum(float(np.sum(grad * grad)) for grad in grads.values())))
        if total_norm > grad_clip_norm:
            scale = grad_clip_norm / (total_norm + 1e-12)
            grads = {name: grad * scale for name, grad in grads.items()}
    network._adam_step(grads, learning_rate)
    return loss, np.abs(errors)


def state_sampling_weights(
    q_targets: np.ndarray,
    *,
    mixed_scale: float,
    margin_scale: float,
) -> np.ndarray:
    flat = q_targets.reshape(4, -1).T
    weights = np.ones(flat.shape[0], dtype=float)
    if mixed_scale <= 0.0 and margin_scale <= 0.0:
        return weights
    for index, matrix in enumerate(flat.reshape(-1, 2, 2)):
        matrix = matrix - matrix.mean()
        game = solve_zero_sum_matrix_game(matrix)
        p_attack = float(game["attacker_strategy"][1])
        p_defend = float(game["defender_strategy"][1])
        sensitivity = 4.0 * (
            p_attack * (1.0 - p_attack) + p_defend * (1.0 - p_defend)
        )
        weights[index] += mixed_scale * sensitivity
        if margin_scale > 0.0:
            defender = np.asarray(game["defender_strategy"], dtype=float)
            attacker = np.asarray(game["attacker_strategy"], dtype=float)
            attacker_payoffs = matrix @ defender
            defender_payoffs = attacker @ matrix
            attacker_margin = float(np.max(attacker_payoffs) - np.min(attacker_payoffs))
            defender_margin = float(np.max(defender_payoffs) - np.min(defender_payoffs))
            margin = min(attacker_margin, defender_margin)
            weights[index] += margin_scale * min(10.0, 1.0 / (margin + 1e-3))
    return weights


def dqn_policy_rows(
    states: list[tuple[int, int, int, int]],
    network: TwoLayerDQN,
    features: np.ndarray,
) -> tuple[list[dict], dict]:
    rows = []
    predictions = network.predict_batch(features).reshape(-1, 2, 2)
    for state, matrix in zip(states, predictions):
        matrix = matrix - matrix.mean()
        game = solve_zero_sum_matrix_game(matrix)
        queues = state[:3]
        row = {
            "method": "nnq",
            "state": list(state),
            "queues": list(queues),
            "position": int(state[3]),
            "total_queue": int(sum(queues)),
            "queue_gap": int(max(queues) - min(queues)),
            "nominal_targets": list(targets(queues, "max")),
            "attacked_targets": list(targets(queues, "min")),
            "p_no_defend": float(game["defender_strategy"][0]),
            "p_defend": float(game["defender_strategy"][1]),
            "p_no_attack": float(game["attacker_strategy"][0]),
            "p_attack": float(game["attacker_strategy"][1]),
            "value": float(game["value"]),
        }
        rows.append(row)
    defend_probs = np.asarray([row["p_defend"] for row in rows], dtype=float)
    attack_probs = np.asarray([row["p_attack"] for row in rows], dtype=float)
    return rows, {
        "num_policy_states": len(rows),
        "defend_probability_mean": float(defend_probs.mean()),
        "defend_probability_max": float(defend_probs.max()),
        "attack_probability_mean": float(attack_probs.mean()),
        "attack_probability_max": float(attack_probs.max()),
        "num_states_p_defend_at_least_threshold": int(np.sum(defend_probs >= 0.5)),
    }


def targets(queues: tuple[int, int, int], mode: str) -> tuple[int, ...]:
    value = min(queues) if mode == "min" else max(queues)
    return tuple(index for index, queue in enumerate(queues) if queue == value)


def save_network(path: Path, network: TwoLayerDQN, *, max_q: int, feature_set: str) -> None:
    payload = {name: value for name, value in network.params().items()}
    payload["max_queue_length"] = np.array([max_q], dtype=int)
    payload["feature_set_name"] = np.array([feature_set])
    payload["architecture"] = np.array([network.architecture])
    np.savez(path, **payload)


if __name__ == "__main__":
    raise SystemExit(main())
