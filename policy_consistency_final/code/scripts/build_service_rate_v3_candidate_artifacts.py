#!/usr/bin/env python3
"""Build service-rate-control v3-candidate BVI and fitted minimax-DQN artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ROUTING_SOURCE = ROOT / "experiments" / "source_faithful_routing_consistency"
for path in (SRC, ROUTING_SOURCE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from adversarial_queueing.algorithms.bvi import run_bounded_value_iteration  # noqa: E402
from adversarial_queueing.algorithms.minimax_solver import solve_zero_sum_matrix_game  # noqa: E402
from adversarial_queueing.envs.service_rate_control import ServiceRateControlEnv  # noqa: E402
from adversarial_queueing.evaluation.policy_grid import bvi_policy_grid  # noqa: E402
from adversarial_queueing.utils.config import (  # noqa: E402
    build_policy_grid_config,
    build_service_rate_config,
    load_config,
)
from routing_bvi_dqn_consistency import TwoLayerDQN  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvi-config", default="configs/service_rate_v3_candidate_bvi.yaml")
    parser.add_argument("--dqn-config", default="configs/service_rate_v3_candidate_dqn.yaml")
    parser.add_argument("--output-dir", default="artifacts/service_rate_v3_candidate")
    args = parser.parse_args()

    bvi_config_path = Path(args.bvi_config)
    dqn_config_path = Path(args.dqn_config)
    bvi_data = load_config(bvi_config_path)
    dqn_data = load_config(dqn_config_path)
    output_dir = Path(args.output_dir)
    bvi_dir = output_dir / "bvi"
    dqn_dir = output_dir / "dqn"
    bvi_dir.mkdir(parents=True, exist_ok=True)
    dqn_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bvi_config_path, bvi_dir / "config.yaml")
    shutil.copyfile(dqn_config_path, dqn_dir / "config.yaml")

    env = ServiceRateControlEnv(build_service_rate_config(bvi_data))
    bvi_cfg = bvi_data["bvi"]
    bvi_result = run_bounded_value_iteration(
        env,
        max_queue_length=int(bvi_cfg["max_queue_length"]),
        tolerance=float(bvi_cfg["tolerance"]),
        max_iterations=int(bvi_cfg["max_iterations"]),
    )
    policy_cfg = build_policy_grid_config(bvi_data)
    bvi_rows, bvi_summary = bvi_policy_grid(env, bvi_result, policy_cfg)
    write_jsonl(bvi_dir / "policy_grid.jsonl", bvi_rows)

    states = list(range(int(bvi_cfg["max_queue_length"]) + 1))
    dqn_fixed_point = run_model_based_minimax_q(
        env,
        states,
        tolerance=float(dqn_data.get("dqn", {}).get("fixed_point_tolerance", bvi_cfg["tolerance"])),
        max_iterations=int(dqn_data.get("dqn", {}).get("fixed_point_max_iterations", bvi_cfg["max_iterations"])),
    )
    q_targets = dqn_fixed_point["q_values"]
    flat_targets = q_targets.reshape(len(states), 4)
    dqn_cfg = dqn_data["dqn"]
    center_targets = bool(dqn_cfg.get("center_targets", True))
    if center_targets:
        flat_targets = flat_targets - flat_targets.mean(axis=1, keepdims=True)
    features = feature_matrix(
        env,
        states,
        max_queue_length=int(bvi_cfg["max_queue_length"]),
        feature_set=str(dqn_cfg.get("feature_set", "service_rate_threshold_poly")),
    )

    rng = np.random.default_rng(int(dqn_cfg.get("seed", 3)))
    network = TwoLayerDQN(
        rng=rng,
        hidden_size=int(dqn_cfg.get("hidden_size", 64)),
        input_size=features.shape[1],
        output_size=4,
        architecture="standard",
    )
    metrics = fit_network(
        network,
        features,
        flat_targets,
        epochs=int(dqn_cfg.get("epochs", 4000)),
        learning_rate=float(dqn_cfg.get("learning_rate", 0.001)),
    )
    refit_metrics = None
    if bool(dqn_cfg.get("last_layer_refit", True)):
        refit_metrics = refit_last_layer(
            network,
            features,
            flat_targets,
            ridge=float(dqn_cfg.get("last_layer_ridge", 1e-8)),
        )

    dqn_rows, dqn_summary = dqn_policy_grid_rows(env, states, network, features)
    write_jsonl(dqn_dir / "policy_grid.jsonl", dqn_rows)
    q_rows, q_summary = dqn_q_diagnostic(env, states, network, features, flat_targets)
    write_jsonl(dqn_dir / "q_diagnostic.jsonl", q_rows)
    save_network(dqn_dir / "dqn_params.npz", network, max_queue_length=len(states) - 1)

    write_json(
        bvi_dir / "summary.json",
        {
            "method": "bvi",
            "benchmark": "service_rate_control_v3_candidate",
            "iterations": bvi_result.iterations,
            "residual": bvi_result.residual,
            "max_queue_length": bvi_result.max_queue_length,
            "policy_grid": bvi_summary,
            "nontrivial_bvi_states": nontrivial_state_summary(bvi_rows),
            "env_semantics": env_semantics(env),
        },
    )
    write_json(
        dqn_dir / "summary.json",
        {
            "method": "fitted_minimax_dqn",
            "benchmark": "service_rate_control_v3_candidate",
            "feature_set": dqn_cfg.get("feature_set", "service_rate_threshold_poly"),
            "input_size": int(features.shape[1]),
            "hidden_size": int(dqn_cfg.get("hidden_size", 64)),
            "epochs": int(dqn_cfg.get("epochs", 4000)),
            "learning_rate": float(dqn_cfg.get("learning_rate", 0.001)),
            "center_targets": center_targets,
            "last_layer_refit": bool(dqn_cfg.get("last_layer_refit", True)),
            "target_source": "independent_model_based_minimax_q_fixed_point",
            "fixed_point_iterations": dqn_fixed_point["iterations"],
            "fixed_point_residual": dqn_fixed_point["residual"],
            "last_layer_refit_metrics": refit_metrics,
            "fit_metrics": metrics[-5:],
            "policy_grid": dqn_summary,
            "policy_similarity_to_bvi": policy_similarity(bvi_rows, dqn_rows),
            "q_diagnostic": q_summary,
            "params_file": "dqn_params.npz",
            "notes": (
                "Fitted minimax-DQN: independently computed model-based minimax-Q "
                "fixed-point targets -> neural Q matrix -> zero-sum matrix-game "
                "policy extraction. No BVI values or BVI policy labels are used "
                "to train the DQN."
            ),
        },
    )
    print(f"wrote {output_dir}")
    print(
        "summary: "
        f"bvi_nontrivial={nontrivial_state_summary(bvi_rows)['count']}/{len(states)} "
        f"policy_similarity={policy_similarity(bvi_rows, dqn_rows)['similarity_percent']:.2f}% "
        f"q_mae={q_summary['centered_q_abs_error_mean']:.3g}"
    )
    return 0


def run_model_based_minimax_q(
    env: ServiceRateControlEnv,
    states: list[int],
    *,
    tolerance: float,
    max_iterations: int,
) -> dict[str, Any]:
    values = {int(state): 0.0 for state in states}
    residual = float("inf")
    q_values = np.zeros((len(states), 2, 2), dtype=float)
    for iteration in range(1, max_iterations + 1):
        old_values = dict(values)
        q_values = bellman_q_targets(env, old_values, states)
        new_values: dict[int, float] = {}
        residual = 0.0
        for si, state in enumerate(states):
            value = float(solve_zero_sum_matrix_game(q_values[si])["value"])
            new_values[int(state)] = value
            residual = max(residual, abs(value - old_values[int(state)]))
        values = new_values
        if residual <= tolerance:
            break
    q_values = bellman_q_targets(env, values, states)
    return {
        "q_values": q_values,
        "values": values,
        "iterations": iteration,
        "residual": residual,
    }


def bellman_q_targets(
    env: ServiceRateControlEnv,
    values: dict[int, float],
    states: list[int],
) -> np.ndarray:
    targets = np.zeros((len(states), 2, 2), dtype=float)
    max_state = max(values)
    for si, state in enumerate(states):
        for ai, attacker_action in enumerate(env.attacker_actions(state)):
            for di, defender_action in enumerate(env.defender_actions(state)):
                expected_next = 0.0
                for next_state, probability in env.transition_probabilities(
                    state, attacker_action, defender_action
                ).items():
                    expected_next += probability * values[min(int(next_state), max_state)]
                targets[si, ai, di] = (
                    env.cost(state, attacker_action, defender_action)
                    + env.discount * expected_next
                )
    return targets


def feature_matrix(
    env: ServiceRateControlEnv,
    states: list[int],
    *,
    max_queue_length: int,
    feature_set: str,
) -> np.ndarray:
    if feature_set != "service_rate_threshold_poly":
        raise ValueError("only service_rate_threshold_poly is implemented")
    scale = max(float(max_queue_length), 1.0)
    rows = []
    for state in states:
        q = float(state) / scale
        level = env.baseline_service_level(state)
        low = 1.0 if level == 0 else 0.0
        medium = 1.0 if level == 1 else 0.0
        high = 1.0 if level == 2 else 0.0
        rows.append(
            [
                q,
                q * q,
                q * q * q,
                float(state),
                float(state * state) / max(scale * scale, 1.0),
                low,
                medium,
                high,
                float(state - env.config.low_threshold) / scale,
                float(state - env.config.high_threshold) / scale,
                float(abs(state - env.config.low_threshold)) / scale,
                float(abs(state - env.config.high_threshold)) / scale,
            ]
        )
    return np.asarray(rows, dtype=float)


def fit_network(
    network: TwoLayerDQN,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
) -> list[dict[str, float]]:
    metrics = []
    for epoch in range(1, epochs + 1):
        loss, abs_errors = network.train_full_batch(
            features,
            targets,
            learning_rate=learning_rate,
            loss_type="mse",
            huber_delta=1.0,
            grad_clip_norm=20.0,
        )
        if epoch == 1 or epoch % 500 == 0 or epoch == epochs:
            predictions = network.predict_batch(features)
            eval_abs = np.abs(predictions - targets)
            metric = {
                "epoch": epoch,
                "loss": float(loss),
                "abs_error_mean": float(eval_abs.mean()),
                "abs_error_max": float(eval_abs.max()),
            }
            metrics.append(metric)
            print(
                "epoch={epoch} loss={loss:.6g} mae={abs_error_mean:.6g}".format(
                    **metric
                ),
                flush=True,
            )
    return metrics


def refit_last_layer(
    network: TwoLayerDQN,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    ridge: float,
) -> dict[str, float]:
    _x, _h1_pre, _h1, _h2_pre, h2, _out = network._forward(features)
    design = np.concatenate([h2, np.ones((h2.shape[0], 1), dtype=float)], axis=1)
    system = design.T @ design + float(ridge) * np.eye(design.shape[1], dtype=float)
    rhs = design.T @ targets
    coeffs = np.linalg.solve(system, rhs)
    network.w3 = coeffs[:-1]
    network.b3 = coeffs[-1]
    predictions = network.predict_batch(features)
    errors = np.abs(predictions - targets)
    return {
        "abs_error_mean": float(errors.mean()),
        "abs_error_max": float(errors.max()),
    }


def dqn_policy_grid_rows(
    env: ServiceRateControlEnv,
    states: list[int],
    network: TwoLayerDQN,
    features: np.ndarray,
) -> tuple[list[dict[str, float | int | str]], dict[str, Any]]:
    rows = []
    for state, matrix in zip(states, network.predict_batch(features).reshape(-1, 2, 2)):
        matrix = matrix - matrix.mean()
        game = solve_zero_sum_matrix_game(matrix)
        rows.append(
            {
                "method": "dqn",
                "state": int(state),
                "p_no_attack": float(game["attacker_strategy"][0]),
                "p_attack": float(game["attacker_strategy"][1]),
                "p_no_defend": float(game["defender_strategy"][0]),
                "p_defend": float(game["defender_strategy"][1]),
                "value": float(game["value"]),
            }
        )
    return rows, policy_summary(rows)


def dqn_q_diagnostic(
    env: ServiceRateControlEnv,
    states: list[int],
    network: TwoLayerDQN,
    features: np.ndarray,
    centered_targets: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    predictions = network.predict_batch(features)
    errors = predictions - centered_targets
    rows = []
    for si, state in enumerate(states):
        for ai, attacker_action in enumerate(env.attacker_actions(state)):
            for di, defender_action in enumerate(env.defender_actions(state)):
                index = ai * 2 + di
                rows.append(
                    {
                        "state": int(state),
                        "attacker_action": int(attacker_action),
                        "defender_action": int(defender_action),
                        "dqn_centered_q": float(predictions[si, index]),
                        "target_centered_q": float(centered_targets[si, index]),
                        "signed_error": float(errors[si, index]),
                        "abs_error": float(abs(errors[si, index])),
                    }
                )
    abs_errors = np.abs(errors)
    return rows, {
        "num_q_entries": int(abs_errors.size),
        "centered_q_abs_error_mean": float(abs_errors.mean()),
        "centered_q_abs_error_max": float(abs_errors.max()),
    }


def policy_summary(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    p_attack = np.asarray([row["p_attack"] for row in rows], dtype=float)
    p_defend = np.asarray([row["p_defend"] for row in rows], dtype=float)
    return {
        "num_states": len(rows),
        "p_attack_mean": float(p_attack.mean()),
        "p_attack_max": float(p_attack.max()),
        "p_defend_mean": float(p_defend.mean()),
        "p_defend_max": float(p_defend.max()),
        "nontrivial": nontrivial_state_summary(rows),
    }


def nontrivial_state_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    active = [
        int(row["state"])
        for row in rows
        if max(float(row.get("p_attack", 0.0)), float(row.get("p_defend", 0.0))) > 0.05
    ]
    return {
        "count": len(active),
        "states": active,
    }


def policy_similarity(
    bvi_rows: list[dict[str, Any]],
    dqn_rows: list[dict[str, Any]],
) -> dict[str, float | int]:
    gaps = []
    for bvi, dqn in zip(bvi_rows, dqn_rows):
        if int(bvi["state"]) != int(dqn["state"]):
            raise ValueError("policy grids are not aligned")
        gaps.append(
            max(
                abs(float(bvi["p_attack"]) - float(dqn["p_attack"])),
                abs(float(bvi["p_defend"]) - float(dqn["p_defend"])),
            )
        )
    gaps_array = np.asarray(gaps, dtype=float)
    return {
        "mean_max_probability_gap": float(gaps_array.mean()),
        "max_probability_gap": float(gaps_array.max()),
        "similarity": float(1.0 - gaps_array.mean()),
        "similarity_percent": float(100.0 * (1.0 - gaps_array.mean())),
        "fraction_gap_le_0_05": float(np.mean(gaps_array <= 0.05)),
    }


def save_network(path: Path, network: TwoLayerDQN, *, max_queue_length: int) -> None:
    payload = {name: value for name, value in network.params().items()}
    payload["max_queue_length"] = np.asarray([max_queue_length], dtype=int)
    payload["feature_set_name"] = np.asarray(["service_rate_threshold_poly"])
    payload["architecture"] = np.asarray([network.architecture])
    np.savez(path, **payload)


def env_semantics(env: ServiceRateControlEnv) -> dict[str, Any]:
    return {
        "version": "service_rate_control_v3_candidate",
        "attacker_actions": {"0": "not_attack", "1": "attack"},
        "defender_actions": {"0": "not_defend", "1": "defend"},
        "threshold_policy": {
            "low_if_q_lt": env.config.low_threshold,
            "medium_if_q_lt": env.config.high_threshold,
            "high_otherwise": True,
        },
        "attack_success_effect": "attack=1 and defend=0 forces high service for one step",
        "bvi_algorithm": "unchanged bounded value iteration with local zero-sum matrix games",
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
