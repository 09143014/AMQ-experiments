#!/usr/bin/env python3
"""Sanity checks for the polling BVI / fitted-DQN policy-consistency chain."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from adversarial_queueing.algorithms.minimax_solver import solve_zero_sum_matrix_game
from adversarial_queueing.envs.polling import PollingEnv
from adversarial_queueing.utils.config import build_polling_config, load_config
from build_polling3_fitted_dqn_artifact import (
    q_targets_from_values,
    run_vectorized_bvi,
    solve_two_by_two_grid,
)


def main() -> int:
    rng = np.random.default_rng(20260521)
    config_path = ROOT / "configs" / "polling3_bvi_policy_probe_max30.yaml"
    config = load_config(config_path)
    env = PollingEnv(build_polling_config(config))
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
        progress_interval=10_000,
    )
    values = bvi_result["values"]
    targets = q_targets_from_values(
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

    max_payoff_gap = 0.0
    for _ in range(500):
        state = tuple(int(x) for x in rng.integers(0, max_q + 1, size=3)) + (
            int(rng.integers(0, 3)),
        )
        for attacker_action in (0, 1):
            for defender_action in (0, 1):
                expected = env.cost(state, attacker_action, defender_action)
                for next_state, probability in env.transition_probabilities(
                    state,
                    attacker_action,
                    defender_action,
                ).items():
                    bounded = tuple(
                        min(int(value), max_q) if index < 3 else int(value)
                        for index, value in enumerate(next_state)
                    )
                    expected += gamma * float(probability) * values[bounded]
                actual = targets[
                    attacker_action,
                    defender_action,
                    state[0],
                    state[1],
                    state[2],
                    state[3],
                ]
                max_payoff_gap = max(max_payoff_gap, abs(float(actual - expected)))

    max_solver_gap = 0.0
    for _ in range(1000):
        matrix = rng.normal(size=(2, 2))
        value_grid, p_attack_grid, p_defend_grid = solve_two_by_two_grid(
            matrix.reshape(2, 2, 1)
        )
        game = solve_zero_sum_matrix_game(matrix)
        max_solver_gap = max(
            max_solver_gap,
            abs(float(value_grid[0]) - float(game["value"])),
            abs(float(p_attack_grid[0]) - float(game["attacker_strategy"][1])),
            abs(float(p_defend_grid[0]) - float(game["defender_strategy"][1])),
        )

    print(f"polling payoff max gap: {max_payoff_gap:.3e}")
    print(f"2x2 solver max gap: {max_solver_gap:.3e}")
    if max_payoff_gap > 1e-8 or max_solver_gap > 1e-8:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
