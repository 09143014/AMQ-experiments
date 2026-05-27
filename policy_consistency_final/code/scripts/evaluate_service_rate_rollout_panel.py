#!/usr/bin/env python3
"""Evaluate service-rate policy consistency on a multi-state rollout panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from adversarial_queueing.envs.service_rate_control import ServiceRateControlEnv
from adversarial_queueing.utils.config import build_service_rate_config, load_config
from build_service_rate_policy_path_figure import (
    _load_bvi_policy_grid,
    _load_nnq_policies,
    _simulate_same_state_path,
)


DEFAULT_INITIALS = ["0", "2", "5", "8", "12", "16", "19"]
DEFAULT_SEEDS = ["301:1729", "302:1730", "303:1731", "401:1801", "501:1901"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvi-run-dir",
        default=str(ROOT / "artifacts" / "service_rate_v3_candidate" / "bvi"),
    )
    parser.add_argument(
        "--dqn-run-dir",
        default=str(ROOT / "artifacts" / "service_rate_v3_candidate" / "dqn"),
    )
    parser.add_argument("--initial-state", action="append", dest="initials")
    parser.add_argument("--seed-pair", action="append", dest="seed_pairs")
    parser.add_argument("--horizon", type=int, default=75)
    parser.add_argument("--json-output")
    args = parser.parse_args()

    initials = [int(raw) for raw in (args.initials or DEFAULT_INITIALS)]
    seed_pairs = [_parse_seed_pair(raw) for raw in (args.seed_pairs or DEFAULT_SEEDS)]
    bvi_run = Path(args.bvi_run_dir)
    dqn_run = Path(args.dqn_run_dir)

    env = ServiceRateControlEnv(build_service_rate_config(load_config(bvi_run / "config.yaml")))
    bvi_policy, bvi_attacker = _load_bvi_policy_grid(bvi_run / "policy_grid.jsonl")
    dqn_policy, dqn_attacker = _load_nnq_policies(dqn_run)

    rows = []
    all_rollout_rows = []
    for initial in initials:
        for env_seed, attacker_seed in seed_pairs:
            rollout = _simulate_same_state_path(
                env,
                bvi_policy,
                dqn_policy,
                bvi_attacker,
                dqn_attacker,
                initial_state=initial,
                horizon=args.horizon,
                env_seed=env_seed,
                attacker_seed=attacker_seed,
                attacker_mode="learned_split",
            )
            joint = int(sum(row["actions_agree"] for row in rollout))
            attacker = int(sum(row["attacker_actions_agree"] for row in rollout))
            defender = int(sum(row["defender_actions_agree"] for row in rollout))
            active = [
                row
                for row in rollout
                if any(
                    int(row[key]) == 1
                    for key in (
                        "bvi_attacker_action",
                        "nnq_attacker_action",
                        "bvi_defender_action",
                        "nnq_defender_action",
                    )
                )
            ]
            active_joint = int(sum(row["actions_agree"] for row in active))
            max_probability_gaps = [
                max(
                    abs(float(row["bvi_p_attack"]) - float(row["nnq_p_attack"])),
                    abs(float(row["bvi_p_defend"]) - float(row["nnq_p_defend"])),
                )
                for row in rollout
            ]
            row = {
                "initial_state": initial,
                "env_seed": env_seed,
                "attacker_seed": attacker_seed,
                "horizon": args.horizon,
                "joint_agreement": joint,
                "attacker_agreement": attacker,
                "defender_agreement": defender,
                "joint_rate": joint / args.horizon,
                "active_steps": len(active),
                "active_joint_agreement": active_joint,
                "active_joint_rate": active_joint / len(active) if active else None,
                "bvi_attack_steps": int(sum(row["bvi_attacker_action"] for row in rollout)),
                "dqn_attack_steps": int(sum(row["nnq_attacker_action"] for row in rollout)),
                "bvi_defend_steps": int(sum(row["bvi_defender_action"] for row in rollout)),
                "dqn_defend_steps": int(sum(row["nnq_defender_action"] for row in rollout)),
                "mean_max_probability_gap": float(np.mean(max_probability_gaps)),
                "max_probability_gap": float(np.max(max_probability_gaps)),
            }
            rows.append(row)
            all_rollout_rows.extend(rollout)

    total_steps = int(args.horizon * len(rows))
    joint_values = np.asarray([row["joint_agreement"] for row in rows], dtype=float)
    active_steps = int(sum(row["active_steps"] for row in rows))
    active_joint = int(sum(row["active_joint_agreement"] for row in rows))
    summary = {
        "num_initial_states": len(initials),
        "num_seed_pairs": len(seed_pairs),
        "num_rollouts": len(rows),
        "horizon": args.horizon,
        "total_joint_agreement": int(joint_values.sum()),
        "total_steps": total_steps,
        "aggregate_joint_rate": float(joint_values.sum() / total_steps),
        "min_joint_agreement": int(joint_values.min()),
        "mean_joint_agreement": float(joint_values.mean()),
        "max_joint_agreement": int(joint_values.max()),
        "num_rollouts_ge_98_percent": int(np.sum(joint_values / args.horizon >= 0.98)),
        "active_steps": active_steps,
        "active_joint_agreement": active_joint,
        "active_joint_rate": float(active_joint / active_steps) if active_steps else None,
        "mean_bvi_attack_steps": float(np.mean([row["bvi_attack_steps"] for row in rows])),
        "mean_bvi_defend_steps": float(np.mean([row["bvi_defend_steps"] for row in rows])),
        "mean_max_probability_gap": float(
            np.mean([row["mean_max_probability_gap"] for row in rows])
        ),
        "max_probability_gap": float(max(row["max_probability_gap"] for row in rows)),
        "rows": rows,
    }
    print(
        "service-rate panel: "
        f"aggregate={summary['total_joint_agreement']}/{summary['total_steps']} "
        f"({100.0 * summary['aggregate_joint_rate']:.2f}%), "
        f"min={summary['min_joint_agreement']}/{args.horizon}, "
        f"ge98={summary['num_rollouts_ge_98_percent']}/{len(rows)}, "
        f"active={summary['active_joint_agreement']}/{summary['active_steps']}"
    )
    for row in rows:
        print(
            f"  init={row['initial_state']} seeds={row['env_seed']}:{row['attacker_seed']} "
            f"joint={row['joint_agreement']}/{args.horizon} "
            f"active={row['active_joint_agreement']}/{row['active_steps']} "
            f"A={row['bvi_attack_steps']} D={row['bvi_defend_steps']}"
        )
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _parse_seed_pair(raw: str) -> tuple[int, int]:
    left, right = raw.split(":", 1)
    return int(left), int(right)


if __name__ == "__main__":
    raise SystemExit(main())
