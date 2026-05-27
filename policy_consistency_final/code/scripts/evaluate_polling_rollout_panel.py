#!/usr/bin/env python3
"""Evaluate polling policy consistency on an aggressive rollout panel."""

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

from adversarial_queueing.envs.polling import PollingEnv
from adversarial_queueing.utils.config import build_polling_config, load_config
from build_polling_policy_path_figure import _load_policies, _simulate_same_state_path


DEFAULT_INITIALS = [
    "5,10,15,0",
    "1,1,16,0",
    "2,14,17,0",
    "0,17,18,0",
    "5,10,18,0",
    "0,20,0,0",
    "3,3,3,0",
    "10,10,10,0",
    "1,15,20,0",
    "8,12,16,0",
]
DEFAULT_SEEDS = [
    "301:1729",
    "302:1730",
    "303:1731",
    "401:1801",
    "501:1901",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvi-run-dir", default=str(ROOT / "artifacts" / "polling3" / "bvi"))
    parser.add_argument("--dqn-run-dir", default=str(ROOT / "artifacts" / "polling3" / "dqn"))
    parser.add_argument("--initial-state", action="append", dest="initials")
    parser.add_argument("--seed-pair", action="append", dest="seed_pairs")
    parser.add_argument("--horizon", type=int, default=75)
    parser.add_argument(
        "--path-mode",
        choices=["coupled", "bvi_locked"],
        default="coupled",
    )
    parser.add_argument("--json-output")
    args = parser.parse_args()

    initials = [_parse_state(raw) for raw in (args.initials or DEFAULT_INITIALS)]
    seed_pairs = [_parse_seed_pair(raw) for raw in (args.seed_pairs or DEFAULT_SEEDS)]
    bvi_run = Path(args.bvi_run_dir)
    dqn_run = Path(args.dqn_run_dir)

    env = PollingEnv(build_polling_config(load_config(bvi_run / "config.yaml")))
    bvi_policy, bvi_attacker = _load_policies(bvi_run / "policy_inspection.jsonl")
    dqn_policy, dqn_attacker = _load_policies(dqn_run / "policy_inspection.jsonl")

    rows = []
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
                path_mode=args.path_mode,
            )
            joint = int(sum(row["actions_agree"] for row in rollout))
            attacker = int(sum(row["attacker_actions_agree"] for row in rollout))
            defender = int(sum(row["defender_actions_agree"] for row in rollout))
            same_state = int(sum(row["same_state"] for row in rollout))
            rows.append(
                {
                    "initial_state": list(initial),
                    "env_seed": env_seed,
                    "attacker_seed": attacker_seed,
                    "horizon": args.horizon,
                    "joint_agreement": joint,
                    "attacker_agreement": attacker,
                    "defender_agreement": defender,
                    "same_state_steps": same_state,
                    "joint_rate": joint / args.horizon,
                }
            )

    joint_values = np.asarray([row["joint_agreement"] for row in rows], dtype=float)
    total_steps = int(args.horizon * len(rows))
    summary = {
        "path_mode": args.path_mode,
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
        "rows": rows,
    }
    print(
        "polling panel: "
        f"path_mode={args.path_mode} "
        f"aggregate={summary['total_joint_agreement']}/{summary['total_steps']} "
        f"({100.0 * summary['aggregate_joint_rate']:.2f}%), "
        f"min={summary['min_joint_agreement']}/{args.horizon}, "
        f"ge98={summary['num_rollouts_ge_98_percent']}/{len(rows)}"
    )
    for row in rows:
        print(
            f"  init={tuple(row['initial_state'])} seeds={row['env_seed']}:{row['attacker_seed']} "
            f"joint={row['joint_agreement']}/{args.horizon} "
            f"same={row['same_state_steps']}/{args.horizon}"
        )
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _parse_state(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if len(values) != 4:
        raise ValueError("polling initial state must be q1,q2,q3,p")
    return values


def _parse_seed_pair(raw: str) -> tuple[int, int]:
    left, right = raw.split(":", 1)
    return int(left), int(right)


if __name__ == "__main__":
    raise SystemExit(main())
