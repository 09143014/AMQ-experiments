#!/usr/bin/env python3
"""Build service-rate-control v2 BVI and NNQ artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adversarial_queueing.algorithms.bvi import run_bounded_value_iteration  # noqa: E402
from adversarial_queueing.algorithms.nnq import NNQTrainer  # noqa: E402
from adversarial_queueing.envs.service_rate_control import ServiceRateControlEnv  # noqa: E402
from adversarial_queueing.evaluation.policy_grid import (  # noqa: E402
    bvi_policy_grid,
    nnq_policy_grid,
)
from adversarial_queueing.evaluation.service_rate_policy import (  # noqa: E402
    service_rate_nnq_q_diagnostic,
)
from adversarial_queueing.utils.config import (  # noqa: E402
    build_nnq_config,
    build_policy_grid_config,
    build_service_rate_config,
    load_config,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvi-config",
        default="configs/service_rate_v2_bvi_probe.yaml",
    )
    parser.add_argument(
        "--nnq-config",
        default="configs/service_rate_v2_nnq_probe_10k.yaml",
    )
    parser.add_argument("--bvi-output-dir", default="artifacts/service_rate_v2/bvi")
    parser.add_argument("--nnq-output-dir", default="artifacts/service_rate_v2/nnq")
    args = parser.parse_args()

    bvi_config_path = Path(args.bvi_config)
    nnq_config_path = Path(args.nnq_config)
    bvi_data = load_config(bvi_config_path)
    nnq_data = load_config(nnq_config_path)
    bvi_output = Path(args.bvi_output_dir)
    nnq_output = Path(args.nnq_output_dir)
    bvi_output.mkdir(parents=True, exist_ok=True)
    nnq_output.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(bvi_config_path, bvi_output / "config.yaml")
    shutil.copyfile(nnq_config_path, nnq_output / "config.yaml")

    bvi_env = ServiceRateControlEnv(build_service_rate_config(bvi_data))
    bvi_config = bvi_data["bvi"]
    bvi_result = run_bounded_value_iteration(
        bvi_env,
        max_queue_length=int(bvi_config["max_queue_length"]),
        tolerance=float(bvi_config["tolerance"]),
        max_iterations=int(bvi_config["max_iterations"]),
    )
    policy_config = build_policy_grid_config(bvi_data)
    bvi_rows, bvi_policy_summary = bvi_policy_grid(bvi_env, bvi_result, policy_config)
    _write_jsonl(bvi_output / "policy_grid.jsonl", bvi_rows)
    _write_json(
        bvi_output / "summary.json",
        {
            "method": "bvi",
            "iterations": bvi_result.iterations,
            "residual": bvi_result.residual,
            "max_queue_length": bvi_result.max_queue_length,
            "policy_grid": bvi_policy_summary,
            "env_semantics": _env_semantics(bvi_env),
        },
    )

    nnq_env = ServiceRateControlEnv(build_service_rate_config(nnq_data))
    trainer = NNQTrainer(nnq_env, build_nnq_config(nnq_data))
    nnq_result = trainer.train()
    _write_jsonl(nnq_output / "metrics.jsonl", nnq_result.metrics)
    nnq_rows, nnq_policy_summary = nnq_policy_grid(
        nnq_env,
        trainer,
        build_policy_grid_config(nnq_data),
    )
    _write_jsonl(nnq_output / "policy_grid.jsonl", nnq_rows)
    q_rows, q_summary = service_rate_nnq_q_diagnostic(nnq_env, trainer, bvi_result)
    _write_jsonl(nnq_output / "q_diagnostic.jsonl", q_rows)
    _write_json(
        nnq_output / "summary.json",
        {
            "method": "nnq",
            "final_state": int(nnq_result.final_state),
            "num_metrics": len(nnq_result.metrics),
            "policy_grid": nnq_policy_summary,
            "q_diagnostic": q_summary,
            "env_semantics": _env_semantics(nnq_env),
        },
    )

    print(f"wrote {bvi_output}")
    print(f"wrote {nnq_output}")
    print(
        "summary: "
        f"bvi_iterations={bvi_result.iterations} "
        f"bvi_residual={bvi_result.residual:.3g} "
        f"nnq_q_entries={q_summary['num_q_entries']}"
    )
    return 0


def _env_semantics(env: ServiceRateControlEnv) -> dict[str, Any]:
    return {
        "version": "service_rate_control_v2",
        "attacker_actions": {"0": "not_attack", "1": "attack"},
        "defender_actions": {"0": "not_defend", "1": "defend"},
        "threshold_policy": {
            "low_if_q_lt": env.config.low_threshold,
            "medium_if_q_lt": env.config.high_threshold,
            "high_otherwise": True,
        },
        "attack_success_effect": "attack=1 and defend=0 forces high service for one step",
        "defend_cost": env.config.defend_cost,
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
