#!/usr/bin/env python3
"""Draw one service-rate-control v2 BVI-vs-NNQ Bellman-target policy path."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adversarial_queueing.algorithms.minimax_solver import solve_zero_sum_matrix_game
from adversarial_queueing.envs.service_rate_control import ServiceRateControlEnv
from adversarial_queueing.utils.config import build_service_rate_config, load_config


Policy = dict[int, tuple[float, ...]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvi-run-dir", required=True)
    parser.add_argument("--nnq-run-dir", required=True)
    parser.add_argument("--initial-state", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=75)
    parser.add_argument("--env-seed", type=int, default=301)
    parser.add_argument("--attacker-seed", type=int, default=1729)
    parser.add_argument(
        "--attacker-mode",
        choices=("learned_split", "learned_bvi", "random"),
        default="learned_split",
    )
    parser.add_argument("--columns", type=int, default=12)
    parser.add_argument(
        "--svg-output",
        default="results/figures/service_rate_policy_path_bvi_vs_nnq_bellman_h75.svg",
    )
    parser.add_argument(
        "--jsonl-output",
        default="results/service_rate_policy_path_bvi_vs_nnq_bellman_h75.jsonl",
    )
    args = parser.parse_args()

    bvi_run = Path(args.bvi_run_dir)
    nnq_run = Path(args.nnq_run_dir)
    env = ServiceRateControlEnv(build_service_rate_config(load_config(bvi_run / "config.yaml")))
    bvi_policy, bvi_attacker = _load_bvi_policy_grid(bvi_run / "policy_grid.jsonl")
    nnq_policy, nnq_attacker = _load_nnq_bellman_target_policies(nnq_run / "q_diagnostic.jsonl")
    rows = _simulate_same_state_path(
        env,
        bvi_policy,
        nnq_policy,
        bvi_attacker,
        nnq_attacker,
        initial_state=args.initial_state,
        horizon=args.horizon,
        env_seed=args.env_seed,
        attacker_seed=args.attacker_seed,
        attacker_mode=args.attacker_mode,
    )

    svg_output = Path(args.svg_output)
    jsonl_output = Path(args.jsonl_output)
    svg_output.parent.mkdir(parents=True, exist_ok=True)
    jsonl_output.parent.mkdir(parents=True, exist_ok=True)
    svg_output.write_text(
        _render_svg(
            env,
            rows,
            title="Service-Rate v2 Policy Path: BVI vs NNQ Bellman Target",
            subtitle=(
                f"initial={args.initial_state}, env_seed={args.env_seed}, "
                f"attacker={args.attacker_mode}, attacker_seed={args.attacker_seed}, "
                f"horizon={len(rows)}"
            ),
            columns=args.columns,
        ),
        encoding="utf-8",
    )
    _write_jsonl(jsonl_output, rows)
    agreements = sum(row["actions_agree"] for row in rows)
    attacker_agreements = sum(row["attacker_actions_agree"] for row in rows)
    defender_agreements = sum(row["defender_actions_agree"] for row in rows)
    print(f"wrote {svg_output}")
    print(f"wrote {jsonl_output}")
    print(
        f"summary: steps={len(rows)} agreements={agreements}/{len(rows)} "
        f"attacker_agreements={attacker_agreements}/{len(rows)} "
        f"defender_agreements={defender_agreements}/{len(rows)} "
        f"bvi_actions={_action_counts(rows, 'bvi_action')} "
        f"nnq_actions={_action_counts(rows, 'nnq_action')} "
        f"bvi_attacks={sum(row['bvi_attacker_action'] for row in rows)} "
        f"nnq_attacks={sum(row['nnq_attacker_action'] for row in rows)}"
    )
    return 0


def _simulate_same_state_path(
    env: ServiceRateControlEnv,
    bvi_policy: Policy,
    nnq_policy: Policy,
    bvi_attacker: Policy,
    nnq_attacker: Policy,
    *,
    initial_state: int,
    horizon: int,
    env_seed: int,
    attacker_seed: int,
    attacker_mode: str,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(env_seed)
    attacker_rng = np.random.default_rng(attacker_seed)
    bvi_state = int(initial_state)
    nnq_state = int(initial_state)
    rows = []
    for step in range(horizon):
        bvi_lookup_state = _clip_state(bvi_state, bvi_policy)
        nnq_lookup_state = _clip_state(nnq_state, nnq_policy)
        bvi_probs = bvi_policy[bvi_lookup_state]
        nnq_probs = nnq_policy[nnq_lookup_state]
        bvi_attacker_probs = bvi_attacker[bvi_lookup_state]
        nnq_attacker_probs = nnq_attacker[nnq_lookup_state]
        attacker_u = float(attacker_rng.random())
        defender_u = float(attacker_rng.random())
        if attacker_mode == "learned_bvi":
            bvi_attacker_action = int(attacker_u < bvi_attacker_probs[1])
            nnq_attacker_action = bvi_attacker_action
        elif attacker_mode == "learned_split":
            bvi_attacker_action = int(attacker_u < bvi_attacker_probs[1])
            nnq_attacker_action = int(attacker_u < nnq_attacker_probs[1])
        else:
            bvi_attacker_action = int(attacker_u < 0.5)
            nnq_attacker_action = bvi_attacker_action
        bvi_action = int(defender_u < bvi_probs[1])
        nnq_action = int(defender_u < nnq_probs[1])
        random_u = float(rng.random())
        next_bvi_state = _sample_transition(
            env, bvi_state, bvi_attacker_action, bvi_action, random_u
        )
        next_nnq_state = _sample_transition(
            env, nnq_state, nnq_attacker_action, nnq_action, random_u
        )
        bvi_baseline_level = env.baseline_service_level(bvi_state)
        nnq_baseline_level = env.baseline_service_level(nnq_state)
        bvi_realized_level = env.realized_service_level(
            bvi_state, bvi_attacker_action, bvi_action
        )
        nnq_realized_level = env.realized_service_level(
            nnq_state, nnq_attacker_action, nnq_action
        )
        bvi_realized_mu = env.realized_mu(bvi_state, bvi_attacker_action, bvi_action)
        nnq_realized_mu = env.realized_mu(nnq_state, nnq_attacker_action, nnq_action)
        rows.append(
            {
                "step": step,
                "state": bvi_state,
                "bvi_state": bvi_state,
                "nnq_state": nnq_state,
                "same_state": bvi_state == nnq_state,
                "lookup_state": bvi_lookup_state,
                "bvi_lookup_state": bvi_lookup_state,
                "nnq_lookup_state": nnq_lookup_state,
                "next_state": next_bvi_state,
                "bvi_next_state": next_bvi_state,
                "nnq_next_state": next_nnq_state,
                "bvi_attacker_action": bvi_attacker_action,
                "nnq_attacker_action": nnq_attacker_action,
                "bvi_defender_action": bvi_action,
                "nnq_defender_action": nnq_action,
                "bvi_p_attack": float(bvi_attacker_probs[1]),
                "nnq_p_attack": float(nnq_attacker_probs[1]),
                "bvi_attacker_p_attack": float(bvi_attacker_probs[1]),
                "nnq_attacker_p_attack": float(nnq_attacker_probs[1]),
                "bvi_action": bvi_action,
                "nnq_action": nnq_action,
                "bvi_p_defend": float(bvi_probs[1]),
                "nnq_p_defend": float(nnq_probs[1]),
                "bvi_policy": list(float(value) for value in bvi_probs),
                "nnq_policy": list(float(value) for value in nnq_probs),
                "bvi_baseline_service_level": int(bvi_baseline_level),
                "nnq_baseline_service_level": int(nnq_baseline_level),
                "bvi_realized_service_level": int(bvi_realized_level),
                "nnq_realized_service_level": int(nnq_realized_level),
                "baseline_service_label": _service_label(bvi_baseline_level),
                "bvi_service_label": _service_label(bvi_realized_level),
                "nnq_service_label": _service_label(nnq_realized_level),
                "realized_mu": float(bvi_realized_mu),
                "bvi_realized_mu": float(bvi_realized_mu),
                "nnq_realized_mu": float(nnq_realized_mu),
                "attacker_actions_agree": bool(bvi_attacker_action == nnq_attacker_action),
                "defender_actions_agree": bool(bvi_action == nnq_action),
                "actions_agree": bool(
                    bvi_attacker_action == nnq_attacker_action and bvi_action == nnq_action
                ),
                "event_label": _event_label(bvi_state, next_bvi_state),
                "bvi_event_label": _event_label(bvi_state, next_bvi_state),
                "nnq_event_label": _event_label(nnq_state, next_nnq_state),
            }
        )
        bvi_state = next_bvi_state
        nnq_state = next_nnq_state
    return rows


def _render_svg(
    env: ServiceRateControlEnv,
    rows: list[dict[str, Any]],
    *,
    title: str,
    subtitle: str,
    columns: int,
) -> str:
    left = 138
    col_w = 108
    right = 56
    top = 156
    panel_h = 432
    width = left + col_w * columns + right
    height = top + panel_h * ((len(rows) + columns - 1) // columns) + 96
    text = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            f'<rect x="22" y="22" width="{width - 44}" height="{height - 44}" '
            'rx="14" fill="#fbfcfe" stroke="#d7dde7"/>'
        ),
        _txt(48, 66, title, 28, weight=800, fill="#26364f"),
        _txt(48, 96, subtitle, 14, fill="#596579"),
        _legend(width - 500, 68),
    ]
    for panel_index, start in enumerate(range(0, len(rows), columns)):
        chunk = rows[start : start + columns]
        y0 = top + panel_index * panel_h
        text.append(
            _txt(
                48,
                y0 - 34,
                f"steps {start}-{start + len(chunk) - 1}",
                15,
                weight=800,
                fill="#26364f",
            )
        )
        _render_panel(text, env, chunk, left, col_w, y0)

    agreements = sum(row["actions_agree"] for row in rows)
    attacker_agreements = sum(row["attacker_actions_agree"] for row in rows)
    defender_agreements = sum(row["defender_actions_agree"] for row in rows)
    counts = _action_counts(rows, "bvi_action")
    summary_y = height - 52
    text.append(
        f'<rect x="48" y="{summary_y - 24}" width="{width - 96}" height="42" '
        'rx="10" fill="#f2f6fb" stroke="#dce4ef"/>'
    )
    text.append(
        _txt(
            70,
            summary_y + 2,
            (
                f"Path check: joint {agreements}/{len(rows)}, "
                f"attacker {attacker_agreements}/{len(rows)}, "
                f"defender {defender_agreements}/{len(rows)}; "
                f"BVI defends/not = {counts[1]}/{counts[0]}; "
                f"BVI/NNQ attacks = {sum(row['bvi_attacker_action'] for row in rows)}/"
                f"{sum(row['nnq_attacker_action'] for row in rows)}."
            ),
            14,
            weight=700,
            fill="#26364f",
        )
    )
    text.append("</svg>")
    return "\n".join(text) + "\n"


def _render_panel(
    text: list[str],
    env: ServiceRateControlEnv,
    rows: list[dict[str, Any]],
    left: int,
    col_w: int,
    y0: int,
) -> None:
    labels = [
        ("state", 0),
        ("mu", 42),
        ("BVI A", 94),
        ("NNQ A", 150),
        ("BVI D", 206),
        ("NNQ D", 262),
        ("event", 332),
        ("step", 364),
    ]
    for label, offset in labels:
        text.append(_txt(48, y0 + offset + 5, label, 13, weight=800, fill="#26364f"))
        if label != "event":
            text.append(
                f'<line x1="{left}" y1="{y0 + offset}" x2="{left + col_w * (len(rows) - 1)}" '
                f'y2="{y0 + offset}" stroke="#e4e9f1" stroke-width="1.1"/>'
            )
    for index, row in enumerate(rows):
        x = left + index * col_w
        state_label = (
            str(int(row["bvi_state"]))
            if row["same_state"]
            else f"{int(row['bvi_state'])}|{int(row['nnq_state'])}"
        )
        state_fill = "#eef3f9" if row["same_state"] else "#fff3d6"
        text.append(f'<circle cx="{x}" cy="{y0}" r="14" fill="{state_fill}" stroke="#cfd8e6"/>')
        text.append(_txt(x, y0 + 5, state_label, 11, anchor="middle", weight=800, fill="#26364f"))
        text.append(
            _txt(
                x,
                y0 + 40,
                f"base {row['baseline_service_label']}",
                9,
                anchor="middle",
                weight=700,
                fill="#596579",
            )
        )
        text.append(
            _txt(
                x,
                y0 + 57,
                f"B/N {row['bvi_service_label']}/{row['nnq_service_label']}",
                9,
                anchor="middle",
                weight=700,
                fill="#1769aa",
            )
        )
        _render_attacker_action(text, x, y0 + 94, int(row["bvi_attacker_action"]), float(row["bvi_p_attack"]))
        _render_attacker_action(text, x, y0 + 150, int(row["nnq_attacker_action"]), float(row["nnq_p_attack"]))
        _render_action(text, x, y0 + 206, int(row["bvi_action"]), float(row["bvi_p_defend"]))
        _render_action(text, x, y0 + 262, int(row["nnq_action"]), float(row["nnq_p_defend"]))
        if not row["actions_agree"]:
            text.append(
                f'<rect x="{x - 26}" y="{y0 + 76}" width="52" height="210" rx="8" '
                'fill="none" stroke="#d92d20" stroke-width="2" stroke-dasharray="4 4"/>'
            )
        event = str(row["bvi_event_label"])
        text.append(_txt(x, y0 + 337, event, 11, anchor="middle", weight=800, fill=_event_color(event)))
        text.append(_txt(x, y0 + 369, str(row["step"]), 10, anchor="middle", weight=800, fill="#8a94a6"))


def _render_action(text: list[str], x: int, y: int, action: int, probability: float) -> None:
    fill = "#2d7d46" if action else "#d3d9e3"
    stroke = "#1f5e35" if action else "#9aa5b5"
    label = "D" if action else "N"
    label_fill = "#ffffff" if action else "#26364f"
    text.append(
        f'<rect x="{x - 15}" y="{y - 15}" width="30" height="30" rx="6" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
    )
    text.append(_txt(x, y + 5, label, 15, anchor="middle", weight=800, fill=label_fill))
    text.append(_txt(x, y + 25, f"p={probability:.2f}", 9, anchor="middle", weight=700, fill="#596579"))


def _render_attacker_action(text: list[str], x: int, y: int, action: int, probability: float) -> None:
    fill = "#c2410c" if action else "#f1f5f9"
    stroke = "#9a3412" if action else "#cbd5e1"
    label = "A" if action else "-"
    label_fill = "#ffffff" if action else "#26364f"
    text.append(
        f'<rect x="{x - 15}" y="{y - 14}" width="30" height="28" rx="6" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
    )
    text.append(_txt(x, y + 5, label, 14, anchor="middle", weight=800, fill=label_fill))
    text.append(_txt(x, y + 25, f"p={probability:.2f}", 9, anchor="middle", weight=700, fill="#596579"))


def _legend(x: float, y: float) -> str:
    return "\n".join(
        [
            f'<rect x="{x}" y="{y - 15}" width="24" height="24" rx="5" fill="#c2410c"/>',
            _txt(x + 34, y + 3, "A = attack", 13, weight=700, fill="#26364f"),
            f'<rect x="{x + 136}" y="{y - 15}" width="24" height="24" rx="5" fill="#2d7d46"/>',
            _txt(x + 170, y + 3, "D = defend", 13, weight=700, fill="#26364f"),
            f'<rect x="{x + 270}" y="{y - 15}" width="24" height="24" rx="5" fill="#d3d9e3" stroke="#9aa5b5"/>',
            _txt(x + 304, y + 3, "N = not defend", 13, weight=700, fill="#26364f"),
        ]
    )


def _load_bvi_policy_grid(path: Path) -> tuple[Policy, Policy]:
    defender: Policy = {}
    attacker: Policy = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        state = int(row["state"])
        defender[state] = (
            float(row["p_no_defend"]),
            float(row["p_defend"]),
        )
        attacker[state] = (float(row["p_no_attack"]), float(row["p_attack"]))
    return defender, attacker


def _load_nnq_bellman_target_policies(path: Path) -> tuple[Policy, Policy]:
    matrices: dict[int, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        matrix = matrices.setdefault(int(row["state"]), np.zeros((2, 2), dtype=float))
        matrix[int(row["attacker_action"]), int(row["defender_action"])] = float(
            row["nnq_bellman_target"]
        )
    defender: Policy = {}
    attacker: Policy = {}
    for state, matrix in matrices.items():
        solution = solve_zero_sum_matrix_game(matrix)
        defender[state] = tuple(float(value) for value in solution["defender_strategy"])
        attacker[state] = tuple(float(value) for value in solution["attacker_strategy"])
    return defender, attacker


def _sample_transition(
    env: ServiceRateControlEnv,
    state: int,
    attacker_action: int,
    defender_action: int,
    random_u: float,
) -> int:
    cumulative = 0.0
    items = sorted(env.transition_probabilities(state, attacker_action, defender_action).items())
    for next_state, probability in items:
        cumulative += float(probability)
        if random_u <= cumulative + 1e-12:
            return int(next_state)
    return int(items[-1][0])


def _event_label(state: int, next_state: int) -> str:
    if next_state > state:
        return "+1"
    if next_state < state:
        return "-1"
    return "stay"


def _event_color(event: str) -> str:
    if event.startswith("+"):
        return "#1769aa"
    if event.startswith("-"):
        return "#6c4ab6"
    return "#8a94a6"


def _mu_label(env: ServiceRateControlEnv, mu: float) -> str:
    for index, level in enumerate(env.config.mu_levels):
        if abs(float(level) - mu) < 1e-9:
            return f"mu{index + 1}"
    return f"{mu:.1f}"


def _service_label(level: int) -> str:
    return {0: "L", 1: "M", 2: "H"}[int(level)]


def _clip_state(state: int, policy: Policy) -> int:
    return min(int(state), max(policy))


def _action_counts(rows: list[dict[str, Any]], key: str) -> dict[int, int]:
    return {action: sum(1 for row in rows if int(row[key]) == action) for action in (0, 1)}


def _txt(
    x: float,
    y: float,
    text: str,
    size: int,
    *,
    anchor: str = "start",
    weight: int = 400,
    fill: str = "#202124",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{html.escape(str(text))}</text>'
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
