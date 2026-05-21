#!/usr/bin/env python3
"""Plot a BVI-vs-neural-minimax-Q routing rollout path."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from routing_bvi_dqn_consistency import (  # noqa: E402
    RoutingSecurityParams,
    State,
    TwoLayerDQN,
    dqn_matrix,
    encode_state,
    matrix_game,
    run_source_bvi,
    sample_binary_action,
    transition_from_uniform,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bound", type=int, default=20)
    parser.add_argument("--num-queues", type=int, default=3)
    parser.add_argument("--initial-state", default="0,5,10")
    parser.add_argument("--horizon", type=int, default=75)
    parser.add_argument("--seed", type=int, default=2401)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--architecture", choices=("standard", "dueling"), default="standard")
    parser.add_argument("--feature-set", choices=("raw", "structural", "poly2"), default="structural")
    parser.add_argument(
        "--params-file",
        default=(
            "experiments/source_faithful_routing_consistency/results/"
            "neural_fixed_point_q_structural_b20_seed2_long.dqn_params.npz"
        ),
    )
    parser.add_argument(
        "--svg-output",
        default=(
            "experiments/source_faithful_routing_consistency/results/figures/"
            "routing_bvi_vs_neural_fixed_point_q_initial_0_5_10_h75.svg"
        ),
    )
    parser.add_argument(
        "--jsonl-output",
        default=(
            "experiments/source_faithful_routing_consistency/results/"
            "routing_bvi_vs_neural_fixed_point_q_initial_0_5_10_h75.jsonl"
        ),
    )
    args = parser.parse_args()

    params = RoutingSecurityParams(bound=args.bound, num_queues=args.num_queues)
    initial_state = _parse_state(args.initial_state, args.num_queues)
    bvi = run_source_bvi(params, tolerance=1e-7, max_iterations=10_000)
    network = _load_network(
        Path(args.params_file),
        params=params,
        hidden_size=args.hidden_size,
        architecture=args.architecture,
        feature_set=args.feature_set,
    )
    rows = _rollout_rows(
        bvi["policy"],
        network,
        params,
        initial_state=initial_state,
        horizon=args.horizon,
        seed=args.seed,
        feature_set=args.feature_set,
    )

    svg_path = Path(args.svg_output)
    jsonl_path = Path(args.jsonl_output)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(
        _render_svg(
            rows,
            title="Routing Policy Path: BVI vs Neural Fixed-point Q",
            subtitle=(
                f"initial={_fmt_state(initial_state)}, B={args.bound}, seed={args.seed}, "
                f"horizon={args.horizon}; both attacker and defender actions are compared"
            ),
        ),
        encoding="utf-8",
    )
    jsonl_path.write_text(
        "\n".join(json.dumps(_jsonable(row), sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    joint = sum(row["joint_match"] for row in rows)
    print(f"wrote {svg_path}")
    print(f"wrote {jsonl_path}")
    print(
        f"summary: joint agreement {joint}/{len(rows)} = {100.0 * joint / len(rows):.2f}%; "
        f"BVI attacks={sum(row['bvi_attacker'] for row in rows)}, "
        f"NNQ attacks={sum(row['nnq_attacker'] for row in rows)}, "
        f"BVI defends={sum(row['bvi_defender'] for row in rows)}, "
        f"NNQ defends={sum(row['nnq_defender'] for row in rows)}"
    )
    return 0


def _load_network(
    path: Path,
    *,
    params: RoutingSecurityParams,
    hidden_size: int,
    architecture: str,
    feature_set: str,
) -> TwoLayerDQN:
    input_size = int(encode_state((0,) * params.num_queues, params.bound, feature_set).size)
    network = TwoLayerDQN(
        np.random.default_rng(0),
        hidden_size=hidden_size,
        input_size=input_size,
        architecture=architecture,
    )
    arrays = np.load(path)
    for name in network.params():
        setattr(network, name, arrays[name])
    network.m = {name: np.zeros_like(value) for name, value in network.params().items()}
    network.v = {name: np.zeros_like(value) for name, value in network.params().items()}
    network.t = 0
    return network


def _rollout_rows(
    bvi_policy: dict[State, dict[str, Any]],
    network: TwoLayerDQN,
    params: RoutingSecurityParams,
    *,
    initial_state: State,
    horizon: int,
    seed: int,
    feature_set: str,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    bvi_state = initial_state
    nnq_state = initial_state
    rows: list[dict[str, Any]] = []
    for step in range(horizon):
        bvi = bvi_policy[bvi_state]
        nnq_game = matrix_game(dqn_matrix(network, nnq_state, params.bound, feature_set))
        attacker_u = float(rng.random())
        defender_u = float(rng.random())
        env_u = float(rng.random())

        bvi_attacker = sample_binary_action(bvi["attacker_strategy"], attacker_u)
        bvi_defender = sample_binary_action(bvi["defender_strategy"], defender_u)
        nnq_attacker = sample_binary_action(nnq_game["attacker_strategy"], attacker_u)
        nnq_defender = sample_binary_action(nnq_game["defender_strategy"], defender_u)

        next_bvi = transition_from_uniform(
            bvi_state,
            bvi_attacker,
            bvi_defender,
            params,
            env_u,
        )
        next_nnq = transition_from_uniform(
            nnq_state,
            nnq_attacker,
            nnq_defender,
            params,
            env_u,
        )
        displayed_next_bvi = initial_state if next_bvi is None else next_bvi
        displayed_next_nnq = initial_state if next_nnq is None else next_nnq
        rows.append(
            {
                "step": step,
                "bvi_state": bvi_state,
                "nnq_state": nnq_state,
                "same_state": bvi_state == nnq_state,
                "bvi_attacker": bvi_attacker,
                "nnq_attacker": nnq_attacker,
                "bvi_defender": bvi_defender,
                "nnq_defender": nnq_defender,
                "bvi_attacker_action": bvi_attacker,
                "nnq_attacker_action": nnq_attacker,
                "bvi_defender_action": bvi_defender,
                "nnq_defender_action": nnq_defender,
                "attacker_match": bvi_attacker == nnq_attacker,
                "defender_match": bvi_defender == nnq_defender,
                "joint_match": bvi_attacker == nnq_attacker and bvi_defender == nnq_defender,
                "bvi_p_attack": float(bvi["attacker_strategy"][1]),
                "nnq_p_attack": float(nnq_game["attacker_strategy"][1]),
                "bvi_p_defend": float(bvi["defender_strategy"][1]),
                "nnq_p_defend": float(nnq_game["defender_strategy"][1]),
                "env_uniform": env_u,
                "bvi_next_state": displayed_next_bvi,
                "nnq_next_state": displayed_next_nnq,
                "bvi_event": "reset" if next_bvi is None else _event_label(bvi_state, next_bvi),
                "nnq_event": "reset" if next_nnq is None else _event_label(nnq_state, next_nnq),
            }
        )
        bvi_state = displayed_next_bvi
        nnq_state = displayed_next_nnq
    return rows


def _render_svg(rows: list[dict[str, Any]], *, title: str, subtitle: str) -> str:
    columns = 12
    left = 128
    col_w = 106
    right = 48
    top = 170
    panel_h = 344
    panels = (len(rows) + columns - 1) // columns
    width = left + col_w * columns + right
    height = top + panel_h * panels + 88
    joint = sum(row["joint_match"] for row in rows)
    attacker = sum(row["attacker_match"] for row in rows)
    defender = sum(row["defender_match"] for row in rows)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<rect x="24" y="22" width="{width - 48}" height="{height - 44}" '
        'rx="14" fill="#fbfcfe" stroke="#d8e0ec"/>',
        _txt(48, 62, title, 27, weight=800, fill="#26364f"),
        _txt(48, 91, subtitle, 13, fill="#596579"),
        _legend(width - 430, 70),
        _txt(
            48,
            118,
            (
                f"agreement: joint {joint}/{len(rows)} ({100 * joint / len(rows):.1f}%), "
                f"attacker {attacker}/{len(rows)}, defender {defender}/{len(rows)}"
            ),
            13,
            weight=800,
            fill="#26364f",
        ),
    ]

    for panel_index, start in enumerate(range(0, len(rows), columns)):
        chunk = rows[start : start + columns]
        y0 = top + panel_index * panel_h
        parts.append(_txt(48, y0 - 22, f"steps {start}-{start + len(chunk) - 1}", 15, weight=800))
        _render_panel(parts, chunk, left, col_w, y0)

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _render_panel(parts: list[str], rows: list[dict[str, Any]], left: int, col_w: int, y0: int) -> None:
    labels = [
        ("state", 0),
        ("BVI A", 52),
        ("NNQ A", 104),
        ("BVI D", 156),
        ("NNQ D", 208),
        ("event", 266),
    ]
    for label, offset in labels:
        parts.append(_txt(48, y0 + offset + 5, label, 12, weight=800, fill="#26364f"))
        if label != "event":
            parts.append(
                f'<line x1="{left - 22}" y1="{y0 + offset}" '
                f'x2="{left + col_w * (len(rows) - 1) + 22}" y2="{y0 + offset}" '
                'stroke="#e4e9f1" stroke-width="1"/>'
            )
    for index, row in enumerate(rows):
        x = left + index * col_w
        state = tuple(int(value) for value in row["bvi_state"])
        state_fill = "#eef3f9" if row["same_state"] else "#fff3d6"
        parts.append(f'<circle cx="{x}" cy="{y0}" r="15" fill="{state_fill}" stroke="#cbd7e6"/>')
        parts.append(_txt(x, y0 - 1, str(sum(state)), 10, anchor="middle", weight=800))
        parts.append(_txt(x, y0 + 24, _fmt_state(state), 8, anchor="middle", fill="#596579"))
        _action_cell(parts, x, y0 + 52, "A", int(row["bvi_attacker"]), float(row["bvi_p_attack"]))
        _action_cell(parts, x, y0 + 104, "A", int(row["nnq_attacker"]), float(row["nnq_p_attack"]))
        _action_cell(parts, x, y0 + 156, "D", int(row["bvi_defender"]), float(row["bvi_p_defend"]))
        _action_cell(parts, x, y0 + 208, "D", int(row["nnq_defender"]), float(row["nnq_p_defend"]))
        if not row["joint_match"]:
            parts.append(
                f'<rect x="{x - 26}" y="{y0 + 34}" width="52" height="194" rx="8" '
                'fill="none" stroke="#d92d20" stroke-width="2" stroke-dasharray="4 4"/>'
            )
        event = row["bvi_event"]
        parts.append(_txt(x, y0 + 271, event, 9, anchor="middle", weight=800, fill=_event_color(event)))
        parts.append(_txt(x, y0 + 296, str(row["step"]), 9, anchor="middle", fill="#8a94a6"))


def _action_cell(parts: list[str], x: int, y: int, action_type: str, action: int, probability: float) -> None:
    if action_type == "A":
        active_fill = "#c2410c"
        active_stroke = "#9a3412"
        inactive_fill = "#f1f5f9"
        inactive_stroke = "#cbd5e1"
        active_label = "A"
        inactive_label = "-"
    else:
        active_fill = "#2d7d46"
        active_stroke = "#1f5e35"
        inactive_fill = "#d3d9e3"
        inactive_stroke = "#9aa5b5"
        active_label = "D"
        inactive_label = "N"
    fill = active_fill if action else inactive_fill
    stroke = active_stroke if action else inactive_stroke
    label = active_label if action else inactive_label
    label_fill = "#ffffff" if action else "#26364f"
    parts.append(
        f'<rect x="{x - 15}" y="{y - 14}" width="30" height="28" rx="6" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
    )
    parts.append(_txt(x, y + 5, label, 13, anchor="middle", weight=800, fill=label_fill))
    parts.append(
        _txt(
            x,
            y + 24,
            f"p={probability:.2f}",
            9,
            anchor="middle",
            weight=700,
            fill="#596579",
        )
    )


def _legend(x: int, y: int) -> str:
    items = [
        f'<rect x="{x}" y="{y - 14}" width="28" height="28" rx="6" fill="#c2410c"/>',
        _txt(x + 36, y + 4, "A = attack", 12, weight=800, fill="#26364f"),
        f'<rect x="{x + 120}" y="{y - 14}" width="28" height="28" rx="6" fill="#2d7d46"/>',
        _txt(x + 156, y + 4, "D = defend", 12, weight=800, fill="#26364f"),
        f'<rect x="{x + 252}" y="{y - 14}" width="28" height="28" rx="6" fill="#d3d9e3" stroke="#9aa5b5"/>',
        _txt(x + 288, y + 4, "N = not defend", 12, weight=800, fill="#26364f"),
    ]
    return "\n".join(items)


def _event_label(state: State, next_state: State) -> str:
    diffs = [nxt - cur for cur, nxt in zip(state, next_state)]
    if all(diff == 0 for diff in diffs):
        return "stay"
    labels = []
    for index, diff in enumerate(diffs, start=1):
        if diff > 0:
            labels.append(f"+q{index}")
        elif diff < 0:
            labels.append(f"-q{index}")
    return ",".join(labels)


def _event_color(event: str) -> str:
    if event == "reset":
        return "#b42318"
    if event.startswith("+"):
        return "#0369a1"
    if event.startswith("-"):
        return "#6d28d9"
    return "#8a94a6"


def _txt(
    x: float,
    y: float,
    text: str,
    size: int,
    *,
    anchor: str = "start",
    weight: int = 500,
    fill: str = "#26364f",
) -> str:
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="Inter, Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{html.escape(text)}</text>'
    )


def _parse_state(raw: str, expected_length: int) -> State:
    state = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if len(state) != expected_length:
        raise ValueError(f"--initial-state must have {expected_length} comma-separated integers")
    return state


def _fmt_state(state: State) -> str:
    return "(" + ",".join(str(value) for value in state) + ")"


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
