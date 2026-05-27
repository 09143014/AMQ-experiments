#!/usr/bin/env python3
"""Draw one polling BVI-vs-NNQ policy path."""

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

from adversarial_queueing.envs.polling import PollingEnv
from adversarial_queueing.utils.config import build_polling_config, load_config


State = tuple[int, ...]
Policy = dict[State, tuple[float, float]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvi-run-dir", required=True)
    parser.add_argument("--nnq-run-dir", required=True)
    parser.add_argument("--initial-state", default="0,3,0")
    parser.add_argument("--horizon", type=int, default=75)
    parser.add_argument("--env-seed", type=int, default=301)
    parser.add_argument("--attacker-seed", type=int, default=1729)
    parser.add_argument("--columns", type=int, default=12)
    parser.add_argument(
        "--path-mode",
        choices=["coupled", "bvi_locked"],
        default="coupled",
        help=(
            "coupled lets BVI and DQN states evolve separately after a mismatch; "
            "bvi_locked compares both policies on the same BVI-driven state path."
        ),
    )
    parser.add_argument(
        "--svg-output",
        default="results/figures/polling_policy_path_bvi_vs_nnq_h75.svg",
    )
    parser.add_argument(
        "--jsonl-output",
        default="results/polling_policy_path_bvi_vs_nnq_h75.jsonl",
    )
    args = parser.parse_args()

    bvi_run = Path(args.bvi_run_dir)
    nnq_run = Path(args.nnq_run_dir)
    env = PollingEnv(build_polling_config(load_config(bvi_run / "config.yaml")))
    bvi_policy, bvi_attacker = _load_policies(bvi_run / "policy_inspection.jsonl")
    nnq_policy, nnq_attacker = _load_policies(nnq_run / "policy_inspection.jsonl")
    rows = _simulate_same_state_path(
        env,
        bvi_policy,
        nnq_policy,
        bvi_attacker,
        nnq_attacker,
        initial_state=_parse_state(args.initial_state),
        horizon=args.horizon,
        env_seed=args.env_seed,
        attacker_seed=args.attacker_seed,
        path_mode=args.path_mode,
    )

    svg_output = Path(args.svg_output)
    jsonl_output = Path(args.jsonl_output)
    svg_output.parent.mkdir(parents=True, exist_ok=True)
    jsonl_output.parent.mkdir(parents=True, exist_ok=True)
    svg_output.write_text(
        _render_svg(
            rows,
            title="Polling Policy Path: BVI vs NNQ",
            subtitle=(
                f"initial={_fmt_state(tuple(rows[0]['state']))}, "
                f"env_seed={args.env_seed}, attacker_seed={args.attacker_seed}, "
                f"horizon={len(rows)}, path_mode={args.path_mode}"
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
        f"bvi_defends={sum(row['bvi_action'] for row in rows)} "
        f"nnq_defends={sum(row['nnq_action'] for row in rows)} "
        f"bvi_attacks={sum(row['bvi_attacker_action'] for row in rows)} "
        f"nnq_attacks={sum(row['nnq_attacker_action'] for row in rows)}"
    )
    return 0


def _simulate_same_state_path(
    env: PollingEnv,
    bvi_policy: Policy,
    nnq_policy: Policy,
    bvi_attacker: Policy,
    nnq_attacker: Policy,
    *,
    initial_state: State,
    horizon: int,
    env_seed: int,
    attacker_seed: int,
    path_mode: str,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(env_seed)
    attacker_rng = np.random.default_rng(attacker_seed)
    bvi_state = initial_state
    nnq_state = initial_state
    rows = []
    for step in range(horizon):
        bvi_probs = _policy_probs(bvi_policy, bvi_state)
        nnq_probs = _policy_probs(nnq_policy, nnq_state)
        bvi_attacker_probs = _policy_probs(bvi_attacker, bvi_state)
        nnq_attacker_probs = _policy_probs(nnq_attacker, nnq_state)
        attacker_u = float(attacker_rng.random())
        defender_u = float(attacker_rng.random())
        bvi_attacker_action = int(attacker_u < bvi_attacker_probs[1])
        nnq_attacker_action = int(attacker_u < nnq_attacker_probs[1])
        bvi_action = int(defender_u < bvi_probs[1])
        nnq_action = int(defender_u < nnq_probs[1])
        bvi_targets = env.polling_targets(bvi_state, bvi_attacker_action, bvi_action)
        nnq_targets = env.polling_targets(nnq_state, nnq_attacker_action, nnq_action)
        random_u = float(rng.random())
        next_bvi_state = _sample_transition(
            env, bvi_state, bvi_attacker_action, bvi_action, random_u
        )
        if path_mode == "bvi_locked":
            next_nnq_state = next_bvi_state
        else:
            next_nnq_state = _sample_transition(
                env, nnq_state, nnq_attacker_action, nnq_action, random_u
            )
        rows.append(
            {
                "step": step,
                "path_mode": path_mode,
                "state": list(bvi_state),
                "bvi_state": list(bvi_state),
                "nnq_state": list(nnq_state),
                "same_state": bvi_state == nnq_state,
                "next_state": list(next_bvi_state),
                "bvi_next_state": list(next_bvi_state),
                "nnq_next_state": list(next_nnq_state),
                "bvi_attacker_action": bvi_attacker_action,
                "nnq_attacker_action": nnq_attacker_action,
                "bvi_action": bvi_action,
                "nnq_action": nnq_action,
                "bvi_defender_action": bvi_action,
                "nnq_defender_action": nnq_action,
                "bvi_p_attack": float(bvi_attacker_probs[1]),
                "nnq_p_attack": float(nnq_attacker_probs[1]),
                "bvi_p_defend": float(bvi_probs[1]),
                "nnq_p_defend": float(nnq_probs[1]),
                "bvi_attacker_p_attack": float(bvi_attacker_probs[1]),
                "nnq_attacker_p_attack": float(nnq_attacker_probs[1]),
                "attacker_actions_agree": bool(bvi_attacker_action == nnq_attacker_action),
                "defender_actions_agree": bool(bvi_action == nnq_action),
                "actions_agree": bool(
                    bvi_attacker_action == nnq_attacker_action and bvi_action == nnq_action
                ),
                "bvi_polling_targets": list(bvi_targets),
                "nnq_polling_targets": list(nnq_targets),
                "polling_targets": list(bvi_targets),
                "event_label": _event_label(bvi_state, next_bvi_state),
                "bvi_event_label": _event_label(bvi_state, next_bvi_state),
                "nnq_event_label": _event_label(nnq_state, next_nnq_state),
            }
        )
        bvi_state = next_bvi_state
        nnq_state = next_nnq_state
    return rows


def _render_svg(
    rows: list[dict[str, Any]],
    *,
    title: str,
    subtitle: str,
    columns: int,
) -> str:
    left = 138
    col_w = 108
    right = 56
    top = 162
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
        _txt(48, 66, title, 30, weight=800, fill="#26364f"),
        _txt(48, 96, subtitle, 14, fill="#596579"),
        _legend(width - 420, 68),
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
        _render_panel(text, chunk, left, col_w, y0)

    agreements = sum(row["actions_agree"] for row in rows)
    attacker_agreements = sum(row["attacker_actions_agree"] for row in rows)
    defender_agreements = sum(row["defender_actions_agree"] for row in rows)
    bvi_defends = sum(row["bvi_action"] for row in rows)
    nnq_defends = sum(row["nnq_action"] for row in rows)
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
                f"BVI defends {bvi_defends}; NNQ defends {nnq_defends}. "
                f"state=({_state_schema(len(rows[0]['state']))}), target is selected queue."
            ),
            14,
            weight=700,
            fill="#26364f",
        )
    )
    text.append("</svg>")
    return "\n".join(text) + "\n"


def _render_panel(text: list[str], rows: list[dict[str, Any]], left: int, col_w: int, y0: int) -> None:
    labels = [
        ("state", 0),
        ("target", 40),
        ("BVI A", 92),
        ("NNQ A", 148),
        ("BVI D", 204),
        ("NNQ D", 260),
        ("event", 330),
        ("step", 362),
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
        state = tuple(int(value) for value in row["bvi_state"])
        queues = state[:-1]
        state_fill = "#eef3f9" if row["same_state"] else "#fff3d6"
        text.append(f'<circle cx="{x}" cy="{y0}" r="14" fill="{state_fill}" stroke="#cfd8e6"/>')
        text.append(_txt(x, y0 + 5, str(sum(queues)), 12, anchor="middle", weight=800, fill="#26364f"))
        text.append(_txt(x, y0 + 28, _fmt_state(state), 8, anchor="middle", fill="#596579"))
        targets = ",".join(f"q{int(target) + 1}" for target in row["bvi_polling_targets"])
        text.append(_txt(x, y0 + 45, targets, 10, anchor="middle", weight=800, fill="#1769aa"))
        _render_attacker_action(text, x, y0 + 92, int(row["bvi_attacker_action"]), float(row["bvi_p_attack"]))
        _render_attacker_action(text, x, y0 + 148, int(row["nnq_attacker_action"]), float(row["nnq_p_attack"]))
        _render_action(text, x, y0 + 204, int(row["bvi_action"]), float(row["bvi_p_defend"]))
        _render_action(text, x, y0 + 260, int(row["nnq_action"]), float(row["nnq_p_defend"]))
        if not row["actions_agree"]:
            text.append(
                f'<rect x="{x - 26}" y="{y0 + 70}" width="52" height="214" rx="8" '
                'fill="none" stroke="#d92d20" stroke-width="2" stroke-dasharray="4 4"/>'
            )
        event = str(row["bvi_event_label"])
        text.append(_txt(x, y0 + 335, event, 10, anchor="middle", weight=800, fill=_event_color(event)))
        text.append(_txt(x, y0 + 367, str(row["step"]), 10, anchor="middle", weight=800, fill="#8a94a6"))


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
            f'<rect x="{x}" y="{y - 15}" width="24" height="24" rx="5" fill="#2d7d46"/>',
            _txt(x + 34, y + 3, "D = defend", 13, weight=700, fill="#26364f"),
            f'<rect x="{x + 126}" y="{y - 15}" width="24" height="24" rx="5" fill="#d3d9e3" stroke="#9aa5b5"/>',
            _txt(x + 160, y + 3, "N = not defend", 13, weight=700, fill="#26364f"),
            f'<rect x="{x + 294}" y="{y - 15}" width="24" height="24" rx="5" fill="#c2410c"/>',
            _txt(x + 328, y + 3, "A = attack", 13, weight=700, fill="#26364f"),
        ]
    )


def _event_label(state: State, next_state: State) -> str:
    queues = state[:-1]
    next_queues = next_state[:-1]
    diffs = [new - old for old, new in zip(queues, next_queues)]
    position = int(next_state[-1])
    if all(diff == 0 for diff in diffs):
        return f"p=q{position + 1}"
    if sum(diffs) == 1 and diffs.count(1) == 1 and all(diff in (0, 1) for diff in diffs):
        return f"+q{diffs.index(1) + 1},p=q{position + 1}"
    if sum(diffs) == -1 and diffs.count(-1) == 1 and all(diff in (0, -1) for diff in diffs):
        return f"-q{diffs.index(-1) + 1},p=q{position + 1}"
    return "?"


def _event_color(event: str) -> str:
    if event.startswith("+"):
        return "#1769aa"
    if event.startswith("-"):
        return "#6c4ab6"
    if event.startswith("p="):
        return "#8a94a6"
    return "#b42318"


def _sample_transition(
    env: PollingEnv,
    state: State,
    attacker_action: int,
    defender_action: int,
    random_u: float,
) -> State:
    cumulative = 0.0
    items = sorted(env.transition_probabilities(state, attacker_action, defender_action).items())
    for next_state, probability in items:
        cumulative += float(probability)
        if random_u <= cumulative + 1e-12:
            return tuple(int(value) for value in next_state)
    return tuple(int(value) for value in items[-1][0])


def _load_policies(path: Path) -> tuple[Policy, Policy]:
    defender: Policy = {}
    attacker: Policy = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        state = tuple(int(value) for value in row["state"])
        defender[state] = (1.0 - float(row["p_defend"]), float(row["p_defend"]))
        attacker[state] = (float(row["p_no_attack"]), float(row["p_attack"]))
    return defender, attacker


def _policy_probs(policy: Policy, state: State) -> tuple[float, float]:
    if state in policy:
        return policy[state]
    max_values = [max(item[index] for item in policy) for index in range(len(state))]
    clipped = tuple(min(int(value), int(max_values[index])) for index, value in enumerate(state))
    return policy[clipped]


def _greedy(probs: tuple[float, ...]) -> int:
    return int(max(range(len(probs)), key=lambda index: probs[index]))


def _fmt_state(state: State) -> str:
    queues = ",".join(str(int(value)) for value in state[:-1])
    return f"({queues},p={int(state[-1]) + 1})"


def _parse_state(raw: str) -> State:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if len(values) < 3:
        raise ValueError("--initial-state must have form q1,q2,...,position")
    return values


def _state_schema(state_len: int) -> str:
    queue_labels = ",".join(f"q{index + 1}" for index in range(state_len - 1))
    return f"{queue_labels},p"


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
