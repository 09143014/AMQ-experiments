#!/usr/bin/env python3
"""Aggregate routing convergence-speed summaries and draw a compact SVG."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


DEFAULT_RESULTS_DIR = (
    Path("/Users/zheqihu/research/minimax_queueing_results_report")
    / "covergence_speed_final"
    / "results"
)
DEFAULT_OUTPUT_JSON = DEFAULT_RESULTS_DIR / "routing_b20_5seed_summary.json"
DEFAULT_OUTPUT_SVG = (
    Path("/Users/zheqihu/research/minimax_queueing_results_report")
    / "covergence_speed_final"
    / "figures"
    / "routing_b20_convergence_speed.svg"
)


def _work_to_checkpoint(snapshot: dict[str, Any]) -> float:
    if "work_to_checkpoint" in snapshot:
        return float(snapshot["work_to_checkpoint"])
    if "primary_work_to_checkpoint" in snapshot:
        return float(snapshot["primary_work_to_checkpoint"])
    if "fixed_point_bellman_backups" in snapshot:
        return float(snapshot["fixed_point_bellman_backups"])
    if "effective_target_updates" in snapshot:
        return float(snapshot["effective_target_updates"])
    if "effective_bellman_backups" in snapshot:
        return float(snapshot["effective_bellman_backups"])
    return float(snapshot.get("num_gradient_updates", 0))


def _secondary_fitting_work(snapshot: dict[str, Any]) -> float:
    return float(snapshot.get("secondary_fitting_target_entries", 0))


def _stable_metadata(stabilization: dict[str, Any]) -> tuple[int | None, int, int, bool, bool]:
    final_checkpoint = int(stabilization["final_checkpoint"])
    raw_stable = stabilization.get("stable_checkpoint")
    stable_checkpoint = None if raw_stable is None else int(raw_stable)
    accounting_checkpoint = stable_checkpoint if stable_checkpoint is not None else final_checkpoint
    stable_before_horizon = bool(
        stabilization.get(
            "stable_before_horizon",
            stable_checkpoint is not None and stable_checkpoint < final_checkpoint,
        )
    )
    censored_at_horizon = bool(
        stabilization.get(
            "censored_at_horizon",
            stable_checkpoint is None or stable_checkpoint == final_checkpoint,
        )
    )
    return stable_checkpoint, accounting_checkpoint, final_checkpoint, stable_before_horizon, censored_at_horizon


def load_summaries(paths: list[Path]) -> list[dict[str, Any]]:
    summaries = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        summaries.append(data)
    return summaries


def aggregate(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    seed_rows = []
    curves: dict[str, dict[int, list[dict[str, float]]]] = {
        "amq": defaultdict(list),
        "dqn": defaultdict(list),
    }
    for data in summaries:
        seed = int(data["amq_config"]["seed"])
        row = {"seed": seed}
        for method in ("amq", "dqn"):
            stabilization = data[f"{method}_stabilization"]
            stable, accounting_checkpoint, final_checkpoint, stable_before_horizon, censored_at_horizon = (
                _stable_metadata(stabilization)
            )
            snapshot = data[f"{method}_snapshots"][str(accounting_checkpoint)]
            row[f"{method}_stable_checkpoint"] = stable
            row[f"{method}_accounting_checkpoint"] = accounting_checkpoint
            row[f"{method}_final_checkpoint"] = final_checkpoint
            row[f"{method}_stable_before_horizon"] = stable_before_horizon
            row[f"{method}_censored_at_horizon"] = censored_at_horizon
            row[f"{method}_not_stabilized_within_horizon"] = stable is None
            row[f"{method}_stable_elapsed_seconds"] = float(snapshot["elapsed_seconds"])
            row[f"{method}_stable_work_to_stable"] = int(_work_to_checkpoint(snapshot))
            row[f"{method}_stable_secondary_fitting_target_entries"] = int(
                _secondary_fitting_work(snapshot)
            )
            for item in stabilization["checkpoints"]:
                checkpoint = int(item["checkpoint"])
                snap = data[f"{method}_snapshots"][str(checkpoint)]
                curves[method][checkpoint].append(
                    {
                        "joint_gap": float(item["joint_gap"]),
                        "policy_similarity_percent": float(item["policy_similarity_percent"]),
                        "elapsed_seconds": float(snap["elapsed_seconds"]),
                        "work_to_checkpoint": _work_to_checkpoint(snap),
                    }
                )
        seed_rows.append(row)

    curve_rows: dict[str, list[dict[str, float]]] = {"amq": [], "dqn": []}
    for method in ("amq", "dqn"):
        for checkpoint, entries in sorted(curves[method].items()):
            curve_rows[method].append(
                {
                    "checkpoint": checkpoint,
                    "mean_joint_gap": mean(entry["joint_gap"] for entry in entries),
                    "mean_policy_similarity_percent": mean(
                        entry["policy_similarity_percent"] for entry in entries
                    ),
                    "mean_elapsed_seconds": mean(entry["elapsed_seconds"] for entry in entries),
                    "mean_work_to_checkpoint": mean(entry["work_to_checkpoint"] for entry in entries),
                }
            )

    aggregate_rows = {}
    for method in ("amq", "dqn"):
        stable_checkpoints = [row[f"{method}_accounting_checkpoint"] for row in seed_rows]
        stable_seconds = [row[f"{method}_stable_elapsed_seconds"] for row in seed_rows]
        stable_work = [row[f"{method}_stable_work_to_stable"] for row in seed_rows]
        stable_secondary_fitting = [
            row[f"{method}_stable_secondary_fitting_target_entries"] for row in seed_rows
        ]
        aggregate_rows[method] = {
            "mean_stable_checkpoint": mean(stable_checkpoints),
            "median_stable_checkpoint": median(stable_checkpoints),
            "auxiliary_mean_stable_elapsed_seconds": mean(stable_seconds),
            "auxiliary_median_stable_elapsed_seconds": median(stable_seconds),
            "mean_work_to_stable": mean(stable_work),
            "median_work_to_stable": median(stable_work),
            "mean_secondary_fitting_target_entries_at_stable": mean(stable_secondary_fitting),
            "median_secondary_fitting_target_entries_at_stable": median(stable_secondary_fitting),
            "stable_before_horizon_count": sum(
                bool(row[f"{method}_stable_before_horizon"]) for row in seed_rows
            ),
            "censored_at_horizon_count": sum(
                bool(row[f"{method}_censored_at_horizon"]) for row in seed_rows
            ),
            "not_stabilized_within_horizon_count": sum(
                bool(row[f"{method}_not_stabilized_within_horizon"]) for row in seed_rows
            ),
        }
        # Backward-compatible aliases for older notes.
        aggregate_rows[method]["mean_stable_elapsed_seconds"] = aggregate_rows[method][
            "auxiliary_mean_stable_elapsed_seconds"
        ]
        aggregate_rows[method]["median_stable_elapsed_seconds"] = aggregate_rows[method][
            "auxiliary_median_stable_elapsed_seconds"
        ]
        aggregate_rows[method]["mean_stable_effective_bellman_backups"] = aggregate_rows[method][
            "mean_work_to_stable"
        ]
        aggregate_rows[method]["median_stable_effective_bellman_backups"] = aggregate_rows[method][
            "median_work_to_stable"
        ]

    amq_work = aggregate_rows["amq"]["mean_work_to_stable"]
    dqn_work = aggregate_rows["dqn"]["mean_work_to_stable"]
    amq_median_work = aggregate_rows["amq"]["median_work_to_stable"]
    dqn_median_work = aggregate_rows["dqn"]["median_work_to_stable"]
    work_ratio = (dqn_work / amq_work) if amq_work else None
    median_work_ratio = (dqn_median_work / amq_median_work) if amq_median_work else None
    per_seed_work_ratios = [
        {
            "seed": row["seed"],
            "dqn_over_amq": (
                row["dqn_stable_work_to_stable"] / row["amq_stable_work_to_stable"]
                if row["amq_stable_work_to_stable"]
                else None
            ),
        }
        for row in sorted(seed_rows, key=lambda item: item["seed"])
    ]
    seed_win_counts = {
        "amq_less_work": sum(
            row["amq_stable_work_to_stable"] < row["dqn_stable_work_to_stable"]
            for row in seed_rows
        ),
        "dqn_less_work": sum(
            row["dqn_stable_work_to_stable"] < row["amq_stable_work_to_stable"]
            for row in seed_rows
        ),
        "tie": sum(
            row["dqn_stable_work_to_stable"] == row["amq_stable_work_to_stable"]
            for row in seed_rows
        ),
    }

    return {
        "benchmark": summaries[0].get("benchmark", "routing") if summaries else "unknown",
        "num_seeds": len(seed_rows),
        "seed_rows": sorted(seed_rows, key=lambda row: row["seed"]),
        "aggregate": aggregate_rows,
        "work_ratio_dqn_over_amq": work_ratio,
        "median_work_ratio_dqn_over_amq": median_work_ratio,
        "per_seed_work_ratios_dqn_over_amq": per_seed_work_ratios,
        "seed_win_counts_by_work": seed_win_counts,
        "curves": curve_rows,
        "note": (
            "Main speed metric is work_to_stable: cumulative effective Bellman/target "
            "evaluations at the first sustained self-stabilizing checkpoint. Runtime "
            "is recorded only as auxiliary implementation metadata."
        ),
    }


def draw_svg(summary: dict[str, Any]) -> str:
    width, height = 1200, 760
    margin = 90
    panel_w = 1000
    panel_h = 390
    colors = {"amq": "#1b7f5a", "dqn": "#b23b3b"}
    label = {"amq": "AMQ", "dqn": "fitted minimax-DQN"}

    all_work = [
        row["mean_work_to_checkpoint"]
        for method in ("amq", "dqn")
        for row in summary["curves"][method]
    ]
    min_log = math.log10(max(1.0, min(all_work)))
    max_log = math.log10(max(all_work))

    def x_scale(work: float) -> float:
        if max_log == min_log:
            return margin
        return margin + (math.log10(max(1.0, work)) - min_log) / (max_log - min_log) * panel_w

    def y_scale(similarity: float) -> float:
        return margin + 28 + (100.0 - similarity) / 100.0 * panel_h

    def fmt(value: float | int) -> str:
        if abs(float(value)) >= 1000:
            return f"{float(value):,.0f}"
        if abs(float(value) - round(float(value))) < 1e-9:
            return f"{float(value):,.0f}"
        return f"{float(value):,.2f}"

    chart_top = margin + 46
    chart_bottom = chart_top + panel_h
    chart_right = margin + panel_w

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfcfe"/>',
        f'<text x="64" y="46" font-family="Arial, sans-serif" font-size="28" font-weight="700" fill="#172033">{summary["benchmark"].title()} convergence speed</text>',
        f'<text x="64" y="74" font-family="Arial, sans-serif" font-size="14" fill="#536173">AMQ vs fitted minimax-DQN; {summary["num_seeds"]} seed(s); self-final joint policy gap.</text>',
        f'<line x1="{margin}" y1="{chart_bottom}" x2="{chart_right}" y2="{chart_bottom}" stroke="#b8c2d0"/>',
        f'<line x1="{margin}" y1="{chart_top}" x2="{margin}" y2="{chart_bottom}" stroke="#b8c2d0"/>',
        f'<text x="{margin}" y="{chart_top - 22}" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#172033">Policy stabilization curve</text>',
        f'<text x="{margin + 230}" y="{chart_bottom + 34}" font-family="Arial, sans-serif" font-size="13" fill="#536173">cumulative effective Bellman / target work (log scale)</text>',
        f'<text x="24" y="{chart_top + 250}" font-family="Arial, sans-serif" font-size="13" fill="#536173" transform="rotate(-90 24 {chart_top + 250})">policy similarity (%)</text>',
    ]
    for y in (0, 25, 50, 75, 100):
        yy = y_scale(y)
        svg.append(f'<line x1="{margin-5}" y1="{yy:.1f}" x2="{chart_right}" y2="{yy:.1f}" stroke="#e4e9f0"/>')
        svg.append(f'<text x="{margin-44}" y="{yy+4:.1f}" font-family="Arial, sans-serif" font-size="12" fill="#536173">{y}</text>')

    for method in ("amq", "dqn"):
        points = [
            (
                x_scale(row["mean_work_to_checkpoint"]),
                y_scale(row["mean_policy_similarity_percent"]),
            )
            for row in summary["curves"][method]
        ]
        path = " ".join(
            ("M" if index == 0 else "L") + f" {x:.1f} {y:.1f}"
            for index, (x, y) in enumerate(points)
        )
        svg.append(f'<path d="{path}" fill="none" stroke="{colors[method]}" stroke-width="3"/>')
        for x, y in points:
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{colors[method]}"/>')
    svg.extend(
        [
            f'<rect x="{chart_right - 272}" y="{chart_top - 36}" width="14" height="14" fill="{colors["amq"]}"/>',
            f'<text x="{chart_right - 252}" y="{chart_top - 24}" font-family="Arial, sans-serif" font-size="13" fill="#172033">AMQ</text>',
            f'<rect x="{chart_right - 170}" y="{chart_top - 36}" width="14" height="14" fill="{colors["dqn"]}"/>',
            f'<text x="{chart_right - 150}" y="{chart_top - 24}" font-family="Arial, sans-serif" font-size="13" fill="#172033">fitted minimax-DQN</text>',
        ]
    )

    agg = summary["aggregate"]
    ratio = summary.get("work_ratio_dqn_over_amq")
    median_ratio = summary.get("median_work_ratio_dqn_over_amq")
    mean_ratio_text = f"{ratio:,.1f}x" if ratio is not None else "n/a"
    median_ratio_text = f"{median_ratio:,.1f}x" if median_ratio is not None else "n/a"

    censored_amq = int(agg["amq"].get("censored_at_horizon_count", 0))
    censored_dqn = int(agg["dqn"].get("censored_at_horizon_count", 0))
    summary_y = chart_bottom + 78
    col_x = [margin + 28, margin + 260, margin + 492, margin + 724]
    svg.extend(
        [
            f'<rect x="{margin}" y="{summary_y - 32}" width="{panel_w}" height="118" rx="6" fill="#ffffff" stroke="#d8e0ea"/>',
            f'<text x="{col_x[0]}" y="{summary_y}" font-family="Arial, sans-serif" font-size="12" font-weight="700" fill="#536173">AMQ mean work</text>',
            f'<text x="{col_x[0]}" y="{summary_y + 30}" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="{colors["amq"]}">{fmt(agg["amq"]["mean_work_to_stable"])}</text>',
            f'<text x="{col_x[1]}" y="{summary_y}" font-family="Arial, sans-serif" font-size="12" font-weight="700" fill="#536173">DQN mean work</text>',
            f'<text x="{col_x[1]}" y="{summary_y + 30}" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="{colors["dqn"]}">{fmt(agg["dqn"]["mean_work_to_stable"])}</text>',
            f'<text x="{col_x[2]}" y="{summary_y}" font-family="Arial, sans-serif" font-size="12" font-weight="700" fill="#536173">Mean ratio</text>',
            f'<text x="{col_x[2]}" y="{summary_y + 30}" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#172033">{mean_ratio_text}</text>',
            f'<text x="{col_x[3]}" y="{summary_y}" font-family="Arial, sans-serif" font-size="12" font-weight="700" fill="#536173">Median ratio</text>',
            f'<text x="{col_x[3]}" y="{summary_y + 30}" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#172033">{median_ratio_text}</text>',
            f'<line x1="{margin + 22}" y1="{summary_y + 48}" x2="{margin + panel_w - 22}" y2="{summary_y + 48}" stroke="#e4e9f0"/>',
            f'<text x="{col_x[0]}" y="{summary_y + 76}" font-family="Arial, sans-serif" font-size="13" fill="#536173">Budget ceiling: AMQ {censored_amq}/{summary["num_seeds"]}; DQN {censored_dqn}/{summary["num_seeds"]}</text>',
            f'<text x="{margin}" y="{height - 42}" font-family="Arial, sans-serif" font-size="12" fill="#536173">Stable condition: joint policy gap &lt;= 0.05 and remains stable. Runtime is auxiliary; native checkpoints are method-specific.</text>',
        ]
    )
    svg.append("</svg>")
    return "\n".join(svg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-svg", type=Path, default=DEFAULT_OUTPUT_SVG)
    args = parser.parse_args()
    summary = aggregate(load_summaries(args.summaries))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    args.output_svg.write_text(draw_svg(summary), encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
