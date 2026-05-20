# AMQ Experiments

This repository contains the final experiment artifacts for adversarial queueing minimax-Q experiments.

The repository is organized around two report-ready modules:

- `policy_consistency_final/`: BVI vs fitted minimax-DQN / NNQ policy consistency experiments.
- `covergence_speed_final/`: AMQ vs fitted minimax-DQN / NNQ convergence-speed experiments.

The directory name `covergence_speed_final` preserves the original experiment folder name.

## Contents

Each module contains:

- a Chinese report Markdown file;
- source code needed to regenerate the experiment figures or rerun the final pipelines;
- figures used by the report;
- compact result summaries.

Large intermediate full-grid policy row dumps are intentionally not versioned. In particular, convergence-speed per-seed full-grid JSONL policy tables can be hundreds of MB per seed; the repository keeps the aggregate summaries and per-seed `summary.json` files instead.

## Environment

The code is plain Python. The core dependencies are:

```bash
pip install numpy scipy pyyaml
```

Some scripts use the local package under `policy_consistency_final/code/src`, so run them with:

```bash
export PYTHONPATH="$PWD/policy_consistency_final/code/src:$PWD/policy_consistency_final/code:$PYTHONPATH"
```

## Main Reports

- Policy consistency report:
  `policy_consistency_final/policy_consistency_report_zh.md`

- Convergence speed report:
  `covergence_speed_final/convergence_speed_report_zh.md`

## Reproduction Notes

Policy-consistency figures can be regenerated from the scripts under:

```text
policy_consistency_final/code/scripts/
policy_consistency_final/code/experiments/source_faithful_routing_consistency/
```

Convergence-speed experiments can be rerun from:

```text
covergence_speed_final/code/
```

The reports record the exact final experiment definitions, work accounting, policy-consistency metric, and the interpretation of horizon-censored convergence runs.

