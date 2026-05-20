# Convergence Speed Final Workspace

This directory contains the cleaned final workspace for the revised convergence-speed experiments.

It is a report workspace, not a standalone Python package. The scripts depend on
the source checkout at:

```text
/Users/zheqihu/research/minimax_queueing_experiments
```

In particular, routing imports the finalized policy-consistency implementation
from `experiments/source_faithful_routing_consistency/routing_bvi_dqn_consistency.py`,
and polling/service-rate import `adversarial_queueing` from the same checkout.

Current focus:

- AMQ follows the paper-form linear minimax-Q update.
- NNQ/DQN is represented by the benchmark-specific fitted minimax-DQN used in
  the final policy-consistency work.
- Stabilization is measured against each method's own final policy.
- Policy distance compares both attacker and defender.

For routing, the selected DQN path is Scheme B: policy-consistency-faithful
`neural_fixed_point_q`. Its checkpoint is the number of neural fitting epochs
after computing the model-based minimax-Q fixed point. It is not comparable to
an AMQ sampled transition index as a raw "step"; use `work_to_stable`, defined
as cumulative effective Bellman/target evaluations, for cross-method compute
comparisons. Runtime is auxiliary only and is not used for the main conclusion.

The pre-registered work accounting schema is:

```text
docs/work_accounting_schema.json
```

## Current Routing Commands

Single-seed routing run:

```bash
.venv/bin/python /Users/zheqihu/research/minimax_queueing_results_report/covergence_speed_final/code/routing_convergence_speed.py \
  --bound 20 \
  --seed 0 \
  --amq-feature-set amq2 \
  --amq-eta0 1e-6 \
  --amq-checkpoints 1,2,5,10,20,50,100,200,500,1000,2000,5000 \
  --dqn-hidden-size 256 \
  --dqn-feature-set structural \
  --dqn-checkpoints 1,2,5,10,20,50,100,200,500,1000,1500 \
  --output-dir /Users/zheqihu/research/minimax_queueing_results_report/covergence_speed_final/results/routing_b20_seed0
```

Aggregate completed routing summaries:

```bash
.venv/bin/python /Users/zheqihu/research/minimax_queueing_results_report/covergence_speed_final/code/build_routing_convergence_summary.py \
  /Users/zheqihu/research/minimax_queueing_results_report/covergence_speed_final/results/routing_b20_seed0/summary.json \
  /Users/zheqihu/research/minimax_queueing_results_report/covergence_speed_final/results/routing_b20_seed1/summary.json \
  /Users/zheqihu/research/minimax_queueing_results_report/covergence_speed_final/results/routing_b20_seed2/summary.json \
  /Users/zheqihu/research/minimax_queueing_results_report/covergence_speed_final/results/routing_b20_seed3/summary.json \
  /Users/zheqihu/research/minimax_queueing_results_report/covergence_speed_final/results/routing_b20_seed4/summary.json
```

## Directory Layout

- `code/`: final experiment scripts.
- `docs/`: Chinese experiment plans and progress reports.
- `figures/`: report-ready figures.
- `results/`: JSON summaries and policy rows.

`results/deprecated_pre_cursor_alignment/` contains outputs generated before
the routing DQN and AMQ feature definitions were aligned with the final protocol.
Those files are retained only for traceability and should not be used in final
convergence-speed conclusions.
