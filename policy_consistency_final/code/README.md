# Policy consistency final code bundle

这个目录只收录最终版 policy consistency 报告直接相关的代码与必要输入产物，目的是和历史探索期脚本区分开。

## 目录结构

```text
code/
  README.md
  scripts/
    build_service_rate_policy_path_figure.py
    build_polling_policy_path_figure.py
  experiments/source_faithful_routing_consistency/
    routing_bvi_dqn_consistency.py
    plot_neural_fixed_point_rollout.py
  src/adversarial_queueing/
    algorithms/
    envs/
    evaluation/
    utils/
  configs/
  artifacts/
```

## 文件角色

- `experiments/source_faithful_routing_consistency/routing_bvi_dqn_consistency.py`
  - Routing 的 source-faithful BVI 与 neural fixed-point minimax-Q / fitted minimax-DQN 实现。
  - 包含 2x2 matrix game solver、BVI Bellman iteration、DQN/Q-network parameter representation、routing rollout transition。
- `experiments/source_faithful_routing_consistency/plot_neural_fixed_point_rollout.py`
  - 生成 routing 的 BVI vs fitted minimax-DQN 75-step 决策链路图。
- `scripts/build_service_rate_policy_path_figure.py`
  - 使用 service-rate-control v2 BVI policy grid 与 NNQ Bellman-target matrix game extraction 生成 75-step 决策链路图。
- `scripts/build_service_rate_v2_artifacts.py`
  - 重建 service-rate-control v2 的 BVI/NNQ artifact；旧 LMH setting 仅保存在 `_legacy/`。
- `scripts/build_polling_policy_path_figure.py`
  - 使用三队列 polling BVI/NNQ policy inspection 生成 75-step 决策链路图。
- `src/adversarial_queueing/`
  - 最终图依赖的核心算法、环境、policy extraction、config 解析代码。
- `configs/`
  - service-rate-control 与 polling 三队列相关正式配置。
- `artifacts/`
  - 为避免依赖历史混乱目录，这里复制了最终图需要读取的模型参数、policy grid、policy inspection、q diagnostic 等输入产物。

## 不包含哪些内容

本目录刻意不包含历史探索期代码，例如：

- one-hot policy imitation routing 实验；
- supervised BVI-policy imitation 实验；
- broad classifier probe；
- tabular minimax-Q diagnostic sweep；
- 不进入最终报告的 `(0,20,0)`、`(5,10,18)` 等诊断图生成产物。

这些历史实验可以在原项目目录中追溯，但不属于最终报告的主链路。

## 复现 9 张正式图

下面命令假设当前工作目录是：

```text
/Users/zheqihu/research/minimax_queueing_results_report/policy_consistency_final/code
```

### Routing

```bash
python3 experiments/source_faithful_routing_consistency/plot_neural_fixed_point_rollout.py \
  --initial-state 5,5,5 --seed 3101 --horizon 75 --bound 20 \
  --params-file artifacts/routing/neural_fixed_point_q_structural_b20_seed2_long.dqn_params.npz \
  --svg-output ../figures/routing_bvi_vs_nnq_initial_5_5_5_h75.svg \
  --jsonl-output ../data/routing_bvi_vs_nnq_initial_5_5_5_h75.jsonl

python3 experiments/source_faithful_routing_consistency/plot_neural_fixed_point_rollout.py \
  --initial-state 2,6,10 --seed 3102 --horizon 75 --bound 20 \
  --params-file artifacts/routing/neural_fixed_point_q_structural_b20_seed2_long.dqn_params.npz \
  --svg-output ../figures/routing_bvi_vs_nnq_initial_2_6_10_h75.svg \
  --jsonl-output ../data/routing_bvi_vs_nnq_initial_2_6_10_h75.jsonl

python3 experiments/source_faithful_routing_consistency/plot_neural_fixed_point_rollout.py \
  --initial-state 2,5,12 --seed 3103 --horizon 75 --bound 20 \
  --params-file artifacts/routing/neural_fixed_point_q_structural_b20_seed2_long.dqn_params.npz \
  --svg-output ../figures/routing_bvi_vs_nnq_initial_2_5_12_h75.svg \
  --jsonl-output ../data/routing_bvi_vs_nnq_initial_2_5_12_h75.jsonl
```

### Service-rate-control

```bash
python3 scripts/build_service_rate_v2_artifacts.py

python3 scripts/build_service_rate_policy_path_figure.py \
  --bvi-run-dir artifacts/service_rate_v2/bvi \
  --nnq-run-dir artifacts/service_rate_v2/nnq \
  --initial-state 0 --env-seed 3301 --attacker-seed 1929 \
  --attacker-mode learned_split --horizon 75 --columns 12 \
  --svg-output ../figures/service_rate_v2_bvi_vs_nnq_initial_0_h75.svg \
  --jsonl-output ../data/service_rate_v2_bvi_vs_nnq_initial_0_h75.jsonl

python3 scripts/build_service_rate_policy_path_figure.py \
  --bvi-run-dir artifacts/service_rate_v2/bvi \
  --nnq-run-dir artifacts/service_rate_v2/nnq \
  --initial-state 8 --env-seed 3302 --attacker-seed 1930 \
  --attacker-mode learned_split --horizon 75 --columns 12 \
  --svg-output ../figures/service_rate_v2_bvi_vs_nnq_initial_8_h75.svg \
  --jsonl-output ../data/service_rate_v2_bvi_vs_nnq_initial_8_h75.jsonl

python3 scripts/build_service_rate_policy_path_figure.py \
  --bvi-run-dir artifacts/service_rate_v2/bvi \
  --nnq-run-dir artifacts/service_rate_v2/nnq \
  --initial-state 16 --env-seed 3303 --attacker-seed 1931 \
  --attacker-mode learned_split --horizon 75 --columns 12 \
  --svg-output ../figures/service_rate_v2_bvi_vs_nnq_initial_16_h75.svg \
  --jsonl-output ../data/service_rate_v2_bvi_vs_nnq_initial_16_h75.jsonl
```

### Polling 三队列

```bash
python3 scripts/build_polling_policy_path_figure.py \
  --bvi-run-dir artifacts/polling3/bvi \
  --nnq-run-dir artifacts/polling3/nnq \
  --initial-state 0,17,18,0 --env-seed 7201 --attacker-seed 8201 \
  --horizon 75 --columns 12 \
  --svg-output ../figures/polling3_bvi_vs_nnq_initial_0_17_18_0_h75.svg \
  --jsonl-output ../data/polling3_bvi_vs_nnq_initial_0_17_18_0_h75.jsonl

python3 scripts/build_polling_policy_path_figure.py \
  --bvi-run-dir artifacts/polling3/bvi \
  --nnq-run-dir artifacts/polling3/nnq \
  --initial-state 1,1,4,1 --env-seed 6534 --attacker-seed 2785 \
  --horizon 75 --columns 12 \
  --svg-output ../figures/polling3_bvi_vs_nnq_initial_1_1_4_1_h75.svg \
  --jsonl-output ../data/polling3_bvi_vs_nnq_initial_1_1_4_1_h75.jsonl

python3 scripts/build_polling_policy_path_figure.py \
  --bvi-run-dir artifacts/polling3/bvi \
  --nnq-run-dir artifacts/polling3/nnq \
  --initial-state 1,2,1,0 --env-seed 3021 --attacker-seed 4959 \
  --horizon 75 --columns 12 \
  --svg-output ../figures/polling3_bvi_vs_nnq_initial_1_2_1_0_h75.svg \
  --jsonl-output ../data/polling3_bvi_vs_nnq_initial_1_2_1_0_h75.jsonl
```

## 结果口径

最终报告使用 rollout-level policy consistency：

```text
policy consistency = matched (attacker action, defender action) steps / total steps
```

新版图使用 probability-aware sampled rollout。Routing 与 service-rate-control v2 三条轨迹均为 `75/75`；polling 因 defender 也改为 mixed-policy sampling，有一条轨迹会暴露明显概率差距。

```text
overall = 654/675 = 96.89%
```
