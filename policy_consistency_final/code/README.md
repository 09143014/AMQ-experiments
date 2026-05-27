# Policy consistency final code bundle

这个目录只收录最终版 policy consistency 报告直接相关的代码与必要输入产物，目的是和历史探索期脚本区分开。

## 目录结构

```text
code/
  README.md
  scripts/
    build_polling3_fitted_dqn_artifact.py
    build_polling_policy_path_figure.py
    build_service_rate_policy_path_figure.py
    build_service_rate_v3_candidate_artifacts.py
    evaluate_polling_rollout_panel.py
    evaluate_service_rate_rollout_panel.py
    validate_polling_chain.py
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
  - 使用 service-rate-control BVI policy grid 与 fitted minimax-DQN matrix-game extraction 生成 75-step 决策链路图。
- `scripts/build_service_rate_v3_candidate_artifacts.py`
  - 重建 service-rate-control v3 candidate 的 BVI 与 fitted minimax-DQN artifact；旧 LMH setting 与低活动 v2 仅作为诊断对照保留。
- `scripts/evaluate_service_rate_rollout_panel.py`
  - 对 service-rate-control v3 candidate 做多 initial、多 seed rollout panel 验证，避免只看三张展示图。
- `scripts/build_polling_policy_path_figure.py`
  - 使用三队列 polling BVI/NNQ policy inspection 生成 75-step 决策链路图。
  - 默认 `--path-mode coupled` 会让 BVI 与 DQN state path 各自演化；可用 `--path-mode bvi_locked` 做 same-state diagnostic，避免早期 mismatch 的路径级联放大。
- `scripts/build_polling3_fitted_dqn_artifact.py`
  - 重建 polling 三队列 solver-fixed BVI 与 fitted minimax-DQN artifact。
  - 正式 polling 结果不再使用旧 online NNQ artifact；旧版本保存在 `artifacts/_legacy/polling3_online_nnq/`。
- `scripts/validate_polling_chain.py`
  - 校验 polling vectorized BVI target 是否与 `PollingEnv` Bellman backup 一致，并校验 vectorized 2x2 solver 是否与统一 `minimax_solver` 一致。
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
/Users/zheqihu/repositories/AMQ-experiments/policy_consistency_final/code
```

建议先运行 polling 链路 sanity check：

```bash
python3 scripts/validate_polling_chain.py
```

当前期望输出约为：

```text
polling payoff max gap: 1.137e-13
2x2 solver max gap: 4.441e-16
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
python3 scripts/build_service_rate_v3_candidate_artifacts.py

python3 scripts/build_service_rate_policy_path_figure.py \
  --bvi-run-dir artifacts/service_rate_v3_candidate/bvi \
  --nnq-run-dir artifacts/service_rate_v3_candidate/dqn \
  --initial-state 0 --env-seed 301 --attacker-seed 1729 \
  --attacker-mode learned_split --horizon 75 --columns 12 \
  --svg-output ../figures/service_rate_v3_bvi_vs_dqn_initial_0_h75.svg \
  --jsonl-output ../data/service_rate_v3_bvi_vs_dqn_initial_0_h75.jsonl

python3 scripts/build_service_rate_policy_path_figure.py \
  --bvi-run-dir artifacts/service_rate_v3_candidate/bvi \
  --nnq-run-dir artifacts/service_rate_v3_candidate/dqn \
  --initial-state 8 --env-seed 302 --attacker-seed 1730 \
  --attacker-mode learned_split --horizon 75 --columns 12 \
  --svg-output ../figures/service_rate_v3_bvi_vs_dqn_initial_8_h75.svg \
  --jsonl-output ../data/service_rate_v3_bvi_vs_dqn_initial_8_h75.jsonl

python3 scripts/build_service_rate_policy_path_figure.py \
  --bvi-run-dir artifacts/service_rate_v3_candidate/bvi \
  --nnq-run-dir artifacts/service_rate_v3_candidate/dqn \
  --initial-state 16 --env-seed 303 --attacker-seed 1731 \
  --attacker-mode learned_split --horizon 75 --columns 12 \
  --svg-output ../figures/service_rate_v3_bvi_vs_dqn_initial_16_h75.svg \
  --jsonl-output ../data/service_rate_v3_bvi_vs_dqn_initial_16_h75.jsonl

python3 scripts/evaluate_service_rate_rollout_panel.py \
  --json-output ../data/_diagnostic/service_rate_v3_candidate_panel.json
```

### Polling 三队列

```bash
python3 scripts/build_polling3_fitted_dqn_artifact.py \
  --feature-set polling_bucket_onehot_v1 \
  --hidden-size 512 \
  --epochs 5000 \
  --batch-size 8192 \
  --seed 3 \
  --last-layer-refit

python3 scripts/build_polling_policy_path_figure.py \
  --bvi-run-dir artifacts/polling3/bvi \
  --nnq-run-dir artifacts/polling3/dqn \
  --initial-state 0,20,0,0 --env-seed 301 --attacker-seed 1729 \
  --horizon 75 --columns 12 \
  --svg-output ../figures/polling3_bvi_vs_nnq_initial_0_20_0_0_h75.svg \
  --jsonl-output ../data/polling3_bvi_vs_nnq_initial_0_20_0_0_h75.jsonl

python3 scripts/build_polling_policy_path_figure.py \
  --bvi-run-dir artifacts/polling3/bvi \
  --nnq-run-dir artifacts/polling3/dqn \
  --initial-state 1,1,16,0 --env-seed 301 --attacker-seed 1729 \
  --horizon 75 --columns 12 \
  --svg-output ../figures/polling3_bvi_vs_nnq_initial_1_1_16_0_h75.svg \
  --jsonl-output ../data/polling3_bvi_vs_nnq_initial_1_1_16_0_h75.jsonl

python3 scripts/build_polling_policy_path_figure.py \
  --bvi-run-dir artifacts/polling3/bvi \
  --nnq-run-dir artifacts/polling3/dqn \
  --initial-state 1,15,20,0 --env-seed 303 --attacker-seed 1731 \
  --horizon 75 --columns 12 \
  --svg-output ../figures/polling3_bvi_vs_nnq_initial_1_15_20_0_h75.svg \
  --jsonl-output ../data/polling3_bvi_vs_nnq_initial_1_15_20_0_h75.jsonl
```

如果需要诊断 polling coupled rollout 中的路径分叉放大效应，可在上述命令加入：

```bash
--path-mode bvi_locked
```

并将输出写到 `../figures/_diagnostic/` 与 `../data/_diagnostic/`。该模式让 BVI 与 DQN 始终在同一条 BVI-driven state path 上比较策略，不替代正式 coupled rollout，只用于解释同一 state 上的 policy 是否接近。

正式 polling 验收还使用 broad aggressive panel，而不是只看上述 3 条展示轨迹：

```bash
python3 scripts/evaluate_polling_rollout_panel.py \
  --json-output ../data/_diagnostic/polling_panel_bucket_h512_last_layer_lstsq.json

python3 scripts/evaluate_polling_rollout_panel.py \
  --path-mode bvi_locked \
  --json-output ../data/_diagnostic/polling_panel_bucket_h512_last_layer_lstsq_bvi_locked.json
```

该 panel 包含 10 组 aggressive initial states、5 组随机种子对、每条 75 steps，共 3750 个 sampled joint-action decisions。

## 结果口径

最终报告使用 rollout-level policy consistency：

```text
policy consistency = matched (attacker action, defender action) steps / total steps
```

新版图使用 probability-aware sampled rollout。Routing 与 service-rate-control v3 candidate 三条轨迹均为 `75/75`；service-rate-control v3 candidate 不再是低活动 preliminary extension：BVI 在 bounded grid `0..20` 上有 `20/21` 个 state 非平凡，三条展示轨迹分别有大量 attack/defend 动作，并且 7 initial states x 5 seed pairs x 75 steps 的 panel 达到 `2625/2625`。

service-rate benchmark 的 v3 candidate 只改变环境参数，不能改变 BVI 算法本身。BVI 必须继续采用论文式 bounded value iteration / adapted Shapley 口径：bounded state grid、同步 Bellman backup、每个 state 构造 local zero-sum matrix game，并由同一个 minimax solver 提取 attacker/defender equilibrium policy。DQN 侧也保持 fitted minimax-DQN：state features -> neural network -> Q matrix -> matrix-game solver -> attacker/defender policy。

Polling 使用 solver-fixed fitted minimax-DQN，正式 artifact 为 `polling_bucket_onehot_v1` features、hidden size 512、5000 fitting epochs，并对最后一层做 least-squares refit。DQN 的训练 target 由独立的 model-based minimax-Q fixed point 产生，不使用 BVI 收敛的 value table 或 BVI policy labels。三条 polling 展示 rollout 从 broad aggressive panel 中选择 action-rich 展示样例，均为 `75/75`：`(0,20,0,p=0)` 有 `26` 次 attack 与 `15` 次 defend，`(1,1,16,p=0)` 有 `13` 次 attack 与 `11` 次 defend，`(1,15,20,p=0)` 有 `5` 次 attack 与 `7` 次 defend。

```text
overall display rollouts = 675/675 = 100.00%
polling broad panel = 3706/3750 = 98.83%
polling broad panel, bvi_locked diagnostic = 3717/3750 = 99.12%
polling full-grid probability similarity = 99.60%
service-rate v3 display rollouts = 225/225 = 100.00%
service-rate v3 panel = 2625/2625 = 100.00%
service-rate v3 active-step panel = 2181/2181 = 100.00%
service-rate v3 nontrivial BVI states = 20/21
service-rate v3 full-grid max policy gap = 8.20e-05
```

Polling 的 broad panel 是最终验收口径；三张图用于展示具体路径与每一步的 `pA` / `pD` 概率。
