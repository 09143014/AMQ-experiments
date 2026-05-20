# Routing Convergence Speed 阶段性记录

本文档记录 Cursor 建议对齐之后的新版 routing convergence speed 工作。此前基于泛二阶 AMQ 特征和 iterative fitted Bellman-Q 的 routing 输出已经移入：

`../results/deprecated_pre_cursor_alignment/`

这些旧结果只保留作溯源，不进入最终结论。

## 1. 当前锁定口径

### AMQ

Routing AMQ 使用 AMQ 论文 Appendix A.1 的 AMQ2 特征，主表预注册为 AMQ2：

```text
phi_{i,1}(x,a,b) = 1
phi_{i,2}(x,a,b) = x_i + delta_i(x,a,b)
phi_{i,3}(x,a,b) = (x_i + delta_i(x,a,b))^2
phi_{i,4}(x,a,b) = a
phi_{i,5}(x,a,b) = b
```

其中 `i` 是 server / queue index。`delta_i(x,a,b)` 的定义与 routing 规则一致：

- 如果 `(a,b)=(attack, not defend)`，则标记最长队列。
- 否则标记最短队列。

训练过程为论文 Algorithm 1 的在线 AMQ：

1. 从状态依赖 behavior policy `alpha(.|x), beta(.|x)` 采样 attacker / defender action。
2. 执行动作，得到 cost 和 next state。
3. 在 next state 上求解 2x2 minimax game，得到 continuation value。
4. 用 Robbins-Monro 步长更新线性权重 `w`。

当前 routing smoke 使用：

```text
eta_k = 1e-6 / k^0.6
```

原因是未归一化 AMQ2 二次特征在 `B=20` 上数值较大，较大的 `eta0` 会导致权重爆炸。这里没有使用 weight clipping，也没有使用 fitted calibration。

### fitted minimax-DQN

Routing DQN 使用 policy consistency 定稿的 `neural_fixed_point_q` 口径：

- 先通过 model-based minimax-Q iteration 求 fixed-point Q target。
- 再用两层 ReLU MLP 拟合每个状态的 2x2 Q matrix。
- 网络输入为 `structural` queue features。
- 网络输出为 4 个 Q entries。
- 训练 target 按状态减去 2x2 Q matrix 的均值，只拟合 action-dependent gap。
- 不使用 BVI policy label，不使用 AMQ label，不使用 one-hot state lookup。

当前 routing smoke 使用：

```text
hidden_size = 256
architecture = standard
batch_size = 1024
learning_rate = 0.0007
loss_type = mse
grad_clip_norm = 20
reward_scale = denominator
```

## 2. Checkpoint 轴

Routing 采用方案 B：严格使用 policy consistency 定稿 DQN。

因此：

- AMQ checkpoint = online sampled transition/update index。
- DQN checkpoint = neural fitting epoch。

这两个 checkpoint 不是同一种 step，不能直接写 “AMQ step / DQN step ratio”。跨方法主比较必须看：

- `work_to_stable`：达到稳定点时累计的 effective Bellman / target evaluations。
- 各自相对自身 final policy 的 stabilization curve。

runtime 只保留为辅助实现信息，不作为主结论证据。

## 3. 当前 smoke 结果

### B=5 smoke

路径：

`../results/routing_cursor_aligned_smoke_b5_seed0/summary.json`

该 smoke 验证了：

- AMQ 参数维度为 `15 = 3 queues * 5 AMQ2 features`。
- DQN 参数是 neural network，非表格策略。
- joint gap 同时比较 attacker 和 defender。

### B=20 smoke

路径：

`../results/routing_cursor_aligned_smoke_b20_eta1e6_seed0/summary.json`

该 smoke 验证了：

- B=20 下 AMQ2 不使用 clipping 可以稳定运行，但需要较小 `eta0`。
- DQN fixed-point target 计算是主要耗时来源。
- 当前 smoke 只使用 `eval_state_limit=512`，用于检查流程；正式实验应改为全 bounded grid，即 `21^3 = 9261` 个状态。

## 4. Full-grid 10-seed routing 结果

正式 full-grid routing 已完成 10 个 seed：

```text
seeds = 0, 1, ..., 9
evaluation grid = 21^3 = 9261 states
```

结果文件：

`../results/routing_b20_10seed_summary.json`

图：

![Routing B20 convergence speed](../figures/routing_b20_10seed_convergence_speed.svg)

聚合结果如下：

| Method | Mean native stable checkpoint | Median native stable checkpoint | Mean work_to_stable | Runtime status |
|---|---:|---:|---:|---|
| AMQ2 | 440.10 online updates | 350 online updates | 440.10 | auxiliary only |
| fitted minimax-DQN | 510 fitting epochs | 500 fitting epochs | 8,557,164 | auxiliary only |

逐 seed 结果：

| Seed | AMQ stable update | AMQ work_to_stable | DQN stable epoch | DQN work_to_stable | DQN censored |
|---:|---:|---:|---:|---:|---|
| 0 | 100 | 100 | 500 | 8,557,164 | no |
| 1 | 50 | 50 | 200 | 8,557,164 | no |
| 2 | 500 | 500 | 100 | 8,557,164 | no |
| 3 | 500 | 500 | 500 | 8,557,164 | no |
| 4 | 1 | 1 | 1000 | 8,557,164 | no |
| 5 | 1000 | 1000 | 100 | 8,557,164 | no |
| 6 | 50 | 50 | 200 | 8,557,164 | no |
| 7 | 200 | 200 | 500 | 8,557,164 | no |
| 8 | 1000 | 1000 | 1500 | 8,557,164 | yes |
| 9 | 1000 | 1000 | 500 | 8,557,164 | no |

解释时需要非常谨慎：

- AMQ 的 checkpoint 是 online sampled update。
- DQN 的 checkpoint 是 fixed-point target 之后的 neural fitting epoch。
- 因此不能说 “AMQ step 比 DQN step 少/多多少倍”。
- 从 `work_to_stable` 看，routing 上 AMQ2 明显更快达到自身 policy stabilization。
- DQN 的 neural fitting target entries/gradient steps 单独作为 secondary work，不计入主 Bellman work。
- 10-seed 中 DQN 有 1/10 个 seed 在 final fitting epoch 才满足稳定判据，应按 budget ceiling 解读。
- DQN secondary fitting target entries at stable 的均值为 18,892,440，中位数为 18,522,000。

## 5. 下一步

Routing 10-seed 已完成，可纳入 final 报告。后续若继续增强，主要是报告层面的图注与 limitations，而不是 routing pipeline 本身。
