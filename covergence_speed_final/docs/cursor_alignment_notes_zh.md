# Cursor 建议采纳记录

本记录用于固定 convergence speed 后续执行口径，避免再次混用旧 NNQ、泛化 DQN 或非论文 AMQ 特征。

## 1. DQN 不再写成一套通用算法

后续所有 convergence speed 实验中的 DQN 都应使用各 benchmark 在 policy consistency 定稿中采用的版本。

| Benchmark | DQN 口径 | 备注 |
|---|---|---|
| Routing | `neural_fixed_point_q` | model-based minimax-Q fixed point target + MLP fitting |
| Polling | policy consistency artifact 中的 fitted/full-action NNQ/DQN config | 迁移时逐字段复制 artifact config |
| Service-rate-control | policy consistency artifact 中的 service-rate DQN config | service-rate 是 extension benchmark |

不得从旧 convergence speed JSON 或 `NNQTrainer` 默认参数反推新版 DQN。

## 2. Routing 采用 checkpoint 方案 B

Routing DQN 使用 policy consistency 定稿的 `neural_fixed_point_q`，因此 checkpoint 轴定义为：

```text
DQN checkpoint = neural fitting epoch
```

AMQ checkpoint 轴定义为：

```text
AMQ checkpoint = online sampled AMQ update index
```

二者不共享同一个 raw step 概念。报告中禁止写 AMQ/DQN step ratio。跨算法主比较只能报告：

- `work_to_stable`：达到 self-stabilization 时累计的 effective Bellman / target evaluations。
- `joint_gap` vs cumulative work 曲线。
- 各自轴上的 native stabilization checkpoint，作为辅助解释。

runtime 只作为附录性质的实现信息，不作为“谁更快”的主证据。

## 3. Routing AMQ 使用论文 AMQ2

Routing 主表使用 AMQ2：

```text
phi_i = [1, x_i + delta_i, (x_i + delta_i)^2, a, b]
```

其中 `delta_i` 按 Appendix A.1：

- attack 且未 defend 时标记最长队列。
- 其他 action pair 标记最短队列。

行为策略使用论文式状态依赖 `alpha(.|x), beta(.|x)`，不是 uniform random。

学习率使用 Robbins-Monro：

```text
eta_k = eta0 / k^p
```

当前 B=20 routing smoke 使用 `eta0=1e-6, p=0.6`，原因是 AMQ2 未归一化二次特征在高 bound 下会放大 TD update。

## 4. Policy extraction

所有 routing policy extraction 均走同一 2x2 matrix-game solver：

```text
Q matrix -> attacker mixed policy + defender mixed policy
```

stabilization gap 必须同时比较 attacker 和 defender：

```text
joint_gap = 0.5 * mean_s |p_attack_t - p_attack_final|
          + 0.5 * mean_s |p_defend_t - p_defend_final|
```

旧 defender-only gap 不再用于新版结论。

## 5. Evaluation grid

正式 routing 实验使用 full bounded grid：

```text
B=20, num_queues=3 -> 21^3 = 9261 states
```

`--eval-state-limit` 只允许用于 smoke，不进入正式报告结论。
