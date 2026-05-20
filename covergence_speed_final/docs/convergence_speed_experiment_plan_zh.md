# Convergence Speed 新实验计划：AMQ vs fitted minimax-DQN

本文档用于重新定义 convergence speed / policy stabilization 实验。旧版收敛速度报告中的“自身最终策略稳定”评价框架可以保留，但旧算法口径不再作为最终结论：旧 NNQ 不是当前敲定的 fitted minimax-DQN，旧 AMQ 也需要进一步贴合 AMQ 论文中的算法形式。

## 1. 核心问题

本模块研究的问题是：

> 在 routing、polling、service-rate-control 三个 queueing benchmark 上，AMQ 是否比 fitted minimax-DQN 更快稳定到各自的最终策略？

这里的“更快”不是指最终 performance 更好，也不是指 AMQ 和 DQN 收敛到同一个策略，而是指：

- AMQ 和 DQN 分别独立训练。
- 对每个算法、每个 seed，保存一系列 checkpoint。
- 对某个 checkpoint 的 policy，与同一个算法、同一个 seed 的最终 checkpoint policy 进行比较。
- 如果从某个 checkpoint 开始，之后所有 checkpoint 都与最终 policy 的差距不超过阈值，就认为该算法在该 seed 上已经达到策略稳定。

## 2. 旧实验状态

旧版报告路径：

`/Users/zheqihu/research/minimax_queueing_results_report/convergence_speed/convergence_speed_experiment_report_zh.md`

旧实验可以保留为历史参考，但不作为新版最终结论，原因如下。

### 2.1 可保留的部分

- “各自对各自最终 policy 稳定”的评价思想是合理的。
- stable step 的定义是合理的：找到第一个 checkpoint，使得它和最终 policy 的 gap 小于阈值，并且之后所有 checkpoint 也都满足这个条件。
- stable step 可以保留为各算法 native axis 下的辅助指标，但跨算法主指标应改为 `work_to_stable`。
- wall-clock runtime 只作为附录性质的实现信息，不进入主结论。

### 2.2 需要重做的部分

- 旧 NNQ 使用的是 `src/adversarial_queueing/algorithms/nnq.py` 中的旧版 NumPy NNQ trainer，不是现在敲定的 fitted minimax-DQN。
- 旧 AMQ 在 routing / polling primary config 中使用常数学习率，这不符合 AMQ 论文中 Robbins-Monro step-size 条件。
- 旧 AMQ 代码中存在 fitted calibration、weight clipping 等工程选项。新版主实验必须排除这些内容，除非单独作为诊断或消融。
- 旧 policy distance 在 routing / polling 中主要比较 defender policy，没有同时比较 attacker policy。新版必须同时评价 attacker 和 defender，因为这是 matrix game。
- 旧 checkpoint 较粗，例如 AMQ 在 routing/polling 上第一个 checkpoint 已经稳定，只能说 `<=20`，无法细分更早稳定时间。

## 3. 新版算法口径

### 3.1 AMQ：严格遵循论文形式

AMQ 主线采用论文中的线性近似 minimax Q-learning：

```text
Q_w(x, a, b) = phi(x, a, b)^T w
```

其中：

- `x` 是 queueing system state。
- `a` 是 attacker action。
- `b` 是 defender action。
- `phi(x,a,b)` 是人工指定的特征。
- `w` 是线性参数。

每次迭代的过程为：

1. 给定当前状态 `X_k`。
2. 根据行为策略采样 attacker action：
   `A_k ~ alpha(. | X_k)`。
3. 根据行为策略采样 defender action：
   `B_k ~ beta(. | X_k)`。
4. 在环境中执行 `(A_k, B_k)`，获得 reward/cost `R_{k+1}` 和下一个状态 `X_{k+1}`。
5. 在 `X_{k+1}` 上求解 2x2 matrix game。论文中 defender 的混合策略 `sigma` 由线性规划得到：

```text
minimize    c
subject to  sum_b sigma(b | x) Q_w(X_{k+1}, a, b) <= c, for all a in {0,1}
            sigma(b | x) >= 0
            sum_b sigma(b | x) = 1
```

该 LP 的最优值是：

```text
c = max_a sum_b sigma(b | x) Q_w(X_{k+1}, a, b)
```

6. 计算 TD error：

```text
Delta_k = R_{k+1} + gamma * c - Q_w(X_k, A_k, B_k)
```

7. 更新线性参数：

```text
w_{k+1} = w_k + eta_k * phi(X_k, A_k, B_k) * Delta_k
```

8. step-size 使用 Robbins-Monro 型衰减：

```text
sum_k eta_k = infinity,    sum_k eta_k^2 < infinity
```

例如：

```text
eta_k = eta0 / (1 + visit_count_or_k)^p,  p in (0.5, 1]
```

新版 AMQ 主实验禁止使用：

- BVI label。
- DQN label。
- fitted calibration。
- least-squares post-fit。
- policy guard。
- 面向结果的 action override。
- 未单独标明的 weight clipping。

### 3.2 AMQ 特征口径

AMQ 论文中使用过线性/仿射特征和二阶多项式特征。新版计划保留两个 AMQ 版本：

- `AMQ-affine`：基础仿射特征，作为最忠实、最简洁版本。
- `AMQ-poly2`：二阶 polynomial traffic-state 特征，作为论文风格增强版本。

最终报告中应明确说明选用哪个作为主线。如果两个都运行，主表只放一个预注册的主版本，另一个放诊断或消融，避免“哪个结果好用哪个”的问题。

### 3.3 DQN：fitted minimax-DQN

新版 NNQ/DQN 使用当前 policy consistency 中敲定的神经网络参数形式，而不是旧 `NNQTrainer`：

```text
Q_theta(x, a, b)
```

其关键特征：

- 最终 policy 由神经网络参数 `theta` 表示。
- 每个状态输出 2x2 matrix game 的 Q values。
- 对每个状态，通过 minimax/matrix-game solver 得到 attacker 和 defender 的混合策略。
- 训练 target 来自 Bellman/minimax backup，不使用 BVI policy label，不使用 AMQ policy label。
- 可以使用 fitted Bellman/minimax-Q 的训练形式来降低采样噪声，但必须清楚说明它是 fitted minimax-DQN，而不是 raw online DQN。

新版 DQN 主实验禁止使用：

- BVI 动作表监督。
- BVI policy imitation。
- one-hot BVI label。
- 从 AMQ 复制 policy。
- 手工 action override。

允许使用的工程稳定化包括：

- 神经网络参数化。
- target network 或 fixed-point iteration 风格的 target freezing。
- batch training。
- state normalization。
- Huber/MSE loss。
- 合理的 optimizer。

这些都必须在报告中列明。

## 4. Policy Stabilization 指标

旧实验只看 defender policy 不够。新版必须同时评价 attacker 和 defender。

对 binary attacker / binary defender 的 benchmark，设：

```text
p_attack_t(s) = checkpoint t 下 attacker 选择 attack 的概率
p_defend_t(s) = checkpoint t 下 defender 选择 defend 的概率
```

相对最终 checkpoint `T` 的 gap 定义为：

```text
attacker_gap(t,T) = mean_s |p_attack_t(s) - p_attack_T(s)|
defender_gap(t,T) = mean_s |p_defend_t(s) - p_defend_T(s)|
joint_gap(t,T)    = 0.5 * attacker_gap(t,T) + 0.5 * defender_gap(t,T)
```

policy similarity 用百分比表示：

```text
policy_similarity(t,T) = (1 - joint_gap(t,T)) * 100%
```

稳定条件：

```text
joint_gap(t,T) <= 0.05
```

并且该 checkpoint 之后所有 checkpoint 也满足该条件。也就是说，稳定后不能再明显反弹。

同时记录 greedy action agreement：

- attacker action 是否一致。
- defender action 是否一致。
- joint action pair `(a,b)` 是否一致。

但主稳定指标优先使用 probability gap，因为 matrix game 中可能存在 mixed policy，单纯 argmax 会在 near-tie 区域造成误判。

## 4.1 Work Accounting 主速度指标

新版 convergence speed 的主速度指标为：

```text
work_to_stable = first stable checkpoint 时累计的 effective Bellman / target evaluations
```

预注册账本见：

```text
/Users/zheqihu/research/minimax_queueing_results_report/covergence_speed_final/docs/work_accounting_schema.json
```

核心原则：

- AMQ online update：每一步记 1 次 sampled TD target / continuation minimax backup。
- Routing fitted minimax-DQN：主 work 只记 model-based fixed point 的 Bellman backups，即 `num_states * 4 * fixed_point_iterations`；后续 MLP fitting 的 gradient steps 和 target entries 单独作为 secondary work。
- Polling full-action DQN：每个 replay batch 记 `distinct sampled states * attacker actions * defender actions` 次 model target。
- Service-rate sampled DQN：每个 replay batch 记 `batch_size` 次 sampled TD target。
- Runtime 不作为主速度指标。

## 5. Benchmark 设计

### 5.1 Routing

优先级最高，作为第一批实现对象。

原因：

- AMQ 源论文直接包含 routing 类 queueing benchmark。
- 现有 policy consistency 中已经有 routing 的 source-faithful BVI / fitted minimax-DQN 代码基础。
- 状态空间和 action 结构清晰，适合作为新版 convergence speed 的首个闭环。

建议设置：

- 三条队列：`(q1, q2, q3)`。
- bound：`B = 20`。
- evaluation states：覆盖低、中、高负载，避免只看 `(0,0,0)` 附近。
- checkpoint 网格要比旧实验更细，尤其前期：

```text
AMQ checkpoints: 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, ...
DQN checkpoints: 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, ...
```

实际最大步数根据 smoke run 的稳定情况调整，但必须事先记录。

### 5.2 Polling

第二优先级。新版应使用三队列 polling，与当前 policy consistency 最终口径一致。

注意事项：

- state 应包含三条队列长度和 server position。
- BVI/AMQ/DQN 的 state clipping 不得错误裁剪 server position。
- policy distance 需要同时比较 attacker 和 defender。
- 图和报告中队列状态应写成 `(q1,q2,q3)`，server position 单独标注，避免旧图中的 `(q1,q2|q3)` 类错误。

### 5.3 Service-rate-control

第三优先级。它不是 AMQ 源论文的原始 benchmark，因此报告中必须写成：

> 将 AMQ 论文形式的线性 minimax-Q update 应用于 service-rate-control extension benchmark。

不能写成“论文原始 benchmark 已覆盖 service-rate-control”。

该 benchmark 的 defender action 可能是 low / medium / high service-rate choice，因此 policy gap 需要扩展为多动作分布距离。attacker 仍然需要单独比较。

## 6. 实验流程

### 6.1 第一阶段：routing smoke

目标不是立刻追求最终结论，而是确认新版 pipeline 没有逻辑错误。

需要完成：

- 实现或整理 source-faithful AMQ runner。
- 实现 fitted minimax-DQN checkpoint runner。
- 保存每个 checkpoint 的参数和 policy。
- 计算 attacker gap、defender gap、joint gap。
- 用 2-3 个 seed 跑 routing smoke。
- 检查 AMQ 参数确实是线性 `w`，DQN 参数确实是神经网络 `theta`。
- 检查没有使用 BVI/AMQ/DQN 互相监督。

### 6.2 第二阶段：routing 正式实验

在 smoke 通过后运行 10 seeds。

输出：

- per-seed work_to_stable。
- per-seed native stable step。
- mean/median work_to_stable。
- policy gap curve。
- runtime auxiliary metadata。
- AMQ vs DQN seed-level earlier/same/later counts。

### 6.3 第三阶段：polling

在 routing pipeline 稳定后迁移到三队列 polling。

重点检查：

- server position 的处理是否正确。
- attacker/defender policy extraction 是否正确。
- 初始状态和 evaluation grid 是否覆盖不同负载。

### 6.4 第四阶段：service-rate-control

最后处理 service-rate-control。

重点检查：

- 多 defender action 的 probability gap 定义是否正确。
- service-rate action 图例和报告文本是否清楚。
- 由于该 benchmark 是 extension，需要在结论中更保守。

## 7. 反作弊与路线约束

新版收敛速度实验必须遵守以下约束：

- 不使用 BVI policy label 训练 DQN。
- 不使用 BVI value table 训练 DQN，除非明确标为 diagnostic upper bound。
- 不使用 AMQ final policy 训练 DQN。
- 不使用 DQN final policy 训练 AMQ。
- 不从多个 seed 中挑好看的 seed 报告。
- 不临时改变稳定阈值来迎合结论。
- 不把 service-rate-control 说成 AMQ 论文原始 benchmark。
- 如果 AMQ 没有更快稳定，必须如实报告，而不是继续调到出现想要结果。

## 8. 预期产物

新版 convergence speed 最终目录建议保持在：

`/Users/zheqihu/research/minimax_queueing_results_report/convergence_speed`

新增或刷新以下内容：

- `convergence_speed_experiment_plan_zh.md`：本计划文件。
- `convergence_speed_experiment_report_zh.md`：新版正式报告。
- `figures/`：三 benchmark 的 stabilization curves。
- `results/`：summary JSON、per-seed JSON、checkpoint policy rows。
- `code/`：只保留复现新版收敛速度实验需要的脚本和配置。

旧结果可以暂时保留，但在新版报告中应标注为 deprecated/preliminary，不进入最终结论。

## 9. 当前执行顺序

建议下一步按以下顺序推进：

1. 写 routing 新版 convergence speed runner。
2. 先跑 routing smoke，确认 AMQ 和 fitted minimax-DQN 都能输出 checkpoint policy。
3. 修正 policy distance 为 attacker + defender joint gap。
4. 跑 routing 10-seed 正式实验。
5. 将同一框架迁移到 polling 三队列。
6. 最后迁移到 service-rate-control。
7. 生成新版 convergence speed 中文报告。

这个顺序的好处是：先在最接近 AMQ 论文和当前 policy consistency 代码基础的 routing 上闭环，再扩展到另外两个 benchmark，避免三个 benchmark 同时改动导致错误来源混在一起。
