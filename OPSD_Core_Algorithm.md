# OPSD / RLSD 风格蒸馏在 Search-R1 中的当前主线方案

本文档只保留当前已经确认的主线设计，不再保留已经被否定的旧方案，例如：

- `hint -> 再重算 step logits` 两层链路
- `alpha` / 1/N step 均分逻辑
- 只基于 `clean_history` 的无 observation 对照
- `reward-based token shaping`

当前认可的方向是：

1. **Teacher / Student 都在完整状态（full state）条件下比较**
2. **被优化的目标 span 只包含模型自己生成的 token**
3. **Teacher 的特权信息来自 hindsight outcome information**
4. **最终做的是 advantage-based shaping，而不是 reward-based shaping**

---

## 0. 核心结论

Search-R1 当前最合理的蒸馏目标不是：

```text
teacher: p(clean_history | privileged_context)
student: p(clean_history | normal_context)
```

而是：

```text
teacher: p(clear_step_text | masked_full_history, privileged_outcome_info)
student: p(clear_step_text | original_rollout_context)
```

其中：

- `clear_step_text`：当前要评估/蒸馏的模型生成片段
- `masked_full_history`：在完整 history 中把这段 `clear_step_text` 挖空后的文本
- `privileged_outcome_info`：训练时 hindsight 才知道的信息，如最终是否正确、最终答案、后续证据摘要

这个设计的好处是：

- **observation 仍然保留在条件上下文里**
- **当前目标 span 不会直接泄露给 teacher**
- **teacher 真正满足 `p(y|x,I)`，student 满足 `p(y|x)`**

---

## 1. 环境奖励与主干 RL 信号

Search-R1 的主干训练仍然是 GRPO / PPO 风格：

- 先由 ORM / EM 打分得到 trajectory-level reward
- 再由 `compute_advantage()` 计算轨迹内 token advantage

因此，主干 RL 的职责是：

- 决定更新方向
- 决定成功/失败的相对比较
- 为失败样本提供负 advantage

蒸馏分支不负责替代这件事，而是负责：

- **在同一条轨迹内部，对 token update magnitude 进行更细粒度的重分配**

---

## 2. Full State 与 Learnable Span 的分离

当前已经确认的关键原则是：

### 2.1 Full State 仍然保留 observation

在 Search-R1 中，一个 token 的合理性往往依赖 observation。

因此，teacher / student 在重算 logprob 时，条件上下文不能只看去掉 `<information>` 后的文本，而应保留：

- `Question`
- 历史模型输出
- 历史 observation
- 当前 observation

也就是说：

```text
observation 参与 delta 计算
```

### 2.2 但 observation token 本身不参与优化

环境 observation 不是 actor 生成的，因此不应该被赋予 reward / advantage 或梯度。

因此还需要一个单独的 **learnable token mask**：

- `<think>...</think>`：可学习
- `<search>...</search>`：可学习
- `<answer>...</answer>`：可学习
- `<information>...</information>`：不可学习

也就是说：

```text
observation 参与判分，但不参与被更新
```

这是当前 agent 场景下最自然的 masking 方式。

---

## 3. 特权信息（Privileged Information）应当是什么

在这个设定下，`observation` 本身**不是**特权信息，因为 student 在当前 state 里也能看到。

真正的特权信息应当是：

> **当前时刻看不到，但训练时回看整条轨迹后才能知道的 hindsight 信息**

最自然的来源包括：

- 这条轨迹最终是否正确
- 最终答案是什么
- 后续 observation 最终证实了什么
- 当前前缀后来是被验证为有效、无效还是误导

因此，当前最合理的 PI 定义是：

```text
privileged_outcome_info = hindsight outcome information
```

它不是当前 state 的一部分，但训练时合法可得。

---

## 4. Masked Full History 方案

### 4.1 基本思想

对于当前要蒸馏的一段 `clear_step_text`：

1. 在完整 history 中把这一段挖空
2. 保留其余完整上下文
3. 在 teacher 侧追加 hindsight outcome information
4. 用 teacher / student 分别对这段被挖空的 `clear_step_text` 算 logprob

记号上：

- `y = clear_step_text`
- `x = masked_full_history`
- `I = privileged_outcome_info`

则：

```text
teacher: p(y | x, I)
student: p(y | x)
```

这就是当前最接近标准 teacher/student distillation 的形式。

### 4.2 为什么比 clean history 更好

它同时解决了三个问题：

1. **保留真实 state 条件**
   - observation 仍然参与 token 合理性的判断

2. **避免 target leakage**
   - 当前要预测的 span 被挖空，不会直接塞进 teacher prompt

3. **PI 真正额外**
   - teacher 多看到的是 hindsight outcome info，而不是 student 本来就能看到的 state

---

## 5. 对 `answer` step 的特殊处理

若当前被挖空的是最终 `<answer>` span，那么把“最终答案原文”直接放进特权信息里会导致强 target leakage。

因此建议分情况：

### 对 `search` / `think` step

可以给 teacher：

- 是否最终成功
- 最终答案
- 后续关键证据摘要

### 对 `answer` step

不要直接给最终答案原文，建议只给：

- 是否最终正确
- 哪些证据支持正确答案
- 当前答案是否应与某些 observation 一致

这样可以减少 oracle 泄露。

---

## 6. Delta 计算与 Masking

### 6.1 Delta 的定义

对于目标 span 中第 $t$ 个 token：

$$
\Delta_t = \ell_t^{(T)} - \ell_t^{(S)}
$$

其中：

- $\ell_t^{(T)} = \log p_{	ext{teacher}}(y_t \mid x, I)$
- $\ell_t^{(S)} = \log p_{	ext{student}}(y_t \mid x)$

### 6.2 Masking 原则

虽然 teacher / student 的条件上下文保留完整 state，但做 token reweighting 时，应只在可学习 token 上归一化：

```python
masked_delta = delta.masked_fill(learnable_token_mask == 0, -1e9)
weights = torch.softmax(masked_delta, dim=-1)
weights = weights * learnable_token_mask
weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
```

这意味着：

- observation token 不参与权重竞争
- softmax 只在模型自己生成的 span 上归一化

---

## 7. Advantage-Based Shaping

当前已经确认的关键修正是：

> **OPSD 在这个项目里应当作用在 advantage 上，而不是直接作用在 reward 上。**

原因是：

- 如果直接做 `reward-based shaping`
- 那么 `reward=0` 的样本即使算了 `weights`，最终也仍然是 0

因此，当前主线应当是：

$$
\widetilde{A}_t = A_t \cdot w_t
$$

其中：

- $A_t$：主干 RL 算出来的 token advantage
- $w_t$：teacher/student delta 导出的 token weight

代码层面的抽象是：

```python
shaped_advantages = advantages * opsd_token_weights
```

所以现在：

- 主干 RL 决定更新方向
- OPSD 决定轨迹内部 token 更新的相对幅度

这与 RLSD 的思想是一致的：

```text
direction <- RL / advantage
magnitude <- self-distillation token difference
```

---

## 8. Clipping 的标准位置

当前已经确认：

- 不再对 `delta` 做 clipping
- clipping 应放在最终进入 actor update 的逐 token optimization signal 上

因此更标准的位置是：

```python
shaped_advantages = advantages * opsd_token_weights
shaped_advantages = shaped_advantages * rescale
shaped_advantages = shaped_advantages.clamp(min=-opsd_advantage_clip, max=opsd_advantage_clip)
```

这里被裁的是：

$$
\widetilde{A}_t
$$

而不是中间分数：

$$
\Delta_t
$$

这比 `delta.clamp()` 更接近标准的 per-token clipping 思路，因为它直接控制最终逐 token 更新强度。

---

## 9. 当前 Search-R1 的实施状态

### 已经开始实施

- `reward-based shaping -> advantage-based shaping`
- `opsd_token_weights` 缓存到 batch
- `compute_advantage()` 后对 `advantages` 做 token 级重分配
- 对最终 `shaped_advantages` 做 clipping

### 尚未完全落地

- `teacher: p(y|x,I), student: p(y|x)` 的 **masked full history + outcome PI** 输入构造
- 显式的 `learnable_token_mask`
- 在 delta softmax 归一化中排除 observation token
- `answer` step 的 anti-leakage 特殊处理

也就是说：

- **advantage-based shaping 主线已经开始实施**
- **teacher/student 的特权输入设计还需要进一步改到 masked full history 版本**

---

## 10. 最终主线总结

当前最合理、且后续应继续推进的版本可以概括为：

1. **主干 RL**  
   用 ORM / EM + GRPO 计算 reward 与 advantage

2. **Teacher / Student 条件分布**  
   用完整 state 做条件，不删除 observation

3. **Target Span**  
   只预测并更新当前被挖空的 `clear_step_text`

4. **Privileged Information**  
   使用 hindsight outcome information，而不是当前 state 自带信息

5. **Masking**  
   observation 参与条件判断，但不参与 token update

6. **Shaping**  
   对 `advantages` 做 token 级重分配，而不是对 `reward` 做重分配

7. **Clipping**  
   对最终 `shaped_advantages` 做 clipping，而不是对 `delta` 做 clipping

这条主线的数学形式是：

$$
	ext{teacher: } p(y \mid x, I), \qquad
	ext{student: } p(y \mid x)
$$

$$
\Delta_t = \log p_{	ext{teacher}}(y_t) - \log p_{	ext{student}}(y_t)
$$

$$
w_t = 	ext{softmax}(\Delta)_t \quad 	ext{(仅在可学习 token 上归一化)}
$$

$$
\widetilde{A}_t = A_t \cdot w_t
$$

最终由：

- 主干 advantage 决定方向
- teacher-conditioned token weights 决定轨迹内部更新分布

这就是当前 Search-R1 中最合理的 OPSD / RLSD 风格实现方向。
