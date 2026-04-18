# Search-R1 当前 OPSD / RLSD 主线方案

这份文档只保留**当前认可的主线**，不再混入已经被否定或暂缓的旧方案。

不纳入本文件的旧思路包括：

- `hint -> 再重算 step logits`
- `alpha` / `1/N` step 均分
- 只基于 `clean_history` 的整体蒸馏
- `reward-based token shaping`
- “整条 history 一次性重算 logits + 全局 learnable mask” 作为当前主线

---

## 1. 当前主线一句话

当前认可的形式是：

```text
teacher: p(y | x, I)
student: p(y | x)
```

但这里的计算单位已经明确为：

```text
step-wise recomputation
```

也就是对每个 step 单独做一次 teacher/student logprob 重算，而不是对整条 history 一次性做更新。

其中：

- `y`：当前 step 的 `clear_step_text`
- `x`：把该 `clear_step_text` 从完整 history 中挖空后的 `masked_history_for_this_step`
- `I`：训练时 hindsight 才知道的 outcome privileged information

最终使用 teacher/student 的 token difference 去**重分布 advantage**，而不是直接重分布 reward。

---

## 2. 为什么主线要改成这样

之前几条路线的问题已经比较明确：

1. `hint` 方案等于先生成额外文本，再拿这个文本去影响另一个目标文本的 logits，链路不干净。
2. `step-level holistic` 评价如果最后落到 `1/N` 或 step 内均摊，会和真正的 token credit assignment 错位。
3. `reward-based shaping` 在 `reward=0` 时会直接把整条样本的 token signal 一起抹掉。
4. `clean_history` 单独作为条件又会丢掉 observation 对 step 合理性的联合影响。

因此当前主线收敛为：

- 仍然按 `step` 做 logits 重算
- 保留完整 history / observation 作为条件上下文
- 只把当前 step 的目标文本从 history 中挖空
- teacher 额外看到 hindsight outcome information
- 最后对主干 advantage 做 token 级 shaping

---

## 3. Step-Wise 输入构造

对于轨迹中的某一个 step，设其目标文本为 `clear_step_text`。

当前 teacher/student 输入应构造成：

1. 从完整 history 中定位该 step 对应的 `clear_step_text`
2. 仅把这一段目标文本挖空
3. 其余 history 全部保留，包括已有 observation
4. student 以这个挖空后的 history 为条件，对 `clear_step_text` 重算 logprob
5. teacher 在相同条件基础上，再额外加入 hindsight PI，对同一段 `clear_step_text` 重算 logprob

对应形式：

```text
teacher: p(clear_step_text | masked_history_for_this_step, privileged_outcome_info)
student: p(clear_step_text | masked_history_for_this_step)
```

这里强调两点：

- 当前仍然是**逐 step**计算，不是整条 history 一次性计算
- 当前需要“挖空”的是**该 step 的目标文本**，不是做一个全局 learnable token mask

---

## 4. Observation 在这里的角色

当前主线里，observation 的角色已经更清楚了：

- observation 是条件上下文的一部分
- observation 不是 privileged information
- observation 也不是当前要被优化的 target span

因为我们现在是按 step 重算 `clear_step_text` 的 logits，所以只要 target span 明确为当前 step 文本，本身就不会把 observation 当成优化目标。

因此在**当前这条主线**里，不需要把“显式 learnable token mask”当作核心前提。那个话题更适合“整条 history 一次性做 token 更新”的方案，不是当前重点。

---

## 5. 特权信息应当是什么

`privileged_outcome_info` 的定义应当是：

```text
hindsight outcome information
```

也就是只有在训练时回看整条轨迹后才知道、而当前 student state 本身拿不到的信息，例如：

- 这条轨迹最终是否答对
- 最终答案是什么
- 后续 observation 最终证实了什么
- 当前 step 在 hindsight 下是有效、无效还是误导

因此：

- `observation` 不是 PI，因为 student 当前就能看到
- `答案 + 是否正确 + hindsight summary` 才是更合理的 PI

---

## 6. 为什么这比旧的 clean history 更合理

这个方案相对旧的 `clean_history` 主线更合理，原因在于：

- 完整 observation 仍然保留在条件里，不会错误地把 step 当成“脱离环境独立成立”的文本
- teacher/student 对齐的是同一段 `clear_step_text`，不会再引入“先生成 hint 再蒸馏另一个文本”的链路偏差
- teacher 真正多看到的是 hindsight PI，而不是 student 当前本来就有的 state
- 计算单位仍然是 step，工程上也更容易和现有 `step extraction -> per-step logprob recomputation` 对齐

---

## 7. Answer Step 的 PI 给法

如果当前被挖空的是最终 `<answer>` 对应的 step，那么当前更合适的做法不是完全跳过它，而是把 PI 收弱成：

```text
correctness only
```

也就是：

- 不给 final answer string
- 只告诉 teacher 这个 `answer step` 最终是对还是错

建议区分两类情况：

### 对 `search` / `think` step

可以给更丰富的 hindsight PI，例如：

- 是否最终成功
- 最终答案
- 后续关键证据摘要

### 对 `answer` step

只给：

- 是否最终正确
- 是否最终错误

对应地，可以把它写成：

```text
teacher: p(answer_step_text | masked_history_for_this_step, correctness_only)
student: p(answer_step_text | masked_history_for_this_step)
```

这样做的含义是：

- `answer step` 仍然参与蒸馏
- teacher 仍然有 hindsight supervision
- 但不会直接因为看到了标准答案原文而变成过强的 oracle copy

---

## 8. Delta 与 Token Weights

对某个 step 的目标 span 中第 `t` 个 token，有：

$$
\Delta_t = \ell_t^{(T)} - \ell_t^{(S)}
$$

其中：

- $\ell_t^{(T)} = \log p_{\text{teacher}}(y_t \mid x, I)$
- $\ell_t^{(S)} = \log p_{\text{student}}(y_t \mid x)$

然后在**该 step 的目标 token 内部**做归一化，得到 token weights：

$$
w_t = \text{softmax}(\Delta)_t
$$

当前这里不再把“全局 masking”写成主线必需项，因为我们并不是在整条 history 的所有 token 上一起做 softmax，而是只对当前 step 的目标 span 做 teacher/student 对齐。

---

## 9. Advantage-Based Shaping

当前主线不再做：

```text
token_reward = reward * weight
```

而是做：

$$
\widetilde{A}_t = A_t \cdot w_t
$$

其中：

- $A_t$：主干 GRPO / PPO 算出的 token advantage
- $w_t$：当前 step 内 teacher/student delta 导出的 token weight

代码抽象就是：

```python
shaped_advantages = advantages * opsd_token_weights
```

这条线的含义是：

- 主干 RL 决定更新方向
- OPSD / RLSD 风格蒸馏决定同一条轨迹内部、同一个 step 内部，哪些 token 应该被更强或更弱地更新

这也是为什么 `reward=0` 样本不必天然失效：

- 它最终是否参与更新，看的是 advantage
- 不是简单看原始 reward 是否为 0

---

## 10. Clipping 的位置

当前更标准的做法是不裁 `delta`，而是裁最终进入 actor update 的 shaped advantage：

```python
shaped_advantages = advantages * opsd_token_weights
shaped_advantages = shaped_advantages * rescale
shaped_advantages = shaped_advantages.clamp(min=-opsd_advantage_clip, max=opsd_advantage_clip)
```

也就是裁：

$$
\widetilde{A}_t
$$

而不是裁：

$$
\Delta_t
$$

---

## 11. 当前实施状态

### 已实施

- `reward-based shaping -> advantage-based shaping`
- 在 batch 中缓存 `opsd_token_weights`
- 在 `compute_advantage()` 后重写 `advantages`
- 对最终 `shaped_advantages` 做 clipping

### 尚未实施

- `masked_history_for_this_step + privileged_outcome_info` 的 teacher/student 输入构造
- 按最新主线重写 `main_ppo.py` 中的 per-step metadata 生成
- 按最新主线重写 `ray_trainer.py` 中的 teacher/student step-wise logits 重算
- `answer` step 的 `correctness_only` PI 特殊处理

---

## 12. 一句话总结

当前 Search-R1 最合理的主线是：

- **按 step 逐个重算 teacher/student logits**
- **把当前 step 的 `clear_step_text` 从完整 history 中挖空**
- **保留完整 history / observation 作为条件上下文**
- **teacher 对普通 step 可看到答案/正确性/hindsight summary，对 `answer step` 只看 correctness**
- **最后用 teacher/student token difference 去 reshape 主干 advantage**

核心形式就是：

$$
\text{teacher: } p(y \mid x, I), \qquad \text{student: } p(y \mid x)
$$

其中当前语境下：

- $y$ 是当前 step 的 `clear_step_text`
- $x$ 是该 step 对应的 `masked_history_for_this_step`
- $I$ 是 outcome privileged information

而不是“整条 history 一次性更新 + 全局 mask”的版本。
