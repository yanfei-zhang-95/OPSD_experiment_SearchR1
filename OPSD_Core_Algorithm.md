# OPSD (On-Policy Self-Distillation) 算法核心要点

OPSD (On-Policy Self-Distillation) 是一种结合了强化学习（如 PPO/GRPO）与自我蒸馏机制的训练算法。在当前基于 `verl` 的 BrowseComp 实验中，OPSD 主要用于在多回合长轨迹（Trajectory）中提供细粒度的 Token 级别奖励和优势引导。

以下是 OPSD 算法的核心要点与处理流：

## 0. 基础奖励计算与 ORM (Outcome Reward Model)
- **概念**：在进行 OPSD Token 级别拆解前，首先需要从环境中获得轨迹级别的全局基础奖励 $R$（Macro Reward）。
- **ORM 逻辑**：框架采用结果奖励模型（Outcome Reward Model，ORM）来评估整条轨迹的最终输出与真实标签（Ground Truth）的一致性。例如，在 BrowseComp 等任务中，通过提取模型给出的最终答案，并利用基准测试中的精确匹配（Exact Match）或子串匹配（Substring Match）规则进行判定。判定后得到的基础分数 $R$（通常为离散值如 $0$ 或 $1$）会作为后续所有步骤优势分配的总基数。

## 1. 轨迹步骤拆解 (Step Extraction)
- **概念**：将 LLM 采样的完整长轨迹（包含多个回合的 `Thought`、`Action`、`Observation` 交互）拆解为离散的逻辑步骤单元。
- **作用**：为后续的细粒度评估和提示生成提供基础评估单元，避免对整条长轨迹进行粗放式的打分。

## 2. LLM 提示生成 (Hint Generation)
- **概念**：针对失败的轨迹或存在优化空间的特定步骤，调用强大的 LLM（或具有特权信息的 Teacher 模型）生成纠错提示（Hint）。在本框架中，默认采用 **Self-Distillation（自我蒸馏）** 模式，即直接使用当前正在训练的 Actor 权重进行本地推理生成 Hint。
- **作用**：指出当前策略在某一步骤中的具体错误，并给出正确的思考方向或行动建议，为模型自我纠正提供监督信号。
- **使用的 Prompt**：
  ```text
  You are generating a compact teacher hint for RL self-distillation.
  Return a JSON object with keys hint and hint_reason.

  Question: {question}
  History: {history_json}
  Current state: {current_state}
  Current action: {current_action}
  Environment observation: {environment_observation}
  ```

## 3. 教师评估与影响力打分 (Teacher Evaluation - $\alpha_i$ 计算)
- **概念**：教师模型对拆解出的每一个步骤进行评估，计算每个步骤的重要性或影响力得分（通常记为 $\alpha_i$）。与 Hint 生成类似，本框架默认采用 Actor 权重进行本地自我评估。
- **作用**：区分轨迹中关键的决策步和无关紧要的边缘步。得分 $\alpha_i$ 决定了该步骤在后续计算 Token 级别奖励时的权重，重点优化导致任务成功或失败的“关键帧”。
- **算法公式**：
  - 如果 **启用** $\alpha$ 评估：每个步骤的优势分配按照 $\alpha_i$ 占比分配基础奖励 $R$：
    $$A_{step, i} = R \times \frac{\alpha_i}{\sum_{k=1}^{N} \alpha_k}$$
  - 如果 **关闭** $\alpha$ 评估：基础奖励在所有步骤之间平均分配：
    $$A_{step, i} = \frac{R}{N}$$
    （其中 $N$ 为当前轨迹的总步骤数）
- **使用的 Prompt**：
  ```text
  Evaluate the usefulness of the current reasoning step for solving the question.
  Return a JSON object with a single key alpha in range [0, 2].

  Question: {question}
  History: {history_json}
  Current step action: {current_action}
  Current observation: {environment_observation}
  ```

## 4. KL 散度重加权 (KL Reweighting)
- **概念**：在计算优势函数时，结合 KL 散度（即 Actor 模型与 Reference 模型的 Token 级别分布差异）进行重加权。在自我蒸馏中，这一步通过计算带有 Hint 的文本分布和原始状态分布之间的差异来实现。
- **作用**：确保模型在学习 Teacher 的 Hint 和高分步骤时，不会偏离参考策略（Reference Policy）过远，从而保证 On-Policy 训练的稳定性和收敛性，防止模型崩溃。
- **算法公式与 Token Reweighting 逻辑**：
  1. 计算在特定上下文下生成第 $j$ 个 Token 的对数概率差异（即 KL 散度）：
     $$KL_j = D_{KL} \left( \pi_{\text{student}}(\cdot | s_t) \parallel \pi_{\text{teacher}}(\cdot | s_t, \text{hint}) \right)[a_j]$$
  2. 对整个步骤内生成的 Token 长度（记为 $L_i$），将 KL 散度进行归一化，得到每个 Token 的相对权重：
     $$w_j = \frac{KL_j}{\sum_{k=1}^{L_i} KL_k}$$
  3. 将之前计算得到的步骤级优势 $A_{step, i}$ 乘上 Token 权重，得到细粒度的 Token 优势：
     $$A_{token, j} = A_{step, i} \times w_j$$
- **具体的文本组装**：
  - **Teacher Text (带 Hint 的输入)**：`{current_state}\n\nHint: {generated_hint}`
  - **Student Text (原始输入)**：`{current_state}`
  - **Answer Text (待测动作)**：`{current_action}`
  （KL 引擎会对比模型在给定 Teacher Text 与 Student Text 下输出 Answer Text 时的 Token 级概率分布差异）

## 5. 优势函数重塑 (Advantage Shaping)
- **概念**：将环境反馈的基础稀疏奖励（如 GRPO 的任务最终成功与否）与 OPSD 计算出的 Token 级别奖励/步骤优势进行合并。
- **作用**：将传统的稀疏奖励（Sequence-level Reward）转化为密集的 Token-level 奖励，使模型不仅知道“最终结果对不对”，还能明确知道“这中间每一步的思考过程到底好不好”。
- **算法公式**：
  最终分配到 Token 级别的奖励信号，由原始的 Macro Reward 和 OPSD Token Reward 按系数 $\lambda_{\text{ospd}}$ 进行组合（代码中实现为直接叠加至 `A_final` 以便于观察，在训练中则作为旁路密集奖励与主干 Actor Loss 合并）：
  $$A_{\text{final}} = A_{\text{macro}} + \lambda_{\text{ospd}} \times \sum_{j} A_{token, j}$$
  *(注：具体实现时 `A_token` 级别优势通过 `reward_extra_info` 透传回 PPO Trainer)*

## 6. 在 verl 框架中的系统集成与实现架构 (最新)
在本项目中，为了保持 `verl` 核心框架的纯洁性并最大化 GPU 利用率，OPSD 被设计为直接在 **Reward 计算阶段的内存拦截（In-memory Interception）** 模块：

### 6.1 零侵入式模型注入 (Zero-Intrusion Injection)
- **原理**：不修改 `verl` 内部的 `ray_trainer.py`，而是在用户层启动脚本（如 `main_ppo.py`）中，利用动态绑定将 `actor_rollout_wg`（即当前正在训练的 Actor 模型引擎）注入到 `RewardManager` 中。
- **优势**：使得 `RewardManager` 获得了直接调用当前 Actor 模型进行推理（生成 Hint）和计算 Logits（计算 KL）的能力，同时完全解耦了算法层和框架层。

### 6.2 内存就地批处理流 (In-memory Batched Processing)
为了避免逐条处理导致的时间开销，以及避免将 Rollout 写入磁盘带来的 I/O 瓶颈，整个 OPSD 计算在 `RewardManager.__call__` 中以 Batched 形式在内存中瞬间完成：
1. **收集与展平 (Flatten)**：在内存中解析当前 Batch 的完整轨迹字符串（`sequences_str`），利用正则表达式实时提取出所有的逻辑步骤（如 `<think>`, `<action>`），并记录它们在原 `response_ids` 序列中的 Token 索引起止位置。
2. **组装 Hint/$\alpha$ 提示词**：为提取出的所有 Step 构造评估所需的 `opsd_prompt`。
3. **一次性生成 (Batched Generation)**：将所有 Prompt 组装成一个扁平化的大 Batch，调用 `actor_rollout_wg.generate_sequences`，利用底层 vLLM/Megatron 引擎的并发能力实现极致吞吐，瞬间完成所有 Hint 和 $\alpha$ 打分的生成。
4. **一次性计算 KL (Batched Forward)**：将 `[Teacher_Text + Answer]` 组成 Batch，调用 `actor_rollout_wg.compute_log_prob()` 拿到带 Hint 的教师 Logits，并完成 KL 散度加权。
5. **映射回填 (Scatter)**：计算出所有 Token 的细粒度密集奖励（Dense Rewards）后，根据第 1 步记录的 Token 索引，将数据就地填回形状为 `[batch_size, response_length]` 的 `reward_tensor` 中，直接返回给 PPO/GRPO Trainer 进行优势计算。

## 7. Search-R1 中的具体实现细节（Search-R1 OPSD Implementation）

### 7.1 整体数据流概览

Search-R1 的 OPSD 实现分布在两个核心文件中：

| 文件 | 职责 |
|------|------|
| `verl/trainer/main_ppo.py`（`RewardManager` 类） | 轨迹拆解、Hint 生成、1/N 奖励分配、opsd_kl_data 构造 |
| `verl/trainer/ppo/ray_trainer.py`（`_apply_opsd_token_level_kl` 方法） | Teacher Logits 计算、Token 级 KL 加权 |

完整数据流如下：

```
Rollout Batch (sequences_str)
    ↓
RewardManager.__call__()
    ├── 1. 正则拆解 <search>/<answer>/<information> 步骤
    ├── 2. 构造 Hint 生成 Prompt（3-item 格式: Progress / Findings / Better query）
    ├── 3. vLLM 批量生成 Hint
    ├── 4. 1/N 等分 macro_reward → 每个 step 获得 dense_reward
    ├── 5. 将 step_info (token_start/end, hint, current_state 等) 存入 opsd_kl_data
    └── 6. 返回 reward_tensor（已在 step['token_end_idx'] 写入 dense_reward）
    ↓
ray_trainer.fit() 中:
    reward_tensor = self.reward_fn(batch)
    batch.batch['token_level_scores'] = reward_tensor
    batch = self._apply_opsd_token_level_kl(batch)   ← 在这里完成 Token 级 KL
    ↓
apply_kl_penalty() / compute_advantage() / update_actor()
```

### 7.2 步骤拆解（Step Extraction）

采用正则表达式从 `sequences_str` 中提取步骤，匹配模式为：

```python
pattern = rf'<({action_types})>([^<]+)</({action_types})>|(<think>[^<]*</think>)'
```

每提取到一个步骤，记录以下关键信息：

| 字段 | 含义 |
|------|------|
| `token_start_idx` | 该步骤在 `response_ids` 中的起始 token 位置 |
| `token_end_idx` | 该步骤在 `response_ids` 中的结束 token 位置 |
| `action_token_start_idx` | `<action>` 标签**后**内容开始的 token 位置 |
| `action_token_end_idx` | `</action>` 标签**前**内容结束的 token 位置 |
| `action_type` | 动作类型：`search`、`answer` 或 `information` |
| `history_str` | 到当前步骤为止的完整历史轨迹文本 |
| `current_state_str` | 恰好在当前步骤**之前**的轨迹文本（即 Teacher 的条件上下文） |
| `original_prefix_str` | 同 `current_state_str`，用于后续 KL 计算时还原学生分布 |
| `hint_prompt` | 用于 Debug 打印的完整 Hint 生成 Prompt |

### 7.3 Hint 生成 Prompt（3-Item 结构）

不同于传统的 JSON 结构化输出，Search-R1 采用纯文本的 3-item Hint 格式：

```
You are a careful teacher for RL self-distillation...
Output exactly 3 numbered items in plain text:
1. Progress: Is this step better, worse, or neutral for solving the question so far? Give a short reason.
2. Findings: What did this step successfully find, and what important information is still missing?
3. Better query or next action: If another search is needed, give one improved search query...

Requirements:
- Focus on step quality, not style.
- If the current step is based on a wrong assumption, say so explicitly.
- If the current answer conflicts with the observation, say so explicitly.
- If the current search query is weak, ambiguous, or too broad, propose a sharper query.
- Keep each item short and specific.
```

其中 `current_state_str` 和 `history_str` 分别注入到 Prompt 的对应位置：

```python
messages = [
    {"role": "system", "content": "You are a careful teacher for RL self-distillation..."},
    {"role": "user", "content": (
        f"Question: {question_text}\n"
        f"History so far: {history_text}\n"
        f"Current step action: <{action_type}>{action_content}</{action_type}>\n"
        f"Current observation: {obs_content[:1000]}\n\n"
        "Evaluate ONLY the current step based on the history so far.\n"
        "Output exactly 3 numbered items..."
    )}
]
```

Tokenize 时使用 `apply_chat_template(..., tokenize=True, add_generation_prompt=True)`，确保只经过一次 chat template 处理，避免双重 special token 化。

### 7.4 奖励分配（1/N 均分 + opsd_kl_data）

```python
opsd_kl_data = np.empty((batch_size,), dtype=object)
for i in range(batch_size):
    macro_score = all_scores[i]          # ORM 最终分数（0 或 1）
    step_info = batch_steps.get(i, [])    # 该轨迹所有步骤的元信息
    
    opsd_kl_data[i] = step_info           # 保存，供后续 _apply_opsd_token_level_kl 使用
    
    if not step_info:
        reward_tensor[i, valid_response_lengths[i] - 1] = macro_score  # 兜底稀疏奖励
        continue
        
    num_steps = len(step_info)
    dense_reward = macro_score / num_steps  # 1/N 等分
    
    for step in step_info:
        reward_tensor[i, step['token_end_idx']] = dense_reward
```

注入 `opsd_kl_data` 到 DataProto 的 `non_tensor_batch` 中：

```python
data.non_tensor_batch['opsd_kl_data'] = opsd_kl_data
```

### 7.5 Teacher Logits 计算（_apply_opsd_token_level_kl）

该方法在 `ray_trainer.fit()` 的主循环中被调用，**位于 `reward_fn` 返回 reward_tensor 之后**，核心步骤如下：

#### 7.5.1 Teacher Prompt 构造

对每个 step，Teacher 的输入文本为：

```
teacher_prompt = current_state + "\n\nHint: " + hint
```

其中 `current_state` 是当前步骤之前的历史文本（不含当前步骤本身），`hint` 是 Teacher 模型生成的改进建议。

#### 7.5.2 Left-Padded Logprob Batch 构造

由于 `compute_log_prob()` 需要对 prompt 和 response 分别编码，采用左填充（Left-Padding）避免右填充带来的 pad token 泄露问题：

```python
def _build_left_padded_logprob_batch(self, prompt_ids_list, response_ids_list, micro_batch_size=128):
    pad_token_id = self.tokenizer.pad_token_id
    max_prompt_len = max(x.numel() for x in prompt_ids_list)
    max_resp_len = max(x.numel() for x in response_ids_list)
    seq_len = max_prompt_len + max_resp_len
    
    input_ids = torch.full((batch_size, seq_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)
    responses = torch.full((batch_size, max_resp_len), pad_token_id, dtype=torch.long)
    
    for i, (prompt_ids, response_ids) in enumerate(zip(prompt_ids_list, response_ids_list)):
        # 左填充：内容靠右对齐
        input_ids[i, max_prompt_len - p_len:max_prompt_len] = prompt_ids
        input_ids[i, max_prompt_len:max_prompt_len + r_len] = response_ids
        attention_mask[i, max_prompt_len - p_len:max_prompt_len + r_len] = 1
        responses[i, :r_len] = response_ids
    
    position_ids = attention_mask.cumsum(dim=1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)
```

#### 7.5.3 DataProto Padding（World_Size Divisibility）

`compute_log_prob()` 要求 DataProto 大小能被 `world_size=8` 整除，因此使用已有的 padding 工具：

```python
teacher_batch_padded, pad_size = pad_dataproto_to_divisor(
    teacher_batch, self.actor_rollout_wg.world_size
)
teacher_logprob_output_padded = self.actor_rollout_wg.compute_log_prob(teacher_batch_padded)
teacher_logprob_output = unpad_dataproto(teacher_logprob_output_padded, pad_size=pad_size)
teacher_log_probs = teacher_logprob_output.batch['old_log_probs']
```

#### 7.5.4 Token 级 KL 重加权（Softmax Smoothing）

对每个 step，对比 Teacher 和 Student 在该 step 对应 action token span 上的 log probabilities：

```python
delta = teacher_lp - student_lp          # shape: [shared_len]
weights = torch.softmax(delta, dim=-1)   # 平滑权重，避免 one-hot collapse

step_reward = macro_reward / num_steps   # 1/N dense reward
token_level_scores[batch_idx, action_start:action_start + shared_len] = step_reward * weights
```

**关键澄清**：`teacher_lp` 和 `student_lp` 本身是**原始 log probabilities**（未经 ReLU/截断），它们来自：
- `student_lp`：Rollout 时原始策略在 `current_state` 条件下采样 `action_str` 的 log prob（来自 `batch.old_log_probs`）
- `teacher_lp`：`compute_log_prob()` 在 `current_state + "\n\nHint: " + hint` 条件下计算 `action_str` 的 log prob

**Softmax vs ReLU**：之前尝试用 `relu(delta)` 发现权重会退化为 one-hot（最大 delta 独占全部权重），改为 `softmax(delta, dim=-1)` 后权重分布更加平滑，避免对单个 token 的过度聚焦。

### 7.6 调试打印（Debug Prints）

实现了两层 Debug 打印，分别在 Hint 生成阶段和 Token 级 KL 计算阶段：

**Hint 生成阶段（main_ppo.py）**：
```python
if opsd_metadata[idx]['batch_idx'] == 0:
    print(
        "[OPSD DEBUG] Traj 0 Prompt/Response Pair\n"
        f"[Prompt]\n{opsd_metadata[idx]['hint_prompt']}\n"
        f"[Response]\n{hints[-1]}\n"
        + "-" * 80
    )
```

**Token 级 KL 阶段（ray_trainer.py）**：
```python
if batch_idx == 0 and meta['step_idx'] < 3:
    print(
        "[OPSD KL DEBUG] Token-Level KL\n"
        f"[Step Idx] {meta['step_idx']}\n"
        f"[Action Text]\n{action_text}\n"
        f"[Student LogProb] {student_lp.tolist()}\n"
        f"[Teacher LogProb] {teacher_lp.tolist()}\n"
        f"[Delta] {delta.tolist()}\n"
        f"[Weights] {weights.tolist()}\n"
        f"[Macro Reward] {float(macro_reward):.6f}\n"
        f"[Step Reward] {float(step_reward):.6f}\n"
        + "-" * 80
    )
```

### 7.7 在 verl fit() 循环中的调用位置

```python
# verl/trainer/ppo/ray_trainer.py fit() 方法，约第 972 行
reward_tensor = self.reward_fn(batch)
batch.batch['token_level_scores'] = reward_tensor
batch = self._apply_opsd_token_level_kl(batch)   # ← OPSD Token KL 在此处注入

# compute rewards. apply_kl_penalty if available
if not self.config.actor_rollout_ref.actor.use_kl_loss:
    batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl, kl_penalty=self.config.algorithm.kl_penalty)
    metrics.update(kl_metrics)
else:
    batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
```
