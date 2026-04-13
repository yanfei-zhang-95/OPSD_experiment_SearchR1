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
