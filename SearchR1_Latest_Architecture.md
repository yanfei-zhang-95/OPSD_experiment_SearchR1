# Search-R1 Latest Architecture

这份文档描述当前仓库里实际在跑的最新训练架构，重点对应如下代码与脚本路径：

- `train_grpo_full8.sh`
- `verl/trainer/ppo/ray_trainer.py`
- `search_r1/llm_agent/generation.py`
- `verl/workers/fsdp_workers.py`
- `verl/workers/actor/dp_actor.py`

本文不再展开旧版本设计，只记录当前代码主路径，尤其是 `do_search=true`、`GRPO + OPSD`、`FSDP actor/ref + vLLM rollout` 这条链路。

---

## 1. 当前训练形态

当前主实验由 `train_grpo_full8.sh` 启动，核心配置是：

- 基座模型：`Qwen/Qwen2.5-7B-Instruct`
- 训练方式：`GRPO`
- 搜索模式：`do_search=true`
- 多轮 agent：`max_turns=20`
- 每条样本扩成 `n_agent=5`
- Actor/Ref 策略：`FSDP`
- Rollout 引擎：`vllm`
- KL 方式：`actor.use_kl_loss=true`
- OPSD：开启，当前 student mode 为 `causal_prefix`
- Teacher mode：`live_actor`

当前脚本里的关键 batch 设置为：

- `data.train_batch_size=64`
- `actor_rollout_ref.actor.ppo_mini_batch_size=256`
- `actor_rollout_ref.actor.ppo_micro_batch_size=32`
- `actor_rollout_ref.rollout.log_prob_micro_batch_size=128`
- `actor_rollout_ref.rollout.temperature=1`

---

## 2. 总体模块关系

当前训练链路可以概括为：

```text
train_grpo_full8.sh
  -> verl.trainer.main_ppo
    -> RayPPOTrainer.fit()
      -> DataProto batch 构建
      -> Search agent 多轮 rollout
      -> final_gen_batch_output
      -> old_log_probs / ref_log_prob / reward / OPSD KL / advantage
      -> update_critic
      -> update_actor
```

更细一点的角色分工如下：

- `train_grpo_full8.sh`
  - 负责注入 Hydra 配置。
  - 定义当前实验的 batch、resume、rollout、OPSD 等运行参数。

- `verl/trainer/ppo/ray_trainer.py`
  - 是训练主控。
  - 负责 dataloader、resume、训练循环、验证、reward、advantage、actor/critic 更新。

- `search_r1/llm_agent/generation.py`
  - 是 `do_search=true` 时的多轮 agent 生成管理器。
  - 负责 active trajectory 的筛选、搜索调用、observation 注入、最终轨迹拼装。

- `verl/workers/fsdp_workers.py`
  - 是 trainer 和底层 actor/ref/rollout 之间的 FSDP worker 边界。
  - 负责 `generate_sequences`、`compute_log_prob`、`compute_ref_log_prob`、`update_actor` 等分布式接口。

- `verl/workers/actor/dp_actor.py`
  - 是 actor 前向、logprob 重算、PPO policy 更新的核心执行逻辑。

---

## 3. 训练主链路

### 3.1 Batch 进入 trainer

在 `ray_trainer.py` 中，每个训练 step 的主路径是：

1. 从 `train_dataloader` 取出 `batch_dict`
2. 转成 `DataProto`
3. 按 `n_agent=5` 做 `repeat`
4. 弹出 `input_ids / attention_mask / position_ids` 作为 `gen_batch`

这里的设计含义是：

- 原始监督样本是一个基础问题样本
- 进入 RL 之后，会复制成多个 agent 轨迹
- 每条轨迹独立进行多轮 search/reasoning 交互

### 3.2 `do_search=false` 与 `do_search=true`

当前真正使用的是 `do_search=true` 分支。

- `do_search=false`
  - 直接调用 `actor_rollout_wg.generate_sequences(gen_batch)`
  - 属于单次 rollout 模式

- `do_search=true`
  - 调 `LLMGenerationManager.run_llm_loop(...)`
  - 进入多轮 agent 循环
  - 每轮根据模型输出判断是 `search`、`answer` 或无效动作
  - 根据搜索结果构造 observation 再回灌到 rolling context
  - 最终拼成 `final_gen_batch_output`

---

## 4. Search Agent 多轮生成架构

`search_r1/llm_agent/generation.py` 是当前 search agent 的核心。

### 4.1 Active trajectory 机制

当前 agent loop 不是每轮都对全量轨迹继续生成，而是维护一个 `active_mask`：

- 活跃轨迹继续 rollout
- 已结束轨迹停止继续生成
- 每轮记录：
  - `turns_stats`
  - `valid_action_stats`
  - `valid_search_stats`
  - `active_num_list`

这就是日志里 `ACTIVE_TRAJ_NUM: [...]` 的来源。

### 4.2 多轮循环

每一轮大致做下面几件事：

1. 用 `cut_to_effective_len()` 截断到当前有效长度
2. 仅保留 active 样本组成 `rollings_active`
3. 调 `_generate_with_gpu_padding()`
4. 后处理 response
5. 执行动作
6. 如动作是 `search`，走检索器，拿到搜索结果
7. 构造下一轮 observation
8. 更新 rolling state

### 4.3 GPU 对齐 padding

`_generate_with_gpu_padding()` 的职责是：

- 如果 active 样本数不能被 `world_size` 整除
- 先用 `pad_dataproto_to_divisor()` 做补齐
- 调 `actor_rollout_wg.generate_sequences(...)`
- 再做 `unpad_dataproto()`

这是 search agent 在多 GPU 下支持小活跃批次的关键。

### 4.4 最终输出拼装

多轮结束后，`_compose_final_output()` 会构造：

- `prompts`
- `responses`
- `responses_with_info_mask`
- `input_ids`
- `attention_mask`
- `info_mask`
- `position_ids`

并把统计信息写回 `meta_info`，例如：

- `turns_stats`
- `active_mask`
- `valid_action_stats`
- `valid_search_stats`

---

## 5. Trainer 后半段计算图

在 `ray_trainer.py` 的 search 分支里，`final_gen_batch_output` 回到 trainer 之后会继续走下面这条链：

1. `compute_log_prob(final_gen_batch_output)`
2. 把输出并回 batch，形成 `old_log_probs`
3. `batch.union(final_gen_batch_output)`
4. `_balance_batch(batch)`
5. 写入 `global_token_num`
6. 如果启用 reference policy，则算 `ref_log_prob`
7. 如果启用 critic，则算 `values`
8. 计算 reward
9. `_apply_opsd_token_level_kl(batch)`
10. `compute_advantage(...)`
11. `update_critic(batch)`
12. `update_actor(batch)`

其中：

- `old_log_probs` 来自 actor 对最终样本的重算
- `ref_log_prob` 来自 ref policy
- `token_level_scores` 来自 reward / rule-based reward
- `advantages` 是 PPO/GRPO 更新的直接输入

---

## 6. OPSD 在当前架构中的位置

当前 OPSD 不是在 rollout 时直接干预，而是在 trainer 中、reward 之后、advantage 之前插入。

顺序是：

```text
rollout -> old_log_probs/ref_log_prob -> reward -> OPSD token KL shaping -> advantage -> actor update
```

`_apply_opsd_token_level_kl()` 的作用是：

- 读取 `opsd_kl_data`
- 对 step 级片段做 student / teacher logprob 重算
- 形成额外 token-level shaping 信号
- 再与主奖励融合

因此当前主线依然是：

- rollout 负责生成轨迹
- OPSD 负责 credit shaping
- PPO/GRPO 负责最终参数更新

---

## 7. Worker 边界与元信息依赖

当前 search 分支最容易出问题的不是 tensor 本身，而是 `DataProto.meta_info`。

### 7.1 `compute_log_prob()` 依赖的元信息

`dp_actor.compute_log_prob()` 依赖：

- `micro_batch_size`
- `temperature`
- `use_dynamic_bsz`
- `max_token_len`

如果调用方只给了 tensor，没有把这些字段补齐，就会在 logprob 重算时出错。

### 7.2 `update_policy()` 依赖的元信息

`dp_actor.update_policy()` 依赖：

- `temperature`

如果 search 分支构造出来的 batch 没保留这个字段，actor update 阶段就会直接 `KeyError`。

---

## 8. 最近补上的两层兜底

这部分是“当前最新架构”和旧版本最不同的地方，因为它们已经直接写进代码主线。

### 8.1 中间 rollout 不再重算旧 logprob

在 `search_r1/llm_agent/generation.py` 的 `_generate_with_gpu_padding()` 中，当前显式写了：

```python
active_batch.meta_info['recompute_log_prob'] = False
```

这样做的原因是：

- agent loop 中间每一轮 rollout 只需要生成 token
- 不需要在 `generate_sequences()` 内部顺手重算 `old_log_probs`
- 真正有训练意义的是最终 `final_gen_batch_output` 的那次显式 `compute_log_prob()`

这让当前架构变成：

```text
中间多轮 rollout: 只生成
最终训练样本: 单独重算 old_log_probs
```

### 8.2 `compute_log_prob()` 入口兜底元信息

在 `verl/workers/fsdp_workers.py` 的 `compute_log_prob()` 中，当前会补默认：

- `micro_batch_size = self.config.rollout.log_prob_micro_batch_size`
- `max_token_len = self.config.rollout.log_prob_max_token_len_per_gpu`
- `use_dynamic_bsz = self.config.rollout.log_prob_use_dynamic_bsz`
- `temperature = self.config.rollout.temperature`

这使得 search 分支拼出来的最终 batch 即使没带全量 `meta_info`，也能安全进入 logprob 重算。

### 8.3 `update_actor()` 入口兜底 temperature

在 `verl/workers/fsdp_workers.py` 的 `update_actor()` 中，当前会做：

```python
if 'temperature' not in data.meta_info:
    data.meta_info['temperature'] = self.config.rollout.temperature
```

这样 actor 更新不再依赖上游必须手动传入 `temperature`。

---

## 9. 当前数据流的真实形态

当前主干数据流可以写成下面这样：

```text
Dataset sample
  -> repeat by n_agent
  -> gen_batch
  -> LLMGenerationManager.run_llm_loop
    -> active subset rollout
    -> search tool call
    -> observation injection
    -> final composed trajectory
  -> actor.compute_log_prob(final batch)
  -> ref.compute_ref_log_prob(batch)
  -> reward + OPSD KL shaping
  -> advantage
  -> critic update
  -> actor update
```

如果从对象角度看，核心载体始终是 `DataProto`：

- `batch` 里放 tensor
- `non_tensor_batch` 里放 uid、index、opsd 辅助信息
- `meta_info` 里放 rollout / update 所需控制参数

因此当前架构的一个关键结论是：

> Search-R1 这条主线本质上不是“几个独立模块松耦合拼起来”，而是“围绕 DataProto 持续扩写 tensor、meta_info、non_tensor 信息”的一条流水线。

---

## 10. 当前最值得注意的工程约束

### 10.1 Search 分支比普通 PPO 分支更脆弱

原因不是算法本身，而是：

- 有 active trajectory 缩减
- 有世界尺寸对齐 padding
- 有最终 batch 重组
- 有额外的 OPSD token-level 处理

这导致 search 分支对：

- `meta_info`
- `attention_mask`
- `responses`
- batch 对齐关系

都更敏感。

### 10.2 `meta_info` 是当前训练可用性的关键

最近两次实际报错已经说明：

- 一次缺 `micro_batch_size`
- 一次缺 `temperature`

所以当前架构里，`meta_info` 不是可有可无的附加字段，而是 worker 接口契约的一部分。

### 10.3 `ppo_mini_batch_size=256` 的含义

当前脚本里设置：

- `ppo_mini_batch_size=256`
- `ppo_micro_batch_size=32`

因此 actor update 时：

- 一个 mini-batch 会被进一步切成多个 micro-batch
- `gradient_accumulation = 256 / 32 = 8`

这和 `dp_actor.update_policy()` 的实现直接绑定。

---

## 11. 当前架构的一句话总结

当前最新架构可以概括为：

> Search-R1 现在是一条以 `DataProto` 为核心载体、以 `ray_trainer.py` 为主控、以 `LLMGenerationManager` 负责多轮 search-agent rollout、以 `fsdp_workers.py` 负责分布式接口兜底、并在 reward 到 advantage 之间插入 OPSD token-level shaping 的 GRPO 训练流水线。

如果后面继续演进，这份文档最值得优先同步的部分应该是：

- search 分支生成后的 batch 结构
- `meta_info` 的接口契约
- OPSD 插入位置
- actor/ref/critic 各自依赖的输入字段
