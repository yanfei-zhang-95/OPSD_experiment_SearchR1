# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
from collections import defaultdict
import torch
import numpy as np
from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re

def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle','r1searcher_stage2']:
        return qa_em.compute_score_em


class RewardManager():
    """The reward manager with OPSD capabilities.
    """

    def __init__(self, tokenizer, num_examine, format_score=0.) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.actor_rollout_wg = None
        
        # Split steps by observation boundaries instead of relying on optional
        # <think> tags, which may be missing in some responses.
        self.action_pattern = re.compile(r'<(search|answer)>(.*?)</\1>', re.DOTALL)
        self.information_pattern = re.compile(r'<information>(.*?)</information>', re.DOTALL)

    def set_rollout_wg(self, wg):
        """Inject the rollout worker group to allow OPSD to compute hints and alpha scores."""
        self.actor_rollout_wg = wg

    def _extract_question_text(self, data_item, prompt_str: str) -> str:
        prompt_chat = data_item.non_tensor_batch.get('prompt')
        if prompt_chat is not None:
            try:
                if isinstance(prompt_chat, list) and len(prompt_chat) > 0:
                    content = prompt_chat[-1].get('content', '')
                    match = re.search(r'Question:\s*(.*)', content, re.DOTALL)
                    if match:
                        return match.group(1).strip()
                    return content.strip()
            except Exception:
                pass

        raw_prompt = data_item.non_tensor_batch.get('raw_prompt')
        if raw_prompt is not None:
            try:
                if isinstance(raw_prompt, list) and len(raw_prompt) > 0:
                    content = raw_prompt[-1].get('content', '')
                    match = re.search(r'Question:\s*(.*)', content, re.DOTALL)
                    if match:
                        return match.group(1).strip()
                    return content.strip()
            except Exception:
                pass

        match = re.search(r'Question:\s*(.*?)(?:\n\s*assistant\b|$)', prompt_str, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return prompt_str.strip()

    def _truncate_by_tokens(self, text: str, max_tokens: int) -> str:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return text
        return self.tokenizer.decode(token_ids[-max_tokens:], skip_special_tokens=True)

    def _strip_information_tags(self, text: str) -> str:
        return text.replace("<information>", "").replace("</information>", "")

    def _remove_information_block(self, text: str) -> str:
        return self.information_pattern.sub("", text)

    def _format_ground_truth_answer(self, ground_truth) -> str:
        if isinstance(ground_truth, dict):
            target = ground_truth.get('target', ground_truth)
        else:
            target = ground_truth

        if isinstance(target, (list, tuple)):
            target = " | ".join(str(x) for x in target)
        elif not isinstance(target, str):
            target = str(target)

        return self._truncate_by_tokens(target, max_tokens=128)

    def _score_to_correctness_label(self, score) -> str:
        try:
            return "correct" if float(score) == 1.0 else "incorrect_or_other"
        except Exception:
            return "incorrect_or_other"

    def _build_masked_history_for_step(self, response_str: str, step_start_char_idx: int, step_end_char_idx: int) -> str:
        step_text = response_str[step_start_char_idx:step_end_char_idx]
        obs_match = self.information_pattern.search(step_text)

        if obs_match is None:
            masked_step_text = "[MASKED_CURRENT_STEP]"
        else:
            observation_block = step_text[obs_match.start():obs_match.end()]
            masked_step_text = "[MASKED_CURRENT_STEP]\n" + observation_block

        return response_str[:step_start_char_idx] + masked_step_text + response_str[step_end_char_idx:]

    def _build_opsd_prompts(self,
                            question_text: str,
                            masked_history_for_step: str,
                            action_type: str,
                            correctness_label: str,
                            ground_truth_answer: str):
        truncated_history = self._truncate_by_tokens(masked_history_for_step, max_tokens=1536)

        student_messages = [
            {"role": "system", "content": (
                "You are predicting the current step text for a trajectory. "
                "Use the provided context to score the current step continuation. "
                "Output only the current step text."
            )},
            {"role": "user", "content": (
                f"Question: {question_text}\n\n"
                f"Trajectory with the current step masked:\n{truncated_history}\n\n"
                "Predict the current step text that should continue from this masked trajectory context. "
                "Output only the current step text."
            )}
        ]

        if action_type == 'answer':
            outcome_info = f"Final correctness: {correctness_label}"
        else:
            outcome_info = (
                f"Final correctness: {correctness_label}\n"
                f"Reference answer: {ground_truth_answer}"
            )

        teacher_messages = [
            {"role": "system", "content": (
                "You are a privileged teacher for step-wise RL self-distillation. "
                "Use the same masked trajectory context as the student, plus hindsight outcome information, "
                "to score the current step continuation. "
                "Output only the current step text."
            )},
            {"role": "user", "content": (
                f"Question: {question_text}\n\n"
                f"Trajectory with the current step masked:\n{truncated_history}\n\n"
                f"Hindsight outcome information:\n{outcome_info}\n\n"
                "Predict the current step text that should continue from this masked trajectory context. "
                "Output only the current step text."
            )}
        ]

        student_prompt_ids = self.tokenizer.apply_chat_template(
            student_messages,
            tokenize=True,
            add_generation_prompt=True
        )
        teacher_prompt_ids = self.tokenizer.apply_chat_template(
            teacher_messages,
            tokenize=True,
            add_generation_prompt=True
        )

        student_prompt = self.tokenizer.decode(student_prompt_ids, skip_special_tokens=False)
        teacher_prompt = self.tokenizer.decode(teacher_prompt_ids, skip_special_tokens=False)
        return student_prompt, teacher_prompt

    def _split_response_into_steps(self, response_str: str):
        """Split a trajectory by </information> and keep the final tail step."""
        steps = []
        cursor = 0
        response_len = len(response_str)
        info_end_tag = "</information>"

        while cursor < response_len:
            next_info_end = response_str.find(info_end_tag, cursor)
            if next_info_end == -1:
                step_end_char_idx = response_len
            else:
                step_end_char_idx = next_info_end + len(info_end_tag)

            step_text = response_str[cursor:step_end_char_idx]
            if step_text.strip():
                steps.append({
                    'step_start_char_idx': cursor,
                    'step_end_char_idx': step_end_char_idx,
                    'step_text': step_text,
                })

            if next_info_end == -1:
                break

            cursor = step_end_char_idx
            while cursor < response_len and response_str[cursor].isspace():
                cursor += 1

        return steps

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""
        
        print(f"\n[OPSD DEBUG] -------- RewardManager called! actor_rollout_wg is {'SET' if self.actor_rollout_wg is not None else 'NONE'} --------")

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}
        all_scores = []
        valid_response_lengths = []
        
        opsd_metadata = []

        batch_size = len(data)
        for i in range(batch_size):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            valid_response_lengths.append(valid_response_length.item())

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            # Macro Reward R
            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score)
            all_scores.append(score)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)
                
            # OPSD Step Extraction
            if self.actor_rollout_wg is not None:
                step_segments = self._split_response_into_steps(response_str)
                for step_segment in step_segments:
                    step_start_char_idx = step_segment['step_start_char_idx']
                    end_char_idx = step_segment['step_end_char_idx']
                    step_text = step_segment['step_text']
                    clean_step_text = self._remove_information_block(step_text)

                    action_match = self.action_pattern.search(step_text)
                    if action_match is None:
                        continue

                    action_type = action_match.group(1)
                    action_content = action_match.group(2)
                    obs_match = self.information_pattern.search(step_text)
                    clean_step_end_char_idx = end_char_idx
                    if obs_match is not None:
                        clean_step_end_char_idx = step_start_char_idx + obs_match.start()

                    prefix_str = response_str[:end_char_idx]
                    clean_prefix_str = self._strip_information_tags(prefix_str)
                    question_text = self._extract_question_text(data_item, prompt_str)
                    action_str = f"<{action_type}>{action_content}</{action_type}>"
                    masked_history_for_step = self._build_masked_history_for_step(
                        response_str=response_str,
                        step_start_char_idx=step_start_char_idx,
                        step_end_char_idx=end_char_idx,
                    )
                    correctness_label = self._score_to_correctness_label(score)
                    ground_truth_answer = self._format_ground_truth_answer(ground_truth)
                    student_prompt, teacher_prompt = self._build_opsd_prompts(
                        question_text=question_text,
                        masked_history_for_step=masked_history_for_step,
                        action_type=action_type,
                        correctness_label=correctness_label,
                        ground_truth_answer=ground_truth_answer,
                    )
                    
                    # Approximate token index by tokenizing the prefix
                    start_prefix_tokens = self.tokenizer.encode(response_str[:step_start_char_idx], add_special_tokens=False)
                    prefix_tokens = self.tokenizer.encode(prefix_str, add_special_tokens=False)
                    token_start_idx = len(start_prefix_tokens)
                    token_end_idx = len(prefix_tokens) - 1
                    token_start_idx = max(0, min(token_start_idx, valid_response_length.item() - 1))
                    token_end_idx = max(0, min(token_end_idx, valid_response_length.item() - 1))

                    clean_step_end_tokens = self.tokenizer.encode(
                        response_str[:clean_step_end_char_idx], add_special_tokens=False
                    )
                    clean_step_token_end_idx = len(clean_step_end_tokens) - 1
                    clean_step_token_end_idx = max(0, min(clean_step_token_end_idx, valid_response_length.item() - 1))

                    action_start_char_idx = response_str.find(action_str, step_start_char_idx, end_char_idx)
                    if action_start_char_idx < 0:
                        action_start_char_idx = step_start_char_idx
                    action_end_char_idx = action_start_char_idx + len(action_str)
                    action_start_tokens = self.tokenizer.encode(
                        response_str[:action_start_char_idx], add_special_tokens=False
                    )
                    action_end_tokens = self.tokenizer.encode(
                        response_str[:action_end_char_idx], add_special_tokens=False
                    )
                    action_token_start_idx = len(action_start_tokens)
                    action_token_end_idx = len(action_end_tokens) - 1
                    action_token_start_idx = max(0, min(action_token_start_idx, valid_response_length.item() - 1))
                    action_token_end_idx = max(0, min(action_token_end_idx, valid_response_length.item() - 1))
                    
                    opsd_metadata.append({
                        'batch_idx': i,
                        'token_start_idx': token_start_idx,
                        'token_end_idx': token_end_idx,
                        'clean_step_token_end_idx': clean_step_token_end_idx,
                        'action_token_start_idx': action_token_start_idx,
                        'action_token_end_idx': action_token_end_idx,
                        'action_type': action_type,
                        'masked_history_for_step': masked_history_for_step,
                        'correctness_label': correctness_label,
                        'action_str': action_str,
                        'step_text': step_text,
                        'clean_step_text': clean_step_text,
                        'original_prefix_str': clean_prefix_str,
                        'student_prompt': student_prompt,
                        'teacher_prompt': teacher_prompt,
                    })
                    
                    if i == 0:  # Debug print for the first trajectory only
                        print(f"[OPSD DEBUG] Extracted Step in Traj 0 | Action: {action_type} | Token End Idx: {token_end_idx}")
                        if len(opsd_metadata) == 1:
                            print(f"[OPSD DEBUG] Exact Student Prompt for Step 1:\n{student_prompt}\n" + "-" * 50)
                            print(f"[OPSD DEBUG] Exact Teacher Prompt for Step 1:\n{teacher_prompt}\n" + "-" * 50)

        # Process OPSD Dense Rewards if we extracted steps. Teacher logits will
        # be recomputed later under this prompt without an intermediate hint generation step.
        if self.actor_rollout_wg is not None and opsd_metadata:
            print(f"\n[OPSD DEBUG] Total OPSD steps extracted across batch: {len(opsd_metadata)}")

            batch_steps = defaultdict(list)
            for meta in opsd_metadata:
                batch_steps[meta['batch_idx']].append({
                    'token_start_idx': meta['token_start_idx'],
                    'token_end_idx': meta['token_end_idx'],
                    'clean_step_token_end_idx': meta['clean_step_token_end_idx'],
                    'action_token_start_idx': meta['action_token_start_idx'],
                    'action_token_end_idx': meta['action_token_end_idx'],
                    'masked_history_for_step': meta['masked_history_for_step'],
                    'correctness_label': meta['correctness_label'],
                    'action_str': meta['action_str'],
                    'action_type': meta['action_type'],
                    'step_text': meta['step_text'],
                    'clean_step_text': meta['clean_step_text'],
                    'original_prefix_str': meta['original_prefix_str'],
                    'student_prompt': meta['student_prompt'],
                    'teacher_prompt': meta['teacher_prompt'],
                })
                
            opsd_kl_data = np.empty((batch_size,), dtype=object)
                
            for i in range(batch_size):
                macro_score = all_scores[i]
                step_info = batch_steps.get(i, [])
                
                opsd_kl_data[i] = step_info  # Save for KL in ray_trainer.py
                reward_tensor[i, valid_response_lengths[i] - 1] = macro_score
                if i == 0:
                    print(f"[OPSD DEBUG] Traj 0 | Macro Reward: {macro_score} | Assigned sparse reward at final token")
                    
            # Inject OPSD data back into the batch for KL computation
            data.non_tensor_batch['opsd_kl_data'] = opsd_kl_data
        else:
            # Fallback to pure sparse reward if OPSD is inactive
            for i in range(batch_size):
                reward_tensor[i, valid_response_lengths[i] - 1] = all_scores[i]

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {
            'TOKENIZERS_PARALLELISM': 'true',
            'NCCL_DEBUG': 'WARN',
            'RAY_DEBUG': 'legacy',
        }})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    
    # Inject the actor rollout worker group into the reward function for OPSD
    # This avoids modifying the internal verl/trainer/ppo/ray_trainer.py
    if hasattr(trainer.reward_fn, 'set_rollout_wg'):
        trainer.reward_fn.set_rollout_wg(trainer.actor_rollout_wg)
        
    trainer.fit()


if __name__ == '__main__':
    main()
