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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
import verl.utils.hdfs_io as hdfs_io
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['info_mask'] if 'info_mask' in data.batch else data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    _ = num_repeat
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError

    opsd_token_delta = data.batch.get('opsd_token_delta', None)
    if opsd_token_delta is not None:
        advantages = data.batch['advantages']
        response_length = data.batch['responses'].size(-1)
        response_mask = data.batch['attention_mask'][:, -response_length:].float()

        opsd_weight_clip = float(data.meta_info.get('opsd_weight_clip', 0.2))
        opsd_mix_lambda = float(data.meta_info.get('opsd_mix_lambda', 1.0))
        opsd_weight_fn = str(data.meta_info.get('opsd_weight_fn', 'sigmoid'))

        signed_delta = torch.sign(advantages) * opsd_token_delta.detach()
        # Compute token-level evidence weights, then apply clipping directly to
        # the per-token advantage as in RLSD-style reweighting.
        if opsd_weight_fn == 'sigmoid':
            opsd_raw_weights = 2.0 * torch.sigmoid(signed_delta)
        elif opsd_weight_fn == 'exp':
            opsd_raw_weights = torch.exp(signed_delta)
        else:
            raise ValueError(
                f"Unsupported opsd_weight_fn={opsd_weight_fn}. "
                "Expected 'sigmoid' or 'exp'."
            )
        opsd_token_weights = opsd_raw_weights.clamp(
            min=1.0 - opsd_weight_clip,
            max=1.0 + opsd_weight_clip,
        )
        reweighted_advantages = advantages * opsd_token_weights
        shaped_advantages = (
            (1.0 - opsd_mix_lambda) * advantages
            + opsd_mix_lambda * reweighted_advantages
        )
        shaped_advantages = shaped_advantages * response_mask

        opsd_step_advantage_norm = str(data.meta_info.get('opsd_step_advantage_norm', 'none'))
        if opsd_step_advantage_norm not in ('none', 'equal_step_mean_abs'):
            raise ValueError(
                f"Unsupported opsd_step_advantage_norm={opsd_step_advantage_norm}. "
                "Expected 'none' or 'equal_step_mean_abs'."
            )
        if opsd_step_advantage_norm == 'equal_step_mean_abs':
            opsd_kl_data = data.non_tensor_batch.get('opsd_kl_data', None)
            if opsd_kl_data is not None:
                opsd_step_advantage_norm_eps = float(data.meta_info.get('opsd_step_advantage_norm_eps', 1e-6))
                opsd_step_advantage_coef_clip = float(data.meta_info.get('opsd_step_advantage_coef_clip', 2.0))
                clip_min = 1.0 / max(opsd_step_advantage_coef_clip, 1.0)
                clip_max = max(opsd_step_advantage_coef_clip, 1.0)

                normalized_advantages = shaped_advantages.clone()
                batch_size = normalized_advantages.size(0)
                for batch_idx in range(min(batch_size, len(opsd_kl_data))):
                    step_list = opsd_kl_data[batch_idx]
                    if not step_list or len(step_list) <= 1:
                        continue

                    step_spans = []
                    step_mean_abs = []
                    for step in step_list:
                        start_idx = int(step.get('target_token_start_idx', -1))
                        end_idx = int(step.get('target_token_end_idx', -1))
                        if end_idx < start_idx:
                            continue

                        step_mask = response_mask[batch_idx, start_idx:end_idx + 1] > 0
                        if not torch.any(step_mask):
                            continue

                        step_values = normalized_advantages[batch_idx, start_idx:end_idx + 1][step_mask]
                        step_spans.append((start_idx, end_idx))
                        step_mean_abs.append(step_values.abs().mean())

                    if len(step_mean_abs) <= 1:
                        continue

                    step_mean_abs_tensor = torch.stack(step_mean_abs)
                    target_mean_abs = step_mean_abs_tensor.mean()
                    step_coefs = target_mean_abs / (step_mean_abs_tensor + opsd_step_advantage_norm_eps)
                    step_coefs = torch.clamp(step_coefs, min=clip_min, max=clip_max)

                    for (start_idx, end_idx), step_coef in zip(step_spans, step_coefs):
                        normalized_advantages[batch_idx, start_idx:end_idx + 1] *= step_coef

                shaped_advantages = normalized_advantages * response_mask

        data.batch['opsd_token_weights'] = opsd_token_weights * response_mask
        data.batch['advantages'] = shaped_advantages
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # metrics for actions
    if 'turns_stats' in batch.meta_info:
        metrics['env/number_of_actions/mean'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).mean())
        metrics['env/number_of_actions/max'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).max())
        metrics['env/number_of_actions/min'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).min())
    if 'active_mask' in batch.meta_info:
        metrics['env/finish_ratio'] = 1 - float(np.array(batch.meta_info['active_mask'], dtype=np.int16).mean())
    if 'valid_action_stats' in batch.meta_info:
        metrics['env/number_of_valid_action'] = float(np.array(batch.meta_info['valid_action_stats'], dtype=np.int16).mean())
        metrics['env/ratio_of_valid_action'] = float((np.array(batch.meta_info['valid_action_stats'], dtype=np.int16) / np.array(batch.meta_info['turns_stats'], dtype=np.int16)).mean())
    if 'valid_search_stats' in batch.meta_info:
        metrics['env/number_of_valid_search'] = float(np.array(batch.meta_info['valid_search_stats'], dtype=np.int16).mean())


    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor', 'rollout']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.resume_from_path = self._get_resume_path()
        self.resume_state = None
        self.resume_epoch = 0
        self.resume_batch_offset = 0
        self.resume_next_global_step = None

        if self.resume_from_path is not None:
            self._configure_resume()

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.resume_state is not None and 'kl_ctrl_value' in self.resume_state:
            self.kl_ctrl.value = float(self.resume_state['kl_ctrl_value'])

        self._create_dataloader()
        self._init_logger()

    def _get_resume_path(self):
        resume_path = self.config.trainer.get('resume_from_path', None)
        if resume_path in (None, '', 'null'):
            return None
        return str(resume_path)

    def _get_trainer_state_path(self, actor_checkpoint_path: str) -> str:
        return os.path.join(actor_checkpoint_path, 'trainer_state.pt')

    def _derive_critic_resume_path(self, actor_checkpoint_path: str) -> str:
        step_dir = os.path.basename(actor_checkpoint_path.rstrip('/'))
        run_root = os.path.dirname(os.path.dirname(actor_checkpoint_path.rstrip('/')))
        return os.path.join(run_root, 'critic', step_dir)

    def _load_trainer_state(self, actor_checkpoint_path: str):
        local_actor_path = copy_local_path_from_hdfs(actor_checkpoint_path)
        trainer_state_path = self._get_trainer_state_path(local_actor_path)
        if not os.path.exists(trainer_state_path):
            raise FileNotFoundError(f'Trainer state not found: {trainer_state_path}')
        return torch.load(trainer_state_path, map_location='cpu')

    def _configure_resume(self):
        self.resume_state = self._load_trainer_state(self.resume_from_path)
        last_saved_step = int(self.resume_state['global_steps'])
        self.resume_next_global_step = last_saved_step + 1

        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.resume_from_path = self.resume_from_path
            if self.config.algorithm.adv_estimator == 'gae':
                self.config.critic.resume_from_path = self._derive_critic_resume_path(self.resume_from_path)

        print(f'[RESUME] Preparing to resume from {self.resume_from_path}. '
              f'Last saved global step={last_saved_step}, next step={self.resume_next_global_step}.')
    
    def _init_logger(self):
        from verl.utils.tracking import Tracking
        self.logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True),
                          wandb_run_id=self.config.trainer.get('wandb_run_id', None),
                          wandb_resume=self.config.trainer.get('wandb_resume', None))

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        if self.config.data.train_data_num is not None:
            if self.config.data.train_data_num > len(self.train_dataset.dataframe):
                print(f"[WARNING] training dataset size is smaller than desired size. Using the dataset as the original size {len(self.train_dataset.dataframe)}")
            else:
                self.train_dataset.dataframe = self.train_dataset.dataframe.sample(self.config.data.train_data_num, random_state=42)
        print(f"filtered training dataset size: {len(self.train_dataset.dataframe)}")

        train_generator = None
        if self.config.data.shuffle_train_dataloader:
            train_seed = int(self.config.trainer.get('seed', 42))
            train_generator = torch.Generator()
            train_generator.manual_seed(train_seed)

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=self.config.data.shuffle_train_dataloader,
                                           drop_last=True,
                                           collate_fn=collate_fn,
                                           generator=train_generator)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        if self.config.data.val_data_num is not None:
            if self.config.data.val_data_num > len(self.val_dataset.dataframe):
                print(f"[WARNING] validation dataset size is smaller than desired size. Using the dataset as the original size {len(self.val_dataset.dataframe)}")
            else:
                self.val_dataset.dataframe = self.val_dataset.dataframe.sample(self.config.data.val_data_num, random_state=42)
        print(f"filtered validation dataset size: {len(self.val_dataset.dataframe)}")

        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.config.data.val_batch_size,
                                         shuffle=False,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')
        
        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        if self.resume_next_global_step is not None:
            completed_steps = max(0, self.resume_next_global_step - 1)
            self.resume_epoch = completed_steps // len(self.train_dataloader)
            self.resume_batch_offset = completed_steps % len(self.train_dataloader)
            print(f'[RESUME] Will skip to epoch {self.resume_epoch}, batch offset {self.resume_batch_offset}.')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _build_left_padded_logprob_batch(self, prompt_ids_list, response_ids_list, micro_batch_size=128):
        pad_token_id = self.tokenizer.pad_token_id
        batch_size = len(prompt_ids_list)
        max_prompt_len = max(x.numel() for x in prompt_ids_list)
        max_resp_len = max(x.numel() for x in response_ids_list)
        seq_len = max_prompt_len + max_resp_len

        input_ids = torch.full((batch_size, seq_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)
        responses = torch.full((batch_size, max_resp_len), pad_token_id, dtype=torch.long)

        for i, (prompt_ids, response_ids) in enumerate(zip(prompt_ids_list, response_ids_list)):
            p_len = prompt_ids.numel()
            r_len = response_ids.numel()
            input_ids[i, max_prompt_len - p_len:max_prompt_len] = prompt_ids
            input_ids[i, max_prompt_len:max_prompt_len + r_len] = response_ids
            attention_mask[i, max_prompt_len - p_len:max_prompt_len + r_len] = 1
            responses[i, :r_len] = response_ids

        position_ids = attention_mask.cumsum(dim=1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)

        return DataProto.from_dict(
            tensors={
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'responses': responses,
            },
            meta_info={
                'micro_batch_size': micro_batch_size,
                'temperature': 1.0,
                'use_dynamic_bsz': False,
                'max_token_len': seq_len,
            }
        )

    def _format_opsd_token_for_display(self, token_id: int) -> str:
        token_text = self.tokenizer.decode([int(token_id)], skip_special_tokens=False)
        if token_text == '':
            token_text = self.tokenizer.convert_ids_to_tokens([int(token_id)])[0]
        token_text = token_text.replace("\n", "\\n").replace("\t", "\\t")
        return token_text

    def _annotate_opsd_trajectory_spans(
        self,
        text: str,
        step_char_span: tuple[int, int],
        action_char_span: tuple[int, int],
    ) -> str:
        text_len = len(text)
        step_start, step_end = step_char_span
        action_start, action_end = action_char_span

        def _clip(idx: int) -> int:
            return max(0, min(int(idx), text_len))

        markers = [
            (_clip(step_start), "[STEP_START]"),
            (_clip(action_start), "[ACTION_START]"),
            (_clip(action_end), "[ACTION_END]"),
            (_clip(step_end), "[STEP_END]"),
        ]
        markers.sort(key=lambda x: (x[0], x[1]))

        parts = []
        cursor = 0
        for idx, marker in markers:
            if idx < cursor:
                idx = cursor
            parts.append(text[cursor:idx])
            parts.append(marker)
            cursor = idx
        parts.append(text[cursor:])
        return "".join(parts).replace("\n", "\\n")

    def _format_opsd_delta_case_report(self, case: dict, delta_threshold: float) -> str:
        changed_entries = []
        unchanged_entries = []
        ordered_entries = []

        for token_idx, (token_id, delta_val, student_lp, teacher_lp) in enumerate(
                zip(case['token_ids'], case['delta'], case['student_logprob'], case['teacher_logprob'])):
            token_text = self._format_opsd_token_for_display(token_id)
            status = 'changed' if abs(delta_val) > delta_threshold else 'unchanged'
            entry = (
                f"{token_idx:02d} | token={repr(token_text)} | "
                f"status={status} | delta={delta_val:+.4f} | "
                f"student_lp={student_lp:+.4f} | teacher_lp={teacher_lp:+.4f}"
            )
            ordered_entries.append(entry)
            if abs(delta_val) > delta_threshold:
                changed_entries.append(entry)
            else:
                unchanged_entries.append(entry)

        def _render_entries(entries: list[str]) -> str:
            if not entries:
                return 'None'
            return "\n".join(entries)

        step_preview = case['step_text'].replace("\n", "\\n")
        trajectory_preview = case['full_trajectory_text'].replace("\n", "\\n")
        annotated_trajectory_preview = self._annotate_opsd_trajectory_spans(
            case['full_trajectory_text'],
            case['step_char_span'],
            case['target_char_span'],
        )
        action_preview = case['action_text'].replace("\n", "\\n")
        target_preview = case['target_text'].replace("\n", "\\n")
        return (
            "[OPSD DELTA REPORT] Sampled Case\n"
            f"[Batch Idx] {case['batch_idx']}\n"
            f"[Step Idx] {case['step_idx']}\n"
            f"[Action Type] {case['action_type']}\n"
            f"[Target Span Mode] {case['target_span_mode']}\n"
            f"[Changed Tokens] {len(changed_entries)} / {len(case['token_ids'])}\n"
            f"[Step Char Span] {case['step_char_span'][0]} -> {case['step_char_span'][1]}\n"
            f"[Action Char Span] {case['action_char_span'][0]} -> {case['action_char_span'][1]}\n"
            f"[Target Char Span] {case['target_char_span'][0]} -> {case['target_char_span'][1]}\n"
            f"[Step Text]\n{step_preview}\n"
            f"[Action Text]\n{action_preview}\n"
            f"[Target Text]\n{target_preview}\n"
            f"[Full Trajectory]\n{trajectory_preview}\n"
            f"[Annotated Trajectory]\n{annotated_trajectory_preview}\n"
            "[All Token Details In Order]\n"
            f"{_render_entries(ordered_entries)}\n"
            "[Changed Token Details]\n"
            f"{_render_entries(changed_entries)}\n"
            "[Unchanged Token Details]\n"
            f"{_render_entries(unchanged_entries)}\n"
            + '-' * 80
        )

    def _build_opsd_delta_report_text(self, delta_values: list[float], sampled_cases: list[dict], delta_threshold: float) -> str:
        if not delta_values:
            return ''

        delta_arr = np.asarray(delta_values, dtype=np.float32)
        abs_delta_arr = np.abs(delta_arr)
        changed_mask = abs_delta_arr > delta_threshold
        positive_count = int((delta_arr > delta_threshold).sum())
        negative_count = int((delta_arr < -delta_threshold).sum())
        unchanged_count = int((~changed_mask).sum())

        report_sections = [
            (
                "[OPSD DELTA REPORT] Summary\n"
                f"[Token Count] {delta_arr.size}\n"
                f"[Change Threshold] {delta_threshold:.2e}\n"
                f"[Mean] {delta_arr.mean():+.6f}\n"
                f"[Std] {delta_arr.std():.6f}\n"
                f"[Min, Max] {delta_arr.min():+.6f}, {delta_arr.max():+.6f}\n"
                f"[Abs Mean, Abs Median] {abs_delta_arr.mean():.6f}, {np.median(abs_delta_arr):.6f}\n"
                f"[P05, P25, P50, P75, P95] "
                f"{np.percentile(delta_arr, 5):+.6f}, "
                f"{np.percentile(delta_arr, 25):+.6f}, "
                f"{np.percentile(delta_arr, 50):+.6f}, "
                f"{np.percentile(delta_arr, 75):+.6f}, "
                f"{np.percentile(delta_arr, 95):+.6f}\n"
                f"[Positive, Negative, Unchanged] {positive_count}, {negative_count}, {unchanged_count}\n"
                f"[Changed Ratio] {changed_mask.mean():.4f}\n"
                + '-' * 80
            )
        ]
        for case in sampled_cases:
            report_sections.append(self._format_opsd_delta_case_report(case, delta_threshold=delta_threshold))
        return "\n".join(report_sections)

    def _print_opsd_delta_report(self, delta_values: list[float], sampled_cases: list[dict], delta_threshold: float):
        report_text = self._build_opsd_delta_report_text(
            delta_values=delta_values,
            sampled_cases=sampled_cases,
            delta_threshold=delta_threshold,
        )
        if not report_text:
            return

        print(report_text)
        if getattr(self, '_opsd_validate_report_sink', None) is not None:
            self._opsd_validate_report_sink.append(report_text)

    def _apply_opsd_token_level_kl(self, batch: DataProto):
        if 'opsd_kl_data' not in batch.non_tensor_batch:
            return batch

        opsd_kl_data = batch.non_tensor_batch['opsd_kl_data']
        if opsd_kl_data is None or len(opsd_kl_data) == 0:
            return batch

        logprob_micro_batch_size = self.config.actor_rollout_ref.rollout.log_prob_micro_batch_size
        try:
            opsd_logprob_chunk_size = int(self.config.actor_rollout_ref.rollout.opsd_logprob_chunk_size)
        except Exception:
            opsd_logprob_chunk_size = min(32, int(logprob_micro_batch_size))
        opsd_logprob_chunk_size = max(1, opsd_logprob_chunk_size)
        try:
            opsd_student_scoring_mode = str(self.config.algorithm.opsd_student_scoring_mode)
        except Exception:
            opsd_student_scoring_mode = 'causal_prefix'
        if opsd_student_scoring_mode == 'masked_prompt':
            opsd_student_scoring_mode = 'causal_prefix'
        if opsd_student_scoring_mode not in ('causal_prefix', 'rollout_old_log_prob'):
            raise ValueError(
                f"Unsupported algorithm.opsd_student_scoring_mode={opsd_student_scoring_mode}. "
                "Expected 'causal_prefix' or 'rollout_old_log_prob'."
            )
        try:
            opsd_teacher_mode = str(self.config.algorithm.opsd_teacher_mode)
        except Exception:
            opsd_teacher_mode = 'stale_ref_policy' if self.use_reference_policy else 'live_actor'
        if opsd_teacher_mode not in ('live_actor', 'stale_ref_policy'):
            raise ValueError(
                f"Unsupported algorithm.opsd_teacher_mode={opsd_teacher_mode}. "
                "Expected 'live_actor' or 'stale_ref_policy'."
            )
        if opsd_teacher_mode == 'stale_ref_policy' and not self.use_reference_policy:
            opsd_teacher_mode = 'live_actor'

        step_records = []
        debug_examples = []

        for batch_idx in range(len(batch)):
            step_list = opsd_kl_data[batch_idx]
            if step_list is None:
                continue
            for step_idx, step in enumerate(step_list):
                student_context_ids = step.get('student_context_ids', None)
                teacher_context_ids = step.get('teacher_context_ids', None)
                target_response_ids = step.get('target_response_ids', None)
                clean_step_text = step.get('clean_step_text', '')
                if teacher_context_ids is None or target_response_ids is None:
                    continue
                if opsd_student_scoring_mode == 'causal_prefix' and student_context_ids is None:
                    continue

                student_ids = None
                if opsd_student_scoring_mode == 'causal_prefix':
                    student_ids = torch.tensor(student_context_ids, dtype=torch.long)
                teacher_ids = torch.tensor(teacher_context_ids, dtype=torch.long)
                response_ids = torch.tensor(target_response_ids, dtype=torch.long)
                if response_ids.numel() == 0:
                    continue

                meta = {
                    'batch_idx': batch_idx,
                    'step_idx': step_idx,
                    'resp_len': response_ids.numel(),
                    'step_token_start_idx': step['target_token_start_idx'],
                    'step_token_end_idx': step['target_token_end_idx'],
                    'reward_anchor_idx': step['token_end_idx'],
                    'student_scoring_mode': opsd_student_scoring_mode,
                    'action_type': step.get('action_type', 'unknown'),
                    'target_span_mode': step.get('target_span_mode', 'clean_step_no_observation'),
                    'full_trajectory_text': step.get('full_trajectory_text', ''),
                    'action_text': step.get('action_str', ''),
                    'target_text': step.get('target_text', clean_step_text),
                    'step_char_span': (
                        step.get('step_start_char_idx', -1),
                        step.get('step_end_char_idx', -1),
                    ),
                    'action_char_span': (
                        step.get('action_start_char_idx', -1),
                        step.get('action_end_char_idx', -1),
                    ),
                    'target_char_span': (
                        step.get('target_char_start_idx', -1),
                        step.get('target_char_end_idx', -1),
                    ),
                }
                step_records.append({
                    'student_ids': student_ids,
                    'teacher_ids': teacher_ids,
                    'response_ids': response_ids,
                    'meta': meta,
                })
                if batch_idx == 0 and len(debug_examples) < 3:
                    debug_examples.append({
                        'step_idx': step_idx,
                        'student_context': self.tokenizer.decode(student_context_ids, skip_special_tokens=False)
                        if student_context_ids is not None else '',
                        'teacher_context': self.tokenizer.decode(teacher_context_ids, skip_special_tokens=False),
                        'full_trajectory_text': step.get('full_trajectory_text', ''),
                        'step_text': step.get('step_text', ''),
                        'action_text': step.get('action_str', ''),
                        'target_text': step.get('target_text', clean_step_text),
                        'target_span_mode': step.get('target_span_mode', 'clean_step_no_observation'),
                        'target_response': step.get('target_text', clean_step_text),
                        'step_token_start_idx': step['target_token_start_idx'],
                        'step_token_end_idx': step['target_token_end_idx'],
                        'step_char_span': (step.get('step_start_char_idx', -1), step.get('step_end_char_idx', -1)),
                        'action_char_span': (step.get('action_start_char_idx', -1), step.get('action_end_char_idx', -1)),
                        'target_char_span': (step.get('target_char_start_idx', -1), step.get('target_char_end_idx', -1)),
                    })

        if not step_records:
            return batch

        for ex in debug_examples:
            print(
                "[OPSD KL DEBUG] Teacher Example\n"
                f"[Step Idx] {ex['step_idx']}\n"
                f"[Teacher Mode] {opsd_teacher_mode}\n"
                f"[Student Scoring Mode] {opsd_student_scoring_mode}\n"
                f"[Student Context]\n{ex['student_context']}\n"
                f"[Teacher Context]\n{ex['teacher_context']}\n"
                f"[Full Trajectory]\n{ex['full_trajectory_text']}\n"
                f"[Original Step Text]\n{ex['step_text']}\n"
                f"[Action Text]\n{ex['action_text']}\n"
                f"[Target Span Mode] {ex['target_span_mode']}\n"
                f"[Target Text]\n{ex['target_text']}\n"
                f"[Step Char Span] {ex['step_char_span'][0]} -> {ex['step_char_span'][1]}\n"
                f"[Action Char Span] {ex['action_char_span'][0]} -> {ex['action_char_span'][1]}\n"
                f"[Target Char Span] {ex['target_char_span'][0]} -> {ex['target_char_span'][1]}\n"
                f"[Target Response]\n{ex['target_response']}\n"
                f"[Step Token Span] {ex['step_token_start_idx']} -> {ex['step_token_end_idx']}\n"
                + "-" * 80
            )

        opsd_token_delta = torch.zeros_like(batch.batch['token_level_scores'], dtype=torch.float32)
        total_step_records = len(step_records)
        total_chunks = (total_step_records + opsd_logprob_chunk_size - 1) // opsd_logprob_chunk_size
        try:
            opsd_report_delta_threshold = float(self.config.algorithm.opsd_report_delta_threshold)
        except Exception:
            opsd_report_delta_threshold = 1e-4
        try:
            opsd_report_case_count = int(self.config.algorithm.opsd_report_case_count)
        except Exception:
            opsd_report_case_count = 8
        opsd_report_case_count = max(1, opsd_report_case_count)
        delta_values = []
        sampled_cases = []
        seen_case_count = 0

        print(
            "[OPSD KL DEBUG] Chunked LogProb Recompute\n"
            f"[Total Step Instances] {total_step_records}\n"
            f"[Chunk Size] {opsd_logprob_chunk_size}\n"
            f"[Total Chunks] {total_chunks}\n"
            f"[Teacher Mode] {opsd_teacher_mode}\n"
            f"[Student Scoring Mode] {opsd_student_scoring_mode}\n"
            + "-" * 80
        )

        for chunk_idx in range(0, total_step_records, opsd_logprob_chunk_size):
            chunk_records = step_records[chunk_idx:chunk_idx + opsd_logprob_chunk_size]

            student_batch = None
            student_batch_padded = None
            student_logprob_output_padded = None
            student_logprob_output = None
            if opsd_student_scoring_mode == 'causal_prefix':
                student_batch = self._build_left_padded_logprob_batch(
                    [record['student_ids'] for record in chunk_records],
                    [record['response_ids'] for record in chunk_records],
                    micro_batch_size=logprob_micro_batch_size,
                )
            teacher_batch = self._build_left_padded_logprob_batch(
                [record['teacher_ids'] for record in chunk_records],
                [record['response_ids'] for record in chunk_records],
                micro_batch_size=logprob_micro_batch_size,
            )

            if opsd_student_scoring_mode == 'causal_prefix':
                student_batch_padded, student_pad_size = pad_dataproto_to_divisor(
                    student_batch, self.actor_rollout_wg.world_size
                )
                student_logprob_output_padded = self.actor_rollout_wg.compute_log_prob(student_batch_padded)
                student_logprob_output = unpad_dataproto(student_logprob_output_padded, pad_size=student_pad_size)
                student_log_probs = student_logprob_output.batch['old_log_probs']
            else:
                student_log_probs = None

            teacher_batch_padded, teacher_pad_size = pad_dataproto_to_divisor(
                teacher_batch, self.actor_rollout_wg.world_size
            )
            if opsd_teacher_mode == 'stale_ref_policy':
                teacher_logprob_output_padded = self.ref_policy_wg.compute_ref_log_prob(teacher_batch_padded)
                teacher_logprob_output = unpad_dataproto(teacher_logprob_output_padded, pad_size=teacher_pad_size)
                teacher_log_probs = teacher_logprob_output.batch['ref_log_prob']
            else:
                teacher_logprob_output_padded = self.actor_rollout_wg.compute_log_prob(teacher_batch_padded)
                teacher_logprob_output = unpad_dataproto(teacher_logprob_output_padded, pad_size=teacher_pad_size)
                teacher_log_probs = teacher_logprob_output.batch['old_log_probs']

            print(
                "[OPSD KL DEBUG] Processing Chunk\n"
                f"[Chunk Idx] {chunk_idx // opsd_logprob_chunk_size + 1}/{total_chunks}\n"
                f"[Chunk Step Instances] {len(chunk_records)}\n"
                + "-" * 80
            )

            if opsd_student_scoring_mode == 'causal_prefix':
                chunk_iter = zip(chunk_records, student_log_probs, teacher_log_probs)
            else:
                chunk_iter = ((record, None, teacher_lp) for record, teacher_lp in zip(chunk_records, teacher_log_probs))

            for record, student_lp, teacher_lp in chunk_iter:
                meta = record['meta']
                batch_idx = meta['batch_idx']
                step_start = meta['step_token_start_idx']
                step_end = meta['step_token_end_idx']
                resp_len = meta['resp_len']
                if step_end < step_start:
                    continue

                if opsd_student_scoring_mode == 'causal_prefix':
                    student_lp = student_lp[:resp_len]
                else:
                    student_lp = batch.batch['old_log_probs'][batch_idx, step_start:step_end + 1]
                teacher_lp = teacher_lp[:resp_len]
                shared_len = min(student_lp.numel(), teacher_lp.numel())
                if shared_len <= 0:
                    continue

                student_lp = student_lp[:shared_len]
                teacher_lp = teacher_lp[:shared_len]

                delta = (teacher_lp - student_lp).detach().float()
                opsd_token_delta[batch_idx, step_start:step_start + shared_len] = delta
                delta_cpu = delta.cpu()
                student_lp_cpu = student_lp.detach().float().cpu()
                teacher_lp_cpu = teacher_lp.detach().float().cpu()
                token_ids = batch.batch['responses'][batch_idx, step_start:step_start + shared_len].detach().cpu().tolist()
                delta_values.extend(delta_cpu.tolist())

                case_payload = {
                    'batch_idx': batch_idx,
                    'step_idx': meta['step_idx'],
                    'action_type': meta.get('action_type', 'unknown'),
                    'target_span_mode': meta.get('target_span_mode', 'clean_step_no_observation'),
                    'full_trajectory_text': meta.get('full_trajectory_text', ''),
                    'step_text': self.tokenizer.decode(token_ids, skip_special_tokens=False),
                    'action_text': meta.get('action_text', ''),
                    'target_text': meta.get('target_text', self.tokenizer.decode(token_ids, skip_special_tokens=False)),
                    'step_char_span': meta.get('step_char_span', (-1, -1)),
                    'action_char_span': meta.get('action_char_span', (-1, -1)),
                    'target_char_span': meta.get('target_char_span', (-1, -1)),
                    'token_ids': token_ids,
                    'delta': delta_cpu.tolist(),
                    'student_logprob': student_lp_cpu.tolist(),
                    'teacher_logprob': teacher_lp_cpu.tolist(),
                }
                seen_case_count += 1
                if len(sampled_cases) < opsd_report_case_count:
                    sampled_cases.append(case_payload)
                else:
                    replace_idx = np.random.randint(0, seen_case_count)
                    if replace_idx < len(sampled_cases):
                        sampled_cases[replace_idx] = case_payload

                if batch_idx == 0 and meta['step_idx'] < 3:
                    step_text = case_payload['step_text']
                    print(
                        "[OPSD KL DEBUG] Token-Level KL\n"
                        f"[Step Idx] {meta['step_idx']}\n"
                        f"[Teacher Mode] {opsd_teacher_mode}\n"
                        f"[Student Scoring Mode] {opsd_student_scoring_mode}\n"
                        f"[Step Text]\n{step_text}\n"
                        f"[Student LogProb] {student_lp.tolist()}\n"
                        f"[Teacher LogProb] {teacher_lp.tolist()}\n"
                        f"[Delta] {delta.tolist()}\n"
                        + "-" * 80
                    )

            del teacher_batch
            del teacher_batch_padded
            del teacher_logprob_output_padded
            del teacher_logprob_output
            del teacher_log_probs
            if student_batch is not None:
                del student_batch
            if student_batch_padded is not None:
                del student_batch_padded
            if student_logprob_output_padded is not None:
                del student_logprob_output_padded
            if student_logprob_output is not None:
                del student_logprob_output
            if student_log_probs is not None:
                del student_log_probs

        self._print_opsd_delta_report(
            delta_values=delta_values,
            sampled_cases=sampled_cases,
            delta_threshold=opsd_report_delta_threshold,
        )
        batch.batch['opsd_token_delta'] = opsd_token_delta
        return batch

    def _refresh_opsd_teacher_snapshot(self):
        if not self.use_reference_policy:
            return

        snapshot_local_path = os.path.join(self.config.trainer.default_local_dir, 'opsd_teacher_snapshot', 'latest')
        print(f"[OPSD SNAPSHOT] Refreshing stale teacher from actor checkpoint at step {self.global_steps}: {snapshot_local_path}")
        self.actor_rollout_wg.save_checkpoint(snapshot_local_path, None)
        self.ref_policy_wg.load_pretrained_model(snapshot_local_path)

    def _validate(self):
        """
        The training loop of PPO with global metric computation.
        Accumulates metrics across all batches before computing final statistics.
        """
        import torch
        reward_tensor_lst = []
        data_source_lst = []
        self._opsd_validate_report_sink = []
        if hasattr(self.val_reward_fn, 'reset_validation_examples'):
            self.val_reward_fn.reset_validation_examples()

        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,
        )

        # Agent config preparation
        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            is_validation = True,
        )

        if not self.config.do_search:
            for test_data in self.val_dataloader:
                test_batch = DataProto.from_single_dict(test_data)

                # we only do validation on rule-based rm
                if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                    return {}

                test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }

                # pad to be divisible by dp_size
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                # unpad
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
                print('validation generation end')

                test_batch = test_batch.union(test_output_gen_batch)
                test_batch.meta_info = dict(getattr(test_batch, 'meta_info', {}) or {})
                test_batch.meta_info['validate'] = True

                # evaluate using reward_function
                if hasattr(self.val_reward_fn, "set_rollout_wg"):
                    self.val_reward_fn.set_rollout_wg(self.actor_rollout_wg)
                else:
                    self.val_reward_fn.actor_rollout_wg = self.actor_rollout_wg
                    
                # for certain reward function (e.g. sandbox), the generation can overlap with reward
                reward_tensor = self.val_reward_fn(test_batch)
                if 'opsd_kl_data' in test_batch.non_tensor_batch:
                    test_batch.batch['token_level_scores'] = reward_tensor
                    self._apply_opsd_token_level_kl(test_batch)

                reward_tensor_lst.append(reward_tensor)
                data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
        else:
            for batch_dict in self.val_dataloader:
                timing_raw = {}
                test_batch: DataProto = DataProto.from_single_dict(batch_dict)
                # test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)
                
                test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }
                with _timer('step', timing_raw):
                    first_input_ids = test_gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
                    with _timer('gen', timing_raw):
                        generation_manager.timing_raw = timing_raw
                        final_gen_batch_output = generation_manager.run_llm_loop(
                            gen_batch=test_gen_batch,
                            initial_input_ids=first_input_ids,
                        )
                    
                    test_batch = test_batch.union(final_gen_batch_output)
                    test_batch.meta_info = dict(getattr(test_batch, 'meta_info', {}) or {})
                    test_batch.meta_info['validate'] = True
                    
                    for key in test_batch.batch.keys():
                        test_batch.batch[key] = test_batch.batch[key].long()
                    
                    # evaluate using reward_function
                    if hasattr(self.val_reward_fn, "set_rollout_wg"):
                        self.val_reward_fn.set_rollout_wg(self.actor_rollout_wg)
                    else:
                        self.val_reward_fn.actor_rollout_wg = self.actor_rollout_wg
                        
                    # for certain reward function (e.g. sandbox), the generation can overlap with reward
                    reward_tensor = self.val_reward_fn(test_batch)
                    if 'opsd_kl_data' in test_batch.non_tensor_batch:
                        test_batch.batch['token_level_scores'] = reward_tensor
                        self._apply_opsd_token_level_kl(test_batch)

                    reward_tensor_lst.append(reward_tensor)
                    data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        reward_tensor = torch.cat([rw.sum(-1) for rw in reward_tensor_lst], dim=0).cpu()  # (batch_size,)
        # reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        validation_examples = []
        if hasattr(self.val_reward_fn, 'pop_validation_examples'):
            validation_examples = self.val_reward_fn.pop_validation_examples()

        validation_detail_path = self.config.trainer.get('validation_detail_path', None)
        if validation_detail_path:
            detail_dir = os.path.dirname(validation_detail_path)
            if detail_dir:
                os.makedirs(detail_dir, exist_ok=True)
            with open(validation_detail_path, 'w', encoding='utf-8') as f:
                for row in validation_examples:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
            print(f'[VALIDATION DETAILS] Saved {len(validation_examples)} examples to {validation_detail_path}')

        if self._opsd_validate_report_sink:
            report_dir = os.path.join(self.config.trainer.default_local_dir, 'opsd_delta_reports')
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(
                report_dir,
                f'validation_step_{self.global_steps}_{uuid.uuid4().hex[:8]}.txt',
            )
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('OPSD delta validation report\n')
                f.write(f'global_step={self.global_steps}\n')
                f.write(f'val_only={bool(self.config.trainer.get("val_only", False))}\n')
                for metric_name, metric_value in sorted(metric_dict.items()):
                    f.write(f'{metric_name}={metric_value}\n')
                f.write('\n')
                f.write("\n\n".join(self._opsd_validate_report_sink))
            print(f'[OPSD DELTA REPORT] Validation report saved to {report_path}')

        self._opsd_validate_report_sink = None
        return metric_dict


    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
            
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

        trainer_state_path = self._get_trainer_state_path(actor_local_path)
        torch.save({
            'global_steps': self.global_steps,
            'kl_ctrl_value': float(self.kl_ctrl.value),
        }, trainer_state_path)
        print(f'[RESUME] Saved trainer state to {trainer_state_path}')
        if actor_remote_path is not None:
            hdfs_io.makedirs(actor_remote_path, exist_ok=True)
            hdfs_io.copy(src=actor_local_path, dst=actor_remote_path)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        logger = self.logger
        is_resuming = self.resume_state is not None
        if is_resuming:
            self.global_steps = self.resume_next_global_step
            print(f'[RESUME] Continuing training from global step {self.global_steps}.')
            if self.global_steps >= self.total_training_steps:
                print(f'[RESUME] global step {self.global_steps} already reached total_training_steps={self.total_training_steps}. Nothing to do.')
                return
        else:
            self.global_steps = 0

        if is_resuming and self.use_reference_policy:
            try:
                opsd_teacher_mode = str(self.config.algorithm.opsd_teacher_mode)
            except Exception:
                opsd_teacher_mode = 'stale_ref_policy'
            if opsd_teacher_mode == 'stale_ref_policy':
                print(f'[RESUME] Refreshing stale teacher snapshot at global step {self.global_steps}.')
                self._refresh_opsd_teacher_snapshot()
        # perform validation before training
        # currently, we only support validation using the reward_function.
        run_initial_validation = self.config.trainer.get('val_before_train', True)
        if is_resuming:
            run_initial_validation = self.config.trainer.get('resume_run_initial_validation', False)
        if self.val_reward_fn is not None and run_initial_validation:
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            init_val_log_step = self.global_steps if not is_resuming else self.global_steps - 1
            logger.log(data=val_metrics, step=init_val_log_step)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        if not is_resuming:
            self.global_steps += 1

        # Agent config preparation
        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,
        )

        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
        )

        # start training loop
        start_epoch = self.resume_epoch if is_resuming else 0
        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            for batch_idx, batch_dict in enumerate(self.train_dataloader):
                if is_resuming and epoch == start_epoch and batch_idx < self.resume_batch_offset:
                    continue
                print(f'epoch {epoch}, step {self.global_steps}')
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                ####################
                # original code here

                with _timer('step', timing_raw):
                    if not self.config.do_search:
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                                dtype=object)
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)

                ####################
                # Below is aLL about agents - the "LLM + forloop"
                ####################
                # with _timer('step', timing_raw):
                    else:
                        first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone().long()

                        with _timer('gen', timing_raw):
                            generation_manager.timing_raw = timing_raw
                            final_gen_batch_output = generation_manager.run_llm_loop(
                                gen_batch=gen_batch,
                                initial_input_ids=first_input_ids,
                            )

                        # final_gen_batch_output.batch.apply(lambda x: x.long(), inplace=True)
                        for key in final_gen_batch_output.batch.keys():
                            final_gen_batch_output.batch[key] = final_gen_batch_output.batch[key].long()

                        with torch.no_grad():
                            output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                            final_gen_batch_output = final_gen_batch_output.union(output)

                        # batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                        #                                         dtype=object)
                        batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()
                                            
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(final_gen_batch_output)

                    ####################
                    ####################

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # batch.batch.apply(lambda x, key: x.long() if key != "old_log_probs" else x, inplace=True, key=True)
                    for key in batch.batch.keys():
                        if key != 'old_log_probs':
                            batch.batch[key] = batch.batch[key].long()

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # Inject the actor_rollout_wg for OPSD
                        if hasattr(self.reward_fn, "set_rollout_wg"):
                            self.reward_fn.set_rollout_wg(self.actor_rollout_wg)
                        else:
                            # Fallback: force inject as attribute if method not found
                            self.reward_fn.actor_rollout_wg = self.actor_rollout_wg
                            
                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor
                        batch = self._apply_opsd_token_level_kl(batch)

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.use_kl_loss:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        try:
                            opsd_weight_clip = float(self.config.algorithm.opsd_weight_clip)
                        except Exception:
                            opsd_weight_clip = 0.2
                        try:
                            opsd_mix_lambda_init = float(self.config.algorithm.opsd_mix_lambda_init)
                        except Exception:
                            opsd_mix_lambda_init = 0.5
                        try:
                            opsd_mix_lambda_decay_steps = int(self.config.algorithm.opsd_mix_lambda_decay_steps)
                        except Exception:
                            opsd_mix_lambda_decay_steps = 50

                        if opsd_mix_lambda_decay_steps > 0:
                            decay_ratio = max(0.0, 1.0 - float(self.global_steps) / float(opsd_mix_lambda_decay_steps))
                            opsd_mix_lambda = opsd_mix_lambda_init * decay_ratio
                        else:
                            opsd_mix_lambda = opsd_mix_lambda_init

                        batch.meta_info['opsd_weight_clip'] = opsd_weight_clip
                        try:
                            opsd_weight_fn = str(self.config.algorithm.opsd_weight_fn)
                        except Exception:
                            opsd_weight_fn = 'sigmoid'
                        batch.meta_info['opsd_weight_fn'] = opsd_weight_fn
                        batch.meta_info['opsd_mix_lambda'] = opsd_mix_lambda

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            if self.config.do_search and self.config.actor_rollout_ref.actor.state_masking:
                                batch, metrics = self._create_loss_mask(batch, metrics)
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                        try:
                            opsd_teacher_mode = str(self.config.algorithm.opsd_teacher_mode)
                        except Exception:
                            opsd_teacher_mode = 'stale_ref_policy' if self.use_reference_policy else 'live_actor'
                        try:
                            opsd_teacher_refresh_interval = int(self.config.algorithm.opsd_teacher_refresh_interval)
                        except Exception:
                            opsd_teacher_refresh_interval = 10

                        if (opsd_teacher_mode == 'stale_ref_policy' and self.use_reference_policy and
                                opsd_teacher_refresh_interval > 0 and
                                self.global_steps % opsd_teacher_refresh_interval == 0):
                            with _timer('opsd_teacher_refresh', timing_raw):
                                self._refresh_opsd_teacher_snapshot()

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return
    
    def _create_loss_mask(self, batch, metrics):
        """Create loss mask for state tokens."""
        response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_length:]
        
        loss_mask = batch.batch['info_mask'][:, -response_length:]
        batch.batch['loss_mask'] = loss_mask

        metrics.update({
            'state_tokens/total': loss_mask.sum().item(),
            'state_tokens/coverage': (loss_mask.sum() / response_mask.sum()).item(),
        })
        
        return batch, metrics
