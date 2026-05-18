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
from typing import Optional
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
        self.config = None
        self._validation_examples = []
        
        # Anchor logical steps on complete action spans instead of relying on
        # perfectly closed observation blocks, which may be truncated.
        self.action_pattern = re.compile(r'<(search|answer)>(.*?)</\1>', re.DOTALL)
        self.information_start_tag = "<information>"
        self.information_end_tag = "</information>"
        self.invalid_action_prefix = "My previous action is invalid."
        self.invalid_action_suffix = "Let me try again.\n"

    def set_rollout_wg(self, wg):
        """Inject the rollout worker group to allow OPSD to compute hints and alpha scores."""
        self.actor_rollout_wg = wg

    def reset_validation_examples(self):
        self._validation_examples = []

    def pop_validation_examples(self):
        validation_examples = self._validation_examples
        self._validation_examples = []
        return validation_examples

    def _score_to_correctness_label(self, score) -> str:
        try:
            return "correct" if float(score) == 1.0 else "incorrect"
        except Exception:
            return "incorrect"

    def _normalize_group_key(self, value, fallback):
        if value is None:
            return fallback
        if hasattr(value, 'item'):
            try:
                return value.item()
            except Exception:
                pass
        return value

    def _is_config_flag_enabled(self, key: str, default: bool = False) -> bool:
        value = default
        try:
            value = self.config.algorithm.get(key, default)
        except Exception:
            value = default

        if isinstance(value, str):
            return value.strip().lower() in ('1', 'true', 'yes', 'on')
        return bool(value)

    def _get_hindsight_info_mode(self) -> str:
        value = 'none'
        try:
            value = self.config.algorithm.get('opsd_hindsight_info_mode', value)
        except Exception:
            pass

        normalized = str(value).strip().lower()
        alias_map = {
            'off': 'none',
            'disabled': 'none',
            'peer': 'peer_traj',
            'peer_correct_trajectory': 'peer_traj',
            'golden': 'golden_rollout',
            'gold': 'golden_rollout',
        }
        normalized = alias_map.get(normalized, normalized)
        if normalized in ('none', 'peer_traj', 'golden_rollout'):
            return normalized
        return 'none'

    def _normalize_optional_text(self, value) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized if normalized else None

    def _extract_golden_rollout(self, extra_info) -> Optional[str]:
        if isinstance(extra_info, dict):
            return self._normalize_optional_text(extra_info.get('golden_rollout'))

        try:
            return self._normalize_optional_text(extra_info.get('golden_rollout'))
        except Exception:
            return None

    def _normalize_ground_truth(self, ground_truth):
        candidate = ground_truth
        if isinstance(candidate, dict):
            candidate = candidate.get('target', candidate.get('golden_answers', candidate))
        elif hasattr(candidate, 'get'):
            try:
                candidate = candidate.get('target', candidate.get('golden_answers', candidate))
            except Exception:
                pass

        if isinstance(candidate, (list, tuple, set)):
            return [str(item).strip() for item in candidate if str(item).strip()]

        normalized = self._normalize_optional_text(candidate)
        return [normalized] if normalized else []

    def _extract_prompt_text(self, data_item) -> Optional[str]:
        raw_prompt = data_item.non_tensor_batch.get('raw_prompt', None)
        if raw_prompt is None:
            return None

        if hasattr(raw_prompt, 'tolist'):
            try:
                raw_prompt = raw_prompt.tolist()
            except Exception:
                pass

        if isinstance(raw_prompt, list):
            prompt_parts = []
            for message in raw_prompt:
                content = None
                if isinstance(message, dict):
                    content = message.get('content')
                elif hasattr(message, 'get'):
                    try:
                        content = message.get('content')
                    except Exception:
                        content = None
                if content is not None:
                    prompt_parts.append(str(content))
            if prompt_parts:
                return "\n".join(prompt_parts)

        return self._normalize_optional_text(raw_prompt)

    def _extract_question_from_prompt_text(self, prompt_text: Optional[str]) -> Optional[str]:
        if not prompt_text:
            return None
        if 'Question:' in prompt_text:
            return self._normalize_optional_text(prompt_text.split('Question:', 1)[1])
        return self._normalize_optional_text(prompt_text)

    def _extract_prediction(self, response_str: str) -> Optional[str]:
        answer_matches = re.findall(r'<answer>(.*?)</answer>', response_str, re.DOTALL)
        if answer_matches:
            return self._normalize_optional_text(answer_matches[-1])
        return self._normalize_optional_text(response_str)

    def _extract_search_queries(self, response_str: str):
        search_matches = re.findall(r'<search>(.*?)</search>', response_str, re.DOTALL)
        return [str(match).strip() for match in search_matches if str(match).strip()]

    def _build_teacher_hindsight_suffix_ids(
        self,
        correctness_label: str,
        peer_correct_trajectory: Optional[str] = None,
        golden_rollout: Optional[str] = None,
    ) -> list[int]:
        include_final_correctness = self._is_config_flag_enabled(
            'opsd_teacher_include_final_correctness',
            default=False,
        )
        hindsight_lines = ["", "<hindsight>"]
        has_hindsight_content = False

        if include_final_correctness:
            hindsight_lines.append(f"final_correctness: {correctness_label}")
            has_hindsight_content = True
        if peer_correct_trajectory:
            hindsight_lines.extend([
                "peer_correct_trajectory:",
                "<peer_correct_trajectory>",
                peer_correct_trajectory.rstrip(),
                "</peer_correct_trajectory>",
            ])
            has_hindsight_content = True
        if golden_rollout:
            hindsight_lines.extend([
                "golden_rollout:",
                "<golden_rollout>",
                golden_rollout.rstrip(),
                "</golden_rollout>",
            ])
            has_hindsight_content = True
        if not has_hindsight_content:
            return []
        hindsight_lines.extend([
            "</hindsight>",
            "",
        ])
        hindsight_block = "\n".join(hindsight_lines)
        return self.tokenizer.encode(hindsight_block, add_special_tokens=False)

    def _find_invalid_feedback_spans(self, response_str: str):
        spans = []
        cursor = 0
        response_len = len(response_str)

        while cursor < response_len:
            span_start = response_str.find(self.invalid_action_prefix, cursor)
            if span_start < 0:
                break

            suffix_start = response_str.find(self.invalid_action_suffix, span_start)
            if suffix_start >= 0:
                span_end = suffix_start + len(self.invalid_action_suffix)
            else:
                span_end = response_len

            spans.append((span_start, span_end))
            cursor = max(span_end, span_start + len(self.invalid_action_prefix))

        return spans

    def _char_idx_in_spans(self, char_idx: int, spans) -> bool:
        for span_start, span_end in spans:
            if span_start <= char_idx < span_end:
                return True
        return False

    def _advance_past_invalid_feedback(self, response_str: str, char_idx: int, invalid_feedback_spans):
        response_len = len(response_str)

        while char_idx < response_len:
            while char_idx < response_len and response_str[char_idx].isspace():
                char_idx += 1

            advanced = False
            for span_start, span_end in invalid_feedback_spans:
                if span_start <= char_idx < span_end:
                    char_idx = span_end
                    advanced = True
                    break

            if not advanced:
                break

        while char_idx < response_len and response_str[char_idx].isspace():
            char_idx += 1

        return char_idx

    def _find_next_observation_span(self, response_str: str, start_char_idx: int, end_char_idx: int, invalid_feedback_spans):
        candidates = []

        info_start_char_idx = response_str.find(
            self.information_start_tag,
            start_char_idx,
            end_char_idx,
        )
        if info_start_char_idx >= 0:
            info_end_char_idx = response_str.find(
                self.information_end_tag,
                info_start_char_idx,
            )
            if info_end_char_idx >= 0:
                info_span_end = info_end_char_idx + len(self.information_end_tag)
            else:
                info_span_end = end_char_idx
            candidates.append((info_start_char_idx, min(info_span_end, end_char_idx)))

        for span_start, span_end in invalid_feedback_spans:
            if start_char_idx <= span_start < end_char_idx:
                candidates.append((span_start, min(span_end, end_char_idx)))

        if not candidates:
            return None

        return min(candidates, key=lambda span: span[0])

    def _extract_response_steps(self, response_str: str):
        """Extract logical steps anchored on complete action spans.

        Each step starts right after the previous observation block (if any) and
        ends at the next observation start or the next action / end of response.
        This keeps reasoning + action tokens in the clean span while excluding
        observation text even when `</information>` is missing due to truncation.
        """
        steps = []
        invalid_feedback_spans = self._find_invalid_feedback_spans(response_str)
        action_matches = [
            action_match
            for action_match in self.action_pattern.finditer(response_str)
            if not self._char_idx_in_spans(action_match.start(), invalid_feedback_spans)
        ]
        if not action_matches:
            return steps
        final_answer_action_idx = None
        for reverse_idx in range(len(action_matches) - 1, -1, -1):
            if action_matches[reverse_idx].group(1) == 'answer':
                final_answer_action_idx = reverse_idx
                break

        response_len = len(response_str)
        step_start_char_idx = self._advance_past_invalid_feedback(
            response_str,
            0,
            invalid_feedback_spans,
        )

        for action_idx, action_match in enumerate(action_matches):
            step_start_char_idx = self._advance_past_invalid_feedback(
                response_str,
                step_start_char_idx,
                invalid_feedback_spans,
            )
            action_start_char_idx = action_match.start()
            action_end_char_idx = action_match.end()
            next_action_start_char_idx = (
                action_matches[action_idx + 1].start()
                if action_idx + 1 < len(action_matches)
                else response_len
            )

            observation_span = self._find_next_observation_span(
                response_str,
                action_end_char_idx,
                next_action_start_char_idx,
                invalid_feedback_spans,
            )

            if observation_span is not None:
                target_end_char_idx = observation_span[0]
                step_end_char_idx = observation_span[1]
            else:
                target_end_char_idx = next_action_start_char_idx
                step_end_char_idx = next_action_start_char_idx

            if target_end_char_idx <= step_start_char_idx:
                step_start_char_idx = self._advance_past_invalid_feedback(
                    response_str,
                    step_end_char_idx,
                    invalid_feedback_spans,
                )
                continue

            step_text = response_str[step_start_char_idx:step_end_char_idx]
            clean_step_text = response_str[step_start_char_idx:target_end_char_idx]
            if clean_step_text.strip():
                steps.append({
                    'step_start_char_idx': step_start_char_idx,
                    'step_end_char_idx': step_end_char_idx,
                    'target_end_char_idx': target_end_char_idx,
                    'step_text': step_text,
                    'clean_step_text': clean_step_text,
                    'action_start_char_idx': action_start_char_idx,
                    'action_end_char_idx': action_end_char_idx,
                    'action_type': action_match.group(1),
                    'action_str': action_match.group(0),
                    'is_final_answer_action': (
                        action_match.group(1) == 'answer' and action_idx == final_answer_action_idx
                    ),
                })

            step_start_char_idx = self._advance_past_invalid_feedback(
                response_str,
                step_end_char_idx,
                invalid_feedback_spans,
            )

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
        item_cache = []
        opsd_metadata = []
        is_validation = bool(getattr(data, 'meta_info', {}).get('validate', False))

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
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            # Macro Reward R
            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score)
            all_scores.append(score)
            correctness_label = self._score_to_correctness_label(score)
            group_key = self._normalize_group_key(data_item.non_tensor_batch.get('index', None), i)
            extra_info = data_item.non_tensor_batch.get('extra_info', {})
            item_cache.append({
                'group_key': group_key,
                'valid_prompt_ids': valid_prompt_ids,
                'valid_response_ids': valid_response_ids,
                'valid_response_length': valid_response_length.item(),
                'response_str': response_str,
                'score': score,
                'correctness_label': correctness_label,
                'golden_rollout': self._extract_golden_rollout(extra_info),
            })

            if is_validation:
                prompt_text = self._extract_prompt_text(data_item)
                search_queries = self._extract_search_queries(response_str)
                self._validation_examples.append({
                    'index': group_key,
                    'data_source': data_source,
                    'prompt_text': prompt_text,
                    'question': self._extract_question_from_prompt_text(prompt_text),
                    'prediction': self._extract_prediction(response_str),
                    'ground_truth': self._normalize_ground_truth(ground_truth),
                    'score': float(score),
                    'is_correct': bool(float(score) == 1.0),
                    'search_count': len(search_queries),
                    'search_queries': search_queries,
                    'valid_response_length': int(valid_response_length.item()),
                    'trajectory_text': response_str,
                })

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        hindsight_info_mode = self._get_hindsight_info_mode()
        mixed_group_first_correct_trajectory = {}
        if (
            hindsight_info_mode == 'peer_traj'
            and self._is_config_flag_enabled('opsd_hindsight_include_first_correct_peer_trajectory', default=False)
        ):
            grouped_items = defaultdict(list)
            for item in item_cache:
                grouped_items[item['group_key']].append(item)

            for group_key, grouped_item_list in grouped_items.items():
                correct_items = [item for item in grouped_item_list if item['correctness_label'] == 'correct']
                if 0 < len(correct_items) < len(grouped_item_list):
                    mixed_group_first_correct_trajectory[group_key] = correct_items[0]['response_str']

        # OPSD Step Extraction
        if self.actor_rollout_wg is not None:
            for i, cached_item in enumerate(item_cache):
                valid_prompt_ids = cached_item['valid_prompt_ids']
                valid_response_ids = cached_item['valid_response_ids']
                valid_response_length = cached_item['valid_response_length']
                response_str = cached_item['response_str']
                correctness_label = cached_item['correctness_label']
                group_key = cached_item['group_key']
                sample_golden_rollout = cached_item.get('golden_rollout')

                step_segments = self._extract_response_steps(response_str)
                for step_segment in step_segments:
                    step_start_char_idx = step_segment['step_start_char_idx']
                    end_char_idx = step_segment['step_end_char_idx']
                    clean_step_end_char_idx = step_segment['target_end_char_idx']
                    step_text = step_segment['step_text']
                    clean_step_text = step_segment['clean_step_text']
                    action_type = step_segment['action_type']
                    action_str = step_segment['action_str']

                    try:
                        opsd_target_span_mode = str(self.config.algorithm.opsd_target_span_mode)
                    except Exception:
                        opsd_target_span_mode = 'clean_step_no_observation'
                    if opsd_target_span_mode not in (
                        'clean_step_no_observation',
                        'action_only',
                        'answer_only',
                        'final_answer_only',
                    ):
                        raise ValueError(
                            f"Unsupported algorithm.opsd_target_span_mode={opsd_target_span_mode}. "
                            "Expected 'clean_step_no_observation', 'action_only', 'answer_only', "
                            "or 'final_answer_only'."
                        )
                    if opsd_target_span_mode in ('answer_only', 'final_answer_only') and action_type != 'answer':
                        continue
                    if opsd_target_span_mode == 'final_answer_only' and not step_segment.get('is_final_answer_action', False):
                        continue

                    # Approximate token indices by tokenizing the corresponding prefixes.
                    step_start_tokens = self.tokenizer.encode(response_str[:step_start_char_idx], add_special_tokens=False)
                    clean_step_end_tokens = self.tokenizer.encode(
                        response_str[:clean_step_end_char_idx], add_special_tokens=False
                    )
                    token_start_idx = len(step_start_tokens)
                    token_start_idx = max(0, min(token_start_idx, valid_response_length))
                    clean_step_token_end_idx = len(clean_step_end_tokens) - 1
                    clean_step_token_end_idx = max(0, min(clean_step_token_end_idx, valid_response_length - 1))

                    action_start_char_idx = step_segment['action_start_char_idx']
                    action_end_char_idx = step_segment['action_end_char_idx']
                    action_start_tokens = self.tokenizer.encode(
                        response_str[:action_start_char_idx], add_special_tokens=False
                    )
                    action_end_tokens = self.tokenizer.encode(
                        response_str[:action_end_char_idx], add_special_tokens=False
                    )
                    action_token_start_idx = len(action_start_tokens)
                    action_token_end_idx = len(action_end_tokens) - 1
                    action_token_start_idx = max(0, min(action_token_start_idx, valid_response_length - 1))
                    action_token_end_idx = max(0, min(action_token_end_idx, valid_response_length - 1))
                    if action_token_end_idx < action_token_start_idx:
                        continue

                    if opsd_target_span_mode in ('action_only', 'answer_only', 'final_answer_only'):
                        target_token_start_idx = action_token_start_idx
                        target_token_end_idx = action_token_end_idx
                        target_char_start_idx = action_start_char_idx
                        target_char_end_idx = action_end_char_idx
                        target_text = action_str
                    else:
                        if clean_step_token_end_idx < token_start_idx:
                            continue
                        target_token_start_idx = token_start_idx
                        target_token_end_idx = clean_step_token_end_idx
                        target_char_start_idx = step_start_char_idx
                        target_char_end_idx = clean_step_end_char_idx
                        target_text = clean_step_text
                    token_end_idx = target_token_end_idx

                    peer_correct_trajectory = None
                    if hindsight_info_mode == 'peer_traj' and correctness_label != 'correct':
                        peer_correct_trajectory = mixed_group_first_correct_trajectory.get(group_key)
                    golden_rollout = sample_golden_rollout if hindsight_info_mode == 'golden_rollout' else None

                    prompt_context_ids = valid_prompt_ids.tolist()
                    student_context_ids = prompt_context_ids + valid_response_ids[:target_token_start_idx].tolist()
                    target_response_ids = valid_response_ids[target_token_start_idx:target_token_end_idx + 1].tolist()
                    teacher_context_ids = student_context_ids + self._build_teacher_hindsight_suffix_ids(
                        correctness_label=correctness_label,
                        peer_correct_trajectory=peer_correct_trajectory,
                        golden_rollout=golden_rollout,
                    )

                    if len(target_response_ids) == 0:
                        continue

                    opsd_metadata.append({
                        'batch_idx': i,
                        'token_start_idx': target_token_start_idx,
                        'token_end_idx': token_end_idx,
                        'clean_step_token_end_idx': clean_step_token_end_idx,
                        'target_span_mode': opsd_target_span_mode,
                        'target_token_start_idx': target_token_start_idx,
                        'target_token_end_idx': target_token_end_idx,
                        'target_char_start_idx': target_char_start_idx,
                        'target_char_end_idx': target_char_end_idx,
                        'action_token_start_idx': action_token_start_idx,
                        'action_token_end_idx': action_token_end_idx,
                        'step_start_char_idx': step_start_char_idx,
                        'step_end_char_idx': end_char_idx,
                        'action_start_char_idx': action_start_char_idx,
                        'action_end_char_idx': action_end_char_idx,
                        'action_type': action_type,
                        'correctness_label': correctness_label,
                        'action_str': action_str,
                        'full_trajectory_text': response_str,
                        'step_text': step_text,
                        'clean_step_text': clean_step_text,
                        'target_text': target_text,
                        'student_context_ids': student_context_ids,
                        'teacher_context_ids': teacher_context_ids,
                        'target_response_ids': target_response_ids,
                    })

                    if i == 0:  # Debug print for the first trajectory only
                        print(f"[OPSD DEBUG] Extracted Step in Traj 0 | Action: {action_type} | Token End Idx: {token_end_idx}")
                        if len(opsd_metadata) == 1:
                            student_context_str = self.tokenizer.decode(student_context_ids, skip_special_tokens=False)
                            teacher_context_str = self.tokenizer.decode(teacher_context_ids, skip_special_tokens=False)
                            print(f"[OPSD DEBUG] Exact Student Context for Step 1:\n{student_context_str}\n" + "-" * 50)
                            print(f"[OPSD DEBUG] Exact Teacher Context for Step 1:\n{teacher_context_str}\n" + "-" * 50)

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
                    'target_span_mode': meta['target_span_mode'],
                    'target_token_start_idx': meta['target_token_start_idx'],
                    'target_token_end_idx': meta['target_token_end_idx'],
                    'target_char_start_idx': meta['target_char_start_idx'],
                    'target_char_end_idx': meta['target_char_end_idx'],
                    'action_token_start_idx': meta['action_token_start_idx'],
                    'action_token_end_idx': meta['action_token_end_idx'],
                    'step_start_char_idx': meta['step_start_char_idx'],
                    'step_end_char_idx': meta['step_end_char_idx'],
                    'action_start_char_idx': meta['action_start_char_idx'],
                    'action_end_char_idx': meta['action_end_char_idx'],
                    'correctness_label': meta['correctness_label'],
                    'action_str': meta['action_str'],
                    'action_type': meta['action_type'],
                    'full_trajectory_text': meta['full_trajectory_text'],
                    'step_text': meta['step_text'],
                    'clean_step_text': meta['clean_step_text'],
                    'target_text': meta['target_text'],
                    'student_context_ids': meta['student_context_ids'],
                    'teacher_context_ids': meta['teacher_context_ids'],
                    'target_response_ids': meta['target_response_ids'],
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
    reward_fn.config = config

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)
    val_reward_fn.config = config

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
