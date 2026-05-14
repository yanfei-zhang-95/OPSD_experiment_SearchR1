#!/usr/bin/env python3
"""
Generate Search-R1 reference trajectories with an external strong model.

The script reads a JSONL file such as `data/r1searcher_stage1/stage_2.jsonl`,
rolls out a Search-R1-style agent step by step with the same action space used
in training, and writes successful trajectories back into `gen_text_store`.

The environment contract intentionally matches the current Search-R1 setup:
- model outputs one step at a time
- each step must contain `<think>...</think>` followed by either
  `<search>...</search>` or `<answer>...</answer>`
- search observations are appended as `<information>...</information>`

The script is designed for OpenAI-compatible chat APIs. DeepSeek-style hosted
endpoints usually work by setting:

  OPENAI_API_KEY=...
  OPENAI_BASE_URL=...
  OPENAI_MODEL=...

Example:
  python etc/external_reference_rollout.py \
    --input-jsonl data/r1searcher_stage1/stage_2.jsonl \
    --output-jsonl data/r1searcher_stage1/stage_2_with_reference.jsonl \
    --search-url http://127.0.0.1:8085/retrieve
"""

import argparse
import asyncio
import collections
import json
import os
import re
import string
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp


SEARCH_R1_PREFIX = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> "
    "and it will return the top searched results between <information> and </information>. "
    "You can search as many times as your want. "
    "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, "
    "without detailed illustrations. For example, <answer> Beijing </answer>."
)

SYSTEM_PROMPT = (
    "You are a high-accuracy Search-R1 trajectory generator. "
    "You are interacting with a QA search environment. "
    "At each turn, output exactly one complete next step and nothing else. "
    "The step must be either:\n"
    "<think> ... </think>\n<search> concise query </search>\n"
    "or:\n"
    "<think> ... </think>\n<answer> short final answer </answer>\n"
    "Strict formatting requirement: every response must contain exactly one <think>...</think> block "
    "followed immediately by exactly one action block. "
    "Never omit <think>. Never output a bare <search>...</search> or bare <answer>...</answer>. "
    "Any extra text before, after, or outside the two tags is invalid. "
    "Never emit <information>; the environment appends it after a search. "
    "Keep search queries short and specific. "
    "When you answer, provide only the final answer span inside <answer>."
)

JUDGE_SYSTEM_PROMPT = (
    "You are a strict QA answer judge. "
    "Decide whether the predicted answer should be accepted as correct for the question, "
    "given one or more reference gold answers. "
    "Accept only semantic equivalence or clearly valid aliases. "
    "Reject partial, broader, narrower, related-but-not-equal, or unsupported answers. "
    "Return JSON only with keys: correct, reason."
)

JUDGE_FALLBACK_SYSTEM_PROMPT = (
    "You are a strict QA answer judge. "
    "Reply in exactly two lines. "
    "Line 1 must be YES or NO. "
    "Line 2 must start with REASON: and give a short reason."
)

THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ACTION_PATTERN = re.compile(r"<(search|answer)>(.*?)</\1>", re.DOTALL)


def load_dotenv(dotenv_path: Path, override: bool = False) -> None:
    if not dotenv_path.exists():
        return

    with dotenv_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue

            if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]

            if override or key not in os.environ:
                os.environ[key] = value


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    def lower(value: str) -> str:
        return value.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def answers_match(prediction: str, gold_answers: List[str], match_mode: str) -> bool:
    normalized_prediction = normalize_answer(prediction)
    for gold in gold_answers:
        normalized_gold = normalize_answer(gold)
        if match_mode in {"substring", "cover_exact"}:
            if normalized_gold and normalized_prediction and (
                normalized_gold in normalized_prediction
                or normalized_prediction in normalized_gold
            ):
                return True
        elif normalized_gold == normalized_prediction:
            return True
    return False


def token_f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_answer_f1(prediction: str, gold_answers: List[str]) -> float:
    best = 0.0
    for gold in gold_answers:
        best = max(best, token_f1_score(prediction, gold))
    return best


def build_judge_messages(
    question: str,
    prediction: str,
    gold_answers: List[str],
    fallback_mode: bool = False,
) -> List[Dict[str, str]]:
    if fallback_mode:
        user_prompt = (
            f"Question: {question}\n"
            f"Predicted answer: {prediction}\n"
            f"Gold answers: {json.dumps(gold_answers, ensure_ascii=False)}\n\n"
            "Should the predicted answer be accepted as correct?\n"
            "Accept only semantic equivalence or a clear alias.\n"
            "Reject partial answers, broader/narrower answers, related facts, and explanations.\n"
            "Output exactly:\n"
            "YES or NO\n"
            "REASON: <short reason>"
        )
        return [
            {"role": "system", "content": JUDGE_FALLBACK_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    user_prompt = (
        f"Question: {question}\n"
        f"Predicted answer: {prediction}\n"
        f"Gold answers: {json.dumps(gold_answers, ensure_ascii=False)}\n\n"
        "Judge whether the predicted answer should be accepted as correct.\n"
        "Be strict:\n"
        "- accept exact answers, clear aliases, abbreviations, or formatting variants\n"
        "- reject partially correct answers\n"
        "- reject answers that are only related to the gold answer\n"
        "- reject answers that require unsupported assumptions\n\n"
        'Respond with JSON only, for example: {"correct": true, "reason": "alias"}'
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def parse_judge_response(raw_text: str) -> Tuple[Optional[bool], str]:
    raw_text = raw_text.strip()
    if not raw_text:
        return None, "empty judge response"

    candidates = [raw_text]
    fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_text)
    if fenced_match:
        candidates.insert(0, fenced_match.group(1).strip())

    json_match = re.search(r"\{[\s\S]*\}", raw_text)
    if json_match:
        candidates.insert(0, json_match.group(0).strip())

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        if isinstance(data, dict):
            correct = data.get("correct")
            reason = str(data.get("reason", "")).strip()
            if isinstance(correct, bool):
                return correct, reason
            if isinstance(correct, str):
                lowered = correct.strip().lower()
                if lowered in {"true", "yes", "correct"}:
                    return True, reason
                if lowered in {"false", "no", "incorrect"}:
                    return False, reason

    lowered = raw_text.lower()
    if lowered in {"true", "yes", "correct"}:
        return True, raw_text[:300]
    if lowered in {"false", "no", "incorrect"}:
        return False, raw_text[:300]
    if re.search(r'"correct"\s*:\s*true', lowered) or re.search(r"\bcorrect\b\s*[:=]\s*true", lowered):
        return True, raw_text[:300]
    if re.search(r'"correct"\s*:\s*false', lowered) or re.search(r"\bcorrect\b\s*[:=]\s*false", lowered):
        return False, raw_text[:300]
    if re.match(r"^\s*yes\b", lowered):
        reason_match = re.search(r"reason\s*:\s*(.+)", raw_text, re.IGNORECASE | re.DOTALL)
        return True, reason_match.group(1).strip()[:300] if reason_match else raw_text[:300]
    if re.match(r"^\s*no\b", lowered):
        reason_match = re.search(r"reason\s*:\s*(.+)", raw_text, re.IGNORECASE | re.DOTALL)
        return False, reason_match.group(1).strip()[:300] if reason_match else raw_text[:300]
    return None, raw_text[:300]


def ensure_question_mark(question: str) -> str:
    question = question.strip()
    if question and question[-1] != "?":
        question += "?"
    return question


def atomic_write_jsonl(records: List[Dict[str, Any]], output_path: Path) -> None:
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(output_path)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def resolve_output_path(_input_path: Path, output_arg: str) -> Path:
    output_path = Path(output_arg)
    default_name = "stage2_rollout.jsonl"

    # If the user passes an existing directory, write a sibling file inside it.
    if output_path.exists() and output_path.is_dir():
        return output_path / default_name

    # If the path syntactically looks like a directory, treat it as one even if it
    # does not exist yet.
    if output_arg.endswith(os.sep) or (not output_path.suffix and not output_path.exists()):
        return output_path / default_name

    return output_path


def coerce_gold_answers(record: Dict[str, Any]) -> List[str]:
    answer = record.get("answer")
    if answer is None:
        raise KeyError("Missing `answer` field in record.")
    if isinstance(answer, list):
        return [str(item).strip() for item in answer if str(item).strip()]
    return [str(answer).strip()]


def extract_next_step(raw_text: str) -> Optional[Tuple[str, str, str]]:
    think_match = THINK_PATTERN.search(raw_text)
    if not think_match:
        return None

    action_match = ACTION_PATTERN.search(raw_text, pos=think_match.end())
    if not action_match:
        return None

    think_text = think_match.group(1).strip()
    action_type = action_match.group(1).strip()
    action_content = action_match.group(2).strip()
    if not think_text or not action_content:
        return None

    step_text = (
        f"<think>{think_text}</think>\n"
        f"<{action_type}>{action_content}</{action_type}>"
    )
    return step_text, action_type, action_content


def stringify_docs(search_result: List[Dict[str, Any]], max_doc_chars: int) -> str:
    chunks: List[str] = []
    for idx, doc_item in enumerate(search_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:]).strip()
        if max_doc_chars > 0:
            text = text[:max_doc_chars]
        chunks.append(f"Doc {idx + 1}(Title: {title}) {text}".strip())
    return "\n".join(chunks)


@dataclass
class RolloutConfig:
    search_url: str
    topk: int
    max_turns: int
    max_step_retries: int
    max_rollout_retries: int
    max_repair_retries: int
    sleep_seconds: float
    request_timeout: int
    search_timeout: int
    max_doc_chars: int
    max_observation_chars: int
    match_mode: str
    enable_f1_match: bool
    f1_threshold: float
    allow_gold_answer_repair: bool
    enable_llm_judge: bool
    judge_model: str
    judge_base_url: str
    judge_api_key: str
    judge_timeout: int
    judge_max_tokens: int
    judge_max_retries: int


async def _search_one(
    session: aiohttp.ClientSession,
    search_url: str,
    query: str,
    topk: int,
    timeout: int,
    max_doc_chars: int,
    max_observation_chars: int,
) -> str:
    payload = {
        "queries": [query],
        "topk": topk,
        "return_scores": True,
    }
    aiohttp_timeout = aiohttp.ClientTimeout(total=timeout)
    async with session.post(
        search_url, json=payload, timeout=aiohttp_timeout
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
    result = data["result"][0]
    obs = stringify_docs(result, max_doc_chars=max_doc_chars)
    if max_observation_chars > 0:
        obs = obs[:max_observation_chars]
    return obs


def build_messages(
    question: str,
    trajectory: str,
    step_feedback: Optional[str],
    attempt_index: int,
    gold_answers: List[str],
    gold_guided: bool,
) -> List[Dict[str, str]]:
    prompt_parts = [
        SEARCH_R1_PREFIX,
        f"Question: {question}",
        "",
        "Current trajectory so far:",
        trajectory.strip() if trajectory.strip() else "(empty)",
        "",
        "Output the next step only.",
        "You must output exactly one complete step in one of these two forms:",
        "<think>reasoning</think>",
        "<search>concise query</search>",
        "or",
        "<think>reasoning</think>",
        "<answer>final answer span</answer>",
        "The <think> block is mandatory on every turn.",
        "A bare <search>...</search> or bare <answer>...</answer> is invalid.",
        "Do not output any text outside the two required tags.",
        "Do not repeat the whole trajectory.",
        "Do not output markdown fences.",
        "Do not output <information> tags yourself.",
    ]

    if attempt_index > 0:
        prompt_parts.extend(
            [
                "",
                f"Previous full-trajectory attempts failed. Current attempt index: {attempt_index}.",
                "Be more careful, search more directly, and only answer when you have enough evidence.",
            ]
        )

    if step_feedback:
        prompt_parts.extend(["", "Validation feedback for the previous output:", step_feedback])

    if gold_guided:
        prompt_parts.extend(
            [
                "",
                "Verified final answer constraint:",
                f"You must eventually finish with exactly this answer span: {gold_answers[0]}",
                "Use search to gather supporting evidence before you finalize.",
            ]
        )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(prompt_parts)},
    ]


@dataclass
class SampleState:
    record_index: int
    record_idx: int
    question: str
    gold_answers: List[str]
    trajectory: str = ""
    active: bool = True
    done: bool = False
    success: bool = False
    final_answer: Optional[str] = None
    attempts: int = 0
    used_gold_guidance: bool = False
    judge_used: bool = False
    judge_reason: str = ""
    match_type: str = ""
    failure_reasons: List[str] = field(default_factory=list)
    step_feedback: Optional[str] = None
    attempt_index: int = 0
    gold_guided: bool = False
    step_retries: int = 0


async def _chat_one(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: int,
    messages: List[Dict[str, str]],
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    aiohttp_timeout = aiohttp.ClientTimeout(total=timeout)
    async with session.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=aiohttp_timeout,
    ) as response:
        response.raise_for_status()
        data = await response.json()
    message = data["choices"][0]["message"]["content"]
    if isinstance(message, list):
        text_parts: List[str] = []
        for item in message:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "".join(text_parts)
    return str(message)


async def _judge_answer(
    session: aiohttp.ClientSession,
    question: str,
    prediction: str,
    gold_answers: List[str],
    rollout_config: RolloutConfig,
) -> Tuple[bool, str]:
    debug_reasons: List[str] = []
    total_retries = max(rollout_config.judge_max_retries, 1)

    for judge_try in range(total_retries):
        fallback_mode = judge_try == total_retries - 1 and total_retries > 1
        messages = build_judge_messages(
            question=question,
            prediction=prediction,
            gold_answers=gold_answers,
            fallback_mode=fallback_mode,
        )
        try:
            raw_response = await _chat_one(
                session=session,
                base_url=rollout_config.judge_base_url,
                api_key=rollout_config.judge_api_key,
                model=rollout_config.judge_model,
                temperature=0.0,
                top_p=1.0,
                max_tokens=rollout_config.judge_max_tokens,
                timeout=rollout_config.judge_timeout,
                messages=messages,
            )
        except Exception as exc:
            debug_reasons.append(f"judge_request_failed[{judge_try + 1}/{total_retries}]: {exc}")
            continue

        decision, reason = parse_judge_response(raw_response)
        if decision is None:
            short_reason = reason or "unknown_parse_failure"
            debug_reasons.append(f"judge_parse_failed[{judge_try + 1}/{total_retries}]: {short_reason}")
            continue
        if decision:
            return True, reason or "accepted by llm judge"
        return False, reason or "rejected by llm judge"

    return False, " | ".join(debug_reasons[:3]) if debug_reasons else "judge_failed_without_response"


async def _process_one_sample(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    chat_timeout: int,
    search_url: str,
    topk: int,
    search_timeout: int,
    max_doc_chars: int,
    max_observation_chars: int,
    rollout_config: RolloutConfig,
    sample: SampleState,
) -> None:

    async def _chat(messages: List[Dict[str, str]]) -> str:
        return await _chat_one(
            session, base_url, api_key, model,
            temperature, top_p, max_tokens, chat_timeout, messages,
        )

    async def _search(query: str) -> str:
        return await _search_one(
            session, search_url, query, topk, search_timeout,
            max_doc_chars, max_observation_chars,
        )

    async def _run_attempt(attempt_index: int, gold_guided: bool) -> Tuple[bool, str, Optional[str], str]:
        trajectory = ""
        step_feedback: Optional[str] = None
        last_reason = "rollout did not produce a final answer"

        for _ in range(rollout_config.max_turns):
            step = None
            raw_response = ""
            for _ in range(rollout_config.max_step_retries):
                messages = build_messages(
                    question=sample.question,
                    trajectory=trajectory,
                    step_feedback=step_feedback,
                    attempt_index=attempt_index,
                    gold_answers=sample.gold_answers,
                    gold_guided=gold_guided,
                )
                raw_response = (await _chat(messages)).strip()
                step = extract_next_step(raw_response)
                if step is not None:
                    break
                step_feedback = (
                    "The output was invalid. Emit exactly one complete tagged step and nothing else. "
                    "You must include <think>...</think> first, then exactly one action tag. "
                    "Valid forms are: <think>...</think> followed by <search>...</search>, "
                    "or <think>...</think> followed by <answer>...</answer>. "
                    "Do not output a bare <search> or bare <answer>."
                )
                if rollout_config.sleep_seconds > 0:
                    await asyncio.sleep(rollout_config.sleep_seconds)

            if step is None:
                return False, trajectory, None, f"invalid step format: {raw_response[:500]}"

            step_text, action_type, action_content = step
            if action_type == "search":
                try:
                    observation = await _search(action_content)
                except Exception as exc:
                    return False, trajectory, None, f"search failed: {exc}"
                trajectory = (
                    trajectory
                    + step_text
                    + "\n\n<information>"
                    + observation.strip()
                    + "</information>\n\n"
                )
                step_feedback = None
            else:
                trajectory = trajectory + step_text
                if answers_match(action_content, sample.gold_answers, rollout_config.match_mode):
                    sample.match_type = rollout_config.match_mode
                    sample.judge_used = False
                    sample.judge_reason = ""
                    return True, trajectory, action_content, "success"
                if rollout_config.enable_f1_match:
                    answer_f1 = best_answer_f1(action_content, sample.gold_answers)
                    if answer_f1 >= rollout_config.f1_threshold:
                        sample.match_type = "token_f1"
                        sample.judge_used = False
                        sample.judge_reason = f"f1={answer_f1:.3f} threshold={rollout_config.f1_threshold:.3f}"
                        return True, trajectory, action_content, f"accepted by token f1: {sample.judge_reason}"
                if rollout_config.enable_llm_judge:
                    judge_ok, judge_reason = await _judge_answer(
                        session=session,
                        question=sample.question,
                        prediction=action_content,
                        gold_answers=sample.gold_answers,
                        rollout_config=rollout_config,
                    )
                    sample.judge_used = True
                    sample.judge_reason = judge_reason
                    if judge_ok:
                        sample.match_type = "llm_judge"
                        return True, trajectory, action_content, f"accepted by llm judge: {judge_reason}"
                last_reason = (
                    f"wrong final answer: predicted={action_content!r}, gold={sample.gold_answers!r}"
                )
                if sample.judge_used and sample.judge_reason:
                    last_reason = f"{last_reason}; judge={sample.judge_reason}"
                return False, trajectory, action_content, last_reason

            if rollout_config.sleep_seconds > 0:
                await asyncio.sleep(rollout_config.sleep_seconds)

        return False, trajectory, None, last_reason

    attempt_logs: List[str] = []
    total_attempts = 0

    for attempt_index in range(rollout_config.max_rollout_retries):
        total_attempts += 1
        success, trajectory, answer, reason = await _run_attempt(
            attempt_index=attempt_index, gold_guided=False
        )
        attempt_logs.append(reason)
        if success:
            sample.success = True
            sample.trajectory = trajectory
            sample.final_answer = answer
            sample.attempts = total_attempts
            sample.used_gold_guidance = False
            sample.failure_reasons = attempt_logs
            return

    if rollout_config.allow_gold_answer_repair:
        for repair_index in range(rollout_config.max_repair_retries):
            total_attempts += 1
            success, trajectory, answer, reason = await _run_attempt(
                attempt_index=repair_index, gold_guided=True
            )
            attempt_logs.append(reason)
            if success:
                sample.success = True
                sample.trajectory = trajectory
                sample.final_answer = answer
                sample.attempts = total_attempts
                sample.used_gold_guidance = True
                sample.failure_reasons = attempt_logs
                return

    sample.success = False
    sample.trajectory = ""
    sample.final_answer = None
    sample.attempts = total_attempts
    sample.used_gold_guidance = False
    sample.failure_reasons = attempt_logs


def heartbeat_loop(
    start_time: float,
    total_count: int,
    progress: Dict[str, int],
    progress_lock: threading.Lock,
    stop_event: threading.Event,
    interval_seconds: float,
) -> None:
    while not stop_event.wait(interval_seconds):
        with progress_lock:
            completed = progress["completed"]
            success = progress["success"]
            failed = progress["failed"]
        elapsed = int(time.time() - start_time)
        running = max(total_count - completed, 0)
        rate = completed / max(elapsed, 1) * 60
        print(
            f"[HEARTBEAT] elapsed={elapsed}s completed={completed}/{total_count} "
            f"success={success} failed={failed} running={running} rate={rate:.1f}/min"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=str, default="/data/yanfeizhang/OPSD_experiment/Search-R1/data/r1searcher_stage1/stage_2.jsonl")
    parser.add_argument("--output-jsonl", type=str, default="/data/yanfeizhang/OPSD_experiment/Search-R1/data/r1searcher_stage1/")
    parser.add_argument("--env-file", type=str, default=str(Path(__file__).with_name(".env")))
    parser.add_argument("--search-url", type=str, default="http://127.0.0.1:8085/retrieve")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--search-timeout", type=int, default=15)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-step-retries", type=int, default=3)
    parser.add_argument("--max-rollout-retries", type=int, default=4)
    parser.add_argument("--max-repair-retries", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=float, default=1)
    parser.add_argument("--max-doc-chars", type=int, default=1200)
    parser.add_argument("--max-observation-chars", type=int, default=3600)
    parser.add_argument(
        "--match-mode",
        choices=["exact", "substring", "cover_exact"],
        default="exact",
    )
    parser.add_argument(
        "--enable-f1-match",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--f1-threshold", type=float, default=0.7)
    parser.add_argument(
        "--enable-llm-judge",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--judge-model", type=str, default=None)
    parser.add_argument("--judge-base-url", type=str, default=None)
    parser.add_argument("--judge-api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--judge-timeout", type=int, default=20)
    parser.add_argument("--judge-max-tokens", type=int, default=128)
    parser.add_argument("--judge-max-retries", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--heartbeat-interval", type=float, default=10.0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed-on-resume", action="store_true")
    parser.add_argument("--disable-gold-answer-repair", action="store_true")
    return parser


async def main_async(args: argparse.Namespace) -> None:
    input_path = Path(args.input_jsonl)
    output_path = resolve_output_path(_input_path=input_path, output_arg=args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        raise ValueError(f"Environment variable `{args.api_key_env}` is empty.")
    judge_api_key = os.getenv(args.judge_api_key_env, "").strip()
    if args.enable_llm_judge and not judge_api_key:
        raise ValueError(f"Environment variable `{args.judge_api_key_env}` is empty.")
    if not args.base_url:
        raise ValueError("`--base-url` or `OPENAI_BASE_URL` must be provided.")
    if args.enable_llm_judge and not (args.judge_base_url or args.base_url):
        raise ValueError("`--judge-base-url` or `OPENAI_BASE_URL` must be provided when LLM judge is enabled.")

    source_path = output_path if args.resume and output_path.exists() else input_path
    records = load_jsonl(source_path)

    rollout_config = RolloutConfig(
        search_url=args.search_url,
        topk=args.topk,
        max_turns=args.max_turns,
        max_step_retries=args.max_step_retries,
        max_rollout_retries=args.max_rollout_retries,
        max_repair_retries=args.max_repair_retries,
        sleep_seconds=args.sleep_seconds,
        request_timeout=args.timeout,
        search_timeout=args.search_timeout,
        max_doc_chars=args.max_doc_chars,
        max_observation_chars=args.max_observation_chars,
        match_mode=args.match_mode,
        enable_f1_match=args.enable_f1_match,
        f1_threshold=args.f1_threshold,
        allow_gold_answer_repair=not args.disable_gold_answer_repair,
        enable_llm_judge=args.enable_llm_judge,
        judge_model=args.judge_model or args.model,
        judge_base_url=args.judge_base_url or args.base_url,
        judge_api_key=judge_api_key if args.enable_llm_judge else "",
        judge_timeout=args.judge_timeout,
        judge_max_tokens=args.judge_max_tokens,
        judge_max_retries=args.judge_max_retries,
    )

    start = max(args.start, 0)
    end = len(records) if args.end is None else min(args.end, len(records))
    failures: List[Dict[str, Any]] = []
    processed = 0
    batch_size = max(args.batch_size, 1)
    work_items: List[Tuple[int, Dict[str, Any]]] = []
    skipped_success = 0
    skipped_failed = 0

    for i in range(start, end):
        record = records[i]
        existing = str(record.get("gen_text_store", "")).strip()
        existing_status = str(record.get("reference_rollout_status", "")).strip().lower()
        if existing and not args.overwrite:
            skipped_success += 1
            continue
        if (
            args.resume
            and not args.overwrite
            and existing_status == "failed"
            and not args.retry_failed_on_resume
        ):
            skipped_failed += 1
            continue
        work_items.append((i, record))

    total_count = len(work_items)
    if total_count == 0:
        print("No work items to process.")
        return

    print(
        f"Starting async rollout with batch_size={batch_size} over {total_count} examples. "
        f"model={args.model} base_url={args.base_url}"
    )
    if args.resume and not args.overwrite:
        print(
            f"Resume filter: skipped_success={skipped_success} "
            f"skipped_failed={skipped_failed} pending={total_count}"
        )

    progress = {"completed": 0, "success": 0, "failed": 0}
    progress_lock = threading.Lock()
    stop_event = threading.Event()
    start_time = time.time()

    heartbeat_thread: Optional[threading.Thread] = None
    if args.heartbeat_interval > 0:
        heartbeat_thread = threading.Thread(
            target=heartbeat_loop,
            args=(start_time, total_count, progress, progress_lock, stop_event, args.heartbeat_interval),
            daemon=True,
        )
        heartbeat_thread.start()

    semaphore = asyncio.Semaphore(batch_size)

    async def _process_with_semaphore(i: int, record: Dict[str, Any]) -> None:
        nonlocal processed
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                question = ensure_question_mark(str(record["question"]))
                gold_answers = coerce_gold_answers(record)

                sample = SampleState(
                    record_index=i,
                    record_idx=record.get("idx", i),
                    question=question,
                    gold_answers=gold_answers,
                )

                await _process_one_sample(
                    session=session,
                    base_url=args.base_url,
                    api_key=api_key,
                    model=args.model,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    chat_timeout=args.timeout,
                    search_url=args.search_url,
                    topk=args.topk,
                    search_timeout=args.search_timeout,
                    max_doc_chars=args.max_doc_chars,
                    max_observation_chars=args.max_observation_chars,
                    rollout_config=rollout_config,
                    sample=sample,
                )

                records[i]["question"] = sample.question
                records[i]["reference_rollout_status"] = "success" if sample.success else "failed"
                records[i]["reference_rollout_attempts"] = sample.attempts
                records[i]["reference_rollout_used_gold_guidance"] = sample.used_gold_guidance
                records[i]["reference_rollout_model"] = args.model
                records[i]["reference_rollout_match_type"] = sample.match_type if sample.success else "failed"
                records[i]["reference_rollout_judge_used"] = sample.judge_used
                records[i]["reference_rollout_judge_reason"] = sample.judge_reason
                records[i]["gen_text_store"] = sample.trajectory if sample.success else ""

                if sample.success:
                    with progress_lock:
                        progress["success"] += 1
                    print(
                        f"[SUCCESS] idx={sample.record_idx} attempts={sample.attempts} "
                        f"gold_guided={sample.used_gold_guidance} match_type={sample.match_type}"
                    )
                else:
                    failure_item = {
                        "record_index": i,
                        "idx": sample.record_idx,
                        "question": sample.question,
                        "gold_answers": sample.gold_answers,
                        "failure_reasons": sample.failure_reasons,
                    }
                    failures.append(failure_item)
                    with progress_lock:
                        progress["failed"] += 1
                    print(f"[FAILED] idx={sample.record_idx} reasons={sample.failure_reasons[-3:]}")

                processed += 1
                with progress_lock:
                    progress["completed"] += 1
                if processed % max(args.save_every, 1) == 0:
                    atomic_write_jsonl(records, output_path)

    try:
        tasks = [_process_with_semaphore(i, record) for i, record in work_items]
        await asyncio.gather(*tasks)
    finally:
        stop_event.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1.0)

    atomic_write_jsonl(records, output_path)

    if failures:
        failure_path = output_path.with_suffix(output_path.suffix + ".failures.jsonl")
        with failure_path.open("w", encoding="utf-8") as f:
            for item in failures:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Wrote {len(failures)} failures to {failure_path}")

    success_count = 0
    for record in records[start:end]:
        if str(record.get("gen_text_store", "")).strip():
            success_count += 1

    elapsed = int(time.time() - start_time)
    print(
        f"Finished. success={success_count}/{total_count} range=[{start}, {end}) "
        f"elapsed={elapsed}s output={output_path}"
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    load_dotenv(Path(args.env_file), override=False)
    if not args.model:
        args.model = os.getenv("OPENAI_MODEL", "deepseek-v4-pro").strip()
    if not args.base_url:
        args.base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if not args.judge_model:
        args.judge_model = os.getenv("OPENAI_JUDGE_MODEL", "").strip() or args.model
    if not args.judge_base_url:
        args.judge_base_url = os.getenv("OPENAI_JUDGE_BASE_URL", "").strip() or args.base_url

    asyncio.run(main_async(args))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
