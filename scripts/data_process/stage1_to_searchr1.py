"""
Convert a generic stage1.jsonl QA-style dataset into Search-R1 parquet format.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import datasets


def make_prefix(question: str, template_type: str) -> str:
    question = question.strip()
    if question and question[-1] != "?":
        question += "?"

    if template_type == "base":
        return (
            "Answer the given question. "
            "You must conduct reasoning inside <think> and </think> first every time you get new information. "
            "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> "
            "and it will return the top searched results between <information> and </information>. "
            "You can search as many times as your want. "
            "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, "
            "without detailed illustrations. For example, <answer> Beijing </answer>. "
            f"Question: {question}\n"
        )
    raise NotImplementedError(f"Unknown template_type: {template_type}")


def _pick_first(record: Dict[str, Any], candidates: List[str]) -> Optional[Any]:
    for key in candidates:
        if key in record and record[key] is not None:
            return record[key]
    return None


def normalize_targets(answer_value: Any) -> List[str]:
    if answer_value is None:
        return []
    if isinstance(answer_value, list):
        targets = []
        for item in answer_value:
            if item is None:
                continue
            if isinstance(item, dict):
                item = _pick_first(item, ["text", "answer", "value", "target"])
                if item is None:
                    continue
            targets.append(str(item).strip())
        return [x for x in targets if x]
    if isinstance(answer_value, dict):
        nested = _pick_first(answer_value, ["target", "targets", "answer", "answers", "text", "value"])
        return normalize_targets(nested)
    return [str(answer_value).strip()]


def extract_question(record: Dict[str, Any]) -> str:
    question = _pick_first(
        record,
        [
            "question",
            "query",
            "instruction",
            "problem",
            "input",
        ],
    )

    if isinstance(question, dict):
        question = _pick_first(question, ["content", "text", "question", "query"])
    if question is None:
        raise KeyError(f"Cannot find question field in record keys: {list(record.keys())}")
    return str(question).strip()


def extract_answers(record: Dict[str, Any]) -> List[str]:
    answer_value = _pick_first(
        record,
        [
            "golden_answers",
            "answers",
            "answer",
            "ground_truth",
            "target",
            "targets",
            "solution",
            "output",
            "response",
        ],
    )
    targets = normalize_targets(answer_value)
    if not targets:
        raise KeyError(f"Cannot find answer field in record keys: {list(record.keys())}")
    return targets


def convert_record(record: Dict[str, Any], idx: int, split: str, data_source: str, template_type: str) -> Dict[str, Any]:
    question = extract_question(record)
    targets = extract_answers(record)

    return {
        "data_source": data_source,
        "prompt": [{
            "role": "user",
            "content": make_prefix(question, template_type=template_type),
        }],
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "target": targets,
            },
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "source_question": question,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--local_dir", type=str, required=True)
    parser.add_argument("--data_source", type=str, default="hotpotqa")
    parser.add_argument("--template_type", type=str, default="base")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    records: List[Dict[str, Any]] = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if args.limit is not None and idx >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            records.append(convert_record(raw, idx=idx, split=args.split, data_source=args.data_source, template_type=args.template_type))

    dataset = datasets.Dataset.from_list(records)
    os.makedirs(args.local_dir, exist_ok=True)
    output_path = os.path.join(args.local_dir, f"{args.split}.parquet")
    dataset.to_parquet(output_path)

    print(f"Saved {len(dataset)} examples to {output_path}")


if __name__ == "__main__":
    main()
