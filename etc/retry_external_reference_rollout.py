#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retry external_reference_rollout.py until it succeeds or retry budget is exhausted."
    )
    parser.add_argument("--max-retries", type=int, default=300)
    parser.add_argument("--retry-delay-seconds", type=float, default=10.0)
    parser.add_argument("--python-bin", type=str, default=sys.executable or "python")
    parser.add_argument(
        "--script-path",
        type=str,
        default="/data/yanfeizhang/OPSD_experiment/Search-R1/etc/external_reference_rollout.py",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Optional extra args appended to external_reference_rollout.py",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    script_path = Path(args.script_path)
    if not script_path.exists():
        print(f"[FATAL] rollout script not found: {script_path}", file=sys.stderr)
        return 2

    base_cmd = [
        args.python_bin,
        str(script_path),
        "--resume",
        "--disable-gold-answer-repair",
    ]

    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    base_cmd.extend(extra_args)

    total_attempts = max(args.max_retries, 0) + 1
    print(
        f"[START] command={' '.join(base_cmd)} | "
        f"max_retries={args.max_retries} | total_attempts={total_attempts}"
    )

    for attempt_idx in range(total_attempts):
        human_attempt = attempt_idx + 1
        print(f"[RUN] attempt={human_attempt}/{total_attempts}")
        try:
            completed = subprocess.run(base_cmd, check=False)
        except KeyboardInterrupt:
            print("[STOP] interrupted by user")
            return 130
        except Exception as exc:
            print(f"[ERROR] failed to launch subprocess: {exc}", file=sys.stderr)
            completed = subprocess.CompletedProcess(base_cmd, returncode=1)

        if completed.returncode == 0:
            print(f"[DONE] rollout finished successfully on attempt {human_attempt}.")
            return 0

        if human_attempt >= total_attempts:
            print(
                f"[FAILED] rollout kept failing after {human_attempt} attempts. "
                f"last_returncode={completed.returncode}",
                file=sys.stderr,
            )
            return completed.returncode or 1

        print(
            f"[RETRY] returncode={completed.returncode}; "
            f"sleeping {args.retry_delay_seconds}s before restart."
        )
        try:
            time.sleep(args.retry_delay_seconds)
        except KeyboardInterrupt:
            print("[STOP] interrupted during retry sleep")
            return 130

    return 1


if __name__ == "__main__":
    sys.exit(main())
