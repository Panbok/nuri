#!/usr/bin/env python3
"""Measure release metadata command startup against the tooling contract."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]


def measure_command(
    executable: Path,
    arguments: Sequence[str],
    *,
    warmups: int,
    iterations: int,
) -> list[float]:
    command = [str(executable), *arguments]
    for _ in range(warmups):
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, check=False
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{' '.join(arguments)} exited {completed.returncode}: {completed.stderr.strip()}"
            )

    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, check=False
        )
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        if completed.returncode != 0:
            raise RuntimeError(
                f"{' '.join(arguments)} exited {completed.returncode}: {completed.stderr.strip()}"
            )
        samples.append(elapsed_ms)
    return samples


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True, type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--budget-ms", type=float, default=200.0)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.budget_ms <= 0 or args.warmups < 0 or args.iterations < 1:
        print("budget must be positive, warmups non-negative, and iterations positive", file=sys.stderr)
        return 2
    if not args.tool.is_file():
        print(f"tool executable not found: {args.tool}", file=sys.stderr)
        return 2

    commands = (("list",), ("explain", "--case", args.case))
    results: list[dict[str, object]] = []
    try:
        for command in commands:
            samples = measure_command(
                args.tool,
                command,
                warmups=args.warmups,
                iterations=args.iterations,
            )
            median_ms = statistics.median(samples)
            results.append(
                {
                    "command": list(command),
                    "medianMs": round(median_ms, 3),
                    "samplesMs": [round(sample, 3) for sample in samples],
                    "budgetMs": args.budget_ms,
                    "passed": median_ms <= args.budget_ms,
                }
            )
    except (OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 2

    passed = all(bool(result["passed"]) for result in results)
    print(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "nuri.tool.startup_check",
                "tool": str(args.tool),
                "passed": passed,
                "results": results,
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
