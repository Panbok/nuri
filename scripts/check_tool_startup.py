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
    parser.add_argument("--tool", type=Path)
    parser.add_argument(
        "--target",
        choices=("nuri-bench", "nuri-snapshot", "nuri-autotest"),
    )
    parser.add_argument("--variant", default="release-fast")
    parser.add_argument("--capability", default="runtime-tools")
    parser.add_argument("--case", required=True)
    parser.add_argument("--budget-ms", type=float, default=200.0)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7)
    return parser.parse_args(argv)


def resolve_tool(args: argparse.Namespace) -> Path:
    if args.tool:
        return args.tool
    if not args.target:
        raise ValueError("provide --tool or --target")
    registry_path = ROOT / "build" / "_registry" / "trees.json"
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read canonical build registry: {error}") from error
    for record in registry.get("trees", {}).values():
        if (
            record.get("variant") != args.variant
            or record.get("capability") != args.capability
            or record.get("state") != "ready"
        ):
            continue
        manifest_path = Path(str(record.get("path", ""))) / ".nuri-artifacts.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        artifact = manifest.get("targets", {}).get(args.target, {})
        receipt = manifest.get("receipts", {}).get(args.target)
        path = Path(str(artifact.get("path", "")))
        if receipt and path.is_file():
            return path
    raise ValueError(
        f"no identity-checked artifact for {args.variant}/{args.capability}/{args.target}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.budget_ms <= 0 or args.warmups < 0 or args.iterations < 1:
        print("budget must be positive, warmups non-negative, and iterations positive", file=sys.stderr)
        return 2
    try:
        tool = resolve_tool(args)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    commands = (("list",), ("explain", "--case", args.case))
    results: list[dict[str, object]] = []
    try:
        for command in commands:
            samples = measure_command(
                tool,
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
                "tool": str(tool),
                "passed": passed,
                "results": results,
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
