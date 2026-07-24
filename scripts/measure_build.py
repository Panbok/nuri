#!/usr/bin/env python3
"""Versioned, fixture-safe Nuri build-performance measurement harness."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
SCENARIOS = (
    "cold-configure",
    "cold-renderer",
    "cold-editor",
    "warm-noop",
    "editor-cpp",
    "renderer-cpp",
    "renderer-header",
    "pch-header",
    "compatible-target-switch",
    "incompatible-variant-switch",
    "contract-validation",
    "concurrent-same-tree",
    "concurrent-distinct-trees",
    "build-run-competing-rebuild",
    "no-build-identity",
)
MUTATION_FILES = {
    "editor-cpp": "editor/src/app/editor_runtime.cpp",
    "renderer-cpp": "lib/nuri/core/runtime_config.cpp",
    "renderer-header": "lib/nuri/core/runtime_config.h",
    "pch-header": "lib/nuri/pch.h",
}


class MeasurementError(RuntimeError):
    pass


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    return_code: int
    duration_seconds: float
    stdout: str
    stderr: str


def run(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    timeout: float,
) -> CommandResult:
    started = time.perf_counter()
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
        check=False,
    )
    return CommandResult(
        command=list(command),
        return_code=completed.returncode,
        duration_seconds=time.perf_counter() - started,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def git_output(workspace: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=workspace,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    if completed.returncode:
        raise MeasurementError(completed.stderr.strip())
    return completed.stdout.strip()


def environment_record(workspace: Path, jobs: int) -> dict[str, object]:
    def version(name: str) -> dict[str, str | None]:
        path = shutil.which(name)
        if not path:
            return {"path": None, "version": None}
        completed = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
        output = (completed.stdout + completed.stderr).strip().splitlines()
        return {"path": str(Path(path).resolve()), "version": output[0] if output else None}

    dirty = bool(git_output(workspace, "status", "--porcelain=v1"))
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "nuri.build_measurement_environment",
        "capturedUtc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {
            "root": str(workspace.resolve()),
            "revision": git_output(workspace, "rev-parse", "HEAD"),
            "dirty": dirty,
        },
        "host": {
            "node": platform.node(),
            "os": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logicalCpuCount": os.cpu_count(),
            "memoryBytes": _memory_bytes(),
            "powerPolicy": os.environ.get("NURI_MEASUREMENT_POWER_POLICY", "unreported"),
            "storage": os.environ.get("NURI_MEASUREMENT_STORAGE", "unreported"),
            "backgroundLoad": os.environ.get(
                "NURI_MEASUREMENT_BACKGROUND_LOAD", "unreported"
            ),
        },
        "tools": {
            name: version(name)
            for name in ("python", "cmake", "ninja", "clang++", "clang-cl", "ctest")
        },
        "jobs": jobs,
        "cacheStateDimensions": {
            "tree": ["cold", "warm"],
            "compilerCache": ["disabled", "cold", "warm"],
            "filesystemPageCache": "reported-not-controlled",
            "vcpkgInstalled": ["cold", "warm"],
            "vcpkgBinaryAndDownloadCache": ["cold", "warm"],
        },
        "comparisonContract": {
            "sameMachine": True,
            "sameRevision": True,
            "minimumRepetitions": 3,
            "warmups": 1,
            "order": "seeded-randomized",
            "noiseFloorMethod": "median absolute pairwise warm-noop delta",
            "relativeRegressionThreshold": 0.10,
            "absoluteDurationFloorSeconds": 0.050,
        },
    }


def _memory_bytes() -> int | None:
    if os.name == "nt":
        import ctypes

        class MemoryStatus(ctypes.Structure):  # type: ignore[name-defined]
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memoryLoad", ctypes.c_ulong),
                ("totalPhys", ctypes.c_ulonglong),
                ("availPhys", ctypes.c_ulonglong),
                ("totalPageFile", ctypes.c_ulonglong),
                ("availPageFile", ctypes.c_ulonglong),
                ("totalVirtual", ctypes.c_ulonglong),
                ("availVirtual", ctypes.c_ulonglong),
                ("availExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        return int(status.totalPhys) if ctypes.windll.kernel32.GlobalMemoryStatusEx(
            ctypes.byref(status)
        ) else None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * size)
    except (AttributeError, OSError, ValueError):
        return None


def driver(workspace: Path, *arguments: str) -> list[str]:
    return [
        sys.executable,
        str(workspace / "scripts" / "nuri_build.py"),
        *arguments,
    ]


def selection(
    workspace: Path,
    target: str,
    *,
    variant: str = "release-checked",
    capability: str = "developer-full",
) -> list[str]:
    return driver(
        workspace,
        "build",
        "--variant",
        variant,
        "--capability",
        capability,
        "--target",
        target,
    )


def scenario_commands(workspace: Path, scenario: str) -> list[list[str]]:
    renderer = selection(workspace, "nuri_renderer")
    editor = selection(workspace, "nuri_editor")
    if scenario == "cold-configure":
        return [
            driver(
                workspace,
                "configure",
                "--variant",
                "release-checked",
                "--capability",
                "developer-full",
                "--target",
                "nuri_renderer",
            )
        ]
    if scenario in {"cold-renderer", "warm-noop", "renderer-cpp", "renderer-header", "pch-header"}:
        return [renderer]
    if scenario in {"cold-editor", "editor-cpp"}:
        return [editor]
    if scenario == "compatible-target-switch":
        return [renderer, editor]
    if scenario == "incompatible-variant-switch":
        return [
            selection(
                workspace,
                "nuri-bench",
                variant="release-fast",
                capability="runtime-tools",
            ),
            selection(
                workspace,
                "nuri-bench",
                variant="release-tracy",
                capability="runtime-tools",
            ),
        ]
    if scenario == "contract-validation":
        return [
            driver(
                workspace,
                "test",
                "--variant",
                "release-checked",
                "--capability",
                "developer-full",
                "--target",
                "nuri-tests-contract",
                "--labels",
                "contract",
            )
        ]
    if scenario == "concurrent-same-tree":
        return [renderer, renderer]
    if scenario == "concurrent-distinct-trees":
        return [
            renderer,
            selection(
                workspace,
                "nuri-bench",
                variant="release-fast",
                capability="runtime-tools",
            ),
        ]
    if scenario == "build-run-competing-rebuild":
        return [
            selection(workspace, "nuri_editor"),
            driver(
                workspace,
                "run",
                "--variant",
                "release-checked",
                "--capability",
                "developer-full",
                "--target",
                "nuri_editor",
                "--no-build",
                "--",
                "--help",
            ),
            selection(workspace, "nuri_editor"),
        ]
    if scenario == "no-build-identity":
        return [
            selection(
                workspace,
                "nuri-bench",
                variant="release-fast",
                capability="runtime-tools",
            ),
            driver(
                workspace,
                "run",
                "--variant",
                "release-fast",
                "--capability",
                "runtime-tools",
                "--target",
                "nuri-bench",
                "--no-build",
                "--",
                "list",
            ),
            driver(
                workspace,
                "run",
                "--variant",
                "release-tracy",
                "--capability",
                "runtime-tools",
                "--target",
                "nuri-bench",
                "--no-build",
                "--",
                "list",
            ),
        ]
    raise MeasurementError(f"unknown scenario: {scenario}")


def command_counts(text: str) -> dict[str, int]:
    lowered = text.lower()
    return {
        "configure": lowered.count("configuring "),
        "compile": len(re.findall(r"building cxx object|\\.cpp\\.obj|\\.cpp\\.o", lowered)),
        "pch": lowered.count("cmake_pch"),
        "archive": lowered.count("linking cxx static library"),
        "link": len(re.findall(r"linking cxx (executable|shared library)", lowered)),
        "vcpkgMutation": lowered.count("installing "),
        "ninjaNoWork": lowered.count("ninja: no work to do"),
        "lockWait": lowered.count("waiting for build lease"),
    }


def measure_scenario(
    workspace: Path,
    scenario: str,
    *,
    environment: dict[str, str],
    timeout: float,
) -> dict[str, object]:
    if scenario.startswith("cold-"):
        target = "nuri_editor" if scenario == "cold-editor" else "nuri_renderer"
        resolved = run(
            driver(
                workspace,
                "resolve",
                "--variant",
                "release-checked",
                "--capability",
                "developer-full",
                "--target",
                target,
            ),
            cwd=workspace,
            environment=environment,
            timeout=timeout,
        )
        if resolved.return_code:
            raise MeasurementError(resolved.stderr)
        tree = Path(json.loads(resolved.stdout)["tree"]).resolve()
        managed = (workspace / "build" / "_trees").resolve()
        if tree == managed or not tree.is_relative_to(managed):
            raise MeasurementError(f"resolved cold tree escaped fixture: {tree}")
        if tree.exists():
            shutil.rmtree(tree)
    commands = scenario_commands(workspace, scenario)
    mutation = MUTATION_FILES.get(scenario)
    original: bytes | None = None
    mutation_path: Path | None = None
    if mutation:
        mutation_path = workspace / mutation
        original = mutation_path.read_bytes()
        mutation_path.write_bytes(
            original + b"\n// nuri-build-measurement deterministic edit\n"
        )
    try:
        results: list[CommandResult] = []
        if scenario in {"concurrent-same-tree", "concurrent-distinct-trees"}:
            started = time.perf_counter()
            processes = [
                subprocess.Popen(
                    command,
                    cwd=workspace,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    errors="replace",
                )
                for command in commands
            ]
            for command, process in zip(commands, processes, strict=True):
                stdout, stderr = process.communicate(timeout=timeout)
                results.append(
                    CommandResult(
                        command,
                        process.returncode,
                        time.perf_counter() - started,
                        stdout,
                        stderr,
                    )
                )
        else:
            for command in commands:
                results.append(
                    run(
                        command,
                        cwd=workspace,
                        environment=environment,
                        timeout=timeout,
                    )
                )
        combined = "\n".join(
            result.stdout + "\n" + result.stderr for result in results
        )
        expected_failure = scenario == "no-build-identity"
        failures = sum(result.return_code != 0 for result in results)
        passed = failures == (1 if expected_failure else 0)
        return {
            "scenario": scenario,
            "passed": passed,
            "durationSeconds": round(
                max((result.duration_seconds for result in results), default=0.0)
                if scenario.startswith("concurrent-")
                else sum(result.duration_seconds for result in results),
                6,
            ),
            "commands": [result.command for result in results],
            "returnCodes": [result.return_code for result in results],
            "counts": command_counts(combined),
            "cacheState": {
                "tree": "cold" if scenario.startswith("cold-") else "warm",
                "compilerCache": (
                    "disabled"
                    if environment.get("NURI_COMPILER_CACHE", "off") == "off"
                    else "reported-by-launcher"
                ),
                "vcpkgInstalled": "warm",
                "vcpkgBinaryAndDownloadCache": "warm",
                "filesystemPageCache": "uncontrolled",
            },
        }
    finally:
        if mutation_path is not None and original is not None:
            mutation_path.write_bytes(original)
            if mutation_path.read_bytes() != original:
                raise MeasurementError(f"failed to restore fixture edit: {mutation_path}")


def require_clean_fixture(workspace: Path) -> None:
    status = git_output(workspace, "status", "--porcelain=v1")
    if status:
        raise MeasurementError(
            "measurement execution requires a clean isolated fixture; "
            "use --create-fixture or provide a clean worktree"
        )


def create_fixture(parent: Path | None) -> tuple[Path, bool]:
    root = parent or Path(tempfile.mkdtemp(prefix="nuri-build-measurement-"))
    if parent:
        root.mkdir(parents=True, exist_ok=True)
    fixture = root / "source"
    revision = git_output(ROOT, "rev-parse", "HEAD")
    completed = subprocess.run(
        ["git", "worktree", "add", "--detach", str(fixture), revision],
        cwd=ROOT,
        check=False,
    )
    if completed.returncode:
        raise MeasurementError("failed to create isolated git worktree")
    subprocess.run(
        ["git", "submodule", "update", "--init", "--recursive"],
        cwd=fixture,
        check=True,
    )
    return fixture, parent is None


def summarize(samples: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for sample in samples:
        grouped.setdefault(str(sample["scenario"]), []).append(sample)
    result: dict[str, object] = {}
    for scenario, values in grouped.items():
        durations = [float(value["durationSeconds"]) for value in values]
        result[scenario] = {
            "medianSeconds": round(statistics.median(durations), 6),
            "rangeSeconds": [
                round(min(durations), 6),
                round(max(durations), 6),
            ],
            "rawSeconds": durations,
            "failures": sum(not bool(value["passed"]) for value in values),
        }
    warm = [
        float(sample["durationSeconds"])
        for sample in grouped.get("warm-noop", [])
    ]
    pairwise = [
        abs(right - left) for left, right in zip(warm, warm[1:])
    ]
    return {
        "scenarios": result,
        "noiseFloorSeconds": (
            round(statistics.median(pairwise), 6) if pairwise else None
        ),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--output", type=Path)
    result.add_argument("--workspace", type=Path)
    result.add_argument("--create-fixture", type=Path)
    result.add_argument("--plan", action="store_true")
    result.add_argument("--scenario", action="append", choices=SCENARIOS)
    result.add_argument("--repetitions", type=int, default=3)
    result.add_argument("--warmups", type=int, default=1)
    result.add_argument("--jobs", type=int, default=0)
    result.add_argument("--seed", type=int, default=20260724)
    result.add_argument("--timeout-seconds", type=float, default=3600.0)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    selected = args.scenario or list(SCENARIOS)
    if args.repetitions < 3 or args.warmups < 0 or args.timeout_seconds <= 0:
        print(
            "repetitions must be >= 3, warmups >= 0, and timeout positive",
            file=sys.stderr,
        )
        return 2
    if args.plan:
        print(
            json.dumps(
                {
                    "schemaVersion": SCHEMA_VERSION,
                    "kind": "nuri.build_measurement_plan",
                    "scenarios": [
                        {
                            "id": scenario,
                            "commands": scenario_commands(ROOT, scenario),
                            "fixtureMutation": MUTATION_FILES.get(scenario),
                        }
                        for scenario in selected
                    ],
                    "repetitions": args.repetitions,
                    "warmups": args.warmups,
                    "seed": args.seed,
                },
                indent=2,
            )
        )
        return 0
    temporary_fixture = False
    workspace = args.workspace
    try:
        if args.create_fixture is not None or workspace is None:
            workspace, temporary_fixture = create_fixture(args.create_fixture)
        assert workspace is not None
        workspace = workspace.resolve()
        require_clean_fixture(workspace)
        environment = dict(os.environ)
        environment["NURI_MEASUREMENT_JOBS"] = str(args.jobs)
        order = [
            scenario
            for scenario in selected
            for _ in range(args.warmups + args.repetitions)
        ]
        random.Random(args.seed).shuffle(order)
        seen: dict[str, int] = {}
        samples: list[dict[str, object]] = []
        for scenario in order:
            index = seen.get(scenario, 0)
            seen[scenario] = index + 1
            sample = measure_scenario(
                workspace,
                scenario,
                environment=environment,
                timeout=args.timeout_seconds,
            )
            if index >= args.warmups:
                samples.append(sample)
        report = {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "nuri.build_performance_report",
            "environment": environment_record(workspace, args.jobs),
            "order": order,
            "samples": samples,
            "summary": summarize(samples),
        }
        output = args.output or (
            ROOT
            / "artifacts"
            / "build-performance"
            / f"measurement-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(output)
        return 0 if all(bool(sample["passed"]) for sample in samples) else 1
    except (MeasurementError, OSError, subprocess.SubprocessError) as error:
        print(f"measure-build: {error}", file=sys.stderr)
        return 2
    finally:
        if temporary_fixture and workspace:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(workspace)],
                cwd=ROOT,
                check=False,
            )
            shutil.rmtree(workspace.parent, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
