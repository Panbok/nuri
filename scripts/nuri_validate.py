#!/usr/bin/env python3
"""Unified contract-plan and artifact lifecycle entry point for renderer tools."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts" / "validate"
ARTIFACT_ROOT = DEFAULT_ARTIFACT_ROOT
ID_RE = re.compile(r"^[a-z0-9_]+(?:[.-][a-z0-9_]+)*$")
RESERVED = {"con", "prn", "aux", "nul"} | {
    f"{prefix}{number}" for prefix in ("com", "lpt") for number in range(1, 10)
}
GOVERNED_APPROVAL_KINDS = {
    "nuri.baseline.approval",
    "nuri.benchmark.baseline_approval",
    "nuri.autotest.baseline_approval",
}


@dataclass(frozen=True)
class Step:
    name: str
    profile: str
    labels: str = "contract"


PRESETS: dict[str, tuple[Step, ...]] = {
    "contract": (
        Step("renderer-contract", "tests"),
        Step("benchmark-tool-core", "bench-tests", "tool-core"),
        Step("snapshot-tool-core", "snapshot-tests", "tool-core"),
        Step("autotest-tool-core", "autotest-tests", "tool-core"),
    ),
    "tool-core": (
        Step("benchmark-tool-core", "bench-tests", "tool-core"),
        Step("snapshot-tool-core", "snapshot-tests", "tool-core"),
        Step("autotest-tool-core", "autotest-tests", "tool-core"),
    ),
    "renderer": (Step("renderer-contract", "tests", "renderer"),),
    "asset": (Step("asset-contract", "tests", "asset"),),
}


def _json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def run_registry_roots() -> tuple[Path, ...]:
    if ARTIFACT_ROOT != DEFAULT_ARTIFACT_ROOT:
        return (ARTIFACT_ROOT,)
    return (
        DEFAULT_ARTIFACT_ROOT,
        ROOT / "artifacts" / "bench",
        ROOT / "artifacts" / "snapshots",
        ROOT / "artifacts" / "autotests",
    )


def _safe_child(root: Path, child: Path) -> Path:
    root_resolved = root.resolve()
    candidate = child.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"path escapes artifact root: {child}") from error
    return candidate


def _valid_identifier(value: object) -> bool:
    if not isinstance(value, str) or not value or len(value) > 255:
        return False
    if not ID_RE.fullmatch(value):
        return False
    return all(part not in RESERVED and len(part) <= 64 for part in re.split(r"[.-]", value))


def _is_governed_approval(document: object) -> bool:
    return (
        isinstance(document, dict)
        and document.get("schemaVersion") == 1
        and document.get("kind") in GOVERNED_APPROVAL_KINDS
        and isinstance(document.get("planDigest"), str)
        and str(document["planDigest"]).startswith("sha256:")
    )


def _validate_nested_ids(document: dict[str, object], display_path: Path) -> list[str]:
    errors: list[str] = []
    keyed_lists = {
        "checkpoints": "id",
        "assertions": "id",
        "captures": "target",
        "timeline": "id",
        "metricWindows": "id",
    }

    def visit(value: object, location: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in keyed_lists and isinstance(child, list):
                    identity_key = keyed_lists[key]
                    identities: set[str] = set()
                    for index, item in enumerate(child):
                        if not isinstance(item, dict):
                            continue
                        identity = item.get(identity_key)
                        if identity is None and key == "captures":
                            continue
                        if not _valid_identifier(identity):
                            errors.append(f"{display_path}: unsafe {location}.{key}[{index}].{identity_key} {identity!r}")
                        elif identity in identities:
                            errors.append(f"{display_path}: duplicate {location}.{key} id {identity!r}")
                        else:
                            identities.add(str(identity))
                visit(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{location}[{index}]")

    visit(document, "root")
    for key in ("samples", "measurementFrames"):
        if key in document and (not isinstance(document[key], int) or int(document[key]) <= 0):
            errors.append(f"{display_path}: {key} must be greater than zero")
    resolution = document.get("resolution")
    if not isinstance(resolution, list) or len(resolution) != 2 or any(not isinstance(item, int) or item <= 0 for item in resolution):
        errors.append(f"{display_path}: resolution must contain two positive integers")
    return errors


def _validate_case_location(document: dict[str, object], path: Path, case_root: Path) -> list[str]:
    """Keep suite, ID namespace, and repository location aligned."""
    errors: list[str] = []
    display_path = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    relative = path.relative_to(case_root)
    suite = document.get("suite")
    case_id = document.get("id")
    if len(relative.parts) < 2:
        errors.append(f"{display_path}: cases must live under a suite directory")
        return errors
    directory_suite = relative.parts[0]
    if suite != directory_suite:
        errors.append(
            f"{display_path}: suite {suite!r} must match directory {directory_suite!r}"
        )
    if isinstance(case_id, str) and isinstance(suite, str):
        id_suite = case_id.partition(".")[0]
        if id_suite != suite:
            errors.append(
                f"{display_path}: id namespace {id_suite!r} must match suite {suite!r}"
            )
    if suite == "perf":
        errors.append(
            f"{display_path}: suite 'perf' is reserved; use 'stress' for non-gating soak work and nuri-bench for performance authority"
        )
    return errors


def _validate_run_wrapper_contract(text: str, suffix: str, display_path: Path) -> list[str]:
    """Keep native wrappers thin and route policy through the canonical driver."""
    errors: list[str] = []
    required = ("nuri_build.py", "legacy-wrapper-run")
    for token in required:
        if token not in text:
            errors.append(f"{display_path}: missing canonical driver token {token!r}")
    return errors


def validate_repository() -> list[str]:
    errors: list[str] = []
    seen: dict[tuple[str, str], Path] = {}
    roots = {
        "benchmark": ROOT / "tools" / "cases" / "benchmarks",
        "snapshot": ROOT / "tools" / "cases" / "snapshots",
        "autotest": ROOT / "tools" / "cases" / "autotests",
    }
    for tool, case_root in roots.items():
        for path in sorted(case_root.rglob("*.json")):
            try:
                with path.open(encoding="utf-8") as stream:
                    document = json.load(stream)
            except (OSError, json.JSONDecodeError) as error:
                errors.append(f"{path.relative_to(ROOT)}: invalid JSON: {error}")
                continue
            if not isinstance(document, dict):
                errors.append(f"{path.relative_to(ROOT)}: root must be an object")
                continue
            case_id = document.get("id")
            suite = document.get("suite")
            if document.get("schemaVersion") != 1:
                errors.append(f"{path.relative_to(ROOT)}: schemaVersion must be 1")
            if not _valid_identifier(case_id):
                errors.append(f"{path.relative_to(ROOT)}: unsafe id {case_id!r}")
            if not _valid_identifier(suite) or "." in str(suite):
                errors.append(f"{path.relative_to(ROOT)}: unsafe suite {suite!r}")
            errors.extend(_validate_case_location(document, path, case_root))
            if isinstance(case_id, str):
                key = (tool, case_id)
                if key in seen:
                    errors.append(
                        f"{path.relative_to(ROOT)}: duplicate id also in {seen[key].relative_to(ROOT)}"
                    )
                else:
                    seen[key] = path
            errors.extend(_validate_nested_ids(document, path.relative_to(ROOT)))

    for path in sorted((ROOT / "tools" / "schemas").glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{path.relative_to(ROOT)}: invalid schema JSON: {error}")
            continue
        if not isinstance(document, dict) or "$schema" not in document or "$id" not in document:
            errors.append(f"{path.relative_to(ROOT)}: schema requires $schema and $id")

    for path in sorted((ROOT / "tools" / "profiles").glob("*.json")):
        try:
            profile = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{path.relative_to(ROOT)}: invalid profile JSON: {error}")
            continue
        if profile.get("kind") != "nuri.baseline.profile" or profile.get("schemaVersion") != 1:
            errors.append(f"{path.relative_to(ROOT)}: invalid profile kind/schema")
        if not _valid_identifier(profile.get("id")):
            errors.append(f"{path.relative_to(ROOT)}: unsafe profile id")
        policy = profile.get("benchmarkPolicy", {})
        if not isinstance(policy, dict) or policy.get("thresholdOwnership") != "baseline":
            errors.append(f"{path.relative_to(ROOT)}: benchmark thresholds must be baseline-owned")

    for name in ("nuri-benchmarks", "nuri-snapshots", "nuri-autotests"):
        path = ROOT / ".codex" / "skills" / name / "SKILL.md"
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as error:
            errors.append(f"{path.relative_to(ROOT)}: {error}")
            continue
        if not text.startswith("---\n") or "\nname:" not in text or "\ndescription:" not in text:
            errors.append(f"{path.relative_to(ROOT)}: missing skill frontmatter")

    windows_build = ROOT / "scripts" / "_nuri_build.bat"
    try:
        build_text = windows_build.read_text(encoding="utf-8")
    except OSError as error:
        errors.append(f"{windows_build.relative_to(ROOT)}: {error}")
    else:
        if "nuri_build.py" not in build_text or "legacy-build" not in build_text:
            errors.append(
                f"{windows_build.relative_to(ROOT)}: compatibility shim must route to nuri_build.py"
            )

    canonical_driver = ROOT / "scripts" / "nuri_build.py"
    try:
        driver_text = canonical_driver.read_text(encoding="utf-8")
    except OSError as error:
        errors.append(f"{canonical_driver.relative_to(ROOT)}: {error}")
    else:
        for token in (
            "validate_manifest",
            "--no-build",
            "FileLock",
            "runtimeSearchPaths",
        ):
            if token not in driver_text:
                errors.append(
                    f"{canonical_driver.relative_to(ROOT)}: missing build contract token {token!r}"
                )

    for wrapper_name in ("run_benchmarks", "run_snapshots", "run_autotests"):
        for suffix in (".bat", ".sh"):
            wrapper = ROOT / "scripts" / f"{wrapper_name}{suffix}"
            try:
                wrapper_text = wrapper.read_text(encoding="utf-8")
            except OSError as error:
                errors.append(f"{wrapper.relative_to(ROOT)}: {error}")
                continue
            errors.extend(
                _validate_run_wrapper_contract(
                    wrapper_text, suffix, wrapper.relative_to(ROOT)
                )
            )

    benchmark_runner = ROOT / "tools" / "benchmark" / "src" / "benchmark_runner.cpp"
    try:
        runner_text = benchmark_runner.read_text(encoding="utf-8")
    except OSError as error:
        errors.append(f"{benchmark_runner.relative_to(ROOT)}: {error}")
    else:
        if any(token in runner_text for token in ("popen(", "_popen(", "system(")):
            errors.append(
                f"{benchmark_runner.relative_to(ROOT)}: diagnostic tools must use the argv process seam"
            )
        if "runProcess(command)" not in runner_text:
            errors.append(
                f"{benchmark_runner.relative_to(ROOT)}: Tracy helpers are not routed through runProcess"
            )
    return errors


def affected_preset(paths: Iterable[str]) -> str:
    values = tuple(path.replace("\\", "/") for path in paths)
    if not values:
        return "contract"
    if all(value.startswith(("docs/", ".codex/")) for value in values):
        return "tool-core"
    if any("asset" in value or "import" in value for value in values):
        return "asset"
    if any(value.startswith(("lib/", "app/", "shaders/")) for value in values):
        return "renderer"
    return "contract"


def select_steps(preset: str, shard_index: int, shard_count: int) -> tuple[Step, ...]:
    steps = PRESETS[preset]
    if shard_count < 1 or shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard index must be in [0, shard-count)")
    return tuple(step for index, step in enumerate(steps) if index % shard_count == shard_index)


def build_dir(profile: str) -> Path:
    raise RuntimeError("profile-derived build directories are no longer supported")


def step_command(
    step: Step,
    no_build: bool,
    jobs: int = 0,
    test_filter: str | None = None,
    repeat: int = 1,
    extra_ctest_args: Iterable[str] = (),
) -> list[str]:
    aggregates = {
        "tests": "nuri-tests-renderer-contract",
        "bench-tests": "nuri-tests-benchmark",
        "snapshot-tests": "nuri-tests-snapshot",
        "autotest-tests": "nuri-tests-autotest",
    }
    command = [
        sys.executable,
        str(ROOT / "scripts" / "nuri_build.py"),
        "test",
        "--variant",
        "release-checked",
        "--capability",
        "developer-full",
        "--target",
        aggregates[step.profile],
        "--labels",
        step.labels,
    ]
    if no_build:
        command.append("--no-build")
    if jobs > 0:
        command.extend(["--jobs", str(jobs)])
    command.append("--")
    if test_filter:
        command.extend(["-R", test_filter])
    if repeat > 1:
        command.extend(["--repeat", f"until-fail:{repeat}"])
    command.extend(extra_ctest_args)
    return command


def step_commands(
    step: Step,
    no_build: bool,
    jobs: int = 0,
    test_filter: str | None = None,
    repeat: int = 1,
    extra_ctest_args: Iterable[str] = (),
) -> list[list[str]]:
    command = step_command(
        step,
        no_build,
        jobs,
        test_filter,
        repeat,
        extra_ctest_args,
    )
    return [command]


def write_junit(path: Path, results: list[dict[str, object]], elapsed: float) -> None:
    suite = ET.Element(
        "testsuite",
        name="nuri-validate",
        tests=str(len(results)),
        failures=str(sum(result["status"] == "failure" for result in results)),
        errors=str(sum(result["status"] == "error" for result in results)),
        time=f"{elapsed:.3f}",
    )
    for result in results:
        case = ET.SubElement(
            suite,
            "testcase",
            name=str(result["name"]),
            time=f"{float(result['durationSeconds']):.3f}",
        )
        if result["status"] != "pass":
            element = "error" if result["status"] == "error" else "failure"
            child = ET.SubElement(case, element, message=f"exit {result['exitCode']}")
            child.text = str(result.get("log", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(suite)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def run_preset(args: argparse.Namespace) -> int:
    if not args.dry_run and args.timeout_seconds <= 0.0:
        raise ValueError("timeout-seconds must be greater than zero")
    if args.jobs < 0 or args.repeat < 1:
        raise ValueError("jobs must be non-negative and repeat must be positive")
    preset = affected_preset(args.affected) if args.preset == "affected" else args.preset
    steps = select_steps(preset, args.shard_index, args.shard_count)
    if not steps:
        print("selected shard contains no steps", file=sys.stderr)
        return 2
    static_errors = validate_repository()
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-{os.getpid()}"
    run_root = ARTIFACT_ROOT / run_id
    logs = run_root / "logs"
    logs.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    results: list[dict[str, object]] = []
    progress_path = Path(args.progress_jsonl) if args.progress_jsonl else None

    def progress(event: dict[str, object]) -> None:
        if progress_path is None:
            return
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        with progress_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
            stream.flush()

    if static_errors:
        results.append(
            {
                "name": "repository-contract",
                "status": "failure",
                "exitCode": 2,
                "durationSeconds": 0.0,
                "log": "\n".join(static_errors),
            }
        )
    if args.dry_run:
        for step in steps:
            print(
                json.dumps(
                    {
                        "name": step.name,
                        "commands": step_commands(
                            step,
                            args.no_build,
                            args.jobs,
                            args.test_filter,
                            args.repeat,
                            args.ctest_args,
                        ),
                    }
                )
            )
    elif not static_errors:
        for step in steps:
            commands = step_commands(
                step,
                args.no_build,
                args.jobs,
                args.test_filter,
                args.repeat,
                args.ctest_args,
            )
            step_started = time.monotonic()
            progress({"event": "step_started", "runId": run_id, "step": step.name, "commands": commands})
            deadline = step_started + args.timeout_seconds
            output_parts: list[str] = []
            raw_exit_code: int | None = 0
            exit_code = 0
            for command in commands:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raw_exit_code = None
                    exit_code = 4
                    output_parts.append(f"step timed out after {args.timeout_seconds} seconds\n")
                    break
                output_parts.append("$ " + subprocess.list2cmdline(command) + "\n")
                try:
                    completed = subprocess.run(
                        command,
                        cwd=ROOT,
                        text=True,
                        capture_output=True,
                        check=False,
                        timeout=remaining,
                    )
                    raw_exit_code = completed.returncode
                    output_parts.append(completed.stdout + completed.stderr)
                    if raw_exit_code != 0:
                        exit_code = 1
                        break
                except subprocess.TimeoutExpired as error:
                    raw_exit_code = None
                    exit_code = 4
                    output_parts.append(
                        (error.stdout or "")
                        + (error.stderr or "")
                        + f"\nstep timed out after {args.timeout_seconds} seconds\n"
                    )
                    break
            output = "".join(output_parts)
            (logs / f"{step.name}.log").write_text(output, encoding="utf-8")
            results.append(
                {
                    "name": step.name,
                    "status": "pass" if exit_code == 0 else "error" if exit_code == 4 else "failure",
                    "exitCode": exit_code,
                    "rawExitCode": raw_exit_code,
                    "durationSeconds": time.monotonic() - step_started,
                    "log": str(Path("logs") / f"{step.name}.log"),
                    "commands": commands,
                }
            )
            progress({"event": "step_finished", "runId": run_id, "step": step.name, "exitCode": exit_code})
            if exit_code != 0 and not args.keep_going:
                break
    elapsed = time.monotonic() - started
    if args.dry_run and not static_errors:
        status = "investigative"
    elif static_errors:
        status = "invalid"
    elif all(result["status"] == "pass" for result in results):
        status = "pass"
    elif any(result["status"] == "error" for result in results):
        status = "error"
    else:
        status = "fail"
    aggregate_exit = (
        0
        if status in {"pass", "investigative"}
        else 2 if status == "invalid"
        else 4 if status == "error"
        else 1
    )
    artifacts: list[dict[str, object]] = []
    for log_path in sorted(logs.glob("*.log")):
        artifacts.append(
            {
                "role": "validate.log",
                "path": str(log_path.relative_to(run_root)).replace("\\", "/"),
                "mediaType": "text/plain",
                "digest": f"sha256:{hashlib.sha256(log_path.read_bytes()).hexdigest()}",
                "status": "complete",
            }
        )
    report = {
        "schemaVersion": 2,
        "kind": "nuri.tool.result",
        "tool": "validate",
        "toolVersion": "2",
        "runId": run_id,
        "status": status,
        "exitCode": aggregate_exit,
        "authoritative": False,
        "durationMs": elapsed * 1000.0,
        "command": ["python", str(Path(__file__).relative_to(ROOT)).replace("\\", "/"), *sys.argv[1:]],
        "selection": {
            "requested": preset,
            "selected": len(steps),
            "attempted": len(results),
            "completed": len(results),
            "passed": sum(result["status"] == "pass" for result in results),
            "warned": 0,
            "failed": sum(result["status"] in {"failure", "error"} for result in results),
            "skipped": 0,
            "unavailable": 0,
            "notRun": max(0, len(steps) - len(results)),
        },
        "diagnostics": [
            {"code": "validate.repository.contract", "severity": "error", "message": error}
            for error in static_errors
        ],
        "artifacts": artifacts,
        "children": [
            {
                "id": str(result["name"]),
                "status": "pass" if result["status"] == "pass" else "error" if result["status"] == "error" else "fail",
                "exitCode": int(result["exitCode"]),
            }
            for result in results
        ],
        "payload": {
            "preset": preset,
            "shard": {"index": args.shard_index, "count": args.shard_count},
            "results": results,
        },
    }
    _json_dump(run_root / "run.json", report)
    if args.junit:
        write_junit(Path(args.junit), results, elapsed)
    print(run_root / "run.json")
    return aggregate_exit


def list_runs(args: argparse.Namespace) -> int:
    entries: list[tuple[Path, dict[str, object]]] = []
    for registry_root in run_registry_roots():
        if not registry_root.exists():
            continue
        for path in registry_root.iterdir():
            try:
                report = json.loads((path / "run.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if args.status and report.get("status") != args.status:
                continue
            entries.append((path, report))
    for path, report in sorted(entries, key=lambda entry: entry[0].stat().st_mtime, reverse=True):
        payload = report.get("payload", {})
        print(json.dumps({"runId": report.get("runId", path.name), "tool": report.get("tool"), "status": report.get("status"), "preset": payload.get("preset") if isinstance(payload, dict) else None}))
    return 0


def find_run(run_id: str) -> Path:
    if not _valid_identifier(run_id.replace("T", "t").replace("Z", "z")):
        raise ValueError("invalid run id")
    matches: list[Path] = []
    for registry_root in run_registry_roots():
        path = _safe_child(registry_root, registry_root / run_id)
        if (path / "run.json").is_file():
            matches.append(path)
    if not matches:
        raise FileNotFoundError(f"unknown run: {run_id}")
    if len(matches) > 1:
        raise ValueError(f"ambiguous run id across tool registries: {run_id}")
    return matches[0]


def inspect_run(args: argparse.Namespace) -> int:
    print((find_run(args.run_id) / "run.json").read_text(encoding="utf-8"), end="")
    return 0


def bundle_run(args: argparse.Namespace) -> int:
    run = find_run(args.run_id)
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w:gz" if output.suffix.lower() in {".gz", ".tgz"} or output.name.endswith(".tar.gz") else "w"
    entries: list[dict[str, object]] = []
    for path in sorted(run.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"run bundle contains a symlink: {path.relative_to(run)}")
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append(
            {
                "path": str(path.relative_to(run)).replace("\\", "/"),
                "size": path.stat().st_size,
                "digest": f"sha256:{digest}",
            }
        )
    manifest = json.dumps(
        {"kind": "nuri.run.bundle", "schemaVersion": 1, "runId": args.run_id, "entries": entries},
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    with tarfile.open(output, mode) as archive:
        archive.add(run, arcname=run.name, recursive=True)
        info = tarfile.TarInfo(f"{run.name}/bundle-manifest.json")
        info.size = len(manifest)
        info.mtime = int(time.time())
        archive.addfile(info, io.BytesIO(manifest))
    verify_bundle_file(output)
    print(output)
    return 0


def verify_bundle_file(path: Path) -> dict[str, object]:
    """Validate archive confinement, entry set, sizes, and SHA-256 digests."""
    archive_path = path.resolve()
    seen_names: set[str] = set()
    roots: set[str] = set()
    files: dict[str, tuple[int, str]] = {}
    manifest_bytes: bytes | None = None
    manifest_name = ""
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            normalized = member.name.replace("\\", "/").rstrip("/")
            parts = [part for part in normalized.split("/") if part]
            if not parts or normalized.startswith("/") or any(part in {".", ".."} for part in parts):
                raise ValueError(f"unsafe bundle member path: {member.name}")
            if member.name in seen_names:
                raise ValueError(f"duplicate bundle member: {member.name}")
            seen_names.add(member.name)
            roots.add(parts[0])
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise ValueError(f"unsupported bundle member type: {member.name}")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"unsupported bundle member: {member.name}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"bundle member cannot be read: {member.name}")
            payload = extracted.read()
            digest = hashlib.sha256(payload).hexdigest()
            files[normalized] = (len(payload), f"sha256:{digest}")
            if parts[-1] == "bundle-manifest.json":
                if manifest_bytes is not None:
                    raise ValueError("bundle contains multiple manifests")
                manifest_bytes = payload
                manifest_name = normalized
    if len(roots) != 1:
        raise ValueError("bundle must contain exactly one run root")
    if manifest_bytes is None:
        raise ValueError("bundle manifest is missing")
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid bundle manifest: {error}") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("kind") != "nuri.run.bundle"
        or manifest.get("schemaVersion") != 1
        or not isinstance(manifest.get("runId"), str)
        or not isinstance(manifest.get("entries"), list)
    ):
        raise ValueError("invalid bundle manifest contract")
    root = next(iter(roots))
    if manifest["runId"] != root:
        raise ValueError("bundle root does not match manifest runId")
    expected: dict[str, tuple[int, str]] = {}
    for index, entry in enumerate(manifest["entries"]):
        if not isinstance(entry, dict) or set(entry) != {"path", "size", "digest"}:
            raise ValueError(f"invalid bundle manifest entry {index}")
        relative = entry["path"]
        if (
            not isinstance(relative, str)
            or relative.startswith(("/", "\\"))
            or any(part in {"", ".", ".."} for part in relative.replace("\\", "/").split("/"))
            or relative in expected
            or not isinstance(entry["size"], int)
            or entry["size"] < 0
            or not isinstance(entry["digest"], str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", entry["digest"])
        ):
            raise ValueError(f"invalid bundle manifest entry {index}")
        expected[relative] = (entry["size"], entry["digest"])
    actual = {
        name[len(root) + 1 :]: value
        for name, value in files.items()
        if name != manifest_name
    }
    if expected != actual:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(
            name for name in set(expected) & set(actual) if expected[name] != actual[name]
        )
        raise ValueError(
            f"bundle checksum mismatch: missing={missing}, extra={extra}, changed={changed}"
        )
    return {
        "kind": "nuri.run.bundle_verification",
        "schemaVersion": 1,
        "status": "pass",
        "runId": root,
        "entries": len(actual),
        "archive": str(archive_path),
    }


def verify_bundle(args: argparse.Namespace) -> int:
    print(json.dumps(verify_bundle_file(Path(args.archive)), indent=2))
    return 0


def prune_runs(args: argparse.Namespace) -> int:
    if not args.dry_run and not args.confirm:
        print("prune requires --dry-run or --confirm", file=sys.stderr)
        return 2
    cutoff = time.time() - args.older_than_days * 86400
    candidates: list[Path] = []
    failures: list[Path] = []
    for registry_root in run_registry_roots():
        if not registry_root.exists():
            continue
        for path in registry_root.iterdir():
            try:
                report = json.loads((path / "run.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if report.get("status") != "pass":
                failures.append(path)
            if path.stat().st_mtime < cutoff:
                candidates.append(path)
    protected = set(sorted(failures, key=lambda path: path.stat().st_mtime, reverse=True)[: args.keep_failures])
    selected = [path for path in candidates if path not in protected]
    print(json.dumps({"dryRun": args.dry_run, "runs": [path.name for path in selected], "count": len(selected)}))
    if not args.dry_run:
        for path in selected:
            owner = next(
                root for root in run_registry_roots() if root == path.parent
            )
            shutil.rmtree(_safe_child(owner, path))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="validate manifests and tracked tool skills")
    check.set_defaults(func=lambda _args: _check_command())
    doctor = sub.add_parser("doctor", help="report build, profile, and runtime capability readiness")
    doctor.set_defaults(func=_doctor_command)
    plan = sub.add_parser("plan", help="print the selected validation plan as JSON")
    plan.add_argument("preset", choices=(*PRESETS, "affected"), default="contract", nargs="?")
    plan.add_argument("--affected", action="append", default=[])
    plan.set_defaults(func=_plan_command)
    run = sub.add_parser("run", help="execute a named validation preset")
    run.add_argument("preset", choices=(*PRESETS, "affected"), default="contract", nargs="?")
    run.add_argument("--affected", action="append", default=[])
    run.add_argument("--no-build", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--keep-going", action="store_true")
    run.add_argument("--junit")
    run.add_argument("--progress-jsonl")
    run.add_argument("--timeout-seconds", type=float, default=1800.0)
    run.add_argument("--jobs", type=int, default=0)
    run.add_argument("--filter", dest="test_filter")
    run.add_argument("--repeat", type=int, default=1)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--shard-count", type=int, default=1)
    run.set_defaults(func=run_preset, ctest_args=[])
    runs = sub.add_parser("runs", help="inspect and manage validation artifacts").add_subparsers(dest="runs_command", required=True)
    list_parser = runs.add_parser("list")
    list_parser.add_argument(
        "--status",
        choices=("pass", "warn", "fail", "invalid", "unavailable", "error", "investigative"),
    )
    list_parser.set_defaults(func=list_runs)
    inspect = runs.add_parser("inspect")
    inspect.add_argument("run_id")
    inspect.set_defaults(func=inspect_run)
    bundle = runs.add_parser("bundle")
    bundle.add_argument("run_id")
    bundle.add_argument("--out", required=True)
    bundle.set_defaults(func=bundle_run)
    verify = runs.add_parser("verify-bundle")
    verify.add_argument("archive")
    verify.set_defaults(func=verify_bundle)
    prune = runs.add_parser("prune")
    prune.add_argument("--older-than-days", type=int, required=True)
    prune.add_argument("--keep-failures", type=int, default=10)
    prune.add_argument("--dry-run", action="store_true")
    prune.add_argument("--confirm", action="store_true")
    prune.set_defaults(func=prune_runs)
    return result


def _check_command() -> int:
    errors = validate_repository()
    for error in errors:
        print(error, file=sys.stderr)
    if not errors:
        print(json.dumps({"kind": "nuri.validation.check", "schemaVersion": 2, "status": "pass"}))
    return 0 if not errors else 2


def _plan_command(args: argparse.Namespace) -> int:
    preset = affected_preset(args.affected) if args.preset == "affected" else args.preset
    print(json.dumps({"preset": preset, "steps": [step.__dict__ for step in PRESETS[preset]]}, indent=2))
    return 0


def _doctor_command(_args: argparse.Namespace) -> int:
    cmake = shutil.which("cmake")
    ctest = shutil.which("ctest")
    vcpkg_root = os.environ.get("VCPKG_ROOT")
    if not vcpkg_root:
        sibling_vcpkg = ROOT.parent / "vcpkg"
        if (sibling_vcpkg / "scripts" / "buildsystems" / "vcpkg.cmake").is_file():
            vcpkg_root = str(sibling_vcpkg)
    profile_documents: list[dict[str, object]] = []
    for path in sorted((ROOT / "tools" / "profiles").glob("*.json")):
        try:
            profile = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        profile_documents.append(
            {
                "id": profile.get("id"),
                "authoritative": bool(profile.get("authority", {}).get("authoritative", False)),
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            }
        )
    binaries: dict[str, str | None] = {
        "benchmark": None,
        "snapshot": None,
        "autotest": None,
    }
    registry_path = ROOT / "build" / "_registry" / "trees.json"
    if registry_path.is_file():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            registry = {}
        for record in registry.get("trees", {}).values():
            manifest_path = Path(str(record.get("path", ""))) / ".nuri-artifacts.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for tool, name in (
                ("benchmark", "nuri-bench"),
                ("snapshot", "nuri-snapshot"),
                ("autotest", "nuri-autotest"),
            ):
                artifact = manifest.get("targets", {}).get(name, {})
                path = Path(str(artifact.get("path", "")))
                if path.is_file():
                    binaries[tool] = str(path)
    approval_files = sorted((ROOT / "tools" / "baselines").rglob("approval.json"))
    governed_approvals = 0
    for approval_path in approval_files:
        try:
            approval = json.loads(approval_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        governed_approvals += int(_is_governed_approval(approval))
    ready = cmake is not None and ctest is not None and bool(vcpkg_root)
    report = {
        "kind": "nuri.validation.doctor",
        "schemaVersion": 2,
        "status": "pass" if ready and len(approval_files) == governed_approvals else "warn",
        "host": {"os": platform.system(), "release": platform.release(), "architecture": platform.machine()},
        "tools": {"python": sys.version.split()[0], "cmake": cmake, "ctest": ctest, "vcpkgRoot": vcpkg_root},
        "binaries": binaries,
        "profiles": profile_documents,
        "baselineGovernance": {
            "approvalFiles": len(approval_files),
            "governed": governed_approvals,
            "legacyOrInvalid": len(approval_files) - governed_approvals,
        },
        "capabilities": {
            "visibleWindow": "runtime-unknown",
            "hiddenWindow": "supported-requires-window-system",
            "headless": "unavailable",
            "offscreen": "unavailable",
            "adapterProbe": "requires renderer tool runtime",
        },
        "diagnostics": (
            [
                "true offscreen/headless execution is not implemented; use hidden only where a window system is available",
                f"{len(approval_files) - governed_approvals} existing approval files predate digest-bound governance and remain investigative",
            ]
            if ready
            else ["CMake, CTest, and a discovered VCPKG_ROOT are required before build-and-run presets"]
        ),
    }
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    try:
        raw_arguments = sys.argv[1:]
        ctest_arguments: list[str] = []
        if "--" in raw_arguments:
            separator = raw_arguments.index("--")
            ctest_arguments = raw_arguments[separator + 1 :]
            raw_arguments = raw_arguments[:separator]
        arguments = parser().parse_args(raw_arguments)
        if ctest_arguments:
            if arguments.command != "run":
                raise ValueError("raw CTest arguments after -- are only valid for run")
            arguments.ctest_args = ctest_arguments
        return int(arguments.func(arguments))
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
