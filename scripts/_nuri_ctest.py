#!/usr/bin/env python3
"""Windows CTest dispatcher that preserves argv without cmd.exe re-parsing."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROFILES = {"tests", "bench-tests", "snapshot-tests", "autotest-tests"}


def _command_for(mode: str, profile: str, args: list[str]) -> list[str]:
    if mode not in {"debug", "release"} or profile not in PROFILES:
        raise ValueError("invalid mode or test profile")
    args = list(args)
    repo_root = Path(__file__).resolve().parents[1]
    suffix = mode if profile == "tests" else f"{mode}-{profile}"
    command = [
        "ctest",
        "--test-dir",
        str(repo_root / "build" / suffix),
        "--output-on-failure",
    ]
    if "-j" not in args and "--parallel" not in args:
        command.extend(["-j", str(os.cpu_count() or 4)])
    command.extend(args)
    return command


def ctest_command(argv: list[str]) -> list[str]:
    args = list(argv)
    mode = "debug"
    profile = "tests"
    if args and args[0].lower() in {"debug", "release"}:
        mode = args.pop(0).lower()
    if args and args[0].lower() in PROFILES:
        profile = args.pop(0).lower()
    elif args and not args[0].startswith("-"):
        raise ValueError(
            "usage: run_tests.bat [debug|release] "
            "[bench-tests|snapshot-tests|autotest-tests] [ctest args...]"
        )

    return _command_for(mode, profile, args)


def ctest_command_from_environment(environment: dict[str, str]) -> list[str]:
    try:
        count = int(environment.get("NURI_CTEST_ARGC", "0"))
    except ValueError as error:
        raise ValueError("invalid NURI_CTEST_ARGC") from error
    if count < 0 or count > 1024:
        raise ValueError("invalid NURI_CTEST_ARGC")
    args: list[str] = []
    for index in range(count):
        key = f"NURI_CTEST_ARG_{index}"
        if key not in environment:
            raise ValueError(f"missing {key}")
        args.append(environment[key])
    return _command_for(
        environment.get("NURI_CTEST_MODE", "debug").lower(),
        environment.get("NURI_CTEST_PROFILE", "tests").lower(),
        args,
    )


def main(argv: list[str]) -> int:
    try:
        command = (
            ctest_command_from_environment(dict(os.environ))
            if argv == ["--from-env"]
            else ctest_command(argv)
        )
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    try:
        return subprocess.run(command, check=False).returncode
    except OSError as error:
        print(f"failed to start CTest: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
