from __future__ import annotations

import argparse
import csv
import fnmatch
import os
import re
import sys
from pathlib import Path


LANGUAGES = {
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".h": "cpp-header",
    ".hh": "cpp-header",
    ".hpp": "cpp-header",
    ".hxx": "cpp-header",
    ".m": "objective-c",
    ".mm": "objective-cpp",
    ".cs": "csharp",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".go": "go",
    ".rs": "rust",
    ".swift": "swift",
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".vue": "vue",
    ".svelte": "svelte",
    ".rb": "ruby",
    ".php": "php",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".fish": "shell",
    ".ps1": "powershell",
    ".psm1": "powershell",
    ".bat": "batch",
    ".cmd": "batch",
    ".lua": "lua",
    ".sql": "sql",
    ".glsl": "glsl",
    ".vert": "glsl",
    ".frag": "glsl",
    ".comp": "glsl",
    ".geom": "glsl",
    ".tesc": "glsl",
    ".tese": "glsl",
    ".hlsl": "hlsl",
    ".wgsl": "wgsl",
    ".cmake": "cmake",
}

SPECIAL_NAMES = {"CMakeLists.txt": "cmake", "Makefile": "make"}
HASH_COMMENTS = {"python", "ruby", "shell", "powershell", "cmake", "make"}
DASH_COMMENTS = {"lua", "sql"}
BRANCH_RE = re.compile(r"\b(?:if|else\s+if|switch|case|catch|match)\b")
ASSERT_RE = re.compile(
    r"\b(?:assert|static_assert|NURI_ASSERT|EXPECT_[A-Z_]+|ASSERT_[A-Z_]+)\b"
)
FIELDS = [
    "file",
    "language",
    "lines",
    "nonblank",
    "comment_lines",
    "branch_tokens",
    "assert_tokens",
    "bytes",
    "disposition",
    "slice",
    "owner",
    "action",
    "expected_lines",
    "current_lines",
    "delta_lines",
    "risk",
    "validation",
    "status",
    "evidence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory every source file in a compression scope into a TSV ledger."
    )
    parser.add_argument("paths", nargs="+", help="Repository-relative files or directories")
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Repository root")
    parser.add_argument("--exclude", action="append", default=[], help="Repository-relative glob")
    parser.add_argument("--extensions", help="Comma-separated extension override")
    parser.add_argument("--output", type=Path, help="TSV output path; defaults to stdout")
    parser.add_argument("--force", action="store_true", help="Replace an existing output file")
    return parser.parse_args()


def language_for(path: Path, extensions: set[str]) -> str | None:
    if path.name in SPECIAL_NAMES:
        return SPECIAL_NAMES[path.name]
    suffix = path.suffix.lower()
    if suffix not in extensions:
        return None
    return LANGUAGES.get(suffix, suffix.removeprefix("."))


def excluded(relative: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(relative, pattern) for pattern in patterns)


def collect_files(
    repo: Path, paths: list[str], extensions: set[str], patterns: list[str]
) -> list[tuple[Path, str]]:
    files: dict[str, Path] = {}
    for raw in paths:
        target = (repo / raw).resolve()
        try:
            target.relative_to(repo)
        except ValueError as error:
            raise SystemExit(f"scope escapes repository: {raw}") from error
        if not target.exists():
            raise SystemExit(f"scope does not exist: {raw}")
        candidates: list[Path] = []
        if target.is_file():
            candidates.append(target)
        else:
            for root, dirs, names in os.walk(target, followlinks=False):
                root_path = Path(root)
                dirs[:] = [
                    name
                    for name in dirs
                    if not excluded(
                        (root_path / name).relative_to(repo).as_posix() + "/", patterns
                    )
                ]
                candidates.extend(root_path / name for name in names)
        for candidate in candidates:
            relative = candidate.relative_to(repo).as_posix()
            if excluded(relative, patterns):
                continue
            language = language_for(candidate, extensions)
            if target.is_file() or language is not None:
                files[relative] = candidate
    return [(files[key], key) for key in sorted(files, key=str.casefold)]


def line_metrics(text: str, language: str) -> tuple[int, int, int, int, int]:
    lines = text.splitlines()
    blank = 0
    comments = 0
    branches = 0
    assertions = 0
    in_block = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            blank += 1
            continue
        comment_only = False
        if in_block:
            comment_only = True
            if "*/" in stripped:
                in_block = False
        elif stripped.startswith("/*"):
            comment_only = True
            in_block = "*/" not in stripped[2:]
        elif stripped.startswith("//"):
            comment_only = True
        elif language in HASH_COMMENTS and stripped.startswith("#"):
            comment_only = True
        elif language in DASH_COMMENTS and stripped.startswith("--"):
            comment_only = True
        if comment_only:
            comments += 1
            continue
        branches += len(BRANCH_RE.findall(stripped))
        assertions += len(ASSERT_RE.findall(stripped))
    return len(lines), len(lines) - blank, comments, branches, assertions


def row_for(path: Path, relative: str, language: str) -> dict[str, object]:
    data = path.read_bytes()
    text = data.decode("utf-8", errors="replace")
    lines, nonblank, comments, branches, assertions = line_metrics(text, language)
    return {
        "file": relative,
        "language": language,
        "lines": lines,
        "nonblank": nonblank,
        "comment_lines": comments,
        "branch_tokens": branches,
        "assert_tokens": assertions,
        "bytes": len(data),
        "disposition": "",
        "slice": "",
        "owner": "",
        "action": "",
        "expected_lines": "",
        "current_lines": "",
        "delta_lines": "",
        "risk": "",
        "validation": "",
        "status": "",
        "evidence": "",
    }


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    extensions = (
        {item.strip().lower() for item in args.extensions.split(",") if item.strip()}
        if args.extensions
        else set(LANGUAGES)
    )
    extensions = {item if item.startswith(".") else "." + item for item in extensions}
    patterns = [".git/", ".git/**", *args.exclude]
    entries = collect_files(repo, args.paths, extensions, patterns)
    rows = [
        row_for(path, relative, language_for(path, extensions) or "explicit")
        for path, relative in entries
    ]
    output = args.output
    if output:
        output = output if output.is_absolute() else repo / output
        if output.exists() and not args.force:
            raise SystemExit(f"output exists; pass --force to replace: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        stream = output.open("w", encoding="utf-8", newline="")
    else:
        stream = sys.stdout
    try:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if output:
            stream.close()
    total_lines = sum(int(row["lines"]) for row in rows)
    total_nonblank = sum(int(row["nonblank"]) for row in rows)
    destination = str(output) if output else "stdout"
    print(
        f"mapped {len(rows)} files, {total_lines} lines, {total_nonblank} nonblank -> {destination}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
