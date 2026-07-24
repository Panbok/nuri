#!/usr/bin/env python3
"""Canonical Nuri build-tree identity, locking, build, run, and retention driver."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "scripts" / "build_variants.json"
BUILD_ROOT = ROOT / "build"
TREE_ROOT = BUILD_ROOT / "_trees"
LOCK_ROOT = BUILD_ROOT / "_locks"
REGISTRY_ROOT = BUILD_ROOT / "_registry"
DEPENDENCY_ROOT = BUILD_ROOT / "_deps" / "vcpkg-installed"
PREPARED_ROOT = BUILD_ROOT / "_deps" / "nvrhi-source"
TRASH_ROOT = BUILD_ROOT / "_trash"
HEALTH_ARCHIVE_ROOT = ROOT / ".scratch" / "build-health"
IDENTITY_FILE = ".nuri-build-identity.json"
ARTIFACT_FILE = ".nuri-artifacts.json"
RECEIPT_FILE = ".nuri-build-receipts.json"
NVRHI_INPUT_FILE = ".nuri-nvrhi-input.json"
REGISTRY_FILE = REGISTRY_ROOT / "trees.json"
LOCK_ORDER = ("registry", "host", "dependency", "prepared-source", "tree")


class NuriBuildError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NuriBuildError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise NuriBuildError(f"JSON root must be an object: {path}")
    return value


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return "sha256:" + hasher.hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_path(path: Path) -> str:
    resolved = str(path.resolve())
    return os.path.normcase(resolved) if os.name == "nt" else resolved


def platform_key() -> str:
    if os.name == "nt":
        return "windows"
    if sys.platform.startswith("linux"):
        return "linux"
    raise NuriBuildError(f"unsupported host platform: {sys.platform}")


def command_output(
    command: Sequence[str],
    *,
    environment: dict[str, str] | None = None,
    cwd: Path = ROOT,
) -> str:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
    except OSError as error:
        raise NuriBuildError(f"failed to start {command[0]}: {error}") from error
    output = (completed.stdout + "\n" + completed.stderr).strip()
    if completed.returncode != 0:
        raise NuriBuildError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{output}"
        )
    return output


def executable_identity(
    name: str, *, environment: dict[str, str], version_args: Sequence[str] = ("--version",)
) -> dict[str, object]:
    path_text = shutil.which(name, path=environment.get("PATH"))
    if not path_text:
        raise NuriBuildError(f"required executable not found: {name}")
    path = Path(path_text).resolve()
    stat = path.stat()
    version = ""
    if version_args:
        try:
            version = command_output(
                [str(path), *version_args], environment=environment
            ).splitlines()[0]
        except NuriBuildError:
            version = "unavailable"
    return {
        "path": canonical_path(path),
        "size": stat.st_size,
        "mtimeNs": stat.st_mtime_ns,
        "version": version,
    }


def _visual_studio_environment(base: dict[str, str]) -> dict[str, str]:
    if os.name != "nt":
        return base
    initialized = all(
        base.get(name)
        for name in (
            "VSINSTALLDIR",
            "VCToolsInstallDir",
            "WindowsSdkDir",
            "LIB",
            "INCLUDE",
        )
    )
    if initialized and shutil.which("link.exe", path=base.get("PATH")):
        result = dict(base)
        if result.get("VSINSTALLDIR"):
            result["VCPKG_VISUAL_STUDIO_PATH"] = result["VSINSTALLDIR"].rstrip(
                "\\/"
            )
        return result
    candidates: list[Path] = []
    vswhere = (
        Path(os.environ.get("ProgramFiles(x86)", ""))
        / "Microsoft Visual Studio"
        / "Installer"
        / "vswhere.exe"
    )
    if vswhere.is_file():
        try:
            install = command_output(
                [
                    str(vswhere),
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ],
                environment=base,
            ).splitlines()
            if install:
                candidates.append(
                    Path(install[-1]) / "Common7" / "Tools" / "VsDevCmd.bat"
                )
        except NuriBuildError:
            pass
    for program_root in (
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramFiles(x86)"),
    ):
        if not program_root:
            continue
        for version in ("18", "2022", "2019"):
            for edition in ("BuildTools", "Community", "Professional", "Enterprise"):
                candidates.append(
                    Path(program_root)
                    / "Microsoft Visual Studio"
                    / version
                    / edition
                    / "Common7"
                    / "Tools"
                    / "VsDevCmd.bat"
                )
    script = next((candidate for candidate in candidates if candidate.is_file()), None)
    if script is None:
        raise NuriBuildError(
            "Visual Studio C++ Build Tools environment was not found"
        )
    short_path_length = ctypes.windll.kernel32.GetShortPathNameW(
        str(script), None, 0
    )
    if not short_path_length:
        raise NuriBuildError(
            f"cannot resolve Visual Studio environment path: {script}"
        )
    short_path_buffer = ctypes.create_unicode_buffer(short_path_length)
    ctypes.windll.kernel32.GetShortPathNameW(
        str(script), short_path_buffer, short_path_length
    )
    command = (
        f"call {short_path_buffer.value} -arch=x64 -host_arch=x64 >nul && set"
    )
    completed = subprocess.run(
        ["cmd.exe", "/d", "/u", "/c", command],
        env=base,
        capture_output=True,
        check=False,
    )
    stdout = completed.stdout.decode("utf-16-le", errors="replace")
    stderr = completed.stderr.decode("utf-16-le", errors="replace")
    if completed.returncode != 0:
        raise NuriBuildError(
            f"Visual Studio environment setup failed: {stderr.strip()}"
        )
    result = dict(base)
    for line in stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            for existing in tuple(result):
                if existing.casefold() == key.casefold() and existing != key:
                    del result[existing]
            result[key] = value
    missing = [
        name
        for name in (
            "VSINSTALLDIR",
            "VCToolsInstallDir",
            "WindowsSdkDir",
            "LIB",
            "INCLUDE",
        )
        if not result.get(name)
    ]
    if missing:
        raise NuriBuildError(
            "Visual Studio environment setup omitted: " + ", ".join(missing)
        )
    if result.get("VSINSTALLDIR"):
        result["VCPKG_VISUAL_STUDIO_PATH"] = result["VSINSTALLDIR"].rstrip("\\/")
    return result


def build_environment() -> dict[str, str]:
    base = dict(os.environ)
    explicit_vcpkg = base.get("VCPKG_ROOT")
    environment = _visual_studio_environment(base)
    sibling = ROOT.parent / "vcpkg"
    if explicit_vcpkg:
        environment["VCPKG_ROOT"] = explicit_vcpkg
    elif sibling.joinpath("scripts", "buildsystems", "vcpkg.cmake").is_file():
        environment["VCPKG_ROOT"] = str(sibling)
    return environment


class FileLock:
    """One-byte OS-backed shared/exclusive file lock with diagnostic metadata."""

    def __init__(
        self,
        path: Path,
        *,
        shared: bool = False,
        timeout: float = 600.0,
        identity: str = "",
        command: str = "",
    ) -> None:
        self.path = path
        self.shared = shared
        self.timeout = timeout
        self.identity = identity
        self.command = command
        self._stream: Any = None

    def _try_lock(self) -> bool:
        if os.name == "nt":
            import msvcrt

            handle = msvcrt.get_osfhandle(self._stream.fileno())
            flags = 0x00000001
            if not self.shared:
                flags |= 0x00000002

            class Overlapped(ctypes.Structure):
                _fields_ = [
                    ("Internal", ctypes.c_size_t),
                    ("InternalHigh", ctypes.c_size_t),
                    ("Offset", ctypes.c_ulong),
                    ("OffsetHigh", ctypes.c_ulong),
                    ("hEvent", ctypes.c_void_p),
                ]

            self._overlapped = Overlapped()
            result = ctypes.windll.kernel32.LockFileEx(
                ctypes.c_void_p(handle),
                flags,
                0,
                1,
                0,
                ctypes.byref(self._overlapped),
            )
            return bool(result)
        import fcntl

        operation = fcntl.LOCK_SH if self.shared else fcntl.LOCK_EX
        try:
            fcntl.flock(self._stream.fileno(), operation | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False

    def __enter__(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a+b")
        started = time.monotonic()
        last_notice = started
        while not self._try_lock():
            now = time.monotonic()
            if now - started >= self.timeout:
                self._stream.close()
                raise NuriBuildError(
                    f"timed out waiting for {'shared' if self.shared else 'exclusive'} "
                    f"lock: {self.path}"
                )
            if now - last_notice >= 10.0:
                print(f"waiting for build lease: {self.path}", file=sys.stderr)
                last_notice = now
            time.sleep(0.1)
        metadata = {
            "schemaVersion": 1,
            "host": socket.gethostname(),
            "bootHint": platform.uname().version,
            "pid": os.getpid(),
            "processStartHint": time.time() - time.monotonic(),
            "acquiredUtc": utc_now(),
            "mode": "shared" if self.shared else "exclusive",
            "identity": self.identity,
            "command": self.command,
        }
        atomic_json(
            self.path.with_name(
                f"{self.path.name}.{os.getpid()}.lease.json"
            ),
            metadata,
        )
        return self

    def __exit__(self, *_args: object) -> None:
        metadata = self.path.with_name(f"{self.path.name}.{os.getpid()}.lease.json")
        try:
            metadata.unlink(missing_ok=True)
        except OSError:
            pass
        if self._stream is None:
            return
        if os.name == "nt":
            import msvcrt

            handle = msvcrt.get_osfhandle(self._stream.fileno())
            ctypes.windll.kernel32.UnlockFileEx(
                ctypes.c_void_p(handle),
                0,
                1,
                0,
                ctypes.byref(self._overlapped),
            )
        else:
            import fcntl

            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        self._stream.close()


@dataclass(frozen=True)
class Request:
    variant: str
    capability: str
    target: str
    configuration: str
    labels: str | None = None


@dataclass(frozen=True)
class BuildContext:
    request: Request
    policy: dict[str, Any]
    environment: dict[str, str]
    tree_identity: dict[str, Any]
    compile_compatibility: dict[str, Any]
    dependency_identity: dict[str, Any]
    tree_digest: str
    compile_digest: str
    dependency_digest: str
    tree: Path
    dependency_root: Path
    triplet: str
    vcpkg_root: Path
    tools: dict[str, Any]


def load_policy() -> dict[str, Any]:
    policy = load_json(POLICY_PATH)
    if policy.get("schemaVersion") != 1:
        raise NuriBuildError("unsupported build policy schema")
    return policy


def profile_record(policy: dict[str, Any], profile: str) -> tuple[str, dict[str, Any]]:
    profiles = policy["legacyProfiles"]
    if profile in profiles:
        return profile, profiles[profile]
    for name, record in profiles.items():
        if profile in record.get("aliases", []):
            return name, record
    raise NuriBuildError(f"unknown legacy profile: {profile}")


def legacy_request(
    mode: str, profile: str, modifiers: Iterable[str], policy: dict[str, Any]
) -> Request:
    mode = mode.lower()
    canonical_profile, record = profile_record(policy, profile.lower())
    values = [value.lower() for value in modifiers if value]
    unknown = set(values) - {"cpu", "off", "devchecks"}
    if unknown or ("cpu" in values and "off" in values):
        raise NuriBuildError(
            "legacy modifiers are [cpu|off] [devchecks]; received "
            + ", ".join(sorted(unknown or set(values)))
        )
    if mode == "debug":
        variant = "debug-dev-no-tracy" if "off" in values else "debug-dev"
    elif mode == "release":
        checked = record["class"] == "checked" or "devchecks" in values
        if "cpu" in values:
            variant = "release-tracy-checked" if checked else "release-tracy"
        else:
            variant = "release-checked" if checked else "release-fast"
    else:
        raise NuriBuildError(f"unknown build mode: {mode}")
    if os.environ.get("NURI_WITH_FSR31", "OFF").upper() == "ON":
        variant = f"{variant}-fsr31"
    variants = policy["variants"]
    if variant not in variants:
        if not variant.endswith("-fsr31"):
            raise NuriBuildError(f"build policy has no variant {variant}")
        base = variant.removesuffix("-fsr31")
        if base not in variants:
            raise NuriBuildError(f"build policy has no variant {base}")
        variants[variant] = json.loads(json.dumps(variants[base]))
        variants[variant]["cmake"]["NURI_WITH_FSR31"] = True
        variants[variant]["derivedFrom"] = base
    return Request(
        variant=variant,
        capability=record["capability"],
        target=record["target"],
        configuration=variants[variant]["configuration"],
        labels=record.get("labels"),
    )


def explicit_request(
    policy: dict[str, Any], variant: str, capability: str, target: str
) -> Request:
    if variant not in policy["variants"]:
        raise NuriBuildError(f"unknown variant: {variant}")
    if capability not in policy["capabilities"]:
        raise NuriBuildError(f"unknown capability: {capability}")
    if target not in policy["capabilities"][capability]["targets"]:
        raise NuriBuildError(
            f"target {target} is not in graph capability {capability}"
        )
    return Request(
        variant=variant,
        capability=capability,
        target=target,
        configuration=policy["variants"][variant]["configuration"],
    )


def _content_record(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "sha256": file_digest(path) if path.is_file() else "missing",
    }


def _vcpkg_root(environment: dict[str, str]) -> Path:
    value = environment.get("VCPKG_ROOT")
    if not value:
        raise NuriBuildError("VCPKG_ROOT is not set and no sibling vcpkg was found")
    root = Path(value).resolve()
    if not root.joinpath("scripts", "buildsystems", "vcpkg.cmake").is_file():
        raise NuriBuildError(f"invalid VCPKG_ROOT: {root}")
    return root


def _triplet_path(triplet: str) -> Path:
    path = ROOT / "cmake" / "triplets" / f"{triplet}.cmake"
    if not path.is_file():
        raise NuriBuildError(f"overlay triplet is missing: {path}")
    return path


def create_context(request: Request, policy: dict[str, Any]) -> BuildContext:
    environment = build_environment()
    host = platform_key()
    variant = policy["variants"][request.variant]
    if policy["generator"] != "Ninja":
        raise NuriBuildError("only the checked-in Ninja contract is supported")
    compiler_name = variant["compilerFrontend"]
    c_compiler_name = (
        "clang-cl"
        if host == "windows" and compiler_name == "clang-cl"
        else "clang"
    )
    tools = {
        "cmake": executable_identity("cmake", environment=environment),
        "ninja": executable_identity("ninja", environment=environment),
        "compiler": executable_identity(compiler_name, environment=environment),
        "cCompiler": executable_identity(
            c_compiler_name, environment=environment
        ),
    }
    linker_name = variant["linker"]
    linker_path = shutil.which(linker_name, path=environment.get("PATH"))
    if not linker_path and host == "linux" and linker_name == "link.exe":
        linker_name = "ld"
        linker_path = shutil.which(linker_name, path=environment.get("PATH"))
    if not linker_path:
        raise NuriBuildError(f"required linker not found: {variant['linker']}")
    linker_stat = Path(linker_path).stat()
    tools["linker"] = {
        "path": canonical_path(Path(linker_path)),
        "size": linker_stat.st_size,
        "mtimeNs": linker_stat.st_mtime_ns,
        "version": "selected-by-toolchain",
    }
    vcpkg_root = _vcpkg_root(environment)
    try:
        vcpkg_revision = command_output(
            ["git", "rev-parse", "HEAD"], cwd=vcpkg_root
        )
        vcpkg_dirty = bool(
            command_output(
                ["git", "status", "--porcelain=v1", "--untracked-files=no"],
                cwd=vcpkg_root,
            )
        )
    except NuriBuildError:
        vcpkg_revision = "unknown"
        vcpkg_dirty = True
    triplet = variant["triplet"][host]
    triplet_path = _triplet_path(triplet)
    features = sorted(policy["capabilities"][request.capability]["features"])
    dependency_identity = {
        "schemaVersion": 1,
        "vcpkgRoot": canonical_path(vcpkg_root),
        "vcpkgRevision": vcpkg_revision,
        "vcpkgDirty": vcpkg_dirty,
        "vcpkgExecutable": _content_record(
            vcpkg_root / ("vcpkg.exe" if os.name == "nt" else "vcpkg")
        ),
        "manifest": _content_record(ROOT / "vcpkg.json"),
        "configuration": _content_record(ROOT / "vcpkg-configuration.json"),
        "triplet": triplet,
        "tripletContent": _content_record(triplet_path),
        "features": features,
        "targetPlatform": host,
        "targetArchitecture": "x64",
        "compilerAbi": tools["compiler"],
        "portCompiler": (
            executable_identity("cl.exe", environment=environment, version_args=())
            if host == "windows"
            else tools["compiler"]
        ),
        "sdk": {
            "visualStudioVersion": environment.get("VisualStudioVersion", ""),
            "vcToolsVersion": environment.get("VCToolsVersion", ""),
            "windowsSdkVersion": environment.get("WindowsSDKVersion", ""),
            "vcpkgVisualStudioPath": environment.get(
                "VCPKG_VISUAL_STUDIO_PATH", ""
            ),
        },
    }
    dependency_digest = digest(dependency_identity)
    compiler_cache = os.environ.get("NURI_COMPILER_CACHE", "off")
    if compiler_cache != "off":
        cache_path = shutil.which(compiler_cache, path=environment.get("PATH"))
        if not cache_path:
            raise NuriBuildError(
                f"NURI_COMPILER_CACHE executable not found: {compiler_cache}"
            )
        compiler_cache = canonical_path(Path(cache_path))
        environment.setdefault(
            "SCCACHE_CACHE_SIZE",
            str(policy["retention"]["compilerCacheMaxBytes"]),
        )
    compile_compatibility = {
        "schemaVersion": 1,
        "variant": request.variant,
        "configuration": request.configuration,
        "compilePolicy": variant["cmake"],
        "compiler": tools["compiler"],
        "linker": tools["linker"],
        "cxxStandard": 20,
        "targetPlatform": host,
        "targetArchitecture": "x64",
        "dependencyIdentityDigest": dependency_digest,
        "compilerCache": compiler_cache,
    }
    compile_digest = digest(compile_compatibility)
    tree_identity = {
        "schemaVersion": 1,
        "policyEpoch": policy["policyEpoch"],
        "layoutEpoch": policy["layoutEpoch"],
        "owner": canonical_path(ROOT),
        "variant": request.variant,
        "graphCapability": request.capability,
        "configurePreset": variant["preset"],
        "generator": policy["generator"],
        "tools": tools,
        "configuration": request.configuration,
        "cacheVariables": {
            **variant["cmake"],
            **policy["capabilities"][request.capability]["cmake"],
        },
        "dependencyIdentityDigest": dependency_digest,
        "compileCompatibilityDigest": compile_digest,
    }
    tree_digest = digest(tree_identity)
    short = tree_digest.split(":", 1)[1][:16]
    friendly = re.sub(
        r"[^a-z0-9-]+",
        "-",
        f"{host}-{request.variant}-{request.capability}".lower(),
    ).strip("-")
    tree = TREE_ROOT / f"{friendly}-{short}"
    dependency_root = DEPENDENCY_ROOT / dependency_digest.split(":", 1)[1]
    return BuildContext(
        request=request,
        policy=policy,
        environment=environment,
        tree_identity=tree_identity,
        compile_compatibility=compile_compatibility,
        dependency_identity=dependency_identity,
        tree_digest=tree_digest,
        compile_digest=compile_digest,
        dependency_digest=dependency_digest,
        tree=tree,
        dependency_root=dependency_root,
        triplet=triplet,
        vcpkg_root=vcpkg_root,
        tools=tools,
    )


def source_provenance() -> dict[str, object]:
    revision = "unknown"
    dirty = True
    try:
        revision = command_output(["git", "rev-parse", "HEAD"])
        dirty = bool(command_output(["git", "status", "--porcelain=v1"]))
    except NuriBuildError:
        pass
    return {
        "revision": revision,
        "dirty": dirty,
        "capturedUtc": utc_now(),
    }


def _nvrhi_input() -> tuple[str, dict[str, object]]:
    source = ROOT / "external" / "nvrhi"
    if not source.joinpath("CMakeLists.txt").is_file():
        raise NuriBuildError(
            "NVRHI submodule is missing; run git submodule update --init --recursive"
        )
    revision = command_output(["git", "rev-parse", "HEAD"], cwd=source)
    status = command_output(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=source,
    )
    record: dict[str, object] = {
        "schemaVersion": 1,
        "revision": revision,
        "status": status,
        "patch": _content_record(
            ROOT / "patches" / "nvrhi" / "framebuffer-resolve.patch"
        ),
        "preparation": "copy-git-apply-ceiling-v4",
    }
    if status:
        record["workingDiff"] = command_output(
            ["git", "diff", "--binary", "--no-ext-diff"], cwd=source
        )
        record["indexDiff"] = command_output(
            ["git", "diff", "--cached", "--binary", "--no-ext-diff"], cwd=source
        )
        untracked = command_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=source
        )
        untracked_records: list[dict[str, object]] = []
        for relative in filter(None, untracked.split("\0")):
            path = source / relative
            if path.is_file():
                untracked_records.append(
                    {"path": relative, "sha256": file_digest(path)}
                )
        record["untracked"] = untracked_records
    return digest(record), record


def prepare_nvrhi(input_digest: str, record: dict[str, object]) -> Path:
    key = input_digest.split(":", 1)[1]
    destination = PREPARED_ROOT / key
    owner = destination / ".nuri-prepared-source.json"
    if owner.is_file():
        existing = load_json(owner)
        if existing.get("inputDigest") != input_digest:
            raise NuriBuildError(f"prepared-source key collision: {destination}")
        return destination
    PREPARED_ROOT.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{key[:12]}.", dir=str(PREPARED_ROOT))
    )
    source = ROOT / "external" / "nvrhi"
    try:
        for child in source.iterdir():
            if child.name == ".git":
                continue
            target = staging / child.name
            if child.is_dir():
                shutil.copytree(child, target, symlinks=True)
            else:
                shutil.copy2(child, target)
        patch = ROOT / "patches" / "nvrhi" / "framebuffer-resolve.patch"
        if patch.is_file():
            apply_environment = dict(os.environ)
            apply_environment["GIT_CEILING_DIRECTORIES"] = str(ROOT)
            command_output(
                [
                    "git",
                    "apply",
                    "--check",
                    "--ignore-space-change",
                    "--whitespace=nowarn",
                    str(patch),
                ],
                cwd=staging,
                environment=apply_environment,
            )
            command_output(
                [
                    "git",
                    "apply",
                    "--ignore-space-change",
                    "--whitespace=nowarn",
                    str(patch),
                ],
                cwd=staging,
                environment=apply_environment,
            )
        atomic_json(
            staging / ".nuri-prepared-source.json",
            {
                "schemaVersion": 1,
                "kind": "nuri.prepared_source",
                "inputDigest": input_digest,
                "input": record,
                "publishedUtc": utc_now(),
            },
        )
        try:
            os.replace(staging, destination)
        except FileExistsError:
            shutil.rmtree(staging)
        return destination
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def _registry() -> dict[str, Any]:
    if not REGISTRY_FILE.is_file():
        return {"schemaVersion": 1, "trees": {}}
    value = load_json(REGISTRY_FILE)
    if value.get("schemaVersion") != 1 or not isinstance(value.get("trees"), dict):
        raise NuriBuildError(f"invalid build registry: {REGISTRY_FILE}")
    return value


def reserve_tree(context: BuildContext) -> None:
    command = " ".join(sys.argv)
    with FileLock(
        LOCK_ROOT / "00-registry.lock",
        timeout=60.0,
        identity=context.tree_digest,
        command=command,
    ):
        registry = _registry()
        trees = registry["trees"]
        current = trees.get(context.tree_digest)
        tree_text = canonical_path(context.tree)
        if current and current.get("path") != tree_text:
            raise NuriBuildError(
                f"tree digest collision: {context.tree_digest} maps to "
                f"{current.get('path')} and {tree_text}"
            )
        if context.tree.exists():
            identity_path = context.tree / IDENTITY_FILE
            if identity_path.is_file():
                existing = load_json(identity_path)
                if existing.get("treeIdentityDigest") != context.tree_digest:
                    raise NuriBuildError(
                        f"refusing identity mismatch in existing tree: {context.tree}"
                    )
        trees[context.tree_digest] = {
            "path": tree_text,
            "state": current.get("state", "reserved") if current else "reserved",
            "owner": canonical_path(ROOT),
            "variant": context.request.variant,
            "capability": context.request.capability,
            "updatedUtc": utc_now(),
        }
        atomic_json(REGISTRY_FILE, registry)


def update_registry(context: BuildContext, state: str, **extra: object) -> None:
    with FileLock(
        LOCK_ROOT / "00-registry.lock",
        timeout=60.0,
        identity=context.tree_digest,
        command=" ".join(sys.argv),
    ):
        registry = _registry()
        record = registry["trees"].setdefault(context.tree_digest, {})
        if state in {"configuring", "ready"}:
            record.pop("lastError", None)
        if state == "ready":
            record.pop("reason", None)
        record.update(
            {
                "path": canonical_path(context.tree),
                "state": state,
                "owner": canonical_path(ROOT),
                "variant": context.request.variant,
                "capability": context.request.capability,
                "updatedUtc": utc_now(),
                **extra,
            }
        )
        atomic_json(REGISTRY_FILE, registry)


def _identity_document(context: BuildContext) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "kind": "nuri.build_tree_identity",
        "treeIdentityDigest": context.tree_digest,
        "compileCompatibilityDigest": context.compile_digest,
        "dependencyIdentityDigest": context.dependency_digest,
        "treeIdentity": context.tree_identity,
        "compileCompatibility": context.compile_compatibility,
        "dependencyIdentity": context.dependency_identity,
        "readyUtc": utc_now(),
    }


def _cmake_bool(value: object) -> str:
    if isinstance(value, bool):
        return "ON" if value else "OFF"
    return str(value)


def configure_command(
    context: BuildContext, prepared_source: Path
) -> list[str]:
    variant = context.policy["variants"][context.request.variant]
    capability = context.policy["capabilities"][context.request.capability]
    compiler = context.tools["compiler"]["path"]
    c_compiler = context.tools["cCompiler"]["path"]
    command = [
        str(context.tools["cmake"]["path"]),
        "--preset",
        variant["preset"],
        "-S",
        str(ROOT),
        "-B",
        str(context.tree),
        f"-DCMAKE_CXX_COMPILER={compiler}",
        f"-DCMAKE_C_COMPILER={c_compiler}",
        f"-DCMAKE_MAKE_PROGRAM={context.tools['ninja']['path']}",
        f"-DCMAKE_TOOLCHAIN_FILE={context.vcpkg_root / 'scripts' / 'buildsystems' / 'vcpkg.cmake'}",
        f"-DVCPKG_EXECUTABLE={context.vcpkg_root / ('vcpkg.exe' if os.name == 'nt' else 'vcpkg')}",
        f"-DVCPKG_INSTALLED_DIR={context.dependency_root}",
        f"-DVCPKG_TARGET_TRIPLET={context.triplet}",
        f"-DVCPKG_OVERLAY_TRIPLETS={ROOT / 'cmake' / 'triplets'}",
        "-DVCPKG_APPLOCAL_DEPS=OFF",
        f"-DVCPKG_MANIFEST_FEATURES={';'.join(capability['features'])}",
        f"-DNURI_GRAPH_CAPABILITY={context.request.capability}",
        f"-DNURI_TREE_IDENTITY_DIGEST={context.tree_digest}",
        f"-DNURI_COMPILE_COMPATIBILITY_DIGEST={context.compile_digest}",
        f"-DNURI_DEPENDENCY_IDENTITY_DIGEST={context.dependency_digest}",
        f"-DNURI_NVRHI_SOURCE_DIR={prepared_source}",
    ]
    for key, value in sorted(
        {**variant["cmake"], **capability["cmake"]}.items()
    ):
        command.append(f"-D{key}={_cmake_bool(value)}")
    compiler_cache = context.compile_compatibility["compilerCache"]
    if compiler_cache != "off":
        command.append(f"-DCMAKE_CXX_COMPILER_LAUNCHER={compiler_cache}")
        command.append("-DNURI_NORMALIZE_COMPILER_PATHS=ON")
    if os.name == "nt" and variant["linker"]:
        command.append(f"-DCMAKE_LINKER={context.tools['linker']['path']}")
    return command


def _dependency_owner(context: BuildContext) -> None:
    owner = context.dependency_root / ".nuri-dependency-identity.json"
    if owner.is_file():
        current = load_json(owner)
        if current.get("dependencyIdentityDigest") != context.dependency_digest:
            raise NuriBuildError(
                f"shared dependency root identity mismatch: {context.dependency_root}"
            )
        return
    context.dependency_root.mkdir(parents=True, exist_ok=True)
    atomic_json(
        owner,
        {
            "schemaVersion": 1,
            "kind": "nuri.dependency_identity",
            "dependencyIdentityDigest": context.dependency_digest,
            "dependencyIdentity": context.dependency_identity,
            "createdUtc": utc_now(),
        },
    )


def _needs_configure(
    context: BuildContext, prepared_digest: str
) -> tuple[bool, str]:
    identity_path = context.tree / IDENTITY_FILE
    if not identity_path.is_file():
        return True, "tree has no ready identity"
    identity = load_json(identity_path)
    if identity.get("treeIdentityDigest") != context.tree_digest:
        raise NuriBuildError(
            f"tree identity mismatch; refusing in-place reconfigure: {context.tree}"
        )
    if not context.tree.joinpath("CMakeCache.txt").is_file():
        return True, "CMakeCache.txt is missing"
    if not context.tree.joinpath("build.ninja").is_file():
        return True, "build.ninja is missing"
    input_path = context.tree / NVRHI_INPUT_FILE
    if not input_path.is_file():
        return True, "prepared NVRHI input record is missing"
    if load_json(input_path).get("inputDigest") != prepared_digest:
        return True, "effective NVRHI source input changed"
    return False, "ready identity matches"


def _generated_manifest(context: BuildContext) -> Path:
    return context.tree / f".nuri-artifacts-{context.request.configuration}.json"


def _publish_manifest(
    context: BuildContext, receipts: dict[str, Any]
) -> dict[str, Any]:
    generated_path = _generated_manifest(context)
    if not generated_path.is_file():
        raise NuriBuildError(
            f"CMake artifact manifest was not generated: {generated_path}"
        )
    generated = load_json(generated_path)
    if generated.get("treeIdentityDigest") != context.tree_digest:
        raise NuriBuildError("generated artifact manifest identity mismatch")
    manifest = {
        **generated,
        "kind": "nuri.artifact_manifest",
        "publishedUtc": utc_now(),
        "receipts": receipts,
        "dependencyInstalledRoot": canonical_path(context.dependency_root),
    }
    atomic_json(context.tree / ARTIFACT_FILE, manifest)
    return manifest


def _build_receipts(context: BuildContext) -> dict[str, Any]:
    path = context.tree / RECEIPT_FILE
    return load_json(path).get("targets", {}) if path.is_file() else {}


def _publish_clangd_database(context: BuildContext) -> None:
    if (
        context.request.configuration != "Debug"
        or context.request.capability != "developer-full"
    ):
        return
    source = context.tree / "compile_commands.json"
    if not source.is_file():
        raise NuriBuildError(f"compilation database was not generated: {source}")
    destination = BUILD_ROOT / "compile_commands.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        shutil.copyfile(source, temp)
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


def run_subprocess(
    command: Sequence[str],
    *,
    environment: dict[str, str],
    cwd: Path = ROOT,
) -> int:
    print("+ " + subprocess.list2cmdline(list(command)), flush=True)
    try:
        return subprocess.run(
            list(command), cwd=cwd, env=environment, check=False
        ).returncode
    except OSError as error:
        raise NuriBuildError(f"failed to start {command[0]}: {error}") from error


def build(
    context: BuildContext,
    *,
    jobs: int = 0,
    lock_timeout: float = 600.0,
    configure_only: bool = False,
) -> dict[str, Any]:
    reserve_tree(context)
    input_digest, input_record = _nvrhi_input()
    configure_before_lock, configure_reason = _needs_configure(
        context, input_digest
    )
    if configure_before_lock:
        update_registry(context, "configuring", reason=configure_reason)
    command_text = " ".join(sys.argv)
    host_lock = FileLock(
        LOCK_ROOT / "10-host-build.lock",
        timeout=lock_timeout,
        identity=context.tree_digest,
        command=command_text,
    )
    dependency_lock = FileLock(
        LOCK_ROOT / f"20-dependency-{context.dependency_digest[7:]}.lock",
        timeout=lock_timeout,
        identity=context.dependency_digest,
        command=command_text,
    )
    prepared_lock = FileLock(
        LOCK_ROOT / f"30-prepared-{input_digest[7:]}.lock",
        timeout=lock_timeout,
        identity=input_digest,
        command=command_text,
    )
    tree_lock = FileLock(
        LOCK_ROOT / f"40-tree-{context.tree_digest[7:]}.lock",
        timeout=lock_timeout,
        identity=context.tree_digest,
        command=command_text,
    )
    try:
        with host_lock, dependency_lock, prepared_lock, tree_lock:
            prepared = prepare_nvrhi(input_digest, input_record)
            _dependency_owner(context)
            configure, reason = _needs_configure(context, input_digest)
            context.tree.mkdir(parents=True, exist_ok=True)
            if configure:
                print(f"configuring {context.tree.name}: {reason}")
                exit_code = run_subprocess(
                    configure_command(context, prepared),
                    environment=context.environment,
                )
                if exit_code:
                    raise NuriBuildError(
                        f"CMake configure failed with exit {exit_code}"
                    )
                atomic_json(
                    context.tree / IDENTITY_FILE, _identity_document(context)
                )
                atomic_json(
                    context.tree / NVRHI_INPUT_FILE,
                    {
                        "schemaVersion": 1,
                        "inputDigest": input_digest,
                        "preparedSource": canonical_path(prepared),
                    },
                )
            else:
                print(f"reusing {context.tree.name}: {reason}")
            if configure_only:
                receipts = _build_receipts(context)
                manifest = _publish_manifest(context, receipts)
                command = []
                elapsed = 0.0
            else:
                command = [
                    str(context.tools["cmake"]["path"]),
                    "--build",
                    str(context.tree),
                    "--target",
                    context.request.target,
                    "--config",
                    context.request.configuration,
                ]
                if jobs > 0:
                    command.extend(["--parallel", str(jobs)])
                started = time.perf_counter()
                exit_code = run_subprocess(command, environment=context.environment)
                elapsed = time.perf_counter() - started
                if exit_code:
                    raise NuriBuildError(f"build failed with exit {exit_code}")
                receipts = _build_receipts(context)
                receipts[context.request.target] = {
                    "target": context.request.target,
                    "configuration": context.request.configuration,
                    "treeIdentityDigest": context.tree_digest,
                    "completedUtc": utc_now(),
                    "durationSeconds": round(elapsed, 6),
                    "source": source_provenance(),
                    "command": command,
                }
                atomic_json(
                    context.tree / RECEIPT_FILE,
                    {"schemaVersion": 1, "targets": receipts},
                )
                manifest = _publish_manifest(context, receipts)
            _publish_clangd_database(context)
    except BaseException as error:
        failure_state = (
            "ready" if context.tree.joinpath(IDENTITY_FILE).is_file() else "reserved"
        )
        update_registry(context, failure_state, lastError=str(error))
        raise
    registry_details: dict[str, object] = {}
    if not configure_only:
        registry_details = {
            "lastTarget": context.request.target,
            "lastBuildUtc": receipts[context.request.target]["completedUtc"],
        }
    update_registry(context, "ready", **registry_details)
    return manifest


def validate_manifest(
    context: BuildContext, target: str, *, executable: bool
) -> tuple[dict[str, Any], Path | None]:
    identity_path = context.tree / IDENTITY_FILE
    manifest_path = context.tree / ARTIFACT_FILE
    if not identity_path.is_file() or not manifest_path.is_file():
        raise NuriBuildError(
            f"--no-build requires a ready identity and artifact manifest: {context.tree}"
        )
    identity = load_json(identity_path)
    manifest = load_json(manifest_path)
    if identity.get("treeIdentityDigest") != context.tree_digest:
        raise NuriBuildError("--no-build tree owner/identity mismatch")
    for key, expected in (
        ("treeIdentityDigest", context.tree_digest),
        ("compileCompatibilityDigest", context.compile_digest),
        ("dependencyIdentityDigest", context.dependency_digest),
        ("configuration", context.request.configuration),
    ):
        if manifest.get(key) != expected:
            raise NuriBuildError(f"--no-build manifest {key} mismatch")
    receipt = manifest.get("receipts", {}).get(target)
    if not isinstance(receipt, dict):
        raise NuriBuildError(f"--no-build has no successful receipt for {target}")
    artifact_record = manifest.get("targets", {}).get(target)
    if not executable:
        return manifest, None
    if not isinstance(artifact_record, dict):
        raise NuriBuildError(f"artifact manifest has no executable target {target}")
    path = Path(str(artifact_record.get("path", "")))
    if not path.is_file():
        raise NuriBuildError(f"artifact is missing: {path}")
    return manifest, path


def _runtime_environment(
    context: BuildContext, manifest: dict[str, Any]
) -> dict[str, str]:
    environment = dict(context.environment)
    paths = [
        str(Path(value))
        for value in manifest.get("runtimeSearchPaths", [])
        if value and Path(value).is_dir()
    ]
    environment["PATH"] = os.pathsep.join(
        [*paths, environment.get("PATH", "")]
    )
    library_var = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    library_paths = [value for value in paths if Path(value).name in {"lib", "bin"}]
    if library_paths:
        environment[library_var] = os.pathsep.join(
            [*library_paths, environment.get(library_var, "")]
        )
    return environment


def run_target(
    context: BuildContext,
    arguments: Sequence[str],
    *,
    no_build: bool,
    jobs: int,
    lock_timeout: float,
) -> int:
    if no_build:
        manifest, executable = validate_manifest(
            context, context.request.target, executable=True
        )
    else:
        manifest = build(context, jobs=jobs, lock_timeout=lock_timeout)
        _, executable = validate_manifest(
            context, context.request.target, executable=True
        )
    assert executable is not None
    record = manifest["targets"][context.request.target]
    lease = FileLock(
        LOCK_ROOT / f"40-tree-{context.tree_digest[7:]}.lock",
        shared=True,
        timeout=lock_timeout,
        identity=context.tree_digest,
        command="run " + context.request.target,
    )
    with lease:
        environment = _runtime_environment(context, manifest)
        environment["NURI_EXEC_START_UTC"] = utc_now()
        return run_subprocess(
            [str(executable), *arguments],
            environment=environment,
            cwd=Path(record["workingDirectory"]),
        )


def test_target(
    context: BuildContext,
    ctest_args: Sequence[str],
    *,
    no_build: bool,
    jobs: int,
    lock_timeout: float,
    labels: str | None,
) -> int:
    if no_build:
        manifest, _ = validate_manifest(
            context, context.request.target, executable=False
        )
    else:
        manifest = build(context, jobs=jobs, lock_timeout=lock_timeout)
    command = [
        "ctest",
        "--test-dir",
        str(context.tree),
        "--build-config",
        context.request.configuration,
        "--output-on-failure",
    ]
    if labels:
        command.extend(["-L", labels])
    if jobs > 0 and "-j" not in ctest_args and "--parallel" not in ctest_args:
        command.extend(["-j", str(jobs)])
    command.extend(ctest_args)
    with FileLock(
        LOCK_ROOT / f"40-tree-{context.tree_digest[7:]}.lock",
        shared=True,
        timeout=lock_timeout,
        identity=context.tree_digest,
        command="ctest " + context.request.target,
    ):
        return run_subprocess(
            command,
            environment=_runtime_environment(context, manifest),
        )


def health_for_tree(
    tree: Path,
    expected_digest: str | None = None,
    expected_target: str | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    state = "healthy"
    has_build_receipts = False
    identity_path = tree / IDENTITY_FILE
    if not identity_path.is_file():
        return {
            "state": "unknown",
            "tree": canonical_path(tree),
            "reasons": ["ready identity is missing"],
        }
    try:
        identity = load_json(identity_path)
    except NuriBuildError as error:
        return {
            "state": "damaged",
            "tree": canonical_path(tree),
            "reasons": [str(error)],
        }
    actual_digest = identity.get("treeIdentityDigest")
    if expected_digest and actual_digest != expected_digest:
        reasons.append("tree identity digest does not match request")
        state = "damaged"
    for name in ("CMakeCache.txt", "build.ninja"):
        if not tree.joinpath(name).is_file():
            reasons.append(f"{name} is missing")
            state = "damaged"
    cache_path = tree / "CMakeCache.txt"
    if cache_path.is_file():
        cache_text = cache_path.read_text(encoding="utf-8", errors="replace")
        if "CMAKE_GENERATOR:INTERNAL=Ninja" not in cache_text:
            reasons.append("configured generator is not Ninja")
            state = "damaged"
    graph_path = tree / "build.ninja"
    if expected_target and graph_path.is_file():
        graph_text = graph_path.read_text(encoding="utf-8", errors="replace")
        target_patterns = (
            f"build {expected_target}:",
            f"build {expected_target}.exe:",
            f" {expected_target} ",
        )
        if not any(pattern in graph_text for pattern in target_patterns):
            reasons.append(f"expected target is unavailable: {expected_target}")
            state = "damaged"
    ninja_record = (
        identity.get("treeIdentity", {})
        .get("tools", {})
        .get("ninja", {})
    )
    ninja_path = Path(str(ninja_record.get("path", "")))
    if ninja_path.is_file():
        try:
            current_version = command_output([str(ninja_path), "--version"])
            if current_version != ninja_record.get("version"):
                reasons.append("Ninja version differs from the tree identity")
                state = "suspect" if state == "healthy" else state
        except NuriBuildError as error:
            reasons.append(f"cannot verify Ninja version: {error}")
            state = "unknown" if state == "healthy" else state
    else:
        reasons.append("identity Ninja executable is unavailable")
        state = "unknown" if state == "healthy" else state
    receipt_path = tree / RECEIPT_FILE
    if receipt_path.is_file():
        try:
            receipt_targets = load_json(receipt_path).get("targets")
            if not isinstance(receipt_targets, dict):
                raise NuriBuildError(
                    f"receipt targets must be an object: {receipt_path}"
                )
            has_build_receipts = bool(receipt_targets)
        except NuriBuildError as error:
            reasons.append(str(error))
            state = "damaged"
    ninja_log = tree / ".ninja_log"
    if ninja_log.is_file():
        try:
            lines = ninja_log.read_text(encoding="utf-8", errors="strict").splitlines()
            if not lines or not lines[0].startswith("# ninja log v"):
                raise ValueError("header is invalid")
            for index, line in enumerate(lines[1:], start=2):
                if line and len(line.split("\t")) < 5:
                    raise ValueError(f"record {index} is truncated")
        except (OSError, UnicodeError, ValueError) as error:
            reasons.append(f".ninja_log is malformed: {error}")
            state = "damaged"
    else:
        if has_build_receipts:
            reasons.append(".ninja_log is absent despite recorded build receipts")
            if state == "healthy":
                state = "suspect"
    ninja_deps = tree / ".ninja_deps"
    if ninja_deps.is_file():
        try:
            data = ninja_deps.read_bytes()
            if len(data) < 16 or not data.startswith(b"# ninjadeps\n"):
                raise ValueError("header or minimum length is invalid")
            if len(data) % 4:
                raise ValueError("file length is not 4-byte aligned")
        except (OSError, ValueError) as error:
            reasons.append(f".ninja_deps is malformed: {error}")
            state = "damaged"
    else:
        if has_build_receipts:
            reasons.append(".ninja_deps is absent despite recorded build receipts")
            if state == "healthy":
                state = "suspect"
    manifest = tree / ARTIFACT_FILE
    if not manifest.is_file():
        reasons.append("published artifact manifest is missing")
        if state == "healthy":
            state = "suspect"
    if not reasons:
        details = (
            "identity, generated graph, Ninja metadata, and manifest passed"
            if has_build_receipts
            else "identity, generated graph, and manifest passed (configured, not built)"
        )
        reasons.append(details)
    return {
        "schemaVersion": 1,
        "kind": "nuri.build_health",
        "state": state,
        "tree": canonical_path(tree),
        "treeIdentityDigest": actual_digest,
        "reasons": reasons,
        "checkedUtc": utc_now(),
        "sideEffectFree": True,
    }


def iter_registered_trees() -> Iterator[tuple[str, dict[str, Any], Path]]:
    registry = _registry()
    for tree_digest, record in sorted(registry["trees"].items()):
        path = Path(str(record.get("path", "")))
        yield tree_digest, record, path


def disk_size(path: Path) -> dict[str, int]:
    logical = 0
    physical = 0
    files = 0
    if not path.exists():
        return {"logicalBytes": 0, "physicalBytes": 0, "files": 0}
    for root, _dirs, names in os.walk(path):
        for name in names:
            candidate = Path(root) / name
            try:
                stat = candidate.stat()
            except OSError:
                continue
            logical += stat.st_size
            blocks = getattr(stat, "st_blocks", 0)
            physical += blocks * 512 if blocks else stat.st_size
            files += 1
    return {"logicalBytes": logical, "physicalBytes": physical, "files": files}


def safe_managed_child(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    managed_root = root.resolve()
    if resolved == managed_root or resolved == ROOT.resolve():
        raise NuriBuildError(f"refusing broad managed path: {resolved}")
    try:
        resolved.relative_to(managed_root)
    except ValueError as error:
        raise NuriBuildError(f"path escapes managed root: {resolved}") from error
    current = managed_root
    for part in resolved.relative_to(managed_root).parts:
        current = current / part
        if current.is_symlink():
            raise NuriBuildError(f"managed path traverses a symlink: {current}")
    return resolved


def cmd_status(_args: argparse.Namespace) -> int:
    records: list[dict[str, Any]] = []
    registered_paths: set[str] = set()
    for tree_digest, record, tree in iter_registered_trees():
        registered_paths.add(canonical_path(tree))
        records.append(
            {
                **record,
                "treeIdentityDigest": tree_digest,
                "health": health_for_tree(tree, tree_digest),
                "disk": disk_size(tree),
            }
        )
    legacy: list[dict[str, object]] = []
    if BUILD_ROOT.is_dir():
        for child in BUILD_ROOT.iterdir():
            if (
                not child.is_dir()
                or child.name.startswith("_")
                or not child.joinpath("CMakeCache.txt").is_file()
            ):
                continue
            if canonical_path(child) not in registered_paths:
                legacy.append(
                    {
                        "path": canonical_path(child),
                        "state": "legacy-stale-candidate",
                        "disk": disk_size(child),
                    }
                )
    print(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "nuri.build_status",
                "lockOrder": list(LOCK_ORDER),
                "trees": records,
                "legacyTrees": legacy,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    if args.tree:
        result = health_for_tree(Path(args.tree))
    elif args.variant:
        policy = load_policy()
        request = explicit_request(policy, args.variant, args.capability, args.target)
        context = create_context(request, policy)
        result = health_for_tree(
            context.tree, context.tree_digest, context.request.target
        )
    else:
        raise NuriBuildError("health requires --tree or --variant/--capability/--target")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["state"] in {"healthy", "suspect"} else 1


def cmd_repair(args: argparse.Namespace) -> int:
    tree = safe_managed_child(Path(args.tree), TREE_ROOT)
    result = health_for_tree(tree)
    print(json.dumps(result, indent=2, sort_keys=True))
    damaged = [
        tree / name
        for name in (".ninja_deps", ".ninja_log")
        if any(name in reason for reason in result["reasons"])
    ]
    if not damaged:
        raise NuriBuildError("repair found no proven damaged Ninja metadata")
    if not args.execute:
        print("dry-run: pass --execute to quarantine only the listed metadata")
        return 0
    identity = load_json(tree / IDENTITY_FILE)
    tree_digest = str(identity["treeIdentityDigest"])
    with FileLock(
        LOCK_ROOT / "10-host-build.lock",
        identity=tree_digest,
        command="repair",
    ), FileLock(
        LOCK_ROOT / f"40-tree-{tree_digest[7:]}.lock",
        identity=tree_digest,
        command="repair",
    ):
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        archive = HEALTH_ARCHIVE_ROOT / f"{tree.name}-{stamp}"
        archive.mkdir(parents=True, exist_ok=False)
        atomic_json(archive / "health.json", result)
        shutil.copy2(tree / IDENTITY_FILE, archive / IDENTITY_FILE)
        for path in damaged:
            if path.is_file():
                shutil.move(str(path), str(archive / path.name))
        atomic_json(
            archive / "repair.json",
            {
                "schemaVersion": 1,
                "tree": canonical_path(tree),
                "treeIdentityDigest": tree_digest,
                "quarantined": [path.name for path in damaged],
                "rebuildRequired": True,
                "createdUtc": utc_now(),
            },
        )
    print(f"quarantined damaged metadata to {archive}; next build is controlled")
    return 0


def cmd_disk_usage(_args: argparse.Namespace) -> int:
    areas = {
        "canonicalTrees": TREE_ROOT,
        "dependencies": DEPENDENCY_ROOT,
        "preparedSources": PREPARED_ROOT,
        "legacyAndCanonicalBuildRoot": BUILD_ROOT,
        "validationArtifacts": ROOT / "artifacts",
    }
    print(
        json.dumps(
            {
                "schemaVersion": 1,
                "kind": "nuri.build_disk_usage",
                "areas": {
                    name: {"path": canonical_path(path), **disk_size(path)}
                    for name, path in areas.items()
                },
                "note": "physicalBytes uses allocated blocks where the host exposes them; otherwise it equals logicalBytes",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    tree = safe_managed_child(Path(args.tree), TREE_ROOT)
    identity = load_json(tree / IDENTITY_FILE)
    tree_digest = str(identity["treeIdentityDigest"])
    command = ["cmake", "--build", str(tree), "--target", "clean"]
    print("dry-run command: " + subprocess.list2cmdline(command))
    if not args.execute:
        return 0
    with FileLock(
        LOCK_ROOT / "10-host-build.lock",
        identity=tree_digest,
        command="clean",
    ), FileLock(
        LOCK_ROOT / f"40-tree-{tree_digest[7:]}.lock",
        identity=tree_digest,
        command="clean",
    ):
        return run_subprocess(command, environment=build_environment())


def cmd_prune(args: argparse.Namespace) -> int:
    candidates: list[tuple[Path, str | None]] = []
    if args.tree:
        path = safe_managed_child(Path(args.tree), TREE_ROOT)
        identity = load_json(path / IDENTITY_FILE)
        candidates.append((path, str(identity["treeIdentityDigest"])))
    if args.legacy_tree:
        path = safe_managed_child(Path(args.legacy_tree), BUILD_ROOT)
        if path.parent != BUILD_ROOT.resolve() or path.name.startswith("_"):
            raise NuriBuildError("legacy prune accepts only a direct legacy build child")
        if path.joinpath(IDENTITY_FILE).exists() or not path.joinpath(
            "CMakeCache.txt"
        ).is_file():
            raise NuriBuildError("legacy prune target is not a recognized legacy tree")
        candidates.append((path, None))
    if args.stale_identities:
        for tree_digest, record, tree in iter_registered_trees():
            if record.get("state") == "stale":
                candidates.append(
                    (safe_managed_child(tree, TREE_ROOT), tree_digest)
                )
    report = [
        {
            "path": canonical_path(path),
            "treeIdentityDigest": tree_digest,
            "kind": "canonical" if tree_digest else "legacy",
            "disk": disk_size(path),
        }
        for path, tree_digest in candidates
    ]
    print(json.dumps({"dryRun": not args.execute, "candidates": report}, indent=2))
    if not args.execute:
        return 0
    if not candidates:
        return 0
    TRASH_ROOT.mkdir(parents=True, exist_ok=True)
    with FileLock(LOCK_ROOT / "00-registry.lock", command="prune"):
        registry = _registry()
        with FileLock(LOCK_ROOT / "10-host-build.lock", command="prune"):
            for path, tree_digest in candidates:
                lock_identity = tree_digest or digest(canonical_path(path))
                lock_suffix = (
                    f"tree-{lock_identity[7:]}"
                    if tree_digest
                    else f"legacy-{lock_identity[7:]}"
                )
                with FileLock(
                    LOCK_ROOT / f"40-{lock_suffix}.lock",
                    timeout=args.lock_timeout,
                    identity=lock_identity,
                    command="prune",
                ):
                    destination = TRASH_ROOT / (
                        f"{path.name}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
                    )
                    os.replace(path, destination)
                    if tree_digest:
                        registry["trees"].pop(tree_digest, None)
                    print(f"moved {path} to recoverable quarantine {destination}")
        atomic_json(REGISTRY_FILE, registry)
    return 0


def cmd_prune_logs(args: argparse.Namespace) -> int:
    policy = load_policy()["retention"]
    cutoff = time.time() - int(policy["logMaxAgeDays"]) * 86400
    candidates: list[Path] = []
    for root in (ROOT / ".scratch", ROOT / "artifacts"):
        if not root.is_dir():
            continue
        for path in root.rglob("*.log"):
            try:
                if path.stat().st_mtime < cutoff or path.stat().st_size > int(
                    policy["logMaxBytes"]
                ):
                    candidates.append(path.resolve())
            except OSError:
                continue
    print(
        json.dumps(
            {
                "dryRun": not args.execute,
                "files": [
                    {"path": str(path), "bytes": path.stat().st_size}
                    for path in candidates
                ],
            },
            indent=2,
        )
    )
    if args.execute:
        for path in candidates:
            if not (
                path.is_relative_to((ROOT / ".scratch").resolve())
                or path.is_relative_to((ROOT / "artifacts").resolve())
            ):
                raise NuriBuildError(f"log path escaped retention roots: {path}")
            path.unlink()
    return 0


def cmd_cache_status(_args: argparse.Namespace) -> int:
    policy = load_policy()
    environment = build_environment()
    launcher = os.environ.get("NURI_COMPILER_CACHE", "off")
    report: dict[str, object] = {
        "schemaVersion": 1,
        "kind": "nuri.compiler_cache_status",
        "launcher": launcher,
        "configuredMaxBytes": policy["retention"]["compilerCacheMaxBytes"],
    }
    if launcher == "off":
        report["state"] = "disabled"
    else:
        path = shutil.which(launcher, path=environment.get("PATH"))
        if not path:
            report.update({"state": "unavailable", "reason": "launcher not found"})
        else:
            report.update({"state": "available", "path": canonical_path(Path(path))})
            if Path(path).name.lower().startswith("sccache"):
                completed = subprocess.run(
                    [path, "--show-stats", "--stats-format", "json"],
                    env=environment,
                    capture_output=True,
                    text=True,
                    errors="replace",
                    check=False,
                )
                report["statsExitCode"] = completed.returncode
                try:
                    report["stats"] = json.loads(completed.stdout)
                except json.JSONDecodeError:
                    report["statsText"] = (
                        completed.stdout + completed.stderr
                    ).strip()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("state") != "unavailable" else 1


def _common_context(args: argparse.Namespace) -> BuildContext:
    policy = load_policy()
    request = explicit_request(policy, args.variant, args.capability, args.target)
    return create_context(request, policy)


def cmd_build(args: argparse.Namespace) -> int:
    build(
        _common_context(args),
        jobs=args.jobs,
        lock_timeout=args.lock_timeout,
    )
    return 0


def cmd_configure(args: argparse.Namespace) -> int:
    build(
        _common_context(args),
        jobs=0,
        lock_timeout=args.lock_timeout,
        configure_only=True,
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    return run_target(
        _common_context(args),
        args.arguments,
        no_build=args.no_build,
        jobs=args.jobs,
        lock_timeout=args.lock_timeout,
    )


def cmd_test(args: argparse.Namespace) -> int:
    context = _common_context(args)
    return test_target(
        context,
        args.arguments,
        no_build=args.no_build,
        jobs=args.jobs,
        lock_timeout=args.lock_timeout,
        labels=args.labels,
    )


def _legacy_parts(
    values: Sequence[str],
) -> tuple[list[str], bool, list[str]]:
    modifiers: list[str] = []
    no_build = False
    index = 0
    while index < len(values):
        value = values[index]
        if value == "--":
            index += 1
            break
        if value == "--no-build":
            no_build = True
            index += 1
            continue
        if value.lower() in {"cpu", "off", "devchecks"}:
            modifiers.append(value.lower())
            index += 1
            continue
        break
    return modifiers, no_build, list(values[index:])


def cmd_legacy_build(args: argparse.Namespace) -> int:
    policy = load_policy()
    request = legacy_request(args.mode, args.profile, args.modifiers, policy)
    context = create_context(request, policy)
    build(context, jobs=args.jobs, lock_timeout=args.lock_timeout)
    return 0


def cmd_legacy_run(args: argparse.Namespace) -> int:
    policy = load_policy()
    modifiers, no_build, arguments = _legacy_parts(args.values)
    request = legacy_request(args.mode, args.profile, modifiers, policy)
    canonical_profile, _ = profile_record(policy, args.profile)
    verbs = {
        "bench": {
            "list",
            "explain",
            "run",
            "check",
            "compare",
            "summarize",
            "graph",
            "baseline",
        },
        "snapshot": {
            "list",
            "explain",
            "capture",
            "compare",
            "run",
            "approve",
            "baseline",
            "diff",
        },
        "autotest": {
            "list",
            "explain",
            "run",
            "record",
            "approve",
            "baseline",
        },
    }
    if canonical_profile in verbs:
        if not arguments:
            arguments = ["list"]
        elif arguments[0] not in verbs[canonical_profile]:
            arguments.insert(0, "run")
    return run_target(
        create_context(request, policy),
        arguments,
        no_build=no_build,
        jobs=args.jobs,
        lock_timeout=args.lock_timeout,
    )


def cmd_legacy_wrapper_run(args: argparse.Namespace) -> int:
    values = list(args.values)
    default_mode = "debug" if args.profile in {"app", "editor"} else "release"
    mode = default_mode
    if values and values[0].lower() in {"debug", "release"}:
        mode = values.pop(0).lower()
    forwarded = argparse.Namespace(
        mode=mode,
        profile=args.profile,
        values=values,
        jobs=args.jobs,
        lock_timeout=args.lock_timeout,
    )
    return cmd_legacy_run(forwarded)


def cmd_legacy_test(args: argparse.Namespace) -> int:
    policy = load_policy()
    request = legacy_request(args.mode, args.profile, (), policy)
    return test_target(
        create_context(request, policy),
        args.arguments,
        no_build=args.no_build,
        jobs=args.jobs,
        lock_timeout=args.lock_timeout,
        labels=args.labels or request.labels,
    )


def cmd_legacy_wrapper_test(args: argparse.Namespace) -> int:
    values = list(args.values)
    mode = "debug"
    profile = "tests"
    if values and values[0].lower() in {"debug", "release"}:
        mode = values.pop(0).lower()
    if values:
        try:
            canonical, _record = profile_record(load_policy(), values[0].lower())
        except NuriBuildError:
            canonical = ""
        if canonical in {"bench-tests", "snapshot-tests", "autotest-tests"}:
            profile = canonical
            values.pop(0)
    no_build = False
    if values and values[0] == "--no-build":
        no_build = True
        values.pop(0)
    forwarded = argparse.Namespace(
        mode=mode,
        profile=profile,
        labels=None,
        no_build=no_build,
        jobs=args.jobs,
        lock_timeout=args.lock_timeout,
        arguments=values,
    )
    return cmd_legacy_test(forwarded)


def cmd_resolve(args: argparse.Namespace) -> int:
    policy = load_policy()
    if args.mode:
        request = legacy_request(args.mode, args.profile, args.modifiers, policy)
    else:
        request = explicit_request(policy, args.variant, args.capability, args.target)
    context = create_context(request, policy)
    print(
        json.dumps(
            {
                "request": request.__dict__,
                "tree": canonical_path(context.tree),
                "dependencyRoot": canonical_path(context.dependency_root),
                "treeIdentityDigest": context.tree_digest,
                "compileCompatibilityDigest": context.compile_digest,
                "dependencyIdentityDigest": context.dependency_digest,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def add_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--variant", required=True)
    parser.add_argument("--capability", required=True)
    parser.add_argument("--target", required=True)


def add_execution(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--lock-timeout", type=float, default=600.0)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="nuri-build")
    commands = result.add_subparsers(dest="command", required=True)

    build_parser = commands.add_parser("build")
    add_selection(build_parser)
    add_execution(build_parser)
    build_parser.set_defaults(func=cmd_build)

    configure_parser = commands.add_parser("configure")
    add_selection(configure_parser)
    configure_parser.add_argument("--lock-timeout", type=float, default=600.0)
    configure_parser.set_defaults(func=cmd_configure)

    run_parser = commands.add_parser("run")
    add_selection(run_parser)
    add_execution(run_parser)
    run_parser.add_argument("--no-build", action="store_true")
    run_parser.add_argument("arguments", nargs=argparse.REMAINDER)
    run_parser.set_defaults(func=cmd_run)

    test_parser = commands.add_parser("test")
    add_selection(test_parser)
    add_execution(test_parser)
    test_parser.add_argument("--labels")
    test_parser.add_argument("--no-build", action="store_true")
    test_parser.add_argument("arguments", nargs=argparse.REMAINDER)
    test_parser.set_defaults(func=cmd_test)

    legacy_build = commands.add_parser("legacy-build")
    legacy_build.add_argument("mode")
    legacy_build.add_argument("profile")
    legacy_build.add_argument("modifiers", nargs="*")
    add_execution(legacy_build)
    legacy_build.set_defaults(func=cmd_legacy_build)

    legacy_run = commands.add_parser("legacy-run")
    legacy_run.add_argument("mode")
    legacy_run.add_argument("profile")
    legacy_run.add_argument("values", nargs=argparse.REMAINDER)
    add_execution(legacy_run)
    legacy_run.set_defaults(func=cmd_legacy_run)

    wrapper_run = commands.add_parser("legacy-wrapper-run")
    wrapper_run.add_argument("profile")
    wrapper_run.add_argument("values", nargs=argparse.REMAINDER)
    add_execution(wrapper_run)
    wrapper_run.set_defaults(func=cmd_legacy_wrapper_run)

    legacy_test = commands.add_parser("legacy-test")
    legacy_test.add_argument("mode")
    legacy_test.add_argument("profile")
    legacy_test.add_argument("--labels")
    legacy_test.add_argument("--no-build", action="store_true")
    add_execution(legacy_test)
    legacy_test.add_argument("arguments", nargs=argparse.REMAINDER)
    legacy_test.set_defaults(func=cmd_legacy_test)

    wrapper_test = commands.add_parser("legacy-wrapper-test")
    wrapper_test.add_argument("values", nargs=argparse.REMAINDER)
    add_execution(wrapper_test)
    wrapper_test.set_defaults(func=cmd_legacy_wrapper_test)

    resolve = commands.add_parser("resolve")
    selection = resolve.add_mutually_exclusive_group(required=True)
    selection.add_argument("--mode")
    selection.add_argument("--variant")
    resolve.add_argument("--profile", default="app")
    resolve.add_argument("--modifiers", nargs="*", default=[])
    resolve.add_argument("--capability", default="developer-full")
    resolve.add_argument("--target", default="nuri")
    resolve.set_defaults(func=cmd_resolve)

    status = commands.add_parser("status")
    status.set_defaults(func=cmd_status)

    health = commands.add_parser("health")
    health.add_argument("--tree")
    health.add_argument("--variant")
    health.add_argument("--capability", default="developer-full")
    health.add_argument("--target", default="nuri_renderer")
    health.set_defaults(func=cmd_health)

    repair = commands.add_parser("repair")
    repair.add_argument("--tree", required=True)
    repair.add_argument("--execute", action="store_true")
    repair.set_defaults(func=cmd_repair)

    usage = commands.add_parser("disk-usage")
    usage.set_defaults(func=cmd_disk_usage)

    clean = commands.add_parser("clean")
    clean.add_argument("--tree", required=True)
    clean.add_argument("--execute", action="store_true")
    clean.set_defaults(func=cmd_clean)

    prune = commands.add_parser("prune")
    prune.add_argument("--tree")
    prune.add_argument("--legacy-tree")
    prune.add_argument("--stale-identities", action="store_true")
    prune.add_argument("--execute", action="store_true")
    prune.add_argument("--lock-timeout", type=float, default=30.0)
    prune.set_defaults(func=cmd_prune)

    prune_logs = commands.add_parser("prune-logs")
    prune_logs.add_argument("--execute", action="store_true")
    prune_logs.set_defaults(func=cmd_prune_logs)

    cache_status = commands.add_parser("cache-status")
    cache_status.set_defaults(func=cmd_cache_status)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        if getattr(args, "jobs", 0) < 0:
            raise NuriBuildError("--jobs must be non-negative")
        if getattr(args, "lock_timeout", 1.0) <= 0:
            raise NuriBuildError("--lock-timeout must be positive")
        arguments = getattr(args, "arguments", None)
        if arguments and arguments[0] == "--":
            args.arguments = arguments[1:]
        return int(args.func(args))
    except (NuriBuildError, ValueError) as error:
        print(f"nuri-build: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
