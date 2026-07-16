# CLAUDE

## Agent skills

### Issue tracker

Issues and PRDs are tracked in this repository's GitHub Issues. See `docs/agents/issue-tracker.md`.

### Triage labels

Triage uses the standard five-role label vocabulary. See `docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repository with root domain docs and ADRs. See `docs/agents/domain.md`.

## Project intent
- This codebase builds a renderer on NVRHI. Keep NVRHI types private so higher-level code depends on Nuri's typed descriptors, handles, and submission contracts.
- Correctness is the top priority, with performance as a critical constraint to be optimized after correctness is ensured. Optimize throughput and frame time only once behavior is correct.

## C++ and memory
- Use C++20 and the STL for core functionality.
- Use `std::string_view` for immutable strings (debug names, string literals, error/debug/info messages etc.)
- Use `std::span` for non-owning views on containers
- Prefer PMR types and custom allocators to control allocations and reduce churn.
- For transient allocations with clear scope lifetime (loop iteration/task/frame slice), prefer `ScratchArena` + `ScopedScratch` from `lib/nuri/core/pmr_scratch.h`.
- Do not let scratch-backed allocations escape the guard scope; all scratch-backed objects must be destroyed before `ScopedScratch` exits.
- Do not nest `ScopedScratch` over the same `ScratchArena`.
- Avoid exceptions when possible; use `lib/nuri/result.h` for error handling.

## Renderer engineering
- For renderer architecture, semantic compression, N+1 design, RHI seams,
  PIMPL decisions, GPU lifetime, and measured performance work, follow
  `.codex/skills/nuri-renderer-design/SKILL.md` and its focused references.

## Shaders
- Shaders are written in GLSL.

## Build and scripts
- Do not run `cmake` manually: it can overwrite generated files, bypass project configuration, and produce inconsistent builds; use the build scripts in `scripts/` instead.
- CMake is the build system.
- Build/run scripts live in `scripts/`.

## Profiling (Tracy)
- Prefer measuring before optimizing. Use `nuri-bench` as the primary renderer performance oracle and Tracy as a benchmark-owned diagnostic artifact.
- Enable CPU profiling through the build/run scripts with `cpu`; do not invoke CMake directly.
- Instrument with `lib/nuri/core/profiling.h` macros and keep zones coarse and meaningful before drilling down.

## Logging
- `fatal`: use for paths that will crash or abort the app.
- `error`: use for paths that return a recoverable error that can be handled by callers.
- `warning`: use for paths where behavior is unexpected but not critical to application function.
- `info`: use for generic app behavior and user-relevant runtime information (device info, startup phases, etc.).
- `debug`: use in non-hot paths to trace creation/initialization of resources and systems.

## Naming conventions
- `ModelData` is CPU-side asset data; `Model` is the GPU/renderable object.
- `Buffer` and `Texture` refer to GPU resources, not raw data containers.

