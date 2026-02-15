# AGENTS

## Project intent
- This codebase builds a renderer on top of LVK (LightweightVK). Keep LVK usage behind clean abstractions where it makes sense so higher-level code is not tightly coupled to the backend.
- Correctness is the top priority, with performance as a critical constraint to be optimized after correctness is ensured. Optimize throughput and frame time only once behavior is correct.

## C++ and memory
- Use C++20 and the STL for core functionality.
- Use `std::string_view` for immutable strings (debug names, string literals, error/debug/info messages etc.)
- Use `std::span` for non-owning views on containers
- Prefer PMR types and custom allocators to control allocations and reduce churn.
- Avoid exceptions when possible; use `lib/nuri/result.h` for error handling.

## Performance style
- Prefer contiguous layouts in hot paths (`std::vector`, SoA when useful) to maximize cache locality.
- Use polymorphism deliberately: avoid virtual dispatch in tight loops unless profiling shows negligible impact.
- Avoid blocking mutexes on frame-critical threads; use lock-free/wait-free approaches only when needed and proven correct.
- Measure before/after performance changes with Tracy, and optimize only verified bottlenecks.
- Use semantic compression where domain intent is clearer for expert readers; prefer explicit code at public/module boundaries and in high-risk logic (concurrency, lifecycle, error handling).

## Architecture and lifecycle principles (N+1 + ZII)
- Apply N+1 design at module boundaries: shape APIs/data so one additional backend/feature/state can be added without rewriting call sites.
- Do not over-generalize hot paths for N+1; keep per-draw/per-dispatch paths direct unless profiling proves abstraction cost is negligible.
- Use ZII (zero-is-initialization) as a reuse strategy for transient state: reset reusable records/metadata to zero or explicit invalid sentinels before reuse.
- ZII does not replace destruction for owning GPU resources: always release backend/driver resources explicitly, then invalidate handles/state.
- Reuse pooled GPU resources only after GPU completion is proven (fence/timeline); never recycle in-flight resources.
- Never `memset` non-trivial C++ objects; prefer explicit `reset()` logic for correctness.

## Abstractions and deps
- Use the PIMPL pattern when you need to hide third-party deps, reduce rebuilds, or keep ABI stable.
- Examples: `GPUDevice` and `Window` use `struct Impl` to hide LVK/GLFW types from public headers.

## Shaders
- Shaders are written in GLSL.

## Build and scripts
- Do not run `cmake` manually: it can overwrite generated files, bypass project configuration, and produce inconsistent builds; use the build scripts in `scripts/` instead.
- CMake is the build system.
- Build/run scripts live in `scripts/`.

## Profiling (Tracy)
- Prefer measuring before optimizing. Use the built-in Tracy hooks when investigating performance.
- Enable profiling via CMake: `-DNURI_WITH_TRACY=ON` (Debug builds do this by default).
- Instrument code using `lib/nuri/core/profiling.h` macros:
  - `NURI_PROFILER_FUNCTION()` / `NURI_PROFILER_FUNCTION_COLOR(color)`
  - `NURI_PROFILER_ZONE(name, color)` / `NURI_PROFILER_ZONE_END()`
  - `NURI_PROFILER_FRAME(name)` and `NURI_PROFILER_THREAD(name)`
- Focus zones on likely bottlenecks (frame loop, render submission, shader compilation, asset loading). Keep zones coarse and meaningful; avoid spamming per-draw zones unless you’re drilling down.

## Logging
- `fatal`: use for paths that will crash or abort the app.
- `error`: use for paths that return a recoverable error that can be handled by callers.
- `warning`: use for paths where behavior is unexpected but not critical to application function.
- `info`: use for generic app behavior and user-relevant runtime information (device info, startup phases, etc.).
- `debug`: use in non-hot paths to trace creation/initialization of resources and systems.

## Naming conventions
- `ModelData` is CPU-side asset data; `Model` is the GPU/renderable object.
- `Buffer` and `Texture` refer to GPU resources (wrapping LVK handles), not raw data containers.
