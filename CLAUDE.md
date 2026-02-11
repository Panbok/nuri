# CLAUDE

## Project intent
- This codebase builds a renderer on top of LVK (LightweightVK). Keep LVK usage behind clean abstractions where it makes sense so higher-level code is not tightly coupled to the backend.
- Correctness is the top priority, with performance as a critical constraint to be optimized after correctness is ensured. Optimize throughput and frame time only once behavior is correct.

## C++ and memory
- Use C++20 and the STL for core functionality.
- Use `std::string_view` for immutable strings (debug names, string literals, error/debug/info messages etc.)
- Use `std::span` for non-owning views on containers
- Prefer PMR types and custom allocators to control allocations and reduce churn.
- Avoid exceptions when possible; use `lib/nuri/result.h` for error handling.

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
- When working on performance-sensitive changes, enable Tracy and validate impact with a capture.
- Enable profiling via CMake: `-DNURI_WITH_TRACY=ON` (Debug builds do this by default).
- Use `lib/nuri/core/profiling.h` macros for instrumentation (`NURI_PROFILER_FUNCTION`, `NURI_PROFILER_ZONE`, `NURI_PROFILER_FRAME`, `NURI_PROFILER_THREAD`).
- Prefer a few high-level zones first (frame loop, renderer submit, resource creation/loading), then add finer-grained zones only where needed.

## Logging
- `fatal`: use for paths that will crash or abort the app.
- `error`: use for paths that return a recoverable error that can be handled by callers.
- `warning`: use for paths where behavior is unexpected but not critical to application function.
- `info`: use for generic app behavior and user-relevant runtime information (device info, startup phases, etc.).
- `debug`: use in non-hot paths to trace creation/initialization of resources and systems.

## Repo layout (high level)
- `app/` contains the application entry point and sample usage.
- `lib/nuri/` contains engine and renderer code (pipeline, shader, result).
- `lib/nuri/platform/` contains platform-facing abstractions (Window, GPUDevice).
- `lib/nuri/resources/gpu` holds GPU-side resources; `lib/nuri/resources/cpu` holds CPU-side data.
- `external/lightweightvk/` is the LVK submodule.
- `assets/` holds runtime assets.

## Naming conventions
- `ModelData` is CPU-side asset data; `Model` is the GPU/renderable object.
- `Buffer` and `Texture` refer to GPU resources (wrapping LVK handles), not raw data containers.

