# CLAUDE

## Project intent
- This codebase builds a renderer on top of LVK (LightweightVK). Keep LVK usage behind clean abstractions where it makes sense so higher-level code is not tightly coupled to the backend.
- Performance is the top priority, followed by correctness. Optimize for throughput and frame time while keeping behavior correct.

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
- DO NOT EXECUTE `cmake` COMMAND ON YOUR OWN NEVER EVER DO IT
- CMake is the build system.
- Build/run scripts live in `scripts/`.

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

