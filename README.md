# nuri

This project uses NVRHI + GLFW on Vulkan. NVRHI is included as a git submodule at `external/nvrhi` and is built as part of the project. NVRHI, Vulkan, and GLFW remain private to the platform layer (`lib/nuri/platform`).

## Prerequisites

- Git
- Python 3 (`python3` or `py -3`)
- Vulkan SDK (so CMake can find `vulkan-1.lib` / `libvulkan.so`)
- CMake 3.29+
- Ninja
- Clang (recommended; the provided scripts use `clang`/`clang++`)
- vcpkg (used for third-party packages)

## Windows (PowerShell)

```powershell
# Clone + submodule
git clone --recurse-submodules <your-repo-url> nuri
cd nuri
git submodule update --init --recursive   # safe to re-run

# Point to your vcpkg installation
$env:VCPKG_ROOT = "E:\install\vcpkg"   # adjust

# Build + run app (debug by default)
.\scripts\run_app.bat

# Build + run editor
.\scripts\run_editor.bat
```

Release build:

```powershell
.\scripts\build_app.bat release
.\scripts\build_editor.bat release
.\scripts\build_lib.bat release
```

RenderDoc capture on Windows:

```powershell
# Build the release editor (or app) first
.\scripts\build_editor.bat release

# Recommended RenderDoc capture settings for nuri
$env:NURI_PRESENT_MODE = "fifo"
$env:NURI_VK_VALIDATION = "1"
$env:NURI_VK_SYNC_VALIDATION = "1"
$env:NURI_VK_DIAGNOSTICS = "1"

# Launch from the same shell so RenderDoc inherits the environment
E:\install\nuri\build\release\nuri_editor.exe
```

Notes for RenderDoc:

- Use `FIFO` present mode for RenderDoc captures. `IMMEDIATE` present mode can trigger a RenderDoc-only `VK_ERROR_DEVICE_LOST` during frame pacing / swapchain synchronization.
- If you capture the sample app instead of the editor, launch `E:\install\nuri\build\release\nuri_app.exe` from the same shell after setting the same environment variables.
- `NURI_VK_VALIDATION=1` enables Vulkan validation in Release builds for capture sessions.
- `NURI_VK_SYNC_VALIDATION=1` enables Vulkan synchronization validation and is useful for debugging swapchain / submit ordering issues.
- `NURI_VK_DIAGNOSTICS=1` enables extra submit / present logging in the Vulkan backend.
- In RenderDoc, disable `DebugOutputMute` if you want validation-layer messages to appear in the app log during capture.

Tests:

```powershell
.\scripts\run_tests.bat
```

`run_tests` enables the manifest `tests` feature automatically. If you consume `nuri` as a vcpkg port instead of using this repo in manifest mode, install `nuri[tests]` before running the test targets.

Benchmarks:

```powershell
# Build the benchmark CLI
.\scripts\build_benchmarks.bat release off

# List available benchmark cases
.\scripts\run_benchmarks.bat release off list

# Explain one case
.\scripts\run_benchmarks.bat release off explain --case smoke.procedural.default

# Run one case and write a report
.\scripts\run_benchmarks.bat release off --case smoke.procedural.default --json-out artifacts\bench\smoke_live.json

# Run a suite and also write an all-metrics graph view
.\scripts\run_benchmarks.bat release off --suite renderer --artifact-dir artifacts\bench\renderer --html-out artifacts\bench\renderer\index.html

# Run the opt-in Bistro moving-camera stress suite
.\scripts\run_benchmarks.bat release off --suite stress --artifact-dir artifacts\bench\stress --html-out artifacts\bench\stress\index.html

# Run a diagnostic benchmark-owned Tracy capture
.\scripts\run_benchmarks.bat release cpu run --case smoke.procedural.default --tracy-diagnostic --artifact-dir artifacts\bench\tracy-smoke

# Compare a report with a saved baseline
.\scripts\run_benchmarks.bat release off compare --current artifacts\bench\current.json --baseline path\to\baseline.json --json-out artifacts\bench\comparison.json --html-out artifacts\bench\comparison.html

# Review, accept, and verify governed investigative evidence
.\scripts\run_benchmarks.bat release off baseline accept --from artifacts\bench\<run-id> --case <case-id> --profile local-nvrhi-visible --reason "reviewed evidence" --actor <name> --dry-run
.\scripts\run_benchmarks.bat release off baseline accept --from artifacts\bench\<run-id> --case <case-id> --profile local-nvrhi-visible --reason "reviewed evidence" --actor <name> --confirm-plan sha256:<digest>
.\scripts\run_benchmarks.bat release off baseline verify --case <case-id> --profile local-nvrhi-visible

# Summarize a report directory
.\scripts\run_benchmarks.bat release off summarize --reports artifacts\bench --json-out artifacts\bench\summary.json

# Build a focused graph from existing reports
.\scripts\run_benchmarks.bat release off graph --reports artifacts\bench\renderer\cases --metric cpu.render_submit_ms gpu.scopes_sum_ms --stat median p95 --html-out artifacts\bench\renderer\index.html
```

`run_benchmarks` builds the benchmark profile before running `nuri-bench`. If the first CLI argument after the build options is not a subcommand, the wrapper treats the command as `run`, so `--case smoke.procedural.default` is equivalent to `run --case smoke.procedural.default`.
Pass `--no-build` before the tool subcommand to reuse an existing configured
binary without invoking CMake or vcpkg, for example
`.\scripts\run_benchmarks.bat release --no-build off list`. The snapshot and
autotest wrappers support the same fast path; they fail clearly when the
matching `build\release-<tool>` executable is absent.
HTML graphs include all numeric metrics by default, grouped by CPU timings, GPU pass timings, render-graph pass timings, render-graph structure, process memory, benchmark PMR pools, GPU frame memory estimates, and renderer work counters. Compare reports also include a dedicated delta/status HTML view and reject incompatible run, render-graph, present-mode, settings, and build/profiling configurations unless `--force` is used. Use `--metric` on `graph` or `--html-metric` on `run`, `compare`, and `summarize` to filter to a focused subset.

## Renderer validation, snapshots, and autotests

Use the unified validator to check repository contracts, choose an affected
preset, shard CI work, or reuse an existing build:

```powershell
python scripts\nuri_validate.py check
python scripts\nuri_validate.py plan affected --affected lib\nuri\gfx\renderer.cpp
python scripts\nuri_validate.py run tool-core --junit artifacts\junit\tool-core.xml
python scripts\nuri_validate.py run tool-core --no-build --shard-index 0 --shard-count 3
```

Snapshot and multi-frame scenario examples:

```powershell
.\scripts\run_snapshots.bat release list
.\scripts\run_snapshots.bat release run --suite smoke --artifact-dir artifacts\snapshots --baseline-profile local-nvrhi-visible
.\scripts\run_snapshots.bat release capture --case smoke.procedural.final_color --window-mode hidden --artifact-dir artifacts\snapshots-hidden
.\scripts\run_autotests.bat release list
.\scripts\run_autotests.bat release run --suite smoke --artifact-dir artifacts\autotests
.\scripts\run_autotests.bat release record --case smoke.procedural.static_multiframe --out artifacts\autotest-record
.\scripts\run_autotests.bat release baseline inspect --case smoke.procedural.static_multiframe --profile local-nvrhi-visible
.\scripts\run_autotests.bat release baseline verify --case smoke.procedural.static_multiframe --profile local-nvrhi-visible
.\scripts\run_autotests.bat release baseline accept --from artifacts\autotest-record --case smoke.procedural.static_multiframe --profile local-nvrhi-visible --reason "reviewed candidate" --actor reviewer --dry-run
```

The curated smoke/correctness/stress workload map, including procedural
serial/parallel scheduling, semantic attachment/cascade captures, temporal mode
transitions, and baseline-free mode-churn soak coverage, is documented in
[`docs/renderer_tooling_coverage_matrix.md`](docs/renderer_tooling_coverage_matrix.md).
New correctness cases are candidates only; use `explain` or `capture` until a
compatible baseline has been explicitly reviewed and accepted.

On POSIX, use the corresponding `.sh` wrappers and `/` paths. The
`local-nvrhi-visible` profile is investigative: it does not establish an
authoritative visual or performance gate. Baseline approval is always an
explicit reviewed action; CI never approves baselines. Forced or dry-run work
can exit successfully while retaining `investigative` status.

Snapshot and autotest approval are two-step transactions: first run `baseline accept ... --actor
<reviewer> --reason <reason> --dry-run`, review every add/replace/remove and
hash, then rerun with the exact emitted `--confirm-plan sha256:...`. Any source
artifact change invalidates confirmation and prior approval records remain in
case-local history. `baseline inspect` is read-only and `baseline verify`
checks the reviewed plan, approval history, exact tree, links, and SHA-256
digests. Autotest `record` only creates a candidate package. Hidden-window capture uses a real hidden window-system
surface; true offscreen/headless is still reported unavailable with exit 3.

Public tool exits are `0` pass/investigative completion, `1` regression, `2`
invalid input or zero selection, `3` unavailable environment, `4` runtime or
artifact failure, and `5` missing required baseline. See
`docs/agents/tooling.md` for gate selection, artifact triage, and safe baseline
policy.

## Linux/macOS (bash)

```bash
git clone --recurse-submodules <your-repo-url> nuri
cd nuri
git submodule update --init --recursive   # safe to re-run

export VCPKG_ROOT="$HOME/vcpkg"   # adjust
./scripts/build_app.sh
./scripts/build_editor.sh
./scripts/build_lib.sh
./scripts/run_app.sh
./scripts/run_editor.sh
```

Tests:

```bash
./scripts/run_tests.sh
```

`run_tests` enables the manifest `tests` feature automatically. If you consume `nuri` as a vcpkg port instead of using this repo in manifest mode, install `nuri[tests]` before running the test targets.

Benchmarks:

```bash
# Build the benchmark CLI
./scripts/build_benchmarks.sh release off

# List available benchmark cases
./scripts/run_benchmarks.sh release off list

# Explain one case
./scripts/run_benchmarks.sh release off explain --case smoke.procedural.default

# Run one case and write a report
./scripts/run_benchmarks.sh release off --case smoke.procedural.default --json-out artifacts/bench/smoke_live.json

# Run a suite and also write an all-metrics graph view
./scripts/run_benchmarks.sh release off --suite renderer --artifact-dir artifacts/bench/renderer --html-out artifacts/bench/renderer/index.html

# Run the opt-in Bistro moving-camera stress suite
./scripts/run_benchmarks.sh release off --suite stress --artifact-dir artifacts/bench/stress --html-out artifacts/bench/stress/index.html

# Run a diagnostic benchmark-owned Tracy capture
./scripts/run_benchmarks.sh release cpu run --case smoke.procedural.default --tracy-diagnostic --artifact-dir artifacts/bench/tracy-smoke

# Compare a report with a saved baseline
./scripts/run_benchmarks.sh release off compare --current artifacts/bench/current.json --baseline path/to/baseline.json --json-out artifacts/bench/comparison.json --html-out artifacts/bench/comparison.html

# Review, accept, and verify governed investigative evidence
./scripts/run_benchmarks.sh release off baseline accept --from artifacts/bench/<run-id> --case <case-id> --profile local-nvrhi-visible --reason "reviewed evidence" --actor <name> --dry-run
./scripts/run_benchmarks.sh release off baseline accept --from artifacts/bench/<run-id> --case <case-id> --profile local-nvrhi-visible --reason "reviewed evidence" --actor <name> --confirm-plan sha256:<digest>
./scripts/run_benchmarks.sh release off baseline verify --case <case-id> --profile local-nvrhi-visible

# Summarize a report directory
./scripts/run_benchmarks.sh release off summarize --reports artifacts/bench --json-out artifacts/bench/summary.json

# Build a focused graph from existing reports
./scripts/run_benchmarks.sh release off graph --reports artifacts/bench/renderer/cases --metric cpu.render_submit_ms gpu.scopes_sum_ms --stat median p95 --html-out artifacts/bench/renderer/index.html
```

`run_benchmarks` builds the benchmark profile before running `nuri-bench`. If the first CLI argument after the build options is not a subcommand, the wrapper treats the command as `run`, so `--case smoke.procedural.default` is equivalent to `run --case smoke.procedural.default`.
HTML graphs include all numeric metrics by default, grouped by CPU timings, GPU pass timings, render-graph pass timings, render-graph structure, process memory, benchmark PMR pools, GPU frame memory estimates, and renderer work counters. Compare reports also include a dedicated delta/status HTML view and reject incompatible run, render-graph, present-mode, settings, and build/profiling configurations unless `--force` is used. Use `--metric` on `graph` or `--html-metric` on `run`, `compare`, and `summarize` to filter to a focused subset.

## Notes

- Target-specific scripts configure a minimal build tree per mode and target set. Debug `app` uses `build/`, other Debug profiles use `build_<target>/`, and Release profiles use `build_release/<target>/`.
- `build_debug`/`build_release` remain as compatibility wrappers and default to the `app` profile.

## Project layout (high level)

- `app/` sample application using engine APIs
- `lib/nuri/platform/` platform-facing abstractions (Window, GPUDevice)
- `lib/nuri/gfx/` backend-neutral renderer and pipeline/shader front-end
- `lib/nuri/resources/` CPU/GPU asset types
- `external/nvrhi/` NVRHI submodule
