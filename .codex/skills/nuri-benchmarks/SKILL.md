---
name: nuri-benchmarks
description: Primary Nuri renderer performance workflow. Use when measuring renderer performance, running perf passes, creating or comparing benchmark baselines, investigating frame-time regressions, collecting benchmark-owned Tracy diagnostic artifacts, or updating benchmark cases/reports/HTML output.
---

# Nuri Benchmarks

For shared exit semantics, validation presets, artifact triage, and baseline safety, use `docs/agents/tooling.md` as the canonical guide.

Use `nuri-bench` as the primary performance oracle for renderer perf work.
Tracy is a benchmark diagnostic artifact, not a separate first-choice workflow.

## Commands

- Build the benchmark tool with `scripts/build_benchmarks.bat release off` on
  Windows or `scripts/build_benchmarks.sh release off` on Unix-like systems.
- Run the tool through `scripts/run_benchmarks.bat release ...` or
  `scripts/run_benchmarks.sh release ...`.
- After a suitable binary exists, put `--no-build` before the subcommand for
  repeated metadata/report work, for example
  `scripts/run_benchmarks.sh release --no-build off list`.
- Use `nuri-bench list`, `nuri-bench explain --case <id>`,
  `nuri-bench run --case <id>`, `nuri-bench compare`, and
  `nuri-bench summarize` as the stable non-interactive surface.
- Use `nuri-bench check --case <id>|--suite <name> --profile <profile>` for
  the canonical governed gate. It verifies every accepted baseline before
  running, executes the profile-owned isolated repetition count, and compares
  with baseline-owned thresholds in one portable invocation workspace.
- Add `--repetitions N` to `run` only when independent process observations
  are needed. Each repetition owns a child report and captured stdout/stderr
  under the parent run bundle. `--repetition-timeout-ms` bounds each child.
  Omitting `--repetitions` keeps the efficient single-process investigative
  path; authoritative profiles require an explicit count meeting their policy.
- Inspect governed evidence with
  `nuri-bench baseline inspect --case <id> --profile <profile>` and verify its
  report, approval, and immutable history with `baseline verify`.
- Acceptance is always two-step: review
  `baseline accept --from <run-root> --case <id> --profile <profile> --reason
  <reason> --actor <actor> --dry-run`, then repeat with the emitted
  `--confirm-plan sha256:...`. Dry-run never mutates baselines.
- Use `nuri-bench graph --reports <dir> --html-out <path>` for a standalone
  self-contained HTML graph view. `run`, `compare`, and `summarize` also accept
  `--html-out`, plus optional `--html-metric` and `--html-stat` filters.
- For a diagnostic trace, build/run with Tracy enabled and ask the benchmark to
  own the capture:
  `scripts/run_benchmarks.bat release cpu run --case <id> --tracy-diagnostic --artifact-dir artifacts/bench/<run-id>`.
  The `cpu` profiling mode enables both CPU and Vulkan GPU Tracy instrumentation.
  Render-graph passes appear as GPU zones on the `Nuri Graphics` context and as
  Vulkan debug-marker ranges for external GPU profilers.
- Visual correctness is still separate: run `nuri-snapshot run` after renderer
  changes. Do not approve visual baselines unless explicitly asked.

## GPU Evidence

- Profiling-off release `nuri-bench` remains the performance oracle. Use
  `gpu.frame_ms` for the graphics-queue frame envelope, relevant
  `gpu.scopes.*` metrics for feature totals, and
  `rendergraph.pass.*.gpu_ms` for pass-level regression localization.
- `gpu.scopes_sum_ms` is not whole-frame GPU time: it excludes untimed work,
  queue gaps, present, and child scopes whose parent is already present.
- A Tracy diagnostic is intentionally non-authoritative because its timestamp
  queries and debug markers perturb the workload. Use it after a profiling-off
  regression is reproduced.
- Start with the benchmark JSON. Under `tracy`, inspect `tracePath`,
  `gpuEventsExportSupported`, `gpuZoneEventCount`, `gpuZones`, and
  `gpuEventsCsvPath`. GPU zone rows report total, mean, P50, P95, maximum, and
  event count. The HTML report shows the same table.
- The raw `.tracy` trace can contain GPU pass zones even when the installed
  `tracy-csvexport` is too old to support `-g`. In that case the report records
  `gpuEventsExportSupported: false`; open the trace in the matching Tracy GUI
  and do not claim that GPU CSV aggregation was collected.
- The benchmark-tools vcpkg feature installs `tracy-capture` and
  `tracy-csvexport`. Keep Tracy capture/client/exporter versions matched. A
  missing capture tool is an unavailable diagnostic environment, not a renderer
  regression.

## Shader And Vendor Diagnostics

- Tracy answers **which pass is slow and when it executes**. It does not explain
  shader occupancy, register pressure, cache behavior, divergence, or stall
  reasons.
- On NVIDIA, open a Tracy-identified pass in Nsight Graphics Shader Profiler.
  Use the Vulkan pass marker to correlate it, then inspect source/ISA hotspots,
  registers per thread, theoretical occupancy, instruction mix, and stall
  reasons.
- On AMD, use Radeon GPU Profiler for queue timing, barriers, wave occupancy,
  and instruction timing. Use Radeon GPU Analyzer for offline GLSL/SPIR-V ISA,
  VGPR/SGPR, and LDS analysis.
- RenderDoc is a correctness and frame-inspection tool here, not the authority
  for frame-time or shader-throughput conclusions.
- Vendor counters, Vulkan pipeline statistics, static SPIR-V instruction counts,
  and pipeline executable properties are diagnostic and device/driver specific.
  Never use them as portable baseline gates. Normalize shader experiments by
  unchanged workload counts where possible and pin GPU, driver, build, settings,
  resolution, and power/clock policy.
- Pair a vendor capture with the owning real-renderer benchmark case. A synthetic
  shader microbenchmark may answer a focused algorithm question, but cannot
  replace the full pass because bandwidth, cache, occupancy, and scheduling
  interactions differ.

## Policy

- Prefer required benchmark metrics for gates: `cpu.render_submit_ms`,
  `gpu.frame_ms`, `gpu.scopes_sum_ms`, relevant `gpu.scopes.*`, and relevant
  `rendergraph.pass.*.{cpu_ms,gpu_ms}` rows.
- Renderer benchmark cases should use the project perf profile unless a case is
  intentionally a smoke exception: GTAO Ultra, Shadows enabled Ultra, TAA Ultra.
- Use Damaged Helmet for simple renderer checks and the `stress` suite's
  Niagara Bistro moving-camera cases for heavy stress testing.
- Use the paired `stress.procedural.reactive_taa_{serial,parallel}_360p`
  cases only for controlled scheduling investigation. Their manifests add no
  calibrated gate and remain non-authoritative until independent repetitions
  and profile-owned policy are complete.
- Do not compare reports as authoritative unless profile fields match:
  benchmark case, resolution, run frame counts, fixed delta, present mode,
  render-graph config, settings/config signatures, backend, build type, tool
  profile, build/profiling flags, Tracy flags, and dev-check flags.
- Missing required metrics make comparison invalid unless `--force` is used for
  investigation.
- In-process sample windows are not independent repetitions. Authoritative
  profile gates require the profile-owned minimum isolated-process repetitions,
  measured stable warmup, and complete registered required metrics. Warmup
  compares median `cpu.render_submit_ms` across two non-overlapping
  profile-sized windows against the profile-owned drift limit. Insufficient
  frames are `unknown`; otherwise reports remain investigative with explicit
  blockers.
- Local smoke runs may use a visible swapchain window; reports must say so.
- Store accepted benchmark baselines under
  `tools/baselines/benchmark/<profile>/<suite>/<case>.json`. Keep raw run
  artifacts under `artifacts/bench/...`.
- Acceptance preserves an existing baseline's gate thresholds. The local
  profile stores investigative evidence only and never becomes authoritative.
