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
  `scripts/run_benchmarks.bat release cpu-gpu run --case <id> --tracy-diagnostic --artifact-dir artifacts/bench/<run-id>`.
- Visual correctness is still separate: run `nuri-snapshot run` after renderer
  changes. Do not approve visual baselines unless explicitly asked.

## Policy

- Treat `gpu.scopes_sum_ms` as a pass-scope sum, not a full GPU frame time. It
  excludes untimed work, queue gaps, and present.
- Prefer required benchmark metrics for gates: `cpu.render_submit_ms`,
  `gpu.scopes_sum_ms`, relevant `gpu.scopes.*`, and relevant
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
