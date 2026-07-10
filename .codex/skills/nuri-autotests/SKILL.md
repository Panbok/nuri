---
name: nuri-autotests
description: Nuri renderer scenario workflow. Use when validating deterministic multi-frame renderer behavior, checkpoint captures, readouts, metric windows, record packages, or the nuri-autotest CLI.
---

# Nuri Renderer Autotests

Use this skill when working on renderer autotest manifests, checkpoint reports, record packages, or the `nuri-autotest` CLI.
For shared exit semantics, validation presets, artifact triage, and baseline safety, use `docs/agents/tooling.md` as the canonical guide.

## Build And Run

- Build the tool with `scripts/build_autotests.bat release` on Windows or `scripts/build_autotests.sh release` on POSIX.
- Run the tool with `scripts/run_autotests.bat release <command>` or `scripts/run_autotests.sh release <command>`.
- Reuse an existing configured binary with `--no-build` before the subcommand,
  for example `scripts/run_autotests.sh release --no-build list`.
- Do not run `cmake` directly; route through scripts so vcpkg features, build profiles, and dev-check defaults stay consistent.

## Common Commands

```bash
scripts/run_autotests.sh release list
scripts/run_autotests.sh release explain --case smoke.procedural.static_multiframe
scripts/run_autotests.sh release run --case smoke.procedural.static_multiframe --json-out artifacts/autotests/report.json
scripts/run_autotests.sh release run --suite smoke --artifact-dir artifacts/autotests-smoke
scripts/run_autotests.sh release record --case smoke.procedural.static_multiframe --out artifacts/autotest-record
scripts/run_autotests.sh release baseline inspect --case smoke.procedural.static_multiframe --profile local-nvrhi-visible
scripts/run_autotests.sh release baseline verify --case smoke.procedural.static_multiframe --profile local-nvrhi-visible
scripts/run_autotests.sh release baseline accept --from artifacts/autotest-record --case smoke.procedural.static_multiframe --profile local-nvrhi-visible --reason "reviewed candidate" --actor reviewer --dry-run
scripts/run_autotests.sh release baseline accept --from artifacts/autotest-record --case smoke.procedural.static_multiframe --profile local-nvrhi-visible --reason "reviewed candidate" --actor reviewer --confirm-plan sha256:<reviewed-digest>
```

## Review Workflow

- Inspect JSON reports first: failed checkpoints, capture target statuses, assertion IDs, metric values, and artifact paths are the stable interface.
- Open generated HTML when visual review is needed.
- Summarize failures by checkpoint, capture target, assertion ID, metric delta, and artifact path.
- `record` creates a candidate package; it never updates a baseline by itself.
- Baseline acceptance requires the exact digest from a reviewed dry-run plan.
  Use `baseline verify` to check plan, approval, history, tree identity, links,
  and every file digest after promotion.
- Treat `local-nvrhi-visible`, dry runs, and forced incompatible work as investigative rather than authoritative passes.
- Preserve user changes and avoid broad baseline updates.
- Use `correctness.procedural.temporal_mode_transitions` for the compact
  camera-cut/history/AA transition path. It needs reviewed checkpoint baselines
  before it can gate.
- Use `stress.procedural.parallel_mode_churn` for longer baseline-free metric
  and scheduling soak. Its diagnostic captures intentionally use
  `compare: false`; performance conclusions still belong to `nuri-bench`.
