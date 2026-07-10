---
name: nuri-snapshots
description: Nuri renderer visual snapshot workflow. Use when validating renderer visual correctness, capture points, image baselines, snapshot manifests, or the nuri-snapshot CLI.
---

# Nuri Visual Snapshots

Use this skill when working on renderer visual snapshot cases, capture points, baselines, or the `nuri-snapshot` CLI.
Use snapshots as the visual correctness gate around performance work; use `nuri-bench` for frame-time and baseline performance data.
For shared exit semantics, validation presets, artifact triage, and baseline safety, use `docs/agents/tooling.md` as the canonical guide.

## Build And Run

- Build the tool with `scripts/build_snapshots.bat release` on Windows or `scripts/build_snapshots.sh release` on POSIX.
- Run the tool with `scripts/run_snapshots.bat release <command>` or `scripts/run_snapshots.sh release <command>`.
- Reuse an existing configured binary with `--no-build` before the subcommand,
  for example `scripts/run_snapshots.sh release --no-build list`.
- Do not run `cmake` directly; route through the scripts so the vcpkg features, build profile, and dev-check defaults stay consistent.
- Release snapshot builds default to `NURI_DEV_CHECKS=OFF`. Pass `devchecks` explicitly when validating debug instrumentation paths.
- For renderer perf changes, run `nuri-snapshot run` after the benchmark pass to catch visual regressions. Do not approve new visual baselines unless the user explicitly asks for baseline approval.

## Common Commands

```bash
scripts/run_snapshots.sh release list
scripts/run_snapshots.sh release explain --case smoke.procedural.final_color
scripts/run_snapshots.sh release capture --case smoke.procedural.final_color --artifact-dir artifacts/snapshots
scripts/run_snapshots.sh release compare --case smoke.procedural.final_color --artifact-dir artifacts/snapshots --baseline-profile local-nvrhi-visible
scripts/run_snapshots.sh release run --suite smoke --artifact-dir artifacts/snapshots --baseline-profile local-nvrhi-visible
scripts/run_snapshots.sh release baseline inspect --case smoke.procedural.final_color --profile local-nvrhi-visible
scripts/run_snapshots.sh release baseline verify --case smoke.procedural.final_color --profile local-nvrhi-visible
scripts/run_snapshots.sh release baseline accept --case smoke.procedural.final_color --from artifacts/snapshots --profile local-nvrhi-visible --reason "reviewed change" --actor "<reviewer>" --dry-run
scripts/run_snapshots.sh release baseline accept --case smoke.procedural.final_color --from artifacts/snapshots --profile local-nvrhi-visible --reason "reviewed change" --actor "<reviewer>" --confirm-plan sha256:<reviewed-plan>
```

## Capture Points

- Source capture catalog: `tools/snapshot/src/snapshot_capture_point.cpp`.
- Renderer publications live in frame providers/features/renderers. Capture points are request-driven through `RenderFrameContext::captureRequests` and published through `RenderFrameContext::captureRegistry`.
- First-slice capture points include `final_color`, `scene_color_hdr`, `frame_color_hdr`, `scene_depth`, `material_normals`, `motion_vectors`, `reactive_mask`, `ambient_occlusion`, `shadow_cascade_0` through `shadow_cascade_3`, `shadow_preview`, `transmission_visibility_depth`, and `transmission_feedback` variants.
- Known-not-capturable catalog entries such as `gtao_edges` should remain listed for diagnostics, but manifests must reject them until implementation publishes them.
- The `correctness` suite contains procedural typed-attachment and all-cascade
  candidates. `capture` is valid before baselines exist; `run --suite
  correctness` requires reviewed compatible baselines and must report missing
  baseline until then.

## Baselines

- Default baseline root: `tools/baselines/render/<profile>/<suite>/<case-id>/`.
- Approval is permitted only from a complete compatible run after reviewing its dry-run plan; it requires a non-empty reason, actor, and exact confirmation digest. Artifact changes after review invalidate the plan.
- Each approved case carries its prior approval/history transactionally. Never edit or copy the baseline directory by hand.
- Keep baseline profile names explicit, for example `local-nvrhi-visible`, because GPU backend, driver, OS, and window mode affect visual output.
- `local-nvrhi-visible` and forced comparisons are investigative and cannot produce an authoritative pass.

## Tests

- No-GPU snapshot tests live in `tests/src/nuri_snapshot_testing_tests.cpp`.
- Prefer exercising manifest validation, catalog behavior, fake texture readback, report JSON, HTML escaping, and deterministic image compare without requiring a visible GPU window.
- Hidden-window execution is available where GLFW can create a window-system surface. True offscreen/headless remains explicitly unavailable rather than falling back to a hidden window.
