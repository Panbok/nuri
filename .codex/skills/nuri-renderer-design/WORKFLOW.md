# Renderer refactor workflow

## 1. Establish evidence

- Record branch, commit, dirty files, backend, build flags, driver/GPU, profile,
  resolution, fixed delta, present mode, warmup, frames, and repetitions.
- Validate manifests with `python scripts/nuri_validate.py check`.
- Build only through `scripts/` wrappers.
- Capture the smallest relevant autotest and snapshot before editing.
- Run release profiling-off benchmark repetitions. Use benchmark-owned Tracy only to
  explain hotspots/regressions.

## 2. Audit every file without forcing every file to change

For each file/group, record:

- domain responsibility and callers;
- owned versus borrowed state and destruction/completion rule;
- allocations, locks, virtual calls, strings, validation, and handle churn;
- duplicated facts/paths and competing authorities;
- interface facts callers must know;
- actual adapters/variation points;
- applicable test seam and evidence artifact;
- keep, deepen, merge, split, move-private, or delete decision.

Apply the deletion test to shallow modules. Coverage means every file is accounted
for; indiscriminate edits are anti-compression.

## 3. Instrument likely hitches

Measure at minimum:

- lazy pipeline-variant count/time;
- synchronous upload waits/bytes;
- framebuffer and command-list creates/reuses;
- graph barriers, native barriers, and dependency resources scanned;
- recording-context capacity/reuse and lock wait/hold time;
- per-range encode time and draw/dispatch counts.

## 4. Refactor order

1. Fix proven correctness/lifetime bugs with regression tests.
2. Remove frame hitches: pipeline compilation, blocking uploads, hot owning-handle
   copies, and non-reused command/framebuffer objects.
3. Establish render graph as the one state authority.
4. Centralize generational handles, slot maps, recording/submission lifecycle,
   retirement, and immutable capabilities.
5. Replace broad invalid-combination records with typed compiled-pass variants.
6. Privatize concrete adapters, remove redundant PIMPL objects where justified, and
   split backend source by lifecycle.
7. Add semantic validation at creation/compile/debug layers.
8. Separate presentation, uploads, geometry, frame encoding, and core device ownership.

Each item is a vertical slice with tests and compatible before/after evidence. Do not
combine state-authority, lifetime, backend representation, and pipeline-cache changes
in one unreviewable patch.

## 5. Verification ladder

1. Focused unit/regression tests.
2. `python scripts/nuri_validate.py run affected --affected <path>`.
3. Relevant Nuri autotest case/suite.
4. Relevant snapshot capture/compare; never accept baselines implicitly.
5. Same-config isolated release benchmark repetitions.
6. Tracy diagnostic only when measurements need explanation.

Stop and report when the visual baseline was already invalid, the profile is
incompatible, required metrics are missing, warmup is unstable, or GPU completion
cannot be proven. Investigative evidence may guide work but cannot support an
authoritative claim.
