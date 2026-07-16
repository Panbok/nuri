# Nuri renderer context

## Purpose

Nuri is a correctness-first, performance-critical renderer implemented on NVRHI.
The renderer exposes a small number of deep, data-oriented modules. NVRHI details
remain private, while higher layers operate on Nuri descriptors, typed handles,
compiled frame plans, and submission tokens.

The governing renderer rules live in
`.codex/skills/nuri-renderer-design/SKILL.md`. The current migration is specified
in `docs/renderer_refactor_plan.md` and
`docs/adr/0001-renderer-rhi-and-lifetime-architecture.md`.

## Domain vocabulary

- **Render settings**: authored/user-facing choices before capability resolution.
- **Resolved render settings**: immutable, sanitized settings for one frame.
- **Frame scene view**: immutable dense renderer-facing view of scene state.
- **Scene draw database**: persistent SoA metadata used by opaque, shadow,
  transparent, transmission, picking, and visibility preparation.
- **Frame resources**: named current-frame and history resources shared by stages.
- **Render stage**: one prepared contribution to a compiled frame plan.
- **Render graph plan**: structural pass order, resource use, barriers, lifetimes,
  queues, and submission dependencies. It excludes changing frame payload values.
- **Frame command arena**: current-frame draw, dispatch, copy, push-constant, and
  late-binding payload storage.
- **Resource use**: canonical exact use of a texture or buffer by a stage,
  including access, state, subresource range, and queue/stage constraints.
- **Recording context**: reusable worker/queue-owned backend encoder state.
- **Submission token**: backend-neutral proof that recorded GPU work was submitted.
- **Completed token**: proof that a submission and every earlier ordered use have
  completed on the relevant queue.
- **Logical destruction**: immediate invalidation of a public generational handle.
- **Physical retirement**: native destruction/reuse after the last referencing
  submission is proven complete.
- **Persistent resource**: application-owned resource spanning graph executions.
- **Transient resource**: graph-owned resource whose lifetime is derived from a
  compiled plan and may alias compatible allocations.
- **Pipeline library**: canonical immutable pipeline and variant ownership outside
  frame encoding.
- **Validation layer**: optional debug/contract validation before hot lowering.
- **NVRHI implementation**: private concrete lowering of Nuri operations to
  NVRHI. It is not a public renderer domain type.

## Authoritative representations

Each fact has one owner:

- Render graph owns transient state transitions and barrier plans.
- Resource records own canonical normalized structural descriptors.
- Frame command arenas own changing pass payloads.
- Submission timelines own completion truth.
- Pipeline library owns immutable pipeline variants.
- Resolved settings own effective per-frame feature configuration.
- Frame scene view owns the renderer-facing scene snapshot.
- Device capabilities are an immutable typed snapshot.

Backends lower these facts; they do not infer a second competing version.

## Target module boundaries

```text
Application loop
  -> RuntimeComposition
     -> Window / Presentation
     -> Renderer
        -> ResolvedRenderSettings
        -> FrameSceneView + SceneDrawDatabase
        -> RenderStage table
        -> RenderGraphPlan + FrameCommandArena
        -> FrameEncoder
     -> GPUDevice resources + DeviceCaps
     -> UploadQueue
     -> GpuRetirementQueue
     -> PipelineLibrary
        -> private NVRHI implementation
```

These are deep modules, not one interface per method. The backend seam is crossed
at resource setup, upload batches, compiled recording ranges, and submission—not
through extra ownership or allocation for each draw.

## Non-negotiable invariants

### Correctness

- No owning GPU resource is destroyed or reused based on guessed frame lag.
- Logical handles become invalid immediately; native resources retire by proven
  submission completion.
- A graph fingerprint changes only for structural changes.
- Cached plans refresh all current-frame payload and imported handles.
- Invalid pass/command combinations are represented by distinct typed records or
  rejected before backend lowering.
- Visual baselines are never changed without explicit approval.

### Hot paths

- No pipeline compilation, blocking wait, heap allocation, string construction,
  owning-refcount churn, or avoidable virtual dispatch per draw/dispatch.
- Recording contexts, command lists, framebuffers, queries, upload pages, and
  scratch storage are reused only after completion.
- Data is contiguous and batched; GPU-bounded collections use fixed capacity.
- Validation happens at creation/compile/debug layers. Proven encoder invariants
  use assertions.

### N+1 and compression

- Adding one backend, pass kind, texture use, material extension, light kind, or
  frame texture changes one descriptor/registration plus its real implementation,
  not unrelated callers.
- Compression follows at least two concrete uses and removes relationships or
  states, not merely lines.
- One adapter is not proof that an extensibility seam is useful.
- Public/module boundaries and lifecycle/concurrency logic remain explicit.

## Verification contract

- Unit/contract tests prove algorithms, lifecycle, cache, and state invariants.
- `nuri-autotest` proves deterministic multi-frame behavior.
- `nuri-snapshot` proves visual equivalence; invalid or incompatible stored
  baselines are reported, never silently accepted.
- Profiling-off release `nuri-bench` provides comparison numbers.
- Benchmark-owned Tracy diagnostics explain measured regressions and hitches.
- The canonical stress workload is the full Niagara Bistro rapid route with
  Reference TAA Ultra, GTAO Ultra, and shadows Ultra.

## Current implementation constraints

- NVRHI is the sole GPU implementation; there is no runtime backend selection.
- The render graph's submission-based transient retirement is the proven lifetime
  model to generalize.
- Existing microbenchmarks remain isolated; the full-pipeline route supplements
  rather than replaces them.
- Large modules are split through verified vertical slices, not a flag-day rewrite.
