---
name: nuri-renderer-design
description: Applies Nuri's renderer architecture, semantic-compression, N+1, RHI, GPU-lifetime, and measured-performance rules. Use when reviewing or refactoring `lib/nuri`, changing GPU backends, render-graph state, resource ownership, command recording, renderer hot paths, or renderer-facing interfaces.
---

# Nuri Renderer Design

## Quick start

1. Read root `CONTEXT.md`, relevant `docs/adr/`, and `AGENTS.md`.
2. Define visual, behavioral, lifetime, and performance invariants before editing.
3. Baseline the smallest relevant autotest/snapshot and a release benchmark.
4. Trace concrete call paths and find at least two real examples before compressing.
5. Refactor one vertical slice, run the cheapest sufficient gate, then measure again.

Read [PRINCIPLES.md](PRINCIPLES.md) for Casey Muratori, Ryan Fleury, and project
N+1 rules. Read [NVRHI_PATTERNS.md](NVRHI_PATTERNS.md) for RHI decisions. Use
[WORKFLOW.md](WORKFLOW.md) for comprehensive audits and migration sequencing.

## Core rules

- Compress observed repetition; do not design reuse from zero or one example.
- Preserve continuous granularity: high-level helpers must have usable lower-level
  equivalents without forcing a rewrite.
- Prefer a few deep modules with small interfaces and strong locality.
- A seam needs real variation. One adapter is hypothetical; two adapters are real.
- Apply the project N+1 test at module interfaces: adding one backend, feature,
  pass kind, resource state, or quality mode must not rewrite unrelated callers.
- N+1 is not permission to generalize per-draw/per-dispatch hot loops.
- Collapse equivalent cases into shared codepaths; make invalid states
  unrepresentable with typed/tagged records where practical.
- Use one authoritative representation for each fact and lower it once.

## Hot-path and data rules

- Prefer contiguous batches, SoA when measured useful, fixed-capacity arrays for
  GPU-bounded collections, and dense generational slot maps.
- No heap allocation, owning-handle/refcount churn, string construction, blocking
  locks, or avoidable virtual dispatch per draw/dispatch.
- Use PMR and `ScratchArena`/`ScopedScratch` for scoped transient work. Scratch data
  never escapes, and `ScopedScratch` is never nested on one arena.
- Use ZII for reusable transient records: reset to zero or explicit invalid
  sentinels. Never `memset` non-trivial objects.
- Keep `Result` at fallible seams; validate at creation/compile/debug layers and use
  `NURI_ASSERT` for already-proven hot-loop invariants.
- Optimize only measured bottlenecks. Release `nuri-bench` profiling-off is the
  oracle; benchmark-owned Tracy traces explain regressions.

## RHI and lifetime rules

- Copy NVRHI invariants, not its class hierarchy or atomic ownership model.
- Render graph owns transient resource state and barriers. Encode exact usage,
  stages/queues when needed, and subresource ranges; batch barrier commits.
- Resource descriptors are structural. Upload, mip generation, and readback are
  commands with explicit synchronization.
- Logical destruction invalidates a handle immediately. Physical destruction and
  pool reuse wait for a proven GPU completion token/timeline.
- Recording contexts retain compact referenced-resource lists through submission
  completion. Reuse command contexts, framebuffers, queries, staging, and scratch
  chunks only after completion.
- Pipelines are immutable and prebuilt; compiling a new variant during frame
  encoding is forbidden.
- Capabilities are immutable typed data, not perpetual vtable growth or `void*`.
- Third-party types stay in private backend headers. Concrete adapters need not be
  public. Never downcast through one abstraction to reach another backend.

## PIMPL decision

- Remove PIMPL when state is small/stable or a private concrete backend header can
  own it directly with better locality.
- Keep an opaque compile seam when exposing third-party or volatile backend state
  would materially couple public headers.
- Do not replace PIMPL with a function table plus `void*`, opaque inline storage, or
  a complete-backend variant without a measured, simpler result.
- Treat PIMPL removal as a locality/build decision until benchmarks prove runtime
  impact.

## Validation

Use `nuri-autotests`, `nuri-snapshots`, and `nuri-benchmarks` skills. Renderer stress
defaults to Niagara Bistro rapid wide-area motion with GTAO Ultra, shadows Ultra,
and the selected Ultra AA/TAA mode. Never approve baselines without explicit user
authorization.
