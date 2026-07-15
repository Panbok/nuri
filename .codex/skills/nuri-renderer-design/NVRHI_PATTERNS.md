# NVRHI patterns for Nuri

Use the vendored `external/nvrhi` source as implementation evidence. Copy mature
invariants while filtering patterns through Nuri's generational handles, ZII, PMR,
and direct-hot-loop requirements.

## Adopt

- Canonical descriptors retained with resource records (`include/nvrhi/nvrhi.h`).
- Explicit resource states and subresource ranges.
- Fixed-capacity storage for GPU-limited attachment/state collections
  (`include/nvrhi/common/containers.h`).
- Command-list-owned state caches and batched barrier commits
  (`src/common/state-tracking.*`, `src/vulkan/vulkan-commandlist.cpp`).
- Timeline submission IDs and completion-gated command/resource retirement
  (`src/vulkan/vulkan-queue.cpp`).
- Reusable upload/scratch chunks with linear suballocation and completion proof
  (`src/vulkan/vulkan-upload.cpp`).
- Immutable pipelines and explicit pipeline/framebuffer compatibility.
- Validation as an optional layer with explicit recording lifecycle states
  (`src/validation/*`).
- Private concrete backend headers split into device, resources, pipelines,
  commands/state, submission/lifetime, and timing/readback source modules.

## Do not copy

- Atomic intrusive ownership for Nuri's public resource handles.
- Huge virtual command interfaces or per-draw virtual dispatch.
- Owning strings/vectors in every hot descriptor.
- Per-resource heap maps for common state tracking.
- Default-uninitialized records, broad `void*` native escape hatches, exception-led
  error handling, or handwritten pass-through validation of the entire interface.

## Current Nuri gaps

- `gfx/gpu_device.h` mixes window/presentation, resources, capabilities, geometry,
  uploads, recording, submission, timing, readback, and temporal integration.
- `gpu_types.h` manually repeats handle structs; prefer `Handle<Tag>` while retaining
  generational stale-use checks.
- `TextureUsage` encodes combination enumerants instead of orthogonal flags.
- `RenderPipelineDesc` has both a format vector and attachment count.
- `TextureDesc`/`BufferDesc` mix structural creation with upload policy/data.
- Render graph barriers, adapter dependency inference, and NVRHI automatic barriers
  are competing state authorities.
- Barrier records are too coarse and transition all texture subresources.
- Both adapters duplicate recording-context, submission, timing, validation,
  geometry, and resource-table plumbing.
- NVRHI adapter creates command-list objects per acquired context and locks a global
  graphics mutex across pass encoding.
- NVRHI owning handles are copied in draw encoding. Do not assume those copies are
  material: a 2026-07-15 raw borrowed-pointer experiment regressed the local route
  benchmark and was reverted. Revisit only with clean interleaved measurement and
  a recording-context retention design that proves pointer lifetime.
- Device-local NVRHI updates submit and wait synchronously.
- Pipeline variants may be created during draw recording.
- Several upper layers infer GPU completion from frame lag rather than a completion
  token.

## Target deep modules

- `GPUDevice`: resource ownership and immutable capabilities.
- `Presentation`: window/swapchain acquire, resize, mode, present output.
- `FrameEncoder`: validated compiled-range recording and submission.
- `UploadQueue`: chunked asynchronous transfer ownership.
- `GeometryPool`: common upper-layer geometry allocation.
- `RetirementQueue`: logical invalidation plus completion-gated physical release.

Do not split these into pass-through interfaces. Each module must own a coherent
lifecycle and hide substantial implementation.

## Backend representation

Prefer factory-only exposure for `LvkGPUDevice` and `NvrhiGPUDevice`. Define concrete
final adapters and direct state in private backend headers, then split their source by
domain like NVRHI. This removes redundant public declarations/PIMPL objects while
preserving third-party dependency hiding. It is primarily a locality and build-shape
change; benchmark runtime separately.
