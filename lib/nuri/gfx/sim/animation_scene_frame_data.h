#pragma once

#include "nuri/defines.h"
#include "nuri/gfx/gpu_render_types.h"

#include <cstdint>
#include <span>

namespace nuri {

// Optional per-renderable animated geometry override produced for a frame.
// When enabled, the renderer uses vertexBuffer instead of the model's default
// animated vertex stream for that renderable.
struct NURI_API AnimatedRenderableGeometryOverride {
  // GPU buffer containing packed animated vertex data for the override.
  BufferHandle vertexBuffer{};
  // Byte offset into vertexBuffer where the override vertex data begins.
  uint64_t vertexByteOffset = 0u;
  // Number of vertices available from vertexBuffer starting at
  // vertexByteOffset.
  uint32_t vertexCount = 0u;
  // Applies the override only when true; false means use the model geometry.
  bool enabled = false;
};

// Non-owning GPU frame view for animation scene submission.
// The spans reference arrays owned elsewhere, typically frame-local storage or
// higher-level frame resources that outlive this view for the frame. Treat the
// buffer handle and referenced arrays as immutable once published.
struct NURI_API AnimationSceneFrameData {
  // Storage buffer handle containing per-instance matrices for the frame.
  BufferHandle instanceMatricesBuffer{};
  // Device address for instanceMatricesBuffer, in bytes, expected to be
  // non-zero.
  uint64_t instanceMatricesAddress = 0u;
  // Non-owning span of pre-dispatch compute work. The backing array must stay
  // alive until the frame graph finishes consuming this
  // AnimationSceneFrameData.
  std::span<const ComputeDispatchItem> preDispatches{};
  // Non-owning span indexed by runtime renderable index. The backing array must
  // outlive this view, usually via frame-local storage kept alive by frame
  // resources.
  std::span<const AnimatedRenderableGeometryOverride>
      geometryOverridesByRenderable{};
  // Monotonic version for the published frame data payload.
  uint64_t version = 0u;
};

} // namespace nuri
