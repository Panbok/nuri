#pragma once

#include "nuri/defines.h"
#include "nuri/gfx/gpu_render_types.h"

#include <cstdint>
#include <span>

namespace nuri {

struct NURI_API AnimatedRenderableGeometryOverride {
  BufferHandle vertexBuffer{};
  uint64_t vertexByteOffset = 0u;
  uint32_t vertexCount = 0u;
  bool enabled = false;
};

struct NURI_API AnimationSceneFrameData {
  BufferHandle instanceMatricesBuffer{};
  uint64_t instanceMatricesAddress = 0u;
  std::span<const ComputeDispatchItem> preDispatches{};
  std::span<const AnimatedRenderableGeometryOverride>
      geometryOverridesByRenderable{};
  uint64_t version = 0u;
};

} // namespace nuri
