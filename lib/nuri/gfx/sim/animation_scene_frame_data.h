#pragma once
#include "nuri/defines.h"
#include "nuri/gfx/gpu_render_types.h"
#include <cstddef>
#include <cstdint>
#include <span>
namespace nuri {

class RenderScene;

struct NURI_API AnimatedRenderableGeometryOverride {
  BufferHandle vertexBuffer{};
  uint64_t vertexByteOffset = 0u;
  uint32_t vertexCount = 0u;
  bool enabled = false;
};

struct NURI_API AnimationSceneFrameData {
  BufferHandle instanceMatricesBuffer{};
  uint64_t instanceMatricesAddress = 0u;
  BufferHandle previousInstanceMatricesBuffer{};
  uint64_t previousInstanceMatricesAddress = 0u;
  std::span<const ComputeDispatchItem> preDispatches{};
  std::span<const RenderGraphImportedBufferUse> bufferUses{};
  std::span<const AnimatedRenderableGeometryOverride>
      geometryOverridesByRenderable{};
  std::span<const AnimatedRenderableGeometryOverride>
      previousGeometryOverridesByRenderable{};
  std::span<const uint32_t> animatedRenderableIndices{};
  const RenderScene *scene = nullptr;
  uint64_t sceneTopologyVersion = 0u;
  size_t renderableCount = 0u;
  uint64_t version = 0u;
};

} // namespace nuri
