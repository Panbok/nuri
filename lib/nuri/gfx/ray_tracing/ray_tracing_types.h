#pragma once

#include "nuri/defines.h"
#include "nuri/gfx/gpu_descriptors.h"
#include "nuri/gfx/render_graph/render_graph.h"
#include <cstdint>
#include <glm/glm.hpp>
#include <span>

namespace nuri {

inline constexpr uint32_t kRayTracingSceneLayoutVersion = 1u;
inline constexpr uint32_t kRayTracingInstanceCustomIndexLimit = 0x00ffffffu;

struct alignas(16) RtInstanceGpuData {
  uint32_t firstGeometryRecord = 0u;
  uint32_t geometryCount = 0u;
  uint32_t renderableIndex = 0u;
  uint32_t flags = 0u;
  glm::mat4 worldFromObject{1.0f};
  glm::mat4 objectFromWorld{1.0f};
};

struct alignas(16) RtGeometryGpuData {
  uint64_t indexBufferAddress = 0u;
  uint64_t vertexBufferAddress = 0u;
  uint64_t vertexDecodeAddress = 0u;
  uint32_t materialIndex = 0u;
  uint32_t indexOffset = 0u;
  uint32_t vertexOffset = 0u;
  uint32_t indexFormatAndVertexFormat = 0u;
  uint32_t flags = 0u;
};

static_assert(sizeof(RtInstanceGpuData) == 144u);
static_assert(sizeof(RtGeometryGpuData) == 48u);
static_assert(alignof(RtInstanceGpuData) == 16u);
static_assert(alignof(RtGeometryGpuData) == 16u);

enum class RayTracingSceneReadiness : uint8_t {
  Disabled = 0,
  Unsupported,
  Building,
  Ready,
  Failed,
};

struct RayTracingSceneFrameView {
  AccelerationStructureHandle topLevelAccelerationStructure{};
  RenderGraphAccelerationStructureId graphTopLevelAccelerationStructure{};
  BufferHandle instanceTable{};
  BufferHandle geometryTable{};
  BufferHandle materialTable{};
  uint64_t instanceTableAddress = 0u;
  uint64_t geometryTableAddress = 0u;
  uint64_t materialTableAddress = 0u;
  std::span<const BufferHandle> indirectSubmissionReferences{};
  std::span<const TextureHandle> indirectSubmissionTextureReferences{};
  uint64_t sceneId = 0u;
  uint64_t topologyVersion = 0u;
  uint64_t transformVersion = 0u;
  uint64_t deformationVersion = 0u;
  uint64_t geometryMutationVersion = 0u;
  uint32_t instanceCount = 0u;
  uint32_t geometryCount = 0u;
  uint32_t completedBlasBuilds = 0u;
  uint32_t totalBlasBuilds = 0u;
  RayTracingSceneReadiness readiness = RayTracingSceneReadiness::Disabled;
  bool ready = false;
};

struct RayTracingSceneFrameMetrics {
  uint32_t staticInstances = 0u;
  uint32_t dynamicInstances = 0u;
  uint32_t excludedDynamicInstances = 0u;
  uint32_t staticBlasCount = 0u;
  uint32_t dynamicBlasCount = 0u;
  uint32_t tlasCount = 0u;
  uint32_t uniqueStaticGeometry = 0u;
  uint32_t geometryRecords = 0u;
  uint64_t triangles = 0u;
  uint32_t queuedBlasBuilds = 0u;
  uint32_t decodedVertices = 0u;
  uint32_t decodeDispatches = 0u;
  uint32_t blasBuilds = 0u;
  uint32_t tlasBuilds = 0u;
  uint32_t tlasUpdates = 0u;
  uint32_t dynamicBlasUpdates = 0u;
  uint32_t dynamicVertexDispatches = 0u;
  uint64_t decodedPositionBytes = 0u;
  uint64_t tableBytes = 0u;
  uint64_t consumedRebuildEpoch = 0u;
  float gpuTimeMs = 0.0f;
  float blasGpuTimeMs = 0.0f;
  float tlasGpuTimeMs = 0.0f;
  uint64_t gpuTimingSourceFrameIndex = UINT64_MAX;
  uint32_t gpuTimingAvailable = 0u;
  uint32_t blasGpuTimingAvailable = 0u;
  uint32_t tlasGpuTimingAvailable = 0u;
  RayTracingSceneReadiness readiness = RayTracingSceneReadiness::Disabled;
};

[[nodiscard]] NURI_API AccelerationStructureTransform
makeAccelerationStructureTransform(const glm::mat4 &worldFromObject) noexcept;
[[nodiscard]] constexpr uint32_t
packRtGeometryFormats(IndexFormat indexFormat,
                      uint32_t packedVertexFormat) noexcept {
  return static_cast<uint32_t>(indexFormat) | (packedVertexFormat << 8u);
}

} // namespace nuri
