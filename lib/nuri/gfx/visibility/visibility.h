#pragma once

#include "nuri/defines.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/renderers/detail/visibility_math.h"
#include "nuri/math/types.h"
#include "nuri/resources/gpu/model.h"

#include <array>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <memory_resource>
#include <span>
#include <vector>

#include <glm/glm.hpp>

namespace nuri {

enum class VisibilityPassKind : uint32_t {
  OpaqueMain = 0,
  OpaqueDepth = 1,
  OpaqueVelocity = 2,
  OpaqueReactiveMask = 3,
  OpaqueNormal = 4,
  OpaquePick = 5,
  ShadowCascade = 6,
  TransmissionVisibilityDepth = 7,
};

enum VisibilityCandidateFlags : uint32_t {
  kVisibilityCandidateConservativeVisible = 1u << 0u,
};

enum VisibilityGpuFlags : uint32_t {
  kVisibilityGpuFlagFrustumCulling = 1u << 0u,
  kVisibilityGpuFlagOcclusionCulling = 1u << 1u,
  kVisibilityGpuFlagVisibleOnUncertain = 1u << 2u,
};

struct VisibilityResolvedSettings {
  VisibilityCullingMode mainViewMode = VisibilityCullingMode::GpuDriven;
  VisibilityCullingMode shadowMode = VisibilityCullingMode::Hybrid;
  VisibilityOcclusionMode occlusionMode =
      VisibilityOcclusionMode::Disabled;
  bool enableCpuMainFrustumCulling = true;
  bool enableGpuInstanceCulling = true;
  bool enableMeshletFrustumCulling = true;
  bool enableMeshletConeCulling = true;
  bool enableShadowMeshletCulling = true;
  bool enableIndirectMeshDispatch = true;
  bool enableGpuIndirectDraw = true;
  bool enableOcclusionCulling = false;
  bool visibleOnUncertain = true;
  uint32_t forcedVisibleListCapacity = std::numeric_limits<uint32_t>::max();
};

struct VisibilityPassSignature {
  VisibilityPassKind kind = VisibilityPassKind::OpaqueMain;
  uint32_t cascadeIndex = 0;
  uint32_t flags = 0;
  uint64_t sceneTopologyVersion = 0;
  uint64_t materialVersion = 0;
  uint64_t transformVersion = 0;
  uint64_t geometryVersion = 0;
  uint64_t viewSignature = 0;
};

struct VisibilityCandidate {
  uint32_t renderableIndex = 0;
  uint32_t templateIndex = 0;
  uint32_t submeshIndex = 0;
  uint32_t materialIndex = 0;
  uint64_t geometryVersion = 0;
  uint64_t transformVersion = 0;
  uint64_t deformationVersion = 0;
  uint32_t flags = 0;
  BoundingBox localBounds{};
  glm::mat4 worldFromLocal{1.0f};
  const Model::ModelMeshletGpuView *meshletView = nullptr;
};

struct alignas(16) VisibilityCandidateGpu {
  glm::uvec4 ids{0u};
  glm::uvec4 ranges{0u};
  glm::vec4 bounds{0.0f};
  glm::vec4 boundsMin{0.0f};
  glm::vec4 boundsMax{0.0f};
};
static_assert(sizeof(VisibilityCandidateGpu) == 80u);
static_assert(offsetof(VisibilityCandidateGpu, ranges) == 16u);
static_assert(offsetof(VisibilityCandidateGpu, bounds) == 32u);
static_assert(offsetof(VisibilityCandidateGpu, boundsMin) == 48u);
static_assert(offsetof(VisibilityCandidateGpu, boundsMax) == 64u);

struct alignas(16) VisibilityPassGpuData {
  glm::mat4 view{1.0f};
  glm::mat4 proj{1.0f};
  glm::mat4 viewProj{1.0f};
  glm::mat4 previousViewProj{1.0f};
  glm::vec4 planes[6]{};
  glm::vec4 cameraOrLightPos{0.0f};
  glm::vec4 volumeMin{0.0f};
  glm::vec4 volumeMax{0.0f};
  glm::uvec4 passInfo{0u};
  glm::uvec4 depthPyramidInfo{0u};
  std::array<glm::uvec4, kSceneDepthPyramidArraySize> depthPyramidTexIds{};
};
static_assert(sizeof(VisibilityPassGpuData) == 496u);
static_assert(offsetof(VisibilityPassGpuData, view) == 0u);
static_assert(offsetof(VisibilityPassGpuData, proj) == 64u);
static_assert(offsetof(VisibilityPassGpuData, viewProj) == 128u);
static_assert(offsetof(VisibilityPassGpuData, previousViewProj) == 192u);
static_assert(offsetof(VisibilityPassGpuData, planes) == 256u);
static_assert(offsetof(VisibilityPassGpuData, cameraOrLightPos) == 352u);
static_assert(offsetof(VisibilityPassGpuData, volumeMin) == 368u);
static_assert(offsetof(VisibilityPassGpuData, volumeMax) == 384u);
static_assert(offsetof(VisibilityPassGpuData, passInfo) == 400u);
static_assert(offsetof(VisibilityPassGpuData, depthPyramidInfo) == 416u);
static_assert(offsetof(VisibilityPassGpuData, depthPyramidTexIds) == 432u);

struct alignas(16) VisibilityCounterGpuData {
  glm::uvec4 main{0u};
  glm::uvec4 status{0u};
  glm::uvec4 indirect{0u};
  glm::uvec4 meshlet{0u};
  glm::uvec4 meshlet2{0u};
};
static_assert(sizeof(VisibilityCounterGpuData) == 80u);
static_assert(offsetof(VisibilityCounterGpuData, status) == 16u);
static_assert(offsetof(VisibilityCounterGpuData, indirect) == 32u);
static_assert(offsetof(VisibilityCounterGpuData, meshlet) == 48u);
static_assert(offsetof(VisibilityCounterGpuData, meshlet2) == 64u);

struct alignas(16) VisibilityGpuPushConstants {
  uint64_t candidateBufferAddress = 0u;
  uint64_t passBufferAddress = 0u;
  uint64_t visibleIndexBufferAddress = 0u;
  uint64_t counterBufferAddress = 0u;
  uint32_t candidateCount = 0u;
  uint32_t visibleCapacity = 0u;
  uint32_t flags = 0u;
  uint32_t sourceFrameIndex = 0u;
};
static_assert(sizeof(VisibilityGpuPushConstants) == 48u);
static_assert(offsetof(VisibilityGpuPushConstants, candidateCount) == 32u);

struct alignas(16) VisibilityIndirectDrawPushConstants {
  uint64_t commandBufferAddress = 0u;
  uint64_t remapBufferAddress = 0u;
  uint64_t candidateBufferAddress = 0u;
  uint64_t passBufferAddress = 0u;
  uint64_t counterBufferAddress = 0u;
  uint32_t commandWordOffset = 0u;
  uint32_t commandCount = 0u;
  uint32_t candidateCount = 0u;
  uint32_t flags = 0u;
  uint32_t sourceFrameIndex = 0u;
};
static_assert(sizeof(VisibilityIndirectDrawPushConstants) == 64u);
static_assert(
    offsetof(VisibilityIndirectDrawPushConstants, commandWordOffset) == 40u);

struct alignas(16) VisibilityMeshletDispatchGpuData {
  glm::uvec4 groups{0u};
  glm::uvec4 batches{0u};
};
static_assert(sizeof(VisibilityMeshletDispatchGpuData) == 32u);
static_assert(offsetof(VisibilityMeshletDispatchGpuData, batches) == 16u);

struct alignas(16) VisibilityIndirectMeshDispatchPushConstants {
  uint64_t commandBufferAddress = 0u;
  uint64_t dispatchBufferAddress = 0u;
  uint64_t meshletBatchBufferAddress = 0u;
  uint64_t remapBufferAddress = 0u;
  uint64_t candidateMapBufferAddress = 0u;
  uint64_t candidateBufferAddress = 0u;
  uint64_t passBufferAddress = 0u;
  uint32_t dispatchCount = 0u;
  uint32_t candidateCount = 0u;
  uint32_t candidateMapStride = 0u;
  uint32_t flags = 0u;
  uint32_t sourceFrameIndex = 0u;
};
static_assert(sizeof(VisibilityIndirectMeshDispatchPushConstants) == 80u);
static_assert(offsetof(VisibilityIndirectMeshDispatchPushConstants,
                       dispatchCount) == 56u);
static_assert(offsetof(VisibilityIndirectMeshDispatchPushConstants,
                       candidateMapStride) == 64u);

struct VisibilityPassRequest {
  VisibilityPassKind kind = VisibilityPassKind::OpaqueMain;
  VisibilityPassSignature signature{};
  visibility_detail::FrustumPlanes frustum{};
  bool enableCpuFrustumCulling = false;
};

struct VisibilityPassResult {
  VisibilityPassSignature signature{};
  std::pmr::vector<uint32_t> visibleCandidateIndices;
  uint32_t cpuCandidates = 0;
  uint32_t cpuVisibleCandidates = 0;
  uint32_t cpuRejected = 0;
  uint32_t uncertainVisible = 0;

  explicit VisibilityPassResult(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());

  void clear();
};

class NURI_API VisibilityFrameState {
public:
  explicit VisibilityFrameState(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());

  void clear();

  [[nodiscard]] VisibilityPassResult evaluateCpu(
      const VisibilityPassRequest &request,
      std::span<const VisibilityCandidate> candidates);

private:
  std::pmr::memory_resource *memory_;
};

[[nodiscard]] NURI_API VisibilityResolvedSettings
visibilitySettingsFromRenderSettings(const RenderSettings &settings) noexcept;

[[nodiscard]] NURI_API VisibilityPassRequest
makeMainViewVisibilityPassRequest(const CameraFrameState &camera,
                                  const VisibilityResolvedSettings &settings) noexcept;

[[nodiscard]] NURI_API VisibilityCandidateGpu
makeVisibilityCandidateGpu(const VisibilityCandidate &candidate) noexcept;

[[nodiscard]] NURI_API VisibilityPassGpuData makeMainViewVisibilityPassGpuData(
    const CameraFrameState &camera, const VisibilityPassRequest &request,
    uint32_t candidateCount, bool occlusionAvailable,
    uint32_t depthPyramidWidth, uint32_t depthPyramidHeight,
    uint32_t depthPyramidLevelCount, uint32_t depthPyramidSamplerId,
    std::span<const glm::uvec4> depthPyramidTexIds) noexcept;

} // namespace nuri
