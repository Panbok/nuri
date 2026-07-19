#include "nuri/gfx/renderers/shadow_renderer.h"
#include "nuri/core/containers/hash_map.h"
#include "nuri/core/log.h"
#include "nuri/core/pmr_scratch.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/frame/render_capture.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/gfx/renderers/detail/animation_rendering.h"
#include "nuri/gfx/renderers/detail/forward_rendering.h"
#include "nuri/gfx/renderers/detail/shadow_math.h"
#include "nuri/pch.h"
#include "nuri/resources/gpu/resource_manager.h"
#include <bit>
namespace nuri {
namespace {
constexpr uint32_t kShadowPassDebugColor = 0xff5a7dffu;
constexpr uint32_t kShadowPreviewPassDebugColor = 0xff7d5affu;
constexpr std::string_view kShadowPassLabel = "ShadowDepthPass";
constexpr std::string_view kShadowPreviewPassLabel = "ShadowDepthPreviewPass";
constexpr std::string_view kShadowPreviewDrawLabel = "ShadowDepthPreview";
constexpr uint64_t kFnvOffsetBasis64 = 14695981039346656037ull;
constexpr uint64_t kFnvPrime64 = 1099511628211ull;
constexpr uint32_t kShadowPreviewFlagInvert = 1u << 0u;
constexpr uint32_t kShadowPreviewFlagLog = 1u << 1u;
constexpr uint32_t kShadowPreviewFlagTiled = 1u << 2u;
constexpr float kCullingPaddingTexelMultiplier = 2.0f;
constexpr uint32_t kStaticCasterLightGridResolution = 64u;
constexpr uint32_t kStaticCasterLightGridDepthResolution = 16u;
constexpr uint32_t kStaticCasterLightGridMaxCellsPerCaster = 128u;
constexpr size_t kStaticCasterLightGridSortedCandidateLimit = 1024u;
constexpr uint32_t kStaticBatchOverlapEmitMinInstances = 4u;
constexpr float kStaticOnlyGuardBandShadowMapFraction = 1.0f / 8.0f;
constexpr float kStaticOnlyGuardBandMaxTexels = 1536.0f;
constexpr float kStaticOnlyGuardBandMinTexels = 8.0f;
constexpr float kStaticOnlyDepthGuardBandShadowMapFraction = 1.0f / 16.0f;
constexpr float kStaticOnlyDepthGuardBandMaxTexels = 768.0f;
constexpr float kStaticOnlyDepthGuardBandMinTexels = 4.0f;
constexpr float kStaticOnlyAdaptiveGuardBandMotionMultiplier = 3.0f;
constexpr float kStaticOnlyAdaptiveGuardBandMaxShadowMapFraction = 1.0f / 4.0f;
constexpr float kStaticOnlyAdaptiveDepthGuardBandMaxShadowMapFraction =
    1.0f / 8.0f;
constexpr float kStaticOnlyPredictiveCenterMotionMultiplier = 1.0f;
constexpr float kStaticOnlyPredictiveExtentGrowthFraction = 0.5f;
constexpr float kStaticOnlyPredictiveTrailingGuardFraction = 0.5f;
constexpr float kStaticOnlyPredictiveMotionEpsilonTexels = 0.5f;
constexpr uint32_t kStaticOnlyAdaptiveRefreshBudgetPerFrame = 1u;
constexpr uint32_t kMaxShadowIndirectCommandsPerDraw = 1024u;
constexpr uint32_t kMinSdsmReduceResultRingCount = 16u;
constexpr uint32_t kSdsmGpuWarmupGraceMissFrames = 2u;
[[nodiscard]] constexpr size_t shadowPipelineIndex(bool doubleSided,
                                                   bool alphaMasked) {
  return (alphaMasked ? 2u : 0u) + (doubleSided ? 1u : 0u);
}
[[nodiscard]] constexpr Format
sanitizeShadowDepthFormat(Format format) noexcept {
  switch (format) {
  case Format::D16_UNORM:
  case Format::D32_FLOAT:
    return format;
  default:
    return kDefaultShadowMapDepthFormat;
  }
}
[[nodiscard]] constexpr uint64_t
shadowDepthTextureBytesPerPixel(Format format) noexcept {
  switch (sanitizeShadowDepthFormat(format)) {
  case Format::D32_FLOAT:
    return sizeof(float);
  case Format::D16_UNORM:
  default:
    return sizeof(uint16_t);
  }
}
constexpr std::array<std::string_view, kMaxShadowCascades>
    kShadowCascadePassLabels{
        "ShadowDepthPass.Cascade0", "ShadowDepthPass.Cascade1",
        "ShadowDepthPass.Cascade2", "ShadowDepthPass.Cascade3"};
constexpr std::array<std::string_view, kMaxShadowCascades>
    kShadowCascadeTextureImportNames{
        "shadow_depth_map_cascade0", "shadow_depth_map_cascade1",
        "shadow_depth_map_cascade2", "shadow_depth_map_cascade3"};
constexpr std::array<std::string_view, kMaxShadowCascades>
    kShadowCascadeCaptureNames{"shadow_cascade_0", "shadow_cascade_1",
                               "shadow_cascade_2", "shadow_cascade_3"};
void logShadowVisibilityCounters(const RenderFrameContext &frame) {
  if (frame.settings == nullptr ||
      !renderSettingsOrDefault(frame).visibility.debug.logCounters) {
    return;
  }
  const VisibilityFrameMetrics &visibility = frame.metrics.visibility;
  NURI_LOG_INFO("ShadowRenderer::visibility counters frame=%llu "
                "cpu(candidates=%u rejected=%u)",
                static_cast<unsigned long long>(frame.frameIndex),
                visibility.shadowCpuCandidates, visibility.shadowCpuRejected);
}
[[nodiscard]] float
elapsedMilliseconds(std::chrono::steady_clock::time_point begin,
                    std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<float, std::milli>(end - begin).count();
}
[[nodiscard]] std::pmr::memory_resource *
resolveMemoryResource(std::pmr::memory_resource *memory) {
  return memory != nullptr ? memory : std::pmr::get_default_resource();
}
[[nodiscard]] uint64_t hashBytes(std::span<const std::byte> bytes) {
  uint64_t hash = kFnvOffsetBasis64;
  for (const std::byte value : bytes) {
    hash ^= static_cast<uint64_t>(std::to_integer<uint8_t>(value));
    hash *= kFnvPrime64;
  }
  return hash;
}
[[nodiscard]] uint64_t hashCombine64(uint64_t hash, uint64_t value) {
  hash ^= value;
  hash *= kFnvPrime64;
  return hash;
}
[[nodiscard]] uint64_t hashCombineBytes(uint64_t hash,
                                        std::span<const std::byte> bytes) {
  for (const std::byte value : bytes) {
    hash ^= static_cast<uint64_t>(std::to_integer<uint8_t>(value));
    hash *= kFnvPrime64;
  }
  return hash;
}
template <typename T>
[[nodiscard]] uint64_t hashCombineValue(uint64_t hash, const T &value) {
  return hashCombineBytes(hash, std::as_bytes(std::span<const T>(&value, 1u)));
}
template <typename... Containers> void clearAll(Containers &...containers) {
  (containers.clear(), ...);
}
template <typename Range> void clearEach(Range &range) {
  for (auto &value : range)
    value.clear();
}
[[nodiscard]] uint64_t foldHandle(uint32_t index, uint32_t generation) {
  return (static_cast<uint64_t>(generation) << 32u) | index;
}
struct ShadowBatchKey {
  uint32_t cascadeIndex = 0u;
  bool dynamicCaster = false;
  RenderPipelineHandle pipeline{};
  BufferHandle vertexBuffer{};
  BufferHandle vertexDecodeBuffer{};
  BufferHandle indexBuffer{};
  uint64_t indexBufferOffset = 0u;
  IndexFormat indexFormat = IndexFormat::U32;
  uint32_t indexCount = 0u;
  uint32_t firstIndex = 0u;
  uint64_t vertexBufferAddress = 0u;
  uint64_t vertexDecodeBufferAddress = 0u;
  uint32_t vertexDecodeIndex = 0u;
  uint32_t packedVertexFormat = 0u;
  uint32_t materialIndex = 0u;
  bool operator==(const ShadowBatchKey &other) const noexcept {
    return cascadeIndex == other.cascadeIndex &&
           dynamicCaster == other.dynamicCaster && pipeline == other.pipeline &&
           vertexBuffer == other.vertexBuffer &&
           vertexDecodeBuffer == other.vertexDecodeBuffer &&
           indexBuffer == other.indexBuffer &&
           indexBufferOffset == other.indexBufferOffset &&
           indexFormat == other.indexFormat && indexCount == other.indexCount &&
           firstIndex == other.firstIndex &&
           vertexBufferAddress == other.vertexBufferAddress &&
           vertexDecodeBufferAddress == other.vertexDecodeBufferAddress &&
           vertexDecodeIndex == other.vertexDecodeIndex &&
           packedVertexFormat == other.packedVertexFormat &&
           materialIndex == other.materialIndex;
  }
};
struct ShadowBatchKeyHash {
  size_t operator()(const ShadowBatchKey &key) const noexcept {
    uint64_t hash = kFnvOffsetBasis64;
    hash = hashCombine64(hash, key.cascadeIndex);
    hash = hashCombine64(hash, key.dynamicCaster ? 1ull : 0ull);
    hash = hashCombine64(
        hash, foldHandle(key.pipeline.index, key.pipeline.generation));
    hash = hashCombine64(
        hash, foldHandle(key.vertexBuffer.index, key.vertexBuffer.generation));
    hash = hashCombine64(hash, foldHandle(key.vertexDecodeBuffer.index,
                                          key.vertexDecodeBuffer.generation));
    hash = hashCombine64(
        hash, foldHandle(key.indexBuffer.index, key.indexBuffer.generation));
    hash = hashCombine64(hash, key.indexBufferOffset);
    hash = hashCombine64(hash, static_cast<uint64_t>(key.indexFormat));
    hash = hashCombine64(hash, (static_cast<uint64_t>(key.indexCount) << 32u) |
                                   key.firstIndex);
    hash = hashCombine64(hash, key.vertexBufferAddress);
    hash = hashCombine64(hash, key.vertexDecodeBufferAddress);
    hash = hashCombine64(hash,
                         (static_cast<uint64_t>(key.vertexDecodeIndex) << 32u) |
                             key.packedVertexFormat);
    hash = hashCombine64(hash, key.materialIndex);
    return static_cast<size_t>(hash);
  }
};
struct ShadowIndirectGroupKey {
  RenderPipelineHandle pipeline{};
  BufferHandle indexBuffer{};
  IndexFormat indexFormat = IndexFormat::U32;
  uint32_t materialIndex = 0u;
  bool operator==(const ShadowIndirectGroupKey &other) const noexcept {
    return pipeline == other.pipeline && indexBuffer == other.indexBuffer &&
           indexFormat == other.indexFormat &&
           materialIndex == other.materialIndex;
  }
};
struct ShadowIndirectGroupKeyHash {
  size_t operator()(const ShadowIndirectGroupKey &key) const noexcept {
    uint64_t hash = kFnvOffsetBasis64;
    hash = hashCombine64(
        hash, foldHandle(key.pipeline.index, key.pipeline.generation));
    hash = hashCombine64(
        hash, foldHandle(key.indexBuffer.index, key.indexBuffer.generation));
    hash = hashCombine64(hash, static_cast<uint64_t>(key.indexFormat));
    hash = hashCombine64(hash, key.materialIndex);
    return static_cast<size_t>(hash);
  }
};
struct alignas(16) ShadowDrawGpuData {
  uint64_t vertexBufferAddress = 0u;
  uint64_t vertexDecodeBufferAddress = 0u;
  uint32_t vertexDecodeIndex = 0u;
  uint32_t packedVertexFormat = 0u;
  uint32_t reserved0 = 0u;
  uint32_t reserved1 = 0u;
};
static_assert(sizeof(ShadowDrawGpuData) == 32u,
              "ShadowDrawGpuData must match the shader layout");
struct DrawIndexedIndirectCommand {
  uint32_t indexCount = 0u;
  uint32_t instanceCount = 0u;
  uint32_t firstIndex = 0u;
  int32_t vertexOffset = 0;
  uint32_t firstInstance = 0u;
};
static_assert(sizeof(DrawIndexedIndirectCommand) == 20u,
              "Vulkan indexed-indirect command layout changed");
struct ShadowBatchEntry {
  ShadowBatchKey key{};
  uint32_t firstInstance = 0u;
  uint32_t inlineInstanceIndex = 0u;
  uint32_t instanceCount = 0u;
  const uint32_t *externalInstanceIndices = nullptr;
  uint32_t externalInstanceCount = 0u;
  uint32_t instanceListOffset = 0u;
  bool hasInstanceList = false;
  void materializeInstances(std::pmr::vector<uint32_t> &instanceLists,
                            uint32_t reserveHint = 0u) {
    if (hasInstanceList) {
      return;
    }
    const uint32_t existingCount = instanceCount;
    const size_t requestedCapacity =
        instanceLists.size() +
        std::max<size_t>(reserveHint, std::max(existingCount, 2u));
    if (requestedCapacity > instanceLists.capacity()) {
      instanceLists.reserve(requestedCapacity);
    }
    instanceListOffset = static_cast<uint32_t>(instanceLists.size());
    hasInstanceList = true;
    if (externalInstanceCount != 0u) {
      instanceLists.insert(instanceLists.end(), externalInstanceIndices,
                           externalInstanceIndices + externalInstanceCount);
      externalInstanceIndices = nullptr;
      externalInstanceCount = 0u;
    } else if (existingCount != 0u) {
      instanceLists.push_back(inlineInstanceIndex);
    }
  }
  void appendInstance(uint32_t instanceIndex,
                      std::pmr::vector<uint32_t> &instanceLists,
                      uint32_t reserveHint = 0u) {
    if (instanceCount == 0u) {
      inlineInstanceIndex = instanceIndex;
      instanceCount = 1u;
      return;
    }
    materializeInstances(instanceLists,
                         std::max(reserveHint, instanceCount + 1u));
    instanceLists.push_back(instanceIndex);
    ++instanceCount;
  }
  void appendInstances(std::span<const uint32_t> indices,
                       std::pmr::vector<uint32_t> &instanceLists) {
    const uint32_t addedCount = static_cast<uint32_t>(
        std::min<size_t>(indices.size(), std::numeric_limits<uint32_t>::max()));
    if (addedCount == 0u) {
      return;
    }
    if (instanceCount == 0u) {
      externalInstanceIndices = indices.data();
      externalInstanceCount = addedCount;
      instanceCount = addedCount;
      return;
    }
    materializeInstances(instanceLists, instanceCount + addedCount);
    instanceLists.insert(instanceLists.end(), indices.begin(), indices.end());
    instanceCount += addedCount;
  }
};
[[nodiscard]] std::string_view shadowSdsmStatusName(ShadowSdsmStatus status) {
  switch (status) {
  case ShadowSdsmStatus::Disabled:
    return "Disabled";
  case ShadowSdsmStatus::Active:
    return "Active";
  case ShadowSdsmStatus::Unavailable:
    return "Unavailable";
  case ShadowSdsmStatus::Stale:
    return "Stale";
  case ShadowSdsmStatus::Invalid:
    return "Invalid";
  case ShadowSdsmStatus::FallbackFixed:
    return "FallbackFixed";
  }
  return "Unknown";
}
[[nodiscard]] std::string_view
shadowSdsmReductionBackendName(ShadowSdsmReductionBackend backend) {
  switch (backend) {
  case ShadowSdsmReductionBackend::Cpu:
    return "Cpu";
  case ShadowSdsmReductionBackend::Gpu:
    return "Gpu";
  }
  return "Unknown";
}
template <typename T>
[[nodiscard]] bool rawBytesEqual(const T &lhs, const T &rhs) {
  static_assert(std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>,
                "rawBytesEqual is only for plain layout values; use operator== "
                "or element-wise comparison for other types.");
  return std::memcmp(&lhs, &rhs, sizeof(T)) == 0;
}
[[nodiscard]] uint64_t buildShadowDiagnosticSignature(
    const ShadowDebugFrameData &debugFrameData,
    const ShadowFrameMetrics &metrics, bool hasActiveShadowLightForFrame,
    bool hasPreparedShadowDepthPasses, bool hasPreparedShadowPreviewPass) {
  uint64_t hash = hashBytes(std::as_bytes(
      std::span<const ShadowDebugFrameData>(&debugFrameData, 1u)));
  hash = hashCombineBytes(
      hash, std::as_bytes(std::span<const ShadowFrameMetrics>(&metrics, 1u)));
  hash = hashCombineValue(hash, hasActiveShadowLightForFrame);
  hash = hashCombineValue(hash, hasPreparedShadowDepthPasses);
  hash = hashCombineValue(hash, hasPreparedShadowPreviewPass);
  return hash;
}
constexpr uint64_t kSdsmMaxCachedSourceFrameLag = 2u;
void appendUniqueBufferDependency(
    std::pmr::vector<BufferDependency> &dependencies, BufferHandle handle,
    RenderGraphAccessMode mode = RenderGraphAccessMode::Read) {
  if (!nuri::isValid(handle)) {
    return;
  }
  for (const BufferDependency &existing : dependencies) {
    if (existing.handle == handle) {
      return;
    }
  }
  dependencies.push_back(BufferDependency{
      .handle = handle,
      .mode = mode,
  });
}
void appendUniqueBufferHandle(std::pmr::vector<BufferHandle> &buffers,
                              BufferHandle handle) {
  if (!nuri::isValid(handle)) {
    return;
  }
  for (const BufferHandle existing : buffers) {
    if (existing == handle) {
      return;
    }
  }
  buffers.push_back(handle);
}
void appendUniqueTextureDependency(
    std::pmr::vector<TextureDependency> &dependencies, TextureHandle handle,
    RenderGraphAccessMode mode = RenderGraphAccessMode::Read) {
  if (!nuri::isValid(handle)) {
    return;
  }
  for (const TextureDependency &existing : dependencies) {
    if (existing.handle.index == handle.index &&
        existing.handle.generation == handle.generation) {
      return;
    }
  }
  dependencies.push_back(TextureDependency{
      .handle = handle,
      .mode = mode,
  });
}
struct ShadowSplitRange {
  float nearDepth = 0.0f;
  float farDepth = 0.0f;
};
[[nodiscard]] ShadowSplitRange
computeFixedShadowSplitRange(const CameraFrameState &camera,
                             float maxDistance) {
  const float nearDepth = std::max(camera.nearPlane, 0.01f);
  const float requestedFar =
      std::min(std::max(maxDistance, nearDepth + 0.01f),
               std::max(camera.farPlane, nearDepth + 0.01f));
  return ShadowSplitRange{
      .nearDepth = nearDepth,
      .farDepth = std::max(nearDepth + 0.01f, requestedFar),
  };
}
[[nodiscard]] float
computeSdsmFarUpdateThreshold(float farCascadeTexelWorldSize) {
  return std::max(std::max(farCascadeTexelWorldSize, 0.0f) * 16.0f, 0.5f);
}
[[nodiscard]] CameraFrameState
cameraWithShadowSplitRange(const CameraFrameState &camera,
                           ShadowSplitRange range) {
  CameraFrameState adjustedCamera = camera;
  adjustedCamera.nearPlane = std::max(range.nearDepth, 0.01f);
  adjustedCamera.farPlane =
      std::max(adjustedCamera.nearPlane + 0.01f, range.farDepth);
  return adjustedCamera;
}
template <typename DependencyT, typename HandleT>
void splitDependencies(std::span<const DependencyT> dependencies,
                       std::pmr::vector<HandleT> &handles,
                       std::pmr::vector<RenderGraphAccessMode> &accessModes) {
  handles.clear();
  accessModes.clear();
  handles.reserve(dependencies.size());
  accessModes.reserve(dependencies.size());
  for (const DependencyT &dependency : dependencies) {
    handles.push_back(dependency.handle);
    accessModes.push_back(dependency.mode);
  }
}
void appendAnimatedGeometryDependencies(
    std::pmr::vector<BufferDependency> &dependencies,
    const AnimationSceneFrameData *animationSceneData) {
  if (animationSceneData == nullptr) {
    return;
  }
  for (const AnimatedRenderableGeometryOverride &geometryOverride :
       animationSceneData->geometryOverridesByRenderable) {
    if (!geometryOverride.enabled ||
        !nuri::isValid(geometryOverride.vertexBuffer)) {
      continue;
    }
    appendUniqueBufferDependency(dependencies, geometryOverride.vertexBuffer);
  }
}
[[nodiscard]] std::span<const ComputeDispatchItem>
shadowCascadePreDispatches(const AnimationSceneFrameData *animationSceneData,
                           uint32_t cascadeIndex) {
  if (cascadeIndex != 0u || animationSceneData == nullptr) {
    return {};
  }
  return animationSceneData->preDispatches;
}
struct StaticOnlyGuardBandTexels {
  float ortho = 0.0f;
  float depth = 0.0f;
};
struct StaticOnlyDirectedTexels {
  glm::vec2 x{0.0f};
  glm::vec2 y{0.0f};
  glm::vec2 z{0.0f};
};
[[nodiscard]] float staticOnlyGuardBandTexels(uint32_t shadowMapSize) {
  return std::clamp(
      static_cast<float>(shadowMapSize) * kStaticOnlyGuardBandShadowMapFraction,
      kStaticOnlyGuardBandMinTexels, kStaticOnlyGuardBandMaxTexels);
}
[[nodiscard]] float staticOnlyDepthGuardBandTexels(uint32_t shadowMapSize) {
  return std::clamp(static_cast<float>(shadowMapSize) *
                        kStaticOnlyDepthGuardBandShadowMapFraction,
                    kStaticOnlyDepthGuardBandMinTexels,
                    kStaticOnlyDepthGuardBandMaxTexels);
}
[[nodiscard]] float
staticOnlyAdaptiveGuardBandMaxTexels(uint32_t shadowMapSize) {
  return std::max(staticOnlyGuardBandTexels(shadowMapSize),
                  static_cast<float>(shadowMapSize) *
                      kStaticOnlyAdaptiveGuardBandMaxShadowMapFraction);
}
[[nodiscard]] float
staticOnlyAdaptiveDepthGuardBandMaxTexels(uint32_t shadowMapSize) {
  return std::max(staticOnlyDepthGuardBandTexels(shadowMapSize),
                  static_cast<float>(shadowMapSize) *
                      kStaticOnlyAdaptiveDepthGuardBandMaxShadowMapFraction);
}
[[nodiscard]] StaticOnlyGuardBandTexels staticOnlyRawMotionTexels(
    const shadow_detail::DirectionalShadowFit &currentRawFit,
    const shadow_detail::DirectionalShadowFit &previousRawFit,
    bool previousRawFitValid) {
  StaticOnlyGuardBandTexels motion{};
  if (!previousRawFitValid ||
      !shadow_detail::canReuseDirectionalShadowFitAnchor(currentRawFit,
                                                         previousRawFit)) {
    return motion;
  }
  const float texelWorldSize = std::max(currentRawFit.texelWorldSize, 1.0e-6f);
  const float orthoMotionWorld = std::max({
      std::abs(currentRawFit.lightSpaceBoundsMin.x -
               previousRawFit.lightSpaceBoundsMin.x),
      std::abs(currentRawFit.lightSpaceBoundsMax.x -
               previousRawFit.lightSpaceBoundsMax.x),
      std::abs(currentRawFit.lightSpaceBoundsMin.y -
               previousRawFit.lightSpaceBoundsMin.y),
      std::abs(currentRawFit.lightSpaceBoundsMax.y -
               previousRawFit.lightSpaceBoundsMax.y),
  });
  const float depthMotionWorld =
      std::max(std::abs(currentRawFit.lightSpaceBoundsMin.z -
                        previousRawFit.lightSpaceBoundsMin.z),
               std::abs(currentRawFit.lightSpaceBoundsMax.z -
                        previousRawFit.lightSpaceBoundsMax.z));
  motion.ortho = std::max(orthoMotionWorld / texelWorldSize, 0.0f);
  motion.depth = std::max(depthMotionWorld / texelWorldSize, 0.0f);
  return motion;
}
[[nodiscard]] StaticOnlyGuardBandTexels staticOnlyMotionGuardBandTexels(
    const shadow_detail::DirectionalShadowFit &currentRawFit,
    const shadow_detail::DirectionalShadowFit &previousRawFit,
    bool previousRawFitValid, uint32_t shadowMapSize) {
  const StaticOnlyGuardBandTexels motion = staticOnlyRawMotionTexels(
      currentRawFit, previousRawFit, previousRawFitValid);
  return StaticOnlyGuardBandTexels{
      .ortho = std::clamp(
          motion.ortho * kStaticOnlyAdaptiveGuardBandMotionMultiplier, 0.0f,
          staticOnlyAdaptiveGuardBandMaxTexels(shadowMapSize)),
      .depth = std::clamp(
          motion.depth * kStaticOnlyAdaptiveGuardBandMotionMultiplier, 0.0f,
          staticOnlyAdaptiveDepthGuardBandMaxTexels(shadowMapSize)),
  };
}
[[nodiscard]] StaticOnlyGuardBandTexels staticOnlyRenderGuardBandTexels(
    const shadow_detail::DirectionalShadowFit &currentRawFit,
    const shadow_detail::DirectionalShadowFit &previousRawFit,
    bool previousRawFitValid, uint32_t shadowMapSize) {
  const StaticOnlyGuardBandTexels motionGuard = staticOnlyMotionGuardBandTexels(
      currentRawFit, previousRawFit, previousRawFitValid, shadowMapSize);
  return StaticOnlyGuardBandTexels{
      .ortho =
          std::max(staticOnlyGuardBandTexels(shadowMapSize), motionGuard.ortho),
      .depth = std::max(staticOnlyDepthGuardBandTexels(shadowMapSize),
                        motionGuard.depth),
  };
}
[[nodiscard]] StaticOnlyDirectedTexels staticOnlyDirectedMotionTexels(
    const shadow_detail::DirectionalShadowFit &currentRawFit,
    const shadow_detail::DirectionalShadowFit &previousRawFit,
    bool previousRawFitValid) {
  StaticOnlyDirectedTexels motion{};
  if (!previousRawFitValid ||
      !shadow_detail::canReuseDirectionalShadowFitAnchor(currentRawFit,
                                                         previousRawFit)) {
    return motion;
  }
  const float texelWorldSize = std::max(currentRawFit.texelWorldSize, 1.0e-6f);
  const auto resolveAxisMotion = [texelWorldSize](
                                     float currentMin, float currentMax,
                                     float previousMin, float previousMax) {
    const float deltaMin = currentMin - previousMin;
    const float deltaMax = currentMax - previousMax;
    return glm::vec2(std::max({-deltaMin, -deltaMax, 0.0f}) / texelWorldSize,
                     std::max({deltaMin, deltaMax, 0.0f}) / texelWorldSize);
  };
  motion.x = resolveAxisMotion(currentRawFit.lightSpaceBoundsMin.x,
                               currentRawFit.lightSpaceBoundsMax.x,
                               previousRawFit.lightSpaceBoundsMin.x,
                               previousRawFit.lightSpaceBoundsMax.x);
  motion.y = resolveAxisMotion(currentRawFit.lightSpaceBoundsMin.y,
                               currentRawFit.lightSpaceBoundsMax.y,
                               previousRawFit.lightSpaceBoundsMin.y,
                               previousRawFit.lightSpaceBoundsMax.y);
  motion.z = resolveAxisMotion(currentRawFit.lightSpaceBoundsMin.z,
                               currentRawFit.lightSpaceBoundsMax.z,
                               previousRawFit.lightSpaceBoundsMin.z,
                               previousRawFit.lightSpaceBoundsMax.z);
  return motion;
}
[[nodiscard]] StaticOnlyDirectedTexels staticOnlyDirectedRemainingMarginTexels(
    const shadow_detail::DirectionalShadowFit &renderedFit,
    const shadow_detail::DirectionalShadowFit &currentRawFit) {
  StaticOnlyDirectedTexels margin{};
  if (!shadow_detail::canReuseDirectionalShadowFitAnchor(renderedFit,
                                                         currentRawFit)) {
    return margin;
  }
  const float texelWorldSize = std::max(currentRawFit.texelWorldSize, 1.0e-6f);
  const auto resolveAxisMargin = [texelWorldSize](
                                     float renderedMin, float renderedMax,
                                     float currentMin, float currentMax) {
    return glm::vec2(std::max(currentMin - renderedMin, 0.0f) / texelWorldSize,
                     std::max(renderedMax - currentMax, 0.0f) / texelWorldSize);
  };
  margin.x = resolveAxisMargin(
      renderedFit.lightSpaceBoundsMin.x, renderedFit.lightSpaceBoundsMax.x,
      currentRawFit.lightSpaceBoundsMin.x, currentRawFit.lightSpaceBoundsMax.x);
  margin.y = resolveAxisMargin(
      renderedFit.lightSpaceBoundsMin.y, renderedFit.lightSpaceBoundsMax.y,
      currentRawFit.lightSpaceBoundsMin.y, currentRawFit.lightSpaceBoundsMax.y);
  margin.z = resolveAxisMargin(
      renderedFit.lightSpaceBoundsMin.z, renderedFit.lightSpaceBoundsMax.z,
      currentRawFit.lightSpaceBoundsMin.z, currentRawFit.lightSpaceBoundsMax.z);
  return margin;
}
[[nodiscard]] bool staticOnlyNeedsAdaptiveRefresh(
    const shadow_detail::DirectionalShadowFit &renderedFit,
    const shadow_detail::DirectionalShadowFit &currentRawFit,
    const shadow_detail::DirectionalShadowFit &previousRawFit,
    bool previousRawFitValid, uint32_t shadowMapSize) {
  const StaticOnlyDirectedTexels motion = staticOnlyDirectedMotionTexels(
      currentRawFit, previousRawFit, previousRawFitValid);
  const StaticOnlyDirectedTexels margin =
      staticOnlyDirectedRemainingMarginTexels(renderedFit, currentRawFit);
  const StaticOnlyDirectedTexels previousMargin =
      staticOnlyDirectedRemainingMarginTexels(renderedFit, previousRawFit);
  const float baseOrthoGuard = staticOnlyGuardBandTexels(shadowMapSize);
  const float baseDepthGuard = staticOnlyDepthGuardBandTexels(shadowMapSize);
  const auto needsAxisRefresh = [](glm::vec2 currentMargin,
                                   glm::vec2 previousAxisMargin,
                                   glm::vec2 axisMotion, float baseGuard) {
    const bool negativeMarginWillExpire =
        currentMargin.x + kStaticOnlyPredictiveMotionEpsilonTexels <
        axisMotion.x;
    const bool positiveMarginWillExpire =
        currentMargin.y + kStaticOnlyPredictiveMotionEpsilonTexels <
        axisMotion.y;
    const bool negativeLargeStep =
        axisMotion.x > baseGuard + kStaticOnlyPredictiveMotionEpsilonTexels;
    const bool positiveLargeStep =
        axisMotion.y > baseGuard + kStaticOnlyPredictiveMotionEpsilonTexels;
    const bool negativeCacheAlreadyOffset =
        previousAxisMargin.x + kStaticOnlyPredictiveMotionEpsilonTexels <
        baseGuard;
    const bool positiveCacheAlreadyOffset =
        previousAxisMargin.y + kStaticOnlyPredictiveMotionEpsilonTexels <
        baseGuard;
    return (negativeMarginWillExpire &&
            (negativeLargeStep || negativeCacheAlreadyOffset)) ||
           (positiveMarginWillExpire &&
            (positiveLargeStep || positiveCacheAlreadyOffset));
  };
  return needsAxisRefresh(margin.x, previousMargin.x, motion.x,
                          baseOrthoGuard) ||
         needsAxisRefresh(margin.y, previousMargin.y, motion.y,
                          baseOrthoGuard) ||
         needsAxisRefresh(margin.z, previousMargin.z, motion.z, baseDepthGuard);
}
void applyStaticOnlyGuardBand(
    shadow_detail::DirectionalShadowFit &fit,
    const shadow_detail::DirectionalShadowFit &previousRawFit,
    bool previousRawFitValid, uint32_t shadowMapSize,
    StaticOnlyGuardBandTexels guard) {
  const float sourceTexelWorldSize = fit.texelWorldSize;
  const float baseOrthoPadding = shadow_detail::quantizeShadowPaddingUp(
      sourceTexelWorldSize * staticOnlyGuardBandTexels(shadowMapSize),
      sourceTexelWorldSize);
  const float baseDepthPadding = shadow_detail::quantizeShadowPaddingUp(
      sourceTexelWorldSize * staticOnlyDepthGuardBandTexels(shadowMapSize),
      sourceTexelWorldSize);
  const float adaptiveOrthoPadding = shadow_detail::quantizeShadowPaddingUp(
      sourceTexelWorldSize * guard.ortho, sourceTexelWorldSize);
  const float adaptiveDepthPadding = shadow_detail::quantizeShadowPaddingUp(
      sourceTexelWorldSize * guard.depth, sourceTexelWorldSize);
  const float extraOrthoPadding =
      std::max(adaptiveOrthoPadding - baseOrthoPadding, 0.0f);
  const float extraDepthPadding =
      std::max(adaptiveDepthPadding - baseDepthPadding, 0.0f);
  const bool hasMotionAnchor =
      previousRawFitValid &&
      shadow_detail::canReuseDirectionalShadowFitAnchor(fit, previousRawFit);
  if (!hasMotionAnchor) {
    shadow_detail::expandDirectionalShadowFitBounds(
        fit, shadowMapSize, adaptiveOrthoPadding, adaptiveDepthPadding);
    return;
  }
  const float predictiveOrthoPadding =
      baseOrthoPadding +
      extraOrthoPadding * kStaticOnlyPredictiveExtentGrowthFraction;
  const float predictiveDepthPadding =
      baseDepthPadding +
      extraDepthPadding * kStaticOnlyPredictiveExtentGrowthFraction;
  const glm::vec3 currentCenter = shadow_detail::lightSpaceCenter(
      fit.lightSpaceBoundsMin, fit.lightSpaceBoundsMax);
  const glm::vec3 currentExtent =
      fit.lightSpaceBoundsMax - fit.lightSpaceBoundsMin;
  const auto resolvePredictiveShift =
      [sourceTexelWorldSize](float deltaMin, float deltaMax,
                             float extentPadding) {
        const float negativeMotion = std::max({-deltaMin, -deltaMax, 0.0f});
        const float positiveMotion = std::max({deltaMin, deltaMax, 0.0f});
        if (std::max(negativeMotion, positiveMotion) <=
            sourceTexelWorldSize * kStaticOnlyPredictiveMotionEpsilonTexels) {
          return 0.0f;
        }
        const float trailingReserve = std::min(
            extentPadding,
            std::max(extentPadding * kStaticOnlyPredictiveTrailingGuardFraction,
                     sourceTexelWorldSize));
        const float maxShift = std::max(extentPadding - trailingReserve, 0.0f);
        const float desiredShift = (positiveMotion - negativeMotion) *
                                   kStaticOnlyPredictiveCenterMotionMultiplier;
        return std::clamp(desiredShift, -maxShift, maxShift);
      };
  const glm::vec3 predictiveShift(
      resolvePredictiveShift(
          fit.lightSpaceBoundsMin.x - previousRawFit.lightSpaceBoundsMin.x,
          fit.lightSpaceBoundsMax.x - previousRawFit.lightSpaceBoundsMax.x,
          predictiveOrthoPadding),
      resolvePredictiveShift(
          fit.lightSpaceBoundsMin.y - previousRawFit.lightSpaceBoundsMin.y,
          fit.lightSpaceBoundsMax.y - previousRawFit.lightSpaceBoundsMax.y,
          predictiveOrthoPadding),
      resolvePredictiveShift(
          fit.lightSpaceBoundsMin.z - previousRawFit.lightSpaceBoundsMin.z,
          fit.lightSpaceBoundsMax.z - previousRawFit.lightSpaceBoundsMax.z,
          predictiveDepthPadding));
  glm::vec3 lightMin = currentCenter + predictiveShift - currentExtent * 0.5f;
  glm::vec3 lightMax = currentCenter + predictiveShift + currentExtent * 0.5f;
  lightMin.x -= predictiveOrthoPadding;
  lightMax.x += predictiveOrthoPadding;
  lightMin.y -= predictiveOrthoPadding;
  lightMax.y += predictiveOrthoPadding;
  lightMin.z -= predictiveDepthPadding;
  lightMax.z += predictiveDepthPadding;
  const glm::mat4 inverseLightView = glm::inverse(fit.lightView);
  shadow_detail::stabilizeOrthoBounds(lightMin, lightMax, shadowMapSize, true,
                                      fit, inverseLightView);
  shadow_detail::quantizeShadowZBounds(lightMin, lightMax, fit.texelWorldSize,
                                       true);
  const float nearPlane = -lightMax.z;
  float farPlane = -lightMin.z;
  if (farPlane <= nearPlane + 0.01f) {
    farPlane = nearPlane + 0.01f;
  }
  fit.lightProj = glm::orthoRH_ZO(lightMin.x, lightMax.x, lightMin.y,
                                  lightMax.y, nearPlane, farPlane);
  fit.lightViewProj = fit.lightProj * fit.lightView;
  fit.lightSpaceBoundsMin = glm::vec3(lightMin.x, lightMin.y, -farPlane);
  fit.lightSpaceBoundsMax = glm::vec3(lightMax.x, lightMax.y, -nearPlane);
}
[[nodiscard]] bool shadowCasterOverlapsDirectionalShadowFit(
    std::span<const glm::vec3, 8> casterWorldCorners,
    const shadow_detail::DirectionalShadowFit &fit) {
  const float cullingPadding =
      std::max(fit.texelWorldSize * kCullingPaddingTexelMultiplier, 0.01f);
  return shadow_detail::shadowCasterOverlapsLightSpaceBounds(
      casterWorldCorners, fit.lightView, fit.lightSpaceBoundsMin,
      fit.lightSpaceBoundsMax, cullingPadding);
}
[[nodiscard]] bool lightSpaceBoundsOverlap(glm::vec3 casterMin,
                                           glm::vec3 casterMax,
                                           glm::vec3 cascadeMin,
                                           glm::vec3 cascadeMax,
                                           float padding) {
  const glm::vec3 normalizedCasterMin = glm::min(casterMin, casterMax);
  const glm::vec3 normalizedCasterMax = glm::max(casterMin, casterMax);
  casterMin = normalizedCasterMin;
  casterMax = normalizedCasterMax;
  const glm::vec3 normalizedCascadeMin = glm::min(cascadeMin, cascadeMax);
  const glm::vec3 normalizedCascadeMax = glm::max(cascadeMin, cascadeMax);
  cascadeMin = normalizedCascadeMin - glm::vec3(padding);
  cascadeMax = normalizedCascadeMax + glm::vec3(padding);
  return casterMax.x >= cascadeMin.x && casterMin.x <= cascadeMax.x &&
         casterMax.y >= cascadeMin.y && casterMin.y <= cascadeMax.y &&
         casterMax.z >= cascadeMin.z && casterMin.z <= cascadeMax.z;
}
[[nodiscard]] bool normalizedLightSpaceBoundsOverlap(
    const glm::vec3 &casterMin, const glm::vec3 &casterMax,
    const glm::vec3 &cascadeMin, const glm::vec3 &cascadeMax) {
  return casterMax.x >= cascadeMin.x && casterMin.x <= cascadeMax.x &&
         casterMax.y >= cascadeMin.y && casterMin.y <= cascadeMax.y &&
         casterMax.z >= cascadeMin.z && casterMin.z <= cascadeMax.z;
}
[[nodiscard]] bool staticOnlyRenderedFitContainsCurrent(
    const shadow_detail::DirectionalShadowFit &renderedFit,
    const shadow_detail::DirectionalShadowFit &currentFit) {
  return shadow_detail::directionalShadowFitContains(renderedFit, currentFit);
}
[[nodiscard]] shadow_detail::DirectionalShadowFit
makeStaticOnlyReuseFitForCurrentFrame(
    const shadow_detail::DirectionalShadowFit &cachedRenderedFit,
    const shadow_detail::DirectionalShadowFit &currentRawFit) {
  shadow_detail::DirectionalShadowFit fit = cachedRenderedFit;
  fit.splitNear = currentRawFit.splitNear;
  fit.splitFar = currentRawFit.splitFar;
  fit.frustumCenter = currentRawFit.frustumCenter;
  fit.frustumCorners = currentRawFit.frustumCorners;
  return fit;
}
[[nodiscard]] glm::vec3 paddedLightSpaceCullingMin(
    const shadow_detail::DirectionalShadowFit &fit) noexcept {
  const float padding =
      std::max(fit.texelWorldSize * kCullingPaddingTexelMultiplier, 0.01f);
  return glm::min(fit.lightSpaceBoundsMin, fit.lightSpaceBoundsMax) -
         glm::vec3(padding);
}
[[nodiscard]] glm::vec3 paddedLightSpaceCullingMax(
    const shadow_detail::DirectionalShadowFit &fit) noexcept {
  const float padding =
      std::max(fit.texelWorldSize * kCullingPaddingTexelMultiplier, 0.01f);
  return glm::max(fit.lightSpaceBoundsMin, fit.lightSpaceBoundsMax) +
         glm::vec3(padding);
}
void writeShadowCascadeFit(const shadow_detail::DirectionalShadowFit &fit,
                           uint32_t cascadeIndex,
                           TextureHandle shadowDepthTexture, GPUDevice &gpu,
                           ShadowFrameGpuData &shadowFrameGpuData,
                           ShadowDebugFrameData &shadowDebugFrameData) {
  shadowFrameGpuData.cascades[cascadeIndex] = ShadowCascadeGpuData{
      .lightViewProj = fit.lightViewProj,
      .splitDepthTexelSize =
          glm::vec4(fit.splitNear, fit.splitFar, fit.texelWorldSize, 0.0f),
  };
  shadowFrameGpuData.cascadeTextureIds[cascadeIndex] =
      gpu.getTextureBindlessIndex(shadowDepthTexture);
  ShadowCascadeDebugFrameData &cascadeDebug =
      shadowDebugFrameData.cascades[cascadeIndex];
  cascadeDebug.splitNear = fit.splitNear;
  cascadeDebug.splitFar = fit.splitFar;
  cascadeDebug.texelWorldSize = fit.texelWorldSize;
  cascadeDebug.lightView = fit.lightView;
  cascadeDebug.lightProj = fit.lightProj;
  cascadeDebug.lightViewProj = fit.lightViewProj;
  cascadeDebug.inverseLightView = glm::inverse(fit.lightView);
  cascadeDebug.inverseLightProj = glm::inverse(fit.lightProj);
  cascadeDebug.lightSpaceBoundsMin = glm::vec4(fit.lightSpaceBoundsMin, 1.0f);
  cascadeDebug.lightSpaceBoundsMax = glm::vec4(fit.lightSpaceBoundsMax, 1.0f);
  cascadeDebug.unsnappedCenter = glm::vec4(fit.unsnappedCenter, 1.0f);
  cascadeDebug.snappedCenter = glm::vec4(fit.snappedCenter, 1.0f);
  for (size_t i = 0u; i < fit.frustumCorners.size(); ++i) {
    cascadeDebug.worldFrustumCorners[i] =
        glm::vec4(fit.frustumCorners[i], 1.0f);
  }
}
[[nodiscard]] shadow_detail::DirectionalShadowFit
makeDirectionalShadowFitFromDebugCascade(
    const ShadowCascadeDebugFrameData &cascade) {
  shadow_detail::DirectionalShadowFit fit{};
  fit.splitNear = cascade.splitNear;
  fit.splitFar = cascade.splitFar;
  fit.texelWorldSize = cascade.texelWorldSize;
  fit.lightView = cascade.lightView;
  fit.lightProj = cascade.lightProj;
  fit.lightViewProj = cascade.lightViewProj;
  fit.lightSpaceBoundsMin = glm::vec3(cascade.lightSpaceBoundsMin);
  fit.lightSpaceBoundsMax = glm::vec3(cascade.lightSpaceBoundsMax);
  fit.unsnappedCenter = glm::vec3(cascade.unsnappedCenter);
  fit.snappedCenter = glm::vec3(cascade.snappedCenter);
  glm::vec3 frustumCenter(0.0f);
  for (size_t i = 0u; i < fit.frustumCorners.size(); ++i) {
    fit.frustumCorners[i] = glm::vec3(cascade.worldFrustumCorners[i]);
    frustumCenter += fit.frustumCorners[i];
  }
  fit.frustumCenter =
      frustumCenter / static_cast<float>(fit.frustumCorners.size());
  const glm::vec3 unsnappedLightSpaceCenter =
      glm::vec3(fit.lightView * glm::vec4(fit.unsnappedCenter, 1.0f));
  const glm::vec3 snappedLightSpaceCenter =
      glm::vec3(fit.lightView * glm::vec4(fit.snappedCenter, 1.0f));
  fit.unsnappedLightSpaceCenter =
      glm::vec2(unsnappedLightSpaceCenter.x, unsnappedLightSpaceCenter.y);
  fit.snappedLightSpaceCenter =
      glm::vec2(snappedLightSpaceCenter.x, snappedLightSpaceCenter.y);
  return fit;
}
[[nodiscard]] shadow_detail::DirectionalShadowFit fitSingleShadowMapForRange(
    const CameraFrameState &camera, ShadowSplitRange range,
    glm::vec3 lightDirection, uint32_t shadowMapSize, bool stabilize,
    bool hasCasterBounds, bool hasAnimatedGeometryOverrides,
    glm::vec3 casterBoundsMin, glm::vec3 casterBoundsMax) {
  const CameraFrameState adjustedCamera =
      cameraWithShadowSplitRange(camera, range);
  if (hasCasterBounds && !hasAnimatedGeometryOverrides) {
    return shadow_detail::fitDirectionalShadowMapToBounds(
        adjustedCamera, casterBoundsMin, casterBoundsMax, lightDirection,
        range.farDepth, shadowMapSize, stabilize);
  }
  return shadow_detail::fitSingleDirectionalShadowMap(
      adjustedCamera, lightDirection, range.farDepth, shadowMapSize, stabilize);
}
[[nodiscard]] RenderPipelineDesc
shadowDepthPipelineDesc(ShaderHandle vertexShader, ShaderHandle fragmentShader,
                        CullMode cullMode, Format depthFormat,
                        RasterPipelineState rasterState) {
  return RenderPipelineDesc{
      .vertexInput = {},
      .vertexShader = vertexShader,
      .fragmentShader = fragmentShader,
      .colorFormats = {Format::RGBA8_UNORM},
      .colorAttachmentCount = 0u,
      .depthFormat = sanitizeShadowDepthFormat(depthFormat),
      .cullMode = cullMode,
      .polygonMode = PolygonMode::Fill,
      .topology = Topology::Triangle,
      .blendEnabled = false,
      .rasterState = canonicalRasterPipelineState(rasterState),
  };
}
[[nodiscard]] RenderPipelineDesc
shadowPreviewPipelineDesc(ShaderHandle vertexShader,
                          ShaderHandle fragmentShader) {
  return RenderPipelineDesc{
      .vertexInput = {},
      .vertexShader = vertexShader,
      .fragmentShader = fragmentShader,
      .colorFormats = {Format::RGBA8_UNORM},
      .depthFormat = Format::Count,
      .cullMode = CullMode::None,
      .polygonMode = PolygonMode::Fill,
      .topology = Topology::Triangle,
      .blendEnabled = false,
  };
}
[[nodiscard]] uint32_t visibilityReadbackAge(uint64_t currentFrame,
                                             uint32_t sourceFrame) {
  if (static_cast<uint64_t>(sourceFrame) >= currentFrame) {
    return 0u;
  }
  const uint64_t age = currentFrame - static_cast<uint64_t>(sourceFrame);
  return age > std::numeric_limits<uint32_t>::max()
             ? std::numeric_limits<uint32_t>::max()
             : static_cast<uint32_t>(age);
}
[[nodiscard]] bool isValidShadowDepthTexture(const GPUDevice &gpu,
                                             TextureHandle texture,
                                             uint32_t shadowMapSize,
                                             Format depthFormat) {
  if (!nuri::isValid(texture)) {
    return false;
  }
  const TextureDimensions dimensions = gpu.getTextureDimensions(texture);
  return dimensions.width == shadowMapSize &&
         dimensions.height == shadowMapSize &&
         gpu.getTextureFormat(texture) ==
             sanitizeShadowDepthFormat(depthFormat);
}
[[nodiscard]] TextureDimensions
shadowPreviewDimensions(uint32_t shadowMapSize, ShadowPreviewMode mode) {
  if (sanitizeShadowPreviewMode(mode) == ShadowPreviewMode::TiledAllCascades) {
    return TextureDimensions{
        .width = shadowMapSize * 2u,
        .height = shadowMapSize * 2u,
        .depth = 1u,
    };
  }
  return TextureDimensions{
      .width = shadowMapSize,
      .height = shadowMapSize,
      .depth = 1u,
  };
}
[[nodiscard]] bool isValidShadowPreviewTexture(const GPUDevice &gpu,
                                               TextureHandle texture,
                                               uint32_t shadowMapSize,
                                               ShadowPreviewMode mode) {
  if (!nuri::isValid(texture) ||
      gpu.getTextureFormat(texture) != Format::RGBA8_UNORM) {
    return false;
  }
  const TextureDimensions dimensions = gpu.getTextureDimensions(texture);
  const TextureDimensions expected =
      shadowPreviewDimensions(shadowMapSize, mode);
  return dimensions.width == expected.width &&
         dimensions.height == expected.height;
}
[[nodiscard]] uint64_t
makeStaticOnlyCascadeRasterSignatureSeed(const ShadowCascadeGpuData &cascade,
                                         float constantBias, float slopeBias) {
  const uint64_t lightViewProjSignature = hashBytes(
      std::as_bytes(std::span<const glm::mat4>(&cascade.lightViewProj, 1u)));
  uint64_t biasSignature = kFnvOffsetBasis64;
  biasSignature = hashCombineValue(biasSignature, constantBias);
  biasSignature = hashCombineValue(biasSignature, slopeBias);
  uint64_t signature = kFnvOffsetBasis64;
  signature = hashCombine64(signature, lightViewProjSignature);
  signature = hashCombine64(signature, biasSignature);
  return signature;
}
[[nodiscard]] uint64_t makeStaticOnlyCascadeLightViewProjSignature(
    const ShadowCascadeGpuData &cascade) {
  return hashBytes(
      std::as_bytes(std::span<const glm::mat4>(&cascade.lightViewProj, 1u)));
}
[[nodiscard]] uint64_t makeStaticOnlyCascadeBiasSignature(float constantBias,
                                                          float slopeBias) {
  uint64_t signature = kFnvOffsetBasis64;
  signature = hashCombineValue(signature, constantBias);
  signature = hashCombineValue(signature, slopeBias);
  return signature;
}
template <typename Entry>
[[nodiscard]] uint64_t
hashStaticShadowCasterRasterSignature(uint64_t signature, const Entry &entry) {
  signature =
      hashCombine64(signature, static_cast<uint64_t>(entry.templateIndex));
  signature =
      hashCombine64(signature, static_cast<uint64_t>(entry.instanceIndex));
  signature =
      hashCombine64(signature, foldHandle(entry.indexBuffer.index,
                                          entry.indexBuffer.generation));
  signature = hashCombineValue(signature, entry.indexBufferOffset);
  signature =
      hashCombine64(signature, static_cast<uint64_t>(entry.indexFormat));
  signature =
      hashCombine64(signature, foldHandle(entry.vertexBuffer.index,
                                          entry.vertexBuffer.generation));
  signature =
      hashCombine64(signature, foldHandle(entry.vertexDecodeBuffer.index,
                                          entry.vertexDecodeBuffer.generation));
  signature = hashCombineValue(signature, entry.vertexBufferAddress);
  signature = hashCombineValue(signature, entry.vertexDecodeBufferAddress);
  signature =
      hashCombine64(signature, static_cast<uint64_t>(entry.vertexDecodeIndex));
  signature =
      hashCombine64(signature, static_cast<uint64_t>(entry.packedVertexFormat));
  signature =
      hashCombine64(signature, static_cast<uint64_t>(entry.materialIndex));
  signature = hashCombine64(signature, static_cast<uint64_t>(entry.indexCount));
  signature = hashCombine64(signature, static_cast<uint64_t>(entry.firstIndex));
  signature = hashCombine64(signature, entry.doubleSided ? 1ull : 0ull);
  signature = hashCombine64(signature, entry.alphaMasked ? 1ull : 0ull);
  return signature;
}
} // namespace
ShadowRenderer::ShadowRenderer(GPUDevice &gpu,
                               std::pmr::memory_resource *memory)
    : ShadowRenderer(gpu, ShadowRendererConfig{}, memory) {}

ShadowRenderer::ShadowRenderer(GPUDevice &gpu,
                               const ShadowRendererConfig &config,
                               std::pmr::memory_resource *memory)
    : gpu_(gpu), config_(config), memory_(resolveMemoryResource(memory)),
      bufferRings_(memory_), frameSlotStates_(memory_),
      meshDrawTemplates_(std::pmr::new_delete_resource()),
      batchBuildScratchArena_(memory_),
      staticShadowTemplateIndices_(std::pmr::new_delete_resource()),
      dynamicShadowTemplateIndices_(std::pmr::new_delete_resource()),
      staticShadowCasterCache_(std::pmr::new_delete_resource()),
      staticShadowBatchTemplates_(std::pmr::new_delete_resource()),
      staticShadowBatchIndexMap_(std::pmr::new_delete_resource()),
      staticShadowBatchInstanceIndices_(std::pmr::new_delete_resource()),
      staticShadowCasterDrawBuffers_(std::pmr::new_delete_resource()),
      staticShadowCasterFitPoints_(std::pmr::new_delete_resource()),
      staticLightGridCells_{
          std::pmr::vector<StaticShadowCasterLightGridCell>(memory_),
          std::pmr::vector<StaticShadowCasterLightGridCell>(memory_)},
      staticLightGridEntries_{std::pmr::vector<uint32_t>(memory_),
                              std::pmr::vector<uint32_t>(memory_)},
      staticLargeLightGridEntries_{std::pmr::vector<uint32_t>(memory_),
                                   std::pmr::vector<uint32_t>(memory_)},
      staticShadowCasterLightGridQueryMarks_(memory_),
      staticShadowCasterLightGridQueryEntries_(memory_),
      instanceMatrices_(std::pmr::new_delete_resource()),
      instanceRemap_(memory_), shadowDrawPacketUploadBytes_(memory_),
      passBufferDependencies_(memory_), passDependencyBuffers_(memory_),
      passDependencyBufferAccessModes_(memory_),
      passDependencyTextures_(memory_),
      passDependencyTextureAccessModes_(memory_),
      preResolvedDrawBuffers_(memory_),
      passTextureDependencies_(std::pmr::new_delete_resource()),
      previewTextureDependencies_(memory_) {
  bufferRings_.resize(BufferRingCount);
  for (uint32_t cascadeIndex = 0u; cascadeIndex < kMaxShadowCascades;
       ++cascadeIndex) {
    cascadePushConstants_[cascadeIndex] =
        std::pmr::vector<PushConstants>(memory_);
    cascadeDrawItems_[cascadeIndex] = std::pmr::vector<DrawItem>(memory_);
    cascadeIndirectPushConstants_[cascadeIndex] =
        std::pmr::vector<PushConstants>(memory_);
    cascadeIndirectDrawItems_[cascadeIndex] =
        std::pmr::vector<DrawItem>(memory_);
  }
}

ShadowRenderer::~ShadowRenderer() {
  destroyShadowResources();
  destroyBuffers();
  destroyPipelineState();
  destroyShaders();
}

Result<bool, std::string> ShadowRenderer::createShaders() {
  destroyShaders();
  if (config_.shaderBasePath.empty()) {
    return Result<bool, std::string>::makeError(
        "ShadowRenderer::createShaders: shader base path is not configured");
  }
  struct ShaderSpec {
    std::string_view name;
    ShaderHandle *handle;
    std::string_view file;
    ShaderStage stage;
  };
  const std::array specs{
      ShaderSpec{"shadow_depth_opaque", &shaders_[ShadowOpaqueVertex],
                 "shadow_depth_opaque.vert", ShaderStage::Vertex},
      ShaderSpec{"shadow_depth", &shaders_[ShadowVertex], "shadow_depth.vert",
                 ShaderStage::Vertex},
      ShaderSpec{"shadow_depth_only", &shaders_[DepthFragment],
                 "opaque_depth.frag", ShaderStage::Fragment},
      ShaderSpec{"shadow_depth_alpha", &shaders_[DepthAlphaFragment],
                 "opaque_depth_alpha.frag", ShaderStage::Fragment},
  };
  for (const ShaderSpec &spec : specs) {
    auto result = Shader{spec.name, gpu_}.compileFromFile(
        (config_.shaderBasePath / spec.file).string(), spec.stage);
    if (result.hasError()) {
      return Result<bool, std::string>::makeError(result.error());
    }
    *spec.handle = result.value();
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> ShadowRenderer::createPreviewShaders() {
  Shader shader{"shadow_preview", gpu_};
  auto vertexResult = shader.compileFromFile(
      (config_.shaderBasePath / "shadow_preview.vert").string(),
      ShaderStage::Vertex);
  if (vertexResult.hasError()) {
    return Result<bool, std::string>::makeError(vertexResult.error());
  }
  auto fragmentResult = shader.compileFromFile(
      (config_.shaderBasePath / "shadow_preview.frag").string(),
      ShaderStage::Fragment);
  if (fragmentResult.hasError()) {
    gpu_.destroyShaderModule(vertexResult.value());
    return Result<bool, std::string>::makeError(fragmentResult.error());
  }
  shaders_[PreviewVertex] = vertexResult.value();
  shaders_[PreviewFragment] = fragmentResult.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> ShadowRenderer::createSdsmReduceShaders() {
  Shader shader{"shadow_sdsm_prev_frame_minmax", gpu_};
  const std::filesystem::path computePath =
      config_.shaderBasePath / "shadow_sdsm_prev_frame_minmax.comp";
  auto computeResult =
      shader.compileFromFile(computePath.string(), ShaderStage::Compute);
  if (computeResult.hasError()) {
    return Result<bool, std::string>::makeError(computeResult.error());
  }
  shaders_[SdsmReduceCompute] = computeResult.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
ShadowRenderer::createPipelines(Format depthFormat,
                                RasterPipelineState rasterState) {
  const Format targetDepthFormat = sanitizeShadowDepthFormat(depthFormat);
  const RasterPipelineState targetRasterState =
      canonicalRasterPipelineState(rasterState);
  struct PipelineSpec {
    ShaderHandle vertex;
    ShaderHandle fragment;
    CullMode cull;
    std::string_view name;
  };
  const std::array specs{
      PipelineSpec{shaders_[ShadowOpaqueVertex], shaders_[DepthFragment],
                   CullMode::Back, "shadow_depth_opaque"},
      PipelineSpec{shaders_[ShadowOpaqueVertex], shaders_[DepthFragment],
                   CullMode::None, "shadow_depth_opaque_double_sided"},
      PipelineSpec{shaders_[ShadowVertex], shaders_[DepthAlphaFragment],
                   CullMode::Back, "shadow_depth_alpha"},
      PipelineSpec{shaders_[ShadowVertex], shaders_[DepthAlphaFragment],
                   CullMode::None, "shadow_depth_alpha_double_sided"},
  };
  std::array<RenderPipelineHandle, 4> created{};
  for (size_t index = 0u; index < specs.size(); ++index) {
    const PipelineSpec &spec = specs[index];
    auto result = gpu_.createRenderPipeline(
        shadowDepthPipelineDesc(spec.vertex, spec.fragment, spec.cull,
                                targetDepthFormat, targetRasterState),
        spec.name);
    if (result.hasError()) {
      for (RenderPipelineHandle pipeline : created) {
        if (nuri::isValid(pipeline)) {
          gpu_.destroyRenderPipeline(pipeline);
        }
      }
      return Result<bool, std::string>::makeError(result.error());
    }
    created[index] = result.value();
  }
  for (RenderPipelineHandle pipeline : shadowPipelines_) {
    if (nuri::isValid(pipeline)) {
      gpu_.destroyRenderPipeline(pipeline);
    }
  }
  shadowPipelines_ = created;
  shadowDepthPipelineFormat_ = targetDepthFormat;
  shadowPipelineRasterState_ = targetRasterState;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> ShadowRenderer::createPreviewPipeline() {
  auto pipelineResult = gpu_.createRenderPipeline(
      shadowPreviewPipelineDesc(shaders_[PreviewVertex],
                                shaders_[PreviewFragment]),
      "shadow_depth_preview");
  if (pipelineResult.hasError()) {
    return Result<bool, std::string>::makeError(pipelineResult.error());
  }
  previewPipelineHandle_ = pipelineResult.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> ShadowRenderer::createSdsmReducePipeline() {
  auto pipelineResult = gpu_.createComputePipeline(
      ComputePipelineDesc{.computeShader = shaders_[SdsmReduceCompute]},
      "shadow_sdsm_prev_frame_minmax");
  if (pipelineResult.hasError()) {
    return Result<bool, std::string>::makeError(pipelineResult.error());
  }
  sdsmReducePipelineHandle_ = pipelineResult.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> ShadowRenderer::ensureInitialized() {
  if (initialized_) {
    return Result<bool, std::string>::makeResult(true);
  }
  const auto run = [this](auto operation) { return (this->*operation)(); };
  for (auto operation :
       {&ShadowRenderer::createShaders, &ShadowRenderer::createPreviewShaders,
        &ShadowRenderer::createSdsmReduceShaders,
        &ShadowRenderer::createPreviewPipeline,
        &ShadowRenderer::createSdsmReducePipeline}) {
    auto result = run(operation);
    if (result.hasError()) {
      return result;
    }
  }
  initialized_ = true;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> ShadowRenderer::ensureShadowResources(
    const RenderSettings::ShadowSettings &settings) {
  const bool previewEnabled = settings.debug.showShadowMapViewport;
  const ShadowPreviewMode previewMode =
      sanitizeShadowPreviewMode(settings.debug.previewMode);
  const auto createPreviewTexture = [&]() {
    return gpu_.createTexture(
        TextureDesc{
            .dimensions =
                shadowPreviewDimensions(settings.shadowMapSize, previewMode),
            .usage = TextureUsage::AttachmentSampled,
        },
        "shadow_depth_preview_phase4");
  };
  const uint32_t requestedCascadeCount =
      std::clamp(settings.cascadeCount, 1u, kMaxShadowCascades);
  const Format targetDepthFormat =
      sanitizeShadowDepthFormat(settings.depthFormat);
  bool depthValid = nuri::isValid(rawDepthSampler_) &&
                    nuri::isValid(compareDepthSampler_) &&
                    activeCascadeCount_ == requestedCascadeCount;
  for (uint32_t cascadeIndex = 0u;
       depthValid && cascadeIndex < requestedCascadeCount; ++cascadeIndex) {
    depthValid =
        isValidShadowDepthTexture(gpu_, shadowDepthTextures_[cascadeIndex],
                                  settings.shadowMapSize, targetDepthFormat);
  }
  for (uint32_t cascadeIndex = requestedCascadeCount;
       depthValid && cascadeIndex < kMaxShadowCascades; ++cascadeIndex) {
    depthValid = !nuri::isValid(shadowDepthTextures_[cascadeIndex]);
  }
  const bool previewTextureValid = isValidShadowPreviewTexture(
      gpu_, shadowDebugPreviewTexture_, settings.shadowMapSize, previewMode);
  const bool previewValid = previewEnabled
                                ? previewTextureValid
                                : !nuri::isValid(shadowDebugPreviewTexture_);
  if (depthValid && previewValid) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (depthValid) {
    if (previewEnabled && !previewTextureValid) {
      const TextureHandle oldPreviewTexture = shadowDebugPreviewTexture_;
      auto previewResult = createPreviewTexture();
      if (previewResult.hasError()) {
        return Result<bool, std::string>::makeError(previewResult.error());
      }
      shadowDebugPreviewTexture_ = previewResult.value();
      if (nuri::isValid(oldPreviewTexture)) {
        gpu_.destroyTexture(oldPreviewTexture);
      }
      return Result<bool, std::string>::makeResult(true);
    }
    if (!previewEnabled && nuri::isValid(shadowDebugPreviewTexture_)) {
      gpu_.destroyTexture(shadowDebugPreviewTexture_);
      shadowDebugPreviewTexture_ = {};
      return Result<bool, std::string>::makeResult(true);
    }
  }
  std::array<TextureHandle, kMaxShadowCascades> newShadowDepthTextures{};
  const auto destroyNewDepthTextures = [&]() {
    for (TextureHandle texture : newShadowDepthTextures) {
      if (nuri::isValid(texture)) {
        gpu_.destroyTexture(texture);
      }
    }
  };
  const TextureDesc shadowDesc{
      .format = targetDepthFormat,
      .dimensions = {.width = settings.shadowMapSize,
                     .height = settings.shadowMapSize,
                     .depth = 1u},
      .usage = TextureUsage::AttachmentSampled,
  };
  for (uint32_t cascadeIndex = 0u; cascadeIndex < requestedCascadeCount;
       ++cascadeIndex) {
    const std::string debugName =
        "shadow_depth_map_phase4_cascade" + std::to_string(cascadeIndex);
    auto textureResult = gpu_.createTexture(shadowDesc, debugName);
    if (textureResult.hasError()) {
      destroyNewDepthTextures();
      return Result<bool, std::string>::makeError(textureResult.error());
    }
    newShadowDepthTextures[cascadeIndex] = textureResult.value();
  }
  const SamplerDesc rawSamplerDesc{
      .minFilter = SamplerFilter::Nearest,
      .magFilter = SamplerFilter::Nearest,
      .wrapU = SamplerWrapMode::Clamp,
      .wrapV = SamplerWrapMode::Clamp,
      .wrapW = SamplerWrapMode::Clamp,
      .mipLodMax = 0.0f,
  };
  auto samplerResult =
      gpu_.createSampler(rawSamplerDesc, "shadow_raw_depth_sampler");
  if (samplerResult.hasError()) {
    destroyNewDepthTextures();
    return Result<bool, std::string>::makeError(samplerResult.error());
  }
  const SamplerHandle newRawDepthSampler = samplerResult.value();
  SamplerDesc compareSamplerDesc = rawSamplerDesc;
  const SamplerFilter compareSamplerFilter =
      gpu_.supportsSampledImageLinearFiltering(targetDepthFormat)
          ? SamplerFilter::Linear
          : SamplerFilter::Nearest;
  compareSamplerDesc.minFilter = compareSamplerFilter;
  compareSamplerDesc.magFilter = compareSamplerFilter;
  compareSamplerDesc.depthCompareEnabled = true;
  compareSamplerDesc.depthCompareOp = CompareOp::LessEqual;
  auto compareSamplerResult =
      gpu_.createSampler(compareSamplerDesc, "shadow_compare_depth_sampler");
  if (compareSamplerResult.hasError()) {
    gpu_.destroySampler(newRawDepthSampler);
    destroyNewDepthTextures();
    return Result<bool, std::string>::makeError(compareSamplerResult.error());
  }
  const SamplerHandle newCompareDepthSampler = compareSamplerResult.value();
  TextureHandle newPreviewTexture{};
  if (previewEnabled) {
    auto previewResult = createPreviewTexture();
    if (previewResult.hasError()) {
      gpu_.destroySampler(newCompareDepthSampler);
      gpu_.destroySampler(newRawDepthSampler);
      destroyNewDepthTextures();
      return Result<bool, std::string>::makeError(previewResult.error());
    }
    newPreviewTexture = previewResult.value();
  }
  const std::array<TextureHandle, kMaxShadowCascades> oldShadowDepthTextures =
      shadowDepthTextures_;
  const TextureHandle oldPreviewTexture = shadowDebugPreviewTexture_;
  const SamplerHandle oldRawDepthSampler = rawDepthSampler_;
  const SamplerHandle oldCompareDepthSampler = compareDepthSampler_;
  shadowDepthTextures_ = newShadowDepthTextures;
  shadowDebugPreviewTexture_ = newPreviewTexture;
  rawDepthSampler_ = newRawDepthSampler;
  compareDepthSampler_ = newCompareDepthSampler;
  shadowMapSize_ = settings.shadowMapSize;
  activeCascadeCount_ = requestedCascadeCount;
  if (nuri::isValid(oldCompareDepthSampler)) {
    gpu_.destroySampler(oldCompareDepthSampler);
  }
  if (nuri::isValid(oldRawDepthSampler)) {
    gpu_.destroySampler(oldRawDepthSampler);
  }
  if (nuri::isValid(oldPreviewTexture)) {
    gpu_.destroyTexture(oldPreviewTexture);
  }
  for (const TextureHandle texture : oldShadowDepthTextures) {
    if (nuri::isValid(texture)) {
      gpu_.destroyTexture(texture);
    }
  }
  resetFrozenShadowFit();
  resetCascadeStabilizationHistory();
  return Result<bool, std::string>::makeResult(true);
}

void ShadowRenderer::ensureRingBufferCount(uint32_t requiredCount) {
  const std::array rings{
      &bufferRings_[InstanceMatricesRing], &bufferRings_[InstanceRemapRing],
      &bufferRings_[ShadowDrawPacketRing], &bufferRings_[ShadowFrameRing]};
  const uint32_t safeCount = growDynamicBufferRings(requiredCount, rings);
  frameSlotStates_.resize(safeCount);
}

Result<bool, std::string> ShadowRenderer::ensureRingCapacity(
    BufferRingSlot slot, size_t requiredBytes, std::string_view debugName,
    uint64_t FrameSlotState::*version, Storage storage) {
  return ensureDynamicBufferRingCapacity(
      gpu_, bufferRings_[slot],
      BufferDesc{.usage = BufferUsage::Storage,
                 .storage = storage,
                 .size = requiredBytes},
      debugName, [this, version](size_t i) {
        frameSlotStates_[i].*version = std::numeric_limits<uint64_t>::max();
      });
}

Result<bool, std::string>
ShadowRenderer::ensureInstanceMatricesRingCapacity(size_t requiredBytes) {
  return ensureRingCapacity(
      InstanceMatricesRing, requiredBytes, "shadow_instance_matrices",
      &FrameSlotState::instanceUploadVersion, Storage::HostVisible);
}

Result<bool, std::string>
ShadowRenderer::ensureInstanceRemapRingCapacity(size_t requiredBytes) {
  return ensureRingCapacity(
      InstanceRemapRing, requiredBytes, "shadow_instance_remap",
      &FrameSlotState::remapUploadSignature, Storage::HostVisible);
}

Result<bool, std::string>
ShadowRenderer::ensureShadowFrameRingCapacity(size_t requiredBytes) {
  return ensureRingCapacity(
      ShadowFrameRing, requiredBytes, "shadow_frame_gpu_data",
      &FrameSlotState::frameUploadSignature, Storage::HostVisible);
}

Result<bool, std::string>
ShadowRenderer::ensureSdsmReduceResultRingCount(uint32_t requiredCount) {
  const uint32_t safeCount = std::max(requiredCount, 1u);
  while (bufferRings_[SdsmReduceResultRing].size() < safeCount) {
    bufferRings_[SdsmReduceResultRing].push_back(DynamicBufferSlot{});
  }
  frameSlotStates_.resize(safeCount);
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
ShadowRenderer::ensureSdsmReduceResultRingCapacity(size_t requiredBytes) {
  const uint64_t invalidPublishedFrame = std::numeric_limits<uint64_t>::max();
  for (size_t slotIndex = 0u;
       slotIndex < bufferRings_[SdsmReduceResultRing].size(); ++slotIndex) {
    DynamicBufferSlot &slot = bufferRings_[SdsmReduceResultRing][slotIndex];
    if (slot.buffer && slot.buffer->valid() &&
        slot.capacityBytes >= requiredBytes) {
      continue;
    }
    const size_t replacementCapacity =
        nextDynamicBufferCapacity(slot.capacityBytes, requiredBytes);
    auto bufferResult =
        Buffer::create(gpu_,
                       BufferDesc{.usage = BufferUsage::Storage,
                                  .storage = Storage::HostVisible,
                                  .size = replacementCapacity},
                       "shadow_sdsm_minmax_result");
    if (bufferResult.hasError()) {
      return Result<bool, std::string>::makeError(bufferResult.error());
    }
    SdsmGpuMinMaxResult clearedResult{};
    auto clearResult = gpu_.updateBuffer(
        bufferResult.value()->handle(),
        std::as_bytes(std::span<const SdsmGpuMinMaxResult>(&clearedResult, 1u)),
        0u);
    if (clearResult.hasError()) {
      return Result<bool, std::string>::makeError(clearResult.error());
    }
    std::unique_ptr<Buffer> previous = std::move(slot.buffer);
    slot.buffer = std::move(bufferResult.value());
    slot.capacityBytes = replacementCapacity;
    frameSlotStates_[slotIndex].sdsmPublishedFrame = invalidPublishedFrame;
    previous.reset();
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> ShadowRenderer::prepareSceneCache(
    SceneDrawDatabase &database, const RenderScene &scene,
    const ResourceManager &resources, const RenderSettings *) {
  auto result = database.update(scene, resources);
  if (result.hasError()) {
    return result;
  }
  rebuildSceneCache(database);
  cachedScene_ = &scene;
  cachedTopologyVersion_ = scene.topologyVersion();
  cachedMaterialVersion_ = resources.materialVersion();
  cachedModelMaterialBindingVersion_ = resources.modelMaterialBindingVersion();
  cachedDeformationVersion_ = scene.deformationVersion();
  cachedGeometryMutationVersion_ = gpu_.geometryMutationVersion();
  return Result<bool, std::string>::makeResult(true);
}
void ShadowRenderer::rebuildSceneCache(const SceneDrawDatabase &database) {
  meshDrawTemplates_.clear();
  staticShadowTemplateIndices_.clear();
  dynamicShadowTemplateIndices_.clear();
  passTextureDependencies_.clear();
  meshDrawTemplates_.reserve(database.draws().size());
  for (const SceneDrawRecord &draw : database.draws()) {
    if (draw.alphaBlended) {
      continue;
    }
    meshDrawTemplates_.push_back(draw);
    const uint32_t index =
        static_cast<uint32_t>(meshDrawTemplates_.size() - 1u);
    (draw.dynamicCaster ? dynamicShadowTemplateIndices_
                        : staticShadowTemplateIndices_)
        .push_back(index);
    if (draw.alphaMasked) {
      appendUniqueTextureDependency(passTextureDependencies_,
                                    draw.baseColorTexture);
    }
  }
  invalidateStaticShadowCasterCache();
}
void ShadowRenderer::rebuildStaticShadowCasterCache(
    const RenderScene &scene, const RenderSettings &settings) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_COPY);
  staticShadowCasterCache_.clear();
  staticShadowBatchTemplates_.clear();
  staticShadowBatchIndexMap_.clear();
  staticShadowBatchInstanceIndices_.clear();
  staticShadowCasterDrawBuffers_.clear();
  staticShadowCasterFitPoints_.clear();
  for (auto &cells : staticLightGridCells_)
    cells.clear();
  for (auto &entries : staticLightGridEntries_)
    entries.clear();
  for (auto &entries : staticLargeLightGridEntries_)
    entries.clear();
  staticShadowCasterLightGridQueryMarks_.clear();
  staticShadowCasterLightGridQueryEntries_.clear();
  staticShadowCasterLightGridQueryMarker_ = 1u;
  staticLightGrids_ = {};
  staticShadowCasterBoundsMin_ = glm::vec3(std::numeric_limits<float>::max());
  staticShadowCasterBoundsMax_ =
      glm::vec3(std::numeric_limits<float>::lowest());
  staticShadowCasterLightSpaceBoundsMin_ =
      glm::vec3(std::numeric_limits<float>::max());
  staticShadowCasterLightSpaceBoundsMax_ =
      glm::vec3(std::numeric_limits<float>::lowest());
  staticShadowCasterCacheContentSignature_ = kFnvOffsetBasis64;
  staticShadowCasterCacheIndexCountEstimate_ = 0u;
  hasStaticShadowCasterBounds_ = false;
  hasStaticShadowCasterLightDepthBounds_ = false;
  hasStaticShadowCasterLightSpaceBounds_ = false;
  if (staticShadowTemplateIndices_.empty()) {
    staticShadowCasterCacheValid_ = true;
    return;
  }
  const std::span<const Renderable> renderables = scene.renderables();
  const bool enableCascadeCasterCulling =
      settings.shadow.debug.enableCascadeCasterCulling;
  staticShadowCasterCache_.reserve(staticShadowTemplateIndices_.size());
  staticShadowBatchTemplates_.reserve(staticShadowTemplateIndices_.size());
  staticShadowBatchIndexMap_.reserve(staticShadowTemplateIndices_.size());
  staticShadowCasterDrawBuffers_.reserve(staticShadowTemplateIndices_.size() *
                                         2u);
  staticShadowCasterFitPoints_.reserve(staticShadowTemplateIndices_.size() *
                                       8u);
  const auto resolveStaticBatchIndex =
      [&](const StaticShadowBatchTemplate &candidate) {
        const StaticShadowBatchKey &key = candidate;
        const auto it = staticShadowBatchIndexMap_.find(key);
        if (it != staticShadowBatchIndexMap_.end()) {
          return it->second;
        }
        const uint32_t batchIndex =
            static_cast<uint32_t>(staticShadowBatchTemplates_.size());
        staticShadowBatchTemplates_.push_back(candidate);
        staticShadowBatchIndexMap_.emplace(key, batchIndex);
        return batchIndex;
      };
  for (const uint32_t templateIndex : staticShadowTemplateIndices_) {
    const MeshDrawTemplate &entry = meshDrawTemplates_[templateIndex];
    const std::optional<SubmeshLod> lod =
        resolveForwardLod(*entry.submesh, settings.opaque.forcedMeshLod, true);
    if (!lod.has_value()) {
      continue;
    }
    StaticShadowCasterCacheEntry cachedEntry{
        .templateIndex = templateIndex,
        .instanceIndex = entry.instanceIndex,
        .indexBuffer = entry.indexBuffer,
        .indexBufferOffset = entry.indexBufferOffset,
        .indexFormat = entry.indexFormat,
        .vertexBuffer = entry.baseVertexBuffer,
        .vertexDecodeBuffer = entry.vertexDecodeBuffer,
        .vertexBufferAddress = entry.vertexBufferAddress,
        .vertexDecodeBufferAddress = entry.vertexDecodeBufferAddress,
        .vertexDecodeIndex = entry.vertexDecodeIndex,
        .packedVertexFormat = entry.packedVertexFormat,
        .materialIndex = entry.materialIndex,
        .indexCount = lod->indexCount,
        .firstIndex = lod->indexOffset,
        .doubleSided = entry.doubleSided,
        .alphaMasked = entry.alphaMasked,
        .hasCasterCullingBounds = false,
        .casterWorldCorners = {},
    };
    cachedEntry.rasterSignature =
        hashStaticShadowCasterRasterSignature(kFnvOffsetBasis64, cachedEntry);
    const glm::mat4 &modelMatrix = renderables[entry.instanceIndex].modelMatrix;
    cachedEntry.rasterSignature = hashCombine64(
        cachedEntry.rasterSignature,
        hashBytes(std::as_bytes(std::span<const glm::mat4>(&modelMatrix, 1u))));
    const StaticShadowBatchTemplate batchTemplate{
        StaticShadowBatchKey{
            .pipeline = shadowPipelines_[shadowPipelineIndex(
                entry.doubleSided, entry.alphaMasked)],
            .vertexBuffer = cachedEntry.vertexBuffer,
            .vertexDecodeBuffer = cachedEntry.vertexDecodeBuffer,
            .indexBuffer = cachedEntry.indexBuffer,
            .indexBufferOffset = cachedEntry.indexBufferOffset,
            .indexFormat = cachedEntry.indexFormat,
            .indexCount = cachedEntry.indexCount,
            .firstIndex = cachedEntry.firstIndex,
            .vertexBufferAddress = cachedEntry.vertexBufferAddress,
            .vertexDecodeBufferAddress = cachedEntry.vertexDecodeBufferAddress,
            .vertexDecodeIndex = cachedEntry.vertexDecodeIndex,
            .packedVertexFormat = cachedEntry.packedVertexFormat,
            .materialIndex = entry.alphaMasked ? cachedEntry.materialIndex : 0u,
        },
        0u, 0u, kFnvOffsetBasis64, 0u};
    cachedEntry.batchIndex = resolveStaticBatchIndex(batchTemplate);
    StaticShadowBatchTemplate &resolvedBatch =
        staticShadowBatchTemplates_[cachedEntry.batchIndex];
    ++resolvedBatch.instanceCount;
    resolvedBatch.rasterSignature = hashCombine64(resolvedBatch.rasterSignature,
                                                  cachedEntry.rasterSignature);
    resolvedBatch.indexCountEstimate += cachedEntry.indexCount;
    const BoundingBox worldBounds =
        entry.submesh->bounds.getTransformed(modelMatrix);
    cachedEntry.casterWorldCorners =
        shadow_detail::computeBoundsCorners(worldBounds.min_, worldBounds.max_);
    cachedEntry.hasCasterCullingBounds = enableCascadeCasterCulling;
    staticShadowCasterFitPoints_.insert(staticShadowCasterFitPoints_.end(),
                                        cachedEntry.casterWorldCorners.begin(),
                                        cachedEntry.casterWorldCorners.end());
    staticShadowCasterBoundsMin_ =
        glm::min(staticShadowCasterBoundsMin_, worldBounds.min_);
    staticShadowCasterBoundsMax_ =
        glm::max(staticShadowCasterBoundsMax_, worldBounds.max_);
    hasStaticShadowCasterBounds_ = true;
    staticShadowCasterCache_.push_back(cachedEntry);
    staticShadowCasterCacheContentSignature_ = hashCombine64(
        staticShadowCasterCacheContentSignature_, cachedEntry.rasterSignature);
    staticShadowCasterCacheIndexCountEstimate_ += cachedEntry.indexCount;
    appendUniqueBufferHandle(staticShadowCasterDrawBuffers_,
                             cachedEntry.vertexBuffer);
    appendUniqueBufferHandle(staticShadowCasterDrawBuffers_,
                             cachedEntry.indexBuffer);
  }
  staticShadowBatchInstanceIndices_.resize(staticShadowCasterCache_.size());
  uint32_t nextBatchInstanceIndex = 0u;
  for (StaticShadowBatchTemplate &batch : staticShadowBatchTemplates_) {
    batch.firstInstanceIndex = nextBatchInstanceIndex;
    nextBatchInstanceIndex += batch.instanceCount;
  }
  std::pmr::vector<uint32_t> staticBatchWriteOffsets(memory_);
  staticBatchWriteOffsets.reserve(staticShadowBatchTemplates_.size());
  for (const StaticShadowBatchTemplate &batch : staticShadowBatchTemplates_) {
    staticBatchWriteOffsets.push_back(batch.firstInstanceIndex);
  }
  for (const StaticShadowCasterCacheEntry &entry : staticShadowCasterCache_) {
    const uint32_t writeIndex = staticBatchWriteOffsets[entry.batchIndex]++;
    staticShadowBatchInstanceIndices_[writeIndex] = entry.instanceIndex;
  }
  staticShadowCasterLightGridQueryMarks_.assign(staticShadowCasterCache_.size(),
                                                0u);
  staticShadowCasterCacheValid_ = true;
}

Result<bool, std::string> ShadowRenderer::updateShadowFrameData(
    RenderFrameContext &frame, const RenderSettings::ShadowSettings &settings,
    uint32_t shadowMapSize, int32_t forcedMeshLod) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_COPY);
  shadowFrameCpuData_ = {};
  shadowDebugFrameData_ = {};
  currentRawShadowFits_ = {};
  hasActiveShadowLightForFrame_ = false;
  const uint32_t cascadeCount =
      std::clamp(activeCascadeCount_, 1u, kMaxShadowCascades);
  const ShadowSplitRange fixedSplitRange =
      computeFixedShadowSplitRange(frame.camera, settings.maxDistance);
  const std::array<float, kMaxShadowCascades + 1u> fixedSplitDepths =
      shadow_detail::computeCascadeSplitDepthsForRange(
          fixedSplitRange.nearDepth, fixedSplitRange.farDepth, cascadeCount,
          settings.splitLambda);
  const float minimumFarDepth = cascadeCount > 1u
                                    ? fixedSplitDepths[cascadeCount - 1u]
                                    : fixedSplitRange.farDepth;
  std::array<float, kMaxShadowCascades + 1u> minMaxSplitDepths =
      fixedSplitDepths;
  ShadowSplitRange effectiveSplitRange = fixedSplitRange;
  std::array<float, kMaxShadowCascades + 1u> effectiveSplitDepths =
      fixedSplitDepths;
  shadowDebugFrameData_.rawSamplerId = frame.sharedResources.shadowRawSamplerId;
  shadowDebugFrameData_.compareSamplerId =
      frame.sharedResources.shadowCompareSamplerId;
  shadowDebugFrameData_.sdsm.status = ShadowSdsmStatus::FallbackFixed;
  shadowDebugFrameData_.sdsm.activeReductionBackend =
      ShadowSdsmReductionBackend::Cpu;
  shadowDebugFrameData_.sdsm.sourceFrameIndex =
      frame.sharedResources.sceneDepthPyramidSourceFrameIndex.value_or(
          std::numeric_limits<uint64_t>::max());
  shadowDebugFrameData_.sdsm.gpuResultSelectedSlot =
      std::numeric_limits<uint32_t>::max();
  shadowDebugFrameData_.sdsm.gpuResultSourceFrameIndex =
      std::numeric_limits<uint64_t>::max();
  shadowDebugFrameData_.sdsm.splitCount = cascadeCount;
  shadowDebugFrameData_.sdsm.fixedRangeNear = fixedSplitRange.nearDepth;
  shadowDebugFrameData_.sdsm.fixedRangeFar = fixedSplitRange.farDepth;
  shadowDebugFrameData_.sdsm.smoothedLinearMin = fixedSplitRange.nearDepth;
  shadowDebugFrameData_.sdsm.smoothedLinearMax = fixedSplitRange.farDepth;
  shadowDebugFrameData_.sdsm.effectiveRangeNear = fixedSplitRange.nearDepth;
  shadowDebugFrameData_.sdsm.effectiveRangeFar = fixedSplitRange.farDepth;
  shadowDebugFrameData_.sdsm.fixedSplitDepths = fixedSplitDepths;
  shadowDebugFrameData_.sdsm.minMaxSplitDepths = fixedSplitDepths;
  shadowDebugFrameData_.sdsm.effectiveSplitDepths = fixedSplitDepths;
  frame.metrics.shadow.sdsmReadbackBytes = 0u;
  bool gpuReductionWarmingUp = false;
  const auto publishGpuReductionResultDebug = [&](uint32_t selectedSlot,
                                                  uint64_t sourceFrameIndex) {
    shadowDebugFrameData_.sdsm.gpuReductionResultAvailable = true;
    shadowDebugFrameData_.sdsm.gpuResultSelectedSlot = selectedSlot;
    shadowDebugFrameData_.sdsm.gpuResultSourceFrameIndex = sourceFrameIndex;
  };
  for (uint32_t cascadeIndex = 0u; cascadeIndex < activeCascadeCount_;
       ++cascadeIndex) {
    shadowDebugFrameData_.cascades[cascadeIndex].texture =
        shadowDepthTextures_[cascadeIndex];
    shadowDebugFrameData_.cascades[cascadeIndex].textureBindlessId =
        gpu_.getTextureBindlessIndex(shadowDepthTextures_[cascadeIndex]);
  }
  const auto publishInactiveShadowFrame = [&]() -> Result<bool, std::string> {
    invalidateReusableStaticOnlyCascadeCache();
    frame.sharedResources.selectedShadowLightId.reset();
    frame.sharedResources.shadowDebugFrameData = shadowDebugFrameData_;
    return Result<bool, std::string>::makeResult(true);
  };
  const auto invalidateSdsmRange = [&]() {
    sdsmState_.hasValidSdsmRange_ = false;
    sdsmState_.hasValidSdsmFarCascadeTexelSize_ = false;
    sdsmState_.lastValidSdsmSourceFrameIndex_ =
        std::numeric_limits<uint64_t>::max();
  };
  const auto canReuseCachedSdsmRange = [&]() {
    if (!sdsmState_.hasValidSdsmRange_ ||
        sdsmState_.lastValidSdsmSourceFrameIndex_ ==
            std::numeric_limits<uint64_t>::max() ||
        frame.frameIndex <= sdsmState_.lastValidSdsmSourceFrameIndex_) {
      return false;
    }
    return (frame.frameIndex - sdsmState_.lastValidSdsmSourceFrameIndex_) <=
           kSdsmMaxCachedSourceFrameLag;
  };
  const auto applySdsmSplitDistribution = [&]() {
    const float visibleFar = std::clamp(
        std::max(sdsmState_.sdsmSmoothedMaxDepth_, minimumFarDepth),
        fixedSplitRange.nearDepth + 1.0e-4f, fixedSplitRange.farDepth);
    minMaxSplitDepths = shadow_detail::computeCascadeSplitDepthsForRange(
        fixedSplitRange.nearDepth, visibleFar, cascadeCount,
        settings.splitLambda);
    minMaxSplitDepths[cascadeCount] = fixedSplitRange.farDepth;
    shadow_detail::enforceMonotonicShadowSplitDepths(
        minMaxSplitDepths, cascadeCount, fixedSplitRange.nearDepth,
        fixedSplitRange.farDepth);
    effectiveSplitDepths = minMaxSplitDepths;
  };
  const bool gpuReductionResourcesAvailable =
      nuri::isValid(sdsmReducePipelineHandle_) &&
      !bufferRings_[SdsmReduceResultRing].empty() &&
      std::all_of(bufferRings_[SdsmReduceResultRing].begin(),
                  bufferRings_[SdsmReduceResultRing].end(),
                  [](const DynamicBufferSlot &slot) {
                    return slot.buffer && slot.buffer->valid();
                  });
  const ShadowSdsmReductionBackend activeReductionBackend =
      gpuReductionResourcesAvailable ? ShadowSdsmReductionBackend::Gpu
                                     : ShadowSdsmReductionBackend::Cpu;
  const auto activateReductionFallback = [&] {
    shadowDebugFrameData_.sdsm.reductionFallbackActive = true;
  };
  if (activeReductionBackend != ShadowSdsmReductionBackend::Gpu) {
    sdsmState_.gpuReductionConsecutiveMissingResultFrames_ = 0u;
  }
  shadowDebugFrameData_.sdsm.activeReductionBackend = activeReductionBackend;
  shadowDebugFrameData_.sdsm.gpuResultRingSlotCount =
      activeReductionBackend == ShadowSdsmReductionBackend::Gpu
          ? saturateToU32(bufferRings_[SdsmReduceResultRing].size())
          : 0u;
  frame.metrics.shadow.sdsmActiveReductionBackend = activeReductionBackend;
  frame.metrics.shadow.sdsmReadbackBytes =
      activeReductionBackend == ShadowSdsmReductionBackend::Gpu
          ? static_cast<uint32_t>(sizeof(SdsmGpuMinMaxResult))
          : static_cast<uint32_t>(sizeof(float) * 2u);
  const auto reuseCachedSdsmRangeOrFallback = [&](ShadowSdsmStatus status) {
    shadowDebugFrameData_.sdsm.status = status;
    if (canReuseCachedSdsmRange()) {
      shadowDebugFrameData_.sdsm.fixedFallbackActive = false;
      shadowDebugFrameData_.sdsm.smoothedLinearMin =
          sdsmState_.sdsmSmoothedMinDepth_;
      shadowDebugFrameData_.sdsm.smoothedLinearMax =
          sdsmState_.sdsmSmoothedMaxDepth_;
      applySdsmSplitDistribution();
      return;
    }
    invalidateSdsmRange();
    shadowDebugFrameData_.sdsm.fixedFallbackActive = true;
  };
  if (frame.scene == nullptr) {
    resetFrozenShadowFit();
    resetCascadeStabilizationHistory();
    resetSdsmState();
    return publishInactiveShadowFrame();
  }
  const std::span<const DirectionalLightGpuData> directionalLights =
      frame.scene->packedDirectionalLights();
  const std::span<const LightId> directionalLightIds =
      frame.scene->packedDirectionalLightIds();
  if (directionalLights.empty() || directionalLightIds.empty()) {
    resetFrozenShadowFit();
    resetCascadeStabilizationHistory();
    resetSdsmState();
    return publishInactiveShadowFrame();
  }
  uint32_t selectedLightIndex = 0u;
  LightId selectedLightId = directionalLightIds.front();
  if (frame.sharedResources.selectedShadowLightId.has_value() &&
      isValid(*frame.sharedResources.selectedShadowLightId)) {
    for (uint32_t i = 0u; i < directionalLightIds.size(); ++i) {
      if (directionalLightIds[i] ==
          *frame.sharedResources.selectedShadowLightId) {
        selectedLightIndex = i;
        selectedLightId = directionalLightIds[i];
        break;
      }
    }
  }
  frame.sharedResources.selectedShadowLightId = selectedLightId;
  const bool freezeShadowFits =
      settings.debug.freezeLightView || settings.debug.freezeCascades;
  if (!freezeShadowFits) {
    resetFrozenShadowFit();
  }
  if (cascadeStabilizationHistory_.valid &&
      (cascadeStabilizationHistory_.lightId != selectedLightId ||
       cascadeStabilizationHistory_.shadowMapSize != shadowMapSize ||
       cascadeStabilizationHistory_.cascadeCount != cascadeCount)) {
    resetCascadeStabilizationHistory();
  }
  const bool canReuseCascadeStabilizationHistory =
      cascadeStabilizationHistory_.valid;
  hasActiveShadowLightForFrame_ = true;
  const DirectionalLightGpuData &light = directionalLights[selectedLightIndex];
  const glm::vec3 lightDirection = shadow_detail::normalizeSafe(
      glm::vec3(light.directionIlluminance), glm::vec3(0.0f, -1.0f, 0.0f));
  const AnimationSceneFrameData *animationSceneData =
      resolveAnimationSceneFrameData(frame);
  const bool staticCasterCacheMatchesFrame =
      frame.scene != nullptr &&
      staticShadowCasterCacheTransformVersion_ ==
          frame.scene->transformVersion() &&
      staticShadowCasterCacheForcedMeshLod_ == forcedMeshLod;
  const bool canUseStaticCasterFitCache = animationSceneData == nullptr &&
                                          staticShadowCasterCacheValid_ &&
                                          staticCasterCacheMatchesFrame;
  const bool needCasterLightDepthBounds = cascadeCount > 1u;
  const glm::mat4 casterLightView =
      needCasterLightDepthBounds
          ? shadow_detail::makeDirectionalLightView(lightDirection)
          : glm::mat4(1.0f);
  glm::vec2 casterLightDepthBounds(std::numeric_limits<float>::max(),
                                   std::numeric_limits<float>::lowest());
  bool hasCasterLightDepthBounds = false;
  const auto combineCasterLightDepthBounds = [&](glm::vec2 bounds) {
    if (bounds.x > bounds.y) {
      return;
    }
    const float casterMin = std::min(bounds.x, bounds.y);
    const float casterMax = std::max(bounds.x, bounds.y);
    if (!std::isfinite(casterMin) || !std::isfinite(casterMax)) {
      return;
    }
    casterLightDepthBounds.x = std::min(casterLightDepthBounds.x, casterMin);
    casterLightDepthBounds.y = std::max(casterLightDepthBounds.y, casterMax);
    hasCasterLightDepthBounds = true;
  };
  const auto computeCasterLightDepthBounds =
      [&](std::span<const glm::vec3> points) {
        glm::vec2 bounds(std::numeric_limits<float>::max(),
                         std::numeric_limits<float>::lowest());
        bool hasBounds = false;
        for (const glm::vec3 point : points) {
          const float z = glm::vec3(casterLightView * glm::vec4(point, 1.0f)).z;
          if (!std::isfinite(z)) {
            continue;
          }
          bounds.x = std::min(bounds.x, z);
          bounds.y = std::max(bounds.y, z);
          hasBounds = true;
        }
        return hasBounds ? bounds
                         : glm::vec2(std::numeric_limits<float>::max(),
                                     std::numeric_limits<float>::lowest());
      };
  const auto ensureStaticCasterLightDepthBounds = [&]() {
    if (!needCasterLightDepthBounds || staticShadowCasterFitPoints_.empty()) {
      hasStaticShadowCasterLightDepthBounds_ = false;
      return;
    }
    if (hasStaticShadowCasterLightDepthBounds_ &&
        rawBytesEqual(staticShadowCasterLightDepthDirection_, lightDirection)) {
      return;
    }
    const glm::vec2 bounds = computeCasterLightDepthBounds(
        std::span<const glm::vec3>(staticShadowCasterFitPoints_.data(),
                                   staticShadowCasterFitPoints_.size()));
    staticShadowCasterLightDepthDirection_ = lightDirection;
    staticShadowCasterLightDepthBounds_ = bounds;
    hasStaticShadowCasterLightDepthBounds_ = bounds.x <= bounds.y;
  };
  BoundingBox casterBounds{};
  bool hasCasterBounds = false;
  bool hasAnimatedGeometryOverrides = false;
  const auto appendCasterFitBounds = [&](const MeshDrawTemplate &entry) {
    bool usesAnimatedOverride = false;
    if (animationSceneData != nullptr) {
      const AnimatedRenderableGeometryOverride &geometryOverride =
          animationSceneData
              ->geometryOverridesByRenderable[entry.instanceIndex];
      usesAnimatedOverride =
          geometryOverride.enabled &&
          nuri::isValid(geometryOverride.vertexBuffer) &&
          animationOverrideCoversSubmesh(geometryOverride, *entry.submesh);
    }
    if (usesAnimatedOverride) {
      hasAnimatedGeometryOverrides = true;
      return;
    }
    const BoundingBox worldBounds =
        entry.submesh->bounds.getTransformed(entry.renderable->modelMatrix);
    if (needCasterLightDepthBounds) {
      const auto worldBoundsCorners = shadow_detail::computeBoundsCorners(
          worldBounds.min_, worldBounds.max_);
      combineCasterLightDepthBounds(computeCasterLightDepthBounds(
          std::span<const glm::vec3, 8>(worldBoundsCorners)));
    }
    casterBounds.combinePoint(worldBounds.min_);
    casterBounds.combinePoint(worldBounds.max_);
    hasCasterBounds = true;
  };
  {
    NURI_PROFILER_ZONE("ShadowRenderer.update.static_caster_bounds",
                       NURI_PROFILER_COLOR_CMD_COPY);
    if (canUseStaticCasterFitCache && hasStaticShadowCasterBounds_) {
      casterBounds.combinePoint(staticShadowCasterBoundsMin_);
      casterBounds.combinePoint(staticShadowCasterBoundsMax_);
      hasCasterBounds = true;
      if (needCasterLightDepthBounds) {
        ensureStaticCasterLightDepthBounds();
        if (hasStaticShadowCasterLightDepthBounds_) {
          combineCasterLightDepthBounds(staticShadowCasterLightDepthBounds_);
        }
      }
    }
    NURI_PROFILER_ZONE_END();
  }
  {
    NURI_PROFILER_ZONE("ShadowRenderer.update.dynamic_caster_bounds",
                       NURI_PROFILER_COLOR_CMD_COPY);
    if (canUseStaticCasterFitCache) {
      for (const uint32_t templateIndex : dynamicShadowTemplateIndices_) {
        appendCasterFitBounds(meshDrawTemplates_[templateIndex]);
      }
    } else {
      for (const MeshDrawTemplate &entry : meshDrawTemplates_) {
        appendCasterFitBounds(entry);
      }
    }
    NURI_PROFILER_ZONE_END();
  }
  shadowDebugFrameData_.cascadeCount = cascadeCount;
  TextureHandle latestSdsmPyramidTexture{};
  bool hasValidSdsmSource =
      frame.sharedResources.sceneDepthPyramidSourceFrameIndex.has_value() &&
      frame.frameIndex > 0u &&
      *frame.sharedResources.sceneDepthPyramidSourceFrameIndex ==
          (frame.frameIndex - 1u) &&
      frame.sharedResources.sceneDepthPyramidLevelCount > 0u;
  if (frame.sharedResources.sceneDepthPyramidLevelCount > 0u) {
    latestSdsmPyramidTexture =
        frame.sharedResources.sceneDepthPyramidTextures
            [frame.sharedResources.sceneDepthPyramidLevelCount - 1u];
    hasValidSdsmSource =
        hasValidSdsmSource && nuri::isValid(latestSdsmPyramidTexture);
  }
  if (!hasValidSdsmSource) {
    sdsmState_.gpuReductionConsecutiveMissingResultFrames_ = 0u;
    const ShadowSdsmStatus sourceStatus =
        frame.sharedResources.sceneDepthPyramidSourceFrameIndex.has_value() &&
                frame.frameIndex > 0u
            ? ShadowSdsmStatus::Stale
            : ShadowSdsmStatus::Unavailable;
    reuseCachedSdsmRangeOrFallback(sourceStatus);
  } else {
    const auto activateStableSdsmCoverage = [&]() {
      shadowDebugFrameData_.sdsm.status = ShadowSdsmStatus::Active;
      shadowDebugFrameData_.sdsm.fixedFallbackActive = false;
      shadowDebugFrameData_.sdsm.smoothedLinearMin =
          sdsmState_.sdsmSmoothedMinDepth_;
      shadowDebugFrameData_.sdsm.smoothedLinearMax =
          sdsmState_.sdsmSmoothedMaxDepth_;
      effectiveSplitRange = fixedSplitRange;
      applySdsmSplitDistribution();
    };
    const auto consumeRawDeviceMinMax = [&](float rawDeviceMin,
                                            float rawDeviceMax) {
      shadowDebugFrameData_.sdsm.rawDeviceMin = rawDeviceMin;
      shadowDebugFrameData_.sdsm.rawDeviceMax = rawDeviceMax;
      const bool rawDepthsValid =
          std::isfinite(rawDeviceMin) && std::isfinite(rawDeviceMax) &&
          rawDeviceMin >= -1.0e-4f && rawDeviceMax <= (1.0f + 1.0e-4f) &&
          rawDeviceMax >= rawDeviceMin;
      if (!rawDepthsValid) {
        reuseCachedSdsmRangeOrFallback(ShadowSdsmStatus::Invalid);
        return;
      }
      const float rawLinearMin = shadow_detail::linearizeDeviceDepthToViewDepth(
          rawDeviceMin, frame.camera);
      const float rawLinearMax = shadow_detail::linearizeDeviceDepthToViewDepth(
          rawDeviceMax, frame.camera);
      shadowDebugFrameData_.sdsm.rawLinearMin = rawLinearMin;
      shadowDebugFrameData_.sdsm.rawLinearMax = rawLinearMax;
      const bool rawLinearValid = std::isfinite(rawLinearMin) &&
                                  std::isfinite(rawLinearMax) &&
                                  rawLinearMax >= rawLinearMin;
      if (!rawLinearValid) {
        reuseCachedSdsmRangeOrFallback(ShadowSdsmStatus::Invalid);
        return;
      }
      if (!sdsmState_.hasValidSdsmRange_) {
        sdsmState_.sdsmSmoothedMinDepth_ = rawLinearMin;
        sdsmState_.sdsmSmoothedMaxDepth_ = rawLinearMax;
      } else {
        const float historyWeight =
            std::clamp(settings.sdsmTemporalBlend, 0.0f, 1.0f);
        const float sampleWeight = 1.0f - historyWeight;
        sdsmState_.sdsmSmoothedMinDepth_ =
            rawLinearMin * sampleWeight +
            sdsmState_.sdsmSmoothedMinDepth_ * historyWeight;
        sdsmState_.sdsmSmoothedMaxDepth_ =
            rawLinearMax * sampleWeight +
            sdsmState_.sdsmSmoothedMaxDepth_ * historyWeight;
      }
      sdsmState_.hasValidSdsmRange_ = true;
      sdsmState_.lastValidSdsmSourceFrameIndex_ =
          *frame.sharedResources.sceneDepthPyramidSourceFrameIndex;
      activateStableSdsmCoverage();
    };
    const auto consumeCpuMinMaxSource = [&]() -> bool {
      const TextureHandle sdsmTexture =
          frame.sharedResources.sceneDepthPyramidTextures
              [frame.sharedResources.sceneDepthPyramidLevelCount - 1u];
      if (!nuri::isValid(sdsmTexture)) {
        reuseCachedSdsmRangeOrFallback(ShadowSdsmStatus::Unavailable);
        return false;
      }
      std::array<std::byte, sizeof(float) * 2u> sdsmBytes{};
      const TextureReadbackRegion readbackRegion{
          .x = 0u,
          .y = 0u,
          .width = 1u,
          .height = 1u,
          .mipLevel = 0u,
          .layer = 0u,
      };
      auto readResult =
          gpu_.readTexture(sdsmTexture, readbackRegion, sdsmBytes);
      if (readResult.hasError()) {
        reuseCachedSdsmRangeOrFallback(ShadowSdsmStatus::Invalid);
        return false;
      }
      std::array<float, 2u> rawDeviceDepths{};
      std::memcpy(rawDeviceDepths.data(), sdsmBytes.data(),
                  sizeof(rawDeviceDepths));
      consumeRawDeviceMinMax(rawDeviceDepths[0], rawDeviceDepths[1]);
      return shadowDebugFrameData_.sdsm.status == ShadowSdsmStatus::Active;
    };
    if (activeReductionBackend == ShadowSdsmReductionBackend::Gpu) {
      const auto reductionStart = std::chrono::steady_clock::now();
      bool anyReadbackError = false;
      std::optional<SdsmGpuMinMaxResult> selectedGpuResult;
      uint32_t selectedGpuResultSlot = std::numeric_limits<uint32_t>::max();
      for (size_t slotIndex = 0u;
           slotIndex < bufferRings_[SdsmReduceResultRing].size(); ++slotIndex) {
        const DynamicBufferSlot &slot =
            bufferRings_[SdsmReduceResultRing][slotIndex];
        const uint64_t expectedPublishedFrame =
            frameSlotStates_[slotIndex].sdsmPublishedFrame;
        if (expectedPublishedFrame == std::numeric_limits<uint64_t>::max()) {
          continue;
        }
        SdsmGpuMinMaxResult gpuResult{};
        auto readResult = gpu_.readBuffer(
            slot.buffer->handle(), 0u,
            std::as_writable_bytes(
                std::span<SdsmGpuMinMaxResult>(&gpuResult, 1u)));
        if (readResult.hasError()) {
          anyReadbackError = true;
          continue;
        }
        if (gpuResult.valid == 0u ||
            static_cast<uint64_t>(gpuResult.sourceFrameIndex) >=
                frame.frameIndex) {
          continue;
        }
        if (static_cast<uint64_t>(gpuResult.sourceFrameIndex) !=
            expectedPublishedFrame) {
          continue;
        }
        if (!selectedGpuResult.has_value() ||
            gpuResult.sourceFrameIndex > selectedGpuResult->sourceFrameIndex) {
          selectedGpuResult = gpuResult;
          selectedGpuResultSlot = saturateToU32(slotIndex);
        }
      }
      if (!selectedGpuResult.has_value()) {
        if (!anyReadbackError) {
          ++sdsmState_.gpuReductionConsecutiveMissingResultFrames_;
          gpuReductionWarmingUp =
              sdsmState_.gpuReductionConsecutiveMissingResultFrames_ <=
              kSdsmGpuWarmupGraceMissFrames;
        } else {
          sdsmState_.gpuReductionConsecutiveMissingResultFrames_ = 0u;
        }
        if (!gpuReductionWarmingUp)
          activateReductionFallback();
        reuseCachedSdsmRangeOrFallback(ShadowSdsmStatus::Stale);
      } else {
        sdsmState_.gpuReductionConsecutiveMissingResultFrames_ = 0u;
        shadowDebugFrameData_.sdsm.sourceFrameIndex =
            selectedGpuResult->sourceFrameIndex;
        publishGpuReductionResultDebug(selectedGpuResultSlot,
                                       selectedGpuResult->sourceFrameIndex);
        consumeRawDeviceMinMax(selectedGpuResult->rawDeviceMinMax.x,
                               selectedGpuResult->rawDeviceMinMax.y);
      }
      frame.metrics.shadow.sdsmReductionSourceSamples = 1u;
      frame.metrics.shadow.sdsmCpuReductionTimeMs =
          elapsedMilliseconds(reductionStart, std::chrono::steady_clock::now());
    } else {
      const auto reductionStart = std::chrono::steady_clock::now();
      sdsmState_.gpuReductionConsecutiveMissingResultFrames_ = 0u;
      (void)consumeCpuMinMaxSource();
      frame.metrics.shadow.sdsmReductionSourceSamples = 1u;
      frame.metrics.shadow.sdsmCpuReductionTimeMs =
          elapsedMilliseconds(reductionStart, std::chrono::steady_clock::now());
    }
  }
  shadowDebugFrameData_.sdsm.effectiveRangeNear = effectiveSplitRange.nearDepth;
  shadowDebugFrameData_.sdsm.effectiveRangeFar = effectiveSplitRange.farDepth;
  shadowDebugFrameData_.sdsm.minMaxSplitDepths = minMaxSplitDepths;
  shadowDebugFrameData_.sdsm.effectiveSplitDepths = effectiveSplitDepths;
  const bool reuseFrozenFit = freezeShadowFits && hasFrozenShadowFit_ &&
                              frozenShadowLightId_ == selectedLightId &&
                              frozenShadowMapSize_ == shadowMapSize &&
                              frozenCascadeCount_ == cascadeCount;
  {
    NURI_PROFILER_ZONE("ShadowRenderer.update.fit_cascades",
                       NURI_PROFILER_COLOR_CMD_COPY);
    if (reuseFrozenFit) {
      for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
           ++cascadeIndex) {
        const shadow_detail::DirectionalShadowFit &fit =
            frozenShadowFits_[cascadeIndex];
        currentRawShadowFits_[cascadeIndex] = fit;
        writeShadowCascadeFit(fit, cascadeIndex,
                              shadowDepthTextures_[cascadeIndex], gpu_,
                              shadowFrameCpuData_, shadowDebugFrameData_);
        cascadeStabilizationHistory_.fits[cascadeIndex] = fit;
      }
    } else {
      for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
           ++cascadeIndex) {
        shadow_detail::DirectionalShadowFit fit{};
        if (cascadeCount == 1u) {
          fit = fitSingleShadowMapForRange(
              frame.camera, effectiveSplitRange, lightDirection, shadowMapSize,
              true, hasCasterBounds, hasAnimatedGeometryOverrides,
              casterBounds.min_, casterBounds.max_);
        } else {
          fit = shadow_detail::
              fitDirectionalShadowCascadeSliceWithCasterDepthBounds(
                  frame.camera, effectiveSplitDepths[cascadeIndex],
                  effectiveSplitDepths[cascadeIndex + 1u], casterLightView,
                  shadowMapSize,
                  !hasAnimatedGeometryOverrides && hasCasterLightDepthBounds,
                  casterLightDepthBounds, true);
        }
        if (canReuseCascadeStabilizationHistory) {
          shadow_detail::applyDirectionalShadowFitHysteresis(
              fit, cascadeStabilizationHistory_.fits[cascadeIndex],
              shadowMapSize);
        }
        frozenShadowFits_[cascadeIndex] = fit;
        currentRawShadowFits_[cascadeIndex] = fit;
        writeShadowCascadeFit(fit, cascadeIndex,
                              shadowDepthTextures_[cascadeIndex], gpu_,
                              shadowFrameCpuData_, shadowDebugFrameData_);
        cascadeStabilizationHistory_.fits[cascadeIndex] = fit;
      }
      if (freezeShadowFits) {
        hasFrozenShadowFit_ = true;
        frozenShadowLightId_ = selectedLightId;
        frozenShadowMapSize_ = shadowMapSize;
        frozenCascadeCount_ = cascadeCount;
      }
    }
    NURI_PROFILER_ZONE_END();
  }
  cascadeStabilizationHistory_.valid = true;
  cascadeStabilizationHistory_.lightId = selectedLightId;
  cascadeStabilizationHistory_.shadowMapSize = shadowMapSize;
  cascadeStabilizationHistory_.cascadeCount = cascadeCount;
  for (uint32_t cascadeIndex = cascadeCount; cascadeIndex < kMaxShadowCascades;
       ++cascadeIndex) {
    cascadeStabilizationHistory_.fits[cascadeIndex] = {};
  }
  if (shadowDebugFrameData_.sdsm.status == ShadowSdsmStatus::Active &&
      cascadeCount > 0u) {
    const float farCascadeTexelWorldSize =
        shadowDebugFrameData_.cascades[cascadeCount - 1u].texelWorldSize;
    if (std::isfinite(farCascadeTexelWorldSize) &&
        farCascadeTexelWorldSize > 0.0f) {
      sdsmState_.hasValidSdsmFarCascadeTexelSize_ = true;
      sdsmState_.sdsmFarCascadeTexelWorldSize_ = farCascadeTexelWorldSize;
    } else {
      sdsmState_.hasValidSdsmFarCascadeTexelSize_ = false;
      sdsmState_.sdsmFarCascadeTexelWorldSize_ = 0.0f;
    }
  }
  const uint32_t shadowFlags =
      kShadowFrameFlagEnabled | shadowDebugFrameFlags(settings.debug);
  shadowFrameCpuData_.flagsCascadeCountLightIndex = glm::uvec4(
      shadowFlags, cascadeCount, selectedLightIndex, settings.pcfSampleCount);
  shadowFrameCpuData_.sharedBiasParams = glm::vec4(
      settings.constantBias, settings.slopeBias, settings.normalBias, 0.0f);
  shadowFrameCpuData_.sharedSamplerMapSize = glm::uvec4(
      frame.sharedResources.shadowCompareSamplerId,
      frame.sharedResources.shadowRawSamplerId, settings.shadowMapSize, 0u);
  const float fadeEnd =
      shadowDebugFrameData_.cascades[cascadeCount - 1u].splitFar;
  const float fadeNear = shadowDebugFrameData_.cascades[0].splitNear;
  const float fadeWidth =
      std::max(fadeEnd - fadeNear, 0.0f) * settings.maxDistanceFadeFraction;
  const float fadeStart = std::max(fadeNear, fadeEnd - fadeWidth);
  shadowFrameCpuData_.fadeParams =
      glm::vec4(fadeStart, fadeEnd, settings.cascadeBlendFraction, 0.0f);
  shadowDebugFrameData_.maxDistanceFadeStart = fadeStart;
  shadowDebugFrameData_.maxDistanceFadeEnd = fadeEnd;
  shadowDebugFrameData_.selectedShadowLightId = selectedLightId;
  float minCascadeTexelWorldSize = std::numeric_limits<float>::max();
  float maxCascadeTexelWorldSize = 0.0f;
  float totalCascadeTexelWorldSize = 0.0f;
  uint32_t validCascadeTexelWorldSizeCount = 0u;
  for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
       ++cascadeIndex) {
    const float texelWorldSize =
        shadowDebugFrameData_.cascades[cascadeIndex].texelWorldSize;
    if (!std::isfinite(texelWorldSize) || texelWorldSize <= 0.0f) {
      continue;
    }
    minCascadeTexelWorldSize =
        std::min(minCascadeTexelWorldSize, texelWorldSize);
    maxCascadeTexelWorldSize =
        std::max(maxCascadeTexelWorldSize, texelWorldSize);
    totalCascadeTexelWorldSize += texelWorldSize;
    ++validCascadeTexelWorldSizeCount;
  }
  if (validCascadeTexelWorldSizeCount == 0u) {
    minCascadeTexelWorldSize = 0.0f;
  }
  frame.metrics.shadow.cascadeCount = cascadeCount;
  frame.metrics.shadow.shadowMapSize = shadowMapSize;
  frame.metrics.shadow.frameGpuBytes = sizeof(ShadowFrameGpuData);
  frame.metrics.shadow.cascadeTextureBytes =
      static_cast<uint64_t>(cascadeCount) * shadowMapSize * shadowMapSize *
      shadowDepthTextureBytesPerPixel(settings.depthFormat);
  frame.metrics.shadow.minCascadeTexelWorldSize = minCascadeTexelWorldSize;
  frame.metrics.shadow.maxCascadeTexelWorldSize = maxCascadeTexelWorldSize;
  frame.metrics.shadow.averageCascadeTexelWorldSize =
      validCascadeTexelWorldSizeCount > 0u
          ? totalCascadeTexelWorldSize /
                static_cast<float>(validCascadeTexelWorldSizeCount)
          : 0.0f;
  frame.metrics.shadow.farCascadeTexelWorldSize =
      cascadeCount > 0u
          ? shadowDebugFrameData_.cascades[cascadeCount - 1u].texelWorldSize
          : 0.0f;
  frame.sharedResources.shadowDebugFrameData = shadowDebugFrameData_;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
ShadowRenderer::buildShadowDraws(RenderFrameContext &frame, uint32_t frameSlot,
                                 const ForwardSceneGpuData &sceneGpu) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_DRAW);
  const RenderSettings &settings = renderSettingsOrDefault(frame);
  RenderSettings::ShadowSettings shadowSettings = settings.shadow;
  sanitizeShadowSettings(shadowSettings);
  const auto emitShadowDiagnostics = [&]() {
    if (!shadowSettings.debug.logDiagnostics) {
      diagnosticLogState_ = {};
      return;
    }
    const uint64_t signature = buildShadowDiagnosticSignature(
        shadowDebugFrameData_, frame.metrics.shadow,
        hasActiveShadowLightForFrame_, hasPreparedShadowDepthPasses_,
        hasPreparedShadowPreviewPass_);
    const bool changed = !diagnosticLogState_.hasLastSignature ||
                         diagnosticLogState_.lastSignature != signature;
    const bool intervalDue =
        diagnosticLogState_.lastLoggedFrameIndex ==
            std::numeric_limits<uint64_t>::max() ||
        frame.frameIndex >=
            diagnosticLogState_.lastLoggedFrameIndex +
                shadowSettings.debug.diagnosticLogIntervalFrames;
    if ((!changed && shadowSettings.debug.diagnosticLogOnlyOnChange) ||
        (!changed && !intervalDue)) {
      return;
    }
    const ShadowSdsmDebugFrameData &sdsm = shadowDebugFrameData_.sdsm;
    const uint64_t light =
        isValid(shadowDebugFrameData_.selectedShadowLightId)
            ? shadowDebugFrameData_.selectedShadowLightId.value
            : 0u;
    logMessagef(
        shadowSettings.debug.diagnosticLogLevel,
        "ShadowRenderer: frame=%llu prepared=(%u,%u) active=%u light=%llu "
        "map=%u cascades=%u draws=(%u,%u) cache=%u sdsm=(%s,%s)",
        static_cast<unsigned long long>(frame.frameIndex),
        hasPreparedShadowDepthPasses_ ? 1u : 0u,
        hasPreparedShadowPreviewPass_ ? 1u : 0u,
        hasActiveShadowLightForFrame_ ? 1u : 0u,
        static_cast<unsigned long long>(light), shadowSettings.shadowMapSize,
        shadowSettings.cascadeCount, frame.metrics.shadow.totalDraws,
        frame.metrics.shadow.totalCulledDraws,
        frame.metrics.shadow.staticCacheReused,
        shadowSdsmReductionBackendName(sdsm.activeReductionBackend).data(),
        shadowSdsmStatusName(sdsm.status).data());
    diagnosticLogState_.hasLastSignature = true;
    diagnosticLogState_.lastSignature = signature;
    diagnosticLogState_.lastLoggedFrameIndex = frame.frameIndex;
  };
  if (frame.scene == nullptr || meshDrawTemplates_.empty()) {
    emitShadowDiagnostics();
    return Result<bool, std::string>::makeResult(true);
  }
  if (activeCascadeCount_ == 0u ||
      shadowFrameCpuData_.flagsCascadeCountLightIndex.y == 0u) {
    emitShadowDiagnostics();
    return Result<bool, std::string>::makeResult(true);
  }
  const std::span<const Renderable> renderables = frame.scene->renderables();
  const AnimationSceneFrameData *animationSceneData =
      resolveAnimationSceneFrameData(frame);
  const uint64_t transformVersion = frame.scene->transformVersion();
  if (cachedTransformVersion_ != transformVersion ||
      instanceMatrices_.size() != renderables.size()) {
    instanceMatrices_.clear();
    instanceMatrices_.reserve(renderables.size());
    for (uint32_t i = 0u; i < static_cast<uint32_t>(renderables.size()); ++i) {
      instanceMatrices_.push_back(makeInstanceData(renderables[i].modelMatrix));
    }
    cachedTransformVersion_ = transformVersion;
    for (FrameSlotState &slot : frameSlotStates_) {
      slot.instanceUploadVersion = std::numeric_limits<uint64_t>::max();
    }
  }
  auto matricesResult = ensureInstanceMatricesRingCapacity(std::max(
      instanceMatrices_.size() * sizeof(InstanceData), sizeof(InstanceData)));
  if (matricesResult.hasError()) {
    return matricesResult;
  }
  const bool needsInstanceUpload =
      frameSlotStates_[frameSlot].instanceUploadVersion !=
      cachedTransformVersion_;
  if (needsInstanceUpload) {
    const std::span<const std::byte> matrixBytes{
        reinterpret_cast<const std::byte *>(instanceMatrices_.data()),
        instanceMatrices_.size() * sizeof(InstanceData)};
    auto updateMatricesResult = gpu_.updateBuffer(
        bufferRings_[InstanceMatricesRing][frameSlot].buffer->handle(),
        matrixBytes, 0u);
    if (updateMatricesResult.hasError()) {
      return updateMatricesResult;
    }
    frameSlotStates_[frameSlot].instanceUploadVersion = cachedTransformVersion_;
  }
  const BufferHandle instanceMatricesBuffer =
      animationSceneData != nullptr
          ? animationSceneData->instanceMatricesBuffer
          : bufferRings_[InstanceMatricesRing][frameSlot].buffer->handle();
  const uint64_t instanceMatricesAddress =
      animationSceneData != nullptr
          ? animationSceneData->instanceMatricesAddress
          : gpu_.getBufferDeviceAddress(instanceMatricesBuffer);
  if (sceneGpu.shadowFrameBufferAddress == 0u) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (sceneGpu.frameDataAddress == 0u || instanceMatricesAddress == 0u) {
    return Result<bool, std::string>::makeError(
        "ShadowRenderer: invalid scene buffer address");
  }
  const uint32_t cascadeCount =
      std::clamp(shadowFrameCpuData_.flagsCascadeCountLightIndex.y, 1u,
                 activeCascadeCount_);
  for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
       ++cascadeIndex) {
    cascadePushConstants_[cascadeIndex].clear();
    cascadeDrawItems_[cascadeIndex].clear();
    cascadeIndirectPushConstants_[cascadeIndex].clear();
    cascadeIndirectDrawItems_[cascadeIndex].clear();
    cascadePushConstants_[cascadeIndex].reserve(meshDrawTemplates_.size());
    cascadeDrawItems_[cascadeIndex].reserve(meshDrawTemplates_.size());
  }
  const bool enableCascadeCasterCulling =
      settings.shadow.debug.enableCascadeCasterCulling;
  const uint64_t cachePipelineSignature =
      shadowPipelineSignature() ^
      (enableCascadeCasterCulling ? 0x9e3779b97f4a7c15ull : 0ull);
  const bool staticCacheMissing = !staticShadowCasterCacheValid_;
  const bool staticCacheTransformChanged =
      staticShadowCasterCacheTransformVersion_ != transformVersion;
  const bool staticCacheForcedMeshLodChanged =
      staticShadowCasterCacheForcedMeshLod_ != settings.opaque.forcedMeshLod;
  const bool staticCachePipelineChanged =
      staticShadowCasterCachePipelineSignature_ != cachePipelineSignature;
  const bool needsStaticCacheRebuild =
      staticCacheMissing || staticCacheTransformChanged ||
      staticCacheForcedMeshLodChanged || staticCachePipelineChanged;
  if (needsStaticCacheRebuild) {
    rebuildStaticShadowCasterCache(*frame.scene, settings);
    staticShadowCasterCacheTransformVersion_ = transformVersion;
    staticShadowCasterCacheForcedMeshLod_ = settings.opaque.forcedMeshLod;
    staticShadowCasterCachePipelineSignature_ = cachePipelineSignature;
  }
  const bool staticCacheReused = !needsStaticCacheRebuild;
  const bool staticCacheRebuiltForTransformsOnly =
      needsStaticCacheRebuild && !staticCacheMissing &&
      staticCacheTransformChanged && !staticCacheForcedMeshLodChanged &&
      !staticCachePipelineChanged;
  std::array<ShadowRenderer::StaticOnlyCascadeReuseState, kMaxShadowCascades>
      currentStaticOnlyCascadeStates{};
  std::pmr::vector<uint32_t> frameDynamicTemplateIndices(memory_);
  frameDynamicTemplateIndices.reserve(dynamicShadowTemplateIndices_.size() +
                                      (animationSceneData != nullptr
                                           ? staticShadowTemplateIndices_.size()
                                           : 0u));
  preResolvedDrawBuffers_.reserve(staticShadowCasterDrawBuffers_.size() +
                                  frameDynamicTemplateIndices.capacity() * 2u);
  preResolvedDrawBuffers_.insert(preResolvedDrawBuffers_.end(),
                                 staticShadowCasterDrawBuffers_.begin(),
                                 staticShadowCasterDrawBuffers_.end());
  frameDynamicTemplateIndices.insert(frameDynamicTemplateIndices.end(),
                                     dynamicShadowTemplateIndices_.begin(),
                                     dynamicShadowTemplateIndices_.end());
  std::pmr::vector<uint8_t> staticTemplatesUsingDynamicPath(
      meshDrawTemplates_.size(), uint8_t{0}, memory_);
  uint32_t overriddenStaticTemplateCount = 0u;
  if (animationSceneData != nullptr) {
    for (const uint32_t templateIndex : staticShadowTemplateIndices_) {
      const MeshDrawTemplate &entry = meshDrawTemplates_[templateIndex];
      const AnimatedRenderableGeometryOverride &geometryOverride =
          animationSceneData
              ->geometryOverridesByRenderable[entry.instanceIndex];
      if (!geometryOverride.enabled ||
          !nuri::isValid(geometryOverride.vertexBuffer) ||
          !animationOverrideCoversSubmesh(geometryOverride, *entry.submesh)) {
        continue;
      }
      staticTemplatesUsingDynamicPath[templateIndex] = 1u;
      ++overriddenStaticTemplateCount;
      frameDynamicTemplateIndices.push_back(templateIndex);
    }
  }
  const size_t maxShadowInstanceRemapCount =
      std::max<size_t>(1u, static_cast<size_t>(cascadeCount) *
                               (staticShadowCasterCache_.size() +
                                frameDynamicTemplateIndices.size()));
  auto remapResult = ensureInstanceRemapRingCapacity(std::max(
      maxShadowInstanceRemapCount * sizeof(uint32_t), sizeof(uint32_t)));
  if (remapResult.hasError()) {
    return remapResult;
  }
  const BufferHandle instanceRemapBuffer =
      bufferRings_[InstanceRemapRing][frameSlot].buffer->handle();
  const uint64_t instanceRemapAddress =
      gpu_.getBufferDeviceAddress(instanceRemapBuffer);
  if (instanceRemapAddress == 0u) {
    return Result<bool, std::string>::makeError(
        "ShadowRenderer: invalid instance remap buffer address");
  }
  instanceRemap_.clear();
  instanceRemap_.reserve(maxShadowInstanceRemapCount);
  const size_t staticBatchEntryCount =
      static_cast<size_t>(cascadeCount) * staticShadowBatchTemplates_.size();
  const size_t dynamicBatchEstimate =
      static_cast<size_t>(cascadeCount) * frameDynamicTemplateIndices.size();
  ScopedScratch batchScratch(batchBuildScratchArena_);
  std::pmr::memory_resource *batchMemory = batchScratch.resource();
  PmrHashMap<ShadowBatchKey, uint32_t, ShadowBatchKeyHash> shadowBatchLookup(
      batchMemory);
  shadowBatchLookup.reserve(dynamicBatchEstimate + 1u);
  std::pmr::vector<ShadowBatchEntry> shadowBatchEntries(batchMemory);
  shadowBatchEntries.reserve(staticBatchEntryCount + dynamicBatchEstimate);
  std::pmr::vector<uint32_t> shadowBatchInstanceIndices(batchMemory);
  shadowBatchInstanceIndices.reserve(
      std::min<size_t>(maxShadowInstanceRemapCount, 4096u));
  std::array<uint32_t, kMaxShadowCascades> cascadeStaticCasterCounts{};
  std::array<uint32_t, kMaxShadowCascades> cascadeDynamicCasterCounts{};
  std::array<uint32_t, kMaxShadowCascades> cascadeStaticBatchFullEmitCounts{};
  std::array<uint32_t, kMaxShadowCascades>
      cascadeStaticLightGridQueryCellCounts{};
  std::array<uint32_t, kMaxShadowCascades>
      cascadeStaticLightGridCandidateCounts{};
  std::array<uint32_t, kMaxShadowCascades>
      cascadeStaticLightGridFallbackScanCounts{};
  uint32_t staticLightGridQueryCount = 0u;
  const uint32_t renderableCount = saturateToU32(renderables.size());
  const auto appendShadowDraw =
      [&](uint32_t cascadeIndex, BufferHandle vertexBuffer,
          BufferHandle indexBuffer, uint64_t indexBufferOffset,
          IndexFormat indexFormat, uint32_t indexCount, uint32_t firstIndex,
          uint32_t firstInstance, uint64_t vertexBufferAddress,
          BufferHandle vertexDecodeBuffer, uint64_t vertexDecodeBufferAddress,
          uint32_t vertexDecodeIndex, uint32_t packedVertexFormat,
          uint32_t materialIndex, bool doubleSided, bool alphaMasked,
          bool dynamicCaster, bool buffersAlreadyPreResolved) {
        if (!buffersAlreadyPreResolved) {
          appendUniqueBufferHandle(preResolvedDrawBuffers_, vertexBuffer);
          appendUniqueBufferHandle(preResolvedDrawBuffers_, indexBuffer);
        }
        const ShadowBatchKey key{
            .cascadeIndex = cascadeIndex,
            .dynamicCaster = dynamicCaster,
            .pipeline =
                shadowPipelines_[shadowPipelineIndex(doubleSided, alphaMasked)],
            .vertexBuffer = vertexBuffer,
            .vertexDecodeBuffer = vertexDecodeBuffer,
            .indexBuffer = indexBuffer,
            .indexBufferOffset = indexBufferOffset,
            .indexFormat = indexFormat,
            .indexCount = indexCount,
            .firstIndex = firstIndex,
            .vertexBufferAddress = vertexBufferAddress,
            .vertexDecodeBufferAddress = vertexDecodeBufferAddress,
            .vertexDecodeIndex = vertexDecodeIndex,
            .packedVertexFormat = packedVertexFormat,
            .materialIndex = alphaMasked ? materialIndex : 0u,
        };
        auto it = shadowBatchLookup.find(key);
        if (it == shadowBatchLookup.end()) {
          const uint32_t batchIndex =
              static_cast<uint32_t>(shadowBatchEntries.size());
          shadowBatchEntries.emplace_back();
          shadowBatchEntries.back().key = key;
          auto [insertedIt, _] = shadowBatchLookup.emplace(key, batchIndex);
          it = insertedIt;
        }
        ShadowBatchEntry &batch = shadowBatchEntries[it->second];
        batch.appendInstance(firstInstance, shadowBatchInstanceIndices);
        if (dynamicCaster) {
          ++cascadeDynamicCasterCounts[cascadeIndex];
        } else {
          ++cascadeStaticCasterCounts[cascadeIndex];
        }
      };
  const auto makeStaticShadowBatchKey =
      [](const StaticShadowBatchTemplate &batchTemplate,
         uint32_t cascadeIndex) {
        return ShadowBatchKey{
            .cascadeIndex = cascadeIndex,
            .dynamicCaster = false,
            .pipeline = batchTemplate.pipeline,
            .vertexBuffer = batchTemplate.vertexBuffer,
            .vertexDecodeBuffer = batchTemplate.vertexDecodeBuffer,
            .indexBuffer = batchTemplate.indexBuffer,
            .indexBufferOffset = batchTemplate.indexBufferOffset,
            .indexFormat = batchTemplate.indexFormat,
            .indexCount = batchTemplate.indexCount,
            .firstIndex = batchTemplate.firstIndex,
            .vertexBufferAddress = batchTemplate.vertexBufferAddress,
            .vertexDecodeBufferAddress =
                batchTemplate.vertexDecodeBufferAddress,
            .vertexDecodeIndex = batchTemplate.vertexDecodeIndex,
            .packedVertexFormat = batchTemplate.packedVertexFormat,
            .materialIndex = batchTemplate.materialIndex,
        };
      };
  std::array<shadow_detail::DirectionalShadowFit, kMaxShadowCascades>
      guardedStaticOnlyFits = currentRawShadowFits_;
  for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
       ++cascadeIndex) {
    const StaticOnlyGuardBandTexels guard = staticOnlyRenderGuardBandTexels(
        currentRawShadowFits_[cascadeIndex],
        cascadeStates_[cascadeIndex].reusable.rawFit,
        cascadeStates_[cascadeIndex].reusableValid,
        settings.shadow.shadowMapSize);
    applyStaticOnlyGuardBand(guardedStaticOnlyFits[cascadeIndex],
                             cascadeStates_[cascadeIndex].reusable.rawFit,
                             cascadeStates_[cascadeIndex].reusableValid,
                             settings.shadow.shadowMapSize, guard);
  }
  std::array<uint8_t, kMaxShadowCascades> cascadeHasGuardedDynamicDraw{};
  {
    NURI_PROFILER_ZONE("ShadowRenderer.dynamic_guard_scan",
                       NURI_PROFILER_COLOR_CMD_COPY);
    for (const uint32_t templateIndex : frameDynamicTemplateIndices) {
      const MeshDrawTemplate &entry = meshDrawTemplates_[templateIndex];
      bool usesAnimatedOverride = false;
      if (animationSceneData != nullptr) {
        const AnimatedRenderableGeometryOverride &geometryOverride =
            animationSceneData
                ->geometryOverridesByRenderable[entry.instanceIndex];
        usesAnimatedOverride =
            geometryOverride.enabled &&
            nuri::isValid(geometryOverride.vertexBuffer) &&
            animationOverrideCoversSubmesh(geometryOverride, *entry.submesh);
      }
      std::array<glm::vec3, 8> casterWorldCorners{};
      bool hasCasterCullingBounds = false;
      if (enableCascadeCasterCulling && !usesAnimatedOverride) {
        const BoundingBox worldBounds =
            entry.submesh->bounds.getTransformed(entry.renderable->modelMatrix);
        casterWorldCorners = shadow_detail::computeBoundsCorners(
            worldBounds.min_, worldBounds.max_);
        hasCasterCullingBounds = true;
      }
      for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
           ++cascadeIndex) {
        if (hasCasterCullingBounds &&
            !shadowCasterOverlapsDirectionalShadowFit(
                std::span<const glm::vec3, 8>(casterWorldCorners),
                guardedStaticOnlyFits[cascadeIndex])) {
          continue;
        }
        cascadeHasGuardedDynamicDraw[cascadeIndex] = 1u;
      }
    }
    NURI_PROFILER_ZONE_END();
  }
  {
    NURI_PROFILER_ZONE("ShadowRenderer.write_cascade_fits",
                       NURI_PROFILER_COLOR_CMD_COPY);
    for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
         ++cascadeIndex) {
      const bool useGuardedStaticOnlyFit =
          cascadeHasGuardedDynamicDraw[cascadeIndex] == 0u;
      const shadow_detail::DirectionalShadowFit &renderedFit =
          useGuardedStaticOnlyFit ? guardedStaticOnlyFits[cascadeIndex]
                                  : currentRawShadowFits_[cascadeIndex];
      writeShadowCascadeFit(renderedFit, cascadeIndex,
                            shadowDepthTextures_[cascadeIndex], gpu_,
                            shadowFrameCpuData_, shadowDebugFrameData_);
      StaticOnlyCascadeReuseState &state =
          currentStaticOnlyCascadeStates[cascadeIndex];
      state.renderedFit = renderedFit;
      state.rawFit = currentRawShadowFits_[cascadeIndex];
      state.lightViewProjSignature =
          makeStaticOnlyCascadeLightViewProjSignature(
              shadowFrameCpuData_.cascades[cascadeIndex]);
      state.biasSignature = makeStaticOnlyCascadeBiasSignature(
          settings.shadow.constantBias, settings.shadow.slopeBias);
      state.casterSignature = kFnvOffsetBasis64;
      state.rasterSignature = makeStaticOnlyCascadeRasterSignatureSeed(
          shadowFrameCpuData_.cascades[cascadeIndex],
          settings.shadow.constantBias, settings.shadow.slopeBias);
    }
    NURI_PROFILER_ZONE_END();
  }
  std::array<uint8_t, kMaxShadowCascades> skipStaticCasterBuildForCascade{};
  const bool canUseFastStaticOnlyReusePath =
      staticCacheReused && overriddenStaticTemplateCount == 0u;
  if (canUseFastStaticOnlyReusePath) {
    uint32_t staticOnlyAdaptiveRefreshesThisFrame = 0u;
    for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
         ++cascadeIndex) {
      StaticOnlyCascadeReuseState &currentState =
          currentStaticOnlyCascadeStates[cascadeIndex];
      const StaticOnlyCascadeReuseState &previousState =
          cascadeStates_[cascadeIndex].reusable;
      const bool staticOnlyCandidate =
          cascadeHasGuardedDynamicDraw[cascadeIndex] == 0u;
      const bool previousValid = cascadeStates_[cascadeIndex].reusableValid;
      const bool biasChanged = previousValid && previousState.biasSignature !=
                                                    currentState.biasSignature;
      const bool cachedRenderedFitContainsCurrent =
          previousValid &&
          staticOnlyRenderedFitContainsCurrent(
              previousState.renderedFit, currentRawShadowFits_[cascadeIndex]);
      const bool needsAdaptiveRefresh =
          cachedRenderedFitContainsCurrent &&
          staticOnlyNeedsAdaptiveRefresh(previousState.renderedFit,
                                         currentRawShadowFits_[cascadeIndex],
                                         previousState.rawFit, previousValid,
                                         settings.shadow.shadowMapSize);
      const bool deferAdaptiveRefresh =
          needsAdaptiveRefresh && staticOnlyAdaptiveRefreshesThisFrame >=
                                      kStaticOnlyAdaptiveRefreshBudgetPerFrame;
      if (!staticOnlyCandidate || !previousValid || biasChanged ||
          !cachedRenderedFitContainsCurrent) {
        continue;
      }
      if (needsAdaptiveRefresh && !deferAdaptiveRefresh) {
        ++staticOnlyAdaptiveRefreshesThisFrame;
        continue;
      }
      skipStaticCasterBuildForCascade[cascadeIndex] = 1u;
      currentState.casterSignature = previousState.casterSignature;
      currentState.rasterSignature = previousState.rasterSignature;
      currentState.staticDrawCount = previousState.staticDrawCount;
      currentState.dynamicDrawCount = 0u;
    }
  }
  if (enableCascadeCasterCulling && !staticShadowCasterCache_.empty()) {
    const glm::mat4 staticCasterLightView =
        shadowDebugFrameData_.cascades[0].lightView;
    if (!hasStaticShadowCasterLightSpaceBounds_ ||
        !rawBytesEqual(staticShadowCasterLightSpaceBoundsView_,
                       staticCasterLightView)) {
      NURI_PROFILER_ZONE("ShadowRenderer.static_caster_light_bounds",
                         NURI_PROFILER_COLOR_CMD_COPY);
      for (StaticShadowBatchTemplate &batch : staticShadowBatchTemplates_) {
        StaticShadowCasterLightSpaceBounds &bounds = batch.lightBounds;
        bounds.min = glm::vec3(std::numeric_limits<float>::max());
        bounds.max = glm::vec3(std::numeric_limits<float>::lowest());
      }
      glm::vec3 staticLightBoundsMin(std::numeric_limits<float>::max());
      glm::vec3 staticLightBoundsMax(std::numeric_limits<float>::lowest());
      for (size_t entryIndex = 0u; entryIndex < staticShadowCasterCache_.size();
           ++entryIndex) {
        const StaticShadowCasterCacheEntry &entry =
            staticShadowCasterCache_[entryIndex];
        glm::vec3 lightMin(std::numeric_limits<float>::max());
        glm::vec3 lightMax(std::numeric_limits<float>::lowest());
        for (const glm::vec3 corner : entry.casterWorldCorners) {
          const glm::vec3 lightCorner =
              glm::vec3(staticCasterLightView * glm::vec4(corner, 1.0f));
          lightMin = glm::min(lightMin, lightCorner);
          lightMax = glm::max(lightMax, lightCorner);
        }
        staticShadowCasterCache_[entryIndex].lightBounds = {
            .min = lightMin,
            .max = lightMax,
        };
        StaticShadowCasterLightSpaceBounds &batchBounds =
            staticShadowBatchTemplates_[entry.batchIndex].lightBounds;
        batchBounds.min = glm::min(batchBounds.min, lightMin);
        batchBounds.max = glm::max(batchBounds.max, lightMax);
        staticLightBoundsMin =
            glm::min(staticLightBoundsMin, glm::min(lightMin, lightMax));
        staticLightBoundsMax =
            glm::max(staticLightBoundsMax, glm::max(lightMin, lightMax));
      }
      staticShadowCasterLightSpaceBoundsView_ = staticCasterLightView;
      staticShadowCasterLightSpaceBoundsMin_ = staticLightBoundsMin;
      staticShadowCasterLightSpaceBoundsMax_ = staticLightBoundsMax;
      hasStaticShadowCasterLightSpaceBounds_ = true;
      for (auto &cells : staticLightGridCells_)
        cells.clear();
      for (auto &entries : staticLightGridEntries_)
        entries.clear();
      for (auto &entries : staticLargeLightGridEntries_)
        entries.clear();
      staticLightGrids_ = {};
      NURI_PROFILER_ZONE_END();
    }
    if ((!staticLightGrids_[CasterGrid].valid ||
         !staticLightGrids_[BatchGrid].valid) &&
        hasStaticShadowCasterLightSpaceBounds_) {
      NURI_PROFILER_ZONE("ShadowRenderer.static_caster_light_grid",
                         NURI_PROFILER_COLOR_CMD_COPY);
      glm::vec3 gridMin = staticShadowCasterLightSpaceBoundsMin_;
      glm::vec3 gridMax = staticShadowCasterLightSpaceBoundsMax_;
      const glm::vec3 gridExtent =
          glm::max(gridMax - gridMin, glm::vec3(0.01f));
      gridMax = gridMin + gridExtent;
      const glm::uvec3 gridDimensions(kStaticCasterLightGridResolution,
                                      kStaticCasterLightGridResolution,
                                      kStaticCasterLightGridDepthResolution);
      const uint32_t gridCellCount =
          gridDimensions.x * gridDimensions.y * gridDimensions.z;
      const glm::vec3 invCellSize = glm::vec3(gridDimensions) / gridExtent;
      struct StaticCasterLightGridRange {
        uint32_t minX = 0u;
        uint32_t minY = 0u;
        uint32_t minZ = 0u;
        uint32_t maxX = 0u;
        uint32_t maxY = 0u;
        uint32_t maxZ = 0u;
      };
      const auto gridCellIndex = [&](uint32_t x, uint32_t y, uint32_t z) {
        return (z * gridDimensions.y + y) * gridDimensions.x + x;
      };
      const auto gridCellRange =
          [&](const StaticShadowCasterLightSpaceBounds &bounds) {
            const glm::vec3 boundsMin = glm::min(bounds.min, bounds.max);
            const glm::vec3 boundsMax = glm::max(bounds.min, bounds.max);
            const glm::ivec3 minCell = glm::clamp(
                glm::ivec3(glm::floor((boundsMin - gridMin) * invCellSize)),
                glm::ivec3(0), glm::ivec3(gridDimensions) - glm::ivec3(1));
            const glm::ivec3 maxCell = glm::clamp(
                glm::ivec3(glm::floor((boundsMax - gridMin) * invCellSize)),
                glm::ivec3(0), glm::ivec3(gridDimensions) - glm::ivec3(1));
            return StaticCasterLightGridRange{
                .minX = static_cast<uint32_t>(std::min(minCell.x, maxCell.x)),
                .minY = static_cast<uint32_t>(std::min(minCell.y, maxCell.y)),
                .minZ = static_cast<uint32_t>(std::min(minCell.z, maxCell.z)),
                .maxX = static_cast<uint32_t>(std::max(minCell.x, maxCell.x)),
                .maxY = static_cast<uint32_t>(std::max(minCell.y, maxCell.y)),
                .maxZ = static_cast<uint32_t>(std::max(minCell.z, maxCell.z)),
            };
          };
      const auto forEachGridItem = [&](size_t kind, auto fn) {
        if (kind == CasterGrid) {
          for (uint32_t index = 0; index < staticShadowCasterCache_.size();
               ++index)
            fn(index, staticShadowCasterCache_[index].lightBounds);
        } else {
          for (uint32_t index = 0; index < staticShadowBatchTemplates_.size();
               ++index)
            if (staticShadowBatchTemplates_[index].instanceCount >=
                kStaticBatchOverlapEmitMinInstances)
              fn(index, staticShadowBatchTemplates_[index].lightBounds);
        }
      };
      std::array<std::pmr::vector<uint32_t>, LightGridCount> counts{
          std::pmr::vector<uint32_t>(batchMemory),
          std::pmr::vector<uint32_t>(batchMemory)};
      for (size_t kind = 0; kind < LightGridCount; ++kind) {
        counts[kind].resize(gridCellCount, 0u);
        staticLargeLightGridEntries_[kind].clear();
        forEachGridItem(
            kind, [&](uint32_t index,
                      const StaticShadowCasterLightSpaceBounds &bounds) {
              const StaticCasterLightGridRange range = gridCellRange(bounds);
              const uint32_t coveredCells = (range.maxX - range.minX + 1u) *
                                            (range.maxY - range.minY + 1u) *
                                            (range.maxZ - range.minZ + 1u);
              if (coveredCells > kStaticCasterLightGridMaxCellsPerCaster) {
                staticLargeLightGridEntries_[kind].push_back(index);
                return;
              }
              for (uint32_t z = range.minZ; z <= range.maxZ; ++z)
                for (uint32_t y = range.minY; y <= range.maxY; ++y)
                  for (uint32_t x = range.minX; x <= range.maxX; ++x)
                    ++counts[kind][gridCellIndex(x, y, z)];
            });
      }
      std::array<std::pmr::vector<uint32_t>, LightGridCount> writeOffsets{
          std::pmr::vector<uint32_t>(batchMemory),
          std::pmr::vector<uint32_t>(batchMemory)};
      for (size_t kind = 0; kind < LightGridCount; ++kind) {
        auto &cells = staticLightGridCells_[kind];
        cells.resize(gridCellCount);
        uint32_t runningCount = 0u;
        for (uint32_t cellIndex = 0; cellIndex < gridCellCount; ++cellIndex) {
          cells[cellIndex] = {.firstEntry = runningCount,
                              .entryCount = counts[kind][cellIndex]};
          runningCount += counts[kind][cellIndex];
        }
        staticLightGridEntries_[kind].resize(runningCount);
        writeOffsets[kind].resize(gridCellCount);
        for (uint32_t cellIndex = 0; cellIndex < gridCellCount; ++cellIndex)
          writeOffsets[kind][cellIndex] = cells[cellIndex].firstEntry;
        forEachGridItem(
            kind, [&](uint32_t index,
                      const StaticShadowCasterLightSpaceBounds &bounds) {
              const StaticCasterLightGridRange range = gridCellRange(bounds);
              const uint32_t coveredCells = (range.maxX - range.minX + 1u) *
                                            (range.maxY - range.minY + 1u) *
                                            (range.maxZ - range.minZ + 1u);
              if (coveredCells > kStaticCasterLightGridMaxCellsPerCaster)
                return;
              for (uint32_t z = range.minZ; z <= range.maxZ; ++z)
                for (uint32_t y = range.minY; y <= range.maxY; ++y)
                  for (uint32_t x = range.minX; x <= range.maxX; ++x) {
                    const uint32_t cell = gridCellIndex(x, y, z);
                    staticLightGridEntries_[kind][writeOffsets[kind][cell]++] =
                        index;
                  }
            });
      }
      staticLightGrids_[CasterGrid] = {
          .min = gridMin,
          .max = gridMax,
          .invCellSize = invCellSize,
          .dimensions = gridDimensions,
          .valid = true,
      };
      staticLightGrids_[BatchGrid] = staticLightGrids_[CasterGrid];
      NURI_PROFILER_ZONE_END();
    }
  }
  std::pmr::vector<uint32_t> staticBatchEntryIndices(batchMemory);
  staticBatchEntryIndices.resize(staticBatchEntryCount,
                                 std::numeric_limits<uint32_t>::max());
  std::pmr::vector<uint8_t> fullyEmittedStaticBatchEntries(batchMemory);
  fullyEmittedStaticBatchEntries.resize(staticBatchEntryCount, uint8_t{0});
  std::pmr::vector<uint32_t> staticBatchLightGridQueryEntries(batchMemory);
  std::pmr::vector<uint32_t> staticBatchLightGridQueryMarks(batchMemory);
  uint32_t staticBatchLightGridQueryMarker = 1u;
  const auto staticBatchEntrySlot = [&](uint32_t cascadeIndex,
                                        uint32_t staticBatchIndex) {
    return static_cast<size_t>(cascadeIndex) *
               staticShadowBatchTemplates_.size() +
           staticBatchIndex;
  };
  const auto resolveStaticBatchEntry =
      [&](uint32_t cascadeIndex,
          uint32_t staticBatchIndex) -> ShadowBatchEntry & {
    const size_t slotIndex =
        staticBatchEntrySlot(cascadeIndex, staticBatchIndex);
    uint32_t &entryIndex = staticBatchEntryIndices[slotIndex];
    if (entryIndex == std::numeric_limits<uint32_t>::max()) {
      entryIndex = static_cast<uint32_t>(shadowBatchEntries.size());
      shadowBatchEntries.emplace_back();
      ShadowBatchEntry &batch = shadowBatchEntries.back();
      const StaticShadowBatchTemplate &batchTemplate =
          staticShadowBatchTemplates_[staticBatchIndex];
      batch.key = makeStaticShadowBatchKey(batchTemplate, cascadeIndex);
    }
    return shadowBatchEntries[entryIndex];
  };
  std::array<glm::vec3, kMaxShadowCascades> cascadeLightBoundsMin{};
  std::array<glm::vec3, kMaxShadowCascades> cascadeLightBoundsMax{};
  std::array<uint8_t, kMaxShadowCascades> cascadeUsesCachedLightBounds{};
  for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
       ++cascadeIndex) {
    const ShadowCascadeDebugFrameData &cascadeDebug =
        shadowDebugFrameData_.cascades[cascadeIndex];
    const float cullingPadding = std::max(
        cascadeDebug.texelWorldSize * kCullingPaddingTexelMultiplier, 0.01f);
    const glm::vec3 rawBoundsMin(cascadeDebug.lightSpaceBoundsMin);
    const glm::vec3 rawBoundsMax(cascadeDebug.lightSpaceBoundsMax);
    cascadeLightBoundsMin[cascadeIndex] =
        glm::min(rawBoundsMin, rawBoundsMax) - glm::vec3(cullingPadding);
    cascadeLightBoundsMax[cascadeIndex] =
        glm::max(rawBoundsMin, rawBoundsMax) + glm::vec3(cullingPadding);
    cascadeUsesCachedLightBounds[cascadeIndex] =
        hasStaticShadowCasterLightSpaceBounds_ &&
                rawBytesEqual(staticShadowCasterLightSpaceBoundsView_,
                              cascadeDebug.lightView)
            ? 1u
            : 0u;
  }
  const uint32_t staticCandidateCasterCount = saturateToU32(
      staticShadowCasterCache_.size() -
      std::min(staticShadowCasterCache_.size(),
               static_cast<size_t>(overriddenStaticTemplateCount)));
  if (staticLightGrids_[CasterGrid].valid) {
    staticShadowCasterLightGridQueryEntries_.reserve(
        std::min(staticShadowCasterCache_.size(),
                 kStaticCasterLightGridSortedCandidateLimit));
  }
  if (staticLightGrids_[BatchGrid].valid) {
    staticBatchLightGridQueryMarks.resize(staticShadowBatchTemplates_.size(),
                                          0u);
    staticBatchLightGridQueryEntries.reserve(
        std::min(staticShadowBatchTemplates_.size(),
                 kStaticCasterLightGridSortedCandidateLimit));
  }
  const auto nextStaticCasterQueryMarker = [&]() {
    ++staticShadowCasterLightGridQueryMarker_;
    if (staticShadowCasterLightGridQueryMarker_ == 0u) {
      std::fill(staticShadowCasterLightGridQueryMarks_.begin(),
                staticShadowCasterLightGridQueryMarks_.end(), 0u);
      staticShadowCasterLightGridQueryMarker_ = 1u;
    }
    return staticShadowCasterLightGridQueryMarker_;
  };
  const auto nextStaticBatchQueryMarker = [&]() {
    ++staticBatchLightGridQueryMarker;
    if (staticBatchLightGridQueryMarker == 0u) {
      std::fill(staticBatchLightGridQueryMarks.begin(),
                staticBatchLightGridQueryMarks.end(), 0u);
      staticBatchLightGridQueryMarker = 1u;
    }
    return staticBatchLightGridQueryMarker;
  };
  const auto staticEntryUsesDynamicPath =
      [&](const StaticShadowCasterCacheEntry &entry) {
        return staticTemplatesUsingDynamicPath[entry.templateIndex] != 0u;
      };
  const auto staticBatchWasFullyEmitted = [&](uint32_t staticBatchIndex,
                                              uint32_t cascadeIndex) {
    const size_t slotIndex =
        staticBatchEntrySlot(cascadeIndex, staticBatchIndex);
    return fullyEmittedStaticBatchEntries[slotIndex] != 0u;
  };
  const auto staticCasterDepthOverlapsCascade = [&](uint32_t entryIndex,
                                                    uint32_t cascadeIndex) {
    const StaticShadowCasterLightSpaceBounds &bounds =
        staticShadowCasterCache_[entryIndex].lightBounds;
    return bounds.max.z >= cascadeLightBoundsMin[cascadeIndex].z &&
           bounds.min.z <= cascadeLightBoundsMax[cascadeIndex].z;
  };
  const auto emitStaticCasterUnchecked = [&](uint32_t entryIndex,
                                             uint32_t cascadeIndex) {
    const StaticShadowCasterCacheEntry &entry =
        staticShadowCasterCache_[entryIndex];
    ShadowBatchEntry &batch =
        resolveStaticBatchEntry(cascadeIndex, entry.batchIndex);
    batch.appendInstance(
        entry.instanceIndex, shadowBatchInstanceIndices,
        staticShadowBatchTemplates_[entry.batchIndex].instanceCount);
    ++cascadeStaticCasterCounts[cascadeIndex];
    StaticOnlyCascadeReuseState &cascadeState =
        currentStaticOnlyCascadeStates[cascadeIndex];
    cascadeState.casterSignature =
        hashCombine64(cascadeState.casterSignature, entry.rasterSignature);
    cascadeState.rasterSignature =
        hashCombine64(cascadeState.rasterSignature, entry.rasterSignature);
    cascadeStates_[cascadeIndex].indexCountEstimate += entry.indexCount;
    return true;
  };
  const auto emitStaticBatch = [&](uint32_t batchIndex, uint32_t cascadeIndex) {
    const StaticShadowBatchTemplate &batchTemplate =
        staticShadowBatchTemplates_[batchIndex];
    const size_t firstInstanceIndex = batchTemplate.firstInstanceIndex;
    ShadowBatchEntry &batch = resolveStaticBatchEntry(cascadeIndex, batchIndex);
    batch.appendInstances(
        std::span<const uint32_t>(staticShadowBatchInstanceIndices_.data() +
                                      firstInstanceIndex,
                                  batchTemplate.instanceCount),
        shadowBatchInstanceIndices);
  };
  const auto emitStaticBatchWithState = [&](uint32_t batchIndex,
                                            uint32_t cascadeIndex) {
    const StaticShadowBatchTemplate &batchTemplate =
        staticShadowBatchTemplates_[batchIndex];
    emitStaticBatch(batchIndex, cascadeIndex);
    cascadeStaticCasterCounts[cascadeIndex] += batchTemplate.instanceCount;
    StaticOnlyCascadeReuseState &cascadeState =
        currentStaticOnlyCascadeStates[cascadeIndex];
    cascadeState.casterSignature = hashCombine64(cascadeState.casterSignature,
                                                 batchTemplate.rasterSignature);
    cascadeState.rasterSignature = hashCombine64(cascadeState.rasterSignature,
                                                 batchTemplate.rasterSignature);
    cascadeStates_[cascadeIndex].indexCountEstimate +=
        batchTemplate.indexCountEstimate;
  };
  const auto emitAllStaticCastersForCascade = [&](uint32_t cascadeIndex) {
    for (uint32_t batchIndex = 0u;
         batchIndex < staticShadowBatchTemplates_.size(); ++batchIndex) {
      ++cascadeStaticBatchFullEmitCounts[cascadeIndex];
      emitStaticBatch(batchIndex, cascadeIndex);
    }
    cascadeStaticCasterCounts[cascadeIndex] += staticCandidateCasterCount;
    StaticOnlyCascadeReuseState &cascadeState =
        currentStaticOnlyCascadeStates[cascadeIndex];
    cascadeState.casterSignature = staticShadowCasterCacheContentSignature_;
    cascadeState.rasterSignature = hashCombine64(
        cascadeState.rasterSignature, staticShadowCasterCacheContentSignature_);
    cascadeStates_[cascadeIndex].indexCountEstimate +=
        staticShadowCasterCacheIndexCountEstimate_;
  };
  const auto cascadeContainsAllStaticCasters = [&](uint32_t cascadeIndex) {
    if (staticShadowCasterCache_.empty()) {
      return true;
    }
    if (!enableCascadeCasterCulling) {
      return true;
    }
    if (cascadeUsesCachedLightBounds[cascadeIndex] == 0u ||
        !hasStaticShadowCasterLightSpaceBounds_) {
      return false;
    }
    const glm::vec3 &cascadeMin = cascadeLightBoundsMin[cascadeIndex];
    const glm::vec3 &cascadeMax = cascadeLightBoundsMax[cascadeIndex];
    return staticShadowCasterLightSpaceBoundsMin_.x >= cascadeMin.x &&
           staticShadowCasterLightSpaceBoundsMin_.y >= cascadeMin.y &&
           staticShadowCasterLightSpaceBoundsMin_.z >= cascadeMin.z &&
           staticShadowCasterLightSpaceBoundsMax_.x <= cascadeMax.x &&
           staticShadowCasterLightSpaceBoundsMax_.y <= cascadeMax.y &&
           staticShadowCasterLightSpaceBoundsMax_.z <= cascadeMax.z;
  };
  const auto emitOverlappingStaticBatch = [&](uint32_t cascadeIndex,
                                              uint32_t batchIndex) {
    const StaticShadowBatchTemplate &batchTemplate =
        staticShadowBatchTemplates_[batchIndex];
    if (batchTemplate.instanceCount < kStaticBatchOverlapEmitMinInstances) {
      return;
    }
    const StaticShadowCasterLightSpaceBounds &batchBounds =
        batchTemplate.lightBounds;
    if (!normalizedLightSpaceBoundsOverlap(
            batchBounds.min, batchBounds.max,
            cascadeLightBoundsMin[cascadeIndex],
            cascadeLightBoundsMax[cascadeIndex])) {
      return;
    }
    const size_t slotIndex = staticBatchEntrySlot(cascadeIndex, batchIndex);
    fullyEmittedStaticBatchEntries[slotIndex] = 1u;
    emitStaticBatchWithState(batchIndex, cascadeIndex);
    ++cascadeStaticBatchFullEmitCounts[cascadeIndex];
  };
  const auto scanOverlappingStaticBatchesForCascade =
      [&](uint32_t cascadeIndex) {
        for (uint32_t batchIndex = 0u;
             batchIndex < staticShadowBatchTemplates_.size(); ++batchIndex) {
          emitOverlappingStaticBatch(cascadeIndex, batchIndex);
        }
      };
  struct StaticLightGridQueryRange {
    uint32_t minX = 0u;
    uint32_t minY = 0u;
    uint32_t minZ = 0u;
    uint32_t maxX = 0u;
    uint32_t maxY = 0u;
    uint32_t maxZ = 0u;
    uint32_t cellCount = 0u;
    bool outside = false;
    bool tooBroad = false;
  };
  const auto makeStaticLightGridQueryRange =
      [&](const StaticShadowCasterLightGrid &grid,
          uint32_t cascadeIndex) -> StaticLightGridQueryRange {
    const glm::vec3 &queryMin = cascadeLightBoundsMin[cascadeIndex];
    const glm::vec3 &queryMax = cascadeLightBoundsMax[cascadeIndex];
    if (queryMax.x < grid.min.x || queryMin.x > grid.max.x ||
        queryMax.y < grid.min.y || queryMin.y > grid.max.y ||
        queryMax.z < grid.min.z || queryMin.z > grid.max.z) {
      return StaticLightGridQueryRange{.outside = true};
    }
    const glm::ivec3 minCell = glm::clamp(
        glm::ivec3(glm::floor((queryMin - grid.min) * grid.invCellSize)),
        glm::ivec3(0), glm::ivec3(grid.dimensions) - glm::ivec3(1));
    const glm::ivec3 maxCell = glm::clamp(
        glm::ivec3(glm::floor((queryMax - grid.min) * grid.invCellSize)),
        glm::ivec3(0), glm::ivec3(grid.dimensions) - glm::ivec3(1));
    StaticLightGridQueryRange range{
        .minX = static_cast<uint32_t>(std::min(minCell.x, maxCell.x)),
        .minY = static_cast<uint32_t>(std::min(minCell.y, maxCell.y)),
        .minZ = static_cast<uint32_t>(std::min(minCell.z, maxCell.z)),
        .maxX = static_cast<uint32_t>(std::max(minCell.x, maxCell.x)),
        .maxY = static_cast<uint32_t>(std::max(minCell.y, maxCell.y)),
        .maxZ = static_cast<uint32_t>(std::max(minCell.z, maxCell.z)),
    };
    range.cellCount = (range.maxX - range.minX + 1u) *
                      (range.maxY - range.minY + 1u) *
                      (range.maxZ - range.minZ + 1u);
    const uint32_t gridCellCount =
        grid.dimensions.x * grid.dimensions.y * grid.dimensions.z;
    range.tooBroad = range.cellCount > gridCellCount / 2u;
    return range;
  };
  const auto queryStaticLightGrid = [&](size_t kind, uint32_t cascadeIndex,
                                        uint32_t queryMarker, auto &marks,
                                        auto &entries, auto acceptCandidate) {
    entries.clear();
    if (!staticLightGrids_[kind].valid ||
        cascadeUsesCachedLightBounds[cascadeIndex] == 0u)
      return false;
    ++staticLightGridQueryCount;
    const StaticShadowCasterLightGrid &grid = staticLightGrids_[kind];
    const StaticLightGridQueryRange query =
        makeStaticLightGridQueryRange(grid, cascadeIndex);
    if (query.outside)
      return true;
    cascadeStaticLightGridQueryCellCounts[cascadeIndex] += query.cellCount;
    if (query.tooBroad)
      return false;
    const auto append = [&](uint32_t index) {
      if (!acceptCandidate(index) || marks[index] == queryMarker)
        return;
      marks[index] = queryMarker;
      entries.push_back(index);
    };
    for (uint32_t index : staticLargeLightGridEntries_[kind])
      append(index);
    for (uint32_t z = query.minZ; z <= query.maxZ; ++z)
      for (uint32_t y = query.minY; y <= query.maxY; ++y)
        for (uint32_t x = query.minX; x <= query.maxX; ++x) {
          const uint32_t cellIndex =
              (z * grid.dimensions.y + y) * grid.dimensions.x + x;
          const StaticShadowCasterLightGridCell &cell =
              staticLightGridCells_[kind][cellIndex];
          for (uint32_t i = cell.firstEntry;
               i < cell.firstEntry + cell.entryCount; ++i)
            append(staticLightGridEntries_[kind][i]);
        }
    return true;
  };
  const auto queryStaticBatchLightGrid = [&](uint32_t cascadeIndex,
                                             uint32_t queryMarker) {
    if (!queryStaticLightGrid(
            BatchGrid, cascadeIndex, queryMarker,
            staticBatchLightGridQueryMarks, staticBatchLightGridQueryEntries,
            [&](uint32_t batchIndex) {
              return staticShadowBatchTemplates_[batchIndex].instanceCount >=
                     kStaticBatchOverlapEmitMinInstances;
            }))
      return false;
    std::sort(staticBatchLightGridQueryEntries.begin(),
              staticBatchLightGridQueryEntries.end());
    cascadeStaticLightGridCandidateCounts[cascadeIndex] +=
        saturateToU32(staticBatchLightGridQueryEntries.size());
    return true;
  };
  const auto emitOverlappingStaticBatchesForCascade =
      [&](uint32_t cascadeIndex) {
        if (overriddenStaticTemplateCount != 0u ||
            cascadeUsesCachedLightBounds[cascadeIndex] == 0u) {
          return;
        }
        const bool queriedBatchGrid = queryStaticBatchLightGrid(
            cascadeIndex, nextStaticBatchQueryMarker());
        if (!queriedBatchGrid) {
          ++cascadeStaticLightGridFallbackScanCounts[cascadeIndex];
          scanOverlappingStaticBatchesForCascade(cascadeIndex);
          return;
        }
        for (const uint32_t batchIndex : staticBatchLightGridQueryEntries) {
          emitOverlappingStaticBatch(cascadeIndex, batchIndex);
        }
      };
  const auto emitGridCandidateIfVisible = [&](uint32_t entryIndex,
                                              uint32_t cascadeIndex) {
    const StaticShadowCasterCacheEntry &entry =
        staticShadowCasterCache_[entryIndex];
    if (staticEntryUsesDynamicPath(entry) ||
        staticBatchWasFullyEmitted(entry.batchIndex, cascadeIndex)) {
      return false;
    }
    if (entry.hasCasterCullingBounds) {
      const bool overlapsCascade = normalizedLightSpaceBoundsOverlap(
          staticShadowCasterCache_[entryIndex].lightBounds.min,
          staticShadowCasterCache_[entryIndex].lightBounds.max,
          cascadeLightBoundsMin[cascadeIndex],
          cascadeLightBoundsMax[cascadeIndex]);
      if (!overlapsCascade) {
        return false;
      }
    }
    return emitStaticCasterUnchecked(entryIndex, cascadeIndex);
  };
  const auto queryStaticCasterLightGrid = [&](uint32_t cascadeIndex,
                                              uint32_t queryMarker,
                                              bool &emitMarkedByCacheOrder) {
    emitMarkedByCacheOrder = false;
    if (!queryStaticLightGrid(
            CasterGrid, cascadeIndex, queryMarker,
            staticShadowCasterLightGridQueryMarks_,
            staticShadowCasterLightGridQueryEntries_, [&](uint32_t entryIndex) {
              const StaticShadowCasterCacheEntry &entry =
                  staticShadowCasterCache_[entryIndex];
              return !staticEntryUsesDynamicPath(entry) &&
                     !staticBatchWasFullyEmitted(entry.batchIndex,
                                                 cascadeIndex) &&
                     staticCasterDepthOverlapsCascade(entryIndex, cascadeIndex);
            }))
      return false;
    if (staticShadowCasterLightGridQueryEntries_.size() >
        kStaticCasterLightGridSortedCandidateLimit) {
      emitMarkedByCacheOrder = true;
    } else {
      std::sort(staticShadowCasterLightGridQueryEntries_.begin(),
                staticShadowCasterLightGridQueryEntries_.end());
    }
    cascadeStaticLightGridCandidateCounts[cascadeIndex] +=
        saturateToU32(staticShadowCasterLightGridQueryEntries_.size());
    return true;
  };
  const auto scanStaticCastersForCascade = [&](uint32_t cascadeIndex) {
    for (uint32_t entryIndex = 0u; entryIndex < staticShadowCasterCache_.size();
         ++entryIndex) {
      const StaticShadowCasterCacheEntry &entry =
          staticShadowCasterCache_[entryIndex];
      if (staticEntryUsesDynamicPath(entry) ||
          staticBatchWasFullyEmitted(entry.batchIndex, cascadeIndex)) {
        continue;
      }
      if (entry.hasCasterCullingBounds) {
        const bool overlapsCascade =
            cascadeUsesCachedLightBounds[cascadeIndex] != 0u
                ? normalizedLightSpaceBoundsOverlap(
                      staticShadowCasterCache_[entryIndex].lightBounds.min,
                      staticShadowCasterCache_[entryIndex].lightBounds.max,
                      cascadeLightBoundsMin[cascadeIndex],
                      cascadeLightBoundsMax[cascadeIndex])
                : shadow_detail::shadowCasterOverlapsLightSpaceBounds(
                      std::span<const glm::vec3, 8>(entry.casterWorldCorners),
                      shadowDebugFrameData_.cascades[cascadeIndex].lightView,
                      cascadeLightBoundsMin[cascadeIndex],
                      cascadeLightBoundsMax[cascadeIndex], 0.0f);
        if (!overlapsCascade) {
          ++cascadeStates_[cascadeIndex].culledCount;
          continue;
        }
      }
      (void)emitStaticCasterUnchecked(entryIndex, cascadeIndex);
    }
  };
  {
    NURI_PROFILER_ZONE("ShadowRenderer.static_caster_emit",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
         ++cascadeIndex) {
      if (skipStaticCasterBuildForCascade[cascadeIndex] != 0u) {
        continue;
      }
      if (overriddenStaticTemplateCount == 0u &&
          cascadeContainsAllStaticCasters(cascadeIndex)) {
        emitAllStaticCastersForCascade(cascadeIndex);
        continue;
      }
      const uint32_t staticDrawCountBefore =
          cascadeStaticCasterCounts[cascadeIndex];
      emitOverlappingStaticBatchesForCascade(cascadeIndex);
      bool emitMarkedByCacheOrder = false;
      const uint32_t queryMarker = nextStaticCasterQueryMarker();
      const bool queriedStaticLightGrid =
          enableCascadeCasterCulling &&
          queryStaticCasterLightGrid(cascadeIndex, queryMarker,
                                     emitMarkedByCacheOrder);
      if (!queriedStaticLightGrid) {
        if (enableCascadeCasterCulling) {
          ++cascadeStaticLightGridFallbackScanCounts[cascadeIndex];
        }
        scanStaticCastersForCascade(cascadeIndex);
        continue;
      }
      if (emitMarkedByCacheOrder) {
        for (uint32_t entryIndex = 0u;
             entryIndex < staticShadowCasterCache_.size(); ++entryIndex) {
          if (staticShadowCasterLightGridQueryMarks_[entryIndex] !=
              queryMarker) {
            continue;
          }
          (void)emitGridCandidateIfVisible(entryIndex, cascadeIndex);
        }
      } else {
        for (const uint32_t entryIndex :
             staticShadowCasterLightGridQueryEntries_) {
          (void)emitGridCandidateIfVisible(entryIndex, cascadeIndex);
        }
      }
      const uint32_t emittedStaticDrawCount =
          cascadeStaticCasterCounts[cascadeIndex] - staticDrawCountBefore;
      if (staticCandidateCasterCount > emittedStaticDrawCount) {
        cascadeStates_[cascadeIndex].culledCount +=
            staticCandidateCasterCount - emittedStaticDrawCount;
      }
    }
    NURI_PROFILER_ZONE_END();
  }
  {
    NURI_PROFILER_ZONE("ShadowRenderer.dynamic_caster_emit",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    for (const uint32_t templateIndex : frameDynamicTemplateIndices) {
      const MeshDrawTemplate &entry = meshDrawTemplates_[templateIndex];
      const std::optional<SubmeshLod> lod = resolveForwardLod(
          *entry.submesh, settings.opaque.forcedMeshLod, true);
      if (!lod.has_value()) {
        continue;
      }
      BufferHandle resolvedVertexBuffer = entry.baseVertexBuffer;
      uint64_t resolvedVertexBufferAddress = entry.vertexBufferAddress;
      uint64_t resolvedVertexDecodeBufferAddress =
          entry.vertexDecodeBufferAddress;
      uint32_t resolvedVertexDecodeIndex = entry.vertexDecodeIndex;
      uint32_t resolvedPackedVertexFormat = entry.packedVertexFormat;
      bool usesAnimatedOverride = false;
      if (animationSceneData != nullptr) {
        const AnimatedRenderableGeometryOverride &geometryOverride =
            animationSceneData
                ->geometryOverridesByRenderable[entry.instanceIndex];
        if (geometryOverride.enabled &&
            nuri::isValid(geometryOverride.vertexBuffer) &&
            animationOverrideCoversSubmesh(geometryOverride, *entry.submesh)) {
          usesAnimatedOverride = true;
          const uint64_t overrideVertexAddress = gpu_.getBufferDeviceAddress(
              geometryOverride.vertexBuffer, geometryOverride.vertexByteOffset);
          resolvedVertexBuffer = geometryOverride.vertexBuffer;
          resolvedVertexBufferAddress = overrideVertexAddress;
          resolvedVertexDecodeBufferAddress = 0u;
          resolvedVertexDecodeIndex = 0u;
          resolvedPackedVertexFormat =
              static_cast<uint32_t>(PackedVertexFormat::AnimatedFloat32);
        }
      }
      std::array<glm::vec3, 8> casterWorldCorners{};
      bool hasCasterCullingBounds = false;
      if (enableCascadeCasterCulling && !usesAnimatedOverride) {
        const BoundingBox worldBounds =
            entry.submesh->bounds.getTransformed(entry.renderable->modelMatrix);
        casterWorldCorners = shadow_detail::computeBoundsCorners(
            worldBounds.min_, worldBounds.max_);
        hasCasterCullingBounds = true;
      }
      for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
           ++cascadeIndex) {
        if (skipStaticCasterBuildForCascade[cascadeIndex] != 0u) {
          continue;
        }
        if (hasCasterCullingBounds) {
          const ShadowCascadeDebugFrameData &cascadeDebug =
              shadowDebugFrameData_.cascades[cascadeIndex];
          const float cullingPadding = std::max(
              cascadeDebug.texelWorldSize * kCullingPaddingTexelMultiplier,
              0.01f);
          if (!shadow_detail::shadowCasterOverlapsLightSpaceBounds(
                  std::span<const glm::vec3, 8>(casterWorldCorners),
                  cascadeDebug.lightView,
                  glm::vec3(cascadeDebug.lightSpaceBoundsMin),
                  glm::vec3(cascadeDebug.lightSpaceBoundsMax),
                  cullingPadding)) {
            ++cascadeStates_[cascadeIndex].culledCount;
            continue;
          }
        }
        appendShadowDraw(cascadeIndex, resolvedVertexBuffer, entry.indexBuffer,
                         entry.indexBufferOffset, entry.indexFormat,
                         lod->indexCount, lod->indexOffset, entry.instanceIndex,
                         resolvedVertexBufferAddress, entry.vertexDecodeBuffer,
                         resolvedVertexDecodeBufferAddress,
                         resolvedVertexDecodeIndex, resolvedPackedVertexFormat,
                         entry.materialIndex, entry.doubleSided,
                         entry.alphaMasked, true, false);
        cascadeStates_[cascadeIndex].indexCountEstimate += lod->indexCount;
      }
    }
    NURI_PROFILER_ZONE_END();
  }
  uint32_t emittedShadowBatchEntryCount = 0u;
  uint32_t emittedShadowInstanceRemapCount = 0u;
  {
    NURI_PROFILER_ZONE("ShadowRenderer.batch_emit",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    std::array<uint32_t, kMaxShadowCascades> cascadeBatchCounts{};
    size_t emittedInstanceRemapCount = 0u;
    for (const ShadowBatchEntry &batch : shadowBatchEntries) {
      ++emittedShadowBatchEntryCount;
      emittedInstanceRemapCount += batch.instanceCount;
      ++cascadeBatchCounts[batch.key.cascadeIndex];
    }
    for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
         ++cascadeIndex) {
      cascadePushConstants_[cascadeIndex].reserve(
          cascadeBatchCounts[cascadeIndex]);
      cascadeDrawItems_[cascadeIndex].reserve(cascadeBatchCounts[cascadeIndex]);
    }
    instanceRemap_.resize(emittedInstanceRemapCount);
    uint32_t *remapWrite = instanceRemap_.data();
    for (ShadowBatchEntry &batch : shadowBatchEntries) {
      if (batch.instanceCount == 0u) {
        continue;
      }
      const ShadowBatchKey &key = batch.key;
      batch.firstInstance = saturateToU32(
          static_cast<size_t>(remapWrite - instanceRemap_.data()));
      if (batch.externalInstanceCount != 0u) {
        std::copy_n(batch.externalInstanceIndices, batch.externalInstanceCount,
                    remapWrite);
        remapWrite += batch.externalInstanceCount;
      } else if (batch.hasInstanceList) {
        std::copy_n(shadowBatchInstanceIndices.data() +
                        batch.instanceListOffset,
                    batch.instanceCount, remapWrite);
        remapWrite += batch.instanceCount;
      } else {
        *remapWrite++ = batch.inlineInstanceIndex;
      }
      PushConstants &pc =
          cascadePushConstants_[key.cascadeIndex].emplace_back();
      pc = PushConstants{
          .frameDataAddress = sceneGpu.frameDataAddress,
          .vertexBufferAddress = key.vertexBufferAddress,
          .vertexDecodeBufferAddress = key.vertexDecodeBufferAddress,
          .instanceMatricesAddress = instanceMatricesAddress,
          .instanceRemapAddress = instanceRemapAddress,
          .instanceCentersPhaseAddress = 0u,
          .instanceBaseMatricesAddress = 0u,
          .instanceCount = renderableCount,
          .materialIndex = key.materialIndex,
          .vertexDecodeIndex = key.vertexDecodeIndex,
          .packedVertexFormat = key.packedVertexFormat,
          .timeSeconds = static_cast<float>(frame.timeSeconds),
          .tessNearDistance = 1.0f,
          .tessFarDistance = 8.0f,
          .tessMinFactor = 1.0f,
          .tessMaxFactor = 1.0f,
          .debugVisualizationMode = 0u,
          .shadowCascadeIndex = key.cascadeIndex,
      };
      DrawItem &draw = cascadeDrawItems_[key.cascadeIndex].emplace_back();
      draw = DrawItem{};
      draw.pipeline = key.pipeline;
      draw.vertexBuffer = key.vertexBuffer;
      draw.indexBuffer = key.indexBuffer;
      draw.indexBufferOffset = key.indexBufferOffset;
      draw.indexFormat = key.indexFormat;
      draw.indexCount = key.indexCount;
      draw.instanceCount = batch.instanceCount;
      draw.firstIndex = key.firstIndex;
      draw.firstInstance = batch.firstInstance;
      draw.useDepthState = true;
      draw.depthState = {.compareOp = CompareOp::Less,
                         .isDepthWriteEnabled = true};
      draw.depthBiasEnable = true;
      draw.depthBiasConstant = shadowSettings.constantBias;
      draw.depthBiasSlope = shadowSettings.slopeBias;
      draw.depthBiasClamp = 0.0f;
      draw.alphaMasked = key.pipeline == shadowPipelines_[2] ||
                         key.pipeline == shadowPipelines_[3];
      draw.pushConstants = std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&pc), sizeof(PushConstants));
      draw.debugLabel = kShadowPassLabel;
      draw.debugColor = kShadowPassDebugColor;
    }
    emittedShadowInstanceRemapCount = saturateToU32(emittedInstanceRemapCount);
    NURI_PROFILER_ZONE_END();
  }
  shadowDrawPacketUploadBytes_.clear();
  BufferHandle shadowDrawPacketBuffer{};
  struct IndirectGroup {
    DrawItem baseDraw{};
    PushConstants baseConstants{};
    std::pmr::vector<DrawIndexedIndirectCommand> commands;
    std::pmr::vector<ShadowDrawGpuData> metadata;
    explicit IndirectGroup(std::pmr::memory_resource *resource)
        : commands(resource), metadata(resource) {}
  };
  for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
       ++cascadeIndex) {
    auto &indirectConstants = cascadeIndirectPushConstants_[cascadeIndex];
    auto &indirectDraws = cascadeIndirectDrawItems_[cascadeIndex];
    indirectConstants.clear();
    indirectDraws.clear();
    const auto &sourceDraws = cascadeDrawItems_[cascadeIndex];
    const auto &sourceConstants = cascadePushConstants_[cascadeIndex];
    if (sourceDraws.empty()) {
      continue;
    }
    PmrHashMap<ShadowIndirectGroupKey, uint32_t, ShadowIndirectGroupKeyHash>
        groupLookup(batchMemory);
    groupLookup.reserve(sourceDraws.size());
    std::pmr::vector<IndirectGroup> groups(batchMemory);
    groups.reserve(sourceDraws.size());
    for (size_t drawIndex = 0u; drawIndex < sourceDraws.size(); ++drawIndex) {
      const DrawItem &sourceDraw = sourceDraws[drawIndex];
      const PushConstants &sourcePc = sourceConstants[drawIndex];
      const uint32_t materialKey =
          sourceDraw.alphaMasked ? sourcePc.materialIndex : 0u;
      const ShadowIndirectGroupKey key{
          .pipeline = sourceDraw.pipeline,
          .indexBuffer = sourceDraw.indexBuffer,
          .indexFormat = sourceDraw.indexFormat,
          .materialIndex = materialKey,
      };
      auto groupIt = groupLookup.find(key);
      if (groupIt == groupLookup.end()) {
        const uint32_t groupIndex = saturateToU32(groups.size());
        groups.emplace_back(batchMemory);
        IndirectGroup &group = groups.back();
        group.baseDraw = sourceDraw;
        group.baseConstants = sourcePc;
        group.commands.reserve(16u);
        group.metadata.reserve(16u);
        groupIt = groupLookup.emplace(key, groupIndex).first;
      }
      IndirectGroup &group = groups[groupIt->second];
      const uint32_t indexStride = sourceDraw.indexFormat == IndexFormat::U16
                                       ? sizeof(uint16_t)
                                       : sizeof(uint32_t);
      const uint64_t firstIndex = static_cast<uint64_t>(sourceDraw.firstIndex) +
                                  sourceDraw.indexBufferOffset / indexStride;
      group.commands.push_back(DrawIndexedIndirectCommand{
          .indexCount = sourceDraw.indexCount,
          .instanceCount = sourceDraw.instanceCount,
          .firstIndex = static_cast<uint32_t>(firstIndex),
          .vertexOffset = sourceDraw.vertexOffset,
          .firstInstance = sourceDraw.firstInstance,
      });
      group.metadata.push_back(ShadowDrawGpuData{
          .vertexBufferAddress = sourcePc.vertexBufferAddress,
          .vertexDecodeBufferAddress = sourcePc.vertexDecodeBufferAddress,
          .vertexDecodeIndex = sourcePc.vertexDecodeIndex,
          .packedVertexFormat = sourcePc.packedVertexFormat,
      });
    }
    indirectConstants.reserve(groups.size());
    indirectDraws.reserve(groups.size());
    for (const IndirectGroup &group : groups) {
      if (group.commands.empty()) {
        continue;
      }
      size_t commandCursor = 0u;
      while (commandCursor < group.commands.size()) {
        const size_t chunkCount =
            std::min<size_t>(group.commands.size() - commandCursor,
                             kMaxShadowIndirectCommandsPerDraw);
        const size_t commandOffset = shadowDrawPacketUploadBytes_.size();
        const std::span<const std::byte> commandBytes =
            std::as_bytes(std::span<const DrawIndexedIndirectCommand>(
                group.commands.data() + commandCursor, chunkCount));
        shadowDrawPacketUploadBytes_.insert(shadowDrawPacketUploadBytes_.end(),
                                            commandBytes.begin(),
                                            commandBytes.end());
        const size_t alignedMetadataOffset =
            (shadowDrawPacketUploadBytes_.size() + 15u) & ~size_t{15u};
        shadowDrawPacketUploadBytes_.resize(alignedMetadataOffset);
        const size_t metadataOffset = shadowDrawPacketUploadBytes_.size();
        const std::span<const std::byte> metadataBytes =
            std::as_bytes(std::span<const ShadowDrawGpuData>(
                group.metadata.data() + commandCursor, chunkCount));
        shadowDrawPacketUploadBytes_.insert(shadowDrawPacketUploadBytes_.end(),
                                            metadataBytes.begin(),
                                            metadataBytes.end());
        PushConstants &indirectPc = indirectConstants.emplace_back();
        indirectPc = group.baseConstants;
        indirectPc.shadowDrawMetadataAddress = metadataOffset;
        DrawItem &indirectDraw = indirectDraws.emplace_back(group.baseDraw);
        indirectDraw.command = DrawCommandType::IndexedIndirect;
        indirectDraw.indexBufferOffset = 0u;
        indirectDraw.indirectBufferOffset = commandOffset;
        indirectDraw.indirectDrawCount = saturateToU32(chunkCount);
        indirectDraw.indirectStride = sizeof(DrawIndexedIndirectCommand);
        commandCursor += chunkCount;
      }
    }
  }
  if (!shadowDrawPacketUploadBytes_.empty()) {
    auto packetCapacityResult =
        ensureShadowDrawPacketRingCapacity(shadowDrawPacketUploadBytes_.size());
    if (packetCapacityResult.hasError()) {
      return packetCapacityResult;
    }
    const BufferHandle packetBuffer =
        bufferRings_[ShadowDrawPacketRing][frameSlot].buffer->handle();
    shadowDrawPacketBuffer = packetBuffer;
    const uint64_t packetAddress = gpu_.getBufferDeviceAddress(packetBuffer);
    if (packetAddress == 0u) {
      return Result<bool, std::string>::makeError(
          "ShadowRenderer: invalid indirect packet address");
    }
    const std::span<const std::byte> packetBytes{
        shadowDrawPacketUploadBytes_.data(),
        shadowDrawPacketUploadBytes_.size()};
    const uint64_t packetSignature = hashBytes(packetBytes);
    if (frameSlotStates_[frameSlot].drawPacketUploadSignature !=
        packetSignature) {
      auto uploadResult = gpu_.updateBuffer(packetBuffer, packetBytes, 0u);
      if (uploadResult.hasError()) {
        return uploadResult;
      }
      frameSlotStates_[frameSlot].drawPacketUploadSignature = packetSignature;
    }
    appendUniqueBufferHandle(preResolvedDrawBuffers_, packetBuffer);
    for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
         ++cascadeIndex) {
      auto &indirectConstants = cascadeIndirectPushConstants_[cascadeIndex];
      auto &indirectDraws = cascadeIndirectDrawItems_[cascadeIndex];
      for (size_t drawIndex = 0u; drawIndex < indirectDraws.size();
           ++drawIndex) {
        indirectConstants[drawIndex].shadowDrawMetadataAddress += packetAddress;
        indirectDraws[drawIndex].indirectBuffer = packetBuffer;
        indirectDraws[drawIndex].pushConstants = std::span<const std::byte>(
            reinterpret_cast<const std::byte *>(&indirectConstants[drawIndex]),
            sizeof(PushConstants));
      }
    }
  }
  if (!instanceRemap_.empty()) {
    const std::span<const std::byte> remapBytes{
        reinterpret_cast<const std::byte *>(instanceRemap_.data()),
        instanceRemap_.size() * sizeof(uint32_t)};
    const uint64_t remapSignature = hashBytes(remapBytes);
    if (frameSlotStates_[frameSlot].remapUploadSignature != remapSignature) {
      auto updateRemapResult =
          gpu_.updateBuffer(instanceRemapBuffer, remapBytes, 0u);
      if (updateRemapResult.hasError()) {
        return updateRemapResult;
      }
      frameSlotStates_[frameSlot].remapUploadSignature = remapSignature;
    }
  } else {
    frameSlotStates_[frameSlot].remapUploadSignature =
        std::numeric_limits<uint64_t>::max();
  }
  passBufferDependencies_.clear();
  appendUniqueBufferDependency(passBufferDependencies_, sceneGpu.buffer);
  appendUniqueBufferDependency(passBufferDependencies_, instanceMatricesBuffer);
  appendUniqueBufferDependency(passBufferDependencies_, instanceRemapBuffer);
  appendUniqueBufferDependency(passBufferDependencies_, shadowDrawPacketBuffer);
  appendAnimatedGeometryDependencies(passBufferDependencies_,
                                     animationSceneData);
  if (frame.sharedResources.shadowFrameGpuData.has_value()) {
    appendUniqueBufferDependency(
        passBufferDependencies_,
        frame.sharedResources.shadowFrameGpuData->buffer);
  }
  if (frame.sharedResources.materialTableGpuData.has_value()) {
    appendUniqueBufferDependency(
        passBufferDependencies_,
        frame.sharedResources.materialTableGpuData->headerBuffer);
  }
  for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
       ++cascadeIndex) {
    const uint32_t staticDrawCount = cascadeStaticCasterCounts[cascadeIndex];
    const uint32_t dynamicDrawCount = cascadeDynamicCasterCounts[cascadeIndex];
    cascadeStates_[cascadeIndex].drawCount = staticDrawCount + dynamicDrawCount;
    cascadeStates_[cascadeIndex].dynamicDrawCount = dynamicDrawCount;
    currentStaticOnlyCascadeStates[cascadeIndex].staticDrawCount =
        staticDrawCount;
    currentStaticOnlyCascadeStates[cascadeIndex].dynamicDrawCount =
        dynamicDrawCount;
    ShadowCascadeDebugFrameData &cascadeDebug =
        shadowDebugFrameData_.cascades[cascadeIndex];
    cascadeDebug.drawCount = cascadeStates_[cascadeIndex].drawCount;
    cascadeDebug.culledCount = cascadeStates_[cascadeIndex].culledCount;
    cascadeDebug.staticDrawCount = staticDrawCount;
    cascadeDebug.dynamicDrawCount = dynamicDrawCount;
    cascadeDebug.staticBatchFullEmitCount =
        cascadeStaticBatchFullEmitCounts[cascadeIndex];
    cascadeDebug.staticLightGridQueryCellCount =
        cascadeStaticLightGridQueryCellCounts[cascadeIndex];
    cascadeDebug.staticLightGridCandidateCount =
        cascadeStaticLightGridCandidateCounts[cascadeIndex];
    cascadeDebug.staticLightGridFallbackScanCount =
        cascadeStaticLightGridFallbackScanCounts[cascadeIndex];
  }
  uint32_t totalDraws = 0u;
  uint32_t totalCulledDraws = 0u;
  uint32_t totalStaticBatchFullEmitCount = 0u;
  uint32_t totalStaticLightGridFallbackScanCount = 0u;
  uint32_t totalStaticLightGridQueryCellCount = 0u;
  uint32_t totalStaticLightGridCandidateCount = 0u;
  uint32_t staticOnlyCandidateCount = 0u;
  uint32_t reusedStaticOnlyCascadeCount = 0u;
  uint32_t staticOnlyReuseMissStaticCacheRebuiltCount = 0u;
  uint32_t staticOnlyReuseMissDynamicCasterCount = 0u;
  uint32_t staticOnlyReuseMissNoPreviousCount = 0u;
  uint32_t staticOnlyReuseMissRasterStateChangedCount = 0u;
  uint32_t staticOnlyReuseMissAdaptiveRefreshCount = 0u;
  uint32_t staticOnlyAdaptiveRefreshesThisFrame = 0u;
  uint64_t actualTotalIndexCountEstimate = 0u;
  for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
       ++cascadeIndex) {
    ShadowCascadeDebugFrameData &cascadeDebug =
        shadowDebugFrameData_.cascades[cascadeIndex];
    StaticOnlyCascadeReuseState &currentState =
        currentStaticOnlyCascadeStates[cascadeIndex];
    const StaticOnlyCascadeReuseState &previousState =
        cascadeStates_[cascadeIndex].reusable;
    const bool staticOnlyCandidate =
        currentState.dynamicDrawCount == 0u &&
        cascadeHasGuardedDynamicDraw[cascadeIndex] == 0u;
    const bool previousValid = cascadeStates_[cascadeIndex].reusableValid;
    const TextureHandle currentShadowDepthTexture =
        shadowDepthTextures_[cascadeIndex];
    const bool previousTextureMatches =
        previousValid &&
        currentShadowDepthTexture.index ==
            previousState.shadowDepthTexture.index &&
        currentShadowDepthTexture.generation ==
            previousState.shadowDepthTexture.generation;
    const bool lightViewProjChanged =
        previousValid && previousState.lightViewProjSignature !=
                             currentState.lightViewProjSignature;
    const bool biasChanged = previousValid && previousState.biasSignature !=
                                                  currentState.biasSignature;
    const bool casterSignatureChanged =
        previousValid &&
        previousState.casterSignature != currentState.casterSignature;
    const bool cachedRenderedFitContainsCurrent =
        previousValid &&
        staticOnlyRenderedFitContainsCurrent(
            previousState.renderedFit, currentRawShadowFits_[cascadeIndex]);
    const bool needsAdaptiveRefresh =
        cachedRenderedFitContainsCurrent &&
        staticOnlyNeedsAdaptiveRefresh(
            previousState.renderedFit, currentRawShadowFits_[cascadeIndex],
            previousState.rawFit, previousValid, settings.shadow.shadowMapSize);
    const StaticOnlyGuardBandTexels rawMotion =
        staticOnlyRawMotionTexels(currentRawShadowFits_[cascadeIndex],
                                  previousState.rawFit, previousValid);
    const bool motionFitsAdaptiveBudget =
        previousValid &&
        rawMotion.ortho <= staticOnlyAdaptiveGuardBandMaxTexels(
                               settings.shadow.shadowMapSize) +
                               kStaticOnlyPredictiveMotionEpsilonTexels &&
        rawMotion.depth <= staticOnlyAdaptiveDepthGuardBandMaxTexels(
                               settings.shadow.shadowMapSize) +
                               kStaticOnlyPredictiveMotionEpsilonTexels;
    const bool adaptiveRefresh =
        needsAdaptiveRefresh ||
        (!cachedRenderedFitContainsCurrent && staticOnlyCandidate &&
         previousValid && !biasChanged && !casterSignatureChanged &&
         motionFitsAdaptiveBudget);
    const bool staticCacheCompatibleForTextureReuse =
        staticCacheReused ||
        (staticCacheRebuiltForTransformsOnly && !casterSignatureChanged);
    const bool canReuseCurrentStaticOnlyFit =
        staticCacheCompatibleForTextureReuse && staticOnlyCandidate &&
        previousValid && previousTextureMatches && !biasChanged &&
        cachedRenderedFitContainsCurrent;
    const bool deferAdaptiveRefresh =
        canReuseCurrentStaticOnlyFit && needsAdaptiveRefresh &&
        staticOnlyAdaptiveRefreshesThisFrame >=
            kStaticOnlyAdaptiveRefreshBudgetPerFrame;
    cascadeStates_[cascadeIndex].reuse =
        canReuseCurrentStaticOnlyFit &&
        (!needsAdaptiveRefresh || deferAdaptiveRefresh);
    if (!cascadeStates_[cascadeIndex].reuse && canReuseCurrentStaticOnlyFit &&
        needsAdaptiveRefresh) {
      ++staticOnlyAdaptiveRefreshesThisFrame;
    }
    if (cascadeStates_[cascadeIndex].reuse) {
      const shadow_detail::DirectionalShadowFit rawFit =
          currentRawShadowFits_[cascadeIndex];
      const shadow_detail::DirectionalShadowFit reuseFit =
          makeStaticOnlyReuseFitForCurrentFrame(previousState.renderedFit,
                                                rawFit);
      writeShadowCascadeFit(reuseFit, cascadeIndex,
                            shadowDepthTextures_[cascadeIndex], gpu_,
                            shadowFrameCpuData_, shadowDebugFrameData_);
      currentState = previousState;
      currentState.renderedFit = reuseFit;
      currentState.rawFit = rawFit;
      cascadeStates_[cascadeIndex].contentSignature =
          previousState.rasterSignature;
    } else {
      cascadeStates_[cascadeIndex].contentSignature =
          currentState.rasterSignature;
    }
    cascadeDebug.staticOnlyReuseCandidate = staticOnlyCandidate;
    cascadeDebug.staticOnlyReusePreviousValid = previousValid;
    cascadeDebug.staticOnlyReuseLightViewProjChanged = lightViewProjChanged;
    cascadeDebug.staticOnlyReuseBiasChanged = biasChanged;
    cascadeDebug.staticOnlyReuseCasterSignatureChanged = casterSignatureChanged;
    cascadeDebug.staticOnlyReuseAdaptiveRefresh = adaptiveRefresh;
    cascadeDebug.currentStaticOnlyRasterSignature =
        currentState.rasterSignature;
    cascadeDebug.previousStaticOnlyRasterSignature =
        previousState.rasterSignature;
    cascadeDebug.currentStaticOnlyLightViewProjSignature =
        currentState.lightViewProjSignature;
    cascadeDebug.previousStaticOnlyLightViewProjSignature =
        previousState.lightViewProjSignature;
    cascadeDebug.currentStaticOnlyBiasSignature = currentState.biasSignature;
    cascadeDebug.previousStaticOnlyBiasSignature = previousState.biasSignature;
    cascadeDebug.currentStaticOnlyCasterSignature =
        currentState.casterSignature;
    cascadeDebug.previousStaticOnlyCasterSignature =
        previousState.casterSignature;
    cascadeDebug.staticDrawCount = currentState.staticDrawCount;
    cascadeDebug.dynamicDrawCount = currentState.dynamicDrawCount;
    if (cascadeStates_[cascadeIndex].reuse) {
      cascadeStates_[cascadeIndex].drawCount = 0u;
      cascadeDebug.drawCount = 0u;
      cascadeDebug.staticOnlyReuseStatus =
          ShadowCascadeDebugFrameData::StaticOnlyReuseStatus::Reused;
    } else {
      actualTotalIndexCountEstimate +=
          cascadeStates_[cascadeIndex].indexCountEstimate;
      if (!staticCacheReused) {
        cascadeDebug.staticOnlyReuseStatus = ShadowCascadeDebugFrameData::
            StaticOnlyReuseStatus::StaticCacheRebuilt;
        ++staticOnlyReuseMissStaticCacheRebuiltCount;
      } else if (!staticOnlyCandidate) {
        cascadeDebug.staticOnlyReuseStatus = ShadowCascadeDebugFrameData::
            StaticOnlyReuseStatus::HasDynamicCasters;
        ++staticOnlyReuseMissDynamicCasterCount;
      } else if (!previousValid) {
        cascadeDebug.staticOnlyReuseStatus = ShadowCascadeDebugFrameData::
            StaticOnlyReuseStatus::NoPreviousStaticOnlyPass;
        ++staticOnlyReuseMissNoPreviousCount;
      } else if (adaptiveRefresh) {
        cascadeDebug.staticOnlyReuseStatus =
            ShadowCascadeDebugFrameData::StaticOnlyReuseStatus::AdaptiveRefresh;
        ++staticOnlyReuseMissAdaptiveRefreshCount;
      } else {
        cascadeDebug.staticOnlyReuseStatus = ShadowCascadeDebugFrameData::
            StaticOnlyReuseStatus::RasterStateChanged;
        ++staticOnlyReuseMissRasterStateChangedCount;
      }
    }
    staticOnlyCandidateCount += staticOnlyCandidate ? 1u : 0u;
    reusedStaticOnlyCascadeCount +=
        cascadeStates_[cascadeIndex].reuse ? 1u : 0u;
    totalDraws += cascadeStates_[cascadeIndex].drawCount;
    totalCulledDraws += cascadeStates_[cascadeIndex].culledCount;
    totalStaticBatchFullEmitCount +=
        cascadeStaticBatchFullEmitCounts[cascadeIndex];
    totalStaticLightGridFallbackScanCount +=
        cascadeStaticLightGridFallbackScanCounts[cascadeIndex];
    totalStaticLightGridQueryCellCount +=
        cascadeStaticLightGridQueryCellCounts[cascadeIndex];
    totalStaticLightGridCandidateCount +=
        cascadeStaticLightGridCandidateCounts[cascadeIndex];
  }
  for (uint32_t cascadeIndex = cascadeCount; cascadeIndex < kMaxShadowCascades;
       ++cascadeIndex) {
    cascadeStates_[cascadeIndex].reuse = false;
    cascadeStates_[cascadeIndex].contentSignature = 0u;
    shadowDebugFrameData_.cascades[cascadeIndex].staticOnlyReuseStatus =
        ShadowCascadeDebugFrameData::StaticOnlyReuseStatus::None;
  }
  float minCascadeTexelWorldSize = std::numeric_limits<float>::max();
  float maxCascadeTexelWorldSize = 0.0f;
  float totalCascadeTexelWorldSize = 0.0f;
  uint32_t validCascadeTexelWorldSizeCount = 0u;
  for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
       ++cascadeIndex) {
    const float texelWorldSize =
        shadowDebugFrameData_.cascades[cascadeIndex].texelWorldSize;
    if (!std::isfinite(texelWorldSize) || texelWorldSize <= 0.0f) {
      continue;
    }
    minCascadeTexelWorldSize =
        std::min(minCascadeTexelWorldSize, texelWorldSize);
    maxCascadeTexelWorldSize =
        std::max(maxCascadeTexelWorldSize, texelWorldSize);
    totalCascadeTexelWorldSize += texelWorldSize;
    ++validCascadeTexelWorldSizeCount;
  }
  if (validCascadeTexelWorldSizeCount == 0u) {
    minCascadeTexelWorldSize = 0.0f;
  }
  frame.metrics.shadow.cascadeCount = cascadeCount;
  frame.metrics.shadow.shadowMapSize = shadowMapSize_;
  frame.metrics.shadow.totalDraws = totalDraws;
  frame.metrics.shadow.totalCulledDraws = totalCulledDraws;
  frame.metrics.visibility.shadowCpuRejected = totalCulledDraws;
  frame.metrics.visibility.shadowCpuCandidates = saturateToU32(
      static_cast<size_t>(totalDraws) + static_cast<size_t>(totalCulledDraws));
  frame.metrics.shadow.totalIndexCountEstimate =
      saturateToU32(actualTotalIndexCountEstimate);
  frame.metrics.shadow.staticCasterEntries =
      saturateToU32(staticShadowCasterCache_.size()) -
      std::min(saturateToU32(staticShadowCasterCache_.size()),
               overriddenStaticTemplateCount);
  frame.metrics.shadow.dynamicCasterEntries =
      saturateToU32(frameDynamicTemplateIndices.size());
  frame.metrics.shadow.staticCacheReused = staticCacheReused ? 1u : 0u;
  frame.metrics.shadow.staticBatchTemplateCount =
      saturateToU32(staticShadowBatchTemplates_.size());
  frame.metrics.shadow.shadowBatchEntryCount = emittedShadowBatchEntryCount;
  frame.metrics.shadow.shadowInstanceRemapCount =
      emittedShadowInstanceRemapCount;
  uint32_t submittedDrawItemCount = 0u;
  for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
       ++cascadeIndex) {
    submittedDrawItemCount +=
        saturateToU32(cascadeIndirectDrawItems_[cascadeIndex].size());
  }
  frame.metrics.shadow.submittedDrawItemCount = submittedDrawItemCount;
  frame.metrics.shadow.indirectCommandCount = emittedShadowBatchEntryCount;
  frame.metrics.shadow.drawPacketBytes =
      saturateToU32(shadowDrawPacketUploadBytes_.size());
  frame.metrics.shadow.staticBatchFullEmitCount = totalStaticBatchFullEmitCount;
  frame.metrics.shadow.staticLightGridQueryCount = staticLightGridQueryCount;
  frame.metrics.shadow.staticLightGridFallbackScanCount =
      totalStaticLightGridFallbackScanCount;
  frame.metrics.shadow.staticLightGridQueryCellCount =
      totalStaticLightGridQueryCellCount;
  frame.metrics.shadow.staticLightGridCandidateCount =
      totalStaticLightGridCandidateCount;
  frame.metrics.shadow.staticOnlyCandidateCount = staticOnlyCandidateCount;
  frame.metrics.shadow.reusedStaticOnlyCascadeCount =
      reusedStaticOnlyCascadeCount;
  frame.metrics.shadow.staticOnlyReuseMissStaticCacheRebuiltCount =
      staticOnlyReuseMissStaticCacheRebuiltCount;
  frame.metrics.shadow.staticOnlyReuseMissDynamicCasterCount =
      staticOnlyReuseMissDynamicCasterCount;
  frame.metrics.shadow.staticOnlyReuseMissNoPreviousCount =
      staticOnlyReuseMissNoPreviousCount;
  frame.metrics.shadow.staticOnlyReuseMissRasterStateChangedCount =
      staticOnlyReuseMissRasterStateChangedCount;
  frame.metrics.shadow.staticOnlyReuseMissAdaptiveRefreshCount =
      staticOnlyReuseMissAdaptiveRefreshCount;
  frame.metrics.shadow.cascadeTextureBytes =
      static_cast<uint64_t>(cascadeCount) * shadowMapSize_ * shadowMapSize_ *
      shadowDepthTextureBytesPerPixel(shadowDepthPipelineFormat_);
  frame.metrics.shadow.minCascadeTexelWorldSize = minCascadeTexelWorldSize;
  frame.metrics.shadow.maxCascadeTexelWorldSize = maxCascadeTexelWorldSize;
  frame.metrics.shadow.averageCascadeTexelWorldSize =
      validCascadeTexelWorldSizeCount > 0u
          ? totalCascadeTexelWorldSize /
                static_cast<float>(validCascadeTexelWorldSizeCount)
          : 0.0f;
  frame.metrics.shadow.farCascadeTexelWorldSize =
      cascadeCount > 0u
          ? shadowDebugFrameData_.cascades[cascadeCount - 1u].texelWorldSize
          : 0.0f;
  frame.sharedResources.shadowDebugFrameData = shadowDebugFrameData_;
  emitShadowDiagnostics();
  return Result<bool, std::string>::makeResult(true);
}

uint64_t ShadowRenderer::shadowPipelineSignature() const noexcept {
  return hashBytes(std::as_bytes(std::span(shadowPipelines_)));
}

void ShadowRenderer::invalidateStaticShadowCasterCache() noexcept {
  staticShadowCasterCache_.clear();
  staticShadowBatchTemplates_.clear();
  staticShadowBatchIndexMap_.clear();
  staticShadowBatchInstanceIndices_.clear();
  staticShadowCasterDrawBuffers_.clear();
  staticShadowCasterFitPoints_.clear();
  for (auto &cells : staticLightGridCells_)
    cells.clear();
  for (auto &entries : staticLightGridEntries_)
    entries.clear();
  for (auto &entries : staticLargeLightGridEntries_)
    entries.clear();
  staticShadowCasterLightGridQueryMarks_.clear();
  staticShadowCasterLightGridQueryEntries_.clear();
  staticShadowCasterLightGridQueryMarker_ = 1u;
  staticLightGrids_ = {};
  staticShadowCasterBoundsMin_ = glm::vec3(std::numeric_limits<float>::max());
  staticShadowCasterBoundsMax_ =
      glm::vec3(std::numeric_limits<float>::lowest());
  staticShadowCasterLightSpaceBoundsMin_ =
      glm::vec3(std::numeric_limits<float>::max());
  staticShadowCasterLightSpaceBoundsMax_ =
      glm::vec3(std::numeric_limits<float>::lowest());
  staticShadowCasterCacheContentSignature_ = 0u;
  staticShadowCasterCacheIndexCountEstimate_ = 0u;
  hasStaticShadowCasterBounds_ = false;
  hasStaticShadowCasterLightDepthBounds_ = false;
  hasStaticShadowCasterLightSpaceBounds_ = false;
  staticShadowCasterCacheTransformVersion_ =
      std::numeric_limits<uint64_t>::max();
  staticShadowCasterCachePipelineSignature_ = 0u;
  staticShadowCasterCacheForcedMeshLod_ = std::numeric_limits<int32_t>::min();
  staticShadowCasterCacheValid_ = false;
}

void ShadowRenderer::invalidateReusableStaticOnlyCascadeCache() noexcept {
  for (CascadeState &cascade : cascadeStates_) {
    cascade.reusableContentSignature = 0u;
    cascade.reusable = {};
    cascade.reusableValid = false;
  }
}

void ShadowRenderer::destroyShadowResources() {
  if (nuri::isValid(shadowDebugPreviewTexture_)) {
    gpu_.destroyTexture(shadowDebugPreviewTexture_);
    shadowDebugPreviewTexture_ = {};
  }
  for (TextureHandle &texture : shadowDepthTextures_) {
    if (nuri::isValid(texture)) {
      gpu_.destroyTexture(texture);
      texture = {};
    }
  }
  if (nuri::isValid(rawDepthSampler_)) {
    gpu_.destroySampler(rawDepthSampler_);
    rawDepthSampler_ = {};
  }
  if (nuri::isValid(compareDepthSampler_)) {
    gpu_.destroySampler(compareDepthSampler_);
    compareDepthSampler_ = {};
  }
  shadowMapSize_ = 0u;
  activeCascadeCount_ = 0u;
  invalidateReusableStaticOnlyCascadeCache();
  resetSdsmState();
  resetFrozenShadowFit();
  resetCascadeStabilizationHistory();
}

void ShadowRenderer::destroyBuffers() {
  for (auto &ring : bufferRings_) {
    retireDynamicBufferRing(ring);
    ring.clear();
  }
  frameSlotStates_.clear();
}

void ShadowRenderer::destroyShaders() {
  for (ShaderHandle &shader : shaders_) {
    if (nuri::isValid(shader)) {
      gpu_.destroyShaderModule(shader);
      shader = {};
    }
  }
  initialized_ = false;
}

void ShadowRenderer::destroyShadowDepthPipelineState() {
  for (RenderPipelineHandle &pipeline : shadowPipelines_) {
    if (nuri::isValid(pipeline)) {
      gpu_.destroyRenderPipeline(pipeline);
      pipeline = {};
    }
  }
  shadowDepthPipelineFormat_ = Format::Count;
  shadowPipelineRasterState_ = {};
}

Result<bool, std::string>
ShadowRenderer::ensureShadowDrawPacketRingCapacity(size_t requiredBytes) {
  return ensureDynamicBufferRingCapacity(
      gpu_, bufferRings_[ShadowDrawPacketRing],
      BufferDesc{.usage = BufferUsage::Storage | BufferUsage::Indirect,
                 .storage = Storage::HostVisible,
                 .size = std::max(requiredBytes, sizeof(uint32_t))},
      "shadow_draw_packet", [this](size_t i) {
        frameSlotStates_[i].drawPacketUploadSignature =
            std::numeric_limits<uint64_t>::max();
      });
}

void ShadowRenderer::destroyPipelineState() {
  if (nuri::isValid(previewPipelineHandle_)) {
    gpu_.destroyRenderPipeline(previewPipelineHandle_);
    previewPipelineHandle_ = {};
  }
  destroyShadowDepthPipelineState();
  if (nuri::isValid(sdsmReducePipelineHandle_)) {
    gpu_.destroyComputePipeline(sdsmReducePipelineHandle_);
    sdsmReducePipelineHandle_ = {};
  }
  initialized_ = false;
}

void ShadowRenderer::resetCachedState() {
  cachedScene_ = nullptr;
  cachedTopologyVersion_ = std::numeric_limits<uint64_t>::max();
  cachedMaterialVersion_ = std::numeric_limits<uint64_t>::max();
  cachedModelMaterialBindingVersion_ = std::numeric_limits<uint64_t>::max();
  cachedTransformVersion_ = std::numeric_limits<uint64_t>::max();
  cachedDeformationVersion_ = std::numeric_limits<uint64_t>::max();
  cachedGeometryMutationVersion_ = std::numeric_limits<uint64_t>::max();
  clearAll(meshDrawTemplates_, staticShadowTemplateIndices_,
           dynamicShadowTemplateIndices_, staticShadowCasterDrawBuffers_);
  invalidateStaticShadowCasterCache();
  invalidateReusableStaticOnlyCascadeCache();
  resetCascadeStabilizationHistory();
  clearAll(instanceMatrices_, instanceRemap_, passTextureDependencies_,
           passDependencyBuffers_, passDependencyBufferAccessModes_,
           passDependencyTextures_, passDependencyTextureAccessModes_,
           preResolvedDrawBuffers_);
}

void ShadowRenderer::resetFrameBuildState() {
  hasPreparedShadowDepthPasses_ = false;
  hasPreparedShadowPreviewPass_ = false;
  hasActiveShadowLightForFrame_ = false;
  for (CascadeState &cascade : cascadeStates_) {
    cascade.drawCount = 0u;
    cascade.culledCount = 0u;
    cascade.dynamicDrawCount = 0u;
    cascade.indexCountEstimate = 0u;
    cascade.contentSignature = 0u;
    cascade.reuse = false;
  }
  currentRawShadowFits_ = {};
  clearEach(cascadePushConstants_);
  clearEach(cascadeDrawItems_);
  clearEach(cascadeIndirectPushConstants_);
  clearEach(cascadeIndirectDrawItems_);
  clearAll(passBufferDependencies_, passDependencyBuffers_,
           passDependencyBufferAccessModes_, passDependencyTextures_,
           passDependencyTextureAccessModes_, preResolvedDrawBuffers_,
           previewTextureDependencies_);
  previewPushConstants_ = {};
  previewDraw_ = {};
}

void ShadowRenderer::resetFrozenShadowFit() {
  hasFrozenShadowFit_ = false;
  frozenShadowLightId_ = kInvalidLightId;
  frozenShadowMapSize_ = 0u;
  frozenCascadeCount_ = 0u;
  frozenShadowFits_ = {};
}

void ShadowRenderer::resetCascadeStabilizationHistory() {
  cascadeStabilizationHistory_ = {};
}

void ShadowRenderer::resetSdsmState() { sdsmState_ = {}; }

Result<bool, std::string>
ShadowRenderer::publishFrameData(RenderFrameContext &frame) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_COPY);
  frame.sharedResources.shadowCascadeTextures = {};
  frame.sharedResources.shadowCascadeGraphTextures = {};
  frame.sharedResources.shadowDebugPreviewTexture = {};
  frame.sharedResources.shadowDebugPreviewGraphTexture = {};
  frame.sharedResources.shadowFrameGpuData.reset();
  frame.sharedResources.shadowSdsmGpuReduceTarget.reset();
  frame.sharedResources.shadowSdsmGpuReducePipeline = {};
  frame.sharedResources.shadowRawSamplerId = kInvalidShadowBindlessIndex;
  frame.sharedResources.shadowCompareSamplerId = kInvalidShadowBindlessIndex;
  frame.sharedResources.shadowDebugFrameData.reset();
  frame.metrics.shadow = {};
  const RenderSettings &frameSettings = renderSettingsOrDefault(frame);
  RenderSettings::ShadowSettings settings = frameSettings.shadow;
  sanitizeShadowSettings(settings);
  if (settings.enabled && !config_.shaderBasePath.empty()) {
    auto initResult = ensureInitialized();
    if (initResult.hasError()) {
      return initResult;
    }
  }
  if (settings.enabled && initialized_) {
    frame.metrics.shadow.filterSampleBudget =
        shadowFilterSampleBudget(settings.pcfSampleCount);
    const GpuTimingReport &timingReport = frame.gpuTiming;
    if (hasGpuTimingScope(timingReport, GpuTimingScope::Shadow)) {
      frame.metrics.shadow.gpuTimeMs = timingReport.shadowTimeMs;
      frame.metrics.shadow.gpuTimingSourceFrameIndex =
          timingReport.shadowSourceFrameIndex;
      frame.metrics.shadow.gpuTimingAvailable = 1u;
    }
    if (hasGpuTimingScope(timingReport, GpuTimingScope::ShadowDepth)) {
      frame.metrics.shadow.depthGpuTimeMs = timingReport.shadowDepthTimeMs;
      frame.metrics.shadow.depthGpuTimingSourceFrameIndex =
          timingReport.shadowDepthSourceFrameIndex;
      frame.metrics.shadow.depthGpuTimingAvailable = 1u;
    }
    if (hasGpuTimingScope(timingReport, GpuTimingScope::ShadowSdsm)) {
      frame.metrics.shadow.sdsmGpuTimeMs = timingReport.shadowSdsmTimeMs;
      frame.metrics.shadow.sdsmGpuTimingSourceFrameIndex =
          timingReport.shadowSdsmSourceFrameIndex;
      frame.metrics.shadow.sdsmGpuTimingAvailable = 1u;
    }
  }
  auto ensureTextureResult = ensureShadowResources(settings);
  if (ensureTextureResult.hasError()) {
    return ensureTextureResult;
  }
  if (settings.enabled && nuri::isValid(sdsmReducePipelineHandle_)) {
    const Format depthFormat = sanitizeShadowDepthFormat(settings.depthFormat);
    const RasterPipelineState rasterState = makeRasterPipelineState(
        DepthState{.compareOp = CompareOp::Less, .isDepthWriteEnabled = true},
        true, settings.constantBias, settings.slopeBias);
    if (shadowDepthPipelineFormat_ != depthFormat ||
        shadowPipelineRasterState_ != rasterState ||
        std::ranges::any_of(shadowPipelines_,
                            [](RenderPipelineHandle pipeline) {
                              return !nuri::isValid(pipeline);
                            })) {
      auto pipelineResult = createPipelines(depthFormat, rasterState);
      if (pipelineResult.hasError()) {
        return pipelineResult;
      }
    }
  }
  ensureRingBufferCount(std::max(2u, gpu_.getSwapchainImageCount() + 1u));
  auto shadowFrameResult =
      ensureShadowFrameRingCapacity(sizeof(ShadowFrameGpuData));
  if (shadowFrameResult.hasError()) {
    return shadowFrameResult;
  }
  const uint32_t rawSamplerId = gpu_.getSamplerBindlessIndex(rawDepthSampler_);
  const uint32_t compareSamplerId =
      gpu_.getSamplerBindlessIndex(compareDepthSampler_);
  const uint32_t frameSlot = static_cast<uint32_t>(
      frame.frameIndex % bufferRings_[ShadowFrameRing].size());
  const BufferHandle shadowFrameBuffer =
      bufferRings_[ShadowFrameRing][frameSlot].buffer->handle();
  const uint64_t shadowFrameAddress =
      gpu_.getBufferDeviceAddress(shadowFrameBuffer);
  for (uint32_t cascadeIndex = 0u; cascadeIndex < activeCascadeCount_;
       ++cascadeIndex) {
    frame.sharedResources.shadowCascadeTextures[cascadeIndex] =
        shadowDepthTextures_[cascadeIndex];
  }
  frame.sharedResources.shadowDebugPreviewTexture = shadowDebugPreviewTexture_;
  frame.sharedResources.shadowRawSamplerId = rawSamplerId;
  frame.sharedResources.shadowCompareSamplerId = compareSamplerId;
  frame.sharedResources.shadowFrameGpuData = ShadowFrameGpuDataHandle{
      .buffer = shadowFrameBuffer,
      .bufferAddress = shadowFrameAddress,
  };
  if (settings.enabled && nuri::isValid(sdsmReducePipelineHandle_)) {
    const auto ringCount = ensureSdsmReduceResultRingCount(std::max(
        kMinSdsmReduceResultRingCount, gpu_.getSwapchainImageCount() + 1u));
    const auto ring =
        ensureSdsmReduceResultRingCapacity(sizeof(SdsmGpuMinMaxResult));
    if (!ringCount.hasError() && !ring.hasError()) {
      const uint32_t slot = static_cast<uint32_t>(
          frame.frameIndex % bufferRings_[SdsmReduceResultRing].size());
      std::array<std::byte, sizeof(SdsmGpuMinMaxResult)> cleared{};
      const auto clear = gpu_.updateBuffer(
          bufferRings_[SdsmReduceResultRing][slot].buffer->handle(), cleared,
          0u);
      if (!clear.hasError()) {
        frameSlotStates_[slot].sdsmPublishedFrame = frame.frameIndex;
        const BufferHandle buffer =
            bufferRings_[SdsmReduceResultRing][slot].buffer->handle();
        frame.sharedResources.shadowSdsmGpuReduceTarget =
            ShadowSdsmGpuReduceTargetHandle{
                .buffer = buffer,
                .bufferAddress = gpu_.getBufferDeviceAddress(buffer)};
        frame.sharedResources.shadowSdsmGpuReducePipeline =
            sdsmReducePipelineHandle_;
      }
    }
  }
  ShadowDebugFrameData debug{};
  debug.cascadeCount = activeCascadeCount_;
  debug.rawSamplerId = rawSamplerId;
  debug.compareSamplerId = compareSamplerId;
  for (uint32_t cascadeIndex = 0u; cascadeIndex < activeCascadeCount_;
       ++cascadeIndex) {
    debug.cascades[cascadeIndex].texture = shadowDepthTextures_[cascadeIndex];
    debug.cascades[cascadeIndex].textureBindlessId =
        gpu_.getTextureBindlessIndex(shadowDepthTextures_[cascadeIndex]);
  }
  frame.sharedResources.shadowDebugFrameData = debug;
  if (!settings.enabled) {
    diagnosticLogState_ = {};
    invalidateReusableStaticOnlyCascadeCache();
    resetFrozenShadowFit();
    resetCascadeStabilizationHistory();
    resetSdsmState();
    shadowFrameCpuData_ = {};
    shadowFrameCpuData_.sharedSamplerMapSize =
        glm::uvec4(compareSamplerId, rawSamplerId, 0u, 0u);
    for (uint32_t cascadeIndex = 0u; cascadeIndex < activeCascadeCount_;
         ++cascadeIndex) {
      shadowFrameCpuData_.cascadeTextureIds[cascadeIndex] =
          gpu_.getTextureBindlessIndex(shadowDepthTextures_[cascadeIndex]);
    }
    const std::span<const std::byte> shadowFrameBytes = std::as_bytes(
        std::span<const ShadowFrameGpuData>(&shadowFrameCpuData_, 1u));
    const uint64_t shadowFrameSignature = hashBytes(shadowFrameBytes);
    if (frameSlotStates_[frameSlot].frameUploadSignature !=
        shadowFrameSignature) {
      auto updateResult =
          gpu_.updateBuffer(shadowFrameBuffer, shadowFrameBytes, 0u);
      if (updateResult.hasError()) {
        return updateResult;
      }
      frameSlotStates_[frameSlot].frameUploadSignature = shadowFrameSignature;
    }
    resetFrameBuildState();
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
ShadowRenderer::prepareShadowGraphPasses(RenderFrameContext &frame) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_COPY);
  resetFrameBuildState();
  frame.metrics.visibility.shadowCpuCandidates = 0u;
  frame.metrics.visibility.shadowCpuRejected = 0u;
  const RenderSettings &frameSettings = renderSettingsOrDefault(frame);
  RenderSettings::ShadowSettings settings = frameSettings.shadow;
  sanitizeShadowSettings(settings);
  if (!settings.enabled) {
    diagnosticLogState_ = {};
    invalidateReusableStaticOnlyCascadeCache();
    resetCascadeStabilizationHistory();
    return Result<bool, std::string>::makeResult(true);
  }
  auto ensureTextureResult = ensureShadowResources(settings);
  if (ensureTextureResult.hasError()) {
    return ensureTextureResult;
  }
  ensureRingBufferCount(std::max(2u, gpu_.getSwapchainImageCount() + 1u));
  auto shadowFrameResult =
      ensureShadowFrameRingCapacity(sizeof(ShadowFrameGpuData));
  if (shadowFrameResult.hasError()) {
    return shadowFrameResult;
  }
  const MaterialTableSnapshot materialSnapshot =
      frame.resources->materialSnapshot();
  const bool topologyDirty =
      cachedScene_ != frame.scene ||
      cachedTopologyVersion_ != frame.scene->topologyVersion() ||
      cachedMaterialVersion_ != materialSnapshot.version ||
      cachedModelMaterialBindingVersion_ !=
          frame.resources->modelMaterialBindingVersion() ||
      cachedDeformationVersion_ != frame.scene->deformationVersion() ||
      cachedGeometryMutationVersion_ != gpu_.geometryMutationVersion();
  if (topologyDirty) {
    rebuildSceneCache(*frame.sharedResources.sceneDrawDatabase);
    cachedScene_ = frame.scene;
    cachedTopologyVersion_ = frame.scene->topologyVersion();
    cachedMaterialVersion_ = materialSnapshot.version;
    cachedModelMaterialBindingVersion_ =
        frame.resources->modelMaterialBindingVersion();
    cachedDeformationVersion_ = frame.scene->deformationVersion();
    cachedGeometryMutationVersion_ = gpu_.geometryMutationVersion();
  }
  auto shadowFrameDataResult =
      updateShadowFrameData(frame, settings, settings.shadowMapSize,
                            frameSettings.opaque.forcedMeshLod);
  if (shadowFrameDataResult.hasError()) {
    return shadowFrameDataResult;
  }
  const uint32_t frameSlot = static_cast<uint32_t>(
      frame.frameIndex % bufferRings_[ShadowFrameRing].size());
  if (frame.sharedResources.forwardSceneGpuData.has_value()) {
    auto drawResult = buildShadowDraws(
        frame, frameSlot, *frame.sharedResources.forwardSceneGpuData);
    if (drawResult.hasError()) {
      return drawResult;
    }
  }
  const std::span<const std::byte> shadowFrameBytes = std::as_bytes(
      std::span<const ShadowFrameGpuData>(&shadowFrameCpuData_, 1u));
  const uint64_t shadowFrameSignature = hashBytes(shadowFrameBytes);
  if (frameSlotStates_[frameSlot].frameUploadSignature !=
      shadowFrameSignature) {
    auto updateResult = gpu_.updateBuffer(
        bufferRings_[ShadowFrameRing][frameSlot].buffer->handle(),
        shadowFrameBytes, 0u);
    if (updateResult.hasError()) {
      return updateResult;
    }
    frameSlotStates_[frameSlot].frameUploadSignature = shadowFrameSignature;
  }
  hasPreparedShadowDepthPasses_ = hasActiveShadowLightForFrame_ &&
                                  activeCascadeCount_ > 0u &&
                                  nuri::isValid(shadowDepthTextures_[0]);
  if (hasActiveShadowLightForFrame_ && activeCascadeCount_ > 0u &&
      settings.debug.showShadowMapViewport &&
      nuri::isValid(shadowDebugPreviewTexture_)) {
    const ShadowPreviewMode previewMode =
        sanitizeShadowPreviewMode(settings.debug.previewMode);
    const float previewDepthRange = std::max(settings.debug.previewDepthMax -
                                                 settings.debug.previewDepthMin,
                                             1.0e-6f);
    uint32_t previewFlags = 0u;
    if (settings.debug.previewDepthInvert) {
      previewFlags |= kShadowPreviewFlagInvert;
    }
    if (settings.debug.previewDepthLog) {
      previewFlags |= kShadowPreviewFlagLog;
    }
    if (previewMode == ShadowPreviewMode::TiledAllCascades) {
      previewFlags |= kShadowPreviewFlagTiled;
    }
    glm::uvec4 previewSourceTexIds{
        kInvalidShadowBindlessIndex, kInvalidShadowBindlessIndex,
        kInvalidShadowBindlessIndex, kInvalidShadowBindlessIndex};
    const uint32_t previewSourceCount =
        previewMode == ShadowPreviewMode::TiledAllCascades ? activeCascadeCount_
                                                           : 1u;
    previewTextureDependencies_.clear();
    for (uint32_t sourceIndex = 0u; sourceIndex < previewSourceCount;
         ++sourceIndex) {
      const uint32_t cascadeIndex =
          previewMode == ShadowPreviewMode::TiledAllCascades
              ? sourceIndex
              : std::min(settings.debug.debugCascadeIndex,
                         activeCascadeCount_ - 1u);
      const uint32_t sourceTexId =
          gpu_.getTextureBindlessIndex(shadowDepthTextures_[cascadeIndex]);
      previewSourceTexIds[sourceIndex] = sourceTexId;
      appendUniqueTextureDependency(previewTextureDependencies_,
                                    shadowDepthTextures_[cascadeIndex]);
    }
    const TextureDimensions previewDimensions =
        gpu_.getTextureDimensions(shadowDebugPreviewTexture_);
    previewPushConstants_ = PreviewPushConstants{
        .sourceTexIds = previewSourceTexIds,
        .previewParams =
            glm::uvec4(previewDimensions.width, previewDimensions.height,
                       previewSourceCount, previewFlags),
        .depthParams = glm::vec4(
            1.0f / previewDepthRange,
            -settings.debug.previewDepthMin / previewDepthRange, 0.0f, 0.0f)};
    previewDraw_ = DrawItem{};
    previewDraw_.pipeline = previewPipelineHandle_;
    previewDraw_.vertexCount = 3u;
    previewDraw_.instanceCount = 1u;
    previewDraw_.pushConstants = std::span<const std::byte>(
        reinterpret_cast<const std::byte *>(&previewPushConstants_),
        sizeof(PreviewPushConstants));
    previewDraw_.debugLabel = kShadowPreviewDrawLabel;
    previewDraw_.debugColor = kShadowPreviewPassDebugColor;
    hasPreparedShadowPreviewPass_ = true;
  }
  logShadowVisibilityCounters(frame);
  return Result<bool, std::string>::makeResult(true);
}

bool ShadowRenderer::hasPreparedShadowDepthPasses() const noexcept {
  return hasPreparedShadowDepthPasses_;
}

Result<bool, std::string>
ShadowRenderer::appendShadowDepthPasses(RenderFrameContext &frame,
                                        RenderGraphBuilder &graph) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_DRAW);
  const AnimationSceneFrameData *animationSceneData =
      resolveAnimationSceneFrameData(frame);
  passDependencyBuffers_.clear();
  passDependencyBufferAccessModes_.clear();
  passDependencyTextures_.clear();
  passDependencyTextureAccessModes_.clear();
  splitDependencies(
      std::span<const BufferDependency>(passBufferDependencies_.data(),
                                        passBufferDependencies_.size()),
      passDependencyBuffers_, passDependencyBufferAccessModes_);
  splitDependencies(
      std::span<const TextureDependency>(passTextureDependencies_.data(),
                                         passTextureDependencies_.size()),
      passDependencyTextures_, passDependencyTextureAccessModes_);
  const auto publishStaticOnlyCascadeState = [&](uint32_t cascadeIndex,
                                                 TextureHandle texture) {
    if (cascadeStates_[cascadeIndex].dynamicDrawCount == 0u) {
      cascadeStates_[cascadeIndex].reusableValid = true;
      cascadeStates_[cascadeIndex].reusableContentSignature =
          cascadeStates_[cascadeIndex].contentSignature;
      cascadeStates_[cascadeIndex].reusable = StaticOnlyCascadeReuseState{
          .renderedFit = makeDirectionalShadowFitFromDebugCascade(
              shadowDebugFrameData_.cascades[cascadeIndex]),
          .rawFit = currentRawShadowFits_[cascadeIndex],
          .shadowDepthTexture = texture,
          .rasterSignature = cascadeStates_[cascadeIndex].contentSignature,
          .lightViewProjSignature =
              shadowDebugFrameData_.cascades[cascadeIndex]
                  .currentStaticOnlyLightViewProjSignature,
          .biasSignature = shadowDebugFrameData_.cascades[cascadeIndex]
                               .currentStaticOnlyBiasSignature,
          .casterSignature = shadowDebugFrameData_.cascades[cascadeIndex]
                                 .currentStaticOnlyCasterSignature,
          .staticDrawCount =
              shadowDebugFrameData_.cascades[cascadeIndex].staticDrawCount,
          .dynamicDrawCount =
              shadowDebugFrameData_.cascades[cascadeIndex].dynamicDrawCount,
      };
    } else {
      cascadeStates_[cascadeIndex].reusableValid = false;
      cascadeStates_[cascadeIndex].reusableContentSignature = 0u;
      cascadeStates_[cascadeIndex].reusable = StaticOnlyCascadeReuseState{};
    }
  };
  for (uint32_t cascadeIndex = 0u; cascadeIndex < activeCascadeCount_;
       ++cascadeIndex) {
    const TextureHandle shadowDepthTexture = shadowDepthTextures_[cascadeIndex];
    const bool reuseStaticOnlyCascade = cascadeStates_[cascadeIndex].reuse;
    const TextureDimensions shadowDepthDimensions =
        gpu_.getTextureDimensions(shadowDepthTexture);
    const float shadowViewportWidth =
        static_cast<float>(std::max(shadowDepthDimensions.width, 1u));
    const float shadowViewportHeight =
        static_cast<float>(std::max(shadowDepthDimensions.height, 1u));
    const RenderGraphTextureId depthGraphTexture =
        graph
            .importTexture(shadowDepthTexture,
                           kShadowCascadeTextureImportNames[cascadeIndex])
            .value();
    frame.sharedResources.shadowCascadeGraphTextures[cascadeIndex] =
        depthGraphTexture;
    publishRequestedCapture(
        frame, gpu_, kShadowCascadeCaptureNames[cascadeIndex],
        shadowDepthTexture, RenderCaptureValueKind::ShadowDepth,
        RenderCaptureLifetimeClass::FeaturePersistentTexture, "linear_depth",
        "shadow_depth", kShadowCascadePassLabels[cascadeIndex]);
    if (reuseStaticOnlyCascade) {
      publishStaticOnlyCascadeState(cascadeIndex, shadowDepthTexture);
      continue;
    }
    const std::span<const ComputeDispatchItem> preDispatches =
        shadowCascadePreDispatches(animationSceneData, cascadeIndex);
    const auto &submittedDraws = cascadeIndirectDrawItems_[cascadeIndex];
    RenderGraphGraphicsPassDesc desc{
        .hasColorAttachment = false,
        .depth = {.loadOp = LoadOp::Clear,
                  .storeOp = StoreOp::Store,
                  .clearDepth = 1.0f},
        .depthTexture = depthGraphTexture,
        .useViewport = true,
        .viewport = {.width = shadowViewportWidth,
                     .height = shadowViewportHeight,
                     .maxDepth = 1.0f},
        .preDispatches = preDispatches,
        .dependencyBuffers = passDependencyBuffers_,
        .dependencyBufferAccessModes = passDependencyBufferAccessModes_,
        .dependencyTextures = passDependencyTextures_,
        .dependencyTextureAccessModes = passDependencyTextureAccessModes_,
        .draws = submittedDraws,
        .drawBuffersPreResolved = true,
        .preResolvedDrawBuffers = preResolvedDrawBuffers_,
        .gpuTimingScope = GpuTimingScope::ShadowDepth,
        .debugLabel = kShadowCascadePassLabels[cascadeIndex],
        .debugColor = kShadowPassDebugColor,
    };
    desc.borrowPayload = false;
    const RenderGraphPassId pass = graph.addGraphicsPass(desc).value();
    (void)graph.markPassSideEffect(pass).value();
    publishStaticOnlyCascadeState(cascadeIndex, shadowDepthTexture);
  }
  for (uint32_t cascadeIndex = activeCascadeCount_;
       cascadeIndex < kMaxShadowCascades; ++cascadeIndex) {
    cascadeStates_[cascadeIndex].reusableValid = false;
    cascadeStates_[cascadeIndex].reusableContentSignature = 0u;
    cascadeStates_[cascadeIndex].reusable = StaticOnlyCascadeReuseState{};
  }
  if (hasPreparedShadowPreviewPass_ &&
      nuri::isValid(shadowDebugPreviewTexture_)) {
    const TextureDimensions previewDimensions =
        gpu_.getTextureDimensions(shadowDebugPreviewTexture_);
    const float previewViewportWidth =
        static_cast<float>(std::max(previewDimensions.width, 1u));
    const float previewViewportHeight =
        static_cast<float>(std::max(previewDimensions.height, 1u));
    const RenderGraphTextureId previewGraphTexture =
        graph.importTexture(shadowDebugPreviewTexture_, "shadow_depth_preview")
            .value();
    frame.sharedResources.shadowDebugPreviewGraphTexture = previewGraphTexture;
    publishRequestedCapture(
        frame, gpu_, "shadow_preview", shadowDebugPreviewTexture_,
        RenderCaptureValueKind::DebugPreview,
        RenderCaptureLifetimeClass::FeaturePersistentTexture, "display_sdr",
        "debug_preview", "Shadow Depth Preview Pass");
    std::pmr::vector<TextureHandle> previewDependencyTextures(memory_);
    std::pmr::vector<RenderGraphAccessMode> previewDependencyTextureAccessModes(
        memory_);
    splitDependencies(
        std::span<const TextureDependency>(previewTextureDependencies_.data(),
                                           previewTextureDependencies_.size()),
        previewDependencyTextures, previewDependencyTextureAccessModes);
    const RenderGraphGraphicsPassDesc previewDesc{
        .color = {.loadOp = LoadOp::Clear,
                  .storeOp = StoreOp::Store,
                  .clearColor = {1.0f, 1.0f, 1.0f, 1.0f}},
        .colorTexture = previewGraphTexture,
        .hasColorAttachment = true,
        .useViewport = true,
        .viewport = {.width = previewViewportWidth,
                     .height = previewViewportHeight,
                     .maxDepth = 1.0f},
        .dependencyTextures = previewDependencyTextures,
        .dependencyTextureAccessModes = previewDependencyTextureAccessModes,
        .draws = std::span<const DrawItem>(&previewDraw_, 1u),
        .debugLabel = kShadowPreviewPassLabel,
        .debugColor = kShadowPreviewPassDebugColor,
        .borrowPayload = true,
    };
    const RenderGraphPassId previewPass =
        graph.addGraphicsPass(previewDesc).value();
    (void)graph.markPassSideEffect(previewPass).value();
  }
  return Result<bool, std::string>::makeResult(true);
}

namespace {
ShadowRenderer *registerShadowRenderer(RenderPipeline &pipeline,
                                       std::unique_ptr<ShadowRenderer> owner) {
  auto *renderer = pipeline.addComponent(
      std::move(owner),
      PipelineComponentDesc{
          .publish =
              [](void *state, FrameBuildContext &ctx) {
                NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_COPY);
                return static_cast<ShadowRenderer *>(state)->publishFrameData(
                    ctx.frame);
              },
          .prepare =
              [](void *state, FrameBuildContext &ctx) {
                NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_COPY);
                return static_cast<ShadowRenderer *>(state)
                    ->prepareShadowGraphPasses(ctx.frame);
              },
          .prepareScene =
              [](void *state, RenderScenePreparationContext &ctx) {
                return static_cast<ShadowRenderer *>(state)->prepareSceneCache(
                    *ctx.sceneDrawDatabase, ctx.scene, ctx.resources,
                    ctx.settings);
              },
      });
  pipeline.addStage(PipelineStageDesc{
      .componentName = "ShadowFeature",
      .name = "ShadowDepthPass",
      .state = renderer,
      .enabled =
          [](const void *state, const FrameBuildContext &ctx) {
            return ctx.frame.settings &&
                   renderSettingsOrDefault(ctx.frame).shadow.enabled &&
                   static_cast<const ShadowRenderer *>(state)
                       ->hasPreparedShadowDepthPasses();
          },
      .build =
          [](void *state, FrameBuildContext &ctx) {
            NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_DRAW);
            return static_cast<ShadowRenderer *>(state)
                ->appendShadowDepthPasses(ctx.frame, ctx.graph);
          },
  });
  return renderer;
}
} // namespace

ShadowRenderer *registerShadowStage(RenderPipeline &pipeline, GPUDevice &gpu,
                                    std::pmr::memory_resource *memory,
                                    SceneDrawDatabase *database) {
  if (!database) {
    pipeline.addProvider(std::make_unique<SceneDrawDatabase>(gpu, memory));
  }
  return registerShadowRenderer(pipeline,
                                std::make_unique<ShadowRenderer>(gpu, memory));
}

ShadowRenderer *registerShadowStage(RenderPipeline &pipeline, GPUDevice &gpu,
                                    const RuntimeOpaqueShaderConfig &config,
                                    std::pmr::memory_resource *memory,
                                    SceneDrawDatabase *database) {
  if (!database) {
    pipeline.addProvider(std::make_unique<SceneDrawDatabase>(gpu, memory));
  }
  return registerShadowRenderer(
      pipeline, std::make_unique<ShadowRenderer>(gpu, config, memory));
}

} // namespace nuri
