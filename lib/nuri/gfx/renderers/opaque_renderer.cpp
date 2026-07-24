#include "nuri/gfx/renderers/opaque_renderer.h"
#include "nuri/core/containers/hash_map.h"
#include "nuri/core/log.h"
#include "nuri/core/pmr_scratch.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/frame/presentation_aa_plan.h"
#include "nuri/gfx/frame/render_capture.h"
#include "nuri/gfx/fullscreen.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/gfx/renderers/detail/animation_rendering.h"
#include "nuri/gfx/renderers/detail/opaque_lod_selection.h"
#include "nuri/gfx/renderers/detail/opaque_meshlet_routing.h"
#include "nuri/gfx/renderers/detail/shadow_math.h"
#include "nuri/gfx/visibility/visibility.h"
#include "nuri/pch.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/scene/render_scene.h"
#include <bit>
#include <cmath>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <unordered_set>
namespace nuri {
namespace {
constexpr float kMinLodRadius = 1.0e-4f;
constexpr float kBoundsRadiusHalf = 0.5f;
constexpr size_t kMaxBatchReserve = 128;
constexpr float kClearDepthOne = 1.0f;
constexpr float kClearColorWhite = 1.0f;
constexpr uint32_t kOpaquePassDebugColor = 0xff0000ff;
constexpr uint32_t kMeshDebugColor = 0xffcc5500;
constexpr uint32_t kComputeDispatchColor = 0xff33aa33;
constexpr uint32_t kComputeWorkgroupSize = 32;
constexpr uint32_t kOpaqueMeshletTaskCandidatesPerGroup = 32u;
constexpr uint32_t kOpaqueMeshletTaskPayloadBytes =
    sizeof(uint32_t) * (1u + 4u * kOpaqueMeshletTaskCandidatesPerGroup);
constexpr size_t kMeshletNormalDepthMergeMinVisibleInstances = 1u;
constexpr uint32_t kTessellationPatchControlPoints = 3;
constexpr size_t surfaceVariantIndex(bool tessellated, bool doubleSided) {
  return static_cast<size_t>(tessellated) |
         (static_cast<size_t>(doubleSided) << 1u);
}
constexpr size_t rasterVariantIndex(CoverageMode coverage, bool alphaMasked,
                                    bool tessellated, bool doubleSided) {
  return static_cast<size_t>(alphaMasked) |
         (static_cast<size_t>(tessellated) << 1u) |
         (static_cast<size_t>(doubleSided) << 2u) |
         (coverageModeIndex(coverage) << 3u);
}
constexpr size_t meshletSceneVariantIndex(bool compacted, CoverageMode coverage,
                                          bool alphaMasked, bool doubleSided) {
  return static_cast<size_t>(doubleSided) |
         (static_cast<size_t>(compacted) << 1u) |
         (static_cast<size_t>(alphaMasked) << 2u) |
         (coverageModeIndex(coverage) << 3u);
}
constexpr size_t meshletDepthVariantIndex(CoverageMode coverage,
                                          bool alphaMasked, bool doubleSided) {
  return static_cast<size_t>(doubleSided) |
         (static_cast<size_t>(alphaMasked) << 1u) |
         (coverageModeIndex(coverage) << 2u);
}
constexpr size_t kIndirectCountHeaderBytes = sizeof(uint32_t);
constexpr uint32_t kMaxIndirectCommandsPerDraw = 1024u;
constexpr size_t kMaxDrawItemsForIndirectPath = 8192u;
constexpr uint32_t kUnlimitedTessInstanceCap = 0u;
constexpr uint32_t kVelocityGeometryFlagPreviousVertexBuffer = 1u << 0u;
constexpr float kOverlayDepthBiasConstant = -1.0f;
constexpr float kOverlayDepthBiasSlope = -1.0f;
constexpr uint32_t kAutoLodCacheInvalidationSeed = 1664525u;
constexpr uint32_t kAutoLodCacheInvalidationMagic = 1013904223u;
constexpr uint32_t kMeshletFlagFrustumCulling = 1u << 0u;
constexpr uint32_t kMeshletFlagConeCulling = 1u << 1u;
constexpr uint32_t kMeshletFlagDoubleSided = 1u << 2u;
constexpr uint32_t kMeshletFlagDebugMeshletId = 1u << 3u;
constexpr uint32_t kMeshletFlagGpuLod = 1u << 4u;
constexpr uint32_t kMeshletFlagDebugSelectedLod = 1u << 5u;
constexpr uint32_t kMeshletFlagOcclusionCulling = 1u << 7u;
constexpr uint32_t kMeshletFlagForcedLodShift = 8u;
constexpr uint32_t kMeshletFlagForcedLodMask = 0x3u;
constexpr uint32_t kMeshletFlagCurrentFrameOcclusion = 1u << 10u;
constexpr uint32_t kMeshletCounterFlagEnabled = 1u << 0u;
enum class MeshletOcclusionSource : uint8_t {
  Disabled = 0,
  PreviousFrame = 1,
  CurrentFrame = 2,
};
struct MeshletOcclusionPlan {
  MeshletOcclusionSource source = MeshletOcclusionSource::Disabled;
  uint32_t pyramidLevelCount = 0u;
  uint64_t sourceFrame = 0u;
  [[nodiscard]] bool active() const noexcept {
    return source != MeshletOcclusionSource::Disabled;
  }
  [[nodiscard]] bool usesCurrentFrame() const noexcept {
    return source == MeshletOcclusionSource::CurrentFrame;
  }
};
constexpr uint32_t kPhaseHashMask = 0x00ffffffu;
constexpr float kPhaseNormDivisor = 16777215.0f;
constexpr uint32_t kPhaseHashMixMultiplier = 2246822519u;
constexpr uint32_t kPhaseHashShift1 = 16u;
constexpr uint32_t kPhaseHashShift2 = 13u;
constexpr uint64_t kFnvOffsetBasis64 = 14695981039346656037ull;
constexpr uint64_t kFnvPrime64 = 1099511628211ull;
constexpr uint64_t kInvalidDrawSignature = std::numeric_limits<uint64_t>::max();
constexpr std::string_view kOpaquePickPassLabel = "Opaque Pick Pass";
constexpr std::string_view kOpaqueMainPassLabel = "Opaque Pass";
constexpr std::array<RasterPipelineState, 1> kOpaqueDepthPreparedMainStates{
    RasterPipelineState{.compareOp = CompareOp::Equal, .depthWrite = false}};
constexpr std::array<RasterPipelineState, 1> kOpaqueReadOnlyAuxiliaryStates{
    RasterPipelineState{.compareOp = CompareOp::LessEqual,
                        .depthWrite = false}};
constexpr RasterPipelineState kOpaqueOverlayRasterState{
    .compareOp = CompareOp::LessEqual,
    .depthWrite = false,
    .depthBiasEnable = true,
    .depthBiasConstant = -1,
    .depthBiasSlope = kOverlayDepthBiasSlope,
};
template <typename PipelineDesc>
[[nodiscard]] PipelineDesc
withOpaqueMainDepthVariants(PipelineDesc desc) noexcept {
  desc.prewarmRasterStates = kOpaqueDepthPreparedMainStates;
  return desc;
}
template <typename PipelineDesc>
[[nodiscard]] PipelineDesc
withOpaqueReadOnlyAuxiliaryVariant(PipelineDesc desc) noexcept {
  desc.prewarmRasterStates = kOpaqueReadOnlyAuxiliaryStates;
  return desc;
}
uint64_t hashCombine64(uint64_t hash, uint64_t value) {
  hash ^= value;
  hash *= kFnvPrime64;
  return hash;
}
template <typename... Containers> void clearAll(Containers &...containers) {
  (containers.clear(), ...);
}
template <typename... Values> void resetAll(Values &...values) {
  ((values = std::remove_cvref_t<Values>{}), ...);
}
template <typename Value, typename... Targets>
void setAll(Value value, Targets &...targets) {
  ((targets = value), ...);
}
uint64_t hashVisibilityVisibleIndexList(std::span<const uint32_t> indices) {
  uint64_t hash = kFnvOffsetBasis64;
  for (const uint32_t index : indices) {
    hash = hashCombine64(hash, index);
  }
  return hash;
}
uint64_t hashSortedVisibilityVisibleIndexList(std::span<uint32_t> indices) {
  std::sort(indices.begin(), indices.end());
  return hashVisibilityVisibleIndexList(indices);
}
uint32_t visibilityReadbackAge(uint64_t currentFrame, uint32_t sourceFrame) {
  if (static_cast<uint64_t>(sourceFrame) >= currentFrame) {
    return 0u;
  }
  const uint64_t age = currentFrame - static_cast<uint64_t>(sourceFrame);
  return age > std::numeric_limits<uint32_t>::max()
             ? std::numeric_limits<uint32_t>::max()
             : static_cast<uint32_t>(age);
}
VisibilityPassResult evaluateCpuVisibilityFromCachedBounds(
    const VisibilityPassRequest &request,
    std::span<const VisibilityCandidate> candidates,
    std::span<const VisibilityCandidateGpu> candidateGpuData,
    std::pmr::memory_resource *memory) {
  const size_t candidateCount = candidates.size();
  VisibilityPassResult result(memory);
  result.signature = request.signature;
  result.cpuCandidates = static_cast<uint32_t>(
      std::min(candidateCount,
               static_cast<size_t>(std::numeric_limits<uint32_t>::max())));
  result.visibleCandidateIndices.reserve(candidateCount);
  for (size_t i = 0; i < candidateCount; ++i) {
    const VisibilityCandidate &candidate = candidates[i];
    bool visible = true;
    if (request.enableCpuFrustumCulling) {
      if ((candidate.flags & kVisibilityCandidateConservativeVisible) != 0u) {
        ++result.uncertainVisible;
      } else {
        const VisibilityCandidateGpu &gpuCandidate = candidateGpuData[i];
        visibility_detail::VisibilityClassification classification =
            visibility_detail::classifySphere(request.frustum,
                                              glm::vec3(gpuCandidate.bounds),
                                              gpuCandidate.bounds.w);
        if (classification ==
            visibility_detail::VisibilityClassification::Intersects) {
          const BoundingBox worldBounds(glm::vec3(gpuCandidate.boundsMin),
                                        glm::vec3(gpuCandidate.boundsMax));
          classification =
              visibility_detail::classifyAabb(request.frustum, worldBounds);
        }
        visible = visibility_detail::isVisible(classification);
      }
    }
    if (!visible) {
      ++result.cpuRejected;
      continue;
    }
    result.visibleCandidateIndices.push_back(static_cast<uint32_t>(i));
  }
  result.cpuVisibleCandidates = static_cast<uint32_t>(
      std::min(result.visibleCandidateIndices.size(),
               static_cast<size_t>(std::numeric_limits<uint32_t>::max())));
  return result;
}
bool matrixNearlyEqual(const glm::mat4 &lhs, const glm::mat4 &rhs,
                       float epsilon = 1.0e-5f) {
  for (int column = 0; column < 4; ++column) {
    for (int row = 0; row < 4; ++row) {
      if (std::abs(lhs[column][row] - rhs[column][row]) > epsilon) {
        return false;
      }
    }
  }
  return true;
}
bool previousDepthPyramidCameraStable(const RenderFrameContext &frame) {
  return frame.sharedResources.sceneDepthPyramidSourceViewProj.has_value() &&
         matrixNearlyEqual(
             *frame.sharedResources.sceneDepthPyramidSourceViewProj,
             frame.camera.currentUnjitteredViewProj);
}
[[nodiscard]] uint32_t fullDepthPyramidLevelCount(uint32_t width,
                                                  uint32_t height) noexcept {
  uint32_t levelCount = 1u;
  uint32_t maxDim = std::max(width, height);
  while (maxDim > 1u && levelCount < kMaxSceneDepthPyramidLevels) {
    maxDim = (maxDim + 1u) >> 1u;
    ++levelCount;
  }
  return levelCount;
}
void logOpaqueVisibilityCounters(const RenderFrameContext &frame) {
  if (frame.settings == nullptr ||
      !renderSettingsOrDefault(frame).visibility.debug.logCounters) {
    return;
  }
  const VisibilityFrameMetrics &visibility = frame.metrics.visibility;
  NURI_LOG_INFO(
      "OpaqueRenderer::visibility frame=%llu cpu=%u/%u gpu=%u/%u "
      "meshlets=%u/%u indirect=%u/%u",
      static_cast<unsigned long long>(frame.frameIndex),
      visibility.cpuMainCandidates, visibility.cpuMainVisibleCandidates,
      visibility.gpuMainCandidates, visibility.gpuMainVisibleCandidates,
      visibility.meshletCandidates, visibility.meshletEmitted,
      visibility.gpuIndirectDrawCommands, visibility.indirectMeshDispatchCount);
}
uint64_t foldHandle(uint32_t index, uint32_t generation) {
  return (static_cast<uint64_t>(generation) << 32u) | index;
}
uint64_t hashBufferHandleSignature(uint64_t signature, BufferHandle handle) {
  return nuri::isValid(handle)
             ? hashCombine64(signature,
                             foldHandle(handle.index, handle.generation))
             : signature;
}
uint64_t hashDrawBufferSignature(uint64_t signature, const DrawItem &draw) {
  signature = hashBufferHandleSignature(signature, draw.vertexBuffer);
  signature = hashBufferHandleSignature(signature, draw.indexBuffer);
  signature = hashBufferHandleSignature(signature, draw.indirectBuffer);
  signature = hashBufferHandleSignature(signature, draw.indirectCountBuffer);
  return signature;
}
std::pmr::memory_resource *
resolveMemoryResource(std::pmr::memory_resource *memory) {
  return memory != nullptr ? memory : std::pmr::get_default_resource();
}
RenderPipelineDesc
meshPipelineDesc(Format swapchainFormat, Format depthFormat,
                 ShaderHandle vertexShader, ShaderHandle tessControlShader,
                 ShaderHandle tessEvalShader, ShaderHandle geometryShader,
                 ShaderHandle fragmentShader, PolygonMode polygonMode,
                 Topology topology = Topology::Triangle,
                 uint32_t patchControlPoints = 0, bool blendEnabled = false,
                 CullMode cullMode = CullMode::Back) {
  return RenderPipelineDesc{
      .vertexInput = {},
      .vertexShader = vertexShader,
      .tessControlShader = tessControlShader,
      .tessEvalShader = tessEvalShader,
      .geometryShader = geometryShader,
      .fragmentShader = fragmentShader,
      .colorFormats = {swapchainFormat},
      .depthFormat = depthFormat,
      .cullMode = cullMode,
      .polygonMode = polygonMode,
      .topology = topology,
      .patchControlPoints = patchControlPoints,
      .blendEnabled = blendEnabled,
  };
}
RenderPipelineDesc
depthPipelineDesc(Format depthFormat, ShaderHandle vertexShader,
                  ShaderHandle tessControlShader, ShaderHandle tessEvalShader,
                  ShaderHandle fragmentShader, CullMode cullMode,
                  Topology topology = Topology::Triangle,
                  uint32_t patchControlPoints = 0) {
  return RenderPipelineDesc{
      .vertexInput = {},
      .vertexShader = vertexShader,
      .tessControlShader = tessControlShader,
      .tessEvalShader = tessEvalShader,
      .fragmentShader = fragmentShader,
      .colorFormats = {Format::RGBA8_UNORM},
      .colorAttachmentCount = 0u,
      .depthFormat = depthFormat,
      .cullMode = cullMode,
      .polygonMode = PolygonMode::Fill,
      .topology = topology,
      .patchControlPoints = patchControlPoints,
      .blendEnabled = false,
  };
}
bool isSamePipelineHandle(RenderPipelineHandle lhs, RenderPipelineHandle rhs) {
  return lhs.index == rhs.index && lhs.generation == rhs.generation;
}
RenderPipelineHandle
selectSurfaceVariant(const std::array<RenderPipelineHandle, 4> &pipelines,
                     bool tessellated, bool doubleSided,
                     bool allowNonTessellatedFallback) {
  for (;;) {
    const RenderPipelineHandle sided =
        pipelines[surfaceVariantIndex(tessellated, doubleSided)];
    if (doubleSided && nuri::isValid(sided)) {
      return sided;
    }
    const RenderPipelineHandle singleSided =
        pipelines[surfaceVariantIndex(tessellated, false)];
    if (nuri::isValid(singleSided) || !tessellated ||
        !allowNonTessellatedFallback) {
      return singleSided;
    }
    tessellated = false;
  }
}
bool matchesVariantBit(RenderPipelineHandle handle,
                       std::span<const RenderPipelineHandle> pipelines,
                       size_t bit) {
  for (size_t i = bit; i < pipelines.size(); ++i) {
    if ((i & bit) != 0u && isSamePipelineHandle(handle, pipelines[i])) {
      return true;
    }
  }
  return false;
}
void destroyPipelineHandle(GPUDevice &gpu, RenderPipelineHandle &handle) {
  if (nuri::isValid(handle)) {
    gpu.destroyRenderPipeline(handle);
    handle = {};
  }
}
void destroyMeshletPipelineHandle(GPUDevice &gpu,
                                  MeshletPipelineHandle &handle) {
  if (nuri::isValid(handle)) {
    gpu.destroyMeshletPipeline(handle);
    handle = {};
  }
}
DrawItem makeBaseMeshDraw(RenderPipelineHandle pipeline,
                          std::string_view debugLabel) {
  DrawItem draw{};
  draw.pipeline = pipeline;
  draw.indexFormat = IndexFormat::U32;
  draw.useDepthState = true;
  draw.depthState = {.compareOp = CompareOp::Less, .isDepthWriteEnabled = true};
  draw.debugLabel = debugLabel;
  draw.debugColor = kMeshDebugColor;
  return draw;
}
float maxAxisScale(const glm::mat4 &transform) {
  const float sx = glm::length(glm::vec3(transform[0]));
  const float sy = glm::length(glm::vec3(transform[1]));
  const float sz = glm::length(glm::vec3(transform[2]));
  return std::max({sx, sy, sz});
}
bool nearlyEqualVec3(const glm::vec3 &a, const glm::vec3 &b, float epsilon) {
  const glm::vec3 delta = glm::abs(a - b);
  return delta.x <= epsilon && delta.y <= epsilon && delta.z <= epsilon;
}
float maxMatrixElementDelta(const glm::mat4 &a, const glm::mat4 &b) {
  float maxDelta = 0.0f;
  for (glm::length_t column = 0; column < 4; ++column) {
    for (glm::length_t row = 0; row < 4; ++row) {
      maxDelta = std::max(maxDelta, std::abs(a[column][row] - b[column][row]));
    }
  }
  return maxDelta;
}
uint64_t textureStorageBytes(GPUDevice &gpu, TextureHandle texture) {
  if (!nuri::isValid(texture)) {
    return 0u;
  }
  const TextureDimensions dimensions = gpu.getTextureDimensions(texture);
  return static_cast<uint64_t>(std::max(dimensions.width, 1u)) *
         static_cast<uint64_t>(std::max(dimensions.height, 1u)) *
         formatTexelBytes(gpu.getTextureFormat(texture));
}
uint64_t computeRemapSignature(std::span<const uint32_t> remap) {
  uint64_t signature = hashCombine64(kFnvOffsetBasis64, remap.size());
  for (const uint32_t value : remap) {
    signature = hashCombine64(signature, static_cast<uint64_t>(value));
  }
  return signature;
}
bool isSameBufferHandle(BufferHandle a, BufferHandle b) {
  return a.index == b.index && a.generation == b.generation;
}
bool isSameTextureHandle(TextureHandle a, TextureHandle b) {
  return a.index == b.index && a.generation == b.generation;
}
void registerOrUpdatePersistentBuffer(RenderGraphBuilder &graph,
                                      PersistentBufferId &persistentId,
                                      BufferHandle &registeredHandle,
                                      BufferHandle currentHandle,
                                      std::string_view name) {
  if (!isValid(persistentId)) {
    persistentId = graph.registerPersistentBuffer(currentHandle, name);
    registeredHandle = currentHandle;
    return;
  }
  if (!isSameBufferHandle(registeredHandle, currentHandle)) {
    graph.updatePersistentBuffer(persistentId, currentHandle);
    registeredHandle = currentHandle;
  }
}
void appendUniqueDependency(
    std::pmr::vector<BufferHandle> &dependencies,
    std::pmr::vector<RenderGraphAccessMode> &accessModes, BufferHandle handle,
    RenderGraphAccessMode accessMode) {
  if (!nuri::isValid(handle)) {
    return;
  }
  for (size_t i = 0; i < dependencies.size(); ++i) {
    if (isSameBufferHandle(dependencies[i], handle)) {
      accessModes[i] = accessModes[i] | accessMode;
      return;
    }
  }
  dependencies.push_back(handle);
  accessModes.push_back(accessMode);
}
template <typename Handle>
void appendUniqueDependency(std::pmr::vector<Handle> &dependencies,
                            Handle handle) {
  if (!nuri::isValid(handle)) {
    return;
  }
  if (std::ranges::find(dependencies, handle) == dependencies.end())
    dependencies.push_back(handle);
}
std::optional<uint32_t> resolveAvailableLod(const Submesh &submesh,
                                            uint32_t desiredLod) {
  const uint32_t lodCount =
      std::clamp(submesh.lodCount, 1u, Submesh::kMaxLodCount);
  uint32_t candidate = std::min(desiredLod, lodCount - 1);
  while (candidate > 0 && submesh.lods[candidate].indexCount == 0) {
    --candidate;
  }
  if (submesh.lods[candidate].indexCount == 0) {
    return std::nullopt;
  }
  return candidate;
}
uint32_t maxMeshletCountForSubmesh(const Submesh &submesh) {
  const uint32_t lodCount =
      std::clamp(submesh.lodCount, 1u, Submesh::kMaxLodCount);
  uint32_t maxMeshletCount = 0;
  for (uint32_t lodIndex = 0; lodIndex < lodCount; ++lodIndex) {
    maxMeshletCount =
        std::max(maxMeshletCount, submesh.lods[lodIndex].meshletCount);
  }
  return maxMeshletCount;
}
float deterministicPhase(uint32_t index) {
  uint32_t hash =
      index * kAutoLodCacheInvalidationSeed + kAutoLodCacheInvalidationMagic;
  hash ^= hash >> kPhaseHashShift1;
  hash *= kPhaseHashMixMultiplier;
  hash ^= hash >> kPhaseHashShift2;
  return static_cast<float>(hash & kPhaseHashMask) / kPhaseNormDivisor *
         glm::two_pi<float>();
}
glm::mat4 makeBuiltInAnimatedModel(const glm::vec4 &centerPhase,
                                   const glm::mat4 &baseMatrix,
                                   float timeSeconds) {
  const glm::mat4 translation =
      glm::translate(glm::mat4(1.0f), glm::vec3(centerPhase));
  return glm::rotate(translation, timeSeconds + centerPhase.w,
                     glm::normalize(glm::vec3(1.0f))) *
         baseMatrix;
}
struct BatchKey {
  RenderPipelineHandle pipeline{};
  BufferHandle indexBuffer{};
  uint64_t indexBufferOffset = 0;
  uint32_t indexCount = 0;
  uint32_t firstIndex = 0;
  uint64_t vertexBufferAddress = 0;
  uint64_t vertexDecodeBufferAddress = 0;
  uint32_t vertexDecodeIndex = 0;
  uint32_t packedVertexFormat = 0;
  uint32_t materialIndex = kInvalidMaterialIndex;
  BufferHandle meshletBuffer{};
  uint32_t meshletOffset = 0;
  uint32_t meshletCount = 0;
  uint32_t meshletSubmeshIndex = 0;
  bool operator==(const BatchKey &other) const {
    return isSamePipelineHandle(pipeline, other.pipeline) &&
           isSameBufferHandle(indexBuffer, other.indexBuffer) &&
           indexBufferOffset == other.indexBufferOffset &&
           indexCount == other.indexCount && firstIndex == other.firstIndex &&
           vertexBufferAddress == other.vertexBufferAddress &&
           vertexDecodeBufferAddress == other.vertexDecodeBufferAddress &&
           vertexDecodeIndex == other.vertexDecodeIndex &&
           packedVertexFormat == other.packedVertexFormat &&
           materialIndex == other.materialIndex &&
           isSameBufferHandle(meshletBuffer, other.meshletBuffer) &&
           meshletOffset == other.meshletOffset &&
           meshletCount == other.meshletCount &&
           meshletSubmeshIndex == other.meshletSubmeshIndex;
  }
};
struct BatchKeyHash {
  size_t operator()(const BatchKey &key) const noexcept {
    uint64_t h64 = kFnvOffsetBasis64;
    const auto mix = [&h64](uint64_t v) {
      h64 ^= v;
      h64 *= kFnvPrime64;
    };
    mix((static_cast<uint64_t>(key.pipeline.generation) << 32u) |
        key.pipeline.index);
    mix((static_cast<uint64_t>(key.indexBuffer.generation) << 32u) |
        key.indexBuffer.index);
    mix(key.indexBufferOffset);
    mix((static_cast<uint64_t>(key.indexCount) << 32u) | key.firstIndex);
    mix(key.vertexBufferAddress);
    mix(key.vertexDecodeBufferAddress);
    mix((static_cast<uint64_t>(key.vertexDecodeIndex) << 32u) |
        key.packedVertexFormat);
    mix(static_cast<uint64_t>(key.materialIndex));
    mix((static_cast<uint64_t>(key.meshletBuffer.generation) << 32u) |
        key.meshletBuffer.index);
    mix((static_cast<uint64_t>(key.meshletOffset) << 32u) | key.meshletCount);
    mix(static_cast<uint64_t>(key.meshletSubmeshIndex));
    return static_cast<size_t>(h64);
  }
};
struct DrawIndexedIndirectCommand {
  uint32_t indexCount = 0;
  uint32_t instanceCount = 0;
  uint32_t firstIndex = 0;
  int32_t vertexOffset = 0;
  uint32_t firstInstance = 0;
};
static_assert(sizeof(DrawIndexedIndirectCommand) == 20);
struct IndirectGroupKey {
  RenderPipelineHandle pipeline{};
  BufferHandle indexBuffer{};
  uint64_t indexBufferOffset = 0;
  IndexFormat indexFormat = IndexFormat::U32;
  uint64_t vertexBufferAddress = 0;
  uint64_t vertexDecodeBufferAddress = 0;
  uint32_t vertexDecodeIndex = 0;
  uint32_t packedVertexFormat = 0;
  uint32_t materialIndex = kInvalidMaterialIndex;
  bool operator==(const IndirectGroupKey &other) const {
    return isSamePipelineHandle(pipeline, other.pipeline) &&
           isSameBufferHandle(indexBuffer, other.indexBuffer) &&
           indexBufferOffset == other.indexBufferOffset &&
           indexFormat == other.indexFormat &&
           vertexBufferAddress == other.vertexBufferAddress &&
           vertexDecodeBufferAddress == other.vertexDecodeBufferAddress &&
           vertexDecodeIndex == other.vertexDecodeIndex &&
           packedVertexFormat == other.packedVertexFormat &&
           materialIndex == other.materialIndex;
  }
};
struct IndirectGroupKeyHash {
  size_t operator()(const IndirectGroupKey &key) const noexcept {
    uint64_t h64 = kFnvOffsetBasis64;
    const auto mix = [&h64](uint64_t v) {
      h64 ^= v;
      h64 *= kFnvPrime64;
    };
    mix((static_cast<uint64_t>(key.pipeline.generation) << 32u) |
        key.pipeline.index);
    mix((static_cast<uint64_t>(key.indexBuffer.generation) << 32u) |
        key.indexBuffer.index);
    mix(key.indexBufferOffset);
    mix(static_cast<uint64_t>(key.indexFormat));
    mix(key.vertexBufferAddress);
    mix(key.vertexDecodeBufferAddress);
    mix((static_cast<uint64_t>(key.vertexDecodeIndex) << 32u) |
        key.packedVertexFormat);
    mix(static_cast<uint64_t>(key.materialIndex));
    return static_cast<size_t>(h64);
  }
};
} // namespace

OpaqueRenderer::OpaqueRenderer(GPUDevice &gpu, OpaqueRendererConfig config,
                               std::pmr::memory_resource *memory)
    : gpu_(gpu), config_(std::move(config)),
      bufferRings_(resolveMemoryResource(memory)),
      singleInstanceBatchCaches_(resolveMemoryResource(memory)),
      staticBatchCache_(resolveMemoryResource(memory)),
      renderableTemplates_(std::pmr::new_delete_resource()),
      meshDrawTemplates_(std::pmr::new_delete_resource()),
      indirectSourceDrawIndices_(resolveMemoryResource(memory)),
      frameSlotStates_(resolveMemoryResource(memory)),
      visibilityVisibleIndexReadback_(resolveMemoryResource(memory)),
      visibilityCandidates_(std::pmr::new_delete_resource()),
      visibilityCandidateGpuData_(std::pmr::new_delete_resource()),
      templateBatchIndices_(resolveMemoryResource(memory)),
      cachedVisibleTemplateBatchIndices_(resolveMemoryResource(memory)),
      visibleBatchActiveRemap_(resolveMemoryResource(memory)),
      cachedVisibleBatchEntries_(resolveMemoryResource(memory)),
      batchWriteOffsets_(resolveMemoryResource(memory)),
      instanceCentersPhase_(std::pmr::new_delete_resource()),
      instanceBaseMatrices_(std::pmr::new_delete_resource()),
      instanceMatricesCpuCache_(std::pmr::new_delete_resource()),
      instanceLodCentersInvRadiusSq_(std::pmr::new_delete_resource()),
      instanceAutoLodWorldErrors_(std::pmr::new_delete_resource()),
      instanceAutoLodCounts_(std::pmr::new_delete_resource()),
      materialTextureAccessHandles_(std::pmr::new_delete_resource()),
      instanceAutoLodLevels_(std::pmr::new_delete_resource()),
      instanceTessSelection_(resolveMemoryResource(memory)),
      tessCandidates_(resolveMemoryResource(memory)),
      instanceRemap_(std::pmr::new_delete_resource()),
      drawPushConstants_(resolveMemoryResource(memory)),
      drawItems_(resolveMemoryResource(memory)),
      drawAlphaMasked_(resolveMemoryResource(memory)),
      meshletBatchInfos_(resolveMemoryResource(memory)),
      indirectDrawItems_(resolveMemoryResource(memory)),
      indirectAlphaMasked_(resolveMemoryResource(memory)),
      indirectCommandUploadBytes_(resolveMemoryResource(memory)),
      overlayDrawItems_(resolveMemoryResource(memory)),
      velocityDrawItems_(resolveMemoryResource(memory)),
      reactiveMaskDrawItems_(resolveMemoryResource(memory)),
      pickDrawItems_(resolveMemoryResource(memory)),
      shadowInspectDrawItems_(resolveMemoryResource(memory)),
      passDrawItems_(resolveMemoryResource(memory)),
      msaaPassDrawItems_(resolveMemoryResource(memory)),
      depthPrepassDrawItems_(resolveMemoryResource(memory)),
      transmissionVisibilityDepthDrawItems_(resolveMemoryResource(memory)),
      normalPrepassDrawItems_(resolveMemoryResource(memory)),
      depthPyramidPushConstants_(resolveMemoryResource(memory)),
      depthPyramidDrawItems_(resolveMemoryResource(memory)),
      meshletDepthPrepassDispatchItems_(resolveMemoryResource(memory)),
      meshletDepthPrepassPushConstants_(resolveMemoryResource(memory)),
      meshletDepthPrepassDispatchDependencyBuffers_(
          resolveMemoryResource(memory)),
      meshletDepthPrepassDependencyBuffers_(resolveMemoryResource(memory)),
      meshletDepthPrepassDependencyBufferAccessModes_(
          resolveMemoryResource(memory)),
      meshletNormalPrepassDispatchItems_(resolveMemoryResource(memory)),
      meshletNormalPrepassPushConstants_(resolveMemoryResource(memory)),
      meshletNormalPrepassDispatchDependencyBuffers_(
          resolveMemoryResource(memory)),
      meshletNormalPrepassDependencyBuffers_(resolveMemoryResource(memory)),
      meshletNormalPrepassDependencyBufferAccessModes_(
          resolveMemoryResource(memory)),
      meshletDispatchItems_(resolveMemoryResource(memory)),
      meshletPushConstants_(resolveMemoryResource(memory)),
      meshletBatchGpuData_(resolveMemoryResource(memory)),
      meshletDispatchDependencyBuffers_(resolveMemoryResource(memory)),
      meshletVelocityDispatchItems_(resolveMemoryResource(memory)),
      meshletVelocityPushConstants_(resolveMemoryResource(memory)),
      meshletVelocityDispatchDependencyBuffers_(resolveMemoryResource(memory)),
      meshletReactiveMaskDispatchItems_(resolveMemoryResource(memory)),
      meshletReactiveMaskPushConstants_(resolveMemoryResource(memory)),
      meshletReactiveMaskDispatchDependencyBuffers_(
          resolveMemoryResource(memory)),
      depthPyramidDependencyTextures_(resolveMemoryResource(memory)),
      shadowSdsmReducePushConstants_(resolveMemoryResource(memory)),
      shadowSdsmReduceDispatches_(resolveMemoryResource(memory)),
      shadowSdsmReduceDependencyBuffers_(resolveMemoryResource(memory)),
      shadowSdsmReduceDependencyTextures_(resolveMemoryResource(memory)),
      preDispatches_(resolveMemoryResource(memory)),
      mainPreDispatches_(resolveMemoryResource(memory)),
      visibilityGpuDispatches_(resolveMemoryResource(memory)),
      visibilityGpuCandidates_(resolveMemoryResource(memory)),
      visibilityPassGpuData_(resolveMemoryResource(memory)),
      visibilityCounterClear_(resolveMemoryResource(memory)),
      visibilityGpuPushConstants_(resolveMemoryResource(memory)),
      visibilityIndirectDrawPushConstants_(resolveMemoryResource(memory)),
      visibilityMeshletDispatchGpuData_(resolveMemoryResource(memory)),
      visibilityMeshletCandidateMap_(resolveMemoryResource(memory)),
      visibilityIndirectMeshDispatchPushConstants_(
          resolveMemoryResource(memory)),
      visibilityMeshletGpuDispatches_(resolveMemoryResource(memory)),
      meshletCompactionWorkItems_(resolveMemoryResource(memory)),
      meshletCompactionPushConstants_(resolveMemoryResource(memory)),
      meshletCompactionDispatches_(resolveMemoryResource(memory)),
      meshletCompactionCounterClear_(resolveMemoryResource(memory)),
      meshletCompactionDependencyBuffers_(resolveMemoryResource(memory)),
      meshletCompactionFinalizeDependencyBuffers_(
          resolveMemoryResource(memory)),
      meshletCompactionDependencyTextures_(resolveMemoryResource(memory)),
      visibilityMeshletGpuDependencyBuffers_(resolveMemoryResource(memory)),
      visibilityMeshletGpuDependencyBufferAccessModes_(
          resolveMemoryResource(memory)),
      visibilityGpuDependencyBuffers_(resolveMemoryResource(memory)),
      visibilityGpuDependencyBufferAccessModes_(resolveMemoryResource(memory)),
      visibilityGpuDependencyTextures_(resolveMemoryResource(memory)),
      passDependencyBuffers_(resolveMemoryResource(memory)),
      passDependencyBufferAccessModes_(resolveMemoryResource(memory)),
      preResolvedDecodeBuffers_(std::pmr::new_delete_resource()),
      preResolvedDrawBuffers_(resolveMemoryResource(memory)),
      dispatchDependencyBuffers_(resolveMemoryResource(memory)),
      passDependencyTextures_(resolveMemoryResource(memory)),
      mainPassDependencyBuffers_(resolveMemoryResource(memory)),
      mainPassDependencyBufferAccessModes_(resolveMemoryResource(memory)),
      mainPassDependencyTextures_(resolveMemoryResource(memory)),
      mainPassDependencyTextureAccessModes_(resolveMemoryResource(memory)),
      velocityPassDependencyBuffers_(resolveMemoryResource(memory)),
      velocityPassDependencyBufferAccessModes_(resolveMemoryResource(memory)),
      reactivePassDependencyBuffers_(resolveMemoryResource(memory)),
      reactivePassDependencyBufferAccessModes_(resolveMemoryResource(memory)),
      previousTransformById_(std::pmr::new_delete_resource()),
      pendingPreviousTransformById_(std::pmr::new_delete_resource()),
      previousInstanceMatricesCpuCache_(resolveMemoryResource(memory)),
      velocityInstanceFlagsCpuCache_(resolveMemoryResource(memory)),
      velocityGeometryCpuCache_(resolveMemoryResource(memory)),
      transmissionVisibilityDepthPushConstants_(resolveMemoryResource(memory)),
      preparedGraphPasses_(resolveMemoryResource(memory)) {
  auto *resource = resolveMemoryResource(memory);
  bufferRings_.resize(BufferRingCount);
  singleInstanceBatchCaches_.reserve(kSingleInstanceCacheVariantCount);
  for (size_t i = 0; i < kSingleInstanceCacheVariantCount; ++i) {
    singleInstanceBatchCaches_.emplace_back(resource);
  }
}

OpaqueRenderer::~OpaqueRenderer() { onDetach(); }

void OpaqueRenderer::onAttach() {
  auto initResult = ensureInitialized();
  if (initResult.hasError() || !gpu_.supportsFeature(GPUFeature::Meshlets) ||
      !meshletPipelinesConfigured()) {
    return;
  }
  (void)createMeshletPipelineState();
}

void OpaqueRenderer::resetPickState() {
  pendingPickRequest_.reset();
  pendingShadowInspectRequest_.reset();
  inFlightPickReadback_.reset();
  inFlightShadowInspectReadback_.reset();
}

void OpaqueRenderer::stagePreviousTransforms(const RenderScene &scene,
                                             uint64_t frameIndex) {
  if (pendingPreviousTransformFrameIndex_ == frameIndex &&
      pendingPreviousTransformSceneId_ == scene.id()) {
    return;
  }
  const uint64_t topologyVersion = scene.topologyVersion();
  const uint64_t transformVersion = scene.transformVersion();
  pendingPreviousTransformDataChanged_ =
      previousTransformSceneId_ != scene.id() ||
      previousTransformCaptureTopologyVersion_ != topologyVersion ||
      previousTransformCaptureTransformVersion_ != transformVersion;
  if (pendingPreviousTransformDataChanged_) {
    const std::span<const Renderable> renderables = scene.renderables();
    pendingPreviousTransformById_.clear();
    pendingPreviousTransformById_.reserve(renderables.size());
    for (const Renderable &renderable : renderables) {
      if (nuri::isValid(renderable.id)) {
        pendingPreviousTransformById_[renderable.id] = renderable.modelMatrix;
      }
    }
  }
  pendingPreviousTransformSceneId_ = scene.id();
  pendingPreviousTransformFrameIndex_ = frameIndex;
  pendingPreviousTransformTopologyVersion_ = topologyVersion;
  pendingPreviousTransformTransformVersion_ = transformVersion;
}

void OpaqueRenderer::commitSubmittedFrame(uint64_t frameIndex) noexcept {
  if (pendingPreviousTransformFrameIndex_ != frameIndex) {
    return;
  }
  if (pendingPreviousTransformDataChanged_) {
    previousTransformById_.swap(pendingPreviousTransformById_);
    pendingPreviousTransformById_.clear();
  }
  previousTransformSceneId_ = pendingPreviousTransformSceneId_;
  previousTransformCaptureFrameIndex_ = frameIndex;
  previousTransformCaptureTopologyVersion_ =
      pendingPreviousTransformTopologyVersion_;
  previousTransformCaptureTransformVersion_ =
      pendingPreviousTransformTransformVersion_;
  pendingPreviousTransformFrameIndex_ = std::numeric_limits<uint64_t>::max();
  pendingPreviousTransformDataChanged_ = false;
}

void OpaqueRenderer::abandonPreparedFrame(uint64_t frameIndex) noexcept {
  if (pendingPreviousTransformFrameIndex_ != frameIndex) {
    return;
  }
  pendingPreviousTransformById_.clear();
  pendingPreviousTransformFrameIndex_ = std::numeric_limits<uint64_t>::max();
  pendingPreviousTransformDataChanged_ = false;
}

void OpaqueRenderer::onDetach() {
  destroyBuffers();
  destroyPickTexture();
  destroyShadowInspectTexture();
  destroyTransmissionVisibilityDepthTexture();
  destroyDepthPyramidTextures();
  if (nuri::isValid(sceneDepthSampler_)) {
    gpu_.destroySampler(sceneDepthSampler_);
    sceneDepthSampler_ = {};
  }
  resetOverlayPipelineState();
  destroyMeshletPipelineState();
  destroyMeshPipelineState();
  meshPipeline_.reset();
  computePipeline_.reset();
  visibilityComputePipeline_.reset();
  visibilityIndirectDrawComputePipeline_.reset();
  visibilityIndirectMeshDispatchComputePipeline_.reset();
  meshletCompactionComputePipeline_.reset();
  shaders_.fill({});
  tessellationUnsupported_ = false;
  clearAll(renderableTemplates_, meshDrawTemplates_, templateBatchIndices_,
           batchWriteOffsets_);
  clearAll(instanceCentersPhase_, instanceBaseMatrices_,
           instanceMatricesCpuCache_, previousInstanceMatricesCpuCache_);
  clearAll(velocityInstanceFlagsCpuCache_, velocityGeometryCpuCache_,
           instanceLodCentersInvRadiusSq_);
  clearAll(instanceAutoLodWorldErrors_, instanceAutoLodCounts_,
           materialTextureAccessHandles_, instanceAutoLodLevels_);
  materialTextureAccessCacheValid_ = false;
  clearAll(instanceTessSelection_, tessCandidates_, instanceRemap_,
           drawPushConstants_);
  clearAll(drawItems_, drawAlphaMasked_, meshletBatchInfos_);
  clearAll(indirectDrawItems_, indirectAlphaMasked_,
           indirectCommandUploadBytes_, overlayDrawItems_);
  clearAll(velocityDrawItems_, reactiveMaskDrawItems_, passDrawItems_,
           depthPrepassDrawItems_);
  clearAll(transmissionVisibilityDepthDrawItems_,
           transmissionVisibilityDepthPushConstants_,
           depthPyramidPushConstants_, depthPyramidDrawItems_);
  clearAll(meshletDispatchItems_, meshletPushConstants_, meshletBatchGpuData_,
           meshletDispatchDependencyBuffers_);
  clearAll(meshletNormalPrepassDispatchItems_,
           meshletNormalPrepassPushConstants_,
           meshletNormalPrepassDispatchDependencyBuffers_,
           meshletNormalPrepassDependencyBuffers_);
  clearAll(meshletNormalPrepassDependencyBufferAccessModes_,
           meshletVelocityDispatchItems_, meshletVelocityPushConstants_,
           meshletVelocityDispatchDependencyBuffers_);
  clearAll(meshletReactiveMaskDispatchItems_, meshletReactiveMaskPushConstants_,
           meshletReactiveMaskDispatchDependencyBuffers_,
           depthPyramidDependencyTextures_);
  clearAll(preDispatches_, mainPreDispatches_, visibilityGpuDispatches_,
           visibilityGpuCandidates_);
  clearAll(visibilityPassGpuData_, visibilityCounterClear_,
           visibilityGpuPushConstants_, visibilityIndirectDrawPushConstants_);
  clearAll(visibilityMeshletDispatchGpuData_, visibilityMeshletCandidateMap_,
           visibilityIndirectMeshDispatchPushConstants_,
           visibilityMeshletGpuDispatches_);
  clearAll(meshletCompactionWorkItems_, meshletCompactionPushConstants_,
           meshletCompactionDispatches_, meshletCompactionCounterClear_);
  clearAll(meshletCompactionDependencyBuffers_,
           meshletCompactionFinalizeDependencyBuffers_,
           meshletCompactionDependencyTextures_,
           visibilityMeshletGpuDependencyBuffers_);
  clearAll(visibilityMeshletGpuDependencyBufferAccessModes_,
           visibilityGpuDependencyBuffers_,
           visibilityGpuDependencyBufferAccessModes_,
           visibilityGpuDependencyTextures_);
  clearAll(visibilityCandidates_, visibilityCandidateGpuData_,
           cachedVisibleTemplateBatchIndices_, visibleBatchActiveRemap_);
  clearAll(cachedVisibleBatchEntries_, passDependencyBuffers_,
           passDependencyBufferAccessModes_, preResolvedDecodeBuffers_);
  clearAll(preResolvedDrawBuffers_, dispatchDependencyBuffers_,
           passDependencyTextures_, mainPassDependencyBuffers_);
  clearAll(mainPassDependencyBufferAccessModes_, mainPassDependencyTextures_,
           mainPassDependencyTextureAccessModes_,
           velocityPassDependencyBuffers_);
  clearAll(velocityPassDependencyBufferAccessModes_,
           reactivePassDependencyBuffers_,
           reactivePassDependencyBufferAccessModes_, previousTransformById_);
  pickDrawItems_.clear();
  setAll(
      std::numeric_limits<uint64_t>::max(), currentDirectDrawBufferSignature_,
      currentIndirectDrawBufferSignature_, cachedTopologyVersion_,
      cachedTransformVersion_, cachedMaterialVersion_,
      cachedModelMaterialBindingVersion_, cachedGeometryMutationVersion_,
      cachedVisibilityCandidateTopologyVersion_,
      cachedVisibilityCandidateTransformVersion_,
      cachedVisibilityCandidateDeformationVersion_,
      cachedVisibilityCandidateGeometryVersion_, cachedAnimationSceneVersion_,
      cachedVisibleBatchTopologyVersion_, cachedVisibleBatchMaterialVersion_,
      cachedVisibleBatchGeometryVersion_, previousTransformCaptureFrameIndex_,
      previousTransformCaptureTopologyVersion_,
      previousTransformCaptureTransformVersion_);
  resetAll(cachedScene_, cachedMeshletCounterValid_,
           cachedMeshletCounterSourceFrame_, cachedMeshletEmitted_,
           cachedMeshletTaskGroupsExecuted_,
           cachedVisibilityCandidatesHadDeformedRenderable_,
           cachedVisibleBatchValid_, cachedVisibleBatchMeshletRequested_,
           cachedVisibleBatchEnableMeshLod_, cachedAnimationSceneActive_,
           previousTransformSceneId_, uniformSingleSubmeshPath_,
           cachedRemapSignatureValid_, initialized_);
  cachedExcludeTransmission_ = true;
  cachedVisibleBatchForcedMeshLod_ = -1;
  instanceStaticBuffersDirty_ = true;
  invalidateAutoLodHistory();
  invalidateSingleInstanceBatchCache();
  invalidateIndirectPackCache();
  cachedRemapSignature_ = kInvalidDrawSignature;
  invalidateStaticBatchCache();
  resetPickState();
}

void OpaqueRenderer::onResize(uint32_t, uint32_t) {
  destroyPickTexture();
  destroyShadowInspectTexture();
  destroyTransmissionVisibilityDepthTexture();
  destroyDepthPyramidTextures();
  resetPickState();
}

void OpaqueRenderer::publishFrameData(RenderFrameContext &frame) {
  frame.sharedResources.sceneDepthSamplerId =
      gpu_.getDefaultSamplerBindlessIndex();
  frame.sharedResources.sceneDepthPyramidLevelCount = 0u;
  frame.sharedResources.sceneDepthPyramidTextures = {};
  frame.sharedResources.sceneDepthPyramidSourceFrameIndex.reset();
  frame.sharedResources.sceneDepthPyramidSourceViewProj.reset();
  const RenderSettings &settings = renderSettingsOrDefault(frame);
  if (!settings.opaque.enabled) {
    return;
  }
  const bool currentFrameVerificationRequired =
      settings.visibility.occlusionMode ==
      VisibilityOcclusionMode::CurrentFrameHiZExperimental;
  if (!requiresDepthPyramid(settings) || ensureInitialized().hasError() ||
      !nuri::isValid(depthPyramidPipelineHandle_) ||
      ensureDepthPyramidTextures(currentFrameVerificationRequired).hasError()) {
    return;
  }
  frame.sharedResources.sceneDepthSamplerId =
      gpu_.getSamplerBindlessIndex(sceneDepthSampler_);
  frame.sharedResources.sceneDepthPyramidLevelCount =
      sceneDepthPyramidLevelCount_;
  frame.sharedResources.sceneDepthPyramidTextures = sceneDepthPyramidTextures_;
  if (sceneDepthPyramidSourceFrameIndex_ && sceneDepthPyramidSourceViewProj_) {
    frame.sharedResources.sceneDepthPyramidSourceFrameIndex =
        sceneDepthPyramidSourceFrameIndex_;
    frame.sharedResources.sceneDepthPyramidSourceViewProj =
        sceneDepthPyramidSourceViewProj_;
  }
}

void OpaqueRenderer::readLatestVisibilityGpuReadback(
    RenderFrameContext &frame) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_COPY);
  VisibilityFrameMetrics &metrics = frame.metrics.visibility;
  std::optional<VisibilityCounterGpuData> selectedCounter;
  size_t selectedSlotIndex = std::numeric_limits<size_t>::max();
  uint32_t selectedSourceFrame = 0u;
  uint32_t counterReadbackErrorCount = 0u;
  const auto readCounterSlot = [&](size_t slotIndex) {
    const uint64_t expectedFrame =
        frameSlotStates_[slotIndex].visibilityPublishedFrame;
    if (expectedFrame == std::numeric_limits<uint64_t>::max()) {
      return;
    }
    const uint64_t safeReadbackAge = bufferRings_[VisibilityCounterRing].size();
    if (expectedFrame >= frame.frameIndex ||
        frame.frameIndex - expectedFrame < safeReadbackAge) {
      return;
    }
    const DynamicBufferSlot &slot =
        bufferRings_[VisibilityCounterRing][slotIndex];
    VisibilityCounterGpuData counter{};
    auto readResult =
        gpu_.readBuffer(slot.buffer->handle(), 0u,
                        std::as_writable_bytes(
                            std::span<VisibilityCounterGpuData>(&counter, 1u)));
    if (readResult.hasError()) {
      ++counterReadbackErrorCount;
      return;
    }
    const uint32_t valid = counter.status.w;
    const uint32_t sourceFrame = counter.status.z;
    if (valid == 0u || sourceFrame != static_cast<uint32_t>(expectedFrame)) {
      return;
    }
    if (!selectedCounter.has_value() || sourceFrame > selectedSourceFrame) {
      selectedCounter = counter;
      selectedSlotIndex = slotIndex;
      selectedSourceFrame = sourceFrame;
    }
  };
  for (size_t slotIndex = 0u;
       slotIndex < bufferRings_[VisibilityCounterRing].size(); ++slotIndex) {
    readCounterSlot(slotIndex);
  }
  metrics.gpuMainReadbackErrorCount = counterReadbackErrorCount;
  metrics.meshletReadbackErrorCount = counterReadbackErrorCount;
  if (!selectedCounter.has_value()) {
    if (cachedMeshletCounterValid_) {
      metrics.meshletEmitted = cachedMeshletEmitted_;
      metrics.meshletTaskGroupsExecuted = cachedMeshletTaskGroupsExecuted_;
      metrics.meshletReadbackSourceFrame = cachedMeshletCounterSourceFrame_;
      metrics.meshletReadbackStaleFrameCount = visibilityReadbackAge(
          frame.frameIndex, cachedMeshletCounterSourceFrame_);
    }
    return;
  }
  const uint32_t staleFrameCount =
      visibilityReadbackAge(frame.frameIndex, selectedSourceFrame);
  metrics.gpuMainCandidates = selectedCounter->main.x;
  metrics.gpuMainVisibleCandidates = selectedCounter->main.y;
  metrics.gpuMainRejectedFrustum = selectedCounter->main.z;
  metrics.gpuMainRejectedOcclusion = selectedCounter->main.w;
  metrics.gpuOutputOverflowCount = selectedCounter->status.x;
  metrics.occlusionAvailable = selectedCounter->status.y;
  metrics.meshletRejectedFrustum = selectedCounter->meshlet.x;
  metrics.meshletRejectedCone = selectedCounter->meshlet.y;
  metrics.meshletRejectedOcclusion = selectedCounter->meshlet.z;
  metrics.meshletPayloadOverflowCount = selectedCounter->meshlet.w;
  metrics.meshletEmitted = selectedCounter->meshlet2.x;
  metrics.meshletTaskGroupsExecuted = selectedCounter->meshlet2.y;
  metrics.meshletPreTaskCompactionActive =
      selectedCounter->meshlet3.z != 0u ? 1u : 0u;
  metrics.meshletPreTaskCandidatesInput = selectedCounter->meshlet3.x;
  metrics.meshletPreTaskCandidatesOutput = selectedCounter->meshlet3.y;
  metrics.meshletPreTaskTaskGroupsInput = selectedCounter->meshlet3.z;
  metrics.meshletPreTaskTaskGroupsOutput = selectedCounter->meshlet2.y;
  metrics.meshletPreTaskTaskGroupsSaved =
      selectedCounter->meshlet3.z > selectedCounter->meshlet2.y
          ? selectedCounter->meshlet3.z - selectedCounter->meshlet2.y
          : 0u;
  metrics.meshletPreTaskOverflowCount = selectedCounter->meshlet3.w;
  metrics.meshletPreTaskMismatchCount =
      selectedCounter->meshlet3.z != 0u &&
              selectedCounter->meshlet3.y != selectedCounter->meshlet2.x
          ? 1u
          : 0u;
  if (selectedCounter->meshlet2.y != 0u) {
    cachedMeshletCounterValid_ = true;
    cachedMeshletCounterSourceFrame_ = selectedSourceFrame;
    cachedMeshletEmitted_ = selectedCounter->meshlet2.x;
    cachedMeshletTaskGroupsExecuted_ = selectedCounter->meshlet2.y;
    metrics.meshletReadbackAvailable = 1u;
    metrics.meshletReadbackSourceFrame = selectedSourceFrame;
    metrics.meshletReadbackStaleFrameCount = staleFrameCount;
  }
  metrics.gpuMainReadbackAvailable = 1u;
  metrics.gpuMainReadbackSourceFrame = selectedSourceFrame;
  metrics.gpuMainReadbackStaleFrameCount = staleFrameCount;
  metrics.gpuMainReadbackVisibleCandidates = selectedCounter->main.y;
  metrics.gpuIndirectDrawReadbackCommands = selectedCounter->indirect.x;
  metrics.gpuIndirectDrawReadbackVisible = selectedCounter->indirect.y;
  metrics.gpuIndirectDrawReadbackTombstoned = selectedCounter->indirect.z;
  const bool validatesVisibleList =
      frameSlotStates_[selectedSlotIndex].expectedVisibleListValid != 0u;
  if (!validatesVisibleList || selectedCounter->status.y != 0u) {
    metrics.gpuMainVisibleListMismatches = 0u;
    return;
  }
  const DynamicBufferSlot &visibleSlot =
      bufferRings_[VisibilityVisibleIndexRing][selectedSlotIndex];
  const size_t visibleCapacity = visibleSlot.capacityBytes / sizeof(uint32_t);
  const size_t visibleCount =
      std::min(static_cast<size_t>(selectedCounter->main.y), visibleCapacity);
  visibilityVisibleIndexReadback_.assign(visibleCount, 0u);
  auto readResult =
      gpu_.readBuffer(visibleSlot.buffer->handle(), 0u,
                      std::as_writable_bytes(std::span<uint32_t>(
                          visibilityVisibleIndexReadback_.data(),
                          visibilityVisibleIndexReadback_.size())));
  if (readResult.hasError()) {
    ++metrics.gpuMainReadbackErrorCount;
    return;
  }
  const uint32_t expectedCount =
      frameSlotStates_[selectedSlotIndex].expectedVisibleCount;
  const uint64_t expectedHash =
      frameSlotStates_[selectedSlotIndex].expectedVisibleHash;
  const uint64_t readbackHash =
      hashSortedVisibilityVisibleIndexList(visibilityVisibleIndexReadback_);
  metrics.gpuMainVisibleListMismatches =
      selectedCounter->status.x != 0u ||
              selectedCounter->main.y != expectedCount ||
              visibleCount != static_cast<size_t>(expectedCount) ||
              readbackHash != expectedHash
          ? 1u
          : 0u;
}

Result<bool, std::string> OpaqueRenderer::appendGpuVisibilityMainPass(
    RenderFrameContext &frame, uint32_t frameSlot,
    std::span<const VisibilityCandidate> candidates,
    std::span<const VisibilityCandidateGpu> candidateGpuData,
    std::span<const uint32_t> candidateIndices,
    const VisibilityPassRequest &request,
    const VisibilityResolvedSettings &settings, bool validateVisibleList,
    std::pmr::vector<PreparedGraphPass> &out) {
  NURI_PROFILER_FUNCTION();
  if (candidateIndices.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }
  const bool visibilityListAvailable =
      nuri::isValid(visibilityComputePipeline_.get());
  const bool gpuIndirectRequested = settings.enableGpuIndirectDraw;
  const bool gpuIndirectPipelineAvailable =
      nuri::isValid(visibilityIndirectDrawComputePipeline_.get());
  if (!visibilityListAvailable &&
      !(gpuIndirectRequested && gpuIndirectPipelineAvailable)) {
    if (gpuIndirectRequested) {
      frame.metrics.visibility.gpuIndirectDrawFallback = 1u;
    }
    return Result<bool, std::string>::makeResult(true);
  }
  const uint32_t candidateCount = static_cast<uint32_t>(
      std::min(candidateIndices.size(),
               static_cast<size_t>(std::numeric_limits<uint32_t>::max())));
  bool gpuIndirectCandidateMapIdentity =
      candidateIndices.size() == candidates.size();
  if (gpuIndirectCandidateMapIdentity) {
    for (size_t i = 0; i < candidateIndices.size(); ++i) {
      if (candidateIndices[i] != i || candidates[i].renderableIndex != i) {
        gpuIndirectCandidateMapIdentity = false;
        break;
      }
    }
  }
  struct GpuIndirectDrawChunk {
    uint32_t commandWordOffset = 0u;
    uint32_t commandCount = 0u;
  };
  ScratchArena scratch;
  ScopedScratch scopedScratch(scratch);
  std::pmr::vector<GpuIndirectDrawChunk> gpuIndirectChunks(
      scopedScratch.resource());
  bool gpuIndirectSafe = gpuIndirectRequested && gpuIndirectPipelineAvailable &&
                         gpuIndirectCandidateMapIdentity &&
                         !indirectDrawItems_.empty();
  uint32_t gpuIndirectCommandCount = 0u;
  if (gpuIndirectSafe) {
    gpuIndirectChunks.reserve(indirectDrawItems_.size());
    for (const DrawItem &indirectDraw : indirectDrawItems_) {
      gpuIndirectChunks.push_back(GpuIndirectDrawChunk{
          .commandWordOffset = static_cast<uint32_t>(
              indirectDraw.indirectBufferOffset / sizeof(uint32_t)),
          .commandCount = indirectDraw.indirectDrawCount,
      });
      gpuIndirectCommandCount += indirectDraw.indirectDrawCount;
    }
  }
  if (gpuIndirectRequested) {
    if (gpuIndirectSafe) {
      frame.metrics.visibility.gpuIndirectDrawUsed = 1u;
      frame.metrics.visibility.gpuIndirectDrawCommands =
          gpuIndirectCommandCount;
    } else {
      frame.metrics.visibility.gpuIndirectDrawFallback = 1u;
    }
  }
  uint32_t visibleCapacity = candidateCount;
  if (settings.forcedVisibleListCapacity !=
      std::numeric_limits<uint32_t>::max()) {
    visibleCapacity =
        std::min(settings.forcedVisibleListCapacity, candidateCount);
  }
  const size_t candidateBytes =
      static_cast<size_t>(candidateCount) * sizeof(VisibilityCandidateGpu);
  const size_t visibleIndexBytes =
      static_cast<size_t>(visibleCapacity) * sizeof(uint32_t);
  auto ringResult =
      ensureVisibilityGpuRingCapacity(candidateBytes, visibleIndexBytes);
  if (ringResult.hasError()) {
    return ringResult;
  }
  if (frame.metrics.visibility.gpuMainReadbackAvailable == 0u) {
    frame.metrics.visibility.gpuMainCandidates = candidateCount;
    frame.metrics.visibility.gpuMainVisibleCandidates = candidateCount;
  }
  uint32_t expectedVisibleCount = 0u;
  uint64_t expectedVisibleHash = kFnvOffsetBasis64;
  {
    NURI_PROFILER_ZONE("OpaqueRenderer.visibility_gpu_candidate_pack",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    visibilityGpuCandidates_.clear();
    visibilityGpuCandidates_.reserve(candidateCount);
    std::pmr::vector<uint32_t> expectedVisibleIndices(
        visibilityGpuCandidates_.get_allocator().resource());
    if (validateVisibleList) {
      expectedVisibleIndices.reserve(candidateCount);
    }
    for (uint32_t i = 0u; i < candidateCount; ++i) {
      const uint32_t sourceIndex = candidateIndices[i];
      const VisibilityCandidate &candidate = candidates[sourceIndex];
      const VisibilityCandidateGpu &gpuCandidate =
          candidateGpuData[sourceIndex];
      visibilityGpuCandidates_.push_back(gpuCandidate);
      if (validateVisibleList) {
        bool expectedVisible = true;
        const bool uncertainVisible =
            (candidate.flags & kVisibilityCandidateConservativeVisible) != 0u &&
            settings.visibleOnUncertain;
        if (!uncertainVisible) {
          expectedVisible =
              visibility_detail::classifySphere(request.frustum,
                                                glm::vec3(gpuCandidate.bounds),
                                                gpuCandidate.bounds.w) !=
              visibility_detail::VisibilityClassification::Outside;
        }
        if (expectedVisible) {
          ++expectedVisibleCount;
          expectedVisibleIndices.push_back(candidate.templateIndex);
        }
      }
    }
    if (validateVisibleList) {
      expectedVisibleHash =
          hashSortedVisibilityVisibleIndexList(std::span<uint32_t>(
              expectedVisibleIndices.data(), expectedVisibleIndices.size()));
    }
    NURI_PROFILER_ZONE_END();
  }
  visibilityGpuDependencyTextures_.clear();
  std::array<glm::uvec4, kSceneDepthPyramidArraySize> depthPyramidTexIds{};
  uint32_t depthPyramidWidth = 0u;
  uint32_t depthPyramidHeight = 0u;
  uint32_t depthPyramidLevelCount = 0u;
  uint32_t depthPyramidSamplerId = 0u;
  bool occlusionAvailable = false;
  const bool wantsPreviousFrameOcclusion =
      settings.enableOcclusionCulling &&
      settings.occlusionMode == VisibilityOcclusionMode::PreviousFrameHiZ;
  if (wantsPreviousFrameOcclusion &&
      frame.sharedResources.sceneDepthPyramidSourceFrameIndex.has_value() &&
      frame.sharedResources.sceneDepthPyramidSourceViewProj.has_value() &&
      previousDepthPyramidCameraStable(frame) &&
      *frame.sharedResources.sceneDepthPyramidSourceFrameIndex + 1u ==
          frame.frameIndex) {
    const uint32_t candidateLevelCount =
        std::min<uint32_t>(frame.sharedResources.sceneDepthPyramidLevelCount,
                           kMaxSceneDepthPyramidLevels);
    visibilityGpuDependencyTextures_.reserve(candidateLevelCount);
    for (uint32_t level = 0u; level < candidateLevelCount; ++level) {
      const TextureHandle texture =
          frame.sharedResources.sceneDepthPyramidTextures[level];
      const uint32_t texId = gpu_.getTextureBindlessIndex(texture);
      const uint32_t packIndex = level / kSceneDepthPyramidTexIdPackWidth;
      const uint32_t componentIndex = level % kSceneDepthPyramidTexIdPackWidth;
      depthPyramidTexIds[packIndex][componentIndex] = texId;
      visibilityGpuDependencyTextures_.push_back(texture);
    }
    depthPyramidLevelCount = candidateLevelCount;
    const TextureDimensions dimensions = gpu_.getTextureDimensions(
        frame.sharedResources.sceneDepthPyramidTextures[0]);
    depthPyramidWidth = std::max(dimensions.width, 1u);
    depthPyramidHeight = std::max(dimensions.height, 1u);
    depthPyramidSamplerId = frame.sharedResources.sceneDepthSamplerId;
    occlusionAvailable = true;
  }
  visibilityPassGpuData_.clear();
  VisibilityPassGpuData mainVisibilityPassData =
      makeMainViewVisibilityPassGpuData(
          frame.camera, request, candidateCount, occlusionAvailable,
          depthPyramidWidth, depthPyramidHeight, depthPyramidLevelCount,
          depthPyramidSamplerId,
          std::span<const glm::uvec4>(depthPyramidTexIds.data(),
                                      depthPyramidTexIds.size()));
  if (occlusionAvailable) {
    mainVisibilityPassData.previousViewProj =
        *frame.sharedResources.sceneDepthPyramidSourceViewProj;
  }
  visibilityPassGpuData_.push_back(mainVisibilityPassData);
  visibilityCounterClear_.clear();
  visibilityCounterClear_.push_back(VisibilityCounterGpuData{});
  const BufferHandle candidateBuffer =
      bufferRings_[VisibilityCandidateRing][frameSlot].buffer->handle();
  const BufferHandle passBuffer =
      bufferRings_[VisibilityPassRing][frameSlot].buffer->handle();
  const BufferHandle visibleIndexBuffer =
      bufferRings_[VisibilityVisibleIndexRing][frameSlot].buffer->handle();
  const BufferHandle counterBuffer =
      bufferRings_[VisibilityCounterRing][frameSlot].buffer->handle();
  {
    NURI_PROFILER_ZONE("OpaqueRenderer.visibility_gpu_upload",
                       NURI_PROFILER_COLOR_CMD_COPY);
    const std::array updates{
        BufferUpdate{.buffer = candidateBuffer,
                     .data =
                         std::as_bytes(std::span<const VisibilityCandidateGpu>(
                             visibilityGpuCandidates_.data(),
                             visibilityGpuCandidates_.size()))},
        BufferUpdate{
            .buffer = passBuffer,
            .data = std::as_bytes(std::span<const VisibilityPassGpuData>(
                visibilityPassGpuData_.data(), visibilityPassGpuData_.size()))},
        BufferUpdate{
            .buffer = counterBuffer,
            .data = std::as_bytes(std::span<const VisibilityCounterGpuData>(
                visibilityCounterClear_.data(),
                visibilityCounterClear_.size()))},
    };
    auto updateResult = gpu_.updateBuffers(updates);
    if (updateResult.hasError()) {
      return updateResult;
    }
    NURI_PROFILER_ZONE_END();
  }
  frameSlotStates_[frameSlot].visibilityPublishedFrame = frame.frameIndex;
  frameSlotStates_[frameSlot].expectedVisibleCount = expectedVisibleCount;
  frameSlotStates_[frameSlot].expectedVisibleHash = expectedVisibleHash;
  frameSlotStates_[frameSlot].expectedVisibleListValid =
      validateVisibleList ? 1u : 0u;
  const uint64_t candidateAddress =
      gpu_.getBufferDeviceAddress(candidateBuffer);
  const uint64_t passAddress = gpu_.getBufferDeviceAddress(passBuffer);
  const uint64_t visibleIndexAddress =
      gpu_.getBufferDeviceAddress(visibleIndexBuffer);
  const uint64_t counterAddress = gpu_.getBufferDeviceAddress(counterBuffer);
  const BufferHandle indirectBuffer =
      gpuIndirectSafe
          ? bufferRings_[IndirectCommandRing][frameSlot].buffer->handle()
          : BufferHandle{};
  const BufferHandle remapBuffer =
      gpuIndirectSafe
          ? bufferRings_[InstanceRemapRing][frameSlot].buffer->handle()
          : BufferHandle{};
  const uint64_t indirectAddress =
      gpuIndirectSafe ? gpu_.getBufferDeviceAddress(indirectBuffer) : 0u;
  const uint64_t remapAddress =
      gpuIndirectSafe ? gpu_.getBufferDeviceAddress(remapBuffer) : 0u;
  if (gpuIndirectSafe && (indirectAddress == 0u || remapAddress == 0u)) {
    gpuIndirectSafe = false;
    frame.metrics.visibility.gpuIndirectDrawUsed = 0u;
    frame.metrics.visibility.gpuIndirectDrawCommands = 0u;
    frame.metrics.visibility.gpuIndirectDrawFallback = 1u;
  }
  if (candidateAddress == 0u || passAddress == 0u ||
      visibleIndexAddress == 0u || counterAddress == 0u) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (!visibilityListAvailable && !gpuIndirectSafe) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (gpuIndirectSafe) {
    const std::span<const std::byte> uploadBytes{
        indirectCommandUploadBytes_.data(), indirectCommandUploadBytes_.size()};
    auto updateIndirectResult =
        gpu_.updateBuffer(indirectBuffer, uploadBytes, 0u);
    if (updateIndirectResult.hasError()) {
      return updateIndirectResult;
    }
  }
  uint32_t flags = kVisibilityGpuFlagFrustumCulling;
  if (settings.visibleOnUncertain) {
    flags |= kVisibilityGpuFlagVisibleOnUncertain;
  }
  if (occlusionAvailable) {
    flags |= kVisibilityGpuFlagOcclusionCulling;
  }
  visibilityGpuPushConstants_.clear();
  if (visibilityListAvailable) {
    visibilityGpuPushConstants_.push_back(VisibilityGpuPushConstants{
        .candidateBufferAddress = candidateAddress,
        .passBufferAddress = passAddress,
        .visibleIndexBufferAddress = visibleIndexAddress,
        .counterBufferAddress = counterAddress,
        .candidateCount = candidateCount,
        .visibleCapacity = visibleCapacity,
        .flags = flags,
        .sourceFrameIndex = static_cast<uint32_t>(frame.frameIndex),
    });
  }
  visibilityIndirectDrawPushConstants_.clear();
  if (gpuIndirectSafe) {
    visibilityIndirectDrawPushConstants_.reserve(gpuIndirectChunks.size());
    for (const GpuIndirectDrawChunk &chunk : gpuIndirectChunks) {
      visibilityIndirectDrawPushConstants_.push_back(
          VisibilityIndirectDrawPushConstants{
              .commandBufferAddress = indirectAddress,
              .remapBufferAddress = remapAddress,
              .candidateBufferAddress = candidateAddress,
              .passBufferAddress = passAddress,
              .counterBufferAddress = counterAddress,
              .commandWordOffset = chunk.commandWordOffset,
              .commandCount = chunk.commandCount,
              .candidateCount = candidateCount,
              .flags = flags,
              .sourceFrameIndex = static_cast<uint32_t>(frame.frameIndex),
          });
    }
  }
  visibilityGpuDependencyBuffers_.clear();
  visibilityGpuDependencyBuffers_.reserve(gpuIndirectSafe ? 6u : 4u);
  visibilityGpuDependencyBuffers_.push_back(candidateBuffer);
  visibilityGpuDependencyBuffers_.push_back(passBuffer);
  visibilityGpuDependencyBuffers_.push_back(visibleIndexBuffer);
  visibilityGpuDependencyBuffers_.push_back(counterBuffer);
  if (gpuIndirectSafe) {
    visibilityGpuDependencyBuffers_.push_back(indirectBuffer);
    visibilityGpuDependencyBuffers_.push_back(remapBuffer);
  }
  visibilityGpuDependencyBufferAccessModes_.clear();
  visibilityGpuDependencyBufferAccessModes_.push_back(
      RenderGraphAccessMode::Read);
  visibilityGpuDependencyBufferAccessModes_.push_back(
      RenderGraphAccessMode::Read);
  visibilityGpuDependencyBufferAccessModes_.push_back(
      RenderGraphAccessMode::Read | RenderGraphAccessMode::Write);
  visibilityGpuDependencyBufferAccessModes_.push_back(
      RenderGraphAccessMode::Read | RenderGraphAccessMode::Write);
  if (gpuIndirectSafe) {
    visibilityGpuDependencyBufferAccessModes_.push_back(
        RenderGraphAccessMode::Read | RenderGraphAccessMode::Write);
    visibilityGpuDependencyBufferAccessModes_.push_back(
        RenderGraphAccessMode::Read);
  }
  visibilityGpuDispatches_.clear();
  visibilityGpuDispatches_.reserve(
      (visibilityListAvailable ? 1u : 0u) +
      static_cast<uint32_t>(visibilityIndirectDrawPushConstants_.size()));
  if (visibilityListAvailable) {
    ComputeDispatchItem dispatch{};
    dispatch.pipeline = visibilityComputePipeline_.get();
    dispatch.dispatch = {
        .x = (candidateCount + 63u) / 64u,
        .y = 1u,
        .z = 1u,
    };
    dispatch.pushConstants =
        std::as_bytes(std::span<const VisibilityGpuPushConstants>(
            visibilityGpuPushConstants_.data(),
            visibilityGpuPushConstants_.size()));
    dispatch.dependencyBuffers =
        std::span<const BufferHandle>(visibilityGpuDependencyBuffers_.data(),
                                      visibilityGpuDependencyBuffers_.size());
    dispatch.dependencyTextures =
        std::span<const TextureHandle>(visibilityGpuDependencyTextures_.data(),
                                       visibilityGpuDependencyTextures_.size());
    dispatch.debugLabel = "Opaque Visibility Cull";
    dispatch.debugColor = kComputeDispatchColor;
    visibilityGpuDispatches_.push_back(dispatch);
  }
  if (gpuIndirectSafe) {
    for (const VisibilityIndirectDrawPushConstants &pushConstants :
         visibilityIndirectDrawPushConstants_) {
      ComputeDispatchItem dispatch{};
      dispatch.pipeline = visibilityIndirectDrawComputePipeline_.get();
      dispatch.dispatch = {
          .x = (pushConstants.commandCount + 63u) / 64u,
          .y = 1u,
          .z = 1u,
      };
      dispatch.pushConstants =
          std::as_bytes(std::span<const VisibilityIndirectDrawPushConstants>(
              &pushConstants, 1u));
      dispatch.dependencyBuffers =
          std::span<const BufferHandle>(visibilityGpuDependencyBuffers_.data(),
                                        visibilityGpuDependencyBuffers_.size());
      dispatch.dependencyTextures = std::span<const TextureHandle>(
          visibilityGpuDependencyTextures_.data(),
          visibilityGpuDependencyTextures_.size());
      dispatch.debugLabel = "Opaque Visibility Indirect Draw";
      dispatch.debugColor = kComputeDispatchColor;
      visibilityGpuDispatches_.push_back(dispatch);
    }
  }
  PreparedGraphPass &visibilityPass =
      out.emplace_back(drawItems_.get_allocator().resource());
  visibilityPass.desc.executionMode = RenderPassExecutionMode::ComputeOnly;
  visibilityPass.desc.hasColorAttachment = false;
  visibilityPass.desc.preDispatches = std::span<const ComputeDispatchItem>(
      visibilityGpuDispatches_.data(), visibilityGpuDispatches_.size());
  visibilityPass.desc.dependencyBuffers =
      std::span<const BufferHandle>(visibilityGpuDependencyBuffers_.data(),
                                    visibilityGpuDependencyBuffers_.size());
  visibilityPass.desc.dependencyBufferAccessModes =
      std::span<const RenderGraphAccessMode>(
          visibilityGpuDependencyBufferAccessModes_.data(),
          visibilityGpuDependencyBufferAccessModes_.size());
  visibilityPass.desc.dependencyTextures =
      std::span<const TextureHandle>(visibilityGpuDependencyTextures_.data(),
                                     visibilityGpuDependencyTextures_.size());
  visibilityPass.desc.debugLabel = "Opaque Visibility Culling";
  visibilityPass.desc.debugColor = kComputeDispatchColor;
  visibilityPass.desc.gpuTimingScope = GpuTimingScope::Opaque;
  visibilityPass.desc.markImplicitOutputSideEffect = true;
  visibilityPass.desc.borrowPayload = false;
  visibilityPass.phase = PreparedPassPhase::PreLighting;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
OpaqueRenderer::buildOpaquePasses(RenderFrameContext &frame,
                                  std::pmr::vector<PreparedGraphPass> &out) {
  NURI_PROFILER_FUNCTION();
  frame.metrics.opaque = {};
  const auto shadowCpuCandidates = frame.metrics.visibility.shadowCpuCandidates;
  const auto shadowCpuRejected = frame.metrics.visibility.shadowCpuRejected;
  frame.metrics.visibility = {};
  frame.metrics.visibility.shadowCpuCandidates = shadowCpuCandidates;
  frame.metrics.visibility.shadowCpuRejected = shadowCpuRejected;
  const GpuTimingReport &timingReport = frame.gpuTiming;
  if (hasGpuTimingScope(timingReport, GpuTimingScope::Opaque)) {
    frame.metrics.opaque.gpuTimeMs = timingReport.opaqueTimeMs;
    frame.metrics.opaque.gpuTimingSourceFrameIndex =
        timingReport.opaqueSourceFrameIndex;
    frame.metrics.opaque.gpuTimingAvailable = 1u;
  }
  frame.opaquePickResult.reset();
  if (frame.opaquePickRequest.has_value()) {
    pendingPickRequest_ = frame.opaquePickRequest;
    frame.opaquePickRequest.reset();
  }
  frame.shadowInspectResult.reset();
  if (frame.shadowInspectRequest.has_value()) {
    pendingShadowInspectRequest_ = frame.shadowInspectRequest;
    frame.shadowInspectRequest.reset();
  }
  const RenderSettings &settings = renderSettingsOrDefault(frame);
  if (!settings.opaque.enabled) {
    return Result<bool, std::string>::makeResult(true);
  }
  const SceneDrawDatabase &drawDatabase =
      *frame.sharedResources.sceneDrawDatabase;
  const MaterialTableSnapshot materialSnapshot =
      frame.resources->materialSnapshot();
  auto initResult = ensureInitialized();
  if (initResult.hasError()) {
    return Result<bool, std::string>::makeError(initResult.error());
  }
  const TextureHandle sceneDepthTexture = resolveFrameDepthTexture(frame);
  frame.sharedResources.transmissionVisibilityDepthTexture = {};
  frame.sharedResources.transmissionVisibilityDepthGraphTexture = {};
  const bool needsPickResources =
      pendingPickRequest_.has_value() || inFlightPickReadback_.has_value();
  if (needsPickResources && !nuri::isValid(pickIdTexture_)) {
    auto pickTextureResult = recreatePickTexture();
    if (pickTextureResult.hasError()) {
      return pickTextureResult;
    }
  }
  const bool needsShadowInspectResources =
      pendingShadowInspectRequest_.has_value() ||
      inFlightShadowInspectReadback_.has_value();
  if (needsShadowInspectResources && !nuri::isValid(shadowInspectTexture_)) {
    auto inspectTextureResult = recreateShadowInspectTexture();
    if (inspectTextureResult.hasError()) {
      return inspectTextureResult;
    }
  }
  if (inFlightPickReadback_.has_value() &&
      frame.frameIndex > inFlightPickReadback_->submissionFrame) {
    NURI_PROFILER_ZONE("OpaqueRenderer.pick_readback",
                       NURI_PROFILER_COLOR_CMD_COPY);
    std::array<std::byte, sizeof(uint32_t)> pickBytes{};
    const TextureReadbackRegion readbackRegion{
        .x = inFlightPickReadback_->request.x,
        .y = inFlightPickReadback_->request.y,
        .width = 1,
        .height = 1,
        .mipLevel = 0,
        .layer = 0,
    };
    auto readResult =
        gpu_.readTexture(pickIdTexture_, readbackRegion, pickBytes);
    if (!readResult.hasError()) {
      uint32_t encodedId = 0;
      std::memcpy(&encodedId, pickBytes.data(), sizeof(encodedId));
      OpaquePickResult result{};
      result.requestId = inFlightPickReadback_->request.requestId;
      result.hit = encodedId > 0;
      result.renderableIndex = result.hit ? (encodedId - 1u) : 0u;
      frame.opaquePickResult = result;
    }
    inFlightPickReadback_.reset();
    NURI_PROFILER_ZONE_END();
  }
  if (inFlightShadowInspectReadback_.has_value() &&
      frame.frameIndex > inFlightShadowInspectReadback_->submissionFrame) {
    NURI_PROFILER_ZONE("OpaqueRenderer.shadow_inspect_readback",
                       NURI_PROFILER_COLOR_CMD_COPY);
    std::array<std::byte, sizeof(float) * 4u> inspectBytes{};
    const TextureReadbackRegion readbackRegion{
        .x = inFlightShadowInspectReadback_->request.x,
        .y = inFlightShadowInspectReadback_->request.y,
        .width = 1,
        .height = 1,
        .mipLevel = 0,
        .layer = 0,
    };
    auto readResult =
        gpu_.readTexture(shadowInspectTexture_, readbackRegion, inspectBytes);
    if (!readResult.hasError()) {
      std::array<float, 4> values{};
      std::memcpy(values.data(), inspectBytes.data(), sizeof(values));
      ShadowInspectResult result{};
      result.requestId = inFlightShadowInspectReadback_->request.requestId;
      result.valid = values[3] >= 0.0f;
      result.receiverDepth = values[0];
      result.receiverCompareDepth = values[1];
      result.sampledDepth = values[2];
      if (result.valid) {
        const float packedCascadeState = values[3];
        const float cascadeIndex = std::floor(packedCascadeState + 1.0e-5f);
        result.cascadeIndex =
            static_cast<uint32_t>(std::max(cascadeIndex, 0.0f));
        result.cascadeBlendFactor =
            std::clamp((packedCascadeState - cascadeIndex) * 8.0f, 0.0f, 1.0f);
      }
      frame.shadowInspectResult = result;
    }
    inFlightShadowInspectReadback_.reset();
    NURI_PROFILER_ZONE_END();
  }
  bool topologyDirty = cachedScene_ != frame.scene ||
                       cachedTopologyVersion_ != frame.scene->topologyVersion();
  const uint64_t modelMaterialBindingVersion =
      frame.resources->modelMaterialBindingVersion();
  bool materialDirty =
      topologyDirty || cachedScene_ != frame.scene ||
      cachedMaterialVersion_ != materialSnapshot.version ||
      cachedModelMaterialBindingVersion_ != modelMaterialBindingVersion;
  const bool excludeTransmission = true;
  bool transmissionPolicyDirty =
      cachedExcludeTransmission_ != excludeTransmission;
  const uint64_t geometryMutationVersion = gpu_.geometryMutationVersion();
  const bool hasGeometryMutationTracking = geometryMutationVersion != 0;
  const bool geometryDirty =
      !hasGeometryMutationTracking ||
      cachedGeometryMutationVersion_ != geometryMutationVersion;
  if (topologyDirty || materialDirty || transmissionPolicyDirty ||
      geometryDirty) {
    rebuildSceneCache(drawDatabase, *frame.scene, excludeTransmission);
    if (hasGeometryMutationTracking) {
      cachedGeometryMutationVersion_ = geometryMutationVersion;
    }
  }
  const bool transformDirty =
      topologyDirty ||
      cachedTransformVersion_ != frame.scene->transformVersion();
  if (topologyDirty) {
    invalidateAutoLodHistory();
  }
  const size_t instanceCount = renderableTemplates_.size();
  const uint32_t swapchainImageCount =
      std::max(1u, gpu_.getSwapchainImageCount());
  auto ringResult = ensureRingBufferCount(swapchainImageCount);
  if (ringResult.hasError()) {
    return ringResult;
  }
  const uint32_t frameSlot =
      static_cast<uint32_t>(frame.frameIndex % swapchainImageCount);
  readLatestVisibilityGpuReadback(frame);
  bool visibilityCounterPreparedForFrame = false;
  const AnimationSceneFrameData *animationSceneData =
      resolveAnimationSceneFrameData(frame);
  if (topologyDirty || transformDirty) {
    instanceCentersPhase_.clear();
    instanceBaseMatrices_.clear();
    instanceMatricesCpuCache_.clear();
    instanceLodCentersInvRadiusSq_.clear();
    instanceAutoLodWorldErrors_.clear();
    instanceAutoLodCounts_.clear();
    instanceCentersPhase_.reserve(instanceCount);
    instanceBaseMatrices_.reserve(instanceCount);
    instanceMatricesCpuCache_.reserve(instanceCount);
    instanceLodCentersInvRadiusSq_.reserve(instanceCount);
    instanceAutoLodWorldErrors_.reserve(instanceCount);
    instanceAutoLodCounts_.reserve(instanceCount);
    const bool animateInstances = settings.opaque.enableInstanceAnimation;
    for (size_t i = 0; i < instanceCount; ++i) {
      const RenderableTemplate &templ = renderableTemplates_[i];
      const Renderable *renderable = templ.renderable;
      const Model *model = templ.model;
      const glm::vec3 center = glm::vec3(renderable->modelMatrix[3]);
      instanceCentersPhase_.push_back(
          glm::vec4(center, animateInstances
                                ? deterministicPhase(static_cast<uint32_t>(i))
                                : 0.0f));
      glm::mat4 baseMatrix = renderable->modelMatrix;
      baseMatrix[3] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
      instanceBaseMatrices_.push_back(baseMatrix);
      instanceMatricesCpuCache_.push_back(
          makeInstanceData(renderable->modelMatrix));
      const BoundingBox &bounds = model->bounds();
      const glm::vec3 localCenter = bounds.getCenter();
      const float localRadius =
          kBoundsRadiusHalf * glm::length(bounds.getSize());
      const glm::vec3 worldCenter =
          glm::vec3(renderable->modelMatrix * glm::vec4(localCenter, 1.0f));
      const float worldScale = maxAxisScale(renderable->modelMatrix);
      const float worldRadius =
          std::max(localRadius * worldScale, kMinLodRadius);
      const float invRadiusSq = 1.0f / (worldRadius * worldRadius);
      instanceLodCentersInvRadiusSq_.push_back(
          glm::vec4(worldCenter, invRadiusSq));
      glm::vec4 worldErrors(0.0f);
      uint32_t availableLodCount = 1u;
      for (const Submesh &submesh : model->submeshes()) {
        const uint32_t stableLodCount = std::min(
            submesh.lodCount, detail::kMaxStableGeneratedOpaqueLod + 1u);
        availableLodCount = std::max(availableLodCount, stableLodCount);
        for (uint32_t lod = 1u; lod < stableLodCount; ++lod) {
          worldErrors[lod] =
              std::max(worldErrors[lod],
                       std::max(submesh.lods[lod].error, 0.0f) * worldScale);
        }
      }
      instanceAutoLodWorldErrors_.push_back(worldErrors);
      instanceAutoLodCounts_.push_back(static_cast<uint8_t>(availableLodCount));
    }
    cachedTransformVersion_ = frame.scene->transformVersion();
    instanceStaticBuffersDirty_ = true;
    for (FrameSlotState &slot : frameSlotStates_) {
      slot.matricesUploadVersion = std::numeric_limits<uint64_t>::max();
    }
  }
  const bool animationSceneStateDirty =
      topologyDirty || geometryDirty ||
      cachedAnimationSceneActive_ != (animationSceneData != nullptr) ||
      (animationSceneData != nullptr &&
       cachedAnimationSceneVersion_ != animationSceneData->version);
  if (animationSceneStateDirty) {
    bool vertexAddressChanged = false;
    bool hasAnimatedGeometry = false;
    for (MeshDrawTemplate &templateEntry : meshDrawTemplates_) {
      BufferHandle resolvedVertexDecodeBuffer =
          templateEntry.baseVertexDecodeBuffer;
      uint64_t resolvedVertexDecodeBufferAddress =
          templateEntry.baseVertexDecodeBufferAddress;
      uint32_t resolvedVertexDecodeIndex = templateEntry.submeshIndex;
      uint32_t resolvedPackedVertexFormat =
          templateEntry.basePackedVertexFormat;
      uint64_t resolvedVertexBufferAddress =
          templateEntry.baseVertexBufferAddress != 0u
              ? templateEntry.baseVertexBufferAddress
              : templateEntry.vertexBufferAddress;
      BufferHandle resolvedVertexBuffer = templateEntry.baseVertexBuffer;
      if (animationSceneData != nullptr) {
        const AnimatedRenderableGeometryOverride &geometryOverride =
            animationSceneData
                ->geometryOverridesByRenderable[templateEntry.instanceIndex];
        if (geometryOverride.enabled &&
            nuri::isValid(geometryOverride.vertexBuffer) &&
            animationOverrideCoversSubmesh(geometryOverride,
                                           *templateEntry.submesh)) {
          const uint64_t overrideVertexAddress = gpu_.getBufferDeviceAddress(
              geometryOverride.vertexBuffer, geometryOverride.vertexByteOffset);
          resolvedVertexBuffer = geometryOverride.vertexBuffer;
          resolvedVertexDecodeBuffer = {};
          resolvedVertexBufferAddress = overrideVertexAddress;
          resolvedVertexDecodeBufferAddress = 0u;
          resolvedVertexDecodeIndex = 0u;
          resolvedPackedVertexFormat =
              static_cast<uint32_t>(PackedVertexFormat::AnimatedFloat32);
          hasAnimatedGeometry = true;
        }
      }
      vertexAddressChanged |=
          !isSameBufferHandle(templateEntry.vertexBuffer,
                              resolvedVertexBuffer) ||
          !isSameBufferHandle(templateEntry.vertexDecodeBuffer,
                              resolvedVertexDecodeBuffer) ||
          templateEntry.vertexBufferAddress != resolvedVertexBufferAddress ||
          templateEntry.vertexDecodeBufferAddress !=
              resolvedVertexDecodeBufferAddress ||
          templateEntry.vertexDecodeIndex != resolvedVertexDecodeIndex ||
          templateEntry.packedVertexFormat != resolvedPackedVertexFormat;
      templateEntry.vertexBuffer = resolvedVertexBuffer;
      templateEntry.vertexDecodeBuffer = resolvedVertexDecodeBuffer;
      templateEntry.vertexBufferAddress = resolvedVertexBufferAddress;
      templateEntry.vertexDecodeBufferAddress =
          resolvedVertexDecodeBufferAddress;
      templateEntry.vertexDecodeIndex = resolvedVertexDecodeIndex;
      templateEntry.packedVertexFormat = resolvedPackedVertexFormat;
    }
    if (vertexAddressChanged) {
      invalidateSingleInstanceBatchCache();
      invalidateStaticBatchCache();
      invalidateIndirectPackCache();
    }
    if (hasAnimatedGeometry) {
      uniformSingleSubmeshPath_ = false;
    }
    cachedAnimationSceneActive_ = animationSceneData != nullptr;
    cachedAnimationSceneVersion_ = animationSceneData != nullptr
                                       ? animationSceneData->version
                                       : std::numeric_limits<uint64_t>::max();
  }
  if (topologyDirty || animationSceneStateDirty) {
    preResolvedDecodeBuffers_.clear();
    preResolvedDecodeBuffers_.reserve(meshDrawTemplates_.size());
    for (const MeshDrawTemplate &templateEntry : meshDrawTemplates_) {
      if (templateEntry.packedVertexFormat !=
              static_cast<uint32_t>(PackedVertexFormat::StaticQuantized20) ||
          !nuri::isValid(templateEntry.vertexDecodeBuffer)) {
        continue;
      }
      appendUniqueDependency(preResolvedDecodeBuffers_,
                             templateEntry.vertexDecodeBuffer);
    }
  }
  const ForwardSceneGpuData *sceneGpu =
      &*frame.sharedResources.forwardSceneGpuData;
  const MaterialTableGpuData *materialGpu =
      &*frame.sharedResources.materialTableGpuData;
  auto staticBuffers = ensureStaticInstanceBufferCapacity(instanceCount);
  if (staticBuffers.hasError()) {
    return staticBuffers;
  }
  auto matricesResult = ensureInstanceMatricesRingCapacity(
      std::max(instanceCount * sizeof(InstanceData), sizeof(InstanceData)));
  if (matricesResult.hasError()) {
    return matricesResult;
  }
  const PresentationAAPlan presentationAA = presentationAAPlanForFrame(frame);
  const bool temporalMotionRequired = presentationAA.needsMotion;
  const bool reactiveMaskRequired = presentationAA.needsReactiveMask;
  const CoverageMode coverage = presentationAA.coverage;
  const bool msaaSelected = coverage != CoverageMode::Sample1;
  RenderSettings::AmbientOcclusionSettings ambientOcclusionSettings =
      settings.ambientOcclusion;
  const auto requestedAmbientOcclusionSettings = ambientOcclusionSettings;
  const AmbientOcclusionExecutionPlan &ambientOcclusionPlan =
      frame.ambientOcclusion;
  ambientOcclusionSettings.active = ambientOcclusionPlan.active;
  ambientOcclusionSettings.temporalAccumulation = ambientOcclusionPlan.temporal;
  ambientOcclusionSettings.preset = ambientOcclusionPlan.preset;
  ambientOcclusionSettings.sliceCount = ambientOcclusionPlan.sliceCount;
  ambientOcclusionSettings.stepCount = ambientOcclusionPlan.stepCount;
  ambientOcclusionSettings.denoisePassCount =
      ambientOcclusionPlan.denoisePassCount;
  AmbientOcclusionFrameMetrics &aoMetrics = frame.metrics.ambientOcclusion;
  aoMetrics.enabled =
      ambientOcclusionSettings.mode != AmbientOcclusionMode::Disabled;
  aoMetrics.inputMode = ambientOcclusionPlan.inputMode;
  aoMetrics.workingResolution = ambientOcclusionPlan.workingResolution;
  aoMetrics.activePreset = ambientOcclusionSettings.preset;
  aoMetrics.strength = ambientOcclusionSettings.strength;
  aoMetrics.requestedSliceCount = requestedAmbientOcclusionSettings.sliceCount;
  aoMetrics.requestedStepCount = requestedAmbientOcclusionSettings.stepCount;
  aoMetrics.requestedDenoisePassCount =
      requestedAmbientOcclusionSettings.denoisePassCount;
  aoMetrics.sliceCount = ambientOcclusionSettings.sliceCount;
  aoMetrics.stepCount = ambientOcclusionSettings.stepCount;
  aoMetrics.denoisePassCount = ambientOcclusionSettings.denoisePassCount;
  aoMetrics.temporalAccumulationEnabled =
      ambientOcclusionSettings.temporalAccumulation;
  aoMetrics.disabledReason = ambientOcclusionSettings.disabledReason;
  aoMetrics.active = ambientOcclusionSettings.active;
  const bool hasTaaVelocityInstances =
      temporalMotionRequired && instanceCount > 0;
  const bool animationPreviousFrameValid =
      hasTaaVelocityInstances && animationSceneData != nullptr &&
      hasTemporalCameraContinuity(frame.camera) &&
      frame.camera.temporalDataValid &&
      nuri::isValid(animationSceneData->previousInstanceMatricesBuffer) &&
      animationSceneData->previousInstanceMatricesAddress != 0u;
  const bool transformPreviousCacheValid =
      hasTaaVelocityInstances && hasTemporalCameraContinuity(frame.camera) &&
      frame.camera.temporalDataValid &&
      previousTransformSceneId_ == frame.scene->id() &&
      previousTransformCaptureFrameIndex_ !=
          std::numeric_limits<uint64_t>::max() &&
      previousTransformCaptureFrameIndex_ < frame.frameIndex;
  const bool previousCacheValid = animationSceneData != nullptr
                                      ? animationPreviousFrameValid
                                      : transformPreviousCacheValid;
  const bool builtInAnimationPreviousFrameValid =
      hasTaaVelocityInstances && settings.opaque.enableInstanceAnimation &&
      animationSceneData == nullptr && transformPreviousCacheValid &&
      previousTransformCaptureFrameIndex_ + 1u == frame.frameIndex &&
      previousTransformCaptureTopologyVersion_ ==
          frame.scene->topologyVersion() &&
      previousTransformCaptureTransformVersion_ ==
          frame.scene->transformVersion() &&
      std::isfinite(frame.timeSeconds) && std::isfinite(frame.deltaSeconds);
  const bool staticVelocityScene = hasTaaVelocityInstances &&
                                   !settings.opaque.enableInstanceAnimation &&
                                   animationSceneData == nullptr;
  const bool canReuseStaticPreviousMatrices =
      staticVelocityScene && transformPreviousCacheValid && !transformDirty &&
      !animationSceneStateDirty;
  const bool canUseAllInvalidVelocityFlags =
      staticVelocityScene && !transformPreviousCacheValid;
  VelocityInstanceFlagsMode velocityInstanceFlagsMode =
      VelocityInstanceFlagsMode::Buffer;
  if (animationSceneData != nullptr) {
    velocityInstanceFlagsMode = animationPreviousFrameValid
                                    ? VelocityInstanceFlagsMode::AllValid
                                    : VelocityInstanceFlagsMode::AllInvalid;
  } else if (canReuseStaticPreviousMatrices) {
    velocityInstanceFlagsMode = VelocityInstanceFlagsMode::AllValid;
  } else if (canUseAllInvalidVelocityFlags) {
    velocityInstanceFlagsMode = VelocityInstanceFlagsMode::AllInvalid;
  }
  const bool needsVelocityInstanceBufferUpload =
      hasTaaVelocityInstances &&
      velocityInstanceFlagsMode == VelocityInstanceFlagsMode::Buffer;
  const bool needsVelocityGeometryUpload =
      hasTaaVelocityInstances && animationPreviousFrameValid &&
      !animationSceneData->previousGeometryOverridesByRenderable.empty();
  const TextureHandle sceneDepthTarget =
      msaaSelected ? frame.sharedResources.msaaSceneDepthTexture
                   : sceneDepthTexture;
  if (needsVelocityInstanceBufferUpload) {
    auto previousMatricesResult = ensurePreviousInstanceMatricesRingCapacity(
        std::max(instanceCount * sizeof(InstanceData), sizeof(InstanceData)));
    if (previousMatricesResult.hasError()) {
      return previousMatricesResult;
    }
    auto velocityFlagsResult = ensureVelocityInstanceFlagsRingCapacity(
        std::max(instanceCount * sizeof(uint32_t), sizeof(uint32_t)));
    if (velocityFlagsResult.hasError()) {
      return velocityFlagsResult;
    }
  }
  if (hasTaaVelocityInstances) {
    auto velocityFrameResult =
        ensureVelocityFrameDataRingCapacity(sizeof(VelocityFrameGpuData));
    if (velocityFrameResult.hasError()) {
      return velocityFrameResult;
    }
  }
  if (needsVelocityGeometryUpload) {
    auto velocityGeometryResult = ensureVelocityGeometryRingCapacity(
        std::max(instanceCount * sizeof(VelocityRenderableGeometryGpuData),
                 sizeof(VelocityRenderableGeometryGpuData)));
    if (velocityGeometryResult.hasError()) {
      return velocityGeometryResult;
    }
  }
  if (instanceStaticBuffersDirty_) {
    std::array<BufferUpdate, 3u> updates{};
    size_t updateCount = 0u;
    if (!instanceCentersPhase_.empty()) {
      const std::span<const std::byte> centersBytes{
          reinterpret_cast<const std::byte *>(instanceCentersPhase_.data()),
          instanceCentersPhase_.size() * sizeof(glm::vec4)};
      updates[updateCount++] = BufferUpdate{
          .buffer = instanceCentersPhaseBuffer_->handle(),
          .data = centersBytes,
      };
    }
    if (!instanceBaseMatrices_.empty()) {
      const std::span<const std::byte> baseMatricesBytes{
          reinterpret_cast<const std::byte *>(instanceBaseMatrices_.data()),
          instanceBaseMatrices_.size() * sizeof(glm::mat4)};
      updates[updateCount++] = BufferUpdate{
          .buffer = instanceBaseMatricesBuffer_->handle(),
          .data = baseMatricesBytes,
      };
    }
    if (!instanceLodCentersInvRadiusSq_.empty()) {
      const std::span<const std::byte> lodBoundsBytes{
          reinterpret_cast<const std::byte *>(
              instanceLodCentersInvRadiusSq_.data()),
          instanceLodCentersInvRadiusSq_.size() * sizeof(glm::vec4)};
      updates[updateCount++] = BufferUpdate{
          .buffer = instanceLodBoundsBuffer_->handle(),
          .data = lodBoundsBytes,
      };
    }
    if (updateCount != 0u) {
      auto updateResult = gpu_.updateBuffers(
          std::span<const BufferUpdate>(updates.data(), updateCount));
      if (updateResult.hasError()) {
        return updateResult;
      }
    }
    instanceStaticBuffersDirty_ = false;
  }
  if (materialDirty) {
    cachedMaterialVersion_ = materialSnapshot.version;
    cachedModelMaterialBindingVersion_ = modelMaterialBindingVersion;
  }
  if (materialDirty || transmissionPolicyDirty ||
      !materialTextureAccessCacheValid_) {
    NURI_PROFILER_ZONE("OpaqueRenderer.material_access_cache",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    auto materialAccessCacheResult = rebuildMaterialTextureAccessCache(
        drawDatabase, *frame.resources, excludeTransmission);
    if (materialAccessCacheResult.hasError()) {
      return materialAccessCacheResult;
    }
    NURI_PROFILER_ZONE_END();
  }
  const uint64_t frameDataAddress = sceneGpu->frameDataAddress;
  const uint64_t instanceCentersPhaseAddress =
      gpu_.getBufferDeviceAddress(instanceCentersPhaseBuffer_->handle());
  const uint64_t instanceLodBoundsAddress =
      gpu_.getBufferDeviceAddress(instanceLodBoundsBuffer_->handle());
  const uint64_t instanceBaseMatricesAddress =
      gpu_.getBufferDeviceAddress(instanceBaseMatricesBuffer_->handle());
  const uint64_t directionalLightBufferAddress =
      sceneGpu->directionalLightBufferAddress;
  const uint64_t localLightBufferAddress = sceneGpu->localLightBufferAddress;
  const BufferHandle instanceMatricesBufferHandle =
      animationSceneData != nullptr
          ? animationSceneData->instanceMatricesBuffer
          : bufferRings_[InstanceMatricesRing][frameSlot].buffer->handle();
  const uint64_t instanceMatricesAddress =
      animationSceneData != nullptr
          ? animationSceneData->instanceMatricesAddress
          : gpu_.getBufferDeviceAddress(instanceMatricesBufferHandle);
  const BufferHandle previousInstanceMatricesBufferHandle =
      animationPreviousFrameValid
          ? animationSceneData->previousInstanceMatricesBuffer
          : (needsVelocityInstanceBufferUpload
                 ? bufferRings_[PreviousInstanceMatricesRing][frameSlot]
                       .buffer->handle()
                 : BufferHandle{});
  const BufferHandle velocityInstanceFlagsBufferHandle =
      needsVelocityInstanceBufferUpload
          ? bufferRings_[VelocityInstanceFlagsRing][frameSlot].buffer->handle()
          : BufferHandle{};
  const BufferHandle velocityFrameDataBufferHandle =
      hasTaaVelocityInstances
          ? bufferRings_[VelocityFrameDataRing][frameSlot].buffer->handle()
          : BufferHandle{};
  const BufferHandle velocityGeometryBufferHandle =
      needsVelocityGeometryUpload
          ? bufferRings_[VelocityGeometryRing][frameSlot].buffer->handle()
          : BufferHandle{};
  const bool reuseCurrentMatricesForVelocity =
      velocityInstanceFlagsMode == VelocityInstanceFlagsMode::AllInvalid ||
      canReuseStaticPreviousMatrices;
  const uint64_t previousInstanceMatricesAddress =
      animationPreviousFrameValid
          ? animationSceneData->previousInstanceMatricesAddress
          : (reuseCurrentMatricesForVelocity
                 ? instanceMatricesAddress
                 : (nuri::isValid(previousInstanceMatricesBufferHandle)
                        ? gpu_.getBufferDeviceAddress(
                              previousInstanceMatricesBufferHandle)
                        : 0u));
  const uint64_t velocityInstanceFlagsAddress =
      nuri::isValid(velocityInstanceFlagsBufferHandle)
          ? gpu_.getBufferDeviceAddress(velocityInstanceFlagsBufferHandle)
          : 0u;
  const uint64_t velocityFrameDataAddress =
      nuri::isValid(velocityFrameDataBufferHandle)
          ? gpu_.getBufferDeviceAddress(velocityFrameDataBufferHandle)
          : 0u;
  const uint64_t velocityGeometryAddress =
      nuri::isValid(velocityGeometryBufferHandle)
          ? gpu_.getBufferDeviceAddress(velocityGeometryBufferHandle)
          : 0u;
  if (hasTaaVelocityInstances) {
    uint32_t validPreviousCount = 0u;
    uint32_t missingPreviousCount = 0u;
    uint32_t animatedResponsiveCount = 0u;
    uint32_t animatedPreviousGeometryCount = 0u;
    double totalObjectMotion = 0.0;
    float maxObjectMotion = 0.0f;
    if (velocityInstanceFlagsMode == VelocityInstanceFlagsMode::AllValid) {
      validPreviousCount = saturateToU32(instanceCount);
    } else if (velocityInstanceFlagsMode ==
               VelocityInstanceFlagsMode::AllInvalid) {
      missingPreviousCount = saturateToU32(instanceCount);
      animatedResponsiveCount =
          animationSceneData != nullptr
              ? saturateToU32(
                    animationSceneData->animatedRenderableIndices.size())
              : 0u;
    } else {
      previousInstanceMatricesCpuCache_.clear();
      velocityInstanceFlagsCpuCache_.clear();
      previousInstanceMatricesCpuCache_.reserve(instanceCount);
      velocityInstanceFlagsCpuCache_.reserve(instanceCount);
      for (size_t i = 0; i < instanceCount; ++i) {
        const RenderableTemplate &templ = renderableTemplates_[i];
        const Renderable *renderable = templ.renderable;
        bool animatedInstance = settings.opaque.enableInstanceAnimation;
        if (!animatedInstance && animationSceneData != nullptr) {
          animatedInstance =
              animationSceneAnimatesRenderable(*animationSceneData, i);
        }
        glm::mat4 currentModel = renderable->modelMatrix;
        glm::mat4 previousModel = currentModel;
        bool hasPrevious = false;
        if (settings.opaque.enableInstanceAnimation &&
            animationSceneData == nullptr) {
          currentModel = makeBuiltInAnimatedModel(
              instanceCentersPhase_[i], instanceBaseMatrices_[i],
              static_cast<float>(frame.timeSeconds));
          if (builtInAnimationPreviousFrameValid) {
            previousModel = makeBuiltInAnimatedModel(
                instanceCentersPhase_[i], instanceBaseMatrices_[i],
                static_cast<float>(frame.timeSeconds - frame.deltaSeconds));
            hasPrevious = true;
          }
        } else if (!animatedInstance && previousCacheValid) {
          if (const auto it = previousTransformById_.find(renderable->id);
              it != previousTransformById_.end()) {
            previousModel = it->second;
            hasPrevious = true;
          }
        }
        if (hasPrevious) {
          ++validPreviousCount;
          const float motion =
              std::max(glm::length(glm::vec3(currentModel[3]) -
                                   glm::vec3(previousModel[3])),
                       maxMatrixElementDelta(currentModel, previousModel));
          totalObjectMotion += static_cast<double>(motion);
          maxObjectMotion = std::max(maxObjectMotion, motion);
        } else {
          ++missingPreviousCount;
          if (animatedInstance) {
            ++animatedResponsiveCount;
          }
        }
        previousInstanceMatricesCpuCache_.push_back(
            makeInstanceData(previousModel));
        velocityInstanceFlagsCpuCache_.push_back(hasPrevious ? 1u : 0u);
      }
      const std::span<const std::byte> previousMatricesBytes{
          reinterpret_cast<const std::byte *>(
              previousInstanceMatricesCpuCache_.data()),
          previousInstanceMatricesCpuCache_.size() * sizeof(InstanceData)};
      const std::span<const std::byte> velocityFlagsBytes{
          reinterpret_cast<const std::byte *>(
              velocityInstanceFlagsCpuCache_.data()),
          velocityInstanceFlagsCpuCache_.size() * sizeof(uint32_t)};
      const std::array updates{
          BufferUpdate{.buffer = previousInstanceMatricesBufferHandle,
                       .data = previousMatricesBytes},
          BufferUpdate{.buffer = velocityInstanceFlagsBufferHandle,
                       .data = velocityFlagsBytes},
      };
      auto updateResult = gpu_.updateBuffers(updates);
      if (updateResult.hasError()) {
        return updateResult;
      }
    }
    if (needsVelocityGeometryUpload) {
      velocityGeometryCpuCache_.clear();
      velocityGeometryCpuCache_.resize(instanceCount);
      const size_t previousOverrideCount = std::min(
          animationSceneData->previousGeometryOverridesByRenderable.size(),
          instanceCount);
      for (size_t i = 0; i < previousOverrideCount; ++i) {
        const AnimatedRenderableGeometryOverride &currentOverride =
            animationSceneData->geometryOverridesByRenderable[i];
        const AnimatedRenderableGeometryOverride &previousOverride =
            animationSceneData->previousGeometryOverridesByRenderable[i];
        if (!currentOverride.enabled ||
            !nuri::isValid(currentOverride.vertexBuffer) ||
            !previousOverride.enabled ||
            !nuri::isValid(previousOverride.vertexBuffer) ||
            previousOverride.vertexCount != currentOverride.vertexCount) {
          continue;
        }
        const uint64_t previousVertexAddress = gpu_.getBufferDeviceAddress(
            previousOverride.vertexBuffer, previousOverride.vertexByteOffset);
        if (previousVertexAddress == 0u) {
          continue;
        }
        velocityGeometryCpuCache_[i] = VelocityRenderableGeometryGpuData{
            .previousVertexBufferAddress = previousVertexAddress,
            .metadata = glm::uvec4(
                kVelocityGeometryFlagPreviousVertexBuffer,
                static_cast<uint32_t>(PackedVertexFormat::AnimatedFloat32),
                previousOverride.vertexCount, 0u),
        };
        ++animatedPreviousGeometryCount;
      }
      const std::span<const std::byte> velocityGeometryBytes{
          reinterpret_cast<const std::byte *>(velocityGeometryCpuCache_.data()),
          velocityGeometryCpuCache_.size() *
              sizeof(VelocityRenderableGeometryGpuData)};
      auto geometryUpdateResult = gpu_.updateBuffer(
          velocityGeometryBufferHandle, velocityGeometryBytes, 0);
      if (geometryUpdateResult.hasError()) {
        return geometryUpdateResult;
      }
    } else {
      velocityGeometryCpuCache_.clear();
    }
    const VelocityFrameGpuData velocityFrameData{
        .currentViewProjNoJitter = frame.camera.currentUnjitteredViewProj,
        .previousViewProjNoJitter = frame.camera.previousUnjitteredViewProj,
        .previousInstanceMatricesAddress = previousInstanceMatricesAddress,
        .velocityInstanceFlagsAddress = velocityInstanceFlagsAddress,
        .instanceFlagsMode = glm::uvec4(
            static_cast<uint32_t>(velocityInstanceFlagsMode), 0u, 0u, 0u),
        .previousGeometryAddress = velocityGeometryAddress,
        .previousGeometryInfo = glm::uvec4(
            needsVelocityGeometryUpload ? instanceCount : 0u, 0u, 0u, 0u),
    };
    const std::span<const std::byte> velocityFrameBytes{
        reinterpret_cast<const std::byte *>(&velocityFrameData),
        sizeof(velocityFrameData)};
    auto frameUpdateResult =
        gpu_.updateBuffer(velocityFrameDataBufferHandle, velocityFrameBytes, 0);
    if (frameUpdateResult.hasError()) {
      return frameUpdateResult;
    }
    AntiAliasingFrameMetrics &aaMetrics = frame.metrics.antiAliasing;
    aaMetrics.velocityInstanceCount = saturateToU32(instanceCount);
    aaMetrics.velocityPreviousTransformValidCount = validPreviousCount;
    aaMetrics.velocityMissingPreviousTransformCount = missingPreviousCount;
    aaMetrics.velocityAnimatedResponsiveCount = animatedResponsiveCount;
    aaMetrics.velocityAnimatedPreviousGeometryCount =
        animatedPreviousGeometryCount;
    aaMetrics.previousTransformCacheValid = previousCacheValid;
    aaMetrics.velocityAverageObjectMotion =
        validPreviousCount > 0u
            ? static_cast<float>(totalObjectMotion / validPreviousCount)
            : 0.0f;
    aaMetrics.velocityMaxObjectMotion = maxObjectMotion;
    aaMetrics.velocityMissingPreviousRatio =
        instanceCount > 0 ? static_cast<float>(missingPreviousCount) /
                                static_cast<float>(instanceCount)
                          : 0.0f;
    aaMetrics.velocityCameraMatrixDelta =
        maxMatrixElementDelta(frame.camera.currentUnjitteredViewProj,
                              frame.camera.previousUnjitteredViewProj);
    aaMetrics.velocityStaticResidualEstimate =
        maxObjectMotion == 0.0f && missingPreviousCount == 0u
            ? aaMetrics.velocityCameraMatrixDelta
            : 0.0f;
    aaMetrics.velocityEstimatedAverageMagnitude =
        aaMetrics.velocityAverageObjectMotion +
        aaMetrics.velocityCameraMatrixDelta;
    aaMetrics.velocityEstimatedMaxMagnitude =
        maxObjectMotion + aaMetrics.velocityCameraMatrixDelta;
  }
  const OpaqueDebugVisualization debugVisualization =
      settings.opaque.debugVisualization;
  const MeshletRenderMode meshletMode = settings.opaque.meshletMode;
  const bool meshletRequested = meshletMode != MeshletRenderMode::Disabled;
  const bool meshletRequired = meshletMode == MeshletRenderMode::Required;
  constexpr bool meshletUsesGpuLod = false;
  frame.metrics.opaque.meshletModeRequired = meshletRequired ? 1u : 0u;
  const bool wireOverlayRequested =
      debugVisualization == OpaqueDebugVisualization::WireframeOverlay;
  const bool wireframeOnlyRequested =
      debugVisualization == OpaqueDebugVisualization::WireframeOnly;
  const bool patchHeatmapRequested =
      debugVisualization == OpaqueDebugVisualization::TessPatchEdgesHeatmap;
  const bool meshletDebugRequested =
      debugVisualization == OpaqueDebugVisualization::MeshletId ||
      debugVisualization == OpaqueDebugVisualization::MeshletSelectedLod;
  const bool overlayRequested = wireOverlayRequested || patchHeatmapRequested;
  const uint32_t debugVisualizationMode =
      static_cast<uint32_t>(debugVisualization);
  DrawItem baseDraw = baseMeshFillDraw_;
  const float tessNearDistance =
      std::max(0.0f, settings.opaque.tessNearDistance);
  const float tessFarDistance =
      std::max(settings.opaque.tessFarDistance, tessNearDistance + 1.0e-3f);
  const float tessFarDistanceSq = tessFarDistance * tessFarDistance;
  const float tessMinFactor =
      std::clamp(settings.opaque.tessMinFactor, 1.0f, 64.0f);
  const float tessMaxFactor =
      std::clamp(settings.opaque.tessMaxFactor, tessMinFactor, 64.0f);
  const size_t tessInstanceCap =
      settings.opaque.tessMaxInstances == kUnlimitedTessInstanceCap
          ? std::numeric_limits<size_t>::max()
          : static_cast<size_t>(settings.opaque.tessMaxInstances);
  const bool tessellationRequested =
      (settings.opaque.enableTessellation || patchHeatmapRequested) &&
      settings.opaque.forcedMeshLod < 1 && !tessellationUnsupported_ &&
      nuri::isValid(meshScenePipelines_[rasterVariantIndex(
          CoverageMode::Sample1, false, true, false)]);
  constexpr uint32_t kInvalidBatchIndex = std::numeric_limits<uint32_t>::max();
  ScratchArena batchScratchArena;
  ScopedScratch batchScratch(batchScratchArena);
  std::pmr::vector<BatchEntry> batches(batchScratch.resource());
  const size_t batchReserve =
      std::min<size_t>(meshDrawTemplates_.size(), kMaxBatchReserve);
  batches.reserve(batchReserve);
  const auto makeBatchEntry =
      [&baseDraw](
          RenderPipelineHandle pipeline, BufferHandle indexBuffer,
          uint64_t indexBufferOffset, const SubmeshLod &lodRange,
          uint32_t vertexOffset, uint32_t submeshIndex, uint32_t resolvedLod,
          uint32_t meshletMaxCount, BufferHandle vertexBuffer,
          BufferHandle vertexDecodeBuffer, uint64_t vertexBufferAddress,
          uint64_t vertexDecodeBufferAddress, uint32_t vertexDecodeIndex,
          uint32_t packedVertexFormat, uint32_t materialIndex,
          const Model::ModelMeshletGpuView *meshletView, bool doubleSided,
          bool alphaMasked, bool materialNormalRequired, size_t count,
          size_t firstInstance) -> BatchEntry {
    BatchEntry entry{};
    entry.draw = baseDraw;
    entry.draw.pipeline = pipeline;
    entry.draw.indexBuffer = indexBuffer;
    entry.draw.indexBufferOffset = indexBufferOffset;
    entry.draw.indexCount = lodRange.indexCount;
    entry.draw.firstIndex = lodRange.indexOffset;
    entry.draw.vertexOffset = 0;
    entry.draw.alphaMasked = alphaMasked;
    entry.vertexBuffer = vertexBuffer;
    entry.vertexDecodeBuffer = vertexDecodeBuffer;
    entry.vertexBufferAddress = vertexBufferAddress;
    entry.vertexDecodeBufferAddress = vertexDecodeBufferAddress;
    entry.vertexDecodeIndex = vertexDecodeIndex;
    entry.packedVertexFormat = packedVertexFormat;
    entry.materialIndex = materialIndex;
    entry.meshletView = meshletView;
    entry.meshletOffset = lodRange.meshletOffset;
    entry.meshletCount = lodRange.meshletCount;
    entry.submeshIndex = submeshIndex;
    entry.resolvedLod = resolvedLod;
    entry.meshletMaxCount = meshletMaxCount;
    entry.vertexOffset = vertexOffset;
    entry.doubleSided = doubleSided;
    entry.alphaMasked = alphaMasked;
    entry.materialNormalRequired = materialNormalRequired;
    entry.instanceCount = count;
    entry.firstInstance = firstInstance;
    return entry;
  };
  const auto appendBatch =
      [&batches, &makeBatchEntry](
          RenderPipelineHandle pipeline, BufferHandle indexBuffer,
          uint64_t indexBufferOffset, const SubmeshLod &lodRange,
          uint32_t vertexOffset, uint32_t submeshIndex, uint32_t resolvedLod,
          uint32_t meshletMaxCount, BufferHandle vertexBuffer,
          BufferHandle vertexDecodeBuffer, uint64_t vertexBufferAddress,
          uint64_t vertexDecodeBufferAddress, uint32_t vertexDecodeIndex,
          uint32_t packedVertexFormat, uint32_t materialIndex,
          const Model::ModelMeshletGpuView *meshletView, bool doubleSided,
          bool alphaMasked, bool materialNormalRequired, size_t count,
          size_t firstInstance) {
        if (count == 0) {
          return;
        }
        batches.push_back(makeBatchEntry(
            pipeline, indexBuffer, indexBufferOffset, lodRange, vertexOffset,
            submeshIndex, resolvedLod, meshletMaxCount, vertexBuffer,
            vertexDecodeBuffer, vertexBufferAddress, vertexDecodeBufferAddress,
            vertexDecodeIndex, packedVertexFormat, materialIndex, meshletView,
            doubleSided, alphaMasked, materialNormalRequired, count,
            firstInstance));
      };
  const VisibilityResolvedSettings visibilitySettings =
      visibilitySettingsFromRenderSettings(settings);
  const bool gpuMainCullingEnabled =
      usesGpuMainVisibility(visibilitySettings.mainViewMode);
  const bool cpuMainCullingEnabled =
      usesCpuMainVisibility(visibilitySettings.mainViewMode);
  const bool cpuMainEvaluationEnabled =
      runsCpuVisibilityEvaluation(visibilitySettings.mainViewMode);
  const bool validateGpuMainVisibility =
      validatesGpuMainVisibility(visibilitySettings.mainViewMode);
  std::pmr::vector<uint8_t> visibleMainTemplates(batchScratch.resource());
  std::pmr::vector<uint32_t> visibleMainTemplateIndices(
      batchScratch.resource());
  std::pmr::vector<uint32_t> gpuVisibilityCandidateIndices(
      batchScratch.resource());
  std::span<const VisibilityCandidate> visibilityCandidates{};
  std::span<const VisibilityCandidateGpu> visibilityCandidateGpuData{};
  VisibilityPassRequest visibilityRequest{};
  bool hasDeformedRenderable = false;
  if (cpuMainEvaluationEnabled || gpuMainCullingEnabled) {
    const uint64_t visibilityTopologyVersion = frame.scene->topologyVersion();
    const uint64_t visibilityTransformVersion = frame.scene->transformVersion();
    const uint64_t visibilityDeformationVersion =
        frame.scene->deformationVersion();
    const uint64_t visibilityGeometryVersion = geometryMutationVersion;
    const bool visibilityCandidateCacheValid =
        hasGeometryMutationTracking &&
        cachedVisibilityCandidateTopologyVersion_ ==
            visibilityTopologyVersion &&
        cachedVisibilityCandidateTransformVersion_ ==
            visibilityTransformVersion &&
        cachedVisibilityCandidateDeformationVersion_ ==
            visibilityDeformationVersion &&
        cachedVisibilityCandidateGeometryVersion_ == visibilityGeometryVersion;
    if (!visibilityCandidateCacheValid) {
      NURI_PROFILER_ZONE("OpaqueRenderer.visibility_candidate_build",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      visibilityCandidates_.clear();
      visibilityCandidateGpuData_.clear();
      visibilityCandidates_.reserve(meshDrawTemplates_.size());
      visibilityCandidateGpuData_.reserve(meshDrawTemplates_.size());
      bool candidatesHadDeformedRenderable = false;
      for (size_t templateIndex = 0; templateIndex < meshDrawTemplates_.size();
           ++templateIndex) {
        const MeshDrawTemplate &templ = meshDrawTemplates_[templateIndex];
        const bool deformed = !templ.renderable->morphWeights.empty() ||
                              !templ.renderable->skinPalette.empty();
        candidatesHadDeformedRenderable |= deformed;
        VisibilityCandidate candidate{
            .renderableIndex = templ.instanceIndex,
            .templateIndex = static_cast<uint32_t>(templateIndex),
            .submeshIndex = templ.submeshIndex,
            .materialIndex = templ.materialIndex,
            .geometryVersion = visibilityGeometryVersion,
            .transformVersion = visibilityTransformVersion,
            .deformationVersion = visibilityDeformationVersion,
            .flags = deformed ? static_cast<uint32_t>(
                                    kVisibilityCandidateConservativeVisible)
                              : 0u,
            .localBounds = templ.submesh->bounds,
            .worldFromLocal = templ.renderable->modelMatrix,
            .meshletView = templ.meshletView,
        };
        visibilityCandidateGpuData_.push_back(
            makeVisibilityCandidateGpu(candidate));
        visibilityCandidates_.push_back(candidate);
      }
      cachedVisibilityCandidateTopologyVersion_ =
          hasGeometryMutationTracking ? visibilityTopologyVersion
                                      : std::numeric_limits<uint64_t>::max();
      cachedVisibilityCandidateTransformVersion_ =
          hasGeometryMutationTracking ? visibilityTransformVersion
                                      : std::numeric_limits<uint64_t>::max();
      cachedVisibilityCandidateDeformationVersion_ =
          hasGeometryMutationTracking ? visibilityDeformationVersion
                                      : std::numeric_limits<uint64_t>::max();
      cachedVisibilityCandidateGeometryVersion_ =
          hasGeometryMutationTracking ? visibilityGeometryVersion
                                      : std::numeric_limits<uint64_t>::max();
      cachedVisibilityCandidatesHadDeformedRenderable_ =
          candidatesHadDeformedRenderable;
      NURI_PROFILER_ZONE_END();
    }
    visibilityCandidates = std::span<const VisibilityCandidate>(
        visibilityCandidates_.data(), visibilityCandidates_.size());
    visibilityCandidateGpuData = std::span<const VisibilityCandidateGpu>(
        visibilityCandidateGpuData_.data(), visibilityCandidateGpuData_.size());
    hasDeformedRenderable = cachedVisibilityCandidatesHadDeformedRenderable_;
    visibilityRequest =
        makeMainViewVisibilityPassRequest(frame.camera, visibilitySettings);
  }
  if (!hasDeformedRenderable &&
      !(cpuMainEvaluationEnabled || gpuMainCullingEnabled)) {
    for (const RenderableTemplate &templ : renderableTemplates_) {
      if (!templ.renderable->morphWeights.empty() ||
          !templ.renderable->skinPalette.empty()) {
        hasDeformedRenderable = true;
        break;
      }
    }
  }
  if (cpuMainEvaluationEnabled) {
    NURI_PROFILER_ZONE("OpaqueRenderer.visibility_cpu_main",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    VisibilityPassResult visibilityResult =
        evaluateCpuVisibilityFromCachedBounds(
            visibilityRequest, visibilityCandidates, visibilityCandidateGpuData,
            batchScratch.resource());
    if (cpuMainCullingEnabled) {
      visibleMainTemplates.assign(meshDrawTemplates_.size(), 0u);
      visibleMainTemplateIndices.clear();
      visibleMainTemplateIndices.reserve(
          visibilityResult.visibleCandidateIndices.size());
      for (const uint32_t candidateIndex :
           visibilityResult.visibleCandidateIndices) {
        const uint32_t templateIndex =
            visibilityCandidates[candidateIndex].templateIndex;
        if (visibleMainTemplates[templateIndex] == 0u) {
          visibleMainTemplates[templateIndex] = 1u;
          visibleMainTemplateIndices.push_back(templateIndex);
        }
      }
    }
    frame.metrics.visibility.cpuMainCandidates = visibilityResult.cpuCandidates;
    frame.metrics.visibility.cpuMainVisibleCandidates =
        visibilityResult.cpuVisibleCandidates;
    frame.metrics.visibility.cpuMainRejected = visibilityResult.cpuRejected;
    frame.metrics.visibility.uncertainVisible =
        visibilityResult.uncertainVisible;
    NURI_PROFILER_ZONE_END();
  }
  if (gpuMainCullingEnabled && gpuVisibilityCandidateIndices.empty()) {
    gpuVisibilityCandidateIndices.reserve(visibilityCandidates.size());
    for (size_t i = 0; i < visibilityCandidates.size(); ++i) {
      gpuVisibilityCandidateIndices.push_back(static_cast<uint32_t>(i));
    }
  }
  constexpr std::array<float, 3> sortedLodThresholds{0.0f, 0.0f, 0.0f};
  const glm::vec3 cameraPosition = glm::vec3(frame.camera.cameraPos);
  const bool useAutoLod = settings.opaque.enableMeshLod &&
                          settings.opaque.forcedMeshLod < 0 &&
                          !meshletUsesGpuLod;
  const bool canUseUniformAutoLodFastPath =
      !cpuMainCullingEnabled && uniformSingleSubmeshPath_ &&
      !meshDrawTemplates_.empty() && useAutoLod &&
      instanceCount == meshDrawTemplates_.size();
  const uint32_t forcedLod =
      settings.opaque.forcedMeshLod < 0
          ? 0u
          : static_cast<uint32_t>(settings.opaque.forcedMeshLod);
  if (useAutoLod) {
    NURI_PROFILER_ZONE("OpaqueRenderer.auto_lod_resolve",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    const float targetPixelError =
        std::max(settings.opaque.meshLodTargetPixelError, 1.0e-3f);
    const float hysteresisRatio =
        std::clamp(settings.opaque.meshLodHysteresisRatio, 0.0f, 0.95f);
    const float projectionScaleY =
        std::abs(frame.camera.currentUnjitteredProj[1][1]);
    const bool cameraCut = frame.camera.historyResetReason ==
                           TemporalHistoryResetReason::CameraCut;
    bool historyReset =
        !autoLodHistoryValid_ || !autoLodWasActive_ ||
        instanceAutoLodLevels_.size() != instanceCount ||
        cachedAutoLodTargetPixelError_ != targetPixelError ||
        cachedAutoLodHysteresisRatio_ != hysteresisRatio ||
        cachedAutoLodProjectionScaleY_ != projectionScaleY ||
        cachedAutoLodNearPlane_ != frame.camera.nearPlane ||
        cachedAutoLodRenderExtent_ != frame.camera.renderExtent ||
        cachedAutoLodProjectionType_ != frame.camera.projectionType ||
        cameraCut;
    if (instanceAutoLodLevels_.size() != instanceCount) {
      instanceAutoLodLevels_.assign(instanceCount, 0u);
    }
    const float pixelScaleY =
        0.5f * static_cast<float>(std::max(frame.camera.renderExtent.y, 1u)) *
        projectionScaleY;
    const bool orthographic =
        frame.camera.projectionType == ProjectionType::Orthographic;
    uint64_t transitionCount = 0u;
    uint64_t lod0Count = 0u;
    uint64_t lod1Count = 0u;
    for (size_t i = 0; i < instanceCount; ++i) {
      const glm::vec4 lodBounds = instanceLodCentersInvRadiusSq_[i];
      const glm::vec4 viewCenter =
          frame.camera.view * glm::vec4(glm::vec3(lodBounds), 1.0f);
      const float worldRadius =
          lodBounds.w > 0.0f ? 1.0f / std::sqrt(lodBounds.w) : kMinLodRadius;
      const float nearestDepth =
          orthographic
              ? 1.0f
              : std::max(-viewCenter.z - worldRadius, frame.camera.nearPlane);
      const detail::OpaqueLodProjection projection{
          .pixelScaleY = pixelScaleY,
          .nearestDepth = nearestDepth,
          .orthographic = orthographic,
      };
      const glm::vec4 worldErrors = instanceAutoLodWorldErrors_[i];
      const std::array<float, Submesh::kMaxLodCount> errors{
          worldErrors.x, worldErrors.y, worldErrors.z, worldErrors.w};
      const uint32_t lodCount = std::clamp<uint32_t>(instanceAutoLodCounts_[i],
                                                     1u, Submesh::kMaxLodCount);
      const uint32_t previousLod = instanceAutoLodLevels_[i];
      const uint32_t selectedLod = detail::selectOpaqueLod(
          std::span<const float>(errors.data(), lodCount), targetPixelError,
          hysteresisRatio, projection,
          historyReset ? std::nullopt : std::optional<uint32_t>(previousLod));
      instanceAutoLodLevels_[i] = selectedLod;
      if (!historyReset && selectedLod != previousLod) {
        ++transitionCount;
      }
      if (selectedLod == 0u) {
        ++lod0Count;
      } else if (selectedLod == 1u) {
        ++lod1Count;
      }
    }
    autoLodHistoryValid_ = true;
    autoLodWasActive_ = true;
    cachedAutoLodTargetPixelError_ = targetPixelError;
    cachedAutoLodHysteresisRatio_ = hysteresisRatio;
    cachedAutoLodProjectionScaleY_ = projectionScaleY;
    cachedAutoLodNearPlane_ = frame.camera.nearPlane;
    cachedAutoLodRenderExtent_ = frame.camera.renderExtent;
    cachedAutoLodProjectionType_ = frame.camera.projectionType;
    frame.metrics.opaque.autoLodActive = 1u;
    frame.metrics.opaque.autoLodHistoryReset = historyReset ? 1u : 0u;
    frame.metrics.opaque.autoLodTransitions = saturateToU32(transitionCount);
    frame.metrics.opaque.autoLodLod0Instances = saturateToU32(lod0Count);
    frame.metrics.opaque.autoLodLod1Instances = saturateToU32(lod1Count);
    NURI_PROFILER_ZONE_END();
  } else {
    autoLodWasActive_ = false;
  }
  bool usedUniformFastPath = false;
  bool usedUniformAutoLodFastPath = false;
  bool usedUniformAutoLodTessSplit = false;
  std::array<size_t, Submesh::kMaxLodCount> autoLodBucketStarts{};
  std::array<size_t, Submesh::kMaxLodCount> autoLodBucketWrites{};
  std::array<size_t, Submesh::kMaxLodCount> autoLodBucketCounts{};
  size_t autoLodTessBucketStart = 0;
  size_t autoLodTessBucketWrite = 0;
  size_t autoLodTessBucketCount = 0;
  size_t remapCount = 0;
  uint64_t remapSignature = kInvalidDrawSignature;
  bool remapSignatureValid = false;
  uint64_t indirectDrawSignature = kInvalidDrawSignature;
  bool indirectDrawSignatureValid = false;
  uint64_t directDrawBufferSignature = kInvalidDrawSignature;
  bool directDrawBufferSignatureValid = false;
  currentDirectDrawBufferSignature_ = kInvalidDrawSignature;
  const std::pmr::vector<uint32_t> *instanceRemapUploadSource = &instanceRemap_;
  const SingleInstanceBatchCache *activeSingleInstanceCache = nullptr;
  const bool canUseStaticBatchCache =
      !cpuMainCullingEnabled && !settings.opaque.enableInstanceAnimation &&
      !useAutoLod && !tessellationRequested && !meshDrawTemplates_.empty() &&
      !uniformSingleSubmeshPath_ && instanceCount > 1;
  const bool canReuseStaticBatchCache =
      canUseStaticBatchCache && staticBatchCache_.valid &&
      staticBatchCache_.meshletRequested == meshletRequested &&
      staticBatchCache_.enableMeshLod == settings.opaque.enableMeshLod &&
      staticBatchCache_.forcedMeshLod == settings.opaque.forcedMeshLod;
  bool reusedStaticBatchCache = false;
  if (canReuseStaticBatchCache) {
    NURI_PROFILER_ZONE("OpaqueRenderer.batch_reuse_static",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    const size_t batchCount = staticBatchCache_.draws.size();
    remapCount = staticBatchCache_.remap.size();
    const bool needsStaticBatchRebind =
        boundStaticBatchGeneration_ != staticBatchCache_.generation ||
        drawItems_.size() != batchCount;
    if (needsStaticBatchRebind) {
      drawItems_ = staticBatchCache_.draws;
      drawPushConstants_ = staticBatchCache_.pushConstantsTemplates;
      drawAlphaMasked_ = staticBatchCache_.alphaMasked;
      meshletBatchInfos_ = staticBatchCache_.meshletBatchInfos;
      for (size_t batchIndex = 0; batchIndex < batchCount; ++batchIndex) {
        drawItems_[batchIndex].alphaMasked = drawAlphaMasked_[batchIndex] != 0u;
        drawItems_[batchIndex].pushConstants =
            std::span<const std::byte>(reinterpret_cast<const std::byte *>(
                                           &drawPushConstants_[batchIndex]),
                                       sizeof(PushConstants));
      }
      boundStaticBatchGeneration_ = staticBatchCache_.generation;
    } else {
      drawPushConstants_.resize(batchCount);
    }
    instanceRemap_.clear();
    instanceRemapUploadSource = &staticBatchCache_.remap;
    remapSignature = staticBatchCache_.remapSignature;
    remapSignatureValid = remapCount > 0;
    if (settings.opaque.enableIndirectDraw &&
        staticBatchCache_.indirectDrawSignature != kInvalidDrawSignature) {
      indirectDrawSignature = staticBatchCache_.indirectDrawSignature;
      indirectDrawSignatureValid = true;
    }
    if (staticBatchCache_.drawBufferSignature != kInvalidDrawSignature) {
      directDrawBufferSignature = staticBatchCache_.drawBufferSignature;
      directDrawBufferSignatureValid = true;
    }
    for (size_t batchIndex = 0; batchIndex < batchCount; ++batchIndex) {
      PushConstants &constants = drawPushConstants_[batchIndex];
      constants.frameDataAddress = frameDataAddress;
      constants.instanceMatricesAddress = instanceMatricesAddress;
      constants.instanceCentersPhaseAddress = instanceCentersPhaseAddress;
      constants.instanceBaseMatricesAddress = instanceBaseMatricesAddress;
      constants.tessNearDistance = tessNearDistance;
      constants.tessFarDistance = tessFarDistance;
      constants.tessMinFactor = tessMinFactor;
      constants.tessMaxFactor = tessMaxFactor;
      constants.debugVisualizationMode = debugVisualizationMode;
    }
    cachedRemapSignature_ = remapSignature;
    cachedRemapSignatureValid_ = remapSignatureValid;
    currentDirectDrawBufferSignature_ = directDrawBufferSignatureValid
                                            ? directDrawBufferSignature
                                            : kInvalidDrawSignature;
    reusedStaticBatchCache = true;
    NURI_PROFILER_ZONE_END();
  }
  if (!reusedStaticBatchCache && canUseUniformAutoLodFastPath) {
    NURI_PROFILER_ZONE("OpaqueRenderer.batch_build_auto_lod",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    MeshDrawTemplate &templateEntry = meshDrawTemplates_.front();
    const Submesh &submesh = *templateEntry.submesh;
    const uint32_t submeshMeshletMaxCount = maxMeshletCountForSubmesh(submesh);
    if (tessellationRequested) {
      instanceTessSelection_.assign(instanceCount, 0u);
      tessCandidates_.clear();
      tessCandidates_.reserve(instanceCount);
    }
    for (size_t i = 0; i < instanceCount; ++i) {
      uint32_t requestedLod = detail::resolveOpaqueAutomaticLod(
          instanceAutoLodLevels_[i], templateEntry.alphaMasked, true);
      const auto resolvedLod = resolveAvailableLod(submesh, requestedLod);
      if (!resolvedLod) {
        continue;
      }
      instanceAutoLodLevels_[i] = *resolvedLod;
      ++autoLodBucketCounts[*resolvedLod];
      ++remapCount;
      if (tessellationRequested && *resolvedLod == 0u) {
        const glm::vec3 delta =
            cameraPosition - glm::vec3(instanceLodCentersInvRadiusSq_[i]);
        const float worldDistanceSq = glm::dot(delta, delta);
        if (worldDistanceSq <= tessFarDistanceSq) {
          tessCandidates_.push_back(TessCandidate{
              .distanceSq = worldDistanceSq,
              .instanceId = static_cast<uint32_t>(i),
          });
        }
      }
    }
    if (tessellationRequested && !tessCandidates_.empty()) {
      const auto candidateCloser = [](const TessCandidate &a,
                                      const TessCandidate &b) {
        if (a.distanceSq != b.distanceSq) {
          return a.distanceSq < b.distanceSq;
        }
        return a.instanceId < b.instanceId;
      };
      const size_t cappedTessCount =
          std::min(tessInstanceCap, tessCandidates_.size());
      if (cappedTessCount < tessCandidates_.size()) {
        std::nth_element(tessCandidates_.begin(),
                         tessCandidates_.begin() + cappedTessCount,
                         tessCandidates_.end(), candidateCloser);
      }
      autoLodTessBucketCount = cappedTessCount;
      for (size_t i = 0; i < cappedTessCount; ++i) {
        const uint32_t instanceId = tessCandidates_[i].instanceId;
        instanceTessSelection_[instanceId] = 1u;
      }
      if (autoLodBucketCounts[0] >= autoLodTessBucketCount) {
        autoLodBucketCounts[0] -= autoLodTessBucketCount;
      } else {
        autoLodBucketCounts[0] = 0;
      }
    }
    size_t firstInstance = 0;
    if (tessellationRequested) {
      const SubmeshLod &lod0Range = submesh.lods[0];
      autoLodBucketStarts[0] = firstInstance;
      autoLodBucketWrites[0] = firstInstance;
      appendBatch(
          selectMeshPipeline(templateEntry.doubleSided, false),
          templateEntry.indexBuffer, templateEntry.indexBufferOffset, lod0Range,
          submesh.vertexOffset, templateEntry.submeshIndex, 0u,
          submeshMeshletMaxCount, templateEntry.vertexBuffer,
          templateEntry.vertexDecodeBuffer, templateEntry.vertexBufferAddress,
          templateEntry.vertexDecodeBufferAddress,
          templateEntry.vertexDecodeIndex, templateEntry.packedVertexFormat,
          templateEntry.materialIndex, templateEntry.meshletView,
          templateEntry.doubleSided, templateEntry.alphaMasked,
          templateEntry.materialNormalRequired, autoLodBucketCounts[0],
          firstInstance);
      firstInstance += autoLodBucketCounts[0];
      autoLodTessBucketStart = firstInstance;
      autoLodTessBucketWrite = firstInstance;
      appendBatch(
          selectMeshPipeline(templateEntry.doubleSided, true),
          templateEntry.indexBuffer, templateEntry.indexBufferOffset, lod0Range,
          submesh.vertexOffset, templateEntry.submeshIndex, 0u,
          submeshMeshletMaxCount, templateEntry.vertexBuffer,
          templateEntry.vertexDecodeBuffer, templateEntry.vertexBufferAddress,
          templateEntry.vertexDecodeBufferAddress,
          templateEntry.vertexDecodeIndex, templateEntry.packedVertexFormat,
          templateEntry.materialIndex, templateEntry.meshletView,
          templateEntry.doubleSided, templateEntry.alphaMasked,
          templateEntry.materialNormalRequired, autoLodTessBucketCount,
          firstInstance);
      if (autoLodTessBucketCount > 0) {
        usedUniformAutoLodTessSplit = true;
      }
      firstInstance += autoLodTessBucketCount;
      for (uint32_t lod = 1; lod < Submesh::kMaxLodCount; ++lod) {
        autoLodBucketStarts[lod] = firstInstance;
        autoLodBucketWrites[lod] = firstInstance;
        const size_t count = autoLodBucketCounts[lod];
        if (count == 0) {
          continue;
        }
        const SubmeshLod &lodRange = submesh.lods[lod];
        appendBatch(
            selectMeshPipeline(templateEntry.doubleSided, false),
            templateEntry.indexBuffer, templateEntry.indexBufferOffset,
            lodRange, submesh.vertexOffset, templateEntry.submeshIndex, lod,
            submeshMeshletMaxCount, templateEntry.vertexBuffer,
            templateEntry.vertexDecodeBuffer, templateEntry.vertexBufferAddress,
            templateEntry.vertexDecodeBufferAddress,
            templateEntry.vertexDecodeIndex, templateEntry.packedVertexFormat,
            templateEntry.materialIndex, templateEntry.meshletView,
            templateEntry.doubleSided, templateEntry.alphaMasked,
            templateEntry.materialNormalRequired, count, firstInstance);
        firstInstance += count;
      }
    } else {
      for (uint32_t lod = 0; lod < Submesh::kMaxLodCount; ++lod) {
        autoLodBucketStarts[lod] = firstInstance;
        autoLodBucketWrites[lod] = firstInstance;
        const size_t count = autoLodBucketCounts[lod];
        if (count == 0) {
          continue;
        }
        const SubmeshLod &lodRange = submesh.lods[lod];
        appendBatch(
            selectMeshPipeline(templateEntry.doubleSided, false),
            templateEntry.indexBuffer, templateEntry.indexBufferOffset,
            lodRange, submesh.vertexOffset, templateEntry.submeshIndex, lod,
            submeshMeshletMaxCount, templateEntry.vertexBuffer,
            templateEntry.vertexDecodeBuffer, templateEntry.vertexBufferAddress,
            templateEntry.vertexDecodeBufferAddress,
            templateEntry.vertexDecodeIndex, templateEntry.packedVertexFormat,
            templateEntry.materialIndex, templateEntry.meshletView,
            templateEntry.doubleSided, templateEntry.alphaMasked,
            templateEntry.materialNormalRequired, count, firstInstance);
        firstInstance += count;
      }
    }
    if (remapCount > 0) {
      usedUniformFastPath = true;
      usedUniformAutoLodFastPath = true;
    }
    NURI_PROFILER_ZONE_END();
  }
  const bool isSingleRenderableInstance = instanceCount == 1;
  if (!useAutoLod && !cpuMainCullingEnabled && !meshletRequested &&
      !reusedStaticBatchCache && !usedUniformFastPath &&
      isSingleRenderableInstance && !meshDrawTemplates_.empty() &&
      !uniformSingleSubmeshPath_) {
    NURI_PROFILER_ZONE("OpaqueRenderer.batch_build_single_instance_cache",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    const uint32_t requestedLod =
        resolveSingleInstanceRequestedLod(settings, forcedLod);
    const bool tessPipelineEnabled = shouldEnableSingleInstanceTessPipeline(
        tessellationRequested, requestedLod, cameraPosition, tessFarDistanceSq);
    auto singleInstanceCacheResult = ensureSingleInstanceBatchCache(
        requestedLod, useAutoLod, tessPipelineEnabled, baseDraw);
    if (singleInstanceCacheResult.hasError()) {
      return singleInstanceCacheResult;
    }
    const size_t cacheIndex =
        singleInstanceCacheIndex(requestedLod, tessPipelineEnabled);
    const SingleInstanceBatchCache &activeCache =
        singleInstanceBatchCaches_[cacheIndex];
    remapCount = activeCache.remapCount;
    if (remapCount > 0) {
      activeSingleInstanceCache = &activeCache;
      usedUniformFastPath = true;
    }
    NURI_PROFILER_ZONE_END();
  }
  if (!cpuMainCullingEnabled && !reusedStaticBatchCache &&
      !usedUniformFastPath && uniformSingleSubmeshPath_ &&
      !tessellationRequested && !meshDrawTemplates_.empty() && !useAutoLod &&
      instanceCount == meshDrawTemplates_.size()) {
    NURI_PROFILER_ZONE("OpaqueRenderer.batch_build_fast",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    MeshDrawTemplate &templateEntry = meshDrawTemplates_.front();
    const uint32_t requestedLod =
        settings.opaque.enableMeshLod ? forcedLod : 0u;
    const auto lodIndex =
        resolveAvailableLod(*templateEntry.submesh, requestedLod);
    if (lodIndex) {
      const SubmeshLod &lodRange = templateEntry.submesh->lods[*lodIndex];
      appendBatch(
          selectMeshPipeline(templateEntry.doubleSided, false),
          templateEntry.indexBuffer, templateEntry.indexBufferOffset, lodRange,
          templateEntry.submesh->vertexOffset, templateEntry.submeshIndex,
          *lodIndex, maxMeshletCountForSubmesh(*templateEntry.submesh),
          templateEntry.vertexBuffer, templateEntry.vertexDecodeBuffer,
          templateEntry.vertexBufferAddress,
          templateEntry.vertexDecodeBufferAddress,
          templateEntry.vertexDecodeIndex, templateEntry.packedVertexFormat,
          templateEntry.materialIndex, templateEntry.meshletView,
          templateEntry.doubleSided, templateEntry.alphaMasked,
          templateEntry.materialNormalRequired, instanceCount, 0);
      remapCount = instanceCount;
      templateBatchIndices_.clear();
      templateBatchIndices_.resize(meshDrawTemplates_.size(), 0u);
      usedUniformFastPath = true;
    }
    NURI_PROFILER_ZONE_END();
  }
  if (!reusedStaticBatchCache && !usedUniformFastPath) {
    templateBatchIndices_.clear();
    templateBatchIndices_.resize(meshDrawTemplates_.size(), kInvalidBatchIndex);
    NURI_PROFILER_ZONE("OpaqueRenderer.batch_build",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    const bool canUseVisibleBatchCache =
        cpuMainCullingEnabled && hasGeometryMutationTracking &&
        !settings.opaque.enableInstanceAnimation && !useAutoLod &&
        !tessellationRequested &&
        debugVisualization == OpaqueDebugVisualization::None &&
        !meshDrawTemplates_.empty();
    bool usedVisibleBatchCache = false;
    if (canUseVisibleBatchCache) {
      const bool cacheValid =
          cachedVisibleBatchValid_ &&
          cachedVisibleBatchTopologyVersion_ ==
              frame.scene->topologyVersion() &&
          cachedVisibleBatchMaterialVersion_ == materialSnapshot.version &&
          cachedVisibleBatchGeometryVersion_ == geometryMutationVersion &&
          cachedVisibleBatchMeshletRequested_ == meshletRequested &&
          cachedVisibleBatchEnableMeshLod_ == settings.opaque.enableMeshLod &&
          cachedVisibleBatchForcedMeshLod_ == settings.opaque.forcedMeshLod &&
          cachedVisibleTemplateBatchIndices_.size() ==
              meshDrawTemplates_.size();
      if (!cacheValid) {
        cachedVisibleBatchValid_ = false;
        cachedVisibleBatchEntries_.clear();
        cachedVisibleTemplateBatchIndices_.clear();
        cachedVisibleTemplateBatchIndices_.resize(meshDrawTemplates_.size(),
                                                  kInvalidBatchIndex);
        PmrHashMap<BatchKey, size_t, BatchKeyHash> batchLookup(
            batchScratch.resource());
        batchLookup.reserve(batchReserve);
        cachedVisibleBatchEntries_.reserve(batchReserve);
        for (size_t templateIndex = 0;
             templateIndex < meshDrawTemplates_.size(); ++templateIndex) {
          MeshDrawTemplate &templateEntry = meshDrawTemplates_[templateIndex];
          uint32_t requestedLod = 0;
          if (!settings.opaque.enableMeshLod) {
            requestedLod = 0;
          } else if (settings.opaque.forcedMeshLod >= 0) {
            requestedLod = forcedLod;
          } else if (meshletUsesGpuLod) {
            requestedLod = 0;
          } else {
            requestedLod = instanceAutoLodLevels_[templateEntry.instanceIndex];
          }
          requestedLod = detail::resolveOpaqueAutomaticLod(
              requestedLod, templateEntry.alphaMasked, useAutoLod);
          const auto lodIndex =
              resolveAvailableLod(*templateEntry.submesh, requestedLod);
          if (!lodIndex) {
            continue;
          }
          const SubmeshLod &lodRange = templateEntry.submesh->lods[*lodIndex];
          const RenderPipelineHandle selectedPipeline =
              selectMeshPipeline(templateEntry.doubleSided, false);
          const BatchKey key{
              .pipeline = selectedPipeline,
              .indexBuffer = templateEntry.indexBuffer,
              .indexBufferOffset = templateEntry.indexBufferOffset,
              .indexCount = lodRange.indexCount,
              .firstIndex = lodRange.indexOffset,
              .vertexBufferAddress = templateEntry.vertexBufferAddress,
              .vertexDecodeBufferAddress =
                  templateEntry.vertexDecodeBufferAddress,
              .vertexDecodeIndex = templateEntry.vertexDecodeIndex,
              .packedVertexFormat = templateEntry.packedVertexFormat,
              .materialIndex = templateEntry.materialIndex,
              .meshletBuffer =
                  meshletRequested && templateEntry.meshletView != nullptr
                      ? templateEntry.meshletView->meshletBuffer
                      : BufferHandle{},
              .meshletOffset = meshletRequested ? lodRange.meshletOffset : 0u,
              .meshletCount = meshletRequested ? lodRange.meshletCount : 0u,
              .meshletSubmeshIndex =
                  meshletRequested ? templateEntry.submeshIndex : 0u,
          };
          auto it = batchLookup.find(key);
          if (it == batchLookup.end()) {
            const size_t insertedIndex = cachedVisibleBatchEntries_.size();
            cachedVisibleBatchEntries_.push_back(makeBatchEntry(
                selectedPipeline, templateEntry.indexBuffer,
                templateEntry.indexBufferOffset, lodRange,
                templateEntry.submesh->vertexOffset, templateEntry.submeshIndex,
                *lodIndex, maxMeshletCountForSubmesh(*templateEntry.submesh),
                templateEntry.vertexBuffer, templateEntry.vertexDecodeBuffer,
                templateEntry.vertexBufferAddress,
                templateEntry.vertexDecodeBufferAddress,
                templateEntry.vertexDecodeIndex,
                templateEntry.packedVertexFormat, templateEntry.materialIndex,
                templateEntry.meshletView, templateEntry.doubleSided,
                templateEntry.alphaMasked, templateEntry.materialNormalRequired,
                0u, 0u));
            auto [insertedIt, _] = batchLookup.emplace(key, insertedIndex);
            it = insertedIt;
          }
          cachedVisibleTemplateBatchIndices_[templateIndex] =
              static_cast<uint32_t>(it->second);
        }
        cachedVisibleBatchTopologyVersion_ = frame.scene->topologyVersion();
        cachedVisibleBatchMaterialVersion_ = materialSnapshot.version;
        cachedVisibleBatchGeometryVersion_ = geometryMutationVersion;
        cachedVisibleBatchMeshletRequested_ = meshletRequested;
        cachedVisibleBatchEnableMeshLod_ = settings.opaque.enableMeshLod;
        cachedVisibleBatchForcedMeshLod_ = settings.opaque.forcedMeshLod;
        cachedVisibleBatchValid_ = true;
      }
      visibleBatchActiveRemap_.assign(cachedVisibleBatchEntries_.size(),
                                      kInvalidBatchIndex);
      for (const uint32_t visibleTemplateIndex : visibleMainTemplateIndices) {
        const size_t templateIndex = static_cast<size_t>(visibleTemplateIndex);
        const uint32_t cachedBatchIndex =
            cachedVisibleTemplateBatchIndices_[templateIndex];
        if (cachedBatchIndex == kInvalidBatchIndex) {
          continue;
        }
        uint32_t activeBatchIndex = visibleBatchActiveRemap_[cachedBatchIndex];
        if (activeBatchIndex == kInvalidBatchIndex) {
          activeBatchIndex = static_cast<uint32_t>(batches.size());
          visibleBatchActiveRemap_[cachedBatchIndex] = activeBatchIndex;
          batches.push_back(cachedVisibleBatchEntries_[cachedBatchIndex]);
          batches.back().instanceCount = 0u;
          batches.back().firstInstance = 0u;
        }
        templateBatchIndices_[templateIndex] = activeBatchIndex;
        ++batches[activeBatchIndex].instanceCount;
        ++remapCount;
      }
      usedVisibleBatchCache = true;
    }
    if (!usedVisibleBatchCache) {
      PmrHashMap<BatchKey, size_t, BatchKeyHash> batchLookup(
          batchScratch.resource());
      batchLookup.reserve(batchReserve);
      const size_t batchTemplateCount = cpuMainCullingEnabled
                                            ? visibleMainTemplateIndices.size()
                                            : meshDrawTemplates_.size();
      for (size_t batchTemplateOffset = 0;
           batchTemplateOffset < batchTemplateCount; ++batchTemplateOffset) {
        const size_t templateIndex =
            cpuMainCullingEnabled
                ? static_cast<size_t>(
                      visibleMainTemplateIndices[batchTemplateOffset])
                : batchTemplateOffset;
        MeshDrawTemplate &templateEntry = meshDrawTemplates_[templateIndex];
        uint32_t requestedLod = 0;
        if (!settings.opaque.enableMeshLod) {
          requestedLod = 0;
        } else if (settings.opaque.forcedMeshLod >= 0) {
          requestedLod = forcedLod;
        } else if (meshletUsesGpuLod) {
          requestedLod = 0;
        } else {
          requestedLod = instanceAutoLodLevels_[templateEntry.instanceIndex];
        }
        requestedLod = detail::resolveOpaqueAutomaticLod(
            requestedLod, templateEntry.alphaMasked, useAutoLod);
        const auto lodIndex =
            resolveAvailableLod(*templateEntry.submesh, requestedLod);
        if (!lodIndex) {
          continue;
        }
        const SubmeshLod &lodRange = templateEntry.submesh->lods[*lodIndex];
        RenderPipelineHandle selectedPipeline =
            selectMeshPipeline(templateEntry.doubleSided, false);
        if (tessellationRequested && *lodIndex == 0 &&
            templateEntry.instanceIndex <
                instanceLodCentersInvRadiusSq_.size()) {
          const glm::vec4 centerInvRadiusSq =
              instanceLodCentersInvRadiusSq_[templateEntry.instanceIndex];
          const float dx = cameraPosition.x - centerInvRadiusSq.x;
          const float dy = cameraPosition.y - centerInvRadiusSq.y;
          const float dz = cameraPosition.z - centerInvRadiusSq.z;
          const float distanceSq = dx * dx + dy * dy + dz * dz;
          if (distanceSq <= tessFarDistanceSq) {
            selectedPipeline =
                selectMeshPipeline(templateEntry.doubleSided, true);
          }
        }
        const BatchKey key{
            .pipeline = selectedPipeline,
            .indexBuffer = templateEntry.indexBuffer,
            .indexBufferOffset = templateEntry.indexBufferOffset,
            .indexCount = lodRange.indexCount,
            .firstIndex = lodRange.indexOffset,
            .vertexBufferAddress = templateEntry.vertexBufferAddress,
            .vertexDecodeBufferAddress =
                templateEntry.vertexDecodeBufferAddress,
            .vertexDecodeIndex = templateEntry.vertexDecodeIndex,
            .packedVertexFormat = templateEntry.packedVertexFormat,
            .materialIndex = templateEntry.materialIndex,
            .meshletBuffer =
                meshletRequested && templateEntry.meshletView != nullptr
                    ? templateEntry.meshletView->meshletBuffer
                    : BufferHandle{},
            .meshletOffset = meshletRequested ? lodRange.meshletOffset : 0u,
            .meshletCount = meshletRequested ? lodRange.meshletCount : 0u,
            .meshletSubmeshIndex =
                meshletRequested ? templateEntry.submeshIndex : 0u,
        };
        auto it = batchLookup.find(key);
        if (it == batchLookup.end()) {
          batches.push_back(makeBatchEntry(
              selectedPipeline, templateEntry.indexBuffer,
              templateEntry.indexBufferOffset, lodRange,
              templateEntry.submesh->vertexOffset, templateEntry.submeshIndex,
              *lodIndex, maxMeshletCountForSubmesh(*templateEntry.submesh),
              templateEntry.vertexBuffer, templateEntry.vertexDecodeBuffer,
              templateEntry.vertexBufferAddress,
              templateEntry.vertexDecodeBufferAddress,
              templateEntry.vertexDecodeIndex, templateEntry.packedVertexFormat,
              templateEntry.materialIndex, templateEntry.meshletView,
              templateEntry.doubleSided, templateEntry.alphaMasked,
              templateEntry.materialNormalRequired, 0u, 0u));
          const size_t insertedIndex = batches.size() - 1;
          auto [insertedIt, _] = batchLookup.emplace(key, insertedIndex);
          it = insertedIt;
        }
        const uint32_t batchIndex = static_cast<uint32_t>(it->second);
        templateBatchIndices_[templateIndex] = batchIndex;
        ++batches[it->second].instanceCount;
        ++remapCount;
      }
    }
    NURI_PROFILER_ZONE_END();
  }
  auto remapCapacityResult = ensureInstanceRemapRingCapacity(
      std::max(remapCount * sizeof(uint32_t), sizeof(uint32_t)));
  if (remapCapacityResult.hasError()) {
    return remapCapacityResult;
  }
  if (!reusedStaticBatchCache) {
    NURI_PROFILER_ZONE("OpaqueRenderer.draw_list_emit",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    const bool useCachedSingleInstanceBatches =
        activeSingleInstanceCache != nullptr;
    const size_t batchCount = useCachedSingleInstanceBatches
                                  ? activeSingleInstanceCache->batches.size()
                                  : batches.size();
    drawItems_.resize(batchCount);
    drawPushConstants_.resize(batchCount);
    drawAlphaMasked_.resize(batchCount, 0u);
    meshletBatchInfos_.resize(batchCount);
    const bool singleRenderableInstance = instanceCount == 1;
    const bool needsBatchWriteOffsets =
        !singleRenderableInstance && !usedUniformFastPath;
    if (needsBatchWriteOffsets) {
      batchWriteOffsets_.resize(batchCount);
    }
    if (singleRenderableInstance) {
      size_t singleRemapCount = 0;
      if (useCachedSingleInstanceBatches) {
        for (size_t batchIndex = 0;
             batchIndex < activeSingleInstanceCache->batches.size();
             ++batchIndex) {
          const SingleInstanceBatchEntry &batch =
              activeSingleInstanceCache->batches[batchIndex];
          singleRemapCount = std::max(singleRemapCount, batch.instanceCount);
        }
      } else {
        for (size_t batchIndex = 0; batchIndex < batches.size(); ++batchIndex) {
          BatchEntry &batch = batches[batchIndex];
          batch.firstInstance = 0;
          singleRemapCount = std::max(singleRemapCount, batch.instanceCount);
        }
      }
      remapCount = singleRemapCount;
    } else {
      size_t firstInstance = 0;
      for (size_t batchIndex = 0; batchIndex < batches.size(); ++batchIndex) {
        BatchEntry &batch = batches[batchIndex];
        batch.firstInstance = firstInstance;
        if (needsBatchWriteOffsets) {
          batchWriteOffsets_[batchIndex] = firstInstance;
        }
        firstInstance += batch.instanceCount;
      }
    }
    if (instanceRemap_.size() != remapCount) {
      instanceRemap_.resize(remapCount);
    }
    remapSignature = hashCombine64(kFnvOffsetBasis64, remapCount);
    remapSignatureValid = true;
    if (usedUniformAutoLodFastPath) {
      for (uint32_t lod = 0; lod < Submesh::kMaxLodCount; ++lod) {
        autoLodBucketWrites[lod] = autoLodBucketStarts[lod];
      }
      if (usedUniformAutoLodTessSplit) {
        autoLodTessBucketWrite = autoLodTessBucketStart;
        for (uint32_t instanceId = 0; instanceId < instanceCount;
             ++instanceId) {
          const uint32_t lod = instanceAutoLodLevels_[instanceId];
          if (lod == 0 && instanceTessSelection_[instanceId] != 0u) {
            instanceRemap_[autoLodTessBucketWrite++] = instanceId;
            remapSignature = hashCombine64(remapSignature,
                                           static_cast<uint64_t>(instanceId));
            continue;
          }
          const size_t writeOffset = autoLodBucketWrites[lod]++;
          instanceRemap_[writeOffset] = instanceId;
          remapSignature =
              hashCombine64(remapSignature, static_cast<uint64_t>(instanceId));
        }
      } else {
        for (uint32_t instanceId = 0; instanceId < instanceCount;
             ++instanceId) {
          const uint32_t lod = instanceAutoLodLevels_[instanceId];
          const size_t writeOffset = autoLodBucketWrites[lod]++;
          instanceRemap_[writeOffset] = instanceId;
          remapSignature =
              hashCombine64(remapSignature, static_cast<uint64_t>(instanceId));
        }
      }
    } else if (singleRenderableInstance) {
      for (size_t i = 0; i < instanceRemap_.size(); ++i) {
        instanceRemap_[i] = 0u;
        remapSignature = hashCombine64(remapSignature, 0u);
      }
    } else if (usedUniformFastPath) {
      for (uint32_t instanceId = 0; instanceId < instanceCount; ++instanceId) {
        instanceRemap_[instanceId] = instanceId;
        remapSignature =
            hashCombine64(remapSignature, static_cast<uint64_t>(instanceId));
      }
    } else {
      const size_t remapTemplateCount = cpuMainCullingEnabled
                                            ? visibleMainTemplateIndices.size()
                                            : meshDrawTemplates_.size();
      for (size_t remapTemplateOffset = 0;
           remapTemplateOffset < remapTemplateCount; ++remapTemplateOffset) {
        const size_t templateIndex =
            cpuMainCullingEnabled
                ? static_cast<size_t>(
                      visibleMainTemplateIndices[remapTemplateOffset])
                : remapTemplateOffset;
        const uint32_t batchIndex = templateBatchIndices_[templateIndex];
        if (batchIndex == kInvalidBatchIndex) {
          continue;
        }
        const size_t writeOffset = batchWriteOffsets_[batchIndex]++;
        const uint32_t instanceId =
            meshDrawTemplates_[templateIndex].instanceIndex;
        instanceRemap_[writeOffset] = instanceId;
        remapSignature =
            hashCombine64(remapSignature, static_cast<uint64_t>(instanceId));
      }
    }
    cachedRemapSignature_ = remapSignature;
    cachedRemapSignatureValid_ = true;
    if (settings.opaque.enableIndirectDraw) {
      indirectDrawSignature = hashCombine64(kFnvOffsetBasis64, batchCount);
      indirectDrawSignature = hashCombine64(indirectDrawSignature, remapCount);
      indirectDrawSignatureValid = true;
    }
    if (batchCount > 0) {
      directDrawBufferSignature = hashCombine64(kFnvOffsetBasis64, batchCount);
      directDrawBufferSignature =
          hashCombine64(directDrawBufferSignature, remapCount);
      directDrawBufferSignatureValid = true;
    }
    if (useCachedSingleInstanceBatches) {
      for (size_t batchIndex = 0;
           batchIndex < activeSingleInstanceCache->batches.size();
           ++batchIndex) {
        const SingleInstanceBatchEntry &batch =
            activeSingleInstanceCache->batches[batchIndex];
        PushConstants &constants = drawPushConstants_[batchIndex];
        constants.frameDataAddress = frameDataAddress;
        constants.vertexBufferAddress = batch.vertexBufferAddress;
        constants.vertexDecodeBufferAddress = batch.vertexDecodeBufferAddress;
        constants.instanceMatricesAddress = instanceMatricesAddress;
        constants.instanceRemapAddress = 0;
        constants.instanceCentersPhaseAddress = instanceCentersPhaseAddress;
        constants.instanceBaseMatricesAddress = instanceBaseMatricesAddress;
        constants.instanceCount = static_cast<uint32_t>(instanceCount);
        constants.materialIndex = batch.materialIndex;
        constants.vertexDecodeIndex = batch.vertexDecodeIndex;
        constants.packedVertexFormat = batch.packedVertexFormat;
        constants.timeSeconds = settings.opaque.enableInstanceAnimation
                                    ? static_cast<float>(frame.timeSeconds)
                                    : 0.0f;
        constants.tessNearDistance = tessNearDistance;
        constants.tessFarDistance = tessFarDistance;
        constants.tessMinFactor = tessMinFactor;
        constants.tessMaxFactor = tessMaxFactor;
        constants.debugVisualizationMode = debugVisualizationMode;
        drawAlphaMasked_[batchIndex] = batch.alphaMasked ? 1u : 0u;
        meshletBatchInfos_[batchIndex] = {};
        DrawItem &draw = drawItems_[batchIndex];
        draw = batch.draw;
        draw.alphaMasked = batch.alphaMasked;
        draw.vertexBuffer = batch.vertexBuffer;
        draw.instanceCount = static_cast<uint32_t>(batch.instanceCount);
        draw.firstInstance = 0;
        draw.pushConstants = std::span<const std::byte>(
            reinterpret_cast<const std::byte *>(&constants),
            sizeof(PushConstants));
        if (directDrawBufferSignatureValid) {
          directDrawBufferSignature =
              hashDrawBufferSignature(directDrawBufferSignature, draw);
        }
        if (indirectDrawSignatureValid) {
          indirectDrawSignature = hashCombine64(
              indirectDrawSignature, static_cast<uint64_t>(draw.command));
          indirectDrawSignature = hashCombine64(
              indirectDrawSignature,
              foldHandle(draw.pipeline.index, draw.pipeline.generation));
          indirectDrawSignature = hashCombine64(
              indirectDrawSignature,
              foldHandle(draw.indexBuffer.index, draw.indexBuffer.generation));
          indirectDrawSignature =
              hashCombine64(indirectDrawSignature, draw.indexBufferOffset);
          indirectDrawSignature = hashCombine64(
              indirectDrawSignature, static_cast<uint64_t>(draw.indexFormat));
          indirectDrawSignature =
              hashCombine64(indirectDrawSignature, draw.indexCount);
          indirectDrawSignature =
              hashCombine64(indirectDrawSignature, draw.instanceCount);
          indirectDrawSignature =
              hashCombine64(indirectDrawSignature, draw.firstIndex);
          indirectDrawSignature = hashCombine64(
              indirectDrawSignature, static_cast<uint64_t>(draw.vertexOffset));
          indirectDrawSignature =
              hashCombine64(indirectDrawSignature, draw.firstInstance);
          indirectDrawSignature = hashCombine64(indirectDrawSignature,
                                                constants.vertexBufferAddress);
          indirectDrawSignature =
              hashCombine64(indirectDrawSignature,
                            static_cast<uint64_t>(constants.materialIndex));
        }
      }
    } else {
      for (size_t batchIndex = 0; batchIndex < batches.size(); ++batchIndex) {
        const BatchEntry &batch = batches[batchIndex];
        PushConstants &constants = drawPushConstants_[batchIndex];
        constants.frameDataAddress = frameDataAddress;
        constants.vertexBufferAddress = batch.vertexBufferAddress;
        constants.vertexDecodeBufferAddress = batch.vertexDecodeBufferAddress;
        constants.instanceMatricesAddress = instanceMatricesAddress;
        constants.instanceRemapAddress = 0;
        constants.instanceCentersPhaseAddress = instanceCentersPhaseAddress;
        constants.instanceBaseMatricesAddress = instanceBaseMatricesAddress;
        constants.instanceCount = static_cast<uint32_t>(instanceCount);
        constants.materialIndex = batch.materialIndex;
        constants.vertexDecodeIndex = batch.vertexDecodeIndex;
        constants.packedVertexFormat = batch.packedVertexFormat;
        constants.timeSeconds = settings.opaque.enableInstanceAnimation
                                    ? static_cast<float>(frame.timeSeconds)
                                    : 0.0f;
        constants.tessNearDistance = tessNearDistance;
        constants.tessFarDistance = tessFarDistance;
        constants.tessMinFactor = tessMinFactor;
        constants.tessMaxFactor = tessMaxFactor;
        constants.debugVisualizationMode = debugVisualizationMode;
        drawAlphaMasked_[batchIndex] = batch.alphaMasked ? 1u : 0u;
        meshletBatchInfos_[batchIndex] = MeshletBatchInfo{
            .view = batch.meshletView,
            .vertexDecodeBuffer = batch.vertexDecodeBuffer,
            .meshletOffset = batch.meshletOffset,
            .meshletCount = batch.meshletCount,
            .submeshIndex = batch.submeshIndex,
            .resolvedLod = batch.resolvedLod,
            .meshletMaxCount = batch.meshletMaxCount,
            .vertexOffset = batch.vertexOffset,
            .doubleSided = batch.doubleSided,
            .alphaMasked = batch.alphaMasked,
            .materialNormalRequired = batch.materialNormalRequired,
        };
        DrawItem &draw = drawItems_[batchIndex];
        draw = batch.draw;
        draw.alphaMasked = batch.alphaMasked;
        draw.vertexBuffer = batch.vertexBuffer;
        draw.instanceCount = static_cast<uint32_t>(batch.instanceCount);
        draw.firstInstance = static_cast<uint32_t>(batch.firstInstance);
        draw.pushConstants = std::span<const std::byte>(
            reinterpret_cast<const std::byte *>(&constants),
            sizeof(PushConstants));
        if (directDrawBufferSignatureValid) {
          directDrawBufferSignature =
              hashDrawBufferSignature(directDrawBufferSignature, draw);
        }
        if (indirectDrawSignatureValid) {
          indirectDrawSignature = hashCombine64(
              indirectDrawSignature, static_cast<uint64_t>(draw.command));
          indirectDrawSignature = hashCombine64(
              indirectDrawSignature,
              foldHandle(draw.pipeline.index, draw.pipeline.generation));
          indirectDrawSignature = hashCombine64(
              indirectDrawSignature,
              foldHandle(draw.indexBuffer.index, draw.indexBuffer.generation));
          indirectDrawSignature =
              hashCombine64(indirectDrawSignature, draw.indexBufferOffset);
          indirectDrawSignature = hashCombine64(
              indirectDrawSignature, static_cast<uint64_t>(draw.indexFormat));
          indirectDrawSignature =
              hashCombine64(indirectDrawSignature, draw.indexCount);
          indirectDrawSignature =
              hashCombine64(indirectDrawSignature, draw.instanceCount);
          indirectDrawSignature =
              hashCombine64(indirectDrawSignature, draw.firstIndex);
          indirectDrawSignature = hashCombine64(
              indirectDrawSignature, static_cast<uint64_t>(draw.vertexOffset));
          indirectDrawSignature =
              hashCombine64(indirectDrawSignature, draw.firstInstance);
          indirectDrawSignature = hashCombine64(indirectDrawSignature,
                                                constants.vertexBufferAddress);
          indirectDrawSignature =
              hashCombine64(indirectDrawSignature,
                            static_cast<uint64_t>(constants.materialIndex));
        }
      }
    }
    NURI_PROFILER_ZONE_END();
    currentDirectDrawBufferSignature_ = directDrawBufferSignatureValid
                                            ? directDrawBufferSignature
                                            : kInvalidDrawSignature;
    if (canUseStaticBatchCache) {
      staticBatchCache_.meshletRequested = meshletRequested;
      staticBatchCache_.enableMeshLod = settings.opaque.enableMeshLod;
      staticBatchCache_.forcedMeshLod = settings.opaque.forcedMeshLod;
      staticBatchCache_.remapSignature =
          remapSignatureValid ? remapSignature : kInvalidDrawSignature;
      staticBatchCache_.indirectDrawSignature = indirectDrawSignatureValid
                                                    ? indirectDrawSignature
                                                    : kInvalidDrawSignature;
      staticBatchCache_.drawBufferSignature = currentDirectDrawBufferSignature_;
      staticBatchCache_.draws = drawItems_;
      staticBatchCache_.pushConstantsTemplates = drawPushConstants_;
      staticBatchCache_.alphaMasked = drawAlphaMasked_;
      staticBatchCache_.meshletBatchInfos = meshletBatchInfos_;
      for (size_t i = 0; i < drawPushConstants_.size(); ++i) {
        staticBatchCache_.draws[i].pushConstants = {};
        staticBatchCache_.pushConstantsTemplates[i].frameDataAddress = 0;
        staticBatchCache_.pushConstantsTemplates[i].instanceMatricesAddress = 0;
        staticBatchCache_.pushConstantsTemplates[i]
            .instanceCentersPhaseAddress = 0;
        staticBatchCache_.pushConstantsTemplates[i]
            .instanceBaseMatricesAddress = 0;
      }
      staticBatchCache_.remap = instanceRemap_;
      staticBatchCache_.valid = true;
    }
  }
  if (!settings.opaque.enableInstancedDraw && !drawItems_.empty()) {
    NURI_PROFILER_ZONE("OpaqueRenderer.instancing_expand",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    size_t expandedDrawCount = 0;
    for (const DrawItem &draw : drawItems_) {
      expandedDrawCount += draw.instanceCount;
    }
    if (expandedDrawCount != drawItems_.size()) {
      std::pmr::vector<PushConstants> expandedPushConstants(
          drawPushConstants_.get_allocator().resource());
      std::pmr::vector<DrawItem> expandedDrawItems(
          drawItems_.get_allocator().resource());
      std::pmr::vector<uint8_t> expandedAlphaMasked(
          drawAlphaMasked_.get_allocator().resource());
      std::pmr::vector<MeshletBatchInfo> expandedMeshletBatchInfos(
          meshletBatchInfos_.get_allocator().resource());
      expandedPushConstants.reserve(expandedDrawCount);
      expandedDrawItems.reserve(expandedDrawCount);
      expandedAlphaMasked.reserve(expandedDrawCount);
      expandedMeshletBatchInfos.reserve(expandedDrawCount);
      for (size_t i = 0; i < drawItems_.size(); ++i) {
        const DrawItem &sourceDraw = drawItems_[i];
        const PushConstants &sourceConstants = drawPushConstants_[i];
        for (uint32_t instanceOffset = 0;
             instanceOffset < sourceDraw.instanceCount; ++instanceOffset) {
          expandedPushConstants.push_back(sourceConstants);
          DrawItem expandedDraw = sourceDraw;
          expandedDraw.alphaMasked = drawAlphaMasked_[i] != 0u;
          expandedDraw.instanceCount = 1;
          expandedDraw.firstInstance =
              sourceDraw.firstInstance + instanceOffset;
          expandedDraw.pushConstants =
              std::span<const std::byte>(reinterpret_cast<const std::byte *>(
                                             &expandedPushConstants.back()),
                                         sizeof(PushConstants));
          expandedDrawItems.push_back(expandedDraw);
          expandedAlphaMasked.push_back(drawAlphaMasked_[i]);
          expandedMeshletBatchInfos.push_back(meshletBatchInfos_[i]);
        }
      }
      drawPushConstants_ = std::move(expandedPushConstants);
      drawItems_ = std::move(expandedDrawItems);
      drawAlphaMasked_ = std::move(expandedAlphaMasked);
      meshletBatchInfos_ = std::move(expandedMeshletBatchInfos);
      for (size_t i = 0; i < drawItems_.size(); ++i) {
        drawItems_[i].pushConstants = std::span<const std::byte>(
            reinterpret_cast<const std::byte *>(&drawPushConstants_[i]),
            sizeof(PushConstants));
      }
    }
    NURI_PROFILER_ZONE_END();
  }
  const uint64_t instanceRemapAddress = gpu_.getBufferDeviceAddress(
      bufferRings_[InstanceRemapRing][frameSlot].buffer->handle());
  if (!instanceRemapUploadSource->empty()) {
    if (!remapSignatureValid) {
      remapSignature = computeRemapSignature(
          std::span<const uint32_t>(instanceRemapUploadSource->data(),
                                    instanceRemapUploadSource->size()));
      remapSignatureValid = true;
      cachedRemapSignature_ = remapSignature;
      cachedRemapSignatureValid_ = true;
    }
    const bool remapAlreadyUploadedForSlot =
        frameSlotStates_[frameSlot].remapUploadSignature == remapSignature;
    if (!remapAlreadyUploadedForSlot) {
      NURI_PROFILER_ZONE("OpaqueRenderer.remap_upload",
                         NURI_PROFILER_COLOR_CMD_COPY);
      const std::span<const std::byte> remapBytes{
          reinterpret_cast<const std::byte *>(
              instanceRemapUploadSource->data()),
          instanceRemapUploadSource->size() * sizeof(uint32_t)};
      auto updateResult = gpu_.updateBuffer(
          bufferRings_[InstanceRemapRing][frameSlot].buffer->handle(),
          remapBytes, 0);
      if (updateResult.hasError()) {
        return updateResult;
      }
      frameSlotStates_[frameSlot].remapUploadSignature = remapSignature;
      NURI_PROFILER_ZONE_END();
    }
  } else {
    frameSlotStates_[frameSlot].remapUploadSignature = kInvalidDrawSignature;
    cachedRemapSignature_ = kInvalidDrawSignature;
    cachedRemapSignatureValid_ = false;
  }
  for (PushConstants &constants : drawPushConstants_) {
    constants.instanceRemapAddress = instanceRemapAddress;
    constants.previousInstanceMatricesAddress = previousInstanceMatricesAddress;
    constants.velocityInstanceFlagsAddress = velocityInstanceFlagsAddress;
    constants.velocityFrameDataAddress = velocityFrameDataAddress;
  }
  if (settings.opaque.enableIndirectDraw) {
    auto indirectBuildResult =
        buildIndirectDraws(frameSlot, remapCount, indirectDrawSignature,
                           indirectDrawSignatureValid);
    if (indirectBuildResult.hasError()) {
      return indirectBuildResult;
    }
  } else {
    invalidateIndirectPackCache();
    indirectDrawItems_.clear();
    indirectAlphaMasked_.clear();
    indirectCommandUploadBytes_.clear();
    currentIndirectDrawBufferSignature_ = kInvalidDrawSignature;
  }
  computePushConstants_ = PushConstants{
      .frameDataAddress = frameDataAddress,
      .vertexBufferAddress = 0,
      .vertexDecodeBufferAddress = 0u,
      .instanceMatricesAddress = instanceMatricesAddress,
      .previousInstanceMatricesAddress = previousInstanceMatricesAddress,
      .instanceRemapAddress = instanceRemapAddress,
      .instanceCentersPhaseAddress = instanceCentersPhaseAddress,
      .instanceBaseMatricesAddress = instanceBaseMatricesAddress,
      .velocityInstanceFlagsAddress = velocityInstanceFlagsAddress,
      .velocityFrameDataAddress = velocityFrameDataAddress,
      .instanceCount = static_cast<uint32_t>(instanceCount),
      .materialIndex = 0u,
      .vertexDecodeIndex = 0u,
      .packedVertexFormat = 0u,
      .timeSeconds = settings.opaque.enableInstanceAnimation
                         ? static_cast<float>(frame.timeSeconds)
                         : 0.0f,
      .tessNearDistance = tessNearDistance,
      .tessFarDistance = tessFarDistance,
      .tessMinFactor = tessMinFactor,
      .tessMaxFactor = tessMaxFactor,
      .debugVisualizationMode = debugVisualizationMode,
  };
  const bool useComputePass =
      settings.opaque.enableInstanceCompute && animationSceneData == nullptr;
  if (!useComputePass && animationSceneData == nullptr && instanceCount > 0) {
    const bool animateInstances = settings.opaque.enableInstanceAnimation;
    const bool needsInstanceMatricesUpload =
        animateInstances || frameSlotStates_[frameSlot].matricesUploadVersion !=
                                cachedTransformVersion_;
    if (needsInstanceMatricesUpload) {
      NURI_PROFILER_ZONE("OpaqueRenderer.instance_matrices_cpu",
                         NURI_PROFILER_COLOR_CMD_COPY);
      if (animateInstances) {
        instanceMatricesCpuCache_.resize(instanceCount);
        for (size_t i = 0; i < instanceCount; ++i) {
          instanceMatricesCpuCache_[i] =
              makeInstanceData(makeBuiltInAnimatedModel(
                  instanceCentersPhase_[i], instanceBaseMatrices_[i],
                  static_cast<float>(frame.timeSeconds)));
        }
      }
      const std::span<const std::byte> matricesBytes{
          reinterpret_cast<const std::byte *>(instanceMatricesCpuCache_.data()),
          instanceMatricesCpuCache_.size() * sizeof(InstanceData)};
      auto updateResult = gpu_.updateBuffer(
          bufferRings_[InstanceMatricesRing][frameSlot].buffer->handle(),
          matricesBytes, 0);
      if (updateResult.hasError()) {
        return updateResult;
      }
      if (!animateInstances) {
        frameSlotStates_[frameSlot].matricesUploadVersion =
            cachedTransformVersion_;
      }
      NURI_PROFILER_ZONE_END();
    }
  }
  uint32_t computeDispatchX = 0;
  {
    NURI_PROFILER_ZONE("OpaqueRenderer.compute_dispatch_submission",
                       NURI_PROFILER_COLOR_CMD_DISPATCH);
    preDispatches_.clear();
    mainPreDispatches_.clear();
    dispatchDependencyBuffers_.clear();
    passDependencyBuffers_.clear();
    passDependencyBufferAccessModes_.clear();
    preResolvedDrawBuffers_.clear();
    passDependencyTextures_.clear();
    const bool hasIndirectDraws = !indirectDrawItems_.empty();
    if (nuri::isValid(sceneGpu->buffer)) {
      appendUniqueDependency(passDependencyBuffers_,
                             passDependencyBufferAccessModes_, sceneGpu->buffer,
                             RenderGraphAccessMode::Read);
    }
    for (const BufferHandle handle : sceneGpu->indirectDependencyBuffers) {
      appendUniqueDependency(passDependencyBuffers_,
                             passDependencyBufferAccessModes_, handle,
                             RenderGraphAccessMode::Read);
    }
    for (const TextureHandle handle : sceneGpu->indirectDependencyTextures) {
      appendUniqueDependency(passDependencyTextures_, handle);
    }
    for (const BufferHandle materialHandle :
         {materialGpu->headerBuffer, materialGpu->clearcoatBuffer,
          materialGpu->sheenBuffer, materialGpu->transmissionBuffer,
          materialGpu->specularBuffer}) {
      appendUniqueDependency(passDependencyBuffers_,
                             passDependencyBufferAccessModes_, materialHandle,
                             RenderGraphAccessMode::Read);
    }
    if (animationSceneData != nullptr) {
      for (const AnimatedRenderableGeometryOverride &geometryOverride :
           animationSceneData->geometryOverridesByRenderable) {
        if (!geometryOverride.enabled ||
            !nuri::isValid(geometryOverride.vertexBuffer)) {
          continue;
        }
        appendUniqueDependency(
            passDependencyBuffers_, passDependencyBufferAccessModes_,
            geometryOverride.vertexBuffer, RenderGraphAccessMode::Read);
      }
    }
    appendUniqueDependency(
        passDependencyBuffers_, passDependencyBufferAccessModes_,
        bufferRings_[InstanceRemapRing][frameSlot].buffer->handle(),
        RenderGraphAccessMode::Read);
    if (hasIndirectDraws) {
      appendUniqueDependency(
          passDependencyBuffers_, passDependencyBufferAccessModes_,
          bufferRings_[IndirectCommandRing][frameSlot].buffer->handle(),
          RenderGraphAccessMode::Read);
    }
    if (instanceCount > 0) {
      appendUniqueDependency(
          passDependencyBuffers_, passDependencyBufferAccessModes_,
          instanceMatricesBufferHandle, RenderGraphAccessMode::Read);
      if (useComputePass) {
        if (instanceCentersPhaseBuffer_ &&
            instanceCentersPhaseBuffer_->valid()) {
          appendUniqueDependency(dispatchDependencyBuffers_,
                                 instanceCentersPhaseBuffer_->handle());
        }
        if (instanceBaseMatricesBuffer_ &&
            instanceBaseMatricesBuffer_->valid()) {
          appendUniqueDependency(dispatchDependencyBuffers_,
                                 instanceBaseMatricesBuffer_->handle());
        }
        appendUniqueDependency(
            dispatchDependencyBuffers_,
            bufferRings_[InstanceMatricesRing][frameSlot].buffer->handle());
        const uint32_t dispatchX = static_cast<uint32_t>(
            (instanceCount + (kComputeWorkgroupSize - 1)) /
            kComputeWorkgroupSize);
        computeDispatchX = std::max(dispatchX, 1u);
        ComputeDispatchItem dispatch{};
        dispatch.pipeline = computePipeline_.get();
        dispatch.dispatch = {.x = computeDispatchX, .y = 1u, .z = 1u};
        dispatch.pushConstants = std::span<const std::byte>(
            reinterpret_cast<const std::byte *>(&computePushConstants_),
            sizeof(computePushConstants_));
        dispatch.dependencyBuffers =
            std::span<const BufferHandle>(dispatchDependencyBuffers_.data(),
                                          dispatchDependencyBuffers_.size());
        dispatch.debugLabel = "Opaque Instance Compute";
        dispatch.debugColor = kComputeDispatchColor;
        preDispatches_.push_back(dispatch);
      }
    }
    NURI_PROFILER_ZONE_END();
  }
  size_t tessellatedDraws = 0;
  size_t tessellatedInstances = 0;
  if (tessellationRequested &&
      nuri::isValid(meshScenePipelines_[rasterVariantIndex(
          CoverageMode::Sample1, false, true, false)])) {
    for (const DrawItem &draw : drawItems_) {
      if (isTessPipeline(draw.pipeline)) {
        ++tessellatedDraws;
        tessellatedInstances += draw.instanceCount;
      }
    }
  }
  const bool hasIndirectBaseDraws = !indirectDrawItems_.empty();
  const auto populateCoverageDependencyBuffers =
      [&](std::pmr::vector<BufferHandle> &dependencies,
          std::pmr::vector<RenderGraphAccessMode> &accessModes) {
        dependencies.clear();
        accessModes.clear();
        auto appendRead = [&](BufferHandle handle) {
          appendUniqueDependency(dependencies, accessModes, handle,
                                 RenderGraphAccessMode::Read);
        };
        if (nuri::isValid(sceneGpu->buffer)) {
          appendRead(sceneGpu->buffer);
        }
        appendRead(materialGpu->headerBuffer);
        if (animationSceneData != nullptr) {
          for (const AnimatedRenderableGeometryOverride &geometryOverride :
               animationSceneData->geometryOverridesByRenderable) {
            if (!geometryOverride.enabled ||
                !nuri::isValid(geometryOverride.vertexBuffer)) {
              continue;
            }
            appendRead(geometryOverride.vertexBuffer);
          }
        }
        appendRead(bufferRings_[InstanceRemapRing][frameSlot].buffer->handle());
        if (hasIndirectBaseDraws) {
          appendRead(
              bufferRings_[IndirectCommandRing][frameSlot].buffer->handle());
        }
        if (instanceCount > 0) {
          appendRead(instanceMatricesBufferHandle);
        }
      };
  const std::span<const DrawItem> baseDrawItems =
      hasIndirectBaseDraws
          ? std::span<const DrawItem>(indirectDrawItems_.data(),
                                      indirectDrawItems_.size())
          : std::span<const DrawItem>(drawItems_.data(), drawItems_.size());
  const std::span<const uint8_t> baseAlphaMasked =
      hasIndirectBaseDraws
          ? std::span<const uint8_t>(indirectAlphaMasked_.data(),
                                     indirectAlphaMasked_.size())
          : std::span<const uint8_t>(drawAlphaMasked_.data(),
                                     drawAlphaMasked_.size());
  bool meshletActive = false;
  if (meshletRequested && !drawItems_.empty()) {
    const bool animationIncompatible = animationSceneData != nullptr;
    const bool tessellationIncompatible = tessellatedDraws != 0u;
    const bool debugVisualizationIncompatible =
        debugVisualization != OpaqueDebugVisualization::None &&
        !meshletDebugRequested;
    const bool hasIncompatibleFrameSettings = animationIncompatible ||
                                              tessellationIncompatible ||
                                              debugVisualizationIncompatible;
    bool missingMeshletData = false;
    for (size_t i = 0; i < meshletBatchInfos_.size() && !missingMeshletData;
         ++i) {
      const MeshletBatchInfo &info = meshletBatchInfos_[i];
      missingMeshletData = info.view == nullptr || info.meshletCount == 0u;
    }
    uint32_t *rejection =
        hasIncompatibleFrameSettings
            ? &frame.metrics.opaque.meshletRejectedIncompatibleFrame
        : missingMeshletData
            ? &frame.metrics.opaque.meshletRejectedMissingAssetData
        : !meshletPipelineInitialized_
            ? &frame.metrics.opaque.meshletRejectedMissingFeature
            : nullptr;
    if (rejection) {
      *rejection = 1u;
      if (meshletRequired)
        return Result<bool, std::string>::makeError(
            "OpaqueRenderer: required meshlet path is unavailable");
    } else {
      meshletActive = true;
    }
  }
  size_t debugOverlayDraws = 0;
  size_t debugOverlayFallbackDraws = 0;
  size_t debugPatchHeatmapDraws = 0;
  overlayDrawItems_.clear();
  passDrawItems_.clear();
  msaaPassDrawItems_.clear();
  depthPrepassDrawItems_.clear();
  transmissionVisibilityDepthDrawItems_.clear();
  transmissionVisibilityDepthPushConstants_.clear();
  normalPrepassDrawItems_.clear();
  meshletDepthPrepassDispatchItems_.clear();
  meshletDepthPrepassPushConstants_.clear();
  meshletDepthPrepassDispatchDependencyBuffers_.clear();
  meshletDepthPrepassDependencyBuffers_.clear();
  meshletDepthPrepassDependencyBufferAccessModes_.clear();
  meshletNormalPrepassDispatchItems_.clear();
  meshletNormalPrepassPushConstants_.clear();
  meshletNormalPrepassDispatchDependencyBuffers_.clear();
  meshletNormalPrepassDependencyBuffers_.clear();
  meshletNormalPrepassDependencyBufferAccessModes_.clear();
  meshletVelocityDispatchItems_.clear();
  meshletVelocityPushConstants_.clear();
  meshletVelocityDispatchDependencyBuffers_.clear();
  meshletReactiveMaskDispatchItems_.clear();
  meshletReactiveMaskPushConstants_.clear();
  meshletReactiveMaskDispatchDependencyBuffers_.clear();
  visibilityMeshletDispatchGpuData_.clear();
  visibilityMeshletCandidateMap_.clear();
  visibilityIndirectMeshDispatchPushConstants_.clear();
  visibilityMeshletGpuDispatches_.clear();
  visibilityMeshletGpuDependencyBuffers_.clear();
  visibilityMeshletGpuDependencyBufferAccessModes_.clear();
  const bool materialNormalInput =
      ambientOcclusionPlan.inputMode ==
      AmbientOcclusionInputMode::MaterialNormalAndDepth;
  const bool depthOnlyInput =
      ambientOcclusionPlan.inputMode ==
      AmbientOcclusionInputMode::DepthOnlyReconstructedNormal;
  const bool normalPrepassRequested =
      ambientOcclusionSettings.active &&
      (depthOnlyInput || nuri::isValid(frame.sharedResources.normalTexture)) &&
      !wireframeOnlyRequested && !baseDrawItems.empty();
  const bool msaaGtaoAuxiliaryPrepass =
      msaaSelected && normalPrepassRequested &&
      nuri::isValid(frame.sharedResources.sceneDepthTexture);
  const bool visibilityDepthPrepassRequested =
      visibilitySettings.enableOcclusionCulling;
  const bool requiresDepthPyramid = this->requiresDepthPyramid(settings);
  frame.metrics.opaque.depthPyramidRequested = requiresDepthPyramid ? 1u : 0u;
  frame.metrics.opaque.hiZRequested =
      sanitizeVisibilityOcclusionMode(visibilitySettings.occlusionMode) !=
              VisibilityOcclusionMode::Disabled
          ? 1u
          : 0u;
  frame.metrics.opaque.hiZSourceFramePolicy =
      visibilitySettings.occlusionMode ==
              VisibilityOcclusionMode::CurrentFrameHiZExperimental
          ? HiZSourceFramePolicy::CurrentFrame
      : frame.metrics.opaque.hiZRequested != 0u
          ? HiZSourceFramePolicy::PreviousFrame
          : HiZSourceFramePolicy::Disabled;
  const bool meshletDepthPrepassRequested =
      settings.opaque.enableDepthPrepass || visibilityDepthPrepassRequested ||
      requiresDepthPyramid;
  bool meshletNormalPrepassEnabled =
      meshletActive && normalPrepassRequested && !drawItems_.empty() &&
      (materialNormalInput
           ? std::ranges::all_of(
                 meshletNormalPipelines_,
                 [](auto pipeline) { return isValid(pipeline); })
           : nuri::isValid(selectMeshletDepthPipeline(CoverageMode::Sample1,
                                                      false, false)) &&
                 nuri::isValid(selectMeshletDepthPipeline(CoverageMode::Sample1,
                                                          false, true)) &&
                 nuri::isValid(selectMeshletDepthPipeline(CoverageMode::Sample1,
                                                          true, false)) &&
                 nuri::isValid(selectMeshletDepthPipeline(CoverageMode::Sample1,
                                                          true, true)));
  const bool meshletNormalDepthMergeEligible =
      remapCount >= kMeshletNormalDepthMergeMinVisibleInstances;
  const bool meshletNormalProvidesRequestedDepth =
      meshletNormalPrepassEnabled && meshletDepthPrepassRequested &&
      meshletNormalDepthMergeEligible && !msaaGtaoAuxiliaryPrepass;
  const bool meshletUsesBatchResolvedLod = !meshletUsesGpuLod;
  const bool classicMeshletDepthPrepassEligible =
      meshletActive && !msaaSelected && meshletDepthPrepassRequested &&
      meshletUsesBatchResolvedLod && !normalPrepassRequested &&
      !meshletNormalProvidesRequestedDepth;
  bool depthPrepassEnabled =
      (!meshletActive || classicMeshletDepthPrepassEligible) &&
      (settings.opaque.enableDepthPrepass || visibilityDepthPrepassRequested ||
       (normalPrepassRequested && !msaaGtaoAuxiliaryPrepass) ||
       requiresDepthPyramid) &&
      !wireframeOnlyRequested && !baseDrawItems.empty();
  bool meshletDepthPrepassEnabled =
      meshletActive && meshletDepthPrepassRequested &&
      !meshletNormalProvidesRequestedDepth && !depthPrepassEnabled &&
      !wireframeOnlyRequested && !drawItems_.empty();
  bool useDepthPreparedMainDrawItems = false;
  std::pmr::vector<uint8_t> meshletMainSourceMask(batchScratch.resource());
  bool meshletHybridRoutingActive = false;
  if (depthPrepassEnabled) {
    NURI_PROFILER_ZONE("OpaqueRenderer.depth_prepass_build",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    depthPrepassDrawItems_.reserve(baseDrawItems.size());
    passDrawItems_.reserve(baseDrawItems.size());
    for (size_t i = 0; i < baseDrawItems.size(); ++i) {
      const DrawItem &source = baseDrawItems[i];
      const bool alphaMasked = baseAlphaMasked[i] != 0u;
      if (msaaSelected && alphaMasked) {
        DrawItem shadeDraw = source;
        shadeDraw.alphaMasked = true;
        shadeDraw.useDepthState = true;
        shadeDraw.depthState = {.compareOp = CompareOp::Less,
                                .isDepthWriteEnabled = true};
        passDrawItems_.push_back(shadeDraw);
        continue;
      }
      const RenderPipelineHandle depthPipeline =
          selectDepthPipeline(source.pipeline, alphaMasked, coverage);
      if (!nuri::isValid(depthPipeline)) {
        depthPrepassEnabled = false;
        depthPrepassDrawItems_.clear();
        passDrawItems_.clear();
        useDepthPreparedMainDrawItems = false;
        break;
      }
      DrawItem depthDraw = source;
      depthDraw.pipeline = depthPipeline;
      depthDraw.useDepthState = true;
      depthDraw.depthState = {.compareOp = CompareOp::Less,
                              .isDepthWriteEnabled = true};
      depthDraw.debugLabel =
          alphaMasked ? "OpaqueMeshDepthAlpha" : "OpaqueMeshDepth";
      depthPrepassDrawItems_.push_back(depthDraw);
      if (!meshletActive) {
        DrawItem shadeDraw = source;
        shadeDraw.useDepthState = true;
        shadeDraw.depthState = {.compareOp = CompareOp::Equal,
                                .isDepthWriteEnabled = false};
        passDrawItems_.push_back(shadeDraw);
      }
    }
    if (depthPrepassEnabled) {
      useDepthPreparedMainDrawItems = !meshletActive && !passDrawItems_.empty();
      if (depthPrepassDrawItems_.empty()) {
        depthPrepassEnabled = false;
      }
    }
    NURI_PROFILER_ZONE_END();
  }
  const bool meshletHybridRoutingEligible =
      meshletActive && meshletMode == MeshletRenderMode::Opportunistic &&
      settings.opaque.hybridClassicMaxMeshlets != 0u &&
      meshletUsesBatchResolvedLod &&
      debugVisualization == OpaqueDebugVisualization::None && !msaaSelected &&
      depthPrepassEnabled && !normalPrepassRequested &&
      !temporalMotionRequired && !reactiveMaskRequired;
  if (meshletHybridRoutingEligible) {
    meshletMainSourceMask.resize(drawItems_.size(), 1u);
    passDrawItems_.clear();
    uint64_t classicInstanceCount = 0u;
    uint64_t coverageClassicInstanceCount = 0u;
    uint64_t meshletInstanceCount = 0u;
    for (size_t i = 0; i < drawItems_.size(); ++i) {
      const DrawItem &source = drawItems_[i];
      const MeshletBatchInfo &info = meshletBatchInfos_[i];
      const bool meshletDataValid = info.view != nullptr &&
                                    info.meshletCount != 0u &&
                                    info.meshletMaxCount != 0u;
      const bool useMeshlets = detail::shouldUseMeshletsForOpaqueBatch(
          meshletMode, true, info.alphaMasked, info.doubleSided,
          meshletDataValid, info.meshletCount,
          settings.opaque.hybridClassicMaxMeshlets);
      meshletMainSourceMask[i] = useMeshlets ? 1u : 0u;
      if (useMeshlets) {
        ++frame.metrics.opaque.meshletHybridMeshletBatches;
        meshletInstanceCount += source.instanceCount;
        continue;
      }
      ++frame.metrics.opaque.meshletHybridClassicBatches;
      classicInstanceCount += source.instanceCount;
      if (info.alphaMasked || info.doubleSided) {
        ++frame.metrics.opaque.meshletHybridCoverageClassicBatches;
        coverageClassicInstanceCount += source.instanceCount;
      }
    }
    passDrawItems_.reserve(drawItems_.size());
    for (size_t i = 0u; i < drawItems_.size(); ++i) {
      if (meshletMainSourceMask[i] != 0u) {
        continue;
      }
      DrawItem shadeDraw = drawItems_[i];
      shadeDraw.useDepthState = true;
      shadeDraw.depthState = {.compareOp = CompareOp::Equal,
                              .isDepthWriteEnabled = false};
      shadeDraw.debugLabel = "OpaqueMeshHybridIndexed";
      passDrawItems_.push_back(shadeDraw);
    }
    frame.metrics.opaque.meshletHybridClassicInstances =
        saturateToU32(classicInstanceCount);
    frame.metrics.opaque.meshletHybridCoverageClassicInstances =
        saturateToU32(coverageClassicInstanceCount);
    frame.metrics.opaque.meshletHybridMeshletInstances =
        saturateToU32(meshletInstanceCount);
    meshletHybridRoutingActive =
        frame.metrics.opaque.meshletHybridClassicBatches != 0u;
    frame.metrics.opaque.meshletHybridActive =
        meshletHybridRoutingActive ? 1u : 0u;
    useDepthPreparedMainDrawItems = meshletHybridRoutingActive;
  }
  bool transmissionVisibilityDepthEnabled =
      !meshletActive &&
      shouldBuildTransmissionVisibilityDepth(frame, settings) &&
      !wireframeOnlyRequested && !baseDrawItems.empty();
  if (transmissionVisibilityDepthEnabled) {
    auto visibilityDepthResult =
        ensureTransmissionVisibilityDepthTexture(sceneDepthTexture);
    if (visibilityDepthResult.hasError()) {
      return visibilityDepthResult;
    }
    frame.sharedResources.transmissionVisibilityDepthTexture =
        transmissionVisibilityDepthTexture_;
    NURI_PROFILER_ZONE("OpaqueRenderer.transmission_visibility_depth_build",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    transmissionVisibilityDepthDrawItems_.reserve(baseDrawItems.size());
    transmissionVisibilityDepthPushConstants_.reserve(baseDrawItems.size());
    for (size_t i = 0; i < baseDrawItems.size(); ++i) {
      const DrawItem &source = baseDrawItems[i];
      const bool alphaMasked = baseAlphaMasked[i] != 0u;
      const RenderPipelineHandle depthPipeline = selectDepthPipeline(
          source.pipeline, alphaMasked, CoverageMode::Sample1);
      if (!nuri::isValid(depthPipeline)) {
        transmissionVisibilityDepthEnabled = false;
        transmissionVisibilityDepthDrawItems_.clear();
        transmissionVisibilityDepthPushConstants_.clear();
        frame.sharedResources.transmissionVisibilityDepthTexture = {};
        break;
      }
      PushConstants constants{};
      std::memcpy(&constants, source.pushConstants.data(), sizeof(constants));
      constants.frameDataAddress = sceneGpu->postTaaFrameDataAddress;
      transmissionVisibilityDepthPushConstants_.push_back(constants);
      DrawItem depthDraw = source;
      depthDraw.pipeline = depthPipeline;
      depthDraw.useDepthState = true;
      depthDraw.depthState = {.compareOp = CompareOp::Less,
                              .isDepthWriteEnabled = true};
      depthDraw.pushConstants = std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(
              &transmissionVisibilityDepthPushConstants_.back()),
          sizeof(PushConstants));
      depthDraw.debugLabel = alphaMasked ? "TransmissionVisibilityDepthAlpha"
                                         : "TransmissionVisibilityDepth";
      transmissionVisibilityDepthDrawItems_.push_back(depthDraw);
    }
    NURI_PROFILER_ZONE_END();
  }
  std::span<const DrawItem> shadedBaseDrawItems =
      useDepthPreparedMainDrawItems
          ? std::span<const DrawItem>(passDrawItems_.data(),
                                      passDrawItems_.size())
          : baseDrawItems;
  bool meshletNormalPrepassWritesDepth =
      meshletNormalPrepassEnabled &&
      (msaaGtaoAuxiliaryPrepass || meshletNormalProvidesRequestedDepth ||
       !meshletDepthPrepassEnabled);
  const bool classicNormalPrepassWritesDepth =
      !meshletNormalPrepassEnabled && normalPrepassRequested &&
      (msaaGtaoAuxiliaryPrepass ||
       (meshletActive && !meshletDepthPrepassEnabled));
  const bool meshletNormalPreparesMainDepth =
      meshletNormalPrepassWritesDepth && !msaaGtaoAuxiliaryPrepass;
  const bool classicNormalPreparesMainDepth =
      classicNormalPrepassWritesDepth && !msaaGtaoAuxiliaryPrepass;
  const std::span<const DrawItem> normalPrepassSourceDrawItems =
      classicNormalPrepassWritesDepth ? baseDrawItems : shadedBaseDrawItems;
  bool normalPrepassEnabled =
      !meshletNormalPrepassEnabled && normalPrepassRequested &&
      !normalPrepassSourceDrawItems.empty() &&
      (classicNormalPrepassWritesDepth || depthPrepassEnabled ||
       meshletDepthPrepassEnabled);
  if (normalPrepassEnabled) {
    normalPrepassDrawItems_.reserve(normalPrepassSourceDrawItems.size());
    for (const DrawItem &source : normalPrepassSourceDrawItems) {
      const RenderPipelineHandle normalPipeline =
          materialNormalInput
              ? selectNormalPipeline(source.pipeline)
              : selectDepthPipeline(source.pipeline, source.alphaMasked,
                                    CoverageMode::Sample1);
      if (!nuri::isValid(normalPipeline)) {
        normalPrepassDrawItems_.clear();
        normalPrepassEnabled = false;
        ambientOcclusionSettings.active = false;
        frame.sharedResources.ambientOcclusionTexture = {};
        frame.sharedResources.ambientOcclusionGraphTexture = {};
        aoMetrics.active = false;
        aoMetrics.disabledReason = AmbientOcclusionDisabledReason::Unsupported;
        break;
      }
      DrawItem normalDraw = source;
      normalDraw.pipeline = normalPipeline;
      normalDraw.useDepthState = true;
      normalDraw.depthState =
          classicNormalPrepassWritesDepth
              ? DepthState{.compareOp = CompareOp::Less,
                           .isDepthWriteEnabled = true}
              : DepthState{.compareOp = CompareOp::LessEqual,
                           .isDepthWriteEnabled = false};
      normalDraw.debugLabel =
          materialNormalInput ? "OpaqueMaterialNormals" : "OpaqueGTAODepth";
      normalDraw.debugColor = 0xff66ddff;
      normalPrepassDrawItems_.push_back(normalDraw);
    }
  }
  const MeshletLimits meshletLimits = gpu_.getMeshletLimits();
  uint32_t maxTaskGroupsX = meshletLimits.maxTaskWorkGroupCountX != 0u
                                ? meshletLimits.maxTaskWorkGroupCountX
                                : std::numeric_limits<uint32_t>::max();
  if (meshletLimits.maxTaskWorkGroupTotalCount != 0u) {
    maxTaskGroupsX =
        std::min(maxTaskGroupsX, meshletLimits.maxTaskWorkGroupTotalCount);
  }
  maxTaskGroupsX = std::max(maxTaskGroupsX, 1u);
  const uint64_t maxCandidateSpanPerDispatch =
      static_cast<uint64_t>(maxTaskGroupsX) *
      kOpaqueMeshletTaskCandidatesPerGroup;
  const uint32_t maxCandidateOffsetStep =
      static_cast<uint32_t>(std::min<uint64_t>(
          maxCandidateSpanPerDispatch, std::numeric_limits<uint32_t>::max()));
  const auto maxTaskGroupsYFor = [&](uint32_t groupsX) -> uint32_t {
    uint32_t maxGroupsY = meshletLimits.maxTaskWorkGroupCountY != 0u
                              ? meshletLimits.maxTaskWorkGroupCountY
                              : std::numeric_limits<uint32_t>::max();
    if (meshletLimits.maxTaskWorkGroupTotalCount != 0u) {
      maxGroupsY = std::min(
          maxGroupsY,
          std::max(meshletLimits.maxTaskWorkGroupTotalCount / groupsX, 1u));
    }
    return std::max(maxGroupsY, 1u);
  };
  const auto meshletDispatchBucket = [](uint32_t groupCount) -> uint32_t {
    uint32_t bucket = 0u;
    uint32_t value = groupCount > 0u ? groupCount - 1u : 0u;
    while (value > 0u) {
      value >>= 1u;
      ++bucket;
    }
    return bucket;
  };
  const bool meshletDispatchUsesGpuLod = meshletUsesGpuLod;
  const auto meshletCandidateSpanForInfo =
      [meshletDispatchUsesGpuLod](const MeshletBatchInfo &info) -> uint32_t {
    return meshletDispatchUsesGpuLod ? info.meshletMaxCount : info.meshletCount;
  };
  const auto estimateMeshletBatchDescriptorCount = [&]() {
    size_t descriptorCount = 0u;
    for (size_t i = 0; i < drawItems_.size(); ++i) {
      const DrawItem &draw = drawItems_[i];
      const MeshletBatchInfo &info = meshletBatchInfos_[i];
      const uint32_t candidateSpan = meshletCandidateSpanForInfo(info);
      const uint64_t candidateCount =
          static_cast<uint64_t>(candidateSpan) * draw.instanceCount;
      descriptorCount += static_cast<size_t>(
          (candidateCount + maxCandidateSpanPerDispatch - 1u) /
          maxCandidateSpanPerDispatch);
    }
    return descriptorCount;
  };
  meshletBatchGpuData_.clear();
  BufferHandle meshletBatchBufferHandle{};
  uint64_t meshletBatchBufferAddress = 0u;
  if (meshletActive) {
    const size_t descriptorCount = estimateMeshletBatchDescriptorCount();
    const size_t descriptorPassCount =
        1u + (meshletDepthPrepassEnabled ? 1u : 0u) +
        (meshletNormalPrepassEnabled ? 1u : 0u) +
        (temporalMotionRequired ? 1u : 0u) + (reactiveMaskRequired ? 1u : 0u);
    const size_t descriptorCapacity = descriptorCount * descriptorPassCount;
    auto batchBufferResult = ensureMeshletBatchRingCapacity(
        std::max(descriptorCapacity * sizeof(MeshletBatchGpuData),
                 sizeof(MeshletBatchGpuData)));
    if (batchBufferResult.hasError()) {
      return batchBufferResult;
    }
    meshletBatchBufferHandle =
        bufferRings_[MeshletBatchRing][frameSlot].buffer->handle();
    meshletBatchBufferAddress =
        gpu_.getBufferDeviceAddress(meshletBatchBufferHandle);
  }
  const bool meshletPreviousFrameDepthPyramidAvailable =
      frame.frameIndex > 0u &&
      frame.sharedResources.sceneDepthPyramidSourceFrameIndex.has_value() &&
      frame.sharedResources.sceneDepthPyramidSourceViewProj.has_value() &&
      previousDepthPyramidCameraStable(frame) &&
      *frame.sharedResources.sceneDepthPyramidSourceFrameIndex ==
          frame.frameIndex - 1u;
  const bool meshletSceneStaticForPreviousOcclusion =
      previousTransformSceneId_ == frame.scene->id() &&
      previousTransformCaptureFrameIndex_ !=
          std::numeric_limits<uint64_t>::max() &&
      previousTransformCaptureFrameIndex_ < frame.frameIndex &&
      previousTransformCaptureTopologyVersion_ ==
          frame.scene->topologyVersion() &&
      previousTransformCaptureTransformVersion_ ==
          frame.scene->transformVersion();
  const bool meshletPreviousFrameOcclusionAvailable =
      meshletActive && visibilitySettings.enableOcclusionCulling &&
      visibilitySettings.occlusionMode ==
          VisibilityOcclusionMode::PreviousFrameHiZ &&
      meshletPreviousFrameDepthPyramidAvailable &&
      meshletSceneStaticForPreviousOcclusion && animationSceneData == nullptr &&
      !settings.opaque.enableInstanceAnimation && !hasDeformedRenderable;
  const bool meshletCurrentFrameOcclusionRequested =
      meshletActive && visibilitySettings.enableOcclusionCulling &&
      visibilitySettings.occlusionMode ==
          VisibilityOcclusionMode::CurrentFrameHiZExperimental &&
      animationSceneData == nullptr &&
      !settings.opaque.enableInstanceAnimation && !hasDeformedRenderable;
  const bool currentDepthVerificationExtentFits =
      sceneDepthPyramidWidth_ > 0u && sceneDepthPyramidWidth_ <= 0xffffu &&
      sceneDepthPyramidHeight_ > 0u && sceneDepthPyramidHeight_ <= 0xffffu;
  const bool currentDepthVerificationAvailable =
      meshletCurrentFrameOcclusionRequested &&
      currentDepthVerificationExtentFits &&
      nuri::isValid(currentFrameDepthVerificationPipelineHandle_) &&
      nuri::isValid(currentFrameDepthVerificationTexture_);
  const uint32_t currentDepthVerificationTexId =
      currentDepthVerificationAvailable
          ? gpu_.getTextureBindlessIndex(currentFrameDepthVerificationTexture_)
          : kInvalidTextureBindlessIndex;
  const uint32_t currentDepthVerificationExtentPacked =
      currentDepthVerificationTexId != kInvalidTextureBindlessIndex
          ? sceneDepthPyramidWidth_ | (sceneDepthPyramidHeight_ << 16u)
          : 0u;
  bool meshletCurrentFrameOcclusionAvailable = false;
  BufferHandle meshletVisibilityCounterBuffer{};
  uint64_t meshletVisibilityCounterBufferAddress = 0u;
  uint32_t meshletCounterFlags = 0u;
  if (meshletActive) {
    auto counterResult =
        ensureVisibilityCounterRingCapacity(sizeof(VisibilityCounterGpuData));
    if (counterResult.hasError()) {
      return counterResult;
    }
    meshletVisibilityCounterBuffer =
        bufferRings_[VisibilityCounterRing][frameSlot].buffer->handle();
    if (!visibilityCounterPreparedForFrame) {
      visibilityCounterClear_.clear();
      visibilityCounterClear_.push_back(VisibilityCounterGpuData{});
      auto clearCounterResult = gpu_.updateBuffer(
          meshletVisibilityCounterBuffer,
          std::as_bytes(std::span<const VisibilityCounterGpuData>(
              visibilityCounterClear_.data(), visibilityCounterClear_.size())),
          0u);
      if (clearCounterResult.hasError()) {
        return clearCounterResult;
      }
      FrameSlotState &slot = frameSlotStates_[frameSlot];
      slot.visibilityPublishedFrame = frame.frameIndex;
      slot.expectedVisibleCount = 0u;
      slot.expectedVisibleHash = kFnvOffsetBasis64;
      slot.expectedVisibleListValid = false;
      visibilityCounterPreparedForFrame = true;
    }
    meshletVisibilityCounterBufferAddress =
        gpu_.getBufferDeviceAddress(meshletVisibilityCounterBuffer);
    meshletCounterFlags = kMeshletCounterFlagEnabled;
  }
  struct MeshletSourceFilter {
    bool alphaMaskedOnly = false;
    bool materialNormalUsesAuxPipelines = false;
    std::span<const uint8_t> enabledSources{};
  };
  const auto buildMeshletDispatches =
      [&](std::pmr::vector<MeshDispatchItem> &dispatchItems,
          std::pmr::vector<MeshletPushConstants> &pushConstants,
          std::pmr::vector<MeshletDispatchDependencyBuffers>
              &dispatchDependencyBuffers,
          MeshletPipelineHandle singleSidedPipeline,
          MeshletPipelineHandle doubleSidedPipeline,
          MeshletPipelineHandle alphaPipeline,
          MeshletPipelineHandle alphaDoubleSidedPipeline,
          CompareOp depthCompareOp, bool depthWriteEnabled,
          std::string_view debugLabel, uint32_t debugColor,
          MeshletOcclusionSource occlusionSource,
          uint64_t visibilityCounterBufferAddress,
          uint32_t meshletCounterFlagsForPass,
          uint64_t velocityFrameDataAddressForPass,
          bool useExactMeshletDispatchGroups,
          bool allowStaticMeshletDispatchCache,
          MeshletSourceFilter sourceFilter) -> uint64_t {
    dispatchItems.clear();
    pushConstants.clear();
    dispatchDependencyBuffers.clear();
    uint64_t dispatchSignature = hashCombine64(kFnvOffsetBasis64, 0x4d455348u);
    dispatchSignature = hashCombine64(
        dispatchSignature,
        foldHandle(singleSidedPipeline.index, singleSidedPipeline.generation));
    dispatchSignature = hashCombine64(
        dispatchSignature,
        foldHandle(doubleSidedPipeline.index, doubleSidedPipeline.generation));
    dispatchSignature =
        hashCombine64(dispatchSignature, foldHandle(alphaPipeline.index,
                                                    alphaPipeline.generation));
    dispatchSignature = hashCombine64(
        dispatchSignature, foldHandle(alphaDoubleSidedPipeline.index,
                                      alphaDoubleSidedPipeline.generation));
    dispatchSignature =
        hashCombine64(dispatchSignature, static_cast<uint64_t>(depthCompareOp));
    dispatchSignature =
        hashCombine64(dispatchSignature, depthWriteEnabled ? 1u : 0u);
    dispatchSignature = hashCombine64(dispatchSignature,
                                      settings.opaque.enableMeshLod ? 1u : 0u);
    dispatchSignature =
        hashCombine64(dispatchSignature, static_cast<uint64_t>(forcedLod));
    dispatchSignature =
        hashCombine64(dispatchSignature,
                      visibilitySettings.enableMeshletFrustumCulling ? 1u : 0u);
    dispatchSignature =
        hashCombine64(dispatchSignature,
                      visibilitySettings.enableMeshletConeCulling ? 1u : 0u);
    dispatchSignature = hashCombine64(
        dispatchSignature, static_cast<uint64_t>(debugVisualization));
    dispatchSignature = hashCombine64(dispatchSignature,
                                      static_cast<uint32_t>(occlusionSource));
    dispatchSignature = hashCombine64(dispatchSignature, maxTaskGroupsX);
    dispatchSignature =
        hashCombine64(dispatchSignature, kOpaqueMeshletTaskCandidatesPerGroup);
    dispatchSignature = hashCombine64(dispatchSignature,
                                      useExactMeshletDispatchGroups ? 1u : 0u);
    dispatchSignature = hashCombine64(dispatchSignature,
                                      sourceFilter.alphaMaskedOnly ? 1u : 0u);
    dispatchSignature =
        hashCombine64(dispatchSignature,
                      sourceFilter.materialNormalUsesAuxPipelines ? 1u : 0u);
    const auto patchCachedDispatches =
        [&](uint64_t cachedCandidateCount) -> uint64_t {
      const auto &cachedDispatches = staticBatchCache_.meshletDispatches;
      const auto &cachedPushConstants =
          staticBatchCache_.meshletPushConstantsTemplates;
      const auto &cachedDependencyBuffers =
          staticBatchCache_.meshletDispatchDependencyBuffers;
      dispatchItems = cachedDispatches;
      pushConstants = cachedPushConstants;
      dispatchDependencyBuffers.clear();
      dispatchDependencyBuffers.reserve(cachedDependencyBuffers.size());
      for (const MeshletDispatchDependencyBuffers &cachedDependencies :
           cachedDependencyBuffers) {
        dispatchDependencyBuffers.emplace_back(
            dispatchDependencyBuffers.get_allocator().resource());
        dispatchDependencyBuffers.back().buffers.assign(
            cachedDependencies.buffers.begin(),
            cachedDependencies.buffers.end());
      }
      for (size_t i = 0; i < pushConstants.size(); ++i) {
        MeshletPushConstants &constants = pushConstants[i];
        constants.frameDataAddress = frameDataAddress;
        constants.instanceMatricesAddress = instanceMatricesAddress;
        constants.instanceRemapAddress = instanceRemapAddress;
        constants.instanceLodBoundsAddress = instanceLodBoundsAddress;
        constants.lodThresholds =
            glm::vec4(sortedLodThresholds[0], sortedLodThresholds[1],
                      sortedLodThresholds[2], 0.0f);
        constants.meshletBatchBufferAddress = meshletBatchBufferAddress;
        constants.visibilityCounterBufferAddress =
            visibilityCounterBufferAddress;
        constants.compactedMeshletBufferAddress = 0u;
        constants.compactionCounterBufferAddress = 0u;
        constants.velocityFrameDataAddress = velocityFrameDataAddressForPass;
        constants.sourceFrameIndex = static_cast<uint32_t>(frame.frameIndex);
        constants.meshletCounterFlags = meshletCounterFlagsForPass;
        dispatchItems[i].pushConstants = std::span<const std::byte>(
            reinterpret_cast<const std::byte *>(&pushConstants[i]),
            sizeof(MeshletPushConstants));
        dispatchItems[i].dependencyBuffers =
            dispatchDependencyBuffers[i].span();
      }
      meshletBatchGpuData_ = staticBatchCache_.meshletBatchGpuData;
      return cachedCandidateCount;
    };
    if (allowStaticMeshletDispatchCache &&
        staticBatchCache_.meshletDispatchCacheValid &&
        staticBatchCache_.meshletDispatchSignature == dispatchSignature) {
      return patchCachedDispatches(staticBatchCache_.meshletCandidateCount);
    }
    struct MeshletBuildSource {
      MeshletBatchGpuData batch{};
      std::array<BufferHandle, 1> dependencies{};
      MeshletPipelineHandle pipeline{};
      uint32_t candidateCount = 0u;
    };
    std::pmr::vector<MeshletBuildSource> sources(
        drawItems_.get_allocator().resource());
    sources.reserve(drawItems_.size());
    uint64_t totalCandidateCount = 0;
    uint32_t maxCandidateCount = 0u;
    for (size_t i = 0; i < drawItems_.size(); ++i) {
      if (!sourceFilter.enabledSources.empty() &&
          sourceFilter.enabledSources[i] == 0u) {
        continue;
      }
      const DrawItem &draw = drawItems_[i];
      const MeshletBatchInfo &info = meshletBatchInfos_[i];
      const Model::ModelMeshletGpuView *view = info.view;
      if (sourceFilter.alphaMaskedOnly && !info.alphaMasked) {
        continue;
      }
      const PushConstants *classicConstants =
          reinterpret_cast<const PushConstants *>(draw.pushConstants.data());
      const uint64_t meshletBufferAddress =
          gpu_.getBufferDeviceAddress(view->meshletBuffer);
      const uint64_t meshletVertexIndexAddress =
          gpu_.getBufferDeviceAddress(view->meshletVertexIndexBuffer);
      const uint64_t meshletPrimitiveIndexAddress =
          gpu_.getBufferDeviceAddress(view->meshletPrimitiveIndexBuffer);
      const uint64_t meshletLodRangeAddress =
          gpu_.getBufferDeviceAddress(view->lodRangeBuffer);
      uint32_t flags = 0u;
      const bool allowsCoverageChangingOptimizations = !info.alphaMasked;
      const bool meshletGpuLod =
          meshletUsesGpuLod && allowsCoverageChangingOptimizations;
      flags |= meshletGpuLod ? kMeshletFlagGpuLod : 0u;
      flags |= visibilitySettings.enableMeshletFrustumCulling
                   ? kMeshletFlagFrustumCulling
                   : 0u;
      flags |= visibilitySettings.enableMeshletConeCulling &&
                       allowsCoverageChangingOptimizations
                   ? kMeshletFlagConeCulling
                   : 0u;
      flags |= debugVisualization == OpaqueDebugVisualization::MeshletId
                   ? kMeshletFlagDebugMeshletId
                   : 0u;
      flags |=
          debugVisualization == OpaqueDebugVisualization::MeshletSelectedLod
              ? kMeshletFlagDebugSelectedLod
              : 0u;
      flags |= occlusionSource != MeshletOcclusionSource::Disabled &&
                       allowsCoverageChangingOptimizations
                   ? kMeshletFlagOcclusionCulling
                   : 0u;
      flags |= occlusionSource == MeshletOcclusionSource::CurrentFrame &&
                       allowsCoverageChangingOptimizations
                   ? kMeshletFlagCurrentFrameOcclusion
                   : 0u;
      flags |= info.doubleSided ? kMeshletFlagDoubleSided : 0u;
      const uint32_t forcedMeshLod =
          settings.opaque.enableMeshLod
              ? std::min(forcedLod, Submesh::kMaxLodCount - 1u)
              : 0u;
      flags |= (forcedMeshLod & kMeshletFlagForcedLodMask)
               << kMeshletFlagForcedLodShift;
      const uint32_t candidateSpan = meshletCandidateSpanForInfo(info);
      const uint32_t candidateCount = static_cast<uint32_t>(
          static_cast<uint64_t>(candidateSpan) * draw.instanceCount);
      totalCandidateCount += candidateCount;
      maxCandidateCount = std::max(maxCandidateCount, candidateCount);
      MeshletBatchGpuData batch{};
      batch.vertexBufferAddress = classicConstants->vertexBufferAddress;
      batch.vertexDecodeBufferAddress =
          classicConstants->vertexDecodeBufferAddress;
      batch.meshletBufferAddress = meshletBufferAddress;
      batch.meshletVertexIndexBufferAddress = meshletVertexIndexAddress;
      batch.meshletPrimitiveIndexBufferAddress = meshletPrimitiveIndexAddress;
      batch.meshletLodRangeBufferAddress = meshletLodRangeAddress;
      batch.draw = glm::uvec4(draw.instanceCount, draw.firstInstance,
                              classicConstants->materialIndex,
                              classicConstants->vertexDecodeIndex);
      batch.mesh =
          glm::uvec4(classicConstants->packedVertexFormat, info.vertexOffset,
                     info.submeshIndex, candidateSpan);
      batch.flags = glm::uvec4(flags, info.resolvedLod, 0u, 0u);
      const bool useAuxPipeline = nuri::isValid(alphaPipeline) &&
                                  nuri::isValid(alphaDoubleSidedPipeline) &&
                                  (sourceFilter.materialNormalUsesAuxPipelines
                                       ? info.materialNormalRequired
                                       : info.alphaMasked);
      const MeshletPipelineHandle selectedPipeline =
          useAuxPipeline
              ? (info.doubleSided ? alphaDoubleSidedPipeline : alphaPipeline)
              : (info.doubleSided ? doubleSidedPipeline : singleSidedPipeline);
      sources.push_back(MeshletBuildSource{
          .batch = batch,
          .dependencies = {draw.vertexBuffer},
          .pipeline = selectedPipeline,
          .candidateCount = candidateCount,
      });
    }
    if (sources.empty()) {
      return totalCandidateCount;
    }
    constexpr uint64_t kExactMeshletDispatchCandidateThreshold = 10u * 1024u;
    const bool useExactMeshletDispatches =
        useExactMeshletDispatchGroups &&
        totalCandidateCount >= kExactMeshletDispatchCandidateThreshold;
    const auto samePipeline = [](MeshletPipelineHandle lhs,
                                 MeshletPipelineHandle rhs) noexcept {
      return lhs.index == rhs.index && lhs.generation == rhs.generation;
    };
    const auto appendMeshletDependency =
        [](MeshletDispatchDependencyBuffers &dependencies,
           PmrHashSet<uint64_t> &dependencyKeys, BufferHandle handle) {
          if (!nuri::isValid(handle)) {
            return;
          }
          const uint64_t key = foldHandle(handle.index, handle.generation);
          if (!dependencyKeys.insert(key).second) {
            return;
          }
          dependencies.buffers.push_back(handle);
        };
    const auto meshletTaskGroupCount = [](uint32_t candidateCount) {
      return (candidateCount + kOpaqueMeshletTaskCandidatesPerGroup - 1u) /
             kOpaqueMeshletTaskCandidatesPerGroup;
    };
    const auto sourceCandidateSpanForDispatch =
        [&](const MeshletBuildSource &source, uint32_t candidateOffset) {
          const uint64_t remaining =
              static_cast<uint64_t>(source.candidateCount - candidateOffset);
          return static_cast<uint32_t>(
              std::min<uint64_t>(maxCandidateSpanPerDispatch, remaining));
        };
    const MeshletPipelineHandle pipelines[4] = {
        singleSidedPipeline, doubleSidedPipeline, alphaPipeline,
        alphaDoubleSidedPipeline};
    constexpr size_t kMeshletPipelineBucketCount = 4u;
    constexpr size_t kMeshletDispatchBucketCount = 32u;
    constexpr size_t kMeshletSourceBucketCount =
        kMeshletPipelineBucketCount * kMeshletDispatchBucketCount;
    std::array<size_t, kMeshletSourceBucketCount> bucketCounts{};
    std::array<size_t, kMeshletSourceBucketCount + 1u> bucketOffsets{};
    std::array<size_t, kMeshletSourceBucketCount> bucketCursors{};
    std::pmr::vector<size_t> bucketedSourceIndices(
        drawItems_.get_allocator().resource());
    const auto visitSourceBuckets =
        [&](size_t sourceIndex, uint32_t candidateOffset, auto &&visitor) {
          const MeshletBuildSource &source = sources[sourceIndex];
          if (source.candidateCount <= candidateOffset) {
            return;
          }
          const uint32_t taskGroupCount = meshletTaskGroupCount(
              sourceCandidateSpanForDispatch(source, candidateOffset));
          const size_t bucket = meshletDispatchBucket(taskGroupCount);
          if (bucket >= kMeshletDispatchBucketCount) {
            return;
          }
          for (size_t pipelineIndex = 0u;
               pipelineIndex < kMeshletPipelineBucketCount; ++pipelineIndex) {
            const MeshletPipelineHandle pipeline = pipelines[pipelineIndex];
            if (!nuri::isValid(pipeline) ||
                !samePipeline(source.pipeline, pipeline)) {
              continue;
            }
            visitor(pipelineIndex * kMeshletDispatchBucketCount + bucket);
          }
        };
    for (uint32_t candidateOffset = 0u; candidateOffset < maxCandidateCount;) {
      bucketCounts.fill(0u);
      size_t bucketedSourceCount = 0u;
      for (size_t sourceIndex = 0; sourceIndex < sources.size();
           ++sourceIndex) {
        visitSourceBuckets(sourceIndex, candidateOffset, [&](size_t bucket) {
          ++bucketCounts[bucket];
          ++bucketedSourceCount;
        });
      }
      size_t bucketOffset = 0u;
      for (size_t bucketIndex = 0u; bucketIndex < bucketCounts.size();
           ++bucketIndex) {
        bucketOffsets[bucketIndex] = bucketOffset;
        bucketOffset += bucketCounts[bucketIndex];
      }
      bucketOffsets.back() = bucketOffset;
      std::copy(bucketOffsets.begin(),
                bucketOffsets.begin() +
                    static_cast<std::ptrdiff_t>(bucketCursors.size()),
                bucketCursors.begin());
      bucketedSourceIndices.resize(bucketedSourceCount);
      for (size_t sourceIndex = 0; sourceIndex < sources.size();
           ++sourceIndex) {
        visitSourceBuckets(sourceIndex, candidateOffset, [&](size_t bucket) {
          bucketedSourceIndices[bucketCursors[bucket]++] = sourceIndex;
        });
      }
      for (size_t pipelineIndex = 0u;
           pipelineIndex < kMeshletPipelineBucketCount; ++pipelineIndex) {
        const MeshletPipelineHandle pipeline = pipelines[pipelineIndex];
        if (!nuri::isValid(pipeline)) {
          continue;
        }
        for (size_t bucket = 0u; bucket < kMeshletDispatchBucketCount;
             ++bucket) {
          const size_t bucketIndex =
              pipelineIndex * kMeshletDispatchBucketCount + bucket;
          const size_t sourceBegin = bucketOffsets[bucketIndex];
          const size_t sourceEnd = bucketOffsets[bucketIndex + 1u];
          if (sourceBegin != sourceEnd) {
            if (useExactMeshletDispatches) {
              const auto taskGroupCountForSourceIndex =
                  [&](size_t sourceIndex) -> uint32_t {
                const MeshletBuildSource &source = sources[sourceIndex];
                return meshletTaskGroupCount(
                    sourceCandidateSpanForDispatch(source, candidateOffset));
              };
              std::sort(bucketedSourceIndices.begin() +
                            static_cast<std::ptrdiff_t>(sourceBegin),
                        bucketedSourceIndices.begin() +
                            static_cast<std::ptrdiff_t>(sourceEnd),
                        [&](size_t lhs, size_t rhs) {
                          return taskGroupCountForSourceIndex(lhs) <
                                 taskGroupCountForSourceIndex(rhs);
                        });
              for (size_t exactBegin = sourceBegin; exactBegin < sourceEnd;) {
                const uint32_t groupsX = taskGroupCountForSourceIndex(
                    bucketedSourceIndices[exactBegin]);
                size_t exactEnd = exactBegin + 1u;
                while (exactEnd < sourceEnd &&
                       taskGroupCountForSourceIndex(
                           bucketedSourceIndices[exactEnd]) == groupsX) {
                  ++exactEnd;
                }
                const size_t groupBatchBase = meshletBatchGpuData_.size();
                MeshletDispatchDependencyBuffers exactDependencies(
                    dispatchDependencyBuffers.get_allocator().resource());
                PmrHashSet<uint64_t> exactDependencyKeys(
                    dispatchDependencyBuffers.get_allocator().resource());
                exactDependencyKeys.reserve(
                    std::min(exactEnd - exactBegin + 1u,
                             kMaxMeshDispatchDependencyResources));
                uint32_t groupBatchCount = 0u;
                for (size_t sourceCursor = exactBegin; sourceCursor < exactEnd;
                     ++sourceCursor) {
                  const size_t sourceIndex =
                      bucketedSourceIndices[sourceCursor];
                  const MeshletBuildSource &source = sources[sourceIndex];
                  meshletBatchGpuData_.push_back(source.batch);
                  for (const BufferHandle handle : source.dependencies) {
                    appendMeshletDependency(exactDependencies,
                                            exactDependencyKeys, handle);
                  }
                  ++groupBatchCount;
                }
                appendMeshletDependency(exactDependencies, exactDependencyKeys,
                                        meshletBatchBufferHandle);
                const uint32_t maxGroupsY = maxTaskGroupsYFor(groupsX);
                uint32_t emittedBatches = 0u;
                while (emittedBatches < groupBatchCount) {
                  const uint32_t groupsY =
                      std::min(maxGroupsY, groupBatchCount - emittedBatches);
                  MeshletPushConstants constants{};
                  constants.frameDataAddress = frameDataAddress;
                  constants.instanceMatricesAddress = instanceMatricesAddress;
                  constants.instanceRemapAddress = instanceRemapAddress;
                  constants.instanceLodBoundsAddress = instanceLodBoundsAddress;
                  constants.lodThresholds =
                      glm::vec4(sortedLodThresholds[0], sortedLodThresholds[1],
                                sortedLodThresholds[2], 0.0f);
                  constants.meshletBatchBufferAddress =
                      meshletBatchBufferAddress;
                  constants.visibilityCounterBufferAddress =
                      visibilityCounterBufferAddress;
                  constants.compactedMeshletBufferAddress = 0u;
                  constants.compactionCounterBufferAddress = 0u;
                  constants.velocityFrameDataAddress =
                      velocityFrameDataAddressForPass;
                  constants.batchBase =
                      static_cast<uint32_t>(groupBatchBase + emittedBatches);
                  constants.candidateOffset = candidateOffset;
                  constants.sourceFrameIndex =
                      static_cast<uint32_t>(frame.frameIndex);
                  constants.meshletCounterFlags = meshletCounterFlagsForPass;
                  constants.currentDepthVerificationTexId =
                      occlusionSource == MeshletOcclusionSource::CurrentFrame
                          ? currentDepthVerificationTexId
                          : kInvalidTextureBindlessIndex;
                  constants.currentDepthVerificationExtentPacked =
                      occlusionSource == MeshletOcclusionSource::CurrentFrame
                          ? currentDepthVerificationExtentPacked
                          : 0u;
                  pushConstants.push_back(constants);
                  MeshDispatchItem dispatch{};
                  dispatch.command = MeshDispatchCommandType::Direct;
                  dispatch.pipeline = pipeline;
                  dispatch.groupsX = groupsX;
                  dispatch.groupsY = groupsY;
                  dispatch.groupsZ = 1u;
                  dispatch.useDepthState = true;
                  dispatch.depthState = {.compareOp = depthCompareOp,
                                         .isDepthWriteEnabled =
                                             depthWriteEnabled};
                  dispatch.debugLabel = debugLabel;
                  dispatch.debugColor = debugColor;
                  dispatchItems.push_back(dispatch);
                  dispatchDependencyBuffers.emplace_back(
                      dispatchDependencyBuffers.get_allocator().resource());
                  MeshletDispatchDependencyBuffers &dependencies =
                      dispatchDependencyBuffers.back();
                  dependencies.buffers.assign(exactDependencies.buffers.begin(),
                                              exactDependencies.buffers.end());
                  emittedBatches += groupsY;
                }
                exactBegin = exactEnd;
              }
              continue;
            }
            const size_t groupBatchBase = meshletBatchGpuData_.size();
            MeshletDispatchDependencyBuffers groupDependencies(
                dispatchDependencyBuffers.get_allocator().resource());
            PmrHashSet<uint64_t> groupDependencyKeys(
                dispatchDependencyBuffers.get_allocator().resource());
            groupDependencyKeys.reserve(
                std::min(sourceEnd - sourceBegin + 1u,
                         kMaxMeshDispatchDependencyResources));
            uint32_t groupBatchCount = 0u;
            uint32_t groupsX = 0u;
            for (size_t sourceCursor = sourceBegin; sourceCursor < sourceEnd;
                 ++sourceCursor) {
              const size_t sourceIndex = bucketedSourceIndices[sourceCursor];
              const MeshletBuildSource &source = sources[sourceIndex];
              const uint32_t taskGroupCount = meshletTaskGroupCount(
                  sourceCandidateSpanForDispatch(source, candidateOffset));
              meshletBatchGpuData_.push_back(source.batch);
              for (const BufferHandle handle : source.dependencies) {
                appendMeshletDependency(groupDependencies, groupDependencyKeys,
                                        handle);
              }
              groupsX = std::max(groupsX, taskGroupCount);
              ++groupBatchCount;
            }
            appendMeshletDependency(groupDependencies, groupDependencyKeys,
                                    meshletBatchBufferHandle);
            const uint32_t maxGroupsY = maxTaskGroupsYFor(groupsX);
            uint32_t emittedBatches = 0u;
            while (emittedBatches < groupBatchCount) {
              const uint32_t groupsY =
                  std::min(maxGroupsY, groupBatchCount - emittedBatches);
              MeshletPushConstants constants{};
              constants.frameDataAddress = frameDataAddress;
              constants.instanceMatricesAddress = instanceMatricesAddress;
              constants.instanceRemapAddress = instanceRemapAddress;
              constants.instanceLodBoundsAddress = instanceLodBoundsAddress;
              constants.lodThresholds =
                  glm::vec4(sortedLodThresholds[0], sortedLodThresholds[1],
                            sortedLodThresholds[2], 0.0f);
              constants.meshletBatchBufferAddress = meshletBatchBufferAddress;
              constants.visibilityCounterBufferAddress =
                  visibilityCounterBufferAddress;
              constants.compactedMeshletBufferAddress = 0u;
              constants.compactionCounterBufferAddress = 0u;
              constants.velocityFrameDataAddress =
                  velocityFrameDataAddressForPass;
              constants.batchBase =
                  static_cast<uint32_t>(groupBatchBase + emittedBatches);
              constants.candidateOffset = candidateOffset;
              constants.sourceFrameIndex =
                  static_cast<uint32_t>(frame.frameIndex);
              constants.meshletCounterFlags = meshletCounterFlagsForPass;
              constants.currentDepthVerificationTexId =
                  occlusionSource == MeshletOcclusionSource::CurrentFrame
                      ? currentDepthVerificationTexId
                      : kInvalidTextureBindlessIndex;
              constants.currentDepthVerificationExtentPacked =
                  occlusionSource == MeshletOcclusionSource::CurrentFrame
                      ? currentDepthVerificationExtentPacked
                      : 0u;
              pushConstants.push_back(constants);
              MeshDispatchItem dispatch{};
              dispatch.command = MeshDispatchCommandType::Direct;
              dispatch.pipeline = pipeline;
              dispatch.groupsX = groupsX;
              dispatch.groupsY = groupsY;
              dispatch.groupsZ = 1u;
              dispatch.useDepthState = true;
              dispatch.depthState = {.compareOp = depthCompareOp,
                                     .isDepthWriteEnabled = depthWriteEnabled};
              dispatch.debugLabel = debugLabel;
              dispatch.debugColor = debugColor;
              dispatchItems.push_back(dispatch);
              dispatchDependencyBuffers.emplace_back(
                  dispatchDependencyBuffers.get_allocator().resource());
              MeshletDispatchDependencyBuffers &dependencies =
                  dispatchDependencyBuffers.back();
              dependencies.buffers.assign(groupDependencies.buffers.begin(),
                                          groupDependencies.buffers.end());
              emittedBatches += groupsY;
            }
          }
        }
      }
      if (static_cast<uint64_t>(maxCandidateCount - candidateOffset) <=
          maxCandidateSpanPerDispatch) {
        break;
      }
      candidateOffset += maxCandidateOffsetStep;
    }
    for (size_t i = 0; i < dispatchItems.size(); ++i) {
      dispatchItems[i].pushConstants = std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&pushConstants[i]),
          sizeof(MeshletPushConstants));
      dispatchItems[i].dependencyBuffers = dispatchDependencyBuffers[i].span();
    }
    if (allowStaticMeshletDispatchCache) {
      auto &cachedDispatches = staticBatchCache_.meshletDispatches;
      auto &cachedPushConstants =
          staticBatchCache_.meshletPushConstantsTemplates;
      auto &cachedDependencyBuffers =
          staticBatchCache_.meshletDispatchDependencyBuffers;
      auto &cachedBatchGpuData = staticBatchCache_.meshletBatchGpuData;
      cachedDispatches = dispatchItems;
      cachedPushConstants = pushConstants;
      for (MeshDispatchItem &cachedDispatch : cachedDispatches) {
        cachedDispatch.pushConstants = {};
        cachedDispatch.dependencyBuffers = {};
        cachedDispatch.dependencyTextures = {};
      }
      cachedDependencyBuffers.clear();
      cachedDependencyBuffers.reserve(dispatchDependencyBuffers.size());
      for (const MeshletDispatchDependencyBuffers &dependencies :
           dispatchDependencyBuffers) {
        cachedDependencyBuffers.emplace_back(
            cachedDependencyBuffers.get_allocator().resource());
        cachedDependencyBuffers.back().buffers.assign(
            dependencies.buffers.begin(), dependencies.buffers.end());
      }
      cachedBatchGpuData = meshletBatchGpuData_;
      staticBatchCache_.meshletCandidateCount = totalCandidateCount;
      staticBatchCache_.meshletDispatchSignature = dispatchSignature;
      staticBatchCache_.meshletDispatchCacheValid = true;
    }
    return totalCandidateCount;
  };
  if (meshletDepthPrepassEnabled) {
    NURI_PROFILER_ZONE("OpaqueRenderer.meshlet_depth_prepass_build",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    meshletDepthPrepassDependencyBuffers_ = passDependencyBuffers_;
    meshletDepthPrepassDependencyBufferAccessModes_ =
        passDependencyBufferAccessModes_;
    appendUniqueDependency(meshletDepthPrepassDependencyBuffers_,
                           meshletDepthPrepassDependencyBufferAccessModes_,
                           instanceLodBoundsBuffer_->handle(),
                           RenderGraphAccessMode::Read);
    buildMeshletDispatches(
        meshletDepthPrepassDispatchItems_, meshletDepthPrepassPushConstants_,
        meshletDepthPrepassDispatchDependencyBuffers_,
        selectMeshletDepthPipeline(coverage, false, false),
        selectMeshletDepthPipeline(coverage, false, true),
        selectMeshletDepthPipeline(coverage, true, false),
        selectMeshletDepthPipeline(coverage, true, true), CompareOp::Less, true,
        "OpaqueMeshletDepth", kMeshDebugColor, MeshletOcclusionSource::Disabled,
        0u, 0u, 0u, false, false, MeshletSourceFilter{});
    if (meshletDepthPrepassDispatchItems_.empty()) {
      meshletDepthPrepassEnabled = false;
      meshletDepthPrepassDependencyBuffers_.clear();
      meshletDepthPrepassDependencyBufferAccessModes_.clear();
    }
    NURI_PROFILER_ZONE_END();
  }
  if (meshletNormalPrepassEnabled) {
    NURI_PROFILER_ZONE("OpaqueRenderer.meshlet_normal_prepass_build",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    meshletNormalPrepassDependencyBuffers_ = passDependencyBuffers_;
    meshletNormalPrepassDependencyBufferAccessModes_ =
        passDependencyBufferAccessModes_;
    appendUniqueDependency(meshletNormalPrepassDependencyBuffers_,
                           meshletNormalPrepassDependencyBufferAccessModes_,
                           instanceLodBoundsBuffer_->handle(),
                           RenderGraphAccessMode::Read);
    const CompareOp normalPrepassCompare = meshletNormalPrepassWritesDepth
                                               ? CompareOp::Less
                                               : CompareOp::LessEqual;
    const MeshletPipelineHandle singleSidedInputPipeline =
        materialNormalInput
            ? meshletNormalPipelines_[0]
            : selectMeshletDepthPipeline(CoverageMode::Sample1, false, false);
    const MeshletPipelineHandle doubleSidedInputPipeline =
        materialNormalInput
            ? meshletNormalPipelines_[1]
            : selectMeshletDepthPipeline(CoverageMode::Sample1, false, true);
    const MeshletPipelineHandle alphaInputPipeline =
        materialNormalInput
            ? meshletNormalPipelines_[2]
            : selectMeshletDepthPipeline(CoverageMode::Sample1, true, false);
    const MeshletPipelineHandle alphaDoubleSidedInputPipeline =
        materialNormalInput
            ? meshletNormalPipelines_[3]
            : selectMeshletDepthPipeline(CoverageMode::Sample1, true, true);
    buildMeshletDispatches(
        meshletNormalPrepassDispatchItems_, meshletNormalPrepassPushConstants_,
        meshletNormalPrepassDispatchDependencyBuffers_,
        singleSidedInputPipeline, doubleSidedInputPipeline, alphaInputPipeline,
        alphaDoubleSidedInputPipeline, normalPrepassCompare,
        meshletNormalPrepassWritesDepth,
        materialNormalInput ? "OpaqueMeshletNormals" : "OpaqueMeshletGTAODepth",
        0xff66ddff, MeshletOcclusionSource::Disabled, 0u, 0u, 0u, false, false,
        MeshletSourceFilter{.materialNormalUsesAuxPipelines =
                                materialNormalInput});
    if (meshletNormalPrepassDispatchItems_.empty()) {
      meshletNormalPrepassEnabled = false;
      meshletNormalPrepassWritesDepth = false;
      meshletNormalPrepassDependencyBuffers_.clear();
      meshletNormalPrepassDependencyBufferAccessModes_.clear();
    }
    NURI_PROFILER_ZONE_END();
  }
  frame.metrics.opaque.msaaDepthPrepassDraws =
      msaaSelected ? saturateToU32(depthPrepassDrawItems_.size()) : 0u;
  frame.metrics.opaque.msaaDepthPrepassDispatches =
      msaaSelected ? saturateToU32(meshletDepthPrepassDispatchItems_.size())
                   : 0u;
  frame.metrics.opaque.gtaoAuxiliaryPrepassDraws =
      msaaGtaoAuxiliaryPrepass ? saturateToU32(normalPrepassDrawItems_.size())
                               : 0u;
  frame.metrics.opaque.gtaoAuxiliaryPrepassDispatches =
      msaaGtaoAuxiliaryPrepass
          ? saturateToU32(meshletNormalPrepassDispatchItems_.size())
          : 0u;
  frame.metrics.opaque.gtaoAuxiliaryWritesSingleSampleDepth =
      msaaGtaoAuxiliaryPrepass && (!normalPrepassDrawItems_.empty() ||
                                   !meshletNormalPrepassDispatchItems_.empty())
          ? 1u
          : 0u;
  std::span<const DrawItem> finalPassDrawItems = shadedBaseDrawItems;
  if (!meshletActive && wireframeOnlyRequested && !baseDrawItems.empty()) {
    const bool lineOverlayAvailable = nuri::isValid(
        overlayPipeline(OverlayPipelineKind::Wireframe, coverage));
    const bool lineTessOverlayAvailable = nuri::isValid(
        overlayPipeline(OverlayPipelineKind::TessWireframe, coverage));
    overlayDrawItems_.reserve(baseDrawItems.size());
    for (const DrawItem &baseItem : baseDrawItems) {
      const bool isTessDraw = isTessPipeline(baseItem.pipeline);
      RenderPipelineHandle wireframePipeline{};
      bool usedFallback = false;
      if (isTessDraw && lineTessOverlayAvailable) {
        wireframePipeline =
            overlayPipeline(OverlayPipelineKind::TessWireframe, coverage);
      } else if (lineOverlayAvailable) {
        wireframePipeline =
            overlayPipeline(OverlayPipelineKind::Wireframe, coverage);
        usedFallback = isTessDraw;
      }
      if (!nuri::isValid(wireframePipeline)) {
        continue;
      }
      DrawItem wireframeItem = baseItem;
      wireframeItem.pipeline = wireframePipeline;
      if (isTessDraw) {
        wireframeItem.debugLabel = usedFallback
                                       ? "OpaqueMeshTessWireframeOnlyFallback"
                                       : "OpaqueMeshTessWireframeOnly";
      } else {
        wireframeItem.debugLabel = "OpaqueMeshWireframeOnly";
      }
      overlayDrawItems_.push_back(wireframeItem);
      ++debugOverlayDraws;
      if (usedFallback) {
        ++debugOverlayFallbackDraws;
      }
    }
    if (!overlayDrawItems_.empty()) {
      finalPassDrawItems = std::span<const DrawItem>(overlayDrawItems_.data(),
                                                     overlayDrawItems_.size());
    }
  } else {
    if (!meshletActive && overlayRequested && !baseDrawItems.empty()) {
      const bool gsOverlayAvailable = nuri::isValid(
          overlayPipeline(OverlayPipelineKind::Geometry, coverage));
      const bool gsTessOverlayAvailable = nuri::isValid(
          overlayPipeline(OverlayPipelineKind::TessGeometry, coverage));
      const bool lineOverlayAvailable =
          !gsOverlayAvailable && nuri::isValid(overlayPipeline(
                                     OverlayPipelineKind::Wireframe, coverage));
      const bool lineTessOverlayAvailable =
          !gsTessOverlayAvailable &&
          nuri::isValid(
              overlayPipeline(OverlayPipelineKind::TessWireframe, coverage));
      overlayDrawItems_.reserve(baseDrawItems.size());
      for (const DrawItem &baseItem : baseDrawItems) {
        const bool isTessDraw = isTessPipeline(baseItem.pipeline);
        RenderPipelineHandle overlayHandle{};
        bool usedFallback = false;
        if (isTessDraw) {
          if (gsTessOverlayAvailable) {
            overlayHandle =
                overlayPipeline(OverlayPipelineKind::TessGeometry, coverage);
          } else if (lineTessOverlayAvailable) {
            overlayHandle =
                overlayPipeline(OverlayPipelineKind::TessWireframe, coverage);
            usedFallback = true;
          }
        } else {
          if (gsOverlayAvailable) {
            overlayHandle =
                overlayPipeline(OverlayPipelineKind::Geometry, coverage);
          } else if (lineOverlayAvailable) {
            overlayHandle =
                overlayPipeline(OverlayPipelineKind::Wireframe, coverage);
            usedFallback = true;
          }
        }
        if (!nuri::isValid(overlayHandle)) {
          continue;
        }
        DrawItem overlayItem = baseItem;
        overlayItem.pipeline = overlayHandle;
        overlayItem.useDepthState = true;
        overlayItem.depthState = {.compareOp = CompareOp::LessEqual,
                                  .isDepthWriteEnabled = false};
        overlayItem.depthBiasEnable = true;
        overlayItem.depthBiasConstant = kOverlayDepthBiasConstant;
        overlayItem.depthBiasSlope = kOverlayDepthBiasSlope;
        overlayItem.depthBiasClamp = 0.0f;
        if (usedFallback) {
          overlayItem.debugLabel = isTessDraw ? "OpaqueMeshTessOverlayFallback"
                                              : "OpaqueMeshOverlayFallback";
        } else {
          overlayItem.debugLabel =
              isTessDraw ? "OpaqueMeshTessOverlay" : "OpaqueMeshOverlay";
        }
        overlayDrawItems_.push_back(overlayItem);
        ++debugOverlayDraws;
        if (usedFallback) {
          ++debugOverlayFallbackDraws;
        }
        if (patchHeatmapRequested && isTessDraw && !usedFallback) {
          ++debugPatchHeatmapDraws;
        }
      }
    }
    if (!overlayDrawItems_.empty()) {
      const size_t baseOffset = passDrawItems_.size();
      passDrawItems_.reserve(baseOffset + shadedBaseDrawItems.size() +
                             overlayDrawItems_.size());
      if (passDrawItems_.empty()) {
        passDrawItems_.insert(passDrawItems_.end(), shadedBaseDrawItems.begin(),
                              shadedBaseDrawItems.end());
      }
      passDrawItems_.insert(passDrawItems_.end(), overlayDrawItems_.begin(),
                            overlayDrawItems_.end());
      finalPassDrawItems = std::span<const DrawItem>(passDrawItems_.data(),
                                                     passDrawItems_.size());
    }
  }
  size_t indirectCommandCount = 0;
  for (const DrawItem &indirectDraw : indirectDrawItems_) {
    indirectCommandCount += indirectDraw.indirectDrawCount;
  }
  frame.metrics.opaque.totalInstances = saturateToU32(instanceCount);
  frame.metrics.opaque.visibleInstances = saturateToU32(remapCount);
  frame.metrics.opaque.instancedDraws = saturateToU32(drawItems_.size());
  frame.metrics.opaque.indirectDrawCalls =
      saturateToU32(indirectDrawItems_.size());
  frame.metrics.opaque.indirectCommands = saturateToU32(indirectCommandCount);
  frame.metrics.opaque.tessellatedDraws = saturateToU32(tessellatedDraws);
  frame.metrics.opaque.tessellatedInstances =
      saturateToU32(tessellatedInstances);
  frame.metrics.opaque.debugOverlayDraws = saturateToU32(debugOverlayDraws);
  frame.metrics.opaque.debugOverlayFallbackDraws =
      saturateToU32(debugOverlayFallbackDraws);
  frame.metrics.opaque.debugPatchHeatmapDraws =
      saturateToU32(debugPatchHeatmapDraws);
  frame.metrics.opaque.computeDispatches = saturateToU32(preDispatches_.size());
  frame.metrics.opaque.computeDispatchX = computeDispatchX;
  const size_t meshletNormalDepthPrepassDraws =
      meshletNormalPrepassWritesDepth
          ? meshletNormalPrepassDispatchItems_.size()
          : 0u;
  frame.metrics.opaque.depthPrepassDraws = saturateToU32(
      depthPrepassDrawItems_.size() + meshletDepthPrepassDispatchItems_.size() +
      meshletNormalDepthPrepassDraws);
  frame.metrics.opaque.depthPyramidLevels = 0u;
  frame.metrics.opaque.depthPrepassEnabled =
      (depthPrepassEnabled || meshletDepthPrepassEnabled ||
       meshletNormalPrepassWritesDepth || classicNormalPrepassWritesDepth)
          ? 1u
          : 0u;
  aoMetrics.inputPassDraws =
      saturateToU32(normalPrepassDrawItems_.size() +
                    meshletNormalPrepassDispatchItems_.size());
  aoMetrics.normalPrepassDraws =
      materialNormalInput ? aoMetrics.inputPassDraws : 0u;
  if (gpuMainCullingEnabled && !gpuVisibilityCandidateIndices.empty()) {
    VisibilityResolvedSettings mainVisibilitySettings = visibilitySettings;
    if (meshletActive) {
      mainVisibilitySettings.enableGpuIndirectDraw = false;
    }
    const size_t visibilityPassCountBefore = out.size();
    auto visibilityPassResult = appendGpuVisibilityMainPass(
        frame, frameSlot, visibilityCandidates, visibilityCandidateGpuData,
        gpuVisibilityCandidateIndices, visibilityRequest,
        mainVisibilitySettings, validateGpuMainVisibility, out);
    if (visibilityPassResult.hasError()) {
      return visibilityPassResult;
    }
    visibilityCounterPreparedForFrame = out.size() != visibilityPassCountBefore;
  }
  const auto makePreparedPass =
      [&](std::span<const DrawItem> draws,
          std::span<const MeshDispatchItem> meshDispatches,
          std::span<const BufferHandle> dependencyBuffers,
          std::span<const RenderGraphAccessMode> dependencyBufferAccessModes,
          std::span<const TextureHandle> dependencyTextures,
          std::string_view debugLabel,
          uint32_t debugColor) -> PreparedGraphPass & {
    PreparedGraphPass &pass =
        out.emplace_back(drawItems_.get_allocator().resource());
    pass.desc.dependencyBuffers = dependencyBuffers;
    pass.desc.dependencyBufferAccessModes = dependencyBufferAccessModes;
    pass.desc.dependencyTextures = dependencyTextures;
    pass.desc.draws = draws;
    pass.desc.meshDispatches = meshDispatches;
    pass.desc.drawBuffersPreResolved = true;
    pass.desc.preResolvedDrawBuffers = preResolvedDrawBuffers_;
    pass.desc.debugLabel = debugLabel;
    pass.desc.debugColor = debugColor;
    return pass;
  };
  const auto makeDispatchPass =
      [&](std::span<const ComputeDispatchItem> dispatches,
          std::span<const BufferHandle> dependencyBuffers,
          std::span<const TextureHandle> dependencyTextures,
          std::string_view debugLabel,
          GpuTimingScope timingScope) -> PreparedGraphPass & {
    PreparedGraphPass &pass =
        out.emplace_back(drawItems_.get_allocator().resource());
    pass.desc.preDispatches = dispatches;
    pass.desc.dependencyBuffers = dependencyBuffers;
    pass.desc.dependencyTextures = dependencyTextures;
    pass.desc.debugLabel = debugLabel;
    pass.desc.debugColor = kComputeDispatchColor;
    pass.desc.gpuTimingScope = timingScope;
    pass.desc.borrowPayload = false;
    return pass;
  };
  bool pickPassSubmitted = false;
  if (pendingPickRequest_.has_value() && nuri::isValid(pickIdTexture_) &&
      nuri::isValid(meshPickPipelines_[surfaceVariantIndex(false, false)])) {
    NURI_PROFILER_ZONE("OpaqueRenderer.pick_pass",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    pickDrawItems_.clear();
    pickDrawItems_.reserve(baseDrawItems.size());
    for (const DrawItem &baseItem : baseDrawItems) {
      DrawItem pickItem = baseItem;
      pickItem.pipeline = selectPickPipeline(baseItem.pipeline);
      pickItem.debugLabel = "OpaqueMeshPick";
      pickItem.debugColor = kOpaquePassDebugColor;
      pickDrawItems_.push_back(pickItem);
    }
    int32_t framebufferWidth = 0;
    int32_t framebufferHeight = 0;
    gpu_.getFramebufferSize(framebufferWidth, framebufferHeight);
    const uint32_t safeWidth =
        static_cast<uint32_t>(std::max(framebufferWidth, 1));
    const uint32_t safeHeight =
        static_cast<uint32_t>(std::max(framebufferHeight, 1));
    pendingPickRequest_->x = std::min(pendingPickRequest_->x, safeWidth - 1u);
    pendingPickRequest_->y = std::min(pendingPickRequest_->y, safeHeight - 1u);
    PreparedGraphPass &pickPass =
        makePreparedPass(pickDrawItems_, {}, passDependencyBuffers_,
                         passDependencyBufferAccessModes_, {},
                         kOpaquePickPassLabel, kOpaquePassDebugColor);
    pickPass.desc.color = {.loadOp = LoadOp::Clear,
                           .storeOp = StoreOp::Store,
                           .clearColor = {0.0f, 0.0f, 0.0f, 0.0f}};
    pickPass.colorTextureHandle = pickIdTexture_;
    pickPass.desc.depth = {.loadOp = LoadOp::Clear,
                           .storeOp = StoreOp::Store,
                           .clearDepth = kClearDepthOne,
                           .clearStencil = 0};
    pickPass.depthTextureHandle = sceneDepthTexture;
    pickPass.desc.preDispatches = std::span<const ComputeDispatchItem>(
        preDispatches_.data(), preDispatches_.size());
    pickPass.desc.borrowPayload = preDispatches_.empty();
    pickPass.phase = PreparedPassPhase::Pick;
    pickPass.kind = PreparedPassKind::Pick;
    inFlightPickReadback_ = InFlightPickReadback{
        .request = *pendingPickRequest_, .submissionFrame = frame.frameIndex};
    pendingPickRequest_.reset();
    pickPassSubmitted = true;
    NURI_PROFILER_ZONE_END();
  }
  if (depthPrepassEnabled) {
    PreparedGraphPass &depthPass = makePreparedPass(
        depthPrepassDrawItems_, {}, passDependencyBuffers_,
        passDependencyBufferAccessModes_, passDependencyTextures_,
        "Opaque Depth Pre-Pass", kOpaquePassDebugColor);
    depthPass.desc.hasColorAttachment = false;
    depthPass.desc.depth = {.loadOp = LoadOp::Clear,
                            .storeOp = StoreOp::Store,
                            .clearDepth = kClearDepthOne,
                            .clearStencil = 0};
    depthPass.depthTextureHandle = sceneDepthTarget;
    if (!pickPassSubmitted) {
      depthPass.desc.preDispatches = std::span<const ComputeDispatchItem>(
          preDispatches_.data(), preDispatches_.size());
    }
    depthPass.desc.gpuTimingScope = GpuTimingScope::OpaqueDepth;
    depthPass.desc.markImplicitOutputSideEffect = true;
    depthPass.desc.borrowPayload = pickPassSubmitted || preDispatches_.empty();
    depthPass.phase = PreparedPassPhase::PreLighting;
    depthPass.kind = PreparedPassKind::Depth;
    depthPass.publishesDepth = true;
  }
  if (meshletDepthPrepassEnabled) {
    PreparedGraphPass &depthPass = makePreparedPass(
        {}, meshletDepthPrepassDispatchItems_,
        meshletDepthPrepassDependencyBuffers_,
        meshletDepthPrepassDependencyBufferAccessModes_,
        passDependencyTextures_, "Opaque Meshlet Depth Pre-Pass",
        kOpaquePassDebugColor);
    depthPass.desc.hasColorAttachment = false;
    depthPass.desc.depth = {.loadOp = LoadOp::Clear,
                            .storeOp = StoreOp::Store,
                            .clearDepth = kClearDepthOne,
                            .clearStencil = 0};
    depthPass.depthTextureHandle = sceneDepthTarget;
    if (!pickPassSubmitted) {
      depthPass.desc.preDispatches = std::span<const ComputeDispatchItem>(
          preDispatches_.data(), preDispatches_.size());
    }
    depthPass.desc.gpuTimingScope = GpuTimingScope::OpaqueDepth;
    depthPass.desc.markImplicitOutputSideEffect = true;
    depthPass.desc.borrowPayload = pickPassSubmitted || preDispatches_.empty();
    depthPass.phase = PreparedPassPhase::PreLighting;
    depthPass.kind = PreparedPassKind::Depth;
    depthPass.publishesDepth = true;
  }
  if (transmissionVisibilityDepthEnabled) {
    PreparedGraphPass &visibilityDepthPass = makePreparedPass(
        transmissionVisibilityDepthDrawItems_, {}, passDependencyBuffers_,
        passDependencyBufferAccessModes_, passDependencyTextures_,
        "Opaque Transmission Visibility Depth", kOpaquePassDebugColor);
    visibilityDepthPass.desc.hasColorAttachment = false;
    visibilityDepthPass.desc.depth = {.loadOp = LoadOp::Clear,
                                      .storeOp = StoreOp::Store,
                                      .clearDepth = kClearDepthOne,
                                      .clearStencil = 0};
    visibilityDepthPass.depthTextureHandle =
        transmissionVisibilityDepthTexture_;
    if (!pickPassSubmitted && !depthPrepassEnabled &&
        !meshletDepthPrepassEnabled) {
      visibilityDepthPass.desc.preDispatches =
          std::span<const ComputeDispatchItem>(preDispatches_.data(),
                                               preDispatches_.size());
    }
    visibilityDepthPass.desc.gpuTimingScope = GpuTimingScope::Opaque;
    visibilityDepthPass.desc.borrowPayload =
        pickPassSubmitted || depthPrepassEnabled ||
        meshletDepthPrepassEnabled || preDispatches_.empty();
    visibilityDepthPass.phase = PreparedPassPhase::PreLighting;
    visibilityDepthPass.kind = PreparedPassKind::TransmissionDepth;
  }
  const bool preDispatchSubmittedBeforeNormal =
      pickPassSubmitted || depthPrepassEnabled || meshletDepthPrepassEnabled ||
      transmissionVisibilityDepthEnabled;
  const bool hasMeshletNormalPrepass =
      meshletNormalPrepassEnabled &&
      !meshletNormalPrepassDispatchItems_.empty();
  const bool hasClassicNormalPrepass =
      normalPrepassEnabled && !normalPrepassDrawItems_.empty();
  const bool hasClassicNormalDepthPrepass =
      hasClassicNormalPrepass && classicNormalPrepassWritesDepth;
  if (hasMeshletNormalPrepass &&
      (!materialNormalInput ||
       nuri::isValid(frame.sharedResources.normalTexture))) {
    PreparedGraphPass &normalPass = makePreparedPass(
        {}, meshletNormalPrepassDispatchItems_,
        meshletNormalPrepassDependencyBuffers_,
        meshletNormalPrepassDependencyBufferAccessModes_,
        passDependencyTextures_,
        materialNormalInput ? "Opaque Meshlet Material Normal Pre-Pass"
                            : "Opaque Meshlet GTAO Depth Input Pass",
        0xff66ddff);
    normalPass.desc.hasColorAttachment = materialNormalInput;
    if (materialNormalInput) {
      normalPass.desc.color = {.loadOp = LoadOp::Clear,
                               .storeOp = StoreOp::Store,
                               .clearColor = kFrameCompositionNormalClearValue};
      normalPass.colorTextureHandle = frame.sharedResources.normalTexture;
    }
    normalPass.desc.depth = {.loadOp = meshletNormalPrepassWritesDepth
                                           ? LoadOp::Clear
                                           : LoadOp::Load,
                             .storeOp = StoreOp::Store,
                             .clearDepth = kClearDepthOne,
                             .clearStencil = 0};
    normalPass.depthTextureHandle =
        msaaGtaoAuxiliaryPrepass ? frame.sharedResources.sceneDepthTexture
                                 : sceneDepthTarget;
    if (meshletNormalPrepassWritesDepth && !preDispatchSubmittedBeforeNormal) {
      normalPass.desc.preDispatches = std::span<const ComputeDispatchItem>(
          preDispatches_.data(), preDispatches_.size());
    }
    normalPass.desc.gpuTimingScope = GpuTimingScope::OpaqueNormal;
    normalPass.desc.markImplicitOutputSideEffect = true;
    normalPass.desc.borrowPayload = !meshletNormalPrepassWritesDepth ||
                                    preDispatchSubmittedBeforeNormal ||
                                    preDispatches_.empty();
    normalPass.phase = PreparedPassPhase::PreLighting;
    normalPass.kind = PreparedPassKind::Normal;
    normalPass.publishesDepth = meshletNormalPrepassWritesDepth;
    aoMetrics.inputPassDraws =
        saturateToU32(meshletNormalPrepassDispatchItems_.size());
    aoMetrics.normalPrepassDraws =
        materialNormalInput ? aoMetrics.inputPassDraws : 0u;
  } else if (hasClassicNormalPrepass &&
             (!materialNormalInput ||
              nuri::isValid(frame.sharedResources.normalTexture))) {
    PreparedGraphPass &normalPass = makePreparedPass(
        normalPrepassDrawItems_, {}, passDependencyBuffers_,
        passDependencyBufferAccessModes_, passDependencyTextures_,
        materialNormalInput ? "Opaque Material Normal Pre-Pass"
                            : "Opaque GTAO Depth Input Pass",
        0xff66ddff);
    normalPass.desc.hasColorAttachment = materialNormalInput;
    if (materialNormalInput) {
      normalPass.desc.color = {.loadOp = LoadOp::Clear,
                               .storeOp = StoreOp::Store,
                               .clearColor = kFrameCompositionNormalClearValue};
      normalPass.colorTextureHandle = frame.sharedResources.normalTexture;
    }
    normalPass.desc.depth = {
        .loadOp = hasClassicNormalDepthPrepass ? LoadOp::Clear : LoadOp::Load,
        .storeOp = StoreOp::Store,
        .clearDepth = kClearDepthOne,
        .clearStencil = 0};
    normalPass.depthTextureHandle =
        msaaGtaoAuxiliaryPrepass ? frame.sharedResources.sceneDepthTexture
                                 : sceneDepthTarget;
    if (hasClassicNormalDepthPrepass) {
      if (!preDispatchSubmittedBeforeNormal) {
        normalPass.desc.preDispatches = std::span<const ComputeDispatchItem>(
            preDispatches_.data(), preDispatches_.size());
      }
    }
    normalPass.desc.gpuTimingScope = GpuTimingScope::OpaqueNormal;
    normalPass.desc.markImplicitOutputSideEffect = true;
    normalPass.desc.borrowPayload = !hasClassicNormalDepthPrepass ||
                                    preDispatchSubmittedBeforeNormal ||
                                    preDispatches_.empty();
    normalPass.phase = PreparedPassPhase::PreLighting;
    normalPass.kind = PreparedPassKind::Normal;
    normalPass.publishesDepth = hasClassicNormalDepthPrepass;
    aoMetrics.inputPassDraws = saturateToU32(normalPrepassDrawItems_.size());
    aoMetrics.normalPrepassDraws =
        materialNormalInput ? aoMetrics.inputPassDraws : 0u;
  } else if (ambientOcclusionSettings.active) {
    aoMetrics.active = false;
    aoMetrics.disabledReason = AmbientOcclusionDisabledReason::MissingResources;
  }
  const bool preDispatchSubmittedBeforeMain =
      pickPassSubmitted || depthPrepassEnabled || meshletDepthPrepassEnabled ||
      transmissionVisibilityDepthEnabled || meshletNormalPrepassWritesDepth ||
      hasClassicNormalDepthPrepass;
  const bool sceneDepthAvailableForPyramid =
      pickPassSubmitted || depthPrepassEnabled || meshletDepthPrepassEnabled ||
      meshletNormalPrepassWritesDepth || hasClassicNormalDepthPrepass;
  const bool depthPyramidEnabled =
      !msaaSelected && requiresDepthPyramid && sceneDepthAvailableForPyramid &&
      (meshletActive ? !drawItems_.empty() : !baseDrawItems.empty());
  frame.metrics.shadow.sdsmComputePassCount = 0u;
  if (depthPyramidEnabled) {
    NURI_PROFILER_ZONE("OpaqueRenderer.depth_pyramid_build",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    depthPyramidPushConstants_.clear();
    depthPyramidDrawItems_.clear();
    depthPyramidDependencyTextures_.clear();
    depthPyramidPushConstants_.reserve(sceneDepthPyramidLevelCount_ + 1u);
    depthPyramidDrawItems_.reserve(sceneDepthPyramidLevelCount_ + 1u);
    depthPyramidDependencyTextures_.reserve(sceneDepthPyramidLevelCount_ + 1u);
    const uint32_t samplerId = gpu_.getSamplerBindlessIndex(sceneDepthSampler_);
    bool currentDepthVerificationBuilt = false;
    if (meshletCurrentFrameOcclusionRequested &&
        currentDepthVerificationTexId != kInvalidTextureBindlessIndex) {
      const uint32_t sourceTexId =
          gpu_.getTextureBindlessIndex(sceneDepthTexture);
      depthPyramidDependencyTextures_.push_back(sceneDepthTexture);
      depthPyramidPushConstants_.push_back(
          glm::uvec4(sourceTexId, samplerId, 2u, 0u));
      DrawItem draw{};
      draw.pipeline = currentFrameDepthVerificationPipelineHandle_;
      draw.vertexCount = 3u;
      draw.pushConstants =
          std::span<const std::byte>(reinterpret_cast<const std::byte *>(
                                         &depthPyramidPushConstants_.back()),
                                     sizeof(glm::uvec4));
      draw.debugLabel = "OpaqueCurrentFrameDepthVerification";
      draw.debugColor = kOpaquePassDebugColor;
      depthPyramidDrawItems_.push_back(draw);
      PreparedGraphPass &verificationPass =
          out.emplace_back(drawItems_.get_allocator().resource());
      verificationPass.desc.color = {.loadOp = LoadOp::Clear,
                                     .storeOp = StoreOp::Store,
                                     .clearColor = {1.0f, 1.0f, 0.0f, 0.0f}};
      verificationPass.colorTextureHandle =
          currentFrameDepthVerificationTexture_;
      verificationPass.desc.dependencyTextures = std::span<const TextureHandle>(
          &depthPyramidDependencyTextures_.back(), 1u);
      verificationPass.desc.draws =
          std::span<const DrawItem>(&depthPyramidDrawItems_.back(), 1u);
      verificationPass.desc.drawBuffersPreResolved = true;
      verificationPass.desc.gpuTimingScope = GpuTimingScope::OpaqueDepth;
      verificationPass.desc.debugLabel =
          "Opaque Current-Frame Depth Verification";
      verificationPass.desc.debugColor = kOpaquePassDebugColor;
      verificationPass.desc.borrowPayload = true;
      verificationPass.phase = PreparedPassPhase::PreLighting;
      currentDepthVerificationBuilt = true;
    }
    for (uint32_t level = 0u; level < sceneDepthPyramidLevelCount_; ++level) {
      const TextureHandle sourceTexture =
          level == 0u ? sceneDepthTexture
                      : sceneDepthPyramidTextures_[level - 1u];
      const TextureHandle destinationTexture =
          sceneDepthPyramidTextures_[level];
      const uint32_t sourceTexId = gpu_.getTextureBindlessIndex(sourceTexture);
      depthPyramidDependencyTextures_.push_back(sourceTexture);
      depthPyramidPushConstants_.push_back(
          glm::uvec4(sourceTexId, samplerId, level == 0u ? 1u : 0u, 0u));
      DrawItem draw{};
      draw.pipeline = depthPyramidPipelineHandle_;
      draw.vertexCount = 3u;
      draw.pushConstants =
          std::span<const std::byte>(reinterpret_cast<const std::byte *>(
                                         &depthPyramidPushConstants_.back()),
                                     sizeof(glm::uvec4));
      draw.debugLabel = "OpaqueDepthMinMaxPyramid";
      draw.debugColor = kOpaquePassDebugColor;
      depthPyramidDrawItems_.push_back(draw);
      PreparedGraphPass &pyramidPass =
          out.emplace_back(drawItems_.get_allocator().resource());
      pyramidPass.desc.color = {.loadOp = LoadOp::Clear,
                                .storeOp = StoreOp::Store,
                                .clearColor = {1.0f, 1.0f, 0.0f, 0.0f}};
      pyramidPass.colorTextureHandle = destinationTexture;
      pyramidPass.desc.dependencyTextures = std::span<const TextureHandle>(
          &depthPyramidDependencyTextures_.back(), 1u);
      pyramidPass.desc.draws =
          std::span<const DrawItem>(&depthPyramidDrawItems_.back(), 1u);
      pyramidPass.desc.drawBuffersPreResolved = true;
      pyramidPass.desc.gpuTimingScope = GpuTimingScope::OpaqueDepth;
      pyramidPass.desc.debugLabel = "Opaque Depth MinMax Pyramid";
      pyramidPass.desc.debugColor = kOpaquePassDebugColor;
      pyramidPass.desc.borrowPayload = true;
      pyramidPass.phase = PreparedPassPhase::PreLighting;
      pyramidPass.kind = PreparedPassKind::DepthPyramid;
      pyramidPass.depthPyramidLevel = level;
    }
    frame.metrics.opaque.depthPyramidLevels = sceneDepthPyramidLevelCount_;
    frame.metrics.opaque.depthPyramidActive =
        sceneDepthPyramidLevelCount_ != 0u ? 1u : 0u;
    sceneDepthPyramidSourceFrameIndex_ = frame.frameIndex;
    sceneDepthPyramidSourceViewProj_ = frame.camera.currentUnjitteredViewProj;
    meshletCurrentFrameOcclusionAvailable =
        meshletCurrentFrameOcclusionRequested && currentDepthVerificationBuilt;
    if (settings.shadow.enabled &&
        frame.sharedResources.shadowSdsmGpuReduceTarget.has_value() &&
        nuri::isValid(frame.sharedResources.shadowSdsmGpuReducePipeline)) {
      const TextureHandle reduceSourceTexture =
          sceneDepthPyramidTextures_[sceneDepthPyramidLevelCount_ - 1u];
      const uint32_t reduceSourceTexId =
          gpu_.getTextureBindlessIndex(reduceSourceTexture);
      shadowSdsmReducePushConstants_.clear();
      shadowSdsmReduceDispatches_.clear();
      shadowSdsmReduceDependencyBuffers_.clear();
      shadowSdsmReduceDependencyTextures_.clear();
      shadowSdsmReducePushConstants_.reserve(1u);
      shadowSdsmReduceDispatches_.reserve(1u);
      shadowSdsmReduceDependencyBuffers_.reserve(1u);
      shadowSdsmReduceDependencyTextures_.reserve(1u);
      shadowSdsmReducePushConstants_.push_back(ShadowSdsmReducePushConstants{
          .resultBufferAddress =
              frame.sharedResources.shadowSdsmGpuReduceTarget->bufferAddress,
          .sourceTexId = reduceSourceTexId,
          .sourceFrameIndex = static_cast<uint32_t>(frame.frameIndex),
      });
      shadowSdsmReduceDependencyBuffers_.push_back(
          frame.sharedResources.shadowSdsmGpuReduceTarget->buffer);
      shadowSdsmReduceDependencyTextures_.push_back(reduceSourceTexture);
      ComputeDispatchItem dispatch{};
      dispatch.pipeline = frame.sharedResources.shadowSdsmGpuReducePipeline;
      dispatch.dispatch = {.x = 1u, .y = 1u, .z = 1u};
      dispatch.pushConstants = std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(
              &shadowSdsmReducePushConstants_.back()),
          sizeof(ShadowSdsmReducePushConstants));
      dispatch.dependencyBuffers = std::span<const BufferHandle>(
          shadowSdsmReduceDependencyBuffers_.data(),
          shadowSdsmReduceDependencyBuffers_.size());
      dispatch.dependencyTextures = std::span<const TextureHandle>(
          shadowSdsmReduceDependencyTextures_.data(),
          shadowSdsmReduceDependencyTextures_.size());
      dispatch.debugLabel = "Shadow SDSM Reduce";
      dispatch.debugColor = kComputeDispatchColor;
      shadowSdsmReduceDispatches_.push_back(dispatch);
      PreparedGraphPass &reducePass = makeDispatchPass(
          shadowSdsmReduceDispatches_, {}, shadowSdsmReduceDependencyTextures_,
          dispatch.debugLabel, GpuTimingScope::ShadowSdsm);
      reducePass.desc.executionMode = RenderPassExecutionMode::ComputeOnly;
      reducePass.desc.hasColorAttachment = false;
      reducePass.desc.markImplicitOutputSideEffect = true;
      reducePass.phase = PreparedPassPhase::PreLighting;
      frame.metrics.shadow.sdsmComputePassCount = 1u;
    }
    NURI_PROFILER_ZONE_END();
  } else {
    sceneDepthPyramidSourceFrameIndex_.reset();
    sceneDepthPyramidSourceViewProj_.reset();
  }
  NURI_PROFILER_ZONE("OpaqueRenderer.main_pass_finalize",
                     NURI_PROFILER_COLOR_CMD_DRAW);
  const size_t mainPassIndex = out.size();
  PreparedGraphPass &pass =
      out.emplace_back(drawItems_.get_allocator().resource());
  pass.desc.color = {.loadOp = LoadOp::Clear,
                     .storeOp = StoreOp::Store,
                     .clearColor = {kClearColorWhite, kClearColorWhite,
                                    kClearColorWhite, kClearColorWhite}};
  pass.desc.depth = {.loadOp =
                         (depthPrepassEnabled || meshletDepthPrepassEnabled ||
                          meshletNormalPreparesMainDepth ||
                          classicNormalPreparesMainDepth)
                             ? LoadOp::Load
                             : LoadOp::Clear,
                     .storeOp = StoreOp::Store,
                     .clearDepth = kClearDepthOne,
                     .clearStencil = 0};
  pass.depthTextureHandle = sceneDepthTarget;
  mainPassDependencyBuffers_ = passDependencyBuffers_;
  mainPassDependencyBufferAccessModes_ = passDependencyBufferAccessModes_;
  mainPassDependencyTextures_ = passDependencyTextures_;
  mainPassDependencyTextureAccessModes_.assign(
      mainPassDependencyTextures_.size(), RenderGraphAccessMode::Read);
  if ((meshletCounterFlags & kMeshletCounterFlagEnabled) != 0u) {
    appendUniqueDependency(
        mainPassDependencyBuffers_, mainPassDependencyBufferAccessModes_,
        meshletVisibilityCounterBuffer,
        RenderGraphAccessMode::Read | RenderGraphAccessMode::Write);
  }
  if ((sceneGpu->shadowFlags & kShadowFrameFlagEnabled) != 0u &&
      frame.sharedResources.shadowFrameGpuData.has_value()) {
    appendUniqueDependency(mainPassDependencyBuffers_,
                           mainPassDependencyBufferAccessModes_,
                           frame.sharedResources.shadowFrameGpuData->buffer,
                           RenderGraphAccessMode::Read);
    for (uint32_t cascadeIndex = 0u; cascadeIndex < kMaxShadowCascades;
         ++cascadeIndex) {
      const TextureHandle texture =
          frame.sharedResources.shadowCascadeTextures[cascadeIndex];
      if (!nuri::isValid(texture)) {
        continue;
      }
      const size_t oldTextureDependencyCount =
          mainPassDependencyTextures_.size();
      appendUniqueDependency(mainPassDependencyTextures_, texture);
      if (mainPassDependencyTextures_.size() != oldTextureDependencyCount) {
        mainPassDependencyTextureAccessModes_.push_back(
            RenderGraphAccessMode::Read);
      }
    }
  }
  if (ambientOcclusionSettings.active &&
      nuri::isValid(frame.sharedResources.ambientOcclusionTexture)) {
    const size_t oldTextureDependencyCount = mainPassDependencyTextures_.size();
    appendUniqueDependency(mainPassDependencyTextures_,
                           frame.sharedResources.ambientOcclusionTexture);
    if (mainPassDependencyTextures_.size() != oldTextureDependencyCount) {
      mainPassDependencyTextureAccessModes_.push_back(
          RenderGraphAccessMode::Read);
    }
  }
  const MeshletOcclusionPlan meshletOcclusion = [&]() {
    MeshletOcclusionPlan plan{};
    plan.pyramidLevelCount =
        std::min<uint32_t>(frame.sharedResources.sceneDepthPyramidLevelCount,
                           kMaxSceneDepthPyramidLevels);
    if (meshletCurrentFrameOcclusionAvailable) {
      plan.source = MeshletOcclusionSource::CurrentFrame;
      plan.sourceFrame = frame.frameIndex;
    } else if (meshletPreviousFrameOcclusionAvailable &&
               frame.sharedResources.sceneDepthPyramidSourceFrameIndex
                   .has_value()) {
      plan.source = MeshletOcclusionSource::PreviousFrame;
      plan.sourceFrame =
          *frame.sharedResources.sceneDepthPyramidSourceFrameIndex;
    }
    return plan;
  }();
  frame.metrics.opaque.hiZActive = meshletOcclusion.active() ? 1u : 0u;
  frame.metrics.opaque.hiZSourceFramePolicy =
      meshletOcclusion.usesCurrentFrame() ? HiZSourceFramePolicy::CurrentFrame
      : meshletOcclusion.source == MeshletOcclusionSource::PreviousFrame
          ? HiZSourceFramePolicy::PreviousFrame
          : HiZSourceFramePolicy::Disabled;
  if (meshletOcclusion.active()) {
    for (uint32_t level = 0u; level < meshletOcclusion.pyramidLevelCount;
         ++level) {
      const TextureHandle texture =
          frame.sharedResources.sceneDepthPyramidTextures[level];
      const size_t oldTextureDependencyCount =
          mainPassDependencyTextures_.size();
      appendUniqueDependency(mainPassDependencyTextures_, texture);
      if (mainPassDependencyTextures_.size() != oldTextureDependencyCount) {
        mainPassDependencyTextureAccessModes_.push_back(
            RenderGraphAccessMode::Read);
      }
    }
    if (meshletOcclusion.usesCurrentFrame()) {
      const size_t oldTextureDependencyCount =
          mainPassDependencyTextures_.size();
      appendUniqueDependency(mainPassDependencyTextures_,
                             currentFrameDepthVerificationTexture_);
      if (mainPassDependencyTextures_.size() != oldTextureDependencyCount) {
        mainPassDependencyTextureAccessModes_.push_back(
            RenderGraphAccessMode::Read);
      }
    }
    frame.metrics.visibility.occlusionAvailable = 1u;
    frame.metrics.visibility.meshletOcclusionAvailable = 1u;
    frame.metrics.visibility.meshletOcclusionMode =
        static_cast<uint32_t>(visibilitySettings.occlusionMode);
    frame.metrics.visibility.meshletOcclusionSourceFrame =
        static_cast<uint32_t>(meshletOcclusion.sourceFrame);
    frame.metrics.visibility.meshletOcclusionSourceAge =
        static_cast<uint32_t>(frame.frameIndex - meshletOcclusion.sourceFrame);
    if (meshletOcclusion.usesCurrentFrame()) {
      frame.metrics.visibility.currentFrameHiZActive = 1u;
    }
  }
  meshletDispatchItems_.clear();
  meshletPushConstants_.clear();
  meshletDispatchDependencyBuffers_.clear();
  meshletCompactionWorkItems_.clear();
  meshletCompactionPushConstants_.clear();
  meshletCompactionDispatches_.clear();
  meshletCompactionCounterClear_.clear();
  meshletCompactionDependencyBuffers_.clear();
  meshletCompactionFinalizeDependencyBuffers_.clear();
  meshletCompactionDependencyTextures_.clear();
  bool meshletPreTaskCompactionUsed = false;
  if (meshletActive) {
    NURI_PROFILER_ZONE("OpaqueRenderer.meshlet_dispatch_build",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    appendUniqueDependency(
        mainPassDependencyBuffers_, mainPassDependencyBufferAccessModes_,
        instanceLodBoundsBuffer_->handle(), RenderGraphAccessMode::Read);
    const bool hasDepthPreparedMeshletMain =
        depthPrepassEnabled || meshletDepthPrepassEnabled ||
        meshletNormalPreparesMainDepth || classicNormalPreparesMainDepth;
    const CompareOp mainDepthCompare =
        hasDepthPreparedMeshletMain ? CompareOp::Equal : CompareOp::Less;
    const bool meshletIndirectDispatchEligible =
        !meshletHybridRoutingActive &&
        visibilitySettings.enableIndirectMeshDispatch &&
        visibilityCounterPreparedForFrame &&
        !gpuVisibilityCandidateIndices.empty() &&
        nuri::isValid(visibilityIndirectMeshDispatchComputePipeline_.get());
    const bool meshletPreTaskCompactionEligible =
        !meshletHybridRoutingActive &&
        visibilitySettings.enableIndirectMeshDispatch &&
        visibilitySettings.enableMeshletPreTaskCompaction &&
        meshletOcclusion.usesCurrentFrame() &&
        nuri::isValid(meshletCompactionComputePipeline_.get()) &&
        nuri::isValid(
            selectMeshletScenePipeline(true, coverage, false, false)) &&
        nuri::isValid(
            selectMeshletScenePipeline(true, coverage, false, true)) &&
        nuri::isValid(
            selectMeshletScenePipeline(true, coverage, true, false)) &&
        nuri::isValid(selectMeshletScenePipeline(true, coverage, true, true)) &&
        meshletVisibilityCounterBufferAddress != 0u;
    const uint64_t mainCandidateCount = buildMeshletDispatches(
        meshletDispatchItems_, meshletPushConstants_,
        meshletDispatchDependencyBuffers_,
        selectMeshletScenePipeline(meshletPreTaskCompactionEligible, coverage,
                                   false, false),
        selectMeshletScenePipeline(meshletPreTaskCompactionEligible, coverage,
                                   false, true),
        selectMeshletScenePipeline(meshletPreTaskCompactionEligible, coverage,
                                   true, false),
        selectMeshletScenePipeline(meshletPreTaskCompactionEligible, coverage,
                                   true, true),
        mainDepthCompare, !hasDepthPreparedMeshletMain, "OpaqueMeshlet",
        kMeshDebugColor, meshletOcclusion.source,
        meshletVisibilityCounterBufferAddress, meshletCounterFlags, 0u,
        !meshletHybridRoutingActive && (meshletIndirectDispatchEligible ||
                                        meshletPreTaskCompactionEligible),
        canUseStaticBatchCache && !meshletDepthPrepassEnabled &&
            !meshletOcclusion.active() && !meshletHybridRoutingActive,
        MeshletSourceFilter{
            .enabledSources = std::span<const uint8_t>(
                meshletMainSourceMask.data(), meshletMainSourceMask.size())});
    bool meshletIndirectDispatchUsed = false;
    constexpr size_t kMeshDispatchCommandBytes = sizeof(uint32_t) * 3u;
    if (meshletPreTaskCompactionEligible && !meshletDispatchItems_.empty()) {
      const size_t dispatchCount = meshletDispatchItems_.size();
      uint64_t workItemCount64 = 0u;
      uint64_t compactRecordCount64 = 0u;
      for (const MeshDispatchItem &dispatch : meshletDispatchItems_) {
        const uint64_t groupCount = static_cast<uint64_t>(dispatch.groupsX) *
                                    dispatch.groupsY * dispatch.groupsZ;
        workItemCount64 += groupCount;
        compactRecordCount64 +=
            groupCount * kOpaqueMeshletTaskCandidatesPerGroup;
      }
      const bool compactCountsAddressable =
          dispatchCount <=
              static_cast<size_t>(std::numeric_limits<uint32_t>::max()) &&
          workItemCount64 <= std::numeric_limits<uint32_t>::max() &&
          compactRecordCount64 <= std::numeric_limits<uint32_t>::max();
      if (compactCountsAddressable) {
        const size_t workItemCount = static_cast<size_t>(workItemCount64);
        const size_t compactRecordCount =
            static_cast<size_t>(compactRecordCount64);
        const size_t counterBytes = dispatchCount * sizeof(uint32_t);
        const size_t compactRecordOffset = (counterBytes + 15u) & ~size_t{15u};
        const size_t compactRecordBytes =
            compactRecordCount * sizeof(CompactedMeshletGpuData);
        const size_t commandBytes = dispatchCount * kMeshDispatchCommandBytes;
        auto indirectRingResult = ensureVisibilityMeshletIndirectRingCapacity(
            workItemCount * sizeof(MeshletCompactionWorkItemGpuData),
            commandBytes);
        if (indirectRingResult.hasError()) {
          return indirectRingResult;
        }
        auto compactRingResult = ensureMeshletCompactionRingCapacity(
            compactRecordOffset + compactRecordBytes);
        if (compactRingResult.hasError()) {
          return compactRingResult;
        }
        const DynamicBufferSlot &workSlot =
            bufferRings_[VisibilityMeshletDispatchRing][frameSlot];
        const DynamicBufferSlot &commandSlot =
            bufferRings_[VisibilityMeshletIndirectCommandRing][frameSlot];
        const DynamicBufferSlot &compactSlot =
            bufferRings_[MeshletCompactionRing][frameSlot];
        if (workSlot.buffer && workSlot.buffer->valid() && commandSlot.buffer &&
            commandSlot.buffer->valid() && compactSlot.buffer &&
            compactSlot.buffer->valid()) {
          meshletCompactionWorkItems_.reserve(workItemCount);
          uint32_t outputRecordBase = 0u;
          for (size_t i = 0; i < dispatchCount; ++i) {
            const MeshDispatchItem &dispatch = meshletDispatchItems_[i];
            const MeshletPushConstants &constants = meshletPushConstants_[i];
            const uint64_t dispatchRecordCapacity =
                static_cast<uint64_t>(dispatch.groupsX) * dispatch.groupsY *
                dispatch.groupsZ * kOpaqueMeshletTaskCandidatesPerGroup;
            for (uint32_t y = 0u; y < dispatch.groupsY; ++y) {
              for (uint32_t x = 0u; x < dispatch.groupsX; ++x) {
                meshletCompactionWorkItems_.push_back(
                    MeshletCompactionWorkItemGpuData{
                        .data = glm::uvec4(
                            constants.batchBase + y,
                            constants.candidateOffset +
                                x * kOpaqueMeshletTaskCandidatesPerGroup,
                            static_cast<uint32_t>(i), outputRecordBase),
                    });
              }
            }
            outputRecordBase += static_cast<uint32_t>(dispatchRecordCapacity);
          }
          meshletCompactionCounterClear_.assign(dispatchCount, 0u);
          const std::array updates{
              BufferUpdate{
                  .buffer = workSlot.buffer->handle(),
                  .data = std::as_bytes(
                      std::span<const MeshletCompactionWorkItemGpuData>(
                          meshletCompactionWorkItems_.data(),
                          meshletCompactionWorkItems_.size()))},
              BufferUpdate{.buffer = compactSlot.buffer->handle(),
                           .data = std::as_bytes(std::span<const uint32_t>(
                               meshletCompactionCounterClear_.data(),
                               meshletCompactionCounterClear_.size()))},
          };
          auto updateResult = gpu_.updateBuffers(updates);
          if (updateResult.hasError()) {
            return updateResult;
          }
          const BufferHandle workBuffer = workSlot.buffer->handle();
          const BufferHandle commandBuffer = commandSlot.buffer->handle();
          const BufferHandle compactBuffer = compactSlot.buffer->handle();
          const BufferHandle remapBuffer =
              bufferRings_[InstanceRemapRing][frameSlot].buffer->handle();
          const uint64_t workAddress = gpu_.getBufferDeviceAddress(workBuffer);
          const uint64_t commandAddress =
              gpu_.getBufferDeviceAddress(commandBuffer);
          const uint64_t compactAddress =
              gpu_.getBufferDeviceAddress(compactBuffer);
          if (workAddress != 0u && commandAddress != 0u &&
              compactAddress != 0u) {
            const MeshletCompactionPushConstants compactionConstants{
                .frameDataAddress = frameDataAddress,
                .instanceMatricesAddress = instanceMatricesAddress,
                .instanceRemapAddress = instanceRemapAddress,
                .instanceLodBoundsAddress = instanceLodBoundsAddress,
                .meshletBatchBufferAddress = meshletBatchBufferAddress,
                .workItemBufferAddress = workAddress,
                .compactedMeshletBufferAddress =
                    compactAddress + compactRecordOffset,
                .compactionCounterBufferAddress = compactAddress,
                .indirectCommandBufferAddress = commandAddress,
                .visibilityCounterBufferAddress =
                    meshletVisibilityCounterBufferAddress,
                .lodThresholds =
                    glm::vec4(sortedLodThresholds[0], sortedLodThresholds[1],
                              sortedLodThresholds[2], 0.0f),
                .workItemCount = static_cast<uint32_t>(workItemCount),
                .dispatchCount = static_cast<uint32_t>(dispatchCount),
                .compactGridWidth = maxTaskGroupsX,
                .sourceFrameIndex = static_cast<uint32_t>(frame.frameIndex),
                .meshletCounterFlags = meshletCounterFlags,
                .flags = 1u,
                .currentDepthVerificationTexId = currentDepthVerificationTexId,
                .currentDepthVerificationExtentPacked =
                    currentDepthVerificationExtentPacked,
            };
            meshletCompactionPushConstants_.push_back(compactionConstants);
            MeshletCompactionPushConstants finalizeConstants =
                compactionConstants;
            finalizeConstants.workItemCount =
                static_cast<uint32_t>(dispatchCount);
            finalizeConstants.flags = 2u;
            meshletCompactionPushConstants_.push_back(finalizeConstants);
            MeshletCompactionPushConstants publishConstants =
                compactionConstants;
            publishConstants.workItemCount = 1u;
            publishConstants.flags = 4u;
            meshletCompactionPushConstants_.push_back(publishConstants);
            const auto appendCompactionDependency =
                [&](BufferHandle handle, RenderGraphAccessMode accessMode) {
                  appendUniqueDependency(meshletCompactionDependencyBuffers_,
                                         handle);
                  appendUniqueDependency(mainPassDependencyBuffers_,
                                         mainPassDependencyBufferAccessModes_,
                                         handle, accessMode);
                };
            for (const BufferHandle handle :
                 {sceneGpu->buffer, instanceMatricesBufferHandle, remapBuffer,
                  instanceLodBoundsBuffer_->handle(), meshletBatchBufferHandle,
                  workBuffer}) {
              appendCompactionDependency(handle, RenderGraphAccessMode::Read);
            }
            for (const BufferHandle handle :
                 {compactBuffer, meshletVisibilityCounterBuffer}) {
              appendCompactionDependency(handle,
                                         RenderGraphAccessMode::Read |
                                             RenderGraphAccessMode::Write);
            }
            appendUniqueDependency(
                mainPassDependencyBuffers_,
                mainPassDependencyBufferAccessModes_, commandBuffer,
                RenderGraphAccessMode::Read | RenderGraphAccessMode::Write);
            meshletCompactionFinalizeDependencyBuffers_.assign(
                {compactBuffer, commandBuffer, meshletVisibilityCounterBuffer});
            for (const MeshletDispatchDependencyBuffers &dependencies :
                 meshletDispatchDependencyBuffers_) {
              for (const BufferHandle handle : dependencies.buffers) {
                appendUniqueDependency(mainPassDependencyBuffers_,
                                       mainPassDependencyBufferAccessModes_,
                                       handle, RenderGraphAccessMode::Read);
              }
            }
            for (uint32_t level = 0u;
                 level < meshletOcclusion.pyramidLevelCount; ++level) {
              appendUniqueDependency(
                  meshletCompactionDependencyTextures_,
                  frame.sharedResources.sceneDepthPyramidTextures[level]);
            }
            appendUniqueDependency(meshletCompactionDependencyTextures_,
                                   currentFrameDepthVerificationTexture_);
            ComputeDispatchItem compactionDispatch{};
            compactionDispatch.pipeline =
                meshletCompactionComputePipeline_.get();
            compactionDispatch.dispatch = {
                .x = static_cast<uint32_t>(
                    std::min<size_t>(workItemCount, maxTaskGroupsX)),
                .y = static_cast<uint32_t>(
                    (workItemCount + maxTaskGroupsX - 1u) / maxTaskGroupsX),
                .z = 1u,
            };
            compactionDispatch.pushConstants = std::span<const std::byte>(
                reinterpret_cast<const std::byte *>(
                    &meshletCompactionPushConstants_[0]),
                sizeof(MeshletCompactionPushConstants));
            compactionDispatch.dependencyBuffers =
                std::span<const BufferHandle>(
                    meshletCompactionDependencyBuffers_.data(),
                    meshletCompactionDependencyBuffers_.size());
            compactionDispatch.dependencyTextures =
                std::span<const TextureHandle>(
                    meshletCompactionDependencyTextures_.data(),
                    meshletCompactionDependencyTextures_.size());
            compactionDispatch.debugLabel =
                "Opaque Meshlet Pre-Task Compaction";
            compactionDispatch.debugColor = kComputeDispatchColor;
            meshletCompactionDispatches_.push_back(compactionDispatch);
            ComputeDispatchItem finalizeDispatch = compactionDispatch;
            finalizeDispatch.dispatch.x =
                (static_cast<uint32_t>(dispatchCount) + 31u) / 32u;
            finalizeDispatch.pushConstants = std::span<const std::byte>(
                reinterpret_cast<const std::byte *>(
                    &meshletCompactionPushConstants_[1]),
                sizeof(MeshletCompactionPushConstants));
            finalizeDispatch.debugLabel = "Opaque Meshlet Compaction Finalize";
            finalizeDispatch.dependencyBuffers = std::span<const BufferHandle>(
                meshletCompactionFinalizeDependencyBuffers_.data(),
                meshletCompactionFinalizeDependencyBuffers_.size());
            finalizeDispatch.dependencyTextures = {};
            meshletCompactionDispatches_.push_back(finalizeDispatch);
            ComputeDispatchItem publishDispatch = finalizeDispatch;
            publishDispatch.dispatch = {.x = 1u, .y = 1u, .z = 1u};
            publishDispatch.pushConstants = std::span<const std::byte>(
                reinterpret_cast<const std::byte *>(
                    &meshletCompactionPushConstants_[2]),
                sizeof(MeshletCompactionPushConstants));
            publishDispatch.debugLabel =
                "Opaque Meshlet Compaction Metrics Finalize";
            publishDispatch.dependencyBuffers = std::span<const BufferHandle>(
                meshletCompactionFinalizeDependencyBuffers_.data() + 2u, 1u);
            meshletCompactionDispatches_.push_back(publishDispatch);
            uint32_t patchedRecordBase = 0u;
            for (size_t i = 0; i < dispatchCount; ++i) {
              MeshDispatchItem &meshDispatch = meshletDispatchItems_[i];
              MeshletPushConstants &constants = meshletPushConstants_[i];
              for (const BufferHandle handle : {compactBuffer, commandBuffer}) {
                appendUniqueDependency(
                    meshletDispatchDependencyBuffers_[i].buffers, handle);
              }
              const uint64_t dispatchRecordCapacity =
                  static_cast<uint64_t>(meshDispatch.groupsX) *
                  meshDispatch.groupsY * meshDispatch.groupsZ *
                  kOpaqueMeshletTaskCandidatesPerGroup;
              constants.compactedMeshletBufferAddress =
                  compactAddress + compactRecordOffset;
              constants.compactionCounterBufferAddress = compactAddress;
              constants.batchBase = static_cast<uint32_t>(i);
              constants.candidateOffset = patchedRecordBase;
              constants.sourceFrameIndex = maxTaskGroupsX;
              constants.meshletCounterFlags = meshletCounterFlags;
              meshDispatch.pushConstants = std::span<const std::byte>(
                  reinterpret_cast<const std::byte *>(&constants),
                  sizeof(constants));
              meshDispatch.dependencyBuffers =
                  meshletDispatchDependencyBuffers_[i].span();
              meshDispatch.command = MeshDispatchCommandType::Indirect;
              meshDispatch.indirectBuffer = commandBuffer;
              meshDispatch.indirectBufferOffset =
                  static_cast<uint64_t>(i) * kMeshDispatchCommandBytes;
              meshDispatch.indirectDispatchCount = 1u;
              patchedRecordBase +=
                  static_cast<uint32_t>(dispatchRecordCapacity);
            }
            meshletPreTaskCompactionUsed = true;
            meshletIndirectDispatchUsed = true;
            frame.metrics.visibility.meshletPreTaskCompactionActive = 1u;
          }
        }
      }
    }
    if (!meshletPreTaskCompactionUsed && meshletIndirectDispatchEligible &&
        !meshletDispatchItems_.empty()) {
      const size_t dispatchCount = meshletDispatchItems_.size();
      if (dispatchCount <=
              static_cast<size_t>(std::numeric_limits<uint32_t>::max()) &&
          instanceCount <=
              static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        uint32_t candidateMapStride = 0u;
        for (const VisibilityCandidate &candidate : visibilityCandidates) {
          candidateMapStride =
              std::max(candidateMapStride, candidate.submeshIndex + 1u);
        }
        const size_t candidateMapCount =
            static_cast<size_t>(instanceCount) * candidateMapStride;
        const size_t dispatchMetadataBytes =
            dispatchCount * sizeof(VisibilityMeshletDispatchGpuData);
        const size_t candidateMapOffset = dispatchMetadataBytes;
        const size_t candidateMapBytes = candidateMapCount * sizeof(uint32_t);
        const bool candidateMapAddressable =
            candidateMapStride != 0u &&
            candidateMapCount <=
                static_cast<size_t>(std::numeric_limits<uint32_t>::max());
        if (candidateMapAddressable) {
          auto indirectRingResult = ensureVisibilityMeshletIndirectRingCapacity(
              dispatchMetadataBytes + candidateMapBytes,
              dispatchCount * kMeshDispatchCommandBytes);
          if (indirectRingResult.hasError()) {
            return indirectRingResult;
          }
        }
        const DynamicBufferSlot &dispatchSlot =
            bufferRings_[VisibilityMeshletDispatchRing][frameSlot];
        const DynamicBufferSlot &commandSlot =
            bufferRings_[VisibilityMeshletIndirectCommandRing][frameSlot];
        const DynamicBufferSlot &candidateSlot =
            bufferRings_[VisibilityCandidateRing][frameSlot];
        const DynamicBufferSlot &passSlot =
            bufferRings_[VisibilityPassRing][frameSlot];
        const DynamicBufferSlot &remapSlot =
            bufferRings_[InstanceRemapRing][frameSlot];
        if (candidateMapAddressable && dispatchSlot.buffer &&
            dispatchSlot.buffer->valid() && commandSlot.buffer &&
            commandSlot.buffer->valid() && candidateSlot.buffer &&
            candidateSlot.buffer->valid() && passSlot.buffer &&
            passSlot.buffer->valid() && remapSlot.buffer &&
            remapSlot.buffer->valid()) {
          visibilityMeshletDispatchGpuData_.clear();
          visibilityMeshletDispatchGpuData_.reserve(dispatchCount);
          for (size_t i = 0; i < dispatchCount; ++i) {
            const MeshDispatchItem &dispatch = meshletDispatchItems_[i];
            const MeshletPushConstants &constants = meshletPushConstants_[i];
            visibilityMeshletDispatchGpuData_.push_back(
                VisibilityMeshletDispatchGpuData{
                    .groups = glm::uvec4(dispatch.groupsX, dispatch.groupsY,
                                         dispatch.groupsZ, 0u),
                    .batches = glm::uvec4(constants.batchBase, dispatch.groupsY,
                                          constants.candidateOffset, 0u),
                });
          }
          visibilityMeshletCandidateMap_.assign(
              candidateMapCount, std::numeric_limits<uint32_t>::max());
          for (size_t gpuCandidateIndex = 0u;
               gpuCandidateIndex < gpuVisibilityCandidateIndices.size();
               ++gpuCandidateIndex) {
            const uint32_t sourceCandidateIndex =
                gpuVisibilityCandidateIndices[gpuCandidateIndex];
            const VisibilityCandidate &candidate =
                visibilityCandidates[sourceCandidateIndex];
            const size_t mapIndex =
                static_cast<size_t>(candidate.renderableIndex) *
                    candidateMapStride +
                candidate.submeshIndex;
            if (mapIndex < visibilityMeshletCandidateMap_.size()) {
              visibilityMeshletCandidateMap_[mapIndex] =
                  static_cast<uint32_t>(gpuCandidateIndex);
            }
          }
          const std::array updates{
              BufferUpdate{
                  .buffer = dispatchSlot.buffer->handle(),
                  .data = std::as_bytes(
                      std::span<const VisibilityMeshletDispatchGpuData>(
                          visibilityMeshletDispatchGpuData_.data(),
                          visibilityMeshletDispatchGpuData_.size()))},
              BufferUpdate{.buffer = dispatchSlot.buffer->handle(),
                           .data = std::as_bytes(std::span<const uint32_t>(
                               visibilityMeshletCandidateMap_.data(),
                               visibilityMeshletCandidateMap_.size())),
                           .offset = candidateMapOffset},
          };
          auto updateResult = gpu_.updateBuffers(updates);
          if (updateResult.hasError()) {
            return updateResult;
          }
          const BufferHandle commandBuffer = commandSlot.buffer->handle();
          const BufferHandle dispatchBuffer = dispatchSlot.buffer->handle();
          const BufferHandle candidateBuffer = candidateSlot.buffer->handle();
          const BufferHandle passBuffer = passSlot.buffer->handle();
          const BufferHandle remapBuffer = remapSlot.buffer->handle();
          const uint64_t commandAddress =
              gpu_.getBufferDeviceAddress(commandBuffer);
          const uint64_t dispatchAddress =
              gpu_.getBufferDeviceAddress(dispatchBuffer);
          const uint64_t candidateMapAddress =
              dispatchAddress + candidateMapOffset;
          const uint64_t remapAddress =
              gpu_.getBufferDeviceAddress(remapBuffer);
          const uint64_t candidateAddress =
              gpu_.getBufferDeviceAddress(candidateBuffer);
          const uint64_t passAddress = gpu_.getBufferDeviceAddress(passBuffer);
          if (commandAddress != 0u && dispatchAddress != 0u &&
              meshletBatchBufferAddress != 0u && remapAddress != 0u &&
              candidateAddress != 0u && passAddress != 0u) {
            visibilityIndirectMeshDispatchPushConstants_.clear();
            uint32_t indirectFlags = kVisibilityGpuFlagFrustumCulling;
            if (visibilitySettings.visibleOnUncertain) {
              indirectFlags |= kVisibilityGpuFlagVisibleOnUncertain;
            }
            visibilityIndirectMeshDispatchPushConstants_.push_back(
                VisibilityIndirectMeshDispatchPushConstants{
                    .commandBufferAddress = commandAddress,
                    .dispatchBufferAddress = dispatchAddress,
                    .meshletBatchBufferAddress = meshletBatchBufferAddress,
                    .remapBufferAddress = remapAddress,
                    .candidateMapBufferAddress = candidateMapAddress,
                    .candidateBufferAddress = candidateAddress,
                    .passBufferAddress = passAddress,
                    .dispatchCount = static_cast<uint32_t>(dispatchCount),
                    .candidateCount = static_cast<uint32_t>(
                        gpuVisibilityCandidateIndices.size()),
                    .candidateMapStride = candidateMapStride,
                    .flags = indirectFlags,
                    .sourceFrameIndex = static_cast<uint32_t>(frame.frameIndex),
                });
            visibilityMeshletGpuDependencyBuffers_.clear();
            visibilityMeshletGpuDependencyBuffers_.reserve(6u);
            visibilityMeshletGpuDependencyBuffers_.push_back(commandBuffer);
            visibilityMeshletGpuDependencyBuffers_.push_back(dispatchBuffer);
            visibilityMeshletGpuDependencyBuffers_.push_back(
                meshletBatchBufferHandle);
            visibilityMeshletGpuDependencyBuffers_.push_back(remapBuffer);
            visibilityMeshletGpuDependencyBuffers_.push_back(candidateBuffer);
            visibilityMeshletGpuDependencyBuffers_.push_back(passBuffer);
            visibilityMeshletGpuDependencyBufferAccessModes_.clear();
            visibilityMeshletGpuDependencyBufferAccessModes_.push_back(
                RenderGraphAccessMode::Read | RenderGraphAccessMode::Write);
            for (size_t i = 1u;
                 i < visibilityMeshletGpuDependencyBuffers_.size(); ++i) {
              visibilityMeshletGpuDependencyBufferAccessModes_.push_back(
                  RenderGraphAccessMode::Read);
            }
            visibilityMeshletGpuDispatches_.clear();
            ComputeDispatchItem dispatch{};
            dispatch.pipeline =
                visibilityIndirectMeshDispatchComputePipeline_.get();
            dispatch.dispatch = {
                .x = (static_cast<uint32_t>(dispatchCount) + 63u) / 64u,
                .y = 1u,
                .z = 1u,
            };
            dispatch.pushConstants = std::as_bytes(
                std::span<const VisibilityIndirectMeshDispatchPushConstants>(
                    visibilityIndirectMeshDispatchPushConstants_.data(),
                    visibilityIndirectMeshDispatchPushConstants_.size()));
            dispatch.dependencyBuffers = std::span<const BufferHandle>(
                visibilityMeshletGpuDependencyBuffers_.data(),
                visibilityMeshletGpuDependencyBuffers_.size());
            dispatch.debugLabel = "Opaque Visibility Indirect Mesh Dispatch";
            dispatch.debugColor = kComputeDispatchColor;
            visibilityMeshletGpuDispatches_.push_back(dispatch);
            for (size_t i = 0; i < meshletDispatchItems_.size(); ++i) {
              MeshDispatchItem &meshDispatch = meshletDispatchItems_[i];
              meshDispatch.command = MeshDispatchCommandType::Indirect;
              meshDispatch.indirectBuffer = commandBuffer;
              meshDispatch.indirectBufferOffset =
                  static_cast<uint64_t>(i) * kMeshDispatchCommandBytes;
              meshDispatch.indirectDispatchCount = 1u;
            }
            meshletIndirectDispatchUsed = true;
          }
        }
      }
    }
    uint64_t submittedMeshletTaskGroups = 0u;
    for (const MeshDispatchItem &dispatch : meshletDispatchItems_) {
      submittedMeshletTaskGroups += static_cast<uint64_t>(dispatch.groupsX) *
                                    dispatch.groupsY * dispatch.groupsZ;
    }
    if (!meshletHybridRoutingActive) {
      finalPassDrawItems = {};
    }
    frame.metrics.opaque.meshletModeActive =
        meshletDispatchItems_.empty() ? 0u : 1u;
    frame.metrics.opaque.meshletDispatches =
        saturateToU32(meshletDispatchItems_.size());
    frame.metrics.opaque.meshletTaskGroups =
        saturateToU32(submittedMeshletTaskGroups);
    frame.metrics.opaque.meshletCandidateCount =
        saturateToU32(mainCandidateCount);
    frame.metrics.visibility.meshletCandidates =
        saturateToU32(mainCandidateCount);
    frame.metrics.visibility.indirectMeshDispatchCount =
        meshletIndirectDispatchUsed
            ? saturateToU32(meshletDispatchItems_.size())
            : 0u;
    NURI_PROFILER_ZONE_END();
  }
  OpaqueFrameMetrics &opaqueMetrics = frame.metrics.opaque;
  opaqueMetrics.classicMainDraws = saturateToU32(finalPassDrawItems.size());
  opaqueMetrics.classicAlphaMaskedMainDraws = saturateToU32(
      std::ranges::count_if(finalPassDrawItems, [](const DrawItem &draw) {
        return draw.alphaMasked;
      }));
  if (meshletActive) {
    uint32_t representedItems = 0u;
    uint32_t alphaRepresentedItems = 0u;
    for (size_t i = 0u; i < meshletBatchInfos_.size(); ++i) {
      if (meshletHybridRoutingActive && meshletMainSourceMask[i] == 0u) {
        continue;
      }
      ++representedItems;
      alphaRepresentedItems += meshletBatchInfos_[i].alphaMasked ? 1u : 0u;
    }
    opaqueMetrics.meshletMainRepresentedItems = representedItems;
    opaqueMetrics.meshletAlphaMaskedMainItems = alphaRepresentedItems;
    opaqueMetrics.meshletMainDispatches =
        saturateToU32(meshletDispatchItems_.size());
    const auto sameMeshletPipeline = [](MeshletPipelineHandle lhs,
                                        MeshletPipelineHandle rhs) noexcept {
      return lhs.index == rhs.index && lhs.generation == rhs.generation;
    };
    uint32_t alphaDispatches = 0u;
    for (const MeshDispatchItem &dispatch : meshletDispatchItems_) {
      bool alphaPipeline = false;
      for (const bool compacted : {false, true}) {
        for (const bool doubleSided : {false, true}) {
          alphaPipeline =
              alphaPipeline ||
              sameMeshletPipeline(dispatch.pipeline,
                                  selectMeshletScenePipeline(
                                      compacted, coverage, true, doubleSided));
        }
      }
      alphaDispatches += alphaPipeline ? 1u : 0u;
    }
    opaqueMetrics.meshletAlphaMaskedMainDispatches = alphaDispatches;
  }
  pass.desc.dependencyBuffers = std::span<const BufferHandle>(
      mainPassDependencyBuffers_.data(), mainPassDependencyBuffers_.size());
  pass.desc.dependencyBufferAccessModes =
      std::span<const RenderGraphAccessMode>(
          mainPassDependencyBufferAccessModes_.data(),
          mainPassDependencyBufferAccessModes_.size());
  pass.desc.dependencyTextures = std::span<const TextureHandle>(
      mainPassDependencyTextures_.data(), mainPassDependencyTextures_.size());
  pass.desc.dependencyTextureAccessModes =
      std::span<const RenderGraphAccessMode>(
          mainPassDependencyTextureAccessModes_.data(),
          mainPassDependencyTextureAccessModes_.size());
  uint32_t msaaAlphaMaskedDrawCount = 0u;
  if (msaaSelected && !finalPassDrawItems.empty()) {
    msaaPassDrawItems_.assign(finalPassDrawItems.begin(),
                              finalPassDrawItems.end());
    for (DrawItem &draw : msaaPassDrawItems_) {
      if (draw.alphaMasked) {
        ++msaaAlphaMaskedDrawCount;
      }
      const RenderPipelineHandle msaaPipeline =
          selectMsaaScenePipeline(draw.pipeline, draw.alphaMasked, coverage);
      if (nuri::isValid(msaaPipeline)) {
        draw.pipeline = msaaPipeline;
      }
    }
    finalPassDrawItems = std::span<const DrawItem>(msaaPassDrawItems_.data(),
                                                   msaaPassDrawItems_.size());
  }
  pass.desc.draws = finalPassDrawItems;
  pass.desc.meshDispatches = std::span<const MeshDispatchItem>(
      meshletDispatchItems_.data(), meshletDispatchItems_.size());
  for (const DrawItem &draw : finalPassDrawItems) {
    const bool equalReadOnly = draw.useDepthState &&
                               draw.depthState.compareOp == CompareOp::Equal &&
                               !draw.depthState.isDepthWriteEnabled;
    opaqueMetrics.mainEqualReadOnlyDraws += equalReadOnly ? 1u : 0u;
    const bool lessWrite =
        draw.useDepthState &&
        (draw.depthState.compareOp == CompareOp::Less ||
         draw.depthState.compareOp == CompareOp::LessEqual) &&
        draw.depthState.isDepthWriteEnabled;
    opaqueMetrics.mainLessWriteDraws += lessWrite ? 1u : 0u;
  }
  for (const MeshDispatchItem &dispatch : meshletDispatchItems_) {
    const bool equalReadOnly =
        dispatch.useDepthState &&
        dispatch.depthState.compareOp == CompareOp::Equal &&
        !dispatch.depthState.isDepthWriteEnabled;
    opaqueMetrics.mainEqualReadOnlyDispatches += equalReadOnly ? 1u : 0u;
    const bool lessWrite =
        dispatch.useDepthState &&
        (dispatch.depthState.compareOp == CompareOp::Less ||
         dispatch.depthState.compareOp == CompareOp::LessEqual) &&
        dispatch.depthState.isDepthWriteEnabled;
    opaqueMetrics.mainLessWriteDispatches += lessWrite ? 1u : 0u;
  }
  pass.desc.drawBuffersPreResolved = true;
  pass.desc.preResolvedDrawBuffers = std::span<const BufferHandle>(
      preResolvedDrawBuffers_.data(), preResolvedDrawBuffers_.size());
  pass.desc.debugLabel = kOpaqueMainPassLabel;
  pass.desc.debugColor = kOpaquePassDebugColor;
  pass.desc.gpuTimingScope = GpuTimingScope::OpaqueMain;
  mainPreDispatches_.clear();
  if (!preDispatchSubmittedBeforeMain) {
    mainPreDispatches_.insert(mainPreDispatches_.end(), preDispatches_.begin(),
                              preDispatches_.end());
  }
  mainPreDispatches_.insert(mainPreDispatches_.end(),
                            visibilityMeshletGpuDispatches_.begin(),
                            visibilityMeshletGpuDispatches_.end());
  pass.desc.preDispatches = std::span<const ComputeDispatchItem>(
      mainPreDispatches_.data(), mainPreDispatches_.size());
  pass.desc.borrowPayload = mainPreDispatches_.empty();
  pass.kind = PreparedPassKind::Main;
  const TextureHandle sceneColorTarget =
      msaaSelected ? frame.sharedResources.msaaSceneColorTexture
                   : frame.sharedResources.sceneColorTexture;
  pass.colorTextureHandle = sceneColorTarget;
  AntiAliasingFrameMetrics &aaMetrics = frame.metrics.antiAliasing;
  aaMetrics.msaaMainColorFormat = kFrameCompositionSceneColorFormat;
  aaMetrics.msaaMainDepthFormat = kFrameCompositionDepthFormat;
  aaMetrics.msaaMainAttachmentSampleCount = coverageSampleCount(coverage);
  if (msaaSelected) {
    aaMetrics.msaaAlphaMaskedDrawCount =
        msaaAlphaMaskedDrawCount + opaqueMetrics.meshletAlphaMaskedMainItems;
    aaMetrics.msaaAlphaToCoverageEnabled =
        aaMetrics.msaaAlphaMaskedDrawCount > 0u &&
        presentationAA.alphaCoverage ==
            AlphaCoveragePolicy::ThresholdedAlphaToCoverage &&
        (opaqueMetrics.classicAlphaMaskedMainDraws == 0u ||
         frame.metrics.antiAliasing.msaaAlphaToCoverageSupported) &&
        (opaqueMetrics.meshletAlphaMaskedMainItems == 0u ||
         opaqueMetrics.meshletAlphaMaskedMainDispatches != 0u);
    aaMetrics.msaaSampleShadingEnabled =
        aaMetrics.msaaAlphaMaskedDrawCount > 0u &&
        presentationAA.sampleShadingEnabled;
  }
  if (meshletPreTaskCompactionUsed) {
    PreparedGraphPass &compactionPass = makeDispatchPass(
        std::span<const ComputeDispatchItem>(
            meshletCompactionDispatches_.data(), 1u),
        meshletCompactionDependencyBuffers_,
        meshletCompactionDependencyTextures_,
        "Opaque Meshlet Pre-Task Compaction", GpuTimingScope::OpaqueMain);
    compactionPass.phase = PreparedPassPhase::PreLighting;
    PreparedGraphPass &finalizePass = makeDispatchPass(
        std::span<const ComputeDispatchItem>(
            meshletCompactionDispatches_.data() + 1u, 1u),
        meshletCompactionFinalizeDependencyBuffers_, {},
        "Opaque Meshlet Compaction Finalize", GpuTimingScope::OpaqueMain);
    finalizePass.phase = PreparedPassPhase::PreLighting;
    makeDispatchPass(
        std::span<const ComputeDispatchItem>(
            meshletCompactionDispatches_.data() + 2u, 1u),
        std::span<const BufferHandle>(
            meshletCompactionFinalizeDependencyBuffers_.data() + 2u, 1u),
        {}, "Opaque Meshlet Compaction Metrics Finalize",
        GpuTimingScope::OpaqueMain);
    std::rotate(out.begin() + static_cast<std::ptrdiff_t>(mainPassIndex),
                out.begin() + static_cast<std::ptrdiff_t>(mainPassIndex + 1u),
                out.end() - 1);
  }
  if (meshletActive && reactiveMaskRequired &&
      nuri::isValid(frame.sharedResources.reactiveMaskTexture) &&
      nuri::isValid(meshletReactiveMaskPipelines_[0])) {
    const bool motionUncertainReactiveMode =
        hasTaaVelocityInstances &&
        velocityInstanceFlagsMode != VelocityInstanceFlagsMode::AllValid;
    const bool jitteredAlphaReactiveMode = frame.camera.jitterEnabled;
    uint32_t alphaMaskedCoverageDraws = 0u;
    uint32_t reactiveAlphaMaskedDraws = 0u;
    uint32_t motionUncertainDraws = 0u;
    uint32_t potentialReactiveDraws = 0u;
    for (size_t i = 0; i < drawItems_.size(); ++i) {
      const bool alphaMasked = drawAlphaMasked_[i] != 0u;
      alphaMaskedCoverageDraws += alphaMasked ? 1u : 0u;
      const bool motionUncertain = motionUncertainReactiveMode;
      const bool alphaReactive = alphaMasked && jitteredAlphaReactiveMode;
      if (!motionUncertain && !alphaReactive) {
        continue;
      }
      motionUncertainDraws += motionUncertain ? 1u : 0u;
      reactiveAlphaMaskedDraws += alphaMasked ? 1u : 0u;
      ++potentialReactiveDraws;
    }
    AntiAliasingFrameMetrics &aaMetrics = frame.metrics.antiAliasing;
    aaMetrics.reactiveAlphaMaskedDrawCount = reactiveAlphaMaskedDraws;
    aaMetrics.reactiveMotionUncertainDrawCount = motionUncertainDraws;
    aaMetrics.reactiveSkippedTessellatedDrawCount = 0u;
    aaMetrics.reactiveMaskPassBandwidthEstimateBytes =
        aaMetrics.reactiveMaskTextureBytes;
    const float drawDenominator =
        static_cast<float>(std::max<size_t>(drawItems_.size(), 1u));
    aaMetrics.taaAlphaMaskedCoverageEstimate =
        static_cast<float>(alphaMaskedCoverageDraws) / drawDenominator;
    aaMetrics.taaReactiveCoverageEstimate =
        static_cast<float>(potentialReactiveDraws) / drawDenominator;
    if (potentialReactiveDraws > 0u) {
      buildMeshletDispatches(
          meshletReactiveMaskDispatchItems_, meshletReactiveMaskPushConstants_,
          meshletReactiveMaskDispatchDependencyBuffers_,
          meshletReactiveMaskPipelines_[0], meshletReactiveMaskPipelines_[1],
          {}, {}, CompareOp::LessEqual, false, "OpaqueMeshletReactiveMask",
          0xff33cc88, MeshletOcclusionSource::Disabled, 0u, 0u,
          velocityFrameDataAddress, false, false,
          MeshletSourceFilter{.alphaMaskedOnly = !motionUncertainReactiveMode});
      if (!meshletReactiveMaskDispatchItems_.empty()) {
        populateCoverageDependencyBuffers(
            reactivePassDependencyBuffers_,
            reactivePassDependencyBufferAccessModes_);
        for (const BufferHandle handle : {instanceLodBoundsBuffer_->handle(),
                                          velocityInstanceFlagsBufferHandle,
                                          velocityFrameDataBufferHandle}) {
          appendUniqueDependency(reactivePassDependencyBuffers_,
                                 reactivePassDependencyBufferAccessModes_,
                                 handle, RenderGraphAccessMode::Read);
        }
        PreparedGraphPass &reactivePass = makePreparedPass(
            {}, meshletReactiveMaskDispatchItems_,
            reactivePassDependencyBuffers_,
            reactivePassDependencyBufferAccessModes_, passDependencyTextures_,
            "Opaque Meshlet Reactive Mask Pass", 0xff33cc88);
        reactivePass.desc.color = AttachmentColor{
            .loadOp = LoadOp::Clear,
            .storeOp = StoreOp::Store,
            .clearColor = kFrameCompositionReactiveMaskClearValue};
        reactivePass.colorTextureHandle =
            frame.sharedResources.reactiveMaskTexture;
        reactivePass.desc.depth = {.loadOp = LoadOp::Load,
                                   .storeOp = StoreOp::Store,
                                   .clearDepth = kClearDepthOne,
                                   .clearStencil = 0};
        reactivePass.depthTextureHandle = sceneDepthTexture;
        reactivePass.desc.dependencyTextureAccessModes =
            std::span<const RenderGraphAccessMode>(
                mainPassDependencyTextureAccessModes_.data(),
                passDependencyTextures_.size());
        reactivePass.desc.gpuTimingScope = GpuTimingScope::ReactiveMask;
        reactivePass.desc.borrowPayload = true;
        reactivePass.phase = presentationAA.gtaoTemporal
                                 ? PreparedPassPhase::PreLighting
                                 : PreparedPassPhase::MainLighting;
        reactivePass.kind = PreparedPassKind::ReactiveMask;
        aaMetrics.reactiveMaskDrawCount =
            saturateToU32(meshletReactiveMaskDispatchItems_.size());
        aaMetrics.reactiveMaskPassCount = 1u;
      }
    }
  }
  if (!meshletActive && reactiveMaskRequired &&
      nuri::isValid(frame.sharedResources.reactiveMaskTexture) &&
      nuri::isValid(meshReactiveMaskPipelines_[0])) {
    reactiveMaskDrawItems_.clear();
    reactiveMaskDrawItems_.reserve(shadedBaseDrawItems.size());
    const bool motionUncertainReactiveMode =
        hasTaaVelocityInstances &&
        velocityInstanceFlagsMode != VelocityInstanceFlagsMode::AllValid;
    const bool jitteredAlphaReactiveMode = frame.camera.jitterEnabled;
    uint32_t alphaMaskedCoverageDraws = 0u;
    uint32_t reactiveAlphaMaskedDraws = 0u;
    uint32_t motionUncertainDraws = 0u;
    uint32_t skippedTessellatedReactiveDraws = 0u;
    for (size_t i = 0; i < shadedBaseDrawItems.size(); ++i) {
      const bool alphaMasked = baseAlphaMasked[i] != 0u;
      alphaMaskedCoverageDraws += alphaMasked ? 1u : 0u;
      const bool motionUncertain = motionUncertainReactiveMode;
      const bool alphaReactive = alphaMasked && jitteredAlphaReactiveMode;
      if (!motionUncertain && !alphaReactive) {
        continue;
      }
      const DrawItem &sourceItem = shadedBaseDrawItems[i];
      if (isTessPipeline(sourceItem.pipeline)) {
        ++skippedTessellatedReactiveDraws;
        continue;
      }
      motionUncertainDraws += motionUncertain ? 1u : 0u;
      reactiveAlphaMaskedDraws += alphaMasked ? 1u : 0u;
      const RenderPipelineHandle reactivePipeline =
          selectReactiveMaskPipeline(sourceItem.pipeline);
      if (!nuri::isValid(reactivePipeline)) {
        continue;
      }
      DrawItem reactiveItem = sourceItem;
      reactiveItem.pipeline = reactivePipeline;
      reactiveItem.useDepthState = true;
      reactiveItem.depthState = {.compareOp = CompareOp::LessEqual,
                                 .isDepthWriteEnabled = false};
      reactiveItem.debugLabel = "OpaqueReactiveMask";
      reactiveItem.debugColor = 0xff33cc88;
      reactiveMaskDrawItems_.push_back(reactiveItem);
    }
    AntiAliasingFrameMetrics &aaMetrics = frame.metrics.antiAliasing;
    aaMetrics.reactiveMaskDrawCount =
        saturateToU32(reactiveMaskDrawItems_.size());
    aaMetrics.reactiveAlphaMaskedDrawCount = reactiveAlphaMaskedDraws;
    aaMetrics.reactiveMotionUncertainDrawCount = motionUncertainDraws;
    aaMetrics.reactiveSkippedTessellatedDrawCount =
        skippedTessellatedReactiveDraws;
    aaMetrics.reactiveMaskPassBandwidthEstimateBytes =
        aaMetrics.reactiveMaskTextureBytes;
    const float drawDenominator =
        static_cast<float>(std::max<size_t>(shadedBaseDrawItems.size(), 1u));
    aaMetrics.taaAlphaMaskedCoverageEstimate =
        static_cast<float>(alphaMaskedCoverageDraws) / drawDenominator;
    aaMetrics.taaReactiveCoverageEstimate =
        static_cast<float>(reactiveMaskDrawItems_.size()) / drawDenominator;
    if (!reactiveMaskDrawItems_.empty()) {
      populateCoverageDependencyBuffers(
          reactivePassDependencyBuffers_,
          reactivePassDependencyBufferAccessModes_);
      for (const BufferHandle handle :
           {velocityInstanceFlagsBufferHandle, velocityFrameDataBufferHandle}) {
        appendUniqueDependency(reactivePassDependencyBuffers_,
                               reactivePassDependencyBufferAccessModes_, handle,
                               RenderGraphAccessMode::Read);
      }
      PreparedGraphPass &reactivePass = makePreparedPass(
          reactiveMaskDrawItems_, {}, reactivePassDependencyBuffers_,
          reactivePassDependencyBufferAccessModes_, passDependencyTextures_,
          "Opaque Reactive Mask Pass", 0xff33cc88);
      reactivePass.desc.color = AttachmentColor{
          .loadOp = LoadOp::Clear,
          .storeOp = StoreOp::Store,
          .clearColor = kFrameCompositionReactiveMaskClearValue};
      reactivePass.colorTextureHandle =
          frame.sharedResources.reactiveMaskTexture;
      reactivePass.desc.depth = {.loadOp = LoadOp::Load,
                                 .storeOp = StoreOp::Store,
                                 .clearDepth = kClearDepthOne,
                                 .clearStencil = 0};
      reactivePass.depthTextureHandle = sceneDepthTexture;
      reactivePass.desc.dependencyTextureAccessModes =
          std::span<const RenderGraphAccessMode>(
              mainPassDependencyTextureAccessModes_.data(),
              passDependencyTextures_.size());
      reactivePass.desc.gpuTimingScope = GpuTimingScope::ReactiveMask;
      reactivePass.desc.borrowPayload = true;
      reactivePass.phase = presentationAA.gtaoTemporal
                               ? PreparedPassPhase::PreLighting
                               : PreparedPassPhase::MainLighting;
      reactivePass.kind = PreparedPassKind::ReactiveMask;
      aaMetrics.reactiveMaskPassCount = 1u;
    }
  }
  const bool gtaoTemporalVelocityRequested =
      presentationAAPlanForFrame(frame).gtaoTemporal &&
      hasTemporalCameraContinuity(frame.camera) &&
      frame.camera.temporalDataValid &&
      nuri::isValid(frame.sharedResources.previousAmbientOcclusionTexture);
  const bool depthMotionVectorEligible =
      hasTaaVelocityInstances && staticVelocityScene &&
      canReuseStaticPreviousMatrices &&
      velocityInstanceFlagsMode == VelocityInstanceFlagsMode::AllValid &&
      !needsVelocityGeometryUpload &&
      hasTemporalCameraContinuity(frame.camera) &&
      frame.camera.temporalDataValid &&
      nuri::isValid(frame.sharedResources.motionVectorTexture) &&
      nuri::isValid(sceneDepthTexture) &&
      nuri::isValid(depthMotionVectorPipelineHandle_);
  bool depthMotionVectorPassBuilt = false;
  if (depthMotionVectorEligible) {
    const uint32_t depthTexId = gpu_.getTextureBindlessIndex(sceneDepthTexture);
    const uint32_t pointSamplerId = gpu_.getDefaultSamplerBindlessIndex();
    if (depthTexId != kInvalidTextureBindlessIndex &&
        pointSamplerId != kInvalidTextureBindlessIndex) {
      const TextureDimensions motionVectorDimensions =
          gpu_.getTextureDimensions(frame.sharedResources.motionVectorTexture);
      const float inverseWidth =
          1.0f / static_cast<float>(std::max(motionVectorDimensions.width, 1u));
      const float inverseHeight =
          1.0f /
          static_cast<float>(std::max(motionVectorDimensions.height, 1u));
      const glm::vec2 currentJitterUv{
          frame.camera.jitterPixelOffset.x * inverseWidth,
          -frame.camera.jitterPixelOffset.y * inverseHeight,
      };
      depthMotionVectorPushConstants_ = DepthMotionVectorPushConstants{
          .depthTexId = depthTexId,
          .pointSamplerId = pointSamplerId,
          .currentJitterUvXBits = std::bit_cast<uint32_t>(currentJitterUv.x),
          .currentJitterUvYBits = std::bit_cast<uint32_t>(currentJitterUv.y),
          .previousFromCurrentJitteredClip =
              frame.camera.previousUnjitteredViewProj *
              glm::inverse(frame.camera.currentJitteredViewProj),
      };
      depthMotionVectorDrawItem_ = {};
      depthMotionVectorDrawItem_.pipeline = depthMotionVectorPipelineHandle_;
      depthMotionVectorDrawItem_.vertexCount = 3u;
      depthMotionVectorDrawItem_.instanceCount = 1u;
      depthMotionVectorDrawItem_.pushConstants = std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&depthMotionVectorPushConstants_),
          sizeof(depthMotionVectorPushConstants_));
      depthMotionVectorDrawItem_.debugLabel = "OpaqueDepthMotionVector";
      depthMotionVectorDrawItem_.debugColor = 0xff44aaff;
      depthMotionVectorDependencyTextures_[0] = sceneDepthTexture;
      PreparedGraphPass &velocityPass = makePreparedPass(
          std::span<const DrawItem>(&depthMotionVectorDrawItem_, 1u), {}, {},
          {}, depthMotionVectorDependencyTextures_,
          "Opaque Depth Motion Vector Pass", 0xff44aaff);
      velocityPass.desc.preResolvedDrawBuffers = {};
      velocityPass.desc.color = AttachmentColor{
          .loadOp = LoadOp::Clear,
          .storeOp = StoreOp::Store,
          .clearColor = kFrameCompositionMotionVectorClearValue};
      velocityPass.colorTextureHandle =
          frame.sharedResources.motionVectorTexture;
      velocityPass.desc.gpuTimingScope = GpuTimingScope::Velocity;
      velocityPass.desc.borrowPayload = true;
      velocityPass.phase = gtaoTemporalVelocityRequested
                               ? PreparedPassPhase::PreLighting
                               : PreparedPassPhase::MainLighting;
      velocityPass.kind = PreparedPassKind::Velocity;
      AntiAliasingFrameMetrics &aaMetrics = frame.metrics.antiAliasing;
      aaMetrics.motionVectorDepthReprojectionPassCount = 1u;
      aaMetrics.motionVectorDepthReprojectionGenerated = true;
      aaMetrics.opaqueVelocityGenerated = true;
      aaMetrics.velocityPassBandwidthEstimateBytes =
          aaMetrics.motionVectorTextureBytes;
      aaMetrics.velocityEdgeDiscontinuityEstimate =
          aaMetrics.velocityMissingPreviousRatio;
      depthMotionVectorPassBuilt = true;
    }
  }
  if (meshletActive && temporalMotionRequired &&
      nuri::isValid(frame.sharedResources.motionVectorTexture) &&
      nuri::isValid(meshletVelocityPipelines_[0]) && instanceCount > 0 &&
      !depthMotionVectorPassBuilt) {
    buildMeshletDispatches(
        meshletVelocityDispatchItems_, meshletVelocityPushConstants_,
        meshletVelocityDispatchDependencyBuffers_, meshletVelocityPipelines_[0],
        meshletVelocityPipelines_[1], {}, {}, CompareOp::LessEqual, false,
        "OpaqueMeshletVelocity", 0xff44aaff, MeshletOcclusionSource::Disabled,
        0u, 0u, velocityFrameDataAddress, false, false, MeshletSourceFilter{});
    if (!meshletVelocityDispatchItems_.empty()) {
      populateCoverageDependencyBuffers(
          velocityPassDependencyBuffers_,
          velocityPassDependencyBufferAccessModes_);
      for (const BufferHandle handle :
           {instanceLodBoundsBuffer_->handle(),
            previousInstanceMatricesBufferHandle,
            velocityInstanceFlagsBufferHandle, velocityFrameDataBufferHandle,
            velocityGeometryBufferHandle}) {
        appendUniqueDependency(velocityPassDependencyBuffers_,
                               velocityPassDependencyBufferAccessModes_, handle,
                               RenderGraphAccessMode::Read);
      }
      PreparedGraphPass &velocityPass = makePreparedPass(
          {}, meshletVelocityDispatchItems_, velocityPassDependencyBuffers_,
          velocityPassDependencyBufferAccessModes_, passDependencyTextures_,
          "Opaque Meshlet Velocity Pass", 0xff44aaff);
      velocityPass.desc.color = AttachmentColor{
          .loadOp = LoadOp::Clear,
          .storeOp = StoreOp::Store,
          .clearColor = kFrameCompositionMotionVectorClearValue};
      velocityPass.colorTextureHandle =
          frame.sharedResources.motionVectorTexture;
      velocityPass.desc.depth = {.loadOp = LoadOp::Load,
                                 .storeOp = StoreOp::Store,
                                 .clearDepth = kClearDepthOne,
                                 .clearStencil = 0};
      velocityPass.depthTextureHandle = sceneDepthTexture;
      velocityPass.desc.dependencyTextureAccessModes =
          std::span<const RenderGraphAccessMode>(
              mainPassDependencyTextureAccessModes_.data(),
              passDependencyTextures_.size());
      velocityPass.desc.gpuTimingScope = GpuTimingScope::Velocity;
      velocityPass.desc.borrowPayload = true;
      velocityPass.phase = gtaoTemporalVelocityRequested
                               ? PreparedPassPhase::PreLighting
                               : PreparedPassPhase::MainLighting;
      velocityPass.kind = PreparedPassKind::Velocity;
      frame.metrics.antiAliasing.velocityTessellatedSkippedDrawCount = 0u;
      frame.metrics.antiAliasing.velocityPassCount = 1u;
      frame.metrics.antiAliasing.velocityDrawCount =
          saturateToU32(drawItems_.size());
      frame.metrics.antiAliasing.opaqueVelocityGenerated = true;
      frame.metrics.antiAliasing.velocityPassBandwidthEstimateBytes =
          frame.metrics.antiAliasing.motionVectorTextureBytes;
    }
    const uint32_t velocityInstanceCount =
        frame.metrics.antiAliasing.velocityInstanceCount;
    if (velocityInstanceCount > 0u) {
      frame.metrics.antiAliasing.velocityEdgeDiscontinuityEstimate =
          frame.metrics.antiAliasing.velocityMissingPreviousRatio;
    }
  }
  if (!meshletActive && temporalMotionRequired &&
      nuri::isValid(frame.sharedResources.motionVectorTexture) &&
      nuri::isValid(
          meshVelocityPipelines_[surfaceVariantIndex(false, false)]) &&
      instanceCount > 0 && !depthMotionVectorPassBuilt) {
    velocityDrawItems_.clear();
    velocityDrawItems_.reserve(shadedBaseDrawItems.size());
    uint32_t skippedTessellatedDraws = 0u;
    for (const DrawItem &sourceItem : shadedBaseDrawItems) {
      const RenderPipelineHandle velocityPipeline =
          selectVelocityPipeline(sourceItem.pipeline);
      if (!nuri::isValid(velocityPipeline)) {
        if (isTessPipeline(sourceItem.pipeline)) {
          ++skippedTessellatedDraws;
          frame.metrics.antiAliasing.opaqueVelocityGenerated = false;
        }
        continue;
      }
      DrawItem velocityItem = sourceItem;
      velocityItem.pipeline = velocityPipeline;
      velocityItem.useDepthState = true;
      velocityItem.depthState = {.compareOp = CompareOp::LessEqual,
                                 .isDepthWriteEnabled = false};
      velocityItem.debugLabel = "OpaqueVelocity";
      velocityItem.debugColor = 0xff44aaff;
      velocityDrawItems_.push_back(velocityItem);
    }
    frame.metrics.antiAliasing.velocityTessellatedSkippedDrawCount =
        skippedTessellatedDraws;
    if (!velocityDrawItems_.empty()) {
      populateCoverageDependencyBuffers(
          velocityPassDependencyBuffers_,
          velocityPassDependencyBufferAccessModes_);
      for (const BufferHandle handle :
           {previousInstanceMatricesBufferHandle,
            velocityInstanceFlagsBufferHandle, velocityFrameDataBufferHandle,
            velocityGeometryBufferHandle}) {
        appendUniqueDependency(velocityPassDependencyBuffers_,
                               velocityPassDependencyBufferAccessModes_, handle,
                               RenderGraphAccessMode::Read);
      }
      PreparedGraphPass &velocityPass = makePreparedPass(
          velocityDrawItems_, {}, velocityPassDependencyBuffers_,
          velocityPassDependencyBufferAccessModes_, passDependencyTextures_,
          "Opaque Velocity Pass", 0xff44aaff);
      velocityPass.desc.color = AttachmentColor{
          .loadOp = LoadOp::Clear,
          .storeOp = StoreOp::Store,
          .clearColor = kFrameCompositionMotionVectorClearValue};
      velocityPass.colorTextureHandle =
          frame.sharedResources.motionVectorTexture;
      velocityPass.desc.depth = {.loadOp = LoadOp::Load,
                                 .storeOp = StoreOp::Store,
                                 .clearDepth = kClearDepthOne,
                                 .clearStencil = 0};
      velocityPass.depthTextureHandle = sceneDepthTexture;
      velocityPass.desc.dependencyTextureAccessModes =
          std::span<const RenderGraphAccessMode>(
              mainPassDependencyTextureAccessModes_.data(),
              passDependencyTextures_.size());
      velocityPass.desc.gpuTimingScope = GpuTimingScope::Velocity;
      velocityPass.desc.borrowPayload = true;
      velocityPass.phase = gtaoTemporalVelocityRequested
                               ? PreparedPassPhase::PreLighting
                               : PreparedPassPhase::MainLighting;
      velocityPass.kind = PreparedPassKind::Velocity;
      frame.metrics.antiAliasing.velocityPassCount = 1u;
      frame.metrics.antiAliasing.velocityDrawCount =
          saturateToU32(velocityDrawItems_.size());
      frame.metrics.antiAliasing.opaqueVelocityGenerated =
          skippedTessellatedDraws == 0u;
      frame.metrics.antiAliasing.velocityPassBandwidthEstimateBytes =
          frame.metrics.antiAliasing.motionVectorTextureBytes;
    }
    const uint32_t velocityInstanceCount =
        frame.metrics.antiAliasing.velocityInstanceCount;
    if (velocityInstanceCount > 0u) {
      const float tessellatedRatio =
          static_cast<float>(skippedTessellatedDraws) /
          static_cast<float>(std::max<size_t>(shadedBaseDrawItems.size(), 1u));
      frame.metrics.antiAliasing.velocityEdgeDiscontinuityEstimate = std::min(
          1.0f, frame.metrics.antiAliasing.velocityMissingPreviousRatio +
                    tessellatedRatio);
    }
  }
  if (meshletActive && !meshletBatchGpuData_.empty()) {
    const std::span<const std::byte> batchBytes{
        reinterpret_cast<const std::byte *>(meshletBatchGpuData_.data()),
        meshletBatchGpuData_.size() * sizeof(MeshletBatchGpuData)};
    auto updateResult =
        gpu_.updateBuffer(meshletBatchBufferHandle, batchBytes, 0u);
    if (updateResult.hasError()) {
      return updateResult;
    }
  }
  if (pendingShadowInspectRequest_.has_value() &&
      nuri::isValid(shadowInspectTexture_) &&
      nuri::isValid(
          meshShadowInspectPipelines_[surfaceVariantIndex(false, false)])) {
    shadowInspectDrawItems_.clear();
    shadowInspectDrawItems_.reserve(shadedBaseDrawItems.size());
    int32_t framebufferWidth = 0;
    int32_t framebufferHeight = 0;
    gpu_.getFramebufferSize(framebufferWidth, framebufferHeight);
    const uint32_t safeWidth =
        static_cast<uint32_t>(std::max(framebufferWidth, 1));
    const uint32_t safeHeight =
        static_cast<uint32_t>(std::max(framebufferHeight, 1));
    pendingShadowInspectRequest_->x =
        std::min(pendingShadowInspectRequest_->x, safeWidth - 1u);
    pendingShadowInspectRequest_->y =
        std::min(pendingShadowInspectRequest_->y, safeHeight - 1u);
    const RectU32 inspectScissor{.x = pendingShadowInspectRequest_->x,
                                 .y = pendingShadowInspectRequest_->y,
                                 .width = 1u,
                                 .height = 1u};
    for (const DrawItem &sourceItem : shadedBaseDrawItems) {
      DrawItem inspectItem = sourceItem;
      inspectItem.pipeline = selectShadowInspectPipeline(sourceItem.pipeline);
      inspectItem.useDepthState = true;
      inspectItem.depthState = {.compareOp = CompareOp::LessEqual,
                                .isDepthWriteEnabled = false};
      inspectItem.useScissor = true;
      inspectItem.scissor = inspectScissor;
      inspectItem.debugLabel = "OpaqueShadowInspect";
      inspectItem.debugColor = kOpaquePassDebugColor;
      shadowInspectDrawItems_.push_back(inspectItem);
    }
    PreparedGraphPass &inspectPass = makePreparedPass(
        shadowInspectDrawItems_, {}, mainPassDependencyBuffers_,
        mainPassDependencyBufferAccessModes_, mainPassDependencyTextures_,
        "Opaque Shadow Inspect Pass", kOpaquePassDebugColor);
    inspectPass.desc.color = {.loadOp = LoadOp::Clear,
                              .storeOp = StoreOp::Store,
                              .clearColor = {0.0f, 0.0f, 0.0f, -1.0f}};
    inspectPass.colorTextureHandle = shadowInspectTexture_;
    inspectPass.desc.depth = {.loadOp = LoadOp::Load,
                              .storeOp = StoreOp::Store,
                              .clearDepth = kClearDepthOne,
                              .clearStencil = 0};
    inspectPass.depthTextureHandle = sceneDepthTexture;
    inspectPass.desc.dependencyTextureAccessModes =
        std::span<const RenderGraphAccessMode>(
            mainPassDependencyTextureAccessModes_.data(),
            mainPassDependencyTextureAccessModes_.size());
    inspectPass.desc.borrowPayload = true;
    inFlightShadowInspectReadback_ =
        InFlightShadowInspectReadback{.request = *pendingShadowInspectRequest_,
                                      .submissionFrame = frame.frameIndex};
    pendingShadowInspectRequest_.reset();
  }
  frame.metrics.opaque.computeDispatches = saturateToU32(preDispatches_.size());
  stagePreviousTransforms(*frame.scene, frame.frameIndex);
  logOpaqueVisibilityCounters(frame);
  NURI_PROFILER_ZONE_END();
  return Result<bool, std::string>::makeResult(true);
}

bool OpaqueRenderer::requiresDepthPyramid(
    const RenderSettings &settings) const {
  const VisibilityOcclusionMode visibilityOcclusionMode =
      sanitizeVisibilityOcclusionMode(settings.visibility.occlusionMode);
  return settings.opaque.enableDepthPyramid ||
         visibilityOcclusionMode == VisibilityOcclusionMode::PreviousFrameHiZ ||
         visibilityOcclusionMode ==
             VisibilityOcclusionMode::CurrentFrameHiZExperimental ||
         settings.shadow.enabled;
}

bool OpaqueRenderer::shouldBuildTransmissionVisibilityDepth(
    const RenderFrameContext &frame, const RenderSettings &settings) const {
  if (!settings.transmission.enabled || !frame.camera.jitterEnabled ||
      sanitizeAntiAliasingMode(settings.antiAliasing.mode) !=
          AntiAliasingMode::TAA) {
    return false;
  }
  if (!frame.sharedResources.forwardSceneGpuData.has_value()) {
    return false;
  }
  const ForwardSceneGpuData &sceneGpu =
      *frame.sharedResources.forwardSceneGpuData;
  return sceneGpu.frameDataAddress != 0u &&
         sceneGpu.postTaaFrameDataAddress != 0u &&
         sceneGpu.frameDataAddress != sceneGpu.postTaaFrameDataAddress;
}

Result<bool, std::string>
OpaqueRenderer::prepareOpaqueGraphPasses(RenderFrameContext &frame) {
  preparedGraphPasses_.clear();
  Result<bool, std::string> result =
      Result<bool, std::string>::makeResult(true);
  NURI_PROFILER_ZONE("OpaqueRenderer.graph_prepare_passes",
                     NURI_PROFILER_COLOR_CMD_DRAW);
  result = buildOpaquePasses(frame, preparedGraphPasses_);
  NURI_PROFILER_ZONE_END();
  return result;
}

Result<bool, std::string>
OpaqueRenderer::prepareSceneCache(SceneDrawDatabase &database,
                                  const RenderScene &scene,
                                  const ResourceManager &resources) {
  auto result = database.update(scene, resources);
  if (result.hasError()) {
    return result;
  }
  rebuildSceneCache(database, scene, true);
  cachedMaterialVersion_ = resources.materialVersion();
  cachedModelMaterialBindingVersion_ = resources.modelMaterialBindingVersion();
  return Result<bool, std::string>::makeResult(true);
}
bool OpaqueRenderer::hasPreparedOpaquePrepassPasses() const noexcept {
  return hasPreparedPasses(PreparedPassPhase::PreLighting);
}

bool OpaqueRenderer::hasPreparedOpaqueMainLightingPasses() const noexcept {
  return hasPreparedPasses(PreparedPassPhase::MainLighting);
}

bool OpaqueRenderer::hasPreparedOpaquePickPasses() const noexcept {
  return hasPreparedPasses(PreparedPassPhase::Pick);
}

bool OpaqueRenderer::shouldPublishSceneDepthGraphTexture(
    const RenderFrameContext &frame) const {
  const RenderSettings &settings = renderSettingsOrDefault(frame);
  if (isRenderCaptureRequested(frame, "scene_depth")) {
    return true;
  }
  if (settings.opaque.enableDepthPyramid) {
    return true;
  }
  if (settings.skybox.enabled) {
    return true;
  }
  if (sanitizeVisibilityOcclusionMode(settings.visibility.occlusionMode) !=
      VisibilityOcclusionMode::Disabled) {
    return true;
  }
  if (settings.transmission.enabled || settings.transparent.enabled) {
    return true;
  }
  if (settings.visibility.debug.showObjectBounds ||
      settings.visibility.debug.showMeshletBounds) {
    return true;
  }
  if (settings.debug.enabled &&
      (settings.debug.grid || settings.debug.modelBounds ||
       settings.debug.lightIcons)) {
    return true;
  }
  return false;
}

bool OpaqueRenderer::hasPreparedPasses(PreparedPassPhase phase) const noexcept {
  return std::ranges::any_of(
      preparedGraphPasses_,
      [phase](const PreparedGraphPass &pass) { return pass.phase == phase; });
}

void OpaqueRenderer::appendPreparedGraphPass(RenderFrameContext &frame,
                                             RenderGraphBuilder &graph,
                                             const PreparedGraphPass &pass,
                                             uint32_t safeWidth,
                                             uint32_t safeHeight) {
  RenderGraphGraphicsPassDesc passDesc = pass.desc;
  struct TextureImport {
    TextureHandle handle;
    std::string_view name;
    RenderGraphTextureId *output;
  };
  for (const TextureImport &spec :
       {TextureImport{pass.colorTextureHandle, "opaque_pass_color_texture",
                      &passDesc.colorTexture},
        TextureImport{pass.colorResolveTextureHandle,
                      "opaque_pass_color_resolve_texture",
                      &passDesc.colorResolveTexture},
        TextureImport{pass.depthTextureHandle, "opaque_pass_depth_texture",
                      &passDesc.depthTexture},
        TextureImport{pass.depthResolveTextureHandle,
                      "opaque_pass_depth_resolve_texture",
                      &passDesc.depthResolveTexture}}) {
    if (!nuri::isValid(spec.handle)) {
      continue;
    }
    *spec.output = graph.importTexture(spec.handle, spec.name).value();
  }
  const RenderGraphPassId passId = graph.addGraphicsPass(passDesc).value();
  if (pass.kind == PreparedPassKind::Pick) {
    frame.sharedResources.opaquePickGraphTexture = passDesc.colorTexture;
    const Format pickDepthFormat =
        nuri::isValid(pass.depthTextureHandle)
            ? gpu_.getTextureFormat(pass.depthTextureHandle)
            : Format::D32_FLOAT;
    const TextureDesc pickDepthTransientDesc{
        .type = TextureType::Texture2D,
        .format = pickDepthFormat,
        .dimensions = {safeWidth, safeHeight, 1},
        .usage = TextureUsage::Attachment,
        .storage = Storage::Device,
        .numLayers = 1,
        .numSamples = 1,
        .numMipLevels = 1,
        .data = {},
        .dataNumMipLevels = 1,
        .generateMipmaps = false,
    };
    frame.sharedResources.opaquePickDepthGraphTexture =
        graph
            .createTransientTexture(pickDepthTransientDesc,
                                    "opaque_pick_transient_depth")
            .value();
    (void)graph
        .bindPassDepthTexture(passId,
                              frame.sharedResources.opaquePickDepthGraphTexture)
        .value();
  }
  const bool publishDepthGraphTexture =
      shouldPublishSceneDepthGraphTexture(frame) ||
      (pass.kind == PreparedPassKind::Main &&
       nuri::isValid(pass.depthResolveTextureHandle));
  if ((pass.publishesDepth || pass.kind == PreparedPassKind::Main) &&
      nuri::isValid(pass.depthTextureHandle) && publishDepthGraphTexture) {
    if (isSameTextureHandle(pass.depthTextureHandle,
                            frame.sharedResources.msaaSceneDepthTexture)) {
      frame.sharedResources.msaaSceneDepthGraphTexture = passDesc.depthTexture;
      frame.metrics.antiAliasing.msaaDepthGraphPublished = true;
    } else {
      frame.sharedResources.sceneDepthGraphTexture = passDesc.depthTexture;
      publishRequestedCapture(
          frame, gpu_, "scene_depth", frame.sharedResources.sceneDepthTexture,
          RenderCaptureValueKind::Depth,
          RenderCaptureLifetimeClass::FrameSharedRingTexture, "linear_depth",
          "depth", pass.desc.debugLabel);
    }
  }
  if (pass.kind == PreparedPassKind::TransmissionDepth &&
      nuri::isValid(pass.depthTextureHandle)) {
    frame.sharedResources.transmissionVisibilityDepthGraphTexture =
        passDesc.depthTexture;
    publishRequestedCapture(
        frame, gpu_, "transmission_visibility_depth", pass.depthTextureHandle,
        RenderCaptureValueKind::Depth,
        RenderCaptureLifetimeClass::FeaturePersistentTexture, "linear_depth",
        "depth", pass.desc.debugLabel);
  }
  if (pass.kind == PreparedPassKind::Main &&
      nuri::isValid(pass.depthResolveTextureHandle) &&
      isSameTextureHandle(pass.depthResolveTextureHandle,
                          frame.sharedResources.sceneDepthTexture)) {
    frame.sharedResources.sceneDepthGraphTexture = passDesc.depthResolveTexture;
    frame.metrics.antiAliasing.msaaDepthResolveTargetBound = true;
    publishRequestedCapture(frame, gpu_, "scene_depth",
                            frame.sharedResources.sceneDepthTexture,
                            RenderCaptureValueKind::Depth,
                            RenderCaptureLifetimeClass::FrameSharedRingTexture,
                            "linear_depth", "depth", pass.desc.debugLabel);
  }
  if ((pass.publishesDepth || pass.kind == PreparedPassKind::Main) &&
      nuri::isValid(frame.sharedResources.sceneDepthGraphTexture)) {
    frame.sharedResources.historyWriteRequirements |=
        FrameTextureRequirementFlags::SceneDepth;
  }
  if (pass.kind == PreparedPassKind::Main &&
      nuri::isValid(pass.colorTextureHandle)) {
    if (isSameTextureHandle(pass.colorTextureHandle,
                            frame.sharedResources.msaaSceneColorTexture)) {
      frame.sharedResources.msaaSceneColorGraphTexture = passDesc.colorTexture;
      frame.metrics.antiAliasing.msaaColorGraphPublished = true;
      if (nuri::isValid(pass.colorResolveTextureHandle) &&
          isSameTextureHandle(pass.colorResolveTextureHandle,
                              frame.sharedResources.sceneColorTexture)) {
        frame.sharedResources.sceneColorGraphTexture =
            passDesc.colorResolveTexture;
        frame.metrics.antiAliasing.msaaColorResolveTargetBound = true;
        publishRequestedCapture(
            frame, gpu_, "scene_color_hdr",
            frame.sharedResources.sceneColorTexture,
            RenderCaptureValueKind::LinearHdrColor,
            RenderCaptureLifetimeClass::FrameSharedRingTexture, "linear_hdr",
            "hdr_color", pass.desc.debugLabel);
        if (renderSettingsOrDefault(frame).shadow.debug.visualizeShadowFactor) {
          publishRequestedCapture(
              frame, gpu_, "shadow_factor",
              frame.sharedResources.sceneColorTexture,
              RenderCaptureValueKind::Scalar,
              RenderCaptureLifetimeClass::FrameSharedRingTexture, "linear",
              "scalar", pass.desc.debugLabel);
        }
      }
    } else {
      frame.sharedResources.sceneColorGraphTexture = passDesc.colorTexture;
      publishRequestedCapture(
          frame, gpu_, "scene_color_hdr",
          frame.sharedResources.sceneColorTexture,
          RenderCaptureValueKind::LinearHdrColor,
          RenderCaptureLifetimeClass::FrameSharedRingTexture, "linear_hdr",
          "hdr_color", pass.desc.debugLabel);
      if (renderSettingsOrDefault(frame).shadow.debug.visualizeShadowFactor) {
        publishRequestedCapture(
            frame, gpu_, "shadow_factor",
            frame.sharedResources.sceneColorTexture,
            RenderCaptureValueKind::Scalar,
            RenderCaptureLifetimeClass::FrameSharedRingTexture, "linear",
            "scalar", pass.desc.debugLabel);
      }
    }
  }
  if (pass.kind == PreparedPassKind::Velocity &&
      nuri::isValid(pass.colorTextureHandle)) {
    frame.sharedResources.motionVectorGraphTexture = passDesc.colorTexture;
    frame.sharedResources.historyWriteRequirements |=
        FrameTextureRequirementFlags::MotionVectors;
    frame.metrics.antiAliasing.motionVectorGraphPublished = true;
    frame.metrics.antiAliasing.motionVectorClearPassCount = 0u;
    frame.metrics.antiAliasing.motionVectorClearBytes = 0u;
    publishRequestedCapture(frame, gpu_, "motion_vectors",
                            pass.colorTextureHandle,
                            RenderCaptureValueKind::Velocity,
                            RenderCaptureLifetimeClass::FrameSharedRingTexture,
                            "uv_velocity", "velocity", pass.desc.debugLabel);
  }
  if (pass.kind == PreparedPassKind::ReactiveMask &&
      nuri::isValid(pass.colorTextureHandle)) {
    frame.sharedResources.reactiveMaskGraphTexture = passDesc.colorTexture;
    frame.metrics.antiAliasing.reactiveMaskGraphPublished = true;
    publishRequestedCapture(frame, gpu_, "reactive_mask",
                            pass.colorTextureHandle,
                            RenderCaptureValueKind::Mask,
                            RenderCaptureLifetimeClass::FrameSharedRingTexture,
                            "linear_scalar", "mask", pass.desc.debugLabel);
  }
  if (pass.kind == PreparedPassKind::Normal &&
      nuri::isValid(pass.colorTextureHandle)) {
    frame.sharedResources.normalGraphTexture = passDesc.colorTexture;
    frame.metrics.ambientOcclusion.normalGraphPublished = true;
    publishRequestedCapture(frame, gpu_, "material_normals",
                            pass.colorTextureHandle,
                            RenderCaptureValueKind::Normal,
                            RenderCaptureLifetimeClass::FrameSharedRingTexture,
                            "world_normal", "normal", pass.desc.debugLabel);
  }
  if (pass.kind == PreparedPassKind::DepthPyramid &&
      pass.depthPyramidLevel < kMaxSceneDepthPyramidLevels &&
      nuri::isValid(passDesc.colorTexture)) {
    frame.sharedResources
        .sceneDepthPyramidGraphTextures[pass.depthPyramidLevel] =
        passDesc.colorTexture;
  }
}
Result<bool, std::string>
OpaqueRenderer::appendPreparedPasses(RenderFrameContext &frame,
                                     RenderGraphBuilder &graph,
                                     PreparedPassPhase phase) {
  int32_t width = 0;
  int32_t height = 0;
  gpu_.getFramebufferSize(width, height);
  if (phase == PreparedPassPhase::PreLighting) {
    struct Registration {
      PersistentBufferId *id;
      BufferHandle *registered;
      BufferHandle handle;
      std::string_view name;
    };
    const std::array<Registration, 2> registrations{
        Registration{&persistentCentersPhaseBuffer_,
                     &registeredCentersPhaseBufferHandle_,
                     instanceCentersPhaseBuffer_
                         ? instanceCentersPhaseBuffer_->handle()
                         : BufferHandle{},
                     "opaque_instance_centers_phase_buffer"},
        Registration{&persistentBaseMatricesBuffer_,
                     &registeredBaseMatricesBufferHandle_,
                     instanceBaseMatricesBuffer_
                         ? instanceBaseMatricesBuffer_->handle()
                         : BufferHandle{},
                     "opaque_instance_base_matrices_buffer"},
    };
    for (const auto &[id, registered, handle, name] : registrations) {
      registerOrUpdatePersistentBuffer(graph, *id, *registered, handle, name);
    }
  }
  NURI_PROFILER_ZONE("OpaqueRenderer.graph_add_passes",
                     NURI_PROFILER_COLOR_CMD_DRAW);
  for (const PreparedGraphPass &pass : preparedGraphPasses_) {
    if (pass.phase != phase) {
      continue;
    }
    appendPreparedGraphPass(frame, graph, pass,
                            static_cast<uint32_t>(std::max(width, 1)),
                            static_cast<uint32_t>(std::max(height, 1)));
  }
  NURI_PROFILER_ZONE_END();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
OpaqueRenderer::appendOpaquePrepassPasses(RenderFrameContext &frame,
                                          RenderGraphBuilder &graph) {
  return appendPreparedPasses(frame, graph, PreparedPassPhase::PreLighting);
}

Result<bool, std::string>
OpaqueRenderer::appendOpaqueMainLightingPasses(RenderFrameContext &frame,
                                               RenderGraphBuilder &graph) {
  return appendPreparedPasses(frame, graph, PreparedPassPhase::MainLighting);
}

Result<bool, std::string>
OpaqueRenderer::appendOpaquePickPasses(RenderFrameContext &frame,
                                       RenderGraphBuilder &graph) {
  return appendPreparedPasses(frame, graph, PreparedPassPhase::Pick);
}
uint32_t OpaqueRenderer::resolveSingleInstanceRequestedLod(
    const RenderSettings &settings, uint32_t forcedLod) const {
  if (!settings.opaque.enableMeshLod) {
    return 0;
  }
  if (settings.opaque.forcedMeshLod >= 0) {
    return forcedLod;
  }
  if (!instanceAutoLodLevels_.empty()) {
    return instanceAutoLodLevels_.front();
  }
  return 0;
}

bool OpaqueRenderer::shouldEnableSingleInstanceTessPipeline(
    bool tessellationRequested, uint32_t requestedLod,
    const glm::vec3 &cameraPosition, float tessFarDistanceSq) const {
  if (!tessellationRequested || requestedLod != 0 ||
      instanceLodCentersInvRadiusSq_.empty()) {
    return false;
  }
  const glm::vec4 centerInvRadiusSq = instanceLodCentersInvRadiusSq_.front();
  const float dx = cameraPosition.x - centerInvRadiusSq.x;
  const float dy = cameraPosition.y - centerInvRadiusSq.y;
  const float dz = cameraPosition.z - centerInvRadiusSq.z;
  const float distanceSq = dx * dx + dy * dy + dz * dz;
  return distanceSq <= tessFarDistanceSq;
}

size_t
OpaqueRenderer::singleInstanceCacheIndex(uint32_t requestedLod,
                                         bool tessPipelineEnabled) const {
  const uint32_t clampedLod =
      std::min(requestedLod, Submesh::kMaxLodCount - 1u);
  return static_cast<size_t>(clampedLod) * 2u + (tessPipelineEnabled ? 1u : 0u);
}

Result<bool, std::string> OpaqueRenderer::ensureSingleInstanceBatchCache(
    uint32_t requestedLod, bool automaticLod, bool tessPipelineEnabled,
    const DrawItem &baseDraw) {
  const size_t cacheIndex =
      singleInstanceCacheIndex(requestedLod, tessPipelineEnabled);
  SingleInstanceBatchCache &cache = singleInstanceBatchCaches_[cacheIndex];
  const bool canReuseSingleInstanceCache =
      cache.valid && cache.requestedLod == requestedLod &&
      cache.automaticLod == automaticLod &&
      cache.tessPipelineEnabled == tessPipelineEnabled &&
      cache.templateRevision == singleInstanceTemplateRevision_ &&
      isSamePipelineHandle(cache.basePipeline, baseDraw.pipeline) &&
      isSamePipelineHandle(cache.doubleSidedBasePipeline,
                           meshScenePipelines_[rasterVariantIndex(
                               CoverageMode::Sample1, false, false, true)]) &&
      isSamePipelineHandle(cache.tessPipeline,
                           meshScenePipelines_[rasterVariantIndex(
                               CoverageMode::Sample1, false, true, false)]) &&
      isSamePipelineHandle(cache.doubleSidedTessPipeline,
                           meshScenePipelines_[rasterVariantIndex(
                               CoverageMode::Sample1, false, true, true)]);
  if (canReuseSingleInstanceCache) {
    return Result<bool, std::string>::makeResult(true);
  }
  ScratchArena scratch;
  ScopedScratch scopedScratch(scratch);
  PmrHashMap<BatchKey, size_t, BatchKeyHash> singleBatchLookup(
      scopedScratch.resource());
  singleBatchLookup.reserve(meshDrawTemplates_.size());
  cache.batches.clear();
  cache.remapCount = 0;
  for (const MeshDrawTemplate &templateEntry : meshDrawTemplates_) {
    const uint32_t materialLod = detail::resolveOpaqueAutomaticLod(
        requestedLod, templateEntry.alphaMasked, automaticLod);
    const auto lodIndex =
        resolveAvailableLod(*templateEntry.submesh, materialLod);
    if (!lodIndex) {
      continue;
    }
    const bool useTessPipeline = tessPipelineEnabled && *lodIndex == 0;
    RenderPipelineHandle selectedPipeline =
        selectMeshPipeline(templateEntry.doubleSided, useTessPipeline);
    const SubmeshLod &lodRange = templateEntry.submesh->lods[*lodIndex];
    const BatchKey key{
        .pipeline = selectedPipeline,
        .indexBuffer = templateEntry.indexBuffer,
        .indexBufferOffset = templateEntry.indexBufferOffset,
        .indexCount = lodRange.indexCount,
        .firstIndex = lodRange.indexOffset,
        .vertexBufferAddress = templateEntry.vertexBufferAddress,
        .vertexDecodeBufferAddress = templateEntry.vertexDecodeBufferAddress,
        .vertexDecodeIndex = templateEntry.vertexDecodeIndex,
        .packedVertexFormat = templateEntry.packedVertexFormat,
        .materialIndex = templateEntry.materialIndex,
    };
    auto it = singleBatchLookup.find(key);
    if (it == singleBatchLookup.end()) {
      SingleInstanceBatchEntry entry{};
      entry.draw = baseDraw;
      entry.draw.pipeline = selectedPipeline;
      entry.draw.indexBuffer = templateEntry.indexBuffer;
      entry.draw.indexBufferOffset = templateEntry.indexBufferOffset;
      entry.draw.indexCount = lodRange.indexCount;
      entry.draw.firstIndex = lodRange.indexOffset;
      entry.draw.vertexOffset = 0;
      entry.draw.alphaMasked = templateEntry.alphaMasked;
      entry.vertexBuffer = templateEntry.vertexBuffer;
      entry.vertexBufferAddress = templateEntry.vertexBufferAddress;
      entry.vertexDecodeBufferAddress = templateEntry.vertexDecodeBufferAddress;
      entry.vertexDecodeIndex = templateEntry.vertexDecodeIndex;
      entry.packedVertexFormat = templateEntry.packedVertexFormat;
      entry.materialIndex = templateEntry.materialIndex;
      entry.alphaMasked = templateEntry.alphaMasked;
      entry.materialNormalRequired = templateEntry.materialNormalRequired;
      cache.batches.push_back(entry);
      const size_t insertedIndex = cache.batches.size() - 1;
      auto [insertedIt, _] = singleBatchLookup.emplace(key, insertedIndex);
      it = insertedIt;
    }
    ++cache.batches[it->second].instanceCount;
    ++cache.remapCount;
  }
  cache.requestedLod = requestedLod;
  cache.automaticLod = automaticLod;
  cache.tessPipelineEnabled = tessPipelineEnabled;
  cache.basePipeline = baseDraw.pipeline;
  cache.doubleSidedBasePipeline = meshScenePipelines_[rasterVariantIndex(
      CoverageMode::Sample1, false, false, true)];
  cache.tessPipeline = meshScenePipelines_[rasterVariantIndex(
      CoverageMode::Sample1, false, true, false)];
  cache.doubleSidedTessPipeline = meshScenePipelines_[rasterVariantIndex(
      CoverageMode::Sample1, false, true, true)];
  cache.templateRevision = singleInstanceTemplateRevision_;
  cache.valid = true;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
OpaqueRenderer::buildIndirectDraws(uint32_t frameSlot, size_t remapCount,
                                   uint64_t drawSignature,
                                   bool drawSignatureValid) {
  const bool canUseIndirectPath =
      !drawItems_.empty() && drawItems_.size() <= kMaxDrawItemsForIndirectPath;
  if (!canUseIndirectPath) {
    invalidateIndirectPackCache();
    indirectDrawItems_.clear();
    indirectCommandUploadBytes_.clear();
    currentIndirectDrawBufferSignature_ = kInvalidDrawSignature;
    return Result<bool, std::string>::makeResult(true);
  }
  Result<bool, std::string> packResult =
      Result<bool, std::string>::makeResult(true);
  NURI_PROFILER_ZONE("OpaqueRenderer.indirect_pack",
                     NURI_PROFILER_COLOR_CMD_DRAW);
  if (!drawSignatureValid) {
    drawSignature = computeIndirectDrawSignature(remapCount);
  }
  const bool canReusePackedCommands =
      indirectPackCache_.valid &&
      indirectPackCache_.drawSignature == drawSignature;
  if (canReusePackedCommands) {
    auto indirectCapacityResult = ensureIndirectCommandRingCapacity(
        std::max(indirectPackCache_.requiredBytes, kIndirectCountHeaderBytes));
    if (indirectCapacityResult.hasError()) {
      return indirectCapacityResult;
    }
  }
  packResult = canReusePackedCommands
                   ? refreshCachedIndirectPack(frameSlot, drawSignature)
                   : rebuildIndirectPack(frameSlot, drawSignature);
  NURI_PROFILER_ZONE_END();
  return packResult;
}

uint64_t OpaqueRenderer::computeIndirectDrawSignature(size_t remapCount) const {
  uint64_t drawSignature = kFnvOffsetBasis64;
  drawSignature = hashCombine64(drawSignature, drawItems_.size());
  drawSignature = hashCombine64(drawSignature, remapCount);
  for (size_t i = 0; i < drawItems_.size(); ++i) {
    const DrawItem &draw = drawItems_[i];
    drawSignature =
        hashCombine64(drawSignature, static_cast<uint64_t>(draw.command));
    drawSignature =
        hashCombine64(drawSignature, foldHandle(draw.pipeline.index,
                                                draw.pipeline.generation));
    drawSignature =
        hashCombine64(drawSignature, foldHandle(draw.indexBuffer.index,
                                                draw.indexBuffer.generation));
    drawSignature = hashCombine64(drawSignature, draw.indexBufferOffset);
    drawSignature =
        hashCombine64(drawSignature, static_cast<uint64_t>(draw.indexFormat));
    drawSignature = hashCombine64(drawSignature, draw.indexCount);
    drawSignature = hashCombine64(drawSignature, draw.instanceCount);
    drawSignature = hashCombine64(drawSignature, draw.firstIndex);
    drawSignature =
        hashCombine64(drawSignature, static_cast<uint64_t>(draw.vertexOffset));
    drawSignature = hashCombine64(drawSignature, draw.firstInstance);
    drawSignature =
        hashCombine64(drawSignature, drawPushConstants_[i].vertexBufferAddress);
    drawSignature =
        hashCombine64(drawSignature, drawPushConstants_[i].materialIndex);
  }
  return drawSignature;
}

Result<bool, std::string>
OpaqueRenderer::rebuildIndirectPack(uint32_t frameSlot,
                                    uint64_t drawSignature) {
  ScratchArena scratch;
  ScopedScratch scopedScratch(scratch);
  struct IndirectGroup {
    DrawItem baseDraw{};
    size_t sourceDrawIndex = 0;
    std::pmr::vector<DrawIndexedIndirectCommand> commands;
    explicit IndirectGroup(std::pmr::memory_resource *mem) : commands(mem) {}
  };
  std::pmr::vector<IndirectGroup> indirectGroups(scopedScratch.resource());
  indirectGroups.reserve(drawItems_.size());
  PmrHashMap<IndirectGroupKey, size_t, IndirectGroupKeyHash> indirectLookup(
      scopedScratch.resource());
  indirectLookup.reserve(drawItems_.size());
  for (size_t i = 0; i < drawItems_.size(); ++i) {
    const DrawItem &draw = drawItems_[i];
    const PushConstants &constants = drawPushConstants_[i];
    const IndirectGroupKey key{
        .pipeline = draw.pipeline,
        .indexBuffer = draw.indexBuffer,
        .indexBufferOffset = draw.indexBufferOffset,
        .indexFormat = draw.indexFormat,
        .vertexBufferAddress = constants.vertexBufferAddress,
        .vertexDecodeBufferAddress = constants.vertexDecodeBufferAddress,
        .vertexDecodeIndex = constants.vertexDecodeIndex,
        .packedVertexFormat = constants.packedVertexFormat,
        .materialIndex = constants.materialIndex,
    };
    auto it = indirectLookup.find(key);
    if (it == indirectLookup.end()) {
      indirectGroups.emplace_back(scopedScratch.resource());
      IndirectGroup &group = indirectGroups.back();
      group.baseDraw = draw;
      group.sourceDrawIndex = i;
      group.commands.reserve(4);
      const size_t groupIndex = indirectGroups.size() - 1;
      auto insertResult = indirectLookup.emplace(key, groupIndex);
      it = insertResult.first;
    }
    DrawIndexedIndirectCommand command{};
    command.indexCount = draw.indexCount;
    command.instanceCount = draw.instanceCount;
    command.firstIndex = draw.firstIndex;
    command.vertexOffset = draw.vertexOffset;
    command.firstInstance = draw.firstInstance;
    indirectGroups[it->second].commands.push_back(command);
  }
  size_t packedRequiredBytes = 0;
  size_t totalIndirectDrawItems = 0;
  for (const IndirectGroup &group : indirectGroups) {
    const size_t commandCount = group.commands.size();
    if (commandCount == 0) {
      continue;
    }
    const size_t chunkCount =
        (commandCount + (kMaxIndirectCommandsPerDraw - 1u)) /
        kMaxIndirectCommandsPerDraw;
    totalIndirectDrawItems += chunkCount;
    packedRequiredBytes += chunkCount * kIndirectCountHeaderBytes +
                           commandCount * sizeof(DrawIndexedIndirectCommand);
  }
  packedRequiredBytes =
      std::max(packedRequiredBytes, kIndirectCountHeaderBytes);
  auto packedCapacityResult =
      ensureIndirectCommandRingCapacity(packedRequiredBytes);
  if (packedCapacityResult.hasError()) {
    return packedCapacityResult;
  }
  indirectDrawItems_.clear();
  indirectAlphaMasked_.clear();
  indirectCommandUploadBytes_.clear();
  indirectSourceDrawIndices_.clear();
  if (!indirectGroups.empty()) {
    indirectCommandUploadBytes_.reserve(packedRequiredBytes);
    indirectDrawItems_.reserve(totalIndirectDrawItems);
    indirectSourceDrawIndices_.reserve(totalIndirectDrawItems);
    const BufferHandle indirectBufferHandle =
        bufferRings_[IndirectCommandRing][frameSlot].buffer->handle();
    uint64_t indirectBufferSignature =
        hashBufferHandleSignature(kFnvOffsetBasis64, indirectBufferHandle);
    for (const IndirectGroup &group : indirectGroups) {
      if (group.commands.empty()) {
        continue;
      }
      size_t commandCursor = 0;
      while (commandCursor < group.commands.size()) {
        const size_t remaining = group.commands.size() - commandCursor;
        const uint32_t drawCount = static_cast<uint32_t>(std::min<size_t>(
            remaining, static_cast<size_t>(kMaxIndirectCommandsPerDraw)));
        const size_t commandByteOffset =
            commandCursor * sizeof(DrawIndexedIndirectCommand);
        const size_t commandByteSize =
            static_cast<size_t>(drawCount) * sizeof(DrawIndexedIndirectCommand);
        const size_t chunkOffset = indirectCommandUploadBytes_.size();
        indirectCommandUploadBytes_.insert(
            indirectCommandUploadBytes_.end(),
            reinterpret_cast<const std::byte *>(&drawCount),
            reinterpret_cast<const std::byte *>(&drawCount) +
                kIndirectCountHeaderBytes);
        indirectCommandUploadBytes_.insert(
            indirectCommandUploadBytes_.end(),
            reinterpret_cast<const std::byte *>(group.commands.data()) +
                commandByteOffset,
            reinterpret_cast<const std::byte *>(group.commands.data()) +
                commandByteOffset + commandByteSize);
        DrawItem indirectDraw = group.baseDraw;
        indirectDraw.alphaMasked =
            drawAlphaMasked_[group.sourceDrawIndex] != 0u;
        indirectDraw.command = DrawCommandType::IndexedIndirect;
        indirectDraw.indirectBuffer = indirectBufferHandle;
        indirectDraw.indirectBufferOffset =
            chunkOffset + kIndirectCountHeaderBytes;
        indirectDraw.indirectDrawCount = drawCount;
        indirectDraw.indirectStride = sizeof(DrawIndexedIndirectCommand);
        indirectDraw.pushConstants = std::span<const std::byte>(
            reinterpret_cast<const std::byte *>(
                &drawPushConstants_[group.sourceDrawIndex]),
            sizeof(PushConstants));
        indirectDrawItems_.push_back(indirectDraw);
        indirectBufferSignature =
            hashDrawBufferSignature(indirectBufferSignature, indirectDraw);
        indirectSourceDrawIndices_.push_back(group.sourceDrawIndex);
        indirectAlphaMasked_.push_back(drawAlphaMasked_[group.sourceDrawIndex]);
        commandCursor += drawCount;
      }
    }
    currentIndirectDrawBufferSignature_ = indirectBufferSignature;
  } else {
    currentIndirectDrawBufferSignature_ = kInvalidDrawSignature;
  }
  const std::span<const std::byte> uploadBytes{
      indirectCommandUploadBytes_.data(), indirectCommandUploadBytes_.size()};
  auto updateIndirectResult = gpu_.updateBuffer(
      bufferRings_[IndirectCommandRing][frameSlot].buffer->handle(),
      uploadBytes, 0);
  if (updateIndirectResult.hasError()) {
    return updateIndirectResult;
  }
  indirectPackCache_.valid = true;
  indirectPackCache_.drawSignature = drawSignature;
  indirectPackCache_.requiredBytes = packedRequiredBytes;
  frameSlotStates_[frameSlot].indirectUploadSignature = drawSignature;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
OpaqueRenderer::refreshCachedIndirectPack(uint32_t frameSlot,
                                          uint64_t drawSignature) {
  indirectAlphaMasked_.resize(indirectDrawItems_.size(), 0u);
  const BufferHandle indirectBufferHandle =
      bufferRings_[IndirectCommandRing][frameSlot].buffer->handle();
  uint64_t indirectBufferSignature =
      hashBufferHandleSignature(kFnvOffsetBasis64, indirectBufferHandle);
  for (size_t i = 0; i < indirectSourceDrawIndices_.size(); ++i) {
    const size_t sourceIndex = indirectSourceDrawIndices_[i];
    indirectDrawItems_[i].indirectBuffer = indirectBufferHandle;
    indirectDrawItems_[i].pushConstants = std::span<const std::byte>(
        reinterpret_cast<const std::byte *>(&drawPushConstants_[sourceIndex]),
        sizeof(PushConstants));
    indirectBufferSignature =
        hashDrawBufferSignature(indirectBufferSignature, indirectDrawItems_[i]);
    indirectAlphaMasked_[i] = drawAlphaMasked_[sourceIndex];
    indirectDrawItems_[i].alphaMasked = indirectAlphaMasked_[i] != 0u;
  }
  currentIndirectDrawBufferSignature_ = indirectDrawItems_.empty()
                                            ? kInvalidDrawSignature
                                            : indirectBufferSignature;
  if (frameSlotStates_[frameSlot].indirectUploadSignature != drawSignature) {
    const std::span<const std::byte> uploadBytes{
        indirectCommandUploadBytes_.data(), indirectCommandUploadBytes_.size()};
    auto updateIndirectResult = gpu_.updateBuffer(
        bufferRings_[IndirectCommandRing][frameSlot].buffer->handle(),
        uploadBytes, 0);
    if (updateIndirectResult.hasError()) {
      return updateIndirectResult;
    }
    frameSlotStates_[frameSlot].indirectUploadSignature = drawSignature;
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> OpaqueRenderer::ensureInitialized() {
  if (initialized_) {
    return Result<bool, std::string>::makeResult(true);
  }
  auto shaderResult = createShaders();
  if (shaderResult.hasError()) {
    return shaderResult;
  }
  auto pickTextureResult = recreatePickTexture();
  if (pickTextureResult.hasError()) {
    return pickTextureResult;
  }
  auto pipelineResult = createPipelines();
  if (pipelineResult.hasError()) {
    destroyPickTexture();
    return pipelineResult;
  }
  auto staticBuffers = ensureStaticInstanceBufferCapacity(1u);
  if (staticBuffers.hasError()) {
    return staticBuffers;
  }
  initialized_ = true;
  return Result<bool, std::string>::makeResult(true);
}

bool OpaqueRenderer::meshletPipelinesConfigured() const noexcept {
  return !config_.meshletTask.empty() && !config_.meshletMesh.empty() &&
         !config_.meshletFragment.empty() &&
         !config_.meshletDepthFragment.empty() &&
         !config_.meshletDepthAlphaFragment.empty();
}

Result<bool, std::string> OpaqueRenderer::ensureSceneDepthSampler() {
  if (nuri::isValid(sceneDepthSampler_)) {
    return Result<bool, std::string>::makeResult(true);
  }
  auto samplerResult =
      gpu_.createSampler(SamplerDesc{.minFilter = SamplerFilter::Nearest,
                                     .magFilter = SamplerFilter::Nearest,
                                     .mipMode = SamplerMipMode::Disabled,
                                     .wrapU = SamplerWrapMode::Clamp,
                                     .wrapV = SamplerWrapMode::Clamp,
                                     .wrapW = SamplerWrapMode::Clamp},
                         "scene_depth_nearest_clamp");
  if (samplerResult.hasError()) {
    return Result<bool, std::string>::makeError(samplerResult.error());
  }
  sceneDepthSampler_ = samplerResult.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> OpaqueRenderer::ensureDepthPyramidTextures(
    bool currentFrameVerificationRequired) {
  int32_t framebufferWidth = 0;
  int32_t framebufferHeight = 0;
  gpu_.getFramebufferSize(framebufferWidth, framebufferHeight);
  const uint32_t safeWidth =
      static_cast<uint32_t>(std::max(framebufferWidth, 1));
  const uint32_t safeHeight =
      static_cast<uint32_t>(std::max(framebufferHeight, 1));
  const uint32_t pyramidWidth = std::max(1u, (safeWidth + 1u) >> 1u);
  const uint32_t pyramidHeight = std::max(1u, (safeHeight + 1u) >> 1u);
  const uint32_t levelCount =
      fullDepthPyramidLevelCount(pyramidWidth, pyramidHeight);
  auto samplerResult = ensureSceneDepthSampler();
  if (samplerResult.hasError()) {
    return samplerResult;
  }
  const bool recreatePyramid = sceneDepthPyramidLevelCount_ != levelCount ||
                               sceneDepthPyramidWidth_ != safeWidth ||
                               sceneDepthPyramidHeight_ != safeHeight;
  const bool recreateVerification =
      currentFrameVerificationRequired !=
      nuri::isValid(currentFrameDepthVerificationTexture_);
  if (!recreatePyramid && !recreateVerification)
    return Result<bool, std::string>::makeResult(true);
  if (recreatePyramid) {
    destroyDepthPyramidTextures();
  } else if (nuri::isValid(currentFrameDepthVerificationTexture_)) {
    gpu_.destroyTexture(currentFrameDepthVerificationTexture_);
    currentFrameDepthVerificationTexture_ = {};
  }
  if (currentFrameVerificationRequired) {
    const TextureDesc verificationDesc{
        .type = TextureType::Texture2D,
        .format = Format::R32_FLOAT,
        .dimensions = {safeWidth, safeHeight, 1u},
        .usage = TextureUsage::AttachmentSampled,
        .storage = Storage::Device,
        .numLayers = 1u,
        .numSamples = 1u,
        .numMipLevels = 1u,
        .data = {},
        .dataNumMipLevels = 1u,
        .generateMipmaps = false,
    };
    auto verificationResult = gpu_.createTexture(
        verificationDesc, "opaque_current_frame_depth_verification");
    if (verificationResult.hasError()) {
      return Result<bool, std::string>::makeError(verificationResult.error());
    }
    currentFrameDepthVerificationTexture_ = verificationResult.value();
  }
  if (!recreatePyramid) {
    return Result<bool, std::string>::makeResult(true);
  }
  uint32_t width = pyramidWidth;
  uint32_t height = pyramidHeight;
  for (uint32_t level = 0u; level < levelCount; ++level) {
    const TextureDesc desc{
        .type = TextureType::Texture2D,
        .format = Format::RG32_FLOAT,
        .dimensions = {width, height, 1u},
        .usage = TextureUsage::AttachmentSampled,
        .storage = Storage::Device,
        .numLayers = 1u,
        .numSamples = 1u,
        .numMipLevels = 1u,
        .data = {},
        .dataNumMipLevels = 1u,
        .generateMipmaps = false,
    };
    auto textureResult = gpu_.createTexture(desc, "opaque_scene_depth_minmax_" +
                                                      std::to_string(level));
    if (textureResult.hasError()) {
      destroyDepthPyramidTextures();
      return Result<bool, std::string>::makeError(textureResult.error());
    }
    sceneDepthPyramidTextures_[level] = textureResult.value();
    width = std::max(1u, (width + 1u) >> 1u);
    height = std::max(1u, (height + 1u) >> 1u);
  }
  sceneDepthPyramidLevelCount_ = levelCount;
  sceneDepthPyramidWidth_ = safeWidth;
  sceneDepthPyramidHeight_ = safeHeight;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> OpaqueRenderer::recreatePickTexture() {
  if (nuri::isValid(pickIdTexture_)) {
    gpu_.destroyTexture(pickIdTexture_);
    pickIdTexture_ = TextureHandle{};
  }
  const TextureDesc pickDesc{
      .type = TextureType::Texture2D,
      .format = Format::R32_UINT,
      .dimensions = {1, 1, 1},
      .usage = TextureUsage::AttachmentSampled,
      .storage = Storage::Device,
      .numLayers = 1,
      .numSamples = 1,
      .numMipLevels = 1,
      .data = {},
      .dataNumMipLevels = 1,
      .generateMipmaps = false,
  };
  auto pickResult =
      gpu_.createFramebufferTexture(pickDesc, "opaque_pick_id_texture");
  if (pickResult.hasError()) {
    return Result<bool, std::string>::makeError(pickResult.error());
  }
  pickIdTexture_ = pickResult.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
OpaqueRenderer::ensureTransmissionVisibilityDepthTexture(
    TextureHandle sceneDepthTexture) {
  const TextureDimensions dimensions =
      gpu_.getTextureDimensions(sceneDepthTexture);
  const Format format = gpu_.getTextureFormat(sceneDepthTexture);
  const uint32_t width = std::max(dimensions.width, 1u);
  const uint32_t height = std::max(dimensions.height, 1u);
  bool recreate = !nuri::isValid(transmissionVisibilityDepthTexture_);
  if (!recreate) {
    const TextureDimensions currentDimensions =
        gpu_.getTextureDimensions(transmissionVisibilityDepthTexture_);
    recreate =
        gpu_.getTextureFormat(transmissionVisibilityDepthTexture_) != format ||
        currentDimensions.width != width || currentDimensions.height != height;
  }
  if (!recreate) {
    return Result<bool, std::string>::makeResult(true);
  }
  destroyTransmissionVisibilityDepthTexture();
  const TextureDesc desc{
      .type = TextureType::Texture2D,
      .format = format,
      .dimensions = {width, height, 1u},
      .usage = TextureUsage::Attachment,
      .storage = Storage::Device,
      .numLayers = 1u,
      .numSamples = 1u,
      .numMipLevels = 1u,
      .data = {},
      .dataNumMipLevels = 1u,
      .generateMipmaps = false,
  };
  auto textureResult =
      gpu_.createTexture(desc, "opaque_transmission_visibility_depth");
  if (textureResult.hasError()) {
    return Result<bool, std::string>::makeError(textureResult.error());
  }
  transmissionVisibilityDepthTexture_ = textureResult.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> OpaqueRenderer::recreateShadowInspectTexture() {
  if (nuri::isValid(shadowInspectTexture_)) {
    gpu_.destroyTexture(shadowInspectTexture_);
    shadowInspectTexture_ = TextureHandle{};
  }
  const TextureDesc inspectDesc{
      .type = TextureType::Texture2D,
      .format = Format::RGBA32_FLOAT,
      .dimensions = {1u, 1u, 1u},
      .usage = TextureUsage::Attachment,
      .storage = Storage::Device,
      .numLayers = 1u,
      .numSamples = 1u,
      .numMipLevels = 1u,
      .data = {},
      .dataNumMipLevels = 1u,
      .generateMipmaps = false,
  };
  auto inspectResult = gpu_.createFramebufferTexture(
      inspectDesc, "opaque_shadow_inspect_texture");
  if (inspectResult.hasError()) {
    return Result<bool, std::string>::makeError(inspectResult.error());
  }
  shadowInspectTexture_ = inspectResult.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
OpaqueRenderer::ensureStaticInstanceBufferCapacity(size_t count) {
  struct BufferSpec {
    std::unique_ptr<Buffer> *buffer;
    size_t *capacity;
    size_t stride;
    std::string_view name;
  };
  const std::array specs{
      BufferSpec{&instanceCentersPhaseBuffer_,
                 &instanceCentersPhaseBufferCapacityBytes_, sizeof(glm::vec4),
                 "opaque_instance_centers_phase_buffer"},
      BufferSpec{&instanceLodBoundsBuffer_,
                 &instanceLodBoundsBufferCapacityBytes_, sizeof(glm::vec4),
                 "opaque_instance_lod_bounds_buffer"},
      BufferSpec{&instanceBaseMatricesBuffer_,
                 &instanceBaseMatricesBufferCapacityBytes_, sizeof(glm::mat4),
                 "opaque_instance_base_matrices_buffer"},
  };
  for (const BufferSpec &spec : specs) {
    auto result = ensureDynamicBufferCapacity(
        gpu_, *spec.buffer, *spec.capacity,
        BufferDesc{.usage = BufferUsage::Storage,
                   .storage = Storage::Device,
                   .size = std::max(count, size_t{1u}) * spec.stride},
        spec.name);
    if (result.hasError()) {
      return result;
    }
    instanceStaticBuffersDirty_ |= result.value();
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
OpaqueRenderer::ensureRingBufferCount(uint32_t requiredCount) {
  if (!resizeDynamicBufferRings(requiredCount, bufferRings_)) {
    return Result<bool, std::string>::makeResult(true);
  }
  requiredCount =
      static_cast<uint32_t>(bufferRings_[InstanceMatricesRing].size());
  frameSlotStates_.assign(
      requiredCount, FrameSlotState{.expectedVisibleHash = kFnvOffsetBasis64});
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
OpaqueRenderer::ensureInstanceMatricesRingCapacity(size_t requiredBytes) {
  const size_t sceneInstanceBytes =
      std::max(instanceMatricesCpuCache_.size(), size_t{1u}) *
      sizeof(InstanceData);
  return ensureDynamicBufferRingCapacity(
      gpu_, bufferRings_[InstanceMatricesRing],
      BufferDesc{.usage = BufferUsage::Storage,
                 .storage = Storage::Device,
                 .size = std::max({requiredBytes, sceneInstanceBytes,
                                   sizeof(InstanceData)})},
      "opaque_instance_matrices_buffer", [this](size_t i) {
        frameSlotStates_[i].matricesUploadVersion =
            std::numeric_limits<uint64_t>::max();
      });
}

Result<bool, std::string> OpaqueRenderer::ensureDynamicRingCapacity(
    std::pmr::vector<DynamicBufferSlot> &ring, size_t requiredBytes,
    size_t minimumBytes, std::string_view debugNamePrefix, Storage storage) {
  return ensureDynamicBufferRingCapacity(
      gpu_, ring,
      BufferDesc{.usage = BufferUsage::Storage,
                 .storage = storage,
                 .size = std::max(requiredBytes, minimumBytes)},
      debugNamePrefix);
}

Result<bool, std::string>
OpaqueRenderer::ensurePreviousInstanceMatricesRingCapacity(
    size_t requiredBytes) {
  return ensureDynamicRingCapacity(bufferRings_[PreviousInstanceMatricesRing],
                                   requiredBytes, sizeof(InstanceData),
                                   "opaque_previous_instance_matrices_buffer",
                                   Storage::HostVisible);
}

Result<bool, std::string>
OpaqueRenderer::ensureVelocityInstanceFlagsRingCapacity(size_t requiredBytes) {
  return ensureDynamicRingCapacity(
      bufferRings_[VelocityInstanceFlagsRing], requiredBytes, sizeof(uint32_t),
      "opaque_velocity_instance_flags_buffer", Storage::HostVisible);
}

Result<bool, std::string>
OpaqueRenderer::ensureVelocityFrameDataRingCapacity(size_t requiredBytes) {
  return ensureDynamicRingCapacity(bufferRings_[VelocityFrameDataRing],
                                   requiredBytes, sizeof(VelocityFrameGpuData),
                                   "opaque_velocity_frame_data_buffer",
                                   Storage::HostVisible);
}

Result<bool, std::string>
OpaqueRenderer::ensureVelocityGeometryRingCapacity(size_t requiredBytes) {
  return ensureDynamicRingCapacity(
      bufferRings_[VelocityGeometryRing], requiredBytes,
      sizeof(VelocityRenderableGeometryGpuData),
      "opaque_velocity_geometry_buffer", Storage::HostVisible);
}

Result<bool, std::string>
OpaqueRenderer::ensureMeshletBatchRingCapacity(size_t requiredBytes) {
  const size_t requested =
      ((std::max(requiredBytes, sizeof(MeshletBatchGpuData)) +
        sizeof(MeshletBatchGpuData) - 1u) /
       sizeof(MeshletBatchGpuData)) *
      sizeof(MeshletBatchGpuData);
  return ensureDynamicBufferRingCapacity(
      gpu_, bufferRings_[MeshletBatchRing],
      BufferDesc{.usage = BufferUsage::Storage,
                 .storage = Storage::HostVisible,
                 .size = requested},
      "opaque_meshlet_batch_buffer");
}

Result<bool, std::string>
OpaqueRenderer::ensureVisibilityGpuRingCapacity(size_t candidateBytes,
                                                size_t visibleIndexBytes) {
  const size_t sceneCandidateCount = std::max(
      {visibilityCandidateGpuData_.size(), visibilityCandidates_.size(),
       meshDrawTemplates_.size(), size_t{1u}});
  const size_t sceneCandidateBytes =
      sceneCandidateCount * sizeof(VisibilityCandidateGpu);
  const size_t sceneVisibleIndexBytes = sceneCandidateCount * sizeof(uint32_t);
  auto candidateResult = ensureDynamicRingCapacity(
      bufferRings_[VisibilityCandidateRing],
      std::max(candidateBytes, sceneCandidateBytes),
      sizeof(VisibilityCandidateGpu), "opaque_visibility_candidate_buffer",
      Storage::HostVisible);
  if (candidateResult.hasError()) {
    return candidateResult;
  }
  auto passResult = ensureDynamicRingCapacity(
      bufferRings_[VisibilityPassRing], sizeof(VisibilityPassGpuData),
      sizeof(VisibilityPassGpuData), "opaque_visibility_pass_buffer",
      Storage::HostVisible);
  if (passResult.hasError()) {
    return passResult;
  }
  auto visibleResult = ensureVisibilityVisibleIndexRingCapacity(
      std::max(visibleIndexBytes, sceneVisibleIndexBytes));
  if (visibleResult.hasError()) {
    return visibleResult;
  }
  return ensureVisibilityCounterRingCapacity(sizeof(VisibilityCounterGpuData));
}

Result<bool, std::string>
OpaqueRenderer::ensureVisibilityMeshletIndirectRingCapacity(
    size_t dispatchBytes, size_t commandBytes) {
  const size_t maxDispatchBytes =
      kMaxDrawItemsForIndirectPath * sizeof(VisibilityMeshletDispatchGpuData);
  auto dispatchResult = ensureDynamicRingCapacity(
      bufferRings_[VisibilityMeshletDispatchRing],
      std::max(dispatchBytes, maxDispatchBytes),
      sizeof(VisibilityMeshletDispatchGpuData),
      "opaque_visibility_meshlet_dispatch_buffer", Storage::HostVisible);
  if (dispatchResult.hasError()) {
    return dispatchResult;
  }
  return ensureVisibilityMeshletIndirectCommandRingCapacity(commandBytes);
}

Result<bool, std::string>
OpaqueRenderer::ensureMeshletCompactionRingCapacity(size_t requiredBytes) {
  return ensureDynamicRingCapacity(
      bufferRings_[MeshletCompactionRing], requiredBytes,
      sizeof(CompactedMeshletGpuData), "opaque_meshlet_compaction_buffer",
      Storage::Device);
}

Result<bool, std::string>
OpaqueRenderer::ensureVisibilityMeshletIndirectCommandRingCapacity(
    size_t requiredBytes) {
  constexpr size_t kMeshDispatchCommandBytes = sizeof(uint32_t) * 3u;
  constexpr size_t kMaxMeshDispatchCommandBytes =
      kMaxDrawItemsForIndirectPath * kMeshDispatchCommandBytes;
  const size_t requested = ((std::max({requiredBytes, kMeshDispatchCommandBytes,
                                       kMaxMeshDispatchCommandBytes}) +
                             kMeshDispatchCommandBytes - 1u) /
                            kMeshDispatchCommandBytes) *
                           kMeshDispatchCommandBytes;
  return ensureDynamicBufferRingCapacity(
      gpu_, bufferRings_[VisibilityMeshletIndirectCommandRing],
      BufferDesc{.usage = BufferUsage::Storage | BufferUsage::Indirect,
                 .storage = Storage::Device,
                 .size = requested},
      "opaque_visibility_meshlet_indirect_commands");
}

Result<bool, std::string>
OpaqueRenderer::ensureVisibilityVisibleIndexRingCapacity(size_t requiredBytes) {
  return ensureDynamicBufferRingCapacity(
      gpu_, bufferRings_[VisibilityVisibleIndexRing],
      BufferDesc{.usage = BufferUsage::Storage,
                 .storage = Storage::HostVisible,
                 .size = std::max(requiredBytes, sizeof(uint32_t))},
      "opaque_visibility_visible_index_buffer",
      [this](size_t i) { invalidateVisibilityReadbackSlot(i); });
}

Result<bool, std::string>
OpaqueRenderer::ensureVisibilityCounterRingCapacity(size_t requiredBytes) {
  return ensureDynamicBufferRingCapacity(
      gpu_, bufferRings_[VisibilityCounterRing],
      BufferDesc{.usage = BufferUsage::Storage,
                 .storage = Storage::HostVisible,
                 .size =
                     std::max(requiredBytes, sizeof(VisibilityCounterGpuData))},
      "opaque_visibility_counter_buffer",
      [this](size_t i) { invalidateVisibilityReadbackSlot(i); });
}

void OpaqueRenderer::invalidateVisibilityReadbackSlot(size_t i) {
  frameSlotStates_[i].visibilityPublishedFrame =
      std::numeric_limits<uint64_t>::max();
  frameSlotStates_[i].expectedVisibleCount = 0u;
  frameSlotStates_[i].expectedVisibleHash = kFnvOffsetBasis64;
  frameSlotStates_[i].expectedVisibleListValid = 0u;
}

Result<bool, std::string>
OpaqueRenderer::ensureInstanceRemapRingCapacity(size_t requiredBytes) {
  const size_t sceneRemapBytes =
      std::max(meshDrawTemplates_.size(), size_t{1u}) * sizeof(uint32_t);
  return ensureDynamicBufferRingCapacity(
      gpu_, bufferRings_[InstanceRemapRing],
      BufferDesc{
          .usage = BufferUsage::Storage,
          .storage = Storage::HostVisible,
          .size = std::max({requiredBytes, sceneRemapBytes, sizeof(uint32_t)})},
      "opaque_instance_remap_buffer", [this](size_t i) {
        frameSlotStates_[i].remapUploadSignature = kInvalidDrawSignature;
      });
}

Result<bool, std::string>
OpaqueRenderer::ensureIndirectCommandRingCapacity(size_t requiredBytes) {
  constexpr size_t kIndirectCommandBytesPerDraw =
      kIndirectCountHeaderBytes + sizeof(DrawIndexedIndirectCommand);
  constexpr size_t kMaxIndirectCommandBytes =
      kMaxDrawItemsForIndirectPath * kIndirectCommandBytesPerDraw;
  return ensureDynamicBufferRingCapacity(
      gpu_, bufferRings_[IndirectCommandRing],
      BufferDesc{.usage = BufferUsage::Storage | BufferUsage::Indirect,
                 .storage = Storage::HostVisible,
                 .size = std::max({requiredBytes, kIndirectCountHeaderBytes,
                                   kMaxIndirectCommandBytes})},
      "opaque_indirect_commands_buffer", [this](size_t i) {
        frameSlotStates_[i].indirectUploadSignature = kInvalidDrawSignature;
      });
}

void OpaqueRenderer::rebuildSceneCache(const SceneDrawDatabase &database,
                                       const RenderScene &scene,
                                       bool excludeTransmission) {
  renderableTemplates_.assign(database.instances().begin(),
                              database.instances().end());
  meshDrawTemplates_.clear();
  meshDrawTemplates_.reserve(database.draws().size());
  for (const SceneDrawRecord &draw : database.draws()) {
    if (!draw.alphaBlended && !(excludeTransmission && draw.transmission)) {
      meshDrawTemplates_.push_back(draw);
    }
  }
  visibilityCandidates_.clear();
  visibilityCandidateGpuData_.clear();
  cachedVisibilityCandidateTopologyVersion_ = UINT64_MAX;
  cachedVisibilityCandidateTransformVersion_ = UINT64_MAX;
  cachedVisibilityCandidateDeformationVersion_ = UINT64_MAX;
  cachedVisibilityCandidateGeometryVersion_ = UINT64_MAX;
  cachedVisibilityCandidatesHadDeformedRenderable_ = false;
  cachedScene_ = &scene;
  cachedTopologyVersion_ = scene.topologyVersion();
  cachedGeometryMutationVersion_ = gpu_.geometryMutationVersion();
  cachedExcludeTransmission_ = excludeTransmission;
  uniformSingleSubmeshPath_ =
      !meshDrawTemplates_.empty() &&
      meshDrawTemplates_.size() == renderableTemplates_.size();
  if (uniformSingleSubmeshPath_) {
    const MeshDrawTemplate &first = meshDrawTemplates_.front();
    for (size_t i = 0; i < meshDrawTemplates_.size(); ++i) {
      const MeshDrawTemplate &draw = meshDrawTemplates_[i];
      if (draw.instanceIndex != i ||
          draw.geometryHandle.index != first.geometryHandle.index ||
          draw.geometryHandle.generation != first.geometryHandle.generation ||
          draw.submesh != first.submesh ||
          draw.materialIndex != first.materialIndex) {
        uniformSingleSubmeshPath_ = false;
        break;
      }
    }
  }
  instanceStaticBuffersDirty_ = true;
  invalidateSingleInstanceBatchCache();
  invalidateStaticBatchCache();
  invalidateIndirectPackCache();
}
Result<bool, std::string> OpaqueRenderer::rebuildMaterialTextureAccessCache(
    const SceneDrawDatabase &database, const ResourceManager &resources,
    bool excludeTransmission) {
  NURI_PROFILER_FUNCTION();
  materialTextureAccessHandles_.clear();
  materialTextureAccessCacheValid_ = false;
  ScratchArena scratch;
  ScopedScratch scopedScratch(scratch);
  PmrHashSet<uint64_t> textureKeys(scopedScratch.resource());
  textureKeys.reserve(database.draws().size());
  materialTextureAccessHandles_.reserve(database.draws().size());
  for (const SceneDrawRecord &draw : database.draws()) {
    if (draw.alphaBlended || (excludeTransmission && draw.transmission)) {
      continue;
    }
    const MaterialRecord *material = resources.tryGet(draw.material);
    if (!material) {
      continue;
    }
    forEachMaterialTextureRef(material->textureRefs, [&](TextureRef ref) {
      const TextureRecord *texture = resources.tryGet(ref);
      if (!texture || !nuri::isValid(texture->texture)) {
        return;
      }
      const uint64_t key =
          foldHandle(texture->texture.index, texture->texture.generation);
      if (textureKeys.insert(key).second) {
        materialTextureAccessHandles_.push_back(texture->texture);
      }
    });
  }
  materialTextureAccessCacheValid_ = true;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> OpaqueRenderer::createShaders() {
  shaders_.fill({});
  tessellationUnsupported_ = false;
  overlayPipelineUnsupported_.fill(false);
  struct ShaderSpec {
    std::string_view name;
    const std::filesystem::path *path = nullptr;
    ShaderStage stage = ShaderStage::Vertex;
    ShaderHandle *outHandle = nullptr;
  };
  const auto compileGroup = [this](std::span<const ShaderSpec> specs) {
    for (const ShaderSpec &spec : specs) {
      auto result = Shader{spec.name, gpu_}.compileFromFile(spec.path->string(),
                                                            spec.stage);
      if (result.hasError()) {
        for (const ShaderSpec &resetSpec : specs) {
          *resetSpec.outHandle = {};
        }
        return result.error();
      }
      *spec.outHandle = result.value();
    }
    return std::string{};
  };
  const auto compileOptional = [&](std::span<const ShaderSpec> specs) {
    return compileGroup(specs).empty();
  };
  const std::array<ShaderSpec, 3> shaderSpecs = {
      ShaderSpec{"main", &config_.meshVertex, ShaderStage::Vertex,
                 &shaders_[MeshVertex]},
      ShaderSpec{"main", &config_.meshFragment, ShaderStage::Fragment,
                 &shaders_[MeshFragment]},
      ShaderSpec{"duck_instances", &config_.computeInstances,
                 ShaderStage::Compute, &shaders_[Compute]},
  };
  if (std::string error = compileGroup(shaderSpecs); !error.empty()) {
    return Result<bool, std::string>::makeError(std::move(error));
  }
  const std::filesystem::path shaderDir = config_.meshFragment.parent_path();
  struct VisibilityShaderSpec {
    std::string_view name;
    std::string_view file;
    ShaderHandle *output;
  };
  for (const VisibilityShaderSpec &spec :
       {VisibilityShaderSpec{"visibility_cull", "visibility_cull.comp",
                             &shaders_[VisibilityCompute]},
        VisibilityShaderSpec{"visibility_indirect_draw",
                             "visibility_indirect_draw.comp",
                             &shaders_[VisibilityIndirectDrawCompute]},
        VisibilityShaderSpec{
            "visibility_indirect_mesh_dispatch",
            "visibility_indirect_mesh_dispatch.comp",
            &shaders_[VisibilityIndirectMeshDispatchCompute]}}) {
    auto result = Shader{spec.name, gpu_}.compileFromFile(
        (shaderDir / spec.file).string(), ShaderStage::Compute);
    if (!result.hasError()) {
      *spec.output = result.value();
    }
  }
  auto compactionResult =
      Shader{"opaque_meshlet_compact", gpu_}.compileFromFile(
          (shaderDir / "opaque_meshlet_compact.comp").string(),
          ShaderStage::Compute);
  if (compactionResult.hasError()) {
    return Result<bool, std::string>::makeError(
        "OpaqueRenderer::createShaders: meshlet pre-task compaction shader "
        "failed: " +
        compactionResult.error());
  }
  shaders_[MeshletCompactionCompute] = compactionResult.value();
  const std::array<std::filesystem::path, 8> requiredPaths = {
      shaderDir / "main_id_only.vert",
      config_.pickFragment,
      config_.shadowInspectFragment.empty() ? shaderDir / "shadow_inspect.frag"
                                            : config_.shadowInspectFragment,
      shaderDir / "opaque_velocity.vert",
      shaderDir / "opaque_velocity.frag",
      shaderDir / "opaque_velocity_tess.vert",
      shaderDir / "opaque_velocity_tess.tesc",
      shaderDir / "opaque_velocity_tess.tese"};
  const std::array<ShaderSpec, 8> requiredSpecs = {
      ShaderSpec{"main_id", &requiredPaths[0], ShaderStage::Vertex,
                 &shaders_[MeshPickVertex]},
      ShaderSpec{"main_id", &requiredPaths[1], ShaderStage::Fragment,
                 &shaders_[MeshPickFragment]},
      ShaderSpec{"shadow_inspect", &requiredPaths[2], ShaderStage::Fragment,
                 &shaders_[MeshShadowInspectFragment]},
      ShaderSpec{"opaque_velocity", &requiredPaths[3], ShaderStage::Vertex,
                 &shaders_[MeshVelocityVertex]},
      ShaderSpec{"opaque_velocity", &requiredPaths[4], ShaderStage::Fragment,
                 &shaders_[MeshVelocityFragment]},
      ShaderSpec{"opaque_velocity", &requiredPaths[5], ShaderStage::Vertex,
                 &shaders_[MeshVelocityTessVertex]},
      ShaderSpec{"opaque_velocity", &requiredPaths[6], ShaderStage::TessControl,
                 &shaders_[MeshVelocityTessControl]},
      ShaderSpec{"opaque_velocity", &requiredPaths[7], ShaderStage::TessEval,
                 &shaders_[MeshVelocityTessEval]}};
  if (std::string error = compileGroup(requiredSpecs); !error.empty()) {
    return Result<bool, std::string>::makeError(std::move(error));
  }
  const std::array<std::filesystem::path, 11> optionalPaths = {
      shaderDir / "opaque_reactive_mask.vert",
      shaderDir / "opaque_reactive_mask.frag",
      shaderDir / "opaque_normal.frag",
      shaderDir / "opaque_depth.vert",
      shaderDir / "opaque_depth.frag",
      shaderDir / "opaque_depth_alpha.vert",
      shaderDir / "opaque_depth_alpha.frag",
      shaderDir / "fullscreen_copy.vert",
      shaderDir / "depth_minmax_pyramid.frag",
      shaderDir / "fullscreen_copy.vert",
      shaderDir / "opaque_depth_motion_vector.frag"};
  const std::array<ShaderSpec, 11> optionalSpecs = {
      ShaderSpec{"opaque_reactive_mask", &optionalPaths[0], ShaderStage::Vertex,
                 &shaders_[MeshReactiveMaskVertex]},
      ShaderSpec{"opaque_reactive_mask", &optionalPaths[1],
                 ShaderStage::Fragment, &shaders_[MeshReactiveMaskFragment]},
      ShaderSpec{"opaque_normal", &optionalPaths[2], ShaderStage::Fragment,
                 &shaders_[MeshNormalFragment]},
      ShaderSpec{"opaque_depth", &optionalPaths[3], ShaderStage::Vertex,
                 &shaders_[DepthVertex]},
      ShaderSpec{"opaque_depth", &optionalPaths[4], ShaderStage::Fragment,
                 &shaders_[DepthFragment]},
      ShaderSpec{"opaque_depth_alpha", &optionalPaths[5], ShaderStage::Vertex,
                 &shaders_[DepthAlphaVertex]},
      ShaderSpec{"opaque_depth_alpha", &optionalPaths[6], ShaderStage::Fragment,
                 &shaders_[DepthAlphaFragment]},
      ShaderSpec{"depth_minmax_pyramid", &optionalPaths[7], ShaderStage::Vertex,
                 &shaders_[DepthPyramidVertex]},
      ShaderSpec{"depth_minmax_pyramid", &optionalPaths[8],
                 ShaderStage::Fragment, &shaders_[DepthPyramidFragment]},
      ShaderSpec{"opaque_depth_motion_vector", &optionalPaths[9],
                 ShaderStage::Vertex, &shaders_[DepthMotionVectorVertex]},
      ShaderSpec{"opaque_depth_motion_vector", &optionalPaths[10],
                 ShaderStage::Fragment, &shaders_[DepthMotionVectorFragment]}};
  compileOptional(std::span(optionalSpecs).subspan(0, 2));
  compileOptional(std::span(optionalSpecs).subspan(2, 1));
  compileOptional(std::span(optionalSpecs).subspan(3, 1));
  compileOptional(std::span(optionalSpecs).subspan(4, 1));
  compileOptional(std::span(optionalSpecs).subspan(5, 1));
  compileOptional(std::span(optionalSpecs).subspan(6, 1));
  compileOptional(std::span(optionalSpecs).subspan(7, 2));
  compileOptional(std::span(optionalSpecs).subspan(9, 2));
  const std::array<ShaderSpec, 3> tessShaderSpecs = {
      ShaderSpec{"main_tess", &config_.tessVertex, ShaderStage::Vertex,
                 &shaders_[MeshTessVertex]},
      ShaderSpec{"main_tess", &config_.tessControl, ShaderStage::TessControl,
                 &shaders_[MeshTessControl]},
      ShaderSpec{"main_tess", &config_.tessEval, ShaderStage::TessEval,
                 &shaders_[MeshTessEval]},
  };
  tessellationUnsupported_ = !compileOptional(tessShaderSpecs);
  if (!tessellationUnsupported_) {
    const std::array<std::filesystem::path, 9> tessPaths = {
        shaderDir / "main_id_tess.vert",
        shaderDir / "main_id_tess.tesc",
        shaderDir / "main_id_tess.tese",
        shaderDir / "opaque_depth_alpha_tess.vert",
        shaderDir / "opaque_depth_alpha_tess.tesc",
        shaderDir / "opaque_depth_alpha_tess.tese",
        shaderDir / "opaque_depth_tess.vert",
        shaderDir / "opaque_depth_tess.tesc",
        shaderDir / "opaque_depth_tess.tese"};
    const std::array<ShaderSpec, 9> tessSpecs = {
        ShaderSpec{"main_id", &tessPaths[0], ShaderStage::Vertex,
                   &shaders_[MeshPickTessVertex]},
        ShaderSpec{"main_id", &tessPaths[1], ShaderStage::TessControl,
                   &shaders_[MeshPickTessControl]},
        ShaderSpec{"main_id", &tessPaths[2], ShaderStage::TessEval,
                   &shaders_[MeshPickTessEval]},
        ShaderSpec{"opaque_depth_alpha", &tessPaths[3], ShaderStage::Vertex,
                   &shaders_[DepthAlphaTessVertex]},
        ShaderSpec{"opaque_depth_alpha", &tessPaths[4],
                   ShaderStage::TessControl, &shaders_[DepthAlphaTessControl]},
        ShaderSpec{"opaque_depth_alpha", &tessPaths[5], ShaderStage::TessEval,
                   &shaders_[DepthAlphaTessEval]},
        ShaderSpec{"opaque_depth", &tessPaths[6], ShaderStage::Vertex,
                   &shaders_[DepthTessVertex]},
        ShaderSpec{"opaque_depth", &tessPaths[7], ShaderStage::TessControl,
                   &shaders_[DepthTessControl]},
        ShaderSpec{"opaque_depth", &tessPaths[8], ShaderStage::TessEval,
                   &shaders_[DepthTessEval]}};
    compileOptional(std::span(tessSpecs).subspan(0, 3));
    compileOptional(std::span(tessSpecs).subspan(3, 3));
    compileOptional(std::span(tessSpecs).subspan(6, 3));
  }
  const std::array<ShaderSpec, 2> overlayShaderSpecs = {
      ShaderSpec{"mesh_debug_overlay", &config_.overlayGeometry,
                 ShaderStage::Geometry, &shaders_[MeshDebugOverlayGeometry]},
      ShaderSpec{"mesh_debug_overlay", &config_.overlayFragment,
                 ShaderStage::Fragment, &shaders_[MeshDebugOverlayFragment]},
  };
  const bool overlayUnsupported = !compileOptional(overlayShaderSpecs);
  overlayPipelineUnsupported_[overlayPipelineIndex(
      OverlayPipelineKind::Geometry, CoverageMode::Sample1)] =
      overlayUnsupported;
  overlayPipelineUnsupported_[overlayPipelineIndex(
      OverlayPipelineKind::TessGeometry, CoverageMode::Sample1)] =
      overlayUnsupported;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> OpaqueRenderer::createPipelines() {
  resetMeshPipelineState();
  resetOverlayPipelineState();
  const auto createPipeline = [this](const RenderPipelineDesc &desc,
                                     std::string_view name,
                                     RenderPipelineHandle &output) {
    auto result = gpu_.createRenderPipeline(desc, name);
    if (result.hasError()) {
      output = {};
      return result.error();
    }
    output = result.value();
    return std::string{};
  };
  const auto createOptional = [&](const RenderPipelineDesc &desc,
                                  std::string_view name,
                                  RenderPipelineHandle &output) {
    return createPipeline(desc, name, output).empty();
  };
  const Format depthFormat = kFrameCompositionDepthFormat;
  const RenderPipelineDesc meshDesc =
      withOpaqueMainDepthVariants(meshPipelineDesc(
          kFrameCompositionSceneColorFormat, depthFormat, shaders_[MeshVertex],
          {}, {}, {}, shaders_[MeshFragment], PolygonMode::Fill));
  auto meshResult = gpu_.createRenderPipeline(meshDesc, "opaque_mesh");
  if (meshResult.hasError()) {
    return Result<bool, std::string>::makeError(meshResult.error());
  }
  meshPipeline_.reset(gpu_, meshResult.value());
  const GpuMultisampleCapabilities multisampleCapabilities =
      gpu_.getMultisampleCapabilities();
  const bool supportsMsaa8x = multisampleCapabilities.sample8Color &&
                              multisampleCapabilities.sample8Depth;
  const auto makeMsaaSceneDesc = [](RenderPipelineDesc desc,
                                    uint32_t sampleCount) {
    desc.numSamples = sampleCount;
    return desc;
  };
  const auto makeAlphaMsaaSceneDesc =
      [multisampleCapabilities](RenderPipelineDesc desc, uint32_t sampleCount) {
        desc.numSamples = sampleCount;
        desc.alphaToCoverageEnabled = multisampleCapabilities.alphaToCoverage;
        desc.minSampleShading = 0.0f;
        return desc;
      };
  constexpr std::array msaaCoverages{CoverageMode::Sample4,
                                     CoverageMode::Sample8};
  const auto createMsaaSceneVariants =
      [&](const RenderPipelineDesc &baseDesc, bool tessellated,
          bool doubleSided,
          const std::array<std::array<std::string_view, 2>, 2> &names) {
        for (size_t coverageIndex = 0; coverageIndex < msaaCoverages.size();
             ++coverageIndex) {
          const CoverageMode coverage = msaaCoverages[coverageIndex];
          if (coverage == CoverageMode::Sample8 && !supportsMsaa8x) {
            continue;
          }
          const uint32_t sampleCount = coverageSampleCount(coverage);
          for (size_t alphaIndex = 0; alphaIndex < 2u; ++alphaIndex) {
            const bool alphaMasked = alphaIndex != 0u;
            RenderPipelineDesc desc =
                alphaMasked ? makeAlphaMsaaSceneDesc(baseDesc, sampleCount)
                            : makeMsaaSceneDesc(baseDesc, sampleCount);
            RenderPipelineHandle &output =
                meshScenePipelines_[rasterVariantIndex(
                    coverage, alphaMasked, tessellated, doubleSided)];
            if (std::string error = createPipeline(
                    desc, names[coverageIndex][alphaIndex], output);
                !error.empty()) {
              return error;
            }
          }
        }
        return std::string{};
      };
  RenderPipelineDesc doubleSidedMeshDesc = meshDesc;
  doubleSidedMeshDesc.cullMode = CullMode::None;
  std::string meshVariantError =
      createPipeline(doubleSidedMeshDesc, "opaque_mesh_double_sided",
                     meshScenePipelines_[rasterVariantIndex(
                         CoverageMode::Sample1, false, false, true)]);
  if (meshVariantError.empty()) {
    meshVariantError = createMsaaSceneVariants(
        meshDesc, false, false,
        {{{"opaque_mesh_msaa4x", "opaque_mesh_alpha_msaa4x"},
          {"opaque_mesh_msaa8x", "opaque_mesh_alpha_msaa8x"}}});
  }
  if (meshVariantError.empty()) {
    meshVariantError =
        createMsaaSceneVariants(doubleSidedMeshDesc, false, true,
                                {{{"opaque_mesh_double_sided_msaa4x",
                                   "opaque_mesh_alpha_double_sided_msaa4x"},
                                  {"opaque_mesh_double_sided_msaa8x",
                                   "opaque_mesh_alpha_double_sided_msaa8x"}}});
  }
  if (!meshVariantError.empty()) {
    return Result<bool, std::string>::makeError(std::move(meshVariantError));
  }
  struct AuxiliaryPipelineSpec {
    Format colorFormat;
    ShaderHandle vertex;
    ShaderHandle fragment;
    std::string_view name;
    std::string_view doubleSidedName;
    RenderPipelineHandle *pipeline;
    RenderPipelineHandle *doubleSidedPipeline;
  };
  for (const AuxiliaryPipelineSpec &spec :
       {AuxiliaryPipelineSpec{
            kFrameCompositionMotionVectorFormat, shaders_[MeshVelocityVertex],
            shaders_[MeshVelocityFragment], "opaque_velocity",
            "opaque_velocity_double_sided",
            &meshVelocityPipelines_[surfaceVariantIndex(false, false)],
            &meshVelocityPipelines_[surfaceVariantIndex(false, true)]},
        AuxiliaryPipelineSpec{
            kFrameCompositionReactiveMaskFormat,
            shaders_[MeshReactiveMaskVertex],
            shaders_[MeshReactiveMaskFragment], "opaque_reactive_mask",
            "opaque_reactive_mask_double_sided", &meshReactiveMaskPipelines_[0],
            &meshReactiveMaskPipelines_[1]},
        AuxiliaryPipelineSpec{
            kFrameCompositionNormalFormat, shaders_[MeshVertex],
            shaders_[MeshNormalFragment], "opaque_material_normals",
            "opaque_material_normals_double_sided",
            &meshNormalPipelines_[surfaceVariantIndex(false, false)],
            &meshNormalPipelines_[surfaceVariantIndex(false, true)]}}) {
    if (!nuri::isValid(spec.vertex) || !nuri::isValid(spec.fragment)) {
      continue;
    }
    RenderPipelineDesc desc =
        meshPipelineDesc(spec.colorFormat, depthFormat, spec.vertex, {}, {}, {},
                         spec.fragment, PolygonMode::Fill);
    desc.prewarmRasterStates = kOpaqueReadOnlyAuxiliaryStates;
    if (!createOptional(desc, spec.name, *spec.pipeline)) {
      continue;
    }
    desc.cullMode = CullMode::None;
    createOptional(desc, spec.doubleSidedName, *spec.doubleSidedPipeline);
  }
  const auto createDepthPipeline =
      [this](const RenderPipelineDesc &desc, std::string_view debugName,
             RenderPipelineHandle &outHandle) -> bool {
    auto result = gpu_.createRenderPipeline(desc, debugName);
    if (result.hasError()) {
      outHandle = {};
      return false;
    }
    outHandle = result.value();
    return true;
  };
  const auto createDepthVariants =
      [&](const RenderPipelineDesc &baseDesc, bool alphaMasked,
          bool tessellated, bool doubleSided,
          const std::array<std::string_view, kCoverageModeCount> &names) {
        for (size_t index = 0; index < kCoverageModeCount; ++index) {
          const CoverageMode coverage = static_cast<CoverageMode>(index);
          if (coverage == CoverageMode::Sample8 && !supportsMsaa8x) {
            continue;
          }
          RenderPipelineDesc desc = baseDesc;
          desc.numSamples = coverageSampleCount(coverage);
          createDepthPipeline(
              desc, names[index],
              meshDepthPipelines_[rasterVariantIndex(
                  coverage, alphaMasked, tessellated, doubleSided)]);
        }
      };
  if (nuri::isValid(shaders_[DepthVertex]) &&
      nuri::isValid(shaders_[DepthFragment])) {
    const RenderPipelineDesc depthDesc =
        depthPipelineDesc(depthFormat, shaders_[DepthVertex], {}, {},
                          shaders_[DepthFragment], CullMode::Back);
    createDepthVariants(depthDesc, false, false, false,
                        {"opaque_mesh_depth", "opaque_mesh_depth_msaa4x",
                         "opaque_mesh_depth_msaa8x"});
    const RenderPipelineDesc doubleSidedDepthDesc =
        depthPipelineDesc(depthFormat, shaders_[DepthVertex], {}, {},
                          shaders_[DepthFragment], CullMode::None);
    createDepthVariants(doubleSidedDepthDesc, false, false, true,
                        {"opaque_mesh_depth_double_sided",
                         "opaque_mesh_depth_double_sided_msaa4x",
                         "opaque_mesh_depth_double_sided_msaa8x"});
  }
  if (nuri::isValid(shaders_[DepthAlphaVertex]) &&
      nuri::isValid(shaders_[DepthAlphaFragment])) {
    const RenderPipelineDesc depthAlphaDesc =
        depthPipelineDesc(depthFormat, shaders_[DepthAlphaVertex], {}, {},
                          shaders_[DepthAlphaFragment], CullMode::Back);
    createDepthVariants(depthAlphaDesc, true, false, false,
                        {"opaque_mesh_depth_alpha",
                         "opaque_mesh_depth_alpha_msaa4x",
                         "opaque_mesh_depth_alpha_msaa8x"});
    const RenderPipelineDesc doubleSidedDepthAlphaDesc =
        depthPipelineDesc(depthFormat, shaders_[DepthAlphaVertex], {}, {},
                          shaders_[DepthAlphaFragment], CullMode::None);
    createDepthVariants(doubleSidedDepthAlphaDesc, true, false, true,
                        {"opaque_mesh_depth_alpha_double_sided",
                         "opaque_mesh_depth_alpha_double_sided_msaa4x",
                         "opaque_mesh_depth_alpha_double_sided_msaa8x"});
  }
  const bool canCreateTessPipeline = !tessellationUnsupported_ &&
                                     nuri::isValid(shaders_[MeshTessVertex]) &&
                                     nuri::isValid(shaders_[MeshTessControl]) &&
                                     nuri::isValid(shaders_[MeshTessEval]) &&
                                     nuri::isValid(shaders_[MeshFragment]);
  const auto canCreatePickTessPipeline = [this]() -> bool {
    return !tessellationUnsupported_ &&
           nuri::isValid(shaders_[MeshPickTessVertex]) &&
           nuri::isValid(shaders_[MeshPickTessControl]) &&
           nuri::isValid(shaders_[MeshPickTessEval]) &&
           nuri::isValid(shaders_[MeshPickFragment]);
  };
  const auto canCreateDepthTessPipeline = [this]() -> bool {
    return !tessellationUnsupported_ &&
           nuri::isValid(shaders_[DepthTessVertex]) &&
           nuri::isValid(shaders_[DepthTessControl]) &&
           nuri::isValid(shaders_[DepthTessEval]) &&
           nuri::isValid(shaders_[DepthFragment]);
  };
  const auto canCreateDepthAlphaTessPipeline = [this]() -> bool {
    return !tessellationUnsupported_ &&
           nuri::isValid(shaders_[DepthAlphaTessVertex]) &&
           nuri::isValid(shaders_[DepthAlphaTessControl]) &&
           nuri::isValid(shaders_[DepthAlphaTessEval]) &&
           nuri::isValid(shaders_[DepthAlphaFragment]);
  };
  if (canCreateTessPipeline) {
    const RenderPipelineDesc tessDesc = withOpaqueMainDepthVariants(
        meshPipelineDesc(kFrameCompositionSceneColorFormat, depthFormat,
                         shaders_[MeshTessVertex], shaders_[MeshTessControl],
                         shaders_[MeshTessEval], {}, shaders_[MeshFragment],
                         PolygonMode::Fill, Topology::Patch,
                         kTessellationPatchControlPoints));
    std::string tessError =
        createPipeline(tessDesc, "opaque_mesh_tess",
                       meshScenePipelines_[rasterVariantIndex(
                           CoverageMode::Sample1, false, true, false)]);
    if (tessError.empty()) {
      tessError = createMsaaSceneVariants(
          tessDesc, true, false,
          {{{"opaque_mesh_tess_msaa4x", "opaque_mesh_tess_alpha_msaa4x"},
            {"opaque_mesh_tess_msaa8x", "opaque_mesh_tess_alpha_msaa8x"}}});
    }
    if (tessError.empty()) {
      const RenderPipelineDesc doubleSidedTessDesc =
          withOpaqueMainDepthVariants(meshPipelineDesc(
              kFrameCompositionSceneColorFormat, depthFormat,
              shaders_[MeshTessVertex], shaders_[MeshTessControl],
              shaders_[MeshTessEval], {}, shaders_[MeshFragment],
              PolygonMode::Fill, Topology::Patch,
              kTessellationPatchControlPoints, false, CullMode::None));
      tessError =
          createPipeline(doubleSidedTessDesc, "opaque_mesh_tess_double_sided",
                         meshScenePipelines_[rasterVariantIndex(
                             CoverageMode::Sample1, false, true, true)]);
      if (tessError.empty()) {
        tessError = createMsaaSceneVariants(
            doubleSidedTessDesc, true, true,
            {{{"opaque_mesh_tess_double_sided_msaa4x",
               "opaque_mesh_tess_alpha_double_sided_msaa4x"},
              {"opaque_mesh_tess_double_sided_msaa8x",
               "opaque_mesh_tess_alpha_double_sided_msaa8x"}}});
      }
    }
    if (!tessError.empty()) {
      tessellationUnsupported_ = true;
      for (RenderPipelineHandle &pipeline : meshScenePipelines_) {
        if (matchesVariantBit(pipeline, meshScenePipelines_, 2u)) {
          destroyPipelineHandle(gpu_, pipeline);
        }
      }
    }
    if (!tessellationUnsupported_) {
      if (canCreateDepthTessPipeline()) {
        const RenderPipelineDesc depthTessDesc = depthPipelineDesc(
            depthFormat, shaders_[DepthTessVertex], shaders_[DepthTessControl],
            shaders_[DepthTessEval], shaders_[DepthFragment], CullMode::Back,
            Topology::Patch, kTessellationPatchControlPoints);
        createDepthVariants(depthTessDesc, false, true, false,
                            {"opaque_mesh_depth_tess",
                             "opaque_mesh_depth_tess_msaa4x",
                             "opaque_mesh_depth_tess_msaa8x"});
        const RenderPipelineDesc depthDoubleSidedTessDesc = depthPipelineDesc(
            depthFormat, shaders_[DepthTessVertex], shaders_[DepthTessControl],
            shaders_[DepthTessEval], shaders_[DepthFragment], CullMode::None,
            Topology::Patch, kTessellationPatchControlPoints);
        createDepthVariants(depthDoubleSidedTessDesc, false, true, true,
                            {"opaque_mesh_depth_tess_double_sided",
                             "opaque_mesh_depth_tess_double_sided_msaa4x",
                             "opaque_mesh_depth_tess_double_sided_msaa8x"});
      }
      if (canCreateDepthAlphaTessPipeline()) {
        const RenderPipelineDesc depthAlphaTessDesc = depthPipelineDesc(
            depthFormat, shaders_[DepthAlphaTessVertex],
            shaders_[DepthAlphaTessControl], shaders_[DepthAlphaTessEval],
            shaders_[DepthAlphaFragment], CullMode::Back, Topology::Patch,
            kTessellationPatchControlPoints);
        createDepthVariants(depthAlphaTessDesc, true, true, false,
                            {"opaque_mesh_depth_alpha_tess",
                             "opaque_mesh_depth_alpha_tess_msaa4x",
                             "opaque_mesh_depth_alpha_tess_msaa8x"});
        const RenderPipelineDesc depthAlphaDoubleSidedTessDesc =
            depthPipelineDesc(depthFormat, shaders_[DepthAlphaTessVertex],
                              shaders_[DepthAlphaTessControl],
                              shaders_[DepthAlphaTessEval],
                              shaders_[DepthAlphaFragment], CullMode::None,
                              Topology::Patch, kTessellationPatchControlPoints);
        createDepthVariants(
            depthAlphaDoubleSidedTessDesc, true, true, true,
            {"opaque_mesh_depth_alpha_tess_double_sided",
             "opaque_mesh_depth_alpha_tess_double_sided_msaa4x",
             "opaque_mesh_depth_alpha_tess_double_sided_msaa8x"});
      }
      if (nuri::isValid(shaders_[MeshNormalFragment])) {
        RenderPipelineDesc normalTessDesc =
            withOpaqueReadOnlyAuxiliaryVariant(meshPipelineDesc(
                kFrameCompositionNormalFormat, depthFormat,
                shaders_[MeshTessVertex], shaders_[MeshTessControl],
                shaders_[MeshTessEval], {}, shaders_[MeshNormalFragment],
                PolygonMode::Fill, Topology::Patch,
                kTessellationPatchControlPoints));
        createOptional(normalTessDesc, "opaque_material_normals_tess",
                       meshNormalPipelines_[surfaceVariantIndex(true, false)]);
        normalTessDesc.cullMode = CullMode::None;
        createOptional(normalTessDesc,
                       "opaque_material_normals_tess_double_sided",
                       meshNormalPipelines_[surfaceVariantIndex(true, true)]);
      }
      if (nuri::isValid(shaders_[MeshVelocityTessVertex]) &&
          nuri::isValid(shaders_[MeshVelocityTessControl]) &&
          nuri::isValid(shaders_[MeshVelocityTessEval]) &&
          nuri::isValid(shaders_[MeshVelocityFragment])) {
        RenderPipelineDesc velocityTessDesc =
            withOpaqueReadOnlyAuxiliaryVariant(meshPipelineDesc(
                kFrameCompositionMotionVectorFormat, depthFormat,
                shaders_[MeshVelocityTessVertex],
                shaders_[MeshVelocityTessControl],
                shaders_[MeshVelocityTessEval], {},
                shaders_[MeshVelocityFragment], PolygonMode::Fill,
                Topology::Patch, kTessellationPatchControlPoints));
        createOptional(
            velocityTessDesc, "opaque_velocity_tess",
            meshVelocityPipelines_[surfaceVariantIndex(true, false)]);
        velocityTessDesc.cullMode = CullMode::None;
        createOptional(velocityTessDesc, "opaque_velocity_tess_double_sided",
                       meshVelocityPipelines_[surfaceVariantIndex(true, true)]);
      }
    }
  } else {
    tessellationUnsupported_ = true;
  }
  RenderPipelineDesc pickDesc = meshPipelineDesc(
      Format::R32_UINT, depthFormat, shaders_[MeshPickVertex], {}, {}, {},
      shaders_[MeshPickFragment], PolygonMode::Fill);
  std::string pickError =
      createPipeline(pickDesc, "opaque_mesh_pick",
                     meshPickPipelines_[surfaceVariantIndex(false, false)]);
  if (pickError.empty()) {
    pickDesc.cullMode = CullMode::None;
    pickError =
        createPipeline(pickDesc, "opaque_mesh_pick_double_sided",
                       meshPickPipelines_[surfaceVariantIndex(false, true)]);
  }
  if (!pickError.empty()) {
    for (RenderPipelineHandle *handle :
         {&meshPickPipelines_[surfaceVariantIndex(false, false)],
          &meshScenePipelines_[rasterVariantIndex(CoverageMode::Sample1, false,
                                                  true, false)],
          &meshScenePipelines_[rasterVariantIndex(CoverageMode::Sample1, false,
                                                  true, true)],
          &meshScenePipelines_[rasterVariantIndex(CoverageMode::Sample1, false,
                                                  false, true)]}) {
      destroyPipelineHandle(gpu_, *handle);
    }
    return Result<bool, std::string>::makeError(std::move(pickError));
  }
  if (canCreatePickTessPipeline()) {
    RenderPipelineDesc pickTessDesc = meshPipelineDesc(
        Format::R32_UINT, depthFormat, shaders_[MeshPickTessVertex],
        shaders_[MeshPickTessControl], shaders_[MeshPickTessEval], {},
        shaders_[MeshPickFragment], PolygonMode::Fill, Topology::Patch,
        kTessellationPatchControlPoints);
    createOptional(pickTessDesc, "opaque_mesh_tess_pick",
                   meshPickPipelines_[surfaceVariantIndex(true, false)]);
    pickTessDesc.cullMode = CullMode::None;
    createOptional(pickTessDesc, "opaque_mesh_tess_pick_double_sided",
                   meshPickPipelines_[surfaceVariantIndex(true, true)]);
  }
  RenderPipelineDesc inspectDesc =
      withOpaqueReadOnlyAuxiliaryVariant(meshPipelineDesc(
          Format::RGBA32_FLOAT, depthFormat, shaders_[MeshVertex], {}, {}, {},
          shaders_[MeshShadowInspectFragment], PolygonMode::Fill));
  createOptional(
      inspectDesc, "opaque_mesh_shadow_inspect",
      meshShadowInspectPipelines_[surfaceVariantIndex(false, false)]);
  inspectDesc.cullMode = CullMode::None;
  createOptional(inspectDesc, "opaque_mesh_shadow_inspect_double_sided",
                 meshShadowInspectPipelines_[surfaceVariantIndex(false, true)]);
  if (canCreateTessPipeline) {
    RenderPipelineDesc inspectTessDesc =
        withOpaqueReadOnlyAuxiliaryVariant(meshPipelineDesc(
            Format::RGBA32_FLOAT, depthFormat, shaders_[MeshTessVertex],
            shaders_[MeshTessControl], shaders_[MeshTessEval], {},
            shaders_[MeshShadowInspectFragment], PolygonMode::Fill,
            Topology::Patch, kTessellationPatchControlPoints));
    createOptional(
        inspectTessDesc, "opaque_mesh_tess_shadow_inspect",
        meshShadowInspectPipelines_[surfaceVariantIndex(true, false)]);
    if (!tessellationUnsupported_) {
      inspectTessDesc.cullMode = CullMode::None;
      createOptional(
          inspectTessDesc, "opaque_mesh_tess_shadow_inspect_double_sided",
          meshShadowInspectPipelines_[surfaceVariantIndex(true, true)]);
    }
  }
  const ComputePipelineDesc computeDesc{
      .computeShader = shaders_[Compute],
  };
  auto computeResult =
      gpu_.createComputePipeline(computeDesc, "opaque_instance_compute");
  if (computeResult.hasError()) {
    destroyMeshPipelineState();
    return Result<bool, std::string>::makeError(computeResult.error());
  }
  computePipeline_.reset(gpu_, computeResult.value());
  visibilityComputePipeline_.reset();
  visibilityIndirectDrawComputePipeline_.reset();
  visibilityIndirectMeshDispatchComputePipeline_.reset();
  struct ComputeSpec {
    ShaderHandle shader;
    std::string_view name;
    OwnedComputePipelineHandle *pipeline;
  };
  for (const ComputeSpec &spec :
       {ComputeSpec{shaders_[VisibilityCompute], "opaque_visibility_cull",
                    &visibilityComputePipeline_},
        ComputeSpec{shaders_[VisibilityIndirectDrawCompute],
                    "opaque_visibility_indirect_draw",
                    &visibilityIndirectDrawComputePipeline_},
        ComputeSpec{shaders_[VisibilityIndirectMeshDispatchCompute],
                    "opaque_visibility_indirect_mesh_dispatch",
                    &visibilityIndirectMeshDispatchComputePipeline_},
        ComputeSpec{shaders_[MeshletCompactionCompute],
                    "opaque_meshlet_pre_task_compaction",
                    &meshletCompactionComputePipeline_}}) {
    if (!nuri::isValid(spec.shader)) {
      continue;
    }
    auto result = gpu_.createComputePipeline(
        ComputePipelineDesc{.computeShader = spec.shader}, spec.name);
    if (!result.hasError()) {
      spec.pipeline->reset(gpu_, result.value());
    }
  }
  if (nuri::isValid(shaders_[DepthPyramidVertex]) &&
      nuri::isValid(shaders_[DepthPyramidFragment])) {
    RenderPipelineDesc pyramidDesc{
        .vertexInput = {},
        .vertexShader = shaders_[DepthPyramidVertex],
        .fragmentShader = shaders_[DepthPyramidFragment],
        .colorFormats = {Format::RG32_FLOAT},
        .colorAttachmentCount = 1u,
        .depthFormat = Format::Count,
        .cullMode = CullMode::None,
        .polygonMode = PolygonMode::Fill,
        .topology = Topology::Triangle,
        .patchControlPoints = 0u,
        .blendEnabled = false,
    };
    createOptional(pyramidDesc, "opaque_depth_minmax_pyramid",
                   depthPyramidPipelineHandle_);
    pyramidDesc.colorFormats[0] = Format::R32_FLOAT;
    createOptional(pyramidDesc, "opaque_current_frame_depth_verification",
                   currentFrameDepthVerificationPipelineHandle_);
  }
  if (nuri::isValid(shaders_[DepthMotionVectorVertex]) &&
      nuri::isValid(shaders_[DepthMotionVectorFragment])) {
    createOptional(fullscreenPipelineDesc(kFrameCompositionMotionVectorFormat,
                                          shaders_[DepthMotionVectorVertex],
                                          shaders_[DepthMotionVectorFragment]),
                   "opaque_depth_motion_vector",
                   depthMotionVectorPipelineHandle_);
  }
  baseMeshFillDraw_ = makeBaseMeshDraw(meshPipeline_.get(), "OpaqueMesh");
  for (uint8_t kind = 0u;
       kind < static_cast<uint8_t>(OverlayPipelineKind::Count); ++kind) {
    ensureOverlayPipeline(static_cast<OverlayPipelineKind>(kind),
                          CoverageMode::Sample4);
    if (supportsMsaa8x) {
      ensureOverlayPipeline(static_cast<OverlayPipelineKind>(kind),
                            CoverageMode::Sample8);
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> OpaqueRenderer::createMeshletPipelineState() {
  Shader shader{"opaque_meshlet", gpu_};
  const std::filesystem::path compactTaskPath =
      config_.meshletTask.parent_path() / "opaque_meshlet_compact.task.glsl";
  const auto meshletShaderDir = config_.meshletMesh.parent_path();
  struct ShaderSpec {
    std::filesystem::path path;
    ShaderStage stage;
    ShaderHandle *handle;
  };
  const std::array shaderSpecs{
      ShaderSpec{config_.meshletTask, ShaderStage::Task,
                 &shaders_[MeshletTask]},
      ShaderSpec{compactTaskPath, ShaderStage::Task,
                 &shaders_[MeshletCompactedTask]},
      ShaderSpec{config_.meshletMesh, ShaderStage::Mesh,
                 &shaders_[MeshletMesh]},
      ShaderSpec{config_.meshletFragment, ShaderStage::Fragment,
                 &shaders_[MeshletFragment]},
      ShaderSpec{config_.meshletDepthFragment, ShaderStage::Fragment,
                 &shaders_[MeshletDepthFragment]},
      ShaderSpec{config_.meshletDepthAlphaFragment, ShaderStage::Fragment,
                 &shaders_[MeshletDepthAlphaFragment]},
      ShaderSpec{meshletShaderDir / "opaque_meshlet_normal_simple.mesh.glsl",
                 ShaderStage::Mesh, &shaders_[MeshletSimpleNormalMesh]},
      ShaderSpec{meshletShaderDir / "opaque_meshlet_normal_simple.frag",
                 ShaderStage::Fragment, &shaders_[MeshletSimpleNormalFragment]},
      ShaderSpec{meshletShaderDir / "opaque_meshlet_normal.frag",
                 ShaderStage::Fragment, &shaders_[MeshletNormalFragment]},
      ShaderSpec{meshletShaderDir / "opaque_meshlet_velocity.mesh.glsl",
                 ShaderStage::Mesh, &shaders_[MeshletVelocityMesh]},
      ShaderSpec{meshletShaderDir / "opaque_meshlet_velocity.frag",
                 ShaderStage::Fragment, &shaders_[MeshletVelocityFragment]},
      ShaderSpec{meshletShaderDir / "opaque_meshlet_reactive_mask.mesh.glsl",
                 ShaderStage::Mesh, &shaders_[MeshletReactiveMaskMesh]},
      ShaderSpec{meshletShaderDir / "opaque_meshlet_reactive_mask.frag",
                 ShaderStage::Fragment, &shaders_[MeshletReactiveMaskFragment]},
  };
  for (const ShaderSpec &spec : shaderSpecs) {
    auto result = shader.compileFromFile(spec.path.string(), spec.stage);
    if (result.hasError()) {
      resetMeshletPipelineState();
      return Result<bool, std::string>::makeError(result.error());
    }
    *spec.handle = result.value();
  }
  MeshletPipelineDesc desc{};
  desc.taskShader = shaders_[MeshletTask];
  desc.meshShader = shaders_[MeshletMesh];
  desc.fragmentShader = shaders_[MeshletFragment];
  desc.colorFormats = {kFrameCompositionSceneColorFormat};
  desc.colorAttachmentCount = 1u;
  desc.depthFormat = kFrameCompositionDepthFormat;
  desc.cullMode = CullMode::Back;
  desc.polygonMode = PolygonMode::Fill;
  desc.numSamples = 1u;
  desc.prewarmRasterStates = kOpaqueDepthPreparedMainStates;
  MeshletPipelineDesc compactedDesc = desc;
  compactedDesc.taskShader = shaders_[MeshletCompactedTask];
  const GpuMultisampleCapabilities multisampleCapabilities =
      gpu_.getMultisampleCapabilities();
  const bool createMeshletMsaa4x = multisampleCapabilities.sample4Color &&
                                   multisampleCapabilities.sample4Depth;
  const bool createMeshletMsaa8x = multisampleCapabilities.sample8Color &&
                                   multisampleCapabilities.sample8Depth;
  MeshletPipelineDesc msaaDesc = desc;
  msaaDesc.numSamples = kMsaa4xSampleCount;
  MeshletPipelineDesc compactedMsaaDesc = compactedDesc;
  compactedMsaaDesc.numSamples = kMsaa4xSampleCount;
  MeshletPipelineDesc msaa8xDesc = desc;
  msaa8xDesc.numSamples = kMsaa8xSampleCount;
  MeshletPipelineDesc compactedMsaa8xDesc = compactedDesc;
  compactedMsaa8xDesc.numSamples = kMsaa8xSampleCount;
  MeshletPipelineDesc msaaAlphaDesc = msaaDesc;
  msaaAlphaDesc.alphaToCoverageEnabled =
      multisampleCapabilities.alphaToCoverage;
  msaaAlphaDesc.minSampleShading = 0.0f;
  MeshletPipelineDesc compactedMsaaAlphaDesc = compactedMsaaDesc;
  compactedMsaaAlphaDesc.alphaToCoverageEnabled =
      multisampleCapabilities.alphaToCoverage;
  compactedMsaaAlphaDesc.minSampleShading = 0.0f;
  MeshletPipelineDesc msaa8xAlphaDesc = msaa8xDesc;
  msaa8xAlphaDesc.alphaToCoverageEnabled =
      multisampleCapabilities.alphaToCoverage;
  msaa8xAlphaDesc.minSampleShading = 0.0f;
  MeshletPipelineDesc compactedMsaa8xAlphaDesc = compactedMsaa8xDesc;
  compactedMsaa8xAlphaDesc.alphaToCoverageEnabled =
      multisampleCapabilities.alphaToCoverage;
  compactedMsaa8xAlphaDesc.minSampleShading = 0.0f;
  MeshletPipelineDesc depthDesc = desc;
  depthDesc.fragmentShader = shaders_[MeshletDepthFragment];
  depthDesc.colorAttachmentCount = 0u;
  depthDesc.prewarmRasterStates = {};
  MeshletPipelineDesc depthAlphaDesc = depthDesc;
  depthAlphaDesc.fragmentShader = shaders_[MeshletDepthAlphaFragment];
  MeshletPipelineDesc msaaDepthDesc = depthDesc;
  msaaDepthDesc.numSamples = kMsaa4xSampleCount;
  MeshletPipelineDesc msaaDepthAlphaDesc = depthAlphaDesc;
  msaaDepthAlphaDesc.numSamples = kMsaa4xSampleCount;
  msaaDepthAlphaDesc.alphaToCoverageEnabled =
      multisampleCapabilities.alphaToCoverage;
  msaaDepthAlphaDesc.minSampleShading = 0.0f;
  MeshletPipelineDesc msaa8xDepthDesc = depthDesc;
  msaa8xDepthDesc.numSamples = kMsaa8xSampleCount;
  MeshletPipelineDesc msaa8xDepthAlphaDesc = depthAlphaDesc;
  msaa8xDepthAlphaDesc.numSamples = kMsaa8xSampleCount;
  msaa8xDepthAlphaDesc.alphaToCoverageEnabled =
      multisampleCapabilities.alphaToCoverage;
  msaa8xDepthAlphaDesc.minSampleShading = 0.0f;
  MeshletPipelineDesc simpleNormalDesc = desc;
  simpleNormalDesc.meshShader = shaders_[MeshletSimpleNormalMesh];
  simpleNormalDesc.fragmentShader = shaders_[MeshletSimpleNormalFragment];
  simpleNormalDesc.colorFormats = {kFrameCompositionNormalFormat};
  simpleNormalDesc.colorAttachmentCount = 1u;
  simpleNormalDesc.prewarmRasterStates = kOpaqueReadOnlyAuxiliaryStates;
  MeshletPipelineDesc normalDesc = desc;
  normalDesc.fragmentShader = shaders_[MeshletNormalFragment];
  normalDesc.colorFormats = {kFrameCompositionNormalFormat};
  normalDesc.colorAttachmentCount = 1u;
  normalDesc.prewarmRasterStates = kOpaqueReadOnlyAuxiliaryStates;
  MeshletPipelineDesc velocityDesc = desc;
  velocityDesc.meshShader = shaders_[MeshletVelocityMesh];
  velocityDesc.fragmentShader = shaders_[MeshletVelocityFragment];
  velocityDesc.colorFormats = {kFrameCompositionMotionVectorFormat};
  velocityDesc.colorAttachmentCount = 1u;
  velocityDesc.prewarmRasterStates = kOpaqueReadOnlyAuxiliaryStates;
  MeshletPipelineDesc reactiveMaskDesc = desc;
  reactiveMaskDesc.meshShader = shaders_[MeshletReactiveMaskMesh];
  reactiveMaskDesc.fragmentShader = shaders_[MeshletReactiveMaskFragment];
  reactiveMaskDesc.colorFormats = {kFrameCompositionReactiveMaskFormat};
  reactiveMaskDesc.colorAttachmentCount = 1u;
  reactiveMaskDesc.prewarmRasterStates = kOpaqueReadOnlyAuxiliaryStates;
  struct PipelineSpec {
    MeshletPipelineDesc desc;
    std::array<std::string_view, 2> names;
    MeshletPipelineHandle *handles;
    bool enabled = true;
  };
  std::array pipelineSpecs{
      PipelineSpec{desc,
                   {"opaque_meshlet", "opaque_meshlet_double_sided"},
                   meshletScenePipelines_.data()},
      PipelineSpec{
          compactedDesc,
          {"opaque_meshlet_compacted", "opaque_meshlet_compacted_double_sided"},
          meshletScenePipelines_.data() +
              meshletSceneVariantIndex(true, CoverageMode::Sample1, false,
                                       false)},
      PipelineSpec{
          msaaDesc,
          {"opaque_meshlet_msaa4x", "opaque_meshlet_msaa4x_double_sided"},
          meshletScenePipelines_.data() +
              meshletSceneVariantIndex(false, CoverageMode::Sample4, false,
                                       false),
          createMeshletMsaa4x},
      PipelineSpec{compactedMsaaDesc,
                   {"opaque_meshlet_compacted_msaa4x",
                    "opaque_meshlet_compacted_msaa4x_double_sided"},
                   meshletScenePipelines_.data() +
                       meshletSceneVariantIndex(true, CoverageMode::Sample4,
                                                false, false),
                   createMeshletMsaa4x},
      PipelineSpec{msaaAlphaDesc,
                   {"opaque_meshlet_alpha_msaa4x",
                    "opaque_meshlet_alpha_msaa4x_double_sided"},
                   meshletScenePipelines_.data() +
                       meshletSceneVariantIndex(false, CoverageMode::Sample4,
                                                true, false),
                   createMeshletMsaa4x},
      PipelineSpec{compactedMsaaAlphaDesc,
                   {"opaque_meshlet_compacted_alpha_msaa4x",
                    "opaque_meshlet_compacted_alpha_msaa4x_double_sided"},
                   meshletScenePipelines_.data() +
                       meshletSceneVariantIndex(true, CoverageMode::Sample4,
                                                true, false),
                   createMeshletMsaa4x},
      PipelineSpec{
          msaa8xDesc,
          {"opaque_meshlet_msaa8x", "opaque_meshlet_msaa8x_double_sided"},
          meshletScenePipelines_.data() +
              meshletSceneVariantIndex(false, CoverageMode::Sample8, false,
                                       false),
          createMeshletMsaa8x},
      PipelineSpec{compactedMsaa8xDesc,
                   {"opaque_meshlet_compacted_msaa8x",
                    "opaque_meshlet_compacted_msaa8x_double_sided"},
                   meshletScenePipelines_.data() +
                       meshletSceneVariantIndex(true, CoverageMode::Sample8,
                                                false, false),
                   createMeshletMsaa8x},
      PipelineSpec{msaa8xAlphaDesc,
                   {"opaque_meshlet_alpha_msaa8x",
                    "opaque_meshlet_alpha_msaa8x_double_sided"},
                   meshletScenePipelines_.data() +
                       meshletSceneVariantIndex(false, CoverageMode::Sample8,
                                                true, false),
                   createMeshletMsaa8x},
      PipelineSpec{compactedMsaa8xAlphaDesc,
                   {"opaque_meshlet_compacted_alpha_msaa8x",
                    "opaque_meshlet_compacted_alpha_msaa8x_double_sided"},
                   meshletScenePipelines_.data() +
                       meshletSceneVariantIndex(true, CoverageMode::Sample8,
                                                true, false),
                   createMeshletMsaa8x},
      PipelineSpec{
          depthDesc,
          {"opaque_meshlet_depth", "opaque_meshlet_depth_double_sided"},
          meshletDepthPipelines_.data()},
      PipelineSpec{depthAlphaDesc,
                   {"opaque_meshlet_depth_alpha",
                    "opaque_meshlet_depth_alpha_double_sided"},
                   meshletDepthPipelines_.data() + 2u},
      PipelineSpec{msaaDepthDesc,
                   {"opaque_meshlet_depth_msaa4x",
                    "opaque_meshlet_depth_msaa4x_double_sided"},
                   meshletDepthPipelines_.data() + 4u,
                   createMeshletMsaa4x},
      PipelineSpec{msaaDepthAlphaDesc,
                   {"opaque_meshlet_depth_alpha_msaa4x",
                    "opaque_meshlet_depth_alpha_msaa4x_double_sided"},
                   meshletDepthPipelines_.data() + 6u,
                   createMeshletMsaa4x},
      PipelineSpec{
          msaa8xDepthDesc,
          {"opaque_meshlet_depth_msaa8x",
           "opaque_meshlet_depth_msaa8x_double_sided"},
          meshletDepthPipelines_.data() +
              meshletDepthVariantIndex(CoverageMode::Sample8, false, false),
          createMeshletMsaa8x},
      PipelineSpec{
          msaa8xDepthAlphaDesc,
          {"opaque_meshlet_depth_alpha_msaa8x",
           "opaque_meshlet_depth_alpha_msaa8x_double_sided"},
          meshletDepthPipelines_.data() +
              meshletDepthVariantIndex(CoverageMode::Sample8, true, false),
          createMeshletMsaa8x},
      PipelineSpec{simpleNormalDesc,
                   {"opaque_meshlet_normal_simple",
                    "opaque_meshlet_normal_simple_double_sided"},
                   meshletNormalPipelines_.data()},
      PipelineSpec{normalDesc,
                   {"opaque_meshlet_normal_material",
                    "opaque_meshlet_normal_double_sided"},
                   meshletNormalPipelines_.data() + 2u},
      PipelineSpec{
          velocityDesc,
          {"opaque_meshlet_velocity", "opaque_meshlet_velocity_double_sided"},
          meshletVelocityPipelines_.data()},
      PipelineSpec{reactiveMaskDesc,
                   {"opaque_meshlet_reactive_mask",
                    "opaque_meshlet_reactive_mask_double_sided"},
                   meshletReactiveMaskPipelines_.data()},
  };
  for (PipelineSpec &spec : pipelineSpecs) {
    if (!spec.enabled) {
      continue;
    }
    for (size_t side = 0; side < 2u; ++side) {
      spec.desc.cullMode = side == 0 ? CullMode::Back : CullMode::None;
      auto result = gpu_.createMeshletPipeline(spec.desc, spec.names[side]);
      if (result.hasError()) {
        destroyMeshletPipelineState();
        return Result<bool, std::string>::makeError(result.error());
      }
      spec.handles[side] = result.value();
    }
  }
  meshletPipelineInitialized_ = true;
  return Result<bool, std::string>::makeResult(true);
}

RenderPipelineHandle
OpaqueRenderer::selectMeshPipeline(bool doubleSided, bool tessellated) const {
  const RenderPipelineHandle selected = meshScenePipelines_[rasterVariantIndex(
      CoverageMode::Sample1, false, tessellated, doubleSided)];
  if (nuri::isValid(selected)) {
    return selected;
  }
  return tessellated ? meshScenePipelines_[rasterVariantIndex(
                           CoverageMode::Sample1, false, true, false)]
                     : meshPipeline_.get();
}

RenderPipelineHandle OpaqueRenderer::selectVelocityPipeline(
    RenderPipelineHandle sourcePipeline) const {
  return selectSurfaceVariant(meshVelocityPipelines_,
                              isTessPipeline(sourcePipeline),
                              isDoubleSidedPipeline(sourcePipeline), false);
}

RenderPipelineHandle OpaqueRenderer::selectReactiveMaskPipeline(
    RenderPipelineHandle sourcePipeline) const {
  const bool doubleSided = isDoubleSidedPipeline(sourcePipeline);
  if (doubleSided && nuri::isValid(meshReactiveMaskPipelines_[1])) {
    return meshReactiveMaskPipelines_[1];
  }
  return meshReactiveMaskPipelines_[0];
}

RenderPipelineHandle OpaqueRenderer::selectNormalPipeline(
    RenderPipelineHandle sourcePipeline) const {
  return selectSurfaceVariant(meshNormalPipelines_,
                              isTessPipeline(sourcePipeline),
                              isDoubleSidedPipeline(sourcePipeline), false);
}

RenderPipelineHandle
OpaqueRenderer::selectDepthPipeline(RenderPipelineHandle sourcePipeline,
                                    bool alphaMasked,
                                    CoverageMode coverage) const {
  const bool tessellated = isTessPipeline(sourcePipeline);
  const bool doubleSided = isDoubleSidedPipeline(sourcePipeline);
  return meshDepthPipelines_[rasterVariantIndex(coverage, alphaMasked,
                                                tessellated, doubleSided)];
}

RenderPipelineHandle
OpaqueRenderer::selectPickPipeline(RenderPipelineHandle sourcePipeline) const {
  return selectSurfaceVariant(meshPickPipelines_,
                              isTessPipeline(sourcePipeline),
                              isDoubleSidedPipeline(sourcePipeline), true);
}

RenderPipelineHandle OpaqueRenderer::selectShadowInspectPipeline(
    RenderPipelineHandle sourcePipeline) const {
  return selectSurfaceVariant(meshShadowInspectPipelines_,
                              isTessPipeline(sourcePipeline),
                              isDoubleSidedPipeline(sourcePipeline), true);
}

RenderPipelineHandle
OpaqueRenderer::selectMsaaScenePipeline(RenderPipelineHandle sourcePipeline,
                                        bool alphaMasked,
                                        CoverageMode coverage) const {
  for (size_t variant = 0; variant < 4u; ++variant) {
    const bool tessellated = (variant & 1u) != 0u;
    const bool doubleSided = (variant & 2u) != 0u;
    const RenderPipelineHandle base =
        variant == 0u
            ? meshPipeline_.get()
            : meshScenePipelines_[rasterVariantIndex(
                  CoverageMode::Sample1, false, tessellated, doubleSided)];
    if (!isSamePipelineHandle(sourcePipeline, base)) {
      continue;
    }
    const RenderPipelineHandle alpha = meshScenePipelines_[rasterVariantIndex(
        coverage, true, tessellated, doubleSided)];
    return alphaMasked && nuri::isValid(alpha)
               ? alpha
               : meshScenePipelines_[rasterVariantIndex(
                     coverage, false, tessellated, doubleSided)];
  }
  for (size_t kind = 0; kind < static_cast<size_t>(OverlayPipelineKind::Count);
       ++kind) {
    if (isSamePipelineHandle(sourcePipeline, overlayPipelines_[kind])) {
      return overlayPipeline(static_cast<OverlayPipelineKind>(kind), coverage);
    }
  }
  return sourcePipeline;
}

MeshletPipelineHandle OpaqueRenderer::selectMeshletScenePipeline(
    bool compacted, CoverageMode coverage, bool alphaMasked,
    bool doubleSided) const {
  return meshletScenePipelines_[meshletSceneVariantIndex(
      compacted, coverage, alphaMasked, doubleSided)];
}

MeshletPipelineHandle OpaqueRenderer::selectMeshletDepthPipeline(
    CoverageMode coverage, bool alphaMasked, bool doubleSided) const {
  return meshletDepthPipelines_[meshletDepthVariantIndex(coverage, alphaMasked,
                                                         doubleSided)];
}

bool OpaqueRenderer::isDoubleSidedPipeline(RenderPipelineHandle handle) const {
  return matchesVariantBit(handle, meshScenePipelines_, 4u) ||
         matchesVariantBit(handle, meshNormalPipelines_, 2u);
}

bool OpaqueRenderer::isTessPipeline(RenderPipelineHandle handle) const {
  return matchesVariantBit(handle, meshScenePipelines_, 2u) ||
         matchesVariantBit(handle, meshNormalPipelines_, 1u);
}

RenderPipelineHandle
OpaqueRenderer::overlayPipeline(OverlayPipelineKind kind,
                                CoverageMode coverage) const noexcept {
  return overlayPipelines_[overlayPipelineIndex(kind, coverage)];
}

bool OpaqueRenderer::ensureOverlayPipeline(OverlayPipelineKind kind,
                                           CoverageMode coverage) {
  const size_t baseIndex = overlayPipelineIndex(kind, CoverageMode::Sample1);
  const size_t requestedIndex = overlayPipelineIndex(kind, coverage);
  RenderPipelineHandle &requestedPipeline = overlayPipelines_[requestedIndex];
  bool &requestedUnsupported = overlayPipelineUnsupported_[requestedIndex];
  if (nuri::isValid(requestedPipeline)) {
    return true;
  }
  if (overlayPipelineUnsupported_[baseIndex] || requestedUnsupported) {
    return false;
  }
  const bool tessellated = kind == OverlayPipelineKind::TessWireframe ||
                           kind == OverlayPipelineKind::TessGeometry;
  const bool geometry = kind == OverlayPipelineKind::Geometry ||
                        kind == OverlayPipelineKind::TessGeometry;
  if (tessellated && !nuri::isValid(meshScenePipelines_[rasterVariantIndex(
                         CoverageMode::Sample1, false, true, false)])) {
    overlayPipelineUnsupported_[baseIndex] = true;
    return false;
  }
  if (geometry && (!nuri::isValid(shaders_[MeshDebugOverlayGeometry]) ||
                   !nuri::isValid(shaders_[MeshDebugOverlayFragment]))) {
    overlayPipelineUnsupported_[baseIndex] = true;
    return false;
  }
  RenderPipelineDesc desc = meshPipelineDesc(
      kFrameCompositionSceneColorFormat, kFrameCompositionDepthFormat,
      tessellated ? shaders_[MeshTessVertex] : shaders_[MeshVertex],
      tessellated ? shaders_[MeshTessControl] : ShaderHandle{},
      tessellated ? shaders_[MeshTessEval] : ShaderHandle{},
      geometry ? shaders_[MeshDebugOverlayGeometry] : ShaderHandle{},
      geometry ? shaders_[MeshDebugOverlayFragment] : shaders_[MeshFragment],
      geometry ? PolygonMode::Fill : PolygonMode::Line,
      tessellated ? Topology::Patch : Topology::Triangle,
      tessellated ? kTessellationPatchControlPoints : 0u, true);
  desc.rasterState = kOpaqueOverlayRasterState;
  constexpr std::array names{
      std::array<std::string_view, 3>{"opaque_mesh_wireframe",
                                      "opaque_mesh_wireframe_msaa4x",
                                      "opaque_mesh_wireframe_msaa8x"},
      std::array<std::string_view, 3>{"opaque_mesh_tess_wireframe",
                                      "opaque_mesh_tess_wireframe_msaa4x",
                                      "opaque_mesh_tess_wireframe_msaa8x"},
      std::array<std::string_view, 3>{"opaque_mesh_overlay_gs",
                                      "opaque_mesh_overlay_gs_msaa4x",
                                      "opaque_mesh_overlay_gs_msaa8x"},
      std::array<std::string_view, 3>{"opaque_mesh_tess_overlay_gs",
                                      "opaque_mesh_tess_overlay_gs_msaa4x",
                                      "opaque_mesh_tess_overlay_gs_msaa8x"},
  };
  const size_t kindIndex = static_cast<size_t>(kind);
  const auto create = [&](CoverageMode variantCoverage) {
    const size_t index = overlayPipelineIndex(kind, variantCoverage);
    RenderPipelineHandle &pipeline = overlayPipelines_[index];
    bool &unsupported = overlayPipelineUnsupported_[index];
    if (nuri::isValid(pipeline)) {
      return true;
    }
    RenderPipelineDesc variant = desc;
    variant.numSamples = coverageSampleCount(variantCoverage);
    auto result = gpu_.createRenderPipeline(
        variant, names[kindIndex][coverageModeIndex(variantCoverage)]);
    if (!result.hasError()) {
      pipeline = result.value();
      return true;
    }
    unsupported = true;
    return false;
  };
  return create(CoverageMode::Sample1) && create(coverage);
}

void OpaqueRenderer::resetOverlayPipelineState() {
  for (RenderPipelineHandle &pipeline : overlayPipelines_) {
    destroyPipelineHandle(gpu_, pipeline);
  }
  overlayPipelineUnsupported_.fill(false);
}

void OpaqueRenderer::invalidateAutoLodHistory() {
  autoLodHistoryValid_ = false;
  autoLodWasActive_ = false;
}

void OpaqueRenderer::invalidateStaticBatchCache() {
  staticBatchCache_.valid = false;
  staticBatchCache_.meshletRequested = false;
  staticBatchCache_.meshletDispatchCacheValid = false;
  staticBatchCache_.enableMeshLod = false;
  staticBatchCache_.forcedMeshLod = -1;
  ++staticBatchCache_.generation;
  if (staticBatchCache_.generation == 0) {
    staticBatchCache_.generation = 1;
  }
  staticBatchCache_.remapSignature = kInvalidDrawSignature;
  staticBatchCache_.indirectDrawSignature = kInvalidDrawSignature;
  staticBatchCache_.drawBufferSignature = kInvalidDrawSignature;
  staticBatchCache_.meshletDispatchSignature = kInvalidDrawSignature;
  staticBatchCache_.meshletCandidateCount = 0u;
  staticBatchCache_.draws.clear();
  staticBatchCache_.pushConstantsTemplates.clear();
  staticBatchCache_.alphaMasked.clear();
  staticBatchCache_.meshletBatchInfos.clear();
  staticBatchCache_.meshletDispatches.clear();
  staticBatchCache_.meshletPushConstantsTemplates.clear();
  staticBatchCache_.meshletDispatchDependencyBuffers.clear();
  staticBatchCache_.meshletBatchGpuData.clear();
  staticBatchCache_.remap.clear();
  boundStaticBatchGeneration_ = 0;
  cachedVisibleBatchValid_ = false;
  cachedVisibleBatchTopologyVersion_ = std::numeric_limits<uint64_t>::max();
  cachedVisibleBatchMaterialVersion_ = std::numeric_limits<uint64_t>::max();
  cachedVisibleBatchGeometryVersion_ = std::numeric_limits<uint64_t>::max();
  cachedVisibleTemplateBatchIndices_.clear();
  visibleBatchActiveRemap_.clear();
  cachedVisibleBatchEntries_.clear();
}

void OpaqueRenderer::invalidateSingleInstanceBatchCache() {
  ++singleInstanceTemplateRevision_;
  if (singleInstanceTemplateRevision_ == 0) {
    singleInstanceTemplateRevision_ = 1;
  }
  for (SingleInstanceBatchCache &cache : singleInstanceBatchCaches_) {
    cache.valid = false;
    cache.requestedLod = 0;
    cache.tessPipelineEnabled = false;
    cache.basePipeline = {};
    cache.doubleSidedBasePipeline = {};
    cache.tessPipeline = {};
    cache.doubleSidedTessPipeline = {};
    cache.templateRevision = 0;
    cache.remapCount = 0;
    cache.batches.clear();
  }
}

void OpaqueRenderer::invalidateIndirectPackCache() {
  indirectPackCache_.valid = false;
  indirectPackCache_.drawSignature = kInvalidDrawSignature;
  indirectPackCache_.requiredBytes = 0;
  currentIndirectDrawBufferSignature_ = kInvalidDrawSignature;
  indirectSourceDrawIndices_.clear();
  indirectAlphaMasked_.clear();
  for (FrameSlotState &slot : frameSlotStates_) {
    slot.indirectUploadSignature = kInvalidDrawSignature;
  }
}

void OpaqueRenderer::destroyMeshPipelineState() {
  const auto destroy = [this](auto &handles) {
    for (RenderPipelineHandle &handle : handles) {
      destroyPipelineHandle(gpu_, handle);
    }
  };
  destroy(meshScenePipelines_);
  destroy(meshPickPipelines_);
  destroy(meshShadowInspectPipelines_);
  destroy(meshVelocityPipelines_);
  destroy(meshReactiveMaskPipelines_);
  destroy(meshNormalPipelines_);
  destroy(meshDepthPipelines_);
  for (RenderPipelineHandle *handle :
       {&depthMotionVectorPipelineHandle_,
        &currentFrameDepthVerificationPipelineHandle_,
        &depthPyramidPipelineHandle_}) {
    destroyPipelineHandle(gpu_, *handle);
  }
  resetMeshPipelineState();
}

void OpaqueRenderer::destroyMeshletPipelineState() {
  const auto destroy = [this](auto &pipelines) {
    for (MeshletPipelineHandle &pipeline : pipelines) {
      destroyMeshletPipelineHandle(gpu_, pipeline);
    }
  };
  destroy(meshletScenePipelines_);
  destroy(meshletDepthPipelines_);
  destroy(meshletNormalPipelines_);
  destroy(meshletVelocityPipelines_);
  destroy(meshletReactiveMaskPipelines_);
  resetMeshletPipelineState();
}

void OpaqueRenderer::resetMeshletPipelineState() {
  meshletScenePipelines_.fill({});
  meshletDepthPipelines_.fill({});
  meshletNormalPipelines_.fill({});
  meshletVelocityPipelines_.fill({});
  meshletReactiveMaskPipelines_.fill({});
  shaders_[MeshletTask] = {};
  shaders_[MeshletCompactedTask] = {};
  shaders_[MeshletMesh] = {};
  shaders_[MeshletFragment] = {};
  shaders_[MeshletDepthFragment] = {};
  shaders_[MeshletDepthAlphaFragment] = {};
  shaders_[MeshletSimpleNormalMesh] = {};
  shaders_[MeshletSimpleNormalFragment] = {};
  shaders_[MeshletNormalFragment] = {};
  shaders_[MeshletVelocityMesh] = {};
  shaders_[MeshletVelocityFragment] = {};
  shaders_[MeshletReactiveMaskMesh] = {};
  shaders_[MeshletReactiveMaskFragment] = {};
  meshletPipelineInitialized_ = false;
}

void OpaqueRenderer::resetMeshPipelineState() {
  meshScenePipelines_.fill({});
  meshPickPipelines_.fill({});
  meshShadowInspectPipelines_.fill({});
  meshVelocityPipelines_.fill({});
  meshReactiveMaskPipelines_.fill({});
  meshNormalPipelines_.fill({});
  meshDepthPipelines_.fill({});
  currentFrameDepthVerificationPipelineHandle_ = {};
  depthPyramidPipelineHandle_ = {};
  depthMotionVectorPipelineHandle_ = {};
  depthMotionVectorDrawItem_ = {};
  baseMeshFillDraw_ = {};
}

void OpaqueRenderer::destroyDepthPyramidTextures() {
  if (nuri::isValid(currentFrameDepthVerificationTexture_)) {
    gpu_.destroyTexture(currentFrameDepthVerificationTexture_);
    currentFrameDepthVerificationTexture_ = {};
  }
  for (TextureHandle &texture : sceneDepthPyramidTextures_) {
    if (nuri::isValid(texture)) {
      gpu_.destroyTexture(texture);
      texture = {};
    }
  }
  sceneDepthPyramidLevelCount_ = 0;
  sceneDepthPyramidWidth_ = 0;
  sceneDepthPyramidHeight_ = 0;
  sceneDepthPyramidSourceFrameIndex_.reset();
  sceneDepthPyramidSourceViewProj_.reset();
}

void OpaqueRenderer::destroyPickTexture() {
  if (nuri::isValid(pickIdTexture_)) {
    gpu_.destroyTexture(pickIdTexture_);
    pickIdTexture_ = TextureHandle{};
  }
}

void OpaqueRenderer::destroyTransmissionVisibilityDepthTexture() {
  if (nuri::isValid(transmissionVisibilityDepthTexture_)) {
    gpu_.destroyTexture(transmissionVisibilityDepthTexture_);
    transmissionVisibilityDepthTexture_ = TextureHandle{};
  }
}

void OpaqueRenderer::destroyShadowInspectTexture() {
  if (nuri::isValid(shadowInspectTexture_)) {
    gpu_.destroyTexture(shadowInspectTexture_);
    shadowInspectTexture_ = TextureHandle{};
  }
}

void OpaqueRenderer::destroyBuffers() {
  retireDynamicBuffer(instanceCentersPhaseBuffer_,
                      instanceCentersPhaseBufferCapacityBytes_);
  retireDynamicBuffer(instanceLodBoundsBuffer_,
                      instanceLodBoundsBufferCapacityBytes_);
  retireDynamicBuffer(instanceBaseMatricesBuffer_,
                      instanceBaseMatricesBufferCapacityBytes_);
  for (auto &ring : bufferRings_) {
    retireDynamicBufferRing(ring);
    ring.clear();
  }
  frameSlotStates_.clear();
  visibilityVisibleIndexReadback_.clear();
  invalidateIndirectPackCache();
}

namespace {
template <auto Ready, auto Build>
void addOpaqueStage(RenderPipeline &pipeline, std::string_view componentName,
                    std::string_view name, OpaqueRenderer &renderer) {
  pipeline.addStage(PipelineStageDesc{
      .componentName = componentName,
      .name = name,
      .state = &renderer,
      .enabled =
          [](const void *state, const FrameBuildContext &ctx) {
            return (!ctx.frame.settings ||
                    renderSettingsOrDefault(ctx.frame).opaque.enabled) &&
                   (static_cast<const OpaqueRenderer *>(state)->*Ready)();
          },
      .build =
          [](void *state, FrameBuildContext &ctx) {
            return (static_cast<OpaqueRenderer *>(state)->*Build)(ctx.frame,
                                                                  ctx.graph);
          },
  });
}
} // namespace

OpaqueRenderer *registerOpaquePrepassStages(RenderPipeline &pipeline,
                                            GPUDevice &gpu,
                                            RuntimeOpaqueShaderConfig config,
                                            std::pmr::memory_resource *memory,
                                            SceneDrawDatabase *database) {
  if (!database) {
    database =
        pipeline.addProvider(std::make_unique<SceneDrawDatabase>(gpu, memory));
  }
  auto owner = std::make_unique<OpaqueRenderer>(gpu, std::move(config), memory);
  owner->onAttach();
  auto *renderer = pipeline.addComponent(
      std::move(owner),
      PipelineComponentDesc{
          .publish =
              [](void *state, FrameBuildContext &ctx) {
                static_cast<OpaqueRenderer *>(state)->publishFrameData(
                    ctx.frame);
                return Result<bool, std::string>::makeResult(true);
              },
          .prepare =
              [](void *state, FrameBuildContext &ctx) {
                return static_cast<OpaqueRenderer *>(state)
                    ->prepareOpaqueGraphPasses(ctx.frame);
              },
          .prepareScene =
              [](void *state, RenderScenePreparationContext &ctx) {
                return static_cast<OpaqueRenderer *>(state)->prepareSceneCache(
                    *ctx.sceneDrawDatabase, ctx.scene, ctx.resources);
              },
          .submitted =
              [](void *state, const RenderFrameContext &frame) noexcept {
                static_cast<OpaqueRenderer *>(state)->commitSubmittedFrame(
                    frame.frameIndex);
              },
          .abandoned =
              [](void *state, const RenderFrameContext &frame) noexcept {
                static_cast<OpaqueRenderer *>(state)->abandonPreparedFrame(
                    frame.frameIndex);
              },
      });
  addOpaqueStage<&OpaqueRenderer::hasPreparedOpaquePickPasses,
                 &OpaqueRenderer::appendOpaquePickPasses>(
      pipeline, "OpaquePrepassFeature", "OpaquePickPass", *renderer);
  addOpaqueStage<&OpaqueRenderer::hasPreparedOpaquePrepassPasses,
                 &OpaqueRenderer::appendOpaquePrepassPasses>(
      pipeline, "OpaquePrepassFeature", "OpaquePrepassPass", *renderer);
  return renderer;
}

void registerOpaqueMainStage(RenderPipeline &pipeline,
                             OpaqueRenderer &renderer) {
  addOpaqueStage<&OpaqueRenderer::hasPreparedOpaqueMainLightingPasses,
                 &OpaqueRenderer::appendOpaqueMainLightingPasses>(
      pipeline, "OpaqueMainFeature", "OpaqueMainLightingPass", renderer);
}

} // namespace nuri
