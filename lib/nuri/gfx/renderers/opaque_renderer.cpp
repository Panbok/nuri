#include "nuri/pch.h"

#include "nuri/gfx/renderers/opaque_renderer.h"

#include "nuri/core/containers/hash_map.h"
#include "nuri/core/containers/hash_set.h"
#include "nuri/core/log.h"
#include "nuri/core/pmr_scratch.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/renderers/detail/renderable_material_resolution.h"
#include "nuri/gfx/renderers/detail/shadow_math.h"
#include "nuri/gfx/visibility/visibility.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/scene/render_scene.h"

#include <cmath>

namespace nuri {
namespace {
constexpr float kMinLodRadius = 1.0e-4f;
constexpr float kAutoLodCameraReuseEpsilon = 2.5e-2f;
constexpr float kAutoLodThresholdReuseEpsilon = 1.0e-4f;
constexpr size_t kAutoLodTemporalReuseMinInstances = 4096u;
constexpr uint64_t kAutoLodTemporalReuseFrameInterval = 2ull;
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
constexpr uint32_t kTessellationPatchControlPoints = 3;
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
constexpr uint32_t kMeshletCounterFlagEnabled = 1u << 0u;
// Phase hash: normalize 24-bit hash to [0, 1] then scale to [0, 2*pi]
constexpr uint32_t kPhaseHashMask = 0x00ffffffu;
constexpr float kPhaseNormDivisor = 16777215.0f; // 2^24 - 1
constexpr uint32_t kPhaseHashMixMultiplier = 2246822519u;
constexpr uint32_t kPhaseHashShift1 = 16u;
constexpr uint32_t kPhaseHashShift2 = 13u;
constexpr uint64_t kFnvOffsetBasis64 = 14695981039346656037ull;
constexpr uint64_t kFnvPrime64 = 1099511628211ull;
constexpr uint64_t kInvalidDrawSignature = std::numeric_limits<uint64_t>::max();
constexpr std::string_view kOpaquePickPassLabel = "Opaque Pick Pass";
constexpr std::string_view kOpaqueMainPassLabel = "Opaque Pass";
uint64_t hashCombine64(uint64_t hash, uint64_t value) {
  hash ^= value;
  hash *= kFnvPrime64;
  return hash;
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

void logOpaqueVisibilityCounters(const RenderFrameContext &frame) {
  if (frame.settings == nullptr ||
      !frame.settings->visibility.debug.logCounters) {
    return;
  }

  const VisibilityFrameMetrics &visibility = frame.metrics.visibility;
  NURI_LOG_INFO(
      "OpaqueRenderer::visibility counters frame=%llu "
      "cpu(main=%u visible=%u rejected=%u uncertain=%u) "
      "gpu(main=%u visible=%u frustum=%u occlusion=%u overflow=%u "
      "readback=%u source=%u stale=%u errors=%u readbackVisible=%u "
      "mismatches=%u) "
      "indirect(drawUsed=%u drawFallback=%u drawCommands=%u meshDispatch=%u) "
      "meshlet(candidates=%u emitted=%u taskGroups=%u frustum=%u cone=%u "
      "occlusion=%u occlusionAvailable=%u payloadOverflow=%u readback=%u "
      "source=%u stale=%u errors=%u) occlusionAvailable=%u",
      static_cast<unsigned long long>(frame.frameIndex),
      visibility.cpuMainCandidates, visibility.cpuMainVisibleCandidates,
      visibility.cpuMainRejected, visibility.uncertainVisible,
      visibility.gpuMainCandidates, visibility.gpuMainVisibleCandidates,
      visibility.gpuMainRejectedFrustum, visibility.gpuMainRejectedOcclusion,
      visibility.gpuOutputOverflowCount, visibility.gpuMainReadbackAvailable,
      visibility.gpuMainReadbackSourceFrame,
      visibility.gpuMainReadbackStaleFrameCount,
      visibility.gpuMainReadbackErrorCount,
      visibility.gpuMainReadbackVisibleCandidates,
      visibility.gpuMainVisibleListMismatches, visibility.gpuIndirectDrawUsed,
      visibility.gpuIndirectDrawFallback, visibility.gpuIndirectDrawCommands,
      visibility.indirectMeshDispatchCount, visibility.meshletCandidates,
      visibility.meshletEmitted, visibility.meshletTaskGroupsExecuted,
      visibility.meshletRejectedFrustum, visibility.meshletRejectedCone,
      visibility.meshletRejectedOcclusion, visibility.meshletOcclusionAvailable,
      visibility.meshletPayloadOverflowCount,
      visibility.meshletReadbackAvailable,
      visibility.meshletReadbackSourceFrame,
      visibility.meshletReadbackStaleFrameCount,
      visibility.meshletReadbackErrorCount, visibility.occlusionAvailable);
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

Result<std::pmr::vector<RenderGraphBufferId>, std::string>
importPreResolvedBuffers(RenderGraphBuilder &graph,
                         std::span<const BufferHandle> buffers,
                         std::pmr::memory_resource *memory,
                         std::string_view debugName) {
  std::pmr::vector<RenderGraphBufferId> bufferIds(
      resolveMemoryResource(memory));
  bufferIds.reserve(buffers.size());
  for (const BufferHandle handle : buffers) {
    if (!nuri::isValid(handle)) {
      continue;
    }
    auto importResult = graph.importBuffer(handle, debugName);
    if (importResult.hasError()) {
      return Result<std::pmr::vector<RenderGraphBufferId>,
                    std::string>::makeError(importResult.error());
    }
    bufferIds.push_back(importResult.value());
  }
  return Result<std::pmr::vector<RenderGraphBufferId>, std::string>::makeResult(
      std::move(bufferIds));
}

const RenderSettings &settingsOrDefault(const RenderFrameContext &frame) {
  static const RenderSettings kDefaultSettings{};
  return frame.settings ? *frame.settings : kDefaultSettings;
}

void publishRequestedCapture(RenderFrameContext &frame, GPUDevice &gpu,
                             std::string_view name, TextureHandle texture,
                             RenderCaptureValueKind kind,
                             RenderCaptureLifetimeClass lifetime,
                             std::string_view colorSpace,
                             std::string_view compareProfile,
                             std::string_view producerPassLabel) {
  if (!isRenderCaptureRequested(frame, name) || !nuri::isValid(texture)) {
    return;
  }
  frame.captureRegistry.publish(RenderCapturePoint{
      .name = name,
      .version = 1u,
      .texture = texture,
      .format = gpu.getTextureFormat(texture),
      .dimensions = gpu.getTextureDimensions(texture),
      .frameIndex = frame.frameIndex,
      .mip = 0u,
      .layer = 0u,
      .kind = kind,
      .lifetime = lifetime,
      .colorSpace = colorSpace,
      .defaultCompareProfile = compareProfile,
      .producerPassLabel = producerPassLabel,
      .debugLabel = name,
  });
}

const AnimationSceneFrameData *
resolveAnimationSceneFrameData(const RenderFrameContext &frame) {
  if (!frame.sharedResources.animationSceneGpuData.has_value()) {
    return nullptr;
  }
  const AnimationSceneFrameData &data =
      *frame.sharedResources.animationSceneGpuData;
  if (!nuri::isValid(data.instanceMatricesBuffer) ||
      data.instanceMatricesAddress == 0u) {
    return nullptr;
  }
  if (frame.scene == nullptr || data.scene != frame.scene ||
      data.sceneTopologyVersion != frame.scene->topologyVersion() ||
      data.renderableCount != frame.scene->renderables().size() ||
      data.geometryOverridesByRenderable.size() != data.renderableCount) {
    return nullptr;
  }
  return &data;
}

bool animationSceneAnimatesRenderable(
    const AnimationSceneFrameData &animationSceneData,
    size_t runtimeRenderableIndex) noexcept {
  for (const uint32_t animatedIndex :
       animationSceneData.animatedRenderableIndices) {
    if (animatedIndex == runtimeRenderableIndex) {
      return true;
    }
  }
  if (runtimeRenderableIndex >=
      animationSceneData.geometryOverridesByRenderable.size()) {
    return false;
  }
  const AnimatedRenderableGeometryOverride &geometryOverride =
      animationSceneData.geometryOverridesByRenderable[runtimeRenderableIndex];
  return geometryOverride.enabled &&
         nuri::isValid(geometryOverride.vertexBuffer);
}

bool animationOverrideCoversSubmesh(
    const AnimatedRenderableGeometryOverride &geometryOverride,
    const Submesh &submesh) noexcept {
  const uint64_t requiredVertexCount =
      static_cast<uint64_t>(submesh.vertexOffset) + submesh.vertexCount;
  return static_cast<uint64_t>(geometryOverride.vertexCount) >=
         requiredVertexCount;
}

[[nodiscard]] bool isTransmissionMaterial(const MaterialRecord &material) {
  return material.desc.alphaMode != MaterialAlphaMode::Blend &&
         (material.desc.featureMask & kMaterialFeatureTransmission) != 0u;
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

RenderPipelineDesc fullscreenPipelineDesc(Format colorFormat,
                                          ShaderHandle vertexShader,
                                          ShaderHandle fragmentShader) {
  return RenderPipelineDesc{
      .vertexInput = {},
      .vertexShader = vertexShader,
      .fragmentShader = fragmentShader,
      .colorFormats = {colorFormat},
      .depthFormat = Format::Count,
      .cullMode = CullMode::None,
      .polygonMode = PolygonMode::Fill,
      .topology = Topology::Triangle,
      .blendEnabled = false,
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

bool isSameGeometryAllocationHandle(GeometryAllocationHandle lhs,
                                    GeometryAllocationHandle rhs) {
  return lhs.index == rhs.index && lhs.generation == rhs.generation;
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

uint64_t textureBytesPerPixel(Format format) {
  switch (format) {
  case Format::R8_UNORM:
    return 1u;
  case Format::RG16_FLOAT:
    return sizeof(uint16_t) * 2u;
  case Format::RG32_FLOAT:
    return sizeof(float) * 2u;
  case Format::R32_UINT:
  case Format::R32_FLOAT:
  case Format::D32_FLOAT:
    return sizeof(uint32_t);
  case Format::D16_UNORM:
    return sizeof(uint16_t);
  case Format::RGBA8_UNORM:
  case Format::RGBA8_SRGB:
  case Format::RGBA8_UINT:
    return 4u;
  case Format::RGBA16_FLOAT:
    return 8u;
  case Format::RGBA32_FLOAT:
    return 16u;
  case Format::BC7_RGBA_UNORM:
  case Format::BC7_RGBA_SRGB:
  case Format::ETC2_RGB8_UNORM:
  case Format::ETC2_RGB8_SRGB:
  case Format::Count:
    break;
  }
  return 0u;
}

uint64_t textureStorageBytes(GPUDevice &gpu, TextureHandle texture) {
  if (!nuri::isValid(texture)) {
    return 0u;
  }
  const TextureDimensions dimensions = gpu.getTextureDimensions(texture);
  return static_cast<uint64_t>(std::max(dimensions.width, 1u)) *
         static_cast<uint64_t>(std::max(dimensions.height, 1u)) *
         textureBytesPerPixel(gpu.getTextureFormat(texture));
}

bool nearlyEqualThresholds(const std::array<float, 3> &a,
                           const std::array<float, 3> &b, float epsilon) {
  return std::abs(a[0] - b[0]) <= epsilon && std::abs(a[1] - b[1]) <= epsilon &&
         std::abs(a[2] - b[2]) <= epsilon;
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

Result<bool, std::string>
appendUniqueDependency(std::pmr::vector<BufferHandle> &dependencies,
                       std::pmr::vector<RenderGraphAccessMode> &accessModes,
                       BufferHandle handle, RenderGraphAccessMode accessMode,
                       std::string_view context) {
  if (!nuri::isValid(handle)) {
    return Result<bool, std::string>::makeResult(true);
  }
  for (size_t i = 0; i < dependencies.size(); ++i) {
    if (isSameBufferHandle(dependencies[i], handle)) {
      if (i < accessModes.size()) {
        accessModes[i] = accessModes[i] | accessMode;
      }
      return Result<bool, std::string>::makeResult(true);
    }
  }
  if (dependencies.size() >= kMaxDependencyResources) {
    return Result<bool, std::string>::makeError(
        std::string(context) + ": dependency buffer count exceeds " +
        std::to_string(kMaxDependencyResources));
  }
  dependencies.push_back(handle);
  accessModes.push_back(accessMode);
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
appendUniqueDependency(std::pmr::vector<BufferHandle> &dependencies,
                       BufferHandle handle, std::string_view context) {
  if (!nuri::isValid(handle)) {
    return Result<bool, std::string>::makeResult(true);
  }
  for (const BufferHandle existing : dependencies) {
    if (isSameBufferHandle(existing, handle)) {
      return Result<bool, std::string>::makeResult(true);
    }
  }
  if (dependencies.size() >= kMaxDependencyResources) {
    return Result<bool, std::string>::makeError(
        std::string(context) + ": dependency buffer count exceeds " +
        std::to_string(kMaxDependencyResources));
  }
  dependencies.push_back(handle);
  return Result<bool, std::string>::makeResult(true);
}

void appendUniqueDrawBuffer(std::pmr::vector<BufferHandle> &dependencies,
                            BufferHandle handle) {
  if (!nuri::isValid(handle)) {
    return;
  }
  for (const BufferHandle existing : dependencies) {
    if (isSameBufferHandle(existing, handle)) {
      return;
    }
  }
  dependencies.push_back(handle);
}

void appendUniqueTextureDependency(
    std::pmr::vector<TextureHandle> &dependencies, TextureHandle handle) {
  if (!nuri::isValid(handle)) {
    return;
  }
  for (const TextureHandle existing : dependencies) {
    if (isSameTextureHandle(existing, handle)) {
      return;
    }
  }
  dependencies.push_back(handle);
}

uint32_t saturateToU32(size_t value) {
  return static_cast<uint32_t>(
      std::min(value, size_t(std::numeric_limits<uint32_t>::max())));
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
      instanceMatricesRing_(resolveMemoryResource(memory)),
      previousInstanceMatricesRing_(resolveMemoryResource(memory)),
      velocityInstanceFlagsRing_(resolveMemoryResource(memory)),
      velocityFrameDataRing_(resolveMemoryResource(memory)),
      velocityGeometryRing_(resolveMemoryResource(memory)),
      instanceRemapRing_(resolveMemoryResource(memory)),
      indirectCommandRing_(resolveMemoryResource(memory)),
      meshletBatchRing_(resolveMemoryResource(memory)),
      visibilityCandidateRing_(resolveMemoryResource(memory)),
      visibilityPassRing_(resolveMemoryResource(memory)),
      visibilityVisibleIndexRing_(resolveMemoryResource(memory)),
      visibilityCounterRing_(resolveMemoryResource(memory)),
      visibilityMeshletDispatchRing_(resolveMemoryResource(memory)),
      visibilityMeshletIndirectCommandRing_(resolveMemoryResource(memory)),
      singleInstanceBatchCaches_(resolveMemoryResource(memory)),
      staticBatchCache_(resolveMemoryResource(memory)),
      renderableTemplates_(resolveMemoryResource(memory)),
      meshDrawTemplates_(resolveMemoryResource(memory)),
      indirectSourceDrawIndices_(resolveMemoryResource(memory)),
      instanceMatricesUploadVersions_(resolveMemoryResource(memory)),
      indirectUploadSignatures_(resolveMemoryResource(memory)),
      remapUploadSignatures_(resolveMemoryResource(memory)),
      visibilityCounterRingPublishedFrames_(resolveMemoryResource(memory)),
      visibilityExpectedVisibleIndexCounts_(resolveMemoryResource(memory)),
      visibilityExpectedVisibleIndexHashes_(resolveMemoryResource(memory)),
      visibilityVisibleIndexReadback_(resolveMemoryResource(memory)),
      templateBatchIndices_(resolveMemoryResource(memory)),
      batchWriteOffsets_(resolveMemoryResource(memory)),
      instanceCentersPhase_(resolveMemoryResource(memory)),
      instanceBaseMatrices_(resolveMemoryResource(memory)),
      instanceMatricesCpuCache_(resolveMemoryResource(memory)),
      instanceLodCentersInvRadiusSq_(resolveMemoryResource(memory)),
      materialTextureAccessHandles_(resolveMemoryResource(memory)),
      instanceAutoLodLevels_(resolveMemoryResource(memory)),
      instanceTessSelection_(resolveMemoryResource(memory)),
      tessCandidates_(resolveMemoryResource(memory)),
      instanceRemap_(resolveMemoryResource(memory)),
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
      shadowSdsmHistogramReducePushConstants_(resolveMemoryResource(memory)),
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
      visibilityMeshletGpuDependencyBuffers_(resolveMemoryResource(memory)),
      visibilityMeshletGpuDependencyBufferAccessModes_(
          resolveMemoryResource(memory)),
      visibilityGpuDependencyBuffers_(resolveMemoryResource(memory)),
      visibilityGpuDependencyBufferAccessModes_(resolveMemoryResource(memory)),
      visibilityGpuDependencyTextures_(resolveMemoryResource(memory)),
      passDependencyBuffers_(resolveMemoryResource(memory)),
      passDependencyBufferAccessModes_(resolveMemoryResource(memory)),
      preResolvedDecodeBuffers_(resolveMemoryResource(memory)),
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
      previousTransformById_(resolveMemoryResource(memory)),
      previousInstanceMatricesCpuCache_(resolveMemoryResource(memory)),
      velocityInstanceFlagsCpuCache_(resolveMemoryResource(memory)),
      velocityGeometryCpuCache_(resolveMemoryResource(memory)),
      transmissionVisibilityDepthPushConstants_(resolveMemoryResource(memory)),
      preparedGraphPasses_(resolveMemoryResource(memory)) {
  auto *resource = resolveMemoryResource(memory);
  singleInstanceBatchCaches_.reserve(kSingleInstanceCacheVariantCount);
  for (size_t i = 0; i < kSingleInstanceCacheVariantCount; ++i) {
    singleInstanceBatchCaches_.emplace_back(resource);
  }
}

OpaqueRenderer::~OpaqueRenderer() { onDetach(); }

void OpaqueRenderer::onAttach() {
  auto initResult = ensureInitialized();
  if (initResult.hasError()) {
    NURI_LOG_WARNING("OpaqueRenderer::onAttach: %s",
                     initResult.error().c_str());
  }
}

void OpaqueRenderer::resetPickState() {
  pendingPickRequest_.reset();
  pendingShadowInspectRequest_.reset();
  inFlightPickReadback_.reset();
  inFlightShadowInspectReadback_.reset();
}

void OpaqueRenderer::capturePreviousTransforms(const RenderScene &scene,
                                               uint64_t frameIndex) {
  if (previousTransformCaptureFrameIndex_ == frameIndex &&
      previousTransformSceneId_ == scene.id()) {
    return;
  }

  const uint64_t topologyVersion = scene.topologyVersion();
  const uint64_t transformVersion = scene.transformVersion();
  if (previousTransformSceneId_ == scene.id() &&
      previousTransformCaptureTopologyVersion_ == topologyVersion &&
      previousTransformCaptureTransformVersion_ == transformVersion) {
    previousTransformCaptureFrameIndex_ = frameIndex;
    return;
  }

  const std::span<const Renderable> renderables = scene.renderables();
  previousTransformById_.clear();
  previousTransformById_.reserve(renderables.size());
  for (const Renderable &renderable : renderables) {
    if (nuri::isValid(renderable.id)) {
      previousTransformById_[renderable.id] = renderable.modelMatrix;
    }
  }
  previousTransformSceneId_ = scene.id();
  previousTransformCaptureFrameIndex_ = frameIndex;
  previousTransformCaptureTopologyVersion_ = topologyVersion;
  previousTransformCaptureTransformVersion_ = transformVersion;
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
  meshletShader_.reset();
  meshShader_.reset();
  meshTessShader_.reset();
  meshDebugOverlayShader_.reset();
  meshPickShader_.reset();
  meshShadowInspectShader_.reset();
  meshVelocityShader_.reset();
  meshReactiveMaskShader_.reset();
  meshNormalShader_.reset();
  depthShader_.reset();
  depthAlphaShader_.reset();
  depthPyramidShader_.reset();
  computeShader_.reset();
  visibilityShader_.reset();
  visibilityIndirectDrawShader_.reset();
  visibilityIndirectMeshDispatchShader_.reset();
  meshVertexShader_ = {};
  meshTessVertexShader_ = {};
  meshTessControlShader_ = {};
  meshTessEvalShader_ = {};
  meshFragmentShader_ = {};
  meshDebugOverlayGeometryShader_ = {};
  meshDebugOverlayFragmentShader_ = {};
  meshPickVertexShader_ = {};
  meshPickTessVertexShader_ = {};
  meshPickTessControlShader_ = {};
  meshPickTessEvalShader_ = {};
  meshPickFragmentShader_ = {};
  meshShadowInspectFragmentShader_ = {};
  meshVelocityVertexShader_ = {};
  meshVelocityFragmentShader_ = {};
  meshReactiveMaskVertexShader_ = {};
  meshReactiveMaskFragmentShader_ = {};
  meshNormalFragmentShader_ = {};
  depthVertexShader_ = {};
  depthTessVertexShader_ = {};
  depthTessControlShader_ = {};
  depthTessEvalShader_ = {};
  depthAlphaVertexShader_ = {};
  depthAlphaTessVertexShader_ = {};
  depthAlphaTessControlShader_ = {};
  depthAlphaTessEvalShader_ = {};
  depthFragmentShader_ = {};
  depthAlphaFragmentShader_ = {};
  depthPyramidVertexShader_ = {};
  depthPyramidFragmentShader_ = {};
  computeShaderHandle_ = {};
  visibilityComputeShader_ = {};
  visibilityIndirectDrawComputeShader_ = {};
  visibilityIndirectMeshDispatchComputeShader_ = {};
  meshletTaskShader_ = {};
  meshletMeshShader_ = {};
  meshletFragmentShader_ = {};
  computePipelineHandle_ = {};
  visibilityPipelineHandle_ = {};
  visibilityIndirectDrawPipelineHandle_ = {};
  visibilityIndirectMeshDispatchPipelineHandle_ = {};
  tessellationUnsupported_ = false;
  renderableTemplates_.clear();
  meshDrawTemplates_.clear();
  templateBatchIndices_.clear();
  batchWriteOffsets_.clear();
  instanceCentersPhase_.clear();
  instanceBaseMatrices_.clear();
  instanceMatricesCpuCache_.clear();
  previousInstanceMatricesCpuCache_.clear();
  velocityInstanceFlagsCpuCache_.clear();
  velocityGeometryCpuCache_.clear();
  instanceMatricesUploadVersions_.clear();
  instanceLodCentersInvRadiusSq_.clear();
  materialTextureAccessHandles_.clear();
  instanceAutoLodLevels_.clear();
  instanceTessSelection_.clear();
  tessCandidates_.clear();
  instanceRemap_.clear();
  drawPushConstants_.clear();
  drawItems_.clear();
  drawAlphaMasked_.clear();
  meshletBatchInfos_.clear();
  indirectUploadSignatures_.clear();
  indirectDrawItems_.clear();
  indirectAlphaMasked_.clear();
  indirectCommandUploadBytes_.clear();
  overlayDrawItems_.clear();
  velocityDrawItems_.clear();
  reactiveMaskDrawItems_.clear();
  passDrawItems_.clear();
  depthPrepassDrawItems_.clear();
  transmissionVisibilityDepthDrawItems_.clear();
  transmissionVisibilityDepthPushConstants_.clear();
  depthPyramidPushConstants_.clear();
  depthPyramidDrawItems_.clear();
  meshletDispatchItems_.clear();
  meshletPushConstants_.clear();
  meshletBatchGpuData_.clear();
  meshletDispatchDependencyBuffers_.clear();
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
  depthPyramidDependencyTextures_.clear();
  preDispatches_.clear();
  mainPreDispatches_.clear();
  visibilityGpuDispatches_.clear();
  visibilityGpuCandidates_.clear();
  visibilityPassGpuData_.clear();
  visibilityCounterClear_.clear();
  visibilityGpuPushConstants_.clear();
  visibilityIndirectDrawPushConstants_.clear();
  visibilityMeshletDispatchGpuData_.clear();
  visibilityMeshletCandidateMap_.clear();
  visibilityIndirectMeshDispatchPushConstants_.clear();
  visibilityMeshletGpuDispatches_.clear();
  visibilityMeshletGpuDependencyBuffers_.clear();
  visibilityMeshletGpuDependencyBufferAccessModes_.clear();
  visibilityGpuDependencyBuffers_.clear();
  visibilityGpuDependencyBufferAccessModes_.clear();
  visibilityGpuDependencyTextures_.clear();
  passDependencyBuffers_.clear();
  passDependencyBufferAccessModes_.clear();
  preResolvedDecodeBuffers_.clear();
  preResolvedDrawBuffers_.clear();
  dispatchDependencyBuffers_.clear();
  passDependencyTextures_.clear();
  mainPassDependencyBuffers_.clear();
  mainPassDependencyBufferAccessModes_.clear();
  mainPassDependencyTextures_.clear();
  mainPassDependencyTextureAccessModes_.clear();
  velocityPassDependencyBuffers_.clear();
  velocityPassDependencyBufferAccessModes_.clear();
  reactivePassDependencyBuffers_.clear();
  reactivePassDependencyBufferAccessModes_.clear();
  previousTransformById_.clear();
  pickDrawItems_.clear();
  cachedPreResolvedBufferSignature_ = std::numeric_limits<uint64_t>::max();
  currentDirectDrawBufferSignature_ = std::numeric_limits<uint64_t>::max();
  currentIndirectDrawBufferSignature_ = std::numeric_limits<uint64_t>::max();
  cachedScene_ = nullptr;
  cachedTopologyVersion_ = std::numeric_limits<uint64_t>::max();
  cachedTransformVersion_ = std::numeric_limits<uint64_t>::max();
  cachedMaterialVersion_ = std::numeric_limits<uint64_t>::max();
  cachedGeometryMutationVersion_ = std::numeric_limits<uint64_t>::max();
  cachedExcludeTransmission_ = true;
  cachedAnimationSceneVersion_ = std::numeric_limits<uint64_t>::max();
  cachedAnimationSceneActive_ = false;
  previousTransformSceneId_ = 0u;
  previousTransformCaptureFrameIndex_ = std::numeric_limits<uint64_t>::max();
  previousTransformCaptureTopologyVersion_ =
      std::numeric_limits<uint64_t>::max();
  previousTransformCaptureTransformVersion_ =
      std::numeric_limits<uint64_t>::max();
  instanceStaticBuffersDirty_ = true;
  uniformSingleSubmeshPath_ = false;
  invalidateAutoLodCache();
  invalidateSingleInstanceBatchCache();
  invalidateIndirectPackCache();
  cachedRemapSignature_ = kInvalidDrawSignature;
  cachedRemapSignatureValid_ = false;
  invalidateStaticBatchCache();
  resetPickState();
  initialized_ = false;
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

  const RenderSettings &settings = settingsOrDefault(frame);
  if (!settings.opaque.enabled) {
    return;
  }
  const bool requiresDepthPyramid = this->requiresDepthPyramid(settings);
  const auto sceneDepthSamplerId = [this]() {
    return nuri::isValid(sceneDepthSampler_)
               ? gpu_.getSamplerBindlessIndex(sceneDepthSampler_)
               : gpu_.getDefaultSamplerBindlessIndex();
  };

  if (requiresDepthPyramid) {
    auto initResult = ensureInitialized();
    if (initResult.hasError()) {
      NURI_LOG_WARNING("OpaqueRenderer::publishFrameData: %s",
                       initResult.error().c_str());
      return;
    }
    auto samplerResult = ensureSceneDepthSampler();
    if (samplerResult.hasError()) {
      NURI_LOG_WARNING("OpaqueRenderer::publishFrameData: %s",
                       samplerResult.error().c_str());
      return;
    }
    // FrameCompositionProvider publishes the scene-depth texture after feature
    // publishFrameData(). Preserve the sampler choice now so later stages can
    // consume the depth texture with the intended sampling mode once the
    // provider fills in the texture handle.
    frame.sharedResources.sceneDepthSamplerId = sceneDepthSamplerId();
  }

  if (requiresDepthPyramid) {
    if (!nuri::isValid(depthPyramidPipelineHandle_)) {
      return;
    }
    auto pyramidResult = ensureDepthPyramidTextures();
    if (pyramidResult.hasError()) {
      if (!loggedDepthPyramidUnsupported_) {
        loggedDepthPyramidUnsupported_ = true;
        NURI_LOG_WARNING("OpaqueRenderer::publishFrameData: %s",
                         pyramidResult.error().c_str());
      }
      return;
    }
    frame.sharedResources.sceneDepthPyramidLevelCount =
        sceneDepthPyramidLevelCount_;
    frame.sharedResources.sceneDepthPyramidTextures =
        sceneDepthPyramidTextures_;
    bool hasValidPublishedPyramidSource =
        sceneDepthPyramidSourceFrameIndex_.has_value() &&
        sceneDepthPyramidSourceViewProj_.has_value() &&
        sceneDepthPyramidLevelCount_ > 0u;
    for (uint32_t level = 0u;
         hasValidPublishedPyramidSource && level < sceneDepthPyramidLevelCount_;
         ++level) {
      hasValidPublishedPyramidSource =
          nuri::isValid(sceneDepthPyramidTextures_[level]) &&
          gpu_.isValid(sceneDepthPyramidTextures_[level]);
    }
    if (hasValidPublishedPyramidSource) {
      frame.sharedResources.sceneDepthPyramidSourceFrameIndex =
          sceneDepthPyramidSourceFrameIndex_;
      frame.sharedResources.sceneDepthPyramidSourceViewProj =
          sceneDepthPyramidSourceViewProj_;
    }
  }
}

void OpaqueRenderer::readLatestVisibilityGpuReadback(
    RenderFrameContext &frame) {
  VisibilityFrameMetrics &metrics = frame.metrics.visibility;
  std::optional<VisibilityCounterGpuData> selectedCounter;
  size_t selectedSlotIndex = std::numeric_limits<size_t>::max();
  uint32_t selectedSourceFrame = 0u;
  uint32_t counterReadbackErrorCount = 0u;
  for (size_t slotIndex = 0u; slotIndex < visibilityCounterRing_.size();
       ++slotIndex) {
    const DynamicBufferSlot &slot = visibilityCounterRing_[slotIndex];
    if (!slot.buffer || !slot.buffer->valid()) {
      continue;
    }
    const uint64_t expectedFrame =
        slotIndex < visibilityCounterRingPublishedFrames_.size()
            ? visibilityCounterRingPublishedFrames_[slotIndex]
            : std::numeric_limits<uint64_t>::max();
    if (expectedFrame == std::numeric_limits<uint64_t>::max()) {
      continue;
    }

    VisibilityCounterGpuData counter{};
    auto readResult =
        gpu_.readBuffer(slot.buffer->handle(), 0u,
                        std::as_writable_bytes(
                            std::span<VisibilityCounterGpuData>(&counter, 1u)));
    if (readResult.hasError()) {
      ++counterReadbackErrorCount;
      continue;
    }
    const uint32_t valid = counter.status.w;
    const uint32_t sourceFrame = counter.status.z;
    if (valid == 0u || static_cast<uint64_t>(sourceFrame) >= frame.frameIndex ||
        sourceFrame != static_cast<uint32_t>(expectedFrame)) {
      continue;
    }
    if (!selectedCounter.has_value() || sourceFrame > selectedSourceFrame) {
      selectedCounter = counter;
      selectedSlotIndex = slotIndex;
      selectedSourceFrame = sourceFrame;
    }
  }

  metrics.gpuMainReadbackErrorCount = counterReadbackErrorCount;
  metrics.meshletReadbackErrorCount = counterReadbackErrorCount;

  if (!selectedCounter.has_value()) {
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
  if (selectedCounter->meshlet2.y != 0u) {
    metrics.meshletReadbackAvailable = 1u;
    metrics.meshletReadbackSourceFrame = selectedSourceFrame;
    metrics.meshletReadbackStaleFrameCount = staleFrameCount;
  }

  if (selectedSlotIndex >= visibilityVisibleIndexRing_.size()) {
    return;
  }
  const DynamicBufferSlot &visibleSlot =
      visibilityVisibleIndexRing_[selectedSlotIndex];
  if (!visibleSlot.buffer || !visibleSlot.buffer->valid()) {
    return;
  }
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

  metrics.gpuMainReadbackAvailable = 1u;
  metrics.gpuMainReadbackSourceFrame = selectedSourceFrame;
  metrics.gpuMainReadbackStaleFrameCount = staleFrameCount;
  metrics.gpuMainReadbackVisibleCandidates = selectedCounter->main.y;
  metrics.gpuIndirectDrawReadbackCommands = selectedCounter->indirect.x;
  metrics.gpuIndirectDrawReadbackVisible = selectedCounter->indirect.y;
  metrics.gpuIndirectDrawReadbackTombstoned = selectedCounter->indirect.z;
  if (selectedCounter->status.y != 0u) {
    metrics.gpuMainVisibleListMismatches = 0u;
    return;
  }

  const bool hasExpected =
      selectedSlotIndex < visibilityExpectedVisibleIndexCounts_.size() &&
      selectedSlotIndex < visibilityExpectedVisibleIndexHashes_.size();
  if (!hasExpected) {
    return;
  }
  const uint32_t expectedCount =
      visibilityExpectedVisibleIndexCounts_[selectedSlotIndex];
  const uint64_t expectedHash =
      visibilityExpectedVisibleIndexHashes_[selectedSlotIndex];
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
    std::span<const uint32_t> candidateIndices,
    const VisibilityPassRequest &request,
    const VisibilityResolvedSettings &settings,
    std::pmr::vector<PreparedGraphPass> &out) {
  if (candidateIndices.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }
  const bool visibilityListAvailable =
      nuri::isValid(visibilityPipelineHandle_) &&
      gpu_.isValid(visibilityPipelineHandle_);
  const bool gpuIndirectRequested = settings.enableGpuIndirectDraw;
  const bool gpuIndirectPipelineAvailable =
      nuri::isValid(visibilityIndirectDrawPipelineHandle_) &&
      gpu_.isValid(visibilityIndirectDrawPipelineHandle_);
  if (!visibilityListAvailable &&
      !(gpuIndirectRequested && gpuIndirectPipelineAvailable)) {
    if (gpuIndirectRequested) {
      frame.metrics.visibility.gpuIndirectDrawFallback = 1u;
    }
    if (!loggedVisibilityGpuUnsupportedWarning_) {
      loggedVisibilityGpuUnsupportedWarning_ = true;
      NURI_LOG_WARNING(
          "OpaqueRenderer::buildOpaquePasses: GPU visibility requested but "
          "visibility compute pipeline is unavailable");
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
  bool gpuIndirectSafe = false;
  uint32_t gpuIndirectCommandCount = 0u;
  if (gpuIndirectRequested && gpuIndirectPipelineAvailable &&
      gpuIndirectCandidateMapIdentity &&
      frameSlot < indirectCommandRing_.size() &&
      frameSlot < instanceRemapRing_.size() &&
      indirectCommandRing_[frameSlot].buffer &&
      instanceRemapRing_[frameSlot].buffer &&
      indirectCommandRing_[frameSlot].buffer->valid() &&
      instanceRemapRing_[frameSlot].buffer->valid() &&
      !indirectDrawItems_.empty()) {
    const BufferHandle indirectBuffer =
        indirectCommandRing_[frameSlot].buffer->handle();
    gpuIndirectSafe = true;
    gpuIndirectChunks.reserve(indirectDrawItems_.size());
    for (const DrawItem &indirectDraw : indirectDrawItems_) {
      if (indirectDraw.command != DrawCommandType::IndexedIndirect ||
          !isSameBufferHandle(indirectDraw.indirectBuffer, indirectBuffer) ||
          indirectDraw.indirectStride != sizeof(DrawIndexedIndirectCommand) ||
          indirectDraw.indirectDrawCount == 0u ||
          indirectDraw.indirectBufferOffset < kIndirectCountHeaderBytes ||
          indirectDraw.indirectBufferOffset % sizeof(uint32_t) != 0u) {
        gpuIndirectSafe = false;
        break;
      }

      const uint64_t commandWordOffset =
          indirectDraw.indirectBufferOffset / sizeof(uint32_t);
      if (commandWordOffset > std::numeric_limits<uint32_t>::max()) {
        gpuIndirectSafe = false;
        break;
      }

      const size_t headerOffset =
          static_cast<size_t>(indirectDraw.indirectBufferOffset) -
          kIndirectCountHeaderBytes;
      const size_t commandBytes =
          static_cast<size_t>(indirectDraw.indirectDrawCount) *
          sizeof(DrawIndexedIndirectCommand);
      if (headerOffset + kIndirectCountHeaderBytes + commandBytes <=
          indirectCommandUploadBytes_.size()) {
        uint32_t packedDrawCount = 0u;
        std::memcpy(&packedDrawCount,
                    indirectCommandUploadBytes_.data() + headerOffset,
                    sizeof(packedDrawCount));
        if (packedDrawCount != indirectDraw.indirectDrawCount) {
          gpuIndirectSafe = false;
          break;
        }
        for (uint32_t i = 0u;
             gpuIndirectSafe && i < indirectDraw.indirectDrawCount; ++i) {
          DrawIndexedIndirectCommand command{};
          const size_t commandOffset =
              headerOffset + kIndirectCountHeaderBytes +
              static_cast<size_t>(i) * sizeof(DrawIndexedIndirectCommand);
          std::memcpy(&command,
                      indirectCommandUploadBytes_.data() + commandOffset,
                      sizeof(command));
          const size_t firstInstance =
              static_cast<size_t>(command.firstInstance);
          const size_t instanceCount =
              static_cast<size_t>(command.instanceCount);
          if (instanceCount == 0u || firstInstance >= instanceRemap_.size() ||
              instanceCount > instanceRemap_.size() - firstInstance) {
            gpuIndirectSafe = false;
          }
        }
        if (!gpuIndirectSafe) {
          break;
        }
        gpuIndirectChunks.push_back(GpuIndirectDrawChunk{
            .commandWordOffset = static_cast<uint32_t>(commandWordOffset),
            .commandCount = indirectDraw.indirectDrawCount,
        });
        gpuIndirectCommandCount += indirectDraw.indirectDrawCount;
      } else {
        gpuIndirectSafe = false;
        break;
      }
    }
    gpuIndirectSafe = gpuIndirectSafe && gpuIndirectCommandCount > 0u;
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
  if (frameSlot >= visibilityCandidateRing_.size() ||
      frameSlot >= visibilityPassRing_.size() ||
      frameSlot >= visibilityVisibleIndexRing_.size() ||
      frameSlot >= visibilityCounterRing_.size() ||
      !visibilityCandidateRing_[frameSlot].buffer ||
      !visibilityPassRing_[frameSlot].buffer ||
      !visibilityVisibleIndexRing_[frameSlot].buffer ||
      !visibilityCounterRing_[frameSlot].buffer) {
    return Result<bool, std::string>::makeError(
        "OpaqueRenderer::buildOpaquePasses: visibility GPU ring is invalid");
  }

  frame.metrics.visibility.gpuMainCandidates = candidateCount;
  frame.metrics.visibility.gpuMainVisibleCandidates = candidateCount;
  readLatestVisibilityGpuReadback(frame);

  visibilityGpuCandidates_.clear();
  visibilityGpuCandidates_.reserve(candidateCount);
  std::pmr::vector<uint32_t> expectedVisibleIndices(
      visibilityGpuCandidates_.get_allocator().resource());
  expectedVisibleIndices.reserve(candidateCount);
  uint32_t expectedVisibleCount = 0u;
  uint64_t expectedVisibleHash = kFnvOffsetBasis64;
  for (uint32_t i = 0u; i < candidateCount; ++i) {
    const uint32_t sourceIndex = candidateIndices[i];
    if (sourceIndex >= candidates.size()) {
      return Result<bool, std::string>::makeError(
          "OpaqueRenderer::buildOpaquePasses: invalid visibility candidate "
          "index");
    }
    const VisibilityCandidate &candidate = candidates[sourceIndex];
    visibilityGpuCandidates_.push_back(makeVisibilityCandidateGpu(candidate));

    bool expectedVisible = true;
    const bool uncertainVisible =
        (candidate.flags & kVisibilityCandidateConservativeVisible) != 0u &&
        settings.visibleOnUncertain;
    if (!uncertainVisible) {
      const glm::vec4 sphere = visibility_detail::transformBoundingSphere(
          candidate.localBounds, candidate.worldFromLocal);
      expectedVisible = visibility_detail::classifySphere(
                            request.frustum, glm::vec3(sphere), sphere.w) !=
                        visibility_detail::VisibilityClassification::Outside;
    }
    if (expectedVisible) {
      ++expectedVisibleCount;
      expectedVisibleIndices.push_back(candidate.templateIndex);
    }
  }
  expectedVisibleHash =
      hashSortedVisibilityVisibleIndexList(std::span<uint32_t>(
          expectedVisibleIndices.data(), expectedVisibleIndices.size()));

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
          frame.frameIndex &&
      frame.sharedResources.sceneDepthPyramidLevelCount > 0u &&
      frame.sharedResources.sceneDepthSamplerId !=
          kInvalidSamplerBindlessIndex) {
    const uint32_t candidateLevelCount =
        std::min<uint32_t>(frame.sharedResources.sceneDepthPyramidLevelCount,
                           kMaxSceneDepthPyramidLevels);
    visibilityGpuDependencyTextures_.reserve(candidateLevelCount);
    for (uint32_t level = 0u; level < candidateLevelCount; ++level) {
      const TextureHandle texture =
          frame.sharedResources.sceneDepthPyramidTextures[level];
      if (!nuri::isValid(texture) || !gpu_.isValid(texture)) {
        break;
      }
      const uint32_t texId = gpu_.getTextureBindlessIndex(texture);
      if (texId == std::numeric_limits<uint32_t>::max()) {
        break;
      }
      const uint32_t packIndex = level / kSceneDepthPyramidTexIdPackWidth;
      const uint32_t componentIndex = level % kSceneDepthPyramidTexIdPackWidth;
      depthPyramidTexIds[packIndex][componentIndex] = texId;
      visibilityGpuDependencyTextures_.push_back(texture);
      depthPyramidLevelCount = level + 1u;
    }
    if (depthPyramidLevelCount > 0u) {
      const TextureDimensions dimensions = gpu_.getTextureDimensions(
          frame.sharedResources.sceneDepthPyramidTextures[0]);
      depthPyramidWidth = std::max(dimensions.width, 1u);
      depthPyramidHeight = std::max(dimensions.height, 1u);
      depthPyramidSamplerId = frame.sharedResources.sceneDepthSamplerId;
      occlusionAvailable = true;
    } else {
      visibilityGpuDependencyTextures_.clear();
    }
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
      visibilityCandidateRing_[frameSlot].buffer->handle();
  const BufferHandle passBuffer =
      visibilityPassRing_[frameSlot].buffer->handle();
  const BufferHandle visibleIndexBuffer =
      visibilityVisibleIndexRing_[frameSlot].buffer->handle();
  const BufferHandle counterBuffer =
      visibilityCounterRing_[frameSlot].buffer->handle();

  auto updateCandidateResult = gpu_.updateBuffer(
      candidateBuffer,
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(visibilityGpuCandidates_.data()),
          visibilityGpuCandidates_.size() * sizeof(VisibilityCandidateGpu)),
      0u);
  if (updateCandidateResult.hasError()) {
    return updateCandidateResult;
  }
  auto updatePassResult = gpu_.updateBuffer(
      passBuffer,
      std::as_bytes(std::span<const VisibilityPassGpuData>(
          visibilityPassGpuData_.data(), visibilityPassGpuData_.size())),
      0u);
  if (updatePassResult.hasError()) {
    return updatePassResult;
  }
  auto clearCounterResult = gpu_.updateBuffer(
      counterBuffer,
      std::as_bytes(std::span<const VisibilityCounterGpuData>(
          visibilityCounterClear_.data(), visibilityCounterClear_.size())),
      0u);
  if (clearCounterResult.hasError()) {
    return clearCounterResult;
  }
  if (frameSlot < visibilityCounterRingPublishedFrames_.size()) {
    visibilityCounterRingPublishedFrames_[frameSlot] = frame.frameIndex;
  }
  if (frameSlot < visibilityExpectedVisibleIndexCounts_.size() &&
      frameSlot < visibilityExpectedVisibleIndexHashes_.size()) {
    visibilityExpectedVisibleIndexCounts_[frameSlot] = expectedVisibleCount;
    visibilityExpectedVisibleIndexHashes_[frameSlot] = expectedVisibleHash;
  }

  const uint64_t candidateAddress =
      gpu_.getBufferDeviceAddress(candidateBuffer);
  const uint64_t passAddress = gpu_.getBufferDeviceAddress(passBuffer);
  const uint64_t visibleIndexAddress =
      gpu_.getBufferDeviceAddress(visibleIndexBuffer);
  const uint64_t counterAddress = gpu_.getBufferDeviceAddress(counterBuffer);
  const BufferHandle indirectBuffer =
      gpuIndirectSafe ? indirectCommandRing_[frameSlot].buffer->handle()
                      : BufferHandle{};
  const BufferHandle remapBuffer =
      gpuIndirectSafe ? instanceRemapRing_[frameSlot].buffer->handle()
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
    if (!loggedVisibilityGpuUnsupportedWarning_) {
      loggedVisibilityGpuUnsupportedWarning_ = true;
      NURI_LOG_WARNING(
          "OpaqueRenderer::buildOpaquePasses: visibility GPU buffer device "
          "addresses are unavailable");
    }
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
    dispatch.pipeline = visibilityPipelineHandle_;
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
      dispatch.pipeline = visibilityIndirectDrawPipelineHandle_;
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
  visibilityPass.hasPreDispatch = true;
  visibilityPass.hasDraws = false;
  visibilityPass.isVisibilityComputePass = true;

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
OpaqueRenderer::buildOpaquePasses(RenderFrameContext &frame,
                                  std::pmr::vector<PreparedGraphPass> &out) {
  NURI_PROFILER_FUNCTION();
  frame.metrics.opaque = {};
  frame.metrics.visibility.cpuMainCandidates = 0u;
  frame.metrics.visibility.cpuMainVisibleCandidates = 0u;
  frame.metrics.visibility.cpuMainRejected = 0u;
  frame.metrics.visibility.gpuMainCandidates = 0u;
  frame.metrics.visibility.gpuMainVisibleCandidates = 0u;
  frame.metrics.visibility.gpuMainRejectedFrustum = 0u;
  frame.metrics.visibility.gpuMainRejectedOcclusion = 0u;
  frame.metrics.visibility.gpuOutputOverflowCount = 0u;
  frame.metrics.visibility.gpuMainReadbackAvailable = 0u;
  frame.metrics.visibility.gpuMainReadbackSourceFrame = 0u;
  frame.metrics.visibility.gpuMainReadbackStaleFrameCount = 0u;
  frame.metrics.visibility.gpuMainReadbackErrorCount = 0u;
  frame.metrics.visibility.gpuMainReadbackVisibleCandidates = 0u;
  frame.metrics.visibility.gpuMainVisibleListMismatches = 0u;
  frame.metrics.visibility.meshletCandidates = 0u;
  frame.metrics.visibility.meshletRejectedFrustum = 0u;
  frame.metrics.visibility.meshletRejectedCone = 0u;
  frame.metrics.visibility.meshletRejectedOcclusion = 0u;
  frame.metrics.visibility.meshletOcclusionAvailable = 0u;
  frame.metrics.visibility.meshletPayloadOverflowCount = 0u;
  frame.metrics.visibility.meshletReadbackAvailable = 0u;
  frame.metrics.visibility.meshletReadbackSourceFrame = 0u;
  frame.metrics.visibility.meshletReadbackStaleFrameCount = 0u;
  frame.metrics.visibility.meshletReadbackErrorCount = 0u;
  frame.metrics.visibility.meshletEmitted = 0u;
  frame.metrics.visibility.meshletTaskGroupsExecuted = 0u;
  frame.metrics.visibility.indirectMeshDispatchCount = 0u;
  frame.metrics.visibility.uncertainVisible = 0u;
  frame.metrics.visibility.occlusionAvailable = 0u;
  const GpuTimingReport timingReport = gpu_.getLatestCompletedGpuTimingReport();
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

  const RenderSettings &settings = settingsOrDefault(frame);
  if (!settings.opaque.enabled) {
    return Result<bool, std::string>::makeResult(true);
  }

  if (!frame.scene) {
    return Result<bool, std::string>::makeError(
        "OpaqueRenderer::buildOpaquePasses: frame scene is null");
  }
  if (!frame.resources) {
    return Result<bool, std::string>::makeError(
        "OpaqueRenderer::buildOpaquePasses: frame resources are null");
  }
  const MaterialTableSnapshot materialSnapshot =
      frame.resources->materialSnapshot();

  auto initResult = ensureInitialized();
  if (initResult.hasError()) {
    return Result<bool, std::string>::makeError(initResult.error());
  }
  const TextureHandle sceneDepthTexture = resolveFrameDepthTexture(frame);
  frame.sharedResources.transmissionVisibilityDepthTexture = {};
  frame.sharedResources.transmissionVisibilityDepthGraphTexture = {};
  if (!nuri::isValid(sceneDepthTexture)) {
    return Result<bool, std::string>::makeError(
        "OpaqueRenderer::buildOpaquePasses: scene depth texture is "
        "unavailable");
  }
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
    if (readResult.hasError()) {
      NURI_LOG_WARNING(
          "OpaqueRenderer::buildOpaquePasses: pick readback failed: %s",
          readResult.error().c_str());
    } else {
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
    if (readResult.hasError()) {
      NURI_LOG_WARNING(
          "OpaqueRenderer::buildOpaquePasses: shadow inspect readback failed: "
          "%s",
          readResult.error().c_str());
    } else {
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
  const bool topologyDirty =
      cachedScene_ != frame.scene ||
      cachedTopologyVersion_ != frame.scene->topologyVersion();
  const bool materialDirty = topologyDirty || cachedScene_ != frame.scene ||
                             cachedMaterialVersion_ != materialSnapshot.version;
  const bool excludeTransmission = true;
  const bool transmissionPolicyDirty =
      cachedExcludeTransmission_ != excludeTransmission;
  const uint64_t geometryMutationVersion = gpu_.geometryMutationVersion();
  const bool hasGeometryMutationTracking = geometryMutationVersion != 0;
  if (topologyDirty || materialDirty || transmissionPolicyDirty) {
    auto cacheResult = rebuildSceneCache(
        *frame.scene, *frame.resources,
        static_cast<uint32_t>(materialSnapshot.headers.size()),
        excludeTransmission);
    if (cacheResult.hasError()) {
      return cacheResult;
    }
    if (hasGeometryMutationTracking) {
      cachedGeometryMutationVersion_ = geometryMutationVersion;
    }
  }
  const bool transformDirty =
      topologyDirty ||
      cachedTransformVersion_ != frame.scene->transformVersion();
  if (topologyDirty || transformDirty) {
    invalidateAutoLodCache();
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
    instanceCentersPhase_.reserve(instanceCount);
    instanceBaseMatrices_.reserve(instanceCount);
    instanceMatricesCpuCache_.reserve(instanceCount);
    instanceLodCentersInvRadiusSq_.reserve(instanceCount);

    const bool animateInstances = settings.opaque.enableInstanceAnimation;
    for (size_t i = 0; i < instanceCount; ++i) {
      const RenderableTemplate &templ = renderableTemplates_[i];
      const Renderable *renderable = templ.renderable;
      const Model *model = templ.model;
      if (!renderable || !model) {
        return Result<bool, std::string>::makeError(
            "OpaqueRenderer::buildOpaquePasses: invalid opaque renderable");
      }

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
      const float worldRadius = std::max(
          localRadius * maxAxisScale(renderable->modelMatrix), kMinLodRadius);
      const float invRadiusSq = 1.0f / (worldRadius * worldRadius);
      instanceLodCentersInvRadiusSq_.push_back(
          glm::vec4(worldCenter, invRadiusSq));
    }

    cachedTransformVersion_ = frame.scene->transformVersion();
    instanceStaticBuffersDirty_ = true;
    std::fill(instanceMatricesUploadVersions_.begin(),
              instanceMatricesUploadVersions_.end(),
              std::numeric_limits<uint64_t>::max());
  }

  const bool animationSceneStateDirty =
      topologyDirty ||
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
      if (animationSceneData != nullptr &&
          templateEntry.instanceIndex <
              animationSceneData->geometryOverridesByRenderable.size()) {
        const AnimatedRenderableGeometryOverride &geometryOverride =
            animationSceneData
                ->geometryOverridesByRenderable[templateEntry.instanceIndex];
        if (geometryOverride.enabled &&
            nuri::isValid(geometryOverride.vertexBuffer) &&
            templateEntry.submesh != nullptr &&
            animationOverrideCoversSubmesh(geometryOverride,
                                           *templateEntry.submesh)) {
          const uint64_t overrideVertexAddress = gpu_.getBufferDeviceAddress(
              geometryOverride.vertexBuffer, geometryOverride.vertexByteOffset);
          if (overrideVertexAddress != 0u) {
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

  if (topologyDirty || animationSceneStateDirty ||
      preResolvedDecodeBuffers_.empty()) {
    preResolvedDecodeBuffers_.clear();
    preResolvedDecodeBuffers_.reserve(meshDrawTemplates_.size());
    for (const MeshDrawTemplate &templateEntry : meshDrawTemplates_) {
      if (templateEntry.packedVertexFormat !=
              static_cast<uint32_t>(PackedVertexFormat::StaticQuantized20) ||
          !nuri::isValid(templateEntry.vertexDecodeBuffer)) {
        continue;
      }
      appendUniqueDrawBuffer(preResolvedDecodeBuffers_,
                             templateEntry.vertexDecodeBuffer);
    }
  }

  if (!frame.sharedResources.forwardSceneGpuData.has_value() ||
      !nuri::isValid(frame.sharedResources.forwardSceneGpuData->buffer) ||
      frame.sharedResources.forwardSceneGpuData->frameDataAddress == 0u) {
    return Result<bool, std::string>::makeError(
        "OpaqueRenderer::buildOpaquePasses: forward scene GPU data is "
        "unavailable");
  }
  if (!frame.sharedResources.materialTableGpuData.has_value()) {
    return Result<bool, std::string>::makeError(
        "OpaqueRenderer::buildOpaquePasses: material table GPU data is "
        "unavailable");
  }
  const ForwardSceneGpuData *sceneGpu =
      &*frame.sharedResources.forwardSceneGpuData;
  const MaterialTableGpuData *materialGpu =
      &*frame.sharedResources.materialTableGpuData;
  auto centersResult = ensureCentersPhaseBufferCapacity(
      std::max(instanceCount * sizeof(glm::vec4), sizeof(glm::vec4)));
  if (centersResult.hasError()) {
    return centersResult;
  }
  auto lodBoundsResult = ensureInstanceLodBoundsBufferCapacity(
      std::max(instanceCount * sizeof(glm::vec4), sizeof(glm::vec4)));
  if (lodBoundsResult.hasError()) {
    return lodBoundsResult;
  }
  auto baseMatricesResult = ensureInstanceBaseMatricesBufferCapacity(
      std::max(instanceCount * sizeof(glm::mat4), sizeof(glm::mat4)));
  if (baseMatricesResult.hasError()) {
    return baseMatricesResult;
  }
  auto matricesResult = ensureInstanceMatricesRingCapacity(
      std::max(instanceCount * sizeof(InstanceData), sizeof(InstanceData)));
  if (matricesResult.hasError()) {
    return matricesResult;
  }
  const bool taaSelected =
      sanitizeAntiAliasingMode(settings.antiAliasing.mode) ==
      AntiAliasingMode::TAA;
  const bool msaaSelected =
      sanitizeAntiAliasingMode(settings.antiAliasing.mode) ==
      AntiAliasingMode::MSAA4x;
  RenderSettings::AmbientOcclusionSettings ambientOcclusionSettings =
      settings.ambientOcclusion;
  const auto requestedAmbientOcclusionSettings = ambientOcclusionSettings;
  sanitizeAmbientOcclusionSettings(ambientOcclusionSettings, settings.opaque,
                                   settings.antiAliasing);
  AmbientOcclusionFrameMetrics &aoMetrics = frame.metrics.ambientOcclusion;
  aoMetrics.enabled =
      ambientOcclusionSettings.mode != AmbientOcclusionMode::Disabled;
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
  const bool hasTaaVelocityInstances = taaSelected && instanceCount > 0;
  const bool animationPreviousFrameValid =
      hasTaaVelocityInstances && animationSceneData != nullptr &&
      frame.camera.historyValid && frame.camera.temporalDataValid &&
      nuri::isValid(animationSceneData->previousInstanceMatricesBuffer) &&
      animationSceneData->previousInstanceMatricesAddress != 0u;
  const bool transformPreviousCacheValid =
      hasTaaVelocityInstances && frame.camera.historyValid &&
      frame.camera.temporalDataValid &&
      previousTransformSceneId_ == frame.scene->id() &&
      previousTransformCaptureFrameIndex_ !=
          std::numeric_limits<uint64_t>::max() &&
      previousTransformCaptureFrameIndex_ < frame.frameIndex;
  const bool previousCacheValid = animationSceneData != nullptr
                                      ? animationPreviousFrameValid
                                      : transformPreviousCacheValid;
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
  if (msaaSelected &&
      (!nuri::isValid(frame.sharedResources.msaaSceneDepthTexture) ||
       !nuri::isValid(frame.sharedResources.msaaSceneColorTexture))) {
    return Result<bool, std::string>::makeError(
        "OpaqueRenderer::buildOpaquePasses: MSAA scene color/depth textures "
        "are unavailable");
  }
  const TextureHandle sceneDepthTarget =
      msaaSelected ? frame.sharedResources.msaaSceneDepthTexture
                   : sceneDepthTexture;
  if (msaaSelected && !nuri::isValid(sceneDepthTarget)) {
    return Result<bool, std::string>::makeError(
        "OpaqueRenderer::buildOpaquePasses: MSAA scene depth texture is "
        "unavailable");
  }
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
    if (!instanceCentersPhase_.empty()) {
      const std::span<const std::byte> centersBytes{
          reinterpret_cast<const std::byte *>(instanceCentersPhase_.data()),
          instanceCentersPhase_.size() * sizeof(glm::vec4)};
      auto updateResult = gpu_.updateBuffer(
          instanceCentersPhaseBuffer_->handle(), centersBytes, 0);
      if (updateResult.hasError()) {
        return updateResult;
      }
    }
    if (!instanceBaseMatrices_.empty()) {
      const std::span<const std::byte> baseMatricesBytes{
          reinterpret_cast<const std::byte *>(instanceBaseMatrices_.data()),
          instanceBaseMatrices_.size() * sizeof(glm::mat4)};
      auto updateResult = gpu_.updateBuffer(
          instanceBaseMatricesBuffer_->handle(), baseMatricesBytes, 0);
      if (updateResult.hasError()) {
        return updateResult;
      }
    }
    if (!instanceLodCentersInvRadiusSq_.empty()) {
      const std::span<const std::byte> lodBoundsBytes{
          reinterpret_cast<const std::byte *>(
              instanceLodCentersInvRadiusSq_.data()),
          instanceLodCentersInvRadiusSq_.size() * sizeof(glm::vec4)};
      auto updateResult = gpu_.updateBuffer(instanceLodBoundsBuffer_->handle(),
                                            lodBoundsBytes, 0);
      if (updateResult.hasError()) {
        return updateResult;
      }
    }
    instanceStaticBuffersDirty_ = false;
  }

  if (materialDirty) {
    cachedMaterialVersion_ = materialSnapshot.version;
  }
  if (materialDirty || transmissionPolicyDirty ||
      materialTextureAccessHandles_.empty()) {
    NURI_PROFILER_ZONE("OpaqueRenderer.material_access_cache",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    auto materialAccessCacheResult = rebuildMaterialTextureAccessCache(
        *frame.scene, *frame.resources, excludeTransmission);
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
          : instanceMatricesRing_[frameSlot].buffer->handle();
  const uint64_t instanceMatricesAddress =
      animationSceneData != nullptr
          ? animationSceneData->instanceMatricesAddress
          : gpu_.getBufferDeviceAddress(instanceMatricesBufferHandle);
  const BufferHandle previousInstanceMatricesBufferHandle =
      animationPreviousFrameValid
          ? animationSceneData->previousInstanceMatricesBuffer
          : (needsVelocityInstanceBufferUpload &&
                     frameSlot < previousInstanceMatricesRing_.size() &&
                     previousInstanceMatricesRing_[frameSlot].buffer
                 ? previousInstanceMatricesRing_[frameSlot].buffer->handle()
                 : BufferHandle{});
  const BufferHandle velocityInstanceFlagsBufferHandle =
      needsVelocityInstanceBufferUpload &&
              frameSlot < velocityInstanceFlagsRing_.size() &&
              velocityInstanceFlagsRing_[frameSlot].buffer
          ? velocityInstanceFlagsRing_[frameSlot].buffer->handle()
          : BufferHandle{};
  const BufferHandle velocityFrameDataBufferHandle =
      hasTaaVelocityInstances && frameSlot < velocityFrameDataRing_.size() &&
              velocityFrameDataRing_[frameSlot].buffer
          ? velocityFrameDataRing_[frameSlot].buffer->handle()
          : BufferHandle{};
  const BufferHandle velocityGeometryBufferHandle =
      needsVelocityGeometryUpload && frameSlot < velocityGeometryRing_.size() &&
              velocityGeometryRing_[frameSlot].buffer
          ? velocityGeometryRing_[frameSlot].buffer->handle()
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
  if (frameDataAddress == 0 || instanceCentersPhaseAddress == 0 ||
      instanceLodBoundsAddress == 0 || instanceBaseMatricesAddress == 0 ||
      instanceMatricesAddress == 0 ||
      (hasTaaVelocityInstances &&
       (previousInstanceMatricesAddress == 0u ||
        (needsVelocityInstanceBufferUpload &&
         velocityInstanceFlagsAddress == 0u) ||
        (needsVelocityGeometryUpload && velocityGeometryAddress == 0u) ||
        velocityFrameDataAddress == 0u)) ||
      materialGpu->headerBufferAddress == 0u ||
      materialGpu->clearcoatBufferAddress == 0u ||
      materialGpu->sheenBufferAddress == 0u ||
      materialGpu->transmissionBufferAddress == 0u ||
      materialGpu->specularBufferAddress == 0u ||
      (sceneGpu->directionalLightCount > 0u &&
       directionalLightBufferAddress == 0u) ||
      (sceneGpu->localLightCount > 0u && localLightBufferAddress == 0u) ||
      ((sceneGpu->shadowFlags & kShadowFrameFlagEnabled) != 0u &&
       sceneGpu->shadowFrameBufferAddress == 0u)) {
    return Result<bool, std::string>::makeError(
        "OpaqueRenderer::buildOpaquePasses: invalid GPU buffer address");
  }

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
        const glm::mat4 currentModel =
            renderable != nullptr ? renderable->modelMatrix : glm::mat4(1.0f);
        glm::mat4 previousModel = currentModel;
        bool hasPrevious = false;
        if (!animatedInstance && previousCacheValid && renderable != nullptr &&
            nuri::isValid(renderable->id)) {
          if (const auto it = previousTransformById_.find(renderable->id);
              it != previousTransformById_.end()) {
            previousModel = it->second;
            hasPrevious = true;
          }
        }

        if (hasPrevious) {
          ++validPreviousCount;
          const float motion = glm::length(glm::vec3(currentModel[3]) -
                                           glm::vec3(previousModel[3]));
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
      auto previousUpdateResult = gpu_.updateBuffer(
          previousInstanceMatricesBufferHandle, previousMatricesBytes, 0);
      if (previousUpdateResult.hasError()) {
        return previousUpdateResult;
      }

      const std::span<const std::byte> velocityFlagsBytes{
          reinterpret_cast<const std::byte *>(
              velocityInstanceFlagsCpuCache_.data()),
          velocityInstanceFlagsCpuCache_.size() * sizeof(uint32_t)};
      auto flagsUpdateResult = gpu_.updateBuffer(
          velocityInstanceFlagsBufferHandle, velocityFlagsBytes, 0);
      if (flagsUpdateResult.hasError()) {
        return flagsUpdateResult;
      }
    }

    if (needsVelocityGeometryUpload) {
      velocityGeometryCpuCache_.clear();
      velocityGeometryCpuCache_.resize(instanceCount);
      const size_t previousOverrideCount = std::min(
          animationSceneData->previousGeometryOverridesByRenderable.size(),
          instanceCount);
      for (size_t i = 0; i < previousOverrideCount; ++i) {
        const AnimatedRenderableGeometryOverride &previousOverride =
            animationSceneData->previousGeometryOverridesByRenderable[i];
        if (!previousOverride.enabled ||
            !nuri::isValid(previousOverride.vertexBuffer)) {
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
      nuri::isValid(meshTessPipelineHandle_);

  struct BatchEntry {
    DrawItem draw{};
    BufferHandle vertexBuffer{};
    BufferHandle vertexDecodeBuffer{};
    uint64_t vertexBufferAddress = 0;
    uint64_t vertexDecodeBufferAddress = 0;
    uint32_t vertexDecodeIndex = 0;
    uint32_t packedVertexFormat = 0;
    uint32_t materialIndex = kInvalidMaterialIndex;
    const Model::ModelMeshletGpuView *meshletView = nullptr;
    uint32_t meshletOffset = 0;
    uint32_t meshletCount = 0;
    uint32_t submeshIndex = 0;
    uint32_t meshletMaxCount = 0;
    uint32_t vertexOffset = 0;
    bool doubleSided = false;
    size_t instanceCount = 0;
    size_t firstInstance = 0;
    bool alphaMasked = false;
  };
  constexpr uint32_t kInvalidBatchIndex = std::numeric_limits<uint32_t>::max();

  ScratchArena batchScratchArena;
  ScopedScratch batchScratch(batchScratchArena);
  std::pmr::vector<BatchEntry> batches(batchScratch.resource());
  const size_t batchReserve =
      std::min<size_t>(meshDrawTemplates_.size(), kMaxBatchReserve);
  batches.reserve(batchReserve);
  const auto appendBatch =
      [&baseDraw, &batches](
          RenderPipelineHandle pipeline, BufferHandle indexBuffer,
          uint64_t indexBufferOffset, const SubmeshLod &lodRange,
          uint32_t vertexOffset, uint32_t submeshIndex,
          uint32_t meshletMaxCount, BufferHandle vertexBuffer,
          BufferHandle vertexDecodeBuffer, uint64_t vertexBufferAddress,
          uint64_t vertexDecodeBufferAddress, uint32_t vertexDecodeIndex,
          uint32_t packedVertexFormat, uint32_t materialIndex,
          const Model::ModelMeshletGpuView *meshletView, bool doubleSided,
          bool alphaMasked, size_t count, size_t firstInstance) {
        if (count == 0) {
          return;
        }
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
        entry.meshletMaxCount = meshletMaxCount;
        entry.vertexOffset = vertexOffset;
        entry.doubleSided = doubleSided;
        entry.alphaMasked = alphaMasked;
        entry.instanceCount = count;
        entry.firstInstance = firstInstance;
        batches.push_back(entry);
      };
  const VisibilityResolvedSettings visibilitySettings =
      visibilitySettingsFromRenderSettings(settings);
  const bool cpuMainCullingEnabled =
      visibilitySettings.enableCpuMainFrustumCulling;
  const bool gpuMainCullingEnabled =
      visibilitySettings.enableGpuInstanceCulling;
  std::pmr::vector<uint8_t> visibleMainTemplates(batchScratch.resource());
  std::pmr::vector<VisibilityCandidate> visibilityCandidates(
      batchScratch.resource());
  std::pmr::vector<uint32_t> gpuVisibilityCandidateIndices(
      batchScratch.resource());
  VisibilityPassRequest visibilityRequest{};
  bool hasDeformedRenderable = false;
  if (cpuMainCullingEnabled || gpuMainCullingEnabled) {
    visibilityCandidates.reserve(meshDrawTemplates_.size());
    for (size_t templateIndex = 0; templateIndex < meshDrawTemplates_.size();
         ++templateIndex) {
      const MeshDrawTemplate &templ = meshDrawTemplates_[templateIndex];
      if (templ.renderable == nullptr || templ.submesh == nullptr) {
        return Result<bool, std::string>::makeError(
            "OpaqueRenderer::buildOpaquePasses: invalid visibility template");
      }
      const bool deformed = !templ.renderable->morphWeights.empty() ||
                            !templ.renderable->skinPalette.empty();
      hasDeformedRenderable |= deformed;
      visibilityCandidates.push_back(VisibilityCandidate{
          .renderableIndex = templ.instanceIndex,
          .templateIndex = static_cast<uint32_t>(templateIndex),
          .submeshIndex = templ.submeshIndex,
          .materialIndex = templ.materialIndex,
          .geometryVersion = gpu_.geometryMutationVersion(),
          .transformVersion = frame.scene->transformVersion(),
          .deformationVersion = frame.scene->deformationVersion(),
          .flags = deformed ? static_cast<uint32_t>(
                                  kVisibilityCandidateConservativeVisible)
                            : 0u,
          .localBounds = templ.submesh->bounds,
          .worldFromLocal = templ.renderable->modelMatrix,
          .meshletView = templ.meshletView,
      });
    }
    visibilityRequest =
        makeMainViewVisibilityPassRequest(frame.camera, visibilitySettings);
  }
  if (!hasDeformedRenderable &&
      !(cpuMainCullingEnabled || gpuMainCullingEnabled)) {
    for (const RenderableTemplate &templ : renderableTemplates_) {
      if (templ.renderable != nullptr &&
          (!templ.renderable->morphWeights.empty() ||
           !templ.renderable->skinPalette.empty())) {
        hasDeformedRenderable = true;
        break;
      }
    }
  }
  if (cpuMainCullingEnabled) {
    NURI_PROFILER_ZONE("OpaqueRenderer.visibility_cpu_main",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    visibleMainTemplates.assign(meshDrawTemplates_.size(), 0u);

    VisibilityFrameState visibilityState(batchScratch.resource());
    VisibilityPassResult visibilityResult =
        visibilityState.evaluateCpu(visibilityRequest, visibilityCandidates);
    for (const uint32_t candidateIndex :
         visibilityResult.visibleCandidateIndices) {
      if (candidateIndex < visibilityCandidates.size()) {
        const uint32_t templateIndex =
            visibilityCandidates[candidateIndex].templateIndex;
        if (templateIndex < visibleMainTemplates.size()) {
          visibleMainTemplates[templateIndex] = 1u;
        }
      }
    }
    frame.metrics.visibility.cpuMainCandidates = visibilityResult.cpuCandidates;
    frame.metrics.visibility.cpuMainVisibleCandidates =
        visibilityResult.cpuVisibleCandidates;
    frame.metrics.visibility.cpuMainRejected = visibilityResult.cpuRejected;
    frame.metrics.visibility.uncertainVisible =
        visibilityResult.uncertainVisible;
    if (gpuMainCullingEnabled) {
      gpuVisibilityCandidateIndices.assign(
          visibilityResult.visibleCandidateIndices.begin(),
          visibilityResult.visibleCandidateIndices.end());
    }
    NURI_PROFILER_ZONE_END();
  }
  if (gpuMainCullingEnabled && !cpuMainCullingEnabled &&
      gpuVisibilityCandidateIndices.empty()) {
    gpuVisibilityCandidateIndices.reserve(visibilityCandidates.size());
    for (size_t i = 0; i < visibilityCandidates.size(); ++i) {
      if (i <= static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        gpuVisibilityCandidateIndices.push_back(static_cast<uint32_t>(i));
      }
    }
  }
  if (settings.opaque.meshLodDistanceThresholds !=
      cachedMeshLodThresholdsInput_) {
    cachedMeshLodThresholdsInput_ = settings.opaque.meshLodDistanceThresholds;
    cachedSortedLodThresholds_ = {
        settings.opaque.meshLodDistanceThresholds.x,
        settings.opaque.meshLodDistanceThresholds.y,
        settings.opaque.meshLodDistanceThresholds.z,
    };
    std::sort(cachedSortedLodThresholds_.begin(),
              cachedSortedLodThresholds_.end());
  }
  const std::array<float, 3> &sortedLodThresholds = cachedSortedLodThresholds_;
  const glm::vec3 cameraPosition = glm::vec3(frame.camera.cameraPos);
  const bool useAutoLod = settings.opaque.enableMeshLod &&
                          settings.opaque.forcedMeshLod < 0 &&
                          !meshletRequested;
  const bool canUseUniformAutoLodFastPath =
      !cpuMainCullingEnabled && uniformSingleSubmeshPath_ &&
      !meshDrawTemplates_.empty() && useAutoLod &&
      instanceCount == meshDrawTemplates_.size();
  const uint32_t forcedLod =
      settings.opaque.forcedMeshLod < 0
          ? 0u
          : static_cast<uint32_t>(settings.opaque.forcedMeshLod);
  if (useAutoLod && !canUseUniformAutoLodFastPath) {
    NURI_PROFILER_ZONE("OpaqueRenderer.auto_lod_resolve",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    if (instanceLodCentersInvRadiusSq_.size() != renderableTemplates_.size()) {
      return Result<bool, std::string>::makeError(
          "OpaqueRenderer::buildOpaquePasses: LOD cache size mismatch");
    }

    instanceAutoLodLevels_.clear();
    instanceAutoLodLevels_.resize(renderableTemplates_.size(), 0u);
    const float lodThreshold0Sq =
        sortedLodThresholds[0] * sortedLodThresholds[0];
    const float lodThreshold1Sq =
        sortedLodThresholds[1] * sortedLodThresholds[1];
    const float lodThreshold2Sq =
        sortedLodThresholds[2] * sortedLodThresholds[2];
    const float cameraX = cameraPosition.x;
    const float cameraY = cameraPosition.y;
    const float cameraZ = cameraPosition.z;
    for (size_t i = 0; i < instanceLodCentersInvRadiusSq_.size(); ++i) {
      const glm::vec4 lodCache = instanceLodCentersInvRadiusSq_[i];
      const float dx = cameraX - lodCache.x;
      const float dy = cameraY - lodCache.y;
      const float dz = cameraZ - lodCache.z;
      const float normalizedDistanceSq =
          (dx * dx + dy * dy + dz * dz) * lodCache.w;

      uint32_t lodIndex = 0;
      if (normalizedDistanceSq >= lodThreshold2Sq) {
        lodIndex = 3;
      } else if (normalizedDistanceSq >= lodThreshold1Sq) {
        lodIndex = 2;
      } else if (normalizedDistanceSq >= lodThreshold0Sq) {
        lodIndex = 1;
      }
      instanceAutoLodLevels_[i] = lodIndex;
    }
    NURI_PROFILER_ZONE_END();
  }
  const auto refreshTemplateGeometry =
      [this](MeshDrawTemplate &templateEntry) -> Result<bool, std::string> {
    GeometryAllocationView geometry{};
    if (!gpu_.resolveGeometry(templateEntry.geometryHandle, geometry)) {
      return Result<bool, std::string>::makeError(
          "OpaqueRenderer::buildOpaquePasses: failed to refresh geometry");
    }
    if (!nuri::isValid(geometry.vertexBuffer) ||
        !nuri::isValid(geometry.indexBuffer) ||
        !gpu_.isValid(geometry.vertexBuffer) ||
        !gpu_.isValid(geometry.indexBuffer)) {
      return Result<bool, std::string>::makeError(
          "OpaqueRenderer::buildOpaquePasses: refreshed geometry is invalid");
    }
    const uint64_t refreshedVertexAddress = gpu_.getBufferDeviceAddress(
        geometry.vertexBuffer, geometry.vertexByteOffset);
    if (refreshedVertexAddress == 0) {
      return Result<bool, std::string>::makeError(
          "OpaqueRenderer::buildOpaquePasses: refreshed vertex address is "
          "invalid");
    }
    const uint64_t previousBaseVertexAddress =
        templateEntry.baseVertexBufferAddress;
    const bool wasUsingBaseVertexAddress =
        templateEntry.vertexBufferAddress == previousBaseVertexAddress;
    templateEntry.indexBuffer = geometry.indexBuffer;
    templateEntry.indexBufferOffset = geometry.indexByteOffset;
    templateEntry.baseVertexBuffer = geometry.vertexBuffer;
    templateEntry.baseVertexBufferAddress = refreshedVertexAddress;
    if (wasUsingBaseVertexAddress) {
      templateEntry.vertexBuffer = templateEntry.baseVertexBuffer;
      templateEntry.vertexBufferAddress = refreshedVertexAddress;
    }
    return Result<bool, std::string>::makeResult(true);
  };
  const bool shouldRefreshTemplateGeometry =
      !meshDrawTemplates_.empty() &&
      (!hasGeometryMutationTracking ||
       cachedGeometryMutationVersion_ != geometryMutationVersion);
  if (shouldRefreshTemplateGeometry) {
    bool templateGeometryChanged = false;
    {
      NURI_PROFILER_ZONE("OpaqueRenderer.sync_template_geometry",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      if (uniformSingleSubmeshPath_) {
        MeshDrawTemplate &firstTemplate = meshDrawTemplates_.front();
        const BufferHandle previousIndexBuffer = firstTemplate.indexBuffer;
        const uint64_t previousIndexBufferOffset =
            firstTemplate.indexBufferOffset;
        const uint64_t previousVertexBufferAddress =
            firstTemplate.vertexBufferAddress;

        auto refreshResult = refreshTemplateGeometry(firstTemplate);
        if (refreshResult.hasError()) {
          return refreshResult;
        }

        const bool geometryChanged =
            !isSameBufferHandle(firstTemplate.indexBuffer,
                                previousIndexBuffer) ||
            firstTemplate.indexBufferOffset != previousIndexBufferOffset ||
            firstTemplate.vertexBufferAddress != previousVertexBufferAddress;
        templateGeometryChanged = geometryChanged;
        if (geometryChanged) {
          for (size_t i = 1; i < meshDrawTemplates_.size(); ++i) {
            MeshDrawTemplate &templateEntry = meshDrawTemplates_[i];
            templateEntry.indexBuffer = firstTemplate.indexBuffer;
            templateEntry.indexBufferOffset = firstTemplate.indexBufferOffset;
            templateEntry.baseVertexBuffer = firstTemplate.baseVertexBuffer;
            templateEntry.vertexBuffer = firstTemplate.vertexBuffer;
            templateEntry.baseVertexDecodeBuffer =
                firstTemplate.baseVertexDecodeBuffer;
            templateEntry.vertexDecodeBuffer = firstTemplate.vertexDecodeBuffer;
            templateEntry.baseVertexBufferAddress =
                firstTemplate.baseVertexBufferAddress;
            templateEntry.baseVertexDecodeBufferAddress =
                firstTemplate.baseVertexDecodeBufferAddress;
            templateEntry.vertexBufferAddress =
                firstTemplate.vertexBufferAddress;
            templateEntry.vertexDecodeBufferAddress =
                firstTemplate.vertexDecodeBufferAddress;
          }
        }
      } else {
        GeometryAllocationHandle cachedHandle{};
        BufferHandle cachedIndexBuffer{};
        BufferHandle cachedVertexBuffer{};
        uint64_t cachedIndexBufferOffset = 0;
        uint64_t cachedVertexBufferAddress = 0;
        bool hasCachedGeometry = false;

        for (MeshDrawTemplate &templateEntry : meshDrawTemplates_) {
          const BufferHandle previousIndexBuffer = templateEntry.indexBuffer;
          const uint64_t previousIndexBufferOffset =
              templateEntry.indexBufferOffset;
          const BufferHandle previousVertexBuffer = templateEntry.vertexBuffer;
          const uint64_t previousVertexBufferAddress =
              templateEntry.vertexBufferAddress;

          const bool sameAsCached =
              hasCachedGeometry &&
              isSameGeometryAllocationHandle(templateEntry.geometryHandle,
                                             cachedHandle);
          if (sameAsCached) {
            templateEntry.indexBuffer = cachedIndexBuffer;
            templateEntry.indexBufferOffset = cachedIndexBufferOffset;
            templateEntry.baseVertexBuffer = cachedVertexBuffer;
            templateEntry.baseVertexBufferAddress = cachedVertexBufferAddress;
            templateEntry.vertexBuffer = cachedVertexBuffer;
            templateEntry.vertexBufferAddress = cachedVertexBufferAddress;
          } else {
            auto refreshResult = refreshTemplateGeometry(templateEntry);
            if (refreshResult.hasError()) {
              return refreshResult;
            }

            cachedHandle = templateEntry.geometryHandle;
            cachedIndexBuffer = templateEntry.indexBuffer;
            cachedVertexBuffer = templateEntry.baseVertexBuffer;
            cachedIndexBufferOffset = templateEntry.indexBufferOffset;
            cachedVertexBufferAddress = templateEntry.baseVertexBufferAddress;
            hasCachedGeometry = true;
          }

          const bool geometryChanged =
              !isSameBufferHandle(templateEntry.indexBuffer,
                                  previousIndexBuffer) ||
              !isSameBufferHandle(templateEntry.vertexBuffer,
                                  previousVertexBuffer) ||
              templateEntry.indexBufferOffset != previousIndexBufferOffset ||
              templateEntry.vertexBufferAddress != previousVertexBufferAddress;
          templateGeometryChanged = templateGeometryChanged || geometryChanged;
        }
      }
      NURI_PROFILER_ZONE_END();
    }
    if (templateGeometryChanged) {
      invalidateSingleInstanceBatchCache();
      invalidateStaticBatchCache();
      invalidateIndirectPackCache();
    }
    if (hasGeometryMutationTracking) {
      cachedGeometryMutationVersion_ = geometryMutationVersion;
    }
  }
  bool usedUniformFastPath = false;
  bool usedUniformAutoLodFastPath = false;
  bool reusedUniformAutoLodFastPath = false;
  bool usedUniformAutoLodTessSplit = false;
  std::array<size_t, Submesh::kMaxLodCount> autoLodBucketStarts{};
  std::array<size_t, Submesh::kMaxLodCount> autoLodBucketWrites{};
  std::array<size_t, Submesh::kMaxLodCount> autoLodBucketCounts{};
  size_t autoLodTessBucketStart = 0;
  size_t autoLodTessBucketWrite = 0;
  size_t autoLodTessBucketCount = 0;
  const Submesh *activeFastAutoLodSubmesh = nullptr;
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
    if (batchCount != staticBatchCache_.pushConstantsTemplates.size()) {
      return Result<bool, std::string>::makeError(
          "OpaqueRenderer::buildOpaquePasses: static batch cache size "
          "mismatch");
    }

    remapCount = staticBatchCache_.remap.size();
    const bool needsStaticBatchRebind =
        boundStaticBatchGeneration_ != staticBatchCache_.generation ||
        drawItems_.size() != batchCount ||
        meshletBatchInfos_.size() != batchCount;
    if (needsStaticBatchRebind) {
      drawItems_ = staticBatchCache_.draws;
      drawPushConstants_ = staticBatchCache_.pushConstantsTemplates;
      drawAlphaMasked_ = staticBatchCache_.alphaMasked;
      meshletBatchInfos_ = staticBatchCache_.meshletBatchInfos;
      if (drawAlphaMasked_.size() != batchCount) {
        drawAlphaMasked_.assign(batchCount, 0u);
      }
      if (meshletBatchInfos_.size() != batchCount) {
        meshletBatchInfos_.assign(batchCount, MeshletBatchInfo{});
      }
      for (size_t batchIndex = 0; batchIndex < batchCount; ++batchIndex) {
        drawItems_[batchIndex].alphaMasked =
            batchIndex < drawAlphaMasked_.size() &&
            drawAlphaMasked_[batchIndex] != 0u;
        drawItems_[batchIndex].pushConstants =
            std::span<const std::byte>(reinterpret_cast<const std::byte *>(
                                           &drawPushConstants_[batchIndex]),
                                       sizeof(PushConstants));
      }
      boundStaticBatchGeneration_ = staticBatchCache_.generation;
    } else {
      drawPushConstants_.resize(batchCount);
      if (drawAlphaMasked_.size() != batchCount) {
        drawAlphaMasked_ = staticBatchCache_.alphaMasked;
        if (drawAlphaMasked_.size() != batchCount) {
          drawAlphaMasked_.assign(batchCount, 0u);
        }
      }
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
    if (!templateEntry.submesh) {
      return Result<bool, std::string>::makeError(
          "OpaqueRenderer::buildOpaquePasses: invalid auto-LOD submesh");
    }

    const Submesh &submesh = *templateEntry.submesh;
    const uint32_t submeshMeshletMaxCount = maxMeshletCountForSubmesh(submesh);
    activeFastAutoLodSubmesh = &submesh;

    const bool canTemporallyReuseFastAutoLod =
        instanceCount >= kAutoLodTemporalReuseMinInstances &&
        autoLodCache_.frameIndex != std::numeric_limits<uint64_t>::max() &&
        frame.frameIndex > autoLodCache_.frameIndex &&
        (frame.frameIndex - autoLodCache_.frameIndex) <
            kAutoLodTemporalReuseFrameInterval;
    const bool cameraStableForReuse = nearlyEqualVec3(
        autoLodCache_.cameraPos, cameraPosition, kAutoLodCameraReuseEpsilon);
    const bool canReuseFastAutoLodCache =
        !tessellationRequested && autoLodCache_.valid &&
        autoLodCache_.submesh == &submesh &&
        autoLodCache_.instanceCount == instanceCount &&
        autoLodCache_.remapCount == instanceRemap_.size() &&
        (cameraStableForReuse || canTemporallyReuseFastAutoLod) &&
        nearlyEqualThresholds(autoLodCache_.thresholds, sortedLodThresholds,
                              kAutoLodThresholdReuseEpsilon);

    if (canReuseFastAutoLodCache) {
      autoLodBucketCounts = autoLodCache_.bucketCounts;
      remapCount = autoLodCache_.remapCount;
      reusedUniformAutoLodFastPath = true;
    } else {
      if (instanceLodCentersInvRadiusSq_.size() != instanceCount) {
        return Result<bool, std::string>::makeError(
            "OpaqueRenderer::buildOpaquePasses: auto-LOD cache size "
            "mismatch");
      }
      instanceAutoLodLevels_.clear();
      instanceAutoLodLevels_.resize(instanceCount, 0u);
      if (tessellationRequested) {
        instanceTessSelection_.clear();
        instanceTessSelection_.resize(instanceCount, 0u);
        tessCandidates_.clear();
        // We gather all near LOD0 candidates before applying the cap.
        tessCandidates_.reserve(instanceCount);
      }

      const std::array<float, 3> squaredLodThresholds{
          sortedLodThresholds[0] * sortedLodThresholds[0],
          sortedLodThresholds[1] * sortedLodThresholds[1],
          sortedLodThresholds[2] * sortedLodThresholds[2],
      };
      const float cameraX = cameraPosition.x;
      const float cameraY = cameraPosition.y;
      const float cameraZ = cameraPosition.z;

      std::array<uint32_t, Submesh::kMaxLodCount> resolvedLodByRequested{};
      std::array<uint8_t, Submesh::kMaxLodCount> hasResolvedLod{};
      for (uint32_t lod = 0; lod < Submesh::kMaxLodCount; ++lod) {
        const auto resolved = resolveAvailableLod(submesh, lod);
        if (resolved) {
          resolvedLodByRequested[lod] = *resolved;
          hasResolvedLod[lod] = 1u;
        }
      }

      for (size_t i = 0; i < instanceCount; ++i) {
        const glm::vec4 lodCache = instanceLodCentersInvRadiusSq_[i];
        const float dx = cameraX - lodCache.x;
        const float dy = cameraY - lodCache.y;
        const float dz = cameraZ - lodCache.z;
        const float worldDistanceSq = dx * dx + dy * dy + dz * dz;
        const float normalizedDistanceSq = worldDistanceSq * lodCache.w;

        uint32_t requestedLod = 0;
        if (normalizedDistanceSq >= squaredLodThresholds[2]) {
          requestedLod = 3;
        } else if (normalizedDistanceSq >= squaredLodThresholds[1]) {
          requestedLod = 2;
        } else if (normalizedDistanceSq >= squaredLodThresholds[0]) {
          requestedLod = 1;
        }

        if (hasResolvedLod[requestedLod] == 0u) {
          continue;
        }

        const uint32_t resolvedLod = resolvedLodByRequested[requestedLod];
        instanceAutoLodLevels_[i] = resolvedLod;
        ++autoLodBucketCounts[resolvedLod];
        ++remapCount;
        if (tessellationRequested && resolvedLod == 0 &&
            worldDistanceSq <= tessFarDistanceSq) {
          tessCandidates_.push_back(TessCandidate{
              .distanceSq = worldDistanceSq,
              .instanceId = static_cast<uint32_t>(i),
          });
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
          if (instanceId >= instanceTessSelection_.size()) {
            continue;
          }
          instanceTessSelection_[instanceId] = 1u;
        }
        if (autoLodBucketCounts[0] >= autoLodTessBucketCount) {
          autoLodBucketCounts[0] -= autoLodTessBucketCount;
        } else {
          autoLodBucketCounts[0] = 0;
        }
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
          submesh.vertexOffset, templateEntry.submeshIndex,
          submeshMeshletMaxCount, templateEntry.vertexBuffer,
          templateEntry.vertexDecodeBuffer, templateEntry.vertexBufferAddress,
          templateEntry.vertexDecodeBufferAddress,
          templateEntry.vertexDecodeIndex, templateEntry.packedVertexFormat,
          templateEntry.materialIndex, templateEntry.meshletView,
          templateEntry.doubleSided, templateEntry.alphaMasked,
          autoLodBucketCounts[0], firstInstance);
      firstInstance += autoLodBucketCounts[0];

      autoLodTessBucketStart = firstInstance;
      autoLodTessBucketWrite = firstInstance;
      appendBatch(
          selectMeshPipeline(templateEntry.doubleSided, true),
          templateEntry.indexBuffer, templateEntry.indexBufferOffset, lod0Range,
          submesh.vertexOffset, templateEntry.submeshIndex,
          submeshMeshletMaxCount, templateEntry.vertexBuffer,
          templateEntry.vertexDecodeBuffer, templateEntry.vertexBufferAddress,
          templateEntry.vertexDecodeBufferAddress,
          templateEntry.vertexDecodeIndex, templateEntry.packedVertexFormat,
          templateEntry.materialIndex, templateEntry.meshletView,
          templateEntry.doubleSided, templateEntry.alphaMasked,
          autoLodTessBucketCount, firstInstance);
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
            lodRange, submesh.vertexOffset, templateEntry.submeshIndex,
            submeshMeshletMaxCount, templateEntry.vertexBuffer,
            templateEntry.vertexDecodeBuffer, templateEntry.vertexBufferAddress,
            templateEntry.vertexDecodeBufferAddress,
            templateEntry.vertexDecodeIndex, templateEntry.packedVertexFormat,
            templateEntry.materialIndex, templateEntry.meshletView,
            templateEntry.doubleSided, templateEntry.alphaMasked, count,
            firstInstance);
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
            lodRange, submesh.vertexOffset, templateEntry.submeshIndex,
            submeshMeshletMaxCount, templateEntry.vertexBuffer,
            templateEntry.vertexDecodeBuffer, templateEntry.vertexBufferAddress,
            templateEntry.vertexDecodeBufferAddress,
            templateEntry.vertexDecodeIndex, templateEntry.packedVertexFormat,
            templateEntry.materialIndex, templateEntry.meshletView,
            templateEntry.doubleSided, templateEntry.alphaMasked, count,
            firstInstance);
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
  if (!cpuMainCullingEnabled && !meshletRequested && !reusedStaticBatchCache &&
      !usedUniformFastPath && isSingleRenderableInstance &&
      !meshDrawTemplates_.empty() && !uniformSingleSubmeshPath_) {
    NURI_PROFILER_ZONE("OpaqueRenderer.batch_build_single_instance_cache",
                       NURI_PROFILER_COLOR_CMD_DRAW);

    const uint32_t requestedLod =
        resolveSingleInstanceRequestedLod(settings, forcedLod);
    const bool tessPipelineEnabled = shouldEnableSingleInstanceTessPipeline(
        tessellationRequested, requestedLod, cameraPosition, tessFarDistanceSq);
    auto singleInstanceCacheResult = ensureSingleInstanceBatchCache(
        requestedLod, tessPipelineEnabled, baseDraw);
    if (singleInstanceCacheResult.hasError()) {
      return singleInstanceCacheResult;
    }

    const size_t cacheIndex =
        singleInstanceCacheIndex(requestedLod, tessPipelineEnabled);
    if (cacheIndex >= singleInstanceBatchCaches_.size()) {
      return Result<bool, std::string>::makeError(
          "OpaqueRenderer::buildOpaquePasses: single-instance cache index out "
          "of "
          "range");
    }
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
    if (!templateEntry.submesh) {
      return Result<bool, std::string>::makeError(
          "OpaqueRenderer::buildOpaquePasses: invalid fast-path submesh");
    }
    const uint32_t requestedLod =
        settings.opaque.enableMeshLod ? forcedLod : 0u;
    const auto lodIndex =
        resolveAvailableLod(*templateEntry.submesh, requestedLod);
    if (lodIndex) {
      const SubmeshLod &lodRange = templateEntry.submesh->lods[*lodIndex];
      appendBatch(selectMeshPipeline(templateEntry.doubleSided, false),
                  templateEntry.indexBuffer, templateEntry.indexBufferOffset,
                  lodRange, templateEntry.submesh->vertexOffset,
                  templateEntry.submeshIndex,
                  maxMeshletCountForSubmesh(*templateEntry.submesh),
                  templateEntry.vertexBuffer, templateEntry.vertexDecodeBuffer,
                  templateEntry.vertexBufferAddress,
                  templateEntry.vertexDecodeBufferAddress,
                  templateEntry.vertexDecodeIndex,
                  templateEntry.packedVertexFormat, templateEntry.materialIndex,
                  templateEntry.meshletView, templateEntry.doubleSided,
                  templateEntry.alphaMasked, instanceCount, 0);
      remapCount = instanceCount;
      templateBatchIndices_.clear();
      templateBatchIndices_.resize(meshDrawTemplates_.size(), 0u);
      usedUniformFastPath = true;
    }
    NURI_PROFILER_ZONE_END();
  }

  if (!reusedStaticBatchCache && !usedUniformFastPath) {
    PmrHashMap<BatchKey, size_t, BatchKeyHash> batchLookup(
        batchScratch.resource());
    batchLookup.reserve(batchReserve);
    templateBatchIndices_.clear();
    templateBatchIndices_.resize(meshDrawTemplates_.size(), kInvalidBatchIndex);
    NURI_PROFILER_ZONE("OpaqueRenderer.batch_build",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    for (size_t templateIndex = 0; templateIndex < meshDrawTemplates_.size();
         ++templateIndex) {
      MeshDrawTemplate &templateEntry = meshDrawTemplates_[templateIndex];
      if (!templateEntry.renderable || !templateEntry.submesh) {
        return Result<bool, std::string>::makeError(
            "OpaqueRenderer::buildOpaquePasses: invalid mesh template");
      }
      if (cpuMainCullingEnabled &&
          (templateIndex >= visibleMainTemplates.size() ||
           visibleMainTemplates[templateIndex] == 0u)) {
        continue;
      }

      uint32_t requestedLod = 0;
      if (!settings.opaque.enableMeshLod) {
        requestedLod = 0;
      } else if (settings.opaque.forcedMeshLod >= 0) {
        requestedLod = forcedLod;
      } else if (meshletRequested) {
        requestedLod = 0;
      } else {
        if (templateEntry.instanceIndex >= instanceAutoLodLevels_.size()) {
          return Result<bool, std::string>::makeError(
              "OpaqueRenderer::buildOpaquePasses: instance LOD cache out of "
              "range");
        }
        requestedLod = instanceAutoLodLevels_[templateEntry.instanceIndex];
      }

      const auto lodIndex =
          resolveAvailableLod(*templateEntry.submesh, requestedLod);
      if (!lodIndex) {
        continue;
      }
      const SubmeshLod &lodRange = templateEntry.submesh->lods[*lodIndex];

      RenderPipelineHandle selectedPipeline =
          selectMeshPipeline(templateEntry.doubleSided, false);
      if (tessellationRequested && *lodIndex == 0 &&
          templateEntry.instanceIndex < instanceLodCentersInvRadiusSq_.size()) {
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
          .vertexDecodeBufferAddress = templateEntry.vertexDecodeBufferAddress,
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
        BatchEntry entry{};
        entry.draw = baseDraw;
        entry.draw.pipeline = selectedPipeline;
        entry.draw.indexBuffer = templateEntry.indexBuffer;
        entry.draw.indexBufferOffset = templateEntry.indexBufferOffset;
        entry.draw.indexCount = lodRange.indexCount;
        entry.draw.firstIndex = lodRange.indexOffset;
        entry.draw.vertexOffset = 0;
        entry.draw.alphaMasked = templateEntry.alphaMasked;
        entry.vertexBuffer = templateEntry.vertexBuffer;
        entry.vertexDecodeBuffer = templateEntry.vertexDecodeBuffer;
        entry.vertexBufferAddress = templateEntry.vertexBufferAddress;
        entry.vertexDecodeBufferAddress =
            templateEntry.vertexDecodeBufferAddress;
        entry.vertexDecodeIndex = templateEntry.vertexDecodeIndex;
        entry.packedVertexFormat = templateEntry.packedVertexFormat;
        entry.materialIndex = templateEntry.materialIndex;
        entry.meshletView = templateEntry.meshletView;
        entry.meshletOffset = lodRange.meshletOffset;
        entry.meshletCount = lodRange.meshletCount;
        entry.submeshIndex = templateEntry.submeshIndex;
        entry.meshletMaxCount =
            maxMeshletCountForSubmesh(*templateEntry.submesh);
        entry.vertexOffset = templateEntry.submesh->vertexOffset;
        entry.doubleSided = templateEntry.doubleSided;
        entry.alphaMasked = templateEntry.alphaMasked;
        batches.push_back(std::move(entry));
        const size_t insertedIndex = batches.size() - 1;
        auto [insertedIt, _] = batchLookup.emplace(key, insertedIndex);
        it = insertedIt;
      }

      const uint32_t batchIndex = static_cast<uint32_t>(it->second);
      templateBatchIndices_[templateIndex] = batchIndex;
      ++batches[it->second].instanceCount;
      ++remapCount;
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
    const bool shouldReuseFastAutoLodRemap =
        usedUniformAutoLodFastPath && reusedUniformAutoLodFastPath;
    const bool shouldBuildRemap = !shouldReuseFastAutoLodRemap;
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
        shouldBuildRemap && !singleRenderableInstance && !usedUniformFastPath;
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
      if (useCachedSingleInstanceBatches) {
        return Result<bool, std::string>::makeError(
            "OpaqueRenderer::buildOpaquePasses: cached single-instance batches "
            "used for multi-instance draw list");
      }
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
    if (shouldBuildRemap) {
      if (instanceRemap_.size() != remapCount) {
        instanceRemap_.resize(remapCount);
      }
    } else if (instanceRemap_.size() != remapCount) {
      return Result<bool, std::string>::makeError(
          "OpaqueRenderer::buildOpaquePasses: auto-LOD remap reuse mismatch");
    }
    if (shouldBuildRemap) {
      remapSignature = hashCombine64(kFnvOffsetBasis64, remapCount);
      remapSignatureValid = true;
    }

    const auto writeRemapEntry =
        [this,
         instanceCount](size_t writeOffset,
                        uint32_t instanceId) -> Result<bool, std::string> {
      if (writeOffset >= instanceRemap_.size()) {
        return Result<bool, std::string>::makeError(
            "OpaqueRenderer::buildOpaquePasses: instance remap write offset "
            "is out of range");
      }
      if (instanceId >= instanceCount) {
        return Result<bool, std::string>::makeError(
            "OpaqueRenderer::buildOpaquePasses: instance remap source index "
            "is out of range");
      }
      instanceRemap_[writeOffset] = instanceId;
      return Result<bool, std::string>::makeResult(true);
    };

    if (shouldBuildRemap && usedUniformAutoLodFastPath) {
      for (uint32_t lod = 0; lod < Submesh::kMaxLodCount; ++lod) {
        autoLodBucketWrites[lod] = autoLodBucketStarts[lod];
      }
      if (usedUniformAutoLodTessSplit) {
        autoLodTessBucketWrite = autoLodTessBucketStart;
        for (uint32_t instanceId = 0; instanceId < instanceCount;
             ++instanceId) {
          const uint32_t lod = instanceAutoLodLevels_[instanceId];
          if (lod == 0 && instanceTessSelection_[instanceId] != 0u) {
            auto writeResult =
                writeRemapEntry(autoLodTessBucketWrite++, instanceId);
            if (writeResult.hasError()) {
              return writeResult;
            }
            remapSignature = hashCombine64(remapSignature,
                                           static_cast<uint64_t>(instanceId));
            continue;
          }
          const size_t writeOffset = autoLodBucketWrites[lod]++;
          auto writeResult = writeRemapEntry(writeOffset, instanceId);
          if (writeResult.hasError()) {
            return writeResult;
          }
          remapSignature =
              hashCombine64(remapSignature, static_cast<uint64_t>(instanceId));
        }
      } else {
        for (uint32_t instanceId = 0; instanceId < instanceCount;
             ++instanceId) {
          const uint32_t lod = instanceAutoLodLevels_[instanceId];
          const size_t writeOffset = autoLodBucketWrites[lod]++;
          auto writeResult = writeRemapEntry(writeOffset, instanceId);
          if (writeResult.hasError()) {
            return writeResult;
          }
          remapSignature =
              hashCombine64(remapSignature, static_cast<uint64_t>(instanceId));
        }
      }
    } else if (shouldBuildRemap && singleRenderableInstance) {
      for (size_t i = 0; i < instanceRemap_.size(); ++i) {
        instanceRemap_[i] = 0u;
        remapSignature = hashCombine64(remapSignature, 0u);
      }
    } else if (shouldBuildRemap && usedUniformFastPath) {
      for (uint32_t instanceId = 0; instanceId < instanceCount; ++instanceId) {
        auto writeResult = writeRemapEntry(instanceId, instanceId);
        if (writeResult.hasError()) {
          return writeResult;
        }
        remapSignature =
            hashCombine64(remapSignature, static_cast<uint64_t>(instanceId));
      }
    } else if (shouldBuildRemap) {
      for (size_t templateIndex = 0; templateIndex < meshDrawTemplates_.size();
           ++templateIndex) {
        const uint32_t batchIndex = templateBatchIndices_[templateIndex];
        if (batchIndex == kInvalidBatchIndex) {
          continue;
        }
        const size_t writeOffset = batchWriteOffsets_[batchIndex]++;
        const uint32_t instanceId =
            meshDrawTemplates_[templateIndex].instanceIndex;
        auto writeResult = writeRemapEntry(writeOffset, instanceId);
        if (writeResult.hasError()) {
          return writeResult;
        }
        remapSignature =
            hashCombine64(remapSignature, static_cast<uint64_t>(instanceId));
      }
    }
    if (shouldBuildRemap) {
      cachedRemapSignature_ = remapSignature;
      cachedRemapSignatureValid_ = true;
    } else if (cachedRemapSignatureValid_) {
      remapSignature = cachedRemapSignature_;
      remapSignatureValid = true;
    }

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
            .meshletMaxCount = batch.meshletMaxCount,
            .vertexOffset = batch.vertexOffset,
            .doubleSided = batch.doubleSided,
            .alphaMasked = batch.alphaMasked,
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
    if (drawItems_.size() != drawPushConstants_.size()) {
      return Result<bool, std::string>::makeError(
          "OpaqueRenderer::buildOpaquePasses: draw and push constant count "
          "mismatch before instancing expansion");
    }

    size_t expandedDrawCount = 0;
    for (const DrawItem &draw : drawItems_) {
      expandedDrawCount += draw.instanceCount;
    }
    if (expandedDrawCount >
        static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
      return Result<bool, std::string>::makeError(
          "OpaqueRenderer::buildOpaquePasses: expanded draw count exceeds "
          "UINT32_MAX");
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
        if (sourceDraw.instanceCount == 0) {
          continue;
        }
        if (sourceDraw.instanceCount > 1 &&
            sourceDraw.firstInstance > (std::numeric_limits<uint32_t>::max() -
                                        (sourceDraw.instanceCount - 1u))) {
          return Result<bool, std::string>::makeError(
              "OpaqueRenderer::buildOpaquePasses: expanded instance range "
              "overflows UINT32_MAX");
        }

        for (uint32_t instanceOffset = 0;
             instanceOffset < sourceDraw.instanceCount; ++instanceOffset) {
          expandedPushConstants.push_back(sourceConstants);

          DrawItem expandedDraw = sourceDraw;
          expandedDraw.alphaMasked =
              i < drawAlphaMasked_.size() && drawAlphaMasked_[i] != 0u;
          expandedDraw.instanceCount = 1;
          expandedDraw.firstInstance =
              sourceDraw.firstInstance + instanceOffset;
          expandedDraw.pushConstants =
              std::span<const std::byte>(reinterpret_cast<const std::byte *>(
                                             &expandedPushConstants.back()),
                                         sizeof(PushConstants));
          expandedDrawItems.push_back(expandedDraw);
          expandedAlphaMasked.push_back(
              i < drawAlphaMasked_.size() ? drawAlphaMasked_[i] : 0u);
          expandedMeshletBatchInfos.push_back(i < meshletBatchInfos_.size()
                                                  ? meshletBatchInfos_[i]
                                                  : MeshletBatchInfo{});
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

  if (usedUniformAutoLodFastPath && !tessellationRequested) {
    updateFastAutoLodCache(activeFastAutoLodSubmesh, cameraPosition,
                           sortedLodThresholds, autoLodBucketCounts, remapCount,
                           instanceCount, frame.frameIndex);
  } else {
    invalidateAutoLodCache();
  }

  const uint64_t instanceRemapAddress = gpu_.getBufferDeviceAddress(
      instanceRemapRing_[frameSlot].buffer->handle());
  if (instanceRemapAddress == 0) {
    return Result<bool, std::string>::makeError(
        "OpaqueRenderer::buildOpaquePasses: invalid instance remap buffer "
        "address");
  }

  if (!instanceRemapUploadSource->empty()) {
    if (!remapSignatureValid) {
      remapSignature = computeRemapSignature(
          std::span<const uint32_t>(instanceRemapUploadSource->data(),
                                    instanceRemapUploadSource->size()));
      remapSignatureValid = true;
      cachedRemapSignature_ = remapSignature;
      cachedRemapSignatureValid_ = true;
    }
    const bool hasRemapSlotSignature =
        frameSlot < remapUploadSignatures_.size();
    const bool remapAlreadyUploadedForSlot =
        hasRemapSlotSignature &&
        remapUploadSignatures_[frameSlot] == remapSignature;
    if (!remapAlreadyUploadedForSlot) {
      NURI_PROFILER_ZONE("OpaqueRenderer.remap_upload",
                         NURI_PROFILER_COLOR_CMD_COPY);
      const std::span<const std::byte> remapBytes{
          reinterpret_cast<const std::byte *>(
              instanceRemapUploadSource->data()),
          instanceRemapUploadSource->size() * sizeof(uint32_t)};
      auto updateResult = gpu_.updateBuffer(
          instanceRemapRing_[frameSlot].buffer->handle(), remapBytes, 0);
      if (updateResult.hasError()) {
        return updateResult;
      }
      if (hasRemapSlotSignature) {
        remapUploadSignatures_[frameSlot] = remapSignature;
      }
      NURI_PROFILER_ZONE_END();
    }
  } else if (frameSlot < remapUploadSignatures_.size()) {
    remapUploadSignatures_[frameSlot] = kInvalidDrawSignature;
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
        animateInstances ||
        frameSlot >= instanceMatricesUploadVersions_.size() ||
        instanceMatricesUploadVersions_[frameSlot] != cachedTransformVersion_;
    if (needsInstanceMatricesUpload) {
      NURI_PROFILER_ZONE("OpaqueRenderer.instance_matrices_cpu",
                         NURI_PROFILER_COLOR_CMD_COPY);
      if (animateInstances) {
        instanceMatricesCpuCache_.resize(instanceCount);
        for (size_t i = 0; i < instanceCount; ++i) {
          const glm::vec3 center = glm::vec3(instanceCentersPhase_[i]);
          const glm::mat4 translation = glm::translate(glm::mat4(1.0f), center);
          instanceMatricesCpuCache_[i] =
              makeInstanceData(translation * instanceBaseMatrices_[i]);
        }
      } else if (instanceMatricesCpuCache_.size() != instanceCount) {
        return Result<bool, std::string>::makeError(
            "OpaqueRenderer::buildOpaquePasses: instance matrix cache size "
            "mismatch");
      }

      const std::span<const std::byte> matricesBytes{
          reinterpret_cast<const std::byte *>(instanceMatricesCpuCache_.data()),
          instanceMatricesCpuCache_.size() * sizeof(InstanceData)};
      auto updateResult = gpu_.updateBuffer(
          instanceMatricesRing_[frameSlot].buffer->handle(), matricesBytes, 0);
      if (updateResult.hasError()) {
        return updateResult;
      }
      if (!animateInstances &&
          frameSlot < instanceMatricesUploadVersions_.size()) {
        instanceMatricesUploadVersions_[frameSlot] = cachedTransformVersion_;
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

    if (animationSceneData != nullptr &&
        !animationSceneData->preDispatches.empty()) {
      preDispatches_.insert(preDispatches_.end(),
                            animationSceneData->preDispatches.begin(),
                            animationSceneData->preDispatches.end());
    }

    const bool hasIndirectDraws = !indirectDrawItems_.empty();
    if (nuri::isValid(sceneGpu->buffer)) {
      auto depResult = appendUniqueDependency(
          passDependencyBuffers_, passDependencyBufferAccessModes_,
          sceneGpu->buffer, RenderGraphAccessMode::Read,
          "OpaqueRenderer::buildOpaquePasses(pass)");
      if (depResult.hasError()) {
        return depResult;
      }
    }
    for (const BufferHandle materialHandle :
         {materialGpu->headerBuffer, materialGpu->clearcoatBuffer,
          materialGpu->sheenBuffer, materialGpu->transmissionBuffer,
          materialGpu->specularBuffer}) {
      auto depResult = appendUniqueDependency(
          passDependencyBuffers_, passDependencyBufferAccessModes_,
          materialHandle, RenderGraphAccessMode::Read,
          "OpaqueRenderer::buildOpaquePasses(pass)");
      if (depResult.hasError()) {
        return depResult;
      }
    }

    if (animationSceneData != nullptr) {
      for (const AnimatedRenderableGeometryOverride &geometryOverride :
           animationSceneData->geometryOverridesByRenderable) {
        if (!geometryOverride.enabled ||
            !nuri::isValid(geometryOverride.vertexBuffer)) {
          continue;
        }
        auto depResult = appendUniqueDependency(
            passDependencyBuffers_, passDependencyBufferAccessModes_,
            geometryOverride.vertexBuffer, RenderGraphAccessMode::Read,
            "OpaqueRenderer::buildOpaquePasses(animated geometry pass)");
        if (depResult.hasError()) {
          return depResult;
        }
      }
    }

    if (frameSlot < instanceRemapRing_.size() &&
        instanceRemapRing_[frameSlot].buffer &&
        instanceRemapRing_[frameSlot].buffer->valid()) {
      auto depResult = appendUniqueDependency(
          passDependencyBuffers_, passDependencyBufferAccessModes_,
          instanceRemapRing_[frameSlot].buffer->handle(),
          RenderGraphAccessMode::Read,
          "OpaqueRenderer::buildOpaquePasses(pass)");
      if (depResult.hasError()) {
        return depResult;
      }
    }

    if (hasIndirectDraws) {
      auto depResult = appendUniqueDependency(
          passDependencyBuffers_, passDependencyBufferAccessModes_,
          indirectCommandRing_[frameSlot].buffer->handle(),
          RenderGraphAccessMode::Read,
          "OpaqueRenderer::buildOpaquePasses(pass)");
      if (depResult.hasError()) {
        return depResult;
      }
    }

    if (instanceCount > 0) {
      auto passDepResult = appendUniqueDependency(
          passDependencyBuffers_, passDependencyBufferAccessModes_,
          instanceMatricesBufferHandle, RenderGraphAccessMode::Read,
          "OpaqueRenderer::buildOpaquePasses(pass)");
      if (passDepResult.hasError()) {
        return passDepResult;
      }

      if (useComputePass) {
        if (instanceCentersPhaseBuffer_ &&
            instanceCentersPhaseBuffer_->valid()) {
          auto depResult = appendUniqueDependency(
              dispatchDependencyBuffers_, instanceCentersPhaseBuffer_->handle(),
              "OpaqueRenderer::buildOpaquePasses(dispatch)");
          if (depResult.hasError()) {
            return depResult;
          }
        }
        if (instanceBaseMatricesBuffer_ &&
            instanceBaseMatricesBuffer_->valid()) {
          auto depResult = appendUniqueDependency(
              dispatchDependencyBuffers_, instanceBaseMatricesBuffer_->handle(),
              "OpaqueRenderer::buildOpaquePasses(dispatch)");
          if (depResult.hasError()) {
            return depResult;
          }
        }
        auto dispatchDepResult = appendUniqueDependency(
            dispatchDependencyBuffers_,
            instanceMatricesRing_[frameSlot].buffer->handle(),
            "OpaqueRenderer::buildOpaquePasses(dispatch)");
        if (dispatchDepResult.hasError()) {
          return dispatchDepResult;
        }

        const uint32_t dispatchX = static_cast<uint32_t>(
            (instanceCount + (kComputeWorkgroupSize - 1)) /
            kComputeWorkgroupSize);
        computeDispatchX = std::max(dispatchX, 1u);

        ComputeDispatchItem dispatch{};
        dispatch.pipeline = computePipelineHandle_;
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
  if (tessellationRequested && nuri::isValid(meshTessPipelineHandle_)) {
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
          std::pmr::vector<RenderGraphAccessMode> &accessModes,
          std::string_view context) -> Result<bool, std::string> {
    dependencies.clear();
    accessModes.clear();

    auto appendRead = [&](BufferHandle handle,
                          std::string_view dependencyContext) {
      return appendUniqueDependency(dependencies, accessModes, handle,
                                    RenderGraphAccessMode::Read,
                                    dependencyContext);
    };

    if (nuri::isValid(sceneGpu->buffer)) {
      auto depResult = appendRead(sceneGpu->buffer, context);
      if (depResult.hasError()) {
        return depResult;
      }
    }
    auto materialHeaderDepResult =
        appendRead(materialGpu->headerBuffer, context);
    if (materialHeaderDepResult.hasError()) {
      return materialHeaderDepResult;
    }

    if (animationSceneData != nullptr) {
      for (const AnimatedRenderableGeometryOverride &geometryOverride :
           animationSceneData->geometryOverridesByRenderable) {
        if (!geometryOverride.enabled ||
            !nuri::isValid(geometryOverride.vertexBuffer)) {
          continue;
        }
        auto depResult = appendRead(geometryOverride.vertexBuffer, context);
        if (depResult.hasError()) {
          return depResult;
        }
      }
    }

    if (frameSlot < instanceRemapRing_.size() &&
        instanceRemapRing_[frameSlot].buffer &&
        instanceRemapRing_[frameSlot].buffer->valid()) {
      auto depResult =
          appendRead(instanceRemapRing_[frameSlot].buffer->handle(), context);
      if (depResult.hasError()) {
        return depResult;
      }
    }

    if (hasIndirectBaseDraws) {
      auto depResult =
          appendRead(indirectCommandRing_[frameSlot].buffer->handle(), context);
      if (depResult.hasError()) {
        return depResult;
      }
    }

    if (instanceCount > 0) {
      auto depResult = appendRead(instanceMatricesBufferHandle, context);
      if (depResult.hasError()) {
        return depResult;
      }
    }
    return Result<bool, std::string>::makeResult(true);
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
    const bool hasIncompatibleFrameSettings =
        animationSceneData != nullptr || msaaSelected ||
        settings.opaque.enableTessellation || pendingPickRequest_.has_value() ||
        (debugVisualization != OpaqueDebugVisualization::None &&
         !meshletDebugRequested);
    bool missingMeshletData = meshletBatchInfos_.size() != drawItems_.size();
    for (size_t i = 0; i < meshletBatchInfos_.size() && !missingMeshletData;
         ++i) {
      const MeshletBatchInfo &info = meshletBatchInfos_[i];
      missingMeshletData =
          info.view == nullptr || info.meshletCount == 0u ||
          info.meshletMaxCount == 0u ||
          info.submeshIndex >= info.view->lodRangeCount ||
          !nuri::isValid(info.view->meshletBuffer) ||
          !nuri::isValid(info.view->meshletVertexIndexBuffer) ||
          !nuri::isValid(info.view->meshletPrimitiveIndexBuffer) ||
          !nuri::isValid(info.view->lodRangeBuffer);
    }

    if (hasIncompatibleFrameSettings) {
      frame.metrics.opaque.meshletRejectedIncompatibleFrame = 1u;
      if (meshletRequired) {
        return Result<bool, std::string>::makeError(
            "OpaqueRenderer::buildOpaquePasses: meshlet mode required but the "
            "current opaque frame uses unsupported passes or material modes");
      }
      if (!loggedMeshletIncompatibleWarning_) {
        loggedMeshletIncompatibleWarning_ = true;
        NURI_LOG_WARNING(
            "OpaqueRenderer::buildOpaquePasses: meshlet mode skipped because "
            "the frame uses unsupported opaque passes or material modes");
      }
    } else if (missingMeshletData) {
      frame.metrics.opaque.meshletRejectedMissingAssetData = 1u;
      if (meshletRequired) {
        return Result<bool, std::string>::makeError(
            "OpaqueRenderer::buildOpaquePasses: meshlet mode required but at "
            "least one opaque batch has no meshlet buffers");
      }
      if (!loggedMeshletAssetWarning_) {
        loggedMeshletAssetWarning_ = true;
        NURI_LOG_WARNING(
            "OpaqueRenderer::buildOpaquePasses: meshlet mode skipped because "
            "an opaque batch has no meshlet buffers");
      }
    } else {
      auto meshletPipelineResult = ensureMeshletPipelineState();
      if (meshletPipelineResult.hasError()) {
        frame.metrics.opaque.meshletRejectedMissingFeature = 1u;
        if (meshletRequired) {
          return meshletPipelineResult;
        }
        if (!loggedMeshletUnsupportedWarning_) {
          loggedMeshletUnsupportedWarning_ = true;
          NURI_LOG_WARNING(
              "OpaqueRenderer::buildOpaquePasses: meshlet mode skipped: %s",
              meshletPipelineResult.error().c_str());
        }
      } else {
        meshletActive = true;
      }
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

  const bool normalPrepassRequested =
      ambientOcclusionSettings.active && !msaaSelected &&
      nuri::isValid(frame.sharedResources.normalTexture) &&
      !wireframeOnlyRequested && !baseDrawItems.empty();
  const bool visibilityDepthPrepassRequested =
      visibilitySettings.enableOcclusionCulling;
  const bool requiresDepthPyramid = this->requiresDepthPyramid(settings);
  const bool meshletDepthPrepassRequested =
      settings.opaque.enableDepthPrepass || visibilityDepthPrepassRequested ||
      normalPrepassRequested || requiresDepthPyramid;
  bool depthPrepassEnabled =
      !meshletActive &&
      (settings.opaque.enableDepthPrepass || visibilityDepthPrepassRequested ||
       normalPrepassRequested || requiresDepthPyramid) &&
      !wireframeOnlyRequested && !baseDrawItems.empty();
  bool meshletDepthPrepassEnabled =
      meshletActive && meshletDepthPrepassRequested &&
      !wireframeOnlyRequested && !drawItems_.empty();
  if (depthPrepassEnabled && baseAlphaMasked.size() != baseDrawItems.size()) {
    depthPrepassEnabled = false;
  }
  bool useDepthPreparedMainDrawItems = false;
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
          selectDepthPipeline(source.pipeline, alphaMasked, msaaSelected);
      if (!nuri::isValid(depthPipeline)) {
        depthPrepassEnabled = false;
        depthPrepassDrawItems_.clear();
        passDrawItems_.clear();
        useDepthPreparedMainDrawItems = false;
        if (!loggedDepthPrepassUnsupported_) {
          loggedDepthPrepassUnsupported_ = true;
          NURI_LOG_WARNING(
              "OpaqueRenderer::buildOpaquePasses: depth pre-pass pipeline is "
              "unavailable, using single opaque pass");
        }
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

      DrawItem shadeDraw = source;
      shadeDraw.useDepthState = true;
      shadeDraw.depthState = {.compareOp = CompareOp::LessEqual,
                              .isDepthWriteEnabled = false};
      passDrawItems_.push_back(shadeDraw);
    }
    if (depthPrepassEnabled) {
      useDepthPreparedMainDrawItems = !passDrawItems_.empty();
      if (depthPrepassDrawItems_.empty()) {
        depthPrepassEnabled = false;
      }
    }
    NURI_PROFILER_ZONE_END();
  }
  bool transmissionVisibilityDepthEnabled =
      !meshletActive &&
      shouldBuildTransmissionVisibilityDepth(frame, settings) &&
      !wireframeOnlyRequested && !baseDrawItems.empty();
  if (transmissionVisibilityDepthEnabled &&
      baseAlphaMasked.size() != baseDrawItems.size()) {
    transmissionVisibilityDepthEnabled = false;
  }
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
      const RenderPipelineHandle depthPipeline =
          selectDepthPipeline(source.pipeline, alphaMasked, false);
      if (!nuri::isValid(depthPipeline) ||
          source.pushConstants.size() != sizeof(PushConstants)) {
        transmissionVisibilityDepthEnabled = false;
        transmissionVisibilityDepthDrawItems_.clear();
        transmissionVisibilityDepthPushConstants_.clear();
        frame.sharedResources.transmissionVisibilityDepthTexture = {};
        if (!loggedTransmissionVisibilityDepthUnsupported_) {
          loggedTransmissionVisibilityDepthUnsupported_ = true;
          NURI_LOG_WARNING(
              "OpaqueRenderer::buildOpaquePasses: transmission visibility "
              "depth is unavailable, falling back to jittered scene depth");
        }
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
    if (transmissionVisibilityDepthEnabled &&
        transmissionVisibilityDepthDrawItems_.empty()) {
      transmissionVisibilityDepthEnabled = false;
      frame.sharedResources.transmissionVisibilityDepthTexture = {};
    }
    NURI_PROFILER_ZONE_END();
  }
  std::span<const DrawItem> shadedBaseDrawItems =
      useDepthPreparedMainDrawItems
          ? std::span<const DrawItem>(passDrawItems_.data(),
                                      passDrawItems_.size())
          : baseDrawItems;
  bool meshletNormalPrepassEnabled =
      meshletActive && normalPrepassRequested && meshletDepthPrepassEnabled &&
      nuri::isValid(meshletNormalPipelineHandle_) &&
      nuri::isValid(meshletNormalDoubleSidedPipelineHandle_);
  const bool classicNormalPrepassWritesDepth =
      meshletActive && normalPrepassRequested && !meshletDepthPrepassEnabled;
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
          selectNormalPipeline(source.pipeline);
      if (!nuri::isValid(normalPipeline)) {
        normalPrepassDrawItems_.clear();
        normalPrepassEnabled = false;
        ambientOcclusionSettings.active = false;
        frame.sharedResources.ambientOcclusionTexture = {};
        frame.sharedResources.ambientOcclusionGraphTexture = {};
        aoMetrics.active = false;
        aoMetrics.disabledReason = AmbientOcclusionDisabledReason::Unsupported;
        if (!loggedNormalPrepassUnsupported_) {
          loggedNormalPrepassUnsupported_ = true;
          NURI_LOG_WARNING(
              "OpaqueRenderer::buildOpaquePasses: material normal pre-pass "
              "pipeline is unavailable, disabling GTAO for this frame");
        }
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
      normalDraw.debugLabel = "OpaqueMaterialNormals";
      normalDraw.debugColor = 0xff66ddff;
      normalPrepassDrawItems_.push_back(normalDraw);
    }
  }

  const MeshletLimits meshletLimits = gpu_.getMeshletLimits();
  if (meshletActive) {
    if (meshletLimits.maxTaskWorkGroupInvocations != 0u &&
        meshletLimits.maxTaskWorkGroupInvocations <
            kOpaqueMeshletTaskCandidatesPerGroup) {
      return Result<bool, std::string>::makeError(
          "OpaqueRenderer::buildOpaquePasses: meshlet task shader requires "
          "32 task invocations");
    }
    if (meshletLimits.maxTaskWorkGroupSizeX != 0u &&
        meshletLimits.maxTaskWorkGroupSizeX <
            kOpaqueMeshletTaskCandidatesPerGroup) {
      return Result<bool, std::string>::makeError(
          "OpaqueRenderer::buildOpaquePasses: meshlet task shader requires "
          "task workgroup size x >= 32");
    }
    if (meshletLimits.maxTaskPayloadBytes != 0u &&
        meshletLimits.maxTaskPayloadBytes < kOpaqueMeshletTaskPayloadBytes) {
      return Result<bool, std::string>::makeError(
          "OpaqueRenderer::buildOpaquePasses: meshlet task payload limit is "
          "below opaque payload size");
    }
  }
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

  const auto estimateMeshletBatchDescriptorCount =
      [&]() -> Result<size_t, std::string> {
    size_t descriptorCount = 0u;
    for (size_t i = 0; i < drawItems_.size(); ++i) {
      const DrawItem &draw = drawItems_[i];
      const MeshletBatchInfo &info = meshletBatchInfos_[i];
      if (info.view == nullptr || draw.instanceCount == 0u ||
          info.meshletCount == 0u) {
        continue;
      }
      const uint64_t candidateCount =
          static_cast<uint64_t>(info.meshletMaxCount) * draw.instanceCount;
      if (candidateCount > std::numeric_limits<uint32_t>::max()) {
        return Result<size_t, std::string>::makeError(
            "OpaqueRenderer::buildOpaquePasses: meshlet candidate count "
            "exceeds UINT32_MAX");
      }
      descriptorCount += static_cast<size_t>(
          (candidateCount + maxCandidateSpanPerDispatch - 1u) /
          maxCandidateSpanPerDispatch);
    }
    return Result<size_t, std::string>::makeResult(descriptorCount);
  };

  meshletBatchGpuData_.clear();
  BufferHandle meshletBatchBufferHandle{};
  uint64_t meshletBatchBufferAddress = 0u;
  if (meshletActive) {
    auto descriptorCountResult = estimateMeshletBatchDescriptorCount();
    if (descriptorCountResult.hasError()) {
      return Result<bool, std::string>::makeError(
          descriptorCountResult.error());
    }
    const size_t descriptorPassCount =
        1u + (meshletDepthPrepassEnabled ? 1u : 0u) +
        (meshletNormalPrepassEnabled ? 1u : 0u) + (taaSelected ? 2u : 0u);
    const size_t maxDescriptorCapacity =
        std::numeric_limits<uint32_t>::max() / sizeof(MeshletBatchGpuData);
    if (descriptorCountResult.value() >
        maxDescriptorCapacity / descriptorPassCount) {
      return Result<bool, std::string>::makeError(
          "OpaqueRenderer::buildOpaquePasses: meshlet batch descriptor count "
          "exceeds UINT32_MAX");
    }
    const size_t descriptorCapacity =
        descriptorCountResult.value() * descriptorPassCount;
    auto batchBufferResult = ensureMeshletBatchRingCapacity(
        std::max(descriptorCapacity * sizeof(MeshletBatchGpuData),
                 sizeof(MeshletBatchGpuData)));
    if (batchBufferResult.hasError()) {
      return batchBufferResult;
    }
    if (frameSlot >= meshletBatchRing_.size() ||
        !meshletBatchRing_[frameSlot].buffer ||
        !meshletBatchRing_[frameSlot].buffer->valid()) {
      return Result<bool, std::string>::makeError(
          "OpaqueRenderer::buildOpaquePasses: meshlet batch buffer is "
          "unavailable");
    }
    meshletBatchBufferHandle = meshletBatchRing_[frameSlot].buffer->handle();
    meshletBatchBufferAddress =
        gpu_.getBufferDeviceAddress(meshletBatchBufferHandle);
    if (meshletBatchBufferAddress == 0u) {
      return Result<bool, std::string>::makeError(
          "OpaqueRenderer::buildOpaquePasses: invalid meshlet batch buffer "
          "address");
    }
  }

  const bool meshletPreviousFrameDepthPyramidAvailable =
      frame.frameIndex > 0u &&
      frame.sharedResources.sceneDepthPyramidSourceFrameIndex.has_value() &&
      frame.sharedResources.sceneDepthPyramidSourceViewProj.has_value() &&
      previousDepthPyramidCameraStable(frame) &&
      *frame.sharedResources.sceneDepthPyramidSourceFrameIndex ==
          frame.frameIndex - 1u &&
      frame.sharedResources.sceneDepthPyramidLevelCount > 0u &&
      frame.sharedResources.sceneDepthSamplerId != kInvalidSamplerBindlessIndex;
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

  BufferHandle meshletVisibilityCounterBuffer{};
  uint64_t meshletVisibilityCounterBufferAddress = 0u;
  uint32_t meshletCounterFlags = 0u;
  if (meshletActive) {
    auto counterResult =
        ensureVisibilityCounterRingCapacity(sizeof(VisibilityCounterGpuData));
    if (counterResult.hasError()) {
      return counterResult;
    }
    if (frameSlot < visibilityCounterRing_.size() &&
        visibilityCounterRing_[frameSlot].buffer &&
        visibilityCounterRing_[frameSlot].buffer->valid()) {
      meshletVisibilityCounterBuffer =
          visibilityCounterRing_[frameSlot].buffer->handle();
      if (!visibilityCounterPreparedForFrame) {
        visibilityCounterClear_.clear();
        visibilityCounterClear_.push_back(VisibilityCounterGpuData{});
        auto clearCounterResult = gpu_.updateBuffer(
            meshletVisibilityCounterBuffer,
            std::as_bytes(std::span<const VisibilityCounterGpuData>(
                visibilityCounterClear_.data(),
                visibilityCounterClear_.size())),
            0u);
        if (clearCounterResult.hasError()) {
          return clearCounterResult;
        }
        if (frameSlot < visibilityCounterRingPublishedFrames_.size()) {
          visibilityCounterRingPublishedFrames_[frameSlot] = frame.frameIndex;
        }
        if (frameSlot < visibilityExpectedVisibleIndexCounts_.size()) {
          visibilityExpectedVisibleIndexCounts_[frameSlot] = 0u;
        }
        if (frameSlot < visibilityExpectedVisibleIndexHashes_.size()) {
          visibilityExpectedVisibleIndexHashes_[frameSlot] = kFnvOffsetBasis64;
        }
        visibilityCounterPreparedForFrame = true;
      }
      meshletVisibilityCounterBufferAddress =
          gpu_.getBufferDeviceAddress(meshletVisibilityCounterBuffer);
      if (meshletVisibilityCounterBufferAddress != 0u) {
        meshletCounterFlags |= kMeshletCounterFlagEnabled;
      }
    }
  }

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
          bool enablePreviousFrameOcclusion,
          uint64_t visibilityCounterBufferAddress,
          uint32_t meshletCounterFlagsForPass,
          uint64_t previousInstanceMatricesAddressForPass,
          uint64_t velocityInstanceFlagsAddressForPass,
          uint64_t velocityFrameDataAddressForPass,
          bool allowStaticMeshletDispatchCache)
      -> Result<uint64_t, std::string> {
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
                                      enablePreviousFrameOcclusion ? 1u : 0u);
    dispatchSignature = hashCombine64(dispatchSignature, maxTaskGroupsX);
    dispatchSignature =
        hashCombine64(dispatchSignature, kOpaqueMeshletTaskCandidatesPerGroup);

    const auto patchCachedDispatches =
        [&](uint64_t cachedCandidateCount) -> Result<uint64_t, std::string> {
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
      if (dispatchItems.size() != pushConstants.size() ||
          dispatchItems.size() != dispatchDependencyBuffers.size()) {
        return Result<uint64_t, std::string>::makeError(
            "OpaqueRenderer::buildOpaquePasses: cached meshlet dispatch size "
            "mismatch");
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
        constants.previousInstanceMatricesAddress =
            previousInstanceMatricesAddressForPass;
        constants.velocityInstanceFlagsAddress =
            velocityInstanceFlagsAddressForPass;
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
      return Result<uint64_t, std::string>::makeResult(cachedCandidateCount);
    };

    if (allowStaticMeshletDispatchCache &&
        staticBatchCache_.meshletDispatchCacheValid &&
        staticBatchCache_.meshletDispatchSignature == dispatchSignature) {
      return patchCachedDispatches(staticBatchCache_.meshletCandidateCount);
    }

    struct MeshletBuildSource {
      MeshletBatchGpuData batch{};
      std::array<BufferHandle, 6> dependencies{};
      MeshletPipelineHandle pipeline{};
      uint32_t candidateCount = 0u;
    };

    std::pmr::vector<MeshletBuildSource> sources(
        drawItems_.get_allocator().resource());
    sources.reserve(drawItems_.size());

    uint64_t totalCandidateCount = 0;
    uint32_t maxCandidateCount = 0u;
    for (size_t i = 0; i < drawItems_.size(); ++i) {
      const DrawItem &draw = drawItems_[i];
      const MeshletBatchInfo &info = meshletBatchInfos_[i];
      const Model::ModelMeshletGpuView *view = info.view;
      if (view == nullptr || draw.instanceCount == 0u ||
          info.meshletCount == 0u) {
        continue;
      }
      if (draw.pushConstants.size() != sizeof(PushConstants)) {
        return Result<uint64_t, std::string>::makeError(
            "OpaqueRenderer::buildOpaquePasses: invalid meshlet source push "
            "constants");
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
      if (meshletBufferAddress == 0u || meshletVertexIndexAddress == 0u ||
          meshletPrimitiveIndexAddress == 0u || meshletLodRangeAddress == 0u) {
        return Result<uint64_t, std::string>::makeError(
            "OpaqueRenderer::buildOpaquePasses: invalid meshlet buffer "
            "address");
      }

      uint32_t flags = 0u;
      const bool meshletGpuLod =
          settings.opaque.enableMeshLod && settings.opaque.forcedMeshLod < 0;
      flags |= meshletGpuLod ? kMeshletFlagGpuLod : 0u;
      flags |= visibilitySettings.enableMeshletFrustumCulling
                   ? kMeshletFlagFrustumCulling
                   : 0u;
      flags |= visibilitySettings.enableMeshletConeCulling
                   ? kMeshletFlagConeCulling
                   : 0u;
      flags |= debugVisualization == OpaqueDebugVisualization::MeshletId
                   ? kMeshletFlagDebugMeshletId
                   : 0u;
      flags |=
          debugVisualization == OpaqueDebugVisualization::MeshletSelectedLod
              ? kMeshletFlagDebugSelectedLod
              : 0u;
      flags |= enablePreviousFrameOcclusion ? kMeshletFlagOcclusionCulling : 0u;
      flags |= info.doubleSided ? kMeshletFlagDoubleSided : 0u;
      const uint32_t forcedMeshLod =
          settings.opaque.enableMeshLod
              ? std::min(forcedLod, Submesh::kMaxLodCount - 1u)
              : 0u;
      flags |= (forcedMeshLod & kMeshletFlagForcedLodMask)
               << kMeshletFlagForcedLodShift;

      const uint32_t candidateCount = static_cast<uint32_t>(
          static_cast<uint64_t>(info.meshletMaxCount) * draw.instanceCount);
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
                     info.submeshIndex, info.meshletMaxCount);
      batch.flags = glm::uvec4(flags, 0u, 0u, 0u);

      const bool useAlphaPipeline = info.alphaMasked &&
                                    nuri::isValid(alphaPipeline) &&
                                    nuri::isValid(alphaDoubleSidedPipeline);
      const MeshletPipelineHandle selectedPipeline =
          useAlphaPipeline
              ? (info.doubleSided ? alphaDoubleSidedPipeline : alphaPipeline)
              : (info.doubleSided ? doubleSidedPipeline : singleSidedPipeline);

      sources.push_back(MeshletBuildSource{
          .batch = batch,
          .dependencies = {draw.vertexBuffer, info.vertexDecodeBuffer,
                           view->meshletBuffer, view->meshletVertexIndexBuffer,
                           view->meshletPrimitiveIndexBuffer,
                           view->lodRangeBuffer},
          .pipeline = selectedPipeline,
          .candidateCount = candidateCount,
      });
    }

    if (sources.empty()) {
      return Result<uint64_t, std::string>::makeResult(totalCandidateCount);
    }

    const auto samePipeline = [](MeshletPipelineHandle lhs,
                                 MeshletPipelineHandle rhs) noexcept {
      return lhs.index == rhs.index && lhs.generation == rhs.generation;
    };
    const auto appendMeshletDependency =
        [](MeshletDispatchDependencyBuffers &dependencies,
           PmrHashSet<uint64_t> &dependencyKeys, BufferHandle handle,
           std::string_view context) -> Result<bool, std::string> {
      if (!nuri::isValid(handle)) {
        return Result<bool, std::string>::makeResult(true);
      }
      const uint64_t key = foldHandle(handle.index, handle.generation);
      if (!dependencyKeys.insert(key).second) {
        return Result<bool, std::string>::makeResult(true);
      }
      dependencies.buffers.push_back(handle);
      return Result<bool, std::string>::makeResult(true);
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
    const auto sourceMatches = [&](const MeshletBuildSource &source,
                                   MeshletPipelineHandle pipeline,
                                   uint32_t candidateOffset, uint32_t bucket) {
      if (!samePipeline(source.pipeline, pipeline) ||
          source.candidateCount <= candidateOffset) {
        return false;
      }
      const uint32_t taskGroupCount = meshletTaskGroupCount(
          sourceCandidateSpanForDispatch(source, candidateOffset));
      return meshletDispatchBucket(taskGroupCount) == bucket;
    };

    const MeshletPipelineHandle pipelines[4] = {
        singleSidedPipeline, doubleSidedPipeline, alphaPipeline,
        alphaDoubleSidedPipeline};
    std::pmr::vector<uint8_t> grouped(drawItems_.get_allocator().resource());
    grouped.resize(sources.size(), 0u);
    for (uint32_t candidateOffset = 0u; candidateOffset < maxCandidateCount;) {
      for (const MeshletPipelineHandle pipeline : pipelines) {
        if (!nuri::isValid(pipeline)) {
          continue;
        }
        for (uint32_t bucket = 0u; bucket < 32u; ++bucket) {
          std::fill(grouped.begin(), grouped.end(), 0u);
          for (size_t seedIndex = 0; seedIndex < sources.size(); ++seedIndex) {
            if (grouped[seedIndex] != 0u ||
                !sourceMatches(sources[seedIndex], pipeline, candidateOffset,
                               bucket)) {
              continue;
            }

            const size_t groupBatchBase = meshletBatchGpuData_.size();
            MeshletDispatchDependencyBuffers groupDependencies(
                dispatchDependencyBuffers.get_allocator().resource());
            PmrHashSet<uint64_t> groupDependencyKeys(
                dispatchDependencyBuffers.get_allocator().resource());
            groupDependencyKeys.reserve(std::min(
                sources.size() * 6u + 1u, kMaxMeshDispatchDependencyResources));
            auto batchDepResult = appendMeshletDependency(
                groupDependencies, groupDependencyKeys,
                meshletBatchBufferHandle,
                "OpaqueRenderer::buildOpaquePasses(meshlet dispatch)");
            if (batchDepResult.hasError()) {
              return Result<uint64_t, std::string>::makeError(
                  batchDepResult.error());
            }
            uint32_t groupBatchCount = 0u;
            uint32_t groupsX = 0u;
            for (size_t sourceIndex = seedIndex; sourceIndex < sources.size();
                 ++sourceIndex) {
              const MeshletBuildSource &source = sources[sourceIndex];
              if (grouped[sourceIndex] != 0u ||
                  !sourceMatches(source, pipeline, candidateOffset, bucket)) {
                continue;
              }
              const uint32_t taskGroupCount = meshletTaskGroupCount(
                  sourceCandidateSpanForDispatch(source, candidateOffset));
              if (meshletBatchGpuData_.size() >=
                  static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
                return Result<uint64_t, std::string>::makeError(
                    "OpaqueRenderer::buildOpaquePasses: meshlet batch index "
                    "exceeds UINT32_MAX");
              }
              grouped[sourceIndex] = 1u;
              meshletBatchGpuData_.push_back(source.batch);
              for (const BufferHandle handle : source.dependencies) {
                auto depResult = appendMeshletDependency(
                    groupDependencies, groupDependencyKeys, handle,
                    "OpaqueRenderer::buildOpaquePasses(meshlet dispatch)");
                if (depResult.hasError()) {
                  return Result<uint64_t, std::string>::makeError(
                      depResult.error());
                }
              }
              groupsX = std::max(groupsX, taskGroupCount);
              ++groupBatchCount;
            }

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
              constants.previousInstanceMatricesAddress =
                  previousInstanceMatricesAddressForPass;
              constants.velocityInstanceFlagsAddress =
                  velocityInstanceFlagsAddressForPass;
              constants.velocityFrameDataAddress =
                  velocityFrameDataAddressForPass;
              constants.batchBase =
                  static_cast<uint32_t>(groupBatchBase + emittedBatches);
              constants.candidateOffset = candidateOffset;
              constants.sourceFrameIndex =
                  static_cast<uint32_t>(frame.frameIndex);
              constants.meshletCounterFlags = meshletCounterFlagsForPass;
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

    return Result<uint64_t, std::string>::makeResult(totalCandidateCount);
  };

  if (meshletDepthPrepassEnabled) {
    NURI_PROFILER_ZONE("OpaqueRenderer.meshlet_depth_prepass_build",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    meshletDepthPrepassDependencyBuffers_ = passDependencyBuffers_;
    meshletDepthPrepassDependencyBufferAccessModes_ =
        passDependencyBufferAccessModes_;
    auto lodBoundsDepResult = appendUniqueDependency(
        meshletDepthPrepassDependencyBuffers_,
        meshletDepthPrepassDependencyBufferAccessModes_,
        instanceLodBoundsBuffer_->handle(), RenderGraphAccessMode::Read,
        "OpaqueRenderer::buildOpaquePasses(meshlet depth pre-pass)");
    if (lodBoundsDepResult.hasError()) {
      return lodBoundsDepResult;
    }

    auto meshletDepthBuildResult = buildMeshletDispatches(
        meshletDepthPrepassDispatchItems_, meshletDepthPrepassPushConstants_,
        meshletDepthPrepassDispatchDependencyBuffers_,
        meshletDepthPipelineHandle_, meshletDepthDoubleSidedPipelineHandle_,
        meshletDepthAlphaPipelineHandle_,
        meshletDepthAlphaDoubleSidedPipelineHandle_, CompareOp::Less, true,
        "OpaqueMeshletDepth", kMeshDebugColor, false, 0u, 0u, 0u, 0u, 0u,
        false);
    if (meshletDepthBuildResult.hasError()) {
      return Result<bool, std::string>::makeError(
          meshletDepthBuildResult.error());
    }
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
    auto lodBoundsDepResult = appendUniqueDependency(
        meshletNormalPrepassDependencyBuffers_,
        meshletNormalPrepassDependencyBufferAccessModes_,
        instanceLodBoundsBuffer_->handle(), RenderGraphAccessMode::Read,
        "OpaqueRenderer::buildOpaquePasses(meshlet normal pre-pass)");
    if (lodBoundsDepResult.hasError()) {
      return lodBoundsDepResult;
    }

    auto meshletNormalBuildResult = buildMeshletDispatches(
        meshletNormalPrepassDispatchItems_, meshletNormalPrepassPushConstants_,
        meshletNormalPrepassDispatchDependencyBuffers_,
        meshletNormalPipelineHandle_, meshletNormalDoubleSidedPipelineHandle_,
        {}, {}, CompareOp::LessEqual, false, "OpaqueMeshletNormals", 0xff66ddff,
        false, 0u, 0u, 0u, 0u, 0u, false);
    if (meshletNormalBuildResult.hasError()) {
      return Result<bool, std::string>::makeError(
          meshletNormalBuildResult.error());
    }
    if (meshletNormalPrepassDispatchItems_.empty()) {
      meshletNormalPrepassEnabled = false;
      meshletNormalPrepassDependencyBuffers_.clear();
      meshletNormalPrepassDependencyBufferAccessModes_.clear();
    }
    NURI_PROFILER_ZONE_END();
  }

  std::span<const DrawItem> finalPassDrawItems = shadedBaseDrawItems;
  if (!meshletActive && wireframeOnlyRequested && !baseDrawItems.empty()) {
    bool lineOverlayAvailable = false;
    bool lineTessOverlayAvailable = false;

    auto lineResult = ensureWireframePipeline(msaaSelected);
    if (lineResult.hasError()) {
      if (!loggedWireframeFallbackUnsupported_) {
        loggedWireframeFallbackUnsupported_ = true;
        NURI_LOG_WARNING("OpaqueRenderer::buildOpaquePasses: failed to create "
                         "wireframe pipeline: %s",
                         lineResult.error().c_str());
      }
    } else {
      lineOverlayAvailable = lineResult.value();
    }

    auto lineTessResult = ensureTessWireframePipeline(msaaSelected);
    if (lineTessResult.hasError()) {
      if (!loggedTessWireframeFallbackUnsupported_) {
        loggedTessWireframeFallbackUnsupported_ = true;
        NURI_LOG_WARNING("OpaqueRenderer::buildOpaquePasses: failed to create "
                         "tess wireframe pipeline: %s",
                         lineTessResult.error().c_str());
      }
    } else {
      lineTessOverlayAvailable = lineTessResult.value();
    }

    overlayDrawItems_.reserve(baseDrawItems.size());
    for (const DrawItem &baseItem : baseDrawItems) {
      const bool isTessDraw = isTessPipeline(baseItem.pipeline);
      RenderPipelineHandle wireframePipeline{};
      bool usedFallback = false;
      if (isTessDraw && lineTessOverlayAvailable) {
        wireframePipeline = msaaSelected ? meshMsaaTessWireframePipelineHandle_
                                         : meshTessWireframePipelineHandle_;
      } else if (lineOverlayAvailable) {
        wireframePipeline = msaaSelected ? meshMsaaWireframePipelineHandle_
                                         : meshWireframePipelineHandle_;
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
      bool gsOverlayAvailable = false;
      bool gsTessOverlayAvailable = false;
      bool lineOverlayAvailable = false;
      bool lineTessOverlayAvailable = false;

      auto gsOverlayResult = ensureGsOverlayPipeline(msaaSelected);
      if (gsOverlayResult.hasError()) {
        if (!loggedGsOverlayUnsupported_) {
          loggedGsOverlayUnsupported_ = true;
          NURI_LOG_WARNING(
              "OpaqueRenderer::buildOpaquePasses: failed to create "
              "GS overlay pipeline: %s",
              gsOverlayResult.error().c_str());
        }
      } else {
        gsOverlayAvailable = gsOverlayResult.value();
      }

      auto gsTessOverlayResult = ensureGsTessOverlayPipeline(msaaSelected);
      if (gsTessOverlayResult.hasError()) {
        if (!loggedGsTessOverlayUnsupported_) {
          loggedGsTessOverlayUnsupported_ = true;
          NURI_LOG_WARNING(
              "OpaqueRenderer::buildOpaquePasses: failed to create "
              "GS tess overlay pipeline: %s",
              gsTessOverlayResult.error().c_str());
        }
      } else {
        gsTessOverlayAvailable = gsTessOverlayResult.value();
      }

      if (!gsOverlayAvailable) {
        auto lineResult = ensureWireframePipeline(msaaSelected);
        if (lineResult.hasError()) {
          if (!loggedWireframeFallbackUnsupported_) {
            loggedWireframeFallbackUnsupported_ = true;
            NURI_LOG_WARNING("OpaqueRenderer::buildOpaquePasses: failed to "
                             "create line overlay fallback pipeline: %s",
                             lineResult.error().c_str());
          }
        } else {
          lineOverlayAvailable = lineResult.value();
        }
      }
      if (!gsTessOverlayAvailable) {
        auto lineTessResult = ensureTessWireframePipeline(msaaSelected);
        if (lineTessResult.hasError()) {
          if (!loggedTessWireframeFallbackUnsupported_) {
            loggedTessWireframeFallbackUnsupported_ = true;
            NURI_LOG_WARNING("OpaqueRenderer::buildOpaquePasses: failed to "
                             "create line tess overlay fallback pipeline: %s",
                             lineTessResult.error().c_str());
          }
        } else {
          lineTessOverlayAvailable = lineTessResult.value();
        }
      }

      overlayDrawItems_.reserve(baseDrawItems.size());
      for (const DrawItem &baseItem : baseDrawItems) {
        const bool isTessDraw = isTessPipeline(baseItem.pipeline);
        RenderPipelineHandle overlayPipeline{};
        bool usedFallback = false;
        if (isTessDraw) {
          if (gsTessOverlayAvailable) {
            overlayPipeline = msaaSelected
                                  ? meshMsaaGsTessOverlayPipelineHandle_
                                  : meshGsTessOverlayPipelineHandle_;
          } else if (lineTessOverlayAvailable) {
            overlayPipeline = msaaSelected
                                  ? meshMsaaTessWireframePipelineHandle_
                                  : meshTessWireframePipelineHandle_;
            usedFallback = true;
          }
        } else {
          if (gsOverlayAvailable) {
            overlayPipeline = msaaSelected ? meshMsaaGsOverlayPipelineHandle_
                                           : meshGsOverlayPipelineHandle_;
          } else if (lineOverlayAvailable) {
            overlayPipeline = msaaSelected ? meshMsaaWireframePipelineHandle_
                                           : meshWireframePipelineHandle_;
            usedFallback = true;
          }
        }

        if (!nuri::isValid(overlayPipeline)) {
          continue;
        }

        DrawItem overlayItem = baseItem;
        overlayItem.pipeline = overlayPipeline;
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
  frame.metrics.opaque.depthPrepassDraws = saturateToU32(
      depthPrepassDrawItems_.size() + meshletDepthPrepassDispatchItems_.size());
  frame.metrics.opaque.depthPrepassIndirectDraws =
      depthPrepassEnabled && hasIndirectBaseDraws
          ? saturateToU32(depthPrepassDrawItems_.size())
          : 0u;
  frame.metrics.opaque.depthPyramidLevels = 0u;
  frame.metrics.opaque.depthPrepassEnabled =
      (depthPrepassEnabled || meshletDepthPrepassEnabled ||
       classicNormalPrepassWritesDepth)
          ? 1u
          : 0u;
  aoMetrics.normalPrepassDraws =
      saturateToU32(normalPrepassDrawItems_.size() +
                    meshletNormalPrepassDispatchItems_.size());

  if (gpuMainCullingEnabled && !gpuVisibilityCandidateIndices.empty()) {
    VisibilityResolvedSettings mainVisibilitySettings = visibilitySettings;
    if (meshletActive) {
      mainVisibilitySettings.enableGpuIndirectDraw = false;
    }
    const size_t visibilityPassCountBefore = out.size();
    auto visibilityPassResult = appendGpuVisibilityMainPass(
        frame, frameSlot, visibilityCandidates, gpuVisibilityCandidateIndices,
        visibilityRequest, mainVisibilitySettings, out);
    if (visibilityPassResult.hasError()) {
      return visibilityPassResult;
    }
    visibilityCounterPreparedForFrame = out.size() != visibilityPassCountBefore;
  }

  bool pickPassSubmitted = false;
  if (pendingPickRequest_.has_value() && nuri::isValid(pickIdTexture_) &&
      nuri::isValid(meshPickPipelineHandle_)) {
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
        out.emplace_back(drawItems_.get_allocator().resource());
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
    pickPass.desc.dependencyBuffers = std::span<const BufferHandle>(
        passDependencyBuffers_.data(), passDependencyBuffers_.size());
    pickPass.desc.dependencyBufferAccessModes =
        std::span<const RenderGraphAccessMode>(
            passDependencyBufferAccessModes_.data(),
            passDependencyBufferAccessModes_.size());
    pickPass.desc.draws =
        std::span<const DrawItem>(pickDrawItems_.data(), pickDrawItems_.size());
    pickPass.desc.drawBuffersPreResolved = true;
    pickPass.desc.preResolvedDrawBuffers = std::span<const BufferHandle>(
        preResolvedDrawBuffers_.data(), preResolvedDrawBuffers_.size());
    pickPass.desc.debugLabel = kOpaquePickPassLabel;
    pickPass.desc.debugColor = kOpaquePassDebugColor;
    pickPass.hasDraws = !pickDrawItems_.empty();
    pickPass.hasPreDispatch = !preDispatches_.empty();
    pickPass.desc.borrowPayload = !pickPass.hasPreDispatch;
    pickPass.hasIndirectDraws = false;
    pickPass.isPickPass = true;

    inFlightPickReadback_ = InFlightPickReadback{
        .request = *pendingPickRequest_, .submissionFrame = frame.frameIndex};
    pendingPickRequest_.reset();
    pickPassSubmitted = true;
    NURI_PROFILER_ZONE_END();
  }

  if (depthPrepassEnabled) {
    PreparedGraphPass &depthPass =
        out.emplace_back(drawItems_.get_allocator().resource());
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
    depthPass.desc.dependencyBuffers = std::span<const BufferHandle>(
        passDependencyBuffers_.data(), passDependencyBuffers_.size());
    depthPass.desc.dependencyBufferAccessModes =
        std::span<const RenderGraphAccessMode>(
            passDependencyBufferAccessModes_.data(),
            passDependencyBufferAccessModes_.size());
    depthPass.desc.dependencyTextures = std::span<const TextureHandle>(
        passDependencyTextures_.data(), passDependencyTextures_.size());
    depthPass.desc.draws = std::span<const DrawItem>(
        depthPrepassDrawItems_.data(), depthPrepassDrawItems_.size());
    depthPass.desc.drawBuffersPreResolved = true;
    depthPass.desc.preResolvedDrawBuffers = std::span<const BufferHandle>(
        preResolvedDrawBuffers_.data(), preResolvedDrawBuffers_.size());
    depthPass.desc.debugLabel = "Opaque Depth Pre-Pass";
    depthPass.desc.debugColor = kOpaquePassDebugColor;
    depthPass.desc.gpuTimingScope = GpuTimingScope::Opaque;
    depthPass.desc.markImplicitOutputSideEffect = true;
    depthPass.hasDraws = !depthPrepassDrawItems_.empty();
    depthPass.hasPreDispatch = !pickPassSubmitted && !preDispatches_.empty();
    depthPass.desc.borrowPayload = !depthPass.hasPreDispatch;
    depthPass.hasIndirectDraws = hasIndirectBaseDraws;
    depthPass.isDepthPrepass = true;
  }

  if (meshletDepthPrepassEnabled) {
    PreparedGraphPass &depthPass =
        out.emplace_back(drawItems_.get_allocator().resource());
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
    depthPass.desc.dependencyBuffers = std::span<const BufferHandle>(
        meshletDepthPrepassDependencyBuffers_.data(),
        meshletDepthPrepassDependencyBuffers_.size());
    depthPass.desc.dependencyBufferAccessModes =
        std::span<const RenderGraphAccessMode>(
            meshletDepthPrepassDependencyBufferAccessModes_.data(),
            meshletDepthPrepassDependencyBufferAccessModes_.size());
    depthPass.desc.dependencyTextures = std::span<const TextureHandle>(
        passDependencyTextures_.data(), passDependencyTextures_.size());
    depthPass.desc.meshDispatches = std::span<const MeshDispatchItem>(
        meshletDepthPrepassDispatchItems_.data(),
        meshletDepthPrepassDispatchItems_.size());
    depthPass.desc.drawBuffersPreResolved = true;
    depthPass.desc.preResolvedDrawBuffers = std::span<const BufferHandle>(
        preResolvedDrawBuffers_.data(), preResolvedDrawBuffers_.size());
    depthPass.desc.debugLabel = "Opaque Meshlet Depth Pre-Pass";
    depthPass.desc.debugColor = kOpaquePassDebugColor;
    depthPass.desc.gpuTimingScope = GpuTimingScope::Opaque;
    depthPass.desc.markImplicitOutputSideEffect = true;
    depthPass.hasDraws = !meshletDepthPrepassDispatchItems_.empty();
    depthPass.hasPreDispatch = !pickPassSubmitted && !preDispatches_.empty();
    depthPass.desc.borrowPayload = !depthPass.hasPreDispatch;
    depthPass.hasIndirectDraws = false;
    depthPass.isDepthPrepass = true;
  }

  if (transmissionVisibilityDepthEnabled) {
    PreparedGraphPass &visibilityDepthPass =
        out.emplace_back(drawItems_.get_allocator().resource());
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
    visibilityDepthPass.desc.dependencyBuffers = std::span<const BufferHandle>(
        passDependencyBuffers_.data(), passDependencyBuffers_.size());
    visibilityDepthPass.desc.dependencyBufferAccessModes =
        std::span<const RenderGraphAccessMode>(
            passDependencyBufferAccessModes_.data(),
            passDependencyBufferAccessModes_.size());
    visibilityDepthPass.desc.dependencyTextures =
        std::span<const TextureHandle>(passDependencyTextures_.data(),
                                       passDependencyTextures_.size());
    visibilityDepthPass.desc.draws =
        std::span<const DrawItem>(transmissionVisibilityDepthDrawItems_.data(),
                                  transmissionVisibilityDepthDrawItems_.size());
    visibilityDepthPass.desc.drawBuffersPreResolved = true;
    visibilityDepthPass.desc.preResolvedDrawBuffers =
        std::span<const BufferHandle>(preResolvedDrawBuffers_.data(),
                                      preResolvedDrawBuffers_.size());
    visibilityDepthPass.desc.debugLabel =
        "Opaque Transmission Visibility Depth";
    visibilityDepthPass.desc.debugColor = kOpaquePassDebugColor;
    visibilityDepthPass.desc.gpuTimingScope = GpuTimingScope::Opaque;
    visibilityDepthPass.hasDraws =
        !transmissionVisibilityDepthDrawItems_.empty();
    visibilityDepthPass.hasPreDispatch =
        !pickPassSubmitted && !depthPrepassEnabled &&
        !meshletDepthPrepassEnabled && !preDispatches_.empty();
    visibilityDepthPass.desc.borrowPayload =
        !visibilityDepthPass.hasPreDispatch;
    visibilityDepthPass.hasIndirectDraws = hasIndirectBaseDraws;
    visibilityDepthPass.isTransmissionVisibilityDepthPass = true;
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
      nuri::isValid(frame.sharedResources.normalTexture)) {
    PreparedGraphPass &normalPass =
        out.emplace_back(drawItems_.get_allocator().resource());
    normalPass.desc.color = {.loadOp = LoadOp::Clear,
                             .storeOp = StoreOp::Store,
                             .clearColor = kFrameCompositionNormalClearValue};
    normalPass.colorTextureHandle = frame.sharedResources.normalTexture;
    normalPass.desc.depth = {.loadOp = LoadOp::Load,
                             .storeOp = StoreOp::Store,
                             .clearDepth = kClearDepthOne,
                             .clearStencil = 0};
    normalPass.depthTextureHandle = sceneDepthTarget;
    normalPass.desc.dependencyBuffers = std::span<const BufferHandle>(
        meshletNormalPrepassDependencyBuffers_.data(),
        meshletNormalPrepassDependencyBuffers_.size());
    normalPass.desc.dependencyBufferAccessModes =
        std::span<const RenderGraphAccessMode>(
            meshletNormalPrepassDependencyBufferAccessModes_.data(),
            meshletNormalPrepassDependencyBufferAccessModes_.size());
    normalPass.desc.dependencyTextures = std::span<const TextureHandle>(
        passDependencyTextures_.data(), passDependencyTextures_.size());
    normalPass.desc.meshDispatches = std::span<const MeshDispatchItem>(
        meshletNormalPrepassDispatchItems_.data(),
        meshletNormalPrepassDispatchItems_.size());
    normalPass.desc.debugLabel = "Opaque Meshlet Material Normal Pre-Pass";
    normalPass.desc.debugColor = 0xff66ddff;
    normalPass.desc.gpuTimingScope = GpuTimingScope::Opaque;
    normalPass.desc.markImplicitOutputSideEffect = true;
    normalPass.hasDraws = true;
    normalPass.hasPreDispatch = false;
    normalPass.desc.borrowPayload = true;
    normalPass.hasIndirectDraws = false;
    normalPass.isNormalPrepass = true;
    aoMetrics.normalPrepassDraws =
        saturateToU32(meshletNormalPrepassDispatchItems_.size());
  } else if (hasClassicNormalPrepass &&
             nuri::isValid(frame.sharedResources.normalTexture)) {
    PreparedGraphPass &normalPass =
        out.emplace_back(drawItems_.get_allocator().resource());
    normalPass.desc.color = {.loadOp = LoadOp::Clear,
                             .storeOp = StoreOp::Store,
                             .clearColor = kFrameCompositionNormalClearValue};
    normalPass.colorTextureHandle = frame.sharedResources.normalTexture;
    normalPass.desc.depth = {
        .loadOp = hasClassicNormalDepthPrepass ? LoadOp::Clear : LoadOp::Load,
        .storeOp = StoreOp::Store,
        .clearDepth = kClearDepthOne,
        .clearStencil = 0};
    normalPass.depthTextureHandle = sceneDepthTarget;
    if (hasClassicNormalDepthPrepass) {
      if (!preDispatchSubmittedBeforeNormal) {
        normalPass.desc.preDispatches = std::span<const ComputeDispatchItem>(
            preDispatches_.data(), preDispatches_.size());
      }
    }
    normalPass.desc.dependencyBuffers = std::span<const BufferHandle>(
        passDependencyBuffers_.data(), passDependencyBuffers_.size());
    normalPass.desc.dependencyBufferAccessModes =
        std::span<const RenderGraphAccessMode>(
            passDependencyBufferAccessModes_.data(),
            passDependencyBufferAccessModes_.size());
    normalPass.desc.dependencyTextures = std::span<const TextureHandle>(
        passDependencyTextures_.data(), passDependencyTextures_.size());
    normalPass.desc.draws = std::span<const DrawItem>(
        normalPrepassDrawItems_.data(), normalPrepassDrawItems_.size());
    normalPass.desc.drawBuffersPreResolved = true;
    normalPass.desc.preResolvedDrawBuffers = std::span<const BufferHandle>(
        preResolvedDrawBuffers_.data(), preResolvedDrawBuffers_.size());
    normalPass.desc.debugLabel = "Opaque Material Normal Pre-Pass";
    normalPass.desc.debugColor = 0xff66ddff;
    normalPass.desc.gpuTimingScope = GpuTimingScope::Opaque;
    normalPass.desc.markImplicitOutputSideEffect = true;
    normalPass.hasDraws = true;
    normalPass.hasPreDispatch = hasClassicNormalDepthPrepass &&
                                !preDispatchSubmittedBeforeNormal &&
                                !preDispatches_.empty();
    normalPass.desc.borrowPayload = !normalPass.hasPreDispatch;
    normalPass.hasIndirectDraws =
        hasClassicNormalPrepass && hasIndirectBaseDraws;
    normalPass.isDepthPrepass = hasClassicNormalDepthPrepass;
    normalPass.isNormalPrepass = true;
    aoMetrics.normalPrepassDraws =
        saturateToU32(normalPrepassDrawItems_.size());
  } else if (ambientOcclusionSettings.active) {
    aoMetrics.active = false;
    aoMetrics.disabledReason = AmbientOcclusionDisabledReason::MissingResources;
  }

  const bool preDispatchSubmittedBeforeMain =
      pickPassSubmitted || depthPrepassEnabled || meshletDepthPrepassEnabled ||
      transmissionVisibilityDepthEnabled || hasClassicNormalDepthPrepass;
  const bool sceneDepthAvailableForPyramid =
      pickPassSubmitted || depthPrepassEnabled || meshletDepthPrepassEnabled ||
      hasClassicNormalDepthPrepass;

  const bool depthPyramidEnabled =
      !msaaSelected && requiresDepthPyramid && sceneDepthAvailableForPyramid &&
      nuri::isValid(sceneDepthTexture) &&
      nuri::isValid(depthPyramidPipelineHandle_) &&
      sceneDepthPyramidLevelCount_ > 0u &&
      (meshletActive ? !drawItems_.empty() : !baseDrawItems.empty());
  frame.metrics.shadow.sdsmComputePassCount = 0u;
  if (depthPyramidEnabled) {
    NURI_PROFILER_ZONE("OpaqueRenderer.depth_pyramid_build",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    depthPyramidPushConstants_.clear();
    depthPyramidDrawItems_.clear();
    depthPyramidDependencyTextures_.clear();
    depthPyramidPushConstants_.reserve(sceneDepthPyramidLevelCount_);
    depthPyramidDrawItems_.reserve(sceneDepthPyramidLevelCount_);
    depthPyramidDependencyTextures_.reserve(sceneDepthPyramidLevelCount_);
    const uint32_t samplerId =
        nuri::isValid(sceneDepthSampler_)
            ? gpu_.getSamplerBindlessIndex(sceneDepthSampler_)
            : gpu_.getDefaultSamplerBindlessIndex();
    for (uint32_t level = 0u; level < sceneDepthPyramidLevelCount_; ++level) {
      const TextureHandle sourceTexture =
          level == 0u ? sceneDepthTexture
                      : sceneDepthPyramidTextures_[level - 1u];
      const TextureHandle destinationTexture =
          sceneDepthPyramidTextures_[level];
      if (!nuri::isValid(sourceTexture) || !nuri::isValid(destinationTexture)) {
        break;
      }
      const uint32_t sourceTexId = gpu_.getTextureBindlessIndex(sourceTexture);
      if (sourceTexId == kInvalidTextureBindlessIndex) {
        break;
      }
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
                                .clearColor = {1.0f, 0.0f, 0.0f, 0.0f}};
      pyramidPass.colorTextureHandle = destinationTexture;
      pyramidPass.desc.dependencyTextures = std::span<const TextureHandle>(
          &depthPyramidDependencyTextures_.back(), 1u);
      pyramidPass.desc.draws =
          std::span<const DrawItem>(&depthPyramidDrawItems_.back(), 1u);
      pyramidPass.desc.drawBuffersPreResolved = true;
      pyramidPass.desc.debugLabel = "Opaque Depth MinMax Pyramid";
      pyramidPass.desc.debugColor = kOpaquePassDebugColor;
      pyramidPass.hasPreDispatch =
          false; // No pre-dispatches for pyramid passes.
      pyramidPass.desc.borrowPayload = !pyramidPass.hasPreDispatch;
      pyramidPass.hasDraws = true;
      pyramidPass.isDepthPyramidPass = true;
      pyramidPass.depthPyramidLevel = level;
    }
    frame.metrics.opaque.depthPyramidLevels =
        saturateToU32(depthPyramidDrawItems_.size());
    const uint32_t builtPyramidLevelCount =
        saturateToU32(depthPyramidDrawItems_.size());
    const bool pyramidBuildComplete =
        builtPyramidLevelCount == sceneDepthPyramidLevelCount_;
    if (pyramidBuildComplete && sceneDepthAvailableForPyramid) {
      sceneDepthPyramidSourceFrameIndex_ = frame.frameIndex;
      sceneDepthPyramidSourceViewProj_ = frame.camera.currentUnjitteredViewProj;
    } else {
      sceneDepthPyramidSourceFrameIndex_.reset();
      sceneDepthPyramidSourceViewProj_.reset();
    }
    const ShadowSdsmMode sdsmMode =
        sanitizeShadowSdsmMode(settings.shadow.sdsmMode);
    if (settings.shadow.enabled &&
        (sdsmMode == ShadowSdsmMode::PreviousFrameMinMax ||
         sdsmMode == ShadowSdsmMode::Histogram) &&
        builtPyramidLevelCount > 0u &&
        frame.sharedResources.shadowSdsmGpuReduceTarget.has_value() &&
        nuri::isValid(frame.sharedResources.shadowSdsmGpuReducePipeline)) {
      uint32_t reduceSourceLevel = builtPyramidLevelCount - 1u;
      glm::uvec2 histogramSourceDimensions{1u};
      if (sdsmMode == ShadowSdsmMode::Histogram) {
        std::array<glm::uvec2, kMaxSceneDepthPyramidLevels> levelDimensions{};
        for (uint32_t level = 0u; level < builtPyramidLevelCount; ++level) {
          const TextureDimensions dimensions =
              gpu_.getTextureDimensions(sceneDepthPyramidTextures_[level]);
          levelDimensions[level] = glm::uvec2(std::max(dimensions.width, 1u),
                                              std::max(dimensions.height, 1u));
        }
        const shadow_detail::ShadowSdsmHistogramSourceSelection selection =
            shadow_detail::selectSdsmHistogramSourceLevel(
                std::span<const glm::uvec2>(levelDimensions.data(),
                                            builtPyramidLevelCount),
                builtPyramidLevelCount, 4096u);
        reduceSourceLevel = selection.level;
        histogramSourceDimensions = selection.dimensions;
      }
      const TextureHandle reduceSourceTexture =
          sceneDepthPyramidTextures_[reduceSourceLevel];
      const uint32_t reduceSourceTexId =
          gpu_.getTextureBindlessIndex(reduceSourceTexture);
      if (nuri::isValid(reduceSourceTexture) &&
          reduceSourceTexId != kInvalidTextureBindlessIndex) {
        shadowSdsmReducePushConstants_.clear();
        shadowSdsmHistogramReducePushConstants_.clear();
        shadowSdsmReduceDispatches_.clear();
        shadowSdsmReduceDependencyBuffers_.clear();
        shadowSdsmReduceDependencyTextures_.clear();
        shadowSdsmReducePushConstants_.reserve(1u);
        shadowSdsmHistogramReducePushConstants_.reserve(1u);
        shadowSdsmReduceDispatches_.reserve(1u);
        shadowSdsmReduceDependencyBuffers_.reserve(1u);
        shadowSdsmReduceDependencyTextures_.reserve(1u);

        std::span<const std::byte> reducePushConstants;
        if (sdsmMode == ShadowSdsmMode::Histogram) {
          const float fixedNear = std::max(frame.camera.nearPlane, 0.01f);
          const float fixedFar = std::max(
              fixedNear + 0.01f,
              std::min(std::max(settings.shadow.maxDistance, fixedNear + 0.01f),
                       std::max(frame.camera.farPlane, fixedNear + 0.01f)));
          shadowSdsmHistogramReducePushConstants_.push_back(
              ShadowSdsmHistogramReducePushConstants{
                  .resultBufferAddress =
                      frame.sharedResources.shadowSdsmGpuReduceTarget
                          ->bufferAddress,
                  .sourceParams = glm::uvec4(
                      reduceSourceTexId,
                      static_cast<uint32_t>(frame.frameIndex),
                      histogramSourceDimensions.x, histogramSourceDimensions.y),
                  .histogramParams = glm::uvec4(
                      std::clamp(settings.shadow.sdsmHistogramBucketCount,
                                 kMinShadowSdsmHistogramBucketCount,
                                 kMaxShadowSdsmHistogramBucketCount),
                      std::clamp(settings.shadow.cascadeCount, 1u,
                                 kMaxShadowCascades),
                      frame.camera.projectionType ==
                              ProjectionType::Orthographic
                          ? 1u
                          : 0u,
                      0u),
                  .cameraParams =
                      glm::vec4(frame.camera.nearPlane, frame.camera.farPlane,
                                fixedNear, fixedFar),
                  .trimParams =
                      glm::vec4(settings.shadow.sdsmHistogramTrimLowPercent,
                                settings.shadow.sdsmHistogramTrimHighPercent,
                                1.0e-4f, 0.0f),
              });
          reducePushConstants = std::span<const std::byte>(
              reinterpret_cast<const std::byte *>(
                  &shadowSdsmHistogramReducePushConstants_.back()),
              sizeof(ShadowSdsmHistogramReducePushConstants));
        } else {
          shadowSdsmReducePushConstants_.push_back(
              ShadowSdsmReducePushConstants{
                  .resultBufferAddress =
                      frame.sharedResources.shadowSdsmGpuReduceTarget
                          ->bufferAddress,
                  .sourceTexId = reduceSourceTexId,
                  .sourceFrameIndex = static_cast<uint32_t>(frame.frameIndex),
              });
          reducePushConstants = std::span<const std::byte>(
              reinterpret_cast<const std::byte *>(
                  &shadowSdsmReducePushConstants_.back()),
              sizeof(ShadowSdsmReducePushConstants));
        }
        shadowSdsmReduceDependencyBuffers_.push_back(
            frame.sharedResources.shadowSdsmGpuReduceTarget->buffer);
        shadowSdsmReduceDependencyTextures_.push_back(reduceSourceTexture);

        ComputeDispatchItem dispatch{};
        dispatch.pipeline = frame.sharedResources.shadowSdsmGpuReducePipeline;
        dispatch.dispatch = {.x = 1u, .y = 1u, .z = 1u};
        dispatch.pushConstants = reducePushConstants;
        dispatch.dependencyBuffers = std::span<const BufferHandle>(
            shadowSdsmReduceDependencyBuffers_.data(),
            shadowSdsmReduceDependencyBuffers_.size());
        dispatch.dependencyTextures = std::span<const TextureHandle>(
            shadowSdsmReduceDependencyTextures_.data(),
            shadowSdsmReduceDependencyTextures_.size());
        dispatch.debugLabel = sdsmMode == ShadowSdsmMode::Histogram
                                  ? "Shadow SDSM Histogram Reduce"
                                  : "Shadow SDSM Reduce";
        dispatch.debugColor = kComputeDispatchColor;
        shadowSdsmReduceDispatches_.push_back(dispatch);

        PreparedGraphPass &reducePass =
            out.emplace_back(drawItems_.get_allocator().resource());
        reducePass.desc.executionMode = RenderPassExecutionMode::ComputeOnly;
        reducePass.desc.hasColorAttachment = false;
        reducePass.desc.preDispatches = std::span<const ComputeDispatchItem>(
            shadowSdsmReduceDispatches_.data(),
            shadowSdsmReduceDispatches_.size());
        reducePass.desc.dependencyTextures = std::span<const TextureHandle>(
            shadowSdsmReduceDependencyTextures_.data(),
            shadowSdsmReduceDependencyTextures_.size());
        reducePass.desc.gpuTimingScope = GpuTimingScope::ShadowSdsm;
        reducePass.desc.debugLabel = dispatch.debugLabel;
        reducePass.desc.debugColor = kComputeDispatchColor;
        reducePass.desc.markImplicitOutputSideEffect = true;
        reducePass.desc.borrowPayload = false;
        reducePass.hasPreDispatch = true;
        reducePass.hasDraws = false;
        frame.metrics.shadow.sdsmComputePassCount = 1u;
      } else if (!loggedShadowSdsmReduceSkipWarning_) {
        loggedShadowSdsmReduceSkipWarning_ = true;
        NURI_LOG_WARNING(
            "OpaqueRenderer::buildOpaquePasses: skipping Shadow SDSM Reduce "
            "frame=%llu sourceTextureValid=%u sourceTexId=%u levelCount=%u",
            static_cast<unsigned long long>(frame.frameIndex),
            nuri::isValid(reduceSourceTexture) ? 1u : 0u, reduceSourceTexId,
            sceneDepthPyramidLevelCount_);
      }
    } else if (settings.shadow.enabled &&
               (sdsmMode == ShadowSdsmMode::PreviousFrameMinMax ||
                sdsmMode == ShadowSdsmMode::Histogram) &&
               !loggedShadowSdsmReduceSkipWarning_) {
      loggedShadowSdsmReduceSkipWarning_ = true;
      NURI_LOG_WARNING(
          "OpaqueRenderer::buildOpaquePasses: Shadow SDSM Reduce unavailable "
          "frame=%llu hasReduceTarget=%u hasReducePipeline=%u",
          static_cast<unsigned long long>(frame.frameIndex),
          frame.sharedResources.shadowSdsmGpuReduceTarget.has_value() ? 1u : 0u,
          nuri::isValid(frame.sharedResources.shadowSdsmGpuReducePipeline)
              ? 1u
              : 0u);
    }
    NURI_PROFILER_ZONE_END();
  } else {
    sceneDepthPyramidSourceFrameIndex_.reset();
    sceneDepthPyramidSourceViewProj_.reset();
  }

  const bool shouldLoadColor = settings.skybox.enabled;
  NURI_PROFILER_ZONE("OpaqueRenderer.main_pass_finalize",
                     NURI_PROFILER_COLOR_CMD_DRAW);
  PreparedGraphPass &pass =
      out.emplace_back(drawItems_.get_allocator().resource());
  pass.desc.color = {.loadOp = shouldLoadColor ? LoadOp::Load : LoadOp::Clear,
                     .storeOp = StoreOp::Store,
                     .clearColor = {kClearColorWhite, kClearColorWhite,
                                    kClearColorWhite, kClearColorWhite}};
  pass.desc.depth = {.loadOp =
                         (depthPrepassEnabled || meshletDepthPrepassEnabled ||
                          hasClassicNormalDepthPrepass)
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
    auto counterDepResult = appendUniqueDependency(
        mainPassDependencyBuffers_, mainPassDependencyBufferAccessModes_,
        meshletVisibilityCounterBuffer,
        RenderGraphAccessMode::Read | RenderGraphAccessMode::Write,
        "OpaqueRenderer::buildOpaquePasses(meshlet counters)");
    if (counterDepResult.hasError()) {
      return counterDepResult;
    }
  }
  if ((sceneGpu->shadowFlags & kShadowFrameFlagEnabled) != 0u &&
      frame.sharedResources.shadowFrameGpuData.has_value()) {
    auto shadowBufferDepResult = appendUniqueDependency(
        mainPassDependencyBuffers_, mainPassDependencyBufferAccessModes_,
        frame.sharedResources.shadowFrameGpuData->buffer,
        RenderGraphAccessMode::Read,
        "OpaqueRenderer::buildOpaquePasses(main shadow pass)");
    if (shadowBufferDepResult.hasError()) {
      return shadowBufferDepResult;
    }

    for (uint32_t cascadeIndex = 0u; cascadeIndex < kMaxShadowCascades;
         ++cascadeIndex) {
      const TextureHandle texture =
          frame.sharedResources.shadowCascadeTextures[cascadeIndex];
      if (!nuri::isValid(texture)) {
        continue;
      }
      const size_t oldTextureDependencyCount =
          mainPassDependencyTextures_.size();
      appendUniqueTextureDependency(mainPassDependencyTextures_, texture);
      if (mainPassDependencyTextures_.size() != oldTextureDependencyCount) {
        mainPassDependencyTextureAccessModes_.push_back(
            RenderGraphAccessMode::Read);
      }
    }
  }
  if (ambientOcclusionSettings.active &&
      nuri::isValid(frame.sharedResources.ambientOcclusionTexture)) {
    const size_t oldTextureDependencyCount = mainPassDependencyTextures_.size();
    appendUniqueTextureDependency(
        mainPassDependencyTextures_,
        frame.sharedResources.ambientOcclusionTexture);
    if (mainPassDependencyTextures_.size() != oldTextureDependencyCount) {
      mainPassDependencyTextureAccessModes_.push_back(
          RenderGraphAccessMode::Read);
    }
  }
  if (meshletPreviousFrameOcclusionAvailable) {
    const uint32_t depthPyramidLevelCount =
        std::min<uint32_t>(frame.sharedResources.sceneDepthPyramidLevelCount,
                           kMaxSceneDepthPyramidLevels);
    for (uint32_t level = 0u; level < depthPyramidLevelCount; ++level) {
      const TextureHandle texture =
          frame.sharedResources.sceneDepthPyramidTextures[level];
      const size_t oldTextureDependencyCount =
          mainPassDependencyTextures_.size();
      appendUniqueTextureDependency(mainPassDependencyTextures_, texture);
      if (mainPassDependencyTextures_.size() != oldTextureDependencyCount) {
        mainPassDependencyTextureAccessModes_.push_back(
            RenderGraphAccessMode::Read);
      }
    }
    frame.metrics.visibility.occlusionAvailable = 1u;
    frame.metrics.visibility.meshletOcclusionAvailable = 1u;
  }
  meshletDispatchItems_.clear();
  meshletPushConstants_.clear();
  meshletDispatchDependencyBuffers_.clear();
  if (meshletActive) {
    NURI_PROFILER_ZONE("OpaqueRenderer.meshlet_dispatch_build",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    auto lodBoundsDepResult = appendUniqueDependency(
        mainPassDependencyBuffers_, mainPassDependencyBufferAccessModes_,
        instanceLodBoundsBuffer_->handle(), RenderGraphAccessMode::Read,
        "OpaqueRenderer::buildOpaquePasses(meshlet pass)");
    if (lodBoundsDepResult.hasError()) {
      return lodBoundsDepResult;
    }

    const bool hasDepthPreparedMeshletMain =
        meshletDepthPrepassEnabled || hasClassicNormalDepthPrepass;
    auto meshletBuildResult = buildMeshletDispatches(
        meshletDispatchItems_, meshletPushConstants_,
        meshletDispatchDependencyBuffers_, meshletPipelineHandle_,
        meshletDoubleSidedPipelineHandle_, {}, {},
        hasDepthPreparedMeshletMain ? CompareOp::LessEqual : CompareOp::Less,
        !hasDepthPreparedMeshletMain, "OpaqueMeshlet", kMeshDebugColor,
        meshletPreviousFrameOcclusionAvailable,
        meshletVisibilityCounterBufferAddress, meshletCounterFlags, 0u, 0u, 0u,
        canUseStaticBatchCache && !meshletDepthPrepassEnabled &&
            !meshletPreviousFrameOcclusionAvailable);
    if (meshletBuildResult.hasError()) {
      return Result<bool, std::string>::makeError(meshletBuildResult.error());
    }
    const uint64_t mainCandidateCount = meshletBuildResult.value();
    bool meshletIndirectDispatchUsed = false;
    constexpr size_t kMeshDispatchCommandBytes = sizeof(uint32_t) * 3u;
    if (visibilitySettings.enableIndirectMeshDispatch &&
        visibilityCounterPreparedForFrame &&
        !gpuVisibilityCandidateIndices.empty() &&
        nuri::isValid(visibilityIndirectMeshDispatchPipelineHandle_) &&
        !meshletDispatchItems_.empty() &&
        meshletDispatchItems_.size() == meshletPushConstants_.size() &&
        frameSlot < visibilityMeshletDispatchRing_.size() &&
        frameSlot < visibilityMeshletIndirectCommandRing_.size() &&
        frameSlot < visibilityCandidateRing_.size() &&
        frameSlot < visibilityPassRing_.size() &&
        frameSlot < instanceRemapRing_.size()) {
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
            visibilityMeshletDispatchRing_[frameSlot];
        const DynamicBufferSlot &commandSlot =
            visibilityMeshletIndirectCommandRing_[frameSlot];
        const DynamicBufferSlot &candidateSlot =
            visibilityCandidateRing_[frameSlot];
        const DynamicBufferSlot &passSlot = visibilityPassRing_[frameSlot];
        const DynamicBufferSlot &remapSlot = instanceRemapRing_[frameSlot];
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
            if (sourceCandidateIndex >= visibilityCandidates.size()) {
              return Result<bool, std::string>::makeError(
                  "OpaqueRenderer::buildOpaquePasses: invalid meshlet "
                  "visibility candidate map source");
            }
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

          auto updateDispatchResult = gpu_.updateBuffer(
              dispatchSlot.buffer->handle(),
              std::as_bytes(std::span<const VisibilityMeshletDispatchGpuData>(
                  visibilityMeshletDispatchGpuData_.data(),
                  visibilityMeshletDispatchGpuData_.size())),
              0u);
          if (updateDispatchResult.hasError()) {
            return updateDispatchResult;
          }
          auto updateCandidateMapResult =
              gpu_.updateBuffer(dispatchSlot.buffer->handle(),
                                std::as_bytes(std::span<const uint32_t>(
                                    visibilityMeshletCandidateMap_.data(),
                                    visibilityMeshletCandidateMap_.size())),
                                candidateMapOffset);
          if (updateCandidateMapResult.hasError()) {
            return updateCandidateMapResult;
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
            dispatch.pipeline = visibilityIndirectMeshDispatchPipelineHandle_;
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
    finalPassDrawItems = {};
    frame.metrics.opaque.meshletModeActive = 1u;
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
          selectMsaaScenePipeline(draw.pipeline, draw.alphaMasked);
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
  pass.desc.drawBuffersPreResolved = true;
  pass.desc.preResolvedDrawBuffers = std::span<const BufferHandle>(
      preResolvedDrawBuffers_.data(), preResolvedDrawBuffers_.size());
  pass.desc.debugLabel = kOpaqueMainPassLabel;
  pass.desc.debugColor = kOpaquePassDebugColor;
  pass.desc.gpuTimingScope = GpuTimingScope::Opaque;
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
  pass.hasDraws = !finalPassDrawItems.empty() || !meshletDispatchItems_.empty();
  pass.hasPreDispatch = !mainPreDispatches_.empty();
  pass.desc.borrowPayload = !pass.hasPreDispatch;
  pass.hasIndirectDraws = !meshletActive && hasIndirectBaseDraws;
  pass.isMainPass = true;
  const TextureHandle sceneColorTarget =
      msaaSelected ? frame.sharedResources.msaaSceneColorTexture
                   : frame.sharedResources.sceneColorTexture;
  if (!nuri::isValid(sceneColorTarget)) {
    return Result<bool, std::string>::makeError(
        msaaSelected ? "OpaqueRenderer::buildOpaquePasses: MSAA scene color "
                       "texture is unavailable"
                     : "OpaqueRenderer::buildOpaquePasses: scene color "
                       "texture is unavailable");
  }
  pass.colorTextureHandle = sceneColorTarget;
  if (msaaSelected) {
    if (!nuri::isValid(frame.sharedResources.sceneColorTexture) ||
        !nuri::isValid(sceneDepthTexture)) {
      return Result<bool, std::string>::makeError(
          "OpaqueRenderer::buildOpaquePasses: MSAA resolve target texture is "
          "unavailable");
    }
    pass.desc.color.storeOp = StoreOp::MsaaResolve;
    pass.desc.color.resolveMode = ResolveMode::Average;
    pass.colorResolveTextureHandle = frame.sharedResources.sceneColorTexture;
    pass.desc.depth.storeOp = StoreOp::MsaaResolve;
    pass.desc.depth.resolveMode = ResolveMode::Min;
    pass.depthResolveTextureHandle = sceneDepthTexture;

    AntiAliasingFrameMetrics &aaMetrics = frame.metrics.antiAliasing;
    aaMetrics.msaaResolvePassCount = 1u;
    aaMetrics.msaaColorResolveTargetBound = true;
    aaMetrics.msaaDepthResolveTargetBound = true;
    aaMetrics.msaaAlphaMaskedDrawCount = msaaAlphaMaskedDrawCount;
    aaMetrics.msaaAlphaToCoverageEnabled = msaaAlphaMaskedDrawCount > 0u;
    aaMetrics.msaaSampleShadingEnabled = msaaAlphaMaskedDrawCount > 0u;
    aaMetrics.msaaResolveBandwidthEstimateBytes =
        aaMetrics.msaaColorTextureBytes + aaMetrics.msaaDepthTextureBytes;
  }

  if (meshletActive && taaSelected &&
      nuri::isValid(frame.sharedResources.reactiveMaskTexture) &&
      nuri::isValid(meshletReactiveMaskPipelineHandle_) &&
      drawAlphaMasked_.size() == drawItems_.size()) {
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
      auto meshletReactiveBuildResult = buildMeshletDispatches(
          meshletReactiveMaskDispatchItems_, meshletReactiveMaskPushConstants_,
          meshletReactiveMaskDispatchDependencyBuffers_,
          meshletReactiveMaskPipelineHandle_,
          meshletReactiveMaskDoubleSidedPipelineHandle_, {}, {},
          CompareOp::LessEqual, false, "OpaqueMeshletReactiveMask", 0xff33cc88,
          false, 0u, 0u, 0u, velocityInstanceFlagsAddress,
          velocityFrameDataAddress, false);
      if (meshletReactiveBuildResult.hasError()) {
        return Result<bool, std::string>::makeError(
            meshletReactiveBuildResult.error());
      }

      if (!meshletReactiveMaskDispatchItems_.empty()) {
        auto reactiveBaseDepResult = populateCoverageDependencyBuffers(
            reactivePassDependencyBuffers_,
            reactivePassDependencyBufferAccessModes_,
            "OpaqueRenderer::buildOpaquePasses(meshlet reactive pass)");
        if (reactiveBaseDepResult.hasError()) {
          return reactiveBaseDepResult;
        }
        for (const BufferHandle handle : {instanceLodBoundsBuffer_->handle(),
                                          velocityInstanceFlagsBufferHandle,
                                          velocityFrameDataBufferHandle}) {
          auto reactiveDepResult = appendUniqueDependency(
              reactivePassDependencyBuffers_,
              reactivePassDependencyBufferAccessModes_, handle,
              RenderGraphAccessMode::Read,
              "OpaqueRenderer::buildOpaquePasses(meshlet reactive pass)");
          if (reactiveDepResult.hasError()) {
            return reactiveDepResult;
          }
        }

        PreparedGraphPass &reactivePass =
            out.emplace_back(drawItems_.get_allocator().resource());
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
        reactivePass.desc.dependencyBuffers = std::span<const BufferHandle>(
            reactivePassDependencyBuffers_.data(),
            reactivePassDependencyBuffers_.size());
        reactivePass.desc.dependencyBufferAccessModes =
            std::span<const RenderGraphAccessMode>(
                reactivePassDependencyBufferAccessModes_.data(),
                reactivePassDependencyBufferAccessModes_.size());
        reactivePass.desc.dependencyTextures = std::span<const TextureHandle>(
            passDependencyTextures_.data(), passDependencyTextures_.size());
        reactivePass.desc.dependencyTextureAccessModes =
            std::span<const RenderGraphAccessMode>(
                mainPassDependencyTextureAccessModes_.data(),
                passDependencyTextures_.size());
        reactivePass.desc.meshDispatches = std::span<const MeshDispatchItem>(
            meshletReactiveMaskDispatchItems_.data(),
            meshletReactiveMaskDispatchItems_.size());
        reactivePass.desc.debugLabel = "Opaque Meshlet Reactive Mask Pass";
        reactivePass.desc.debugColor = 0xff33cc88;
        reactivePass.hasDraws = true;
        reactivePass.desc.borrowPayload = true;
        reactivePass.hasIndirectDraws = false;
        reactivePass.isReactiveMaskPass = true;
        aaMetrics.reactiveMaskDrawCount =
            saturateToU32(meshletReactiveMaskDispatchItems_.size());
        aaMetrics.reactiveMaskPassCount = 1u;
      }
    }
  }

  if (!meshletActive && taaSelected &&
      nuri::isValid(frame.sharedResources.reactiveMaskTexture) &&
      nuri::isValid(meshReactiveMaskPipelineHandle_) &&
      baseAlphaMasked.size() == shadedBaseDrawItems.size()) {
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
      auto reactiveBaseDepResult = populateCoverageDependencyBuffers(
          reactivePassDependencyBuffers_,
          reactivePassDependencyBufferAccessModes_,
          "OpaqueRenderer::buildOpaquePasses(reactive pass)");
      if (reactiveBaseDepResult.hasError()) {
        return reactiveBaseDepResult;
      }
      for (const BufferHandle handle :
           {velocityInstanceFlagsBufferHandle, velocityFrameDataBufferHandle}) {
        auto reactiveDepResult = appendUniqueDependency(
            reactivePassDependencyBuffers_,
            reactivePassDependencyBufferAccessModes_, handle,
            RenderGraphAccessMode::Read,
            "OpaqueRenderer::buildOpaquePasses(reactive pass)");
        if (reactiveDepResult.hasError()) {
          return reactiveDepResult;
        }
      }

      PreparedGraphPass &reactivePass =
          out.emplace_back(drawItems_.get_allocator().resource());
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
      reactivePass.desc.dependencyBuffers =
          std::span<const BufferHandle>(reactivePassDependencyBuffers_.data(),
                                        reactivePassDependencyBuffers_.size());
      reactivePass.desc.dependencyBufferAccessModes =
          std::span<const RenderGraphAccessMode>(
              reactivePassDependencyBufferAccessModes_.data(),
              reactivePassDependencyBufferAccessModes_.size());
      reactivePass.desc.dependencyTextures = std::span<const TextureHandle>(
          passDependencyTextures_.data(), passDependencyTextures_.size());
      reactivePass.desc.dependencyTextureAccessModes =
          std::span<const RenderGraphAccessMode>(
              mainPassDependencyTextureAccessModes_.data(),
              passDependencyTextures_.size());
      reactivePass.desc.draws = std::span<const DrawItem>(
          reactiveMaskDrawItems_.data(), reactiveMaskDrawItems_.size());
      reactivePass.desc.drawBuffersPreResolved = true;
      reactivePass.desc.preResolvedDrawBuffers = std::span<const BufferHandle>(
          preResolvedDrawBuffers_.data(), preResolvedDrawBuffers_.size());
      reactivePass.desc.debugLabel = "Opaque Reactive Mask Pass";
      reactivePass.desc.debugColor = 0xff33cc88;
      reactivePass.hasDraws = true;
      reactivePass.desc.borrowPayload = true;
      reactivePass.hasIndirectDraws = hasIndirectBaseDraws;
      reactivePass.isReactiveMaskPass = true;
      aaMetrics.reactiveMaskPassCount = 1u;
    }
  }

  if (meshletActive && taaSelected &&
      nuri::isValid(frame.sharedResources.motionVectorTexture) &&
      nuri::isValid(meshletVelocityPipelineHandle_) && instanceCount > 0) {
    auto meshletVelocityBuildResult = buildMeshletDispatches(
        meshletVelocityDispatchItems_, meshletVelocityPushConstants_,
        meshletVelocityDispatchDependencyBuffers_,
        meshletVelocityPipelineHandle_,
        meshletVelocityDoubleSidedPipelineHandle_, {}, {}, CompareOp::LessEqual,
        false, "OpaqueMeshletVelocity", 0xff44aaff, false, 0u, 0u,
        previousInstanceMatricesAddress, velocityInstanceFlagsAddress,
        velocityFrameDataAddress, false);
    if (meshletVelocityBuildResult.hasError()) {
      return Result<bool, std::string>::makeError(
          meshletVelocityBuildResult.error());
    }

    if (!meshletVelocityDispatchItems_.empty()) {
      auto velocityBaseDepResult = populateCoverageDependencyBuffers(
          velocityPassDependencyBuffers_,
          velocityPassDependencyBufferAccessModes_,
          "OpaqueRenderer::buildOpaquePasses(meshlet velocity pass)");
      if (velocityBaseDepResult.hasError()) {
        return velocityBaseDepResult;
      }
      for (const BufferHandle handle :
           {instanceLodBoundsBuffer_->handle(),
            previousInstanceMatricesBufferHandle,
            velocityInstanceFlagsBufferHandle, velocityFrameDataBufferHandle,
            velocityGeometryBufferHandle}) {
        auto velocityDepResult = appendUniqueDependency(
            velocityPassDependencyBuffers_,
            velocityPassDependencyBufferAccessModes_, handle,
            RenderGraphAccessMode::Read,
            "OpaqueRenderer::buildOpaquePasses(meshlet velocity pass)");
        if (velocityDepResult.hasError()) {
          return velocityDepResult;
        }
      }

      PreparedGraphPass &velocityPass =
          out.emplace_back(drawItems_.get_allocator().resource());
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
      velocityPass.desc.dependencyBuffers =
          std::span<const BufferHandle>(velocityPassDependencyBuffers_.data(),
                                        velocityPassDependencyBuffers_.size());
      velocityPass.desc.dependencyBufferAccessModes =
          std::span<const RenderGraphAccessMode>(
              velocityPassDependencyBufferAccessModes_.data(),
              velocityPassDependencyBufferAccessModes_.size());
      velocityPass.desc.dependencyTextures = std::span<const TextureHandle>(
          passDependencyTextures_.data(), passDependencyTextures_.size());
      velocityPass.desc.dependencyTextureAccessModes =
          std::span<const RenderGraphAccessMode>(
              mainPassDependencyTextureAccessModes_.data(),
              passDependencyTextures_.size());
      velocityPass.desc.meshDispatches = std::span<const MeshDispatchItem>(
          meshletVelocityDispatchItems_.data(),
          meshletVelocityDispatchItems_.size());
      velocityPass.desc.debugLabel = "Opaque Meshlet Velocity Pass";
      velocityPass.desc.debugColor = 0xff44aaff;
      velocityPass.hasDraws = true;
      velocityPass.desc.borrowPayload = true;
      velocityPass.hasIndirectDraws = false;
      velocityPass.isVelocityPass = true;
      const bool gtaoTemporalVelocityRequested =
          ambientOcclusionSettings.active &&
          ambientOcclusionSettings.temporalAccumulation && taaSelected &&
          frame.camera.historyValid && frame.camera.temporalDataValid &&
          nuri::isValid(frame.sharedResources.previousAmbientOcclusionTexture);
      velocityPass.isEarlyVelocityPass = gtaoTemporalVelocityRequested;

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

  if (!meshletActive && taaSelected &&
      nuri::isValid(frame.sharedResources.motionVectorTexture) &&
      nuri::isValid(meshVelocityPipelineHandle_) && instanceCount > 0) {
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
      auto velocityBaseDepResult = populateCoverageDependencyBuffers(
          velocityPassDependencyBuffers_,
          velocityPassDependencyBufferAccessModes_,
          "OpaqueRenderer::buildOpaquePasses(velocity pass)");
      if (velocityBaseDepResult.hasError()) {
        return velocityBaseDepResult;
      }
      for (const BufferHandle handle :
           {previousInstanceMatricesBufferHandle,
            velocityInstanceFlagsBufferHandle, velocityFrameDataBufferHandle,
            velocityGeometryBufferHandle}) {
        auto velocityDepResult = appendUniqueDependency(
            velocityPassDependencyBuffers_,
            velocityPassDependencyBufferAccessModes_, handle,
            RenderGraphAccessMode::Read,
            "OpaqueRenderer::buildOpaquePasses(velocity pass)");
        if (velocityDepResult.hasError()) {
          return velocityDepResult;
        }
      }

      PreparedGraphPass &velocityPass =
          out.emplace_back(drawItems_.get_allocator().resource());
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
      velocityPass.desc.dependencyBuffers =
          std::span<const BufferHandle>(velocityPassDependencyBuffers_.data(),
                                        velocityPassDependencyBuffers_.size());
      velocityPass.desc.dependencyBufferAccessModes =
          std::span<const RenderGraphAccessMode>(
              velocityPassDependencyBufferAccessModes_.data(),
              velocityPassDependencyBufferAccessModes_.size());
      velocityPass.desc.dependencyTextures = std::span<const TextureHandle>(
          passDependencyTextures_.data(), passDependencyTextures_.size());
      velocityPass.desc.dependencyTextureAccessModes =
          std::span<const RenderGraphAccessMode>(
              mainPassDependencyTextureAccessModes_.data(),
              passDependencyTextures_.size());
      velocityPass.desc.draws = std::span<const DrawItem>(
          velocityDrawItems_.data(), velocityDrawItems_.size());
      velocityPass.desc.drawBuffersPreResolved = true;
      velocityPass.desc.preResolvedDrawBuffers = std::span<const BufferHandle>(
          preResolvedDrawBuffers_.data(), preResolvedDrawBuffers_.size());
      velocityPass.desc.debugLabel = "Opaque Velocity Pass";
      velocityPass.desc.debugColor = 0xff44aaff;
      velocityPass.hasDraws = true;
      velocityPass.desc.borrowPayload = true;
      velocityPass.hasIndirectDraws = hasIndirectBaseDraws;
      velocityPass.isVelocityPass = true;
      const bool gtaoTemporalVelocityRequested =
          ambientOcclusionSettings.active &&
          ambientOcclusionSettings.temporalAccumulation && taaSelected &&
          frame.camera.historyValid && frame.camera.temporalDataValid &&
          nuri::isValid(frame.sharedResources.previousAmbientOcclusionTexture);
      velocityPass.isEarlyVelocityPass = gtaoTemporalVelocityRequested;

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
      nuri::isValid(meshShadowInspectPipelineHandle_)) {
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

    PreparedGraphPass &inspectPass =
        out.emplace_back(drawItems_.get_allocator().resource());
    inspectPass.desc.color = {.loadOp = LoadOp::Clear,
                              .storeOp = StoreOp::Store,
                              .clearColor = {0.0f, 0.0f, 0.0f, -1.0f}};
    inspectPass.colorTextureHandle = shadowInspectTexture_;
    inspectPass.desc.depth = {.loadOp = LoadOp::Load,
                              .storeOp = StoreOp::Store,
                              .clearDepth = kClearDepthOne,
                              .clearStencil = 0};
    inspectPass.depthTextureHandle = sceneDepthTexture;
    inspectPass.desc.dependencyBuffers = std::span<const BufferHandle>(
        mainPassDependencyBuffers_.data(), mainPassDependencyBuffers_.size());
    inspectPass.desc.dependencyBufferAccessModes =
        std::span<const RenderGraphAccessMode>(
            mainPassDependencyBufferAccessModes_.data(),
            mainPassDependencyBufferAccessModes_.size());
    inspectPass.desc.dependencyTextures = std::span<const TextureHandle>(
        mainPassDependencyTextures_.data(), mainPassDependencyTextures_.size());
    inspectPass.desc.dependencyTextureAccessModes =
        std::span<const RenderGraphAccessMode>(
            mainPassDependencyTextureAccessModes_.data(),
            mainPassDependencyTextureAccessModes_.size());
    inspectPass.desc.draws = std::span<const DrawItem>(
        shadowInspectDrawItems_.data(), shadowInspectDrawItems_.size());
    inspectPass.desc.drawBuffersPreResolved = true;
    inspectPass.desc.preResolvedDrawBuffers = std::span<const BufferHandle>(
        preResolvedDrawBuffers_.data(), preResolvedDrawBuffers_.size());
    inspectPass.desc.debugLabel = "Opaque Shadow Inspect Pass";
    inspectPass.desc.debugColor = kOpaquePassDebugColor;
    inspectPass.hasDraws = !shadowInspectDrawItems_.empty();
    inspectPass.hasPreDispatch = false;
    inspectPass.desc.borrowPayload = true;
    inspectPass.hasIndirectDraws = false;

    inFlightShadowInspectReadback_ =
        InFlightShadowInspectReadback{.request = *pendingShadowInspectRequest_,
                                      .submissionFrame = frame.frameIndex};
    pendingShadowInspectRequest_.reset();
  }

  frame.metrics.opaque.computeDispatches = saturateToU32(preDispatches_.size());
  capturePreviousTransforms(*frame.scene, frame.frameIndex);
  logOpaqueVisibilityCounters(frame);
  NURI_PROFILER_ZONE_END();
  return Result<bool, std::string>::makeResult(true);
}

bool OpaqueRenderer::requiresDepthPyramid(
    const RenderSettings &settings) const {
  const ShadowSdsmMode sanitizedSdsm =
      sanitizeShadowSdsmMode(settings.shadow.sdsmMode);
  const VisibilityOcclusionMode visibilityOcclusionMode =
      sanitizeVisibilityOcclusionMode(settings.visibility.occlusionMode);
  return settings.opaque.enableDepthPyramid ||
         visibilityOcclusionMode == VisibilityOcclusionMode::PreviousFrameHiZ ||
         visibilityOcclusionMode ==
             VisibilityOcclusionMode::CurrentFrameHiZExperimental ||
         (settings.shadow.enabled &&
          (sanitizedSdsm == ShadowSdsmMode::PreviousFrameMinMax ||
           sanitizedSdsm == ShadowSdsmMode::Histogram));
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

  Result<bool, std::string> buildResult =
      Result<bool, std::string>::makeResult(true);
  {
    NURI_PROFILER_ZONE("OpaqueRenderer.graph_prepare_passes",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    buildResult = buildOpaquePasses(frame, preparedGraphPasses_);
    NURI_PROFILER_ZONE_END();
  }
  if (buildResult.hasError()) {
    return buildResult;
  }
  return Result<bool, std::string>::makeResult(true);
}

bool OpaqueRenderer::hasPreparedOpaqueMainPasses() const noexcept {
  for (const PreparedGraphPass &pass : preparedGraphPasses_) {
    if (pass.isMainPass) {
      return true;
    }
  }
  return false;
}

bool OpaqueRenderer::hasPreparedOpaquePrepassPasses() const noexcept {
  for (const PreparedGraphPass &pass : preparedGraphPasses_) {
    if (isPreLightingPass(pass)) {
      return true;
    }
  }
  return false;
}

bool OpaqueRenderer::hasPreparedOpaqueMainLightingPasses() const noexcept {
  for (const PreparedGraphPass &pass : preparedGraphPasses_) {
    if (!pass.isPickPass && !isPreLightingPass(pass)) {
      return true;
    }
  }
  return false;
}

bool OpaqueRenderer::hasPreparedOpaquePickPasses() const noexcept {
  for (const PreparedGraphPass &pass : preparedGraphPasses_) {
    if (pass.isPickPass) {
      return true;
    }
  }
  return false;
}

bool OpaqueRenderer::shouldPublishSceneDepthGraphTexture(
    const RenderFrameContext &frame) const {
  const RenderSettings &settings = settingsOrDefault(frame);
  if (isRenderCaptureRequested(frame, "scene_depth")) {
    return true;
  }
  if (settings.opaque.enableDepthPyramid) {
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

void OpaqueRenderer::cachePreparedGraphPassMetadata(PreparedGraphPass &) const {
}

bool OpaqueRenderer::isPreLightingPass(const PreparedGraphPass &pass) noexcept {
  return pass.isDepthPrepass || pass.isNormalPrepass ||
         pass.isTransmissionVisibilityDepthPass || pass.isDepthPyramidPass ||
         pass.isEarlyVelocityPass || pass.isVisibilityComputePass ||
         pass.desc.gpuTimingScope == GpuTimingScope::ShadowSdsm;
}

Result<bool, std::string> OpaqueRenderer::appendPreparedGraphPass(
    RenderFrameContext &frame, RenderGraphBuilder &graph,
    const PreparedGraphPass &pass, uint32_t safeWidth, uint32_t safeHeight,
    std::span<const RenderGraphBufferId> preResolvedDrawBufferIds) {
  RenderGraphPreparedGraphicsPassDesc passDesc{};
  passDesc.executionMode = pass.desc.executionMode;
  passDesc.color = pass.desc.color;
  passDesc.hasColorAttachment = pass.desc.hasColorAttachment;
  passDesc.depth = pass.desc.depth;
  passDesc.useViewport = pass.desc.useViewport;
  passDesc.viewport = pass.desc.viewport;
  passDesc.preDispatches = pass.desc.preDispatches;
  passDesc.dependencyBuffers = pass.desc.dependencyBuffers;
  passDesc.draws = pass.desc.draws;
  passDesc.meshDispatches = pass.desc.meshDispatches;
  passDesc.drawBuffersPreResolved = pass.desc.drawBuffersPreResolved;
  passDesc.gpuTimingScope = pass.desc.gpuTimingScope;
  passDesc.debugLabel = pass.desc.debugLabel;
  passDesc.debugColor = pass.desc.debugColor;
  passDesc.markColorAsFrameOutput = pass.desc.markColorAsFrameOutput;
  passDesc.markImplicitOutputSideEffect =
      pass.desc.markImplicitOutputSideEffect;
  passDesc.borrowPayload = pass.desc.borrowPayload;

  if (nuri::isValid(pass.colorTextureHandle)) {
    auto colorImportResult = graph.importTexture(pass.colorTextureHandle,
                                                 "opaque_pass_color_texture");
    if (colorImportResult.hasError()) {
      return Result<bool, std::string>::makeError(colorImportResult.error());
    }
    passDesc.colorTexture = colorImportResult.value();
  }

  if (nuri::isValid(pass.colorResolveTextureHandle)) {
    auto colorResolveImportResult = graph.importTexture(
        pass.colorResolveTextureHandle, "opaque_pass_color_resolve_texture");
    if (colorResolveImportResult.hasError()) {
      return Result<bool, std::string>::makeError(
          colorResolveImportResult.error());
    }
    passDesc.colorResolveTexture = colorResolveImportResult.value();
  }

  if (nuri::isValid(pass.depthTextureHandle)) {
    auto depthImportResult = graph.importTexture(pass.depthTextureHandle,
                                                 "opaque_pass_depth_texture");
    if (depthImportResult.hasError()) {
      return Result<bool, std::string>::makeError(depthImportResult.error());
    }
    passDesc.depthTexture = depthImportResult.value();
  }

  if (nuri::isValid(pass.depthResolveTextureHandle)) {
    auto depthResolveImportResult = graph.importTexture(
        pass.depthResolveTextureHandle, "opaque_pass_depth_resolve_texture");
    if (depthResolveImportResult.hasError()) {
      return Result<bool, std::string>::makeError(
          depthResolveImportResult.error());
    }
    passDesc.depthResolveTexture = depthResolveImportResult.value();
  }

  ScratchArena passBuildScratchArena;
  ScopedScratch passBuildScratch(passBuildScratchArena);

  std::pmr::vector<RenderGraphPreparedDependencyBufferBinding>
      dependencyBufferBindings(passBuildScratch.resource());
  dependencyBufferBindings.reserve(pass.desc.dependencyBuffers.size());
  for (size_t i = 0; i < pass.desc.dependencyBuffers.size(); ++i) {
    const BufferHandle dependency = pass.desc.dependencyBuffers[i];
    if (!nuri::isValid(dependency)) {
      continue;
    }
    const RenderGraphAccessMode accessMode =
        i < pass.desc.dependencyBufferAccessModes.size()
            ? pass.desc.dependencyBufferAccessModes[i]
            : (RenderGraphAccessMode::Read | RenderGraphAccessMode::Write);
    auto importResult =
        graph.importBuffer(dependency, "opaque_pass_dependency_buffer");
    if (importResult.hasError()) {
      return Result<bool, std::string>::makeError(importResult.error());
    }
    dependencyBufferBindings.push_back(
        RenderGraphPreparedDependencyBufferBinding{
            .dependencyIndex = static_cast<uint32_t>(i),
            .buffer = importResult.value(),
            .mode = accessMode,
        });
  }
  passDesc.dependencyBufferBindings =
      std::span<const RenderGraphPreparedDependencyBufferBinding>(
          dependencyBufferBindings.data(), dependencyBufferBindings.size());

  std::pmr::vector<RenderGraphPreparedDependencyTextureBinding>
      dependencyTextureBindings(passBuildScratch.resource());
  dependencyTextureBindings.reserve(pass.desc.dependencyTextures.size());
  for (size_t i = 0; i < pass.desc.dependencyTextures.size(); ++i) {
    const TextureHandle dependency = pass.desc.dependencyTextures[i];
    if (!nuri::isValid(dependency)) {
      continue;
    }
    const RenderGraphAccessMode accessMode =
        i < pass.desc.dependencyTextureAccessModes.size()
            ? pass.desc.dependencyTextureAccessModes[i]
            : RenderGraphAccessMode::Read;
    auto importResult =
        graph.importTexture(dependency, "opaque_pass_dependency_texture");
    if (importResult.hasError()) {
      return Result<bool, std::string>::makeError(importResult.error());
    }
    dependencyTextureBindings.push_back(
        RenderGraphPreparedDependencyTextureBinding{
            .texture = importResult.value(),
            .mode = accessMode,
        });
  }
  passDesc.dependencyTextureBindings =
      std::span<const RenderGraphPreparedDependencyTextureBinding>(
          dependencyTextureBindings.data(), dependencyTextureBindings.size());

  size_t preDispatchDependencyCount = 0u;
  for (const ComputeDispatchItem &dispatch : pass.desc.preDispatches) {
    preDispatchDependencyCount += dispatch.dependencyBuffers.size();
  }
  std::pmr::vector<RenderGraphPreparedPreDispatchDependencyBinding>
      preDispatchDependencyBindings(passBuildScratch.resource());
  preDispatchDependencyBindings.reserve(preDispatchDependencyCount);
  for (size_t dispatchIndex = 0; dispatchIndex < pass.desc.preDispatches.size();
       ++dispatchIndex) {
    const ComputeDispatchItem &dispatch =
        pass.desc.preDispatches[dispatchIndex];
    for (size_t dependencyIndex = 0;
         dependencyIndex < dispatch.dependencyBuffers.size();
         ++dependencyIndex) {
      const BufferHandle dependency =
          dispatch.dependencyBuffers[dependencyIndex];
      if (!nuri::isValid(dependency)) {
        continue;
      }
      auto importResult = graph.importBuffer(
          dependency, "opaque_pass_pre_dispatch_dependency_buffer");
      if (importResult.hasError()) {
        return Result<bool, std::string>::makeError(importResult.error());
      }
      preDispatchDependencyBindings.push_back(
          RenderGraphPreparedPreDispatchDependencyBinding{
              .preDispatchIndex = static_cast<uint32_t>(dispatchIndex),
              .dependencyIndex = static_cast<uint32_t>(dependencyIndex),
              .buffer = importResult.value(),
              .mode =
                  RenderGraphAccessMode::Read | RenderGraphAccessMode::Write,
          });
    }
  }
  passDesc.preDispatchDependencyBindings =
      std::span<const RenderGraphPreparedPreDispatchDependencyBinding>(
          preDispatchDependencyBindings.data(),
          preDispatchDependencyBindings.size());

  std::pmr::vector<RenderGraphPreparedDrawBufferBinding> drawBufferBindings(
      passBuildScratch.resource());
  if (pass.desc.drawBuffersPreResolved) {
    const bool hasSharedPreResolvedDrawBuffers =
        !pass.desc.preResolvedDrawBuffers.empty();
    if (hasSharedPreResolvedDrawBuffers) {
      passDesc.preResolvedDrawBufferIds = preResolvedDrawBufferIds;
      passDesc.preResolvedDrawBuffers = pass.desc.preResolvedDrawBuffers;
    } else {
      passDesc.preResolvedDrawBufferIds =
          std::span<const RenderGraphBufferId>();
      passDesc.preResolvedDrawBuffers = std::span<const BufferHandle>();
    }
  } else {
    drawBufferBindings.reserve(pass.desc.draws.size() * 4u);
    for (size_t drawIndex = 0; drawIndex < pass.desc.draws.size();
         ++drawIndex) {
      const DrawItem &draw = pass.desc.draws[drawIndex];
      const std::array<
          std::pair<BufferHandle, RenderGraphDrawBufferBindingTarget>, 4>
          bindings = {{
              {draw.vertexBuffer, RenderGraphDrawBufferBindingTarget::Vertex},
              {draw.indexBuffer, RenderGraphDrawBufferBindingTarget::Index},
              {draw.indirectBuffer,
               RenderGraphDrawBufferBindingTarget::Indirect},
              {draw.indirectCountBuffer,
               RenderGraphDrawBufferBindingTarget::IndirectCount},
          }};
      for (const auto &[buffer, target] : bindings) {
        if (!nuri::isValid(buffer)) {
          continue;
        }
        auto importResult =
            graph.importBuffer(buffer, "opaque_pass_draw_buffer");
        if (importResult.hasError()) {
          return Result<bool, std::string>::makeError(importResult.error());
        }
        drawBufferBindings.push_back(RenderGraphPreparedDrawBufferBinding{
            .drawIndex = static_cast<uint32_t>(drawIndex),
            .target = target,
            .buffer = importResult.value(),
            .mode = RenderGraphAccessMode::Read,
        });
      }
    }
    passDesc.drawBufferBindings =
        std::span<const RenderGraphPreparedDrawBufferBinding>(
            drawBufferBindings.data(), drawBufferBindings.size());
  }

  auto addResult = graph.addPreparedGraphicsPass(passDesc);
  if (addResult.hasError()) {
    return Result<bool, std::string>::makeError(addResult.error());
  }
  const RenderGraphPassId passId = addResult.value();

  if (pass.isPickPass) {
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
    auto pickDepthResult = graph.createTransientTexture(
        pickDepthTransientDesc, "opaque_pick_transient_depth");
    if (pickDepthResult.hasError()) {
      return Result<bool, std::string>::makeError(pickDepthResult.error());
    }
    auto bindPickDepthResult =
        graph.bindPassDepthTexture(passId, pickDepthResult.value());
    if (bindPickDepthResult.hasError()) {
      return Result<bool, std::string>::makeError(bindPickDepthResult.error());
    }
    frame.sharedResources.opaquePickDepthGraphTexture = pickDepthResult.value();
  }

  const bool publishDepthGraphTexture =
      shouldPublishSceneDepthGraphTexture(frame) ||
      (pass.isMainPass && nuri::isValid(pass.depthResolveTextureHandle));
  if ((pass.isDepthPrepass || pass.isMainPass) &&
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
  if (pass.isTransmissionVisibilityDepthPass &&
      nuri::isValid(pass.depthTextureHandle)) {
    frame.sharedResources.transmissionVisibilityDepthGraphTexture =
        passDesc.depthTexture;
    publishRequestedCapture(
        frame, gpu_, "transmission_visibility_depth", pass.depthTextureHandle,
        RenderCaptureValueKind::Depth,
        RenderCaptureLifetimeClass::FeaturePersistentTexture, "linear_depth",
        "depth", pass.desc.debugLabel);
  }
  if (pass.isMainPass && nuri::isValid(pass.depthResolveTextureHandle) &&
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

  if (pass.isMainPass && nuri::isValid(pass.colorTextureHandle)) {
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
      }
    } else {
      frame.sharedResources.sceneColorGraphTexture = passDesc.colorTexture;
      publishRequestedCapture(
          frame, gpu_, "scene_color_hdr",
          frame.sharedResources.sceneColorTexture,
          RenderCaptureValueKind::LinearHdrColor,
          RenderCaptureLifetimeClass::FrameSharedRingTexture, "linear_hdr",
          "hdr_color", pass.desc.debugLabel);
    }
  }
  if (pass.isVelocityPass && nuri::isValid(pass.colorTextureHandle)) {
    frame.sharedResources.motionVectorGraphTexture = passDesc.colorTexture;
    frame.metrics.antiAliasing.motionVectorGraphPublished = true;
    frame.metrics.antiAliasing.motionVectorClearPassCount = 0u;
    frame.metrics.antiAliasing.motionVectorClearBytes = 0u;
    publishRequestedCapture(frame, gpu_, "motion_vectors",
                            pass.colorTextureHandle,
                            RenderCaptureValueKind::Velocity,
                            RenderCaptureLifetimeClass::FrameSharedRingTexture,
                            "uv_velocity", "velocity", pass.desc.debugLabel);
  }
  if (pass.isReactiveMaskPass && nuri::isValid(pass.colorTextureHandle)) {
    frame.sharedResources.reactiveMaskGraphTexture = passDesc.colorTexture;
    frame.metrics.antiAliasing.reactiveMaskGraphPublished = true;
    publishRequestedCapture(frame, gpu_, "reactive_mask",
                            pass.colorTextureHandle,
                            RenderCaptureValueKind::Mask,
                            RenderCaptureLifetimeClass::FrameSharedRingTexture,
                            "linear_scalar", "mask", pass.desc.debugLabel);
  }
  if (pass.isNormalPrepass && nuri::isValid(pass.colorTextureHandle)) {
    frame.sharedResources.normalGraphTexture = passDesc.colorTexture;
    frame.metrics.ambientOcclusion.normalGraphPublished = true;
    publishRequestedCapture(frame, gpu_, "material_normals",
                            pass.colorTextureHandle,
                            RenderCaptureValueKind::Normal,
                            RenderCaptureLifetimeClass::FrameSharedRingTexture,
                            "world_normal", "normal", pass.desc.debugLabel);
  }
  if (pass.isDepthPyramidPass &&
      pass.depthPyramidLevel < kMaxSceneDepthPyramidLevels &&
      nuri::isValid(passDesc.colorTexture)) {
    frame.sharedResources
        .sceneDepthPyramidGraphTextures[pass.depthPyramidLevel] =
        passDesc.colorTexture;
  }

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
OpaqueRenderer::appendOpaqueMainPasses(RenderFrameContext &frame,
                                       RenderGraphBuilder &graph) {
  int32_t framebufferWidth = 0;
  int32_t framebufferHeight = 0;
  gpu_.getFramebufferSize(framebufferWidth, framebufferHeight);
  const uint32_t safeWidth =
      static_cast<uint32_t>(std::max(framebufferWidth, 1));
  const uint32_t safeHeight =
      static_cast<uint32_t>(std::max(framebufferHeight, 1));

  {
    const BufferHandle centersHandle =
        (instanceCentersPhaseBuffer_ && instanceCentersPhaseBuffer_->valid())
            ? instanceCentersPhaseBuffer_->handle()
            : BufferHandle{};
    registerOrUpdatePersistentBuffer(graph, persistentCentersPhaseBuffer_,
                                     registeredCentersPhaseBufferHandle_,
                                     centersHandle,
                                     "opaque_instance_centers_phase_buffer");

    const BufferHandle baseMatHandle =
        (instanceBaseMatricesBuffer_ && instanceBaseMatricesBuffer_->valid())
            ? instanceBaseMatricesBuffer_->handle()
            : BufferHandle{};
    registerOrUpdatePersistentBuffer(graph, persistentBaseMatricesBuffer_,
                                     registeredBaseMatricesBufferHandle_,
                                     baseMatHandle,
                                     "opaque_instance_base_matrices_buffer");
  }

  NURI_PROFILER_ZONE("OpaqueRenderer.graph_add_main_passes",
                     NURI_PROFILER_COLOR_CMD_DRAW);
  auto preResolvedDrawBufferIdsResult = importPreResolvedBuffers(
      graph,
      std::span<const BufferHandle>(preResolvedDrawBuffers_.data(),
                                    preResolvedDrawBuffers_.size()),
      preparedGraphPasses_.get_allocator().resource(),
      "opaque_pass_draw_buffer");
  if (preResolvedDrawBufferIdsResult.hasError()) {
    return Result<bool, std::string>::makeError(
        preResolvedDrawBufferIdsResult.error());
  }
  std::pmr::vector<RenderGraphBufferId> preResolvedDrawBufferIds =
      std::move(preResolvedDrawBufferIdsResult).value();
  for (const PreparedGraphPass &pass : preparedGraphPasses_) {
    if (pass.isPickPass) {
      continue;
    }
    auto appendResult = appendPreparedGraphPass(
        frame, graph, pass, safeWidth, safeHeight, preResolvedDrawBufferIds);
    if (appendResult.hasError()) {
      return appendResult;
    }
  }
  NURI_PROFILER_ZONE_END();

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
OpaqueRenderer::appendOpaquePrepassPasses(RenderFrameContext &frame,
                                          RenderGraphBuilder &graph) {
  int32_t framebufferWidth = 0;
  int32_t framebufferHeight = 0;
  gpu_.getFramebufferSize(framebufferWidth, framebufferHeight);
  const uint32_t safeWidth =
      static_cast<uint32_t>(std::max(framebufferWidth, 1));
  const uint32_t safeHeight =
      static_cast<uint32_t>(std::max(framebufferHeight, 1));

  const BufferHandle centersHandle =
      (instanceCentersPhaseBuffer_ && instanceCentersPhaseBuffer_->valid())
          ? instanceCentersPhaseBuffer_->handle()
          : BufferHandle{};
  registerOrUpdatePersistentBuffer(
      graph, persistentCentersPhaseBuffer_, registeredCentersPhaseBufferHandle_,
      centersHandle, "opaque_instance_centers_phase_buffer");

  const BufferHandle baseMatHandle =
      (instanceBaseMatricesBuffer_ && instanceBaseMatricesBuffer_->valid())
          ? instanceBaseMatricesBuffer_->handle()
          : BufferHandle{};
  registerOrUpdatePersistentBuffer(
      graph, persistentBaseMatricesBuffer_, registeredBaseMatricesBufferHandle_,
      baseMatHandle, "opaque_instance_base_matrices_buffer");

  NURI_PROFILER_ZONE("OpaqueRenderer.graph_add_prepass_passes",
                     NURI_PROFILER_COLOR_CMD_DRAW);
  auto preResolvedDrawBufferIdsResult = importPreResolvedBuffers(
      graph,
      std::span<const BufferHandle>(preResolvedDrawBuffers_.data(),
                                    preResolvedDrawBuffers_.size()),
      preparedGraphPasses_.get_allocator().resource(),
      "opaque_pass_draw_buffer");
  if (preResolvedDrawBufferIdsResult.hasError()) {
    return Result<bool, std::string>::makeError(
        preResolvedDrawBufferIdsResult.error());
  }
  std::pmr::vector<RenderGraphBufferId> preResolvedDrawBufferIds =
      std::move(preResolvedDrawBufferIdsResult).value();
  for (const PreparedGraphPass &pass : preparedGraphPasses_) {
    if (!isPreLightingPass(pass)) {
      continue;
    }
    auto appendResult = appendPreparedGraphPass(
        frame, graph, pass, safeWidth, safeHeight, preResolvedDrawBufferIds);
    if (appendResult.hasError()) {
      return appendResult;
    }
  }
  NURI_PROFILER_ZONE_END();

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
OpaqueRenderer::appendOpaqueMainLightingPasses(RenderFrameContext &frame,
                                               RenderGraphBuilder &graph) {
  int32_t framebufferWidth = 0;
  int32_t framebufferHeight = 0;
  gpu_.getFramebufferSize(framebufferWidth, framebufferHeight);
  const uint32_t safeWidth =
      static_cast<uint32_t>(std::max(framebufferWidth, 1));
  const uint32_t safeHeight =
      static_cast<uint32_t>(std::max(framebufferHeight, 1));

  NURI_PROFILER_ZONE("OpaqueRenderer.graph_add_main_lighting_passes",
                     NURI_PROFILER_COLOR_CMD_DRAW);
  auto preResolvedDrawBufferIdsResult = importPreResolvedBuffers(
      graph,
      std::span<const BufferHandle>(preResolvedDrawBuffers_.data(),
                                    preResolvedDrawBuffers_.size()),
      preparedGraphPasses_.get_allocator().resource(),
      "opaque_pass_draw_buffer");
  if (preResolvedDrawBufferIdsResult.hasError()) {
    return Result<bool, std::string>::makeError(
        preResolvedDrawBufferIdsResult.error());
  }
  std::pmr::vector<RenderGraphBufferId> preResolvedDrawBufferIds =
      std::move(preResolvedDrawBufferIdsResult).value();
  for (const PreparedGraphPass &pass : preparedGraphPasses_) {
    if (pass.isPickPass || isPreLightingPass(pass)) {
      continue;
    }
    auto appendResult = appendPreparedGraphPass(
        frame, graph, pass, safeWidth, safeHeight, preResolvedDrawBufferIds);
    if (appendResult.hasError()) {
      return appendResult;
    }
  }
  NURI_PROFILER_ZONE_END();

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
OpaqueRenderer::appendOpaquePickPasses(RenderFrameContext &frame,
                                       RenderGraphBuilder &graph) {
  int32_t framebufferWidth = 0;
  int32_t framebufferHeight = 0;
  gpu_.getFramebufferSize(framebufferWidth, framebufferHeight);
  const uint32_t safeWidth =
      static_cast<uint32_t>(std::max(framebufferWidth, 1));
  const uint32_t safeHeight =
      static_cast<uint32_t>(std::max(framebufferHeight, 1));

  NURI_PROFILER_ZONE("OpaqueRenderer.graph_add_pick_passes",
                     NURI_PROFILER_COLOR_CMD_DRAW);
  auto preResolvedDrawBufferIdsResult = importPreResolvedBuffers(
      graph,
      std::span<const BufferHandle>(preResolvedDrawBuffers_.data(),
                                    preResolvedDrawBuffers_.size()),
      preparedGraphPasses_.get_allocator().resource(),
      "opaque_pass_draw_buffer");
  if (preResolvedDrawBufferIdsResult.hasError()) {
    return Result<bool, std::string>::makeError(
        preResolvedDrawBufferIdsResult.error());
  }
  std::pmr::vector<RenderGraphBufferId> preResolvedDrawBufferIds =
      std::move(preResolvedDrawBufferIdsResult).value();
  for (const PreparedGraphPass &pass : preparedGraphPasses_) {
    if (!pass.isPickPass) {
      continue;
    }
    auto appendResult = appendPreparedGraphPass(
        frame, graph, pass, safeWidth, safeHeight, preResolvedDrawBufferIds);
    if (appendResult.hasError()) {
      return appendResult;
    }
  }
  NURI_PROFILER_ZONE_END();

  return Result<bool, std::string>::makeResult(true);
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
    uint32_t requestedLod, bool tessPipelineEnabled, const DrawItem &baseDraw) {
  const size_t cacheIndex =
      singleInstanceCacheIndex(requestedLod, tessPipelineEnabled);
  if (cacheIndex >= singleInstanceBatchCaches_.size()) {
    return Result<bool, std::string>::makeError(
        "OpaqueRenderer::buildOpaquePasses: single-instance cache index out of "
        "range");
  }
  SingleInstanceBatchCache &cache = singleInstanceBatchCaches_[cacheIndex];
  const bool canReuseSingleInstanceCache =
      cache.valid && cache.requestedLod == requestedLod &&
      cache.tessPipelineEnabled == tessPipelineEnabled &&
      cache.templateRevision == singleInstanceTemplateRevision_ &&
      isSamePipelineHandle(cache.basePipeline, baseDraw.pipeline) &&
      isSamePipelineHandle(cache.doubleSidedBasePipeline,
                           meshDoubleSidedFillPipelineHandle_) &&
      isSamePipelineHandle(cache.tessPipeline, meshTessPipelineHandle_) &&
      isSamePipelineHandle(cache.doubleSidedTessPipeline,
                           meshDoubleSidedTessPipelineHandle_);
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
    if (!templateEntry.renderable || !templateEntry.submesh) {
      return Result<bool, std::string>::makeError(
          "OpaqueRenderer::buildOpaquePasses: invalid mesh template");
    }

    const auto lodIndex =
        resolveAvailableLod(*templateEntry.submesh, requestedLod);
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
      cache.batches.push_back(entry);
      const size_t insertedIndex = cache.batches.size() - 1;
      auto [insertedIt, _] = singleBatchLookup.emplace(key, insertedIndex);
      it = insertedIt;
    }

    ++cache.batches[it->second].instanceCount;
    ++cache.remapCount;
  }

  cache.requestedLod = requestedLod;
  cache.tessPipelineEnabled = tessPipelineEnabled;
  cache.basePipeline = baseDraw.pipeline;
  cache.doubleSidedBasePipeline = meshDoubleSidedFillPipelineHandle_;
  cache.tessPipeline = meshTessPipelineHandle_;
  cache.doubleSidedTessPipeline = meshDoubleSidedTessPipelineHandle_;
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

  NURI_PROFILER_ZONE("OpaqueRenderer.indirect_pack",
                     NURI_PROFILER_COLOR_CMD_DRAW);
  if (drawItems_.size() != drawPushConstants_.size()) {
    return Result<bool, std::string>::makeError(
        "OpaqueRenderer::buildIndirectDraws: draw and push constant count "
        "mismatch");
  }

  if (!drawSignatureValid) {
    drawSignature = computeIndirectDrawSignature(remapCount);
  }
  const bool canReusePackedCommands = canReuseIndirectPack(drawSignature);

  if (canReusePackedCommands) {
    auto indirectCapacityResult = ensureIndirectCommandRingCapacity(
        std::max(indirectPackCache_.requiredBytes, kIndirectCountHeaderBytes));
    if (indirectCapacityResult.hasError()) {
      return indirectCapacityResult;
    }
  }

  if (indirectUploadSignatures_.size() != indirectCommandRing_.size()) {
    indirectUploadSignatures_.assign(indirectCommandRing_.size(),
                                     kInvalidDrawSignature);
  }

  auto packResult =
      canReusePackedCommands
          ? refreshCachedIndirectPack(frameSlot, drawSignature)
          : rebuildIndirectPack(frameSlot, remapCount, drawSignature);
  if (packResult.hasError()) {
    return packResult;
  }
  if (frameSlot >= indirectCommandRing_.size() ||
      !indirectCommandRing_[frameSlot].buffer ||
      !indirectCommandRing_[frameSlot].buffer->valid()) {
    return Result<bool, std::string>::makeError(
        "OpaqueRenderer::buildIndirectDraws: indirect ring buffer slot is "
        "invalid");
  }

  NURI_PROFILER_ZONE_END();
  return Result<bool, std::string>::makeResult(true);
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

bool OpaqueRenderer::canReuseIndirectPack(uint64_t drawSignature) const {
  return indirectPackCache_.valid &&
         indirectPackCache_.drawSignature == drawSignature &&
         indirectSourceDrawIndices_.size() == indirectDrawItems_.size() &&
         indirectPackCache_.requiredBytes >= kIndirectCountHeaderBytes &&
         indirectCommandUploadBytes_.size() <= indirectPackCache_.requiredBytes;
}

Result<bool, std::string>
OpaqueRenderer::rebuildIndirectPack(uint32_t frameSlot, size_t remapCount,
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
    if (draw.indexCount == 0) {
      return Result<bool, std::string>::makeError(
          "OpaqueRenderer::buildIndirectDraws: non-indexed opaque draws are "
          "not "
          "supported in indirect mode");
    }
    if (draw.instanceCount == 0) {
      continue;
    }
    if (draw.firstInstance > remapCount ||
        draw.instanceCount > remapCount - draw.firstInstance) {
      return Result<bool, std::string>::makeError(
          "OpaqueRenderer::buildIndirectDraws: indirect draw instance range is "
          "out of remap bounds");
    }

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

  if (frameSlot >= indirectCommandRing_.size() ||
      !indirectCommandRing_[frameSlot].buffer ||
      !indirectCommandRing_[frameSlot].buffer->valid()) {
    return Result<bool, std::string>::makeError(
        "OpaqueRenderer::buildIndirectDraws: indirect ring buffer slot is "
        "invalid");
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
        indirectCommandRing_[frameSlot].buffer->handle();
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
            group.sourceDrawIndex < drawAlphaMasked_.size() &&
            drawAlphaMasked_[group.sourceDrawIndex] != 0u;
        indirectDraw.command = DrawCommandType::IndexedIndirect;
        indirectDraw.indirectBuffer = indirectBufferHandle;
        indirectDraw.indirectBufferOffset =
            chunkOffset + kIndirectCountHeaderBytes;
        indirectDraw.indirectDrawCount = drawCount;
        indirectDraw.indirectStride = sizeof(DrawIndexedIndirectCommand);
        if (group.sourceDrawIndex >= drawPushConstants_.size()) {
          return Result<bool, std::string>::makeError(
              "OpaqueRenderer::buildIndirectDraws: indirect source index is "
              "out "
              "of range");
        }
        indirectDraw.pushConstants = std::span<const std::byte>(
            reinterpret_cast<const std::byte *>(
                &drawPushConstants_[group.sourceDrawIndex]),
            sizeof(PushConstants));
        indirectDrawItems_.push_back(indirectDraw);
        indirectBufferSignature =
            hashDrawBufferSignature(indirectBufferSignature, indirectDraw);
        indirectSourceDrawIndices_.push_back(group.sourceDrawIndex);
        indirectAlphaMasked_.push_back(
            group.sourceDrawIndex < drawAlphaMasked_.size()
                ? drawAlphaMasked_[group.sourceDrawIndex]
                : 0u);

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
      indirectCommandRing_[frameSlot].buffer->handle(), uploadBytes, 0);
  if (updateIndirectResult.hasError()) {
    return updateIndirectResult;
  }

  indirectPackCache_.valid = true;
  indirectPackCache_.drawSignature = drawSignature;
  indirectPackCache_.requiredBytes = packedRequiredBytes;
  indirectUploadSignatures_[frameSlot] = drawSignature;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
OpaqueRenderer::refreshCachedIndirectPack(uint32_t frameSlot,
                                          uint64_t drawSignature) {
  if (indirectSourceDrawIndices_.size() != indirectDrawItems_.size()) {
    return Result<bool, std::string>::makeError(
        "OpaqueRenderer::buildIndirectDraws: cached indirect source mapping is "
        "invalid");
  }
  indirectAlphaMasked_.resize(indirectDrawItems_.size(), 0u);
  const BufferHandle indirectBufferHandle =
      indirectCommandRing_[frameSlot].buffer->handle();
  uint64_t indirectBufferSignature =
      hashBufferHandleSignature(kFnvOffsetBasis64, indirectBufferHandle);
  for (size_t i = 0; i < indirectSourceDrawIndices_.size(); ++i) {
    const size_t sourceIndex = indirectSourceDrawIndices_[i];
    if (sourceIndex >= drawPushConstants_.size()) {
      return Result<bool, std::string>::makeError(
          "OpaqueRenderer::buildIndirectDraws: cached indirect source index is "
          "out of range");
    }
    indirectDrawItems_[i].indirectBuffer = indirectBufferHandle;
    indirectDrawItems_[i].pushConstants = std::span<const std::byte>(
        reinterpret_cast<const std::byte *>(&drawPushConstants_[sourceIndex]),
        sizeof(PushConstants));
    indirectBufferSignature =
        hashDrawBufferSignature(indirectBufferSignature, indirectDrawItems_[i]);
    indirectAlphaMasked_[i] = sourceIndex < drawAlphaMasked_.size()
                                  ? drawAlphaMasked_[sourceIndex]
                                  : 0u;
    indirectDrawItems_[i].alphaMasked = indirectAlphaMasked_[i] != 0u;
  }
  currentIndirectDrawBufferSignature_ = indirectDrawItems_.empty()
                                            ? kInvalidDrawSignature
                                            : indirectBufferSignature;

  if (indirectUploadSignatures_[frameSlot] != drawSignature) {
    const std::span<const std::byte> uploadBytes{
        indirectCommandUploadBytes_.data(), indirectCommandUploadBytes_.size()};
    auto updateIndirectResult = gpu_.updateBuffer(
        indirectCommandRing_[frameSlot].buffer->handle(), uploadBytes, 0);
    if (updateIndirectResult.hasError()) {
      return updateIndirectResult;
    }
    indirectUploadSignatures_[frameSlot] = drawSignature;
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
    meshShader_.reset();
    meshTessShader_.reset();
    meshDebugOverlayShader_.reset();
    meshPickShader_.reset();
    meshShadowInspectShader_.reset();
    meshVelocityShader_.reset();
    meshReactiveMaskShader_.reset();
    meshNormalShader_.reset();
    depthShader_.reset();
    depthAlphaShader_.reset();
    depthPyramidShader_.reset();
    computeShader_.reset();
    meshVertexShader_ = {};
    meshTessVertexShader_ = {};
    meshTessControlShader_ = {};
    meshTessEvalShader_ = {};
    meshFragmentShader_ = {};
    meshDebugOverlayGeometryShader_ = {};
    meshDebugOverlayFragmentShader_ = {};
    meshPickVertexShader_ = {};
    meshPickTessVertexShader_ = {};
    meshPickTessControlShader_ = {};
    meshPickTessEvalShader_ = {};
    meshPickFragmentShader_ = {};
    meshShadowInspectFragmentShader_ = {};
    meshVelocityVertexShader_ = {};
    meshVelocityFragmentShader_ = {};
    meshReactiveMaskVertexShader_ = {};
    meshReactiveMaskFragmentShader_ = {};
    meshNormalFragmentShader_ = {};
    depthVertexShader_ = {};
    depthTessVertexShader_ = {};
    depthTessControlShader_ = {};
    depthTessEvalShader_ = {};
    depthAlphaVertexShader_ = {};
    depthAlphaTessVertexShader_ = {};
    depthAlphaTessControlShader_ = {};
    depthAlphaTessEvalShader_ = {};
    depthFragmentShader_ = {};
    depthAlphaFragmentShader_ = {};
    depthPyramidVertexShader_ = {};
    depthPyramidFragmentShader_ = {};
    computeShaderHandle_ = {};
    resetMeshPipelineState();
    tessellationUnsupported_ = false;
    return pickTextureResult;
  }

  auto pipelineResult = createPipelines();
  if (pipelineResult.hasError()) {
    resetOverlayPipelineState();
    destroyMeshPipelineState();
    meshPipeline_.reset();
    computePipeline_.reset();
    meshShader_.reset();
    meshTessShader_.reset();
    meshDebugOverlayShader_.reset();
    meshPickShader_.reset();
    meshShadowInspectShader_.reset();
    meshVelocityShader_.reset();
    meshReactiveMaskShader_.reset();
    meshNormalShader_.reset();
    depthShader_.reset();
    depthAlphaShader_.reset();
    depthPyramidShader_.reset();
    computeShader_.reset();
    meshVertexShader_ = {};
    meshTessVertexShader_ = {};
    meshTessControlShader_ = {};
    meshTessEvalShader_ = {};
    meshFragmentShader_ = {};
    meshDebugOverlayGeometryShader_ = {};
    meshDebugOverlayFragmentShader_ = {};
    meshPickVertexShader_ = {};
    meshPickTessVertexShader_ = {};
    meshPickTessControlShader_ = {};
    meshPickTessEvalShader_ = {};
    meshPickFragmentShader_ = {};
    meshShadowInspectFragmentShader_ = {};
    meshVelocityVertexShader_ = {};
    meshVelocityFragmentShader_ = {};
    meshReactiveMaskVertexShader_ = {};
    meshReactiveMaskFragmentShader_ = {};
    meshNormalFragmentShader_ = {};
    depthVertexShader_ = {};
    depthTessVertexShader_ = {};
    depthTessControlShader_ = {};
    depthTessEvalShader_ = {};
    depthAlphaVertexShader_ = {};
    depthAlphaTessVertexShader_ = {};
    depthAlphaTessControlShader_ = {};
    depthAlphaTessEvalShader_ = {};
    depthFragmentShader_ = {};
    depthAlphaFragmentShader_ = {};
    depthPyramidVertexShader_ = {};
    depthPyramidFragmentShader_ = {};
    computeShaderHandle_ = {};
    computePipelineHandle_ = {};
    tessellationUnsupported_ = false;
    destroyPickTexture();
    return pipelineResult;
  }

  auto centersResult = ensureCentersPhaseBufferCapacity(sizeof(glm::vec4));
  if (centersResult.hasError()) {
    return centersResult;
  }
  auto lodBoundsResult =
      ensureInstanceLodBoundsBufferCapacity(sizeof(glm::vec4));
  if (lodBoundsResult.hasError()) {
    return lodBoundsResult;
  }
  auto baseMatricesResult =
      ensureInstanceBaseMatricesBufferCapacity(sizeof(glm::mat4));
  if (baseMatricesResult.hasError()) {
    return baseMatricesResult;
  }

  initialized_ = true;
  return Result<bool, std::string>::makeResult(true);
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

Result<bool, std::string> OpaqueRenderer::ensureDepthPyramidTextures() {
  int32_t framebufferWidth = 0;
  int32_t framebufferHeight = 0;
  gpu_.getFramebufferSize(framebufferWidth, framebufferHeight);
  const uint32_t safeWidth =
      static_cast<uint32_t>(std::max(framebufferWidth, 1));
  const uint32_t safeHeight =
      static_cast<uint32_t>(std::max(framebufferHeight, 1));

  uint32_t levelCount = 1u;
  uint32_t maxDim = std::max(safeWidth, safeHeight);
  while (maxDim > 1u && levelCount < kMaxSceneDepthPyramidLevels) {
    maxDim = (maxDim + 1u) >> 1u;
    ++levelCount;
  }

  auto samplerResult = ensureSceneDepthSampler();
  if (samplerResult.hasError()) {
    return samplerResult;
  }

  const bool recreate = sceneDepthPyramidLevelCount_ != levelCount ||
                        sceneDepthPyramidWidth_ != safeWidth ||
                        sceneDepthPyramidHeight_ != safeHeight;
  if (!recreate) {
    bool allValid = true;
    for (uint32_t i = 0; i < levelCount; ++i) {
      allValid = allValid && nuri::isValid(sceneDepthPyramidTextures_[i]) &&
                 gpu_.isValid(sceneDepthPyramidTextures_[i]);
    }
    if (allValid) {
      return Result<bool, std::string>::makeResult(true);
    }
  }

  destroyDepthPyramidTextures();

  uint32_t width = safeWidth;
  uint32_t height = safeHeight;
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
  if (!nuri::isValid(sceneDepthTexture)) {
    return Result<bool, std::string>::makeError(
        "OpaqueRenderer::ensureTransmissionVisibilityDepthTexture: scene "
        "depth texture is unavailable");
  }

  const TextureDimensions dimensions =
      gpu_.getTextureDimensions(sceneDepthTexture);
  const Format format = gpu_.getTextureFormat(sceneDepthTexture);
  const uint32_t width = std::max(dimensions.width, 1u);
  const uint32_t height = std::max(dimensions.height, 1u);
  bool recreate = !nuri::isValid(transmissionVisibilityDepthTexture_) ||
                  !gpu_.isValid(transmissionVisibilityDepthTexture_);
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
OpaqueRenderer::ensureCentersPhaseBufferCapacity(size_t requiredBytes) {
  const size_t requested = std::max(requiredBytes, sizeof(glm::vec4));
  if (instanceCentersPhaseBuffer_ && instanceCentersPhaseBuffer_->valid() &&
      instanceCentersPhaseBufferCapacityBytes_ >= requested) {
    return Result<bool, std::string>::makeResult(true);
  }

  if (instanceCentersPhaseBuffer_ && instanceCentersPhaseBuffer_->valid()) {
    gpu_.waitIdle();
    gpu_.destroyBuffer(instanceCentersPhaseBuffer_->handle());
    instanceCentersPhaseBuffer_.reset();
    instanceCentersPhaseBufferCapacityBytes_ = 0;
  }

  const BufferDesc desc{
      .usage = BufferUsage::Storage,
      .storage = Storage::Device,
      .size = requested,
  };
  auto createResult =
      Buffer::create(gpu_, desc, "opaque_instance_centers_phase_buffer");
  if (createResult.hasError()) {
    return Result<bool, std::string>::makeError(createResult.error());
  }
  instanceCentersPhaseBuffer_ = std::move(createResult.value());
  instanceCentersPhaseBufferCapacityBytes_ = requested;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
OpaqueRenderer::ensureInstanceLodBoundsBufferCapacity(size_t requiredBytes) {
  const size_t requested = std::max(requiredBytes, sizeof(glm::vec4));
  if (instanceLodBoundsBuffer_ && instanceLodBoundsBuffer_->valid() &&
      instanceLodBoundsBufferCapacityBytes_ >= requested) {
    return Result<bool, std::string>::makeResult(true);
  }

  if (instanceLodBoundsBuffer_ && instanceLodBoundsBuffer_->valid()) {
    gpu_.waitIdle();
    gpu_.destroyBuffer(instanceLodBoundsBuffer_->handle());
    instanceLodBoundsBuffer_.reset();
    instanceLodBoundsBufferCapacityBytes_ = 0;
  }

  const BufferDesc desc{
      .usage = BufferUsage::Storage,
      .storage = Storage::Device,
      .size = requested,
  };
  auto createResult =
      Buffer::create(gpu_, desc, "opaque_instance_lod_bounds_buffer");
  if (createResult.hasError()) {
    return Result<bool, std::string>::makeError(createResult.error());
  }
  instanceLodBoundsBuffer_ = std::move(createResult.value());
  instanceLodBoundsBufferCapacityBytes_ = requested;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
OpaqueRenderer::ensureInstanceBaseMatricesBufferCapacity(size_t requiredBytes) {
  const size_t requested = std::max(requiredBytes, sizeof(glm::mat4));
  if (instanceBaseMatricesBuffer_ && instanceBaseMatricesBuffer_->valid() &&
      instanceBaseMatricesBufferCapacityBytes_ >= requested) {
    return Result<bool, std::string>::makeResult(true);
  }

  if (instanceBaseMatricesBuffer_ && instanceBaseMatricesBuffer_->valid()) {
    gpu_.waitIdle();
    gpu_.destroyBuffer(instanceBaseMatricesBuffer_->handle());
    instanceBaseMatricesBuffer_.reset();
    instanceBaseMatricesBufferCapacityBytes_ = 0;
  }

  const BufferDesc desc{
      .usage = BufferUsage::Storage,
      .storage = Storage::Device,
      .size = requested,
  };
  auto createResult =
      Buffer::create(gpu_, desc, "opaque_instance_base_matrices_buffer");
  if (createResult.hasError()) {
    return Result<bool, std::string>::makeError(createResult.error());
  }
  instanceBaseMatricesBuffer_ = std::move(createResult.value());
  instanceBaseMatricesBufferCapacityBytes_ = requested;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
OpaqueRenderer::ensureRingBufferCount(uint32_t requiredCount) {
  if (requiredCount == 0) {
    requiredCount = 1;
  }
  if (instanceMatricesRing_.size() == requiredCount &&
      previousInstanceMatricesRing_.size() == requiredCount &&
      velocityInstanceFlagsRing_.size() == requiredCount &&
      velocityFrameDataRing_.size() == requiredCount &&
      velocityGeometryRing_.size() == requiredCount &&
      instanceRemapRing_.size() == requiredCount &&
      indirectCommandRing_.size() == requiredCount &&
      meshletBatchRing_.size() == requiredCount &&
      visibilityCandidateRing_.size() == requiredCount &&
      visibilityPassRing_.size() == requiredCount &&
      visibilityVisibleIndexRing_.size() == requiredCount &&
      visibilityCounterRing_.size() == requiredCount &&
      visibilityMeshletDispatchRing_.size() == requiredCount &&
      visibilityMeshletIndirectCommandRing_.size() == requiredCount) {
    return Result<bool, std::string>::makeResult(true);
  }

  gpu_.waitIdle();

  for (DynamicBufferSlot &slot : instanceMatricesRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
  }
  for (DynamicBufferSlot &slot : previousInstanceMatricesRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
  }
  for (DynamicBufferSlot &slot : velocityInstanceFlagsRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
  }
  for (DynamicBufferSlot &slot : velocityFrameDataRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
  }
  for (DynamicBufferSlot &slot : velocityGeometryRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
  }
  for (DynamicBufferSlot &slot : instanceRemapRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
  }
  for (DynamicBufferSlot &slot : indirectCommandRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
  }
  for (DynamicBufferSlot &slot : meshletBatchRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
  }
  for (DynamicBufferSlot &slot : visibilityCandidateRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
  }
  for (DynamicBufferSlot &slot : visibilityPassRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
  }
  for (DynamicBufferSlot &slot : visibilityVisibleIndexRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
  }
  for (DynamicBufferSlot &slot : visibilityCounterRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
  }
  for (DynamicBufferSlot &slot : visibilityMeshletDispatchRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
  }
  for (DynamicBufferSlot &slot : visibilityMeshletIndirectCommandRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
  }

  instanceMatricesRing_.clear();
  previousInstanceMatricesRing_.clear();
  velocityInstanceFlagsRing_.clear();
  velocityFrameDataRing_.clear();
  velocityGeometryRing_.clear();
  instanceRemapRing_.clear();
  indirectCommandRing_.clear();
  meshletBatchRing_.clear();
  visibilityCandidateRing_.clear();
  visibilityPassRing_.clear();
  visibilityVisibleIndexRing_.clear();
  visibilityCounterRing_.clear();
  visibilityMeshletDispatchRing_.clear();
  visibilityMeshletIndirectCommandRing_.clear();
  instanceMatricesRing_.resize(requiredCount);
  previousInstanceMatricesRing_.resize(requiredCount);
  velocityInstanceFlagsRing_.resize(requiredCount);
  velocityFrameDataRing_.resize(requiredCount);
  velocityGeometryRing_.resize(requiredCount);
  instanceRemapRing_.resize(requiredCount);
  indirectCommandRing_.resize(requiredCount);
  meshletBatchRing_.resize(requiredCount);
  visibilityCandidateRing_.resize(requiredCount);
  visibilityPassRing_.resize(requiredCount);
  visibilityVisibleIndexRing_.resize(requiredCount);
  visibilityCounterRing_.resize(requiredCount);
  visibilityMeshletDispatchRing_.resize(requiredCount);
  visibilityMeshletIndirectCommandRing_.resize(requiredCount);
  instanceMatricesUploadVersions_.assign(requiredCount,
                                         std::numeric_limits<uint64_t>::max());
  indirectUploadSignatures_.assign(requiredCount, kInvalidDrawSignature);
  remapUploadSignatures_.assign(requiredCount, kInvalidDrawSignature);
  visibilityCounterRingPublishedFrames_.assign(
      requiredCount, std::numeric_limits<uint64_t>::max());
  visibilityExpectedVisibleIndexCounts_.assign(requiredCount, 0u);
  visibilityExpectedVisibleIndexHashes_.assign(requiredCount,
                                               kFnvOffsetBasis64);
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
OpaqueRenderer::ensureInstanceMatricesRingCapacity(size_t requiredBytes) {
  const size_t requested = std::max(requiredBytes, sizeof(glm::mat4));
  bool needsGrowth = false;
  for (const DynamicBufferSlot &slot : instanceMatricesRing_) {
    if (slot.buffer && slot.buffer->valid() && slot.capacityBytes < requested) {
      needsGrowth = true;
      break;
    }
  }
  if (needsGrowth) {
    gpu_.waitIdle();
  }
  for (size_t i = 0; i < instanceMatricesRing_.size(); ++i) {
    DynamicBufferSlot &slot = instanceMatricesRing_[i];
    if (slot.buffer && slot.buffer->valid() &&
        slot.capacityBytes >= requested) {
      continue;
    }
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
      slot.buffer.reset();
      slot.capacityBytes = 0;
    }

    const BufferDesc desc{
        .usage = BufferUsage::Storage,
        .storage = Storage::Device,
        .size = requested,
    };
    auto createResult = Buffer::create(
        gpu_, desc, "opaque_instance_matrices_buffer_" + std::to_string(i));
    if (createResult.hasError()) {
      return Result<bool, std::string>::makeError(createResult.error());
    }
    slot.buffer = std::move(createResult.value());
    slot.capacityBytes = requested;
    if (i < instanceMatricesUploadVersions_.size()) {
      instanceMatricesUploadVersions_[i] = std::numeric_limits<uint64_t>::max();
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> OpaqueRenderer::ensureDynamicRingCapacity(
    std::pmr::vector<DynamicBufferSlot> &ring, size_t requiredBytes,
    size_t minimumBytes, std::string_view debugNamePrefix) {
  const size_t requested = std::max(requiredBytes, minimumBytes);
  bool needsGrowth = false;
  for (const DynamicBufferSlot &slot : ring) {
    if (slot.buffer && slot.buffer->valid() && slot.capacityBytes < requested) {
      needsGrowth = true;
      break;
    }
  }
  if (needsGrowth) {
    gpu_.waitIdle();
  }
  for (size_t i = 0; i < ring.size(); ++i) {
    DynamicBufferSlot &slot = ring[i];
    if (slot.buffer && slot.buffer->valid() &&
        slot.capacityBytes >= requested) {
      continue;
    }
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
      slot.buffer.reset();
      slot.capacityBytes = 0;
    }

    const BufferDesc desc{
        .usage = BufferUsage::Storage,
        .storage = Storage::Device,
        .size = requested,
    };
    std::string debugName(debugNamePrefix);
    debugName += "_";
    debugName += std::to_string(i);
    auto createResult = Buffer::create(gpu_, desc, debugName);
    if (createResult.hasError()) {
      return Result<bool, std::string>::makeError(createResult.error());
    }
    slot.buffer = std::move(createResult.value());
    slot.capacityBytes = requested;
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
OpaqueRenderer::ensurePreviousInstanceMatricesRingCapacity(
    size_t requiredBytes) {
  return ensureDynamicRingCapacity(previousInstanceMatricesRing_, requiredBytes,
                                   sizeof(InstanceData),
                                   "opaque_previous_instance_matrices_buffer");
}

Result<bool, std::string>
OpaqueRenderer::ensureVelocityInstanceFlagsRingCapacity(size_t requiredBytes) {
  return ensureDynamicRingCapacity(velocityInstanceFlagsRing_, requiredBytes,
                                   sizeof(uint32_t),
                                   "opaque_velocity_instance_flags_buffer");
}

Result<bool, std::string>
OpaqueRenderer::ensureVelocityFrameDataRingCapacity(size_t requiredBytes) {
  return ensureDynamicRingCapacity(velocityFrameDataRing_, requiredBytes,
                                   sizeof(VelocityFrameGpuData),
                                   "opaque_velocity_frame_data_buffer");
}

Result<bool, std::string>
OpaqueRenderer::ensureVelocityGeometryRingCapacity(size_t requiredBytes) {
  return ensureDynamicRingCapacity(velocityGeometryRing_, requiredBytes,
                                   sizeof(VelocityRenderableGeometryGpuData),
                                   "opaque_velocity_geometry_buffer");
}

Result<bool, std::string>
OpaqueRenderer::ensureMeshletBatchRingCapacity(size_t requiredBytes) {
  const size_t requested =
      ((std::max(requiredBytes, sizeof(MeshletBatchGpuData)) +
        sizeof(MeshletBatchGpuData) - 1u) /
       sizeof(MeshletBatchGpuData)) *
      sizeof(MeshletBatchGpuData);
  for (size_t i = 0; i < meshletBatchRing_.size(); ++i) {
    DynamicBufferSlot &slot = meshletBatchRing_[i];
    if (slot.buffer && slot.buffer->valid() &&
        slot.capacityBytes >= requested) {
      continue;
    }
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
      slot.buffer.reset();
      slot.capacityBytes = 0;
    }

    const BufferDesc desc{
        .usage = BufferUsage::Storage,
        .storage = Storage::HostVisible,
        .size = requested,
    };
    std::string debugName("opaque_meshlet_batch_buffer_");
    debugName += std::to_string(i);
    auto createResult = Buffer::create(gpu_, desc, debugName);
    if (createResult.hasError()) {
      return Result<bool, std::string>::makeError(createResult.error());
    }
    slot.buffer = std::move(createResult.value());
    slot.capacityBytes = requested;
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
OpaqueRenderer::ensureVisibilityGpuRingCapacity(size_t candidateBytes,
                                                size_t visibleIndexBytes) {
  auto candidateResult = ensureDynamicRingCapacity(
      visibilityCandidateRing_, candidateBytes, sizeof(VisibilityCandidateGpu),
      "opaque_visibility_candidate_buffer");
  if (candidateResult.hasError()) {
    return candidateResult;
  }
  auto passResult = ensureDynamicRingCapacity(
      visibilityPassRing_, sizeof(VisibilityPassGpuData),
      sizeof(VisibilityPassGpuData), "opaque_visibility_pass_buffer");
  if (passResult.hasError()) {
    return passResult;
  }
  auto visibleResult =
      ensureVisibilityVisibleIndexRingCapacity(visibleIndexBytes);
  if (visibleResult.hasError()) {
    return visibleResult;
  }
  return ensureVisibilityCounterRingCapacity(sizeof(VisibilityCounterGpuData));
}

Result<bool, std::string>
OpaqueRenderer::ensureVisibilityMeshletIndirectRingCapacity(
    size_t dispatchBytes, size_t commandBytes) {
  auto dispatchResult =
      ensureDynamicRingCapacity(visibilityMeshletDispatchRing_, dispatchBytes,
                                sizeof(VisibilityMeshletDispatchGpuData),
                                "opaque_visibility_meshlet_dispatch_buffer");
  if (dispatchResult.hasError()) {
    return dispatchResult;
  }
  return ensureVisibilityMeshletIndirectCommandRingCapacity(commandBytes);
}

Result<bool, std::string>
OpaqueRenderer::ensureVisibilityMeshletIndirectCommandRingCapacity(
    size_t requiredBytes) {
  constexpr size_t kMeshDispatchCommandBytes = sizeof(uint32_t) * 3u;
  const size_t requested =
      ((std::max(requiredBytes, kMeshDispatchCommandBytes) +
        kMeshDispatchCommandBytes - 1u) /
       kMeshDispatchCommandBytes) *
      kMeshDispatchCommandBytes;
  bool needsGrowth = false;
  for (const DynamicBufferSlot &slot : visibilityMeshletIndirectCommandRing_) {
    if (slot.buffer && slot.buffer->valid() && slot.capacityBytes < requested) {
      needsGrowth = true;
      break;
    }
  }
  if (needsGrowth) {
    gpu_.waitIdle();
  }
  for (size_t i = 0; i < visibilityMeshletIndirectCommandRing_.size(); ++i) {
    DynamicBufferSlot &slot = visibilityMeshletIndirectCommandRing_[i];
    if (slot.buffer && slot.buffer->valid() &&
        slot.capacityBytes >= requested) {
      continue;
    }
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
      slot.buffer.reset();
      slot.capacityBytes = 0;
    }

    const BufferDesc desc{
        .usage = BufferUsage::Storage | BufferUsage::Indirect,
        .storage = Storage::Device,
        .size = requested,
    };
    std::string debugName("opaque_visibility_meshlet_indirect_commands_");
    debugName += std::to_string(i);
    auto createResult = Buffer::create(gpu_, desc, debugName);
    if (createResult.hasError()) {
      return Result<bool, std::string>::makeError(createResult.error());
    }
    slot.buffer = std::move(createResult.value());
    slot.capacityBytes = requested;
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
OpaqueRenderer::ensureVisibilityVisibleIndexRingCapacity(size_t requiredBytes) {
  const size_t requested = std::max(requiredBytes, sizeof(uint32_t));
  bool needsGrowth = false;
  for (const DynamicBufferSlot &slot : visibilityVisibleIndexRing_) {
    if (slot.buffer && slot.buffer->valid() && slot.capacityBytes < requested) {
      needsGrowth = true;
      break;
    }
  }
  if (needsGrowth) {
    gpu_.waitIdle();
  }
  for (size_t i = 0; i < visibilityVisibleIndexRing_.size(); ++i) {
    DynamicBufferSlot &slot = visibilityVisibleIndexRing_[i];
    if (slot.buffer && slot.buffer->valid() &&
        slot.capacityBytes >= requested) {
      continue;
    }
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
      slot.buffer.reset();
      slot.capacityBytes = 0;
    }

    const BufferDesc desc{
        .usage = BufferUsage::Storage,
        .storage = Storage::HostVisible,
        .size = requested,
    };
    std::string debugName("opaque_visibility_visible_index_buffer_");
    debugName += std::to_string(i);
    auto createResult = Buffer::create(gpu_, desc, debugName);
    if (createResult.hasError()) {
      return Result<bool, std::string>::makeError(createResult.error());
    }
    slot.buffer = std::move(createResult.value());
    slot.capacityBytes = requested;
    if (i < visibilityCounterRingPublishedFrames_.size()) {
      visibilityCounterRingPublishedFrames_[i] =
          std::numeric_limits<uint64_t>::max();
    }
    if (i < visibilityExpectedVisibleIndexCounts_.size()) {
      visibilityExpectedVisibleIndexCounts_[i] = 0u;
    }
    if (i < visibilityExpectedVisibleIndexHashes_.size()) {
      visibilityExpectedVisibleIndexHashes_[i] = kFnvOffsetBasis64;
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
OpaqueRenderer::ensureVisibilityCounterRingCapacity(size_t requiredBytes) {
  const size_t requested =
      std::max(requiredBytes, sizeof(VisibilityCounterGpuData));
  bool needsGrowth = false;
  for (const DynamicBufferSlot &slot : visibilityCounterRing_) {
    if (slot.buffer && slot.buffer->valid() && slot.capacityBytes < requested) {
      needsGrowth = true;
      break;
    }
  }
  if (needsGrowth) {
    gpu_.waitIdle();
  }
  for (size_t i = 0; i < visibilityCounterRing_.size(); ++i) {
    DynamicBufferSlot &slot = visibilityCounterRing_[i];
    if (slot.buffer && slot.buffer->valid() &&
        slot.capacityBytes >= requested) {
      continue;
    }
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
      slot.buffer.reset();
      slot.capacityBytes = 0;
    }

    const BufferDesc desc{
        .usage = BufferUsage::Storage,
        .storage = Storage::HostVisible,
        .size = requested,
    };
    std::string debugName("opaque_visibility_counter_buffer_");
    debugName += std::to_string(i);
    auto createResult = Buffer::create(gpu_, desc, debugName);
    if (createResult.hasError()) {
      return Result<bool, std::string>::makeError(createResult.error());
    }
    slot.buffer = std::move(createResult.value());
    slot.capacityBytes = requested;
    if (i < visibilityCounterRingPublishedFrames_.size()) {
      visibilityCounterRingPublishedFrames_[i] =
          std::numeric_limits<uint64_t>::max();
    }
    if (i < visibilityExpectedVisibleIndexCounts_.size()) {
      visibilityExpectedVisibleIndexCounts_[i] = 0u;
    }
    if (i < visibilityExpectedVisibleIndexHashes_.size()) {
      visibilityExpectedVisibleIndexHashes_[i] = kFnvOffsetBasis64;
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
OpaqueRenderer::ensureInstanceRemapRingCapacity(size_t requiredBytes) {
  const size_t requested = std::max(requiredBytes, sizeof(uint32_t));
  bool needsGrowth = false;
  for (const DynamicBufferSlot &slot : instanceRemapRing_) {
    if (slot.buffer && slot.buffer->valid() && slot.capacityBytes < requested) {
      needsGrowth = true;
      break;
    }
  }
  if (needsGrowth) {
    gpu_.waitIdle();
  }
  for (size_t i = 0; i < instanceRemapRing_.size(); ++i) {
    DynamicBufferSlot &slot = instanceRemapRing_[i];
    if (slot.buffer && slot.buffer->valid() &&
        slot.capacityBytes >= requested) {
      continue;
    }
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
      slot.buffer.reset();
      slot.capacityBytes = 0;
    }

    const BufferDesc desc{
        .usage = BufferUsage::Storage,
        .storage = Storage::Device,
        .size = requested,
    };
    auto createResult = Buffer::create(
        gpu_, desc, "opaque_instance_remap_buffer_" + std::to_string(i));
    if (createResult.hasError()) {
      return Result<bool, std::string>::makeError(createResult.error());
    }
    slot.buffer = std::move(createResult.value());
    slot.capacityBytes = requested;
    if (i < remapUploadSignatures_.size()) {
      remapUploadSignatures_[i] = kInvalidDrawSignature;
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
OpaqueRenderer::ensureIndirectCommandRingCapacity(size_t requiredBytes) {
  const size_t requested = std::max(requiredBytes, kIndirectCountHeaderBytes);
  bool needsGrowth = false;
  for (const DynamicBufferSlot &slot : indirectCommandRing_) {
    if (slot.buffer && slot.buffer->valid() && slot.capacityBytes < requested) {
      needsGrowth = true;
      break;
    }
  }
  if (needsGrowth) {
    gpu_.waitIdle();
  }
  for (size_t i = 0; i < indirectCommandRing_.size(); ++i) {
    DynamicBufferSlot &slot = indirectCommandRing_[i];
    if (slot.buffer && slot.buffer->valid() &&
        slot.capacityBytes >= requested) {
      continue;
    }
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
      slot.buffer.reset();
      slot.capacityBytes = 0;
    }

    const BufferDesc desc{
        .usage = BufferUsage::Storage | BufferUsage::Indirect,
        .storage = Storage::Device,
        .size = requested,
    };
    auto createResult = Buffer::create(
        gpu_, desc, "opaque_indirect_commands_buffer_" + std::to_string(i));
    if (createResult.hasError()) {
      return Result<bool, std::string>::makeError(createResult.error());
    }
    slot.buffer = std::move(createResult.value());
    slot.capacityBytes = requested;
    if (i < indirectUploadSignatures_.size()) {
      indirectUploadSignatures_[i] = kInvalidDrawSignature;
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> OpaqueRenderer::rebuildSceneCache(
    const RenderScene &scene, const ResourceManager &resources,
    uint32_t materialCount, bool excludeTransmission) {
  renderableTemplates_.clear();
  meshDrawTemplates_.clear();

  const std::span<const Renderable> renderables = scene.renderables();
  if (renderables.size() >
      static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    return Result<bool, std::string>::makeError(
        "OpaqueRenderer::rebuildSceneCache: renderables count exceeds "
        "UINT32_MAX");
  }
  renderableTemplates_.reserve(renderables.size());

  size_t totalMeshDraws = 0;
  for (const Renderable &renderable : renderables) {
    const ModelRecord *modelRecord = resources.tryGet(renderable.model);
    if (!modelRecord || !modelRecord->model) {
      return Result<bool, std::string>::makeError(
          "OpaqueRenderer::rebuildSceneCache: renderable model handle is "
          "invalid");
    }
    totalMeshDraws += modelRecord->model->submeshes().size();
  }
  meshDrawTemplates_.reserve(totalMeshDraws);
  size_t invalidMaterialFallbackCount = 0;
  size_t skippedBlendSubmeshCount = 0;

  for (uint32_t index = 0; index < static_cast<uint32_t>(renderables.size());
       ++index) {
    const Renderable &renderable = renderables[index];
    const ModelRecord *modelRecord = resources.tryGet(renderable.model);
    if (!modelRecord || !modelRecord->model) {
      return Result<bool, std::string>::makeError(
          "OpaqueRenderer::rebuildSceneCache: failed to resolve model handle");
    }
    const Model *model = modelRecord->model.get();
    GeometryAllocationView geometry{};
    if (!gpu_.resolveGeometry(model->geometryHandle(), geometry)) {
      return Result<bool, std::string>::makeError(
          "OpaqueRenderer::rebuildSceneCache: failed to resolve geometry "
          "allocation");
    }
    if (!nuri::isValid(geometry.vertexBuffer) ||
        !nuri::isValid(geometry.indexBuffer)) {
      return Result<bool, std::string>::makeError(
          "OpaqueRenderer::rebuildSceneCache: resolved geometry uses invalid "
          "buffers");
    }
    const uint64_t vertexBufferAddress = gpu_.getBufferDeviceAddress(
        geometry.vertexBuffer, geometry.vertexByteOffset);
    if (vertexBufferAddress == 0) {
      return Result<bool, std::string>::makeError(
          "OpaqueRenderer::rebuildSceneCache: invalid geometry vertex buffer "
          "address");
    }

    renderableTemplates_.push_back(
        RenderableTemplate{.renderable = &renderable, .model = model});

    const std::span<const Submesh> submeshes = model->submeshes();
    for (size_t submeshIndex = 0; submeshIndex < submeshes.size();
         ++submeshIndex) {
      const MaterialRef resolvedMaterial = resolveRenderableMaterial(
          renderable, *modelRecord, static_cast<uint32_t>(submeshIndex));
      const MaterialRecord *materialRecord = resources.tryGet(resolvedMaterial);
      const bool doubleSided =
          materialRecord != nullptr && materialRecord->desc.doubleSided;
      const bool alphaMasked =
          materialRecord != nullptr &&
          materialRecord->desc.alphaMode == MaterialAlphaMode::Mask;
      if (materialRecord != nullptr &&
          (materialRecord->desc.alphaMode == MaterialAlphaMode::Blend ||
           (excludeTransmission && isTransmissionMaterial(*materialRecord)))) {
        ++skippedBlendSubmeshCount;
        continue;
      }
      uint32_t finalMaterialIndex =
          resources.materialTableIndex(resolvedMaterial);
      if (materialCount == 0u || finalMaterialIndex >= materialCount) {
        finalMaterialIndex = 0u;
        ++invalidMaterialFallbackCount;
      }
      meshDrawTemplates_.push_back(MeshDrawTemplate{
          .renderable = &renderable,
          .submesh = &submeshes[submeshIndex],
          .submeshIndex = static_cast<uint32_t>(submeshIndex),
          .instanceIndex = index,
          .geometryHandle = model->geometryHandle(),
          .indexBuffer = geometry.indexBuffer,
          .indexBufferOffset = geometry.indexByteOffset,
          .baseVertexBuffer = geometry.vertexBuffer,
          .vertexBuffer = geometry.vertexBuffer,
          .baseVertexDecodeBuffer = model->vertexDecodeBuffer(),
          .vertexDecodeBuffer = model->vertexDecodeBuffer(),
          .baseVertexBufferAddress = vertexBufferAddress,
          .baseVertexDecodeBufferAddress = model->vertexDecodeBufferAddress(),
          .vertexBufferAddress = vertexBufferAddress,
          .vertexDecodeBufferAddress = model->vertexDecodeBufferAddress(),
          .basePackedVertexFormat =
              static_cast<uint32_t>(model->drawVertexFormat()),
          .vertexDecodeIndex = static_cast<uint32_t>(submeshIndex),
          .packedVertexFormat =
              static_cast<uint32_t>(model->drawVertexFormat()),
          .materialIndex = finalMaterialIndex,
          .meshletView =
              model->hasMeshlets() ? &model->meshletGpuView() : nullptr,
          .doubleSided = doubleSided,
          .alphaMasked = alphaMasked,
      });
    }
  }

  if (invalidMaterialFallbackCount > 0u) {
    if (!loggedMaterialFallbackWarning_) {
      NURI_LOG_WARNING(
          "OpaqueRenderer::rebuildSceneCache: %zu submesh draw(s) used "
          "fallback "
          "material index 0 due to missing/out-of-range material mapping",
          invalidMaterialFallbackCount);
      loggedMaterialFallbackWarning_ = true;
    }
  } else {
    loggedMaterialFallbackWarning_ = false;
  }
  loggedBlendMaterialUnsupportedWarning_ = false;

  cachedScene_ = &scene;
  cachedTopologyVersion_ = scene.topologyVersion();
  cachedGeometryMutationVersion_ = gpu_.geometryMutationVersion();
  cachedExcludeTransmission_ = excludeTransmission;
  uniformSingleSubmeshPath_ = false;
  if (!meshDrawTemplates_.empty() &&
      meshDrawTemplates_.size() == renderableTemplates_.size()) {
    const MeshDrawTemplate &first = meshDrawTemplates_.front();
    uniformSingleSubmeshPath_ = true;
    for (size_t i = 0; i < meshDrawTemplates_.size(); ++i) {
      const MeshDrawTemplate &entry = meshDrawTemplates_[i];
      if (entry.instanceIndex != i ||
          entry.geometryHandle.index != first.geometryHandle.index ||
          entry.geometryHandle.generation != first.geometryHandle.generation ||
          entry.submesh != first.submesh ||
          entry.materialIndex != first.materialIndex) {
        uniformSingleSubmeshPath_ = false;
        break;
      }
    }
  }
  instanceStaticBuffersDirty_ = true;
  invalidateSingleInstanceBatchCache();
  invalidateStaticBatchCache();
  invalidateIndirectPackCache();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> OpaqueRenderer::rebuildMaterialTextureAccessCache(
    const RenderScene &scene, const ResourceManager &resources,
    bool excludeTransmission) {
  NURI_PROFILER_FUNCTION();
  materialTextureAccessHandles_.clear();
  const std::span<const Renderable> renderables = scene.renderables();
  if (renderables.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }

  ScratchArena scratch;
  ScopedScratch scopedScratch(scratch);
  PmrHashSet<uint64_t> textureKeys(scopedScratch.resource());
  textureKeys.reserve(renderables.size());
  materialTextureAccessHandles_.reserve(renderables.size());

  for (const Renderable &renderable : renderables) {
    const ModelRecord *modelRecord = resources.tryGet(renderable.model);
    if (modelRecord == nullptr || modelRecord->model == nullptr) {
      continue;
    }
    for (size_t submeshIndex = 0;
         submeshIndex < modelRecord->model->submeshes().size();
         ++submeshIndex) {
      const MaterialRef resolvedMaterial = resolveRenderableMaterial(
          renderable, *modelRecord, static_cast<uint32_t>(submeshIndex));
      const MaterialRecord *materialRecord = resources.tryGet(resolvedMaterial);
      if (materialRecord == nullptr ||
          materialRecord->desc.alphaMode == MaterialAlphaMode::Blend ||
          (excludeTransmission && isTransmissionMaterial(*materialRecord))) {
        continue;
      }
      forEachMaterialTextureRef(
          materialRecord->textureRefs, [&](TextureRef ref) {
            const TextureRecord *record = resources.tryGet(ref);
            if (record == nullptr || !nuri::isValid(record->texture)) {
              return;
            }
            const uint64_t key =
                foldHandle(record->texture.index, record->texture.generation);
            if (!textureKeys.insert(key).second) {
              return;
            }
            materialTextureAccessHandles_.push_back(record->texture);
          });
    }
  }

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> OpaqueRenderer::createShaders() {
  meshShader_ = Shader::create("main", gpu_);
  meshTessShader_ = Shader::create("main_tess", gpu_);
  meshDebugOverlayShader_ = Shader::create("mesh_debug_overlay", gpu_);
  meshPickShader_ = Shader::create("main_id", gpu_);
  meshShadowInspectShader_ = Shader::create("shadow_inspect", gpu_);
  meshVelocityShader_ = Shader::create("opaque_velocity", gpu_);
  meshReactiveMaskShader_ = Shader::create("opaque_reactive_mask", gpu_);
  meshNormalShader_ = Shader::create("opaque_normal", gpu_);
  depthShader_ = Shader::create("opaque_depth", gpu_);
  depthAlphaShader_ = Shader::create("opaque_depth_alpha", gpu_);
  depthPyramidShader_ = Shader::create("depth_minmax_pyramid", gpu_);
  computeShader_ = Shader::create("duck_instances", gpu_);
  visibilityShader_ = Shader::create("visibility_cull", gpu_);
  visibilityIndirectDrawShader_ =
      Shader::create("visibility_indirect_draw", gpu_);
  visibilityIndirectMeshDispatchShader_ =
      Shader::create("visibility_indirect_mesh_dispatch", gpu_);
  if (!meshShader_ || !meshTessShader_ || !meshPickShader_ ||
      !meshShadowInspectShader_ || !meshVelocityShader_ ||
      !meshReactiveMaskShader_ || !meshNormalShader_ || !computeShader_ ||
      !depthShader_ || !depthAlphaShader_ || !depthPyramidShader_) {
    return Result<bool, std::string>::makeError(
        "OpaqueRenderer::createShaders: failed to create shader objects");
  }

  meshVertexShader_ = {};
  meshTessVertexShader_ = {};
  meshTessControlShader_ = {};
  meshTessEvalShader_ = {};
  meshFragmentShader_ = {};
  meshDebugOverlayGeometryShader_ = {};
  meshDebugOverlayFragmentShader_ = {};
  meshPickVertexShader_ = {};
  meshPickFragmentShader_ = {};
  meshPickTessVertexShader_ = {};
  meshPickTessControlShader_ = {};
  meshPickTessEvalShader_ = {};
  meshShadowInspectFragmentShader_ = {};
  meshVelocityVertexShader_ = {};
  meshVelocityTessVertexShader_ = {};
  meshVelocityTessControlShader_ = {};
  meshVelocityTessEvalShader_ = {};
  meshVelocityFragmentShader_ = {};
  meshReactiveMaskVertexShader_ = {};
  meshReactiveMaskFragmentShader_ = {};
  meshNormalFragmentShader_ = {};
  depthVertexShader_ = {};
  depthFragmentShader_ = {};
  depthTessVertexShader_ = {};
  depthTessControlShader_ = {};
  depthTessEvalShader_ = {};
  depthAlphaVertexShader_ = {};
  depthAlphaTessVertexShader_ = {};
  depthAlphaTessControlShader_ = {};
  depthAlphaTessEvalShader_ = {};
  depthAlphaFragmentShader_ = {};
  depthPyramidVertexShader_ = {};
  depthPyramidFragmentShader_ = {};
  computeShaderHandle_ = {};
  visibilityComputeShader_ = {};
  visibilityIndirectDrawComputeShader_ = {};
  visibilityIndirectMeshDispatchComputeShader_ = {};
  tessellationUnsupported_ = false;
  gsOverlayPipelineUnsupported_ = false;
  gsTessOverlayPipelineUnsupported_ = false;

  struct ShaderSpec {
    Shader *shader = nullptr;
    const std::filesystem::path *path = nullptr;
    ShaderStage stage = ShaderStage::Vertex;
    ShaderHandle *outHandle = nullptr;
  };
  const std::array<ShaderSpec, 3> shaderSpecs = {
      ShaderSpec{meshShader_.get(), &config_.meshVertex, ShaderStage::Vertex,
                 &meshVertexShader_},
      ShaderSpec{meshShader_.get(), &config_.meshFragment,
                 ShaderStage::Fragment, &meshFragmentShader_},
      ShaderSpec{computeShader_.get(), &config_.computeInstances,
                 ShaderStage::Compute, &computeShaderHandle_},
  };

  for (const ShaderSpec &spec : shaderSpecs) {
    if (!spec.shader || !spec.outHandle || !spec.path) {
      return Result<bool, std::string>::makeError(
          "OpaqueRenderer::createShaders: invalid shader spec");
    }
    const std::string shaderPath = spec.path->string();
    auto compileResult = spec.shader->compileFromFile(shaderPath, spec.stage);
    if (compileResult.hasError()) {
      return Result<bool, std::string>::makeError(compileResult.error());
    }
    *spec.outHandle = compileResult.value();
  }

  if (visibilityShader_) {
    const std::filesystem::path shaderDir = config_.meshFragment.parent_path();
    auto visibilityResult = visibilityShader_->compileFromFile(
        (shaderDir / "visibility_cull.comp").string(), ShaderStage::Compute);
    if (visibilityResult.hasError()) {
      if (!loggedVisibilityGpuUnsupportedWarning_) {
        loggedVisibilityGpuUnsupportedWarning_ = true;
        NURI_LOG_WARNING(
            "OpaqueRenderer::createShaders: visibility culling shader failed, "
            "GPU visibility pass disabled: %s",
            visibilityResult.error().c_str());
      }
      visibilityComputeShader_ = {};
    } else {
      visibilityComputeShader_ = visibilityResult.value();
    }
  }

  if (visibilityIndirectDrawShader_) {
    const std::filesystem::path shaderDir = config_.meshFragment.parent_path();
    auto indirectResult = visibilityIndirectDrawShader_->compileFromFile(
        (shaderDir / "visibility_indirect_draw.comp").string(),
        ShaderStage::Compute);
    if (indirectResult.hasError()) {
      if (!loggedVisibilityGpuUnsupportedWarning_) {
        loggedVisibilityGpuUnsupportedWarning_ = true;
        NURI_LOG_WARNING(
            "OpaqueRenderer::createShaders: visibility indirect draw shader "
            "failed, GPU indirect draw disabled: %s",
            indirectResult.error().c_str());
      }
      visibilityIndirectDrawComputeShader_ = {};
    } else {
      visibilityIndirectDrawComputeShader_ = indirectResult.value();
    }
  }

  if (visibilityIndirectMeshDispatchShader_) {
    const std::filesystem::path shaderDir = config_.meshFragment.parent_path();
    auto indirectMeshResult =
        visibilityIndirectMeshDispatchShader_->compileFromFile(
            (shaderDir / "visibility_indirect_mesh_dispatch.comp").string(),
            ShaderStage::Compute);
    if (indirectMeshResult.hasError()) {
      if (!loggedVisibilityGpuUnsupportedWarning_) {
        loggedVisibilityGpuUnsupportedWarning_ = true;
        NURI_LOG_WARNING(
            "OpaqueRenderer::createShaders: visibility indirect mesh dispatch "
            "shader failed, GPU mesh dispatch args disabled: %s",
            indirectMeshResult.error().c_str());
      }
      visibilityIndirectMeshDispatchComputeShader_ = {};
    } else {
      visibilityIndirectMeshDispatchComputeShader_ = indirectMeshResult.value();
    }
  }

  {
    const std::filesystem::path shaderDir = config_.meshFragment.parent_path();
    auto vertexResult = meshPickShader_->compileFromFile(
        (shaderDir / "main_id_only.vert").string(), ShaderStage::Vertex);
    auto fragmentResult = meshPickShader_->compileFromFile(
        config_.pickFragment.string(), ShaderStage::Fragment);
    if (vertexResult.hasError() || fragmentResult.hasError()) {
      return Result<bool, std::string>::makeError(vertexResult.hasError()
                                                      ? vertexResult.error()
                                                      : fragmentResult.error());
    }
    meshPickVertexShader_ = vertexResult.value();
    meshPickFragmentShader_ = fragmentResult.value();
  }

  {
    const std::filesystem::path shadowInspectPath =
        config_.shadowInspectFragment.empty()
            ? config_.meshFragment.parent_path() / "shadow_inspect.frag"
            : config_.shadowInspectFragment;
    auto compileResult = meshShadowInspectShader_->compileFromFile(
        shadowInspectPath.string(), ShaderStage::Fragment);
    if (compileResult.hasError()) {
      return Result<bool, std::string>::makeError(compileResult.error());
    }
    meshShadowInspectFragmentShader_ = compileResult.value();
  }

  {
    const std::filesystem::path shaderDir = config_.meshFragment.parent_path();
    auto vertexResult = meshVelocityShader_->compileFromFile(
        (shaderDir / "opaque_velocity.vert").string(), ShaderStage::Vertex);
    auto fragmentResult = meshVelocityShader_->compileFromFile(
        (shaderDir / "opaque_velocity.frag").string(), ShaderStage::Fragment);
    auto tessVertexResult = meshVelocityShader_->compileFromFile(
        (shaderDir / "opaque_velocity_tess.vert").string(),
        ShaderStage::Vertex);
    auto tessControlResult = meshVelocityShader_->compileFromFile(
        (shaderDir / "opaque_velocity_tess.tesc").string(),
        ShaderStage::TessControl);
    auto tessEvalResult = meshVelocityShader_->compileFromFile(
        (shaderDir / "opaque_velocity_tess.tese").string(),
        ShaderStage::TessEval);
    if (vertexResult.hasError() || fragmentResult.hasError()) {
      const std::string error = vertexResult.hasError()
                                    ? vertexResult.error()
                                    : fragmentResult.error();
      return Result<bool, std::string>::makeError(
          "OpaqueRenderer::createShaders: velocity shader compile failed: " +
          error);
    }
    meshVelocityVertexShader_ = vertexResult.value();
    meshVelocityFragmentShader_ = fragmentResult.value();

    if (tessVertexResult.hasError() || tessControlResult.hasError() ||
        tessEvalResult.hasError()) {
      const std::string error =
          tessVertexResult.hasError()
              ? tessVertexResult.error()
              : (tessControlResult.hasError() ? tessControlResult.error()
                                              : tessEvalResult.error());
      return Result<bool, std::string>::makeError(
          "OpaqueRenderer::createShaders: tessellated velocity shader compile "
          "failed: " +
          error);
    }
    meshVelocityTessVertexShader_ = tessVertexResult.value();
    meshVelocityTessControlShader_ = tessControlResult.value();
    meshVelocityTessEvalShader_ = tessEvalResult.value();
  }

  {
    const std::filesystem::path shaderDir = config_.meshFragment.parent_path();
    auto vertexResult = meshReactiveMaskShader_->compileFromFile(
        (shaderDir / "opaque_reactive_mask.vert").string(),
        ShaderStage::Vertex);
    auto fragmentResult = meshReactiveMaskShader_->compileFromFile(
        (shaderDir / "opaque_reactive_mask.frag").string(),
        ShaderStage::Fragment);
    if (vertexResult.hasError() || fragmentResult.hasError()) {
      const std::string error = vertexResult.hasError()
                                    ? vertexResult.error()
                                    : fragmentResult.error();
      NURI_LOG_WARNING(
          "OpaqueRenderer::createShaders: reactive mask shader failed, "
          "alpha-mask reactive tracking will be disabled: %s",
          error.c_str());
      meshReactiveMaskVertexShader_ = {};
      meshReactiveMaskFragmentShader_ = {};
    } else {
      meshReactiveMaskVertexShader_ = vertexResult.value();
      meshReactiveMaskFragmentShader_ = fragmentResult.value();
    }
  }

  {
    const std::filesystem::path shaderDir = config_.meshFragment.parent_path();
    auto fragmentResult = meshNormalShader_->compileFromFile(
        (shaderDir / "opaque_normal.frag").string(), ShaderStage::Fragment);
    if (fragmentResult.hasError()) {
      NURI_LOG_WARNING(
          "OpaqueRenderer::createShaders: material normal shader failed, "
          "GTAO normal pre-pass will be disabled: %s",
          fragmentResult.error().c_str());
      meshNormalFragmentShader_ = {};
    } else {
      meshNormalFragmentShader_ = fragmentResult.value();
    }
  }

  {
    const std::filesystem::path shaderDir = config_.meshFragment.parent_path();
    auto depthVertexCompileResult = depthShader_->compileFromFile(
        (shaderDir / "opaque_depth.vert").string(), ShaderStage::Vertex);
    if (depthVertexCompileResult.hasError()) {
      NURI_LOG_WARNING(
          "OpaqueRenderer::createShaders: depth pre-pass vertex shader "
          "failed, depth pre-pass will be disabled: %s",
          depthVertexCompileResult.error().c_str());
      depthVertexShader_ = {};
    } else {
      depthVertexShader_ = depthVertexCompileResult.value();
    }

    const std::filesystem::path depthPath = shaderDir / "opaque_depth.frag";
    auto depthCompileResult = depthShader_->compileFromFile(
        depthPath.string(), ShaderStage::Fragment);
    if (depthCompileResult.hasError()) {
      NURI_LOG_WARNING(
          "OpaqueRenderer::createShaders: depth pre-pass shader failed, "
          "depth pre-pass will be disabled: %s",
          depthCompileResult.error().c_str());
      depthFragmentShader_ = {};
    } else {
      depthFragmentShader_ = depthCompileResult.value();
    }

    auto depthAlphaVertexCompileResult = depthAlphaShader_->compileFromFile(
        (shaderDir / "opaque_depth_alpha.vert").string(), ShaderStage::Vertex);
    if (depthAlphaVertexCompileResult.hasError()) {
      NURI_LOG_WARNING(
          "OpaqueRenderer::createShaders: masked depth pre-pass vertex "
          "shader failed, masked depth pre-pass will be disabled: %s",
          depthAlphaVertexCompileResult.error().c_str());
      depthAlphaVertexShader_ = {};
    } else {
      depthAlphaVertexShader_ = depthAlphaVertexCompileResult.value();
    }

    const std::filesystem::path depthAlphaPath =
        shaderDir / "opaque_depth_alpha.frag";
    auto compileResult = depthAlphaShader_->compileFromFile(
        depthAlphaPath.string(), ShaderStage::Fragment);
    if (compileResult.hasError()) {
      NURI_LOG_WARNING(
          "OpaqueRenderer::createShaders: masked depth pre-pass shader "
          "failed, depth pre-pass will be disabled: %s",
          compileResult.error().c_str());
      depthAlphaFragmentShader_ = {};
    } else {
      depthAlphaFragmentShader_ = compileResult.value();
    }

    const std::filesystem::path fullscreenVertexPath =
        shaderDir / "fullscreen_copy.vert";
    auto vertexResult = depthPyramidShader_->compileFromFile(
        fullscreenVertexPath.string(), ShaderStage::Vertex);
    auto fragmentResult = depthPyramidShader_->compileFromFile(
        (shaderDir / "depth_minmax_pyramid.frag").string(),
        ShaderStage::Fragment);
    if (vertexResult.hasError() || fragmentResult.hasError()) {
      const std::string error = vertexResult.hasError()
                                    ? vertexResult.error()
                                    : fragmentResult.error();
      NURI_LOG_WARNING(
          "OpaqueRenderer::createShaders: depth pyramid shaders failed, depth "
          "pyramid will be disabled: %s",
          error.c_str());
      depthPyramidVertexShader_ = {};
      depthPyramidFragmentShader_ = {};
    } else {
      depthPyramidVertexShader_ = vertexResult.value();
      depthPyramidFragmentShader_ = fragmentResult.value();
    }
  }

  const std::array<ShaderSpec, 3> tessShaderSpecs = {
      ShaderSpec{meshTessShader_.get(), &config_.tessVertex,
                 ShaderStage::Vertex, &meshTessVertexShader_},
      ShaderSpec{meshTessShader_.get(), &config_.tessControl,
                 ShaderStage::TessControl, &meshTessControlShader_},
      ShaderSpec{meshTessShader_.get(), &config_.tessEval,
                 ShaderStage::TessEval, &meshTessEvalShader_},
  };
  for (const ShaderSpec &spec : tessShaderSpecs) {
    if (!spec.shader || !spec.outHandle || !spec.path) {
      tessellationUnsupported_ = true;
      meshTessVertexShader_ = {};
      meshTessControlShader_ = {};
      meshTessEvalShader_ = {};
      NURI_LOG_WARNING(
          "OpaqueRenderer::createShaders: invalid tessellation shader spec");
      break;
    }

    const std::string shaderPath = spec.path->string();
    auto compileResult = spec.shader->compileFromFile(shaderPath, spec.stage);
    if (compileResult.hasError()) {
      tessellationUnsupported_ = true;
      meshTessVertexShader_ = {};
      meshTessControlShader_ = {};
      meshTessEvalShader_ = {};
      NURI_LOG_WARNING(
          "OpaqueRenderer::createShaders: Tessellation shader path "
          "'%s' failed, fallback to non-tessellation path: %s",
          shaderPath.c_str(), compileResult.error().c_str());
      break;
    }
    *spec.outHandle = compileResult.value();
  }

  if (!tessellationUnsupported_) {
    const std::filesystem::path shaderDir = config_.meshFragment.parent_path();
    const std::array<std::filesystem::path, 3> pickTessPaths = {
        shaderDir / "main_id_tess.vert", shaderDir / "main_id_tess.tesc",
        shaderDir / "main_id_tess.tese"};
    const std::array<ShaderSpec, 3> pickTessShaderSpecs = {
        ShaderSpec{meshPickShader_.get(), &pickTessPaths[0],
                   ShaderStage::Vertex, &meshPickTessVertexShader_},
        ShaderSpec{meshPickShader_.get(), &pickTessPaths[1],
                   ShaderStage::TessControl, &meshPickTessControlShader_},
        ShaderSpec{meshPickShader_.get(), &pickTessPaths[2],
                   ShaderStage::TessEval, &meshPickTessEvalShader_},
    };
    for (const ShaderSpec &spec : pickTessShaderSpecs) {
      const std::string shaderPath = spec.path->string();
      auto compileResult = spec.shader->compileFromFile(shaderPath, spec.stage);
      if (compileResult.hasError()) {
        meshPickTessVertexShader_ = {};
        meshPickTessControlShader_ = {};
        meshPickTessEvalShader_ = {};
        NURI_LOG_WARNING(
            "OpaqueRenderer::createShaders: pick tessellation shader path "
            "'%s' failed, tessellated picking will fall back to the "
            "non-tessellation path: %s",
            shaderPath.c_str(), compileResult.error().c_str());
        break;
      }
      *spec.outHandle = compileResult.value();
    }

    const std::array<std::filesystem::path, 3> depthAlphaTessPaths = {
        shaderDir / "opaque_depth_alpha_tess.vert",
        shaderDir / "opaque_depth_alpha_tess.tesc",
        shaderDir / "opaque_depth_alpha_tess.tese"};
    const std::array<ShaderSpec, 3> depthAlphaTessShaderSpecs = {
        ShaderSpec{depthAlphaShader_.get(), &depthAlphaTessPaths[0],
                   ShaderStage::Vertex, &depthAlphaTessVertexShader_},
        ShaderSpec{depthAlphaShader_.get(), &depthAlphaTessPaths[1],
                   ShaderStage::TessControl, &depthAlphaTessControlShader_},
        ShaderSpec{depthAlphaShader_.get(), &depthAlphaTessPaths[2],
                   ShaderStage::TessEval, &depthAlphaTessEvalShader_},
    };
    for (const ShaderSpec &spec : depthAlphaTessShaderSpecs) {
      const std::string shaderPath = spec.path->string();
      auto compileResult = spec.shader->compileFromFile(shaderPath, spec.stage);
      if (compileResult.hasError()) {
        depthAlphaTessVertexShader_ = {};
        depthAlphaTessControlShader_ = {};
        depthAlphaTessEvalShader_ = {};
        NURI_LOG_WARNING(
            "OpaqueRenderer::createShaders: masked depth tessellation shader "
            "path '%s' failed, tessellated masked depth pre-pass will fall "
            "back to the non-tessellation depth path: %s",
            shaderPath.c_str(), compileResult.error().c_str());
        break;
      }
      *spec.outHandle = compileResult.value();
    }

    const std::array<std::filesystem::path, 3> depthTessPaths = {
        shaderDir / "opaque_depth_tess.vert",
        shaderDir / "opaque_depth_tess.tesc",
        shaderDir / "opaque_depth_tess.tese"};
    const std::array<ShaderSpec, 3> depthTessShaderSpecs = {
        ShaderSpec{depthShader_.get(), &depthTessPaths[0], ShaderStage::Vertex,
                   &depthTessVertexShader_},
        ShaderSpec{depthShader_.get(), &depthTessPaths[1],
                   ShaderStage::TessControl, &depthTessControlShader_},
        ShaderSpec{depthShader_.get(), &depthTessPaths[2],
                   ShaderStage::TessEval, &depthTessEvalShader_},
    };
    for (const ShaderSpec &spec : depthTessShaderSpecs) {
      const std::string shaderPath = spec.path->string();
      auto compileResult = spec.shader->compileFromFile(shaderPath, spec.stage);
      if (compileResult.hasError()) {
        depthTessVertexShader_ = {};
        depthTessControlShader_ = {};
        depthTessEvalShader_ = {};
        NURI_LOG_WARNING(
            "OpaqueRenderer::createShaders: depth tessellation shader path "
            "'%s' failed, tessellated depth pre-pass will fall back to the "
            "non-tessellation depth path: %s",
            shaderPath.c_str(), compileResult.error().c_str());
        break;
      }
      *spec.outHandle = compileResult.value();
    }
  }

  if (!meshDebugOverlayShader_) {
    gsOverlayPipelineUnsupported_ = true;
    gsTessOverlayPipelineUnsupported_ = true;
    NURI_LOG_WARNING("OpaqueRenderer::createShaders: failed to create debug "
                     "overlay shader object, fallback to line pipelines");
    return Result<bool, std::string>::makeResult(true);
  }

  const std::array<ShaderSpec, 2> overlayShaderSpecs = {
      ShaderSpec{meshDebugOverlayShader_.get(), &config_.overlayGeometry,
                 ShaderStage::Geometry, &meshDebugOverlayGeometryShader_},
      ShaderSpec{meshDebugOverlayShader_.get(), &config_.overlayFragment,
                 ShaderStage::Fragment, &meshDebugOverlayFragmentShader_},
  };
  for (const ShaderSpec &spec : overlayShaderSpecs) {
    if (!spec.shader || !spec.outHandle || !spec.path) {
      gsOverlayPipelineUnsupported_ = true;
      gsTessOverlayPipelineUnsupported_ = true;
      meshDebugOverlayGeometryShader_ = {};
      meshDebugOverlayFragmentShader_ = {};
      NURI_LOG_WARNING(
          "OpaqueRenderer::createShaders: invalid debug overlay shader spec");
      break;
    }
    const std::string shaderPath = spec.path->string();
    auto compileResult = spec.shader->compileFromFile(shaderPath, spec.stage);
    if (compileResult.hasError()) {
      gsOverlayPipelineUnsupported_ = true;
      gsTessOverlayPipelineUnsupported_ = true;
      meshDebugOverlayGeometryShader_ = {};
      meshDebugOverlayFragmentShader_ = {};
      NURI_LOG_WARNING(
          "OpaqueRenderer::createShaders: Debug overlay shader path "
          "'%s' failed, fallback to line pipelines: %s",
          shaderPath.c_str(), compileResult.error().c_str());
      break;
    }
    *spec.outHandle = compileResult.value();
  }

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> OpaqueRenderer::createPipelines() {
  meshPipeline_ = Pipeline::create(gpu_);
  computePipeline_ = Pipeline::create(gpu_);
  if (!meshPipeline_ || !computePipeline_) {
    return Result<bool, std::string>::makeError(
        "OpaqueRenderer::createPipelines: failed to create pipeline wrappers");
  }

  resetMeshPipelineState();
  const Format depthFormat = kFrameCompositionDepthFormat;
  const RenderPipelineDesc meshDesc = meshPipelineDesc(
      kFrameCompositionSceneColorFormat, depthFormat, meshVertexShader_, {}, {},
      {}, meshFragmentShader_, PolygonMode::Fill);
  auto meshResult =
      meshPipeline_->createRenderPipeline(meshDesc, "opaque_mesh");
  if (meshResult.hasError()) {
    return Result<bool, std::string>::makeError(meshResult.error());
  }
  meshFillPipelineHandle_ = meshPipeline_->getRenderPipeline();

  const auto makeMsaaSceneDesc = [](RenderPipelineDesc desc) {
    desc.numSamples = kMsaa4xSampleCount;
    return desc;
  };
  const auto makeAlphaMsaaSceneDesc = [](RenderPipelineDesc desc) {
    desc.numSamples = kMsaa4xSampleCount;
    desc.alphaToCoverageEnabled = true;
    desc.minSampleShading = 1.0f;
    return desc;
  };

  RenderPipelineDesc meshMsaaDesc = makeMsaaSceneDesc(meshDesc);
  auto meshMsaaResult =
      gpu_.createRenderPipeline(meshMsaaDesc, "opaque_mesh_msaa4x");
  if (meshMsaaResult.hasError()) {
    return Result<bool, std::string>::makeError(meshMsaaResult.error());
  }
  meshMsaaFillPipelineHandle_ = meshMsaaResult.value();

  RenderPipelineDesc meshAlphaMsaaDesc = makeAlphaMsaaSceneDesc(meshDesc);
  auto meshAlphaMsaaResult =
      gpu_.createRenderPipeline(meshAlphaMsaaDesc, "opaque_mesh_alpha_msaa4x");
  if (meshAlphaMsaaResult.hasError()) {
    return Result<bool, std::string>::makeError(meshAlphaMsaaResult.error());
  }
  meshMsaaAlphaFillPipelineHandle_ = meshAlphaMsaaResult.value();

  {
    const RenderPipelineDesc doubleSidedMeshDesc = meshPipelineDesc(
        kFrameCompositionSceneColorFormat, depthFormat, meshVertexShader_, {},
        {}, {}, meshFragmentShader_, PolygonMode::Fill, Topology::Triangle, 0,
        false, CullMode::None);
    auto doubleSidedMeshResult = gpu_.createRenderPipeline(
        doubleSidedMeshDesc, "opaque_mesh_double_sided");
    if (doubleSidedMeshResult.hasError()) {
      return Result<bool, std::string>::makeError(
          doubleSidedMeshResult.error());
    }
    meshDoubleSidedFillPipelineHandle_ = doubleSidedMeshResult.value();

    RenderPipelineDesc doubleSidedMsaaDesc =
        makeMsaaSceneDesc(doubleSidedMeshDesc);
    auto doubleSidedMsaaResult = gpu_.createRenderPipeline(
        doubleSidedMsaaDesc, "opaque_mesh_double_sided_msaa4x");
    if (doubleSidedMsaaResult.hasError()) {
      return Result<bool, std::string>::makeError(
          doubleSidedMsaaResult.error());
    }
    meshMsaaDoubleSidedFillPipelineHandle_ = doubleSidedMsaaResult.value();

    RenderPipelineDesc doubleSidedAlphaMsaaDesc =
        makeAlphaMsaaSceneDesc(doubleSidedMeshDesc);
    auto doubleSidedAlphaMsaaResult = gpu_.createRenderPipeline(
        doubleSidedAlphaMsaaDesc, "opaque_mesh_alpha_double_sided_msaa4x");
    if (doubleSidedAlphaMsaaResult.hasError()) {
      return Result<bool, std::string>::makeError(
          doubleSidedAlphaMsaaResult.error());
    }
    meshMsaaAlphaDoubleSidedFillPipelineHandle_ =
        doubleSidedAlphaMsaaResult.value();
  }

  if (nuri::isValid(meshVelocityVertexShader_) &&
      nuri::isValid(meshVelocityFragmentShader_)) {
    const RenderPipelineDesc velocityDesc =
        meshPipelineDesc(kFrameCompositionMotionVectorFormat, depthFormat,
                         meshVelocityVertexShader_, {}, {}, {},
                         meshVelocityFragmentShader_, PolygonMode::Fill);
    auto velocityResult =
        gpu_.createRenderPipeline(velocityDesc, "opaque_velocity");
    if (velocityResult.hasError()) {
      NURI_LOG_WARNING(
          "OpaqueRenderer::createPipelines: velocity pipeline failed, opaque "
          "velocity generation disabled: %s",
          velocityResult.error().c_str());
      meshVelocityPipelineHandle_ = {};
      meshVelocityDoubleSidedPipelineHandle_ = {};
    } else {
      meshVelocityPipelineHandle_ = velocityResult.value();
      const RenderPipelineDesc doubleSidedVelocityDesc = meshPipelineDesc(
          kFrameCompositionMotionVectorFormat, depthFormat,
          meshVelocityVertexShader_, {}, {}, {}, meshVelocityFragmentShader_,
          PolygonMode::Fill, Topology::Triangle, 0, false, CullMode::None);
      auto doubleSidedVelocityResult = gpu_.createRenderPipeline(
          doubleSidedVelocityDesc, "opaque_velocity_double_sided");
      if (doubleSidedVelocityResult.hasError()) {
        NURI_LOG_WARNING(
            "OpaqueRenderer::createPipelines: double-sided velocity pipeline "
            "failed, double-sided velocity draws will use back-face culling: "
            "%s",
            doubleSidedVelocityResult.error().c_str());
        meshVelocityDoubleSidedPipelineHandle_ = {};
      } else {
        meshVelocityDoubleSidedPipelineHandle_ =
            doubleSidedVelocityResult.value();
      }
    }
  }

  if (nuri::isValid(meshReactiveMaskVertexShader_) &&
      nuri::isValid(meshReactiveMaskFragmentShader_)) {
    const RenderPipelineDesc reactiveMaskDesc =
        meshPipelineDesc(kFrameCompositionReactiveMaskFormat, depthFormat,
                         meshReactiveMaskVertexShader_, {}, {}, {},
                         meshReactiveMaskFragmentShader_, PolygonMode::Fill);
    auto reactiveMaskResult =
        gpu_.createRenderPipeline(reactiveMaskDesc, "opaque_reactive_mask");
    if (reactiveMaskResult.hasError()) {
      NURI_LOG_WARNING(
          "OpaqueRenderer::createPipelines: reactive mask pipeline failed, "
          "alpha-mask reactive tracking disabled: %s",
          reactiveMaskResult.error().c_str());
      meshReactiveMaskPipelineHandle_ = {};
      meshReactiveMaskDoubleSidedPipelineHandle_ = {};
    } else {
      meshReactiveMaskPipelineHandle_ = reactiveMaskResult.value();
      const RenderPipelineDesc doubleSidedReactiveMaskDesc =
          meshPipelineDesc(kFrameCompositionReactiveMaskFormat, depthFormat,
                           meshReactiveMaskVertexShader_, {}, {}, {},
                           meshReactiveMaskFragmentShader_, PolygonMode::Fill,
                           Topology::Triangle, 0, false, CullMode::None);
      auto doubleSidedReactiveMaskResult = gpu_.createRenderPipeline(
          doubleSidedReactiveMaskDesc, "opaque_reactive_mask_double_sided");
      if (doubleSidedReactiveMaskResult.hasError()) {
        NURI_LOG_WARNING(
            "OpaqueRenderer::createPipelines: double-sided reactive mask "
            "pipeline failed, double-sided masked draws will use back-face "
            "culling: %s",
            doubleSidedReactiveMaskResult.error().c_str());
        meshReactiveMaskDoubleSidedPipelineHandle_ = {};
      } else {
        meshReactiveMaskDoubleSidedPipelineHandle_ =
            doubleSidedReactiveMaskResult.value();
      }
    }
  }

  if (nuri::isValid(meshVertexShader_) &&
      nuri::isValid(meshNormalFragmentShader_)) {
    const RenderPipelineDesc normalDesc = meshPipelineDesc(
        kFrameCompositionNormalFormat, depthFormat, meshVertexShader_, {}, {},
        {}, meshNormalFragmentShader_, PolygonMode::Fill);
    auto normalResult =
        gpu_.createRenderPipeline(normalDesc, "opaque_material_normals");
    if (normalResult.hasError()) {
      NURI_LOG_WARNING(
          "OpaqueRenderer::createPipelines: material normal pipeline failed, "
          "GTAO normal pre-pass disabled: %s",
          normalResult.error().c_str());
      meshNormalPipelineHandle_ = {};
      meshNormalDoubleSidedPipelineHandle_ = {};
    } else {
      meshNormalPipelineHandle_ = normalResult.value();
      const RenderPipelineDesc doubleSidedNormalDesc = meshPipelineDesc(
          kFrameCompositionNormalFormat, depthFormat, meshVertexShader_, {}, {},
          {}, meshNormalFragmentShader_, PolygonMode::Fill, Topology::Triangle,
          0, false, CullMode::None);
      auto doubleSidedNormalResult = gpu_.createRenderPipeline(
          doubleSidedNormalDesc, "opaque_material_normals_double_sided");
      if (doubleSidedNormalResult.hasError()) {
        NURI_LOG_WARNING(
            "OpaqueRenderer::createPipelines: double-sided material normal "
            "pipeline failed, double-sided normals will use back-face culling: "
            "%s",
            doubleSidedNormalResult.error().c_str());
        meshNormalDoubleSidedPipelineHandle_ = {};
      } else {
        meshNormalDoubleSidedPipelineHandle_ = doubleSidedNormalResult.value();
      }
    }
  }

  const auto createDepthPipeline =
      [this](const RenderPipelineDesc &desc, std::string_view debugName,
             RenderPipelineHandle &outHandle) -> bool {
    auto result = gpu_.createRenderPipeline(desc, debugName);
    if (result.hasError()) {
      if (!loggedDepthPrepassUnsupported_) {
        loggedDepthPrepassUnsupported_ = true;
        NURI_LOG_WARNING(
            "OpaqueRenderer::createPipelines: depth pre-pass pipeline '%.*s' "
            "failed, depth pre-pass disabled: %s",
            static_cast<int>(debugName.size()), debugName.data(),
            result.error().c_str());
      }
      outHandle = {};
      return false;
    }
    outHandle = result.value();
    return true;
  };
  const auto createMsaaDepthPipeline =
      [&createDepthPipeline](RenderPipelineDesc desc,
                             std::string_view debugName,
                             RenderPipelineHandle &outHandle) -> bool {
    desc.numSamples = kMsaa4xSampleCount;
    return createDepthPipeline(desc, debugName, outHandle);
  };
  if (nuri::isValid(depthVertexShader_) &&
      nuri::isValid(depthFragmentShader_)) {
    const RenderPipelineDesc depthDesc =
        depthPipelineDesc(depthFormat, depthVertexShader_, {}, {},
                          depthFragmentShader_, CullMode::Back);
    createDepthPipeline(depthDesc, "opaque_mesh_depth",
                        meshDepthPipelineHandle_);
    createMsaaDepthPipeline(depthDesc, "opaque_mesh_depth_msaa4x",
                            meshMsaaDepthPipelineHandle_);
    const RenderPipelineDesc doubleSidedDepthDesc =
        depthPipelineDesc(depthFormat, depthVertexShader_, {}, {},
                          depthFragmentShader_, CullMode::None);
    createDepthPipeline(doubleSidedDepthDesc, "opaque_mesh_depth_double_sided",
                        meshDepthDoubleSidedPipelineHandle_);
    createMsaaDepthPipeline(doubleSidedDepthDesc,
                            "opaque_mesh_depth_double_sided_msaa4x",
                            meshMsaaDepthDoubleSidedPipelineHandle_);
  }
  if (nuri::isValid(depthAlphaVertexShader_) &&
      nuri::isValid(depthAlphaFragmentShader_)) {
    const RenderPipelineDesc depthAlphaDesc =
        depthPipelineDesc(depthFormat, depthAlphaVertexShader_, {}, {},
                          depthAlphaFragmentShader_, CullMode::Back);
    createDepthPipeline(depthAlphaDesc, "opaque_mesh_depth_alpha",
                        meshDepthAlphaPipelineHandle_);
    createMsaaDepthPipeline(depthAlphaDesc, "opaque_mesh_depth_alpha_msaa4x",
                            meshMsaaDepthAlphaPipelineHandle_);
    const RenderPipelineDesc doubleSidedDepthAlphaDesc =
        depthPipelineDesc(depthFormat, depthAlphaVertexShader_, {}, {},
                          depthAlphaFragmentShader_, CullMode::None);
    createDepthPipeline(doubleSidedDepthAlphaDesc,
                        "opaque_mesh_depth_alpha_double_sided",
                        meshDepthAlphaDoubleSidedPipelineHandle_);
    createMsaaDepthPipeline(doubleSidedDepthAlphaDesc,
                            "opaque_mesh_depth_alpha_double_sided_msaa4x",
                            meshMsaaDepthAlphaDoubleSidedPipelineHandle_);
  }
  const bool canCreateTessPipeline =
      !tessellationUnsupported_ && nuri::isValid(meshTessVertexShader_) &&
      nuri::isValid(meshTessControlShader_) &&
      nuri::isValid(meshTessEvalShader_) && nuri::isValid(meshFragmentShader_);
  const auto canCreatePickTessPipeline = [this]() -> bool {
    return !tessellationUnsupported_ &&
           nuri::isValid(meshPickTessVertexShader_) &&
           nuri::isValid(meshPickTessControlShader_) &&
           nuri::isValid(meshPickTessEvalShader_) &&
           nuri::isValid(meshPickFragmentShader_);
  };
  const auto canCreateDepthTessPipeline = [this]() -> bool {
    return !tessellationUnsupported_ && nuri::isValid(depthTessVertexShader_) &&
           nuri::isValid(depthTessControlShader_) &&
           nuri::isValid(depthTessEvalShader_) &&
           nuri::isValid(depthFragmentShader_);
  };
  const auto canCreateDepthAlphaTessPipeline = [this]() -> bool {
    return !tessellationUnsupported_ &&
           nuri::isValid(depthAlphaTessVertexShader_) &&
           nuri::isValid(depthAlphaTessControlShader_) &&
           nuri::isValid(depthAlphaTessEvalShader_) &&
           nuri::isValid(depthAlphaFragmentShader_);
  };
  if (canCreateTessPipeline) {
    const RenderPipelineDesc tessDesc = meshPipelineDesc(
        kFrameCompositionSceneColorFormat, depthFormat, meshTessVertexShader_,
        meshTessControlShader_, meshTessEvalShader_, {}, meshFragmentShader_,
        PolygonMode::Fill, Topology::Patch, kTessellationPatchControlPoints);
    auto tessResult = gpu_.createRenderPipeline(tessDesc, "opaque_mesh_tess");
    if (tessResult.hasError()) {
      tessellationUnsupported_ = true;
      meshTessPipelineHandle_ = {};
      NURI_LOG_WARNING("OpaqueRenderer::createPipelines: Tessellation pipeline "
                       "failed, fallback to non-tessellation path: %s",
                       tessResult.error().c_str());
    } else {
      meshTessPipelineHandle_ = tessResult.value();
      RenderPipelineDesc tessMsaaDesc = makeMsaaSceneDesc(tessDesc);
      auto tessMsaaResult =
          gpu_.createRenderPipeline(tessMsaaDesc, "opaque_mesh_tess_msaa4x");
      if (tessMsaaResult.hasError()) {
        tessellationUnsupported_ = true;
        gpu_.destroyRenderPipeline(meshTessPipelineHandle_);
        meshTessPipelineHandle_ = {};
        meshMsaaTessPipelineHandle_ = {};
        NURI_LOG_WARNING("OpaqueRenderer::createPipelines: MSAA tessellation "
                         "pipeline failed, fallback to non-tessellation path: "
                         "%s",
                         tessMsaaResult.error().c_str());
      } else {
        meshMsaaTessPipelineHandle_ = tessMsaaResult.value();
        RenderPipelineDesc tessAlphaMsaaDesc = makeAlphaMsaaSceneDesc(tessDesc);
        auto tessAlphaMsaaResult = gpu_.createRenderPipeline(
            tessAlphaMsaaDesc, "opaque_mesh_tess_alpha_msaa4x");
        if (tessAlphaMsaaResult.hasError()) {
          tessellationUnsupported_ = true;
          gpu_.destroyRenderPipeline(meshTessPipelineHandle_);
          meshTessPipelineHandle_ = {};
          gpu_.destroyRenderPipeline(meshMsaaTessPipelineHandle_);
          meshMsaaTessPipelineHandle_ = {};
          meshMsaaAlphaTessPipelineHandle_ = {};
          NURI_LOG_WARNING(
              "OpaqueRenderer::createPipelines: alpha MSAA tessellation "
              "pipeline failed, fallback to non-tessellation path: %s",
              tessAlphaMsaaResult.error().c_str());
        } else {
          meshMsaaAlphaTessPipelineHandle_ = tessAlphaMsaaResult.value();
        }
      }
    }
    if (!tessellationUnsupported_) {
      const RenderPipelineDesc doubleSidedTessDesc = meshPipelineDesc(
          kFrameCompositionSceneColorFormat, depthFormat, meshTessVertexShader_,
          meshTessControlShader_, meshTessEvalShader_, {}, meshFragmentShader_,
          PolygonMode::Fill, Topology::Patch, kTessellationPatchControlPoints,
          false, CullMode::None);
      auto doubleSidedTessResult = gpu_.createRenderPipeline(
          doubleSidedTessDesc, "opaque_mesh_tess_double_sided");
      if (doubleSidedTessResult.hasError()) {
        tessellationUnsupported_ = true;
        if (nuri::isValid(meshTessPipelineHandle_)) {
          gpu_.destroyRenderPipeline(meshTessPipelineHandle_);
          meshTessPipelineHandle_ = {};
        }
        if (nuri::isValid(meshMsaaTessPipelineHandle_)) {
          gpu_.destroyRenderPipeline(meshMsaaTessPipelineHandle_);
          meshMsaaTessPipelineHandle_ = {};
        }
        if (nuri::isValid(meshMsaaAlphaTessPipelineHandle_)) {
          gpu_.destroyRenderPipeline(meshMsaaAlphaTessPipelineHandle_);
          meshMsaaAlphaTessPipelineHandle_ = {};
        }
        meshDoubleSidedTessPipelineHandle_ = {};
        NURI_LOG_WARNING("OpaqueRenderer::createPipelines: double-sided "
                         "tessellation pipeline failed, fallback to "
                         "non-tessellation path: %s",
                         doubleSidedTessResult.error().c_str());
      } else {
        meshDoubleSidedTessPipelineHandle_ = doubleSidedTessResult.value();
        RenderPipelineDesc doubleSidedTessMsaaDesc =
            makeMsaaSceneDesc(doubleSidedTessDesc);
        auto doubleSidedTessMsaaResult = gpu_.createRenderPipeline(
            doubleSidedTessMsaaDesc, "opaque_mesh_tess_double_sided_msaa4x");
        if (doubleSidedTessMsaaResult.hasError()) {
          tessellationUnsupported_ = true;
          gpu_.destroyRenderPipeline(meshDoubleSidedTessPipelineHandle_);
          meshDoubleSidedTessPipelineHandle_ = {};
          meshMsaaDoubleSidedTessPipelineHandle_ = {};
          if (nuri::isValid(meshTessPipelineHandle_)) {
            gpu_.destroyRenderPipeline(meshTessPipelineHandle_);
            meshTessPipelineHandle_ = {};
          }
          if (nuri::isValid(meshMsaaTessPipelineHandle_)) {
            gpu_.destroyRenderPipeline(meshMsaaTessPipelineHandle_);
            meshMsaaTessPipelineHandle_ = {};
          }
          if (nuri::isValid(meshMsaaAlphaTessPipelineHandle_)) {
            gpu_.destroyRenderPipeline(meshMsaaAlphaTessPipelineHandle_);
            meshMsaaAlphaTessPipelineHandle_ = {};
          }
          NURI_LOG_WARNING(
              "OpaqueRenderer::createPipelines: double-sided MSAA "
              "tessellation pipeline failed, fallback to non-tessellation "
              "path: %s",
              doubleSidedTessMsaaResult.error().c_str());
        } else {
          meshMsaaDoubleSidedTessPipelineHandle_ =
              doubleSidedTessMsaaResult.value();
          RenderPipelineDesc doubleSidedTessAlphaMsaaDesc =
              makeAlphaMsaaSceneDesc(doubleSidedTessDesc);
          auto doubleSidedTessAlphaMsaaResult = gpu_.createRenderPipeline(
              doubleSidedTessAlphaMsaaDesc,
              "opaque_mesh_tess_alpha_double_sided_msaa4x");
          if (doubleSidedTessAlphaMsaaResult.hasError()) {
            tessellationUnsupported_ = true;
            gpu_.destroyRenderPipeline(meshDoubleSidedTessPipelineHandle_);
            meshDoubleSidedTessPipelineHandle_ = {};
            gpu_.destroyRenderPipeline(meshMsaaDoubleSidedTessPipelineHandle_);
            meshMsaaDoubleSidedTessPipelineHandle_ = {};
            meshMsaaAlphaDoubleSidedTessPipelineHandle_ = {};
            if (nuri::isValid(meshTessPipelineHandle_)) {
              gpu_.destroyRenderPipeline(meshTessPipelineHandle_);
              meshTessPipelineHandle_ = {};
            }
            if (nuri::isValid(meshMsaaTessPipelineHandle_)) {
              gpu_.destroyRenderPipeline(meshMsaaTessPipelineHandle_);
              meshMsaaTessPipelineHandle_ = {};
            }
            if (nuri::isValid(meshMsaaAlphaTessPipelineHandle_)) {
              gpu_.destroyRenderPipeline(meshMsaaAlphaTessPipelineHandle_);
              meshMsaaAlphaTessPipelineHandle_ = {};
            }
            NURI_LOG_WARNING(
                "OpaqueRenderer::createPipelines: double-sided alpha MSAA "
                "tessellation pipeline failed, fallback to non-tessellation "
                "path: %s",
                doubleSidedTessAlphaMsaaResult.error().c_str());
          } else {
            meshMsaaAlphaDoubleSidedTessPipelineHandle_ =
                doubleSidedTessAlphaMsaaResult.value();
          }
        }
      }
    }
    if (!tessellationUnsupported_) {
      if (canCreateDepthTessPipeline()) {
        const RenderPipelineDesc depthTessDesc = depthPipelineDesc(
            depthFormat, depthTessVertexShader_, depthTessControlShader_,
            depthTessEvalShader_, depthFragmentShader_, CullMode::Back,
            Topology::Patch, kTessellationPatchControlPoints);
        createDepthPipeline(depthTessDesc, "opaque_mesh_depth_tess",
                            meshDepthTessPipelineHandle_);
        createMsaaDepthPipeline(depthTessDesc, "opaque_mesh_depth_tess_msaa4x",
                                meshMsaaDepthTessPipelineHandle_);
        const RenderPipelineDesc depthDoubleSidedTessDesc = depthPipelineDesc(
            depthFormat, depthTessVertexShader_, depthTessControlShader_,
            depthTessEvalShader_, depthFragmentShader_, CullMode::None,
            Topology::Patch, kTessellationPatchControlPoints);
        createDepthPipeline(depthDoubleSidedTessDesc,
                            "opaque_mesh_depth_tess_double_sided",
                            meshDepthDoubleSidedTessPipelineHandle_);
        createMsaaDepthPipeline(depthDoubleSidedTessDesc,
                                "opaque_mesh_depth_tess_double_sided_msaa4x",
                                meshMsaaDepthDoubleSidedTessPipelineHandle_);
      }
      if (canCreateDepthAlphaTessPipeline()) {
        const RenderPipelineDesc depthAlphaTessDesc = depthPipelineDesc(
            depthFormat, depthAlphaTessVertexShader_,
            depthAlphaTessControlShader_, depthAlphaTessEvalShader_,
            depthAlphaFragmentShader_, CullMode::Back, Topology::Patch,
            kTessellationPatchControlPoints);
        createDepthPipeline(depthAlphaTessDesc, "opaque_mesh_depth_alpha_tess",
                            meshDepthAlphaTessPipelineHandle_);
        createMsaaDepthPipeline(depthAlphaTessDesc,
                                "opaque_mesh_depth_alpha_tess_msaa4x",
                                meshMsaaDepthAlphaTessPipelineHandle_);
        const RenderPipelineDesc depthAlphaDoubleSidedTessDesc =
            depthPipelineDesc(depthFormat, depthAlphaTessVertexShader_,
                              depthAlphaTessControlShader_,
                              depthAlphaTessEvalShader_,
                              depthAlphaFragmentShader_, CullMode::None,
                              Topology::Patch, kTessellationPatchControlPoints);
        createDepthPipeline(depthAlphaDoubleSidedTessDesc,
                            "opaque_mesh_depth_alpha_tess_double_sided",
                            meshDepthAlphaDoubleSidedTessPipelineHandle_);
        createMsaaDepthPipeline(
            depthAlphaDoubleSidedTessDesc,
            "opaque_mesh_depth_alpha_tess_double_sided_msaa4x",
            meshMsaaDepthAlphaDoubleSidedTessPipelineHandle_);
      }
      if (nuri::isValid(meshNormalFragmentShader_)) {
        const RenderPipelineDesc normalTessDesc = meshPipelineDesc(
            kFrameCompositionNormalFormat, depthFormat, meshTessVertexShader_,
            meshTessControlShader_, meshTessEvalShader_, {},
            meshNormalFragmentShader_, PolygonMode::Fill, Topology::Patch,
            kTessellationPatchControlPoints);
        auto normalTessResult = gpu_.createRenderPipeline(
            normalTessDesc, "opaque_material_normals_tess");
        if (normalTessResult.hasError()) {
          meshNormalTessPipelineHandle_ = {};
          NURI_LOG_WARNING(
              "OpaqueRenderer::createPipelines: tessellation material normal "
              "pipeline failed, tessellated GTAO normals disabled: %s",
              normalTessResult.error().c_str());
        } else {
          meshNormalTessPipelineHandle_ = normalTessResult.value();
        }
        const RenderPipelineDesc doubleSidedNormalTessDesc = meshPipelineDesc(
            kFrameCompositionNormalFormat, depthFormat, meshTessVertexShader_,
            meshTessControlShader_, meshTessEvalShader_, {},
            meshNormalFragmentShader_, PolygonMode::Fill, Topology::Patch,
            kTessellationPatchControlPoints, false, CullMode::None);
        auto doubleSidedNormalTessResult = gpu_.createRenderPipeline(
            doubleSidedNormalTessDesc,
            "opaque_material_normals_tess_double_sided");
        if (doubleSidedNormalTessResult.hasError()) {
          meshNormalDoubleSidedTessPipelineHandle_ = {};
          NURI_LOG_WARNING(
              "OpaqueRenderer::createPipelines: double-sided tessellation "
              "material normal pipeline failed, double-sided tessellated GTAO "
              "normals disabled: %s",
              doubleSidedNormalTessResult.error().c_str());
        } else {
          meshNormalDoubleSidedTessPipelineHandle_ =
              doubleSidedNormalTessResult.value();
        }
      }
      if (nuri::isValid(meshVelocityTessVertexShader_) &&
          nuri::isValid(meshVelocityTessControlShader_) &&
          nuri::isValid(meshVelocityTessEvalShader_) &&
          nuri::isValid(meshVelocityFragmentShader_)) {
        const RenderPipelineDesc velocityTessDesc = meshPipelineDesc(
            kFrameCompositionMotionVectorFormat, depthFormat,
            meshVelocityTessVertexShader_, meshVelocityTessControlShader_,
            meshVelocityTessEvalShader_, {}, meshVelocityFragmentShader_,
            PolygonMode::Fill, Topology::Patch,
            kTessellationPatchControlPoints);
        auto velocityTessResult =
            gpu_.createRenderPipeline(velocityTessDesc, "opaque_velocity_tess");
        if (velocityTessResult.hasError()) {
          meshVelocityTessPipelineHandle_ = {};
          NURI_LOG_WARNING(
              "OpaqueRenderer::createPipelines: tessellated velocity pipeline "
              "failed, tessellated opaque velocity draws will be marked "
              "missing: %s",
              velocityTessResult.error().c_str());
        } else {
          meshVelocityTessPipelineHandle_ = velocityTessResult.value();
        }

        const RenderPipelineDesc doubleSidedVelocityTessDesc = meshPipelineDesc(
            kFrameCompositionMotionVectorFormat, depthFormat,
            meshVelocityTessVertexShader_, meshVelocityTessControlShader_,
            meshVelocityTessEvalShader_, {}, meshVelocityFragmentShader_,
            PolygonMode::Fill, Topology::Patch, kTessellationPatchControlPoints,
            false, CullMode::None);
        auto doubleSidedVelocityTessResult = gpu_.createRenderPipeline(
            doubleSidedVelocityTessDesc, "opaque_velocity_tess_double_sided");
        if (doubleSidedVelocityTessResult.hasError()) {
          meshVelocityDoubleSidedTessPipelineHandle_ = {};
          NURI_LOG_WARNING(
              "OpaqueRenderer::createPipelines: double-sided tessellated "
              "velocity pipeline failed, double-sided tessellated velocity "
              "draws will use the culling variant when available: %s",
              doubleSidedVelocityTessResult.error().c_str());
        } else {
          meshVelocityDoubleSidedTessPipelineHandle_ =
              doubleSidedVelocityTessResult.value();
        }
      }
    }
  } else {
    tessellationUnsupported_ = true;
  }

  {
    const RenderPipelineDesc pickDesc = meshPipelineDesc(
        Format::R32_UINT, depthFormat, meshPickVertexShader_, {}, {}, {},
        meshPickFragmentShader_, PolygonMode::Fill);
    auto pickPipelineResult =
        gpu_.createRenderPipeline(pickDesc, "opaque_mesh_pick");
    if (pickPipelineResult.hasError()) {
      if (nuri::isValid(meshTessPipelineHandle_)) {
        gpu_.destroyRenderPipeline(meshTessPipelineHandle_);
        meshTessPipelineHandle_ = {};
      }
      if (nuri::isValid(meshDoubleSidedTessPipelineHandle_)) {
        gpu_.destroyRenderPipeline(meshDoubleSidedTessPipelineHandle_);
        meshDoubleSidedTessPipelineHandle_ = {};
      }
      if (nuri::isValid(meshDoubleSidedFillPipelineHandle_)) {
        gpu_.destroyRenderPipeline(meshDoubleSidedFillPipelineHandle_);
        meshDoubleSidedFillPipelineHandle_ = {};
      }
      return Result<bool, std::string>::makeError(pickPipelineResult.error());
    }
    meshPickPipelineHandle_ = pickPipelineResult.value();
  }

  {
    const RenderPipelineDesc doubleSidedPickDesc =
        meshPipelineDesc(Format::R32_UINT, depthFormat, meshPickVertexShader_,
                         {}, {}, {}, meshPickFragmentShader_, PolygonMode::Fill,
                         Topology::Triangle, 0, false, CullMode::None);
    auto doubleSidedPickResult = gpu_.createRenderPipeline(
        doubleSidedPickDesc, "opaque_mesh_pick_double_sided");
    if (doubleSidedPickResult.hasError()) {
      if (nuri::isValid(meshPickPipelineHandle_)) {
        gpu_.destroyRenderPipeline(meshPickPipelineHandle_);
        meshPickPipelineHandle_ = {};
      }
      if (nuri::isValid(meshTessPipelineHandle_)) {
        gpu_.destroyRenderPipeline(meshTessPipelineHandle_);
        meshTessPipelineHandle_ = {};
      }
      if (nuri::isValid(meshDoubleSidedTessPipelineHandle_)) {
        gpu_.destroyRenderPipeline(meshDoubleSidedTessPipelineHandle_);
        meshDoubleSidedTessPipelineHandle_ = {};
      }
      if (nuri::isValid(meshDoubleSidedFillPipelineHandle_)) {
        gpu_.destroyRenderPipeline(meshDoubleSidedFillPipelineHandle_);
        meshDoubleSidedFillPipelineHandle_ = {};
      }
      return Result<bool, std::string>::makeError(
          doubleSidedPickResult.error());
    }
    meshPickDoubleSidedPipelineHandle_ = doubleSidedPickResult.value();
  }

  if (canCreatePickTessPipeline()) {
    const RenderPipelineDesc pickTessDesc = meshPipelineDesc(
        Format::R32_UINT, depthFormat, meshPickTessVertexShader_,
        meshPickTessControlShader_, meshPickTessEvalShader_, {},
        meshPickFragmentShader_, PolygonMode::Fill, Topology::Patch,
        kTessellationPatchControlPoints);
    auto pickTessResult =
        gpu_.createRenderPipeline(pickTessDesc, "opaque_mesh_tess_pick");
    if (pickTessResult.hasError()) {
      meshPickTessPipelineHandle_ = {};
      NURI_LOG_WARNING(
          "OpaqueRenderer::createPipelines: tessellation pick pipeline failed, "
          "falling back to non-tessellation pick pipeline: %s",
          pickTessResult.error().c_str());
    } else {
      meshPickTessPipelineHandle_ = pickTessResult.value();
    }
    if (!tessellationUnsupported_) {
      const RenderPipelineDesc doubleSidedPickTessDesc = meshPipelineDesc(
          Format::R32_UINT, depthFormat, meshPickTessVertexShader_,
          meshPickTessControlShader_, meshPickTessEvalShader_, {},
          meshPickFragmentShader_, PolygonMode::Fill, Topology::Patch,
          kTessellationPatchControlPoints, false, CullMode::None);
      auto doubleSidedPickTessResult = gpu_.createRenderPipeline(
          doubleSidedPickTessDesc, "opaque_mesh_tess_pick_double_sided");
      if (doubleSidedPickTessResult.hasError()) {
        meshPickDoubleSidedTessPipelineHandle_ = {};
        NURI_LOG_WARNING("OpaqueRenderer::createPipelines: double-sided "
                         "tessellation pick pipeline failed, falling back to "
                         "non-tessellation pick pipeline: %s",
                         doubleSidedPickTessResult.error().c_str());
      } else {
        meshPickDoubleSidedTessPipelineHandle_ =
            doubleSidedPickTessResult.value();
      }
    }
  }

  {
    const RenderPipelineDesc inspectDesc = meshPipelineDesc(
        Format::RGBA32_FLOAT, depthFormat, meshVertexShader_, {}, {}, {},
        meshShadowInspectFragmentShader_, PolygonMode::Fill);
    auto inspectPipelineResult =
        gpu_.createRenderPipeline(inspectDesc, "opaque_mesh_shadow_inspect");
    if (inspectPipelineResult.hasError()) {
      meshShadowInspectPipelineHandle_ = {};
      NURI_LOG_WARNING(
          "OpaqueRenderer::createPipelines: shadow inspect pipeline disabled: "
          "%s",
          inspectPipelineResult.error().c_str());
    } else {
      meshShadowInspectPipelineHandle_ = inspectPipelineResult.value();
    }
  }

  {
    const RenderPipelineDesc doubleSidedInspectDesc = meshPipelineDesc(
        Format::RGBA32_FLOAT, depthFormat, meshVertexShader_, {}, {}, {},
        meshShadowInspectFragmentShader_, PolygonMode::Fill, Topology::Triangle,
        0, false, CullMode::None);
    auto doubleSidedInspectResult = gpu_.createRenderPipeline(
        doubleSidedInspectDesc, "opaque_mesh_shadow_inspect_double_sided");
    if (doubleSidedInspectResult.hasError()) {
      meshShadowInspectDoubleSidedPipelineHandle_ = {};
      NURI_LOG_WARNING(
          "OpaqueRenderer::createPipelines: double-sided shadow inspect "
          "pipeline disabled: %s",
          doubleSidedInspectResult.error().c_str());
    } else {
      meshShadowInspectDoubleSidedPipelineHandle_ =
          doubleSidedInspectResult.value();
    }
  }

  if (canCreateTessPipeline) {
    const RenderPipelineDesc inspectTessDesc = meshPipelineDesc(
        Format::RGBA32_FLOAT, depthFormat, meshTessVertexShader_,
        meshTessControlShader_, meshTessEvalShader_, {},
        meshShadowInspectFragmentShader_, PolygonMode::Fill, Topology::Patch,
        kTessellationPatchControlPoints);
    auto inspectTessResult = gpu_.createRenderPipeline(
        inspectTessDesc, "opaque_mesh_tess_shadow_inspect");
    if (inspectTessResult.hasError()) {
      meshShadowInspectTessPipelineHandle_ = {};
      NURI_LOG_WARNING(
          "OpaqueRenderer::createPipelines: tessellation shadow inspect "
          "pipeline failed, falling back to non-tessellation inspect "
          "pipeline: %s",
          inspectTessResult.error().c_str());
    } else {
      meshShadowInspectTessPipelineHandle_ = inspectTessResult.value();
    }
    if (!tessellationUnsupported_) {
      const RenderPipelineDesc doubleSidedInspectTessDesc = meshPipelineDesc(
          Format::RGBA32_FLOAT, depthFormat, meshTessVertexShader_,
          meshTessControlShader_, meshTessEvalShader_, {},
          meshShadowInspectFragmentShader_, PolygonMode::Fill, Topology::Patch,
          kTessellationPatchControlPoints, false, CullMode::None);
      auto doubleSidedInspectTessResult = gpu_.createRenderPipeline(
          doubleSidedInspectTessDesc,
          "opaque_mesh_tess_shadow_inspect_double_sided");
      if (doubleSidedInspectTessResult.hasError()) {
        meshShadowInspectDoubleSidedTessPipelineHandle_ = {};
        NURI_LOG_WARNING(
            "OpaqueRenderer::createPipelines: double-sided tessellation "
            "shadow inspect pipeline failed, falling back to "
            "non-tessellation inspect pipeline: %s",
            doubleSidedInspectTessResult.error().c_str());
      } else {
        meshShadowInspectDoubleSidedTessPipelineHandle_ =
            doubleSidedInspectTessResult.value();
      }
    }
  }

  const ComputePipelineDesc computeDesc{
      .computeShader = computeShaderHandle_,
  };
  auto computeResult = computePipeline_->createComputePipeline(
      computeDesc, "opaque_instance_compute");
  if (computeResult.hasError()) {
    destroyMeshPipelineState();
    return Result<bool, std::string>::makeError(computeResult.error());
  }
  computePipelineHandle_ = computePipeline_->getComputePipeline();

  visibilityComputePipeline_.reset();
  visibilityIndirectDrawComputePipeline_.reset();
  visibilityIndirectMeshDispatchComputePipeline_.reset();
  visibilityPipelineHandle_ = {};
  visibilityIndirectDrawPipelineHandle_ = {};
  visibilityIndirectMeshDispatchPipelineHandle_ = {};
  if (nuri::isValid(visibilityComputeShader_)) {
    visibilityComputePipeline_ = Pipeline::create(gpu_);
    if (visibilityComputePipeline_) {
      auto visibilityPipelineResult =
          visibilityComputePipeline_->createComputePipeline(
              ComputePipelineDesc{.computeShader = visibilityComputeShader_},
              "opaque_visibility_cull");
      if (visibilityPipelineResult.hasError()) {
        if (!loggedVisibilityGpuUnsupportedWarning_) {
          loggedVisibilityGpuUnsupportedWarning_ = true;
          NURI_LOG_WARNING(
              "OpaqueRenderer::createPipelines: visibility culling pipeline "
              "failed, GPU visibility pass disabled: %s",
              visibilityPipelineResult.error().c_str());
        }
        visibilityComputePipeline_.reset();
      } else {
        visibilityPipelineHandle_ =
            visibilityComputePipeline_->getComputePipeline();
      }
    }
  }
  if (nuri::isValid(visibilityIndirectDrawComputeShader_)) {
    visibilityIndirectDrawComputePipeline_ = Pipeline::create(gpu_);
    if (visibilityIndirectDrawComputePipeline_) {
      auto indirectPipelineResult =
          visibilityIndirectDrawComputePipeline_->createComputePipeline(
              ComputePipelineDesc{.computeShader =
                                      visibilityIndirectDrawComputeShader_},
              "opaque_visibility_indirect_draw");
      if (indirectPipelineResult.hasError()) {
        if (!loggedVisibilityGpuUnsupportedWarning_) {
          loggedVisibilityGpuUnsupportedWarning_ = true;
          NURI_LOG_WARNING(
              "OpaqueRenderer::createPipelines: visibility indirect draw "
              "pipeline failed, GPU indirect draw disabled: %s",
              indirectPipelineResult.error().c_str());
        }
        visibilityIndirectDrawComputePipeline_.reset();
      } else {
        visibilityIndirectDrawPipelineHandle_ =
            visibilityIndirectDrawComputePipeline_->getComputePipeline();
      }
    }
  }
  if (nuri::isValid(visibilityIndirectMeshDispatchComputeShader_)) {
    visibilityIndirectMeshDispatchComputePipeline_ = Pipeline::create(gpu_);
    if (visibilityIndirectMeshDispatchComputePipeline_) {
      auto indirectMeshPipelineResult =
          visibilityIndirectMeshDispatchComputePipeline_->createComputePipeline(
              ComputePipelineDesc{
                  .computeShader =
                      visibilityIndirectMeshDispatchComputeShader_},
              "opaque_visibility_indirect_mesh_dispatch");
      if (indirectMeshPipelineResult.hasError()) {
        if (!loggedVisibilityGpuUnsupportedWarning_) {
          loggedVisibilityGpuUnsupportedWarning_ = true;
          NURI_LOG_WARNING(
              "OpaqueRenderer::createPipelines: visibility indirect mesh "
              "dispatch pipeline failed, GPU mesh dispatch args disabled: %s",
              indirectMeshPipelineResult.error().c_str());
        }
        visibilityIndirectMeshDispatchComputePipeline_.reset();
      } else {
        visibilityIndirectMeshDispatchPipelineHandle_ =
            visibilityIndirectMeshDispatchComputePipeline_
                ->getComputePipeline();
      }
    }
  }

  if (nuri::isValid(depthPyramidVertexShader_) &&
      nuri::isValid(depthPyramidFragmentShader_)) {
    RenderPipelineDesc pyramidDesc{
        .vertexInput = {},
        .vertexShader = depthPyramidVertexShader_,
        .fragmentShader = depthPyramidFragmentShader_,
        .colorFormats = {Format::RG32_FLOAT},
        .colorAttachmentCount = 1u,
        .depthFormat = Format::Count,
        .cullMode = CullMode::None,
        .polygonMode = PolygonMode::Fill,
        .topology = Topology::Triangle,
        .patchControlPoints = 0u,
        .blendEnabled = false,
    };
    auto pyramidResult =
        gpu_.createRenderPipeline(pyramidDesc, "opaque_depth_minmax_pyramid");
    if (pyramidResult.hasError()) {
      if (!loggedDepthPyramidUnsupported_) {
        loggedDepthPyramidUnsupported_ = true;
        NURI_LOG_WARNING(
            "OpaqueRenderer::createPipelines: depth pyramid pipeline failed, "
            "depth pyramid disabled: %s",
            pyramidResult.error().c_str());
      }
      depthPyramidPipelineHandle_ = {};
    } else {
      depthPyramidPipelineHandle_ = pyramidResult.value();
    }
  }

  baseMeshFillDraw_ = makeBaseMeshDraw(meshFillPipelineHandle_, "OpaqueMesh");
  resetOverlayPipelineState();

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> OpaqueRenderer::ensureMeshletPipelineState() {
  if (meshletPipelineInitialized_) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (meshletPipelineUnsupported_) {
    return Result<bool, std::string>::makeError(
        "OpaqueRenderer::ensureMeshletPipelineState: meshlet pipeline is "
        "unsupported");
  }
  if (!gpu_.supportsFeature(GPUFeature::Meshlets)) {
    meshletPipelineUnsupported_ = true;
    return Result<bool, std::string>::makeError(
        "OpaqueRenderer::ensureMeshletPipelineState: GPU meshlets are "
        "unsupported");
  }
  if (config_.meshletTask.empty() || config_.meshletMesh.empty() ||
      config_.meshletFragment.empty() || config_.meshletDepthFragment.empty() ||
      config_.meshletDepthAlphaFragment.empty()) {
    return Result<bool, std::string>::makeError(
        "OpaqueRenderer::ensureMeshletPipelineState: meshlet shader paths are "
        "not configured");
  }

  meshletShader_ = Shader::create("opaque_meshlet", gpu_);
  if (!meshletShader_) {
    return Result<bool, std::string>::makeError(
        "OpaqueRenderer::ensureMeshletPipelineState: failed to create shader "
        "object");
  }

  auto taskResult = meshletShader_->compileFromFile(
      config_.meshletTask.string(), ShaderStage::Task);
  if (taskResult.hasError()) {
    meshletShader_.reset();
    return Result<bool, std::string>::makeError(taskResult.error());
  }
  auto meshResult = meshletShader_->compileFromFile(
      config_.meshletMesh.string(), ShaderStage::Mesh);
  if (meshResult.hasError()) {
    meshletShader_.reset();
    meshletTaskShader_ = {};
    return Result<bool, std::string>::makeError(meshResult.error());
  }
  auto fragmentResult = meshletShader_->compileFromFile(
      config_.meshletFragment.string(), ShaderStage::Fragment);
  if (fragmentResult.hasError()) {
    meshletShader_.reset();
    meshletTaskShader_ = {};
    meshletMeshShader_ = {};
    return Result<bool, std::string>::makeError(fragmentResult.error());
  }
  auto depthFragmentResult = meshletShader_->compileFromFile(
      config_.meshletDepthFragment.string(), ShaderStage::Fragment);
  if (depthFragmentResult.hasError()) {
    meshletShader_.reset();
    meshletTaskShader_ = {};
    meshletMeshShader_ = {};
    meshletFragmentShader_ = {};
    return Result<bool, std::string>::makeError(depthFragmentResult.error());
  }
  auto depthAlphaFragmentResult = meshletShader_->compileFromFile(
      config_.meshletDepthAlphaFragment.string(), ShaderStage::Fragment);
  if (depthAlphaFragmentResult.hasError()) {
    meshletShader_.reset();
    meshletTaskShader_ = {};
    meshletMeshShader_ = {};
    meshletFragmentShader_ = {};
    meshletDepthFragmentShader_ = {};
    return Result<bool, std::string>::makeError(
        depthAlphaFragmentResult.error());
  }
  const auto meshletShaderDir = config_.meshletMesh.parent_path();
  auto normalFragmentResult = meshletShader_->compileFromFile(
      (meshletShaderDir / "opaque_meshlet_normal.frag").string(),
      ShaderStage::Fragment);
  if (normalFragmentResult.hasError()) {
    meshletShader_.reset();
    meshletTaskShader_ = {};
    meshletMeshShader_ = {};
    meshletFragmentShader_ = {};
    meshletDepthFragmentShader_ = {};
    meshletDepthAlphaFragmentShader_ = {};
    return Result<bool, std::string>::makeError(normalFragmentResult.error());
  }
  auto velocityMeshResult = meshletShader_->compileFromFile(
      (meshletShaderDir / "opaque_meshlet_velocity.mesh.glsl").string(),
      ShaderStage::Mesh);
  if (velocityMeshResult.hasError()) {
    meshletShader_.reset();
    meshletTaskShader_ = {};
    meshletMeshShader_ = {};
    meshletFragmentShader_ = {};
    meshletDepthFragmentShader_ = {};
    meshletDepthAlphaFragmentShader_ = {};
    meshletNormalFragmentShader_ = {};
    return Result<bool, std::string>::makeError(velocityMeshResult.error());
  }
  auto velocityFragmentResult = meshletShader_->compileFromFile(
      (meshletShaderDir / "opaque_meshlet_velocity.frag").string(),
      ShaderStage::Fragment);
  if (velocityFragmentResult.hasError()) {
    meshletShader_.reset();
    meshletTaskShader_ = {};
    meshletMeshShader_ = {};
    meshletFragmentShader_ = {};
    meshletDepthFragmentShader_ = {};
    meshletDepthAlphaFragmentShader_ = {};
    meshletNormalFragmentShader_ = {};
    meshletVelocityMeshShader_ = {};
    return Result<bool, std::string>::makeError(velocityFragmentResult.error());
  }
  auto reactiveMaskMeshResult = meshletShader_->compileFromFile(
      (meshletShaderDir / "opaque_meshlet_reactive_mask.mesh.glsl").string(),
      ShaderStage::Mesh);
  if (reactiveMaskMeshResult.hasError()) {
    meshletShader_.reset();
    meshletTaskShader_ = {};
    meshletMeshShader_ = {};
    meshletFragmentShader_ = {};
    meshletDepthFragmentShader_ = {};
    meshletDepthAlphaFragmentShader_ = {};
    meshletNormalFragmentShader_ = {};
    meshletVelocityMeshShader_ = {};
    meshletVelocityFragmentShader_ = {};
    return Result<bool, std::string>::makeError(reactiveMaskMeshResult.error());
  }
  auto reactiveMaskFragmentResult = meshletShader_->compileFromFile(
      (meshletShaderDir / "opaque_meshlet_reactive_mask.frag").string(),
      ShaderStage::Fragment);
  if (reactiveMaskFragmentResult.hasError()) {
    meshletShader_.reset();
    meshletTaskShader_ = {};
    meshletMeshShader_ = {};
    meshletFragmentShader_ = {};
    meshletDepthFragmentShader_ = {};
    meshletDepthAlphaFragmentShader_ = {};
    meshletNormalFragmentShader_ = {};
    meshletVelocityMeshShader_ = {};
    meshletVelocityFragmentShader_ = {};
    meshletReactiveMaskMeshShader_ = {};
    return Result<bool, std::string>::makeError(
        reactiveMaskFragmentResult.error());
  }
  meshletTaskShader_ = taskResult.value();
  meshletMeshShader_ = meshResult.value();
  meshletFragmentShader_ = fragmentResult.value();
  meshletDepthFragmentShader_ = depthFragmentResult.value();
  meshletDepthAlphaFragmentShader_ = depthAlphaFragmentResult.value();
  meshletNormalFragmentShader_ = normalFragmentResult.value();
  meshletVelocityMeshShader_ = velocityMeshResult.value();
  meshletVelocityFragmentShader_ = velocityFragmentResult.value();
  meshletReactiveMaskMeshShader_ = reactiveMaskMeshResult.value();
  meshletReactiveMaskFragmentShader_ = reactiveMaskFragmentResult.value();

  MeshletPipelineDesc desc{};
  desc.taskShader = meshletTaskShader_;
  desc.meshShader = meshletMeshShader_;
  desc.fragmentShader = meshletFragmentShader_;
  desc.colorFormats = {kFrameCompositionSceneColorFormat};
  desc.colorAttachmentCount = 1u;
  desc.depthFormat = kFrameCompositionDepthFormat;
  desc.cullMode = CullMode::Back;
  desc.polygonMode = PolygonMode::Fill;
  desc.numSamples = 1u;
  auto pipelineResult = gpu_.createMeshletPipeline(desc, "opaque_meshlet");
  if (pipelineResult.hasError()) {
    meshletShader_.reset();
    resetMeshletPipelineState();
    meshletPipelineUnsupported_ = true;
    return Result<bool, std::string>::makeError(pipelineResult.error());
  }
  meshletPipelineHandle_ = pipelineResult.value();

  desc.cullMode = CullMode::None;
  auto doubleSidedResult =
      gpu_.createMeshletPipeline(desc, "opaque_meshlet_double_sided");
  if (doubleSidedResult.hasError()) {
    destroyMeshletPipelineHandle(gpu_, meshletPipelineHandle_);
    meshletShader_.reset();
    resetMeshletPipelineState();
    meshletPipelineUnsupported_ = true;
    return Result<bool, std::string>::makeError(doubleSidedResult.error());
  }
  meshletDoubleSidedPipelineHandle_ = doubleSidedResult.value();

  MeshletPipelineDesc depthDesc = desc;
  depthDesc.fragmentShader = meshletDepthFragmentShader_;
  depthDesc.colorAttachmentCount = 0u;
  depthDesc.cullMode = CullMode::Back;
  auto depthPipelineResult =
      gpu_.createMeshletPipeline(depthDesc, "opaque_meshlet_depth");
  if (depthPipelineResult.hasError()) {
    destroyMeshletPipelineState();
    meshletShader_.reset();
    meshletPipelineUnsupported_ = true;
    return Result<bool, std::string>::makeError(depthPipelineResult.error());
  }
  meshletDepthPipelineHandle_ = depthPipelineResult.value();

  depthDesc.cullMode = CullMode::None;
  auto depthDoubleSidedResult = gpu_.createMeshletPipeline(
      depthDesc, "opaque_meshlet_depth_double_sided");
  if (depthDoubleSidedResult.hasError()) {
    destroyMeshletPipelineState();
    meshletShader_.reset();
    meshletPipelineUnsupported_ = true;
    return Result<bool, std::string>::makeError(depthDoubleSidedResult.error());
  }
  meshletDepthDoubleSidedPipelineHandle_ = depthDoubleSidedResult.value();

  MeshletPipelineDesc depthAlphaDesc = depthDesc;
  depthAlphaDesc.fragmentShader = meshletDepthAlphaFragmentShader_;
  depthAlphaDesc.cullMode = CullMode::Back;
  auto depthAlphaResult =
      gpu_.createMeshletPipeline(depthAlphaDesc, "opaque_meshlet_depth_alpha");
  if (depthAlphaResult.hasError()) {
    destroyMeshletPipelineState();
    meshletShader_.reset();
    meshletPipelineUnsupported_ = true;
    return Result<bool, std::string>::makeError(depthAlphaResult.error());
  }
  meshletDepthAlphaPipelineHandle_ = depthAlphaResult.value();

  depthAlphaDesc.cullMode = CullMode::None;
  auto depthAlphaDoubleSidedResult = gpu_.createMeshletPipeline(
      depthAlphaDesc, "opaque_meshlet_depth_alpha_double_sided");
  if (depthAlphaDoubleSidedResult.hasError()) {
    destroyMeshletPipelineState();
    meshletShader_.reset();
    meshletPipelineUnsupported_ = true;
    return Result<bool, std::string>::makeError(
        depthAlphaDoubleSidedResult.error());
  }
  meshletDepthAlphaDoubleSidedPipelineHandle_ =
      depthAlphaDoubleSidedResult.value();

  MeshletPipelineDesc normalDesc = desc;
  normalDesc.fragmentShader = meshletNormalFragmentShader_;
  normalDesc.colorFormats = {kFrameCompositionNormalFormat};
  normalDesc.colorAttachmentCount = 1u;
  normalDesc.cullMode = CullMode::Back;
  auto normalPipelineResult =
      gpu_.createMeshletPipeline(normalDesc, "opaque_meshlet_normal");
  if (normalPipelineResult.hasError()) {
    destroyMeshletPipelineState();
    meshletShader_.reset();
    meshletPipelineUnsupported_ = true;
    return Result<bool, std::string>::makeError(normalPipelineResult.error());
  }
  meshletNormalPipelineHandle_ = normalPipelineResult.value();

  normalDesc.cullMode = CullMode::None;
  auto normalDoubleSidedResult = gpu_.createMeshletPipeline(
      normalDesc, "opaque_meshlet_normal_double_sided");
  if (normalDoubleSidedResult.hasError()) {
    destroyMeshletPipelineState();
    meshletShader_.reset();
    meshletPipelineUnsupported_ = true;
    return Result<bool, std::string>::makeError(
        normalDoubleSidedResult.error());
  }
  meshletNormalDoubleSidedPipelineHandle_ = normalDoubleSidedResult.value();

  MeshletPipelineDesc velocityDesc = desc;
  velocityDesc.meshShader = meshletVelocityMeshShader_;
  velocityDesc.fragmentShader = meshletVelocityFragmentShader_;
  velocityDesc.colorFormats = {kFrameCompositionMotionVectorFormat};
  velocityDesc.colorAttachmentCount = 1u;
  velocityDesc.cullMode = CullMode::Back;
  auto velocityPipelineResult =
      gpu_.createMeshletPipeline(velocityDesc, "opaque_meshlet_velocity");
  if (velocityPipelineResult.hasError()) {
    destroyMeshletPipelineState();
    meshletShader_.reset();
    meshletPipelineUnsupported_ = true;
    return Result<bool, std::string>::makeError(velocityPipelineResult.error());
  }
  meshletVelocityPipelineHandle_ = velocityPipelineResult.value();

  velocityDesc.cullMode = CullMode::None;
  auto velocityDoubleSidedResult = gpu_.createMeshletPipeline(
      velocityDesc, "opaque_meshlet_velocity_double_sided");
  if (velocityDoubleSidedResult.hasError()) {
    destroyMeshletPipelineState();
    meshletShader_.reset();
    meshletPipelineUnsupported_ = true;
    return Result<bool, std::string>::makeError(
        velocityDoubleSidedResult.error());
  }
  meshletVelocityDoubleSidedPipelineHandle_ = velocityDoubleSidedResult.value();

  MeshletPipelineDesc reactiveMaskDesc = desc;
  reactiveMaskDesc.meshShader = meshletReactiveMaskMeshShader_;
  reactiveMaskDesc.fragmentShader = meshletReactiveMaskFragmentShader_;
  reactiveMaskDesc.colorFormats = {kFrameCompositionReactiveMaskFormat};
  reactiveMaskDesc.colorAttachmentCount = 1u;
  reactiveMaskDesc.cullMode = CullMode::Back;
  auto reactiveMaskPipelineResult = gpu_.createMeshletPipeline(
      reactiveMaskDesc, "opaque_meshlet_reactive_mask");
  if (reactiveMaskPipelineResult.hasError()) {
    destroyMeshletPipelineState();
    meshletShader_.reset();
    meshletPipelineUnsupported_ = true;
    return Result<bool, std::string>::makeError(
        reactiveMaskPipelineResult.error());
  }
  meshletReactiveMaskPipelineHandle_ = reactiveMaskPipelineResult.value();

  reactiveMaskDesc.cullMode = CullMode::None;
  auto reactiveMaskDoubleSidedResult = gpu_.createMeshletPipeline(
      reactiveMaskDesc, "opaque_meshlet_reactive_mask_double_sided");
  if (reactiveMaskDoubleSidedResult.hasError()) {
    destroyMeshletPipelineState();
    meshletShader_.reset();
    meshletPipelineUnsupported_ = true;
    return Result<bool, std::string>::makeError(
        reactiveMaskDoubleSidedResult.error());
  }
  meshletReactiveMaskDoubleSidedPipelineHandle_ =
      reactiveMaskDoubleSidedResult.value();
  meshletPipelineInitialized_ = true;
  meshletPipelineUnsupported_ = false;
  return Result<bool, std::string>::makeResult(true);
}

RenderPipelineHandle
OpaqueRenderer::selectMeshPipeline(bool doubleSided, bool tessellated) const {
  if (tessellated) {
    if (doubleSided && nuri::isValid(meshDoubleSidedTessPipelineHandle_)) {
      return meshDoubleSidedTessPipelineHandle_;
    }
    return meshTessPipelineHandle_;
  }
  if (doubleSided && nuri::isValid(meshDoubleSidedFillPipelineHandle_)) {
    return meshDoubleSidedFillPipelineHandle_;
  }
  return meshFillPipelineHandle_;
}

RenderPipelineHandle OpaqueRenderer::selectVelocityPipeline(
    RenderPipelineHandle sourcePipeline) const {
  const bool tessellated = isTessPipeline(sourcePipeline);
  const bool doubleSided = isDoubleSidedPipeline(sourcePipeline);
  if (tessellated) {
    if (doubleSided &&
        nuri::isValid(meshVelocityDoubleSidedTessPipelineHandle_)) {
      return meshVelocityDoubleSidedTessPipelineHandle_;
    }
    if (nuri::isValid(meshVelocityTessPipelineHandle_)) {
      return meshVelocityTessPipelineHandle_;
    }
    return {};
  }
  if (doubleSided && nuri::isValid(meshVelocityDoubleSidedPipelineHandle_)) {
    return meshVelocityDoubleSidedPipelineHandle_;
  }
  return meshVelocityPipelineHandle_;
}

RenderPipelineHandle OpaqueRenderer::selectReactiveMaskPipeline(
    RenderPipelineHandle sourcePipeline) const {
  const bool doubleSided = isDoubleSidedPipeline(sourcePipeline);
  if (doubleSided &&
      nuri::isValid(meshReactiveMaskDoubleSidedPipelineHandle_)) {
    return meshReactiveMaskDoubleSidedPipelineHandle_;
  }
  return meshReactiveMaskPipelineHandle_;
}

RenderPipelineHandle OpaqueRenderer::selectNormalPipeline(
    RenderPipelineHandle sourcePipeline) const {
  const bool tessellated = isTessPipeline(sourcePipeline);
  const bool doubleSided = isDoubleSidedPipeline(sourcePipeline);
  if (tessellated) {
    if (doubleSided &&
        nuri::isValid(meshNormalDoubleSidedTessPipelineHandle_)) {
      return meshNormalDoubleSidedTessPipelineHandle_;
    }
    if (nuri::isValid(meshNormalTessPipelineHandle_)) {
      return meshNormalTessPipelineHandle_;
    }
    return {};
  }
  if (doubleSided && nuri::isValid(meshNormalDoubleSidedPipelineHandle_)) {
    return meshNormalDoubleSidedPipelineHandle_;
  }
  return meshNormalPipelineHandle_;
}

RenderPipelineHandle
OpaqueRenderer::selectDepthPipeline(RenderPipelineHandle sourcePipeline,
                                    bool alphaMasked, bool msaa) const {
  const bool doubleSided = isDoubleSidedPipeline(sourcePipeline);
  const bool tessellated = isTessPipeline(sourcePipeline);
  if (alphaMasked) {
    if (tessellated) {
      if (msaa) {
        return doubleSided ? meshMsaaDepthAlphaDoubleSidedTessPipelineHandle_
                           : meshMsaaDepthAlphaTessPipelineHandle_;
      }
      return doubleSided ? meshDepthAlphaDoubleSidedTessPipelineHandle_
                         : meshDepthAlphaTessPipelineHandle_;
    }
    if (msaa) {
      return doubleSided ? meshMsaaDepthAlphaDoubleSidedPipelineHandle_
                         : meshMsaaDepthAlphaPipelineHandle_;
    }
    return doubleSided ? meshDepthAlphaDoubleSidedPipelineHandle_
                       : meshDepthAlphaPipelineHandle_;
  }
  if (tessellated) {
    if (msaa) {
      return doubleSided ? meshMsaaDepthDoubleSidedTessPipelineHandle_
                         : meshMsaaDepthTessPipelineHandle_;
    }
    return doubleSided ? meshDepthDoubleSidedTessPipelineHandle_
                       : meshDepthTessPipelineHandle_;
  }
  if (msaa) {
    return doubleSided ? meshMsaaDepthDoubleSidedPipelineHandle_
                       : meshMsaaDepthPipelineHandle_;
  }
  return doubleSided ? meshDepthDoubleSidedPipelineHandle_
                     : meshDepthPipelineHandle_;
}

RenderPipelineHandle
OpaqueRenderer::selectPickPipeline(RenderPipelineHandle sourcePipeline) const {
  const bool tessellated = isTessPipeline(sourcePipeline);
  const bool doubleSided = isDoubleSidedPipeline(sourcePipeline);
  if (tessellated) {
    if (doubleSided && nuri::isValid(meshPickDoubleSidedTessPipelineHandle_)) {
      return meshPickDoubleSidedTessPipelineHandle_;
    }
    if (nuri::isValid(meshPickTessPipelineHandle_)) {
      return meshPickTessPipelineHandle_;
    }
  }
  if (doubleSided && nuri::isValid(meshPickDoubleSidedPipelineHandle_)) {
    return meshPickDoubleSidedPipelineHandle_;
  }
  return meshPickPipelineHandle_;
}

RenderPipelineHandle OpaqueRenderer::selectShadowInspectPipeline(
    RenderPipelineHandle sourcePipeline) const {
  const bool tessellated = isTessPipeline(sourcePipeline);
  const bool doubleSided = isDoubleSidedPipeline(sourcePipeline);
  if (tessellated) {
    if (doubleSided &&
        nuri::isValid(meshShadowInspectDoubleSidedTessPipelineHandle_)) {
      return meshShadowInspectDoubleSidedTessPipelineHandle_;
    }
    if (nuri::isValid(meshShadowInspectTessPipelineHandle_)) {
      return meshShadowInspectTessPipelineHandle_;
    }
  }
  if (doubleSided &&
      nuri::isValid(meshShadowInspectDoubleSidedPipelineHandle_)) {
    return meshShadowInspectDoubleSidedPipelineHandle_;
  }
  return meshShadowInspectPipelineHandle_;
}

RenderPipelineHandle
OpaqueRenderer::selectMsaaScenePipeline(RenderPipelineHandle sourcePipeline,
                                        bool alphaMasked) const {
  if (isSamePipelineHandle(sourcePipeline, meshFillPipelineHandle_)) {
    if (alphaMasked && nuri::isValid(meshMsaaAlphaFillPipelineHandle_)) {
      return meshMsaaAlphaFillPipelineHandle_;
    }
    return meshMsaaFillPipelineHandle_;
  }
  if (isSamePipelineHandle(sourcePipeline,
                           meshDoubleSidedFillPipelineHandle_)) {
    if (alphaMasked &&
        nuri::isValid(meshMsaaAlphaDoubleSidedFillPipelineHandle_)) {
      return meshMsaaAlphaDoubleSidedFillPipelineHandle_;
    }
    return meshMsaaDoubleSidedFillPipelineHandle_;
  }
  if (isSamePipelineHandle(sourcePipeline, meshTessPipelineHandle_)) {
    if (alphaMasked && nuri::isValid(meshMsaaAlphaTessPipelineHandle_)) {
      return meshMsaaAlphaTessPipelineHandle_;
    }
    return meshMsaaTessPipelineHandle_;
  }
  if (isSamePipelineHandle(sourcePipeline,
                           meshDoubleSidedTessPipelineHandle_)) {
    if (alphaMasked &&
        nuri::isValid(meshMsaaAlphaDoubleSidedTessPipelineHandle_)) {
      return meshMsaaAlphaDoubleSidedTessPipelineHandle_;
    }
    return meshMsaaDoubleSidedTessPipelineHandle_;
  }
  if (isSamePipelineHandle(sourcePipeline, meshWireframePipelineHandle_)) {
    return meshMsaaWireframePipelineHandle_;
  }
  if (isSamePipelineHandle(sourcePipeline, meshTessWireframePipelineHandle_)) {
    return meshMsaaTessWireframePipelineHandle_;
  }
  if (isSamePipelineHandle(sourcePipeline, meshGsOverlayPipelineHandle_)) {
    return meshMsaaGsOverlayPipelineHandle_;
  }
  if (isSamePipelineHandle(sourcePipeline, meshGsTessOverlayPipelineHandle_)) {
    return meshMsaaGsTessOverlayPipelineHandle_;
  }
  return sourcePipeline;
}

bool OpaqueRenderer::isDoubleSidedPipeline(RenderPipelineHandle handle) const {
  return isSamePipelineHandle(handle, meshDoubleSidedFillPipelineHandle_) ||
         isSamePipelineHandle(handle, meshDoubleSidedTessPipelineHandle_) ||
         isSamePipelineHandle(handle, meshNormalDoubleSidedPipelineHandle_) ||
         isSamePipelineHandle(handle,
                              meshNormalDoubleSidedTessPipelineHandle_) ||
         isSamePipelineHandle(handle, meshMsaaDoubleSidedFillPipelineHandle_) ||
         isSamePipelineHandle(handle, meshMsaaDoubleSidedTessPipelineHandle_) ||
         isSamePipelineHandle(handle,
                              meshMsaaAlphaDoubleSidedFillPipelineHandle_) ||
         isSamePipelineHandle(handle,
                              meshMsaaAlphaDoubleSidedTessPipelineHandle_);
}

bool OpaqueRenderer::isTessPipeline(RenderPipelineHandle handle) const {
  return isSamePipelineHandle(handle, meshTessPipelineHandle_) ||
         isSamePipelineHandle(handle, meshDoubleSidedTessPipelineHandle_) ||
         isSamePipelineHandle(handle, meshNormalTessPipelineHandle_) ||
         isSamePipelineHandle(handle,
                              meshNormalDoubleSidedTessPipelineHandle_) ||
         isSamePipelineHandle(handle, meshMsaaTessPipelineHandle_) ||
         isSamePipelineHandle(handle, meshMsaaDoubleSidedTessPipelineHandle_) ||
         isSamePipelineHandle(handle, meshMsaaAlphaTessPipelineHandle_) ||
         isSamePipelineHandle(handle,
                              meshMsaaAlphaDoubleSidedTessPipelineHandle_);
}

Result<bool, std::string>
OpaqueRenderer::ensureWireframePipeline(bool requireMsaa) {
  if (wireframePipelineInitialized_ &&
      nuri::isValid(meshWireframePipelineHandle_) &&
      (!requireMsaa || (msaaWireframePipelineInitialized_ &&
                        nuri::isValid(meshMsaaWireframePipelineHandle_)))) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (wireframePipelineUnsupported_ ||
      (requireMsaa && msaaWireframePipelineUnsupported_)) {
    return Result<bool, std::string>::makeResult(false);
  }
  if (!nuri::isValid(meshFillPipelineHandle_)) {
    return Result<bool, std::string>::makeError(
        "OpaqueRenderer::ensureWireframePipeline: fill pipeline is invalid");
  }

  const Format depthFormat = kFrameCompositionDepthFormat;
  const RenderPipelineDesc wireframeDesc = meshPipelineDesc(
      kFrameCompositionSceneColorFormat, depthFormat, meshVertexShader_, {}, {},
      {}, meshFragmentShader_, PolygonMode::Line, Topology::Triangle, 0, true);

  if (!nuri::isValid(meshWireframePipelineHandle_)) {
    auto pipelineResult =
        gpu_.createRenderPipeline(wireframeDesc, "opaque_mesh_wireframe");
    if (pipelineResult.hasError()) {
      wireframePipelineUnsupported_ = true;
      if (!loggedWireframeFallbackUnsupported_) {
        loggedWireframeFallbackUnsupported_ = true;
        NURI_LOG_WARNING("OpaqueRenderer::ensureWireframePipeline: %s",
                         pipelineResult.error().c_str());
      }
      return Result<bool, std::string>::makeResult(false);
    }

    meshWireframePipelineHandle_ = pipelineResult.value();
    wireframePipelineInitialized_ = true;

    baseMeshWireframeDraw_ = baseMeshFillDraw_;
    baseMeshWireframeDraw_.pipeline = meshWireframePipelineHandle_;
    baseMeshWireframeDraw_.debugLabel = "OpaqueMeshWireframe";
  }

  if (requireMsaa && !msaaWireframePipelineInitialized_) {
    RenderPipelineDesc msaaDesc = wireframeDesc;
    msaaDesc.numSamples = kMsaa4xSampleCount;
    auto msaaResult =
        gpu_.createRenderPipeline(msaaDesc, "opaque_mesh_wireframe_msaa4x");
    if (msaaResult.hasError()) {
      msaaWireframePipelineUnsupported_ = true;
      if (!loggedWireframeFallbackUnsupported_) {
        loggedWireframeFallbackUnsupported_ = true;
        NURI_LOG_WARNING("OpaqueRenderer::ensureWireframePipeline: %s",
                         msaaResult.error().c_str());
      }
      return Result<bool, std::string>::makeResult(false);
    }
    meshMsaaWireframePipelineHandle_ = msaaResult.value();
    msaaWireframePipelineInitialized_ = true;
  }

  return Result<bool, std::string>::makeResult(
      !requireMsaa || nuri::isValid(meshMsaaWireframePipelineHandle_));
}

Result<bool, std::string>
OpaqueRenderer::ensureTessWireframePipeline(bool requireMsaa) {
  if (tessWireframePipelineInitialized_ &&
      nuri::isValid(meshTessWireframePipelineHandle_) &&
      (!requireMsaa || (msaaTessWireframePipelineInitialized_ &&
                        nuri::isValid(meshMsaaTessWireframePipelineHandle_)))) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (tessWireframePipelineUnsupported_ ||
      (requireMsaa && msaaTessWireframePipelineUnsupported_)) {
    return Result<bool, std::string>::makeResult(false);
  }
  if (!nuri::isValid(meshTessPipelineHandle_)) {
    return Result<bool, std::string>::makeResult(false);
  }

  const Format depthFormat = kFrameCompositionDepthFormat;
  const RenderPipelineDesc wireframeDesc = meshPipelineDesc(
      kFrameCompositionSceneColorFormat, depthFormat, meshTessVertexShader_,
      meshTessControlShader_, meshTessEvalShader_, {}, meshFragmentShader_,
      PolygonMode::Line, Topology::Patch, kTessellationPatchControlPoints,
      true);

  if (!nuri::isValid(meshTessWireframePipelineHandle_)) {
    auto pipelineResult =
        gpu_.createRenderPipeline(wireframeDesc, "opaque_mesh_tess_wireframe");
    if (pipelineResult.hasError()) {
      tessWireframePipelineUnsupported_ = true;
      if (!loggedTessWireframeFallbackUnsupported_) {
        loggedTessWireframeFallbackUnsupported_ = true;
        NURI_LOG_WARNING("OpaqueRenderer::ensureTessWireframePipeline: %s",
                         pipelineResult.error().c_str());
      }
      return Result<bool, std::string>::makeResult(false);
    }

    meshTessWireframePipelineHandle_ = pipelineResult.value();
    tessWireframePipelineInitialized_ = true;
  }

  if (requireMsaa && !msaaTessWireframePipelineInitialized_) {
    RenderPipelineDesc msaaDesc = wireframeDesc;
    msaaDesc.numSamples = kMsaa4xSampleCount;
    auto msaaResult = gpu_.createRenderPipeline(
        msaaDesc, "opaque_mesh_tess_wireframe_msaa4x");
    if (msaaResult.hasError()) {
      msaaTessWireframePipelineUnsupported_ = true;
      if (!loggedTessWireframeFallbackUnsupported_) {
        loggedTessWireframeFallbackUnsupported_ = true;
        NURI_LOG_WARNING("OpaqueRenderer::ensureTessWireframePipeline: %s",
                         msaaResult.error().c_str());
      }
      return Result<bool, std::string>::makeResult(false);
    }
    meshMsaaTessWireframePipelineHandle_ = msaaResult.value();
    msaaTessWireframePipelineInitialized_ = true;
  }

  return Result<bool, std::string>::makeResult(
      !requireMsaa || nuri::isValid(meshMsaaTessWireframePipelineHandle_));
}

Result<bool, std::string>
OpaqueRenderer::ensureGsOverlayPipeline(bool requireMsaa) {
  if (gsOverlayPipelineInitialized_ &&
      nuri::isValid(meshGsOverlayPipelineHandle_) &&
      (!requireMsaa || (msaaGsOverlayPipelineInitialized_ &&
                        nuri::isValid(meshMsaaGsOverlayPipelineHandle_)))) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (gsOverlayPipelineUnsupported_ ||
      (requireMsaa && msaaGsOverlayPipelineUnsupported_)) {
    return Result<bool, std::string>::makeResult(false);
  }
  if (!nuri::isValid(meshDebugOverlayGeometryShader_) ||
      !nuri::isValid(meshDebugOverlayFragmentShader_)) {
    gsOverlayPipelineUnsupported_ = true;
    if (!loggedGsOverlayUnsupported_) {
      loggedGsOverlayUnsupported_ = true;
      NURI_LOG_WARNING("OpaqueRenderer::ensureGsOverlayPipeline: debug overlay "
                       "shaders are unavailable, fallback to line overlay");
    }
    return Result<bool, std::string>::makeResult(false);
  }

  const Format depthFormat = kFrameCompositionDepthFormat;
  const RenderPipelineDesc overlayDesc = meshPipelineDesc(
      kFrameCompositionSceneColorFormat, depthFormat, meshVertexShader_, {}, {},
      meshDebugOverlayGeometryShader_, meshDebugOverlayFragmentShader_,
      PolygonMode::Fill, Topology::Triangle, 0, true);

  if (!nuri::isValid(meshGsOverlayPipelineHandle_)) {
    auto pipelineResult =
        gpu_.createRenderPipeline(overlayDesc, "opaque_mesh_overlay_gs");
    if (pipelineResult.hasError()) {
      gsOverlayPipelineUnsupported_ = true;
      if (!loggedGsOverlayUnsupported_) {
        loggedGsOverlayUnsupported_ = true;
        NURI_LOG_WARNING("OpaqueRenderer::ensureGsOverlayPipeline: %s",
                         pipelineResult.error().c_str());
      }
      return Result<bool, std::string>::makeResult(false);
    }

    meshGsOverlayPipelineHandle_ = pipelineResult.value();
    gsOverlayPipelineInitialized_ = true;
  }

  if (requireMsaa && !msaaGsOverlayPipelineInitialized_) {
    RenderPipelineDesc msaaDesc = overlayDesc;
    msaaDesc.numSamples = kMsaa4xSampleCount;
    auto msaaResult =
        gpu_.createRenderPipeline(msaaDesc, "opaque_mesh_overlay_gs_msaa4x");
    if (msaaResult.hasError()) {
      msaaGsOverlayPipelineUnsupported_ = true;
      if (!loggedGsOverlayUnsupported_) {
        loggedGsOverlayUnsupported_ = true;
        NURI_LOG_WARNING("OpaqueRenderer::ensureGsOverlayPipeline: %s",
                         msaaResult.error().c_str());
      }
      return Result<bool, std::string>::makeResult(false);
    }
    meshMsaaGsOverlayPipelineHandle_ = msaaResult.value();
    msaaGsOverlayPipelineInitialized_ = true;
  }
  return Result<bool, std::string>::makeResult(
      !requireMsaa || nuri::isValid(meshMsaaGsOverlayPipelineHandle_));
}

Result<bool, std::string>
OpaqueRenderer::ensureGsTessOverlayPipeline(bool requireMsaa) {
  if (gsTessOverlayPipelineInitialized_ &&
      nuri::isValid(meshGsTessOverlayPipelineHandle_) &&
      (!requireMsaa || (msaaGsTessOverlayPipelineInitialized_ &&
                        nuri::isValid(meshMsaaGsTessOverlayPipelineHandle_)))) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (gsTessOverlayPipelineUnsupported_ ||
      (requireMsaa && msaaGsTessOverlayPipelineUnsupported_)) {
    return Result<bool, std::string>::makeResult(false);
  }
  if (!nuri::isValid(meshTessPipelineHandle_)) {
    return Result<bool, std::string>::makeResult(false);
  }
  if (!nuri::isValid(meshDebugOverlayGeometryShader_) ||
      !nuri::isValid(meshDebugOverlayFragmentShader_)) {
    gsTessOverlayPipelineUnsupported_ = true;
    if (!loggedGsTessOverlayUnsupported_) {
      loggedGsTessOverlayUnsupported_ = true;
      NURI_LOG_WARNING("OpaqueRenderer::ensureGsTessOverlayPipeline: debug "
                       "overlay shaders are unavailable, fallback to line "
                       "overlay");
    }
    return Result<bool, std::string>::makeResult(false);
  }

  const Format depthFormat = kFrameCompositionDepthFormat;
  const RenderPipelineDesc overlayDesc =
      meshPipelineDesc(kFrameCompositionSceneColorFormat, depthFormat,
                       meshTessVertexShader_, meshTessControlShader_,
                       meshTessEvalShader_, meshDebugOverlayGeometryShader_,
                       meshDebugOverlayFragmentShader_, PolygonMode::Fill,
                       Topology::Patch, kTessellationPatchControlPoints, true);

  if (!nuri::isValid(meshGsTessOverlayPipelineHandle_)) {
    auto pipelineResult =
        gpu_.createRenderPipeline(overlayDesc, "opaque_mesh_tess_overlay_gs");
    if (pipelineResult.hasError()) {
      gsTessOverlayPipelineUnsupported_ = true;
      if (!loggedGsTessOverlayUnsupported_) {
        loggedGsTessOverlayUnsupported_ = true;
        NURI_LOG_WARNING("OpaqueRenderer::ensureGsTessOverlayPipeline: %s",
                         pipelineResult.error().c_str());
      }
      return Result<bool, std::string>::makeResult(false);
    }

    meshGsTessOverlayPipelineHandle_ = pipelineResult.value();
    gsTessOverlayPipelineInitialized_ = true;
  }

  if (requireMsaa && !msaaGsTessOverlayPipelineInitialized_) {
    RenderPipelineDesc msaaDesc = overlayDesc;
    msaaDesc.numSamples = kMsaa4xSampleCount;
    auto msaaResult = gpu_.createRenderPipeline(
        msaaDesc, "opaque_mesh_tess_overlay_gs_msaa4x");
    if (msaaResult.hasError()) {
      msaaGsTessOverlayPipelineUnsupported_ = true;
      if (!loggedGsTessOverlayUnsupported_) {
        loggedGsTessOverlayUnsupported_ = true;
        NURI_LOG_WARNING("OpaqueRenderer::ensureGsTessOverlayPipeline: %s",
                         msaaResult.error().c_str());
      }
      return Result<bool, std::string>::makeResult(false);
    }
    meshMsaaGsTessOverlayPipelineHandle_ = msaaResult.value();
    msaaGsTessOverlayPipelineInitialized_ = true;
  }
  return Result<bool, std::string>::makeResult(
      !requireMsaa || nuri::isValid(meshMsaaGsTessOverlayPipelineHandle_));
}

void OpaqueRenderer::resetOverlayPipelineState() {
  if (nuri::isValid(meshGsOverlayPipelineHandle_)) {
    gpu_.destroyRenderPipeline(meshGsOverlayPipelineHandle_);
  }
  if (nuri::isValid(meshGsTessOverlayPipelineHandle_)) {
    gpu_.destroyRenderPipeline(meshGsTessOverlayPipelineHandle_);
  }
  if (nuri::isValid(meshWireframePipelineHandle_)) {
    gpu_.destroyRenderPipeline(meshWireframePipelineHandle_);
  }
  if (nuri::isValid(meshTessWireframePipelineHandle_)) {
    gpu_.destroyRenderPipeline(meshTessWireframePipelineHandle_);
  }
  if (nuri::isValid(meshMsaaGsOverlayPipelineHandle_)) {
    gpu_.destroyRenderPipeline(meshMsaaGsOverlayPipelineHandle_);
  }
  if (nuri::isValid(meshMsaaGsTessOverlayPipelineHandle_)) {
    gpu_.destroyRenderPipeline(meshMsaaGsTessOverlayPipelineHandle_);
  }
  if (nuri::isValid(meshMsaaWireframePipelineHandle_)) {
    gpu_.destroyRenderPipeline(meshMsaaWireframePipelineHandle_);
  }
  if (nuri::isValid(meshMsaaTessWireframePipelineHandle_)) {
    gpu_.destroyRenderPipeline(meshMsaaTessWireframePipelineHandle_);
  }
  meshGsOverlayPipelineHandle_ = {};
  meshGsTessOverlayPipelineHandle_ = {};
  meshWireframePipelineHandle_ = {};
  meshTessWireframePipelineHandle_ = {};
  meshMsaaGsOverlayPipelineHandle_ = {};
  meshMsaaGsTessOverlayPipelineHandle_ = {};
  meshMsaaWireframePipelineHandle_ = {};
  meshMsaaTessWireframePipelineHandle_ = {};
  gsOverlayPipelineInitialized_ = false;
  gsTessOverlayPipelineInitialized_ = false;
  wireframePipelineInitialized_ = false;
  tessWireframePipelineInitialized_ = false;
  msaaWireframePipelineInitialized_ = false;
  msaaTessWireframePipelineInitialized_ = false;
  msaaGsOverlayPipelineInitialized_ = false;
  msaaGsTessOverlayPipelineInitialized_ = false;
  gsOverlayPipelineUnsupported_ = false;
  gsTessOverlayPipelineUnsupported_ = false;
  wireframePipelineUnsupported_ = false;
  tessWireframePipelineUnsupported_ = false;
  msaaWireframePipelineUnsupported_ = false;
  msaaTessWireframePipelineUnsupported_ = false;
  msaaGsOverlayPipelineUnsupported_ = false;
  msaaGsTessOverlayPipelineUnsupported_ = false;
  loggedWireframeFallbackUnsupported_ = false;
  loggedTessWireframeFallbackUnsupported_ = false;
  loggedGsOverlayUnsupported_ = false;
  loggedGsTessOverlayUnsupported_ = false;
  baseMeshWireframeDraw_ = {};
}

void OpaqueRenderer::invalidateAutoLodCache() {
  autoLodCache_.valid = false;
  autoLodCache_.remapCount = 0;
  autoLodCache_.instanceCount = 0;
  autoLodCache_.submesh = nullptr;
  autoLodCache_.frameIndex = std::numeric_limits<uint64_t>::max();
  autoLodCache_.bucketCounts.fill(0);
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
  for (uint64_t &slotSignature : indirectUploadSignatures_) {
    slotSignature = kInvalidDrawSignature;
  }
}

void OpaqueRenderer::destroyMeshPipelineState() {
  destroyPipelineHandle(gpu_, depthPyramidPipelineHandle_);
  destroyPipelineHandle(gpu_, meshMsaaDepthAlphaDoubleSidedTessPipelineHandle_);
  destroyPipelineHandle(gpu_, meshMsaaDepthAlphaTessPipelineHandle_);
  destroyPipelineHandle(gpu_, meshMsaaDepthAlphaDoubleSidedPipelineHandle_);
  destroyPipelineHandle(gpu_, meshMsaaDepthAlphaPipelineHandle_);
  destroyPipelineHandle(gpu_, meshMsaaDepthDoubleSidedTessPipelineHandle_);
  destroyPipelineHandle(gpu_, meshMsaaDepthTessPipelineHandle_);
  destroyPipelineHandle(gpu_, meshMsaaDepthDoubleSidedPipelineHandle_);
  destroyPipelineHandle(gpu_, meshMsaaDepthPipelineHandle_);
  destroyPipelineHandle(gpu_, meshDepthAlphaDoubleSidedTessPipelineHandle_);
  destroyPipelineHandle(gpu_, meshDepthAlphaTessPipelineHandle_);
  destroyPipelineHandle(gpu_, meshDepthAlphaDoubleSidedPipelineHandle_);
  destroyPipelineHandle(gpu_, meshDepthAlphaPipelineHandle_);
  destroyPipelineHandle(gpu_, meshDepthDoubleSidedTessPipelineHandle_);
  destroyPipelineHandle(gpu_, meshDepthTessPipelineHandle_);
  destroyPipelineHandle(gpu_, meshDepthDoubleSidedPipelineHandle_);
  destroyPipelineHandle(gpu_, meshDepthPipelineHandle_);
  destroyPipelineHandle(gpu_, meshPickDoubleSidedTessPipelineHandle_);
  destroyPipelineHandle(gpu_, meshPickTessPipelineHandle_);
  destroyPipelineHandle(gpu_, meshPickDoubleSidedPipelineHandle_);
  destroyPipelineHandle(gpu_, meshPickPipelineHandle_);
  destroyPipelineHandle(gpu_, meshShadowInspectDoubleSidedTessPipelineHandle_);
  destroyPipelineHandle(gpu_, meshShadowInspectTessPipelineHandle_);
  destroyPipelineHandle(gpu_, meshShadowInspectDoubleSidedPipelineHandle_);
  destroyPipelineHandle(gpu_, meshShadowInspectPipelineHandle_);
  destroyPipelineHandle(gpu_, meshVelocityDoubleSidedTessPipelineHandle_);
  destroyPipelineHandle(gpu_, meshVelocityTessPipelineHandle_);
  destroyPipelineHandle(gpu_, meshVelocityDoubleSidedPipelineHandle_);
  destroyPipelineHandle(gpu_, meshVelocityPipelineHandle_);
  destroyPipelineHandle(gpu_, meshReactiveMaskDoubleSidedPipelineHandle_);
  destroyPipelineHandle(gpu_, meshReactiveMaskPipelineHandle_);
  destroyPipelineHandle(gpu_, meshNormalDoubleSidedTessPipelineHandle_);
  destroyPipelineHandle(gpu_, meshNormalTessPipelineHandle_);
  destroyPipelineHandle(gpu_, meshNormalDoubleSidedPipelineHandle_);
  destroyPipelineHandle(gpu_, meshNormalPipelineHandle_);
  destroyPipelineHandle(gpu_, meshMsaaAlphaDoubleSidedTessPipelineHandle_);
  destroyPipelineHandle(gpu_, meshMsaaAlphaTessPipelineHandle_);
  destroyPipelineHandle(gpu_, meshMsaaAlphaDoubleSidedFillPipelineHandle_);
  destroyPipelineHandle(gpu_, meshMsaaAlphaFillPipelineHandle_);
  destroyPipelineHandle(gpu_, meshMsaaDoubleSidedTessPipelineHandle_);
  destroyPipelineHandle(gpu_, meshMsaaTessPipelineHandle_);
  destroyPipelineHandle(gpu_, meshMsaaDoubleSidedFillPipelineHandle_);
  destroyPipelineHandle(gpu_, meshMsaaFillPipelineHandle_);
  destroyPipelineHandle(gpu_, meshDoubleSidedTessPipelineHandle_);
  destroyPipelineHandle(gpu_, meshTessPipelineHandle_);
  destroyPipelineHandle(gpu_, meshDoubleSidedFillPipelineHandle_);
  resetMeshPipelineState();
}

void OpaqueRenderer::destroyMeshletPipelineState() {
  destroyMeshletPipelineHandle(gpu_,
                               meshletReactiveMaskDoubleSidedPipelineHandle_);
  destroyMeshletPipelineHandle(gpu_, meshletReactiveMaskPipelineHandle_);
  destroyMeshletPipelineHandle(gpu_, meshletVelocityDoubleSidedPipelineHandle_);
  destroyMeshletPipelineHandle(gpu_, meshletVelocityPipelineHandle_);
  destroyMeshletPipelineHandle(gpu_, meshletNormalDoubleSidedPipelineHandle_);
  destroyMeshletPipelineHandle(gpu_, meshletNormalPipelineHandle_);
  destroyMeshletPipelineHandle(gpu_,
                               meshletDepthAlphaDoubleSidedPipelineHandle_);
  destroyMeshletPipelineHandle(gpu_, meshletDepthAlphaPipelineHandle_);
  destroyMeshletPipelineHandle(gpu_, meshletDepthDoubleSidedPipelineHandle_);
  destroyMeshletPipelineHandle(gpu_, meshletDepthPipelineHandle_);
  destroyMeshletPipelineHandle(gpu_, meshletDoubleSidedPipelineHandle_);
  destroyMeshletPipelineHandle(gpu_, meshletPipelineHandle_);
  resetMeshletPipelineState();
}

void OpaqueRenderer::resetMeshletPipelineState() {
  meshletPipelineHandle_ = {};
  meshletDoubleSidedPipelineHandle_ = {};
  meshletDepthPipelineHandle_ = {};
  meshletDepthDoubleSidedPipelineHandle_ = {};
  meshletDepthAlphaPipelineHandle_ = {};
  meshletDepthAlphaDoubleSidedPipelineHandle_ = {};
  meshletNormalPipelineHandle_ = {};
  meshletNormalDoubleSidedPipelineHandle_ = {};
  meshletVelocityPipelineHandle_ = {};
  meshletVelocityDoubleSidedPipelineHandle_ = {};
  meshletReactiveMaskPipelineHandle_ = {};
  meshletReactiveMaskDoubleSidedPipelineHandle_ = {};
  meshletTaskShader_ = {};
  meshletMeshShader_ = {};
  meshletFragmentShader_ = {};
  meshletDepthFragmentShader_ = {};
  meshletDepthAlphaFragmentShader_ = {};
  meshletNormalFragmentShader_ = {};
  meshletVelocityMeshShader_ = {};
  meshletVelocityFragmentShader_ = {};
  meshletReactiveMaskMeshShader_ = {};
  meshletReactiveMaskFragmentShader_ = {};
  meshletPipelineInitialized_ = false;
  meshletPipelineUnsupported_ = false;
}

void OpaqueRenderer::resetMeshPipelineState() {
  meshFillPipelineHandle_ = {};
  meshDoubleSidedFillPipelineHandle_ = {};
  meshTessPipelineHandle_ = {};
  meshDoubleSidedTessPipelineHandle_ = {};
  meshMsaaFillPipelineHandle_ = {};
  meshMsaaDoubleSidedFillPipelineHandle_ = {};
  meshMsaaTessPipelineHandle_ = {};
  meshMsaaDoubleSidedTessPipelineHandle_ = {};
  meshMsaaAlphaFillPipelineHandle_ = {};
  meshMsaaAlphaDoubleSidedFillPipelineHandle_ = {};
  meshMsaaAlphaTessPipelineHandle_ = {};
  meshMsaaAlphaDoubleSidedTessPipelineHandle_ = {};
  meshPickPipelineHandle_ = {};
  meshPickDoubleSidedPipelineHandle_ = {};
  meshPickTessPipelineHandle_ = {};
  meshPickDoubleSidedTessPipelineHandle_ = {};
  meshShadowInspectPipelineHandle_ = {};
  meshShadowInspectDoubleSidedPipelineHandle_ = {};
  meshShadowInspectTessPipelineHandle_ = {};
  meshShadowInspectDoubleSidedTessPipelineHandle_ = {};
  meshVelocityPipelineHandle_ = {};
  meshVelocityDoubleSidedPipelineHandle_ = {};
  meshVelocityTessPipelineHandle_ = {};
  meshVelocityDoubleSidedTessPipelineHandle_ = {};
  meshReactiveMaskPipelineHandle_ = {};
  meshReactiveMaskDoubleSidedPipelineHandle_ = {};
  meshNormalPipelineHandle_ = {};
  meshNormalDoubleSidedPipelineHandle_ = {};
  meshNormalTessPipelineHandle_ = {};
  meshNormalDoubleSidedTessPipelineHandle_ = {};
  meshDepthPipelineHandle_ = {};
  meshDepthDoubleSidedPipelineHandle_ = {};
  meshDepthTessPipelineHandle_ = {};
  meshDepthDoubleSidedTessPipelineHandle_ = {};
  meshDepthAlphaPipelineHandle_ = {};
  meshDepthAlphaDoubleSidedPipelineHandle_ = {};
  meshDepthAlphaTessPipelineHandle_ = {};
  meshDepthAlphaDoubleSidedTessPipelineHandle_ = {};
  meshMsaaDepthPipelineHandle_ = {};
  meshMsaaDepthDoubleSidedPipelineHandle_ = {};
  meshMsaaDepthTessPipelineHandle_ = {};
  meshMsaaDepthDoubleSidedTessPipelineHandle_ = {};
  meshMsaaDepthAlphaPipelineHandle_ = {};
  meshMsaaDepthAlphaDoubleSidedPipelineHandle_ = {};
  meshMsaaDepthAlphaTessPipelineHandle_ = {};
  meshMsaaDepthAlphaDoubleSidedTessPipelineHandle_ = {};
  depthPyramidPipelineHandle_ = {};
  baseMeshFillDraw_ = {};
}

void OpaqueRenderer::updateFastAutoLodCache(
    const Submesh *submesh, const glm::vec3 &cameraPosition,
    const std::array<float, 3> &sortedLodThresholds,
    const std::array<size_t, Submesh::kMaxLodCount> &bucketCounts,
    size_t remapCount, size_t instanceCount, uint64_t frameIndex) {
  if (submesh == nullptr) {
    invalidateAutoLodCache();
    return;
  }

  autoLodCache_.valid = true;
  autoLodCache_.cameraPos = cameraPosition;
  autoLodCache_.thresholds = sortedLodThresholds;
  autoLodCache_.bucketCounts = bucketCounts;
  autoLodCache_.remapCount = remapCount;
  autoLodCache_.instanceCount = instanceCount;
  autoLodCache_.submesh = submesh;
  autoLodCache_.frameIndex = frameIndex;
}

void OpaqueRenderer::destroyDepthPyramidTextures() {
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
  if (instanceCentersPhaseBuffer_ && instanceCentersPhaseBuffer_->valid()) {
    gpu_.destroyBuffer(instanceCentersPhaseBuffer_->handle());
  }
  instanceCentersPhaseBuffer_.reset();
  instanceCentersPhaseBufferCapacityBytes_ = 0;

  if (instanceLodBoundsBuffer_ && instanceLodBoundsBuffer_->valid()) {
    gpu_.destroyBuffer(instanceLodBoundsBuffer_->handle());
  }
  instanceLodBoundsBuffer_.reset();
  instanceLodBoundsBufferCapacityBytes_ = 0;

  if (instanceBaseMatricesBuffer_ && instanceBaseMatricesBuffer_->valid()) {
    gpu_.destroyBuffer(instanceBaseMatricesBuffer_->handle());
  }
  instanceBaseMatricesBuffer_.reset();
  instanceBaseMatricesBufferCapacityBytes_ = 0;

  for (DynamicBufferSlot &slot : instanceMatricesRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
    slot.buffer.reset();
    slot.capacityBytes = 0;
  }
  for (DynamicBufferSlot &slot : previousInstanceMatricesRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
    slot.buffer.reset();
    slot.capacityBytes = 0;
  }
  for (DynamicBufferSlot &slot : velocityInstanceFlagsRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
    slot.buffer.reset();
    slot.capacityBytes = 0;
  }
  for (DynamicBufferSlot &slot : velocityFrameDataRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
    slot.buffer.reset();
    slot.capacityBytes = 0;
  }
  for (DynamicBufferSlot &slot : velocityGeometryRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
    slot.buffer.reset();
    slot.capacityBytes = 0;
  }
  for (DynamicBufferSlot &slot : instanceRemapRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
    slot.buffer.reset();
    slot.capacityBytes = 0;
  }
  for (DynamicBufferSlot &slot : indirectCommandRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
    slot.buffer.reset();
    slot.capacityBytes = 0;
  }
  for (DynamicBufferSlot &slot : meshletBatchRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
    slot.buffer.reset();
    slot.capacityBytes = 0;
  }
  for (DynamicBufferSlot &slot : visibilityCandidateRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
    slot.buffer.reset();
    slot.capacityBytes = 0;
  }
  for (DynamicBufferSlot &slot : visibilityPassRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
    slot.buffer.reset();
    slot.capacityBytes = 0;
  }
  for (DynamicBufferSlot &slot : visibilityVisibleIndexRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
    slot.buffer.reset();
    slot.capacityBytes = 0;
  }
  for (DynamicBufferSlot &slot : visibilityCounterRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
    slot.buffer.reset();
    slot.capacityBytes = 0;
  }
  for (DynamicBufferSlot &slot : visibilityMeshletDispatchRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
    slot.buffer.reset();
    slot.capacityBytes = 0;
  }
  for (DynamicBufferSlot &slot : visibilityMeshletIndirectCommandRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
    slot.buffer.reset();
    slot.capacityBytes = 0;
  }
  instanceMatricesRing_.clear();
  previousInstanceMatricesRing_.clear();
  velocityInstanceFlagsRing_.clear();
  velocityFrameDataRing_.clear();
  velocityGeometryRing_.clear();
  instanceRemapRing_.clear();
  indirectCommandRing_.clear();
  meshletBatchRing_.clear();
  visibilityCandidateRing_.clear();
  visibilityPassRing_.clear();
  visibilityVisibleIndexRing_.clear();
  visibilityCounterRing_.clear();
  visibilityMeshletDispatchRing_.clear();
  visibilityMeshletIndirectCommandRing_.clear();
  instanceMatricesUploadVersions_.clear();
  remapUploadSignatures_.clear();
  indirectUploadSignatures_.clear();
  visibilityCounterRingPublishedFrames_.clear();
  visibilityExpectedVisibleIndexCounts_.clear();
  visibilityExpectedVisibleIndexHashes_.clear();
  visibilityVisibleIndexReadback_.clear();
  invalidateIndirectPackCache();
}

} // namespace nuri
