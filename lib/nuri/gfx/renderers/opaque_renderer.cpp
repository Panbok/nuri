#include "nuri/pch.h"

#include "nuri/gfx/renderers/opaque_renderer.h"

#include "nuri/core/containers/hash_map.h"
#include "nuri/core/containers/hash_set.h"
#include "nuri/core/log.h"
#include "nuri/core/pmr_scratch.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/renderers/detail/renderable_material_resolution.h"
#include "nuri/gfx/renderers/detail/shadow_math.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/scene/render_scene.h"

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
constexpr uint32_t kTessellationPatchControlPoints = 3;
constexpr size_t kIndirectCountHeaderBytes = sizeof(uint32_t);
constexpr uint32_t kMaxIndirectCommandsPerDraw = 1024u;
constexpr size_t kMaxDrawItemsForIndirectPath = 8192u;
constexpr uint32_t kUnlimitedTessInstanceCap = 0u;
constexpr float kOverlayDepthBiasConstant = -1.0f;
constexpr float kOverlayDepthBiasSlope = -1.0f;
constexpr uint32_t kAutoLodCacheInvalidationSeed = 1664525u;
constexpr uint32_t kAutoLodCacheInvalidationMagic = 1013904223u;
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

  bool operator==(const BatchKey &other) const {
    return isSamePipelineHandle(pipeline, other.pipeline) &&
           isSameBufferHandle(indexBuffer, other.indexBuffer) &&
           indexBufferOffset == other.indexBufferOffset &&
           indexCount == other.indexCount && firstIndex == other.firstIndex &&
           vertexBufferAddress == other.vertexBufferAddress &&
           vertexDecodeBufferAddress == other.vertexDecodeBufferAddress &&
           vertexDecodeIndex == other.vertexDecodeIndex &&
           packedVertexFormat == other.packedVertexFormat &&
           materialIndex == other.materialIndex;
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
      instanceRemapRing_(resolveMemoryResource(memory)),
      indirectCommandRing_(resolveMemoryResource(memory)),
      singleInstanceBatchCaches_(resolveMemoryResource(memory)),
      staticBatchCache_(resolveMemoryResource(memory)),
      renderableTemplates_(resolveMemoryResource(memory)),
      meshDrawTemplates_(resolveMemoryResource(memory)),
      indirectSourceDrawIndices_(resolveMemoryResource(memory)),
      instanceMatricesUploadVersions_(resolveMemoryResource(memory)),
      indirectUploadSignatures_(resolveMemoryResource(memory)),
      remapUploadSignatures_(resolveMemoryResource(memory)),
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
      depthPyramidDependencyTextures_(resolveMemoryResource(memory)),
      shadowSdsmReducePushConstants_(resolveMemoryResource(memory)),
      shadowSdsmHistogramReducePushConstants_(resolveMemoryResource(memory)),
      shadowSdsmReduceDispatches_(resolveMemoryResource(memory)),
      shadowSdsmReduceDependencyBuffers_(resolveMemoryResource(memory)),
      shadowSdsmReduceDependencyTextures_(resolveMemoryResource(memory)),
      preDispatches_(resolveMemoryResource(memory)),
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
  meshPickFragmentShader_ = {};
  meshShadowInspectFragmentShader_ = {};
  meshVelocityVertexShader_ = {};
  meshVelocityFragmentShader_ = {};
  meshReactiveMaskVertexShader_ = {};
  meshReactiveMaskFragmentShader_ = {};
  meshNormalFragmentShader_ = {};
  depthFragmentShader_ = {};
  depthAlphaFragmentShader_ = {};
  depthPyramidVertexShader_ = {};
  depthPyramidFragmentShader_ = {};
  computeShaderHandle_ = {};
  computePipelineHandle_ = {};
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
  depthPyramidDependencyTextures_.clear();
  preDispatches_.clear();
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
    }
  }
}

Result<bool, std::string>
OpaqueRenderer::buildOpaquePasses(RenderFrameContext &frame,
                                  std::pmr::vector<PreparedGraphPass> &out) {
  NURI_PROFILER_FUNCTION();
  frame.metrics.opaque = {};
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
  const uint64_t geometryMutationVersion = gpu_.geometryMutationVersion();
  const bool hasGeometryMutationTracking = geometryMutationVersion != 0;
  if (topologyDirty || materialDirty) {
    auto cacheResult = rebuildSceneCache(
        *frame.scene, *frame.resources,
        static_cast<uint32_t>(materialSnapshot.headers.size()));
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
  const bool previousCacheValid =
      hasTaaVelocityInstances && frame.camera.historyValid &&
      frame.camera.temporalDataValid &&
      previousTransformSceneId_ == frame.scene->id() &&
      previousTransformCaptureFrameIndex_ !=
          std::numeric_limits<uint64_t>::max() &&
      previousTransformCaptureFrameIndex_ < frame.frameIndex;
  const bool staticVelocityScene = hasTaaVelocityInstances &&
                                   !settings.opaque.enableInstanceAnimation &&
                                   animationSceneData == nullptr;
  const bool canReuseStaticPreviousMatrices =
      staticVelocityScene && previousCacheValid && !transformDirty &&
      !animationSceneStateDirty;
  const bool canUseAllInvalidVelocityFlags =
      staticVelocityScene && !previousCacheValid;
  const VelocityInstanceFlagsMode velocityInstanceFlagsMode =
      canReuseStaticPreviousMatrices
          ? VelocityInstanceFlagsMode::AllValid
          : (canUseAllInvalidVelocityFlags
                 ? VelocityInstanceFlagsMode::AllInvalid
                 : VelocityInstanceFlagsMode::Buffer);
  const bool needsVelocityInstanceBufferUpload =
      hasTaaVelocityInstances &&
      velocityInstanceFlagsMode == VelocityInstanceFlagsMode::Buffer;
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
    instanceStaticBuffersDirty_ = false;
  }

  if (materialDirty) {
    cachedMaterialVersion_ = materialSnapshot.version;
  }
  if (materialDirty || materialTextureAccessHandles_.empty()) {
    NURI_PROFILER_ZONE("OpaqueRenderer.material_access_cache",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    auto materialAccessCacheResult =
        rebuildMaterialTextureAccessCache(*frame.scene, *frame.resources);
    if (materialAccessCacheResult.hasError()) {
      return materialAccessCacheResult;
    }
    NURI_PROFILER_ZONE_END();
  }

  const uint64_t frameDataAddress = sceneGpu->frameDataAddress;
  const uint64_t instanceCentersPhaseAddress =
      gpu_.getBufferDeviceAddress(instanceCentersPhaseBuffer_->handle());
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
      needsVelocityInstanceBufferUpload &&
              frameSlot < previousInstanceMatricesRing_.size() &&
              previousInstanceMatricesRing_[frameSlot].buffer
          ? previousInstanceMatricesRing_[frameSlot].buffer->handle()
          : BufferHandle{};
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
  const bool reuseCurrentMatricesForVelocity =
      canReuseStaticPreviousMatrices || canUseAllInvalidVelocityFlags;
  const uint64_t previousInstanceMatricesAddress =
      reuseCurrentMatricesForVelocity
          ? instanceMatricesAddress
          : (nuri::isValid(previousInstanceMatricesBufferHandle)
                 ? gpu_.getBufferDeviceAddress(
                       previousInstanceMatricesBufferHandle)
                 : 0u);
  const uint64_t velocityInstanceFlagsAddress =
      nuri::isValid(velocityInstanceFlagsBufferHandle)
          ? gpu_.getBufferDeviceAddress(velocityInstanceFlagsBufferHandle)
          : 0u;
  const uint64_t velocityFrameDataAddress =
      nuri::isValid(velocityFrameDataBufferHandle)
          ? gpu_.getBufferDeviceAddress(velocityFrameDataBufferHandle)
          : 0u;
  if (frameDataAddress == 0 || instanceCentersPhaseAddress == 0 ||
      instanceBaseMatricesAddress == 0 || instanceMatricesAddress == 0 ||
      (hasTaaVelocityInstances && (previousInstanceMatricesAddress == 0u ||
                                   (needsVelocityInstanceBufferUpload &&
                                    velocityInstanceFlagsAddress == 0u) ||
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
    double totalObjectMotion = 0.0;
    float maxObjectMotion = 0.0f;

    if (canReuseStaticPreviousMatrices) {
      validPreviousCount = saturateToU32(instanceCount);
    } else if (canUseAllInvalidVelocityFlags) {
      missingPreviousCount = saturateToU32(instanceCount);
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

    const VelocityFrameGpuData velocityFrameData{
        .currentViewProjNoJitter = frame.camera.currentUnjitteredViewProj,
        .previousViewProjNoJitter = frame.camera.previousUnjitteredViewProj,
        .instanceFlagsMode = glm::uvec4(
            static_cast<uint32_t>(velocityInstanceFlagsMode), 0u, 0u, 0u),
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
  const bool wireOverlayRequested =
      debugVisualization == OpaqueDebugVisualization::WireframeOverlay;
  const bool wireframeOnlyRequested =
      debugVisualization == OpaqueDebugVisualization::WireframeOnly;
  const bool patchHeatmapRequested =
      debugVisualization == OpaqueDebugVisualization::TessPatchEdgesHeatmap;
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
    uint64_t vertexBufferAddress = 0;
    uint64_t vertexDecodeBufferAddress = 0;
    uint32_t vertexDecodeIndex = 0;
    uint32_t packedVertexFormat = 0;
    uint32_t materialIndex = kInvalidMaterialIndex;
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
      [&baseDraw,
       &batches](RenderPipelineHandle pipeline, BufferHandle indexBuffer,
                 uint64_t indexBufferOffset, const SubmeshLod &lodRange,
                 BufferHandle vertexBuffer, uint64_t vertexBufferAddress,
                 uint64_t vertexDecodeBufferAddress, uint32_t vertexDecodeIndex,
                 uint32_t packedVertexFormat, uint32_t materialIndex,
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
        entry.vertexBufferAddress = vertexBufferAddress;
        entry.vertexDecodeBufferAddress = vertexDecodeBufferAddress;
        entry.vertexDecodeIndex = vertexDecodeIndex;
        entry.packedVertexFormat = packedVertexFormat;
        entry.materialIndex = materialIndex;
        entry.alphaMasked = alphaMasked;
        entry.instanceCount = count;
        entry.firstInstance = firstInstance;
        batches.push_back(entry);
      };
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
  const bool useAutoLod =
      settings.opaque.enableMeshLod && settings.opaque.forcedMeshLod < 0;
  const bool canUseUniformAutoLodFastPath =
      uniformSingleSubmeshPath_ && !meshDrawTemplates_.empty() && useAutoLod &&
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
      !settings.opaque.enableInstanceAnimation && !useAutoLod &&
      !tessellationRequested && !meshDrawTemplates_.empty() &&
      !uniformSingleSubmeshPath_ && instanceCount > 1;
  const bool canReuseStaticBatchCache =
      canUseStaticBatchCache && staticBatchCache_.valid &&
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
        drawItems_.size() != batchCount;
    if (needsStaticBatchRebind) {
      drawItems_ = staticBatchCache_.draws;
      drawPushConstants_ = staticBatchCache_.pushConstantsTemplates;
      drawAlphaMasked_ = staticBatchCache_.alphaMasked;
      if (drawAlphaMasked_.size() != batchCount) {
        drawAlphaMasked_.assign(batchCount, 0u);
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
          templateEntry.vertexBuffer, templateEntry.vertexBufferAddress,
          templateEntry.vertexDecodeBufferAddress,
          templateEntry.vertexDecodeIndex, templateEntry.packedVertexFormat,
          templateEntry.materialIndex, templateEntry.alphaMasked,
          autoLodBucketCounts[0], firstInstance);
      firstInstance += autoLodBucketCounts[0];

      autoLodTessBucketStart = firstInstance;
      autoLodTessBucketWrite = firstInstance;
      appendBatch(
          selectMeshPipeline(templateEntry.doubleSided, true),
          templateEntry.indexBuffer, templateEntry.indexBufferOffset, lod0Range,
          templateEntry.vertexBuffer, templateEntry.vertexBufferAddress,
          templateEntry.vertexDecodeBufferAddress,
          templateEntry.vertexDecodeIndex, templateEntry.packedVertexFormat,
          templateEntry.materialIndex, templateEntry.alphaMasked,
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

        appendBatch(selectMeshPipeline(templateEntry.doubleSided, false),
                    templateEntry.indexBuffer, templateEntry.indexBufferOffset,
                    lodRange, templateEntry.vertexBuffer,
                    templateEntry.vertexBufferAddress,
                    templateEntry.vertexDecodeBufferAddress,
                    templateEntry.vertexDecodeIndex,
                    templateEntry.packedVertexFormat,
                    templateEntry.materialIndex, templateEntry.alphaMasked,
                    count, firstInstance);
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

        appendBatch(selectMeshPipeline(templateEntry.doubleSided, false),
                    templateEntry.indexBuffer, templateEntry.indexBufferOffset,
                    lodRange, templateEntry.vertexBuffer,
                    templateEntry.vertexBufferAddress,
                    templateEntry.vertexDecodeBufferAddress,
                    templateEntry.vertexDecodeIndex,
                    templateEntry.packedVertexFormat,
                    templateEntry.materialIndex, templateEntry.alphaMasked,
                    count, firstInstance);
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
  if (!reusedStaticBatchCache && !usedUniformFastPath &&
      isSingleRenderableInstance && !meshDrawTemplates_.empty() &&
      !uniformSingleSubmeshPath_) {
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

  if (!reusedStaticBatchCache && !usedUniformFastPath &&
      uniformSingleSubmeshPath_ && !tessellationRequested &&
      !meshDrawTemplates_.empty() && !useAutoLod &&
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
                  lodRange, templateEntry.vertexBuffer,
                  templateEntry.vertexBufferAddress,
                  templateEntry.vertexDecodeBufferAddress,
                  templateEntry.vertexDecodeIndex,
                  templateEntry.packedVertexFormat, templateEntry.materialIndex,
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

      uint32_t requestedLod = 0;
      if (!settings.opaque.enableMeshLod) {
        requestedLod = 0;
      } else if (settings.opaque.forcedMeshLod >= 0) {
        requestedLod = forcedLod;
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
        entry.vertexBufferAddress = templateEntry.vertexBufferAddress;
        entry.vertexDecodeBufferAddress =
            templateEntry.vertexDecodeBufferAddress;
        entry.vertexDecodeIndex = templateEntry.vertexDecodeIndex;
        entry.packedVertexFormat = templateEntry.packedVertexFormat;
        entry.materialIndex = templateEntry.materialIndex;
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
      expandedPushConstants.reserve(expandedDrawCount);
      expandedDrawItems.reserve(expandedDrawCount);
      expandedAlphaMasked.reserve(expandedDrawCount);

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
        }
      }

      drawPushConstants_ = std::move(expandedPushConstants);
      drawItems_ = std::move(expandedDrawItems);
      drawAlphaMasked_ = std::move(expandedAlphaMasked);
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

  const bool normalPrepassRequested =
      ambientOcclusionSettings.active && !msaaSelected &&
      nuri::isValid(frame.sharedResources.normalTexture) &&
      !wireframeOnlyRequested && !baseDrawItems.empty();
  bool depthPrepassEnabled =
      (settings.opaque.enableDepthPrepass || normalPrepassRequested) &&
      !wireframeOnlyRequested && !baseDrawItems.empty();
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
  bool normalPrepassEnabled = normalPrepassRequested && depthPrepassEnabled &&
                              !shadedBaseDrawItems.empty();
  if (normalPrepassEnabled) {
    normalPrepassDrawItems_.reserve(shadedBaseDrawItems.size());
    for (const DrawItem &source : shadedBaseDrawItems) {
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
      normalDraw.depthState = {.compareOp = CompareOp::LessEqual,
                               .isDepthWriteEnabled = false};
      normalDraw.debugLabel = "OpaqueMaterialNormals";
      normalDraw.debugColor = 0xff66ddff;
      normalPrepassDrawItems_.push_back(normalDraw);
    }
  }
  std::span<const DrawItem> finalPassDrawItems = shadedBaseDrawItems;
  if (wireframeOnlyRequested && !baseDrawItems.empty()) {
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
    if (overlayRequested && !baseDrawItems.empty()) {
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
  frame.metrics.opaque.depthPrepassDraws =
      saturateToU32(depthPrepassDrawItems_.size());
  frame.metrics.opaque.depthPrepassIndirectDraws =
      depthPrepassEnabled && hasIndirectBaseDraws
          ? saturateToU32(depthPrepassDrawItems_.size())
          : 0u;
  frame.metrics.opaque.depthPyramidLevels = 0u;
  frame.metrics.opaque.depthPrepassEnabled = depthPrepassEnabled ? 1u : 0u;
  aoMetrics.normalPrepassDraws = saturateToU32(normalPrepassDrawItems_.size());

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
    if (!pickPassSubmitted && !depthPrepassEnabled) {
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
        !pickPassSubmitted && !depthPrepassEnabled && !preDispatches_.empty();
    visibilityDepthPass.desc.borrowPayload =
        !visibilityDepthPass.hasPreDispatch;
    visibilityDepthPass.hasIndirectDraws = hasIndirectBaseDraws;
    visibilityDepthPass.isTransmissionVisibilityDepthPass = true;
  }

  const bool preDispatchSubmittedBeforeMain =
      pickPassSubmitted || depthPrepassEnabled ||
      transmissionVisibilityDepthEnabled;

  if (normalPrepassEnabled && !normalPrepassDrawItems_.empty() &&
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
    normalPass.hasPreDispatch = false;
    normalPass.desc.borrowPayload = true;
    normalPass.hasIndirectDraws = hasIndirectBaseDraws;
    normalPass.isNormalPrepass = true;
    aoMetrics.normalPrepassDraws =
        saturateToU32(normalPrepassDrawItems_.size());
  } else if (ambientOcclusionSettings.active) {
    aoMetrics.active = false;
    aoMetrics.disabledReason = AmbientOcclusionDisabledReason::MissingResources;
  }

  const bool requiresDepthPyramid = this->requiresDepthPyramid(settings);
  const bool depthPyramidEnabled = !msaaSelected && requiresDepthPyramid &&
                                   nuri::isValid(sceneDepthTexture) &&
                                   nuri::isValid(depthPyramidPipelineHandle_) &&
                                   sceneDepthPyramidLevelCount_ > 0u &&
                                   !baseDrawItems.empty();
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
    if (pyramidBuildComplete) {
      sceneDepthPyramidSourceFrameIndex_ = frame.frameIndex;
    } else {
      sceneDepthPyramidSourceFrameIndex_.reset();
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
                         depthPrepassEnabled ? LoadOp::Load : LoadOp::Clear,
                     .storeOp = StoreOp::Store,
                     .clearDepth = kClearDepthOne,
                     .clearStencil = 0};
  pass.depthTextureHandle = sceneDepthTarget;
  if (!preDispatchSubmittedBeforeMain) {
    pass.desc.preDispatches = std::span<const ComputeDispatchItem>(
        preDispatches_.data(), preDispatches_.size());
  }
  mainPassDependencyBuffers_ = passDependencyBuffers_;
  mainPassDependencyBufferAccessModes_ = passDependencyBufferAccessModes_;
  mainPassDependencyTextures_ = passDependencyTextures_;
  mainPassDependencyTextureAccessModes_.assign(
      mainPassDependencyTextures_.size(), RenderGraphAccessMode::Read);
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
  pass.desc.drawBuffersPreResolved = true;
  pass.desc.preResolvedDrawBuffers = std::span<const BufferHandle>(
      preResolvedDrawBuffers_.data(), preResolvedDrawBuffers_.size());
  pass.desc.debugLabel = kOpaqueMainPassLabel;
  pass.desc.debugColor = kOpaquePassDebugColor;
  pass.desc.gpuTimingScope = GpuTimingScope::Opaque;
  pass.hasDraws = !finalPassDrawItems.empty();
  pass.hasPreDispatch =
      !preDispatchSubmittedBeforeMain && !preDispatches_.empty();
  pass.desc.borrowPayload = !pass.hasPreDispatch;
  pass.hasIndirectDraws = hasIndirectBaseDraws;
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

  if (taaSelected && nuri::isValid(frame.sharedResources.reactiveMaskTexture) &&
      nuri::isValid(meshReactiveMaskPipelineHandle_) &&
      baseAlphaMasked.size() == shadedBaseDrawItems.size()) {
    reactiveMaskDrawItems_.clear();
    reactiveMaskDrawItems_.reserve(shadedBaseDrawItems.size());
    const bool motionUncertainReactiveMode =
        hasTaaVelocityInstances &&
        velocityInstanceFlagsMode != VelocityInstanceFlagsMode::AllValid;
    uint32_t alphaMaskedCoverageDraws = 0u;
    uint32_t reactiveAlphaMaskedDraws = 0u;
    uint32_t motionUncertainDraws = 0u;
    uint32_t skippedTessellatedReactiveDraws = 0u;
    for (size_t i = 0; i < shadedBaseDrawItems.size(); ++i) {
      const bool alphaMasked = baseAlphaMasked[i] != 0u;
      alphaMaskedCoverageDraws += alphaMasked ? 1u : 0u;
      reactiveAlphaMaskedDraws += alphaMasked ? 1u : 0u;
      const bool motionUncertain = motionUncertainReactiveMode;
      if (!motionUncertain) {
        continue;
      }
      const DrawItem &sourceItem = shadedBaseDrawItems[i];
      if (isTessPipeline(sourceItem.pipeline)) {
        ++skippedTessellatedReactiveDraws;
        continue;
      }
      ++motionUncertainDraws;
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
      reactivePassDependencyBuffers_ = passDependencyBuffers_;
      reactivePassDependencyBufferAccessModes_ =
          passDependencyBufferAccessModes_;
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

  if (taaSelected && nuri::isValid(frame.sharedResources.motionVectorTexture) &&
      nuri::isValid(meshVelocityPipelineHandle_) && instanceCount > 0) {
    velocityDrawItems_.clear();
    velocityDrawItems_.reserve(shadedBaseDrawItems.size());
    uint32_t skippedTessellatedDraws = 0u;
    for (const DrawItem &sourceItem : shadedBaseDrawItems) {
      if (isTessPipeline(sourceItem.pipeline)) {
        ++skippedTessellatedDraws;
        frame.metrics.antiAliasing.opaqueVelocityGenerated = false;
        continue;
      }
      const RenderPipelineHandle velocityPipeline =
          selectVelocityPipeline(sourceItem.pipeline);
      if (!nuri::isValid(velocityPipeline)) {
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
      velocityPassDependencyBuffers_ = passDependencyBuffers_;
      velocityPassDependencyBufferAccessModes_ =
          passDependencyBufferAccessModes_;
      for (const BufferHandle handle :
           {previousInstanceMatricesBufferHandle,
            velocityInstanceFlagsBufferHandle, velocityFrameDataBufferHandle}) {
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
  NURI_PROFILER_ZONE_END();
  return Result<bool, std::string>::makeResult(true);
}

bool OpaqueRenderer::requiresDepthPyramid(
    const RenderSettings &settings) const {
  const ShadowSdsmMode sanitizedSdsm =
      sanitizeShadowSdsmMode(settings.shadow.sdsmMode);
  return settings.opaque.enableDepthPyramid ||
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
  if (settings.opaque.enableDepthPyramid) {
    return true;
  }
  if (settings.transmission.enabled || settings.transparent.enabled) {
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
         pass.isEarlyVelocityPass ||
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
    }
  }
  if (pass.isTransmissionVisibilityDepthPass &&
      nuri::isValid(pass.depthTextureHandle)) {
    frame.sharedResources.transmissionVisibilityDepthGraphTexture =
        passDesc.depthTexture;
  }
  if (pass.isMainPass && nuri::isValid(pass.depthResolveTextureHandle) &&
      isSameTextureHandle(pass.depthResolveTextureHandle,
                          frame.sharedResources.sceneDepthTexture)) {
    frame.sharedResources.sceneDepthGraphTexture = passDesc.depthResolveTexture;
    frame.metrics.antiAliasing.msaaDepthResolveTargetBound = true;
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
      }
    } else {
      frame.sharedResources.sceneColorGraphTexture = passDesc.colorTexture;
    }
  }
  if (pass.isVelocityPass && nuri::isValid(pass.colorTextureHandle)) {
    frame.sharedResources.motionVectorGraphTexture = passDesc.colorTexture;
    frame.metrics.antiAliasing.motionVectorGraphPublished = true;
    frame.metrics.antiAliasing.motionVectorClearPassCount = 0u;
    frame.metrics.antiAliasing.motionVectorClearBytes = 0u;
  }
  if (pass.isReactiveMaskPass && nuri::isValid(pass.colorTextureHandle)) {
    frame.sharedResources.reactiveMaskGraphTexture = passDesc.colorTexture;
    frame.metrics.antiAliasing.reactiveMaskGraphPublished = true;
  }
  if (pass.isNormalPrepass && nuri::isValid(pass.colorTextureHandle)) {
    frame.sharedResources.normalGraphTexture = passDesc.colorTexture;
    frame.metrics.ambientOcclusion.normalGraphPublished = true;
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
    meshPickFragmentShader_ = {};
    meshShadowInspectFragmentShader_ = {};
    meshVelocityVertexShader_ = {};
    meshVelocityFragmentShader_ = {};
    meshReactiveMaskVertexShader_ = {};
    meshReactiveMaskFragmentShader_ = {};
    meshNormalFragmentShader_ = {};
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
    meshPickFragmentShader_ = {};
    meshShadowInspectFragmentShader_ = {};
    meshVelocityVertexShader_ = {};
    meshVelocityFragmentShader_ = {};
    meshReactiveMaskVertexShader_ = {};
    meshReactiveMaskFragmentShader_ = {};
    meshNormalFragmentShader_ = {};
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
      instanceRemapRing_.size() == requiredCount &&
      indirectCommandRing_.size() == requiredCount) {
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

  instanceMatricesRing_.clear();
  previousInstanceMatricesRing_.clear();
  velocityInstanceFlagsRing_.clear();
  velocityFrameDataRing_.clear();
  instanceRemapRing_.clear();
  indirectCommandRing_.clear();
  instanceMatricesRing_.resize(requiredCount);
  previousInstanceMatricesRing_.resize(requiredCount);
  velocityInstanceFlagsRing_.resize(requiredCount);
  velocityFrameDataRing_.resize(requiredCount);
  instanceRemapRing_.resize(requiredCount);
  indirectCommandRing_.resize(requiredCount);
  instanceMatricesUploadVersions_.assign(requiredCount,
                                         std::numeric_limits<uint64_t>::max());
  indirectUploadSignatures_.assign(requiredCount, kInvalidDrawSignature);
  remapUploadSignatures_.assign(requiredCount, kInvalidDrawSignature);
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

Result<bool, std::string>
OpaqueRenderer::rebuildSceneCache(const RenderScene &scene,
                                  const ResourceManager &resources,
                                  uint32_t materialCount) {
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
           isTransmissionMaterial(*materialRecord))) {
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
    const RenderScene &scene, const ResourceManager &resources) {
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
          isTransmissionMaterial(*materialRecord)) {
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
  meshPickFragmentShader_ = {};
  meshShadowInspectFragmentShader_ = {};
  meshVelocityVertexShader_ = {};
  meshVelocityFragmentShader_ = {};
  meshReactiveMaskVertexShader_ = {};
  meshReactiveMaskFragmentShader_ = {};
  meshNormalFragmentShader_ = {};
  depthFragmentShader_ = {};
  depthAlphaFragmentShader_ = {};
  depthPyramidVertexShader_ = {};
  depthPyramidFragmentShader_ = {};
  computeShaderHandle_ = {};
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

  {
    const std::string shaderPath = config_.pickFragment.string();
    auto compileResult =
        meshPickShader_->compileFromFile(shaderPath, ShaderStage::Fragment);
    if (compileResult.hasError()) {
      return Result<bool, std::string>::makeError(compileResult.error());
    }
    meshPickFragmentShader_ = compileResult.value();
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
    if (vertexResult.hasError() || fragmentResult.hasError()) {
      const std::string error = vertexResult.hasError()
                                    ? vertexResult.error()
                                    : fragmentResult.error();
      NURI_LOG_WARNING(
          "OpaqueRenderer::createShaders: velocity shaders failed, opaque "
          "velocity generation will be disabled: %s",
          error.c_str());
      meshVelocityVertexShader_ = {};
      meshVelocityFragmentShader_ = {};
    } else {
      meshVelocityVertexShader_ = vertexResult.value();
      meshVelocityFragmentShader_ = fragmentResult.value();
    }
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
  if (nuri::isValid(depthFragmentShader_)) {
    const RenderPipelineDesc depthDesc =
        depthPipelineDesc(depthFormat, meshVertexShader_, {}, {},
                          depthFragmentShader_, CullMode::Back);
    createDepthPipeline(depthDesc, "opaque_mesh_depth",
                        meshDepthPipelineHandle_);
    createMsaaDepthPipeline(depthDesc, "opaque_mesh_depth_msaa4x",
                            meshMsaaDepthPipelineHandle_);
    const RenderPipelineDesc doubleSidedDepthDesc =
        depthPipelineDesc(depthFormat, meshVertexShader_, {}, {},
                          depthFragmentShader_, CullMode::None);
    createDepthPipeline(doubleSidedDepthDesc, "opaque_mesh_depth_double_sided",
                        meshDepthDoubleSidedPipelineHandle_);
    createMsaaDepthPipeline(doubleSidedDepthDesc,
                            "opaque_mesh_depth_double_sided_msaa4x",
                            meshMsaaDepthDoubleSidedPipelineHandle_);
  }
  if (nuri::isValid(depthAlphaFragmentShader_)) {
    const RenderPipelineDesc depthAlphaDesc =
        depthPipelineDesc(depthFormat, meshVertexShader_, {}, {},
                          depthAlphaFragmentShader_, CullMode::Back);
    createDepthPipeline(depthAlphaDesc, "opaque_mesh_depth_alpha",
                        meshDepthAlphaPipelineHandle_);
    createMsaaDepthPipeline(depthAlphaDesc, "opaque_mesh_depth_alpha_msaa4x",
                            meshMsaaDepthAlphaPipelineHandle_);
    const RenderPipelineDesc doubleSidedDepthAlphaDesc =
        depthPipelineDesc(depthFormat, meshVertexShader_, {}, {},
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
      if (nuri::isValid(depthFragmentShader_)) {
        const RenderPipelineDesc depthTessDesc = depthPipelineDesc(
            depthFormat, meshTessVertexShader_, meshTessControlShader_,
            meshTessEvalShader_, depthFragmentShader_, CullMode::Back,
            Topology::Patch, kTessellationPatchControlPoints);
        createDepthPipeline(depthTessDesc, "opaque_mesh_depth_tess",
                            meshDepthTessPipelineHandle_);
        createMsaaDepthPipeline(depthTessDesc, "opaque_mesh_depth_tess_msaa4x",
                                meshMsaaDepthTessPipelineHandle_);
        const RenderPipelineDesc depthDoubleSidedTessDesc = depthPipelineDesc(
            depthFormat, meshTessVertexShader_, meshTessControlShader_,
            meshTessEvalShader_, depthFragmentShader_, CullMode::None,
            Topology::Patch, kTessellationPatchControlPoints);
        createDepthPipeline(depthDoubleSidedTessDesc,
                            "opaque_mesh_depth_tess_double_sided",
                            meshDepthDoubleSidedTessPipelineHandle_);
        createMsaaDepthPipeline(depthDoubleSidedTessDesc,
                                "opaque_mesh_depth_tess_double_sided_msaa4x",
                                meshMsaaDepthDoubleSidedTessPipelineHandle_);
      }
      if (nuri::isValid(depthAlphaFragmentShader_)) {
        const RenderPipelineDesc depthAlphaTessDesc = depthPipelineDesc(
            depthFormat, meshTessVertexShader_, meshTessControlShader_,
            meshTessEvalShader_, depthAlphaFragmentShader_, CullMode::Back,
            Topology::Patch, kTessellationPatchControlPoints);
        createDepthPipeline(depthAlphaTessDesc, "opaque_mesh_depth_alpha_tess",
                            meshDepthAlphaTessPipelineHandle_);
        createMsaaDepthPipeline(depthAlphaTessDesc,
                                "opaque_mesh_depth_alpha_tess_msaa4x",
                                meshMsaaDepthAlphaTessPipelineHandle_);
        const RenderPipelineDesc depthAlphaDoubleSidedTessDesc =
            depthPipelineDesc(depthFormat, meshTessVertexShader_,
                              meshTessControlShader_, meshTessEvalShader_,
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
    }
  } else {
    tessellationUnsupported_ = true;
  }

  {
    const RenderPipelineDesc pickDesc =
        meshPipelineDesc(Format::R32_UINT, depthFormat, meshVertexShader_, {},
                         {}, {}, meshPickFragmentShader_, PolygonMode::Fill);
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
        meshPipelineDesc(Format::R32_UINT, depthFormat, meshVertexShader_, {},
                         {}, {}, meshPickFragmentShader_, PolygonMode::Fill,
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

  if (canCreateTessPipeline) {
    const RenderPipelineDesc pickTessDesc =
        meshPipelineDesc(Format::R32_UINT, depthFormat, meshTessVertexShader_,
                         meshTessControlShader_, meshTessEvalShader_, {},
                         meshPickFragmentShader_, PolygonMode::Fill,
                         Topology::Patch, kTessellationPatchControlPoints);
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
          Format::R32_UINT, depthFormat, meshTessVertexShader_,
          meshTessControlShader_, meshTessEvalShader_, {},
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
      destroyMeshPipelineState();
      return Result<bool, std::string>::makeError(
          inspectPipelineResult.error());
    }
    meshShadowInspectPipelineHandle_ = inspectPipelineResult.value();
  }

  {
    const RenderPipelineDesc doubleSidedInspectDesc = meshPipelineDesc(
        Format::RGBA32_FLOAT, depthFormat, meshVertexShader_, {}, {}, {},
        meshShadowInspectFragmentShader_, PolygonMode::Fill, Topology::Triangle,
        0, false, CullMode::None);
    auto doubleSidedInspectResult = gpu_.createRenderPipeline(
        doubleSidedInspectDesc, "opaque_mesh_shadow_inspect_double_sided");
    if (doubleSidedInspectResult.hasError()) {
      destroyMeshPipelineState();
      return Result<bool, std::string>::makeError(
          doubleSidedInspectResult.error());
    }
    meshShadowInspectDoubleSidedPipelineHandle_ =
        doubleSidedInspectResult.value();
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
  const bool doubleSided = isDoubleSidedPipeline(sourcePipeline);
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
  staticBatchCache_.enableMeshLod = false;
  staticBatchCache_.forcedMeshLod = -1;
  ++staticBatchCache_.generation;
  if (staticBatchCache_.generation == 0) {
    staticBatchCache_.generation = 1;
  }
  staticBatchCache_.remapSignature = kInvalidDrawSignature;
  staticBatchCache_.indirectDrawSignature = kInvalidDrawSignature;
  staticBatchCache_.drawBufferSignature = kInvalidDrawSignature;
  staticBatchCache_.draws.clear();
  staticBatchCache_.pushConstantsTemplates.clear();
  staticBatchCache_.alphaMasked.clear();
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
  instanceMatricesRing_.clear();
  previousInstanceMatricesRing_.clear();
  velocityInstanceFlagsRing_.clear();
  velocityFrameDataRing_.clear();
  instanceRemapRing_.clear();
  indirectCommandRing_.clear();
  instanceMatricesUploadVersions_.clear();
  remapUploadSignatures_.clear();
  indirectUploadSignatures_.clear();
  invalidateIndirectPackCache();
}

} // namespace nuri
