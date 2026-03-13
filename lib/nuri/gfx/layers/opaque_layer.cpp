#include "nuri/pch.h"

#include "nuri/gfx/layers/opaque_layer.h"

#include "nuri/core/containers/hash_map.h"
#include "nuri/core/containers/hash_set.h"
#include "nuri/core/log.h"
#include "nuri/core/pmr_scratch.h"
#include "nuri/core/profiling.h"
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

std::pmr::memory_resource *
resolveMemoryResource(std::pmr::memory_resource *memory) {
  return memory != nullptr ? memory : std::pmr::get_default_resource();
}

const RenderSettings &settingsOrDefault(const RenderFrameContext &frame) {
  static const RenderSettings kDefaultSettings{};
  return frame.settings ? *frame.settings : kDefaultSettings;
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
  if (dependencies.size() >= kMaxDependencyBuffers) {
    return Result<bool, std::string>::makeError(
        std::string(context) + ": dependency buffer count exceeds " +
        std::to_string(kMaxDependencyBuffers));
  }
  dependencies.push_back(handle);
  return Result<bool, std::string>::makeResult(true);
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
  uint32_t materialIndex = kInvalidMaterialIndex;

  bool operator==(const BatchKey &other) const {
    return isSamePipelineHandle(pipeline, other.pipeline) &&
           isSameBufferHandle(indexBuffer, other.indexBuffer) &&
           indexBufferOffset == other.indexBufferOffset &&
           indexCount == other.indexCount && firstIndex == other.firstIndex &&
           vertexBufferAddress == other.vertexBufferAddress &&
           materialIndex == other.materialIndex;
  }
};

struct BatchKeyHash {
  size_t operator()(const BatchKey &key) const noexcept {
    size_t h = kFnvOffsetBasis64;
    const auto mix = [&h](uint64_t v) {
      h ^= static_cast<size_t>(v);
      h *= 1099511628211ull;
    };
    mix((static_cast<uint64_t>(key.pipeline.generation) << 32u) |
        key.pipeline.index);
    mix((static_cast<uint64_t>(key.indexBuffer.generation) << 32u) |
        key.indexBuffer.index);
    mix(key.indexBufferOffset);
    mix((static_cast<uint64_t>(key.indexCount) << 32u) | key.firstIndex);
    mix(key.vertexBufferAddress);
    mix(static_cast<uint64_t>(key.materialIndex));
    return h;
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
  uint32_t materialIndex = kInvalidMaterialIndex;

  bool operator==(const IndirectGroupKey &other) const {
    return isSamePipelineHandle(pipeline, other.pipeline) &&
           isSameBufferHandle(indexBuffer, other.indexBuffer) &&
           indexBufferOffset == other.indexBufferOffset &&
           indexFormat == other.indexFormat &&
           vertexBufferAddress == other.vertexBufferAddress &&
           materialIndex == other.materialIndex;
  }
};

struct IndirectGroupKeyHash {
  size_t operator()(const IndirectGroupKey &key) const noexcept {
    size_t h = kFnvOffsetBasis64;
    const auto mix = [&h](uint64_t v) {
      h ^= static_cast<size_t>(v);
      h *= 1099511628211ull;
    };
    mix((static_cast<uint64_t>(key.pipeline.generation) << 32u) |
        key.pipeline.index);
    mix((static_cast<uint64_t>(key.indexBuffer.generation) << 32u) |
        key.indexBuffer.index);
    mix(key.indexBufferOffset);
    mix(static_cast<uint64_t>(key.indexFormat));
    mix(key.vertexBufferAddress);
    mix(static_cast<uint64_t>(key.materialIndex));
    return h;
  }
};

} // namespace

OpaqueLayer::OpaqueLayer(GPUDevice &gpu, OpaqueLayerConfig config,
                         std::pmr::memory_resource *memory)
    : gpu_(gpu), config_(std::move(config)),
      instanceMatricesRing_(resolveMemoryResource(memory)),
      instanceRemapRing_(resolveMemoryResource(memory)),
      indirectCommandRing_(resolveMemoryResource(memory)),
      singleInstanceBatchCaches_(resolveMemoryResource(memory)),
      renderableTemplates_(resolveMemoryResource(memory)),
      meshDrawTemplates_(resolveMemoryResource(memory)),
      indirectSourceDrawIndices_(resolveMemoryResource(memory)),
      indirectUploadSignatures_(resolveMemoryResource(memory)),
      remapUploadSignatures_(resolveMemoryResource(memory)),
      templateBatchIndices_(resolveMemoryResource(memory)),
      batchWriteOffsets_(resolveMemoryResource(memory)),
      instanceCentersPhase_(resolveMemoryResource(memory)),
      instanceBaseMatrices_(resolveMemoryResource(memory)),
      instanceLodCentersInvRadiusSq_(resolveMemoryResource(memory)),
      materialGpuDataCache_(resolveMemoryResource(memory)),
      materialTextureAccessHandles_(resolveMemoryResource(memory)),
      instanceAutoLodLevels_(resolveMemoryResource(memory)),
      instanceTessSelection_(resolveMemoryResource(memory)),
      tessCandidates_(resolveMemoryResource(memory)),
      instanceRemap_(resolveMemoryResource(memory)),
      drawPushConstants_(resolveMemoryResource(memory)),
      drawItems_(resolveMemoryResource(memory)),
      indirectDrawItems_(resolveMemoryResource(memory)),
      indirectCommandUploadBytes_(resolveMemoryResource(memory)),
      overlayDrawItems_(resolveMemoryResource(memory)),
      pickDrawItems_(resolveMemoryResource(memory)),
      passDrawItems_(resolveMemoryResource(memory)),
      preDispatches_(resolveMemoryResource(memory)),
      passDependencyBuffers_(resolveMemoryResource(memory)),
      dispatchDependencyBuffers_(resolveMemoryResource(memory)) {
  auto *resource = resolveMemoryResource(memory);
  singleInstanceBatchCaches_.reserve(kSingleInstanceCacheVariantCount);
  for (size_t i = 0; i < kSingleInstanceCacheVariantCount; ++i) {
    singleInstanceBatchCaches_.emplace_back(resource);
  }
}

OpaqueLayer::~OpaqueLayer() { onDetach(); }

void OpaqueLayer::onAttach() {
  auto initResult = ensureInitialized();
  if (initResult.hasError()) {
    NURI_LOG_WARNING("OpaqueLayer::onAttach: %s", initResult.error().c_str());
  }
}

void OpaqueLayer::resetPickState() {
  pendingPickRequest_.reset();
  inFlightPickReadback_.reset();
}

void OpaqueLayer::onDetach() {
  destroyBuffers();
  destroyDepthTexture();
  destroyPickTexture();
  resetOverlayPipelineState();
  destroyMeshPipelineState();
  meshPipeline_.reset();
  computePipeline_.reset();
  meshShader_.reset();
  meshTessShader_.reset();
  meshDebugOverlayShader_.reset();
  meshPickShader_.reset();
  computeShader_.reset();
  meshVertexShader_ = {};
  meshTessVertexShader_ = {};
  meshTessControlShader_ = {};
  meshTessEvalShader_ = {};
  meshFragmentShader_ = {};
  meshDebugOverlayGeometryShader_ = {};
  meshDebugOverlayFragmentShader_ = {};
  meshPickFragmentShader_ = {};
  computeShaderHandle_ = {};
  computePipelineHandle_ = {};
  tessellationUnsupported_ = false;
  renderableTemplates_.clear();
  meshDrawTemplates_.clear();
  templateBatchIndices_.clear();
  batchWriteOffsets_.clear();
  instanceLodCentersInvRadiusSq_.clear();
  materialGpuDataCache_.clear();
  materialTextureAccessHandles_.clear();
  instanceAutoLodLevels_.clear();
  instanceTessSelection_.clear();
  tessCandidates_.clear();
  instanceRemap_.clear();
  drawPushConstants_.clear();
  drawItems_.clear();
  indirectUploadSignatures_.clear();
  indirectDrawItems_.clear();
  indirectCommandUploadBytes_.clear();
  overlayDrawItems_.clear();
  passDrawItems_.clear();
  preDispatches_.clear();
  passDependencyBuffers_.clear();
  dispatchDependencyBuffers_.clear();
  pickDrawItems_.clear();
  cachedScene_ = nullptr;
  cachedTopologyVersion_ = std::numeric_limits<uint64_t>::max();
  cachedTransformVersion_ = std::numeric_limits<uint64_t>::max();
  cachedMaterialVersion_ = std::numeric_limits<uint64_t>::max();
  cachedGeometryMutationVersion_ = std::numeric_limits<uint64_t>::max();
  instanceStaticBuffersDirty_ = true;
  uniformSingleSubmeshPath_ = false;
  invalidateAutoLodCache();
  invalidateSingleInstanceBatchCache();
  invalidateIndirectPackCache();
  cachedRemapSignature_ = kInvalidDrawSignature;
  cachedRemapSignatureValid_ = false;
  resetPickState();
  initialized_ = false;
}

void OpaqueLayer::onResize(int32_t, int32_t) {
  destroyDepthTexture();
  destroyPickTexture();
  resetPickState();
}

Result<bool, std::string>
OpaqueLayer::buildOpaquePasses(RenderFrameContext &frame,
                               std::pmr::vector<PreparedGraphPass> &out) {
  NURI_PROFILER_FUNCTION();
  frame.metrics.opaque = {};
  frame.opaquePickResult.reset();
  if (frame.opaquePickRequest.has_value()) {
    pendingPickRequest_ = frame.opaquePickRequest;
    frame.opaquePickRequest.reset();
  }

  const RenderSettings &settings = settingsOrDefault(frame);
  if (!settings.opaque.enabled) {
    return Result<bool, std::string>::makeResult(true);
  }

  if (!frame.scene) {
    return Result<bool, std::string>::makeError(
        "OpaqueLayer::buildOpaquePasses: frame scene is null");
  }
  if (!frame.resources) {
    return Result<bool, std::string>::makeError(
        "OpaqueLayer::buildOpaquePasses: frame resources are null");
  }
  const MaterialTableSnapshot materialSnapshot =
      frame.resources->materialSnapshot();

  auto initResult = ensureInitialized();
  if (initResult.hasError()) {
    return Result<bool, std::string>::makeError(initResult.error());
  }
  if (!nuri::isValid(depthTexture_)) {
    auto depthResult = recreateDepthTexture();
    if (depthResult.hasError()) {
      return depthResult;
    }
  }
  if (!nuri::isValid(pickIdTexture_)) {
    auto pickTextureResult = recreatePickTexture();
    if (pickTextureResult.hasError()) {
      return pickTextureResult;
    }
  }
  if (inFlightPickReadback_.has_value() &&
      frame.frameIndex > inFlightPickReadback_->submissionFrame) {
    NURI_PROFILER_ZONE("OpaqueLayer.pick_readback",
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
          "OpaqueLayer::buildOpaquePasses: pick readback failed: %s",
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
        static_cast<uint32_t>(materialSnapshot.gpuData.size()));
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

  if (topologyDirty || transformDirty) {
    instanceCentersPhase_.clear();
    instanceBaseMatrices_.clear();
    instanceLodCentersInvRadiusSq_.clear();
    instanceCentersPhase_.reserve(instanceCount);
    instanceBaseMatrices_.reserve(instanceCount);
    instanceLodCentersInvRadiusSq_.reserve(instanceCount);

    const bool animateInstances = settings.opaque.enableInstanceAnimation;
    for (size_t i = 0; i < instanceCount; ++i) {
      const RenderableTemplate &templ = renderableTemplates_[i];
      const Renderable *renderable = templ.renderable;
      const Model *model = templ.model;
      if (!renderable || !model) {
        return Result<bool, std::string>::makeError(
            "OpaqueLayer::buildOpaquePasses: invalid opaque renderable");
      }

      const glm::vec3 center = glm::vec3(renderable->modelMatrix[3]);
      instanceCentersPhase_.push_back(
          glm::vec4(center, animateInstances
                                ? deterministicPhase(static_cast<uint32_t>(i))
                                : 0.0f));
      glm::mat4 baseMatrix = renderable->modelMatrix;
      baseMatrix[3] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
      instanceBaseMatrices_.push_back(baseMatrix);

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
  }

  uint32_t cubemapTexId = kInvalidTextureBindlessIndex;
  const uint32_t cubemapSamplerId = gpu_.getCubemapSamplerBindlessIndex();
  uint32_t hasCubemap = 0;
  const EnvironmentHandles &environment = frame.scene->environment();
  if (const TextureRecord *cubemap =
          frame.resources->tryGet(environment.cubemap);
      cubemap != nullptr && nuri::isValid(cubemap->texture)) {
    cubemapTexId = cubemap->bindlessIndex;
    hasCubemap = 1;
  }

  uint32_t irradianceTexId = kInvalidTextureBindlessIndex;
  uint32_t prefilteredGgxTexId = kInvalidTextureBindlessIndex;
  uint32_t prefilteredCharlieTexId = kInvalidTextureBindlessIndex;
  uint32_t brdfLutTexId = kInvalidTextureBindlessIndex;
  uint32_t frameFlags = 0;

  if (const TextureRecord *irradiance =
          frame.resources->tryGet(environment.irradiance);
      irradiance != nullptr && nuri::isValid(irradiance->texture)) {
    irradianceTexId = irradiance->bindlessIndex;
    frameFlags |= FrameDataFlags::HasIblDiffuse;
  } else if (hasCubemap != 0u) {
    irradianceTexId = cubemapTexId;
    frameFlags |= FrameDataFlags::HasIblDiffuse;
  }

  if (const TextureRecord *prefilteredGgx =
          frame.resources->tryGet(environment.prefilteredGgx);
      prefilteredGgx != nullptr && nuri::isValid(prefilteredGgx->texture)) {
    prefilteredGgxTexId = prefilteredGgx->bindlessIndex;
    frameFlags |= FrameDataFlags::HasIblSpecular;
  } else if (hasCubemap != 0u) {
    prefilteredGgxTexId = cubemapTexId;
    frameFlags |= FrameDataFlags::HasIblSpecular;
  }

  if (const TextureRecord *prefilteredCharlie =
          frame.resources->tryGet(environment.prefilteredCharlie);
      prefilteredCharlie != nullptr &&
      nuri::isValid(prefilteredCharlie->texture)) {
    prefilteredCharlieTexId = prefilteredCharlie->bindlessIndex;
    frameFlags |= FrameDataFlags::HasIblSheen;
  } else if ((frameFlags & FrameDataFlags::HasIblSpecular) != 0u) {
    prefilteredCharlieTexId = prefilteredGgxTexId;
    frameFlags |= FrameDataFlags::HasIblSheen;
  }

  if (const TextureRecord *brdfLut =
          frame.resources->tryGet(environment.brdfLut);
      brdfLut != nullptr && nuri::isValid(brdfLut->texture)) {
    brdfLutTexId = brdfLut->bindlessIndex;
    frameFlags |= FrameDataFlags::HasBrdfLut;
  }
  if (gpu_.getSwapchainFormat() != Format::RGBA8_SRGB) {
    frameFlags |= FrameDataFlags::OutputLinearToSrgb;
  }

  frameData_ = FrameData{
      .view = frame.camera.view,
      .proj = frame.camera.proj,
      .cameraPos = frame.camera.cameraPos,
      .cubemapTexId = cubemapTexId,
      .hasCubemap = hasCubemap,
      .irradianceTexId = irradianceTexId,
      .prefilteredGgxTexId = prefilteredGgxTexId,
      .prefilteredCharlieTexId = prefilteredCharlieTexId,
      .brdfLutTexId = brdfLutTexId,
      .flags = frameFlags,
      .cubemapSamplerId = cubemapSamplerId,
  };

  auto frameDataResult = ensureFrameDataBufferCapacity(sizeof(FrameData));
  if (frameDataResult.hasError()) {
    return frameDataResult;
  }
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
      std::max(instanceCount * sizeof(glm::mat4), sizeof(glm::mat4)));
  if (matricesResult.hasError()) {
    return matricesResult;
  }
  const size_t sceneMaterialCount =
      std::max<size_t>(materialSnapshot.gpuData.size(), 1u);
  auto materialBufferResult = ensureMaterialBufferCapacity(
      sceneMaterialCount * sizeof(MaterialGpuData));
  if (materialBufferResult.hasError()) {
    return materialBufferResult;
  }

  {
    const bool shouldUploadFrameData =
        !frameDataUploadValid_ ||
        std::memcmp(&uploadedFrameData_, &frameData_, sizeof(FrameData)) != 0;
    if (shouldUploadFrameData) {
      const std::span<const std::byte> frameDataBytes{
          reinterpret_cast<const std::byte *>(&frameData_), sizeof(frameData_)};
      auto updateResult =
          gpu_.updateBuffer(frameDataBuffer_->handle(), frameDataBytes, 0);
      if (updateResult.hasError()) {
        return updateResult;
      }
      uploadedFrameData_ = frameData_;
      frameDataUploadValid_ = true;
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

  if (materialDirty || materialGpuDataCache_.empty()) {
    materialGpuDataCache_.clear();
    materialGpuDataCache_.reserve(materialSnapshot.gpuData.size());
    materialGpuDataCache_.insert(materialGpuDataCache_.end(),
                                 materialSnapshot.gpuData.begin(),
                                 materialSnapshot.gpuData.end());
    if (materialGpuDataCache_.empty()) {
      materialGpuDataCache_.push_back(MaterialGpuData{});
    }

    const std::span<const std::byte> materialBytes{
        reinterpret_cast<const std::byte *>(materialGpuDataCache_.data()),
        materialGpuDataCache_.size() * sizeof(MaterialGpuData)};
    auto updateResult =
        gpu_.updateBuffer(materialBuffer_->handle(), materialBytes, 0);
    if (updateResult.hasError()) {
      return updateResult;
    }
    cachedMaterialVersion_ = materialSnapshot.version;
  }
  if (materialDirty || materialTextureAccessHandles_.empty()) {
    auto materialAccessCacheResult =
        rebuildMaterialTextureAccessCache(*frame.scene, *frame.resources);
    if (materialAccessCacheResult.hasError()) {
      return materialAccessCacheResult;
    }
  }

  const uint64_t frameDataAddress =
      gpu_.getBufferDeviceAddress(frameDataBuffer_->handle());
  const uint64_t instanceCentersPhaseAddress =
      gpu_.getBufferDeviceAddress(instanceCentersPhaseBuffer_->handle());
  const uint64_t instanceBaseMatricesAddress =
      gpu_.getBufferDeviceAddress(instanceBaseMatricesBuffer_->handle());
  const uint64_t materialBufferAddress =
      gpu_.getBufferDeviceAddress(materialBuffer_->handle());
  const uint64_t instanceMatricesAddress = gpu_.getBufferDeviceAddress(
      instanceMatricesRing_[frameSlot].buffer->handle());
  if (frameDataAddress == 0 || instanceCentersPhaseAddress == 0 ||
      instanceBaseMatricesAddress == 0 || materialBufferAddress == 0 ||
      instanceMatricesAddress == 0) {
    return Result<bool, std::string>::makeError(
        "OpaqueLayer::buildOpaquePasses: invalid GPU buffer address");
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
    uint64_t vertexBufferAddress = 0;
    uint32_t materialIndex = kInvalidMaterialIndex;
    size_t instanceCount = 0;
    size_t firstInstance = 0;
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
                 uint64_t vertexBufferAddress, uint32_t materialIndex,
                 size_t count, size_t firstInstance) {
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
        entry.vertexBufferAddress = vertexBufferAddress;
        entry.materialIndex = materialIndex;
        entry.instanceCount = count;
        entry.firstInstance = firstInstance;
        batches.push_back(entry);
      };
  std::array<float, 3> sortedLodThresholds = {
      settings.opaque.meshLodDistanceThresholds.x,
      settings.opaque.meshLodDistanceThresholds.y,
      settings.opaque.meshLodDistanceThresholds.z,
  };
  std::sort(sortedLodThresholds.begin(), sortedLodThresholds.end());
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
    NURI_PROFILER_ZONE("OpaqueLayer.auto_lod_resolve",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    if (instanceLodCentersInvRadiusSq_.size() != renderableTemplates_.size()) {
      return Result<bool, std::string>::makeError(
          "OpaqueLayer::buildOpaquePasses: LOD cache size mismatch");
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
          "OpaqueLayer::buildOpaquePasses: failed to refresh geometry");
    }
    if (!nuri::isValid(geometry.vertexBuffer) ||
        !nuri::isValid(geometry.indexBuffer) ||
        !gpu_.isValid(geometry.vertexBuffer) ||
        !gpu_.isValid(geometry.indexBuffer)) {
      return Result<bool, std::string>::makeError(
          "OpaqueLayer::buildOpaquePasses: refreshed geometry is invalid");
    }
    const uint64_t refreshedVertexAddress = gpu_.getBufferDeviceAddress(
        geometry.vertexBuffer, geometry.vertexByteOffset);
    if (refreshedVertexAddress == 0) {
      return Result<bool, std::string>::makeError(
          "OpaqueLayer::buildOpaquePasses: refreshed vertex address is "
          "invalid");
    }
    templateEntry.indexBuffer = geometry.indexBuffer;
    templateEntry.indexBufferOffset = geometry.indexByteOffset;
    templateEntry.vertexBufferAddress = refreshedVertexAddress;
    return Result<bool, std::string>::makeResult(true);
  };
  const bool shouldRefreshTemplateGeometry =
      !meshDrawTemplates_.empty() &&
      (!hasGeometryMutationTracking ||
       cachedGeometryMutationVersion_ != geometryMutationVersion);
  if (shouldRefreshTemplateGeometry) {
    bool templateGeometryChanged = false;
    {
      NURI_PROFILER_ZONE("OpaqueLayer.sync_template_geometry",
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
            templateEntry.vertexBufferAddress =
                firstTemplate.vertexBufferAddress;
          }
        }
      } else {
        GeometryAllocationHandle cachedHandle{};
        BufferHandle cachedIndexBuffer{};
        uint64_t cachedIndexBufferOffset = 0;
        uint64_t cachedVertexBufferAddress = 0;
        bool hasCachedGeometry = false;

        for (MeshDrawTemplate &templateEntry : meshDrawTemplates_) {
          const BufferHandle previousIndexBuffer = templateEntry.indexBuffer;
          const uint64_t previousIndexBufferOffset =
              templateEntry.indexBufferOffset;
          const uint64_t previousVertexBufferAddress =
              templateEntry.vertexBufferAddress;

          const bool sameAsCached =
              hasCachedGeometry &&
              isSameGeometryAllocationHandle(templateEntry.geometryHandle,
                                             cachedHandle);
          if (sameAsCached) {
            templateEntry.indexBuffer = cachedIndexBuffer;
            templateEntry.indexBufferOffset = cachedIndexBufferOffset;
            templateEntry.vertexBufferAddress = cachedVertexBufferAddress;
          } else {
            auto refreshResult = refreshTemplateGeometry(templateEntry);
            if (refreshResult.hasError()) {
              return refreshResult;
            }

            cachedHandle = templateEntry.geometryHandle;
            cachedIndexBuffer = templateEntry.indexBuffer;
            cachedIndexBufferOffset = templateEntry.indexBufferOffset;
            cachedVertexBufferAddress = templateEntry.vertexBufferAddress;
            hasCachedGeometry = true;
          }

          const bool geometryChanged =
              !isSameBufferHandle(templateEntry.indexBuffer,
                                  previousIndexBuffer) ||
              templateEntry.indexBufferOffset != previousIndexBufferOffset ||
              templateEntry.vertexBufferAddress != previousVertexBufferAddress;
          templateGeometryChanged = templateGeometryChanged || geometryChanged;
        }
      }
      NURI_PROFILER_ZONE_END();
    }
    if (templateGeometryChanged) {
      invalidateSingleInstanceBatchCache();
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
  const SingleInstanceBatchCache *activeSingleInstanceCache = nullptr;

  if (canUseUniformAutoLodFastPath) {
    NURI_PROFILER_ZONE("OpaqueLayer.batch_build_auto_lod",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    MeshDrawTemplate &templateEntry = meshDrawTemplates_.front();
    if (!templateEntry.submesh) {
      return Result<bool, std::string>::makeError(
          "OpaqueLayer::buildOpaquePasses: invalid auto-LOD submesh");
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
            "OpaqueLayer::buildOpaquePasses: auto-LOD cache size "
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
      appendBatch(selectMeshPipeline(templateEntry.doubleSided, false),
                  templateEntry.indexBuffer, templateEntry.indexBufferOffset,
                  lod0Range, templateEntry.vertexBufferAddress,
                  templateEntry.materialIndex, autoLodBucketCounts[0],
                  firstInstance);
      firstInstance += autoLodBucketCounts[0];

      autoLodTessBucketStart = firstInstance;
      autoLodTessBucketWrite = firstInstance;
      appendBatch(selectMeshPipeline(templateEntry.doubleSided, true),
                  templateEntry.indexBuffer, templateEntry.indexBufferOffset,
                  lod0Range, templateEntry.vertexBufferAddress,
                  templateEntry.materialIndex, autoLodTessBucketCount,
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

        appendBatch(selectMeshPipeline(templateEntry.doubleSided, false),
                    templateEntry.indexBuffer, templateEntry.indexBufferOffset,
                    lodRange, templateEntry.vertexBufferAddress,
                    templateEntry.materialIndex, count, firstInstance);
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
                    lodRange, templateEntry.vertexBufferAddress,
                    templateEntry.materialIndex, count, firstInstance);
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
  if (!usedUniformFastPath && isSingleRenderableInstance &&
      !meshDrawTemplates_.empty() && !uniformSingleSubmeshPath_) {
    NURI_PROFILER_ZONE("OpaqueLayer.batch_build_single_instance_cache",
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
          "OpaqueLayer::buildOpaquePasses: single-instance cache index out of "
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

  if (!usedUniformFastPath && uniformSingleSubmeshPath_ &&
      !tessellationRequested && !meshDrawTemplates_.empty() && !useAutoLod &&
      instanceCount == meshDrawTemplates_.size()) {
    NURI_PROFILER_ZONE("OpaqueLayer.batch_build_fast",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    MeshDrawTemplate &templateEntry = meshDrawTemplates_.front();
    if (!templateEntry.submesh) {
      return Result<bool, std::string>::makeError(
          "OpaqueLayer::buildOpaquePasses: invalid fast-path submesh");
    }
    const uint32_t requestedLod =
        settings.opaque.enableMeshLod ? forcedLod : 0u;
    const auto lodIndex =
        resolveAvailableLod(*templateEntry.submesh, requestedLod);
    if (lodIndex) {
      const SubmeshLod &lodRange = templateEntry.submesh->lods[*lodIndex];
      appendBatch(selectMeshPipeline(templateEntry.doubleSided, false),
                  templateEntry.indexBuffer, templateEntry.indexBufferOffset,
                  lodRange, templateEntry.vertexBufferAddress,
                  templateEntry.materialIndex, instanceCount, 0);
      remapCount = instanceCount;
      templateBatchIndices_.clear();
      templateBatchIndices_.resize(meshDrawTemplates_.size(), 0u);
      usedUniformFastPath = true;
    }
    NURI_PROFILER_ZONE_END();
  }

  if (!usedUniformFastPath) {
    PmrHashMap<BatchKey, size_t, BatchKeyHash> batchLookup(
        batchScratch.resource());
    batchLookup.reserve(batchReserve);
    templateBatchIndices_.clear();
    templateBatchIndices_.resize(meshDrawTemplates_.size(), kInvalidBatchIndex);
    NURI_PROFILER_ZONE("OpaqueLayer.batch_build", NURI_PROFILER_COLOR_CMD_DRAW);
    for (size_t templateIndex = 0; templateIndex < meshDrawTemplates_.size();
         ++templateIndex) {
      MeshDrawTemplate &templateEntry = meshDrawTemplates_[templateIndex];
      if (!templateEntry.renderable || !templateEntry.submesh) {
        return Result<bool, std::string>::makeError(
            "OpaqueLayer::buildOpaquePasses: invalid mesh template");
      }

      uint32_t requestedLod = 0;
      if (!settings.opaque.enableMeshLod) {
        requestedLod = 0;
      } else if (settings.opaque.forcedMeshLod >= 0) {
        requestedLod = forcedLod;
      } else {
        if (templateEntry.instanceIndex >= instanceAutoLodLevels_.size()) {
          return Result<bool, std::string>::makeError(
              "OpaqueLayer::buildOpaquePasses: instance LOD cache out of "
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
        entry.vertexBufferAddress = templateEntry.vertexBufferAddress;
        entry.materialIndex = templateEntry.materialIndex;
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

  uint64_t remapSignature = kInvalidDrawSignature;
  bool remapSignatureValid = false;
  uint64_t indirectDrawSignature = kInvalidDrawSignature;
  bool indirectDrawSignatureValid = false;
  {
    NURI_PROFILER_ZONE("OpaqueLayer.draw_list_emit",
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
            "OpaqueLayer::buildOpaquePasses: cached single-instance batches "
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
          "OpaqueLayer::buildOpaquePasses: auto-LOD remap reuse mismatch");
    }
    if (shouldBuildRemap) {
      remapSignature = hashCombine64(kFnvOffsetBasis64, remapCount);
      remapSignatureValid = true;
    }

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
    } else if (shouldBuildRemap && singleRenderableInstance) {
      for (size_t i = 0; i < instanceRemap_.size(); ++i) {
        instanceRemap_[i] = 0u;
        remapSignature = hashCombine64(remapSignature, 0u);
      }
    } else if (shouldBuildRemap && usedUniformFastPath) {
      for (uint32_t instanceId = 0; instanceId < instanceCount; ++instanceId) {
        instanceRemap_[instanceId] = instanceId;
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
        instanceRemap_[writeOffset] = instanceId;
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
    if (useCachedSingleInstanceBatches) {
      for (size_t batchIndex = 0;
           batchIndex < activeSingleInstanceCache->batches.size();
           ++batchIndex) {
        const SingleInstanceBatchEntry &batch =
            activeSingleInstanceCache->batches[batchIndex];
        PushConstants &constants = drawPushConstants_[batchIndex];
        constants.frameDataAddress = frameDataAddress;
        constants.vertexBufferAddress = batch.vertexBufferAddress;
        constants.instanceMatricesAddress = instanceMatricesAddress;
        constants.instanceRemapAddress = 0;
        constants.materialBufferAddress = materialBufferAddress;
        constants.instanceCentersPhaseAddress = instanceCentersPhaseAddress;
        constants.instanceBaseMatricesAddress = instanceBaseMatricesAddress;
        constants.instanceCount = static_cast<uint32_t>(instanceCount);
        constants.materialIndex = batch.materialIndex;
        constants.timeSeconds = settings.opaque.enableInstanceAnimation
                                    ? static_cast<float>(frame.timeSeconds)
                                    : 0.0f;
        constants.tessNearDistance = tessNearDistance;
        constants.tessFarDistance = tessFarDistance;
        constants.tessMinFactor = tessMinFactor;
        constants.tessMaxFactor = tessMaxFactor;
        constants.debugVisualizationMode = debugVisualizationMode;

        DrawItem &draw = drawItems_[batchIndex];
        draw = batch.draw;
        draw.instanceCount = static_cast<uint32_t>(batch.instanceCount);
        draw.firstInstance = 0;
        draw.pushConstants = std::span<const std::byte>(
            reinterpret_cast<const std::byte *>(&constants),
            sizeof(PushConstants));

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
        constants.instanceMatricesAddress = instanceMatricesAddress;
        constants.instanceRemapAddress = 0;
        constants.materialBufferAddress = materialBufferAddress;
        constants.instanceCentersPhaseAddress = instanceCentersPhaseAddress;
        constants.instanceBaseMatricesAddress = instanceBaseMatricesAddress;
        constants.instanceCount = static_cast<uint32_t>(instanceCount);
        constants.materialIndex = batch.materialIndex;
        constants.timeSeconds = settings.opaque.enableInstanceAnimation
                                    ? static_cast<float>(frame.timeSeconds)
                                    : 0.0f;
        constants.tessNearDistance = tessNearDistance;
        constants.tessFarDistance = tessFarDistance;
        constants.tessMinFactor = tessMinFactor;
        constants.tessMaxFactor = tessMaxFactor;
        constants.debugVisualizationMode = debugVisualizationMode;

        DrawItem &draw = drawItems_[batchIndex];
        draw = batch.draw;
        draw.instanceCount = static_cast<uint32_t>(batch.instanceCount);
        draw.firstInstance = static_cast<uint32_t>(batch.firstInstance);
        draw.pushConstants = std::span<const std::byte>(
            reinterpret_cast<const std::byte *>(&constants),
            sizeof(PushConstants));

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
  }

  if (!settings.opaque.enableInstancedDraw && !drawItems_.empty()) {
    NURI_PROFILER_ZONE("OpaqueLayer.instancing_expand",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    if (drawItems_.size() != drawPushConstants_.size()) {
      return Result<bool, std::string>::makeError(
          "OpaqueLayer::buildOpaquePasses: draw and push constant count "
          "mismatch before instancing expansion");
    }

    size_t expandedDrawCount = 0;
    for (const DrawItem &draw : drawItems_) {
      expandedDrawCount += draw.instanceCount;
    }
    if (expandedDrawCount >
        static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
      return Result<bool, std::string>::makeError(
          "OpaqueLayer::buildOpaquePasses: expanded draw count exceeds "
          "UINT32_MAX");
    }

    if (expandedDrawCount != drawItems_.size()) {
      std::pmr::vector<PushConstants> expandedPushConstants(
          drawPushConstants_.get_allocator().resource());
      std::pmr::vector<DrawItem> expandedDrawItems(
          drawItems_.get_allocator().resource());
      expandedPushConstants.reserve(expandedDrawCount);
      expandedDrawItems.reserve(expandedDrawCount);

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
              "OpaqueLayer::buildOpaquePasses: expanded instance range "
              "overflows UINT32_MAX");
        }

        for (uint32_t instanceOffset = 0;
             instanceOffset < sourceDraw.instanceCount; ++instanceOffset) {
          expandedPushConstants.push_back(sourceConstants);

          DrawItem expandedDraw = sourceDraw;
          expandedDraw.instanceCount = 1;
          expandedDraw.firstInstance =
              sourceDraw.firstInstance + instanceOffset;
          expandedDraw.pushConstants =
              std::span<const std::byte>(reinterpret_cast<const std::byte *>(
                                             &expandedPushConstants.back()),
                                         sizeof(PushConstants));
          expandedDrawItems.push_back(expandedDraw);
        }
      }

      drawPushConstants_ = std::move(expandedPushConstants);
      drawItems_ = std::move(expandedDrawItems);
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
        "OpaqueLayer::buildOpaquePasses: invalid instance remap buffer "
        "address");
  }

  if (!instanceRemap_.empty()) {
    if (!remapSignatureValid) {
      remapSignature = computeRemapSignature(std::span<const uint32_t>(
          instanceRemap_.data(), instanceRemap_.size()));
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
      NURI_PROFILER_ZONE("OpaqueLayer.remap_upload",
                         NURI_PROFILER_COLOR_CMD_COPY);
      const std::span<const std::byte> remapBytes{
          reinterpret_cast<const std::byte *>(instanceRemap_.data()),
          instanceRemap_.size() * sizeof(uint32_t)};
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
    indirectCommandUploadBytes_.clear();
  }

  computePushConstants_ = PushConstants{
      .frameDataAddress = frameDataAddress,
      .vertexBufferAddress = 0,
      .instanceMatricesAddress = instanceMatricesAddress,
      .instanceRemapAddress = instanceRemapAddress,
      .materialBufferAddress = materialBufferAddress,
      .instanceCentersPhaseAddress = instanceCentersPhaseAddress,
      .instanceBaseMatricesAddress = instanceBaseMatricesAddress,
      .instanceCount = static_cast<uint32_t>(instanceCount),
      .materialIndex = 0u,
      .timeSeconds = settings.opaque.enableInstanceAnimation
                         ? static_cast<float>(frame.timeSeconds)
                         : 0.0f,
      .tessNearDistance = tessNearDistance,
      .tessFarDistance = tessFarDistance,
      .tessMinFactor = tessMinFactor,
      .tessMaxFactor = tessMaxFactor,
      .debugVisualizationMode = debugVisualizationMode,
  };

  const bool useComputePass = settings.opaque.enableInstanceCompute;
  if (!useComputePass && instanceCount > 0) {
    NURI_PROFILER_ZONE("OpaqueLayer.instance_matrices_cpu",
                       NURI_PROFILER_COLOR_CMD_COPY);
    ScratchArena scratch;
    ScopedScratch scopedScratch(scratch);
    std::pmr::vector<glm::mat4> instanceMatrices(scopedScratch.resource());
    instanceMatrices.resize(instanceCount);

    for (size_t i = 0; i < instanceCount; ++i) {
      const glm::vec3 center = glm::vec3(instanceCentersPhase_[i]);
      const glm::mat4 translation = glm::translate(glm::mat4(1.0f), center);
      instanceMatrices[i] = translation * instanceBaseMatrices_[i];
    }

    const std::span<const std::byte> matricesBytes{
        reinterpret_cast<const std::byte *>(instanceMatrices.data()),
        instanceMatrices.size() * sizeof(glm::mat4)};
    auto updateResult = gpu_.updateBuffer(
        instanceMatricesRing_[frameSlot].buffer->handle(), matricesBytes, 0);
    if (updateResult.hasError()) {
      return updateResult;
    }
    NURI_PROFILER_ZONE_END();
  }

  uint32_t computeDispatchX = 0;
  {
    NURI_PROFILER_ZONE("OpaqueLayer.compute_dispatch_submission",
                       NURI_PROFILER_COLOR_CMD_DISPATCH);
    preDispatches_.clear();
    dispatchDependencyBuffers_.clear();
    passDependencyBuffers_.clear();

    const bool hasIndirectDraws = !indirectDrawItems_.empty();
    if (!hasIndirectDraws && frameDataBuffer_ && frameDataBuffer_->valid()) {
      auto depResult = appendUniqueDependency(
          passDependencyBuffers_, frameDataBuffer_->handle(),
          "OpaqueLayer::buildOpaquePasses(pass)");
      if (depResult.hasError()) {
        return depResult;
      }
    }
    if (materialBuffer_ && materialBuffer_->valid()) {
      auto depResult = appendUniqueDependency(
          passDependencyBuffers_, materialBuffer_->handle(),
          "OpaqueLayer::buildOpaquePasses(pass)");
      if (depResult.hasError()) {
        return depResult;
      }
    }
    if (frameSlot < instanceRemapRing_.size() &&
        instanceRemapRing_[frameSlot].buffer &&
        instanceRemapRing_[frameSlot].buffer->valid()) {
      auto depResult =
          appendUniqueDependency(passDependencyBuffers_,
                                 instanceRemapRing_[frameSlot].buffer->handle(),
                                 "OpaqueLayer::buildOpaquePasses(pass)");
      if (depResult.hasError()) {
        return depResult;
      }
    }

    if (hasIndirectDraws) {
      auto depResult = appendUniqueDependency(
          passDependencyBuffers_,
          indirectCommandRing_[frameSlot].buffer->handle(),
          "OpaqueLayer::buildOpaquePasses(pass)");
      if (depResult.hasError()) {
        return depResult;
      }
    }

    if (instanceCount > 0) {
      auto passDepResult = appendUniqueDependency(
          passDependencyBuffers_,
          instanceMatricesRing_[frameSlot].buffer->handle(),
          "OpaqueLayer::buildOpaquePasses(pass)");
      if (passDepResult.hasError()) {
        return passDepResult;
      }

      if (useComputePass) {
        if (instanceCentersPhaseBuffer_ &&
            instanceCentersPhaseBuffer_->valid()) {
          auto depResult = appendUniqueDependency(
              dispatchDependencyBuffers_, instanceCentersPhaseBuffer_->handle(),
              "OpaqueLayer::buildOpaquePasses(dispatch)");
          if (depResult.hasError()) {
            return depResult;
          }
        }
        if (instanceBaseMatricesBuffer_ &&
            instanceBaseMatricesBuffer_->valid()) {
          auto depResult = appendUniqueDependency(
              dispatchDependencyBuffers_, instanceBaseMatricesBuffer_->handle(),
              "OpaqueLayer::buildOpaquePasses(dispatch)");
          if (depResult.hasError()) {
            return depResult;
          }
        }
        auto dispatchDepResult = appendUniqueDependency(
            dispatchDependencyBuffers_,
            instanceMatricesRing_[frameSlot].buffer->handle(),
            "OpaqueLayer::buildOpaquePasses(dispatch)");
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

  size_t debugOverlayDraws = 0;
  size_t debugOverlayFallbackDraws = 0;
  size_t debugPatchHeatmapDraws = 0;
  overlayDrawItems_.clear();
  passDrawItems_.clear();
  std::span<const DrawItem> finalPassDrawItems = baseDrawItems;
  if (wireframeOnlyRequested && !baseDrawItems.empty()) {
    bool lineOverlayAvailable = false;
    bool lineTessOverlayAvailable = false;

    auto lineResult = ensureWireframePipeline();
    if (lineResult.hasError()) {
      if (!loggedWireframeFallbackUnsupported_) {
        loggedWireframeFallbackUnsupported_ = true;
        NURI_LOG_WARNING("OpaqueLayer::buildOpaquePasses: failed to create "
                         "wireframe pipeline: %s",
                         lineResult.error().c_str());
      }
    } else {
      lineOverlayAvailable = lineResult.value();
    }

    auto lineTessResult = ensureTessWireframePipeline();
    if (lineTessResult.hasError()) {
      if (!loggedTessWireframeFallbackUnsupported_) {
        loggedTessWireframeFallbackUnsupported_ = true;
        NURI_LOG_WARNING("OpaqueLayer::buildOpaquePasses: failed to create "
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
        wireframePipeline = meshTessWireframePipelineHandle_;
      } else if (lineOverlayAvailable) {
        wireframePipeline = meshWireframePipelineHandle_;
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

      auto gsOverlayResult = ensureGsOverlayPipeline();
      if (gsOverlayResult.hasError()) {
        if (!loggedGsOverlayUnsupported_) {
          loggedGsOverlayUnsupported_ = true;
          NURI_LOG_WARNING("OpaqueLayer::buildOpaquePasses: failed to create "
                           "GS overlay pipeline: %s",
                           gsOverlayResult.error().c_str());
        }
      } else {
        gsOverlayAvailable = gsOverlayResult.value();
      }

      auto gsTessOverlayResult = ensureGsTessOverlayPipeline();
      if (gsTessOverlayResult.hasError()) {
        if (!loggedGsTessOverlayUnsupported_) {
          loggedGsTessOverlayUnsupported_ = true;
          NURI_LOG_WARNING("OpaqueLayer::buildOpaquePasses: failed to create "
                           "GS tess overlay pipeline: %s",
                           gsTessOverlayResult.error().c_str());
        }
      } else {
        gsTessOverlayAvailable = gsTessOverlayResult.value();
      }

      if (!gsOverlayAvailable) {
        auto lineResult = ensureWireframePipeline();
        if (lineResult.hasError()) {
          if (!loggedWireframeFallbackUnsupported_) {
            loggedWireframeFallbackUnsupported_ = true;
            NURI_LOG_WARNING("OpaqueLayer::buildOpaquePasses: failed to "
                             "create line overlay fallback pipeline: %s",
                             lineResult.error().c_str());
          }
        } else {
          lineOverlayAvailable = lineResult.value();
        }
      }
      if (!gsTessOverlayAvailable) {
        auto lineTessResult = ensureTessWireframePipeline();
        if (lineTessResult.hasError()) {
          if (!loggedTessWireframeFallbackUnsupported_) {
            loggedTessWireframeFallbackUnsupported_ = true;
            NURI_LOG_WARNING("OpaqueLayer::buildOpaquePasses: failed to "
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
            overlayPipeline = meshGsTessOverlayPipelineHandle_;
          } else if (lineTessOverlayAvailable) {
            overlayPipeline = meshTessWireframePipelineHandle_;
            usedFallback = true;
          }
        } else {
          if (gsOverlayAvailable) {
            overlayPipeline = meshGsOverlayPipelineHandle_;
          } else if (lineOverlayAvailable) {
            overlayPipeline = meshWireframePipelineHandle_;
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
      passDrawItems_.reserve(baseDrawItems.size() + overlayDrawItems_.size());
      passDrawItems_.insert(passDrawItems_.end(), baseDrawItems.begin(),
                            baseDrawItems.end());
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

  ++statsLogFrameCounter_;
  const bool shouldLogStats = (statsLogFrameCounter_ & 511ull) == 0ull;
  if (shouldLogStats) {
    // NURI_LOG_DEBUG("OpaqueLayer::buildOpaquePasses: totalInstances=%zu "
    //                "visibleInstances=%zu "
    //                "instancedDraws=%zu tessellatedDraws=%zu "
    //                "tessellatedInstances=%zu indirectDrawCalls=%zu "
    //                "indirectCommands=%zu "
    //                "overlayDraws=%zu "
    //                "overlayFallbackDraws=%zu",
    //                instanceCount, remapCount, drawItems_.size(),
    //                tessellatedDraws, tessellatedInstances,
    //                indirectDrawItems_.size(), indirectCommandCount,
    //                debugOverlayDraws, debugOverlayFallbackDraws);
  }

  bool pickPassSubmitted = false;
  if (pendingPickRequest_.has_value() && nuri::isValid(pickIdTexture_) &&
      nuri::isValid(meshPickPipelineHandle_)) {
    NURI_PROFILER_ZONE("OpaqueLayer.pick_pass", NURI_PROFILER_COLOR_CMD_DRAW);
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

    PreparedGraphPass pickPass{};
    pickPass.desc.color = {.loadOp = LoadOp::Clear,
                           .storeOp = StoreOp::Store,
                           .clearColor = {0.0f, 0.0f, 0.0f, 0.0f}};
    pickPass.colorTextureHandle = pickIdTexture_;
    pickPass.desc.depth = {.loadOp = LoadOp::Clear,
                           .storeOp = StoreOp::Store,
                           .clearDepth = kClearDepthOne,
                           .clearStencil = 0};
    pickPass.depthTextureHandle = depthTexture_;
    pickPass.desc.preDispatches = std::span<const ComputeDispatchItem>(
        preDispatches_.data(), preDispatches_.size());
    pickPass.desc.dependencyBuffers = std::span<const BufferHandle>(
        passDependencyBuffers_.data(), passDependencyBuffers_.size());
    pickPass.desc.draws =
        std::span<const DrawItem>(pickDrawItems_.data(), pickDrawItems_.size());
    pickPass.desc.debugLabel = kOpaquePickPassLabel;
    pickPass.desc.debugColor = kOpaquePassDebugColor;
    pickPass.hasDraws = !pickDrawItems_.empty();
    pickPass.hasPreDispatch = !preDispatches_.empty();
    pickPass.hasIndirectDraws = false;
    pickPass.isPickPass = true;
    out.push_back(pickPass);

    inFlightPickReadback_ = InFlightPickReadback{
        .request = *pendingPickRequest_, .submissionFrame = frame.frameIndex};
    pendingPickRequest_.reset();
    pickPassSubmitted = true;
    NURI_PROFILER_ZONE_END();
  }

  const bool shouldLoadColor = settings.skybox.enabled;
  PreparedGraphPass pass{};
  pass.desc.color = {.loadOp = shouldLoadColor ? LoadOp::Load : LoadOp::Clear,
                     .storeOp = StoreOp::Store,
                     .clearColor = {kClearColorWhite, kClearColorWhite,
                                    kClearColorWhite, kClearColorWhite}};
  pass.desc.depth = {.loadOp = LoadOp::Clear,
                     .storeOp = StoreOp::Store,
                     .clearDepth = kClearDepthOne,
                     .clearStencil = 0};
  pass.depthTextureHandle = depthTexture_;
  if (!pickPassSubmitted) {
    pass.desc.preDispatches = std::span<const ComputeDispatchItem>(
        preDispatches_.data(), preDispatches_.size());
  }
  pass.desc.dependencyBuffers = std::span<const BufferHandle>(
      passDependencyBuffers_.data(), passDependencyBuffers_.size());
  pass.desc.draws = finalPassDrawItems;
  pass.desc.debugLabel = kOpaqueMainPassLabel;
  pass.desc.debugColor = kOpaquePassDebugColor;
  pass.hasDraws = !finalPassDrawItems.empty();
  pass.hasPreDispatch = !pickPassSubmitted && !preDispatches_.empty();
  pass.hasIndirectDraws = hasIndirectBaseDraws;
  pass.isMainPass = true;

  frame.sharedDepthTexture = depthTexture_;
  frame.channels.publish<TextureHandle>(kFrameChannelSceneDepthTexture,
                                        depthTexture_);
  out.push_back(pass);
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
OpaqueLayer::buildRenderGraph(RenderFrameContext &frame,
                              RenderGraphBuilder &graph) {
  NURI_PROFILER_FUNCTION();

  std::pmr::memory_resource *const memory =
      renderableTemplates_.get_allocator().resource();
  std::pmr::vector<PreparedGraphPass> localPasses(memory);
  auto buildResult = buildOpaquePasses(frame, localPasses);
  if (buildResult.hasError()) {
    return buildResult;
  }

  std::pmr::vector<RenderGraphPassId> addedPassIds(memory);
  addedPassIds.reserve(localPasses.size());
  std::pmr::vector<RenderGraphPassId> opaqueShadingPassIds(memory);
  opaqueShadingPassIds.reserve(localPasses.size());
  std::pmr::vector<RenderGraphPassId> opaqueWorkPassIds(memory);
  opaqueWorkPassIds.reserve(localPasses.size());
  std::pmr::vector<RenderGraphPassId> opaquePreDispatchPassIds(memory);
  opaquePreDispatchPassIds.reserve(localPasses.size());
  std::pmr::vector<RenderGraphPassId> opaqueIndirectPassIds(memory);
  opaqueIndirectPassIds.reserve(localPasses.size());
  int32_t framebufferWidth = 0;
  int32_t framebufferHeight = 0;
  gpu_.getFramebufferSize(framebufferWidth, framebufferHeight);
  const uint32_t safeWidth =
      static_cast<uint32_t>(std::max(framebufferWidth, 1));
  const uint32_t safeHeight =
      static_cast<uint32_t>(std::max(framebufferHeight, 1));

  for (const PreparedGraphPass &pass : localPasses) {
    RenderGraphGraphicsPassDesc passDesc = pass.desc;

    if (nuri::isValid(pass.colorTextureHandle)) {
      auto colorImportResult = graph.importTexture(pass.colorTextureHandle,
                                                   "opaque_pass_color_texture");
      if (colorImportResult.hasError()) {
        return Result<bool, std::string>::makeError(colorImportResult.error());
      }
      passDesc.colorTexture = colorImportResult.value();
    }

    if (nuri::isValid(pass.depthTextureHandle)) {
      auto depthImportResult = graph.importTexture(pass.depthTextureHandle,
                                                   "opaque_pass_depth_texture");
      if (depthImportResult.hasError()) {
        return Result<bool, std::string>::makeError(depthImportResult.error());
      }
      passDesc.depthTexture = depthImportResult.value();
    }

    auto addResult = graph.addGraphicsPass(passDesc);
    if (addResult.hasError()) {
      return Result<bool, std::string>::makeError(addResult.error());
    }
    const RenderGraphPassId passId = addResult.value();
    addedPassIds.push_back(passId);

    const bool hasPreDispatch = pass.hasPreDispatch;
    const bool hasDraws = pass.hasDraws;
    const bool hasIndirectDraws = pass.hasIndirectDraws;
    if (hasDraws || hasPreDispatch) {
      opaqueWorkPassIds.push_back(passId);
    }
    if (hasPreDispatch) {
      opaquePreDispatchPassIds.push_back(passId);
    }
    if (hasPreDispatch || hasIndirectDraws) {
      opaqueIndirectPassIds.push_back(passId);
    }
    if (pass.isMainPass) {
      opaqueShadingPassIds.push_back(passId);
    }

    if (pass.isPickPass) {
      frame.channels.publish<RenderGraphTextureId>(
          kFrameChannelOpaquePickGraphTexture, passDesc.colorTexture);
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
        return Result<bool, std::string>::makeError(
            bindPickDepthResult.error());
      }
      frame.channels.publish<RenderGraphTextureId>(
          kFrameChannelOpaquePickDepthGraphTexture, pickDepthResult.value());
    }

    if (pass.isMainPass && nuri::isValid(pass.depthTextureHandle)) {
      const Format sceneDepthFormat =
          gpu_.getTextureFormat(pass.depthTextureHandle);
      const TextureDesc sceneDepthTransientDesc{
          .type = TextureType::Texture2D,
          .format = sceneDepthFormat,
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
      auto sceneDepthResult = graph.createTransientTexture(
          sceneDepthTransientDesc, "opaque_scene_transient_depth");
      if (sceneDepthResult.hasError()) {
        return Result<bool, std::string>::makeError(sceneDepthResult.error());
      }
      auto bindSceneDepthResult =
          graph.bindPassDepthTexture(passId, sceneDepthResult.value());
      if (bindSceneDepthResult.hasError()) {
        return Result<bool, std::string>::makeError(
            bindSceneDepthResult.error());
      }
      frame.channels.publish<RenderGraphTextureId>(
          kFrameChannelSceneDepthGraphTexture, sceneDepthResult.value());
    }
  }

  if (addedPassIds.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }

  const auto registerBufferAccessForPasses =
      [&graph](std::span<const RenderGraphPassId> passIds, BufferHandle handle,
               RenderGraphAccessMode mode,
               std::string_view debugName) -> Result<bool, std::string> {
    if (passIds.empty()) {
      return Result<bool, std::string>::makeResult(true);
    }
    if (!nuri::isValid(handle)) {
      return Result<bool, std::string>::makeResult(true);
    }

    auto importResult = graph.importBuffer(handle, debugName);
    if (importResult.hasError()) {
      return Result<bool, std::string>::makeError(importResult.error());
    }

    for (const RenderGraphPassId passId : passIds) {
      auto accessResult =
          graph.addBufferAccess(passId, importResult.value(), mode);
      if (accessResult.hasError()) {
        return Result<bool, std::string>::makeError(accessResult.error());
      }
    }
    return Result<bool, std::string>::makeResult(true);
  };
  const auto registerTextureAccessForPasses =
      [&graph](std::span<const RenderGraphPassId> passIds, TextureHandle handle,
               RenderGraphAccessMode mode,
               std::string_view debugName) -> Result<bool, std::string> {
    if (passIds.empty()) {
      return Result<bool, std::string>::makeResult(true);
    }
    if (!nuri::isValid(handle)) {
      return Result<bool, std::string>::makeResult(true);
    }

    auto importResult = graph.importTexture(handle, debugName);
    if (importResult.hasError()) {
      return Result<bool, std::string>::makeError(importResult.error());
    }

    for (const RenderGraphPassId passId : passIds) {
      auto accessResult =
          graph.addTextureAccess(passId, importResult.value(), mode);
      if (accessResult.hasError()) {
        return Result<bool, std::string>::makeError(accessResult.error());
      }
    }
    return Result<bool, std::string>::makeResult(true);
  };

  constexpr RenderGraphAccessMode kReadOnly = RenderGraphAccessMode::Read;
  constexpr RenderGraphAccessMode kReadWrite =
      RenderGraphAccessMode::Read | RenderGraphAccessMode::Write;
  auto registerResult = registerBufferAccessForPasses(
      opaqueWorkPassIds,
      frameDataBuffer_ && frameDataBuffer_->valid() ? frameDataBuffer_->handle()
                                                    : BufferHandle{},
      kReadOnly, "opaque_frame_data_buffer");
  if (registerResult.hasError()) {
    return registerResult;
  }
  registerResult = registerBufferAccessForPasses(
      opaqueWorkPassIds,
      instanceCentersPhaseBuffer_ && instanceCentersPhaseBuffer_->valid()
          ? instanceCentersPhaseBuffer_->handle()
          : BufferHandle{},
      kReadOnly, "opaque_instance_centers_phase_buffer");
  if (registerResult.hasError()) {
    return registerResult;
  }
  registerResult = registerBufferAccessForPasses(
      opaqueWorkPassIds,
      instanceBaseMatricesBuffer_ && instanceBaseMatricesBuffer_->valid()
          ? instanceBaseMatricesBuffer_->handle()
          : BufferHandle{},
      kReadOnly, "opaque_instance_base_matrices_buffer");
  if (registerResult.hasError()) {
    return registerResult;
  }
  registerResult = registerBufferAccessForPasses(
      opaqueShadingPassIds,
      materialBuffer_ && materialBuffer_->valid() ? materialBuffer_->handle()
                                                  : BufferHandle{},
      kReadOnly, "opaque_material_buffer");
  if (registerResult.hasError()) {
    return registerResult;
  }

  const uint32_t swapchainImageCount =
      std::max(1u, gpu_.getSwapchainImageCount());
  const uint32_t frameSlot =
      static_cast<uint32_t>(frame.frameIndex % swapchainImageCount);
  if (frameSlot < instanceMatricesRing_.size() &&
      instanceMatricesRing_[frameSlot].buffer &&
      instanceMatricesRing_[frameSlot].buffer->valid()) {
    auto registerResult = registerBufferAccessForPasses(
        opaqueWorkPassIds, instanceMatricesRing_[frameSlot].buffer->handle(),
        kReadWrite, "opaque_instance_matrices_ring_buffer");
    if (registerResult.hasError()) {
      return registerResult;
    }
  }
  if (frameSlot < instanceRemapRing_.size() &&
      instanceRemapRing_[frameSlot].buffer &&
      instanceRemapRing_[frameSlot].buffer->valid()) {
    auto registerResult = registerBufferAccessForPasses(
        opaquePreDispatchPassIds,
        instanceRemapRing_[frameSlot].buffer->handle(), kReadWrite,
        "opaque_instance_remap_ring_buffer");
    if (registerResult.hasError()) {
      return registerResult;
    }
  }
  if (frameSlot < indirectCommandRing_.size() &&
      indirectCommandRing_[frameSlot].buffer &&
      indirectCommandRing_[frameSlot].buffer->valid()) {
    auto registerResult = registerBufferAccessForPasses(
        opaqueIndirectPassIds, indirectCommandRing_[frameSlot].buffer->handle(),
        kReadWrite, "opaque_indirect_command_ring_buffer");
    if (registerResult.hasError()) {
      return registerResult;
    }
  }

  if (frame.scene != nullptr && frame.resources != nullptr) {
    const EnvironmentHandles &environment = frame.scene->environment();
    const std::array<std::pair<TextureRef, std::string_view>, 5> envTextures = {
        {
            {environment.cubemap, "opaque_env_cubemap"},
            {environment.irradiance, "opaque_env_irradiance"},
            {environment.prefilteredGgx, "opaque_env_prefiltered_ggx"},
            {environment.prefilteredCharlie, "opaque_env_prefiltered_charlie"},
            {environment.brdfLut, "opaque_env_brdf_lut"},
        }};

    for (const auto &[ref, name] : envTextures) {
      const TextureRecord *record = frame.resources->tryGet(ref);
      if (record == nullptr || !nuri::isValid(record->texture)) {
        continue;
      }
      auto registerResult = registerTextureAccessForPasses(
          opaqueShadingPassIds, record->texture, kReadOnly, name);
      if (registerResult.hasError()) {
        return registerResult;
      }
    }
    for (const TextureHandle texture : materialTextureAccessHandles_) {
      auto registerResult = registerTextureAccessForPasses(
          opaqueShadingPassIds, texture, kReadOnly, "opaque_material_texture");
      if (registerResult.hasError()) {
        return registerResult;
      }
    }
  }

  return Result<bool, std::string>::makeResult(true);
}

uint32_t
OpaqueLayer::resolveSingleInstanceRequestedLod(const RenderSettings &settings,
                                               uint32_t forcedLod) const {
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

bool OpaqueLayer::shouldEnableSingleInstanceTessPipeline(
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

size_t OpaqueLayer::singleInstanceCacheIndex(uint32_t requestedLod,
                                             bool tessPipelineEnabled) const {
  const uint32_t clampedLod =
      std::min(requestedLod, Submesh::kMaxLodCount - 1u);
  return static_cast<size_t>(clampedLod) * 2u + (tessPipelineEnabled ? 1u : 0u);
}

Result<bool, std::string> OpaqueLayer::ensureSingleInstanceBatchCache(
    uint32_t requestedLod, bool tessPipelineEnabled, const DrawItem &baseDraw) {
  const size_t cacheIndex =
      singleInstanceCacheIndex(requestedLod, tessPipelineEnabled);
  if (cacheIndex >= singleInstanceBatchCaches_.size()) {
    return Result<bool, std::string>::makeError(
        "OpaqueLayer::buildOpaquePasses: single-instance cache index out of "
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
          "OpaqueLayer::buildOpaquePasses: invalid mesh template");
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
      entry.vertexBufferAddress = templateEntry.vertexBufferAddress;
      entry.materialIndex = templateEntry.materialIndex;
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
OpaqueLayer::buildIndirectDraws(uint32_t frameSlot, size_t remapCount,
                                uint64_t drawSignature,
                                bool drawSignatureValid) {
  const bool canUseIndirectPath =
      !drawItems_.empty() && drawItems_.size() <= kMaxDrawItemsForIndirectPath;
  if (!canUseIndirectPath) {
    invalidateIndirectPackCache();
    indirectDrawItems_.clear();
    indirectCommandUploadBytes_.clear();
    return Result<bool, std::string>::makeResult(true);
  }

  NURI_PROFILER_ZONE("OpaqueLayer.indirect_pack", NURI_PROFILER_COLOR_CMD_DRAW);
  if (drawItems_.size() != drawPushConstants_.size()) {
    return Result<bool, std::string>::makeError(
        "OpaqueLayer::buildIndirectDraws: draw and push constant count "
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
        "OpaqueLayer::buildIndirectDraws: indirect ring buffer slot is "
        "invalid");
  }

  NURI_PROFILER_ZONE_END();
  return Result<bool, std::string>::makeResult(true);
}

uint64_t OpaqueLayer::computeIndirectDrawSignature(size_t remapCount) const {
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

bool OpaqueLayer::canReuseIndirectPack(uint64_t drawSignature) const {
  return indirectPackCache_.valid &&
         indirectPackCache_.drawSignature == drawSignature &&
         indirectSourceDrawIndices_.size() == indirectDrawItems_.size() &&
         indirectPackCache_.requiredBytes >= kIndirectCountHeaderBytes &&
         indirectCommandUploadBytes_.size() <= indirectPackCache_.requiredBytes;
}

Result<bool, std::string>
OpaqueLayer::rebuildIndirectPack(uint32_t frameSlot, size_t remapCount,
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
          "OpaqueLayer::buildIndirectDraws: non-indexed opaque draws are not "
          "supported in indirect mode");
    }
    if (draw.instanceCount == 0) {
      continue;
    }
    if (draw.firstInstance > remapCount ||
        draw.instanceCount > remapCount - draw.firstInstance) {
      return Result<bool, std::string>::makeError(
          "OpaqueLayer::buildIndirectDraws: indirect draw instance range is "
          "out of remap bounds");
    }

    const PushConstants &constants = drawPushConstants_[i];
    const IndirectGroupKey key{
        .pipeline = draw.pipeline,
        .indexBuffer = draw.indexBuffer,
        .indexBufferOffset = draw.indexBufferOffset,
        .indexFormat = draw.indexFormat,
        .vertexBufferAddress = constants.vertexBufferAddress,
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
      indirectLookup.emplace(key, groupIndex);
      it = indirectLookup.find(key);
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
        "OpaqueLayer::buildIndirectDraws: indirect ring buffer slot is "
        "invalid");
  }

  indirectDrawItems_.clear();
  indirectCommandUploadBytes_.clear();
  indirectSourceDrawIndices_.clear();

  if (!indirectGroups.empty()) {
    indirectCommandUploadBytes_.reserve(packedRequiredBytes);
    indirectDrawItems_.reserve(totalIndirectDrawItems);
    indirectSourceDrawIndices_.reserve(totalIndirectDrawItems);

    const BufferHandle indirectBufferHandle =
        indirectCommandRing_[frameSlot].buffer->handle();

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
        indirectDraw.command = DrawCommandType::IndexedIndirect;
        indirectDraw.indirectBuffer = indirectBufferHandle;
        indirectDraw.indirectBufferOffset =
            chunkOffset + kIndirectCountHeaderBytes;
        indirectDraw.indirectDrawCount = drawCount;
        indirectDraw.indirectStride = sizeof(DrawIndexedIndirectCommand);
        if (group.sourceDrawIndex >= drawPushConstants_.size()) {
          return Result<bool, std::string>::makeError(
              "OpaqueLayer::buildIndirectDraws: indirect source index is out "
              "of range");
        }
        indirectDraw.pushConstants = std::span<const std::byte>(
            reinterpret_cast<const std::byte *>(
                &drawPushConstants_[group.sourceDrawIndex]),
            sizeof(PushConstants));
        indirectDrawItems_.push_back(indirectDraw);
        indirectSourceDrawIndices_.push_back(group.sourceDrawIndex);

        commandCursor += drawCount;
      }
    }
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
OpaqueLayer::refreshCachedIndirectPack(uint32_t frameSlot,
                                       uint64_t drawSignature) {
  if (indirectSourceDrawIndices_.size() != indirectDrawItems_.size()) {
    return Result<bool, std::string>::makeError(
        "OpaqueLayer::buildIndirectDraws: cached indirect source mapping is "
        "invalid");
  }
  const BufferHandle indirectBufferHandle =
      indirectCommandRing_[frameSlot].buffer->handle();
  for (size_t i = 0; i < indirectSourceDrawIndices_.size(); ++i) {
    const size_t sourceIndex = indirectSourceDrawIndices_[i];
    if (sourceIndex >= drawPushConstants_.size()) {
      return Result<bool, std::string>::makeError(
          "OpaqueLayer::buildIndirectDraws: cached indirect source index is "
          "out of range");
    }
    indirectDrawItems_[i].indirectBuffer = indirectBufferHandle;
    indirectDrawItems_[i].pushConstants = std::span<const std::byte>(
        reinterpret_cast<const std::byte *>(&drawPushConstants_[sourceIndex]),
        sizeof(PushConstants));
  }

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

Result<bool, std::string> OpaqueLayer::ensureInitialized() {
  if (initialized_) {
    return Result<bool, std::string>::makeResult(true);
  }

  auto shaderResult = createShaders();
  if (shaderResult.hasError()) {
    return shaderResult;
  }

  auto depthResult = recreateDepthTexture();
  if (depthResult.hasError()) {
    meshShader_.reset();
    meshTessShader_.reset();
    meshDebugOverlayShader_.reset();
    meshPickShader_.reset();
    computeShader_.reset();
    meshVertexShader_ = {};
    meshTessVertexShader_ = {};
    meshTessControlShader_ = {};
    meshTessEvalShader_ = {};
    meshFragmentShader_ = {};
    meshDebugOverlayGeometryShader_ = {};
    meshDebugOverlayFragmentShader_ = {};
    meshPickFragmentShader_ = {};
    computeShaderHandle_ = {};
    resetMeshPipelineState();
    tessellationUnsupported_ = false;
    return depthResult;
  }

  auto pickTextureResult = recreatePickTexture();
  if (pickTextureResult.hasError()) {
    meshShader_.reset();
    meshTessShader_.reset();
    meshDebugOverlayShader_.reset();
    meshPickShader_.reset();
    computeShader_.reset();
    meshVertexShader_ = {};
    meshTessVertexShader_ = {};
    meshTessControlShader_ = {};
    meshTessEvalShader_ = {};
    meshFragmentShader_ = {};
    meshDebugOverlayGeometryShader_ = {};
    meshDebugOverlayFragmentShader_ = {};
    meshPickFragmentShader_ = {};
    computeShaderHandle_ = {};
    resetMeshPipelineState();
    tessellationUnsupported_ = false;
    destroyDepthTexture();
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
    computeShader_.reset();
    meshVertexShader_ = {};
    meshTessVertexShader_ = {};
    meshTessControlShader_ = {};
    meshTessEvalShader_ = {};
    meshFragmentShader_ = {};
    meshDebugOverlayGeometryShader_ = {};
    meshDebugOverlayFragmentShader_ = {};
    meshPickFragmentShader_ = {};
    computeShaderHandle_ = {};
    computePipelineHandle_ = {};
    tessellationUnsupported_ = false;
    destroyDepthTexture();
    destroyPickTexture();
    return pipelineResult;
  }

  auto frameBufferResult = ensureFrameDataBufferCapacity(sizeof(FrameData));
  if (frameBufferResult.hasError()) {
    return frameBufferResult;
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
  auto materialResult = ensureMaterialBufferCapacity(sizeof(MaterialGpuData));
  if (materialResult.hasError()) {
    return materialResult;
  }

  initialized_ = true;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> OpaqueLayer::recreateDepthTexture() {
  if (nuri::isValid(depthTexture_)) {
    gpu_.destroyTexture(depthTexture_);
    depthTexture_ = TextureHandle{};
  }

  auto depthResult = gpu_.createDepthBuffer();
  if (depthResult.hasError()) {
    return Result<bool, std::string>::makeError(depthResult.error());
  }
  depthTexture_ = depthResult.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> OpaqueLayer::recreatePickTexture() {
  if (nuri::isValid(pickIdTexture_)) {
    gpu_.destroyTexture(pickIdTexture_);
    pickIdTexture_ = TextureHandle{};
  }

  const TextureDesc pickDesc{
      .type = TextureType::Texture2D,
      .format = Format::R32_UINT,
      .dimensions = {1, 1, 1},
      .usage = TextureUsage::Attachment,
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
OpaqueLayer::ensureFrameDataBufferCapacity(size_t requiredBytes) {
  const size_t requested = std::max(requiredBytes, sizeof(FrameData));
  if (frameDataBuffer_ && frameDataBuffer_->valid() &&
      frameDataBufferCapacityBytes_ >= requested) {
    return Result<bool, std::string>::makeResult(true);
  }

  if (frameDataBuffer_ && frameDataBuffer_->valid()) {
    gpu_.waitIdle();
    gpu_.destroyBuffer(frameDataBuffer_->handle());
    frameDataBuffer_.reset();
    frameDataBufferCapacityBytes_ = 0;
    frameDataUploadValid_ = false;
  }

  const BufferDesc desc{
      .usage = BufferUsage::Storage,
      .storage = Storage::Device,
      .size = requested,
  };
  auto createResult = Buffer::create(gpu_, desc, "opaque_frame_data_buffer");
  if (createResult.hasError()) {
    return Result<bool, std::string>::makeError(createResult.error());
  }
  frameDataBuffer_ = std::move(createResult.value());
  frameDataBufferCapacityBytes_ = requested;
  frameDataUploadValid_ = false;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
OpaqueLayer::ensureCentersPhaseBufferCapacity(size_t requiredBytes) {
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
OpaqueLayer::ensureInstanceBaseMatricesBufferCapacity(size_t requiredBytes) {
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
OpaqueLayer::ensureMaterialBufferCapacity(size_t requiredBytes) {
  const size_t requested = std::max(requiredBytes, sizeof(MaterialGpuData));
  if (materialBuffer_ && materialBuffer_->valid() &&
      materialBufferCapacityBytes_ >= requested) {
    return Result<bool, std::string>::makeResult(true);
  }

  if (materialBuffer_ && materialBuffer_->valid()) {
    gpu_.waitIdle();
    gpu_.destroyBuffer(materialBuffer_->handle());
    materialBuffer_.reset();
    materialBufferCapacityBytes_ = 0;
  }

  const BufferDesc desc{
      .usage = BufferUsage::Storage,
      .storage = Storage::Device,
      .size = requested,
  };
  auto createResult = Buffer::create(gpu_, desc, "opaque_material_buffer");
  if (createResult.hasError()) {
    return Result<bool, std::string>::makeError(createResult.error());
  }
  materialBuffer_ = std::move(createResult.value());
  materialBufferCapacityBytes_ = requested;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
OpaqueLayer::ensureRingBufferCount(uint32_t requiredCount) {
  if (requiredCount == 0) {
    requiredCount = 1;
  }
  if (instanceMatricesRing_.size() == requiredCount &&
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
  instanceRemapRing_.clear();
  indirectCommandRing_.clear();
  instanceMatricesRing_.resize(requiredCount);
  instanceRemapRing_.resize(requiredCount);
  indirectCommandRing_.resize(requiredCount);
  indirectUploadSignatures_.assign(requiredCount, kInvalidDrawSignature);
  remapUploadSignatures_.assign(requiredCount, kInvalidDrawSignature);
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
OpaqueLayer::ensureInstanceMatricesRingCapacity(size_t requiredBytes) {
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
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
OpaqueLayer::ensureInstanceRemapRingCapacity(size_t requiredBytes) {
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
OpaqueLayer::ensureIndirectCommandRingCapacity(size_t requiredBytes) {
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
OpaqueLayer::rebuildSceneCache(const RenderScene &scene,
                               const ResourceManager &resources,
                               uint32_t materialCount) {
  renderableTemplates_.clear();
  meshDrawTemplates_.clear();

  const std::span<const Renderable> renderables = scene.renderables();
  if (renderables.size() >
      static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    return Result<bool, std::string>::makeError(
        "OpaqueLayer::rebuildSceneCache: renderables count exceeds UINT32_MAX");
  }
  renderableTemplates_.reserve(renderables.size());

  size_t totalMeshDraws = 0;
  for (const Renderable &renderable : renderables) {
    const ModelRecord *modelRecord = resources.tryGet(renderable.model);
    if (!modelRecord || !modelRecord->model) {
      return Result<bool, std::string>::makeError(
          "OpaqueLayer::rebuildSceneCache: renderable model handle is invalid");
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
          "OpaqueLayer::rebuildSceneCache: failed to resolve model handle");
    }
    const Model *model = modelRecord->model.get();
    GeometryAllocationView geometry{};
    if (!gpu_.resolveGeometry(model->geometryHandle(), geometry)) {
      return Result<bool, std::string>::makeError(
          "OpaqueLayer::rebuildSceneCache: failed to resolve geometry "
          "allocation");
    }
    if (!nuri::isValid(geometry.vertexBuffer) ||
        !nuri::isValid(geometry.indexBuffer)) {
      return Result<bool, std::string>::makeError(
          "OpaqueLayer::rebuildSceneCache: resolved geometry uses invalid "
          "buffers");
    }
    const uint64_t vertexBufferAddress = gpu_.getBufferDeviceAddress(
        geometry.vertexBuffer, geometry.vertexByteOffset);
    if (vertexBufferAddress == 0) {
      return Result<bool, std::string>::makeError(
          "OpaqueLayer::rebuildSceneCache: invalid geometry vertex buffer "
          "address");
    }

    renderableTemplates_.push_back(
        RenderableTemplate{.renderable = &renderable, .model = model});

    const std::span<const Submesh> submeshes = model->submeshes();
    for (size_t submeshIndex = 0; submeshIndex < submeshes.size();
         ++submeshIndex) {
      const MaterialRef resolvedModelMaterial =
          modelRecord->materialForSubmesh(static_cast<uint32_t>(submeshIndex));
      const MaterialRef resolvedMaterial = isValid(resolvedModelMaterial)
                                               ? resolvedModelMaterial
                                               : renderable.material;
      const MaterialRecord *materialRecord = resources.tryGet(resolvedMaterial);
      const bool doubleSided =
          materialRecord != nullptr && materialRecord->desc.doubleSided;
      if (materialRecord != nullptr &&
          materialRecord->desc.alphaMode == MaterialAlphaMode::Blend) {
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
          .vertexBufferAddress = vertexBufferAddress,
          .materialIndex = finalMaterialIndex,
          .doubleSided = doubleSided,
      });
    }
  }

  if (invalidMaterialFallbackCount > 0u) {
    if (!loggedMaterialFallbackWarning_) {
      NURI_LOG_WARNING(
          "OpaqueLayer::rebuildSceneCache: %zu submesh draw(s) used fallback "
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
  invalidateIndirectPackCache();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> OpaqueLayer::rebuildMaterialTextureAccessCache(
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
    const MaterialRecord *materialRecord =
        resources.tryGet(renderable.material);
    if (materialRecord == nullptr) {
      continue;
    }

    const std::array<TextureRef, 8> refs = {
        materialRecord->textureRefs.baseColor,
        materialRecord->textureRefs.metallicRoughness,
        materialRecord->textureRefs.normal,
        materialRecord->textureRefs.occlusion,
        materialRecord->textureRefs.emissive,
        materialRecord->textureRefs.clearcoat,
        materialRecord->textureRefs.clearcoatRoughness,
        materialRecord->textureRefs.clearcoatNormal,
    };
    for (const TextureRef ref : refs) {
      const TextureRecord *record = resources.tryGet(ref);
      if (record == nullptr || !nuri::isValid(record->texture)) {
        continue;
      }
      const uint64_t key =
          foldHandle(record->texture.index, record->texture.generation);
      if (!textureKeys.insert(key).second) {
        continue;
      }
      materialTextureAccessHandles_.push_back(record->texture);
    }
  }

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> OpaqueLayer::createShaders() {
  meshShader_ = Shader::create("main", gpu_);
  meshTessShader_ = Shader::create("main_tess", gpu_);
  meshDebugOverlayShader_ = Shader::create("mesh_debug_overlay", gpu_);
  meshPickShader_ = Shader::create("main_id", gpu_);
  computeShader_ = Shader::create("duck_instances", gpu_);
  if (!meshShader_ || !meshTessShader_ || !meshPickShader_ || !computeShader_) {
    return Result<bool, std::string>::makeError(
        "OpaqueLayer::createShaders: failed to create shader objects");
  }

  meshVertexShader_ = {};
  meshTessVertexShader_ = {};
  meshTessControlShader_ = {};
  meshTessEvalShader_ = {};
  meshFragmentShader_ = {};
  meshDebugOverlayGeometryShader_ = {};
  meshDebugOverlayFragmentShader_ = {};
  meshPickFragmentShader_ = {};
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
          "OpaqueLayer::createShaders: invalid shader spec");
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
          "OpaqueLayer::createShaders: invalid tessellation shader spec");
      break;
    }

    const std::string shaderPath = spec.path->string();
    auto compileResult = spec.shader->compileFromFile(shaderPath, spec.stage);
    if (compileResult.hasError()) {
      tessellationUnsupported_ = true;
      meshTessVertexShader_ = {};
      meshTessControlShader_ = {};
      meshTessEvalShader_ = {};
      NURI_LOG_WARNING("OpaqueLayer::createShaders: Tessellation shader path "
                       "'%s' failed, fallback to non-tessellation path: %s",
                       shaderPath.c_str(), compileResult.error().c_str());
      break;
    }
    *spec.outHandle = compileResult.value();
  }

  if (!meshDebugOverlayShader_) {
    gsOverlayPipelineUnsupported_ = true;
    gsTessOverlayPipelineUnsupported_ = true;
    NURI_LOG_WARNING("OpaqueLayer::createShaders: failed to create debug "
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
          "OpaqueLayer::createShaders: invalid debug overlay shader spec");
      break;
    }
    const std::string shaderPath = spec.path->string();
    auto compileResult = spec.shader->compileFromFile(shaderPath, spec.stage);
    if (compileResult.hasError()) {
      gsOverlayPipelineUnsupported_ = true;
      gsTessOverlayPipelineUnsupported_ = true;
      meshDebugOverlayGeometryShader_ = {};
      meshDebugOverlayFragmentShader_ = {};
      NURI_LOG_WARNING("OpaqueLayer::createShaders: Debug overlay shader path "
                       "'%s' failed, fallback to line pipelines: %s",
                       shaderPath.c_str(), compileResult.error().c_str());
      break;
    }
    *spec.outHandle = compileResult.value();
  }

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> OpaqueLayer::createPipelines() {
  meshPipeline_ = Pipeline::create(gpu_);
  computePipeline_ = Pipeline::create(gpu_);
  if (!meshPipeline_ || !computePipeline_) {
    return Result<bool, std::string>::makeError(
        "OpaqueLayer::createPipelines: failed to create pipeline wrappers");
  }

  resetMeshPipelineState();
  const Format depthFormat = nuri::isValid(depthTexture_)
                                 ? gpu_.getTextureFormat(depthTexture_)
                                 : Format::D32_FLOAT;
  const RenderPipelineDesc meshDesc = meshPipelineDesc(
      gpu_.getSwapchainFormat(), depthFormat, meshVertexShader_, {}, {}, {},
      meshFragmentShader_, PolygonMode::Fill);
  auto meshResult =
      meshPipeline_->createRenderPipeline(meshDesc, "opaque_mesh");
  if (meshResult.hasError()) {
    return Result<bool, std::string>::makeError(meshResult.error());
  }
  meshFillPipelineHandle_ = meshPipeline_->getRenderPipeline();

  {
    const RenderPipelineDesc doubleSidedMeshDesc = meshPipelineDesc(
        gpu_.getSwapchainFormat(), depthFormat, meshVertexShader_, {}, {}, {},
        meshFragmentShader_, PolygonMode::Fill, Topology::Triangle, 0, false,
        CullMode::None);
    auto doubleSidedMeshResult = gpu_.createRenderPipeline(
        doubleSidedMeshDesc, "opaque_mesh_double_sided");
    if (doubleSidedMeshResult.hasError()) {
      return Result<bool, std::string>::makeError(
          doubleSidedMeshResult.error());
    }
    meshDoubleSidedFillPipelineHandle_ = doubleSidedMeshResult.value();
  }

  const bool canCreateTessPipeline =
      !tessellationUnsupported_ && nuri::isValid(meshTessVertexShader_) &&
      nuri::isValid(meshTessControlShader_) &&
      nuri::isValid(meshTessEvalShader_) && nuri::isValid(meshFragmentShader_);
  if (canCreateTessPipeline) {
    const RenderPipelineDesc tessDesc = meshPipelineDesc(
        gpu_.getSwapchainFormat(), depthFormat, meshTessVertexShader_,
        meshTessControlShader_, meshTessEvalShader_, {}, meshFragmentShader_,
        PolygonMode::Fill, Topology::Patch, kTessellationPatchControlPoints);
    auto tessResult = gpu_.createRenderPipeline(tessDesc, "opaque_mesh_tess");
    if (tessResult.hasError()) {
      tessellationUnsupported_ = true;
      meshTessPipelineHandle_ = {};
      NURI_LOG_WARNING("OpaqueLayer::createPipelines: Tessellation pipeline "
                       "failed, fallback to non-tessellation path: %s",
                       tessResult.error().c_str());
    } else {
      meshTessPipelineHandle_ = tessResult.value();
    }
    if (!tessellationUnsupported_) {
      const RenderPipelineDesc doubleSidedTessDesc = meshPipelineDesc(
          gpu_.getSwapchainFormat(), depthFormat, meshTessVertexShader_,
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
        meshDoubleSidedTessPipelineHandle_ = {};
        NURI_LOG_WARNING("OpaqueLayer::createPipelines: double-sided "
                         "tessellation pipeline failed, fallback to "
                         "non-tessellation path: %s",
                         doubleSidedTessResult.error().c_str());
      } else {
        meshDoubleSidedTessPipelineHandle_ = doubleSidedTessResult.value();
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
          "OpaqueLayer::createPipelines: tessellation pick pipeline failed, "
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
        NURI_LOG_WARNING("OpaqueLayer::createPipelines: double-sided "
                         "tessellation pick pipeline failed, falling back to "
                         "non-tessellation pick pipeline: %s",
                         doubleSidedPickTessResult.error().c_str());
      } else {
        meshPickDoubleSidedTessPipelineHandle_ =
            doubleSidedPickTessResult.value();
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

  baseMeshFillDraw_ = makeBaseMeshDraw(meshFillPipelineHandle_, "OpaqueMesh");
  resetOverlayPipelineState();

  return Result<bool, std::string>::makeResult(true);
}

RenderPipelineHandle OpaqueLayer::selectMeshPipeline(bool doubleSided,
                                                     bool tessellated) const {
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

RenderPipelineHandle
OpaqueLayer::selectPickPipeline(RenderPipelineHandle sourcePipeline) const {
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

bool OpaqueLayer::isDoubleSidedPipeline(RenderPipelineHandle handle) const {
  return isSamePipelineHandle(handle, meshDoubleSidedFillPipelineHandle_) ||
         isSamePipelineHandle(handle, meshDoubleSidedTessPipelineHandle_);
}

bool OpaqueLayer::isTessPipeline(RenderPipelineHandle handle) const {
  return isSamePipelineHandle(handle, meshTessPipelineHandle_) ||
         isSamePipelineHandle(handle, meshDoubleSidedTessPipelineHandle_);
}

Result<bool, std::string> OpaqueLayer::ensureWireframePipeline() {
  if (wireframePipelineInitialized_ &&
      nuri::isValid(meshWireframePipelineHandle_)) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (wireframePipelineUnsupported_) {
    return Result<bool, std::string>::makeResult(false);
  }
  if (!nuri::isValid(meshFillPipelineHandle_)) {
    return Result<bool, std::string>::makeError(
        "OpaqueLayer::ensureWireframePipeline: fill pipeline is invalid");
  }

  const Format depthFormat = nuri::isValid(depthTexture_)
                                 ? gpu_.getTextureFormat(depthTexture_)
                                 : Format::D32_FLOAT;
  const RenderPipelineDesc wireframeDesc = meshPipelineDesc(
      gpu_.getSwapchainFormat(), depthFormat, meshVertexShader_, {}, {}, {},
      meshFragmentShader_, PolygonMode::Line, Topology::Triangle, 0, true);

  auto pipelineResult =
      gpu_.createRenderPipeline(wireframeDesc, "opaque_mesh_wireframe");
  if (pipelineResult.hasError()) {
    wireframePipelineUnsupported_ = true;
    if (!loggedWireframeFallbackUnsupported_) {
      loggedWireframeFallbackUnsupported_ = true;
      NURI_LOG_WARNING("OpaqueLayer::ensureWireframePipeline: %s",
                       pipelineResult.error().c_str());
    }
    return Result<bool, std::string>::makeResult(false);
  }

  meshWireframePipelineHandle_ = pipelineResult.value();
  wireframePipelineInitialized_ = true;

  baseMeshWireframeDraw_ = baseMeshFillDraw_;
  baseMeshWireframeDraw_.pipeline = meshWireframePipelineHandle_;
  baseMeshWireframeDraw_.debugLabel = "OpaqueMeshWireframe";

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> OpaqueLayer::ensureTessWireframePipeline() {
  if (tessWireframePipelineInitialized_ &&
      nuri::isValid(meshTessWireframePipelineHandle_)) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (tessWireframePipelineUnsupported_) {
    return Result<bool, std::string>::makeResult(false);
  }
  if (!nuri::isValid(meshTessPipelineHandle_)) {
    return Result<bool, std::string>::makeResult(false);
  }

  const Format depthFormat = nuri::isValid(depthTexture_)
                                 ? gpu_.getTextureFormat(depthTexture_)
                                 : Format::D32_FLOAT;
  const RenderPipelineDesc wireframeDesc = meshPipelineDesc(
      gpu_.getSwapchainFormat(), depthFormat, meshTessVertexShader_,
      meshTessControlShader_, meshTessEvalShader_, {}, meshFragmentShader_,
      PolygonMode::Line, Topology::Patch, kTessellationPatchControlPoints,
      true);

  auto pipelineResult =
      gpu_.createRenderPipeline(wireframeDesc, "opaque_mesh_tess_wireframe");
  if (pipelineResult.hasError()) {
    tessWireframePipelineUnsupported_ = true;
    if (!loggedTessWireframeFallbackUnsupported_) {
      loggedTessWireframeFallbackUnsupported_ = true;
      NURI_LOG_WARNING("OpaqueLayer::ensureTessWireframePipeline: %s",
                       pipelineResult.error().c_str());
    }
    return Result<bool, std::string>::makeResult(false);
  }

  meshTessWireframePipelineHandle_ = pipelineResult.value();
  tessWireframePipelineInitialized_ = true;

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> OpaqueLayer::ensureGsOverlayPipeline() {
  if (gsOverlayPipelineInitialized_ &&
      nuri::isValid(meshGsOverlayPipelineHandle_)) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (gsOverlayPipelineUnsupported_) {
    return Result<bool, std::string>::makeResult(false);
  }
  if (!nuri::isValid(meshDebugOverlayGeometryShader_) ||
      !nuri::isValid(meshDebugOverlayFragmentShader_)) {
    gsOverlayPipelineUnsupported_ = true;
    if (!loggedGsOverlayUnsupported_) {
      loggedGsOverlayUnsupported_ = true;
      NURI_LOG_WARNING("OpaqueLayer::ensureGsOverlayPipeline: debug overlay "
                       "shaders are unavailable, fallback to line overlay");
    }
    return Result<bool, std::string>::makeResult(false);
  }

  const Format depthFormat = nuri::isValid(depthTexture_)
                                 ? gpu_.getTextureFormat(depthTexture_)
                                 : Format::D32_FLOAT;
  const RenderPipelineDesc overlayDesc = meshPipelineDesc(
      gpu_.getSwapchainFormat(), depthFormat, meshVertexShader_, {}, {},
      meshDebugOverlayGeometryShader_, meshDebugOverlayFragmentShader_,
      PolygonMode::Fill, Topology::Triangle, 0, true);

  auto pipelineResult =
      gpu_.createRenderPipeline(overlayDesc, "opaque_mesh_overlay_gs");
  if (pipelineResult.hasError()) {
    gsOverlayPipelineUnsupported_ = true;
    if (!loggedGsOverlayUnsupported_) {
      loggedGsOverlayUnsupported_ = true;
      NURI_LOG_WARNING("OpaqueLayer::ensureGsOverlayPipeline: %s",
                       pipelineResult.error().c_str());
    }
    return Result<bool, std::string>::makeResult(false);
  }

  meshGsOverlayPipelineHandle_ = pipelineResult.value();
  gsOverlayPipelineInitialized_ = true;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> OpaqueLayer::ensureGsTessOverlayPipeline() {
  if (gsTessOverlayPipelineInitialized_ &&
      nuri::isValid(meshGsTessOverlayPipelineHandle_)) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (gsTessOverlayPipelineUnsupported_) {
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
      NURI_LOG_WARNING("OpaqueLayer::ensureGsTessOverlayPipeline: debug "
                       "overlay shaders are unavailable, fallback to line "
                       "overlay");
    }
    return Result<bool, std::string>::makeResult(false);
  }

  const Format depthFormat = nuri::isValid(depthTexture_)
                                 ? gpu_.getTextureFormat(depthTexture_)
                                 : Format::D32_FLOAT;
  const RenderPipelineDesc overlayDesc =
      meshPipelineDesc(gpu_.getSwapchainFormat(), depthFormat,
                       meshTessVertexShader_, meshTessControlShader_,
                       meshTessEvalShader_, meshDebugOverlayGeometryShader_,
                       meshDebugOverlayFragmentShader_, PolygonMode::Fill,
                       Topology::Patch, kTessellationPatchControlPoints, true);

  auto pipelineResult =
      gpu_.createRenderPipeline(overlayDesc, "opaque_mesh_tess_overlay_gs");
  if (pipelineResult.hasError()) {
    gsTessOverlayPipelineUnsupported_ = true;
    if (!loggedGsTessOverlayUnsupported_) {
      loggedGsTessOverlayUnsupported_ = true;
      NURI_LOG_WARNING("OpaqueLayer::ensureGsTessOverlayPipeline: %s",
                       pipelineResult.error().c_str());
    }
    return Result<bool, std::string>::makeResult(false);
  }

  meshGsTessOverlayPipelineHandle_ = pipelineResult.value();
  gsTessOverlayPipelineInitialized_ = true;
  return Result<bool, std::string>::makeResult(true);
}

void OpaqueLayer::resetOverlayPipelineState() {
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
  meshGsOverlayPipelineHandle_ = {};
  meshGsTessOverlayPipelineHandle_ = {};
  meshWireframePipelineHandle_ = {};
  meshTessWireframePipelineHandle_ = {};
  gsOverlayPipelineInitialized_ = false;
  gsTessOverlayPipelineInitialized_ = false;
  wireframePipelineInitialized_ = false;
  tessWireframePipelineInitialized_ = false;
  gsOverlayPipelineUnsupported_ = false;
  gsTessOverlayPipelineUnsupported_ = false;
  wireframePipelineUnsupported_ = false;
  tessWireframePipelineUnsupported_ = false;
  loggedWireframeFallbackUnsupported_ = false;
  loggedTessWireframeFallbackUnsupported_ = false;
  loggedGsOverlayUnsupported_ = false;
  loggedGsTessOverlayUnsupported_ = false;
  baseMeshWireframeDraw_ = {};
}

void OpaqueLayer::invalidateAutoLodCache() {
  autoLodCache_.valid = false;
  autoLodCache_.remapCount = 0;
  autoLodCache_.instanceCount = 0;
  autoLodCache_.submesh = nullptr;
  autoLodCache_.frameIndex = std::numeric_limits<uint64_t>::max();
  autoLodCache_.bucketCounts.fill(0);
}

void OpaqueLayer::invalidateSingleInstanceBatchCache() {
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

void OpaqueLayer::invalidateIndirectPackCache() {
  indirectPackCache_.valid = false;
  indirectPackCache_.drawSignature = kInvalidDrawSignature;
  indirectPackCache_.requiredBytes = 0;
  indirectSourceDrawIndices_.clear();
  for (uint64_t &slotSignature : indirectUploadSignatures_) {
    slotSignature = kInvalidDrawSignature;
  }
}

void OpaqueLayer::destroyMeshPipelineState() {
  destroyPipelineHandle(gpu_, meshPickDoubleSidedTessPipelineHandle_);
  destroyPipelineHandle(gpu_, meshPickTessPipelineHandle_);
  destroyPipelineHandle(gpu_, meshPickDoubleSidedPipelineHandle_);
  destroyPipelineHandle(gpu_, meshPickPipelineHandle_);
  destroyPipelineHandle(gpu_, meshDoubleSidedTessPipelineHandle_);
  destroyPipelineHandle(gpu_, meshTessPipelineHandle_);
  destroyPipelineHandle(gpu_, meshDoubleSidedFillPipelineHandle_);
  resetMeshPipelineState();
}

void OpaqueLayer::resetMeshPipelineState() {
  meshFillPipelineHandle_ = {};
  meshDoubleSidedFillPipelineHandle_ = {};
  meshTessPipelineHandle_ = {};
  meshDoubleSidedTessPipelineHandle_ = {};
  meshPickPipelineHandle_ = {};
  meshPickDoubleSidedPipelineHandle_ = {};
  meshPickTessPipelineHandle_ = {};
  meshPickDoubleSidedTessPipelineHandle_ = {};
  baseMeshFillDraw_ = {};
}

void OpaqueLayer::updateFastAutoLodCache(
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

void OpaqueLayer::destroyDepthTexture() {
  if (nuri::isValid(depthTexture_)) {
    gpu_.destroyTexture(depthTexture_);
    depthTexture_ = TextureHandle{};
  }
}

void OpaqueLayer::destroyPickTexture() {
  if (nuri::isValid(pickIdTexture_)) {
    gpu_.destroyTexture(pickIdTexture_);
    pickIdTexture_ = TextureHandle{};
  }
}

void OpaqueLayer::destroyBuffers() {
  if (frameDataBuffer_ && frameDataBuffer_->valid()) {
    gpu_.destroyBuffer(frameDataBuffer_->handle());
  }
  frameDataBuffer_.reset();
  frameDataBufferCapacityBytes_ = 0;
  frameDataUploadValid_ = false;

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

  if (materialBuffer_ && materialBuffer_->valid()) {
    gpu_.destroyBuffer(materialBuffer_->handle());
  }
  materialBuffer_.reset();
  materialBufferCapacityBytes_ = 0;

  for (DynamicBufferSlot &slot : instanceMatricesRing_) {
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
  instanceRemapRing_.clear();
  indirectCommandRing_.clear();
  remapUploadSignatures_.clear();
  indirectUploadSignatures_.clear();
  invalidateIndirectPackCache();
}

} // namespace nuri
