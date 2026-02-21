#include "nuri/pch.h"

#include "nuri/gfx/layers/opaque_layer.h"

#include "nuri/core/log.h"
#include "nuri/core/pmr_scratch.h"
#include "nuri/core/profiling.h"
#include "nuri/scene/render_scene.h"

namespace nuri {
namespace {
constexpr float kMinLodRadius = 1.0e-4f;
constexpr float kAutoLodCacheEpsilon = 1.0e-4f;
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
constexpr uint64_t kFnvOffsetBasis64 = 1469598103934665603ull;
constexpr uint64_t kFnvPrime64 = 1099511628211ull;
constexpr uint64_t kInvalidDrawSignature = std::numeric_limits<uint64_t>::max();

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

RenderPipelineDesc meshPipelineDesc(
    Format swapchainFormat, Format depthFormat, ShaderHandle vertexShader,
    ShaderHandle tessControlShader, ShaderHandle tessEvalShader,
    ShaderHandle geometryShader, ShaderHandle fragmentShader,
    PolygonMode polygonMode, Topology topology = Topology::Triangle,
    uint32_t patchControlPoints = 0, bool blendEnabled = false) {
  return RenderPipelineDesc{
      .vertexInput = {},
      .vertexShader = vertexShader,
      .tessControlShader = tessControlShader,
      .tessEvalShader = tessEvalShader,
      .geometryShader = geometryShader,
      .fragmentShader = fragmentShader,
      .colorFormats = {swapchainFormat},
      .depthFormat = depthFormat,
      .cullMode = CullMode::Back,
      .polygonMode = polygonMode,
      .topology = topology,
      .patchControlPoints = patchControlPoints,
      .blendEnabled = blendEnabled,
  };
}

bool isTessPipelineHandle(RenderPipelineHandle handle,
                          RenderPipelineHandle tessPipeline) {
  return handle.index == tessPipeline.index &&
         handle.generation == tessPipeline.generation;
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

  bool operator==(const BatchKey &other) const {
    return pipeline.index == other.pipeline.index &&
           pipeline.generation == other.pipeline.generation &&
           indexBuffer.index == other.indexBuffer.index &&
           indexBuffer.generation == other.indexBuffer.generation &&
           indexBufferOffset == other.indexBufferOffset &&
           indexCount == other.indexCount && firstIndex == other.firstIndex &&
           vertexBufferAddress == other.vertexBufferAddress;
  }
};

struct BatchKeyHash {
  size_t operator()(const BatchKey &key) const noexcept {
    size_t h = 1469598103934665603ull;
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

  bool operator==(const IndirectGroupKey &other) const {
    return pipeline.index == other.pipeline.index &&
           pipeline.generation == other.pipeline.generation &&
           indexBuffer.index == other.indexBuffer.index &&
           indexBuffer.generation == other.indexBuffer.generation &&
           indexBufferOffset == other.indexBufferOffset &&
           indexFormat == other.indexFormat &&
           vertexBufferAddress == other.vertexBufferAddress;
  }
};

struct IndirectGroupKeyHash {
  size_t operator()(const IndirectGroupKey &key) const noexcept {
    size_t h = 1469598103934665603ull;
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
    return h;
  }
};

} // namespace

OpaqueLayer::OpaqueLayer(GPUDevice &gpu, OpaqueLayerConfig config,
                         std::pmr::memory_resource *memory)
    : gpu_(gpu), config_(std::move(config)),
      singleInstanceBatchCache_(resolveMemoryResource(memory)),
      renderableTemplates_(resolveMemoryResource(memory)),
      meshDrawTemplates_(resolveMemoryResource(memory)),
      indirectSourceDrawIndices_(resolveMemoryResource(memory)),
      indirectUploadSignatures_(resolveMemoryResource(memory)),
      templateBatchIndices_(resolveMemoryResource(memory)),
      batchWriteOffsets_(resolveMemoryResource(memory)),
      instanceCentersPhase_(resolveMemoryResource(memory)),
      instanceBaseMatrices_(resolveMemoryResource(memory)),
      instanceLodCentersInvRadiusSq_(resolveMemoryResource(memory)),
      instanceAlbedoTexIds_(resolveMemoryResource(memory)),
      instanceAutoLodLevels_(resolveMemoryResource(memory)),
      instanceTessSelection_(resolveMemoryResource(memory)),
      tessCandidates_(resolveMemoryResource(memory)),
      instanceRemap_(resolveMemoryResource(memory)),
      drawPushConstants_(resolveMemoryResource(memory)),
      drawItems_(resolveMemoryResource(memory)),
      indirectPushConstants_(resolveMemoryResource(memory)),
      indirectDrawItems_(resolveMemoryResource(memory)),
      indirectCommandUploadBytes_(resolveMemoryResource(memory)),
      overlayDrawItems_(resolveMemoryResource(memory)),
      passDrawItems_(resolveMemoryResource(memory)),
      preDispatches_(resolveMemoryResource(memory)),
      passDependencyBuffers_(resolveMemoryResource(memory)),
      dispatchDependencyBuffers_(resolveMemoryResource(memory)) {}

OpaqueLayer::~OpaqueLayer() { onDetach(); }

void OpaqueLayer::onAttach() {
  auto initResult = ensureInitialized();
  if (initResult.hasError()) {
    NURI_LOG_WARNING("OpaqueLayer::onAttach: %s", initResult.error().c_str());
  }
}

void OpaqueLayer::onDetach() {
  destroyBuffers();
  destroyDepthTexture();
  resetOverlayPipelineState();
  if (nuri::isValid(meshTessPipelineHandle_)) {
    gpu_.destroyRenderPipeline(meshTessPipelineHandle_);
    meshTessPipelineHandle_ = {};
  }
  meshPipeline_.reset();
  computePipeline_.reset();
  meshShader_.reset();
  meshTessShader_.reset();
  meshDebugOverlayShader_.reset();
  computeShader_.reset();
  meshVertexShader_ = {};
  meshTessVertexShader_ = {};
  meshTessControlShader_ = {};
  meshTessEvalShader_ = {};
  meshFragmentShader_ = {};
  meshDebugOverlayGeometryShader_ = {};
  meshDebugOverlayFragmentShader_ = {};
  computeShaderHandle_ = {};
  meshFillPipelineHandle_ = {};
  meshTessPipelineHandle_ = {};
  computePipelineHandle_ = {};
  tessellationUnsupported_ = false;
  baseMeshFillDraw_ = {};
  baseMeshWireframeDraw_ = {};
  renderableTemplates_.clear();
  meshDrawTemplates_.clear();
  templateBatchIndices_.clear();
  batchWriteOffsets_.clear();
  instanceLodCentersInvRadiusSq_.clear();
  instanceAutoLodLevels_.clear();
  instanceTessSelection_.clear();
  tessCandidates_.clear();
  instanceRemap_.clear();
  drawPushConstants_.clear();
  drawItems_.clear();
  indirectUploadSignatures_.clear();
  indirectPushConstants_.clear();
  indirectDrawItems_.clear();
  indirectCommandUploadBytes_.clear();
  overlayDrawItems_.clear();
  passDrawItems_.clear();
  preDispatches_.clear();
  passDependencyBuffers_.clear();
  dispatchDependencyBuffers_.clear();
  cachedScene_ = nullptr;
  cachedTopologyVersion_ = std::numeric_limits<uint64_t>::max();
  cachedTransformVersion_ = std::numeric_limits<uint64_t>::max();
  instanceStaticBuffersDirty_ = true;
  uniformSingleSubmeshPath_ = false;
  invalidateAutoLodCache();
  invalidateSingleInstanceBatchCache();
  invalidateIndirectPackCache();
  initialized_ = false;
}

void OpaqueLayer::onResize(int32_t, int32_t) { destroyDepthTexture(); }

Result<bool, std::string>
OpaqueLayer::buildRenderPasses(RenderFrameContext &frame, RenderPassList &out) {
  NURI_PROFILER_FUNCTION();
  frame.metrics.opaque = {};

  const RenderSettings &settings = settingsOrDefault(frame);
  if (!settings.opaque.enabled) {
    return Result<bool, std::string>::makeResult(true);
  }

  if (!frame.scene) {
    return Result<bool, std::string>::makeError(
        "OpaqueLayer::buildRenderPasses: frame scene is null");
  }

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
  const bool topologyDirty =
      cachedScene_ != frame.scene ||
      cachedTopologyVersion_ != frame.scene->topologyVersion();
  if (topologyDirty) {
    auto cacheResult = rebuildSceneCache(*frame.scene);
    if (cacheResult.hasError()) {
      return cacheResult;
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
    instanceAlbedoTexIds_.clear();
    instanceCentersPhase_.reserve(instanceCount);
    instanceBaseMatrices_.reserve(instanceCount);
    instanceLodCentersInvRadiusSq_.reserve(instanceCount);
    instanceAlbedoTexIds_.reserve(instanceCount);

    const bool animateInstances = settings.opaque.enableInstanceAnimation;
    for (size_t i = 0; i < instanceCount; ++i) {
      const OpaqueRenderable *renderable = renderableTemplates_[i].renderable;
      if (!renderable || !renderable->albedoTexture || !renderable->model) {
        return Result<bool, std::string>::makeError(
            "OpaqueLayer::buildRenderPasses: invalid opaque renderable");
      }
      if (!renderable->albedoTexture->valid()) {
        return Result<bool, std::string>::makeError(
            "OpaqueLayer::buildRenderPasses: invalid albedo texture handle");
      }

      const glm::vec3 center = glm::vec3(renderable->modelMatrix[3]);
      instanceCentersPhase_.push_back(glm::vec4(
          center, animateInstances ? deterministicPhase(static_cast<uint32_t>(i))
                                   : 0.0f));
      glm::mat4 baseMatrix = renderable->modelMatrix;
      baseMatrix[3] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
      instanceBaseMatrices_.push_back(baseMatrix);

      const BoundingBox &bounds = renderable->model->bounds();
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

      instanceAlbedoTexIds_.push_back(
          gpu_.getTextureBindlessIndex(renderable->albedoTexture->handle()));
    }

    cachedTransformVersion_ = frame.scene->transformVersion();
    instanceStaticBuffersDirty_ = true;
  }

  uint32_t cubemapTexId = 0;
  uint32_t hasCubemap = 0;
  if (const Texture *cubemap = frame.scene->environmentCubemap();
      cubemap && cubemap->valid()) {
    cubemapTexId = gpu_.getTextureBindlessIndex(cubemap->handle());
    hasCubemap = 1;
  }

  frameData_ = FrameData{
      .view = frame.camera.view,
      .proj = frame.camera.proj,
      .cameraPos = frame.camera.cameraPos,
      .cubemapTexId = cubemapTexId,
      .hasCubemap = hasCubemap,
      ._padding0 = 0,
      ._padding1 = 0,
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
  auto metaResult = ensureInstanceMetaBufferCapacity(
      std::max(instanceCount * sizeof(uint32_t), sizeof(uint32_t)));
  if (metaResult.hasError()) {
    return metaResult;
  }
  auto matricesResult = ensureInstanceMatricesRingCapacity(
      std::max(instanceCount * sizeof(glm::mat4), sizeof(glm::mat4)));
  if (matricesResult.hasError()) {
    return matricesResult;
  }

  {
    const std::span<const std::byte> frameDataBytes{
        reinterpret_cast<const std::byte *>(&frameData_), sizeof(frameData_)};
    auto updateResult =
        gpu_.updateBuffer(frameDataBuffer_->handle(), frameDataBytes, 0);
    if (updateResult.hasError()) {
      return updateResult;
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
    if (!instanceAlbedoTexIds_.empty()) {
      const std::span<const std::byte> metaBytes{
          reinterpret_cast<const std::byte *>(instanceAlbedoTexIds_.data()),
          instanceAlbedoTexIds_.size() * sizeof(uint32_t)};
      auto updateResult =
          gpu_.updateBuffer(instanceMetaBuffer_->handle(), metaBytes, 0);
      if (updateResult.hasError()) {
        return updateResult;
      }
    }
    instanceStaticBuffersDirty_ = false;
  }

  const uint64_t frameDataAddress =
      gpu_.getBufferDeviceAddress(frameDataBuffer_->handle());
  const uint64_t instanceCentersPhaseAddress =
      gpu_.getBufferDeviceAddress(instanceCentersPhaseBuffer_->handle());
  const uint64_t instanceBaseMatricesAddress =
      gpu_.getBufferDeviceAddress(instanceBaseMatricesBuffer_->handle());
  const uint64_t instanceMetaAddress =
      gpu_.getBufferDeviceAddress(instanceMetaBuffer_->handle());
  const uint64_t instanceMatricesAddress = gpu_.getBufferDeviceAddress(
      instanceMatricesRing_[frameSlot].buffer->handle());
  if (frameDataAddress == 0 || instanceCentersPhaseAddress == 0 ||
      instanceBaseMatricesAddress == 0 || instanceMetaAddress == 0 ||
      instanceMatricesAddress == 0) {
    return Result<bool, std::string>::makeError(
        "OpaqueLayer::buildRenderPasses: invalid GPU buffer address");
  }

  const OpaqueDebugVisualization debugVisualization =
      settings.opaque.debugVisualization;
  const bool overlayRequested =
      debugVisualization != OpaqueDebugVisualization::None;
  const bool patchHeatmapRequested =
      debugVisualization == OpaqueDebugVisualization::TessPatchEdgesHeatmap;
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
    size_t instanceCount = 0;
    size_t firstInstance = 0;
  };
  constexpr uint32_t kInvalidBatchIndex = std::numeric_limits<uint32_t>::max();

  std::vector<BatchEntry> batches;
  const size_t batchReserve =
      std::min<size_t>(meshDrawTemplates_.size(), kMaxBatchReserve);
  batches.reserve(batchReserve);
  const auto appendBatch = [&baseDraw, &batches](
                               RenderPipelineHandle pipeline,
                               BufferHandle indexBuffer,
                               uint64_t indexBufferOffset,
                               const SubmeshLod &lodRange,
                               uint64_t vertexBufferAddress, size_t count,
                               size_t firstInstance) {
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
    entry.instanceCount = count;
    entry.firstInstance = firstInstance;
    batches.push_back(entry);
  };
  std::unordered_map<BatchKey, size_t, BatchKeyHash> batchLookup;
  batchLookup.reserve(batchReserve);
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
          "OpaqueLayer::buildRenderPasses: LOD cache size mismatch");
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
          "OpaqueLayer::buildRenderPasses: failed to refresh geometry");
    }
    if (!nuri::isValid(geometry.vertexBuffer) ||
        !nuri::isValid(geometry.indexBuffer) ||
        !gpu_.isValid(geometry.vertexBuffer) ||
        !gpu_.isValid(geometry.indexBuffer)) {
      return Result<bool, std::string>::makeError(
          "OpaqueLayer::buildRenderPasses: refreshed geometry is invalid");
    }
    const uint64_t refreshedVertexAddress = gpu_.getBufferDeviceAddress(
        geometry.vertexBuffer, geometry.vertexByteOffset);
    if (refreshedVertexAddress == 0) {
      return Result<bool, std::string>::makeError(
          "OpaqueLayer::buildRenderPasses: refreshed vertex address is "
          "invalid");
    }
    templateEntry.indexBuffer = geometry.indexBuffer;
    templateEntry.indexBufferOffset = geometry.indexByteOffset;
    templateEntry.vertexBufferAddress = refreshedVertexAddress;
    return Result<bool, std::string>::makeResult(true);
  };
  uint64_t meshTemplateSignature = kFnvOffsetBasis64;
  {
    NURI_PROFILER_ZONE("OpaqueLayer.sync_template_geometry",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    GeometryAllocationHandle cachedHandle{};
    BufferHandle cachedIndexBuffer{};
    uint64_t cachedIndexBufferOffset = 0;
    uint64_t cachedVertexBufferAddress = 0;
    bool hasCachedGeometry = false;

    for (MeshDrawTemplate &templateEntry : meshDrawTemplates_) {
      const bool sameAsCached =
          hasCachedGeometry &&
          templateEntry.geometryHandle.index == cachedHandle.index &&
          templateEntry.geometryHandle.generation == cachedHandle.generation;
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

      meshTemplateSignature =
          hashCombine64(meshTemplateSignature,
                        foldHandle(templateEntry.geometryHandle.index,
                                   templateEntry.geometryHandle.generation));
      meshTemplateSignature = hashCombine64(
          meshTemplateSignature,
          foldHandle(templateEntry.indexBuffer.index,
                     templateEntry.indexBuffer.generation));
      meshTemplateSignature =
          hashCombine64(meshTemplateSignature, templateEntry.indexBufferOffset);
      meshTemplateSignature =
          hashCombine64(meshTemplateSignature, templateEntry.vertexBufferAddress);
      meshTemplateSignature = hashCombine64(
          meshTemplateSignature,
          reinterpret_cast<uint64_t>(templateEntry.submesh));
    }
    NURI_PROFILER_ZONE_END();
  }
  meshTemplateSignature =
      hashCombine64(meshTemplateSignature, meshDrawTemplates_.size());
  meshTemplateSignature = hashCombine64(
      meshTemplateSignature,
      foldHandle(baseDraw.pipeline.index, baseDraw.pipeline.generation));
  meshTemplateSignature = hashCombine64(
      meshTemplateSignature,
      foldHandle(meshTessPipelineHandle_.index, meshTessPipelineHandle_.generation));
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

  if (canUseUniformAutoLodFastPath) {
    NURI_PROFILER_ZONE("OpaqueLayer.batch_build_auto_lod",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    MeshDrawTemplate &templateEntry = meshDrawTemplates_.front();
    if (!templateEntry.submesh) {
      return Result<bool, std::string>::makeError(
          "OpaqueLayer::buildRenderPasses: invalid auto-LOD submesh");
    }

    const Submesh &submesh = *templateEntry.submesh;
    activeFastAutoLodSubmesh = &submesh;

    const bool canReuseFastAutoLodCache =
        !tessellationRequested && autoLodCache_.valid &&
        autoLodCache_.submesh == &submesh &&
        autoLodCache_.instanceCount == instanceCount &&
        autoLodCache_.remapCount == instanceRemap_.size() &&
        nearlyEqualVec3(autoLodCache_.cameraPos, cameraPosition,
                        kAutoLodCacheEpsilon) &&
        nearlyEqualThresholds(autoLodCache_.thresholds, sortedLodThresholds,
                              kAutoLodCacheEpsilon);

    if (canReuseFastAutoLodCache) {
      autoLodBucketCounts = autoLodCache_.bucketCounts;
      remapCount = autoLodCache_.remapCount;
      reusedUniformAutoLodFastPath = true;
    } else {
      if (instanceLodCentersInvRadiusSq_.size() != instanceCount) {
        return Result<bool, std::string>::makeError(
            "OpaqueLayer::buildRenderPasses: auto-LOD cache size "
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
      appendBatch(baseDraw.pipeline, templateEntry.indexBuffer,
                  templateEntry.indexBufferOffset, lod0Range,
                  templateEntry.vertexBufferAddress, autoLodBucketCounts[0],
                  firstInstance);
      firstInstance += autoLodBucketCounts[0];

      autoLodTessBucketStart = firstInstance;
      autoLodTessBucketWrite = firstInstance;
      appendBatch(meshTessPipelineHandle_, templateEntry.indexBuffer,
                  templateEntry.indexBufferOffset, lod0Range,
                  templateEntry.vertexBufferAddress, autoLodTessBucketCount,
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

        appendBatch(baseDraw.pipeline, templateEntry.indexBuffer,
                    templateEntry.indexBufferOffset, lodRange,
                    templateEntry.vertexBufferAddress, count, firstInstance);
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

        appendBatch(baseDraw.pipeline, templateEntry.indexBuffer,
                    templateEntry.indexBufferOffset, lodRange,
                    templateEntry.vertexBufferAddress, count, firstInstance);
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

    uint32_t requestedLod = 0;
    if (!settings.opaque.enableMeshLod) {
      requestedLod = 0;
    } else if (settings.opaque.forcedMeshLod >= 0) {
      requestedLod = forcedLod;
    } else if (!instanceAutoLodLevels_.empty()) {
      requestedLod = instanceAutoLodLevels_.front();
    }

    bool tessPipelineEnabled = false;
    if (tessellationRequested && requestedLod == 0 &&
        !instanceLodCentersInvRadiusSq_.empty()) {
      const glm::vec4 centerInvRadiusSq = instanceLodCentersInvRadiusSq_.front();
      const float dx = cameraPosition.x - centerInvRadiusSq.x;
      const float dy = cameraPosition.y - centerInvRadiusSq.y;
      const float dz = cameraPosition.z - centerInvRadiusSq.z;
      const float distanceSq = dx * dx + dy * dy + dz * dz;
      tessPipelineEnabled = distanceSq <= tessFarDistanceSq;
    }

    const bool canReuseSingleInstanceCache =
        singleInstanceBatchCache_.valid &&
        singleInstanceBatchCache_.requestedLod == requestedLod &&
        singleInstanceBatchCache_.tessPipelineEnabled == tessPipelineEnabled &&
        singleInstanceBatchCache_.templateSignature == meshTemplateSignature;

    if (!canReuseSingleInstanceCache) {
      ScratchArena scratch;
      ScopedScratch scopedScratch(scratch);
      std::pmr::unordered_map<BatchKey, size_t, BatchKeyHash> singleBatchLookup(
          scopedScratch.resource());
      singleBatchLookup.reserve(meshDrawTemplates_.size());
      singleInstanceBatchCache_.batches.clear();
      singleInstanceBatchCache_.remapCount = 0;

      for (const MeshDrawTemplate &templateEntry : meshDrawTemplates_) {
        if (!templateEntry.renderable || !templateEntry.submesh) {
          return Result<bool, std::string>::makeError(
              "OpaqueLayer::buildRenderPasses: invalid mesh template");
        }

        const auto lodIndex =
            resolveAvailableLod(*templateEntry.submesh, requestedLod);
        if (!lodIndex) {
          continue;
        }

        RenderPipelineHandle selectedPipeline = baseDraw.pipeline;
        if (tessPipelineEnabled && *lodIndex == 0) {
          selectedPipeline = meshTessPipelineHandle_;
        }

        const SubmeshLod &lodRange = templateEntry.submesh->lods[*lodIndex];
        const BatchKey key{
            .pipeline = selectedPipeline,
            .indexBuffer = templateEntry.indexBuffer,
            .indexBufferOffset = templateEntry.indexBufferOffset,
            .indexCount = lodRange.indexCount,
            .firstIndex = lodRange.indexOffset,
            .vertexBufferAddress = templateEntry.vertexBufferAddress,
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
          singleInstanceBatchCache_.batches.push_back(entry);
          const size_t insertedIndex = singleInstanceBatchCache_.batches.size() - 1;
          singleBatchLookup.emplace(key, insertedIndex);
          it = singleBatchLookup.find(key);
        }

        ++singleInstanceBatchCache_.batches[it->second].instanceCount;
        ++singleInstanceBatchCache_.remapCount;
      }

      singleInstanceBatchCache_.requestedLod = requestedLod;
      singleInstanceBatchCache_.tessPipelineEnabled = tessPipelineEnabled;
      singleInstanceBatchCache_.templateSignature = meshTemplateSignature;
      singleInstanceBatchCache_.valid = true;
    }

    remapCount = singleInstanceBatchCache_.remapCount;
    if (remapCount > 0) {
      batches.clear();
      batches.reserve(singleInstanceBatchCache_.batches.size());
      for (const SingleInstanceBatchEntry &cachedBatch :
           singleInstanceBatchCache_.batches) {
        BatchEntry batch{};
        batch.draw = cachedBatch.draw;
        batch.vertexBufferAddress = cachedBatch.vertexBufferAddress;
        batch.instanceCount = cachedBatch.instanceCount;
        batch.firstInstance = 0;
        batches.push_back(batch);
      }
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
          "OpaqueLayer::buildRenderPasses: invalid fast-path submesh");
    }
    const uint32_t requestedLod =
        settings.opaque.enableMeshLod ? forcedLod : 0u;
    const auto lodIndex =
        resolveAvailableLod(*templateEntry.submesh, requestedLod);
    if (lodIndex) {
      const SubmeshLod &lodRange = templateEntry.submesh->lods[*lodIndex];
      appendBatch(baseDraw.pipeline, templateEntry.indexBuffer,
                  templateEntry.indexBufferOffset, lodRange,
                  templateEntry.vertexBufferAddress, instanceCount, 0);
      remapCount = instanceCount;
      templateBatchIndices_.clear();
      templateBatchIndices_.resize(meshDrawTemplates_.size(), 0u);
      usedUniformFastPath = true;
    }
    NURI_PROFILER_ZONE_END();
  }

  if (!usedUniformFastPath) {
    templateBatchIndices_.clear();
    templateBatchIndices_.resize(meshDrawTemplates_.size(), kInvalidBatchIndex);
    NURI_PROFILER_ZONE("OpaqueLayer.batch_build", NURI_PROFILER_COLOR_CMD_DRAW);
    for (size_t templateIndex = 0; templateIndex < meshDrawTemplates_.size();
         ++templateIndex) {
      MeshDrawTemplate &templateEntry = meshDrawTemplates_[templateIndex];
      if (!templateEntry.renderable || !templateEntry.submesh) {
        return Result<bool, std::string>::makeError(
            "OpaqueLayer::buildRenderPasses: invalid mesh template");
      }

      uint32_t requestedLod = 0;
      if (!settings.opaque.enableMeshLod) {
        requestedLod = 0;
      } else if (settings.opaque.forcedMeshLod >= 0) {
        requestedLod = forcedLod;
      } else {
        if (templateEntry.instanceIndex >= instanceAutoLodLevels_.size()) {
          return Result<bool, std::string>::makeError(
              "OpaqueLayer::buildRenderPasses: instance LOD cache out of "
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

      RenderPipelineHandle selectedPipeline = baseDraw.pipeline;
      if (tessellationRequested && *lodIndex == 0 &&
          templateEntry.instanceIndex < instanceLodCentersInvRadiusSq_.size()) {
        const glm::vec4 centerInvRadiusSq =
            instanceLodCentersInvRadiusSq_[templateEntry.instanceIndex];
        const float dx = cameraPosition.x - centerInvRadiusSq.x;
        const float dy = cameraPosition.y - centerInvRadiusSq.y;
        const float dz = cameraPosition.z - centerInvRadiusSq.z;
        const float distanceSq = dx * dx + dy * dy + dz * dz;
        if (distanceSq <= tessFarDistanceSq) {
          selectedPipeline = meshTessPipelineHandle_;
        }
      }

      const BatchKey key{
          .pipeline = selectedPipeline,
          .indexBuffer = templateEntry.indexBuffer,
          .indexBufferOffset = templateEntry.indexBufferOffset,
          .indexCount = lodRange.indexCount,
          .firstIndex = lodRange.indexOffset,
          .vertexBufferAddress = templateEntry.vertexBufferAddress,
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
        batches.push_back(std::move(entry));
        const size_t insertedIndex = batches.size() - 1;
        batchLookup.emplace(key, insertedIndex);
        it = batchLookup.find(key);
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

  {
    NURI_PROFILER_ZONE("OpaqueLayer.draw_list_emit",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    const bool shouldReuseFastAutoLodRemap =
        usedUniformAutoLodFastPath && reusedUniformAutoLodFastPath;
    const bool shouldBuildRemap = !shouldReuseFastAutoLodRemap;
    if (shouldBuildRemap) {
      instanceRemap_.clear();
      instanceRemap_.resize(remapCount);
    } else if (instanceRemap_.size() != remapCount) {
      return Result<bool, std::string>::makeError(
          "OpaqueLayer::buildRenderPasses: auto-LOD remap reuse mismatch");
    }
    drawItems_.clear();
    drawPushConstants_.clear();
    drawItems_.reserve(batches.size());
    drawPushConstants_.reserve(batches.size());
    batchWriteOffsets_.clear();
    batchWriteOffsets_.resize(batches.size(), 0);

    const bool singleRenderableInstance = instanceCount == 1;
    if (singleRenderableInstance) {
      size_t singleRemapCount = 0;
      for (size_t batchIndex = 0; batchIndex < batches.size(); ++batchIndex) {
        BatchEntry &batch = batches[batchIndex];
        batch.firstInstance = 0;
        batchWriteOffsets_[batchIndex] = 0;
        singleRemapCount = std::max(singleRemapCount, batch.instanceCount);
      }
      remapCount = singleRemapCount;
    } else {
      size_t firstInstance = 0;
      for (size_t batchIndex = 0; batchIndex < batches.size(); ++batchIndex) {
        BatchEntry &batch = batches[batchIndex];
        batch.firstInstance = firstInstance;
        batchWriteOffsets_[batchIndex] = firstInstance;
        firstInstance += batch.instanceCount;
      }
    }

    if (shouldBuildRemap && usedUniformAutoLodFastPath) {
      for (uint32_t lod = 0; lod < Submesh::kMaxLodCount; ++lod) {
        autoLodBucketWrites[lod] = autoLodBucketStarts[lod];
      }
      autoLodTessBucketWrite = autoLodTessBucketStart;
      for (uint32_t instanceId = 0; instanceId < instanceCount; ++instanceId) {
        const uint32_t lod = instanceAutoLodLevels_[instanceId];
        if (lod >= Submesh::kMaxLodCount) {
          continue;
        }
        if (usedUniformAutoLodTessSplit && lod == 0 &&
            instanceId < instanceTessSelection_.size() &&
            instanceTessSelection_[instanceId] != 0u) {
          instanceRemap_[autoLodTessBucketWrite++] = instanceId;
          continue;
        }
        const size_t writeOffset = autoLodBucketWrites[lod]++;
        instanceRemap_[writeOffset] = instanceId;
      }
    } else if (shouldBuildRemap && singleRenderableInstance) {
      std::fill(instanceRemap_.begin(), instanceRemap_.end(), 0u);
    } else if (shouldBuildRemap && usedUniformFastPath) {
      for (uint32_t instanceId = 0; instanceId < instanceCount; ++instanceId) {
        instanceRemap_[instanceId] = instanceId;
      }
    } else if (shouldBuildRemap) {
      for (size_t templateIndex = 0; templateIndex < meshDrawTemplates_.size();
           ++templateIndex) {
        const uint32_t batchIndex = templateBatchIndices_[templateIndex];
        if (batchIndex == kInvalidBatchIndex) {
          continue;
        }
        const size_t writeOffset = batchWriteOffsets_[batchIndex]++;
        instanceRemap_[writeOffset] =
            meshDrawTemplates_[templateIndex].instanceIndex;
      }
    }

    for (const BatchEntry &batch : batches) {
      PushConstants constants{};
      constants.frameDataAddress = frameDataAddress;
      constants.vertexBufferAddress = batch.vertexBufferAddress;
      constants.instanceMatricesAddress = instanceMatricesAddress;
      constants.instanceMetaAddress = instanceMetaAddress;
      constants.instanceCentersPhaseAddress = instanceCentersPhaseAddress;
      constants.instanceBaseMatricesAddress = instanceBaseMatricesAddress;
      constants.instanceCount = static_cast<uint32_t>(instanceCount);
      constants.timeSeconds = settings.opaque.enableInstanceAnimation
                                  ? static_cast<float>(frame.timeSeconds)
                                  : 0.0f;
      constants.tessNearDistance = tessNearDistance;
      constants.tessFarDistance = tessFarDistance;
      constants.tessMinFactor = tessMinFactor;
      constants.tessMaxFactor = tessMaxFactor;
      constants.debugVisualizationMode = debugVisualizationMode;
      drawPushConstants_.push_back(constants);

      DrawItem draw = batch.draw;
      draw.instanceCount = static_cast<uint32_t>(batch.instanceCount);
      draw.firstInstance = static_cast<uint32_t>(batch.firstInstance);
      draw.pushConstants = std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&drawPushConstants_.back()),
          sizeof(PushConstants));
      drawItems_.push_back(draw);
    }
    NURI_PROFILER_ZONE_END();
  }

  if (usedUniformAutoLodFastPath && !tessellationRequested) {
    updateFastAutoLodCache(activeFastAutoLodSubmesh, cameraPosition,
                           sortedLodThresholds, autoLodBucketCounts, remapCount,
                           instanceCount);
  } else {
    invalidateAutoLodCache();
  }

  const uint64_t instanceRemapAddress = gpu_.getBufferDeviceAddress(
      instanceRemapRing_[frameSlot].buffer->handle());
  if (instanceRemapAddress == 0) {
    return Result<bool, std::string>::makeError(
        "OpaqueLayer::buildRenderPasses: invalid instance remap buffer "
        "address");
  }

  if (!instanceRemap_.empty()) {
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
    NURI_PROFILER_ZONE_END();
  }

  for (PushConstants &constants : drawPushConstants_) {
    constants.instanceRemapAddress = instanceRemapAddress;
  }

  auto indirectBuildResult = buildIndirectDraws(frameSlot, remapCount);
  if (indirectBuildResult.hasError()) {
    return indirectBuildResult;
  }

  computePushConstants_ = PushConstants{
      .frameDataAddress = frameDataAddress,
      .vertexBufferAddress = 0,
      .instanceMatricesAddress = instanceMatricesAddress,
      .instanceRemapAddress = instanceRemapAddress,
      .instanceMetaAddress = instanceMetaAddress,
      .instanceCentersPhaseAddress = instanceCentersPhaseAddress,
      .instanceBaseMatricesAddress = instanceBaseMatricesAddress,
      .instanceCount = static_cast<uint32_t>(instanceCount),
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
          "OpaqueLayer::buildRenderPasses(pass)");
      if (depResult.hasError()) {
        return depResult;
      }
    }
    if (instanceMetaBuffer_ && instanceMetaBuffer_->valid()) {
      auto depResult = appendUniqueDependency(
          passDependencyBuffers_, instanceMetaBuffer_->handle(),
          "OpaqueLayer::buildRenderPasses(pass)");
      if (depResult.hasError()) {
        return depResult;
      }
    }
    if (frameSlot < instanceRemapRing_.size() && instanceRemapRing_[frameSlot].buffer &&
        instanceRemapRing_[frameSlot].buffer->valid()) {
      auto depResult = appendUniqueDependency(
          passDependencyBuffers_,
          instanceRemapRing_[frameSlot].buffer->handle(),
          "OpaqueLayer::buildRenderPasses(pass)");
      if (depResult.hasError()) {
        return depResult;
      }
    }

    if (hasIndirectDraws) {
      auto depResult = appendUniqueDependency(
          passDependencyBuffers_,
          indirectCommandRing_[frameSlot].buffer->handle(),
          "OpaqueLayer::buildRenderPasses(pass)");
      if (depResult.hasError()) {
        return depResult;
      }
    }

    if (instanceCount > 0) {
      auto passDepResult = appendUniqueDependency(
          passDependencyBuffers_,
          instanceMatricesRing_[frameSlot].buffer->handle(),
          "OpaqueLayer::buildRenderPasses(pass)");
      if (passDepResult.hasError()) {
        return passDepResult;
      }

      if (useComputePass) {
        if (instanceCentersPhaseBuffer_ && instanceCentersPhaseBuffer_->valid()) {
          auto depResult = appendUniqueDependency(
              dispatchDependencyBuffers_, instanceCentersPhaseBuffer_->handle(),
              "OpaqueLayer::buildRenderPasses(dispatch)");
          if (depResult.hasError()) {
            return depResult;
          }
        }
        if (instanceBaseMatricesBuffer_ &&
            instanceBaseMatricesBuffer_->valid()) {
          auto depResult = appendUniqueDependency(
              dispatchDependencyBuffers_, instanceBaseMatricesBuffer_->handle(),
              "OpaqueLayer::buildRenderPasses(dispatch)");
          if (depResult.hasError()) {
            return depResult;
          }
        }
        auto dispatchDepResult = appendUniqueDependency(
            dispatchDependencyBuffers_,
            instanceMatricesRing_[frameSlot].buffer->handle(),
            "OpaqueLayer::buildRenderPasses(dispatch)");
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
        dispatch.dependencyBuffers = std::span<const BufferHandle>(
            dispatchDependencyBuffers_.data(),
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
  if (nuri::isValid(meshTessPipelineHandle_)) {
    for (const DrawItem &draw : drawItems_) {
      if (isTessPipelineHandle(draw.pipeline, meshTessPipelineHandle_)) {
        ++tessellatedDraws;
        tessellatedInstances += draw.instanceCount;
      }
    }
  }

  size_t debugOverlayDraws = 0;
  size_t debugOverlayFallbackDraws = 0;
  size_t debugPatchHeatmapDraws = 0;
  overlayDrawItems_.clear();
  passDrawItems_.clear();
  if (overlayRequested && !drawItems_.empty()) {
    bool gsOverlayAvailable = false;
    bool gsTessOverlayAvailable = false;
    bool lineOverlayAvailable = false;
    bool lineTessOverlayAvailable = false;

    auto gsOverlayResult = ensureGsOverlayPipeline();
    if (gsOverlayResult.hasError()) {
      if (!loggedGsOverlayUnsupported_) {
        loggedGsOverlayUnsupported_ = true;
        NURI_LOG_WARNING("OpaqueLayer::buildRenderPasses: failed to create "
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
        NURI_LOG_WARNING("OpaqueLayer::buildRenderPasses: failed to create "
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
          NURI_LOG_WARNING("OpaqueLayer::buildRenderPasses: failed to create "
                           "line overlay fallback pipeline: %s",
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
          NURI_LOG_WARNING("OpaqueLayer::buildRenderPasses: failed to create "
                           "line tess overlay fallback pipeline: %s",
                           lineTessResult.error().c_str());
        }
      } else {
        lineTessOverlayAvailable = lineTessResult.value();
      }
    }

    overlayDrawItems_.reserve(drawItems_.size());
    for (const DrawItem &baseItem : drawItems_) {
      const bool isTessDraw =
          isTessPipelineHandle(baseItem.pipeline, meshTessPipelineHandle_);
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

  const std::span<const DrawItem> baseDrawItems =
      indirectDrawItems_.empty()
          ? std::span<const DrawItem>(drawItems_.data(), drawItems_.size())
          : std::span<const DrawItem>(indirectDrawItems_.data(),
                                      indirectDrawItems_.size());
  passDrawItems_.reserve(baseDrawItems.size() + overlayDrawItems_.size());
  passDrawItems_.insert(passDrawItems_.end(), baseDrawItems.begin(),
                        baseDrawItems.end());
  passDrawItems_.insert(passDrawItems_.end(), overlayDrawItems_.begin(),
                        overlayDrawItems_.end());

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
    NURI_LOG_DEBUG("OpaqueLayer::buildRenderPasses: totalInstances=%zu "
                   "visibleInstances=%zu "
                   "instancedDraws=%zu tessellatedDraws=%zu "
                   "tessellatedInstances=%zu indirectDrawCalls=%zu "
                   "indirectCommands=%zu "
                   "overlayDraws=%zu "
                   "overlayFallbackDraws=%zu",
                   instanceCount, remapCount, drawItems_.size(),
                   tessellatedDraws, tessellatedInstances,
                   indirectDrawItems_.size(), indirectCommandCount,
                   debugOverlayDraws,
                   debugOverlayFallbackDraws);
  }

  const bool shouldLoadColor = settings.skybox.enabled;
  RenderPass pass{};
  pass.color = {.loadOp = shouldLoadColor ? LoadOp::Load : LoadOp::Clear,
                .storeOp = StoreOp::Store,
                .clearColor = {kClearColorWhite, kClearColorWhite,
                               kClearColorWhite, kClearColorWhite}};
  pass.depth = {.loadOp = LoadOp::Clear,
                .storeOp = StoreOp::Store,
                .clearDepth = kClearDepthOne,
                .clearStencil = 0};
  pass.depthTexture = depthTexture_;
  pass.preDispatches = std::span<const ComputeDispatchItem>(
      preDispatches_.data(), preDispatches_.size());
  pass.dependencyBuffers = std::span<const BufferHandle>(
      passDependencyBuffers_.data(), passDependencyBuffers_.size());
  pass.draws = std::span<const DrawItem>(passDrawItems_.data(),
                                         passDrawItems_.size());
  pass.debugLabel = "Opaque Pass";
  pass.debugColor = kOpaquePassDebugColor;

  frame.sharedDepthTexture = depthTexture_;
  out.push_back(pass);
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> OpaqueLayer::buildIndirectDraws(uint32_t frameSlot,
                                                           size_t remapCount) {
  const bool canUseIndirectPath =
      !drawItems_.empty() &&
      drawItems_.size() <= kMaxDrawItemsForIndirectPath;
  if (!canUseIndirectPath) {
    invalidateIndirectPackCache();
    indirectPushConstants_.clear();
    indirectDrawItems_.clear();
    indirectCommandUploadBytes_.clear();
    return Result<bool, std::string>::makeResult(true);
  }

  NURI_PROFILER_ZONE("OpaqueLayer.indirect_pack",
                     NURI_PROFILER_COLOR_CMD_DRAW);
  if (drawItems_.size() != drawPushConstants_.size()) {
    return Result<bool, std::string>::makeError(
        "OpaqueLayer::buildIndirectDraws: draw and push constant count "
        "mismatch");
  }

  const uint64_t drawSignature = computeIndirectDrawSignature(remapCount);
  const bool canReusePackedCommands = canReuseIndirectPack(drawSignature);

  const size_t requiredBytes = canReusePackedCommands
                                   ? indirectPackCache_.requiredBytes
                                   : kIndirectCountHeaderBytes;
  auto indirectCapacityResult =
      ensureIndirectCommandRingCapacity(std::max(requiredBytes, kIndirectCountHeaderBytes));
  if (indirectCapacityResult.hasError()) {
    return indirectCapacityResult;
  }
  if (frameSlot >= indirectCommandRing_.size() ||
      !indirectCommandRing_[frameSlot].buffer ||
      !indirectCommandRing_[frameSlot].buffer->valid()) {
    return Result<bool, std::string>::makeError(
        "OpaqueLayer::buildIndirectDraws: indirect ring buffer slot is "
        "invalid");
  }

  if (indirectUploadSignatures_.size() != indirectCommandRing_.size()) {
    indirectUploadSignatures_.assign(indirectCommandRing_.size(),
                                     kInvalidDrawSignature);
  }

  auto packResult = canReusePackedCommands
                        ? refreshCachedIndirectPack(frameSlot, drawSignature)
                        : rebuildIndirectPack(frameSlot, remapCount, drawSignature);
  if (packResult.hasError()) {
    return packResult;
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
    drawSignature = hashCombine64(
        drawSignature, foldHandle(draw.pipeline.index, draw.pipeline.generation));
    drawSignature = hashCombine64(
        drawSignature,
        foldHandle(draw.indexBuffer.index, draw.indexBuffer.generation));
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
  }
  return drawSignature;
}

bool OpaqueLayer::canReuseIndirectPack(uint64_t drawSignature) const {
  return indirectPackCache_.valid &&
         indirectPackCache_.drawSignature == drawSignature &&
         indirectSourceDrawIndices_.size() == indirectDrawItems_.size() &&
         indirectSourceDrawIndices_.size() == indirectPushConstants_.size() &&
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
    PushConstants constants{};
    size_t sourceDrawIndex = 0;
    std::pmr::vector<DrawIndexedIndirectCommand> commands;

    explicit IndirectGroup(std::pmr::memory_resource *mem) : commands(mem) {}
  };

  std::pmr::vector<IndirectGroup> indirectGroups(scopedScratch.resource());
  indirectGroups.reserve(drawItems_.size());
  std::pmr::unordered_map<IndirectGroupKey, size_t, IndirectGroupKeyHash>
      indirectLookup(scopedScratch.resource());
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
    };

    auto it = indirectLookup.find(key);
    if (it == indirectLookup.end()) {
      indirectGroups.emplace_back(scopedScratch.resource());
      IndirectGroup &group = indirectGroups.back();
      group.baseDraw = draw;
      group.constants = constants;
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
    const size_t chunkCount = (commandCount + (kMaxIndirectCommandsPerDraw - 1u)) /
                              kMaxIndirectCommandsPerDraw;
    totalIndirectDrawItems += chunkCount;
    packedRequiredBytes += chunkCount * kIndirectCountHeaderBytes +
                           commandCount * sizeof(DrawIndexedIndirectCommand);
  }
  packedRequiredBytes = std::max(packedRequiredBytes, kIndirectCountHeaderBytes);

  auto packedCapacityResult = ensureIndirectCommandRingCapacity(packedRequiredBytes);
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

  indirectPushConstants_.clear();
  indirectDrawItems_.clear();
  indirectCommandUploadBytes_.clear();
  indirectSourceDrawIndices_.clear();

  if (!indirectGroups.empty()) {
    indirectCommandUploadBytes_.reserve(packedRequiredBytes);
    indirectPushConstants_.reserve(totalIndirectDrawItems);
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

        indirectPushConstants_.push_back(group.constants);
        DrawItem indirectDraw = group.baseDraw;
        indirectDraw.command = DrawCommandType::IndexedIndirect;
        indirectDraw.indirectBuffer = indirectBufferHandle;
        indirectDraw.indirectBufferOffset = chunkOffset + kIndirectCountHeaderBytes;
        indirectDraw.indirectDrawCount = drawCount;
        indirectDraw.indirectStride = sizeof(DrawIndexedIndirectCommand);
        indirectDraw.pushConstants = std::span<const std::byte>(
            reinterpret_cast<const std::byte *>(&indirectPushConstants_.back()),
            sizeof(PushConstants));
        indirectDrawItems_.push_back(indirectDraw);
        indirectSourceDrawIndices_.push_back(group.sourceDrawIndex);

        commandCursor += drawCount;
      }
    }
  }

  const std::span<const std::byte> uploadBytes{indirectCommandUploadBytes_.data(),
                                                indirectCommandUploadBytes_.size()};
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
OpaqueLayer::refreshCachedIndirectPack(uint32_t frameSlot, uint64_t drawSignature) {
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
    indirectPushConstants_[i] = drawPushConstants_[sourceIndex];
    indirectDrawItems_[i].indirectBuffer = indirectBufferHandle;
    indirectDrawItems_[i].pushConstants = std::span<const std::byte>(
        reinterpret_cast<const std::byte *>(&indirectPushConstants_[i]),
        sizeof(PushConstants));
  }

  if (indirectUploadSignatures_[frameSlot] != drawSignature) {
    const std::span<const std::byte> uploadBytes{indirectCommandUploadBytes_.data(),
                                                 indirectCommandUploadBytes_.size()};
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
    computeShader_.reset();
    meshVertexShader_ = {};
    meshTessVertexShader_ = {};
    meshTessControlShader_ = {};
    meshTessEvalShader_ = {};
    meshFragmentShader_ = {};
    meshDebugOverlayGeometryShader_ = {};
    meshDebugOverlayFragmentShader_ = {};
    computeShaderHandle_ = {};
    meshTessPipelineHandle_ = {};
    tessellationUnsupported_ = false;
    return depthResult;
  }

  auto pipelineResult = createPipelines();
  if (pipelineResult.hasError()) {
    resetOverlayPipelineState();
    if (nuri::isValid(meshTessPipelineHandle_)) {
      gpu_.destroyRenderPipeline(meshTessPipelineHandle_);
    }
    meshShader_.reset();
    meshTessShader_.reset();
    meshDebugOverlayShader_.reset();
    computeShader_.reset();
    meshVertexShader_ = {};
    meshTessVertexShader_ = {};
    meshTessControlShader_ = {};
    meshTessEvalShader_ = {};
    meshFragmentShader_ = {};
    meshDebugOverlayGeometryShader_ = {};
    meshDebugOverlayFragmentShader_ = {};
    computeShaderHandle_ = {};
    meshFillPipelineHandle_ = {};
    meshTessPipelineHandle_ = {};
    computePipelineHandle_ = {};
    tessellationUnsupported_ = false;
    destroyDepthTexture();
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
  auto metaResult = ensureInstanceMetaBufferCapacity(sizeof(uint32_t));
  if (metaResult.hasError()) {
    return metaResult;
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
OpaqueLayer::ensureInstanceMetaBufferCapacity(size_t requiredBytes) {
  const size_t requested = std::max(requiredBytes, sizeof(uint32_t));
  if (instanceMetaBuffer_ && instanceMetaBuffer_->valid() &&
      instanceMetaBufferCapacityBytes_ >= requested) {
    return Result<bool, std::string>::makeResult(true);
  }

  if (instanceMetaBuffer_ && instanceMetaBuffer_->valid()) {
    gpu_.waitIdle();
    gpu_.destroyBuffer(instanceMetaBuffer_->handle());
    instanceMetaBuffer_.reset();
    instanceMetaBufferCapacityBytes_ = 0;
  }

  const BufferDesc desc{
      .usage = BufferUsage::Storage,
      .storage = Storage::Device,
      .size = requested,
  };
  auto createResult = Buffer::create(gpu_, desc, "opaque_instance_meta_buffer");
  if (createResult.hasError()) {
    return Result<bool, std::string>::makeError(createResult.error());
  }
  instanceMetaBuffer_ = std::move(createResult.value());
  instanceMetaBufferCapacityBytes_ = requested;
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
OpaqueLayer::rebuildSceneCache(const RenderScene &scene) {
  renderableTemplates_.clear();
  meshDrawTemplates_.clear();

  const std::span<const OpaqueRenderable> renderables =
      scene.opaqueRenderables();
  if (renderables.size() >
      static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    return Result<bool, std::string>::makeError(
        "OpaqueLayer::rebuildSceneCache: renderables count exceeds UINT32_MAX");
  }
  renderableTemplates_.reserve(renderables.size());

  size_t totalMeshDraws = 0;
  for (const OpaqueRenderable &renderable : renderables) {
    if (!renderable.model) {
      return Result<bool, std::string>::makeError(
          "OpaqueLayer::rebuildSceneCache: renderable model is null");
    }
    totalMeshDraws += renderable.model->submeshes().size();
  }
  meshDrawTemplates_.reserve(totalMeshDraws);

  for (uint32_t index = 0; index < static_cast<uint32_t>(renderables.size());
       ++index) {
    const OpaqueRenderable &renderable = renderables[index];
    GeometryAllocationView geometry{};
    if (!gpu_.resolveGeometry(renderable.model->geometryHandle(), geometry)) {
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
        RenderableTemplate{.renderable = &renderable});

    const std::span<const Submesh> submeshes = renderable.model->submeshes();
    for (size_t submeshIndex = 0; submeshIndex < submeshes.size();
         ++submeshIndex) {
      meshDrawTemplates_.push_back(MeshDrawTemplate{
          .renderable = &renderable,
          .submesh = &submeshes[submeshIndex],
          .instanceIndex = index,
          .geometryHandle = renderable.model->geometryHandle(),
          .indexBuffer = geometry.indexBuffer,
          .indexBufferOffset = geometry.indexByteOffset,
          .vertexBufferAddress = vertexBufferAddress,
      });
    }
  }

  cachedScene_ = &scene;
  cachedTopologyVersion_ = scene.topologyVersion();
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
          entry.submesh != first.submesh) {
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

Result<bool, std::string> OpaqueLayer::createShaders() {
  meshShader_ = Shader::create("main", gpu_);
  meshTessShader_ = Shader::create("main_tess", gpu_);
  meshDebugOverlayShader_ = Shader::create("mesh_debug_overlay", gpu_);
  computeShader_ = Shader::create("duck_instances", gpu_);
  if (!meshShader_ || !meshTessShader_ || !computeShader_) {
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
    auto compileResult =
        spec.shader->compileFromFile(shaderPath, spec.stage);
    if (compileResult.hasError()) {
      return Result<bool, std::string>::makeError(compileResult.error());
    }
    *spec.outHandle = compileResult.value();
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
    auto compileResult =
        spec.shader->compileFromFile(shaderPath, spec.stage);
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
    auto compileResult =
        spec.shader->compileFromFile(shaderPath, spec.stage);
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

  const Format depthFormat = nuri::isValid(depthTexture_)
                                 ? gpu_.getTextureFormat(depthTexture_)
                                 : Format::D32_FLOAT;
  const RenderPipelineDesc meshDesc = meshPipelineDesc(
      gpu_.getSwapchainFormat(), depthFormat, meshVertexShader_, {}, {},
      {}, meshFragmentShader_, PolygonMode::Fill);
  auto meshResult =
      meshPipeline_->createRenderPipeline(meshDesc, "opaque_mesh");
  if (meshResult.hasError()) {
    return Result<bool, std::string>::makeError(meshResult.error());
  }
  meshFillPipelineHandle_ = meshPipeline_->getRenderPipeline();
  meshTessPipelineHandle_ = {};

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
  } else {
    tessellationUnsupported_ = true;
  }

  const ComputePipelineDesc computeDesc{
      .computeShader = computeShaderHandle_,
  };
  auto computeResult = computePipeline_->createComputePipeline(
      computeDesc, "opaque_instance_compute");
  if (computeResult.hasError()) {
    if (nuri::isValid(meshTessPipelineHandle_)) {
      gpu_.destroyRenderPipeline(meshTessPipelineHandle_);
      meshTessPipelineHandle_ = {};
    }
    return Result<bool, std::string>::makeError(computeResult.error());
  }
  computePipelineHandle_ = computePipeline_->getComputePipeline();

  baseMeshFillDraw_ = makeBaseMeshDraw(meshFillPipelineHandle_, "OpaqueMesh");
  resetOverlayPipelineState();

  return Result<bool, std::string>::makeResult(true);
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
      gpu_.getSwapchainFormat(), depthFormat, meshVertexShader_, {}, {},
      {}, meshFragmentShader_, PolygonMode::Line, Topology::Triangle, 0, true);

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

  auto pipelineResult = gpu_.createRenderPipeline(
      wireframeDesc, "opaque_mesh_tess_wireframe");
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
  const RenderPipelineDesc overlayDesc = meshPipelineDesc(
      gpu_.getSwapchainFormat(), depthFormat, meshTessVertexShader_,
      meshTessControlShader_, meshTessEvalShader_, meshDebugOverlayGeometryShader_,
      meshDebugOverlayFragmentShader_, PolygonMode::Fill, Topology::Patch,
      kTessellationPatchControlPoints, true);

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
  autoLodCache_.bucketCounts.fill(0);
}

void OpaqueLayer::invalidateSingleInstanceBatchCache() {
  singleInstanceBatchCache_.valid = false;
  singleInstanceBatchCache_.requestedLod = 0;
  singleInstanceBatchCache_.tessPipelineEnabled = false;
  singleInstanceBatchCache_.templateSignature = std::numeric_limits<uint64_t>::max();
  singleInstanceBatchCache_.remapCount = 0;
  singleInstanceBatchCache_.batches.clear();
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

void OpaqueLayer::updateFastAutoLodCache(
    const Submesh *submesh, const glm::vec3 &cameraPosition,
    const std::array<float, 3> &sortedLodThresholds,
    const std::array<size_t, Submesh::kMaxLodCount> &bucketCounts,
    size_t remapCount, size_t instanceCount) {
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
}

void OpaqueLayer::destroyDepthTexture() {
  if (nuri::isValid(depthTexture_)) {
    gpu_.destroyTexture(depthTexture_);
    depthTexture_ = TextureHandle{};
  }
}

void OpaqueLayer::destroyBuffers() {
  if (frameDataBuffer_ && frameDataBuffer_->valid()) {
    gpu_.destroyBuffer(frameDataBuffer_->handle());
  }
  frameDataBuffer_.reset();
  frameDataBufferCapacityBytes_ = 0;

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

  if (instanceMetaBuffer_ && instanceMetaBuffer_->valid()) {
    gpu_.destroyBuffer(instanceMetaBuffer_->handle());
  }
  instanceMetaBuffer_.reset();
  instanceMetaBufferCapacityBytes_ = 0;

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
  indirectUploadSignatures_.clear();
  invalidateIndirectPackCache();
}

} // namespace nuri
