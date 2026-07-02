#include "nuri/pch.h"

#include "nuri/gfx/renderers/transmission_renderer.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/renderers/detail/renderable_material_resolution.h"
#include "nuri/gfx/shader.h"
#include "nuri/resources/gpu/resource_manager.h"

namespace nuri {
namespace {

constexpr uint32_t kTransmissionPassDebugColor = 0x33ffaaeeu;
constexpr uint32_t kTransmissionMeshDebugColor = 0x33ffaaeeu;
constexpr std::string_view kTransmissionPassLabel = "Transmission Pass";
constexpr std::string_view kTransmissionMeshLabel = "TransmissionMesh";
constexpr uint64_t kFnvOffsetBasis64 = 14695981039346656037ull;
constexpr uint64_t kFnvPrime64 = 1099511628211ull;
constexpr float kDefaultTaaCurrentFrameWeight = 0.045f;
constexpr uint32_t kForwardSceneHasSceneColor = 1u << 5u;

[[nodiscard]] std::pmr::memory_resource *
resolveMemoryResource(std::pmr::memory_resource *memory) {
  return memory != nullptr ? memory : std::pmr::get_default_resource();
}

[[nodiscard]] const RenderSettings &
settingsOrDefault(const RenderFrameContext &frame) {
  static const RenderSettings kDefaultSettings{};
  return frame.settings ? *frame.settings : kDefaultSettings;
}

[[nodiscard]] bool
isAntiAliasingDebugOutputView(AntiAliasingDebugView view) noexcept {
  return view != AntiAliasingDebugView::None &&
         view != AntiAliasingDebugView::Settings;
}

[[nodiscard]] bool shouldSuppressTransmissionForAntiAliasingDebugView(
    AntiAliasingDebugView view) noexcept {
  return isAntiAliasingDebugOutputView(view) &&
         view != AntiAliasingDebugView::TAATransmissionMipSource;
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

bool animationOverrideCoversSubmesh(
    const AnimatedRenderableGeometryOverride &geometryOverride,
    const Submesh &submesh) noexcept {
  const uint64_t requiredVertexCount =
      static_cast<uint64_t>(submesh.vertexOffset) + submesh.vertexCount;
  return static_cast<uint64_t>(geometryOverride.vertexCount) >=
         requiredVertexCount;
}

[[nodiscard]] uint64_t foldHandle(uint32_t index, uint32_t generation) {
  return (static_cast<uint64_t>(generation) << 32u) | index;
}

[[nodiscard]] uint64_t hashCombine64(uint64_t hash, uint64_t value) {
  hash ^= value;
  hash *= kFnvPrime64;
  return hash;
}

[[nodiscard]] bool isSameTextureHandle(TextureHandle lhs,
                                       TextureHandle rhs) noexcept {
  return lhs.index == rhs.index && lhs.generation == rhs.generation;
}

[[nodiscard]] bool isSameBufferHandle(BufferHandle lhs,
                                      BufferHandle rhs) noexcept {
  return lhs.index == rhs.index && lhs.generation == rhs.generation;
}

[[nodiscard]] bool isTransmissionMaterial(const MaterialRecord &material) {
  return (material.desc.featureMask & kMaterialFeatureTransmission) != 0u;
}

[[nodiscard]] bool
usesSortedTransmissionFeedback(const MaterialRecord &material) {
  return isTransmissionMaterial(material) &&
         material.desc.alphaMode != MaterialAlphaMode::Mask;
}

[[nodiscard]] float
transmissionSortDepth(const glm::mat4 &view,
                      const std::array<glm::vec3, 8> &worldCorners) {
  float depth = std::numeric_limits<float>::lowest();
  for (const glm::vec3 &corner : worldCorners) {
    const glm::vec4 viewPos = view * glm::vec4(corner, 1.0f);
    depth = std::max(depth, -viewPos.z);
  }
  return std::isfinite(depth) ? depth : 0.0f;
}

[[nodiscard]] std::array<glm::vec3, 8>
transmissionWorldBoundsCorners(const glm::mat4 &model,
                               const BoundingBox &bounds) {
  const glm::vec3 min = bounds.min_;
  const glm::vec3 max = bounds.max_;
  const std::array<glm::vec3, 8> localCorners = {
      glm::vec3(min.x, min.y, min.z), glm::vec3(min.x, max.y, min.z),
      glm::vec3(min.x, min.y, max.z), glm::vec3(min.x, max.y, max.z),
      glm::vec3(max.x, min.y, min.z), glm::vec3(max.x, max.y, min.z),
      glm::vec3(max.x, min.y, max.z), glm::vec3(max.x, max.y, max.z),
  };
  std::array<glm::vec3, 8> worldCorners{};
  for (size_t i = 0; i < localCorners.size(); ++i) {
    worldCorners[i] = glm::vec3(model * glm::vec4(localCorners[i], 1.0f));
  }
  return worldCorners;
}

[[nodiscard]] uint32_t levelDimension(uint32_t value, uint32_t mipLevel) {
  return std::max(1u, value >> mipLevel);
}

[[nodiscard]] TextureDesc makeTransparentFeedbackTextureDesc(uint32_t width,
                                                             uint32_t height) {
  return TextureDesc{
      .type = TextureType::Texture2D,
      .format = kFrameCompositionFrameColorFormat,
      .dimensions = {.width = width, .height = height, .depth = 1u},
      .usage = TextureUsage::AttachmentSampled,
      .storage = Storage::Device,
      .numLayers = 1u,
      .numSamples = 1u,
      .numMipLevels = 1u,
      .data = {},
      .dataNumMipLevels = 1u,
      .generateMipmaps = false,
  };
}

[[nodiscard]] glm::vec3 transmissionScaleForDraw(const Renderable &renderable,
                                                 const Submesh &submesh) {
  const glm::mat4 &model = renderable.modelMatrix;
  const glm::vec3 authoredScale = glm::abs(submesh.authoredScale);
  return glm::vec3(
      std::max(glm::length(glm::vec3(model[0])) * authoredScale.x, 1.0e-4f),
      std::max(glm::length(glm::vec3(model[1])) * authoredScale.y, 1.0e-4f),
      std::max(glm::length(glm::vec3(model[2])) * authoredScale.z, 1.0e-4f));
}

[[nodiscard]] float smoothStep(float edge0, float edge1, float value) noexcept {
  if (edge1 <= edge0) {
    return value >= edge1 ? 1.0f : 0.0f;
  }
  const float t = std::clamp((value - edge0) / (edge1 - edge0), 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}

[[nodiscard]] float
transmissionTaaJitterMinLod(const RenderFrameContext &frame,
                            const RenderSettings &settings) noexcept {
  if (!settings.transmission.taaJitterPrefilter ||
      sanitizeAntiAliasingMode(settings.antiAliasing.mode) !=
          AntiAliasingMode::TAA ||
      !frame.camera.jitterEnabled) {
    return 0.0f;
  }

  const RenderSettings::AntiAliasingDebugSettings aaDebug =
      effectiveTemporalAADebugSettings(settings.antiAliasing);
  const float jitterScale = std::isfinite(aaDebug.taaJitterScale)
                                ? std::clamp(aaDebug.taaJitterScale, 0.0f, 1.0f)
                                : 0.75f;
  const float currentWeight =
      std::isfinite(aaDebug.taaCurrentFrameWeight)
          ? std::clamp(aaDebug.taaCurrentFrameWeight, 0.0f, 1.0f)
          : kDefaultTaaCurrentFrameWeight;
  const float maxLod =
      std::isfinite(settings.transmission.taaJitterPrefilterMaxLod)
          ? std::clamp(settings.transmission.taaJitterPrefilterMaxLod, 0.0f,
                       2.0f)
          : 1.0f;
  const float jitterFactor = smoothStep(0.05f, 0.50f, jitterScale);
  const float weightFactor =
      smoothStep(0.0f, kDefaultTaaCurrentFrameWeight, currentWeight);
  return std::clamp(maxLod * jitterFactor * weightFactor, 0.0f, maxLod);
}

[[nodiscard]] float transmissionTaaJitterDepthBiasConstant(
    const RenderSettings &settings) noexcept {
  if (!std::isfinite(settings.transmission.taaJitterDepthBiasConstant)) {
    return -8.0f;
  }
  return std::clamp(settings.transmission.taaJitterDepthBiasConstant, -64.0f,
                    0.0f);
}

[[nodiscard]] bool transmissionUsesJitteredPostTaaDepthBias(
    const RenderFrameContext &frame, const RenderSettings &settings,
    uint64_t frameDataAddress, const ForwardSceneGpuData &sceneGpu) noexcept {
  return frame.camera.jitterEnabled &&
         sanitizeAntiAliasingMode(settings.antiAliasing.mode) ==
             AntiAliasingMode::TAA &&
         frameDataAddress != 0u && sceneGpu.postTaaFrameDataAddress != 0u &&
         frameDataAddress == sceneGpu.postTaaFrameDataAddress &&
         sceneGpu.postTaaFrameDataAddress != sceneGpu.frameDataAddress;
}

Result<ForwardSceneFrameData, std::string>
resolveForwardSceneFrameData(const ForwardSceneGpuData &sceneGpu,
                             uint64_t frameDataAddress) {
  if (!nuri::isValid(sceneGpu.buffer) || frameDataAddress == 0u) {
    return Result<ForwardSceneFrameData, std::string>::makeError(
        "TransmissionRenderer::prepareTransmissionPasses: invalid forward "
        "scene frame data source");
  }
  if (frameDataAddress == sceneGpu.postTaaFrameDataAddress &&
      sceneGpu.postTaaFrameDataAddress != 0u) {
    return Result<ForwardSceneFrameData, std::string>::makeResult(
        sceneGpu.postTaaFrameData);
  }
  if (frameDataAddress == sceneGpu.frameDataAddress &&
      sceneGpu.frameDataAddress != 0u) {
    return Result<ForwardSceneFrameData, std::string>::makeResult(
        sceneGpu.frameData);
  }
  return Result<ForwardSceneFrameData, std::string>::makeError(
      "TransmissionRenderer::prepareTransmissionPasses: unknown forward "
      "scene frame data address");
}

void appendUniqueTexture(std::pmr::vector<TextureHandle> &handles,
                         TextureHandle handle) {
  if (!nuri::isValid(handle)) {
    return;
  }
  if (std::find_if(handles.begin(), handles.end(),
                   [handle](const TextureHandle existing) {
                     return existing.index == handle.index &&
                            existing.generation == handle.generation;
                   }) != handles.end()) {
    return;
  }
  handles.push_back(handle);
}

[[nodiscard]] uint64_t
textureReadSignature(std::span<const TextureHandle> staticHandles,
                     std::span<const TextureHandle> dynamicHandles) {
  uint64_t hash = hashCombine64(kFnvOffsetBasis64, staticHandles.size());
  for (const TextureHandle handle : staticHandles) {
    if (!nuri::isValid(handle)) {
      continue;
    }
    hash = hashCombine64(hash, foldHandle(handle.index, handle.generation));
  }
  hash = hashCombine64(hash, 0x9e3779b97f4a7c15ull);
  hash = hashCombine64(hash, dynamicHandles.size());
  for (const TextureHandle handle : dynamicHandles) {
    if (!nuri::isValid(handle)) {
      continue;
    }
    hash = hashCombine64(hash, foldHandle(handle.index, handle.generation));
  }
  return hash;
}

void rebuildTextureReads(std::pmr::vector<TextureHandle> &out,
                         std::span<const TextureHandle> staticHandles,
                         std::span<const TextureHandle> dynamicHandles) {
  out.clear();
  out.reserve(staticHandles.size() + dynamicHandles.size());
  for (const TextureHandle handle : staticHandles) {
    appendUniqueTexture(out, handle);
  }
  for (const TextureHandle handle : dynamicHandles) {
    appendUniqueTexture(out, handle);
  }
}

[[nodiscard]] uint64_t transmissionDrawLayoutSignature(
    RenderPipelineHandle pipeline, BufferHandle vertexBuffer,
    BufferHandle indexBuffer, uint64_t indexBufferOffset, const SubmeshLod &lod,
    uint32_t instanceIndex, bool hasDepthAttachment, bool depthBiasEnabled,
    uint32_t depthBiasBits) {
  uint64_t hash = hashCombine64(
      kFnvOffsetBasis64, foldHandle(pipeline.index, pipeline.generation));
  hash = hashCombine64(hash,
                       foldHandle(vertexBuffer.index, vertexBuffer.generation));
  hash = hashCombine64(hash,
                       foldHandle(indexBuffer.index, indexBuffer.generation));
  hash = hashCombine64(hash, indexBufferOffset);
  hash = hashCombine64(hash, lod.indexOffset);
  hash = hashCombine64(hash, lod.indexCount);
  hash = hashCombine64(hash, instanceIndex);
  hash = hashCombine64(hash, hasDepthAttachment ? 1u : 0u);
  hash = hashCombine64(hash, depthBiasEnabled ? 1u : 0u);
  return hashCombine64(hash, depthBiasBits);
}

void appendUniqueBuffer(std::pmr::vector<BufferHandle> &handles,
                        BufferHandle handle) {
  if (!nuri::isValid(handle)) {
    return;
  }
  if (std::find_if(handles.begin(), handles.end(),
                   [handle](const BufferHandle existing) {
                     return existing.index == handle.index &&
                            existing.generation == handle.generation;
                   }) != handles.end()) {
    return;
  }
  handles.push_back(handle);
}

void appendPreResolvedDrawBuffers(std::pmr::vector<BufferHandle> &handles,
                                  std::span<const DrawItem> draws) {
  for (const DrawItem &draw : draws) {
    appendUniqueBuffer(handles, draw.vertexBuffer);
    appendUniqueBuffer(handles, draw.indexBuffer);
    appendUniqueBuffer(handles, draw.indirectBuffer);
    appendUniqueBuffer(handles, draw.indirectCountBuffer);
  }
}

[[nodiscard]] bool isSameTextureRef(TextureRef lhs, TextureRef rhs) {
  return lhs == rhs;
}

[[nodiscard]] bool isSameEnvironmentHandles(const EnvironmentHandles &lhs,
                                            const EnvironmentHandles &rhs) {
  return isSameTextureRef(lhs.cubemap, rhs.cubemap) &&
         isSameTextureRef(lhs.irradiance, rhs.irradiance) &&
         isSameTextureRef(lhs.prefilteredGgx, rhs.prefilteredGgx) &&
         isSameTextureRef(lhs.prefilteredCharlie, rhs.prefilteredCharlie) &&
         isSameTextureRef(lhs.brdfLut, rhs.brdfLut);
}

[[nodiscard]] std::optional<SubmeshLod>
resolveTransmissionLod(const Submesh &submesh, const RenderSettings &settings) {
  if (settings.opaque.forcedMeshLod < 0) {
    if (submesh.indexCount > 0) {
      return SubmeshLod{.indexOffset = submesh.indexOffset,
                        .indexCount = submesh.indexCount,
                        .error = 0.0f};
    }
    for (uint32_t lod = 0; lod < std::max(submesh.lodCount, 1u); ++lod) {
      if (submesh.lods[lod].indexCount > 0u) {
        return submesh.lods[lod];
      }
    }
    return std::nullopt;
  }

  const uint32_t lodCount =
      std::clamp(submesh.lodCount, 1u, Submesh::kMaxLodCount);
  uint32_t candidate = std::min(
      static_cast<uint32_t>(settings.opaque.forcedMeshLod), lodCount - 1u);
  while (candidate > 0u && submesh.lods[candidate].indexCount == 0u) {
    --candidate;
  }
  if (submesh.lods[candidate].indexCount > 0u) {
    return submesh.lods[candidate];
  }
  if (submesh.indexCount > 0u) {
    return SubmeshLod{.indexOffset = submesh.indexOffset,
                      .indexCount = submesh.indexCount,
                      .error = 0.0f};
  }
  return std::nullopt;
}

RenderPipelineDesc meshPipelineDesc(Format colorFormat, Format depthFormat,
                                    ShaderHandle vertexShader,
                                    ShaderHandle fragmentShader,
                                    CullMode cullMode, bool blendEnabled) {
  return RenderPipelineDesc{
      .vertexInput = {},
      .vertexShader = vertexShader,
      .fragmentShader = fragmentShader,
      .colorFormats = {colorFormat},
      .depthFormat = depthFormat,
      .cullMode = cullMode,
      .polygonMode = PolygonMode::Fill,
      .topology = Topology::Triangle,
      .blendEnabled = blendEnabled,
  };
}

uint32_t saturateToU32(size_t value) {
  return static_cast<uint32_t>(
      std::min(value, size_t(std::numeric_limits<uint32_t>::max())));
}

} // namespace

TransmissionRenderer::TransmissionRenderer(
    GPUDevice &gpu, const TransmissionRendererConfig &config,
    std::pmr::memory_resource *memory)
    : gpu_(gpu), config_(config), memory_(resolveMemoryResource(memory)),
      instanceMatricesRing_(memory_), instanceRemapRing_(memory_),
      blendedFrameDataRing_(memory_), meshDrawTemplates_(memory_),
      instanceMatrices_(memory_), instanceRemap_(memory_),
      instanceDataRingUploadVersions_(memory_),
      materialTextureAccessHandles_(memory_),
      environmentTextureAccessHandles_(memory_),
      staticPassTextureReads_(memory_), meshPushConstants_(memory_),
      blendedPushConstants_(memory_), passDrawItems_(memory_),
      blendedSortableDraws_(memory_), sortedDepthTemplates_(memory_),
      passTextureReads_(memory_), blendedTextureReads_(memory_),
      passDependencyBuffers_(memory_), blendedDependencyBuffers_(memory_),
      passDependencyBufferAccessModes_(memory_),
      preResolvedTemplateBuffers_(memory_),
      cachedPreResolvedDrawBuffers_(memory_),
      transparentFeedbackTextures_{std::pmr::vector<TextureHandle>(memory_),
                                   std::pmr::vector<TextureHandle>(memory_),
                                   std::pmr::vector<TextureHandle>(memory_)} {
  const std::filesystem::path basePath = config_.meshFragment.parent_path();
  transmissionVertexPath_ = basePath / "transmission.vert";
  transmissionFragmentPath_ = basePath / "transmission.frag";
}

TransmissionRenderer::~TransmissionRenderer() { onDetach(); }

void TransmissionRenderer::onAttach() {
  auto initResult = ensureInitialized();
  if (initResult.hasError()) {
    NURI_LOG_WARNING("TransmissionRenderer::onAttach: %s",
                     initResult.error().c_str());
  }
}

void TransmissionRenderer::onDetach() {
  destroyBuffers();
  destroyFeedbackTextures();
  destroyPipelineState();
  destroyShaders();
  meshShader_.reset();
  resetFrameBuildState();
  resetCachedState();
  initialized_ = false;
}

void TransmissionRenderer::publishFrameData(RenderFrameContext &frame) {
  const RenderSettings &settings = settingsOrDefault(frame);
  if (!settings.transmission.enabled) {
    return;
  }
  frame.sharedResources.transparentTransmissionStageEnabled = true;
  frame.transparentContributors.publish(TransparentContributionCollector{
      .user = this,
      .collect =
          [](void *user, RenderFrameContext &frame,
             TransparentStageContribution &out) -> Result<bool, std::string> {
        return static_cast<TransmissionRenderer *>(user)
            ->buildTransparentStageContribution(frame, out);
      },
  });
}

Result<bool, std::string>
TransmissionRenderer::prepareTransmissionPasses(RenderFrameContext &frame) {
  resetFrameBuildState();
  preparedSceneColorTexture_ = {};
  preparedSceneColorHalfResTexture_ = {};
  preparedSceneColorQuarterResTexture_ = {};
  preparedFrameColorTexture_ = {};
  preparedDepthTexture_ = {};
  preparedSceneDepthGraphTexture_ = {};
  frame.sharedResources.transparentTransmissionFeedbackTexture = {};
  frame.sharedResources.transparentTransmissionFeedbackHalfResTexture = {};
  frame.sharedResources.transparentTransmissionFeedbackQuarterResTexture = {};

  const RenderSettings &settings = settingsOrDefault(frame);
  if (!settings.transmission.enabled) {
    passTextureReads_.clear();
    blendedTextureReads_.clear();
    cachedPassTextureReadSignature_ = std::numeric_limits<uint64_t>::max();
    cachedBlendedTextureReadSignature_ = std::numeric_limits<uint64_t>::max();
    return Result<bool, std::string>::makeResult(true);
  }
  if (!frame.scene) {
    return Result<bool, std::string>::makeError(
        "TransmissionRenderer::buildRenderGraph: frame scene is null");
  }
  if (!frame.resources) {
    return Result<bool, std::string>::makeError(
        "TransmissionRenderer::buildRenderGraph: frame resources are null");
  }

  auto initResult = ensureInitialized();
  if (initResult.hasError()) {
    return initResult;
  }

  if (!nuri::isValid(frame.sharedResources.sceneColorTexture)) {
    return Result<bool, std::string>::makeError(
        "TransmissionRenderer::prepareTransmissionPasses: scene color texture "
        "is "
        "unavailable");
  }
  if (!nuri::isValid(frame.sharedResources.frameColorTexture)) {
    return Result<bool, std::string>::makeError(
        "TransmissionRenderer::prepareTransmissionPasses: frame color texture "
        "is "
        "unavailable");
  }
  const TextureHandle sceneColorTexture =
      frame.sharedResources.sceneColorTexture;
  const TextureHandle sceneColorHalfResTexture =
      frame.sharedResources.sceneColorHalfResTexture;
  const TextureHandle sceneColorQuarterResTexture =
      frame.sharedResources.sceneColorQuarterResTexture;
  const TextureHandle frameColorTexture =
      frame.sharedResources.frameColorTexture;
  const bool stableVisibilityDepth =
      nuri::isValid(frame.sharedResources.transmissionVisibilityDepthTexture);
  const TextureHandle depthTexture =
      stableVisibilityDepth
          ? frame.sharedResources.transmissionVisibilityDepthTexture
          : resolveFrameDepthTexture(frame);
  const RenderGraphTextureId sceneDepthGraphTexture =
      stableVisibilityDepth
          ? frame.sharedResources.transmissionVisibilityDepthGraphTexture
          : resolveSceneDepthGraphTexture(frame);
  frame.metrics.antiAliasing.taaTransmissionStableVisibilityDepth =
      stableVisibilityDepth;

  const MaterialTableSnapshot materialSnapshot =
      frame.resources->materialSnapshot();
  const bool topologyDirty =
      cachedScene_ != frame.scene ||
      cachedTopologyVersion_ != frame.scene->topologyVersion();
  const bool materialDirty =
      topologyDirty || cachedMaterialVersion_ != materialSnapshot.version;
  const bool transformDirty =
      topologyDirty ||
      cachedTransformVersion_ != frame.scene->transformVersion();
  const uint64_t geometryMutationVersion = gpu_.geometryMutationVersion();
  const bool geometryDirty =
      geometryMutationVersion != 0u &&
      geometryMutationVersion != cachedGeometryMutationVersion_;
  const bool needsGeometryRebuild =
      geometryDirty && !meshDrawTemplates_.empty();
  bool drawTemplatesRebuilt = false;
  if (topologyDirty || materialDirty || needsGeometryRebuild) {
    Result<bool, std::string> rebuildResult =
        [&]() -> Result<bool, std::string> {
      std::optional<Result<bool, std::string>> result;
      NURI_PROFILER_ZONE("TransmissionRenderer.cache_rebuild",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      result.emplace(rebuildSceneCache(
          *frame.scene, *frame.resources,
          static_cast<uint32_t>(materialSnapshot.headers.size())));
      NURI_PROFILER_ZONE_END();
      return std::move(*result);
    }();
    if (rebuildResult.hasError()) {
      return rebuildResult;
    }
    drawTemplatesRebuilt = true;
    cachedMaterialVersion_ = materialSnapshot.version;
  } else if (geometryDirty) {
    cachedGeometryMutationVersion_ = geometryMutationVersion;
  }

  const EnvironmentHandles &environment = frame.scene->environment();
  const bool environmentDirty =
      !isSameEnvironmentHandles(cachedEnvironmentHandles_, environment);
  if (environmentDirty || !environmentTextureAccessCacheValid_) {
    NURI_PROFILER_ZONE("TransmissionRenderer.env_texture_collect",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    collectEnvironmentTextureReads(*frame.scene, *frame.resources);
    cachedEnvironmentHandles_ = environment;
    environmentTextureAccessCacheValid_ = true;
    NURI_PROFILER_ZONE_END();
  }
  if (materialDirty || !materialTextureAccessCacheValid_) {
    Result<bool, std::string> cacheResult = [&]() -> Result<bool, std::string> {
      std::optional<Result<bool, std::string>> result;
      NURI_PROFILER_ZONE("TransmissionRenderer.material_texture_cache",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      result.emplace(rebuildMaterialTextureAccessCache(*frame.resources));
      NURI_PROFILER_ZONE_END();
      return std::move(*result);
    }();
    if (cacheResult.hasError()) {
      return cacheResult;
    }
    materialTextureAccessCacheValid_ = true;
  }
  if (environmentDirty || materialDirty || !staticPassTextureReadsValid_) {
    staticPassTextureReads_.clear();
    staticPassTextureReads_.reserve(environmentTextureAccessHandles_.size() +
                                    materialTextureAccessHandles_.size());
    for (const TextureHandle handle : environmentTextureAccessHandles_) {
      appendUniqueTexture(staticPassTextureReads_, handle);
    }
    for (const TextureHandle handle : materialTextureAccessHandles_) {
      appendUniqueTexture(staticPassTextureReads_, handle);
    }
    staticPassTextureReadsValid_ = true;
  }

  if (!frame.sharedResources.forwardSceneGpuData.has_value() ||
      !nuri::isValid(frame.sharedResources.forwardSceneGpuData->buffer) ||
      frame.sharedResources.forwardSceneGpuData->frameDataAddress == 0u) {
    return Result<bool, std::string>::makeError(
        "TransmissionRenderer::prepareTransmissionPasses: forward scene GPU "
        "data is "
        "unavailable");
  }
  if (!frame.sharedResources.materialTableGpuData.has_value()) {
    return Result<bool, std::string>::makeError(
        "TransmissionRenderer::prepareTransmissionPasses: material table GPU "
        "data is unavailable");
  }
  const ForwardSceneGpuData *sceneGpu =
      &*frame.sharedResources.forwardSceneGpuData;
  const MaterialTableGpuData *materialGpu =
      &*frame.sharedResources.materialTableGpuData;
  const AnimationSceneFrameData *animationSceneData =
      resolveAnimationSceneFrameData(frame);
  const uint64_t frameDataAddress = sceneGpu->postTaaFrameDataAddress != 0u
                                        ? sceneGpu->postTaaFrameDataAddress
                                        : sceneGpu->frameDataAddress;
  const bool hasSortedFeedbackDraws = std::any_of(
      meshDrawTemplates_.begin(), meshDrawTemplates_.end(),
      [](const MeshDrawTemplate &entry) { return entry.sortedFeedback; });
  if (hasSortedFeedbackDraws) {
    auto feedbackResult = ensureTransparentFeedbackTextures(frame);
    if (feedbackResult.hasError()) {
      return feedbackResult;
    }
    frame.metrics.antiAliasing.transparentTransmissionFeedbackSourceAvailable =
        nuri::isValid(
            frame.sharedResources.transparentTransmissionFeedbackTexture) &&
                nuri::isValid(
                    frame.sharedResources
                        .transparentTransmissionFeedbackHalfResTexture) &&
                nuri::isValid(
                    frame.sharedResources
                        .transparentTransmissionFeedbackQuarterResTexture)
            ? 1u
            : 0u;
  }
  const bool jitteredPostTaaDepthBias =
      !stableVisibilityDepth && nuri::isValid(depthTexture) &&
      transmissionUsesJitteredPostTaaDepthBias(frame, settings,
                                               frameDataAddress, *sceneGpu);
  const float jitterDepthBiasConstant =
      jitteredPostTaaDepthBias
          ? transmissionTaaJitterDepthBiasConstant(settings)
          : 0.0f;
  frame.metrics.antiAliasing.taaTransmissionDepthBiasConstant =
      jitterDepthBiasConstant;

  const std::span<const Renderable> renderables = frame.scene->renderables();
  if (!meshDrawTemplates_.empty()) {
    auto ringResult =
        ensureRingBufferCount(std::max(1u, gpu_.getSwapchainImageCount()));
    if (ringResult.hasError()) {
      return ringResult;
    }
  }

  const uint32_t frameSlot =
      instanceMatricesRing_.empty()
          ? 0u
          : static_cast<uint32_t>(frame.frameIndex %
                                  instanceMatricesRing_.size());
  uint64_t blendedFrameDataAddress = 0u;
  BufferHandle blendedFrameDataBufferHandle{};
  if (hasSortedFeedbackDraws) {
    auto frameDataRingResult =
        ensureBlendedFrameDataRingCapacity(sizeof(FrameData));
    if (frameDataRingResult.hasError()) {
      return frameDataRingResult;
    }
    const uint32_t blendFrameSlot =
        blendedFrameDataRing_.empty()
            ? 0u
            : static_cast<uint32_t>(frame.frameIndex %
                                    blendedFrameDataRing_.size());
    DynamicBufferSlot &blendSlot = blendedFrameDataRing_[blendFrameSlot];
    if (!blendSlot.buffer || !blendSlot.buffer->valid()) {
      return Result<bool, std::string>::makeError(
          "TransmissionRenderer::prepareTransmissionPasses: blended frame "
          "data buffer is unavailable");
    }
    auto blendedFrameDataResult =
        resolveForwardSceneFrameData(*sceneGpu, frameDataAddress);
    if (blendedFrameDataResult.hasError()) {
      return Result<bool, std::string>::makeError(
          blendedFrameDataResult.error());
    }
    FrameData blendedFrameData = blendedFrameDataResult.value();
    const TextureHandle feedbackFull =
        frame.sharedResources.transparentTransmissionFeedbackTexture;
    const TextureHandle feedbackHalf =
        frame.sharedResources.transparentTransmissionFeedbackHalfResTexture;
    const TextureHandle feedbackQuarter =
        frame.sharedResources.transparentTransmissionFeedbackQuarterResTexture;
    blendedFrameData.sceneColorTexId =
        gpu_.getTextureBindlessIndex(feedbackFull);
    blendedFrameData.sceneColorHalfResTexId =
        gpu_.getTextureBindlessIndex(feedbackHalf);
    blendedFrameData.sceneColorQuarterResTexId =
        gpu_.getTextureBindlessIndex(feedbackQuarter);
    blendedFrameData.sceneColorSamplerId =
        gpu_.getLinearRepeatSamplerBindlessIndex(true, 1u);
    if (blendedFrameData.sceneColorTexId == kInvalidTextureBindlessIndex ||
        blendedFrameData.sceneColorHalfResTexId ==
            kInvalidTextureBindlessIndex ||
        blendedFrameData.sceneColorQuarterResTexId ==
            kInvalidTextureBindlessIndex) {
      return Result<bool, std::string>::makeError(
          "TransmissionRenderer::prepareTransmissionPasses: invalid "
          "transparent transmission feedback texture bindless index");
    }
    blendedFrameData.flags |= kForwardSceneHasSceneColor;
    const std::span<const std::byte> frameDataBytes{
        reinterpret_cast<const std::byte *>(&blendedFrameData),
        sizeof(blendedFrameData)};
    auto updateResult =
        gpu_.updateBuffer(blendSlot.buffer->handle(), frameDataBytes, 0u);
    if (updateResult.hasError()) {
      return updateResult;
    }
    blendedFrameDataBufferHandle = blendSlot.buffer->handle();
    blendedFrameDataAddress =
        gpu_.getBufferDeviceAddress(blendedFrameDataBufferHandle);
  }

  if (!meshDrawTemplates_.empty() && (transformDirty || drawTemplatesRebuilt)) {
    {
      NURI_PROFILER_ZONE("TransmissionRenderer.draw_template_refresh",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      refreshDrawTemplateTransforms(renderables);
      NURI_PROFILER_ZONE_END();
    }
  }
  if (!meshDrawTemplates_.empty() && transformDirty) {
    instanceMatrices_.clear();
    instanceRemap_.clear();
    instanceMatrices_.reserve(renderables.size());
    instanceRemap_.reserve(renderables.size());
    for (uint32_t i = 0; i < static_cast<uint32_t>(renderables.size()); ++i) {
      instanceMatrices_.push_back(makeInstanceData(renderables[i].modelMatrix));
      instanceRemap_.push_back(i);
    }
    cachedTransformVersion_ = frame.scene->transformVersion();
    std::fill(instanceDataRingUploadVersions_.begin(),
              instanceDataRingUploadVersions_.end(),
              std::numeric_limits<uint64_t>::max());
  } else if (meshDrawTemplates_.empty()) {
    cachedTransformVersion_ = frame.scene->transformVersion();
  }

  if (!meshDrawTemplates_.empty()) {
    if (animationSceneData == nullptr) {
      auto matricesBufferResult = ensureInstanceMatricesRingCapacity(
          std::max(instanceMatrices_.size() * sizeof(InstanceData),
                   sizeof(InstanceData)));
      if (matricesBufferResult.hasError()) {
        return matricesBufferResult;
      }
    }
    auto remapBufferResult = ensureInstanceRemapRingCapacity(
        std::max(instanceRemap_.size() * sizeof(uint32_t), sizeof(uint32_t)));
    if (remapBufferResult.hasError()) {
      return remapBufferResult;
    }
  }

  const Format depthFormat = nuri::isValid(depthTexture)
                                 ? gpu_.getTextureFormat(depthTexture)
                                 : Format::Count;
  auto pipelineResult =
      ensurePipelines(gpu_.getTextureFormat(frameColorTexture), depthFormat);
  if (pipelineResult.hasError()) {
    return pipelineResult;
  }

  passDrawItems_.clear();
  blendedSortableDraws_.clear();
  passTextureReads_.clear();
  blendedTextureReads_.clear();
  passDependencyBuffers_.clear();
  blendedDependencyBuffers_.clear();
  meshPushConstants_.clear();
  blendedPushConstants_.clear();

  if (!meshDrawTemplates_.empty()) {
    NURI_PROFILER_ZONE("TransmissionRenderer.material_instance_uploads",
                       NURI_PROFILER_COLOR_CMD_COPY);
    const bool needsInstanceDataUpload =
        animationSceneData == nullptr &&
        instanceDataRingUploadVersions_[frameSlot] != cachedTransformVersion_;
    if (needsInstanceDataUpload && !instanceMatrices_.empty()) {
      const std::span<const std::byte> matrixBytes{
          reinterpret_cast<const std::byte *>(instanceMatrices_.data()),
          instanceMatrices_.size() * sizeof(InstanceData)};
      auto updateResult = gpu_.updateBuffer(
          instanceMatricesRing_[frameSlot].buffer->handle(), matrixBytes, 0);
      if (updateResult.hasError()) {
        return updateResult;
      }
    }
    if (needsInstanceDataUpload && !instanceRemap_.empty()) {
      const std::span<const std::byte> remapBytes{
          reinterpret_cast<const std::byte *>(instanceRemap_.data()),
          instanceRemap_.size() * sizeof(uint32_t)};
      auto updateResult = gpu_.updateBuffer(
          instanceRemapRing_[frameSlot].buffer->handle(), remapBytes, 0);
      if (updateResult.hasError()) {
        return updateResult;
      }
    }
    if (needsInstanceDataUpload) {
      instanceDataRingUploadVersions_[frameSlot] = cachedTransformVersion_;
    }
    NURI_PROFILER_ZONE_END();

    NURI_PROFILER_ZONE("TransmissionRenderer.mesh_draw_build",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    const float taaJitterMinLod = transmissionTaaJitterMinLod(frame, settings);
    frame.metrics.antiAliasing.taaTransmissionJitterMinLod = taaJitterMinLod;
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
    const uint64_t instanceRemapAddress = gpu_.getBufferDeviceAddress(
        instanceRemapRing_[frameSlot].buffer->handle());
    if (frameDataAddress == 0u || materialGpu->headerBufferAddress == 0u ||
        materialGpu->clearcoatBufferAddress == 0u ||
        materialGpu->sheenBufferAddress == 0u ||
        materialGpu->transmissionBufferAddress == 0u ||
        materialGpu->specularBufferAddress == 0u ||
        instanceMatricesAddress == 0u || instanceRemapAddress == 0u ||
        (hasSortedFeedbackDraws &&
         (blendedFrameDataAddress == 0u ||
          !nuri::isValid(blendedFrameDataBufferHandle))) ||
        (sceneGpu->directionalLightCount > 0u &&
         directionalLightBufferAddress == 0u) ||
        (sceneGpu->localLightCount > 0u && localLightBufferAddress == 0u)) {
      return Result<bool, std::string>::makeError(
          "TransmissionRenderer::prepareTransmissionPasses: invalid GPU buffer "
          "address");
    }

    meshPushConstants_.reserve(meshDrawTemplates_.size());
    blendedPushConstants_.reserve(meshDrawTemplates_.size());
    blendedSortableDraws_.reserve(meshDrawTemplates_.size());
    const uint32_t renderableCount = saturateToU32(renderables.size());
    if (instanceRemap_.size() < renderables.size()) {
      return Result<bool, std::string>::makeError(
          "TransmissionRenderer::prepareTransmissionPasses: instance remap "
          "buffer is smaller than the renderable set");
    }
    const bool hasDepthAttachment =
        (nuri::isValid(depthTexture) || nuri::isValid(sceneDepthGraphTexture));
    constexpr uint32_t debugFlags = 0u;
    sortedDepthTemplates_.clear();
    if (hasSortedFeedbackDraws) {
      sortedDepthTemplates_.reserve(meshDrawTemplates_.size());
    }
    uint32_t jitterDepthBiasBits = 0u;
    std::memcpy(&jitterDepthBiasBits, &jitterDepthBiasConstant,
                sizeof(jitterDepthBiasBits));
    NURI_PROFILER_ZONE("TransmissionRenderer.push_constant_patch",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    for (MeshDrawTemplate &entry : meshDrawTemplates_) {
      if (entry.renderable == nullptr || entry.submesh == nullptr) {
        continue;
      }

      const std::optional<SubmeshLod> lod =
          resolveTransmissionLod(*entry.submesh, settings);
      if (!lod.has_value()) {
        continue;
      }
      if (entry.instanceIndex >= renderableCount ||
          entry.instanceIndex >= instanceRemap_.size()) {
        return Result<bool, std::string>::makeError(
            "TransmissionRenderer::prepareTransmissionPasses: draw instance "
            "index is out of range");
      }
      if (entry.materialIndex >= materialSnapshot.headers.size()) {
        return Result<bool, std::string>::makeError(
            "TransmissionRenderer::prepareTransmissionPasses: draw material "
            "index is out of range");
      }
      if (!nuri::isValid(entry.indexBuffer)) {
        return Result<bool, std::string>::makeError(
            "TransmissionRenderer::prepareTransmissionPasses: draw index "
            "buffer is invalid");
      }

      BufferHandle vertexBuffer = nuri::isValid(entry.vertexBuffer)
                                      ? entry.vertexBuffer
                                      : entry.baseVertexBuffer;
      uint64_t vertexBufferAddress =
          isSameBufferHandle(vertexBuffer, entry.vertexBuffer)
              ? entry.vertexBufferAddress
              : gpu_.getBufferDeviceAddress(vertexBuffer,
                                            entry.vertexBufferByteOffset);
      uint64_t vertexDecodeBufferAddress = entry.vertexDecodeBufferAddress;
      uint32_t vertexDecodeIndex = entry.vertexDecodeIndex;
      uint32_t packedVertexFormat = entry.packedVertexFormat;
      bool animatedOverrideApplied = false;
      if (vertexBufferAddress == 0u) {
        return Result<bool, std::string>::makeError(
            "TransmissionRenderer::prepareTransmissionPasses: invalid live "
            "vertex buffer address");
      }
      if (packedVertexFormat ==
              static_cast<uint32_t>(PackedVertexFormat::StaticQuantized20) &&
          vertexDecodeBufferAddress == 0u) {
        return Result<bool, std::string>::makeError(
            "TransmissionRenderer::prepareTransmissionPasses: invalid live "
            "vertex decode buffer address");
      }
      if (animationSceneData != nullptr &&
          entry.instanceIndex <
              animationSceneData->geometryOverridesByRenderable.size()) {
        const AnimatedRenderableGeometryOverride &geometryOverride =
            animationSceneData
                ->geometryOverridesByRenderable[entry.instanceIndex];
        if (geometryOverride.enabled &&
            nuri::isValid(geometryOverride.vertexBuffer) &&
            animationOverrideCoversSubmesh(geometryOverride, *entry.submesh)) {
          const uint64_t overrideVertexAddress = gpu_.getBufferDeviceAddress(
              geometryOverride.vertexBuffer, geometryOverride.vertexByteOffset);
          if (overrideVertexAddress != 0u) {
            vertexBuffer = geometryOverride.vertexBuffer;
            vertexBufferAddress = overrideVertexAddress;
            vertexDecodeBufferAddress = 0u;
            vertexDecodeIndex = 0u;
            packedVertexFormat =
                static_cast<uint32_t>(PackedVertexFormat::AnimatedFloat32);
            animatedOverrideApplied = true;
          }
        }
      }

      const bool sortedFeedbackDraw = entry.sortedFeedback;
      MeshPushConstants constants{
          .frameDataAddress =
              sortedFeedbackDraw ? blendedFrameDataAddress : frameDataAddress,
          .vertexBufferAddress = vertexBufferAddress,
          .vertexDecodeBufferAddress = vertexDecodeBufferAddress,
          .instanceMatricesAddress = instanceMatricesAddress,
          .instanceRemapAddress = instanceRemapAddress,
          .instanceCentersPhaseAddress = 0u,
          .instanceBaseMatricesAddress = 0u,
          .instanceCount = renderableCount,
          .materialIndex = entry.materialIndex,
          .vertexDecodeIndex = vertexDecodeIndex,
          .packedVertexFormat = packedVertexFormat,
          .timeSeconds = static_cast<float>(frame.timeSeconds),
          // Alias tessMaxFactor as taaJitterMinLod: a scene-color pyramid LOD
          // in mip levels. Transmission shaders do not tessellate, so this
          // push-constant slot is otherwise unused in this pipeline.
          .tessMaxFactor = sortedFeedbackDraw ? 0.0f : taaJitterMinLod,
          .debugVisualizationMode = debugFlags,
      };
      constants.setTransmissionScale(entry.transmissionScale);
      std::pmr::vector<MeshPushConstants> &pushConstants =
          sortedFeedbackDraw ? blendedPushConstants_ : meshPushConstants_;
      pushConstants.push_back(constants);
      const MeshPushConstants &pc = pushConstants.back();

      const RenderPipelineHandle pipeline =
          selectMeshPipeline(entry.doubleSided, sortedFeedbackDraw);
      const bool depthBiasEnabled =
          jitteredPostTaaDepthBias && jitterDepthBiasConstant != 0.0f;
      const uint64_t drawLayoutSignature = transmissionDrawLayoutSignature(
          pipeline, vertexBuffer, entry.indexBuffer, entry.indexBufferOffset,
          *lod, entry.instanceIndex, hasDepthAttachment, depthBiasEnabled,
          jitterDepthBiasBits);

      DrawItem draw{};
      if (!animatedOverrideApplied &&
          entry.cachedDrawLayoutSignature == drawLayoutSignature) {
        draw = entry.cachedDrawItem;
      } else {
        draw.pipeline = pipeline;
        draw.vertexBuffer = vertexBuffer;
        draw.indexBuffer = entry.indexBuffer;
        draw.indexBufferOffset = entry.indexBufferOffset;
        draw.indexFormat = IndexFormat::U32;
        draw.indexCount = lod->indexCount;
        draw.instanceCount = 1u;
        draw.firstIndex = lod->indexOffset;
        draw.firstInstance = entry.instanceIndex;
        if (hasDepthAttachment) {
          draw.useDepthState = true;
          draw.depthState = {.compareOp = CompareOp::LessEqual,
                             .isDepthWriteEnabled = false};
          if (depthBiasEnabled) {
            draw.depthBiasEnable = true;
            draw.depthBiasConstant = jitterDepthBiasConstant;
          }
        }
        draw.debugLabel = kTransmissionMeshLabel;
        draw.debugColor = kTransmissionMeshDebugColor;
        if (!animatedOverrideApplied) {
          entry.cachedDrawItem = draw;
          entry.cachedDrawLayoutSignature = drawLayoutSignature;
        }
      }
      draw.pushConstants = std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&pc), sizeof(MeshPushConstants));
      if (sortedFeedbackDraw) {
        sortedDepthTemplates_.push_back(&entry);
        blendedSortableDraws_.push_back(TransparentStageSortableDraw{
            .draw = draw,
            .sortDepth = 0.0f,
            .stableOrder = static_cast<uint32_t>(blendedSortableDraws_.size()),
            .flags = kTransparentStageDrawFlagRequiresFrameColorFeedback,
        });
      } else {
        passDrawItems_.push_back(draw);
      }
    }
    NURI_PROFILER_ZONE_END();
    if (!sortedDepthTemplates_.empty()) {
      NURI_PROFILER_ZONE("TransmissionRenderer.sorted_depth_update",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      for (size_t i = 0; i < sortedDepthTemplates_.size(); ++i) {
        blendedSortableDraws_[i].sortDepth = transmissionSortDepth(
            frame.camera.view, sortedDepthTemplates_[i]->worldBoundsCorners);
      }
      NURI_PROFILER_ZONE_END();
    }

    NURI_PROFILER_ZONE("TransmissionRenderer.dependency_refresh",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    appendUniqueBuffer(passDependencyBuffers_, sceneGpu->buffer);
    appendUniqueBuffer(passDependencyBuffers_, materialGpu->headerBuffer);
    appendUniqueBuffer(passDependencyBuffers_, materialGpu->clearcoatBuffer);
    appendUniqueBuffer(passDependencyBuffers_, materialGpu->sheenBuffer);
    appendUniqueBuffer(passDependencyBuffers_, materialGpu->transmissionBuffer);
    appendUniqueBuffer(passDependencyBuffers_, materialGpu->specularBuffer);
    appendUniqueBuffer(passDependencyBuffers_, instanceMatricesBufferHandle);
    appendUniqueBuffer(passDependencyBuffers_,
                       instanceRemapRing_[frameSlot].buffer->handle());
    blendedDependencyBuffers_ = passDependencyBuffers_;
    if (nuri::isValid(blendedFrameDataBufferHandle)) {
      appendUniqueBuffer(blendedDependencyBuffers_,
                         blendedFrameDataBufferHandle);
    }
    const std::array<TextureHandle, 3> passFrameTextures = {
        sceneColorTexture, sceneColorHalfResTexture,
        sceneColorQuarterResTexture};
    const uint64_t passTextureReadSignature =
        textureReadSignature(staticPassTextureReads_, passFrameTextures);
    if (cachedPassTextureReadSignature_ != passTextureReadSignature ||
        passTextureReads_.empty()) {
      rebuildTextureReads(passTextureReads_, staticPassTextureReads_,
                          passFrameTextures);
      cachedPassTextureReadSignature_ = passTextureReadSignature;
    }
    if (hasSortedFeedbackDraws) {
      const std::array<TextureHandle, 3> blendedFeedbackTextures = {
          frame.sharedResources.transparentTransmissionFeedbackTexture,
          frame.sharedResources.transparentTransmissionFeedbackHalfResTexture,
          frame.sharedResources
              .transparentTransmissionFeedbackQuarterResTexture};
      const uint64_t blendedTextureReadSignature = textureReadSignature(
          staticPassTextureReads_, blendedFeedbackTextures);
      if (cachedBlendedTextureReadSignature_ != blendedTextureReadSignature ||
          blendedTextureReads_.empty()) {
        rebuildTextureReads(blendedTextureReads_, staticPassTextureReads_,
                            blendedFeedbackTextures);
        cachedBlendedTextureReadSignature_ = blendedTextureReadSignature;
      }
    } else {
      blendedTextureReads_.clear();
      cachedBlendedTextureReadSignature_ = std::numeric_limits<uint64_t>::max();
    }
    NURI_PROFILER_ZONE_END();
    frame.metrics.antiAliasing.transparentTransmissionBlendDrawCount =
        saturateToU32(blendedSortableDraws_.size());
    if (loggedAddressProbeTopologyVersion_ != frame.scene->topologyVersion() &&
        !meshPushConstants_.empty()) {
      loggedAddressProbeTopologyVersion_ = frame.scene->topologyVersion();
      const MeshPushConstants &probe = meshPushConstants_.front();
      NURI_LOG_TRACE(
          "TransmissionRenderer::prepareTransmissionPasses probe: "
          "frameData=0x%llx vertex=0x%llx vertexDecode=0x%llx "
          "instanceMatrices=0x%llx instanceRemap=0x%llx "
          "dirLights=0x%llx localLights=0x%llx materialHeader=0x%llx "
          "materialTransmission=0x%llx materialIndex=%u decodeIndex=%u "
          "debugFlags=0x%08x packedFormat=%u draws=%zu deps=%zu",
          static_cast<unsigned long long>(probe.frameDataAddress),
          static_cast<unsigned long long>(probe.vertexBufferAddress),
          static_cast<unsigned long long>(probe.vertexDecodeBufferAddress),
          static_cast<unsigned long long>(probe.instanceMatricesAddress),
          static_cast<unsigned long long>(probe.instanceRemapAddress),
          static_cast<unsigned long long>(
              sceneGpu->directionalLightBufferAddress),
          static_cast<unsigned long long>(sceneGpu->localLightBufferAddress),
          static_cast<unsigned long long>(materialGpu->headerBufferAddress),
          static_cast<unsigned long long>(
              materialGpu->transmissionBufferAddress),
          probe.materialIndex, probe.vertexDecodeIndex, debugFlags,
          probe.packedVertexFormat, passDrawItems_.size(),
          passDependencyBuffers_.size());
    }
    NURI_PROFILER_ZONE_END();
  } else {
    passTextureReads_.clear();
    blendedTextureReads_.clear();
    cachedPassTextureReadSignature_ = std::numeric_limits<uint64_t>::max();
    cachedBlendedTextureReadSignature_ = std::numeric_limits<uint64_t>::max();
  }

  preparedSceneColorTexture_ = sceneColorTexture;
  preparedSceneColorHalfResTexture_ = sceneColorHalfResTexture;
  preparedSceneColorQuarterResTexture_ = sceneColorQuarterResTexture;
  preparedFrameColorTexture_ = frameColorTexture;
  preparedDepthTexture_ = depthTexture;
  preparedSceneDepthGraphTexture_ = sceneDepthGraphTexture;

  if (!meshDrawTemplates_.empty() && passDrawItems_.empty() &&
      blendedSortableDraws_.empty()) {
    NURI_LOG_WARNING("TransmissionRenderer::prepareTransmissionPasses: "
                     "transmission cache has "
                     "%zu template draw(s) but produced no pass draw(s)",
                     meshDrawTemplates_.size());
  }
  return Result<bool, std::string>::makeResult(true);
}

bool TransmissionRenderer::hasPreparedTransmissionMainPass() const noexcept {
  return !passDrawItems_.empty();
}

Result<bool, std::string>
TransmissionRenderer::buildTransparentStageContribution(
    RenderFrameContext &frame, TransparentStageContribution &out) {
  (void)frame;
  out.sortableDraws = std::span<const TransparentStageSortableDraw>(
      blendedSortableDraws_.data(), blendedSortableDraws_.size());
  out.fixedDraws = {};
  out.dependencyBuffers = std::span<const BufferHandle>(
      blendedDependencyBuffers_.data(), blendedDependencyBuffers_.size());
  out.textureReads = std::span<const TextureHandle>(
      blendedTextureReads_.data(), blendedTextureReads_.size());
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TransmissionRenderer::appendTransmissionMainPass(RenderFrameContext &frame,
                                                 RenderGraphBuilder &graph) {
  if (!hasPreparedTransmissionMainPass()) {
    return Result<bool, std::string>::makeResult(true);
  }

  if (!nuri::isValid(preparedFrameColorTexture_)) {
    return Result<bool, std::string>::makeError(
        "TransmissionRenderer::appendTransmissionMainPass: frame color target "
        "is "
        "unavailable at build time");
  }
  AntiAliasingFrameMetrics &aaMetrics = frame.metrics.antiAliasing;
  const RenderSettings &settings = settingsOrDefault(frame);
  const AntiAliasingDebugView debugView =
      sanitizeAntiAliasingDebugView(settings.antiAliasing.debug.view);
  if (shouldSuppressTransmissionForAntiAliasingDebugView(debugView)) {
    return Result<bool, std::string>::makeResult(true);
  }
  const bool normalTaaResolve =
      sanitizeAntiAliasingMode(settings.antiAliasing.mode) ==
          AntiAliasingMode::TAA &&
      (debugView == AntiAliasingDebugView::None ||
       debugView == AntiAliasingDebugView::TAAResolved);
  const bool hasSceneColorInputs = nuri::isValid(preparedSceneColorTexture_);
  const bool consumedPostTaaSceneColor =
      hasSceneColorInputs && aaMetrics.taaResolvedSceneColorPublished &&
      aaMetrics.taaPostResolveSceneColorMipChainGenerated;
  aaMetrics.taaTransmissionPostResolveSceneColorConsumed =
      consumedPostTaaSceneColor;
  if (normalTaaResolve && !consumedPostTaaSceneColor) {
    ++aaMetrics.taaTransmissionStaleSceneColorFrameCount;
  }
  if (debugView == AntiAliasingDebugView::TAATransmissionMipSource) {
    aaMetrics.taaTransmissionMipDebugViewRendered = true;
    ++aaMetrics.taaTransmissionMipDebugPassCount;
  }
  const GpuTimingReport timingReport = gpu_.getLatestCompletedGpuTimingReport();
  if (hasGpuTimingScope(timingReport, GpuTimingScope::Transmission)) {
    aaMetrics.taaTransmissionGpuTimeMs = timingReport.transmissionTimeMs;
    aaMetrics.taaTransmissionGpuTimingSourceFrameIndex =
        timingReport.transmissionSourceFrameIndex;
    aaMetrics.taaTransmissionGpuTimingAvailable = 1u;
  }
  aaMetrics.taaTransmissionFlickerEstimate =
      normalTaaResolve && !consumedPostTaaSceneColor ? 1.0f : 0.0f;

  NURI_PROFILER_ZONE("TransmissionRenderer.main_pass_build",
                     NURI_PROFILER_COLOR_CMD_DRAW);
  uint64_t preResolvedDrawBufferSignature =
      foldHandle(static_cast<uint32_t>(preResolvedTemplateBuffers_.size()),
                 static_cast<uint32_t>(passDrawItems_.size()));
  for (const BufferHandle handle : preResolvedTemplateBuffers_) {
    preResolvedDrawBufferSignature =
        hashCombine64(preResolvedDrawBufferSignature,
                      foldHandle(handle.index, handle.generation));
  }
  for (const DrawItem &draw : passDrawItems_) {
    preResolvedDrawBufferSignature = hashCombine64(
        preResolvedDrawBufferSignature,
        foldHandle(draw.vertexBuffer.index, draw.vertexBuffer.generation));
    preResolvedDrawBufferSignature = hashCombine64(
        preResolvedDrawBufferSignature,
        foldHandle(draw.indexBuffer.index, draw.indexBuffer.generation));
    preResolvedDrawBufferSignature = hashCombine64(
        preResolvedDrawBufferSignature,
        foldHandle(draw.indirectBuffer.index, draw.indirectBuffer.generation));
    preResolvedDrawBufferSignature =
        hashCombine64(preResolvedDrawBufferSignature,
                      foldHandle(draw.indirectCountBuffer.index,
                                 draw.indirectCountBuffer.generation));
  }
  if (cachedPreResolvedDrawBufferSignature_ != preResolvedDrawBufferSignature ||
      cachedPreResolvedDrawBuffers_.empty()) {
    ScratchArena preResolveScratchArena;
    ScopedScratch preResolveScratch(preResolveScratchArena);
    PmrHashSet<uint64_t> seenBuffers(preResolveScratch.resource());
    seenBuffers.reserve(preResolvedTemplateBuffers_.size() +
                        passDrawItems_.size() * 4u);
    cachedPreResolvedDrawBuffers_.clear();
    const auto appendPreResolvedBuffer = [&](BufferHandle handle) {
      if (!nuri::isValid(handle)) {
        return;
      }
      const uint64_t handleKey = foldHandle(handle.index, handle.generation);
      if (!seenBuffers.insert(handleKey).second) {
        return;
      }
      cachedPreResolvedDrawBuffers_.push_back(handle);
    };
    cachedPreResolvedDrawBuffers_.reserve(preResolvedTemplateBuffers_.size() +
                                          passDrawItems_.size() * 4u);
    for (const BufferHandle handle : preResolvedTemplateBuffers_) {
      appendPreResolvedBuffer(handle);
    }
    for (const DrawItem &draw : passDrawItems_) {
      appendPreResolvedBuffer(draw.vertexBuffer);
      appendPreResolvedBuffer(draw.indexBuffer);
      appendPreResolvedBuffer(draw.indirectBuffer);
      appendPreResolvedBuffer(draw.indirectCountBuffer);
    }
    cachedPreResolvedDrawBufferSignature_ = preResolvedDrawBufferSignature;
  }
  passDependencyBufferAccessModes_.resize(passDependencyBuffers_.size(),
                                          RenderGraphAccessMode::Read);

  ScratchArena passBuildScratchArena;
  ScopedScratch passBuildScratch(passBuildScratchArena);
  std::pmr::vector<RenderGraphPreparedDependencyBufferBinding>
      dependencyBufferBindings(passBuildScratch.resource());
  dependencyBufferBindings.reserve(passDependencyBuffers_.size());
  for (size_t i = 0; i < passDependencyBuffers_.size(); ++i) {
    const BufferHandle dependency = passDependencyBuffers_[i];
    if (!nuri::isValid(dependency)) {
      continue;
    }
    auto importResult =
        graph.importBuffer(dependency, "transmission_pass_dependency_buffer");
    if (importResult.hasError()) {
      return Result<bool, std::string>::makeError(importResult.error());
    }
    const RenderGraphAccessMode accessMode =
        i < passDependencyBufferAccessModes_.size()
            ? passDependencyBufferAccessModes_[i]
            : RenderGraphAccessMode::Read;
    dependencyBufferBindings.push_back(
        RenderGraphPreparedDependencyBufferBinding{
            .dependencyIndex = static_cast<uint32_t>(i),
            .buffer = importResult.value(),
            .mode = accessMode,
        });
  }

  std::pmr::vector<RenderGraphPreparedDependencyTextureBinding>
      dependencyTextureBindings(passBuildScratch.resource());
  dependencyTextureBindings.reserve(passTextureReads_.size());
  for (const TextureHandle dependency : passTextureReads_) {
    if (!nuri::isValid(dependency)) {
      continue;
    }
    auto importResult =
        graph.importTexture(dependency, "transmission_pass_dependency_texture");
    if (importResult.hasError()) {
      return Result<bool, std::string>::makeError(importResult.error());
    }
    dependencyTextureBindings.push_back(
        RenderGraphPreparedDependencyTextureBinding{
            .texture = importResult.value(),
            .mode = RenderGraphAccessMode::Read,
        });
  }

  std::pmr::vector<RenderGraphBufferId> preResolvedDrawBufferIds(
      passBuildScratch.resource());
  preResolvedDrawBufferIds.reserve(cachedPreResolvedDrawBuffers_.size());
  for (const BufferHandle buffer : cachedPreResolvedDrawBuffers_) {
    if (!nuri::isValid(buffer)) {
      continue;
    }
    auto importResult =
        graph.importBuffer(buffer, "transmission_pre_resolved_draw_buffer");
    if (importResult.hasError()) {
      return Result<bool, std::string>::makeError(importResult.error());
    }
    preResolvedDrawBufferIds.push_back(importResult.value());
  }

  RenderGraphPreparedGraphicsPassDesc passDesc{};
  passDesc.color = {.loadOp = LoadOp::Load,
                    .storeOp = StoreOp::Store,
                    .clearColor = {1.0f, 1.0f, 1.0f, 1.0f}};
  auto colorImportResult = graph.importTexture(preparedFrameColorTexture_,
                                               "transmission_frame_color");
  if (colorImportResult.hasError()) {
    return Result<bool, std::string>::makeError(colorImportResult.error());
  }
  passDesc.colorTexture = colorImportResult.value();
  if (nuri::isValid(preparedDepthTexture_) ||
      nuri::isValid(preparedSceneDepthGraphTexture_)) {
    passDesc.depth = {.loadOp = LoadOp::Load,
                      .storeOp = StoreOp::Store,
                      .clearDepth = 1.0f,
                      .clearStencil = 0u};
    RenderGraphTextureId depthGraphTexture = preparedSceneDepthGraphTexture_;
    if (nuri::isValid(
            frame.sharedResources.transmissionVisibilityDepthGraphTexture) &&
        nuri::isValid(
            frame.sharedResources.transmissionVisibilityDepthTexture) &&
        isSameTextureHandle(
            preparedDepthTexture_,
            frame.sharedResources.transmissionVisibilityDepthTexture)) {
      depthGraphTexture =
          frame.sharedResources.transmissionVisibilityDepthGraphTexture;
    }
    if (nuri::isValid(depthGraphTexture)) {
      passDesc.depthTexture = depthGraphTexture;
    } else {
      auto depthImportResult = graph.importTexture(preparedDepthTexture_,
                                                   "transmission_scene_depth");
      if (depthImportResult.hasError()) {
        return Result<bool, std::string>::makeError(depthImportResult.error());
      }
      passDesc.depthTexture = depthImportResult.value();
    }
  }
  passDesc.draws =
      std::span<const DrawItem>(passDrawItems_.data(), passDrawItems_.size());
  passDesc.drawBuffersPreResolved = true;
  passDesc.preResolvedDrawBufferIds = std::span<const RenderGraphBufferId>(
      preResolvedDrawBufferIds.data(), preResolvedDrawBufferIds.size());
  passDesc.dependencyBuffers = std::span<const BufferHandle>(
      passDependencyBuffers_.data(), passDependencyBuffers_.size());
  passDesc.dependencyBufferBindings =
      std::span<const RenderGraphPreparedDependencyBufferBinding>(
          dependencyBufferBindings.data(), dependencyBufferBindings.size());
  passDesc.dependencyTextureBindings =
      std::span<const RenderGraphPreparedDependencyTextureBinding>(
          dependencyTextureBindings.data(), dependencyTextureBindings.size());
  passDesc.debugLabel = kTransmissionPassLabel;
  passDesc.debugColor = kTransmissionPassDebugColor;
  passDesc.borrowPayload = true;
  passDesc.gpuTimingScope = GpuTimingScope::Transmission;

  auto addResult = graph.addPreparedGraphicsPass(passDesc);
  if (addResult.hasError()) {
    return Result<bool, std::string>::makeError(addResult.error());
  }
  NURI_PROFILER_ZONE_END();

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> TransmissionRenderer::ensureInitialized() {
  if (initialized_) {
    return Result<bool, std::string>::makeResult(true);
  }
  auto shaderResult = createShaders();
  if (shaderResult.hasError()) {
    return shaderResult;
  }
  initialized_ = true;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> TransmissionRenderer::createShaders() {
  destroyShaders();
  meshShader_ = Shader::create("transmission_main", gpu_);
  if (!meshShader_) {
    return Result<bool, std::string>::makeError(
        "TransmissionRenderer::createShaders: failed to create shader "
        "wrappers");
  }

  auto meshVertexResult = meshShader_->compileFromFile(
      transmissionVertexPath_.string(), ShaderStage::Vertex);
  if (meshVertexResult.hasError()) {
    return Result<bool, std::string>::makeError(meshVertexResult.error());
  }
  auto meshFragmentResult = meshShader_->compileFromFile(
      transmissionFragmentPath_.string(), ShaderStage::Fragment);
  if (meshFragmentResult.hasError()) {
    return Result<bool, std::string>::makeError(meshFragmentResult.error());
  }
  meshVertexShader_ = meshVertexResult.value();
  meshFragmentShader_ = meshFragmentResult.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TransmissionRenderer::ensurePipelines(Format colorFormat, Format depthFormat) {
  const bool meshPipelinesValid =
      nuri::isValid(meshPipelineHandle_) &&
      nuri::isValid(meshDoubleSidedPipelineHandle_) &&
      nuri::isValid(meshBlendPipelineHandle_) &&
      nuri::isValid(meshBlendDoubleSidedPipelineHandle_) &&
      meshPipelineColorFormat_ == colorFormat &&
      meshPipelineDepthFormat_ == depthFormat;
  if (meshPipelinesValid) {
    return Result<bool, std::string>::makeResult(true);
  }

  destroyPipelineState();

  auto meshResult = gpu_.createRenderPipeline(
      meshPipelineDesc(colorFormat, depthFormat, meshVertexShader_,
                       meshFragmentShader_, CullMode::Back, false),
      "transmission_mesh");
  if (meshResult.hasError()) {
    return Result<bool, std::string>::makeError(meshResult.error());
  }
  meshPipelineHandle_ = meshResult.value();

  auto meshDoubleResult = gpu_.createRenderPipeline(
      meshPipelineDesc(colorFormat, depthFormat, meshVertexShader_,
                       meshFragmentShader_, CullMode::None, false),
      "transmission_mesh_double_sided");
  if (meshDoubleResult.hasError()) {
    destroyPipelineState();
    return Result<bool, std::string>::makeError(meshDoubleResult.error());
  }
  meshDoubleSidedPipelineHandle_ = meshDoubleResult.value();

  auto blendResult = gpu_.createRenderPipeline(
      meshPipelineDesc(colorFormat, depthFormat, meshVertexShader_,
                       meshFragmentShader_, CullMode::Back, true),
      "transmission_mesh_blend");
  if (blendResult.hasError()) {
    destroyPipelineState();
    return Result<bool, std::string>::makeError(blendResult.error());
  }
  meshBlendPipelineHandle_ = blendResult.value();

  auto blendDoubleResult = gpu_.createRenderPipeline(
      meshPipelineDesc(colorFormat, depthFormat, meshVertexShader_,
                       meshFragmentShader_, CullMode::None, true),
      "transmission_mesh_blend_double_sided");
  if (blendDoubleResult.hasError()) {
    destroyPipelineState();
    return Result<bool, std::string>::makeError(blendDoubleResult.error());
  }
  meshBlendDoubleSidedPipelineHandle_ = blendDoubleResult.value();

  meshPipelineColorFormat_ = colorFormat;
  meshPipelineDepthFormat_ = depthFormat;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TransmissionRenderer::ensureRingBufferCount(uint32_t requiredCount) {
  const uint32_t count = std::max(requiredCount, 1u);
  while (instanceMatricesRing_.size() < count) {
    instanceMatricesRing_.push_back(DynamicBufferSlot{});
  }
  while (instanceRemapRing_.size() < count) {
    instanceRemapRing_.push_back(DynamicBufferSlot{});
  }
  instanceDataRingUploadVersions_.resize(count,
                                         std::numeric_limits<uint64_t>::max());
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TransmissionRenderer::ensureInstanceMatricesRingCapacity(size_t requiredBytes) {
  bool needsResize = false;
  bool hasLiveBuffers = false;
  for (const DynamicBufferSlot &slot : instanceMatricesRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      hasLiveBuffers = true;
      if (slot.capacityBytes < requiredBytes) {
        needsResize = true;
        break;
      }
    }
  }
  if (needsResize && hasLiveBuffers) {
    gpu_.waitIdle();
  }
  for (size_t i = 0; i < instanceMatricesRing_.size(); ++i) {
    DynamicBufferSlot &slot = instanceMatricesRing_[i];
    if (slot.buffer && slot.buffer->valid() &&
        slot.capacityBytes >= requiredBytes) {
      continue;
    }
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
    slot.buffer.reset();
    auto bufferResult = Buffer::create(gpu_,
                                       BufferDesc{.usage = BufferUsage::Storage,
                                                  .storage = Storage::Device,
                                                  .size = requiredBytes},
                                       "transmission_instance_matrices");
    if (bufferResult.hasError()) {
      return Result<bool, std::string>::makeError(bufferResult.error());
    }
    slot.buffer = std::move(bufferResult.value());
    slot.capacityBytes = requiredBytes;
    instanceDataRingUploadVersions_[i] = std::numeric_limits<uint64_t>::max();
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TransmissionRenderer::ensureInstanceRemapRingCapacity(size_t requiredBytes) {
  bool needsResize = false;
  bool hasLiveBuffers = false;
  for (const DynamicBufferSlot &slot : instanceRemapRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      hasLiveBuffers = true;
      if (slot.capacityBytes < requiredBytes) {
        needsResize = true;
        break;
      }
    }
  }
  if (needsResize && hasLiveBuffers) {
    gpu_.waitIdle();
  }
  for (size_t i = 0; i < instanceRemapRing_.size(); ++i) {
    DynamicBufferSlot &slot = instanceRemapRing_[i];
    if (slot.buffer && slot.buffer->valid() &&
        slot.capacityBytes >= requiredBytes) {
      continue;
    }
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
    slot.buffer.reset();
    auto bufferResult = Buffer::create(gpu_,
                                       BufferDesc{.usage = BufferUsage::Storage,
                                                  .storage = Storage::Device,
                                                  .size = requiredBytes},
                                       "transmission_instance_remap");
    if (bufferResult.hasError()) {
      return Result<bool, std::string>::makeError(bufferResult.error());
    }
    slot.buffer = std::move(bufferResult.value());
    slot.capacityBytes = requiredBytes;
    instanceDataRingUploadVersions_[i] = std::numeric_limits<uint64_t>::max();
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TransmissionRenderer::ensureBlendedFrameDataRingCapacity(size_t requiredBytes) {
  const uint32_t count = std::max(1u, gpu_.getSwapchainImageCount());
  while (blendedFrameDataRing_.size() < count) {
    blendedFrameDataRing_.push_back(DynamicBufferSlot{});
  }
  bool waitedForResize = false;
  for (DynamicBufferSlot &slot : blendedFrameDataRing_) {
    if (slot.buffer && slot.buffer->valid() &&
        slot.capacityBytes >= requiredBytes) {
      continue;
    }
    if (slot.buffer && slot.buffer->valid()) {
      if (slot.capacityBytes < requiredBytes && !waitedForResize) {
        gpu_.waitIdle();
        waitedForResize = true;
      }
      gpu_.destroyBuffer(slot.buffer->handle());
    }
    slot.buffer.reset();
    auto bufferResult =
        Buffer::create(gpu_,
                       BufferDesc{.usage = BufferUsage::Storage,
                                  .storage = Storage::HostVisible,
                                  .size = requiredBytes},
                       "transparent_transmission_frame_data");
    if (bufferResult.hasError()) {
      return Result<bool, std::string>::makeError(bufferResult.error());
    }
    slot.buffer = std::move(bufferResult.value());
    slot.capacityBytes = requiredBytes;
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TransmissionRenderer::ensureTransparentFeedbackTextures(
    RenderFrameContext &frame) {
  int32_t framebufferWidth = 0;
  int32_t framebufferHeight = 0;
  gpu_.getFramebufferSize(framebufferWidth, framebufferHeight);
  const uint32_t safeWidth =
      static_cast<uint32_t>(std::max(framebufferWidth, 1));
  const uint32_t safeHeight =
      static_cast<uint32_t>(std::max(framebufferHeight, 1));
  const uint32_t ringCount = std::max(1u, gpu_.getSwapchainImageCount());
  const bool needsRecreate =
      transparentFeedbackWidth_ != safeWidth ||
      transparentFeedbackHeight_ != safeHeight ||
      transparentFeedbackRingCount_ != ringCount ||
      transparentFeedbackTextures_[0].size() != ringCount ||
      transparentFeedbackTextures_[1].size() != ringCount ||
      transparentFeedbackTextures_[2].size() != ringCount;
  if (needsRecreate) {
    destroyFeedbackTextures();
    transparentFeedbackWidth_ = safeWidth;
    transparentFeedbackHeight_ = safeHeight;
    transparentFeedbackRingCount_ = ringCount;
    for (uint32_t mipLevel = 0u; mipLevel < kFrameCompositionSceneColorMipCount;
         ++mipLevel) {
      std::pmr::vector<TextureHandle> &textures =
          transparentFeedbackTextures_[mipLevel];
      textures.resize(ringCount);
      const TextureDesc desc = makeTransparentFeedbackTextureDesc(
          levelDimension(safeWidth, mipLevel),
          levelDimension(safeHeight, mipLevel));
      for (uint32_t i = 0u; i < ringCount; ++i) {
        const std::string debugName = "transparent_transmission_feedback_" +
                                      std::to_string(mipLevel) + "_" +
                                      std::to_string(i);
        auto createResult = gpu_.createTexture(desc, debugName);
        if (createResult.hasError()) {
          destroyFeedbackTextures();
          return Result<bool, std::string>::makeError(createResult.error());
        }
        textures[i] = createResult.value();
      }
    }
  }
  const uint32_t frameSlot =
      static_cast<uint32_t>(frame.frameIndex % ringCount);
  frame.sharedResources.transparentTransmissionFeedbackTexture =
      transparentFeedbackTextures_[0][frameSlot];
  frame.sharedResources.transparentTransmissionFeedbackHalfResTexture =
      transparentFeedbackTextures_[1][frameSlot];
  frame.sharedResources.transparentTransmissionFeedbackQuarterResTexture =
      transparentFeedbackTextures_[2][frameSlot];
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TransmissionRenderer::rebuildSceneCache(const RenderScene &scene,
                                        const ResourceManager &resources,
                                        uint32_t materialCount) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_DRAW);
  meshDrawTemplates_.clear();

  const std::span<const Renderable> renderables = scene.renderables();
  if (renderables.size() >
      static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    return Result<bool, std::string>::makeError(
        "TransmissionRenderer::rebuildSceneCache: renderables count exceeds "
        "UINT32_MAX");
  }

  size_t invalidMaterialFallbackCount = 0u;
  bool hasTransmissionContent = false;
  for (uint32_t renderableIndex = 0;
       renderableIndex < static_cast<uint32_t>(renderables.size());
       ++renderableIndex) {
    const Renderable &renderable = renderables[renderableIndex];
    const ModelRecord *modelRecord = resources.tryGet(renderable.model);
    if (!modelRecord || !modelRecord->model) {
      return Result<bool, std::string>::makeError(
          "TransmissionRenderer::rebuildSceneCache: failed to resolve model");
    }

    GeometryAllocationView geometry{};
    if (!gpu_.resolveGeometry(modelRecord->model->geometryHandle(), geometry)) {
      return Result<bool, std::string>::makeError(
          "TransmissionRenderer::rebuildSceneCache: failed to resolve "
          "geometry");
    }
    const uint64_t vertexBufferAddress = gpu_.getBufferDeviceAddress(
        geometry.vertexBuffer, geometry.vertexByteOffset);
    if (vertexBufferAddress == 0u) {
      return Result<bool, std::string>::makeError(
          "TransmissionRenderer::rebuildSceneCache: invalid vertex buffer "
          "address");
    }

    const std::span<const Submesh> submeshes = modelRecord->model->submeshes();
    for (size_t submeshIndex = 0; submeshIndex < submeshes.size();
         ++submeshIndex) {
      const MaterialRef resolvedMaterial = resolveRenderableMaterial(
          renderable, *modelRecord, static_cast<uint32_t>(submeshIndex));
      const MaterialRecord *materialRecord = resources.tryGet(resolvedMaterial);
      if (materialRecord == nullptr ||
          !isTransmissionMaterial(*materialRecord)) {
        continue;
      }
      hasTransmissionContent = true;
      const bool sortedFeedback =
          usesSortedTransmissionFeedback(*materialRecord);

      uint32_t materialIndex = resources.materialTableIndex(resolvedMaterial);
      if (materialCount == 0u || materialIndex >= materialCount) {
        materialIndex = 0u;
        ++invalidMaterialFallbackCount;
      }

      meshDrawTemplates_.push_back(MeshDrawTemplate{
          .renderable = &renderable,
          .submesh = &submeshes[submeshIndex],
          .submeshIndex = static_cast<uint32_t>(submeshIndex),
          .indexBuffer = geometry.indexBuffer,
          .indexBufferOffset = geometry.indexByteOffset,
          .baseVertexBuffer = geometry.vertexBuffer,
          .vertexBufferByteOffset = geometry.vertexByteOffset,
          .vertexBuffer = geometry.vertexBuffer,
          .baseVertexDecodeBuffer = modelRecord->model->vertexDecodeBuffer(),
          .vertexBufferAddress = vertexBufferAddress,
          .vertexDecodeBufferAddress =
              modelRecord->model->vertexDecodeBufferAddress(),
          .vertexDecodeIndex = static_cast<uint32_t>(submeshIndex),
          .packedVertexFormat =
              static_cast<uint32_t>(modelRecord->model->drawVertexFormat()),
          .materialIndex = materialIndex,
          .instanceIndex = renderableIndex,
          .doubleSided = materialRecord->desc.doubleSided,
          .sortedFeedback = sortedFeedback,
      });
    }
  }

  if (invalidMaterialFallbackCount > 0u) {
    if (!loggedMaterialFallbackWarning_) {
      NURI_LOG_WARNING(
          "TransmissionRenderer::rebuildSceneCache: %zu submesh draw(s) used "
          "fallback material index 0 due to missing/out-of-range material "
          "mapping",
          invalidMaterialFallbackCount);
      loggedMaterialFallbackWarning_ = true;
    }
  } else {
    loggedMaterialFallbackWarning_ = false;
  }

  preResolvedTemplateBuffers_.clear();
  preResolvedTemplateBuffers_.reserve(meshDrawTemplates_.size() * 3u);
  for (const MeshDrawTemplate &entry : meshDrawTemplates_) {
    appendUniqueBuffer(preResolvedTemplateBuffers_, entry.baseVertexBuffer);
    appendUniqueBuffer(preResolvedTemplateBuffers_,
                       entry.baseVertexDecodeBuffer);
    appendUniqueBuffer(preResolvedTemplateBuffers_, entry.indexBuffer);
  }
  cachedPreResolvedDrawBuffers_.clear();
  cachedPreResolvedDrawBufferSignature_ = std::numeric_limits<uint64_t>::max();

  cachedScene_ = &scene;
  cachedTopologyVersion_ = scene.topologyVersion();
  cachedMaterialVersion_ = resources.materialSnapshot().version;
  cachedTransmissionContentScene_ = &scene;
  cachedTransmissionContentTopologyVersion_ = scene.topologyVersion();
  cachedTransmissionContentMaterialVersion_ =
      resources.materialSnapshot().version;
  cachedTransmissionContentValid_ = true;
  cachedTransmissionContent_ = hasTransmissionContent;
  cachedGeometryMutationVersion_ = gpu_.geometryMutationVersion();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TransmissionRenderer::rebuildMaterialTextureAccessCache(
    const ResourceManager &resources) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_DRAW);
  materialTextureAccessHandles_.clear();
  for (const MeshDrawTemplate &entry : meshDrawTemplates_) {
    if (entry.renderable == nullptr || entry.submesh == nullptr) {
      continue;
    }
    const ModelRecord *modelRecord = resources.tryGet(entry.renderable->model);
    if (!modelRecord || !modelRecord->model) {
      continue;
    }
    const std::span<const Submesh> submeshes = modelRecord->model->submeshes();
    if (entry.submeshIndex >= submeshes.size()) {
      continue;
    }
    const MaterialRef resolvedMaterial = resolveRenderableMaterial(
        *entry.renderable, *modelRecord, entry.submeshIndex);
    const MaterialRecord *materialRecord = resources.tryGet(resolvedMaterial);
    if (materialRecord == nullptr) {
      continue;
    }

    forEachMaterialTextureRef(materialRecord->textureRefs, [&](TextureRef ref) {
      const TextureRecord *record = resources.tryGet(ref);
      if (record == nullptr || !nuri::isValid(record->texture)) {
        return;
      }
      appendUniqueTexture(materialTextureAccessHandles_, record->texture);
    });
  }
  return Result<bool, std::string>::makeResult(true);
}

void TransmissionRenderer::collectEnvironmentTextureReads(
    const RenderScene &scene, const ResourceManager &resources) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_DRAW);
  environmentTextureAccessHandles_.clear();
  const EnvironmentHandles environment = scene.environment();
  const std::array<TextureRef, 5> refs = {
      environment.cubemap, environment.irradiance, environment.prefilteredGgx,
      environment.prefilteredCharlie, environment.brdfLut};
  for (const TextureRef ref : refs) {
    const TextureRecord *record = resources.tryGet(ref);
    if (record == nullptr || !nuri::isValid(record->texture)) {
      continue;
    }
    appendUniqueTexture(environmentTextureAccessHandles_, record->texture);
  }
}

void TransmissionRenderer::refreshDrawTemplateTransforms(
    std::span<const Renderable> renderables) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_DRAW);
  for (MeshDrawTemplate &entry : meshDrawTemplates_) {
    if (entry.submesh == nullptr || entry.instanceIndex >= renderables.size()) {
      continue;
    }
    const Renderable &renderable = renderables[entry.instanceIndex];
    entry.transmissionScale =
        transmissionScaleForDraw(renderable, *entry.submesh);
    entry.worldBoundsCorners = transmissionWorldBoundsCorners(
        renderable.modelMatrix, entry.submesh->bounds);
  }
}

void TransmissionRenderer::resetCachedState() {
  cachedScene_ = nullptr;
  cachedTopologyVersion_ = std::numeric_limits<uint64_t>::max();
  cachedMaterialVersion_ = std::numeric_limits<uint64_t>::max();
  cachedTransformVersion_ = std::numeric_limits<uint64_t>::max();
  cachedGeometryMutationVersion_ = std::numeric_limits<uint64_t>::max();
  cachedEnvironmentHandles_ = {};
  cachedTransmissionContentScene_ = nullptr;
  cachedTransmissionContentTopologyVersion_ =
      std::numeric_limits<uint64_t>::max();
  cachedTransmissionContentMaterialVersion_ =
      std::numeric_limits<uint64_t>::max();
  cachedTransmissionContentValid_ = false;
  cachedTransmissionContent_ = false;
  loggedMaterialFallbackWarning_ = false;
  loggedAddressProbeTopologyVersion_ = std::numeric_limits<uint64_t>::max();
  environmentTextureAccessCacheValid_ = false;
  materialTextureAccessCacheValid_ = false;
  staticPassTextureReadsValid_ = false;

  meshDrawTemplates_.clear();
  instanceMatrices_.clear();
  instanceRemap_.clear();
  instanceDataRingUploadVersions_.clear();
  sortedDepthTemplates_.clear();
  materialTextureAccessHandles_.clear();
  environmentTextureAccessHandles_.clear();
  staticPassTextureReads_.clear();
  passTextureReads_.clear();
  blendedTextureReads_.clear();
  blendedDependencyBuffers_.clear();
  passDependencyBufferAccessModes_.clear();
  preResolvedTemplateBuffers_.clear();
  cachedPreResolvedDrawBuffers_.clear();
  cachedPreResolvedDrawBufferSignature_ = std::numeric_limits<uint64_t>::max();
  cachedPassTextureReadSignature_ = std::numeric_limits<uint64_t>::max();
  cachedBlendedTextureReadSignature_ = std::numeric_limits<uint64_t>::max();
}

void TransmissionRenderer::resetFrameBuildState() {
  meshPushConstants_.clear();
  blendedPushConstants_.clear();
  passDrawItems_.clear();
  blendedSortableDraws_.clear();
  sortedDepthTemplates_.clear();
  passDependencyBuffers_.clear();
  blendedDependencyBuffers_.clear();
  passDependencyBufferAccessModes_.clear();
  preparedSceneColorTexture_ = {};
  preparedSceneColorHalfResTexture_ = {};
  preparedSceneColorQuarterResTexture_ = {};
  preparedFrameColorTexture_ = {};
  preparedDepthTexture_ = {};
  preparedSceneDepthGraphTexture_ = {};
}

void TransmissionRenderer::destroyPipelineState() {
  if (nuri::isValid(meshBlendDoubleSidedPipelineHandle_)) {
    gpu_.destroyRenderPipeline(meshBlendDoubleSidedPipelineHandle_);
  }
  if (nuri::isValid(meshBlendPipelineHandle_)) {
    gpu_.destroyRenderPipeline(meshBlendPipelineHandle_);
  }
  if (nuri::isValid(meshDoubleSidedPipelineHandle_)) {
    gpu_.destroyRenderPipeline(meshDoubleSidedPipelineHandle_);
  }
  if (nuri::isValid(meshPipelineHandle_)) {
    gpu_.destroyRenderPipeline(meshPipelineHandle_);
  }
  meshPipelineHandle_ = {};
  meshDoubleSidedPipelineHandle_ = {};
  meshBlendPipelineHandle_ = {};
  meshBlendDoubleSidedPipelineHandle_ = {};
  meshPipelineColorFormat_ = Format::Count;
  meshPipelineDepthFormat_ = Format::Count;
}

void TransmissionRenderer::destroyShaders() {
  if (nuri::isValid(meshVertexShader_)) {
    gpu_.destroyShaderModule(meshVertexShader_);
  }
  if (nuri::isValid(meshFragmentShader_)) {
    gpu_.destroyShaderModule(meshFragmentShader_);
  }
  meshVertexShader_ = {};
  meshFragmentShader_ = {};
}

void TransmissionRenderer::destroyBuffers() {
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
  for (DynamicBufferSlot &slot : blendedFrameDataRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
    slot.buffer.reset();
    slot.capacityBytes = 0;
  }
  instanceMatricesRing_.clear();
  instanceRemapRing_.clear();
  blendedFrameDataRing_.clear();
  instanceDataRingUploadVersions_.clear();
}

void TransmissionRenderer::destroyFeedbackTextures() {
  for (std::pmr::vector<TextureHandle> &textures :
       transparentFeedbackTextures_) {
    for (const TextureHandle texture : textures) {
      if (nuri::isValid(texture)) {
        gpu_.destroyTexture(texture);
      }
    }
    textures.clear();
  }
  transparentFeedbackWidth_ = 0u;
  transparentFeedbackHeight_ = 0u;
  transparentFeedbackRingCount_ = 0u;
}

bool TransmissionRenderer::hasTransmissionContent(
    const RenderFrameContext &frame) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_DRAW);
  if (frame.scene == nullptr || frame.resources == nullptr) {
    return false;
  }

  const MaterialTableSnapshot materialSnapshot =
      frame.resources->materialSnapshot();
  const uint64_t topologyVersion = frame.scene->topologyVersion();
  if (cachedTransmissionContentValid_ &&
      cachedTransmissionContentScene_ == frame.scene &&
      cachedTransmissionContentTopologyVersion_ == topologyVersion &&
      cachedTransmissionContentMaterialVersion_ == materialSnapshot.version) {
    return cachedTransmissionContent_;
  }
  auto rebuildResult =
      rebuildSceneCache(*frame.scene, *frame.resources,
                        static_cast<uint32_t>(materialSnapshot.headers.size()));
  if (rebuildResult.hasError()) {
    NURI_LOG_WARNING("TransmissionRenderer::hasTransmissionContent: %s",
                     rebuildResult.error().c_str());
    return false;
  }
  return cachedTransmissionContentValid_ &&
         cachedTransmissionContentScene_ == frame.scene &&
         cachedTransmissionContentTopologyVersion_ == topologyVersion &&
         cachedTransmissionContentMaterialVersion_ ==
             materialSnapshot.version &&
         cachedTransmissionContent_;
}

RenderPipelineHandle
TransmissionRenderer::selectMeshPipeline(bool doubleSided,
                                         bool useBlendPipeline) const {
  if (useBlendPipeline) {
    if (doubleSided && nuri::isValid(meshBlendDoubleSidedPipelineHandle_)) {
      return meshBlendDoubleSidedPipelineHandle_;
    }
    return meshBlendPipelineHandle_;
  }
  if (doubleSided && nuri::isValid(meshDoubleSidedPipelineHandle_)) {
    return meshDoubleSidedPipelineHandle_;
  }
  return meshPipelineHandle_;
}

} // namespace nuri
