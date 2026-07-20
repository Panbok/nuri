#include "nuri/gfx/renderers/transmission_renderer.h"
#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/frame/render_capture.h"
#include "nuri/gfx/fullscreen.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/gfx/renderers/detail/animation_rendering.h"
#include "nuri/gfx/renderers/detail/forward_rendering.h"
#include "nuri/gfx/renderers/detail/visibility_math.h"
#include "nuri/gfx/shader.h"
#include "nuri/pch.h"
#include "nuri/resources/gpu/resource_manager.h"
namespace nuri {
namespace {
constexpr uint32_t kTransmissionPassDebugColor = 0x33ffaaeeu;
constexpr uint32_t kTransmissionMeshDebugColor = 0x33ffaaeeu;
constexpr uint32_t kTransparentPassDebugColor = 0x66aaffffu;
constexpr std::string_view kTransmissionPassLabel = "Transmission Pass";
constexpr std::string_view kTransmissionMeshLabel = "TransmissionMesh";
constexpr std::array kTransparentTransmissionFeedbackLabels{
    std::string_view{"Transparent Transmission Feedback Copy"},
    std::string_view{"Transparent Transmission Feedback Half"},
    std::string_view{"Transparent Transmission Feedback Quarter"},
};
constexpr std::array kTransparentTransmissionFeedbackCaptureNames{
    std::string_view{"transmission_feedback"},
    std::string_view{"transmission_feedback_half"},
    std::string_view{"transmission_feedback_quarter"},
};
constexpr uint32_t kSceneCopyFlagDownsample = 1u << 0u;
constexpr uint32_t kMaxExactTransparentFeedbackDraws = 64u;
constexpr uint64_t kFnvOffsetBasis64 = 14695981039346656037ull;
constexpr uint64_t kFnvPrime64 = 1099511628211ull;
constexpr float kDefaultTaaCurrentFrameWeight = 0.045f;
constexpr uint32_t kForwardSceneHasSceneColor = 1u << 5u;
struct CopyPushConstants {
  uint32_t sourceTexId = 0u;
  uint32_t sourceSamplerId = 0u;
  uint32_t flags = 0u;
  uint32_t reserved0 = 0u;
};
static_assert(sizeof(CopyPushConstants) <= 128);
[[nodiscard]] std::pmr::memory_resource *
resolveMemoryResource(std::pmr::memory_resource *memory) {
  return memory != nullptr ? memory : std::pmr::get_default_resource();
}
[[nodiscard]] const RenderSettings &
settingsOrDefault(const RenderFrameContext &frame) {
  return renderSettingsOrDefault(frame);
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
[[nodiscard]] uint64_t foldHandle(uint32_t index, uint32_t generation) {
  return (static_cast<uint64_t>(generation) << 32u) | index;
}
[[nodiscard]] uint64_t hashCombine64(uint64_t hash, uint64_t value) {
  hash ^= value;
  hash *= kFnvPrime64;
  return hash;
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
ForwardSceneFrameData
resolveForwardSceneFrameData(const ForwardSceneGpuData &sceneGpu,
                             uint64_t frameDataAddress) {
  if (frameDataAddress == sceneGpu.postTaaFrameDataAddress &&
      sceneGpu.postTaaFrameDataAddress != 0u) {
    return sceneGpu.postTaaFrameData;
  }
  return sceneGpu.frameData;
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
    appendUniqueForwardHandle(out, handle);
  }
  for (const TextureHandle handle : dynamicHandles) {
    appendUniqueForwardHandle(out, handle);
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
void appendPreResolvedDrawBuffers(std::pmr::vector<BufferHandle> &handles,
                                  std::span<const DrawItem> draws) {
  for (const DrawItem &draw : draws) {
    appendUniqueForwardHandle(handles, draw.vertexBuffer);
    appendUniqueForwardHandle(handles, draw.indexBuffer);
    appendUniqueForwardHandle(handles, draw.indirectBuffer);
    appendUniqueForwardHandle(handles, draw.indirectCountBuffer);
  }
}
RenderPipelineDesc meshPipelineDesc(Format colorFormat, Format depthFormat,
                                    ShaderHandle vertexShader,
                                    ShaderHandle fragmentShader,
                                    CullMode cullMode, bool blendEnabled,
                                    RasterPipelineState rasterState) {
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
      .rasterState = canonicalRasterPipelineState(rasterState),
  };
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
  feedbackCopyVertexPath_ = basePath / "fullscreen_copy.vert";
  feedbackCopyFragmentPath_ = basePath / "scene_copy.frag";
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
  const RenderSettings &settings = settingsOrDefault(frame);
  if (!settings.transmission.enabled) {
    passTextureReads_.clear();
    blendedTextureReads_.clear();
    cachedPassTextureReadSignature_ = std::numeric_limits<uint64_t>::max();
    cachedBlendedTextureReadSignature_ = std::numeric_limits<uint64_t>::max();
    return Result<bool, std::string>::makeResult(true);
  }
  auto initResult = ensureInitialized();
  if (initResult.hasError()) {
    return initResult;
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
          : frame.sharedResources.sceneDepthGraphTexture;
  frame.metrics.antiAliasing.taaTransmissionStableVisibilityDepth =
      stableVisibilityDepth;
  const MaterialTableSnapshot materialSnapshot =
      frame.resources->materialSnapshot();
  const bool topologyDirty =
      cachedScene_ != frame.scene ||
      cachedTopologyVersion_ != frame.scene->topologyVersion();
  const uint64_t modelMaterialBindingVersion =
      frame.resources->modelMaterialBindingVersion();
  const bool materialDirty =
      topologyDirty || cachedMaterialVersion_ != materialSnapshot.version ||
      cachedModelMaterialBindingVersion_ != modelMaterialBindingVersion;
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
    NURI_PROFILER_ZONE("TransmissionRenderer.cache_rebuild",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    rebuildSceneCache(*frame.sharedResources.sceneDrawDatabase, *frame.scene);
    NURI_PROFILER_ZONE_END();
    drawTemplatesRebuilt = true;
    cachedMaterialVersion_ = materialSnapshot.version;
    cachedModelMaterialBindingVersion_ = modelMaterialBindingVersion;
  } else if (geometryDirty) {
    cachedGeometryMutationVersion_ = geometryMutationVersion;
  }
  const EnvironmentHandles &environment = frame.scene->environment();
  const bool environmentDirty = cachedEnvironmentHandles_ != environment;
  if (environmentDirty || !environmentTextureAccessCacheValid_) {
    NURI_PROFILER_ZONE("TransmissionRenderer.env_texture_collect",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    collectForwardEnvironmentTextures(*frame.scene, *frame.resources,
                                      environmentTextureAccessHandles_);
    cachedEnvironmentHandles_ = environment;
    environmentTextureAccessCacheValid_ = true;
    NURI_PROFILER_ZONE_END();
  }
  if (materialDirty || !materialTextureAccessCacheValid_) {
    NURI_PROFILER_ZONE("TransmissionRenderer.material_texture_cache",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    collectForwardMaterialTextures(*frame.resources, meshDrawTemplates_,
                                   materialTextureAccessHandles_);
    NURI_PROFILER_ZONE_END();
    materialTextureAccessCacheValid_ = true;
  }
  if (environmentDirty || materialDirty || !staticPassTextureReadsValid_) {
    staticPassTextureReads_.clear();
    staticPassTextureReads_.reserve(environmentTextureAccessHandles_.size() +
                                    materialTextureAccessHandles_.size());
    for (const TextureHandle handle : environmentTextureAccessHandles_) {
      appendUniqueForwardHandle(staticPassTextureReads_, handle);
    }
    for (const TextureHandle handle : materialTextureAccessHandles_) {
      appendUniqueForwardHandle(staticPassTextureReads_, handle);
    }
    staticPassTextureReadsValid_ = true;
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
  preparedTransparentFeedbackCandidateCount_ = saturateToU32(std::count_if(
      meshDrawTemplates_.begin(), meshDrawTemplates_.end(),
      [](const MeshDrawTemplate &entry) { return entry.sortedFeedback; }));
  const bool hasSortedFeedbackDraws =
      preparedTransparentFeedbackCandidateCount_ != 0u;
  if (hasSortedFeedbackDraws) {
    auto feedbackResult = ensureTransparentFeedbackTextures(frame);
    if (feedbackResult.hasError()) {
      return feedbackResult;
    }
    auto feedbackPipelineResult = ensureFeedbackCopyPipeline(
        gpu_.getTextureFormat(preparedTransparentFeedbackTextures_[0]));
    if (feedbackPipelineResult.hasError()) {
      return feedbackPipelineResult;
    }
    frame.metrics.antiAliasing.transparentTransmissionFeedbackSourceAvailable =
        1u;
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
    const std::array rings{&instanceMatricesRing_, &instanceRemapRing_};
    const uint32_t ringCount =
        growDynamicBufferRings(gpu_.getSwapchainImageCount(), rings);
    instanceDataRingUploadVersions_.resize(
        ringCount, std::numeric_limits<uint64_t>::max());
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
    FrameData blendedFrameData =
        resolveForwardSceneFrameData(*sceneGpu, frameDataAddress);
    const auto &[feedbackFull, feedbackHalf, feedbackQuarter] =
        preparedTransparentFeedbackTextures_;
    blendedFrameData.sceneColorTexId =
        gpu_.getTextureBindlessIndex(feedbackFull);
    blendedFrameData.sceneColorHalfResTexId =
        gpu_.getTextureBindlessIndex(feedbackHalf);
    blendedFrameData.sceneColorQuarterResTexId =
        gpu_.getTextureBindlessIndex(feedbackQuarter);
    blendedFrameData.sceneColorSamplerId =
        gpu_.getLinearRepeatSamplerBindlessIndex(true, 1u);
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
      refreshDrawTemplateTransforms();
      NURI_PROFILER_ZONE_END();
    }
  }
  if (!meshDrawTemplates_.empty() && transformDirty) {
    rebuildForwardInstances(renderables, instanceMatrices_, instanceRemap_,
                            instanceDataRingUploadVersions_);
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
  const RasterPipelineState targetRasterState =
      depthFormat != Format::Count
          ? makeRasterPipelineState(
                DepthState{.compareOp = CompareOp::LessEqual,
                           .isDepthWriteEnabled = false},
                jitteredPostTaaDepthBias && jitterDepthBiasConstant != 0.0f,
                jitterDepthBiasConstant)
          : RasterPipelineState{};
  auto pipelineResult = ensurePipelines(
      gpu_.getTextureFormat(frameColorTexture), depthFormat, targetRasterState);
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
    auto instanceUpload = uploadForwardInstances(
        gpu_, instanceMatricesRing_, instanceRemapRing_, instanceMatrices_,
        instanceRemap_, instanceDataRingUploadVersions_, frameSlot,
        cachedTransformVersion_, animationSceneData != nullptr);
    if (instanceUpload.hasError()) {
      return instanceUpload;
    }
    NURI_PROFILER_ZONE_END();
    NURI_PROFILER_ZONE("TransmissionRenderer.mesh_draw_build",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    const float taaJitterMinLod = transmissionTaaJitterMinLod(frame, settings);
    frame.metrics.antiAliasing.taaTransmissionJitterMinLod = taaJitterMinLod;
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
    meshPushConstants_.reserve(meshDrawTemplates_.size());
    blendedPushConstants_.reserve(meshDrawTemplates_.size());
    blendedSortableDraws_.reserve(meshDrawTemplates_.size());
    const uint32_t renderableCount = saturateToU32(renderables.size());
    const bool hasDepthAttachment =
        (nuri::isValid(depthTexture) || nuri::isValid(sceneDepthGraphTexture));
    constexpr uint32_t debugFlags = 0u;
    const visibility_detail::FrustumPlanes transmissionFrustum =
        visibility_detail::buildCameraFrustumPlanes(frame.camera);
    const bool enableCpuFrustumCulling =
        settings.opaque.enableCpuFrustumCulling &&
        animationSceneData == nullptr;
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
      if (enableCpuFrustumCulling &&
          !visibility_detail::isVisible(
              visibility_detail::classifyTransformedBounds(
                  transmissionFrustum, entry.submesh->bounds,
                  entry.renderable->modelMatrix))) {
        continue;
      }
      const std::optional<SubmeshLod> lod =
          resolveForwardLod(*entry.submesh, settings.opaque.forcedMeshLod);
      if (!lod.has_value()) {
        continue;
      }
      BufferHandle vertexBuffer = nuri::isValid(entry.vertexBuffer)
                                      ? entry.vertexBuffer
                                      : entry.baseVertexBuffer;
      uint64_t vertexBufferAddress =
          vertexBuffer == entry.vertexBuffer
              ? entry.vertexBufferAddress
              : gpu_.getBufferDeviceAddress(vertexBuffer,
                                            entry.vertexBufferByteOffset);
      uint64_t vertexDecodeBufferAddress = entry.vertexDecodeBufferAddress;
      uint32_t vertexDecodeIndex = entry.vertexDecodeIndex;
      uint32_t packedVertexFormat = entry.packedVertexFormat;
      bool animatedOverrideApplied = false;
      if (animationSceneData != nullptr) {
        const AnimatedRenderableGeometryOverride &geometryOverride =
            animationSceneData
                ->geometryOverridesByRenderable[entry.instanceIndex];
        if (geometryOverride.enabled &&
            nuri::isValid(geometryOverride.vertexBuffer) &&
            animationOverrideCoversSubmesh(geometryOverride, *entry.submesh)) {
          const uint64_t overrideVertexAddress = gpu_.getBufferDeviceAddress(
              geometryOverride.vertexBuffer, geometryOverride.vertexByteOffset);
          vertexBuffer = geometryOverride.vertexBuffer;
          vertexBufferAddress = overrideVertexAddress;
          vertexDecodeBufferAddress = 0u;
          vertexDecodeIndex = 0u;
          packedVertexFormat =
              static_cast<uint32_t>(PackedVertexFormat::AnimatedFloat32);
          animatedOverrideApplied = true;
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
          .tessMaxFactor = sortedFeedbackDraw ? 0.0f : taaJitterMinLod,
          .debugVisualizationMode = debugFlags,
      };
      constants.tessNearDistance = entry.transmissionScale.x;
      constants.tessFarDistance = entry.transmissionScale.y;
      constants.tessMinFactor = entry.transmissionScale.z;
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
    appendUniqueForwardHandle(passDependencyBuffers_, sceneGpu->buffer);
    appendUniqueForwardHandle(passDependencyBuffers_,
                              materialGpu->headerBuffer);
    appendUniqueForwardHandle(passDependencyBuffers_,
                              materialGpu->clearcoatBuffer);
    appendUniqueForwardHandle(passDependencyBuffers_, materialGpu->sheenBuffer);
    appendUniqueForwardHandle(passDependencyBuffers_,
                              materialGpu->transmissionBuffer);
    appendUniqueForwardHandle(passDependencyBuffers_,
                              materialGpu->specularBuffer);
    appendUniqueForwardHandle(passDependencyBuffers_,
                              instanceMatricesBufferHandle);
    appendUniqueForwardHandle(passDependencyBuffers_,
                              instanceRemapRing_[frameSlot].buffer->handle());
    blendedDependencyBuffers_ = passDependencyBuffers_;
    if (nuri::isValid(blendedFrameDataBufferHandle)) {
      appendUniqueForwardHandle(blendedDependencyBuffers_,
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
      const uint64_t blendedTextureReadSignature = textureReadSignature(
          staticPassTextureReads_, preparedTransparentFeedbackTextures_);
      if (cachedBlendedTextureReadSignature_ != blendedTextureReadSignature ||
          blendedTextureReads_.empty()) {
        rebuildTextureReads(blendedTextureReads_, staticPassTextureReads_,
                            preparedTransparentFeedbackTextures_);
        cachedBlendedTextureReadSignature_ = blendedTextureReadSignature;
      }
    } else {
      blendedTextureReads_.clear();
      cachedBlendedTextureReadSignature_ = std::numeric_limits<uint64_t>::max();
    }
    NURI_PROFILER_ZONE_END();
    frame.metrics.antiAliasing.transparentTransmissionBlendDrawCount =
        saturateToU32(blendedSortableDraws_.size());
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
  return Result<bool, std::string>::makeResult(true);
}

bool TransmissionRenderer::hasPreparedTransmissionMainPass() const noexcept {
  return !passDrawItems_.empty();
}

Result<bool, std::string>
TransmissionRenderer::buildTransparentStageContribution(
    RenderFrameContext &frame, TransparentStageContribution &out) {
  (void)frame;
  out = {};
  out.sortableDraws = std::span<const TransparentStageSortableDraw>(
      blendedSortableDraws_.data(), blendedSortableDraws_.size());
  out.fixedDraws = {};
  out.dependencyBuffers = std::span<const BufferHandle>(
      blendedDependencyBuffers_.data(), blendedDependencyBuffers_.size());
  out.textureReads = std::span<const TextureHandle>(
      blendedTextureReads_.data(), blendedTextureReads_.size());
  if (!out.sortableDraws.empty()) {
    out.feedbackRefresh = TransparentStageFeedbackRefresh{
        .user = this,
        .appendRefresh =
            [](void *user, RenderFrameContext &frame,
               RenderGraphBuilder &graph) -> Result<bool, std::string> {
          return static_cast<TransmissionRenderer *>(user)
              ->appendTransparentFeedbackRefresh(frame, graph);
        },
        .mode = selectTransparentFeedbackRefreshMode(
            saturateToU32(out.sortableDraws.size())),
    };
  }
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
  const GpuTimingReport &timingReport = frame.gpuTiming;
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
  RenderGraphGraphicsPassDesc passDesc{};
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
        preparedDepthTexture_ ==
            frame.sharedResources.transmissionVisibilityDepthTexture) {
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
  passDesc.preResolvedDrawBuffers = cachedPreResolvedDrawBuffers_;
  passDesc.dependencyBuffers = std::span<const BufferHandle>(
      passDependencyBuffers_.data(), passDependencyBuffers_.size());
  passDesc.dependencyBufferAccessModes = passDependencyBufferAccessModes_;
  passDesc.dependencyTextures = passTextureReads_;
  passDesc.debugLabel = kTransmissionPassLabel;
  passDesc.debugColor = kTransmissionPassDebugColor;
  passDesc.borrowPayload = true;
  passDesc.gpuTimingScope = GpuTimingScope::Transmission;
  auto addResult = graph.addGraphicsPass(passDesc);
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
  struct ShaderSpec {
    std::string_view name;
    const std::filesystem::path *path;
    ShaderStage stage;
  };
  const std::array specs{
      ShaderSpec{"transmission_main", &transmissionVertexPath_,
                 ShaderStage::Vertex},
      ShaderSpec{"transmission_main", &transmissionFragmentPath_,
                 ShaderStage::Fragment},
      ShaderSpec{"transparent_transmission_feedback", &feedbackCopyVertexPath_,
                 ShaderStage::Vertex},
      ShaderSpec{"transparent_transmission_feedback",
                 &feedbackCopyFragmentPath_, ShaderStage::Fragment},
  };
  for (size_t index = 0; index < shaders_.size(); ++index) {
    auto compiler = Shader::create(specs[index].name, gpu_);
    auto result = compiler->compileFromFile(specs[index].path->string(),
                                            specs[index].stage);
    if (result.hasError()) {
      destroyShaders();
      return Result<bool, std::string>::makeError(result.error());
    }
    shaders_[index] = result.value();
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TransmissionRenderer::ensureFeedbackCopyPipeline(Format colorFormat) {
  if (nuri::isValid(feedbackCopyPipelineHandle_) &&
      feedbackCopyPipelineColorFormat_ == colorFormat) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (nuri::isValid(feedbackCopyPipelineHandle_)) {
    gpu_.destroyRenderPipeline(feedbackCopyPipelineHandle_);
    feedbackCopyPipelineHandle_ = {};
  }
  auto pipelineResult = gpu_.createRenderPipeline(
      fullscreenPipelineDesc(colorFormat, shaders_[2], shaders_[3]),
      "transparent_transmission_feedback_copy");
  if (pipelineResult.hasError()) {
    return Result<bool, std::string>::makeError(pipelineResult.error());
  }
  feedbackCopyPipelineHandle_ = pipelineResult.value();
  feedbackCopyPipelineColorFormat_ = colorFormat;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TransmissionRenderer::ensurePipelines(Format colorFormat, Format depthFormat,
                                      RasterPipelineState rasterState) {
  const RasterPipelineState targetRasterState =
      canonicalRasterPipelineState(rasterState);
  const bool meshPipelinesValid =
      std::ranges::all_of(meshPipelines_,
                          [](RenderPipelineHandle pipeline) {
                            return nuri::isValid(pipeline);
                          }) &&
      meshPipelineColorFormat_ == colorFormat &&
      meshPipelineDepthFormat_ == depthFormat &&
      meshPipelineRasterState_ == targetRasterState;
  if (meshPipelinesValid) {
    return Result<bool, std::string>::makeResult(true);
  }
  struct Variant {
    CullMode cull;
    bool blend;
    std::string_view name;
  };
  constexpr std::array variants{
      Variant{CullMode::Back, false, "transmission_mesh"},
      Variant{CullMode::None, false, "transmission_mesh_double_sided"},
      Variant{CullMode::Back, true, "transmission_mesh_blend"},
      Variant{CullMode::None, true, "transmission_mesh_blend_double_sided"},
  };
  std::array<RenderPipelineHandle, 4> pipelines{};
  for (size_t index = 0; index < pipelines.size(); ++index) {
    const Variant &variant = variants[index];
    auto result = gpu_.createRenderPipeline(
        meshPipelineDesc(colorFormat, depthFormat, shaders_[0], shaders_[1],
                         variant.cull, variant.blend, targetRasterState),
        variant.name);
    if (result.hasError()) {
      for (RenderPipelineHandle pipeline : pipelines) {
        if (nuri::isValid(pipeline)) {
          gpu_.destroyRenderPipeline(pipeline);
        }
      }
      return Result<bool, std::string>::makeError(result.error());
    }
    pipelines[index] = result.value();
  }
  const auto oldPipelines = std::exchange(meshPipelines_, pipelines);
  meshPipelineColorFormat_ = colorFormat;
  meshPipelineDepthFormat_ = depthFormat;
  meshPipelineRasterState_ = targetRasterState;
  for (RenderPipelineHandle pipeline : oldPipelines) {
    if (nuri::isValid(pipeline)) {
      gpu_.destroyRenderPipeline(pipeline);
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TransmissionRenderer::ensureInstanceMatricesRingCapacity(size_t requiredBytes) {
  return ensureDynamicBufferRingCapacity(
      gpu_, instanceMatricesRing_,
      BufferDesc{.usage = BufferUsage::Storage,
                 .storage = Storage::Device,
                 .size = requiredBytes},
      "transmission_instance_matrices", [this](size_t i) {
        instanceDataRingUploadVersions_[i] =
            std::numeric_limits<uint64_t>::max();
      });
}

Result<bool, std::string>
TransmissionRenderer::ensureInstanceRemapRingCapacity(size_t requiredBytes) {
  return ensureDynamicBufferRingCapacity(
      gpu_, instanceRemapRing_,
      BufferDesc{.usage = BufferUsage::Storage,
                 .storage = Storage::Device,
                 .size = requiredBytes},
      "transmission_instance_remap", [this](size_t i) {
        instanceDataRingUploadVersions_[i] =
            std::numeric_limits<uint64_t>::max();
      });
}

Result<bool, std::string>
TransmissionRenderer::ensureBlendedFrameDataRingCapacity(size_t requiredBytes) {
  const std::array rings{&blendedFrameDataRing_};
  growDynamicBufferRings(gpu_.getSwapchainImageCount(), rings);
  return ensureDynamicBufferRingCapacity(
      gpu_, blendedFrameDataRing_,
      BufferDesc{.usage = BufferUsage::Storage,
                 .storage = Storage::HostVisible,
                 .size = requiredBytes},
      "transparent_transmission_frame_data");
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
  for (size_t level = 0; level < preparedTransparentFeedbackTextures_.size();
       ++level) {
    preparedTransparentFeedbackTextures_[level] =
        transparentFeedbackTextures_[level][frameSlot];
  }
  return Result<bool, std::string>::makeResult(true);
}

TransparentStageFeedbackRefreshMode
TransmissionRenderer::selectTransparentFeedbackRefreshMode(
    uint32_t visibleDrawCount) {
  if (preparedTransparentFeedbackCandidateCount_ <=
      kMaxExactTransparentFeedbackDraws) {
    return TransparentStageFeedbackRefreshMode::BeforeEachDraw;
  }
  if (!loggedTransparentFeedbackFallbackWarning_) {
    loggedTransparentFeedbackFallbackWarning_ = true;
    NURI_LOG_WARNING(
        "TransmissionRenderer: %u visible blended transmission draw(s) from "
        "%u candidate(s); candidate count exceeds exact feedback budget %u; "
        "using one shared feedback refresh for this frame",
        visibleDrawCount, preparedTransparentFeedbackCandidateCount_,
        kMaxExactTransparentFeedbackDraws);
  }
  return TransparentStageFeedbackRefreshMode::OnceBeforeFirstDraw;
}

Result<bool, std::string>
TransmissionRenderer::appendTransparentFeedbackRefresh(
    RenderFrameContext &frame, RenderGraphBuilder &graph) {
  const TextureHandle frameColorTexture =
      frame.sharedResources.frameColorTexture;
  const auto &[feedbackFull, feedbackHalf, feedbackQuarter] =
      preparedTransparentFeedbackTextures_;
  if (!nuri::isValid(frameColorTexture) || !nuri::isValid(feedbackFull) ||
      !nuri::isValid(feedbackHalf) || !nuri::isValid(feedbackQuarter)) {
    return Result<bool, std::string>::makeError(
        "TransmissionRenderer::appendTransparentFeedbackRefresh: transparent "
        "transmission feedback pyramid is incomplete");
  }
  if (!nuri::isValid(feedbackCopyPipelineHandle_)) {
    return Result<bool, std::string>::makeError(
        "TransmissionRenderer::appendTransparentFeedbackRefresh: feedback "
        "copy pipeline is invalid");
  }
  const auto appendCopyPass =
      [this, &graph](TextureHandle source, TextureHandle destination,
                     std::string_view debugLabel,
                     bool downsample) -> Result<bool, std::string> {
    const uint32_t sourceTexId = gpu_.getTextureBindlessIndex(source);
    if (sourceTexId == kInvalidTextureBindlessIndex) {
      return Result<bool, std::string>::makeError(
          "TransmissionRenderer::appendTransparentFeedbackRefresh: invalid "
          "source texture bindless index");
    }
    const CopyPushConstants pushConstants{
        .sourceTexId = sourceTexId,
        .sourceSamplerId = gpu_.getLinearRepeatSamplerBindlessIndex(true, 1u),
        .flags = downsample ? kSceneCopyFlagDownsample : 0u,
        .reserved0 = 0u,
    };
    const DrawItem draw{
        .command = DrawCommandType::Direct,
        .pipeline = feedbackCopyPipelineHandle_,
        .vertexCount = 3u,
        .instanceCount = 1u,
        .pushConstants = std::span<const std::byte>(
            reinterpret_cast<const std::byte *>(&pushConstants),
            sizeof(pushConstants)),
        .debugLabel = debugLabel,
        .debugColor = kTransparentPassDebugColor,
    };
    auto colorImportResult =
        graph.importTexture(destination, "transparent_transmission_feedback");
    if (colorImportResult.hasError()) {
      return Result<bool, std::string>::makeError(colorImportResult.error());
    }
    RenderGraphGraphicsPassDesc passDesc{};
    passDesc.color = {.loadOp = LoadOp::Clear,
                      .storeOp = StoreOp::Store,
                      .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
    passDesc.colorTexture = colorImportResult.value();
    passDesc.draws = std::span<const DrawItem>(&draw, 1u);
    passDesc.dependencyTextures = std::span<const TextureHandle>(&source, 1u);
    passDesc.debugLabel = debugLabel;
    passDesc.debugColor = kTransparentPassDebugColor;
    auto addResult = graph.addGraphicsPass(passDesc);
    if (addResult.hasError()) {
      return Result<bool, std::string>::makeError(addResult.error());
    }
    return Result<bool, std::string>::makeResult(true);
  };
  const std::array textures{frameColorTexture, feedbackFull, feedbackHalf,
                            feedbackQuarter};
  for (size_t index = 0; index < kTransparentTransmissionFeedbackLabels.size();
       ++index) {
    auto result = appendCopyPass(textures[index], textures[index + 1u],
                                 kTransparentTransmissionFeedbackLabels[index],
                                 index != 0u);
    if (result.hasError()) {
      return result;
    }
  }
  for (size_t index = 0; index < kTransparentTransmissionFeedbackLabels.size();
       ++index) {
    publishRequestedCapture(
        frame, gpu_, kTransparentTransmissionFeedbackCaptureNames[index],
        textures[index + 1u], RenderCaptureValueKind::LinearHdrColor,
        RenderCaptureLifetimeClass::FeaturePersistentTexture, "linear_hdr",
        "hdr_color", kTransparentTransmissionFeedbackLabels[index]);
  }
  ++frame.metrics.antiAliasing.transparentTransmissionFeedbackRefreshCount;
  return Result<bool, std::string>::makeResult(true);
}

void TransmissionRenderer::rebuildSceneCache(const SceneDrawDatabase &database,
                                             const RenderScene &scene) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_DRAW);
  meshDrawTemplates_.clear();
  for (const SceneDrawRecord &draw : database.draws()) {
    if (draw.transmission) {
      meshDrawTemplates_.emplace_back(draw);
    }
  }
  preResolvedTemplateBuffers_.clear();
  preResolvedTemplateBuffers_.reserve(meshDrawTemplates_.size() * 3u);
  for (const MeshDrawTemplate &entry : meshDrawTemplates_) {
    appendUniqueForwardHandle(preResolvedTemplateBuffers_,
                              entry.baseVertexBuffer);
    appendUniqueForwardHandle(preResolvedTemplateBuffers_,
                              entry.baseVertexDecodeBuffer);
    appendUniqueForwardHandle(preResolvedTemplateBuffers_, entry.indexBuffer);
  }
  cachedPreResolvedDrawBuffers_.clear();
  cachedPreResolvedDrawBufferSignature_ = std::numeric_limits<uint64_t>::max();
  cachedScene_ = &scene;
  cachedTopologyVersion_ = scene.topologyVersion();
  cachedGeometryMutationVersion_ = gpu_.geometryMutationVersion();
}

void TransmissionRenderer::refreshDrawTemplateTransforms() {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_DRAW);
  for (MeshDrawTemplate &entry : meshDrawTemplates_) {
    const Renderable &renderable = *entry.renderable;
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
  cachedModelMaterialBindingVersion_ = std::numeric_limits<uint64_t>::max();
  cachedTransformVersion_ = std::numeric_limits<uint64_t>::max();
  cachedGeometryMutationVersion_ = std::numeric_limits<uint64_t>::max();
  loggedTransparentFeedbackFallbackWarning_ = false;
  cachedEnvironmentHandles_ = {};
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
  preparedTransparentFeedbackTextures_ = {};
  preparedTransparentFeedbackCandidateCount_ = 0u;
}

void TransmissionRenderer::destroyPipelineState() {
  if (nuri::isValid(feedbackCopyPipelineHandle_)) {
    gpu_.destroyRenderPipeline(feedbackCopyPipelineHandle_);
    feedbackCopyPipelineHandle_ = {};
  }
  for (RenderPipelineHandle &pipeline : meshPipelines_) {
    if (nuri::isValid(pipeline)) {
      gpu_.destroyRenderPipeline(pipeline);
    }
    pipeline = {};
  }
  meshPipelineColorFormat_ = Format::Count;
  meshPipelineDepthFormat_ = Format::Count;
  meshPipelineRasterState_ = {};
  feedbackCopyPipelineColorFormat_ = Format::Count;
}

void TransmissionRenderer::destroyShaders() {
  for (ShaderHandle &shader : shaders_) {
    if (nuri::isValid(shader)) {
      gpu_.destroyShaderModule(shader);
    }
    shader = {};
  }
}

void TransmissionRenderer::destroyBuffers() {
  retireDynamicBufferRing(instanceMatricesRing_);
  retireDynamicBufferRing(instanceRemapRing_);
  retireDynamicBufferRing(blendedFrameDataRing_);
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

RenderPipelineHandle
TransmissionRenderer::selectMeshPipeline(bool doubleSided,
                                         bool useBlendPipeline) const {
  return meshPipelines_[(useBlendPipeline ? 2u : 0u) + (doubleSided ? 1u : 0u)];
}

void registerTransmissionStage(RenderPipeline &pipeline, GPUDevice &gpu,
                               const RuntimeOpaqueShaderConfig &config,
                               std::pmr::memory_resource *memory,
                               SceneDrawDatabase *database) {
  if (!database) {
    pipeline.addProvider(std::make_unique<SceneDrawDatabase>(gpu, memory));
  }
  auto owner = std::make_unique<TransmissionRenderer>(gpu, config, memory);
  owner->onAttach();
  auto *renderer = pipeline.addComponent(
      std::move(owner),
      PipelineComponentDesc{
          .publish =
              [](void *state, FrameBuildContext &ctx) {
                static_cast<TransmissionRenderer *>(state)->publishFrameData(
                    ctx.frame);
                return Result<bool, std::string>::makeResult(true);
              },
          .prepare =
              [](void *state, FrameBuildContext &ctx) {
                return static_cast<TransmissionRenderer *>(state)
                    ->prepareTransmissionPasses(ctx.frame);
              },
      });
  pipeline.addStage(PipelineStageDesc{
      .componentName = "TransmissionFeature",
      .name = "TransmissionMainPass",
      .state = renderer,
      .enabled =
          [](const void *state, const FrameBuildContext &ctx) {
            return (!ctx.frame.settings ||
                    renderSettingsOrDefault(ctx.frame).transmission.enabled) &&
                   static_cast<const TransmissionRenderer *>(state)
                       ->hasPreparedTransmissionMainPass();
          },
      .build =
          [](void *state, FrameBuildContext &ctx) {
            return static_cast<TransmissionRenderer *>(state)
                ->appendTransmissionMainPass(ctx.frame, ctx.graph);
          },
  });
}

} // namespace nuri
