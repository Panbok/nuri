#include "nuri/gfx/pipeline/providers/frame_composition_provider.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/frame/presentation_aa_plan.h"
#include "nuri/gfx/frame/render_capture.h"
#include "nuri/resources/gpu/resource_manager.h"
namespace nuri {
namespace {
uint32_t levelDimensions(uint32_t base, uint32_t mipLevel) {
  return std::max(1u, base >> std::min(mipLevel, 31u));
}
TextureDesc makeTextureDesc(Format format, uint32_t width, uint32_t height,
                            TextureUsage usage) {
  return TextureDesc{
      .type = TextureType::Texture2D,
      .format = format,
      .dimensions = {.width = width, .height = height, .depth = 1u},
      .usage = usage,
      .storage = Storage::Device,
      .numLayers = 1u,
      .numSamples = 1u,
      .numMipLevels = 1u,
      .data = {},
      .dataNumMipLevels = 1u,
      .generateMipmaps = false,
  };
}

bool needsMsaaResolve(const FrameBuildContext &ctx) {
  const AntiAliasingFrameMetrics &metrics = ctx.frame.metrics.antiAliasing;
  return ctx.frame.presentationAA.coverage != CoverageMode::Sample1 &&
         !(metrics.msaaColorResolveTargetBound &&
           metrics.msaaDepthResolveTargetBound);
}

Result<bool, std::string> appendMsaaResolve(FrameBuildContext &ctx,
                                            GPUDevice &gpu) {
  const auto import = [&ctx](RenderGraphTextureId id, TextureHandle handle,
                             std::string_view name) {
    return nuri::isValid(id) ? id
                             : ctx.graph.importTexture(handle, name).value();
  };
  ctx.shared[FrameTextureSlot::MsaaSceneColor].graph =
      import(ctx.shared[FrameTextureSlot::MsaaSceneColor].graph,
             ctx.shared[FrameTextureSlot::MsaaSceneColor].texture,
             "msaa_resolve_scene_color");
  ctx.shared[FrameTextureSlot::MsaaSceneDepth].graph =
      import(ctx.shared[FrameTextureSlot::MsaaSceneDepth].graph,
             ctx.shared[FrameTextureSlot::MsaaSceneDepth].texture,
             "msaa_resolve_scene_depth");
  const RenderGraphTextureId colorTarget =
      ctx.graph
          .importTexture(ctx.shared[FrameTextureSlot::SceneColor].texture,
                         "msaa_resolve_scene_color_target")
          .value();
  const RenderGraphTextureId depthTarget =
      ctx.graph
          .importTexture(ctx.shared[FrameTextureSlot::SceneDepth].texture,
                         "msaa_resolve_scene_depth_target")
          .value();
  [[maybe_unused]] const RenderGraphPassId resolvePass =
      ctx.graph
          .addGraphicsPass(RenderGraphGraphicsPassDesc{
              .color = {.loadOp = LoadOp::Load,
                        .storeOp = StoreOp::MsaaResolve,
                        .resolveMode = ResolveMode::Average},
              .colorTexture =
                  ctx.shared[FrameTextureSlot::MsaaSceneColor].graph,
              .colorResolveTexture = colorTarget,
              .depth = {.loadOp = LoadOp::Load,
                        .storeOp = StoreOp::MsaaResolve,
                        .resolveMode = ResolveMode::Min},
              .depthTexture =
                  ctx.shared[FrameTextureSlot::MsaaSceneDepth].graph,
              .depthResolveTexture = depthTarget,
              .gpuTimingScope = GpuTimingScope::MsaaResolve,
              .debugLabel = "MSAA Resolve Pass",
              .debugColor = 0xff44ccff,
              .markImplicitOutputSideEffect = true,
          })
          .value();
  ctx.shared[FrameTextureSlot::SceneColor].graph = colorTarget;
  ctx.shared[FrameTextureSlot::SceneDepth].graph = depthTarget;
  AntiAliasingFrameMetrics &metrics = ctx.frame.metrics.antiAliasing;
  metrics.msaaResolvePassCount = 1u;
  metrics.msaaColorResolveCount = 1u;
  metrics.msaaDepthResolveCount = 1u;
  metrics.msaaResolvedSampleCount =
      coverageSampleCount(ctx.frame.presentationAA.coverage);
  metrics.msaaResolvePlacement = MsaaResolvePlacement::ExplicitPass;
  metrics.msaaColorGraphPublished = true;
  metrics.msaaDepthGraphPublished = true;
  metrics.msaaColorResolveTargetBound = true;
  metrics.msaaDepthResolveTargetBound = true;
  publishRequestedCapture(ctx.frame, gpu, "scene_color_hdr",
                          ctx.shared[FrameTextureSlot::SceneColor].texture,
                          RenderCaptureValueKind::LinearHdrColor,
                          RenderCaptureLifetimeClass::FrameSharedRingTexture,
                          "linear_hdr", "hdr_color", "MSAA Resolve Pass");
  publishRequestedCapture(ctx.frame, gpu, "scene_depth",
                          ctx.shared[FrameTextureSlot::SceneDepth].texture,
                          RenderCaptureValueKind::Depth,
                          RenderCaptureLifetimeClass::FrameSharedRingTexture,
                          "linear_depth", "depth", "MSAA Resolve Pass");
  metrics.msaaResolveBandwidthEstimateBytes =
      metrics.msaaResolveReadEstimateBytes;
  if (hasGpuTimingScope(ctx.frame.gpuTiming, GpuTimingScope::MsaaResolve)) {
    metrics.msaaResolveGpuTimeMs =
        ctx.frame.gpuTiming[GpuTimingScope::MsaaResolve].timeMs;
    metrics.msaaResolveGpuTimingSourceFrameIndex =
        ctx.frame.gpuTiming[GpuTimingScope::MsaaResolve].sourceFrameIndex;
    metrics.msaaResolveGpuTimingAvailable = 1u;
  }
  return Result<bool, std::string>::makeResult(true);
}
} // namespace

void registerMsaaResolveStage(RenderPipeline &pipeline, GPUDevice &gpu) {
  pipeline.addStage(PipelineStageDesc{
      .componentName = "MsaaResolveFeature",
      .name = "MsaaResolvePass",
      .state = &gpu,
      .enabled =
          [](const void *, const FrameBuildContext &ctx) {
            return needsMsaaResolve(ctx);
          },
      .build =
          [](void *state, FrameBuildContext &ctx) {
            return appendMsaaResolve(ctx, *static_cast<GPUDevice *>(state));
          },
  });
}

FrameCompositionProvider::FrameCompositionProvider(
    GPUDevice &gpu, std::pmr::memory_resource *memory)
    : gpu_(gpu),
      memory_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      textureRings_(memory_) {
  textureRings_.reserve(static_cast<size_t>(Ring::Count));
  for (size_t i = 0; i < static_cast<size_t>(Ring::Count); ++i) {
    textureRings_.emplace_back();
  }
}

FrameCompositionProvider::~FrameCompositionProvider() {
  for (size_t i = 0; i < textureRings_.size(); ++i) {
    destroyTextureRing(static_cast<Ring>(i));
  }
}

Result<bool, std::string>
FrameCompositionProvider::prepare(FrameBuildContext &ctx) {
  NURI_PROFILER_FUNCTION();
  ctx.shared.textureRequirements |= kBaselineFrameTextureRequirements;
  ctx.shared.historyWriteRequirements = FrameTextureRequirementFlags::None;
  ctx.shared[FrameTextureSlot::SceneColor].graph = {};
  ctx.shared[FrameTextureSlot::FrameColor].graph = {};
  ctx.shared[FrameTextureSlot::PresentCapture].graph = {};
  ctx.shared[FrameTextureSlot::SceneDepth].graph = {};
  ctx.shared[FrameHistoryTextureSlot::PreviousSceneDepth].graph = {};
  ctx.shared[FrameTextureSlot::TransmissionVisibilityDepth].texture = {};
  ctx.shared[FrameTextureSlot::TransmissionVisibilityDepth].graph = {};
  ctx.shared[FrameTextureSlot::MsaaSceneColor].graph = {};
  ctx.shared[FrameTextureSlot::MsaaSceneDepth].graph = {};
  ctx.shared[FrameTextureSlot::Normal].graph = {};
  ctx.shared[FrameTextureSlot::AmbientOcclusion].graph = {};
  ctx.shared[FrameTextureSlot::DdgiOpaqueSurfaceCache].graph = {};
  ctx.shared[FrameHistoryTextureSlot::PreviousAmbientOcclusion].texture = {};
  ctx.shared.sceneDepthPyramidGraphTextures = {};
  ctx.shared[FrameTextureSlot::MotionVector].graph = {};
  ctx.shared[FrameHistoryTextureSlot::PreviousMotionVector].graph = {};
  ctx.shared[FrameTextureSlot::ReactiveMask].graph = {};
  ctx.shared[FrameTextureSlot::MotionClass].graph = {};
  ctx.shared.historyColorReadValid = false;
  ctx.shared[FrameHistoryTextureSlot::ExposureRead].graph = {};
  ctx.shared[FrameHistoryTextureSlot::ExposureWrite].graph = {};
  ctx.shared.exposureHistoryValid = false;
  const RenderSettings &settings = ctx.frame.settings;
  if (isRenderCaptureRequested(ctx.frame, "final_color")) {
    ctx.shared.textureRequirements |=
        FrameTextureRequirementFlags::PresentCapture;
  }
  const CoverageMode coverage = ctx.frame.presentationAA.coverage;
  const uint32_t sampleCount = coverageSampleCount(coverage);
  if (coverage != CoverageMode::Sample1) {
    ctx.shared.textureRequirements |=
        FrameTextureRequirementFlags::MsaaSceneColor |
        FrameTextureRequirementFlags::MsaaSceneDepth;
  }
  auto ensureResult = ensureTextures(ctx.shared.textureRequirements, coverage);
  if (ensureResult.hasError()) {
    return ensureResult;
  }
  if (historyRegistry_.lease().pendingCommit &&
      historyRegistry_.lease().frameIndex != ctx.frame.frameIndex &&
      ctx.frame.temporalFrameService == nullptr) {
    historyRegistry_.abandonFrame(historyRegistry_.lease().frameIndex);
  }
  auto historyLeaseResult = historyRegistry_.prepareFrame(
      ctx.frame.frameIndex, std::max(textureRingCount_, 2u));
  if (historyLeaseResult.hasError()) {
    return Result<bool, std::string>::makeError(historyLeaseResult.error());
  }
  const HistoryLease historyLease = historyLeaseResult.value();
  pendingHistoryRequirements_ = ctx.shared.textureRequirements;
  const bool declaredHistoryValid =
      historyLease.readValid || ctx.frame.camera.historyValid;
  const uint32_t historyReadSlot =
      historyLease.readValid
          ? historyLease.readSlot
          : (historyLease.writeSlot + 1u) % std::max(textureRingCount_, 2u);
  const auto historyRequirementWasWritten =
      [&](FrameTextureRequirementFlags requirement) {
        return hasFrameTextureRequirementFlag(
            historyLease.readValid ? committedHistoryRequirements_
                                   : pendingHistoryRequirements_,
            requirement);
      };
  ctx.shared[FrameTextureSlot::SceneColor].texture =
      currentTexture(Ring::SceneColor, ctx.frame.frameIndex);
  ctx.shared.sceneColorHalfResTexture =
      currentTexture(Ring::SceneColorHalf, ctx.frame.frameIndex);
  ctx.shared.sceneColorQuarterResTexture =
      currentTexture(Ring::SceneColorQuarter, ctx.frame.frameIndex);
  ctx.shared[FrameTextureSlot::FrameColor].texture =
      currentTexture(Ring::FrameColor, ctx.frame.frameIndex);
  ctx.shared[FrameTextureSlot::PresentCapture].texture =
      currentTexture(Ring::PresentCapture, ctx.frame.frameIndex);
  ctx.shared[FrameTextureSlot::SceneDepth].texture =
      currentTexture(Ring::SceneDepth, historyLease.writeSlot);
  ctx.shared[FrameHistoryTextureSlot::PreviousSceneDepth].texture =
      hasTemporalCameraContinuity(ctx.frame.camera) && declaredHistoryValid &&
              historyRequirementWasWritten(
                  FrameTextureRequirementFlags::SceneDepth)
          ? currentTexture(Ring::SceneDepth, historyReadSlot)
          : TextureHandle{};
  ctx.shared[FrameTextureSlot::MsaaSceneColor].texture =
      currentTexture(Ring::MsaaSceneColor, ctx.frame.frameIndex);
  ctx.shared[FrameTextureSlot::MsaaSceneDepth].texture =
      currentTexture(Ring::MsaaSceneDepth, ctx.frame.frameIndex);
  ctx.shared[FrameHistoryTextureSlot::ColorRead].texture =
      currentTexture(Ring::HistoryColor, historyReadSlot);
  ctx.shared[FrameHistoryTextureSlot::ColorWrite].texture =
      currentTexture(Ring::HistoryColor, historyLease.writeSlot);
  ctx.shared.historyColorReadValid =
      declaredHistoryValid &&
      historyRequirementWasWritten(
          FrameTextureRequirementFlags::HistoryColor) &&
      nuri::isValid(ctx.shared[FrameHistoryTextureSlot::ColorRead].texture);
  ctx.shared[FrameTextureSlot::MotionVector].texture =
      currentTexture(Ring::MotionVectors, historyLease.writeSlot);
  ctx.shared[FrameHistoryTextureSlot::PreviousMotionVector].texture =
      hasTemporalCameraContinuity(ctx.frame.camera) && declaredHistoryValid &&
              historyRequirementWasWritten(
                  FrameTextureRequirementFlags::MotionVectors)
          ? currentTexture(Ring::MotionVectors, historyReadSlot)
          : TextureHandle{};
  ctx.shared[FrameTextureSlot::ReactiveMask].texture =
      currentTexture(Ring::ReactiveMask, ctx.frame.frameIndex);
  ctx.shared[FrameTextureSlot::MotionClass].texture =
      currentTexture(Ring::MotionClass, ctx.frame.frameIndex);
  ctx.shared[FrameTextureSlot::Normal].texture =
      currentTexture(Ring::Normals, ctx.frame.frameIndex);
  ctx.shared[FrameTextureSlot::AmbientOcclusion].texture =
      currentTexture(Ring::AmbientOcclusion, historyLease.writeSlot);
  ctx.shared[FrameTextureSlot::DdgiOpaqueSurfaceCache].texture =
      currentTexture(Ring::DDGIOpaqueSurfaceCache, ctx.frame.frameIndex);
  ctx.shared[FrameHistoryTextureSlot::PreviousAmbientOcclusion].texture =
      hasTemporalCameraContinuity(ctx.frame.camera) && declaredHistoryValid &&
              historyRequirementWasWritten(
                  FrameTextureRequirementFlags::AmbientOcclusion)
          ? currentTexture(Ring::AmbientOcclusion, historyReadSlot)
          : TextureHandle{};
  ctx.shared[FrameHistoryTextureSlot::ExposureRead].texture =
      currentTexture(Ring::Exposure, historyReadSlot);
  ctx.shared[FrameHistoryTextureSlot::ExposureWrite].texture =
      currentTexture(Ring::Exposure, historyLease.writeSlot);
  ctx.shared.exposureHistoryValid =
      declaredHistoryValid &&
      historyRequirementWasWritten(FrameTextureRequirementFlags::Exposure) &&
      nuri::isValid(ctx.shared[FrameHistoryTextureSlot::ExposureRead].texture);
  const auto ringSize = [this](Ring ring) {
    return textureRings_[static_cast<size_t>(ring)].size();
  };
  const auto allocations = [this](Ring ring) {
    return allocationCounts_[static_cast<size_t>(ring)];
  };
  const auto reallocations = [this](Ring ring) {
    return reallocationCounts_[static_cast<size_t>(ring)];
  };
  HDRPostProcessFrameMetrics &hdrMetrics = ctx.frame.metrics.hdrPostProcess;
  hdrMetrics.exposureHistoryValid = ctx.shared.exposureHistoryValid;
  hdrMetrics.exposureTextureAllocationCount = allocations(Ring::Exposure);
  hdrMetrics.exposureTextureReallocationCount = reallocations(Ring::Exposure);
  hdrMetrics.exposureHistoryAllocationCount = allocations(Ring::Exposure);
  hdrMetrics.exposureHistoryReallocationCount = reallocations(Ring::Exposure);
  AmbientOcclusionFrameMetrics &aoMetrics = ctx.frame.metrics.ambientOcclusion;
  aoMetrics.inputMode = ctx.frame.ambientOcclusion.inputMode;
  aoMetrics.workingResolution = ctx.frame.ambientOcclusion.workingResolution;
  aoMetrics.normalFormat =
      nuri::isValid(ctx.shared[FrameTextureSlot::Normal].texture)
          ? kFrameCompositionNormalFormat
          : Format::Count;
  aoMetrics.ambientOcclusionFormat = kFrameCompositionAmbientOcclusionFormat;
  aoMetrics.width = framebufferWidth_;
  aoMetrics.height = framebufferHeight_;
  aoMetrics.normalsAllocated =
      nuri::isValid(ctx.shared[FrameTextureSlot::Normal].texture);
  aoMetrics.ambientOcclusionAllocated =
      nuri::isValid(ctx.shared[FrameTextureSlot::AmbientOcclusion].texture);
  aoMetrics.temporalHistoryValid = nuri::isValid(
      ctx.shared[FrameHistoryTextureSlot::PreviousAmbientOcclusion].texture);
  aoMetrics.normalTextureAllocationCount = allocations(Ring::Normals);
  aoMetrics.normalTextureReallocationCount = reallocations(Ring::Normals);
  aoMetrics.ambientOcclusionTextureAllocationCount =
      allocations(Ring::AmbientOcclusion);
  aoMetrics.ambientOcclusionTextureReallocationCount =
      reallocations(Ring::AmbientOcclusion);
  const uint64_t fullResPixels = static_cast<uint64_t>(framebufferWidth_) *
                                 static_cast<uint64_t>(framebufferHeight_);
  aoMetrics.normalTextureBytes =
      aoMetrics.normalsAllocated
          ? fullResPixels * static_cast<uint64_t>(
                                formatTexelBytes(kFrameCompositionNormalFormat))
          : 0u;
  aoMetrics.ambientOcclusionTextureBytes =
      aoMetrics.ambientOcclusionAllocated
          ? fullResPixels * static_cast<uint64_t>(formatTexelBytes(
                                kFrameCompositionAmbientOcclusionFormat))
          : 0u;
  aoMetrics.normalTextureCount = static_cast<uint32_t>(ringSize(Ring::Normals));
  aoMetrics.ambientOcclusionTextureCount =
      static_cast<uint32_t>(ringSize(Ring::AmbientOcclusion));
  aoMetrics.textureCount =
      aoMetrics.normalTextureCount + aoMetrics.ambientOcclusionTextureCount;
  aoMetrics.totalTextureBytes =
      aoMetrics.normalTextureBytes *
          static_cast<uint64_t>(aoMetrics.normalTextureCount) +
      aoMetrics.ambientOcclusionTextureBytes *
          static_cast<uint64_t>(aoMetrics.ambientOcclusionTextureCount);
  aoMetrics.providerTextureBytes = aoMetrics.totalTextureBytes;
  aoMetrics.logicalActiveTextureBytes =
      aoMetrics.normalTextureBytes + aoMetrics.ambientOcclusionTextureBytes;
  AntiAliasingFrameMetrics &aaMetrics = ctx.frame.metrics.antiAliasing;
  aaMetrics.postAAPlan = ctx.frame.presentationAA.postAA;
  aaMetrics.postAA.specularSelected =
      aaMetrics.postAAPlan.specular == PostAASpecularAlgorithm::BakedClean;
  aaMetrics.msaaEnabled = coverage != CoverageMode::Sample1;
  aaMetrics.msaaSampleCount = sampleCount;
  aaMetrics.msaaAlphaCoverageRequested = isMsaaMode(settings.antiAliasing.mode);
  aaMetrics.msaaSpatialCleanupRequested =
      aaMetrics.postAAPlan.requested &&
      aaMetrics.postAAPlan.spatial == PostAASpatialAlgorithm::Smaa1x;
  aaMetrics.msaaSpatialCleanupEnabled =
      aaMetrics.postAAPlan.active &&
      aaMetrics.postAAPlan.spatial == PostAASpatialAlgorithm::Smaa1x;
  const PoolStats poolStats = ctx.resources.stats();
  aaMetrics.normalVarianceContractMaterialsLive =
      poolStats.normalVarianceContractMaterialsLive;
  aaMetrics.normalVarianceContractTexturesLive =
      poolStats.normalVarianceContractTexturesLive;
  aaMetrics.normalVarianceUnavailableSlotsLive =
      aaMetrics.postAA.specularSelected
          ? poolStats.normalVarianceUnavailableSlotsLive
          : 0u;
  aaMetrics.normalVarianceContractTextureBytesLive =
      poolStats.normalVarianceContractTextureBytesLive;
  aaMetrics.msaaResolvePlacement = aaMetrics.msaaEnabled
                                       ? MsaaResolvePlacement::ExplicitPass
                                       : MsaaResolvePlacement::None;
  aaMetrics.msaaWidth = framebufferWidth_;
  aaMetrics.msaaHeight = framebufferHeight_;
  aaMetrics.msaaColorAllocated =
      nuri::isValid(ctx.shared[FrameTextureSlot::MsaaSceneColor].texture);
  aaMetrics.msaaDepthAllocated =
      nuri::isValid(ctx.shared[FrameTextureSlot::MsaaSceneDepth].texture);
  aaMetrics.msaaColorTextureCount =
      static_cast<uint32_t>(ringSize(Ring::MsaaSceneColor));
  aaMetrics.msaaDepthTextureCount =
      static_cast<uint32_t>(ringSize(Ring::MsaaSceneDepth));
  aaMetrics.msaaRingSlots = std::max(aaMetrics.msaaColorTextureCount,
                                     aaMetrics.msaaDepthTextureCount);
  aaMetrics.msaaColorAllocationCount = allocations(Ring::MsaaSceneColor);
  aaMetrics.msaaColorReallocationCount = reallocations(Ring::MsaaSceneColor);
  aaMetrics.msaaDepthAllocationCount = allocations(Ring::MsaaSceneDepth);
  aaMetrics.msaaDepthReallocationCount = reallocations(Ring::MsaaSceneDepth);
  const uint64_t msaaPixelCount = static_cast<uint64_t>(framebufferWidth_) *
                                  static_cast<uint64_t>(framebufferHeight_);
  aaMetrics.msaaExtentWidth = framebufferWidth_;
  aaMetrics.msaaExtentHeight = framebufferHeight_;
  aaMetrics.msaaColorTexelBytes =
      formatTexelBytes(kFrameCompositionSceneColorFormat);
  aaMetrics.msaaDepthTexelBytes =
      formatTexelBytes(kFrameCompositionDepthFormat);
  aaMetrics.msaaColorTextureBytes =
      aaMetrics.msaaColorAllocated
          ? msaaPixelCount *
                static_cast<uint64_t>(
                    formatTexelBytes(kFrameCompositionSceneColorFormat)) *
                static_cast<uint64_t>(sampleCount)
          : 0u;
  aaMetrics.msaaDepthTextureBytes =
      aaMetrics.msaaDepthAllocated ? msaaPixelCount *
                                         static_cast<uint64_t>(formatTexelBytes(
                                             kFrameCompositionDepthFormat)) *
                                         static_cast<uint64_t>(sampleCount)
                                   : 0u;
  aaMetrics.msaaRingColorBytes =
      aaMetrics.msaaColorTextureBytes *
      static_cast<uint64_t>(aaMetrics.msaaColorTextureCount);
  aaMetrics.msaaRingDepthBytes =
      aaMetrics.msaaDepthTextureBytes *
      static_cast<uint64_t>(aaMetrics.msaaDepthTextureCount);
  aaMetrics.msaaTotalBytes =
      aaMetrics.msaaRingColorBytes + aaMetrics.msaaRingDepthBytes;
  aaMetrics.msaaResolveReadEstimateBytes =
      aaMetrics.msaaColorTextureBytes + aaMetrics.msaaDepthTextureBytes;
  aaMetrics.msaaResolveWriteEstimateBytes =
      aaMetrics.msaaEnabled
          ? msaaPixelCount *
                static_cast<uint64_t>(
                    formatTexelBytes(kFrameCompositionSceneColorFormat) +
                    formatTexelBytes(kFrameCompositionDepthFormat))
          : 0u;
  aaMetrics.msaaResolveBandwidthEstimateBytes =
      aaMetrics.msaaResolveReadEstimateBytes;
  aaMetrics.motionVectorFormat = kFrameCompositionMotionVectorFormat;
  aaMetrics.motionVectorWidth = framebufferWidth_;
  aaMetrics.motionVectorHeight = framebufferHeight_;
  aaMetrics.motionVectorTextureCount =
      static_cast<uint32_t>(ringSize(Ring::MotionVectors));
  aaMetrics.motionVectorAllocationCount = allocations(Ring::MotionVectors);
  aaMetrics.motionVectorReallocationCount = reallocations(Ring::MotionVectors);
  aaMetrics.motionVectorRg32FallbackCount = 0u;
  aaMetrics.motionVectorAllocated =
      nuri::isValid(ctx.shared[FrameTextureSlot::MotionVector].texture);
  aaMetrics.previousMotionVectorValid = nuri::isValid(
      ctx.shared[FrameHistoryTextureSlot::PreviousMotionVector].texture);
  aaMetrics.previousSceneDepthValid = nuri::isValid(
      ctx.shared[FrameHistoryTextureSlot::PreviousSceneDepth].texture);
  aaMetrics.motionVectorFormatSupported =
      kFrameCompositionMotionVectorFormat == Format::RG16_FLOAT;
  const uint64_t bytesPerTexture = static_cast<uint64_t>(framebufferWidth_) *
                                   static_cast<uint64_t>(framebufferHeight_) *
                                   static_cast<uint64_t>(formatTexelBytes(
                                       kFrameCompositionMotionVectorFormat));
  aaMetrics.motionVectorTextureBytes =
      aaMetrics.motionVectorAllocated ? bytesPerTexture : 0u;
  aaMetrics.previousMotionVectorTextureBytes =
      ringSize(Ring::MotionVectors) != 0u ? bytesPerTexture : 0u;
  aaMetrics.motionVectorTotalBytes =
      bytesPerTexture * static_cast<uint64_t>(ringSize(Ring::MotionVectors));
  aaMetrics.motionVectorClearBytes = 0u;
  aaMetrics.previousSceneDepthTextureBytes =
      aaMetrics.previousSceneDepthValid
          ? static_cast<uint64_t>(framebufferWidth_) *
                static_cast<uint64_t>(framebufferHeight_) *
                static_cast<uint64_t>(
                    formatTexelBytes(kFrameCompositionDepthFormat))
          : 0u;
  aaMetrics.reactiveMaskWidth = framebufferWidth_;
  aaMetrics.reactiveMaskHeight = framebufferHeight_;
  aaMetrics.reactiveMaskTextureCount =
      static_cast<uint32_t>(ringSize(Ring::ReactiveMask));
  aaMetrics.reactiveMaskAllocationCount = allocations(Ring::ReactiveMask);
  aaMetrics.reactiveMaskReallocationCount = reallocations(Ring::ReactiveMask);
  aaMetrics.reactiveMaskAllocated =
      nuri::isValid(ctx.shared[FrameTextureSlot::ReactiveMask].texture);
  aaMetrics.reactiveMaskFormatSupported =
      formatTexelBytes(kFrameCompositionReactiveMaskFormat) != 0u;
  const uint64_t reactiveBytesPerTexture =
      static_cast<uint64_t>(framebufferWidth_) *
      static_cast<uint64_t>(framebufferHeight_) *
      static_cast<uint64_t>(
          formatTexelBytes(kFrameCompositionReactiveMaskFormat));
  aaMetrics.reactiveMaskTextureBytes =
      aaMetrics.reactiveMaskAllocated ? reactiveBytesPerTexture : 0u;
  aaMetrics.reactiveMaskTotalBytes =
      reactiveBytesPerTexture *
      static_cast<uint64_t>(ringSize(Ring::ReactiveMask));
  aaMetrics.motionClassTextureCount =
      static_cast<uint32_t>(ringSize(Ring::MotionClass));
  const uint64_t motionClassBytesPerTexture =
      static_cast<uint64_t>(framebufferWidth_) *
      static_cast<uint64_t>(framebufferHeight_) *
      static_cast<uint64_t>(
          formatTexelBytes(kFrameCompositionMotionClassFormat));
  aaMetrics.motionClassTotalBytes =
      motionClassBytesPerTexture *
      static_cast<uint64_t>(ringSize(Ring::MotionClass));
  aaMetrics.historyColorTextureCount =
      static_cast<uint32_t>(ringSize(Ring::HistoryColor));
  const uint64_t historyColorBytesPerTexture =
      static_cast<uint64_t>(framebufferWidth_) *
      static_cast<uint64_t>(framebufferHeight_) *
      static_cast<uint64_t>(
          formatTexelBytes(kFrameCompositionSceneColorFormat));
  aaMetrics.historyColorTextureBytes =
      ringSize(Ring::HistoryColor) == 0u ? 0u : historyColorBytesPerTexture;
  aaMetrics.historyColorTotalBytes =
      historyColorBytesPerTexture *
      static_cast<uint64_t>(ringSize(Ring::HistoryColor));
  if (ctx.shared.sceneDepthSamplerId == 0u) {
    ctx.shared.sceneDepthSamplerId = gpu_.getDefaultSamplerBindlessIndex();
  }
  ctx.frame.sharedDepthTexture =
      ctx.shared[FrameTextureSlot::SceneDepth].texture;
  allocationCounts_.fill(0u);
  reallocationCounts_.fill(0u);
  return Result<bool, std::string>::makeResult(true);
}

void FrameCompositionProvider::onFrameSubmitted(
    const RenderFrameContext &frame) noexcept {
  if (!historyRegistry_.commitFrame(frame.frameIndex)) {
    return;
  }
  committedHistoryRequirements_ =
      pendingHistoryRequirements_ &
      frame.sharedResources.historyWriteRequirements;
  pendingHistoryRequirements_ = FrameTextureRequirementFlags::None;
}

void FrameCompositionProvider::onFrameAbandoned(
    const RenderFrameContext &frame) noexcept {
  historyRegistry_.abandonFrame(frame.frameIndex);
  pendingHistoryRequirements_ = FrameTextureRequirementFlags::None;
}

Result<bool, std::string> FrameCompositionProvider::ensureTextures(
    FrameTextureRequirementFlags requirements, CoverageMode coverage) {
  const uint32_t sampleCount = coverageSampleCount(coverage);
  const std::array<RingDesc, static_cast<size_t>(Ring::Count)> rings{{
      {Ring::SceneColor, FrameTextureRequirementFlags::SceneColor,
       kFrameCompositionSceneColorFormat, TextureUsage::AttachmentSampled,
       "frame_scene_color"},
      {Ring::SceneColorHalf, FrameTextureRequirementFlags::SceneColorMipChain,
       kFrameCompositionSceneColorFormat, TextureUsage::AttachmentSampled,
       "frame_scene_color_half", 1u},
      {Ring::SceneColorQuarter,
       FrameTextureRequirementFlags::SceneColorMipChain,
       kFrameCompositionSceneColorFormat, TextureUsage::AttachmentSampled,
       "frame_scene_color_quarter", 2u},
      {Ring::FrameColor, FrameTextureRequirementFlags::FrameColor,
       kFrameCompositionFrameColorFormat, TextureUsage::AttachmentSampled,
       "frame_output_color"},
      {Ring::SceneDepth, FrameTextureRequirementFlags::SceneDepth,
       kFrameCompositionDepthFormat, TextureUsage::AttachmentSampled,
       "frame_scene_depth"},
      {Ring::MsaaSceneColor, FrameTextureRequirementFlags::MsaaSceneColor,
       kFrameCompositionSceneColorFormat, TextureUsage::Attachment,
       "frame_msaa_scene_color", 0u, static_cast<uint8_t>(sampleCount)},
      {Ring::MsaaSceneDepth, FrameTextureRequirementFlags::MsaaSceneDepth,
       kFrameCompositionDepthFormat, TextureUsage::Attachment,
       "frame_msaa_scene_depth", 0u, static_cast<uint8_t>(sampleCount)},
      {Ring::MotionVectors, FrameTextureRequirementFlags::MotionVectors,
       kFrameCompositionMotionVectorFormat, TextureUsage::AttachmentSampled,
       "frame_motion_vectors"},
      {Ring::ReactiveMask, FrameTextureRequirementFlags::ReactiveMask,
       kFrameCompositionReactiveMaskFormat, TextureUsage::AttachmentSampled,
       "frame_reactive_mask"},
      {Ring::MotionClass, FrameTextureRequirementFlags::MotionClass,
       kFrameCompositionMotionClassFormat, TextureUsage::AttachmentSampled,
       "frame_motion_class"},
      {Ring::Normals, FrameTextureRequirementFlags::Normals,
       kFrameCompositionNormalFormat, TextureUsage::AttachmentSampled,
       "frame_material_normals"},
      {Ring::AmbientOcclusion, FrameTextureRequirementFlags::AmbientOcclusion,
       kFrameCompositionAmbientOcclusionFormat, TextureUsage::StorageSampled,
       "frame_ambient_occlusion"},
      {Ring::DDGIOpaqueSurfaceCache,
       FrameTextureRequirementFlags::DDGIOpaqueSurfaceCache,
       kFrameCompositionDDGIOpaqueSurfaceCacheFormat,
       TextureUsage::StorageSampled, "frame_ddgi_opaque_surface_cache"},
      {Ring::Exposure, FrameTextureRequirementFlags::Exposure,
       kFrameCompositionExposureFormat, TextureUsage::AttachmentSampled,
       "frame_exposure_ev", 0u, 1u, true},
      {Ring::PresentCapture, FrameTextureRequirementFlags::PresentCapture,
       Format::RGBA8_UNORM, TextureUsage::AttachmentSampled,
       "frame_present_capture"},
      {Ring::HistoryColor, FrameTextureRequirementFlags::HistoryColor,
       kFrameCompositionFrameColorFormat, TextureUsage::AttachmentSampled,
       "frame_history_color"},
  }};
  int32_t framebufferWidth = 0;
  int32_t framebufferHeight = 0;
  gpu_.getFramebufferSize(framebufferWidth, framebufferHeight);
  const uint32_t safeWidth =
      static_cast<uint32_t>(std::max(framebufferWidth, 1));
  const uint32_t safeHeight =
      static_cast<uint32_t>(std::max(framebufferHeight, 1));
  const uint32_t ringCount = std::max(2u, gpu_.getSwapchainImageCount());
  const bool dimensionsChanged =
      framebufferWidth_ != safeWidth || framebufferHeight_ != safeHeight;
  const bool ringChanged = textureRingCount_ != ringCount;
  const bool requirementsChanged = allocatedRequirements_ != requirements;
  const bool coverageChanged = allocatedCoverage_ != coverage;
  if (!dimensionsChanged && !ringChanged && !requirementsChanged &&
      !coverageChanged) {
    return Result<bool, std::string>::makeResult(true);
  }
  const FrameTextureRequirementFlags previousRequirements =
      allocatedRequirements_;
  const bool fullRecreate = dimensionsChanged || ringChanged;
  if (fullRecreate) {
    historyRegistry_.invalidate(HistoryInvalidationReason::ResourceRecreation);
    committedHistoryRequirements_ = FrameTextureRequirementFlags::None;
  }
  framebufferWidth_ = safeWidth;
  framebufferHeight_ = safeHeight;
  textureRingCount_ = ringCount;
  allocatedCoverage_ = coverage;
  allocatedRequirements_ = requirements;
  for (const RingDesc &ring : rings) {
    const bool needs =
        hasFrameTextureRequirementFlag(requirements, ring.requirement);
    const bool had =
        hasFrameTextureRequirementFlag(previousRequirements, ring.requirement);
    const bool ringCoverageChanged =
        coverageChanged && (ring.ring == Ring::MsaaSceneColor ||
                            ring.ring == Ring::MsaaSceneDepth);
    if (needs && (fullRecreate || !had || ringCoverageChanged)) {
      auto result = recreateTextureRing(ring);
      if (result.hasError()) {
        invalidateAllocationState();
        return result;
      }
    } else if (!needs && had) {
      destroyTextureRing(ring.ring);
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

void FrameCompositionProvider::invalidateAllocationState() noexcept {
  framebufferWidth_ = 0u;
  framebufferHeight_ = 0u;
  textureRingCount_ = 0u;
  allocatedCoverage_ = CoverageMode::Sample1;
  allocatedRequirements_ = FrameTextureRequirementFlags::None;
  for (size_t i = 0; i < textureRings_.size(); ++i) {
    destroyTextureRing(static_cast<Ring>(i));
  }
}

Result<bool, std::string>
FrameCompositionProvider::recreateTextureRing(const RingDesc &ring) {
  const size_t index = static_cast<size_t>(ring.ring);
  TextureRing &textures = textureRings_[index];
  const bool replacing = !textures.empty();
  destroyTextureRing(ring.ring);
  textures.resize(textureRingCount_);
  const uint32_t width =
      ring.fixedSize ? 1u : levelDimensions(framebufferWidth_, ring.mipLevel);
  const uint32_t height =
      ring.fixedSize ? 1u : levelDimensions(framebufferHeight_, ring.mipLevel);
  TextureDesc textureDesc =
      makeTextureDesc(ring.format, width, height, ring.usage);
  textureDesc.numSamples = ring.samples;
  for (uint32_t i = 0; i < textureRingCount_; ++i) {
    const std::string name = std::string(ring.name) + "_" + std::to_string(i);
    auto result = gpu_.createTexture(textureDesc, name);
    if (result.hasError()) {
      destroyTextureRing(ring.ring);
      return Result<bool, std::string>::makeError(result.error());
    }
    textures[i] = result.value();
  }
  reallocationCounts_[index] += replacing ? 1u : 0u;
  allocationCounts_[index] += textureRingCount_;
  return Result<bool, std::string>::makeResult(true);
}

void FrameCompositionProvider::destroyTextureRing(Ring ring) {
  TextureRing &textures = textureRings_[static_cast<size_t>(ring)];
  for (TextureHandle texture : textures) {
    if (nuri::isValid(texture)) {
      gpu_.destroyTexture(texture);
    }
  }
  textures.clear();
}

TextureHandle
FrameCompositionProvider::currentTexture(Ring ring,
                                         uint64_t frameIndex) const noexcept {
  const TextureRing &textures = textureRings_[static_cast<size_t>(ring)];
  if (textures.empty()) {
    return {};
  }
  return textures[static_cast<size_t>(frameIndex % textures.size())];
}

} // namespace nuri
