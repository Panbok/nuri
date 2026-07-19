#include "nuri/gfx/renderer.h"
#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/frame/temporal_frame_service.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/render_graph/render_graph_telemetry.h"
#include "nuri/pch.h"
#include "nuri/utils/env_utils.h"
namespace nuri {

namespace {
[[nodiscard]] bool resolveSuppressInferredSideEffectsFlag() {
  const std::optional<std::string> env =
      readEnvVar("NURI_RENDER_GRAPH_SUPPRESS_INFERRED_SIDE_EFFECTS");
  if (!env.has_value()) {
    return false;
  }
  const std::string &value = *env;
  return value == "1" || value == "true" || value == "TRUE";
}
} // namespace

Renderer::Renderer(GPUDevice &gpu, std::pmr::memory_resource &memory)
    : gpu_(gpu), resources_(gpu, &memory), assets_(gpu, resources_),
      renderGraphRuntime_(&renderGraphMemory_),
      renderGraphBuilder_(&renderGraphMemory_),
      renderGraphExecutor_(&renderGraphMemory_),
      renderGraphTelemetry_(&renderGraphMemory_),
      suppressInferredSideEffects_(resolveSuppressInferredSideEffectsFlag()) {
  renderGraphBuilder_.setInferredSideEffectSuppression(
      suppressInferredSideEffects_);
  if (suppressInferredSideEffects_) {
    NURI_LOG_INFO(
        "Renderer: inferred render-graph side-effect suppression is "
        "enabled via NURI_RENDER_GRAPH_SUPPRESS_INFERRED_SIDE_EFFECTS");
  }
  NURI_LOG_DEBUG("Renderer::Renderer: Renderer created");
}

Result<bool, std::string> Renderer::render() {
  NURI_PROFILER_FUNCTION();
  const uint64_t frameIndex = standaloneFrameIndex_++;
  Result<bool, std::string> frameResult = beginFrameSequence(frameIndex);
  if (frameResult.hasError()) {
    return frameResult;
  }
  renderGraphBeginFrame(frameIndex);
  return endFrameSequence(frameIndex);
}

Result<bool, std::string> Renderer::render(RenderPipeline &pipeline,
                                           RenderFrameContext &frameContext) {
  NURI_PROFILER_FUNCTION();
  resolveRenderSettingsForFrame(frameContext);
  resetFrameSharedResources(frameContext);
  Result<bool, std::string> frameResult =
      beginFrameSequence(frameContext.frameIndex, frameContext.scene);
  if (frameResult.hasError()) {
    if (frameContext.temporalFrameService != nullptr) {
      frameContext.temporalFrameService->abandonFrame(frameContext.frameIndex);
    }
    return frameResult;
  }
  const AssetCpuSchedulerStats cpuStats = assets_.cpuStats();
  frameContext.metrics.assets = {
      .cpuCompletions = lastAssetPublicationStats_.cpuCompletions,
      .cpuWorkers = cpuStats.workerCount,
      .cpuActiveWorkerLimit = cpuStats.activeWorkerLimit,
      .cpuInteractiveMode = cpuStats.interactiveMode ? 1u : 0u,
      .cpuQueuedJobs = cpuStats.queuedJobs,
      .cpuRunningJobs = cpuStats.runningJobs,
      .cpuRunningIo =
          cpuStats.runningByClass[static_cast<size_t>(AssetWorkClass::Io)],
      .cpuRunningDecode =
          cpuStats.runningByClass[static_cast<size_t>(AssetWorkClass::Decode)],
      .cpuRunningCook =
          cpuStats.runningByClass[static_cast<size_t>(AssetWorkClass::Cook)],
      .cpuRunningTranscode =
          cpuStats
              .runningByClass[static_cast<size_t>(AssetWorkClass::Transcode)],
      .cpuRunningMetadata =
          cpuStats
              .runningByClass[static_cast<size_t>(AssetWorkClass::Metadata)],
      .dedicatedCopyQueue = gpu_.assetUploadsUseDedicatedCopyQueue() ? 1u : 0u,
      .gpuMaterialized = lastAssetPublicationStats_.gpuMaterialized,
      .published = lastAssetPublicationStats_.published,
      .cancelled = lastAssetPublicationStats_.cancelled,
      .failed = lastAssetPublicationStats_.failed,
      .scenePatches = lastAssetPublicationStats_.scenePatches,
      .sceneCommits = lastAssetPublicationStats_.sceneCommits,
      .deferredCpuCompletions =
          lastAssetPublicationStats_.deferredCpuCompletions,
      .publicationDeadlineExceeded =
          lastAssetPublicationStats_.deadlineExceeded ? 1u : 0u,
      .publicationMainThreadMilliseconds =
          lastAssetPublicationStats_.mainThreadMilliseconds,
      .publicationMaxOperationMilliseconds =
          lastAssetPublicationStats_.maxOperationMilliseconds,
      .cpuInFlightBytes = cpuStats.inFlightBytes,
      .uploadBytes = lastAssetPublicationStats_.uploadBytes,
      .submittedJobs = cpuStats.submittedJobs,
      .completedJobs = cpuStats.completedJobs,
      .cancelledJobs = cpuStats.cancelledJobs,
      .rejectedJobs = cpuStats.rejectedJobs,
  };
  frameContext.gpuTiming = gpu_.getLatestCompletedGpuTimingReport();
  renderGraphBeginFrame(frameContext.frameIndex);
  auto pipelineResult =
      pipeline.buildRenderGraph(frameContext, resources_, renderGraphBuilder_);
  if (pipelineResult.hasError()) {
    pipeline.onFrameAbandoned(frameContext);
    if (frameContext.temporalFrameService != nullptr) {
      frameContext.temporalFrameService->abandonFrame(frameContext.frameIndex);
    }
    return Result<bool, std::string>::makeError(pipelineResult.error());
  }
  Result<bool, std::string> submitResult =
      endFrameSequence(frameContext.frameIndex);
  if (frameContext.temporalFrameService != nullptr) {
    if (submitResult.hasError()) {
      pipeline.onFrameAbandoned(frameContext);
      frameContext.temporalFrameService->abandonFrame(frameContext.frameIndex);
    } else {
      pipeline.onFrameSubmitted(frameContext);
      const bool committed = frameContext.temporalFrameService->commitFrame(
          frameContext.frameIndex);
    }
  } else if (submitResult.hasError()) {
    pipeline.onFrameAbandoned(frameContext);
  } else {
    pipeline.onFrameSubmitted(frameContext);
  }
  return submitResult;
}

Result<bool, std::string> Renderer::beginFrameSequence(uint64_t frameIndex,
                                                       RenderScene *scene) {
  Result<bool, std::string> frameResult =
      Result<bool, std::string>::makeResult(true);
  {
    NURI_PROFILER_ZONE("Renderer.gpu_begin_frame", NURI_PROFILER_COLOR_WAIT);
    frameResult = gpu_.beginFrame(frameIndex);
    NURI_PROFILER_ZONE_END();
  }
  if (frameResult.hasError()) {
    return frameResult;
  }
  Result<AssetPublicationStats, std::string> assetResult =
      Result<AssetPublicationStats, std::string>::makeResult({});
  {
    NURI_PROFILER_ZONE("Renderer.begin_frame", NURI_PROFILER_COLOR_CMD_COPY);
    assetResult = assets_.prepareFrame(AssetPublicationContext{
        .scene = scene,
    });
    NURI_PROFILER_ZONE_END();
  }
  if (assetResult.hasError()) {
    return Result<bool, std::string>::makeError(assetResult.error());
  }
  lastAssetPublicationStats_ = assetResult.value();
  return frameResult;
}

void Renderer::renderGraphBeginFrame(uint64_t frameIndex) {
  NURI_PROFILER_ZONE("Renderer.render_graph_begin_frame",
                     NURI_PROFILER_COLOR_CMD_COPY);
  renderGraphBuilder_.beginFrame(frameIndex);
  NURI_PROFILER_ZONE_END();
}

Result<bool, std::string> Renderer::endFrameSequence(uint64_t frameIndex) {
  Result<bool, std::string> submitResult =
      Result<bool, std::string>::makeResult(true);
  {
    NURI_PROFILER_ZONE("Renderer.compile_execute", NURI_PROFILER_COLOR_SUBMIT);
    submitResult = compileAndExecuteRenderGraph(frameIndex);
    NURI_PROFILER_ZONE_END();
  }
  {
    NURI_PROFILER_ZONE("Renderer.resource_gc", NURI_PROFILER_COLOR_DESTROY);
    resources_.collectGarbage();
    NURI_PROFILER_ZONE_END();
  }
  return submitResult;
}

Result<bool, std::string>
Renderer::compileAndExecuteRenderGraph(uint64_t frameIndex) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_SUBMIT);
  const RenderGraphBuilder::GraphFingerprint fingerprint =
      renderGraphBuilder_.computeGraphFingerprint();
  if (cachedCompileResult_.has_value() && fingerprint == cachedFingerprint_) {
    NURI_PROFILER_ZONE("Renderer.render_graph_compile_cache_hit",
                       NURI_PROFILER_COLOR_BARRIER);
    renderGraphBuilder_.refreshHandlesInCompileResult(*cachedCompileResult_);
    cachedCompileResult_->frameIndex = frameIndex;
    NURI_PROFILER_ZONE_END();
  } else {
    std::string compileError;
    {
      NURI_PROFILER_ZONE("Renderer.render_graph_compile",
                         NURI_PROFILER_COLOR_BARRIER);
      auto compileResult = renderGraphBuilder_.compile(renderGraphRuntime_);
      if (compileResult.hasError()) {
        compileError = compileResult.error();
      } else {
        cachedCompileResult_ = std::move(compileResult.value());
        cachedFingerprint_ = fingerprint;
      }
      NURI_PROFILER_ZONE_END();
    }
    if (!compileError.empty()) {
      cachedCompileResult_.reset();
      return Result<bool, std::string>::makeError(std::move(compileError));
    }
  }
  const RenderGraphTelemetryLevel telemetryLevel =
      renderGraphTelemetry_.requestedCaptureLevel();
  const auto executeResult =
      [&]() -> Result<RenderGraphExecutionMetadata, std::string> {
    std::optional<Result<RenderGraphExecutionMetadata, std::string>> result;
    NURI_PROFILER_ZONE("Renderer.render_graph_execute",
                       NURI_PROFILER_COLOR_SUBMIT);
    result.emplace(renderGraphExecutor_.execute(
        renderGraphRuntime_, gpu_, *cachedCompileResult_,
        RenderGraphExecutionOptions{.telemetry = telemetryLevel}));
    NURI_PROFILER_ZONE_END();
    return std::move(*result);
  }();
  if (executeResult.hasError()) {
    cachedCompileResult_.reset();
    return Result<bool, std::string>::makeError(executeResult.error());
  }
  if (telemetryLevel != RenderGraphTelemetryLevel::None) {
    NURI_PROFILER_ZONE("Renderer.render_graph_telemetry",
                       NURI_PROFILER_COLOR_CMD_COPY);
    renderGraphTelemetry_.capture(*cachedCompileResult_, executeResult.value());
    NURI_PROFILER_ZONE_END();
  }
  return Result<bool, std::string>::makeResult(true);
}

void Renderer::invalidateCompileCache() { cachedCompileResult_.reset(); }

void Renderer::onResize(uint32_t width, uint32_t height) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  gpu_.resizeSwapchain(static_cast<int32_t>(width),
                       static_cast<int32_t>(height));
  invalidateCompileCache();
}

} // namespace nuri
