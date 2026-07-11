#include "nuri/pch.h"

#include "nuri/gfx/renderer.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/frame/temporal_frame_service.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/render_graph/render_graph_telemetry.h"
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

[[nodiscard]] uint64_t estimateCompletedFrameIndex(const GPUDevice &gpu,
                                                   uint64_t frameIndex) {
  const uint64_t lag = std::max<uint64_t>(1u, gpu.getSwapchainImageCount());
  if (frameIndex <= lag) {
    return 0u;
  }
  return frameIndex - lag;
}

} // namespace

Renderer::Renderer(GPUDevice &gpu, std::pmr::memory_resource &memory)
    : gpu_(gpu), resources_(gpu, &memory), renderGraphRuntime_(&memory),
      renderGraphBuilder_(&memory), renderGraphExecutor_(&memory),
      renderGraphTelemetry_(&memory),
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
  resetFrameSharedResources(frameContext);
  Result<bool, std::string> frameResult =
      beginFrameSequence(frameContext.frameIndex);
  if (frameResult.hasError()) {
    if (frameContext.temporalFrameService != nullptr) {
      frameContext.temporalFrameService->abandonFrame(frameContext.frameIndex);
    }
    return frameResult;
  }

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
      NURI_ASSERT(committed,
                  "Temporal frame commit did not match submitted frame");
    }
  } else if (submitResult.hasError()) {
    pipeline.onFrameAbandoned(frameContext);
  } else {
    pipeline.onFrameSubmitted(frameContext);
  }
  return submitResult;
}

Result<bool, std::string> Renderer::beginFrameSequence(uint64_t frameIndex) {
  {
    NURI_PROFILER_ZONE("Renderer.begin_frame", NURI_PROFILER_COLOR_CMD_COPY);
    resources_.beginFrame(frameIndex);
    NURI_PROFILER_ZONE_END();
  }

  Result<bool, std::string> frameResult =
      Result<bool, std::string>::makeResult(true);
  {
    NURI_PROFILER_ZONE("Renderer.gpu_begin_frame", NURI_PROFILER_COLOR_WAIT);
    frameResult = gpu_.beginFrame(frameIndex);
    NURI_PROFILER_ZONE_END();
  }
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
    resources_.collectGarbage(estimateCompletedFrameIndex(gpu_, frameIndex));
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

  const auto executeResult =
      [&]() -> Result<RenderGraphExecutionMetadata, std::string> {
    std::optional<Result<RenderGraphExecutionMetadata, std::string>> result;
    NURI_PROFILER_ZONE("Renderer.render_graph_execute",
                       NURI_PROFILER_COLOR_SUBMIT);
    result.emplace(renderGraphExecutor_.execute(renderGraphRuntime_, gpu_,
                                                *cachedCompileResult_));
    NURI_PROFILER_ZONE_END();
    return std::move(*result);
  }();
  if (executeResult.hasError()) {
    cachedCompileResult_.reset();
    return Result<bool, std::string>::makeError(executeResult.error());
  }

  {
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
