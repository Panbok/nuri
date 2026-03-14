#include "nuri/pch.h"

#include "nuri/gfx/renderer.h"

#include "nuri/core/layer_stack.h"
#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
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
  if (frameResult.hasError()) {
    return frameResult;
  }
  {
    NURI_PROFILER_ZONE("Renderer.render_graph_begin_frame",
                       NURI_PROFILER_COLOR_CMD_COPY);
    renderGraphBuilder_.beginFrame(frameIndex);
    NURI_PROFILER_ZONE_END();
  }
  Result<bool, std::string> submitResult =
      Result<bool, std::string>::makeResult(true);
  {
    NURI_PROFILER_ZONE("Renderer.compile_execute", NURI_PROFILER_COLOR_SUBMIT);
    submitResult = compileAndExecuteRenderGraph();
    NURI_PROFILER_ZONE_END();
  }
  {
    NURI_PROFILER_ZONE("Renderer.resource_gc", NURI_PROFILER_COLOR_DESTROY);
    resources_.collectGarbage(frameIndex);
    NURI_PROFILER_ZONE_END();
  }
  return submitResult;
}

Result<bool, std::string> Renderer::render(LayerStack &layers,
                                           RenderFrameContext &frameContext) {
  NURI_PROFILER_FUNCTION();
  {
    NURI_PROFILER_ZONE("Renderer.begin_frame", NURI_PROFILER_COLOR_CMD_COPY);
    resources_.beginFrame(frameContext.frameIndex);
    NURI_PROFILER_ZONE_END();
  }
  Result<bool, std::string> frameResult =
      Result<bool, std::string>::makeResult(true);
  {
    NURI_PROFILER_ZONE("Renderer.gpu_begin_frame", NURI_PROFILER_COLOR_WAIT);
    frameResult = gpu_.beginFrame(frameContext.frameIndex);
    NURI_PROFILER_ZONE_END();
  }
  if (frameResult.hasError()) {
    return frameResult;
  }

  {
    NURI_PROFILER_ZONE("Renderer.render_graph_begin_frame",
                       NURI_PROFILER_COLOR_CMD_COPY);
    renderGraphBuilder_.beginFrame(frameContext.frameIndex);
    NURI_PROFILER_ZONE_END();
  }

  if (!layers.empty()) {
    Result<bool, std::string> layerResult =
        Result<bool, std::string>::makeResult(true);
    {
      NURI_PROFILER_ZONE("Renderer.layer_graph_build",
                         NURI_PROFILER_COLOR_CMD_DRAW);
      layerResult = layers.buildRenderGraph(frameContext, renderGraphBuilder_);
      NURI_PROFILER_ZONE_END();
    }
    if (layerResult.hasError()) {
      return Result<bool, std::string>::makeError(layerResult.error());
    }
  }

  Result<bool, std::string> submitResult =
      Result<bool, std::string>::makeResult(true);
  {
    NURI_PROFILER_ZONE("Renderer.compile_execute", NURI_PROFILER_COLOR_SUBMIT);
    submitResult = compileAndExecuteRenderGraph();
    NURI_PROFILER_ZONE_END();
  }
  {
    NURI_PROFILER_ZONE("Renderer.resource_gc", NURI_PROFILER_COLOR_DESTROY);
    resources_.collectGarbage(frameContext.frameIndex);
    NURI_PROFILER_ZONE_END();
  }
  return submitResult;
}

Result<bool, std::string> Renderer::compileAndExecuteRenderGraph() {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_SUBMIT);
  Result<RenderGraphCompileResult, std::string> compileResult =
      Result<RenderGraphCompileResult, std::string>::makeError(std::string{});
  {
    NURI_PROFILER_ZONE("Renderer.render_graph_compile",
                       NURI_PROFILER_COLOR_BARRIER);
    compileResult = renderGraphBuilder_.compile(renderGraphRuntime_);
    NURI_PROFILER_ZONE_END();
  }
  if (compileResult.hasError()) {
    return Result<bool, std::string>::makeError(compileResult.error());
  }

  Result<RenderGraphExecutionMetadata, std::string> executeResult =
      Result<RenderGraphExecutionMetadata, std::string>::makeError(
          std::string{});
  {
    NURI_PROFILER_ZONE("Renderer.render_graph_execute",
                       NURI_PROFILER_COLOR_SUBMIT);
    executeResult = renderGraphExecutor_.execute(renderGraphRuntime_, gpu_,
                                                 compileResult.value());
    NURI_PROFILER_ZONE_END();
  }
  if (executeResult.hasError()) {
    return Result<bool, std::string>::makeError(executeResult.error());
  }
  {
    NURI_PROFILER_ZONE("Renderer.render_graph_telemetry",
                       NURI_PROFILER_COLOR_CMD_COPY);
    renderGraphTelemetry_.capture(compileResult.value(), executeResult.value());
    NURI_PROFILER_ZONE_END();
  }
  return Result<bool, std::string>::makeResult(true);
}

void Renderer::onResize(uint32_t width, uint32_t height) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  gpu_.resizeSwapchain(static_cast<int32_t>(width),
                       static_cast<int32_t>(height));
}

} // namespace nuri
