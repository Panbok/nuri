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
    : gpu_(gpu), resources_(gpu, &memory), renderGraphBuilder_(&memory),
      renderGraphExecutor_(&memory), renderGraphTelemetry_(&memory),
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
  resources_.beginFrame(frameIndex);
  auto frameResult = gpu_.beginFrame(frameIndex);
  if (frameResult.hasError()) {
    return frameResult;
  }
  renderGraphBuilder_.beginFrame(frameIndex);
  auto submitResult = compileAndExecuteRenderGraph();
  resources_.collectGarbage(frameIndex);
  return submitResult;
}

Result<bool, std::string> Renderer::render(LayerStack &layers,
                                           RenderFrameContext &frameContext) {
  NURI_PROFILER_FUNCTION();
  resources_.beginFrame(frameContext.frameIndex);
  auto frameResult = gpu_.beginFrame(frameContext.frameIndex);
  if (frameResult.hasError()) {
    return frameResult;
  }

  renderGraphBuilder_.beginFrame(frameContext.frameIndex);

  if (!layers.empty()) {
    auto layerResult =
        layers.buildRenderGraph(frameContext, renderGraphBuilder_);
    if (layerResult.hasError()) {
      return Result<bool, std::string>::makeError(layerResult.error());
    }
  }

  auto submitResult = compileAndExecuteRenderGraph();
  resources_.collectGarbage(frameContext.frameIndex);
  return submitResult;
}

Result<bool, std::string> Renderer::compileAndExecuteRenderGraph() {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_SUBMIT);
  auto compileResult = renderGraphBuilder_.compile();
  if (compileResult.hasError()) {
    return Result<bool, std::string>::makeError(compileResult.error());
  }
  renderGraphTelemetry_.capture(compileResult.value());

  auto executeResult =
      renderGraphExecutor_.execute(gpu_, compileResult.value());
  return executeResult;
}

void Renderer::onResize(uint32_t width, uint32_t height) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  gpu_.resizeSwapchain(static_cast<int32_t>(width),
                       static_cast<int32_t>(height));
}

} // namespace nuri
