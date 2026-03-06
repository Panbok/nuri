#include "nuri/pch.h"

#include "nuri/gfx/renderer.h"

#include "nuri/core/layer_stack.h"
#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/render_graph/render_graph_debug_dump.h"
#include "nuri/utils/env_utils.h"

namespace nuri {

namespace {

[[nodiscard]] std::filesystem::path resolveRenderGraphDumpDirectory() {
  const std::optional<std::string> dumpEnv =
      readEnvVar("NURI_RENDER_GRAPH_DUMP");
  if (!dumpEnv.has_value() || dumpEnv->front() == '0') {
    return {};
  }

  const std::string &value = *dumpEnv;
  if (value == "1" || value == "true" || value == "TRUE") {
    return std::filesystem::path("logs/render_graph");
  }
  return std::filesystem::path(value);
}

[[nodiscard]] bool resolveSuppressInferredSideEffectsFlag() {
  const std::optional<std::string> env =
      readEnvVar("NURI_RENDER_GRAPH_SUPPRESS_INFERRED_SIDE_EFFECTS");
  if (!env.has_value()) {
    return false;
  }

  const std::string &value = *env;
  return value == "1" || value == "true" || value == "TRUE";
}

[[nodiscard]] std::filesystem::path
makeRenderGraphDumpPath(uint64_t frameIndex) {
  const std::filesystem::path directory = resolveRenderGraphDumpDirectory();
  if (directory.empty()) {
    return {};
  }

  std::ostringstream fileName;
  fileName << "render_graph_frame_" << frameIndex << ".txt";
  return directory / fileName.str();
}

} // namespace

Renderer::Renderer(GPUDevice &gpu, std::pmr::memory_resource &memory)
    : gpu_(gpu), resources_(gpu, &memory), renderGraphBuilder_(&memory),
      renderGraphExecutor_(&memory),
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
  renderGraphBuilder_.setInferredSideEffectSuppression(
      suppressInferredSideEffects_);
  auto submitResult = compileAndExecuteRenderGraph(frameIndex);
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
  renderGraphBuilder_.setInferredSideEffectSuppression(
      suppressInferredSideEffects_);

  if (!layers.empty()) {
    auto layerResult =
        layers.buildRenderGraph(frameContext, renderGraphBuilder_);
    if (layerResult.hasError()) {
      return Result<bool, std::string>::makeError(layerResult.error());
    }
  }

  auto submitResult = compileAndExecuteRenderGraph(frameContext.frameIndex);
  resources_.collectGarbage(frameContext.frameIndex);
  return submitResult;
}

Result<bool, std::string>
Renderer::compileAndExecuteRenderGraph(uint64_t frameIndex) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_SUBMIT);
  auto compileResult = renderGraphBuilder_.compile();
  if (compileResult.hasError()) {
    return Result<bool, std::string>::makeError(compileResult.error());
  }

  const std::filesystem::path dumpPath = makeRenderGraphDumpPath(frameIndex);
  if (!dumpPath.empty()) {
    auto dumpResult =
        writeRenderGraphTextDump(compileResult.value(), dumpPath.string());
    if (dumpResult.hasError()) {
      NURI_LOG_WARNING(
          "Renderer::compileAndExecuteRenderGraph: failed to write graph dump "
          "'%s': %s",
          dumpPath.string().c_str(), dumpResult.error().c_str());
    }
  }

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
