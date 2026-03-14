#include "tests_pch.h"

#include "render_graph_test_support.h"

#include <gtest/gtest.h>

#include "nuri/core/layer.h"
#include "nuri/core/layer_stack.h"
#include "nuri/gfx/renderer.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory_resource>
#include <span>

namespace {

using namespace nuri;
using namespace nuri::test_support;

std::filesystem::path makeTempRendererPath(std::string_view stem) {
  const auto tick =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() /
         ("nuri_" + std::string(stem) + "_" + std::to_string(tick));
}

class ExplicitFrameOutputLayer final : public Layer {
public:
  explicit ExplicitFrameOutputLayer(TextureHandle outputTexture)
      : outputTexture(outputTexture) {}

  Result<bool, std::string>
  buildRenderGraph(RenderFrameContext &, RenderGraphBuilder &graph) override {
    RenderPass pass{};
    pass.colorTexture = outputTexture;
    pass.debugLabel = "Layer Explicit Output Pass";

    auto addResult = addTestGraphicsPass(graph, pass, pass.debugLabel);
    if (addResult.hasError()) {
      return Result<bool, std::string>::makeError(addResult.error());
    }

    auto importResult = graph.importTexture(outputTexture, "layer_output_tex");
    if (importResult.hasError()) {
      return Result<bool, std::string>::makeError(importResult.error());
    }
    auto outputResult = graph.markTextureAsFrameOutput(importResult.value());
    if (outputResult.hasError()) {
      return Result<bool, std::string>::makeError(outputResult.error());
    }

    return Result<bool, std::string>::makeResult(true);
  }

private:
  TextureHandle outputTexture{};
};

class BaseImplicitOutputLayer final : public Layer {
public:
  Result<bool, std::string>
  buildRenderGraph(RenderFrameContext &, RenderGraphBuilder &graph) override {
    RenderGraphGraphicsPassDesc desc{};
    desc.debugLabel = "Base Implicit Output Pass";
    desc.debugColor = 0xff778899u;
    auto addResult = graph.addGraphicsPass(desc);
    if (addResult.hasError()) {
      return Result<bool, std::string>::makeError(addResult.error());
    }
    return Result<bool, std::string>::makeResult(true);
  }
};

TEST(RenderGraphRendererTest,
     RendererKeepsBaseImplicitPassUnderSuppressionWithExplicitOutputRoot) {
  EnvVarGuard envGuard("NURI_RENDER_GRAPH_SUPPRESS_INFERRED_SIDE_EFFECTS", "1");

  std::array<std::byte, 64 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeRendererGPUDevice gpu;
  Renderer renderer(gpu, memory);

  LayerStack layers(&memory);
  auto *baseLayer =
      layers.pushLayer(std::make_unique<BaseImplicitOutputLayer>());
  ASSERT_NE(baseLayer, nullptr) << "pushLayer for base pass should succeed";
  const TextureHandle explicitOutputTexture{.index = 501u, .generation = 1u};
  auto *layer = layers.pushLayer(
      std::make_unique<ExplicitFrameOutputLayer>(explicitOutputTexture));
  ASSERT_NE(layer, nullptr) << "pushLayer should succeed";

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 1u;

  auto renderResult = renderer.render(layers, frameContext);
  if (renderResult.hasError() || !renderResult.value()) {
    ADD_FAILURE() << "renderer render should succeed";
    if (renderResult.hasError()) {
      std::cerr << renderResult.error() << "\n";
    }
    return;
  }

  ASSERT_EQ(gpu.submitCount, 1u) << "renderer should submit one frame";
  ASSERT_EQ(gpu.submittedPassCount, 2u)
      << "renderer should keep base implicit pass and explicit output layer "
         "pass";
  ASSERT_TRUE(hasPassLabel(gpu, "Base Implicit Output Pass") &&
              hasPassLabel(gpu, "Layer Explicit Output Pass"))
      << "submitted frame should contain both base implicit and layer output "
         "passes";
}

TEST(RenderGraphRendererTest, RendererCapturesTelemetryWithoutAutomaticDump) {
  const std::filesystem::path dumpDirectory =
      makeTempRendererPath("renderer_telemetry_dir");
  EnvVarGuard envGuard("NURI_RENDER_GRAPH_DUMP",
                       dumpDirectory.generic_string());

  std::array<std::byte, 64 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeRendererGPUDevice gpu;
  Renderer renderer(gpu, memory);

  LayerStack layers(&memory);
  auto *baseLayer =
      layers.pushLayer(std::make_unique<BaseImplicitOutputLayer>());
  ASSERT_NE(baseLayer, nullptr);

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 7u;

  auto renderResult = renderer.render(layers, frameContext);
  ASSERT_FALSE(renderResult.hasError());
  ASSERT_TRUE(renderResult.value());

  const RenderGraphTelemetrySnapshot *snapshot =
      renderer.renderGraphTelemetry().latestSnapshot();
  ASSERT_NE(snapshot, nullptr);
  EXPECT_EQ(snapshot->summary.frameIndex, 7u);
  EXPECT_EQ(snapshot->summary.passCount, 1u);
  EXPECT_EQ(snapshot->summary.recordedCommandBufferCount, 1u);
  EXPECT_EQ(snapshot->summary.submitBatchCount, 1u);
  EXPECT_EQ(snapshot->summary.passRangeCount, 1u);
  EXPECT_FALSE(snapshot->summary.usedParallelRecording);
  ASSERT_EQ(snapshot->recordedCommandBuffers.size(), 1u);
  EXPECT_EQ(snapshot->recordedCommandBuffers[0].firstOrderedPassIndex, 0u);
  ASSERT_EQ(snapshot->submitBatches.size(), 1u);
  EXPECT_TRUE(snapshot->submitBatches[0].presentsFrameOutput);

  const std::filesystem::path suggested =
      renderer.renderGraphTelemetry().suggestDumpPath();
  EXPECT_EQ(suggested.parent_path(), dumpDirectory);
  EXPECT_EQ(suggested.filename(), "render_graph_frame_7.txt");
  EXPECT_FALSE(std::filesystem::exists(dumpDirectory))
      << "renderer should not create or write the dump directory per frame";
}

} // namespace
