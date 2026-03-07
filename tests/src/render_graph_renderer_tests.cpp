#include "nuri/pch.h"

#include "render_graph_test_support.h"

#include <gtest/gtest.h>

#include "nuri/core/layer.h"
#include "nuri/core/layer_stack.h"
#include "nuri/gfx/renderer.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <memory_resource>
#include <span>

namespace {

using namespace nuri;
using namespace nuri::test_support;

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

} // namespace
