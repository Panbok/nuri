#include "tests_pch.h"

#include "render_graph_test_support.h"

#include <gtest/gtest.h>

#include "nuri/gfx/pipeline/render_feature.h"
#include "nuri/gfx/pipeline/render_feature_pass.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
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

class SinglePassFeaturePass final : public RenderFeaturePass {
public:
  struct Config {
    std::string_view passName{};
    TextureHandle outputTexture{};
    bool useExplicitFrameOutput = false;
    uint32_t debugColor = 0xffffffffu;
  };

  explicit SinglePassFeaturePass(Config config) : config_(config) {}

  std::string_view name() const noexcept override { return config_.passName; }
  bool isEnabled(const FrameBuildContext &) const override { return true; }

  Result<bool, std::string> prepare(FrameBuildContext &) override {
    return Result<bool, std::string>::makeResult(true);
  }

  Result<bool, std::string> build(FrameBuildContext &ctx) override {
    if (config_.useExplicitFrameOutput) {
      auto importResult =
          ctx.graph.importTexture(config_.outputTexture, "layer_output_tex");
      if (importResult.hasError()) {
        return Result<bool, std::string>::makeError(importResult.error());
      }

      RenderGraphGraphicsPassDesc desc{};
      desc.colorTexture = importResult.value();
      desc.debugLabel = config_.passName;
      desc.debugColor = config_.debugColor;
      desc.markColorAsFrameOutput = true;
      auto addResult = ctx.graph.addGraphicsPass(desc);
      if (addResult.hasError()) {
        return Result<bool, std::string>::makeError(addResult.error());
      }
      return Result<bool, std::string>::makeResult(true);
    }

    RenderGraphGraphicsPassDesc desc{};
    desc.debugLabel = config_.passName;
    desc.debugColor = config_.debugColor;
    auto addResult = ctx.graph.addGraphicsPass(desc);
    if (addResult.hasError()) {
      return Result<bool, std::string>::makeError(addResult.error());
    }
    return Result<bool, std::string>::makeResult(true);
  }

private:
  Config config_{};
};

class SinglePassFeature final : public RenderFeature {
public:
  explicit SinglePassFeature(SinglePassFeaturePass::Config config)
      : pass_(config), passes_{&pass_} {}

  std::string_view name() const noexcept override {
    return "SinglePassFeature";
  }

  std::span<RenderFeaturePass *const> passes() noexcept override {
    return std::span<RenderFeaturePass *const>(passes_.data(), passes_.size());
  }

private:
  SinglePassFeaturePass pass_;
  std::array<RenderFeaturePass *, 1> passes_{};
};

class BaseImplicitOutputFeature final : public RenderFeature {
public:
  BaseImplicitOutputFeature()
      : pass_({.passName = "Base Implicit Output Pass",
               .outputTexture = {},
               .useExplicitFrameOutput = false,
               .debugColor = 0xff778899u}),
        passes_{&pass_} {}

  std::string_view name() const noexcept override {
    return "BaseImplicitOutputFeature";
  }

  std::span<RenderFeaturePass *const> passes() noexcept override {
    return std::span<RenderFeaturePass *const>(passes_.data(), passes_.size());
  }

private:
  SinglePassFeaturePass pass_;
  std::array<RenderFeaturePass *, 1> passes_{};
};

class ExplicitFrameOutputFeature final : public RenderFeature {
public:
  explicit ExplicitFrameOutputFeature(TextureHandle outputTexture)
      : pass_({.passName = "Layer Explicit Output Pass",
               .outputTexture = outputTexture,
               .useExplicitFrameOutput = true,
               .debugColor = 0xffffffffu}),
        passes_{&pass_} {}

  std::string_view name() const noexcept override {
    return "ExplicitFrameOutputFeature";
  }

  std::span<RenderFeaturePass *const> passes() noexcept override {
    return std::span<RenderFeaturePass *const>(passes_.data(), passes_.size());
  }

private:
  SinglePassFeaturePass pass_;
  std::array<RenderFeaturePass *, 1> passes_{};
};

TEST(RenderGraphRendererTest,
     RendererKeepsBaseImplicitPassUnderSuppressionWithExplicitOutputRoot) {
  EnvVarGuard envGuard("NURI_RENDER_GRAPH_SUPPRESS_INFERRED_SIDE_EFFECTS", "1");

  std::array<std::byte, 64 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeRendererGPUDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);

  auto *baseFeature =
      pipeline.addFeature(std::make_unique<BaseImplicitOutputFeature>());
  ASSERT_NE(baseFeature, nullptr) << "addFeature for base pass should succeed";
  const TextureHandle explicitOutputTexture{.index = 501u, .generation = 1u};
  auto *feature = pipeline.addFeature(
      std::make_unique<ExplicitFrameOutputFeature>(explicitOutputTexture));
  ASSERT_NE(feature, nullptr) << "addFeature should succeed";

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 1u;
  frameContext.resources = &renderer.resources();

  auto renderResult = renderer.render(pipeline, frameContext);
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
  RenderPipeline pipeline(&memory);

  auto *baseFeature =
      pipeline.addFeature(std::make_unique<BaseImplicitOutputFeature>());
  ASSERT_NE(baseFeature, nullptr);

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 7u;
  frameContext.resources = &renderer.resources();

  auto renderResult = renderer.render(pipeline, frameContext);
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
