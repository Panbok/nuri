#include "tests_pch.h"

#include "nuri/bakery/smaa_lut_baker.h"
#include "nuri/gfx/frame/presentation_aa_plan.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/pipeline/features/spatial_aa_feature.h"
#include "nuri/gfx/smaa_lut_contract.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "render_graph_test_support.h"

#include <array>
#include <chrono>
#include <filesystem>
#include <string_view>

#include <gtest/gtest.h>

namespace {

using namespace nuri;
using nuri::test_support::FakeGPUDeviceBase;

PresentationAAPlan requirePlan(const RenderSettings &settings) {
  constexpr PresentationAAGpuCapabilities capabilities{
      .sample4Color = true,
      .sample4Depth = true,
      .sample8Color = true,
      .sample8Depth = true,
      .depthResolveMin = true,
      .alphaToCoverage = true,
  };
  auto result = buildPresentationAAPlan(settings, {}, capabilities);
  EXPECT_FALSE(result.hasError()) << (result.hasError() ? result.error() : "");
  return result.hasError() ? PresentationAAPlan{} : result.value();
}

class FakeSpatialAAGpuDevice final : public FakeGPUDeviceBase {
public:
  Result<ShaderHandle, std::string>
  createShaderModule(const ShaderDesc &) override {
    return Result<ShaderHandle, std::string>::makeResult(
        ShaderHandle{.index = nextShaderIndex_++, .generation = 1u});
  }

  Result<RenderPipelineHandle, std::string>
  createRenderPipeline(const RenderPipelineDesc &, std::string_view) override {
    return Result<RenderPipelineHandle, std::string>::makeResult(
        RenderPipelineHandle{.index = nextPipelineIndex_++, .generation = 1u});
  }

  [[nodiscard]] uint32_t pipelineCreateCount() const noexcept {
    return nextPipelineIndex_ - 1u;
  }

private:
  uint32_t nextShaderIndex_ = 1u;
  uint32_t nextPipelineIndex_ = 1u;
};

class ScopedTempDirectory final {
public:
  explicit ScopedTempDirectory(std::string_view stem) {
    const auto tick =
        std::chrono::high_resolution_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            (std::string(stem) + "_" + std::to_string(tick));
    std::filesystem::create_directories(path_);
  }

  ~ScopedTempDirectory() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }

  [[nodiscard]] const std::filesystem::path &path() const noexcept {
    return path_;
  }

private:
  std::filesystem::path path_{};
};

TextureHandle createFrameTexture(FakeSpatialAAGpuDevice &gpu,
                                 std::string_view name,
                                 uint32_t dimension = 32u) {
  auto texture = gpu.createTexture(
      TextureDesc{
          .type = TextureType::Texture2D,
          .format = Format::RGBA8_UNORM,
          .dimensions = {.width = dimension, .height = dimension, .depth = 1u},
          .usage = TextureUsage::AttachmentSampled,
          .storage = Storage::Device,
          .numLayers = 1u,
          .numSamples = 1u,
          .numMipLevels = 1u,
      },
      name);
  EXPECT_FALSE(texture.hasError()) << texture.error();
  return texture.hasError() ? TextureHandle{} : texture.value();
}

void publishSpatialAAAssets(const std::filesystem::path &destination) {
  const std::filesystem::path sourceShaders =
      std::filesystem::path(PROJECT_SOURCE_DIR) / "assets" / "shaders";
  constexpr std::array shaderFiles{
      "fullscreen_copy.vert",
      "spatial_aa_edge.frag",
      "spatial_aa_blend.frag",
      "spatial_aa_neighborhood.frag",
  };
  for (std::string_view filename : shaderFiles) {
    std::error_code ec;
    std::filesystem::copy_file(
        sourceShaders / filename, destination / filename,
        std::filesystem::copy_options::overwrite_existing, ec);
    ASSERT_FALSE(ec) << filename << ": " << ec.message();
  }
  const bakery::detail::SmaaLutBakePlan plan{
      .shouldBake = true,
      .areaOutputPath = destination / smaa_lut::kAreaFilename,
      .searchOutputPath = destination / smaa_lut::kSearchFilename,
  };
  auto bake = bakery::detail::bakeSmaaLutsToDisk(plan);
  ASSERT_FALSE(bake.hasError()) << bake.error();
}

TEST(SpatialAAFeatureTests, RetriesInitializationAfterBakeryPublishesLuts) {
  ScopedTempDirectory temp("nuri_spatial_aa_live_bake");
  const std::filesystem::path sourceShaders =
      std::filesystem::path(PROJECT_SOURCE_DIR) / "assets" / "shaders";
  constexpr std::array shaderFiles{
      "fullscreen_copy.vert",
      "spatial_aa_edge.frag",
      "spatial_aa_blend.frag",
      "spatial_aa_neighborhood.frag",
  };
  for (std::string_view filename : shaderFiles) {
    std::error_code ec;
    std::filesystem::copy_file(
        sourceShaders / filename, temp.path() / filename,
        std::filesystem::copy_options::overwrite_existing, ec);
    ASSERT_FALSE(ec) << filename << ": " << ec.message();
  }

  FakeSpatialAAGpuDevice gpu;
  ResourceManager resources(gpu);
  RenderGraphBuilder graph;
  RenderFrameContext frame{};
  RenderSettings settings{};
  settings.antiAliasing.mode = AntiAliasingMode::TAA;
  frame.settings = &settings;
  frame.camera.historyValid = false;
  frame.sharedResources.sceneColorTexture =
      createFrameTexture(gpu, "spatial_aa_test_scene_color");
  frame.sharedResources.frameColorTexture =
      createFrameTexture(gpu, "spatial_aa_test_frame_color");
  ASSERT_TRUE(isValid(frame.sharedResources.sceneColorTexture));
  ASSERT_TRUE(isValid(frame.sharedResources.frameColorTexture));
  FrameBuildContext ctx{
      .frame = frame,
      .graph = graph,
      .resources = resources,
      .shared = frame.sharedResources,
  };

  SpatialAAPass pass(gpu,
                     RuntimeCompositeConfig{.shaderBasePath = temp.path()});
  ASSERT_TRUE(pass.isEnabled(ctx));
  auto prepare = pass.prepare(ctx);
  ASSERT_TRUE(prepare.hasError());
  EXPECT_NE(prepare.error().find("Invalid SMAA LUT"), std::string::npos);
  const uint32_t pipelineCreateCount = gpu.pipelineCreateCount();

  const bakery::detail::SmaaLutBakePlan plan{
      .shouldBake = true,
      .areaOutputPath = temp.path() / smaa_lut::kAreaFilename,
      .searchOutputPath = temp.path() / smaa_lut::kSearchFilename,
  };
  auto bake = bakery::detail::bakeSmaaLutsToDisk(plan);
  ASSERT_FALSE(bake.hasError()) << bake.error();

  prepare = pass.prepare(ctx);
  ASSERT_FALSE(prepare.hasError()) << prepare.error();
  EXPECT_TRUE(prepare.value());
  EXPECT_EQ(gpu.pipelineCreateCount(), pipelineCreateCount);
}

TEST(SpatialAAFeatureTests,
     PostAAScratchLeasesRespectSubmissionAbandonAndCompletion) {
  ScopedTempDirectory temp("nuri_post_aa_lifecycle");
  publishSpatialAAAssets(temp.path());
  FakeSpatialAAGpuDevice gpu;
  gpu.swapchainImageCount = 2u;
  ResourceManager resources(gpu);
  SpatialAAPass pass(gpu, RuntimeCompositeConfig{.shaderBasePath = temp.path()},
                     SpatialAAPlacement::PostTransparent);
  RenderSettings settings{};
  settings.antiAliasing.mode = AntiAliasingMode::MSAA4x;
  settings.antiAliasing.postAA.enabled = true;
  settings.antiAliasing.postAA.specular =
      PostAASpecularAlgorithm::InheritCurrent;
  settings.antiAliasing.postAA.spatial = PostAASpatialAlgorithm::Smaa1x;
  const TextureHandle frameColor = createFrameTexture(gpu, "post_aa_color");

  for (uint64_t frameIndex = 0u; frameIndex < 2u; ++frameIndex) {
    ASSERT_FALSE(gpu.beginFrame(frameIndex).hasError());
    RenderGraphBuilder graph;
    RenderFrameContext frame{};
    frame.frameIndex = frameIndex;
    frame.settings = &settings;
    frame.presentationAA = requirePlan(settings);
    frame.sharedResources.frameColorTexture = frameColor;
    FrameBuildContext ctx{.frame = frame,
                          .graph = graph,
                          .resources = resources,
                          .shared = frame.sharedResources};
    ASSERT_TRUE(pass.isEnabled(ctx));
    auto prepare = pass.prepare(ctx);
    ASSERT_FALSE(prepare.hasError()) << prepare.error();
    ASSERT_TRUE(prepare.value());
    EXPECT_EQ(pass.lifecycleSnapshot().recordingLeaseCount, 1u);
    auto build = pass.build(ctx);
    ASSERT_FALSE(build.hasError()) << build.error();
    ASSERT_TRUE(build.value());
    EXPECT_TRUE(frame.metrics.antiAliasing.postAA.smaaPlanned);
    EXPECT_FALSE(frame.metrics.antiAliasing.postAA.smaaSubmitted);
    auto submission = gpu.submitRecordedGraphicsFrame(
        std::span<const RecordedCommandBufferHandle>{},
        std::span<const SubmitBatchMeta>{});
    ASSERT_FALSE(submission.hasError()) << submission.error();
    frame.submission = submission.value().submission;
    pass.onFrameSubmitted(frame);
    EXPECT_TRUE(frame.metrics.antiAliasing.postAA.smaaSubmitted);
    EXPECT_EQ(frame.metrics.antiAliasing.postAA.smaaSubmittedPassCount, 4u);
    const SpatialAALifecycleSnapshot snapshot = pass.lifecycleSnapshot();
    EXPECT_EQ(snapshot.recordingLeaseCount, 0u);
    EXPECT_EQ(snapshot.submittedScratchCount, frameIndex + 1u);
    EXPECT_EQ(snapshot.submittedPostAALedgerCount, frameIndex + 1u);
  }

  ASSERT_FALSE(gpu.beginFrame(2u).hasError());
  RenderGraphBuilder saturatedGraph;
  RenderFrameContext saturatedFrame{};
  saturatedFrame.frameIndex = 2u;
  saturatedFrame.settings = &settings;
  saturatedFrame.presentationAA = requirePlan(settings);
  saturatedFrame.sharedResources.frameColorTexture = frameColor;
  FrameBuildContext saturatedCtx{.frame = saturatedFrame,
                                 .graph = saturatedGraph,
                                 .resources = resources,
                                 .shared = saturatedFrame.sharedResources};
  auto saturated = pass.prepare(saturatedCtx);
  ASSERT_FALSE(saturated.hasError()) << saturated.error();
  EXPECT_FALSE(saturated.value());
  EXPECT_NE(
      static_cast<uint32_t>(
          saturatedFrame.metrics.antiAliasing.postAA.degradation) &
          static_cast<uint32_t>(PostAADegradation::SmaaScratchRingSaturated),
      0u);

  gpu.waitIdle();
  ASSERT_FALSE(gpu.beginFrame(3u).hasError());
  RenderGraphBuilder abandonedGraph;
  RenderFrameContext abandonedFrame{};
  abandonedFrame.frameIndex = 3u;
  abandonedFrame.settings = &settings;
  abandonedFrame.presentationAA = requirePlan(settings);
  abandonedFrame.sharedResources.frameColorTexture = frameColor;
  FrameBuildContext abandonedCtx{.frame = abandonedFrame,
                                 .graph = abandonedGraph,
                                 .resources = resources,
                                 .shared = abandonedFrame.sharedResources};
  auto prepared = pass.prepare(abandonedCtx);
  ASSERT_FALSE(prepared.hasError()) << prepared.error();
  ASSERT_TRUE(prepared.value());
  const SpatialAALifecycleSnapshot beforeAbandon = pass.lifecycleSnapshot();
  EXPECT_EQ(beforeAbandon.recordingLeaseCount, 1u);
  pass.onFrameAbandoned(abandonedFrame);
  const SpatialAALifecycleSnapshot abandoned = pass.lifecycleSnapshot();
  EXPECT_EQ(abandoned.recordingLeaseCount, 0u);
  EXPECT_EQ(abandoned.submittedScratchCount,
            beforeAbandon.submittedScratchCount);
  EXPECT_EQ(abandoned.submittedPostAALedgerCount, 0u);
}

TEST(SpatialAAFeatureTests,
     PostAAResizeRetiresInFlightScratchAndCompletionRequiresLedgerMatch) {
  ScopedTempDirectory temp("nuri_post_aa_retirement");
  publishSpatialAAAssets(temp.path());
  FakeSpatialAAGpuDevice gpu;
  gpu.swapchainImageCount = 2u;
  ResourceManager resources(gpu);
  SpatialAAPass pass(gpu, RuntimeCompositeConfig{.shaderBasePath = temp.path()},
                     SpatialAAPlacement::PostTransparent);
  RenderSettings settings{};
  settings.antiAliasing.mode = AntiAliasingMode::MSAA4x;
  settings.antiAliasing.postAA.enabled = true;
  settings.antiAliasing.postAA.specular =
      PostAASpecularAlgorithm::InheritCurrent;
  settings.antiAliasing.postAA.spatial = PostAASpatialAlgorithm::Smaa1x;

  ASSERT_FALSE(gpu.beginFrame(0u).hasError());
  RenderGraphBuilder firstGraph;
  RenderFrameContext first{};
  first.frameIndex = 0u;
  first.settings = &settings;
  first.presentationAA = requirePlan(settings);
  first.sharedResources.frameColorTexture =
      createFrameTexture(gpu, "post_aa_32", 32u);
  FrameBuildContext firstCtx{.frame = first,
                             .graph = firstGraph,
                             .resources = resources,
                             .shared = first.sharedResources};
  ASSERT_TRUE(pass.prepare(firstCtx).value());
  ASSERT_TRUE(pass.build(firstCtx).value());
  auto submission = gpu.submitRecordedGraphicsFrame(
      std::span<const RecordedCommandBufferHandle>{},
      std::span<const SubmitBatchMeta>{});
  ASSERT_FALSE(submission.hasError()) << submission.error();
  first.submission = submission.value().submission;
  pass.onFrameSubmitted(first);

  ASSERT_FALSE(gpu.beginFrame(1u).hasError());
  RenderGraphBuilder resizedGraph;
  RenderFrameContext resized{};
  resized.frameIndex = 1u;
  resized.settings = &settings;
  resized.presentationAA = requirePlan(settings);
  resized.sharedResources.frameColorTexture =
      createFrameTexture(gpu, "post_aa_64", 64u);
  resized.gpuTiming.availableScopeMask =
      gpuTimingScopeToBit(GpuTimingScope::SpatialAA);
  resized.gpuTiming.spatialAASourceFrameIndex = 999u;
  FrameBuildContext resizedCtx{.frame = resized,
                               .graph = resizedGraph,
                               .resources = resources,
                               .shared = resized.sharedResources};
  auto prepared = pass.prepare(resizedCtx);
  ASSERT_FALSE(prepared.hasError()) << prepared.error();
  ASSERT_TRUE(prepared.value());
  EXPECT_FALSE(resized.metrics.antiAliasing.postAA.smaaCompleted);
  EXPECT_EQ(pass.lifecycleSnapshot().retiredScratchGroupCount, 1u);
  pass.onFrameAbandoned(resized);

  ASSERT_FALSE(gpu.beginFrame(3u).hasError());
  RenderGraphBuilder completedGraph;
  RenderFrameContext completed{};
  completed.frameIndex = 3u;
  completed.settings = &settings;
  completed.presentationAA = requirePlan(settings);
  completed.sharedResources.frameColorTexture =
      resized.sharedResources.frameColorTexture;
  FrameBuildContext completedCtx{.frame = completed,
                                 .graph = completedGraph,
                                 .resources = resources,
                                 .shared = completed.sharedResources};
  prepared = pass.prepare(completedCtx);
  ASSERT_FALSE(prepared.hasError()) << prepared.error();
  ASSERT_TRUE(prepared.value());
  EXPECT_TRUE(completed.metrics.antiAliasing.postAA.smaaCompleted);
  EXPECT_EQ(completed.metrics.antiAliasing.postAA.smaaCompletedSourceFrameIndex,
            0u);
  EXPECT_EQ(pass.lifecycleSnapshot().retiredScratchGroupCount, 0u);
  pass.onFrameAbandoned(completed);
}

} // namespace
