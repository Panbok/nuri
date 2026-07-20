#include "tests_pch.h"

#include "nuri/bakery/smaa_lut_baker.h"
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
                                 std::string_view name) {
  auto texture = gpu.createTexture(
      TextureDesc{
          .type = TextureType::Texture2D,
          .format = Format::RGBA8_UNORM,
          .dimensions = {.width = 32u, .height = 32u, .depth = 1u},
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

} // namespace
