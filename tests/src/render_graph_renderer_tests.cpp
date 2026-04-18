#include "tests_pch.h"

#include "render_graph_test_support.h"

#include <gtest/gtest.h>

#include "nuri/core/runtime_config.h"
#include "nuri/gfx/pipeline/features/composite_feature.h"
#include "nuri/gfx/pipeline/features/debug_feature.h"
#include "nuri/gfx/pipeline/features/opaque_feature.h"
#include "nuri/gfx/pipeline/features/shadow_feature.h"
#include "nuri/gfx/pipeline/providers/frame_composition_provider.h"
#include "nuri/gfx/pipeline/providers/material_table_gpu_provider.h"
#include "nuri/gfx/pipeline/providers/scene_lighting_provider.h"
#include "nuri/gfx/pipeline/render_feature.h"
#include "nuri/gfx/pipeline/render_feature_pass.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/gfx/renderer.h"
#include "nuri/gfx/renderers/detail/shadow_math.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/scene/light.h"
#include "nuri/scene/render_scene.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <memory_resource>
#include <span>

#include <lvk/LVK.h>
#include <vulkan/VulkanUtils.h>
#include <vulkan/vulkan_core.h>

namespace {

using namespace nuri;
using namespace nuri::test_support;

constexpr uint32_t kShadowPreviewProbeFlagInvert = 1u << 0u;
constexpr uint32_t kShadowPreviewProbeFlagLog = 1u << 1u;

struct ShadowPreviewPushConstantsProbe {
  uint32_t sourceTexId = 0u;
  float depthScale = 0.0f;
  float depthBias = 0.0f;
  uint32_t flags = 0u;
};

std::filesystem::path makeTempRendererPath(std::string_view stem) {
  const auto tick =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() /
         ("nuri_" + std::string(stem) + "_" + std::to_string(tick));
}

RuntimeCompositeConfig makeCompositeConfig(const std::filesystem::path &root) {
  const std::filesystem::path shaders = root / "assets" / "shaders";
  const std::filesystem::path textures = root / "assets" / "textures";
  return RuntimeCompositeConfig{
      .shaderBasePath = shaders,
      .fullscreenVertex = shaders / "fullscreen_copy.vert",
      .sceneCopyFragment = shaders / "scene_copy.frag",
      .presentFragment = shaders / "tonemap_present.frag",
      .aces2SdrLut = textures / "tonemap_aces2_sdr_64.ktx2",
      .agxLut = textures / "tonemap_agx_sdr_64.ktx2",
  };
}

RuntimeOpaqueShaderConfig makeOpaqueConfig(const std::filesystem::path &root) {
  const std::filesystem::path shaders = root / "assets" / "shaders";
  return RuntimeOpaqueShaderConfig{
      .shaderBasePath = shaders,
      .meshVertex = shaders / "main.vert",
      .meshFragment = shaders / "main.frag",
      .pickFragment = shaders / "main_id.frag",
      .shadowInspectFragment = shaders / "shadow_inspect.frag",
      .computeInstances = shaders / "duck_instances.comp",
      .tessVertex = shaders / "main.vert",
      .tessControl = shaders / "main.tesc",
      .tessEval = shaders / "main.tese",
      .overlayGeometry = shaders / "mesh_debug_overlay.geom",
      .overlayFragment = shaders / "mesh_debug_overlay.frag",
  };
}

class FakeFullscreenGpuDevice : public FakeGPUDeviceBase {
public:
  Result<ShaderHandle, std::string>
  createShaderModule(const ShaderDesc &) override {
    return Result<ShaderHandle, std::string>::makeResult(
        ShaderHandle{.index = nextShaderIndex_++, .generation = 1u});
  }

  Result<RenderPipelineHandle, std::string>
  createRenderPipeline(const RenderPipelineDesc &desc,
                       std::string_view debugName) override {
    createdRenderPipelineDescs.push_back(desc);
    createdRenderPipelineNames.emplace_back(debugName);
    return Result<RenderPipelineHandle, std::string>::makeResult(
        RenderPipelineHandle{.index = nextPipelineIndex_++, .generation = 1u});
  }

  Result<ComputePipelineHandle, std::string>
  createComputePipeline(const ComputePipelineDesc &,
                        std::string_view) override {
    return Result<ComputePipelineHandle, std::string>::makeResult(
        ComputePipelineHandle{.index = nextComputePipelineIndex_++,
                              .generation = 1u});
  }

  Result<SubmissionHandle, std::string> submitRecordedGraphicsFrame(
      std::span<const RecordedCommandBufferHandle> commandBuffers,
      std::span<const SubmitBatchMeta> batches) override {
    auto baseResult =
        FakeGPUDeviceBase::submitRecordedGraphicsFrame(commandBuffers, batches);
    if (baseResult.hasError()) {
      return baseResult;
    }
    submittedPassCount = recordedPasses.size();
    submittedPassLabels.clear();
    submittedPassLabels.reserve(recordedPasses.size());
    for (const RenderPass &pass : recordedPasses) {
      submittedPassLabels.emplace_back(pass.debugLabel);
    }
    return baseResult;
  }

  size_t submittedPassCount = 0u;
  std::vector<std::string> submittedPassLabels{};
  std::vector<RenderPipelineDesc> createdRenderPipelineDescs{};
  std::vector<std::string> createdRenderPipelineNames{};

private:
  uint32_t nextShaderIndex_ = 1u;
  uint32_t nextPipelineIndex_ = 1u;
  uint32_t nextComputePipelineIndex_ = 1u;
};

class FakeShadowSceneGpuDevice final : public FakeFullscreenGpuDevice {
public:
  Result<GeometryAllocationHandle, std::string>
  allocateGeometry(std::span<const std::byte> vertexBytes, uint32_t vertexCount,
                   std::span<const std::byte> indexBytes, uint32_t indexCount,
                   std::string_view) override {
    auto vertexBufferResult = createBufferImpl(BufferDesc{
        .usage = BufferUsage::Vertex | BufferUsage::Storage,
        .storage = Storage::Device,
        .size = vertexBytes.size(),
        .data = vertexBytes,
    });
    if (vertexBufferResult.hasError()) {
      return Result<GeometryAllocationHandle, std::string>::makeError(
          vertexBufferResult.error());
    }
    auto indexBufferResult = createBufferImpl(BufferDesc{
        .usage = BufferUsage::Index,
        .storage = Storage::Device,
        .size = indexBytes.size(),
        .data = indexBytes,
    });
    if (indexBufferResult.hasError()) {
      destroyBuffer(vertexBufferResult.value());
      return Result<GeometryAllocationHandle, std::string>::makeError(
          indexBufferResult.error());
    }

    const GeometryAllocationHandle handle{
        .index = static_cast<uint32_t>(allocations_.size() + 1u),
        .generation = 1u,
    };
    const size_t resolvedIndexByteSize =
        forcedIndexStrideBytes != 0u
            ? static_cast<size_t>(indexCount) * forcedIndexStrideBytes
            : indexBytes.size();
    allocations_.push_back(Allocation{
        .handle = handle,
        .view =
            GeometryAllocationView{
                .vertexBuffer = vertexBufferResult.value(),
                .vertexByteOffset = 0u,
                .vertexByteSize = vertexBytes.size(),
                .indexBuffer = indexBufferResult.value(),
                .indexByteOffset = 0u,
                .indexByteSize = resolvedIndexByteSize,
                .vertexCount = vertexCount,
                .indexCount = indexCount,
            },
    });
    return Result<GeometryAllocationHandle, std::string>::makeResult(handle);
  }

  void releaseGeometry(GeometryAllocationHandle handle) override {
    for (Allocation &allocation : allocations_) {
      if (allocation.handle.index != handle.index ||
          allocation.handle.generation != handle.generation) {
        continue;
      }
      if (nuri::isValid(allocation.view.vertexBuffer)) {
        destroyBuffer(allocation.view.vertexBuffer);
        allocation.view.vertexBuffer = {};
      }
      if (nuri::isValid(allocation.view.indexBuffer)) {
        destroyBuffer(allocation.view.indexBuffer);
        allocation.view.indexBuffer = {};
      }
      allocation.handle = {};
      return;
    }
  }

  bool resolveGeometry(GeometryAllocationHandle handle,
                       GeometryAllocationView &out) const override {
    for (const Allocation &allocation : allocations_) {
      if (allocation.handle.index == handle.index &&
          allocation.handle.generation == handle.generation) {
        out = allocation.view;
        return true;
      }
    }
    return false;
  }

  uint32_t forcedIndexStrideBytes = 0u;

private:
  struct Allocation {
    GeometryAllocationHandle handle{};
    GeometryAllocationView view{};
  };

  std::vector<Allocation> allocations_{};
};

struct PresentPushConstantsProbe {
  uint32_t sourceTexId = 0u;
  uint32_t sourceSamplerId = 0u;
  uint32_t acesLutTexId = 0u;
  uint32_t agxLutTexId = 0u;
  uint32_t lutSamplerId = 0u;
  uint32_t flags = 0u;
  float acesExposureScale = 1.0f;
  float agxExposureScale = 1.0f;
  float compareSplit = 0.5f;
  float shaperMinLog2 = 0.0f;
  float shaperInvRange = 0.0f;
};
static_assert(sizeof(PresentPushConstantsProbe) <= 128u);

constexpr uint32_t kPresentFlagManualSrgbEncode = 1u << 0u;
constexpr uint32_t kPresentFlagPrimaryUseAgx = 1u << 1u;
constexpr uint32_t kPresentFlagCompareEnabled = 1u << 2u;
constexpr uint32_t kPresentFlagGrayCardDebug = 1u << 3u;
constexpr uint32_t kPresentFlagAcesLutAvailable = 1u << 4u;
constexpr uint32_t kPresentFlagAgxLutAvailable = 1u << 5u;
constexpr uint32_t kPresentFlagPrimaryLegacyFallback = 1u << 6u;
constexpr uint32_t kPresentFlagCompareLegacyFallback = 1u << 7u;

const RenderPass *findRecordedPass(const FakeGPUDeviceBase &gpu,
                                   std::string_view label) {
  for (const RenderPass &pass : gpu.recordedPasses) {
    if (pass.debugLabel == label) {
      return &pass;
    }
  }
  return nullptr;
}

PresentPushConstantsProbe
readPresentPushConstants(const FakeGPUDeviceBase &gpu) {
  const RenderPass *pass = findRecordedPass(gpu, "Present ToneMap Pass");
  EXPECT_NE(pass, nullptr);
  if (pass == nullptr) {
    return {};
  }
  EXPECT_FALSE(pass->draws.empty());
  if (pass->draws.empty()) {
    return {};
  }
  const DrawItem &draw = pass->draws.front();
  EXPECT_EQ(draw.pushConstants.size(), sizeof(PresentPushConstantsProbe));
  PresentPushConstantsProbe probe{};
  if (draw.pushConstants.size() == sizeof(PresentPushConstantsProbe)) {
    std::memcpy(&probe, draw.pushConstants.data(), sizeof(probe));
  }
  return probe;
}

void writeTextFile(const std::filesystem::path &path, std::string_view text) {
  std::ofstream out(path, std::ios::binary);
  ASSERT_TRUE(out.is_open());
  out.write(text.data(), static_cast<std::streamsize>(text.size()));
  ASSERT_TRUE(out.good());
}

void writeTinyPngFile(const std::filesystem::path &path) {
  static constexpr std::array<unsigned char, 68> kPngBytes = {
      0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D,
      0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
      0x08, 0x04, 0x00, 0x00, 0x00, 0xB5, 0x1C, 0x0C, 0x02, 0x00, 0x00, 0x00,
      0x0B, 0x49, 0x44, 0x41, 0x54, 0x78, 0xDA, 0x63, 0xFC, 0xFF, 0x1F, 0x00,
      0x03, 0x03, 0x01, 0xFF, 0xA5, 0xF7, 0x45, 0xEA, 0x00, 0x00, 0x00, 0x00,
      0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
  };
  std::ofstream out(path, std::ios::binary);
  ASSERT_TRUE(out.is_open());
  out.write(reinterpret_cast<const char *>(kPngBytes.data()),
            static_cast<std::streamsize>(kPngBytes.size()));
  ASSERT_TRUE(out.good());
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

class TerminalPassFeature final : public RenderFeature {
public:
  explicit TerminalPassFeature(std::string_view passName)
      : pass_({.passName = passName,
               .outputTexture = {},
               .useExplicitFrameOutput = false,
               .debugColor = 0xff55aa55u}),
        passes_{&pass_} {}

  std::string_view name() const noexcept override {
    return "TerminalPassFeature";
  }

  bool isTerminalFeature() const noexcept override { return true; }

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

TEST(RenderGraphRendererTest, RenderSettingsDefaultToAces2SdrToneMapping) {
  const RenderSettings settings{};
  EXPECT_EQ(settings.toneMap.operator_, ToneMapper::ACES2_SDR);
  EXPECT_FLOAT_EQ(settings.toneMap.exposureEv, 0.0f);
  EXPECT_FLOAT_EQ(settings.toneMap.acesExposureOffsetEv, 0.35f);
  EXPECT_FLOAT_EQ(settings.toneMap.agxExposureOffsetEv, -0.35f);
  EXPECT_FALSE(settings.toneMap.grayCardDebug);
  EXPECT_FALSE(settings.toneMap.sideBySideCompare);
  EXPECT_FLOAT_EQ(settings.toneMap.compareSplit, 0.5f);
}

TEST(RenderGraphRendererTest, RenderSettingsDefaultToShadowsEnabled) {
  const RenderSettings settings{};
  EXPECT_TRUE(settings.shadow.enabled);
  EXPECT_EQ(settings.shadow.cascadeCount, kMaxShadowCascades);
  EXPECT_EQ(settings.shadow.filterMode, ShadowFilterMode::Hard);
  EXPECT_EQ(settings.shadow.sdsmMode, ShadowSdsmMode::Disabled);
}

TEST(RenderGraphRendererTest, ShadowSettingsSanitizeClampsCoreValues) {
  RenderSettings::ShadowSettings settings{};
  settings.cascadeCount = 0u;
  settings.shadowMapSize = 0u;
  settings.maxDistance = -10.0f;
  settings.splitLambda = 2.0f;
  settings.cascadeBlendFraction = -1.0f;
  settings.pcfSampleCount = 0u;
  settings.pcssBlockerSampleCount = 0u;
  settings.pcssFilterSampleCount = 0u;
  settings.debug.debugCascadeIndex = 99u;
  settings.debug.previewDepthMin = -1.0f;
  settings.debug.previewDepthMax = 2.0f;

  sanitizeShadowSettings(settings);

  EXPECT_EQ(settings.cascadeCount, 1u);
  EXPECT_EQ(settings.shadowMapSize, 1u);
  EXPECT_FLOAT_EQ(settings.maxDistance, 150.0f);
  EXPECT_FLOAT_EQ(settings.splitLambda, 1.0f);
  EXPECT_FLOAT_EQ(settings.cascadeBlendFraction, 0.0f);
  EXPECT_EQ(settings.pcfSampleCount, 1u);
  EXPECT_EQ(settings.pcssBlockerSampleCount, 1u);
  EXPECT_EQ(settings.pcssFilterSampleCount, 1u);
  EXPECT_EQ(settings.debug.debugCascadeIndex, 0u);
  EXPECT_FLOAT_EQ(settings.debug.previewDepthMin, 0.0f);
  EXPECT_FLOAT_EQ(settings.debug.previewDepthMax, 1.0f);

  settings.cascadeCount = kMaxShadowCascades + 8u;
  settings.splitLambda = -2.0f;
  settings.cascadeBlendFraction = 2.0f;
  settings.pcfSampleCount = kMaxShadowPcfSamples + 1u;
  settings.debug.previewDepthMin = 0.8f;
  settings.debug.previewDepthMax = 0.2f;
  sanitizeShadowSettings(settings);
  EXPECT_EQ(settings.cascadeCount, kMaxShadowCascades);
  EXPECT_FLOAT_EQ(settings.splitLambda, 0.0f);
  EXPECT_FLOAT_EQ(settings.cascadeBlendFraction, 1.0f);
  EXPECT_EQ(settings.pcfSampleCount, kMaxShadowPcfSamples);
  EXPECT_LT(settings.debug.previewDepthMin, settings.debug.previewDepthMax);
  EXPECT_NEAR(settings.debug.previewDepthMax - settings.debug.previewDepthMin,
              1.0e-4f, 1.0e-6f);
}

TEST(RenderGraphRendererTest, LvkD16DepthFormatMapsToVulkanD16Unorm) {
  EXPECT_EQ(lvk::formatToVkFormat(lvk::Format_Z_UN16), VK_FORMAT_D16_UNORM);
}

TEST(RenderGraphRendererTest,
     MakeCameraFrameStateCopiesPerspectiveProjectionMetadata) {
  Camera camera{};
  PerspectiveParams params{};
  params.fovYRadians = glm::radians(45.0f);
  params.nearPlane = 0.25f;
  params.farPlane = 250.0f;
  camera.setProjectionType(ProjectionType::Perspective);
  camera.setPerspective(params);
  camera.setLookAt(glm::vec3(1.0f, 2.0f, 3.0f), glm::vec3(1.0f, 2.0f, 2.0f),
                   glm::vec3(0.0f, 1.0f, 0.0f));

  const CameraFrameState state = makeCameraFrameState(camera, 16.0f / 9.0f);

  EXPECT_EQ(state.projectionType, ProjectionType::Perspective);
  EXPECT_FLOAT_EQ(state.aspectRatio, 16.0f / 9.0f);
  EXPECT_FLOAT_EQ(state.nearPlane, params.nearPlane);
  EXPECT_FLOAT_EQ(state.farPlane, params.farPlane);
  EXPECT_FLOAT_EQ(state.fovYRadians, params.fovYRadians);
  EXPECT_EQ(state.cameraPos, glm::vec4(camera.position(), 1.0f));
}

TEST(RenderGraphRendererTest,
     MakeCameraFrameStateCopiesOrthographicProjectionMetadata) {
  Camera camera{};
  OrthographicParams params{};
  params.height = 24.0f;
  params.nearPlane = 0.5f;
  params.farPlane = 500.0f;
  camera.setProjectionType(ProjectionType::Orthographic);
  camera.setOrthographic(params);

  const CameraFrameState state = makeCameraFrameState(camera, 4.0f / 3.0f);

  EXPECT_EQ(state.projectionType, ProjectionType::Orthographic);
  EXPECT_FLOAT_EQ(state.aspectRatio, 4.0f / 3.0f);
  EXPECT_FLOAT_EQ(state.nearPlane, params.nearPlane);
  EXPECT_FLOAT_EQ(state.farPlane, params.farPlane);
  EXPECT_FLOAT_EQ(state.orthoHeight, params.height);
}

TEST(RenderGraphRendererTest, DirectionalShadowBoundsFitIsCameraInvariant) {
  CameraFrameState cameraA{};
  cameraA.view =
      glm::lookAt(glm::vec3(0.0f, 2.0f, 6.0f), glm::vec3(0.0f, 0.5f, 0.0f),
                  glm::vec3(0.0f, 1.0f, 0.0f));
  cameraA.cameraPos = glm::vec4(0.0f, 2.0f, 6.0f, 1.0f);
  cameraA.aspectRatio = 1.0f;
  cameraA.projectionType = ProjectionType::Perspective;
  cameraA.nearPlane = 0.1f;
  cameraA.farPlane = 100.0f;
  cameraA.fovYRadians = glm::radians(60.0f);

  CameraFrameState cameraB = cameraA;
  cameraB.view =
      glm::lookAt(glm::vec3(8.0f, 4.0f, -7.0f), glm::vec3(2.0f, 1.0f, 0.0f),
                  glm::vec3(0.0f, 1.0f, 0.0f));
  cameraB.cameraPos = glm::vec4(8.0f, 4.0f, -7.0f, 1.0f);

  const glm::vec3 boundsMin(-3.0f, -0.25f, -2.0f);
  const glm::vec3 boundsMax(4.0f, 2.0f, 5.0f);
  const glm::vec3 lightDirection =
      glm::normalize(glm::vec3(-0.35f, -1.0f, -0.2f));
  const shadow_detail::DirectionalShadowFit fitA =
      shadow_detail::fitDirectionalShadowMapToBounds(
          cameraA, boundsMin, boundsMax, lightDirection, 50.0f, 1024u, true);
  const shadow_detail::DirectionalShadowFit fitB =
      shadow_detail::fitDirectionalShadowMapToBounds(
          cameraB, boundsMin, boundsMax, lightDirection, 50.0f, 1024u, true);

  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < 4; ++r) {
      EXPECT_NEAR(fitA.lightViewProj[c][r], fitB.lightViewProj[c][r], 1.0e-6f);
    }
  }
  EXPECT_GT(fitA.texelWorldSize, 0.0f);
}

TEST(RenderGraphRendererTest, DirectionalShadowBoundsFitUsesZeroToOneDepth) {
  CameraFrameState camera{};
  camera.nearPlane = 0.1f;
  camera.farPlane = 100.0f;

  const glm::vec3 boundsMin(-3.0f, -0.25f, -2.0f);
  const glm::vec3 boundsMax(4.0f, 2.0f, 5.0f);
  const glm::vec3 lightDirection =
      glm::normalize(glm::vec3(-0.35f, -1.0f, -0.2f));
  const shadow_detail::DirectionalShadowFit fit =
      shadow_detail::fitDirectionalShadowMapToBounds(
          camera, boundsMin, boundsMax, lightDirection, 50.0f, 1024u, true);

  for (const glm::vec3 corner : fit.frustumCorners) {
    const glm::vec4 clip = fit.lightViewProj * glm::vec4(corner, 1.0f);
    ASSERT_GT(std::abs(clip.w), 1.0e-6f);
    const glm::vec3 ndc = glm::vec3(clip) / clip.w;
    EXPECT_GE(ndc.x, -1.0f - 1.0e-4f);
    EXPECT_LE(ndc.x, 1.0f + 1.0e-4f);
    EXPECT_GE(ndc.y, -1.0f - 1.0e-4f);
    EXPECT_LE(ndc.y, 1.0f + 1.0e-4f);
    EXPECT_GE(ndc.z, -1.0e-4f);
    EXPECT_LE(ndc.z, 1.0f + 1.0e-4f);
  }
}

TEST(RenderGraphRendererTest,
     DirectionalShadowFitStabilizationKeepsSubTexelMotionSnapped) {
  CameraFrameState camera{};
  const glm::vec3 eye(0.0f, 2.0f, 6.0f);
  const glm::vec3 target(0.0f, 0.5f, 0.0f);
  const glm::vec3 up(0.0f, 1.0f, 0.0f);
  camera.view = glm::lookAt(eye, target, up);
  camera.cameraPos = glm::vec4(eye, 1.0f);
  camera.aspectRatio = 1.0f;
  camera.projectionType = ProjectionType::Perspective;
  camera.nearPlane = 0.1f;
  camera.farPlane = 100.0f;
  camera.fovYRadians = glm::radians(60.0f);

  const glm::vec3 lightDirection =
      glm::normalize(glm::vec3(-0.35f, -1.0f, -0.2f));
  const shadow_detail::DirectionalShadowFit baseFit =
      shadow_detail::fitSingleDirectionalShadowMap(camera, lightDirection,
                                                   50.0f, 1024u, true);

  const glm::vec3 subTexelMotion =
      glm::vec3(glm::inverse(baseFit.lightView) *
                glm::vec4(baseFit.texelWorldSize * 0.25f, 0.0f, 0.0f, 0.0f));
  CameraFrameState movedCamera = camera;
  movedCamera.view =
      glm::lookAt(eye + subTexelMotion, target + subTexelMotion, up);
  movedCamera.cameraPos = glm::vec4(eye + subTexelMotion, 1.0f);

  const shadow_detail::DirectionalShadowFit movedFit =
      shadow_detail::fitSingleDirectionalShadowMap(movedCamera, lightDirection,
                                                   50.0f, 1024u, true);

  EXPECT_GT(glm::length(movedFit.unsnappedCenter - baseFit.unsnappedCenter),
            1.0e-5f);
  EXPECT_NEAR(baseFit.snappedLightSpaceCenter.x,
              movedFit.snappedLightSpaceCenter.x, 1.0e-6f);
  EXPECT_NEAR(baseFit.snappedLightSpaceCenter.y,
              movedFit.snappedLightSpaceCenter.y, 1.0e-6f);
}

TEST(RenderGraphRendererTest,
     DirectionalShadowFitStabilizationAdvancesAfterTexelThreshold) {
  CameraFrameState camera{};
  const glm::vec3 eye(0.0f, 2.0f, 6.0f);
  const glm::vec3 target(0.0f, 0.5f, 0.0f);
  const glm::vec3 up(0.0f, 1.0f, 0.0f);
  camera.view = glm::lookAt(eye, target, up);
  camera.cameraPos = glm::vec4(eye, 1.0f);
  camera.aspectRatio = 1.0f;
  camera.projectionType = ProjectionType::Perspective;
  camera.nearPlane = 0.1f;
  camera.farPlane = 100.0f;
  camera.fovYRadians = glm::radians(60.0f);

  const glm::vec3 lightDirection =
      glm::normalize(glm::vec3(-0.35f, -1.0f, -0.2f));
  const shadow_detail::DirectionalShadowFit baseFit =
      shadow_detail::fitSingleDirectionalShadowMap(camera, lightDirection,
                                                   50.0f, 1024u, true);

  const glm::vec3 aboveTexelMotion =
      glm::vec3(glm::inverse(baseFit.lightView) *
                glm::vec4(baseFit.texelWorldSize * 1.25f, 0.0f, 0.0f, 0.0f));
  CameraFrameState movedCamera = camera;
  movedCamera.view =
      glm::lookAt(eye + aboveTexelMotion, target + aboveTexelMotion, up);
  movedCamera.cameraPos = glm::vec4(eye + aboveTexelMotion, 1.0f);

  const shadow_detail::DirectionalShadowFit movedFit =
      shadow_detail::fitSingleDirectionalShadowMap(movedCamera, lightDirection,
                                                   50.0f, 1024u, true);

  EXPECT_GT(std::abs(movedFit.snappedLightSpaceCenter.x -
                     baseFit.snappedLightSpaceCenter.x),
            baseFit.texelWorldSize * 0.5f);
}

TEST(RenderGraphRendererTest, RuntimeConfigResolvesCompositeDefaults) {
  const std::filesystem::path sourceRoot =
      std::filesystem::path(PROJECT_SOURCE_DIR);
  const std::filesystem::path tempDir = makeTempRendererPath("runtime_config");
  ASSERT_TRUE(std::filesystem::create_directories(tempDir));
  const std::filesystem::path configPath = tempDir / "app.config.json";

  const std::string configText = std::format(
      R"({{
  "window": {{
    "title": "Runtime Config Test",
    "width": 1280,
    "height": 720,
    "mode": "windowed"
  }},
  "roots": {{
    "assets": "{}",
    "shaders": "{}",
    "models": "{}",
    "textures": "{}",
    "fonts": "{}"
  }}
}})",
      (sourceRoot / "assets").generic_string(),
      (sourceRoot / "assets" / "shaders").generic_string(),
      (sourceRoot / "assets" / "models").generic_string(),
      (sourceRoot / "assets" / "textures").generic_string(),
      (sourceRoot / "assets" / "fonts").generic_string());
  writeTextFile(configPath, configText);

  auto configResult = loadRuntimeConfig(configPath);
  ASSERT_FALSE(configResult.hasError()) << configResult.error();
  const RuntimeShaderConfig &shaders = configResult.value().shaders;
  EXPECT_EQ(shaders.composite.fullscreenVertex,
            sourceRoot / "assets" / "shaders" / "fullscreen_copy.vert");
  EXPECT_EQ(shaders.composite.sceneCopyFragment,
            sourceRoot / "assets" / "shaders" / "scene_copy.frag");
  EXPECT_EQ(shaders.composite.presentFragment,
            sourceRoot / "assets" / "shaders" / "tonemap_present.frag");
  EXPECT_EQ(shaders.composite.aces2SdrLut,
            sourceRoot / "assets" / "textures" / "tonemap_aces2_sdr_64.ktx2");
  EXPECT_EQ(shaders.composite.agxLut,
            sourceRoot / "assets" / "textures" / "tonemap_agx_sdr_64.ktx2");
}

TEST(RenderGraphRendererTest, RuntimeConfigParsesExplicitCompositeOverrides) {
  const std::filesystem::path sourceRoot =
      std::filesystem::path(PROJECT_SOURCE_DIR);
  const std::filesystem::path tempDir =
      makeTempRendererPath("runtime_config_explicit");
  ASSERT_TRUE(std::filesystem::create_directories(tempDir));
  const std::filesystem::path configPath = tempDir / "app.config.json";

  const std::string configText = std::format(
      R"({{
  "window": {{
    "title": "Runtime Config Override Test",
    "width": 1280,
    "height": 720,
    "mode": "windowed"
  }},
  "roots": {{
    "assets": "{}",
    "shaders": "{}",
    "models": "{}",
    "textures": "{}",
    "fonts": "{}"
  }},
  "shaders": {{
    "composite": {{
      "fullscreen_vertex": "fullscreen_copy.vert",
      "scene_copy_fragment": "scene_copy.frag",
      "present_fragment": "tonemap_present.frag",
      "aces2_sdr_lut": "tonemap_aces2_sdr_64.ktx2",
      "agx_lut": "tonemap_agx_sdr_64.ktx2"
    }}
  }}
}})",
      (sourceRoot / "assets").generic_string(),
      (sourceRoot / "assets" / "shaders").generic_string(),
      (sourceRoot / "assets" / "models").generic_string(),
      (sourceRoot / "assets" / "textures").generic_string(),
      (sourceRoot / "assets" / "fonts").generic_string());
  writeTextFile(configPath, configText);

  auto configResult = loadRuntimeConfig(configPath);
  ASSERT_FALSE(configResult.hasError()) << configResult.error();
  const RuntimeCompositeConfig &composite =
      configResult.value().shaders.composite;
  EXPECT_EQ(composite.fullscreenVertex,
            sourceRoot / "assets" / "shaders" / "fullscreen_copy.vert");
  EXPECT_EQ(composite.aces2SdrLut,
            sourceRoot / "assets" / "textures" / "tonemap_aces2_sdr_64.ktx2");
  EXPECT_EQ(composite.agxLut,
            sourceRoot / "assets" / "textures" / "tonemap_agx_sdr_64.ktx2");
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

TEST(RenderGraphRendererTest,
     FrameCompositionProviderPublishesOffscreenResourcesWithoutTransmission) {
  std::array<std::byte, 32 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  FrameCompositionProvider provider(gpu, &memory);
  RenderGraphBuilder graph(&memory);
  RenderFrameContext frameContext{};
  FrameBuildContext ctx{
      .frame = frameContext,
      .graph = graph,
      .resources = renderer.resources(),
      .shared = frameContext.sharedResources,
  };

  frameContext.frameIndex = 0u;
  auto prepareResult = provider.prepare(ctx);
  ASSERT_FALSE(prepareResult.hasError());
  ASSERT_TRUE(prepareResult.value());
  EXPECT_TRUE(isValid(frameContext.sharedResources.sceneColorTexture));
  EXPECT_TRUE(isValid(frameContext.sharedResources.sceneColorHalfResTexture));
  EXPECT_TRUE(
      isValid(frameContext.sharedResources.sceneColorQuarterResTexture));
  EXPECT_TRUE(isValid(frameContext.sharedResources.frameColorTexture));
  EXPECT_TRUE(isValid(frameContext.sharedResources.sceneDepthTexture));
  EXPECT_TRUE(isValid(frameContext.sharedResources.historyColorReadTexture));
  EXPECT_TRUE(isValid(frameContext.sharedResources.historyColorWriteTexture));
}

TEST(RenderGraphRendererTest,
     FramePresentFeaturePushesToneMapModeAndSwapchainEncodingFlags) {
  std::array<std::byte, 128 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  auto *provider = pipeline.addProvider(
      std::make_unique<FrameCompositionProvider>(gpu, &memory));
  ASSERT_NE(provider, nullptr);
  auto *compositionFeature =
      pipeline.addFeature(std::make_unique<FrameCompositionFeature>(
          gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));
  ASSERT_NE(compositionFeature, nullptr);
  auto *presentFeature =
      pipeline.addFeature(std::make_unique<FramePresentFeature>(
          gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));
  ASSERT_NE(presentFeature, nullptr);

  RenderSettings settings{};
  settings.toneMap.operator_ = ToneMapper::AgX;
  settings.toneMap.exposureEv = 1.5f;
  settings.toneMap.acesExposureOffsetEv = -0.25f;
  settings.toneMap.agxExposureOffsetEv = 0.50f;
  settings.toneMap.grayCardDebug = true;
  settings.toneMap.sideBySideCompare = true;
  settings.toneMap.compareSplit = 0.35f;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 3u;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;

  auto renderResult = renderer.render(pipeline, frameContext);
  ASSERT_FALSE(renderResult.hasError()) << renderResult.error();
  ASSERT_TRUE(renderResult.value());

  const PresentPushConstantsProbe unormProbe = readPresentPushConstants(gpu);
  EXPECT_NE(unormProbe.acesLutTexId, 0u);
  EXPECT_NE(unormProbe.agxLutTexId, 0u);
  EXPECT_NE(unormProbe.lutSamplerId, 0u);
  EXPECT_NE(unormProbe.flags & kPresentFlagPrimaryUseAgx, 0u);
  EXPECT_NE(unormProbe.flags & kPresentFlagCompareEnabled, 0u);
  EXPECT_NE(unormProbe.flags & kPresentFlagGrayCardDebug, 0u);
  EXPECT_NE(unormProbe.flags & kPresentFlagAcesLutAvailable, 0u);
  EXPECT_NE(unormProbe.flags & kPresentFlagAgxLutAvailable, 0u);
  EXPECT_NE(unormProbe.flags & kPresentFlagManualSrgbEncode, 0u);
  EXPECT_EQ(unormProbe.flags & kPresentFlagPrimaryLegacyFallback, 0u);
  EXPECT_EQ(unormProbe.flags & kPresentFlagCompareLegacyFallback, 0u);
  EXPECT_NEAR(unormProbe.acesExposureScale, std::exp2(1.25f), 1.0e-6f);
  EXPECT_NEAR(unormProbe.agxExposureScale, std::exp2(2.0f), 1.0e-6f);
  EXPECT_NEAR(unormProbe.compareSplit, 0.35f, 1.0e-6f);
  EXPECT_FLOAT_EQ(unormProbe.shaperMinLog2, -10.0f);
  EXPECT_NEAR(unormProbe.shaperInvRange, 1.0f / 16.5f, 1.0e-6f);

  gpu.recordedPasses.clear();
  gpu.submittedPassLabels.clear();
  gpu.submittedPassCount = 0u;
  gpu.submitCount = 0u;
  gpu.swapchainFormat = Format::RGBA8_SRGB;
  settings.toneMap.operator_ = ToneMapper::ACES2_SDR;
  frameContext.frameIndex = 4u;

  renderResult = renderer.render(pipeline, frameContext);
  ASSERT_FALSE(renderResult.hasError()) << renderResult.error();
  ASSERT_TRUE(renderResult.value());

  const PresentPushConstantsProbe srgbProbe = readPresentPushConstants(gpu);
  EXPECT_NE(srgbProbe.flags & kPresentFlagAcesLutAvailable, 0u);
  EXPECT_NE(srgbProbe.flags & kPresentFlagAgxLutAvailable, 0u);
  EXPECT_EQ(srgbProbe.flags & kPresentFlagPrimaryUseAgx, 0u);
  EXPECT_EQ(srgbProbe.flags & kPresentFlagManualSrgbEncode, 0u);
  EXPECT_NE(srgbProbe.flags & kPresentFlagCompareEnabled, 0u);
  EXPECT_NE(srgbProbe.flags & kPresentFlagGrayCardDebug, 0u);
  EXPECT_EQ(srgbProbe.flags & kPresentFlagPrimaryLegacyFallback, 0u);
  EXPECT_EQ(srgbProbe.flags & kPresentFlagCompareLegacyFallback, 0u);
}

TEST(RenderGraphRendererTest,
     FramePresentFeatureFallsBackWhenToneMapLutIsMissing) {
  std::array<std::byte, 128 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  auto *provider = pipeline.addProvider(
      std::make_unique<FrameCompositionProvider>(gpu, &memory));
  ASSERT_NE(provider, nullptr);

  RuntimeCompositeConfig missingConfig =
      makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR));
  missingConfig.aces2SdrLut =
      std::filesystem::temp_directory_path() / "missing_aces2_sdr_lut.ktx2";
  auto *presentFeature = pipeline.addFeature(
      std::make_unique<FramePresentFeature>(gpu, missingConfig));
  ASSERT_NE(presentFeature, nullptr);

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 5u;
  frameContext.resources = &renderer.resources();

  auto renderResult = renderer.render(pipeline, frameContext);
  ASSERT_FALSE(renderResult.hasError()) << renderResult.error();
  ASSERT_TRUE(renderResult.value());

  const PresentPushConstantsProbe probe = readPresentPushConstants(gpu);
  EXPECT_EQ(probe.flags & kPresentFlagAcesLutAvailable, 0u);
  EXPECT_NE(probe.flags & kPresentFlagPrimaryLegacyFallback, 0u);
}

TEST(RenderGraphRendererTest,
     FrameCompositionProviderRecreatesTexturesWhenFramebufferChanges) {
  std::array<std::byte, 32 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeRendererGPUDevice gpu;
  Renderer renderer(gpu, memory);
  FrameCompositionProvider provider(gpu, &memory);
  RenderGraphBuilder graph(&memory);
  RenderFrameContext frameContext{};
  FrameBuildContext ctx{
      .frame = frameContext,
      .graph = graph,
      .resources = renderer.resources(),
      .shared = frameContext.sharedResources,
  };

  frameContext.frameIndex = 0u;
  auto prepareResult = provider.prepare(ctx);
  ASSERT_FALSE(prepareResult.hasError());
  const TextureHandle originalSceneColor =
      frameContext.sharedResources.sceneColorTexture;
  const TextureHandle originalFrameColor =
      frameContext.sharedResources.frameColorTexture;

  gpu.resizeSwapchain(1920, 1080);
  frameContext.frameIndex = 1u;
  prepareResult = provider.prepare(ctx);
  ASSERT_FALSE(prepareResult.hasError());
  EXPECT_FALSE(sameTexture(originalSceneColor,
                           frameContext.sharedResources.sceneColorTexture));
  EXPECT_FALSE(sameTexture(originalFrameColor,
                           frameContext.sharedResources.frameColorTexture));
}

TEST(RenderGraphRendererTest,
     TerminalFeaturePassesStayLastWhenLaterFeaturesAreRegistered) {
  std::array<std::byte, 8 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  RenderPipeline pipeline(&memory);

  auto *base = pipeline.addFeature(std::make_unique<SinglePassFeature>(
      SinglePassFeaturePass::Config{.passName = "Base Pass",
                                    .outputTexture = {},
                                    .useExplicitFrameOutput = false,
                                    .debugColor = 0xff111111u}));
  ASSERT_NE(base, nullptr);
  auto *terminal = pipeline.addFeature(
      std::make_unique<TerminalPassFeature>("Present Pass"));
  ASSERT_NE(terminal, nullptr);
  auto *late = pipeline.addFeature(std::make_unique<SinglePassFeature>(
      SinglePassFeaturePass::Config{.passName = "Late Pass",
                                    .outputTexture = {},
                                    .useExplicitFrameOutput = false,
                                    .debugColor = 0xff222222u}));
  ASSERT_NE(late, nullptr);

  ASSERT_EQ(pipeline.passCount(), 3u);
  const auto first = pipeline.passInfo(0u);
  const auto second = pipeline.passInfo(1u);
  const auto third = pipeline.passInfo(2u);
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());
  ASSERT_TRUE(third.has_value());
  EXPECT_EQ(first->passName, "Base Pass");
  EXPECT_EQ(second->passName, "Late Pass");
  EXPECT_EQ(third->passName, "Present Pass");
}

TEST(RenderGraphRendererTest, ShadowFeatureRegistersDepthPassPlaceholder) {
  std::array<std::byte, 8 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeRendererGPUDevice gpu;
  RenderPipeline pipeline(&memory);

  auto *shadow =
      pipeline.addFeature(std::make_unique<ShadowFeature>(gpu, &memory));

  ASSERT_NE(shadow, nullptr);
  ASSERT_EQ(pipeline.passCount(), 1u);
  const auto pass = pipeline.passInfo(0u);
  ASSERT_TRUE(pass.has_value());
  EXPECT_EQ(pass->featureName, "ShadowFeature");
  EXPECT_EQ(pass->passName, "ShadowDepthPass");
}

TEST(RenderGraphRendererTest, ShadowFeatureBuildsNoGraphPassesWhenDisabled) {
  std::array<std::byte, 32 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeRendererGPUDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  auto *shadow =
      pipeline.addFeature(std::make_unique<ShadowFeature>(gpu, &memory));
  ASSERT_NE(shadow, nullptr);

  RenderSettings settings{};
  settings.shadow.enabled = false;
  RenderFrameContext frameContext{};
  frameContext.frameIndex = 1u;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);

  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);

  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  EXPECT_TRUE(buildResult.value());
  EXPECT_EQ(graph.passCount(), 0u);
  ASSERT_EQ(gpu.createdTextureDescs.size(), 1u);
  ASSERT_EQ(gpu.createdSamplerDescs.size(), 2u);
  EXPECT_FALSE(gpu.createdSamplerDescs[0].depthCompareEnabled);
  EXPECT_TRUE(gpu.createdSamplerDescs[1].depthCompareEnabled);
  EXPECT_TRUE(frameContext.sharedResources.shadowFrameGpuData.has_value());
  EXPECT_EQ(frameContext.sharedResources.shadowRawSamplerId, 1u);
  EXPECT_EQ(frameContext.sharedResources.shadowCompareSamplerId, 2u);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureAllocatesD16DepthOnlyShadowResource) {
  std::array<std::byte, 32 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeRendererGPUDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  auto *shadow =
      pipeline.addFeature(std::make_unique<ShadowFeature>(gpu, &memory));
  ASSERT_NE(shadow, nullptr);

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.shadowMapSize = 1024u;
  RenderFrameContext frameContext{};
  frameContext.frameIndex = 1u;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);

  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);

  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  EXPECT_TRUE(buildResult.value());
  ASSERT_EQ(gpu.createdTextureDescs.size(), 1u);
  const TextureDesc &shadowDesc = gpu.createdTextureDescs.front();
  EXPECT_EQ(shadowDesc.type, TextureType::Texture2D);
  EXPECT_EQ(shadowDesc.format, kDefaultShadowMapDepthFormat);
  EXPECT_EQ(shadowDesc.format, Format::D16_UNORM);
  EXPECT_EQ(shadowDesc.dimensions.width, 1024u);
  EXPECT_EQ(shadowDesc.dimensions.height, 1024u);
  EXPECT_EQ(shadowDesc.dimensions.depth, 1u);
  EXPECT_EQ(shadowDesc.usage, TextureUsage::AttachmentSampled);
  EXPECT_EQ(shadowDesc.storage, Storage::Device);
  EXPECT_EQ(shadowDesc.numLayers, 1u);
  EXPECT_EQ(shadowDesc.numSamples, 1u);
  EXPECT_EQ(shadowDesc.numMipLevels, 1u);

  ASSERT_EQ(gpu.createdSamplerDescs.size(), 2u);
  const SamplerDesc &rawSamplerDesc = gpu.createdSamplerDescs.front();
  EXPECT_EQ(rawSamplerDesc.minFilter, SamplerFilter::Nearest);
  EXPECT_EQ(rawSamplerDesc.magFilter, SamplerFilter::Nearest);
  EXPECT_EQ(rawSamplerDesc.mipMode, SamplerMipMode::Disabled);
  EXPECT_EQ(rawSamplerDesc.wrapU, SamplerWrapMode::Clamp);
  EXPECT_EQ(rawSamplerDesc.wrapV, SamplerWrapMode::Clamp);
  EXPECT_EQ(rawSamplerDesc.wrapW, SamplerWrapMode::Clamp);
  EXPECT_FALSE(rawSamplerDesc.depthCompareEnabled);
  const SamplerDesc &compareSamplerDesc = gpu.createdSamplerDescs[1];
  EXPECT_EQ(compareSamplerDesc.minFilter, SamplerFilter::Nearest);
  EXPECT_EQ(compareSamplerDesc.magFilter, SamplerFilter::Nearest);
  EXPECT_EQ(compareSamplerDesc.mipMode, SamplerMipMode::Disabled);
  EXPECT_EQ(compareSamplerDesc.wrapU, SamplerWrapMode::Clamp);
  EXPECT_EQ(compareSamplerDesc.wrapV, SamplerWrapMode::Clamp);
  EXPECT_EQ(compareSamplerDesc.wrapW, SamplerWrapMode::Clamp);
  EXPECT_TRUE(compareSamplerDesc.depthCompareEnabled);
  EXPECT_EQ(compareSamplerDesc.depthCompareOp, CompareOp::LessEqual);

  EXPECT_TRUE(
      nuri::isValid(frameContext.sharedResources.shadowCascadeTextures[0]));
  EXPECT_EQ(frameContext.sharedResources.shadowRawSamplerId, 1u);
  EXPECT_EQ(frameContext.sharedResources.shadowCompareSamplerId, 2u);
  ASSERT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());
  EXPECT_EQ(frameContext.sharedResources.shadowDebugFrameData->cascadeCount,
            1u);
  EXPECT_EQ(frameContext.sharedResources.shadowDebugFrameData->rawSamplerId,
            1u);
  EXPECT_EQ(frameContext.sharedResources.shadowDebugFrameData->compareSamplerId,
            2u);
  EXPECT_EQ(graph.passCount(), 1u);

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const RenderGraphCompileResult &compiled = compileResult.value();
  ASSERT_EQ(compiled.orderedPasses.size(), 1u);
  const RenderPass &shadowPass = compiled.orderedPasses.front();
  EXPECT_FALSE(shadowPass.hasColorAttachment);
  EXPECT_TRUE(
      sameTexture(shadowPass.depthTexture,
                  frameContext.sharedResources.shadowCascadeTextures[0]));
  EXPECT_EQ(shadowPass.depth.loadOp, LoadOp::Clear);
  EXPECT_EQ(shadowPass.depth.storeOp, StoreOp::Store);
  EXPECT_EQ(shadowPass.depth.clearStencil, 0u);
  EXPECT_TRUE(nuri::isValid(
      frameContext.sharedResources.shadowCascadeGraphTextures[0]));
}

TEST(RenderGraphRendererTest,
     ShadowFeatureBuildsRgba8DebugPreviewPassWhenEnabled) {
  std::array<std::byte, 32 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  auto *shadow =
      pipeline.addFeature(std::make_unique<ShadowFeature>(gpu, &memory));
  ASSERT_NE(shadow, nullptr);

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.shadowMapSize = 1024u;
  settings.shadow.debug.showShadowMapViewport = true;
  settings.shadow.debug.previewDepthMin = 0.2f;
  settings.shadow.debug.previewDepthMax = 0.4f;
  settings.shadow.debug.previewDepthInvert = true;
  settings.shadow.debug.previewDepthLog = true;
  RenderFrameContext frameContext{};
  frameContext.frameIndex = 2u;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);

  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);

  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  EXPECT_TRUE(buildResult.value());
  ASSERT_EQ(gpu.createdTextureDescs.size(), 2u);
  const TextureDesc &shadowDesc = gpu.createdTextureDescs[0];
  const TextureDesc &previewDesc = gpu.createdTextureDescs[1];
  EXPECT_EQ(shadowDesc.format, Format::D16_UNORM);
  EXPECT_EQ(previewDesc.format, Format::RGBA8_UNORM);
  EXPECT_EQ(previewDesc.usage, TextureUsage::AttachmentSampled);
  EXPECT_EQ(previewDesc.dimensions.width, 1024u);
  EXPECT_EQ(previewDesc.dimensions.height, 1024u);
  EXPECT_TRUE(
      nuri::isValid(frameContext.sharedResources.shadowDebugPreviewTexture));
  EXPECT_EQ(graph.passCount(), 2u);

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const RenderGraphCompileResult &compiled = compileResult.value();
  ASSERT_EQ(compiled.orderedPasses.size(), 2u);
  const RenderPass &shadowPass = compiled.orderedPasses[0];
  const RenderPass &previewPass = compiled.orderedPasses[1];
  EXPECT_FALSE(shadowPass.hasColorAttachment);
  EXPECT_TRUE(previewPass.hasColorAttachment);
  EXPECT_TRUE(
      sameTexture(previewPass.colorTexture,
                  frameContext.sharedResources.shadowDebugPreviewTexture));
  ASSERT_FALSE(previewPass.draws.empty());
  const DrawItem &previewDraw = previewPass.draws.front();
  ASSERT_EQ(previewDraw.pushConstants.size(),
            sizeof(ShadowPreviewPushConstantsProbe));
  ShadowPreviewPushConstantsProbe previewPc{};
  std::memcpy(&previewPc, previewDraw.pushConstants.data(), sizeof(previewPc));
  EXPECT_EQ(previewPc.sourceTexId, 1u);
  EXPECT_NEAR(previewPc.depthScale, 5.0f, 1.0e-5f);
  EXPECT_NEAR(previewPc.depthBias, -1.0f, 1.0e-5f);
  EXPECT_EQ(previewPc.flags,
            kShadowPreviewProbeFlagInvert | kShadowPreviewProbeFlagLog);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureBuildsOpaqueCasterDrawsWithDirectionalLight) {
  std::array<std::byte, 256 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeShadowSceneGpuDevice gpu;
  gpu.forcedIndexStrideBytes = sizeof(uint16_t);
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(std::make_unique<MaterialTableGpuProvider>(gpu));
  pipeline.addProvider(std::make_unique<SceneLightingProvider>(gpu));
  auto *shadow = pipeline.addFeature(std::make_unique<ShadowFeature>(
      gpu, makeOpaqueConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
      &memory));
  ASSERT_NE(shadow, nullptr);

  const std::filesystem::path tempDir =
      makeTempRendererPath("shadow_feature_scene");
  ASSERT_TRUE(std::filesystem::create_directories(tempDir));
  const std::filesystem::path objPath = tempDir / "shadow_triangle.obj";
  writeTextFile(objPath, "o ShadowTriangle\n"
                         "v -1.0 0.0 0.0\n"
                         "v 1.0 0.0 0.0\n"
                         "v 0.0 1.0 0.0\n"
                         "vt 0.0 0.0\n"
                         "vt 1.0 0.0\n"
                         "vt 0.5 1.0\n"
                         "vn 0.0 0.0 1.0\n"
                         "f 1/1/1 2/2/1 3/3/1\n");

  auto modelResult = renderer.resources().acquireModel(ModelRequest{
      .path = objPath.string(),
      .debugName = "shadow_triangle",
  });
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();

  MaterialRequest materialRequest{};
  materialRequest.debugName = "shadow_material";
  auto materialResult = renderer.resources().acquireMaterial(materialRequest);
  ASSERT_FALSE(materialResult.hasError()) << materialResult.error();

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  auto renderableResult = scene.graph().addRenderable(
      scene.graph().rootNode(), modelResult.value(), materialResult.value());
  ASSERT_FALSE(renderableResult.hasError()) << renderableResult.error();

  LightDesc light{};
  light.type = LightType::Directional;
  light.intensity = 6.0f;
  auto addLightResult = scene.graph().addLight(scene.graph().rootNode(), light);
  ASSERT_FALSE(addLightResult.hasError()) << addLightResult.error();

  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();
  ASSERT_TRUE(commitResult.value());

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.shadowMapSize = 512u;
  settings.shadow.maxDistance = 25.0f;
  settings.shadow.pcfSampleCount = 37u;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 3u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  frameContext.camera.view =
      glm::lookAt(glm::vec3(0.0f, 1.5f, 4.0f), glm::vec3(0.0f, 0.5f, 0.0f),
                  glm::vec3(0.0f, 1.0f, 0.0f));
  frameContext.camera.proj =
      glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 30.0f);
  frameContext.camera.cameraPos = glm::vec4(0.0f, 1.5f, 4.0f, 1.0f);
  frameContext.camera.aspectRatio = 1.0f;
  frameContext.camera.projectionType = ProjectionType::Perspective;
  frameContext.camera.nearPlane = 0.1f;
  frameContext.camera.farPlane = 30.0f;
  frameContext.camera.fovYRadians = glm::radians(60.0f);

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);

  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);

  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());
  ASSERT_EQ(graph.passCount(), 1u);
  ASSERT_TRUE(frameContext.sharedResources.shadowFrameGpuData.has_value());
  EXPECT_NE(frameContext.sharedResources.shadowFrameGpuData->bufferAddress, 0u);
  ShadowFrameGpuData shadowFrame{};
  auto shadowFrameReadResult = gpu.readBuffer(
      frameContext.sharedResources.shadowFrameGpuData->buffer, 0u,
      std::as_writable_bytes(std::span<ShadowFrameGpuData>(&shadowFrame, 1u)));
  ASSERT_FALSE(shadowFrameReadResult.hasError())
      << shadowFrameReadResult.error();
  EXPECT_EQ(shadowFrame.cascades[0].textureSampler.w,
            settings.shadow.pcfSampleCount);
  ASSERT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());
  ASSERT_TRUE(frameContext.sharedResources.selectedShadowLightId.has_value());
  EXPECT_GT(
      frameContext.sharedResources.shadowDebugFrameData->cascades[0].splitFar,
      frameContext.camera.nearPlane);
  EXPECT_NEAR(
      frameContext.sharedResources.shadowDebugFrameData->cascades[0].splitFar,
      settings.shadow.maxDistance, 1.0e-4f);

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const RenderGraphCompileResult &compiled = compileResult.value();
  ASSERT_EQ(compiled.orderedPasses.size(), 1u);
  const RenderPass &shadowPass = compiled.orderedPasses.front();
  EXPECT_FALSE(shadowPass.hasColorAttachment);
  ASSERT_FALSE(shadowPass.draws.empty());
  EXPECT_TRUE(shadowPass.draws.front().depthBiasEnable);
  EXPECT_TRUE(shadowPass.draws.front().useDepthState);
  EXPECT_EQ(shadowPass.draws.front().indexFormat, IndexFormat::U16);
  EXPECT_EQ(shadowPass.draws.front().instanceCount, 1u);
  EXPECT_FALSE(shadowPass.draws.front().pushConstants.empty());
  EXPECT_FALSE(shadowPass.dependencyBuffers.empty());
  EXPECT_EQ(
      frameContext.sharedResources.shadowDebugFrameData->cascades[0].drawCount,
      static_cast<uint32_t>(shadowPass.draws.size()));
}

TEST(RenderGraphRendererTest, ShadowFeatureUsesAlphaPipelineForMaskedCasters) {
  std::array<std::byte, 256 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeShadowSceneGpuDevice gpu;
  gpu.forcedIndexStrideBytes = sizeof(uint16_t);
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(std::make_unique<MaterialTableGpuProvider>(gpu));
  pipeline.addProvider(std::make_unique<SceneLightingProvider>(gpu));
  pipeline.addFeature(std::make_unique<ShadowFeature>(
      gpu, makeOpaqueConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
      &memory));

  const std::filesystem::path tempDir =
      makeTempRendererPath("shadow_masked_scene");
  ASSERT_TRUE(std::filesystem::create_directories(tempDir));
  const std::filesystem::path objPath = tempDir / "shadow_masked.obj";
  writeTextFile(objPath, "o ShadowMasked\n"
                         "v -1.0 0.0 0.0\n"
                         "v 1.0 0.0 0.0\n"
                         "v 0.0 1.0 0.0\n"
                         "vt 0.0 0.0\n"
                         "vt 1.0 0.0\n"
                         "vt 0.5 1.0\n"
                         "vn 0.0 0.0 1.0\n"
                         "f 1/1/1 2/2/1 3/3/1\n");

  auto modelResult = renderer.resources().acquireModel(ModelRequest{
      .path = objPath.string(),
      .debugName = "shadow_masked",
  });
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();

  MaterialRequest materialRequest{};
  materialRequest.desc.alphaMode = MaterialAlphaMode::Mask;
  materialRequest.debugName = "shadow_masked_material";
  auto materialResult = renderer.resources().acquireMaterial(materialRequest);
  ASSERT_FALSE(materialResult.hasError()) << materialResult.error();

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  auto renderableResult = scene.graph().addRenderable(
      scene.graph().rootNode(), modelResult.value(), materialResult.value());
  ASSERT_FALSE(renderableResult.hasError()) << renderableResult.error();

  LightDesc light{};
  light.type = LightType::Directional;
  auto addLightResult = scene.graph().addLight(scene.graph().rootNode(), light);
  ASSERT_FALSE(addLightResult.hasError()) << addLightResult.error();
  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();
  ASSERT_TRUE(commitResult.value());

  RenderSettings settings{};
  settings.shadow.enabled = true;
  RenderFrameContext frameContext{};
  frameContext.frameIndex = 7u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);

  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const RenderGraphCompileResult &compiled = compileResult.value();
  ASSERT_EQ(compiled.orderedPasses.size(), 1u);
  const RenderPass &shadowPass = compiled.orderedPasses.front();
  ASSERT_FALSE(shadowPass.draws.empty());

  auto alphaPipelineIt =
      std::find(gpu.createdRenderPipelineNames.begin(),
                gpu.createdRenderPipelineNames.end(), "shadow_depth_alpha");
  ASSERT_NE(alphaPipelineIt, gpu.createdRenderPipelineNames.end());
  const uint32_t alphaPipelineIndex =
      static_cast<uint32_t>(std::distance(
          gpu.createdRenderPipelineNames.begin(), alphaPipelineIt)) +
      1u;
  EXPECT_EQ(shadowPass.draws.front().pipeline.index, alphaPipelineIndex);
}

TEST(RenderGraphRendererTest,
     ShadowFeaturePublishesBaseColorDependencyForMaskedCasters) {
  std::array<std::byte, 512 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeShadowSceneGpuDevice gpu;
  gpu.forcedIndexStrideBytes = sizeof(uint16_t);
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(std::make_unique<MaterialTableGpuProvider>(gpu));
  pipeline.addProvider(std::make_unique<SceneLightingProvider>(gpu));
  pipeline.addFeature(std::make_unique<ShadowFeature>(
      gpu, makeOpaqueConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
      &memory));

  const std::filesystem::path tempDir =
      makeTempRendererPath("shadow_masked_dependency_scene");
  ASSERT_TRUE(std::filesystem::create_directories(tempDir));
  const std::filesystem::path objPath =
      tempDir / "shadow_masked_dependency.obj";
  writeTextFile(objPath, "o ShadowMaskedDependency\n"
                         "v -1.0 0.0 0.0\n"
                         "v 1.0 0.0 0.0\n"
                         "v 0.0 1.0 0.0\n"
                         "vt 0.0 0.0\n"
                         "vt 1.0 0.0\n"
                         "vt 0.5 1.0\n"
                         "vn 0.0 0.0 1.0\n"
                         "f 1/1/1 2/2/1 3/3/1\n");

  auto modelResult = renderer.resources().acquireModel(ModelRequest{
      .path = objPath.string(),
      .debugName = "shadow_masked_dependency_model",
  });
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();

  const std::filesystem::path baseColorPath =
      std::filesystem::path(PROJECT_SOURCE_DIR) / "build" / "debug" /
      "shadow_masked_dependency.png";
  ASSERT_TRUE(std::filesystem::exists(baseColorPath.parent_path()) ||
              std::filesystem::create_directories(baseColorPath.parent_path()));
  writeTinyPngFile(baseColorPath);
  auto textureResult = renderer.resources().acquireTexture(TextureRequest{
      .path = baseColorPath.string(),
      .loadOptions = TextureLoadOptions{.srgb = true, .generateMipmaps = false},
      .kind = TextureRequestKind::Texture2D,
      .debugName = "shadow_masked_dependency_albedo",
  });
  ASSERT_FALSE(textureResult.hasError()) << textureResult.error();
  const TextureRecord *baseColorRecord =
      renderer.resources().tryGet(textureResult.value());
  ASSERT_NE(baseColorRecord, nullptr);

  MaterialRequest materialRequest{};
  materialRequest.desc.alphaMode = MaterialAlphaMode::Mask;
  materialRequest.desc.textures.baseColor = baseColorRecord->texture;
  materialRequest.textureRefs.baseColor = textureResult.value();
  materialRequest.debugName = "shadow_masked_dependency_material";
  auto materialResult = renderer.resources().acquireMaterial(materialRequest);
  ASSERT_FALSE(materialResult.hasError()) << materialResult.error();

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  auto renderableResult = scene.graph().addRenderable(
      scene.graph().rootNode(), modelResult.value(), materialResult.value());
  ASSERT_FALSE(renderableResult.hasError()) << renderableResult.error();

  LightDesc light{};
  light.type = LightType::Directional;
  auto addLightResult = scene.graph().addLight(scene.graph().rootNode(), light);
  ASSERT_FALSE(addLightResult.hasError()) << addLightResult.error();
  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();
  ASSERT_TRUE(commitResult.value());

  RenderSettings settings{};
  settings.shadow.enabled = true;
  RenderFrameContext frameContext{};
  frameContext.frameIndex = 8u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const RenderGraphCompileResult &compiled = compileResult.value();
  ASSERT_EQ(compiled.orderedPasses.size(), 1u);

  uint32_t baseColorResourceIndex = UINT32_MAX;
  for (uint32_t i = 0u; i < compiled.textureHandlesByResource.size(); ++i) {
    if (sameTexture(compiled.textureHandlesByResource[i],
                    baseColorRecord->texture)) {
      baseColorResourceIndex = i;
      break;
    }
  }
  ASSERT_NE(baseColorResourceIndex, UINT32_MAX);

  const PassBarrierPlan &shadowBarrierPlan = compiled.passBarrierPlans.front();
  bool shadowReadsBaseColor = false;
  for (uint32_t i = 0u; i < shadowBarrierPlan.barrierCount; ++i) {
    const RenderGraphBarrierRecord &barrier =
        compiled.passBarrierRecords[shadowBarrierPlan.barrierOffset + i];
    shadowReadsBaseColor |=
        barrier.resourceKind == RenderGraphBarrierResourceKind::Texture &&
        barrier.resourceIndex == baseColorResourceIndex &&
        hasAccessFlag(barrier.afterAccess, RenderGraphAccessMode::Read);
  }
  EXPECT_TRUE(shadowReadsBaseColor);
}

TEST(RenderGraphRendererTest, ShadowFeatureSkipsBlendMaterialsAsCasters) {
  std::array<std::byte, 256 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeShadowSceneGpuDevice gpu;
  gpu.forcedIndexStrideBytes = sizeof(uint16_t);
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(std::make_unique<MaterialTableGpuProvider>(gpu));
  pipeline.addProvider(std::make_unique<SceneLightingProvider>(gpu));
  pipeline.addFeature(std::make_unique<ShadowFeature>(
      gpu, makeOpaqueConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
      &memory));

  const std::filesystem::path tempDir =
      makeTempRendererPath("shadow_blend_skip_scene");
  ASSERT_TRUE(std::filesystem::create_directories(tempDir));
  const std::filesystem::path objPath = tempDir / "shadow_blend_skip.obj";
  writeTextFile(objPath, "o ShadowBlendSkip\n"
                         "v -1.0 0.0 0.0\n"
                         "v 1.0 0.0 0.0\n"
                         "v 0.0 1.0 0.0\n"
                         "vt 0.0 0.0\n"
                         "vt 1.0 0.0\n"
                         "vt 0.5 1.0\n"
                         "vn 0.0 0.0 1.0\n"
                         "f 1/1/1 2/2/1 3/3/1\n");

  auto modelResult = renderer.resources().acquireModel(ModelRequest{
      .path = objPath.string(),
      .debugName = "shadow_blend_skip_model",
  });
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();

  MaterialRequest materialRequest{};
  materialRequest.desc.alphaMode = MaterialAlphaMode::Blend;
  materialRequest.debugName = "shadow_blend_skip_material";
  auto materialResult = renderer.resources().acquireMaterial(materialRequest);
  ASSERT_FALSE(materialResult.hasError()) << materialResult.error();

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  auto renderableResult = scene.graph().addRenderable(
      scene.graph().rootNode(), modelResult.value(), materialResult.value());
  ASSERT_FALSE(renderableResult.hasError()) << renderableResult.error();

  LightDesc light{};
  light.type = LightType::Directional;
  auto addLightResult = scene.graph().addLight(scene.graph().rootNode(), light);
  ASSERT_FALSE(addLightResult.hasError()) << addLightResult.error();
  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();
  ASSERT_TRUE(commitResult.value());

  RenderSettings settings{};
  settings.shadow.enabled = true;
  RenderFrameContext frameContext{};
  frameContext.frameIndex = 9u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const RenderGraphCompileResult &compiled = compileResult.value();
  ASSERT_EQ(compiled.orderedPasses.size(), 1u);
  EXPECT_TRUE(compiled.orderedPasses.front().draws.empty());
}

TEST(RenderGraphRendererTest,
     SceneLightingProviderPublishesShadowFlagsWhenShadowFrameDataExists) {
  std::array<std::byte, 128 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(std::make_unique<MaterialTableGpuProvider>(gpu));
  pipeline.addProvider(std::make_unique<SceneLightingProvider>(gpu));
  pipeline.addFeature(std::make_unique<ShadowFeature>(gpu, &memory));

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  LightDesc light{};
  light.type = LightType::Directional;
  light.intensity = 4.0f;
  auto addLightResult = scene.graph().addLight(scene.graph().rootNode(), light);
  ASSERT_FALSE(addLightResult.hasError()) << addLightResult.error();
  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();
  ASSERT_TRUE(commitResult.value());

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.debug.visualizeShadowFactor = true;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 4u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);

  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());
  ASSERT_TRUE(frameContext.sharedResources.shadowFrameGpuData.has_value());
  ASSERT_TRUE(frameContext.sharedResources.forwardSceneGpuData.has_value());
  EXPECT_NE(frameContext.sharedResources.shadowFrameGpuData->bufferAddress, 0u);
  EXPECT_EQ(frameContext.sharedResources.forwardSceneGpuData->shadowFlags &
                kShadowFrameFlagEnabled,
            kShadowFrameFlagEnabled);
  EXPECT_EQ(frameContext.sharedResources.forwardSceneGpuData->shadowFlags &
                kShadowFrameFlagVisualizeShadowFactor,
            kShadowFrameFlagVisualizeShadowFactor);
}

TEST(RenderGraphRendererTest, ShadowFeatureUsesRepeatableCameraMaxDistanceFit) {
  std::array<std::byte, 128 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(std::make_unique<MaterialTableGpuProvider>(gpu));
  pipeline.addProvider(std::make_unique<SceneLightingProvider>(gpu));
  pipeline.addFeature(std::make_unique<ShadowFeature>(gpu, &memory));

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  LightDesc light{};
  light.type = LightType::Directional;
  light.intensity = 4.0f;
  auto addLightResult = scene.graph().addLight(scene.graph().rootNode(), light);
  ASSERT_FALSE(addLightResult.hasError()) << addLightResult.error();
  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();
  ASSERT_TRUE(commitResult.value());

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.maxDistance = 20.0f;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 5u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  frameContext.camera.view =
      glm::lookAt(glm::vec3(0.0f, 2.0f, 6.0f), glm::vec3(0.0f, 0.5f, 0.0f),
                  glm::vec3(0.0f, 1.0f, 0.0f));
  frameContext.camera.proj =
      glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 100.0f);
  frameContext.camera.cameraPos = glm::vec4(0.0f, 2.0f, 6.0f, 1.0f);
  frameContext.camera.aspectRatio = 1.0f;
  frameContext.camera.projectionType = ProjectionType::Perspective;
  frameContext.camera.nearPlane = 0.1f;
  frameContext.camera.farPlane = 100.0f;
  frameContext.camera.fovYRadians = glm::radians(60.0f);

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);

  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());
  ASSERT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());
  const ShadowCascadeDebugFrameData &cascade =
      frameContext.sharedResources.shadowDebugFrameData->cascades[0];
  EXPECT_NEAR(cascade.splitNear, frameContext.camera.nearPlane, 1.0e-5f);
  EXPECT_NEAR(cascade.splitFar, settings.shadow.maxDistance, 1.0e-5f);
  EXPECT_GT(cascade.texelWorldSize, 0.0f);
}

TEST(RenderGraphRendererTest, ShadowFeatureFreezeLightViewReusesCachedFit) {
  std::array<std::byte, 128 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(std::make_unique<MaterialTableGpuProvider>(gpu));
  pipeline.addProvider(std::make_unique<SceneLightingProvider>(gpu));
  pipeline.addFeature(std::make_unique<ShadowFeature>(gpu, &memory));

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  LightDesc light{};
  light.type = LightType::Directional;
  light.intensity = 4.0f;
  auto addLightResult = scene.graph().addLight(scene.graph().rootNode(), light);
  ASSERT_FALSE(addLightResult.hasError()) << addLightResult.error();
  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();
  ASSERT_TRUE(commitResult.value());

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.debug.freezeLightView = true;

  RenderFrameContext frameA{};
  frameA.frameIndex = 10u;
  frameA.scene = &scene;
  frameA.resources = &renderer.resources();
  frameA.settings = &settings;
  frameA.camera.view =
      glm::lookAt(glm::vec3(0.0f, 2.0f, 6.0f), glm::vec3(0.0f, 0.5f, 0.0f),
                  glm::vec3(0.0f, 1.0f, 0.0f));
  frameA.camera.proj =
      glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 100.0f);
  frameA.camera.cameraPos = glm::vec4(0.0f, 2.0f, 6.0f, 1.0f);
  frameA.camera.aspectRatio = 1.0f;
  frameA.camera.projectionType = ProjectionType::Perspective;
  frameA.camera.nearPlane = 0.1f;
  frameA.camera.farPlane = 100.0f;
  frameA.camera.fovYRadians = glm::radians(60.0f);

  RenderGraphBuilder graphA(&memory);
  graphA.beginFrame(frameA.frameIndex);
  auto buildResultA =
      pipeline.buildRenderGraph(frameA, renderer.resources(), graphA);
  ASSERT_FALSE(buildResultA.hasError()) << buildResultA.error();
  ASSERT_TRUE(buildResultA.value());
  ASSERT_TRUE(frameA.sharedResources.shadowDebugFrameData.has_value());
  const ShadowCascadeDebugFrameData cascadeA =
      frameA.sharedResources.shadowDebugFrameData->cascades[0];

  RenderFrameContext frameB = frameA;
  frameB.frameIndex = 11u;
  frameB.camera.view =
      glm::lookAt(glm::vec3(7.0f, 3.0f, -5.0f), glm::vec3(1.5f, 0.8f, 0.0f),
                  glm::vec3(0.0f, 1.0f, 0.0f));
  frameB.camera.proj =
      glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 100.0f);
  frameB.camera.cameraPos = glm::vec4(7.0f, 3.0f, -5.0f, 1.0f);

  RenderGraphBuilder graphB(&memory);
  graphB.beginFrame(frameB.frameIndex);
  auto buildResultB =
      pipeline.buildRenderGraph(frameB, renderer.resources(), graphB);
  ASSERT_FALSE(buildResultB.hasError()) << buildResultB.error();
  ASSERT_TRUE(buildResultB.value());
  ASSERT_TRUE(frameB.sharedResources.shadowDebugFrameData.has_value());
  const ShadowCascadeDebugFrameData &cascadeB =
      frameB.sharedResources.shadowDebugFrameData->cascades[0];

  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < 4; ++r) {
      EXPECT_NEAR(cascadeA.lightViewProj[c][r], cascadeB.lightViewProj[c][r],
                  1.0e-6f);
    }
  }
  EXPECT_NEAR(cascadeA.snappedCenter.x, cascadeB.snappedCenter.x, 1.0e-6f);
  EXPECT_NEAR(cascadeA.snappedCenter.y, cascadeB.snappedCenter.y, 1.0e-6f);
  EXPECT_NEAR(cascadeA.snappedCenter.z, cascadeB.snappedCenter.z, 1.0e-6f);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureDisablingFreezeLightViewResumesFitUpdates) {
  std::array<std::byte, 128 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(std::make_unique<MaterialTableGpuProvider>(gpu));
  pipeline.addProvider(std::make_unique<SceneLightingProvider>(gpu));
  pipeline.addFeature(std::make_unique<ShadowFeature>(gpu, &memory));

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  LightDesc light{};
  light.type = LightType::Directional;
  light.intensity = 4.0f;
  auto addLightResult = scene.graph().addLight(scene.graph().rootNode(), light);
  ASSERT_FALSE(addLightResult.hasError()) << addLightResult.error();
  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();
  ASSERT_TRUE(commitResult.value());

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.debug.freezeLightView = true;

  RenderFrameContext frozenFrame{};
  frozenFrame.frameIndex = 12u;
  frozenFrame.scene = &scene;
  frozenFrame.resources = &renderer.resources();
  frozenFrame.settings = &settings;
  frozenFrame.camera.view =
      glm::lookAt(glm::vec3(0.0f, 2.0f, 6.0f), glm::vec3(0.0f, 0.5f, 0.0f),
                  glm::vec3(0.0f, 1.0f, 0.0f));
  frozenFrame.camera.proj =
      glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 100.0f);
  frozenFrame.camera.cameraPos = glm::vec4(0.0f, 2.0f, 6.0f, 1.0f);
  frozenFrame.camera.aspectRatio = 1.0f;
  frozenFrame.camera.projectionType = ProjectionType::Perspective;
  frozenFrame.camera.nearPlane = 0.1f;
  frozenFrame.camera.farPlane = 100.0f;
  frozenFrame.camera.fovYRadians = glm::radians(60.0f);

  RenderGraphBuilder frozenGraph(&memory);
  frozenGraph.beginFrame(frozenFrame.frameIndex);
  auto frozenBuildResult =
      pipeline.buildRenderGraph(frozenFrame, renderer.resources(), frozenGraph);
  ASSERT_FALSE(frozenBuildResult.hasError()) << frozenBuildResult.error();
  ASSERT_TRUE(frozenBuildResult.value());
  ASSERT_TRUE(frozenFrame.sharedResources.shadowDebugFrameData.has_value());
  const ShadowCascadeDebugFrameData frozenCascade =
      frozenFrame.sharedResources.shadowDebugFrameData->cascades[0];

  RenderFrameContext movedFrozenFrame = frozenFrame;
  movedFrozenFrame.frameIndex = 13u;
  movedFrozenFrame.camera.view =
      glm::lookAt(glm::vec3(6.0f, 4.0f, -6.0f), glm::vec3(2.0f, 1.0f, 0.0f),
                  glm::vec3(0.0f, 1.0f, 0.0f));
  movedFrozenFrame.camera.proj =
      glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 100.0f);
  movedFrozenFrame.camera.cameraPos = glm::vec4(6.0f, 4.0f, -6.0f, 1.0f);

  RenderGraphBuilder movedFrozenGraph(&memory);
  movedFrozenGraph.beginFrame(movedFrozenFrame.frameIndex);
  auto movedFrozenBuildResult = pipeline.buildRenderGraph(
      movedFrozenFrame, renderer.resources(), movedFrozenGraph);
  ASSERT_FALSE(movedFrozenBuildResult.hasError())
      << movedFrozenBuildResult.error();
  ASSERT_TRUE(movedFrozenBuildResult.value());
  ASSERT_TRUE(
      movedFrozenFrame.sharedResources.shadowDebugFrameData.has_value());

  settings.shadow.debug.freezeLightView = false;
  RenderFrameContext unfrozenFrame = movedFrozenFrame;
  unfrozenFrame.frameIndex = 14u;
  unfrozenFrame.settings = &settings;

  RenderGraphBuilder unfrozenGraph(&memory);
  unfrozenGraph.beginFrame(unfrozenFrame.frameIndex);
  auto unfrozenBuildResult = pipeline.buildRenderGraph(
      unfrozenFrame, renderer.resources(), unfrozenGraph);
  ASSERT_FALSE(unfrozenBuildResult.hasError()) << unfrozenBuildResult.error();
  ASSERT_TRUE(unfrozenBuildResult.value());
  ASSERT_TRUE(unfrozenFrame.sharedResources.shadowDebugFrameData.has_value());
  const ShadowCascadeDebugFrameData &unfrozenCascade =
      unfrozenFrame.sharedResources.shadowDebugFrameData->cascades[0];

  bool matrixChanged = false;
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < 4; ++r) {
      matrixChanged |= std::abs(unfrozenCascade.lightViewProj[c][r] -
                                frozenCascade.lightViewProj[c][r]) > 1.0e-5f;
    }
  }
  EXPECT_TRUE(matrixChanged);
}

TEST(RenderGraphRendererTest, OpaqueMainPassReadsShadowDepthAfterShadowPass) {
  std::vector<std::byte> scratchBytes(1024 * 1024);
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeShadowSceneGpuDevice gpu;
  gpu.forcedIndexStrideBytes = sizeof(uint16_t);
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(std::make_unique<MaterialTableGpuProvider>(gpu));
  pipeline.addProvider(
      std::make_unique<FrameCompositionProvider>(gpu, &memory));
  pipeline.addProvider(std::make_unique<SceneLightingProvider>(gpu));
  pipeline.addFeature(std::make_unique<ShadowFeature>(
      gpu, makeOpaqueConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
      &memory));
  pipeline.addFeature(std::make_unique<OpaqueFeature>(
      gpu, makeOpaqueConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
      &memory));

  const std::filesystem::path tempDir =
      makeTempRendererPath("shadow_opaque_dependency_scene");
  ASSERT_TRUE(std::filesystem::create_directories(tempDir));
  const std::filesystem::path objPath = tempDir / "shadow_triangle.obj";
  writeTextFile(objPath, "o ShadowTriangle\n"
                         "v -1.0 0.0 0.0\n"
                         "v 1.0 0.0 0.0\n"
                         "v 0.0 1.0 0.0\n"
                         "vt 0.0 0.0\n"
                         "vt 1.0 0.0\n"
                         "vt 0.5 1.0\n"
                         "vn 0.0 0.0 1.0\n"
                         "f 1/1/1 2/2/1 3/3/1\n");

  auto modelResult = renderer.resources().acquireModel(ModelRequest{
      .path = objPath.string(),
      .debugName = "shadow_opaque_dependency_triangle",
  });
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();

  MaterialRequest materialRequest{};
  materialRequest.debugName = "shadow_opaque_dependency_material";
  auto materialResult = renderer.resources().acquireMaterial(materialRequest);
  ASSERT_FALSE(materialResult.hasError()) << materialResult.error();

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  auto renderableResult = scene.graph().addRenderable(
      scene.graph().rootNode(), modelResult.value(), materialResult.value());
  ASSERT_FALSE(renderableResult.hasError()) << renderableResult.error();

  LightDesc light{};
  light.type = LightType::Directional;
  light.intensity = 6.0f;
  auto addLightResult = scene.graph().addLight(scene.graph().rootNode(), light);
  ASSERT_FALSE(addLightResult.hasError()) << addLightResult.error();
  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();
  ASSERT_TRUE(commitResult.value());

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.shadowMapSize = 512u;
  settings.opaque.enableInstanceCompute = false;
  settings.opaque.enableIndirectDraw = false;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 6u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  frameContext.camera.view =
      glm::lookAt(glm::vec3(0.0f, 1.5f, 4.0f), glm::vec3(0.0f, 0.5f, 0.0f),
                  glm::vec3(0.0f, 1.0f, 0.0f));
  frameContext.camera.proj =
      glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 30.0f);
  frameContext.camera.cameraPos = glm::vec4(0.0f, 1.5f, 4.0f, 1.0f);
  frameContext.camera.aspectRatio = 1.0f;
  frameContext.camera.projectionType = ProjectionType::Perspective;
  frameContext.camera.nearPlane = 0.1f;
  frameContext.camera.farPlane = 30.0f;
  frameContext.camera.fovYRadians = glm::radians(60.0f);

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);

  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const RenderGraphCompileResult &compiled = compileResult.value();

  const TextureHandle shadowTexture =
      frameContext.sharedResources.shadowCascadeTextures[0];
  uint32_t shadowTextureResourceIndex = UINT32_MAX;
  for (uint32_t i = 0u; i < compiled.textureHandlesByResource.size(); ++i) {
    if (sameTexture(compiled.textureHandlesByResource[i], shadowTexture)) {
      shadowTextureResourceIndex = i;
      break;
    }
  }
  ASSERT_NE(shadowTextureResourceIndex, UINT32_MAX);

  bool sawShadowPass = false;
  bool sawOpaquePass = false;
  size_t shadowPassIndex = 0u;
  size_t opaquePassIndex = 0u;
  for (size_t i = 0u; i < compiled.orderedPasses.size(); ++i) {
    const RenderPass &pass = compiled.orderedPasses[i];
    if (pass.debugLabel == "ShadowDepthPass") {
      sawShadowPass = true;
      shadowPassIndex = i;
      EXPECT_TRUE(sameTexture(pass.depthTexture, shadowTexture));
    }
    if (pass.debugLabel == "Opaque Pass") {
      sawOpaquePass = true;
      opaquePassIndex = i;
    }
  }
  ASSERT_TRUE(sawShadowPass);
  ASSERT_TRUE(sawOpaquePass);
  EXPECT_LT(shadowPassIndex, opaquePassIndex);
  ASSERT_LT(opaquePassIndex, compiled.passBarrierPlans.size());
  const PassBarrierPlan &opaqueBarrierPlan =
      compiled.passBarrierPlans[opaquePassIndex];
  bool opaqueReadsShadowTexture = false;
  for (uint32_t i = 0u; i < opaqueBarrierPlan.barrierCount; ++i) {
    const RenderGraphBarrierRecord &barrier =
        compiled.passBarrierRecords[opaqueBarrierPlan.barrierOffset + i];
    opaqueReadsShadowTexture |=
        barrier.resourceKind == RenderGraphBarrierResourceKind::Texture &&
        barrier.resourceIndex == shadowTextureResourceIndex &&
        hasAccessFlag(barrier.afterAccess, RenderGraphAccessMode::Read);
  }
  EXPECT_TRUE(opaqueReadsShadowTexture);
  ASSERT_TRUE(frameContext.sharedResources.forwardSceneGpuData.has_value());
  EXPECT_EQ(frameContext.sharedResources.forwardSceneGpuData->shadowFlags &
                kShadowFrameFlagEnabled,
            kShadowFrameFlagEnabled);
}

TEST(RenderGraphRendererTest,
     DebugFeatureBuildsShadowOverlayFromFrameDebugData) {
  std::array<std::byte, 64 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  auto *debug = pipeline.addFeature(
      std::make_unique<DebugFeature>(gpu, DebugRendererConfig{}, &memory));
  ASSERT_NE(debug, nullptr);

  auto colorResult =
      gpu.createTexture(TextureDesc{.type = TextureType::Texture2D,
                                    .format = Format::RGBA8_UNORM,
                                    .dimensions = {640u, 480u, 1u},
                                    .usage = TextureUsage::AttachmentSampled,
                                    .storage = Storage::Device,
                                    .numLayers = 1u,
                                    .numSamples = 1u,
                                    .numMipLevels = 1u},
                        "test_frame_color");
  ASSERT_FALSE(colorResult.hasError()) << colorResult.error();
  auto depthResult =
      gpu.createTexture(TextureDesc{.type = TextureType::Texture2D,
                                    .format = Format::D32_FLOAT,
                                    .dimensions = {640u, 480u, 1u},
                                    .usage = TextureUsage::AttachmentSampled,
                                    .storage = Storage::Device,
                                    .numLayers = 1u,
                                    .numSamples = 1u,
                                    .numMipLevels = 1u},
                        "test_scene_depth");
  ASSERT_FALSE(depthResult.hasError()) << depthResult.error();

  ShadowDebugFrameData shadowDebug{};
  shadowDebug.cascadeCount = 1u;
  ShadowCascadeDebugFrameData &cascade = shadowDebug.cascades[0];
  cascade.splitNear = 0.1f;
  cascade.splitFar = 10.0f;
  cascade.texelWorldSize = 0.01f;
  cascade.inverseLightView = glm::mat4(1.0f);
  cascade.lightSpaceBoundsMin = glm::vec4(-2.0f, -2.0f, -4.0f, 1.0f);
  cascade.lightSpaceBoundsMax = glm::vec4(2.0f, 2.0f, -0.1f, 1.0f);
  cascade.snappedCenter = glm::vec4(0.0f, 0.0f, -2.0f, 1.0f);
  cascade.unsnappedCenter = cascade.snappedCenter;
  cascade.worldFrustumCorners = {
      glm::vec4(-1.0f, -1.0f, -1.0f, 1.0f), glm::vec4(1.0f, -1.0f, -1.0f, 1.0f),
      glm::vec4(1.0f, 1.0f, -1.0f, 1.0f),   glm::vec4(-1.0f, 1.0f, -1.0f, 1.0f),
      glm::vec4(-2.0f, -2.0f, -4.0f, 1.0f), glm::vec4(2.0f, -2.0f, -4.0f, 1.0f),
      glm::vec4(2.0f, 2.0f, -4.0f, 1.0f),   glm::vec4(-2.0f, 2.0f, -4.0f, 1.0f),
  };

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.debug.showCascadeFrusta = true;
  settings.shadow.debug.showLightViewBounds = true;
  settings.shadow.debug.showTexelGridSnap = true;
  settings.debug.lightIcons = false;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 4u;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  frameContext.camera.view =
      glm::lookAt(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                  glm::vec3(0.0f, 1.0f, 0.0f));
  frameContext.camera.proj =
      glm::perspective(glm::radians(60.0f), 4.0f / 3.0f, 0.1f, 100.0f);
  frameContext.sharedResources.frameColorTexture = colorResult.value();
  frameContext.sharedResources.sceneDepthTexture = depthResult.value();
  frameContext.sharedResources.shadowDebugFrameData = shadowDebug;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);

  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);

  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());
  ASSERT_EQ(graph.passCount(), 1u);

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const RenderGraphCompileResult &compiled = compileResult.value();
  ASSERT_EQ(compiled.orderedPasses.size(), 1u);
  const RenderPass &overlayPass = compiled.orderedPasses.front();
  EXPECT_TRUE(overlayPass.hasColorAttachment);
  EXPECT_TRUE(sameTexture(overlayPass.colorTexture, colorResult.value()));
  EXPECT_FALSE(isValid(overlayPass.depthTexture));
  ASSERT_FALSE(overlayPass.draws.empty());
  EXPECT_FALSE(overlayPass.draws.front().useDepthState);
  EXPECT_GT(overlayPass.draws.front().vertexCount, 54u);
}

TEST(RenderGraphRendererTest, DebugGridPipelineUsesFrameColorTextureFormat) {
  std::array<std::byte, 64 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  const std::filesystem::path tempRoot = makeTempRendererPath("debug_grid");
  std::filesystem::create_directories(tempRoot);
  const std::filesystem::path vertexPath = tempRoot / "debug_grid.vert";
  const std::filesystem::path fragmentPath = tempRoot / "debug_grid.frag";
  {
    std::ofstream vertex(vertexPath);
    vertex << "#version 460\nvoid main() {}\n";
    std::ofstream fragment(fragmentPath);
    fragment << "#version 460\nvoid main() {}\n";
  }
  DebugRendererConfig debugConfig{};
  debugConfig.vertex = vertexPath;
  debugConfig.fragment = fragmentPath;
  auto *debug = pipeline.addFeature(
      std::make_unique<DebugFeature>(gpu, debugConfig, &memory));
  ASSERT_NE(debug, nullptr);

  auto colorResult =
      gpu.createTexture(TextureDesc{.type = TextureType::Texture2D,
                                    .format = Format::RGBA16_FLOAT,
                                    .dimensions = {640u, 480u, 1u},
                                    .usage = TextureUsage::AttachmentSampled,
                                    .storage = Storage::Device,
                                    .numLayers = 1u,
                                    .numSamples = 1u,
                                    .numMipLevels = 1u},
                        "hdr_frame_color");
  ASSERT_FALSE(colorResult.hasError()) << colorResult.error();

  RenderSettings settings{};
  settings.debug.grid = true;
  settings.debug.lightIcons = false;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 7u;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  frameContext.camera.view = glm::mat4(1.0f);
  frameContext.camera.proj = glm::mat4(1.0f);
  frameContext.sharedResources.frameColorTexture = colorResult.value();

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);

  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());
  ASSERT_EQ(graph.passCount(), 1u);

  ASSERT_EQ(gpu.createdRenderPipelineDescs.size(), 1u);
  EXPECT_EQ(gpu.createdRenderPipelineNames.front(), "debug_grid");
  const RenderPipelineDesc &desc = gpu.createdRenderPipelineDescs.front();
  ASSERT_EQ(desc.colorFormats.size(), 1u);
  EXPECT_EQ(desc.colorFormats.front(), Format::RGBA16_FLOAT);

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  ASSERT_EQ(compileResult.value().orderedPasses.size(), 1u);
  const RenderPass &gridPass = compileResult.value().orderedPasses.front();
  EXPECT_TRUE(sameTexture(gridPass.colorTexture, colorResult.value()));

  std::filesystem::remove_all(tempRoot);
}

TEST(RenderGraphRendererTest,
     SceneLightingProviderSkipsUnchangedStaticUploadsPerSlot) {
  std::array<std::byte, 32 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeRendererGPUDevice gpu;
  gpu.swapchainImageCount = 1u;
  Renderer renderer(gpu, memory);
  RenderScene scene(&memory);
  SceneLightingProvider provider(gpu);
  RenderGraphBuilder graph(&memory);
  RenderFrameContext frameContext{};

  LightDesc light{};
  light.type = LightType::Point;
  light.intensity = 4.0f;
  light.range = 12.0f;
  auto addLightResult = scene.graph().addLight(scene.graph().rootNode(), light);
  ASSERT_FALSE(addLightResult.hasError());
  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError());
  ASSERT_TRUE(commitResult.value());

  auto materialBufferResult =
      gpu.createBuffer(BufferDesc{.usage = BufferUsage::Storage,
                                  .storage = Storage::Device,
                                  .size = 256u},
                       "test_material_table");
  ASSERT_FALSE(materialBufferResult.hasError());
  const BufferHandle materialBuffer = materialBufferResult.value();
  ASSERT_TRUE(gpu.isValid(materialBuffer));
  const uint64_t materialBufferAddress = 0x200000000ull;

  frameContext.frameIndex = 0u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.camera.view = glm::mat4(1.0f);
  frameContext.camera.proj = glm::mat4(1.0f);
  frameContext.camera.cameraPos = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
  frameContext.sharedResources.materialTableGpuData = MaterialTableGpuData{
      .headerBuffer = materialBuffer,
      .clearcoatBuffer = materialBuffer,
      .sheenBuffer = materialBuffer,
      .transmissionBuffer = materialBuffer,
      .specularBuffer = materialBuffer,
      .headerBufferAddress = materialBufferAddress,
      .clearcoatBufferAddress = materialBufferAddress,
      .sheenBufferAddress = materialBufferAddress,
      .transmissionBufferAddress = materialBufferAddress,
      .specularBufferAddress = materialBufferAddress,
      .version = 1u,
  };

  FrameBuildContext ctx{
      .frame = frameContext,
      .graph = graph,
      .resources = renderer.resources(),
      .shared = frameContext.sharedResources,
  };

  auto prepareResult = provider.prepare(ctx);
  ASSERT_FALSE(prepareResult.hasError());
  ASSERT_TRUE(prepareResult.value());
  const uint32_t firstUploadCount = gpu.updateBufferCallCount;
  EXPECT_EQ(firstUploadCount, 3u);

  prepareResult = provider.prepare(ctx);
  ASSERT_FALSE(prepareResult.hasError());
  ASSERT_TRUE(prepareResult.value());
  EXPECT_EQ(gpu.updateBufferCallCount, firstUploadCount);

  const auto sunNodeResult =
      scene.graph().createNode(scene.graph().rootNode(), "Sun Node");
  ASSERT_FALSE(sunNodeResult.hasError());
  ASSERT_TRUE(scene.graph().setLightNode(addLightResult.value(),
                                         sunNodeResult.value()));
  ASSERT_TRUE(scene.graph().setNodeLocalTransform(
      sunNodeResult.value(),
      glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 0.0f, 0.0f))));
  commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError());

  prepareResult = provider.prepare(ctx);
  ASSERT_FALSE(prepareResult.hasError());
  ASSERT_TRUE(prepareResult.value());
  EXPECT_EQ(gpu.updateBufferCallCount, firstUploadCount + 1u);
}

} // namespace
