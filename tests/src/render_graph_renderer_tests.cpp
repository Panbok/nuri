#include "tests_pch.h"

#include "render_graph_test_support.h"

#include <gtest/gtest.h>

#include "nuri/core/log.h"
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
#include <limits>
#include <memory_resource>
#include <span>
#include <vector>

#include <lvk/LVK.h>
#include <vulkan/VulkanUtils.h>
#include <vulkan/vulkan_core.h>

namespace {

using namespace nuri;
using namespace nuri::test_support;

constexpr uint32_t kShadowPreviewProbeFlagInvert = 1u << 0u;
constexpr uint32_t kShadowPreviewProbeFlagLog = 1u << 1u;
constexpr uint32_t kShadowPreviewProbeFlagTiled = 1u << 2u;

struct ShadowPreviewPushConstantsProbe {
  std::array<uint32_t, 4> sourceTexIds{};
  std::array<uint32_t, 4> previewParams{};
  std::array<float, 4> depthParams{};
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

RuntimeOpaqueShaderConfig makeShadowConfig(const std::filesystem::path &root) {
  return RuntimeOpaqueShaderConfig{
      .shaderBasePath = root / "assets" / "shaders",
  };
}

CameraFrameState makeSdsmPerspectiveCamera(float farPlane = 100.0f) {
  return CameraFrameState{
      .view =
          glm::lookAt(glm::vec3(0.0f, 2.0f, 6.0f), glm::vec3(0.0f, 0.5f, 0.0f),
                      glm::vec3(0.0f, 1.0f, 0.0f)),
      .proj = glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, farPlane),
      .cameraPos = glm::vec4(0.0f, 2.0f, 6.0f, 1.0f),
      .aspectRatio = 1.0f,
      .projectionType = ProjectionType::Perspective,
      .nearPlane = 0.1f,
      .farPlane = farPlane,
      .fovYRadians = glm::radians(60.0f),
  };
}

CameraFrameState makeSdsmOrthographicCamera(float farPlane = 100.0f,
                                            float orthoHeight = 24.0f) {
  const float halfHeight = orthoHeight * 0.5f;
  return CameraFrameState{
      .view =
          glm::lookAt(glm::vec3(0.0f, 2.0f, 6.0f), glm::vec3(0.0f, 0.5f, 0.0f),
                      glm::vec3(0.0f, 1.0f, 0.0f)),
      .proj = glm::ortho(-halfHeight, halfHeight, -halfHeight, halfHeight, 0.1f,
                         farPlane),
      .cameraPos = glm::vec4(0.0f, 2.0f, 6.0f, 1.0f),
      .aspectRatio = 1.0f,
      .projectionType = ProjectionType::Orthographic,
      .nearPlane = 0.1f,
      .farPlane = farPlane,
      .orthoHeight = orthoHeight,
  };
}

float deviceDepthForViewDepth(float viewDepth, const CameraFrameState &camera) {
  const float clampedDepth =
      std::clamp(viewDepth, camera.nearPlane + 1.0e-4f, camera.farPlane);
  if (camera.projectionType == ProjectionType::Orthographic) {
    const float depthRange =
        std::max(camera.farPlane - camera.nearPlane, 1.0e-4f);
    return std::clamp((clampedDepth - camera.nearPlane) / depthRange, 0.0f,
                      1.0f);
  }
  const float numerator =
      camera.farPlane - (camera.nearPlane * camera.farPlane) / clampedDepth;
  return std::clamp(numerator / (camera.farPlane - camera.nearPlane), 0.0f,
                    1.0f);
}

std::array<float, kMaxShadowCascades + 1u>
expectedHistogramEffectiveSplitDepths(const ShadowSdsmDebugFrameData &sdsm) {
  std::array<float, kMaxShadowCascades + 1u> expected =
      sdsm.histogramSplitDepths;
  const uint32_t cascadeCount =
      std::clamp(sdsm.splitCount, 1u, kMaxShadowCascades);
  const float effectiveNear = std::max(sdsm.effectiveRangeNear, 0.01f);
  const float effectiveFar =
      std::max(effectiveNear + 1.0e-4f, sdsm.effectiveRangeFar);
  expected[0] = effectiveNear;
  if (cascadeCount == 1u) {
    expected[1] = effectiveFar;
    return expected;
  }

  const float histogramNear = sdsm.histogramSplitDepths[0];
  const float histogramFar = std::max(sdsm.histogramSplitDepths[cascadeCount],
                                      histogramNear + 1.0e-4f);
  const float histogramSpan = std::max(histogramFar - histogramNear, 1.0e-4f);
  const float effectiveSpan = effectiveFar - effectiveNear;
  for (uint32_t cascadeIndex = 1u; cascadeIndex < cascadeCount;
       ++cascadeIndex) {
    const float normalized =
        std::clamp((sdsm.histogramSplitDepths[cascadeIndex] - histogramNear) /
                       histogramSpan,
                   0.0f, 1.0f);
    expected[cascadeIndex] = effectiveNear + normalized * effectiveSpan;
  }
  shadow_detail::enforceMonotonicShadowSplitDepths(expected, cascadeCount,
                                                   effectiveNear, effectiveFar);
  return expected;
}

void addDirectionalLightToScene(RenderScene &scene) {
  LightDesc light{};
  light.type = LightType::Directional;
  light.intensity = 4.0f;
  auto addLightResult = scene.graph().addLight(scene.graph().rootNode(), light);
  ASSERT_FALSE(addLightResult.hasError()) << addLightResult.error();
  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();
  ASSERT_TRUE(commitResult.value());
}

ShadowSdsmDebugFrameData
buildShadowSdsmFrame(RenderPipeline &pipeline, Renderer &renderer,
                     std::pmr::memory_resource &memory, RenderScene &scene,
                     RenderSettings &settings, const CameraFrameState &camera,
                     uint64_t frameIndex,
                     std::optional<uint64_t> sourceFrameIndex,
                     std::span<const TextureHandle> pyramidTextures) {
  RenderFrameContext frameContext{};
  frameContext.frameIndex = frameIndex;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  frameContext.camera = camera;
  for (size_t i = 0u;
       i < pyramidTextures.size() &&
       i < frameContext.sharedResources.sceneDepthPyramidTextures.size();
       ++i) {
    frameContext.sharedResources.sceneDepthPyramidTextures[i] =
        pyramidTextures[i];
  }
  frameContext.sharedResources.sceneDepthPyramidLevelCount =
      static_cast<uint32_t>(pyramidTextures.size());
  frameContext.sharedResources.sceneDepthPyramidSourceFrameIndex =
      sourceFrameIndex;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  EXPECT_FALSE(buildResult.hasError()) << buildResult.error();
  EXPECT_TRUE(buildResult.value());
  EXPECT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());
  return frameContext.sharedResources.shadowDebugFrameData->sdsm;
}

uint64_t currentLogSequence() {
  std::vector<LogEntry> entries;
  const LogReadResult readResult = readLogEntriesSince(0u, entries);
  return readResult.lastSequence;
}

size_t countLogEntriesSince(uint64_t afterSequence, std::string_view needle) {
  std::vector<LogEntry> entries;
  (void)readLogEntriesSince(afterSequence, entries);

  size_t count = 0u;
  for (const LogEntry &entry : entries) {
    if (entry.message.find(needle) != std::string::npos) {
      ++count;
    }
  }
  return count;
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

class FakeNoComputeFullscreenGpuDevice final : public FakeFullscreenGpuDevice {
public:
  Result<ComputePipelineHandle, std::string>
  createComputePipeline(const ComputePipelineDesc &,
                        std::string_view) override {
    return Result<ComputePipelineHandle, std::string>::makeError(
        "compute pipelines unavailable");
  }
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
  EXPECT_EQ(settings.shadow.splitMode, ShadowCascadeSplitMode::Practical);
  EXPECT_EQ(settings.shadow.filterMode, ShadowFilterMode::Hard);
  EXPECT_EQ(settings.shadow.pcssBlockerSampleCount, 16u);
  EXPECT_EQ(settings.shadow.pcssFilterSampleCount, 32u);
  EXPECT_FLOAT_EQ(settings.shadow.pcssLightRadiusScale,
                  kDefaultPcssLightRadiusScale);
  EXPECT_FLOAT_EQ(settings.shadow.pcssSearchRadiusClampTexels,
                  kDefaultPcssSearchRadiusClampTexels);
  EXPECT_FLOAT_EQ(settings.shadow.pcssFilterRadiusClampTexels,
                  kDefaultPcssFilterRadiusClampTexels);
  EXPECT_EQ(settings.shadow.sdsmMode, ShadowSdsmMode::Disabled);
  EXPECT_FALSE(settings.shadow.debug.logDiagnostics);
  EXPECT_EQ(settings.shadow.debug.diagnosticLogLevel, LogLevel::Trace);
  EXPECT_EQ(settings.shadow.debug.diagnosticLogIntervalFrames, 1u);
  EXPECT_FALSE(settings.shadow.debug.diagnosticLogOnlyOnChange);
}

TEST(RenderGraphRendererTest, ShadowSettingsSanitizeClampsCoreValues) {
  RenderSettings::ShadowSettings settings{};
  settings.cascadeCount = 0u;
  settings.shadowMapSize = 0u;
  settings.maxDistance = -10.0f;
  settings.splitMode = static_cast<ShadowCascadeSplitMode>(99u);
  settings.splitLambda = 2.0f;
  settings.cascadeBlendFraction = -1.0f;
  settings.pcfSampleCount = 0u;
  settings.pcssBlockerSampleCount = 0u;
  settings.pcssFilterSampleCount = 0u;
  settings.pcssLightRadiusScale = std::numeric_limits<float>::quiet_NaN();
  settings.pcssSearchRadiusClampTexels =
      std::numeric_limits<float>::quiet_NaN();
  settings.pcssFilterRadiusClampTexels =
      std::numeric_limits<float>::quiet_NaN();
  settings.debug.showLightPerspectiveViewport = true;
  settings.debug.diagnosticLogLevel = static_cast<LogLevel>(99u);
  settings.debug.diagnosticLogIntervalFrames = 0u;
  settings.debug.debugCascadeIndex = 99u;
  settings.debug.previewDepthMin = -1.0f;
  settings.debug.previewDepthMax = 2.0f;

  sanitizeShadowSettings(settings);

  EXPECT_EQ(settings.cascadeCount, 1u);
  EXPECT_EQ(settings.shadowMapSize, 1u);
  EXPECT_FLOAT_EQ(settings.maxDistance, 150.0f);
  EXPECT_EQ(settings.splitMode, ShadowCascadeSplitMode::Practical);
  EXPECT_FLOAT_EQ(settings.splitLambda, 1.0f);
  EXPECT_FLOAT_EQ(settings.cascadeBlendFraction, 0.0f);
  EXPECT_EQ(settings.pcfSampleCount, 1u);
  EXPECT_EQ(settings.pcssBlockerSampleCount, 1u);
  EXPECT_EQ(settings.pcssFilterSampleCount, 1u);
  EXPECT_FLOAT_EQ(settings.pcssLightRadiusScale, kDefaultPcssLightRadiusScale);
  EXPECT_FLOAT_EQ(settings.pcssSearchRadiusClampTexels,
                  kDefaultPcssSearchRadiusClampTexels);
  EXPECT_FLOAT_EQ(settings.pcssFilterRadiusClampTexels,
                  kDefaultPcssFilterRadiusClampTexels);
  EXPECT_FALSE(settings.debug.showLightPerspectiveViewport);
  EXPECT_EQ(settings.debug.diagnosticLogLevel, LogLevel::Trace);
  EXPECT_EQ(settings.debug.diagnosticLogIntervalFrames, 1u);
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
     ShadowFeatureLogsDiagnosticsAtConfiguredInterval) {
  std::array<std::byte, 128 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeShadowSceneGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(std::make_unique<MaterialTableGpuProvider>(gpu));
  pipeline.addProvider(std::make_unique<SceneLightingProvider>(gpu));
  pipeline.addFeature(std::make_unique<ShadowFeature>(
      gpu, makeOpaqueConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
      &memory));

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  addDirectionalLightToScene(scene);

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.debug.logDiagnostics = true;
  settings.shadow.debug.diagnosticLogLevel = LogLevel::Debug;
  settings.shadow.debug.diagnosticLogIntervalFrames = 2u;

  const CameraFrameState camera{
      .projectionType = ProjectionType::Perspective,
      .nearPlane = 0.1f,
      .farPlane = 30.0f,
  };

  const uint64_t baselineSequence = currentLogSequence();
  for (uint64_t frameIndex = 0u; frameIndex < 3u; ++frameIndex) {
    RenderFrameContext frameContext{};
    frameContext.frameIndex = frameIndex;
    frameContext.scene = &scene;
    frameContext.resources = &renderer.resources();
    frameContext.settings = &settings;
    frameContext.camera = camera;

    RenderGraphBuilder graph(&memory);
    graph.beginFrame(frameContext.frameIndex);
    auto buildResult =
        pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
    ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
    ASSERT_TRUE(buildResult.value());
  }

  EXPECT_EQ(
      countLogEntriesSince(baselineSequence, "Shadow diagnostics summary"), 2u);
  EXPECT_EQ(
      countLogEntriesSince(baselineSequence, "Shadow diagnostics cascade"), 8u);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureLogsLightFitChangeComponentsWhenCameraMoves) {
  std::array<std::byte, 128 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeShadowSceneGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(std::make_unique<MaterialTableGpuProvider>(gpu));
  pipeline.addProvider(std::make_unique<SceneLightingProvider>(gpu));
  pipeline.addFeature(std::make_unique<ShadowFeature>(
      gpu, makeOpaqueConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
      &memory));

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  addDirectionalLightToScene(scene);

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.debug.logDiagnostics = true;
  settings.shadow.debug.diagnosticLogLevel = LogLevel::Debug;
  settings.shadow.debug.diagnosticLogIntervalFrames = 1u;

  const CameraFrameState cameraA = makeSdsmPerspectiveCamera(30.0f);
  CameraFrameState cameraB = cameraA;
  cameraB.view =
      glm::lookAt(glm::vec3(0.4f, 2.0f, 6.0f), glm::vec3(0.4f, 0.5f, 0.0f),
                  glm::vec3(0.0f, 1.0f, 0.0f));
  cameraB.cameraPos = glm::vec4(0.4f, 2.0f, 6.0f, 1.0f);

  const uint64_t baselineSequence = currentLogSequence();
  for (uint64_t frameIndex = 0u; frameIndex < 2u; ++frameIndex) {
    RenderFrameContext frameContext{};
    frameContext.frameIndex = frameIndex;
    frameContext.scene = &scene;
    frameContext.resources = &renderer.resources();
    frameContext.settings = &settings;
    frameContext.camera = frameIndex == 0u ? cameraA : cameraB;

    RenderGraphBuilder graph(&memory);
    graph.beginFrame(frameContext.frameIndex);
    auto buildResult =
        pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
    ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
    ASSERT_TRUE(buildResult.value());
  }

  EXPECT_GE(
      countLogEntriesSince(baselineSequence, "Shadow diagnostics light change"),
      1u);
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
     ShadowCasterLightSpaceVolumeCullingRejectsOutsideBounds) {
  const glm::mat4 lightView(1.0f);
  const glm::vec3 boundsMin(-1.0f, -1.0f, -1.0f);
  const glm::vec3 boundsMax(1.0f, 1.0f, 1.0f);

  const auto overlappingCaster =
      shadow_detail::computeBoundsCorners(glm::vec3(-0.25f), glm::vec3(0.25f));
  EXPECT_TRUE(shadow_detail::shadowCasterOverlapsLightSpaceBounds(
      std::span<const glm::vec3, 8>(overlappingCaster), lightView, boundsMin,
      boundsMax, 0.01f));

  const auto outsideCaster = shadow_detail::computeBoundsCorners(
      glm::vec3(4.0f, -0.25f, -0.25f), glm::vec3(5.0f, 0.25f, 0.25f));
  EXPECT_FALSE(shadow_detail::shadowCasterOverlapsLightSpaceBounds(
      std::span<const glm::vec3, 8>(outsideCaster), lightView, boundsMin,
      boundsMax, 0.01f));

  const auto paddedEdgeCaster = shadow_detail::computeBoundsCorners(
      glm::vec3(1.05f, -0.25f, -0.25f), glm::vec3(1.10f, 0.25f, 0.25f));
  EXPECT_TRUE(shadow_detail::shadowCasterOverlapsLightSpaceBounds(
      std::span<const glm::vec3, 8>(paddedEdgeCaster), lightView, boundsMin,
      boundsMax, 0.20f));
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
  EXPECT_NEAR(baseFit.texelWorldSize, movedFit.texelWorldSize, 1.0e-6f);
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < 4; ++r) {
      EXPECT_NEAR(baseFit.lightProj[c][r], movedFit.lightProj[c][r], 1.0e-6f);
      EXPECT_NEAR(baseFit.lightViewProj[c][r], movedFit.lightViewProj[c][r],
                  1.0e-6f);
    }
  }
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

TEST(RenderGraphRendererTest,
     DirectionalShadowFitHysteresisKeepsPreviousCenterWithinDeadband) {
  constexpr uint32_t kShadowMapSize = 1024u;
  const CameraFrameState camera = makeSdsmPerspectiveCamera(100.0f);
  const glm::vec3 eye(0.0f, 2.0f, 6.0f);
  const glm::vec3 target(0.0f, 0.5f, 0.0f);
  const glm::vec3 up(0.0f, 1.0f, 0.0f);
  const glm::vec3 lightDirection =
      glm::normalize(glm::vec3(-0.35f, -1.0f, -0.2f));
  const shadow_detail::DirectionalShadowFit baseFit =
      shadow_detail::fitSingleDirectionalShadowMap(camera, lightDirection,
                                                   50.0f, kShadowMapSize, true);

  const glm::vec3 hysteresisMotion =
      glm::vec3(glm::inverse(baseFit.lightView) *
                glm::vec4(baseFit.texelWorldSize * 0.75f, 0.0f, 0.0f, 0.0f));
  CameraFrameState movedCamera = camera;
  movedCamera.view =
      glm::lookAt(eye + hysteresisMotion, target + hysteresisMotion, up);
  movedCamera.cameraPos = glm::vec4(eye + hysteresisMotion, 1.0f);

  shadow_detail::DirectionalShadowFit movedFit =
      shadow_detail::fitSingleDirectionalShadowMap(movedCamera, lightDirection,
                                                   50.0f, kShadowMapSize, true);
  EXPECT_GT(std::abs(movedFit.snappedLightSpaceCenter.x -
                     baseFit.snappedLightSpaceCenter.x),
            baseFit.texelWorldSize * 0.5f);

  shadow_detail::applyDirectionalShadowFitHysteresis(movedFit, baseFit,
                                                     kShadowMapSize);

  EXPECT_NEAR(movedFit.snappedLightSpaceCenter.x,
              baseFit.snappedLightSpaceCenter.x, 1.0e-6f);
  EXPECT_NEAR(movedFit.snappedLightSpaceCenter.y,
              baseFit.snappedLightSpaceCenter.y, 1.0e-6f);
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < 4; ++r) {
      EXPECT_NEAR(movedFit.lightProj[c][r], baseFit.lightProj[c][r], 1.0e-6f);
      EXPECT_NEAR(movedFit.lightViewProj[c][r], baseFit.lightViewProj[c][r],
                  1.0e-6f);
    }
  }
}

TEST(RenderGraphRendererTest,
     DirectionalShadowFitHysteresisReleasesPreviousCenterPastDeadband) {
  constexpr uint32_t kShadowMapSize = 1024u;
  const CameraFrameState camera = makeSdsmPerspectiveCamera(100.0f);
  const glm::vec3 eye(0.0f, 2.0f, 6.0f);
  const glm::vec3 target(0.0f, 0.5f, 0.0f);
  const glm::vec3 up(0.0f, 1.0f, 0.0f);
  const glm::vec3 lightDirection =
      glm::normalize(glm::vec3(-0.35f, -1.0f, -0.2f));
  const shadow_detail::DirectionalShadowFit baseFit =
      shadow_detail::fitSingleDirectionalShadowMap(camera, lightDirection,
                                                   50.0f, kShadowMapSize, true);

  const glm::vec3 releaseMotion =
      glm::vec3(glm::inverse(baseFit.lightView) *
                glm::vec4(baseFit.texelWorldSize * 1.25f, 0.0f, 0.0f, 0.0f));
  CameraFrameState movedCamera = camera;
  movedCamera.view =
      glm::lookAt(eye + releaseMotion, target + releaseMotion, up);
  movedCamera.cameraPos = glm::vec4(eye + releaseMotion, 1.0f);

  shadow_detail::DirectionalShadowFit movedFit =
      shadow_detail::fitSingleDirectionalShadowMap(movedCamera, lightDirection,
                                                   50.0f, kShadowMapSize, true);
  shadow_detail::applyDirectionalShadowFitHysteresis(movedFit, baseFit,
                                                     kShadowMapSize);

  EXPECT_GT(std::abs(movedFit.snappedLightSpaceCenter.x -
                     baseFit.snappedLightSpaceCenter.x),
            baseFit.texelWorldSize * 0.5f);
}

TEST(RenderGraphRendererTest,
     DirectionalShadowFitHysteresisKeepsPreviousExtentWithinDeadband) {
  constexpr uint32_t kShadowMapSize = 1024u;
  const CameraFrameState camera = makeSdsmPerspectiveCamera(100.0f);
  const glm::vec3 lightDirection =
      glm::normalize(glm::vec3(-0.35f, -1.0f, -0.2f));
  const shadow_detail::DirectionalShadowFit baseFit =
      shadow_detail::fitSingleDirectionalShadowMap(camera, lightDirection,
                                                   50.0f, kShadowMapSize, true);
  const glm::vec2 baseExtent = shadow_detail::orthoExtentFromShadowFit(baseFit);

  shadow_detail::DirectionalShadowFit shrunkFit = baseFit;
  const float shrinkAmount = baseFit.texelWorldSize * 1.5f;
  shrunkFit.lightSpaceBoundsMin.x += shrinkAmount * 0.5f;
  shrunkFit.lightSpaceBoundsMax.x -= shrinkAmount * 0.5f;
  shrunkFit.lightSpaceBoundsMin.y += shrinkAmount * 0.5f;
  shrunkFit.lightSpaceBoundsMax.y -= shrinkAmount * 0.5f;
  const glm::vec2 shrunkExtent =
      shadow_detail::orthoExtentFromShadowFit(shrunkFit);
  shrunkFit.texelWorldSize = std::max(shrunkExtent.x, shrunkExtent.y) /
                             static_cast<float>(kShadowMapSize);

  shadow_detail::applyDirectionalShadowFitHysteresis(shrunkFit, baseFit,
                                                     kShadowMapSize);

  const glm::vec2 stabilizedExtent =
      shadow_detail::orthoExtentFromShadowFit(shrunkFit);
  EXPECT_NEAR(stabilizedExtent.x, baseExtent.x, 1.0e-6f);
  EXPECT_NEAR(stabilizedExtent.y, baseExtent.y, 1.0e-6f);
  EXPECT_NEAR(shrunkFit.texelWorldSize, baseFit.texelWorldSize, 1.0e-6f);
}

TEST(RenderGraphRendererTest,
     DirectionalShadowFitHysteresisShrinksExtentAfterDeadband) {
  constexpr uint32_t kShadowMapSize = 1024u;
  const CameraFrameState camera = makeSdsmPerspectiveCamera(100.0f);
  const glm::vec3 lightDirection =
      glm::normalize(glm::vec3(-0.35f, -1.0f, -0.2f));
  const shadow_detail::DirectionalShadowFit baseFit =
      shadow_detail::fitSingleDirectionalShadowMap(camera, lightDirection,
                                                   50.0f, kShadowMapSize, true);

  shadow_detail::DirectionalShadowFit shrunkFit = baseFit;
  const float shrinkAmount = baseFit.texelWorldSize * 3.0f;
  shrunkFit.lightSpaceBoundsMin.x += shrinkAmount * 0.5f;
  shrunkFit.lightSpaceBoundsMax.x -= shrinkAmount * 0.5f;
  shrunkFit.lightSpaceBoundsMin.y += shrinkAmount * 0.5f;
  shrunkFit.lightSpaceBoundsMax.y -= shrinkAmount * 0.5f;
  const glm::vec2 expectedExtent =
      shadow_detail::orthoExtentFromShadowFit(shrunkFit);
  shrunkFit.texelWorldSize = std::max(expectedExtent.x, expectedExtent.y) /
                             static_cast<float>(kShadowMapSize);

  shadow_detail::applyDirectionalShadowFitHysteresis(shrunkFit, baseFit,
                                                     kShadowMapSize);

  const glm::vec2 stabilizedExtent =
      shadow_detail::orthoExtentFromShadowFit(shrunkFit);
  EXPECT_NEAR(stabilizedExtent.x, expectedExtent.x, 1.0e-6f);
  EXPECT_NEAR(stabilizedExtent.y, expectedExtent.y, 1.0e-6f);
}

TEST(RenderGraphRendererTest,
     DirectionalShadowFitHysteresisKeepsPreviousDepthWithinDeadband) {
  constexpr uint32_t kShadowMapSize = 1024u;
  const CameraFrameState camera = makeSdsmPerspectiveCamera(100.0f);
  const glm::vec3 lightDirection =
      glm::normalize(glm::vec3(-0.35f, -1.0f, -0.2f));
  const shadow_detail::DirectionalShadowFit baseFit =
      shadow_detail::fitSingleDirectionalShadowMap(camera, lightDirection,
                                                   50.0f, kShadowMapSize, true);

  shadow_detail::DirectionalShadowFit shrunkFit = baseFit;
  const float shrinkAmount = baseFit.texelWorldSize * 1.5f;
  shrunkFit.lightSpaceBoundsMin.z += shrinkAmount;
  shrunkFit.lightSpaceBoundsMax.z -= shrinkAmount;

  shadow_detail::applyDirectionalShadowFitHysteresis(shrunkFit, baseFit,
                                                     kShadowMapSize);

  EXPECT_EQ(shrunkFit.lightSpaceBoundsMin.z, baseFit.lightSpaceBoundsMin.z);
  EXPECT_EQ(shrunkFit.lightSpaceBoundsMax.z, baseFit.lightSpaceBoundsMax.z);
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < 4; ++r) {
      EXPECT_NEAR(shrunkFit.lightProj[c][r], baseFit.lightProj[c][r], 1.0e-6f);
      EXPECT_NEAR(shrunkFit.lightViewProj[c][r], baseFit.lightViewProj[c][r],
                  1.0e-6f);
    }
  }
}

TEST(RenderGraphRendererTest,
     DirectionalShadowFitHysteresisShrinksDepthAfterDeadband) {
  constexpr uint32_t kShadowMapSize = 1024u;
  const CameraFrameState camera = makeSdsmPerspectiveCamera(100.0f);
  const glm::vec3 lightDirection =
      glm::normalize(glm::vec3(-0.35f, -1.0f, -0.2f));
  const shadow_detail::DirectionalShadowFit baseFit =
      shadow_detail::fitSingleDirectionalShadowMap(camera, lightDirection,
                                                   50.0f, kShadowMapSize, true);

  shadow_detail::DirectionalShadowFit shrunkFit = baseFit;
  const float shrinkAmount = baseFit.texelWorldSize * 3.0f;
  shrunkFit.lightSpaceBoundsMin.z += shrinkAmount;
  shrunkFit.lightSpaceBoundsMax.z -= shrinkAmount;
  const float expectedMinZ = shrunkFit.lightSpaceBoundsMin.z;
  const float expectedMaxZ = shrunkFit.lightSpaceBoundsMax.z;

  shadow_detail::applyDirectionalShadowFitHysteresis(shrunkFit, baseFit,
                                                     kShadowMapSize);

  EXPECT_EQ(shrunkFit.lightSpaceBoundsMin.z, expectedMinZ);
  EXPECT_EQ(shrunkFit.lightSpaceBoundsMax.z, expectedMaxZ);
}

TEST(RenderGraphRendererTest,
     LinearizeDeviceDepthToViewDepthMapsPerspectiveNearAndFar) {
  CameraFrameState camera{};
  camera.projectionType = ProjectionType::Perspective;
  camera.nearPlane = 0.25f;
  camera.farPlane = 200.0f;

  EXPECT_NEAR(shadow_detail::linearizeDeviceDepthToViewDepth(0.0f, camera),
              camera.nearPlane, 1.0e-6f);
  EXPECT_NEAR(shadow_detail::linearizeDeviceDepthToViewDepth(1.0f, camera),
              camera.farPlane, 1.0e-4f);
}

TEST(RenderGraphRendererTest,
     LinearizeDeviceDepthToViewDepthMapsOrthographicNearAndFar) {
  CameraFrameState camera{};
  camera.projectionType = ProjectionType::Orthographic;
  camera.nearPlane = 2.0f;
  camera.farPlane = 18.0f;

  EXPECT_FLOAT_EQ(shadow_detail::linearizeDeviceDepthToViewDepth(0.0f, camera),
                  camera.nearPlane);
  EXPECT_FLOAT_EQ(shadow_detail::linearizeDeviceDepthToViewDepth(1.0f, camera),
                  camera.farPlane);
}

TEST(RenderGraphRendererTest, CascadeSplitDepthsStayMonotonicAcrossModes) {
  CameraFrameState camera{};
  camera.nearPlane = 0.1f;
  camera.farPlane = 100.0f;

  const auto uniformSplits = shadow_detail::computeCascadeSplitDepths(
      camera, 50.0f, 4u, ShadowCascadeSplitMode::Uniform, 0.75f);
  const auto logSplits = shadow_detail::computeCascadeSplitDepths(
      camera, 50.0f, 4u, ShadowCascadeSplitMode::Logarithmic, 0.75f);
  const auto practicalSplits = shadow_detail::computeCascadeSplitDepths(
      camera, 50.0f, 4u, ShadowCascadeSplitMode::Practical, 0.75f);

  for (uint32_t i = 0u; i < 4u; ++i) {
    EXPECT_LT(uniformSplits[i], uniformSplits[i + 1u]);
    EXPECT_LT(logSplits[i], logSplits[i + 1u]);
    EXPECT_LT(practicalSplits[i], practicalSplits[i + 1u]);
  }
  EXPECT_NEAR(uniformSplits[0], 0.1f, 1.0e-6f);
  EXPECT_NEAR(logSplits[0], 0.1f, 1.0e-6f);
  EXPECT_NEAR(practicalSplits[0], 0.1f, 1.0e-6f);
  EXPECT_NEAR(uniformSplits[4], 50.0f, 1.0e-6f);
  EXPECT_NEAR(logSplits[4], 50.0f, 1.0e-6f);
  EXPECT_NEAR(practicalSplits[4], 50.0f, 1.0e-6f);
  EXPECT_NE(uniformSplits[1], practicalSplits[1]);

  const auto rangedSplits = shadow_detail::computeCascadeSplitDepthsForRange(
      2.0f, 20.0f, 4u, ShadowCascadeSplitMode::Practical, 0.5f);
  for (uint32_t i = 0u; i < 4u; ++i) {
    EXPECT_LT(rangedSplits[i], rangedSplits[i + 1u]);
  }
  EXPECT_NEAR(rangedSplits[0], 2.0f, 1.0e-6f);
  EXPECT_NEAR(rangedSplits[4], 20.0f, 1.0e-6f);
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
  settings.shadow.enabled = false;
  settings.shadow.cascadeCount = 1u;
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
     ShadowFeaturePublishesLatestCompletedGpuTimingMetric) {
  std::array<std::byte, 32 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeRendererGPUDevice gpu;
  gpu.latestCompletedGpuTimingReport = GpuTimingReport{
      .sourceFrameIndex = 7u,
      .shadowDepthSourceFrameIndex = 7u,
      .shadowSdsmSourceFrameIndex = 6u,
      .shadowTimeMs = 1.75f,
      .shadowDepthTimeMs = 1.75f,
      .shadowSdsmTimeMs = 0.11f,
      .availableScopeMask = kGpuTimingScopeShadowBit |
                            kGpuTimingScopeShadowDepthBit |
                            kGpuTimingScopeShadowSdsmBit,
  };
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  auto *shadow =
      pipeline.addFeature(std::make_unique<ShadowFeature>(gpu, &memory));
  ASSERT_NE(shadow, nullptr);

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  addDirectionalLightToScene(scene);

  RenderSettings settings{};
  settings.shadow.enabled = true;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 8u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  frameContext.camera = CameraFrameState{
      .projectionType = ProjectionType::Perspective,
      .nearPlane = 0.1f,
      .farPlane = 30.0f,
  };

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);

  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());
  EXPECT_EQ(frameContext.metrics.shadow.gpuTimingAvailable, 1u);
  EXPECT_FLOAT_EQ(frameContext.metrics.shadow.gpuTimeMs, 1.75f);
  EXPECT_EQ(frameContext.metrics.shadow.depthGpuTimingAvailable, 1u);
  EXPECT_FLOAT_EQ(frameContext.metrics.shadow.depthGpuTimeMs, 1.75f);
  EXPECT_EQ(frameContext.metrics.shadow.depthGpuTimingSourceFrameIndex, 7u);
  EXPECT_EQ(frameContext.metrics.shadow.sdsmGpuTimingAvailable, 1u);
  EXPECT_FLOAT_EQ(frameContext.metrics.shadow.sdsmGpuTimeMs, 0.11f);
  EXPECT_EQ(frameContext.metrics.shadow.sdsmGpuTimingSourceFrameIndex, 6u);
  EXPECT_EQ(frameContext.metrics.shadow.gpuTimingSourceFrameIndex, 7u);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureBuildsNoGraphPassesWithoutDirectionalLights) {
  std::array<std::byte, 64 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeRendererGPUDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(std::make_unique<MaterialTableGpuProvider>(gpu));
  pipeline.addProvider(std::make_unique<SceneLightingProvider>(gpu));
  auto *shadow =
      pipeline.addFeature(std::make_unique<ShadowFeature>(gpu, &memory));
  ASSERT_NE(shadow, nullptr);

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();
  ASSERT_TRUE(commitResult.value());

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.cascadeCount = 4u;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 2u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);

  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);

  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  EXPECT_TRUE(buildResult.value());
  EXPECT_EQ(graph.passCount(), 0u);
  ASSERT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());
  EXPECT_EQ(frameContext.sharedResources.shadowDebugFrameData->cascadeCount,
            0u);
  EXPECT_FALSE(frameContext.sharedResources.selectedShadowLightId.has_value());
}

TEST(RenderGraphRendererTest,
     ShadowFeatureAttachesAnimationPreDispatchesToFirstCascadeOnly) {
  std::array<std::byte, 64 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeRendererGPUDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  auto *shadow =
      pipeline.addFeature(std::make_unique<ShadowFeature>(gpu, &memory));
  ASSERT_NE(shadow, nullptr);

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

  auto instanceMatricesResult =
      gpu.createBuffer(BufferDesc{.usage = BufferUsage::Storage,
                                  .storage = Storage::Device,
                                  .size = 256u},
                       "shadow_animation_instances");
  ASSERT_FALSE(instanceMatricesResult.hasError())
      << instanceMatricesResult.error();
  const BufferHandle instanceMatricesBuffer = instanceMatricesResult.value();
  const uint64_t instanceMatricesAddress =
      gpu.getBufferDeviceAddress(instanceMatricesBuffer, 0u);
  ASSERT_NE(instanceMatricesAddress, 0u);

  const std::array<std::byte, 4u> dispatchPushConstants = {
      std::byte{0x11}, std::byte{0x22}, std::byte{0x33}, std::byte{0x44}};
  const std::array<BufferHandle, 1u> dispatchDependencyBuffers = {
      instanceMatricesBuffer};
  ComputeDispatchItem animationDispatch{};
  animationDispatch.pipeline =
      ComputePipelineHandle{.index = 77u, .generation = 1u};
  animationDispatch.dispatch = {.x = 1u, .y = 1u, .z = 1u};
  animationDispatch.pushConstants = std::span<const std::byte>(
      dispatchPushConstants.data(), dispatchPushConstants.size());
  animationDispatch.dependencyBuffers = std::span<const BufferHandle>(
      dispatchDependencyBuffers.data(), dispatchDependencyBuffers.size());
  animationDispatch.debugLabel = "AnimationPose Scatter";
  const std::array<ComputeDispatchItem, 1u> preDispatches = {animationDispatch};

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.cascadeCount = 4u;
  settings.shadow.shadowMapSize = 1024u;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 2u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  frameContext.sharedResources.animationSceneGpuData = AnimationSceneFrameData{
      .instanceMatricesBuffer = instanceMatricesBuffer,
      .instanceMatricesAddress = instanceMatricesAddress,
      .preDispatches = std::span<const ComputeDispatchItem>(
          preDispatches.data(), preDispatches.size()),
      .geometryOverridesByRenderable = {},
      .scene = &scene,
      .sceneTopologyVersion = scene.topologyVersion(),
      .renderableCount = scene.renderables().size(),
      .version = 1u,
  };
  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);

  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);

  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  EXPECT_TRUE(buildResult.value());
  EXPECT_EQ(graph.passCount(), 4u);

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const RenderGraphCompileResult &compiled = compileResult.value();
  ASSERT_EQ(compiled.orderedPasses.size(), 4u);
  ASSERT_EQ(compiled.orderedPasses[0].preDispatches.size(), 1u);
  EXPECT_EQ(compiled.orderedPasses[0].preDispatches[0].debugLabel,
            "AnimationPose Scatter");
  for (size_t passIndex = 1u; passIndex < compiled.orderedPasses.size();
       ++passIndex) {
    EXPECT_TRUE(compiled.orderedPasses[passIndex].preDispatches.empty());
  }
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
  settings.shadow.cascadeCount = 4u;
  settings.shadow.shadowMapSize = 1024u;
  RenderFrameContext frameContext{};
  frameContext.frameIndex = 1u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);

  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);

  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  EXPECT_TRUE(buildResult.value());
  ASSERT_EQ(gpu.createdTextureDescs.size(), 4u);
  for (const TextureDesc &shadowDesc : gpu.createdTextureDescs) {
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
  }

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

  for (uint32_t cascadeIndex = 0u; cascadeIndex < 4u; ++cascadeIndex) {
    EXPECT_TRUE(nuri::isValid(
        frameContext.sharedResources.shadowCascadeTextures[cascadeIndex]));
  }
  EXPECT_EQ(frameContext.sharedResources.shadowRawSamplerId, 1u);
  EXPECT_EQ(frameContext.sharedResources.shadowCompareSamplerId, 2u);
  ASSERT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());
  EXPECT_EQ(frameContext.sharedResources.shadowDebugFrameData->cascadeCount,
            4u);
  EXPECT_EQ(frameContext.sharedResources.shadowDebugFrameData->rawSamplerId,
            1u);
  EXPECT_EQ(frameContext.sharedResources.shadowDebugFrameData->compareSamplerId,
            2u);
  EXPECT_EQ(graph.passCount(), 4u);

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const RenderGraphCompileResult &compiled = compileResult.value();
  ASSERT_EQ(compiled.orderedPasses.size(), 4u);
  for (uint32_t cascadeIndex = 0u; cascadeIndex < 4u; ++cascadeIndex) {
    const RenderPass &shadowPass = compiled.orderedPasses[cascadeIndex];
    EXPECT_FALSE(shadowPass.hasColorAttachment);
    EXPECT_TRUE(sameTexture(
        shadowPass.depthTexture,
        frameContext.sharedResources.shadowCascadeTextures[cascadeIndex]));
    EXPECT_EQ(shadowPass.depth.loadOp, LoadOp::Clear);
    EXPECT_EQ(shadowPass.depth.storeOp, StoreOp::Store);
    EXPECT_EQ(shadowPass.depth.clearStencil, 0u);
    EXPECT_TRUE(nuri::isValid(
        frameContext.sharedResources.shadowCascadeGraphTextures[cascadeIndex]));
  }
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
  settings.shadow.cascadeCount = 4u;
  settings.shadow.shadowMapSize = 1024u;
  settings.shadow.debug.showShadowMapViewport = true;
  settings.shadow.debug.debugCascadeIndex = 2u;
  settings.shadow.debug.previewDepthMin = 0.2f;
  settings.shadow.debug.previewDepthMax = 0.4f;
  settings.shadow.debug.previewDepthInvert = true;
  settings.shadow.debug.previewDepthLog = true;
  RenderFrameContext frameContext{};
  frameContext.frameIndex = 2u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);

  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);

  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  EXPECT_TRUE(buildResult.value());
  ASSERT_EQ(gpu.createdTextureDescs.size(), 5u);
  for (uint32_t cascadeIndex = 0u; cascadeIndex < 4u; ++cascadeIndex) {
    EXPECT_EQ(gpu.createdTextureDescs[cascadeIndex].format, Format::D16_UNORM);
  }
  const TextureDesc &previewDesc = gpu.createdTextureDescs.back();
  EXPECT_EQ(previewDesc.format, Format::RGBA8_UNORM);
  EXPECT_EQ(previewDesc.usage, TextureUsage::AttachmentSampled);
  EXPECT_EQ(previewDesc.dimensions.width, 1024u);
  EXPECT_EQ(previewDesc.dimensions.height, 1024u);
  EXPECT_TRUE(
      nuri::isValid(frameContext.sharedResources.shadowDebugPreviewTexture));
  EXPECT_EQ(graph.passCount(), 5u);

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const RenderGraphCompileResult &compiled = compileResult.value();
  ASSERT_EQ(compiled.orderedPasses.size(), 5u);
  for (uint32_t cascadeIndex = 0u; cascadeIndex < 4u; ++cascadeIndex) {
    EXPECT_FALSE(compiled.orderedPasses[cascadeIndex].hasColorAttachment);
  }
  const RenderPass &previewPass = compiled.orderedPasses.back();
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
  EXPECT_EQ(previewPc.sourceTexIds[0],
            gpu.getTextureBindlessIndex(
                frameContext.sharedResources.shadowCascadeTextures[2]));
  EXPECT_EQ(previewPc.previewParams[0], 1024u);
  EXPECT_EQ(previewPc.previewParams[1], 1024u);
  EXPECT_EQ(previewPc.previewParams[2], 1u);
  EXPECT_NEAR(previewPc.depthParams[0], 5.0f, 1.0e-5f);
  EXPECT_NEAR(previewPc.depthParams[1], -1.0f, 1.0e-5f);
  EXPECT_EQ(previewPc.previewParams[3],
            kShadowPreviewProbeFlagInvert | kShadowPreviewProbeFlagLog);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureBuildsTiledAllCascadeDebugPreviewPassWhenEnabled) {
  std::array<std::byte, 32 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  auto *shadow =
      pipeline.addFeature(std::make_unique<ShadowFeature>(gpu, &memory));
  ASSERT_NE(shadow, nullptr);

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
  settings.shadow.cascadeCount = 4u;
  settings.shadow.shadowMapSize = 1024u;
  settings.shadow.debug.showShadowMapViewport = true;
  settings.shadow.debug.previewMode = ShadowPreviewMode::TiledAllCascades;
  RenderFrameContext frameContext{};
  frameContext.frameIndex = 2u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);

  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);

  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  EXPECT_TRUE(buildResult.value());
  ASSERT_EQ(gpu.createdTextureDescs.size(), 5u);
  const TextureDesc &previewDesc = gpu.createdTextureDescs.back();
  EXPECT_EQ(previewDesc.dimensions.width, 2048u);
  EXPECT_EQ(previewDesc.dimensions.height, 2048u);
  EXPECT_TRUE(
      nuri::isValid(frameContext.sharedResources.shadowDebugPreviewTexture));

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const RenderGraphCompileResult &compiled = compileResult.value();
  const RenderPass &previewPass = compiled.orderedPasses.back();
  ASSERT_FALSE(previewPass.draws.empty());
  const DrawItem &previewDraw = previewPass.draws.front();
  ASSERT_EQ(previewDraw.pushConstants.size(),
            sizeof(ShadowPreviewPushConstantsProbe));
  ShadowPreviewPushConstantsProbe previewPc{};
  std::memcpy(&previewPc, previewDraw.pushConstants.data(), sizeof(previewPc));
  EXPECT_EQ(previewPc.previewParams[0], 2048u);
  EXPECT_EQ(previewPc.previewParams[1], 2048u);
  EXPECT_EQ(previewPc.previewParams[2], 4u);
  EXPECT_EQ(previewPc.previewParams[3], kShadowPreviewProbeFlagTiled);
  for (uint32_t cascadeIndex = 0u; cascadeIndex < 4u; ++cascadeIndex) {
    EXPECT_EQ(
        previewPc.sourceTexIds[cascadeIndex],
        gpu.getTextureBindlessIndex(
            frameContext.sharedResources.shadowCascadeTextures[cascadeIndex]));
  }
}

TEST(RenderGraphRendererTest,
     OpaqueFeatureAutoEnablesDepthPyramidWhenPreviousFrameSdsmIsRequested) {
  std::array<std::byte, 512 * 1024> scratchBytes{};
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
  pipeline.addFeature(std::make_unique<OpaqueFeature>(
      gpu, makeOpaqueConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
      &memory));

  const std::filesystem::path tempDir =
      makeTempRendererPath("sdsm_forced_depth_pyramid_scene");
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
      .debugName = "sdsm_forced_depth_pyramid_triangle",
  });
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();
  auto materialResult = renderer.resources().acquireMaterial(MaterialRequest{
      .debugName = "sdsm_forced_depth_pyramid_material",
  });
  ASSERT_FALSE(materialResult.hasError()) << materialResult.error();

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  auto renderableResult = scene.graph().addRenderable(
      scene.graph().rootNode(), modelResult.value(), materialResult.value());
  ASSERT_FALSE(renderableResult.hasError()) << renderableResult.error();
  LightDesc light{};
  light.type = LightType::Directional;
  light.intensity = 4.0f;
  auto addLightResult = scene.graph().addLight(scene.graph().rootNode(), light);
  ASSERT_FALSE(addLightResult.hasError()) << addLightResult.error();
  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();
  ASSERT_TRUE(commitResult.value());

  RenderSettings settings{};
  settings.opaque.enableDepthPyramid = false;
  settings.opaque.enableInstanceCompute = false;
  settings.opaque.enableIndirectDraw = false;
  settings.shadow.enabled = true;
  settings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 1u;
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
  EXPECT_GT(frameContext.sharedResources.sceneDepthPyramidLevelCount, 0u);
  EXPECT_FALSE(frameContext.sharedResources.sceneDepthPyramidSourceFrameIndex
                   .has_value());

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  bool sawDepthPyramidPass = false;
  for (const RenderPass &pass : compileResult.value().orderedPasses) {
    if (pass.debugLabel == "Opaque Depth MinMax Pyramid") {
      sawDepthPyramidPass = true;
      break;
    }
  }
  EXPECT_TRUE(sawDepthPyramidPass);
  bool createdMinMaxTexture = false;
  for (const TextureDesc &textureDesc : gpu.createdTextureDescs) {
    if (textureDesc.format == Format::RG32_FLOAT) {
      createdMinMaxTexture = true;
      break;
    }
  }
  EXPECT_TRUE(createdMinMaxTexture);
}

TEST(RenderGraphRendererTest,
     OpaqueFeatureAutoEnablesDepthPyramidWhenHistogramSdsmIsRequested) {
  std::array<std::byte, 512 * 1024> scratchBytes{};
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
  pipeline.addFeature(std::make_unique<OpaqueFeature>(
      gpu, makeOpaqueConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
      &memory));

  const std::filesystem::path tempDir =
      makeTempRendererPath("sdsm_histogram_forced_depth_pyramid_scene");
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
      .debugName = "sdsm_histogram_forced_depth_pyramid_triangle",
  });
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();
  auto materialResult = renderer.resources().acquireMaterial(MaterialRequest{
      .debugName = "sdsm_histogram_forced_depth_pyramid_material",
  });
  ASSERT_FALSE(materialResult.hasError()) << materialResult.error();

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  auto renderableResult = scene.graph().addRenderable(
      scene.graph().rootNode(), modelResult.value(), materialResult.value());
  ASSERT_FALSE(renderableResult.hasError()) << renderableResult.error();
  LightDesc light{};
  light.type = LightType::Directional;
  light.intensity = 4.0f;
  auto addLightResult = scene.graph().addLight(scene.graph().rootNode(), light);
  ASSERT_FALSE(addLightResult.hasError()) << addLightResult.error();
  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();
  ASSERT_TRUE(commitResult.value());

  RenderSettings settings{};
  settings.opaque.enableDepthPyramid = false;
  settings.opaque.enableInstanceCompute = false;
  settings.opaque.enableIndirectDraw = false;
  settings.shadow.enabled = true;
  settings.shadow.sdsmMode = ShadowSdsmMode::Histogram;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 1u;
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
  EXPECT_GT(frameContext.sharedResources.sceneDepthPyramidLevelCount, 0u);
  EXPECT_FALSE(frameContext.sharedResources.sceneDepthPyramidSourceFrameIndex
                   .has_value());

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  bool sawDepthPyramidPass = false;
  for (const RenderPass &pass : compileResult.value().orderedPasses) {
    if (pass.debugLabel == "Opaque Depth MinMax Pyramid") {
      sawDepthPyramidPass = true;
      break;
    }
  }
  EXPECT_TRUE(sawDepthPyramidPass);
  bool createdMinMaxTexture = false;
  for (const TextureDesc &textureDesc : gpu.createdTextureDescs) {
    if (textureDesc.format == Format::RG32_FLOAT) {
      createdMinMaxTexture = true;
      break;
    }
  }
  EXPECT_TRUE(createdMinMaxTexture);
}

TEST(RenderGraphRendererTest,
     OpaqueFeatureClearsPublishedDepthPyramidSourceAfterPyramidRecreation) {
  std::array<std::byte, 512 * 1024> scratchBytes{};
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
  pipeline.addFeature(std::make_unique<OpaqueFeature>(
      gpu, makeOpaqueConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
      &memory));

  const std::filesystem::path tempDir =
      makeTempRendererPath("sdsm_depth_pyramid_recreate_scene");
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
      .debugName = "sdsm_depth_pyramid_recreate_triangle",
  });
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();
  auto materialResult = renderer.resources().acquireMaterial(MaterialRequest{
      .debugName = "sdsm_depth_pyramid_recreate_material",
  });
  ASSERT_FALSE(materialResult.hasError()) << materialResult.error();

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  auto renderableResult = scene.graph().addRenderable(
      scene.graph().rootNode(), modelResult.value(), materialResult.value());
  ASSERT_FALSE(renderableResult.hasError()) << renderableResult.error();
  LightDesc light{};
  light.type = LightType::Directional;
  light.intensity = 4.0f;
  auto addLightResult = scene.graph().addLight(scene.graph().rootNode(), light);
  ASSERT_FALSE(addLightResult.hasError()) << addLightResult.error();
  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();
  ASSERT_TRUE(commitResult.value());

  RenderSettings settings{};
  settings.opaque.enableDepthPyramid = false;
  settings.opaque.enableInstanceCompute = false;
  settings.opaque.enableIndirectDraw = false;
  settings.shadow.enabled = true;
  settings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;

  const auto buildFrame = [&](uint64_t frameIndex) {
    RenderFrameContext frameContext{};
    frameContext.frameIndex = frameIndex;
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
    EXPECT_FALSE(buildResult.hasError()) << buildResult.error();
    EXPECT_TRUE(buildResult.value());
    return frameContext;
  };

  const RenderFrameContext firstFrame = buildFrame(1u);
  ASSERT_GT(firstFrame.sharedResources.sceneDepthPyramidLevelCount, 0u);
  EXPECT_FALSE(
      firstFrame.sharedResources.sceneDepthPyramidSourceFrameIndex.has_value());

  for (uint32_t level = 0u;
       level < firstFrame.sharedResources.sceneDepthPyramidLevelCount;
       ++level) {
    const TextureHandle texture =
        firstFrame.sharedResources.sceneDepthPyramidTextures[level];
    ASSERT_TRUE(nuri::isValid(texture));
    gpu.destroyTexture(texture);
  }

  const RenderFrameContext secondFrame = buildFrame(2u);
  EXPECT_GT(secondFrame.sharedResources.sceneDepthPyramidLevelCount, 0u);
  EXPECT_FALSE(secondFrame.sharedResources.sceneDepthPyramidSourceFrameIndex
                   .has_value());
}

TEST(RenderGraphRendererTest,
     OpaqueFeatureAppendsShadowSdsmReducePassForGpuMinMaxAndPublishesMetrics) {
  std::array<std::byte, 512 * 1024> scratchBytes{};
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
      gpu, makeShadowConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
      &memory));
  pipeline.addFeature(std::make_unique<OpaqueFeature>(
      gpu, makeOpaqueConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
      &memory));

  const std::filesystem::path tempDir =
      makeTempRendererPath("sdsm_gpu_reduce_scene");
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
      .debugName = "sdsm_gpu_reduce_triangle",
  });
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();
  auto materialResult = renderer.resources().acquireMaterial(MaterialRequest{
      .debugName = "sdsm_gpu_reduce_material",
  });
  ASSERT_FALSE(materialResult.hasError()) << materialResult.error();

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  auto renderableResult = scene.graph().addRenderable(
      scene.graph().rootNode(), modelResult.value(), materialResult.value());
  ASSERT_FALSE(renderableResult.hasError()) << renderableResult.error();
  addDirectionalLightToScene(scene);

  RenderSettings settings{};
  settings.opaque.enableDepthPyramid = false;
  settings.opaque.enableInstanceCompute = false;
  settings.opaque.enableIndirectDraw = false;
  settings.shadow.enabled = true;
  settings.shadow.filterMode = ShadowFilterMode::PCSS;
  settings.shadow.pcssBlockerSampleCount = 19u;
  settings.shadow.pcssFilterSampleCount = 41u;
  settings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
  settings.shadow.sdsmReductionBackend = ShadowSdsmReductionBackend::Gpu;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 1u;
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
  EXPECT_TRUE(
      frameContext.sharedResources.shadowSdsmGpuReduceTarget.has_value());
  EXPECT_TRUE(
      nuri::isValid(frameContext.sharedResources.shadowSdsmGpuReducePipeline));
  EXPECT_EQ(frameContext.metrics.shadow.sdsmComputePassCount, 1u);
  EXPECT_GT(frameContext.metrics.shadow.cascadeTextureBytes, 0u);
  EXPECT_GT(frameContext.metrics.shadow.totalDraws, 0u);
  EXPECT_EQ(frameContext.metrics.shadow.shadowMapSize,
            settings.shadow.shadowMapSize);
  EXPECT_EQ(frameContext.metrics.shadow.filterSampleBudget, 60u);
  EXPECT_EQ(frameContext.metrics.shadow.pcssBlockerSampleBudget, 19u);
  EXPECT_EQ(frameContext.metrics.shadow.pcssFilterSampleBudget, 41u);

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const RenderPass *reducePass = nullptr;
  for (const RenderPass &pass : compileResult.value().orderedPasses) {
    if (pass.debugLabel == "Shadow SDSM Reduce") {
      reducePass = &pass;
      break;
    }
  }
  ASSERT_NE(reducePass, nullptr);
  EXPECT_EQ(reducePass->executionMode, RenderPassExecutionMode::ComputeOnly);
  EXPECT_TRUE(reducePass->draws.empty());
  ASSERT_EQ(reducePass->preDispatches.size(), 1u);
  EXPECT_EQ(reducePass->preDispatches[0].debugLabel, "Shadow SDSM Reduce");
  ASSERT_EQ(reducePass->preDispatches[0].dependencyBuffers.size(), 1u);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureHistogramModeAlwaysUsesCpuReductionBackend) {
  std::array<std::byte, 64 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addFeature(std::make_unique<ShadowFeature>(gpu, &memory));

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  addDirectionalLightToScene(scene);

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 1u, 1u),
                        "sdsm_histogram_force_cpu");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.sdsmMode = ShadowSdsmMode::Histogram;
  settings.shadow.sdsmReductionBackend = ShadowSdsmReductionBackend::Gpu;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 1u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  frameContext.camera = CameraFrameState{
      .projectionType = ProjectionType::Perspective,
      .nearPlane = 0.1f,
      .farPlane = 30.0f,
  };
  frameContext.sharedResources.sceneDepthPyramidTextures[0] =
      pyramidTextureResult.value();
  frameContext.sharedResources.sceneDepthPyramidLevelCount = 1u;
  frameContext.sharedResources.sceneDepthPyramidSourceFrameIndex = 0u;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());
  ASSERT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());
  EXPECT_EQ(frameContext.sharedResources.shadowDebugFrameData->sdsm
                .requestedReductionBackend,
            ShadowSdsmReductionBackend::Gpu);
  EXPECT_EQ(frameContext.sharedResources.shadowDebugFrameData->sdsm
                .activeReductionBackend,
            ShadowSdsmReductionBackend::Cpu);
  EXPECT_FALSE(frameContext.sharedResources.shadowDebugFrameData->sdsm
                   .reductionFallbackActive);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureAutoFallsBackToCpuWhenGpuReductionIsUnavailable) {
  std::array<std::byte, 64 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeNoComputeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addFeature(std::make_unique<ShadowFeature>(gpu, &memory));

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  addDirectionalLightToScene(scene);

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 1u, 1u),
                        "sdsm_auto_cpu_fallback");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
  settings.shadow.sdsmReductionBackend = ShadowSdsmReductionBackend::Auto;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 1u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  frameContext.camera = CameraFrameState{
      .projectionType = ProjectionType::Perspective,
      .nearPlane = 0.1f,
      .farPlane = 30.0f,
  };
  frameContext.sharedResources.sceneDepthPyramidTextures[0] =
      pyramidTextureResult.value();
  frameContext.sharedResources.sceneDepthPyramidLevelCount = 1u;
  frameContext.sharedResources.sceneDepthPyramidSourceFrameIndex = 0u;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());
  ASSERT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());
  EXPECT_FALSE(
      frameContext.sharedResources.shadowSdsmGpuReduceTarget.has_value());
  EXPECT_EQ(frameContext.sharedResources.shadowDebugFrameData->sdsm
                .requestedReductionBackend,
            ShadowSdsmReductionBackend::Auto);
  EXPECT_EQ(frameContext.sharedResources.shadowDebugFrameData->sdsm
                .activeReductionBackend,
            ShadowSdsmReductionBackend::Cpu);
  EXPECT_FALSE(frameContext.sharedResources.shadowDebugFrameData->sdsm
                   .reductionFallbackActive);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureGpuSdsmConsumesPreviousFrameBufferAndMatchesCpuPath) {
  struct GpuSdsmMinMaxResult {
    glm::vec2 rawDeviceMinMax{1.0f, 1.0f};
    uint32_t sourceFrameIndex = 0u;
    uint32_t valid = 0u;
  };

  const CameraFrameState camera{
      .projectionType = ProjectionType::Perspective,
      .nearPlane = 0.1f,
      .farPlane = 30.0f,
  };

  const auto buildSdsm =
      [&](ShadowSdsmReductionBackend backend) -> ShadowSdsmDebugFrameData {
    std::array<std::byte, 64 * 1024> scratchBytes{};
    std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                               scratchBytes.size());
    FakeFullscreenGpuDevice gpu;
    Renderer renderer(gpu, memory);
    RenderPipeline pipeline(&memory);
    pipeline.addFeature(std::make_unique<ShadowFeature>(
        gpu, makeShadowConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
        &memory));

    RenderScene scene(&memory);
    scene.bindResources(&renderer.resources());
    addDirectionalLightToScene(scene);

    auto pyramidTextureResult =
        gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 1u, 1u),
                          "sdsm_compare_texture");
    EXPECT_FALSE(pyramidTextureResult.hasError())
        << pyramidTextureResult.error();
    if (pyramidTextureResult.hasError()) {
      return ShadowSdsmDebugFrameData{};
    }

    if (backend == ShadowSdsmReductionBackend::Cpu) {
      const std::array<float, 2u> deviceDepths{0.2f, 0.5f};
      auto seedResult =
          gpu.seedTextureBytes(pyramidTextureResult.value(),
                               std::as_bytes(std::span<const float>(
                                   deviceDepths.data(), deviceDepths.size())));
      EXPECT_FALSE(seedResult.hasError()) << seedResult.error();
    } else {
      RenderSettings publishSettings{};
      publishSettings.shadow.enabled = true;
      publishSettings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
      publishSettings.shadow.sdsmReductionBackend = backend;

      RenderFrameContext publishFrame{};
      publishFrame.frameIndex = 0u;
      publishFrame.scene = &scene;
      publishFrame.resources = &renderer.resources();
      publishFrame.settings = &publishSettings;
      publishFrame.camera = camera;

      RenderGraphBuilder publishGraph(&memory);
      publishGraph.beginFrame(publishFrame.frameIndex);
      auto publishResult = pipeline.buildRenderGraph(
          publishFrame, renderer.resources(), publishGraph);
      EXPECT_FALSE(publishResult.hasError()) << publishResult.error();
      EXPECT_TRUE(publishResult.value());
      EXPECT_TRUE(
          publishFrame.sharedResources.shadowSdsmGpuReduceTarget.has_value());
      if (!publishFrame.sharedResources.shadowSdsmGpuReduceTarget.has_value()) {
        return ShadowSdsmDebugFrameData{};
      }

      const GpuSdsmMinMaxResult gpuResult{
          .rawDeviceMinMax = glm::vec2(0.2f, 0.5f),
          .sourceFrameIndex = 0u,
          .valid = 1u,
      };
      auto writeResult = gpu.updateBuffer(
          publishFrame.sharedResources.shadowSdsmGpuReduceTarget->buffer,
          std::as_bytes(std::span<const GpuSdsmMinMaxResult>(&gpuResult, 1u)),
          0u);
      EXPECT_FALSE(writeResult.hasError()) << writeResult.error();
    }

    RenderSettings settings{};
    settings.shadow.enabled = true;
    settings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
    settings.shadow.sdsmReductionBackend = backend;
    settings.shadow.sdsmTemporalBlend = 0.0f;

    RenderFrameContext frameContext{};
    frameContext.frameIndex = 1u;
    frameContext.scene = &scene;
    frameContext.resources = &renderer.resources();
    frameContext.settings = &settings;
    frameContext.camera = camera;
    frameContext.sharedResources.sceneDepthPyramidTextures[0] =
        pyramidTextureResult.value();
    frameContext.sharedResources.sceneDepthPyramidLevelCount = 1u;
    frameContext.sharedResources.sceneDepthPyramidSourceFrameIndex = 0u;

    RenderGraphBuilder graph(&memory);
    graph.beginFrame(frameContext.frameIndex);
    auto buildResult =
        pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
    EXPECT_FALSE(buildResult.hasError()) << buildResult.error();
    EXPECT_TRUE(buildResult.value());
    EXPECT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());
    if (!frameContext.sharedResources.shadowDebugFrameData.has_value()) {
      return ShadowSdsmDebugFrameData{};
    }
    return frameContext.sharedResources.shadowDebugFrameData->sdsm;
  };

  const ShadowSdsmDebugFrameData gpuSdsm =
      buildSdsm(ShadowSdsmReductionBackend::Gpu);
  const ShadowSdsmDebugFrameData cpuSdsm =
      buildSdsm(ShadowSdsmReductionBackend::Cpu);

  ASSERT_EQ(gpuSdsm.status, ShadowSdsmStatus::Active);
  ASSERT_EQ(cpuSdsm.status, ShadowSdsmStatus::Active);
  EXPECT_EQ(gpuSdsm.activeReductionBackend, ShadowSdsmReductionBackend::Gpu);
  EXPECT_EQ(cpuSdsm.activeReductionBackend, ShadowSdsmReductionBackend::Cpu);
  EXPECT_NEAR(gpuSdsm.rawLinearMin, cpuSdsm.rawLinearMin, 1.0e-5f);
  EXPECT_NEAR(gpuSdsm.rawLinearMax, cpuSdsm.rawLinearMax, 1.0e-5f);
  EXPECT_NEAR(gpuSdsm.effectiveRangeNear, cpuSdsm.effectiveRangeNear, 1.0e-5f);
  EXPECT_NEAR(gpuSdsm.effectiveRangeFar, cpuSdsm.effectiveRangeFar, 1.0e-5f);
  EXPECT_EQ(gpuSdsm.effectiveSplitDepths, cpuSdsm.effectiveSplitDepths);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureGpuSdsmUsesNewestCompletedRingResultWhenLatestSlotIsInvalid) {
  struct GpuSdsmMinMaxResult {
    glm::vec2 rawDeviceMinMax{1.0f, 1.0f};
    uint32_t sourceFrameIndex = 0u;
    uint32_t valid = 0u;
  };

  std::array<std::byte, 64 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addFeature(std::make_unique<ShadowFeature>(
      gpu, makeShadowConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
      &memory));

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  addDirectionalLightToScene(scene);

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 1u, 1u),
                        "sdsm_delayed_gpu_result");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();

  const CameraFrameState camera{
      .projectionType = ProjectionType::Perspective,
      .nearPlane = 0.1f,
      .farPlane = 30.0f,
  };

  RenderSettings publishSettings{};
  publishSettings.shadow.enabled = true;
  publishSettings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
  publishSettings.shadow.sdsmReductionBackend = ShadowSdsmReductionBackend::Gpu;

  for (uint64_t publishFrameIndex = 0u; publishFrameIndex < 2u;
       ++publishFrameIndex) {
    RenderFrameContext publishFrame{};
    publishFrame.frameIndex = publishFrameIndex;
    publishFrame.scene = &scene;
    publishFrame.resources = &renderer.resources();
    publishFrame.settings = &publishSettings;
    publishFrame.camera = camera;

    RenderGraphBuilder publishGraph(&memory);
    publishGraph.beginFrame(publishFrame.frameIndex);
    auto publishResult = pipeline.buildRenderGraph(
        publishFrame, renderer.resources(), publishGraph);
    ASSERT_FALSE(publishResult.hasError()) << publishResult.error();
    ASSERT_TRUE(publishResult.value());
    ASSERT_TRUE(
        publishFrame.sharedResources.shadowSdsmGpuReduceTarget.has_value());

    if (publishFrameIndex == 0u) {
      const GpuSdsmMinMaxResult gpuResult{
          .rawDeviceMinMax = glm::vec2(0.2f, 0.5f),
          .sourceFrameIndex = 0u,
          .valid = 1u,
      };
      auto writeResult = gpu.updateBuffer(
          publishFrame.sharedResources.shadowSdsmGpuReduceTarget->buffer,
          std::as_bytes(std::span<const GpuSdsmMinMaxResult>(&gpuResult, 1u)),
          0u);
      ASSERT_FALSE(writeResult.hasError()) << writeResult.error();
    }
  }

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
  settings.shadow.sdsmReductionBackend = ShadowSdsmReductionBackend::Gpu;
  settings.shadow.sdsmTemporalBlend = 0.0f;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 2u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  frameContext.camera = camera;
  frameContext.sharedResources.sceneDepthPyramidTextures[0] =
      pyramidTextureResult.value();
  frameContext.sharedResources.sceneDepthPyramidLevelCount = 1u;
  frameContext.sharedResources.sceneDepthPyramidSourceFrameIndex = 1u;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());
  ASSERT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());

  const ShadowSdsmDebugFrameData &sdsm =
      frameContext.sharedResources.shadowDebugFrameData->sdsm;
  EXPECT_EQ(sdsm.status, ShadowSdsmStatus::Active);
  EXPECT_EQ(sdsm.activeReductionBackend, ShadowSdsmReductionBackend::Gpu);
  EXPECT_FALSE(sdsm.fixedFallbackActive);
  EXPECT_FALSE(sdsm.reductionFallbackActive);
  EXPECT_EQ(sdsm.sourceFrameIndex, 0u);
  EXPECT_FLOAT_EQ(sdsm.rawDeviceMin, 0.2f);
  EXPECT_FLOAT_EQ(sdsm.rawDeviceMax, 0.5f);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureGpuSdsmMarksMissingCompletedResultAsStale) {
  std::array<std::byte, 64 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addFeature(std::make_unique<ShadowFeature>(
      gpu, makeShadowConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
      &memory));

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  addDirectionalLightToScene(scene);

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 1u, 1u),
                        "sdsm_missing_gpu_result");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();

  const CameraFrameState camera{
      .projectionType = ProjectionType::Perspective,
      .nearPlane = 0.1f,
      .farPlane = 30.0f,
  };

  RenderSettings publishSettings{};
  publishSettings.shadow.enabled = true;
  publishSettings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
  publishSettings.shadow.sdsmReductionBackend = ShadowSdsmReductionBackend::Gpu;

  RenderFrameContext publishFrame{};
  publishFrame.frameIndex = 0u;
  publishFrame.scene = &scene;
  publishFrame.resources = &renderer.resources();
  publishFrame.settings = &publishSettings;
  publishFrame.camera = camera;

  RenderGraphBuilder publishGraph(&memory);
  publishGraph.beginFrame(publishFrame.frameIndex);
  auto publishResult = pipeline.buildRenderGraph(
      publishFrame, renderer.resources(), publishGraph);
  ASSERT_FALSE(publishResult.hasError()) << publishResult.error();
  ASSERT_TRUE(publishResult.value());
  ASSERT_TRUE(
      publishFrame.sharedResources.shadowSdsmGpuReduceTarget.has_value());

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
  settings.shadow.sdsmReductionBackend = ShadowSdsmReductionBackend::Gpu;
  settings.shadow.sdsmTemporalBlend = 0.0f;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 1u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  frameContext.camera = camera;
  frameContext.sharedResources.sceneDepthPyramidTextures[0] =
      pyramidTextureResult.value();
  frameContext.sharedResources.sceneDepthPyramidLevelCount = 1u;
  frameContext.sharedResources.sceneDepthPyramidSourceFrameIndex = 0u;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());
  ASSERT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());

  const ShadowSdsmDebugFrameData &sdsm =
      frameContext.sharedResources.shadowDebugFrameData->sdsm;
  EXPECT_EQ(sdsm.status, ShadowSdsmStatus::Stale);
  EXPECT_EQ(sdsm.activeReductionBackend, ShadowSdsmReductionBackend::Gpu);
  EXPECT_TRUE(sdsm.fixedFallbackActive);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureGpuSdsmSuppressesWarmupWarningForFirstMiss) {
  std::array<std::byte, 64 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addFeature(std::make_unique<ShadowFeature>(
      gpu, makeShadowConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
      &memory));

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  addDirectionalLightToScene(scene);

  auto pyramidTextureResult = gpu.createTexture(
      makeTransientTextureDesc(Format::RG32_FLOAT, 1u, 1u), "sdsm_warmup_miss");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();

  const CameraFrameState camera{
      .projectionType = ProjectionType::Perspective,
      .nearPlane = 0.1f,
      .farPlane = 30.0f,
  };

  RenderSettings publishSettings{};
  publishSettings.shadow.enabled = true;
  publishSettings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
  publishSettings.shadow.sdsmReductionBackend = ShadowSdsmReductionBackend::Gpu;

  RenderFrameContext publishFrame{};
  publishFrame.frameIndex = 0u;
  publishFrame.scene = &scene;
  publishFrame.resources = &renderer.resources();
  publishFrame.settings = &publishSettings;
  publishFrame.camera = camera;

  RenderGraphBuilder publishGraph(&memory);
  publishGraph.beginFrame(publishFrame.frameIndex);
  auto publishResult = pipeline.buildRenderGraph(
      publishFrame, renderer.resources(), publishGraph);
  ASSERT_FALSE(publishResult.hasError()) << publishResult.error();
  ASSERT_TRUE(publishResult.value());

  const uint64_t baselineSequence = currentLogSequence();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
  settings.shadow.sdsmReductionBackend = ShadowSdsmReductionBackend::Gpu;
  settings.shadow.sdsmTemporalBlend = 0.0f;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 1u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  frameContext.camera = camera;
  frameContext.sharedResources.sceneDepthPyramidTextures[0] =
      pyramidTextureResult.value();
  frameContext.sharedResources.sceneDepthPyramidLevelCount = 1u;
  frameContext.sharedResources.sceneDepthPyramidSourceFrameIndex = 0u;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());
  ASSERT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());

  const ShadowSdsmDebugFrameData &sdsm =
      frameContext.sharedResources.shadowDebugFrameData->sdsm;
  EXPECT_EQ(sdsm.status, ShadowSdsmStatus::Stale);
  EXPECT_EQ(sdsm.activeReductionBackend, ShadowSdsmReductionBackend::Gpu);
  EXPECT_TRUE(sdsm.fixedFallbackActive);
  EXPECT_FALSE(sdsm.reductionFallbackActive);
  EXPECT_EQ(countLogEntriesSince(baselineSequence, "GPU SDSM ring diagnostics"),
            0u);
  EXPECT_EQ(countLogEntriesSince(baselineSequence, "SDSM source warning"), 0u);
  EXPECT_EQ(countLogEntriesSince(
                baselineSequence,
                "requested GPU SDSM reduction fell back to CPU path"),
            0u);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureGpuSdsmLogsPersistentMissAfterWarmupGrace) {
  std::array<std::byte, 64 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addFeature(std::make_unique<ShadowFeature>(
      gpu, makeShadowConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
      &memory));

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  addDirectionalLightToScene(scene);

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 1u, 1u),
                        "sdsm_persistent_miss");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();

  const CameraFrameState camera{
      .projectionType = ProjectionType::Perspective,
      .nearPlane = 0.1f,
      .farPlane = 30.0f,
  };

  RenderSettings publishSettings{};
  publishSettings.shadow.enabled = true;
  publishSettings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
  publishSettings.shadow.sdsmReductionBackend = ShadowSdsmReductionBackend::Gpu;

  RenderFrameContext publishFrame{};
  publishFrame.frameIndex = 0u;
  publishFrame.scene = &scene;
  publishFrame.resources = &renderer.resources();
  publishFrame.settings = &publishSettings;
  publishFrame.camera = camera;

  RenderGraphBuilder publishGraph(&memory);
  publishGraph.beginFrame(publishFrame.frameIndex);
  auto publishResult = pipeline.buildRenderGraph(
      publishFrame, renderer.resources(), publishGraph);
  ASSERT_FALSE(publishResult.hasError()) << publishResult.error();
  ASSERT_TRUE(publishResult.value());

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
  settings.shadow.sdsmReductionBackend = ShadowSdsmReductionBackend::Gpu;
  settings.shadow.sdsmTemporalBlend = 0.0f;

  const uint64_t baselineSequence = currentLogSequence();
  for (uint64_t frameIndex = 1u; frameIndex <= 2u; ++frameIndex) {
    RenderFrameContext frameContext{};
    frameContext.frameIndex = frameIndex;
    frameContext.scene = &scene;
    frameContext.resources = &renderer.resources();
    frameContext.settings = &settings;
    frameContext.camera = camera;
    frameContext.sharedResources.sceneDepthPyramidTextures[0] =
        pyramidTextureResult.value();
    frameContext.sharedResources.sceneDepthPyramidLevelCount = 1u;
    frameContext.sharedResources.sceneDepthPyramidSourceFrameIndex =
        frameIndex - 1u;

    RenderGraphBuilder graph(&memory);
    graph.beginFrame(frameContext.frameIndex);
    auto buildResult =
        pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
    ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
    ASSERT_TRUE(buildResult.value());
  }

  EXPECT_EQ(countLogEntriesSince(baselineSequence, "GPU SDSM ring diagnostics"),
            0u);
  EXPECT_EQ(countLogEntriesSince(baselineSequence, "SDSM source warning"), 0u);
  EXPECT_EQ(countLogEntriesSince(
                baselineSequence,
                "requested GPU SDSM reduction fell back to CPU path"),
            0u);

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 3u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  frameContext.camera = camera;
  frameContext.sharedResources.sceneDepthPyramidTextures[0] =
      pyramidTextureResult.value();
  frameContext.sharedResources.sceneDepthPyramidLevelCount = 1u;
  frameContext.sharedResources.sceneDepthPyramidSourceFrameIndex = 2u;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());

  EXPECT_EQ(countLogEntriesSince(baselineSequence, "GPU SDSM ring diagnostics"),
            1u);
  EXPECT_EQ(countLogEntriesSince(baselineSequence, "SDSM source warning"), 1u);
  EXPECT_EQ(countLogEntriesSince(
                baselineSequence,
                "requested GPU SDSM reduction fell back to CPU path"),
            1u);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureGpuSdsmUsesDeepResultRingBeyondSwapchainDepth) {
  std::array<std::byte, 512 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  gpu.swapchainImageCount = 1u;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addFeature(std::make_unique<ShadowFeature>(
      gpu, makeShadowConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
      &memory));

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  addDirectionalLightToScene(scene);

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
  settings.shadow.sdsmReductionBackend = ShadowSdsmReductionBackend::Gpu;

  const CameraFrameState camera{
      .projectionType = ProjectionType::Perspective,
      .nearPlane = 0.1f,
      .farPlane = 30.0f,
  };

  std::vector<BufferHandle> publishedTargets;
  publishedTargets.reserve(16u);
  for (uint64_t frameIndex = 0u; frameIndex < 16u; ++frameIndex) {
    RenderFrameContext frameContext{};
    frameContext.frameIndex = frameIndex;
    frameContext.scene = &scene;
    frameContext.resources = &renderer.resources();
    frameContext.settings = &settings;
    frameContext.camera = camera;

    RenderGraphBuilder graph(&memory);
    graph.beginFrame(frameContext.frameIndex);
    auto buildResult =
        pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
    ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
    ASSERT_TRUE(buildResult.value());
    ASSERT_TRUE(
        frameContext.sharedResources.shadowSdsmGpuReduceTarget.has_value());

    const BufferHandle targetBuffer =
        frameContext.sharedResources.shadowSdsmGpuReduceTarget->buffer;
    ASSERT_TRUE(nuri::isValid(targetBuffer));
    for (const BufferHandle publishedBuffer : publishedTargets) {
      EXPECT_FALSE(sameBuffer(targetBuffer, publishedBuffer));
    }
    publishedTargets.push_back(targetBuffer);
  }
}

TEST(RenderGraphRendererTest,
     ShadowFeatureGpuSdsmReadsPublishedResultBeyondSwapchainDepth) {
  struct GpuSdsmMinMaxResult {
    glm::vec2 rawDeviceMinMax{1.0f, 1.0f};
    uint32_t sourceFrameIndex = 0u;
    uint32_t valid = 0u;
  };

  std::array<std::byte, 512 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  gpu.swapchainImageCount = 1u;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addFeature(std::make_unique<ShadowFeature>(
      gpu, makeShadowConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
      &memory));

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  addDirectionalLightToScene(scene);

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 1u, 1u),
                        "sdsm_deep_ring_pyramid");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();

  const CameraFrameState camera{
      .projectionType = ProjectionType::Perspective,
      .nearPlane = 0.1f,
      .farPlane = 30.0f,
  };

  RenderSettings publishSettings{};
  publishSettings.shadow.enabled = true;
  publishSettings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
  publishSettings.shadow.sdsmReductionBackend = ShadowSdsmReductionBackend::Gpu;

  for (uint64_t publishFrameIndex = 0u; publishFrameIndex <= 10u;
       ++publishFrameIndex) {
    RenderFrameContext publishFrame{};
    publishFrame.frameIndex = publishFrameIndex;
    publishFrame.scene = &scene;
    publishFrame.resources = &renderer.resources();
    publishFrame.settings = &publishSettings;
    publishFrame.camera = camera;

    RenderGraphBuilder publishGraph(&memory);
    publishGraph.beginFrame(publishFrame.frameIndex);
    auto publishResult = pipeline.buildRenderGraph(
        publishFrame, renderer.resources(), publishGraph);
    ASSERT_FALSE(publishResult.hasError()) << publishResult.error();
    ASSERT_TRUE(publishResult.value());
    ASSERT_TRUE(
        publishFrame.sharedResources.shadowSdsmGpuReduceTarget.has_value());

    if (publishFrameIndex == 10u) {
      const GpuSdsmMinMaxResult gpuResult{
          .rawDeviceMinMax = glm::vec2(0.2f, 0.5f),
          .sourceFrameIndex = 10u,
          .valid = 1u,
      };
      auto writeResult = gpu.updateBuffer(
          publishFrame.sharedResources.shadowSdsmGpuReduceTarget->buffer,
          std::as_bytes(std::span<const GpuSdsmMinMaxResult>(&gpuResult, 1u)),
          0u);
      ASSERT_FALSE(writeResult.hasError()) << writeResult.error();
    }
  }

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
  settings.shadow.sdsmReductionBackend = ShadowSdsmReductionBackend::Gpu;
  settings.shadow.sdsmTemporalBlend = 0.0f;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 11u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  frameContext.camera = camera;
  frameContext.sharedResources.sceneDepthPyramidTextures[0] =
      pyramidTextureResult.value();
  frameContext.sharedResources.sceneDepthPyramidLevelCount = 1u;
  frameContext.sharedResources.sceneDepthPyramidSourceFrameIndex = 10u;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());
  ASSERT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());

  const ShadowSdsmDebugFrameData &sdsm =
      frameContext.sharedResources.shadowDebugFrameData->sdsm;
  EXPECT_EQ(sdsm.status, ShadowSdsmStatus::Active);
  EXPECT_EQ(sdsm.activeReductionBackend, ShadowSdsmReductionBackend::Gpu);
  EXPECT_FALSE(sdsm.fixedFallbackActive);
  EXPECT_FALSE(sdsm.reductionFallbackActive);
  EXPECT_EQ(sdsm.sourceFrameIndex, 10u);
  EXPECT_FLOAT_EQ(sdsm.rawDeviceMin, 0.2f);
  EXPECT_FLOAT_EQ(sdsm.rawDeviceMax, 0.5f);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureGpuSdsmIgnoresRingResultWithMismatchedSourceFrameIndex) {
  struct GpuSdsmMinMaxResult {
    glm::vec2 rawDeviceMinMax{1.0f, 1.0f};
    uint32_t sourceFrameIndex = 0u;
    uint32_t valid = 0u;
  };

  std::array<std::byte, 64 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addFeature(std::make_unique<ShadowFeature>(
      gpu, makeShadowConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
      &memory));

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  addDirectionalLightToScene(scene);

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 1u, 1u),
                        "sdsm_mismatched_gpu_result");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();

  const CameraFrameState camera{
      .projectionType = ProjectionType::Perspective,
      .nearPlane = 0.1f,
      .farPlane = 30.0f,
  };

  RenderSettings publishSettings{};
  publishSettings.shadow.enabled = true;
  publishSettings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
  publishSettings.shadow.sdsmReductionBackend = ShadowSdsmReductionBackend::Gpu;

  for (uint64_t publishFrameIndex = 0u; publishFrameIndex < 2u;
       ++publishFrameIndex) {
    RenderFrameContext publishFrame{};
    publishFrame.frameIndex = publishFrameIndex;
    publishFrame.scene = &scene;
    publishFrame.resources = &renderer.resources();
    publishFrame.settings = &publishSettings;
    publishFrame.camera = camera;

    RenderGraphBuilder publishGraph(&memory);
    publishGraph.beginFrame(publishFrame.frameIndex);
    auto publishResult = pipeline.buildRenderGraph(
        publishFrame, renderer.resources(), publishGraph);
    ASSERT_FALSE(publishResult.hasError()) << publishResult.error();
    ASSERT_TRUE(publishResult.value());
    ASSERT_TRUE(
        publishFrame.sharedResources.shadowSdsmGpuReduceTarget.has_value());

    const GpuSdsmMinMaxResult gpuResult{
        .rawDeviceMinMax = publishFrameIndex == 0u ? glm::vec2(0.2f, 0.5f)
                                                   : glm::vec2(0.8f, 0.9f),
        .sourceFrameIndex = 0u,
        .valid = 1u,
    };
    auto writeResult = gpu.updateBuffer(
        publishFrame.sharedResources.shadowSdsmGpuReduceTarget->buffer,
        std::as_bytes(std::span<const GpuSdsmMinMaxResult>(&gpuResult, 1u)),
        0u);
    ASSERT_FALSE(writeResult.hasError()) << writeResult.error();
  }

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
  settings.shadow.sdsmReductionBackend = ShadowSdsmReductionBackend::Gpu;
  settings.shadow.sdsmTemporalBlend = 0.0f;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 2u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  frameContext.camera = camera;
  frameContext.sharedResources.sceneDepthPyramidTextures[0] =
      pyramidTextureResult.value();
  frameContext.sharedResources.sceneDepthPyramidLevelCount = 1u;
  frameContext.sharedResources.sceneDepthPyramidSourceFrameIndex = 1u;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());
  ASSERT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());

  const ShadowSdsmDebugFrameData &sdsm =
      frameContext.sharedResources.shadowDebugFrameData->sdsm;
  EXPECT_EQ(sdsm.status, ShadowSdsmStatus::Active);
  EXPECT_EQ(sdsm.activeReductionBackend, ShadowSdsmReductionBackend::Gpu);
  EXPECT_FALSE(sdsm.fixedFallbackActive);
  EXPECT_FALSE(sdsm.reductionFallbackActive);
  EXPECT_EQ(sdsm.sourceFrameIndex, 0u);
  EXPECT_FLOAT_EQ(sdsm.rawDeviceMin, 0.2f);
  EXPECT_FLOAT_EQ(sdsm.rawDeviceMax, 0.5f);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureSdsmFallsBackWithoutPreviousFrameData) {
  std::array<std::byte, 64 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
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
  settings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 1u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  frameContext.camera.view = glm::mat4(1.0f);
  frameContext.camera.proj = glm::mat4(1.0f);
  frameContext.camera.cameraPos = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
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
  ASSERT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());
  const ShadowSdsmDebugFrameData &sdsm =
      frameContext.sharedResources.shadowDebugFrameData->sdsm;
  EXPECT_EQ(sdsm.status, ShadowSdsmStatus::Unavailable);
  EXPECT_EQ(sdsm.mode, ShadowSdsmMode::PreviousFrameMinMax);
  EXPECT_EQ(sdsm.splitCount, settings.shadow.cascadeCount);
  EXPECT_TRUE(sdsm.fixedFallbackActive);
  EXPECT_EQ(sdsm.fixedSplitDepths, sdsm.effectiveSplitDepths);
  EXPECT_FLOAT_EQ(sdsm.fixedRangeNear, sdsm.effectiveRangeNear);
  EXPECT_FLOAT_EQ(sdsm.fixedRangeFar, sdsm.effectiveRangeFar);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureSdsmMarksMismatchedSourceFrameAsStale) {
  std::array<std::byte, 64 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
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

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 1u, 1u),
                        "sdsm_stale_texture");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 2u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  frameContext.camera.view = glm::mat4(1.0f);
  frameContext.camera.proj = glm::mat4(1.0f);
  frameContext.camera.cameraPos = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
  frameContext.camera.aspectRatio = 1.0f;
  frameContext.camera.projectionType = ProjectionType::Perspective;
  frameContext.camera.nearPlane = 0.1f;
  frameContext.camera.farPlane = 30.0f;
  frameContext.camera.fovYRadians = glm::radians(60.0f);
  frameContext.sharedResources.sceneDepthPyramidTextures[0] =
      pyramidTextureResult.value();
  frameContext.sharedResources.sceneDepthPyramidLevelCount = 1u;
  frameContext.sharedResources.sceneDepthPyramidSourceFrameIndex = 0u;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);

  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());
  ASSERT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());
  const ShadowSdsmDebugFrameData &sdsm =
      frameContext.sharedResources.shadowDebugFrameData->sdsm;
  EXPECT_EQ(sdsm.status, ShadowSdsmStatus::Stale);
  EXPECT_TRUE(sdsm.fixedFallbackActive);
  EXPECT_EQ(sdsm.fixedSplitDepths, sdsm.effectiveSplitDepths);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureSdsmReusesLastValidRangeWhenNextFrameDataIsUnavailable) {
  std::array<std::byte, 64 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
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

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 1u, 1u),
                        "sdsm_unavailable_reuse_texture");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();
  const TextureHandle pyramidTexture = pyramidTextureResult.value();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
  settings.shadow.sdsmTemporalBlend = 0.0f;

  const CameraFrameState camera{
      .projectionType = ProjectionType::Perspective,
      .nearPlane = 0.1f,
      .farPlane = 30.0f,
  };

  auto buildFrame = [&](uint64_t frameIndex,
                        std::optional<uint64_t> sourceFrameIndex,
                        std::optional<std::array<float, 2u>> deviceDepths)
      -> ShadowSdsmDebugFrameData {
    if (deviceDepths.has_value()) {
      auto seedResult = gpu.seedTextureBytes(
          pyramidTexture, std::as_bytes(std::span<const float>(
                              deviceDepths->data(), deviceDepths->size())));
      EXPECT_FALSE(seedResult.hasError()) << seedResult.error();
    }

    RenderFrameContext frameContext{};
    frameContext.frameIndex = frameIndex;
    frameContext.scene = &scene;
    frameContext.resources = &renderer.resources();
    frameContext.settings = &settings;
    frameContext.camera = camera;
    frameContext.sharedResources.sceneDepthPyramidTextures[0] = pyramidTexture;
    frameContext.sharedResources.sceneDepthPyramidLevelCount =
        deviceDepths.has_value() ? 1u : 0u;
    frameContext.sharedResources.sceneDepthPyramidSourceFrameIndex =
        sourceFrameIndex;

    RenderGraphBuilder graph(&memory);
    graph.beginFrame(frameContext.frameIndex);
    auto buildResult =
        pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
    EXPECT_FALSE(buildResult.hasError()) << buildResult.error();
    EXPECT_TRUE(buildResult.value());
    EXPECT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());
    return frameContext.sharedResources.shadowDebugFrameData->sdsm;
  };

  const ShadowSdsmDebugFrameData activeSdsm =
      buildFrame(2u, 1u, std::array<float, 2u>{0.2f, 0.5f});
  ASSERT_EQ(activeSdsm.status, ShadowSdsmStatus::Active);
  ASSERT_FALSE(activeSdsm.fixedFallbackActive);

  const ShadowSdsmDebugFrameData unavailableSdsm =
      buildFrame(3u, std::nullopt, std::nullopt);
  EXPECT_EQ(unavailableSdsm.status, ShadowSdsmStatus::Unavailable);
  EXPECT_FALSE(unavailableSdsm.fixedFallbackActive);
  EXPECT_NEAR(unavailableSdsm.smoothedLinearMin, activeSdsm.smoothedLinearMin,
              1.0e-5f);
  EXPECT_NEAR(unavailableSdsm.smoothedLinearMax, activeSdsm.smoothedLinearMax,
              1.0e-5f);
  EXPECT_NEAR(unavailableSdsm.effectiveRangeNear, activeSdsm.effectiveRangeNear,
              1.0e-5f);
  EXPECT_NEAR(unavailableSdsm.effectiveRangeFar, activeSdsm.effectiveRangeFar,
              1.0e-5f);
  EXPECT_EQ(unavailableSdsm.effectiveSplitDepths,
            activeSdsm.effectiveSplitDepths);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureSdsmReusesLastValidRangeWhenSourceFrameIsStale) {
  std::array<std::byte, 64 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
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

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 1u, 1u),
                        "sdsm_stale_reuse_texture");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();
  const TextureHandle pyramidTexture = pyramidTextureResult.value();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
  settings.shadow.sdsmTemporalBlend = 0.0f;

  const CameraFrameState camera{
      .projectionType = ProjectionType::Perspective,
      .nearPlane = 0.1f,
      .farPlane = 30.0f,
  };

  auto buildFrame =
      [&](uint64_t frameIndex, uint64_t sourceFrameIndex,
          std::array<float, 2u> deviceDepths) -> ShadowSdsmDebugFrameData {
    auto seedResult = gpu.seedTextureBytes(
        pyramidTexture, std::as_bytes(std::span<const float>(
                            deviceDepths.data(), deviceDepths.size())));
    EXPECT_FALSE(seedResult.hasError()) << seedResult.error();

    RenderFrameContext frameContext{};
    frameContext.frameIndex = frameIndex;
    frameContext.scene = &scene;
    frameContext.resources = &renderer.resources();
    frameContext.settings = &settings;
    frameContext.camera = camera;
    frameContext.sharedResources.sceneDepthPyramidTextures[0] = pyramidTexture;
    frameContext.sharedResources.sceneDepthPyramidLevelCount = 1u;
    frameContext.sharedResources.sceneDepthPyramidSourceFrameIndex =
        sourceFrameIndex;

    RenderGraphBuilder graph(&memory);
    graph.beginFrame(frameContext.frameIndex);
    auto buildResult =
        pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
    EXPECT_FALSE(buildResult.hasError()) << buildResult.error();
    EXPECT_TRUE(buildResult.value());
    EXPECT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());
    return frameContext.sharedResources.shadowDebugFrameData->sdsm;
  };

  const ShadowSdsmDebugFrameData activeSdsm = buildFrame(2u, 1u, {0.2f, 0.5f});
  ASSERT_EQ(activeSdsm.status, ShadowSdsmStatus::Active);
  ASSERT_FALSE(activeSdsm.fixedFallbackActive);

  const ShadowSdsmDebugFrameData staleSdsm = buildFrame(3u, 0u, {0.4f, 0.8f});
  EXPECT_EQ(staleSdsm.status, ShadowSdsmStatus::Stale);
  EXPECT_FALSE(staleSdsm.fixedFallbackActive);
  EXPECT_NEAR(staleSdsm.smoothedLinearMin, activeSdsm.smoothedLinearMin,
              1.0e-5f);
  EXPECT_NEAR(staleSdsm.smoothedLinearMax, activeSdsm.smoothedLinearMax,
              1.0e-5f);
  EXPECT_NEAR(staleSdsm.effectiveRangeNear, activeSdsm.effectiveRangeNear,
              1.0e-5f);
  EXPECT_NEAR(staleSdsm.effectiveRangeFar, activeSdsm.effectiveRangeFar,
              1.0e-5f);
  EXPECT_EQ(staleSdsm.effectiveSplitDepths, activeSdsm.effectiveSplitDepths);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureSdsmUsesPreviousFrameMinMaxAndSmoothsThenRejectsInvalidData) {
  std::array<std::byte, 64 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
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

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 1u, 1u),
                        "sdsm_history_texture");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();
  const TextureHandle pyramidTexture = pyramidTextureResult.value();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
  settings.shadow.sdsmTemporalBlend = 0.75f;

  const CameraFrameState camera{
      .projectionType = ProjectionType::Perspective,
      .nearPlane = 0.1f,
      .farPlane = 30.0f,
  };

  auto buildFrame = [&](uint64_t frameIndex, uint64_t sourceFrameIndex,
                        float rawMin,
                        float rawMax) -> ShadowSdsmDebugFrameData {
    const std::array<float, 2u> depths = {rawMin, rawMax};
    auto seedResult = gpu.seedTextureBytes(
        pyramidTexture,
        std::as_bytes(std::span<const float>(depths.data(), depths.size())));
    EXPECT_FALSE(seedResult.hasError()) << seedResult.error();

    RenderFrameContext frameContext{};
    frameContext.frameIndex = frameIndex;
    frameContext.scene = &scene;
    frameContext.resources = &renderer.resources();
    frameContext.settings = &settings;
    frameContext.camera = camera;
    frameContext.sharedResources.sceneDepthPyramidTextures[0] = pyramidTexture;
    frameContext.sharedResources.sceneDepthPyramidLevelCount = 1u;
    frameContext.sharedResources.sceneDepthPyramidSourceFrameIndex =
        sourceFrameIndex;

    RenderGraphBuilder graph(&memory);
    graph.beginFrame(frameContext.frameIndex);
    auto buildResult =
        pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
    EXPECT_FALSE(buildResult.hasError()) << buildResult.error();
    EXPECT_TRUE(buildResult.value());
    EXPECT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());
    return frameContext.sharedResources.shadowDebugFrameData->sdsm;
  };

  const ShadowSdsmDebugFrameData firstSdsm = buildFrame(2u, 1u, 0.2f, 0.5f);
  EXPECT_EQ(firstSdsm.status, ShadowSdsmStatus::Active);
  const float firstLinearMin =
      shadow_detail::linearizeDeviceDepthToViewDepth(0.2f, camera);
  const float firstLinearMax =
      shadow_detail::linearizeDeviceDepthToViewDepth(0.5f, camera);
  EXPECT_NEAR(firstSdsm.rawLinearMin, firstLinearMin, 1.0e-5f);
  EXPECT_NEAR(firstSdsm.rawLinearMax, firstLinearMax, 1.0e-5f);
  EXPECT_NEAR(firstSdsm.smoothedLinearMin, firstLinearMin, 1.0e-5f);
  EXPECT_NEAR(firstSdsm.smoothedLinearMax, firstLinearMax, 1.0e-5f);
  const float firstMinimumSdsmFarDepth =
      firstSdsm.splitCount > 1u
          ? firstSdsm.fixedSplitDepths[firstSdsm.splitCount - 1u]
          : firstSdsm.fixedRangeFar;
  EXPECT_NEAR(firstSdsm.effectiveRangeNear, camera.nearPlane, 1.0e-5f);
  EXPECT_NEAR(firstSdsm.effectiveRangeFar, firstSdsm.fixedRangeFar, 1.0e-5f);
  EXPECT_GT(firstSdsm.effectiveRangeFar, firstMinimumSdsmFarDepth);
  EXPECT_GT(firstSdsm.effectiveRangeFar, firstSdsm.smoothedLinearMax);
  if (firstSdsm.splitCount > 1u) {
    EXPECT_EQ(firstSdsm.fixedSplitDepths, firstSdsm.effectiveSplitDepths);
  }

  const ShadowSdsmDebugFrameData secondSdsm = buildFrame(3u, 2u, 0.4f, 0.8f);
  EXPECT_EQ(secondSdsm.status, ShadowSdsmStatus::Active);
  const float secondLinearMin =
      shadow_detail::linearizeDeviceDepthToViewDepth(0.4f, camera);
  const float secondLinearMax =
      shadow_detail::linearizeDeviceDepthToViewDepth(0.8f, camera);
  const float expectedSmoothedMin =
      firstLinearMin * settings.shadow.sdsmTemporalBlend +
      secondLinearMin * (1.0f - settings.shadow.sdsmTemporalBlend);
  const float expectedSmoothedMax =
      firstLinearMax * settings.shadow.sdsmTemporalBlend +
      secondLinearMax * (1.0f - settings.shadow.sdsmTemporalBlend);
  EXPECT_NEAR(secondSdsm.smoothedLinearMin, expectedSmoothedMin, 1.0e-5f);
  EXPECT_NEAR(secondSdsm.smoothedLinearMax, expectedSmoothedMax, 1.0e-5f);
  const float secondMinimumSdsmFarDepth =
      secondSdsm.splitCount > 1u
          ? secondSdsm.fixedSplitDepths[secondSdsm.splitCount - 1u]
          : secondSdsm.fixedRangeFar;
  EXPECT_NEAR(secondSdsm.effectiveRangeNear, camera.nearPlane, 1.0e-5f);
  EXPECT_NEAR(secondSdsm.effectiveRangeFar, secondSdsm.fixedRangeFar, 1.0e-5f);
  EXPECT_GT(secondSdsm.effectiveRangeFar, secondMinimumSdsmFarDepth);

  const ShadowSdsmDebugFrameData invalidSdsm = buildFrame(4u, 3u, 0.8f, 0.2f);
  EXPECT_EQ(invalidSdsm.status, ShadowSdsmStatus::Invalid);
  EXPECT_EQ(invalidSdsm.fixedSplitDepths, invalidSdsm.effectiveSplitDepths);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureSdsmKeepsOrthographicPreviousFrameMinMaxRangeFixed) {
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
  addDirectionalLightToScene(scene);

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 1u, 1u),
                        "sdsm_ortho_minmax_texture");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();
  const TextureHandle pyramidTexture = pyramidTextureResult.value();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.cascadeCount = 4u;
  settings.shadow.maxDistance = 150.0f;
  settings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
  settings.shadow.sdsmTemporalBlend = 0.0f;

  const CameraFrameState camera = makeSdsmOrthographicCamera(150.0f, 48.0f);
  const std::array<float, 2u> depths = {
      deviceDepthForViewDepth(2.0f, camera),
      deviceDepthForViewDepth(100.0f, camera),
  };
  auto seedResult = gpu.seedTextureBytes(
      pyramidTexture,
      std::as_bytes(std::span<const float>(depths.data(), depths.size())));
  ASSERT_FALSE(seedResult.hasError()) << seedResult.error();

  const ShadowSdsmDebugFrameData sdsm = buildShadowSdsmFrame(
      pipeline, renderer, memory, scene, settings, camera, 2u, 1u,
      std::span<const TextureHandle>(&pyramidTexture, 1u));
  ASSERT_EQ(sdsm.status, ShadowSdsmStatus::Active);
  EXPECT_FALSE(sdsm.fixedFallbackActive);
  EXPECT_FLOAT_EQ(sdsm.effectiveRangeFar, sdsm.fixedRangeFar);
  EXPECT_EQ(sdsm.effectiveSplitDepths, sdsm.fixedSplitDepths);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureSdsmReusesLastValidRangeWhenSourceDataIsInvalid) {
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

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 1u, 1u),
                        "sdsm_invalid_reuse_texture");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();
  const TextureHandle pyramidTexture = pyramidTextureResult.value();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.cascadeCount = 4u;
  settings.shadow.maxDistance = 100.0f;
  settings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
  settings.shadow.sdsmTemporalBlend = 0.0f;

  const CameraFrameState camera{
      .view =
          glm::lookAt(glm::vec3(0.0f, 2.0f, 6.0f), glm::vec3(0.0f, 0.5f, 0.0f),
                      glm::vec3(0.0f, 1.0f, 0.0f)),
      .proj = glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 100.0f),
      .cameraPos = glm::vec4(0.0f, 2.0f, 6.0f, 1.0f),
      .aspectRatio = 1.0f,
      .projectionType = ProjectionType::Perspective,
      .nearPlane = 0.1f,
      .farPlane = 100.0f,
      .fovYRadians = glm::radians(60.0f),
  };

  const auto deviceDepthForViewDepth = [&](float viewDepth) {
    const float clampedDepth =
        std::clamp(viewDepth, camera.nearPlane + 1.0e-4f, camera.farPlane);
    const float numerator =
        camera.farPlane - (camera.nearPlane * camera.farPlane) / clampedDepth;
    return std::clamp(numerator / (camera.farPlane - camera.nearPlane), 0.0f,
                      1.0f);
  };

  auto buildFrame =
      [&](uint64_t frameIndex, uint64_t sourceFrameIndex,
          std::array<float, 2u> deviceDepths) -> ShadowSdsmDebugFrameData {
    auto seedResult = gpu.seedTextureBytes(
        pyramidTexture, std::as_bytes(std::span<const float>(
                            deviceDepths.data(), deviceDepths.size())));
    EXPECT_FALSE(seedResult.hasError()) << seedResult.error();

    RenderFrameContext frameContext{};
    frameContext.frameIndex = frameIndex;
    frameContext.scene = &scene;
    frameContext.resources = &renderer.resources();
    frameContext.settings = &settings;
    frameContext.camera = camera;
    frameContext.sharedResources.sceneDepthPyramidTextures[0] = pyramidTexture;
    frameContext.sharedResources.sceneDepthPyramidLevelCount = 1u;
    frameContext.sharedResources.sceneDepthPyramidSourceFrameIndex =
        sourceFrameIndex;

    RenderGraphBuilder graph(&memory);
    graph.beginFrame(frameContext.frameIndex);
    auto buildResult =
        pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
    EXPECT_FALSE(buildResult.hasError()) << buildResult.error();
    EXPECT_TRUE(buildResult.value());
    EXPECT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());
    return frameContext.sharedResources.shadowDebugFrameData->sdsm;
  };

  const ShadowSdsmDebugFrameData activeSdsm =
      buildFrame(2u, 1u,
                 std::array<float, 2u>{deviceDepthForViewDepth(2.0f),
                                       deviceDepthForViewDepth(60.0f)});
  ASSERT_EQ(activeSdsm.status, ShadowSdsmStatus::Active);
  ASSERT_FLOAT_EQ(activeSdsm.effectiveRangeFar, activeSdsm.fixedRangeFar);
  ASSERT_EQ(activeSdsm.effectiveSplitDepths, activeSdsm.fixedSplitDepths);

  const ShadowSdsmDebugFrameData invalidSdsm =
      buildFrame(3u, 2u, std::array<float, 2u>{0.8f, 0.2f});
  EXPECT_EQ(invalidSdsm.status, ShadowSdsmStatus::Invalid);
  EXPECT_FALSE(invalidSdsm.fixedFallbackActive);
  EXPECT_NEAR(invalidSdsm.smoothedLinearMin, activeSdsm.smoothedLinearMin,
              1.0e-5f);
  EXPECT_NEAR(invalidSdsm.smoothedLinearMax, activeSdsm.smoothedLinearMax,
              1.0e-5f);
  EXPECT_NEAR(invalidSdsm.effectiveRangeFar, activeSdsm.effectiveRangeFar,
              0.05f);
  EXPECT_EQ(invalidSdsm.effectiveSplitDepths, activeSdsm.effectiveSplitDepths);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureSdsmTreatsCollapsedLinearDepthAsValidSourceData) {
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

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 1u, 1u),
                        "sdsm_collapsed_linear_depth_texture");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();
  const TextureHandle pyramidTexture = pyramidTextureResult.value();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.cascadeCount = 4u;
  settings.shadow.maxDistance = 150.0f;
  settings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
  settings.shadow.sdsmTemporalBlend = 0.0f;

  const CameraFrameState camera{
      .view =
          glm::lookAt(glm::vec3(0.0f, 2.0f, 6.0f), glm::vec3(0.0f, 0.5f, 0.0f),
                      glm::vec3(0.0f, 1.0f, 0.0f)),
      .proj = glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 150.0f),
      .cameraPos = glm::vec4(0.0f, 2.0f, 6.0f, 1.0f),
      .aspectRatio = 1.0f,
      .projectionType = ProjectionType::Perspective,
      .nearPlane = 0.1f,
      .farPlane = 150.0f,
      .fovYRadians = glm::radians(60.0f),
  };

  auto buildFrame =
      [&](uint64_t frameIndex, uint64_t sourceFrameIndex,
          std::array<float, 2u> deviceDepths) -> ShadowSdsmDebugFrameData {
    auto seedResult = gpu.seedTextureBytes(
        pyramidTexture, std::as_bytes(std::span<const float>(
                            deviceDepths.data(), deviceDepths.size())));
    EXPECT_FALSE(seedResult.hasError()) << seedResult.error();

    RenderFrameContext frameContext{};
    frameContext.frameIndex = frameIndex;
    frameContext.scene = &scene;
    frameContext.resources = &renderer.resources();
    frameContext.settings = &settings;
    frameContext.camera = camera;
    frameContext.sharedResources.sceneDepthPyramidTextures[0] = pyramidTexture;
    frameContext.sharedResources.sceneDepthPyramidLevelCount = 1u;
    frameContext.sharedResources.sceneDepthPyramidSourceFrameIndex =
        sourceFrameIndex;

    RenderGraphBuilder graph(&memory);
    graph.beginFrame(frameContext.frameIndex);
    auto buildResult =
        pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
    EXPECT_FALSE(buildResult.hasError()) << buildResult.error();
    EXPECT_TRUE(buildResult.value());
    EXPECT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());
    return frameContext.sharedResources.shadowDebugFrameData->sdsm;
  };

  const ShadowSdsmDebugFrameData firstSdsm =
      buildFrame(2u, 1u, std::array<float, 2u>{0.2f, 0.6f});
  ASSERT_EQ(firstSdsm.status, ShadowSdsmStatus::Active);

  const ShadowSdsmDebugFrameData collapsedSdsm =
      buildFrame(3u, 2u, std::array<float, 2u>{0.754985f, 0.754986f});
  EXPECT_EQ(collapsedSdsm.status, ShadowSdsmStatus::Active);
  EXPECT_FALSE(collapsedSdsm.fixedFallbackActive);
  EXPECT_GE(collapsedSdsm.rawLinearMax, collapsedSdsm.rawLinearMin);
  EXPECT_LT(collapsedSdsm.rawLinearMax - collapsedSdsm.rawLinearMin, 1.0e-3f);
  EXPECT_FLOAT_EQ(collapsedSdsm.effectiveRangeFar, collapsedSdsm.fixedRangeFar);
  EXPECT_EQ(collapsedSdsm.effectiveSplitDepths, collapsedSdsm.fixedSplitDepths);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureSdsmTreatsFarPlaneOnlyDepthAsValidSourceData) {
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

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 1u, 1u),
                        "sdsm_far_plane_only_texture");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();
  const TextureHandle pyramidTexture = pyramidTextureResult.value();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.cascadeCount = 4u;
  settings.shadow.maxDistance = 150.0f;
  settings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
  settings.shadow.sdsmTemporalBlend = 0.85f;

  const CameraFrameState camera{
      .view =
          glm::lookAt(glm::vec3(0.0f, 2.0f, 6.0f), glm::vec3(0.0f, 0.5f, 0.0f),
                      glm::vec3(0.0f, 1.0f, 0.0f)),
      .proj = glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 1699.661f),
      .cameraPos = glm::vec4(0.0f, 2.0f, 6.0f, 1.0f),
      .aspectRatio = 1.0f,
      .projectionType = ProjectionType::Perspective,
      .nearPlane = 0.073898f,
      .farPlane = 1699.661f,
      .fovYRadians = glm::radians(60.0f),
  };

  auto buildFrame =
      [&](uint64_t frameIndex, uint64_t sourceFrameIndex,
          std::array<float, 2u> deviceDepths) -> ShadowSdsmDebugFrameData {
    auto seedResult = gpu.seedTextureBytes(
        pyramidTexture, std::as_bytes(std::span<const float>(
                            deviceDepths.data(), deviceDepths.size())));
    EXPECT_FALSE(seedResult.hasError()) << seedResult.error();

    RenderFrameContext frameContext{};
    frameContext.frameIndex = frameIndex;
    frameContext.scene = &scene;
    frameContext.resources = &renderer.resources();
    frameContext.settings = &settings;
    frameContext.camera = camera;
    frameContext.sharedResources.sceneDepthPyramidTextures[0] = pyramidTexture;
    frameContext.sharedResources.sceneDepthPyramidLevelCount = 1u;
    frameContext.sharedResources.sceneDepthPyramidSourceFrameIndex =
        sourceFrameIndex;

    RenderGraphBuilder graph(&memory);
    graph.beginFrame(frameContext.frameIndex);
    auto buildResult =
        pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
    EXPECT_FALSE(buildResult.hasError()) << buildResult.error();
    EXPECT_TRUE(buildResult.value());
    EXPECT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());
    return frameContext.sharedResources.shadowDebugFrameData->sdsm;
  };

  const ShadowSdsmDebugFrameData farOnlySdsm =
      buildFrame(2u, 1u, std::array<float, 2u>{1.0f, 1.0f});
  EXPECT_EQ(farOnlySdsm.status, ShadowSdsmStatus::Active);
  EXPECT_FALSE(farOnlySdsm.fixedFallbackActive);
  EXPECT_FLOAT_EQ(farOnlySdsm.effectiveRangeFar, farOnlySdsm.fixedRangeFar);
  EXPECT_EQ(farOnlySdsm.effectiveSplitDepths, farOnlySdsm.fixedSplitDepths);
  EXPECT_GT(farOnlySdsm.rawLinearMin, farOnlySdsm.fixedRangeFar);
  EXPECT_GT(farOnlySdsm.rawLinearMax, farOnlySdsm.fixedRangeFar);
  EXPECT_GT(farOnlySdsm.smoothedLinearMax, farOnlySdsm.fixedRangeFar);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureSdsmKeepsFixedRangeWhenDepthMaxStaysAtFarPlane) {
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

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 1u, 1u),
                        "sdsm_far_plane_max_regression_texture");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();
  const TextureHandle pyramidTexture = pyramidTextureResult.value();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.cascadeCount = 4u;
  settings.shadow.maxDistance = 150.0f;
  settings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
  settings.shadow.sdsmTemporalBlend = 0.85f;

  const CameraFrameState camera{
      .view =
          glm::lookAt(glm::vec3(0.0f, 2.0f, 6.0f), glm::vec3(0.0f, 0.5f, 0.0f),
                      glm::vec3(0.0f, 1.0f, 0.0f)),
      .proj = glm::perspective(glm::radians(60.0f), 1.0f, 0.073898f, 1699.661f),
      .cameraPos = glm::vec4(0.0f, 2.0f, 6.0f, 1.0f),
      .aspectRatio = 1.0f,
      .projectionType = ProjectionType::Perspective,
      .nearPlane = 0.073898f,
      .farPlane = 1699.661f,
      .fovYRadians = glm::radians(60.0f),
  };

  auto buildFrame =
      [&](uint64_t frameIndex, uint64_t sourceFrameIndex,
          std::array<float, 2u> deviceDepths) -> ShadowSdsmDebugFrameData {
    auto seedResult = gpu.seedTextureBytes(
        pyramidTexture, std::as_bytes(std::span<const float>(
                            deviceDepths.data(), deviceDepths.size())));
    EXPECT_FALSE(seedResult.hasError()) << seedResult.error();

    RenderFrameContext frameContext{};
    frameContext.frameIndex = frameIndex;
    frameContext.scene = &scene;
    frameContext.resources = &renderer.resources();
    frameContext.settings = &settings;
    frameContext.camera = camera;
    frameContext.sharedResources.sceneDepthPyramidTextures[0] = pyramidTexture;
    frameContext.sharedResources.sceneDepthPyramidLevelCount = 1u;
    frameContext.sharedResources.sceneDepthPyramidSourceFrameIndex =
        sourceFrameIndex;

    RenderGraphBuilder graph(&memory);
    graph.beginFrame(frameContext.frameIndex);
    auto buildResult =
        pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
    EXPECT_FALSE(buildResult.hasError()) << buildResult.error();
    EXPECT_TRUE(buildResult.value());
    EXPECT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());
    return frameContext.sharedResources.shadowDebugFrameData->sdsm;
  };

  ShadowSdsmDebugFrameData sdsm =
      buildFrame(2u, 1u, std::array<float, 2u>{0.97f, 0.971f});
  ASSERT_EQ(sdsm.status, ShadowSdsmStatus::Active);
  EXPECT_FLOAT_EQ(sdsm.effectiveRangeFar, sdsm.fixedRangeFar);
  EXPECT_EQ(sdsm.effectiveSplitDepths, sdsm.fixedSplitDepths);

  for (uint64_t frameIndex = 3u; frameIndex <= 8u; ++frameIndex) {
    sdsm = buildFrame(frameIndex, frameIndex - 1u,
                      std::array<float, 2u>{0.878543f, 1.0f});
    ASSERT_EQ(sdsm.status, ShadowSdsmStatus::Active);
    EXPECT_FLOAT_EQ(sdsm.fixedRangeFar, 150.0f);
    EXPECT_FLOAT_EQ(sdsm.effectiveRangeFar, sdsm.fixedRangeFar);
    EXPECT_EQ(sdsm.effectiveSplitDepths, sdsm.fixedSplitDepths);
    EXPECT_GT(sdsm.rawLinearMax, sdsm.fixedRangeFar);
  }
}

TEST(RenderGraphRendererTest,
     ShadowFeatureSdsmKeepsFixedCoverageAcrossNearOnlyMultiCascadeUpdates) {
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

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 1u, 1u),
                        "sdsm_fixed_coverage_texture");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();
  const TextureHandle pyramidTexture = pyramidTextureResult.value();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.cascadeCount = 4u;
  settings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
  settings.shadow.sdsmTemporalBlend = 0.0f;

  const CameraFrameState camera{
      .view =
          glm::lookAt(glm::vec3(0.0f, 2.0f, 6.0f), glm::vec3(0.0f, 0.5f, 0.0f),
                      glm::vec3(0.0f, 1.0f, 0.0f)),
      .proj = glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 100.0f),
      .cameraPos = glm::vec4(0.0f, 2.0f, 6.0f, 1.0f),
      .aspectRatio = 1.0f,
      .projectionType = ProjectionType::Perspective,
      .nearPlane = 0.1f,
      .farPlane = 100.0f,
      .fovYRadians = glm::radians(60.0f),
  };

  const auto deviceDepthForViewDepth = [&](float viewDepth) {
    const float clampedDepth =
        std::clamp(viewDepth, camera.nearPlane + 1.0e-4f, camera.farPlane);
    const float numerator =
        camera.farPlane - (camera.nearPlane * camera.farPlane) / clampedDepth;
    return std::clamp(numerator / (camera.farPlane - camera.nearPlane), 0.0f,
                      1.0f);
  };

  auto buildFrame = [&](uint64_t frameIndex, uint64_t sourceFrameIndex,
                        float linearMin,
                        float linearMax) -> ShadowSdsmDebugFrameData {
    const std::array<float, 2u> depths = {deviceDepthForViewDepth(linearMin),
                                          deviceDepthForViewDepth(linearMax)};
    auto seedResult = gpu.seedTextureBytes(
        pyramidTexture,
        std::as_bytes(std::span<const float>(depths.data(), depths.size())));
    EXPECT_FALSE(seedResult.hasError()) << seedResult.error();

    RenderFrameContext frameContext{};
    frameContext.frameIndex = frameIndex;
    frameContext.scene = &scene;
    frameContext.resources = &renderer.resources();
    frameContext.settings = &settings;
    frameContext.camera = camera;
    frameContext.sharedResources.sceneDepthPyramidTextures[0] = pyramidTexture;
    frameContext.sharedResources.sceneDepthPyramidLevelCount = 1u;
    frameContext.sharedResources.sceneDepthPyramidSourceFrameIndex =
        sourceFrameIndex;

    RenderGraphBuilder graph(&memory);
    graph.beginFrame(frameContext.frameIndex);
    auto buildResult =
        pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
    EXPECT_FALSE(buildResult.hasError()) << buildResult.error();
    EXPECT_TRUE(buildResult.value());
    EXPECT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());
    return frameContext.sharedResources.shadowDebugFrameData->sdsm;
  };

  const ShadowSdsmDebugFrameData firstSdsm = buildFrame(2u, 1u, 2.0f, 60.0f);
  ASSERT_EQ(firstSdsm.status, ShadowSdsmStatus::Active);
  ASSERT_EQ(firstSdsm.splitCount, settings.shadow.cascadeCount);
  const float minimumFarDepth =
      firstSdsm.fixedSplitDepths[firstSdsm.splitCount - 1u];
  const float shrinkDepth =
      std::max((firstSdsm.fixedRangeFar - minimumFarDepth) * 0.05f, 1.0f);
  ASSERT_GT(firstSdsm.effectiveRangeFar, minimumFarDepth);
  EXPECT_NEAR(firstSdsm.smoothedLinearMax, 60.0f, 0.05f);
  EXPECT_FLOAT_EQ(firstSdsm.effectiveRangeFar, firstSdsm.fixedRangeFar);
  EXPECT_EQ(firstSdsm.effectiveSplitDepths, firstSdsm.fixedSplitDepths);

  const ShadowSdsmDebugFrameData heldSdsm =
      buildFrame(3u, 2u, 2.0f, 60.0f - shrinkDepth * 0.5f);
  ASSERT_EQ(heldSdsm.status, ShadowSdsmStatus::Active);
  EXPECT_NEAR(heldSdsm.smoothedLinearMax, 60.0f - shrinkDepth * 0.5f, 0.05f);
  EXPECT_FLOAT_EQ(heldSdsm.effectiveRangeFar, firstSdsm.effectiveRangeFar);
  EXPECT_EQ(heldSdsm.effectiveSplitDepths, heldSdsm.fixedSplitDepths);

  const ShadowSdsmDebugFrameData stableSdsm =
      buildFrame(4u, 3u, 2.0f, 60.0f - shrinkDepth * 2.0f);
  ASSERT_EQ(stableSdsm.status, ShadowSdsmStatus::Active);
  EXPECT_LT(stableSdsm.smoothedLinearMax, heldSdsm.smoothedLinearMax);
  EXPECT_FLOAT_EQ(stableSdsm.effectiveRangeFar, heldSdsm.effectiveRangeFar);
  EXPECT_EQ(stableSdsm.effectiveSplitDepths, stableSdsm.fixedSplitDepths);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureSdsmIgnoresSmallEffectiveFarExpansionsFromFarCascadeTexels) {
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

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 1u, 1u),
                        "sdsm_far_threshold_texture");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();
  const TextureHandle pyramidTexture = pyramidTextureResult.value();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.cascadeCount = 4u;
  settings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
  settings.shadow.sdsmTemporalBlend = 0.0f;

  const CameraFrameState camera{
      .view =
          glm::lookAt(glm::vec3(0.0f, 2.0f, 6.0f), glm::vec3(0.0f, 0.5f, 0.0f),
                      glm::vec3(0.0f, 1.0f, 0.0f)),
      .proj = glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 100.0f),
      .cameraPos = glm::vec4(0.0f, 2.0f, 6.0f, 1.0f),
      .aspectRatio = 1.0f,
      .projectionType = ProjectionType::Perspective,
      .nearPlane = 0.1f,
      .farPlane = 100.0f,
      .fovYRadians = glm::radians(60.0f),
  };

  const auto deviceDepthForViewDepth = [&](float viewDepth) {
    const float clampedDepth =
        std::clamp(viewDepth, camera.nearPlane + 1.0e-4f, camera.farPlane);
    const float numerator =
        camera.farPlane - (camera.nearPlane * camera.farPlane) / clampedDepth;
    return std::clamp(numerator / (camera.farPlane - camera.nearPlane), 0.0f,
                      1.0f);
  };

  auto buildFrame = [&](uint64_t frameIndex, uint64_t sourceFrameIndex,
                        float linearMin,
                        float linearMax) -> ShadowDebugFrameData {
    const std::array<float, 2u> depths = {deviceDepthForViewDepth(linearMin),
                                          deviceDepthForViewDepth(linearMax)};
    auto seedResult = gpu.seedTextureBytes(
        pyramidTexture,
        std::as_bytes(std::span<const float>(depths.data(), depths.size())));
    EXPECT_FALSE(seedResult.hasError()) << seedResult.error();

    RenderFrameContext frameContext{};
    frameContext.frameIndex = frameIndex;
    frameContext.scene = &scene;
    frameContext.resources = &renderer.resources();
    frameContext.settings = &settings;
    frameContext.camera = camera;
    frameContext.sharedResources.sceneDepthPyramidTextures[0] = pyramidTexture;
    frameContext.sharedResources.sceneDepthPyramidLevelCount = 1u;
    frameContext.sharedResources.sceneDepthPyramidSourceFrameIndex =
        sourceFrameIndex;

    RenderGraphBuilder graph(&memory);
    graph.beginFrame(frameContext.frameIndex);
    auto buildResult =
        pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
    EXPECT_FALSE(buildResult.hasError()) << buildResult.error();
    EXPECT_TRUE(buildResult.value());
    EXPECT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());
    return *frameContext.sharedResources.shadowDebugFrameData;
  };

  const ShadowDebugFrameData firstFrame = buildFrame(2u, 1u, 2.0f, 60.0f);
  ASSERT_EQ(firstFrame.sdsm.status, ShadowSdsmStatus::Active);
  ASSERT_EQ(firstFrame.cascadeCount, settings.shadow.cascadeCount);
  EXPECT_FLOAT_EQ(firstFrame.sdsm.effectiveRangeFar,
                  firstFrame.sdsm.fixedRangeFar);
  EXPECT_EQ(firstFrame.sdsm.effectiveSplitDepths,
            firstFrame.sdsm.fixedSplitDepths);
  const float farCascadeTexel =
      firstFrame.cascades[firstFrame.cascadeCount - 1u].texelWorldSize;
  ASSERT_GT(farCascadeTexel, 0.0f);
  const float updateThreshold = std::max(farCascadeTexel * 16.0f, 0.5f);

  const ShadowDebugFrameData heldFrame =
      buildFrame(3u, 2u, 2.0f, 60.0f + updateThreshold * 0.5f);
  ASSERT_EQ(heldFrame.sdsm.status, ShadowSdsmStatus::Active);
  EXPECT_NEAR(heldFrame.sdsm.smoothedLinearMax, 60.0f + updateThreshold * 0.5f,
              0.1f);
  EXPECT_FLOAT_EQ(heldFrame.sdsm.effectiveRangeFar,
                  firstFrame.sdsm.effectiveRangeFar);
  EXPECT_EQ(heldFrame.sdsm.effectiveSplitDepths,
            heldFrame.sdsm.fixedSplitDepths);

  const ShadowDebugFrameData expandedFrame =
      buildFrame(4u, 3u, 2.0f, 60.0f + updateThreshold * 2.0f);
  ASSERT_EQ(expandedFrame.sdsm.status, ShadowSdsmStatus::Active);
  EXPECT_GT(expandedFrame.sdsm.smoothedLinearMax,
            heldFrame.sdsm.smoothedLinearMax);
  EXPECT_FLOAT_EQ(expandedFrame.sdsm.effectiveRangeFar,
                  heldFrame.sdsm.effectiveRangeFar);
  EXPECT_EQ(expandedFrame.sdsm.effectiveSplitDepths,
            expandedFrame.sdsm.fixedSplitDepths);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureSdsmKeepsFixedSplitDepthsUntilVisibleRangeEntersLastCascade) {
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

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 1u, 1u),
                        "sdsm_fixed_split_gate_texture");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();
  const TextureHandle pyramidTexture = pyramidTextureResult.value();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.cascadeCount = 4u;
  settings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
  settings.shadow.sdsmTemporalBlend = 0.0f;

  const CameraFrameState camera{
      .view =
          glm::lookAt(glm::vec3(0.0f, 2.0f, 6.0f), glm::vec3(0.0f, 0.5f, 0.0f),
                      glm::vec3(0.0f, 1.0f, 0.0f)),
      .proj = glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 100.0f),
      .cameraPos = glm::vec4(0.0f, 2.0f, 6.0f, 1.0f),
      .aspectRatio = 1.0f,
      .projectionType = ProjectionType::Perspective,
      .nearPlane = 0.1f,
      .farPlane = 100.0f,
      .fovYRadians = glm::radians(60.0f),
  };

  const auto deviceDepthForViewDepth = [&](float viewDepth) {
    const float clampedDepth =
        std::clamp(viewDepth, camera.nearPlane + 1.0e-4f, camera.farPlane);
    const float numerator =
        camera.farPlane - (camera.nearPlane * camera.farPlane) / clampedDepth;
    return std::clamp(numerator / (camera.farPlane - camera.nearPlane), 0.0f,
                      1.0f);
  };

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 2u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  frameContext.camera = camera;
  frameContext.sharedResources.sceneDepthPyramidTextures[0] = pyramidTexture;
  frameContext.sharedResources.sceneDepthPyramidLevelCount = 1u;
  frameContext.sharedResources.sceneDepthPyramidSourceFrameIndex = 1u;

  const std::array<float, 2u> depths = {deviceDepthForViewDepth(2.0f),
                                        deviceDepthForViewDepth(35.0f)};
  auto seedResult = gpu.seedTextureBytes(
      pyramidTexture,
      std::as_bytes(std::span<const float>(depths.data(), depths.size())));
  ASSERT_FALSE(seedResult.hasError()) << seedResult.error();

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());
  ASSERT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());
  const ShadowSdsmDebugFrameData &sdsm =
      frameContext.sharedResources.shadowDebugFrameData->sdsm;
  ASSERT_EQ(sdsm.status, ShadowSdsmStatus::Active);
  ASSERT_EQ(sdsm.splitCount, settings.shadow.cascadeCount);
  EXPECT_FLOAT_EQ(sdsm.effectiveRangeFar, sdsm.fixedRangeFar);
  EXPECT_EQ(sdsm.effectiveSplitDepths, sdsm.fixedSplitDepths);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureSdsmKeepsFixedRangeWhenVisibleDepthStaysNearMaxDistance) {
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

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 1u, 1u),
                        "sdsm_near_far_boundary_texture");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();
  const TextureHandle pyramidTexture = pyramidTextureResult.value();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.cascadeCount = 4u;
  settings.shadow.maxDistance = 150.0f;
  settings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
  settings.shadow.sdsmTemporalBlend = 0.0f;

  const CameraFrameState camera{
      .view =
          glm::lookAt(glm::vec3(0.0f, 2.0f, 6.0f), glm::vec3(0.0f, 0.5f, 0.0f),
                      glm::vec3(0.0f, 1.0f, 0.0f)),
      .proj = glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 150.0f),
      .cameraPos = glm::vec4(0.0f, 2.0f, 6.0f, 1.0f),
      .aspectRatio = 1.0f,
      .projectionType = ProjectionType::Perspective,
      .nearPlane = 0.1f,
      .farPlane = 150.0f,
      .fovYRadians = glm::radians(60.0f),
  };

  const auto deviceDepthForViewDepth = [&](float viewDepth) {
    const float clampedDepth =
        std::clamp(viewDepth, camera.nearPlane + 1.0e-4f, camera.farPlane);
    const float numerator =
        camera.farPlane - (camera.nearPlane * camera.farPlane) / clampedDepth;
    return std::clamp(numerator / (camera.farPlane - camera.nearPlane), 0.0f,
                      1.0f);
  };

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 2u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  frameContext.camera = camera;
  frameContext.sharedResources.sceneDepthPyramidTextures[0] = pyramidTexture;
  frameContext.sharedResources.sceneDepthPyramidLevelCount = 1u;
  frameContext.sharedResources.sceneDepthPyramidSourceFrameIndex = 1u;

  const std::array<float, 2u> depths = {deviceDepthForViewDepth(2.0f),
                                        deviceDepthForViewDepth(130.0f)};
  auto seedResult = gpu.seedTextureBytes(
      pyramidTexture,
      std::as_bytes(std::span<const float>(depths.data(), depths.size())));
  ASSERT_FALSE(seedResult.hasError()) << seedResult.error();

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());
  ASSERT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());
  const ShadowSdsmDebugFrameData &sdsm =
      frameContext.sharedResources.shadowDebugFrameData->sdsm;
  ASSERT_EQ(sdsm.status, ShadowSdsmStatus::Active);
  EXPECT_FLOAT_EQ(sdsm.fixedRangeFar, 150.0f);
  EXPECT_FLOAT_EQ(sdsm.effectiveRangeFar, sdsm.fixedRangeFar);
  EXPECT_EQ(sdsm.effectiveSplitDepths, sdsm.fixedSplitDepths);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureSdsmKeepsFixedRangeWhenTemporalBlendDecaysInto120Band) {
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

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 1u, 1u),
                        "sdsm_temporal_120_band_texture");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();
  const TextureHandle pyramidTexture = pyramidTextureResult.value();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.cascadeCount = 4u;
  settings.shadow.maxDistance = 150.0f;
  settings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
  settings.shadow.sdsmTemporalBlend = 0.85f;

  const CameraFrameState camera{
      .view =
          glm::lookAt(glm::vec3(0.0f, 2.0f, 6.0f), glm::vec3(0.0f, 0.5f, 0.0f),
                      glm::vec3(0.0f, 1.0f, 0.0f)),
      .proj = glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 150.0f),
      .cameraPos = glm::vec4(0.0f, 2.0f, 6.0f, 1.0f),
      .aspectRatio = 1.0f,
      .projectionType = ProjectionType::Perspective,
      .nearPlane = 0.1f,
      .farPlane = 150.0f,
      .fovYRadians = glm::radians(60.0f),
  };

  const auto deviceDepthForViewDepth = [&](float viewDepth) {
    const float clampedDepth =
        std::clamp(viewDepth, camera.nearPlane + 1.0e-4f, camera.farPlane);
    const float numerator =
        camera.farPlane - (camera.nearPlane * camera.farPlane) / clampedDepth;
    return std::clamp(numerator / (camera.farPlane - camera.nearPlane), 0.0f,
                      1.0f);
  };

  auto buildFrame = [&](uint64_t frameIndex, uint64_t sourceFrameIndex,
                        float linearMin,
                        float linearMax) -> ShadowSdsmDebugFrameData {
    const std::array<float, 2u> depths = {deviceDepthForViewDepth(linearMin),
                                          deviceDepthForViewDepth(linearMax)};
    auto seedResult = gpu.seedTextureBytes(
        pyramidTexture,
        std::as_bytes(std::span<const float>(depths.data(), depths.size())));
    EXPECT_FALSE(seedResult.hasError()) << seedResult.error();

    RenderFrameContext frameContext{};
    frameContext.frameIndex = frameIndex;
    frameContext.scene = &scene;
    frameContext.resources = &renderer.resources();
    frameContext.settings = &settings;
    frameContext.camera = camera;
    frameContext.sharedResources.sceneDepthPyramidTextures[0] = pyramidTexture;
    frameContext.sharedResources.sceneDepthPyramidLevelCount = 1u;
    frameContext.sharedResources.sceneDepthPyramidSourceFrameIndex =
        sourceFrameIndex;

    RenderGraphBuilder graph(&memory);
    graph.beginFrame(frameContext.frameIndex);
    auto buildResult =
        pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
    EXPECT_FALSE(buildResult.hasError()) << buildResult.error();
    EXPECT_TRUE(buildResult.value());
    EXPECT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());
    return frameContext.sharedResources.shadowDebugFrameData->sdsm;
  };

  ShadowSdsmDebugFrameData sdsm = buildFrame(2u, 1u, 2.0f, 149.0f);
  ASSERT_EQ(sdsm.status, ShadowSdsmStatus::Active);
  EXPECT_FLOAT_EQ(sdsm.effectiveRangeFar, sdsm.fixedRangeFar);
  EXPECT_EQ(sdsm.effectiveSplitDepths, sdsm.fixedSplitDepths);

  for (uint64_t frameIndex = 3u; frameIndex < 15u; ++frameIndex) {
    sdsm = buildFrame(frameIndex, frameIndex - 1u, 2.0f, 120.0f);
    ASSERT_EQ(sdsm.status, ShadowSdsmStatus::Active);
    EXPECT_FLOAT_EQ(sdsm.fixedRangeFar, 150.0f);
    EXPECT_FLOAT_EQ(sdsm.effectiveRangeFar, sdsm.fixedRangeFar);
    EXPECT_EQ(sdsm.effectiveSplitDepths, sdsm.fixedSplitDepths);
  }

  EXPECT_GT(sdsm.smoothedLinearMax, 120.0f);
  EXPECT_LT(sdsm.smoothedLinearMax, 130.0f);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureSdsmKeepsFixedRangeWhenOnlyHistoryRemainsInLastCascade) {
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

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 1u, 1u),
                        "sdsm_history_only_far_cascade_texture");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();
  const TextureHandle pyramidTexture = pyramidTextureResult.value();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.cascadeCount = 4u;
  settings.shadow.maxDistance = 150.0f;
  settings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
  settings.shadow.sdsmTemporalBlend = 0.85f;

  const CameraFrameState camera{
      .view =
          glm::lookAt(glm::vec3(0.0f, 2.0f, 6.0f), glm::vec3(0.0f, 0.5f, 0.0f),
                      glm::vec3(0.0f, 1.0f, 0.0f)),
      .proj = glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 150.0f),
      .cameraPos = glm::vec4(0.0f, 2.0f, 6.0f, 1.0f),
      .aspectRatio = 1.0f,
      .projectionType = ProjectionType::Perspective,
      .nearPlane = 0.1f,
      .farPlane = 150.0f,
      .fovYRadians = glm::radians(60.0f),
  };

  const auto deviceDepthForViewDepth = [&](float viewDepth) {
    const float clampedDepth =
        std::clamp(viewDepth, camera.nearPlane + 1.0e-4f, camera.farPlane);
    const float numerator =
        camera.farPlane - (camera.nearPlane * camera.farPlane) / clampedDepth;
    return std::clamp(numerator / (camera.farPlane - camera.nearPlane), 0.0f,
                      1.0f);
  };

  auto buildFrame = [&](uint64_t frameIndex, uint64_t sourceFrameIndex,
                        float linearMin,
                        float linearMax) -> ShadowSdsmDebugFrameData {
    const std::array<float, 2u> depths = {deviceDepthForViewDepth(linearMin),
                                          deviceDepthForViewDepth(linearMax)};
    auto seedResult = gpu.seedTextureBytes(
        pyramidTexture,
        std::as_bytes(std::span<const float>(depths.data(), depths.size())));
    EXPECT_FALSE(seedResult.hasError()) << seedResult.error();

    RenderFrameContext frameContext{};
    frameContext.frameIndex = frameIndex;
    frameContext.scene = &scene;
    frameContext.resources = &renderer.resources();
    frameContext.settings = &settings;
    frameContext.camera = camera;
    frameContext.sharedResources.sceneDepthPyramidTextures[0] = pyramidTexture;
    frameContext.sharedResources.sceneDepthPyramidLevelCount = 1u;
    frameContext.sharedResources.sceneDepthPyramidSourceFrameIndex =
        sourceFrameIndex;

    RenderGraphBuilder graph(&memory);
    graph.beginFrame(frameContext.frameIndex);
    auto buildResult =
        pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
    EXPECT_FALSE(buildResult.hasError()) << buildResult.error();
    EXPECT_TRUE(buildResult.value());
    EXPECT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());
    return frameContext.sharedResources.shadowDebugFrameData->sdsm;
  };

  ShadowSdsmDebugFrameData sdsm = buildFrame(2u, 1u, 2.0f, 149.0f);
  ASSERT_EQ(sdsm.status, ShadowSdsmStatus::Active);
  EXPECT_FLOAT_EQ(sdsm.effectiveRangeFar, sdsm.fixedRangeFar);
  EXPECT_EQ(sdsm.effectiveSplitDepths, sdsm.fixedSplitDepths);

  for (uint64_t frameIndex = 3u; frameIndex <= 5u; ++frameIndex) {
    sdsm = buildFrame(frameIndex, frameIndex - 1u, 2.1f, 2.16f);
    ASSERT_EQ(sdsm.status, ShadowSdsmStatus::Active);
    EXPECT_FLOAT_EQ(sdsm.fixedRangeFar, 150.0f);
    EXPECT_FLOAT_EQ(sdsm.effectiveRangeFar, sdsm.fixedRangeFar);
    EXPECT_EQ(sdsm.effectiveSplitDepths, sdsm.fixedSplitDepths);
  }

  EXPECT_GT(sdsm.smoothedLinearMax, 89.0f);
  EXPECT_LT(sdsm.smoothedLinearMax, 96.0f);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureSdsmKeepsFixedRangeOnceSmoothedFarReenters120Band) {
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

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 1u, 1u),
                        "sdsm_recover_fixed_far_texture");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();
  const TextureHandle pyramidTexture = pyramidTextureResult.value();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.cascadeCount = 4u;
  settings.shadow.maxDistance = 150.0f;
  settings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
  settings.shadow.sdsmTemporalBlend = 0.85f;

  const CameraFrameState camera{
      .view =
          glm::lookAt(glm::vec3(0.0f, 2.0f, 6.0f), glm::vec3(0.0f, 0.5f, 0.0f),
                      glm::vec3(0.0f, 1.0f, 0.0f)),
      .proj = glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 150.0f),
      .cameraPos = glm::vec4(0.0f, 2.0f, 6.0f, 1.0f),
      .aspectRatio = 1.0f,
      .projectionType = ProjectionType::Perspective,
      .nearPlane = 0.1f,
      .farPlane = 150.0f,
      .fovYRadians = glm::radians(60.0f),
  };

  const auto deviceDepthForViewDepth = [&](float viewDepth) {
    const float clampedDepth =
        std::clamp(viewDepth, camera.nearPlane + 1.0e-4f, camera.farPlane);
    const float numerator =
        camera.farPlane - (camera.nearPlane * camera.farPlane) / clampedDepth;
    return std::clamp(numerator / (camera.farPlane - camera.nearPlane), 0.0f,
                      1.0f);
  };

  auto buildFrame = [&](uint64_t frameIndex, uint64_t sourceFrameIndex,
                        float linearMin,
                        float linearMax) -> ShadowSdsmDebugFrameData {
    const std::array<float, 2u> depths = {deviceDepthForViewDepth(linearMin),
                                          deviceDepthForViewDepth(linearMax)};
    auto seedResult = gpu.seedTextureBytes(
        pyramidTexture,
        std::as_bytes(std::span<const float>(depths.data(), depths.size())));
    EXPECT_FALSE(seedResult.hasError()) << seedResult.error();

    RenderFrameContext frameContext{};
    frameContext.frameIndex = frameIndex;
    frameContext.scene = &scene;
    frameContext.resources = &renderer.resources();
    frameContext.settings = &settings;
    frameContext.camera = camera;
    frameContext.sharedResources.sceneDepthPyramidTextures[0] = pyramidTexture;
    frameContext.sharedResources.sceneDepthPyramidLevelCount = 1u;
    frameContext.sharedResources.sceneDepthPyramidSourceFrameIndex =
        sourceFrameIndex;

    RenderGraphBuilder graph(&memory);
    graph.beginFrame(frameContext.frameIndex);
    auto buildResult =
        pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
    EXPECT_FALSE(buildResult.hasError()) << buildResult.error();
    EXPECT_TRUE(buildResult.value());
    EXPECT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());
    return frameContext.sharedResources.shadowDebugFrameData->sdsm;
  };

  ShadowSdsmDebugFrameData sdsm = buildFrame(2u, 1u, 2.0f, 60.0f);
  ASSERT_EQ(sdsm.status, ShadowSdsmStatus::Active);
  ASSERT_FLOAT_EQ(sdsm.effectiveRangeFar, sdsm.fixedRangeFar);
  ASSERT_EQ(sdsm.effectiveSplitDepths, sdsm.fixedSplitDepths);

  bool sawSmoothedMaxInBand = false;
  for (uint64_t frameIndex = 3u; frameIndex < 16u; ++frameIndex) {
    sdsm = buildFrame(frameIndex, frameIndex - 1u, 2.0f, 149.0f);
    ASSERT_EQ(sdsm.status, ShadowSdsmStatus::Active);
    if (sdsm.smoothedLinearMax > 120.0f && sdsm.smoothedLinearMax < 130.0f) {
      sawSmoothedMaxInBand = true;
      EXPECT_FLOAT_EQ(sdsm.effectiveRangeFar, sdsm.fixedRangeFar);
      EXPECT_EQ(sdsm.effectiveSplitDepths, sdsm.fixedSplitDepths);
      break;
    }
  }

  EXPECT_TRUE(sawSmoothedMaxInBand);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureSdsmKeepsFixedSplitsAroundLastCascadeBoundary) {
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

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 1u, 1u),
                        "sdsm_split_boundary_texture");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();
  const TextureHandle pyramidTexture = pyramidTextureResult.value();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.cascadeCount = 4u;
  settings.shadow.sdsmMode = ShadowSdsmMode::PreviousFrameMinMax;
  settings.shadow.sdsmTemporalBlend = 0.0f;

  const CameraFrameState camera{
      .view =
          glm::lookAt(glm::vec3(0.0f, 2.0f, 6.0f), glm::vec3(0.0f, 0.5f, 0.0f),
                      glm::vec3(0.0f, 1.0f, 0.0f)),
      .proj = glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 100.0f),
      .cameraPos = glm::vec4(0.0f, 2.0f, 6.0f, 1.0f),
      .aspectRatio = 1.0f,
      .projectionType = ProjectionType::Perspective,
      .nearPlane = 0.1f,
      .farPlane = 100.0f,
      .fovYRadians = glm::radians(60.0f),
  };

  const auto deviceDepthForViewDepth = [&](float viewDepth) {
    const float clampedDepth =
        std::clamp(viewDepth, camera.nearPlane + 1.0e-4f, camera.farPlane);
    const float numerator =
        camera.farPlane - (camera.nearPlane * camera.farPlane) / clampedDepth;
    return std::clamp(numerator / (camera.farPlane - camera.nearPlane), 0.0f,
                      1.0f);
  };

  auto buildFrame = [&](uint64_t frameIndex, uint64_t sourceFrameIndex,
                        float linearMin,
                        float linearMax) -> ShadowSdsmDebugFrameData {
    const std::array<float, 2u> depths = {deviceDepthForViewDepth(linearMin),
                                          deviceDepthForViewDepth(linearMax)};
    auto seedResult = gpu.seedTextureBytes(
        pyramidTexture,
        std::as_bytes(std::span<const float>(depths.data(), depths.size())));
    EXPECT_FALSE(seedResult.hasError()) << seedResult.error();

    RenderFrameContext frameContext{};
    frameContext.frameIndex = frameIndex;
    frameContext.scene = &scene;
    frameContext.resources = &renderer.resources();
    frameContext.settings = &settings;
    frameContext.camera = camera;
    frameContext.sharedResources.sceneDepthPyramidTextures[0] = pyramidTexture;
    frameContext.sharedResources.sceneDepthPyramidLevelCount = 1u;
    frameContext.sharedResources.sceneDepthPyramidSourceFrameIndex =
        sourceFrameIndex;

    RenderGraphBuilder graph(&memory);
    graph.beginFrame(frameContext.frameIndex);
    auto buildResult =
        pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
    EXPECT_FALSE(buildResult.hasError()) << buildResult.error();
    EXPECT_TRUE(buildResult.value());
    EXPECT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());
    return frameContext.sharedResources.shadowDebugFrameData->sdsm;
  };

  const ShadowSdsmDebugFrameData fixedSdsm = buildFrame(2u, 1u, 2.0f, 35.0f);
  ASSERT_EQ(fixedSdsm.status, ShadowSdsmStatus::Active);
  ASSERT_EQ(fixedSdsm.splitCount, settings.shadow.cascadeCount);
  ASSERT_EQ(fixedSdsm.effectiveSplitDepths, fixedSdsm.fixedSplitDepths);

  const float lastFixedSplitNear =
      fixedSdsm.fixedSplitDepths[fixedSdsm.splitCount - 1u];
  const float splitActivationDepth =
      std::max((fixedSdsm.fixedRangeFar - lastFixedSplitNear) * 0.05f, 1.0f);

  const ShadowSdsmDebugFrameData activatedSdsm = buildFrame(
      3u, 2u, 2.0f, lastFixedSplitNear + splitActivationDepth * 2.0f);
  ASSERT_EQ(activatedSdsm.status, ShadowSdsmStatus::Active);
  EXPECT_FLOAT_EQ(activatedSdsm.effectiveRangeFar, activatedSdsm.fixedRangeFar);
  EXPECT_EQ(activatedSdsm.effectiveSplitDepths, activatedSdsm.fixedSplitDepths);

  const ShadowSdsmDebugFrameData heldSdsm = buildFrame(
      4u, 3u, 2.0f, lastFixedSplitNear + splitActivationDepth * 0.75f);
  ASSERT_EQ(heldSdsm.status, ShadowSdsmStatus::Active);
  EXPECT_FLOAT_EQ(heldSdsm.effectiveRangeFar, heldSdsm.fixedRangeFar);
  EXPECT_EQ(heldSdsm.effectiveSplitDepths, heldSdsm.fixedSplitDepths);

  const ShadowSdsmDebugFrameData releasedSdsm = buildFrame(
      5u, 4u, 2.0f, lastFixedSplitNear + splitActivationDepth * 0.25f);
  ASSERT_EQ(releasedSdsm.status, ShadowSdsmStatus::Active);
  EXPECT_FLOAT_EQ(releasedSdsm.effectiveRangeFar, releasedSdsm.fixedRangeFar);
  EXPECT_EQ(releasedSdsm.effectiveSplitDepths, releasedSdsm.fixedSplitDepths);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureHistogramSdsmProducesNonFixedSplitsForSkewedDistribution) {
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
  addDirectionalLightToScene(scene);

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 4u, 4u),
                        "sdsm_histogram_skewed_texture");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();
  const TextureHandle pyramidTexture = pyramidTextureResult.value();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.cascadeCount = 4u;
  settings.shadow.maxDistance = 100.0f;
  settings.shadow.sdsmMode = ShadowSdsmMode::Histogram;
  settings.shadow.sdsmTemporalBlend = 0.0f;
  settings.shadow.sdsmHistogramTrimLowPercent = 0.0f;
  settings.shadow.sdsmHistogramTrimHighPercent = 0.0f;

  const CameraFrameState camera = makeSdsmPerspectiveCamera(100.0f);
  std::vector<float> depths(4u * 4u * 2u, 0.0f);
  for (size_t tile = 0u; tile < 12u; ++tile) {
    depths[tile * 2u] = deviceDepthForViewDepth(2.0f, camera);
    depths[tile * 2u + 1u] = deviceDepthForViewDepth(6.0f, camera);
  }
  for (size_t tile = 12u; tile < 16u; ++tile) {
    depths[tile * 2u] = deviceDepthForViewDepth(20.0f, camera);
    depths[tile * 2u + 1u] = deviceDepthForViewDepth(40.0f, camera);
  }
  auto seedResult = gpu.seedTextureBytes(
      pyramidTexture,
      std::as_bytes(std::span<const float>(depths.data(), depths.size())));
  ASSERT_FALSE(seedResult.hasError()) << seedResult.error();

  const ShadowSdsmDebugFrameData sdsm = buildShadowSdsmFrame(
      pipeline, renderer, memory, scene, settings, camera, 2u, 1u,
      std::span<const TextureHandle>(&pyramidTexture, 1u));
  ASSERT_EQ(sdsm.status, ShadowSdsmStatus::Active);
  EXPECT_EQ(sdsm.histogramBucketCount,
            settings.shadow.sdsmHistogramBucketCount);
  EXPECT_EQ(sdsm.histogramValidTileCount, 16u);
  EXPECT_GT(sdsm.histogramTotalWeight, 0.0f);
  EXPECT_NE(sdsm.histogramSplitDepths, sdsm.fixedSplitDepths);
  EXPECT_EQ(sdsm.effectiveSplitDepths,
            expectedHistogramEffectiveSplitDepths(sdsm));
  EXPECT_NE(sdsm.effectiveSplitDepths, sdsm.fixedSplitDepths);
  EXPECT_NE(sdsm.effectiveSplitDepths,
            shadow_detail::computeCascadeSplitDepthsForRange(
                sdsm.effectiveRangeNear, sdsm.effectiveRangeFar,
                sdsm.splitCount, settings.shadow.splitMode,
                settings.shadow.splitLambda));
}

TEST(RenderGraphRendererTest,
     ShadowFeatureHistogramSdsmTrimSuppressesSparseFarOutliers) {
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
  addDirectionalLightToScene(scene);

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 4u, 4u),
                        "sdsm_histogram_trim_texture");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();
  const TextureHandle pyramidTexture = pyramidTextureResult.value();
  const CameraFrameState camera = makeSdsmPerspectiveCamera(100.0f);

  std::vector<float> depths(4u * 4u * 2u, 0.0f);
  for (size_t tile = 0u; tile < 15u; ++tile) {
    depths[tile * 2u] = deviceDepthForViewDepth(2.0f, camera);
    depths[tile * 2u + 1u] = deviceDepthForViewDepth(6.0f, camera);
  }
  depths[30] = deviceDepthForViewDepth(80.0f, camera);
  depths[31] = deviceDepthForViewDepth(90.0f, camera);
  auto seedResult = gpu.seedTextureBytes(
      pyramidTexture,
      std::as_bytes(std::span<const float>(depths.data(), depths.size())));
  ASSERT_FALSE(seedResult.hasError()) << seedResult.error();

  RenderSettings noTrimSettings{};
  noTrimSettings.shadow.enabled = true;
  noTrimSettings.shadow.cascadeCount = 4u;
  noTrimSettings.shadow.maxDistance = 100.0f;
  noTrimSettings.shadow.sdsmMode = ShadowSdsmMode::Histogram;
  noTrimSettings.shadow.sdsmTemporalBlend = 0.0f;

  RenderSettings trimmedSettings = noTrimSettings;
  trimmedSettings.shadow.sdsmHistogramTrimHighPercent = 10.0f;

  const ShadowSdsmDebugFrameData noTrim = buildShadowSdsmFrame(
      pipeline, renderer, memory, scene, noTrimSettings, camera, 2u, 1u,
      std::span<const TextureHandle>(&pyramidTexture, 1u));
  const ShadowSdsmDebugFrameData trimmed = buildShadowSdsmFrame(
      pipeline, renderer, memory, scene, trimmedSettings, camera, 2u, 1u,
      std::span<const TextureHandle>(&pyramidTexture, 1u));
  ASSERT_EQ(noTrim.status, ShadowSdsmStatus::Active);
  ASSERT_EQ(trimmed.status, ShadowSdsmStatus::Active);
  EXPECT_LT(trimmed.histogramTrimmedRangeFar, noTrim.histogramTrimmedRangeFar);
  EXPECT_LT(trimmed.histogramSplitDepths[3], noTrim.histogramSplitDepths[3]);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureHistogramSdsmBucketCountChangesEffectiveSplits) {
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
  addDirectionalLightToScene(scene);

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 4u, 4u),
                        "sdsm_histogram_bucket_count_texture");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();
  const TextureHandle pyramidTexture = pyramidTextureResult.value();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.cascadeCount = 4u;
  settings.shadow.maxDistance = 100.0f;
  settings.shadow.sdsmMode = ShadowSdsmMode::Histogram;
  settings.shadow.sdsmTemporalBlend = 0.0f;

  const CameraFrameState camera = makeSdsmPerspectiveCamera(100.0f);
  std::vector<float> depths(4u * 4u * 2u, 0.0f);
  for (size_t tile = 0u; tile < 12u; ++tile) {
    depths[tile * 2u] = deviceDepthForViewDepth(2.0f, camera);
    depths[tile * 2u + 1u] = deviceDepthForViewDepth(6.0f, camera);
  }
  for (size_t tile = 12u; tile < 16u; ++tile) {
    depths[tile * 2u] = deviceDepthForViewDepth(20.0f, camera);
    depths[tile * 2u + 1u] = deviceDepthForViewDepth(40.0f, camera);
  }
  auto seedResult = gpu.seedTextureBytes(
      pyramidTexture,
      std::as_bytes(std::span<const float>(depths.data(), depths.size())));
  ASSERT_FALSE(seedResult.hasError()) << seedResult.error();

  settings.shadow.sdsmHistogramBucketCount = kMinShadowSdsmHistogramBucketCount;
  const ShadowSdsmDebugFrameData coarseSdsm = buildShadowSdsmFrame(
      pipeline, renderer, memory, scene, settings, camera, 2u, 1u,
      std::span<const TextureHandle>(&pyramidTexture, 1u));
  ASSERT_EQ(coarseSdsm.status, ShadowSdsmStatus::Active);

  settings.shadow.sdsmHistogramBucketCount = kMaxShadowSdsmHistogramBucketCount;
  const ShadowSdsmDebugFrameData fineSdsm = buildShadowSdsmFrame(
      pipeline, renderer, memory, scene, settings, camera, 3u, 2u,
      std::span<const TextureHandle>(&pyramidTexture, 1u));
  ASSERT_EQ(fineSdsm.status, ShadowSdsmStatus::Active);

  EXPECT_NE(coarseSdsm.histogramSplitDepths, fineSdsm.histogramSplitDepths);
  EXPECT_NE(coarseSdsm.effectiveSplitDepths, fineSdsm.effectiveSplitDepths);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureHistogramSdsmSelectsFinestPyramidLevelWithinTexelBudget) {
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
  addDirectionalLightToScene(scene);

  auto level0Result =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 128u, 64u),
                        "sdsm_histogram_level0");
  ASSERT_FALSE(level0Result.hasError()) << level0Result.error();
  auto level1Result =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 64u, 32u),
                        "sdsm_histogram_level1");
  ASSERT_FALSE(level1Result.hasError()) << level1Result.error();
  const TextureHandle level0 = level0Result.value();
  const TextureHandle level1 = level1Result.value();

  const CameraFrameState camera = makeSdsmPerspectiveCamera(100.0f);
  std::vector<float> level0Depths(128u * 64u * 2u, 0.0f);
  std::vector<float> level1Depths(64u * 32u * 2u, 0.0f);
  for (size_t tile = 0u; tile < level0Depths.size() / 2u; ++tile) {
    level0Depths[tile * 2u] = deviceDepthForViewDepth(2.0f, camera);
    level0Depths[tile * 2u + 1u] = deviceDepthForViewDepth(8.0f, camera);
  }
  for (size_t tile = 0u; tile < level1Depths.size() / 2u; ++tile) {
    level1Depths[tile * 2u] = deviceDepthForViewDepth(2.0f, camera);
    level1Depths[tile * 2u + 1u] = deviceDepthForViewDepth(8.0f, camera);
  }
  auto seed0Result = gpu.seedTextureBytes(
      level0, std::as_bytes(std::span<const float>(level0Depths.data(),
                                                   level0Depths.size())));
  ASSERT_FALSE(seed0Result.hasError()) << seed0Result.error();
  auto seed1Result = gpu.seedTextureBytes(
      level1, std::as_bytes(std::span<const float>(level1Depths.data(),
                                                   level1Depths.size())));
  ASSERT_FALSE(seed1Result.hasError()) << seed1Result.error();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.cascadeCount = 4u;
  settings.shadow.maxDistance = 100.0f;
  settings.shadow.sdsmMode = ShadowSdsmMode::Histogram;
  settings.shadow.sdsmTemporalBlend = 0.0f;

  const std::array<TextureHandle, 2u> pyramidTextures = {level0, level1};
  const ShadowSdsmDebugFrameData sdsm =
      buildShadowSdsmFrame(pipeline, renderer, memory, scene, settings, camera,
                           2u, 1u, pyramidTextures);
  ASSERT_EQ(sdsm.status, ShadowSdsmStatus::Active);
  EXPECT_EQ(sdsm.histogramSourceLevel, 1u);
  EXPECT_EQ(sdsm.histogramSourceDimensions, glm::uvec2(64u, 32u));
}

TEST(
    RenderGraphRendererTest,
    ShadowFeatureHistogramSdsmReusesCachedSplitsThenFallsBackWhenCacheExpires) {
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
  addDirectionalLightToScene(scene);

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 4u, 4u),
                        "sdsm_histogram_cache_texture");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();
  const TextureHandle pyramidTexture = pyramidTextureResult.value();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.cascadeCount = 4u;
  settings.shadow.maxDistance = 100.0f;
  settings.shadow.sdsmMode = ShadowSdsmMode::Histogram;
  settings.shadow.sdsmTemporalBlend = 0.0f;

  const CameraFrameState camera = makeSdsmPerspectiveCamera(100.0f);
  std::vector<float> depths(4u * 4u * 2u, 0.0f);
  for (size_t tile = 0u; tile < 12u; ++tile) {
    depths[tile * 2u] = deviceDepthForViewDepth(2.0f, camera);
    depths[tile * 2u + 1u] = deviceDepthForViewDepth(6.0f, camera);
  }
  for (size_t tile = 12u; tile < 16u; ++tile) {
    depths[tile * 2u] = deviceDepthForViewDepth(20.0f, camera);
    depths[tile * 2u + 1u] = deviceDepthForViewDepth(30.0f, camera);
  }
  auto seedResult = gpu.seedTextureBytes(
      pyramidTexture,
      std::as_bytes(std::span<const float>(depths.data(), depths.size())));
  ASSERT_FALSE(seedResult.hasError()) << seedResult.error();

  const ShadowSdsmDebugFrameData active = buildShadowSdsmFrame(
      pipeline, renderer, memory, scene, settings, camera, 2u, 1u,
      std::span<const TextureHandle>(&pyramidTexture, 1u));
  ASSERT_EQ(active.status, ShadowSdsmStatus::Active);

  const ShadowSdsmDebugFrameData reused =
      buildShadowSdsmFrame(pipeline, renderer, memory, scene, settings, camera,
                           3u, std::nullopt, {});
  EXPECT_EQ(reused.status, ShadowSdsmStatus::Unavailable);
  EXPECT_FALSE(reused.fixedFallbackActive);
  EXPECT_EQ(reused.effectiveSplitDepths, active.effectiveSplitDepths);

  const ShadowSdsmDebugFrameData expired =
      buildShadowSdsmFrame(pipeline, renderer, memory, scene, settings, camera,
                           4u, std::nullopt, {});
  EXPECT_EQ(expired.status, ShadowSdsmStatus::Unavailable);
  EXPECT_TRUE(expired.fixedFallbackActive);
  EXPECT_EQ(expired.effectiveSplitDepths, expired.fixedSplitDepths);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureHistogramSdsmSmoothsInternalSplitDepthsTemporally) {
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
  addDirectionalLightToScene(scene);

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 4u, 4u),
                        "sdsm_histogram_smoothing_texture");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();
  const TextureHandle pyramidTexture = pyramidTextureResult.value();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.cascadeCount = 4u;
  settings.shadow.maxDistance = 100.0f;
  settings.shadow.sdsmMode = ShadowSdsmMode::Histogram;
  settings.shadow.sdsmTemporalBlend = 0.5f;

  const CameraFrameState camera = makeSdsmPerspectiveCamera(100.0f);
  std::vector<float> nearDepths(4u * 4u * 2u, 0.0f);
  std::vector<float> farDepths(4u * 4u * 2u, 0.0f);
  for (size_t tile = 0u; tile < nearDepths.size() / 2u; ++tile) {
    nearDepths[tile * 2u] = deviceDepthForViewDepth(2.0f, camera);
    nearDepths[tile * 2u + 1u] = deviceDepthForViewDepth(6.0f, camera);
    farDepths[tile * 2u] = deviceDepthForViewDepth(20.0f, camera);
    farDepths[tile * 2u + 1u] = deviceDepthForViewDepth(40.0f, camera);
  }

  auto seedNearResult = gpu.seedTextureBytes(
      pyramidTexture, std::as_bytes(std::span<const float>(nearDepths.data(),
                                                           nearDepths.size())));
  ASSERT_FALSE(seedNearResult.hasError()) << seedNearResult.error();
  const ShadowSdsmDebugFrameData nearSdsm = buildShadowSdsmFrame(
      pipeline, renderer, memory, scene, settings, camera, 2u, 1u,
      std::span<const TextureHandle>(&pyramidTexture, 1u));
  ASSERT_EQ(nearSdsm.status, ShadowSdsmStatus::Active);

  auto seedFarResult = gpu.seedTextureBytes(
      pyramidTexture, std::as_bytes(std::span<const float>(farDepths.data(),
                                                           farDepths.size())));
  ASSERT_FALSE(seedFarResult.hasError()) << seedFarResult.error();
  const ShadowSdsmDebugFrameData farSdsm = buildShadowSdsmFrame(
      pipeline, renderer, memory, scene, settings, camera, 3u, 2u,
      std::span<const TextureHandle>(&pyramidTexture, 1u));
  ASSERT_EQ(farSdsm.status, ShadowSdsmStatus::Active);
  EXPECT_GT(farSdsm.smoothedLinearMax, nearSdsm.smoothedLinearMax);
  EXPECT_LT(farSdsm.smoothedLinearMax, farSdsm.histogramTrimmedRangeFar);
  EXPECT_EQ(farSdsm.effectiveSplitDepths,
            expectedHistogramEffectiveSplitDepths(farSdsm));
  EXPECT_NE(farSdsm.effectiveSplitDepths,
            shadow_detail::computeCascadeSplitDepthsForRange(
                farSdsm.effectiveRangeNear, farSdsm.effectiveRangeFar,
                farSdsm.splitCount, settings.shadow.splitMode,
                settings.shadow.splitLambda));
}

TEST(RenderGraphRendererTest,
     ShadowFeatureHistogramSdsmUsesHistogramDistributionForEffectiveSplits) {
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
  addDirectionalLightToScene(scene);

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 4u, 4u),
                        "sdsm_histogram_range_gate_texture");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();
  const TextureHandle pyramidTexture = pyramidTextureResult.value();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.cascadeCount = 4u;
  settings.shadow.maxDistance = 150.0f;
  settings.shadow.sdsmMode = ShadowSdsmMode::Histogram;
  settings.shadow.sdsmTemporalBlend = 0.85f;
  settings.shadow.sdsmHistogramTrimHighPercent = 0.0f;

  const CameraFrameState camera = makeSdsmPerspectiveCamera(150.0f);
  std::vector<float> nearDepths(4u * 4u * 2u, 0.0f);
  std::vector<float> outlierDepths(4u * 4u * 2u, 0.0f);
  for (size_t tile = 0u; tile < nearDepths.size() / 2u; ++tile) {
    nearDepths[tile * 2u] = deviceDepthForViewDepth(2.0f, camera);
    nearDepths[tile * 2u + 1u] = deviceDepthForViewDepth(6.0f, camera);
    outlierDepths[tile * 2u] = deviceDepthForViewDepth(2.0f, camera);
    outlierDepths[tile * 2u + 1u] = deviceDepthForViewDepth(6.0f, camera);
  }
  for (size_t tile = 12u; tile < 16u; ++tile) {
    outlierDepths[tile * 2u] = deviceDepthForViewDepth(70.0f, camera);
    outlierDepths[tile * 2u + 1u] = deviceDepthForViewDepth(80.0f, camera);
  }

  auto seedNearResult = gpu.seedTextureBytes(
      pyramidTexture, std::as_bytes(std::span<const float>(nearDepths.data(),
                                                           nearDepths.size())));
  ASSERT_FALSE(seedNearResult.hasError()) << seedNearResult.error();
  const ShadowSdsmDebugFrameData nearSdsm = buildShadowSdsmFrame(
      pipeline, renderer, memory, scene, settings, camera, 2u, 1u,
      std::span<const TextureHandle>(&pyramidTexture, 1u));
  ASSERT_EQ(nearSdsm.status, ShadowSdsmStatus::Active);

  auto seedOutlierResult = gpu.seedTextureBytes(
      pyramidTexture, std::as_bytes(std::span<const float>(
                          outlierDepths.data(), outlierDepths.size())));
  ASSERT_FALSE(seedOutlierResult.hasError()) << seedOutlierResult.error();
  const ShadowSdsmDebugFrameData outlierSdsm = buildShadowSdsmFrame(
      pipeline, renderer, memory, scene, settings, camera, 3u, 2u,
      std::span<const TextureHandle>(&pyramidTexture, 1u));
  ASSERT_EQ(outlierSdsm.status, ShadowSdsmStatus::Active);
  EXPECT_FLOAT_EQ(outlierSdsm.effectiveRangeFar, outlierSdsm.fixedRangeFar);
  const auto expectedEffectiveSplits =
      expectedHistogramEffectiveSplitDepths(outlierSdsm);
  EXPECT_EQ(outlierSdsm.effectiveSplitDepths, expectedEffectiveSplits);
  EXPECT_NE(outlierSdsm.effectiveSplitDepths, outlierSdsm.fixedSplitDepths);
  EXPECT_GT(outlierSdsm.histogramTrimmedRangeFar,
            outlierSdsm.fixedSplitDepths[outlierSdsm.splitCount - 1u]);
  EXPECT_LT(outlierSdsm.smoothedLinearMax,
            outlierSdsm.fixedSplitDepths[outlierSdsm.splitCount - 1u]);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureHistogramSdsmKeepsFixedCoverageForNearOnlyContent) {
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
  addDirectionalLightToScene(scene);

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 4u, 4u),
                        "sdsm_histogram_near_only_texture");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();
  const TextureHandle pyramidTexture = pyramidTextureResult.value();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.cascadeCount = 4u;
  settings.shadow.maxDistance = 150.0f;
  settings.shadow.sdsmMode = ShadowSdsmMode::Histogram;
  settings.shadow.sdsmTemporalBlend = 0.5f;
  settings.shadow.sdsmHistogramTrimLowPercent = 0.0f;
  settings.shadow.sdsmHistogramTrimHighPercent = 0.0f;

  const CameraFrameState camera = makeSdsmPerspectiveCamera(150.0f);
  std::vector<float> depths(4u * 4u * 2u, 0.0f);
  for (size_t tile = 0u; tile < depths.size() / 2u; ++tile) {
    depths[tile * 2u] = deviceDepthForViewDepth(2.0f, camera);
    depths[tile * 2u + 1u] = deviceDepthForViewDepth(6.0f, camera);
  }

  auto seedResult = gpu.seedTextureBytes(
      pyramidTexture,
      std::as_bytes(std::span<const float>(depths.data(), depths.size())));
  ASSERT_FALSE(seedResult.hasError()) << seedResult.error();

  const ShadowSdsmDebugFrameData sdsm = buildShadowSdsmFrame(
      pipeline, renderer, memory, scene, settings, camera, 2u, 1u,
      std::span<const TextureHandle>(&pyramidTexture, 1u));
  ASSERT_EQ(sdsm.status, ShadowSdsmStatus::Active);
  const float minimumFarDepth = sdsm.fixedSplitDepths[sdsm.splitCount - 1u];
  EXPECT_LT(sdsm.smoothedLinearMax, minimumFarDepth);
  EXPECT_FLOAT_EQ(sdsm.effectiveRangeFar, sdsm.fixedRangeFar);
  EXPECT_EQ(sdsm.effectiveSplitDepths, sdsm.fixedSplitDepths);
  EXPECT_EQ(sdsm.effectiveSplitDepths,
            shadow_detail::computeCascadeSplitDepthsForRange(
                sdsm.effectiveRangeNear, sdsm.effectiveRangeFar,
                sdsm.splitCount, settings.shadow.splitMode,
                settings.shadow.splitLambda));
}

TEST(
    RenderGraphRendererTest,
    ShadowFeatureHistogramSdsmKeepsOrthographicRangeFixedWhileRedistributingSplits) {
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
  addDirectionalLightToScene(scene);

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 4u, 4u),
                        "sdsm_histogram_ortho_texture");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();
  const TextureHandle pyramidTexture = pyramidTextureResult.value();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.cascadeCount = 4u;
  settings.shadow.maxDistance = 150.0f;
  settings.shadow.sdsmMode = ShadowSdsmMode::Histogram;
  settings.shadow.sdsmTemporalBlend = 0.0f;
  settings.shadow.sdsmHistogramTrimLowPercent = 0.0f;
  settings.shadow.sdsmHistogramTrimHighPercent = 0.0f;

  const CameraFrameState camera = makeSdsmOrthographicCamera(150.0f, 48.0f);
  std::vector<float> depths(4u * 4u * 2u, 0.0f);
  for (size_t tile = 0u; tile < depths.size() / 2u; ++tile) {
    depths[tile * 2u] = deviceDepthForViewDepth(2.0f, camera);
    depths[tile * 2u + 1u] = deviceDepthForViewDepth(100.0f, camera);
  }

  auto seedResult = gpu.seedTextureBytes(
      pyramidTexture,
      std::as_bytes(std::span<const float>(depths.data(), depths.size())));
  ASSERT_FALSE(seedResult.hasError()) << seedResult.error();

  const ShadowSdsmDebugFrameData sdsm = buildShadowSdsmFrame(
      pipeline, renderer, memory, scene, settings, camera, 2u, 1u,
      std::span<const TextureHandle>(&pyramidTexture, 1u));
  ASSERT_EQ(sdsm.status, ShadowSdsmStatus::Active);
  EXPECT_FALSE(sdsm.fixedFallbackActive);
  EXPECT_FLOAT_EQ(sdsm.effectiveRangeFar, sdsm.fixedRangeFar);
  EXPECT_EQ(sdsm.effectiveSplitDepths,
            expectedHistogramEffectiveSplitDepths(sdsm));
}

TEST(
    RenderGraphRendererTest,
    ShadowFeatureHistogramSdsmKeepsFloorClampedEffectiveSplitsStableAcrossNearOnlyUpdates) {
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
  addDirectionalLightToScene(scene);

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 4u, 4u),
                        "sdsm_histogram_floor_stability_texture");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();
  const TextureHandle pyramidTexture = pyramidTextureResult.value();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.cascadeCount = 4u;
  settings.shadow.maxDistance = 150.0f;
  settings.shadow.sdsmMode = ShadowSdsmMode::Histogram;
  settings.shadow.sdsmTemporalBlend = 0.0f;
  settings.shadow.sdsmHistogramTrimLowPercent = 0.0f;
  settings.shadow.sdsmHistogramTrimHighPercent = 0.0f;

  const CameraFrameState camera = makeSdsmPerspectiveCamera(150.0f);
  std::vector<float> shallowDepths(4u * 4u * 2u, 0.0f);
  std::vector<float> deeperNearDepths(4u * 4u * 2u, 0.0f);
  for (size_t tile = 0u; tile < shallowDepths.size() / 2u; ++tile) {
    shallowDepths[tile * 2u] = deviceDepthForViewDepth(2.0f, camera);
    shallowDepths[tile * 2u + 1u] = deviceDepthForViewDepth(6.0f, camera);
    deeperNearDepths[tile * 2u] = deviceDepthForViewDepth(5.0f, camera);
    deeperNearDepths[tile * 2u + 1u] = deviceDepthForViewDepth(20.0f, camera);
  }

  auto seedShallowResult = gpu.seedTextureBytes(
      pyramidTexture, std::as_bytes(std::span<const float>(
                          shallowDepths.data(), shallowDepths.size())));
  ASSERT_FALSE(seedShallowResult.hasError()) << seedShallowResult.error();
  const ShadowSdsmDebugFrameData firstSdsm = buildShadowSdsmFrame(
      pipeline, renderer, memory, scene, settings, camera, 2u, 1u,
      std::span<const TextureHandle>(&pyramidTexture, 1u));
  ASSERT_EQ(firstSdsm.status, ShadowSdsmStatus::Active);

  auto seedDeeperNearResult = gpu.seedTextureBytes(
      pyramidTexture, std::as_bytes(std::span<const float>(
                          deeperNearDepths.data(), deeperNearDepths.size())));
  ASSERT_FALSE(seedDeeperNearResult.hasError()) << seedDeeperNearResult.error();
  const ShadowSdsmDebugFrameData secondSdsm = buildShadowSdsmFrame(
      pipeline, renderer, memory, scene, settings, camera, 3u, 2u,
      std::span<const TextureHandle>(&pyramidTexture, 1u));
  ASSERT_EQ(secondSdsm.status, ShadowSdsmStatus::Active);

  const auto expectedFloorClampedSplits =
      shadow_detail::computeCascadeSplitDepthsForRange(
          secondSdsm.effectiveRangeNear, secondSdsm.effectiveRangeFar,
          secondSdsm.splitCount, settings.shadow.splitMode,
          settings.shadow.splitLambda);
  EXPECT_FLOAT_EQ(firstSdsm.effectiveRangeFar, firstSdsm.fixedRangeFar);
  EXPECT_FLOAT_EQ(secondSdsm.effectiveRangeFar, secondSdsm.fixedRangeFar);
  EXPECT_NEAR(secondSdsm.effectiveRangeFar, firstSdsm.effectiveRangeFar,
              1.0e-4f);
  EXPECT_NE(secondSdsm.histogramSplitDepths, firstSdsm.histogramSplitDepths);
  EXPECT_EQ(firstSdsm.effectiveSplitDepths, expectedFloorClampedSplits);
  EXPECT_EQ(secondSdsm.effectiveSplitDepths, expectedFloorClampedSplits);
  EXPECT_EQ(secondSdsm.effectiveSplitDepths, firstSdsm.effectiveSplitDepths);
}

TEST(
    RenderGraphRendererTest,
    ShadowFeatureHistogramSdsmKeepsEffectiveHistogramSplitsStableAcrossTemporalUpdates) {
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
  addDirectionalLightToScene(scene);

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 4u, 4u),
                        "sdsm_histogram_fixed_range_texture");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();
  const TextureHandle pyramidTexture = pyramidTextureResult.value();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.cascadeCount = 4u;
  settings.shadow.maxDistance = 150.0f;
  settings.shadow.sdsmMode = ShadowSdsmMode::Histogram;
  settings.shadow.sdsmTemporalBlend = 0.85f;

  const CameraFrameState camera = makeSdsmPerspectiveCamera(150.0f);
  std::vector<float> depths(4u * 4u * 2u, 0.0f);
  for (size_t tile = 0u; tile < depths.size() / 2u; ++tile) {
    depths[tile * 2u] = deviceDepthForViewDepth(2.0f, camera);
    depths[tile * 2u + 1u] = deviceDepthForViewDepth(60.0f, camera);
  }

  auto seedResult = gpu.seedTextureBytes(
      pyramidTexture,
      std::as_bytes(std::span<const float>(depths.data(), depths.size())));
  ASSERT_FALSE(seedResult.hasError()) << seedResult.error();

  const ShadowSdsmDebugFrameData firstSdsm = buildShadowSdsmFrame(
      pipeline, renderer, memory, scene, settings, camera, 2u, 1u,
      std::span<const TextureHandle>(&pyramidTexture, 1u));
  ASSERT_EQ(firstSdsm.status, ShadowSdsmStatus::Active);
  EXPECT_FLOAT_EQ(firstSdsm.effectiveRangeFar, firstSdsm.fixedRangeFar);
  EXPECT_EQ(firstSdsm.effectiveSplitDepths,
            expectedHistogramEffectiveSplitDepths(firstSdsm));

  const ShadowSdsmDebugFrameData secondSdsm = buildShadowSdsmFrame(
      pipeline, renderer, memory, scene, settings, camera, 3u, 2u,
      std::span<const TextureHandle>(&pyramidTexture, 1u));
  ASSERT_EQ(secondSdsm.status, ShadowSdsmStatus::Active);
  EXPECT_FLOAT_EQ(secondSdsm.effectiveRangeFar, secondSdsm.fixedRangeFar);
  EXPECT_EQ(secondSdsm.effectiveSplitDepths,
            expectedHistogramEffectiveSplitDepths(secondSdsm));
  EXPECT_NE(secondSdsm.effectiveSplitDepths,
            shadow_detail::computeCascadeSplitDepthsForRange(
                secondSdsm.effectiveRangeNear, secondSdsm.effectiveRangeFar,
                secondSdsm.splitCount, settings.shadow.splitMode,
                settings.shadow.splitLambda));
  EXPECT_NEAR(secondSdsm.effectiveRangeFar, firstSdsm.effectiveRangeFar,
              1.0e-5f);
  EXPECT_EQ(secondSdsm.histogramSplitDepths, firstSdsm.histogramSplitDepths);
  EXPECT_EQ(secondSdsm.effectiveSplitDepths, firstSdsm.effectiveSplitDepths);
  EXPECT_GT(secondSdsm.smoothedLinearMax,
            secondSdsm.fixedSplitDepths[secondSdsm.splitCount - 1u]);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureHistogramSdsmIgnoresClearDepthContaminatedTileMaxima) {
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
  addDirectionalLightToScene(scene);

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 4u, 4u),
                        "sdsm_histogram_clear_contamination_texture");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();
  const TextureHandle pyramidTexture = pyramidTextureResult.value();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.cascadeCount = 4u;
  settings.shadow.maxDistance = 150.0f;
  settings.shadow.sdsmMode = ShadowSdsmMode::Histogram;
  settings.shadow.sdsmTemporalBlend = 0.85f;
  settings.shadow.sdsmHistogramTrimLowPercent = 0.0f;
  settings.shadow.sdsmHistogramTrimHighPercent = 0.0f;

  const CameraFrameState camera = makeSdsmPerspectiveCamera(150.0f);
  std::vector<float> stableDepths(4u * 4u * 2u, 0.0f);
  std::vector<float> contaminatedDepths = stableDepths;
  const float stableMinDepth = deviceDepthForViewDepth(3.0f, camera);
  const float stableMaxDepth = deviceDepthForViewDepth(5.5f, camera);
  for (size_t tile = 0u; tile < stableDepths.size() / 2u; ++tile) {
    stableDepths[tile * 2u] = stableMinDepth;
    stableDepths[tile * 2u + 1u] = stableMaxDepth;
    contaminatedDepths[tile * 2u] = stableMinDepth;
    contaminatedDepths[tile * 2u + 1u] = stableMaxDepth;
  }
  for (size_t tile = 0u; tile < 12u; ++tile) {
    contaminatedDepths[tile * 2u + 1u] = 1.0f;
  }

  auto seedStableResult = gpu.seedTextureBytes(
      pyramidTexture, std::as_bytes(std::span<const float>(
                          stableDepths.data(), stableDepths.size())));
  ASSERT_FALSE(seedStableResult.hasError()) << seedStableResult.error();
  const ShadowSdsmDebugFrameData firstSdsm = buildShadowSdsmFrame(
      pipeline, renderer, memory, scene, settings, camera, 2u, 1u,
      std::span<const TextureHandle>(&pyramidTexture, 1u));
  ASSERT_EQ(firstSdsm.status, ShadowSdsmStatus::Active);

  auto seedContaminatedResult = gpu.seedTextureBytes(
      pyramidTexture,
      std::as_bytes(std::span<const float>(contaminatedDepths.data(),
                                           contaminatedDepths.size())));
  ASSERT_FALSE(seedContaminatedResult.hasError())
      << seedContaminatedResult.error();
  const ShadowSdsmDebugFrameData secondSdsm = buildShadowSdsmFrame(
      pipeline, renderer, memory, scene, settings, camera, 3u, 2u,
      std::span<const TextureHandle>(&pyramidTexture, 1u));
  ASSERT_EQ(secondSdsm.status, ShadowSdsmStatus::Active);
  ASSERT_FLOAT_EQ(secondSdsm.rawDeviceMax, 1.0f);
  ASSERT_GT(secondSdsm.rawLinearMax, firstSdsm.rawLinearMax);
  EXPECT_FLOAT_EQ(secondSdsm.histogramTrimmedRangeNear,
                  firstSdsm.histogramTrimmedRangeNear);
  EXPECT_FLOAT_EQ(secondSdsm.histogramTrimmedRangeFar,
                  firstSdsm.histogramTrimmedRangeFar);
  EXPECT_EQ(secondSdsm.histogramSplitDepths, firstSdsm.histogramSplitDepths);
  EXPECT_EQ(secondSdsm.effectiveSplitDepths, firstSdsm.effectiveSplitDepths);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureHistogramSdsmTreatsClearOnlySourceAsValidFixedRange) {
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
  addDirectionalLightToScene(scene);

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 4u, 4u),
                        "sdsm_histogram_clear_only_texture");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();
  const TextureHandle pyramidTexture = pyramidTextureResult.value();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.cascadeCount = 4u;
  settings.shadow.maxDistance = 150.0f;
  settings.shadow.sdsmMode = ShadowSdsmMode::Histogram;
  settings.shadow.sdsmTemporalBlend = 0.85f;
  settings.shadow.sdsmHistogramTrimLowPercent = 0.0f;
  settings.shadow.sdsmHistogramTrimHighPercent = 0.0f;

  const CameraFrameState camera = makeSdsmPerspectiveCamera(1699.661f);
  std::vector<float> nearDepths(4u * 4u * 2u, 0.0f);
  std::vector<float> clearOnlyDepths(4u * 4u * 2u, 1.0f);
  for (size_t tile = 0u; tile < nearDepths.size() / 2u; ++tile) {
    nearDepths[tile * 2u] = deviceDepthForViewDepth(2.0f, camera);
    nearDepths[tile * 2u + 1u] = deviceDepthForViewDepth(6.0f, camera);
  }

  auto seedNearResult = gpu.seedTextureBytes(
      pyramidTexture, std::as_bytes(std::span<const float>(nearDepths.data(),
                                                           nearDepths.size())));
  ASSERT_FALSE(seedNearResult.hasError()) << seedNearResult.error();
  const ShadowSdsmDebugFrameData firstSdsm = buildShadowSdsmFrame(
      pipeline, renderer, memory, scene, settings, camera, 2u, 1u,
      std::span<const TextureHandle>(&pyramidTexture, 1u));
  ASSERT_EQ(firstSdsm.status, ShadowSdsmStatus::Active);
  ASSERT_FLOAT_EQ(firstSdsm.effectiveRangeFar, firstSdsm.fixedRangeFar);

  auto seedClearOnlyResult = gpu.seedTextureBytes(
      pyramidTexture, std::as_bytes(std::span<const float>(
                          clearOnlyDepths.data(), clearOnlyDepths.size())));
  ASSERT_FALSE(seedClearOnlyResult.hasError()) << seedClearOnlyResult.error();
  const ShadowSdsmDebugFrameData secondSdsm = buildShadowSdsmFrame(
      pipeline, renderer, memory, scene, settings, camera, 3u, 2u,
      std::span<const TextureHandle>(&pyramidTexture, 1u));
  ASSERT_EQ(secondSdsm.status, ShadowSdsmStatus::Active);
  EXPECT_FALSE(secondSdsm.fixedFallbackActive);
  EXPECT_FLOAT_EQ(secondSdsm.rawDeviceMin, 1.0f);
  EXPECT_FLOAT_EQ(secondSdsm.rawDeviceMax, 1.0f);
  EXPECT_GT(secondSdsm.rawLinearMin, secondSdsm.fixedRangeFar);
  EXPECT_GT(secondSdsm.rawLinearMax, secondSdsm.fixedRangeFar);
  EXPECT_GT(secondSdsm.smoothedLinearMax, secondSdsm.fixedRangeFar);
  EXPECT_FLOAT_EQ(secondSdsm.effectiveRangeFar, secondSdsm.fixedRangeFar);
  EXPECT_EQ(secondSdsm.histogramSplitDepths, secondSdsm.fixedSplitDepths);
  for (uint32_t cascadeIndex = 0u;
       cascadeIndex <=
       std::clamp(secondSdsm.splitCount, 1u, kMaxShadowCascades);
       ++cascadeIndex) {
    EXPECT_NEAR(secondSdsm.effectiveSplitDepths[cascadeIndex],
                secondSdsm.fixedSplitDepths[cascadeIndex], 1.0e-5f);
  }
  EXPECT_EQ(secondSdsm.histogramValidTileCount, 16u);
  EXPECT_FLOAT_EQ(secondSdsm.histogramTotalWeight, 0.0f);
}

TEST(
    RenderGraphRendererTest,
    ShadowFeatureHistogramSdsmKeepsFixedCoverageWhenTargetHitsFarCascadeFloor) {
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
  addDirectionalLightToScene(scene);

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 4u, 4u),
                        "sdsm_histogram_floor_stability_texture");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();
  const TextureHandle pyramidTexture = pyramidTextureResult.value();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.cascadeCount = 4u;
  settings.shadow.maxDistance = 150.0f;
  settings.shadow.sdsmMode = ShadowSdsmMode::Histogram;
  settings.shadow.sdsmTemporalBlend = 0.85f;
  settings.shadow.sdsmHistogramTrimLowPercent = 0.0f;
  settings.shadow.sdsmHistogramTrimHighPercent = 0.0f;

  const CameraFrameState camera = makeSdsmPerspectiveCamera(150.0f);
  std::vector<float> farDepths(4u * 4u * 2u, 0.0f);
  std::vector<float> nearDepths(4u * 4u * 2u, 0.0f);
  for (size_t tile = 0u; tile < farDepths.size() / 2u; ++tile) {
    farDepths[tile * 2u] = deviceDepthForViewDepth(2.0f, camera);
    farDepths[tile * 2u + 1u] = deviceDepthForViewDepth(130.0f, camera);
    nearDepths[tile * 2u] = deviceDepthForViewDepth(2.0f, camera);
    nearDepths[tile * 2u + 1u] = deviceDepthForViewDepth(6.0f, camera);
  }

  auto seedFarResult = gpu.seedTextureBytes(
      pyramidTexture, std::as_bytes(std::span<const float>(farDepths.data(),
                                                           farDepths.size())));
  ASSERT_FALSE(seedFarResult.hasError()) << seedFarResult.error();
  const ShadowSdsmDebugFrameData firstSdsm = buildShadowSdsmFrame(
      pipeline, renderer, memory, scene, settings, camera, 2u, 1u,
      std::span<const TextureHandle>(&pyramidTexture, 1u));
  ASSERT_EQ(firstSdsm.status, ShadowSdsmStatus::Active);
  EXPECT_FLOAT_EQ(firstSdsm.effectiveRangeFar, firstSdsm.fixedRangeFar);

  auto seedNearResult = gpu.seedTextureBytes(
      pyramidTexture, std::as_bytes(std::span<const float>(nearDepths.data(),
                                                           nearDepths.size())));
  ASSERT_FALSE(seedNearResult.hasError()) << seedNearResult.error();
  const ShadowSdsmDebugFrameData secondSdsm = buildShadowSdsmFrame(
      pipeline, renderer, memory, scene, settings, camera, 3u, 2u,
      std::span<const TextureHandle>(&pyramidTexture, 1u));
  ASSERT_EQ(secondSdsm.status, ShadowSdsmStatus::Active);
  EXPECT_FLOAT_EQ(secondSdsm.effectiveRangeFar, secondSdsm.fixedRangeFar);
  EXPECT_FLOAT_EQ(secondSdsm.effectiveRangeFar, firstSdsm.effectiveRangeFar);
  EXPECT_EQ(secondSdsm.effectiveSplitDepths, secondSdsm.fixedSplitDepths);

  const float minimumFarDepth =
      secondSdsm.fixedSplitDepths[secondSdsm.splitCount - 1u];
  ShadowSdsmDebugFrameData currentSdsm = secondSdsm;
  bool sawFloorClampedTarget = currentSdsm.smoothedLinearMax < minimumFarDepth;
  for (uint64_t frameIndex = 4u; frameIndex <= 40u; ++frameIndex) {
    currentSdsm = buildShadowSdsmFrame(
        pipeline, renderer, memory, scene, settings, camera, frameIndex,
        frameIndex - 1u, std::span<const TextureHandle>(&pyramidTexture, 1u));
    ASSERT_EQ(currentSdsm.status, ShadowSdsmStatus::Active);
    sawFloorClampedTarget = sawFloorClampedTarget ||
                            currentSdsm.smoothedLinearMax < minimumFarDepth;
  }

  EXPECT_TRUE(sawFloorClampedTarget);
  EXPECT_LT(currentSdsm.smoothedLinearMax, minimumFarDepth);
  EXPECT_FLOAT_EQ(currentSdsm.effectiveRangeFar, currentSdsm.fixedRangeFar);
  EXPECT_EQ(currentSdsm.effectiveSplitDepths, currentSdsm.fixedSplitDepths);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureHistogramSdsmDoesNotWarnDuringStableNearOnlyUpdates) {
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
  addDirectionalLightToScene(scene);

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 4u, 4u),
                        "sdsm_histogram_stable_log_texture");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();
  const TextureHandle pyramidTexture = pyramidTextureResult.value();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.cascadeCount = 4u;
  settings.shadow.maxDistance = 150.0f;
  settings.shadow.sdsmMode = ShadowSdsmMode::Histogram;
  settings.shadow.sdsmTemporalBlend = 0.85f;
  settings.shadow.sdsmHistogramTrimLowPercent = 0.0f;
  settings.shadow.sdsmHistogramTrimHighPercent = 0.0f;
  settings.shadow.debug.diagnosticLogLevel = LogLevel::Debug;

  const CameraFrameState camera = makeSdsmPerspectiveCamera(150.0f);
  std::vector<float> farDepths(4u * 4u * 2u, 0.0f);
  std::vector<float> nearDepths(4u * 4u * 2u, 0.0f);
  for (size_t tile = 0u; tile < farDepths.size() / 2u; ++tile) {
    farDepths[tile * 2u] = deviceDepthForViewDepth(2.0f, camera);
    farDepths[tile * 2u + 1u] = deviceDepthForViewDepth(130.0f, camera);
    nearDepths[tile * 2u] = deviceDepthForViewDepth(2.0f, camera);
    nearDepths[tile * 2u + 1u] = deviceDepthForViewDepth(6.0f, camera);
  }

  auto seedFarResult = gpu.seedTextureBytes(
      pyramidTexture, std::as_bytes(std::span<const float>(farDepths.data(),
                                                           farDepths.size())));
  ASSERT_FALSE(seedFarResult.hasError()) << seedFarResult.error();
  const ShadowSdsmDebugFrameData firstSdsm = buildShadowSdsmFrame(
      pipeline, renderer, memory, scene, settings, camera, 2u, 1u,
      std::span<const TextureHandle>(&pyramidTexture, 1u));
  ASSERT_EQ(firstSdsm.status, ShadowSdsmStatus::Active);

  auto seedNearResult = gpu.seedTextureBytes(
      pyramidTexture, std::as_bytes(std::span<const float>(nearDepths.data(),
                                                           nearDepths.size())));
  ASSERT_FALSE(seedNearResult.hasError()) << seedNearResult.error();

  const float minimumFarDepth =
      firstSdsm.fixedSplitDepths[firstSdsm.splitCount - 1u];
  ShadowSdsmDebugFrameData currentSdsm = firstSdsm;
  bool sawSmoothedTargetEnterFarFloor =
      currentSdsm.smoothedLinearMax < minimumFarDepth;
  uint64_t frameIndex = 3u;
  for (; frameIndex <= 64u; ++frameIndex) {
    currentSdsm = buildShadowSdsmFrame(
        pipeline, renderer, memory, scene, settings, camera, frameIndex,
        frameIndex - 1u, std::span<const TextureHandle>(&pyramidTexture, 1u));
    ASSERT_EQ(currentSdsm.status, ShadowSdsmStatus::Active);
    if (currentSdsm.smoothedLinearMax < minimumFarDepth) {
      sawSmoothedTargetEnterFarFloor = true;
      ++frameIndex;
      break;
    }
  }
  ASSERT_TRUE(sawSmoothedTargetEnterFarFloor);

  const uint64_t baselineSequence = currentLogSequence();
  const ShadowSdsmDebugFrameData postFloorStartSdsm = currentSdsm;
  for (uint32_t postFloorFrameCount = 0u;
       postFloorFrameCount < 8u && frameIndex <= 72u;
       ++postFloorFrameCount, ++frameIndex) {
    currentSdsm = buildShadowSdsmFrame(
        pipeline, renderer, memory, scene, settings, camera, frameIndex,
        frameIndex - 1u, std::span<const TextureHandle>(&pyramidTexture, 1u));
    ASSERT_EQ(currentSdsm.status, ShadowSdsmStatus::Active);
    ASSERT_FLOAT_EQ(currentSdsm.effectiveRangeFar, currentSdsm.fixedRangeFar);
    ASSERT_EQ(currentSdsm.effectiveSplitDepths, currentSdsm.fixedSplitDepths);
  }

  EXPECT_GT(postFloorStartSdsm.smoothedLinearMax -
                currentSdsm.smoothedLinearMax,
            1.0f);
  EXPECT_EQ(countLogEntriesSince(baselineSequence, "SDSM source warning"), 0u);
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
  settings.shadow.cascadeCount = 4u;
  settings.shadow.shadowMapSize = 512u;
  settings.shadow.maxDistance = 25.0f;
  settings.shadow.debug.enableCascadeCasterCulling = false;
  settings.shadow.cascadeBlendFraction = 0.125f;
  settings.shadow.normalBias = 0.25f;
  settings.shadow.pcfSampleCount = 37u;
  settings.shadow.pcssBlockerSampleCount = 19u;
  settings.shadow.pcssFilterSampleCount = 41u;
  settings.shadow.pcssLightRadiusScale = 2.5f;
  settings.shadow.pcssSearchRadiusClampTexels = 17.0f;
  settings.shadow.pcssFilterRadiusClampTexels = 29.0f;
  settings.shadow.debug.visualizePCFResult = true;
  settings.shadow.debug.visualizeReceiverDepth = true;
  settings.shadow.debug.visualizeShadowMapDepth = true;
  settings.shadow.debug.visualizePCSSBlockers = true;
  settings.shadow.debug.visualizePCSSAverageBlockerDepth = true;
  settings.shadow.debug.visualizePCSSFilterRadius = true;
  settings.shadow.debug.fixedPoissonRotation = true;
  settings.shadow.debug.poissonRotationSeed = 1234u;

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
  ASSERT_EQ(graph.passCount(), 4u);
  ASSERT_TRUE(frameContext.sharedResources.shadowFrameGpuData.has_value());
  EXPECT_NE(frameContext.sharedResources.shadowFrameGpuData->bufferAddress, 0u);
  ShadowFrameGpuData shadowFrame{};
  auto shadowFrameReadResult = gpu.readBuffer(
      frameContext.sharedResources.shadowFrameGpuData->buffer, 0u,
      std::as_writable_bytes(std::span<ShadowFrameGpuData>(&shadowFrame, 1u)));
  ASSERT_FALSE(shadowFrameReadResult.hasError())
      << shadowFrameReadResult.error();
  EXPECT_FLOAT_EQ(shadowFrame.fadeParams.x, frameContext.camera.nearPlane);
  EXPECT_NEAR(shadowFrame.fadeParams.y, settings.shadow.maxDistance, 1.0e-4f);
  EXPECT_FLOAT_EQ(shadowFrame.fadeParams.z,
                  settings.shadow.cascadeBlendFraction);
  EXPECT_EQ(shadowFrame.flagsCascadeCountLightIndex.x &
                kShadowFrameFlagFixedPoissonRotation,
            kShadowFrameFlagFixedPoissonRotation);
  EXPECT_EQ(shadowFrame.flagsCascadeCountLightIndex.x &
                kShadowFrameFlagVisualizePCFResult,
            kShadowFrameFlagVisualizePCFResult);
  EXPECT_EQ(shadowFrame.flagsCascadeCountLightIndex.x &
                kShadowFrameFlagVisualizeReceiverDepth,
            kShadowFrameFlagVisualizeReceiverDepth);
  EXPECT_EQ(shadowFrame.flagsCascadeCountLightIndex.x &
                kShadowFrameFlagVisualizeShadowMapDepth,
            kShadowFrameFlagVisualizeShadowMapDepth);
  EXPECT_EQ(shadowFrame.flagsCascadeCountLightIndex.x &
                kShadowFrameFlagVisualizePCSSBlockers,
            kShadowFrameFlagVisualizePCSSBlockers);
  EXPECT_EQ(shadowFrame.flagsCascadeCountLightIndex.x &
                kShadowFrameFlagVisualizePCSSAverageBlockerDepth,
            kShadowFrameFlagVisualizePCSSAverageBlockerDepth);
  EXPECT_EQ(shadowFrame.flagsCascadeCountLightIndex.x &
                kShadowFrameFlagVisualizePCSSFilterRadius,
            kShadowFrameFlagVisualizePCSSFilterRadius);
  EXPECT_FLOAT_EQ(shadowFrame.fadeParams.w,
                  settings.shadow.pcssLightRadiusScale);
  EXPECT_EQ(shadowFrame.filterParams.x, settings.shadow.pcfSampleCount);
  EXPECT_EQ(shadowFrame.filterParams.y, settings.shadow.pcssBlockerSampleCount);
  EXPECT_EQ(shadowFrame.filterParams.z, settings.shadow.pcssFilterSampleCount);
  EXPECT_EQ(shadowFrame.filterParams.w,
            settings.shadow.debug.poissonRotationSeed);
  EXPECT_FLOAT_EQ(shadowFrame.cascades[0].biasParams.z,
                  settings.shadow.normalBias);
  EXPECT_FLOAT_EQ(shadowFrame.cascades[3].biasParams.z,
                  settings.shadow.normalBias);
  EXPECT_FLOAT_EQ(shadowFrame.cascades[0].pcssParams.y,
                  settings.shadow.pcssSearchRadiusClampTexels);
  EXPECT_FLOAT_EQ(shadowFrame.cascades[0].pcssParams.z,
                  settings.shadow.pcssFilterRadiusClampTexels);
  EXPECT_GT(shadowFrame.cascades[0].pcssParams.x, 0.0f);
  EXPECT_FLOAT_EQ(shadowFrame.cascades[3].pcssParams.y,
                  settings.shadow.pcssSearchRadiusClampTexels);
  EXPECT_FLOAT_EQ(shadowFrame.cascades[3].pcssParams.z,
                  settings.shadow.pcssFilterRadiusClampTexels);
  EXPECT_GT(shadowFrame.cascades[3].pcssParams.x, 0.0f);
  ASSERT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());
  ASSERT_TRUE(frameContext.sharedResources.selectedShadowLightId.has_value());
  EXPECT_EQ(frameContext.sharedResources.shadowDebugFrameData->cascadeCount,
            4u);
  EXPECT_GT(
      frameContext.sharedResources.shadowDebugFrameData->cascades[0].splitFar,
      frameContext.camera.nearPlane);
  EXPECT_NEAR(
      frameContext.sharedResources.shadowDebugFrameData->cascades[3].splitFar,
      settings.shadow.maxDistance, 1.0e-4f);

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const RenderGraphCompileResult &compiled = compileResult.value();
  ASSERT_EQ(compiled.orderedPasses.size(), 4u);
  for (uint32_t cascadeIndex = 0u; cascadeIndex < 4u; ++cascadeIndex) {
    const RenderPass &shadowPass = compiled.orderedPasses[cascadeIndex];
    EXPECT_FALSE(shadowPass.hasColorAttachment);
    ASSERT_FALSE(shadowPass.draws.empty());
    EXPECT_TRUE(shadowPass.draws.front().depthBiasEnable);
    EXPECT_TRUE(shadowPass.draws.front().useDepthState);
    EXPECT_EQ(shadowPass.draws.front().indexFormat, IndexFormat::U16);
    EXPECT_EQ(shadowPass.draws.front().instanceCount, 1u);
    EXPECT_FALSE(shadowPass.draws.front().pushConstants.empty());
    EXPECT_FALSE(shadowPass.dependencyBuffers.empty());
    EXPECT_EQ(frameContext.sharedResources.shadowDebugFrameData
                  ->cascades[cascadeIndex]
                  .drawCount,
              static_cast<uint32_t>(shadowPass.draws.size()));
  }
}

TEST(RenderGraphRendererTest,
     ShadowFeatureDebugCasterCullingReducesCascadeDrawsWhenEnabled) {
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
      makeTempRendererPath("shadow_cascade_culling_scene");
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
      .debugName = "shadow_cascade_culling_triangle",
  });
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();

  MaterialRequest materialRequest{};
  materialRequest.debugName = "shadow_cascade_culling_material";
  auto materialResult = renderer.resources().acquireMaterial(materialRequest);
  ASSERT_FALSE(materialResult.hasError()) << materialResult.error();

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  auto nearRenderableResult = scene.graph().addRenderable(
      scene.graph().rootNode(), modelResult.value(), materialResult.value());
  ASSERT_FALSE(nearRenderableResult.hasError()) << nearRenderableResult.error();

  auto farNodeResult = scene.graph().createNode(
      scene.graph().rootNode(), "FarShadowCaster",
      glm::translate(glm::mat4(1.0f), glm::vec3(1000.0f, 0.0f, 0.0f)));
  ASSERT_FALSE(farNodeResult.hasError()) << farNodeResult.error();
  auto farRenderableResult = scene.graph().addRenderable(
      farNodeResult.value(), modelResult.value(), materialResult.value());
  ASSERT_FALSE(farRenderableResult.hasError()) << farRenderableResult.error();

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
  settings.shadow.cascadeCount = 4u;
  settings.shadow.shadowMapSize = 512u;
  settings.shadow.maxDistance = 25.0f;
  settings.shadow.debug.enableCascadeCasterCulling = false;

  RenderFrameContext uncullFrame{};
  uncullFrame.frameIndex = 30u;
  uncullFrame.scene = &scene;
  uncullFrame.resources = &renderer.resources();
  uncullFrame.settings = &settings;
  uncullFrame.camera.view =
      glm::lookAt(glm::vec3(0.0f, 1.5f, 4.0f), glm::vec3(0.0f, 0.5f, 0.0f),
                  glm::vec3(0.0f, 1.0f, 0.0f));
  uncullFrame.camera.proj =
      glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 30.0f);
  uncullFrame.camera.cameraPos = glm::vec4(0.0f, 1.5f, 4.0f, 1.0f);
  uncullFrame.camera.aspectRatio = 1.0f;
  uncullFrame.camera.projectionType = ProjectionType::Perspective;
  uncullFrame.camera.nearPlane = 0.1f;
  uncullFrame.camera.farPlane = 30.0f;
  uncullFrame.camera.fovYRadians = glm::radians(60.0f);

  RenderGraphBuilder uncullGraph(&memory);
  uncullGraph.beginFrame(uncullFrame.frameIndex);
  auto uncullBuildResult =
      pipeline.buildRenderGraph(uncullFrame, renderer.resources(), uncullGraph);
  ASSERT_FALSE(uncullBuildResult.hasError()) << uncullBuildResult.error();
  ASSERT_TRUE(uncullBuildResult.value());
  ASSERT_TRUE(uncullFrame.sharedResources.shadowDebugFrameData.has_value());
  const ShadowDebugFrameData uncullDebug =
      *uncullFrame.sharedResources.shadowDebugFrameData;

  settings.shadow.debug.enableCascadeCasterCulling = true;
  RenderFrameContext cullFrame = uncullFrame;
  cullFrame.frameIndex = 31u;
  cullFrame.settings = &settings;

  RenderGraphBuilder cullGraph(&memory);
  cullGraph.beginFrame(cullFrame.frameIndex);
  auto cullBuildResult =
      pipeline.buildRenderGraph(cullFrame, renderer.resources(), cullGraph);
  ASSERT_FALSE(cullBuildResult.hasError()) << cullBuildResult.error();
  ASSERT_TRUE(cullBuildResult.value());
  ASSERT_TRUE(cullFrame.sharedResources.shadowDebugFrameData.has_value());
  const ShadowDebugFrameData &cullDebug =
      *cullFrame.sharedResources.shadowDebugFrameData;

  bool sawCulledCascade = false;
  for (uint32_t cascadeIndex = 0u; cascadeIndex < 4u; ++cascadeIndex) {
    EXPECT_EQ(uncullDebug.cascades[cascadeIndex].drawCount, 2u);
    EXPECT_EQ(uncullDebug.cascades[cascadeIndex].culledCount, 0u);
    EXPECT_EQ(cullDebug.cascades[cascadeIndex].drawCount +
                  cullDebug.cascades[cascadeIndex].culledCount,
              uncullDebug.cascades[cascadeIndex].drawCount);
    sawCulledCascade =
        sawCulledCascade || cullDebug.cascades[cascadeIndex].culledCount > 0u;
  }
  EXPECT_TRUE(sawCulledCascade);

  RenderGraphRuntime runtime;
  auto compileResult = cullGraph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  uint32_t shadowPassCount = 0u;
  for (const RenderPass &pass : compileResult.value().orderedPasses) {
    if (std::string_view(pass.debugLabel)
            .starts_with("ShadowDepthPass.Cascade")) {
      ++shadowPassCount;
    }
  }
  EXPECT_EQ(shadowPassCount, 4u);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureReusesStaticShadowCasterCacheAcrossFrames) {
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
      makeTempRendererPath("shadow_static_cache_scene");
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
      .debugName = "shadow_static_cache_triangle",
  });
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();

  MaterialRequest materialRequest{};
  materialRequest.debugName = "shadow_static_cache_material";
  auto materialResult = renderer.resources().acquireMaterial(materialRequest);
  ASSERT_FALSE(materialResult.hasError()) << materialResult.error();

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  auto staticRenderableResult = scene.graph().addRenderable(
      scene.graph().rootNode(), modelResult.value(), materialResult.value());
  ASSERT_FALSE(staticRenderableResult.hasError())
      << staticRenderableResult.error();

  auto dynamicNodeResult = scene.graph().createNode(
      scene.graph().rootNode(), "DynamicShadowCaster",
      glm::translate(glm::mat4(1.0f), glm::vec3(1.5f, 0.0f, 0.0f)));
  ASSERT_FALSE(dynamicNodeResult.hasError()) << dynamicNodeResult.error();
  auto dynamicRenderableResult = scene.graph().addRenderable(
      dynamicNodeResult.value(), modelResult.value(), materialResult.value());
  ASSERT_FALSE(dynamicRenderableResult.hasError())
      << dynamicRenderableResult.error();

  const float morphWeight = 1.0f;
  EXPECT_TRUE(scene.graph().setRenderableMorphWeights(
      dynamicRenderableResult.value(),
      std::span<const float>(&morphWeight, 1u)));

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
  settings.shadow.cascadeCount = 4u;
  settings.shadow.shadowMapSize = 512u;
  settings.shadow.maxDistance = 25.0f;
  settings.shadow.debug.enableCascadeCasterCulling = true;

  const auto buildFrame = [&](uint64_t frameIndex) {
    RenderFrameContext frameContext{};
    frameContext.frameIndex = frameIndex;
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
    EXPECT_FALSE(buildResult.hasError()) << buildResult.error();
    EXPECT_TRUE(buildResult.value());
    return frameContext;
  };

  const RenderFrameContext firstFrame = buildFrame(40u);
  EXPECT_EQ(firstFrame.metrics.shadow.staticCasterEntries, 1u);
  EXPECT_EQ(firstFrame.metrics.shadow.dynamicCasterEntries, 1u);
  EXPECT_EQ(firstFrame.metrics.shadow.staticCacheReused, 0u);
  EXPECT_EQ(firstFrame.metrics.shadow.reusedStaticOnlyCascadeCount, 0u);
  EXPECT_GT(firstFrame.metrics.shadow.totalDraws, 0u);

  const RenderFrameContext secondFrame = buildFrame(41u);
  EXPECT_EQ(secondFrame.metrics.shadow.staticCasterEntries, 1u);
  EXPECT_EQ(secondFrame.metrics.shadow.dynamicCasterEntries, 1u);
  EXPECT_EQ(secondFrame.metrics.shadow.staticCacheReused, 1u);
  EXPECT_EQ(secondFrame.metrics.shadow.reusedStaticOnlyCascadeCount, 0u);
  EXPECT_GT(secondFrame.metrics.shadow.totalDraws, 0u);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureReusesStaticOnlyShadowCascadesAcrossFrames) {
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
      makeTempRendererPath("shadow_static_only_cascade_reuse_scene");
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
      .debugName = "shadow_static_only_cache_triangle",
  });
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();

  MaterialRequest materialRequest{};
  materialRequest.debugName = "shadow_static_only_cache_material";
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
  settings.shadow.cascadeCount = 4u;
  settings.shadow.shadowMapSize = 512u;
  settings.shadow.maxDistance = 25.0f;
  settings.shadow.debug.enableCascadeCasterCulling = true;

  const auto buildFrame = [&](uint64_t frameIndex)
      -> std::pair<RenderFrameContext, RenderGraphCompileResult> {
    RenderFrameContext frameContext{};
    RenderGraphCompileResult compiled(&memory);
    frameContext.frameIndex = frameIndex;
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
    if (buildResult.hasError()) {
      ADD_FAILURE() << buildResult.error();
      return {frameContext, std::move(compiled)};
    }
    if (!buildResult.value()) {
      ADD_FAILURE() << "shadow static-only reuse frame build returned false";
      return {frameContext, std::move(compiled)};
    }

    RenderGraphRuntime runtime;
    auto compileResult = graph.compile(runtime);
    if (compileResult.hasError()) {
      ADD_FAILURE() << compileResult.error();
      return {frameContext, std::move(compiled)};
    }
    return std::pair<RenderFrameContext, RenderGraphCompileResult>(
        std::move(frameContext), std::move(compileResult.value()));
  };

  const auto [firstFrame, firstCompiled] = buildFrame(60u);
  EXPECT_EQ(firstFrame.metrics.shadow.staticCacheReused, 0u);
  EXPECT_EQ(firstFrame.metrics.shadow.staticOnlyCandidateCount, 4u);
  EXPECT_EQ(firstFrame.metrics.shadow.reusedStaticOnlyCascadeCount, 0u);
  EXPECT_EQ(
      firstFrame.metrics.shadow.staticOnlyReuseMissStaticCacheRebuiltCount, 4u);
  EXPECT_EQ(
      firstFrame.metrics.shadow.staticOnlyReuseMissRasterStateChangedCount, 0u);
  EXPECT_GT(firstFrame.metrics.shadow.totalDraws, 0u);
  EXPECT_GT(firstFrame.metrics.shadow.totalIndexCountEstimate, 0u);
  uint32_t firstShadowPassCount = 0u;
  for (const RenderPass &pass : firstCompiled.orderedPasses) {
    if (!std::string_view(pass.debugLabel)
             .starts_with("ShadowDepthPass.Cascade")) {
      continue;
    }
    ++firstShadowPassCount;
    EXPECT_EQ(pass.depth.loadOp, LoadOp::Clear);
    EXPECT_FALSE(pass.draws.empty());
  }
  EXPECT_EQ(firstShadowPassCount, 4u);

  const auto [secondFrame, secondCompiled] = buildFrame(61u);
  EXPECT_EQ(secondFrame.metrics.shadow.staticCacheReused, 1u);
  EXPECT_EQ(secondFrame.metrics.shadow.staticOnlyCandidateCount, 4u);
  EXPECT_EQ(secondFrame.metrics.shadow.reusedStaticOnlyCascadeCount, 4u);
  EXPECT_EQ(
      secondFrame.metrics.shadow.staticOnlyReuseMissRasterStateChangedCount,
      0u);
  EXPECT_EQ(secondFrame.metrics.shadow.totalDraws, 0u);
  EXPECT_EQ(secondFrame.metrics.shadow.totalIndexCountEstimate, 0u);
  uint32_t secondShadowPassCount = 0u;
  for (const RenderPass &pass : secondCompiled.orderedPasses) {
    if (!std::string_view(pass.debugLabel)
             .starts_with("ShadowDepthPass.Cascade")) {
      continue;
    }
    ++secondShadowPassCount;
    EXPECT_EQ(pass.depth.loadOp, LoadOp::Load);
    EXPECT_TRUE(pass.preDispatches.empty());
    EXPECT_TRUE(pass.draws.empty());
  }
  EXPECT_EQ(secondShadowPassCount, 4u);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureReusesStaticOnlyShadowCascadesForSubTexelCameraMotion) {
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
      makeTempRendererPath("shadow_static_only_reuse_subtexel_motion_scene");
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
      .debugName = "shadow_static_only_reuse_subtexel_motion_triangle",
  });
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();

  MaterialRequest materialRequest{};
  materialRequest.debugName = "shadow_static_only_reuse_subtexel_motion_material";
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
  settings.shadow.cascadeCount = 4u;
  settings.shadow.shadowMapSize = 512u;
  settings.shadow.maxDistance = 25.0f;
  settings.shadow.debug.enableCascadeCasterCulling = false;

  const glm::vec3 baseEye(0.0f, 1.5f, 4.0f);
  const glm::vec3 baseTarget(0.0f, 0.5f, 0.0f);
  const glm::vec3 up(0.0f, 1.0f, 0.0f);
  const auto buildFrame = [&](uint64_t frameIndex, const glm::vec3 &eye,
                              const glm::vec3 &target) {
    RenderFrameContext frameContext{};
    frameContext.frameIndex = frameIndex;
    frameContext.scene = &scene;
    frameContext.resources = &renderer.resources();
    frameContext.settings = &settings;
    frameContext.camera.view = glm::lookAt(eye, target, up);
    frameContext.camera.proj =
        glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 30.0f);
    frameContext.camera.cameraPos = glm::vec4(eye, 1.0f);
    frameContext.camera.aspectRatio = 1.0f;
    frameContext.camera.projectionType = ProjectionType::Perspective;
    frameContext.camera.nearPlane = 0.1f;
    frameContext.camera.farPlane = 30.0f;
    frameContext.camera.fovYRadians = glm::radians(60.0f);

    RenderGraphBuilder graph(&memory);
    graph.beginFrame(frameContext.frameIndex);
    auto buildResult =
        pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
    EXPECT_FALSE(buildResult.hasError()) << buildResult.error();
    EXPECT_TRUE(buildResult.value());
    return frameContext;
  };

  const RenderFrameContext firstFrame = buildFrame(70u, baseEye, baseTarget);
  ASSERT_TRUE(firstFrame.sharedResources.shadowDebugFrameData.has_value());
  const ShadowCascadeDebugFrameData &baseCascade =
      firstFrame.sharedResources.shadowDebugFrameData->cascades[0];
  ASSERT_GT(baseCascade.texelWorldSize, 0.0f);

  const glm::vec3 subTexelMotion = glm::vec3(
      glm::inverse(baseCascade.lightView) *
      glm::vec4(baseCascade.texelWorldSize * 0.75f, 0.0f, 0.0f, 0.0f));
  const RenderFrameContext movedFrame =
      buildFrame(71u, baseEye + subTexelMotion, baseTarget + subTexelMotion);

  EXPECT_EQ(movedFrame.metrics.shadow.staticCacheReused, 1u);
  EXPECT_EQ(movedFrame.metrics.shadow.staticOnlyCandidateCount, 4u);
  EXPECT_EQ(movedFrame.metrics.shadow.reusedStaticOnlyCascadeCount, 4u);
  EXPECT_EQ(
      movedFrame.metrics.shadow.staticOnlyReuseMissRasterStateChangedCount, 0u);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureReportsStaticOnlyReuseRasterMismatchForMovedStaticCamera) {
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
      makeTempRendererPath("shadow_static_only_reuse_moved_camera_scene");
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
      .debugName = "shadow_static_only_reuse_moved_camera_triangle",
  });
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();

  MaterialRequest materialRequest{};
  materialRequest.debugName = "shadow_static_only_reuse_moved_camera_material";
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
  settings.shadow.cascadeCount = 4u;
  settings.shadow.shadowMapSize = 512u;
  settings.shadow.maxDistance = 25.0f;
  settings.shadow.debug.enableCascadeCasterCulling = false;

  const auto buildFrame = [&](uint64_t frameIndex, const glm::vec3 &cameraPos) {
    RenderFrameContext frameContext{};
    frameContext.frameIndex = frameIndex;
    frameContext.scene = &scene;
    frameContext.resources = &renderer.resources();
    frameContext.settings = &settings;
    frameContext.camera.view = glm::lookAt(
        cameraPos, glm::vec3(0.0f, 0.5f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    frameContext.camera.proj =
        glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 30.0f);
    frameContext.camera.cameraPos = glm::vec4(cameraPos, 1.0f);
    frameContext.camera.aspectRatio = 1.0f;
    frameContext.camera.projectionType = ProjectionType::Perspective;
    frameContext.camera.nearPlane = 0.1f;
    frameContext.camera.farPlane = 30.0f;
    frameContext.camera.fovYRadians = glm::radians(60.0f);

    RenderGraphBuilder graph(&memory);
    graph.beginFrame(frameContext.frameIndex);
    auto buildResult =
        pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
    EXPECT_FALSE(buildResult.hasError()) << buildResult.error();
    EXPECT_TRUE(buildResult.value());
    return frameContext;
  };

  const RenderFrameContext firstFrame =
      buildFrame(70u, glm::vec3(0.0f, 1.5f, 4.0f));
  EXPECT_EQ(firstFrame.metrics.shadow.staticOnlyCandidateCount, 4u);
  EXPECT_EQ(firstFrame.metrics.shadow.reusedStaticOnlyCascadeCount, 0u);
  EXPECT_EQ(
      firstFrame.metrics.shadow.staticOnlyReuseMissStaticCacheRebuiltCount, 4u);

  const RenderFrameContext movedFrame =
      buildFrame(71u, glm::vec3(0.4f, 1.5f, 4.0f));
  EXPECT_EQ(movedFrame.metrics.shadow.staticCacheReused, 1u);
  EXPECT_EQ(movedFrame.metrics.shadow.staticOnlyCandidateCount, 4u);
  EXPECT_EQ(movedFrame.metrics.shadow.reusedStaticOnlyCascadeCount, 0u);
  EXPECT_EQ(
      movedFrame.metrics.shadow.staticOnlyReuseMissRasterStateChangedCount, 4u);
  ASSERT_TRUE(movedFrame.sharedResources.shadowDebugFrameData.has_value());
  const ShadowCascadeDebugFrameData &cascade =
      movedFrame.sharedResources.shadowDebugFrameData->cascades[0];
  EXPECT_EQ(
      cascade.staticOnlyReuseStatus,
      ShadowCascadeDebugFrameData::StaticOnlyReuseStatus::RasterStateChanged);
  EXPECT_TRUE(cascade.staticOnlyReuseCandidate);
  EXPECT_TRUE(cascade.staticOnlyReusePreviousValid);
  EXPECT_TRUE(cascade.staticOnlyReuseLightViewProjChanged);
  EXPECT_FALSE(cascade.staticOnlyReuseBiasChanged);
  EXPECT_FALSE(cascade.staticOnlyReuseCasterSignatureChanged);
  EXPECT_NE(cascade.currentStaticOnlyRasterSignature,
            cascade.previousStaticOnlyRasterSignature);
  EXPECT_NE(cascade.currentStaticOnlyLightViewProjSignature,
            cascade.previousStaticOnlyLightViewProjSignature);
  EXPECT_EQ(cascade.currentStaticOnlyCasterSignature,
            cascade.previousStaticOnlyCasterSignature);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureInvalidatesStaticOnlyCascadeReuseWhenShadowsDisable) {
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
      makeTempRendererPath("shadow_static_only_disable_invalidate_scene");
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
      .debugName = "shadow_static_only_disable_triangle",
  });
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();

  MaterialRequest materialRequest{};
  materialRequest.debugName = "shadow_static_only_disable_material";
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
  settings.shadow.cascadeCount = 4u;
  settings.shadow.shadowMapSize = 512u;
  settings.shadow.maxDistance = 25.0f;
  settings.shadow.debug.enableCascadeCasterCulling = true;

  const auto buildFrame = [&](uint64_t frameIndex)
      -> std::pair<RenderFrameContext, RenderGraphCompileResult> {
    RenderFrameContext frameContext{};
    RenderGraphCompileResult compiled(&memory);
    frameContext.frameIndex = frameIndex;
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
    if (buildResult.hasError()) {
      ADD_FAILURE() << buildResult.error();
      return {frameContext, std::move(compiled)};
    }
    if (!buildResult.value()) {
      ADD_FAILURE()
          << "shadow static-only disable invalidation frame build returned "
             "false";
      return {frameContext, std::move(compiled)};
    }

    RenderGraphRuntime runtime;
    auto compileResult = graph.compile(runtime);
    if (compileResult.hasError()) {
      ADD_FAILURE() << compileResult.error();
      return {frameContext, std::move(compiled)};
    }
    return std::pair<RenderFrameContext, RenderGraphCompileResult>(
        std::move(frameContext), std::move(compileResult.value()));
  };

  const auto [firstFrame, firstCompiled] = buildFrame(80u);
  EXPECT_EQ(firstFrame.metrics.shadow.reusedStaticOnlyCascadeCount, 0u);
  EXPECT_FALSE(firstCompiled.orderedPasses.empty());

  settings.shadow.enabled = false;
  const auto [disabledFrame, disabledCompiled] = buildFrame(81u);
  EXPECT_EQ(disabledFrame.metrics.shadow.cascadeCount, 0u);
  uint32_t disabledShadowPassCount = 0u;
  for (const RenderPass &pass : disabledCompiled.orderedPasses) {
    if (std::string_view(pass.debugLabel)
            .starts_with("ShadowDepthPass.Cascade")) {
      ++disabledShadowPassCount;
    }
  }
  EXPECT_EQ(disabledShadowPassCount, 0u);

  settings.shadow.enabled = true;
  const auto [reenabledFrame, reenabledCompiled] = buildFrame(82u);
  EXPECT_EQ(reenabledFrame.metrics.shadow.reusedStaticOnlyCascadeCount, 0u);
  EXPECT_GT(reenabledFrame.metrics.shadow.totalDraws, 0u);
  uint32_t reenabledShadowPassCount = 0u;
  for (const RenderPass &pass : reenabledCompiled.orderedPasses) {
    if (!std::string_view(pass.debugLabel)
             .starts_with("ShadowDepthPass.Cascade")) {
      continue;
    }
    ++reenabledShadowPassCount;
    EXPECT_EQ(pass.depth.loadOp, LoadOp::Clear);
    EXPECT_FALSE(pass.draws.empty());
  }
  EXPECT_EQ(reenabledShadowPassCount, 4u);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureStaticOnlyCascadeReuseIgnoresSamplingOnlyShadowParams) {
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
      makeTempRendererPath("shadow_static_only_sampling_param_scene");
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
      .debugName = "shadow_static_only_sampling_triangle",
  });
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();

  MaterialRequest materialRequest{};
  materialRequest.debugName = "shadow_static_only_sampling_material";
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
  settings.shadow.cascadeCount = 4u;
  settings.shadow.shadowMapSize = 512u;
  settings.shadow.maxDistance = 25.0f;
  settings.shadow.debug.enableCascadeCasterCulling = true;
  settings.shadow.filterMode = ShadowFilterMode::Hard;
  settings.shadow.pcssLightRadiusScale = 0.05f;

  const auto buildFrame = [&](uint64_t frameIndex)
      -> std::pair<RenderFrameContext, RenderGraphCompileResult> {
    RenderFrameContext frameContext{};
    RenderGraphCompileResult compiled(&memory);
    frameContext.frameIndex = frameIndex;
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
    if (buildResult.hasError()) {
      ADD_FAILURE() << buildResult.error();
      return {frameContext, std::move(compiled)};
    }
    if (!buildResult.value()) {
      ADD_FAILURE()
          << "shadow static-only sampling-param frame build returned false";
      return {frameContext, std::move(compiled)};
    }

    RenderGraphRuntime runtime;
    auto compileResult = graph.compile(runtime);
    if (compileResult.hasError()) {
      ADD_FAILURE() << compileResult.error();
      return {frameContext, std::move(compiled)};
    }
    return std::pair<RenderFrameContext, RenderGraphCompileResult>(
        std::move(frameContext), std::move(compileResult.value()));
  };

  const auto [firstFrame, firstCompiled] = buildFrame(100u);
  EXPECT_EQ(firstFrame.metrics.shadow.reusedStaticOnlyCascadeCount, 0u);
  uint32_t firstShadowPassCount = 0u;
  for (const RenderPass &pass : firstCompiled.orderedPasses) {
    if (!std::string_view(pass.debugLabel)
             .starts_with("ShadowDepthPass.Cascade")) {
      continue;
    }
    ++firstShadowPassCount;
    EXPECT_EQ(pass.depth.loadOp, LoadOp::Clear);
    EXPECT_FALSE(pass.draws.empty());
  }
  EXPECT_EQ(firstShadowPassCount, 4u);

  settings.shadow.pcssLightRadiusScale = 2.0f;
  const auto [secondFrame, secondCompiled] = buildFrame(101u);
  EXPECT_EQ(secondFrame.metrics.shadow.staticCacheReused, 1u);
  EXPECT_EQ(secondFrame.metrics.shadow.reusedStaticOnlyCascadeCount, 4u);
  EXPECT_EQ(secondFrame.metrics.shadow.totalDraws, 0u);
  uint32_t secondShadowPassCount = 0u;
  for (const RenderPass &pass : secondCompiled.orderedPasses) {
    if (!std::string_view(pass.debugLabel)
             .starts_with("ShadowDepthPass.Cascade")) {
      continue;
    }
    ++secondShadowPassCount;
    EXPECT_EQ(pass.depth.loadOp, LoadOp::Load);
    EXPECT_TRUE(pass.draws.empty());
  }
  EXPECT_EQ(secondShadowPassCount, 4u);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureUsesAnimatedGeometryOverrideForShadowCasters) {
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
      makeTempRendererPath("shadow_feature_anim_override");
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
      .debugName = "shadow_triangle_anim_override",
  });
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();

  MaterialRequest materialRequest{};
  materialRequest.debugName = "shadow_material_anim_override";
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

  auto instanceMatricesResult =
      gpu.createBuffer(BufferDesc{.usage = BufferUsage::Storage,
                                  .storage = Storage::Device,
                                  .size = 256u},
                       "shadow_anim_instance_matrices");
  ASSERT_FALSE(instanceMatricesResult.hasError())
      << instanceMatricesResult.error();
  const BufferHandle instanceMatricesBuffer = instanceMatricesResult.value();
  const uint64_t instanceMatricesAddress =
      gpu.getBufferDeviceAddress(instanceMatricesBuffer, 0u);
  ASSERT_NE(instanceMatricesAddress, 0u);

  auto overrideVertexResult = gpu.createBuffer(
      BufferDesc{.usage = BufferUsage::Vertex | BufferUsage::Storage,
                 .storage = Storage::Device,
                 .size = 256u},
      "shadow_anim_override_vertices");
  ASSERT_FALSE(overrideVertexResult.hasError()) << overrideVertexResult.error();
  const BufferHandle overrideVertexBuffer = overrideVertexResult.value();
  const uint64_t overrideVertexAddress =
      gpu.getBufferDeviceAddress(overrideVertexBuffer, 0u);
  ASSERT_NE(overrideVertexAddress, 0u);

  std::array<AnimatedRenderableGeometryOverride, 1> overrides{};
  overrides[0] = AnimatedRenderableGeometryOverride{
      .vertexBuffer = overrideVertexBuffer,
      .vertexByteOffset = 0u,
      .vertexCount = 3u,
      .enabled = true,
  };

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.cascadeCount = 1u;
  settings.shadow.shadowMapSize = 512u;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 4u;
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
  frameContext.sharedResources.animationSceneGpuData = AnimationSceneFrameData{
      .instanceMatricesBuffer = instanceMatricesBuffer,
      .instanceMatricesAddress = instanceMatricesAddress,
      .preDispatches = {},
      .geometryOverridesByRenderable = overrides,
      .scene = &scene,
      .sceneTopologyVersion = scene.topologyVersion(),
      .renderableCount = scene.renderables().size(),
      .version = 1u,
  };

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
  EXPECT_TRUE(
      sameBuffer(shadowPass.draws.front().vertexBuffer, overrideVertexBuffer));
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
  settings.shadow.cascadeCount = 1u;
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
  settings.shadow.cascadeCount = 1u;
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
  settings.shadow.cascadeCount = 1u;
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
  settings.shadow.debug.visualizeCascadeIndex = true;
  settings.shadow.debug.visualizePCFResult = true;
  settings.shadow.debug.visualizeReceiverDepth = true;
  settings.shadow.debug.visualizeShadowMapDepth = true;
  settings.shadow.debug.visualizePCSSBlockers = true;
  settings.shadow.debug.visualizePCSSAverageBlockerDepth = true;
  settings.shadow.debug.visualizePCSSFilterRadius = true;
  settings.shadow.debug.fixedPoissonRotation = true;

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
  EXPECT_EQ(frameContext.sharedResources.forwardSceneGpuData->shadowFlags &
                kShadowFrameFlagVisualizeCascadeIndex,
            kShadowFrameFlagVisualizeCascadeIndex);
  EXPECT_EQ(frameContext.sharedResources.forwardSceneGpuData->shadowFlags &
                kShadowFrameFlagVisualizePCFResult,
            kShadowFrameFlagVisualizePCFResult);
  EXPECT_EQ(frameContext.sharedResources.forwardSceneGpuData->shadowFlags &
                kShadowFrameFlagVisualizeReceiverDepth,
            kShadowFrameFlagVisualizeReceiverDepth);
  EXPECT_EQ(frameContext.sharedResources.forwardSceneGpuData->shadowFlags &
                kShadowFrameFlagVisualizeShadowMapDepth,
            kShadowFrameFlagVisualizeShadowMapDepth);
  EXPECT_EQ(frameContext.sharedResources.forwardSceneGpuData->shadowFlags &
                kShadowFrameFlagVisualizePCSSBlockers,
            kShadowFrameFlagVisualizePCSSBlockers);
  EXPECT_EQ(frameContext.sharedResources.forwardSceneGpuData->shadowFlags &
                kShadowFrameFlagVisualizePCSSAverageBlockerDepth,
            kShadowFrameFlagVisualizePCSSAverageBlockerDepth);
  EXPECT_EQ(frameContext.sharedResources.forwardSceneGpuData->shadowFlags &
                kShadowFrameFlagVisualizePCSSFilterRadius,
            kShadowFrameFlagVisualizePCSSFilterRadius);
  EXPECT_EQ(frameContext.sharedResources.forwardSceneGpuData->shadowFlags &
                kShadowFrameFlagFixedPoissonRotation,
            kShadowFrameFlagFixedPoissonRotation);
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
  settings.shadow.cascadeCount = 1u;
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

TEST(RenderGraphRendererTest,
     ShadowFeatureStabilizesCascadeCentersForSmallCameraMotion) {
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
  settings.shadow.cascadeCount = 4u;
  settings.shadow.stabilizeCascades = true;

  const glm::vec3 eyeA(0.0f, 2.0f, 6.0f);
  const glm::vec3 targetA(0.0f, 0.5f, 0.0f);
  const glm::vec3 up(0.0f, 1.0f, 0.0f);

  RenderFrameContext frameA{};
  frameA.frameIndex = 15u;
  frameA.scene = &scene;
  frameA.resources = &renderer.resources();
  frameA.settings = &settings;
  frameA.camera.view = glm::lookAt(eyeA, targetA, up);
  frameA.camera.proj =
      glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 100.0f);
  frameA.camera.cameraPos = glm::vec4(eyeA, 1.0f);
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
  ASSERT_GT(cascadeA.texelWorldSize, 0.0f);

  const float cameraOffset = cascadeA.texelWorldSize * 0.25f;
  const glm::vec3 motion(cameraOffset, 0.0f, 0.0f);
  RenderFrameContext frameB = frameA;
  frameB.frameIndex = 16u;
  frameB.camera.view = glm::lookAt(eyeA + motion, targetA + motion, up);
  frameB.camera.cameraPos = glm::vec4(eyeA + motion, 1.0f);

  RenderGraphBuilder graphB(&memory);
  graphB.beginFrame(frameB.frameIndex);
  auto buildResultB =
      pipeline.buildRenderGraph(frameB, renderer.resources(), graphB);
  ASSERT_FALSE(buildResultB.hasError()) << buildResultB.error();
  ASSERT_TRUE(buildResultB.value());
  ASSERT_TRUE(frameB.sharedResources.shadowDebugFrameData.has_value());
  const ShadowCascadeDebugFrameData &cascadeB =
      frameB.sharedResources.shadowDebugFrameData->cascades[0];

  EXPECT_GT(std::abs(cascadeA.unsnappedCenter.x - cascadeB.unsnappedCenter.x),
            1.0e-5f);
  EXPECT_NEAR(cascadeA.snappedCenter.x, cascadeB.snappedCenter.x, 1.0e-5f);
  EXPECT_NEAR(cascadeA.snappedCenter.y, cascadeB.snappedCenter.y, 1.0e-5f);
  EXPECT_NEAR(cascadeA.snappedCenter.z, cascadeB.snappedCenter.z, 1.0e-5f);
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
  settings.shadow.cascadeCount = 1u;
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
  settings.shadow.cascadeCount = 1u;
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

TEST(RenderGraphRendererTest,
     ShadowFeatureFreezeCascadesReusesCachedCascadeFits) {
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
  settings.shadow.cascadeCount = 4u;
  settings.shadow.debug.freezeCascades = true;

  RenderFrameContext frameA{};
  frameA.frameIndex = 21u;
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
  const ShadowDebugFrameData frozenDebug =
      *frameA.sharedResources.shadowDebugFrameData;
  ASSERT_EQ(frozenDebug.cascadeCount, 4u);

  RenderFrameContext frameB = frameA;
  frameB.frameIndex = 22u;
  frameB.camera.view =
      glm::lookAt(glm::vec3(8.0f, 4.0f, -7.0f), glm::vec3(2.0f, 1.0f, 0.0f),
                  glm::vec3(0.0f, 1.0f, 0.0f));
  frameB.camera.proj =
      glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 100.0f);
  frameB.camera.cameraPos = glm::vec4(8.0f, 4.0f, -7.0f, 1.0f);

  RenderGraphBuilder graphB(&memory);
  graphB.beginFrame(frameB.frameIndex);
  auto buildResultB =
      pipeline.buildRenderGraph(frameB, renderer.resources(), graphB);
  ASSERT_FALSE(buildResultB.hasError()) << buildResultB.error();
  ASSERT_TRUE(buildResultB.value());
  ASSERT_TRUE(frameB.sharedResources.shadowDebugFrameData.has_value());
  const ShadowDebugFrameData &movedDebug =
      *frameB.sharedResources.shadowDebugFrameData;
  ASSERT_EQ(movedDebug.cascadeCount, 4u);

  for (uint32_t cascadeIndex = 0u; cascadeIndex < 4u; ++cascadeIndex) {
    const ShadowCascadeDebugFrameData &frozen =
        frozenDebug.cascades[cascadeIndex];
    const ShadowCascadeDebugFrameData &moved =
        movedDebug.cascades[cascadeIndex];
    EXPECT_NEAR(frozen.splitNear, moved.splitNear, 1.0e-6f);
    EXPECT_NEAR(frozen.splitFar, moved.splitFar, 1.0e-6f);
    EXPECT_NEAR(frozen.snappedCenter.x, moved.snappedCenter.x, 1.0e-6f);
    EXPECT_NEAR(frozen.snappedCenter.y, moved.snappedCenter.y, 1.0e-6f);
    EXPECT_NEAR(frozen.snappedCenter.z, moved.snappedCenter.z, 1.0e-6f);
    for (int c = 0; c < 4; ++c) {
      for (int r = 0; r < 4; ++r) {
        EXPECT_NEAR(frozen.lightViewProj[c][r], moved.lightViewProj[c][r],
                    1.0e-6f);
      }
    }
  }
}

TEST(RenderGraphRendererTest,
     ShadowFeatureDisablingFreezeCascadesResumesCascadeFitUpdates) {
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
  settings.shadow.cascadeCount = 4u;
  settings.shadow.debug.freezeCascades = true;

  RenderFrameContext frozenFrame{};
  frozenFrame.frameIndex = 23u;
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
  const ShadowDebugFrameData frozenDebug =
      *frozenFrame.sharedResources.shadowDebugFrameData;

  settings.shadow.debug.freezeCascades = false;
  RenderFrameContext unfrozenFrame = frozenFrame;
  unfrozenFrame.frameIndex = 24u;
  unfrozenFrame.settings = &settings;
  unfrozenFrame.camera.view =
      glm::lookAt(glm::vec3(8.0f, 4.0f, -7.0f), glm::vec3(2.0f, 1.0f, 0.0f),
                  glm::vec3(0.0f, 1.0f, 0.0f));
  unfrozenFrame.camera.proj =
      glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 100.0f);
  unfrozenFrame.camera.cameraPos = glm::vec4(8.0f, 4.0f, -7.0f, 1.0f);

  RenderGraphBuilder unfrozenGraph(&memory);
  unfrozenGraph.beginFrame(unfrozenFrame.frameIndex);
  auto unfrozenBuildResult = pipeline.buildRenderGraph(
      unfrozenFrame, renderer.resources(), unfrozenGraph);
  ASSERT_FALSE(unfrozenBuildResult.hasError()) << unfrozenBuildResult.error();
  ASSERT_TRUE(unfrozenBuildResult.value());
  ASSERT_TRUE(unfrozenFrame.sharedResources.shadowDebugFrameData.has_value());
  const ShadowDebugFrameData &unfrozenDebug =
      *unfrozenFrame.sharedResources.shadowDebugFrameData;

  bool changed = false;
  for (uint32_t cascadeIndex = 0u; cascadeIndex < 4u; ++cascadeIndex) {
    const ShadowCascadeDebugFrameData &frozen =
        frozenDebug.cascades[cascadeIndex];
    const ShadowCascadeDebugFrameData &unfrozen =
        unfrozenDebug.cascades[cascadeIndex];
    changed |= std::abs(frozen.splitNear - unfrozen.splitNear) > 1.0e-5f;
    changed |= std::abs(frozen.splitFar - unfrozen.splitFar) > 1.0e-5f;
    for (int c = 0; c < 4; ++c) {
      for (int r = 0; r < 4; ++r) {
        changed |= std::abs(frozen.lightViewProj[c][r] -
                            unfrozen.lightViewProj[c][r]) > 1.0e-5f;
      }
    }
  }
  EXPECT_TRUE(changed);
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
  settings.shadow.cascadeCount = 4u;
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

  std::array<uint32_t, kMaxShadowCascades> shadowTextureResourceIndices{};
  shadowTextureResourceIndices.fill(UINT32_MAX);
  for (uint32_t cascadeIndex = 0u; cascadeIndex < kMaxShadowCascades;
       ++cascadeIndex) {
    const TextureHandle shadowTexture =
        frameContext.sharedResources.shadowCascadeTextures[cascadeIndex];
    for (uint32_t i = 0u; i < compiled.textureHandlesByResource.size(); ++i) {
      if (sameTexture(compiled.textureHandlesByResource[i], shadowTexture)) {
        shadowTextureResourceIndices[cascadeIndex] = i;
        break;
      }
    }
    ASSERT_NE(shadowTextureResourceIndices[cascadeIndex], UINT32_MAX);
  }

  uint32_t sawShadowPassCount = 0u;
  bool sawOpaquePass = false;
  size_t lastShadowPassIndex = 0u;
  size_t opaquePassIndex = 0u;
  for (size_t i = 0u; i < compiled.orderedPasses.size(); ++i) {
    const RenderPass &pass = compiled.orderedPasses[i];
    if (std::string_view(pass.debugLabel)
            .starts_with("ShadowDepthPass.Cascade")) {
      ++sawShadowPassCount;
      lastShadowPassIndex = i;
    }
    if (pass.debugLabel == "Opaque Pass") {
      sawOpaquePass = true;
      opaquePassIndex = i;
    }
  }
  ASSERT_EQ(sawShadowPassCount, 4u);
  ASSERT_TRUE(sawOpaquePass);
  EXPECT_LT(lastShadowPassIndex, opaquePassIndex);
  ASSERT_LT(opaquePassIndex, compiled.passBarrierPlans.size());
  const PassBarrierPlan &opaqueBarrierPlan =
      compiled.passBarrierPlans[opaquePassIndex];
  std::array<bool, kMaxShadowCascades> opaqueReadsShadowTexture{};
  opaqueReadsShadowTexture.fill(false);
  for (uint32_t i = 0u; i < opaqueBarrierPlan.barrierCount; ++i) {
    const RenderGraphBarrierRecord &barrier =
        compiled.passBarrierRecords[opaqueBarrierPlan.barrierOffset + i];
    if (barrier.resourceKind != RenderGraphBarrierResourceKind::Texture ||
        !hasAccessFlag(barrier.afterAccess, RenderGraphAccessMode::Read)) {
      continue;
    }
    for (uint32_t cascadeIndex = 0u; cascadeIndex < kMaxShadowCascades;
         ++cascadeIndex) {
      opaqueReadsShadowTexture[cascadeIndex] |=
          barrier.resourceIndex == shadowTextureResourceIndices[cascadeIndex];
    }
  }
  for (bool readsTexture : opaqueReadsShadowTexture) {
    EXPECT_TRUE(readsTexture);
  }
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
