#include "tests_pch.h"

#include "render_graph_test_support.h"

#include <gtest/gtest.h>

#include "nuri/core/log.h"
#include "nuri/core/runtime_config.h"
#include "nuri/gfx/pipeline/features/composite_feature.h"
#include "nuri/gfx/pipeline/features/debug_feature.h"
#include "nuri/gfx/pipeline/features/msaa_resolve_feature.h"
#include "nuri/gfx/pipeline/features/opaque_feature.h"
#include "nuri/gfx/pipeline/features/shadow_feature.h"
#include "nuri/gfx/pipeline/features/skybox_feature.h"
#include "nuri/gfx/pipeline/features/spatial_aa_feature.h"
#include "nuri/gfx/pipeline/features/temporal_aa_feature.h"
#include "nuri/gfx/pipeline/features/transmission_feature.h"
#include "nuri/gfx/pipeline/features/transparent_feature.h"
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
#include <iterator>
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
constexpr uint32_t kForwardSceneTransmissionMipDebugProbe = 1u << 8u;

struct ShadowPreviewPushConstantsProbe {
  std::array<uint32_t, 4> sourceTexIds{};
  std::array<uint32_t, 4> previewParams{};
  std::array<float, 4> depthParams{};
};

struct GpuSdsmMinMaxResultProbe {
  glm::vec2 rawDeviceMinMax{1.0f, 1.0f};
  uint32_t sourceFrameIndex = 0u;
  uint32_t valid = 0u;
};
static_assert(sizeof(GpuSdsmMinMaxResultProbe) == 16u);

struct alignas(16) GpuSdsmHistogramResultProbe {
  glm::vec4 rawDeviceMinMaxLinearMinMax{1.0f, 0.0f, 0.0f, 0.0f};
  glm::vec4 histogramRangeWeightClear{0.0f};
  glm::uvec4 metadata{0u};
  glm::vec4 splitDepths0{0.0f};
  glm::vec4 splitDepths1{0.0f};
  std::array<float, kMaxShadowSdsmHistogramBucketCount> bucketWeights{};
};
static_assert(sizeof(GpuSdsmHistogramResultProbe) == 592u);

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

RuntimeSkyboxShaderConfig makeSkyboxConfig(const std::filesystem::path &root) {
  const std::filesystem::path shaders = root / "assets" / "shaders";
  return RuntimeSkyboxShaderConfig{
      .vertex = shaders / "skybox.vert",
      .fragment = shaders / "skybox.frag",
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

bool mat4Near(const glm::mat4 &a, const glm::mat4 &b, float epsilon) {
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < 4; ++r) {
      if (std::abs(a[c][r] - b[c][r]) > epsilon) {
        return false;
      }
    }
  }
  return true;
}

glm::vec2 projectedPixelShift(const glm::mat4 &baseProjection,
                              const glm::mat4 &jitteredProjection,
                              const glm::vec4 &viewPosition,
                              glm::uvec2 renderExtent) {
  const glm::vec4 baseClip = baseProjection * viewPosition;
  const glm::vec4 jitteredClip = jitteredProjection * viewPosition;
  const glm::vec2 baseNdc = glm::vec2(baseClip) / baseClip.w;
  const glm::vec2 jitteredNdc = glm::vec2(jitteredClip) / jitteredClip.w;
  return (jitteredNdc - baseNdc) * (0.5f * glm::vec2(renderExtent));
}

glm::vec2 taaScreenUvFromClipNdc(glm::vec2 ndc) {
  return glm::vec2(ndc.x * 0.5f + 0.5f, 0.5f - ndc.y * 0.5f);
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

void expectGpuSdsmResultAvailable(const ShadowSdsmDebugFrameData &sdsm,
                                  uint64_t sourceFrameIndex,
                                  bool splitPayloadValid) {
  EXPECT_GT(sdsm.gpuResultRingSlotCount, 0u);
  EXPECT_NE(sdsm.gpuResultSelectedSlot, std::numeric_limits<uint32_t>::max());
  EXPECT_TRUE(sdsm.gpuReductionResultAvailable);
  EXPECT_EQ(sdsm.gpuSplitPayloadValid, splitPayloadValid);
  EXPECT_EQ(sdsm.gpuResultSourceFrameIndex, sourceFrameIndex);
}

void expectNoGpuSdsmResult(const ShadowSdsmDebugFrameData &sdsm) {
  EXPECT_GT(sdsm.gpuResultRingSlotCount, 0u);
  EXPECT_EQ(sdsm.gpuResultSelectedSlot, std::numeric_limits<uint32_t>::max());
  EXPECT_FALSE(sdsm.gpuReductionResultAvailable);
  EXPECT_FALSE(sdsm.gpuSplitPayloadValid);
  EXPECT_EQ(sdsm.gpuResultSourceFrameIndex,
            std::numeric_limits<uint64_t>::max());
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

class FakeShadowSceneGpuDevice : public FakeFullscreenGpuDevice {
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

class FakeStableBindlessShadowSceneGpuDevice final
    : public FakeShadowSceneGpuDevice {
public:
  uint32_t getTextureBindlessIndex(TextureHandle h) const override {
    return nuri::isValid(h) ? h.index : 0xFFFFFFFFu;
  }
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

struct SpatialAAPushConstantsProbe {
  uint32_t sourceTexId = 0u;
  uint32_t edgeTexId = 0u;
  uint32_t blendTexId = 0u;
  uint32_t areaTexId = 0u;
  uint32_t searchTexId = 0u;
  uint32_t linearSamplerId = 0u;
  uint32_t pointSamplerId = 0u;
  uint32_t mode = 0u;
  uint32_t inverseWidthBits = 0u;
  uint32_t inverseHeightBits = 0u;
  uint32_t edgeThresholdBits = 0u;
  uint32_t maxSearchSteps = 0u;
  uint32_t resolveStrengthBits = 0u;
  uint32_t localContrastFactorBits = 0u;
  uint32_t cornerRoundingBits = 0u;
};
static_assert(sizeof(SpatialAAPushConstantsProbe) <= 128u);

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

std::string readTextFile(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  EXPECT_TRUE(in.is_open());
  if (!in.is_open()) {
    return {};
  }
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

uint32_t compiledPassIndex(std::span<const RenderPass> passes,
                           std::string_view label) {
  for (uint32_t i = 0u; i < passes.size(); ++i) {
    if (passes[i].debugLabel == label) {
      return i;
    }
  }
  return UINT32_MAX;
}

TEST(RenderGraphRendererTest,
     ShadowFeatureGpuHistogramShaderSkipsClearOnlyTiles) {
  const std::string shader =
      readTextFile(std::filesystem::path(PROJECT_SOURCE_DIR) / "assets" /
                   "shaders" / "shadow_sdsm_histogram.comp");
  ASSERT_FALSE(shader.empty());

  const size_t clearOnlySkip = shader.find("if (tileClearMin && tileClearMax)");
  const size_t rawMinUpdate =
      shader.find("rawDeviceMin = min(rawDeviceMin", clearOnlySkip);
  const size_t histogramAdd = shader.find("addHistogramInterval", rawMinUpdate);
  ASSERT_NE(clearOnlySkip, std::string::npos);
  ASSERT_NE(rawMinUpdate, std::string::npos);
  ASSERT_NE(histogramAdd, std::string::npos);
  EXPECT_LT(clearOnlySkip, rawMinUpdate);
  EXPECT_LT(rawMinUpdate, histogramAdd);
  EXPECT_NE(shader.find("bool clearOnly = !anyNonClear && rawValidTiles > 0u"),
            std::string::npos);
  EXPECT_NE(shader.find("splits[i] = clearOnly ? mix(fixedNear, fixedFar"),
            std::string::npos);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureLightingShaderUsesGeometricNormalAndCheapPcssBlend) {
  const std::string shader =
      readTextFile(std::filesystem::path(PROJECT_SOURCE_DIR) / "assets" /
                   "shaders" / "material_lighting.sp");
  ASSERT_FALSE(shader.empty());

  EXPECT_NE(shader.find("shadow, worldPos, sm.nGeom, l,"), std::string::npos);
  EXPECT_EQ(shader.find("shadow, worldPos, sm.nBase, l,"), std::string::npos);
  EXPECT_NE(shader.find("directionalShadowBlendFactorFromContext"),
            std::string::npos);
  EXPECT_NE(shader.find("min(directionalShadowPcssBlockerSampleCount(shadow), "
                        "8)"),
            std::string::npos);
  EXPECT_NE(shader.find("min(directionalShadowPcssFilterSampleCount(shadow), "
                        "16)"),
            std::string::npos);
  EXPECT_NE(shader.find("const uint shadowRawSamplerId = "
                        "ctx.cascade.textureSampler.z"),
            std::string::npos);
  EXPECT_NE(shader.find("const uint storedShadowSize = "
                        "ctx.cascade.textureSampler.w"),
            std::string::npos);
  EXPECT_NE(shader.find("textureBindless2D(shadowTexId, shadowRawSamplerId"),
            std::string::npos);
  EXPECT_NE(shader.find("if (sampleCount <= 1) {\n"
                        "    return hardDirectionalShadowFactor(ctx);"),
            std::string::npos);
  EXPECT_NE(shader.find("if (sampleCount >= 16)"), std::string::npos);
  EXPECT_NE(shader.find("tryResolveUniformPoissonPcf"), std::string::npos);
  EXPECT_NE(shader.find("const vec2 probes[8]"), std::string::npos);
  const size_t fixedRotation =
      shader.find("kShadowFrameFlagFixedPoissonRotation) != 0u");
  const size_t fixedRotationReturn =
      shader.find("return mat2(1.0)", fixedRotation);
  const size_t dynamicRotationSin =
      shader.find("sin(angle)", fixedRotationReturn);
  ASSERT_NE(fixedRotation, std::string::npos);
  ASSERT_NE(fixedRotationReturn, std::string::npos);
  ASSERT_NE(dynamicRotationSin, std::string::npos);
  EXPECT_LT(fixedRotationReturn, dynamicRotationSin);
  EXPECT_NE(shader.find("const vec2 kShadowPoissonDisk[64]"),
            std::string::npos);
  EXPECT_NE(shader.find("return kShadowPoissonDisk[sampleIndex & 63]"),
            std::string::npos);
  const size_t poissonSample =
      shader.find("vec2 shadowPoissonDiskSample(int sampleIndex)");
  const size_t shadowHash = shader.find("uint shadowHash", poissonSample);
  ASSERT_NE(poissonSample, std::string::npos);
  ASSERT_NE(shadowHash, std::string::npos);
  EXPECT_EQ(
      shader.substr(poissonSample, shadowHash - poissonSample).find("switch"),
      std::string::npos);
  const size_t rawTapBoundsCheck =
      shader.find("if (any(lessThan(sampleUv, vec2(0.0)))");
  const size_t rawTapSample =
      shader.find("textureBindless2D(shadowTexId, shadowRawSamplerId, "
                  "sampleUv)");
  ASSERT_NE(rawTapBoundsCheck, std::string::npos);
  ASSERT_NE(rawTapSample, std::string::npos);
  EXPECT_LT(rawTapBoundsCheck, rawTapSample);
}

float spatialAAPushFloat(uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

SpatialAAPushConstantsProbe
readSpatialAAPushConstants(std::span<const RenderPass> passes,
                           std::string_view label) {
  const uint32_t passIndex = compiledPassIndex(passes, label);
  EXPECT_NE(passIndex, UINT32_MAX);
  if (passIndex == UINT32_MAX) {
    return {};
  }
  const RenderPass &pass = passes[passIndex];
  EXPECT_FALSE(pass.draws.empty());
  if (pass.draws.empty()) {
    return {};
  }
  const DrawItem &draw = pass.draws.front();
  EXPECT_EQ(draw.pushConstants.size(), sizeof(SpatialAAPushConstantsProbe));
  SpatialAAPushConstantsProbe probe{};
  if (draw.pushConstants.size() == sizeof(SpatialAAPushConstantsProbe)) {
    std::memcpy(&probe, draw.pushConstants.data(), sizeof(probe));
  }
  return probe;
}

void writeTransmissionTriangleObj(const std::filesystem::path &path) {
  writeTextFile(path, "o TaaTransmissionTriangle\n"
                      "v -1.0 0.0 0.0\n"
                      "v 1.0 0.0 0.0\n"
                      "v 0.0 1.0 0.0\n"
                      "vt 0.0 0.0\n"
                      "vt 1.0 0.0\n"
                      "vt 0.5 1.0\n"
                      "vn 0.0 0.0 1.0\n"
                      "f 1/1/1 2/2/1 3/3/1\n");
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

TEST(RenderGraphRendererTest, RenderSettingsDefaultToAntiAliasingDisabled) {
  const RenderSettings settings{};
  EXPECT_EQ(settings.antiAliasing.mode, AntiAliasingMode::None);
  EXPECT_FALSE(settings.antiAliasing.debug.jitterEnabled);
  EXPECT_FALSE(settings.antiAliasing.debug.freezeJitter);
  EXPECT_FALSE(settings.antiAliasing.debug.resetHistoryRequested);
  EXPECT_FALSE(settings.antiAliasing.debug.logDiagnostics);
  EXPECT_FALSE(settings.antiAliasing.debug.spatialPostTaaCleanup);
  EXPECT_TRUE(settings.antiAliasing.debug.spatialPostMsaaCleanup);
  EXPECT_EQ(settings.antiAliasing.debug.view, AntiAliasingDebugView::None);
  EXPECT_EQ(
      sanitizeAntiAliasingDebugView(AntiAliasingDebugView::TAAPreviousVelocity),
      AntiAliasingDebugView::TAAPreviousVelocity);
  EXPECT_EQ(sanitizeAntiAliasingDebugView(AntiAliasingDebugView::TAAHdrWeight),
            AntiAliasingDebugView::TAAHdrWeight);
  EXPECT_EQ(sanitizeAntiAliasingDebugView(
                AntiAliasingDebugView::TAAHistoryFilterDelta),
            AntiAliasingDebugView::TAAHistoryFilterDelta);
  EXPECT_EQ(sanitizeAntiAliasingDebugView(
                AntiAliasingDebugView::TAADisocclusionFallback),
            AntiAliasingDebugView::TAADisocclusionFallback);
  EXPECT_EQ(
      sanitizeAntiAliasingDebugView(AntiAliasingDebugView::TAASplitCompare),
      AntiAliasingDebugView::TAASplitCompare);
  EXPECT_EQ(
      sanitizeAntiAliasingDebugView(AntiAliasingDebugView::SpatialAAEdges),
      AntiAliasingDebugView::SpatialAAEdges);
  EXPECT_EQ(sanitizeAntiAliasingDebugView(
                AntiAliasingDebugView::SpatialAABlendWeights),
            AntiAliasingDebugView::SpatialAABlendWeights);
  EXPECT_EQ(sanitizeAntiAliasingDebugView(
                AntiAliasingDebugView::SpatialAACleanupMask),
            AntiAliasingDebugView::SpatialAACleanupMask);
  EXPECT_EQ(sanitizeAntiAliasingDebugView(
                AntiAliasingDebugView::SpatialAASplitCompare),
            AntiAliasingDebugView::SpatialAASplitCompare);
  EXPECT_FLOAT_EQ(settings.antiAliasing.debug.diagnosticLogIntervalSeconds,
                  0.25f);
  EXPECT_FLOAT_EQ(settings.antiAliasing.debug.taaJitterScale, 0.75f);
  EXPECT_FLOAT_EQ(settings.antiAliasing.debug.taaVelocityRejectionThreshold,
                  0.0015f);
  EXPECT_FLOAT_EQ(settings.antiAliasing.debug.taaVelocityBlendScale, 0.35f);
  EXPECT_FLOAT_EQ(settings.antiAliasing.debug.taaCurrentFrameWeight, 0.06f);
  EXPECT_FLOAT_EQ(settings.antiAliasing.debug.taaMotionCurrentWeight, 0.35f);
  EXPECT_FLOAT_EQ(settings.antiAliasing.debug.taaDisocclusionCurrentWeight,
                  0.65f);
  EXPECT_FLOAT_EQ(settings.antiAliasing.debug.taaClampCurrentWeight, 0.50f);
  EXPECT_EQ(settings.antiAliasing.debug.taaClampMode,
            TemporalAAClampMode::Variance);
  EXPECT_FLOAT_EQ(settings.antiAliasing.debug.taaVarianceGamma, 1.50f);
}

TEST(RenderGraphRendererTest, Msaa4xAntiAliasingModeSanitizesTemporalState) {
  RenderSettings::AntiAliasingSettings aa{};
  aa.mode = AntiAliasingMode::MSAA4x;
  aa.debug.jitterEnabled = true;
  aa.debug.freezeJitter = true;

  sanitizeAntiAliasingSettings(aa);

  EXPECT_EQ(aa.mode, AntiAliasingMode::MSAA4x);
  EXPECT_EQ(sanitizeAntiAliasingMode(AntiAliasingMode::MSAA4x),
            AntiAliasingMode::MSAA4x);
  EXPECT_FALSE(aa.debug.jitterEnabled);
  EXPECT_FALSE(aa.debug.freezeJitter);
}

TEST(RenderGraphRendererTest, RenderSettingsDefaultToShadowsEnabled) {
  const RenderSettings settings{};
  EXPECT_TRUE(settings.shadow.enabled);
  EXPECT_EQ(settings.shadow.qualityPreset, ShadowQualityPreset::Custom);
  EXPECT_EQ(settings.shadow.cascadeCount, kMaxShadowCascades);
  EXPECT_FLOAT_EQ(settings.shadow.maxDistanceFadeFraction, 0.0f);
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
  settings.qualityPreset = static_cast<ShadowQualityPreset>(99u);
  settings.shadowMapSize = 0u;
  settings.maxDistance = -10.0f;
  settings.maxDistanceFadeFraction = -1.0f;
  settings.splitMode = static_cast<ShadowCascadeSplitMode>(99u);
  settings.splitLambda = 2.0f;
  settings.cascadeBlendFraction = -1.0f;
  settings.constantBias = std::numeric_limits<float>::quiet_NaN();
  settings.slopeBias = std::numeric_limits<float>::quiet_NaN();
  settings.normalBias = std::numeric_limits<float>::quiet_NaN();
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
  EXPECT_EQ(settings.qualityPreset, ShadowQualityPreset::Custom);
  EXPECT_EQ(settings.shadowMapSize, 1u);
  EXPECT_FLOAT_EQ(settings.maxDistance, 150.0f);
  EXPECT_FLOAT_EQ(settings.maxDistanceFadeFraction, 0.0f);
  EXPECT_EQ(settings.splitMode, ShadowCascadeSplitMode::Practical);
  EXPECT_FLOAT_EQ(settings.splitLambda, 1.0f);
  EXPECT_FLOAT_EQ(settings.cascadeBlendFraction, 0.0f);
  EXPECT_FLOAT_EQ(settings.constantBias, 0.0005f);
  EXPECT_FLOAT_EQ(settings.slopeBias, 1.5f);
  EXPECT_FLOAT_EQ(settings.normalBias, 0.0f);
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
  settings.maxDistanceFadeFraction = 2.0f;
  settings.splitLambda = -2.0f;
  settings.cascadeBlendFraction = 2.0f;
  settings.pcfSampleCount = kMaxShadowPcfSamples + 1u;
  settings.debug.previewDepthMin = 0.8f;
  settings.debug.previewDepthMax = 0.2f;
  sanitizeShadowSettings(settings);
  EXPECT_EQ(settings.cascadeCount, kMaxShadowCascades);
  EXPECT_FLOAT_EQ(settings.maxDistanceFadeFraction, 1.0f);
  EXPECT_FLOAT_EQ(settings.splitLambda, 0.0f);
  EXPECT_FLOAT_EQ(settings.cascadeBlendFraction, 1.0f);
  EXPECT_EQ(settings.pcfSampleCount, kMaxShadowPcfSamples);
  EXPECT_LT(settings.debug.previewDepthMin, settings.debug.previewDepthMax);
  EXPECT_NEAR(settings.debug.previewDepthMax - settings.debug.previewDepthMin,
              1.0e-4f, 1.0e-6f);
}

TEST(RenderGraphRendererTest, ShadowQualityPresetAppliesExpectedValues) {
  RenderSettings::ShadowSettings settings{};
  settings.debug.showShadowMapViewport = true;
  applyShadowQualityPreset(settings, ShadowQualityPreset::Low);
  EXPECT_EQ(settings.qualityPreset, ShadowQualityPreset::Low);
  EXPECT_EQ(settings.cascadeCount, 1u);
  EXPECT_EQ(settings.shadowMapSize, 1024u);
  EXPECT_FLOAT_EQ(settings.maxDistance, 80.0f);
  EXPECT_FLOAT_EQ(settings.maxDistanceFadeFraction, 0.0f);
  EXPECT_FLOAT_EQ(settings.splitLambda, 0.35f);
  EXPECT_EQ(settings.filterMode, ShadowFilterMode::PCF3x3);
  EXPECT_EQ(settings.pcfSampleCount, 9u);
  EXPECT_EQ(settings.sdsmMode, ShadowSdsmMode::Disabled);
  EXPECT_FALSE(settings.debug.fixedPoissonRotation);
  EXPECT_FLOAT_EQ(settings.constantBias, 0.0008f);
  EXPECT_FLOAT_EQ(settings.slopeBias, 2.0f);
  EXPECT_FLOAT_EQ(settings.normalBias, 0.25f);
  EXPECT_TRUE(settings.debug.showShadowMapViewport);

  applyShadowQualityPreset(settings, ShadowQualityPreset::Medium);
  EXPECT_EQ(settings.qualityPreset, ShadowQualityPreset::Medium);
  EXPECT_EQ(settings.cascadeCount, 3u);
  EXPECT_EQ(settings.shadowMapSize, 2048u);
  EXPECT_FLOAT_EQ(settings.maxDistance, 120.0f);
  EXPECT_FLOAT_EQ(settings.maxDistanceFadeFraction, 0.0f);
  EXPECT_FLOAT_EQ(settings.splitLambda, 0.30f);
  EXPECT_EQ(settings.filterMode, ShadowFilterMode::PoissonPCF);
  EXPECT_EQ(settings.pcfSampleCount, 16u);
  EXPECT_EQ(settings.sdsmMode, ShadowSdsmMode::PreviousFrameMinMax);
  EXPECT_EQ(settings.sdsmReductionBackend, ShadowSdsmReductionBackend::Auto);
  EXPECT_TRUE(settings.debug.fixedPoissonRotation);
  EXPECT_FLOAT_EQ(settings.constantBias, 0.0006f);
  EXPECT_FLOAT_EQ(settings.slopeBias, 1.75f);
  EXPECT_FLOAT_EQ(settings.normalBias, 0.35f);

  applyShadowQualityPreset(settings, ShadowQualityPreset::High);
  EXPECT_EQ(settings.qualityPreset, ShadowQualityPreset::High);
  EXPECT_EQ(settings.cascadeCount, 4u);
  EXPECT_EQ(settings.shadowMapSize, 4096u);
  EXPECT_FLOAT_EQ(settings.maxDistance, 150.0f);
  EXPECT_FLOAT_EQ(settings.maxDistanceFadeFraction, 0.0f);
  EXPECT_FLOAT_EQ(settings.splitLambda, 0.25f);
  EXPECT_EQ(settings.filterMode, ShadowFilterMode::PoissonPCF);
  EXPECT_EQ(settings.pcfSampleCount, 24u);
  EXPECT_EQ(settings.sdsmMode, ShadowSdsmMode::PreviousFrameMinMax);
  EXPECT_EQ(settings.sdsmReductionBackend, ShadowSdsmReductionBackend::Auto);
  EXPECT_TRUE(settings.debug.fixedPoissonRotation);
  EXPECT_FLOAT_EQ(settings.constantBias, 0.0005f);
  EXPECT_FLOAT_EQ(settings.slopeBias, 1.5f);
  EXPECT_FLOAT_EQ(settings.normalBias, 0.50f);

  applyShadowQualityPreset(settings, ShadowQualityPreset::Ultra);
  EXPECT_EQ(settings.qualityPreset, ShadowQualityPreset::Ultra);
  EXPECT_EQ(settings.cascadeCount, 4u);
  EXPECT_EQ(settings.shadowMapSize, 8192u);
  EXPECT_FLOAT_EQ(settings.maxDistance, 220.0f);
  EXPECT_FLOAT_EQ(settings.maxDistanceFadeFraction, 0.0f);
  EXPECT_FLOAT_EQ(settings.splitLambda, 0.0f);
  EXPECT_EQ(settings.filterMode, ShadowFilterMode::PoissonPCF);
  EXPECT_EQ(settings.pcfSampleCount, 32u);
  EXPECT_EQ(settings.pcssBlockerSampleCount, 16u);
  EXPECT_EQ(settings.pcssFilterSampleCount, 32u);
  EXPECT_EQ(settings.sdsmMode, ShadowSdsmMode::PreviousFrameMinMax);
  EXPECT_EQ(settings.sdsmReductionBackend, ShadowSdsmReductionBackend::Auto);
  EXPECT_TRUE(settings.debug.fixedPoissonRotation);
  EXPECT_FLOAT_EQ(settings.constantBias, 0.0004f);
  EXPECT_FLOAT_EQ(settings.slopeBias, 1.3f);
  EXPECT_FLOAT_EQ(settings.normalBias, 0.60f);
}

TEST(RenderGraphRendererTest, LvkD16DepthFormatMapsToVulkanD16Unorm) {
  EXPECT_EQ(lvk::formatToVkFormat(lvk::Format_Z_UN16), VK_FORMAT_D16_UNORM);
}

TEST(RenderGraphRendererTest, LvkRg16FloatFormatMapsToVulkanRg16Sfloat) {
  EXPECT_EQ(lvk::formatToVkFormat(lvk::Format_RG_F16), VK_FORMAT_R16G16_SFLOAT);
  EXPECT_EQ(lvk::vkFormatToFormat(VK_FORMAT_R16G16_SFLOAT), lvk::Format_RG_F16);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureLogsDiagnosticsAtConfiguredInterval) {
  std::array<std::byte, 128 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
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

TEST(RenderGraphRendererTest,
     TemporalHaltonJitterSequenceStaysWithinHalfPixel) {
  for (uint32_t i = 0u; i < kTemporalJitterSequenceLength; ++i) {
    const glm::vec2 offset = temporalJitterPixelOffset(i);
    EXPECT_LE(std::abs(offset.x), 0.5f);
    EXPECT_LE(std::abs(offset.y), 0.5f);
    EXPECT_TRUE(jitterOffsetWithinHalfPixel(offset));
  }

  const glm::vec2 firstOffset = temporalJitterPixelOffset(0u);
  EXPECT_FLOAT_EQ(firstOffset.x, 0.0f);
  EXPECT_FLOAT_EQ(firstOffset.y, -1.0f / 6.0f);
}

TEST(RenderGraphRendererTest,
     TemporalProjectionJitterMovesSamplesByConfiguredPixelOffset) {
  const glm::uvec2 renderExtent{1920u, 1080u};
  const glm::vec2 jitterOffset{0.375f, -0.25f};

  const glm::mat4 perspective =
      glm::perspective(glm::radians(60.0f), 16.0f / 9.0f, 0.1f, 100.0f);
  const glm::mat4 jitteredPerspective = applyProjectionJitter(
      perspective, jitterOffset, renderExtent, ProjectionType::Perspective);
  const glm::vec2 perspectiveShift =
      projectedPixelShift(perspective, jitteredPerspective,
                          glm::vec4(0.2f, -0.1f, -5.0f, 1.0f), renderExtent);
  EXPECT_NEAR(perspectiveShift.x, jitterOffset.x, 1.0e-4f);
  EXPECT_NEAR(perspectiveShift.y, jitterOffset.y, 1.0e-4f);

  const glm::mat4 orthographic =
      glm::ortho(-8.0f, 8.0f, -4.5f, 4.5f, 0.1f, 100.0f);
  const glm::mat4 jitteredOrthographic = applyProjectionJitter(
      orthographic, jitterOffset, renderExtent, ProjectionType::Orthographic);
  const glm::vec2 orthographicShift =
      projectedPixelShift(orthographic, jitteredOrthographic,
                          glm::vec4(0.2f, -0.1f, -5.0f, 1.0f), renderExtent);
  EXPECT_NEAR(orthographicShift.x, jitterOffset.x, 1.0e-4f);
  EXPECT_NEAR(orthographicShift.y, jitterOffset.y, 1.0e-4f);
}

TEST(RenderGraphRendererTest,
     TemporalVelocityConventionMatchesTaaResolveScreenUv) {
  const glm::vec2 currentNdc{0.25f, -0.40f};
  const glm::vec2 previousNdc{-0.15f, 0.30f};
  const glm::vec2 currentUv = taaScreenUvFromClipNdc(currentNdc);
  const glm::vec2 previousUv = taaScreenUvFromClipNdc(previousNdc);
  const glm::vec2 velocity = previousUv - currentUv;
  const glm::vec2 historyUv = currentUv + velocity;

  EXPECT_NEAR(historyUv.x, previousUv.x, 1.0e-6f);
  EXPECT_NEAR(historyUv.y, previousUv.y, 1.0e-6f);
  EXPECT_FLOAT_EQ(taaScreenUvFromClipNdc(glm::vec2(0.0f, 1.0f)).y, 0.0f);
  EXPECT_FLOAT_EQ(taaScreenUvFromClipNdc(glm::vec2(0.0f, -1.0f)).y, 1.0f);
}

TEST(RenderGraphRendererTest,
     TemporalCameraFrameStateAppliesJitterAndTracksPreviousMatrices) {
  Camera camera{};
  camera.setLookAt(glm::vec3(0.0f, 1.0f, 4.0f), glm::vec3(0.0f, 1.0f, 3.0f),
                   glm::vec3(0.0f, 1.0f, 0.0f));

  RenderSettings::AntiAliasingSettings aa{};
  aa.mode = AntiAliasingMode::TAA;
  aa.debug.jitterEnabled = true;

  TemporalCameraHistoryState history{};
  const TemporalCameraFrameDesc desc{.renderExtent = glm::uvec2(1920u, 1080u)};
  const CameraFrameState first =
      makeTemporalCameraFrameState(camera, 16.0f / 9.0f, aa, desc, history);

  EXPECT_TRUE(first.temporalDataValid);
  EXPECT_TRUE(first.jitterEnabled);
  EXPECT_FALSE(first.historyValid);
  EXPECT_EQ(first.historyResetReason, TemporalHistoryResetReason::FirstFrame);
  EXPECT_EQ(first.jitterIndex, 0u);
  EXPECT_EQ(first.jitterSequenceLength, kTemporalJitterSequenceLength);
  EXPECT_TRUE(jitterOffsetWithinHalfPixel(first.jitterPixelOffset));
  EXPECT_FALSE(mat4Near(first.proj, first.currentUnjitteredProj, 1.0e-8f));
  EXPECT_TRUE(mat4Near(first.previousUnjitteredViewProj,
                       first.currentUnjitteredViewProj, 1.0e-6f));
  EXPECT_TRUE(mat4Near(first.previousJitteredViewProj,
                       first.currentJitteredViewProj, 1.0e-6f));

  const CameraFrameState second =
      makeTemporalCameraFrameState(camera, 16.0f / 9.0f, aa, desc, history);

  EXPECT_TRUE(second.historyValid);
  EXPECT_EQ(second.historyResetReason, TemporalHistoryResetReason::None);
  EXPECT_EQ(second.jitterIndex, 1u);
  EXPECT_TRUE(mat4Near(second.previousUnjitteredViewProj,
                       first.currentUnjitteredViewProj, 1.0e-6f));
  EXPECT_TRUE(mat4Near(second.previousJitteredViewProj,
                       first.currentJitteredViewProj, 1.0e-6f));
  EXPECT_FLOAT_EQ(second.previousJitterPixelOffset.x,
                  first.jitterPixelOffset.x);
  EXPECT_FLOAT_EQ(second.previousJitterPixelOffset.y,
                  first.jitterPixelOffset.y);
  EXPECT_NEAR(second.previousCameraPos.x, first.cameraPos.x, 1.0e-6f);
  EXPECT_NEAR(second.previousCameraPos.y, first.cameraPos.y, 1.0e-6f);
  EXPECT_NEAR(second.previousCameraPos.z, first.cameraPos.z, 1.0e-6f);
}

TEST(RenderGraphRendererTest, TemporalCameraFrameStateScalesJitterOffset) {
  Camera camera{};
  RenderSettings::AntiAliasingSettings aa{};
  aa.mode = AntiAliasingMode::TAA;
  aa.debug.jitterEnabled = true;
  aa.debug.taaJitterScale = 0.5f;

  TemporalCameraHistoryState history{};
  const TemporalCameraFrameDesc desc{.renderExtent = glm::uvec2(1920u, 1080u)};
  const CameraFrameState state =
      makeTemporalCameraFrameState(camera, 16.0f / 9.0f, aa, desc, history);

  const glm::vec2 expected = temporalJitterPixelOffset(0u) * 0.5f;
  EXPECT_FLOAT_EQ(state.jitterPixelOffset.x, expected.x);
  EXPECT_FLOAT_EQ(state.jitterPixelOffset.y, expected.y);
  EXPECT_TRUE(jitterOffsetWithinHalfPixel(state.jitterPixelOffset));
}

TEST(RenderGraphRendererTest,
     TemporalCameraFrameStateForMsaa4xDisablesJitterAndHistory) {
  Camera camera{};
  RenderSettings::AntiAliasingSettings aa{};
  aa.mode = AntiAliasingMode::MSAA4x;
  aa.debug.jitterEnabled = true;
  aa.debug.freezeJitter = true;

  TemporalCameraHistoryState history{};
  history.initialized = true;
  history.previousRenderExtent = glm::uvec2(1920u, 1080u);
  history.previousAntiAliasingMode = AntiAliasingMode::MSAA4x;
  history.previousUnjitteredViewProj = glm::mat4(2.0f);
  history.previousJitteredViewProj = glm::mat4(3.0f);
  history.previousJitterPixelOffset = glm::vec2(0.25f, -0.25f);
  const TemporalCameraFrameDesc desc{.renderExtent = glm::uvec2(1920u, 1080u)};

  const CameraFrameState state =
      makeTemporalCameraFrameState(camera, 16.0f / 9.0f, aa, desc, history);

  EXPECT_TRUE(state.temporalDataValid);
  EXPECT_FALSE(state.jitterEnabled);
  EXPECT_FALSE(state.jitterFrozen);
  EXPECT_FALSE(state.historyValid);
  EXPECT_EQ(state.jitterPixelOffset, glm::vec2(0.0f));
  EXPECT_EQ(state.previousJitterPixelOffset, glm::vec2(0.0f));
  EXPECT_TRUE(mat4Near(state.proj, state.currentUnjitteredProj, 1.0e-8f));
}

TEST(RenderGraphRendererTest, TemporalCameraFrameStateFreezeJitterHoldsIndex) {
  Camera camera{};
  RenderSettings::AntiAliasingSettings aa{};
  aa.mode = AntiAliasingMode::TAA;
  aa.debug.jitterEnabled = true;

  TemporalCameraHistoryState history{};
  const TemporalCameraFrameDesc desc{.renderExtent = glm::uvec2(1280u, 720u)};
  const CameraFrameState first =
      makeTemporalCameraFrameState(camera, 16.0f / 9.0f, aa, desc, history);

  aa.debug.freezeJitter = true;
  const CameraFrameState second =
      makeTemporalCameraFrameState(camera, 16.0f / 9.0f, aa, desc, history);
  const CameraFrameState third =
      makeTemporalCameraFrameState(camera, 16.0f / 9.0f, aa, desc, history);

  EXPECT_EQ(second.jitterIndex, first.jitterIndex);
  EXPECT_EQ(third.jitterIndex, second.jitterIndex);
  EXPECT_FLOAT_EQ(third.jitterPixelOffset.x, second.jitterPixelOffset.x);
  EXPECT_FLOAT_EQ(third.jitterPixelOffset.y, second.jitterPixelOffset.y);
  EXPECT_TRUE(third.jitterFrozen);

  const AntiAliasingFrameMetrics metrics = makeAntiAliasingFrameMetrics(third);
  EXPECT_TRUE(metrics.taaQualityValidationInvalidatedByFrozenJitter);
}

TEST(RenderGraphRendererTest, TemporalCameraFrameStateResetsOnResize) {
  Camera camera{};
  RenderSettings::AntiAliasingSettings aa{};
  aa.mode = AntiAliasingMode::TAA;
  aa.debug.jitterEnabled = true;

  TemporalCameraHistoryState history{};
  const CameraFrameState first = makeTemporalCameraFrameState(
      camera, 16.0f / 9.0f, aa,
      TemporalCameraFrameDesc{.renderExtent = glm::uvec2(1920u, 1080u)},
      history);
  ASSERT_EQ(first.historyResetReason, TemporalHistoryResetReason::FirstFrame);

  const CameraFrameState resized = makeTemporalCameraFrameState(
      camera, 16.0f / 9.0f, aa,
      TemporalCameraFrameDesc{.renderExtent = glm::uvec2(1280u, 720u)},
      history);

  EXPECT_FALSE(resized.historyValid);
  EXPECT_EQ(resized.historyResetReason, TemporalHistoryResetReason::Resize);
  EXPECT_EQ(resized.framesSinceHistoryReset, 0u);
  EXPECT_EQ(resized.historyResetCount, 2u);
}

TEST(RenderGraphRendererTest,
     TemporalCameraFrameStateTracksRenderScaleChanges) {
  Camera camera{};
  RenderSettings::AntiAliasingSettings aa{};
  aa.mode = AntiAliasingMode::TAA;
  aa.debug.jitterEnabled = true;

  TemporalCameraHistoryState history{};
  const TemporalCameraFrameDesc scaledDesc{
      .renderExtent = glm::uvec2(1920u, 1080u),
      .renderScale = glm::vec2(0.75f, 0.75f),
  };
  const CameraFrameState first = makeTemporalCameraFrameState(
      camera, 16.0f / 9.0f, aa, scaledDesc, history);
  ASSERT_EQ(first.historyResetReason, TemporalHistoryResetReason::FirstFrame);

  const CameraFrameState stable = makeTemporalCameraFrameState(
      camera, 16.0f / 9.0f, aa, scaledDesc, history);

  EXPECT_TRUE(stable.historyValid);
  EXPECT_EQ(stable.historyResetReason, TemporalHistoryResetReason::None);
  EXPECT_EQ(stable.historyResetCount, 1u);

  TemporalCameraFrameDesc changedDesc = scaledDesc;
  changedDesc.renderScale = glm::vec2(1.0f, 1.0f);
  const CameraFrameState changed = makeTemporalCameraFrameState(
      camera, 16.0f / 9.0f, aa, changedDesc, history);

  EXPECT_FALSE(changed.historyValid);
  EXPECT_EQ(changed.historyResetReason,
            TemporalHistoryResetReason::RenderScaleChanged);
  EXPECT_EQ(changed.framesSinceHistoryReset, 0u);
  EXPECT_EQ(changed.historyResetCount, 2u);
}

TEST(RenderGraphRendererTest,
     TemporalCameraFrameStateResetsOnInvalidHistoryTexture) {
  Camera camera{};
  RenderSettings::AntiAliasingSettings aa{};
  aa.mode = AntiAliasingMode::TAA;
  aa.debug.jitterEnabled = true;

  TemporalCameraHistoryState history{};
  const TemporalCameraFrameDesc validDesc{
      .renderExtent = glm::uvec2(1920u, 1080u),
  };
  const CameraFrameState first = makeTemporalCameraFrameState(
      camera, 16.0f / 9.0f, aa, validDesc, history);
  ASSERT_EQ(first.historyResetReason, TemporalHistoryResetReason::FirstFrame);

  TemporalCameraFrameDesc invalidHistoryDesc = validDesc;
  invalidHistoryDesc.historyTextureValid = false;
  const CameraFrameState invalidHistory = makeTemporalCameraFrameState(
      camera, 16.0f / 9.0f, aa, invalidHistoryDesc, history);

  EXPECT_FALSE(invalidHistory.historyValid);
  EXPECT_EQ(invalidHistory.historyResetReason,
            TemporalHistoryResetReason::InvalidHistoryTexture);
  EXPECT_EQ(invalidHistory.framesSinceHistoryReset, 0u);
  EXPECT_EQ(invalidHistory.historyResetCount, 2u);
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
     DirectionalShadowCascadeFitKeepsStableTexelSizeAcrossCameraYaw) {
  constexpr uint32_t kShadowMapSize = 4096u;
  const glm::vec3 eye(0.0f, 2.0f, 6.0f);
  const glm::vec3 up(0.0f, 1.0f, 0.0f);

  CameraFrameState cameraA{};
  cameraA.view = glm::lookAt(eye, glm::vec3(0.0f, 0.5f, 0.0f), up);
  cameraA.cameraPos = glm::vec4(eye, 1.0f);
  cameraA.aspectRatio = 16.0f / 9.0f;
  cameraA.projectionType = ProjectionType::Perspective;
  cameraA.nearPlane = 0.1f;
  cameraA.farPlane = 220.0f;
  cameraA.fovYRadians = glm::radians(60.0f);

  CameraFrameState cameraB = cameraA;
  cameraB.view = glm::lookAt(eye, glm::vec3(0.6f, 0.5f, 0.0f), up);

  const glm::vec3 lightDirection =
      glm::normalize(glm::vec3(-0.35f, -1.0f, -0.2f));
  const glm::mat4 lightView =
      shadow_detail::makeDirectionalLightView(lightDirection);

  const shadow_detail::DirectionalShadowFit fitA =
      shadow_detail::fitDirectionalShadowCascadeSliceWithCasterDepthBounds(
          cameraA, 0.1f, 55.0f, lightView, kShadowMapSize, false,
          glm::vec2(0.0f), true);
  const shadow_detail::DirectionalShadowFit fitB =
      shadow_detail::fitDirectionalShadowCascadeSliceWithCasterDepthBounds(
          cameraB, 0.1f, 55.0f, lightView, kShadowMapSize, false,
          glm::vec2(0.0f), true);

  EXPECT_NEAR(fitA.texelWorldSize, fitB.texelWorldSize, 1.0e-6f);
  const glm::vec2 extentA = shadow_detail::orthoExtentFromShadowFit(fitA);
  const glm::vec2 extentB = shadow_detail::orthoExtentFromShadowFit(fitB);
  EXPECT_NEAR(extentA.x, extentB.x, 1.0e-5f);
  EXPECT_NEAR(extentA.y, extentB.y, 1.0e-5f);
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
  FakeShadowSceneGpuDevice gpu;
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
  EXPECT_FALSE(isValid(frameContext.sharedResources.motionVectorTexture));
  EXPECT_FALSE(
      isValid(frameContext.sharedResources.previousMotionVectorTexture));
}

TEST(
    RenderGraphRendererTest,
    FrameCompositionProviderPublishesPreviousMotionVectorsOnlyForValidHistory) {
  std::array<std::byte, 32 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  gpu.swapchainImageCount = 3u;
  Renderer renderer(gpu, memory);
  FrameCompositionProvider provider(gpu, &memory);
  RenderGraphBuilder graph(&memory);
  RenderFrameContext frameContext{};
  frameContext.sharedResources.textureRequirements =
      kBaselineFrameTextureRequirements |
      FrameTextureRequirementFlags::MotionVectors;
  FrameBuildContext ctx{
      .frame = frameContext,
      .graph = graph,
      .resources = renderer.resources(),
      .shared = frameContext.sharedResources,
  };

  frameContext.frameIndex = 0u;
  frameContext.camera.historyValid = false;
  auto prepareResult = provider.prepare(ctx);
  ASSERT_FALSE(prepareResult.hasError());
  ASSERT_TRUE(isValid(frameContext.sharedResources.motionVectorTexture));
  EXPECT_FALSE(
      isValid(frameContext.sharedResources.previousMotionVectorTexture));
  const TextureHandle firstMotionVector =
      frameContext.sharedResources.motionVectorTexture;
  const TextureHandle firstHistoryWrite =
      frameContext.sharedResources.historyColorWriteTexture;

  frameContext.frameIndex = 1u;
  frameContext.camera.historyValid = true;
  prepareResult = provider.prepare(ctx);
  ASSERT_FALSE(prepareResult.hasError());
  ASSERT_TRUE(isValid(frameContext.sharedResources.motionVectorTexture));
  ASSERT_TRUE(
      isValid(frameContext.sharedResources.previousMotionVectorTexture));
  EXPECT_TRUE(
      sameTexture(frameContext.sharedResources.previousMotionVectorTexture,
                  firstMotionVector));
  EXPECT_FALSE(sameTexture(frameContext.sharedResources.motionVectorTexture,
                           firstMotionVector));
  EXPECT_TRUE(sameTexture(frameContext.sharedResources.historyColorReadTexture,
                          firstHistoryWrite));
  const TextureHandle secondHistoryWrite =
      frameContext.sharedResources.historyColorWriteTexture;
  EXPECT_FALSE(sameTexture(secondHistoryWrite, firstHistoryWrite));

  frameContext.frameIndex = 2u;
  prepareResult = provider.prepare(ctx);
  ASSERT_FALSE(prepareResult.hasError());
  EXPECT_TRUE(sameTexture(frameContext.sharedResources.historyColorReadTexture,
                          secondHistoryWrite));
  EXPECT_FALSE(
      sameTexture(frameContext.sharedResources.historyColorWriteTexture,
                  firstHistoryWrite));
  EXPECT_FALSE(
      sameTexture(frameContext.sharedResources.historyColorWriteTexture,
                  secondHistoryWrite));

  uint32_t motionVectorTextureCount = 0u;
  for (const TextureDesc &desc : gpu.createdTextureDescs) {
    if (desc.format == kFrameCompositionMotionVectorFormat) {
      ++motionVectorTextureCount;
    }
  }
  EXPECT_EQ(motionVectorTextureCount, 3u);
}

TEST(
    RenderGraphRendererTest,
    FrameCompositionProviderAddsMotionVectorsWithoutRecreatingBaselineTextures) {
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
  const TextureHandle sceneColor =
      frameContext.sharedResources.sceneColorTexture;
  const TextureHandle frameColor =
      frameContext.sharedResources.frameColorTexture;
  const TextureHandle sceneDepth =
      frameContext.sharedResources.sceneDepthTexture;
  const TextureHandle historyRead =
      frameContext.sharedResources.historyColorReadTexture;
  const TextureHandle historyWrite =
      frameContext.sharedResources.historyColorWriteTexture;
  const uint32_t destroyedBeforeMotion = gpu.destroyedTextureCount;

  frameContext.sharedResources.textureRequirements |=
      FrameTextureRequirementFlags::MotionVectors;
  frameContext.camera.historyValid = true;
  prepareResult = provider.prepare(ctx);
  ASSERT_FALSE(prepareResult.hasError());

  EXPECT_TRUE(
      sameTexture(frameContext.sharedResources.sceneColorTexture, sceneColor));
  EXPECT_TRUE(
      sameTexture(frameContext.sharedResources.frameColorTexture, frameColor));
  EXPECT_TRUE(
      sameTexture(frameContext.sharedResources.sceneDepthTexture, sceneDepth));
  EXPECT_TRUE(sameTexture(frameContext.sharedResources.historyColorReadTexture,
                          historyRead));
  EXPECT_TRUE(sameTexture(frameContext.sharedResources.historyColorWriteTexture,
                          historyWrite));
  EXPECT_EQ(gpu.destroyedTextureCount, destroyedBeforeMotion);
  EXPECT_TRUE(isValid(frameContext.sharedResources.motionVectorTexture));
  EXPECT_TRUE(
      isValid(frameContext.sharedResources.previousMotionVectorTexture));
  EXPECT_TRUE(frameContext.metrics.antiAliasing.motionVectorAllocated);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.previousMotionVectorValid);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.motionVectorFormatSupported);
  EXPECT_EQ(frameContext.metrics.antiAliasing.motionVectorFormat,
            kFrameCompositionMotionVectorFormat);
  EXPECT_EQ(frameContext.metrics.antiAliasing.motionVectorWidth, 1280u);
  EXPECT_EQ(frameContext.metrics.antiAliasing.motionVectorHeight, 720u);
  EXPECT_EQ(frameContext.metrics.antiAliasing.motionVectorTextureCount, 2u);
  EXPECT_EQ(frameContext.metrics.antiAliasing.motionVectorAllocationCount, 2u);
  EXPECT_EQ(frameContext.metrics.antiAliasing.motionVectorReallocationCount,
            0u);
  EXPECT_EQ(frameContext.metrics.antiAliasing.motionVectorRg32FallbackCount,
            0u);
  EXPECT_EQ(frameContext.metrics.antiAliasing.motionVectorTextureBytes,
            1280ull * 720ull * 4ull);
  EXPECT_EQ(frameContext.metrics.antiAliasing.previousMotionVectorTextureBytes,
            1280ull * 720ull * 4ull);
  EXPECT_EQ(frameContext.metrics.antiAliasing.motionVectorTotalBytes,
            2ull * 1280ull * 720ull * 4ull);
  EXPECT_EQ(frameContext.metrics.antiAliasing.motionVectorClearBytes, 0ull);

  gpu.resizeSwapchain(1920, 1080);
  frameContext.frameIndex = 1u;
  prepareResult = provider.prepare(ctx);
  ASSERT_FALSE(prepareResult.hasError());
  EXPECT_TRUE(frameContext.metrics.antiAliasing.motionVectorAllocated);
  EXPECT_EQ(frameContext.metrics.antiAliasing.motionVectorAllocationCount, 4u);
  EXPECT_EQ(frameContext.metrics.antiAliasing.motionVectorReallocationCount,
            1u);
  EXPECT_EQ(frameContext.metrics.antiAliasing.motionVectorTextureBytes,
            1920ull * 1080ull * 4ull);
  EXPECT_EQ(frameContext.metrics.antiAliasing.motionVectorTotalBytes,
            2ull * 1920ull * 1080ull * 4ull);
  EXPECT_EQ(frameContext.metrics.antiAliasing.motionVectorClearBytes, 0ull);
}

TEST(RenderGraphRendererTest,
     FrameCompositionProviderAllocatesMsaa4xSceneTexturesOnlyForMsaaMode) {
  std::array<std::byte, 32 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  FrameCompositionProvider provider(gpu, &memory);
  RenderGraphBuilder graph(&memory);
  RenderSettings settings{};
  settings.antiAliasing.mode = AntiAliasingMode::MSAA4x;
  RenderFrameContext frameContext{};
  frameContext.settings = &settings;
  FrameBuildContext ctx{
      .frame = frameContext,
      .graph = graph,
      .resources = renderer.resources(),
      .shared = frameContext.sharedResources,
  };

  auto prepareResult = provider.prepare(ctx);
  ASSERT_FALSE(prepareResult.hasError());
  ASSERT_TRUE(prepareResult.value());

  EXPECT_TRUE(isValid(frameContext.sharedResources.sceneColorTexture));
  EXPECT_TRUE(isValid(frameContext.sharedResources.sceneDepthTexture));
  EXPECT_TRUE(isValid(frameContext.sharedResources.msaaSceneColorTexture));
  EXPECT_TRUE(isValid(frameContext.sharedResources.msaaSceneDepthTexture));
  EXPECT_TRUE(hasFrameTextureRequirementFlag(
      frameContext.sharedResources.textureRequirements,
      FrameTextureRequirementFlags::MsaaSceneColor));
  EXPECT_TRUE(hasFrameTextureRequirementFlag(
      frameContext.sharedResources.textureRequirements,
      FrameTextureRequirementFlags::MsaaSceneDepth));
  EXPECT_TRUE(frameContext.metrics.antiAliasing.msaaEnabled);
  EXPECT_EQ(frameContext.metrics.antiAliasing.msaaSampleCount,
            kMsaa4xSampleCount);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.msaaColorAllocated);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.msaaDepthAllocated);

  uint32_t msaaSceneColorCount = 0u;
  uint32_t msaaSceneDepthCount = 0u;
  for (const TextureDesc &desc : gpu.createdTextureDescs) {
    if (desc.numSamples != kMsaa4xSampleCount) {
      continue;
    }
    EXPECT_EQ(desc.usage, TextureUsage::Attachment);
    EXPECT_EQ(desc.storage, Storage::Device);
    if (desc.format == kFrameCompositionSceneColorFormat) {
      ++msaaSceneColorCount;
    } else if (desc.format == kFrameCompositionDepthFormat) {
      ++msaaSceneDepthCount;
    }
  }
  EXPECT_EQ(msaaSceneColorCount, gpu.swapchainImageCount);
  EXPECT_EQ(msaaSceneDepthCount, gpu.swapchainImageCount);

  FakeFullscreenGpuDevice nonMsaaGpu;
  Renderer nonMsaaRenderer(nonMsaaGpu, memory);
  FrameCompositionProvider nonMsaaProvider(nonMsaaGpu, &memory);
  RenderGraphBuilder nonMsaaGraph(&memory);
  RenderFrameContext nonMsaaFrame{};
  FrameBuildContext nonMsaaCtx{
      .frame = nonMsaaFrame,
      .graph = nonMsaaGraph,
      .resources = nonMsaaRenderer.resources(),
      .shared = nonMsaaFrame.sharedResources,
  };
  prepareResult = nonMsaaProvider.prepare(nonMsaaCtx);
  ASSERT_FALSE(prepareResult.hasError());
  EXPECT_FALSE(isValid(nonMsaaFrame.sharedResources.msaaSceneColorTexture));
  EXPECT_FALSE(isValid(nonMsaaFrame.sharedResources.msaaSceneDepthTexture));
  for (const TextureDesc &desc : nonMsaaGpu.createdTextureDescs) {
    EXPECT_EQ(desc.numSamples, 1u);
  }
}

TEST(
    RenderGraphRendererTest,
    FrameCompositionProviderAddsReactiveMaskWithoutRecreatingBaselineTextures) {
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
  const TextureHandle sceneColor =
      frameContext.sharedResources.sceneColorTexture;
  const TextureHandle frameColor =
      frameContext.sharedResources.frameColorTexture;
  const TextureHandle sceneDepth =
      frameContext.sharedResources.sceneDepthTexture;
  const TextureHandle historyRead =
      frameContext.sharedResources.historyColorReadTexture;
  const TextureHandle historyWrite =
      frameContext.sharedResources.historyColorWriteTexture;
  const uint32_t destroyedBeforeReactive = gpu.destroyedTextureCount;

  frameContext.sharedResources.textureRequirements |=
      FrameTextureRequirementFlags::ReactiveMask;
  prepareResult = provider.prepare(ctx);
  ASSERT_FALSE(prepareResult.hasError());

  EXPECT_TRUE(
      sameTexture(frameContext.sharedResources.sceneColorTexture, sceneColor));
  EXPECT_TRUE(
      sameTexture(frameContext.sharedResources.frameColorTexture, frameColor));
  EXPECT_TRUE(
      sameTexture(frameContext.sharedResources.sceneDepthTexture, sceneDepth));
  EXPECT_TRUE(sameTexture(frameContext.sharedResources.historyColorReadTexture,
                          historyRead));
  EXPECT_TRUE(sameTexture(frameContext.sharedResources.historyColorWriteTexture,
                          historyWrite));
  EXPECT_EQ(gpu.destroyedTextureCount, destroyedBeforeReactive);
  EXPECT_TRUE(isValid(frameContext.sharedResources.reactiveMaskTexture));
  EXPECT_TRUE(frameContext.metrics.antiAliasing.reactiveMaskAllocated);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.reactiveMaskFormatSupported);
  EXPECT_EQ(frameContext.metrics.antiAliasing.reactiveMaskWidth, 1280u);
  EXPECT_EQ(frameContext.metrics.antiAliasing.reactiveMaskHeight, 720u);
  EXPECT_EQ(frameContext.metrics.antiAliasing.reactiveMaskTextureCount, 2u);
  EXPECT_EQ(frameContext.metrics.antiAliasing.reactiveMaskAllocationCount, 2u);
  EXPECT_EQ(frameContext.metrics.antiAliasing.reactiveMaskReallocationCount,
            0u);
  EXPECT_EQ(frameContext.metrics.antiAliasing.reactiveMaskTextureBytes,
            1280ull * 720ull * 4ull);
  EXPECT_EQ(frameContext.metrics.antiAliasing.reactiveMaskTotalBytes,
            2ull * 1280ull * 720ull * 4ull);
}

TEST(RenderGraphRendererTest,
     TemporalAAFeatureRequestsAndClearsMotionVectorTexture) {
  std::array<std::byte, 32 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(
      std::make_unique<FrameCompositionProvider>(gpu, &memory));
  pipeline.addFeature(std::make_unique<TemporalAAFeature>(
      gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));

  RenderSettings settings{};
  settings.antiAliasing.mode = AntiAliasingMode::TAA;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 0u;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());

  EXPECT_TRUE(hasFrameTextureRequirementFlag(
      frameContext.sharedResources.textureRequirements,
      FrameTextureRequirementFlags::MotionVectors));
  EXPECT_TRUE(hasFrameTextureRequirementFlag(
      frameContext.sharedResources.textureRequirements,
      FrameTextureRequirementFlags::ReactiveMask));
  ASSERT_TRUE(isValid(frameContext.sharedResources.motionVectorTexture));
  ASSERT_TRUE(isValid(frameContext.sharedResources.reactiveMaskTexture));
  EXPECT_FALSE(
      isValid(frameContext.sharedResources.previousMotionVectorTexture));
  EXPECT_TRUE(isValid(frameContext.sharedResources.motionVectorGraphTexture));
  EXPECT_FALSE(
      isValid(frameContext.sharedResources.previousMotionVectorGraphTexture));
  EXPECT_TRUE(frameContext.metrics.antiAliasing.motionVectorAllocated);
  EXPECT_FALSE(frameContext.metrics.antiAliasing.previousMotionVectorValid);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.motionVectorGraphPublished);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.reactiveMaskGraphPublished);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.reactiveMaskAllocated);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.reactiveMaskFormatSupported);
  EXPECT_FALSE(
      frameContext.metrics.antiAliasing.previousMotionVectorGraphPublished);
  EXPECT_EQ(frameContext.metrics.antiAliasing.motionVectorClearPassCount, 1u);
  EXPECT_EQ(frameContext.metrics.antiAliasing.motionVectorClearBytes,
            1280ull * 720ull * 4ull);
  EXPECT_EQ(frameContext.metrics.antiAliasing.taaResolvePassCount, 1u);
  EXPECT_EQ(frameContext.metrics.antiAliasing.taaCopyBackPassCount, 1u);
  EXPECT_EQ(frameContext.metrics.antiAliasing.taaCurrentFallbackFrameCount, 1u);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.taaResolvedSceneColorPublished);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.taaOutOfBoundsFallbackEnabled);
  EXPECT_FALSE(frameContext.metrics.antiAliasing.taaBilinearHistorySampling);
  EXPECT_EQ(sanitizeTemporalAAHistoryFilterMode(
                static_cast<TemporalAAHistoryFilterMode>(0)),
            TemporalAAHistoryFilterMode::CatmullRom);
  EXPECT_EQ(frameContext.metrics.antiAliasing.taaHistoryFilterMode,
            TemporalAAHistoryFilterMode::CatmullRom);
  EXPECT_EQ(frameContext.metrics.antiAliasing.taaResolveWidth, 1280u);
  EXPECT_EQ(frameContext.metrics.antiAliasing.taaResolveHeight, 720u);
  EXPECT_FLOAT_EQ(frameContext.metrics.antiAliasing.taaCurrentFrameWeight,
                  0.06f);
  EXPECT_FLOAT_EQ(frameContext.metrics.antiAliasing.taaHistoryFrameWeight,
                  0.94f);
  EXPECT_FLOAT_EQ(frameContext.metrics.antiAliasing.taaJitterScale, 0.75f);
  EXPECT_FLOAT_EQ(
      frameContext.metrics.antiAliasing.taaVelocityRejectionThreshold, 0.0015f);
  EXPECT_FLOAT_EQ(frameContext.metrics.antiAliasing.taaVelocityBlendScale,
                  0.35f);
  EXPECT_FLOAT_EQ(frameContext.metrics.antiAliasing.taaMotionCurrentWeight,
                  0.35f);
  EXPECT_FLOAT_EQ(frameContext.metrics.antiAliasing.taaClampCurrentWeight,
                  0.50f);
  EXPECT_EQ(frameContext.metrics.antiAliasing.taaClampMode,
            TemporalAAClampMode::Variance);
  EXPECT_FLOAT_EQ(frameContext.metrics.antiAliasing.taaVarianceGamma, 1.50f);
  EXPECT_EQ(
      gpu.getTextureFormat(frameContext.sharedResources.motionVectorTexture),
      kFrameCompositionMotionVectorFormat);
  EXPECT_EQ(
      gpu.getTextureFormat(frameContext.sharedResources.reactiveMaskTexture),
      kFrameCompositionReactiveMaskFormat);
  const TextureDimensions motionVectorDimensions = gpu.getTextureDimensions(
      frameContext.sharedResources.motionVectorTexture);
  EXPECT_EQ(motionVectorDimensions.width, 1280u);
  EXPECT_EQ(motionVectorDimensions.height, 720u);
  EXPECT_EQ(motionVectorDimensions.depth, 1u);

  uint32_t motionVectorTextureCount = 0u;
  for (const TextureDesc &desc : gpu.createdTextureDescs) {
    if (desc.format != kFrameCompositionMotionVectorFormat) {
      continue;
    }
    ++motionVectorTextureCount;
    EXPECT_EQ(desc.usage, TextureUsage::AttachmentSampled);
    EXPECT_EQ(desc.dimensions.width, 1280u);
    EXPECT_EQ(desc.dimensions.height, 720u);
  }
  EXPECT_EQ(motionVectorTextureCount, 2u);
  uint32_t reactiveMaskTextureCount = 0u;
  for (const TextureDesc &desc : gpu.createdTextureDescs) {
    if (desc.format != kFrameCompositionReactiveMaskFormat) {
      continue;
    }
    ++reactiveMaskTextureCount;
    EXPECT_EQ(desc.usage, TextureUsage::AttachmentSampled);
    EXPECT_EQ(desc.dimensions.width, 1280u);
    EXPECT_EQ(desc.dimensions.height, 720u);
  }
  EXPECT_EQ(reactiveMaskTextureCount, 2u);

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const RenderGraphCompileResult &compiled = compileResult.value();
  ASSERT_EQ(compiled.orderedPasses.size(), 4u);
  const RenderPass &clearPass = compiled.orderedPasses[0];
  EXPECT_EQ(clearPass.debugLabel, "Temporal AA Motion Vector Clear");
  EXPECT_TRUE(clearPass.hasColorAttachment);
  EXPECT_TRUE(sameTexture(clearPass.colorTexture,
                          frameContext.sharedResources.motionVectorTexture));
  EXPECT_EQ(clearPass.color.loadOp, LoadOp::Clear);
  EXPECT_EQ(clearPass.color.storeOp, StoreOp::Store);
  EXPECT_FLOAT_EQ(clearPass.color.clearColor.r,
                  kFrameCompositionMotionVectorClearValue.r);
  EXPECT_FLOAT_EQ(clearPass.color.clearColor.g,
                  kFrameCompositionMotionVectorClearValue.g);
  EXPECT_FLOAT_EQ(clearPass.color.clearColor.b,
                  kFrameCompositionMotionVectorClearValue.b);
  EXPECT_FLOAT_EQ(clearPass.color.clearColor.a,
                  kFrameCompositionMotionVectorClearValue.a);

  const RenderPass &reactiveClearPass = compiled.orderedPasses[1];
  EXPECT_EQ(reactiveClearPass.debugLabel, "Temporal AA Reactive Mask Clear");
  EXPECT_TRUE(sameTexture(reactiveClearPass.colorTexture,
                          frameContext.sharedResources.reactiveMaskTexture));
  EXPECT_EQ(reactiveClearPass.color.loadOp, LoadOp::Clear);
  EXPECT_FLOAT_EQ(reactiveClearPass.color.clearColor.r,
                  kFrameCompositionReactiveMaskClearValue.r);

  const RenderPass &resolvePass = compiled.orderedPasses[2];
  EXPECT_EQ(resolvePass.debugLabel, "TAA Resolve Pass");
  EXPECT_TRUE(
      sameTexture(resolvePass.colorTexture,
                  frameContext.sharedResources.historyColorWriteTexture));
  EXPECT_FALSE(resolvePass.draws.empty());

  const RenderPass &copyBackPass = compiled.orderedPasses[3];
  EXPECT_EQ(copyBackPass.debugLabel, "TAA Copy Back Pass");
  EXPECT_TRUE(sameTexture(copyBackPass.colorTexture,
                          frameContext.sharedResources.sceneColorTexture));
  EXPECT_TRUE(sameTexture(frameContext.sharedResources.sceneColorTexture,
                          copyBackPass.colorTexture));
  EXPECT_TRUE(isValid(frameContext.sharedResources.sceneColorGraphTexture));
}

TEST(RenderGraphRendererTest,
     MsaaResolveFeaturePublishesSingleSampleSceneTargets) {
  std::array<std::byte, 64 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  gpu.latestCompletedGpuTimingReport = GpuTimingReport{
      .msaaResolveSourceFrameIndex = 42u,
      .msaaResolveTimeMs = 0.21f,
      .availableScopeMask = kGpuTimingScopeMsaaResolveBit,
  };
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(
      std::make_unique<FrameCompositionProvider>(gpu, &memory));
  pipeline.addFeature(std::make_unique<MsaaResolveFeature>(gpu));

  RenderSettings settings{};
  settings.antiAliasing.mode = AntiAliasingMode::MSAA4x;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 0u;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());

  EXPECT_FALSE(hasFrameTextureRequirementFlag(
      frameContext.sharedResources.textureRequirements,
      FrameTextureRequirementFlags::MotionVectors));
  EXPECT_FALSE(hasFrameTextureRequirementFlag(
      frameContext.sharedResources.textureRequirements,
      FrameTextureRequirementFlags::ReactiveMask));
  EXPECT_TRUE(isValid(frameContext.sharedResources.msaaSceneColorTexture));
  EXPECT_TRUE(isValid(frameContext.sharedResources.msaaSceneDepthTexture));
  EXPECT_TRUE(isValid(frameContext.sharedResources.sceneColorGraphTexture));
  EXPECT_TRUE(isValid(frameContext.sharedResources.sceneDepthGraphTexture));
  EXPECT_EQ(frameContext.metrics.antiAliasing.msaaResolvePassCount, 1u);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.msaaColorResolveTargetBound);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.msaaDepthResolveTargetBound);
  EXPECT_EQ(frameContext.metrics.antiAliasing.msaaResolveGpuTimingAvailable,
            1u);
  EXPECT_FLOAT_EQ(frameContext.metrics.antiAliasing.msaaResolveGpuTimeMs,
                  0.21f);
  EXPECT_EQ(
      frameContext.metrics.antiAliasing.msaaResolveGpuTimingSourceFrameIndex,
      42u);

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const auto &passes = compileResult.value().orderedPasses;
  ASSERT_EQ(passes.size(), 1u);
  const RenderPass &resolvePass = passes.front();
  EXPECT_EQ(resolvePass.debugLabel, "MSAA Resolve Pass");
  EXPECT_TRUE(sameTexture(resolvePass.colorTexture,
                          frameContext.sharedResources.msaaSceneColorTexture));
  EXPECT_TRUE(sameTexture(resolvePass.depthTexture,
                          frameContext.sharedResources.msaaSceneDepthTexture));
  EXPECT_TRUE(sameTexture(resolvePass.colorResolveTexture,
                          frameContext.sharedResources.sceneColorTexture));
  EXPECT_TRUE(sameTexture(resolvePass.depthResolveTexture,
                          frameContext.sharedResources.sceneDepthTexture));
  EXPECT_EQ(resolvePass.color.storeOp, StoreOp::MsaaResolve);
  EXPECT_EQ(resolvePass.depth.storeOp, StoreOp::MsaaResolve);
  EXPECT_EQ(resolvePass.color.resolveMode, ResolveMode::Average);
  EXPECT_EQ(resolvePass.depth.resolveMode, ResolveMode::Min);
  EXPECT_EQ(resolvePass.gpuTimingScope, GpuTimingScope::MsaaResolve);
  EXPECT_TRUE(resolvePass.draws.empty());
}

TEST(RenderGraphRendererTest,
     Msaa4xRendererPipelinesKeepSampleCountScopedToSceneTargets) {
  std::array<std::byte, 128 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  gpu.latestCompletedGpuTimingReport = GpuTimingReport{
      .opaqueSourceFrameIndex = 43u,
      .opaqueTimeMs = 1.25f,
      .availableScopeMask = kGpuTimingScopeOpaqueBit,
  };
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(
      std::make_unique<FrameCompositionProvider>(gpu, &memory));
  pipeline.addProvider(std::make_unique<MaterialTableGpuProvider>(gpu));
  pipeline.addProvider(std::make_unique<SceneLightingProvider>(gpu));

  const std::filesystem::path root = std::filesystem::path(PROJECT_SOURCE_DIR);
  pipeline.addFeature(
      std::make_unique<SkyboxFeature>(gpu, makeSkyboxConfig(root)));
  pipeline.addFeature(
      std::make_unique<OpaqueFeature>(gpu, makeOpaqueConfig(root), &memory));
  pipeline.addFeature(std::make_unique<MsaaResolveFeature>(gpu));

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());

  RenderSettings settings{};
  settings.antiAliasing.mode = AntiAliasingMode::MSAA4x;
  settings.opaque.enableDepthPyramid = false;
  settings.opaque.enableInstanceAnimation = false;
  settings.opaque.enableInstanceCompute = false;
  settings.opaque.enableIndirectDraw = false;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 0u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());

  const auto pipelineDescFor =
      [&gpu](std::string_view debugName) -> const RenderPipelineDesc * {
    for (size_t i = 0u; i < gpu.createdRenderPipelineNames.size(); ++i) {
      if (gpu.createdRenderPipelineNames[i] == debugName) {
        return &gpu.createdRenderPipelineDescs[i];
      }
    }
    return nullptr;
  };
  const auto expectSampleCount = [&pipelineDescFor](std::string_view debugName,
                                                    uint32_t sampleCount) {
    const RenderPipelineDesc *desc = pipelineDescFor(debugName);
    ASSERT_NE(desc, nullptr) << debugName;
    EXPECT_EQ(desc->numSamples, sampleCount) << debugName;
  };
  const auto expectMsaaCoverageState =
      [&pipelineDescFor](std::string_view debugName, bool alphaToCoverage,
                         float minSampleShading) {
        const RenderPipelineDesc *desc = pipelineDescFor(debugName);
        ASSERT_NE(desc, nullptr) << debugName;
        EXPECT_EQ(desc->numSamples, kMsaa4xSampleCount) << debugName;
        EXPECT_EQ(desc->alphaToCoverageEnabled, alphaToCoverage) << debugName;
        EXPECT_FLOAT_EQ(desc->minSampleShading, minSampleShading) << debugName;
      };

  expectSampleCount("skybox", 1u);
  expectSampleCount("skybox_msaa4x", kMsaa4xSampleCount);
  expectSampleCount("opaque_mesh", 1u);
  expectSampleCount("opaque_mesh_msaa4x", kMsaa4xSampleCount);
  expectSampleCount("opaque_mesh_double_sided", 1u);
  expectSampleCount("opaque_mesh_double_sided_msaa4x", kMsaa4xSampleCount);
  expectMsaaCoverageState("opaque_mesh_msaa4x", false, 0.0f);
  expectMsaaCoverageState("opaque_mesh_double_sided_msaa4x", false, 0.0f);
  expectMsaaCoverageState("opaque_mesh_alpha_msaa4x", true, 1.0f);
  expectMsaaCoverageState("opaque_mesh_alpha_double_sided_msaa4x", true, 1.0f);
  if (pipelineDescFor("opaque_mesh_tess_msaa4x") != nullptr) {
    expectMsaaCoverageState("opaque_mesh_tess_msaa4x", false, 0.0f);
    expectMsaaCoverageState("opaque_mesh_tess_alpha_msaa4x", true, 1.0f);
  }
  if (pipelineDescFor("opaque_mesh_tess_double_sided_msaa4x") != nullptr) {
    expectMsaaCoverageState("opaque_mesh_tess_double_sided_msaa4x", false,
                            0.0f);
    expectMsaaCoverageState("opaque_mesh_tess_alpha_double_sided_msaa4x", true,
                            1.0f);
  }
  expectSampleCount("opaque_mesh_depth", 1u);
  expectSampleCount("opaque_mesh_depth_msaa4x", kMsaa4xSampleCount);
  expectSampleCount("opaque_mesh_depth_alpha", 1u);
  expectSampleCount("opaque_mesh_depth_alpha_msaa4x", kMsaa4xSampleCount);
  expectSampleCount("opaque_mesh_pick", 1u);
  expectSampleCount("opaque_velocity", 1u);
  expectSampleCount("opaque_reactive_mask", 1u);
  expectSampleCount("opaque_mesh_shadow_inspect", 1u);

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const RenderPass *opaquePass = nullptr;
  uint32_t standaloneResolvePassCount = 0u;
  for (const RenderPass &pass : compileResult.value().orderedPasses) {
    if (pass.debugLabel == "Opaque Pass") {
      opaquePass = &pass;
    } else if (pass.debugLabel == "MSAA Resolve Pass") {
      ++standaloneResolvePassCount;
    }
  }
  ASSERT_NE(opaquePass, nullptr);
  EXPECT_EQ(standaloneResolvePassCount, 0u);
  EXPECT_EQ(opaquePass->gpuTimingScope, GpuTimingScope::Opaque);
  EXPECT_EQ(frameContext.metrics.opaque.gpuTimingAvailable, 1u);
  EXPECT_FLOAT_EQ(frameContext.metrics.opaque.gpuTimeMs, 1.25f);
  EXPECT_EQ(frameContext.metrics.opaque.gpuTimingSourceFrameIndex, 43u);
  EXPECT_TRUE(sameTexture(opaquePass->colorTexture,
                          frameContext.sharedResources.msaaSceneColorTexture));
  EXPECT_TRUE(sameTexture(opaquePass->colorResolveTexture,
                          frameContext.sharedResources.sceneColorTexture));
  EXPECT_TRUE(sameTexture(opaquePass->depthTexture,
                          frameContext.sharedResources.msaaSceneDepthTexture));
  EXPECT_TRUE(sameTexture(opaquePass->depthResolveTexture,
                          frameContext.sharedResources.sceneDepthTexture));
  EXPECT_EQ(opaquePass->color.storeOp, StoreOp::MsaaResolve);
  EXPECT_EQ(opaquePass->depth.storeOp, StoreOp::MsaaResolve);
  EXPECT_EQ(opaquePass->color.resolveMode, ResolveMode::Average);
  EXPECT_EQ(opaquePass->depth.resolveMode, ResolveMode::Min);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.msaaColorResolveTargetBound);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.msaaDepthResolveTargetBound);
}

TEST(RenderGraphRendererTest,
     Msaa4xOpaquePipelineSelectionSurvivesModeSwitches) {
  std::array<std::byte, 512 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeShadowSceneGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(
      std::make_unique<FrameCompositionProvider>(gpu, &memory));
  pipeline.addProvider(std::make_unique<MaterialTableGpuProvider>(gpu));
  pipeline.addProvider(std::make_unique<SceneLightingProvider>(gpu));
  pipeline.addFeature(std::make_unique<OpaqueFeature>(
      gpu, makeOpaqueConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
      &memory));
  pipeline.addFeature(std::make_unique<MsaaResolveFeature>(gpu));

  const std::filesystem::path tempDir =
      makeTempRendererPath("msaa_pipeline_switch");
  ASSERT_TRUE(std::filesystem::create_directories(tempDir));
  const std::filesystem::path objPath = tempDir / "triangle.obj";
  writeTextFile(objPath, "o MsaaSwitchTriangle\n"
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
      .debugName = "msaa_pipeline_switch_triangle",
  });
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();

  MaterialRequest materialRequest{};
  materialRequest.debugName = "msaa_pipeline_switch_material";
  auto materialResult = renderer.resources().acquireMaterial(materialRequest);
  ASSERT_FALSE(materialResult.hasError()) << materialResult.error();

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  auto renderableResult = scene.graph().addRenderable(
      scene.graph().rootNode(), modelResult.value(), materialResult.value());
  ASSERT_FALSE(renderableResult.hasError()) << renderableResult.error();
  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();
  ASSERT_TRUE(commitResult.value());

  RenderSettings settings{};
  settings.skybox.enabled = false;
  settings.opaque.enableDepthPrepass = false;
  settings.opaque.enableDepthPyramid = false;
  settings.opaque.enableInstanceAnimation = false;
  settings.opaque.enableInstanceCompute = false;
  settings.opaque.enableIndirectDraw = false;

  const auto pipelineNameFor =
      [&gpu](RenderPipelineHandle handle) -> std::string_view {
    if (handle.index == 0u) {
      return {};
    }
    const size_t index = static_cast<size_t>(handle.index - 1u);
    if (index >= gpu.createdRenderPipelineNames.size()) {
      return {};
    }
    return gpu.createdRenderPipelineNames[index];
  };

  const auto buildAndExpect = [&](uint64_t frameIndex, AntiAliasingMode mode,
                                  std::string_view expectedPipelineName,
                                  bool expectMsaa) {
    settings.antiAliasing.mode = mode;

    RenderFrameContext frameContext{};
    frameContext.frameIndex = frameIndex;
    frameContext.scene = &scene;
    frameContext.resources = &renderer.resources();
    frameContext.settings = &settings;
    frameContext.camera = makeSdsmPerspectiveCamera(30.0f);

    RenderGraphBuilder graph(&memory);
    graph.beginFrame(frameContext.frameIndex);
    auto buildResult =
        pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
    ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
    ASSERT_TRUE(buildResult.value());

    RenderGraphRuntime runtime;
    auto compileResult = graph.compile(runtime);
    ASSERT_FALSE(compileResult.hasError()) << compileResult.error();

    const RenderPass *opaquePass = nullptr;
    for (const RenderPass &pass : compileResult.value().orderedPasses) {
      if (pass.debugLabel == "Opaque Pass") {
        opaquePass = &pass;
        break;
      }
    }
    ASSERT_NE(opaquePass, nullptr);
    ASSERT_FALSE(opaquePass->draws.empty());
    EXPECT_EQ(pipelineNameFor(opaquePass->draws.front().pipeline),
              expectedPipelineName);

    if (expectMsaa) {
      EXPECT_TRUE(frameContext.metrics.antiAliasing.msaaEnabled);
      EXPECT_EQ(frameContext.metrics.antiAliasing.msaaSampleCount,
                kMsaa4xSampleCount);
      EXPECT_TRUE(
          sameTexture(opaquePass->colorTexture,
                      frameContext.sharedResources.msaaSceneColorTexture));
      EXPECT_TRUE(sameTexture(opaquePass->colorResolveTexture,
                              frameContext.sharedResources.sceneColorTexture));
      EXPECT_TRUE(
          sameTexture(opaquePass->depthTexture,
                      frameContext.sharedResources.msaaSceneDepthTexture));
      EXPECT_TRUE(sameTexture(opaquePass->depthResolveTexture,
                              frameContext.sharedResources.sceneDepthTexture));
      EXPECT_EQ(opaquePass->color.storeOp, StoreOp::MsaaResolve);
      EXPECT_EQ(opaquePass->depth.storeOp, StoreOp::MsaaResolve);
      EXPECT_TRUE(
          frameContext.metrics.antiAliasing.msaaColorResolveTargetBound);
      EXPECT_TRUE(
          frameContext.metrics.antiAliasing.msaaDepthResolveTargetBound);
    } else {
      EXPECT_FALSE(frameContext.metrics.antiAliasing.msaaEnabled);
      EXPECT_EQ(frameContext.metrics.antiAliasing.msaaSampleCount, 1u);
      EXPECT_TRUE(sameTexture(opaquePass->colorTexture,
                              frameContext.sharedResources.sceneColorTexture));
      EXPECT_FALSE(isValid(opaquePass->colorResolveTexture));
      EXPECT_FALSE(isValid(opaquePass->depthResolveTexture));
      EXPECT_EQ(opaquePass->color.storeOp, StoreOp::Store);
      EXPECT_FALSE(isValid(frameContext.sharedResources.msaaSceneColorTexture));
      EXPECT_FALSE(isValid(frameContext.sharedResources.msaaSceneDepthTexture));
    }
  };

  buildAndExpect(0u, AntiAliasingMode::None, "opaque_mesh", false);
  buildAndExpect(1u, AntiAliasingMode::MSAA4x, "opaque_mesh_msaa4x", true);
  buildAndExpect(2u, AntiAliasingMode::None, "opaque_mesh", false);
}

TEST(RenderGraphRendererTest,
     Msaa4xAlphaMaskedOpaqueDrawsUseAlphaCoveragePipeline) {
  std::array<std::byte, 512 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeShadowSceneGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(
      std::make_unique<FrameCompositionProvider>(gpu, &memory));
  pipeline.addProvider(std::make_unique<MaterialTableGpuProvider>(gpu));
  pipeline.addProvider(std::make_unique<SceneLightingProvider>(gpu));
  pipeline.addFeature(std::make_unique<OpaqueFeature>(
      gpu, makeOpaqueConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
      &memory));
  pipeline.addFeature(std::make_unique<MsaaResolveFeature>(gpu));

  const std::filesystem::path tempDir = makeTempRendererPath("msaa_masked");
  ASSERT_TRUE(std::filesystem::create_directories(tempDir));
  const std::filesystem::path objPath = tempDir / "triangle.obj";
  writeTextFile(objPath, "o MsaaMaskedTriangle\n"
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
      .debugName = "msaa_masked_triangle",
  });
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();

  MaterialRequest materialRequest{};
  materialRequest.debugName = "msaa_masked_material";
  materialRequest.desc.alphaMode = MaterialAlphaMode::Mask;
  materialRequest.desc.alphaCutoff = 0.5f;
  auto materialResult = renderer.resources().acquireMaterial(materialRequest);
  ASSERT_FALSE(materialResult.hasError()) << materialResult.error();

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  auto renderableResult = scene.graph().addRenderable(
      scene.graph().rootNode(), modelResult.value(), materialResult.value());
  ASSERT_FALSE(renderableResult.hasError()) << renderableResult.error();
  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();
  ASSERT_TRUE(commitResult.value());

  RenderSettings settings{};
  settings.skybox.enabled = false;
  settings.antiAliasing.mode = AntiAliasingMode::MSAA4x;
  settings.opaque.enableDepthPrepass = true;
  settings.opaque.enableDepthPyramid = false;
  settings.opaque.enableInstanceAnimation = false;
  settings.opaque.enableInstanceCompute = false;
  settings.opaque.enableIndirectDraw = false;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 0u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  frameContext.camera = makeSdsmPerspectiveCamera(30.0f);

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());

  const AntiAliasingFrameMetrics &metrics = frameContext.metrics.antiAliasing;
  EXPECT_GT(metrics.msaaAlphaMaskedDrawCount, 0u);
  EXPECT_TRUE(metrics.msaaAlphaToCoverageEnabled);
  EXPECT_TRUE(metrics.msaaSampleShadingEnabled);

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const auto &passes = compileResult.value().orderedPasses;
  EXPECT_EQ(compiledPassIndex(passes, "Opaque Depth Pre-Pass"), UINT32_MAX);

  const RenderPass *opaquePass = nullptr;
  for (const RenderPass &pass : passes) {
    if (pass.debugLabel == "Opaque Pass") {
      opaquePass = &pass;
      break;
    }
  }
  ASSERT_NE(opaquePass, nullptr);
  ASSERT_FALSE(opaquePass->draws.empty());
  EXPECT_EQ(opaquePass->depth.loadOp, LoadOp::Clear);

  const auto pipelineNameFor =
      [&gpu](RenderPipelineHandle handle) -> std::string_view {
    if (handle.index == 0u) {
      return {};
    }
    const size_t index = static_cast<size_t>(handle.index - 1u);
    if (index >= gpu.createdRenderPipelineNames.size()) {
      return {};
    }
    return gpu.createdRenderPipelineNames[index];
  };

  bool sawMaskedDraw = false;
  for (const DrawItem &draw : opaquePass->draws) {
    if (!draw.alphaMasked) {
      continue;
    }
    sawMaskedDraw = true;
    EXPECT_EQ(pipelineNameFor(draw.pipeline), "opaque_mesh_alpha_msaa4x");
    EXPECT_TRUE(draw.useDepthState);
    EXPECT_EQ(draw.depthState.compareOp, CompareOp::Less);
    EXPECT_TRUE(draw.depthState.isDepthWriteEnabled);
  }
  EXPECT_TRUE(sawMaskedDraw);
}

TEST(RenderGraphRendererTest, SpatialFallbackModeRunsSmaaPassesAndCopyBack) {
  std::array<std::byte, 64 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  gpu.latestCompletedGpuTimingReport = GpuTimingReport{
      .spatialAASourceFrameIndex = 7u,
      .spatialAATimeMs = 0.42f,
      .availableScopeMask = kGpuTimingScopeSpatialAABit,
  };
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(
      std::make_unique<FrameCompositionProvider>(gpu, &memory));
  pipeline.addFeature(std::make_unique<SpatialAAFeature>(
      gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));

  RenderSettings settings{};
  settings.antiAliasing.mode = AntiAliasingMode::SpatialFallback;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 0u;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());

  const AntiAliasingFrameMetrics &metrics = frameContext.metrics.antiAliasing;
  EXPECT_TRUE(metrics.spatialAAEnabled);
  EXPECT_TRUE(metrics.spatialAAFallbackActive);
  EXPECT_FALSE(metrics.spatialAACleanupActive);
  EXPECT_EQ(metrics.spatialAAWidth, 1280u);
  EXPECT_EQ(metrics.spatialAAHeight, 720u);
  EXPECT_EQ(metrics.spatialAAPassCount, 4u);
  EXPECT_EQ(metrics.spatialAAEdgePassCount, 1u);
  EXPECT_EQ(metrics.spatialAABlendPassCount, 1u);
  EXPECT_EQ(metrics.spatialAANeighborhoodPassCount, 1u);
  EXPECT_EQ(metrics.spatialAACopyBackPassCount, 1u);
  EXPECT_EQ(metrics.spatialAAFallbackFrameCount, 1u);
  EXPECT_EQ(metrics.spatialAATextureCount, 4u);
  EXPECT_EQ(metrics.spatialAALutTextureCount, 2u);
  EXPECT_EQ(metrics.spatialAAGpuTimingAvailable, 1u);
  EXPECT_FLOAT_EQ(metrics.spatialAAGpuTimeMs, 0.42f);
  EXPECT_EQ(metrics.spatialAAGpuTimingSourceFrameIndex, 7u);

  uint32_t rgba8TextureCount = 0u;
  uint32_t lutTextureCount = 0u;
  for (const TextureDesc &desc : gpu.createdTextureDescs) {
    if (desc.format != Format::RGBA8_UNORM) {
      continue;
    }
    ++rgba8TextureCount;
    if (desc.usage == TextureUsage::Sampled) {
      ++lutTextureCount;
    } else {
      EXPECT_EQ(desc.usage, TextureUsage::AttachmentSampled);
      EXPECT_EQ(desc.dimensions.width, 1280u);
      EXPECT_EQ(desc.dimensions.height, 720u);
    }
  }
  EXPECT_EQ(rgba8TextureCount, 6u);
  EXPECT_EQ(lutTextureCount, 2u);

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const auto &passes = compileResult.value().orderedPasses;
  ASSERT_EQ(passes.size(), 4u);
  EXPECT_EQ(passes[0].debugLabel, "SMAA Edge Detection Pass");
  EXPECT_EQ(passes[0].gpuTimingScope, GpuTimingScope::SpatialAA);
  EXPECT_EQ(passes[1].debugLabel, "SMAA Blend Weight Pass");
  EXPECT_EQ(passes[2].debugLabel, "SMAA Neighborhood Pass");
  EXPECT_EQ(passes[3].debugLabel, "SMAA Copy Back Pass");
  EXPECT_TRUE(sameTexture(passes[3].colorTexture,
                          frameContext.sharedResources.sceneColorTexture));
  const SpatialAAPushConstantsProbe fallbackConstants =
      readSpatialAAPushConstants(passes, "SMAA Edge Detection Pass");
  EXPECT_FLOAT_EQ(spatialAAPushFloat(fallbackConstants.edgeThresholdBits),
                  0.12f);
  EXPECT_FLOAT_EQ(spatialAAPushFloat(fallbackConstants.resolveStrengthBits),
                  0.30f);
  EXPECT_FLOAT_EQ(spatialAAPushFloat(fallbackConstants.localContrastFactorBits),
                  2.0f);
  EXPECT_FLOAT_EQ(spatialAAPushFloat(fallbackConstants.cornerRoundingBits),
                  0.25f);
  EXPECT_EQ(fallbackConstants.maxSearchSteps, 16u);
}

TEST(RenderGraphRendererTest,
     SpatialFallbackRunsAfterTaaResolveWhenHistoryIsInvalid) {
  std::array<std::byte, 96 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(
      std::make_unique<FrameCompositionProvider>(gpu, &memory));
  pipeline.addFeature(std::make_unique<TemporalAAFeature>(
      gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));
  pipeline.addFeature(std::make_unique<SpatialAAFeature>(
      gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));

  RenderSettings settings{};
  settings.antiAliasing.mode = AntiAliasingMode::TAA;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 0u;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  frameContext.camera.temporalDataValid = true;
  frameContext.camera.historyValid = false;
  frameContext.camera.framesSinceHistoryReset = 0u;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());

  EXPECT_TRUE(frameContext.metrics.antiAliasing.spatialAAEnabled);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.spatialAAFallbackActive);
  EXPECT_EQ(frameContext.metrics.antiAliasing.spatialAAPassCount, 4u);

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const auto &passes = compileResult.value().orderedPasses;
  const uint32_t taaCopyBackIndex =
      compiledPassIndex(passes, "TAA Copy Back Pass");
  const uint32_t spatialEdgeIndex =
      compiledPassIndex(passes, "SMAA Edge Detection Pass");
  ASSERT_NE(taaCopyBackIndex, UINT32_MAX);
  ASSERT_NE(spatialEdgeIndex, UINT32_MAX);
  EXPECT_LT(taaCopyBackIndex, spatialEdgeIndex);
  const SpatialAAPushConstantsProbe fallbackConstants =
      readSpatialAAPushConstants(passes, "SMAA Edge Detection Pass");
  EXPECT_FLOAT_EQ(spatialAAPushFloat(fallbackConstants.edgeThresholdBits),
                  0.12f);
  EXPECT_FLOAT_EQ(spatialAAPushFloat(fallbackConstants.resolveStrengthBits),
                  0.30f);
  EXPECT_FLOAT_EQ(spatialAAPushFloat(fallbackConstants.localContrastFactorBits),
                  2.0f);
}

TEST(RenderGraphRendererTest,
     SpatialCleanupRunsAfterTaaWhenDebugToggleIsEnabled) {
  std::array<std::byte, 96 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(
      std::make_unique<FrameCompositionProvider>(gpu, &memory));
  pipeline.addFeature(std::make_unique<TemporalAAFeature>(
      gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));
  pipeline.addFeature(std::make_unique<SpatialAAFeature>(
      gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));

  RenderSettings settings{};
  settings.antiAliasing.mode = AntiAliasingMode::TAA;
  settings.antiAliasing.debug.spatialPostTaaCleanup = true;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 0u;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  frameContext.camera.temporalDataValid = true;
  frameContext.camera.historyValid = true;
  frameContext.camera.framesSinceHistoryReset = kTemporalJitterSequenceLength;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());

  EXPECT_TRUE(frameContext.metrics.antiAliasing.spatialAAEnabled);
  EXPECT_FALSE(frameContext.metrics.antiAliasing.spatialAAFallbackActive);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.spatialAACleanupActive);
  EXPECT_EQ(frameContext.metrics.antiAliasing.spatialAAPassCount, 4u);
  EXPECT_EQ(frameContext.metrics.antiAliasing.spatialAACleanupFrameCount, 1u);

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const auto &passes = compileResult.value().orderedPasses;
  const uint32_t taaCopyBackIndex =
      compiledPassIndex(passes, "TAA Copy Back Pass");
  const uint32_t spatialEdgeIndex =
      compiledPassIndex(passes, "SMAA Edge Detection Pass");
  ASSERT_NE(taaCopyBackIndex, UINT32_MAX);
  ASSERT_NE(spatialEdgeIndex, UINT32_MAX);
  EXPECT_LT(taaCopyBackIndex, spatialEdgeIndex);
  const SpatialAAPushConstantsProbe cleanupConstants =
      readSpatialAAPushConstants(passes, "SMAA Edge Detection Pass");
  EXPECT_FLOAT_EQ(spatialAAPushFloat(cleanupConstants.edgeThresholdBits),
                  0.14f);
  EXPECT_FLOAT_EQ(spatialAAPushFloat(cleanupConstants.resolveStrengthBits),
                  0.10f);
  EXPECT_FLOAT_EQ(spatialAAPushFloat(cleanupConstants.localContrastFactorBits),
                  1.45f);
  EXPECT_FLOAT_EQ(spatialAAPushFloat(cleanupConstants.cornerRoundingBits),
                  0.40f);
  EXPECT_EQ(cleanupConstants.maxSearchSteps, 16u);
}

TEST(RenderGraphRendererTest, SpatialCleanupRunsAfterMsaaResolveWhenEnabled) {
  std::array<std::byte, 128 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(
      std::make_unique<FrameCompositionProvider>(gpu, &memory));
  pipeline.addProvider(std::make_unique<MaterialTableGpuProvider>(gpu));
  pipeline.addProvider(std::make_unique<SceneLightingProvider>(gpu));
  pipeline.addFeature(std::make_unique<OpaqueFeature>(
      gpu, makeOpaqueConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
      &memory));
  pipeline.addFeature(std::make_unique<MsaaResolveFeature>(gpu));
  pipeline.addFeature(std::make_unique<SpatialAAFeature>(
      gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());

  RenderSettings settings{};
  settings.skybox.enabled = false;
  settings.antiAliasing.mode = AntiAliasingMode::MSAA4x;
  settings.antiAliasing.debug.spatialPostMsaaCleanup = true;
  settings.opaque.enableDepthPyramid = false;
  settings.opaque.enableInstanceAnimation = false;
  settings.opaque.enableInstanceCompute = false;
  settings.opaque.enableIndirectDraw = false;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 0u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());

  const AntiAliasingFrameMetrics &metrics = frameContext.metrics.antiAliasing;
  EXPECT_TRUE(metrics.spatialAAEnabled);
  EXPECT_FALSE(metrics.spatialAAFallbackActive);
  EXPECT_TRUE(metrics.spatialAACleanupActive);
  EXPECT_TRUE(metrics.msaaSpatialCleanupEnabled);
  EXPECT_TRUE(metrics.msaaSpatialCleanupActive);
  EXPECT_EQ(metrics.spatialAAPassCount, 4u);

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const auto &passes = compileResult.value().orderedPasses;
  const uint32_t opaqueIndex = compiledPassIndex(passes, "Opaque Pass");
  const uint32_t standaloneResolveIndex =
      compiledPassIndex(passes, "MSAA Resolve Pass");
  const uint32_t spatialEdgeIndex =
      compiledPassIndex(passes, "SMAA Edge Detection Pass");
  ASSERT_NE(opaqueIndex, UINT32_MAX);
  EXPECT_EQ(standaloneResolveIndex, UINT32_MAX);
  ASSERT_NE(spatialEdgeIndex, UINT32_MAX);
  EXPECT_LT(opaqueIndex, spatialEdgeIndex);
  const SpatialAAPushConstantsProbe msaaCleanupConstants =
      readSpatialAAPushConstants(passes, "SMAA Edge Detection Pass");
  EXPECT_FLOAT_EQ(spatialAAPushFloat(msaaCleanupConstants.edgeThresholdBits),
                  0.12f);
  EXPECT_FLOAT_EQ(spatialAAPushFloat(msaaCleanupConstants.resolveStrengthBits),
                  0.16f);
  EXPECT_FLOAT_EQ(
      spatialAAPushFloat(msaaCleanupConstants.localContrastFactorBits), 2.0f);
  EXPECT_FLOAT_EQ(spatialAAPushFloat(msaaCleanupConstants.cornerRoundingBits),
                  0.25f);
}

TEST(RenderGraphRendererTest, SpatialCleanupDoesNotRunAfterMsaaWhenDisabled) {
  std::array<std::byte, 96 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(
      std::make_unique<FrameCompositionProvider>(gpu, &memory));
  pipeline.addFeature(std::make_unique<SpatialAAFeature>(
      gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));

  RenderSettings settings{};
  settings.antiAliasing.mode = AntiAliasingMode::MSAA4x;
  settings.antiAliasing.debug.spatialPostMsaaCleanup = false;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 0u;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());

  EXPECT_FALSE(frameContext.metrics.antiAliasing.spatialAAEnabled);
  EXPECT_FALSE(frameContext.metrics.antiAliasing.msaaSpatialCleanupEnabled);
  EXPECT_FALSE(frameContext.metrics.antiAliasing.msaaSpatialCleanupActive);

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const auto &passes = compileResult.value().orderedPasses;
  EXPECT_EQ(compiledPassIndex(passes, "SMAA Edge Detection Pass"), UINT32_MAX);
}

TEST(RenderGraphRendererTest, SpatialAABypassesTaaDebugViews) {
  std::array<std::byte, 96 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(
      std::make_unique<FrameCompositionProvider>(gpu, &memory));
  pipeline.addFeature(std::make_unique<TemporalAAFeature>(
      gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));
  pipeline.addFeature(std::make_unique<SpatialAAFeature>(
      gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));

  RenderSettings settings{};
  settings.antiAliasing.mode = AntiAliasingMode::TAA;
  settings.antiAliasing.debug.view = AntiAliasingDebugView::TAAHdrWeight;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 0u;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  frameContext.camera.temporalDataValid = true;
  frameContext.camera.historyValid = true;
  frameContext.camera.framesSinceHistoryReset = kTemporalJitterSequenceLength;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());
  EXPECT_FALSE(frameContext.metrics.antiAliasing.spatialAAEnabled);

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const auto &passes = compileResult.value().orderedPasses;
  EXPECT_EQ(compiledPassIndex(passes, "SMAA Edge Detection Pass"), UINT32_MAX);
}

TEST(RenderGraphRendererTest, SpatialAADebugViewsRenderExpectedPasses) {
  struct DebugCase {
    AntiAliasingDebugView view = AntiAliasingDebugView::None;
    std::string_view label{};
  };
  const std::array<DebugCase, 4> cases{{
      {.view = AntiAliasingDebugView::SpatialAAEdges,
       .label = "SMAA Edge Debug Pass"},
      {.view = AntiAliasingDebugView::SpatialAABlendWeights,
       .label = "SMAA Blend Weight Debug Pass"},
      {.view = AntiAliasingDebugView::SpatialAACleanupMask,
       .label = "SMAA Cleanup Mask Debug Pass"},
      {.view = AntiAliasingDebugView::SpatialAASplitCompare,
       .label = "SMAA Split Compare Debug Pass"},
  }};

  for (const DebugCase &debugCase : cases) {
    std::array<std::byte, 96 * 1024> scratchBytes{};
    std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                               scratchBytes.size());
    FakeFullscreenGpuDevice gpu;
    Renderer renderer(gpu, memory);
    RenderPipeline pipeline(&memory);
    pipeline.addProvider(
        std::make_unique<FrameCompositionProvider>(gpu, &memory));
    pipeline.addFeature(std::make_unique<SpatialAAFeature>(
        gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));

    RenderSettings settings{};
    settings.antiAliasing.mode = AntiAliasingMode::TAA;
    settings.antiAliasing.debug.view = debugCase.view;

    RenderFrameContext frameContext{};
    frameContext.frameIndex = 0u;
    frameContext.resources = &renderer.resources();
    frameContext.settings = &settings;
    frameContext.camera.temporalDataValid = true;
    frameContext.camera.historyValid = true;
    frameContext.camera.framesSinceHistoryReset = kTemporalJitterSequenceLength;

    RenderGraphBuilder graph(&memory);
    graph.beginFrame(frameContext.frameIndex);
    auto buildResult =
        pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
    ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
    ASSERT_TRUE(buildResult.value());
    EXPECT_TRUE(frameContext.metrics.antiAliasing.spatialAAEnabled);
    EXPECT_EQ(frameContext.metrics.antiAliasing.spatialAADebugPassCount, 1u);

    if (debugCase.view == AntiAliasingDebugView::SpatialAAEdges) {
      EXPECT_TRUE(
          frameContext.metrics.antiAliasing.spatialAAEdgesDebugViewRendered);
    } else if (debugCase.view == AntiAliasingDebugView::SpatialAABlendWeights) {
      EXPECT_TRUE(frameContext.metrics.antiAliasing
                      .spatialAABlendWeightsDebugViewRendered);
    } else if (debugCase.view == AntiAliasingDebugView::SpatialAACleanupMask) {
      EXPECT_TRUE(frameContext.metrics.antiAliasing
                      .spatialAACleanupMaskDebugViewRendered);
    } else if (debugCase.view == AntiAliasingDebugView::SpatialAASplitCompare) {
      EXPECT_TRUE(frameContext.metrics.antiAliasing
                      .spatialAASplitCompareDebugViewRendered);
    }

    RenderGraphRuntime runtime;
    auto compileResult = graph.compile(runtime);
    ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
    const auto &passes = compileResult.value().orderedPasses;
    EXPECT_NE(compiledPassIndex(passes, debugCase.label), UINT32_MAX);
    EXPECT_NE(compiledPassIndex(passes, "SMAA Copy Back Pass"), UINT32_MAX);
  }
}

TEST(RenderGraphRendererTest, OpaqueVelocityPassPublishesMotionVectorsForTaa) {
  std::array<std::byte, 512 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeShadowSceneGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(
      std::make_unique<FrameCompositionProvider>(gpu, &memory));
  pipeline.addProvider(std::make_unique<MaterialTableGpuProvider>(gpu));
  pipeline.addProvider(std::make_unique<SceneLightingProvider>(gpu));
  pipeline.addFeature(std::make_unique<OpaqueFeature>(
      gpu, makeOpaqueConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
      &memory));
  pipeline.addFeature(std::make_unique<TemporalAAFeature>(
      gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));

  const std::filesystem::path tempDir =
      makeTempRendererPath("taa_opaque_velocity");
  ASSERT_TRUE(std::filesystem::create_directories(tempDir));
  const std::filesystem::path objPath = tempDir / "triangle.obj";
  writeTextFile(objPath, "o TaaVelocityTriangle\n"
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
      .debugName = "taa_velocity_triangle",
  });
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();
  auto materialResult = renderer.resources().acquireMaterial(MaterialRequest{
      .debugName = "taa_velocity_material",
  });
  ASSERT_FALSE(materialResult.hasError()) << materialResult.error();

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  auto renderableResult = scene.graph().addRenderable(
      scene.graph().rootNode(), modelResult.value(), materialResult.value());
  ASSERT_FALSE(renderableResult.hasError()) << renderableResult.error();
  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();
  ASSERT_TRUE(commitResult.value());

  RenderSettings settings{};
  settings.skybox.enabled = false;
  settings.antiAliasing.mode = AntiAliasingMode::TAA;
  settings.antiAliasing.debug.view = AntiAliasingDebugView::VelocityMagnitude;
  settings.opaque.enableInstanceAnimation = false;
  settings.opaque.enableInstanceCompute = false;
  settings.opaque.enableIndirectDraw = false;

  CameraFrameState camera = makeSdsmPerspectiveCamera(30.0f);
  camera.currentUnjitteredViewProj = camera.proj * camera.view;
  camera.previousUnjitteredViewProj = camera.currentUnjitteredViewProj;
  camera.temporalDataValid = true;
  camera.historyValid = true;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 0u;
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

  EXPECT_TRUE(frameContext.metrics.antiAliasing.opaqueVelocityGenerated);
  EXPECT_EQ(frameContext.metrics.antiAliasing.velocityPassCount, 1u);
  EXPECT_GT(frameContext.metrics.antiAliasing.velocityDrawCount, 0u);
  EXPECT_EQ(frameContext.metrics.antiAliasing.velocityDebugPassCount, 1u);
  EXPECT_GT(
      frameContext.metrics.antiAliasing.velocityPassBandwidthEstimateBytes, 0u);
  EXPECT_GT(
      frameContext.metrics.antiAliasing.velocityDebugBandwidthEstimateBytes,
      0u);
  EXPECT_EQ(frameContext.metrics.antiAliasing.motionVectorClearPassCount, 0u);
  EXPECT_EQ(frameContext.metrics.antiAliasing.motionVectorClearBytes, 0ull);
  EXPECT_EQ(frameContext.metrics.antiAliasing.taaResolvePassCount, 1u);
  EXPECT_EQ(frameContext.metrics.antiAliasing.taaCopyBackPassCount, 1u);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.motionVectorGraphPublished);
  EXPECT_FALSE(frameContext.metrics.antiAliasing.previousTransformCacheValid);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.velocityDebugViewRendered);
  EXPECT_TRUE(isValid(frameContext.sharedResources.motionVectorGraphTexture));
  EXPECT_TRUE(isValid(frameContext.sharedResources.sceneColorGraphTexture));

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();

  bool sawOpaquePass = false;
  bool sawVelocityPass = false;
  bool sawVelocityDebugPass = false;
  bool sawTaaResolvePass = false;
  bool sawClearFallback = false;
  for (const RenderPass &pass : compileResult.value().orderedPasses) {
    if (pass.debugLabel == "Opaque Pass") {
      sawOpaquePass = true;
    } else if (pass.debugLabel == "Opaque Velocity Pass") {
      sawVelocityPass = true;
      EXPECT_TRUE(sameTexture(
          pass.colorTexture, frameContext.sharedResources.motionVectorTexture));
      EXPECT_EQ(pass.color.loadOp, LoadOp::Clear);
      EXPECT_EQ(pass.depth.loadOp, LoadOp::Load);
      EXPECT_FALSE(pass.draws.empty());
    } else if (pass.debugLabel == "TAA Velocity Debug Pass") {
      sawVelocityDebugPass = true;
      EXPECT_TRUE(sameTexture(pass.colorTexture,
                              frameContext.sharedResources.sceneColorTexture));
      EXPECT_EQ(pass.color.loadOp, LoadOp::Clear);
      EXPECT_FALSE(pass.draws.empty());
    } else if (pass.debugLabel == "TAA Resolve Pass") {
      sawTaaResolvePass = true;
    } else if (pass.debugLabel == "Temporal AA Motion Vector Clear") {
      sawClearFallback = true;
    }
  }
  EXPECT_TRUE(sawOpaquePass);
  EXPECT_TRUE(sawVelocityPass);
  EXPECT_TRUE(sawTaaResolvePass);
  EXPECT_TRUE(sawVelocityDebugPass);
  EXPECT_FALSE(sawClearFallback);

  RenderFrameContext secondFrameContext{};
  secondFrameContext.frameIndex = 1u;
  secondFrameContext.scene = &scene;
  secondFrameContext.resources = &renderer.resources();
  secondFrameContext.settings = &settings;
  secondFrameContext.camera = camera;

  RenderGraphBuilder secondGraph(&memory);
  secondGraph.beginFrame(secondFrameContext.frameIndex);
  auto secondBuildResult = pipeline.buildRenderGraph(
      secondFrameContext, renderer.resources(), secondGraph);
  ASSERT_FALSE(secondBuildResult.hasError()) << secondBuildResult.error();
  ASSERT_TRUE(secondBuildResult.value());
  EXPECT_TRUE(
      secondFrameContext.metrics.antiAliasing.previousTransformCacheValid);
  EXPECT_GT(secondFrameContext.metrics.antiAliasing
                .velocityPreviousTransformValidCount,
            0u);
  EXPECT_EQ(secondFrameContext.metrics.antiAliasing.motionVectorClearBytes,
            0ull);

  RenderFrameContext sameFrameRebuildContext{};
  sameFrameRebuildContext.frameIndex = 1u;
  sameFrameRebuildContext.scene = &scene;
  sameFrameRebuildContext.resources = &renderer.resources();
  sameFrameRebuildContext.settings = &settings;
  sameFrameRebuildContext.camera = camera;

  RenderGraphBuilder sameFrameGraph(&memory);
  sameFrameGraph.beginFrame(sameFrameRebuildContext.frameIndex);
  auto sameFrameBuildResult = pipeline.buildRenderGraph(
      sameFrameRebuildContext, renderer.resources(), sameFrameGraph);
  ASSERT_FALSE(sameFrameBuildResult.hasError()) << sameFrameBuildResult.error();
  ASSERT_TRUE(sameFrameBuildResult.value());
  EXPECT_FALSE(
      sameFrameRebuildContext.metrics.antiAliasing.previousTransformCacheValid);

  RenderFrameContext invalidTemporalContext{};
  invalidTemporalContext.frameIndex = 2u;
  invalidTemporalContext.scene = &scene;
  invalidTemporalContext.resources = &renderer.resources();
  invalidTemporalContext.settings = &settings;
  invalidTemporalContext.camera = camera;
  invalidTemporalContext.camera.temporalDataValid = false;

  RenderGraphBuilder invalidTemporalGraph(&memory);
  invalidTemporalGraph.beginFrame(invalidTemporalContext.frameIndex);
  auto invalidTemporalBuildResult = pipeline.buildRenderGraph(
      invalidTemporalContext, renderer.resources(), invalidTemporalGraph);
  ASSERT_FALSE(invalidTemporalBuildResult.hasError())
      << invalidTemporalBuildResult.error();
  ASSERT_TRUE(invalidTemporalBuildResult.value());
  EXPECT_FALSE(
      invalidTemporalContext.metrics.antiAliasing.previousTransformCacheValid);
}

TEST(RenderGraphRendererTest, OpaqueReactiveMaskPassPublishesMaskForTaa) {
  std::array<std::byte, 512 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeShadowSceneGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(
      std::make_unique<FrameCompositionProvider>(gpu, &memory));
  pipeline.addProvider(std::make_unique<MaterialTableGpuProvider>(gpu));
  pipeline.addProvider(std::make_unique<SceneLightingProvider>(gpu));
  pipeline.addFeature(std::make_unique<OpaqueFeature>(
      gpu, makeOpaqueConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
      &memory));
  pipeline.addFeature(std::make_unique<TemporalAAFeature>(
      gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));

  const std::filesystem::path tempDir =
      makeTempRendererPath("taa_opaque_reactive");
  ASSERT_TRUE(std::filesystem::create_directories(tempDir));
  const std::filesystem::path objPath = tempDir / "triangle.obj";
  writeTextFile(objPath, "o TaaReactiveTriangle\n"
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
      .debugName = "taa_reactive_triangle",
  });
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();

  MaterialRequest materialRequest{};
  materialRequest.debugName = "taa_reactive_material";
  materialRequest.desc.alphaMode = MaterialAlphaMode::Mask;
  materialRequest.desc.alphaCutoff = 0.5f;
  auto materialResult = renderer.resources().acquireMaterial(materialRequest);
  ASSERT_FALSE(materialResult.hasError()) << materialResult.error();

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  auto renderableResult = scene.graph().addRenderable(
      scene.graph().rootNode(), modelResult.value(), materialResult.value());
  ASSERT_FALSE(renderableResult.hasError()) << renderableResult.error();
  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();
  ASSERT_TRUE(commitResult.value());

  RenderSettings settings{};
  settings.skybox.enabled = false;
  settings.antiAliasing.mode = AntiAliasingMode::TAA;
  settings.opaque.enableInstanceAnimation = false;
  settings.opaque.enableInstanceCompute = false;
  settings.opaque.enableIndirectDraw = false;

  CameraFrameState camera = makeSdsmPerspectiveCamera(30.0f);
  camera.currentUnjitteredViewProj = camera.proj * camera.view;
  camera.previousUnjitteredViewProj = camera.currentUnjitteredViewProj;
  camera.temporalDataValid = true;
  camera.historyValid = true;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 0u;
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

  EXPECT_TRUE(frameContext.metrics.antiAliasing.reactiveMaskAllocated);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.reactiveMaskGraphPublished);
  EXPECT_EQ(frameContext.metrics.antiAliasing.reactiveMaskPassCount, 1u);
  EXPECT_GT(frameContext.metrics.antiAliasing.reactiveMaskDrawCount, 0u);
  EXPECT_GT(frameContext.metrics.antiAliasing.reactiveAlphaMaskedDrawCount, 0u);
  EXPECT_GT(
      frameContext.metrics.antiAliasing.reactiveMaskPassBandwidthEstimateBytes,
      0ull);
  EXPECT_GT(frameContext.metrics.antiAliasing.taaReactiveCoverageEstimate,
            0.0f);

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();

  bool sawReactivePass = false;
  bool sawReactiveClearFallback = false;
  for (const RenderPass &pass : compileResult.value().orderedPasses) {
    if (pass.debugLabel == "Opaque Reactive Mask Pass") {
      sawReactivePass = true;
      EXPECT_TRUE(sameTexture(
          pass.colorTexture, frameContext.sharedResources.reactiveMaskTexture));
      EXPECT_EQ(pass.color.loadOp, LoadOp::Clear);
      EXPECT_EQ(pass.depth.loadOp, LoadOp::Load);
      EXPECT_FALSE(pass.draws.empty());
    } else if (pass.debugLabel == "Temporal AA Reactive Mask Clear") {
      sawReactiveClearFallback = true;
    }
  }
  EXPECT_TRUE(sawReactivePass);
  EXPECT_FALSE(sawReactiveClearFallback);
}

TEST(RenderGraphRendererTest, TaaVelocityBuffersDoNotOverflowShadowedMainPass) {
  std::vector<std::byte> scratchBytes(1024 * 1024);
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeShadowSceneGpuDevice gpu;
  gpu.forcedIndexStrideBytes = sizeof(uint16_t);
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(
      std::make_unique<FrameCompositionProvider>(gpu, &memory));
  pipeline.addProvider(std::make_unique<MaterialTableGpuProvider>(gpu));
  pipeline.addProvider(std::make_unique<SceneLightingProvider>(gpu));
  pipeline.addFeature(std::make_unique<ShadowFeature>(
      gpu, makeOpaqueConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
      &memory));
  pipeline.addFeature(std::make_unique<OpaqueFeature>(
      gpu, makeOpaqueConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
      &memory));
  pipeline.addFeature(std::make_unique<TemporalAAFeature>(
      gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));

  const std::filesystem::path tempDir =
      makeTempRendererPath("taa_shadow_velocity_dependencies");
  ASSERT_TRUE(std::filesystem::create_directories(tempDir));
  const std::filesystem::path objPath = tempDir / "triangle.obj";
  writeTextFile(objPath, "o TaaShadowVelocityTriangle\n"
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
      .debugName = "taa_shadow_velocity_triangle",
  });
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();
  auto materialResult = renderer.resources().acquireMaterial(MaterialRequest{
      .debugName = "taa_shadow_velocity_material",
  });
  ASSERT_FALSE(materialResult.hasError()) << materialResult.error();

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  auto renderableResult = scene.graph().addRenderable(
      scene.graph().rootNode(), modelResult.value(), materialResult.value());
  ASSERT_FALSE(renderableResult.hasError()) << renderableResult.error();
  addDirectionalLightToScene(scene);

  RenderSettings settings{};
  settings.skybox.enabled = false;
  settings.shadow.enabled = true;
  settings.shadow.cascadeCount = 4u;
  settings.shadow.shadowMapSize = 512u;
  settings.shadow.sdsmMode = ShadowSdsmMode::Disabled;
  settings.antiAliasing.mode = AntiAliasingMode::TAA;
  settings.antiAliasing.debug.view = AntiAliasingDebugView::None;
  settings.opaque.enableDepthPyramid = false;
  settings.opaque.enableInstanceAnimation = false;
  settings.opaque.enableInstanceCompute = false;
  settings.opaque.enableIndirectDraw = false;

  CameraFrameState camera = makeSdsmPerspectiveCamera(30.0f);
  camera.currentUnjitteredViewProj = camera.proj * camera.view;
  camera.previousUnjitteredViewProj = camera.currentUnjitteredViewProj;
  camera.temporalDataValid = true;
  camera.historyValid = true;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 4u;
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
  ASSERT_TRUE(frameContext.sharedResources.shadowFrameGpuData.has_value());
  EXPECT_TRUE(frameContext.metrics.antiAliasing.opaqueVelocityGenerated);

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();

  const RenderPass *opaquePass = nullptr;
  const RenderPass *velocityPass = nullptr;
  uint32_t shadowPassCount = 0u;
  for (const RenderPass &pass : compileResult.value().orderedPasses) {
    if (std::string_view(pass.debugLabel)
            .starts_with("ShadowDepthPass.Cascade")) {
      ++shadowPassCount;
    } else if (pass.debugLabel == "Opaque Pass") {
      opaquePass = &pass;
    } else if (pass.debugLabel == "Opaque Velocity Pass") {
      velocityPass = &pass;
    }
  }

  EXPECT_EQ(shadowPassCount, 4u);
  ASSERT_NE(opaquePass, nullptr);
  ASSERT_NE(velocityPass, nullptr);
  EXPECT_LE(opaquePass->dependencyBuffers.size(), kMaxDependencyResources);
  EXPECT_LE(velocityPass->dependencyBuffers.size(), kMaxDependencyResources);
  EXPECT_LT(opaquePass->dependencyBuffers.size(),
            velocityPass->dependencyBuffers.size());
}

TEST(RenderGraphRendererTest,
     TemporalAAFeatureImportsPreviousMotionVectorWhenResolveConsumesIt) {
  std::array<std::byte, 32 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(
      std::make_unique<FrameCompositionProvider>(gpu, &memory));
  pipeline.addFeature(std::make_unique<TemporalAAFeature>(
      gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));

  RenderSettings settings{};
  settings.antiAliasing.mode = AntiAliasingMode::TAA;
  settings.antiAliasing.debug.taaClampMode = TemporalAAClampMode::Variance;
  settings.antiAliasing.debug.taaHdrWeightingMode =
      TemporalAAHdrWeightingMode::LogLuminance;
  settings.antiAliasing.debug.taaVarianceGamma = 1.75f;
  settings.antiAliasing.debug.taaHdrWeightStrength = 0.65f;
  settings.antiAliasing.debug.taaMotionCurrentWeight = 0.33f;
  settings.antiAliasing.debug.taaClampCurrentWeight = 0.52f;
  settings.antiAliasing.debug.taaHistoryFilterMode =
      TemporalAAHistoryFilterMode::Bilinear;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 1u;
  frameContext.camera.historyValid = true;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());

  EXPECT_TRUE(
      isValid(frameContext.sharedResources.previousMotionVectorTexture));
  EXPECT_TRUE(
      isValid(frameContext.sharedResources.previousMotionVectorGraphTexture));
  EXPECT_TRUE(frameContext.metrics.antiAliasing.previousMotionVectorValid);
  EXPECT_TRUE(
      frameContext.metrics.antiAliasing.previousMotionVectorGraphPublished);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.taaDepthRejectionEnabled);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.taaVelocityRejectionEnabled);
  EXPECT_TRUE(
      frameContext.metrics.antiAliasing.taaPreviousVelocityDisocclusionEnabled);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.taaNeighborhoodClampEnabled);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.taaAdaptiveBlendEnabled);
  EXPECT_TRUE(
      frameContext.metrics.antiAliasing.taaClampBlendAttenuationEnabled);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.taaNeighborhoodFallbackEnabled);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.taaHdrWeightingEnabled);
  EXPECT_EQ(frameContext.metrics.antiAliasing.taaClampMode,
            TemporalAAClampMode::Variance);
  EXPECT_EQ(frameContext.metrics.antiAliasing.taaHdrWeightingMode,
            TemporalAAHdrWeightingMode::LogLuminance);
  EXPECT_EQ(frameContext.metrics.antiAliasing.taaHistoryFilterMode,
            TemporalAAHistoryFilterMode::Bilinear);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.taaBilinearHistorySampling);
  EXPECT_FLOAT_EQ(frameContext.metrics.antiAliasing.taaMotionCurrentWeight,
                  0.33f);
  EXPECT_FLOAT_EQ(frameContext.metrics.antiAliasing.taaClampCurrentWeight,
                  0.52f);
  EXPECT_FLOAT_EQ(frameContext.metrics.antiAliasing.taaVarianceGamma, 1.75f);
  EXPECT_FLOAT_EQ(frameContext.metrics.antiAliasing.taaHdrWeightStrength,
                  0.65f);

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  ASSERT_EQ(compileResult.value().orderedPasses.size(), 4u);
  EXPECT_EQ(compileResult.value().orderedPasses[0].debugLabel,
            "Temporal AA Motion Vector Clear");
  EXPECT_EQ(compileResult.value().orderedPasses[1].debugLabel,
            "Temporal AA Reactive Mask Clear");
  EXPECT_EQ(compileResult.value().orderedPasses[2].debugLabel,
            "TAA Resolve Pass");
  EXPECT_EQ(compileResult.value().orderedPasses[3].debugLabel,
            "TAA Copy Back Pass");
}

TEST(RenderGraphRendererTest,
     TemporalAAFeaturePublishesResolveAndDebugGpuTimingMetrics) {
  std::array<std::byte, 32 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  gpu.latestCompletedGpuTimingReport = GpuTimingReport{
      .temporalAAResolveSourceFrameIndex = 9u,
      .temporalAADebugSourceFrameIndex = 9u,
      .temporalAAResolveTimeMs = 0.42f,
      .temporalAADebugTimeMs = 0.17f,
      .availableScopeMask = kGpuTimingScopeTemporalAAResolveBit |
                            kGpuTimingScopeTemporalAADebugBit,
  };
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(
      std::make_unique<FrameCompositionProvider>(gpu, &memory));
  pipeline.addFeature(std::make_unique<TemporalAAFeature>(
      gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));

  RenderSettings settings{};
  settings.antiAliasing.mode = AntiAliasingMode::TAA;
  settings.antiAliasing.debug.view = AntiAliasingDebugView::TAAHdrWeight;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 10u;
  frameContext.camera.historyValid = true;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());

  EXPECT_EQ(frameContext.metrics.antiAliasing.taaResolveGpuTimingAvailable, 1u);
  EXPECT_FLOAT_EQ(frameContext.metrics.antiAliasing.taaResolveGpuTimeMs, 0.42f);
  EXPECT_EQ(
      frameContext.metrics.antiAliasing.taaResolveGpuTimingSourceFrameIndex,
      9u);
  EXPECT_EQ(frameContext.metrics.antiAliasing.taaDebugGpuTimingAvailable, 1u);
  EXPECT_FLOAT_EQ(frameContext.metrics.antiAliasing.taaDebugGpuTimeMs, 0.17f);
  EXPECT_EQ(frameContext.metrics.antiAliasing.taaDebugGpuTimingSourceFrameIndex,
            9u);

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const auto &orderedPasses = compileResult.value().orderedPasses;
  ASSERT_EQ(orderedPasses.size(), 5u);
  EXPECT_EQ(orderedPasses[2].debugLabel, "TAA Resolve Pass");
  EXPECT_EQ(orderedPasses[2].gpuTimingScope, GpuTimingScope::TemporalAAResolve);
  EXPECT_EQ(orderedPasses[3].debugLabel, "TAA HDR Weight Debug Pass");
  EXPECT_EQ(orderedPasses[3].gpuTimingScope, GpuTimingScope::TemporalAADebug);
  EXPECT_EQ(orderedPasses[4].debugLabel, "TAA Debug Copy Back Pass");
  EXPECT_EQ(orderedPasses[4].gpuTimingScope, GpuTimingScope::TemporalAADebug);
}

TEST(RenderGraphRendererTest,
     TemporalAAFeatureCopyBackRunsBeforeFrameCompositionReadsSceneColor) {
  std::array<std::byte, 32 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(
      std::make_unique<FrameCompositionProvider>(gpu, &memory));
  pipeline.addFeature(std::make_unique<TemporalAAFeature>(
      gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));
  pipeline.addFeature(std::make_unique<FrameCompositionFeature>(
      gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));

  RenderSettings settings{};
  settings.antiAliasing.mode = AntiAliasingMode::TAA;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 1u;
  frameContext.camera.historyValid = true;
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
  const auto &passes = compileResult.value().orderedPasses;

  uint32_t taaResolveIndex = UINT32_MAX;
  uint32_t taaCopyBackIndex = UINT32_MAX;
  uint32_t downsampleHalfIndex = UINT32_MAX;
  uint32_t sceneResolveIndex = UINT32_MAX;
  for (uint32_t i = 0u; i < passes.size(); ++i) {
    if (passes[i].debugLabel == "TAA Resolve Pass") {
      taaResolveIndex = i;
    } else if (passes[i].debugLabel == "TAA Copy Back Pass") {
      taaCopyBackIndex = i;
    } else if (passes[i].debugLabel == "Scene Color Downsample Half") {
      downsampleHalfIndex = i;
    } else if (passes[i].debugLabel == "Scene Resolve Pass") {
      sceneResolveIndex = i;
    }
  }

  ASSERT_NE(taaResolveIndex, UINT32_MAX);
  ASSERT_NE(taaCopyBackIndex, UINT32_MAX);
  ASSERT_NE(downsampleHalfIndex, UINT32_MAX);
  ASSERT_NE(sceneResolveIndex, UINT32_MAX);
  EXPECT_LT(taaResolveIndex, taaCopyBackIndex);
  EXPECT_LT(taaCopyBackIndex, downsampleHalfIndex);
  EXPECT_LT(taaCopyBackIndex, sceneResolveIndex);
  EXPECT_EQ(passes[taaResolveIndex].gpuTimingScope,
            GpuTimingScope::TemporalAAResolve);
  EXPECT_EQ(passes[taaCopyBackIndex].gpuTimingScope,
            GpuTimingScope::TemporalAAResolve);
  EXPECT_TRUE(sameTexture(passes[taaCopyBackIndex].colorTexture,
                          frameContext.sharedResources.sceneColorTexture));
  EXPECT_TRUE(sameTexture(passes[sceneResolveIndex].colorTexture,
                          frameContext.sharedResources.frameColorTexture));
}

TEST(RenderGraphRendererTest,
     TemporalAAFeatureFeedsPostTaaSceneColorMipsToTransmission) {
  std::vector<std::byte> scratchBytes(1024 * 1024);
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeStableBindlessShadowSceneGpuDevice gpu;
  gpu.latestCompletedGpuTimingReport = GpuTimingReport{
      .sceneColorDownsampleSourceFrameIndex = 2u,
      .transmissionSourceFrameIndex = 2u,
      .sceneColorDownsampleTimeMs = 0.21f,
      .transmissionTimeMs = 0.37f,
      .availableScopeMask = kGpuTimingScopeSceneColorDownsampleBit |
                            kGpuTimingScopeTransmissionBit,
  };
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(
      std::make_unique<FrameCompositionProvider>(gpu, &memory));
  pipeline.addProvider(std::make_unique<MaterialTableGpuProvider>(gpu));
  pipeline.addProvider(std::make_unique<SceneLightingProvider>(gpu));
  pipeline.addFeature(std::make_unique<TemporalAAFeature>(
      gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));
  pipeline.addFeature(std::make_unique<FrameCompositionFeature>(
      gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));
  pipeline.addFeature(std::make_unique<TransmissionFeature>(
      gpu, makeOpaqueConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
      &memory));

  const std::filesystem::path tempDir =
      makeTempRendererPath("taa_transmission_mips");
  ASSERT_TRUE(std::filesystem::create_directories(tempDir));
  const std::filesystem::path objPath = tempDir / "triangle.obj";
  writeTransmissionTriangleObj(objPath);

  auto modelResult = renderer.resources().acquireModel(ModelRequest{
      .path = objPath.string(),
      .debugName = "taa_transmission_triangle",
  });
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();
  MaterialRequest materialRequest{};
  materialRequest.debugName = "taa_transmission_material";
  materialRequest.desc.transmissionFactor = 1.0f;
  auto materialResult = renderer.resources().acquireMaterial(materialRequest);
  ASSERT_FALSE(materialResult.hasError()) << materialResult.error();

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  auto renderableResult = scene.graph().addRenderable(
      scene.graph().rootNode(), modelResult.value(), materialResult.value());
  ASSERT_FALSE(renderableResult.hasError()) << renderableResult.error();
  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();
  ASSERT_TRUE(commitResult.value());

  RenderSettings settings{};
  settings.antiAliasing.mode = AntiAliasingMode::TAA;
  settings.transmission.enabled = true;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 3u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  frameContext.camera = makeSdsmPerspectiveCamera(30.0f);
  frameContext.camera.historyValid = true;
  frameContext.camera.temporalDataValid = true;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const std::span<const RenderPass> passes(
      compileResult.value().orderedPasses.data(),
      compileResult.value().orderedPasses.size());

  const uint32_t copyBackIndex =
      compiledPassIndex(passes, "TAA Copy Back Pass");
  const uint32_t halfIndex =
      compiledPassIndex(passes, "Scene Color Downsample Half");
  const uint32_t quarterIndex =
      compiledPassIndex(passes, "Scene Color Downsample Quarter");
  const uint32_t transmissionIndex =
      compiledPassIndex(passes, "Transmission Pass");
  ASSERT_NE(copyBackIndex, UINT32_MAX);
  ASSERT_NE(halfIndex, UINT32_MAX);
  ASSERT_NE(quarterIndex, UINT32_MAX);
  ASSERT_NE(transmissionIndex, UINT32_MAX);
  EXPECT_LT(copyBackIndex, halfIndex);
  EXPECT_LT(halfIndex, quarterIndex);
  EXPECT_LT(quarterIndex, transmissionIndex);
  EXPECT_EQ(passes[halfIndex].gpuTimingScope,
            GpuTimingScope::SceneColorDownsample);
  EXPECT_EQ(passes[quarterIndex].gpuTimingScope,
            GpuTimingScope::SceneColorDownsample);
  EXPECT_EQ(passes[transmissionIndex].gpuTimingScope,
            GpuTimingScope::Transmission);

  EXPECT_TRUE(nuri::isValid(frameContext.sharedResources.sceneColorTexture));
  EXPECT_TRUE(
      nuri::isValid(frameContext.sharedResources.sceneColorHalfResTexture));
  EXPECT_TRUE(
      nuri::isValid(frameContext.sharedResources.sceneColorQuarterResTexture));
  EXPECT_EQ(
      frameContext.metrics.antiAliasing.taaPostResolveSceneColorMipPassCount,
      2u);
  EXPECT_TRUE(frameContext.metrics.antiAliasing
                  .taaPostResolveSceneColorMipChainGenerated);
  EXPECT_TRUE(frameContext.metrics.antiAliasing
                  .taaTransmissionPostResolveSceneColorConsumed);
  EXPECT_EQ(frameContext.metrics.antiAliasing
                .taaTransmissionStaleSceneColorFrameCount,
            0u);
  EXPECT_EQ(frameContext.metrics.antiAliasing
                .taaSceneColorDownsampleGpuTimingAvailable,
            1u);
  EXPECT_FLOAT_EQ(
      frameContext.metrics.antiAliasing.taaSceneColorDownsampleGpuTimeMs,
      0.21f);
  EXPECT_EQ(frameContext.metrics.antiAliasing.taaTransmissionGpuTimingAvailable,
            1u);
  EXPECT_FLOAT_EQ(frameContext.metrics.antiAliasing.taaTransmissionGpuTimeMs,
                  0.37f);
  EXPECT_FLOAT_EQ(
      frameContext.metrics.antiAliasing.taaTransmissionFlickerEstimate, 0.0f);

  std::filesystem::remove_all(tempDir);
}

TEST(RenderGraphRendererTest,
     SceneLightingFrameDataSceneColorIdsMatchPostTaaTextureHandles) {
  std::vector<std::byte> scratchBytes(1024 * 1024);
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeStableBindlessShadowSceneGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(
      std::make_unique<FrameCompositionProvider>(gpu, &memory));
  pipeline.addProvider(std::make_unique<MaterialTableGpuProvider>(gpu));
  pipeline.addProvider(std::make_unique<SceneLightingProvider>(gpu));
  pipeline.addFeature(std::make_unique<TemporalAAFeature>(
      gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));
  pipeline.addFeature(std::make_unique<FrameCompositionFeature>(
      gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();

  RenderSettings settings{};
  settings.antiAliasing.mode = AntiAliasingMode::TAA;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 5u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  frameContext.camera = makeSdsmPerspectiveCamera(30.0f);
  frameContext.camera.historyValid = true;
  frameContext.camera.temporalDataValid = true;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());

  ASSERT_TRUE(frameContext.sharedResources.forwardSceneGpuData.has_value());
  const ForwardSceneGpuData &sceneGpu =
      *frameContext.sharedResources.forwardSceneGpuData;
  ASSERT_TRUE(nuri::isValid(sceneGpu.buffer));
  ForwardSceneFrameData uploaded{};
  std::span<std::byte> uploadedBytes(reinterpret_cast<std::byte *>(&uploaded),
                                     sizeof(uploaded));
  auto readResult = gpu.readBuffer(sceneGpu.buffer, 0u, uploadedBytes);
  ASSERT_FALSE(readResult.hasError()) << readResult.error();

  EXPECT_EQ(uploaded.sceneColorTexId,
            gpu.getTextureBindlessIndex(
                frameContext.sharedResources.sceneColorTexture));
  EXPECT_EQ(uploaded.sceneColorHalfResTexId,
            gpu.getTextureBindlessIndex(
                frameContext.sharedResources.sceneColorHalfResTexture));
  EXPECT_EQ(uploaded.sceneColorQuarterResTexId,
            gpu.getTextureBindlessIndex(
                frameContext.sharedResources.sceneColorQuarterResTexture));

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const std::span<const RenderPass> passes(
      compileResult.value().orderedPasses.data(),
      compileResult.value().orderedPasses.size());
  const uint32_t copyBackIndex =
      compiledPassIndex(passes, "TAA Copy Back Pass");
  ASSERT_NE(copyBackIndex, UINT32_MAX);
  EXPECT_TRUE(sameTexture(passes[copyBackIndex].colorTexture,
                          frameContext.sharedResources.sceneColorTexture));
  EXPECT_TRUE(frameContext.metrics.antiAliasing
                  .taaPostResolveSceneColorMipChainGenerated);
}

TEST(RenderGraphRendererTest,
     FrameCompositionRendersPostTaaSceneColorMipDebugViews) {
  struct DebugMipCase {
    AntiAliasingDebugView view = AntiAliasingDebugView::None;
    TextureHandle FrameSharedResources::*texture = nullptr;
  };
  const std::array<DebugMipCase, 2> cases{{
      {.view = AntiAliasingDebugView::TAASceneColorHalfRes,
       .texture = &FrameSharedResources::sceneColorHalfResTexture},
      {.view = AntiAliasingDebugView::TAASceneColorQuarterRes,
       .texture = &FrameSharedResources::sceneColorQuarterResTexture},
  }};

  for (const DebugMipCase &debugCase : cases) {
    std::vector<std::byte> scratchBytes(64 * 1024);
    std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                               scratchBytes.size());
    FakeStableBindlessShadowSceneGpuDevice gpu;
    Renderer renderer(gpu, memory);
    RenderPipeline pipeline(&memory);
    pipeline.addProvider(
        std::make_unique<FrameCompositionProvider>(gpu, &memory));
    pipeline.addFeature(std::make_unique<TemporalAAFeature>(
        gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));
    pipeline.addFeature(std::make_unique<FrameCompositionFeature>(
        gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));

    RenderSettings settings{};
    settings.antiAliasing.mode = AntiAliasingMode::TAA;
    settings.antiAliasing.debug.view = debugCase.view;

    RenderFrameContext frameContext{};
    frameContext.frameIndex = 7u;
    frameContext.resources = &renderer.resources();
    frameContext.settings = &settings;
    frameContext.camera.historyValid = true;
    frameContext.camera.temporalDataValid = true;

    RenderGraphBuilder graph(&memory);
    graph.beginFrame(frameContext.frameIndex);
    auto buildResult =
        pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
    ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
    ASSERT_TRUE(buildResult.value());

    RenderGraphRuntime runtime;
    auto compileResult = graph.compile(runtime);
    ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
    const std::span<const RenderPass> passes(
        compileResult.value().orderedPasses.data(),
        compileResult.value().orderedPasses.size());
    const uint32_t resolveIndex =
        compiledPassIndex(passes, "Scene Resolve Pass");
    ASSERT_NE(resolveIndex, UINT32_MAX);
    ASSERT_FALSE(passes[resolveIndex].draws.empty());
    const DrawItem &draw = passes[resolveIndex].draws.front();
    ASSERT_EQ(draw.pushConstants.size(), sizeof(CopyPushConstants));
    CopyPushConstants probe{};
    std::memcpy(&probe, draw.pushConstants.data(), sizeof(probe));
    const TextureHandle expectedSource =
        frameContext.sharedResources.*(debugCase.texture);
    EXPECT_EQ(probe.sourceTexId, gpu.getTextureBindlessIndex(expectedSource));
    EXPECT_TRUE(
        frameContext.metrics.antiAliasing.taaSceneColorMipDebugViewRendered);
    EXPECT_TRUE(frameContext.metrics.antiAliasing
                    .taaPostResolveSceneColorMipChainGenerated);
  }
}

TEST(RenderGraphRendererTest,
     TransmissionMipSourceDebugViewPublishesShaderFlagAndMetric) {
  std::vector<std::byte> scratchBytes(1024 * 1024);
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeStableBindlessShadowSceneGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(
      std::make_unique<FrameCompositionProvider>(gpu, &memory));
  pipeline.addProvider(std::make_unique<MaterialTableGpuProvider>(gpu));
  pipeline.addProvider(std::make_unique<SceneLightingProvider>(gpu));
  pipeline.addFeature(std::make_unique<TemporalAAFeature>(
      gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));
  pipeline.addFeature(std::make_unique<FrameCompositionFeature>(
      gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));
  pipeline.addFeature(std::make_unique<TransmissionFeature>(
      gpu, makeOpaqueConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
      &memory));

  const std::filesystem::path tempDir =
      makeTempRendererPath("taa_transmission_mip_debug");
  ASSERT_TRUE(std::filesystem::create_directories(tempDir));
  const std::filesystem::path objPath = tempDir / "triangle.obj";
  writeTransmissionTriangleObj(objPath);

  auto modelResult = renderer.resources().acquireModel(ModelRequest{
      .path = objPath.string(),
      .debugName = "taa_transmission_mip_debug_triangle",
  });
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();
  MaterialRequest materialRequest{};
  materialRequest.debugName = "taa_transmission_mip_debug_material";
  materialRequest.desc.transmissionFactor = 1.0f;
  auto materialResult = renderer.resources().acquireMaterial(materialRequest);
  ASSERT_FALSE(materialResult.hasError()) << materialResult.error();

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  auto renderableResult = scene.graph().addRenderable(
      scene.graph().rootNode(), modelResult.value(), materialResult.value());
  ASSERT_FALSE(renderableResult.hasError()) << renderableResult.error();
  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();

  RenderSettings settings{};
  settings.antiAliasing.mode = AntiAliasingMode::TAA;
  settings.antiAliasing.debug.view =
      AntiAliasingDebugView::TAATransmissionMipSource;
  settings.transmission.enabled = true;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 9u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  frameContext.camera = makeSdsmPerspectiveCamera(30.0f);
  frameContext.camera.historyValid = true;
  frameContext.camera.temporalDataValid = true;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());

  ASSERT_TRUE(frameContext.sharedResources.forwardSceneGpuData.has_value());
  const ForwardSceneGpuData &sceneGpu =
      *frameContext.sharedResources.forwardSceneGpuData;
  ForwardSceneFrameData uploaded{};
  std::span<std::byte> uploadedBytes(reinterpret_cast<std::byte *>(&uploaded),
                                     sizeof(uploaded));
  auto readResult = gpu.readBuffer(sceneGpu.buffer, 0u, uploadedBytes);
  ASSERT_FALSE(readResult.hasError()) << readResult.error();
  EXPECT_NE(uploaded.flags & kForwardSceneTransmissionMipDebugProbe, 0u);

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const std::span<const RenderPass> passes(
      compileResult.value().orderedPasses.data(),
      compileResult.value().orderedPasses.size());
  const uint32_t transmissionIndex =
      compiledPassIndex(passes, "Transmission Pass");
  ASSERT_NE(transmissionIndex, UINT32_MAX);
  EXPECT_EQ(passes[transmissionIndex].gpuTimingScope,
            GpuTimingScope::Transmission);
  EXPECT_TRUE(
      frameContext.metrics.antiAliasing.taaTransmissionMipDebugViewRendered);
  EXPECT_EQ(frameContext.metrics.antiAliasing.taaTransmissionMipDebugPassCount,
            1u);
  EXPECT_TRUE(frameContext.metrics.antiAliasing
                  .taaTransmissionPostResolveSceneColorConsumed);

  std::filesystem::remove_all(tempDir);
}

TEST(RenderGraphRendererTest,
     TemporalAAFeatureKeepsDebugOverlayOutsideHistory) {
  std::array<std::byte, 256 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(
      std::make_unique<FrameCompositionProvider>(gpu, &memory));
  pipeline.addFeature(std::make_unique<TemporalAAFeature>(
      gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));
  pipeline.addFeature(std::make_unique<FrameCompositionFeature>(
      gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));
  DebugRendererConfig debugConfig{};
  const std::filesystem::path shaderRoot =
      std::filesystem::path(PROJECT_SOURCE_DIR) / "assets" / "shaders";
  debugConfig.vertex = shaderRoot / "grid.vert";
  debugConfig.fragment = shaderRoot / "grid.frag";
  pipeline.addFeature(
      std::make_unique<DebugFeature>(gpu, debugConfig, &memory));

  RenderSettings settings{};
  settings.antiAliasing.mode = AntiAliasingMode::TAA;
  settings.debug.grid = true;
  settings.debug.lightIcons = false;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 10u;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  frameContext.camera.view = glm::mat4(1.0f);
  frameContext.camera.proj = glm::mat4(1.0f);
  frameContext.camera.historyValid = true;
  frameContext.camera.temporalDataValid = true;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const std::span<const RenderPass> passes(
      compileResult.value().orderedPasses.data(),
      compileResult.value().orderedPasses.size());
  const uint32_t copyBackIndex =
      compiledPassIndex(passes, "TAA Copy Back Pass");
  const uint32_t resolveIndex = compiledPassIndex(passes, "Scene Resolve Pass");
  const uint32_t debugGridIndex = compiledPassIndex(passes, "DebugGrid Pass");
  ASSERT_NE(copyBackIndex, UINT32_MAX);
  ASSERT_NE(resolveIndex, UINT32_MAX);
  ASSERT_NE(debugGridIndex, UINT32_MAX);
  EXPECT_LT(copyBackIndex, resolveIndex);
  EXPECT_LT(resolveIndex, debugGridIndex);
  EXPECT_EQ(frameContext.metrics.antiAliasing.taaOverlayPostTaaDrawCount, 1u);
  EXPECT_EQ(frameContext.metrics.antiAliasing
                .taaOverlayHistoryContaminationFrameCount,
            0u);
}

TEST(RenderGraphRendererTest,
     TemporalAAFeatureTracksTransparentEdgeJitterOutsideHistory) {
  std::vector<std::byte> scratchBytes(1024 * 1024);
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeStableBindlessShadowSceneGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(
      std::make_unique<FrameCompositionProvider>(gpu, &memory));
  pipeline.addProvider(std::make_unique<MaterialTableGpuProvider>(gpu));
  pipeline.addProvider(std::make_unique<SceneLightingProvider>(gpu));
  pipeline.addFeature(std::make_unique<TemporalAAFeature>(
      gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));
  pipeline.addFeature(std::make_unique<FrameCompositionFeature>(
      gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));
  pipeline.addFeature(std::make_unique<TransparentFeature>(
      gpu, makeOpaqueConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
      &memory));

  const std::filesystem::path tempDir =
      makeTempRendererPath("taa_transparent_jitter");
  ASSERT_TRUE(std::filesystem::create_directories(tempDir));
  const std::filesystem::path objPath = tempDir / "transparent.obj";
  writeTransmissionTriangleObj(objPath);

  auto modelResult = renderer.resources().acquireModel(ModelRequest{
      .path = objPath.string(),
      .debugName = "taa_transparent_triangle",
  });
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();
  MaterialRequest materialRequest{};
  materialRequest.debugName = "taa_transparent_material";
  materialRequest.desc.alphaMode = MaterialAlphaMode::Blend;
  materialRequest.desc.baseColorFactor = glm::vec4(1.0f, 1.0f, 1.0f, 0.45f);
  auto materialResult = renderer.resources().acquireMaterial(materialRequest);
  ASSERT_FALSE(materialResult.hasError()) << materialResult.error();

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  auto renderableResult = scene.graph().addRenderable(
      scene.graph().rootNode(), modelResult.value(), materialResult.value());
  ASSERT_FALSE(renderableResult.hasError()) << renderableResult.error();
  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();

  RenderSettings settings{};
  settings.antiAliasing.mode = AntiAliasingMode::TAA;
  settings.transparent.enabled = true;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 11u;
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;
  frameContext.camera = makeSdsmPerspectiveCamera(30.0f);
  frameContext.camera.historyValid = true;
  frameContext.camera.temporalDataValid = true;
  frameContext.camera.jitterEnabled = true;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const std::span<const RenderPass> passes(
      compileResult.value().orderedPasses.data(),
      compileResult.value().orderedPasses.size());
  const uint32_t copyBackIndex =
      compiledPassIndex(passes, "TAA Copy Back Pass");
  const uint32_t transparentIndex =
      compiledPassIndex(passes, "Transparent Pass");
  ASSERT_NE(copyBackIndex, UINT32_MAX);
  ASSERT_NE(transparentIndex, UINT32_MAX);
  EXPECT_LT(copyBackIndex, transparentIndex);
  EXPECT_GT(frameContext.metrics.antiAliasing.taaTransparentPostTaaDrawCount,
            0u);
  EXPECT_EQ(
      frameContext.metrics.antiAliasing.taaTransparentPostTaaMeshDrawCount,
      frameContext.metrics.antiAliasing.taaTransparentPostTaaDrawCount);
  EXPECT_EQ(frameContext.metrics.antiAliasing
                .taaTransparentPostTaaContributorDrawCount,
            0u);
  EXPECT_EQ(
      frameContext.metrics.antiAliasing.taaTransparentPostTaaFixedDrawCount,
      0u);
  EXPECT_TRUE(
      frameContext.metrics.antiAliasing.taaTransparentEdgeJitterTracked);
  EXPECT_FLOAT_EQ(
      frameContext.metrics.antiAliasing.taaTransparentEdgeJitterEstimate, 1.0f);

  std::filesystem::remove_all(tempDir);
}

TEST(RenderGraphRendererTest,
     TemporalAAFeatureAdvancesHistoryForPreviousHistoryDebugView) {
  std::array<std::byte, 32 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(
      std::make_unique<FrameCompositionProvider>(gpu, &memory));
  pipeline.addFeature(std::make_unique<TemporalAAFeature>(
      gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));

  RenderSettings settings{};
  settings.antiAliasing.mode = AntiAliasingMode::TAA;
  settings.antiAliasing.debug.view = AntiAliasingDebugView::TAAPreviousHistory;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 1u;
  frameContext.camera.historyValid = true;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());

  EXPECT_TRUE(
      isValid(frameContext.sharedResources.previousMotionVectorTexture));
  EXPECT_TRUE(
      isValid(frameContext.sharedResources.previousMotionVectorGraphTexture));
  EXPECT_TRUE(frameContext.metrics.antiAliasing.previousMotionVectorValid);
  EXPECT_TRUE(
      frameContext.metrics.antiAliasing.previousMotionVectorGraphPublished);
  EXPECT_EQ(frameContext.metrics.antiAliasing.taaResolvePassCount, 1u);
  EXPECT_EQ(frameContext.metrics.antiAliasing.taaCopyBackPassCount, 1u);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.taaDebugViewRendered);
  EXPECT_FALSE(
      frameContext.metrics.antiAliasing.taaResolvedSceneColorPublished);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.taaDepthRejectionEnabled);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.taaVelocityRejectionEnabled);

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  ASSERT_EQ(compileResult.value().orderedPasses.size(), 4u);
  EXPECT_EQ(compileResult.value().orderedPasses[0].debugLabel,
            "Temporal AA Motion Vector Clear");
  EXPECT_EQ(compileResult.value().orderedPasses[1].debugLabel,
            "Temporal AA Reactive Mask Clear");
  EXPECT_EQ(compileResult.value().orderedPasses[2].debugLabel,
            "TAA Resolve Pass");
  EXPECT_EQ(compileResult.value().orderedPasses[3].debugLabel,
            "TAA Previous History Debug Pass");
}

TEST(RenderGraphRendererTest,
     TemporalAAFeatureRendersPhase5DebugViewsAfterHiddenHistoryResolve) {
  std::array<std::byte, 32 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(
      std::make_unique<FrameCompositionProvider>(gpu, &memory));
  pipeline.addFeature(std::make_unique<TemporalAAFeature>(
      gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));

  RenderSettings settings{};
  settings.antiAliasing.mode = AntiAliasingMode::TAA;
  settings.antiAliasing.debug.view = AntiAliasingDebugView::TAARejectionMask;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 1u;
  frameContext.camera.historyValid = true;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());

  EXPECT_TRUE(
      isValid(frameContext.sharedResources.previousMotionVectorTexture));
  EXPECT_TRUE(
      isValid(frameContext.sharedResources.previousMotionVectorGraphTexture));
  EXPECT_TRUE(frameContext.metrics.antiAliasing.previousMotionVectorValid);
  EXPECT_TRUE(
      frameContext.metrics.antiAliasing.previousMotionVectorGraphPublished);
  EXPECT_EQ(frameContext.metrics.antiAliasing.taaResolvePassCount, 1u);
  EXPECT_EQ(frameContext.metrics.antiAliasing.taaCopyBackPassCount, 2u);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.taaDebugViewRendered);
  EXPECT_FALSE(
      frameContext.metrics.antiAliasing.taaResolvedSceneColorPublished);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.taaDepthRejectionEnabled);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.taaVelocityRejectionEnabled);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.taaNeighborhoodClampEnabled);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.taaHdrWeightingEnabled);
  EXPECT_FLOAT_EQ(
      frameContext.metrics.antiAliasing.taaDepthDiscontinuityThreshold, 0.01f);
  EXPECT_FLOAT_EQ(
      frameContext.metrics.antiAliasing.taaVelocityRejectionThreshold, 0.0015f);

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const auto &orderedPasses = compileResult.value().orderedPasses;
  ASSERT_EQ(orderedPasses.size(), 5u);
  EXPECT_EQ(orderedPasses[0].debugLabel, "Temporal AA Motion Vector Clear");
  EXPECT_EQ(orderedPasses[1].debugLabel, "Temporal AA Reactive Mask Clear");
  EXPECT_EQ(orderedPasses[2].debugLabel, "TAA Resolve Pass");
  EXPECT_EQ(orderedPasses[3].debugLabel, "TAA Rejection Mask Debug Pass");
  EXPECT_TRUE(sameTexture(orderedPasses[3].colorTexture,
                          frameContext.sharedResources.frameColorTexture));
  EXPECT_FALSE(sameTexture(orderedPasses[3].colorTexture,
                           frameContext.sharedResources.sceneColorTexture));
  EXPECT_EQ(orderedPasses[4].debugLabel, "TAA Debug Copy Back Pass");
  EXPECT_TRUE(sameTexture(orderedPasses[4].colorTexture,
                          frameContext.sharedResources.sceneColorTexture));
}

TEST(RenderGraphRendererTest, TemporalAAFeatureRendersPixelInspectorDebugView) {
  std::array<std::byte, 32 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(
      std::make_unique<FrameCompositionProvider>(gpu, &memory));
  pipeline.addFeature(std::make_unique<TemporalAAFeature>(
      gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));

  RenderSettings settings{};
  settings.antiAliasing.mode = AntiAliasingMode::TAA;
  settings.antiAliasing.debug.view = AntiAliasingDebugView::TAAPixelInspector;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 1u;
  frameContext.camera.historyValid = true;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());

  EXPECT_TRUE(
      isValid(frameContext.sharedResources.previousMotionVectorGraphTexture));
  EXPECT_TRUE(frameContext.metrics.antiAliasing.taaDebugViewRendered);
  EXPECT_TRUE(
      frameContext.metrics.antiAliasing.taaPixelInspectorDebugViewRendered);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.taaNeighborhoodClampEnabled);
  EXPECT_EQ(frameContext.metrics.antiAliasing.taaResolvePassCount, 1u);
  EXPECT_EQ(frameContext.metrics.antiAliasing.taaCopyBackPassCount, 2u);

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const auto &orderedPasses = compileResult.value().orderedPasses;
  ASSERT_EQ(orderedPasses.size(), 5u);
  EXPECT_EQ(orderedPasses[0].debugLabel, "Temporal AA Motion Vector Clear");
  EXPECT_EQ(orderedPasses[1].debugLabel, "Temporal AA Reactive Mask Clear");
  EXPECT_EQ(orderedPasses[2].debugLabel, "TAA Resolve Pass");
  EXPECT_EQ(orderedPasses[3].debugLabel, "TAA Pixel Inspector Debug Pass");
  EXPECT_TRUE(sameTexture(orderedPasses[3].colorTexture,
                          frameContext.sharedResources.frameColorTexture));
  EXPECT_FALSE(sameTexture(orderedPasses[3].colorTexture,
                           frameContext.sharedResources.sceneColorTexture));
  EXPECT_EQ(orderedPasses[4].debugLabel, "TAA Debug Copy Back Pass");
  EXPECT_TRUE(sameTexture(orderedPasses[4].colorTexture,
                          frameContext.sharedResources.sceneColorTexture));
}

TEST(RenderGraphRendererTest, TemporalAAFeatureRendersResolveDebugViews) {
  struct DebugViewCase {
    AntiAliasingDebugView view = AntiAliasingDebugView::None;
    std::string_view label{};
  };
  const std::array<DebugViewCase, 11> cases{{
      {.view = AntiAliasingDebugView::TAAReactiveMask,
       .label = "TAA Reactive Mask Debug Pass"},
      {.view = AntiAliasingDebugView::TAADisocclusionMask,
       .label = "TAA Disocclusion Mask Debug Pass"},
      {.view = AntiAliasingDebugView::TAAVelocityDilation,
       .label = "TAA Velocity Dilation Debug Pass"},
      {.view = AntiAliasingDebugView::TAAReprojectedHistory,
       .label = "TAA Reprojected History Debug Pass"},
      {.view = AntiAliasingDebugView::TAAResolveConfidence,
       .label = "TAA Resolve Confidence Debug Pass"},
      {.view = AntiAliasingDebugView::TAAClampDiagnostics,
       .label = "TAA Clamp Diagnostics Debug Pass"},
      {.view = AntiAliasingDebugView::TAAPreviousVelocity,
       .label = "TAA Previous Velocity Debug Pass"},
      {.view = AntiAliasingDebugView::TAAHdrWeight,
       .label = "TAA HDR Weight Debug Pass"},
      {.view = AntiAliasingDebugView::TAAHistoryFilterDelta,
       .label = "TAA History Filter Delta Debug Pass"},
      {.view = AntiAliasingDebugView::TAADisocclusionFallback,
       .label = "TAA Disocclusion Fallback Debug Pass"},
      {.view = AntiAliasingDebugView::TAASplitCompare,
       .label = "TAA Split Compare Debug Pass"},
  }};

  for (const DebugViewCase &debugCase : cases) {
    std::array<std::byte, 32 * 1024> scratchBytes{};
    std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                               scratchBytes.size());
    FakeFullscreenGpuDevice gpu;
    Renderer renderer(gpu, memory);
    RenderPipeline pipeline(&memory);
    pipeline.addProvider(
        std::make_unique<FrameCompositionProvider>(gpu, &memory));
    pipeline.addFeature(std::make_unique<TemporalAAFeature>(
        gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));

    RenderSettings settings{};
    settings.antiAliasing.mode = AntiAliasingMode::TAA;
    settings.antiAliasing.debug.view = debugCase.view;
    settings.antiAliasing.debug.taaVelocityDilationMode =
        TemporalAAVelocityDilationMode::ClosestDepth;

    RenderFrameContext frameContext{};
    frameContext.frameIndex = 1u;
    frameContext.camera.historyValid = true;
    frameContext.resources = &renderer.resources();
    frameContext.settings = &settings;

    RenderGraphBuilder graph(&memory);
    graph.beginFrame(frameContext.frameIndex);
    auto buildResult =
        pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
    ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
    ASSERT_TRUE(buildResult.value());

    EXPECT_TRUE(frameContext.metrics.antiAliasing.taaDebugViewRendered);
    EXPECT_TRUE(frameContext.metrics.antiAliasing.taaReactiveMaskEnabled);
    EXPECT_TRUE(frameContext.metrics.antiAliasing.taaVelocityDilationEnabled);
    const bool temporalEvaluationDebug =
        debugCase.view != AntiAliasingDebugView::TAAReactiveMask &&
        debugCase.view != AntiAliasingDebugView::TAAPreviousVelocity;
    EXPECT_EQ(frameContext.metrics.antiAliasing.taaCopyBackPassCount,
              temporalEvaluationDebug ? 2u : 1u);
    if (debugCase.view == AntiAliasingDebugView::TAAReactiveMask) {
      EXPECT_TRUE(
          frameContext.metrics.antiAliasing.taaReactiveMaskDebugViewRendered);
    } else if (debugCase.view == AntiAliasingDebugView::TAADisocclusionMask) {
      EXPECT_TRUE(frameContext.metrics.antiAliasing
                      .taaDisocclusionMaskDebugViewRendered);
    } else if (debugCase.view == AntiAliasingDebugView::TAAVelocityDilation) {
      EXPECT_TRUE(frameContext.metrics.antiAliasing
                      .taaVelocityDilationDebugViewRendered);
    } else if (debugCase.view == AntiAliasingDebugView::TAAPreviousVelocity) {
      EXPECT_TRUE(frameContext.metrics.antiAliasing
                      .taaPreviousVelocityDebugViewRendered);
    } else if (debugCase.view == AntiAliasingDebugView::TAAHdrWeight) {
      EXPECT_TRUE(
          frameContext.metrics.antiAliasing.taaHdrWeightDebugViewRendered);
    } else if (debugCase.view == AntiAliasingDebugView::TAAHistoryFilterDelta) {
      EXPECT_TRUE(frameContext.metrics.antiAliasing
                      .taaHistoryFilterDeltaDebugViewRendered);
    } else if (debugCase.view ==
               AntiAliasingDebugView::TAADisocclusionFallback) {
      EXPECT_TRUE(frameContext.metrics.antiAliasing
                      .taaDisocclusionFallbackDebugViewRendered);
    } else if (debugCase.view == AntiAliasingDebugView::TAASplitCompare) {
      EXPECT_TRUE(
          frameContext.metrics.antiAliasing.taaSplitCompareDebugViewRendered);
    }

    RenderGraphRuntime runtime;
    auto compileResult = graph.compile(runtime);
    ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
    const auto &orderedPasses = compileResult.value().orderedPasses;
    ASSERT_EQ(orderedPasses.size(), temporalEvaluationDebug ? 5u : 4u);
    EXPECT_EQ(orderedPasses[3].debugLabel, debugCase.label);
    if (temporalEvaluationDebug) {
      EXPECT_TRUE(sameTexture(orderedPasses[3].colorTexture,
                              frameContext.sharedResources.frameColorTexture));
      EXPECT_FALSE(sameTexture(orderedPasses[3].colorTexture,
                               frameContext.sharedResources.sceneColorTexture));
      EXPECT_EQ(orderedPasses[4].debugLabel, "TAA Debug Copy Back Pass");
      EXPECT_TRUE(sameTexture(orderedPasses[4].colorTexture,
                              frameContext.sharedResources.sceneColorTexture));
    } else {
      EXPECT_TRUE(sameTexture(orderedPasses[3].colorTexture,
                              frameContext.sharedResources.sceneColorTexture));
    }
  }
}

TEST(RenderGraphRendererTest,
     TemporalAAFeatureDisplaysCurrentColorWhileAdvancingHistory) {
  std::array<std::byte, 32 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  pipeline.addProvider(
      std::make_unique<FrameCompositionProvider>(gpu, &memory));
  pipeline.addFeature(std::make_unique<TemporalAAFeature>(
      gpu, makeCompositeConfig(std::filesystem::path(PROJECT_SOURCE_DIR))));

  RenderSettings settings{};
  settings.antiAliasing.mode = AntiAliasingMode::TAA;
  settings.antiAliasing.debug.view = AntiAliasingDebugView::TAACurrentColor;

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 1u;
  frameContext.camera.historyValid = true;
  frameContext.resources = &renderer.resources();
  frameContext.settings = &settings;

  RenderGraphBuilder graph(&memory);
  graph.beginFrame(frameContext.frameIndex);
  auto buildResult =
      pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
  ASSERT_FALSE(buildResult.hasError()) << buildResult.error();
  ASSERT_TRUE(buildResult.value());

  EXPECT_TRUE(
      isValid(frameContext.sharedResources.previousMotionVectorGraphTexture));
  EXPECT_TRUE(
      frameContext.metrics.antiAliasing.previousMotionVectorGraphPublished);
  EXPECT_EQ(frameContext.metrics.antiAliasing.taaResolvePassCount, 1u);
  EXPECT_EQ(frameContext.metrics.antiAliasing.taaCopyBackPassCount, 0u);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.taaDebugViewRendered);
  EXPECT_FALSE(
      frameContext.metrics.antiAliasing.taaResolvedSceneColorPublished);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.taaDepthRejectionEnabled);
  EXPECT_TRUE(frameContext.metrics.antiAliasing.taaVelocityRejectionEnabled);
  EXPECT_TRUE(isValid(frameContext.sharedResources.sceneColorGraphTexture));

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  ASSERT_EQ(compileResult.value().orderedPasses.size(), 3u);
  EXPECT_EQ(compileResult.value().orderedPasses[0].debugLabel,
            "Temporal AA Motion Vector Clear");
  EXPECT_EQ(compileResult.value().orderedPasses[1].debugLabel,
            "Temporal AA Reactive Mask Clear");
  EXPECT_EQ(compileResult.value().orderedPasses[2].debugLabel,
            "TAA Resolve Pass");
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

TEST(RenderGraphRendererTest,
     TemporalAAFeatureRegistersBetweenOpaqueAndFrameComposition) {
  std::array<std::byte, 32 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeRendererGPUDevice gpu;
  RenderPipeline pipeline(&memory);

  pipeline.addFeature(std::make_unique<ShadowFeature>(gpu, &memory));
  pipeline.addFeature(
      std::make_unique<SkyboxFeature>(gpu, RuntimeSkyboxShaderConfig{}));
  pipeline.addFeature(
      std::make_unique<OpaqueFeature>(gpu, OpaqueRendererConfig{}, &memory));
  pipeline.addFeature(
      std::make_unique<TemporalAAFeature>(gpu, RuntimeCompositeConfig{}));
  pipeline.addFeature(
      std::make_unique<SpatialAAFeature>(gpu, RuntimeCompositeConfig{}));
  pipeline.addFeature(
      std::make_unique<FrameCompositionFeature>(gpu, RuntimeCompositeConfig{}));

  ASSERT_EQ(pipeline.passCount(), 10u);
  const auto skybox = pipeline.passInfo(1u);
  const auto opaque = pipeline.passInfo(2u);
  const auto opaquePick = pipeline.passInfo(3u);
  const auto temporalClear = pipeline.passInfo(4u);
  const auto temporalReactiveClear = pipeline.passInfo(5u);
  const auto temporalResolve = pipeline.passInfo(6u);
  const auto spatial = pipeline.passInfo(7u);
  const auto composition = pipeline.passInfo(8u);
  ASSERT_TRUE(skybox.has_value());
  ASSERT_TRUE(opaque.has_value());
  ASSERT_TRUE(opaquePick.has_value());
  ASSERT_TRUE(temporalClear.has_value());
  ASSERT_TRUE(temporalReactiveClear.has_value());
  ASSERT_TRUE(temporalResolve.has_value());
  ASSERT_TRUE(spatial.has_value());
  ASSERT_TRUE(composition.has_value());
  EXPECT_EQ(skybox->featureName, "SkyboxFeature");
  EXPECT_EQ(opaque->featureName, "OpaqueFeature");
  EXPECT_EQ(opaquePick->featureName, "OpaqueFeature");
  EXPECT_EQ(temporalClear->featureName, "TemporalAAFeature");
  EXPECT_EQ(temporalClear->passName, "TemporalAAMotionVectorClearPass");
  EXPECT_EQ(temporalReactiveClear->featureName, "TemporalAAFeature");
  EXPECT_EQ(temporalReactiveClear->passName, "TemporalAAReactiveMaskClearPass");
  EXPECT_EQ(temporalResolve->featureName, "TemporalAAFeature");
  EXPECT_EQ(temporalResolve->passName, "TemporalAAResolvePass");
  EXPECT_EQ(spatial->featureName, "SpatialAAFeature");
  EXPECT_EQ(spatial->passName, "SpatialAAPass");
  EXPECT_EQ(composition->featureName, "FrameCompositionFeature");
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
      .shadowSourceFrameIndex = 7u,
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
  EXPECT_EQ(frameContext.metrics.shadow.pcssMaxSamplesPerReceiver, 60u);
  EXPECT_EQ(frameContext.metrics.shadow.pcssMaxSamplesPerBlendedReceiver, 120u);

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
  ASSERT_EQ(reducePass->preDispatches[0].dependencyTextures.size(), 1u);
}

TEST(RenderGraphRendererTest,
     OpaqueFeatureAppendsShadowSdsmHistogramReducePassForGpuHistogram) {
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
      makeTempRendererPath("sdsm_gpu_histogram_scene");
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
      .debugName = "sdsm_gpu_histogram_triangle",
  });
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();
  auto materialResult = renderer.resources().acquireMaterial(MaterialRequest{
      .debugName = "sdsm_gpu_histogram_material",
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
  settings.shadow.sdsmMode = ShadowSdsmMode::Histogram;
  settings.shadow.sdsmReductionBackend = ShadowSdsmReductionBackend::Gpu;
  settings.shadow.sdsmHistogramBucketCount = 16u;

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

  RenderGraphRuntime runtime;
  auto compileResult = graph.compile(runtime);
  ASSERT_FALSE(compileResult.hasError()) << compileResult.error();
  const RenderPass *reducePass = nullptr;
  for (const RenderPass &pass : compileResult.value().orderedPasses) {
    if (pass.debugLabel == "Shadow SDSM Histogram Reduce") {
      reducePass = &pass;
      break;
    }
  }
  ASSERT_NE(reducePass, nullptr);
  EXPECT_EQ(reducePass->executionMode, RenderPassExecutionMode::ComputeOnly);
  EXPECT_TRUE(reducePass->draws.empty());
  ASSERT_EQ(reducePass->preDispatches.size(), 1u);
  EXPECT_EQ(reducePass->preDispatches[0].debugLabel,
            "Shadow SDSM Histogram Reduce");
  ASSERT_EQ(reducePass->preDispatches[0].dependencyBuffers.size(), 1u);
  ASSERT_EQ(reducePass->preDispatches[0].dependencyTextures.size(), 1u);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureHistogramGpuReductionFallsBackToCpuWhenResourcesUnavailable) {
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
  EXPECT_FALSE(
      frameContext.sharedResources.shadowSdsmGpuReduceTarget.has_value());
  EXPECT_EQ(frameContext.sharedResources.shadowDebugFrameData->sdsm
                .requestedReductionBackend,
            ShadowSdsmReductionBackend::Gpu);
  EXPECT_EQ(frameContext.sharedResources.shadowDebugFrameData->sdsm
                .activeReductionBackend,
            ShadowSdsmReductionBackend::Cpu);
  EXPECT_TRUE(frameContext.sharedResources.shadowDebugFrameData->sdsm
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

      const GpuSdsmMinMaxResultProbe gpuResult{
          .rawDeviceMinMax = glm::vec2(0.2f, 0.5f),
          .sourceFrameIndex = 0u,
          .valid = 1u,
      };
      auto writeResult = gpu.updateBuffer(
          publishFrame.sharedResources.shadowSdsmGpuReduceTarget->buffer,
          std::as_bytes(
              std::span<const GpuSdsmMinMaxResultProbe>(&gpuResult, 1u)),
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
  expectGpuSdsmResultAvailable(gpuSdsm, 0u, false);
  EXPECT_EQ(cpuSdsm.gpuResultRingSlotCount, 0u);
  EXPECT_FALSE(cpuSdsm.gpuReductionResultAvailable);
  EXPECT_NEAR(gpuSdsm.rawLinearMin, cpuSdsm.rawLinearMin, 1.0e-5f);
  EXPECT_NEAR(gpuSdsm.rawLinearMax, cpuSdsm.rawLinearMax, 1.0e-5f);
  EXPECT_NEAR(gpuSdsm.effectiveRangeNear, cpuSdsm.effectiveRangeNear, 1.0e-5f);
  EXPECT_NEAR(gpuSdsm.effectiveRangeFar, cpuSdsm.effectiveRangeFar, 1.0e-5f);
  EXPECT_EQ(gpuSdsm.effectiveSplitDepths, cpuSdsm.effectiveSplitDepths);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureGpuHistogramSdsmConsumesPreviousFrameBuffer) {
  std::array<std::byte, 128 * 1024> scratchBytes{};
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
                        "sdsm_histogram_gpu_result");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();

  const CameraFrameState camera{
      .projectionType = ProjectionType::Perspective,
      .nearPlane = 0.1f,
      .farPlane = 30.0f,
  };

  RenderSettings publishSettings{};
  publishSettings.shadow.enabled = true;
  publishSettings.shadow.sdsmMode = ShadowSdsmMode::Histogram;
  publishSettings.shadow.sdsmReductionBackend = ShadowSdsmReductionBackend::Gpu;
  publishSettings.shadow.cascadeCount = 4u;
  publishSettings.shadow.sdsmHistogramBucketCount = 8u;

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
  ASSERT_TRUE(
      nuri::isValid(publishFrame.sharedResources.shadowSdsmGpuReducePipeline));

  GpuSdsmHistogramResultProbe gpuResult{};
  gpuResult.rawDeviceMinMaxLinearMinMax = glm::vec4(0.2f, 0.6f, 2.0f, 24.0f);
  gpuResult.histogramRangeWeightClear = glm::vec4(2.5f, 22.0f, 4.0f, 0.0f);
  gpuResult.metadata = glm::uvec4(0u, 1u, 4u, 8u);
  gpuResult.splitDepths0 = glm::vec4(0.1f, 4.0f, 9.0f, 15.0f);
  gpuResult.splitDepths1 = glm::vec4(30.0f, 24.0f, 0.0f, 0.0f);
  gpuResult.bucketWeights[0] = 1.0f;
  gpuResult.bucketWeights[3] = 2.0f;
  gpuResult.bucketWeights[7] = 1.0f;
  auto writeResult = gpu.updateBuffer(
      publishFrame.sharedResources.shadowSdsmGpuReduceTarget->buffer,
      std::as_bytes(
          std::span<const GpuSdsmHistogramResultProbe>(&gpuResult, 1u)),
      0u);
  ASSERT_FALSE(writeResult.hasError()) << writeResult.error();

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.sdsmMode = ShadowSdsmMode::Histogram;
  settings.shadow.sdsmReductionBackend = ShadowSdsmReductionBackend::Gpu;
  settings.shadow.cascadeCount = 4u;
  settings.shadow.sdsmHistogramBucketCount = 8u;
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
  EXPECT_EQ(sdsm.status, ShadowSdsmStatus::Active);
  EXPECT_EQ(sdsm.activeReductionBackend, ShadowSdsmReductionBackend::Gpu);
  EXPECT_FALSE(sdsm.fixedFallbackActive);
  EXPECT_FALSE(sdsm.reductionFallbackActive);
  EXPECT_EQ(sdsm.sourceFrameIndex, 0u);
  expectGpuSdsmResultAvailable(sdsm, 0u, true);
  EXPECT_EQ(sdsm.histogramBucketCount, 8u);
  EXPECT_EQ(sdsm.histogramValidTileCount, 4u);
  EXPECT_FLOAT_EQ(sdsm.histogramTotalWeight, 4.0f);
  EXPECT_FLOAT_EQ(sdsm.histogramTrimmedRangeNear, 2.5f);
  EXPECT_FLOAT_EQ(sdsm.histogramTrimmedRangeFar, 22.0f);
  EXPECT_FLOAT_EQ(sdsm.rawDeviceMin, 0.2f);
  EXPECT_FLOAT_EQ(sdsm.rawDeviceMax, 0.6f);
  EXPECT_FLOAT_EQ(sdsm.rawLinearMin, 2.0f);
  EXPECT_FLOAT_EQ(sdsm.rawLinearMax, 24.0f);
  EXPECT_FLOAT_EQ(sdsm.histogramBucketWeights[3], 2.0f);
  EXPECT_FLOAT_EQ(sdsm.histogramSplitDepths[1], 4.0f);
  EXPECT_FLOAT_EQ(sdsm.histogramSplitDepths[2], 9.0f);
  EXPECT_FLOAT_EQ(sdsm.histogramSplitDepths[3], 15.0f);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureGpuHistogramSdsmFallsBackToCpuWhenResultIsStale) {
  std::array<std::byte, 128 * 1024> scratchBytes{};
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
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 4u, 4u),
                        "sdsm_histogram_stale_gpu_cpu_fallback");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();
  const TextureHandle pyramidTexture = pyramidTextureResult.value();

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

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.sdsmMode = ShadowSdsmMode::Histogram;
  settings.shadow.sdsmReductionBackend = ShadowSdsmReductionBackend::Gpu;
  settings.shadow.cascadeCount = 4u;
  settings.shadow.maxDistance = 150.0f;
  settings.shadow.sdsmHistogramBucketCount = 8u;
  settings.shadow.sdsmHistogramTrimLowPercent = 0.0f;
  settings.shadow.sdsmHistogramTrimHighPercent = 0.0f;
  settings.shadow.sdsmTemporalBlend = 0.0f;

  RenderFrameContext publishFrame{};
  publishFrame.frameIndex = 0u;
  publishFrame.scene = &scene;
  publishFrame.resources = &renderer.resources();
  publishFrame.settings = &settings;
  publishFrame.camera = camera;

  RenderGraphBuilder publishGraph(&memory);
  publishGraph.beginFrame(publishFrame.frameIndex);
  auto publishResult = pipeline.buildRenderGraph(
      publishFrame, renderer.resources(), publishGraph);
  ASSERT_FALSE(publishResult.hasError()) << publishResult.error();
  ASSERT_TRUE(publishResult.value());
  ASSERT_TRUE(
      publishFrame.sharedResources.shadowSdsmGpuReduceTarget.has_value());

  ShadowSdsmDebugFrameData sdsm{};
  for (uint64_t frameIndex = 1u; frameIndex <= 3u; ++frameIndex) {
    sdsm = buildShadowSdsmFrame(
        pipeline, renderer, memory, scene, settings, camera, frameIndex,
        frameIndex - 1u, std::span<const TextureHandle>(&pyramidTexture, 1u));
    ASSERT_EQ(sdsm.status, ShadowSdsmStatus::Active);
    ASSERT_FALSE(sdsm.fixedFallbackActive);
  }

  EXPECT_EQ(sdsm.activeReductionBackend, ShadowSdsmReductionBackend::Gpu);
  EXPECT_TRUE(sdsm.reductionFallbackActive);
  expectNoGpuSdsmResult(sdsm);
  EXPECT_EQ(sdsm.sourceFrameIndex, 2u);
  EXPECT_EQ(sdsm.histogramBucketCount, 8u);
  EXPECT_EQ(sdsm.histogramValidTileCount, 16u);
  EXPECT_FLOAT_EQ(sdsm.histogramTotalWeight, 16.0f);
  EXPECT_GT(sdsm.histogramTrimmedRangeFar, sdsm.histogramTrimmedRangeNear);
  EXPECT_GT(sdsm.smoothedLinearMax, sdsm.smoothedLinearMin);
  EXPECT_EQ(sdsm.effectiveSplitDepths,
            expectedHistogramEffectiveSplitDepths(sdsm));
}

TEST(RenderGraphRendererTest,
     ShadowFeatureGpuSdsmUsesNewestCompletedRingResultWhenLatestSlotIsInvalid) {
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
      const GpuSdsmMinMaxResultProbe gpuResult{
          .rawDeviceMinMax = glm::vec2(0.2f, 0.5f),
          .sourceFrameIndex = 0u,
          .valid = 1u,
      };
      auto writeResult = gpu.updateBuffer(
          publishFrame.sharedResources.shadowSdsmGpuReduceTarget->buffer,
          std::as_bytes(
              std::span<const GpuSdsmMinMaxResultProbe>(&gpuResult, 1u)),
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
  expectGpuSdsmResultAvailable(sdsm, 0u, false);
  EXPECT_FLOAT_EQ(sdsm.rawDeviceMin, 0.2f);
  EXPECT_FLOAT_EQ(sdsm.rawDeviceMax, 0.5f);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureGpuSdsmFallsBackToCpuWhenResultIsMissing) {
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
  EXPECT_EQ(sdsm.status, ShadowSdsmStatus::Active);
  EXPECT_EQ(sdsm.activeReductionBackend, ShadowSdsmReductionBackend::Gpu);
  EXPECT_FALSE(sdsm.fixedFallbackActive);
  EXPECT_FALSE(sdsm.reductionFallbackActive);
  expectNoGpuSdsmResult(sdsm);
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
  EXPECT_EQ(sdsm.status, ShadowSdsmStatus::Active);
  EXPECT_EQ(sdsm.activeReductionBackend, ShadowSdsmReductionBackend::Gpu);
  EXPECT_FALSE(sdsm.fixedFallbackActive);
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
  ASSERT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());
  EXPECT_EQ(frameContext.sharedResources.shadowDebugFrameData->sdsm.status,
            ShadowSdsmStatus::Active);
  EXPECT_FALSE(frameContext.sharedResources.shadowDebugFrameData->sdsm
                   .fixedFallbackActive);
  EXPECT_TRUE(frameContext.sharedResources.shadowDebugFrameData->sdsm
                  .reductionFallbackActive);

  EXPECT_EQ(countLogEntriesSince(baselineSequence, "GPU SDSM ring diagnostics"),
            1u);
  EXPECT_EQ(countLogEntriesSince(baselineSequence, "SDSM source warning"), 0u);
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
      const GpuSdsmMinMaxResultProbe gpuResult{
          .rawDeviceMinMax = glm::vec2(0.2f, 0.5f),
          .sourceFrameIndex = 10u,
          .valid = 1u,
      };
      auto writeResult = gpu.updateBuffer(
          publishFrame.sharedResources.shadowSdsmGpuReduceTarget->buffer,
          std::as_bytes(
              std::span<const GpuSdsmMinMaxResultProbe>(&gpuResult, 1u)),
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
  expectGpuSdsmResultAvailable(sdsm, 10u, false);
  EXPECT_FLOAT_EQ(sdsm.rawDeviceMin, 0.2f);
  EXPECT_FLOAT_EQ(sdsm.rawDeviceMax, 0.5f);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureGpuSdsmIgnoresRingResultWithMismatchedSourceFrameIndex) {
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

    const GpuSdsmMinMaxResultProbe gpuResult{
        .rawDeviceMinMax = publishFrameIndex == 0u ? glm::vec2(0.2f, 0.5f)
                                                   : glm::vec2(0.8f, 0.9f),
        .sourceFrameIndex = 0u,
        .valid = 1u,
    };
    auto writeResult = gpu.updateBuffer(
        publishFrame.sharedResources.shadowSdsmGpuReduceTarget->buffer,
        std::as_bytes(
            std::span<const GpuSdsmMinMaxResultProbe>(&gpuResult, 1u)),
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
  expectGpuSdsmResultAvailable(sdsm, 0u, false);
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
  settings.shadow.maxDistanceFadeFraction = 0.20f;
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
  const float expectedFadeEnd = settings.shadow.maxDistance;
  const float expectedFadeStart =
      expectedFadeEnd - (expectedFadeEnd - frameContext.camera.nearPlane) *
                            settings.shadow.maxDistanceFadeFraction;
  EXPECT_NEAR(shadowFrame.fadeParams.x, expectedFadeStart, 1.0e-4f);
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
  EXPECT_EQ(shadowFrame.cascades[0].textureSampler.w,
            settings.shadow.shadowMapSize);
  EXPECT_EQ(shadowFrame.cascades[3].textureSampler.w,
            settings.shadow.shadowMapSize);
  EXPECT_FLOAT_EQ(shadowFrame.cascades[0].pcssParams.y,
                  settings.shadow.pcssSearchRadiusClampTexels);
  EXPECT_FLOAT_EQ(shadowFrame.cascades[0].pcssParams.z,
                  settings.shadow.pcssFilterRadiusClampTexels);
  EXPECT_GT(shadowFrame.cascades[0].pcssParams.x, 0.0f);
  EXPECT_LE(shadowFrame.cascades[0].pcssParams.x,
            (shadowFrame.cascades[0].splitDepthTexelSize.y -
             shadowFrame.cascades[0].splitDepthTexelSize.x) *
                    2.0f +
                1.0e-4f);
  EXPECT_FLOAT_EQ(shadowFrame.cascades[3].pcssParams.y,
                  settings.shadow.pcssSearchRadiusClampTexels);
  EXPECT_FLOAT_EQ(shadowFrame.cascades[3].pcssParams.z,
                  settings.shadow.pcssFilterRadiusClampTexels);
  EXPECT_GT(shadowFrame.cascades[3].pcssParams.x, 0.0f);
  EXPECT_LE(shadowFrame.cascades[3].pcssParams.x,
            (shadowFrame.cascades[3].splitDepthTexelSize.y -
             shadowFrame.cascades[3].splitDepthTexelSize.x) *
                    2.0f +
                1.0e-4f);
  ASSERT_TRUE(frameContext.sharedResources.shadowDebugFrameData.has_value());
  ASSERT_TRUE(frameContext.sharedResources.selectedShadowLightId.has_value());
  EXPECT_EQ(frameContext.sharedResources.shadowDebugFrameData->cascadeCount,
            4u);
  EXPECT_NEAR(
      frameContext.sharedResources.shadowDebugFrameData->maxDistanceFadeStart,
      expectedFadeStart, 1.0e-4f);
  EXPECT_NEAR(
      frameContext.sharedResources.shadowDebugFrameData->maxDistanceFadeEnd,
      expectedFadeEnd, 1.0e-4f);
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
  materialRequest.debugName =
      "shadow_static_only_reuse_subtexel_motion_material";
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
     ShadowFeatureReusesStaticOnlyShadowCascadesForGuardBandedCameraMotion) {
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

  const std::filesystem::path tempDir = makeTempRendererPath(
      "shadow_static_only_reuse_guard_banded_motion_scene");
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
      .debugName = "shadow_static_only_reuse_guard_banded_motion_triangle",
  });
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();

  MaterialRequest materialRequest{};
  materialRequest.debugName =
      "shadow_static_only_reuse_guard_banded_motion_material";
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
  settings.shadow.shadowMapSize = 8192u;
  settings.shadow.maxDistance = 150.0f;
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

  const glm::vec3 guardBandedMotion = glm::vec3(
      glm::inverse(baseCascade.lightView) *
      glm::vec4(baseCascade.texelWorldSize * 96.0f, 0.0f, 0.0f, 0.0f));
  const RenderFrameContext movedFrame = buildFrame(
      71u, baseEye + guardBandedMotion, baseTarget + guardBandedMotion);

  EXPECT_EQ(movedFrame.metrics.shadow.staticCacheReused, 1u);
  EXPECT_EQ(movedFrame.metrics.shadow.staticOnlyCandidateCount, 4u);
  EXPECT_EQ(movedFrame.metrics.shadow.reusedStaticOnlyCascadeCount, 4u);
  EXPECT_EQ(
      movedFrame.metrics.shadow.staticOnlyReuseMissRasterStateChangedCount, 0u);
  EXPECT_EQ(movedFrame.metrics.shadow.totalDraws, 0u);
  EXPECT_EQ(movedFrame.metrics.shadow.totalIndexCountEstimate, 0u);
  ASSERT_TRUE(movedFrame.sharedResources.shadowDebugFrameData.has_value());
  const ShadowCascadeDebugFrameData &movedCascade =
      movedFrame.sharedResources.shadowDebugFrameData->cascades[0];
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < 4; ++r) {
      EXPECT_NEAR(movedCascade.lightViewProj[c][r],
                  baseCascade.lightViewProj[c][r], 1.0e-6f);
    }
  }

  const glm::vec3 fastStep = glm::vec3(
      glm::inverse(baseCascade.lightView) *
      glm::vec4(baseCascade.texelWorldSize * 512.0f, 0.0f, 0.0f, 0.0f));
  const RenderFrameContext fastFrame =
      buildFrame(72u, baseEye + guardBandedMotion + fastStep,
                 baseTarget + guardBandedMotion + fastStep);
  EXPECT_EQ(fastFrame.metrics.shadow.staticCacheReused, 1u);
  EXPECT_EQ(fastFrame.metrics.shadow.staticOnlyCandidateCount, 4u);
  EXPECT_LT(fastFrame.metrics.shadow.reusedStaticOnlyCascadeCount, 4u);
  EXPECT_GT(fastFrame.metrics.shadow.staticOnlyReuseMissAdaptiveRefreshCount,
            0u);
  EXPECT_EQ(fastFrame.metrics.shadow.staticOnlyReuseMissRasterStateChangedCount,
            0u);
  ASSERT_TRUE(fastFrame.sharedResources.shadowDebugFrameData.has_value());
  const ShadowCascadeDebugFrameData &fastCascade =
      fastFrame.sharedResources.shadowDebugFrameData->cascades[0];
  EXPECT_EQ(
      fastCascade.staticOnlyReuseStatus,
      ShadowCascadeDebugFrameData::StaticOnlyReuseStatus::AdaptiveRefresh);
  EXPECT_TRUE(fastCascade.staticOnlyReuseAdaptiveRefresh);
  EXPECT_LE(baseCascade.texelWorldSize, fastCascade.texelWorldSize * 1.15f);

  const RenderFrameContext fastNextFrame =
      buildFrame(73u, baseEye + guardBandedMotion + fastStep * 2.0f,
                 baseTarget + guardBandedMotion + fastStep * 2.0f);
  EXPECT_EQ(fastNextFrame.metrics.shadow.staticCacheReused, 1u);
  EXPECT_EQ(fastNextFrame.metrics.shadow.staticOnlyCandidateCount, 4u);
  EXPECT_LT(fastNextFrame.metrics.shadow.reusedStaticOnlyCascadeCount, 4u);
  EXPECT_GT(
      fastNextFrame.metrics.shadow.staticOnlyReuseMissAdaptiveRefreshCount, 0u);
  EXPECT_EQ(
      fastNextFrame.metrics.shadow.staticOnlyReuseMissRasterStateChangedCount,
      0u);
  EXPECT_GT(fastNextFrame.metrics.shadow.totalDraws, 0u);
  EXPECT_GT(fastNextFrame.metrics.shadow.totalIndexCountEstimate, 0u);
  ASSERT_TRUE(fastNextFrame.sharedResources.shadowDebugFrameData.has_value());
  const ShadowCascadeDebugFrameData &fastNextCascade =
      fastNextFrame.sharedResources.shadowDebugFrameData->cascades[0];
  EXPECT_LE(fastNextCascade.texelWorldSize, fastCascade.texelWorldSize * 1.05f);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureReusesStaticOnlyShadowCascadesAcrossSplitDepthDrift) {
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
      makeTempRendererPath("shadow_static_only_reuse_split_drift_scene");
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
      .debugName = "shadow_static_only_reuse_split_drift_triangle",
  });
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();

  MaterialRequest materialRequest{};
  materialRequest.debugName = "shadow_static_only_reuse_split_drift_material";
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
  settings.shadow.shadowMapSize = 2048u;
  settings.shadow.maxDistance = 150.0f;
  settings.shadow.debug.enableCascadeCasterCulling = false;

  const glm::vec3 eye(0.0f, 1.5f, 4.0f);
  const glm::vec3 target(0.0f, 0.5f, 0.0f);
  const glm::vec3 up(0.0f, 1.0f, 0.0f);
  const auto buildFrame = [&](uint64_t frameIndex) {
    RenderFrameContext frameContext{};
    frameContext.frameIndex = frameIndex;
    frameContext.scene = &scene;
    frameContext.resources = &renderer.resources();
    frameContext.settings = &settings;
    frameContext.camera.view = glm::lookAt(eye, target, up);
    frameContext.camera.proj =
        glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 180.0f);
    frameContext.camera.cameraPos = glm::vec4(eye, 1.0f);
    frameContext.camera.aspectRatio = 1.0f;
    frameContext.camera.projectionType = ProjectionType::Perspective;
    frameContext.camera.nearPlane = 0.1f;
    frameContext.camera.farPlane = 180.0f;
    frameContext.camera.fovYRadians = glm::radians(60.0f);

    RenderGraphBuilder graph(&memory);
    graph.beginFrame(frameContext.frameIndex);
    auto buildResult =
        pipeline.buildRenderGraph(frameContext, renderer.resources(), graph);
    EXPECT_FALSE(buildResult.hasError()) << buildResult.error();
    EXPECT_TRUE(buildResult.value());
    return frameContext;
  };

  const RenderFrameContext firstFrame = buildFrame(80u);
  ASSERT_TRUE(firstFrame.sharedResources.shadowDebugFrameData.has_value());
  const float firstCascadeFar =
      firstFrame.sharedResources.shadowDebugFrameData->cascades[0].splitFar;

  settings.shadow.maxDistance = 149.0f;
  const RenderFrameContext splitDriftFrame = buildFrame(81u);

  ASSERT_TRUE(splitDriftFrame.sharedResources.shadowDebugFrameData.has_value());
  const ShadowDebugFrameData &debug =
      *splitDriftFrame.sharedResources.shadowDebugFrameData;
  EXPECT_EQ(splitDriftFrame.metrics.shadow.staticCacheReused, 1u);
  EXPECT_EQ(splitDriftFrame.metrics.shadow.staticOnlyCandidateCount, 4u);
  EXPECT_EQ(splitDriftFrame.metrics.shadow.reusedStaticOnlyCascadeCount, 4u);
  EXPECT_EQ(
      splitDriftFrame.metrics.shadow.staticOnlyReuseMissRasterStateChangedCount,
      0u);
  EXPECT_EQ(splitDriftFrame.metrics.shadow.totalDraws, 0u);
  EXPECT_LT(debug.cascades[0].splitFar, firstCascadeFar);
  EXPECT_NEAR(debug.cascades[0].splitFar, debug.sdsm.effectiveSplitDepths[1],
              1.0e-5f);
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
      buildFrame(71u, glm::vec3(48.0f, 1.5f, 4.0f));
  EXPECT_EQ(movedFrame.metrics.shadow.staticCacheReused, 1u);
  EXPECT_EQ(movedFrame.metrics.shadow.staticOnlyCandidateCount, 4u);
  EXPECT_EQ(movedFrame.metrics.shadow.reusedStaticOnlyCascadeCount, 0u);
  EXPECT_EQ(
      movedFrame.metrics.shadow.staticOnlyReuseMissRasterStateChangedCount +
          movedFrame.metrics.shadow.staticOnlyReuseMissAdaptiveRefreshCount,
      4u);
  EXPECT_GT(
      movedFrame.metrics.shadow.staticOnlyReuseMissRasterStateChangedCount, 0u);
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
  settings.shadow.normalBias = 0.0f;
  settings.shadow.maxDistanceFadeFraction = 0.10f;

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

  settings.shadow.normalBias = 1.25f;
  settings.shadow.maxDistanceFadeFraction = 0.35f;
  const auto [thirdFrame, thirdCompiled] = buildFrame(102u);
  EXPECT_EQ(thirdFrame.metrics.shadow.staticCacheReused, 1u);
  EXPECT_EQ(thirdFrame.metrics.shadow.reusedStaticOnlyCascadeCount, 4u);
  EXPECT_EQ(thirdFrame.metrics.shadow.totalDraws, 0u);
  uint32_t thirdShadowPassCount = 0u;
  for (const RenderPass &pass : thirdCompiled.orderedPasses) {
    if (!std::string_view(pass.debugLabel)
             .starts_with("ShadowDepthPass.Cascade")) {
      continue;
    }
    ++thirdShadowPassCount;
    EXPECT_EQ(pass.depth.loadOp, LoadOp::Load);
    EXPECT_TRUE(pass.draws.empty());
  }
  EXPECT_EQ(thirdShadowPassCount, 4u);
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
  auto opaquePipelineIt =
      std::find(gpu.createdRenderPipelineNames.begin(),
                gpu.createdRenderPipelineNames.end(), "shadow_depth_opaque");
  ASSERT_NE(opaquePipelineIt, gpu.createdRenderPipelineNames.end());
  const size_t opaquePipelineDescIndex = static_cast<size_t>(
      std::distance(gpu.createdRenderPipelineNames.begin(), opaquePipelineIt));
  const size_t alphaPipelineDescIndex = static_cast<size_t>(
      std::distance(gpu.createdRenderPipelineNames.begin(), alphaPipelineIt));
  ASSERT_LT(opaquePipelineDescIndex, gpu.createdRenderPipelineDescs.size());
  ASSERT_LT(alphaPipelineDescIndex, gpu.createdRenderPipelineDescs.size());
  EXPECT_NE(gpu.createdRenderPipelineDescs[opaquePipelineDescIndex]
                .vertexShader.index,
            gpu.createdRenderPipelineDescs[alphaPipelineDescIndex]
                .vertexShader.index);
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
  EXPECT_FLOAT_EQ(frameContext.metrics.shadow.minCascadeTexelWorldSize,
                  cascade.texelWorldSize);
  EXPECT_FLOAT_EQ(frameContext.metrics.shadow.averageCascadeTexelWorldSize,
                  cascade.texelWorldSize);
  EXPECT_FLOAT_EQ(frameContext.metrics.shadow.maxCascadeTexelWorldSize,
                  cascade.texelWorldSize);
  EXPECT_FLOAT_EQ(frameContext.metrics.shadow.farCascadeTexelWorldSize,
                  cascade.texelWorldSize);
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
