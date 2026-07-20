#include "tests_pch.h"

#include "render_graph_test_support.h"

#include <gtest/gtest.h>

#include "nuri/core/log.h"
#include "nuri/core/runtime_config.h"
#include "nuri/gfx/frame/presentation_aa_plan.h"
#include "nuri/gfx/renderers/opaque_renderer.h"
#define private public
#include "nuri/gfx/renderers/shadow_renderer.h"
#undef private
#include "nuri/gfx/pipeline/features/skybox_feature.h"
#include "nuri/gfx/pipeline/providers/frame_composition_provider.h"
#include "nuri/gfx/pipeline/providers/material_table_gpu_provider.h"
#include "nuri/gfx/pipeline/providers/scene_lighting_provider.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/gfx/renderer.h"
#include "nuri/gfx/renderers/detail/opaque_lod_selection.h"
#include "nuri/gfx/renderers/detail/opaque_meshlet_routing.h"
#include "nuri/gfx/renderers/detail/shadow_math.h"
#include "nuri/gfx/renderers/shadow_renderer.h"
#include "nuri/gfx/visibility/visibility.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/scene/light.h"
#include "nuri/scene/render_scene.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
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

namespace {

using namespace nuri;
using namespace nuri::test_support;

TEST(RenderFrameMetricsTests, ResolvesPostCullGpuWorkWhenReadbackIsAvailable) {
  RenderFrameMetrics metrics{};
  metrics.opaque.totalInstances = 2909u;
  metrics.opaque.visibleInstances = 2567u;
  metrics.opaque.indirectCommands = 506u;
  metrics.opaque.meshletTaskGroups = 2382u;
  metrics.visibility.gpuMainCandidates = 2567u;
  metrics.visibility.gpuMainVisibleCandidates = 73u;
  metrics.visibility.gpuMainReadbackAvailable = 1u;
  metrics.visibility.gpuIndirectDrawUsed = 1u;
  metrics.visibility.gpuIndirectDrawReadbackCommands = 506u;
  metrics.visibility.gpuIndirectDrawReadbackVisible = 31u;
  metrics.visibility.meshletCandidates = 66312u;
  metrics.visibility.meshletReadbackAvailable = 1u;
  metrics.visibility.meshletEmitted = 2582u;
  metrics.visibility.meshletTaskGroupsExecuted = 1697u;

  const ResolvedGeometryWorkMetrics resolved =
      resolveGeometryWorkMetrics(metrics);
  EXPECT_EQ(resolved.instanceCandidates, 2567u);
  EXPECT_EQ(resolved.visibleInstances, 73u);
  EXPECT_EQ(resolved.indirectCommands, 506u);
  EXPECT_EQ(resolved.visibleIndirectCommands, 31u);
  EXPECT_EQ(resolved.meshletCandidates, 66312u);
  EXPECT_EQ(resolved.emittedMeshlets, 2582u);
  EXPECT_EQ(resolved.executedMeshletTaskGroups, 1697u);
  EXPECT_TRUE(resolved.mainReadbackAvailable);
  EXPECT_TRUE(resolved.indirectReadbackAvailable);
  EXPECT_TRUE(resolved.meshletReadbackAvailable);
}

TEST(RenderFrameMetricsTests, FallsBackToSubmittedWorkWhileReadbackIsPending) {
  RenderFrameMetrics metrics{};
  metrics.opaque.totalInstances = 2909u;
  metrics.opaque.visibleInstances = 2567u;
  metrics.opaque.indirectCommands = 506u;
  metrics.opaque.meshletTaskGroups = 2382u;
  metrics.visibility.meshletCandidates = 66312u;

  const ResolvedGeometryWorkMetrics resolved =
      resolveGeometryWorkMetrics(metrics);
  EXPECT_EQ(resolved.instanceCandidates, 2909u);
  EXPECT_EQ(resolved.visibleInstances, 2567u);
  EXPECT_EQ(resolved.indirectCommands, 506u);
  EXPECT_EQ(resolved.visibleIndirectCommands, 506u);
  EXPECT_EQ(resolved.emittedMeshlets, 66312u);
  EXPECT_EQ(resolved.executedMeshletTaskGroups, 2382u);
  EXPECT_FALSE(resolved.mainReadbackAvailable);
  EXPECT_FALSE(resolved.indirectReadbackAvailable);
  EXPECT_FALSE(resolved.meshletReadbackAvailable);
}

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

std::filesystem::path makeTempRendererPath(std::string_view stem) {
  const auto tick =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() /
         ("nuri_" + std::string(stem) + "_" + std::to_string(tick));
}

RuntimeOpaqueShaderConfig makeOpaqueConfig(const std::filesystem::path &root) {
  const std::filesystem::path shaders = root / "assets" / "shaders";
  return RuntimeOpaqueShaderConfig{
      .shaderBasePath = shaders,
      .meshVertex = shaders / "main.vert",
      .meshFragment = shaders / "main.frag",
      .meshletTask = shaders / "opaque_meshlet.task.glsl",
      .meshletMesh = shaders / "opaque_meshlet.mesh.glsl",
      .meshletFragment = shaders / "opaque_meshlet.frag",
      .meshletDepthFragment = shaders / "opaque_meshlet_depth.frag",
      .meshletDepthAlphaFragment = shaders / "opaque_meshlet_depth_alpha.frag",
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

MeshData makeShadowTriangleMesh(std::pmr::memory_resource *memory) {
  MeshData mesh(memory);
  mesh.vertices.resize(3u);
  mesh.vertices[0].position = glm::vec3(-0.5f, -0.5f, 0.0f);
  mesh.vertices[0].normal = glm::vec3(0.0f, 0.0f, 1.0f);
  mesh.vertices[1].position = glm::vec3(0.5f, -0.5f, 0.0f);
  mesh.vertices[1].normal = glm::vec3(0.0f, 0.0f, 1.0f);
  mesh.vertices[2].position = glm::vec3(0.0f, 0.5f, 0.0f);
  mesh.vertices[2].normal = glm::vec3(0.0f, 0.0f, 1.0f);
  mesh.indices = {0u, 1u, 2u};

  Submesh submesh{};
  submesh.vertexOffset = 0u;
  submesh.vertexCount = 3u;
  submesh.indexOffset = 0u;
  submesh.indexCount = 3u;
  submesh.bounds =
      BoundingBox(glm::vec3(-0.5f, -0.5f, 0.0f), glm::vec3(0.5f, 0.5f, 0.0f));
  submesh.authoredScale = glm::vec3(1.0f);
  submesh.lodCount = 1u;
  submesh.lods[0] = SubmeshLod{.indexOffset = 0u, .indexCount = 3u};
  mesh.submeshes.push_back(submesh);
  return mesh;
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

void expectGpuSdsmResultAvailable(const ShadowSdsmDebugFrameData &sdsm,
                                  uint64_t sourceFrameIndex) {
  EXPECT_GT(sdsm.gpuResultRingSlotCount, 0u);
  EXPECT_NE(sdsm.gpuResultSelectedSlot, std::numeric_limits<uint32_t>::max());
  EXPECT_TRUE(sdsm.gpuReductionResultAvailable);
  EXPECT_EQ(sdsm.gpuResultSourceFrameIndex, sourceFrameIndex);
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

  Result<SubmittedGraphicsFrame, std::string> submitRecordedGraphicsFrame(
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
    ++geometryMutationVersion_;
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
      ++geometryMutationVersion_;
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

  uint64_t geometryMutationVersion() const override {
    return geometryMutationVersion_;
  }

  Result<GeometryAllocationView, std::string> relocateFirstGeometry() {
    if (allocations_.empty() || !nuri::isValid(allocations_.front().handle)) {
      return Result<GeometryAllocationView, std::string>::makeError(
          "no live geometry allocation");
    }

    Allocation &allocation = allocations_.front();
    auto vertexBufferResult = createBufferImpl(BufferDesc{
        .usage = BufferUsage::Vertex | BufferUsage::Storage,
        .storage = Storage::Device,
        .size = allocation.view.vertexByteSize,
    });
    if (vertexBufferResult.hasError()) {
      return Result<GeometryAllocationView, std::string>::makeError(
          vertexBufferResult.error());
    }
    auto indexBufferResult = createBufferImpl(BufferDesc{
        .usage = BufferUsage::Index,
        .storage = Storage::Device,
        .size = allocation.view.indexByteSize,
    });
    if (indexBufferResult.hasError()) {
      destroyBuffer(vertexBufferResult.value());
      return Result<GeometryAllocationView, std::string>::makeError(
          indexBufferResult.error());
    }

    const BufferHandle previousVertexBuffer = allocation.view.vertexBuffer;
    const BufferHandle previousIndexBuffer = allocation.view.indexBuffer;
    allocation.view.vertexBuffer = vertexBufferResult.value();
    allocation.view.indexBuffer = indexBufferResult.value();
    destroyBuffer(previousVertexBuffer);
    destroyBuffer(previousIndexBuffer);
    ++geometryMutationVersion_;
    return Result<GeometryAllocationView, std::string>::makeResult(
        allocation.view);
  }

  uint32_t forcedIndexStrideBytes = 0u;

private:
  struct Allocation {
    GeometryAllocationHandle handle{};
    GeometryAllocationView view{};
  };

  std::vector<Allocation> allocations_{};
  uint64_t geometryMutationVersion_ = 1u;
};

class FakeMeshletPipelineGpuDevice final : public FakeFullscreenGpuDevice {
public:
  bool supportsFeature(GPUFeature feature) const override {
    return feature == GPUFeature::Meshlets;
  }

  Result<MeshletPipelineHandle, std::string>
  createMeshletPipeline(const MeshletPipelineDesc &desc,
                        std::string_view debugName) override {
    createdMeshletPipelineDescs.push_back(desc);
    createdMeshletPipelineNames.emplace_back(debugName);
    return Result<MeshletPipelineHandle, std::string>::makeResult(
        MeshletPipelineHandle{.index = nextMeshletPipelineIndex_++,
                              .generation = 1u});
  }

  std::vector<MeshletPipelineDesc> createdMeshletPipelineDescs{};
  std::vector<std::string> createdMeshletPipelineNames{};

private:
  uint32_t nextMeshletPipelineIndex_ = 1u;
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

struct HDRExposurePushConstantsProbe {
  uint32_t sourceTexId = 0u;
  uint32_t previousExposureTexId = 0u;
  uint32_t sourceSamplerId = 0u;
  uint32_t flags = 0u;
  float targetGray = 0.0f;
  float speed = 0.0f;
  float minEv = 0.0f;
  float maxEv = 0.0f;
  float deltaSeconds = 0.0f;
  float reserved0 = 0.0f;
  float reserved1 = 0.0f;
  float reserved2 = 0.0f;
};
static_assert(sizeof(HDRExposurePushConstantsProbe) <= 128u);

struct TAAResolvePushConstantsProbe {
  uint32_t currentTexId = 0u;
  uint32_t historyTexId = 0u;
  uint32_t depthTexId = 0u;
  uint32_t previousDepthTexId = 0u;
  uint32_t velocityTexId = 0u;
  uint32_t previousVelocityTexId = 0u;
  uint32_t reactiveMaskTexId = 0u;
  uint32_t linearSamplerId = 0u;
  uint32_t pointSamplerId = 0u;
  uint32_t flags = 0u;
  uint32_t mode = 0u;
  uint32_t currentWeightBits = 0u;
  uint32_t inverseWidthBits = 0u;
  uint32_t inverseHeightBits = 0u;
  uint32_t depthThresholdBits = 0u;
  uint32_t velocityThresholdBits = 0u;
  uint32_t velocityBlendScaleBits = 0u;
  uint32_t disocclusionWeightBits = 0u;
  uint32_t clampAttenuationBits = 0u;
  uint32_t varianceGammaBits = 0u;
  uint32_t hdrWeightStrengthBits = 0u;
  uint32_t reactiveCurrentWeightBits = 0u;
  uint32_t reactiveStrengthBits = 0u;
  uint32_t velocityDilationDepthThresholdBits = 0u;
  uint32_t clampMode = 0u;
  uint32_t hdrWeightingMode = 0u;
  uint32_t velocityDilationMode = 0u;
  uint32_t motionCurrentWeightBits = 0u;
  uint32_t clampCurrentWeightBits = 0u;
  uint32_t historyFilterMode = 0u;
  uint32_t previousRawJitterDeltaUvXBits = 0u;
  uint32_t previousRawJitterDeltaUvYBits = 0u;
};
static_assert(sizeof(TAAResolvePushConstantsProbe) <= 128u);

struct OpaquePushConstantsProbe {
  uint64_t frameDataAddress = 0u;
  uint64_t vertexBufferAddress = 0u;
  uint64_t vertexDecodeBufferAddress = 0u;
  uint64_t instanceMatricesAddress = 0u;
  uint64_t previousInstanceMatricesAddress = 0u;
  uint64_t instanceRemapAddress = 0u;
  uint64_t instanceCentersPhaseAddress = 0u;
  uint64_t instanceBaseMatricesAddress = 0u;
  uint64_t velocityInstanceFlagsAddress = 0u;
  uint64_t velocityFrameDataAddress = 0u;
  uint32_t instanceCount = 0u;
  uint32_t materialIndex = 0u;
  uint32_t vertexDecodeIndex = 0u;
  uint32_t packedVertexFormat = 0u;
  float timeSeconds = 0.0f;
  float tessNearDistance = 1.0f;
  float tessFarDistance = 8.0f;
  float tessMinFactor = 1.0f;
  float tessMaxFactor = 6.0f;
  uint32_t debugVisualizationMode = 0u;
  uint32_t shadowCascadeIndex = 0u;
};
static_assert(sizeof(OpaquePushConstantsProbe) == 128u);

struct alignas(16) DepthMotionVectorPushConstantsProbe {
  uint32_t depthTexId = 0u;
  uint32_t pointSamplerId = 0u;
  uint32_t currentJitterUvXBits = 0u;
  uint32_t currentJitterUvYBits = 0u;
  glm::mat4 previousFromCurrentJitteredClip{1.0f};
};
static_assert(sizeof(DepthMotionVectorPushConstantsProbe) <= 128u);

constexpr uint32_t kPresentFlagManualSrgbEncode = 1u << 0u;
constexpr uint32_t kPresentFlagPrimaryUseAgx = 1u << 1u;
constexpr uint32_t kPresentFlagCompareEnabled = 1u << 2u;
constexpr uint32_t kPresentFlagGrayCardDebug = 1u << 3u;
constexpr uint32_t kPresentFlagAcesLutAvailable = 1u << 4u;
constexpr uint32_t kPresentFlagAgxLutAvailable = 1u << 5u;
constexpr uint32_t kPresentFlagPrimaryLegacyFallback = 1u << 6u;
constexpr uint32_t kPresentFlagCompareLegacyFallback = 1u << 7u;
constexpr uint32_t kHDRPostFlagReducedLuminanceSourceProbe = 1u << 3u;
constexpr float kTransmissionJitterDepthBiasConstantProbe = -8.0f;
constexpr uint32_t kTaaResolveFlagSharpenProbe = 1u << 11u;
constexpr uint32_t kTaaResolveFlagStaticFrameProbe = 1u << 12u;
constexpr uint32_t kTaaResolveModeResolveProbe = 0u;
constexpr uint32_t kTaaResolveModeCopyHistoryToSceneProbe = 21u;
constexpr uint32_t kTaaResolveModeStabilityOwnershipProbe = 25u;
constexpr uint32_t kTaaResolveModePatchProbeValue = 26u;
constexpr uint32_t kTaaResolveModeMotionFilterProbe = 27u;

void writeTextFile(const std::filesystem::path &path, std::string_view text) {
  std::ofstream out(path, std::ios::binary);
  ASSERT_TRUE(out.is_open());
  out.write(text.data(), static_cast<std::streamsize>(text.size()));
  ASSERT_TRUE(out.good());
}

class SinglePassFeaturePass final {
public:
  struct Config {
    std::string_view passName{};
    TextureHandle outputTexture{};
    bool useExplicitFrameOutput = false;
    uint32_t debugColor = 0xffffffffu;
  };

  explicit SinglePassFeaturePass(Config config) : config_(config) {}

  bool isEnabled(const FrameBuildContext &) const { return true; }

  Result<bool, std::string> prepare(FrameBuildContext &) {
    return Result<bool, std::string>::makeResult(true);
  }

  Result<bool, std::string> build(FrameBuildContext &ctx) {
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

void addTestStage(RenderPipeline &pipeline, std::string_view featureName,
                  SinglePassFeaturePass::Config config) {
  const std::string_view passName = config.passName;
  pipeline.addStage(std::make_unique<SinglePassFeaturePass>(config),
                    featureName, passName);
}

void addBaseImplicitOutput(RenderPipeline &pipeline) {
  addTestStage(
      pipeline, "BaseImplicitOutputFeature",
      {.passName = "Base Implicit Output Pass", .debugColor = 0xff778899u});
}

TEST(OpaqueMeshletRoutingTest, OpportunisticHybridUsesResolvedLodBoundary) {
  using detail::shouldUseMeshletsForOpaqueBatch;

  EXPECT_FALSE(shouldUseMeshletsForOpaqueBatch(
      MeshletRenderMode::Opportunistic, true, false, false, true, 8u, 8u));
  EXPECT_TRUE(shouldUseMeshletsForOpaqueBatch(
      MeshletRenderMode::Opportunistic, true, false, false, true, 9u, 8u));
  EXPECT_TRUE(shouldUseMeshletsForOpaqueBatch(
      MeshletRenderMode::Opportunistic, true, false, false, true, 1u, 0u));
}

TEST(OpaqueMeshletRoutingTest, RequiredModePreservesAllMeshletSemantics) {
  using detail::shouldUseMeshletsForOpaqueBatch;

  EXPECT_TRUE(shouldUseMeshletsForOpaqueBatch(MeshletRenderMode::Required, true,
                                              true, true, true, 1u, 64u));
  EXPECT_TRUE(shouldUseMeshletsForOpaqueBatch(
      MeshletRenderMode::Required, false, true, true, true, 1u, 64u));
  EXPECT_FALSE(shouldUseMeshletsForOpaqueBatch(
      MeshletRenderMode::Required, true, false, false, false, 1u, 64u));
}

TEST(OpaqueMeshletRoutingTest, IneligibleOpportunisticFramesStayAllMeshlet) {
  using detail::shouldUseMeshletsForOpaqueBatch;

  EXPECT_TRUE(shouldUseMeshletsForOpaqueBatch(
      MeshletRenderMode::Opportunistic, false, true, true, true, 1u, 64u));
  EXPECT_FALSE(shouldUseMeshletsForOpaqueBatch(
      MeshletRenderMode::Disabled, true, false, false, true, 128u, 64u));
  EXPECT_FALSE(shouldUseMeshletsForOpaqueBatch(
      MeshletRenderMode::Opportunistic, true, false, false, true, 0u, 64u));
}

TEST(OpaqueMeshletRoutingTest,
     OpportunisticHybridRoutesCoverageSensitiveBatchesIndexed) {
  using detail::shouldUseMeshletsForOpaqueBatch;

  EXPECT_FALSE(shouldUseMeshletsForOpaqueBatch(
      MeshletRenderMode::Opportunistic, true, true, false, true, 128u, 64u));
  EXPECT_FALSE(shouldUseMeshletsForOpaqueBatch(
      MeshletRenderMode::Opportunistic, true, false, true, true, 128u, 64u));
}

TEST(OpaqueLodSelectionTest, ChoosesCoarsestLodInsidePixelBudget) {
  const std::array<float, 3> worldErrors{0.0f, 0.5f, 1.5f};
  const detail::OpaqueLodProjection projection{.pixelScaleY = 2.0f,
                                               .nearestDepth = 2.0f};

  EXPECT_EQ(detail::selectOpaqueLod(worldErrors, 1.0f, 0.2f, projection), 1u);
}

TEST(OpaqueLodSelectionTest, HysteresisPreventsBoundaryOscillation) {
  const std::array<float, 2> worldErrors{0.0f, 1.0f};
  detail::OpaqueLodProjection projection{.pixelScaleY = 1.0f,
                                         .nearestDepth = 1.0f};

  EXPECT_EQ(detail::selectOpaqueLod(worldErrors, 1.0f, 0.2f, projection, 0u),
            0u);
  EXPECT_EQ(detail::selectOpaqueLod(worldErrors, 1.0f, 0.2f, projection, 1u),
            1u);

  projection.nearestDepth = 0.8f;
  EXPECT_EQ(detail::selectOpaqueLod(worldErrors, 1.0f, 0.2f, projection, 1u),
            0u);
  projection.nearestDepth = 1.25f;
  EXPECT_EQ(detail::selectOpaqueLod(worldErrors, 1.0f, 0.2f, projection, 0u),
            1u);
}

TEST(OpaqueLodSelectionTest, OrthographicProjectionIgnoresDepth) {
  const std::array<float, 2> worldErrors{0.0f, 0.25f};
  const detail::OpaqueLodProjection projection{
      .pixelScaleY = 4.0f, .nearestDepth = 0.01f, .orthographic = true};

  EXPECT_EQ(detail::selectOpaqueLod(worldErrors, 1.0f, 0.0f, projection), 1u);
}

TEST(OpaqueMeshletRoutingTest, AutomaticLodUsesOnlyStableGeneratedLevels) {
  using detail::resolveOpaqueAutomaticLod;

  EXPECT_EQ(resolveOpaqueAutomaticLod(3u, true, true), 0u);
  EXPECT_EQ(resolveOpaqueAutomaticLod(3u, false, true), 1u);
  EXPECT_EQ(resolveOpaqueAutomaticLod(2u, true, false), 2u);
  EXPECT_EQ(resolveOpaqueAutomaticLod(3u, false, false), 3u);
}

TEST(RenderGraphRendererTest,
     RendererKeepsBaseImplicitPassUnderSuppressionWithExplicitOutputRoot) {
  EnvVarGuard envGuard("NURI_RENDER_GRAPH_SUPPRESS_INFERRED_SIDE_EFFECTS", "1");

  std::array<std::byte, 64 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeRendererGPUDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);

  addBaseImplicitOutput(pipeline);
  const TextureHandle explicitOutputTexture{.index = 501u, .generation = 1u};
  addTestStage(pipeline, "ExplicitFrameOutputFeature",
               {.passName = "Layer Explicit Output Pass",
                .outputTexture = explicitOutputTexture,
                .useExplicitFrameOutput = true});

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

TEST(RenderGraphRendererTest, RendererCapturesGraphTelemetryOnlyOnRequest) {
  EnvVarGuard dumpEnv("NURI_RENDER_GRAPH_DUMP", "0");
  EnvVarGuard suppressionEnv("NURI_RENDER_GRAPH_SUPPRESS_INFERRED_SIDE_EFFECTS",
                             "0");
  std::pmr::unsynchronized_pool_resource memory;
  FakeRendererGPUDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  addBaseImplicitOutput(pipeline);

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 1u;
  frameContext.resources = &renderer.resources();
  auto renderResult = renderer.render(pipeline, frameContext);
  ASSERT_FALSE(renderResult.hasError());
  EXPECT_EQ(renderer.renderGraphTelemetry().latestSnapshot(), nullptr);

  renderer.renderGraphTelemetry().requestCapture(
      RenderGraphTelemetryLevel::PassTimings);
  frameContext.frameIndex = 2u;
  renderResult = renderer.render(pipeline, frameContext);
  ASSERT_FALSE(renderResult.hasError());
  const RenderGraphTelemetrySnapshot *snapshot =
      renderer.renderGraphTelemetry().latestSnapshot();
  ASSERT_NE(snapshot, nullptr);
  EXPECT_EQ(snapshot->summary.frameIndex, 2u);
  EXPECT_EQ(snapshot->summary.passCount, 1u);
  EXPECT_EQ(snapshot->summary.passTimingCount, 1u);
}

TEST(RenderGraphRendererTest, RendererPublishesOneGpuTimingSnapshotPerFrame) {
  EnvVarGuard dumpEnv("NURI_RENDER_GRAPH_DUMP", "0");
  std::pmr::unsynchronized_pool_resource memory;
  FakeRendererGPUDevice gpu;
  gpu.latestCompletedGpuTimingReport.opaqueSourceFrameIndex = 41u;
  gpu.latestCompletedGpuTimingReport.opaqueTimeMs = 3.25f;
  gpu.latestCompletedGpuTimingReport.availableScopeMask =
      gpuTimingScopeToBit(GpuTimingScope::Opaque);
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  addBaseImplicitOutput(pipeline);

  RenderFrameContext frameContext{};
  frameContext.frameIndex = 42u;
  frameContext.resources = &renderer.resources();
  auto renderResult = renderer.render(pipeline, frameContext);

  ASSERT_FALSE(renderResult.hasError());
  EXPECT_EQ(gpu.latestGpuTimingReportFetchCount, 1u);
  EXPECT_FLOAT_EQ(frameContext.gpuTiming.opaqueTimeMs, 3.25f);
  EXPECT_EQ(frameContext.gpuTiming.opaqueSourceFrameIndex, 41u);
  EXPECT_TRUE(
      hasGpuTimingScope(frameContext.gpuTiming, GpuTimingScope::Opaque));
}

TEST(RenderGraphRendererTest,
     RendererResolvesOneImmutableSettingsSnapshotPerFrame) {
  EnvVarGuard dumpEnv("NURI_RENDER_GRAPH_DUMP", "0");
  std::pmr::unsynchronized_pool_resource memory;
  FakeRendererGPUDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  addBaseImplicitOutput(pipeline);

  RenderSettings authored{};
  authored.hdrPostProcess.bloomStrength =
      std::numeric_limits<float>::quiet_NaN();
  authored.transmission.taaJitterPrefilterMaxLod = 99.0f;
  authored.antiAliasing.temporalProvider =
      static_cast<TemporalReconstructionProvider>(0xffu);

  RenderFrameContext frameContext{};
  frameContext.settings = &authored;
  frameContext.resources = &renderer.resources();
  auto renderResult = renderer.render(pipeline, frameContext);

  ASSERT_FALSE(renderResult.hasError());
  ASSERT_TRUE(frameContext.settingsResolved);
  const RenderSettings &resolved = renderSettingsOrDefault(frameContext);
  EXPECT_FLOAT_EQ(resolved.hdrPostProcess.bloomStrength,
                  kDefaultHDRBloomStrength);
  EXPECT_FLOAT_EQ(resolved.transmission.taaJitterPrefilterMaxLod, 2.0f);
  EXPECT_EQ(resolved.antiAliasing.temporalProvider,
            TemporalReconstructionProvider::Legacy);
  EXPECT_TRUE(std::isnan(authored.hdrPostProcess.bloomStrength));
  EXPECT_FLOAT_EQ(authored.transmission.taaJitterPrefilterMaxLod, 99.0f);
}

TEST(RenderGraphRendererTest, SanitizeHDRPostProcessSettingsClampsInputs) {
  RenderSettings::HDRPostProcessSettings hdr{};
  hdr.bloomStrength = std::numeric_limits<float>::quiet_NaN();
  hdr.bloomThreshold = -1.0f;
  hdr.bloomSoftKnee = 10.0f;
  hdr.bloomMaxMipCount = 0u;
  hdr.adaptationTargetGray = 0.0f;
  hdr.adaptationSpeed = std::numeric_limits<float>::infinity();
  hdr.adaptationMinEv = 12.0f;
  hdr.adaptationMaxEv = -4.0f;
  hdr.debugView = static_cast<HDRPostProcessDebugView>(255u);

  sanitizeHDRPostProcessSettings(hdr);

  EXPECT_FLOAT_EQ(hdr.bloomStrength, kDefaultHDRBloomStrength);
  EXPECT_FLOAT_EQ(hdr.bloomThreshold, 0.0f);
  EXPECT_FLOAT_EQ(hdr.bloomSoftKnee, 4.0f);
  EXPECT_EQ(hdr.bloomMaxMipCount, 1u);
  EXPECT_FLOAT_EQ(hdr.adaptationTargetGray, 0.001f);
  EXPECT_FLOAT_EQ(hdr.adaptationSpeed, kDefaultHDRAdaptationSpeed);
  EXPECT_FLOAT_EQ(hdr.adaptationMinEv, -4.0f);
  EXPECT_FLOAT_EQ(hdr.adaptationMaxEv, 12.0f);
  EXPECT_EQ(hdr.debugView, HDRPostProcessDebugView::None);
}

TEST(RenderGraphRendererTest, TemporalAAQualityPresetDrivesEffectiveSettings) {
  RenderSettings::AntiAliasingSettings aa{};
  aa.mode = AntiAliasingMode::TAA;
  aa.qualityPreset = TemporalAAQualityPreset::Quality;
  aa.debug.jitterEnabled = true;
  aa.debug.taaCurrentFrameWeight = 0.77f;
  aa.debug.taaMotionCurrentWeight = 0.88f;
  aa.debug.taaVarianceGamma = 0.25f;
  aa.debug.spatialPostTaaCleanup = false;
  aa.debug.transparentPostTaaSpatialCleanup = false;

  const RenderSettings::AntiAliasingDebugSettings effective =
      effectiveTemporalAADebugSettings(aa);

  EXPECT_TRUE(effective.jitterEnabled);
  EXPECT_FLOAT_EQ(effective.taaJitterScale, 0.75f);
  EXPECT_FLOAT_EQ(effective.taaCurrentFrameWeight, 0.045f);
  EXPECT_FLOAT_EQ(effective.taaMotionCurrentWeight, 0.22f);
  EXPECT_FLOAT_EQ(effective.taaDisocclusionCurrentWeight, 0.62f);
  EXPECT_FLOAT_EQ(effective.taaClampCurrentWeight, 0.38f);
  EXPECT_FLOAT_EQ(effective.taaVelocityBlendScale, 0.22f);
  EXPECT_FLOAT_EQ(effective.taaVarianceGamma, 1.85f);
  EXPECT_TRUE(effective.taaSharpenEnabled);
  EXPECT_FLOAT_EQ(effective.taaSharpenStrength, 0.14f);
  EXPECT_FLOAT_EQ(effective.taaSharpenConfidenceThreshold, 0.82f);
  EXPECT_TRUE(effective.spatialPostTaaCleanup);
  EXPECT_TRUE(effective.transparentPostTaaSpatialCleanup);
  EXPECT_EQ(effective.taaHistoryFilterMode,
            TemporalAAHistoryFilterMode::CatmullRom);
  EXPECT_EQ(effective.taaClampMode, TemporalAAClampMode::VarianceYCoCg);
  EXPECT_EQ(effective.taaHdrWeightingMode,
            TemporalAAHdrWeightingMode::Luminance);
}

TEST(RenderGraphRendererTest, MsaaAntiAliasingModesSanitizeTemporalState) {
  RenderSettings::AntiAliasingSettings aa{};
  aa.mode = AntiAliasingMode::MSAA8x;
  aa.qualityPreset = static_cast<TemporalAAQualityPreset>(UINT8_MAX);
  aa.debug.jitterEnabled = true;
  aa.debug.freezeJitter = true;

  sanitizeAntiAliasingSettings(aa);

  EXPECT_EQ(aa.mode, AntiAliasingMode::MSAA8x);
  EXPECT_EQ(aa.qualityPreset, TemporalAAQualityPreset::Quality);
  EXPECT_EQ(sanitizeAntiAliasingMode(AntiAliasingMode::MSAA4x),
            AntiAliasingMode::MSAA4x);
  EXPECT_EQ(sanitizeAntiAliasingMode(AntiAliasingMode::MSAA8x),
            AntiAliasingMode::MSAA8x);
  EXPECT_EQ(coverageSampleCount(CoverageMode::Sample4), 4u);
  EXPECT_EQ(coverageSampleCount(CoverageMode::Sample8), 8u);
  EXPECT_EQ(coverageSampleCount(CoverageMode::Sample1), 1u);
  EXPECT_FALSE(aa.debug.jitterEnabled);
  EXPECT_FALSE(aa.debug.freezeJitter);
}

TEST(RenderGraphRendererTest,
     PresentationPlanReportsMsaaCapabilityAndCoveragePolicy) {
  RenderSettings settings{};
  settings.antiAliasing.mode = AntiAliasingMode::MSAA4x;
  const PresentationAAGpuCapabilities supported{
      .sample4Color = true,
      .sample4Depth = true,
      .sample8Color = true,
      .sample8Depth = true,
      .depthResolveMin = true,
      .alphaToCoverage = true,
      .sampleRateShading = true,
  };

  auto planResult = buildPresentationAAPlan(settings, {}, supported);
  ASSERT_FALSE(planResult.hasError()) << planResult.error();
  EXPECT_EQ(planResult.value().coverage, CoverageMode::Sample4);
  EXPECT_EQ(planResult.value().alphaCoverage,
            AlphaCoveragePolicy::ThresholdedAlphaToCoverage);
  EXPECT_EQ(planResult.value().transparency,
            TransparencyAAPolicy::SingleSamplePostResolve);
  EXPECT_TRUE(planResult.value().sampleShadingSupported);
  EXPECT_FALSE(planResult.value().sampleShadingEnabled);

  settings.antiAliasing.mode = AntiAliasingMode::MSAA8x;
  planResult = buildPresentationAAPlan(settings, {}, supported);
  ASSERT_FALSE(planResult.hasError()) << planResult.error();
  EXPECT_EQ(planResult.value().coverage, CoverageMode::Sample8);

  settings.antiAliasing.mode = AntiAliasingMode::MSAA4x;

  const auto expectUnsupported = [&settings, &supported](
                                     PresentationAAGpuCapabilities caps,
                                     PresentationAAUnsupportedReason reason) {
    auto result = buildPresentationAAPlan(settings, {}, caps);
    ASSERT_TRUE(result.hasError());
    EXPECT_EQ(msaaUnsupportedReason(settings.antiAliasing.mode, caps), reason);
    EXPECT_NE(result.error().find(presentationAAUnsupportedReasonName(reason)),
              std::string::npos);
  };
  auto caps = supported;
  caps.sample4Color = false;
  expectUnsupported(caps, PresentationAAUnsupportedReason::Sample4Color);
  caps = supported;
  caps.sample4Depth = false;
  expectUnsupported(caps, PresentationAAUnsupportedReason::Sample4Depth);
  caps = supported;
  caps.depthResolveMin = false;
  expectUnsupported(caps, PresentationAAUnsupportedReason::DepthResolveMin);
  caps = supported;
  caps.alphaToCoverage = false;
  expectUnsupported(caps, PresentationAAUnsupportedReason::AlphaToCoverage);

  settings.antiAliasing.mode = AntiAliasingMode::MSAA8x;
  caps = supported;
  caps.sample8Color = false;
  auto result = buildPresentationAAPlan(settings, {}, caps);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(msaaUnsupportedReason(settings.antiAliasing.mode, caps),
            PresentationAAUnsupportedReason::Sample8Color);
  caps = supported;
  caps.sample8Depth = false;
  result = buildPresentationAAPlan(settings, {}, caps);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(msaaUnsupportedReason(settings.antiAliasing.mode, caps),
            PresentationAAUnsupportedReason::Sample8Depth);
}

TEST(RenderGraphRendererTest,
     AntiAliasingVelocityRejectionThresholdSanitizesPixels) {
  RenderSettings::AntiAliasingSettings aa{};
  aa.debug.taaVelocityRejectionThreshold =
      std::numeric_limits<float>::quiet_NaN();
  sanitizeAntiAliasingSettings(aa);
  EXPECT_FLOAT_EQ(aa.debug.taaVelocityRejectionThreshold, 1.5f);

  aa.debug.taaVelocityRejectionThreshold = -1.0f;
  sanitizeAntiAliasingSettings(aa);
  EXPECT_FLOAT_EQ(aa.debug.taaVelocityRejectionThreshold, 0.0f);

  aa.debug.taaVelocityRejectionThreshold = 128.0f;
  sanitizeAntiAliasingSettings(aa);
  EXPECT_FLOAT_EQ(aa.debug.taaVelocityRejectionThreshold, 64.0f);
}

TEST(RenderGraphRendererTest, ShadowSettingsSanitizeClampsCoreValues) {
  RenderSettings::ShadowSettings settings{};
  settings.cascadeCount = 0u;
  settings.qualityPreset = static_cast<ShadowQualityPreset>(99u);
  settings.shadowMapSize = 0u;
  settings.depthFormat = Format::RGBA8_UNORM;
  settings.maxDistance = -10.0f;
  settings.maxDistanceFadeFraction = -1.0f;
  settings.splitLambda = 2.0f;
  settings.cascadeBlendFraction = -1.0f;
  settings.constantBias = std::numeric_limits<float>::quiet_NaN();
  settings.slopeBias = std::numeric_limits<float>::quiet_NaN();
  settings.normalBias = std::numeric_limits<float>::quiet_NaN();
  settings.pcfSampleCount = 0u;
  settings.debug.diagnosticLogLevel = static_cast<LogLevel>(99u);
  settings.debug.diagnosticLogIntervalFrames = 0u;
  settings.debug.debugCascadeIndex = 99u;
  settings.debug.previewDepthMin = -1.0f;
  settings.debug.previewDepthMax = 2.0f;

  sanitizeShadowSettings(settings);

  EXPECT_EQ(settings.cascadeCount, 1u);
  EXPECT_EQ(settings.qualityPreset, ShadowQualityPreset::Custom);
  EXPECT_EQ(settings.shadowMapSize, 1u);
  EXPECT_EQ(settings.depthFormat, kDefaultShadowMapDepthFormat);
  EXPECT_FLOAT_EQ(settings.maxDistance, 150.0f);
  EXPECT_FLOAT_EQ(settings.maxDistanceFadeFraction, 0.0f);
  EXPECT_FLOAT_EQ(settings.splitLambda, 1.0f);
  EXPECT_FLOAT_EQ(settings.cascadeBlendFraction, 0.0f);
  EXPECT_FLOAT_EQ(settings.constantBias, 0.0005f);
  EXPECT_FLOAT_EQ(settings.slopeBias, 1.5f);
  EXPECT_FLOAT_EQ(settings.normalBias, 0.0f);
  EXPECT_EQ(settings.pcfSampleCount, 1u);
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
  EXPECT_EQ(settings.depthFormat, kDefaultShadowMapDepthFormat);
  EXPECT_FLOAT_EQ(settings.maxDistance, 80.0f);
  EXPECT_FLOAT_EQ(settings.maxDistanceFadeFraction, 0.15f);
  EXPECT_FLOAT_EQ(settings.splitLambda, 0.35f);
  EXPECT_EQ(settings.pcfSampleCount, 9u);
  EXPECT_FLOAT_EQ(settings.constantBias, 0.0008f);
  EXPECT_FLOAT_EQ(settings.slopeBias, 2.0f);
  EXPECT_FLOAT_EQ(settings.normalBias, 0.25f);
  EXPECT_TRUE(settings.debug.showShadowMapViewport);

  applyShadowQualityPreset(settings, ShadowQualityPreset::Medium);
  EXPECT_EQ(settings.qualityPreset, ShadowQualityPreset::Medium);
  EXPECT_EQ(settings.cascadeCount, 3u);
  EXPECT_EQ(settings.shadowMapSize, 2048u);
  EXPECT_EQ(settings.depthFormat, kDefaultShadowMapDepthFormat);
  EXPECT_FLOAT_EQ(settings.maxDistance, 120.0f);
  EXPECT_FLOAT_EQ(settings.maxDistanceFadeFraction, 0.12f);
  EXPECT_FLOAT_EQ(settings.splitLambda, 0.30f);
  EXPECT_EQ(settings.pcfSampleCount, 16u);
  EXPECT_FLOAT_EQ(settings.constantBias, 0.0006f);
  EXPECT_FLOAT_EQ(settings.slopeBias, 1.75f);
  EXPECT_FLOAT_EQ(settings.normalBias, 0.35f);

  applyShadowQualityPreset(settings, ShadowQualityPreset::High);
  EXPECT_EQ(settings.qualityPreset, ShadowQualityPreset::High);
  EXPECT_EQ(settings.cascadeCount, 4u);
  EXPECT_EQ(settings.shadowMapSize, 4096u);
  EXPECT_EQ(settings.depthFormat, kDefaultShadowMapDepthFormat);
  EXPECT_FLOAT_EQ(settings.maxDistance, 150.0f);
  EXPECT_FLOAT_EQ(settings.maxDistanceFadeFraction, 0.10f);
  EXPECT_FLOAT_EQ(settings.splitLambda, 0.25f);
  EXPECT_EQ(settings.pcfSampleCount, 24u);
  EXPECT_FLOAT_EQ(settings.constantBias, 0.0005f);
  EXPECT_FLOAT_EQ(settings.slopeBias, 1.5f);
  EXPECT_FLOAT_EQ(settings.normalBias, 0.50f);

  applyShadowQualityPreset(settings, ShadowQualityPreset::Ultra);
  EXPECT_EQ(settings.qualityPreset, ShadowQualityPreset::Ultra);
  EXPECT_EQ(settings.cascadeCount, 4u);
  EXPECT_EQ(settings.shadowMapSize, 8192u);
  EXPECT_EQ(settings.depthFormat, Format::D32_FLOAT);
  EXPECT_FLOAT_EQ(settings.maxDistance, 220.0f);
  EXPECT_FLOAT_EQ(settings.maxDistanceFadeFraction, 0.08f);
  EXPECT_FLOAT_EQ(settings.splitLambda, 0.50f);
  EXPECT_EQ(settings.pcfSampleCount, 32u);
  EXPECT_FLOAT_EQ(settings.constantBias, 0.00035f);
  EXPECT_FLOAT_EQ(settings.slopeBias, 1.15f);
  EXPECT_FLOAT_EQ(settings.normalBias, 0.40f);
}

TEST(RenderGraphRendererTest, ShadowQualityPresetResetsPresetOwnedState) {
  RenderSettings::ShadowSettings settings{};
  settings.cascadeBlendFraction = 0.75f;
  settings.sdsmTemporalBlend = 0.1f;

  applyShadowQualityPreset(settings, ShadowQualityPreset::Ultra);

  EXPECT_FLOAT_EQ(settings.cascadeBlendFraction, 0.08f);
  EXPECT_FLOAT_EQ(settings.sdsmTemporalBlend, 0.85f);
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
     BackgroundRotationMotionIgnoresTranslationAndUsesPreviousUvConvention) {
  const glm::mat4 projection =
      glm::perspective(glm::radians(60.0f), 16.0f / 9.0f, 0.1f, 100.0f);
  const glm::vec3 previousPosition{7.0f, 2.0f, -3.0f};
  const glm::vec3 currentPosition{-11.0f, 5.0f, 9.0f};
  const glm::vec3 previousForward{0.0f, 0.0f, -1.0f};
  const glm::mat4 previousView =
      glm::lookAt(previousPosition, previousPosition + previousForward,
                  glm::vec3(0.0f, 1.0f, 0.0f));

  CameraFrameState translationOnly{};
  translationOnly.view =
      glm::lookAt(currentPosition, currentPosition + previousForward,
                  glm::vec3(0.0f, 1.0f, 0.0f));
  translationOnly.proj = projection;
  translationOnly.currentUnjitteredProj = projection;
  translationOnly.currentUnjitteredViewProj = projection * translationOnly.view;
  translationOnly.previousUnjitteredViewProj = projection * previousView;
  translationOnly.historyValid = true;
  EXPECT_TRUE(mat4Near(makeBackgroundRotationReprojection(translationOnly),
                       glm::mat4(1.0f), 1.0e-5f));

  const float yaw = glm::radians(20.0f);
  const glm::vec3 currentForward{std::sin(yaw), 0.0f, -std::cos(yaw)};
  CameraFrameState rotated = translationOnly;
  rotated.view = glm::lookAt(currentPosition, currentPosition + currentForward,
                             glm::vec3(0.0f, 1.0f, 0.0f));
  rotated.currentUnjitteredViewProj = projection * rotated.view;
  const glm::mat4 reprojection = makeBackgroundRotationReprojection(rotated);
  const glm::vec2 currentUv{0.5f, 0.5f};
  const glm::vec4 previousClip =
      reprojection * glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);
  ASSERT_NE(previousClip.w, 0.0f);
  const glm::vec2 previousUv =
      taaScreenUvFromClipNdc(glm::vec2(previousClip) / previousClip.w);
  const glm::vec2 motionUv = previousUv - currentUv;

  EXPECT_GT(motionUv.x, 0.0f);
  EXPECT_NEAR((currentUv + motionUv).x, previousUv.x, 1.0e-6f);
  EXPECT_NEAR((currentUv + motionUv).y, previousUv.y, 1.0e-6f);
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
     TemporalCameraFrameStateResetsOnSceneContentChange) {
  Camera camera{};
  RenderSettings::AntiAliasingSettings aa{};
  aa.mode = AntiAliasingMode::TAA;
  aa.debug.jitterEnabled = true;

  TemporalCameraHistoryState history{};
  TemporalCameraFrameDesc desc{
      .renderExtent = glm::uvec2(1920u, 1080u),
      .sceneContent =
          TemporalSceneContentState{
              .lightTopologyVersion = 1u,
              .lightTransformVersion = 2u,
              .materialTableVersion = 3u,
              .environmentVersion = 4u,
          },
  };
  const CameraFrameState first =
      makeTemporalCameraFrameState(camera, 16.0f / 9.0f, aa, desc, history);
  ASSERT_EQ(first.historyResetReason, TemporalHistoryResetReason::FirstFrame);

  const CameraFrameState stable =
      makeTemporalCameraFrameState(camera, 16.0f / 9.0f, aa, desc, history);
  EXPECT_TRUE(stable.historyValid);
  EXPECT_EQ(stable.historyResetReason, TemporalHistoryResetReason::None);
  EXPECT_EQ(stable.historyResetCount, 1u);

  desc.sceneContent.lightTransformVersion += 1u;
  const CameraFrameState changed =
      makeTemporalCameraFrameState(camera, 16.0f / 9.0f, aa, desc, history);

  EXPECT_FALSE(changed.historyValid);
  EXPECT_EQ(changed.historyResetReason,
            TemporalHistoryResetReason::SceneContentChanged);
  EXPECT_EQ(changed.framesSinceHistoryReset, 0u);
  EXPECT_EQ(changed.historyResetCount, 2u);
  EXPECT_EQ(history.previousSceneContent, desc.sceneContent);
  EXPECT_EQ(temporalHistoryResetReasonName(
                TemporalHistoryResetReason::SceneContentChanged),
            "Scene Content Changed");
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

TEST(RenderGraphRendererTest, PracticalCascadeSplitDepthsStayMonotonic) {
  CameraFrameState camera{};
  camera.nearPlane = 0.1f;
  camera.farPlane = 100.0f;

  const auto practicalSplits =
      shadow_detail::computeCascadeSplitDepths(camera, 50.0f, 4u, 0.75f);

  for (uint32_t i = 0u; i < 4u; ++i) {
    EXPECT_LT(practicalSplits[i], practicalSplits[i + 1u]);
  }
  EXPECT_NEAR(practicalSplits[0], 0.1f, 1.0e-6f);
  EXPECT_NEAR(practicalSplits[4], 50.0f, 1.0e-6f);

  const auto rangedSplits =
      shadow_detail::computeCascadeSplitDepthsForRange(2.0f, 20.0f, 4u, 0.5f);
  for (uint32_t i = 0u; i < 4u; ++i) {
    EXPECT_LT(rangedSplits[i], rangedSplits[i + 1u]);
  }
  EXPECT_NEAR(rangedSplits[0], 2.0f, 1.0e-6f);
  EXPECT_NEAR(rangedSplits[4], 20.0f, 1.0e-6f);
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
      FrameTextureRequirementFlags::MotionVectors |
      FrameTextureRequirementFlags::HistoryColor;
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
  const TextureHandle firstSceneDepth =
      frameContext.sharedResources.sceneDepthTexture;
  const TextureHandle firstHistoryWrite =
      frameContext.sharedResources.historyColorWriteTexture;
  EXPECT_EQ(frameContext.metrics.antiAliasing.historyColorTextureCount, 3u);
  EXPECT_EQ(frameContext.metrics.antiAliasing.historyColorTextureBytes,
            1280ull * 720ull * 8ull);
  EXPECT_EQ(frameContext.metrics.antiAliasing.historyColorTotalBytes,
            3ull * 1280ull * 720ull * 8ull);
  EXPECT_FALSE(isValid(frameContext.sharedResources.previousSceneDepthTexture));
  EXPECT_FALSE(frameContext.metrics.antiAliasing.previousSceneDepthValid);

  frameContext.sharedResources.historyWriteRequirements |=
      FrameTextureRequirementFlags::SceneDepth |
      FrameTextureRequirementFlags::MotionVectors |
      FrameTextureRequirementFlags::HistoryColor;
  provider.onFrameSubmitted(frameContext);
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
  ASSERT_TRUE(isValid(frameContext.sharedResources.previousSceneDepthTexture));
  EXPECT_TRUE(sameTexture(
      frameContext.sharedResources.previousSceneDepthTexture, firstSceneDepth));
  EXPECT_FALSE(sameTexture(frameContext.sharedResources.sceneDepthTexture,
                           firstSceneDepth));
  EXPECT_TRUE(frameContext.metrics.antiAliasing.previousSceneDepthValid);
  EXPECT_EQ(frameContext.metrics.antiAliasing.previousSceneDepthTextureBytes,
            1280ull * 720ull * 4ull);
  EXPECT_FALSE(sameTexture(frameContext.sharedResources.motionVectorTexture,
                           firstMotionVector));
  EXPECT_TRUE(sameTexture(frameContext.sharedResources.historyColorReadTexture,
                          firstHistoryWrite));
  const TextureHandle secondHistoryWrite =
      frameContext.sharedResources.historyColorWriteTexture;
  EXPECT_FALSE(sameTexture(secondHistoryWrite, firstHistoryWrite));

  frameContext.sharedResources.historyWriteRequirements |=
      FrameTextureRequirementFlags::SceneDepth |
      FrameTextureRequirementFlags::MotionVectors |
      FrameTextureRequirementFlags::HistoryColor;
  provider.onFrameSubmitted(frameContext);
  frameContext.frameIndex = 2u;
  prepareResult = provider.prepare(ctx);
  ASSERT_FALSE(prepareResult.hasError());
  EXPECT_TRUE(isValid(frameContext.sharedResources.previousSceneDepthTexture));
  EXPECT_FALSE(sameTexture(
      frameContext.sharedResources.previousSceneDepthTexture, firstSceneDepth));
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

TEST(RenderGraphRendererTest, SkyboxFrameDataUsesOneBufferPerLogicalFrameSlot) {
  std::array<std::byte, 32 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  gpu.swapchainImageCount = 2u;
  Renderer renderer(gpu, memory);
  RenderGraphBuilder graph(&memory);
  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());

  RenderFrameContext frameContext{};
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  FrameBuildContext ctx{
      .frame = frameContext,
      .graph = graph,
      .resources = renderer.resources(),
      .shared = frameContext.sharedResources,
  };

  const std::filesystem::path shaderRoot =
      std::filesystem::path(PROJECT_SOURCE_DIR) / "assets" / "shaders";
  const SkyboxFeatureConfig config{
      .vertex = shaderRoot / "skybox.vert",
      .fragment = shaderRoot / "skybox.frag",
  };
  const uint32_t initialCreatedBuffers = gpu.createdBufferCount;
  const uint32_t initialDestroyedBuffers = gpu.destroyedBufferCount;

  {
    SkyboxPass pass(gpu, config);

    for (uint64_t frameIndex = 0u; frameIndex < 3u; ++frameIndex) {
      graph.beginFrame(frameIndex);
      frameContext.frameIndex = frameIndex;
      auto prepareResult = pass.prepare(ctx);
      ASSERT_FALSE(prepareResult.hasError()) << prepareResult.error();

      const uint32_t expectedBuffers = frameIndex == 0u ? 1u : 2u;
      EXPECT_EQ(gpu.createdBufferCount - initialCreatedBuffers,
                expectedBuffers);
    }

    gpu.swapchainImageCount = 3u;
    graph.beginFrame(3u);
    frameContext.frameIndex = 3u;
    auto prepareResult = pass.prepare(ctx);
    ASSERT_FALSE(prepareResult.hasError()) << prepareResult.error();
    EXPECT_EQ(gpu.createdBufferCount - initialCreatedBuffers, 3u);
    EXPECT_EQ(gpu.destroyedBufferCount - initialDestroyedBuffers, 2u);
  }

  EXPECT_EQ(gpu.destroyedBufferCount - initialDestroyedBuffers, 3u);
}

TEST(RenderGraphRendererTest,
     OpaqueFeaturePrewarmsConfiguredRasterStateVariantsOnAttach) {
  std::pmr::unsynchronized_pool_resource memory;
  FakeMeshletPipelineGpuDevice gpu;

  RenderPipeline pipeline(&memory);
  ASSERT_NE(registerOpaquePrepassStages(
                pipeline, gpu,
                makeOpaqueConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
                &memory),
            nullptr);

  const auto meshletIt =
      std::find(gpu.createdMeshletPipelineNames.begin(),
                gpu.createdMeshletPipelineNames.end(), "opaque_meshlet");
  ASSERT_NE(meshletIt, gpu.createdMeshletPipelineNames.end());
  const size_t meshletIndex = static_cast<size_t>(
      std::distance(gpu.createdMeshletPipelineNames.begin(), meshletIt));
  ASSERT_LT(meshletIndex, gpu.createdMeshletPipelineDescs.size());
  const auto hasDepthState = [](std::span<const RasterPipelineState> states,
                                CompareOp compareOp, bool depthWrite) {
    return std::any_of(
        states.begin(), states.end(),
        [compareOp, depthWrite](const RasterPipelineState &state) {
          return state.compareOp == compareOp && state.depthWrite == depthWrite;
        });
  };
  EXPECT_TRUE(hasDepthState(
      gpu.createdMeshletPipelineDescs[meshletIndex].prewarmRasterStates,
      CompareOp::Equal, false));

  const auto classicIt =
      std::find(gpu.createdRenderPipelineNames.begin(),
                gpu.createdRenderPipelineNames.end(), "opaque_mesh");
  ASSERT_NE(classicIt, gpu.createdRenderPipelineNames.end());
  const size_t classicIndex = static_cast<size_t>(
      std::distance(gpu.createdRenderPipelineNames.begin(), classicIt));
  ASSERT_LT(classicIndex, gpu.createdRenderPipelineDescs.size());
  EXPECT_TRUE(hasDepthState(
      gpu.createdRenderPipelineDescs[classicIndex].prewarmRasterStates,
      CompareOp::Equal, false));
}

TEST(RenderGraphRendererTest, ShadowFeatureBuildsNoGraphPassesWhenDisabled) {
  std::array<std::byte, 32 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeRendererGPUDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  auto *shadow = registerShadowStage(pipeline, gpu, &memory);
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
     ShadowRendererRefreshesCachedGeometryAfterPoolMutation) {
  std::array<std::byte, 256 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeShadowSceneGpuDevice gpu;
  ResourceManager resources(gpu, &memory);
  RenderScene scene(&memory);
  scene.bindResources(&resources);

  MeshData mesh = makeShadowTriangleMesh(&memory);
  auto modelResult =
      resources.acquireGeneratedModel(mesh, "shadow_geometry_refresh");
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();
  auto materialResult = resources.acquireMaterial(MaterialRequest{});
  ASSERT_FALSE(materialResult.hasError()) << materialResult.error();
  auto renderableResult = scene.graph().addRenderable(
      scene.graph().rootNode(), modelResult.value(), materialResult.value());
  ASSERT_FALSE(renderableResult.hasError()) << renderableResult.error();
  addDirectionalLightToScene(scene);

  RenderSettings settings{};
  settings.shadow.enabled = true;
  settings.shadow.cascadeCount = 1u;
  settings.shadow.shadowMapSize = 128u;
  settings.shadow.maxDistance = 20.0f;

  RenderFrameContext frame{};
  frame.frameIndex = 1u;
  frame.scene = &scene;
  frame.resources = &resources;
  frame.settings = &settings;
  frame.camera.view = glm::lookAt(glm::vec3(0.0f, 1.0f, 4.0f), glm::vec3(0.0f),
                                  glm::vec3(0.0f, 1.0f, 0.0f));
  frame.camera.proj = glm::perspective(glm::radians(60.0f), 1.0f, 0.1f, 20.0f);
  frame.camera.cameraPos = glm::vec4(0.0f, 1.0f, 4.0f, 1.0f);
  frame.camera.aspectRatio = 1.0f;
  frame.camera.projectionType = ProjectionType::Perspective;
  frame.camera.nearPlane = 0.1f;
  frame.camera.farPlane = 20.0f;
  frame.camera.fovYRadians = glm::radians(60.0f);

  ShadowRenderer shadow(
      gpu, makeOpaqueConfig(std::filesystem::path(PROJECT_SOURCE_DIR)),
      &memory);
  SceneDrawDatabase drawDatabase(gpu, &memory);
  auto drawDatabaseResult = drawDatabase.update(scene, resources);
  ASSERT_FALSE(drawDatabaseResult.hasError()) << drawDatabaseResult.error();
  frame.sharedResources.sceneDrawDatabase = &drawDatabase;
  auto publishResult = shadow.publishFrameData(frame);
  ASSERT_FALSE(publishResult.hasError()) << publishResult.error();
  auto prepareResult = shadow.prepareShadowGraphPasses(frame);
  ASSERT_FALSE(prepareResult.hasError()) << prepareResult.error();
  ASSERT_EQ(shadow.meshDrawTemplates_.size(), 1u);

  const BufferHandle previousVertexBuffer =
      shadow.meshDrawTemplates_.front().baseVertexBuffer;
  const BufferHandle previousIndexBuffer =
      shadow.meshDrawTemplates_.front().indexBuffer;
  auto relocationResult = gpu.relocateFirstGeometry();
  ASSERT_FALSE(relocationResult.hasError()) << relocationResult.error();
  ASSERT_FALSE(
      sameHandle(previousVertexBuffer, relocationResult.value().vertexBuffer));
  ASSERT_FALSE(
      sameHandle(previousIndexBuffer, relocationResult.value().indexBuffer));
  drawDatabaseResult = drawDatabase.update(scene, resources);
  ASSERT_FALSE(drawDatabaseResult.hasError()) << drawDatabaseResult.error();

  frame.frameIndex = 2u;
  publishResult = shadow.publishFrameData(frame);
  ASSERT_FALSE(publishResult.hasError()) << publishResult.error();
  prepareResult = shadow.prepareShadowGraphPasses(frame);
  ASSERT_FALSE(prepareResult.hasError()) << prepareResult.error();
  ASSERT_EQ(shadow.meshDrawTemplates_.size(), 1u);
  EXPECT_TRUE(sameHandle(shadow.meshDrawTemplates_.front().baseVertexBuffer,
                         relocationResult.value().vertexBuffer));
  EXPECT_TRUE(sameHandle(shadow.meshDrawTemplates_.front().indexBuffer,
                         relocationResult.value().indexBuffer));
  EXPECT_EQ(shadow.cachedGeometryMutationVersion_,
            gpu.geometryMutationVersion());
}

TEST(RenderGraphRendererTest,
     ShadowFeatureUsesCpuWhenGpuReductionIsUnavailable) {
  std::array<std::byte, 64 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeNoComputeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  registerShadowStage(pipeline, gpu, &memory);

  RenderScene scene(&memory);
  scene.bindResources(&renderer.resources());
  addDirectionalLightToScene(scene);

  auto pyramidTextureResult =
      gpu.createTexture(makeTransientTextureDesc(Format::RG32_FLOAT, 1u, 1u),
                        "sdsm_auto_cpu_fallback");
  ASSERT_FALSE(pyramidTextureResult.hasError()) << pyramidTextureResult.error();

  RenderSettings settings{};
  settings.shadow.enabled = true;

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
                .activeReductionBackend,
            ShadowSdsmReductionBackend::Cpu);
  EXPECT_FALSE(frameContext.sharedResources.shadowDebugFrameData->sdsm
                   .reductionFallbackActive);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureGpuSdsmUsesNewestCompletedRingResultWhenLatestSlotIsInvalid) {
  std::array<std::byte, 64 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  registerShadowStage(
      pipeline, gpu,
      makeShadowConfig(std::filesystem::path(PROJECT_SOURCE_DIR)), &memory);

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
  expectGpuSdsmResultAvailable(sdsm, 0u);
  EXPECT_FLOAT_EQ(sdsm.rawDeviceMin, 0.2f);
  EXPECT_FLOAT_EQ(sdsm.rawDeviceMax, 0.5f);
  ASSERT_EQ(sdsm.splitCount, 4u);
  EXPECT_EQ(sdsm.effectiveSplitDepths, sdsm.minMaxSplitDepths);
  EXPECT_NE(sdsm.effectiveSplitDepths, sdsm.fixedSplitDepths);
  EXPECT_LT(sdsm.effectiveSplitDepths[1], sdsm.fixedSplitDepths[1]);
  EXPECT_FLOAT_EQ(sdsm.effectiveSplitDepths[sdsm.splitCount],
                  sdsm.fixedSplitDepths[sdsm.splitCount]);
}

TEST(RenderGraphRendererTest,
     ShadowFeatureGpuSdsmIgnoresRingResultWithMismatchedSourceFrameIndex) {
  std::array<std::byte, 64 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeFullscreenGpuDevice gpu;
  Renderer renderer(gpu, memory);
  RenderPipeline pipeline(&memory);
  registerShadowStage(
      pipeline, gpu,
      makeShadowConfig(std::filesystem::path(PROJECT_SOURCE_DIR)), &memory);

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
  expectGpuSdsmResultAvailable(sdsm, 0u);
  EXPECT_FLOAT_EQ(sdsm.rawDeviceMin, 0.2f);
  EXPECT_FLOAT_EQ(sdsm.rawDeviceMax, 0.5f);
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
  registerShadowStage(
      pipeline, gpu,
      makeOpaqueConfig(std::filesystem::path(PROJECT_SOURCE_DIR)), &memory);

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
  registerShadowStage(
      pipeline, gpu,
      makeOpaqueConfig(std::filesystem::path(PROJECT_SOURCE_DIR)), &memory);

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
     MaterialTableProviderUploadsFiveTablesAsOneOrderedBatch) {
  std::array<std::byte, 16 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeRendererGPUDevice gpu;
  Renderer renderer(gpu, memory);
  MaterialTableGpuProvider provider(gpu);
  RenderGraphBuilder graph(&memory);
  RenderFrameContext frame{};
  frame.resources = &renderer.resources();
  FrameBuildContext ctx{
      .frame = frame,
      .graph = graph,
      .resources = renderer.resources(),
      .shared = frame.sharedResources,
  };

  auto prepareResult = provider.prepare(ctx);
  ASSERT_TRUE(prepareResult.hasValue());
  EXPECT_TRUE(prepareResult.value());
  ASSERT_EQ(gpu.updateBufferBatchCallCount, 1u);
  ASSERT_EQ(gpu.updateBufferBatchSizes.size(), 1u);
  EXPECT_EQ(gpu.updateBufferBatchSizes.front(), 5u);
  EXPECT_EQ(gpu.updateBufferCallCount, 5u);

  prepareResult = provider.prepare(ctx);
  ASSERT_TRUE(prepareResult.hasValue());
  EXPECT_TRUE(prepareResult.value());
  EXPECT_EQ(gpu.updateBufferBatchCallCount, 1u);
  EXPECT_EQ(gpu.updateBufferCallCount, 5u);
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
  EXPECT_GE(firstUploadCount, 3u);

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
