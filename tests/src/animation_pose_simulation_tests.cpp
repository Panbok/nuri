#include "tests_pch.h"

#include "render_graph_test_support.h"

#include "nuri/gfx/sim/animation_gpu_services.h"
#include "nuri/resources/gpu/model.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/resources/scene_importer.h"
#include "nuri/scene/render_scene.h"
#include "nuri/scene_runtime/scene_runtime_host.h"
#include "nuri/sim/animation_pose_simulation.h"

#include <unordered_map>

namespace {

using nuri::test_support::FakeGPUDeviceBase;

struct TestPackedMorphDeltaGpu {
  glm::vec4 positionDelta{0.0f};
  glm::vec4 normalDelta{0.0f};
  glm::vec4 tangentDelta{0.0f};
};
static_assert(sizeof(TestPackedMorphDeltaGpu) == 48);

[[nodiscard]] std::filesystem::path foxPath() {
  return std::filesystem::path(PROJECT_SOURCE_DIR) / "assets" / "models" /
         "Fox" / "Fox.gltf";
}

[[nodiscard]] std::filesystem::path animatedMorphCubePath() {
  return std::filesystem::path(PROJECT_SOURCE_DIR) / "assets" / "models" /
         "AnimatedMorphCube" / "AnimatedMorphCube.gltf";
}

[[nodiscard]] std::filesystem::path shaderRootPath() {
  return std::filesystem::path(PROJECT_SOURCE_DIR) / "assets" / "shaders";
}

class FakeAnimationGpuDevice final : public FakeGPUDeviceBase {
public:
  nuri::Result<nuri::ShaderHandle, std::string>
  createShaderModule(const nuri::ShaderDesc &) override {
    ++shaderCreateCalls;
    if (failShaderCreateAtCall != 0u &&
        shaderCreateCalls == failShaderCreateAtCall) {
      return nuri::Result<nuri::ShaderHandle, std::string>::makeError(
          "injected shader creation failure");
    }
    return nuri::Result<nuri::ShaderHandle, std::string>::makeResult(
        nuri::ShaderHandle{.index = nextShaderIndex_++, .generation = 1u});
  }

  nuri::Result<nuri::ComputePipelineHandle, std::string>
  createComputePipeline(const nuri::ComputePipelineDesc &,
                        std::string_view) override {
    ++computePipelineCreateCalls;
    if (failComputePipelineCreateAtCall != 0u &&
        computePipelineCreateCalls == failComputePipelineCreateAtCall) {
      return nuri::Result<nuri::ComputePipelineHandle, std::string>::makeError(
          "injected compute pipeline creation failure");
    }
    return nuri::Result<nuri::ComputePipelineHandle, std::string>::makeResult(
        nuri::ComputePipelineHandle{.index = nextComputePipelineIndex_++,
                                    .generation = 1u});
  }

  void destroyShaderModule(nuri::ShaderHandle) override {
    ++destroyedShaderCount;
  }

  void destroyComputePipeline(nuri::ComputePipelineHandle) override {
    ++destroyedComputePipelineCount;
  }

  uint64_t getBufferDeviceAddress(nuri::BufferHandle h,
                                  size_t offset) const override {
    if (!nuri::isValid(h)) {
      return 0u;
    }
    return (static_cast<uint64_t>(h.index) << 32u) +
           static_cast<uint64_t>(offset) + 1u;
  }

  nuri::Result<nuri::GeometryAllocationHandle, std::string>
  allocateGeometry(std::span<const std::byte> vertexBytes, uint32_t vertexCount,
                   std::span<const std::byte> indexBytes, uint32_t indexCount,
                   std::string_view) override {
    auto vertexBuffer = createBufferImpl();
    if (vertexBuffer.hasError()) {
      return nuri::Result<nuri::GeometryAllocationHandle,
                          std::string>::makeError(vertexBuffer.error());
    }
    auto indexBuffer = createBufferImpl();
    if (indexBuffer.hasError()) {
      destroyBufferImpl(vertexBuffer.value());
      return nuri::Result<nuri::GeometryAllocationHandle,
                          std::string>::makeError(indexBuffer.error());
    }

    const nuri::GeometryAllocationHandle handle{.index = nextGeometryIndex_++,
                                                .generation = 1u};
    geometries_.emplace(handle.index,
                        GeometryEntry{
                            .generation = handle.generation,
                            .view =
                                nuri::GeometryAllocationView{
                                    .vertexBuffer = vertexBuffer.value(),
                                    .vertexByteOffset = 0u,
                                    .vertexByteSize = vertexBytes.size(),
                                    .indexBuffer = indexBuffer.value(),
                                    .indexByteOffset = 0u,
                                    .indexByteSize = indexBytes.size(),
                                    .vertexCount = vertexCount,
                                    .indexCount = indexCount,
                                },
                        });
    ++geometryMutationVersion_;
    return nuri::Result<nuri::GeometryAllocationHandle,
                        std::string>::makeResult(handle);
  }

  void releaseGeometry(nuri::GeometryAllocationHandle h) override {
    const auto it = geometries_.find(h.index);
    if (it == geometries_.end() || it->second.generation != h.generation) {
      return;
    }
    destroyBufferImpl(it->second.view.vertexBuffer);
    destroyBufferImpl(it->second.view.indexBuffer);
    geometries_.erase(it);
    ++geometryMutationVersion_;
  }

  bool resolveGeometry(nuri::GeometryAllocationHandle h,
                       nuri::GeometryAllocationView &out) const override {
    const auto it = geometries_.find(h.index);
    if (it == geometries_.end() || it->second.generation != h.generation) {
      return false;
    }
    out = it->second.view;
    return true;
  }

  uint64_t geometryMutationVersion() const override {
    return geometryMutationVersion_;
  }

  uint32_t shaderCreateCalls = 0u;
  uint32_t computePipelineCreateCalls = 0u;
  uint32_t failShaderCreateAtCall = 0u;
  uint32_t failComputePipelineCreateAtCall = 0u;
  uint32_t destroyedShaderCount = 0u;
  uint32_t destroyedComputePipelineCount = 0u;

private:
  struct GeometryEntry {
    uint32_t generation = 0u;
    nuri::GeometryAllocationView view{};
  };

  uint32_t nextShaderIndex_ = 1u;
  uint32_t nextComputePipelineIndex_ = 1u;
  uint32_t nextGeometryIndex_ = 1u;
  uint64_t geometryMutationVersion_ = 1u;
  std::unordered_map<uint32_t, GeometryEntry> geometries_{};
};

[[nodiscard]] size_t
countDispatches(const nuri::AnimationSceneFrameData &frameData,
                std::string_view label) {
  return static_cast<size_t>(std::count_if(
      frameData.preDispatches.begin(), frameData.preDispatches.end(),
      [label](const nuri::ComputeDispatchItem &dispatch) {
        return dispatch.debugLabel == label;
      }));
}

[[nodiscard]] bool
allDependencyBuffersValid(const FakeAnimationGpuDevice &gpu,
                          const nuri::AnimationSceneFrameData &frameData) {
  return std::ranges::all_of(
      frameData.preDispatches, [&](const nuri::ComputeDispatchItem &dispatch) {
        return std::ranges::all_of(
            dispatch.dependencyBuffers, [&](nuri::BufferHandle handle) {
              return nuri::isValid(handle) && gpu.isValid(handle);
            });
      });
}

[[nodiscard]] std::vector<nuri::BufferHandle>
collectDependencyBuffers(const nuri::AnimationSceneFrameData &frameData) {
  std::vector<nuri::BufferHandle> handles;
  for (const nuri::ComputeDispatchItem &dispatch : frameData.preDispatches) {
    for (nuri::BufferHandle handle : dispatch.dependencyBuffers) {
      if (nuri::isValid(handle)) {
        handles.push_back(handle);
      }
    }
  }
  return handles;
}

[[nodiscard]] uint32_t findClipIndex(const nuri::ScenePrefab &prefab,
                                     std::string_view clipName) {
  for (uint32_t i = 0; i < prefab.animations.size(); ++i) {
    if (std::string_view(prefab.animations[i].name) == clipName) {
      return i;
    }
  }
  return std::numeric_limits<uint32_t>::max();
}

TEST(AnimationGpuServicesTests,
     PartialCreationFailuresRetireResourcesAndAllowCleanRetry) {
  FakeAnimationGpuDevice gpu;
  {
    nuri::AnimationGpuServices services(gpu, shaderRootPath());

    gpu.failShaderCreateAtCall = 4u;
    auto shaderFailure = services.ensureInitialized();
    ASSERT_TRUE(shaderFailure.hasError());
    EXPECT_EQ(gpu.destroyedShaderCount, 3u);
    EXPECT_EQ(gpu.computePipelineCreateCalls, 0u);
    EXPECT_EQ(gpu.destroyedComputePipelineCount, 0u);

    gpu.failShaderCreateAtCall = 0u;
    gpu.failComputePipelineCreateAtCall = 3u;
    auto pipelineFailure = services.ensureInitialized();
    ASSERT_TRUE(pipelineFailure.hasError());
    EXPECT_EQ(gpu.destroyedShaderCount, 10u);
    EXPECT_EQ(gpu.computePipelineCreateCalls, 3u);
    EXPECT_EQ(gpu.destroyedComputePipelineCount, 2u);

    gpu.failComputePipelineCreateAtCall = 0u;
    auto retry = services.ensureInitialized();
    ASSERT_FALSE(retry.hasError()) << retry.error();
    EXPECT_TRUE(nuri::isValid(services.samplePipeline()));
    EXPECT_TRUE(nuri::isValid(services.skinPipeline()));
    EXPECT_EQ(gpu.destroyedShaderCount, 10u);
    EXPECT_EQ(gpu.destroyedComputePipelineCount, 2u);
  }

  EXPECT_EQ(gpu.destroyedShaderCount, 17u);
  EXPECT_EQ(gpu.destroyedComputePipelineCount, 9u);
}

TEST(AnimationPoseSimulationTests, MakeDescRejectsInvalidInputs) {
  nuri::ScenePrefab prefab;
  nuri::SceneInstantiationMap instantiationMap;

  auto missingPrefab = nuri::makeAnimationPoseSimulationDesc(
      nuri::AnimationPoseSimulationCreateInfo{
          .prefab = nullptr,
          .instantiationMap = &instantiationMap,
          .rootNode = nuri::NodeId{1u},
      });
  EXPECT_TRUE(missingPrefab.hasError());

  auto missingMap = nuri::makeAnimationPoseSimulationDesc(
      nuri::AnimationPoseSimulationCreateInfo{
          .prefab = &prefab,
          .instantiationMap = nullptr,
          .rootNode = nuri::NodeId{1u},
      });
  EXPECT_TRUE(missingMap.hasError());

  auto missingRoot = nuri::makeAnimationPoseSimulationDesc(
      nuri::AnimationPoseSimulationCreateInfo{
          .prefab = &prefab,
          .instantiationMap = &instantiationMap,
          .rootNode = nuri::kInvalidNodeId,
      });
  EXPECT_TRUE(missingRoot.hasError());
}

TEST(AnimationPoseSimulationTests, DecodeAnimationPoseParamsRoundTripsPayload) {
  const nuri::AnimationPoseSimulationParams params{
      .primary =
          nuri::AnimationPoseClipState{
              .clipIndex = 1u,
              .timeSeconds = 1.25f,
              .playbackMode = nuri::AnimationPosePlaybackMode::Loop,
              .playing = true,
          },
      .secondary =
          nuri::AnimationPoseClipState{
              .clipIndex = 2u,
              .timeSeconds = 0.5f,
              .playbackMode = nuri::AnimationPosePlaybackMode::Once,
              .playing = false,
          },
      .blendWeight = 0.35f,
      .blendMode = nuri::AnimationPoseBlendMode::Lerp,
      .blendSyncMode = nuri::AnimationPoseBlendSyncMode::NormalizedTime,
  };

  auto decodeResult =
      nuri::decodeAnimationPoseSimulationParams(nuri::asBytes(params));
  ASSERT_FALSE(decodeResult.hasError()) << decodeResult.error();
  const nuri::AnimationPoseSimulationParams &decoded = decodeResult.value();
  EXPECT_EQ(decoded.primary.clipIndex, params.primary.clipIndex);
  EXPECT_FLOAT_EQ(decoded.primary.timeSeconds, params.primary.timeSeconds);
  EXPECT_EQ(decoded.primary.playbackMode, params.primary.playbackMode);
  EXPECT_EQ(decoded.primary.playing, params.primary.playing);
  EXPECT_EQ(decoded.secondary.clipIndex, params.secondary.clipIndex);
  EXPECT_FLOAT_EQ(decoded.secondary.timeSeconds, params.secondary.timeSeconds);
  EXPECT_EQ(decoded.secondary.playbackMode, params.secondary.playbackMode);
  EXPECT_EQ(decoded.secondary.playing, params.secondary.playing);
  EXPECT_FLOAT_EQ(decoded.blendWeight, params.blendWeight);
  EXPECT_EQ(decoded.blendMode, params.blendMode);
  EXPECT_EQ(decoded.blendSyncMode, params.blendSyncMode);
}

TEST(AnimationPoseSimulationTests,
     DecodeAnimationPoseParamsRejectsInvalidSize) {
  const nuri::AnimationPoseSimulationParams params{};
  const std::span<const std::byte> bytes = nuri::asBytes(params);
  auto decodeResult =
      nuri::decodeAnimationPoseSimulationParams(bytes.first(bytes.size() - 1u));
  EXPECT_TRUE(decodeResult.hasError());
}

TEST(AnimationPoseSimulationTests, SanitizeClampsBlendWeightAndDisablesBlend) {
  nuri::AnimationPoseSimulationParams params{
      .primary =
          nuri::AnimationPoseClipState{
              .clipIndex = 0u,
              .playing = true,
          },
      .secondary =
          nuri::AnimationPoseClipState{
              .clipIndex = 1u,
              .playing = true,
          },
      .blendWeight = -0.25f,
      .blendMode = nuri::AnimationPoseBlendMode::Lerp,
  };

  nuri::sanitizeAnimationPoseSimulationParams(params);
  EXPECT_FLOAT_EQ(params.blendWeight, 0.0f);
  EXPECT_EQ(params.blendMode, nuri::AnimationPoseBlendMode::Single);
}

TEST(AnimationPoseSimulationTests, MakeDescBuildsGpuOnlyAnimationBinding) {
  nuri::RenderScene scene;
  nuri::ScenePrefab prefab;
  nuri::SceneInstantiationMap instantiationMap;

  auto descResult = nuri::makeAnimationPoseSimulationDesc(
      nuri::AnimationPoseSimulationCreateInfo{
          .prefab = &prefab,
          .instantiationMap = &instantiationMap,
          .rootNode = scene.graph().rootNode(),
          .debugName = "BookAnimation",
          .params =
              nuri::AnimationPoseSimulationParams{
                  .primary =
                      nuri::AnimationPoseClipState{
                          .clipIndex = 0u,
                          .timeSeconds = 1.25f,
                          .playbackMode = nuri::AnimationPosePlaybackMode::Loop,
                          .playing = true,
                      },
              },
      });

  ASSERT_FALSE(descResult.hasError()) << descResult.error();
  const nuri::SimulationDesc &desc = descResult.value();
  EXPECT_EQ(desc.kind, nuri::SimulationKind::AnimationPose);
  EXPECT_EQ(desc.debugName, "BookAnimation");
  EXPECT_EQ(desc.binding.primaryTarget.type,
            nuri::SimulationBindingTargetType::PrefabRoot);
  EXPECT_EQ(desc.binding.primaryTarget.prefabRoot, scene.graph().rootNode());
  EXPECT_EQ(desc.backendPreference, nuri::SimulationBackendPreference::GPUOnly);
  EXPECT_TRUE(desc.allowGpuExecution);
  EXPECT_FALSE(desc.startPaused);
}

TEST(AnimationPoseSimulationTests, RuntimeCreationFailsWithoutGpuServices) {
  nuri::RenderScene scene;
  nuri::SceneRuntimeHost runtime;
  runtime.bindScene(&scene);

  nuri::ScenePrefab prefab;
  nuri::SceneInstantiationMap instantiationMap;
  auto createResult = runtime.createAnimationPoseSimulation(
      nuri::AnimationPoseSimulationCreateInfo{
          .prefab = &prefab,
          .instantiationMap = &instantiationMap,
          .rootNode = scene.graph().rootNode(),
          .debugName = "BookAnimation",
      });

  EXPECT_TRUE(createResult.hasError());
  EXPECT_NE(createResult.error().find("GPU services"), std::string::npos);
}

TEST(AnimationPoseSimulationTests,
     AnimationHistoryAdvancesOnlyAfterSubmittedFramesForFox) {
  const std::filesystem::path path = foxPath();
  ASSERT_TRUE(std::filesystem::exists(path)) << path.string();

  auto prefabResult =
      nuri::SceneImporter::loadScenePrefabFromFile(path.string());
  ASSERT_FALSE(prefabResult.hasError()) << prefabResult.error();
  const nuri::ScenePrefab &prefab = prefabResult.value();

  FakeAnimationGpuDevice gpu;
  nuri::ResourceManager resources(gpu);
  auto assetsResult = resources.acquireScenePrefabAssets(prefab);
  ASSERT_FALSE(assetsResult.hasError()) << assetsResult.error();

  nuri::RenderScene scene;
  scene.bindResources(&resources);

  nuri::SceneInstantiationMap instantiation;
  auto instantiateResult = scene.graph().instantiatePrefab(
      prefab, scene.graph().rootNode(), assetsResult.value(), &instantiation);
  ASSERT_FALSE(instantiateResult.hasError()) << instantiateResult.error();

  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();

  nuri::SceneRuntimeHost runtime;
  runtime.bindScene(&scene);
  nuri::AnimationGpuServices services(gpu, shaderRootPath());
  runtime.attachAnimationGpuServices(&services);

  auto createResult = runtime.createAnimationPoseSimulation(
      nuri::AnimationPoseSimulationCreateInfo{
          .prefab = &prefab,
          .instantiationMap = &instantiation,
          .rootNode = instantiateResult.value(),
          .debugName = "FoxAnimation",
          .params =
              nuri::AnimationPoseSimulationParams{
                  .primary =
                      nuri::AnimationPoseClipState{
                          .clipIndex = 0u,
                          .timeSeconds = 0.0f,
                          .playbackMode = nuri::AnimationPosePlaybackMode::Loop,
                          .playing = true,
                      },
              },
      });
  ASSERT_FALSE(createResult.hasError()) << createResult.error();

  auto prepareResult = runtime.prepareAnimationSceneFrame(0u);
  ASSERT_FALSE(prepareResult.hasError()) << prepareResult.error();

  const nuri::AnimationSceneFrameData *frameData =
      runtime.animationSceneFrameData();
  ASSERT_NE(frameData, nullptr);
  EXPECT_EQ(frameData->scene, &scene);
  EXPECT_EQ(frameData->sceneTopologyVersion, scene.topologyVersion());
  EXPECT_EQ(frameData->renderableCount, scene.renderables().size());
  EXPECT_EQ(frameData->geometryOverridesByRenderable.size(),
            scene.renderables().size());
  EXPECT_FALSE(nuri::isValid(frameData->previousInstanceMatricesBuffer));
  EXPECT_EQ(frameData->previousInstanceMatricesAddress, 0u);
  EXPECT_TRUE(frameData->previousGeometryOverridesByRenderable.empty());
  EXPECT_GT(countDispatches(*frameData, "AnimationPose Sample"), 0u);
  EXPECT_GT(countDispatches(*frameData, "AnimationPose World"), 0u);
  EXPECT_GT(countDispatches(*frameData, "AnimationPose Scatter"), 0u);
  EXPECT_GT(countDispatches(*frameData, "AnimationPose SkinPalette"), 0u);
  EXPECT_GT(countDispatches(*frameData, "AnimationPose Skin"), 0u);
  EXPECT_EQ(countDispatches(*frameData, "AnimationPose Blend"), 0u);

  const bool hasGeometryOverride = std::any_of(
      frameData->geometryOverridesByRenderable.begin(),
      frameData->geometryOverridesByRenderable.end(),
      [](const nuri::AnimatedRenderableGeometryOverride &overrideEntry) {
        return overrideEntry.enabled &&
               nuri::isValid(overrideEntry.vertexBuffer);
      });
  EXPECT_TRUE(hasGeometryOverride);
  const auto firstOverrideIt = std::find_if(
      frameData->geometryOverridesByRenderable.begin(),
      frameData->geometryOverridesByRenderable.end(),
      [](const nuri::AnimatedRenderableGeometryOverride &overrideEntry) {
        return overrideEntry.enabled &&
               nuri::isValid(overrideEntry.vertexBuffer);
      });
  ASSERT_NE(firstOverrideIt, frameData->geometryOverridesByRenderable.end());
  const nuri::BufferHandle firstInstanceMatricesBuffer =
      frameData->instanceMatricesBuffer;
  const nuri::BufferHandle firstOverrideBuffer = firstOverrideIt->vertexBuffer;
  runtime.commitAnimationSceneFrame(0u);

  prepareResult = runtime.prepareAnimationSceneFrame(1u);
  ASSERT_FALSE(prepareResult.hasError()) << prepareResult.error();
  frameData = runtime.animationSceneFrameData();
  ASSERT_NE(frameData, nullptr);
  EXPECT_TRUE(nuri::isValid(frameData->previousInstanceMatricesBuffer));
  EXPECT_TRUE(nuri::test_support::sameBuffer(
      frameData->previousInstanceMatricesBuffer, firstInstanceMatricesBuffer));
  EXPECT_NE(frameData->previousInstanceMatricesAddress, 0u);
  EXPECT_EQ(frameData->previousGeometryOverridesByRenderable.size(),
            scene.renderables().size());
  const auto previousOverrideIt = std::find_if(
      frameData->previousGeometryOverridesByRenderable.begin(),
      frameData->previousGeometryOverridesByRenderable.end(),
      [](const nuri::AnimatedRenderableGeometryOverride &overrideEntry) {
        return overrideEntry.enabled &&
               nuri::isValid(overrideEntry.vertexBuffer);
      });
  ASSERT_NE(previousOverrideIt,
            frameData->previousGeometryOverridesByRenderable.end());
  EXPECT_TRUE(nuri::test_support::sameBuffer(previousOverrideIt->vertexBuffer,
                                             firstOverrideBuffer));
  const auto secondOverrideIt = std::find_if(
      frameData->geometryOverridesByRenderable.begin(),
      frameData->geometryOverridesByRenderable.end(),
      [](const nuri::AnimatedRenderableGeometryOverride &overrideEntry) {
        return overrideEntry.enabled &&
               nuri::isValid(overrideEntry.vertexBuffer);
      });
  ASSERT_NE(secondOverrideIt, frameData->geometryOverridesByRenderable.end());
  EXPECT_FALSE(nuri::test_support::sameBuffer(
      secondOverrideIt->vertexBuffer, previousOverrideIt->vertexBuffer));

  runtime.abandonAnimationSceneFrame(1u);
  EXPECT_EQ(runtime.animationSceneFrameData(), nullptr);

  prepareResult = runtime.prepareAnimationSceneFrame(2u);
  ASSERT_FALSE(prepareResult.hasError()) << prepareResult.error();
  frameData = runtime.animationSceneFrameData();
  ASSERT_NE(frameData, nullptr);
  EXPECT_TRUE(nuri::test_support::sameBuffer(
      frameData->previousInstanceMatricesBuffer, firstInstanceMatricesBuffer));
  EXPECT_FALSE(nuri::test_support::sameBuffer(frameData->instanceMatricesBuffer,
                                              firstInstanceMatricesBuffer));
  const auto frameTwoOverrideIt = std::find_if(
      frameData->geometryOverridesByRenderable.begin(),
      frameData->geometryOverridesByRenderable.end(),
      [](const nuri::AnimatedRenderableGeometryOverride &overrideEntry) {
        return overrideEntry.enabled &&
               nuri::isValid(overrideEntry.vertexBuffer);
      });
  ASSERT_NE(frameTwoOverrideIt, frameData->geometryOverridesByRenderable.end());
  const nuri::BufferHandle frameTwoInstanceMatricesBuffer =
      frameData->instanceMatricesBuffer;
  const nuri::BufferHandle frameTwoOverrideBuffer =
      frameTwoOverrideIt->vertexBuffer;
  runtime.commitAnimationSceneFrame(2u);

  prepareResult = runtime.prepareAnimationSceneFrame(3u);
  ASSERT_FALSE(prepareResult.hasError()) << prepareResult.error();
  frameData = runtime.animationSceneFrameData();
  ASSERT_NE(frameData, nullptr);
  EXPECT_TRUE(
      nuri::test_support::sameBuffer(frameData->previousInstanceMatricesBuffer,
                                     frameTwoInstanceMatricesBuffer));
  const auto frameThreePreviousOverrideIt = std::find_if(
      frameData->previousGeometryOverridesByRenderable.begin(),
      frameData->previousGeometryOverridesByRenderable.end(),
      [](const nuri::AnimatedRenderableGeometryOverride &overrideEntry) {
        return overrideEntry.enabled &&
               nuri::isValid(overrideEntry.vertexBuffer);
      });
  ASSERT_NE(frameThreePreviousOverrideIt,
            frameData->previousGeometryOverridesByRenderable.end());
  EXPECT_TRUE(nuri::test_support::sameBuffer(
      frameThreePreviousOverrideIt->vertexBuffer, frameTwoOverrideBuffer));
  EXPECT_FALSE(nuri::test_support::sameBuffer(
      frameData->instanceMatricesBuffer,
      frameData->previousInstanceMatricesBuffer));

  const nuri::BufferHandle finalInstanceMatricesBuffer =
      frameData->instanceMatricesBuffer;
  const auto finalOverrideIt = std::find_if(
      frameData->geometryOverridesByRenderable.begin(),
      frameData->geometryOverridesByRenderable.end(),
      [](const nuri::AnimatedRenderableGeometryOverride &overrideEntry) {
        return overrideEntry.enabled &&
               nuri::isValid(overrideEntry.vertexBuffer);
      });
  ASSERT_NE(finalOverrideIt, frameData->geometryOverridesByRenderable.end());
  const nuri::BufferHandle finalOverrideBuffer = finalOverrideIt->vertexBuffer;
  const uint32_t destroyedBefore = gpu.destroyedBufferCount;
  EXPECT_TRUE(runtime.destroyAnimationPoseSimulation(createResult.value()));
  EXPECT_EQ(runtime.animationSceneFrameData(), nullptr);
  EXPECT_GT(gpu.destroyedBufferCount, destroyedBefore);
  EXPECT_FALSE(gpu.isValid(finalInstanceMatricesBuffer));
  EXPECT_FALSE(gpu.isValid(finalOverrideBuffer));
}

TEST(AnimationPoseSimulationTests,
     PrepareSceneFramePublishesMorphDispatchesForAnimatedMorphCube) {
  const std::filesystem::path path = animatedMorphCubePath();
  ASSERT_TRUE(std::filesystem::exists(path)) << path.string();

  auto prefabResult =
      nuri::SceneImporter::loadScenePrefabFromFile(path.string());
  ASSERT_FALSE(prefabResult.hasError()) << prefabResult.error();
  const nuri::ScenePrefab &prefab = prefabResult.value();

  FakeAnimationGpuDevice gpu;
  nuri::ResourceManager resources(gpu);
  auto assetsResult = resources.acquireScenePrefabAssets(prefab);
  ASSERT_FALSE(assetsResult.hasError()) << assetsResult.error();

  nuri::RenderScene scene;
  scene.bindResources(&resources);

  nuri::SceneInstantiationMap instantiation;
  auto instantiateResult = scene.graph().instantiatePrefab(
      prefab, scene.graph().rootNode(), assetsResult.value(), &instantiation);
  ASSERT_FALSE(instantiateResult.hasError()) << instantiateResult.error();

  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();

  nuri::SceneRuntimeHost runtime;
  runtime.bindScene(&scene);
  nuri::AnimationGpuServices services(gpu, shaderRootPath());
  runtime.attachAnimationGpuServices(&services);

  auto createResult = runtime.createAnimationPoseSimulation(
      nuri::AnimationPoseSimulationCreateInfo{
          .prefab = &prefab,
          .instantiationMap = &instantiation,
          .rootNode = instantiateResult.value(),
          .debugName = "AnimatedMorphCubeAnimation",
          .params =
              nuri::AnimationPoseSimulationParams{
                  .primary =
                      nuri::AnimationPoseClipState{
                          .clipIndex = 0u,
                          .timeSeconds = 0.25f,
                          .playbackMode = nuri::AnimationPosePlaybackMode::Loop,
                          .playing = true,
                      },
              },
      });
  ASSERT_FALSE(createResult.hasError()) << createResult.error();

  auto prepareResult = runtime.prepareAnimationSceneFrame(0u);
  ASSERT_FALSE(prepareResult.hasError()) << prepareResult.error();

  const nuri::AnimationSceneFrameData *frameData =
      runtime.animationSceneFrameData();
  ASSERT_NE(frameData, nullptr);
  EXPECT_GT(countDispatches(*frameData, "AnimationPose Sample"), 0u);
  EXPECT_GT(countDispatches(*frameData, "AnimationPose World"), 0u);
  EXPECT_GT(countDispatches(*frameData, "AnimationPose Scatter"), 0u);
  EXPECT_GT(countDispatches(*frameData, "AnimationPose Morph"), 0u);
  EXPECT_EQ(countDispatches(*frameData, "AnimationPose Skin"), 0u);
  EXPECT_EQ(countDispatches(*frameData, "AnimationPose Blend"), 0u);

  const bool hasGeometryOverride = std::any_of(
      frameData->geometryOverridesByRenderable.begin(),
      frameData->geometryOverridesByRenderable.end(),
      [](const nuri::AnimatedRenderableGeometryOverride &overrideEntry) {
        return overrideEntry.enabled &&
               nuri::isValid(overrideEntry.vertexBuffer);
      });
  EXPECT_TRUE(hasGeometryOverride);

  EXPECT_TRUE(runtime.destroyAnimationPoseSimulation(createResult.value()));
  EXPECT_EQ(runtime.animationSceneFrameData(), nullptr);
}

} // namespace
