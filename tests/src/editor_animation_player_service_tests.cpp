#include "tests_pch.h"

#include "render_graph_test_support.h"

#include "nuri/app/editor_animation_player_service.h"
#include "nuri/gfx/sim/animation_gpu_services.h"
#include "nuri/resources/gpu/model.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/resources/scene_importer.h"
#include "nuri/scene/render_scene.h"
#include "nuri/scene_runtime/scene_runtime_host.h"
#include "nuri/ui/editor_services.h"

#include <unordered_map>

namespace {

using nuri::test_support::FakeGPUDeviceBase;

[[nodiscard]] std::filesystem::path foxPath() {
  return std::filesystem::path(PROJECT_SOURCE_DIR) / "assets" / "models" /
         "Fox" / "Fox.gltf";
}

[[nodiscard]] std::filesystem::path medievalFantasyBookPath() {
  return std::filesystem::path(PROJECT_SOURCE_DIR) / "assets" / "models" /
         "MedievalFantasyBook" / "scene.gltf";
}

[[nodiscard]] std::filesystem::path shaderRootPath() {
  return std::filesystem::path(PROJECT_SOURCE_DIR) / "assets" / "shaders";
}

class FakeAnimationGpuDevice final : public FakeGPUDeviceBase {
public:
  nuri::Result<nuri::ShaderHandle, std::string>
  createShaderModule(const nuri::ShaderDesc &) override {
    return nuri::Result<nuri::ShaderHandle, std::string>::makeResult(
        nuri::ShaderHandle{.index = nextShaderIndex_++, .generation = 1u});
  }

  nuri::Result<nuri::ComputePipelineHandle, std::string>
  createComputePipeline(const nuri::ComputePipelineDesc &,
                        std::string_view) override {
    return nuri::Result<nuri::ComputePipelineHandle, std::string>::makeResult(
        nuri::ComputePipelineHandle{.index = nextComputePipelineIndex_++,
                                    .generation = 1u});
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

struct ServiceTestContext {
  FakeAnimationGpuDevice gpu{};
  nuri::ResourceManager resources{gpu};
  nuri::RenderScene scene{};
  nuri::SceneRuntimeHost runtime{};
  nuri::AnimationGpuServices animationGpuServices{gpu, shaderRootPath()};
  nuri::ScenePrefab prefab{};
  nuri::ScenePrefabAssets assets{};
  nuri::SceneInstantiationMap instantiation{};
  nuri::SceneEditorSelectionState selection{};
  uint64_t simulationFrameIndex = 1u;
  nuri::EditorAnimationPlayerService service;
  nuri::NodeId rootNode = nuri::kInvalidNodeId;
  nuri::RenderableId selectedRenderable = nuri::kInvalidRenderableId;
  uint32_t selectedRenderableIndex = 0u;

  ServiceTestContext()
      : service(
            scene, runtime, selection, []() { return 0.0; },
            [this]() { return simulationFrameIndex++; }) {}
};

[[nodiscard]] nuri::Result<std::unique_ptr<ServiceTestContext>, std::string>
createServiceTestContext(const std::filesystem::path &path,
                         std::string_view instanceLabel) {
  auto ctx = std::make_unique<ServiceTestContext>();
  ctx->scene.bindResources(&ctx->resources);
  ctx->runtime.bindScene(&ctx->scene);
  ctx->runtime.attachAnimationGpuServices(&ctx->animationGpuServices);

  auto prefabResult = nuri::SceneImporter::loadSceneFromFile(path.string());
  if (prefabResult.hasError()) {
    return nuri::Result<std::unique_ptr<ServiceTestContext>,
                        std::string>::makeError(prefabResult.error());
  }
  ctx->prefab = std::move(prefabResult.value().prefab);

  auto assetsResult = ctx->resources.acquireScenePrefabAssets(ctx->prefab);
  if (assetsResult.hasError()) {
    return nuri::Result<std::unique_ptr<ServiceTestContext>,
                        std::string>::makeError(assetsResult.error());
  }
  ctx->assets = std::move(assetsResult.value());

  auto instantiateResult = ctx->scene.graph().instantiatePrefab(
      ctx->prefab, ctx->scene.graph().rootNode(), ctx->assets,
      &ctx->instantiation);
  if (instantiateResult.hasError()) {
    return nuri::Result<std::unique_ptr<ServiceTestContext>,
                        std::string>::makeError(instantiateResult.error());
  }
  ctx->rootNode = instantiateResult.value();

  auto commitResult = ctx->scene.commit();
  if (commitResult.hasError()) {
    return nuri::Result<std::unique_ptr<ServiceTestContext>,
                        std::string>::makeError(commitResult.error());
  }

  ctx->service.registerPrefabInstance(instanceLabel, ctx->prefab,
                                      ctx->instantiation, ctx->rootNode);
  return nuri::Result<std::unique_ptr<ServiceTestContext>,
                      std::string>::makeResult(std::move(ctx));
}

bool selectRenderableByNodeName(ServiceTestContext &ctx,
                                std::string_view nodeName) {
  for (uint32_t prefabRenderableIndex = 0u;
       prefabRenderableIndex < ctx.prefab.renderables.size();
       ++prefabRenderableIndex) {
    const nuri::ScenePrefabRenderable &prefabRenderable =
        ctx.prefab.renderables[prefabRenderableIndex];
    if (prefabRenderable.nodeIndex >= ctx.prefab.nodes.size() ||
        std::string_view(ctx.prefab.nodes[prefabRenderable.nodeIndex].name) !=
            nodeName ||
        prefabRenderableIndex >= ctx.instantiation.renderables.size()) {
      continue;
    }

    const nuri::RenderableId renderableId =
        ctx.instantiation.renderables[prefabRenderableIndex];
    if (!nuri::isValid(renderableId)) {
      continue;
    }

    const auto renderableIndex = ctx.scene.findRenderableIndex(renderableId);
    if (!renderableIndex.has_value()) {
      return false;
    }
    const nuri::Renderable *renderable = ctx.scene.renderable(*renderableIndex);
    if (renderable == nullptr) {
      return false;
    }

    ctx.selectedRenderable = renderableId;
    ctx.selectedRenderableIndex = *renderableIndex;
    ctx.selection.kind = nuri::SceneSelectionKind::NodeRenderable;
    ctx.selection.node = renderable->node;
    ctx.selection.renderableId = renderableId;
    ctx.selection.renderableIndex = *renderableIndex;
    return true;
  }
  return false;
}

[[nodiscard]] nuri::Result<std::unique_ptr<ServiceTestContext>, std::string>
createFoxServiceTestContext() {
  auto ctxResult = createServiceTestContext(foxPath(), "Fox");
  if (ctxResult.hasError()) {
    return ctxResult;
  }
  std::unique_ptr<ServiceTestContext> ctx = std::move(ctxResult.value());

  for (nuri::RenderableId renderableId : ctx->instantiation.renderables) {
    if (nuri::isValid(renderableId)) {
      ctx->selectedRenderable = renderableId;
      break;
    }
  }
  if (!nuri::isValid(ctx->selectedRenderable)) {
    return nuri::Result<std::unique_ptr<ServiceTestContext>, std::string>::
        makeError("No renderable instantiated for Fox prefab");
  }

  const auto renderableIndex =
      ctx->scene.findRenderableIndex(ctx->selectedRenderable);
  if (!renderableIndex.has_value()) {
    return nuri::Result<std::unique_ptr<ServiceTestContext>, std::string>::
        makeError("Failed to resolve selected renderable index");
  }
  ctx->selectedRenderableIndex = *renderableIndex;
  const nuri::Renderable *renderable =
      ctx->scene.renderable(ctx->selectedRenderableIndex);
  if (renderable == nullptr) {
    return nuri::Result<std::unique_ptr<ServiceTestContext>, std::string>::
        makeError("Failed to resolve selected renderable");
  }

  ctx->selection.kind = nuri::SceneSelectionKind::NodeRenderable;
  ctx->selection.node = renderable->node;
  ctx->selection.renderableId = ctx->selectedRenderable;
  ctx->selection.renderableIndex = ctx->selectedRenderableIndex;
  return nuri::Result<std::unique_ptr<ServiceTestContext>,
                      std::string>::makeResult(std::move(ctx));
}

[[nodiscard]] nuri::Result<std::unique_ptr<ServiceTestContext>, std::string>
createBookServiceTestContext() {
  auto ctxResult = createServiceTestContext(medievalFantasyBookPath(),
                                            "MedievalFantasyBook");
  if (ctxResult.hasError()) {
    return ctxResult;
  }
  std::unique_ptr<ServiceTestContext> ctx = std::move(ctxResult.value());
  if (!selectRenderableByNodeName(*ctx, "Mill-wind-wheel_Texture-base_0")) {
    return nuri::Result<std::unique_ptr<ServiceTestContext>, std::string>::
        makeError("Failed to select book wind wheel renderable");
  }
  return nuri::Result<std::unique_ptr<ServiceTestContext>,
                      std::string>::makeResult(std::move(ctx));
}

TEST(EditorAnimationPlayerServiceTests,
     SelectedRenderableExposesAnimatedViewAndClipList) {
  auto ctxResult = createFoxServiceTestContext();
  ASSERT_FALSE(ctxResult.hasError()) << ctxResult.error();
  std::unique_ptr<ServiceTestContext> ctx = std::move(ctxResult.value());

  const nuri::EditorAnimationPlayerView view = ctx->service.selectedView();
  EXPECT_EQ(view.availability,
            nuri::EditorAnimationPlayerAvailability::Animated);
  EXPECT_NE(view.instanceLabel.find("Fox"), std::string::npos);
  EXPECT_FALSE(view.selectionLabel.empty());
  EXPECT_EQ(view.clips.size(), ctx->prefab.animations.size());
  EXPECT_EQ(view.primaryClipIndex, 0u);
  EXPECT_EQ(view.secondaryClipIndex, nuri::kInvalidScenePrefabIndex);
  EXPECT_FLOAT_EQ(view.blendWeight, 0.0f);
  EXPECT_GT(view.timelineDurationSeconds, 0.0f);
  EXPECT_FALSE(view.hasSimulation);
}

TEST(EditorAnimationPlayerServiceTests,
     PausingSelectedBookObjectDoesNotPauseOtherAnimatedObjects) {
  auto ctxResult = createBookServiceTestContext();
  ASSERT_FALSE(ctxResult.hasError()) << ctxResult.error();
  std::unique_ptr<ServiceTestContext> ctx = std::move(ctxResult.value());

  nuri::AnimationPoseSimulationParams params{};
  params.primary.clipIndex = 0u;
  params.primary.timeSeconds = 0.0f;
  params.primary.playbackMode = nuri::AnimationPosePlaybackMode::Loop;
  params.primary.playing = true;
  ASSERT_TRUE(
      ctx->service.startPrefabInstancePlayback(ctx->rootNode, params, "Book"));

  nuri::EditorAnimationPlayerView wheelView = ctx->service.selectedView();
  EXPECT_TRUE(wheelView.hasSimulation);
  EXPECT_TRUE(wheelView.running);
  ASSERT_TRUE(ctx->service.pauseSelectionPlayback());
  wheelView = ctx->service.selectedView();
  EXPECT_TRUE(wheelView.paused);
  EXPECT_FALSE(wheelView.running);

  ASSERT_TRUE(selectRenderableByNodeName(*ctx, "0"));
  const nuri::EditorAnimationPlayerView flagView = ctx->service.selectedView();
  EXPECT_EQ(flagView.availability,
            nuri::EditorAnimationPlayerAvailability::Animated);
  EXPECT_TRUE(flagView.hasSimulation);
  EXPECT_TRUE(flagView.running);
  EXPECT_FALSE(flagView.paused);
}

} // namespace
