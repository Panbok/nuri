#include "tests_pch.h"

#include "nuri/scene/render_scene.h"

#include <array>

namespace {

[[nodiscard]] glm::mat4 translate(const glm::vec3 &value) {
  return glm::translate(glm::mat4(1.0f), value);
}

[[nodiscard]] bool mat4Near(const glm::mat4 &lhs, const glm::mat4 &rhs,
                            float epsilon = 1.0e-5f) {
  for (uint32_t column = 0; column < 4u; ++column) {
    for (uint32_t row = 0; row < 4u; ++row) {
      if (std::abs(lhs[column][row] - rhs[column][row]) > epsilon) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] uint32_t countChildren(const nuri::SceneGraph &graph,
                                     nuri::NodeId parent) {
  nuri::NodeId child = nuri::kInvalidNodeId;
  if (!graph.getNodeFirstChild(parent, child)) {
    return 0u;
  }

  uint32_t count = 0u;
  while (nuri::isValid(child)) {
    ++count;
    nuri::NodeId next = nuri::kInvalidNodeId;
    if (!graph.getNodeNextSibling(child, next)) {
      break;
    }
    child = next;
  }
  return count;
}

TEST(RenderSceneGraphTests, CommitPropagatesHierarchyWorldTransforms) {
  nuri::RenderScene scene;
  nuri::SceneGraph &graph = scene.graph();

  auto parentResult = graph.createNode(graph.rootNode(), "Parent",
                                       translate(glm::vec3(1.0f, 0.0f, 0.0f)));
  ASSERT_FALSE(parentResult.hasError()) << parentResult.error();
  auto childResult = graph.createNode(parentResult.value(), "Child",
                                      translate(glm::vec3(0.0f, 2.0f, 0.0f)));
  ASSERT_FALSE(childResult.hasError()) << childResult.error();

  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();

  glm::mat4 childWorld(1.0f);
  ASSERT_TRUE(
      graph.getCachedNodeWorldTransform(childResult.value(), childWorld));
  EXPECT_TRUE(mat4Near(childWorld, translate(glm::vec3(1.0f, 2.0f, 0.0f))));
}

TEST(RenderSceneGraphTests,
     SyncWorldTransformsUpdatesCachedHierarchyWorldData) {
  nuri::RenderScene scene;
  nuri::SceneGraph &graph = scene.graph();

  auto parentResult = graph.createNode(graph.rootNode(), "Parent",
                                       translate(glm::vec3(2.0f, 0.0f, 0.0f)));
  ASSERT_FALSE(parentResult.hasError()) << parentResult.error();
  auto childResult = graph.createNode(parentResult.value(), "Child",
                                      translate(glm::vec3(0.0f, 3.0f, 0.0f)));
  ASSERT_FALSE(childResult.hasError()) << childResult.error();

  EXPECT_TRUE(graph.syncWorldTransforms());

  glm::mat4 childWorld(1.0f);
  ASSERT_TRUE(
      graph.getCachedNodeWorldTransform(childResult.value(), childWorld));
  EXPECT_TRUE(mat4Near(childWorld, translate(glm::vec3(2.0f, 3.0f, 0.0f))));
}

TEST(RenderSceneGraphTests, ReparentPreserveWorldKeepsWorldTransformStable) {
  nuri::RenderScene scene;
  nuri::SceneGraph &graph = scene.graph();

  auto parentAResult = graph.createNode(graph.rootNode(), "A",
                                        translate(glm::vec3(1.0f, 0.0f, 0.0f)));
  ASSERT_FALSE(parentAResult.hasError()) << parentAResult.error();
  auto parentBResult = graph.createNode(graph.rootNode(), "B",
                                        translate(glm::vec3(5.0f, 0.0f, 0.0f)));
  ASSERT_FALSE(parentBResult.hasError()) << parentBResult.error();
  auto childResult = graph.createNode(parentAResult.value(), "Child",
                                      translate(glm::vec3(0.0f, 2.0f, 0.0f)));
  ASSERT_FALSE(childResult.hasError()) << childResult.error();

  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();

  glm::mat4 before(1.0f);
  ASSERT_TRUE(graph.getCachedNodeWorldTransform(childResult.value(), before));

  ASSERT_TRUE(
      graph.setNodeParent(childResult.value(), parentBResult.value(), true));
  commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();

  glm::mat4 after(1.0f);
  ASSERT_TRUE(graph.getCachedNodeWorldTransform(childResult.value(), after));
  EXPECT_TRUE(mat4Near(after, before));

  glm::mat4 local(1.0f);
  ASSERT_TRUE(graph.getNodeLocalTransform(childResult.value(), local));
  EXPECT_TRUE(mat4Near(local, translate(glm::vec3(-4.0f, 2.0f, 0.0f))));
}

TEST(RenderSceneGraphTests, RenderableCacheTracksOwningNodeWorldTransform) {
  nuri::RenderScene scene;
  nuri::SceneGraph &graph = scene.graph();

  constexpr nuri::ModelRef kModel = nuri::makeModelRef(1u, 1u);
  constexpr nuri::MaterialRef kMaterial = nuri::makeMaterialRef(2u, 1u);

  auto nodeResult = graph.createNode(graph.rootNode(), "Renderable",
                                     translate(glm::vec3(3.0f, 0.0f, 0.0f)));
  ASSERT_FALSE(nodeResult.hasError()) << nodeResult.error();
  auto renderableResult =
      graph.addRenderable(nodeResult.value(), kModel, kMaterial);
  ASSERT_FALSE(renderableResult.hasError()) << renderableResult.error();

  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();
  ASSERT_EQ(scene.renderables().size(), 1u);
  EXPECT_EQ(scene.renderables()[0].id, renderableResult.value());
  EXPECT_EQ(scene.renderables()[0].node, nodeResult.value());
  EXPECT_EQ(scene.renderables()[0].model.value, kModel.value);
  EXPECT_EQ(scene.renderables()[0].material.value, kMaterial.value);
  EXPECT_TRUE(mat4Near(scene.renderables()[0].modelMatrix,
                       translate(glm::vec3(3.0f, 0.0f, 0.0f))));

  const uint64_t transformVersionBefore = scene.transformVersion();
  ASSERT_TRUE(graph.setNodeLocalTransform(
      nodeResult.value(), translate(glm::vec3(7.0f, 0.0f, 0.0f))));
  commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();
  EXPECT_EQ(scene.transformVersion(), transformVersionBefore + 1u);
  EXPECT_TRUE(mat4Near(scene.renderables()[0].modelMatrix,
                       translate(glm::vec3(7.0f, 0.0f, 0.0f))));
}

TEST(RenderSceneGraphTests, InstantiatePrefabComposesParentTransform) {
  nuri::RenderScene scene;
  nuri::SceneGraph &graph = scene.graph();

  nuri::ScenePrefab prefab;
  prefab.nodes.resize(2u);
  prefab.nodes[0].name = "PrefabRoot";
  prefab.nodes[0].localFromParent = translate(glm::vec3(1.0f, 0.0f, 0.0f));
  prefab.nodes[1].name = "LightNode";
  prefab.nodes[1].parentIndex = 0u;
  prefab.nodes[1].localFromParent = translate(glm::vec3(0.0f, 2.0f, 0.0f));
  prefab.meshAssets.push_back(nuri::ScenePrefabMeshAssetRef{
      .sourceSceneMeshIndex = 0u,
      .sourceName = "mesh_0",
  });
  prefab.materialAssets.push_back(nuri::ScenePrefabMaterialAssetRef{
      .sourceMaterialIndex = 0u,
      .sourceName = "material_0",
  });
  prefab.renderables.push_back(nuri::ScenePrefabRenderable{
      .nodeIndex = 0u,
      .meshIndex = 0u,
      .materialIndex = 0u,
  });
  nuri::LightDesc light{};
  light.type = nuri::LightType::Point;
  light.range = 4.0f;
  light.intensity = 2.0f;
  prefab.lights.push_back(nuri::ScenePrefabLight{
      .nodeIndex = 1u,
      .light = light,
  });

  nuri::ScenePrefabAssets assets;
  assets.models.push_back(nuri::makeModelRef(1u, 1u));
  assets.materials.push_back(nuri::makeMaterialRef(1u, 1u));

  auto parentResult = graph.createNode(graph.rootNode(), "Parent",
                                       translate(glm::vec3(10.0f, 0.0f, 0.0f)));
  ASSERT_FALSE(parentResult.hasError()) << parentResult.error();

  nuri::SceneInstantiationMap instantiated;
  auto instantiateResult = graph.instantiatePrefab(prefab, parentResult.value(),
                                                   assets, &instantiated);
  ASSERT_FALSE(instantiateResult.hasError()) << instantiateResult.error();
  ASSERT_EQ(instantiated.nodes.size(), 2u);
  ASSERT_EQ(instantiated.renderables.size(), 1u);
  ASSERT_EQ(instantiated.lights.size(), 1u);

  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();

  ASSERT_EQ(scene.renderables().size(), 1u);
  EXPECT_TRUE(mat4Near(scene.renderables()[0].modelMatrix,
                       translate(glm::vec3(11.0f, 0.0f, 0.0f))));

  nuri::LightDesc worldLight{};
  ASSERT_TRUE(
      graph.getCachedLightWorldDesc(instantiated.lights[0], worldLight));
  EXPECT_NEAR(worldLight.position.x, 11.0f, 1.0e-5f);
  EXPECT_NEAR(worldLight.position.y, 2.0f, 1.0e-5f);
  EXPECT_NEAR(worldLight.position.z, 0.0f, 1.0e-5f);
}

TEST(RenderSceneGraphTests, ChildTraversalEnumeratesLiveChildren) {
  nuri::RenderScene scene;
  nuri::SceneGraph &graph = scene.graph();

  auto parentResult = graph.createNode(graph.rootNode(), "Parent");
  ASSERT_FALSE(parentResult.hasError()) << parentResult.error();
  auto childAResult = graph.createNode(parentResult.value(), "A");
  ASSERT_FALSE(childAResult.hasError()) << childAResult.error();
  auto childBResult = graph.createNode(parentResult.value(), "B");
  ASSERT_FALSE(childBResult.hasError()) << childBResult.error();
  auto childCResult = graph.createNode(parentResult.value(), "C");
  ASSERT_FALSE(childCResult.hasError()) << childCResult.error();

  std::vector<nuri::NodeId> children;
  nuri::NodeId child = nuri::kInvalidNodeId;
  ASSERT_TRUE(graph.getNodeFirstChild(parentResult.value(), child));
  // getNodeFirstChild/getNodeNextSibling walk the current child list in LIFO
  // insertion order, so the newest child appears first.
  while (nuri::isValid(child)) {
    children.push_back(child);
    nuri::NodeId next = nuri::kInvalidNodeId;
    ASSERT_TRUE(graph.getNodeNextSibling(child, next));
    child = next;
  }

  ASSERT_EQ(children.size(), 3u);
  EXPECT_EQ(children[0], childCResult.value());
  EXPECT_EQ(children[1], childBResult.value());
  EXPECT_EQ(children[2], childAResult.value());
}

TEST(RenderSceneGraphTests, ForEachRenderableOnNodeReturnsAttachedRenderables) {
  nuri::RenderScene scene;
  nuri::SceneGraph &graph = scene.graph();
  constexpr nuri::ModelRef kModel = nuri::makeModelRef(1u, 1u);
  constexpr nuri::MaterialRef kMaterial = nuri::makeMaterialRef(2u, 1u);

  auto nodeResult = graph.createNode(graph.rootNode(), "Node");
  ASSERT_FALSE(nodeResult.hasError()) << nodeResult.error();
  auto renderableA = graph.addRenderable(nodeResult.value(), kModel, kMaterial);
  ASSERT_FALSE(renderableA.hasError()) << renderableA.error();
  auto renderableB = graph.addRenderable(nodeResult.value(), kModel, kMaterial);
  ASSERT_FALSE(renderableB.hasError()) << renderableB.error();

  std::vector<nuri::RenderableId> renderables;
  // forEachRenderableOnNode preserves attachment order, unlike the child-node
  // traversal above which is intentionally LIFO.
  graph.forEachRenderableOnNode(nodeResult.value(), [&](nuri::RenderableId id) {
    renderables.push_back(id);
  });

  ASSERT_EQ(renderables.size(), 2u);
  EXPECT_EQ(renderables[0], renderableA.value());
  EXPECT_EQ(renderables[1], renderableB.value());
}

TEST(RenderSceneGraphTests, RenderableMaterialOverrideRoundTrip) {
  nuri::RenderScene scene;
  nuri::SceneGraph &graph = scene.graph();
  constexpr nuri::ModelRef kModel = nuri::makeModelRef(1u, 1u);
  constexpr nuri::MaterialRef kMaterial = nuri::makeMaterialRef(2u, 1u);
  constexpr nuri::MaterialRef kOverride = nuri::makeMaterialRef(3u, 1u);

  auto nodeResult = graph.createNode(graph.rootNode(), "Node");
  ASSERT_FALSE(nodeResult.hasError()) << nodeResult.error();
  auto renderableResult =
      graph.addRenderable(nodeResult.value(), kModel, kMaterial);
  ASSERT_FALSE(renderableResult.hasError()) << renderableResult.error();

  nuri::MaterialRef overrideRef = nuri::kInvalidMaterialRef;
  ASSERT_TRUE(graph.getRenderableMaterialOverride(renderableResult.value(),
                                                  overrideRef));
  EXPECT_EQ(overrideRef.value, nuri::kInvalidMaterialRef.value);

  ASSERT_TRUE(
      graph.setRenderableMaterialOverride(renderableResult.value(), kOverride));
  ASSERT_TRUE(graph.getRenderableMaterialOverride(renderableResult.value(),
                                                  overrideRef));
  EXPECT_EQ(overrideRef.value, kOverride.value);

  ASSERT_TRUE(graph.clearRenderableMaterialOverride(renderableResult.value()));
  ASSERT_TRUE(graph.getRenderableMaterialOverride(renderableResult.value(),
                                                  overrideRef));
  EXPECT_EQ(overrideRef.value, nuri::kInvalidMaterialRef.value);
}

TEST(RenderSceneGraphTests, CommitCarriesRenderableMaterialOverride) {
  nuri::RenderScene scene;
  nuri::SceneGraph &graph = scene.graph();
  constexpr nuri::ModelRef kModel = nuri::makeModelRef(1u, 1u);
  constexpr nuri::MaterialRef kMaterial = nuri::makeMaterialRef(2u, 1u);
  constexpr nuri::MaterialRef kOverride = nuri::makeMaterialRef(4u, 1u);

  auto nodeResult = graph.createNode(graph.rootNode(), "Node");
  ASSERT_FALSE(nodeResult.hasError()) << nodeResult.error();
  auto renderableResult =
      graph.addRenderable(nodeResult.value(), kModel, kMaterial);
  ASSERT_FALSE(renderableResult.hasError()) << renderableResult.error();
  ASSERT_TRUE(
      graph.setRenderableMaterialOverride(renderableResult.value(), kOverride));

  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();
  ASSERT_EQ(scene.renderables().size(), 1u);
  EXPECT_EQ(scene.renderables()[0].materialOverride.value, kOverride.value);
}

TEST(RenderSceneGraphTests, DestroyNodeSubtreeClearsRenderableOverrides) {
  nuri::RenderScene scene;
  nuri::SceneGraph &graph = scene.graph();
  constexpr nuri::ModelRef kModel = nuri::makeModelRef(1u, 1u);
  constexpr nuri::MaterialRef kMaterial = nuri::makeMaterialRef(2u, 1u);
  constexpr nuri::MaterialRef kOverride = nuri::makeMaterialRef(5u, 1u);

  auto nodeResult = graph.createNode(graph.rootNode(), "Node");
  ASSERT_FALSE(nodeResult.hasError()) << nodeResult.error();
  auto renderableResult =
      graph.addRenderable(nodeResult.value(), kModel, kMaterial);
  ASSERT_FALSE(renderableResult.hasError()) << renderableResult.error();
  ASSERT_TRUE(
      graph.setRenderableMaterialOverride(renderableResult.value(), kOverride));

  ASSERT_TRUE(graph.destroyNodeSubtree(nodeResult.value()));

  nuri::MaterialRef overrideRef = kOverride;
  EXPECT_FALSE(graph.getRenderableMaterialOverride(renderableResult.value(),
                                                   overrideRef));
}

TEST(RenderSceneGraphTests, DestroyedNodeSlotIsReusedWithNewGeneration) {
  nuri::RenderScene scene;
  nuri::SceneGraph &graph = scene.graph();

  auto firstResult = graph.createNode(graph.rootNode(), "First");
  ASSERT_FALSE(firstResult.hasError()) << firstResult.error();
  const nuri::NodeId first = firstResult.value();

  ASSERT_TRUE(graph.destroyNodeSubtree(first));

  auto secondResult = graph.createNode(graph.rootNode(), "Second");
  ASSERT_FALSE(secondResult.hasError()) << secondResult.error();
  const nuri::NodeId second = secondResult.value();

  EXPECT_EQ(nuri::indexOf(first), nuri::indexOf(second));
  EXPECT_NE(nuri::generationOf(first), nuri::generationOf(second));
}

TEST(RenderSceneGraphTests, RemovedRenderableSlotIsReusedWithNewGeneration) {
  nuri::RenderScene scene;
  nuri::SceneGraph &graph = scene.graph();
  constexpr nuri::ModelRef kModel = nuri::makeModelRef(1u, 1u);
  constexpr nuri::MaterialRef kMaterial = nuri::makeMaterialRef(2u, 1u);

  auto nodeResult = graph.createNode(graph.rootNode(), "Node");
  ASSERT_FALSE(nodeResult.hasError()) << nodeResult.error();

  auto firstResult = graph.addRenderable(nodeResult.value(), kModel, kMaterial);
  ASSERT_FALSE(firstResult.hasError()) << firstResult.error();
  const nuri::RenderableId first = firstResult.value();

  ASSERT_TRUE(graph.removeRenderable(first));

  auto secondResult =
      graph.addRenderable(nodeResult.value(), kModel, kMaterial);
  ASSERT_FALSE(secondResult.hasError()) << secondResult.error();
  const nuri::RenderableId second = secondResult.value();

  EXPECT_EQ(nuri::indexOf(first), nuri::indexOf(second));
  EXPECT_NE(nuri::generationOf(first), nuri::generationOf(second));
}

TEST(RenderSceneGraphTests,
     AddRenderablesInstancedRollsBackCreatedNodesOnFailure) {
  nuri::RenderScene scene;
  nuri::SceneGraph &graph = scene.graph();

  constexpr nuri::ModelRef kModel = nuri::makeModelRef(1u, 1u);
  const std::array<glm::mat4, 2u> modelMatrices = {
      translate(glm::vec3(1.0f, 0.0f, 0.0f)),
      translate(glm::vec3(2.0f, 0.0f, 0.0f)),
  };

  auto addResult = graph.addRenderablesInstanced(
      kModel, nuri::kInvalidMaterialRef, modelMatrices);
  EXPECT_TRUE(addResult.hasError());
  EXPECT_EQ(countChildren(graph, graph.rootNode()), 0u);
}

TEST(RenderSceneGraphTests,
     InstantiatePrefabRollsBackCreatedObjectsAndReturnsPartialMapOnFailure) {
  nuri::RenderScene scene;
  nuri::SceneGraph &graph = scene.graph();

  nuri::ScenePrefab prefab;
  prefab.nodes.resize(2u);
  prefab.nodes[0].name = "PrefabRoot";
  prefab.nodes[1].name = "LightNode";
  prefab.nodes[1].parentIndex = 0u;
  prefab.renderables.push_back(nuri::ScenePrefabRenderable{
      .nodeIndex = 0u,
      .meshIndex = 0u,
      .materialIndex = 0u,
  });
  prefab.lights.push_back(nuri::ScenePrefabLight{
      .nodeIndex = nuri::kInvalidScenePrefabIndex,
      .light = nuri::LightDesc{.type = nuri::LightType::Point},
  });

  nuri::ScenePrefabAssets assets;
  assets.models.push_back(nuri::makeModelRef(1u, 1u));
  assets.materials.push_back(nuri::makeMaterialRef(2u, 1u));

  nuri::SceneInstantiationMap instantiated;
  auto instantiateResult =
      graph.instantiatePrefab(prefab, graph.rootNode(), assets, &instantiated);
  EXPECT_TRUE(instantiateResult.hasError());
  EXPECT_EQ(instantiated.nodes.size(), 2u);
  EXPECT_EQ(instantiated.renderables.size(), 1u);
  EXPECT_TRUE(instantiated.lights.empty());
  EXPECT_EQ(countChildren(graph, graph.rootNode()), 0u);

  nuri::NodeId node = nuri::kInvalidNodeId;
  EXPECT_FALSE(graph.getNodeParent(instantiated.nodes[0], node));
  nuri::MaterialRef material = nuri::kInvalidMaterialRef;
  EXPECT_FALSE(
      graph.getRenderableMaterial(instantiated.renderables[0], material));
}

} // namespace
