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

TEST(RenderSceneGraphTests, CommitCarriesRenderableMaterialOverride) {
  nuri::RenderScene scene;
  nuri::SceneGraph &graph = scene.graph();
  constexpr nuri::ModelRef kModel{nuri::packResourceHandle(1u, 1u)};
  constexpr nuri::MaterialRef kMaterial{nuri::packResourceHandle(2u, 1u)};
  constexpr nuri::MaterialRef kOverride{nuri::packResourceHandle(4u, 1u)};

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
  constexpr nuri::ModelRef kModel{nuri::packResourceHandle(1u, 1u)};
  constexpr nuri::MaterialRef kMaterial{nuri::packResourceHandle(2u, 1u)};
  constexpr nuri::MaterialRef kOverride{nuri::packResourceHandle(5u, 1u)};

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

TEST(RenderSceneGraphTests,
     AddRenderablesInstancedRollsBackCreatedNodesOnFailure) {
  nuri::RenderScene scene;
  nuri::SceneGraph &graph = scene.graph();

  constexpr nuri::ModelRef kModel{nuri::packResourceHandle(1u, 1u)};
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
      .meshAssetIndex = 0u,
      .materialAssetIndex = 0u,
  });
  prefab.lights.push_back(nuri::ScenePrefabLight{
      .nodeIndex = nuri::kInvalidScenePrefabIndex,
      .light = nuri::LightDesc{.type = nuri::LightType::Point},
  });

  nuri::ScenePrefabAssets assets;
  assets.models.push_back(nuri::ModelRef{nuri::packResourceHandle(1u, 1u)});
  assets.materials.push_back(
      nuri::MaterialRef{nuri::packResourceHandle(2u, 1u)});

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

TEST(RenderSceneGraphTests,
     PrefabStructureCanPublishBeforeRenderableDependenciesResolve) {
  nuri::RenderScene scene;
  nuri::SceneGraph &graph = scene.graph();

  nuri::ScenePrefab prefab;
  prefab.nodes.resize(2u);
  prefab.nodes[0].name = "ProgressiveRoot";
  prefab.nodes[1].name = "ProgressiveMesh";
  prefab.nodes[1].parentIndex = 0u;
  prefab.nodes[1].morphWeights = {0.25f, 0.75f};
  prefab.renderables.push_back(nuri::ScenePrefabRenderable{
      .nodeIndex = 1u,
      .meshAssetIndex = 0u,
      .materialAssetIndex = 0u,
  });

  nuri::SceneInstantiationMap instantiated;
  auto structure =
      graph.instantiatePrefabStructure(prefab, graph.rootNode(), &instantiated);
  ASSERT_FALSE(structure.hasError()) << structure.error();
  ASSERT_EQ(instantiated.nodes.size(), 2u);
  ASSERT_EQ(instantiated.renderables.size(), 1u);
  EXPECT_FALSE(nuri::isValid(instantiated.renderables[0]));

  auto commit = scene.commit();
  ASSERT_FALSE(commit.hasError()) << commit.error();
  EXPECT_TRUE(scene.renderables().empty());

  constexpr nuri::ModelRef kModel{nuri::packResourceHandle(1u, 1u)};
  constexpr nuri::MaterialRef kMaterial{nuri::packResourceHandle(2u, 1u)};
  auto attached =
      graph.attachPrefabRenderable(prefab, 0u, kModel, kMaterial, instantiated);
  ASSERT_FALSE(attached.hasError()) << attached.error();
  EXPECT_EQ(instantiated.renderables[0], attached.value());

  commit = scene.commit();
  ASSERT_FALSE(commit.hasError()) << commit.error();
  ASSERT_EQ(scene.renderables().size(), 1u);
  EXPECT_EQ(scene.renderables()[0].model.value, kModel.value);
  EXPECT_EQ(scene.renderables()[0].material.value, kMaterial.value);
  ASSERT_EQ(scene.renderables()[0].morphWeights.size(), 2u);
  EXPECT_FLOAT_EQ(scene.renderables()[0].morphWeights[0], 0.25f);
  EXPECT_FLOAT_EQ(scene.renderables()[0].morphWeights[1], 0.75f);
}

TEST(RenderSceneGraphTests, WorldTransformSynchronizationYieldsByNodeBudget) {
  nuri::RenderScene scene;
  nuri::SceneGraph &graph = scene.graph();
  auto parent = graph.createNode(graph.rootNode(), "Parent",
                                 translate(glm::vec3(1.0f, 0.0f, 0.0f)));
  ASSERT_FALSE(parent.hasError()) << parent.error();
  auto child = graph.createNode(parent.value(), "Child",
                                translate(glm::vec3(0.0f, 2.0f, 0.0f)));
  ASSERT_FALSE(child.hasError()) << child.error();
  auto grandchild = graph.createNode(child.value(), "Grandchild",
                                     translate(glm::vec3(0.0f, 0.0f, 3.0f)));
  ASSERT_FALSE(grandchild.hasError()) << grandchild.error();

  EXPECT_FALSE(graph.syncWorldTransformsStep(1u));
  bool complete = false;
  for (uint32_t step = 0u; step < 8u && !complete; ++step) {
    complete = graph.syncWorldTransformsStep(1u);
  }
  ASSERT_TRUE(complete);
  glm::mat4 world(1.0f);
  ASSERT_TRUE(graph.getCachedNodeWorldTransform(grandchild.value(), world));
  EXPECT_TRUE(mat4Near(world, translate(glm::vec3(1.0f, 2.0f, 3.0f))));
}

TEST(RenderSceneGraphTests,
     PrefabStructureConstructionYieldsByOperationBudget) {
  nuri::RenderScene scene;
  nuri::SceneGraph &graph = scene.graph();
  nuri::ScenePrefab prefab;
  prefab.nodes.resize(3u);
  prefab.nodes[0].name = "Root";
  prefab.nodes[1].name = "Child";
  prefab.nodes[1].parentIndex = 0u;
  prefab.nodes[2].name = "Grandchild";
  prefab.nodes[2].parentIndex = 1u;
  prefab.lights.push_back(nuri::ScenePrefabLight{
      .nodeIndex = 2u,
      .light = nuri::LightDesc{.type = nuri::LightType::Point},
  });

  nuri::SceneInstantiationMap map;
  nuri::ScenePrefabStructureCursor cursor;
  uint32_t incompleteSteps = 0u;
  bool complete = false;
  for (uint32_t step = 0u; step < 8u && !complete; ++step) {
    auto result = graph.instantiatePrefabStructureStep(prefab, graph.rootNode(),
                                                       map, cursor, 1u);
    ASSERT_FALSE(result.hasError()) << result.error();
    complete = result.value();
    incompleteSteps += complete ? 0u : 1u;
  }
  EXPECT_TRUE(complete);
  EXPECT_GE(incompleteSteps, 3u);
  EXPECT_EQ(cursor.nextNode, 3u);
  EXPECT_EQ(cursor.nextLight, 1u);
  EXPECT_EQ(countChildren(graph, graph.rootNode()), 1u);
}

TEST(RenderSceneGraphTests, PrefabRejectsParentThatIsNotAnEarlierNode) {
  nuri::RenderScene scene;
  nuri::SceneGraph &graph = scene.graph();
  nuri::ScenePrefab prefab;
  prefab.nodes.resize(1u);
  prefab.nodes[0].name = "MalformedRoot";
  prefab.nodes[0].parentIndex = 7u;

  nuri::SceneInstantiationMap map;
  nuri::ScenePrefabStructureCursor cursor;
  auto result = graph.instantiatePrefabStructureStep(prefab, graph.rootNode(),
                                                     map, cursor, 1u);

  ASSERT_TRUE(result.hasError());
  EXPECT_NE(result.error().find("parent index"), std::string::npos);
  EXPECT_EQ(countChildren(graph, graph.rootNode()), 0u);
}

} // namespace
