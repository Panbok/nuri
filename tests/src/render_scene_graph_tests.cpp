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
