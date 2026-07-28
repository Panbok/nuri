#include "tests_pch.h"

#include "nuri/scene/render_scene.h"

#include <bit>

namespace {

[[nodiscard]] glm::mat4 makeNodeTransform(const glm::vec3 &position,
                                          const glm::quat &rotation) {
  return glm::translate(glm::mat4(1.0f), position) * glm::mat4_cast(rotation);
}

TEST(RenderSceneLightStoreTests,
     DirectionalLightProducesPackedDirectionalEntryAfterCommit) {
  nuri::RenderScene scene;
  nuri::SceneGraph &graph = scene.graph();

  nuri::LightDesc light{};
  light.type = nuri::LightType::Directional;
  light.name = "Sun";
  light.color = glm::vec3(0.75f, 0.5f, 0.25f);
  light.intensity = 3.5f;
  light.angularRadiusDegrees = 1.25f;
  auto addResult = graph.addLight(graph.rootNode(), light);
  ASSERT_FALSE(addResult.hasError()) << addResult.error();

  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();
  EXPECT_TRUE(commitResult.value());
  EXPECT_EQ(scene.lightTopologyVersion(), 1u);
  EXPECT_EQ(scene.lightTransformVersion(), 1u);
  EXPECT_EQ(scene.packedDirectionalLights().size(), 1u);
  EXPECT_TRUE(scene.packedLocalLights().empty());

  nuri::LightDesc stored{};
  ASSERT_TRUE(graph.getLightDesc(addResult.value(), stored));
  EXPECT_EQ(stored.type, nuri::LightType::Directional);
  EXPECT_EQ(stored.name, "Sun");
  EXPECT_EQ(stored.range, 0.0f);

  const nuri::DirectionalLightGpuData gpu = scene.packedDirectionalLights()[0];
  EXPECT_NEAR(gpu.directionIlluminance.x, 0.0f, 1.0e-5f);
  EXPECT_NEAR(gpu.directionIlluminance.y, 0.0f, 1.0e-5f);
  EXPECT_NEAR(gpu.directionIlluminance.z, -1.0f, 1.0e-5f);
  EXPECT_NEAR(gpu.directionIlluminance.w, 3.5f, 1.0e-5f);
  EXPECT_NEAR(gpu.colorReserved.x, 0.75f, 1.0e-5f);
  EXPECT_NEAR(gpu.colorReserved.y, 0.5f, 1.0e-5f);
  EXPECT_NEAR(gpu.colorReserved.z, 0.25f, 1.0e-5f);
  EXPECT_NEAR(gpu.colorReserved.w, glm::radians(1.25f), 1.0e-5f);
}

TEST(RenderSceneLightStoreTests,
     EnvironmentVersionIncrementsOnlyOnActualHandleChanges) {
  nuri::RenderScene scene;
  EXPECT_EQ(scene.environmentVersion(), 0u);

  scene.setEnvironment(nuri::EnvironmentHandles{});
  EXPECT_EQ(scene.environmentVersion(), 0u);

  nuri::EnvironmentHandles environment{};
  environment.cubemap = nuri::TextureRef{nuri::packResourceHandle(1u, 1u)};
  scene.setEnvironment(environment);
  EXPECT_EQ(scene.environmentVersion(), 1u);

  scene.setEnvironment(environment);
  EXPECT_EQ(scene.environmentVersion(), 1u);

  environment.irradiance = nuri::TextureRef{nuri::packResourceHandle(2u, 1u)};
  scene.setEnvironment(environment);
  EXPECT_EQ(scene.environmentVersion(), 2u);

  scene.setEnvironment(nuri::EnvironmentHandles{});
  EXPECT_EQ(scene.environmentVersion(), 3u);

  scene.setEnvironment(nuri::EnvironmentHandles{});
  EXPECT_EQ(scene.environmentVersion(), 3u);
}

TEST(RenderSceneLightStoreTests,
     MovingLightNodeUpdatesPackedDirectionalDataWithoutTopologyChange) {
  nuri::RenderScene scene;
  nuri::SceneGraph &graph = scene.graph();

  auto nodeResult = graph.createNode(graph.rootNode(), "Sun");
  ASSERT_FALSE(nodeResult.hasError()) << nodeResult.error();

  nuri::LightDesc light{};
  light.type = nuri::LightType::Directional;
  auto addResult = graph.addLight(nodeResult.value(), light);
  ASSERT_FALSE(addResult.hasError()) << addResult.error();
  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();

  const glm::quat rotation =
      glm::angleAxis(glm::pi<float>(), glm::vec3(0.0f, 1.0f, 0.0f));
  ASSERT_TRUE(graph.setNodeLocalTransform(
      nodeResult.value(), makeNodeTransform(glm::vec3(2.0f), rotation)));
  commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();

  EXPECT_EQ(scene.lightTopologyVersion(), 1u);
  EXPECT_EQ(scene.lightTransformVersion(), 2u);
  ASSERT_EQ(scene.packedDirectionalLights().size(), 1u);
  const nuri::DirectionalLightGpuData gpu = scene.packedDirectionalLights()[0];
  EXPECT_NEAR(gpu.directionIlluminance.z, 1.0f, 1.0e-4f);
}

TEST(RenderSceneLightStoreTests,
     RemovedLightInvalidatesHandleAndReusesSlotGeneration) {
  nuri::RenderScene scene;
  nuri::SceneGraph &graph = scene.graph();

  nuri::LightDesc light{};
  light.type = nuri::LightType::Point;

  auto firstResult = graph.addLight(graph.rootNode(), light);
  ASSERT_FALSE(firstResult.hasError()) << firstResult.error();
  const nuri::LightId first = firstResult.value();

  ASSERT_TRUE(graph.removeLight(first));
  EXPECT_FALSE(graph.getLightDesc(first, light));

  auto secondResult = graph.addLight(graph.rootNode(), light);
  ASSERT_FALSE(secondResult.hasError()) << secondResult.error();
  const nuri::LightId second = secondResult.value();

  EXPECT_EQ(nuri::indexOf(first), nuri::indexOf(second));
  EXPECT_NE(nuri::generationOf(first), nuri::generationOf(second));
}

TEST(RenderSceneLightStoreTests, LocalLightCapIsEnforced) {
  nuri::RenderScene scene;
  nuri::SceneGraph &graph = scene.graph();

  nuri::LightDesc light{};
  light.type = nuri::LightType::Point;
  for (uint32_t i = 0; i < 128u; ++i) {
    auto addResult = graph.addLight(graph.rootNode(), light);
    ASSERT_FALSE(addResult.hasError()) << addResult.error();
  }

  auto overflowResult = graph.addLight(graph.rootNode(), light);
  EXPECT_TRUE(overflowResult.hasError());
}

} // namespace
