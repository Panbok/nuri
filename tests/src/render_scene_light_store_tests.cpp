#include "tests_pch.h"

#include "nuri/scene/render_scene.h"

#include <bit>

namespace {

TEST(RenderSceneLightStoreTests,
     DirectionalLightProducesPackedDirectionalEntry) {
  nuri::RenderScene scene;

  nuri::LightDesc light{};
  light.type = nuri::LightType::Directional;
  light.name = "Sun";
  light.color = glm::vec3(0.75f, 0.5f, 0.25f);
  light.intensity = 3.5f;

  auto addResult = scene.addLight(light);
  ASSERT_FALSE(addResult.hasError()) << addResult.error();
  EXPECT_EQ(scene.lightTopologyVersion(), 1u);
  EXPECT_EQ(scene.lightTransformVersion(), 1u);
  EXPECT_EQ(scene.packedDirectionalLights().size(), 1u);
  EXPECT_TRUE(scene.packedLocalLights().empty());

  nuri::LightDesc stored{};
  ASSERT_TRUE(scene.getLightDesc(addResult.value(), stored));
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
}

TEST(RenderSceneLightStoreTests,
     DisabledLocalLightStaysOutOfPackedTableUntilEnabled) {
  nuri::RenderScene scene;

  nuri::LightDesc light{};
  light.type = nuri::LightType::Point;
  light.enabled = false;
  light.range = 12.0f;

  auto addResult = scene.addLight(light);
  ASSERT_FALSE(addResult.hasError()) << addResult.error();
  EXPECT_TRUE(scene.packedLocalLights().empty());
  EXPECT_EQ(scene.lightTopologyVersion(), 1u);
  EXPECT_EQ(scene.lightTransformVersion(), 1u);

  light.enabled = true;
  ASSERT_TRUE(scene.updateLight(addResult.value(), light));
  EXPECT_EQ(scene.packedLocalLights().size(), 1u);
  EXPECT_EQ(scene.lightTopologyVersion(), 2u);
  EXPECT_EQ(scene.lightTransformVersion(), 2u);

  light.enabled = false;
  ASSERT_TRUE(scene.updateLight(addResult.value(), light));
  EXPECT_TRUE(scene.packedLocalLights().empty());
  EXPECT_EQ(scene.lightTopologyVersion(), 3u);
  EXPECT_EQ(scene.lightTransformVersion(), 3u);
}

TEST(RenderSceneLightStoreTests,
     SetLightTransformUpdatesPackedDirectionalDataWithoutTopologyChange) {
  nuri::RenderScene scene;

  nuri::LightDesc light{};
  light.type = nuri::LightType::Directional;
  auto addResult = scene.addLight(light);
  ASSERT_FALSE(addResult.hasError()) << addResult.error();

  const glm::quat rotation =
      glm::angleAxis(glm::pi<float>(), glm::vec3(0.0f, 1.0f, 0.0f));
  ASSERT_TRUE(
      scene.setLightTransform(addResult.value(), glm::vec3(2.0f), rotation));

  EXPECT_EQ(scene.lightTopologyVersion(), 1u);
  EXPECT_EQ(scene.lightTransformVersion(), 2u);
  ASSERT_EQ(scene.packedDirectionalLights().size(), 1u);
  const nuri::DirectionalLightGpuData gpu = scene.packedDirectionalLights()[0];
  EXPECT_NEAR(gpu.directionIlluminance.z, 1.0f, 1.0e-4f);
}

TEST(RenderSceneLightStoreTests,
     RemovedLightInvalidatesHandleAndReusesSlotGeneration) {
  nuri::RenderScene scene;

  nuri::LightDesc light{};
  light.type = nuri::LightType::Point;

  auto firstResult = scene.addLight(light);
  ASSERT_FALSE(firstResult.hasError()) << firstResult.error();
  const nuri::LightId first = firstResult.value();

  ASSERT_TRUE(scene.removeLight(first));
  EXPECT_FALSE(scene.getLightDesc(first, light));

  auto secondResult = scene.addLight(light);
  ASSERT_FALSE(secondResult.hasError()) << secondResult.error();
  const nuri::LightId second = secondResult.value();

  EXPECT_EQ(nuri::indexOf(first), nuri::indexOf(second));
  EXPECT_NE(nuri::generationOf(first), nuri::generationOf(second));
}

TEST(RenderSceneLightStoreTests, SpotLightPackingStoresTypeAnglesAndRange) {
  nuri::RenderScene scene;

  nuri::LightDesc light{};
  light.type = nuri::LightType::Spot;
  light.position = glm::vec3(1.0f, 2.0f, 3.0f);
  light.range = 5.0f;
  light.intensity = 7.0f;
  light.innerConeAngleRadians = glm::radians(10.0f);
  light.outerConeAngleRadians = glm::radians(20.0f);

  auto addResult = scene.addLight(light);
  ASSERT_FALSE(addResult.hasError()) << addResult.error();
  ASSERT_EQ(scene.packedLocalLights().size(), 1u);
  const nuri::LocalLightGpuData gpu = scene.packedLocalLights()[0];
  EXPECT_NEAR(gpu.positionRange.x, 1.0f, 1.0e-5f);
  EXPECT_NEAR(gpu.positionRange.y, 2.0f, 1.0e-5f);
  EXPECT_NEAR(gpu.positionRange.z, 3.0f, 1.0e-5f);
  EXPECT_NEAR(gpu.positionRange.w, 5.0f, 1.0e-5f);
  EXPECT_NEAR(gpu.colorIntensity.w, 7.0f, 1.0e-5f);
  EXPECT_EQ(gpu.innerCosTypeEnabledReserved.y,
            static_cast<uint32_t>(nuri::LocalLightGpuType::Spot));
  EXPECT_NEAR(std::bit_cast<float>(gpu.innerCosTypeEnabledReserved.x),
              std::cos(glm::radians(10.0f)), 1.0e-5f);
  EXPECT_NEAR(gpu.directionOuterCos.w, std::cos(glm::radians(20.0f)), 1.0e-5f);
}

TEST(RenderSceneLightStoreTests, LocalLightCapIsEnforced) {
  nuri::RenderScene scene;

  nuri::LightDesc light{};
  light.type = nuri::LightType::Point;
  for (uint32_t i = 0; i < 64u; ++i) {
    auto addResult = scene.addLight(light);
    ASSERT_FALSE(addResult.hasError()) << addResult.error();
  }

  auto overflowResult = scene.addLight(light);
  EXPECT_TRUE(overflowResult.hasError());
}

TEST(RenderSceneLightStoreTests,
     NameOnlyUpdateLeavesVersionsAndPackedGpuDataUnchanged) {
  nuri::RenderScene scene;

  nuri::LightDesc light{};
  light.type = nuri::LightType::Point;
  light.name = "Point A";
  light.position = glm::vec3(1.0f, 2.0f, 3.0f);
  light.color = glm::vec3(0.4f, 0.5f, 0.6f);
  light.intensity = 4.0f;
  light.range = 8.0f;

  auto addResult = scene.addLight(light);
  ASSERT_FALSE(addResult.hasError()) << addResult.error();
  ASSERT_EQ(scene.packedLocalLights().size(), 1u);

  const uint64_t topologyVersionBefore = scene.lightTopologyVersion();
  const uint64_t transformVersionBefore = scene.lightTransformVersion();
  const nuri::LocalLightGpuData gpuBefore = scene.packedLocalLights()[0];

  light.name = "Point B";
  ASSERT_TRUE(scene.updateLight(addResult.value(), light));

  EXPECT_EQ(scene.lightTopologyVersion(), topologyVersionBefore);
  EXPECT_EQ(scene.lightTransformVersion(), transformVersionBefore);
  EXPECT_EQ(0, std::memcmp(&gpuBefore, &scene.packedLocalLights()[0],
                           sizeof(gpuBefore)));

  nuri::LightDesc stored{};
  ASSERT_TRUE(scene.getLightDesc(addResult.value(), stored));
  EXPECT_EQ(stored.name, "Point B");
}

TEST(RenderSceneLightStoreTests,
     ForEachLightEnumeratesLiveLightsWhilePackedIdsTrackEnabledSubset) {
  nuri::RenderScene scene;

  nuri::LightDesc directional{};
  directional.type = nuri::LightType::Directional;
  directional.name = "Sun";

  nuri::LightDesc point{};
  point.type = nuri::LightType::Point;
  point.name = "Point";
  point.enabled = false;

  nuri::LightDesc spot{};
  spot.type = nuri::LightType::Spot;
  spot.name = "Spot";

  auto directionalResult = scene.addLight(directional);
  auto pointResult = scene.addLight(point);
  auto spotResult = scene.addLight(spot);
  ASSERT_FALSE(directionalResult.hasError()) << directionalResult.error();
  ASSERT_FALSE(pointResult.hasError()) << pointResult.error();
  ASSERT_FALSE(spotResult.hasError()) << spotResult.error();

  std::vector<nuri::LightId> enumeratedLights;
  scene.forEachLightId(
      [&](nuri::LightId lightId) { enumeratedLights.push_back(lightId); });

  ASSERT_EQ(enumeratedLights.size(), 3u);
  EXPECT_EQ(enumeratedLights[0], directionalResult.value());
  EXPECT_EQ(enumeratedLights[1], pointResult.value());
  EXPECT_EQ(enumeratedLights[2], spotResult.value());

  ASSERT_EQ(scene.packedDirectionalLightIds().size(), 1u);
  EXPECT_EQ(scene.packedDirectionalLightIds()[0], directionalResult.value());

  ASSERT_EQ(scene.packedLocalLightIds().size(), 1u);
  EXPECT_EQ(scene.packedLocalLightIds()[0], spotResult.value());
}

TEST(RenderSceneLightStoreTests,
     ClearLightsInvalidatesHandlesAndClearsPackedTables) {
  nuri::RenderScene scene;

  nuri::LightDesc directional{};
  directional.type = nuri::LightType::Directional;

  nuri::LightDesc point{};
  point.type = nuri::LightType::Point;

  auto directionalResult = scene.addLight(directional);
  auto pointResult = scene.addLight(point);
  ASSERT_FALSE(directionalResult.hasError()) << directionalResult.error();
  ASSERT_FALSE(pointResult.hasError()) << pointResult.error();

  scene.clearLights();

  EXPECT_TRUE(scene.packedDirectionalLights().empty());
  EXPECT_TRUE(scene.packedLocalLights().empty());
  EXPECT_FALSE(scene.getLightDesc(directionalResult.value(), directional));
  EXPECT_FALSE(scene.getLightDesc(pointResult.value(), point));

  std::vector<nuri::LightId> enumeratedLights;
  scene.forEachLightId(
      [&](nuri::LightId lightId) { enumeratedLights.push_back(lightId); });
  EXPECT_TRUE(enumeratedLights.empty());
}

} // namespace
