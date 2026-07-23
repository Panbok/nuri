#include "tests_pch.h"

#include "nuri/gfx/ddgi/ddgi_dirty_regions.h"

#include <array>
#include <glm/gtc/matrix_transform.hpp>

namespace {

// These tests own pure targeting and submission-lifetime contracts. Renderer
// autotests and snapshots cannot directly or deterministically observe an
// internal dirty record being lost or consumed before submission.

[[nodiscard]] nuri::DDGIDirtyVolume
volume(uint32_t index, nuri::DDGIEffectiveTier tier,
       glm::vec3 worldCenter = glm::vec3(0.0f)) {
  return {
      .localFromWorld = glm::translate(glm::mat4(1.0f), -worldCenter),
      .probeCounts = glm::uvec3(5u),
      .probeSpacing = glm::vec3(2.0f),
      .cameraCell = glm::ivec3(0),
      .queryBias = 0.25f,
      .tier = tier,
      .effectiveIndex = index,
  };
}

[[nodiscard]] nuri::DDGISceneChangeRegion
localChange(glm::vec3 minimum = glm::vec3(-0.1f),
            glm::vec3 maximum = glm::vec3(0.1f)) {
  return {
      .worldBounds = nuri::BoundingBox(minimum, maximum),
      .kind = nuri::DDGISceneChangeKind::DynamicTransform,
      .sourceId = 42u,
      .sourceVersion = 9u,
      .submissionSequence = 13u,
      .boundsKnown = true,
  };
}

TEST(DDGIDirtyRegionsTests,
     LocalChangesTargetOnlyIntersectingVolumeAndProbeInfluence) {
  const std::array volumes{
      volume(0u, nuri::DDGIEffectiveTier::Clipmap0),
      volume(1u, nuri::DDGIEffectiveTier::SceneFitCoarse, glm::vec3(100.0f)),
  };
  nuri::DDGIDirtyRegionRing ring;

  EXPECT_EQ(ring.publish(localChange(), volumes),
            nuri::DDGIDirtyPublishResult::Published);
  ASSERT_TRUE(ring.prepareConsumption(1u));
  const auto regions = ring.pendingRegions();
  ASSERT_EQ(regions.size(), 1u);
  EXPECT_FALSE(regions[0].global);
  EXPECT_EQ(regions[0].affectedVolumeMask, 1u << 0u);
  EXPECT_EQ(regions[0].affectedTierMask,
            1u << static_cast<uint32_t>(nuri::DDGIEffectiveTier::Clipmap0));
  EXPECT_TRUE(regions[0].probeRanges[0].valid);
  EXPECT_EQ(regions[0].probeRanges[0].minimum, glm::uvec3(1u));
  EXPECT_EQ(regions[0].probeRanges[0].maximum, glm::uvec3(3u));
  EXPECT_FALSE(regions[0].probeRanges[1].valid);
  EXPECT_EQ(regions[0].kind, nuri::DDGISceneChangeKind::DynamicTransform);
  EXPECT_EQ(regions[0].sourceId, 42u);
  EXPECT_EQ(regions[0].sourceVersion, 9u);
  EXPECT_EQ(regions[0].submissionSequence, 13u);
  EXPECT_GT(regions[0].generation, 0u);
  EXPECT_EQ(regions[0].worldBounds.min_, glm::vec3(-0.1f));
  EXPECT_EQ(regions[0].worldBounds.max_, glm::vec3(0.1f));
  EXPECT_EQ(regions[0].response, nuri::DDGIDirtyResponseFlags::Wake |
                                     nuri::DDGIDirtyResponseFlags::Irradiance |
                                     nuri::DDGIDirtyResponseFlags::Distance);
}

TEST(DDGIDirtyRegionsTests, OverflowCollapsesWithoutDroppingGlobalResponse) {
  const std::array volumes{
      volume(0u, nuri::DDGIEffectiveTier::Clipmap0),
  };
  nuri::DDGIDirtyRegionRing ring;
  for (uint32_t index = 0u; index < nuri::kMaxDDGIDirtyRegions; ++index) {
    auto change = localChange();
    change.sourceId = index + 1u;
    ASSERT_EQ(ring.publish(change, volumes),
              nuri::DDGIDirtyPublishResult::Published);
  }

  EXPECT_EQ(ring.publish(localChange(), volumes),
            nuri::DDGIDirtyPublishResult::CollapsedToGlobal);
  EXPECT_EQ(ring.unconsumedCount(), 1u);
  ASSERT_TRUE(ring.prepareConsumption(2u));
  const auto regions = ring.pendingRegions();
  ASSERT_EQ(regions.size(), 1u);
  EXPECT_TRUE(regions[0].global);
  EXPECT_EQ(regions[0].response, nuri::DDGIDirtyResponseFlags::All);
  EXPECT_EQ(regions[0].affectedVolumeMask,
            (1u << nuri::kMaxDDGIEffectiveVolumes) - 1u);
  EXPECT_EQ(ring.metrics().produced,
            static_cast<uint64_t>(nuri::kMaxDDGIDirtyRegions) + 1u);
  EXPECT_EQ(ring.metrics().overflowed, 1u);
}

TEST(DDGIDirtyRegionsTests,
     ConsumptionCommitsOnlyPreparedGenerationsAndAbandonRetries) {
  const std::array volumes{
      volume(0u, nuri::DDGIEffectiveTier::Clipmap0),
  };
  nuri::DDGIDirtyRegionRing ring;
  ASSERT_EQ(ring.publish(localChange(), volumes),
            nuri::DDGIDirtyPublishResult::Published);
  ASSERT_TRUE(ring.prepareConsumption(7u));
  const uint64_t firstGeneration = ring.pendingRegions()[0].generation;

  ring.abandonConsumption(7u);
  EXPECT_EQ(ring.unconsumedCount(), 1u);
  ASSERT_TRUE(ring.prepareConsumption(8u));
  ASSERT_EQ(ring.pendingRegions().size(), 1u);
  EXPECT_EQ(ring.pendingRegions()[0].generation, firstGeneration);

  auto later = localChange(glm::vec3(0.2f), glm::vec3(0.3f));
  later.sourceVersion = 10u;
  ASSERT_EQ(ring.publish(later, volumes),
            nuri::DDGIDirtyPublishResult::Published);
  ASSERT_TRUE(ring.commitConsumption(8u));
  EXPECT_EQ(ring.unconsumedCount(), 1u);
  ASSERT_TRUE(ring.prepareConsumption(9u));
  ASSERT_EQ(ring.pendingRegions().size(), 1u);
  EXPECT_EQ(ring.pendingRegions()[0].sourceVersion, 10u);
  EXPECT_GT(ring.pendingRegions()[0].generation, firstGeneration);
}

TEST(DDGIDirtyRegionsTests,
     GlobalRadiometricAndUnknownDeformationAreAlwaysGlobal) {
  const std::array volumes{
      volume(0u, nuri::DDGIEffectiveTier::Clipmap0),
  };
  nuri::DDGIDirtyRegionRing ring;
  auto radiometric = localChange();
  radiometric.kind = nuri::DDGISceneChangeKind::GlobalRadiometric;
  ASSERT_EQ(ring.publish(radiometric, volumes),
            nuri::DDGIDirtyPublishResult::Published);
  auto deformation = localChange();
  deformation.kind = nuri::DDGISceneChangeKind::Deformation;
  deformation.boundsKnown = false;
  ASSERT_EQ(ring.publish(deformation, volumes),
            nuri::DDGIDirtyPublishResult::Published);

  ASSERT_TRUE(ring.prepareConsumption(3u));
  const auto regions = ring.pendingRegions();
  ASSERT_EQ(regions.size(), 2u);
  EXPECT_TRUE(regions[0].global);
  EXPECT_EQ(regions[0].response, nuri::DDGIDirtyResponseFlags::Irradiance);
  EXPECT_TRUE(regions[1].global);
  EXPECT_EQ(regions[1].response, nuri::DDGIDirtyResponseFlags::Wake |
                                     nuri::DDGIDirtyResponseFlags::Irradiance |
                                     nuri::DDGIDirtyResponseFlags::Distance);
}

TEST(DDGIDirtyRegionsTests,
     LocalLightChangesUseTheUnionOfOldAndNewInfluenceBounds) {
  const nuri::LocalLightGpuData previous = nuri::packPointLight(
      glm::vec3(-4.0f, 1.0f, 0.0f), glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
      glm::vec3(1.0f), 10.0f, 2.0f, true);
  const nuri::LocalLightGpuData current = nuri::packPointLight(
      glm::vec3(3.0f, 1.0f, 0.0f), glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
      glm::vec3(1.0f), 10.0f, 4.0f, true);
  const nuri::DDGISceneChangeRegion moved =
      nuri::makeDDGILocalLightChangeRegion(&previous, &current, 17u, 4u);
  EXPECT_TRUE(moved.boundsKnown);
  EXPECT_EQ(moved.worldBounds.min_, glm::vec3(-6.0f, -3.0f, -4.0f));
  EXPECT_EQ(moved.worldBounds.max_, glm::vec3(7.0f, 5.0f, 4.0f));
  EXPECT_EQ(moved.kind, nuri::DDGISceneChangeKind::LocalLight);
  EXPECT_EQ(moved.sourceId, 17u);
  EXPECT_EQ(moved.sourceVersion, 4u);

  const nuri::LocalLightGpuData unbounded =
      nuri::packPointLight(glm::vec3(0.0f), glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
                           glm::vec3(1.0f), 1.0f, 0.0f, true);
  const nuri::DDGISceneChangeRegion global =
      nuri::makeDDGILocalLightChangeRegion(nullptr, &unbounded, 18u, 5u);
  EXPECT_FALSE(global.boundsKnown);
}

} // namespace
