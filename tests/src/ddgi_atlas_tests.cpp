#include "tests_pch.h"

#include "nuri/gfx/ddgi/ddgi_atlas.h"

#include <glm/gtc/matrix_transform.hpp>

namespace {

TEST(DDGIAtlasTests, PacksEachAtlasAgainstItsOwnTileExtent) {
  auto irradiance = nuri::packDDGIAtlas(2048u, nuri::kDDGIIrradianceTileExtent,
                                        glm::uvec2(16'384u));
  auto distance = nuri::packDDGIAtlas(2048u, nuri::kDDGIDistanceTileExtent,
                                      glm::uvec2(16'384u));
  ASSERT_TRUE(irradiance.hasValue());
  ASSERT_TRUE(distance.hasValue());
  EXPECT_EQ(irradiance->columns, 46u);
  EXPECT_EQ(irradiance->rows, 45u);
  EXPECT_EQ(irradiance->textureExtent, glm::uvec2(460u, 450u));
  EXPECT_EQ(distance->columns, 46u);
  EXPECT_EQ(distance->rows, 45u);
  EXPECT_EQ(distance->textureExtent, glm::uvec2(828u, 810u));
  EXPECT_EQ(nuri::ddgiAtlasTileCoordinate(47u, *irradiance),
            glm::uvec2(1u, 1u));
}

TEST(DDGIAtlasTests, RejectsPackingThatExceedsDeviceDimensions) {
  auto result = nuri::packDDGIAtlas(65'536u, nuri::kDDGIDistanceTileExtent,
                                    glm::uvec2(512u));
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error(), nuri::DDGIAtlasError::TextureLimitExceeded);
}

TEST(DDGIAtlasTests, EstimatesAllocatedTextureAndStateBytesExactly) {
  auto irradiance = nuri::packDDGIAtlas(2048u, nuri::kDDGIIrradianceTileExtent,
                                        glm::uvec2(16'384u));
  auto distance = nuri::packDDGIAtlas(2048u, nuri::kDDGIDistanceTileExtent,
                                      glm::uvec2(16'384u));
  ASSERT_TRUE(irradiance.hasValue());
  ASSERT_TRUE(distance.hasValue());
  auto estimate = nuri::estimateDDGIMemory(2048u, *irradiance, *distance);
  ASSERT_TRUE(estimate.hasValue());
  EXPECT_EQ(estimate->irradianceBytes, 460u * 450u * 8u);
  EXPECT_EQ(estimate->distanceBytes, 828u * 810u * 4u);
  EXPECT_EQ(estimate->probeStateBytes, 2048u * 32u);
  EXPECT_EQ(estimate->persistentBytes, estimate->irradianceBytes +
                                           estimate->distanceBytes +
                                           estimate->probeStateBytes);
}

TEST(DDGIAtlasTests, CanonicalIndexAndRingMappingRoundTrip) {
  constexpr glm::uvec3 counts(4u, 3u, 2u);
  for (uint32_t index = 0u; index < 24u; ++index) {
    const glm::uvec3 coordinate = nuri::ddgiProbeCoordinate(index, counts);
    EXPECT_EQ(nuri::ddgiProbeIndex(coordinate, counts), index);
  }
  EXPECT_EQ(nuri::ddgiPhysicalProbeCoordinate(glm::uvec3(3u, 2u, 1u),
                                              glm::uvec3(2u, 2u, 1u), counts),
            glm::uvec3(1u, 1u, 0u));
}

TEST(DDGIAtlasTests, OctahedralBordersCopyWrappedInteriorTexels) {
  constexpr uint32_t interior = 8u;
  EXPECT_EQ(nuri::ddgiAtlasBorderCopyCoordinate({0u, 0u}, interior),
            glm::uvec2(8u, 8u));
  EXPECT_EQ(nuri::ddgiAtlasBorderCopyCoordinate({9u, 0u}, interior),
            glm::uvec2(1u, 8u));
  EXPECT_EQ(nuri::ddgiAtlasBorderCopyCoordinate({0u, 9u}, interior),
            glm::uvec2(8u, 1u));
  EXPECT_EQ(nuri::ddgiAtlasBorderCopyCoordinate({9u, 9u}, interior),
            glm::uvec2(1u, 1u));
  EXPECT_EQ(nuri::ddgiAtlasBorderCopyCoordinate({2u, 0u}, interior),
            glm::uvec2(7u, 1u));
  EXPECT_EQ(nuri::ddgiAtlasBorderCopyCoordinate({2u, 9u}, interior),
            glm::uvec2(7u, 8u));
  EXPECT_EQ(nuri::ddgiAtlasBorderCopyCoordinate({0u, 3u}, interior),
            glm::uvec2(1u, 6u));
  EXPECT_EQ(nuri::ddgiAtlasBorderCopyCoordinate({9u, 3u}, interior),
            glm::uvec2(8u, 6u));
}

TEST(DDGIAtlasTests, CameraCellUsesMathematicalFloorAcrossZero) {
  EXPECT_EQ(
      nuri::ddgiCameraCell(glm::vec3(-0.01f, 1.99f, -2.0f), glm::vec3(1.0f)),
      glm::ivec3(-1, 1, -2));
}

TEST(DDGIAtlasTests, ScrollAdvancesRingAndInvalidatesOnlyExposedPlanes) {
  constexpr glm::uvec3 counts(4u, 3u, 2u);
  const nuri::DDGIScrollPlan plan = nuri::makeDDGIScrollPlan(
      glm::ivec3(0), glm::uvec3(0u), glm::ivec3(1, -1, 0), counts);
  EXPECT_TRUE(plan.changed);
  EXPECT_FALSE(plan.fullInvalidation);
  EXPECT_EQ(plan.ringOrigin, glm::uvec3(1u, 2u, 0u));
  EXPECT_TRUE(
      nuri::isDDGINewlyExposedCoordinate(glm::uvec3(3u, 1u, 0u), plan, counts));
  EXPECT_TRUE(
      nuri::isDDGINewlyExposedCoordinate(glm::uvec3(1u, 0u, 1u), plan, counts));
  EXPECT_FALSE(
      nuri::isDDGINewlyExposedCoordinate(glm::uvec3(1u, 1u, 1u), plan, counts));
}

TEST(DDGIAtlasTests, FullAxisScrollInvalidatesEveryCoordinate) {
  constexpr glm::uvec3 counts(4u, 3u, 2u);
  const nuri::DDGIScrollPlan plan = nuri::makeDDGIScrollPlan(
      glm::ivec3(5, 4, 3), glm::uvec3(2u, 1u, 0u), glm::ivec3(5, 7, 3), counts);
  EXPECT_TRUE(plan.fullInvalidation);
  EXPECT_EQ(plan.ringOrigin, glm::uvec3(2u, 1u, 0u));
  for (uint32_t index = 0u; index < 24u; ++index) {
    EXPECT_TRUE(nuri::isDDGINewlyExposedCoordinate(
        nuri::ddgiProbeCoordinate(index, counts), plan, counts));
  }
}

TEST(DDGIAtlasTests, LayoutUsesCountMinusOneExtentsAndRejectsScale) {
  nuri::DDGIVolumeDesc desc{};
  desc.probeCounts = glm::uvec3(4u, 3u, 2u);
  desc.probeSpacing = glm::vec3(2.0f, 3.0f, 4.0f);
  desc.blendDistance = 1.0f;
  auto atlas = nuri::packDDGIAtlas(24u, nuri::kDDGIIrradianceTileExtent,
                                   glm::uvec2(1024u));
  ASSERT_TRUE(atlas.hasValue());
  const glm::mat4 rigid =
      glm::translate(glm::mat4(1.0f), glm::vec3(5.0f, 0.0f, -2.0f)) *
      glm::rotate(glm::mat4(1.0f), glm::half_pi<float>(),
                  glm::vec3(0.0f, 1.0f, 0.0f));
  auto layout = nuri::makeDDGIVolumeLayout(nuri::makeDDGIVolumeId(1u, 1u), desc,
                                           rigid, *atlas, *atlas, 7u);
  ASSERT_TRUE(layout.hasValue());
  EXPECT_EQ(layout->probeCenterHalfExtents, glm::vec3(3.0f, 3.0f, 2.0f));
  EXPECT_EQ(nuri::ddgiLocalProbePosition(*layout, glm::uvec3(0u)),
            -layout->probeCenterHalfExtents);

  auto tracked =
      nuri::makeDDGIVolumeLayout(nuri::makeDDGIVolumeId(1u, 1u), desc, rigid,
                                 *atlas, *atlas, 7u, glm::ivec3(2, -1, 3));
  ASSERT_TRUE(tracked.hasValue());
  EXPECT_EQ(nuri::ddgiLocalProbePosition(*tracked, glm::uvec3(0u)),
            -tracked->probeCenterHalfExtents +
                glm::vec3(tracked->cameraCell) * tracked->probeSpacing);

  const glm::mat4 scaled = glm::scale(rigid, glm::vec3(2.0f, 1.0f, 1.0f));
  auto invalid = nuri::makeDDGIVolumeLayout(nuri::makeDDGIVolumeId(1u, 1u),
                                            desc, scaled, *atlas, *atlas, 8u);
  ASSERT_TRUE(invalid.hasError());
  EXPECT_EQ(invalid.error().reason,
            nuri::DDGIVolumeValidationReason::NonRigidTransform);
}

TEST(DDGIAtlasTests, PriorityBlendPreservesSkyBoundaryRemainder) {
  const nuri::DDGIVolumeBlendWeights weights =
      nuri::ddgiPriorityBlendWeights(0.5f, 0.25f);
  EXPECT_FLOAT_EQ(weights.first, 0.5f);
  EXPECT_FLOAT_EQ(weights.second, 0.125f);
  EXPECT_FLOAT_EQ(weights.sky, 0.375f);
  EXPECT_FLOAT_EQ(weights.first + weights.second + weights.sky, 1.0f);
}

} // namespace
