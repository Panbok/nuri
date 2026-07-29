#include "tests_pch.h"

#include "nuri/gfx/ddgi/ddgi_coverage.h"

#include <array>
#include <limits>

namespace {

[[nodiscard]] nuri::DDGISceneCoverageBounds completeBounds(glm::vec3 minimum,
                                                           glm::vec3 maximum) {
  return {.bounds = nuri::BoundingBox(minimum, maximum),
          .generation = 7u,
          .valid = true,
          .complete = true};
}

TEST(DDGICoverageTests, SanitizationPreservesManualDefaultAndClampsProfile) {
  nuri::DDGICoverageSettings settings{};
  EXPECT_EQ(settings.mode, nuri::DDGICoverageMode::Manual);

  settings.mode = static_cast<nuri::DDGICoverageMode>(255u);
  settings.constraintPolicy =
      static_cast<nuri::DDGICoverageConstraintPolicy>(255u);
  settings.sceneBoundsSource = static_cast<nuri::DDGISceneBoundsSource>(255u);
  settings.cascadeCount = 99u;
  settings.cascadeProbeCounts = glm::uvec3(1u, 99u, 20u);
  settings.requestedNearSpacing =
      glm::vec3(0.0f, std::numeric_limits<float>::infinity(), 200.0f);
  settings.spacingRatio = 99.0f;
  settings.requestedCoverageHalfExtents = glm::vec3(-1.0f);
  settings.scenePaddingCells = 0u;
  settings.transitionCells = 99u;

  nuri::sanitizeDDGICoverageSettings(settings);

  EXPECT_EQ(settings.mode, nuri::DDGICoverageMode::Manual);
  EXPECT_EQ(settings.constraintPolicy,
            nuri::DDGICoverageConstraintPolicy::PreserveCoverage);
  EXPECT_EQ(settings.sceneBoundsSource,
            nuri::DDGISceneBoundsSource::ActivationSnapshot);
  EXPECT_EQ(settings.cascadeCount, nuri::kMaxDDGIClipmapCascades);
  EXPECT_LE(nuri::ddgiProbeCount(settings.cascadeProbeCounts),
            nuri::kDDGIMaxProbeCount);
  for (uint32_t axis = 0u; axis < 3u; ++axis) {
    EXPECT_GT(settings.cascadeProbeCounts[axis], 2u * settings.transitionCells);
  }
  EXPECT_EQ(settings.requestedNearSpacing, glm::vec3(0.1f, 1.5f, 100.0f));
  EXPECT_FLOAT_EQ(settings.spacingRatio, 4.0f);
  EXPECT_EQ(settings.requestedCoverageHalfExtents,
            glm::vec3(55.0f, 30.0f, 55.0f));
  EXPECT_EQ(settings.scenePaddingCells, 1u);
  EXPECT_EQ(settings.transitionCells, 3u);
}

TEST(DDGICoverageTests, SceneFitUsesSmallestCountsAndContainsEveryCorner) {
  nuri::DDGICoverageSettings settings{};
  settings.requestedNearSpacing = glm::vec3(2.0f);
  settings.scenePaddingCells = 1u;
  auto result = nuri::solveDDGISceneFit(
      completeBounds(glm::vec3(-10.0f), glm::vec3(10.0f)), settings, {});

  ASSERT_TRUE(result.hasValue());
  EXPECT_EQ(result->probeCounts, glm::uvec3(13u));
  EXPECT_EQ(result->achievedSpacing, glm::vec3(2.0f));
  EXPECT_EQ(result->interiorHalfExtents, glm::vec3(10.0f));
  EXPECT_TRUE(result->fullCoverage);
  EXPECT_FLOAT_EQ(result->requestedVolumeCoverage, 1.0f);
  EXPECT_TRUE(nuri::ddgiBoundsContain(result->achievedInteriorBounds,
                                      result->requestedBounds));
  EXPECT_EQ(result->limitingConstraint, nuri::DDGICoverageLimit::None);
}

TEST(DDGICoverageTests, PreserveCoverageCoarsensSpacingWithoutClippingBounds) {
  nuri::DDGICoverageSettings settings{};
  settings.constraintPolicy =
      nuri::DDGICoverageConstraintPolicy::PreserveCoverage;
  settings.requestedNearSpacing = glm::vec3(1.0f);
  settings.scenePaddingCells = 1u;
  auto result = nuri::solveDDGISceneFit(
      completeBounds(glm::vec3(-100.0f), glm::vec3(100.0f)), settings, {});

  ASSERT_TRUE(result.hasValue());
  EXPECT_TRUE(result->fullCoverage);
  EXPECT_GT(result->achievedSpacing.x, settings.requestedNearSpacing.x);
  EXPECT_LE(result->probeCounts.x, nuri::kDDGIMaxProbeCountPerAxis);
  EXPECT_EQ(result->limitingConstraint,
            nuri::DDGICoverageLimit::TotalProbeCount);
}

TEST(DDGICoverageTests, PreserveNearSpacingReportsPartialAchievedCoverage) {
  nuri::DDGICoverageSettings settings{};
  settings.constraintPolicy =
      nuri::DDGICoverageConstraintPolicy::PreserveNearSpacing;
  settings.requestedNearSpacing = glm::vec3(1.0f);
  settings.scenePaddingCells = 1u;
  auto result = nuri::solveDDGISceneFit(
      completeBounds(glm::vec3(-100.0f), glm::vec3(100.0f)), settings, {});

  ASSERT_TRUE(result.hasValue());
  EXPECT_EQ(result->achievedSpacing, settings.requestedNearSpacing);
  EXPECT_FALSE(result->fullCoverage);
  EXPECT_GT(result->requestedVolumeCoverage, 0.0f);
  EXPECT_LT(result->requestedVolumeCoverage, 1.0f);
  EXPECT_EQ(result->limitingConstraint,
            nuri::DDGICoverageLimit::TotalProbeCount);
}

TEST(DDGICoverageTests, RejectUnsatisfiedReturnsTheTypedLimitingConstraint) {
  nuri::DDGICoverageSettings settings{};
  settings.constraintPolicy =
      nuri::DDGICoverageConstraintPolicy::RejectUnsatisfied;
  settings.requestedNearSpacing = glm::vec3(1.0f);
  settings.scenePaddingCells = 1u;
  auto result = nuri::solveDDGISceneFit(
      completeBounds(glm::vec3(-100.0f), glm::vec3(100.0f)), settings, {});

  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().limit, nuri::DDGICoverageLimit::ProbeCountAxis);
  EXPECT_EQ(result.error().axis, 0u);
}

TEST(DDGICoverageTests, RejectsIncompleteBoundsAndImpossibleMemoryBudget) {
  nuri::DDGICoverageSettings settings{};
  nuri::DDGISceneCoverageBounds incomplete =
      completeBounds(glm::vec3(-1.0f), glm::vec3(1.0f));
  incomplete.complete = false;
  auto incompleteResult = nuri::solveDDGISceneFit(incomplete, settings, {});
  ASSERT_TRUE(incompleteResult.hasError());
  EXPECT_EQ(incompleteResult.error().limit,
            nuri::DDGICoverageLimit::SceneBoundsIncomplete);

  nuri::DDGICoverageSolveLimits limits{};
  limits.maxPersistentBytes = 1u;
  limits.maxReplacementPeakBytes = 1u;
  auto memoryResult = nuri::solveDDGISceneFit(
      completeBounds(glm::vec3(-1.0f), glm::vec3(1.0f)), settings, limits);
  ASSERT_TRUE(memoryResult.hasError());
  EXPECT_EQ(memoryResult.error().limit,
            nuri::DDGICoverageLimit::PersistentMemory);
}

TEST(DDGICoverageTests, IndependentMemoryCeilingsNeedNotBeOrdered) {
  nuri::DDGICoverageSettings settings{};
  nuri::DDGICoverageSolveLimits limits{};
  limits.maxPersistentBytes = 4u * 1024u * 1024u;
  limits.maxReplacementPeakBytes = 2u * 1024u * 1024u;

  auto result = nuri::solveDDGISceneFit(
      completeBounds(glm::vec3(-1.0f), glm::vec3(1.0f)), settings, limits);

  ASSERT_TRUE(result.hasValue());
  EXPECT_LE(result->memory.persistentBytes, limits.maxReplacementPeakBytes);
}

TEST(DDGICoverageTests, DegenerateAxisUsesZeroInteriorCells) {
  nuri::DDGICoverageSettings settings{};
  settings.requestedNearSpacing = glm::vec3(2.0f);
  settings.scenePaddingCells = 1u;

  auto result =
      nuri::solveDDGISceneFit(completeBounds(glm::vec3(-10.0f, 0.0f, -10.0f),
                                             glm::vec3(10.0f, 0.0f, 10.0f)),
                              settings, {});

  ASSERT_TRUE(result.hasValue());
  EXPECT_EQ(result->probeCounts, glm::uvec3(13u, 3u, 13u));
  EXPECT_FLOAT_EQ(result->interiorHalfExtents.y, 0.0f);
  EXPECT_TRUE(result->fullCoverage);
}

TEST(DDGICoverageTests, RejectsDerivedBoundsOverflowAndUsesWideRefreshMath) {
  nuri::DDGICoverageSettings settings{};
  const float maximum = std::numeric_limits<float>::max();
  auto boundsResult = nuri::solveDDGISceneFit(
      completeBounds(glm::vec3(-maximum), glm::vec3(maximum)), settings, {});
  ASSERT_TRUE(boundsResult.hasError());
  EXPECT_EQ(boundsResult.error().limit,
            nuri::DDGICoverageLimit::SceneBoundsUnavailable);

  nuri::DDGICoverageSolveLimits limits{};
  limits.maxProbeUpdatesPerFrame = std::numeric_limits<uint32_t>::max();
  auto refreshResult = nuri::solveDDGISceneFit(
      completeBounds(glm::vec3(-10.0f), glm::vec3(10.0f)), settings, limits);
  ASSERT_TRUE(refreshResult.hasValue());
  EXPECT_EQ(refreshResult->estimatedMinimumRefreshFrames, 1u);
}

TEST(DDGICoverageTests, ClipmapPoliciesReportRequestedAndAchievedCoverage) {
  nuri::DDGICoverageSettings settings{};
  settings.mode = nuri::DDGICoverageMode::CameraClipmaps;
  settings.requestedNearSpacing = glm::vec3(1.5f);
  settings.requestedCoverageHalfExtents = glm::vec3(55.0f, 30.0f, 55.0f);

  auto preserveCoverage = nuri::solveDDGIClipmaps(settings, {});
  ASSERT_TRUE(preserveCoverage.hasValue());
  EXPECT_EQ(preserveCoverage->cascadeCount, 3u);
  EXPECT_TRUE(preserveCoverage->fullCoverage);
  EXPECT_GT(preserveCoverage->achievedNearSpacing.y,
            settings.requestedNearSpacing.y);
  EXPECT_EQ(preserveCoverage->limitingConstraint,
            nuri::DDGICoverageLimit::CoverageExtents);
  for (uint32_t cascade = 1u; cascade < preserveCoverage->cascadeCount;
       ++cascade) {
    EXPECT_EQ(preserveCoverage->cascades[cascade].spacing,
              preserveCoverage->cascades[cascade - 1u].spacing * 2.0f);
  }

  settings.constraintPolicy =
      nuri::DDGICoverageConstraintPolicy::PreserveNearSpacing;
  auto preserveSpacing = nuri::solveDDGIClipmaps(settings, {});
  ASSERT_TRUE(preserveSpacing.hasValue());
  EXPECT_EQ(preserveSpacing->achievedNearSpacing,
            settings.requestedNearSpacing);
  EXPECT_FALSE(preserveSpacing->fullCoverage);

  settings.constraintPolicy =
      nuri::DDGICoverageConstraintPolicy::RejectUnsatisfied;
  auto reject = nuri::solveDDGIClipmaps(settings, {});
  ASSERT_TRUE(reject.hasError());
  EXPECT_EQ(reject.error().limit, nuri::DDGICoverageLimit::CoverageExtents);
}

TEST(DDGICoverageTests,
     GeneratedKeysIgnoreCameraCellButTrackProfileGeneration) {
  nuri::DDGICoverageResolveInput input{};
  input.sceneId = 42u;
  input.coverageGeneration = 7u;
  input.settings.mode = nuri::DDGICoverageMode::CameraClipmaps;
  input.settings.requestedCoverageHalfExtents = glm::vec3(40.0f, 20.0f, 40.0f);
  input.settings.includeAuthoredVolumes = false;
  input.cameraWorldPosition = glm::vec3(-0.1f);
  nuri::DDGIEffectiveVolumePlan negative;
  auto negativeResult = nuri::resolveDDGIEffectiveVolumePlan(input, negative);
  ASSERT_TRUE(negativeResult.hasValue());
  ASSERT_EQ(negative.volumeCount, input.settings.cascadeCount);
  EXPECT_EQ(negative.volumes[0].cameraCell, glm::ivec3(-1));

  input.cameraWorldPosition = glm::vec3(10.0f);
  nuri::DDGIEffectiveVolumePlan positive;
  auto positiveResult = nuri::resolveDDGIEffectiveVolumePlan(input, positive);
  ASSERT_TRUE(positiveResult.hasValue());
  ASSERT_EQ(positive.volumeCount, negative.volumeCount);
  for (uint32_t index = 0u; index < positive.volumeCount; ++index) {
    EXPECT_EQ(positive.volumes[index].key, negative.volumes[index].key);
  }
  EXPECT_NE(positive.volumes[0].cameraCell, negative.volumes[0].cameraCell);

  ++input.coverageGeneration;
  nuri::DDGIEffectiveVolumePlan changedProfile;
  ASSERT_TRUE(
      nuri::resolveDDGIEffectiveVolumePlan(input, changedProfile).hasValue());
  EXPECT_NE(changedProfile.volumes[0].key, positive.volumes[0].key);
}

TEST(DDGICoverageTests, AuthoredKeysAreIsolatedBySceneActivation) {
  nuri::RenderDDGIVolume authored{};
  authored.id = nuri::DDGIVolumeId::fromParts(3u, 1u);
  std::array volumes{authored};
  nuri::DDGICoverageResolveInput input{};
  input.sceneId = 41u;
  input.authoredVolumes = volumes;
  input.settings.mode = nuri::DDGICoverageMode::Manual;

  nuri::DDGIEffectiveVolumePlan first;
  ASSERT_TRUE(nuri::resolveDDGIEffectiveVolumePlan(input, first).hasValue());
  ASSERT_EQ(first.volumeCount, 1u);

  input.sceneId = 42u;
  nuri::DDGIEffectiveVolumePlan second;
  ASSERT_TRUE(nuri::resolveDDGIEffectiveVolumePlan(input, second).hasValue());
  ASSERT_EQ(second.volumeCount, 1u);
  EXPECT_NE(first.volumes[0].key, second.volumes[0].key);
  EXPECT_EQ(first.volumes[0].key.authoredId, second.volumes[0].key.authoredId);
}

TEST(DDGICoverageTests, EffectivePlanOrdersAndReportsEveryOmittedKey) {
  std::array<nuri::RenderDDGIVolume, 10u> authored{};
  for (uint32_t index = 0u; index < authored.size(); ++index) {
    authored[index].id = nuri::DDGIVolumeId::fromParts(index, 1u);
    authored[index].name = "authored";
    authored[index].priority = 100 - static_cast<int32_t>(index);
  }
  nuri::DDGICoverageResolveInput input{};
  input.sceneId = 9u;
  input.coverageGeneration = 3u;
  input.authoredVolumes = authored;
  input.settings.mode = nuri::DDGICoverageMode::CameraClipmaps;
  input.settings.requestedCoverageHalfExtents = glm::vec3(40.0f, 20.0f, 40.0f);
  input.settings.constraintPolicy =
      nuri::DDGICoverageConstraintPolicy::PreserveCoverage;
  nuri::DDGIEffectiveVolumePlan plan;
  auto result = nuri::resolveDDGIEffectiveVolumePlan(input, plan);

  ASSERT_TRUE(result.hasValue());
  EXPECT_TRUE(plan.ready);
  EXPECT_TRUE(plan.overflowed);
  EXPECT_EQ(plan.candidateCount, 13u);
  EXPECT_EQ(plan.volumeCount, nuri::kMaxDDGIEffectiveVolumes);
  EXPECT_EQ(plan.omittedKeys.size(), 5u);
  uint64_t activePersistentBytes = 0u;
  for (const nuri::DDGIEffectiveVolume &volume : plan.activeVolumes()) {
    activePersistentBytes += volume.memory.persistentBytes;
  }
  EXPECT_EQ(plan.persistentBytes, activePersistentBytes);
  for (uint32_t index = 1u; index < plan.volumeCount; ++index) {
    EXPECT_GE(plan.volumes[index - 1u].priority, plan.volumes[index].priority);
  }

  input.settings.constraintPolicy =
      nuri::DDGICoverageConstraintPolicy::RejectUnsatisfied;
  nuri::DDGIEffectiveVolumePlan rejected;
  auto reject = nuri::resolveDDGIEffectiveVolumePlan(input, rejected);
  ASSERT_TRUE(reject.hasError());
  EXPECT_FALSE(rejected.ready);
  EXPECT_EQ(rejected.error.limit,
            nuri::DDGICoverageLimit::EffectiveVolumeCapacity);
  EXPECT_EQ(rejected.omittedKeys.size(), 5u);
}

TEST(DDGICoverageTests, HybridPlanPublishesClipmapsAndContainingSceneFit) {
  nuri::DDGICoverageResolveInput input{};
  input.sceneId = 11u;
  input.coverageGeneration = 1u;
  input.sceneBounds = completeBounds(glm::vec3(-20.0f), glm::vec3(20.0f));
  input.settings.mode = nuri::DDGICoverageMode::Hybrid;
  input.settings.includeAuthoredVolumes = false;
  nuri::DDGIEffectiveVolumePlan plan;
  auto result = nuri::resolveDDGIEffectiveVolumePlan(input, plan);

  ASSERT_TRUE(result.hasValue());
  EXPECT_TRUE(plan.ready);
  EXPECT_TRUE(plan.fullSceneCoverage);
  EXPECT_EQ(plan.volumeCount, input.settings.cascadeCount + 1u);
  EXPECT_TRUE(nuri::ddgiBoundsContain(plan.sceneFit.achievedInteriorBounds,
                                      input.sceneBounds.bounds));
  const glm::vec3 outerClipmapSpacing =
      plan.clipmaps.cascades[plan.clipmaps.cascadeCount - 1u].spacing;
  EXPECT_EQ(plan.sceneFit.requestedSpacing, outerClipmapSpacing);
  EXPECT_TRUE(glm::all(glm::greaterThanEqual(plan.sceneFit.achievedSpacing,
                                             outerClipmapSpacing)));
  EXPECT_EQ(plan.volumes[plan.volumeCount - 1u].key.kind,
            nuri::DDGIEffectiveVolumeKind::SceneFit);
  EXPECT_GT(plan.persistentBytes, 0u);
  EXPECT_GE(plan.replacementPeakBytes, plan.persistentBytes);
}

TEST(DDGICoverageTests, RedundancyRequiresFullCoverageAndNoDensityBenefit) {
  nuri::DDGIEffectiveVolume sceneFit{
      .key = {.kind = nuri::DDGIEffectiveVolumeKind::SceneFit},
      .probeSpacing = glm::vec3(1.0f),
      .worldFromLocal = glm::mat4(1.0f),
      .probeCenterHalfExtents = glm::vec3(10.0f),
      .fadeEndHalfExtents = glm::vec3(10.0f),
  };
  nuri::DDGIEffectiveVolume authored{
      .key = {.kind = nuri::DDGIEffectiveVolumeKind::Authored},
      .probeSpacing = glm::vec3(2.0f),
      .worldFromLocal =
          glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 0.0f, 0.0f)),
      .probeCenterHalfExtents = glm::vec3(3.0f),
      .fadeEndHalfExtents = glm::vec3(3.0f),
  };

  const nuri::DDGIRedundancyAnalysis redundant =
      nuri::analyzeDDGIVolumeRedundancy(authored, sceneFit);
  EXPECT_TRUE(redundant.fullyCovered);
  EXPECT_TRUE(redundant.densityRedundant);
  EXPECT_TRUE(redundant.fullyRedundant);

  authored.probeSpacing = glm::vec3(0.5f);
  const nuri::DDGIRedundancyAnalysis denser =
      nuri::analyzeDDGIVolumeRedundancy(authored, sceneFit);
  EXPECT_TRUE(denser.fullyCovered);
  EXPECT_FALSE(denser.densityRedundant);
  EXPECT_FALSE(denser.fullyRedundant);

  authored.probeSpacing = glm::vec3(2.0f);
  authored.worldFromLocal =
      glm::translate(glm::mat4(1.0f), glm::vec3(20.0f, 0.0f, 0.0f));
  const nuri::DDGIRedundancyAnalysis outside =
      nuri::analyzeDDGIVolumeRedundancy(authored, sceneFit);
  EXPECT_FALSE(outside.fullyCovered);
  EXPECT_TRUE(outside.densityRedundant);
  EXPECT_FALSE(outside.fullyRedundant);
}

} // namespace
