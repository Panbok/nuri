#include "tests_pch.h"

#include "nuri/gfx/ddgi/ddgi_scheduler.h"

#include <array>

namespace {

TEST(DDGISchedulerTests, OrdersGlobalCandidatesByStateAgeAndStableIdentity) {
  const std::array candidates{
      nuri::DDGIProbeScheduleCandidate{2u, 9u, nuri::DDGIProbeState::Vigilant,
                                       3u},
      nuri::DDGIProbeScheduleCandidate{1u, 7u, nuri::DDGIProbeState::NewlyAwake,
                                       9u},
      nuri::DDGIProbeScheduleCandidate{1u, 3u,
                                       nuri::DDGIProbeState::Uninitialized, 0u},
      nuri::DDGIProbeScheduleCandidate{1u, 2u, nuri::DDGIProbeState::Vigilant,
                                       3u},
      nuri::DDGIProbeScheduleCandidate{3u, 1u, nuri::DDGIProbeState::Vigilant,
                                       1u},
  };
  std::array<nuri::DDGIProbeScheduleCandidate, candidates.size()> workspace{};
  std::array<nuri::DDGIProbeUpdateEntry, candidates.size()> output{};
  auto result = nuri::scheduleDDGIProbeUpdates(
      candidates,
      {.raysPerProbe = 16u, .maxProbeUpdates = 5u, .maxRayQueries = 160u},
      workspace, output);
  ASSERT_TRUE(result.hasValue());
  ASSERT_EQ(result->updatedProbes, 5u);
  EXPECT_EQ(output[0].probeId, 3u);
  EXPECT_EQ(output[1].probeId, 7u);
  EXPECT_EQ(output[2].volumeStableId, 3u);
  EXPECT_EQ(output[3].volumeStableId, 1u);
  EXPECT_EQ(output[4].volumeStableId, 2u);
}

TEST(DDGISchedulerTests, ConservativelyReservesSecondaryQueryForEveryPrimary) {
  std::array<nuri::DDGIProbeScheduleCandidate, 4> candidates{};
  for (uint32_t index = 0u; index < candidates.size(); ++index) {
    candidates[index] = {1u, index, nuri::DDGIProbeState::Vigilant, index};
  }
  std::array<nuri::DDGIProbeScheduleCandidate, 4> workspace{};
  std::array<nuri::DDGIProbeUpdateEntry, 4> output{};
  auto result = nuri::scheduleDDGIProbeUpdates(
      candidates,
      {.raysPerProbe = 32u, .maxProbeUpdates = 4u, .maxRayQueries = 128u},
      workspace, output);
  ASSERT_TRUE(result.hasValue());
  EXPECT_EQ(result->updatedProbes, 2u);
  EXPECT_EQ(result->primaryQueries, 64u);
  EXPECT_EQ(result->secondaryQueriesReserved, 64u);
  EXPECT_EQ(result->unusedQueryCapacity, 0u);
  EXPECT_TRUE(result->truncated);
}

TEST(DDGISchedulerTests, LendsUnreservedSecondaryCapacityToPrimaryQueries) {
  std::array<nuri::DDGIProbeScheduleCandidate, 4> candidates{};
  for (uint32_t index = 0u; index < candidates.size(); ++index) {
    candidates[index] = {1u, index, nuri::DDGIProbeState::Vigilant, index};
  }
  std::array<nuri::DDGIProbeScheduleCandidate, 4> workspace{};
  std::array<nuri::DDGIProbeUpdateEntry, 4> output{};
  auto result =
      nuri::scheduleDDGIProbeUpdates(candidates,
                                     {.raysPerProbe = 32u,
                                      .maxProbeUpdates = 4u,
                                      .maxRayQueries = 128u,
                                      .secondaryQueriesPer1024Primary = 256u},
                                     workspace, output);
  ASSERT_TRUE(result.hasValue());
  EXPECT_EQ(result->updatedProbes, 3u);
  EXPECT_EQ(result->primaryQueries, 96u);
  EXPECT_EQ(result->secondaryQueriesReserved, 24u);
  EXPECT_EQ(result->unusedQueryCapacity, 8u);
  EXPECT_TRUE(result->truncated);
}

TEST(DDGISchedulerTests, AccountsLowRayClassificationSeparatelyFromIrradiance) {
  const std::array candidates{
      nuri::DDGIProbeScheduleCandidate{1u, 0u,
                                       nuri::DDGIProbeState::Uninitialized, 0u},
      nuri::DDGIProbeScheduleCandidate{1u, 1u, nuri::DDGIProbeState::Vigilant,
                                       1u},
  };
  std::array<nuri::DDGIProbeScheduleCandidate, candidates.size()> workspace{};
  std::array<nuri::DDGIProbeUpdateEntry, candidates.size()> output{};

  auto result =
      nuri::scheduleDDGIProbeUpdates(candidates,
                                     {.raysPerProbe = 64u,
                                      .classificationRaysPerProbe = 16u,
                                      .maxProbeUpdates = 2u,
                                      .maxRayQueries = 160u,
                                      .secondaryQueriesPer1024Primary = 0u},
                                     workspace, output);

  ASSERT_TRUE(result.hasValue());
  ASSERT_EQ(result->updatedProbes, 2u);
  EXPECT_EQ(output[0].rayBase, 0u);
  EXPECT_EQ(output[0].rayCount, 16u);
  EXPECT_EQ(output[1].rayBase, 16u);
  EXPECT_EQ(output[1].rayCount, 64u);
  EXPECT_EQ(result->primaryQueries, 80u);
  EXPECT_EQ(result->classificationProbeUpdates, 1u);
  EXPECT_EQ(result->classificationPrimaryQueries, 16u);
  EXPECT_EQ(result->irradiancePrimaryQueries, 64u);
}

TEST(DDGISchedulerTests, OffAndSleepingProbesRemainUnscheduledWhenForced) {
  const std::array candidates{
      nuri::DDGIProbeScheduleCandidate{1u, 0u, nuri::DDGIProbeState::Off},
      nuri::DDGIProbeScheduleCandidate{1u, 1u, nuri::DDGIProbeState::Sleeping},
      nuri::DDGIProbeScheduleCandidate{1u, 2u, nuri::DDGIProbeState::Awake,
                                       20u},
  };
  std::array<nuri::DDGIProbeScheduleCandidate, 3> workspace{};
  std::array<nuri::DDGIProbeUpdateEntry, 3> output{};
  auto result = nuri::scheduleDDGIProbeUpdates(candidates,
                                               {.raysPerProbe = 16u,
                                                .maxProbeUpdates = 3u,
                                                .maxRayQueries = 96u,
                                                .forceFullUpdate = true},
                                               workspace, output);
  ASSERT_TRUE(result.hasValue());
  ASSERT_EQ(result->updatedProbes, 1u);
  EXPECT_EQ(output[0].probeId, 2u);
}

TEST(DDGISchedulerTests,
     RejectsBorrowedWorkspaceThatCannotCopyFrameCandidates) {
  const std::array candidates{
      nuri::DDGIProbeScheduleCandidate{1u, 0u, nuri::DDGIProbeState::Vigilant},
      nuri::DDGIProbeScheduleCandidate{1u, 1u, nuri::DDGIProbeState::Vigilant},
  };
  std::array<nuri::DDGIProbeScheduleCandidate, 1> workspace{};
  std::array<nuri::DDGIProbeUpdateEntry, 2> output{};
  auto result =
      nuri::scheduleDDGIProbeUpdates(candidates, {}, workspace, output);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error(), nuri::DDGISchedulerError::WorkspaceTooSmall);
}

TEST(DDGISchedulerTests, TieredSchedulerGuaranteesOneProbePerReadyTier) {
  const std::array tiers{
      nuri::DDGITierScheduleInput{
          .stableKey = 11u, .effectiveOrder = 0u, .weight = 8u},
      nuri::DDGITierScheduleInput{
          .stableKey = 22u, .effectiveOrder = 1u, .weight = 1u},
  };
  const std::array candidates{
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 11u,
          .volumeStableId = 0u,
          .probeId = 10u,
          .state = nuri::DDGIProbeState::Vigilant},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 11u,
          .volumeStableId = 0u,
          .probeId = 11u,
          .state = nuri::DDGIProbeState::Vigilant},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 22u,
          .volumeStableId = 1u,
          .probeId = 20u,
          .state = nuri::DDGIProbeState::Vigilant},
  };
  std::array<nuri::DDGITieredProbeScheduleCandidate, candidates.size()>
      workspace{};
  std::array<nuri::DDGIProbeUpdateEntry, 2> output{};

  auto result = nuri::scheduleDDGITieredProbeUpdates(
      candidates, tiers,
      {.raysPerProbe = 16u, .maxProbeUpdates = 2u, .maxRayQueries = 64u},
      workspace, output);

  ASSERT_TRUE(result.hasValue());
  ASSERT_EQ(result->schedule.updatedProbes, 2u);
  EXPECT_EQ(result->tiers[0].usedQuota, 1u);
  EXPECT_EQ(result->tiers[1].usedQuota, 1u);
  EXPECT_EQ(result->tiers[0].pendingDeficit, 7);
  EXPECT_EQ(result->tiers[1].pendingDeficit, -7);
  EXPECT_EQ(result->tiers[0].pendingStarvationFrames, 0u);
  EXPECT_EQ(result->tiers[1].pendingStarvationFrames, 0u);
}

TEST(DDGISchedulerTests, TieredSchedulerReservesHalfCapacityForUrgentWork) {
  const std::array tiers{
      nuri::DDGITierScheduleInput{
          .stableKey = 1u, .effectiveOrder = 0u, .weight = 1u},
      nuri::DDGITierScheduleInput{
          .stableKey = 2u, .effectiveOrder = 1u, .weight = 1u},
  };
  const std::array candidates{
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 2u,
          .volumeStableId = 1u,
          .probeId = 3u,
          .state = nuri::DDGIProbeState::NewlyAwake},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 2u,
          .volumeStableId = 1u,
          .probeId = 1u,
          .state = nuri::DDGIProbeState::Uninitialized},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 2u,
          .volumeStableId = 1u,
          .probeId = 2u,
          .state = nuri::DDGIProbeState::Uninitialized},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 1u,
          .volumeStableId = 0u,
          .probeId = 9u,
          .state = nuri::DDGIProbeState::Vigilant},
  };
  std::array<nuri::DDGITieredProbeScheduleCandidate, candidates.size()>
      workspace{};
  std::array<nuri::DDGIProbeUpdateEntry, 4> output{};

  auto result = nuri::scheduleDDGITieredProbeUpdates(
      candidates, tiers,
      {.raysPerProbe = 8u, .maxProbeUpdates = 4u, .maxRayQueries = 64u},
      workspace, output);

  ASSERT_TRUE(result.hasValue());
  EXPECT_EQ(result->urgentReservation, 2u);
  EXPECT_EQ(result->urgentReservationUsed, 2u);
  EXPECT_EQ(output[0].volumeStableId, 1u);
  EXPECT_EQ(output[1].volumeStableId, 1u);
  EXPECT_EQ(result->tiers[0].usedQuota, 1u);
  EXPECT_EQ(result->schedule.primaryQueries, 32u);
  EXPECT_EQ(result->schedule.secondaryQueriesReserved, 32u);
  EXPECT_EQ(result->schedule.unusedQueryCapacity, 0u);
}

TEST(DDGISchedulerTests, TieredSchedulerRollsUnusedQuotaInEffectiveOrder) {
  const std::array tiers{
      nuri::DDGITierScheduleInput{
          .stableKey = 10u, .effectiveOrder = 0u, .weight = 3u},
      nuri::DDGITierScheduleInput{
          .stableKey = 20u, .effectiveOrder = 1u, .weight = 1u},
  };
  const std::array candidates{
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 10u,
          .volumeStableId = 0u,
          .probeId = 0u,
          .state = nuri::DDGIProbeState::Vigilant},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 20u,
          .volumeStableId = 1u,
          .probeId = 0u,
          .state = nuri::DDGIProbeState::Vigilant},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 20u,
          .volumeStableId = 1u,
          .probeId = 1u,
          .state = nuri::DDGIProbeState::Vigilant},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 20u,
          .volumeStableId = 1u,
          .probeId = 2u,
          .state = nuri::DDGIProbeState::Vigilant},
  };
  std::array<nuri::DDGITieredProbeScheduleCandidate, candidates.size()>
      workspace{};
  std::array<nuri::DDGIProbeUpdateEntry, 4> output{};

  auto result = nuri::scheduleDDGITieredProbeUpdates(
      candidates, tiers,
      {.raysPerProbe = 8u, .maxProbeUpdates = 4u, .maxRayQueries = 64u},
      workspace, output);

  ASSERT_TRUE(result.hasValue());
  EXPECT_EQ(result->tiers[0].scheduledQuota, 3u);
  EXPECT_EQ(result->tiers[0].usedQuota, 1u);
  EXPECT_EQ(result->tiers[1].scheduledQuota, 1u);
  EXPECT_EQ(result->tiers[1].usedQuota, 3u);
  EXPECT_EQ(result->schedule.updatedProbes, 4u);
}

TEST(DDGISchedulerTests, TieredSchedulerReturnsPendingStarvationFacts) {
  const std::array tiers{
      nuri::DDGITierScheduleInput{
          .stableKey = 100u, .effectiveOrder = 0u, .weight = 1u},
      nuri::DDGITierScheduleInput{.stableKey = 200u,
                                  .submittedStarvationFrames = 7u,
                                  .effectiveOrder = 1u,
                                  .weight = 1u},
  };
  const std::array candidates{
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 100u,
          .volumeStableId = 0u,
          .probeId = 0u,
          .state = nuri::DDGIProbeState::Vigilant},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 200u,
          .volumeStableId = 1u,
          .probeId = 0u,
          .state = nuri::DDGIProbeState::Vigilant},
  };
  std::array<nuri::DDGITieredProbeScheduleCandidate, candidates.size()>
      workspace{};
  std::array<nuri::DDGIProbeUpdateEntry, 1> output{};

  auto result = nuri::scheduleDDGITieredProbeUpdates(
      candidates, tiers,
      {.raysPerProbe = 8u, .maxProbeUpdates = 1u, .maxRayQueries = 16u},
      workspace, output);

  ASSERT_TRUE(result.hasValue());
  EXPECT_EQ(result->tiers[0].usedQuota, 1u);
  EXPECT_EQ(result->tiers[1].usedQuota, 0u);
  EXPECT_EQ(result->tiers[1].pendingStarvationFrames, 8u);
  EXPECT_TRUE(result->schedule.truncated);
}

TEST(DDGISchedulerTests, TieredDeficitIgnoresUnusableFrameCapacity) {
  const std::array tiers{nuri::DDGITierScheduleInput{
      .stableKey = 1u, .effectiveOrder = 0u, .weight = 4u}};
  const std::array candidates{nuri::DDGITieredProbeScheduleCandidate{
      .tierStableKey = 1u,
      .volumeStableId = 0u,
      .probeId = 0u,
      .state = nuri::DDGIProbeState::Vigilant}};
  std::array<nuri::DDGITieredProbeScheduleCandidate, 1> workspace{};
  std::array<nuri::DDGIProbeUpdateEntry, 8> output{};

  auto result = nuri::scheduleDDGITieredProbeUpdates(
      candidates, tiers,
      {.raysPerProbe = 8u, .maxProbeUpdates = 8u, .maxRayQueries = 128u},
      workspace, output);

  ASSERT_TRUE(result.hasValue());
  EXPECT_EQ(result->schedule.updatedProbes, 1u);
  EXPECT_EQ(result->tiers[0].pendingDeficit, 0);
  EXPECT_EQ(result->tiers[0].scheduledQuota, 1u);
}

} // namespace
