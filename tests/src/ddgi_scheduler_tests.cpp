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
                                      .maxRayQueries = 144u},
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
  EXPECT_EQ(result->secondaryQueriesReserved, 64u);
  EXPECT_EQ(result->unusedQueryCapacity, 0u);
}

TEST(DDGISchedulerTests, AccountsLowRayRadiancePopulationAsIrradiance) {
  const std::array candidates{
      nuri::DDGIProbeScheduleCandidate{
          .volumeStableId = 1u,
          .probeId = 0u,
          .state = nuri::DDGIProbeState::NewlyVigilant,
          .invalidated = true,
          .radianceRayCount = 16u,
      },
      nuri::DDGIProbeScheduleCandidate{
          .volumeStableId = 1u,
          .probeId = 1u,
          .state = nuri::DDGIProbeState::Vigilant,
          .lastSubmittedUpdate = 1u,
      },
  };
  std::array<nuri::DDGIProbeScheduleCandidate, candidates.size()> workspace{};
  std::array<nuri::DDGIProbeUpdateEntry, candidates.size()> output{};

  auto result = nuri::scheduleDDGIProbeUpdates(
      candidates,
      {.raysPerProbe = 64u, .maxProbeUpdates = 2u, .maxRayQueries = 160u},
      workspace, output);

  ASSERT_TRUE(result.hasValue());
  ASSERT_EQ(result->updatedProbes, 2u);
  EXPECT_EQ(output[0].rayCount, 16u);
  EXPECT_EQ(output[0].flags & nuri::kDDGIProbeUpdateClassificationGeometry, 0u);
  EXPECT_EQ(output[1].rayBase, 16u);
  EXPECT_EQ(output[1].rayCount, 64u);
  EXPECT_EQ(result->classificationProbeUpdates, 0u);
  EXPECT_EQ(result->irradiancePrimaryQueries, 80u);
  EXPECT_EQ(result->secondaryQueriesReserved, 80u);
}

TEST(DDGISchedulerTests,
     ClassificationCanUseTotalCapacityWithoutExceedingRadianceCap) {
  const std::array candidates{
      nuri::DDGIProbeScheduleCandidate{1u, 0u,
                                       nuri::DDGIProbeState::Uninitialized},
      nuri::DDGIProbeScheduleCandidate{1u, 1u,
                                       nuri::DDGIProbeState::Uninitialized},
      nuri::DDGIProbeScheduleCandidate{1u, 2u,
                                       nuri::DDGIProbeState::Uninitialized},
      nuri::DDGIProbeScheduleCandidate{1u, 3u,
                                       nuri::DDGIProbeState::NewlyAwake},
      nuri::DDGIProbeScheduleCandidate{1u, 4u,
                                       nuri::DDGIProbeState::NewlyAwake},
      nuri::DDGIProbeScheduleCandidate{1u, 5u,
                                       nuri::DDGIProbeState::NewlyAwake},
  };
  std::array<nuri::DDGIProbeScheduleCandidate, candidates.size()> workspace{};
  std::array<nuri::DDGIProbeUpdateEntry, candidates.size()> output{};

  auto result =
      nuri::scheduleDDGIProbeUpdates(candidates,
                                     {.raysPerProbe = 64u,
                                      .classificationRaysPerProbe = 16u,
                                      .maxProbeUpdates = 6u,
                                      .maxRadianceProbeUpdates = 2u,
                                      .maxRayQueries = 512u},
                                     workspace, output);

  ASSERT_TRUE(result.hasValue());
  EXPECT_EQ(result->updatedProbes, 5u);
  EXPECT_EQ(result->classificationProbeUpdates, 3u);
  EXPECT_EQ(result->irradiancePrimaryQueries, 128u);
  EXPECT_TRUE(result->truncated);
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
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 0u,
                  .probeId = 10u,
                  .state = nuri::DDGIProbeState::Vigilant}},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 11u,
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 0u,
                  .probeId = 11u,
                  .state = nuri::DDGIProbeState::Vigilant}},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 22u,
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 1u,
                  .probeId = 20u,
                  .state = nuri::DDGIProbeState::Vigilant}},
  };
  std::array<nuri::DDGITieredProbeScheduleCandidate, candidates.size()>
      workspace{};
  std::array<nuri::DDGIProbeUpdateEntry, 2> output{};

  auto result =
      nuri::scheduleDDGITieredProbeUpdates(candidates, tiers,
                                           {.raysPerProbe = 16u,
                                            .maxProbeUpdates = 2u,
                                            .maxMaintenanceProbeUpdates = 1u,
                                            .maxRayQueries = 64u},
                                           workspace, output);

  ASSERT_TRUE(result.hasValue());
  ASSERT_EQ(result->schedule.updatedProbes, 2u);
  EXPECT_EQ(result->schedule.requestedMaintenanceProbeCapacity, 1u);
  EXPECT_EQ(result->schedule.effectiveMaintenanceProbeCapacity, 2u);
  EXPECT_EQ(result->schedule.maintenanceProbeUpdates, 2u);
  EXPECT_EQ(result->tiers[0].usedQuota, 1u);
  EXPECT_EQ(result->tiers[1].usedQuota, 1u);
  EXPECT_EQ(result->tiers[0].pendingDeficit, 7);
  EXPECT_EQ(result->tiers[1].pendingDeficit, -7);
  EXPECT_EQ(result->tiers[0].pendingStarvationFrames, 0u);
  EXPECT_EQ(result->tiers[1].pendingStarvationFrames, 0u);
}

TEST(DDGISchedulerTests,
     TieredBoundedSelectionPreservesExactPriorityAgeAndIdentityPrefix) {
  const std::array tiers{nuri::DDGITierScheduleInput{
      .stableKey = 1u, .effectiveOrder = 0u, .weight = 1u}};
  const std::array candidates{
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 1u,
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 2u,
                  .probeId = 4u,
                  .state = nuri::DDGIProbeState::Vigilant,
                  .lastSubmittedUpdate = 8u}},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 1u,
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 2u,
                  .probeId = 7u,
                  .state = nuri::DDGIProbeState::Vigilant,
                  .lastSubmittedUpdate = 2u}},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 1u,
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 1u,
                  .probeId = 9u,
                  .state = nuri::DDGIProbeState::Vigilant,
                  .lastSubmittedUpdate = 3u}},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 1u,
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 1u,
                  .probeId = 3u,
                  .state = nuri::DDGIProbeState::Vigilant,
                  .lastSubmittedUpdate = 1u}},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 1u,
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 1u,
                  .probeId = 5u,
                  .state = nuri::DDGIProbeState::Vigilant,
                  .lastSubmittedUpdate = 2u}},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 1u,
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 3u,
                  .probeId = 0u,
                  .state = nuri::DDGIProbeState::Vigilant,
                  .lastSubmittedUpdate = 9u}},
  };
  std::array<nuri::DDGITieredProbeScheduleCandidate, candidates.size()>
      workspace{};
  std::array<nuri::DDGIProbeUpdateEntry, 3> output{};

  auto result =
      nuri::scheduleDDGITieredProbeUpdates(candidates, tiers,
                                           {.raysPerProbe = 8u,
                                            .maxProbeUpdates = 3u,
                                            .maxMaintenanceProbeUpdates = 3u,
                                            .maxRayQueries = 48u},
                                           workspace, output);

  ASSERT_TRUE(result.hasValue());
  ASSERT_EQ(result->schedule.updatedProbes, 3u);
  EXPECT_EQ(result->tiers[0].eligibleProbes, candidates.size());
  EXPECT_EQ(output[0].probeId, 3u);
  EXPECT_EQ(output[1].volumeStableId, 1u);
  EXPECT_EQ(output[1].probeId, 5u);
  EXPECT_EQ(output[2].volumeStableId, 2u);
  EXPECT_EQ(output[2].probeId, 7u);
}

TEST(DDGISchedulerTests,
     TieredBootstrapFinishesInProgressClassificationBeforeFreshProbes) {
  const std::array tiers{nuri::DDGITierScheduleInput{
      .stableKey = 1u, .effectiveOrder = 0u, .weight = 1u}};
  const std::array candidates{
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 1u,
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 0u,
                  .probeId = 10u,
                  .state = nuri::DDGIProbeState::Uninitialized,
                  .lastSubmittedUpdate = 1u,
                  .classificationIteration = 0u}},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 1u,
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 0u,
                  .probeId = 20u,
                  .state = nuri::DDGIProbeState::Uninitialized,
                  .lastSubmittedUpdate = 9u,
                  .classificationIteration = 4u}},
  };
  std::array<nuri::DDGITieredProbeScheduleCandidate, candidates.size()>
      workspace{};
  std::array<nuri::DDGIProbeUpdateEntry, 1> output{};

  auto result =
      nuri::scheduleDDGITieredProbeUpdates(candidates, tiers,
                                           {.raysPerProbe = 64u,
                                            .classificationRaysPerProbe = 16u,
                                            .maxProbeUpdates = 1u,
                                            .maxRayQueries = 128u},
                                           workspace, output);

  ASSERT_TRUE(result.hasValue());
  ASSERT_EQ(result->schedule.updatedProbes, 1u);
  EXPECT_EQ(output[0].probeId, 20u);
}

TEST(DDGISchedulerTests,
     TieredMaintenanceLimitPreservesAllUrgentWorkAndOneMaintenanceSlot) {
  const std::array tiers{
      nuri::DDGITierScheduleInput{
          .stableKey = 1u, .effectiveOrder = 0u, .weight = 1u},
      nuri::DDGITierScheduleInput{
          .stableKey = 2u, .effectiveOrder = 1u, .weight = 1u},
  };
  const std::array candidates{
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 1u,
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 0u,
                  .probeId = 10u,
                  .state = nuri::DDGIProbeState::Vigilant}},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 1u,
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 0u,
                  .probeId = 11u,
                  .state = nuri::DDGIProbeState::Awake}},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 2u,
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 1u,
                  .probeId = 20u,
                  .state = nuri::DDGIProbeState::Uninitialized}},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 2u,
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 1u,
                  .probeId = 21u,
                  .state = nuri::DDGIProbeState::NewlyAwake}},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 2u,
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 1u,
                  .probeId = 22u,
                  .state = nuri::DDGIProbeState::Vigilant,
                  .invalidated = true}},
  };
  std::array<nuri::DDGITieredProbeScheduleCandidate, candidates.size()>
      workspace{};
  std::array<nuri::DDGIProbeUpdateEntry, 4> output{};

  auto result =
      nuri::scheduleDDGITieredProbeUpdates(candidates, tiers,
                                           {.raysPerProbe = 8u,
                                            .maxProbeUpdates = 4u,
                                            .maxMaintenanceProbeUpdates = 1u,
                                            .maxRayQueries = 64u},
                                           workspace, output);

  ASSERT_TRUE(result.hasValue());
  EXPECT_EQ(result->schedule.updatedProbes, 4u);
  EXPECT_EQ(result->schedule.maintenanceProbeUpdates, 1u);
  EXPECT_EQ(result->schedule.requestedMaintenanceProbeCapacity, 1u);
  EXPECT_EQ(result->schedule.effectiveMaintenanceProbeCapacity, 1u);
  EXPECT_EQ(result->tiers[0].usedQuota, 1u);
  EXPECT_EQ(result->tiers[1].usedQuota, 3u);
  EXPECT_EQ(output[0].volumeStableId, 1u);
  EXPECT_EQ(output[1].volumeStableId, 1u);
  EXPECT_EQ(output[2].volumeStableId, 0u);
  EXPECT_EQ(output[3].volumeStableId, 1u);
  EXPECT_NE(output[3].flags & nuri::kDDGIProbeUpdateReasonWake, 0u);
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
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 1u,
                  .probeId = 3u,
                  .state = nuri::DDGIProbeState::NewlyAwake}},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 2u,
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 1u,
                  .probeId = 1u,
                  .state = nuri::DDGIProbeState::Uninitialized}},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 2u,
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 1u,
                  .probeId = 2u,
                  .state = nuri::DDGIProbeState::Uninitialized}},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 1u,
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 0u,
                  .probeId = 9u,
                  .state = nuri::DDGIProbeState::Vigilant}},
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
  EXPECT_EQ(result->schedule.secondaryQueriesReserved, 16u);
  EXPECT_EQ(result->schedule.unusedQueryCapacity, 16u);
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
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 0u,
                  .probeId = 0u,
                  .state = nuri::DDGIProbeState::Vigilant}},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 20u,
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 1u,
                  .probeId = 0u,
                  .state = nuri::DDGIProbeState::Vigilant}},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 20u,
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 1u,
                  .probeId = 1u,
                  .state = nuri::DDGIProbeState::Vigilant}},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 20u,
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 1u,
                  .probeId = 2u,
                  .state = nuri::DDGIProbeState::Vigilant}},
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
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 0u,
                  .probeId = 0u,
                  .state = nuri::DDGIProbeState::Vigilant}},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 200u,
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 1u,
                  .probeId = 0u,
                  .state = nuri::DDGIProbeState::Vigilant}},
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
      .probe = nuri::DDGIProbeScheduleCandidate{
          .volumeStableId = 0u,
          .probeId = 0u,
          .state = nuri::DDGIProbeState::Vigilant}}};
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

TEST(DDGISchedulerTests,
     TieredScheduleUsesRecoveredClassificationQueryCapacityExactly) {
  const std::array tiers{nuri::DDGITierScheduleInput{
      .stableKey = 1u, .effectiveOrder = 0u, .weight = 1u}};
  const std::array candidates{
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 1u,
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 0u,
                  .probeId = 0u,
                  .state = nuri::DDGIProbeState::Uninitialized}},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 1u,
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 0u,
                  .probeId = 1u,
                  .state = nuri::DDGIProbeState::Uninitialized}},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 1u,
          .probe = nuri::DDGIProbeScheduleCandidate{
              .volumeStableId = 0u,
              .probeId = 2u,
              .state = nuri::DDGIProbeState::Vigilant}}};
  std::array<nuri::DDGITieredProbeScheduleCandidate, candidates.size()>
      workspace{};
  std::array<nuri::DDGIProbeUpdateEntry, candidates.size()> output{};

  auto result =
      nuri::scheduleDDGITieredProbeUpdates(candidates, tiers,
                                           {.raysPerProbe = 64u,
                                            .classificationRaysPerProbe = 16u,
                                            .maxProbeUpdates = 3u,
                                            .maxRayQueries = 160u},
                                           workspace, output);

  ASSERT_TRUE(result.hasValue());
  EXPECT_EQ(result->schedule.updatedProbes, 3u);
  EXPECT_EQ(result->schedule.primaryQueries, 96u);
  EXPECT_EQ(result->schedule.classificationProbeUpdates, 2u);
  EXPECT_EQ(result->schedule.classificationPrimaryQueries, 32u);
  EXPECT_EQ(result->schedule.irradiancePrimaryQueries, 64u);
  EXPECT_EQ(result->schedule.secondaryQueriesReserved, 64u);
  EXPECT_EQ(result->schedule.unusedQueryCapacity, 0u);
  EXPECT_FALSE(result->schedule.truncated);
  EXPECT_EQ(output[0].flags & nuri::kDDGIProbeUpdateClassificationGeometry,
            nuri::kDDGIProbeUpdateClassificationGeometry);
  EXPECT_EQ(output[1].flags & nuri::kDDGIProbeUpdateClassificationGeometry,
            nuri::kDDGIProbeUpdateClassificationGeometry);
  EXPECT_EQ(output[2].flags & nuri::kDDGIProbeUpdateClassificationGeometry, 0u);
}

TEST(DDGISchedulerTests,
     TieredClassificationRollsOverCapacityAfterRadianceCapIsReached) {
  const std::array tiers{
      nuri::DDGITierScheduleInput{
          .stableKey = 1u, .effectiveOrder = 0u, .weight = 1u},
      nuri::DDGITierScheduleInput{
          .stableKey = 2u, .effectiveOrder = 1u, .weight = 1u},
  };
  const std::array candidates{
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 1u,
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 0u,
                  .probeId = 0u,
                  .state = nuri::DDGIProbeState::NewlyAwake}},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 1u,
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 0u,
                  .probeId = 1u,
                  .state = nuri::DDGIProbeState::NewlyAwake}},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 1u,
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 0u,
                  .probeId = 2u,
                  .state = nuri::DDGIProbeState::NewlyAwake}},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 2u,
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 1u,
                  .probeId = 0u,
                  .state = nuri::DDGIProbeState::Uninitialized}},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 2u,
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 1u,
                  .probeId = 1u,
                  .state = nuri::DDGIProbeState::Uninitialized}},
      nuri::DDGITieredProbeScheduleCandidate{
          .tierStableKey = 2u,
          .probe =
              nuri::DDGIProbeScheduleCandidate{
                  .volumeStableId = 1u,
                  .probeId = 2u,
                  .state = nuri::DDGIProbeState::Uninitialized}},
  };
  std::array<nuri::DDGITieredProbeScheduleCandidate, candidates.size()>
      workspace{};
  std::array<nuri::DDGIProbeUpdateEntry, candidates.size()> output{};

  auto result =
      nuri::scheduleDDGITieredProbeUpdates(candidates, tiers,
                                           {.raysPerProbe = 64u,
                                            .classificationRaysPerProbe = 16u,
                                            .maxProbeUpdates = 6u,
                                            .maxRadianceProbeUpdates = 2u,
                                            .maxRayQueries = 512u},
                                           workspace, output);

  ASSERT_TRUE(result.hasValue());
  EXPECT_EQ(result->schedule.updatedProbes, 5u);
  EXPECT_EQ(result->schedule.classificationProbeUpdates, 3u);
  EXPECT_EQ(result->schedule.irradiancePrimaryQueries, 128u);
  EXPECT_EQ(result->tiers[0].usedQuota, 2u);
  EXPECT_EQ(result->tiers[1].usedQuota, 3u);
  EXPECT_TRUE(result->schedule.truncated);
}

} // namespace
