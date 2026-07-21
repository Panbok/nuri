#include "tests_pch.h"

#include "nuri/gfx/ddgi/ddgi_scheduler.h"

#include <array>

namespace {

TEST(DDGISchedulerTests, OrdersGlobalCandidatesByStateAgeAndStableIdentity) {
  const std::array candidates{
      nuri::DDGIProbeScheduleCandidate{2u, 9u,
                                       nuri::DDGIProbeState::Vigilant, 3u},
      nuri::DDGIProbeScheduleCandidate{1u, 7u,
                                       nuri::DDGIProbeState::NewlyAwake, 9u},
      nuri::DDGIProbeScheduleCandidate{1u, 3u,
                                       nuri::DDGIProbeState::Uninitialized, 0u},
      nuri::DDGIProbeScheduleCandidate{1u, 2u,
                                       nuri::DDGIProbeState::Vigilant, 3u},
      nuri::DDGIProbeScheduleCandidate{3u, 1u,
                                       nuri::DDGIProbeState::Vigilant, 1u},
  };
  std::array<nuri::DDGIProbeScheduleCandidate, candidates.size()> workspace{};
  std::array<nuri::DDGIProbeUpdateEntry, candidates.size()> output{};
  auto result = nuri::scheduleDDGIProbeUpdates(
      candidates, {.raysPerProbe = 16u,
                   .maxProbeUpdates = 5u,
                   .maxRayQueries = 160u},
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
      candidates, {.raysPerProbe = 32u,
                   .maxProbeUpdates = 4u,
                   .maxRayQueries = 128u},
      workspace, output);
  ASSERT_TRUE(result.hasValue());
  EXPECT_EQ(result->updatedProbes, 2u);
  EXPECT_EQ(result->primaryQueries, 64u);
  EXPECT_EQ(result->secondaryQueriesReserved, 64u);
  EXPECT_EQ(result->unusedQueryCapacity, 0u);
  EXPECT_TRUE(result->truncated);
}

TEST(DDGISchedulerTests, OffAndSleepingProbesRemainUnscheduledWhenForced) {
  const std::array candidates{
      nuri::DDGIProbeScheduleCandidate{1u, 0u, nuri::DDGIProbeState::Off},
      nuri::DDGIProbeScheduleCandidate{1u, 1u,
                                       nuri::DDGIProbeState::Sleeping},
      nuri::DDGIProbeScheduleCandidate{1u, 2u, nuri::DDGIProbeState::Awake,
                                       20u},
  };
  std::array<nuri::DDGIProbeScheduleCandidate, 3> workspace{};
  std::array<nuri::DDGIProbeUpdateEntry, 3> output{};
  auto result = nuri::scheduleDDGIProbeUpdates(
      candidates, {.raysPerProbe = 16u,
                   .maxProbeUpdates = 3u,
                   .maxRayQueries = 96u,
                   .forceFullUpdate = true},
      workspace, output);
  ASSERT_TRUE(result.hasValue());
  ASSERT_EQ(result->updatedProbes, 1u);
  EXPECT_EQ(output[0].probeId, 2u);
}

TEST(DDGISchedulerTests, RejectsBorrowedWorkspaceThatCannotCopyFrameCandidates) {
  const std::array candidates{
      nuri::DDGIProbeScheduleCandidate{1u, 0u,
                                       nuri::DDGIProbeState::Vigilant},
      nuri::DDGIProbeScheduleCandidate{1u, 1u,
                                       nuri::DDGIProbeState::Vigilant},
  };
  std::array<nuri::DDGIProbeScheduleCandidate, 1> workspace{};
  std::array<nuri::DDGIProbeUpdateEntry, 2> output{};
  auto result = nuri::scheduleDDGIProbeUpdates(
      candidates, {}, workspace, output);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error(), nuri::DDGISchedulerError::WorkspaceTooSmall);
}

} // namespace
