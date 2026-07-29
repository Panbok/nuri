#include "tests_pch.h"

#include "nuri/core/containers/slot_pool.h"
#include "nuri/platform/detail/recording_retirement_tracker.h"

#include <memory>

namespace {

TEST(SlotPoolTests, ReacquireReusesIndexWithNewGeneration) {
  nuri::SlotPool<> pool;

  const nuri::SlotReservation first = pool.acquire();
  pool.release(first.index);
  const nuri::SlotReservation second = pool.acquire();

  EXPECT_EQ(second.index, first.index);
  EXPECT_NE(second.generation, first.generation);
  EXPECT_FALSE(second.appended);
  EXPECT_FALSE(pool.isValid(first.index, first.generation));
  EXPECT_TRUE(pool.isValid(second.index, second.generation));
}

TEST(SlotPoolTests, ClearResetsCountsAndNextAcquireRestarts) {
  nuri::SlotPool<> pool;

  const nuri::SlotReservation first = pool.acquire();
  pool.release(first.index);
  pool.clear();

  EXPECT_EQ(pool.slotCount(), 0u);
  EXPECT_EQ(pool.liveCount(), 0u);

  const nuri::SlotReservation next = pool.acquire();
  EXPECT_EQ(next.index, 0u);
  EXPECT_EQ(next.generation, 1u);
  EXPECT_TRUE(next.appended);
}

TEST(SlotPoolTests, RetiredIndexIsInvalidButUnavailableUntilRecycled) {
  nuri::SlotPool<> pool;

  const nuri::SlotReservation first = pool.acquire();
  pool.retire(first.index);

  EXPECT_FALSE(pool.isValid(first.index, first.generation));
  EXPECT_TRUE(pool.isRetired(first.index));
  EXPECT_EQ(pool.liveCount(), 0u);

  const nuri::SlotReservation appended = pool.acquire();
  EXPECT_NE(appended.index, first.index);
  EXPECT_TRUE(appended.appended);

  pool.recycle(first.index);
  const nuri::SlotReservation recycled = pool.acquire();
  EXPECT_EQ(recycled.index, first.index);
  EXPECT_NE(recycled.generation, first.generation);
  EXPECT_FALSE(recycled.appended);
}

TEST(SlotPoolTests, MaskedGenerationPolicyWrapsAndSkipsZero) {
  using Policy = nuri::MaskedNonZeroGenerationPolicy<0x3u>;

  EXPECT_EQ(Policy::next(0u), 1u);
  EXPECT_EQ(Policy::next(1u), 2u);
  EXPECT_EQ(Policy::next(2u), 3u);
  EXPECT_EQ(Policy::next(3u), 1u);
}

TEST(SlotPoolTests,
     SequentialFrameReuseStaysBoundedAcrossOneHundredThousandFrames) {
  nuri::SlotPool<> recordingContexts;
  nuri::SlotPool<> recordedCommands;
  nuri::SlotPool<> submissions;

  for (uint32_t frame = 0u; frame < 100'000u; ++frame) {
    const nuri::SlotReservation context = recordingContexts.acquire();
    const nuri::SlotReservation command = recordedCommands.acquire();
    const nuri::SlotReservation submission = submissions.acquire();
    recordingContexts.release(context.index);
    recordedCommands.release(command.index);
    submissions.release(submission.index);
  }

  EXPECT_EQ(recordingContexts.slotCount(), 1u);
  EXPECT_EQ(recordedCommands.slotCount(), 1u);
  EXPECT_EQ(submissions.slotCount(), 1u);
  EXPECT_EQ(recordingContexts.liveCount(), 0u);
  EXPECT_EQ(recordedCommands.liveCount(), 0u);
  EXPECT_EQ(submissions.liveCount(), 0u);
}

TEST(RecordingRetirementTrackerTests,
     OutOfOrderResolutionDoesNotAdvancePastUnresolvedRecording) {
  nuri::RecordingRetirementTracker tracker;
  const uint64_t first = tracker.beginRecording();
  const uint64_t second = tracker.beginRecording();

  EXPECT_TRUE(tracker.resolveRecording(second, 8u));
  EXPECT_EQ(tracker.resolvedThroughSerial(), 0u);
  EXPECT_FALSE(tracker.tryResolveLastUse(second, 3u).has_value());

  EXPECT_TRUE(tracker.resolveRecording(first, 5u));
  EXPECT_EQ(tracker.resolvedThroughSerial(), second);
  EXPECT_EQ(tracker.resolvedSubmissionMax(), 8u);
  ASSERT_TRUE(tracker.tryResolveLastUse(second, 3u).has_value());
  EXPECT_EQ(*tracker.tryResolveLastUse(second, 3u), 8u);
}

TEST(RecordingRetirementTrackerTests,
     AbandonedRecordingResolvesWithoutInventingSubmission) {
  nuri::RecordingRetirementTracker tracker;
  const uint64_t serial = tracker.beginRecording();

  EXPECT_TRUE(tracker.resolveRecording(serial, 0u));
  ASSERT_TRUE(tracker.tryResolveLastUse(serial, 11u).has_value());
  EXPECT_EQ(*tracker.tryResolveLastUse(serial, 11u), 11u);
  EXPECT_FALSE(tracker.resolveRecording(serial, 12u));
}

} // namespace
