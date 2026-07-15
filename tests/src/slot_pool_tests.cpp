#include "tests_pch.h"

#include "nuri/core/containers/slot_pool.h"
#include "nuri/gfx/gpu_retirement_queue.h"

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

TEST(GpuRetirementQueueTests, RetainsUntilCompletionAndReleasesExactlyOnce) {
  struct CountingDelete {
    uint32_t *count = nullptr;
    void operator()(uint32_t *value) const noexcept {
      ++*count;
      delete value;
    }
  };

  using Resource = std::unique_ptr<uint32_t, CountingDelete>;
  nuri::GpuRetirementQueue<Resource, uint64_t> queue;
  uint32_t destructionCount = 0u;
  queue.retire(Resource(new uint32_t(1u), CountingDelete{&destructionCount}),
               4u);
  queue.retire(Resource(new uint32_t(2u), CountingDelete{&destructionCount}),
               9u);

  EXPECT_EQ(queue.collectThrough(3u), 0u);
  EXPECT_EQ(queue.pendingCount(), 2u);
  EXPECT_EQ(destructionCount, 0u);

  EXPECT_EQ(queue.collectThrough(4u), 1u);
  EXPECT_EQ(queue.pendingCount(), 1u);
  EXPECT_EQ(destructionCount, 1u);

  EXPECT_EQ(queue.collectThrough(9u), 1u);
  EXPECT_TRUE(queue.empty());
  EXPECT_EQ(destructionCount, 2u);
}

TEST(GpuRetirementQueueTests, PredicateSupportsOpaqueCompletionTokens) {
  using Resource = std::unique_ptr<uint32_t>;
  nuri::GpuRetirementQueue<Resource, uint32_t> queue;
  queue.retire(std::make_unique<uint32_t>(7u), 17u);

  const size_t collected =
      queue.collectIf([](uint32_t token) { return token == 17u; });
  EXPECT_EQ(collected, 1u);
  EXPECT_TRUE(queue.empty());
}

} // namespace
