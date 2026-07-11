#include "tests_pch.h"

#include "nuri/core/containers/slot_pool.h"

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

} // namespace
