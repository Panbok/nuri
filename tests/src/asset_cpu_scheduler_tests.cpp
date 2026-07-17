#include "tests_pch.h"

#include "nuri/resources/async/asset_cpu_scheduler.h"

#include <atomic>
#include <chrono>
#include <latch>
#include <mutex>
#include <thread>
#include <vector>

namespace {

using namespace std::chrono_literals;

TEST(AssetCpuSchedulerTests, EnforcesInFlightBudgets) {
  nuri::AssetCpuScheduler scheduler(nuri::AssetCpuSchedulerConfig{
      .workerCount = 1u,
      .maxInFlightJobs = 1u,
      .maxInFlightBytes = 16u,
  });
  std::latch entered(1);
  std::latch release(1);
  auto first = scheduler.enqueue(nuri::AssetCpuJob{
      .estimatedBytes = 16u,
      .execute =
          [&](std::stop_token) {
            entered.count_down();
            release.wait();
          },
  });
  ASSERT_FALSE(first.hasError()) << first.error();
  entered.wait();

  auto countRejected = scheduler.enqueue(nuri::AssetCpuJob{
      .estimatedBytes = 1u,
      .execute = [](std::stop_token) {},
  });
  EXPECT_TRUE(countRejected.hasError());

  release.count_down();
  scheduler.waitIdle();
  EXPECT_EQ(scheduler.stats().rejectedJobs, 1u);
}

TEST(AssetCpuSchedulerTests, RunsQueuedJobsByPriority) {
  nuri::AssetCpuScheduler scheduler(nuri::AssetCpuSchedulerConfig{
      .workerCount = 1u,
      .maxInFlightJobs = 8u,
      .maxInFlightBytes = 1024u,
  });
  std::latch blockerEntered(1);
  std::latch releaseBlocker(1);
  std::mutex orderMutex;
  std::vector<int> order;

  ASSERT_FALSE(scheduler
                   .enqueue(nuri::AssetCpuJob{
                       .priority = nuri::AssetPriority::Critical,
                       .execute =
                           [&](std::stop_token) {
                             blockerEntered.count_down();
                             releaseBlocker.wait();
                           },
                   })
                   .hasError());
  blockerEntered.wait();

  ASSERT_FALSE(scheduler
                   .enqueue(nuri::AssetCpuJob{
                       .priority = nuri::AssetPriority::Background,
                       .execute =
                           [&](std::stop_token) {
                             std::lock_guard lock(orderMutex);
                             order.push_back(3);
                           },
                   })
                   .hasError());
  ASSERT_FALSE(scheduler
                   .enqueue(nuri::AssetCpuJob{
                       .priority = nuri::AssetPriority::Visible,
                       .execute =
                           [&](std::stop_token) {
                             std::lock_guard lock(orderMutex);
                             order.push_back(1);
                           },
                   })
                   .hasError());
  ASSERT_FALSE(scheduler
                   .enqueue(nuri::AssetCpuJob{
                       .priority = nuri::AssetPriority::Normal,
                       .execute =
                           [&](std::stop_token) {
                             std::lock_guard lock(orderMutex);
                             order.push_back(2);
                           },
                   })
                   .hasError());

  releaseBlocker.count_down();
  scheduler.waitIdle();
  EXPECT_EQ(order, (std::vector<int>{1, 2, 3}));
}

TEST(AssetCpuSchedulerTests, ByteBudgetQueuesWorkUntilCapacityIsAvailable) {
  nuri::AssetCpuScheduler scheduler(nuri::AssetCpuSchedulerConfig{
      .workerCount = 2u,
      .maxInFlightJobs = 4u,
      .maxInFlightBytes = 16u,
  });
  std::latch firstEntered(1);
  std::latch releaseFirst(1);
  std::atomic<uint32_t> secondRuns = 0u;

  auto first = scheduler.enqueue(nuri::AssetCpuJob{
      .estimatedBytes = 16u,
      .execute =
          [&](std::stop_token) {
            firstEntered.count_down();
            releaseFirst.wait();
          },
  });
  ASSERT_FALSE(first.hasError()) << first.error();
  firstEntered.wait();

  auto second = scheduler.enqueue(nuri::AssetCpuJob{
      .estimatedBytes = 16u,
      .execute =
          [&](std::stop_token) {
            secondRuns.fetch_add(1u, std::memory_order_release);
          },
  });
  ASSERT_FALSE(second.hasError()) << second.error();
  std::this_thread::sleep_for(5ms);
  EXPECT_EQ(secondRuns.load(std::memory_order_acquire), 0u);
  EXPECT_EQ(scheduler.stats().queuedJobs, 1u);

  releaseFirst.count_down();
  scheduler.waitIdle();
  EXPECT_EQ(secondRuns.load(std::memory_order_acquire), 1u);
  EXPECT_EQ(scheduler.stats().rejectedJobs, 0u);
}

TEST(AssetCpuSchedulerTests, CancellationIsVisibleToRunningJob) {
  nuri::AssetCpuScheduler scheduler(nuri::AssetCpuSchedulerConfig{
      .workerCount = 1u,
      .maxInFlightJobs = 4u,
      .maxInFlightBytes = 1024u,
  });
  std::latch entered(1);
  std::atomic<bool> observedStop = false;
  auto task = scheduler.enqueue(nuri::AssetCpuJob{
      .execute =
          [&](std::stop_token stopToken) {
            entered.count_down();
            while (!stopToken.stop_requested()) {
              std::this_thread::yield();
            }
            observedStop.store(true, std::memory_order_release);
          },
  });
  ASSERT_FALSE(task.hasError()) << task.error();
  entered.wait();
  EXPECT_TRUE(scheduler.cancel(task.value()));
  scheduler.waitIdle();
  EXPECT_TRUE(observedStop.load(std::memory_order_acquire));
  EXPECT_EQ(scheduler.stats().cancelledJobs, 1u);
}

TEST(AssetCpuSchedulerTests, WorkClassConcurrencyDoesNotBlockOtherClasses) {
  nuri::AssetCpuScheduler scheduler(nuri::AssetCpuSchedulerConfig{
      .workerCount = 2u,
      .maxInFlightJobs = 8u,
      .maxInFlightBytes = 1024u,
      .transcodeConcurrency = 1u,
  });
  std::latch transcodeEntered(1);
  std::latch releaseTranscode(1);
  std::atomic<uint32_t> secondTranscodeRuns = 0u;
  std::latch metadataRan(1);

  ASSERT_FALSE(scheduler
                   .enqueue(nuri::AssetCpuJob{
                       .workClass = nuri::AssetWorkClass::Transcode,
                       .execute =
                           [&](std::stop_token) {
                             transcodeEntered.count_down();
                             releaseTranscode.wait();
                           },
                   })
                   .hasError());
  transcodeEntered.wait();
  ASSERT_FALSE(scheduler
                   .enqueue(nuri::AssetCpuJob{
                       .workClass = nuri::AssetWorkClass::Transcode,
                       .execute =
                           [&](std::stop_token) {
                             secondTranscodeRuns.fetch_add(
                                 1u, std::memory_order_release);
                           },
                   })
                   .hasError());
  ASSERT_FALSE(scheduler
                   .enqueue(nuri::AssetCpuJob{
                       .workClass = nuri::AssetWorkClass::Metadata,
                       .execute =
                           [&](std::stop_token) {
                             metadataRan.count_down();
                           },
                   })
                   .hasError());

  for (uint32_t attempt = 0u;
       attempt < 100u && !metadataRan.try_wait(); ++attempt) {
    std::this_thread::sleep_for(1ms);
  }
  EXPECT_TRUE(metadataRan.try_wait());
  EXPECT_EQ(secondTranscodeRuns.load(std::memory_order_acquire), 0u);
  releaseTranscode.count_down();
  scheduler.waitIdle();
  EXPECT_EQ(secondTranscodeRuns.load(std::memory_order_acquire), 1u);
}

} // namespace
