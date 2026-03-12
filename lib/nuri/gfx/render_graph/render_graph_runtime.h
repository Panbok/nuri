#pragma once

#include "nuri/core/log.h"
#include "nuri/core/pmr_scratch.h"
#include "nuri/core/profiling.h"
#include "nuri/defines.h"

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace nuri {

struct NURI_API RenderGraphRuntimeConfig {
  uint32_t workerCount = 1u;
  bool parallelCompile = true;
  bool parallelGraphicsRecording = true;
};

struct NURI_API RenderGraphContiguousRange {
  uint32_t offset = 0u;
  uint32_t count = 0u;
};

class NURI_API RenderGraphRuntime {
public:
  explicit RenderGraphRuntime(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  explicit RenderGraphRuntime(
      RenderGraphRuntimeConfig config,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~RenderGraphRuntime();

  RenderGraphRuntime(const RenderGraphRuntime &) = delete;
  RenderGraphRuntime &operator=(const RenderGraphRuntime &) = delete;
  RenderGraphRuntime(RenderGraphRuntime &&) = delete;
  RenderGraphRuntime &operator=(RenderGraphRuntime &&) = delete;

  [[nodiscard]] const RenderGraphRuntimeConfig &config() const noexcept {
    return config_;
  }
  [[nodiscard]] uint32_t workerCount() const noexcept {
    return config_.workerCount;
  }
  [[nodiscard]] bool parallelCompileEnabled() const noexcept {
    return config_.parallelCompile && config_.workerCount > 1u;
  }
  [[nodiscard]] bool parallelGraphicsRecordingEnabled() const noexcept {
    return config_.parallelGraphicsRecording && config_.workerCount > 1u;
  }
  [[nodiscard]] std::pmr::memory_resource *memory() const noexcept {
    return memory_;
  }
  [[nodiscard]] ScratchArena &
  workerScratchArena(uint32_t workerIndex) noexcept {
    NURI_ASSERT(workerIndex < workers_.size(), "Worker index out of bounds");
    if (workerIndex >= workers_.size()) [[unlikely]] {
      return workers_.front()->scratch;
    }
    return workers_[workerIndex]->scratch;
  }

  [[nodiscard]] static std::vector<RenderGraphContiguousRange>
  makeRanges(uint32_t itemCount, uint32_t maxRangeCount);

  template <typename Fn>
  void runRanges(std::span<const RenderGraphContiguousRange> ranges, Fn &&fn) {
    std::function<void(uint32_t, RenderGraphContiguousRange)> task =
        std::forward<Fn>(fn);
    runRangesImpl(ranges, task);
  }

private:
  struct WorkerState {
    ScratchArena scratch;
    std::jthread thread;
  };

  void runRangesImpl(
      std::span<const RenderGraphContiguousRange> ranges,
      const std::function<void(uint32_t, RenderGraphContiguousRange)> &task);
  void workerLoop(uint32_t workerIndex, std::stop_token stopToken);

  std::pmr::memory_resource *memory_ = nullptr;
  RenderGraphRuntimeConfig config_{};
  std::vector<std::unique_ptr<WorkerState>> workers_{};
  std::mutex mutex_{};
  std::condition_variable cvWork_{};
  std::condition_variable cvDone_{};
  std::vector<RenderGraphContiguousRange> currentRanges_{};
  std::function<void(uint32_t, RenderGraphContiguousRange)> currentTask_{};
  uint64_t generation_ = 0u;
  uint64_t completedGeneration_ = 0u;
  uint32_t activeRangeCount_ = 0u;
  uint32_t pendingWorkers_ = 0u;
};

} // namespace nuri
