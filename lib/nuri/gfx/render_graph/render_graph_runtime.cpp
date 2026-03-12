#include "nuri/pch.h"

#include "nuri/gfx/render_graph/render_graph_runtime.h"

#include "nuri/utils/env_utils.h"

namespace nuri {
namespace {

[[nodiscard]] bool parseEnvBool(std::string_view name, bool defaultValue) {
  const std::optional<std::string> value = readEnvVar(name);
  if (!value.has_value()) {
    return defaultValue;
  }

  return *value == "1" || *value == "true" || *value == "TRUE";
}

[[nodiscard]] uint32_t parseWorkerCountEnv() {
  const std::optional<std::string> value =
      readEnvVar("NURI_RENDER_GRAPH_WORKER_COUNT");
  if (!value.has_value() || value->empty()) {
    return 0u;
  }

  char *end = nullptr;
  const unsigned long parsed = std::strtoul(value->c_str(), &end, 10);
  if (end == value->c_str() || parsed == 0ul) {
    return 0u;
  }
  return static_cast<uint32_t>(std::min<unsigned long>(parsed, 64ul));
}

[[nodiscard]] RenderGraphRuntimeConfig resolveDefaultConfig() {
  const uint32_t envWorkerCount = parseWorkerCountEnv();
  const uint32_t hardwareCount =
      std::max(1u, static_cast<uint32_t>(std::thread::hardware_concurrency()));
  RenderGraphRuntimeConfig config{};
  config.workerCount =
      std::clamp(envWorkerCount == 0u ? hardwareCount : envWorkerCount, 1u, 8u);
  config.parallelCompile =
      !parseEnvBool("NURI_RENDER_GRAPH_DISABLE_PARALLEL_COMPILE", false);
  config.parallelGraphicsRecording =
      !parseEnvBool("NURI_RENDER_GRAPH_DISABLE_PARALLEL_RECORDING", false);
  return config;
}

} // namespace

RenderGraphRuntime::RenderGraphRuntime(std::pmr::memory_resource *memory)
    : RenderGraphRuntime(resolveDefaultConfig(), memory) {}

RenderGraphRuntime::RenderGraphRuntime(RenderGraphRuntimeConfig config,
                                       std::pmr::memory_resource *memory)
    : memory_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      config_(config) {
  config_.workerCount = std::max(1u, config_.workerCount);
  workers_.reserve(config_.workerCount);
  for (uint32_t workerIndex = 0u; workerIndex < config_.workerCount;
       ++workerIndex) {
    workers_.push_back(std::make_unique<WorkerState>());
  }
  for (uint32_t workerIndex = 1u; workerIndex < config_.workerCount;
       ++workerIndex) {
    workers_[workerIndex]->thread =
        std::jthread([this, workerIndex](std::stop_token stopToken) {
          workerLoop(workerIndex, stopToken);
        });
  }
}

RenderGraphRuntime::~RenderGraphRuntime() {
  for (uint32_t workerIndex = 1u; workerIndex < workers_.size();
       ++workerIndex) {
    if (workers_[workerIndex]->thread.joinable()) {
      workers_[workerIndex]->thread.request_stop();
    }
  }
  cvWork_.notify_all();
}

std::vector<RenderGraphContiguousRange>
RenderGraphRuntime::makeRanges(uint32_t itemCount, uint32_t maxRangeCount) {
  std::vector<RenderGraphContiguousRange> ranges{};
  if (itemCount == 0u || maxRangeCount == 0u) {
    return ranges;
  }

  const uint32_t rangeCount = std::min(itemCount, maxRangeCount);
  ranges.resize(rangeCount);

  const uint32_t baseCount = itemCount / rangeCount;
  const uint32_t remainder = itemCount % rangeCount;
  uint32_t offset = 0u;
  for (uint32_t i = 0u; i < rangeCount; ++i) {
    const uint32_t count = baseCount + (i < remainder ? 1u : 0u);
    ranges[i] = RenderGraphContiguousRange{.offset = offset, .count = count};
    offset += count;
  }

  return ranges;
}

void RenderGraphRuntime::runRangesImpl(
    std::span<const RenderGraphContiguousRange> ranges,
    const std::function<void(uint32_t, RenderGraphContiguousRange)> &task) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_COPY);
  if (ranges.empty()) {
    return;
  }
  if (config_.workerCount <= 1u || ranges.size() <= 1u) {
    for (uint32_t i = 0u; i < ranges.size(); ++i) {
      task(i, ranges[i]);
    }
    return;
  }

  {
    NURI_PROFILER_ZONE("RenderGraphRuntime.run_ranges.schedule",
                       NURI_PROFILER_COLOR_CMD_COPY);
    std::unique_lock lock(mutex_);
    currentRanges_.assign(ranges.begin(), ranges.end());
    currentTask_ = task;
    activeRangeCount_ = static_cast<uint32_t>(ranges.size());
    pendingWorkers_ = activeRangeCount_ > 0u ? activeRangeCount_ - 1u : 0u;
    ++generation_;
    NURI_PROFILER_ZONE_END();
  }
  cvWork_.notify_all();

  task(0u, ranges[0u]);

  {
    NURI_PROFILER_ZONE("RenderGraphRuntime.run_ranges.wait",
                       NURI_PROFILER_COLOR_CMD_COPY);
    std::unique_lock lock(mutex_);
    cvDone_.wait(lock, [this] { return pendingWorkers_ == 0u; });
    currentTask_ = {};
    currentRanges_.clear();
    completedGeneration_ = generation_;
    NURI_PROFILER_ZONE_END();
  }
}

void RenderGraphRuntime::workerLoop(uint32_t workerIndex,
                                    std::stop_token stopToken) {
  NURI_PROFILER_THREAD("RenderGraphWorker");
  uint64_t observedGeneration = 0u;

  while (!stopToken.stop_requested()) {
    std::function<void(uint32_t, RenderGraphContiguousRange)> task{};
    RenderGraphContiguousRange range{};
    bool shouldRun = false;
    {
      std::unique_lock lock(mutex_);
      cvWork_.wait(lock, [this, &stopToken, &observedGeneration] {
        return stopToken.stop_requested() || generation_ != observedGeneration;
      });
      if (stopToken.stop_requested()) {
        return;
      }

      observedGeneration = generation_;
      if (workerIndex < activeRangeCount_) {
        task = currentTask_;
        range = currentRanges_[workerIndex];
        shouldRun = static_cast<bool>(task);
      }
    }

    if (shouldRun) {
      NURI_PROFILER_ZONE("RenderGraphRuntime.worker_task",
                         NURI_PROFILER_COLOR_CMD_COPY);
      task(workerIndex, range);
      NURI_PROFILER_ZONE_END();
    }

    {
      std::lock_guard lock(mutex_);
      if (workerIndex < activeRangeCount_ && pendingWorkers_ > 0u) {
        --pendingWorkers_;
      }
    }
    cvDone_.notify_one();
  }
}

} // namespace nuri
