#pragma once
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <stop_token>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
namespace nuri {

enum class AssetPriority : uint8_t {
  Critical = 0,
  Visible,
  Normal,
  Background,
  Count,
};

enum class AssetWorkClass : uint8_t {
  Io = 0,
  Decode,
  Cook,
  Transcode,
  Metadata,
  GpuMaterialization,
  Count,
};

struct AssetCpuSchedulerConfig {
  uint32_t workerCount = 0u;
  uint32_t reservedLogicalThreads = 4u;
  uint32_t interactiveWorkerCount = 0u;
  uint32_t maxInFlightJobs = 4096u;
  uint64_t maxInFlightBytes = 512ull * 1024ull * 1024ull;
  uint32_t ioConcurrency = 0u;
  uint32_t decodeConcurrency = 0u;
  uint32_t cookConcurrency = 0u;
  uint32_t transcodeConcurrency = 1u;
  uint32_t metadataConcurrency = 0u;
  uint32_t gpuMaterializationConcurrency = 1u;
};

struct AssetCpuTaskHandle {
  uint64_t value = 0u;
  constexpr bool
  operator==(const AssetCpuTaskHandle &) const noexcept = default;
};

struct AssetCpuJob {
  AssetPriority priority = AssetPriority::Normal;
  AssetWorkClass workClass = AssetWorkClass::Metadata;
  uint64_t estimatedBytes = 0u;
  std::string debugName{};
  std::function<void(std::stop_token)> execute{};
  std::function<void()> onCancelled{};
};

struct AssetCpuSchedulerStats {
  uint32_t workerCount = 0u;
  uint32_t activeWorkerLimit = 0u;
  bool interactiveMode = false;
  uint32_t queuedJobs = 0u;
  uint32_t runningJobs = 0u;
  uint64_t inFlightBytes = 0u;
  uint64_t submittedJobs = 0u;
  uint64_t completedJobs = 0u;
  uint64_t cancelledJobs = 0u;
  uint64_t rejectedJobs = 0u;
  std::array<uint32_t, static_cast<size_t>(AssetWorkClass::Count)>
      runningByClass{};
};

class NURI_API AssetCpuScheduler final {
public:
  explicit AssetCpuScheduler(AssetCpuSchedulerConfig config = {});
  ~AssetCpuScheduler();
  AssetCpuScheduler(const AssetCpuScheduler &) = delete;
  AssetCpuScheduler &operator=(const AssetCpuScheduler &) = delete;
  AssetCpuScheduler(AssetCpuScheduler &&) = delete;
  AssetCpuScheduler &operator=(AssetCpuScheduler &&) = delete;
  [[nodiscard]] Result<AssetCpuTaskHandle, std::string>
  enqueue(AssetCpuJob job);
  [[nodiscard]] bool cancel(AssetCpuTaskHandle handle);
  [[nodiscard]] bool setPriority(AssetCpuTaskHandle handle,
                                 AssetPriority priority);
  void setInteractiveMode(bool enabled);
  void requestStop();
  void waitIdle();
  [[nodiscard]] AssetCpuSchedulerStats stats() const;

private:
  struct JobControl {
    std::stop_source stop{};
    AssetPriority priority = AssetPriority::Normal;
  };
  struct QueuedJob {
    AssetCpuTaskHandle handle{};
    AssetCpuJob job{};
    std::shared_ptr<JobControl> control{};
  };
  static constexpr size_t kPriorityCount =
      static_cast<size_t>(AssetPriority::Count);
  static constexpr size_t kWorkClassCount =
      static_cast<size_t>(AssetWorkClass::Count);
  [[nodiscard]] bool hasRunnableJobLocked() const noexcept;
  [[nodiscard]] QueuedJob popNextJobLocked();
  void workerMain(std::stop_token stopToken, uint32_t workerIndex);
  AssetCpuSchedulerConfig config_{};
  mutable std::mutex mutex_{};
  std::condition_variable_any workCv_{};
  std::condition_variable idleCv_{};
  std::array<std::deque<QueuedJob>, kPriorityCount> queues_{};
  std::unordered_map<uint64_t, std::weak_ptr<JobControl>> controls_{};
  std::vector<std::jthread> workers_{};
  uint64_t nextTaskId_ = 1u;
  uint32_t queuedJobs_ = 0u;
  uint32_t runningJobs_ = 0u;
  uint32_t activeWorkerLimit_ = 1u;
  bool interactiveMode_ = false;
  std::array<uint32_t, kWorkClassCount> runningByClass_{};
  std::array<uint32_t, kWorkClassCount> concurrencyByClass_{};
  uint64_t inFlightBytes_ = 0u;
  uint64_t submittedJobs_ = 0u;
  uint64_t completedJobs_ = 0u;
  uint64_t cancelledJobs_ = 0u;
  uint64_t rejectedJobs_ = 0u;
  bool stopping_ = false;
};

} // namespace nuri
