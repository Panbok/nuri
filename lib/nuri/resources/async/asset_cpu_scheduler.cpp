#include "nuri/pch.h"

#include "nuri/resources/async/asset_cpu_scheduler.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"

#include <algorithm>
#include <exception>
#include <limits>

namespace nuri {
namespace {

[[nodiscard]] uint32_t
resolveWorkerCount(const AssetCpuSchedulerConfig &config) noexcept {
  if (config.workerCount != 0u) {
    return config.workerCount;
  }
  const uint32_t hardware = std::thread::hardware_concurrency();
  return std::max(1u, hardware > 2u ? hardware - 2u : 1u);
}

[[nodiscard]] size_t priorityIndex(AssetPriority priority) noexcept {
  return std::min(static_cast<size_t>(priority),
                  static_cast<size_t>(AssetPriority::Background));
}

[[nodiscard]] size_t workClassIndex(AssetWorkClass workClass) noexcept {
  return std::min(static_cast<size_t>(workClass),
                  static_cast<size_t>(AssetWorkClass::Metadata));
}

} // namespace

AssetCpuScheduler::AssetCpuScheduler(AssetCpuSchedulerConfig config)
    : config_(config) {
  config_.workerCount = resolveWorkerCount(config_);
  config_.maxInFlightJobs = std::max(config_.maxInFlightJobs, 1u);
  config_.maxInFlightBytes = std::max(config_.maxInFlightBytes, 1ull);
  const auto resolveConcurrency = [this](uint32_t requested) {
    return std::clamp(requested == 0u ? config_.workerCount : requested, 1u,
                      config_.workerCount);
  };
  concurrencyByClass_ = {
      resolveConcurrency(config_.ioConcurrency),
      resolveConcurrency(config_.decodeConcurrency),
      resolveConcurrency(config_.cookConcurrency),
      resolveConcurrency(config_.transcodeConcurrency),
      resolveConcurrency(config_.metadataConcurrency),
  };
  workers_.reserve(config_.workerCount);
  for (uint32_t workerIndex = 0u; workerIndex < config_.workerCount;
       ++workerIndex) {
    workers_.emplace_back(
        [this, workerIndex](std::stop_token stopToken) {
          workerMain(stopToken, workerIndex);
        });
  }
}

AssetCpuScheduler::~AssetCpuScheduler() {
  requestStop();
  workers_.clear();
}

Result<AssetCpuTaskHandle, std::string>
AssetCpuScheduler::enqueue(AssetCpuJob job) {
  if (!job.execute) {
    return Result<AssetCpuTaskHandle, std::string>::makeError(
        "AssetCpuScheduler::enqueue: execute callback is empty");
  }

  std::lock_guard lock(mutex_);
  if (stopping_) {
    return Result<AssetCpuTaskHandle, std::string>::makeError(
        "AssetCpuScheduler::enqueue: scheduler is stopping");
  }
  if (queuedJobs_ + runningJobs_ >= config_.maxInFlightJobs) {
    ++rejectedJobs_;
    return Result<AssetCpuTaskHandle, std::string>::makeError(
        "AssetCpuScheduler::enqueue: in-flight job budget exhausted");
  }
  if (job.estimatedBytes > config_.maxInFlightBytes) {
    ++rejectedJobs_;
    return Result<AssetCpuTaskHandle, std::string>::makeError(
        "AssetCpuScheduler::enqueue: job exceeds the in-flight byte budget");
  }

  const AssetCpuTaskHandle handle{.value = nextTaskId_++};
  auto control = std::make_shared<JobControl>();
  control->priority = job.priority;
  queues_[priorityIndex(job.priority)].push_back(QueuedJob{
      .handle = handle,
      .job = std::move(job),
      .control = control,
  });
  controls_[handle.value] = control;
  ++queuedJobs_;
  ++submittedJobs_;
  workCv_.notify_one();
  return Result<AssetCpuTaskHandle, std::string>::makeResult(handle);
}

bool AssetCpuScheduler::cancel(AssetCpuTaskHandle handle) {
  if (handle.value == 0u) {
    return false;
  }
  std::lock_guard lock(mutex_);
  auto it = controls_.find(handle.value);
  if (it == controls_.end()) {
    return false;
  }
  const std::shared_ptr<JobControl> control = it->second.lock();
  return control != nullptr && control->stop.request_stop();
}

bool AssetCpuScheduler::setPriority(AssetCpuTaskHandle handle,
                                    AssetPriority priority) {
  if (handle.value == 0u || priority == AssetPriority::Count) {
    return false;
  }
  std::lock_guard lock(mutex_);
  auto controlIt = controls_.find(handle.value);
  if (controlIt == controls_.end()) {
    return false;
  }
  const std::shared_ptr<JobControl> control = controlIt->second.lock();
  if (!control) {
    return false;
  }
  if (control->priority == priority) {
    return true;
  }

  for (auto &queue : queues_) {
    auto jobIt = std::find_if(
        queue.begin(), queue.end(), [handle](const QueuedJob &queued) {
          return queued.handle == handle;
        });
    if (jobIt == queue.end()) {
      continue;
    }
    QueuedJob moved = std::move(*jobIt);
    queue.erase(jobIt);
    moved.job.priority = priority;
    moved.control->priority = priority;
    queues_[priorityIndex(priority)].push_back(std::move(moved));
    workCv_.notify_one();
    return true;
  }

  control->priority = priority;
  return true;
}

void AssetCpuScheduler::requestStop() {
  std::vector<std::function<void()>> cancellationCallbacks{};
  {
    std::lock_guard lock(mutex_);
    if (stopping_) {
      return;
    }
    stopping_ = true;
    for (auto &queue : queues_) {
      for (QueuedJob &queued : queue) {
        queued.control->stop.request_stop();
        if (queued.job.onCancelled) {
          cancellationCallbacks.push_back(std::move(queued.job.onCancelled));
        }
        controls_.erase(queued.handle.value);
        ++cancelledJobs_;
      }
      queue.clear();
    }
    queuedJobs_ = 0u;
    for (auto &[id, weakControl] : controls_) {
      (void)id;
      if (const std::shared_ptr<JobControl> control = weakControl.lock()) {
        control->stop.request_stop();
      }
    }
    for (std::jthread &worker : workers_) {
      worker.request_stop();
    }
  }
  for (auto &callback : cancellationCallbacks) {
    callback();
  }
  workCv_.notify_all();
  idleCv_.notify_all();
}

void AssetCpuScheduler::waitIdle() {
  std::unique_lock lock(mutex_);
  idleCv_.wait(lock, [this] {
    return queuedJobs_ == 0u && runningJobs_ == 0u;
  });
}

AssetCpuSchedulerStats AssetCpuScheduler::stats() const {
  std::lock_guard lock(mutex_);
  return AssetCpuSchedulerStats{
      .workerCount = config_.workerCount,
      .queuedJobs = queuedJobs_,
      .runningJobs = runningJobs_,
      .inFlightBytes = inFlightBytes_,
      .submittedJobs = submittedJobs_,
      .completedJobs = completedJobs_,
      .cancelledJobs = cancelledJobs_,
      .rejectedJobs = rejectedJobs_,
      .runningByClass = runningByClass_,
  };
}

bool AssetCpuScheduler::hasQueuedJobsLocked() const noexcept {
  return std::any_of(queues_.begin(), queues_.end(),
                     [](const auto &queue) { return !queue.empty(); });
}

bool AssetCpuScheduler::hasRunnableJobLocked() const noexcept {
  for (const auto &queue : queues_) {
    for (const QueuedJob &queued : queue) {
      const size_t classIndex = workClassIndex(queued.job.workClass);
      if (runningByClass_[classIndex] <
              concurrencyByClass_[classIndex] &&
          queued.job.estimatedBytes <=
          config_.maxInFlightBytes - inFlightBytes_) {
        return true;
      }
    }
  }
  return false;
}

AssetCpuScheduler::QueuedJob AssetCpuScheduler::popNextJobLocked() {
  for (auto &queue : queues_) {
    auto jobIt = std::find_if(
        queue.begin(), queue.end(), [this](const QueuedJob &queued) {
          const size_t classIndex =
              workClassIndex(queued.job.workClass);
          return runningByClass_[classIndex] <
                     concurrencyByClass_[classIndex] &&
                 queued.job.estimatedBytes <=
                 config_.maxInFlightBytes - inFlightBytes_;
        });
    if (jobIt == queue.end()) {
      continue;
    }
    QueuedJob job = std::move(*jobIt);
    queue.erase(jobIt);
    --queuedJobs_;
    ++runningJobs_;
    ++runningByClass_[workClassIndex(job.job.workClass)];
    inFlightBytes_ += job.job.estimatedBytes;
    return job;
  }
  NURI_ASSERT(false, "AssetCpuScheduler::popNextJobLocked: no runnable job");
  return {};
}

void AssetCpuScheduler::workerMain(std::stop_token stopToken,
                                   uint32_t workerIndex) {
  const std::string threadName =
      "Asset CPU " + std::to_string(workerIndex);
  NURI_PROFILER_THREAD(threadName.c_str());
  while (!stopToken.stop_requested()) {
    QueuedJob queued{};
    {
      std::unique_lock lock(mutex_);
      workCv_.wait(lock, stopToken, [this] {
        return stopping_ || hasRunnableJobLocked();
      });
      if (stopToken.stop_requested() ||
          (stopping_ && !hasQueuedJobsLocked())) {
        break;
      }
      queued = popNextJobLocked();
    }

    const bool cancelledBeforeRun = queued.control->stop.stop_requested();
    if (cancelledBeforeRun) {
      if (queued.job.onCancelled) {
        queued.job.onCancelled();
      }
    } else {
      try {
        NURI_PROFILER_ZONE("AssetCpuScheduler.execute",
                           NURI_PROFILER_COLOR_CREATE);
        queued.job.execute(queued.control->stop.get_token());
        NURI_PROFILER_ZONE_END();
      } catch (const std::exception &exception) {
        NURI_LOG_ERROR("AssetCpuScheduler: job '%s' threw: %s",
                       queued.job.debugName.c_str(), exception.what());
      } catch (...) {
        NURI_LOG_ERROR("AssetCpuScheduler: job '%s' threw an unknown exception",
                       queued.job.debugName.c_str());
      }
    }

    {
      std::lock_guard lock(mutex_);
      NURI_ASSERT(runningJobs_ > 0u,
                  "AssetCpuScheduler: running job count underflow");
      --runningJobs_;
      const size_t classIndex = workClassIndex(queued.job.workClass);
      NURI_ASSERT(runningByClass_[classIndex] > 0u,
                  "AssetCpuScheduler: running work-class count underflow");
      --runningByClass_[classIndex];
      inFlightBytes_ -= queued.job.estimatedBytes;
      controls_.erase(queued.handle.value);
      if (cancelledBeforeRun ||
          queued.control->stop.stop_requested()) {
        ++cancelledJobs_;
      } else {
        ++completedJobs_;
      }
      if (queuedJobs_ == 0u && runningJobs_ == 0u) {
        idleCv_.notify_all();
      }
      workCv_.notify_all();
    }
  }
}

} // namespace nuri
