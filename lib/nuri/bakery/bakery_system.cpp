#include "nuri/pch.h"

#include "nuri/bakery/bakery_system.h"

#include "nuri/bakery/brdf_lut_baker.h"
#include "nuri/bakery/envmap_prefilter_baker.h"
#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/gpu_device.h"

namespace nuri::bakery {
namespace {

[[nodiscard]] bool isTerminal(BakeJobState state) {
  return state == BakeJobState::Succeeded || state == BakeJobState::Skipped ||
         state == BakeJobState::Failed || state == BakeJobState::Canceled;
}

[[nodiscard]] size_t maxGpuStepsPerTick(BakeryExecutionProfile profile) {
  switch (profile) {
  case BakeryExecutionProfile::Interactive:
    return 1u;
  case BakeryExecutionProfile::Balanced:
    return 2u;
  case BakeryExecutionProfile::Fast:
    return std::numeric_limits<size_t>::max();
  }
  return 1u;
}

[[nodiscard]] size_t
maxWriteCompletionsPerTick(BakeryExecutionProfile profile) {
  switch (profile) {
  case BakeryExecutionProfile::Interactive:
    return 1u;
  case BakeryExecutionProfile::Balanced:
    return 2u;
  case BakeryExecutionProfile::Fast:
    return std::numeric_limits<size_t>::max();
  }
  return 1u;
}

[[nodiscard]] std::string_view jobKindName(BakeJobKind kind) {
  switch (kind) {
  case BakeJobKind::BrdfLut:
    return "BRDF LUT";
  case BakeJobKind::EnvmapPrefilter:
    return "Envmap Prefilter";
  }
  return "Unknown";
}

[[nodiscard]] uint32_t bakeTileSizeForProfile(BakeryExecutionProfile profile) {
  switch (profile) {
  case BakeryExecutionProfile::Interactive:
    return 64u;
  case BakeryExecutionProfile::Balanced:
    return 128u;
  case BakeryExecutionProfile::Fast:
    return 4096u;
  }
  return 64u;
}

constexpr size_t kMaxQueuedJobs = 512u;
constexpr size_t kMaxWriteQueueEntries = 32u;
constexpr size_t kMaxRetainedTerminalJobs = 64u;

} // namespace

struct BakerySystem::Impl {
  struct BrdfJobData {
    detail::BrdfBakePlan plan{};
    detail::BrdfBakeGpuState gpu{};
    std::vector<std::byte> outputBytes{};
  };

  struct EnvJobData {
    detail::EnvBakePlan plan{};
    detail::EnvBakeGpuState gpu{};
    std::optional<detail::EnvWritePayload> payload{};
  };

  struct JobRecord {
    BakeJobId id{};
    BakeRequest request{};
    BakeJobKind kind = BakeJobKind::BrdfLut;
    BakeJobState state = BakeJobState::Queued;
    bool cancelRequested = false;
    uint32_t completedSteps = 0;
    uint32_t totalSteps = 0;
    std::string summary{};
    std::string error{};
    std::variant<std::monostate, BrdfJobData, EnvJobData> data{};
  };

  struct BrdfWriteTask {
    std::filesystem::path outputPath{};
    std::vector<std::byte> bytes{};
  };

  struct EnvWriteTask {
    detail::EnvWritePayload payload{};
  };

  struct WriteTask {
    BakeJobId jobId{};
    std::variant<BrdfWriteTask, EnvWriteTask> payload{};
  };

  struct WriteCompletion {
    BakeJobId jobId{};
    Result<bool, std::string> result =
        Result<bool, std::string>::makeResult(true);
    std::string summary{};
  };

  struct WriteWorker {
    std::mutex mutex{};
    std::condition_variable cv{};
    std::deque<WriteTask> queue{};
    std::deque<WriteCompletion> completions{};
    std::thread worker{};
    bool stopRequested = false;
  };

  explicit Impl(BakerySystem::CreateDesc desc)
      : gpu(&desc.gpu), config(std::move(desc.config)), profile(desc.profile) {
    worker.worker = std::thread([this]() { workerLoop(); });
  }

  ~Impl() {
    {
      std::scoped_lock lock(worker.mutex);
      worker.stopRequested = true;
    }
    worker.cv.notify_one();
    if (worker.worker.joinable()) {
      worker.worker.join();
    }

    for (JobRecord &job : jobs) {
      cleanupJobGpu(job);
    }
  }

  [[nodiscard]] Result<BakeJobId, std::string> enqueue(BakeRequest request) {
    if (jobs.size() >= kMaxQueuedJobs) {
      return Result<BakeJobId, std::string>::makeError(
          "BakerySystem: job queue is full");
    }

    const BakeJobId id{.value = nextJobId++};
    JobRecord job{};
    job.id = id;
    job.kind = std::holds_alternative<BrdfLutBakeRequest>(request)
                   ? BakeJobKind::BrdfLut
                   : BakeJobKind::EnvmapPrefilter;
    job.request = std::move(request);
    job.state = BakeJobState::Queued;
    job.summary = "Queued";
    jobs.push_back(std::move(job));
    const std::string_view kindName = jobKindName(jobs.back().kind);
    NURI_LOG_INFO("BakerySystem: queued job #%llu (%.*s)",
                  static_cast<unsigned long long>(id.value),
                  static_cast<int>(kindName.size()), kindName.data());
    return Result<BakeJobId, std::string>::makeResult(id);
  }

  void requestCancel(BakeJobId id) {
    for (JobRecord &job : jobs) {
      if (job.id.value == id.value) {
        job.cancelRequested = true;
        if (job.summary.empty()) {
          job.summary = "Cancel requested";
        }
        const std::string_view kindName = jobKindName(job.kind);
        NURI_LOG_INFO("BakerySystem: cancel requested for job #%llu (%.*s)",
                      static_cast<unsigned long long>(job.id.value),
                      static_cast<int>(kindName.size()), kindName.data());
        return;
      }
    }
  }

  [[nodiscard]] bool hasActiveJobs() const noexcept {
    for (const JobRecord &job : jobs) {
      if (!isTerminal(job.state)) {
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] std::pmr::vector<BakeJobSnapshot>
  snapshotJobs(std::pmr::memory_resource *mem) const {
    std::pmr::vector<BakeJobSnapshot> out(
        mem != nullptr ? mem : std::pmr::get_default_resource());
    out.reserve(jobs.size());
    for (const JobRecord &job : jobs) {
      const uint32_t total = job.totalSteps;
      float progress = 0.0f;
      if (job.state == BakeJobState::Succeeded ||
          job.state == BakeJobState::Skipped) {
        progress = 1.0f;
      } else if (total > 0u) {
        progress = std::clamp(static_cast<float>(job.completedSteps) /
                                  static_cast<float>(total),
                              0.0f, 1.0f);
      }
      out.push_back(BakeJobSnapshot{
          .id = job.id,
          .kind = job.kind,
          .state = job.state,
          .completedSteps = job.completedSteps,
          .totalSteps = job.totalSteps,
          .progress01 = progress,
          .summary = job.summary,
          .error = job.error,
      });
    }
    return out;
  }

  void tick() {
    NURI_PROFILER_ZONE("BakerySystem::tick", NURI_PROFILER_COLOR_CREATE);
    consumeWriteCompletions(maxWriteCompletionsPerTick(profile));

    const size_t maxGpuSteps = maxGpuStepsPerTick(profile);
    size_t gpuStepsUsed = 0u;

    for (JobRecord &job : jobs) {
      if (isTerminal(job.state)) {
        continue;
      }

      if (job.cancelRequested && (job.state == BakeJobState::Queued ||
                                  job.state == BakeJobState::CacheCheck ||
                                  job.state == BakeJobState::GpuSetup ||
                                  job.state == BakeJobState::GpuStep ||
                                  job.state == BakeJobState::WriteQueued ||
                                  job.state == BakeJobState::WriteInFlight)) {
        cleanupJobGpu(job);
        job.state = BakeJobState::Canceled;
        job.summary = "Canceled";
        continue;
      }

      switch (job.state) {
      case BakeJobState::Queued:
        job.state = BakeJobState::CacheCheck;
        job.summary = "Cache check pending";
        break;
      case BakeJobState::CacheCheck:
        handleCacheCheck(job);
        break;
      case BakeJobState::GpuSetup:
        handleGpuSetup(job);
        break;
      case BakeJobState::GpuStep:
        if (gpuStepsUsed >= maxGpuSteps) {
          break;
        }
        ++gpuStepsUsed;
        handleGpuStep(job);
        break;
      case BakeJobState::WriteQueued:
        handleWriteQueued(job);
        break;
      case BakeJobState::WriteInFlight:
      case BakeJobState::Succeeded:
      case BakeJobState::Skipped:
      case BakeJobState::Failed:
      case BakeJobState::Canceled:
        break;
      }
    }
    pruneTerminalJobs();
    NURI_PROFILER_ZONE_END();
  }

  void setExecutionProfile(BakeryExecutionProfile value) { profile = value; }

  [[nodiscard]] BakeryExecutionProfile executionProfile() const noexcept {
    return profile;
  }

  void workerLoop() {
    NURI_PROFILER_THREAD("BakeryWorker");
    while (true) {
      WriteTask task{};
      {
        std::unique_lock<std::mutex> lock(worker.mutex);
        worker.cv.wait(lock, [this]() {
          return worker.stopRequested || !worker.queue.empty();
        });
        if (worker.queue.empty()) {
          if (worker.stopRequested) {
            return;
          }
          continue;
        }

        task = std::move(worker.queue.front());
        worker.queue.pop_front();
      }

      WriteCompletion completion{};
      completion.jobId = task.jobId;
      {
        NURI_PROFILER_ZONE("BakeryWorker::writeTask",
                           NURI_PROFILER_COLOR_CREATE);
        completion.result = std::visit(
            [](auto &payload) -> Result<bool, std::string> {
              using T = std::decay_t<decltype(payload)>;
              if constexpr (std::is_same_v<T, BrdfWriteTask>) {
                return detail::writeBrdfLutKtx2(
                    std::span<const std::byte>(payload.bytes.data(),
                                               payload.bytes.size()),
                    payload.outputPath);
              } else {
                return detail::writeEnvmapPrefilterOutputs(payload.payload);
              }
            },
            task.payload);
        NURI_PROFILER_ZONE_END();
      }
      completion.summary = "Write completed";
      if (completion.result.hasError()) {
        completion.summary = "Write failed";
      }

      {
        std::scoped_lock lock(worker.mutex);
        worker.completions.push_back(std::move(completion));
      }
    }
  }

  void consumeWriteCompletions(size_t maxCount) {
    NURI_PROFILER_ZONE("BakerySystem::consumeWriteCompletions",
                       NURI_PROFILER_COLOR_CREATE);
    size_t consumedCount = 0u;
    while (consumedCount < maxCount) {
      WriteCompletion completion{};
      {
        std::scoped_lock lock(worker.mutex);
        if (worker.completions.empty()) {
          break;
        }
        completion = std::move(worker.completions.front());
        worker.completions.pop_front();
      }
      ++consumedCount;

      JobRecord *job = findJob(completion.jobId);
      if (job == nullptr) {
        continue;
      }
      if (job->state != BakeJobState::WriteInFlight) {
        continue;
      }

      if (job->cancelRequested) {
        job->state = BakeJobState::Canceled;
        job->summary = "Canceled";
        const std::string_view kindName = jobKindName(job->kind);
        NURI_LOG_INFO("BakerySystem: job #%llu (%.*s) canceled",
                      static_cast<unsigned long long>(job->id.value),
                      static_cast<int>(kindName.size()), kindName.data());
        continue;
      }

      if (completion.result.hasError()) {
        setFailed(*job, completion.result.error());
      } else {
        job->state = BakeJobState::Succeeded;
        job->summary = completion.summary;
        job->error.clear();
        const std::string_view kindName = jobKindName(job->kind);
        NURI_LOG_INFO("BakerySystem: job #%llu (%.*s) succeeded",
                      static_cast<unsigned long long>(job->id.value),
                      static_cast<int>(kindName.size()), kindName.data());
      }
    }
    NURI_PROFILER_ZONE_END();
  }

  void handleCacheCheck(JobRecord &job) {
    NURI_PROFILER_ZONE("BakerySystem::cacheCheck", NURI_PROFILER_COLOR_CREATE);
    if (job.kind == BakeJobKind::BrdfLut) {
      const auto *request = std::get_if<BrdfLutBakeRequest>(&job.request);
      if (request == nullptr) {
        setFailed(job, "BakerySystem: BRDF request payload mismatch");
        return;
      }

      auto planResult = detail::planBrdfLutBake(config, request->forceRebuild);
      if (planResult.hasError()) {
        setFailed(job, planResult.error());
        return;
      }

      detail::BrdfBakePlan plan = std::move(planResult.value());
      if (!plan.shouldBake) {
        job.state = BakeJobState::Skipped;
        job.summary = "Up-to-date";
        job.error.clear();
        return;
      }

      BrdfJobData data{};
      data.plan = std::move(plan);
      job.data = std::move(data);
      job.totalSteps = detail::brdfTotalSteps();
      job.completedSteps = 0u;
      job.summary = "Cache check complete";
      job.state = BakeJobState::GpuSetup;
      return;
    }

    const auto *request = std::get_if<EnvmapPrefilterBakeRequest>(&job.request);
    if (request == nullptr) {
      setFailed(job, "BakerySystem: envmap request payload mismatch");
      return;
    }

    auto planResult = detail::planEnvmapPrefilterBake(
        config, request->environmentHdrPath, request->forceRebuild);
    if (planResult.hasError()) {
      setFailed(job, planResult.error());
      return;
    }

    detail::EnvBakePlan plan = std::move(planResult.value());
    if (!plan.shouldBake) {
      job.state = BakeJobState::Skipped;
      job.summary = "Up-to-date";
      job.error.clear();
      return;
    }

    EnvJobData data{};
    data.plan = std::move(plan);
    job.data = std::move(data);
    job.totalSteps = 0u;
    job.completedSteps = 0u;
    job.summary = "Cache check complete";
    job.state = BakeJobState::GpuSetup;
    NURI_PROFILER_ZONE_END();
  }

  void handleGpuSetup(JobRecord &job) {
    NURI_PROFILER_ZONE("BakerySystem::gpuSetup", NURI_PROFILER_COLOR_CREATE);
    if (job.kind == BakeJobKind::BrdfLut) {
      auto *data = std::get_if<BrdfJobData>(&job.data);
      if (data == nullptr) {
        setFailed(job, "BakerySystem: missing BRDF job data");
        return;
      }
      auto setupResult =
          detail::setupBrdfLutBake(*gpu, data->plan.shaderPath, data->gpu);
      if (setupResult.hasError()) {
        setFailed(job, setupResult.error());
        return;
      }
      job.state = BakeJobState::GpuStep;
      job.summary = "GPU setup complete";
      return;
    }

    auto *data = std::get_if<EnvJobData>(&job.data);
    if (data == nullptr) {
      setFailed(job, "BakerySystem: missing envmap job data");
      return;
    }
    if (data->gpu.bakeTileSize == 0u) {
      data->gpu.bakeTileSize = bakeTileSizeForProfile(profile);
    }
    auto setupResult =
        detail::advanceEnvmapPrefilterSetup(*gpu, data->plan, data->gpu);
    if (setupResult.hasError()) {
      setFailed(job, setupResult.error());
      return;
    }
    job.summary = setupResult.value().summary;
    if (setupResult.value().status == detail::EnvSetupStatus::InProgress) {
      return;
    }
    if (setupResult.value().status == detail::EnvSetupStatus::Skipped) {
      job.state = BakeJobState::Skipped;
      job.summary = setupResult.value().summary;
      job.error.clear();
      return;
    }

    job.totalSteps = data->gpu.totalSteps;
    job.completedSteps = 0u;
    job.state = BakeJobState::GpuStep;
    job.summary = setupResult.value().summary;
    NURI_PROFILER_ZONE_END();
  }

  void handleGpuStep(JobRecord &job) {
    NURI_PROFILER_ZONE("BakerySystem::gpuStep", NURI_PROFILER_COLOR_CREATE);
    if (job.kind == BakeJobKind::BrdfLut) {
      auto *data = std::get_if<BrdfJobData>(&job.data);
      if (data == nullptr) {
        setFailed(job, "BakerySystem: missing BRDF job data");
        return;
      }

      auto stepResult = detail::runBrdfLutBakeStep(*gpu, data->gpu);
      if (stepResult.hasError()) {
        setFailed(job, stepResult.error());
        return;
      }
      data->outputBytes = std::move(stepResult.value());
      job.completedSteps = job.totalSteps;
      detail::cleanupBrdfLutBake(*gpu, data->gpu);
      job.state = BakeJobState::WriteQueued;
      job.summary = "GPU bake complete";
      return;
    }

    auto *data = std::get_if<EnvJobData>(&job.data);
    if (data == nullptr) {
      setFailed(job, "BakerySystem: missing envmap job data");
      return;
    }

    auto stepResult = detail::runEnvmapPrefilterBakeOneStep(*gpu, data->gpu);
    if (stepResult.hasError()) {
      setFailed(job, stepResult.error());
      return;
    }

    job.completedSteps = stepResult.value().completedSteps;
    job.totalSteps = stepResult.value().totalSteps;
    if (!stepResult.value().finished) {
      job.summary = "GPU bake in progress";
      if (job.cancelRequested) {
        detail::cleanupEnvmapPrefilterBake(*gpu, data->gpu);
        job.state = BakeJobState::Canceled;
        job.summary = "Canceled";
      }
      return;
    }

    data->payload = detail::collectEnvWritePayload(data->gpu);
    detail::cleanupEnvmapPrefilterBake(*gpu, data->gpu);
    job.state = BakeJobState::WriteQueued;
    job.summary = "GPU bake complete";
    NURI_PROFILER_ZONE_END();
  }

  void handleWriteQueued(JobRecord &job) {
    if (job.cancelRequested) {
      cleanupJobGpu(job);
      job.state = BakeJobState::Canceled;
      job.summary = "Canceled";
      return;
    }

    WriteTask task{};
    task.jobId = job.id;
    if (job.kind == BakeJobKind::BrdfLut) {
      auto *data = std::get_if<BrdfJobData>(&job.data);
      if (data == nullptr) {
        setFailed(job, "BakerySystem: missing BRDF write data");
        return;
      }
      task.payload = BrdfWriteTask{
          .outputPath = data->plan.outputPath,
          .bytes = std::move(data->outputBytes),
      };
    } else {
      auto *data = std::get_if<EnvJobData>(&job.data);
      if (data == nullptr || !data->payload.has_value()) {
        setFailed(job, "BakerySystem: missing envmap write payload");
        return;
      }
      task.payload = EnvWriteTask{
          .payload = std::move(*data->payload),
      };
      data->payload.reset();
    }

    std::string enqueueError;
    {
      std::scoped_lock lock(worker.mutex);
      if (worker.stopRequested) {
        enqueueError = "BakerySystem: write worker is stopping";
      } else if (worker.queue.size() >= kMaxWriteQueueEntries) {
        enqueueError = "BakerySystem: write queue is full";
      } else {
        worker.queue.push_back(std::move(task));
      }
    }
    if (!enqueueError.empty()) {
      setFailed(job, std::move(enqueueError));
      return;
    }
    worker.cv.notify_one();

    job.state = BakeJobState::WriteInFlight;
    job.summary = "Write queued";
  }

  void cleanupJobGpu(JobRecord &job) {
    if (job.kind == BakeJobKind::BrdfLut) {
      if (auto *data = std::get_if<BrdfJobData>(&job.data); data != nullptr) {
        detail::cleanupBrdfLutBake(*gpu, data->gpu);
      }
      return;
    }

    if (auto *data = std::get_if<EnvJobData>(&job.data); data != nullptr) {
      detail::cleanupEnvmapPrefilterBake(*gpu, data->gpu);
    }
  }

  void setFailed(JobRecord &job, std::string errorText) {
    cleanupJobGpu(job);
    job.state = BakeJobState::Failed;
    job.summary = "Failed";
    job.error = std::move(errorText);
    const std::string_view kindName = jobKindName(job.kind);
    NURI_LOG_WARNING("BakerySystem: job #%llu (%.*s) failed: %s",
                     static_cast<unsigned long long>(job.id.value),
                     static_cast<int>(kindName.size()), kindName.data(),
                     job.error.c_str());
  }

  [[nodiscard]] JobRecord *findJob(BakeJobId id) {
    for (JobRecord &job : jobs) {
      if (job.id.value == id.value) {
        return &job;
      }
    }
    return nullptr;
  }

  void pruneTerminalJobs() {
    size_t terminalCount = 0u;
    for (const JobRecord &job : jobs) {
      if (isTerminal(job.state)) {
        ++terminalCount;
      }
    }
    if (terminalCount <= kMaxRetainedTerminalJobs) {
      return;
    }
    size_t toRemove = terminalCount - kMaxRetainedTerminalJobs;
    std::erase_if(jobs, [&](const JobRecord &job) {
      if (toRemove == 0u) {
        return false;
      }
      if (isTerminal(job.state)) {
        --toRemove;
        return true;
      }
      return false;
    });
  }

  GPUDevice *gpu = nullptr;
  RuntimeConfig config{};
  BakeryExecutionProfile profile = BakeryExecutionProfile::Interactive;
  std::vector<JobRecord> jobs{};
  uint64_t nextJobId = 1u;
  WriteWorker worker{};
};

Result<std::unique_ptr<BakerySystem>, std::string>
BakerySystem::create(CreateDesc desc) {
  auto system =
      std::unique_ptr<BakerySystem>(new BakerySystem(std::move(desc)));
  return Result<std::unique_ptr<BakerySystem>, std::string>::makeResult(
      std::move(system));
}

BakerySystem::BakerySystem(CreateDesc desc)
    : impl_(std::make_unique<Impl>(std::move(desc))) {}

BakerySystem::~BakerySystem() = default;

Result<BakeJobId, std::string> BakerySystem::enqueue(BakeRequest request) {
  return impl_->enqueue(std::move(request));
}

void BakerySystem::tick() { impl_->tick(); }

std::pmr::vector<BakeJobSnapshot>
BakerySystem::snapshotJobs(std::pmr::memory_resource *mem) const {
  return impl_->snapshotJobs(mem);
}

bool BakerySystem::hasActiveJobs() const noexcept {
  return impl_->hasActiveJobs();
}

void BakerySystem::requestCancel(BakeJobId id) { impl_->requestCancel(id); }

void BakerySystem::setExecutionProfile(BakeryExecutionProfile profile) {
  impl_->setExecutionProfile(profile);
}

BakeryExecutionProfile BakerySystem::executionProfile() const noexcept {
  return impl_->executionProfile();
}

} // namespace nuri::bakery
