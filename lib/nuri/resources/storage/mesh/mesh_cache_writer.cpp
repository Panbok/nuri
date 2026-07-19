#include "nuri/resources/storage/mesh/mesh_cache_writer.h"
#include "nuri/core/log.h"
#include "nuri/pch.h"
#include "nuri/resources/storage/cache_utils.h"
namespace nuri {

MeshCacheWriterService &MeshCacheWriterService::instance() {
  static MeshCacheWriterService writer;
  return writer;
}

MeshCacheWriterService::MeshCacheWriterService()
    : worker_([this]() { workerLoop(); }) {}

MeshCacheWriterService::~MeshCacheWriterService() {
  {
    std::scoped_lock lock(mutex_);
    stopRequested_ = true;
  }
  cv_.notify_one();
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
  {
    std::unique_lock<std::mutex> lock(mutex_);
    const bool drained = drainedCv_.wait_until(
        lock, deadline, [this]() { return queue_.empty() && !activeWrite_; });
    if (!drained) {
      queue_.clear();
    }
  }
  if (worker_.joinable()) {
    worker_.join();
  }
}

void MeshCacheWriterService::enqueue(std::filesystem::path destinationPath,
                                     std::vector<std::byte> fileBytes) {
  if (destinationPath.empty() || fileBytes.empty()) {
    return;
  }
  constexpr size_t kMaxQueueEntries = 32;
  {
    std::scoped_lock lock(mutex_);
    if (stopRequested_) {
      return;
    }
    if (queue_.size() >= kMaxQueueEntries) {
      NURI_LOG_WARNING(
          "MeshCacheWriterService::enqueue: queue full, dropping cache write");
      return;
    }
    queue_.push_back(WriteJob{
        .destinationPath = std::move(destinationPath),
        .fileBytes = std::move(fileBytes),
    });
  }
  cv_.notify_one();
}

void MeshCacheWriterService::workerLoop() {
  while (true) {
    WriteJob job{};
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this]() { return stopRequested_ || !queue_.empty(); });
      if (queue_.empty()) {
        drainedCv_.notify_all();
        if (stopRequested_) {
          return;
        }
        continue;
      }
      job = std::move(queue_.front());
      queue_.pop_front();
      activeWrite_ = true;
    }
    const auto writeResult =
        writeBinaryFileAtomic(job.destinationPath, job.fileBytes);
    bool logWriteError = false;
    {
      std::scoped_lock lock(mutex_);
      logWriteError = !stopRequested_;
    }
    if (logWriteError && writeResult.hasError()) {
      NURI_LOG_WARNING(
          "MeshCacheWriterService::workerLoop: failed to write cache '%s': %s",
          job.destinationPath.string().c_str(), writeResult.error().c_str());
    }
    {
      std::scoped_lock lock(mutex_);
      activeWrite_ = false;
      if (queue_.empty()) {
        drainedCv_.notify_all();
      }
    }
  }
}

} // namespace nuri
