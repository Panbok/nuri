#include "nuri/pch.h"

#include "nuri/resources/storage/mesh/mesh_cache_writer.h"

#include "nuri/core/log.h"
#include "nuri/resources/storage/mesh/mesh_cache_utils.h"

namespace nuri {

struct MeshCacheWriterService::WriteJob {
  std::filesystem::path destinationPath;
  std::vector<std::byte> fileBytes;
};

struct MeshCacheWriterService::Impl {
  std::mutex mutex;
  std::condition_variable cv;
  std::condition_variable drainedCv;
  std::deque<WriteJob> queue;
  std::thread worker;
  bool stopRequested = false;
  bool activeWrite = false;
};

MeshCacheWriterService &MeshCacheWriterService::instance() {
  static MeshCacheWriterService writer;
  return writer;
}

MeshCacheWriterService::MeshCacheWriterService()
    : impl_(std::make_unique<Impl>()) {
  impl_->worker = std::thread([this]() { workerLoop(); });
}

MeshCacheWriterService::~MeshCacheWriterService() {
  if (impl_ == nullptr) {
    return;
  }

  {
    std::scoped_lock lock(impl_->mutex);
    impl_->stopRequested = true;
  }
  impl_->cv.notify_one();

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
  {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    const bool drained = impl_->drainedCv.wait_until(lock, deadline, [this]() {
      return impl_->queue.empty() && !impl_->activeWrite;
    });
    if (!drained) {
      impl_->queue.clear();
    }
  }

  if (impl_->worker.joinable()) {
    impl_->worker.join();
  }
}

void MeshCacheWriterService::enqueue(std::filesystem::path destinationPath,
                                     std::vector<std::byte> fileBytes) {
  if (destinationPath.empty() || fileBytes.empty()) {
    return;
  }

  constexpr size_t kMaxQueueEntries = 32;
  {
    std::scoped_lock lock(impl_->mutex);
    if (impl_->stopRequested) {
      return;
    }
    if (impl_->queue.size() >= kMaxQueueEntries) {
      NURI_LOG_WARNING(
          "MeshCacheWriterService::enqueue: queue full, dropping cache write");
      return;
    }
    impl_->queue.push_back(WriteJob{
        .destinationPath = std::move(destinationPath),
        .fileBytes = std::move(fileBytes),
    });
  }
  impl_->cv.notify_one();
}

void MeshCacheWriterService::workerLoop() {
  while (true) {
    WriteJob job{};
    {
      std::unique_lock<std::mutex> lock(impl_->mutex);
      impl_->cv.wait(lock, [this]() {
        return impl_->stopRequested || !impl_->queue.empty();
      });

      if (impl_->queue.empty()) {
        impl_->drainedCv.notify_all();
        if (impl_->stopRequested) {
          return;
        }
        continue;
      }

      job = std::move(impl_->queue.front());
      impl_->queue.pop_front();
      impl_->activeWrite = true;
    }

    const auto writeResult =
        writeBinaryFileAtomic(job.destinationPath, job.fileBytes);

    bool logWriteError = false;
    {
      std::scoped_lock lock(impl_->mutex);
      logWriteError = !impl_->stopRequested;
    }
    if (logWriteError && writeResult.hasError()) {
      NURI_LOG_WARNING(
          "MeshCacheWriterService::workerLoop: failed to write cache '%s': %s",
          job.destinationPath.string().c_str(), writeResult.error().c_str());
    }

    {
      std::scoped_lock lock(impl_->mutex);
      impl_->activeWrite = false;
      if (impl_->queue.empty()) {
        impl_->drainedCv.notify_all();
      }
    }
  }
}

} // namespace nuri
