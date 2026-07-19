#pragma once
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <filesystem>
#include <mutex>
#include <thread>
#include <vector>
namespace nuri {

class MeshCacheWriterService final {
public:
  static MeshCacheWriterService &instance();
  ~MeshCacheWriterService();
  MeshCacheWriterService(const MeshCacheWriterService &) = delete;
  MeshCacheWriterService &operator=(const MeshCacheWriterService &) = delete;
  MeshCacheWriterService(MeshCacheWriterService &&) = delete;
  MeshCacheWriterService &operator=(MeshCacheWriterService &&) = delete;
  void enqueue(std::filesystem::path destinationPath,
               std::vector<std::byte> fileBytes);

private:
  MeshCacheWriterService();
  void workerLoop();
  struct WriteJob {
    std::filesystem::path destinationPath;
    std::vector<std::byte> fileBytes;
  };
  std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable drainedCv_;
  std::deque<WriteJob> queue_;
  bool stopRequested_ = false;
  bool activeWrite_ = false;
  std::thread worker_;
};

} // namespace nuri
