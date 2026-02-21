#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
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

  struct WriteJob;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace nuri
