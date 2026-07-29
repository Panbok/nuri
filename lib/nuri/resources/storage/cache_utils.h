#pragma once
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>
namespace nuri {

struct MeshImportOptions;

constexpr uint64_t kFnv1a64Offset = 1469598103934665603ull;

class Fnv1a64 {
public:
  explicit Fnv1a64(uint64_t seed = kFnv1a64Offset) : value_(seed) {}
  void add(std::span<const std::byte> bytes) noexcept;
  void add(std::string_view value) noexcept {
    add({reinterpret_cast<const std::byte *>(value.data()), value.size()});
  }
  template <typename T>
    requires std::is_trivially_copyable_v<T>
  void add(const T &value) noexcept {
    add({reinterpret_cast<const std::byte *>(&value), sizeof(value)});
  }
  template <typename... T> void addAll(const T &...values) noexcept {
    (add(values), ...);
  }
  [[nodiscard]] uint64_t value() const noexcept { return value_; }

private:
  uint64_t value_;
};

struct SourceFingerprint {
  bool exists = false;
  uint64_t sizeBytes = 0;
  int64_t mtimeNs = 0;
};

struct SceneMaterialCacheKey {
  std::filesystem::path normalizedSourcePath;
  std::filesystem::path cachePath;
  uint64_t sourcePathHash = 0;
};

struct MeshCacheKey {
  std::filesystem::path normalizedSourcePath;
  std::filesystem::path cachePath;
  uint64_t sourcePathHash = 0;
  uint64_t optionsHash = 0;
};

class RandomAccessFile {
public:
  RandomAccessFile() = default;
  ~RandomAccessFile();
  RandomAccessFile(const RandomAccessFile &) = delete;
  RandomAccessFile &operator=(const RandomAccessFile &) = delete;
  [[nodiscard]] bool open(const std::filesystem::path &path);
  [[nodiscard]] bool readAt(uint64_t offset,
                            std::span<std::byte> destination) const;
  [[nodiscard]] uint64_t size() const noexcept { return size_; }

private:
  void close() noexcept;
  intptr_t file_ = -1;
  size_t size_ = 0;
};

[[nodiscard]] NURI_API std::filesystem::path
normalizeSourcePath(const std::filesystem::path &path);
[[nodiscard]] NURI_API bool
hasExtensionCaseInsensitive(std::string_view path, std::string_view extension);
[[nodiscard]] NURI_API bool
pathHasExtensionCaseInsensitive(const std::filesystem::path &path,
                                std::string_view extension);
[[nodiscard]] NURI_API std::string hexU64(uint64_t value);
[[nodiscard]] NURI_API SourceFingerprint
querySourceFingerprint(const std::filesystem::path &path);
[[nodiscard]] NURI_API uint64_t
hashMeshImportOptions(const MeshImportOptions &options);
[[nodiscard]] NURI_API Result<SceneMaterialCacheKey, std::string>
buildSceneMaterialCacheKey(const std::filesystem::path &sourcePath);
[[nodiscard]] NURI_API Result<MeshCacheKey, std::string>
buildMeshCacheKey(const std::filesystem::path &sourcePath,
                  const MeshImportOptions &options);
[[nodiscard]] NURI_API std::filesystem::path
temporarySiblingPath(const std::filesystem::path &path);
[[nodiscard]] NURI_API bool
replaceFileAtomic(const std::filesystem::path &temporaryPath,
                  const std::filesystem::path &destinationPath) noexcept;
[[nodiscard]] NURI_API Result<std::vector<std::byte>, std::string>
readBinaryFile(const std::filesystem::path &path);
[[nodiscard]] NURI_API Result<bool, std::string>
writeBinaryFileAtomic(const std::filesystem::path &path,
                      std::span<const std::byte> bytes);

} // namespace nuri
