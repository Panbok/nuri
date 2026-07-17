#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/resources/storage/texture/texture_cache_utils.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace nuri {

constexpr uint32_t kDdsTexturePackProfileVersion = 1u;
constexpr size_t kMinAutomaticDdsTexturePackEntries = 8u;

struct DdsTexturePackSource {
  std::string path{};
};

struct DdsTexturePackTelemetry {
  uint64_t hits = 0u;
  uint64_t misses = 0u;
  uint64_t stale = 0u;
  uint64_t corrupt = 0u;
  uint64_t builds = 0u;
  uint64_t buildFailures = 0u;
  uint64_t readFailures = 0u;
  uint64_t entriesServed = 0u;
  uint64_t bytesServed = 0u;
  uint64_t buildSourceBytesRead = 0u;
  uint64_t buildTimeNs = 0u;
  uint64_t openTimeNs = 0u;
  uint64_t readTimeNs = 0u;
};

struct DdsTexturePackOpenResult;

class NURI_API DdsTexturePack final {
public:
  struct Impl;

  ~DdsTexturePack();

  DdsTexturePack(const DdsTexturePack &) = delete;
  DdsTexturePack &operator=(const DdsTexturePack &) = delete;
  DdsTexturePack(DdsTexturePack &&) = delete;
  DdsTexturePack &operator=(DdsTexturePack &&) = delete;

  [[nodiscard]] Result<std::span<const std::byte>, std::string>
  read(std::string_view canonicalSourcePath);
  // Thread-safe owned read for asynchronous workers. The returned bytes do not
  // alias the pack's legacy reusable read buffer.
  [[nodiscard]] Result<std::vector<std::byte>, std::string>
  readOwned(std::string_view canonicalSourcePath) const;
  [[nodiscard]] std::optional<TextureSourceFingerprint>
  sourceFingerprint(std::string_view canonicalSourcePath) const noexcept;
  [[nodiscard]] uint32_t entryCount() const noexcept;
  [[nodiscard]] uint64_t artifactSizeBytes() const noexcept;
  [[nodiscard]] const std::filesystem::path &path() const noexcept;

private:
  explicit DdsTexturePack(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;

  friend NURI_API Result<DdsTexturePackOpenResult, std::string>
  ensureDdsTexturePack(const std::filesystem::path &,
                       std::span<const DdsTexturePackSource>);
};

struct DdsTexturePackOpenResult {
  std::unique_ptr<DdsTexturePack> pack{};
  bool built = false;
};

[[nodiscard]] NURI_API Result<std::filesystem::path, std::string>
buildDdsTexturePackPath(const std::filesystem::path &sceneSourcePath);

[[nodiscard]] NURI_API Result<DdsTexturePackOpenResult, std::string>
ensureDdsTexturePack(const std::filesystem::path &sceneSourcePath,
                     std::span<const DdsTexturePackSource> sources);

[[nodiscard]] NURI_API DdsTexturePackTelemetry
ddsTexturePackTelemetry() noexcept;

} // namespace nuri
