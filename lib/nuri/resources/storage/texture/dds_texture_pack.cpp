#include "nuri/resources/storage/texture/dds_texture_pack.h"
#include "nuri/pch.h"
#include "nuri/resources/gpu/resource_keys.h"
#include "nuri/resources/storage/binary_io.h"
#include "nuri/resources/storage/cache_utils.h"
#include "nuri/resources/storage/material/material_cache_utils.h"
#include "nuri/utils/env_utils.h"
#include <atomic>
#include <fstream>
#include <numeric>
#include <unordered_set>
namespace nuri {
namespace {
constexpr std::array<char, 8> kPackMagic{'N', 'U', 'R', 'I',
                                         'D', 'D', 'S', '\0'};
constexpr uint32_t kPackSchemaVersion = 1u;
constexpr uint64_t kDataAlignment = 4096u;
constexpr size_t kCopyBufferSize = 4u * 1024u * 1024u;
#pragma pack(push, 1)
struct DdsTexturePackHeaderDisk {
  std::array<char, 8> magic = kPackMagic;
  uint32_t schemaVersion = kPackSchemaVersion;
  uint32_t profileVersion = kDdsTexturePackProfileVersion;
  uint32_t entryCount = 0u;
  uint32_t reserved = 0u;
  uint64_t sceneSourceSizeBytes = 0u;
  int64_t sceneSourceWriteTimeTicks = 0;
  uint64_t indexOffset = 0u;
  uint64_t pathTableOffset = 0u;
  uint64_t dataOffset = 0u;
  uint64_t fileSizeBytes = 0u;
};
struct DdsTexturePackEntryDisk {
  uint64_t pathHash = 0u;
  uint64_t sourceSizeBytes = 0u;
  int64_t sourceWriteTimeTicks = 0;
  uint64_t dataOffset = 0u;
  uint64_t dataSizeBytes = 0u;
  uint64_t pathOffset = 0u;
  uint32_t pathSizeBytes = 0u;
  uint32_t reserved = 0u;
};
#pragma pack(pop)
struct AtomicDdsTexturePackTelemetry {
  std::atomic<uint64_t> hits{0u};
  std::atomic<uint64_t> misses{0u};
  std::atomic<uint64_t> stale{0u};
  std::atomic<uint64_t> corrupt{0u};
  std::atomic<uint64_t> builds{0u};
  std::atomic<uint64_t> buildFailures{0u};
  std::atomic<uint64_t> readFailures{0u};
  std::atomic<uint64_t> entriesServed{0u};
  std::atomic<uint64_t> bytesServed{0u};
  std::atomic<uint64_t> buildSourceBytesRead{0u};
  std::atomic<uint64_t> buildTimeNs{0u};
  std::atomic<uint64_t> openTimeNs{0u};
  std::atomic<uint64_t> readTimeNs{0u};
};
AtomicDdsTexturePackTelemetry gDdsTexturePackTelemetry{};
[[nodiscard]] uint64_t
elapsedNanoseconds(std::chrono::steady_clock::time_point begin) noexcept {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - begin)
          .count());
}
[[nodiscard]] uint64_t hashCanonicalPath(std::string_view path) noexcept {
  Fnv1a64 hash;
  hash.add(path);
  return hash.value();
}
[[nodiscard]] uint64_t alignUp(uint64_t value, uint64_t alignment) noexcept {
  const uint64_t remainder = value % alignment;
  return remainder == 0u ? value : value + (alignment - remainder);
}
struct NormalizedPackSource {
  std::filesystem::path filesystemPath{};
  std::string canonicalPath{};
  TextureSourceFingerprint fingerprint{};
  uint64_t pathHash = 0u;
  uint64_t dataOffset = 0u;
};
[[nodiscard]] Result<std::vector<NormalizedPackSource>, std::string>
normalizeSources(std::span<const DdsTexturePackSource> sources) {
  std::vector<NormalizedPackSource> normalized{};
  normalized.reserve(sources.size());
  std::unordered_set<std::string> seen{};
  seen.reserve(sources.size());
  for (const DdsTexturePackSource &source : sources) {
    const std::filesystem::path sourcePath(source.path);
    std::filesystem::path filesystemPath = normalizeSourcePath(sourcePath);
    const std::string canonicalPath =
        canonicalizeResourcePath(filesystemPath.string());
    if (canonicalPath.empty() || !seen.emplace(canonicalPath).second) {
      continue;
    }
    auto fingerprint = queryTextureSourceFingerprint(filesystemPath);
    if (fingerprint.hasError()) {
      return Result<std::vector<NormalizedPackSource>, std::string>::makeError(
          fingerprint.error());
    }
    if (fingerprint.value().sizeBytes == 0u) {
      return Result<std::vector<NormalizedPackSource>, std::string>::makeError(
          "DdsTexturePack: source is empty '" + canonicalPath + "'");
    }
    normalized.push_back(NormalizedPackSource{
        .filesystemPath = std::move(filesystemPath),
        .canonicalPath = canonicalPath,
        .fingerprint = fingerprint.value(),
        .pathHash = hashCanonicalPath(canonicalPath),
    });
  }
  if (normalized.empty()) {
    return Result<std::vector<NormalizedPackSource>, std::string>::makeError(
        "DdsTexturePack: source list is empty");
  }
  return Result<std::vector<NormalizedPackSource>, std::string>::makeResult(
      std::move(normalized));
}
[[nodiscard]] bool writePadding(std::ofstream &stream, uint64_t byteCount) {
  static constexpr std::array<char, 4096> kZeros{};
  while (byteCount != 0u) {
    const size_t chunk =
        static_cast<size_t>(std::min<uint64_t>(byteCount, kZeros.size()));
    stream.write(kZeros.data(), static_cast<std::streamsize>(chunk));
    byteCount -= chunk;
  }
  return stream.good();
}
[[nodiscard]] Result<uint64_t, std::string>
buildPackFile(const std::filesystem::path &packPath,
              const TextureSourceFingerprint &sceneFingerprint,
              std::vector<NormalizedPackSource> &sources) {
  const auto buildBegin = std::chrono::steady_clock::now();
  const uint64_t indexBytes = sources.size() * sizeof(DdsTexturePackEntryDisk);
  const uint64_t pathTableOffset =
      sizeof(DdsTexturePackHeaderDisk) + indexBytes;
  std::vector<size_t> sortedIndices(sources.size());
  std::iota(sortedIndices.begin(), sortedIndices.end(), 0u);
  std::ranges::sort(sortedIndices, [&](size_t lhs, size_t rhs) {
    if (sources[lhs].pathHash != sources[rhs].pathHash) {
      return sources[lhs].pathHash < sources[rhs].pathHash;
    }
    return sources[lhs].canonicalPath < sources[rhs].canonicalPath;
  });
  uint64_t pathTableEnd = pathTableOffset;
  for (const size_t index : sortedIndices) {
    pathTableEnd += sources[index].canonicalPath.size();
  }
  const uint64_t dataOffset = alignUp(pathTableEnd, kDataAlignment);
  uint64_t fileSizeBytes = dataOffset;
  for (NormalizedPackSource &source : sources) {
    fileSizeBytes = alignUp(fileSizeBytes, kDataAlignment);
    source.dataOffset = fileSizeBytes;
    fileSizeBytes += source.fingerprint.sizeBytes;
  }
  DdsTexturePackHeaderDisk header{
      .magic = kPackMagic,
      .schemaVersion = kPackSchemaVersion,
      .profileVersion = kDdsTexturePackProfileVersion,
      .entryCount = static_cast<uint32_t>(sources.size()),
      .reserved = 0u,
      .sceneSourceSizeBytes = sceneFingerprint.sizeBytes,
      .sceneSourceWriteTimeTicks = sceneFingerprint.writeTimeTicks,
      .indexOffset = sizeof(DdsTexturePackHeaderDisk),
      .pathTableOffset = pathTableOffset,
      .dataOffset = dataOffset,
      .fileSizeBytes = fileSizeBytes,
  };
  std::vector<DdsTexturePackEntryDisk> diskEntries{};
  diskEntries.reserve(sources.size());
  uint64_t pathOffset = pathTableOffset;
  for (const size_t index : sortedIndices) {
    const NormalizedPackSource &source = sources[index];
    diskEntries.push_back(DdsTexturePackEntryDisk{
        .pathHash = source.pathHash,
        .sourceSizeBytes = source.fingerprint.sizeBytes,
        .sourceWriteTimeTicks = source.fingerprint.writeTimeTicks,
        .dataOffset = source.dataOffset,
        .dataSizeBytes = source.fingerprint.sizeBytes,
        .pathOffset = pathOffset,
        .pathSizeBytes = static_cast<uint32_t>(source.canonicalPath.size()),
        .reserved = 0u,
    });
    pathOffset += source.canonicalPath.size();
  }
  std::error_code ec;
  std::filesystem::create_directories(packPath.parent_path(), ec);
  if (ec) {
    return Result<uint64_t, std::string>::makeError(
        "DdsTexturePack: failed to create cache directory '" +
        packPath.parent_path().string() + "': " + ec.message());
  }
  const std::filesystem::path tempPath = temporarySiblingPath(packPath);
  std::ofstream output(tempPath, std::ios::binary | std::ios::trunc);
  const auto fail = [&](std::string message) -> Result<uint64_t, std::string> {
    output.close();
    std::error_code removeEc;
    std::filesystem::remove(tempPath, removeEc);
    gDdsTexturePackTelemetry.buildFailures.fetch_add(1u,
                                                     std::memory_order_relaxed);
    gDdsTexturePackTelemetry.buildTimeNs.fetch_add(
        elapsedNanoseconds(buildBegin), std::memory_order_relaxed);
    return Result<uint64_t, std::string>::makeError(std::move(message));
  };
  if (!output.is_open()) {
    return fail("DdsTexturePack: failed to open temp artifact '" +
                tempPath.string() + "'");
  }
  output.write(reinterpret_cast<const char *>(&header), sizeof(header));
  output.write(reinterpret_cast<const char *>(diskEntries.data()),
               static_cast<std::streamsize>(diskEntries.size() *
                                            sizeof(diskEntries.front())));
  for (const size_t index : sortedIndices) {
    output.write(
        sources[index].canonicalPath.data(),
        static_cast<std::streamsize>(sources[index].canonicalPath.size()));
  }
  if (!output.good() || !writePadding(output, dataOffset - pathTableEnd)) {
    return fail("DdsTexturePack: failed to write artifact index '" +
                tempPath.string() + "'");
  }
  std::vector<char> copyBuffer(kCopyBufferSize);
  uint64_t writeOffset = dataOffset;
  for (const NormalizedPackSource &source : sources) {
    if (source.dataOffset < writeOffset ||
        !writePadding(output, source.dataOffset - writeOffset)) {
      return fail("DdsTexturePack: failed to align artifact payload");
    }
    std::ifstream input(source.filesystemPath, std::ios::binary);
    if (!input.is_open()) {
      return fail("DdsTexturePack: failed to open source '" +
                  source.canonicalPath + "'");
    }
    uint64_t remaining = source.fingerprint.sizeBytes;
    while (remaining != 0u) {
      const size_t chunk =
          static_cast<size_t>(std::min<uint64_t>(remaining, copyBuffer.size()));
      input.read(copyBuffer.data(), static_cast<std::streamsize>(chunk));
      if (input.gcount() != static_cast<std::streamsize>(chunk)) {
        return fail("DdsTexturePack: failed to read source '" +
                    source.canonicalPath + "'");
      }
      output.write(copyBuffer.data(), static_cast<std::streamsize>(chunk));
      if (!output.good()) {
        return fail("DdsTexturePack: failed to write payload for '" +
                    source.canonicalPath + "'");
      }
      remaining -= chunk;
    }
    auto finalFingerprint =
        queryTextureSourceFingerprint(source.filesystemPath);
    if (finalFingerprint.hasError() ||
        finalFingerprint.value() != source.fingerprint) {
      return fail("DdsTexturePack: source changed while packing '" +
                  source.canonicalPath + "'");
    }
    gDdsTexturePackTelemetry.buildSourceBytesRead.fetch_add(
        source.fingerprint.sizeBytes, std::memory_order_relaxed);
    writeOffset = source.dataOffset + source.fingerprint.sizeBytes;
  }
  output.flush();
  if (!output.good()) {
    return fail("DdsTexturePack: failed to flush artifact '" +
                tempPath.string() + "'");
  }
  output.close();
  const uint64_t actualSize = std::filesystem::file_size(tempPath, ec);
  if (ec || actualSize != fileSizeBytes) {
    return fail("DdsTexturePack: artifact size mismatch for '" +
                tempPath.string() + "'");
  }
  if (!replaceFileAtomic(tempPath, packPath)) {
    return fail("DdsTexturePack: failed to publish artifact '" +
                packPath.string() + "'");
  }
  gDdsTexturePackTelemetry.builds.fetch_add(1u, std::memory_order_relaxed);
  gDdsTexturePackTelemetry.buildTimeNs.fetch_add(elapsedNanoseconds(buildBegin),
                                                 std::memory_order_relaxed);
  return Result<uint64_t, std::string>::makeResult(fileSizeBytes);
}
enum class ProbeStatus : uint8_t { Hit, Missing, Stale, Corrupt };
struct ProbeResult {
  ProbeStatus status = ProbeStatus::Missing;
  std::unique_ptr<DdsTexturePack::Impl> impl{};
  std::string error{};
};
} // namespace

struct DdsTexturePack::Impl {
  struct Entry {
    uint64_t pathHash = 0u;
    TextureSourceFingerprint fingerprint{};
    uint64_t dataOffset = 0u;
    uint64_t dataSizeBytes = 0u;
    std::string canonicalPath{};
  };
  RandomAccessFile file{};
  std::filesystem::path path{};
  std::vector<Entry> entries{};
  uint64_t artifactSizeBytes = 0u;
  [[nodiscard]] const Entry *
  findEntry(std::string_view canonicalPath) const noexcept {
    const uint64_t pathHash = hashCanonicalPath(canonicalPath);
    auto it =
        std::ranges::lower_bound(entries, pathHash, {}, [](const Entry &entry) {
          return entry.pathHash;
        });
    for (; it != entries.end() && it->pathHash == pathHash; ++it) {
      if (it->canonicalPath == canonicalPath) {
        return &*it;
      }
    }
    return nullptr;
  }
};

namespace {
[[nodiscard]] ProbeResult
probePack(const std::filesystem::path &packPath,
          const TextureSourceFingerprint &sceneFingerprint,
          std::span<const NormalizedPackSource> sources) {
  std::error_code ec;
  if (!std::filesystem::exists(packPath, ec) || ec) {
    return {.status = ProbeStatus::Missing};
  }
  auto impl = std::make_unique<DdsTexturePack::Impl>();
  const auto openBegin = std::chrono::steady_clock::now();
  const bool opened = impl->file.open(packPath);
  gDdsTexturePackTelemetry.openTimeNs.fetch_add(elapsedNanoseconds(openBegin),
                                                std::memory_order_relaxed);
  if (!opened) {
    return {.status = ProbeStatus::Corrupt,
            .error = "DdsTexturePack: failed to open artifact"};
  }
  const uint64_t fileSizeBytes = impl->file.size();
  if (fileSizeBytes < sizeof(DdsTexturePackHeaderDisk)) {
    return {.status = ProbeStatus::Corrupt,
            .error = "DdsTexturePack: artifact header is truncated"};
  }
  DdsTexturePackHeaderDisk header{};
  if (!impl->file.readAt(
          0u, {reinterpret_cast<std::byte *>(&header), sizeof(header)})) {
    return {.status = ProbeStatus::Corrupt,
            .error = "DdsTexturePack: failed to read artifact header"};
  }
  if (header.magic != kPackMagic ||
      header.schemaVersion != kPackSchemaVersion ||
      header.profileVersion != kDdsTexturePackProfileVersion) {
    return {.status = ProbeStatus::Corrupt,
            .error = "DdsTexturePack: artifact header is incompatible"};
  }
  if (header.fileSizeBytes != fileSizeBytes ||
      header.entryCount != sources.size()) {
    return {.status = ProbeStatus::Stale,
            .error = "DdsTexturePack: artifact size or entry count changed"};
  }
  if (header.sceneSourceSizeBytes != sceneFingerprint.sizeBytes ||
      header.sceneSourceWriteTimeTicks != sceneFingerprint.writeTimeTicks) {
    return {.status = ProbeStatus::Stale,
            .error = "DdsTexturePack: scene source fingerprint changed"};
  }
  const uint64_t indexBytes =
      uint64_t{header.entryCount} * sizeof(DdsTexturePackEntryDisk);
  if (header.indexOffset != sizeof(DdsTexturePackHeaderDisk) ||
      !binaryRangeValid(static_cast<size_t>(fileSizeBytes), header.indexOffset,
                        indexBytes) ||
      header.pathTableOffset != header.indexOffset + indexBytes ||
      header.pathTableOffset > header.dataOffset ||
      header.dataOffset > fileSizeBytes) {
    return {.status = ProbeStatus::Corrupt,
            .error = "DdsTexturePack: artifact index bounds are invalid"};
  }
  impl->path = packPath;
  impl->artifactSizeBytes = fileSizeBytes;
  impl->entries.reserve(header.entryCount);
  uint64_t previousHash = 0u;
  bool firstEntry = true;
  for (uint32_t i = 0u; i < header.entryCount; ++i) {
    DdsTexturePackEntryDisk diskEntry{};
    const uint64_t offset =
        header.indexOffset +
        static_cast<uint64_t>(i) * sizeof(DdsTexturePackEntryDisk);
    if (!impl->file.readAt(offset, {reinterpret_cast<std::byte *>(&diskEntry),
                                    sizeof(diskEntry)})) {
      return {.status = ProbeStatus::Corrupt,
              .error = "DdsTexturePack: failed to read artifact entry"};
    }
    if ((!firstEntry && diskEntry.pathHash < previousHash) ||
        diskEntry.sourceSizeBytes == 0u ||
        diskEntry.dataSizeBytes != diskEntry.sourceSizeBytes ||
        diskEntry.pathSizeBytes == 0u ||
        diskEntry.pathOffset < header.pathTableOffset ||
        !binaryRangeValid(static_cast<size_t>(header.dataOffset),
                          diskEntry.pathOffset, diskEntry.pathSizeBytes) ||
        diskEntry.dataOffset < header.dataOffset ||
        !binaryRangeValid(static_cast<size_t>(fileSizeBytes),
                          diskEntry.dataOffset, diskEntry.dataSizeBytes)) {
      return {.status = ProbeStatus::Corrupt,
              .error = "DdsTexturePack: artifact entry bounds are invalid"};
    }
    firstEntry = false;
    previousHash = diskEntry.pathHash;
    std::string canonicalPath(diskEntry.pathSizeBytes, '\0');
    if (!impl->file.readAt(diskEntry.pathOffset,
                           {reinterpret_cast<std::byte *>(canonicalPath.data()),
                            canonicalPath.size()})) {
      return {.status = ProbeStatus::Corrupt,
              .error = "DdsTexturePack: failed to read artifact path"};
    }
    impl->entries.push_back(DdsTexturePack::Impl::Entry{
        .pathHash = diskEntry.pathHash,
        .fingerprint =
            {
                .sizeBytes = diskEntry.sourceSizeBytes,
                .writeTimeTicks = diskEntry.sourceWriteTimeTicks,
            },
        .dataOffset = diskEntry.dataOffset,
        .dataSizeBytes = diskEntry.dataSizeBytes,
        .canonicalPath = std::move(canonicalPath),
    });
  }
  for (const NormalizedPackSource &source : sources) {
    const DdsTexturePack::Impl::Entry *entry =
        impl->findEntry(source.canonicalPath);
    if (entry == nullptr || entry->fingerprint != source.fingerprint) {
      return {.status = ProbeStatus::Stale,
              .error = "DdsTexturePack: source dependency changed"};
    }
  }
  return {
      .status = ProbeStatus::Hit,
      .impl = std::move(impl),
  };
}
} // namespace

DdsTexturePack::DdsTexturePack(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

DdsTexturePack::~DdsTexturePack() = default;

Result<std::vector<std::byte>, std::string>
DdsTexturePack::readOwned(std::string_view canonicalSourcePath) const {
  if (!impl_) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "DdsTexturePack::readOwned: pack is not open");
  }
  const Impl::Entry *entry = impl_->findEntry(canonicalSourcePath);
  if (entry == nullptr) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "DdsTexturePack::readOwned: source is not present in the pack");
  }
  const auto readBegin = std::chrono::steady_clock::now();
  std::vector<std::byte> bytes(static_cast<size_t>(entry->dataSizeBytes));
  const bool read = impl_->file.readAt(entry->dataOffset, bytes);
  gDdsTexturePackTelemetry.readTimeNs.fetch_add(elapsedNanoseconds(readBegin),
                                                std::memory_order_relaxed);
  if (!read) {
    gDdsTexturePackTelemetry.readFailures.fetch_add(1u,
                                                    std::memory_order_relaxed);
    return Result<std::vector<std::byte>, std::string>::makeError(
        "DdsTexturePack::readOwned: failed to read artifact");
  }
  gDdsTexturePackTelemetry.entriesServed.fetch_add(1u,
                                                   std::memory_order_relaxed);
  gDdsTexturePackTelemetry.bytesServed.fetch_add(entry->dataSizeBytes,
                                                 std::memory_order_relaxed);
  return Result<std::vector<std::byte>, std::string>::makeResult(
      std::move(bytes));
}

std::optional<TextureSourceFingerprint> DdsTexturePack::sourceFingerprint(
    std::string_view canonicalSourcePath) const noexcept {
  if (!impl_) {
    return std::nullopt;
  }
  const Impl::Entry *entry = impl_->findEntry(canonicalSourcePath);
  if (entry == nullptr) {
    return std::nullopt;
  }
  return entry->fingerprint;
}

uint32_t DdsTexturePack::entryCount() const noexcept {
  return impl_ ? static_cast<uint32_t>(impl_->entries.size()) : 0u;
}

uint64_t DdsTexturePack::artifactSizeBytes() const noexcept {
  return impl_ ? impl_->artifactSizeBytes : 0u;
}

const std::filesystem::path &DdsTexturePack::path() const noexcept {
  static const std::filesystem::path kEmptyPath{};
  return impl_ ? impl_->path : kEmptyPath;
}

Result<std::filesystem::path, std::string>
buildDdsTexturePackPath(const std::filesystem::path &sceneSourcePath) {
  auto cacheKey = buildSceneMaterialCacheKey(sceneSourcePath);
  if (cacheKey.hasError()) {
    return Result<std::filesystem::path, std::string>::makeError(
        cacheKey.error());
  }
  const std::optional<std::string> configuredRoot =
      readEnvVar("NURI_TEXTURE_CACHE_ROOT");
  const std::filesystem::path root =
      configuredRoot.has_value()
          ? std::filesystem::path(*configuredRoot) / "scene_packs"
          : cacheKey.value().normalizedSourcePath.parent_path() /
                ".nuri_scene_cache" / "scene_packs";
  const std::string stem =
      cacheKey.value().normalizedSourcePath.stem().string();
  return Result<std::filesystem::path, std::string>::makeResult(
      root / std::format("{}_{}_dds_v{}.ntxp", stem,
                         hexU64(cacheKey.value().sourcePathHash),
                         kDdsTexturePackProfileVersion));
}

Result<DdsTexturePackOpenResult, std::string>
ensureDdsTexturePack(const std::filesystem::path &sceneSourcePath,
                     std::span<const DdsTexturePackSource> sources) {
  auto packPath = buildDdsTexturePackPath(sceneSourcePath);
  if (packPath.hasError()) {
    return Result<DdsTexturePackOpenResult, std::string>::makeError(
        packPath.error());
  }
  auto normalizedSources = normalizeSources(sources);
  if (normalizedSources.hasError()) {
    return Result<DdsTexturePackOpenResult, std::string>::makeError(
        normalizedSources.error());
  }
  auto sceneFingerprint = queryTextureSourceFingerprint(sceneSourcePath);
  if (sceneFingerprint.hasError()) {
    return Result<DdsTexturePackOpenResult, std::string>::makeError(
        sceneFingerprint.error());
  }
  ProbeResult probe = probePack(packPath.value(), sceneFingerprint.value(),
                                normalizedSources.value());
  switch (probe.status) {
  case ProbeStatus::Hit:
    gDdsTexturePackTelemetry.hits.fetch_add(1u, std::memory_order_relaxed);
    return Result<DdsTexturePackOpenResult, std::string>::makeResult(
        DdsTexturePackOpenResult{.pack = std::unique_ptr<DdsTexturePack>(
                                     new DdsTexturePack(std::move(probe.impl))),
                                 .built = false});
  case ProbeStatus::Missing:
    gDdsTexturePackTelemetry.misses.fetch_add(1u, std::memory_order_relaxed);
    break;
  case ProbeStatus::Stale:
    gDdsTexturePackTelemetry.stale.fetch_add(1u, std::memory_order_relaxed);
    break;
  case ProbeStatus::Corrupt:
    gDdsTexturePackTelemetry.corrupt.fetch_add(1u, std::memory_order_relaxed);
    break;
  }
  auto buildResult = buildPackFile(packPath.value(), sceneFingerprint.value(),
                                   normalizedSources.value());
  if (buildResult.hasError()) {
    return Result<DdsTexturePackOpenResult, std::string>::makeError(
        buildResult.error());
  }
  ProbeResult rebuilt = probePack(packPath.value(), sceneFingerprint.value(),
                                  normalizedSources.value());
  if (rebuilt.status != ProbeStatus::Hit || !rebuilt.impl) {
    gDdsTexturePackTelemetry.buildFailures.fetch_add(1u,
                                                     std::memory_order_relaxed);
    return Result<DdsTexturePackOpenResult, std::string>::makeError(
        rebuilt.error.empty()
            ? "DdsTexturePack: rebuilt artifact failed validation"
            : rebuilt.error);
  }
  return Result<DdsTexturePackOpenResult, std::string>::makeResult(
      DdsTexturePackOpenResult{.pack = std::unique_ptr<DdsTexturePack>(
                                   new DdsTexturePack(std::move(rebuilt.impl))),
                               .built = true});
}

DdsTexturePackTelemetry ddsTexturePackTelemetry() noexcept {
  return DdsTexturePackTelemetry{
      .hits = gDdsTexturePackTelemetry.hits.load(std::memory_order_relaxed),
      .misses = gDdsTexturePackTelemetry.misses.load(std::memory_order_relaxed),
      .stale = gDdsTexturePackTelemetry.stale.load(std::memory_order_relaxed),
      .corrupt =
          gDdsTexturePackTelemetry.corrupt.load(std::memory_order_relaxed),
      .builds = gDdsTexturePackTelemetry.builds.load(std::memory_order_relaxed),
      .buildFailures = gDdsTexturePackTelemetry.buildFailures.load(
          std::memory_order_relaxed),
      .readFailures =
          gDdsTexturePackTelemetry.readFailures.load(std::memory_order_relaxed),
      .entriesServed = gDdsTexturePackTelemetry.entriesServed.load(
          std::memory_order_relaxed),
      .bytesServed =
          gDdsTexturePackTelemetry.bytesServed.load(std::memory_order_relaxed),
      .buildSourceBytesRead =
          gDdsTexturePackTelemetry.buildSourceBytesRead.load(
              std::memory_order_relaxed),
      .buildTimeNs =
          gDdsTexturePackTelemetry.buildTimeNs.load(std::memory_order_relaxed),
      .openTimeNs =
          gDdsTexturePackTelemetry.openTimeNs.load(std::memory_order_relaxed),
      .readTimeNs =
          gDdsTexturePackTelemetry.readTimeNs.load(std::memory_order_relaxed),
  };
}

} // namespace nuri
