#include "nuri/resources/storage/cache_utils.h"
#include "nuri/resources/mesh_importer.h"
#include "nuri/resources/storage/binary_io.h"
#include "nuri/resources/storage/material/material_binary_format.h"
#include "nuri/resources/storage/mesh/mesh_binary_format.h"
#if defined(_WIN32)
#include <Windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
namespace nuri {
namespace {
constexpr uint32_t kMeshCacheContentVersion = 14u;

struct SourceCacheIdentity {
  std::filesystem::path normalizedPath;
  uint64_t pathHash = 0u;
  std::string stem;
};

Result<SourceCacheIdentity, std::string>
compileSourceCacheIdentity(const std::filesystem::path &sourcePath,
                           std::string_view fallbackStem,
                           std::string_view caller, uint64_t hashSeed) {
  SourceCacheIdentity identity{};
  identity.normalizedPath = normalizeSourcePath(sourcePath);
  const std::string normalized = identity.normalizedPath.generic_string();
  if (normalized.empty()) {
    return Result<SourceCacheIdentity, std::string>::makeError(
        std::string(caller) + ": normalized source path is empty");
  }
  Fnv1a64 hash(hashSeed);
  hash.add(normalized);
  identity.pathHash = hash.value();
  identity.stem = identity.normalizedPath.stem().string();
  if (identity.stem.empty())
    identity.stem = fallbackStem;
  return Result<SourceCacheIdentity, std::string>::makeResult(
      std::move(identity));
}
} // namespace

void Fnv1a64::add(std::span<const std::byte> bytes) noexcept {
  for (const std::byte byte : bytes) {
    value_ = (value_ ^ static_cast<uint8_t>(byte)) * 1099511628211ull;
  }
}

RandomAccessFile::~RandomAccessFile() { close(); }

bool RandomAccessFile::open(const std::filesystem::path &path) {
  close();
#if defined(_WIN32)
  const HANDLE handle =
      CreateFileW(path.c_str(), GENERIC_READ,
                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                  nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    return false;
  }
  file_ = reinterpret_cast<intptr_t>(handle);
  LARGE_INTEGER size{};
  if (!GetFileSizeEx(handle, &size) || size.QuadPart <= 0) {
    close();
    return false;
  }
  size_ = static_cast<size_t>(size.QuadPart);
#else
  file_ = ::open(path.c_str(), O_RDONLY);
  struct stat info{};
  if (file_ < 0 || ::fstat(static_cast<int>(file_), &info) != 0 ||
      info.st_size <= 0) {
    close();
    return false;
  }
  size_ = static_cast<size_t>(info.st_size);
#endif
  return true;
}

bool RandomAccessFile::readAt(uint64_t offset,
                              std::span<std::byte> destination) const {
  if (!binaryRangeValid(size_, offset, destination.size())) {
    return false;
  }
  size_t completed = 0;
  while (completed < destination.size()) {
    const size_t remaining = destination.size() - completed;
#if defined(_WIN32)
    const DWORD chunk = static_cast<DWORD>(
        std::min<size_t>(remaining, std::numeric_limits<DWORD>::max()));
    const uint64_t chunkOffset = offset + completed;
    OVERLAPPED overlapped{};
    overlapped.Offset = static_cast<DWORD>(chunkOffset);
    overlapped.OffsetHigh = static_cast<DWORD>(chunkOffset >> 32u);
    DWORD bytesRead = 0;
    if (!ReadFile(reinterpret_cast<HANDLE>(file_),
                  destination.data() + completed, chunk, &bytesRead,
                  &overlapped) ||
        bytesRead != chunk) {
      return false;
    }
#else
    const size_t chunk = std::min<size_t>(remaining, SSIZE_MAX);
    if (::pread(static_cast<int>(file_), destination.data() + completed, chunk,
                static_cast<off_t>(offset + completed)) !=
        static_cast<ssize_t>(chunk)) {
      return false;
    }
#endif
    completed += chunk;
  }
  return true;
}

void RandomAccessFile::close() noexcept {
  if (file_ < 0) {
    return;
  }
#if defined(_WIN32)
  CloseHandle(reinterpret_cast<HANDLE>(file_));
#else
  ::close(static_cast<int>(file_));
#endif
  file_ = -1;
  size_ = 0;
}

std::filesystem::path normalizeSourcePath(const std::filesystem::path &path) {
  std::error_code ec;
  auto normalized = std::filesystem::weakly_canonical(path, ec);
  if (ec) {
    normalized = std::filesystem::absolute(path, ec);
  }
  return (ec ? path : normalized).lexically_normal();
}

bool hasExtensionCaseInsensitive(std::string_view path,
                                 std::string_view extension) {
  if (path.size() < extension.size()) {
    return false;
  }
  path.remove_prefix(path.size() - extension.size());
  return std::ranges::equal(path, extension, [](char lhs, char rhs) {
    return std::tolower(static_cast<unsigned char>(lhs)) ==
           std::tolower(static_cast<unsigned char>(rhs));
  });
}

bool pathHasExtensionCaseInsensitive(const std::filesystem::path &path,
                                     std::string_view extension) {
  const std::string pathExtension = path.extension().string();
  if (extension.starts_with('.')) {
    extension.remove_prefix(1u);
  }
  return hasExtensionCaseInsensitive(
      pathExtension.starts_with('.')
          ? std::string_view(pathExtension).substr(1u)
          : std::string_view(pathExtension),
      extension);
}

std::string hexU64(uint64_t value) { return std::format("{:016x}", value); }

SourceFingerprint querySourceFingerprint(const std::filesystem::path &path) {
  std::error_code ec;
  if (!std::filesystem::is_regular_file(path, ec) || ec) {
    return {};
  }
  const uint64_t size = std::filesystem::file_size(path, ec);
  if (ec) {
    return {};
  }
  const auto writeTime = std::filesystem::last_write_time(path, ec);
  if (ec) {
    return {};
  }
  return SourceFingerprint{
      .exists = true,
      .sizeBytes = size,
      .mtimeNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                     writeTime.time_since_epoch())
                     .count(),
  };
}

uint64_t hashMeshImportOptions(const MeshImportOptions &options) {
  Fnv1a64 hash;
  hash.add(kMeshCacheContentVersion);
  for (const bool flag :
       {options.triangulate, options.genNormals, options.calcTangents,
        options.flipUVs, options.joinIdenticalVertices, options.genUVCoords,
        options.removeRedundantMaterials, options.limitBoneWeights,
        options.optimize, options.generateLods, options.generateMeshlets}) {
    hash.add(static_cast<uint8_t>(flag));
  }
  hash.add(options.lodCount);
  hash.add(options.meshletMaxVertices);
  hash.add(options.meshletMaxPrimitives);
  hash.add(std::bit_cast<uint32_t>(options.meshletConeWeight));
  hash.add(static_cast<uint32_t>(options.lodTriangleRatios.size()));
  for (const float ratio : options.lodTriangleRatios)
    hash.add(std::bit_cast<uint32_t>(ratio));
  hash.add(std::bit_cast<uint32_t>(options.lodTargetError));
  return hash.value();
}

Result<SceneMaterialCacheKey, std::string>
buildSceneMaterialCacheKey(const std::filesystem::path &sourcePath) {
  auto identity = compileSourceCacheIdentity(sourcePath, "scene",
                                             "buildSceneMaterialCacheKey",
                                             14695981039346656037ull);
  if (identity.hasError())
    return Result<SceneMaterialCacheKey, std::string>::makeError(
        identity.error());
  SourceCacheIdentity source = std::move(identity.value());
  SceneMaterialCacheKey key{
      .normalizedSourcePath = std::move(source.normalizedPath),
      .sourcePathHash = source.pathHash,
  };
  key.cachePath =
      key.normalizedSourcePath.parent_path() / ".nuri_scene_cache" /
      std::format("{}_{}_v{}.nmat", source.stem, hexU64(key.sourcePathHash),
                  kMaterialBinaryFormatMajorVersion);
  return Result<SceneMaterialCacheKey, std::string>::makeResult(std::move(key));
}

Result<MeshCacheKey, std::string>
buildMeshCacheKey(const std::filesystem::path &sourcePath,
                  const MeshImportOptions &options) {
  auto identity = compileSourceCacheIdentity(
      sourcePath, "mesh", "buildMeshCacheKey", kFnv1a64Offset);
  if (identity.hasError())
    return Result<MeshCacheKey, std::string>::makeError(identity.error());
  SourceCacheIdentity source = std::move(identity.value());
  MeshCacheKey key{
      .normalizedSourcePath = std::move(source.normalizedPath),
      .sourcePathHash = source.pathHash,
      .optionsHash = hashMeshImportOptions(options),
  };
  key.cachePath =
      key.normalizedSourcePath.parent_path() / ".nuri_mesh_cache" /
      std::format("{}_{}_{}_v{}.nmesh", source.stem, hexU64(key.sourcePathHash),
                  hexU64(key.optionsHash), kMeshBinaryFormatMajorVersion);
  return Result<MeshCacheKey, std::string>::makeResult(std::move(key));
}

std::filesystem::path temporarySiblingPath(const std::filesystem::path &path) {
  static std::atomic<uint64_t> counter{0};
  auto temporary = path;
  temporary += std::format(
      ".tmp.{:x}.{}", std::hash<std::thread::id>{}(std::this_thread::get_id()),
      counter.fetch_add(1, std::memory_order_relaxed));
  return temporary;
}

bool replaceFileAtomic(const std::filesystem::path &temporaryPath,
                       const std::filesystem::path &destinationPath) noexcept {
#if defined(_WIN32)
  if (MoveFileExW(temporaryPath.c_str(), destinationPath.c_str(),
                  MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    return true;
  }
  static std::atomic<uint64_t> counter{0};
  auto retired = destinationPath;
  retired += std::format(".retired.{}",
                         counter.fetch_add(1, std::memory_order_relaxed));
  if (!MoveFileExW(destinationPath.c_str(), retired.c_str(),
                   MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    return false;
  }
  if (MoveFileExW(temporaryPath.c_str(), destinationPath.c_str(),
                  MOVEFILE_WRITE_THROUGH)) {
    DeleteFileW(retired.c_str());
    return true;
  }
  MoveFileExW(retired.c_str(), destinationPath.c_str(),
              MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
  return false;
#else
  std::error_code ec;
  std::filesystem::rename(temporaryPath, destinationPath, ec);
  return !ec;
#endif
}

Result<std::vector<std::byte>, std::string>
readBinaryFile(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  const auto size = input.tellg();
  if (!input || size < 0) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "failed to open binary file '" + path.string() + "'");
  }
  std::vector<std::byte> bytes(static_cast<size_t>(size));
  input.seekg(0);
  input.read(reinterpret_cast<char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "failed to read binary file '" + path.string() + "'");
  }
  return Result<std::vector<std::byte>, std::string>::makeResult(
      std::move(bytes));
}

Result<bool, std::string>
writeBinaryFileAtomic(const std::filesystem::path &path,
                      std::span<const std::byte> bytes) {
  if (path.empty()) {
    return Result<bool, std::string>::makeError("empty binary destination");
  }
  std::error_code ec;
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path(), ec);
  }
  if (ec) {
    return Result<bool, std::string>::makeError(
        "failed to prepare binary destination '" + path.string() + "'");
  }
  const auto temp = temporarySiblingPath(path);
  {
    std::ofstream output(temp, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char *>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    if (!output) {
      std::filesystem::remove(temp, ec);
      return Result<bool, std::string>::makeError(
          "failed to write binary file '" + path.string() + "'");
    }
  }
  if (!replaceFileAtomic(temp, path)) {
    std::filesystem::remove(temp, ec);
    return Result<bool, std::string>::makeError(
        "failed to publish binary file '" + path.string() + "'");
  }
  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri
