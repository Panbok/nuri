#include "nuri/pch.h"

#include "nuri/resources/storage/mesh/mesh_cache_utils.h"

#include "nuri/resources/storage/mesh/mesh_binary_format.h"

namespace nuri {
namespace {

constexpr uint64_t kFnvOffsetBasis = 1469598103934665603ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;
constexpr uint32_t kMeshCacheContentVersion = 2u;

void fnv1aAddByte(uint64_t &hash, uint8_t byte) {
  hash ^= byte;
  hash *= kFnvPrime;
}

void fnv1aAddBytes(uint64_t &hash, std::span<const std::byte> bytes) {
  for (const std::byte value : bytes) {
    fnv1aAddByte(hash, static_cast<uint8_t>(value));
  }
}

template <typename T> void fnv1aAddPod(uint64_t &hash, const T &value) {
  static_assert(std::is_trivially_copyable_v<T>);
  const std::span<const std::byte> bytes(
      reinterpret_cast<const std::byte *>(&value), sizeof(T));
  fnv1aAddBytes(hash, bytes);
}

std::string hexU64(uint64_t value) { return std::format("{:016x}", value); }

} // namespace

std::filesystem::path
normalizeMeshSourcePath(const std::filesystem::path &path) {
  std::error_code ec;
  std::filesystem::path normalized =
      std::filesystem::weakly_canonical(path, ec);
  if (ec) {
    ec.clear();
    normalized = std::filesystem::absolute(path, ec);
  }
  if (ec) {
    normalized = path;
  }
  return normalized.lexically_normal();
}

uint64_t hashMeshImportOptions(const MeshImportOptions &options) {
  uint64_t hash = kFnvOffsetBasis;

  // Bump when cached mesh content semantics change (import/LOD/packing rules).
  fnv1aAddPod(hash, kMeshCacheContentVersion);

  auto addBool = [&hash](bool value) { fnv1aAddByte(hash, value ? 1u : 0u); };

  addBool(options.triangulate);
  addBool(options.genNormals);
  addBool(options.genTangents);
  addBool(options.flipUVs);
  addBool(options.joinIdenticalVertices);
  addBool(options.optimize);
  addBool(options.generateLods);
  fnv1aAddPod(hash, options.lodCount);
  const uint32_t lodRatioCount =
      static_cast<uint32_t>(options.lodTriangleRatios.size());
  fnv1aAddPod(hash, lodRatioCount);
  for (const float ratio : options.lodTriangleRatios) {
    const uint32_t bits = std::bit_cast<uint32_t>(ratio);
    fnv1aAddPod(hash, bits);
  }
  fnv1aAddPod(hash, std::bit_cast<uint32_t>(options.lodTargetError));

  return hash;
}

Result<MeshCacheKey, std::string>
buildMeshCacheKey(const std::filesystem::path &sourcePath,
                  const MeshImportOptions &options) {
  MeshCacheKey key{};
  key.normalizedSourcePath = normalizeMeshSourcePath(sourcePath);

  const std::string normalizedString =
      key.normalizedSourcePath.generic_string();
  if (normalizedString.empty()) {
    return Result<MeshCacheKey, std::string>::makeError(
        "buildMeshCacheKey: normalized source path is empty");
  }

  uint64_t pathHash = kFnvOffsetBasis;
  const std::span<const std::byte> pathBytes(
      reinterpret_cast<const std::byte *>(normalizedString.data()),
      normalizedString.size());
  fnv1aAddBytes(pathHash, pathBytes);

  const uint64_t optionsHash = hashMeshImportOptions(options);

  key.sourcePathHash = pathHash;
  key.optionsHash = optionsHash;

  std::filesystem::path cacheDir =
      key.normalizedSourcePath.parent_path() / ".nuri_mesh_cache";
  std::string stem = key.normalizedSourcePath.stem().string();
  if (stem.empty()) {
    stem = "mesh";
  }

  const std::string fileName =
      std::format("{}_{}_{}_v{}.nmesh", stem, hexU64(pathHash),
                  hexU64(optionsHash), kMeshBinaryFormatMajorVersion);
  key.cachePath = cacheDir / fileName;

  return Result<MeshCacheKey, std::string>::makeResult(std::move(key));
}

MeshSourceFingerprint
queryMeshSourceFingerprint(const std::filesystem::path &sourcePath) {
  MeshSourceFingerprint fingerprint{};

  std::error_code ec;
  const bool exists = std::filesystem::exists(sourcePath, ec);
  if (ec || !exists) {
    return fingerprint;
  }
  if (!std::filesystem::is_regular_file(sourcePath, ec) || ec) {
    return fingerprint;
  }

  const uint64_t sizeBytes = std::filesystem::file_size(sourcePath, ec);
  if (ec) {
    return fingerprint;
  }

  const auto mtime = std::filesystem::last_write_time(sourcePath, ec);
  if (ec) {
    return fingerprint;
  }

  const auto mtimeNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           mtime.time_since_epoch())
                           .count();

  fingerprint.exists = true;
  fingerprint.sizeBytes = sizeBytes;
  fingerprint.mtimeNs = static_cast<int64_t>(mtimeNs);
  return fingerprint;
}

Result<std::vector<std::byte>, std::string>
readBinaryFile(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input.is_open()) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "readBinaryFile: failed to open file '" + path.string() + "'");
  }

  const std::streampos sizePos = input.tellg();
  if (sizePos < 0) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "readBinaryFile: failed to query file size for '" + path.string() +
        "'");
  }

  const uint64_t fileSize = static_cast<uint64_t>(sizePos);
  if (fileSize > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "readBinaryFile: file too large '" + path.string() + "'");
  }

  std::vector<std::byte> bytes(static_cast<size_t>(fileSize));
  input.seekg(0, std::ios::beg);
  if (input.fail()) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "readBinaryFile: failed to seek in file '" + path.string() + "'");
  }
  if (!bytes.empty()) {
    input.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
  }
  if (input.bad() ||
      (!bytes.empty() &&
       input.gcount() != static_cast<std::streamsize>(bytes.size()))) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "readBinaryFile: failed to read file '" + path.string() + "'");
  }

  return Result<std::vector<std::byte>, std::string>::makeResult(
      std::move(bytes));
}

Result<bool, std::string>
writeBinaryFileAtomic(const std::filesystem::path &path,
                      std::span<const std::byte> bytes) {
  if (path.empty()) {
    return Result<bool, std::string>::makeError(
        "writeBinaryFileAtomic: destination path is empty");
  }

  std::error_code ec;
  const std::filesystem::path parent = path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent, ec);
    if (ec) {
      return Result<bool, std::string>::makeError(
          "writeBinaryFileAtomic: failed to create directories for '" +
          path.string() + "'");
    }
  }

  static std::atomic<uint64_t> counter{0};
  const auto threadIdHash = std::hash<std::thread::id>{}(std::this_thread::get_id());
  const std::string tempSuffix =
      std::format(".tmp.{:x}.{}", threadIdHash,
                  counter.fetch_add(1, std::memory_order_relaxed));
  const std::filesystem::path tempPath = path.string() + tempSuffix;
  {
    std::ofstream output(tempPath, std::ios::binary | std::ios::trunc);
    if (!output.is_open()) {
      return Result<bool, std::string>::makeError(
          "writeBinaryFileAtomic: failed to open temp file '" +
          tempPath.string() + "'");
    }
    if (!bytes.empty()) {
      output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    }
    if (!output.good()) {
      return Result<bool, std::string>::makeError(
          "writeBinaryFileAtomic: failed to write temp file '" +
          tempPath.string() + "'");
    }
  }

  std::filesystem::rename(tempPath, path, ec);
  if (ec) {
    ec.clear();
    std::filesystem::remove(path, ec);
    ec.clear();
    std::filesystem::rename(tempPath, path, ec);
    if (ec) {
      std::filesystem::remove(tempPath, ec);
      return Result<bool, std::string>::makeError(
          "writeBinaryFileAtomic: failed to rename temp file to '" +
          path.string() + "'");
    }
  }

  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri
