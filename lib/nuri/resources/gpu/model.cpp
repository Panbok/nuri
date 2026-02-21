#include "nuri/pch.h"

#include "nuri/resources/gpu/model.h"

#include "nuri/core/log.h"
#include "nuri/core/pmr_scratch.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/resources/mesh_importer.h"
#include "nuri/resources/storage/mesh/mesh_binary_format.h"
#include "nuri/resources/storage/mesh/mesh_binary_serializer.h"
#include "nuri/resources/storage/mesh/mesh_cache_utils.h"
#include "nuri/resources/storage/mesh/mesh_cache_writer.h"

namespace nuri {
namespace {

struct PackedVertexWords {
  uint32_t word0 = 0;
  uint32_t word1 = 0;
  uint32_t word2 = 0;
  uint32_t word3 = 0;
  uint32_t word4 = 0;
  uint32_t word5 = 0;
  uint32_t word6 = 0;
  uint32_t word7 = 0;
};

static_assert(sizeof(PackedVertexWords) == 32);

uint16_t packSnorm16(float value) {
  const float clamped = std::clamp(value, -1.0f, 1.0f);
  const int32_t quantized =
      static_cast<int32_t>(std::round(clamped * 32767.0f));
  const int32_t clampedQuantized = std::clamp(quantized, -32767, 32767);
  return static_cast<uint16_t>(static_cast<int16_t>(clampedQuantized));
}

uint32_t packSnorm2x16Custom(const glm::vec2 &value) {
  const uint32_t x = packSnorm16(value.x);
  const uint32_t y = packSnorm16(value.y);
  return x | (y << 16u);
}

PackedVertexWords packVertex(const Vertex &vertex) {
  PackedVertexWords packed{};

  glm::vec3 normal = vertex.normal;
  if (glm::dot(normal, normal) > 0.0f) {
    normal = glm::normalize(normal);
  }

  glm::vec4 tangent = vertex.tangent;
  const glm::vec3 tangentXYZ = glm::vec3(tangent);
  if (glm::dot(tangentXYZ, tangentXYZ) > 0.0f) {
    tangent = glm::vec4(glm::normalize(tangentXYZ), tangent.w);
  }

  packed.word0 = std::bit_cast<uint32_t>(vertex.position.x);
  packed.word1 = std::bit_cast<uint32_t>(vertex.position.y);
  packed.word2 = std::bit_cast<uint32_t>(vertex.position.z);
  packed.word3 = glm::packHalf2x16(vertex.uv);
  packed.word4 = packSnorm2x16Custom(glm::vec2(normal.x, normal.y));
  packed.word5 = packSnorm2x16Custom(glm::vec2(normal.z, 0.0f));
  packed.word6 = packSnorm2x16Custom(glm::vec2(tangent.x, tangent.y));
  packed.word7 = packSnorm2x16Custom(glm::vec2(tangent.z, tangent.w));

  return packed;
}

template <typename PackedVector>
void packVertices(std::span<const Vertex> vertices, PackedVector &packed) {
  packed.resize(vertices.size());
  for (size_t i = 0; i < vertices.size(); ++i) {
    packed[i] = packVertex(vertices[i]);
  }
}

BoundingBox computeModelBounds(std::span<const Vertex> vertices) {
  if (vertices.empty()) {
    return BoundingBox(glm::vec3(0.0f), glm::vec3(0.0f));
  }

  glm::vec3 minPos = vertices.front().position;
  glm::vec3 maxPos = vertices.front().position;
  for (size_t i = 1; i < vertices.size(); ++i) {
    const Vertex &vertex = vertices[i];
    minPos = glm::min(minPos, vertex.position);
    maxPos = glm::max(maxPos, vertex.position);
  }
  return BoundingBox(minPos, maxPos);
}

std::vector<std::byte>
packVerticesToByteBuffer(std::span<const Vertex> vertices) {
  ScratchArena scratch;
  ScopedScratch scopedScratch(scratch);
  std::pmr::vector<PackedVertexWords> packed(scopedScratch.resource());
  packVertices(vertices, packed);

  std::vector<std::byte> packedBytes(packed.size() * sizeof(PackedVertexWords));
  if (!packedBytes.empty()) {
    std::memcpy(packedBytes.data(), packed.data(), packedBytes.size());
  }
  return packedBytes;
}

bool isMeshCacheReadEnabled() {
  std::optional<std::string> envValueStorage;
#if defined(_WIN32)
  char *rawValue = nullptr;
  size_t valueLength = 0;
  if (_dupenv_s(&rawValue, &valueLength, "NURI_MESH_CACHE_READ") == 0 &&
      rawValue != nullptr) {
    envValueStorage = rawValue;
    std::free(rawValue);
  }
#else
  if (const char *value = std::getenv("NURI_MESH_CACHE_READ");
      value != nullptr) {
    envValueStorage = value;
  }
#endif
  if (!envValueStorage.has_value()) {
    return true;
  }

  const std::string_view value = envValueStorage.value();
  if (value == "0" || value == "false" || value == "FALSE" || value == "off" ||
      value == "OFF") {
    return false;
  }
  return true;
}

void maybeQueueMeshCacheWrite(const MeshCacheKey &cacheKey,
                              const MeshImportOptions &options,
                              std::span<const std::byte> packedVertexBytes,
                              uint32_t vertexCount,
                              std::span<const uint32_t> indices,
                              std::span<const Submesh> submeshes,
                              const BoundingBox &bounds) {
  if (packedVertexBytes.empty() || indices.empty()) {
    return;
  }

  const MeshSourceFingerprint fingerprint =
      queryMeshSourceFingerprint(cacheKey.normalizedSourcePath);

  MeshBinarySerializeInput input{};
  input.sourcePathHash = cacheKey.sourcePathHash;
  input.importOptionsHash = hashMeshImportOptions(options);
  input.sourceSizeBytes = fingerprint.exists ? fingerprint.sizeBytes : 0u;
  input.sourceMtimeNs = fingerprint.exists ? fingerprint.mtimeNs : 0;
  input.bounds = bounds;
  input.packedVertexBytes = packedVertexBytes;
  input.vertexCount = vertexCount;
  input.vertexStrideBytes = kMeshBinaryPackedVertexStrideBytes;
  input.indices = indices;
  input.submeshes = submeshes;

  auto serializeResult = meshBinarySerialize(input);
  if (serializeResult.hasError()) {
    NURI_LOG_WARNING(
        "Model::createFromFile: Failed to serialize mesh cache '%s': %s",
        cacheKey.cachePath.string().c_str(), serializeResult.error().c_str());
    return;
  }

  MeshCacheWriterService::instance().enqueue(
      cacheKey.cachePath, std::move(serializeResult.value()));
  NURI_LOG_DEBUG("Model::createFromFile: Queued mesh cache write '%s'",
                 cacheKey.cachePath.string().c_str());
}

std::optional<MeshBinaryDecodedMesh>
tryLoadMeshCache(std::string_view sourcePath, const MeshCacheKey &cacheKey,
                 const MeshImportOptions &options) {
  std::error_code ec;
  const bool cacheExists =
      std::filesystem::exists(cacheKey.cachePath, ec) && !ec &&
      std::filesystem::is_regular_file(cacheKey.cachePath, ec) && !ec;
  if (!cacheExists) {
    return std::nullopt;
  }

  auto cacheReadResult = readBinaryFile(cacheKey.cachePath);
  if (cacheReadResult.hasError()) {
    NURI_LOG_WARNING(
        "Model::createFromFile: Failed to read mesh cache '%s': %s",
        cacheKey.cachePath.string().c_str(), cacheReadResult.error().c_str());
    return std::nullopt;
  }

  const MeshSourceFingerprint sourceFingerprint =
      queryMeshSourceFingerprint(cacheKey.normalizedSourcePath);
  MeshBinaryDeserializeContext context{};
  context.expectedSourcePathHash = cacheKey.sourcePathHash;
  context.expectedImportOptionsHash = hashMeshImportOptions(options);
  context.validateSourceFingerprint = true;
  context.sourceExists = sourceFingerprint.exists;
  context.sourceSizeBytes = sourceFingerprint.sizeBytes;
  context.sourceMtimeNs = sourceFingerprint.mtimeNs;

  auto decodeResult = meshBinaryDeserialize(cacheReadResult.value(), context);
  if (decodeResult.hasError()) {
    const MeshBinaryDeserializeError &error = decodeResult.error();
    if (error.isStale()) {
      NURI_LOG_DEBUG(
          "Model::createFromFile: Mesh cache is stale '%s': %s",
          cacheKey.cachePath.string().c_str(), error.message.c_str());
    } else {
      NURI_LOG_WARNING(
          "Model::createFromFile: Failed to decode mesh cache '%s': %s",
          cacheKey.cachePath.string().c_str(), error.message.c_str());
    }
    return std::nullopt;
  }

  NURI_LOG_DEBUG("Model::createFromFile: Loaded mesh cache for '%.*s'",
                 static_cast<int>(sourcePath.size()), sourcePath.data());
  return std::move(decodeResult.value());
}

} // namespace

Model::~Model() {
  if (gpu_ != nullptr && nuri::isValid(geometry_)) {
    gpu_->releaseGeometry(geometry_);
    geometry_ = {};
  }
}

Result<std::unique_ptr<Model>, std::string>
Model::create(GPUDevice &gpu, const MeshData &data,
              std::string_view debugName) {
  const std::vector<std::byte> packedBytes =
      packVerticesToByteBuffer(data.vertices);
  return createFromPackedVertices(
      gpu, data,
      std::span<const std::byte>(packedBytes.data(), packedBytes.size()),
      debugName);
}

Result<std::unique_ptr<Model>, std::string> Model::createFromPackedVertices(
    GPUDevice &gpu, const MeshData &data,
    std::span<const std::byte> packedVertexBytes, std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  const size_t expectedPackedByteCount =
      data.vertices.size() * sizeof(PackedVertexWords);
  if (packedVertexBytes.size() != expectedPackedByteCount) {
    return Result<std::unique_ptr<Model>, std::string>::makeError(
        "Model::createFromPackedVertices: packed vertex byte count mismatch");
  }

  const std::span<const std::byte> vertexBytes{
      packedVertexBytes.data(), packedVertexBytes.size()};
  const std::span<const std::byte> indexBytes{
      reinterpret_cast<const std::byte *>(data.indices.data()),
      data.indices.size() * sizeof(uint32_t)};
  const BoundingBox bounds = computeModelBounds(data.vertices);

  auto geometryResult = gpu.allocateGeometry(
      vertexBytes, static_cast<uint32_t>(data.vertices.size()), indexBytes,
      static_cast<uint32_t>(data.indices.size()), debugName);
  if (geometryResult.hasError()) {
    return Result<std::unique_ptr<Model>, std::string>::makeError(
        geometryResult.error());
  }

  std::vector<Submesh> ownedSubmeshes(data.submeshes.begin(),
                                      data.submeshes.end());
  return Result<std::unique_ptr<Model>, std::string>::makeResult(
      std::unique_ptr<Model>(
          new Model(gpu, geometryResult.value(), std::move(ownedSubmeshes),
                    static_cast<uint32_t>(data.vertices.size()),
                    static_cast<uint32_t>(data.indices.size()), bounds)));
}

Result<std::unique_ptr<Model>, std::string> Model::createFromFile(
    GPUDevice &gpu, std::string_view path, const MeshImportOptions &options,
    std::pmr::memory_resource *mem, std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  const std::filesystem::path sourcePath{std::string(path)};
  auto cacheKeyResult = buildMeshCacheKey(sourcePath, options);
  if (cacheKeyResult.hasError()) {
    NURI_LOG_WARNING(
        "Model::createFromFile: Failed to build mesh cache key for '%.*s': %s",
        static_cast<int>(path.size()), path.data(),
        cacheKeyResult.error().c_str());
  } else if (isMeshCacheReadEnabled()) {
    const MeshCacheKey &cacheKey = cacheKeyResult.value();
    if (auto cachedMesh = tryLoadMeshCache(path, cacheKey, options);
        cachedMesh.has_value()) {
      const std::span<const std::byte> vertexBytes{
          cachedMesh->packedVertexBytes.data(),
          cachedMesh->packedVertexBytes.size()};
      const std::span<const std::byte> indexBytes{
          reinterpret_cast<const std::byte *>(cachedMesh->indices.data()),
          cachedMesh->indices.size() * sizeof(uint32_t)};
      auto geometryResult = gpu.allocateGeometry(
          vertexBytes, cachedMesh->vertexCount, indexBytes,
          static_cast<uint32_t>(cachedMesh->indices.size()), debugName);
      if (!geometryResult.hasError()) {
        return Result<std::unique_ptr<Model>, std::string>::makeResult(
            std::unique_ptr<Model>(new Model(
                gpu, geometryResult.value(), std::move(cachedMesh->submeshes),
                cachedMesh->vertexCount,
                static_cast<uint32_t>(cachedMesh->indices.size()),
                cachedMesh->bounds)));
      }
      NURI_LOG_WARNING(
          "Model::createFromFile: Failed to create model from cache '%s': %s",
          cacheKey.cachePath.string().c_str(), geometryResult.error().c_str());
    }
  } else {
    NURI_LOG_DEBUG("Model::createFromFile: Mesh cache read disabled for '%.*s'",
                   static_cast<int>(path.size()), path.data());
  }

  auto meshDataResult = MeshImporter::loadFromFile(path, options, mem);
  if (meshDataResult.hasError()) {
    const std::string pathStr{path};
    NURI_LOG_WARNING("Model::createFromFile: Failed to load mesh '%s': %s",
                     pathStr.c_str(), meshDataResult.error().c_str());
    return Result<std::unique_ptr<Model>, std::string>::makeError(
        meshDataResult.error());
  }

  const MeshData &meshData = meshDataResult.value();
  const bool canWriteMeshCache = !cacheKeyResult.hasError();
  std::vector<std::byte> packedBytes;
  if (canWriteMeshCache) {
    packedBytes = packVerticesToByteBuffer(meshData.vertices);
  }

  auto modelResult = canWriteMeshCache
                         ? createFromPackedVertices(
                               gpu, meshData,
                               std::span<const std::byte>(packedBytes.data(),
                                                          packedBytes.size()),
                               debugName)
                         : create(gpu, meshData, debugName);
  if (modelResult.hasError()) {
    const std::string pathStr{path};
    NURI_LOG_WARNING(
        "Model::createFromFile: Failed to create model from '%s': %s",
        pathStr.c_str(), modelResult.error().c_str());
    return Result<std::unique_ptr<Model>, std::string>::makeError(
        modelResult.error());
  }

  if (canWriteMeshCache) {
    maybeQueueMeshCacheWrite(
        cacheKeyResult.value(), options, packedBytes,
        static_cast<uint32_t>(meshData.vertices.size()),
        std::span<const uint32_t>(meshData.indices.data(), meshData.indices.size()),
        std::span<const Submesh>(meshData.submeshes.data(),
                                 meshData.submeshes.size()),
        modelResult.value()->bounds());
  }

  NURI_LOG_DEBUG("Model::createFromFile: Created model from file '%.*s'",
                 static_cast<int>(path.size()), path.data());
  return modelResult;
}

} // namespace nuri
