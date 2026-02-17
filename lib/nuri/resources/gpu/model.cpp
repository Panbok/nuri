#include "nuri/resources/gpu/model.h"

#include "nuri/core/log.h"
#include "nuri/core/pmr_scratch.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/resources/mesh_importer.h"

#include <bit>
#include <cmath>

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

void packVertices(std::span<const Vertex> vertices,
                  std::pmr::vector<PackedVertexWords> &packed) {
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
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  ScratchArena scratch;
  ScopedScratch scopedScratch(scratch);
  std::pmr::vector<PackedVertexWords> packedVertices(scopedScratch.resource());
  packVertices(data.vertices, packedVertices);

  const std::span<const std::byte> vertexBytes{
      reinterpret_cast<const std::byte *>(packedVertices.data()),
      packedVertices.size() * sizeof(PackedVertexWords)};
  const std::span<const std::byte> indexBytes{
      reinterpret_cast<const std::byte *>(data.indices.data()),
      data.indices.size() * sizeof(uint32_t)};

  auto geometryResult = gpu.allocateGeometry(
      vertexBytes, static_cast<uint32_t>(data.vertices.size()), indexBytes,
      static_cast<uint32_t>(data.indices.size()), debugName);
  if (geometryResult.hasError()) {
    return Result<std::unique_ptr<Model>, std::string>::makeError(
        geometryResult.error());
  }

  std::vector<Submesh> submeshes(data.submeshes.begin(), data.submeshes.end());
  const BoundingBox bounds = computeModelBounds(data.vertices);

  return Result<std::unique_ptr<Model>, std::string>::makeResult(
      std::unique_ptr<Model>(new Model(
          gpu, geometryResult.value(), std::move(submeshes),
          static_cast<uint32_t>(data.vertices.size()),
          static_cast<uint32_t>(data.indices.size()), std::move(bounds))));
}

Result<std::unique_ptr<Model>, std::string> Model::createFromFile(
    GPUDevice &gpu, std::string_view path, const MeshImportOptions &options,
    std::pmr::memory_resource *mem, std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  auto meshDataResult = MeshImporter::loadFromFile(path, options, mem);
  if (meshDataResult.hasError()) {
    const std::string pathStr{path};
    NURI_LOG_WARNING("Model::createFromFile: Failed to load mesh '%s': %s",
                     pathStr.c_str(), meshDataResult.error().c_str());
    return Result<std::unique_ptr<Model>, std::string>::makeError(
        meshDataResult.error());
  }

  auto modelResult = create(gpu, meshDataResult.value(), debugName);
  if (modelResult.hasError()) {
    const std::string pathStr{path};
    NURI_LOG_WARNING(
        "Model::createFromFile: Failed to create model from '%s': %s",
        pathStr.c_str(), modelResult.error().c_str());
    return Result<std::unique_ptr<Model>, std::string>::makeError(
        modelResult.error());
  }

  NURI_LOG_DEBUG("Model::createFromFile: Created model from file '%.*s'",
                 static_cast<int>(path.size()), path.data());
  return modelResult;
}

} // namespace nuri
