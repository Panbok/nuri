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

struct ModelAnimationPackedData {
  BufferLayout<std::vector<std::byte>> skinInfluences{};
  BufferLayout<std::vector<std::byte>> morphMeta{};
  BufferLayout<std::vector<std::byte>> morphDeltas{};
};

namespace {

struct PackedStaticVertexWords {
  uint32_t word0 = 0;
  uint32_t word1 = 0;
  uint32_t word2 = 0;
  uint32_t word3 = 0;
  uint32_t word4 = 0;
};

static_assert(sizeof(PackedStaticVertexWords) == 20);

struct PackedAnimatedVertexWords {
  uint32_t word0 = 0;
  uint32_t word1 = 0;
  uint32_t word2 = 0;
  uint32_t word3 = 0;
  uint32_t word4 = 0;
  uint32_t word5 = 0;
  uint32_t word6 = 0;
  uint32_t word7 = 0;
};

static_assert(sizeof(PackedAnimatedVertexWords) == 32);

struct ModelPackedVertexData {
  PackedVertexFormat format = PackedVertexFormat::StaticQuantized20;
  std::vector<std::byte> vertexBytes;
  BufferLayout<std::vector<std::byte>> staticDecode{};
};

struct PackedSkinInfluenceGpu {
  glm::u32vec4 joints{0u};
  glm::vec4 weights{0.0f};
};
static_assert(sizeof(PackedSkinInfluenceGpu) == 32);

struct PackedMorphDeltaGpu {
  glm::vec4 positionDelta{0.0f};
  glm::vec4 normalDelta{0.0f};
  glm::vec4 tangentDelta{0.0f};
};
static_assert(sizeof(PackedMorphDeltaGpu) == 48);

constexpr float kDirectionEpsilon = 1.0e-10f;

struct ModelAnimationGpuBuffers {
  Model::ModelAnimationGpuView view{};
  std::unique_ptr<Buffer> vertexDecodeBuffer;
  uint64_t vertexDecodeBufferAddress = 0u;
  std::unique_ptr<Buffer> skinInfluenceBuffer;
  std::unique_ptr<Buffer> morphMetaBuffer;
  std::unique_ptr<Buffer> morphDeltaBuffer;
};

void destroyBuffer(GPUDevice &gpu, std::unique_ptr<Buffer> &buffer) {
  if (buffer && buffer->valid()) {
    gpu.destroyBuffer(buffer->handle());
  }
  buffer.reset();
}

void releaseAnimationGpuBuffers(GPUDevice &gpu,
                                ModelAnimationGpuBuffers &buffers) {
  destroyBuffer(gpu, buffers.vertexDecodeBuffer);
  destroyBuffer(gpu, buffers.skinInfluenceBuffer);
  destroyBuffer(gpu, buffers.morphMetaBuffer);
  destroyBuffer(gpu, buffers.morphDeltaBuffer);
  buffers.view = {};
  buffers.vertexDecodeBufferAddress = 0u;
}

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

glm::vec2 encodeOctNormal(glm::vec3 normal) {
  if (glm::dot(normal, normal) <= 0.0f) {
    return glm::vec2(0.0f, 0.0f);
  }
  normal = glm::normalize(normal);
  normal /= (std::abs(normal.x) + std::abs(normal.y) + std::abs(normal.z));
  glm::vec2 encoded = glm::vec2(normal.x, normal.y);
  if (normal.z < 0.0f) {
    const glm::vec2 signBits = glm::vec2(encoded.x >= 0.0f ? 1.0f : -1.0f,
                                         encoded.y >= 0.0f ? 1.0f : -1.0f);
    encoded = (glm::vec2(1.0f) - glm::abs(glm::vec2(encoded.y, encoded.x))) *
              signBits;
  }
  return glm::clamp(encoded, glm::vec2(-1.0f), glm::vec2(1.0f));
}

glm::vec3 sanitizeNormal(glm::vec3 normal) {
  if (glm::dot(normal, normal) > kDirectionEpsilon) {
    return glm::normalize(normal);
  }
  return glm::vec3(0.0f, 1.0f, 0.0f);
}

glm::vec4 sanitizeTangent(glm::vec4 tangent, glm::vec3 normal) {
  glm::vec3 tangentXyz(tangent);
  tangentXyz -= normal * glm::dot(tangentXyz, normal);
  if (glm::dot(tangentXyz, tangentXyz) > kDirectionEpsilon) {
    tangentXyz = glm::normalize(tangentXyz);
  } else {
    tangentXyz = glm::vec3(0.0f);
  }
  const float handedness = tangent.w < 0.0f ? -1.0f : 1.0f;
  return glm::vec4(tangentXyz, handedness);
}

PackedAnimatedVertexWords packAnimatedVertex(const Vertex &vertex) {
  PackedAnimatedVertexWords packed{};
  const glm::vec3 normal = sanitizeNormal(vertex.normal);
  const glm::vec4 tangent = sanitizeTangent(vertex.tangent, normal);
  const glm::vec2 normalOct = encodeOctNormal(normal);
  packed.word0 = std::bit_cast<uint32_t>(vertex.position.x);
  packed.word1 = std::bit_cast<uint32_t>(vertex.position.y);
  packed.word2 = std::bit_cast<uint32_t>(vertex.position.z);
  packed.word3 = packSnorm2x16Custom(normalOct);
  const glm::vec3 tangentXyz(tangent);
  if (glm::dot(tangentXyz, tangentXyz) > kDirectionEpsilon) {
    const glm::vec2 tangentOct = encodeOctNormal(tangentXyz);
    packed.word4 = packSnorm2x16Custom(tangentOct);
    packed.word5 = std::bit_cast<uint32_t>(tangent.w);
  }
  packed.word6 = glm::packHalf2x16(vertex.uv);
  packed.word7 = glm::packHalf2x16(vertex.uv1);
  return packed;
}

PackedStaticVertexWords
packStaticVertex(const Vertex &vertex,
                 const StaticVertexDecodeGpuData &decode) {
  PackedStaticVertexWords packed{};
  const glm::vec3 scale = glm::max(glm::vec3(decode.scale), glm::vec3(1.0e-6f));
  const glm::vec3 normalizedPosition =
      glm::clamp((vertex.position - glm::vec3(decode.offset)) / scale,
                 glm::vec3(0.0f), glm::vec3(1.0f));
  const glm::vec3 normal = sanitizeNormal(vertex.normal);
  const glm::vec2 oct = encodeOctNormal(normal);

  packed.word0 =
      glm::packUnorm2x16(glm::vec2(normalizedPosition.x, normalizedPosition.y));
  packed.word1 = glm::packUnorm2x16(glm::vec2(normalizedPosition.z, 0.0f));
  packed.word2 = packSnorm2x16Custom(oct);
  packed.word3 = glm::packHalf2x16(vertex.uv);
  packed.word4 = glm::packHalf2x16(vertex.uv1);
  return packed;
}

StaticVertexDecodeGpuData
computeStaticDecodeRecord(std::span<const Vertex> vertices) {
  StaticVertexDecodeGpuData decode{};
  if (vertices.empty()) {
    return decode;
  }
  glm::vec3 minPos = vertices.front().position;
  glm::vec3 maxPos = vertices.front().position;
  for (size_t i = 1; i < vertices.size(); ++i) {
    minPos = glm::min(minPos, vertices[i].position);
    maxPos = glm::max(maxPos, vertices[i].position);
  }
  decode.offset = glm::vec4(minPos, 0.0f);
  decode.scale = glm::vec4(glm::max(maxPos - minPos, glm::vec3(1.0e-6f)), 0.0f);
  return decode;
}

template <typename T>
std::vector<std::byte> packPodVectorToBytes(const std::vector<T> &values) {
  static_assert(std::is_trivially_copyable_v<T>);
  std::vector<std::byte> bytes(values.size() * sizeof(T));
  if (!bytes.empty()) {
    std::memcpy(bytes.data(), values.data(), bytes.size());
  }
  return bytes;
}

ModelPackedVertexData packVerticesForModel(const MeshData &data) {
  ModelPackedVertexData packed{};
  const bool animatedFormat =
      !data.skinInfluences.empty() || !data.morphTargets.empty();
  packed.format = animatedFormat ? PackedVertexFormat::AnimatedFloat32
                                 : PackedVertexFormat::StaticQuantized20;

  if (packed.format == PackedVertexFormat::AnimatedFloat32) {
    std::vector<PackedAnimatedVertexWords> vertices(data.vertices.size());
    for (size_t i = 0; i < data.vertices.size(); ++i) {
      vertices[i] = packAnimatedVertex(data.vertices[i]);
    }
    packed.vertexBytes = packPodVectorToBytes(vertices);
    return packed;
  }

  std::vector<PackedStaticVertexWords> vertices(data.vertices.size());
  std::vector<StaticVertexDecodeGpuData> decodeRecords(data.submeshes.size());
  for (size_t submeshIndex = 0; submeshIndex < data.submeshes.size();
       ++submeshIndex) {
    const Submesh &submesh = data.submeshes[submeshIndex];
    if (submesh.vertexCount == 0u) {
      decodeRecords[submeshIndex] = StaticVertexDecodeGpuData{};
      continue;
    }
    const size_t start = submesh.vertexOffset;
    const size_t count = submesh.vertexCount;
    if (start > data.vertices.size() || count > data.vertices.size() - start) {
      decodeRecords[submeshIndex] = StaticVertexDecodeGpuData{};
      continue;
    }
    decodeRecords[submeshIndex] = computeStaticDecodeRecord(
        std::span<const Vertex>(data.vertices.data() + start, count));
    for (size_t vertexIndex = 0; vertexIndex < count; ++vertexIndex) {
      vertices[start + vertexIndex] = packStaticVertex(
          data.vertices[start + vertexIndex], decodeRecords[submeshIndex]);
    }
  }
  packed.vertexBytes = packPodVectorToBytes(vertices);
  packed.staticDecode.data = packPodVectorToBytes(decodeRecords);
  packed.staticDecode.count = static_cast<uint32_t>(decodeRecords.size());
  packed.staticDecode.strideBytes = sizeof(StaticVertexDecodeGpuData);
  return packed;
}

Result<bool, std::string>
validatePackedAnimationSection(std::span<const std::byte> bytes, uint32_t count,
                               uint32_t strideBytes,
                               std::string_view sectionName) {
  if (bytes.empty()) {
    if (count != 0u || strideBytes != 0u) {
      return Result<bool, std::string>::makeError(
          std::string("Model::create: ") + std::string(sectionName) +
          " metadata must be zero when payload is empty");
    }
    return Result<bool, std::string>::makeResult(true);
  }
  if (strideBytes == 0u || (bytes.size() % strideBytes) != 0u) {
    return Result<bool, std::string>::makeError(
        std::string("Model::create: invalid ") + std::string(sectionName) +
        " byte layout");
  }
  if ((bytes.size() / strideBytes) != count) {
    return Result<bool, std::string>::makeError(std::string("Model::create: ") +
                                                std::string(sectionName) +
                                                " count mismatch");
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<ModelAnimationPackedData, std::string>
packAnimationData(const MeshData &data) {
  ModelAnimationPackedData packed{};
  if (!data.skinInfluences.empty()) {
    if (data.skinInfluences.size() != data.vertices.size()) {
      return Result<ModelAnimationPackedData, std::string>::makeError(
          "Model::create: skin influence count must match vertex count");
    }
    std::vector<PackedSkinInfluenceGpu> influences;
    influences.resize(data.skinInfluences.size());
    for (size_t i = 0; i < data.skinInfluences.size(); ++i) {
      influences[i].joints = glm::u32vec4(data.skinInfluences[i].joints);
      influences[i].weights = data.skinInfluences[i].weights;
    }
    packed.skinInfluences.data = packPodVectorToBytes(influences);
    packed.skinInfluences.count = static_cast<uint32_t>(influences.size());
    packed.skinInfluences.strideBytes = sizeof(PackedSkinInfluenceGpu);
  }

  if (!data.morphTargets.empty()) {
    if (data.vertices.empty()) {
      return Result<ModelAnimationPackedData, std::string>::makeError(
          "Model::create: morph targets require a non-empty vertex buffer");
    }

    uint32_t logicalMorphTargetCount = 0u;
    for (const Submesh &submesh : data.submeshes) {
      logicalMorphTargetCount =
          std::max(logicalMorphTargetCount, submesh.morphTargetCount);
    }
    if (logicalMorphTargetCount == 0u) {
      return Result<ModelAnimationPackedData, std::string>::makeError(
          "Model::create: morph target payload has no submesh references");
    }

    std::vector<PackedMorphDeltaGpu> morphDeltas;
    const uint64_t morphDeltaCount =
        static_cast<uint64_t>(logicalMorphTargetCount) *
        static_cast<uint64_t>(data.vertices.size());
    if (morphDeltaCount >
        static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
      return Result<ModelAnimationPackedData, std::string>::makeError(
          "Model::create: morphDeltas allocation for PackedMorphDeltaGpu "
          "overflowed (morphTargets=" +
          std::to_string(logicalMorphTargetCount) +
          ", vertices=" + std::to_string(data.vertices.size()) + ")");
    }
    morphDeltas.resize(static_cast<size_t>(morphDeltaCount));
    for (const Submesh &submesh : data.submeshes) {
      if (submesh.vertexCount == 0u || submesh.morphTargetCount == 0u) {
        continue;
      }
      const uint64_t vertexEnd =
          static_cast<uint64_t>(submesh.vertexOffset) + submesh.vertexCount;
      const uint64_t morphEnd =
          static_cast<uint64_t>(submesh.morphTargetFirst) +
          submesh.morphTargetCount;
      if (vertexEnd > data.vertices.size() ||
          morphEnd > data.morphTargets.size()) {
        return Result<ModelAnimationPackedData, std::string>::makeError(
            "Model::create: morph target range exceeds mesh bounds");
      }
      for (uint32_t localMorphIndex = 0;
           localMorphIndex < submesh.morphTargetCount; ++localMorphIndex) {
        const uint32_t morphIndex = submesh.morphTargetFirst + localMorphIndex;
        const MorphTarget &target = data.morphTargets[morphIndex];
        if (!target.positionDeltas.empty() &&
            target.positionDeltas.size() != submesh.vertexCount) {
          return Result<ModelAnimationPackedData, std::string>::makeError(
              "Model::create: morph position delta count mismatch");
        }
        if (!target.normalDeltas.empty() &&
            target.normalDeltas.size() != submesh.vertexCount) {
          return Result<ModelAnimationPackedData, std::string>::makeError(
              "Model::create: morph normal delta count mismatch");
        }
        if (!target.tangentDeltas.empty() &&
            target.tangentDeltas.size() != submesh.vertexCount) {
          return Result<ModelAnimationPackedData, std::string>::makeError(
              "Model::create: morph tangent delta count mismatch");
        }

        for (uint32_t localVertexIndex = 0;
             localVertexIndex < submesh.vertexCount; ++localVertexIndex) {
          const size_t deltaIndex =
              static_cast<size_t>(localMorphIndex) * data.vertices.size() +
              (submesh.vertexOffset + localVertexIndex);
          PackedMorphDeltaGpu &delta = morphDeltas[deltaIndex];
          if (!target.positionDeltas.empty()) {
            delta.positionDelta =
                glm::vec4(target.positionDeltas[localVertexIndex], 0.0f);
          }
          if (!target.normalDeltas.empty()) {
            delta.normalDelta =
                glm::vec4(target.normalDeltas[localVertexIndex], 0.0f);
          }
          if (!target.tangentDeltas.empty()) {
            delta.tangentDelta =
                glm::vec4(target.tangentDeltas[localVertexIndex], 0.0f);
          }
        }
      }
    }

    const MeshBinaryMorphMetaRecord meta{
        .morphTargetCount = logicalMorphTargetCount,
        .vertexCount = static_cast<uint32_t>(data.vertices.size()),
    };
    packed.morphMeta.data.resize(sizeof(meta));
    std::memcpy(packed.morphMeta.data.data(), &meta, sizeof(meta));
    packed.morphMeta.count = 1u;
    packed.morphMeta.strideBytes = sizeof(MeshBinaryMorphMetaRecord);
    packed.morphDeltas.data = packPodVectorToBytes(morphDeltas);
    packed.morphDeltas.count = static_cast<uint32_t>(morphDeltas.size());
    packed.morphDeltas.strideBytes = sizeof(PackedMorphDeltaGpu);
  }

  return Result<ModelAnimationPackedData, std::string>::makeResult(
      std::move(packed));
}

ModelAnimationPackedData
reconstructAnimationPackedDataFromCache(MeshBinaryDecodedMesh &cachedMesh) {
  ModelAnimationPackedData packed{};
  packed.skinInfluences = std::move(cachedMesh.skinInfluences);
  packed.morphMeta = std::move(cachedMesh.morphMeta);
  packed.morphDeltas = std::move(cachedMesh.morphDeltas);
  return packed;
}

Result<std::unique_ptr<Buffer>, std::string>
createStorageBuffer(GPUDevice &gpu, std::span<const std::byte> bytes,
                    std::string_view debugName) {
  if (bytes.empty()) {
    return Result<std::unique_ptr<Buffer>, std::string>::makeResult(nullptr);
  }
  BufferDesc desc{
      .usage = BufferUsage::Storage,
      .storage = Storage::Device,
      .size = bytes.size(),
      .data = bytes,
  };
  return Buffer::create(gpu, desc, debugName);
}

Result<std::unique_ptr<Buffer>, std::string> createStaticVertexDecodeBuffer(
    GPUDevice &gpu, std::span<const std::byte> staticDecodeBytes,
    std::string_view debugName, uint64_t &outBufferAddress) {
  outBufferAddress = 0u;
  if (staticDecodeBytes.empty()) {
    return Result<std::unique_ptr<Buffer>, std::string>::makeResult(nullptr);
  }

  auto bufferResult = createStorageBuffer(
      gpu, staticDecodeBytes, std::string(debugName) + "_vertex_decode");
  if (bufferResult.hasError()) {
    return Result<std::unique_ptr<Buffer>, std::string>::makeError(
        bufferResult.error());
  }

  std::unique_ptr<Buffer> buffer = std::move(bufferResult.value());
  outBufferAddress = gpu.getBufferDeviceAddress(buffer->handle());
  if (outBufferAddress == 0u) {
    return Result<std::unique_ptr<Buffer>, std::string>::makeError(
        "invalid vertex decode buffer address");
  }

  return Result<std::unique_ptr<Buffer>, std::string>::makeResult(
      std::move(buffer));
}

Result<ModelAnimationGpuBuffers, std::string>
createAnimationGpuBuffers(GPUDevice &gpu,
                          const ModelAnimationPackedData &packedData,
                          std::string_view debugName) {
  ModelAnimationGpuBuffers buffers{};
  auto validateSkinResult = validatePackedAnimationSection(
      std::span<const std::byte>(packedData.skinInfluences.data.data(),
                                 packedData.skinInfluences.data.size()),
      packedData.skinInfluences.count, packedData.skinInfluences.strideBytes,
      "skin influence");
  if (validateSkinResult.hasError()) {
    return Result<ModelAnimationGpuBuffers, std::string>::makeError(
        validateSkinResult.error());
  }
  auto validateMorphMetaResult = validatePackedAnimationSection(
      std::span<const std::byte>(packedData.morphMeta.data.data(),
                                 packedData.morphMeta.data.size()),
      packedData.morphMeta.count, packedData.morphMeta.strideBytes,
      "morph meta");
  if (validateMorphMetaResult.hasError()) {
    return Result<ModelAnimationGpuBuffers, std::string>::makeError(
        validateMorphMetaResult.error());
  }
  auto validateMorphDeltaResult = validatePackedAnimationSection(
      std::span<const std::byte>(packedData.morphDeltas.data.data(),
                                 packedData.morphDeltas.data.size()),
      packedData.morphDeltas.count, packedData.morphDeltas.strideBytes,
      "morph delta");
  if (validateMorphDeltaResult.hasError()) {
    return Result<ModelAnimationGpuBuffers, std::string>::makeError(
        validateMorphDeltaResult.error());
  }

  const bool hasMorphMetaBytes = !packedData.morphMeta.empty();
  const bool hasMorphDeltaBytes = !packedData.morphDeltas.empty();
  if (hasMorphMetaBytes != hasMorphDeltaBytes) {
    return Result<ModelAnimationGpuBuffers, std::string>::makeError(
        "Model::create: morph meta and morph delta payload must be present "
        "together");
  }
  MeshBinaryMorphMetaRecord morphMeta{};
  if (hasMorphMetaBytes) {
    if (packedData.morphMeta.count != 1u ||
        packedData.morphMeta.strideBytes != sizeof(MeshBinaryMorphMetaRecord) ||
        packedData.morphMeta.data.size() != sizeof(MeshBinaryMorphMetaRecord)) {
      return Result<ModelAnimationGpuBuffers, std::string>::makeError(
          "Model::create: invalid morph meta payload");
    }
    std::memcpy(&morphMeta, packedData.morphMeta.data.data(),
                sizeof(morphMeta));
  }
  if (!packedData.skinInfluences.empty()) {
    if (packedData.skinInfluences.strideBytes !=
        sizeof(PackedSkinInfluenceGpu)) {
      return Result<ModelAnimationGpuBuffers, std::string>::makeError(
          "Model::create: invalid skin influence stride");
    }
    auto bufferResult = createStorageBuffer(
        gpu,
        std::span<const std::byte>(packedData.skinInfluences.data.data(),
                                   packedData.skinInfluences.data.size()),
        std::string(debugName) + "_skin_influences");
    if (bufferResult.hasError()) {
      return Result<ModelAnimationGpuBuffers, std::string>::makeError(
          bufferResult.error());
    }
    buffers.skinInfluenceBuffer = std::move(bufferResult.value());
    buffers.view.skinInfluenceBuffer = buffers.skinInfluenceBuffer->handle();
    buffers.view.skinInfluenceCount = packedData.skinInfluences.count;
  }
  if (!packedData.morphMeta.empty()) {
    auto metaResult = createStorageBuffer(
        gpu,
        std::span<const std::byte>(packedData.morphMeta.data.data(),
                                   packedData.morphMeta.data.size()),
        std::string(debugName) + "_morph_meta");
    if (metaResult.hasError()) {
      releaseAnimationGpuBuffers(gpu, buffers);
      return Result<ModelAnimationGpuBuffers, std::string>::makeError(
          metaResult.error());
    }
    buffers.morphMetaBuffer = std::move(metaResult.value());
    buffers.view.morphMetaBuffer = buffers.morphMetaBuffer->handle();
  }
  if (!packedData.morphDeltas.empty()) {
    if (packedData.morphDeltas.strideBytes != sizeof(PackedMorphDeltaGpu)) {
      return Result<ModelAnimationGpuBuffers, std::string>::makeError(
          "Model::create: invalid morph delta stride");
    }
    auto deltaResult = createStorageBuffer(
        gpu,
        std::span<const std::byte>(packedData.morphDeltas.data.data(),
                                   packedData.morphDeltas.data.size()),
        std::string(debugName) + "_morph_deltas");
    if (deltaResult.hasError()) {
      releaseAnimationGpuBuffers(gpu, buffers);
      return Result<ModelAnimationGpuBuffers, std::string>::makeError(
          deltaResult.error());
    }
    buffers.morphDeltaBuffer = std::move(deltaResult.value());
    buffers.view.morphDeltaBuffer = buffers.morphDeltaBuffer->handle();
    buffers.view.morphTargetCount = morphMeta.morphTargetCount;
    buffers.view.morphVertexCount = morphMeta.vertexCount;
  }
  return Result<ModelAnimationGpuBuffers, std::string>::makeResult(
      std::move(buffers));
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

Result<uint32_t, std::string>
computeSourceMaterialCount(std::span<const Submesh> submeshes) {
  if (submeshes.empty()) {
    return Result<uint32_t, std::string>::makeResult(0u);
  }

  uint64_t maxSourceMaterial = 0;
  for (const Submesh &submesh : submeshes) {
    maxSourceMaterial = std::max(maxSourceMaterial,
                                 static_cast<uint64_t>(submesh.materialIndex));
  }
  if (maxSourceMaterial >=
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    return Result<uint32_t, std::string>::makeError(
        "Model::create: source material index exceeds UINT32_MAX");
  }
  return Result<uint32_t, std::string>::makeResult(
      static_cast<uint32_t>(maxSourceMaterial + 1u));
}

Result<bool, std::string>
validateMeshTopology(std::span<const uint32_t> indices, uint32_t vertexCount,
                     std::span<const Submesh> submeshes,
                     std::string_view context) {
  const std::string contextString(context);
  if (vertexCount == 0) {
    return Result<bool, std::string>::makeError(contextString +
                                                ": vertex count is zero");
  }
  if (indices.empty()) {
    return Result<bool, std::string>::makeError(contextString +
                                                ": index buffer is empty");
  }
  if (submeshes.empty()) {
    return Result<bool, std::string>::makeError(contextString +
                                                ": submesh list is empty");
  }

  uint32_t maxIndex = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    const uint32_t indexValue = indices[i];
    maxIndex = std::max(maxIndex, indexValue);
    if (indexValue >= vertexCount) {
      return Result<bool, std::string>::makeError(
          contextString + ": index out of range at position " +
          std::to_string(i) + " (" + std::to_string(indexValue) +
          " >= " + std::to_string(vertexCount) + ")");
    }
  }

  for (size_t submeshIndex = 0; submeshIndex < submeshes.size();
       ++submeshIndex) {
    const Submesh &submesh = submeshes[submeshIndex];
    if (submesh.lodCount == 0 || submesh.lodCount > Submesh::kMaxLodCount) {
      return Result<bool, std::string>::makeError(
          contextString + ": invalid LOD count in submesh " +
          std::to_string(submeshIndex));
    }
    for (uint32_t lodIndex = 0; lodIndex < submesh.lodCount; ++lodIndex) {
      const SubmeshLod &lod = submesh.lods[lodIndex];
      const uint64_t end =
          static_cast<uint64_t>(lod.indexOffset) + lod.indexCount;
      if (end > indices.size()) {
        return Result<bool, std::string>::makeError(
            contextString + ": submesh " + std::to_string(submeshIndex) +
            " LOD " + std::to_string(lodIndex) +
            " index range exceeds index buffer");
      }
    }
  }

  NURI_LOG_DEBUG("%s: mesh validated (vertices=%u indices=%zu submeshes=%zu "
                 "maxIndex=%u)",
                 contextString.c_str(), vertexCount, indices.size(),
                 submeshes.size(), maxIndex);
  return Result<bool, std::string>::makeResult(true);
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

uint32_t meshBinaryLayoutIdForVertexFormat(PackedVertexFormat format) {
  switch (format) {
  case PackedVertexFormat::StaticQuantized20:
    return kMeshBinaryLayoutIdStaticQuantized20;
  case PackedVertexFormat::AnimatedFloat24:
    return kMeshBinaryLayoutIdAnimatedFloat24;
  case PackedVertexFormat::AnimatedFloat32:
    return kMeshBinaryLayoutIdAnimatedFloat32;
  }
  return kMeshBinaryLayoutIdStaticQuantized20;
}

uint32_t packedVertexStrideBytes(PackedVertexFormat format) {
  switch (format) {
  case PackedVertexFormat::StaticQuantized20:
    return sizeof(PackedStaticVertexWords);
  case PackedVertexFormat::AnimatedFloat24:
  case PackedVertexFormat::AnimatedFloat32:
    return sizeof(PackedAnimatedVertexWords);
  }
  return sizeof(PackedStaticVertexWords);
}

PackedVertexFormat packedVertexFormatFromLayoutId(uint32_t layoutId) {
  switch (layoutId) {
  case kMeshBinaryLayoutIdAnimatedFloat32:
    return PackedVertexFormat::AnimatedFloat32;
  case kMeshBinaryLayoutIdAnimatedFloat24:
    return PackedVertexFormat::AnimatedFloat24;
  case kMeshBinaryLayoutIdStaticQuantized20:
  default:
    return PackedVertexFormat::StaticQuantized20;
  }
}

void maybeQueueMeshCacheWrite(
    const MeshCacheKey &cacheKey, const MeshImportOptions &options,
    const ModelPackedVertexData &packedVertexData, uint32_t vertexCount,
    std::span<const uint32_t> indices, std::span<const Submesh> submeshes,
    const BoundingBox &bounds, const ModelAnimationPackedData &animationData) {
  if (packedVertexData.vertexBytes.empty() || indices.empty()) {
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
  input.vertexLayoutId =
      meshBinaryLayoutIdForVertexFormat(packedVertexData.format);
  input.vertices = {
      std::span<const std::byte>(packedVertexData.vertexBytes.data(),
                                 packedVertexData.vertexBytes.size()),
      vertexCount, packedVertexStrideBytes(packedVertexData.format)};
  input.staticVertexDecode = {
      std::span<const std::byte>(packedVertexData.staticDecode.data.data(),
                                 packedVertexData.staticDecode.data.size()),
      packedVertexData.staticDecode.count,
      packedVertexData.staticDecode.strideBytes,
  };
  input.indices = indices;
  input.submeshes = submeshes;
  input.skinInfluences = {
      std::span<const std::byte>(animationData.skinInfluences.data.data(),
                                 animationData.skinInfluences.data.size()),
      animationData.skinInfluences.count,
      animationData.skinInfluences.strideBytes,
  };
  input.morphMeta = {
      std::span<const std::byte>(animationData.morphMeta.data.data(),
                                 animationData.morphMeta.data.size()),
      animationData.morphMeta.count,
      animationData.morphMeta.strideBytes,
  };
  input.morphDeltas = {
      std::span<const std::byte>(animationData.morphDeltas.data.data(),
                                 animationData.morphDeltas.data.size()),
      animationData.morphDeltas.count,
      animationData.morphDeltas.strideBytes,
  };

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
      NURI_LOG_DEBUG("Model::createFromFile: Mesh cache is stale '%s': %s",
                     cacheKey.cachePath.string().c_str(),
                     error.message.c_str());
    } else {
      NURI_LOG_WARNING(
          "Model::createFromFile: Failed to decode mesh cache '%s': %s",
          cacheKey.cachePath.string().c_str(), error.message.c_str());
    }
    return std::nullopt;
  }

  MeshBinaryDecodedMesh decodedMesh = std::move(decodeResult.value());
  auto topologyValidation = validateMeshTopology(
      std::span<const uint32_t>(decodedMesh.indices.data(),
                                decodedMesh.indices.size()),
      decodedMesh.vertices.count,
      std::span<const Submesh>(decodedMesh.submeshes.data(),
                               decodedMesh.submeshes.size()),
      "Model::createFromFile cache validation");
  if (topologyValidation.hasError()) {
    NURI_LOG_WARNING("Model::createFromFile: Rejected mesh cache '%s': %s",
                     cacheKey.cachePath.string().c_str(),
                     topologyValidation.error().c_str());
    return std::nullopt;
  }

  NURI_LOG_DEBUG("Model::createFromFile: Loaded mesh cache for '%.*s'",
                 static_cast<int>(sourcePath.size()), sourcePath.data());
  return decodedMesh;
}

} // namespace

bool ModelAsyncLoad::valid() const noexcept {
  return !sourcePath_.empty() &&
         (warmupCompleted_ || warmupFuture_.valid() || finalized_);
}

bool ModelAsyncLoad::isInFlight() const noexcept {
  if (!valid() || warmupCompleted_ || !warmupFuture_.valid()) {
    return false;
  }
  return warmupFuture_.wait_for(std::chrono::milliseconds(0)) !=
         std::future_status::ready;
}

bool ModelAsyncLoad::isReady() const {
  if (!valid()) {
    return false;
  }
  if (warmupCompleted_) {
    return true;
  }
  if (!warmupFuture_.valid()) {
    return false;
  }
  return warmupFuture_.wait_for(std::chrono::milliseconds(0)) ==
         std::future_status::ready;
}

std::optional<bool> ModelAsyncLoad::cacheHit() const noexcept {
  if (!warmupCompleted_ || !warmupError_.empty()) {
    return std::nullopt;
  }
  return warmupCacheHit_;
}

std::string_view ModelAsyncLoad::warmupError() const noexcept {
  return warmupError_;
}

Result<bool, std::string> ModelAsyncLoad::resolveWarmup() {
  if (!valid()) {
    return Result<bool, std::string>::makeError(
        "ModelAsyncLoad::resolveWarmup: async load handle is invalid");
  }
  if (warmupCompleted_) {
    if (!warmupError_.empty()) {
      return Result<bool, std::string>::makeError(warmupError_);
    }
    return Result<bool, std::string>::makeResult(warmupCacheHit_);
  }
  if (!warmupFuture_.valid()) {
    warmupCompleted_ = true;
    warmupError_ = "ModelAsyncLoad::resolveWarmup: warmup future is invalid";
    return Result<bool, std::string>::makeError(warmupError_);
  }
  if (warmupFuture_.wait_for(std::chrono::milliseconds(0)) !=
      std::future_status::ready) {
    return Result<bool, std::string>::makeError(
        "ModelAsyncLoad::resolveWarmup: warmup is still in progress");
  }

  try {
    Result<bool, std::string> warmupResult = warmupFuture_.get();
    warmupCompleted_ = true;
    if (warmupResult.hasError()) {
      warmupError_ = warmupResult.error();
      return Result<bool, std::string>::makeError(warmupError_);
    }
    warmupCacheHit_ = warmupResult.value();
    return Result<bool, std::string>::makeResult(warmupCacheHit_);
  } catch (const std::exception &e) {
    warmupCompleted_ = true;
    warmupError_ =
        std::string("ModelAsyncLoad::resolveWarmup exception: ") + e.what();
    return Result<bool, std::string>::makeError(warmupError_);
  } catch (...) {
    warmupCompleted_ = true;
    warmupError_ = "ModelAsyncLoad::resolveWarmup unknown exception";
    return Result<bool, std::string>::makeError(warmupError_);
  }
}

Result<std::unique_ptr<Model>, std::string>
ModelAsyncLoad::finalize(GPUDevice &gpu, std::pmr::memory_resource *mem,
                         std::string_view debugName) {
  if (!valid()) {
    return Result<std::unique_ptr<Model>, std::string>::makeError(
        "ModelAsyncLoad::finalize: async load handle is invalid");
  }
  if (finalized_) {
    return Result<std::unique_ptr<Model>, std::string>::makeError(
        "ModelAsyncLoad::finalize: model was already finalized");
  }

  auto warmupResult = resolveWarmup();
  if (warmupResult.hasError() && !warmupCompleted_) {
    return Result<std::unique_ptr<Model>, std::string>::makeError(
        warmupResult.error());
  }

  // Warmup is best-effort; fall back to direct createFromFile path.
  auto modelResult =
      Model::createFromFile(gpu, sourcePath_, options_, mem, debugName);
  if (modelResult.hasError()) {
    return modelResult;
  }
  finalized_ = true;
  return modelResult;
}

Model::~Model() {
  if (gpu_ != nullptr) {
    destroyBuffer(*gpu_, vertexDecodeBuffer_);
    destroyBuffer(*gpu_, skinInfluenceBuffer_);
    destroyBuffer(*gpu_, morphMetaBuffer_);
    destroyBuffer(*gpu_, morphDeltaBuffer_);
    if (nuri::isValid(geometry_)) {
      gpu_->releaseGeometry(geometry_);
      geometry_ = {};
    }
  }
}

uint32_t
Model::materialIndexForSource(uint32_t sourceMaterialIndex) const noexcept {
  if (sourceMaterialIndex >= sourceMaterialToRuntime_.size()) {
    return kInvalidMaterialIndex;
  }
  return sourceMaterialToRuntime_[sourceMaterialIndex];
}

uint32_t Model::materialIndexForSubmesh(uint32_t submeshIndex) const noexcept {
  if (submeshIndex >= submeshes_.size()) {
    return kInvalidMaterialIndex;
  }
  return materialIndexForSource(submeshes_[submeshIndex].materialIndex);
}

bool Model::setMaterialIndexForSource(uint32_t sourceMaterialIndex,
                                      uint32_t materialIndex) noexcept {
  if (sourceMaterialIndex >= sourceMaterialToRuntime_.size()) {
    return false;
  }
  sourceMaterialToRuntime_[sourceMaterialIndex] = materialIndex;
  return true;
}

void Model::setMaterialIndexForAllSources(uint32_t materialIndex) noexcept {
  for (uint32_t &mappedIndex : sourceMaterialToRuntime_) {
    mappedIndex = materialIndex;
  }
}

Result<std::unique_ptr<Model>, std::string>
Model::create(GPUDevice &gpu, const MeshData &data,
              std::string_view debugName) {
  const ModelPackedVertexData packedVertices = packVerticesForModel(data);
  return createFromPackedVertices(
      gpu, data,
      std::span<const std::byte>(packedVertices.vertexBytes.data(),
                                 packedVertices.vertexBytes.size()),
      packedVertices.format,
      std::span<const std::byte>(packedVertices.staticDecode.data.data(),
                                 packedVertices.staticDecode.data.size()),
      packedVertices.staticDecode.count, nullptr, debugName);
}

Result<std::unique_ptr<Model>, std::string> Model::createFromPackedVertices(
    GPUDevice &gpu, const MeshData &data,
    std::span<const std::byte> packedVertexBytes,
    PackedVertexFormat packedVertexFormat,
    std::span<const std::byte> staticDecodeBytes, uint32_t staticDecodeCount,
    const ModelAnimationPackedData *animationPackedData,
    std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  std::pmr::memory_resource *const storageMemory =
      std::pmr::get_default_resource();
  auto sourceMaterialCountResult = computeSourceMaterialCount(
      std::span<const Submesh>(data.submeshes.data(), data.submeshes.size()));
  if (sourceMaterialCountResult.hasError()) {
    return Result<std::unique_ptr<Model>, std::string>::makeError(
        sourceMaterialCountResult.error());
  }
  const size_t expectedPackedByteCount =
      data.vertices.size() * packedVertexStrideBytes(packedVertexFormat);
  if (packedVertexBytes.size() != expectedPackedByteCount) {
    return Result<std::unique_ptr<Model>, std::string>::makeError(
        "Model::createFromPackedVertices: packed vertex byte count mismatch");
  }
  if (packedVertexFormat == PackedVertexFormat::StaticQuantized20) {
    const size_t expectedDecodeBytes = static_cast<size_t>(staticDecodeCount) *
                                       sizeof(StaticVertexDecodeGpuData);
    if (staticDecodeCount != data.submeshes.size() ||
        staticDecodeBytes.size() != expectedDecodeBytes) {
      return Result<std::unique_ptr<Model>, std::string>::makeError(
          "Model::createFromPackedVertices: static vertex decode mismatch");
    }
  } else if (!staticDecodeBytes.empty() || staticDecodeCount != 0u) {
    return Result<std::unique_ptr<Model>, std::string>::makeError(
        "Model::createFromPackedVertices: animated vertex data must not carry "
        "static decode metadata");
  }
  auto topologyValidation = validateMeshTopology(
      std::span<const uint32_t>(data.indices.data(), data.indices.size()),
      static_cast<uint32_t>(data.vertices.size()),
      std::span<const Submesh>(data.submeshes.data(), data.submeshes.size()),
      "Model::createFromPackedVertices");
  if (topologyValidation.hasError()) {
    return Result<std::unique_ptr<Model>, std::string>::makeError(
        topologyValidation.error());
  }

  const std::span<const std::byte> vertexBytes{packedVertexBytes.data(),
                                               packedVertexBytes.size()};
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

  Result<ModelAnimationPackedData, std::string> animationPackResult =
      Result<ModelAnimationPackedData, std::string>::makeResult(
          ModelAnimationPackedData{});
  const ModelAnimationPackedData *resolvedAnimationData = animationPackedData;
  if (resolvedAnimationData == nullptr) {
    animationPackResult = packAnimationData(data);
    if (animationPackResult.hasError()) {
      gpu.releaseGeometry(geometryResult.value());
      return Result<std::unique_ptr<Model>, std::string>::makeError(
          animationPackResult.error());
    }
    resolvedAnimationData = &animationPackResult.value();
  }
  auto animationBuffersResult =
      createAnimationGpuBuffers(gpu, *resolvedAnimationData, debugName);
  if (animationBuffersResult.hasError()) {
    gpu.releaseGeometry(geometryResult.value());
    return Result<std::unique_ptr<Model>, std::string>::makeError(
        animationBuffersResult.error());
  }
  ModelAnimationGpuBuffers animationBuffers =
      std::move(animationBuffersResult.value());
  if (!staticDecodeBytes.empty()) {
    auto decodeBufferResult = createStaticVertexDecodeBuffer(
        gpu, staticDecodeBytes, debugName,
        animationBuffers.vertexDecodeBufferAddress);
    if (decodeBufferResult.hasError()) {
      releaseAnimationGpuBuffers(gpu, animationBuffers);
      gpu.releaseGeometry(geometryResult.value());
      return Result<std::unique_ptr<Model>, std::string>::makeError(
          "Model::createFromPackedVertices: " + decodeBufferResult.error());
    }
    animationBuffers.vertexDecodeBuffer = std::move(decodeBufferResult.value());
  }

  std::pmr::vector<Submesh> ownedSubmeshes(storageMemory);
  ownedSubmeshes.assign(data.submeshes.begin(), data.submeshes.end());
  std::pmr::vector<uint32_t> sourceMaterialToRuntime(
      sourceMaterialCountResult.value(), Model::kInvalidMaterialIndex,
      storageMemory);
  return Result<std::unique_ptr<Model>, std::string>::makeResult(
      std::unique_ptr<Model>(
          new Model(gpu, geometryResult.value(), std::move(ownedSubmeshes),
                    static_cast<uint32_t>(data.vertices.size()),
                    static_cast<uint32_t>(data.indices.size()), bounds,
                    packedVertexFormat, animationBuffers.view,
                    animationBuffers.vertexDecodeBufferAddress,
                    std::move(animationBuffers.vertexDecodeBuffer),
                    std::move(animationBuffers.skinInfluenceBuffer),
                    std::move(animationBuffers.morphMetaBuffer),
                    std::move(animationBuffers.morphDeltaBuffer),
                    std::move(sourceMaterialToRuntime))));
}

Result<std::unique_ptr<Model>, std::string> Model::createFromFile(
    GPUDevice &gpu, std::string_view path, const MeshImportOptions &options,
    std::pmr::memory_resource *mem, std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  std::pmr::memory_resource *const storageMemory =
      std::pmr::get_default_resource();
  std::pmr::memory_resource *const importMemory =
      mem ? mem : std::pmr::get_default_resource();
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
      const PackedVertexFormat cachedFormat =
          packedVertexFormatFromLayoutId(cachedMesh->vertexLayoutId);
      const size_t expectedPackedByteCount =
          static_cast<size_t>(cachedMesh->vertices.count) *
          packedVertexStrideBytes(cachedFormat);
      if (cachedMesh->vertices.data.size() != expectedPackedByteCount) {
        NURI_LOG_WARNING(
            "Model::createFromFile: Cache vertex byte count mismatch for '%s' "
            "(expected=%zu actual=%zu), rebuilding from source",
            cacheKey.cachePath.string().c_str(), expectedPackedByteCount,
            cachedMesh->vertices.data.size());
      } else {
        auto sourceMaterialCountResult =
            computeSourceMaterialCount(std::span<const Submesh>(
                cachedMesh->submeshes.data(), cachedMesh->submeshes.size()));
        if (sourceMaterialCountResult.hasError()) {
          return Result<std::unique_ptr<Model>, std::string>::makeError(
              sourceMaterialCountResult.error());
        }
        const std::span<const std::byte> vertexBytes{
            cachedMesh->vertices.data.data(), cachedMesh->vertices.data.size()};
        const std::span<const std::byte> indexBytes{
            reinterpret_cast<const std::byte *>(cachedMesh->indices.data()),
            cachedMesh->indices.size() * sizeof(uint32_t)};
        auto geometryResult = gpu.allocateGeometry(
            vertexBytes, cachedMesh->vertices.count, indexBytes,
            static_cast<uint32_t>(cachedMesh->indices.size()), debugName);
        if (!geometryResult.hasError()) {
          ModelAnimationPackedData animationPacked =
              reconstructAnimationPackedDataFromCache(*cachedMesh);
          auto animationBuffersResult =
              createAnimationGpuBuffers(gpu, animationPacked, debugName);
          if (animationBuffersResult.hasError()) {
            gpu.releaseGeometry(geometryResult.value());
            return Result<std::unique_ptr<Model>, std::string>::makeError(
                animationBuffersResult.error());
          }
          ModelAnimationGpuBuffers animationBuffers =
              std::move(animationBuffersResult.value());
          if (!cachedMesh->staticVertexDecode.empty()) {
            auto decodeBufferResult = createStaticVertexDecodeBuffer(
                gpu,
                std::span<const std::byte>(
                    cachedMesh->staticVertexDecode.data.data(),
                    cachedMesh->staticVertexDecode.data.size()),
                debugName, animationBuffers.vertexDecodeBufferAddress);
            if (decodeBufferResult.hasError()) {
              releaseAnimationGpuBuffers(gpu, animationBuffers);
              gpu.releaseGeometry(geometryResult.value());
              return Result<std::unique_ptr<Model>, std::string>::makeError(
                  "Model::createFromFile: " + decodeBufferResult.error());
            }
            animationBuffers.vertexDecodeBuffer =
                std::move(decodeBufferResult.value());
          }
          std::pmr::vector<Submesh> ownedSubmeshes(storageMemory);
          ownedSubmeshes.assign(cachedMesh->submeshes.begin(),
                                cachedMesh->submeshes.end());
          std::pmr::vector<uint32_t> sourceMaterialToRuntime(
              sourceMaterialCountResult.value(), Model::kInvalidMaterialIndex,
              storageMemory);
          return Result<std::unique_ptr<Model>, std::string>::makeResult(
              std::unique_ptr<Model>(new Model(
                  gpu, geometryResult.value(), std::move(ownedSubmeshes),
                  cachedMesh->vertices.count,
                  static_cast<uint32_t>(cachedMesh->indices.size()),
                  cachedMesh->bounds, cachedFormat, animationBuffers.view,
                  animationBuffers.vertexDecodeBufferAddress,
                  std::move(animationBuffers.vertexDecodeBuffer),
                  std::move(animationBuffers.skinInfluenceBuffer),
                  std::move(animationBuffers.morphMetaBuffer),
                  std::move(animationBuffers.morphDeltaBuffer),
                  std::move(sourceMaterialToRuntime))));
        }
        NURI_LOG_WARNING(
            "Model::createFromFile: Failed to create model from cache '%s': "
            "%s",
            cacheKey.cachePath.string().c_str(),
            geometryResult.error().c_str());
      }
    }
  } else {
    NURI_LOG_DEBUG("Model::createFromFile: Mesh cache read disabled for '%.*s'",
                   static_cast<int>(path.size()), path.data());
  }

  auto meshDataResult = MeshImporter::loadFromFile(path, options, importMemory);
  if (meshDataResult.hasError()) {
    const std::string pathStr{path};
    NURI_LOG_WARNING("Model::createFromFile: Failed to load mesh '%s': %s",
                     pathStr.c_str(), meshDataResult.error().c_str());
    return Result<std::unique_ptr<Model>, std::string>::makeError(
        meshDataResult.error());
  }

  const MeshData &meshData = meshDataResult.value();
  const bool canWriteMeshCache = !cacheKeyResult.hasError();
  ModelPackedVertexData packedVertices{};
  ModelAnimationPackedData animationPacked{};
  if (canWriteMeshCache) {
    packedVertices = packVerticesForModel(meshData);
    auto animationPackResult = packAnimationData(meshData);
    if (animationPackResult.hasError()) {
      return Result<std::unique_ptr<Model>, std::string>::makeError(
          animationPackResult.error());
    }
    animationPacked = std::move(animationPackResult.value());
  }

  auto modelResult =
      canWriteMeshCache
          ? createFromPackedVertices(
                gpu, meshData,
                std::span<const std::byte>(packedVertices.vertexBytes.data(),
                                           packedVertices.vertexBytes.size()),
                packedVertices.format,
                std::span<const std::byte>(
                    packedVertices.staticDecode.data.data(),
                    packedVertices.staticDecode.data.size()),
                packedVertices.staticDecode.count, &animationPacked, debugName)
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
        cacheKeyResult.value(), options, packedVertices,
        static_cast<uint32_t>(meshData.vertices.size()),
        std::span<const uint32_t>(meshData.indices.data(),
                                  meshData.indices.size()),
        std::span<const Submesh>(meshData.submeshes.data(),
                                 meshData.submeshes.size()),
        modelResult.value()->bounds(), animationPacked);
  }

  NURI_LOG_DEBUG("Model::createFromFile: Created model from file '%.*s'",
                 static_cast<int>(path.size()), path.data());
  return modelResult;
}

Result<ModelAsyncLoad, std::string>
Model::createFromFileAsync(std::string_view path,
                           const MeshImportOptions &options) {
  if (path.empty()) {
    return Result<ModelAsyncLoad, std::string>::makeError(
        "Model::createFromFileAsync: path is empty");
  }

  std::filesystem::path sourcePath{std::string(path)};
  std::error_code ec;
  std::filesystem::path normalizedPath =
      std::filesystem::weakly_canonical(sourcePath, ec);
  if (ec) {
    normalizedPath = sourcePath.lexically_normal();
  }
  const std::string sourcePathString = normalizedPath.string();

  std::error_code existsEc;
  if (!std::filesystem::exists(sourcePathString, existsEc)) {
    return Result<ModelAsyncLoad, std::string>::makeError(
        "Model::createFromFileAsync: source path does not exist: " +
        sourcePathString + (existsEc ? (" (" + existsEc.message() + ")") : ""));
  }
  std::error_code isRegEc;
  if (!std::filesystem::is_regular_file(sourcePathString, isRegEc)) {
    return Result<ModelAsyncLoad, std::string>::makeError(
        "Model::createFromFileAsync: source path is not a regular file: " +
        sourcePathString + (isRegEc ? (" (" + isRegEc.message() + ")") : ""));
  }

  std::future<Result<bool, std::string>> warmupFuture;
  try {
    warmupFuture = std::async(
        std::launch::async,
        [sourcePathString, options]() -> Result<bool, std::string> {
          std::pmr::unsynchronized_pool_resource importMemory;
          return Model::warmFileCache(sourcePathString, options, &importMemory);
        });
  } catch (const std::exception &e) {
    return Result<ModelAsyncLoad, std::string>::makeError(
        std::string("Model::createFromFileAsync: failed to launch warmup: ") +
        e.what());
  } catch (...) {
    return Result<ModelAsyncLoad, std::string>::makeError(
        "Model::createFromFileAsync: failed to launch warmup");
  }

  return Result<ModelAsyncLoad, std::string>::makeResult(
      ModelAsyncLoad(sourcePathString, options, std::move(warmupFuture)));
}

Result<bool, std::string> Model::warmFileCache(std::string_view path,
                                               const MeshImportOptions &options,
                                               std::pmr::memory_resource *mem) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  const std::filesystem::path sourcePath{std::string(path)};
  auto cacheKeyResult = buildMeshCacheKey(sourcePath, options);
  if (cacheKeyResult.hasError()) {
    return Result<bool, std::string>::makeError(
        "Model::warmFileCache: Failed to build mesh cache key: " +
        cacheKeyResult.error());
  }
  const MeshCacheKey &cacheKey = cacheKeyResult.value();

  if (auto cachedMesh = tryLoadMeshCache(path, cacheKey, options);
      cachedMesh.has_value()) {
    return Result<bool, std::string>::makeResult(true);
  }

  auto meshDataResult = MeshImporter::loadFromFile(path, options, mem);
  if (meshDataResult.hasError()) {
    return Result<bool, std::string>::makeError(
        "Model::warmFileCache: Failed to import mesh: " +
        meshDataResult.error());
  }

  const MeshData &meshData = meshDataResult.value();
  if (meshData.vertices.empty() || meshData.indices.empty()) {
    return Result<bool, std::string>::makeError(
        "Model::warmFileCache: Imported mesh has no vertices or indices");
  }
  auto topologyValidation =
      validateMeshTopology(std::span<const uint32_t>(meshData.indices.data(),
                                                     meshData.indices.size()),
                           static_cast<uint32_t>(meshData.vertices.size()),
                           std::span<const Submesh>(meshData.submeshes.data(),
                                                    meshData.submeshes.size()),
                           "Model::warmFileCache");
  if (topologyValidation.hasError()) {
    return Result<bool, std::string>::makeError(topologyValidation.error());
  }

  const ModelPackedVertexData packedVertices = packVerticesForModel(meshData);
  if (packedVertices.vertexBytes.empty()) {
    return Result<bool, std::string>::makeError(
        "Model::warmFileCache: Packed vertex buffer is empty");
  }
  auto animationPackResult = packAnimationData(meshData);
  if (animationPackResult.hasError()) {
    return Result<bool, std::string>::makeError(
        "Model::warmFileCache: Failed to pack animation payload: " +
        animationPackResult.error());
  }
  const ModelAnimationPackedData animationPacked =
      std::move(animationPackResult.value());

  const MeshSourceFingerprint fingerprint =
      queryMeshSourceFingerprint(cacheKey.normalizedSourcePath);
  MeshBinarySerializeInput input{};
  input.sourcePathHash = cacheKey.sourcePathHash;
  input.importOptionsHash = hashMeshImportOptions(options);
  input.sourceSizeBytes = fingerprint.exists ? fingerprint.sizeBytes : 0u;
  input.sourceMtimeNs = fingerprint.exists ? fingerprint.mtimeNs : 0;
  input.bounds = computeModelBounds(meshData.vertices);
  input.vertexLayoutId =
      meshBinaryLayoutIdForVertexFormat(packedVertices.format);
  input.vertices = {
      std::span<const std::byte>(packedVertices.vertexBytes.data(),
                                 packedVertices.vertexBytes.size()),
      static_cast<uint32_t>(meshData.vertices.size()),
      packedVertexStrideBytes(packedVertices.format),
  };
  input.staticVertexDecode = {
      std::span<const std::byte>(packedVertices.staticDecode.data.data(),
                                 packedVertices.staticDecode.data.size()),
      packedVertices.staticDecode.count,
      packedVertices.staticDecode.strideBytes,
  };
  input.indices = std::span<const uint32_t>(meshData.indices.data(),
                                            meshData.indices.size());
  input.submeshes = std::span<const Submesh>(meshData.submeshes.data(),
                                             meshData.submeshes.size());
  input.skinInfluences = {
      std::span<const std::byte>(animationPacked.skinInfluences.data.data(),
                                 animationPacked.skinInfluences.data.size()),
      animationPacked.skinInfluences.count,
      animationPacked.skinInfluences.strideBytes,
  };
  input.morphMeta = {
      std::span<const std::byte>(animationPacked.morphMeta.data.data(),
                                 animationPacked.morphMeta.data.size()),
      animationPacked.morphMeta.count,
      animationPacked.morphMeta.strideBytes,
  };
  input.morphDeltas = {
      std::span<const std::byte>(animationPacked.morphDeltas.data.data(),
                                 animationPacked.morphDeltas.data.size()),
      animationPacked.morphDeltas.count,
      animationPacked.morphDeltas.strideBytes,
  };

  auto serializeResult = meshBinarySerialize(input);
  if (serializeResult.hasError()) {
    return Result<bool, std::string>::makeError(
        "Model::warmFileCache: Failed to serialize mesh cache: " +
        serializeResult.error());
  }

  auto writeResult =
      writeBinaryFileAtomic(cacheKey.cachePath, serializeResult.value());
  if (writeResult.hasError()) {
    return Result<bool, std::string>::makeError(
        "Model::warmFileCache: Failed to write mesh cache '" +
        cacheKey.cachePath.string() + "': " + writeResult.error());
  }

  NURI_LOG_INFO("Model::warmFileCache: Built mesh cache '%s'",
                cacheKey.cachePath.string().c_str());
  return Result<bool, std::string>::makeResult(false);
}

} // namespace nuri
