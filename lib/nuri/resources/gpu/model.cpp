#include "nuri/resources/gpu/model.h"
#include "nuri/core/log.h"
#include "nuri/core/pmr_scratch.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/pch.h"
#include "nuri/resources/mesh_importer.h"
#include "nuri/resources/storage/mesh/mesh_binary_format.h"
#include "nuri/resources/storage/mesh/mesh_binary_serializer.h"
#include "nuri/resources/storage/mesh/mesh_cache_utils.h"
#include "nuri/resources/storage/mesh/mesh_cache_writer.h"
#include "nuri/utils/env_utils.h"
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
struct ModelMeshletGpuBuffers {
  Model::ModelMeshletGpuView view{};
  std::unique_ptr<Buffer> meshletBuffer;
  std::unique_ptr<Buffer> meshletVertexIndexBuffer;
  std::unique_ptr<Buffer> meshletPrimitiveIndexBuffer;
  std::unique_ptr<Buffer> meshletLodRangeBuffer;
};
struct PreparedMeshletBufferData {
  std::vector<std::byte> meshletDescriptors;
  std::vector<std::byte> vertexIndices;
  std::vector<std::byte> primitiveIndices;
  std::vector<std::byte> lodRanges;
  uint32_t meshletCount = 0u;
  uint32_t vertexIndexCount = 0u;
  uint32_t primitiveIndexCount = 0u;
  uint32_t lodRangeCount = 0u;
};
struct MeshletDescriptorGpu {
  glm::uvec4 offsetsCounts{0u};
  glm::vec4 boundsSphere{0.0f};
  glm::vec4 coneApex{0.0f};
  glm::vec4 coneAxisCutoff{0.0f};
};
static_assert(sizeof(MeshletDescriptorGpu) == 64);
static_assert(sizeof(Model::SubmeshMeshletLodRangeGpu) == 64);
constexpr uint32_t kMeshletShaderMaxVertices = 64u;
constexpr uint32_t kMeshletShaderMaxPrimitives = 124u;
std::pmr::vector<glm::vec4>
copyMeshletBoundsSpheres(std::span<const MeshletDescriptor> meshlets,
                         std::pmr::memory_resource *memory) {
  std::pmr::vector<glm::vec4> bounds(memory);
  bounds.reserve(meshlets.size());
  for (const MeshletDescriptor &meshlet : meshlets) {
    bounds.push_back(meshlet.boundsSphere);
  }
  return bounds;
}
void destroyBuffer(GPUDevice &, std::unique_ptr<Buffer> &buffer) {
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
void releaseMeshletGpuBuffers(GPUDevice &gpu, ModelMeshletGpuBuffers &buffers) {
  destroyBuffer(gpu, buffers.meshletBuffer);
  destroyBuffer(gpu, buffers.meshletVertexIndexBuffer);
  destroyBuffer(gpu, buffers.meshletPrimitiveIndexBuffer);
  destroyBuffer(gpu, buffers.meshletLodRangeBuffer);
  buffers.view = {};
}
template <typename T>
std::span<const std::byte> podBytes(std::span<const T> values) {
  if (values.empty()) {
    return {};
  }
  return {reinterpret_cast<const std::byte *>(values.data()),
          values.size() * sizeof(T)};
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
      .immutable = true,
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
template <typename T>
std::vector<std::byte> copyPodBytes(std::span<const T> values) {
  std::vector<std::byte> bytes(values.size_bytes());
  if (!bytes.empty()) {
    std::memcpy(bytes.data(), values.data(), bytes.size());
  }
  return bytes;
}
Result<PreparedMeshletBufferData, std::string>
prepareMeshletBufferData(std::span<const MeshletDescriptor> meshlets,
                         std::span<const uint32_t> meshletVertexIndices,
                         std::span<const uint8_t> meshletPrimitiveIndices,
                         std::span<const Submesh> submeshes) {
  PreparedMeshletBufferData prepared{};
  const bool hasMeshletPayload = !meshlets.empty() ||
                                 !meshletVertexIndices.empty() ||
                                 !meshletPrimitiveIndices.empty();
  if (!hasMeshletPayload) {
    return Result<PreparedMeshletBufferData, std::string>::makeResult(
        std::move(prepared));
  }
  if (meshlets.empty() || meshletVertexIndices.empty() ||
      meshletPrimitiveIndices.empty()) {
    return Result<PreparedMeshletBufferData, std::string>::makeError(
        "Model::create: meshlet descriptor, vertex index, and primitive "
        "index streams must be present together");
  }
  std::vector<MeshletDescriptorGpu> descriptors;
  descriptors.reserve(meshlets.size());
  for (const MeshletDescriptor &meshlet : meshlets) {
    if (meshlet.vertexCount > kMeshletShaderMaxVertices) {
      return Result<PreparedMeshletBufferData, std::string>::makeError(
          "Model::create: meshlet vertex count exceeds shader output limit");
    }
    if (meshlet.primitiveCount > kMeshletShaderMaxPrimitives) {
      return Result<PreparedMeshletBufferData, std::string>::makeError(
          "Model::create: meshlet primitive count exceeds shader output "
          "limit");
    }
    const uint64_t vertexEnd =
        static_cast<uint64_t>(meshlet.vertexOffset) + meshlet.vertexCount;
    if (vertexEnd > meshletVertexIndices.size()) {
      return Result<PreparedMeshletBufferData, std::string>::makeError(
          "Model::create: meshlet vertex index range out of bounds");
    }
    const uint64_t primitiveIndexCount =
        static_cast<uint64_t>(meshlet.primitiveCount) * 3u;
    const uint64_t primitiveEnd =
        static_cast<uint64_t>(meshlet.primitiveOffset) + primitiveIndexCount;
    if (primitiveEnd > meshletPrimitiveIndices.size()) {
      return Result<PreparedMeshletBufferData, std::string>::makeError(
          "Model::create: meshlet primitive index range out of bounds");
    }
    if ((meshlet.primitiveOffset % 3u) != 0u) {
      return Result<PreparedMeshletBufferData, std::string>::makeError(
          "Model::create: meshlet primitive offset is not triangle-aligned");
    }
    for (uint64_t index = meshlet.primitiveOffset; index < primitiveEnd;
         ++index) {
      if (meshletPrimitiveIndices[static_cast<size_t>(index)] >=
          meshlet.vertexCount) {
        return Result<PreparedMeshletBufferData, std::string>::makeError(
            "Model::create: meshlet primitive local index out of bounds");
      }
    }
    descriptors.push_back(MeshletDescriptorGpu{
        .offsetsCounts =
            glm::uvec4(meshlet.vertexOffset, meshlet.primitiveOffset / 3u,
                       meshlet.vertexCount, meshlet.primitiveCount),
        .boundsSphere = meshlet.boundsSphere,
        .coneApex = meshlet.coneApex,
        .coneAxisCutoff = meshlet.coneAxisCutoff,
    });
  }
  if ((meshletPrimitiveIndices.size() % 3u) != 0u) {
    return Result<PreparedMeshletBufferData, std::string>::makeError(
        "Model::create: meshlet primitive index stream is not "
        "triangle-aligned");
  }
  std::vector<Model::SubmeshMeshletLodRangeGpu> lodRanges;
  lodRanges.reserve(submeshes.size());
  for (const Submesh &submesh : submeshes) {
    Model::SubmeshMeshletLodRangeGpu range{};
    range.lodCount = submesh.lodCount;
    for (uint32_t lodIndex = 0u; lodIndex < submesh.lodCount; ++lodIndex) {
      const SubmeshLod &lod = submesh.lods[lodIndex];
      const uint64_t meshletEnd =
          static_cast<uint64_t>(lod.meshletOffset) + lod.meshletCount;
      if (meshletEnd > meshlets.size()) {
        return Result<PreparedMeshletBufferData, std::string>::makeError(
            "Model::create: submesh meshlet range out of bounds");
      }
      for (uint64_t meshletIndex = lod.meshletOffset; meshletIndex < meshletEnd;
           ++meshletIndex) {
        const MeshletDescriptor &meshlet =
            meshlets[static_cast<size_t>(meshletIndex)];
        const uint64_t vertexEnd =
            static_cast<uint64_t>(meshlet.vertexOffset) + meshlet.vertexCount;
        for (uint64_t vertexIndex = meshlet.vertexOffset;
             vertexIndex < vertexEnd; ++vertexIndex) {
          if (meshletVertexIndices[static_cast<size_t>(vertexIndex)] >=
              submesh.vertexCount) {
            return Result<PreparedMeshletBufferData, std::string>::makeError(
                "Model::create: meshlet vertex local index out of submesh "
                "bounds");
          }
        }
      }
      range.meshletOffset[lodIndex] = lod.meshletOffset;
      range.meshletCount[lodIndex] = lod.meshletCount;
      range.error[lodIndex] = lod.error;
    }
    lodRanges.push_back(range);
  }
  std::vector<uint32_t> primitiveIndices;
  primitiveIndices.reserve(meshletPrimitiveIndices.size() / 3u);
  for (size_t index = 0u; index < meshletPrimitiveIndices.size(); index += 3u) {
    const uint32_t i0 = static_cast<uint32_t>(meshletPrimitiveIndices[index]);
    const uint32_t i1 =
        static_cast<uint32_t>(meshletPrimitiveIndices[index + 1u]);
    const uint32_t i2 =
        static_cast<uint32_t>(meshletPrimitiveIndices[index + 2u]);
    primitiveIndices.push_back(i0 | (i1 << 8u) | (i2 << 16u));
  }
  prepared.meshletDescriptors =
      copyPodBytes(std::span<const MeshletDescriptorGpu>(descriptors.data(),
                                                         descriptors.size()));
  prepared.vertexIndices = copyPodBytes(meshletVertexIndices);
  prepared.primitiveIndices = copyPodBytes(std::span<const uint32_t>(
      primitiveIndices.data(), primitiveIndices.size()));
  prepared.lodRanges =
      copyPodBytes(std::span<const Model::SubmeshMeshletLodRangeGpu>(
          lodRanges.data(), lodRanges.size()));
  prepared.meshletCount = static_cast<uint32_t>(meshlets.size());
  prepared.vertexIndexCount =
      static_cast<uint32_t>(meshletVertexIndices.size());
  prepared.primitiveIndexCount = static_cast<uint32_t>(primitiveIndices.size());
  prepared.lodRangeCount = static_cast<uint32_t>(lodRanges.size());
  return Result<PreparedMeshletBufferData, std::string>::makeResult(
      std::move(prepared));
}
Result<ModelMeshletGpuBuffers, std::string> createMeshletGpuBuffers(
    GPUDevice &gpu, std::span<const MeshletDescriptor> meshlets,
    std::span<const uint32_t> meshletVertexIndices,
    std::span<const uint8_t> meshletPrimitiveIndices,
    std::span<const Submesh> submeshes, std::string_view debugName) {
  ModelMeshletGpuBuffers buffers{};
  const bool hasMeshletPayload = !meshlets.empty() ||
                                 !meshletVertexIndices.empty() ||
                                 !meshletPrimitiveIndices.empty();
  if (!hasMeshletPayload) {
    return Result<ModelMeshletGpuBuffers, std::string>::makeResult(
        std::move(buffers));
  }
  if (meshlets.empty() || meshletVertexIndices.empty() ||
      meshletPrimitiveIndices.empty()) {
    return Result<ModelMeshletGpuBuffers, std::string>::makeError(
        "Model::create: meshlet descriptor, vertex index, and primitive index "
        "streams must be present together");
  }
  std::vector<MeshletDescriptorGpu> meshletDescriptorsGpu;
  meshletDescriptorsGpu.reserve(meshlets.size());
  for (const MeshletDescriptor &meshlet : meshlets) {
    if (meshlet.vertexCount > kMeshletShaderMaxVertices) {
      return Result<ModelMeshletGpuBuffers, std::string>::makeError(
          "Model::create: meshlet vertex count exceeds shader output limit");
    }
    if (meshlet.primitiveCount > kMeshletShaderMaxPrimitives) {
      return Result<ModelMeshletGpuBuffers, std::string>::makeError(
          "Model::create: meshlet primitive count exceeds shader output "
          "limit");
    }
    const uint64_t vertexEnd =
        static_cast<uint64_t>(meshlet.vertexOffset) + meshlet.vertexCount;
    if (vertexEnd > meshletVertexIndices.size()) {
      return Result<ModelMeshletGpuBuffers, std::string>::makeError(
          "Model::create: meshlet vertex index range out of bounds");
    }
    const uint64_t primitiveIndexCount =
        static_cast<uint64_t>(meshlet.primitiveCount) * 3u;
    const uint64_t primitiveEnd =
        static_cast<uint64_t>(meshlet.primitiveOffset) + primitiveIndexCount;
    if (primitiveEnd > meshletPrimitiveIndices.size()) {
      return Result<ModelMeshletGpuBuffers, std::string>::makeError(
          "Model::create: meshlet primitive index range out of bounds");
    }
    if ((meshlet.primitiveOffset % 3u) != 0u) {
      return Result<ModelMeshletGpuBuffers, std::string>::makeError(
          "Model::create: meshlet primitive offset is not triangle-aligned");
    }
    for (uint64_t index = meshlet.primitiveOffset; index < primitiveEnd;
         ++index) {
      if (meshletPrimitiveIndices[static_cast<size_t>(index)] >=
          meshlet.vertexCount) {
        return Result<ModelMeshletGpuBuffers, std::string>::makeError(
            "Model::create: meshlet primitive local index out of bounds");
      }
    }
    meshletDescriptorsGpu.push_back(MeshletDescriptorGpu{
        .offsetsCounts =
            glm::uvec4(meshlet.vertexOffset, meshlet.primitiveOffset / 3u,
                       meshlet.vertexCount, meshlet.primitiveCount),
        .boundsSphere = meshlet.boundsSphere,
        .coneApex = meshlet.coneApex,
        .coneAxisCutoff = meshlet.coneAxisCutoff,
    });
  }
  if ((meshletPrimitiveIndices.size() % 3u) != 0u) {
    return Result<ModelMeshletGpuBuffers, std::string>::makeError(
        "Model::create: meshlet primitive index stream is not "
        "triangle-aligned");
  }
  std::vector<Model::SubmeshMeshletLodRangeGpu> lodRanges;
  lodRanges.reserve(submeshes.size());
  for (const Submesh &submesh : submeshes) {
    Model::SubmeshMeshletLodRangeGpu range{};
    range.lodCount = submesh.lodCount;
    for (uint32_t lodIndex = 0; lodIndex < submesh.lodCount; ++lodIndex) {
      const SubmeshLod &lod = submesh.lods[lodIndex];
      const uint64_t meshletEnd =
          static_cast<uint64_t>(lod.meshletOffset) + lod.meshletCount;
      if (meshletEnd > meshlets.size()) {
        return Result<ModelMeshletGpuBuffers, std::string>::makeError(
            "Model::create: submesh meshlet range out of bounds");
      }
      for (uint64_t meshletIndex = lod.meshletOffset; meshletIndex < meshletEnd;
           ++meshletIndex) {
        const MeshletDescriptor &meshlet =
            meshlets[static_cast<size_t>(meshletIndex)];
        const uint64_t vertexEnd =
            static_cast<uint64_t>(meshlet.vertexOffset) + meshlet.vertexCount;
        for (uint64_t vertexIndex = meshlet.vertexOffset;
             vertexIndex < vertexEnd; ++vertexIndex) {
          if (meshletVertexIndices[static_cast<size_t>(vertexIndex)] >=
              submesh.vertexCount) {
            return Result<ModelMeshletGpuBuffers, std::string>::makeError(
                "Model::create: meshlet vertex local index out of submesh "
                "bounds");
          }
        }
      }
      range.meshletOffset[lodIndex] = lod.meshletOffset;
      range.meshletCount[lodIndex] = lod.meshletCount;
      range.error[lodIndex] = lod.error;
    }
    lodRanges.push_back(range);
  }
  auto meshletBufferResult = createStorageBuffer(
      gpu,
      podBytes(std::span<const MeshletDescriptorGpu>(
          meshletDescriptorsGpu.data(), meshletDescriptorsGpu.size())),
      std::string(debugName) + "_meshlets");
  if (meshletBufferResult.hasError()) {
    return Result<ModelMeshletGpuBuffers, std::string>::makeError(
        meshletBufferResult.error());
  }
  buffers.meshletBuffer = std::move(meshletBufferResult.value());
  buffers.view.meshletBuffer = buffers.meshletBuffer->handle();
  buffers.view.meshletCount = static_cast<uint32_t>(meshlets.size());
  auto vertexIndexBufferResult = createStorageBuffer(
      gpu,
      podBytes(std::span<const uint32_t>(meshletVertexIndices.data(),
                                         meshletVertexIndices.size())),
      std::string(debugName) + "_meshlet_vertices");
  if (vertexIndexBufferResult.hasError()) {
    releaseMeshletGpuBuffers(gpu, buffers);
    return Result<ModelMeshletGpuBuffers, std::string>::makeError(
        vertexIndexBufferResult.error());
  }
  buffers.meshletVertexIndexBuffer = std::move(vertexIndexBufferResult.value());
  buffers.view.meshletVertexIndexBuffer =
      buffers.meshletVertexIndexBuffer->handle();
  buffers.view.meshletVertexIndexCount =
      static_cast<uint32_t>(meshletVertexIndices.size());
  std::vector<uint32_t> meshletPrimitiveIndicesGpu;
  meshletPrimitiveIndicesGpu.reserve(meshletPrimitiveIndices.size() / 3u);
  for (size_t i = 0u; i < meshletPrimitiveIndices.size(); i += 3u) {
    const uint32_t i0 = static_cast<uint32_t>(meshletPrimitiveIndices[i]);
    const uint32_t i1 = static_cast<uint32_t>(meshletPrimitiveIndices[i + 1u]);
    const uint32_t i2 = static_cast<uint32_t>(meshletPrimitiveIndices[i + 2u]);
    meshletPrimitiveIndicesGpu.push_back(i0 | (i1 << 8u) | (i2 << 16u));
  }
  auto primitiveIndexBufferResult = createStorageBuffer(
      gpu,
      podBytes(std::span<const uint32_t>(meshletPrimitiveIndicesGpu.data(),
                                         meshletPrimitiveIndicesGpu.size())),
      std::string(debugName) + "_meshlet_primitives");
  if (primitiveIndexBufferResult.hasError()) {
    releaseMeshletGpuBuffers(gpu, buffers);
    return Result<ModelMeshletGpuBuffers, std::string>::makeError(
        primitiveIndexBufferResult.error());
  }
  buffers.meshletPrimitiveIndexBuffer =
      std::move(primitiveIndexBufferResult.value());
  buffers.view.meshletPrimitiveIndexBuffer =
      buffers.meshletPrimitiveIndexBuffer->handle();
  buffers.view.meshletPrimitiveIndexCount =
      static_cast<uint32_t>(meshletPrimitiveIndicesGpu.size());
  auto lodRangeBufferResult = createStorageBuffer(
      gpu,
      podBytes(std::span<const Model::SubmeshMeshletLodRangeGpu>(
          lodRanges.data(), lodRanges.size())),
      std::string(debugName) + "_meshlet_lod_ranges");
  if (lodRangeBufferResult.hasError()) {
    releaseMeshletGpuBuffers(gpu, buffers);
    return Result<ModelMeshletGpuBuffers, std::string>::makeError(
        lodRangeBufferResult.error());
  }
  buffers.meshletLodRangeBuffer = std::move(lodRangeBufferResult.value());
  buffers.view.lodRangeBuffer = buffers.meshletLodRangeBuffer->handle();
  buffers.view.lodRangeCount = static_cast<uint32_t>(lodRanges.size());
  return Result<ModelMeshletGpuBuffers, std::string>::makeResult(
      std::move(buffers));
}
Result<ModelAnimationGpuBuffers, std::string>
createAnimationGpuBuffers(GPUDevice &gpu,
                          const ModelAnimationPackedData &packedData,
                          std::string_view debugName) {
  ModelAnimationGpuBuffers buffers{};
  using Layout = BufferLayout<std::vector<std::byte>>;
  const std::array uploads{
      std::tuple<const Layout *, std::unique_ptr<Buffer> *, std::string_view>{
          &packedData.skinInfluences, &buffers.skinInfluenceBuffer,
          "_skin_influences"},
      std::tuple<const Layout *, std::unique_ptr<Buffer> *, std::string_view>{
          &packedData.morphMeta, &buffers.morphMetaBuffer, "_morph_meta"},
      std::tuple<const Layout *, std::unique_ptr<Buffer> *, std::string_view>{
          &packedData.morphDeltas, &buffers.morphDeltaBuffer, "_morph_deltas"},
  };
  for (const auto &[layout, destination, suffix] : uploads) {
    auto created = createStorageBuffer(gpu, layout->data,
                                       std::string(debugName) + suffix.data());
    if (created.hasError()) {
      releaseAnimationGpuBuffers(gpu, buffers);
      return Result<ModelAnimationGpuBuffers, std::string>::makeError(
          created.error());
    }
    *destination = std::move(created.value());
  }
  if (buffers.skinInfluenceBuffer) {
    buffers.view.skinInfluenceBuffer = buffers.skinInfluenceBuffer->handle();
    buffers.view.skinInfluenceCount = packedData.skinInfluences.count;
  }
  if (buffers.morphMetaBuffer) {
    buffers.view.morphMetaBuffer = buffers.morphMetaBuffer->handle();
    MeshBinaryMorphMetaRecord meta{};
    std::memcpy(&meta, packedData.morphMeta.data.data(), sizeof(meta));
    buffers.view.morphTargetCount = meta.morphTargetCount;
    buffers.view.morphVertexCount = meta.vertexCount;
  }
  if (buffers.morphDeltaBuffer) {
    buffers.view.morphDeltaBuffer = buffers.morphDeltaBuffer->handle();
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
                     std::span<const Submesh> submeshes) {
  if (vertexCount == 0u || indices.empty() || submeshes.empty()) {
    return Result<bool, std::string>::makeError("invalid empty mesh topology");
  }
  if (*std::max_element(indices.begin(), indices.end()) >= vertexCount) {
    return Result<bool, std::string>::makeError("mesh index out of range");
  }
  for (const Submesh &submesh : submeshes) {
    if (submesh.lodCount == 0 || submesh.lodCount > Submesh::kMaxLodCount) {
      return Result<bool, std::string>::makeError(
          "invalid mesh submesh LOD count");
    }
    for (uint32_t lodIndex = 0; lodIndex < submesh.lodCount; ++lodIndex) {
      const SubmeshLod &lod = submesh.lods[lodIndex];
      if (static_cast<uint64_t>(lod.indexOffset) + lod.indexCount >
          indices.size()) {
        return Result<bool, std::string>::makeError(
            "mesh submesh index range out of bounds");
      }
    }
  }
  return Result<bool, std::string>::makeResult(true);
}
bool isMeshCacheReadEnabled() {
  return readEnvBoolOverride("NURI_MESH_CACHE_READ").value_or(true);
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
    return kMeshBinaryAnimatedVertexStrideBytes;
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
Result<std::vector<std::byte>, std::string> serializeMeshCache(
    const MeshCacheKey &cacheKey, const MeshImportOptions &options,
    const ModelPackedVertexData &packedVertexData, const MeshData &mesh,
    const BoundingBox &bounds, const ModelAnimationPackedData &animationData) {
  const SourceFingerprint fingerprint =
      querySourceFingerprint(cacheKey.normalizedSourcePath);
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
      static_cast<uint32_t>(mesh.vertices.size()),
      packedVertexStrideBytes(packedVertexData.format)};
  input.staticVertexDecode = {
      std::span<const std::byte>(packedVertexData.staticDecode.data.data(),
                                 packedVertexData.staticDecode.data.size()),
      packedVertexData.staticDecode.count,
      packedVertexData.staticDecode.strideBytes,
  };
  input.indices = mesh.indices;
  input.submeshes = mesh.submeshes;
  input.meshlets = mesh.meshlets;
  input.meshletVertexIndices = mesh.meshletVertexIndices;
  input.meshletPrimitiveIndices = mesh.meshletPrimitiveIndices;
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
  return meshBinarySerialize(input);
}
void maybeQueueMeshCacheWrite(const MeshCacheKey &cacheKey,
                              const MeshImportOptions &options,
                              const ModelPackedVertexData &packedVertexData,
                              const MeshData &mesh, const BoundingBox &bounds,
                              const ModelAnimationPackedData &animationData) {
  auto serialized = serializeMeshCache(cacheKey, options, packedVertexData,
                                       mesh, bounds, animationData);
  if (serialized.hasError()) {
    NURI_LOG_WARNING(
        "Model::createFromFile: Failed to serialize mesh cache '%s': %s",
        cacheKey.cachePath.string().c_str(), serialized.error().c_str());
    return;
  }
  MeshCacheWriterService::instance().enqueue(cacheKey.cachePath,
                                             std::move(serialized.value()));
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
  const SourceFingerprint sourceFingerprint =
      querySourceFingerprint(cacheKey.normalizedSourcePath);
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
                               decodedMesh.submeshes.size()));
  if (topologyValidation.hasError()) {
    NURI_LOG_WARNING("Model::createFromFile: Rejected mesh cache '%s': %s",
                     cacheKey.cachePath.string().c_str(),
                     topologyValidation.error().c_str());
    return std::nullopt;
  }
  if (options.generateMeshlets && decodedMesh.meshlets.empty()) {
    NURI_LOG_DEBUG(
        "Model::createFromFile: Mesh cache '%s' has no meshlet payload, "
        "rebuilding from source",
        cacheKey.cachePath.string().c_str());
    return std::nullopt;
  }
  NURI_LOG_DEBUG("Model::createFromFile: Loaded mesh cache for '%.*s'",
                 static_cast<int>(sourcePath.size()), sourcePath.data());
  return decodedMesh;
}
} // namespace

struct Model::PackedSource {
  uint32_t vertexCount = 0u;
  std::span<const std::byte> vertices{};
  std::span<const uint32_t> indices{};
  std::span<const Submesh> submeshes{};
  std::span<const MeshletDescriptor> meshlets{};
  std::span<const uint32_t> meshletVertexIndices{};
  std::span<const uint8_t> meshletPrimitiveIndices{};
  BoundingBox bounds{};
  PackedVertexFormat vertexFormat = PackedVertexFormat::StaticQuantized20;
  std::span<const std::byte> staticDecode{};
  const ModelAnimationPackedData *animation = nullptr;
};

struct PreparedGpuModelData::Impl {
  enum Slot : size_t {
    Vertex,
    Index,
    VertexDecode,
    SkinInfluences,
    MorphMeta,
    MorphDeltas,
    MeshletDescriptors,
    MeshletVertexIndices,
    MeshletPrimitiveIndices,
    MeshletLodRanges,
    SlotCount,
  };
  struct BufferSlot {
    std::unique_ptr<PreparedGpuBuffer> prepared{};
    size_t size = 0u;
    BufferUsage usage = BufferUsage::None;
    std::string debugName{};
  };
  explicit Impl(PreparedModelData sourceData) : source(std::move(sourceData)) {}
  PreparedModelData source;
  PreparedMeshletBufferData meshlets{};
  std::array<BufferSlot, SlotCount> buffers{};
  BoundingBox bounds{};
  uint32_t sourceMaterialCount = 0u;
  uint32_t morphTargetCount = 0u;
  uint32_t morphVertexCount = 0u;
};

PreparedGpuModelData::~PreparedGpuModelData() = default;

PreparedGpuModelData::PreparedGpuModelData(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

Model::~Model() {
  if (gpu_ != nullptr) {
    destroyBuffer(*gpu_, vertexDecodeBuffer_);
    destroyBuffer(*gpu_, skinInfluenceBuffer_);
    destroyBuffer(*gpu_, morphMetaBuffer_);
    destroyBuffer(*gpu_, morphDeltaBuffer_);
    destroyBuffer(*gpu_, meshletBuffer_);
    destroyBuffer(*gpu_, meshletVertexIndexBuffer_);
    destroyBuffer(*gpu_, meshletPrimitiveIndexBuffer_);
    destroyBuffer(*gpu_, meshletLodRangeBuffer_);
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
  auto animation = packAnimationData(data);
  if (animation.hasError()) {
    return Result<std::unique_ptr<Model>, std::string>::makeError(
        animation.error());
  }
  return createPacked(
      gpu,
      PackedSource{
          .vertexCount = static_cast<uint32_t>(data.vertices.size()),
          .vertices = packedVertices.vertexBytes,
          .indices = data.indices,
          .submeshes = data.submeshes,
          .meshlets = data.meshlets,
          .meshletVertexIndices = data.meshletVertexIndices,
          .meshletPrimitiveIndices = data.meshletPrimitiveIndices,
          .bounds = computeModelBounds(data.vertices),
          .vertexFormat = packedVertices.format,
          .staticDecode = packedVertices.staticDecode.data,
          .animation = &animation.value(),
      },
      debugName);
}

Result<PreparedModelData, std::string> Model::prepare(MeshData data) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (data.vertices.empty() || data.indices.empty()) {
    return Result<PreparedModelData, std::string>::makeError(
        "Model::prepare: mesh has no vertices or indices");
  }
  ModelPackedVertexData packedVertices = packVerticesForModel(data);
  if (packedVertices.vertexBytes.empty()) {
    return Result<PreparedModelData, std::string>::makeError(
        "Model::prepare: packed vertex payload is empty");
  }
  auto animationResult = packAnimationData(data);
  if (animationResult.hasError()) {
    return Result<PreparedModelData, std::string>::makeError(
        animationResult.error());
  }
  ModelAnimationPackedData animation = std::move(animationResult.value());
  PreparedModelData prepared{};
  prepared.mesh = std::move(data);
  prepared.packedVertexBytes = std::move(packedVertices.vertexBytes);
  prepared.packedVertexFormat = packedVertices.format;
  prepared.staticDecode = PreparedModelBufferData{
      .bytes = std::move(packedVertices.staticDecode.data),
      .count = packedVertices.staticDecode.count,
      .stride = packedVertices.staticDecode.strideBytes,
  };
  prepared.skinInfluences = PreparedModelBufferData{
      .bytes = std::move(animation.skinInfluences.data),
      .count = animation.skinInfluences.count,
      .stride = animation.skinInfluences.strideBytes,
  };
  prepared.morphMeta = PreparedModelBufferData{
      .bytes = std::move(animation.morphMeta.data),
      .count = animation.morphMeta.count,
      .stride = animation.morphMeta.strideBytes,
  };
  prepared.morphDeltas = PreparedModelBufferData{
      .bytes = std::move(animation.morphDeltas.data),
      .count = animation.morphDeltas.count,
      .stride = animation.morphDeltas.strideBytes,
  };
  return Result<PreparedModelData, std::string>::makeResult(
      std::move(prepared));
}

Result<PreparedModelData, std::string>
Model::prepareFromFile(std::string_view path, const MeshImportOptions &options,
                       std::pmr::memory_resource *memory) {
  auto meshResult = MeshImporter::loadFromFile(path, options, memory);
  if (meshResult.hasError()) {
    return Result<PreparedModelData, std::string>::makeError(
        meshResult.error());
  }
  return prepare(std::move(meshResult.value()));
}

Result<PreparedModelData, std::string>
Model::prepareSceneMeshFromFile(std::string_view path, uint32_t sceneMeshIndex,
                                const MeshImportOptions &options,
                                std::pmr::memory_resource *memory) {
  auto meshResult = MeshImporter::loadSceneMeshFromFile(path, sceneMeshIndex,
                                                        options, memory);
  if (meshResult.hasError()) {
    return Result<PreparedModelData, std::string>::makeError(
        meshResult.error());
  }
  return prepare(std::move(meshResult.value()));
}

Result<std::unique_ptr<Model>, std::string>
Model::createPrepared(GPUDevice &gpu, PreparedModelData data,
                      std::string_view debugName) {
  ModelAnimationPackedData animation{};
  animation.skinInfluences = BufferLayout<std::vector<std::byte>>{
      .data = std::move(data.skinInfluences.bytes),
      .count = data.skinInfluences.count,
      .strideBytes = data.skinInfluences.stride,
  };
  animation.morphMeta = BufferLayout<std::vector<std::byte>>{
      .data = std::move(data.morphMeta.bytes),
      .count = data.morphMeta.count,
      .strideBytes = data.morphMeta.stride,
  };
  animation.morphDeltas = BufferLayout<std::vector<std::byte>>{
      .data = std::move(data.morphDeltas.bytes),
      .count = data.morphDeltas.count,
      .strideBytes = data.morphDeltas.stride,
  };
  return createPacked(
      gpu,
      PackedSource{
          .vertexCount = static_cast<uint32_t>(data.mesh.vertices.size()),
          .vertices = data.packedVertexBytes,
          .indices = data.mesh.indices,
          .submeshes = data.mesh.submeshes,
          .meshlets = data.mesh.meshlets,
          .meshletVertexIndices = data.mesh.meshletVertexIndices,
          .meshletPrimitiveIndices = data.mesh.meshletPrimitiveIndices,
          .bounds = computeModelBounds(data.mesh.vertices),
          .vertexFormat = data.packedVertexFormat,
          .staticDecode = data.staticDecode.bytes,
          .animation = &animation,
      },
      debugName);
}

Result<std::unique_ptr<PreparedGpuModelData>, std::string>
Model::prepareGpu(GPUDevice &gpu, PreparedModelData data,
                  std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!gpu.supportsBackgroundBufferPreparation() ||
      !gpu.supportsBackgroundBufferBatchPreparation() ||
      !gpu.supportsBackgroundGeometryPreparation()) {
    return Result<std::unique_ptr<PreparedGpuModelData>, std::string>::
        makeError("background model GPU preparation is unsupported");
  }
  auto sourceMaterialCountResult =
      computeSourceMaterialCount(std::span<const Submesh>(
          data.mesh.submeshes.data(), data.mesh.submeshes.size()));
  if (sourceMaterialCountResult.hasError()) {
    return Result<std::unique_ptr<PreparedGpuModelData>,
                  std::string>::makeError(sourceMaterialCountResult.error());
  }
  MeshBinaryMorphMetaRecord morphMeta{};
  if (!data.morphMeta.bytes.empty()) {
    std::memcpy(&morphMeta, data.morphMeta.bytes.data(), sizeof(morphMeta));
  }
  auto meshletResult = prepareMeshletBufferData(
      std::span<const MeshletDescriptor>(data.mesh.meshlets.data(),
                                         data.mesh.meshlets.size()),
      std::span<const uint32_t>(data.mesh.meshletVertexIndices.data(),
                                data.mesh.meshletVertexIndices.size()),
      std::span<const uint8_t>(data.mesh.meshletPrimitiveIndices.data(),
                               data.mesh.meshletPrimitiveIndices.size()),
      std::span<const Submesh>(data.mesh.submeshes.data(),
                               data.mesh.submeshes.size()));
  if (meshletResult.hasError()) {
    return Result<std::unique_ptr<PreparedGpuModelData>,
                  std::string>::makeError(meshletResult.error());
  }
  auto impl = std::make_unique<PreparedGpuModelData::Impl>(std::move(data));
  impl->meshlets = std::move(meshletResult.value());
  impl->bounds = computeModelBounds(std::span<const Vertex>(
      impl->source.mesh.vertices.data(), impl->source.mesh.vertices.size()));
  impl->sourceMaterialCount = sourceMaterialCountResult.value();
  impl->morphTargetCount = morphMeta.morphTargetCount;
  impl->morphVertexCount = morphMeta.vertexCount;
  const std::string baseName =
      debugName.empty() ? std::string("model") : std::string(debugName);
  std::vector<PreparedBufferRequest> requests{};
  std::vector<PreparedGpuModelData::Impl::BufferSlot *> requestSlots{};
  requests.reserve(10u);
  requestSlots.reserve(10u);
  const auto appendSlot = [&baseName, &requests, &requestSlots](
                              PreparedGpuModelData::Impl::BufferSlot &slot,
                              std::span<const std::byte> bytes,
                              BufferUsage usage, std::string_view suffix) {
    if (bytes.empty()) {
      return;
    }
    slot.size = bytes.size();
    slot.usage = usage;
    slot.debugName = baseName + std::string(suffix);
    requests.push_back(PreparedBufferRequest{
        .desc =
            BufferDesc{
                .usage = usage,
                .storage = Storage::Device,
                .size = bytes.size(),
                .data = bytes,
                .immutable = true,
            },
        .debugName = slot.debugName,
    });
    requestSlots.push_back(&slot);
  };
  const std::span<const std::byte> indexBytes{
      reinterpret_cast<const std::byte *>(impl->source.mesh.indices.data()),
      impl->source.mesh.indices.size() * sizeof(uint32_t)};
  appendSlot(impl->buffers[PreparedGpuModelData::Impl::Vertex],
             impl->source.packedVertexBytes,
             BufferUsage::Storage | BufferUsage::Vertex, "_vertices");
  appendSlot(impl->buffers[PreparedGpuModelData::Impl::Index], indexBytes,
             BufferUsage::Index, "_indices");
  appendSlot(impl->buffers[PreparedGpuModelData::Impl::VertexDecode],
             impl->source.staticDecode.bytes, BufferUsage::Storage,
             "_vertex_decode");
  appendSlot(impl->buffers[PreparedGpuModelData::Impl::SkinInfluences],
             impl->source.skinInfluences.bytes, BufferUsage::Storage,
             "_skin_influences");
  appendSlot(impl->buffers[PreparedGpuModelData::Impl::MorphMeta],
             impl->source.morphMeta.bytes, BufferUsage::Storage, "_morph_meta");
  appendSlot(impl->buffers[PreparedGpuModelData::Impl::MorphDeltas],
             impl->source.morphDeltas.bytes, BufferUsage::Storage,
             "_morph_deltas");
  appendSlot(impl->buffers[PreparedGpuModelData::Impl::MeshletDescriptors],
             impl->meshlets.meshletDescriptors, BufferUsage::Storage,
             "_meshlets");
  appendSlot(impl->buffers[PreparedGpuModelData::Impl::MeshletVertexIndices],
             impl->meshlets.vertexIndices, BufferUsage::Storage,
             "_meshlet_vertices");
  appendSlot(impl->buffers[PreparedGpuModelData::Impl::MeshletPrimitiveIndices],
             impl->meshlets.primitiveIndices, BufferUsage::Storage,
             "_meshlet_primitives");
  appendSlot(impl->buffers[PreparedGpuModelData::Impl::MeshletLodRanges],
             impl->meshlets.lodRanges, BufferUsage::Storage,
             "_meshlet_lod_ranges");
  auto batch = gpu.prepareBufferBatch(requests);
  if (batch.hasError()) {
    return Result<std::unique_ptr<PreparedGpuModelData>,
                  std::string>::makeError(batch.error());
  }
  for (size_t index = 0u; index < requestSlots.size(); ++index) {
    requestSlots[index]->prepared = std::move(batch.value()[index]);
  }
  return Result<std::unique_ptr<PreparedGpuModelData>, std::string>::makeResult(
      std::unique_ptr<PreparedGpuModelData>(
          new PreparedGpuModelData(std::move(impl))));
}

Result<std::unique_ptr<Model>, std::string>
Model::publishPreparedGpu(GPUDevice &gpu,
                          std::unique_ptr<PreparedGpuModelData> prepared) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (prepared == nullptr || prepared->impl_ == nullptr) {
    return Result<std::unique_ptr<Model>, std::string>::makeError(
        "Model::publishPreparedGpu: prepared model is null");
  }
  PreparedGpuModelData::Impl &impl = *prepared->impl_;
  auto &slots = impl.buffers;
  auto vertexResult = gpu.publishPreparedBuffer(
      std::move(slots[PreparedGpuModelData::Impl::Vertex].prepared));
  if (vertexResult.hasError()) {
    return Result<std::unique_ptr<Model>, std::string>::makeError(
        vertexResult.error());
  }
  const BufferHandle vertexBuffer = vertexResult.value();
  auto indexResult = gpu.publishPreparedBuffer(
      std::move(slots[PreparedGpuModelData::Impl::Index].prepared));
  if (indexResult.hasError()) {
    gpu.destroyBuffer(vertexBuffer);
    return Result<std::unique_ptr<Model>, std::string>::makeError(
        indexResult.error());
  }
  const BufferHandle indexBuffer = indexResult.value();
  const auto &vertexSlot = slots[PreparedGpuModelData::Impl::Vertex];
  const auto &indexSlot = slots[PreparedGpuModelData::Impl::Index];
  auto geometryResult = gpu.adoptPreparedGeometry(
      vertexBuffer, vertexSlot.size,
      static_cast<uint32_t>(impl.source.mesh.vertices.size()), indexBuffer,
      indexSlot.size, static_cast<uint32_t>(impl.source.mesh.indices.size()),
      vertexSlot.debugName);
  if (geometryResult.hasError()) {
    gpu.destroyBuffer(vertexBuffer);
    gpu.destroyBuffer(indexBuffer);
    return Result<std::unique_ptr<Model>, std::string>::makeError(
        geometryResult.error());
  }
  const GeometryAllocationHandle geometry = geometryResult.value();
  std::array<std::unique_ptr<Buffer>, PreparedGpuModelData::Impl::SlotCount>
      buffers{};
  for (size_t index = PreparedGpuModelData::Impl::VertexDecode;
       index < PreparedGpuModelData::Impl::SlotCount; ++index) {
    auto &slot = slots[index];
    if (slot.prepared == nullptr) {
      continue;
    }
    auto result = Buffer::publishPrepared(gpu, std::move(slot.prepared),
                                          BufferDesc{.usage = slot.usage,
                                                     .storage = Storage::Device,
                                                     .size = slot.size,
                                                     .immutable = true},
                                          slot.debugName);
    if (result.hasError()) {
      gpu.releaseGeometry(geometry);
      return Result<std::unique_ptr<Model>, std::string>::makeError(
          result.error());
    }
    buffers[index] = std::move(result.value());
  }
  uint64_t vertexDecodeAddress = 0u;
  if (const auto &decode = buffers[PreparedGpuModelData::Impl::VertexDecode]) {
    vertexDecodeAddress = gpu.getBufferDeviceAddress(decode->handle());
    if (vertexDecodeAddress == 0u) {
      gpu.releaseGeometry(geometry);
      return Result<std::unique_ptr<Model>, std::string>::makeError(
          "Model::publishPreparedGpu: invalid vertex decode buffer address");
    }
  }
  ModelAnimationGpuView animationView{};
  if (const auto &skin = buffers[PreparedGpuModelData::Impl::SkinInfluences]) {
    animationView.skinInfluenceBuffer = skin->handle();
    animationView.skinInfluenceCount = impl.source.skinInfluences.count;
  }
  if (const auto &meta = buffers[PreparedGpuModelData::Impl::MorphMeta]) {
    animationView.morphMetaBuffer = meta->handle();
  }
  if (const auto &deltas = buffers[PreparedGpuModelData::Impl::MorphDeltas]) {
    animationView.morphDeltaBuffer = deltas->handle();
    animationView.morphTargetCount = impl.morphTargetCount;
    animationView.morphVertexCount = impl.morphVertexCount;
  }
  ModelMeshletGpuView meshletView{};
  if (const auto &meshlets =
          buffers[PreparedGpuModelData::Impl::MeshletDescriptors]) {
    meshletView.meshletBuffer = meshlets->handle();
    meshletView.meshletCount = impl.meshlets.meshletCount;
    meshletView.meshletVertexIndexBuffer =
        buffers[PreparedGpuModelData::Impl::MeshletVertexIndices]->handle();
    meshletView.meshletVertexIndexCount = impl.meshlets.vertexIndexCount;
    meshletView.meshletPrimitiveIndexBuffer =
        buffers[PreparedGpuModelData::Impl::MeshletPrimitiveIndices]->handle();
    meshletView.meshletPrimitiveIndexCount = impl.meshlets.primitiveIndexCount;
    meshletView.lodRangeBuffer =
        buffers[PreparedGpuModelData::Impl::MeshletLodRanges]->handle();
    meshletView.lodRangeCount = impl.meshlets.lodRangeCount;
  }
  std::pmr::memory_resource *const storageMemory =
      std::pmr::get_default_resource();
  std::pmr::vector<Submesh> ownedSubmeshes(storageMemory);
  ownedSubmeshes.assign(impl.source.mesh.submeshes.begin(),
                        impl.source.mesh.submeshes.end());
  std::pmr::vector<glm::vec4> meshletBoundsSpheres = copyMeshletBoundsSpheres(
      std::span<const MeshletDescriptor>(impl.source.mesh.meshlets.data(),
                                         impl.source.mesh.meshlets.size()),
      storageMemory);
  std::pmr::vector<uint32_t> sourceMaterialToRuntime(
      impl.sourceMaterialCount, Model::kInvalidMaterialIndex, storageMemory);
  return Result<std::unique_ptr<Model>, std::string>::makeResult(
      std::unique_ptr<Model>(new Model(
          gpu, geometry, std::move(ownedSubmeshes),
          static_cast<uint32_t>(impl.source.mesh.vertices.size()),
          static_cast<uint32_t>(impl.source.mesh.indices.size()), impl.bounds,
          impl.source.packedVertexFormat, animationView, meshletView,
          vertexDecodeAddress,
          std::move(buffers[PreparedGpuModelData::Impl::VertexDecode]),
          std::move(buffers[PreparedGpuModelData::Impl::SkinInfluences]),
          std::move(buffers[PreparedGpuModelData::Impl::MorphMeta]),
          std::move(buffers[PreparedGpuModelData::Impl::MorphDeltas]),
          std::move(buffers[PreparedGpuModelData::Impl::MeshletDescriptors]),
          std::move(buffers[PreparedGpuModelData::Impl::MeshletVertexIndices]),
          std::move(
              buffers[PreparedGpuModelData::Impl::MeshletPrimitiveIndices]),
          std::move(buffers[PreparedGpuModelData::Impl::MeshletLodRanges]),
          std::move(meshletBoundsSpheres),
          std::move(sourceMaterialToRuntime))));
}

Result<std::unique_ptr<Model>, std::string>
Model::createPacked(GPUDevice &gpu, const PackedSource &source,
                    std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  auto materialCount = computeSourceMaterialCount(source.submeshes);
  if (materialCount.hasError()) {
    return Result<std::unique_ptr<Model>, std::string>::makeError(
        materialCount.error());
  }
  const std::span<const std::byte> indexBytes{
      reinterpret_cast<const std::byte *>(source.indices.data()),
      source.indices.size_bytes()};
  auto geometry = gpu.allocateGeometry(
      source.vertices, source.vertexCount, indexBytes,
      static_cast<uint32_t>(source.indices.size()), debugName);
  if (geometry.hasError()) {
    return Result<std::unique_ptr<Model>, std::string>::makeError(
        geometry.error());
  }
  auto animation = createAnimationGpuBuffers(gpu, *source.animation, debugName);
  if (animation.hasError()) {
    gpu.releaseGeometry(geometry.value());
    return Result<std::unique_ptr<Model>, std::string>::makeError(
        animation.error());
  }
  ModelAnimationGpuBuffers animationBuffers = std::move(animation.value());
  if (!source.staticDecode.empty()) {
    auto decode = createStaticVertexDecodeBuffer(
        gpu, source.staticDecode, debugName,
        animationBuffers.vertexDecodeBufferAddress);
    if (decode.hasError()) {
      releaseAnimationGpuBuffers(gpu, animationBuffers);
      gpu.releaseGeometry(geometry.value());
      return Result<std::unique_ptr<Model>, std::string>::makeError(
          decode.error());
    }
    animationBuffers.vertexDecodeBuffer = std::move(decode.value());
  }
  auto meshlets = createMeshletGpuBuffers(
      gpu, source.meshlets, source.meshletVertexIndices,
      source.meshletPrimitiveIndices, source.submeshes, debugName);
  if (meshlets.hasError()) {
    releaseAnimationGpuBuffers(gpu, animationBuffers);
    gpu.releaseGeometry(geometry.value());
    return Result<std::unique_ptr<Model>, std::string>::makeError(
        meshlets.error());
  }
  ModelMeshletGpuBuffers meshletBuffers = std::move(meshlets.value());
  auto *storage = std::pmr::get_default_resource();
  std::pmr::vector<Submesh> ownedSubmeshes(source.submeshes.begin(),
                                           source.submeshes.end(), storage);
  std::pmr::vector<glm::vec4> meshletBoundsSpheres =
      copyMeshletBoundsSpheres(source.meshlets, storage);
  std::pmr::vector<uint32_t> sourceMaterialToRuntime(
      materialCount.value(), Model::kInvalidMaterialIndex, storage);
  return Result<std::unique_ptr<Model>, std::string>::makeResult(
      std::unique_ptr<Model>(new Model(
          gpu, geometry.value(), std::move(ownedSubmeshes), source.vertexCount,
          static_cast<uint32_t>(source.indices.size()), source.bounds,
          source.vertexFormat, animationBuffers.view, meshletBuffers.view,
          animationBuffers.vertexDecodeBufferAddress,
          std::move(animationBuffers.vertexDecodeBuffer),
          std::move(animationBuffers.skinInfluenceBuffer),
          std::move(animationBuffers.morphMetaBuffer),
          std::move(animationBuffers.morphDeltaBuffer),
          std::move(meshletBuffers.meshletBuffer),
          std::move(meshletBuffers.meshletVertexIndexBuffer),
          std::move(meshletBuffers.meshletPrimitiveIndexBuffer),
          std::move(meshletBuffers.meshletLodRangeBuffer),
          std::move(meshletBoundsSpheres),
          std::move(sourceMaterialToRuntime))));
}

Result<std::unique_ptr<Model>, std::string> Model::createFromFile(
    GPUDevice &gpu, std::string_view path, const MeshImportOptions &options,
    std::pmr::memory_resource *mem, std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
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
      ModelAnimationPackedData animation =
          reconstructAnimationPackedDataFromCache(*cachedMesh);
      auto model = createPacked(
          gpu,
          PackedSource{
              .vertexCount = cachedMesh->vertices.count,
              .vertices = cachedMesh->vertices.data,
              .indices = cachedMesh->indices,
              .submeshes = cachedMesh->submeshes,
              .meshlets = cachedMesh->meshlets,
              .meshletVertexIndices = cachedMesh->meshletVertexIndices,
              .meshletPrimitiveIndices = cachedMesh->meshletPrimitiveIndices,
              .bounds = cachedMesh->bounds,
              .vertexFormat =
                  packedVertexFormatFromLayoutId(cachedMesh->vertexLayoutId),
              .staticDecode = cachedMesh->staticVertexDecode.data,
              .animation = &animation,
          },
          debugName);
      if (!model.hasError()) {
        return model;
      }
      NURI_LOG_WARNING(
          "Model::createFromFile: Failed to create model from cache '%s': %s",
          cacheKey.cachePath.string().c_str(), model.error().c_str());
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
  ModelPackedVertexData packedVertices = packVerticesForModel(meshData);
  auto animationResult = packAnimationData(meshData);
  if (animationResult.hasError()) {
    return Result<std::unique_ptr<Model>, std::string>::makeError(
        animationResult.error());
  }
  ModelAnimationPackedData animationPacked = std::move(animationResult.value());
  auto modelResult = createPacked(
      gpu,
      PackedSource{
          .vertexCount = static_cast<uint32_t>(meshData.vertices.size()),
          .vertices = packedVertices.vertexBytes,
          .indices = meshData.indices,
          .submeshes = meshData.submeshes,
          .meshlets = meshData.meshlets,
          .meshletVertexIndices = meshData.meshletVertexIndices,
          .meshletPrimitiveIndices = meshData.meshletPrimitiveIndices,
          .bounds = computeModelBounds(meshData.vertices),
          .vertexFormat = packedVertices.format,
          .staticDecode = packedVertices.staticDecode.data,
          .animation = &animationPacked,
      },
      debugName);
  if (modelResult.hasError()) {
    const std::string pathStr{path};
    NURI_LOG_WARNING(
        "Model::createFromFile: Failed to create model from '%s': %s",
        pathStr.c_str(), modelResult.error().c_str());
    return Result<std::unique_ptr<Model>, std::string>::makeError(
        modelResult.error());
  }
  if (canWriteMeshCache) {
    maybeQueueMeshCacheWrite(cacheKeyResult.value(), options, packedVertices,
                             meshData, modelResult.value()->bounds(),
                             animationPacked);
  }
  NURI_LOG_DEBUG("Model::createFromFile: Created model from file '%.*s'",
                 static_cast<int>(path.size()), path.data());
  return modelResult;
}

} // namespace nuri
