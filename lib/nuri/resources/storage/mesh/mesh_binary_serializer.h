#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "nuri/core/result.h"
#include "nuri/math/types.h"
#include "nuri/resources/cpu/mesh_data.h"

namespace nuri {

template <typename BytesView> struct BufferLayout {
  BytesView data{};
  uint32_t count = 0;
  uint32_t strideBytes = 0;

  [[nodiscard]] bool empty() const noexcept { return data.empty(); }
  [[nodiscard]] size_t size() const noexcept { return data.size(); }

  [[nodiscard]] bool validate() const noexcept {
    if (data.empty()) {
      return count == 0u && strideBytes == 0u;
    }
    if (strideBytes == 0u) {
      return false;
    }
    const uint64_t expectedByteCount =
        static_cast<uint64_t>(count) * static_cast<uint64_t>(strideBytes);
    return expectedByteCount == static_cast<uint64_t>(data.size());
  }
};

struct MeshBinarySerializeInput {
  uint64_t sourcePathHash = 0;
  uint64_t importOptionsHash = 0;
  uint64_t sourceSizeBytes = 0;
  int64_t sourceMtimeNs = 0;
  BoundingBox bounds{glm::vec3(0.0f), glm::vec3(0.0f)};
  uint32_t vertexLayoutId = 0u;
  BufferLayout<std::span<const std::byte>> vertices{};
  BufferLayout<std::span<const std::byte>> staticVertexDecode{};
  std::span<const uint32_t> indices{};
  std::span<const Submesh> submeshes{};
  BufferLayout<std::span<const std::byte>> skinInfluences{};
  BufferLayout<std::span<const std::byte>> morphMeta{};
  BufferLayout<std::span<const std::byte>> morphDeltas{};
};

struct MeshBinaryDeserializeContext {
  uint64_t expectedSourcePathHash = 0;
  uint64_t expectedImportOptionsHash = 0;
  bool validateSourceFingerprint = false;
  bool sourceExists = false;
  uint64_t sourceSizeBytes = 0;
  int64_t sourceMtimeNs = 0;
};

struct MeshBinaryDecodedMesh {
  uint32_t vertexLayoutId = 0u;
  BufferLayout<std::vector<std::byte>> vertices{};
  BufferLayout<std::vector<std::byte>> staticVertexDecode{};
  std::vector<uint32_t> indices;
  std::vector<Submesh> submeshes;
  BoundingBox bounds{glm::vec3(0.0f), glm::vec3(0.0f)};
  BufferLayout<std::vector<std::byte>> skinInfluences{};
  BufferLayout<std::vector<std::byte>> morphMeta{};
  BufferLayout<std::vector<std::byte>> morphDeltas{};
};

enum class MeshBinaryDeserializeErrorCode : uint8_t {
  InvalidData = 0,
  StaleCache = 1,
};

struct MeshBinaryDeserializeError {
  MeshBinaryDeserializeErrorCode code =
      MeshBinaryDeserializeErrorCode::InvalidData;
  std::string message;

  [[nodiscard]] bool isStale() const noexcept {
    return code == MeshBinaryDeserializeErrorCode::StaleCache;
  }
};

[[nodiscard]] Result<std::vector<std::byte>, std::string>
meshBinarySerialize(const MeshBinarySerializeInput &input);

[[nodiscard]] Result<MeshBinaryDecodedMesh, MeshBinaryDeserializeError>
meshBinaryDeserialize(std::span<const std::byte> fileBytes,
                      const MeshBinaryDeserializeContext &context);

} // namespace nuri
