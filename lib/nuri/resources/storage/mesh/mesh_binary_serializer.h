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

struct MeshBinarySerializeInput {
  uint64_t sourcePathHash = 0;
  uint64_t importOptionsHash = 0;
  uint64_t sourceSizeBytes = 0;
  int64_t sourceMtimeNs = 0;
  BoundingBox bounds{glm::vec3(0.0f), glm::vec3(0.0f)};
  std::span<const std::byte> packedVertexBytes{};
  uint32_t vertexCount = 0;
  uint32_t vertexStrideBytes = 0;
  std::span<const uint32_t> indices{};
  std::span<const Submesh> submeshes{};
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
  std::vector<std::byte> packedVertexBytes;
  uint32_t vertexCount = 0;
  uint32_t vertexStrideBytes = 0;
  std::vector<uint32_t> indices;
  std::vector<Submesh> submeshes;
  BoundingBox bounds{glm::vec3(0.0f), glm::vec3(0.0f)};
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
