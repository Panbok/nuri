#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/resources/cpu/material_data.h"
#include "nuri/resources/gpu/material.h"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace nuri {

struct SceneMaterialTextureCacheRecord {
  uint64_t artifactIdentityHash = 0u;
};

struct SceneMaterialRecord {
  uint32_t sourceMaterialIndex = 0;
  MaterialData sourceMaterial{};
  std::array<SceneMaterialTextureCacheRecord, kMaterialTextureSlotCount>
      textureCache{};
};

struct SceneMaterialCacheData {
  std::vector<SceneMaterialRecord> materials{};
};

struct MaterialBinarySerializeInput {
  uint64_t sourcePathHash = 0;
  uint64_t sourceSizeBytes = 0;
  int64_t sourceMtimeNs = 0;
  std::span<const SceneMaterialRecord> materials{};
};

struct MaterialBinaryDeserializeContext {
  uint64_t expectedSourcePathHash = 0;
  bool validateSourceFingerprint = true;
  bool sourceExists = false;
  uint64_t sourceSizeBytes = 0;
  int64_t sourceMtimeNs = 0;
};

struct MaterialBinaryDeserializeError {
  std::string message{};
  bool stale = false;

  [[nodiscard]] bool isStale() const noexcept { return stale; }
};

[[nodiscard]] NURI_API Result<std::vector<std::byte>, std::string>
materialBinarySerialize(const MaterialBinarySerializeInput &input);

[[nodiscard]] NURI_API
    Result<SceneMaterialCacheData, MaterialBinaryDeserializeError>
    materialBinaryDeserialize(std::span<const std::byte> fileBytes,
                              const MaterialBinaryDeserializeContext &context);

} // namespace nuri
