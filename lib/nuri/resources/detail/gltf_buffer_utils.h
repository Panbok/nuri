#pragma once

#include "nuri/core/result.h"

#include <cstdint>
#include <filesystem>
#include <memory_resource>
#include <span>
#include <vector>

#include <glm/glm.hpp>
#include <yyjson.h>

namespace nuri::detail {

struct GltfAccessorInfo {
  uint32_t count = 0;
  uint32_t componentType = 0;
  uint32_t componentCount = 0;
  bool normalized = false;
};

[[nodiscard]] Result<std::pmr::vector<std::pmr::vector<std::byte>>, std::string>
loadGltfBuffers(const std::filesystem::path &path, yyjson_val *root,
                std::pmr::memory_resource *memory);

[[nodiscard]] Result<GltfAccessorInfo, std::string>
describeGltfAccessor(yyjson_val *root, uint32_t accessorIndex);

[[nodiscard]] Result<std::pmr::vector<float>, std::string>
readGltfAccessorAsFloatArray(
    yyjson_val *root, std::span<const std::pmr::vector<std::byte>> buffers,
    uint32_t accessorIndex, std::pmr::memory_resource *memory);

[[nodiscard]] Result<std::pmr::vector<uint16_t>, std::string>
readGltfAccessorAsU16Array(yyjson_val *root,
                           std::span<const std::pmr::vector<std::byte>> buffers,
                           uint32_t accessorIndex,
                           std::pmr::memory_resource *memory);

[[nodiscard]] Result<std::pmr::vector<glm::mat4>, std::string>
readGltfAccessorAsMat4Array(
    yyjson_val *root, std::span<const std::pmr::vector<std::byte>> buffers,
    uint32_t accessorIndex, std::pmr::memory_resource *memory);

} // namespace nuri::detail
