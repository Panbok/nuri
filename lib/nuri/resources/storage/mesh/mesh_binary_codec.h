#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "nuri/core/result.h"

namespace nuri {

[[nodiscard]] Result<std::vector<std::byte>, std::string>
meshBinaryEncodeVertexBuffer(std::span<const std::byte> vertexBytes,
                             uint32_t vertexStrideBytes);

[[nodiscard]] Result<std::vector<std::byte>, std::string>
meshBinaryDecodeVertexBuffer(std::span<const std::byte> encodedBytes,
                             uint32_t vertexCount,
                             uint32_t vertexStrideBytes);

[[nodiscard]] Result<std::vector<std::byte>, std::string>
meshBinaryEncodeIndexBuffer(std::span<const uint32_t> indices,
                            uint32_t vertexCount);

[[nodiscard]] Result<std::vector<std::byte>, std::string>
meshBinaryDecodeIndexBuffer(std::span<const std::byte> encodedBytes,
                            uint32_t indexCount, uint32_t indexStrideBytes);

} // namespace nuri
