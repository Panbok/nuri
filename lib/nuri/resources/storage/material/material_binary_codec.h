#pragma once

#include "nuri/core/result.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace nuri::material_binary_codec {

class Writer {
public:
  void writeBytes(std::span<const std::byte> bytes);
  void writeU8(uint8_t value);
  void writeU32(uint32_t value);
  void writeU64(uint64_t value);
  void writeI64(int64_t value);
  void writeF32(float value);
  void writeString(std::string_view value);
  void overwrite(size_t offset, std::span<const std::byte> bytes);

  [[nodiscard]] const std::vector<std::byte> &bytes() const noexcept {
    return bytes_;
  }

private:
  std::vector<std::byte> bytes_{};
};

class Reader {
public:
  explicit Reader(std::span<const std::byte> bytes) : bytes_(bytes) {}

  [[nodiscard]] Result<uint8_t, std::string> readU8();
  [[nodiscard]] Result<uint32_t, std::string> readU32();
  [[nodiscard]] Result<uint64_t, std::string> readU64();
  [[nodiscard]] Result<int64_t, std::string> readI64();
  [[nodiscard]] Result<float, std::string> readF32();
  [[nodiscard]] Result<std::string, std::string> readString();
  [[nodiscard]] Result<std::span<const std::byte>, std::string>
  readBytes(size_t sizeBytes);
  [[nodiscard]] bool empty() const noexcept { return offset_ >= bytes_.size(); }

private:
  std::span<const std::byte> bytes_{};
  size_t offset_ = 0u;
};

} // namespace nuri::material_binary_codec
