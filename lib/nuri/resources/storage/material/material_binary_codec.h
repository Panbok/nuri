#pragma once

#include "nuri/core/result.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace nuri::material_binary_codec {

// Writer appends values to an in-memory byte buffer in little-endian order.
// `writeBytes()` appends raw bytes exactly as provided, and `bytes()` exposes
// the accumulated storage by reference without copying it. `overwrite()`
// writes in-place only within the current buffer range and returns `false`
// when `offset + bytes.size()` exceeds `bytes().size()`.
class Writer {
public:
  // Appends raw bytes exactly as provided.
  void writeBytes(std::span<const std::byte> bytes);
  // Serializes the value as a single byte.
  void writeU8(uint8_t value);
  // Serializes the value in little-endian order.
  void writeU32(uint32_t value);
  // Serializes the value in little-endian order.
  void writeU64(uint64_t value);
  // Serializes the value in little-endian order.
  void writeI64(int64_t value);
  // Serializes the IEEE-754 bit pattern in little-endian order.
  void writeF32(float value);
  void writeString(std::string_view value);
  [[nodiscard]] bool overwrite(size_t offset, std::span<const std::byte> bytes);

  [[nodiscard]] const std::vector<std::byte> &bytes() const noexcept {
    return bytes_;
  }

private:
  std::vector<std::byte> bytes_{};
};

class Reader {
public:
  // Reader stores a non-owning view of `bytes`; it does not copy or take
  // ownership of the underlying buffer, so callers must keep that storage
  // alive for the entire Reader lifetime.
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
