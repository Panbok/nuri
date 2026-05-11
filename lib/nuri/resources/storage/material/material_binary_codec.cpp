#include "nuri/pch.h"

#include "nuri/resources/storage/material/material_binary_codec.h"

#include <bit>

namespace nuri::material_binary_codec {

namespace {

void appendBytes(std::vector<std::byte> &bytes,
                 std::span<const std::byte> data) {
  const auto *src = data.data();
  bytes.insert(bytes.end(), src, src + data.size());
}

template <typename T> [[nodiscard]] T byteSwap(T value) {
  static_assert(std::is_integral_v<T>);
  if constexpr (sizeof(T) == 1u) {
    return value;
  } else {
    T out = 0;
    for (size_t i = 0; i < sizeof(T); ++i) {
      out <<= 8u;
      out |= static_cast<T>(value & static_cast<T>(0xffu));
      value >>= 8u;
    }
    return out;
  }
}

template <typename T> [[nodiscard]] T byteSwapIfNeeded(T value) {
  static_assert(std::is_integral_v<T>);
  if constexpr (sizeof(T) == 1u || std::endian::native == std::endian::little) {
    return value;
  } else {
    return byteSwap(value);
  }
}

void appendLittleEndianU32(std::vector<std::byte> &bytes, uint32_t value) {
  const uint32_t encoded = byteSwapIfNeeded(value);
  const auto *src = reinterpret_cast<const std::byte *>(&encoded);
  bytes.insert(bytes.end(), src, src + sizeof(encoded));
}

void appendLittleEndianU64(std::vector<std::byte> &bytes, uint64_t value) {
  const uint64_t encoded = byteSwapIfNeeded(value);
  const auto *src = reinterpret_cast<const std::byte *>(&encoded);
  bytes.insert(bytes.end(), src, src + sizeof(encoded));
}

template <typename T>
[[nodiscard]] Result<T, std::string>
readLittleEndianIntegral(std::span<const std::byte> bytes, size_t &offset) {
  static_assert(std::is_integral_v<T>);
  if (offset > bytes.size() || sizeof(T) > (bytes.size() - offset)) {
    return Result<T, std::string>::makeError(
        "material_binary_codec: unexpected end of buffer");
  }
  T value{};
  std::memcpy(&value, bytes.data() + offset, sizeof(T));
  offset += sizeof(T);
  return Result<T, std::string>::makeResult(byteSwapIfNeeded(value));
}

} // namespace

void Writer::writeBytes(std::span<const std::byte> bytes) {
  appendBytes(bytes_, bytes);
}

void Writer::writeU8(uint8_t value) {
  const auto *src = reinterpret_cast<const std::byte *>(&value);
  appendBytes(bytes_, {src, 1u});
}

void Writer::writeU32(uint32_t value) { appendLittleEndianU32(bytes_, value); }

void Writer::writeU64(uint64_t value) { appendLittleEndianU64(bytes_, value); }

void Writer::writeI64(int64_t value) {
  writeU64(std::bit_cast<uint64_t>(value));
}

void Writer::writeF32(float value) { writeU32(std::bit_cast<uint32_t>(value)); }

void Writer::writeString(std::string_view value) {
  if (value.size() > std::numeric_limits<uint32_t>::max()) {
    throw std::length_error("material_binary_codec::Writer::writeString: "
                            "string exceeds uint32_t length");
  }
  writeU32(static_cast<uint32_t>(value.size()));
  const auto *src = reinterpret_cast<const std::byte *>(value.data());
  writeBytes({src, value.size()});
}

bool Writer::overwrite(size_t offset, std::span<const std::byte> bytes) {
  if (offset > bytes_.size() || bytes.size() > (bytes_.size() - offset)) {
    return false;
  }
  std::memcpy(bytes_.data() + offset, bytes.data(), bytes.size());
  return true;
}

Result<uint8_t, std::string> Reader::readU8() {
  return readLittleEndianIntegral<uint8_t>(bytes_, offset_);
}

Result<uint32_t, std::string> Reader::readU32() {
  return readLittleEndianIntegral<uint32_t>(bytes_, offset_);
}

Result<uint64_t, std::string> Reader::readU64() {
  return readLittleEndianIntegral<uint64_t>(bytes_, offset_);
}

Result<int64_t, std::string> Reader::readI64() {
  auto value = readU64();
  if (value.hasError()) {
    return Result<int64_t, std::string>::makeError(value.error());
  }
  return Result<int64_t, std::string>::makeResult(
      std::bit_cast<int64_t>(value.value()));
}

Result<float, std::string> Reader::readF32() {
  auto value = readU32();
  if (value.hasError()) {
    return Result<float, std::string>::makeError(value.error());
  }
  return Result<float, std::string>::makeResult(
      std::bit_cast<float>(value.value()));
}

Result<std::string, std::string> Reader::readString() {
  auto sizeResult = readU32();
  if (sizeResult.hasError()) {
    return Result<std::string, std::string>::makeError(sizeResult.error());
  }
  const uint32_t sizeBytes = sizeResult.value();
  auto bytesResult = readBytes(sizeBytes);
  if (bytesResult.hasError()) {
    return Result<std::string, std::string>::makeError(bytesResult.error());
  }
  const auto bytes = bytesResult.value();
  return Result<std::string, std::string>::makeResult(
      std::string(reinterpret_cast<const char *>(bytes.data()), bytes.size()));
}

Result<std::span<const std::byte>, std::string>
Reader::readBytes(size_t sizeBytes) {
  if (offset_ > bytes_.size() || sizeBytes > (bytes_.size() - offset_)) {
    return Result<std::span<const std::byte>, std::string>::makeError(
        "material_binary_codec: unexpected end of buffer");
  }
  const std::span<const std::byte> bytes(bytes_.data() + offset_, sizeBytes);
  offset_ += sizeBytes;
  return Result<std::span<const std::byte>, std::string>::makeResult(bytes);
}

} // namespace nuri::material_binary_codec
