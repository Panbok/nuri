#include "nuri/pch.h"

#include "nuri/resources/storage/material/material_binary_codec.h"

namespace nuri::material_binary_codec {

namespace {

template <typename T>
void appendPod(std::vector<std::byte> &bytes, const T &value) {
  static_assert(std::is_trivially_copyable_v<T>);
  const auto *src = reinterpret_cast<const std::byte *>(&value);
  bytes.insert(bytes.end(), src, src + sizeof(T));
}

template <typename T>
[[nodiscard]] Result<T, std::string> readPod(std::span<const std::byte> bytes,
                                             size_t &offset) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (offset > bytes.size() || sizeof(T) > (bytes.size() - offset)) {
    return Result<T, std::string>::makeError(
        "material_binary_codec: unexpected end of buffer");
  }
  T value{};
  std::memcpy(&value, bytes.data() + offset, sizeof(T));
  offset += sizeof(T);
  return Result<T, std::string>::makeResult(value);
}

} // namespace

void Writer::writeBytes(std::span<const std::byte> bytes) {
  bytes_.insert(bytes_.end(), bytes.begin(), bytes.end());
}

void Writer::writeU8(uint8_t value) { appendPod(bytes_, value); }

void Writer::writeU32(uint32_t value) { appendPod(bytes_, value); }

void Writer::writeU64(uint64_t value) { appendPod(bytes_, value); }

void Writer::writeI64(int64_t value) { appendPod(bytes_, value); }

void Writer::writeF32(float value) { appendPod(bytes_, value); }

void Writer::writeString(std::string_view value) {
  writeU32(static_cast<uint32_t>(value.size()));
  const auto *src = reinterpret_cast<const std::byte *>(value.data());
  writeBytes({src, value.size()});
}

void Writer::overwrite(size_t offset, std::span<const std::byte> bytes) {
  if (offset > bytes_.size() || bytes.size() > (bytes_.size() - offset)) {
    return;
  }
  std::memcpy(bytes_.data() + offset, bytes.data(), bytes.size());
}

Result<uint8_t, std::string> Reader::readU8() {
  return readPod<uint8_t>(bytes_, offset_);
}

Result<uint32_t, std::string> Reader::readU32() {
  return readPod<uint32_t>(bytes_, offset_);
}

Result<uint64_t, std::string> Reader::readU64() {
  return readPod<uint64_t>(bytes_, offset_);
}

Result<int64_t, std::string> Reader::readI64() {
  return readPod<int64_t>(bytes_, offset_);
}

Result<float, std::string> Reader::readF32() {
  return readPod<float>(bytes_, offset_);
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
