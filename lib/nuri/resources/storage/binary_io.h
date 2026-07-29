#pragma once
#include "nuri/core/result.h"
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
namespace nuri {

static_assert(std::endian::native == std::endian::little);

class BinaryWriter {
public:
  template <typename T> void write(const T &value) {
    static_assert(std::is_trivially_copyable_v<T>);
    writeBytes({reinterpret_cast<const std::byte *>(&value), sizeof(T)});
  }
  void writeBytes(std::span<const std::byte> bytes) {
    bytes_.insert(bytes_.end(), bytes.begin(), bytes.end());
  }
  void writeString(std::string_view value) {
    if (value.size() > std::numeric_limits<uint32_t>::max()) {
      valid_ = false;
      return;
    }
    write(static_cast<uint32_t>(value.size()));
    writeBytes(
        {reinterpret_cast<const std::byte *>(value.data()), value.size()});
  }
  [[nodiscard]] bool valid() const noexcept { return valid_; }
  [[nodiscard]] const std::vector<std::byte> &bytes() const noexcept {
    return bytes_;
  }
  [[nodiscard]] std::vector<std::byte> take() && { return std::move(bytes_); }

private:
  std::vector<std::byte> bytes_{};
  bool valid_ = true;
};

class BinaryReader {
public:
  explicit BinaryReader(std::span<const std::byte> bytes) : bytes_(bytes) {}
  template <typename T> [[nodiscard]] T read() {
    static_assert(std::is_trivially_copyable_v<T>);
    T value{};
    const auto bytes = readBytes(sizeof(T));
    if (valid_) {
      std::memcpy(&value, bytes.data(), sizeof(T));
    }
    return value;
  }
  [[nodiscard]] std::string readString() {
    const uint32_t size = read<uint32_t>();
    const auto bytes = readBytes(size);
    return valid_ ? std::string(reinterpret_cast<const char *>(bytes.data()),
                                bytes.size())
                  : std::string{};
  }
  [[nodiscard]] std::span<const std::byte> readBytes(size_t size) {
    if (!valid_ || size > bytes_.size() - offset_) {
      valid_ = false;
      return {};
    }
    const auto result = bytes_.subspan(offset_, size);
    offset_ += size;
    return result;
  }
  [[nodiscard]] bool valid() const noexcept { return valid_; }
  [[nodiscard]] bool empty() const noexcept {
    return valid_ && offset_ == bytes_.size();
  }

private:
  std::span<const std::byte> bytes_{};
  size_t offset_ = 0u;
  bool valid_ = true;
};

[[nodiscard]] inline bool binaryRangeValid(size_t total, uint64_t offset,
                                           uint64_t size);
template <typename T, typename Allocator>
void readPodArrayAt(std::span<const std::byte> bytes, uint64_t offset,
                    uint32_t count, std::vector<T, Allocator> &out);

struct BinarySection {
  uint32_t fourcc = 0u;
  uint32_t flags = 0u;
  uint32_t count = 0u;
  uint32_t stride = 0u;
  std::span<const std::byte> payload{};
};

[[nodiscard]] inline bool alignBinaryOffset(uint64_t value, uint64_t alignment,
                                            uint64_t &out) noexcept {
  if (alignment == 0u) {
    return false;
  }
  const uint64_t remainder = value % alignment;
  const uint64_t padding = remainder == 0u ? 0u : alignment - remainder;
  if (padding > std::numeric_limits<uint64_t>::max() - value) {
    return false;
  }
  out = value + padding;
  return true;
}

template <typename Header, typename TocEntry>
[[nodiscard]] Result<std::vector<std::byte>, std::string>
writeSectionedBinary(Header header, std::span<const BinarySection> sections,
                     uint64_t alignment, std::string_view context) {
  const auto fail = [context](std::string_view reason) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        std::string(context) + ": " + std::string(reason));
  };
  if (sections.size() > std::numeric_limits<uint32_t>::max()) {
    return fail("section count exceeds uint32");
  }
  const uint64_t tocBytes =
      static_cast<uint64_t>(sections.size()) * sizeof(TocEntry);
  if (tocBytes > std::numeric_limits<uint64_t>::max() - sizeof(Header)) {
    return fail("TOC layout overflow");
  }
  uint64_t cursor = 0u;
  if (!alignBinaryOffset(sizeof(Header) + tocBytes, alignment, cursor)) {
    return fail("section layout overflow");
  }
  for (const BinarySection &section : sections) {
    if (!alignBinaryOffset(cursor, alignment, cursor) ||
        section.payload.size() >
            std::numeric_limits<uint64_t>::max() - cursor) {
      return fail("section layout overflow");
    }
    cursor += section.payload.size();
  }
  uint64_t fileSize = 0u;
  if (!alignBinaryOffset(cursor, alignment, fileSize) ||
      fileSize > std::numeric_limits<size_t>::max()) {
    return fail("output is too large");
  }
  header.fileSize = fileSize;
  header.tocOffset = sizeof(Header);
  header.tocCount = static_cast<uint32_t>(sections.size());
  std::vector<std::byte> out(static_cast<size_t>(fileSize), std::byte{0});
  std::memcpy(out.data(), &header, sizeof(header));
  cursor = 0u;
  (void)alignBinaryOffset(sizeof(Header) + tocBytes, alignment, cursor);
  for (size_t i = 0; i < sections.size(); ++i) {
    (void)alignBinaryOffset(cursor, alignment, cursor);
    const BinarySection &section = sections[i];
    const TocEntry entry{.fourcc = section.fourcc,
                         .flags = section.flags,
                         .offset = cursor,
                         .sizeBytes = section.payload.size(),
                         .count = section.count,
                         .stride = section.stride};
    std::memcpy(out.data() + sizeof(Header) + i * sizeof(TocEntry), &entry,
                sizeof(entry));
    if (!section.payload.empty()) {
      std::memcpy(out.data() + static_cast<size_t>(cursor),
                  section.payload.data(), section.payload.size());
    }
    cursor += section.payload.size();
  }
  return Result<std::vector<std::byte>, std::string>::makeResult(
      std::move(out));
}

template <typename TocEntry, typename Allocator>
[[nodiscard]] Result<bool, std::string>
readSectionToc(std::span<const std::byte> bytes, uint64_t offset,
               uint32_t count, std::vector<TocEntry, Allocator> &out,
               std::string_view context,
               uint32_t maxCount = std::numeric_limits<uint32_t>::max()) {
  const auto fail = [context](std::string_view reason) {
    return Result<bool, std::string>::makeError(std::string(context) + ": " +
                                                std::string(reason));
  };
  if (count > maxCount ||
      count > std::numeric_limits<uint64_t>::max() / sizeof(TocEntry)) {
    return fail("invalid TOC count");
  }
  const uint64_t tocBytes = static_cast<uint64_t>(count) * sizeof(TocEntry);
  if (!binaryRangeValid(bytes.size(), offset, tocBytes)) {
    return fail("invalid TOC range");
  }
  readPodArrayAt(bytes, offset, count, out);
  for (const TocEntry &entry : out) {
    if (!binaryRangeValid(bytes.size(), entry.offset, entry.sizeBytes)) {
      return fail("invalid section range");
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

template <typename T>
void appendPod(std::vector<std::byte> &out, const T &value) {
  static_assert(std::is_trivially_copyable_v<T>);
  const size_t offset = out.size();
  out.resize(offset + sizeof(T));
  std::memcpy(out.data() + offset, &value, sizeof(T));
}

template <typename T>
void appendPodArray(std::vector<std::byte> &out, std::span<const T> values) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (values.empty()) {
    return;
  }
  const size_t offset = out.size();
  out.resize(offset + values.size_bytes());
  std::memcpy(out.data() + offset, values.data(), values.size_bytes());
}

[[nodiscard]] inline bool binaryRangeValid(size_t total, uint64_t offset,
                                           uint64_t size) {
  return offset <= total && size <= total - static_cast<size_t>(offset);
}

template <typename T>
[[nodiscard]] T readPodAt(std::span<const std::byte> bytes, uint64_t offset) {
  T value{};
  std::memcpy(&value, bytes.data() + static_cast<size_t>(offset), sizeof(T));
  return value;
}

template <typename T, typename Allocator>
void readPodArrayAt(std::span<const std::byte> bytes, uint64_t offset,
                    uint32_t count, std::vector<T, Allocator> &out) {
  static_assert(std::is_trivially_copyable_v<T>);
  out.resize(count);
  if (count != 0u) {
    std::memcpy(out.data(), bytes.data() + static_cast<size_t>(offset),
                static_cast<size_t>(count) * sizeof(T));
  }
}

} // namespace nuri
