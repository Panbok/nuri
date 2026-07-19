#pragma once
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
    writeBytes({reinterpret_cast<const std::byte *>(value.data()),
                value.size()});
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

}
