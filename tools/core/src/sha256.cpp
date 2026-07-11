#include "nuri/tools/core/sha256.h"

#include <array>
#include <bit>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace nuri::tools::core {
namespace {

constexpr std::array<uint32_t, 64> kRoundConstants{
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
    0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
    0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
    0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
    0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
    0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
    0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
    0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

class Sha256 {
public:
  void update(std::span<const std::byte> data) {
    for (const std::byte value : data) {
      block_[blockSize_++] = static_cast<uint8_t>(value);
      if (blockSize_ == block_.size()) {
        transform();
        bitLength_ += 512u;
        blockSize_ = 0u;
      }
    }
  }

  [[nodiscard]] std::array<uint8_t, 32> finish() {
    const uint64_t totalBitLength =
        bitLength_ + static_cast<uint64_t>(blockSize_) * 8u;
    block_[blockSize_++] = 0x80u;
    if (blockSize_ > 56u) {
      while (blockSize_ < block_.size()) {
        block_[blockSize_++] = 0u;
      }
      transform();
      blockSize_ = 0u;
    }
    while (blockSize_ < 56u) {
      block_[blockSize_++] = 0u;
    }
    for (uint32_t index = 0u; index < 8u; ++index) {
      block_[63u - index] =
          static_cast<uint8_t>(totalBitLength >> (index * 8u));
    }
    transform();

    std::array<uint8_t, 32> digest{};
    for (uint32_t word = 0u; word < state_.size(); ++word) {
      for (uint32_t byte = 0u; byte < 4u; ++byte) {
        digest[word * 4u + byte] =
            static_cast<uint8_t>(state_[word] >> (24u - byte * 8u));
      }
    }
    return digest;
  }

private:
  void transform() {
    std::array<uint32_t, 64> schedule{};
    for (uint32_t index = 0u; index < 16u; ++index) {
      const uint32_t offset = index * 4u;
      schedule[index] = (static_cast<uint32_t>(block_[offset]) << 24u) |
                        (static_cast<uint32_t>(block_[offset + 1u]) << 16u) |
                        (static_cast<uint32_t>(block_[offset + 2u]) << 8u) |
                        static_cast<uint32_t>(block_[offset + 3u]);
    }
    for (uint32_t index = 16u; index < schedule.size(); ++index) {
      const uint32_t s0 = std::rotr(schedule[index - 15u], 7) ^
                          std::rotr(schedule[index - 15u], 18) ^
                          (schedule[index - 15u] >> 3u);
      const uint32_t s1 = std::rotr(schedule[index - 2u], 17) ^
                          std::rotr(schedule[index - 2u], 19) ^
                          (schedule[index - 2u] >> 10u);
      schedule[index] = schedule[index - 16u] + s0 + schedule[index - 7u] + s1;
    }

    uint32_t a = state_[0];
    uint32_t b = state_[1];
    uint32_t c = state_[2];
    uint32_t d = state_[3];
    uint32_t e = state_[4];
    uint32_t f = state_[5];
    uint32_t g = state_[6];
    uint32_t h = state_[7];
    for (uint32_t index = 0u; index < schedule.size(); ++index) {
      const uint32_t s1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
      const uint32_t choose = (e & f) ^ (~e & g);
      const uint32_t temporary1 =
          h + s1 + choose + kRoundConstants[index] + schedule[index];
      const uint32_t s0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
      const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t temporary2 = s0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary1;
      d = c;
      c = b;
      b = a;
      a = temporary1 + temporary2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<uint32_t, 8> state_{0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u,
                                 0xa54ff53au, 0x510e527fu, 0x9b05688cu,
                                 0x1f83d9abu, 0x5be0cd19u};
  std::array<uint8_t, 64> block_{};
  size_t blockSize_ = 0u;
  uint64_t bitLength_ = 0u;
};

[[nodiscard]] std::string digestHex(const std::array<uint8_t, 32> &digest) {
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (const uint8_t value : digest) {
    out << std::setw(2) << static_cast<uint32_t>(value);
  }
  return out.str();
}

[[nodiscard]] std::string pathToUtf8(const std::filesystem::path &path) {
  const std::u8string encoded = path.generic_u8string();
  return {reinterpret_cast<const char *>(encoded.data()), encoded.size()};
}

} // namespace

std::string sha256Hex(std::span<const std::byte> data) {
  Sha256 hash;
  hash.update(data);
  return digestHex(hash.finish());
}

Result<std::string, std::string> sha256File(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return Result<std::string, std::string>::makeError(
        "sha256File: failed to open " + pathToUtf8(path));
  }
  Sha256 hash;
  std::array<std::byte, 64u * 1024u> buffer{};
  while (file) {
    file.read(reinterpret_cast<char *>(buffer.data()),
              static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = file.gcount();
    if (count > 0) {
      hash.update(std::span(buffer.data(), static_cast<size_t>(count)));
    }
  }
  if (!file.eof()) {
    return Result<std::string, std::string>::makeError(
        "sha256File: failed to read " + pathToUtf8(path));
  }
  return Result<std::string, std::string>::makeResult(digestHex(hash.finish()));
}

} // namespace nuri::tools::core
