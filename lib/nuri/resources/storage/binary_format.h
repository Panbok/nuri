#pragma once
#include <cstdint>
#include <limits>
namespace nuri {

constexpr uint32_t kBinaryHeaderFlagLittleEndian = 1u;

constexpr uint32_t makeBinaryFourCC(char a, char b, char c, char d) {
  return static_cast<uint32_t>(static_cast<uint8_t>(a)) |
         static_cast<uint32_t>(static_cast<uint8_t>(b)) << 8u |
         static_cast<uint32_t>(static_cast<uint8_t>(c)) << 16u |
         static_cast<uint32_t>(static_cast<uint8_t>(d)) << 24u;
}

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);

}
