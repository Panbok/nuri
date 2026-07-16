#pragma once

#include <array>
#include <cstdint>
#include <type_traits>

namespace nuri {

constexpr uint16_t kMaterialBinaryFormatMajorVersion = 5;
constexpr uint16_t kMaterialBinaryFormatMinorVersion = 0;

constexpr std::array<char, 8> kMaterialBinaryMagic = {'N', 'U', 'R', 'I',
                                                      'M', 'A', 'T', '\0'};

#pragma pack(push, 1)
struct MaterialBinaryHeader {
  std::array<char, 8> magic{};
  uint16_t majorVersion = 0;
  uint16_t minorVersion = 0;
  uint32_t fileSize = 0;
  uint64_t sourcePathHash = 0;
  uint64_t sourceSizeBytes = 0;
  int64_t sourceMtimeNs = 0;
  uint32_t materialCount = 0;
  uint32_t reserved = 0;
};
#pragma pack(pop)

static_assert(sizeof(MaterialBinaryHeader) == 48);
static_assert(std::is_trivially_copyable_v<MaterialBinaryHeader>);

} // namespace nuri
