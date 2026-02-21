#pragma once

#include <array>
#include <cstdint>
#include <type_traits>

namespace nuri {

constexpr uint16_t kMeshBinaryFormatMajorVersion = 1;
constexpr uint16_t kMeshBinaryFormatMinorVersion = 0;

constexpr std::array<char, 8> kMeshBinaryMagic = {'N', 'U', 'R', 'I',
                                                  'M', 'S', 'H', '\0'};

constexpr uint32_t kMeshBinaryHeaderFlagLittleEndian = 1u << 0u;
constexpr uint32_t kMeshBinaryHeaderFlagCompressed = 1u << 1u;

constexpr uint32_t kMeshBinarySectionFlagCompressed = 1u << 0u;

constexpr uint32_t makeMeshBinaryFourCC(char a, char b, char c, char d) {
  return static_cast<uint32_t>(static_cast<uint8_t>(a)) |
         (static_cast<uint32_t>(static_cast<uint8_t>(b)) << 8u) |
         (static_cast<uint32_t>(static_cast<uint8_t>(c)) << 16u) |
         (static_cast<uint32_t>(static_cast<uint8_t>(d)) << 24u);
}

constexpr uint32_t kMeshBinarySectionVlay =
    makeMeshBinaryFourCC('V', 'L', 'A', 'Y');
constexpr uint32_t kMeshBinarySectionSmes =
    makeMeshBinaryFourCC('S', 'M', 'E', 'S');
constexpr uint32_t kMeshBinarySectionLods =
    makeMeshBinaryFourCC('L', 'O', 'D', 'S');
constexpr uint32_t kMeshBinarySectionVbuf =
    makeMeshBinaryFourCC('V', 'B', 'U', 'F');
constexpr uint32_t kMeshBinarySectionIbuf =
    makeMeshBinaryFourCC('I', 'B', 'U', 'F');

constexpr uint32_t kMeshBinaryLayoutIdPacked32 = 0u;
constexpr uint32_t kMeshBinaryPackedVertexStrideBytes = 32u;

// Attribute bitmask for the packed v1 shader layout.
constexpr uint32_t kMeshBinaryPackedAttributePosition = 1u << 0u;
constexpr uint32_t kMeshBinaryPackedAttributeUv = 1u << 1u;
constexpr uint32_t kMeshBinaryPackedAttributeNormal = 1u << 2u;
constexpr uint32_t kMeshBinaryPackedAttributeTangent = 1u << 3u;

#pragma pack(push, 1)
struct MeshBinaryHeader {
  std::array<char, 8> magic{};
  uint16_t majorVersion = 0;
  uint16_t minorVersion = 0;
  uint16_t headerSize = 0;
  uint16_t tocEntrySize = 0;
  uint32_t flags = 0;
  uint64_t fileSize = 0;
  uint64_t tocOffset = 0;
  uint32_t tocCount = 0;
  uint32_t reserved0 = 0;
  uint64_t sourcePathHash = 0;
  uint64_t importOptionsHash = 0;
  uint64_t sourceSizeBytes = 0;
  int64_t sourceMtimeNs = 0;
  float modelBoundsMin[3] = {0.0f, 0.0f, 0.0f};
  float modelBoundsMax[3] = {0.0f, 0.0f, 0.0f};
  uint32_t reserved1[4] = {0, 0, 0, 0};
};
#pragma pack(pop)

struct MeshBinarySectionTocEntry {
  uint32_t fourcc = 0;
  uint32_t flags = 0;
  uint64_t offset = 0;
  uint64_t sizeBytes = 0;
  uint32_t count = 0;
  uint32_t stride = 0;
};

struct MeshBinaryVertexLayoutRecord {
  uint32_t layoutId = kMeshBinaryLayoutIdPacked32;
  uint32_t strideBytes = kMeshBinaryPackedVertexStrideBytes;
  uint32_t attributeMask =
      kMeshBinaryPackedAttributePosition | kMeshBinaryPackedAttributeUv |
      kMeshBinaryPackedAttributeNormal | kMeshBinaryPackedAttributeTangent;
  uint32_t reserved = 0;
};

struct MeshBinarySubmeshRecord {
  uint32_t materialIndex = 0;
  uint32_t lodFirst = 0;
  uint32_t lodCount = 0;
  uint32_t layoutId = kMeshBinaryLayoutIdPacked32;
  float boundsMin[3] = {0.0f, 0.0f, 0.0f};
  float boundsMax[3] = {0.0f, 0.0f, 0.0f};
  uint32_t reserved[2] = {0, 0};
};

struct MeshBinaryLodRecord {
  uint32_t indexOffset = 0;
  uint32_t indexCount = 0;
  float error = 0.0f;
  uint32_t reserved = 0;
};

struct MeshBinaryBufferSectionHeader {
  uint32_t elementCount = 0;
  uint32_t elementStrideBytes = 0;
  uint32_t encodedSizeBytes = 0;
  uint32_t reserved = 0;
};

static_assert(sizeof(MeshBinaryHeader) == 116);
static_assert(sizeof(MeshBinarySectionTocEntry) == 32);
static_assert(sizeof(MeshBinaryVertexLayoutRecord) == 16);
static_assert(sizeof(MeshBinarySubmeshRecord) == 48);
static_assert(sizeof(MeshBinaryLodRecord) == 16);
static_assert(sizeof(MeshBinaryBufferSectionHeader) == 16);
static_assert(std::is_standard_layout_v<MeshBinaryHeader>);
static_assert(std::is_standard_layout_v<MeshBinarySectionTocEntry>);
static_assert(std::is_standard_layout_v<MeshBinaryVertexLayoutRecord>);
static_assert(std::is_standard_layout_v<MeshBinarySubmeshRecord>);
static_assert(std::is_standard_layout_v<MeshBinaryLodRecord>);
static_assert(std::is_standard_layout_v<MeshBinaryBufferSectionHeader>);
static_assert(std::is_trivially_copyable_v<MeshBinaryHeader>);
static_assert(std::is_trivially_copyable_v<MeshBinarySectionTocEntry>);
static_assert(std::is_trivially_copyable_v<MeshBinaryVertexLayoutRecord>);
static_assert(std::is_trivially_copyable_v<MeshBinarySubmeshRecord>);
static_assert(std::is_trivially_copyable_v<MeshBinaryLodRecord>);
static_assert(std::is_trivially_copyable_v<MeshBinaryBufferSectionHeader>);

} // namespace nuri
