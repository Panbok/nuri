#pragma once
#include "nuri/resources/storage/binary_format.h"
#include <array>
#include <cstdint>
namespace nuri {

constexpr uint16_t kMeshBinaryFormatMajorVersion = 2;
constexpr uint16_t kMeshBinaryFormatMinorVersion = 2;

constexpr std::array<char, 8> kMeshBinaryMagic = {'N', 'U', 'R', 'I',
                                                  'M', 'S', 'H', '\0'};

constexpr uint32_t kMeshBinaryHeaderFlagCompressed = 1u << 1u;
constexpr uint32_t kMeshBinarySectionFlagCompressed = 1u << 0u;

constexpr uint32_t kMeshBinarySectionVlay =
    makeBinaryFourCC('V', 'L', 'A', 'Y');
constexpr uint32_t kMeshBinarySectionSmes =
    makeBinaryFourCC('S', 'M', 'E', 'S');
constexpr uint32_t kMeshBinarySectionLods =
    makeBinaryFourCC('L', 'O', 'D', 'S');
constexpr uint32_t kMeshBinarySectionVbuf =
    makeBinaryFourCC('V', 'B', 'U', 'F');
constexpr uint32_t kMeshBinarySectionIbuf =
    makeBinaryFourCC('I', 'B', 'U', 'F');
constexpr uint32_t kMeshBinarySectionVinf =
    makeBinaryFourCC('V', 'I', 'N', 'F');
constexpr uint32_t kMeshBinarySectionVdec =
    makeBinaryFourCC('V', 'D', 'E', 'C');
constexpr uint32_t kMeshBinarySectionMmta =
    makeBinaryFourCC('M', 'M', 'T', 'A');
constexpr uint32_t kMeshBinarySectionMdel =
    makeBinaryFourCC('M', 'D', 'E', 'L');
constexpr uint32_t kMeshBinarySectionMlds =
    makeBinaryFourCC('M', 'L', 'D', 'S');
constexpr uint32_t kMeshBinarySectionMlvi =
    makeBinaryFourCC('M', 'L', 'V', 'I');
constexpr uint32_t kMeshBinarySectionMlpi =
    makeBinaryFourCC('M', 'L', 'P', 'I');
constexpr uint32_t kMeshBinarySectionMlrg =
    makeBinaryFourCC('M', 'L', 'R', 'G');

constexpr uint32_t kMeshBinaryLayoutIdStaticQuantized20 = 0u;
constexpr uint32_t kMeshBinaryLayoutIdAnimatedFloat24 = 1u;
constexpr uint32_t kMeshBinaryLayoutIdAnimatedFloat32 = 2u;
constexpr uint32_t kMeshBinaryStaticVertexStrideBytes = 20u;
constexpr uint32_t kMeshBinaryAnimatedVertexStrideBytes = 24u;
constexpr uint32_t kMeshBinaryAnimatedFloat32VertexStrideBytes = 32u;

constexpr uint32_t kMeshBinaryPackedAttributePosition = 1u << 0u;
constexpr uint32_t kMeshBinaryPackedAttributeUv = 1u << 1u;
constexpr uint32_t kMeshBinaryPackedAttributeUv1 = 1u << 2u;
constexpr uint32_t kMeshBinaryPackedAttributeNormal = 1u << 3u;
constexpr uint32_t kMeshBinaryPackedAttributeTangent = 1u << 4u;

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
  uint32_t layoutId = kMeshBinaryLayoutIdStaticQuantized20;
  uint32_t strideBytes = kMeshBinaryStaticVertexStrideBytes;
  uint32_t attributeMask =
      kMeshBinaryPackedAttributePosition | kMeshBinaryPackedAttributeUv |
      kMeshBinaryPackedAttributeUv1 | kMeshBinaryPackedAttributeNormal;
  uint32_t reserved = 0;
};

struct MeshBinarySubmeshRecord {
  uint32_t vertexOffset = 0;
  uint32_t vertexCount = 0;
  uint32_t materialIndex = 0;
  uint32_t lodFirst = 0;
  uint32_t lodCount = 0;
  uint32_t layoutId = kMeshBinaryLayoutIdStaticQuantized20;
  uint32_t morphTargetFirst = 0;
  uint32_t morphTargetCount = 0;
  float boundsMin[3] = {0.0f, 0.0f, 0.0f};
  float boundsMax[3] = {0.0f, 0.0f, 0.0f};
  float authoredScale[3] = {1.0f, 1.0f, 1.0f};
  uint32_t reserved = 0;
};

struct MeshBinarySubmeshRecordV0 {
  uint32_t materialIndex = 0;
  uint32_t lodFirst = 0;
  uint32_t lodCount = 0;
  uint32_t layoutId = kMeshBinaryLayoutIdStaticQuantized20;
  float boundsMin[3] = {0.0f, 0.0f, 0.0f};
  float boundsMax[3] = {0.0f, 0.0f, 0.0f};
  uint32_t reserved[2] = {0, 0};
};

struct MeshBinarySubmeshRecordV1 {
  uint32_t materialIndex = 0;
  uint32_t lodFirst = 0;
  uint32_t lodCount = 0;
  uint32_t layoutId = kMeshBinaryLayoutIdStaticQuantized20;
  float boundsMin[3] = {0.0f, 0.0f, 0.0f};
  float boundsMax[3] = {0.0f, 0.0f, 0.0f};
  float authoredScale[3] = {1.0f, 1.0f, 1.0f};
  uint32_t reserved = 0;
};

struct MeshBinaryLodRecord {
  uint32_t indexOffset = 0;
  uint32_t indexCount = 0;
  float error = 0.0f;
  uint32_t reserved = 0;
};

struct MeshBinaryLodMeshletRangeRecord {
  uint32_t meshletOffset = 0;
  uint32_t meshletCount = 0;
};

struct MeshBinaryMeshletRecord {
  uint32_t vertexOffset = 0;
  uint32_t vertexCount = 0;
  uint32_t primitiveOffset = 0;
  uint32_t primitiveCount = 0;
  float boundsSphere[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float coneApex[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float coneAxisCutoff[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};

struct MeshBinaryBufferSectionHeader {
  uint32_t elementCount = 0;
  uint32_t elementStrideBytes = 0;
  uint32_t encodedSizeBytes = 0;
  uint32_t reserved = 0;
};

struct MeshBinaryMorphMetaRecord {
  uint32_t morphTargetCount = 0;
  uint32_t vertexCount = 0;
  uint32_t reserved0 = 0;
  uint32_t reserved1 = 0;
};

static_assert(sizeof(MeshBinaryHeader) == 116);
static_assert(sizeof(MeshBinarySectionTocEntry) == 32);
static_assert(sizeof(MeshBinaryVertexLayoutRecord) == 16);
static_assert(sizeof(MeshBinarySubmeshRecord) == 72);
static_assert(sizeof(MeshBinarySubmeshRecordV0) == 48);
static_assert(sizeof(MeshBinarySubmeshRecordV1) == 56);
static_assert(sizeof(MeshBinaryLodRecord) == 16);
static_assert(sizeof(MeshBinaryLodMeshletRangeRecord) == 8);
static_assert(sizeof(MeshBinaryMeshletRecord) == 64);
static_assert(sizeof(MeshBinaryBufferSectionHeader) == 16);
static_assert(sizeof(MeshBinaryMorphMetaRecord) == 16);
} // namespace nuri
