#pragma once
#include "nuri/resources/storage/binary_format.h"
#include <array>
#include <cstdint>
namespace nuri {

constexpr uint16_t kNFontBinaryFormatMajorVersion = 1;
constexpr uint16_t kNFontBinaryFormatMinorVersion = 0;

constexpr std::array<char, 8> kNFontBinaryMagic = {'N', 'U', 'R', 'I',
                                                   'F', 'O', 'N', 'T'};

constexpr uint32_t kNFontBinarySectionHead =
    makeBinaryFourCC('H', 'E', 'A', 'D');
constexpr uint32_t kNFontBinarySectionMetr =
    makeBinaryFourCC('M', 'E', 'T', 'R');
constexpr uint32_t kNFontBinarySectionCmap =
    makeBinaryFourCC('C', 'M', 'A', 'P');
constexpr uint32_t kNFontBinarySectionGlyp =
    makeBinaryFourCC('G', 'L', 'Y', 'P');
constexpr uint32_t kNFontBinarySectionAtls =
    makeBinaryFourCC('A', 'T', 'L', 'S');
constexpr uint32_t kNFontBinarySectionImag =
    makeBinaryFourCC('I', 'M', 'A', 'G');

#pragma pack(push, 1)
struct NFontBinaryHeader {
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
};
#pragma pack(pop)

#pragma pack(push, 1)
struct NFontBinarySectionTocEntry {
  uint32_t fourcc = 0;
  uint32_t flags = 0;
  uint64_t offset = 0;
  uint64_t sizeBytes = 0;
  uint32_t count = 0;
  uint32_t stride = 0;
};

struct NFontBinaryHeadRecord {
  uint32_t glyphCount = 0;
  uint32_t cmapCount = 0;
  uint32_t atlasPageCount = 0;
  uint32_t reserved0 = 0;
  float pxRange = 4.0f;
  uint32_t reserved1[3] = {0, 0, 0};
};

struct NFontBinaryMetricsRecord {
  float ascent = 0.0f;
  float descent = 0.0f;
  float lineGap = 0.0f;
  float unitsPerEm = 1.0f;
};

struct NFontBinaryCmapRecord {
  uint32_t codepoint = 0;
  uint32_t glyphId = 0;
};

struct NFontBinaryGlyphRecord {
  uint32_t glyphId = 0;
  uint16_t localPageIndex = 0;
  uint16_t reserved0 = 0;
  float advance = 0.0f;
  float bearingX = 0.0f;
  float bearingY = 0.0f;
  float planeMinX = 0.0f;
  float planeMinY = 0.0f;
  float planeMaxX = 0.0f;
  float planeMaxY = 0.0f;
  float uvMinX = 0.0f;
  float uvMinY = 0.0f;
  float uvMaxX = 0.0f;
  float uvMaxY = 0.0f;
};

struct NFontBinaryAtlasPageRecord {
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t imageOffset = 0;
  uint32_t imageSize = 0;
};
#pragma pack(pop)

static_assert(sizeof(NFontBinaryHeader) == 44);
static_assert(sizeof(NFontBinarySectionTocEntry) == 32);
static_assert(sizeof(NFontBinaryHeadRecord) == 32);
static_assert(sizeof(NFontBinaryMetricsRecord) == 16);
static_assert(sizeof(NFontBinaryCmapRecord) == 8);
static_assert(sizeof(NFontBinaryGlyphRecord) == 52);
static_assert(sizeof(NFontBinaryAtlasPageRecord) == 16);
} // namespace nuri
