#pragma once

#include <array>
#include <cstdint>
#include <type_traits>

namespace nuri {

constexpr uint16_t kNFontBinaryFormatMajorVersion = 1;
constexpr uint16_t kNFontBinaryFormatMinorVersion = 0;

constexpr std::array<char, 8> kNFontBinaryMagic = {'N', 'U', 'R', 'I',
                                                   'F', 'O', 'N', 'T'};

constexpr uint32_t kNFontBinaryHeaderFlagLittleEndian = 1u << 0u;

constexpr uint32_t makeNFontBinaryFourCC(char a, char b, char c, char d) {
  return static_cast<uint32_t>(static_cast<uint8_t>(a)) |
         (static_cast<uint32_t>(static_cast<uint8_t>(b)) << 8u) |
         (static_cast<uint32_t>(static_cast<uint8_t>(c)) << 16u) |
         (static_cast<uint32_t>(static_cast<uint8_t>(d)) << 24u);
}

constexpr uint32_t kNFontBinarySectionHead =
    makeNFontBinaryFourCC('H', 'E', 'A', 'D');
constexpr uint32_t kNFontBinarySectionMetr =
    makeNFontBinaryFourCC('M', 'E', 'T', 'R');
constexpr uint32_t kNFontBinarySectionCmap =
    makeNFontBinaryFourCC('C', 'M', 'A', 'P');
constexpr uint32_t kNFontBinarySectionGlyp =
    makeNFontBinaryFourCC('G', 'L', 'Y', 'P');
constexpr uint32_t kNFontBinarySectionAtls =
    makeNFontBinaryFourCC('A', 'T', 'L', 'S');
constexpr uint32_t kNFontBinarySectionImag =
    makeNFontBinaryFourCC('I', 'M', 'A', 'G');

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

static_assert(sizeof(NFontBinaryHeader) == 44);
static_assert(sizeof(NFontBinarySectionTocEntry) == 32);
static_assert(sizeof(NFontBinaryHeadRecord) == 32);
static_assert(sizeof(NFontBinaryMetricsRecord) == 16);
static_assert(sizeof(NFontBinaryCmapRecord) == 8);
static_assert(sizeof(NFontBinaryGlyphRecord) == 52);
static_assert(sizeof(NFontBinaryAtlasPageRecord) == 16);
static_assert(std::is_standard_layout_v<NFontBinaryHeader>);
static_assert(std::is_standard_layout_v<NFontBinarySectionTocEntry>);
static_assert(std::is_standard_layout_v<NFontBinaryHeadRecord>);
static_assert(std::is_standard_layout_v<NFontBinaryMetricsRecord>);
static_assert(std::is_standard_layout_v<NFontBinaryCmapRecord>);
static_assert(std::is_standard_layout_v<NFontBinaryGlyphRecord>);
static_assert(std::is_standard_layout_v<NFontBinaryAtlasPageRecord>);
static_assert(std::is_trivially_copyable_v<NFontBinaryHeader>);
static_assert(std::is_trivially_copyable_v<NFontBinarySectionTocEntry>);
static_assert(std::is_trivially_copyable_v<NFontBinaryHeadRecord>);
static_assert(std::is_trivially_copyable_v<NFontBinaryMetricsRecord>);
static_assert(std::is_trivially_copyable_v<NFontBinaryCmapRecord>);
static_assert(std::is_trivially_copyable_v<NFontBinaryGlyphRecord>);
static_assert(std::is_trivially_copyable_v<NFontBinaryAtlasPageRecord>);

} // namespace nuri
