#include "nuri/resources/storage/font/nfont_binary_codec.h"
#include "nuri/resources/storage/binary_format.h"
#include "nuri/resources/storage/binary_io.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <ranges>
#include <utility>
namespace nuri {
namespace {
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

struct Section {
  uint32_t fourcc = 0u;
  uint32_t count = 0u;
  uint32_t stride = 0u;
  std::vector<std::byte> payload{};
};
struct RequiredSection {
  uint32_t fourcc;
  uint32_t stride;
  uint32_t exactCount;
  const NFontBinarySectionTocEntry *entry = nullptr;
};
constexpr uint32_t kAnyCount = std::numeric_limits<uint32_t>::max();
[[nodiscard]] Result<std::vector<std::byte>, std::string>
serializeError(std::string message) {
  return Result<std::vector<std::byte>, std::string>::makeError(
      std::move(message));
}
[[nodiscard]] Result<NFontBinaryData, std::string>
deserializeError(std::string message) {
  return Result<NFontBinaryData, std::string>::makeError(std::move(message));
}
[[nodiscard]] bool validAtlasPage(const NFontBinaryAtlasImage &page) {
  const uint64_t pixels = static_cast<uint64_t>(page.width) * page.height;
  if (pixels == 0u || page.imageBytes.empty() ||
      page.imageBytes.size() % pixels != 0u) {
    return false;
  }
  const uint64_t bytesPerPixel = page.imageBytes.size() / pixels;
  return bytesPerPixel == 4u || bytesPerPixel == 8u;
}
} // namespace

Result<std::vector<std::byte>, std::string>
nfontBinarySerialize(const NFontBinaryData &input) {
  if (input.glyphs.size() > std::numeric_limits<uint32_t>::max() ||
      input.cmap.size() > std::numeric_limits<uint32_t>::max() ||
      input.atlasPages.size() > std::numeric_limits<uint32_t>::max()) {
    return serializeError("nfontBinarySerialize: table count exceeds uint32");
  }
  for (const NFontBinaryAtlasImage &page : input.atlasPages) {
    if (!validAtlasPage(page)) {
      return serializeError("nfontBinarySerialize: invalid atlas page");
    }
  }
  Section head{kNFontBinarySectionHead, 1u, sizeof(NFontBinaryHeadRecord)};
  appendPod(
      head.payload,
      NFontBinaryHeadRecord{
          .glyphCount = static_cast<uint32_t>(input.glyphs.size()),
          .cmapCount = static_cast<uint32_t>(input.cmap.size()),
          .atlasPageCount = static_cast<uint32_t>(input.atlasPages.size()),
          .pxRange = input.pxRange,
      });
  Section metrics{kNFontBinarySectionMetr, 1u,
                  sizeof(NFontBinaryMetricsRecord)};
  appendPod(metrics.payload, NFontBinaryMetricsRecord{
                                 .ascent = input.metrics.ascent,
                                 .descent = input.metrics.descent,
                                 .lineGap = input.metrics.lineGap,
                                 .unitsPerEm = input.metrics.unitsPerEm,
                             });
  Section cmap{kNFontBinarySectionCmap,
               static_cast<uint32_t>(input.cmap.size()),
               sizeof(NFontBinaryCmapRecord)};
  cmap.payload.reserve(input.cmap.size() * sizeof(NFontBinaryCmapRecord));
  for (const NFontBinaryCmapEntry &entry : input.cmap) {
    appendPod(cmap.payload, NFontBinaryCmapRecord{
                                .codepoint = entry.codepoint,
                                .glyphId = entry.glyphId,
                            });
  }
  Section glyphs{kNFontBinarySectionGlyp,
                 static_cast<uint32_t>(input.glyphs.size()),
                 sizeof(NFontBinaryGlyphRecord)};
  glyphs.payload.reserve(input.glyphs.size() * sizeof(NFontBinaryGlyphRecord));
  for (const GlyphMetrics &glyph : input.glyphs) {
    appendPod(glyphs.payload, NFontBinaryGlyphRecord{
                                  .glyphId = glyph.glyphId,
                                  .localPageIndex = glyph.localPageIndex,
                                  .advance = glyph.advance,
                                  .bearingX = glyph.bearingX,
                                  .bearingY = glyph.bearingY,
                                  .planeMinX = glyph.planeMinX,
                                  .planeMinY = glyph.planeMinY,
                                  .planeMaxX = glyph.planeMaxX,
                                  .planeMaxY = glyph.planeMaxY,
                                  .uvMinX = glyph.uvMinX,
                                  .uvMinY = glyph.uvMinY,
                                  .uvMaxX = glyph.uvMaxX,
                                  .uvMaxY = glyph.uvMaxY,
                              });
  }
  Section atlas{kNFontBinarySectionAtls,
                static_cast<uint32_t>(input.atlasPages.size()),
                sizeof(NFontBinaryAtlasPageRecord)};
  Section images{kNFontBinarySectionImag, 0u, 1u};
  for (const NFontBinaryAtlasImage &page : input.atlasPages) {
    if (page.imageBytes.size() > std::numeric_limits<uint32_t>::max() ||
        images.payload.size() >
            std::numeric_limits<uint32_t>::max() - page.imageBytes.size()) {
      return serializeError("nfontBinarySerialize: atlas payload exceeds 4GB");
    }
    appendPod(atlas.payload,
              NFontBinaryAtlasPageRecord{
                  .width = page.width,
                  .height = page.height,
                  .imageOffset = static_cast<uint32_t>(images.payload.size()),
                  .imageSize = static_cast<uint32_t>(page.imageBytes.size()),
              });
    appendPodArray(images.payload, std::span<const std::byte>(page.imageBytes));
  }
  images.count = static_cast<uint32_t>(images.payload.size());
  std::array<Section, 6> sections{
      std::move(head),   std::move(metrics), std::move(cmap),
      std::move(glyphs), std::move(atlas),   std::move(images),
  };
  NFontBinaryHeader header{
      .magic = kNFontBinaryMagic,
      .majorVersion = kNFontBinaryFormatMajorVersion,
      .minorVersion = kNFontBinaryFormatMinorVersion,
      .headerSize = sizeof(NFontBinaryHeader),
      .tocEntrySize = sizeof(NFontBinarySectionTocEntry),
      .flags = kBinaryHeaderFlagLittleEndian,
      .tocOffset = sizeof(NFontBinaryHeader),
      .tocCount = static_cast<uint32_t>(sections.size()),
  };
  std::array<BinarySection, 6> sectionViews{};
  for (size_t i = 0; i < sections.size(); ++i) {
    sectionViews[i] = {
        .fourcc = sections[i].fourcc,
        .count = sections[i].count,
        .stride = sections[i].stride,
        .payload = sections[i].payload,
    };
  }
  return writeSectionedBinary<NFontBinaryHeader, NFontBinarySectionTocEntry>(
      header, sectionViews, 1u, "nfontBinarySerialize");
}

Result<NFontBinaryData, std::string>
nfontBinaryDeserialize(std::span<const std::byte> fileBytes) {
  if (fileBytes.size() < sizeof(NFontBinaryHeader)) {
    return deserializeError("nfontBinaryDeserialize: file too small");
  }
  const NFontBinaryHeader header = readPodAt<NFontBinaryHeader>(fileBytes, 0u);
  if (header.magic != kNFontBinaryMagic ||
      header.majorVersion != kNFontBinaryFormatMajorVersion ||
      (header.flags & kBinaryHeaderFlagLittleEndian) == 0u ||
      header.headerSize != sizeof(NFontBinaryHeader) ||
      header.tocEntrySize != sizeof(NFontBinarySectionTocEntry) ||
      header.fileSize != fileBytes.size() || header.tocCount == 0u ||
      header.tocCount > 1024u) {
    return deserializeError("nfontBinaryDeserialize: invalid header");
  }
  std::vector<NFontBinarySectionTocEntry> toc;
  auto tocResult = readSectionToc(fileBytes, header.tocOffset, header.tocCount,
                                  toc, "nfontBinaryDeserialize", 1024u);
  if (tocResult.hasError()) {
    return deserializeError(tocResult.error());
  }
  std::array<RequiredSection, 6> required{{
      {kNFontBinarySectionHead, sizeof(NFontBinaryHeadRecord), 1u},
      {kNFontBinarySectionMetr, sizeof(NFontBinaryMetricsRecord), 1u},
      {kNFontBinarySectionCmap, sizeof(NFontBinaryCmapRecord), kAnyCount},
      {kNFontBinarySectionGlyp, sizeof(NFontBinaryGlyphRecord), kAnyCount},
      {kNFontBinarySectionAtls, sizeof(NFontBinaryAtlasPageRecord), kAnyCount},
      {kNFontBinarySectionImag, 1u, kAnyCount},
  }};
  for (const NFontBinarySectionTocEntry &entry : toc) {
    const auto spec = std::find_if(required.begin(), required.end(),
                                   [&](const RequiredSection &value) {
                                     return value.fourcc == entry.fourcc;
                                   });
    if (spec == required.end()) {
      continue;
    }
    if (spec->entry != nullptr || entry.stride != spec->stride ||
        entry.sizeBytes != static_cast<uint64_t>(entry.count) * entry.stride ||
        (spec->exactCount != kAnyCount && entry.count != spec->exactCount)) {
      return deserializeError(
          "nfontBinaryDeserialize: invalid required section");
    }
    spec->entry = &entry;
  }
  if (std::ranges::any_of(required, [](const RequiredSection &spec) {
        return spec.entry == nullptr;
      })) {
    return deserializeError("nfontBinaryDeserialize: missing required section");
  }
  const auto &headEntry = *required[0].entry;
  const auto &metricsEntry = *required[1].entry;
  const auto &cmapEntry = *required[2].entry;
  const auto &glyphEntry = *required[3].entry;
  const auto &atlasEntry = *required[4].entry;
  const auto &imageEntry = *required[5].entry;
  const auto head =
      readPodAt<NFontBinaryHeadRecord>(fileBytes, headEntry.offset);
  const auto metrics =
      readPodAt<NFontBinaryMetricsRecord>(fileBytes, metricsEntry.offset);
  if (head.glyphCount != glyphEntry.count ||
      head.cmapCount != cmapEntry.count ||
      head.atlasPageCount != atlasEntry.count ||
      (head.atlasPageCount == 0u && head.glyphCount != 0u)) {
    return deserializeError("nfontBinaryDeserialize: inconsistent counts");
  }
  std::vector<NFontBinaryCmapRecord> cmapRecords;
  std::vector<NFontBinaryGlyphRecord> glyphRecords;
  std::vector<NFontBinaryAtlasPageRecord> atlasRecords;
  readPodArrayAt(fileBytes, cmapEntry.offset, cmapEntry.count, cmapRecords);
  readPodArrayAt(fileBytes, glyphEntry.offset, glyphEntry.count, glyphRecords);
  readPodArrayAt(fileBytes, atlasEntry.offset, atlasEntry.count, atlasRecords);
  const auto imagePayload =
      fileBytes.subspan(static_cast<size_t>(imageEntry.offset),
                        static_cast<size_t>(imageEntry.sizeBytes));
  NFontBinaryData out;
  out.pxRange = head.pxRange;
  out.metrics = {
      .ascent = metrics.ascent,
      .descent = metrics.descent,
      .lineGap = metrics.lineGap,
      .unitsPerEm = metrics.unitsPerEm,
  };
  out.cmap.reserve(cmapRecords.size());
  for (const NFontBinaryCmapRecord &record : cmapRecords) {
    out.cmap.push_back({record.codepoint, record.glyphId});
  }
  out.glyphs.reserve(glyphRecords.size());
  for (const NFontBinaryGlyphRecord &record : glyphRecords) {
    if (record.localPageIndex >= head.atlasPageCount) {
      return deserializeError(
          "nfontBinaryDeserialize: invalid glyph atlas page");
    }
    out.glyphs.push_back({
        .glyphId = record.glyphId,
        .advance = record.advance,
        .bearingX = record.bearingX,
        .bearingY = record.bearingY,
        .planeMinX = record.planeMinX,
        .planeMinY = record.planeMinY,
        .planeMaxX = record.planeMaxX,
        .planeMaxY = record.planeMaxY,
        .uvMinX = record.uvMinX,
        .uvMinY = record.uvMinY,
        .uvMaxX = record.uvMaxX,
        .uvMaxY = record.uvMaxY,
        .localPageIndex = record.localPageIndex,
    });
  }
  out.atlasPages.reserve(atlasRecords.size());
  for (const NFontBinaryAtlasPageRecord &record : atlasRecords) {
    const uint64_t pixels = static_cast<uint64_t>(record.width) * record.height;
    const bool validRange = binaryRangeValid(
        imagePayload.size(), record.imageOffset, record.imageSize);
    const uint64_t bytesPerPixel =
        pixels == 0u ? 0u : static_cast<uint64_t>(record.imageSize) / pixels;
    if (pixels == 0u || record.imageSize == 0u || !validRange ||
        record.imageSize % pixels != 0u ||
        (bytesPerPixel != 4u && bytesPerPixel != 8u)) {
      return deserializeError("nfontBinaryDeserialize: invalid atlas page");
    }
    const auto bytes =
        imagePayload.subspan(record.imageOffset, record.imageSize);
    out.atlasPages.push_back({
        .width = record.width,
        .height = record.height,
        .imageBytes = std::vector<std::byte>(bytes.begin(), bytes.end()),
    });
  }
  return Result<NFontBinaryData, std::string>::makeResult(std::move(out));
}

} // namespace nuri
