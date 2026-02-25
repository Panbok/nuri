#include "nuri/pch.h"

#include "nuri/resources/storage/font/nfont_binary_codec.h"

#include "nuri/resources/storage/font/nfont_binary_format.h"

namespace nuri {
namespace {

template <typename... Args>
[[nodiscard]] Result<std::vector<std::byte>, std::string>
makeSerializeError(Args &&...args) {
  std::ostringstream oss;
  (oss << ... << std::forward<Args>(args));
  return Result<std::vector<std::byte>, std::string>::makeError(oss.str());
}

template <typename... Args>
[[nodiscard]] Result<NFontBinaryData, std::string>
makeDeserializeError(Args &&...args) {
  std::ostringstream oss;
  (oss << ... << std::forward<Args>(args));
  return Result<NFontBinaryData, std::string>::makeError(oss.str());
}

[[nodiscard]] bool checkedAddToU64(uint64_t a, uint64_t b, uint64_t &out) {
  if (a > (std::numeric_limits<uint64_t>::max() - b)) {
    return false;
  }
  out = a + b;
  return true;
}

[[nodiscard]] bool checkedMulToU64(uint64_t a, uint64_t b, uint64_t &out) {
  if (a == 0 || b == 0) {
    out = 0;
    return true;
  }
  if (a > (std::numeric_limits<uint64_t>::max() / b)) {
    return false;
  }
  out = a * b;
  return true;
}

template <typename T> void appendPod(std::vector<std::byte> &out, const T &v) {
  static_assert(std::is_trivially_copyable_v<T>);
  const size_t offset = out.size();
  out.resize(offset + sizeof(T));
  std::memcpy(out.data() + offset, &v, sizeof(T));
}

template <typename T>
[[nodiscard]] bool appendPodArray(std::vector<std::byte> &out,
                                  std::span<const T> values) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (values.empty()) {
    return true;
  }

  uint64_t byteCount = 0;
  if (!checkedMulToU64(static_cast<uint64_t>(values.size()), sizeof(T),
                       byteCount)) {
    return false;
  }
  const uint64_t oldSize = static_cast<uint64_t>(out.size());
  uint64_t newSize = 0;
  if (!checkedAddToU64(oldSize, byteCount, newSize) ||
      newSize > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return false;
  }

  const size_t offset = out.size();
  out.resize(static_cast<size_t>(newSize));
  std::memcpy(out.data() + offset, values.data(), static_cast<size_t>(byteCount));
  return true;
}

template <typename T>
[[nodiscard]] bool readPod(std::span<const std::byte> bytes, uint64_t offset,
                           T &out) {
  static_assert(std::is_trivially_copyable_v<T>);
  uint64_t end = 0;
  if (!checkedAddToU64(offset, sizeof(T), end) ||
      end > static_cast<uint64_t>(bytes.size())) {
    return false;
  }
  std::memcpy(&out, bytes.data() + offset, sizeof(T));
  return true;
}

template <typename T>
[[nodiscard]] bool readPodArray(std::span<const std::byte> bytes,
                                uint64_t offset, uint32_t count,
                                std::vector<T> &out) {
  static_assert(std::is_trivially_copyable_v<T>);
  uint64_t totalBytes = 0;
  if (!checkedMulToU64(static_cast<uint64_t>(count), sizeof(T), totalBytes)) {
    return false;
  }
  uint64_t end = 0;
  if (!checkedAddToU64(offset, totalBytes, end) ||
      end > static_cast<uint64_t>(bytes.size())) {
    return false;
  }
  out.resize(count);
  if (totalBytes > 0) {
    std::memcpy(out.data(), bytes.data() + offset, static_cast<size_t>(totalBytes));
  }
  return true;
}

[[nodiscard]] bool isLittleEndianHost() {
#if defined(__cpp_lib_endian) && (__cpp_lib_endian >= 201907L)
  return std::endian::native == std::endian::little;
#else
  constexpr uint16_t kValue = 0x1;
  return *reinterpret_cast<const uint8_t *>(&kValue) == 0x1;
#endif
}

struct SerializedSection {
  uint32_t fourcc = 0;
  uint32_t flags = 0;
  uint32_t count = 0;
  uint32_t stride = 0;
  std::vector<std::byte> payload;
};

[[nodiscard]] bool validateSectionBounds(const NFontBinarySectionTocEntry &entry,
                                         size_t fileSize) {
  uint64_t end = 0;
  if (!checkedAddToU64(entry.offset, entry.sizeBytes, end)) {
    return false;
  }
  return end <= static_cast<uint64_t>(fileSize);
}

[[nodiscard]] bool
sectionSizeMatchesCountStride(const NFontBinarySectionTocEntry &entry) {
  uint64_t expected = 0;
  if (!checkedMulToU64(entry.count, entry.stride, expected)) {
    return false;
  }
  return expected == entry.sizeBytes;
}

[[nodiscard]] const NFontBinarySectionTocEntry *
findSection(std::span<const NFontBinarySectionTocEntry> toc, uint32_t fourcc) {
  const NFontBinarySectionTocEntry *result = nullptr;
  for (const NFontBinarySectionTocEntry &entry : toc) {
    if (entry.fourcc != fourcc) {
      continue;
    }
    if (result != nullptr) {
      return nullptr;
    }
    result = &entry;
  }
  return result;
}

} // namespace

Result<std::vector<std::byte>, std::string>
nfontBinarySerialize(const NFontBinaryData &input) {
  if (!isLittleEndianHost()) {
    return makeSerializeError("nfontBinarySerialize: unsupported host endianness");
  }

  if (input.glyphs.size() > std::numeric_limits<uint32_t>::max()) {
    return makeSerializeError("nfontBinarySerialize: glyph count exceeds uint32");
  }
  if (input.cmap.size() > std::numeric_limits<uint32_t>::max()) {
    return makeSerializeError("nfontBinarySerialize: cmap count exceeds uint32");
  }
  if (input.atlasPages.size() > std::numeric_limits<uint32_t>::max()) {
    return makeSerializeError(
        "nfontBinarySerialize: atlas page count exceeds uint32");
  }

  for (const NFontBinaryAtlasImage &page : input.atlasPages) {
    if (page.width == 0 || page.height == 0) {
      return makeSerializeError(
          "nfontBinarySerialize: atlas page has invalid dimensions");
    }
    if (page.imageBytes.empty()) {
      return makeSerializeError(
          "nfontBinarySerialize: atlas page has empty image payload");
    }

    uint64_t pixelCount = 0;
    if (!checkedMulToU64(static_cast<uint64_t>(page.width),
                         static_cast<uint64_t>(page.height), pixelCount) ||
        pixelCount == 0) {
      return makeSerializeError(
          "nfontBinarySerialize: atlas page pixel count overflow");
    }
    if ((static_cast<uint64_t>(page.imageBytes.size()) % pixelCount) != 0u) {
      return makeSerializeError(
          "nfontBinarySerialize: atlas image size is not divisible by pixel count");
    }
    const uint64_t bytesPerPixel =
        static_cast<uint64_t>(page.imageBytes.size()) / pixelCount;
    if (bytesPerPixel != 4u && bytesPerPixel != 8u) {
      return makeSerializeError(
          "nfontBinarySerialize: unsupported atlas bytes-per-pixel ",
          bytesPerPixel, " (expected 4 or 8)");
    }
  }

  SerializedSection headSection{};
  headSection.fourcc = kNFontBinarySectionHead;
  headSection.count = 1;
  headSection.stride = sizeof(NFontBinaryHeadRecord);
  NFontBinaryHeadRecord headRecord{};
  headRecord.glyphCount = static_cast<uint32_t>(input.glyphs.size());
  headRecord.cmapCount = static_cast<uint32_t>(input.cmap.size());
  headRecord.atlasPageCount = static_cast<uint32_t>(input.atlasPages.size());
  headRecord.pxRange = input.pxRange;
  appendPod(headSection.payload, headRecord);

  SerializedSection metricsSection{};
  metricsSection.fourcc = kNFontBinarySectionMetr;
  metricsSection.count = 1;
  metricsSection.stride = sizeof(NFontBinaryMetricsRecord);
  NFontBinaryMetricsRecord metricsRecord{};
  metricsRecord.ascent = input.metrics.ascent;
  metricsRecord.descent = input.metrics.descent;
  metricsRecord.lineGap = input.metrics.lineGap;
  metricsRecord.unitsPerEm = input.metrics.unitsPerEm;
  appendPod(metricsSection.payload, metricsRecord);

  SerializedSection cmapSection{};
  cmapSection.fourcc = kNFontBinarySectionCmap;
  cmapSection.count = static_cast<uint32_t>(input.cmap.size());
  cmapSection.stride = sizeof(NFontBinaryCmapRecord);
  cmapSection.payload.reserve(input.cmap.size() * sizeof(NFontBinaryCmapRecord));
  for (const NFontBinaryCmapEntry &entry : input.cmap) {
    NFontBinaryCmapRecord record{};
    record.codepoint = entry.codepoint;
    record.glyphId = entry.glyphId;
    appendPod(cmapSection.payload, record);
  }

  SerializedSection glyphSection{};
  glyphSection.fourcc = kNFontBinarySectionGlyp;
  glyphSection.count = static_cast<uint32_t>(input.glyphs.size());
  glyphSection.stride = sizeof(NFontBinaryGlyphRecord);
  glyphSection.payload.reserve(input.glyphs.size() * sizeof(NFontBinaryGlyphRecord));
  for (const GlyphMetrics &glyph : input.glyphs) {
    NFontBinaryGlyphRecord record{};
    record.glyphId = glyph.glyphId;
    record.localPageIndex = glyph.localPageIndex;
    record.advance = glyph.advance;
    record.bearingX = glyph.bearingX;
    record.bearingY = glyph.bearingY;
    record.planeMinX = glyph.planeMinX;
    record.planeMinY = glyph.planeMinY;
    record.planeMaxX = glyph.planeMaxX;
    record.planeMaxY = glyph.planeMaxY;
    record.uvMinX = glyph.uvMinX;
    record.uvMinY = glyph.uvMinY;
    record.uvMaxX = glyph.uvMaxX;
    record.uvMaxY = glyph.uvMaxY;
    appendPod(glyphSection.payload, record);
  }

  SerializedSection atlasSection{};
  atlasSection.fourcc = kNFontBinarySectionAtls;
  atlasSection.count = static_cast<uint32_t>(input.atlasPages.size());
  atlasSection.stride = sizeof(NFontBinaryAtlasPageRecord);

  SerializedSection imageSection{};
  imageSection.fourcc = kNFontBinarySectionImag;
  imageSection.count = 0;
  imageSection.stride = 1;

  for (const NFontBinaryAtlasImage &page : input.atlasPages) {
    if (page.imageBytes.size() > std::numeric_limits<uint32_t>::max()) {
      return makeSerializeError(
          "nfontBinarySerialize: atlas page image exceeds uint32 size");
    }
    if (imageSection.payload.size() > std::numeric_limits<uint32_t>::max()) {
      return makeSerializeError(
          "nfontBinarySerialize: concatenated image payload exceeds uint32");
    }

    NFontBinaryAtlasPageRecord record{};
    record.width = page.width;
    record.height = page.height;
    record.imageOffset = static_cast<uint32_t>(imageSection.payload.size());
    record.imageSize = static_cast<uint32_t>(page.imageBytes.size());
    appendPod(atlasSection.payload, record);

    if (!appendPodArray(imageSection.payload,
                        std::span<const std::byte>(page.imageBytes))) {
      return makeSerializeError(
          "nfontBinarySerialize: atlas image payload size overflow");
    }
  }
  if (imageSection.payload.size() > std::numeric_limits<uint32_t>::max()) {
    return makeSerializeError(
        "nfontBinarySerialize: concatenated image payload exceeds uint32");
  }
  imageSection.count = static_cast<uint32_t>(imageSection.payload.size());

  std::array<SerializedSection, 6> sections = {
      std::move(headSection), std::move(metricsSection), std::move(cmapSection),
      std::move(glyphSection), std::move(atlasSection), std::move(imageSection)};

  NFontBinaryHeader header{};
  header.magic = kNFontBinaryMagic;
  header.majorVersion = kNFontBinaryFormatMajorVersion;
  header.minorVersion = kNFontBinaryFormatMinorVersion;
  header.headerSize = sizeof(NFontBinaryHeader);
  header.tocEntrySize = sizeof(NFontBinarySectionTocEntry);
  header.flags = kNFontBinaryHeaderFlagLittleEndian;
  header.tocCount = static_cast<uint32_t>(sections.size());
  header.tocOffset = sizeof(NFontBinaryHeader);

  uint64_t tocBytes = 0;
  if (!checkedMulToU64(sizeof(NFontBinarySectionTocEntry), sections.size(),
                       tocBytes)) {
    return makeSerializeError("nfontBinarySerialize: TOC size overflow");
  }

  uint64_t cursor = 0;
  if (!checkedAddToU64(header.tocOffset, tocBytes, cursor)) {
    return makeSerializeError("nfontBinarySerialize: file layout overflow");
  }

  std::array<NFontBinarySectionTocEntry, 6> toc{};
  for (size_t i = 0; i < sections.size(); ++i) {
    NFontBinarySectionTocEntry entry{};
    entry.fourcc = sections[i].fourcc;
    entry.flags = sections[i].flags;
    entry.offset = cursor;
    entry.sizeBytes = sections[i].payload.size();
    entry.count = sections[i].count;
    entry.stride = sections[i].stride;
    toc[i] = entry;

    if (!checkedAddToU64(cursor, entry.sizeBytes, cursor)) {
      return makeSerializeError("nfontBinarySerialize: section layout overflow");
    }
  }

  header.fileSize = cursor;
  if (header.fileSize > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return makeSerializeError(
        "nfontBinarySerialize: output exceeds addressable size");
  }

  std::vector<std::byte> out(static_cast<size_t>(header.fileSize));
  std::memcpy(out.data(), &header, sizeof(header));
  std::memcpy(out.data() + header.tocOffset, toc.data(), static_cast<size_t>(tocBytes));
  for (size_t i = 0; i < sections.size(); ++i) {
    if (!sections[i].payload.empty()) {
      std::memcpy(out.data() + toc[i].offset, sections[i].payload.data(),
                  sections[i].payload.size());
    }
  }

  return Result<std::vector<std::byte>, std::string>::makeResult(std::move(out));
}

Result<NFontBinaryData, std::string>
nfontBinaryDeserialize(std::span<const std::byte> fileBytes) {
  if (!isLittleEndianHost()) {
    return makeDeserializeError(
        "nfontBinaryDeserialize: unsupported host endianness");
  }
  if (fileBytes.size() < sizeof(NFontBinaryHeader)) {
    return makeDeserializeError("nfontBinaryDeserialize: file too small");
  }

  NFontBinaryHeader header{};
  if (!readPod(fileBytes, 0, header)) {
    return makeDeserializeError("nfontBinaryDeserialize: failed to read header");
  }

  if (header.magic != kNFontBinaryMagic) {
    return makeDeserializeError("nfontBinaryDeserialize: invalid magic");
  }
  if (header.majorVersion != kNFontBinaryFormatMajorVersion) {
    return makeDeserializeError(
        "nfontBinaryDeserialize: unsupported major version");
  }
  if ((header.flags & kNFontBinaryHeaderFlagLittleEndian) == 0) {
    return makeDeserializeError("nfontBinaryDeserialize: unsupported endianness");
  }
  if (header.headerSize != sizeof(NFontBinaryHeader) ||
      header.tocEntrySize != sizeof(NFontBinarySectionTocEntry)) {
    return makeDeserializeError(
        "nfontBinaryDeserialize: unexpected header/toc entry size");
  }
  if (header.fileSize != fileBytes.size()) {
    return makeDeserializeError("nfontBinaryDeserialize: file size mismatch");
  }
  if (header.tocCount == 0) {
    return makeDeserializeError("nfontBinaryDeserialize: TOC is empty");
  }
  if (header.tocCount > 1024u) {
    return makeDeserializeError("nfontBinaryDeserialize: TOC count is unreasonable");
  }

  uint64_t tocBytes = 0;
  if (!checkedMulToU64(header.tocCount, sizeof(NFontBinarySectionTocEntry),
                       tocBytes)) {
    return makeDeserializeError("nfontBinaryDeserialize: TOC size overflow");
  }
  uint64_t tocEnd = 0;
  if (!checkedAddToU64(header.tocOffset, tocBytes, tocEnd) ||
      tocEnd > fileBytes.size()) {
    return makeDeserializeError("nfontBinaryDeserialize: invalid TOC range");
  }

  std::vector<NFontBinarySectionTocEntry> toc;
  if (!readPodArray(fileBytes, header.tocOffset, header.tocCount, toc)) {
    return makeDeserializeError("nfontBinaryDeserialize: failed to read TOC");
  }
  for (const NFontBinarySectionTocEntry &entry : toc) {
    if (!validateSectionBounds(entry, fileBytes.size())) {
      return makeDeserializeError(
          "nfontBinaryDeserialize: section out of file bounds");
    }
  }

  const NFontBinarySectionTocEntry *headEntry =
      findSection(toc, kNFontBinarySectionHead);
  const NFontBinarySectionTocEntry *metrEntry =
      findSection(toc, kNFontBinarySectionMetr);
  const NFontBinarySectionTocEntry *cmapEntry =
      findSection(toc, kNFontBinarySectionCmap);
  const NFontBinarySectionTocEntry *glypEntry =
      findSection(toc, kNFontBinarySectionGlyp);
  const NFontBinarySectionTocEntry *atlsEntry =
      findSection(toc, kNFontBinarySectionAtls);
  const NFontBinarySectionTocEntry *imagEntry =
      findSection(toc, kNFontBinarySectionImag);
  if (!headEntry || !metrEntry || !cmapEntry || !glypEntry || !atlsEntry ||
      !imagEntry) {
    return makeDeserializeError(
        "nfontBinaryDeserialize: missing or duplicate required sections");
  }

  if (headEntry->count != 1 || headEntry->stride != sizeof(NFontBinaryHeadRecord) ||
      !sectionSizeMatchesCountStride(*headEntry)) {
    return makeDeserializeError(
        "nfontBinaryDeserialize: invalid HEAD section metadata");
  }
  if (metrEntry->count != 1 ||
      metrEntry->stride != sizeof(NFontBinaryMetricsRecord) ||
      !sectionSizeMatchesCountStride(*metrEntry)) {
    return makeDeserializeError(
        "nfontBinaryDeserialize: invalid METR section metadata");
  }
  if (cmapEntry->stride != sizeof(NFontBinaryCmapRecord) ||
      !sectionSizeMatchesCountStride(*cmapEntry)) {
    return makeDeserializeError(
        "nfontBinaryDeserialize: invalid CMAP section metadata");
  }
  if (glypEntry->stride != sizeof(NFontBinaryGlyphRecord) ||
      !sectionSizeMatchesCountStride(*glypEntry)) {
    return makeDeserializeError(
        "nfontBinaryDeserialize: invalid GLYP section metadata");
  }
  if (atlsEntry->stride != sizeof(NFontBinaryAtlasPageRecord) ||
      !sectionSizeMatchesCountStride(*atlsEntry)) {
    return makeDeserializeError(
        "nfontBinaryDeserialize: invalid ATLS section metadata");
  }
  if (imagEntry->stride != 1 ||
      !sectionSizeMatchesCountStride(*imagEntry)) {
    return makeDeserializeError(
        "nfontBinaryDeserialize: invalid IMAG section metadata");
  }

  NFontBinaryHeadRecord head{};
  if (!readPod(fileBytes, headEntry->offset, head)) {
    return makeDeserializeError("nfontBinaryDeserialize: failed reading HEAD");
  }
  NFontBinaryMetricsRecord metr{};
  if (!readPod(fileBytes, metrEntry->offset, metr)) {
    return makeDeserializeError("nfontBinaryDeserialize: failed reading METR");
  }

  if (head.glyphCount != glypEntry->count || head.cmapCount != cmapEntry->count ||
      head.atlasPageCount != atlsEntry->count) {
    return makeDeserializeError(
        "nfontBinaryDeserialize: HEAD counts do not match table counts");
  }
  if (head.atlasPageCount == 0 && head.glyphCount > 0) {
    return makeDeserializeError(
        "nfontBinaryDeserialize: glyph table is present without atlas pages");
  }

  std::vector<NFontBinaryCmapRecord> cmapRecords;
  if (!readPodArray(fileBytes, cmapEntry->offset, cmapEntry->count, cmapRecords)) {
    return makeDeserializeError("nfontBinaryDeserialize: failed reading CMAP");
  }

  std::vector<NFontBinaryGlyphRecord> glyphRecords;
  if (!readPodArray(fileBytes, glypEntry->offset, glypEntry->count, glyphRecords)) {
    return makeDeserializeError("nfontBinaryDeserialize: failed reading GLYP");
  }

  std::vector<NFontBinaryAtlasPageRecord> atlasRecords;
  if (!readPodArray(fileBytes, atlsEntry->offset, atlsEntry->count, atlasRecords)) {
    return makeDeserializeError("nfontBinaryDeserialize: failed reading ATLS");
  }

  const std::span<const std::byte> imagePayload(
      fileBytes.data() + imagEntry->offset, static_cast<size_t>(imagEntry->sizeBytes));

  NFontBinaryData out{};
  out.pxRange = head.pxRange;
  out.metrics = FontMetrics{
      .ascent = metr.ascent,
      .descent = metr.descent,
      .lineGap = metr.lineGap,
      .unitsPerEm = metr.unitsPerEm,
  };
  out.cmap.reserve(cmapRecords.size());
  for (const NFontBinaryCmapRecord &record : cmapRecords) {
    out.cmap.push_back(NFontBinaryCmapEntry{
        .codepoint = record.codepoint,
        .glyphId = record.glyphId,
    });
  }

  out.glyphs.reserve(glyphRecords.size());
  for (const NFontBinaryGlyphRecord &record : glyphRecords) {
    if (record.localPageIndex >= head.atlasPageCount) {
      return makeDeserializeError(
          "nfontBinaryDeserialize: glyph localPageIndex is out of range");
    }
    out.glyphs.push_back(GlyphMetrics{
        .glyphId = record.glyphId,
        .localPageIndex = record.localPageIndex,
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
    });
  }

  out.atlasPages.reserve(atlasRecords.size());
  for (const NFontBinaryAtlasPageRecord &record : atlasRecords) {
    if (record.width == 0 || record.height == 0) {
      return makeDeserializeError(
          "nfontBinaryDeserialize: atlas page has invalid dimensions");
    }
    if (record.imageSize == 0) {
      return makeDeserializeError(
          "nfontBinaryDeserialize: atlas page has empty image payload");
    }

    uint64_t imageEnd = 0;
    if (!checkedAddToU64(record.imageOffset, record.imageSize, imageEnd) ||
        imageEnd > imagePayload.size()) {
      return makeDeserializeError(
          "nfontBinaryDeserialize: atlas image range is out of bounds");
    }

    uint64_t pixelCount = 0;
    if (!checkedMulToU64(static_cast<uint64_t>(record.width),
                         static_cast<uint64_t>(record.height), pixelCount) ||
        pixelCount == 0) {
      return makeDeserializeError(
          "nfontBinaryDeserialize: atlas page pixel count overflow");
    }
    if ((static_cast<uint64_t>(record.imageSize) % pixelCount) != 0u) {
      return makeDeserializeError(
          "nfontBinaryDeserialize: atlas image size is not divisible by pixel count");
    }
    const uint64_t bytesPerPixel =
        static_cast<uint64_t>(record.imageSize) / pixelCount;
    if (bytesPerPixel != 4u && bytesPerPixel != 8u) {
      return makeDeserializeError(
          "nfontBinaryDeserialize: unsupported atlas bytes-per-pixel");
    }

    NFontBinaryAtlasImage page{};
    page.width = record.width;
    page.height = record.height;
    const auto *imgSrc = imagePayload.data() + record.imageOffset;
    page.imageBytes.assign(imgSrc, imgSrc + record.imageSize);
    out.atlasPages.push_back(std::move(page));
  }

  return Result<NFontBinaryData, std::string>::makeResult(std::move(out));
}

} // namespace nuri
