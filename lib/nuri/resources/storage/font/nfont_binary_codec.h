#pragma once

#include "nuri/core/result.h"
#include "nuri/text/font_types.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace nuri {

struct NFontBinaryCmapEntry {
  uint32_t codepoint = 0;
  GlyphId glyphId = 0;
};

struct NFontBinaryAtlasImage {
  uint32_t width = 0;
  uint32_t height = 0;
  std::vector<std::byte> imageBytes;
};

struct NFontBinaryData {
  float pxRange = 4.0f;
  FontMetrics metrics{};
  std::vector<NFontBinaryCmapEntry> cmap;
  std::vector<GlyphMetrics> glyphs;
  std::vector<NFontBinaryAtlasImage> atlasPages;
};

[[nodiscard]] Result<std::vector<std::byte>, std::string>
nfontBinarySerialize(const NFontBinaryData &input);

[[nodiscard]] Result<NFontBinaryData, std::string>
nfontBinaryDeserialize(std::span<const std::byte> fileBytes);

} // namespace nuri
