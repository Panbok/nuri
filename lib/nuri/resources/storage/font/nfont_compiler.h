#pragma once

#include "nuri/core/result.h"

#include <cstdint>
#include <filesystem>
#include <string>

namespace nuri {

enum class NFontCharsetPreset : uint8_t {
  Ascii,
  Latin1,
};

struct NFontCompileConfig {
  std::filesystem::path sourceFontPath;
  std::filesystem::path outputFontPath;
  NFontCharsetPreset charset = NFontCharsetPreset::Ascii;
  float minimumEmSize = 64.0f;
  float pxRange = 8.0f;
  float outerPixelPadding = 2.0f;
  uint32_t atlasSpacing = 1;
  bool useRgba16fAtlas = true;
  uint32_t maxAtlasWidth = 0;
  uint32_t maxAtlasHeight = 0;
  uint32_t threadCount = 0;
};

struct NFontCompileReport {
  std::filesystem::path outputPath;
  uint32_t glyphCount = 0;
  uint32_t atlasWidth = 0;
  uint32_t atlasHeight = 0;
  size_t bytesWritten = 0;
};

[[nodiscard]] Result<NFontCompileReport, std::string>
compileNFontFromFontFile(const NFontCompileConfig &config);

} // namespace nuri
