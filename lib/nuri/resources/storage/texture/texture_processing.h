#pragma once

#include <cstdint>

namespace nuri {

enum class TextureMipSemantic : uint8_t {
  Generic = 0,
  AlphaCoverage = 1,
  NormalMap = 2,
  RoughnessG = 3,
  RoughnessA = 4,
};

struct TextureLoadOptions {
  bool srgb = false;
  bool generateMipmaps = false;
  TextureMipSemantic mipSemantic = TextureMipSemantic::Generic;
  float alphaCoverageCutoff = 0.5f;
};

} // namespace nuri
