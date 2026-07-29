#pragma once
#include "nuri/defines.h"
#include "nuri/resources/gpu/resource_handles.h"
#include <array>
#include <cassert>
#include <cstdint>
#include <memory_resource>
#include <string>
#include <vector>
namespace nuri {

using FontHandle = PackedHandle<struct FontHandleTag>;
using AtlasPageHandle = PackedHandle<struct AtlasPageHandleTag>;
using ShaperFaceHandle = PackedHandle<struct ShaperFaceHandleTag>;

inline constexpr FontHandle kInvalidFontHandle{};
inline constexpr AtlasPageHandle kInvalidAtlasPageHandle{};
inline constexpr ShaperFaceHandle kInvalidShaperFaceHandle{};

using GlyphId = uint32_t;

enum class TextDirection : uint8_t { Ltr, Rtl, Auto };
enum class TextAlignH : uint8_t { Left, Center, Right };
enum class TextAlignV : uint8_t { Top, Middle, Baseline, Bottom };
enum class TextWrapMode : uint8_t { None, Word, Grapheme };
enum class TextBillboardMode : uint8_t { None, Spherical, CylindricalY };

struct FontMetrics {
  float ascent = 0.0f;
  float descent = 0.0f;
  float lineGap = 0.0f;
  float unitsPerEm = 1.0f;
};

struct GlyphMetrics {
  GlyphId glyphId = 0;
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
  uint16_t localPageIndex = 0;
};

struct GlyphLookupResult {
  const GlyphMetrics *metrics = nullptr;
  AtlasPageHandle atlasPage = kInvalidAtlasPageHandle;
};

struct FontLoadDesc {
  std::string path;
  std::string debugName;
  std::pmr::memory_resource *memory = std::pmr::get_default_resource();
};

struct FontFallbackChain {
  std::pmr::vector<FontHandle> handles;
};

struct TextStyle {
  FontHandle font = kInvalidFontHandle;
  float pxSize = 16.0f;
  float lineHeightScale = 1.0f;
  float letterSpacing = 0.0f;
  float wordSpacing = 0.0f;
  FontFallbackChain fallback;
};

struct TextLayoutParams {
  float maxWidthPx = 0.0f;
  float maxHeightPx = 0.0f;
  TextWrapMode wrapMode = TextWrapMode::None;
  TextDirection direction = TextDirection::Auto;
  TextAlignH alignH = TextAlignH::Left;
  TextAlignV alignV = TextAlignV::Baseline;
  bool enableKerning = true;
  bool enableLigatures = true;
  bool allowFallback = true;
};

struct TextColor {
  float r = 1.0f;
  float g = 1.0f;
  float b = 1.0f;
  float a = 1.0f;
};

struct MtsdfParams {
  float pxRange = 4.0f;
  float edgeSoftness = 0.125f;
  float outlineWidth = 0.0f;
  float glow = 0.0f;
};

struct Text2DDesc {
  std::string utf8;
  TextStyle style;
  TextLayoutParams layout;
  TextColor fillColor;
  TextColor outlineColor;
  MtsdfParams mtsdf;
  float x = 0.0f;
  float y = 0.0f;
  float zOrder = 0.0f;
};

struct Text3DDesc {
  std::string utf8;
  TextStyle style;
  TextLayoutParams layout;
  TextColor fillColor;
  TextColor outlineColor;
  MtsdfParams mtsdf;
  std::array<float, 16> worldFromText{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                                      0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                                      0.0f, 0.0f, 0.0f, 1.0f};
  TextBillboardMode billboard = TextBillboardMode::None;
  float maxScreenSizePx = 0.0f;
};

struct TextBounds {
  float minX = 0.0f;
  float minY = 0.0f;
  float maxX = 0.0f;
  float maxY = 0.0f;
};

} // namespace nuri
