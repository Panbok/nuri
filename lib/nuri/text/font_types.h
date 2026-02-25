#pragma once

#include "nuri/defines.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <span>
#include <string_view>
#include <vector>

namespace nuri {

static constexpr uint32_t kTextHandleIndexBits = 20;
static constexpr uint32_t kTextHandleGenerationBits = 12;
static constexpr uint32_t kTextHandleIndexMask =
    (1u << kTextHandleIndexBits) - 1u;
static constexpr uint32_t kTextHandleGenerationMask =
    (1u << kTextHandleGenerationBits) - 1u;

struct FontHandle {
  uint32_t value = 0;
};
struct AtlasPageHandle {
  uint32_t value = 0;
};
struct ShaperFaceHandle {
  uint32_t value = 0;
};

static constexpr FontHandle kInvalidFontHandle{};
static constexpr AtlasPageHandle kInvalidAtlasPageHandle{};
static constexpr ShaperFaceHandle kInvalidShaperFaceHandle{};

[[nodiscard]] constexpr bool isValid(FontHandle h) noexcept {
  return h.value != 0;
}
[[nodiscard]] constexpr bool isValid(AtlasPageHandle h) noexcept {
  return h.value != 0;
}
[[nodiscard]] constexpr bool isValid(ShaperFaceHandle h) noexcept {
  return h.value != 0;
}

[[nodiscard]] constexpr uint32_t
packTextHandleValue(uint32_t index, uint32_t generation) noexcept {
  if (generation == 0) {
    return 0;
  }
  return ((generation & kTextHandleGenerationMask) << kTextHandleIndexBits) |
         (index & kTextHandleIndexMask);
}

[[nodiscard]] constexpr uint32_t textHandleIndex(uint32_t packed) noexcept {
  return packed & kTextHandleIndexMask;
}

[[nodiscard]] constexpr uint32_t
textHandleGeneration(uint32_t packed) noexcept {
  return (packed >> kTextHandleIndexBits) & kTextHandleGenerationMask;
}

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
  uint16_t localPageIndex = 0;
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

struct FontLoadDesc {
  std::string_view path;
  std::string_view debugName;
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
  float shadow = 0.0f;
};

struct Text2DDesc {
  std::string_view utf8;
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
  std::string_view utf8;
  TextStyle style;
  TextLayoutParams layout;
  TextColor fillColor;
  TextColor outlineColor;
  MtsdfParams mtsdf;
  std::array<float, 16> worldFromText{};
  TextBillboardMode billboard = TextBillboardMode::Spherical;
  float maxScreenSizePx = 0.0f;
};

struct TextBounds {
  float minX = 0.0f;
  float minY = 0.0f;
  float maxX = 0.0f;
  float maxY = 0.0f;
};

} // namespace nuri
