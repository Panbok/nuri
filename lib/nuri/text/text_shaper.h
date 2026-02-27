#pragma once

#include "nuri/core/containers/hash_map.h"
#include "nuri/core/result.h"
#include "nuri/text/font_manager.h"

#include <memory_resource>
#include <string>
#include <string_view>

struct hb_buffer_t;
struct hb_font_t;

namespace nuri {

struct ShapedGlyph {
  FontHandle font = kInvalidFontHandle;
  GlyphId glyphId = 0;
  uint32_t cluster = 0;
  uint32_t codepoint = 0;
  float advanceX = 0.0f;
  float advanceY = 0.0f;
  float offsetX = 0.0f;
  float offsetY = 0.0f;
};

struct ShapedRun {
  ShapedRun() = default;
  explicit ShapedRun(std::pmr::memory_resource *r) : glyphs(r) {}
  std::pmr::vector<ShapedGlyph> glyphs;
  TextDirection direction = TextDirection::Ltr;
  TextBounds inkBounds{};
};

struct HbFontContext {
  const FontManager *fonts = nullptr;
  FontHandle font = kInvalidFontHandle;
  float scale = 1.0f;
};

class NURI_API TextShaper {
public:
  struct CreateDesc {
    FontManager &fonts;
    std::pmr::memory_resource &memory;
  };

  explicit TextShaper(const CreateDesc &desc);
  ~TextShaper();

  TextShaper(const TextShaper &) = delete;
  TextShaper &operator=(const TextShaper &) = delete;
  TextShaper(TextShaper &&) = delete;
  TextShaper &operator=(TextShaper &&) = delete;

  Result<ShapedRun, std::string>
  shapeUtf8(std::string_view utf8, const TextStyle &style,
            const TextLayoutParams &params,
            std::pmr::memory_resource &scratch);

private:
  Result<bool, std::string> ensureHbObjects();

  FontManager &fonts_;
  std::pmr::memory_resource &memory_;
  HashMap<uint64_t, uint32_t> missingGlyphCounts_;
  HashMap<uint64_t, uint32_t> fallbackSwitchCounts_;
  HbFontContext hbContext_{};
  hb_buffer_t *hbBuffer_ = nullptr;
  hb_font_t *hbFont_ = nullptr;
};

} // namespace nuri
