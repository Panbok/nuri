#pragma once

#include "nuri/core/result.h"
#include "nuri/text/font_manager.h"

#include <memory>
#include <memory_resource>
#include <string>
#include <vector>

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
  std::pmr::vector<ShapedGlyph> glyphs;
  TextDirection direction = TextDirection::Ltr;
  TextBounds inkBounds{};
};

class NURI_API TextShaper {
public:
  struct CreateDesc {
    FontManager &fonts;
    std::pmr::memory_resource &memory;
  };

  static std::unique_ptr<TextShaper> create(const CreateDesc &desc);
  virtual ~TextShaper() = default;

  virtual Result<ShapedRun, std::string>
  shapeUtf8(std::string_view utf8, const TextStyle &style,
            const TextLayoutParams &params,
            std::pmr::memory_resource &scratch) = 0;
};

} // namespace nuri
