#pragma once
#include "nuri/core/result.h"
#include "nuri/gfx/gpu_types.h"
#include "nuri/text/font_types.h"
#include <memory>
#include <memory_resource>
#include <span>
#include <string>
namespace nuri {

class GPUDevice;

class FontManager final {
public:
  struct CreateDesc {
    GPUDevice &gpu;
    std::pmr::memory_resource &memory;
    uint32_t initialFontCapacity = 64;
    uint32_t initialAtlasPageCapacity = 256;
  };
  explicit FontManager(const CreateDesc &desc);
  ~FontManager();
  FontManager(const FontManager &) = delete;
  FontManager &operator=(const FontManager &) = delete;
  FontManager(FontManager &&) = delete;
  FontManager &operator=(FontManager &&) = delete;
  Result<FontHandle, std::string> loadFont(const FontLoadDesc &desc);
  Result<bool, std::string> unloadFont(FontHandle font);
  bool isValid(FontHandle font) const;
  FontMetrics metrics(FontHandle font) const;
  float pxRange(FontHandle font) const;
  const GlyphMetrics *findGlyph(FontHandle font, GlyphId glyph) const;
  GlyphId lookupGlyphForCodepoint(FontHandle font, uint32_t codepoint) const;
  std::span<const FontHandle> fallbackChain(FontHandle font) const;
  AtlasPageHandle resolveAtlasPage(FontHandle font,
                                   uint16_t localPageIndex) const;
  GlyphLookupResult resolveGlyph(FontHandle font, GlyphId glyph) const;
  TextureHandle atlasTexture(AtlasPageHandle page) const;
  uint32_t atlasBindlessIndex(AtlasPageHandle page) const;
  Result<bool, std::string> setFallbackChain(FontHandle font,
                                             std::span<const FontHandle> chain);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace nuri
