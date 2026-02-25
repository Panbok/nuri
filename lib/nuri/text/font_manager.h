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

class NURI_API FontManager {
public:
  FontManager() = default;

  struct CreateDesc {
    GPUDevice &gpu;
    std::pmr::memory_resource &memory;
    uint32_t initialFontCapacity = 64;
    uint32_t initialAtlasPageCapacity = 256;
  };

  struct PoolStats {
    uint32_t liveFonts = 0;
    uint32_t liveAtlasPages = 0;
    uint32_t liveShaperFaces = 0;
  };

  static std::unique_ptr<FontManager> create(const CreateDesc &desc);
  virtual ~FontManager() = default;

  FontManager(const FontManager &) = delete;
  FontManager &operator=(const FontManager &) = delete;
  FontManager(FontManager &&) = delete;
  FontManager &operator=(FontManager &&) = delete;

  virtual Result<FontHandle, std::string>
  loadFont(const FontLoadDesc &desc) = 0;
  virtual Result<bool, std::string> unloadFont(FontHandle font) = 0;

  virtual bool isValid(FontHandle font) const = 0;
  virtual FontMetrics metrics(FontHandle font) const = 0;
  virtual float pxRange(FontHandle font) const = 0;
  virtual const GlyphMetrics *findGlyph(FontHandle font,
                                        GlyphId glyph) const = 0;
  virtual GlyphId lookupGlyphForCodepoint(FontHandle font,
                                          uint32_t codepoint) const = 0;
  virtual std::span<const FontHandle> fallbackChain(FontHandle font) const = 0;
  virtual AtlasPageHandle resolveAtlasPage(FontHandle font,
                                           uint16_t localPageIndex) const = 0;
  virtual TextureHandle atlasTexture(AtlasPageHandle page) const = 0;
  virtual uint32_t atlasBindlessIndex(AtlasPageHandle page) const = 0;

  virtual Result<bool, std::string>
  setFallbackChain(FontHandle font, std::span<const FontHandle> chain) = 0;

  virtual void collectGarbage(uint64_t completedTimelineValue) = 0;
  virtual PoolStats poolStats() const = 0;
};

} // namespace nuri
