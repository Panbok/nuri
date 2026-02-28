#include "nuri/pch.h"

#include "nuri/text/text_layouter.h"

#include "nuri/core/profiling.h"

namespace nuri {
namespace {

template <typename... Args>
[[nodiscard]] Result<TextLayout, std::string> makeError(Args &&...args) {
  std::ostringstream oss;
  (oss << ... << std::forward<Args>(args));
  return Result<TextLayout, std::string>::makeError(oss.str());
}

struct FontScaleInfo {
  float scale;
  float lineAdvance;
};

[[nodiscard]] FontScaleInfo computeFontScaleInfo(const FontManager &fonts,
                                                 FontHandle font, float pxSize,
                                                 float lineHeightScale) {
  FontMetrics m = fonts.metrics(font);
  if (m.unitsPerEm <= 0.0f) {
    m.unitsPerEm = 1.0f;
  }

  const float scale = pxSize / m.unitsPerEm;

  float advance = (m.ascent - m.descent + m.lineGap) * scale;
  if (!std::isfinite(advance) || advance <= 0.0f) {
    advance = pxSize * 1.2f;
  }
  const float scaled = advance * std::max(lineHeightScale, 0.01f);
  float lineAdvance;
  if (!std::isfinite(scaled) || scaled <= 0.0f) {
    lineAdvance = std::max(pxSize, 1.0f);
  } else {
    lineAdvance = scaled;
  }
  return {scale, lineAdvance};
}

[[nodiscard]] GlyphMetrics scaleGlyphMetrics(const GlyphMetrics &source,
                                             float scale) {
  GlyphMetrics s = source;
  s.advance *= scale;
  s.bearingX *= scale;
  s.bearingY *= scale;
  s.planeMinX *= scale;
  s.planeMinY *= scale;
  s.planeMaxX *= scale;
  s.planeMaxY *= scale;
  return s;
}

[[nodiscard]] bool isBreakableWhitespace(uint32_t codepoint) {
  return codepoint == static_cast<uint32_t>(' ') ||
         codepoint == static_cast<uint32_t>('\t');
}

constexpr float kAdvanceCollapseAbsEpsilon = 1.0e-4f;
constexpr float kAdvanceCollapseRelEpsilon = 0.05f;

void growBounds(TextBounds &bounds, bool &hasBounds, float minX, float minY,
                float maxX, float maxY) {
  if (!hasBounds) {
    bounds.minX = minX;
    bounds.minY = minY;
    bounds.maxX = maxX;
    bounds.maxY = maxY;
    hasBounds = true;
    return;
  }
  bounds.minX = std::min(bounds.minX, minX);
  bounds.minY = std::min(bounds.minY, minY);
  bounds.maxX = std::max(bounds.maxX, maxX);
  bounds.maxY = std::max(bounds.maxY, maxY);
}

[[nodiscard]] uint64_t hashBytes(const void *data, size_t len) {
  constexpr uint64_t kSeed = 0x9E3779B97F4A7C15ull;
  constexpr uint64_t kMul = 0x517CC1B727220A95ull;
  const auto *p = static_cast<const uint8_t *>(data);
  uint64_t h = kSeed ^ (len * kMul);
  while (len >= 8) {
    uint64_t w;
    std::memcpy(&w, p, 8);
    h ^= w;
    h *= kMul;
    h ^= h >> 28;
    p += 8;
    len -= 8;
  }
  if (len > 0) {
    uint64_t tail = 0;
    std::memcpy(&tail, p, len);
    h ^= tail;
    h *= kMul;
    h ^= h >> 28;
  }
  return h;
}

[[nodiscard]] uint32_t floatBits(float value) {
  uint32_t bits = 0u;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

[[nodiscard]] uint64_t hashMix(uint64_t hash, uint64_t value) {
  constexpr uint64_t kMul = 0x517CC1B727220A95ull;
  hash ^= value;
  hash *= kMul;
  hash ^= hash >> 28;
  return hash;
}

[[nodiscard]] uint64_t hashTextLayoutKey(std::string_view utf8,
                                         const TextStyle &style,
                                         const TextLayoutParams &params) {
  uint64_t h = hashBytes(utf8.data(), utf8.size());

  h = hashMix(h, style.font.value);
  h = hashMix(h, floatBits(style.pxSize));
  h = hashMix(h, floatBits(style.lineHeightScale));
  h = hashMix(h, floatBits(style.letterSpacing));
  h = hashMix(h, floatBits(style.wordSpacing));
  h = hashMix(h, static_cast<uint64_t>(style.fallback.handles.size()));
  for (const FontHandle fallback : style.fallback.handles) {
    h = hashMix(h, fallback.value);
  }

  h = hashMix(h, floatBits(params.maxWidthPx));
  h = hashMix(h, floatBits(params.maxHeightPx));
  h = hashMix(h, static_cast<uint64_t>(params.wrapMode));
  h = hashMix(h, static_cast<uint64_t>(params.direction));
  h = hashMix(h, static_cast<uint64_t>(params.alignH));
  h = hashMix(h, static_cast<uint64_t>(params.alignV));
  h = hashMix(h, params.enableKerning ? 1u : 0u);
  h = hashMix(h, params.enableLigatures ? 1u : 0u);
  h = hashMix(h, params.allowFallback ? 1u : 0u);
  return h;
}

[[nodiscard]] TextLayout
cloneLayoutToMemory(const TextLayout &source,
                    std::pmr::memory_resource &memory) {
  TextLayout out{
      .glyphs = std::pmr::vector<LayoutGlyph>(&memory),
      .bounds = source.bounds,
      .lineCount = source.lineCount,
  };
  out.glyphs.assign(source.glyphs.begin(), source.glyphs.end());
  return out;
}

} // namespace

TextLayouter::CacheEntry::CacheEntry(std::pmr::memory_resource *memory)
    : utf8(memory), fallbackHandles(memory),
      layout{
          .glyphs = std::pmr::vector<LayoutGlyph>(memory),
          .bounds = {},
          .lineCount = 0,
      } {}

TextLayouter::TextLayouter(const CreateDesc &desc)
    : fonts_(desc.fonts), shaper_(desc.shaper), memory_(desc.memory),
      pool_(&memory_) {
  pool_.reserve(kMaxCacheEntries);
}

Result<TextLayout, std::string>
TextLayouter::layoutUtf8(std::string_view utf8, const TextStyle &style,
                         const TextLayoutParams &params,
                         std::pmr::memory_resource &outMemory,
                         std::pmr::memory_resource &scratch) {
  NURI_PROFILER_FUNCTION();
  const uint64_t keyHash = hashTextLayoutKey(utf8, style, params);

  CacheEntry *cachedEntry = nullptr;
  NURI_PROFILER_ZONE("TextLayouter::cacheLookup", NURI_PROFILER_COLOR_CMD_DRAW);
  if (const auto mapIt = cacheMap_.find(keyHash); mapIt != cacheMap_.end()) {
    const uint32_t slot = mapIt->second;
    CacheEntry &entry = pool_[slot];
    if (cacheKeyEquals(entry, utf8, style, params)) {
      lruPromote(slot);
      cachedEntry = &entry;
    }
  }
  NURI_PROFILER_ZONE_END();
  if (cachedEntry != nullptr) {
    return Result<TextLayout, std::string>::makeResult(
        cloneLayoutToMemory(cachedEntry->layout, outMemory));
  }

  auto shapedResult = shaper_.shapeUtf8(utf8, style, params, scratch);
  if (shapedResult.hasError()) {
    return makeError("TextLayouter::layoutUtf8: ", shapedResult.error());
  }

  const uint32_t slot = allocateSlot();
  CacheEntry &ce = pool_[slot];
  fillCacheKey(ce, keyHash, utf8, style, params);

  TextLayout &layout = ce.layout;
  layout.glyphs.clear();
  layout.bounds = {};
  layout.lineCount = 1;

  const ShapedRun &run = shapedResult.value();
  if (run.glyphs.empty()) {
    layout.lineCount = 0;
    insertIntoCache(keyHash, slot);
    return Result<TextLayout, std::string>::makeResult(
        cloneLayoutToMemory(layout, outMemory));
  }

  layout.glyphs.reserve(run.glyphs.size());

  const bool wrapEnabled =
      params.wrapMode != TextWrapMode::None && params.maxWidthPx > 0.0f;
  const float maxWidthPx = params.maxWidthPx;
  FontScaleInfo fsi = computeFontScaleInfo(fonts_, style.font, style.pxSize,
                                           style.lineHeightScale);
  float penX = 0.0f;
  float penY = 0.0f;
  float lineAdvance = fsi.lineAdvance;
  size_t lineStartGlyphIndex = 0;
  bool inWord = false;
  size_t wordStartGlyphIndex = 0;
  float wordStartPenX = 0.0f;
  FontHandle cachedFontForMetrics = kInvalidFontHandle;
  float cachedScale = fsi.scale;
  float cachedLineAdvance = fsi.lineAdvance;

  bool hasBounds = false;
  bool boundsValid = true;
  TextBounds runningBounds{};

  NURI_PROFILER_ZONE("TextLayouter::buildLayoutGlyphs",
                     NURI_PROFILER_COLOR_CMD_DRAW);
  for (const ShapedGlyph &sg : run.glyphs) {
    const uint32_t cp = sg.codepoint;
    if (cp == static_cast<uint32_t>('\r')) {
      continue;
    }
    if (cp == static_cast<uint32_t>('\n')) {
      penX = 0.0f;
      penY += lineAdvance;
      ++layout.lineCount;
      lineStartGlyphIndex = layout.glyphs.size();
      inWord = false;
      continue;
    }

    const GlyphLookupResult resolved = fonts_.resolveGlyph(sg.font, sg.glyphId);
    if (resolved.metrics == nullptr || !::nuri::isValid(resolved.atlasPage)) {
      penX += sg.advanceX;
      penY += sg.advanceY;
      continue;
    }

    if (sg.font.value != cachedFontForMetrics.value) {
      cachedFontForMetrics = sg.font;
      fsi = computeFontScaleInfo(fonts_, sg.font, style.pxSize,
                                 style.lineHeightScale);
      cachedScale = fsi.scale;
      cachedLineAdvance = fsi.lineAdvance;
    }
    lineAdvance = cachedLineAdvance;
    const GlyphMetrics glyphMetrics =
        scaleGlyphMetrics(*resolved.metrics, cachedScale);

    // HarfBuzz can return collapsed advances with our custom face/callback
    // path. Fall back to the font's own glyph advance when shaped advance
    // is invalid.
    const float fallbackAdvanceX = glyphMetrics.advance;
    float advanceX = sg.advanceX;
    if (!std::isfinite(advanceX) ||
        std::abs(advanceX) <
            std::max(kAdvanceCollapseAbsEpsilon,
                     std::abs(fallbackAdvanceX) * kAdvanceCollapseRelEpsilon)) {
      advanceX = fallbackAdvanceX;
    }
    float advanceY = sg.advanceY;
    if (!std::isfinite(advanceY)) {
      advanceY = 0.0f;
    }

    const bool breakable = isBreakableWhitespace(cp);
    if (params.wrapMode == TextWrapMode::Word) {
      if (breakable) {
        inWord = false;
      } else if (!inWord) {
        inWord = true;
        wordStartGlyphIndex = layout.glyphs.size();
        wordStartPenX = penX;
      }
    }

    LayoutGlyph glyph{};
    glyph.font = sg.font;
    glyph.glyphId = sg.glyphId;
    glyph.atlasPage = resolved.atlasPage;
    glyph.x = penX + sg.offsetX;
    glyph.y = penY + sg.offsetY;
    glyph.metrics = glyphMetrics;
    layout.glyphs.push_back(glyph);

    if (boundsValid) {
      growBounds(runningBounds, hasBounds, glyph.x + glyphMetrics.planeMinX,
                 glyph.y - glyphMetrics.planeMaxY,
                 glyph.x + glyphMetrics.planeMaxX,
                 glyph.y - glyphMetrics.planeMinY);
    }

    const size_t currentGlyphIndex = layout.glyphs.size() - 1u;
    penX += advanceX;
    penY += advanceY;

    if (wrapEnabled && penX > maxWidthPx &&
        currentGlyphIndex > lineStartGlyphIndex) {
      size_t breakIndex = currentGlyphIndex;
      float breakPenX = penX - advanceX;
      if (params.wrapMode == TextWrapMode::Word && inWord &&
          wordStartGlyphIndex > lineStartGlyphIndex) {
        breakIndex = wordStartGlyphIndex;
        breakPenX = wordStartPenX;
      }

      for (size_t i = breakIndex; i < layout.glyphs.size(); ++i) {
        layout.glyphs[i].x -= breakPenX;
        layout.glyphs[i].y += lineAdvance;
      }

      penX -= breakPenX;
      penY += lineAdvance;
      ++layout.lineCount;
      lineStartGlyphIndex = breakIndex;

      boundsValid = false;

      if (params.wrapMode == TextWrapMode::Word &&
          breakIndex == wordStartGlyphIndex) {
        wordStartPenX = 0.0f;
      } else {
        inWord = false;
      }
    }
  }
  NURI_PROFILER_ZONE_END();

  // --- Finalize bounds ---
  if (layout.glyphs.empty()) {
    layout.bounds = run.inkBounds;
  } else if (boundsValid && hasBounds) {
    layout.bounds = runningBounds;
  } else {
    hasBounds = false;
    for (const LayoutGlyph &g : layout.glyphs) {
      growBounds(layout.bounds, hasBounds, g.x + g.metrics.planeMinX,
                 g.y - g.metrics.planeMaxY, g.x + g.metrics.planeMaxX,
                 g.y - g.metrics.planeMinY);
    }
    if (!hasBounds) {
      layout.bounds = run.inkBounds;
    }
  }

  insertIntoCache(keyHash, slot);
  return Result<TextLayout, std::string>::makeResult(
      cloneLayoutToMemory(layout, outMemory));
}

void TextLayouter::lruRemove(uint32_t idx) {
  CacheEntry &e = pool_[idx];
  if (e.lruPrev != kNoSlot) {
    pool_[e.lruPrev].lruNext = e.lruNext;
  } else if (lruHead_ == idx) {
    lruHead_ = e.lruNext;
  }
  if (e.lruNext != kNoSlot) {
    pool_[e.lruNext].lruPrev = e.lruPrev;
  } else if (lruTail_ == idx) {
    lruTail_ = e.lruPrev;
  }
  e.lruPrev = kNoSlot;
  e.lruNext = kNoSlot;
}

void TextLayouter::lruPushFront(uint32_t idx) {
  CacheEntry &e = pool_[idx];
  e.lruPrev = kNoSlot;
  e.lruNext = lruHead_;
  if (lruHead_ != kNoSlot) {
    pool_[lruHead_].lruPrev = idx;
  }
  lruHead_ = idx;
  if (lruTail_ == kNoSlot) {
    lruTail_ = idx;
  }
}

void TextLayouter::lruPromote(uint32_t idx) {
  if (lruHead_ == idx) {
    return;
  }
  lruRemove(idx);
  lruPushFront(idx);
}

uint32_t TextLayouter::allocateSlot() {
  if (freeHead_ != kNoSlot) {
    const uint32_t s = freeHead_;
    freeHead_ = pool_[s].lruNext;
    pool_[s].lruPrev = kNoSlot;
    pool_[s].lruNext = kNoSlot;
    return s;
  }
  if (pool_.size() < kMaxCacheEntries) {
    pool_.emplace_back(&memory_);
    return static_cast<uint32_t>(pool_.size() - 1);
  }
  const uint32_t victim = lruTail_;
  assert(victim != kNoSlot);
  lruRemove(victim);
  if (auto it = cacheMap_.find(pool_[victim].hash);
      it != cacheMap_.end() && it->second == victim) {
    cacheMap_.erase(it);
  }
  pool_[victim].layout.glyphs.clear();
  pool_[victim].utf8.clear();
  pool_[victim].fallbackHandles.clear();
  return victim;
}

// Handle hash collision (astronomically rare with 64-bit keys and <=256
// entries) by freeing the stale slot before overwriting the map entry.
void TextLayouter::insertIntoCache(uint64_t keyHash, uint32_t slot) {
  if (auto it = cacheMap_.find(keyHash); it != cacheMap_.end()) {
    const uint32_t oldSlot = it->second;
    if (oldSlot != slot) {
      lruRemove(oldSlot);
      pool_[oldSlot].layout.glyphs.clear();
      pool_[oldSlot].utf8.clear();
      pool_[oldSlot].fallbackHandles.clear();
      pool_[oldSlot].lruNext = freeHead_;
      pool_[oldSlot].lruPrev = kNoSlot;
      freeHead_ = oldSlot;
    }
  }
  cacheMap_[keyHash] = slot;
  lruPushFront(slot);
}

void TextLayouter::fillCacheKey(CacheEntry &entry, uint64_t hash,
                                std::string_view utf8, const TextStyle &style,
                                const TextLayoutParams &params) {
  entry.hash = hash;
  entry.utf8.assign(utf8.begin(), utf8.end());
  entry.font = style.font;
  entry.pxSize = style.pxSize;
  entry.lineHeightScale = style.lineHeightScale;
  entry.letterSpacing = style.letterSpacing;
  entry.wordSpacing = style.wordSpacing;
  entry.fallbackHandles.assign(style.fallback.handles.begin(),
                               style.fallback.handles.end());
  entry.params = params;
}

bool TextLayouter::cacheKeyEquals(const CacheEntry &entry,
                                  std::string_view utf8, const TextStyle &style,
                                  const TextLayoutParams &params) {
  if (entry.utf8 != utf8) {
    return false;
  }
  if (entry.font.value != style.font.value || entry.pxSize != style.pxSize ||
      entry.lineHeightScale != style.lineHeightScale ||
      entry.letterSpacing != style.letterSpacing ||
      entry.wordSpacing != style.wordSpacing) {
    return false;
  }
  if (entry.fallbackHandles.size() != style.fallback.handles.size()) {
    return false;
  }
  for (size_t i = 0; i < entry.fallbackHandles.size(); ++i) {
    if (entry.fallbackHandles[i].value != style.fallback.handles[i].value) {
      return false;
    }
  }

  const TextLayoutParams &c = entry.params;
  return c.maxWidthPx == params.maxWidthPx &&
         c.maxHeightPx == params.maxHeightPx && c.wrapMode == params.wrapMode &&
         c.direction == params.direction && c.alignH == params.alignH &&
         c.alignV == params.alignV && c.enableKerning == params.enableKerning &&
         c.enableLigatures == params.enableLigatures &&
         c.allowFallback == params.allowFallback;
}

} // namespace nuri
