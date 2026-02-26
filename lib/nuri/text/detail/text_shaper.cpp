#include "nuri/pch.h"

#include "nuri/text/text_shaper.h"

#include "nuri/core/containers/hash_map.h"
#include "nuri/core/log.h"
#include "nuri/core/profiling.h"

#include <hb.h>

namespace nuri {
namespace {

template <typename... Args>
[[nodiscard]] Result<ShapedRun, std::string> makeError(Args &&...args) {
  std::ostringstream oss;
  (oss << ... << std::forward<Args>(args));
  return Result<ShapedRun, std::string>::makeError(oss.str());
}

struct HbFontContext {
  const FontManager *fonts = nullptr;
  FontHandle font = kInvalidFontHandle;
  float scale = 1.0f;
};

struct ResolvedGlyph {
  FontHandle font = kInvalidFontHandle;
  GlyphId glyphId = 0;
  const GlyphMetrics *metrics = nullptr;
};

[[nodiscard]] hb_position_t toHbPos(float value) {
  return static_cast<hb_position_t>(
      std::lround(static_cast<double>(value) * 64.0));
}

[[nodiscard]] hb_bool_t hbNominalGlyph(hb_font_t *, void *fontData,
                                       hb_codepoint_t unicode,
                                       hb_codepoint_t *glyph, void *) {
  const auto *ctx = static_cast<const HbFontContext *>(fontData);
  if (ctx == nullptr || ctx->fonts == nullptr || glyph == nullptr) {
    return 0;
  }
  const GlyphId glyphId =
      ctx->fonts->lookupGlyphForCodepoint(ctx->font, unicode);
  if (glyphId == 0) {
    return 0;
  }
  *glyph = glyphId;
  return 1;
}

[[nodiscard]] hb_position_t hbHAdvance(hb_font_t *, void *fontData,
                                       hb_codepoint_t glyph, void *) {
  const auto *ctx = static_cast<const HbFontContext *>(fontData);
  if (ctx == nullptr || ctx->fonts == nullptr) {
    return 0;
  }
  const GlyphMetrics *metrics = ctx->fonts->findGlyph(ctx->font, glyph);
  if (metrics == nullptr) {
    return 0;
  }
  return toHbPos(metrics->advance * ctx->scale);
}

[[nodiscard]] hb_bool_t hbGlyphExtents(hb_font_t *, void *fontData,
                                       hb_codepoint_t glyph,
                                       hb_glyph_extents_t *extents, void *) {
  const auto *ctx = static_cast<const HbFontContext *>(fontData);
  if (ctx == nullptr || ctx->fonts == nullptr || extents == nullptr) {
    return 0;
  }
  const GlyphMetrics *metrics = ctx->fonts->findGlyph(ctx->font, glyph);
  if (metrics == nullptr) {
    return 0;
  }

  const float minX = metrics->planeMinX * ctx->scale;
  const float maxX = metrics->planeMaxX * ctx->scale;
  const float minY = metrics->planeMinY * ctx->scale;
  const float maxY = metrics->planeMaxY * ctx->scale;

  extents->x_bearing = toHbPos(minX);
  extents->y_bearing = toHbPos(maxY);
  extents->width = toHbPos(maxX - minX);
  extents->height = toHbPos(minY - maxY);
  return 1;
}

[[nodiscard]] hb_font_funcs_t *hbFontFuncs() {
  static hb_font_funcs_t *funcs = []() {
    hb_font_funcs_t *created = hb_font_funcs_create();
    hb_font_funcs_set_nominal_glyph_func(created, hbNominalGlyph, nullptr,
                                         nullptr);
    hb_font_funcs_set_glyph_h_advance_func(created, hbHAdvance, nullptr,
                                           nullptr);
    hb_font_funcs_set_glyph_extents_func(created, hbGlyphExtents, nullptr,
                                         nullptr);
    hb_font_funcs_make_immutable(created);
    return created;
  }();
  return funcs;
}

[[nodiscard]] FontHandle pickPrimaryFont(const FontManager &fonts,
                                         const TextStyle &style) {
  if (::nuri::isValid(style.font) && fonts.isValid(style.font)) {
    return style.font;
  }
  for (const FontHandle fallback : style.fallback.handles) {
    if (::nuri::isValid(fallback) && fonts.isValid(fallback)) {
      return fallback;
    }
  }
  return kInvalidFontHandle;
}

void buildFallbackCandidates(const FontManager &fonts, const TextStyle &style,
                             FontHandle primaryFont, bool allowFallback,
                             std::pmr::vector<FontHandle> &out) {
  out.clear();
  HashMap<uint32_t, bool> seen;
  const auto tryAppend = [&](FontHandle font) {
    if (!::nuri::isValid(font) || !fonts.isValid(font)) {
      return;
    }
    auto [_, inserted] = seen.try_emplace(font.value, true);
    if (inserted) {
      out.push_back(font);
    }
  };

  tryAppend(primaryFont);
  if (!allowFallback) {
    return;
  }

  for (const FontHandle fallback : style.fallback.handles) {
    tryAppend(fallback);
  }

  constexpr size_t kMaxCandidates = 64;
  for (size_t i = 0; i < out.size() && out.size() < kMaxCandidates; ++i) {
    const std::span<const FontHandle> chain = fonts.fallbackChain(out[i]);
    for (const FontHandle chainFont : chain) {
      tryAppend(chainFont);
      if (out.size() >= kMaxCandidates) {
        break;
      }
    }
  }
}

[[nodiscard]] GlyphId lookupGlyphFromCandidates(
    const FontManager &fonts, std::span<const FontHandle> candidates,
    uint32_t codepoint, FontHandle *outFont, uint32_t *outCandidateIndex) {
  if (codepoint == 0) {
    return 0;
  }
  for (size_t i = 0; i < candidates.size(); ++i) {
    const FontHandle candidateFont = candidates[i];
    const GlyphId glyphId =
        fonts.lookupGlyphForCodepoint(candidateFont, codepoint);
    if (glyphId == 0) {
      continue;
    }
    if (outFont != nullptr) {
      *outFont = candidateFont;
    }
    if (outCandidateIndex != nullptr) {
      *outCandidateIndex = static_cast<uint32_t>(i);
    }
    return glyphId;
  }
  return 0;
}

[[nodiscard]] uint64_t makeTelemetryKey(uint32_t left, uint32_t right) {
  return (static_cast<uint64_t>(left) << 32u) | static_cast<uint64_t>(right);
}

[[nodiscard]] hb_direction_t toHbDirection(TextDirection direction) {
  switch (direction) {
  case TextDirection::Ltr:
    return HB_DIRECTION_LTR;
  case TextDirection::Rtl:
    return HB_DIRECTION_RTL;
  case TextDirection::Auto:
    return HB_DIRECTION_INVALID;
  }
  return HB_DIRECTION_INVALID;
}

[[nodiscard]] uint32_t decodeUtf8CodepointAt(std::string_view utf8,
                                             uint32_t byteOffset) {
  if (byteOffset >= utf8.size()) {
    return 0;
  }

  const auto *bytes = reinterpret_cast<const uint8_t *>(utf8.data());
  const size_t index = static_cast<size_t>(byteOffset);
  const uint8_t b0 = bytes[index];
  if ((b0 & 0x80u) == 0) {
    return b0;
  }

  const auto continuation = [&](size_t offset) -> uint8_t {
    if (offset >= utf8.size()) {
      return 0xffu;
    }
    const uint8_t b = bytes[offset];
    return ((b & 0xc0u) == 0x80u) ? static_cast<uint8_t>(b & 0x3fu) : 0xffu;
  };

  if ((b0 & 0xe0u) == 0xc0u) {
    const uint8_t c1 = continuation(index + 1u);
    if (c1 == 0xffu) {
      return 0;
    }
    return (static_cast<uint32_t>(b0 & 0x1fu) << 6u) | c1;
  }
  if ((b0 & 0xf0u) == 0xe0u) {
    const uint8_t c1 = continuation(index + 1u);
    const uint8_t c2 = continuation(index + 2u);
    if (c1 == 0xffu || c2 == 0xffu) {
      return 0;
    }
    return (static_cast<uint32_t>(b0 & 0x0fu) << 12u) |
           (static_cast<uint32_t>(c1) << 6u) | c2;
  }
  if ((b0 & 0xf8u) == 0xf0u) {
    const uint8_t c1 = continuation(index + 1u);
    const uint8_t c2 = continuation(index + 2u);
    const uint8_t c3 = continuation(index + 3u);
    if (c1 == 0xffu || c2 == 0xffu || c3 == 0xffu) {
      return 0;
    }
    return (static_cast<uint32_t>(b0 & 0x07u) << 18u) |
           (static_cast<uint32_t>(c1) << 12u) |
           (static_cast<uint32_t>(c2) << 6u) | c3;
  }
  return 0;
}

void preDecodeUtf8Offsets(std::string_view utf8,
                          std::pmr::vector<uint32_t> &out) {
  out.assign(utf8.size(), 0u);
  size_t i = 0;
  while (i < utf8.size()) {
    out[i] = decodeUtf8CodepointAt(utf8, static_cast<uint32_t>(i));
    const auto b0 = static_cast<uint8_t>(utf8[i]);
    if ((b0 & 0x80u) == 0) {
      i += 1;
    } else if ((b0 & 0xe0u) == 0xc0u) {
      i += 2;
    } else if ((b0 & 0xf0u) == 0xe0u) {
      i += 3;
    } else if ((b0 & 0xf8u) == 0xf0u) {
      i += 4;
    } else {
      i += 1;
    }
  }
}

[[nodiscard]] TextDirection fromHbDirection(hb_direction_t direction) {
  if (direction == HB_DIRECTION_RTL) {
    return TextDirection::Rtl;
  }
  return TextDirection::Ltr;
}

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

[[nodiscard]] float fontScaleForPxSize(const FontManager &fonts,
                                       FontHandle font, float pxSize) {
  FontMetrics metrics = fonts.metrics(font);
  if (metrics.unitsPerEm <= 0.0f) {
    metrics.unitsPerEm = 1.0f;
  }
  return pxSize / metrics.unitsPerEm;
}

class TextShaperHb final : public TextShaper {
public:
  explicit TextShaperHb(const CreateDesc &desc)
      : fonts_(desc.fonts), memory_(desc.memory) {}
  ~TextShaperHb() override {
    if (hbFont_ != nullptr) {
      hb_font_destroy(hbFont_);
      hbFont_ = nullptr;
    }
    if (hbBuffer_ != nullptr) {
      hb_buffer_destroy(hbBuffer_);
      hbBuffer_ = nullptr;
    }
  }

  Result<ShapedRun, std::string>
  shapeUtf8(std::string_view utf8, const TextStyle &style,
            const TextLayoutParams &params,
            std::pmr::memory_resource &scratch) override {
    NURI_PROFILER_FUNCTION();
    ShapedRun run{
        .glyphs = std::pmr::vector<ShapedGlyph>(&scratch),
        .direction = TextDirection::Ltr,
        .inkBounds = {},
    };

    if (utf8.empty()) {
      return Result<ShapedRun, std::string>::makeResult(std::move(run));
    }

    const FontHandle primaryFont = pickPrimaryFont(fonts_, style);
    if (!::nuri::isValid(primaryFont)) {
      return makeError(
          "TextShaper::shapeUtf8: no valid font in style/fallback");
    }
    std::pmr::vector<FontHandle> candidates(&scratch);
    candidates.reserve(16);
    NURI_PROFILER_ZONE("TextShaper::buildFallbackCandidates",
                       NURI_PROFILER_COLOR_CMD_DISPATCH);
    buildFallbackCandidates(fonts_, style, primaryFont, params.allowFallback,
                            candidates);
    NURI_PROFILER_ZONE_END();
    if (candidates.empty()) {
      return makeError(
          "TextShaper::shapeUtf8: fallback candidate set is empty");
    }
    const std::span<const FontHandle> candidateSpan(candidates.data(),
                                                    candidates.size());

    const float primaryScale =
        fontScaleForPxSize(fonts_, primaryFont, style.pxSize);
    auto ensureHb = ensureHbObjects();
    if (ensureHb.hasError()) {
      return makeError("TextShaper::shapeUtf8: ", ensureHb.error());
    }

    hbContext_.fonts = &fonts_;
    hbContext_.font = primaryFont;
    hbContext_.scale = primaryScale;

    hb_buffer_reset(hbBuffer_);
    hb_buffer_add_utf8(hbBuffer_, utf8.data(), static_cast<int>(utf8.size()), 0,
                       static_cast<int>(utf8.size()));
    const hb_direction_t requestedDirection = toHbDirection(params.direction);
    if (requestedDirection != HB_DIRECTION_INVALID) {
      hb_buffer_set_direction(hbBuffer_, requestedDirection);
    }
    hb_buffer_guess_segment_properties(hbBuffer_);
    NURI_PROFILER_ZONE("TextShaper::hbShape", NURI_PROFILER_COLOR_CMD_DISPATCH);
    hb_shape(hbFont_, hbBuffer_, nullptr, 0);
    NURI_PROFILER_ZONE_END();

    const unsigned int count = hb_buffer_get_length(hbBuffer_);
    const hb_glyph_info_t *infos =
        hb_buffer_get_glyph_infos(hbBuffer_, nullptr);
    const hb_glyph_position_t *positions =
        hb_buffer_get_glyph_positions(hbBuffer_, nullptr);

    run.glyphs.reserve(count);
    run.direction = fromHbDirection(hb_buffer_get_direction(hbBuffer_));

    std::pmr::vector<uint32_t> codepointByOffset(&scratch);
    preDecodeUtf8Offsets(utf8, codepointByOffset);

    ResolvedGlyph replacementGlyph{};
    {
      FontHandle rFont{};
      uint32_t rIdx = 0;
      GlyphId rId = lookupGlyphFromCandidates(fonts_, candidateSpan, 0xfffdu,
                                               &rFont, &rIdx);
      if (rId == 0) {
        rId = lookupGlyphFromCandidates(
            fonts_, candidateSpan, static_cast<uint32_t>('?'), &rFont, &rIdx);
      }
      if (rId != 0) {
        replacementGlyph = {rFont, rId, fonts_.findGlyph(rFont, rId)};
      }
    }

    HashMap<uint32_t, ResolvedGlyph> resolveCache;
    resolveCache.reserve(std::min(count, 256u));

    HashMap<uint32_t, float> scaleCache;
    scaleCache.reserve(candidates.size());
    scaleCache[primaryFont.value] = primaryScale;

    float penX = 0.0f;
    float penY = 0.0f;
    bool hasBounds = false;
    uint32_t fallbackGlyphCount = 0;
    uint32_t missingGlyphCount = 0;
    NURI_PROFILER_ZONE("TextShaper::resolveGlyphs",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    for (unsigned int i = 0; i < count; ++i) {
      const uint32_t cluster = infos[i].cluster;
      const uint32_t sourceCodepoint =
          (cluster < codepointByOffset.size()) ? codepointByOffset[cluster] : 0;

      ResolvedGlyph resolved{};
      bool fromCache = false;
      if (sourceCodepoint != 0) {
        auto cacheIt = resolveCache.find(sourceCodepoint);
        if (cacheIt != resolveCache.end()) {
          resolved = cacheIt->second;
          fromCache = true;
        }
      }

      if (!fromCache) {
        FontHandle rFont = primaryFont;
        uint32_t rIdx = 0;
        GlyphId rId = lookupGlyphFromCandidates(
            fonts_, candidateSpan, sourceCodepoint, &rFont, &rIdx);
        if (rId == 0 &&
            fonts_.findGlyph(primaryFont, infos[i].codepoint) != nullptr) {
          rId = static_cast<GlyphId>(infos[i].codepoint);
          rFont = primaryFont;
        }
        if (rId != 0) {
          resolved = {rFont, rId, fonts_.findGlyph(rFont, rId)};
        } else {
          resolved = replacementGlyph;
          if (resolved.glyphId == 0) {
            resolved.font = primaryFont;
          }
        }
        if (sourceCodepoint != 0) {
          resolveCache[sourceCodepoint] = resolved;
        }
      }

      if (resolved.glyphId == 0) {
        ++missingGlyphCount;
        const uint32_t missingCp =
            sourceCodepoint != 0 ? sourceCodepoint : 0xfffdu;
        const uint64_t key = makeTelemetryKey(primaryFont.value, missingCp);
        const uint32_t hitCount = ++missingGlyphCounts_[key];
        if (hitCount == 1u) {
          NURI_LOG_WARNING("TextShaper: missing glyph U+%06X for font=0x%08X",
                           static_cast<unsigned int>(missingCp),
                           static_cast<unsigned int>(primaryFont.value));
        } else if ((hitCount & 63u) == 0u) {
          NURI_LOG_DEBUG("TextShaper: missing glyph U+%06X repeated %u times "
                         "for font=0x%08X",
                         static_cast<unsigned int>(missingCp),
                         static_cast<unsigned int>(hitCount),
                         static_cast<unsigned int>(primaryFont.value));
        }
      } else if (resolved.font.value != primaryFont.value) {
        ++fallbackGlyphCount;
        const uint64_t key =
            makeTelemetryKey(primaryFont.value, resolved.font.value);
        const uint32_t switchCount = ++fallbackSwitchCounts_[key];
        if (switchCount == 1u) {
          NURI_LOG_INFO(
              "TextShaper: fallback font switch primary=0x%08X "
              "fallback=0x%08X",
              static_cast<unsigned int>(primaryFont.value),
              static_cast<unsigned int>(resolved.font.value));
        }
      }

      auto [scaleIt, scaleInserted] =
          scaleCache.try_emplace(resolved.font.value, 0.0f);
      if (scaleInserted) {
        scaleIt->second =
            fontScaleForPxSize(fonts_, resolved.font, style.pxSize);
      }
      const float resolvedScale = scaleIt->second;

      ShapedGlyph glyph{};
      glyph.font = resolved.font;
      glyph.glyphId = resolved.glyphId;
      glyph.cluster = cluster;
      glyph.codepoint = sourceCodepoint;
      if (resolved.font.value != primaryFont.value &&
          resolved.metrics != nullptr) {
        glyph.advanceX = resolved.metrics->advance * resolvedScale;
      } else {
        glyph.advanceX = static_cast<float>(positions[i].x_advance) / 64.0f;
      }
      glyph.advanceY = 0.0f;
      glyph.offsetX = static_cast<float>(positions[i].x_offset) / 64.0f;
      glyph.offsetY = 0.0f;
      run.glyphs.push_back(glyph);

      if (resolved.metrics != nullptr) {
        const float minX =
            penX + glyph.offsetX +
            (resolved.metrics->planeMinX * resolvedScale);
        const float minY =
            penY + glyph.offsetY +
            (resolved.metrics->planeMinY * resolvedScale);
        const float maxX =
            penX + glyph.offsetX +
            (resolved.metrics->planeMaxX * resolvedScale);
        const float maxY =
            penY + glyph.offsetY +
            (resolved.metrics->planeMaxY * resolvedScale);
        growBounds(run.inkBounds, hasBounds, minX, minY, maxX, maxY);
      }

      penX += glyph.advanceX;
      penY += glyph.advanceY;
    }
    NURI_PROFILER_ZONE_END();

    if (fallbackGlyphCount > 0 || missingGlyphCount > 0) {
      NURI_LOG_DEBUG("TextShaper: run telemetry primary=0x%08X glyphs=%u "
                     "fallback=%u missing=%u",
                     static_cast<unsigned int>(primaryFont.value),
                     static_cast<unsigned int>(count),
                     static_cast<unsigned int>(fallbackGlyphCount),
                     static_cast<unsigned int>(missingGlyphCount));
    }

    return Result<ShapedRun, std::string>::makeResult(std::move(run));
  }

private:
  Result<bool, std::string> ensureHbObjects() {
    if (hbBuffer_ == nullptr) {
      hbBuffer_ = hb_buffer_create();
      if (hbBuffer_ == nullptr) {
        return Result<bool, std::string>::makeError("hb_buffer_create failed");
      }
    }
    if (hbFont_ == nullptr) {
      hbFont_ = hb_font_create(hb_face_get_empty());
      if (hbFont_ == nullptr) {
        return Result<bool, std::string>::makeError("hb_font_create failed");
      }
      hb_font_set_funcs(hbFont_, hbFontFuncs(), &hbContext_, nullptr);
    }
    return Result<bool, std::string>::makeResult(true);
  }

  FontManager &fonts_;
  std::pmr::memory_resource &memory_;
  HashMap<uint64_t, uint32_t> missingGlyphCounts_;
  HashMap<uint64_t, uint32_t> fallbackSwitchCounts_;
  HbFontContext hbContext_{};
  hb_buffer_t *hbBuffer_ = nullptr;
  hb_font_t *hbFont_ = nullptr;
};

} // namespace

std::unique_ptr<TextShaper> TextShaper::create(const CreateDesc &desc) {
  return std::make_unique<TextShaperHb>(desc);
}

} // namespace nuri
