#pragma once

#include "nuri/core/containers/hash_map.h"
#include "nuri/core/result.h"
#include "nuri/text/text_shaper.h"

#include <memory_resource>
#include <string>

namespace nuri {

struct LayoutGlyph {
  FontHandle font = kInvalidFontHandle;
  GlyphId glyphId = 0;
  AtlasPageHandle atlasPage = kInvalidAtlasPageHandle;
  float x = 0.0f;
  float y = 0.0f;
  GlyphMetrics metrics{};
};

struct TextLayout {
  std::pmr::vector<LayoutGlyph> glyphs;
  TextBounds bounds{};
  uint32_t lineCount = 0;
};

class NURI_API TextLayouter {
public:
  struct CreateDesc {
    FontManager &fonts;
    TextShaper &shaper;
    std::pmr::memory_resource &memory;
  };

  explicit TextLayouter(const CreateDesc &desc);
  ~TextLayouter() = default;

  TextLayouter(const TextLayouter &) = delete;
  TextLayouter &operator=(const TextLayouter &) = delete;
  TextLayouter(TextLayouter &&) = delete;
  TextLayouter &operator=(TextLayouter &&) = delete;

  Result<TextLayout, std::string>
  layoutUtf8(std::string_view utf8, const TextStyle &style,
             const TextLayoutParams &params,
             std::pmr::memory_resource &outMemory,
             std::pmr::memory_resource &scratch);

private:
  static constexpr uint32_t kNoSlot = UINT32_MAX;
  static constexpr size_t kMaxCacheEntries = 256;

  struct IdentityHash {
    using is_avalanching = void;
    [[nodiscard]] uint64_t operator()(uint64_t k) const noexcept { return k; }
  };

  struct CacheEntry {
    uint64_t hash = 0;
    uint32_t lruPrev = kNoSlot;
    uint32_t lruNext = kNoSlot;
    std::pmr::string utf8;
    FontHandle font = kInvalidFontHandle;
    float pxSize = 0.0f;
    float lineHeightScale = 1.0f;
    float letterSpacing = 0.0f;
    float wordSpacing = 0.0f;
    std::pmr::vector<FontHandle> fallbackHandles;
    TextLayoutParams params{};
    TextLayout layout;

    explicit CacheEntry(std::pmr::memory_resource *memory);
  };

  void lruRemove(uint32_t idx);
  void lruPushFront(uint32_t idx);
  void lruPromote(uint32_t idx);
  uint32_t allocateSlot();
  void insertIntoCache(uint64_t keyHash, uint32_t slot);
  static void fillCacheKey(CacheEntry &entry, uint64_t hash,
                           std::string_view utf8, const TextStyle &style,
                           const TextLayoutParams &params);
  static bool cacheKeyEquals(const CacheEntry &entry, std::string_view utf8,
                             const TextStyle &style,
                             const TextLayoutParams &params);

  FontManager &fonts_;
  TextShaper &shaper_;
  std::pmr::memory_resource &memory_;
  std::pmr::vector<CacheEntry> pool_;
  HashMap<uint64_t, uint32_t, IdentityHash> cacheMap_;
  uint32_t lruHead_ = kNoSlot;
  uint32_t lruTail_ = kNoSlot;
  uint32_t freeHead_ = kNoSlot;
};

} // namespace nuri
