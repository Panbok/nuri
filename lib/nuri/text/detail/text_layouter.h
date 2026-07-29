#pragma once
#include "nuri/core/containers/hash_map.h"
#include "nuri/core/result.h"
#include "nuri/text/detail/text_shaper.h"
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

class TextLayouter {
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
  Result<const TextLayout *, std::string>
  layoutUtf8(std::string_view utf8, const TextStyle &style,
             const TextLayoutParams &params,
             std::pmr::memory_resource &scratch);

private:
  static constexpr size_t kMaxCacheEntries = 256;
  struct IdentityHash {
    using is_avalanching = void;
    [[nodiscard]] uint64_t operator()(uint64_t k) const noexcept { return k; }
  };
  struct CacheEntry {
    uint64_t hash = 0;
    uint64_t lastUse = 0;
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
  uint64_t useCounter_ = 0;
};

} // namespace nuri
