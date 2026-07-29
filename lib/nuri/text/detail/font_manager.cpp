#include "nuri/text/detail/font_manager.h"
#include "nuri/core/containers/hash_map.h"
#include "nuri/core/containers/slot_pool.h"
#include "nuri/core/log.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/resources/storage/cache_utils.h"
#include "nuri/resources/storage/font/nfont_binary_codec.h"
#include "nuri/resources/storage/texture/texture_processing.h"
namespace nuri {
namespace {
template <typename T, typename... Args>
[[nodiscard]] Result<T, std::string> makeError(Args &&...args) {
  std::ostringstream oss;
  (oss << ... << std::forward<Args>(args));
  return Result<T, std::string>::makeError(oss.str());
}
} // namespace

struct FontManager::Impl {
public:
  explicit Impl(const CreateDesc &desc)
      : gpu_(desc.gpu), memory_(desc.memory), fontRecords_(&memory_),
        atlasPageRecords_(&memory_), fontSlots_(&memory_),
        atlasPageSlots_(&memory_) {
    fontRecords_.reserve(desc.initialFontCapacity);
    atlasPageRecords_.reserve(desc.initialAtlasPageCapacity);
    fontSlots_.reserve(desc.initialFontCapacity);
    atlasPageSlots_.reserve(desc.initialAtlasPageCapacity);
  }
  ~Impl() { destroyAll(); }
  Result<FontHandle, std::string> loadFont(const FontLoadDesc &desc) {
    if (desc.path.empty()) {
      return makeError<FontHandle>("FontManager::loadFont: empty path");
    }
    const std::filesystem::path path{std::string(desc.path)};
    auto fileBytesResult = readBinaryFile(path);
    if (fileBytesResult.hasError()) {
      return makeError<FontHandle>("FontManager::loadFont: ",
                                   fileBytesResult.error());
    }
    auto decodedResult = nfontBinaryDeserialize(
        std::span<const std::byte>(fileBytesResult.value()));
    if (decodedResult.hasError()) {
      return makeError<FontHandle>("FontManager::loadFont: failed to parse '",
                                   path.string(), "' (", decodedResult.error(),
                                   ")");
    }
    NFontBinaryData &decoded = decodedResult.value();
    std::pmr::vector<AtlasPageHandle> createdPages(&memory_);
    createdPages.reserve(decoded.atlasPages.size());
    for (size_t pageIndex = 0; pageIndex < decoded.atlasPages.size();
         ++pageIndex) {
      const std::string debugName = buildAtlasDebugName(desc, path, pageIndex);
      auto createPageResult =
          createAtlasPage(decoded.atlasPages[pageIndex], debugName);
      if (createPageResult.hasError()) {
        destroyAtlasPages(createdPages);
        return makeError<FontHandle>("FontManager::loadFont: ",
                                     createPageResult.error());
      }
      createdPages.push_back(createPageResult.value());
    }
    const SlotReservation slot = allocateFontSlot();
    const uint32_t index = slot.index;
    FontRecord &record = fontRecords_[index];
    record.metrics = decoded.metrics;
    record.pxRange = decoded.pxRange;
    record.glyphs.assign(decoded.glyphs.begin(), decoded.glyphs.end());
    record.atlasPages.assign(createdPages.begin(), createdPages.end());
    record.debugName = buildFontDebugName(desc, path);
    record.sourcePath = path.string();
    record.glyphToIndex.clear();
    record.glyphToIndex.reserve(record.glyphs.size());
    for (size_t glyphIndex = 0; glyphIndex < record.glyphs.size();
         ++glyphIndex) {
      const GlyphMetrics &glyph = record.glyphs[glyphIndex];
      record.glyphToIndex.emplace(glyph.glyphId, glyphIndex);
    }
    record.cmap.clear();
    record.cmap.reserve(decoded.cmap.size());
    for (const NFontBinaryCmapEntry &entry : decoded.cmap) {
      record.cmap.emplace(entry.codepoint, entry.glyphId);
    }
    record.fallback.clear();
    const FontHandle handle = FontHandle::fromParts(index, slot.generation);
    NURI_LOG_INFO(
        "FontManager: loaded font '%s' from '%s' (glyphs=%zu pages=%zu)",
        record.debugName.c_str(), record.sourcePath.c_str(),
        record.glyphs.size(), record.atlasPages.size());
    return Result<FontHandle, std::string>::makeResult(handle);
  }
  Result<bool, std::string> unloadFont(FontHandle font) {
    FontRecord *record = resolveFont(font);
    if (record == nullptr) {
      return makeError<bool>("FontManager::unloadFont: invalid font handle");
    }
    const uint32_t index = indexOf(font);
    releaseFontRecord(index);
    return Result<bool, std::string>::makeResult(true);
  }
  bool isValid(FontHandle font) const { return resolveFont(font) != nullptr; }
  FontMetrics metrics(FontHandle font) const {
    return resolveFont(font)->metrics;
  }
  float pxRange(FontHandle font) const { return resolveFont(font)->pxRange; }
  const GlyphMetrics *findGlyph(FontHandle font, GlyphId glyph) const {
    const FontRecord *record = resolveFont(font);
    const auto it = record->glyphToIndex.find(glyph);
    if (it == record->glyphToIndex.end()) {
      return nullptr;
    }
    return &record->glyphs[it->second];
  }
  GlyphId lookupGlyphForCodepoint(FontHandle font, uint32_t codepoint) const {
    const FontRecord *record = resolveFont(font);
    const auto it = record->cmap.find(codepoint);
    if (it == record->cmap.end()) {
      return 0;
    }
    return it->second;
  }
  std::span<const FontHandle> fallbackChain(FontHandle font) const {
    const FontRecord *record = resolveFont(font);
    return record->fallback;
  }
  AtlasPageHandle resolveAtlasPage(FontHandle font,
                                   uint16_t localPageIndex) const {
    const FontRecord *record = resolveFont(font);
    return record->atlasPages[localPageIndex];
  }
  GlyphLookupResult resolveGlyph(FontHandle font, GlyphId glyph) const {
    const FontRecord *record = resolveFont(font);
    const auto it = record->glyphToIndex.find(glyph);
    if (it == record->glyphToIndex.end()) {
      return {};
    }
    const GlyphMetrics &m = record->glyphs[it->second];
    return {&m, record->atlasPages[m.localPageIndex]};
  }
  TextureHandle atlasTexture(AtlasPageHandle page) const {
    return resolveAtlasPageRecord(page)->texture;
  }
  uint32_t atlasBindlessIndex(AtlasPageHandle page) const {
    return resolveAtlasPageRecord(page)->bindlessIndex;
  }
  Result<bool, std::string>
  setFallbackChain(FontHandle font, std::span<const FontHandle> chain) {
    FontRecord *record = resolveFont(font);
    if (record == nullptr) {
      return makeError<bool>(
          "FontManager::setFallbackChain: invalid font handle");
    }
    std::pmr::vector<FontHandle> pending(&memory_);
    PmrHashMap<uint32_t, bool> visited(&memory_);
    for (const FontHandle chainFont : chain) {
      if (chainFont.value == font.value) {
        return makeError<bool>("FontManager::setFallbackChain: chain contains "
                               "direct self-reference");
      }
      if (resolveFont(chainFont) == nullptr) {
        return makeError<bool>("FontManager::setFallbackChain: chain contains "
                               "invalid font handle");
      }
      pending.clear();
      visited.clear();
      pending.push_back(chainFont);
      visited.emplace(chainFont.value, true);
      while (!pending.empty()) {
        const FontHandle current = pending.back();
        pending.pop_back();
        const FontRecord *currentRecord = resolveFont(current);
        for (const FontHandle fallbackFont : currentRecord->fallback) {
          if (fallbackFont.value == font.value) {
            return makeError<bool>("FontManager::setFallbackChain: chain "
                                   "introduces fallback cycle");
          }
          if (visited.emplace(fallbackFont.value, true).second) {
            pending.push_back(fallbackFont);
          }
        }
      }
    }
    record->fallback.assign(chain.begin(), chain.end());
    return Result<bool, std::string>::makeResult(true);
  }

private:
  struct FontRecord {
    FontMetrics metrics{};
    float pxRange = 4.0f;
    std::pmr::vector<GlyphMetrics> glyphs;
    PmrHashMap<GlyphId, size_t> glyphToIndex;
    PmrHashMap<uint32_t, GlyphId> cmap;
    std::pmr::vector<AtlasPageHandle> atlasPages;
    std::pmr::vector<FontHandle> fallback;
    std::pmr::string debugName;
    std::pmr::string sourcePath;
    explicit FontRecord(std::pmr::memory_resource *memory)
        : glyphs(memory), glyphToIndex(memory), cmap(memory),
          atlasPages(memory), fallback(memory), debugName(memory),
          sourcePath(memory) {}
  };
  struct AtlasPageRecord {
    TextureHandle texture{};
    uint32_t bindlessIndex = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    std::pmr::string debugName;
    explicit AtlasPageRecord(std::pmr::memory_resource *memory)
        : debugName(memory) {}
  };
  [[nodiscard]] SlotReservation allocateFontSlot() {
    const SlotReservation slot = fontSlots_.acquire();
    if (slot.appended) {
      fontRecords_.emplace_back(&memory_);
    }
    return slot;
  }
  [[nodiscard]] SlotReservation allocateAtlasPageSlot() {
    const SlotReservation slot = atlasPageSlots_.acquire();
    if (slot.appended) {
      atlasPageRecords_.emplace_back(&memory_);
    }
    return slot;
  }
  [[nodiscard]] Result<AtlasPageHandle, std::string>
  createAtlasPage(const NFontBinaryAtlasImage &page,
                  std::string_view debugName) {
    const uint64_t pixelCount =
        static_cast<uint64_t>(page.width) * static_cast<uint64_t>(page.height);
    const uint64_t bytesPerPixel =
        static_cast<uint64_t>(page.imageBytes.size()) / pixelCount;
    const Format format =
        bytesPerPixel == 4 ? Format::RGBA8_UNORM : Format::RGBA16_FLOAT;
    TextureDesc textureDesc{};
    textureDesc.type = TextureType::Texture2D;
    textureDesc.format = format;
    textureDesc.dimensions = TextureDimensions{page.width, page.height, 1};
    textureDesc.usage = TextureUsage::Sampled;
    textureDesc.storage = Storage::Device;
    textureDesc.numLayers = 1;
    textureDesc.numSamples = 1;
    textureDesc.numMipLevels = textureMipLevelCount(page.width, page.height);
    textureDesc.data = std::span<const std::byte>(page.imageBytes.data(),
                                                  page.imageBytes.size());
    textureDesc.dataNumMipLevels = 1;
    textureDesc.generateMipmaps = textureDesc.numMipLevels > 1;
    auto createTextureResult = gpu_.createTexture(textureDesc, debugName);
    if (createTextureResult.hasError()) {
      return makeError<AtlasPageHandle>(
          "FontManager::loadFont: failed to create atlas texture '", debugName,
          "' (", createTextureResult.error(), ")");
    }
    const TextureHandle textureHandle = createTextureResult.value();
    const SlotReservation slot = allocateAtlasPageSlot();
    const uint32_t index = slot.index;
    AtlasPageRecord &record = atlasPageRecords_[index];
    record.texture = textureHandle;
    record.bindlessIndex = gpu_.getTextureBindlessIndex(textureHandle);
    record.width = page.width;
    record.height = page.height;
    record.debugName = debugName;
    return Result<AtlasPageHandle, std::string>::makeResult(
        AtlasPageHandle::fromParts(index, slot.generation));
  }
  void destroyAtlasPages(std::span<const AtlasPageHandle> pages) {
    for (const AtlasPageHandle pageHandle : pages) {
      releaseAtlasPageRecord(pageHandle);
    }
  }
  void releaseFontRecord(uint32_t index) {
    FontRecord &record = fontRecords_[index];
    destroyAtlasPages(record.atlasPages);
    record.metrics = FontMetrics{};
    record.pxRange = 4.0f;
    record.glyphs.clear();
    record.glyphToIndex.clear();
    record.cmap.clear();
    record.atlasPages.clear();
    record.fallback.clear();
    record.debugName.clear();
    record.sourcePath.clear();
    fontSlots_.release(index);
  }
  void releaseAtlasPageRecord(AtlasPageHandle page) {
    const uint32_t index = indexOf(page);
    AtlasPageRecord &record = atlasPageRecords_[index];
    gpu_.destroyTexture(record.texture);
    record.texture = TextureHandle{};
    record.bindlessIndex = 0;
    record.width = 0;
    record.height = 0;
    record.debugName.clear();
    atlasPageSlots_.release(index);
  }
  void destroyAll() {
    for (size_t fontIndex = 0; fontIndex < fontRecords_.size(); ++fontIndex) {
      if (fontSlots_.isLive(static_cast<uint32_t>(fontIndex))) {
        releaseFontRecord(static_cast<uint32_t>(fontIndex));
      }
    }
  }
  [[nodiscard]] FontRecord *resolveFont(FontHandle font) {
    const uint32_t index = indexOf(font);
    const uint32_t generation = generationOf(font);
    if (!fontSlots_.isValid(index, generation)) {
      return nullptr;
    }
    return &fontRecords_[index];
  }
  [[nodiscard]] const FontRecord *resolveFont(FontHandle font) const {
    const uint32_t index = indexOf(font);
    const uint32_t generation = generationOf(font);
    if (!fontSlots_.isValid(index, generation)) {
      return nullptr;
    }
    return &fontRecords_[index];
  }
  [[nodiscard]] const AtlasPageRecord *
  resolveAtlasPageRecord(AtlasPageHandle page) const {
    const uint32_t index = indexOf(page);
    const uint32_t generation = generationOf(page);
    if (!atlasPageSlots_.isValid(index, generation)) {
      return nullptr;
    }
    return &atlasPageRecords_[index];
  }
  [[nodiscard]] std::string
  buildFontDebugName(const FontLoadDesc &desc,
                     const std::filesystem::path &path) const {
    if (!desc.debugName.empty()) {
      return std::string(desc.debugName);
    }
    const std::string stem = path.stem().string();
    return stem.empty() ? path.filename().string() : stem;
  }
  [[nodiscard]] std::string
  buildAtlasDebugName(const FontLoadDesc &desc,
                      const std::filesystem::path &path,
                      size_t pageIndex) const {
    std::ostringstream oss;
    oss << buildFontDebugName(desc, path) << "_atlas_" << pageIndex;
    return oss.str();
  }
  GPUDevice &gpu_;
  std::pmr::memory_resource &memory_;
  std::pmr::vector<FontRecord> fontRecords_;
  std::pmr::vector<AtlasPageRecord> atlasPageRecords_;
  SlotPool<MaskedNonZeroGenerationPolicy<kResourceHandleGenerationMask>>
      fontSlots_;
  SlotPool<MaskedNonZeroGenerationPolicy<kResourceHandleGenerationMask>>
      atlasPageSlots_;
};

FontManager::FontManager(const CreateDesc &desc)
    : impl_(std::make_unique<Impl>(desc)) {}
FontManager::~FontManager() = default;
Result<FontHandle, std::string>
FontManager::loadFont(const FontLoadDesc &desc) {
  return impl_->loadFont(desc);
}
Result<bool, std::string> FontManager::unloadFont(FontHandle font) {
  return impl_->unloadFont(font);
}
bool FontManager::isValid(FontHandle font) const {
  return impl_->isValid(font);
}
FontMetrics FontManager::metrics(FontHandle font) const {
  return impl_->metrics(font);
}
float FontManager::pxRange(FontHandle font) const {
  return impl_->pxRange(font);
}
const GlyphMetrics *FontManager::findGlyph(FontHandle font,
                                           GlyphId glyph) const {
  return impl_->findGlyph(font, glyph);
}
GlyphId FontManager::lookupGlyphForCodepoint(FontHandle font,
                                             uint32_t codepoint) const {
  return impl_->lookupGlyphForCodepoint(font, codepoint);
}
std::span<const FontHandle> FontManager::fallbackChain(FontHandle font) const {
  return impl_->fallbackChain(font);
}
AtlasPageHandle FontManager::resolveAtlasPage(FontHandle font,
                                              uint16_t localPageIndex) const {
  return impl_->resolveAtlasPage(font, localPageIndex);
}
GlyphLookupResult FontManager::resolveGlyph(FontHandle font,
                                            GlyphId glyph) const {
  return impl_->resolveGlyph(font, glyph);
}
TextureHandle FontManager::atlasTexture(AtlasPageHandle page) const {
  return impl_->atlasTexture(page);
}
uint32_t FontManager::atlasBindlessIndex(AtlasPageHandle page) const {
  return impl_->atlasBindlessIndex(page);
}
Result<bool, std::string>
FontManager::setFallbackChain(FontHandle font,
                              std::span<const FontHandle> chain) {
  return impl_->setFallbackChain(font, chain);
}

} // namespace nuri
