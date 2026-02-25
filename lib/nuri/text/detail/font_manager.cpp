#include "nuri/pch.h"

#include "nuri/text/font_manager.h"

#include "nuri/core/log.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/resources/storage/font/nfont_binary_codec.h"

namespace nuri {
namespace {

template <typename T, typename... Args>
[[nodiscard]] Result<T, std::string> makeError(Args &&...args) {
  std::ostringstream oss;
  (oss << ... << std::forward<Args>(args));
  return Result<T, std::string>::makeError(oss.str());
}

[[nodiscard]] uint32_t bumpTextHandleGeneration(uint32_t generation) {
  const uint32_t next = (generation + 1u) & kTextHandleGenerationMask;
  return next == 0u ? 1u : next;
}

[[nodiscard]] Result<std::vector<std::byte>, std::string>
readBinaryFile(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    return makeError<std::vector<std::byte>>(
        "FontManager: failed to open file '", path.string(), "'");
  }

  const std::streampos endPos = input.tellg();
  if (endPos < 0) {
    return makeError<std::vector<std::byte>>(
        "FontManager: failed to query file size for '", path.string(), "'");
  }

  const uint64_t size64 = static_cast<uint64_t>(endPos);
  if (size64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return makeError<std::vector<std::byte>>("FontManager: file too large '",
                                             path.string(), "'");
  }

  input.seekg(0, std::ios::beg);
  if (!input) {
    return makeError<std::vector<std::byte>>(
        "FontManager: failed to seek file '", path.string(), "'");
  }

  std::vector<std::byte> data(static_cast<size_t>(size64));
  if (!data.empty()) {
    input.read(reinterpret_cast<char *>(data.data()),
               static_cast<std::streamsize>(data.size()));
    if (!input) {
      return makeError<std::vector<std::byte>>(
          "FontManager: failed to read file '", path.string(), "'");
    }
  }

  return Result<std::vector<std::byte>, std::string>::makeResult(std::move(data));
}

[[nodiscard]] uint32_t computeMipLevels2D(uint32_t width, uint32_t height) {
  uint32_t levels = 1;
  uint32_t w = std::max(width, 1u);
  uint32_t h = std::max(height, 1u);
  while (w > 1u || h > 1u) {
    w = std::max(1u, w >> 1u);
    h = std::max(1u, h >> 1u);
    ++levels;
  }
  return levels;
}

class FontManagerImpl final : public FontManager {
public:
  explicit FontManagerImpl(const CreateDesc &desc)
      : gpu_(desc.gpu), memory_(desc.memory), fontRecords_(&memory_),
        atlasPageRecords_(&memory_), fontFreeList_(&memory_),
        atlasPageFreeList_(&memory_), retiredTextures_(&memory_) {
    fontRecords_.reserve(desc.initialFontCapacity);
    atlasPageRecords_.reserve(desc.initialAtlasPageCapacity);
    fontFreeList_.reserve(desc.initialFontCapacity);
    atlasPageFreeList_.reserve(desc.initialAtlasPageCapacity);
    retiredTextures_.reserve(desc.initialAtlasPageCapacity);
    gcSafetyLag_ = std::max<uint64_t>(1u, gpu_.getSwapchainImageCount());
  }

  ~FontManagerImpl() override { destroyAll(); }

  Result<FontHandle, std::string> loadFont(const FontLoadDesc &desc) override {
    if (desc.path.empty()) {
      return makeError<FontHandle>("FontManager::loadFont: empty path");
    }

    const std::filesystem::path path{std::string(desc.path)};
    auto fileBytesResult = readBinaryFile(path);
    if (fileBytesResult.hasError()) {
      return makeError<FontHandle>("FontManager::loadFont: ",
                                   fileBytesResult.error());
    }

    auto decodedResult =
        nfontBinaryDeserialize(std::span<const std::byte>(fileBytesResult.value()));
    if (decodedResult.hasError()) {
      return makeError<FontHandle>("FontManager::loadFont: failed to parse '",
                                   path.string(), "' (",
                                   decodedResult.error(), ")");
    }

    NFontBinaryData &decoded = decodedResult.value();

    std::pmr::vector<AtlasPageHandle> createdPages(&memory_);
    createdPages.reserve(decoded.atlasPages.size());
    for (size_t pageIndex = 0; pageIndex < decoded.atlasPages.size(); ++pageIndex) {
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

    auto indexResult = allocateFontSlot();
    if (indexResult.hasError()) {
      destroyAtlasPages(createdPages);
      return makeError<FontHandle>("FontManager::loadFont: ",
                                   indexResult.error());
    }

    const uint32_t index = indexResult.value();
    FontRecord &record = fontRecords_[index];

    record.live = true;
    record.metrics = decoded.metrics;
    record.pxRange = decoded.pxRange;
    record.glyphs.assign(decoded.glyphs.begin(), decoded.glyphs.end());
    record.atlasPages.assign(createdPages.begin(), createdPages.end());
    record.debugName = buildFontDebugName(desc, path);
    record.sourcePath = path.string();
    record.glyphToIndex.clear();
    record.glyphToIndex.reserve(record.glyphs.size());
    for (size_t glyphIndex = 0; glyphIndex < record.glyphs.size(); ++glyphIndex) {
      const GlyphMetrics &glyph = record.glyphs[glyphIndex];
      if (glyph.localPageIndex >= record.atlasPages.size()) {
        releaseFontRecord(index);
        destroyAtlasPages(createdPages);
        return makeError<FontHandle>(
            "FontManager::loadFont: glyph page index out of range in '",
            path.string(), "'");
      }
      const auto [_, inserted] =
          record.glyphToIndex.emplace(glyph.glyphId, glyphIndex);
      if (!inserted) {
        releaseFontRecord(index);
        destroyAtlasPages(createdPages);
        return makeError<FontHandle>(
            "FontManager::loadFont: duplicate glyph id ", glyph.glyphId,
            " in '", path.string(), "'");
      }
    }

    record.cmap.clear();
    record.cmap.reserve(decoded.cmap.size());
    for (const NFontBinaryCmapEntry &entry : decoded.cmap) {
      const auto inserted = record.cmap.emplace(entry.codepoint, entry.glyphId);
      if (!inserted.second) {
        releaseFontRecord(index);
        destroyAtlasPages(createdPages);
        return makeError<FontHandle>(
            "FontManager::loadFont: duplicate codepoint ", entry.codepoint,
            " in '", path.string(), "'");
      }
    }

    record.fallback.clear();
    ++liveFonts_;
    const FontHandle handle{
        .value = packTextHandleValue(index, record.generation),
    };

    NURI_LOG_INFO("FontManager: loaded font '%s' from '%s' (glyphs=%zu pages=%zu)",
                  record.debugName.c_str(), record.sourcePath.c_str(),
                  record.glyphs.size(), record.atlasPages.size());
    return Result<FontHandle, std::string>::makeResult(handle);
  }

  Result<bool, std::string> unloadFont(FontHandle font) override {
    FontRecord *record = resolveFont(font);
    if (record == nullptr) {
      return makeError<bool>("FontManager::unloadFont: invalid font handle");
    }

    const uint32_t index = textHandleIndex(font.value);
    releaseFontRecord(index);
    --liveFonts_;
    return Result<bool, std::string>::makeResult(true);
  }

  bool isValid(FontHandle font) const override {
    return resolveFont(font) != nullptr;
  }

  FontMetrics metrics(FontHandle font) const override {
    const FontRecord *record = resolveFont(font);
    if (record == nullptr) {
      return FontMetrics{};
    }
    return record->metrics;
  }

  float pxRange(FontHandle font) const override {
    const FontRecord *record = resolveFont(font);
    if (record == nullptr) {
      return 4.0f;
    }
    return record->pxRange;
  }

  const GlyphMetrics *findGlyph(FontHandle font, GlyphId glyph) const override {
    const FontRecord *record = resolveFont(font);
    if (record == nullptr) {
      return nullptr;
    }
    const auto it = record->glyphToIndex.find(glyph);
    if (it == record->glyphToIndex.end()) {
      return nullptr;
    }
    return &record->glyphs[it->second];
  }

  GlyphId lookupGlyphForCodepoint(FontHandle font,
                                  uint32_t codepoint) const override {
    const FontRecord *record = resolveFont(font);
    if (record == nullptr) {
      return 0;
    }
    const auto it = record->cmap.find(codepoint);
    if (it == record->cmap.end()) {
      return 0;
    }
    return it->second;
  }

  std::span<const FontHandle> fallbackChain(FontHandle font) const override {
    const FontRecord *record = resolveFont(font);
    if (record == nullptr || record->fallback.empty()) {
      return {};
    }
    return std::span<const FontHandle>(record->fallback.data(),
                                       record->fallback.size());
  }

  AtlasPageHandle resolveAtlasPage(FontHandle font,
                                   uint16_t localPageIndex) const override {
    const FontRecord *record = resolveFont(font);
    if (record == nullptr || localPageIndex >= record->atlasPages.size()) {
      return kInvalidAtlasPageHandle;
    }
    return record->atlasPages[localPageIndex];
  }

  TextureHandle atlasTexture(AtlasPageHandle page) const override {
    const AtlasPageRecord *record = resolveAtlasPageRecord(page);
    if (record == nullptr) {
      return TextureHandle{};
    }
    return record->texture;
  }

  uint32_t atlasBindlessIndex(AtlasPageHandle page) const override {
    const AtlasPageRecord *record = resolveAtlasPageRecord(page);
    if (record == nullptr) {
      return 0;
    }
    return record->bindlessIndex;
  }

  Result<bool, std::string>
  setFallbackChain(FontHandle font, std::span<const FontHandle> chain) override {
    FontRecord *record = resolveFont(font);
    if (record == nullptr) {
      return makeError<bool>("FontManager::setFallbackChain: invalid font handle");
    }

    for (const FontHandle chainFont : chain) {
      if (resolveFont(chainFont) == nullptr) {
        return makeError<bool>(
            "FontManager::setFallbackChain: chain contains invalid font handle");
      }
    }

    record->fallback.assign(chain.begin(), chain.end());
    return Result<bool, std::string>::makeResult(true);
  }

  void collectGarbage(uint64_t completedTimelineValue) override {
    lastCollectedTimelineValue_ =
        std::max(lastCollectedTimelineValue_, completedTimelineValue);
    if (retiredTextures_.empty()) {
      return;
    }

    size_t writeIndex = 0;
    uint32_t destroyed = 0;
    for (size_t i = 0; i < retiredTextures_.size(); ++i) {
      RetiredTextureRecord &record = retiredTextures_[i];
      if (record.retireAfterValue <= lastCollectedTimelineValue_) {
        if (::nuri::isValid(record.texture)) {
          gpu_.destroyTexture(record.texture);
        }
        ++destroyed;
        continue;
      }
      if (writeIndex != i) {
        retiredTextures_[writeIndex] = std::move(record);
      }
      ++writeIndex;
    }
    retiredTextures_.resize(writeIndex);

    if (destroyed > 0) {
      NURI_LOG_DEBUG(
          "FontManager: garbage-collected %u retired atlas textures (remaining=%zu)",
          destroyed, retiredTextures_.size());
    }
  }

  PoolStats poolStats() const override {
    return PoolStats{
        .liveFonts = liveFonts_,
        .liveAtlasPages = liveAtlasPages_,
        .liveShaperFaces = 0,
    };
  }

private:
  struct FontRecord {
    bool live = false;
    uint32_t generation = 0;
    FontMetrics metrics{};
    float pxRange = 4.0f;
    std::pmr::vector<GlyphMetrics> glyphs;
    std::pmr::unordered_map<GlyphId, size_t> glyphToIndex;
    std::pmr::unordered_map<uint32_t, GlyphId> cmap;
    std::pmr::vector<AtlasPageHandle> atlasPages;
    std::pmr::vector<FontHandle> fallback;
    std::pmr::string debugName;
    std::pmr::string sourcePath;

    explicit FontRecord(std::pmr::memory_resource *memory)
        : glyphs(memory), glyphToIndex(memory), cmap(memory), atlasPages(memory),
          fallback(memory), debugName(memory), sourcePath(memory) {}
  };

  struct AtlasPageRecord {
    bool live = false;
    uint32_t generation = 0;
    TextureHandle texture{};
    uint32_t bindlessIndex = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    std::pmr::string debugName;

    explicit AtlasPageRecord(std::pmr::memory_resource *memory)
        : debugName(memory) {}
  };

  struct RetiredTextureRecord {
    TextureHandle texture{};
    uint64_t retireAfterValue = 0;
  };

  [[nodiscard]] Result<uint32_t, std::string> allocateFontSlot() {
    uint32_t index = 0;
    if (!fontFreeList_.empty()) {
      index = fontFreeList_.back();
      fontFreeList_.pop_back();
    } else {
      if (fontRecords_.size() > kTextHandleIndexMask) {
        return makeError<uint32_t>("FontManager: font pool exhausted");
      }
      index = static_cast<uint32_t>(fontRecords_.size());
      fontRecords_.emplace_back(&memory_);
    }

    FontRecord &record = fontRecords_[index];
    record.generation = bumpTextHandleGeneration(record.generation);
    return Result<uint32_t, std::string>::makeResult(index);
  }

  [[nodiscard]] Result<uint32_t, std::string> allocateAtlasPageSlot() {
    uint32_t index = 0;
    if (!atlasPageFreeList_.empty()) {
      index = atlasPageFreeList_.back();
      atlasPageFreeList_.pop_back();
    } else {
      if (atlasPageRecords_.size() > kTextHandleIndexMask) {
        return makeError<uint32_t>("FontManager: atlas page pool exhausted");
      }
      index = static_cast<uint32_t>(atlasPageRecords_.size());
      atlasPageRecords_.emplace_back(&memory_);
    }

    AtlasPageRecord &record = atlasPageRecords_[index];
    record.generation = bumpTextHandleGeneration(record.generation);
    return Result<uint32_t, std::string>::makeResult(index);
  }

  [[nodiscard]] Result<AtlasPageHandle, std::string>
  createAtlasPage(const NFontBinaryAtlasImage &page, std::string_view debugName) {
    if (page.width == 0 || page.height == 0) {
      return makeError<AtlasPageHandle>(
          "FontManager::loadFont: atlas page has invalid dimensions");
    }

    const uint64_t pixelCount =
        static_cast<uint64_t>(page.width) * static_cast<uint64_t>(page.height);
    if (pixelCount == 0) {
      return makeError<AtlasPageHandle>(
          "FontManager::loadFont: atlas page pixel count is zero");
    }
    if (page.imageBytes.size() % pixelCount != 0) {
      return makeError<AtlasPageHandle>(
          "FontManager::loadFont: atlas image size is not divisible by pixel count");
    }

    const uint64_t bytesPerPixel =
        static_cast<uint64_t>(page.imageBytes.size()) / pixelCount;
    Format format = Format::Count;
    if (bytesPerPixel == 4) {
      format = Format::RGBA8_UNORM;
    } else if (bytesPerPixel == 8) {
      format = Format::RGBA16_FLOAT;
    } else {
      return makeError<AtlasPageHandle>(
          "FontManager::loadFont: unsupported atlas bytes-per-pixel ",
          bytesPerPixel, " (expected 4 or 8)");
    }

    TextureDesc textureDesc{};
    textureDesc.type = TextureType::Texture2D;
    textureDesc.format = format;
    textureDesc.dimensions = TextureDimensions{page.width, page.height, 1};
    textureDesc.usage = TextureUsage::Sampled;
    textureDesc.storage = Storage::Device;
    textureDesc.numLayers = 1;
    textureDesc.numSamples = 1;
    textureDesc.numMipLevels = computeMipLevels2D(page.width, page.height);
    textureDesc.data =
        std::span<const std::byte>(page.imageBytes.data(), page.imageBytes.size());
    textureDesc.dataNumMipLevels = 1;
    textureDesc.generateMipmaps = textureDesc.numMipLevels > 1;

    auto createTextureResult = gpu_.createTexture(textureDesc, debugName);
    if (createTextureResult.hasError()) {
      return makeError<AtlasPageHandle>(
          "FontManager::loadFont: failed to create atlas texture '", debugName,
          "' (", createTextureResult.error(), ")");
    }

    const TextureHandle textureHandle = createTextureResult.value();
    auto indexResult = allocateAtlasPageSlot();
    if (indexResult.hasError()) {
      gpu_.destroyTexture(textureHandle);
      return makeError<AtlasPageHandle>("FontManager::loadFont: ",
                                        indexResult.error());
    }

    const uint32_t index = indexResult.value();
    AtlasPageRecord &record = atlasPageRecords_[index];
    record.live = true;
    record.texture = textureHandle;
    record.bindlessIndex = gpu_.getTextureBindlessIndex(textureHandle);
    record.width = page.width;
    record.height = page.height;
    record.debugName = debugName;
    ++liveAtlasPages_;

    return Result<AtlasPageHandle, std::string>::makeResult(AtlasPageHandle{
        .value = packTextHandleValue(index, record.generation),
    });
  }

  void destroyAtlasPages(std::span<const AtlasPageHandle> pages) {
    for (const AtlasPageHandle pageHandle : pages) {
      releaseAtlasPageRecord(pageHandle);
    }
  }

  void releaseFontRecord(uint32_t index) {
    if (index >= fontRecords_.size()) {
      return;
    }

    FontRecord &record = fontRecords_[index];
    if (!record.live) {
      return;
    }

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
    record.live = false;
    fontFreeList_.push_back(index);
  }

  void releaseAtlasPageRecord(AtlasPageHandle page) {
    const uint32_t index = textHandleIndex(page.value);
    const uint32_t generation = textHandleGeneration(page.value);
    if (index >= atlasPageRecords_.size()) {
      return;
    }

    AtlasPageRecord &record = atlasPageRecords_[index];
    if (!record.live || record.generation != generation) {
      return;
    }

    retireTexture(record.texture);
    record.texture = TextureHandle{};
    record.bindlessIndex = 0;
    record.width = 0;
    record.height = 0;
    record.debugName.clear();
    record.live = false;
    atlasPageFreeList_.push_back(index);
    if (liveAtlasPages_ > 0) {
      --liveAtlasPages_;
    }
  }

  void retireTexture(TextureHandle texture) {
    if (!::nuri::isValid(texture)) {
      return;
    }
    const uint64_t retireAfter =
        lastCollectedTimelineValue_ + std::max<uint64_t>(1u, gcSafetyLag_);
    retiredTextures_.push_back(RetiredTextureRecord{
        .texture = texture,
        .retireAfterValue = retireAfter,
    });
  }

  void destroyAll() {
    for (size_t fontIndex = 0; fontIndex < fontRecords_.size(); ++fontIndex) {
      if (fontRecords_[fontIndex].live) {
        releaseFontRecord(static_cast<uint32_t>(fontIndex));
      }
    }
    liveFonts_ = 0;
    collectGarbage(std::numeric_limits<uint64_t>::max());
  }

  [[nodiscard]] FontRecord *resolveFont(FontHandle font) {
    if (!::nuri::isValid(font)) {
      return nullptr;
    }

    const uint32_t index = textHandleIndex(font.value);
    const uint32_t generation = textHandleGeneration(font.value);
    if (index >= fontRecords_.size()) {
      return nullptr;
    }

    FontRecord &record = fontRecords_[index];
    if (!record.live || record.generation != generation) {
      return nullptr;
    }
    return &record;
  }

  [[nodiscard]] const FontRecord *resolveFont(FontHandle font) const {
    if (!::nuri::isValid(font)) {
      return nullptr;
    }

    const uint32_t index = textHandleIndex(font.value);
    const uint32_t generation = textHandleGeneration(font.value);
    if (index >= fontRecords_.size()) {
      return nullptr;
    }

    const FontRecord &record = fontRecords_[index];
    if (!record.live || record.generation != generation) {
      return nullptr;
    }
    return &record;
  }

  [[nodiscard]] const AtlasPageRecord *
  resolveAtlasPageRecord(AtlasPageHandle page) const {
    if (!::nuri::isValid(page)) {
      return nullptr;
    }

    const uint32_t index = textHandleIndex(page.value);
    const uint32_t generation = textHandleGeneration(page.value);
    if (index >= atlasPageRecords_.size()) {
      return nullptr;
    }

    const AtlasPageRecord &record = atlasPageRecords_[index];
    if (!record.live || record.generation != generation) {
      return nullptr;
    }
    return &record;
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
  buildAtlasDebugName(const FontLoadDesc &desc, const std::filesystem::path &path,
                      size_t pageIndex) const {
    std::ostringstream oss;
    oss << buildFontDebugName(desc, path) << "_atlas_" << pageIndex;
    return oss.str();
  }

  GPUDevice &gpu_;
  std::pmr::memory_resource &memory_;
  std::pmr::vector<FontRecord> fontRecords_;
  std::pmr::vector<AtlasPageRecord> atlasPageRecords_;
  std::pmr::vector<uint32_t> fontFreeList_;
  std::pmr::vector<uint32_t> atlasPageFreeList_;
  std::pmr::vector<RetiredTextureRecord> retiredTextures_;
  uint64_t lastCollectedTimelineValue_ = 0;
  uint64_t gcSafetyLag_ = 1;
  uint32_t liveFonts_ = 0;
  uint32_t liveAtlasPages_ = 0;
};

} // namespace

std::unique_ptr<FontManager> FontManager::create(const CreateDesc &desc) {
  return std::make_unique<FontManagerImpl>(desc);
}

} // namespace nuri
