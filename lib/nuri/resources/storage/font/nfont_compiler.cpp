#include "nuri/pch.h"

#include "nuri/resources/storage/font/nfont_compiler.h"

#include "nuri/resources/storage/font/nfont_binary_codec.h"

#include <msdf-atlas-gen/msdf-atlas-gen.h>

#include <cstddef>

namespace nuri {
namespace {

// Latin-1 (ISO-8859-1) code point range: space through U+00FF
constexpr uint32_t kLatin1FirstCodePoint = 0x20u; // space, first printable
constexpr uint32_t kLatin1LastCodePoint = 0x00FFu;

template <typename T, typename... Args>
[[nodiscard]] Result<T, std::string> makeError(Args &&...args) {
  std::ostringstream oss;
  (oss << ... << std::forward<Args>(args));
  return Result<T, std::string>::makeError(oss.str());
}

[[nodiscard]] msdf_atlas::Charset buildCharset(NFontCharsetPreset preset) {
  if (preset == NFontCharsetPreset::Ascii) {
    return msdf_atlas::Charset::ASCII;
  }

  msdf_atlas::Charset charset;
  for (uint32_t cp = kLatin1FirstCodePoint; cp <= kLatin1LastCodePoint; ++cp) {
    charset.add(cp);
  }
  return charset;
}

[[nodiscard]] std::vector<std::byte>
convertAtlasToRgba8(const msdfgen::BitmapConstRef<float, 4> &atlas) {
  const int w = atlas.width;
  const int h = atlas.height;
  std::vector<std::byte> out(static_cast<size_t>(w) * static_cast<size_t>(h) *
                             4u);

  auto *dst = reinterpret_cast<uint8_t *>(out.data());
  for (int y = 0; y < h; ++y) {
    // Y_DOWNWARD: source row index is (h - 1 - y)
    const float *src =
        atlas.pixels + static_cast<ptrdiff_t>((h - 1 - y) * w * 4);
    const float *const src_end = src + static_cast<ptrdiff_t>(w * 4);
    while (src != src_end) {
      const float clamped = std::clamp(*src++, 0.0f, 1.0f);
      *dst++ = static_cast<uint8_t>(clamped * 255.0f + 0.5f);
    }
  }
  return out;
}

[[nodiscard]] std::vector<std::byte>
convertAtlasToRgba16f(const msdfgen::BitmapConstRef<float, 4> &atlas) {
  const int w = atlas.width;
  const int h = atlas.height;
  std::vector<std::byte> out(static_cast<size_t>(w) * static_cast<size_t>(h) *
                             8u);

  auto *dst = reinterpret_cast<uint16_t *>(out.data());
  for (int y = 0; y < h; ++y) {
    const float *src =
        atlas.pixels + static_cast<ptrdiff_t>((h - 1 - y) * w * 4);
    for (int x = 0; x < w; ++x, src += 4, dst += 4) {
      const glm::vec4 clamped =
          glm::clamp(glm::vec4(src[0], src[1], src[2], src[3]), 0.0f, 1.0f);
      const uint64_t packed = glm::packHalf4x16(clamped);
      std::memcpy(dst, &packed, sizeof(uint64_t));
    }
  }
  return out;
}

[[nodiscard]] Result<bool, std::string>
writeFileBytes(const std::filesystem::path &path,
               std::span<const std::byte> bytes) {
  std::error_code ec;
  const std::filesystem::path parent = path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent, ec);
    if (ec) {
      return makeError<bool>(
          "compileNFontFromFontFile: failed to create directory '",
          parent.string(), "' (", ec.message(), ")");
    }
  }

  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output.is_open()) {
    return makeError<bool>(
        "compileNFontFromFontFile: failed to open output file '", path.string(),
        "'");
  }

  if (!bytes.empty()) {
    output.write(reinterpret_cast<const char *>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    if (!output) {
      return makeError<bool>(
          "compileNFontFromFontFile: failed writing output file '",
          path.string(), "'");
    }
  }
  output.flush();
  if (!output) {
    return makeError<bool>(
        "compileNFontFromFontFile: failed flushing output file '",
        path.string(), "'");
  }
  return Result<bool, std::string>::makeResult(true);
}

struct FreetypeHandleDeleter {
  void operator()(msdfgen::FreetypeHandle *handle) const noexcept {
    if (handle != nullptr) {
      msdfgen::deinitializeFreetype(handle);
    }
  }
};

struct FontHandleDeleter {
  void operator()(msdfgen::FontHandle *handle) const noexcept {
    if (handle != nullptr) {
      msdfgen::destroyFont(handle);
    }
  }
};

} // namespace

Result<NFontCompileReport, std::string>
compileNFontFromFontFile(const NFontCompileConfig &config) {
  if (config.sourceFontPath.empty()) {
    return makeError<NFontCompileReport>(
        "compileNFontFromFontFile: source font path is empty");
  }
  if (config.outputFontPath.empty()) {
    return makeError<NFontCompileReport>(
        "compileNFontFromFontFile: output font path is empty");
  }
  if (config.minimumEmSize <= 0.0f) {
    return makeError<NFontCompileReport>(
        "compileNFontFromFontFile: minimumEmSize must be > 0");
  }
  if (config.pxRange <= 0.0f) {
    return makeError<NFontCompileReport>(
        "compileNFontFromFontFile: pxRange must be > 0");
  }
  if (config.outerPixelPadding < 0.0f) {
    return makeError<NFontCompileReport>(
        "compileNFontFromFontFile: outerPixelPadding must be >= 0");
  }

  std::unique_ptr<msdfgen::FreetypeHandle, FreetypeHandleDeleter> ft(
      msdfgen::initializeFreetype());
  if (ft == nullptr) {
    return makeError<NFontCompileReport>(
        "compileNFontFromFontFile: initializeFreetype failed");
  }

  std::unique_ptr<msdfgen::FontHandle, FontHandleDeleter> font(
      msdfgen::loadFont(ft.get(), config.sourceFontPath.string().c_str()));
  if (font == nullptr) {
    return makeError<NFontCompileReport>(
        "compileNFontFromFontFile: failed to load font '",
        config.sourceFontPath.string(), "'");
  }

  std::vector<msdf_atlas::GlyphGeometry> glyphs;
  msdf_atlas::FontGeometry fontGeometry(&glyphs);
  const msdf_atlas::Charset charset = buildCharset(config.charset);
  const int loaded =
      fontGeometry.loadCharset(font.get(), 1.0, charset, true, true);
  if (loaded <= 0 || glyphs.empty()) {
    return makeError<NFontCompileReport>(
        "compileNFontFromFontFile: no glyphs were loaded from charset");
  }

  const double cornerAngle = 3.0;
  for (msdf_atlas::GlyphGeometry &glyph : glyphs) {
    glyph.edgeColoring(&msdfgen::edgeColoringInkTrap, cornerAngle, 0);
  }

  msdf_atlas::TightAtlasPacker packer;
  if (config.maxAtlasWidth > 0 && config.maxAtlasHeight > 0) {
    packer.setDimensions(static_cast<int>(config.maxAtlasWidth),
                         static_cast<int>(config.maxAtlasHeight));
  } else {
    packer.setDimensionsConstraint(msdf_atlas::DimensionsConstraint::SQUARE);
  }
  packer.setMinimumScale(config.minimumEmSize);
  packer.setPixelRange(config.pxRange);
  packer.setSpacing(static_cast<int>(config.atlasSpacing));
  packer.setOuterPixelPadding(msdf_atlas::Padding(config.outerPixelPadding));
  packer.setMiterLimit(1.0);

  const int packResult =
      packer.pack(glyphs.data(), static_cast<int>(glyphs.size()));
  if (packResult != 0) {
    return makeError<NFontCompileReport>(
        "compileNFontFromFontFile: atlas packing failed with code ",
        packResult);
  }

  int width = 0;
  int height = 0;
  packer.getDimensions(width, height);
  if (width <= 0 || height <= 0) {
    return makeError<NFontCompileReport>(
        "compileNFontFromFontFile: invalid atlas dimensions");
  }

  using AtlasGenerator = msdf_atlas::ImmediateAtlasGenerator<
      float, 4, msdf_atlas::mtsdfGenerator,
      msdf_atlas::BitmapAtlasStorage<float, 4>>;
  AtlasGenerator generator(width, height);
  msdf_atlas::GeneratorAttributes attributes{};
  attributes.scanlinePass = true;
  generator.setAttributes(attributes);
  generator.setThreadCount(
      config.threadCount > 0
          ? static_cast<int>(config.threadCount)
          : static_cast<int>(std::thread::hardware_concurrency()));
  generator.generate(glyphs.data(), static_cast<int>(glyphs.size()));

  const msdfgen::BitmapConstRef<float, 4> atlas = generator.atlasStorage();
  NFontBinaryData nfontData{};
  nfontData.pxRange = config.pxRange;

  const msdfgen::FontMetrics &fontMetrics = fontGeometry.getMetrics();
  nfontData.metrics = FontMetrics{
      .ascent = static_cast<float>(fontMetrics.ascenderY),
      .descent = static_cast<float>(fontMetrics.descenderY),
      .lineGap =
          static_cast<float>(fontMetrics.lineHeight -
                             (fontMetrics.ascenderY - fontMetrics.descenderY)),
      .unitsPerEm = static_cast<float>(
          fontMetrics.emSize > 0.0 ? fontMetrics.emSize : 1.0),
  };

  nfontData.cmap.reserve(glyphs.size());
  nfontData.glyphs.reserve(glyphs.size());
  for (const msdf_atlas::GlyphGeometry &glyph : glyphs) {
    double planeL = 0.0;
    double planeB = 0.0;
    double planeR = 0.0;
    double planeT = 0.0;
    glyph.getQuadPlaneBounds(planeL, planeB, planeR, planeT);

    double atlasL = 0.0;
    double atlasB = 0.0;
    double atlasR = 0.0;
    double atlasT = 0.0;
    glyph.getQuadAtlasBounds(atlasL, atlasB, atlasR, atlasT);

    const float uvMinX =
        static_cast<float>(atlasL / static_cast<double>(width));
    const float uvMaxX =
        static_cast<float>(atlasR / static_cast<double>(width));
    const float uvMinY =
        static_cast<float>(1.0 - atlasT / static_cast<double>(height));
    const float uvMaxY =
        static_cast<float>(1.0 - atlasB / static_cast<double>(height));

    nfontData.glyphs.emplace_back(GlyphMetrics{
        .glyphId = static_cast<GlyphId>(glyph.getIndex()),
        .localPageIndex = 0,
        .advance = static_cast<float>(glyph.getAdvance()),
        .bearingX = static_cast<float>(planeL),
        .bearingY = static_cast<float>(planeT),
        .planeMinX = static_cast<float>(planeL),
        .planeMinY = static_cast<float>(planeB),
        .planeMaxX = static_cast<float>(planeR),
        .planeMaxY = static_cast<float>(planeT),
        .uvMinX = std::clamp(uvMinX, 0.0f, 1.0f),
        .uvMinY = std::clamp(uvMinY, 0.0f, 1.0f),
        .uvMaxX = std::clamp(uvMaxX, 0.0f, 1.0f),
        .uvMaxY = std::clamp(uvMaxY, 0.0f, 1.0f),
    });

    const uint32_t codepoint = glyph.getCodepoint();
    if (codepoint != 0) {
      nfontData.cmap.emplace_back(NFontBinaryCmapEntry{
          .codepoint = codepoint,
          .glyphId = static_cast<GlyphId>(glyph.getIndex()),
      });
    }
  }

  std::sort(nfontData.glyphs.begin(), nfontData.glyphs.end(),
            [](const GlyphMetrics &a, const GlyphMetrics &b) {
              return a.glyphId < b.glyphId;
            });
  std::sort(nfontData.cmap.begin(), nfontData.cmap.end(),
            [](const NFontBinaryCmapEntry &a, const NFontBinaryCmapEntry &b) {
              return a.codepoint < b.codepoint;
            });

  std::vector<std::byte> atlasBytes = config.useRgba16fAtlas
                                         ? convertAtlasToRgba16f(atlas)
                                         : convertAtlasToRgba8(atlas);
  nfontData.atlasPages.push_back(NFontBinaryAtlasImage{
      .width = static_cast<uint32_t>(width),
      .height = static_cast<uint32_t>(height),
      .imageBytes = std::move(atlasBytes),
  });

  auto serializeResult = nfontBinarySerialize(nfontData);
  if (serializeResult.hasError()) {
    return makeError<NFontCompileReport>(
        "compileNFontFromFontFile: failed to serialize nfont (",
        serializeResult.error(), ")");
  }

  const std::span<const std::byte> serializedBytes = serializeResult.value();
  const size_t bytesWritten = serializedBytes.size();
  auto writeResult = writeFileBytes(config.outputFontPath, serializedBytes);
  if (writeResult.hasError()) {
    return makeError<NFontCompileReport>(writeResult.error());
  }

  return Result<NFontCompileReport, std::string>::makeResult(NFontCompileReport{
      .outputPath = config.outputFontPath,
      .glyphCount = static_cast<uint32_t>(nfontData.glyphs.size()),
      .atlasWidth = static_cast<uint32_t>(width),
      .atlasHeight = static_cast<uint32_t>(height),
      .bytesWritten = bytesWritten,
  });
}

} // namespace nuri
