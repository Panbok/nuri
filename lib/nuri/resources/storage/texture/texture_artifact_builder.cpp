#include "nuri/pch.h"

#include "nuri/resources/storage/texture/texture_artifact_builder.h"

#include "nuri/core/log.h"
#include "nuri/core/pmr_scratch.h"
#include "nuri/core/profiling.h"
#include "nuri/resources/storage/mesh/mesh_cache_utils.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <ktx.h>
#include <stb_image.h>
#include <stb_image_resize2.h>
#include <thread>

namespace nuri {
namespace {

constexpr ktx_uint32_t kKtxVkFormatR8G8B8A8Unorm = 37u;
constexpr ktx_uint32_t kKtxVkFormatR8G8B8A8Srgb = 43u;
constexpr ktx_uint32_t kKtxVkFormatBc7Unorm = 145u;
constexpr ktx_uint32_t kKtxVkFormatBc7Srgb = 146u;
constexpr ktx_uint32_t kKtxVkFormatEtc2Rgb8Unorm = 147u;
constexpr ktx_uint32_t kKtxVkFormatEtc2Rgb8Srgb = 148u;

struct AtomicTextureArtifactCacheTelemetry {
  std::atomic<uint64_t> nativeHits{0u};
  std::atomic<uint64_t> nativeMisses{0u};
  std::atomic<uint64_t> nativeStale{0u};
  std::atomic<uint64_t> nativeCorrupt{0u};
  std::atomic<uint64_t> nativeWrites{0u};
  std::atomic<uint64_t> nativeWriteFailures{0u};
  std::atomic<uint64_t> artifactBuilds{0u};
  std::atomic<uint64_t> authoredSourceBytesRead{0u};
  std::atomic<uint64_t> nativeArtifactBytesRead{0u};
  std::atomic<uint64_t> artifactBuildTimeNs{0u};
};

AtomicTextureArtifactCacheTelemetry gTelemetry{};

[[nodiscard]] uint64_t
elapsedNanoseconds(std::chrono::steady_clock::time_point start) noexcept {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start)
          .count());
}

[[nodiscard]] bool hasExtension(const std::filesystem::path &path,
                                std::string_view extension) {
  std::string value = path.extension().string();
  std::ranges::transform(value, value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value == extension;
}

[[nodiscard]] bool isKtxPath(const std::filesystem::path &path) {
  return hasExtension(path, ".ktx") || hasExtension(path, ".ktx2");
}

[[nodiscard]] uint32_t computeMipLevelCount(uint32_t width,
                                            uint32_t height) noexcept {
  uint32_t mipCount = 1u;
  uint32_t maxDim = std::max(width, height);
  while (maxDim > 1u) {
    maxDim >>= 1u;
    ++mipCount;
  }
  return mipCount;
}

[[nodiscard]] uint32_t cappedEncodeThreadCount() noexcept {
  const uint32_t hardware = std::max(1u, std::thread::hardware_concurrency());
  return std::min(hardware, 8u);
}

struct KtxTextureDeleter {
  void operator()(ktxTexture *texture) const noexcept {
    if (texture != nullptr) {
      ktxTexture_Destroy(texture);
    }
  }

  void operator()(ktxTexture2 *texture) const noexcept {
    if (texture != nullptr) {
      ktxTexture_Destroy(ktxTexture(texture));
    }
  }
};

using KtxTexturePtr = std::unique_ptr<ktxTexture, KtxTextureDeleter>;
using KtxTexture2Ptr = std::unique_ptr<ktxTexture2, KtxTextureDeleter>;

struct FileCloser {
  void operator()(std::FILE *file) const noexcept {
    if (file != nullptr) {
      std::fclose(file);
    }
  }
};

using FilePtr = std::unique_ptr<std::FILE, FileCloser>;

[[nodiscard]] FilePtr openFile(const std::filesystem::path &path,
                               bool write) {
  std::FILE *file = nullptr;
#ifdef _WIN32
  if (_wfopen_s(&file, path.c_str(), write ? L"wb" : L"rb") != 0) {
    file = nullptr;
  }
#else
  file = std::fopen(path.string().c_str(), write ? "wb" : "rb");
#endif
  return FilePtr(file);
}

[[nodiscard]] Result<KtxTexturePtr, std::string>
loadKtxTextureFile(const std::filesystem::path &path) {
  FilePtr file = openFile(path, false);
  if (!file) {
    return Result<KtxTexturePtr, std::string>::makeError(
        "Texture artifact builder: failed to open KTX source '" +
        path.string() + "'");
  }
  ktxTexture *texture = nullptr;
  const KTX_error_code loadError = ktxTexture_CreateFromStdioStream(
      file.get(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &texture);
  if (loadError != KTX_SUCCESS || texture == nullptr) {
    return Result<KtxTexturePtr, std::string>::makeError(
        "Texture artifact builder: failed to read KTX source '" +
        path.string() + "' (error " +
        std::to_string(static_cast<int>(loadError)) + ")");
  }
  return Result<KtxTexturePtr, std::string>::makeResult(KtxTexturePtr(texture));
}

struct ImageRgba8 {
  int32_t width = 0;
  int32_t height = 0;
  uint32_t sourceComponentCount = 4u;
  std::pmr::vector<std::byte> bytes;

  explicit ImageRgba8(std::pmr::memory_resource *memory)
      : bytes(memory != nullptr ? memory : std::pmr::get_default_resource()) {}
};

[[nodiscard]] uint8_t clampByteFromUnit(float value) {
  return static_cast<uint8_t>(
      std::round(std::clamp(value, 0.0f, 1.0f) * 255.0f));
}

[[nodiscard]] float byteToUnit(std::byte value) {
  return static_cast<float>(std::to_integer<uint32_t>(value)) / 255.0f;
}

[[nodiscard]] float sanitizeAlphaCoverageCutoff(float cutoff) {
  if (!std::isfinite(cutoff)) {
    return 0.5f;
  }
  return std::clamp(cutoff, 1.0f / 255.0f, 254.0f / 255.0f);
}

[[nodiscard]] float alphaCoverage(std::span<const std::byte> rgba,
                                  float cutoff) {
  uint32_t covered = 0u;
  const size_t texelCount = rgba.size() / 4u;
  for (size_t i = 0u; i < texelCount; ++i) {
    covered += byteToUnit(rgba[i * 4u + 3u]) >= cutoff ? 1u : 0u;
  }
  return texelCount == 0u
             ? 0.0f
             : static_cast<float>(covered) / static_cast<float>(texelCount);
}

[[nodiscard]] float scaledAlphaCoverage(std::span<const std::byte> rgba,
                                        float cutoff, float scale) {
  uint32_t covered = 0u;
  const size_t texelCount = rgba.size() / 4u;
  for (size_t i = 0u; i < texelCount; ++i) {
    const float alpha = std::min(byteToUnit(rgba[i * 4u + 3u]) * scale, 1.0f);
    covered += alpha >= cutoff ? 1u : 0u;
  }
  return texelCount == 0u
             ? 0.0f
             : static_cast<float>(covered) / static_cast<float>(texelCount);
}

void preserveAlphaCoverage(std::pmr::vector<std::byte> &rgba, float cutoff,
                           float targetCoverage) {
  if (rgba.empty() || targetCoverage <= 0.0f) {
    return;
  }
  float lo = 0.0f;
  float hi = 1.0f;
  while (scaledAlphaCoverage(rgba, cutoff, hi) < targetCoverage && hi < 16.0f) {
    hi *= 2.0f;
  }
  for (uint32_t i = 0u; i < 10u; ++i) {
    const float mid = (lo + hi) * 0.5f;
    if (scaledAlphaCoverage(rgba, cutoff, mid) < targetCoverage) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  for (size_t i = 3u; i < rgba.size(); i += 4u) {
    rgba[i] =
        static_cast<std::byte>(clampByteFromUnit(byteToUnit(rgba[i]) * hi));
  }
}

[[nodiscard]] Result<std::pmr::vector<std::byte>, std::string>
generateNextMipLevel(std::span<const std::byte> srcBytes, int32_t srcWidth,
                     int32_t srcHeight, int32_t dstWidth, int32_t dstHeight,
                     const TextureLoadOptions &options,
                     std::pmr::memory_resource *memory) {
  std::pmr::vector<std::byte> out(memory);
  out.resize(static_cast<size_t>(dstWidth) * static_cast<size_t>(dstHeight) *
             4u);

  if (options.mipSemantic == TextureMipSemantic::NormalMap) {
    std::pmr::vector<float> srcFloats(memory);
    std::pmr::vector<float> dstFloats(memory);
    srcFloats.resize(static_cast<size_t>(srcWidth) *
                     static_cast<size_t>(srcHeight) * 4u);
    dstFloats.resize(static_cast<size_t>(dstWidth) *
                     static_cast<size_t>(dstHeight) * 4u);
    const size_t srcPixelCount =
        static_cast<size_t>(srcWidth) * static_cast<size_t>(srcHeight);
    for (size_t i = 0u; i < srcPixelCount; ++i) {
      const size_t offset = i * 4u;
      glm::vec3 normal(byteToUnit(srcBytes[offset + 0u]) * 2.0f - 1.0f,
                       byteToUnit(srcBytes[offset + 1u]) * 2.0f - 1.0f,
                       byteToUnit(srcBytes[offset + 2u]) * 2.0f - 1.0f);
      normal = glm::dot(normal, normal) > 1.0e-8f
                   ? glm::normalize(normal)
                   : glm::vec3(0.0f, 0.0f, 1.0f);
      srcFloats[offset + 0u] = normal.x;
      srcFloats[offset + 1u] = normal.y;
      srcFloats[offset + 2u] = normal.z;
      srcFloats[offset + 3u] = byteToUnit(srcBytes[offset + 3u]);
    }
    if (stbir_resize_float_linear(srcFloats.data(), srcWidth, srcHeight, 0,
                                  dstFloats.data(), dstWidth, dstHeight, 0,
                                  STBIR_RGBA) == nullptr) {
      return Result<std::pmr::vector<std::byte>, std::string>::makeError(
          "Texture artifact builder: failed to resize normal-map mip");
    }
    const size_t dstPixelCount =
        static_cast<size_t>(dstWidth) * static_cast<size_t>(dstHeight);
    for (size_t i = 0u; i < dstPixelCount; ++i) {
      const size_t offset = i * 4u;
      glm::vec3 normal(dstFloats[offset + 0u], dstFloats[offset + 1u],
                       dstFloats[offset + 2u]);
      normal = glm::dot(normal, normal) > 1.0e-8f
                   ? glm::normalize(normal)
                   : glm::vec3(0.0f, 0.0f, 1.0f);
      out[offset + 0u] = static_cast<std::byte>(
          clampByteFromUnit(normal.x * 0.5f + 0.5f));
      out[offset + 1u] = static_cast<std::byte>(
          clampByteFromUnit(normal.y * 0.5f + 0.5f));
      out[offset + 2u] = static_cast<std::byte>(
          clampByteFromUnit(normal.z * 0.5f + 0.5f));
      out[offset + 3u] =
          static_cast<std::byte>(clampByteFromUnit(dstFloats[offset + 3u]));
    }
    return Result<std::pmr::vector<std::byte>, std::string>::makeResult(
        std::move(out));
  }

  const auto *src = reinterpret_cast<const unsigned char *>(srcBytes.data());
  auto *dst = reinterpret_cast<unsigned char *>(out.data());
  unsigned char *resizeResult =
      options.srgb
          ? stbir_resize_uint8_srgb(src, srcWidth, srcHeight, 0, dst, dstWidth,
                                    dstHeight, 0, STBIR_RGBA)
          : stbir_resize_uint8_linear(src, srcWidth, srcHeight, 0, dst,
                                      dstWidth, dstHeight, 0, STBIR_RGBA);
  if (resizeResult == nullptr) {
    return Result<std::pmr::vector<std::byte>, std::string>::makeError(
        "Texture artifact builder: failed to resize mip");
  }

  if (options.mipSemantic == TextureMipSemantic::AlphaCoverage) {
    const float cutoff = sanitizeAlphaCoverageCutoff(options.alphaCoverageCutoff);
    preserveAlphaCoverage(out, cutoff, alphaCoverage(srcBytes, cutoff));
  } else if (options.mipSemantic == TextureMipSemantic::RoughnessG ||
             options.mipSemantic == TextureMipSemantic::RoughnessA) {
    const uint32_t channel =
        options.mipSemantic == TextureMipSemantic::RoughnessA ? 3u : 1u;
    for (int32_t y = 0; y < dstHeight; ++y) {
      for (int32_t x = 0; x < dstWidth; ++x) {
        float roughnessSqSum = 0.0f;
        uint32_t count = 0u;
        for (int32_t dy = 0; dy < 2; ++dy) {
          const int32_t sy = y * 2 + dy;
          if (sy >= srcHeight) {
            continue;
          }
          for (int32_t dx = 0; dx < 2; ++dx) {
            const int32_t sx = x * 2 + dx;
            if (sx >= srcWidth) {
              continue;
            }
            const size_t srcOffset =
                (static_cast<size_t>(sy) * static_cast<size_t>(srcWidth) +
                 static_cast<size_t>(sx)) *
                    4u +
                channel;
            const float roughness = byteToUnit(srcBytes[srcOffset]);
            roughnessSqSum += roughness * roughness;
            ++count;
          }
        }
        const float roughness =
            count == 0u
                ? 1.0f
                : std::sqrt(roughnessSqSum / static_cast<float>(count));
        const size_t dstOffset =
            (static_cast<size_t>(y) * static_cast<size_t>(dstWidth) +
             static_cast<size_t>(x)) *
                4u +
            channel;
        out[dstOffset] = static_cast<std::byte>(clampByteFromUnit(roughness));
      }
    }
  }

  return Result<std::pmr::vector<std::byte>, std::string>::makeResult(
      std::move(out));
}

[[nodiscard]] Result<ImageRgba8, std::string>
loadKtxRgba(const std::filesystem::path &path,
            std::pmr::memory_resource *memory) {
  auto textureResult = loadKtxTextureFile(path);
  if (textureResult.hasError()) {
    return Result<ImageRgba8, std::string>::makeError(
        textureResult.error());
  }
  KtxTexturePtr texturePtr = std::move(textureResult.value());
  ktxTexture *texture = texturePtr.get();
  if (texture->numFaces != 1u || texture->numLayers > 1u) {
    return Result<ImageRgba8, std::string>::makeError(
        "Texture artifact builder: source KTX must be a non-array 2D texture");
  }
  uint32_t componentCount = 4u;
  if (texture->classId == ktxTexture2_c) {
    auto *texture2 = reinterpret_cast<ktxTexture2 *>(texture);
    componentCount = ktxTexture2_GetNumComponents(texture2);
    if (ktxTexture2_NeedsTranscoding(texture2)) {
      const KTX_error_code transcodeError =
          ktxTexture2_TranscodeBasis(texture2, KTX_TTF_RGBA32, 0u);
      if (transcodeError != KTX_SUCCESS) {
        return Result<ImageRgba8, std::string>::makeError(
            "Texture artifact builder: failed to decode authored Basis KTX2");
      }
    }
  }
  const size_t expectedSize = static_cast<size_t>(texture->baseWidth) *
                              static_cast<size_t>(texture->baseHeight) * 4u;
  const ktx_size_t imageSize = ktxTexture_GetImageSize(texture, 0u);
  if (static_cast<size_t>(imageSize) != expectedSize) {
    return Result<ImageRgba8, std::string>::makeError(
        "Texture artifact builder: native KTX source is not RGBA8 and cannot "
        "be converted to the selected target");
  }
  ktx_size_t offset = 0u;
  if (ktxTexture_GetImageOffset(texture, 0u, 0u, 0u, &offset) != KTX_SUCCESS) {
    return Result<ImageRgba8, std::string>::makeError(
        "Texture artifact builder: failed to resolve KTX base mip");
  }
  const uint8_t *data = ktxTexture_GetData(texture);
  if (data == nullptr) {
    return Result<ImageRgba8, std::string>::makeError(
        "Texture artifact builder: KTX source payload is empty");
  }
  ImageRgba8 out(memory);
  out.width = static_cast<int32_t>(texture->baseWidth);
  out.height = static_cast<int32_t>(texture->baseHeight);
  out.sourceComponentCount = componentCount;
  out.bytes.assign(reinterpret_cast<const std::byte *>(data + offset),
                   reinterpret_cast<const std::byte *>(data + offset) +
                       expectedSize);
  return Result<ImageRgba8, std::string>::makeResult(std::move(out));
}

[[nodiscard]] Result<ImageRgba8, std::string>
loadExternalRgba(const std::filesystem::path &path,
                 std::pmr::memory_resource *memory) {
  if (isKtxPath(path)) {
    return loadKtxRgba(path, memory);
  }
  int32_t width = 0;
  int32_t height = 0;
  int32_t channels = 0;
  stbi_uc *pixels =
      stbi_load(path.string().c_str(), &width, &height, &channels, 4);
  if (pixels == nullptr) {
    const char *reason = stbi_failure_reason();
    return Result<ImageRgba8, std::string>::makeError(
        "Texture artifact builder: failed to load source '" + path.string() +
        "': " + (reason != nullptr ? reason : "unknown error"));
  }
  ImageRgba8 out(memory);
  out.width = width;
  out.height = height;
  out.sourceComponentCount =
      static_cast<uint32_t>(std::clamp(channels, 1, 4));
  const size_t byteCount =
      static_cast<size_t>(width) * static_cast<size_t>(height) * 4u;
  out.bytes.assign(reinterpret_cast<std::byte *>(pixels),
                   reinterpret_cast<std::byte *>(pixels) + byteCount);
  stbi_image_free(pixels);
  return Result<ImageRgba8, std::string>::makeResult(std::move(out));
}

[[nodiscard]] Result<ImageRgba8, std::string>
loadEmbeddedRgba(const aiScene &scene, uint32_t embeddedIndex,
                 std::pmr::memory_resource *memory) {
  if (embeddedIndex >= scene.mNumTextures ||
      scene.mTextures[embeddedIndex] == nullptr) {
    return Result<ImageRgba8, std::string>::makeError(
        "Texture artifact builder: embedded texture index is invalid");
  }
  const aiTexture &texture = *scene.mTextures[embeddedIndex];
  if (texture.mHeight == 0u) {
    int32_t width = 0;
    int32_t height = 0;
    int32_t channels = 0;
    stbi_uc *pixels = stbi_load_from_memory(
        reinterpret_cast<const stbi_uc *>(texture.pcData),
        static_cast<int>(texture.mWidth), &width, &height, &channels, 4);
    if (pixels == nullptr) {
      return Result<ImageRgba8, std::string>::makeError(
          "Texture artifact builder: failed to decode embedded texture");
    }
    ImageRgba8 out(memory);
    out.width = width;
    out.height = height;
    out.sourceComponentCount =
        static_cast<uint32_t>(std::clamp(channels, 1, 4));
    const size_t byteCount =
        static_cast<size_t>(width) * static_cast<size_t>(height) * 4u;
    out.bytes.assign(reinterpret_cast<std::byte *>(pixels),
                     reinterpret_cast<std::byte *>(pixels) + byteCount);
    stbi_image_free(pixels);
    return Result<ImageRgba8, std::string>::makeResult(std::move(out));
  }

  ImageRgba8 out(memory);
  out.width = static_cast<int32_t>(texture.mWidth);
  out.height = static_cast<int32_t>(texture.mHeight);
  out.sourceComponentCount = 4u;
  out.bytes.resize(static_cast<size_t>(out.width) *
                   static_cast<size_t>(out.height) * 4u);
  for (int32_t y = 0; y < out.height; ++y) {
    for (int32_t x = 0; x < out.width; ++x) {
      const aiTexel &src =
          texture.pcData[static_cast<size_t>(y) *
                             static_cast<size_t>(out.width) +
                         static_cast<size_t>(x)];
      const size_t offset =
          (static_cast<size_t>(y) * static_cast<size_t>(out.width) +
           static_cast<size_t>(x)) *
          4u;
      out.bytes[offset + 0u] = static_cast<std::byte>(src.r);
      out.bytes[offset + 1u] = static_cast<std::byte>(src.g);
      out.bytes[offset + 2u] = static_cast<std::byte>(src.b);
      out.bytes[offset + 3u] = static_cast<std::byte>(src.a);
    }
  }
  return Result<ImageRgba8, std::string>::makeResult(std::move(out));
}

[[nodiscard]] Result<Format, std::string>
resolveNativeKtxFormat(const ktxTexture &texture) {
  if (texture.classId != ktxTexture2_c) {
    return Result<Format, std::string>::makeError(
        "Texture artifact builder: authored native KTX1 is not a target "
        "artifact");
  }
  const auto &texture2 = reinterpret_cast<const ktxTexture2 &>(texture);
  switch (texture2.vkFormat) {
  case kKtxVkFormatR8G8B8A8Unorm:
    return Result<Format, std::string>::makeResult(Format::RGBA8_UNORM);
  case kKtxVkFormatR8G8B8A8Srgb:
    return Result<Format, std::string>::makeResult(Format::RGBA8_SRGB);
  case kKtxVkFormatBc7Unorm:
    return Result<Format, std::string>::makeResult(Format::BC7_RGBA_UNORM);
  case kKtxVkFormatBc7Srgb:
    return Result<Format, std::string>::makeResult(Format::BC7_RGBA_SRGB);
  case kKtxVkFormatEtc2Rgb8Unorm:
    return Result<Format, std::string>::makeResult(Format::ETC2_RGB8_UNORM);
  case kKtxVkFormatEtc2Rgb8Srgb:
    return Result<Format, std::string>::makeResult(Format::ETC2_RGB8_SRGB);
  default:
    return Result<Format, std::string>::makeError(
        "Texture artifact builder: authored native KTX2 format is unsupported");
  }
}

[[nodiscard]] Result<ktx_transcode_fmt_e, std::string>
resolveTranscodeFormat(Format format) {
  auto profile = resolveTextureArtifactTranscodeFormat(format);
  if (profile.hasError()) {
    return Result<ktx_transcode_fmt_e, std::string>::makeError(profile.error());
  }
  switch (profile.value()) {
  case TextureArtifactTranscodeFormat::BC7_RGBA:
    return Result<ktx_transcode_fmt_e, std::string>::makeResult(
        KTX_TTF_BC7_RGBA);
  case TextureArtifactTranscodeFormat::ETC1_RGB:
    return Result<ktx_transcode_fmt_e, std::string>::makeResult(
        KTX_TTF_ETC1_RGB);
  case TextureArtifactTranscodeFormat::RGBA32:
    return Result<ktx_transcode_fmt_e, std::string>::makeResult(KTX_TTF_RGBA32);
  }
  return Result<ktx_transcode_fmt_e, std::string>::makeError(
      "Texture artifact builder: unsupported transcode format");
}

[[nodiscard]] bool textureFormatIsSrgb(Format format) noexcept {
  return format == Format::RGBA8_SRGB || format == Format::BC7_RGBA_SRGB ||
         format == Format::ETC2_RGB8_SRGB;
}

[[nodiscard]] Result<KtxTexture2Ptr, std::string>
createNativeCopy(const ktxTexture &source, Format targetFormat) {
  if (source.classId != ktxTexture2_c) {
    return Result<KtxTexture2Ptr, std::string>::makeError(
        "Texture artifact builder: native artifact source is not KTX2");
  }
  const auto &source2 = reinterpret_cast<const ktxTexture2 &>(source);
  auto sourceFormat = resolveNativeKtxFormat(source);
  if (sourceFormat.hasError() || sourceFormat.value() != targetFormat) {
    return Result<KtxTexture2Ptr, std::string>::makeError(
        "Texture artifact builder: native KTX2 format does not match target");
  }
  ktxTexture2 *copy = nullptr;
  if (ktxTexture2_CreateCopy(const_cast<ktxTexture2 *>(&source2), &copy) !=
          KTX_SUCCESS ||
      copy == nullptr) {
    return Result<KtxTexture2Ptr, std::string>::makeError(
        "Texture artifact builder: failed to copy native KTX2");
  }
  return Result<KtxTexture2Ptr, std::string>::makeResult(
      KtxTexture2Ptr(copy));
}

[[nodiscard]] Result<KtxTexture2Ptr, std::string>
transcodeBasisToNative(ktxTexture2 &source, Format targetFormat) {
  ktxTexture2 *copy = nullptr;
  if (ktxTexture2_CreateCopy(&source, &copy) != KTX_SUCCESS || copy == nullptr) {
    return Result<KtxTexture2Ptr, std::string>::makeError(
        "Texture artifact builder: failed to copy Basis KTX2");
  }
  KtxTexture2Ptr copyPtr(copy);
  auto transcodeFormat = resolveTranscodeFormat(targetFormat);
  if (transcodeFormat.hasError()) {
    return Result<KtxTexture2Ptr, std::string>::makeError(
        transcodeFormat.error());
  }
  const khr_df_transfer_e transferFunction =
      textureFormatIsSrgb(targetFormat) ? KHR_DF_TRANSFER_SRGB
                                        : KHR_DF_TRANSFER_LINEAR;
  if (ktxTexture2_SetTransferFunction(copyPtr.get(), transferFunction) !=
      KTX_SUCCESS) {
    return Result<KtxTexture2Ptr, std::string>::makeError(
        "Texture artifact builder: failed to set Basis transfer function");
  }
  const ktx_transcode_flags flags =
      textureArtifactTranscodeUsesHighQuality(targetFormat)
          ? KTX_TF_HIGH_QUALITY
          : 0u;
  if (ktxTexture2_TranscodeBasis(copyPtr.get(), transcodeFormat.value(),
                                 flags) != KTX_SUCCESS) {
    return Result<KtxTexture2Ptr, std::string>::makeError(
        "Texture artifact builder: Basis transcode failed");
  }
  auto nativeFormat = resolveNativeKtxFormat(*ktxTexture(copyPtr.get()));
  if (nativeFormat.hasError() || nativeFormat.value() != targetFormat) {
    return Result<KtxTexture2Ptr, std::string>::makeError(
        "Texture artifact builder: Basis transcode produced unexpected format");
  }
  return Result<KtxTexture2Ptr, std::string>::makeResult(std::move(copyPtr));
}

[[nodiscard]] Result<KtxTexture2Ptr, std::string>
createRgbaSourceTexture(const ImageRgba8 &image,
                        const TextureArtifactBuildOptions &options,
                        std::pmr::memory_resource *memory) {
  if (image.width <= 0 || image.height <= 0 || image.bytes.empty()) {
    return Result<KtxTexture2Ptr, std::string>::makeError(
        "Texture artifact builder: decoded image is empty");
  }
  const uint32_t mipLevels =
      options.loadOptions.generateMipmaps
          ? computeMipLevelCount(static_cast<uint32_t>(image.width),
                                 static_cast<uint32_t>(image.height))
          : 1u;
  const ktxTextureCreateInfo createInfo{
      .glInternalformat = 0u,
      .vkFormat = options.loadOptions.srgb ? kKtxVkFormatR8G8B8A8Srgb
                                          : kKtxVkFormatR8G8B8A8Unorm,
      .pDfd = nullptr,
      .baseWidth = static_cast<ktx_uint32_t>(image.width),
      .baseHeight = static_cast<ktx_uint32_t>(image.height),
      .baseDepth = 1u,
      .numDimensions = 2u,
      .numLevels = mipLevels,
      .numLayers = 1u,
      .numFaces = 1u,
      .isArray = KTX_FALSE,
      .generateMipmaps = KTX_FALSE,
  };
  ktxTexture2 *texture = nullptr;
  if (ktxTexture2_Create(&createInfo, KTX_TEXTURE_CREATE_ALLOC_STORAGE,
                         &texture) != KTX_SUCCESS ||
      texture == nullptr) {
    return Result<KtxTexture2Ptr, std::string>::makeError(
        "Texture artifact builder: failed to allocate RGBA KTX2");
  }
  KtxTexture2Ptr texturePtr(texture);
  int32_t width = image.width;
  int32_t height = image.height;
  std::span<const std::byte> current(image.bytes.data(), image.bytes.size());
  std::pmr::vector<std::byte> owned(memory);
  for (uint32_t level = 0u; level < mipLevels; ++level) {
    if (ktxTexture_SetImageFromMemory(
            ktxTexture(texturePtr.get()), level, 0u, 0u,
            reinterpret_cast<const ktx_uint8_t *>(current.data()),
            current.size()) != KTX_SUCCESS) {
      return Result<KtxTexture2Ptr, std::string>::makeError(
          "Texture artifact builder: failed to populate RGBA mip");
    }
    if (level + 1u == mipLevels) {
      break;
    }
    const int32_t nextWidth = std::max(1, width >> 1);
    const int32_t nextHeight = std::max(1, height >> 1);
    auto next = generateNextMipLevel(current, width, height, nextWidth,
                                     nextHeight, options.loadOptions, memory);
    if (next.hasError()) {
      return Result<KtxTexture2Ptr, std::string>::makeError(next.error());
    }
    owned = std::move(next.value());
    current = std::span<const std::byte>(owned.data(), owned.size());
    width = nextWidth;
    height = nextHeight;
  }
  return Result<KtxTexture2Ptr, std::string>::makeResult(
      std::move(texturePtr));
}

[[nodiscard]] Result<KtxTexture2Ptr, std::string>
buildNativeFromImage(const ImageRgba8 &image, Format targetFormat,
                     const TextureArtifactBuildOptions &options,
                     std::pmr::memory_resource *memory) {
  auto rgbaTexture = createRgbaSourceTexture(image, options, memory);
  if (rgbaTexture.hasError()) {
    return rgbaTexture;
  }
  if (targetFormat == Format::RGBA8_UNORM ||
      targetFormat == Format::RGBA8_SRGB) {
    return createNativeCopy(*ktxTexture(rgbaTexture.value().get()),
                            targetFormat);
  }
  ktxBasisParams params{};
  params.structSize = sizeof(params);
  params.threadCount = cappedEncodeThreadCount();
  params.normalMap = KTX_FALSE;
  if (options.encoding == TextureArtifactEncoding::Uastc) {
    params.uastc = KTX_TRUE;
    params.uastcFlags = KTX_PACK_UASTC_LEVEL_DEFAULT;
    params.uastcRDO = KTX_FALSE;
  } else {
    params.uastc = KTX_FALSE;
    params.compressionLevel = KTX_ETC1S_DEFAULT_COMPRESSION_LEVEL;
    params.qualityLevel = 128u;
  }
  if (ktxTexture2_CompressBasisEx(rgbaTexture.value().get(), &params) !=
      KTX_SUCCESS) {
    return Result<KtxTexture2Ptr, std::string>::makeError(
        "Texture artifact builder: Basis encoding failed");
  }
  return transcodeBasisToNative(*rgbaTexture.value(), targetFormat);
}

[[nodiscard]] Result<std::optional<KtxTexture2Ptr>, std::string>
tryBuildDirectlyFromAuthoredKtx(const std::filesystem::path &sourcePath,
                                Format targetFormat) {
  auto textureResult = loadKtxTextureFile(sourcePath);
  if (textureResult.hasError()) {
    return Result<std::optional<KtxTexture2Ptr>, std::string>::makeError(
        textureResult.error());
  }
  KtxTexturePtr texturePtr = std::move(textureResult.value());
  ktxTexture *texture = texturePtr.get();
  if (texture->classId == ktxTexture2_c) {
    auto *texture2 = reinterpret_cast<ktxTexture2 *>(texture);
    if (ktxTexture2_NeedsTranscoding(texture2)) {
      auto native = transcodeBasisToNative(*texture2, targetFormat);
      if (native.hasError()) {
        return Result<std::optional<KtxTexture2Ptr>, std::string>::makeError(
            native.error());
      }
      return Result<std::optional<KtxTexture2Ptr>, std::string>::makeResult(
          std::optional<KtxTexture2Ptr>(std::move(native.value())));
    }
  }
  auto sourceFormat = resolveNativeKtxFormat(*texture);
  if (!sourceFormat.hasError() && sourceFormat.value() == targetFormat) {
    auto native = createNativeCopy(*texture, targetFormat);
    if (native.hasError()) {
      return Result<std::optional<KtxTexture2Ptr>, std::string>::makeError(
          native.error());
    }
    return Result<std::optional<KtxTexture2Ptr>, std::string>::makeResult(
        std::optional<KtxTexture2Ptr>(std::move(native.value())));
  }
  return Result<std::optional<KtxTexture2Ptr>, std::string>::makeResult(
      std::nullopt);
}

[[nodiscard]] Result<uint64_t, std::string>
writeKtxTextureAtomic(const std::filesystem::path &path,
                      ktxTexture2 &texture) {
  if (path.empty()) {
    return Result<uint64_t, std::string>::makeError(
        "Texture artifact builder: artifact path is empty");
  }

  std::error_code ec;
  const std::filesystem::path parent = path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent, ec);
    if (ec) {
      return Result<uint64_t, std::string>::makeError(
          "Texture artifact builder: failed to create artifact directory '" +
          parent.string() + "'");
    }
  }

  static std::atomic<uint64_t> counter{0u};
  const uint64_t threadIdHash =
      std::hash<std::thread::id>{}(std::this_thread::get_id());
  const std::filesystem::path tempPath =
      path.string() +
      std::format(".tmp.{:x}.{}", threadIdHash,
                  counter.fetch_add(1u, std::memory_order_relaxed));
  FilePtr output = openFile(tempPath, true);
  if (!output) {
    return Result<uint64_t, std::string>::makeError(
        "Texture artifact builder: failed to open KTX2 artifact '" +
        tempPath.string() + "'");
  }
  const KTX_error_code writeError =
      ktxTexture2_WriteToStdioStream(&texture, output.get());
  output.reset();
  if (writeError != KTX_SUCCESS) {
    std::filesystem::remove(tempPath, ec);
    return Result<uint64_t, std::string>::makeError(
        "Texture artifact builder: failed to write KTX2 artifact (error " +
        std::to_string(static_cast<int>(writeError)) + ")");
  }

  auto validationTexture = loadKtxTextureFile(tempPath);
  if (validationTexture.hasError()) {
    std::filesystem::remove(tempPath, ec);
    return Result<uint64_t, std::string>::makeError(
        "Texture artifact builder: written KTX2 artifact failed validation: " +
        validationTexture.error());
  }

  const uint64_t artifactSizeBytes =
      static_cast<uint64_t>(std::filesystem::file_size(tempPath, ec));
  if (ec || artifactSizeBytes == 0u) {
    std::filesystem::remove(tempPath, ec);
    return Result<uint64_t, std::string>::makeError(
        "Texture artifact builder: failed to query KTX2 artifact size");
  }

  std::filesystem::rename(tempPath, path, ec);
  if (ec) {
    ec.clear();
    std::filesystem::remove(path, ec);
    ec.clear();
    std::filesystem::rename(tempPath, path, ec);
    if (ec) {
      std::filesystem::remove(tempPath, ec);
      return Result<uint64_t, std::string>::makeError(
          "Texture artifact builder: failed to commit KTX2 artifact '" +
          path.string() + "'");
    }
  }
  return Result<uint64_t, std::string>::makeResult(artifactSizeBytes);
}

[[nodiscard]] uint64_t tightPayloadSize(const ktxTexture &texture) noexcept {
  uint64_t size = 0u;
  ktxTexture *mutableTexture = const_cast<ktxTexture *>(&texture);
  for (uint32_t level = 0u; level < std::max(1u, texture.numLevels); ++level) {
    size += static_cast<uint64_t>(
                ktxTexture_GetImageSize(mutableTexture, level)) *
            static_cast<uint64_t>(std::max(1u, texture.numLayers)) *
            static_cast<uint64_t>(std::max(1u, texture.numFaces));
  }
  return size;
}

void recordProbeStatus(NativeTextureCacheProbeStatus status) noexcept {
  switch (status) {
  case NativeTextureCacheProbeStatus::Hit:
    break;
  case NativeTextureCacheProbeStatus::Missing:
    gTelemetry.nativeMisses.fetch_add(1u, std::memory_order_relaxed);
    break;
  case NativeTextureCacheProbeStatus::Stale:
    gTelemetry.nativeStale.fetch_add(1u, std::memory_order_relaxed);
    break;
  case NativeTextureCacheProbeStatus::Corrupt:
    gTelemetry.nativeCorrupt.fetch_add(1u, std::memory_order_relaxed);
    break;
  }
}

} // namespace

struct SceneTextureArtifactBuilder::Impl {
  std::filesystem::path sceneSourcePath{};
  Assimp::Importer importer{};
  const aiScene *scene = nullptr;
  ScratchArena scratch{};

  [[nodiscard]] Result<const aiScene *, std::string> ensureScene() {
    if (scene != nullptr) {
      return Result<const aiScene *, std::string>::makeResult(scene);
    }
    const std::string path = sceneSourcePath.string();
    scene = importer.ReadFile(path,
                              aiProcess_SortByPType | aiProcess_FindInvalidData);
    if (scene == nullptr) {
      return Result<const aiScene *, std::string>::makeError(
          std::string("Texture artifact builder: Assimp error: ") +
          importer.GetErrorString());
    }
    return Result<const aiScene *, std::string>::makeResult(scene);
  }
};

SceneTextureArtifactBuilder::~SceneTextureArtifactBuilder() = default;
SceneTextureArtifactBuilder::SceneTextureArtifactBuilder(
    std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
SceneTextureArtifactBuilder::SceneTextureArtifactBuilder(
    SceneTextureArtifactBuilder &&) noexcept = default;
SceneTextureArtifactBuilder &
SceneTextureArtifactBuilder::operator=(SceneTextureArtifactBuilder &&) noexcept =
    default;

Result<SceneTextureArtifactBuilder, std::string>
SceneTextureArtifactBuilder::create(
    const std::filesystem::path &sceneSourcePath) {
  const std::filesystem::path normalized =
      normalizeMeshSourcePath(sceneSourcePath);
  if (normalized.empty()) {
    return Result<SceneTextureArtifactBuilder, std::string>::makeError(
        "Texture artifact builder: scene source path is empty");
  }
  auto impl = std::make_unique<Impl>();
  impl->sceneSourcePath = normalized;
  return Result<SceneTextureArtifactBuilder, std::string>::makeResult(
      SceneTextureArtifactBuilder(std::move(impl)));
}

Result<TextureArtifactBuildResult, std::string>
SceneTextureArtifactBuilder::ensure(
    const MaterialTextureSlotData &source, uint64_t sourceIdentityHash,
    Format targetFormat, const TextureArtifactBuildOptions &options,
    bool forceRebuild) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (impl_ == nullptr || sourceIdentityHash == 0u) {
    return Result<TextureArtifactBuildResult, std::string>::makeError(
        "Texture artifact builder: invalid build context or identity");
  }
  if (!shouldPersistNativeTextureArtifact(targetFormat)) {
    return Result<TextureArtifactBuildResult, std::string>::makeError(
        "Texture artifact builder: target format is not artifact-compatible");
  }
  if (source.sourceKind == MaterialTextureSourceKind::None) {
    return Result<TextureArtifactBuildResult, std::string>::makeError(
        "Texture artifact builder: texture source is empty");
  }
  if (source.sourceKind == MaterialTextureSourceKind::ExternalFile &&
      source.path.empty()) {
    return Result<TextureArtifactBuildResult, std::string>::makeError(
        "Texture artifact builder: external texture path is empty");
  }
  const std::filesystem::path freshnessPath =
      source.sourceKind == MaterialTextureSourceKind::ExternalFile
          ? std::filesystem::path(source.path)
          : impl_->sceneSourcePath;
  if (hasExtension(freshnessPath, ".dds")) {
    return Result<TextureArtifactBuildResult, std::string>::makeError(
        "Texture artifact builder: DDS textures are source-native and are not "
        "converted into KTX artifacts");
  }
  auto artifactPath = buildNativeTextureArtifactPath(
      impl_->sceneSourcePath, sourceIdentityHash, targetFormat);
  if (artifactPath.hasError()) {
    return Result<TextureArtifactBuildResult, std::string>::makeError(
        artifactPath.error());
  }

  NativeTextureCacheProbe probe{};
  if (!forceRebuild) {
    probe = probeNativeTextureCache(artifactPath.value(), freshnessPath,
                                    sourceIdentityHash, targetFormat);
    recordProbeStatus(probe.status);
    if (probe.status == NativeTextureCacheProbeStatus::Hit) {
      gTelemetry.nativeHits.fetch_add(1u, std::memory_order_relaxed);
      gTelemetry.nativeArtifactBytesRead.fetch_add(
          probe.metadata.artifactSizeBytes, std::memory_order_relaxed);
      return Result<TextureArtifactBuildResult, std::string>::makeResult(
          TextureArtifactBuildResult{
              .artifactPath = artifactPath.value(),
              .targetFormat = targetFormat,
              .previousStatus = probe.status,
              .built = false,
              .artifactSizeBytes = probe.metadata.artifactSizeBytes,
          });
    }
  } else {
    probe.status = NativeTextureCacheProbeStatus::Stale;
    recordProbeStatus(probe.status);
  }

  auto initialFingerprint = queryTextureSourceFingerprint(freshnessPath);
  if (initialFingerprint.hasError()) {
    return Result<TextureArtifactBuildResult, std::string>::makeError(
        initialFingerprint.error());
  }

  const auto buildStart = std::chrono::steady_clock::now();
  ScopedScratch scopedScratch(impl_->scratch);
  std::pmr::memory_resource *memory = scopedScratch.resource();
  Result<KtxTexture2Ptr, std::string> native =
      Result<KtxTexture2Ptr, std::string>::makeError("uninitialized");
  if (source.sourceKind == MaterialTextureSourceKind::ExternalFile &&
      isKtxPath(freshnessPath)) {
    auto direct = tryBuildDirectlyFromAuthoredKtx(freshnessPath, targetFormat);
    if (direct.hasError()) {
      return Result<TextureArtifactBuildResult, std::string>::makeError(
          direct.error());
    }
    if (direct.value().has_value()) {
      native = Result<KtxTexture2Ptr, std::string>::makeResult(
          std::move(*direct.value()));
    }
  }

  if (native.hasError() && native.error() == "uninitialized") {
    Result<ImageRgba8, std::string> image =
        Result<ImageRgba8, std::string>::makeError("uninitialized");
    if (source.sourceKind == MaterialTextureSourceKind::ExternalFile) {
      image = loadExternalRgba(freshnessPath, memory);
    } else {
      auto scene = impl_->ensureScene();
      if (scene.hasError()) {
        return Result<TextureArtifactBuildResult, std::string>::makeError(
            scene.error());
      }
      image = loadEmbeddedRgba(*scene.value(), source.embeddedIndex, memory);
    }
    if (image.hasError()) {
      return Result<TextureArtifactBuildResult, std::string>::makeError(
          image.error());
    }
    native = buildNativeFromImage(image.value(), targetFormat, options, memory);
  }
  if (native.hasError()) {
    return Result<TextureArtifactBuildResult, std::string>::makeError(
        native.error());
  }

  auto finalFingerprint = queryTextureSourceFingerprint(freshnessPath);
  if (finalFingerprint.hasError() ||
      finalFingerprint.value() != initialFingerprint.value()) {
    gTelemetry.nativeWriteFailures.fetch_add(1u, std::memory_order_relaxed);
    return Result<TextureArtifactBuildResult, std::string>::makeError(
        "Texture artifact builder: source changed during artifact build");
  }
  auto artifactWrite =
      writeKtxTextureAtomic(artifactPath.value(), *native.value());
  if (artifactWrite.hasError()) {
    gTelemetry.nativeWriteFailures.fetch_add(1u, std::memory_order_relaxed);
    return Result<TextureArtifactBuildResult, std::string>::makeError(
        artifactWrite.error());
  }

  const ktxTexture &texture = *ktxTexture(native.value().get());
  const NativeTextureCacheMetadata metadata{
      .profileVersion = kNativeTextureArtifactProfileVersion,
      .sourceIdentityHash = sourceIdentityHash,
      .source = initialFingerprint.value(),
      .targetFormat = targetFormat,
      .textureType = TextureType::Texture2D,
      .dimensions =
          {
              .width = std::max(1u, texture.baseWidth),
              .height = std::max(1u, texture.baseHeight),
              .depth = std::max(1u, texture.baseDepth),
          },
      .numLayers = std::max(1u, texture.numLayers),
      .numFaces = std::max(1u, texture.numFaces),
      .numMipLevels = std::max(1u, texture.numLevels),
      .payloadSizeBytes = tightPayloadSize(texture),
      .artifactSizeBytes = artifactWrite.value(),
  };
  auto metadataWrite =
      writeNativeTextureCacheMetadataAtomic(artifactPath.value(), metadata);
  if (metadataWrite.hasError()) {
    gTelemetry.nativeWriteFailures.fetch_add(1u, std::memory_order_relaxed);
    return Result<TextureArtifactBuildResult, std::string>::makeError(
        metadataWrite.error());
  }

  gTelemetry.authoredSourceBytesRead.fetch_add(
      initialFingerprint.value().sizeBytes, std::memory_order_relaxed);
  gTelemetry.artifactBuildTimeNs.fetch_add(elapsedNanoseconds(buildStart),
                                           std::memory_order_relaxed);
  gTelemetry.artifactBuilds.fetch_add(1u, std::memory_order_relaxed);
  gTelemetry.nativeWrites.fetch_add(1u, std::memory_order_relaxed);
  return Result<TextureArtifactBuildResult, std::string>::makeResult(
      TextureArtifactBuildResult{
          .artifactPath = artifactPath.value(),
          .targetFormat = targetFormat,
          .previousStatus = probe.status,
          .built = true,
          .artifactSizeBytes = artifactWrite.value(),
      });
}

Result<TextureArtifactBuildResult, std::string> ensureTextureArtifactFromFile(
    const std::filesystem::path &sourcePath, uint64_t sourceIdentityHash,
    Format targetFormat, const TextureArtifactBuildOptions &options,
    bool forceRebuild) {
  auto builder = SceneTextureArtifactBuilder::create(sourcePath);
  if (builder.hasError()) {
    return Result<TextureArtifactBuildResult, std::string>::makeError(
        builder.error());
  }
  return builder.value().ensure(
      MaterialTextureSlotData{
          .path = sourcePath.string(),
          .sourceKind = MaterialTextureSourceKind::ExternalFile,
      },
      sourceIdentityHash, targetFormat, options, forceRebuild);
}

uint32_t textureArtifactProcessingTag(
    const TextureArtifactBuildOptions &options) noexcept {
  uint32_t hash = 2166136261u;
  const auto mix = [&hash](uint32_t value) {
    hash ^= value;
    hash *= 16777619u;
  };
  mix(options.loadOptions.srgb ? 1u : 0u);
  mix(options.loadOptions.generateMipmaps ? 1u : 0u);
  mix(static_cast<uint32_t>(options.loadOptions.mipSemantic));
  mix(std::bit_cast<uint32_t>(options.loadOptions.alphaCoverageCutoff));
  mix(static_cast<uint32_t>(options.encoding));
  return hash;
}

TextureArtifactCacheTelemetry textureArtifactCacheTelemetry() noexcept {
  return TextureArtifactCacheTelemetry{
      .nativeHits = gTelemetry.nativeHits.load(std::memory_order_relaxed),
      .nativeMisses = gTelemetry.nativeMisses.load(std::memory_order_relaxed),
      .nativeStale = gTelemetry.nativeStale.load(std::memory_order_relaxed),
      .nativeCorrupt = gTelemetry.nativeCorrupt.load(std::memory_order_relaxed),
      .nativeWrites = gTelemetry.nativeWrites.load(std::memory_order_relaxed),
      .nativeWriteFailures =
          gTelemetry.nativeWriteFailures.load(std::memory_order_relaxed),
      .artifactBuilds =
          gTelemetry.artifactBuilds.load(std::memory_order_relaxed),
      .authoredSourceBytesRead =
          gTelemetry.authoredSourceBytesRead.load(std::memory_order_relaxed),
      .nativeArtifactBytesRead =
          gTelemetry.nativeArtifactBytesRead.load(std::memory_order_relaxed),
      .artifactBuildTimeNs =
          gTelemetry.artifactBuildTimeNs.load(std::memory_order_relaxed),
  };
}

} // namespace nuri
