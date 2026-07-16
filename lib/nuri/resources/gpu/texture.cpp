#include "nuri/pch.h"

#include "nuri/resources/gpu/texture.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/resources/cpu/bitmap.h"
#include "nuri/resources/storage/mesh/mesh_cache_utils.h"
#include "nuri/resources/storage/texture/texture_artifact_builder.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <ktx.h>
#include <stb_image.h>

namespace nuri {
namespace {

[[nodiscard]] uint32_t computeMipLevelCount(uint32_t width, uint32_t height) {
  uint32_t mipCount = 1u;
  uint32_t maxDim = std::max(width, height);
  while (maxDim > 1u) {
    maxDim >>= 1u;
    ++mipCount;
  }
  return mipCount;
}

[[nodiscard]] uint32_t mipDimension(uint32_t base, uint32_t mip) {
  return std::max(1u, base >> std::min(mip, 31u));
}

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
  if (rgba.empty()) {
    return 0.0f;
  }
  uint32_t covered = 0u;
  const size_t texelCount = rgba.size() / 4u;
  for (size_t i = 0u; i < texelCount; ++i) {
    covered += byteToUnit(rgba[i * 4u + 3u]) >= cutoff ? 1u : 0u;
  }
  return texelCount > 0u
             ? static_cast<float>(covered) / static_cast<float>(texelCount)
             : 0.0f;
}

[[nodiscard]] float scaledAlphaCoverage(std::span<const std::byte> rgba,
                                        float cutoff, float scale) {
  if (rgba.empty()) {
    return 0.0f;
  }
  uint32_t covered = 0u;
  const size_t texelCount = rgba.size() / 4u;
  for (size_t i = 0u; i < texelCount; ++i) {
    const float alpha = std::min(byteToUnit(rgba[i * 4u + 3u]) * scale, 1.0f);
    covered += alpha >= cutoff ? 1u : 0u;
  }
  return texelCount > 0u
             ? static_cast<float>(covered) / static_cast<float>(texelCount)
             : 0.0f;
}

void scaleAlphaCoverage(std::vector<std::byte> &rgba, float cutoff,
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

  for (size_t i = 0u; i + 3u < rgba.size(); i += 4u) {
    rgba[i + 3u] = static_cast<std::byte>(
        clampByteFromUnit(byteToUnit(rgba[i + 3u]) * hi));
  }
}

[[nodiscard]] std::vector<std::byte>
generateNextRgba8BoxMip(std::span<const std::byte> src, uint32_t srcWidth,
                        uint32_t srcHeight) {
  const uint32_t dstWidth = std::max(1u, srcWidth >> 1u);
  const uint32_t dstHeight = std::max(1u, srcHeight >> 1u);
  std::vector<std::byte> dst(static_cast<size_t>(dstWidth) * dstHeight * 4u);
  for (uint32_t y = 0u; y < dstHeight; ++y) {
    for (uint32_t x = 0u; x < dstWidth; ++x) {
      std::array<uint32_t, 4> sum{};
      uint32_t count = 0u;
      for (uint32_t dy = 0u; dy < 2u; ++dy) {
        const uint32_t sy = y * 2u + dy;
        if (sy >= srcHeight) {
          continue;
        }
        for (uint32_t dx = 0u; dx < 2u; ++dx) {
          const uint32_t sx = x * 2u + dx;
          if (sx >= srcWidth) {
            continue;
          }
          const size_t srcOffset =
              (static_cast<size_t>(sy) * srcWidth + sx) * 4u;
          for (uint32_t c = 0u; c < 4u; ++c) {
            sum[c] += std::to_integer<uint32_t>(src[srcOffset + c]);
          }
          ++count;
        }
      }
      const size_t dstOffset = (static_cast<size_t>(y) * dstWidth + x) * 4u;
      for (uint32_t c = 0u; c < 4u; ++c) {
        dst[dstOffset + c] =
            static_cast<std::byte>((sum[c] + count / 2u) / count);
      }
    }
  }
  return dst;
}

[[nodiscard]] std::vector<std::byte>
generateNextAlphaCoverageMip(std::span<const std::byte> src, uint32_t srcWidth,
                             uint32_t srcHeight, float alphaCutoff) {
  std::vector<std::byte> dst =
      generateNextRgba8BoxMip(src, srcWidth, srcHeight);
  scaleAlphaCoverage(dst, alphaCutoff, alphaCoverage(src, alphaCutoff));
  return dst;
}

[[nodiscard]] std::vector<std::byte>
generateNextNormalMip(std::span<const std::byte> src, uint32_t srcWidth,
                      uint32_t srcHeight) {
  const uint32_t dstWidth = std::max(1u, srcWidth >> 1u);
  const uint32_t dstHeight = std::max(1u, srcHeight >> 1u);
  std::vector<std::byte> dst(static_cast<size_t>(dstWidth) * dstHeight * 4u);
  for (uint32_t y = 0u; y < dstHeight; ++y) {
    for (uint32_t x = 0u; x < dstWidth; ++x) {
      glm::vec3 normalSum(0.0f);
      uint32_t alphaSum = 0u;
      uint32_t count = 0u;
      for (uint32_t dy = 0u; dy < 2u; ++dy) {
        const uint32_t sy = y * 2u + dy;
        if (sy >= srcHeight) {
          continue;
        }
        for (uint32_t dx = 0u; dx < 2u; ++dx) {
          const uint32_t sx = x * 2u + dx;
          if (sx >= srcWidth) {
            continue;
          }
          const size_t srcOffset =
              (static_cast<size_t>(sy) * srcWidth + sx) * 4u;
          normalSum += glm::vec3(byteToUnit(src[srcOffset + 0u]) * 2.0f - 1.0f,
                                 byteToUnit(src[srcOffset + 1u]) * 2.0f - 1.0f,
                                 byteToUnit(src[srcOffset + 2u]) * 2.0f - 1.0f);
          alphaSum += std::to_integer<uint32_t>(src[srcOffset + 3u]);
          ++count;
        }
      }

      const float lenSq = glm::dot(normalSum, normalSum);
      const glm::vec3 normal = lenSq > 1.0e-8f ? glm::normalize(normalSum)
                                               : glm::vec3(0.0f, 0.0f, 1.0f);
      const size_t dstOffset = (static_cast<size_t>(y) * dstWidth + x) * 4u;
      dst[dstOffset + 0u] =
          static_cast<std::byte>(clampByteFromUnit(normal.x * 0.5f + 0.5f));
      dst[dstOffset + 1u] =
          static_cast<std::byte>(clampByteFromUnit(normal.y * 0.5f + 0.5f));
      dst[dstOffset + 2u] =
          static_cast<std::byte>(clampByteFromUnit(normal.z * 0.5f + 0.5f));
      dst[dstOffset + 3u] =
          static_cast<std::byte>((alphaSum + count / 2u) / count);
    }
  }
  return dst;
}

[[nodiscard]] std::vector<std::byte>
generateNextRoughnessMip(std::span<const std::byte> src, uint32_t srcWidth,
                         uint32_t srcHeight, uint32_t roughnessChannel) {
  const uint32_t dstWidth = std::max(1u, srcWidth >> 1u);
  const uint32_t dstHeight = std::max(1u, srcHeight >> 1u);
  std::vector<std::byte> dst =
      generateNextRgba8BoxMip(src, srcWidth, srcHeight);
  for (uint32_t y = 0u; y < dstHeight; ++y) {
    for (uint32_t x = 0u; x < dstWidth; ++x) {
      float roughnessSqSum = 0.0f;
      uint32_t count = 0u;
      for (uint32_t dy = 0u; dy < 2u; ++dy) {
        const uint32_t sy = y * 2u + dy;
        if (sy >= srcHeight) {
          continue;
        }
        for (uint32_t dx = 0u; dx < 2u; ++dx) {
          const uint32_t sx = x * 2u + dx;
          if (sx >= srcWidth) {
            continue;
          }
          const size_t srcOffset =
              (static_cast<size_t>(sy) * srcWidth + sx) * 4u;
          const float roughness = byteToUnit(src[srcOffset + roughnessChannel]);
          roughnessSqSum += roughness * roughness;
          ++count;
        }
      }
      const float roughness =
          count > 0u ? std::sqrt(roughnessSqSum / static_cast<float>(count))
                     : 1.0f;
      const size_t dstOffset = (static_cast<size_t>(y) * dstWidth + x) * 4u;
      dst[dstOffset + roughnessChannel] =
          static_cast<std::byte>(clampByteFromUnit(roughness));
    }
  }
  return dst;
}

[[nodiscard]] Result<std::vector<std::byte>, std::string>
generateSemanticRgba8MipChain(std::span<const std::byte> baseData,
                              uint32_t width, uint32_t height,
                              uint32_t mipLevels,
                              const TextureLoadOptions &options) {
  const size_t requiredBaseBytes = static_cast<size_t>(std::max(width, 1u)) *
                                   static_cast<size_t>(std::max(height, 1u)) *
                                   4u;
  if (baseData.size() < requiredBaseBytes) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "Texture::generateSemanticRgba8MipChain: base mip data is truncated");
  }

  size_t totalBytes = 0u;
  for (uint32_t mip = 0u; mip < mipLevels; ++mip) {
    totalBytes += static_cast<size_t>(mipDimension(width, mip)) *
                  static_cast<size_t>(mipDimension(height, mip)) * 4u;
  }

  std::vector<std::byte> mipChain;
  mipChain.reserve(totalBytes);
  std::vector<std::byte> current(baseData.begin(),
                                 baseData.begin() +
                                     static_cast<ptrdiff_t>(requiredBaseBytes));
  uint32_t srcWidth = width;
  uint32_t srcHeight = height;
  const float alphaCutoff =
      sanitizeAlphaCoverageCutoff(options.alphaCoverageCutoff);

  for (uint32_t mip = 0u; mip < mipLevels; ++mip) {
    mipChain.insert(mipChain.end(), current.begin(), current.end());
    if (mip + 1u == mipLevels) {
      break;
    }

    if (options.mipSemantic == TextureMipSemantic::AlphaCoverage) {
      current = generateNextAlphaCoverageMip(current, srcWidth, srcHeight,
                                             alphaCutoff);
    } else if (options.mipSemantic == TextureMipSemantic::NormalMap) {
      current = generateNextNormalMip(current, srcWidth, srcHeight);
    } else {
      const uint32_t roughnessChannel =
          options.mipSemantic == TextureMipSemantic::RoughnessA ? 3u : 1u;
      current = generateNextRoughnessMip(current, srcWidth, srcHeight,
                                         roughnessChannel);
    }
    srcWidth = std::max(1u, srcWidth >> 1u);
    srcHeight = std::max(1u, srcHeight >> 1u);
  }

  return Result<std::vector<std::byte>, std::string>::makeResult(
      std::move(mipChain));
}

[[nodiscard]] bool
shouldGenerateSemanticRgba8MipChain(const TextureLoadOptions &options,
                                    uint32_t mipLevels) noexcept {
  return options.generateMipmaps && mipLevels > 1u &&
         (options.mipSemantic == TextureMipSemantic::AlphaCoverage ||
          options.mipSemantic == TextureMipSemantic::NormalMap ||
          options.mipSemantic == TextureMipSemantic::RoughnessG ||
          options.mipSemantic == TextureMipSemantic::RoughnessA);
}

struct KtxLoadPayload {
  TextureDesc desc{};
  std::vector<std::byte> bytes{};
  std::span<const std::byte> sourceBytes{};
  size_t dataOffset = 0u;
  size_t dataSize = 0u;
  std::string debugName{};

  void bindData() noexcept {
    const std::span<const std::byte> storage =
        bytes.empty() ? sourceBytes
                      : std::span<const std::byte>(bytes.data(), bytes.size());
    if (dataOffset > storage.size()) {
      desc.data = {};
      return;
    }
    const size_t available = storage.size() - dataOffset;
    const size_t boundedSize = std::min(dataSize, available);
    desc.data = storage.subspan(dataOffset, boundedSize);
  }
};

struct AtomicTextureCacheTelemetry {
  std::atomic<uint64_t> ddsSourceBytesRead{0u};
  std::atomic<uint64_t> ddsReadTimeNs{0u};
};

AtomicTextureCacheTelemetry gTextureCacheTelemetry{};

[[nodiscard]] uint64_t
elapsedNanoseconds(std::chrono::steady_clock::time_point start) noexcept {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start)
          .count());
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

struct FileCloser {
  void operator()(std::FILE *file) const noexcept {
    if (file != nullptr) {
      std::fclose(file);
    }
  }
};

using FilePtr = std::unique_ptr<std::FILE, FileCloser>;

[[nodiscard]] FilePtr openFileForRead(const std::filesystem::path &path) {
  std::FILE *file = nullptr;
#ifdef _WIN32
  if (_wfopen_s(&file, path.c_str(), L"rb") != 0) {
    file = nullptr;
  }
#else
  file = std::fopen(path.string().c_str(), "rb");
#endif
  return FilePtr(file);
}

constexpr ktx_uint32_t kGlRgba8 = 0x8058u;
constexpr ktx_uint32_t kGlSrgb8Alpha8 = 0x8C43u;
constexpr ktx_uint32_t kGlRgba16f = 0x881Au;
constexpr ktx_uint32_t kGlRgba32f = 0x8814u;
constexpr ktx_uint32_t kKtxVkFormatR8G8B8A8Unorm = 37u;
constexpr ktx_uint32_t kKtxVkFormatR8G8B8A8Srgb = 43u;
constexpr ktx_uint32_t kKtxVkFormatR16G16B16A16Sfloat = 97u;
constexpr ktx_uint32_t kKtxVkFormatR32G32B32A32Sfloat = 109u;
constexpr ktx_uint32_t kKtxVkFormatBc7Unorm = 145u;
constexpr ktx_uint32_t kKtxVkFormatBc7Srgb = 146u;
constexpr ktx_uint32_t kKtxVkFormatEtc2Rgb8Unorm = 147u;
constexpr ktx_uint32_t kKtxVkFormatEtc2Rgb8Srgb = 148u;
constexpr uint32_t kDdsMagic = 0x20534444u;
constexpr uint32_t kDdsFourCcDx10 = 0x30315844u;
constexpr uint32_t kDxgiFormatBc7Unorm = 98u;
constexpr uint32_t kDxgiFormatBc7Srgb = 99u;
constexpr float kMaxFiniteHalf = 65504.0f;

#pragma pack(push, 1)
struct DdsPixelFormatHeader {
  uint32_t size = 0;
  uint32_t flags = 0;
  uint32_t fourCc = 0;
  uint32_t rgbBitCount = 0;
  uint32_t rBitMask = 0;
  uint32_t gBitMask = 0;
  uint32_t bBitMask = 0;
  uint32_t aBitMask = 0;
};

struct DdsHeader {
  uint32_t size = 0;
  uint32_t flags = 0;
  uint32_t height = 0;
  uint32_t width = 0;
  uint32_t pitchOrLinearSize = 0;
  uint32_t depth = 0;
  uint32_t mipMapCount = 0;
  uint32_t reserved1[11]{};
  DdsPixelFormatHeader pixelFormat{};
  uint32_t caps = 0;
  uint32_t caps2 = 0;
  uint32_t caps3 = 0;
  uint32_t caps4 = 0;
  uint32_t reserved2 = 0;
};

struct DdsHeaderDx10 {
  uint32_t dxgiFormat = 0;
  uint32_t resourceDimension = 0;
  uint32_t miscFlag = 0;
  uint32_t arraySize = 0;
  uint32_t miscFlags2 = 0;
};
#pragma pack(pop)

static_assert(sizeof(DdsPixelFormatHeader) == 32u,
              "DDS pixel format header must match the DDS spec");
static_assert(sizeof(DdsHeader) == 124u, "DDS header must match the DDS spec");
static_assert(sizeof(DdsHeaderDx10) == 20u,
              "DDS DX10 header must match the DDS spec");

[[nodiscard]] bool hasExtension(std::string_view path,
                                std::string_view extension) {
  if (path.size() < extension.size()) {
    return false;
  }
  const size_t start = path.size() - extension.size();
  for (size_t i = 0; i < extension.size(); ++i) {
    const char lhs = static_cast<char>(
        std::tolower(static_cast<unsigned char>(path[start + i])));
    const char rhs = static_cast<char>(
        std::tolower(static_cast<unsigned char>(extension[i])));
    if (lhs != rhs) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] Result<Format, std::string>
resolveKtxTextureFormat(const ktxTexture *texture, std::string_view filePath) {
  if (texture == nullptr) {
    return Result<Format, std::string>::makeError(
        "Texture::resolveKtxTextureFormat: texture is null");
  }

  if (texture->classId == ktxTexture2_c) {
    const auto *texture2 = reinterpret_cast<const ktxTexture2 *>(texture);
    switch (texture2->vkFormat) {
    case kKtxVkFormatR8G8B8A8Unorm:
      return Result<Format, std::string>::makeResult(Format::RGBA8_UNORM);
    case kKtxVkFormatR8G8B8A8Srgb:
      return Result<Format, std::string>::makeResult(Format::RGBA8_SRGB);
    case kKtxVkFormatR16G16B16A16Sfloat:
      return Result<Format, std::string>::makeResult(Format::RGBA16_FLOAT);
    case kKtxVkFormatR32G32B32A32Sfloat:
      return Result<Format, std::string>::makeResult(Format::RGBA32_FLOAT);
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
          "Texture::resolveKtxTextureFormat: unsupported KTX2 vkFormat " +
          std::to_string(texture2->vkFormat) + " in '" + std::string(filePath) +
          "'");
    }
  }

  if (texture->classId == ktxTexture1_c) {
    const auto *texture1 = reinterpret_cast<const ktxTexture1 *>(texture);
    switch (texture1->glInternalformat) {
    case kGlRgba8:
      return Result<Format, std::string>::makeResult(Format::RGBA8_UNORM);
    case kGlSrgb8Alpha8:
      return Result<Format, std::string>::makeResult(Format::RGBA8_SRGB);
    case kGlRgba16f:
      return Result<Format, std::string>::makeResult(Format::RGBA16_FLOAT);
    case kGlRgba32f:
      return Result<Format, std::string>::makeResult(Format::RGBA32_FLOAT);
    default:
      return Result<Format, std::string>::makeError(
          "Texture::resolveKtxTextureFormat: unsupported KTX1 "
          "glInternalformat " +
          std::to_string(texture1->glInternalformat) + " in '" +
          std::string(filePath) + "'");
    }
  }

  return Result<Format, std::string>::makeError(
      "Texture::resolveKtxTextureFormat: unsupported KTX class in '" +
      std::string(filePath) + "'");
}

[[nodiscard]] bool isCubeTexture(const ktxTexture &texture) noexcept {
  return texture.numFaces == 6u;
}

[[nodiscard]] std::vector<std::byte>
convertFloatBitmapToHalfBytes(std::span<const uint8_t> srcBytes) {
  if ((srcBytes.size() % sizeof(float)) != 0u) {
    return {};
  }

  const size_t floatCount = srcBytes.size() / sizeof(float);
  std::vector<std::byte> dstBytes(floatCount * sizeof(uint16_t));

  for (size_t i = 0; i < floatCount; ++i) {
    float value = 0.0f;
    std::memcpy(&value, srcBytes.data() + (i * sizeof(float)), sizeof(float));
    if (std::isnan(value)) {
      value = 0.0f;
    } else if (std::isinf(value)) {
      value = std::signbit(value) ? 0.0f : kMaxFiniteHalf;
    } else {
      static std::atomic_bool warnedNegativeRadiance{false};
      if (value < 0.0f &&
          !warnedNegativeRadiance.exchange(true, std::memory_order_relaxed)) {
        NURI_LOG_WARNING(
            "Texture: finite negative radiance value encountered; clamping to "
            "0.0 for half-float upload");
      }
      value = std::clamp(value, 0.0f, kMaxFiniteHalf);
    }
    const uint16_t half = static_cast<uint16_t>(glm::packHalf1x16(value));
    std::memcpy(dstBytes.data() + (i * sizeof(uint16_t)), &half,
                sizeof(uint16_t));
  }

  return dstBytes;
}

[[nodiscard]] Result<KtxTexturePtr, std::string>
loadKtxTextureFromFile(std::string_view filePath,
                       ktxTextureCreateFlags createFlags =
                           KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT) {
  const std::string filePathStr(filePath);
  if (filePathStr.empty()) {
    return Result<KtxTexturePtr, std::string>::makeError(
        "Texture::loadKtxTextureFromFile: file path is empty");
  }
  FilePtr file = openFileForRead(std::filesystem::path(filePathStr));
  if (!file) {
    return Result<KtxTexturePtr, std::string>::makeError(
        "Texture::loadKtxTextureFromFile: failed to open KTX file '" +
        filePathStr + "'");
  }
  ktxTexture *texture = nullptr;
  const KTX_error_code createError =
      ktxTexture_CreateFromStdioStream(file.get(), createFlags, &texture);
  if (createError != KTX_SUCCESS || texture == nullptr) {
    return Result<KtxTexturePtr, std::string>::makeError(
        "Texture::loadKtxTextureFromFile: failed to read KTX file '" +
        filePathStr + "' (error " +
        std::to_string(static_cast<int>(createError)) + ")");
  }
  return Result<KtxTexturePtr, std::string>::makeResult(KtxTexturePtr(texture));
}

[[nodiscard]] Result<KtxLoadPayload, std::string>
makeKtxPayloadFromTexture(const ktxTexture &texture, std::string_view filePath,
                          std::string_view debugName, TextureType expectedType,
                          bool allowBasisPayload = false) {
  if (expectedType == TextureType::TextureCube && !isCubeTexture(texture)) {
    return Result<KtxLoadPayload, std::string>::makeError(
        "Texture::makeKtxPayloadFromTexture: expected a cubemap KTX file: '" +
        std::string(filePath) + "'");
  }
  if (expectedType == TextureType::Texture2D && isCubeTexture(texture)) {
    return Result<KtxLoadPayload, std::string>::makeError(
        "Texture::makeKtxPayloadFromTexture: expected a 2D KTX file but got "
        "cubemap: '" +
        std::string(filePath) + "'");
  }

  if (!allowBasisPayload && texture.classId == ktxTexture2_c) {
    auto *texture2 = const_cast<ktxTexture2 *>(
        reinterpret_cast<const ktxTexture2 *>(&texture));
    if (ktxTexture2_NeedsTranscoding(texture2)) {
      return Result<KtxLoadPayload, std::string>::makeError(
          "Texture::makeKtxPayloadFromTexture: Basis-compressed KTX2 file "
          "must be resolved through the texture artifact builder: '" +
          std::string(filePath) + "'");
    }
  }

  auto formatResult = resolveKtxTextureFormat(&texture, filePath);
  if (formatResult.hasError()) {
    return Result<KtxLoadPayload, std::string>::makeError(formatResult.error());
  }

  const uint8_t *srcData =
      ktxTexture_GetData(const_cast<ktxTexture *>(&texture));
  const size_t srcDataSize = static_cast<size_t>(
      ktxTexture_GetDataSize(const_cast<ktxTexture *>(&texture)));
  if (srcData == nullptr || srcDataSize == 0u) {
    return Result<KtxLoadPayload, std::string>::makeError(
        "Texture::makeKtxPayloadFromTexture: KTX texture has no image "
        "payload: '" +
        std::string(filePath) + "'");
  }

  KtxLoadPayload payload{};
  payload.desc.type = expectedType;
  payload.desc.format = formatResult.value();
  payload.desc.dimensions = {std::max(1u, texture.baseWidth),
                             std::max(1u, texture.baseHeight),
                             std::max(1u, texture.baseDepth)};
  payload.desc.usage = TextureUsage::Sampled;
  payload.desc.storage = Storage::Device;
  payload.desc.numLayers = std::max(1u, texture.numLayers);
  payload.desc.numSamples = 1u;
  payload.desc.numMipLevels = std::max(1u, texture.numLevels);
  payload.desc.dataNumMipLevels = payload.desc.numMipLevels;
  payload.desc.generateMipmaps = false;
  payload.debugName =
      debugName.empty() ? std::string(filePath) : std::string(debugName);

  const uint32_t numLayers = std::max(1u, texture.numLayers);
  const uint32_t numFaces = std::max(1u, texture.numFaces);
  size_t tightSizeBytes = 0u;
  ktxTexture *textureMutable = const_cast<ktxTexture *>(&texture);
  for (uint32_t level = 0u; level < payload.desc.numMipLevels; ++level) {
    const ktx_size_t imageSize = ktxTexture_GetImageSize(textureMutable, level);
    tightSizeBytes += static_cast<size_t>(imageSize) *
                      static_cast<size_t>(numLayers) *
                      static_cast<size_t>(numFaces);
  }
  if (tightSizeBytes == 0u) {
    return Result<KtxLoadPayload, std::string>::makeError(
        "Texture::makeKtxPayloadFromTexture: tight KTX payload size resolved "
        "to zero: '" +
        std::string(filePath) + "'");
  }

  payload.bytes.resize(tightSizeBytes);
  size_t dstOffset = 0u;
  for (uint32_t level = 0u; level < payload.desc.numMipLevels; ++level) {
    const ktx_size_t imageSize = ktxTexture_GetImageSize(textureMutable, level);
    for (uint32_t layer = 0u; layer < numLayers; ++layer) {
      for (uint32_t face = 0u; face < numFaces; ++face) {
        ktx_size_t srcOffset = 0u;
        const KTX_error_code offsetError = ktxTexture_GetImageOffset(
            textureMutable, level, layer, face, &srcOffset);
        if (offsetError != KTX_SUCCESS) {
          return Result<KtxLoadPayload, std::string>::makeError(
              "Texture::makeKtxPayloadFromTexture: failed to query KTX image "
              "offset in '" +
              std::string(filePath) + "'");
        }

        if (static_cast<size_t>(srcOffset) > srcDataSize ||
            static_cast<size_t>(imageSize) >
                (srcDataSize - static_cast<size_t>(srcOffset))) {
          return Result<KtxLoadPayload, std::string>::makeError(
              "Texture::makeKtxPayloadFromTexture: KTX image offset is out of "
              "bounds in '" +
              std::string(filePath) + "'");
        }
        if (dstOffset > payload.bytes.size() ||
            static_cast<size_t>(imageSize) >
                (payload.bytes.size() - dstOffset)) {
          return Result<KtxLoadPayload, std::string>::makeError(
              "Texture::makeKtxPayloadFromTexture: packed KTX output buffer "
              "overflow in '" +
              std::string(filePath) + "'");
        }

        std::memcpy(payload.bytes.data() + dstOffset,
                    reinterpret_cast<const std::byte *>(srcData) +
                        static_cast<size_t>(srcOffset),
                    static_cast<size_t>(imageSize));
        dstOffset += static_cast<size_t>(imageSize);
      }
    }
  }
  if (dstOffset != payload.bytes.size()) {
    return Result<KtxLoadPayload, std::string>::makeError(
        "Texture::makeKtxPayloadFromTexture: packed KTX output size mismatch "
        "in '" +
        std::string(filePath) + "'");
  }

  payload.dataOffset = 0u;
  payload.dataSize = payload.bytes.size();
  payload.bindData();
  return Result<KtxLoadPayload, std::string>::makeResult(std::move(payload));
}

[[nodiscard]] Result<KtxLoadPayload, std::string>
loadKtxPayload(std::string_view filePath, std::string_view debugName,
               TextureType expectedType) {
  auto textureResult = loadKtxTextureFromFile(filePath);
  if (textureResult.hasError()) {
    return Result<KtxLoadPayload, std::string>::makeError(
        textureResult.error());
  }
  return makeKtxPayloadFromTexture(*textureResult.value(), filePath, debugName,
                                   expectedType);
}

[[nodiscard]] Result<KtxLoadPayload, std::string>
parseDdsPayload(std::span<const std::byte> bytes, std::string_view sourceName,
                std::string_view debugName) {
  if (bytes.size() < sizeof(uint32_t) + sizeof(DdsHeader)) {
    return Result<KtxLoadPayload, std::string>::makeError(
        "Texture::loadDdsPayload: DDS file is too small");
  }

  const uint32_t magic = *reinterpret_cast<const uint32_t *>(bytes.data());
  if (magic != kDdsMagic) {
    return Result<KtxLoadPayload, std::string>::makeError(
        "Texture::loadDdsPayload: invalid DDS magic");
  }

  const DdsHeader &header =
      *reinterpret_cast<const DdsHeader *>(bytes.data() + sizeof(uint32_t));
  if (header.size != 124u || header.pixelFormat.size != 32u) {
    return Result<KtxLoadPayload, std::string>::makeError(
        "Texture::loadDdsPayload: unsupported DDS header layout");
  }
  if (header.width == 0u || header.height == 0u) {
    return Result<KtxLoadPayload, std::string>::makeError(
        "Texture::loadDdsPayload: DDS texture has invalid dimensions");
  }
  if (header.pixelFormat.fourCc != kDdsFourCcDx10) {
    return Result<KtxLoadPayload, std::string>::makeError(
        "Texture::loadDdsPayload: only DX10 DDS files are supported");
  }
  if (bytes.size() <
      sizeof(uint32_t) + sizeof(DdsHeader) + sizeof(DdsHeaderDx10)) {
    return Result<KtxLoadPayload, std::string>::makeError(
        "Texture::loadDdsPayload: DDS DX10 header is truncated");
  }

  const DdsHeaderDx10 &headerDx10 = *reinterpret_cast<const DdsHeaderDx10 *>(
      bytes.data() + sizeof(uint32_t) + sizeof(DdsHeader));
  if (headerDx10.resourceDimension != 3u || headerDx10.arraySize != 1u ||
      header.caps2 != 0u) {
    return Result<KtxLoadPayload, std::string>::makeError(
        "Texture::loadDdsPayload: only 2D non-array DDS textures are "
        "supported");
  }

  Format format = Format::RGBA8_UNORM;
  switch (headerDx10.dxgiFormat) {
  case kDxgiFormatBc7Unorm:
    format = Format::BC7_RGBA_UNORM;
    break;
  case kDxgiFormatBc7Srgb:
    format = Format::BC7_RGBA_SRGB;
    break;
  default:
    return Result<KtxLoadPayload, std::string>::makeError(
        "Texture::loadDdsPayload: unsupported DDS DXGI format " +
        std::to_string(headerDx10.dxgiFormat));
  }

  const size_t payloadOffset =
      sizeof(uint32_t) + sizeof(DdsHeader) + sizeof(DdsHeaderDx10);
  if (payloadOffset >= bytes.size()) {
    return Result<KtxLoadPayload, std::string>::makeError(
        "Texture::loadDdsPayload: DDS payload is empty");
  }
  const uint32_t numMipLevels = std::max(1u, header.mipMapCount);
  uint64_t expectedPayloadSize = 0u;
  for (uint32_t level = 0u; level < numMipLevels; ++level) {
    const uint32_t mipW = std::max(1u, header.width >> level);
    const uint32_t mipH = std::max(1u, header.height >> level);
    const uint32_t blocksX = (mipW + 3u) / 4u;
    const uint32_t blocksY = (mipH + 3u) / 4u;
    expectedPayloadSize +=
        static_cast<uint64_t>(blocksX) * static_cast<uint64_t>(blocksY) * 16u;
  }
  if (static_cast<uint64_t>(bytes.size() - payloadOffset) <
      expectedPayloadSize) {
    return Result<KtxLoadPayload, std::string>::makeError(
        "Texture::loadDdsPayload: truncated or insufficient DDS payload for "
        "BC7 format");
  }

  KtxLoadPayload payload{};
  payload.desc.type = TextureType::Texture2D;
  payload.desc.format = format;
  payload.desc.dimensions = {std::max(1u, header.width),
                             std::max(1u, header.height), 1u};
  payload.desc.usage = TextureUsage::Sampled;
  payload.desc.storage = Storage::Device;
  payload.desc.numLayers = 1u;
  payload.desc.numSamples = 1u;
  payload.desc.numMipLevels = numMipLevels;
  payload.desc.dataNumMipLevels = payload.desc.numMipLevels;
  payload.desc.generateMipmaps = false;
  payload.debugName =
      debugName.empty() ? std::string(sourceName) : std::string(debugName);
  payload.sourceBytes = bytes;
  payload.dataOffset = payloadOffset;
  payload.dataSize = static_cast<size_t>(expectedPayloadSize);
  payload.bindData();
  return Result<KtxLoadPayload, std::string>::makeResult(std::move(payload));
}

[[nodiscard]] Result<KtxLoadPayload, std::string>
loadDdsPayload(std::string_view filePath, std::string_view debugName) {
  const auto readStart = std::chrono::steady_clock::now();
  auto bytesResult =
      readBinaryFile(std::filesystem::path(std::string(filePath)));
  if (bytesResult.hasError()) {
    return Result<KtxLoadPayload, std::string>::makeError(bytesResult.error());
  }

  std::vector<std::byte> bytes = std::move(bytesResult.value());
  gTextureCacheTelemetry.ddsSourceBytesRead.fetch_add(
      bytes.size(), std::memory_order_relaxed);
  gTextureCacheTelemetry.ddsReadTimeNs.fetch_add(elapsedNanoseconds(readStart),
                                                 std::memory_order_relaxed);
  auto payloadResult = parseDdsPayload(bytes, filePath, debugName);
  if (payloadResult.hasError()) {
    return payloadResult;
  }
  KtxLoadPayload payload = std::move(payloadResult.value());
  payload.bytes = std::move(bytes);
  payload.sourceBytes = {};
  payload.bindData();
  return Result<KtxLoadPayload, std::string>::makeResult(std::move(payload));
}

[[nodiscard]] Result<std::unique_ptr<Texture>, std::string>
createTextureFromPayload(GPUDevice &gpu, KtxLoadPayload payload) {
  payload.bindData();
  return Texture::create(gpu, payload.desc, payload.debugName);
}

} // namespace

Result<std::unique_ptr<Texture>, std::string>
Texture::create(GPUDevice &gpu, const TextureDesc &desc,
                std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  auto result = gpu.createTexture(desc, debugName);
  if (result.hasError()) {
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        result.error());
  }

  return Result<std::unique_ptr<Texture>, std::string>::makeResult(
      std::unique_ptr<Texture>(
          new Texture(gpu, result.value(), desc, std::string(debugName))));
}

Result<std::unique_ptr<Texture>, std::string>
Texture::loadTexture(GPUDevice &gpu, std::string_view filePath,
                     std::string_view debugName) {
  return loadTexture(gpu, filePath, TextureLoadOptions{}, debugName);
}

Result<std::unique_ptr<Texture>, std::string>
Texture::loadTexture(GPUDevice &gpu, std::string_view filePath,
                     const TextureLoadOptions &options,
                     std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (hasExtension(filePath, ".dds")) {
    auto payloadResult = loadDdsPayload(filePath, debugName);
    if (payloadResult.hasError()) {
      return Result<std::unique_ptr<Texture>, std::string>::makeError(
          payloadResult.error());
    }
    return createTextureFromPayload(gpu, std::move(payloadResult.value()));
  }

  const std::string filePathStr(filePath);
  int32_t width = 0;
  int32_t height = 0;
  int32_t channels = 0;
  void *pixels = stbi_load(filePathStr.c_str(), &width, &height, &channels, 4);
  if (!pixels) {
    NURI_LOG_WARNING("Texture::loadTexture: Failed to load texture '%s': %s",
                     filePathStr.c_str(), stbi_failure_reason());
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        "Failed to load texture from file: " + filePathStr + " " +
        stbi_failure_reason());
  }

  const size_t dataSize =
      static_cast<size_t>(width) * static_cast<size_t>(height) * 4u;
  const std::span<const std::byte> initialData{
      static_cast<const std::byte *>(pixels), dataSize};

  const uint32_t widthU32 = static_cast<uint32_t>(width);
  const uint32_t heightU32 = static_cast<uint32_t>(height);
  const uint32_t mipLevels =
      options.generateMipmaps ? computeMipLevelCount(widthU32, heightU32) : 1u;
  std::vector<std::byte> semanticMipData;
  std::span<const std::byte> textureData = initialData;
  uint32_t dataMipLevels = 1u;
  bool generateMipmaps = options.generateMipmaps;
  if (shouldGenerateSemanticRgba8MipChain(options, mipLevels)) {
    auto mipResult = generateSemanticRgba8MipChain(
        initialData, widthU32, heightU32, mipLevels, options);
    if (mipResult.hasError()) {
      stbi_image_free(pixels);
      return Result<std::unique_ptr<Texture>, std::string>::makeError(
          mipResult.error());
    }
    semanticMipData = std::move(mipResult).value();
    textureData = std::span<const std::byte>(semanticMipData.data(),
                                             semanticMipData.size());
    dataMipLevels = mipLevels;
    generateMipmaps = false;
  }

  TextureDesc desc{
      .type = TextureType::Texture2D,
      .format = options.srgb ? Format::RGBA8_SRGB : Format::RGBA8_UNORM,
      .dimensions = {widthU32, heightU32, 1},
      .usage = TextureUsage::Sampled,
      .storage = Storage::Device,
      .numLayers = 1,
      .numSamples = 1,
      .numMipLevels = mipLevels,
      .data = textureData,
      .dataNumMipLevels = dataMipLevels,
      .generateMipmaps = generateMipmaps,
  };
  auto result = gpu.createTexture(desc, debugName);
  if (result.hasError()) {
    NURI_LOG_WARNING("Texture::loadTexture: Failed to create texture '%s': %s",
                     filePathStr.c_str(), result.error().c_str());
    stbi_image_free(pixels);
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        result.error());
  }

  stbi_image_free(pixels);

  NURI_LOG_DEBUG("Texture::loadTexture: Created texture from file '%s'",
                 filePathStr.c_str());

  return Result<std::unique_ptr<Texture>, std::string>::makeResult(
      std::unique_ptr<Texture>(
          new Texture(gpu, result.value(), desc, std::string(debugName))));
}

Result<std::unique_ptr<Texture>, std::string>
Texture::loadDdsTexture(GPUDevice &gpu, std::span<const std::byte> fileBytes,
                        std::string_view sourceName,
                        std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  auto payloadResult = parseDdsPayload(fileBytes, sourceName, debugName);
  if (payloadResult.hasError()) {
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        payloadResult.error());
  }
  return createTextureFromPayload(gpu, std::move(payloadResult.value()));
}

Result<std::unique_ptr<Texture>, std::string>
Texture::loadCubemapFromEquirectangularHDR(GPUDevice &gpu,
                                           std::string_view filePath,
                                           std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  const std::string filePathStr(filePath);
  int32_t width = 0;
  int32_t height = 0;
  int32_t channels = 0;
  float *pixels =
      stbi_loadf(filePathStr.c_str(), &width, &height, &channels, 4);
  if (!pixels) {
    const char *reason = stbi_failure_reason();
    NURI_LOG_WARNING(
        "Texture::loadCubemapFromEquirectangularHDR: Failed to load '%s': %s",
        filePathStr.c_str(), reason ? reason : "unknown error");
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        "Failed to load HDR texture from file: " + filePathStr + " " +
        (reason ? std::string(reason) : std::string("unknown error")));
  }

  const Bitmap equirectangular(width, height, 4, BitmapFormat::F32, pixels);
  stbi_image_free(pixels);

  const Bitmap cubemapFaces =
      equirectangular.convertEquirectangularMapToCubeMapFaces();
  if (cubemapFaces.empty()) {
    NURI_LOG_WARNING("Texture::loadCubemapFromEquirectangularHDR: Failed to "
                     "convert equirectangular HDR to cubemap faces '%s'",
                     filePathStr.c_str());
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        "Failed to convert HDR texture to cubemap faces: " + filePathStr);
  }

  std::vector<std::byte> halfBytes =
      convertFloatBitmapToHalfBytes(cubemapFaces.data());
  if (halfBytes.empty()) {
    NURI_LOG_WARNING("Texture::loadCubemapFromEquirectangularHDR: Failed to "
                     "convert cubemap data to RGBA16F '%s'",
                     filePathStr.c_str());
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        "Failed to convert cubemap face data to RGBA16F: " + filePathStr);
  }

  const std::span<const std::byte> initialData(halfBytes.data(),
                                               halfBytes.size());

  TextureDesc desc{
      .type = TextureType::TextureCube,
      .format = Format::RGBA16_FLOAT,
      .dimensions = {static_cast<uint32_t>(cubemapFaces.width()),
                     static_cast<uint32_t>(cubemapFaces.height()), 1},
      .usage = TextureUsage::Sampled,
      .storage = Storage::Device,
      .numLayers = 1,
      .numSamples = 1,
      .numMipLevels = 1,
      .data = initialData,
      .dataNumMipLevels = 1,
      .generateMipmaps = false,
  };
  const std::string resolvedDebugName =
      debugName.empty() ? filePathStr : std::string(debugName);
  auto result = gpu.createTexture(desc, resolvedDebugName);
  if (result.hasError()) {
    NURI_LOG_WARNING("Texture::loadCubemapFromEquirectangularHDR: Failed to "
                     "create cubemap texture '%s': %s",
                     filePathStr.c_str(), result.error().c_str());
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        result.error());
  }

  NURI_LOG_DEBUG("Texture::loadCubemapFromEquirectangularHDR: Created cubemap "
                 "from file '%s'",
                 filePathStr.c_str());

  return Result<std::unique_ptr<Texture>, std::string>::makeResult(
      std::unique_ptr<Texture>(
          new Texture(gpu, result.value(), desc, resolvedDebugName)));
}

Result<std::unique_ptr<Texture>, std::string>
Texture::loadTextureKtx2(GPUDevice &gpu, std::string_view filePath,
                         std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  auto payloadResult =
      loadKtxPayload(filePath, debugName, TextureType::Texture2D);
  if (payloadResult.hasError()) {
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        payloadResult.error());
  }
  return createTextureFromPayload(gpu, std::move(payloadResult.value()));
}

Result<std::unique_ptr<Texture>, std::string>
Texture::loadCubemapKtx2(GPUDevice &gpu, std::string_view filePath,
                         std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  auto payloadResult =
      loadKtxPayload(filePath, debugName, TextureType::TextureCube);
  if (payloadResult.hasError()) {
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        payloadResult.error());
  }
  return createTextureFromPayload(gpu, std::move(payloadResult.value()));
}

TextureCacheTelemetry Texture::cacheTelemetry() noexcept {
  const TextureArtifactCacheTelemetry artifact =
      textureArtifactCacheTelemetry();
  return TextureCacheTelemetry{
      .nativeHits = artifact.nativeHits,
      .nativeMisses = artifact.nativeMisses,
      .nativeStale = artifact.nativeStale,
      .nativeCorrupt = artifact.nativeCorrupt,
      .nativeWrites = artifact.nativeWrites,
      .nativeWriteFailures = artifact.nativeWriteFailures,
      .artifactBuilds = artifact.artifactBuilds,
      .authoredSourceBytesRead = artifact.authoredSourceBytesRead,
      .nativeArtifactBytesRead = artifact.nativeArtifactBytesRead,
      .ddsSourceBytesRead = gTextureCacheTelemetry.ddsSourceBytesRead.load(
          std::memory_order_relaxed),
      .artifactBuildTimeNs = artifact.artifactBuildTimeNs,
      .ddsReadTimeNs =
          gTextureCacheTelemetry.ddsReadTimeNs.load(std::memory_order_relaxed),
  };
}

} // namespace nuri
