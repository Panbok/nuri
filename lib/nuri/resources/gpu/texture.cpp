#include "nuri/pch.h"

#include "nuri/resources/gpu/texture.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/resources/cpu/bitmap.h"
#include "nuri/resources/storage/mesh/mesh_cache_utils.h"
#include "nuri/resources/storage/texture/texture_cache_utils.h"

#include <cmath>
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

struct KtxLoadPayload {
  TextureDesc desc{};
  std::vector<std::byte> bytes{};
  std::string debugName{};
};

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
      value = std::clamp(value, 0.0f, kMaxFiniteHalf);
    }
    const uint16_t half = static_cast<uint16_t>(glm::packHalf1x16(value));
    std::memcpy(dstBytes.data() + (i * sizeof(uint16_t)), &half,
                sizeof(uint16_t));
  }

  return dstBytes;
}

[[nodiscard]] Result<KtxTexturePtr, std::string>
loadKtxTextureFromFile(std::string_view filePath) {
  const std::string filePathStr(filePath);
  if (filePathStr.empty()) {
    return Result<KtxTexturePtr, std::string>::makeError(
        "Texture::loadKtxTextureFromFile: file path is empty");
  }
  ktxTexture *texture = nullptr;
  const KTX_error_code createError = ktxTexture_CreateFromNamedFile(
      filePathStr.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &texture);
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
          "requires the portable texture path: '" +
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

  payload.desc.data =
      std::span<const std::byte>(payload.bytes.data(), payload.bytes.size());
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
loadDdsPayload(std::string_view filePath, std::string_view debugName) {
  auto bytesResult =
      readBinaryFile(std::filesystem::path(std::string(filePath)));
  if (bytesResult.hasError()) {
    return Result<KtxLoadPayload, std::string>::makeError(bytesResult.error());
  }

  std::vector<std::byte> bytes = std::move(bytesResult.value());
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
      debugName.empty() ? std::string(filePath) : std::string(debugName);
  payload.bytes.assign(bytes.begin() + static_cast<ptrdiff_t>(payloadOffset),
                       bytes.end());
  payload.desc.data =
      std::span<const std::byte>(payload.bytes.data(), payload.bytes.size());
  return Result<KtxLoadPayload, std::string>::makeResult(std::move(payload));
}

[[nodiscard]] Result<std::unique_ptr<Texture>, std::string>
createTextureFromPayload(GPUDevice &gpu, KtxLoadPayload payload) {
  payload.desc.data =
      std::span<const std::byte>(payload.bytes.data(), payload.bytes.size());
  return Texture::create(gpu, payload.desc, payload.debugName);
}

[[nodiscard]] Format
selectPortableRuntimeFormat(const GPUDevice &gpu, bool srgb,
                            uint32_t componentCount) noexcept {
  if (!srgb) {
    return Format::RGBA8_UNORM;
  }
  const TextureCompressionCaps caps = gpu.getTextureCompressionCaps();
  if (caps.bc7) {
    return Format::BC7_RGBA_SRGB;
  }
  if (caps.etc2 && componentCount <= 3u) {
    return Format::ETC2_RGB8_SRGB;
  }
  if (caps.astc) {
    // ASTC support is reported separately, but the backend Format enum does not
    // expose an ASTC target yet, so portable runtime transcoding still falls
    // back to uncompressed RGBA until ASTC formats are added end-to-end.
  }
  return Format::RGBA8_SRGB;
}

[[nodiscard]] bool shouldPersistPortableNativeCache(Format format) noexcept {
  return format != Format::RGBA8_UNORM;
}

[[nodiscard]] std::filesystem::path
resolvePortableNativeCachePath(std::string_view portablePath,
                               Format targetFormat, bool persistNativeCache) {
  if (!persistNativeCache) {
    return {};
  }
  return buildNativeTextureCachePath(std::filesystem::path(portablePath),
                                     targetFormat);
}

[[nodiscard]] Result<ktx_uint32_t, std::string>
resolveKtxVkFormat(Format format) {
  switch (format) {
  case Format::RGBA8_UNORM:
    return Result<ktx_uint32_t, std::string>::makeResult(
        kKtxVkFormatR8G8B8A8Unorm);
  case Format::RGBA8_SRGB:
    return Result<ktx_uint32_t, std::string>::makeResult(
        kKtxVkFormatR8G8B8A8Srgb);
  case Format::BC7_RGBA_UNORM:
    return Result<ktx_uint32_t, std::string>::makeResult(kKtxVkFormatBc7Unorm);
  case Format::BC7_RGBA_SRGB:
    return Result<ktx_uint32_t, std::string>::makeResult(kKtxVkFormatBc7Srgb);
  case Format::ETC2_RGB8_UNORM:
    return Result<ktx_uint32_t, std::string>::makeResult(
        kKtxVkFormatEtc2Rgb8Unorm);
  case Format::ETC2_RGB8_SRGB:
    return Result<ktx_uint32_t, std::string>::makeResult(
        kKtxVkFormatEtc2Rgb8Srgb);
  default:
    return Result<ktx_uint32_t, std::string>::makeError(
        "Texture::resolveKtxVkFormat: unsupported target format");
  }
}

[[nodiscard]] Result<ktx_transcode_fmt_e, std::string>
resolveBasisTranscodeFormat(Format format) {
  switch (format) {
  case Format::BC7_RGBA_UNORM:
  case Format::BC7_RGBA_SRGB:
    return Result<ktx_transcode_fmt_e, std::string>::makeResult(
        KTX_TTF_BC7_RGBA);
  case Format::ETC2_RGB8_UNORM:
  case Format::ETC2_RGB8_SRGB:
    return Result<ktx_transcode_fmt_e, std::string>::makeResult(
        KTX_TTF_ETC1_RGB);
  case Format::RGBA8_UNORM:
  case Format::RGBA8_SRGB:
    return Result<ktx_transcode_fmt_e, std::string>::makeResult(KTX_TTF_RGBA32);
  default:
    return Result<ktx_transcode_fmt_e, std::string>::makeError(
        "Texture::resolveBasisTranscodeFormat: unsupported target format");
  }
}

[[nodiscard]] Result<std::unique_ptr<ktxTexture2, KtxTextureDeleter>,
                     std::string>
createNativeKtx2Copy(const ktxTexture &source, Format targetFormat) {
  auto vkFormatResult = resolveKtxVkFormat(targetFormat);
  if (vkFormatResult.hasError()) {
    return Result<std::unique_ptr<ktxTexture2, KtxTextureDeleter>,
                  std::string>::makeError(vkFormatResult.error());
  }

  const ktxTextureCreateInfo createInfo{
      .glInternalformat = 0u,
      .vkFormat = vkFormatResult.value(),
      .pDfd = nullptr,
      .baseWidth = source.baseWidth,
      .baseHeight = source.baseHeight,
      .baseDepth = source.baseDepth,
      .numDimensions = source.numDimensions,
      .numLevels = source.numLevels,
      .numLayers = source.numLayers,
      .numFaces = source.numFaces,
      .isArray = source.isArray,
      .generateMipmaps = KTX_FALSE,
  };

  ktxTexture2 *outTexture = nullptr;
  const KTX_error_code createError = ktxTexture2_Create(
      &createInfo, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &outTexture);
  if (createError != KTX_SUCCESS || outTexture == nullptr) {
    return Result<
        std::unique_ptr<ktxTexture2, KtxTextureDeleter>,
        std::string>::makeError("Texture::createNativeKtx2Copy: "
                                "ktxTexture2_Create failed with code " +
                                std::to_string(static_cast<int>(createError)));
  }

  std::unique_ptr<ktxTexture2, KtxTextureDeleter> out(outTexture);
  ktxTexture *outBase = ktxTexture(out.get());
  ktxTexture *srcBase = const_cast<ktxTexture *>(&source);
  const uint8_t *srcData = ktxTexture_GetData(srcBase);
  if (srcData == nullptr) {
    return Result<std::unique_ptr<ktxTexture2, KtxTextureDeleter>,
                  std::string>::
        makeError("Texture::createNativeKtx2Copy: source texture data is null");
  }

  for (uint32_t level = 0u; level < source.numLevels; ++level) {
    const ktx_size_t imageSize = ktxTexture_GetImageSize(srcBase, level);
    for (uint32_t layer = 0u; layer < std::max(1u, source.numLayers); ++layer) {
      for (uint32_t face = 0u; face < std::max(1u, source.numFaces); ++face) {
        ktx_size_t srcOffset = 0u;
        const KTX_error_code offsetError =
            ktxTexture_GetImageOffset(srcBase, level, layer, face, &srcOffset);
        if (offsetError != KTX_SUCCESS) {
          return Result<
              std::unique_ptr<ktxTexture2, KtxTextureDeleter>,
              std::string>::makeError("Texture::createNativeKtx2Copy: failed "
                                      "to query image offset");
        }
        const KTX_error_code setError = ktxTexture_SetImageFromMemory(
            outBase, level, layer, face, srcData + srcOffset, imageSize);
        if (setError != KTX_SUCCESS) {
          return Result<
              std::unique_ptr<ktxTexture2, KtxTextureDeleter>,
              std::string>::makeError("Texture::createNativeKtx2Copy: failed "
                                      "to copy image payload");
        }
      }
    }
  }

  return Result<std::unique_ptr<ktxTexture2, KtxTextureDeleter>,
                std::string>::makeResult(std::move(out));
}

[[nodiscard]] Result<std::vector<std::byte>, std::string>
serializeKtxTextureToBytes(ktxTexture2 &texture) {
  ktx_uint8_t *bytes = nullptr;
  ktx_size_t sizeBytes = 0u;
  const KTX_error_code writeError =
      ktxTexture2_WriteToMemory(&texture, &bytes, &sizeBytes);
  if (writeError != KTX_SUCCESS || bytes == nullptr || sizeBytes == 0u) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "Texture::serializeKtxTextureToBytes: ktxTexture2_WriteToMemory "
        "failed with code " +
        std::to_string(static_cast<int>(writeError)));
  }

  std::vector<std::byte> out(reinterpret_cast<std::byte *>(bytes),
                             reinterpret_cast<std::byte *>(bytes) + sizeBytes);
  std::free(bytes);
  return Result<std::vector<std::byte>, std::string>::makeResult(
      std::move(out));
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
          new Texture(result.value(), desc, std::string(debugName))));
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
  TextureDesc desc{
      .type = TextureType::Texture2D,
      .format = options.srgb ? Format::RGBA8_SRGB : Format::RGBA8_UNORM,
      .dimensions = {widthU32, heightU32, 1},
      .usage = TextureUsage::Sampled,
      .storage = Storage::Device,
      .numLayers = 1,
      .numSamples = 1,
      .numMipLevels = mipLevels,
      .data = initialData,
      .dataNumMipLevels = 1,
      .generateMipmaps = options.generateMipmaps,
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
          new Texture(result.value(), desc, std::string(debugName))));
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
          new Texture(result.value(), desc, resolvedDebugName)));
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
Texture::loadPortableTextureKtx2(GPUDevice &gpu, std::string_view filePath,
                                 const TextureLoadOptions &options,
                                 std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  auto portableTextureResult = loadKtxTextureFromFile(filePath);
  if (portableTextureResult.hasError()) {
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        portableTextureResult.error());
  }

  ktxTexture &portableTexture = *portableTextureResult.value();
  if (portableTexture.classId != ktxTexture2_c) {
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        "Texture::loadPortableTextureKtx2: portable source must be a KTX2 "
        "texture");
  }
  auto *portableTexture2 =
      reinterpret_cast<ktxTexture2 *>(portableTextureResult.value().get());
  if (isCubeTexture(portableTexture)) {
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        "Texture::loadPortableTextureKtx2: portable scene textures must be 2D");
  }

  const uint32_t componentCount =
      ktxTexture2_GetNumComponents(portableTexture2);
  if (!ktxTexture2_NeedsTranscoding(portableTexture2)) {
    auto payloadResult = makeKtxPayloadFromTexture(
        portableTexture, filePath, debugName, TextureType::Texture2D);
    if (payloadResult.hasError()) {
      return Result<std::unique_ptr<Texture>, std::string>::makeError(
          payloadResult.error());
    }
    return createTextureFromPayload(gpu, std::move(payloadResult.value()));
  }
  const Format targetFormat =
      selectPortableRuntimeFormat(gpu, options.srgb, componentCount);
  const bool persistNativeCache =
      shouldPersistPortableNativeCache(targetFormat);
  const std::filesystem::path nativeCachePath = resolvePortableNativeCachePath(
      filePath, targetFormat, persistNativeCache);
  if (persistNativeCache &&
      isTextureCacheUpToDate(nativeCachePath,
                             std::filesystem::path(filePath))) {
    auto nativePayload = loadKtxPayload(nativeCachePath.string(), debugName,
                                        TextureType::Texture2D);
    if (!nativePayload.hasError()) {
      NURI_LOG_DEBUG("Texture::loadPortableTextureKtx2: native cache hit '%s'",
                     nativeCachePath.string().c_str());
      return createTextureFromPayload(gpu, std::move(nativePayload.value()));
    }
    NURI_LOG_WARNING(
        "Texture::loadPortableTextureKtx2: failed to load native cache '%s': "
        "%s",
        nativeCachePath.string().c_str(), nativePayload.error().c_str());
  }

  auto transcodeFmtResult = resolveBasisTranscodeFormat(targetFormat);
  if (transcodeFmtResult.hasError()) {
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        transcodeFmtResult.error());
  }
  const ktx_transcode_flags flags = (targetFormat == Format::ETC2_RGB8_UNORM ||
                                     targetFormat == Format::ETC2_RGB8_SRGB)
                                        ? KTX_TF_HIGH_QUALITY
                                        : 0u;
  const KTX_error_code transcodeError = ktxTexture2_TranscodeBasis(
      portableTexture2, transcodeFmtResult.value(), flags);
  if (transcodeError != KTX_SUCCESS) {
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        "Texture::loadPortableTextureKtx2: failed to transcode Basis payload "
        "'" +
        std::string(filePath) + "' (error " +
        std::to_string(static_cast<int>(transcodeError)) + ")");
  }

  auto nativeTextureResult =
      createNativeKtx2Copy(portableTexture, targetFormat);
  if (nativeTextureResult.hasError()) {
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        nativeTextureResult.error());
  }

  if (persistNativeCache) {
    auto nativeFileBytesResult =
        serializeKtxTextureToBytes(*nativeTextureResult.value());
    if (!nativeFileBytesResult.hasError()) {
      auto writeResult =
          writeBinaryFileAtomic(nativeCachePath, nativeFileBytesResult.value());
      if (writeResult.hasError()) {
        NURI_LOG_WARNING(
            "Texture::loadPortableTextureKtx2: failed to write native cache "
            "'%s': %s",
            nativeCachePath.string().c_str(), writeResult.error().c_str());
      } else {
        NURI_LOG_DEBUG(
            "Texture::loadPortableTextureKtx2: native cache built '%s'",
            nativeCachePath.string().c_str());
      }
    } else {
      NURI_LOG_WARNING(
          "Texture::loadPortableTextureKtx2: failed to serialize native cache "
          "'%s': %s",
          nativeCachePath.string().c_str(),
          nativeFileBytesResult.error().c_str());
    }
  }

  const std::string payloadSourcePath =
      (persistNativeCache && !nativeCachePath.empty())
          ? nativeCachePath.string()
          : (debugName.empty() ? std::string(filePath)
                               : std::string(debugName));
  auto payloadResult = makeKtxPayloadFromTexture(
      *ktxTexture(nativeTextureResult.value().get()), payloadSourcePath,
      debugName, TextureType::Texture2D);
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

} // namespace nuri
