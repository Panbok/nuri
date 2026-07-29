#include "nuri/resources/gpu/texture.h"
#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/resources/cpu/bitmap.h"
#include "nuri/resources/storage/cache_utils.h"
#include "nuri/resources/storage/texture/texture_artifact_builder.h"
#include "nuri/resources/storage/texture/texture_ktx.h"
#include "nuri/resources/storage/texture/texture_processing.h"
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <stb_image.h>
namespace nuri {
namespace {
using detail::KtxTexturePtr;
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
  return detail::loadKtxTexture(std::filesystem::path(filePath), "Texture",
                                createFlags);
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
  auto formatResult = detail::resolveKtxFormat(texture, filePath);
  if (formatResult.hasError()) {
    return Result<KtxLoadPayload, std::string>::makeError(formatResult.error());
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
    for (uint32_t layer = 0u; layer < numLayers; ++layer) {
      for (uint32_t face = 0u; face < numFaces; ++face) {
        auto image = detail::viewKtxImage(*textureMutable, level, layer, face,
                                          "Texture::makeKtxPayloadFromTexture");
        if (image.hasError()) {
          return Result<KtxLoadPayload, std::string>::makeError(
              image.error() + " in '" + std::string(filePath) + "'");
        }
        if (dstOffset > payload.bytes.size() ||
            image.value().size() > payload.bytes.size() - dstOffset) {
          return Result<KtxLoadPayload, std::string>::makeError(
              "Texture::makeKtxPayloadFromTexture: packed KTX output buffer "
              "overflow in '" +
              std::string(filePath) + "'");
        }
        std::memcpy(payload.bytes.data() + dstOffset, image.value().data(),
                    image.value().size());
        dstOffset += image.value().size();
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
[[nodiscard]] Result<PreparedTextureData, std::string>
prepareTextureFromPayload(KtxLoadPayload payload) {
  payload.bindData();
  PreparedTextureData prepared{};
  prepared.createDesc = payload.desc;
  prepared.createDesc.data = {};
  prepared.bytes.assign(payload.desc.data.begin(), payload.desc.data.end());
  prepared.debugName = std::move(payload.debugName);
  return Result<PreparedTextureData, std::string>::makeResult(
      std::move(prepared));
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
Texture::createPrepared(GPUDevice &gpu, PreparedTextureData data) {
  const TextureDesc desc = data.descriptor();
  return create(gpu, desc, data.debugName);
}

std::unique_ptr<Texture> Texture::adoptPrepared(GPUDevice &gpu,
                                                TextureHandle handle,
                                                const TextureDesc &desc,
                                                std::string debugName) {
  if (!nuri::isValid(handle)) {
    return nullptr;
  }
  return std::unique_ptr<Texture>(
      new Texture(gpu, handle, desc, std::move(debugName)));
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
  auto prepared = prepareTexture(filePath, options, debugName);
  if (prepared.hasError()) {
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        prepared.error());
  }
  return createPrepared(gpu, std::move(prepared.value()));
}

Result<PreparedTextureData, std::string>
Texture::prepareTexture(std::string_view filePath,
                        const TextureLoadOptions &options,
                        std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (hasExtensionCaseInsensitive(filePath, ".dds")) {
    auto payloadResult = loadDdsPayload(filePath, debugName);
    if (payloadResult.hasError()) {
      return Result<PreparedTextureData, std::string>::makeError(
          payloadResult.error());
    }
    return prepareTextureFromPayload(std::move(payloadResult.value()));
  }
  const std::string filePathStr(filePath);
  int32_t width = 0;
  int32_t height = 0;
  int32_t channels = 0;
  void *pixels = stbi_load(filePathStr.c_str(), &width, &height, &channels, 4);
  if (!pixels) {
    NURI_LOG_WARNING("Texture::loadTexture: Failed to load texture '%s': %s",
                     filePathStr.c_str(), stbi_failure_reason());
    return Result<PreparedTextureData, std::string>::makeError(
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
      options.generateMipmaps ? textureMipLevelCount(widthU32, heightU32) : 1u;
  std::vector<std::byte> semanticMipData;
  std::span<const std::byte> textureData = initialData;
  uint32_t dataMipLevels = 1u;
  bool generateMipmaps = options.generateMipmaps;
  if (shouldGenerateSemanticRgba8MipChain(options, mipLevels)) {
    auto mipResult = generateSemanticRgba8MipChain(
        initialData, widthU32, heightU32, mipLevels, options);
    if (mipResult.hasError()) {
      stbi_image_free(pixels);
      return Result<PreparedTextureData, std::string>::makeError(
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
  PreparedTextureData prepared{};
  prepared.createDesc = desc;
  prepared.createDesc.data = {};
  prepared.debugName = debugName.empty() ? filePathStr : std::string(debugName);
  if (!semanticMipData.empty()) {
    prepared.bytes = std::move(semanticMipData);
  } else {
    prepared.bytes.assign(initialData.begin(), initialData.end());
  }
  stbi_image_free(pixels);
  NURI_LOG_DEBUG("Texture::prepareTexture: Prepared texture from file '%s'",
                 filePathStr.c_str());
  return Result<PreparedTextureData, std::string>::makeResult(
      std::move(prepared));
}

Result<std::unique_ptr<Texture>, std::string>
Texture::loadDdsTexture(GPUDevice &gpu, std::span<const std::byte> fileBytes,
                        std::string_view sourceName,
                        std::string_view debugName) {
  auto prepared = prepareDdsTexture(fileBytes, sourceName, debugName);
  if (prepared.hasError()) {
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        prepared.error());
  }
  return createPrepared(gpu, std::move(prepared.value()));
}

Result<PreparedTextureData, std::string>
Texture::prepareDdsTexture(std::span<const std::byte> fileBytes,
                           std::string_view sourceName,
                           std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  auto payloadResult = parseDdsPayload(fileBytes, sourceName, debugName);
  if (payloadResult.hasError()) {
    return Result<PreparedTextureData, std::string>::makeError(
        payloadResult.error());
  }
  return prepareTextureFromPayload(std::move(payloadResult.value()));
}

Result<std::unique_ptr<Texture>, std::string>
Texture::loadCubemapFromEquirectangularHDR(GPUDevice &gpu,
                                           std::string_view filePath,
                                           std::string_view debugName) {
  auto prepared = prepareCubemapFromEquirectangularHDR(filePath, debugName);
  if (prepared.hasError()) {
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        prepared.error());
  }
  return createPrepared(gpu, std::move(prepared.value()));
}

Result<PreparedTextureData, std::string>
Texture::prepareCubemapFromEquirectangularHDR(std::string_view filePath,
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
    return Result<PreparedTextureData, std::string>::makeError(
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
    return Result<PreparedTextureData, std::string>::makeError(
        "Failed to convert HDR texture to cubemap faces: " + filePathStr);
  }
  std::vector<std::byte> halfBytes =
      convertFloatBitmapToHalfBytes(cubemapFaces.data());
  if (halfBytes.empty()) {
    NURI_LOG_WARNING("Texture::loadCubemapFromEquirectangularHDR: Failed to "
                     "convert cubemap data to RGBA16F '%s'",
                     filePathStr.c_str());
    return Result<PreparedTextureData, std::string>::makeError(
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
  NURI_LOG_DEBUG(
      "Texture::prepareCubemapFromEquirectangularHDR: Prepared cubemap "
      "from file '%s'",
      filePathStr.c_str());
  desc.data = {};
  return Result<PreparedTextureData, std::string>::makeResult(
      PreparedTextureData{
          .createDesc = desc,
          .bytes = std::move(halfBytes),
          .debugName = resolvedDebugName,
      });
}

Result<std::unique_ptr<Texture>, std::string>
Texture::loadTextureKtx2(GPUDevice &gpu, std::string_view filePath,
                         std::string_view debugName) {
  auto prepared = prepareTextureKtx2(filePath, debugName);
  if (prepared.hasError()) {
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        prepared.error());
  }
  return createPrepared(gpu, std::move(prepared.value()));
}

Result<PreparedTextureData, std::string>
Texture::prepareTextureKtx2(std::string_view filePath,
                            std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  auto payloadResult =
      loadKtxPayload(filePath, debugName, TextureType::Texture2D);
  if (payloadResult.hasError()) {
    return Result<PreparedTextureData, std::string>::makeError(
        payloadResult.error());
  }
  return prepareTextureFromPayload(std::move(payloadResult.value()));
}

Result<std::unique_ptr<Texture>, std::string>
Texture::loadCubemapKtx2(GPUDevice &gpu, std::string_view filePath,
                         std::string_view debugName) {
  auto prepared = prepareCubemapKtx2(filePath, debugName);
  if (prepared.hasError()) {
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        prepared.error());
  }
  return createPrepared(gpu, std::move(prepared.value()));
}

Result<PreparedTextureData, std::string>
Texture::prepareCubemapKtx2(std::string_view filePath,
                            std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  auto payloadResult =
      loadKtxPayload(filePath, debugName, TextureType::TextureCube);
  if (payloadResult.hasError()) {
    return Result<PreparedTextureData, std::string>::makeError(
        payloadResult.error());
  }
  return prepareTextureFromPayload(std::move(payloadResult.value()));
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
      .normalVarianceArtifactBuilds = artifact.normalVarianceArtifactBuilds,
      .normalVarianceCleanTexels = artifact.normalVarianceCleanTexels,
      .normalVarianceToksvigFallbackTexels =
          artifact.normalVarianceToksvigFallbackTexels,
      .normalVarianceContractRejections =
          artifact.normalVarianceContractRejections,
      .normalVarianceArtifactBytesWritten =
          artifact.normalVarianceArtifactBytesWritten,
      .normalVarianceArtifactBuildTimeNs =
          artifact.normalVarianceArtifactBuildTimeNs,
  };
}

} // namespace nuri
