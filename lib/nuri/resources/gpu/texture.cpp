#include "nuri/resources/gpu/texture.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/resources/cpu/bitmap.h"

#include "nuri/pch.h"

#include <ktx.h>
#include <stb_image.h>
#include <vulkan/vulkan_core.h>

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
};

constexpr ktx_uint32_t kGlRgba8 = 0x8058u;
constexpr ktx_uint32_t kGlSrgb8Alpha8 = 0x8C43u;
constexpr ktx_uint32_t kGlRgba16f = 0x881Au;
constexpr ktx_uint32_t kGlRgba32f = 0x8814u;

[[nodiscard]] Result<Format, std::string>
resolveKtxTextureFormat(const ktxTexture *texture,
                        std::string_view filePath) {
  if (texture == nullptr) {
    return Result<Format, std::string>::makeError(
        "Texture::resolveKtxTextureFormat: texture is null");
  }

  if (texture->classId == ktxTexture2_c) {
    const auto *texture2 = reinterpret_cast<const ktxTexture2 *>(texture);
    switch (texture2->vkFormat) {
    case VK_FORMAT_R8G8B8A8_UNORM:
      return Result<Format, std::string>::makeResult(Format::RGBA8_UNORM);
    case VK_FORMAT_R8G8B8A8_SRGB:
      return Result<Format, std::string>::makeResult(Format::RGBA8_SRGB);
    case VK_FORMAT_R16G16B16A16_SFLOAT:
      return Result<Format, std::string>::makeResult(Format::RGBA16_FLOAT);
    case VK_FORMAT_R32G32B32A32_SFLOAT:
      return Result<Format, std::string>::makeResult(Format::RGBA32_FLOAT);
    default:
      return Result<Format, std::string>::makeError(
          "Texture::resolveKtxTextureFormat: unsupported KTX2 vkFormat " +
          std::to_string(texture2->vkFormat) + " in '" +
          std::string(filePath) + "'");
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
          "Texture::resolveKtxTextureFormat: unsupported KTX1 glInternalformat " +
          std::to_string(texture1->glInternalformat) + " in '" +
          std::string(filePath) + "'");
    }
  }

  return Result<Format, std::string>::makeError(
      "Texture::resolveKtxTextureFormat: unsupported KTX class in '" +
      std::string(filePath) + "'");
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
    const uint16_t half = static_cast<uint16_t>(glm::packHalf1x16(value));
    std::memcpy(dstBytes.data() + (i * sizeof(uint16_t)), &half,
                sizeof(uint16_t));
  }

  return dstBytes;
}

[[nodiscard]] Result<KtxLoadPayload, std::string>
loadKtxPayload(std::string_view filePath, std::string_view debugName,
               TextureType expectedType) {
  const std::string filePathStr(filePath);
  if (filePathStr.empty()) {
    return Result<KtxLoadPayload, std::string>::makeError(
        "Texture::loadKtxPayload: file path is empty");
  }

  ktxTexture *texture = nullptr;
  const KTX_error_code createError = ktxTexture_CreateFromNamedFile(
      filePathStr.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &texture);
  if (createError != KTX_SUCCESS || texture == nullptr) {
    return Result<KtxLoadPayload, std::string>::makeError(
        "Texture::loadKtxPayload: failed to read KTX file '" + filePathStr +
        "' (error " + std::to_string(static_cast<int>(createError)) + ")");
  }
  std::unique_ptr<ktxTexture, KtxTextureDeleter> textureGuard(texture);

  const bool isCube = texture->numFaces == 6u;
  if (expectedType == TextureType::TextureCube && !isCube) {
    return Result<KtxLoadPayload, std::string>::makeError(
        "Texture::loadKtxPayload: expected a cubemap KTX file: '" +
        filePathStr + "'");
  }
  if (expectedType == TextureType::Texture2D && isCube) {
    return Result<KtxLoadPayload, std::string>::makeError(
        "Texture::loadKtxPayload: expected a 2D KTX file but got cubemap: '" +
        filePathStr + "'");
  }

  const uint32_t width = std::max(1u, texture->baseWidth);
  const uint32_t height = std::max(1u, texture->baseHeight);
  const uint32_t depth = std::max(1u, texture->baseDepth);
  const uint32_t mipLevels = std::max(1u, texture->numLevels);
  const uint32_t numLayers = std::max(1u, texture->numLayers);
  const size_t srcDataSize = static_cast<size_t>(ktxTexture_GetDataSize(texture));
  const uint8_t *srcData = ktxTexture_GetData(texture);
  if (srcData == nullptr || srcDataSize == 0u) {
    return Result<KtxLoadPayload, std::string>::makeError(
        "Texture::loadKtxPayload: KTX2 file has no image payload: '" +
        filePathStr + "'");
  }

  KtxLoadPayload payload{};
  payload.desc.type = expectedType;
  payload.desc.dimensions = {width, height, depth};
  payload.desc.usage = TextureUsage::Sampled;
  payload.desc.storage = Storage::Device;
  payload.desc.numLayers = numLayers;
  payload.desc.numSamples = 1u;
  payload.desc.numMipLevels = mipLevels;
  payload.desc.dataNumMipLevels = mipLevels;
  payload.desc.generateMipmaps = false;
  payload.debugName = debugName.empty() ? filePathStr : std::string(debugName);

  auto formatResult = resolveKtxTextureFormat(texture, filePathStr);
  if (formatResult.hasError()) {
    return Result<KtxLoadPayload, std::string>::makeError(
        formatResult.error());
  }
  payload.desc.format = formatResult.value();

  const auto bytesPerPixelForFormat = [](Format format) -> uint32_t {
    switch (format) {
    case Format::RGBA8_UNORM:
    case Format::RGBA8_SRGB:
      return 4u;
    case Format::RGBA16_FLOAT:
      return 8u;
    case Format::RGBA32_FLOAT:
      return 16u;
    default:
      return 0u;
    }
  };

  const uint32_t bytesPerPixel = bytesPerPixelForFormat(payload.desc.format);
  if (bytesPerPixel == 0u) {
    return Result<KtxLoadPayload, std::string>::makeError(
        "Texture::loadKtxPayload: unsupported pixel size for resolved format "
        "in '" +
        filePathStr + "'");
  }

  const uint32_t numFaces = std::max(1u, texture->numFaces);
  size_t tightSizeBytes = 0u;
  for (uint32_t level = 0u; level < mipLevels; ++level) {
    const uint32_t levelWidth = std::max(1u, width >> level);
    const uint32_t levelHeight = std::max(1u, height >> level);
    const uint32_t levelDepth = std::max(1u, depth >> level);
    const size_t imageBytes = static_cast<size_t>(levelWidth) *
                              static_cast<size_t>(levelHeight) *
                              static_cast<size_t>(levelDepth) *
                              static_cast<size_t>(bytesPerPixel);
    tightSizeBytes += imageBytes * static_cast<size_t>(numLayers) *
                      static_cast<size_t>(numFaces);
  }

  payload.bytes.resize(tightSizeBytes);
  size_t dstOffset = 0u;
  for (uint32_t level = 0u; level < mipLevels; ++level) {
    const uint32_t levelWidth = std::max(1u, width >> level);
    const uint32_t levelHeight = std::max(1u, height >> level);
    const uint32_t levelDepth = std::max(1u, depth >> level);
    const size_t imageBytes = static_cast<size_t>(levelWidth) *
                              static_cast<size_t>(levelHeight) *
                              static_cast<size_t>(levelDepth) *
                              static_cast<size_t>(bytesPerPixel);
    for (uint32_t layer = 0u; layer < numLayers; ++layer) {
      for (uint32_t face = 0u; face < numFaces; ++face) {
        ktx_size_t srcOffset = 0u;
        const KTX_error_code offsetError = ktxTexture_GetImageOffset(
            texture, level, layer, face, &srcOffset);
        if (offsetError != KTX_SUCCESS) {
          return Result<KtxLoadPayload, std::string>::makeError(
              "Texture::loadKtxPayload: failed to get KTX image offset in '" +
              filePathStr + "' (error " +
              std::to_string(static_cast<int>(offsetError)) + ")");
        }

        if (static_cast<size_t>(srcOffset) > srcDataSize ||
            imageBytes > (srcDataSize - static_cast<size_t>(srcOffset))) {
          return Result<KtxLoadPayload, std::string>::makeError(
              "Texture::loadKtxPayload: KTX image offset is out of bounds in '" +
              filePathStr + "'");
        }

        if (dstOffset > payload.bytes.size() ||
            imageBytes > (payload.bytes.size() - dstOffset)) {
          return Result<KtxLoadPayload, std::string>::makeError(
              "Texture::loadKtxPayload: packed KTX output buffer overflow in '" +
              filePathStr + "'");
        }

        std::memcpy(payload.bytes.data() + dstOffset,
                    srcData + static_cast<size_t>(srcOffset), imageBytes);
        dstOffset += imageBytes;
      }
    }
  }

  if (dstOffset != payload.bytes.size()) {
    return Result<KtxLoadPayload, std::string>::makeError(
        "Texture::loadKtxPayload: packed KTX output size mismatch in '" +
        filePathStr + "'");
  }

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

  // Bitmap copies pixel data via memcpy; safe to free stbi buffer after
  // construction.
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
