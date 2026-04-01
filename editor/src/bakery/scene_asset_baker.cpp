#include "nuri/editor_pch.h"

#include "nuri/bakery/scene_asset_baker.h"

#include "nuri/core/log.h"
#include "nuri/core/pmr_scratch.h"
#include "nuri/core/profiling.h"
#include "nuri/resources/gpu/resource_keys.h"
#include "nuri/resources/mesh_importer.h"
#include "nuri/resources/storage/material/material_binary_serializer.h"
#include "nuri/resources/storage/material/material_cache_utils.h"
#include "nuri/resources/storage/mesh/mesh_cache_utils.h"
#include "nuri/resources/storage/texture/texture_cache_utils.h"

#include <array>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <atomic>
#include <cstdlib>
#include <format>
#include <ktx.h>
#include <memory_resource>
#include <stb_image.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>
#include <thread>

namespace nuri::bakery::detail {
namespace {

enum class PortableTextureEncodeMode : uint8_t { Raw, ETC1S, UASTC };

struct SlotBakeSpec {
  MaterialTextureSlotData MaterialData::*slot = nullptr;
  bool srgb = false;
  PortableTextureEncodeMode encodeMode = PortableTextureEncodeMode::ETC1S;
  bool normalMap = false;
};

constexpr std::array<SlotBakeSpec, kMaterialTextureSlotCount> kSlotBakeSpecs{{
    {&MaterialData::baseColor, true, PortableTextureEncodeMode::UASTC, false},
    {&MaterialData::metallicRoughness, false, PortableTextureEncodeMode::ETC1S,
     false},
    {&MaterialData::normal, false, PortableTextureEncodeMode::UASTC, true},
    {&MaterialData::occlusion, false, PortableTextureEncodeMode::ETC1S, false},
    {&MaterialData::emissive, true, PortableTextureEncodeMode::UASTC, false},
    {&MaterialData::clearcoat, false, PortableTextureEncodeMode::ETC1S, false},
    {&MaterialData::clearcoatRoughness, false, PortableTextureEncodeMode::ETC1S,
     false},
    {&MaterialData::clearcoatNormal, false, PortableTextureEncodeMode::UASTC,
     true},
    {&MaterialData::specular, false, PortableTextureEncodeMode::ETC1S, false},
    {&MaterialData::specularColor, true, PortableTextureEncodeMode::UASTC,
     false},
    {&MaterialData::sheenColor, true, PortableTextureEncodeMode::UASTC, false},
    {&MaterialData::sheenRoughness, false, PortableTextureEncodeMode::ETC1S,
     false},
    {&MaterialData::transmission, false, PortableTextureEncodeMode::ETC1S,
     false},
    {&MaterialData::thickness, false, PortableTextureEncodeMode::ETC1S, false},
}};

constexpr ktx_uint32_t kKtxVkFormatR8G8B8A8Unorm = 37u;
constexpr ktx_uint32_t kKtxVkFormatR8G8B8A8Srgb = 43u;
constexpr ktx_uint32_t kKtxVkFormatBc7Unorm = 145u;
constexpr ktx_uint32_t kKtxVkFormatBc7Srgb = 146u;
constexpr ktx_uint32_t kKtxVkFormatEtc2Rgb8Unorm = 147u;
constexpr ktx_uint32_t kKtxVkFormatEtc2Rgb8Srgb = 148u;

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

[[nodiscard]] uint32_t
buildSlotBakeSettingsTag(const SlotBakeSpec &spec) noexcept {
  constexpr uint32_t kMipGenerationPolicyFullChain = 1u << 9u;
  return (spec.srgb ? 1u : 0u) |
         (static_cast<uint32_t>(spec.encodeMode) << 1u) |
         (spec.normalMap ? (1u << 8u) : 0u) | kMipGenerationPolicyFullChain;
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

using KtxTexture2Ptr = std::unique_ptr<ktxTexture2, KtxTextureDeleter>;
using KtxTexturePtr = std::unique_ptr<ktxTexture, KtxTextureDeleter>;

struct NativeTargetBuild {
  ScenePortableTextureTarget target = ScenePortableTextureTarget::BC7;
  std::filesystem::path path{};
};

using NativeTargetBuildList = std::pmr::vector<NativeTargetBuild>;

struct ImageRgba8 {
  int32_t width = 0;
  int32_t height = 0;
  std::pmr::vector<std::byte> bytes;

  explicit ImageRgba8(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : bytes(memory != nullptr ? memory : std::pmr::get_default_resource()) {}
};

[[nodiscard]] std::pmr::memory_resource *
resolveMemoryResource(std::pmr::memory_resource *memory) noexcept {
  return memory != nullptr ? memory : std::pmr::get_default_resource();
}

[[nodiscard]] Result<std::pmr::vector<std::byte>, std::string>
generateNextMipLevel(const std::byte *srcBytes, int32_t srcWidth,
                     int32_t srcHeight, int32_t dstWidth, int32_t dstHeight,
                     const SlotBakeSpec &spec,
                     std::pmr::memory_resource *memory) {
  auto *resource = resolveMemoryResource(memory);
  std::pmr::vector<std::byte> out(resource);
  out.resize(static_cast<size_t>(dstWidth) * static_cast<size_t>(dstHeight) *
             4u);

  if (spec.normalMap) {
    std::pmr::vector<float> srcFloats(resource);
    std::pmr::vector<float> dstFloats(resource);
    srcFloats.resize(static_cast<size_t>(srcWidth) *
                     static_cast<size_t>(srcHeight) * 4u);
    dstFloats.resize(static_cast<size_t>(dstWidth) *
                     static_cast<size_t>(dstHeight) * 4u);

    const size_t srcPixelCount =
        static_cast<size_t>(srcWidth) * static_cast<size_t>(srcHeight);
    for (size_t i = 0; i < srcPixelCount; ++i) {
      const size_t byteOffset = i * 4u;
      const size_t floatOffset = i * 4u;
      const glm::vec3 normal = glm::normalize(
          glm::vec3((static_cast<float>(
                         std::to_integer<uint8_t>(srcBytes[byteOffset + 0u])) /
                     255.0f) *
                            2.0f -
                        1.0f,
                    (static_cast<float>(
                         std::to_integer<uint8_t>(srcBytes[byteOffset + 1u])) /
                     255.0f) *
                            2.0f -
                        1.0f,
                    (static_cast<float>(
                         std::to_integer<uint8_t>(srcBytes[byteOffset + 2u])) /
                     255.0f) *
                            2.0f -
                        1.0f));
      srcFloats[floatOffset + 0u] = normal.x;
      srcFloats[floatOffset + 1u] = normal.y;
      srcFloats[floatOffset + 2u] = normal.z;
      srcFloats[floatOffset + 3u] = static_cast<float>(std::to_integer<uint8_t>(
                                        srcBytes[byteOffset + 3u])) /
                                    255.0f;
    }

    if (stbir_resize_float_linear(srcFloats.data(), srcWidth, srcHeight, 0,
                                  dstFloats.data(), dstWidth, dstHeight, 0,
                                  STBIR_RGBA) == nullptr) {
      return Result<std::pmr::vector<std::byte>, std::string>::makeError(
          "Scene asset baker: failed to resize normal-map mip level");
    }

    const size_t dstPixelCount =
        static_cast<size_t>(dstWidth) * static_cast<size_t>(dstHeight);
    for (size_t i = 0; i < dstPixelCount; ++i) {
      const size_t byteOffset = i * 4u;
      const size_t floatOffset = i * 4u;
      glm::vec3 normal(dstFloats[floatOffset + 0u], dstFloats[floatOffset + 1u],
                       dstFloats[floatOffset + 2u]);
      if (glm::dot(normal, normal) > 0.0f) {
        normal = glm::normalize(normal);
      } else {
        normal = glm::vec3(0.0f, 0.0f, 1.0f);
      }
      out[byteOffset + 0u] = static_cast<std::byte>(
          std::clamp(std::lround((normal.x * 0.5f + 0.5f) * 255.0f), 0l, 255l));
      out[byteOffset + 1u] = static_cast<std::byte>(
          std::clamp(std::lround((normal.y * 0.5f + 0.5f) * 255.0f), 0l, 255l));
      out[byteOffset + 2u] = static_cast<std::byte>(
          std::clamp(std::lround((normal.z * 0.5f + 0.5f) * 255.0f), 0l, 255l));
      out[byteOffset + 3u] = static_cast<std::byte>(std::clamp(
          std::lround(dstFloats[floatOffset + 3u] * 255.0f), 0l, 255l));
    }

    return Result<std::pmr::vector<std::byte>, std::string>::makeResult(
        std::move(out));
  }

  unsigned char *dstBytes = reinterpret_cast<unsigned char *>(out.data());
  const unsigned char *src = reinterpret_cast<const unsigned char *>(srcBytes);
  unsigned char *resizeResult =
      spec.srgb
          ? stbir_resize_uint8_srgb(src, srcWidth, srcHeight, 0, dstBytes,
                                    dstWidth, dstHeight, 0, STBIR_RGBA)
          : stbir_resize_uint8_linear(src, srcWidth, srcHeight, 0, dstBytes,
                                      dstWidth, dstHeight, 0, STBIR_RGBA);
  if (resizeResult == nullptr) {
    return Result<std::pmr::vector<std::byte>, std::string>::makeError(
        "Scene asset baker: failed to resize mip level");
  }
  return Result<std::pmr::vector<std::byte>, std::string>::makeResult(
      std::move(out));
}

[[nodiscard]] Result<ImageRgba8, std::string>
loadExternalTextureKtxRgba(const std::filesystem::path &path,
                           std::pmr::memory_resource *memory) {
  ktxTexture *texture = nullptr;
  const std::string pathString = path.string();
  const KTX_error_code loadError = ktxTexture_CreateFromNamedFile(
      pathString.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &texture);
  if (loadError != KTX_SUCCESS || texture == nullptr) {
    return Result<ImageRgba8, std::string>::makeError(
        "Scene asset baker: failed to read KTX texture '" + pathString +
        "' (error " + std::to_string(static_cast<int>(loadError)) + ")");
  }

  KtxTexturePtr texturePtr(texture);
  if (texture->numFaces != 1u || texture->numLayers > 1u) {
    return Result<ImageRgba8, std::string>::makeError(
        "Scene asset baker: only 2D KTX textures are supported for scene "
        "asset baking");
  }

  if (texture->classId == ktxTexture2_c) {
    auto *texture2 = reinterpret_cast<ktxTexture2 *>(texturePtr.get());
    if (ktxTexture2_NeedsTranscoding(texture2)) {
      const KTX_error_code transcodeError =
          ktxTexture2_TranscodeBasis(texture2, KTX_TTF_RGBA32, 0u);
      if (transcodeError != KTX_SUCCESS) {
        return Result<ImageRgba8, std::string>::makeError(
            "Scene asset baker: failed to transcode KTX texture '" +
            pathString + "' (error " +
            std::to_string(static_cast<int>(transcodeError)) + ")");
      }
    }
  }

  if (texture->baseWidth == 0u || texture->baseHeight == 0u) {
    return Result<ImageRgba8, std::string>::makeError(
        "Scene asset baker: KTX texture has invalid dimensions");
  }

  const ktx_size_t imageSize = ktxTexture_GetImageSize(texturePtr.get(), 0u);
  const size_t expectedSize = static_cast<size_t>(texture->baseWidth) *
                              static_cast<size_t>(texture->baseHeight) * 4u;
  if (static_cast<size_t>(imageSize) != expectedSize) {
    return Result<ImageRgba8, std::string>::makeError(
        "Scene asset baker: KTX texture is not RGBA8 after load/transcode");
  }

  ktx_size_t imageOffset = 0u;
  const KTX_error_code offsetError =
      ktxTexture_GetImageOffset(texturePtr.get(), 0u, 0u, 0u, &imageOffset);
  if (offsetError != KTX_SUCCESS) {
    return Result<ImageRgba8, std::string>::makeError(
        "Scene asset baker: failed to query KTX image offset");
  }

  const uint8_t *srcData = ktxTexture_GetData(texturePtr.get());
  if (srcData == nullptr) {
    return Result<ImageRgba8, std::string>::makeError(
        "Scene asset baker: KTX texture payload is empty");
  }

  ImageRgba8 out(memory);
  out.width = static_cast<int32_t>(texture->baseWidth);
  out.height = static_cast<int32_t>(texture->baseHeight);
  out.bytes.assign(reinterpret_cast<const std::byte *>(srcData + imageOffset),
                   reinterpret_cast<const std::byte *>(srcData + imageOffset) +
                       expectedSize);
  return Result<ImageRgba8, std::string>::makeResult(std::move(out));
}

[[nodiscard]] Result<ImageRgba8, std::string>
loadExternalTextureRgba(const std::filesystem::path &path,
                        std::pmr::memory_resource *memory) {
  if (path.has_extension()) {
    std::string extension = path.extension().string();
    std::ranges::transform(extension, extension.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    if (extension == ".ktx" || extension == ".ktx2") {
      return loadExternalTextureKtxRgba(path, memory);
    }
  }

  int32_t width = 0;
  int32_t height = 0;
  int32_t channels = 0;
  stbi_uc *pixels =
      stbi_load(path.string().c_str(), &width, &height, &channels, 4);
  if (pixels == nullptr) {
    const char *reason = stbi_failure_reason();
    return Result<ImageRgba8, std::string>::makeError(
        "Scene asset baker: failed to load texture '" + path.string() +
        "': " + (reason ? std::string(reason) : std::string("unknown error")));
  }

  ImageRgba8 out(memory);
  out.width = width;
  out.height = height;
  out.bytes.assign(reinterpret_cast<std::byte *>(pixels),
                   reinterpret_cast<std::byte *>(pixels) +
                       static_cast<size_t>(width) *
                           static_cast<size_t>(height) * 4u);
  stbi_image_free(pixels);
  return Result<ImageRgba8, std::string>::makeResult(std::move(out));
}

[[nodiscard]] Result<ImageRgba8, std::string>
loadEmbeddedTextureRgba(const aiScene &scene, uint32_t embeddedIndex,
                        std::pmr::memory_resource *memory) {
  if (embeddedIndex >= scene.mNumTextures) {
    return Result<ImageRgba8, std::string>::makeError(
        "Scene asset baker: embedded texture index is out of range");
  }
  const aiTexture *texture = scene.mTextures[embeddedIndex];
  if (texture == nullptr) {
    return Result<ImageRgba8, std::string>::makeError(
        "Scene asset baker: embedded texture is null");
  }

  if (texture->mHeight == 0u) {
    int32_t width = 0;
    int32_t height = 0;
    int32_t channels = 0;
    stbi_uc *pixels = stbi_load_from_memory(
        reinterpret_cast<const stbi_uc *>(texture->pcData),
        static_cast<int>(texture->mWidth), &width, &height, &channels, 4);
    if (pixels == nullptr) {
      const char *reason = stbi_failure_reason();
      return Result<ImageRgba8, std::string>::makeError(
          "Scene asset baker: failed to decode embedded texture: " +
          std::string(reason ? reason : "unknown error"));
    }

    ImageRgba8 out(memory);
    out.width = width;
    out.height = height;
    out.bytes.assign(reinterpret_cast<std::byte *>(pixels),
                     reinterpret_cast<std::byte *>(pixels) +
                         static_cast<size_t>(width) *
                             static_cast<size_t>(height) * 4u);
    stbi_image_free(pixels);
    return Result<ImageRgba8, std::string>::makeResult(std::move(out));
  }

  ImageRgba8 out(memory);
  out.width = static_cast<int32_t>(texture->mWidth);
  out.height = static_cast<int32_t>(texture->mHeight);
  out.bytes.resize(static_cast<size_t>(out.width) *
                   static_cast<size_t>(out.height) * 4u);
  for (int32_t y = 0; y < out.height; ++y) {
    for (int32_t x = 0; x < out.width; ++x) {
      const aiTexel &src =
          texture
              ->pcData[static_cast<size_t>(y) * static_cast<size_t>(out.width) +
                       static_cast<size_t>(x)];
      const size_t dstOffset =
          (static_cast<size_t>(y) * static_cast<size_t>(out.width) +
           static_cast<size_t>(x)) *
          4u;
      out.bytes[dstOffset + 0u] = static_cast<std::byte>(src.r);
      out.bytes[dstOffset + 1u] = static_cast<std::byte>(src.g);
      out.bytes[dstOffset + 2u] = static_cast<std::byte>(src.b);
      out.bytes[dstOffset + 3u] = static_cast<std::byte>(src.a);
    }
  }
  return Result<ImageRgba8, std::string>::makeResult(std::move(out));
}

[[nodiscard]] Result<ImageRgba8, std::string>
loadSceneTextureRgba(const MaterialTextureSlotData &slot, const aiScene &scene,
                     std::pmr::memory_resource *memory) {
  switch (slot.sourceKind) {
  case MaterialTextureSourceKind::ExternalFile:
    return loadExternalTextureRgba(std::filesystem::path(slot.path), memory);
  case MaterialTextureSourceKind::EmbeddedSceneTexture:
    return loadEmbeddedTextureRgba(scene, slot.embeddedIndex, memory);
  case MaterialTextureSourceKind::None:
    break;
  }
  return Result<ImageRgba8, std::string>::makeError(
      "Scene asset baker: texture slot has no source");
}

[[nodiscard]] Result<KtxTexture2Ptr, std::string>
createPortableTexture(const ImageRgba8 &image, const SlotBakeSpec &spec,
                      std::pmr::memory_resource *memory) {
  if (image.width <= 0 || image.height <= 0 || image.bytes.empty()) {
    return Result<KtxTexture2Ptr, std::string>::makeError(
        "Scene asset baker: mip generation produced no levels");
  }
  const uint32_t mipLevels = computeMipLevelCount(
      static_cast<uint32_t>(image.width), static_cast<uint32_t>(image.height));

  const ktxTextureCreateInfo createInfo{
      .glInternalformat = 0u,
      .vkFormat =
          spec.srgb ? kKtxVkFormatR8G8B8A8Srgb : kKtxVkFormatR8G8B8A8Unorm,
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
  const KTX_error_code createError = ktxTexture2_Create(
      &createInfo, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &texture);
  if (createError != KTX_SUCCESS || texture == nullptr) {
    return Result<KtxTexture2Ptr, std::string>::makeError(
        "Scene asset baker: ktxTexture2_Create failed with code " +
        std::to_string(static_cast<int>(createError)));
  }

  KtxTexture2Ptr texturePtr(texture);
  int32_t currentWidth = image.width;
  int32_t currentHeight = image.height;
  const std::byte *currentBytes = image.bytes.data();
  size_t currentSize = image.bytes.size();
  std::pmr::vector<std::byte> currentOwned(resolveMemoryResource(memory));
  for (uint32_t level = 0u; level < mipLevels; ++level) {
    const KTX_error_code setError = ktxTexture_SetImageFromMemory(
        ktxTexture(texture), level, 0u, 0u,
        reinterpret_cast<const ktx_uint8_t *>(currentBytes), currentSize);
    if (setError != KTX_SUCCESS) {
      return Result<KtxTexture2Ptr, std::string>::makeError(
          "Scene asset baker: failed to populate mip level " +
          std::to_string(level));
    }

    if (level + 1u >= mipLevels) {
      break;
    }

    const int32_t nextWidth = std::max(1, currentWidth >> 1);
    const int32_t nextHeight = std::max(1, currentHeight >> 1);
    auto nextLevelResult =
        generateNextMipLevel(currentBytes, currentWidth, currentHeight,
                             nextWidth, nextHeight, spec, memory);
    if (nextLevelResult.hasError()) {
      return Result<KtxTexture2Ptr, std::string>::makeError(
          nextLevelResult.error());
    }
    currentOwned = std::move(nextLevelResult.value());
    currentBytes = currentOwned.data();
    currentSize = currentOwned.size();
    currentWidth = nextWidth;
    currentHeight = nextHeight;
  }

  ktxBasisParams params{};
  params.structSize = sizeof(params);
  params.threadCount = cappedEncodeThreadCount();
  // KTX Basis normal-map mode stores XY and expects Z reconstruction in the
  // shader. Nuri samples conventional RGB tangent-space normals, so keep
  // normal maps as ordinary linear RGBA data and only use the custom mip
  // renormalization path above.
  params.normalMap = KTX_FALSE;
  if (spec.encodeMode == PortableTextureEncodeMode::Raw) {
    return Result<KtxTexture2Ptr, std::string>::makeResult(
        std::move(texturePtr));
  }
  if (spec.encodeMode == PortableTextureEncodeMode::UASTC) {
    params.uastc = KTX_TRUE;
    params.uastcFlags = KTX_PACK_UASTC_LEVEL_DEFAULT;
    params.uastcRDO = KTX_FALSE;
    params.uastcRDOQualityScalar = 0.0f;
  } else {
    params.uastc = KTX_FALSE;
    params.compressionLevel = KTX_ETC1S_DEFAULT_COMPRESSION_LEVEL;
    params.qualityLevel = 128u;
  }
  const KTX_error_code compressError =
      ktxTexture2_CompressBasisEx(texturePtr.get(), &params);
  if (compressError != KTX_SUCCESS) {
    return Result<KtxTexture2Ptr, std::string>::makeError(
        "Scene asset baker: ktxTexture2_CompressBasisEx failed with code " +
        std::to_string(static_cast<int>(compressError)));
  }

  return Result<KtxTexture2Ptr, std::string>::makeResult(std::move(texturePtr));
}

[[nodiscard]] Result<ktx_uint32_t, std::string>
resolveNativeVkFormat(ScenePortableTextureTarget target, bool srgb) {
  switch (target) {
  case ScenePortableTextureTarget::BC7:
    return Result<ktx_uint32_t, std::string>::makeResult(
        srgb ? kKtxVkFormatBc7Srgb : kKtxVkFormatBc7Unorm);
  case ScenePortableTextureTarget::ETC2:
    return Result<ktx_uint32_t, std::string>::makeResult(
        srgb ? kKtxVkFormatEtc2Rgb8Srgb : kKtxVkFormatEtc2Rgb8Unorm);
  case ScenePortableTextureTarget::RGBA8:
    return Result<ktx_uint32_t, std::string>::makeResult(
        srgb ? kKtxVkFormatR8G8B8A8Srgb : kKtxVkFormatR8G8B8A8Unorm);
  }
  return Result<ktx_uint32_t, std::string>::makeError(
      "Scene asset baker: unsupported native target");
}

[[nodiscard]] Result<ktx_transcode_fmt_e, std::string>
resolveNativeTranscodeFormat(ScenePortableTextureTarget target) {
  switch (target) {
  case ScenePortableTextureTarget::BC7:
    return Result<ktx_transcode_fmt_e, std::string>::makeResult(
        KTX_TTF_BC7_RGBA);
  case ScenePortableTextureTarget::ETC2:
    return Result<ktx_transcode_fmt_e, std::string>::makeResult(
        KTX_TTF_ETC1_RGB);
  case ScenePortableTextureTarget::RGBA8:
    return Result<ktx_transcode_fmt_e, std::string>::makeResult(KTX_TTF_RGBA32);
  }
  return Result<ktx_transcode_fmt_e, std::string>::makeError(
      "Scene asset baker: unsupported native target");
}

[[nodiscard]] Result<KtxTexture2Ptr, std::string>
loadKtxTextureFromFile(const std::filesystem::path &path) {
  ktxTexture *texture = nullptr;
  const std::string pathString = path.string();
  const KTX_error_code loadError = ktxTexture_CreateFromNamedFile(
      pathString.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &texture);
  if (loadError != KTX_SUCCESS || texture == nullptr) {
    return Result<KtxTexture2Ptr, std::string>::makeError(
        "Scene asset baker: failed to read KTX file '" + pathString +
        "' (error " + std::to_string(static_cast<int>(loadError)) + ")");
  }
  if (texture->classId != ktxTexture2_c) {
    ktxTexture_Destroy(texture);
    return Result<KtxTexture2Ptr, std::string>::makeError(
        "Scene asset baker: expected a KTX2 texture '" + pathString + "'");
  }
  return Result<KtxTexture2Ptr, std::string>::makeResult(
      KtxTexture2Ptr(reinterpret_cast<ktxTexture2 *>(texture)));
}

[[nodiscard]] Result<uint64_t, std::string>
writeKtxTextureAtomic(ktxTexture *texture, const std::filesystem::path &path) {
  if (texture == nullptr) {
    return Result<uint64_t, std::string>::makeError(
        "Scene asset baker: KTX texture is null");
  }
  if (path.empty()) {
    return Result<uint64_t, std::string>::makeError(
        "Scene asset baker: destination path is empty");
  }

  std::error_code ec;
  const std::filesystem::path parent = path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent, ec);
    if (ec) {
      return Result<uint64_t, std::string>::makeError(
          "Scene asset baker: failed to create directories for '" +
          path.string() + "'");
    }
  }

  static std::atomic<uint64_t> counter{0};
  const auto threadIdHash =
      std::hash<std::thread::id>{}(std::this_thread::get_id());
  const std::string tempSuffix =
      std::format(".tmp.{:x}.{}", threadIdHash,
                  counter.fetch_add(1, std::memory_order_relaxed));
  const std::filesystem::path tempPath = path.string() + tempSuffix;
  const std::string tempPathString = tempPath.string();
  const KTX_error_code writeError =
      ktxTexture_WriteToNamedFile(texture, tempPathString.c_str());
  if (writeError != KTX_SUCCESS) {
    std::filesystem::remove(tempPath, ec);
    return Result<uint64_t, std::string>::makeError(
        "Scene asset baker: failed to write KTX temp file '" +
        tempPath.string() + "' (error " +
        std::to_string(static_cast<int>(writeError)) + ")");
  }

  std::filesystem::rename(tempPath, path, ec);
  if (ec) {
    const std::error_code renameError = ec;
    ec.clear();
    const bool pathExists = std::filesystem::exists(path, ec);
    if (ec || !pathExists) {
      ec.clear();
      std::filesystem::remove(tempPath, ec);
      return Result<uint64_t, std::string>::makeError(
          "Scene asset baker: failed to rename temp KTX file to '" +
          path.string() + "' (error " + std::to_string(renameError.value()) +
          ")");
    }

    const std::filesystem::path pathBackup =
        path.string() + tempSuffix + ".bak";
    std::filesystem::rename(path, pathBackup, ec);
    if (ec) {
      ec.clear();
      std::filesystem::remove(tempPath, ec);
      return Result<uint64_t, std::string>::makeError(
          "Scene asset baker: failed to move existing KTX file out of the way "
          "for '" +
          path.string() + "'");
    }

    ec.clear();
    std::filesystem::rename(tempPath, path, ec);
    if (ec) {
      const std::error_code secondRenameError = ec;
      ec.clear();
      (void)std::filesystem::rename(pathBackup, path, ec);
      ec.clear();
      std::filesystem::remove(tempPath, ec);
      return Result<uint64_t, std::string>::makeError(
          "Scene asset baker: failed to rename temp KTX file to '" +
          path.string() + "' (error " +
          std::to_string(secondRenameError.value()) + ")");
    }

    ec.clear();
    std::filesystem::remove(pathBackup, ec);
  }

  const uint64_t sizeBytes = std::filesystem::file_size(path, ec);
  if (ec) {
    return Result<uint64_t, std::string>::makeError(
        "Scene asset baker: failed to query KTX size for '" + path.string() +
        "'");
  }
  return Result<uint64_t, std::string>::makeResult(sizeBytes);
}

[[nodiscard]] Result<KtxTexture2Ptr, std::string>
buildNativeTexture(ktxTexture2 &portableTexture,
                   ScenePortableTextureTarget requestedTarget, bool srgb,
                   uint32_t componentCount) {
  ScenePortableTextureTarget target = requestedTarget;
  if (target == ScenePortableTextureTarget::ETC2 && componentCount > 3u) {
    target = ScenePortableTextureTarget::RGBA8;
  }

  ktxTexture2 *copy = nullptr;
  const KTX_error_code copyError =
      ktxTexture2_CreateCopy(&portableTexture, &copy);
  if (copyError != KTX_SUCCESS || copy == nullptr) {
    return Result<KtxTexture2Ptr, std::string>::makeError(
        "Scene asset baker: ktxTexture2_CreateCopy failed with code " +
        std::to_string(static_cast<int>(copyError)));
  }
  KtxTexture2Ptr copyPtr(copy);

  auto transcodeFmt = resolveNativeTranscodeFormat(target);
  if (transcodeFmt.hasError()) {
    return Result<KtxTexture2Ptr, std::string>::makeError(transcodeFmt.error());
  }
  const KTX_error_code transcodeError = ktxTexture2_TranscodeBasis(
      copyPtr.get(), transcodeFmt.value(),
      target == ScenePortableTextureTarget::ETC2 ? KTX_TF_HIGH_QUALITY : 0u);
  if (transcodeError != KTX_SUCCESS) {
    return Result<KtxTexture2Ptr, std::string>::makeError(
        "Scene asset baker: ktxTexture2_TranscodeBasis failed with code " +
        std::to_string(static_cast<int>(transcodeError)));
  }

  auto vkFormatResult = resolveNativeVkFormat(target, srgb);
  if (vkFormatResult.hasError()) {
    return Result<KtxTexture2Ptr, std::string>::makeError(
        vkFormatResult.error());
  }

  const ktxTextureCreateInfo createInfo{
      .glInternalformat = 0u,
      .vkFormat = vkFormatResult.value(),
      .pDfd = nullptr,
      .baseWidth = copyPtr->baseWidth,
      .baseHeight = copyPtr->baseHeight,
      .baseDepth = copyPtr->baseDepth,
      .numDimensions = copyPtr->numDimensions,
      .numLevels = copyPtr->numLevels,
      .numLayers = copyPtr->numLayers,
      .numFaces = copyPtr->numFaces,
      .isArray = copyPtr->isArray,
      .generateMipmaps = KTX_FALSE,
  };

  ktxTexture2 *finalTexture = nullptr;
  const KTX_error_code createError = ktxTexture2_Create(
      &createInfo, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &finalTexture);
  if (createError != KTX_SUCCESS || finalTexture == nullptr) {
    return Result<KtxTexture2Ptr, std::string>::makeError(
        "Scene asset baker: native ktxTexture2_Create failed with code " +
        std::to_string(static_cast<int>(createError)));
  }
  KtxTexture2Ptr finalTexturePtr(finalTexture);

  ktxTexture *srcBase = ktxTexture(copyPtr.get());
  ktxTexture *dstBase = ktxTexture(finalTexturePtr.get());
  const uint8_t *srcData = ktxTexture_GetData(srcBase);
  for (uint32_t level = 0u; level < copyPtr->numLevels; ++level) {
    const ktx_size_t imageSize = ktxTexture_GetImageSize(srcBase, level);
    for (uint32_t layer = 0u; layer < std::max(1u, copyPtr->numLayers);
         ++layer) {
      for (uint32_t face = 0u; face < std::max(1u, copyPtr->numFaces); ++face) {
        ktx_size_t srcOffset = 0u;
        const KTX_error_code offsetError =
            ktxTexture_GetImageOffset(srcBase, level, layer, face, &srcOffset);
        if (offsetError != KTX_SUCCESS) {
          return Result<KtxTexture2Ptr, std::string>::makeError(
              "Scene asset baker: failed to resolve native texture payload "
              "offset (error " +
              std::to_string(static_cast<int>(offsetError)) + ")");
        }
        const KTX_error_code setError = ktxTexture_SetImageFromMemory(
            dstBase, level, layer, face, srcData + srcOffset, imageSize);
        if (setError != KTX_SUCCESS) {
          return Result<KtxTexture2Ptr, std::string>::makeError(
              "Scene asset baker: failed to populate native texture payload");
        }
      }
    }
  }

  return Result<KtxTexture2Ptr, std::string>::makeResult(
      std::move(finalTexturePtr));
}

[[nodiscard]] Format
resolveNativeCacheFormat(ScenePortableTextureTarget requestedTarget, bool srgb,
                         uint32_t componentCount) noexcept {
  if (!srgb) {
    return Format::RGBA8_UNORM;
  }
  ScenePortableTextureTarget target = requestedTarget;
  if (target == ScenePortableTextureTarget::ETC2 && componentCount > 3u) {
    target = ScenePortableTextureTarget::RGBA8;
  }

  switch (target) {
  case ScenePortableTextureTarget::BC7:
    return Format::BC7_RGBA_SRGB;
  case ScenePortableTextureTarget::ETC2:
    return Format::ETC2_RGB8_SRGB;
  case ScenePortableTextureTarget::RGBA8:
    return Format::RGBA8_SRGB;
  }
  return Format::RGBA8_SRGB;
}

[[nodiscard]] bool shouldPersistNativeCache(Format format) noexcept {
  return format != Format::RGBA8_UNORM;
}

[[nodiscard]] NativeTargetBuildList collectNativeTargetsToBuild(
    const std::vector<ScenePortableTextureTarget> &prebuildNativeTargets,
    const std::filesystem::path &portablePath, bool portableFresh, bool srgb,
    uint32_t componentCount, std::pmr::memory_resource *memory) {
  NativeTargetBuildList nativeTargetsToBuild(resolveMemoryResource(memory));
  nativeTargetsToBuild.reserve(prebuildNativeTargets.size());
  for (const ScenePortableTextureTarget target : prebuildNativeTargets) {
    const Format nativeFormat =
        resolveNativeCacheFormat(target, srgb, componentCount);
    if (!shouldPersistNativeCache(nativeFormat)) {
      continue;
    }
    const std::filesystem::path nativePath =
        buildNativeTextureCachePath(portablePath, nativeFormat);
    if (!portableFresh || !isTextureCacheUpToDate(nativePath, portablePath)) {
      nativeTargetsToBuild.push_back(NativeTargetBuild{
          .target = target,
          .path = nativePath,
      });
    }
  }
  return nativeTargetsToBuild;
}

struct SlotBakeResult {
  uint32_t portableTexturesWritten = 0u;
  uint32_t nativeTexturesWritten = 0u;
  uint64_t portableBytesWritten = 0u;
  uint64_t nativeBytesWritten = 0u;

  [[nodiscard]] bool wroteAnyFiles() const noexcept {
    return portableTexturesWritten != 0u || nativeTexturesWritten != 0u;
  }
};

void accumulateSlotBakeResult(ScenePortableBakeStats &stats,
                              const SlotBakeResult &slotResult) {
  stats.wroteAnyFiles = stats.wroteAnyFiles || slotResult.wroteAnyFiles();
  stats.portableTexturesWritten += slotResult.portableTexturesWritten;
  stats.nativeTexturesWritten += slotResult.nativeTexturesWritten;
  stats.portableBytesWritten += slotResult.portableBytesWritten;
  stats.nativeBytesWritten += slotResult.nativeBytesWritten;
}

[[nodiscard]] std::filesystem::path
resolveSceneTextureFreshnessSourcePath(const ScenePortableBakePlan &plan,
                                       const MaterialTextureSlotData &slot) {
  if (slot.sourceKind == MaterialTextureSourceKind::ExternalFile) {
    return std::filesystem::path(slot.path);
  }
  return plan.scenePath;
}

[[nodiscard]] Result<SlotBakeResult, std::string> bakeMaterialTextureSlotToDisk(
    const ScenePortableBakePlan &plan, const aiScene &scene,
    std::string_view canonicalScenePath, const SlotBakeSpec &spec,
    const MaterialTextureSlotData &slot,
    SceneMaterialTextureCacheRecord &cacheRecord, ScratchArena &slotScratch) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  const uint32_t bakeSettingsTag = buildSlotBakeSettingsTag(spec);
  const uint64_t sourceIdentityHash = hashSceneTextureSourceIdentity(
      canonicalScenePath, slot, spec.srgb, bakeSettingsTag);
  cacheRecord.sourceIdentityHash = sourceIdentityHash;
  cacheRecord.srgb = spec.srgb;

  auto portablePathResult =
      buildPortableTextureCachePath(plan.scenePath, sourceIdentityHash);
  if (portablePathResult.hasError()) {
    return Result<SlotBakeResult, std::string>::makeError(
        portablePathResult.error());
  }

  const std::filesystem::path portablePath = portablePathResult.value();
  cacheRecord.portablePath = portablePath.string();
  if (slot.sourceKind == MaterialTextureSourceKind::ExternalFile &&
      std::filesystem::path(slot.path).extension() == ".dds") {
    // DDS scene textures stay authored in this iteration; the portable asset
    // bakery only emits mipmapped KTX2 caches for PNG/KTX sources.
    cacheRecord.portablePath.clear();
    return Result<SlotBakeResult, std::string>::makeResult(SlotBakeResult{});
  }
  const bool portableFresh =
      !portablePath.empty() &&
      isTextureCacheUpToDate(
          portablePath, resolveSceneTextureFreshnessSourcePath(plan, slot));

  ScopedScratch scopedScratch(slotScratch);
  auto *scratchMemory = scopedScratch.resource();
  const uint32_t componentCount = 4u;
  NativeTargetBuildList nativeTargetsToBuild = collectNativeTargetsToBuild(
      plan.prebuildNativeTargets, portablePath, portableFresh, spec.srgb,
      componentCount, scratchMemory);
  if (portableFresh && nativeTargetsToBuild.empty()) {
    return Result<SlotBakeResult, std::string>::makeResult(SlotBakeResult{});
  }

  Result<KtxTexture2Ptr, std::string> portableTextureResult =
      Result<KtxTexture2Ptr, std::string>::makeError("uninitialized");
  SlotBakeResult result{};

  if (portableFresh) {
    NURI_PROFILER_ZONE("SceneBake.LoadPortableKtx", NURI_PROFILER_COLOR_CREATE);
    portableTextureResult = loadKtxTextureFromFile(portablePath);
    NURI_PROFILER_ZONE_END();
  } else {
    NURI_PROFILER_ZONE("SceneBake.EncodePortableKtx",
                       NURI_PROFILER_COLOR_CREATE);
    auto imageResult = loadSceneTextureRgba(slot, scene, scratchMemory);
    if (!imageResult.hasError()) {
      portableTextureResult =
          createPortableTexture(imageResult.value(), spec, scratchMemory);
    } else {
      portableTextureResult =
          Result<KtxTexture2Ptr, std::string>::makeError(imageResult.error());
    }

    if (!portableTextureResult.hasError()) {
      auto portableWriteResult = writeKtxTextureAtomic(
          ktxTexture(portableTextureResult.value().get()), portablePath);
      if (portableWriteResult.hasError()) {
        portableTextureResult = Result<KtxTexture2Ptr, std::string>::makeError(
            portableWriteResult.error());
      } else {
        ++result.portableTexturesWritten;
        result.portableBytesWritten += portableWriteResult.value();
      }
    }
    NURI_PROFILER_ZONE_END();
  }

  if (portableTextureResult.hasError()) {
    return Result<SlotBakeResult, std::string>::makeError(
        portableTextureResult.error());
  }

  if (!nativeTargetsToBuild.empty()) {
    std::string nativeBuildError{};
    NURI_PROFILER_ZONE("SceneBake.BuildNativeCaches",
                       NURI_PROFILER_COLOR_CREATE);
    for (const NativeTargetBuild &nativeTarget : nativeTargetsToBuild) {
      auto nativeTextureResult =
          buildNativeTexture(*portableTextureResult.value(),
                             nativeTarget.target, spec.srgb, componentCount);
      if (nativeTextureResult.hasError()) {
        nativeBuildError = nativeTextureResult.error();
        break;
      }
      auto nativeWriteResult = writeKtxTextureAtomic(
          ktxTexture(nativeTextureResult.value().get()), nativeTarget.path);
      if (nativeWriteResult.hasError()) {
        nativeBuildError = nativeWriteResult.error();
        break;
      }
      ++result.nativeTexturesWritten;
      result.nativeBytesWritten += nativeWriteResult.value();
    }
    NURI_PROFILER_ZONE_END();
    if (!nativeBuildError.empty()) {
      return Result<SlotBakeResult, std::string>::makeError(nativeBuildError);
    }
  }

  return Result<SlotBakeResult, std::string>::makeResult(std::move(result));
}

[[nodiscard]] Result<bool, std::string>
writeSceneMaterialCacheToDisk(const ScenePortableBakePlan &plan,
                              const SceneMaterialCacheData &cacheData) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  const bool materialCacheFresh =
      isTextureCacheUpToDate(plan.materialCachePath, plan.scenePath);
  if (materialCacheFresh) {
    return Result<bool, std::string>::makeResult(false);
  }

  auto cacheKeyResult = buildSceneMaterialCacheKey(plan.scenePath);
  if (cacheKeyResult.hasError()) {
    return Result<bool, std::string>::makeError(cacheKeyResult.error());
  }

  const SceneSourceFingerprint sourceFingerprint =
      querySceneSourceFingerprint(cacheKeyResult.value().normalizedSourcePath);
  MaterialBinarySerializeInput serializeInput{};
  serializeInput.sourcePathHash = cacheKeyResult.value().sourcePathHash;
  serializeInput.sourceSizeBytes = sourceFingerprint.sizeBytes;
  serializeInput.sourceMtimeNs = sourceFingerprint.mtimeNs;
  serializeInput.materials = std::span<const SceneMaterialRecord>(
      cacheData.materials.data(), cacheData.materials.size());

  auto materialBytesResult = materialBinarySerialize(serializeInput);
  if (materialBytesResult.hasError()) {
    return Result<bool, std::string>::makeError(materialBytesResult.error());
  }

  auto materialWriteResult = writeBinaryFileAtomic(plan.materialCachePath,
                                                   materialBytesResult.value());
  if (materialWriteResult.hasError()) {
    return Result<bool, std::string>::makeError(materialWriteResult.error());
  }
  return Result<bool, std::string>::makeResult(true);
}

} // namespace

Result<ScenePortableBakePlan, std::string>
planScenePortableAssetsBake(const ScenePortableAssetsBakeRequest &request) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (request.scenePath.empty()) {
    return Result<ScenePortableBakePlan, std::string>::makeError(
        "Scene asset baker: scene path is empty");
  }
  auto cacheKeyResult = buildSceneMaterialCacheKey(request.scenePath);
  if (cacheKeyResult.hasError()) {
    return Result<ScenePortableBakePlan, std::string>::makeError(
        cacheKeyResult.error());
  }

  ScenePortableBakePlan plan{};
  plan.shouldBake = true;
  plan.scenePath = request.scenePath;
  plan.materialCachePath = cacheKeyResult.value().cachePath;
  plan.prebuildNativeTargets = request.prebuildNativeTargets;
  return Result<ScenePortableBakePlan, std::string>::makeResult(
      std::move(plan));
}

Result<ScenePortableBakeStats, std::string>
bakeScenePortableAssetsToDisk(const ScenePortableBakePlan &plan) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);

  auto materialInfoResult =
      Result<ImportedMaterialSet, std::string>::makeError("uninitialized");
  NURI_PROFILER_ZONE("SceneBake.LoadMaterialInfo", NURI_PROFILER_COLOR_CREATE);
  materialInfoResult =
      MeshImporter::loadMaterialInfoFromFile(plan.scenePath.string());
  NURI_PROFILER_ZONE_END();
  if (materialInfoResult.hasError()) {
    return Result<ScenePortableBakeStats, std::string>::makeError(
        "Scene asset baker: failed to read material info: " +
        materialInfoResult.error());
  }

  Assimp::Importer importer;
  const std::string scenePathStr = plan.scenePath.string();
  const aiScene *scene = nullptr;
  NURI_PROFILER_ZONE("SceneBake.LoadAssimpScene", NURI_PROFILER_COLOR_CREATE);
  scene = importer.ReadFile(scenePathStr,
                            aiProcess_SortByPType | aiProcess_FindInvalidData);
  NURI_PROFILER_ZONE_END();
  if (scene == nullptr) {
    return Result<ScenePortableBakeStats, std::string>::makeError(
        std::string("Scene asset baker: Assimp error: ") +
        importer.GetErrorString());
  }

  const std::string canonicalScenePath =
      canonicalizeResourcePath(plan.scenePath.string());
  ScratchArena slotScratch{};
  SceneMaterialCacheData cacheData{};
  cacheData.materials.reserve(materialInfoResult.value().materials.size());
  ScenePortableBakeStats stats{};
  std::string materialLoopError{};

  NURI_PROFILER_ZONE("SceneBake.MaterialLoop", NURI_PROFILER_COLOR_CREATE);
  for (uint32_t materialIndex = 0;
       materialIndex < materialInfoResult.value().materials.size();
       ++materialIndex) {
    const MaterialData &material =
        materialInfoResult.value().materials[materialIndex];
    SceneMaterialRecord record{};
    record.sourceMaterialIndex = materialIndex;
    record.sourceMaterial = material;

    for (size_t slotIndex = 0; slotIndex < kSlotBakeSpecs.size(); ++slotIndex) {
      const SlotBakeSpec &spec = kSlotBakeSpecs[slotIndex];
      const MaterialTextureSlotData &slot = material.*(spec.slot);
      if (slot.sourceKind == MaterialTextureSourceKind::None) {
        continue;
      }

      auto slotBakeResult = bakeMaterialTextureSlotToDisk(
          plan, *scene, canonicalScenePath, spec, slot,
          record.textureCache[slotIndex], slotScratch);
      if (slotBakeResult.hasError()) {
        materialLoopError = slotBakeResult.error();
        break;
      }
      accumulateSlotBakeResult(stats, slotBakeResult.value());
    }
    if (!materialLoopError.empty()) {
      break;
    }

    cacheData.materials.push_back(std::move(record));
  }
  NURI_PROFILER_ZONE_END();
  if (!materialLoopError.empty()) {
    return Result<ScenePortableBakeStats, std::string>::makeError(
        materialLoopError);
  }

  auto materialCacheWriteResult =
      writeSceneMaterialCacheToDisk(plan, cacheData);
  if (materialCacheWriteResult.hasError()) {
    return Result<ScenePortableBakeStats, std::string>::makeError(
        materialCacheWriteResult.error());
  }
  if (!stats.wroteAnyFiles && !materialCacheWriteResult.value()) {
    return Result<ScenePortableBakeStats, std::string>::makeResult(
        std::move(stats));
  }
  stats.wroteAnyFiles = true;
  stats.wroteMaterialCache = materialCacheWriteResult.value();

  return Result<ScenePortableBakeStats, std::string>::makeResult(
      std::move(stats));
}

} // namespace nuri::bakery::detail
