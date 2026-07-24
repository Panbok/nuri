#include "nuri/resources/storage/texture/texture_artifact_builder.h"
#include "nuri/core/log.h"
#include "nuri/core/pmr_scratch.h"
#include "nuri/core/profiling.h"
#include "nuri/pch.h"
#include "nuri/resources/storage/cache_utils.h"
#include "nuri/resources/storage/texture/texture_ktx.h"
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <atomic>
#include <stb_image.h>
#include <thread>
namespace nuri {

using ArtifactResult = Result<TextureArtifactBuildResult, std::string>;

namespace {
using detail::FilePtr;
using detail::KtxTexture2Ptr;
using detail::KtxTexturePtr;
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
  std::atomic<uint64_t> normalVarianceArtifactBuilds{0u};
  std::atomic<uint64_t> normalVarianceCleanTexels{0u};
  std::atomic<uint64_t> normalVarianceToksvigFallbackTexels{0u};
  std::atomic<uint64_t> normalVarianceContractRejections{0u};
  std::atomic<uint64_t> normalVarianceArtifactBytesWritten{0u};
  std::atomic<uint64_t> normalVarianceArtifactBuildTimeNs{0u};
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
[[nodiscard]] uint32_t cappedEncodeThreadCount() noexcept {
  const uint32_t hardware = std::max(1u, std::thread::hardware_concurrency());
  return std::min(hardware, 8u);
}
[[nodiscard]] Result<KtxTexturePtr, std::string>
loadKtxTextureFile(const std::filesystem::path &path) {
  return detail::loadKtxTexture(path, "Texture artifact builder");
}
struct ImageRgba8 {
  int32_t width = 0;
  int32_t height = 0;
  uint32_t sourceComponentCount = 4u;
  std::pmr::vector<std::byte> bytes;
  explicit ImageRgba8(std::pmr::memory_resource *memory)
      : bytes(memory != nullptr ? memory : std::pmr::get_default_resource()) {}
};
[[nodiscard]] Result<ImageRgba8, std::string>
adoptDecodedRgba(stbi_uc *pixels, int32_t width, int32_t height,
                 int32_t channels, std::pmr::memory_resource *memory,
                 std::string error) {
  if (!pixels) {
    return Result<ImageRgba8, std::string>::makeError(std::move(error));
  }
  ImageRgba8 image(memory);
  image.width = width;
  image.height = height;
  image.sourceComponentCount =
      static_cast<uint32_t>(std::clamp(channels, 1, 4));
  const size_t size =
      static_cast<size_t>(width) * static_cast<size_t>(height) * 4u;
  image.bytes.assign(reinterpret_cast<std::byte *>(pixels),
                     reinterpret_cast<std::byte *>(pixels) + size);
  stbi_image_free(pixels);
  return Result<ImageRgba8, std::string>::makeResult(std::move(image));
}
[[nodiscard]] Result<ImageRgba8, std::string>
loadKtxRgba(const std::filesystem::path &path,
            std::pmr::memory_resource *memory) {
  auto textureResult = loadKtxTextureFile(path);
  if (textureResult.hasError()) {
    return Result<ImageRgba8, std::string>::makeError(textureResult.error());
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
  const char *reason = stbi_failure_reason();
  return adoptDecodedRgba(pixels, width, height, channels, memory,
                          "Texture artifact builder: failed to load source '" +
                              path.string() +
                              "': " + (reason ? reason : "unknown error"));
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
    return adoptDecodedRgba(
        pixels, width, height, channels, memory,
        "Texture artifact builder: failed to decode embedded texture");
  }
  ImageRgba8 out(memory);
  out.width = static_cast<int32_t>(texture.mWidth);
  out.height = static_cast<int32_t>(texture.mHeight);
  out.sourceComponentCount = 4u;
  out.bytes.resize(static_cast<size_t>(out.width) *
                   static_cast<size_t>(out.height) * 4u);
  for (size_t i = 0;
       i < static_cast<size_t>(out.width) * static_cast<size_t>(out.height);
       ++i) {
    const aiTexel &src = texture.pcData[i];
    out.bytes[i * 4u] = static_cast<std::byte>(src.r);
    out.bytes[i * 4u + 1u] = static_cast<std::byte>(src.g);
    out.bytes[i * 4u + 2u] = static_cast<std::byte>(src.b);
    out.bytes[i * 4u + 3u] = static_cast<std::byte>(src.a);
  }
  return Result<ImageRgba8, std::string>::makeResult(std::move(out));
}
[[nodiscard]] Result<ImageRgba8, std::string>
loadEmbeddedRgba(std::span<const EmbeddedSceneTextureData> textures,
                 uint32_t embeddedIndex, std::pmr::memory_resource *memory) {
  if (embeddedIndex >= textures.size()) {
    return Result<ImageRgba8, std::string>::makeError(
        "Texture artifact builder: adapted embedded texture index is invalid");
  }
  const EmbeddedSceneTextureData &texture = textures[embeddedIndex];
  if (texture.compressed) {
    if (texture.bytes.empty() ||
        texture.bytes.size() >
            static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
      return Result<ImageRgba8, std::string>::makeError(
          "Texture artifact builder: adapted embedded texture is empty or "
          "too large");
    }
    int32_t width = 0;
    int32_t height = 0;
    int32_t channels = 0;
    stbi_uc *pixels = stbi_load_from_memory(
        reinterpret_cast<const stbi_uc *>(texture.bytes.data()),
        static_cast<int32_t>(texture.bytes.size()), &width, &height, &channels,
        4);
    return adoptDecodedRgba(
        pixels, width, height, channels, memory,
        "Texture artifact builder: failed to decode adapted embedded texture");
  }
  const size_t expectedBytes = static_cast<size_t>(texture.width) *
                               static_cast<size_t>(texture.height) * 4u;
  if (texture.width == 0u || texture.height == 0u ||
      texture.bytes.size() != expectedBytes) {
    return Result<ImageRgba8, std::string>::makeError(
        "Texture artifact builder: adapted embedded RGBA payload is invalid");
  }
  ImageRgba8 out(memory);
  out.width = static_cast<int32_t>(texture.width);
  out.height = static_cast<int32_t>(texture.height);
  out.sourceComponentCount = 4u;
  out.bytes.assign(texture.bytes.begin(), texture.bytes.end());
  return Result<ImageRgba8, std::string>::makeResult(std::move(out));
}
[[nodiscard]] Result<Format, std::string>
resolveNativeKtxFormat(const ktxTexture &texture) {
  if (texture.classId != ktxTexture2_c) {
    return Result<Format, std::string>::makeError(
        "Texture artifact builder: authored native KTX1 is not a target "
        "artifact");
  }
  return detail::resolveKtxFormat(texture, "Texture artifact builder");
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
[[nodiscard]] bool
textureFormatPreservesIndependentAlpha(Format format) noexcept {
  return format == Format::RGBA8_UNORM || format == Format::BC7_RGBA_UNORM;
}

[[nodiscard]] Result<bool, std::string>
validateContentContract(const TextureArtifactBuildOptions &options,
                        Format targetFormat) {
  if (options.contentContract !=
      TextureContentContract::NormalRgbCleanVarianceA) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (options.loadOptions.mipSemantic != TextureMipSemantic::NormalMap) {
    return Result<bool, std::string>::makeError(
        "NormalRgbCleanVarianceA requires NormalMap mip semantics");
  }
  if (options.loadOptions.srgb) {
    return Result<bool, std::string>::makeError(
        "NormalRgbCleanVarianceA requires linear texture data");
  }
  if (!textureFormatPreservesIndependentAlpha(targetFormat)) {
    return Result<bool, std::string>::makeError(
        "NormalRgbCleanVarianceA requires an alpha-preserving target format");
  }
  return Result<bool, std::string>::makeResult(true);
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
  return Result<KtxTexture2Ptr, std::string>::makeResult(KtxTexture2Ptr(copy));
}
[[nodiscard]] Result<KtxTexture2Ptr, std::string>
transcodeBasisToNative(ktxTexture2 &source, Format targetFormat) {
  ktxTexture2 *copy = nullptr;
  if (ktxTexture2_CreateCopy(&source, &copy) != KTX_SUCCESS ||
      copy == nullptr) {
    return Result<KtxTexture2Ptr, std::string>::makeError(
        "Texture artifact builder: failed to copy Basis KTX2");
  }
  KtxTexture2Ptr copyPtr(copy);
  auto transcodeFormat = resolveTranscodeFormat(targetFormat);
  if (transcodeFormat.hasError()) {
    return Result<KtxTexture2Ptr, std::string>::makeError(
        transcodeFormat.error());
  }
  const khr_df_transfer_e transferFunction = textureFormatIsSrgb(targetFormat)
                                                 ? KHR_DF_TRANSFER_SRGB
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
                        NormalVarianceBuildStats *varianceStats) {
  if (image.width <= 0 || image.height <= 0 || image.bytes.empty()) {
    return Result<KtxTexture2Ptr, std::string>::makeError(
        "Texture artifact builder: decoded image is empty");
  }
  const uint32_t mipLevels =
      options.loadOptions.generateMipmaps
          ? textureMipLevelCount(static_cast<uint32_t>(image.width),
                                 static_cast<uint32_t>(image.height))
          : 1u;
  const ktxTextureCreateInfo createInfo{
      .glInternalformat = 0u,
      .vkFormat = options.loadOptions.srgb ? detail::kVkRgba8Srgb
                                           : detail::kVkRgba8Unorm,
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
  std::vector<std::byte> owned;
  std::vector<std::byte> cleanChain;
  size_t cleanOffset = 0u;
  if (options.contentContract ==
      TextureContentContract::NormalRgbCleanVarianceA) {
    auto chain = generateSemanticRgba8MipChain(
        current, static_cast<uint32_t>(width), static_cast<uint32_t>(height),
        mipLevels, options.loadOptions, options.contentContract, varianceStats);
    if (chain.hasError()) {
      return Result<KtxTexture2Ptr, std::string>::makeError(chain.error());
    }
    cleanChain = std::move(chain.value());
    current = std::span<const std::byte>(cleanChain.data(),
                                         static_cast<size_t>(width) *
                                             static_cast<size_t>(height) * 4u);
  }
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
    if (!cleanChain.empty()) {
      cleanOffset += current.size();
      const uint32_t nextWidth = std::max(1, width >> 1);
      const uint32_t nextHeight = std::max(1, height >> 1);
      current = std::span<const std::byte>(cleanChain.data() + cleanOffset,
                                           size_t{nextWidth} * nextHeight * 4u);
    } else {
      auto next =
          generateRgba8Mip(current, static_cast<uint32_t>(width),
                           static_cast<uint32_t>(height), options.loadOptions);
      if (next.hasError()) {
        return Result<KtxTexture2Ptr, std::string>::makeError(next.error());
      }
      owned = std::move(next.value());
      current = std::span<const std::byte>(owned.data(), owned.size());
    }
    width = std::max(1, width >> 1);
    height = std::max(1, height >> 1);
  }
  return Result<KtxTexture2Ptr, std::string>::makeResult(std::move(texturePtr));
}
[[nodiscard]] Result<KtxTexture2Ptr, std::string>
buildNativeFromImage(const ImageRgba8 &image, Format targetFormat,
                     const TextureArtifactBuildOptions &options,
                     NormalVarianceBuildStats *varianceStats) {
  auto rgbaTexture = createRgbaSourceTexture(image, options, varianceStats);
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
writeKtxTextureAtomic(const std::filesystem::path &path, ktxTexture2 &texture) {
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
  const std::filesystem::path tempPath = temporarySiblingPath(path);
  FilePtr output = detail::openFile(tempPath, true);
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
  if (!replaceFileAtomic(tempPath, path)) {
    std::filesystem::remove(tempPath, ec);
    return Result<uint64_t, std::string>::makeError(
        "Texture artifact builder: failed to commit KTX2 artifact '" +
        path.string() + "'");
  }
  return Result<uint64_t, std::string>::makeResult(artifactSizeBytes);
}
[[nodiscard]] uint64_t tightPayloadSize(const ktxTexture &texture) noexcept {
  uint64_t size = 0u;
  ktxTexture *mutableTexture = const_cast<ktxTexture *>(&texture);
  for (uint32_t level = 0u; level < std::max(1u, texture.numLevels); ++level) {
    size +=
        static_cast<uint64_t>(ktxTexture_GetImageSize(mutableTexture, level)) *
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
  std::span<const EmbeddedSceneTextureData> embeddedTextures{};
  Assimp::Importer importer{};
  const aiScene *scene = nullptr;
  ScratchArena scratch{};
  [[nodiscard]] Result<const aiScene *, std::string> ensureScene() {
    if (scene != nullptr) {
      return Result<const aiScene *, std::string>::makeResult(scene);
    }
    const std::string path = sceneSourcePath.string();
    scene = importer.ReadFile(path, aiProcess_SortByPType |
                                        aiProcess_FindInvalidData);
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
SceneTextureArtifactBuilder &SceneTextureArtifactBuilder::operator=(
    SceneTextureArtifactBuilder &&) noexcept = default;

Result<SceneTextureArtifactBuilder, std::string>
SceneTextureArtifactBuilder::create(
    const std::filesystem::path &sceneSourcePath,
    std::span<const EmbeddedSceneTextureData> embeddedTextures) {
  const std::filesystem::path normalized = normalizeSourcePath(sceneSourcePath);
  if (normalized.empty()) {
    return Result<SceneTextureArtifactBuilder, std::string>::makeError(
        "Texture artifact builder: scene source path is empty");
  }
  auto impl = std::make_unique<Impl>();
  impl->sceneSourcePath = normalized;
  impl->embeddedTextures = embeddedTextures;
  return Result<SceneTextureArtifactBuilder, std::string>::makeResult(
      SceneTextureArtifactBuilder(std::move(impl)));
}

ArtifactResult SceneTextureArtifactBuilder::ensure(
    const MaterialTextureSlotData &source, uint64_t sourceIdentityHash,
    Format targetFormat, const TextureArtifactBuildOptions &options,
    bool forceRebuild) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!impl_ || !sourceIdentityHash ||
      !shouldPersistNativeTextureArtifact(targetFormat) ||
      source.sourceKind == MaterialTextureSourceKind::None ||
      (source.sourceKind == MaterialTextureSourceKind::ExternalFile &&
       source.path.empty())) {
    return ArtifactResult::makeError(
        "Texture artifact builder: invalid artifact request");
  }
  auto contractValidation = validateContentContract(options, targetFormat);
  if (contractValidation.hasError()) {
    gTelemetry.normalVarianceContractRejections.fetch_add(
        1u, std::memory_order_relaxed);
    return ArtifactResult::makeError(contractValidation.error());
  }
  const uint32_t contentEncodingVersion =
      options.contentContract == TextureContentContract::NormalRgbCleanVarianceA
          ? kNormalVarianceEncodingVersion
          : 0u;
  const std::filesystem::path freshnessPath =
      source.sourceKind == MaterialTextureSourceKind::ExternalFile
          ? std::filesystem::path(source.path)
          : impl_->sceneSourcePath;
  if (hasExtension(freshnessPath, ".dds")) {
    return ArtifactResult::makeError(
        "Texture artifact builder: DDS textures are source-native and are not "
        "converted into KTX artifacts");
  }
  auto artifactPath = buildNativeTextureArtifactPath(
      impl_->sceneSourcePath, sourceIdentityHash, targetFormat);
  if (artifactPath.hasError()) {
    return ArtifactResult::makeError(artifactPath.error());
  }
  NativeTextureCacheProbe probe{};
  if (!forceRebuild) {
    probe = probeNativeTextureCache(
        artifactPath.value(), freshnessPath, sourceIdentityHash, targetFormat,
        options.contentContract, contentEncodingVersion);
    recordProbeStatus(probe.status);
    if (probe.status == NativeTextureCacheProbeStatus::Hit) {
      gTelemetry.nativeHits.fetch_add(1u, std::memory_order_relaxed);
      gTelemetry.nativeArtifactBytesRead.fetch_add(
          probe.metadata.artifactSizeBytes, std::memory_order_relaxed);
      return ArtifactResult::makeResult(TextureArtifactBuildResult{
          .artifactPath = artifactPath.value(),
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
    return ArtifactResult::makeError(initialFingerprint.error());
  }
  const auto buildStart = std::chrono::steady_clock::now();
  ScopedScratch scopedScratch(impl_->scratch);
  std::pmr::memory_resource *memory = scopedScratch.resource();
  KtxTexture2Ptr native;
  if (options.contentContract !=
          TextureContentContract::NormalRgbCleanVarianceA &&
      source.sourceKind == MaterialTextureSourceKind::ExternalFile &&
      isKtxPath(freshnessPath)) {
    auto direct = tryBuildDirectlyFromAuthoredKtx(freshnessPath, targetFormat);
    if (direct.hasError()) {
      return ArtifactResult::makeError(direct.error());
    }
    if (direct.value().has_value()) {
      native = std::move(*direct.value());
    }
  }
  if (!native) {
    auto image = [&]() -> Result<ImageRgba8, std::string> {
      if (source.sourceKind == MaterialTextureSourceKind::ExternalFile) {
        return loadExternalRgba(freshnessPath, memory);
      }
      if (!impl_->embeddedTextures.empty()) {
        return loadEmbeddedRgba(impl_->embeddedTextures, source.embeddedIndex,
                                memory);
      }
      auto scene = impl_->ensureScene();
      return scene.hasError()
                 ? Result<ImageRgba8, std::string>::makeError(scene.error())
                 : loadEmbeddedRgba(*scene.value(), source.embeddedIndex,
                                    memory);
    }();
    if (image.hasError()) {
      return ArtifactResult::makeError(image.error());
    }
    NormalVarianceBuildStats varianceStats{};
    auto built = buildNativeFromImage(image.value(), targetFormat, options,
                                      &varianceStats);
    if (built.hasError()) {
      return ArtifactResult::makeError(built.error());
    }
    native = std::move(built.value());
    if (options.contentContract ==
        TextureContentContract::NormalRgbCleanVarianceA) {
      gTelemetry.normalVarianceArtifactBuilds.fetch_add(
          1u, std::memory_order_relaxed);
      gTelemetry.normalVarianceCleanTexels.fetch_add(varianceStats.cleanTexels,
                                                     std::memory_order_relaxed);
      gTelemetry.normalVarianceToksvigFallbackTexels.fetch_add(
          varianceStats.toksvigFallbackTexels, std::memory_order_relaxed);
    }
  }
  auto finalFingerprint = queryTextureSourceFingerprint(freshnessPath);
  if (finalFingerprint.hasError() ||
      finalFingerprint.value() != initialFingerprint.value()) {
    gTelemetry.nativeWriteFailures.fetch_add(1u, std::memory_order_relaxed);
    return ArtifactResult::makeError(
        "Texture artifact builder: source changed during artifact build");
  }
  auto artifactWrite = writeKtxTextureAtomic(artifactPath.value(), *native);
  if (artifactWrite.hasError()) {
    gTelemetry.nativeWriteFailures.fetch_add(1u, std::memory_order_relaxed);
    return ArtifactResult::makeError(artifactWrite.error());
  }
  const ktxTexture &texture = *ktxTexture(native.get());
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
      .contentContract = options.contentContract,
      .contentEncodingVersion = contentEncodingVersion,
  };
  auto metadataWrite =
      writeNativeTextureCacheMetadataAtomic(artifactPath.value(), metadata);
  if (metadataWrite.hasError()) {
    gTelemetry.nativeWriteFailures.fetch_add(1u, std::memory_order_relaxed);
    return ArtifactResult::makeError(metadataWrite.error());
  }
  gTelemetry.authoredSourceBytesRead.fetch_add(
      initialFingerprint.value().sizeBytes, std::memory_order_relaxed);
  const uint64_t buildTimeNs = elapsedNanoseconds(buildStart);
  gTelemetry.artifactBuildTimeNs.fetch_add(buildTimeNs,
                                           std::memory_order_relaxed);
  if (options.contentContract ==
      TextureContentContract::NormalRgbCleanVarianceA) {
    gTelemetry.normalVarianceArtifactBytesWritten.fetch_add(
        artifactWrite.value(), std::memory_order_relaxed);
    gTelemetry.normalVarianceArtifactBuildTimeNs.fetch_add(
        buildTimeNs, std::memory_order_relaxed);
  }
  gTelemetry.artifactBuilds.fetch_add(1u, std::memory_order_relaxed);
  gTelemetry.nativeWrites.fetch_add(1u, std::memory_order_relaxed);
  return ArtifactResult::makeResult(TextureArtifactBuildResult{
      .artifactPath = artifactPath.value(),
      .built = true,
      .artifactSizeBytes = artifactWrite.value(),
  });
}

ArtifactResult
ensureTextureArtifactFromFile(const std::filesystem::path &sourcePath,
                              uint64_t sourceIdentityHash, Format targetFormat,
                              const TextureArtifactBuildOptions &options,
                              bool forceRebuild) {
  auto builder = SceneTextureArtifactBuilder::create(sourcePath);
  if (builder.hasError()) {
    return ArtifactResult::makeError(builder.error());
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
  mix(static_cast<uint32_t>(options.contentContract));
  if (options.contentContract ==
      TextureContentContract::NormalRgbCleanVarianceA) {
    mix(kNormalVarianceEncodingVersion);
    mix(std::bit_cast<uint32_t>(kEncodedSlopeVarianceMax));
    mix(std::bit_cast<uint32_t>(kCleanNormalZFloor));
    mix(std::bit_cast<uint32_t>(kCleanValidWeightEpsilon));
  }
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
      .normalVarianceArtifactBuilds =
          gTelemetry.normalVarianceArtifactBuilds.load(
              std::memory_order_relaxed),
      .normalVarianceCleanTexels =
          gTelemetry.normalVarianceCleanTexels.load(std::memory_order_relaxed),
      .normalVarianceToksvigFallbackTexels =
          gTelemetry.normalVarianceToksvigFallbackTexels.load(
              std::memory_order_relaxed),
      .normalVarianceContractRejections =
          gTelemetry.normalVarianceContractRejections.load(
              std::memory_order_relaxed),
      .normalVarianceArtifactBytesWritten =
          gTelemetry.normalVarianceArtifactBytesWritten.load(
              std::memory_order_relaxed),
      .normalVarianceArtifactBuildTimeNs =
          gTelemetry.normalVarianceArtifactBuildTimeNs.load(
              std::memory_order_relaxed),
  };
}

} // namespace nuri
