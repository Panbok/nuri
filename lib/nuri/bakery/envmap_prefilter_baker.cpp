#include "nuri/pch.h"

#include "nuri/bakery/envmap_prefilter_baker.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/gpu_descriptors.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/gpu_render_types.h"
#include "nuri/gfx/shader.h"
#include "nuri/resources/cpu/bitmap.h"
#include "nuri/resources/gpu/texture.h"

#include <ktx.h>
#include <stb_image.h>
#include <vulkan/vulkan_core.h>

namespace nuri::bakery::detail {
namespace {

constexpr uint32_t kIrradianceFaceSize = 64u;
constexpr uint32_t kLambertSamples = 2048u;
constexpr uint32_t kGgxBaseSamples = 1024u;
constexpr uint32_t kCharlieBaseSamples = 1024u;
constexpr uint32_t kLocalSize = 8u;
constexpr ktx_uint32_t kGlRgba32f = 0x8814u;
constexpr std::string_view kPrefilterShaderFileName = "envmap_prefilter.comp";
constexpr std::string_view kSourceCubemapDebugName = "env_prefilter_source";
constexpr std::string_view kComputePipelineDebugName =
    "envmap_prefilter_compute";
constexpr std::string_view kComputeOutputBufferDebugName =
    "env_prefilter_output";

struct PrefilterPushConstants {
  uint32_t faceSize = 0;
  uint32_t tileOffsetX = 0;
  uint32_t tileOffsetY = 0;
  uint32_t tileWidth = 0;
  uint32_t tileHeight = 0;
  float roughness = 0.0f;
  uint32_t sampleCount = 0;
  uint32_t envMapTexId = 0;
  uint32_t envMapSamplerId = 0;
  uint32_t envMapWidth = 0;
  uint32_t envMapHeight = 0;
  uint32_t distribution = 0;
  uint32_t faceIndex = 0;
  uint32_t _padding0 = 0;
  uint64_t outputBufferAddress = 0;
};
static_assert(sizeof(PrefilterPushConstants) <= 128,
              "PrefilterPushConstants exceeds Vulkan minimum guarantee");

uint32_t computeMipLevelCount(uint32_t faceSize) {
  if (faceSize == 0u) {
    return 0u;
  }
  uint32_t levels = 1u;
  while (faceSize > 1u) {
    faceSize >>= 1u;
    ++levels;
  }
  return levels;
}

EnvOutputPaths makeOutputPaths(const RuntimeConfig &config,
                               const std::filesystem::path &hdrPath) {
  const std::string stem = hdrPath.stem().string();
  return EnvOutputPaths{
      .irradiance = config.roots.textures / (stem + "_irradiance.ktx2"),
      .ggx = config.roots.textures / (stem + "_prefilter_ggx.ktx2"),
      .charlie = config.roots.textures / (stem + "_prefilter_charlie.ktx2"),
  };
}

Result<EnvPreparedCubemapCpuData, std::string>
prepareSourceCubemapCpu(const std::filesystem::path &hdrPath) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  const std::string hdrPathStr = hdrPath.string();
  int32_t width = 0;
  int32_t height = 0;
  int32_t channels = 0;
  float *pixels = stbi_loadf(hdrPathStr.c_str(), &width, &height, &channels, 4);
  if (!pixels) {
    const char *reason = stbi_failure_reason();
    return Result<EnvPreparedCubemapCpuData, std::string>::makeError(
        "Env prefilter baker: failed to load HDR '" + hdrPathStr +
        "': " + (reason ? std::string(reason) : std::string("unknown error")));
  }

  const Bitmap equirectangular(width, height, 4, BitmapFormat::F32, pixels);
  stbi_image_free(pixels);

  const Bitmap cubemapFaces =
      equirectangular.convertEquirectangularMapToCubeMapFaces();
  if (cubemapFaces.empty()) {
    return Result<EnvPreparedCubemapCpuData, std::string>::makeError(
        "Env prefilter baker: failed to convert HDR to cubemap faces: '" +
        hdrPathStr + "'");
  }

  const std::span<const uint8_t> srcBytes = cubemapFaces.data();
  if (srcBytes.empty()) {
    return Result<EnvPreparedCubemapCpuData, std::string>::makeError(
        "Env prefilter baker: prepared cubemap face bytes are empty");
  }
  std::vector<std::byte> floatBytes(srcBytes.size());
  std::memcpy(floatBytes.data(), srcBytes.data(), srcBytes.size());

  return Result<EnvPreparedCubemapCpuData, std::string>::makeResult(
      EnvPreparedCubemapCpuData{
          .faceSize = static_cast<uint32_t>(cubemapFaces.width()),
          .floatBytes = std::move(floatBytes),
      });
}

Result<std::unique_ptr<Texture>, std::string>
createSourceCubemapTextureFromPrepared(
    GPUDevice &gpu, const EnvPreparedCubemapCpuData &prepared,
    std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (prepared.faceSize == 0u || prepared.floatBytes.empty()) {
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        "Env prefilter baker: prepared cubemap data is invalid");
  }

  constexpr uint32_t kChannelCount = 4u;
  constexpr uint32_t kBytesPerChannel = sizeof(float);
  constexpr uint32_t kBytesPerPixel = kChannelCount * kBytesPerChannel;
  const size_t baseFaceBytes = static_cast<size_t>(prepared.faceSize) *
                               static_cast<size_t>(prepared.faceSize) *
                               static_cast<size_t>(kBytesPerPixel);
  const size_t expectedBaseBytes = baseFaceBytes * 6u;
  if (prepared.floatBytes.size() != expectedBaseBytes) {
    return Result<std::unique_ptr<Texture>, std::string>::makeError(
        "Env prefilter baker: source cubemap base data size mismatch");
  }

  const uint32_t mipLevels = computeMipLevelCount(prepared.faceSize);
  std::vector<std::byte> mipChainBytes;
  {
    NURI_PROFILER_ZONE("EnvPrefilter::buildSourceMipChain",
                       NURI_PROFILER_COLOR_CREATE);
    std::vector<size_t> mipBaseOffsets(mipLevels, 0u);
    std::vector<size_t> mipFaceBytesByLevel(mipLevels, 0u);
    std::vector<uint32_t> mipFaceSizes(mipLevels, 1u);
    size_t totalBytes = 0u;
    for (uint32_t mip = 0u; mip < mipLevels; ++mip) {
      const uint32_t mipFaceSize = std::max(1u, prepared.faceSize >> mip);
      const size_t mipFaceBytes = static_cast<size_t>(mipFaceSize) *
                                  static_cast<size_t>(mipFaceSize) *
                                  static_cast<size_t>(kBytesPerPixel);
      mipFaceSizes[mip] = mipFaceSize;
      mipFaceBytesByLevel[mip] = mipFaceBytes;
      mipBaseOffsets[mip] = totalBytes;
      totalBytes += mipFaceBytes * 6u;
    }
    mipChainBytes.resize(totalBytes);

    const size_t maxFaceElements = static_cast<size_t>(prepared.faceSize) *
                                   static_cast<size_t>(prepared.faceSize) * 4u;
    std::vector<float> prevLevel(maxFaceElements);
    std::vector<float> nextLevel(maxFaceElements);
    for (uint32_t face = 0u; face < 6u; ++face) {
      uint32_t prevWidth = prepared.faceSize;
      uint32_t prevHeight = prepared.faceSize;
      {
        const std::byte *srcFace = prepared.floatBytes.data() +
                                   static_cast<size_t>(face) * baseFaceBytes;
        std::memcpy(prevLevel.data(), srcFace, baseFaceBytes);
      }

      for (uint32_t mip = 0u; mip < mipLevels; ++mip) {
        const uint32_t mipFaceSize = mipFaceSizes[mip];
        const size_t mipFaceBytes = mipFaceBytesByLevel[mip];
        const size_t dstOffset =
            mipBaseOffsets[mip] + static_cast<size_t>(face) * mipFaceBytes;
        std::memcpy(mipChainBytes.data() + dstOffset, prevLevel.data(),
                    mipFaceBytes);

        if (mip + 1u >= mipLevels) {
          continue;
        }

        const uint32_t nextWidth = std::max(1u, prevWidth >> 1u);
        const uint32_t nextHeight = std::max(1u, prevHeight >> 1u);
        for (uint32_t y = 0u; y < nextHeight; ++y) {
          const uint32_t srcY0 = std::min(y * 2u, prevHeight - 1u);
          const uint32_t srcY1 = std::min(srcY0 + 1u, prevHeight - 1u);
          for (uint32_t x = 0u; x < nextWidth; ++x) {
            const uint32_t srcX0 = std::min(x * 2u, prevWidth - 1u);
            const uint32_t srcX1 = std::min(srcX0 + 1u, prevWidth - 1u);
            const size_t dstBase =
                (static_cast<size_t>(y) * static_cast<size_t>(nextWidth) +
                 static_cast<size_t>(x)) *
                4u;
            const size_t s00 =
                (static_cast<size_t>(srcY0) * static_cast<size_t>(prevWidth) +
                 static_cast<size_t>(srcX0)) *
                4u;
            const size_t s10 =
                (static_cast<size_t>(srcY0) * static_cast<size_t>(prevWidth) +
                 static_cast<size_t>(srcX1)) *
                4u;
            const size_t s01 =
                (static_cast<size_t>(srcY1) * static_cast<size_t>(prevWidth) +
                 static_cast<size_t>(srcX0)) *
                4u;
            const size_t s11 =
                (static_cast<size_t>(srcY1) * static_cast<size_t>(prevWidth) +
                 static_cast<size_t>(srcX1)) *
                4u;
            for (uint32_t c = 0u; c < 4u; ++c) {
              nextLevel[dstBase + c] =
                  (prevLevel[s00 + c] + prevLevel[s10 + c] +
                   prevLevel[s01 + c] + prevLevel[s11 + c]) *
                  0.25f;
            }
          }
        }
        prevLevel.swap(nextLevel);
        prevWidth = nextWidth;
        prevHeight = nextHeight;
      }
    }
    NURI_PROFILER_ZONE_END();
  }

  const std::span<const std::byte> initialData(mipChainBytes.data(),
                                               mipChainBytes.size());
  TextureDesc desc{
      .type = TextureType::TextureCube,
      .format = Format::RGBA32_FLOAT,
      .dimensions = {prepared.faceSize, prepared.faceSize, 1u},
      .usage = TextureUsage::Sampled,
      .storage = Storage::Device,
      .numLayers = 1u,
      .numSamples = 1u,
      .numMipLevels = std::max(1u, mipLevels),
      .data = initialData,
      .dataNumMipLevels = std::max(1u, mipLevels),
      .generateMipmaps = false,
  };
  return Texture::create(gpu, desc, debugName);
}

bool hasExpectedKtxProperties(const std::filesystem::path &path,
                              VkFormat expectedFormat,
                              uint32_t expectedBaseWidth,
                              uint32_t expectedBaseHeight,
                              uint32_t expectedMipLevels) {
  ktxTexture *texture = nullptr;
  const KTX_error_code createError = ktxTexture_CreateFromNamedFile(
      path.string().c_str(), KTX_TEXTURE_CREATE_NO_FLAGS, &texture);
  if (createError != KTX_SUCCESS || texture == nullptr) {
    return false;
  }

  bool formatMatches = false;
  bool shapeMatches = texture->baseWidth == expectedBaseWidth &&
                      texture->baseHeight == expectedBaseHeight &&
                      texture->numLevels == expectedMipLevels;
  if (texture->classId == ktxTexture2_c) {
    const auto *texture2 = reinterpret_cast<const ktxTexture2 *>(texture);
    formatMatches =
        texture2->vkFormat == static_cast<ktx_uint32_t>(expectedFormat);
  } else if (texture->classId == ktxTexture1_c) {
    const auto *texture1 = reinterpret_cast<const ktxTexture1 *>(texture);
    formatMatches = texture1->glInternalformat == kGlRgba32f;
  }

  ktxTexture_Destroy(texture);
  return formatMatches && shapeMatches;
}

bool outputsMatchExpectedShape(const EnvBakePlan &plan) {
  return hasExpectedKtxProperties(
             plan.outputs.irradiance, VK_FORMAT_R32G32B32A32_SFLOAT,
             plan.irradianceFaceSize, plan.irradianceFaceSize, 1u) &&
         hasExpectedKtxProperties(
             plan.outputs.ggx, VK_FORMAT_R32G32B32A32_SFLOAT,
             plan.specularFaceSize, plan.specularFaceSize, plan.mipCounts[1]) &&
         hasExpectedKtxProperties(
             plan.outputs.charlie, VK_FORMAT_R32G32B32A32_SFLOAT,
             plan.specularFaceSize, plan.specularFaceSize, plan.mipCounts[2]);
}

bool outputsUpToDateByTimestampOnly(const EnvBakePlan &plan) {
  std::error_code ecHdr;
  const auto hdrTime = std::filesystem::last_write_time(plan.hdrPath, ecHdr);
  if (ecHdr) {
    return false;
  }
  std::error_code ecShader;
  const auto shaderTime =
      std::filesystem::last_write_time(plan.shaderPath, ecShader);
  if (ecShader) {
    return false;
  }

  const std::array<std::filesystem::path, 3> outputs = {
      plan.outputs.irradiance, plan.outputs.ggx, plan.outputs.charlie};
  for (const auto &output : outputs) {
    std::error_code ecExists;
    if (!std::filesystem::exists(output, ecExists) || ecExists) {
      return false;
    }
    std::error_code ecOut;
    const auto outTime = std::filesystem::last_write_time(output, ecOut);
    if (ecOut || outTime < hdrTime || outTime < shaderTime) {
      return false;
    }
  }
  return true;
}

Result<bool, std::string> dispatchPrefilterTile(
    GPUDevice &gpu, ComputePipelineHandle pipeline, BufferHandle outputBuffer,
    uint64_t outputBufferAddress, uint32_t envMapTexId,
    uint32_t envMapSamplerId, uint32_t envMapWidth, uint32_t envMapHeight,
    EnvDistribution distribution, uint32_t faceSize, uint32_t faceIndex,
    uint32_t tileOffsetX, uint32_t tileOffsetY, uint32_t tileWidth,
    uint32_t tileHeight, float roughness, uint32_t sampleCount,
    std::span<std::byte> outBytes) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (faceSize == 0u) {
    return Result<bool, std::string>::makeError(
        "Env prefilter dispatch: face size must be > 0");
  }

  if (faceIndex >= 6u) {
    return Result<bool, std::string>::makeError(
        "Env prefilter dispatch: face index out of range");
  }
  if (envMapWidth == 0u || envMapHeight == 0u) {
    return Result<bool, std::string>::makeError(
        "Env prefilter dispatch: source env map dimensions must be non-zero");
  }
  if (tileWidth == 0u || tileHeight == 0u) {
    return Result<bool, std::string>::makeError(
        "Env prefilter dispatch: tile size must be non-zero");
  }
  if (tileOffsetX >= faceSize || tileOffsetY >= faceSize ||
      tileOffsetX + tileWidth > faceSize ||
      tileOffsetY + tileHeight > faceSize) {
    return Result<bool, std::string>::makeError(
        "Env prefilter dispatch: tile bounds exceed face size");
  }
  if (!nuri::isValid(outputBuffer)) {
    return Result<bool, std::string>::makeError(
        "Env prefilter dispatch: output buffer is invalid");
  }

  const uint64_t pixelCount =
      static_cast<uint64_t>(tileWidth) * static_cast<uint64_t>(tileHeight);
  const uint64_t faceBytes64 =
      pixelCount * 4ull * static_cast<uint64_t>(sizeof(float));
  if (faceBytes64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return Result<bool, std::string>::makeError(
        "Env prefilter dispatch: output buffer size overflows size_t");
  }
  const size_t faceBytes = static_cast<size_t>(faceBytes64);
  if (outBytes.size() < faceBytes) {
    return Result<bool, std::string>::makeError(
        "Env prefilter dispatch: output byte span is too small");
  }

  if (outputBufferAddress == 0u) {
    return Result<bool, std::string>::makeError(
        "Env prefilter dispatch: invalid output buffer address");
  }

  PrefilterPushConstants push{
      .faceSize = faceSize,
      .tileOffsetX = tileOffsetX,
      .tileOffsetY = tileOffsetY,
      .tileWidth = tileWidth,
      .tileHeight = tileHeight,
      .roughness = roughness,
      .sampleCount = sampleCount,
      .envMapTexId = envMapTexId,
      .envMapSamplerId = envMapSamplerId,
      .envMapWidth = envMapWidth,
      .envMapHeight = envMapHeight,
      .distribution = static_cast<uint32_t>(distribution),
      .faceIndex = faceIndex,
      ._padding0 = 0u,
      .outputBufferAddress = outputBufferAddress,
  };

  const ComputeDispatchItem dispatch{
      .pipeline = pipeline,
      .dispatch =
          {
              .x = (tileWidth + (kLocalSize - 1u)) / kLocalSize,
              .y = (tileHeight + (kLocalSize - 1u)) / kLocalSize,
              .z = 1u,
          },
      .pushConstants = std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(&push), sizeof(push)),
      .debugLabel = "Env Prefilter Compute",
      .debugColor = 0xff33aa33u,
  };

  std::string errorText;
  NURI_PROFILER_ZONE("EnvPrefilter::dispatchAndReadback",
                     NURI_PROFILER_COLOR_CREATE);
  auto submitResult = gpu.submitComputeDispatches(
      std::span<const ComputeDispatchItem>(&dispatch, 1));
  if (submitResult.hasError()) {
    errorText = "Env prefilter dispatch: compute submission failed: " +
                submitResult.error();
  } else {
    auto readResult =
        gpu.readBuffer(outputBuffer, 0u, outBytes.first(faceBytes));
    if (readResult.hasError()) {
      errorText = "Env prefilter dispatch: failed to read output buffer: " +
                  readResult.error();
    }
  }
  NURI_PROFILER_ZONE_END();
  if (!errorText.empty()) {
    return Result<bool, std::string>::makeError(std::move(errorText));
  }

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
writeCubemapKtx2Rgba32f(const std::filesystem::path &outputPath,
                        uint32_t baseFaceSize,
                        std::span<const std::vector<std::byte>> levelData) {
  if (baseFaceSize == 0u) {
    return Result<bool, std::string>::makeError(
        "Env prefilter write: base face size must be > 0");
  }
  if (levelData.empty()) {
    return Result<bool, std::string>::makeError(
        "Env prefilter write: no mip data to write");
  }

  const std::filesystem::path parent = outputPath.parent_path();
  if (!parent.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
    if (ec) {
      return Result<bool, std::string>::makeError(
          "Env prefilter write: failed to create output directory '" +
          parent.string() + "': " + ec.message());
    }
  }

  const ktxTextureCreateInfo createInfo{
      .glInternalformat = 0u,
      .vkFormat = VK_FORMAT_R32G32B32A32_SFLOAT,
      .baseWidth = baseFaceSize,
      .baseHeight = baseFaceSize,
      .baseDepth = 1u,
      .numDimensions = 2u,
      .numLevels = static_cast<ktx_uint32_t>(levelData.size()),
      .numLayers = 1u,
      .numFaces = 6u,
      .isArray = KTX_FALSE,
      .generateMipmaps = KTX_FALSE,
  };

  ktxTexture2 *textureKtx2 = nullptr;
  const KTX_error_code createError = ktxTexture2_Create(
      &createInfo, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &textureKtx2);
  if (createError != KTX_SUCCESS || textureKtx2 == nullptr) {
    return Result<bool, std::string>::makeError(
        "Env prefilter write: ktxTexture2_Create failed with code " +
        std::to_string(static_cast<int>(createError)));
  }

  ktxTexture *texture = ktxTexture(textureKtx2);

  for (uint32_t level = 0; level < levelData.size(); ++level) {
    const uint32_t faceSize = std::max(1u, baseFaceSize >> level);
    const size_t faceBytes = static_cast<size_t>(faceSize) *
                             static_cast<size_t>(faceSize) * 4u * sizeof(float);
    const size_t expectedLevelBytes = faceBytes * 6u;
    if (levelData[level].size() != expectedLevelBytes) {
      ktxTexture_Destroy(texture);
      return Result<bool, std::string>::makeError(
          "Env prefilter write: invalid level byte size at level " +
          std::to_string(level));
    }

    for (uint32_t face = 0; face < 6u; ++face) {
      const std::byte *src =
          levelData[level].data() + (face * static_cast<size_t>(faceBytes));
      const KTX_error_code setError = ktxTexture_SetImageFromMemory(
          texture, level, 0u, face, reinterpret_cast<const ktx_uint8_t *>(src),
          faceBytes);
      if (setError != KTX_SUCCESS) {
        ktxTexture_Destroy(texture);
        return Result<bool, std::string>::makeError(
            "Env prefilter write: ktxTexture_SetImageFromMemory failed at "
            "level " +
            std::to_string(level) + ", face " + std::to_string(face) +
            " with code " + std::to_string(static_cast<int>(setError)));
      }
    }
  }

  const std::string outputPathStr = outputPath.string();
  const KTX_error_code writeError =
      ktxTexture_WriteToNamedFile(texture, outputPathStr.c_str());
  ktxTexture_Destroy(texture);
  if (writeError != KTX_SUCCESS) {
    return Result<bool, std::string>::makeError(
        "Env prefilter write: ktxTexture_WriteToNamedFile failed with code " +
        std::to_string(static_cast<int>(writeError)));
  }

  return Result<bool, std::string>::makeResult(true);
}

} // namespace

std::string_view envDistributionName(EnvDistribution distribution) {
  switch (distribution) {
  case EnvDistribution::Lambertian:
    return "Lambertian";
  case EnvDistribution::GGX:
    return "GGX";
  case EnvDistribution::Charlie:
    return "Charlie";
  default:
    break;
  }
  return "Unknown";
}

Result<EnvBakePlan, std::string>
planEnvmapPrefilterBake(const RuntimeConfig &config,
                        const std::filesystem::path &environmentHdrPath,
                        bool forceRebuild) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  EnvBakePlan plan{};
  plan.shaderPath = config.roots.shaders / kPrefilterShaderFileName;

  std::filesystem::path hdrPath(environmentHdrPath);
  if (!hdrPath.is_absolute()) {
    hdrPath = config.roots.textures / hdrPath;
  }
  plan.hdrPath = hdrPath.lexically_normal();

  if (!std::filesystem::exists(plan.hdrPath)) {
    return Result<EnvBakePlan, std::string>::makeError(
        "Env prefilter baker: input HDR does not exist: '" +
        plan.hdrPath.string() + "'");
  }
  if (!std::filesystem::exists(plan.shaderPath)) {
    return Result<EnvBakePlan, std::string>::makeError(
        "Env prefilter baker: shader does not exist: '" +
        plan.shaderPath.string() + "'");
  }

  plan.outputs = makeOutputPaths(config, plan.hdrPath);
  plan.irradianceFaceSize = kIrradianceFaceSize;
  plan.specularFaceSize = 0u;
  plan.mipCounts = {1u, 0u, 0u};

  if (forceRebuild) {
    plan.shouldBake = true;
    return Result<EnvBakePlan, std::string>::makeResult(std::move(plan));
  }

  plan.shouldBake = !outputsUpToDateByTimestampOnly(plan);
  return Result<EnvBakePlan, std::string>::makeResult(std::move(plan));
}

Result<EnvSetupProgress, std::string>
advanceEnvmapPrefilterSetup(GPUDevice &gpu, const EnvBakePlan &plan,
                            EnvBakeGpuState &state) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  switch (state.setupPhase) {
  case EnvSetupPhase::StartCpuPrep: {
    cleanupEnvmapPrefilterBake(gpu, state);
    try {
      state.sourcePrepFuture =
          std::async(std::launch::async, [hdrPath = plan.hdrPath]() {
            return prepareSourceCubemapCpu(hdrPath);
          }).share();
    } catch (const std::exception &e) {
      return Result<EnvSetupProgress, std::string>::makeError(
          std::string("Env prefilter baker: failed to launch CPU prepare: ") +
          e.what());
    } catch (...) {
      return Result<EnvSetupProgress, std::string>::makeError(
          "Env prefilter baker: failed to launch CPU prepare");
    }
    state.sourcePrepInFlight = true;
    state.setupPhase = EnvSetupPhase::WaitCpuPrep;
    return Result<EnvSetupProgress, std::string>::makeResult(
        EnvSetupProgress{.status = EnvSetupStatus::InProgress,
                         .summary = "Preparing HDR (CPU)"});
  }

  case EnvSetupPhase::WaitCpuPrep: {
    if (!state.sourcePrepInFlight || !state.sourcePrepFuture.valid()) {
      return Result<EnvSetupProgress, std::string>::makeError(
          "Env prefilter baker: CPU prepare future is invalid");
    }
    const auto waitResult =
        state.sourcePrepFuture.wait_for(std::chrono::seconds(0));
    if (waitResult != std::future_status::ready) {
      return Result<EnvSetupProgress, std::string>::makeResult(
          EnvSetupProgress{.status = EnvSetupStatus::InProgress,
                           .summary = "Preparing HDR (CPU)"});
    }

    auto preparedResult = state.sourcePrepFuture.get();
    state.sourcePrepInFlight = false;
    state.sourcePrepFuture = {};
    if (preparedResult.hasError()) {
      return Result<EnvSetupProgress, std::string>::makeError(
          preparedResult.error());
    }
    state.preparedSourceCpu = std::move(preparedResult.value());
    if (!state.preparedSourceCpu.has_value() ||
        state.preparedSourceCpu->faceSize == 0u) {
      return Result<EnvSetupProgress, std::string>::makeError(
          "Env prefilter baker: prepared source cubemap is invalid");
    }

    const uint32_t sourceFace = state.preparedSourceCpu->faceSize;
    if (sourceFace == 0u) {
      return Result<EnvSetupProgress, std::string>::makeError(
          "Env prefilter baker: source face size resolved to 0");
    }
    const uint32_t specularFace =
        (plan.specularFaceSize > 0u) ? plan.specularFaceSize : sourceFace;
    const uint32_t ggxMip = (plan.mipCounts[1] > 0u)
                                ? plan.mipCounts[1]
                                : computeMipLevelCount(specularFace);
    const uint32_t charlieMip = (plan.mipCounts[2] > 0u)
                                    ? plan.mipCounts[2]
                                    : computeMipLevelCount(specularFace);

    EnvBakePlan resolvedPlan = plan;
    resolvedPlan.specularFaceSize = specularFace;
    resolvedPlan.mipCounts = {1u, ggxMip, charlieMip};
    if (outputsUpToDateByTimestampOnly(resolvedPlan) &&
        outputsMatchExpectedShape(resolvedPlan)) {
      cleanupEnvmapPrefilterBake(gpu, state);
      state.setupPhase = EnvSetupPhase::Done;
      return Result<EnvSetupProgress, std::string>::makeResult(EnvSetupProgress{
          .status = EnvSetupStatus::Skipped, .summary = "Up-to-date"});
    }

    state.setupPhase = EnvSetupPhase::CreateTexture;
    return Result<EnvSetupProgress, std::string>::makeResult(EnvSetupProgress{
        .status = EnvSetupStatus::InProgress, .summary = "HDR prepared"});
  }

  case EnvSetupPhase::CreateTexture: {
    if (!state.preparedSourceCpu.has_value()) {
      return Result<EnvSetupProgress, std::string>::makeError(
          "Env prefilter baker: missing prepared source cubemap data");
    }
    auto cubemapResult = createSourceCubemapTextureFromPrepared(
        gpu, *state.preparedSourceCpu, kSourceCubemapDebugName);
    state.preparedSourceCpu.reset();
    if (cubemapResult.hasError()) {
      return Result<EnvSetupProgress, std::string>::makeError(
          "Env prefilter baker: source cubemap creation failed: " +
          cubemapResult.error());
    }
    state.sourceCubemap = std::move(cubemapResult.value());
    if (!state.sourceCubemap || !state.sourceCubemap->valid()) {
      return Result<EnvSetupProgress, std::string>::makeError(
          "Env prefilter baker: source cubemap is invalid");
    }
    const TextureDimensions dims = state.sourceCubemap->dimensions();
    if (dims.width == 0u || dims.height == 0u || dims.width != dims.height) {
      return Result<EnvSetupProgress, std::string>::makeError(
          "Env prefilter baker: source cubemap dimensions are invalid");
    }
    state.sourceCubemapWidth = dims.width;
    state.sourceCubemapHeight = dims.height;
    state.setupPhase = EnvSetupPhase::CompileShader;
    return Result<EnvSetupProgress, std::string>::makeResult(
        EnvSetupProgress{.status = EnvSetupStatus::InProgress,
                         .summary = "Source cubemap created"});
  }

  case EnvSetupPhase::CompileShader: {
    auto shader = Shader::create("envmap_prefilter_baker", gpu);
    if (!shader) {
      return Result<EnvSetupProgress, std::string>::makeError(
          "Env prefilter baker: failed to create shader helper");
    }
    auto compileResult =
        shader->compileFromFile(plan.shaderPath.string(), ShaderStage::Compute);
    if (compileResult.hasError()) {
      return Result<EnvSetupProgress, std::string>::makeError(
          "Env prefilter baker: shader compile failed: " +
          compileResult.error());
    }
    state.computeShader = compileResult.value();
    state.setupPhase = EnvSetupPhase::CreatePipeline;
    return Result<EnvSetupProgress, std::string>::makeResult(EnvSetupProgress{
        .status = EnvSetupStatus::InProgress, .summary = "Shader compiled"});
  }

  case EnvSetupPhase::CreatePipeline: {
    const ComputePipelineDesc pipelineDesc{.computeShader =
                                               state.computeShader};
    auto pipelineResult =
        gpu.createComputePipeline(pipelineDesc, kComputePipelineDebugName);
    if (pipelineResult.hasError()) {
      return Result<EnvSetupProgress, std::string>::makeError(
          "Env prefilter baker: compute pipeline creation failed: " +
          pipelineResult.error());
    }
    state.computePipeline = pipelineResult.value();
    state.envMapTexId =
        gpu.getTextureBindlessIndex(state.sourceCubemap->handle());
    state.envMapSamplerId = gpu.getCubemapSamplerBindlessIndex();
    state.setupPhase = EnvSetupPhase::Finalize;
    return Result<EnvSetupProgress, std::string>::makeResult(EnvSetupProgress{
        .status = EnvSetupStatus::InProgress, .summary = "Pipeline created"});
  }

  case EnvSetupPhase::Finalize: {
    const uint32_t sourceFace = state.sourceCubemap->dimensions().width;
    if (sourceFace == 0u) {
      return Result<EnvSetupProgress, std::string>::makeError(
          "Env prefilter baker: source face size resolved to 0");
    }
    const uint32_t specularFace =
        (plan.specularFaceSize > 0u) ? plan.specularFaceSize : sourceFace;
    const uint32_t ggxMip = (plan.mipCounts[1] > 0u)
                                ? plan.mipCounts[1]
                                : computeMipLevelCount(specularFace);
    const uint32_t charlieMip = (plan.mipCounts[2] > 0u)
                                    ? plan.mipCounts[2]
                                    : computeMipLevelCount(specularFace);

    if (!nuri::isValid(state.tileOutputBuffer)) {
      const uint32_t tileSize =
          (state.bakeTileSize > 0u) ? state.bakeTileSize : 64u;
      const uint32_t maxFace = std::max(kIrradianceFaceSize, specularFace);
      const uint32_t effectiveTile = std::min(tileSize, maxFace);
      const size_t maxTileBytes = static_cast<size_t>(effectiveTile) *
                                  static_cast<size_t>(effectiveTile) * 4u *
                                  sizeof(float);

      const BufferDesc outputDesc{
          .usage = BufferUsage::Storage,
          .storage = Storage::HostVisible,
          .size = maxTileBytes,
      };
      auto outputBufferResult =
          gpu.createBuffer(outputDesc, kComputeOutputBufferDebugName);
      if (outputBufferResult.hasError()) {
        return Result<EnvSetupProgress, std::string>::makeError(
            "Env prefilter baker: tile output buffer creation failed: " +
            outputBufferResult.error());
      }
      state.tileOutputBuffer = outputBufferResult.value();
      state.tileOutputBufferBytes = maxTileBytes;
      state.tileScratchBytes.resize(maxTileBytes);
    }
    state.tileOutputBufferAddress =
        gpu.getBufferDeviceAddress(state.tileOutputBuffer);
    if (state.tileOutputBufferAddress == 0u) {
      return Result<EnvSetupProgress, std::string>::makeError(
          "Env prefilter baker: tile output buffer address is invalid");
    }

    state.runs[0] = EnvDistributionRuntime{
        .distribution = EnvDistribution::Lambertian,
        .baseFaceSize = kIrradianceFaceSize,
        .sampleCount = kLambertSamples,
        .mipLevels = 1u,
        .outputPath = plan.outputs.irradiance,
        .levels = std::vector<std::vector<std::byte>>(1u),
    };
    state.runs[1] = EnvDistributionRuntime{
        .distribution = EnvDistribution::GGX,
        .baseFaceSize = specularFace,
        .sampleCount = kGgxBaseSamples,
        .mipLevels = ggxMip,
        .outputPath = plan.outputs.ggx,
        .levels = std::vector<std::vector<std::byte>>(ggxMip),
    };
    state.runs[2] = EnvDistributionRuntime{
        .distribution = EnvDistribution::Charlie,
        .baseFaceSize = specularFace,
        .sampleCount = kCharlieBaseSamples,
        .mipLevels = charlieMip,
        .outputPath = plan.outputs.charlie,
        .levels = std::vector<std::vector<std::byte>>(charlieMip),
    };

    state.activeDistributionIndex = 0u;
    state.activeMipLevel = 0u;
    state.activeFace = 0u;
    state.activeTileIndex = 0u;
    state.completedSteps = 0u;
    state.totalSteps = 0u;
    {
      const uint32_t ts = (state.bakeTileSize > 0u) ? state.bakeTileSize : 64u;
      for (const EnvDistributionRuntime &run : state.runs) {
        for (uint32_t mip = 0u; mip < run.mipLevels; ++mip) {
          const uint32_t mipFace = std::max(1u, run.baseFaceSize >> mip);
          const uint32_t tilesX = (mipFace + (ts - 1u)) / ts;
          const uint32_t tilesY = (mipFace + (ts - 1u)) / ts;
          state.totalSteps += tilesX * tilesY * 6u;
        }
      }
    }
    state.initialized = true;
    state.setupPhase = EnvSetupPhase::Done;
    return Result<EnvSetupProgress, std::string>::makeResult(EnvSetupProgress{
        .status = EnvSetupStatus::Ready, .summary = "GPU setup complete"});
  }

  case EnvSetupPhase::Done:
    if (state.initialized) {
      return Result<EnvSetupProgress, std::string>::makeResult(EnvSetupProgress{
          .status = EnvSetupStatus::Ready, .summary = "GPU setup complete"});
    }
    return Result<EnvSetupProgress, std::string>::makeResult(EnvSetupProgress{
        .status = EnvSetupStatus::Skipped, .summary = "Up-to-date"});
  }

  return Result<EnvSetupProgress, std::string>::makeError(
      "Env prefilter baker: invalid setup phase");
}

Result<EnvGpuStepProgress, std::string>
runEnvmapPrefilterBakeOneStep(GPUDevice &gpu, EnvBakeGpuState &state) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!state.initialized || !nuri::isValid(state.computePipeline)) {
    return Result<EnvGpuStepProgress, std::string>::makeError(
        "Env prefilter bake: GPU state is not initialized");
  }

  if (state.activeDistributionIndex >= state.runs.size()) {
    return Result<EnvGpuStepProgress, std::string>::makeResult(
        EnvGpuStepProgress{
            .completedSteps = state.completedSteps,
            .totalSteps = state.totalSteps,
            .finished = true,
        });
  }

  EnvDistributionRuntime &run = state.runs[state.activeDistributionIndex];
  if (state.activeMipLevel >= run.mipLevels) {
    return Result<EnvGpuStepProgress, std::string>::makeError(
        "Env prefilter bake: active mip is out of range");
  }
  if (state.activeFace >= 6u) {
    return Result<EnvGpuStepProgress, std::string>::makeError(
        "Env prefilter bake: active face is out of range");
  }

  const uint32_t ts = (state.bakeTileSize > 0u) ? state.bakeTileSize : 64u;
  const uint32_t faceSize =
      std::max(1u, run.baseFaceSize >> state.activeMipLevel);
  const uint32_t tilesX = (faceSize + (ts - 1u)) / ts;
  const uint32_t tilesY = (faceSize + (ts - 1u)) / ts;
  const uint32_t tilesPerFace = tilesX * tilesY;
  if (tilesPerFace == 0u) {
    return Result<EnvGpuStepProgress, std::string>::makeError(
        "Env prefilter bake: zero tiles per face");
  }
  if (state.activeTileIndex >= tilesPerFace) {
    return Result<EnvGpuStepProgress, std::string>::makeError(
        "Env prefilter bake: active tile index out of range");
  }

  const uint32_t tileXIndex = state.activeTileIndex % tilesX;
  const uint32_t tileYIndex = state.activeTileIndex / tilesX;
  const uint32_t tileOffsetX = tileXIndex * ts;
  const uint32_t tileOffsetY = tileYIndex * ts;
  const uint32_t tileWidth = std::min(ts, faceSize - tileOffsetX);
  const uint32_t tileHeight = std::min(ts, faceSize - tileOffsetY);
  const size_t tileBytes = static_cast<size_t>(tileWidth) *
                           static_cast<size_t>(tileHeight) * 4u * sizeof(float);
  if (tileBytes > state.tileOutputBufferBytes ||
      tileBytes > state.tileScratchBytes.size()) {
    return Result<EnvGpuStepProgress, std::string>::makeError(
        "Env prefilter bake: tile buffer is smaller than requested tile");
  }
  if (state.tileOutputBufferAddress == 0u) {
    return Result<EnvGpuStepProgress, std::string>::makeError(
        "Env prefilter bake: tile output buffer address is invalid");
  }
  const float roughness = (run.mipLevels > 1u)
                              ? static_cast<float>(state.activeMipLevel) /
                                    static_cast<float>(run.mipLevels - 1u)
                              : 0.0f;
  if (state.sourceCubemapWidth == 0u || state.sourceCubemapHeight == 0u) {
    return Result<EnvGpuStepProgress, std::string>::makeError(
        "Env prefilter bake: source cubemap dimensions are invalid");
  }
  auto tileResult = dispatchPrefilterTile(
      gpu, state.computePipeline, state.tileOutputBuffer,
      state.tileOutputBufferAddress, state.envMapTexId, state.envMapSamplerId,
      state.sourceCubemapWidth, state.sourceCubemapHeight, run.distribution,
      faceSize, state.activeFace, tileOffsetX, tileOffsetY, tileWidth,
      tileHeight, roughness, run.sampleCount,
      std::span<std::byte>(state.tileScratchBytes.data(), tileBytes));
  if (tileResult.hasError()) {
    return Result<EnvGpuStepProgress, std::string>::makeError(
        "Env prefilter bake (" +
        std::string(envDistributionName(run.distribution)) +
        ", mip=" + std::to_string(state.activeMipLevel) +
        ", face=" + std::to_string(state.activeFace) + ", tile=" +
        std::to_string(state.activeTileIndex) + "): " + tileResult.error());
  }

  constexpr size_t kPixelBytes = 4u * sizeof(float);
  const size_t singleFaceBytes = static_cast<size_t>(faceSize) *
                                 static_cast<size_t>(faceSize) * 4u *
                                 sizeof(float);
  const size_t levelBytes = singleFaceBytes * 6u;
  std::vector<std::byte> &level = run.levels[state.activeMipLevel];
  if (level.empty()) {
    level.resize(levelBytes);
  }
  if (level.size() != levelBytes) {
    return Result<EnvGpuStepProgress, std::string>::makeError(
        "Env prefilter bake: invalid level byte size allocation");
  }

  const size_t faceBaseOffset =
      static_cast<size_t>(state.activeFace) * singleFaceBytes;
  if (tileWidth == faceSize && tileHeight == faceSize) {
    std::memcpy(level.data() + faceBaseOffset, state.tileScratchBytes.data(),
                singleFaceBytes);
  } else {
    for (uint32_t row = 0u; row < tileHeight; ++row) {
      const size_t srcOffset = static_cast<size_t>(row) *
                               static_cast<size_t>(tileWidth) * kPixelBytes;
      const size_t dstOffset =
          faceBaseOffset +
          ((static_cast<size_t>(tileOffsetY) + static_cast<size_t>(row)) *
               static_cast<size_t>(faceSize) +
           static_cast<size_t>(tileOffsetX)) *
              kPixelBytes;
      std::memcpy(level.data() + dstOffset,
                  state.tileScratchBytes.data() + srcOffset,
                  static_cast<size_t>(tileWidth) * kPixelBytes);
    }
  }

  ++state.completedSteps;
  ++state.activeTileIndex;
  if (state.activeTileIndex >= tilesPerFace) {
    state.activeTileIndex = 0u;
    ++state.activeFace;
  }
  if (state.activeFace >= 6u) {
    state.activeFace = 0u;
    ++state.activeMipLevel;
  }
  if (state.activeMipLevel >= run.mipLevels) {
    state.activeDistributionIndex++;
    state.activeMipLevel = 0u;
    state.activeFace = 0u;
    state.activeTileIndex = 0u;
  }

  const bool finished = state.activeDistributionIndex >= state.runs.size();
  return Result<EnvGpuStepProgress, std::string>::makeResult(EnvGpuStepProgress{
      .completedSteps = state.completedSteps,
      .totalSteps = state.totalSteps,
      .finished = finished,
  });
}

EnvWritePayload collectEnvWritePayload(EnvBakeGpuState &state) {
  EnvWritePayload payload{};
  for (size_t i = 0; i < state.runs.size(); ++i) {
    payload.distributions[i].distribution = state.runs[i].distribution;
    payload.distributions[i].baseFaceSize = state.runs[i].baseFaceSize;
    payload.distributions[i].outputPath = state.runs[i].outputPath;
    payload.distributions[i].levelData = std::move(state.runs[i].levels);
  }
  return payload;
}

void cleanupEnvmapPrefilterBake(GPUDevice &gpu, EnvBakeGpuState &state) {
  if (state.sourceCubemap && state.sourceCubemap->valid()) {
    gpu.destroyTexture(state.sourceCubemap->handle());
  }
  state.sourceCubemap.reset();

  if (nuri::isValid(state.computePipeline)) {
    gpu.destroyComputePipeline(state.computePipeline);
    state.computePipeline = {};
  }
  if (nuri::isValid(state.computeShader)) {
    gpu.destroyShaderModule(state.computeShader);
    state.computeShader = {};
  }
  if (nuri::isValid(state.tileOutputBuffer)) {
    gpu.destroyBuffer(state.tileOutputBuffer);
    state.tileOutputBuffer = {};
  }
  if (state.sourcePrepInFlight && state.sourcePrepFuture.valid()) {
    state.sourcePrepFuture.wait();
  }

  state.envMapTexId = 0u;
  state.envMapSamplerId = 0u;
  state.sourceCubemapWidth = 0u;
  state.sourceCubemapHeight = 0u;
  state.tileOutputBufferAddress = 0u;
  state.tileOutputBufferBytes = 0u;
  state.tileScratchBytes.clear();
  state.preparedSourceCpu.reset();
  state.sourcePrepFuture = {};
  state.sourcePrepInFlight = false;
  state.setupPhase = EnvSetupPhase::StartCpuPrep;
  state.activeDistributionIndex = 0u;
  state.activeMipLevel = 0u;
  state.activeFace = 0u;
  state.activeTileIndex = 0u;
  state.bakeTileSize = 0u;
  state.completedSteps = 0u;
  state.totalSteps = 0u;
  state.initialized = false;
}

Result<bool, std::string>
writeEnvmapPrefilterOutputs(const EnvWritePayload &payload) {
  for (const auto &distribution : payload.distributions) {
    auto writeResult = writeCubemapKtx2Rgba32f(
        distribution.outputPath, distribution.baseFaceSize,
        std::span<const std::vector<std::byte>>(distribution.levelData.data(),
                                                distribution.levelData.size()));
    if (writeResult.hasError()) {
      return Result<bool, std::string>::makeError(
          "Env prefilter write (" +
          std::string(envDistributionName(distribution.distribution)) +
          "): " + writeResult.error());
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri::bakery::detail
