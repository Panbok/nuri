#include "nuri/pch.h"

#include "nuri/gfx/pipeline/providers/scene_lighting_provider.h"

#include "nuri/core/profiling.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/scene/render_scene.h"
#include "nuri/utils/utils.h"

namespace nuri {
namespace {

enum ForwardSceneFlags : uint8_t {
  kForwardSceneHasIblDiffuse = 1u << 0u,
  kForwardSceneHasIblSpecular = 1u << 1u,
  kForwardSceneHasIblSheen = 1u << 2u,
  kForwardSceneHasBrdfLut = 1u << 3u,
  kForwardSceneOutputLinearToSrgb = 1u << 4u,
  kForwardSceneHasSceneColor = 1u << 5u,
  kForwardSceneHasSceneDepth = 1u << 6u,
  kForwardSceneHasSceneDepthPyramid = 1u << 7u,
};

struct SceneDataBufferLayout {
  size_t frameDataOffset = 0u;
  size_t directionalLightsOffset = 0u;
  size_t localLightsOffset = 0u;
  size_t totalBytes = 0u;
};

[[nodiscard]] SceneDataBufferLayout
makeSceneDataBufferLayout(size_t frameDataBytes, size_t directionalLightBytes,
                          size_t localLightBytes) {
  const size_t directionalOffset =
      alignUp(frameDataBytes, alignof(DirectionalLightGpuData));
  const size_t localOffset = alignUp(directionalOffset + directionalLightBytes,
                                     alignof(LocalLightGpuData));
  return SceneDataBufferLayout{
      .frameDataOffset = 0u,
      .directionalLightsOffset = directionalOffset,
      .localLightsOffset = localOffset,
      .totalBytes = std::max(localOffset + localLightBytes, frameDataBytes),
  };
}

[[nodiscard]] const RenderSettings::TextureFilteringSettings &
textureFilteringSettingsOrDefault(const RenderFrameContext &frame) {
  static const RenderSettings kDefaultSettings{};
  return frame.settings ? frame.settings->textureFiltering
                        : kDefaultSettings.textureFiltering;
}

[[nodiscard]] uint32_t resolveMaterialSamplerId(
    GPUDevice &gpu, const RenderSettings::TextureFilteringSettings &settings) {
  switch (sanitizeTextureFilterMode(settings.mode)) {
  case TextureFilterMode::Bilinear:
    return gpu.getLinearRepeatSamplerBindlessIndex(false, 1u);
  case TextureFilterMode::Anisotropic:
    return gpu.getLinearRepeatSamplerBindlessIndex(
        true, sanitizeTextureFilterAnisotropy(settings.anisotropy));
  case TextureFilterMode::Trilinear:
  default:
    return gpu.getLinearRepeatSamplerBindlessIndex(true, 1u);
  }
}

} // namespace

SceneLightingProvider::SceneLightingProvider(GPUDevice &gpu) : gpu_(gpu) {}

SceneLightingProvider::~SceneLightingProvider() { destroyBuffer(); }

Result<bool, std::string>
SceneLightingProvider::prepare(FrameBuildContext &ctx) {
  NURI_PROFILER_FUNCTION();
  RenderFrameContext &frame = ctx.frame;
  if (frame.scene == nullptr) {
    return Result<bool, std::string>::makeError(
        "SceneLightingProvider::prepare: frame scene is null");
  }
  if (frame.resources == nullptr) {
    return Result<bool, std::string>::makeError(
        "SceneLightingProvider::prepare: frame resources are null");
  }
  if (!ctx.shared.materialTableGpuData.has_value()) {
    return Result<bool, std::string>::makeError(
        "SceneLightingProvider::prepare: material table GPU data is "
        "unavailable");
  }

  const std::span<const DirectionalLightGpuData> directionalLights =
      frame.scene->packedDirectionalLights();
  const std::span<const LocalLightGpuData> localLights =
      frame.scene->packedLocalLights();
  const SceneDataBufferLayout layout = makeSceneDataBufferLayout(
      sizeof(ForwardSceneFrameData),
      directionalLights.size() * sizeof(DirectionalLightGpuData),
      localLights.size() * sizeof(LocalLightGpuData));

  auto bufferResult = ensureBufferCapacity(layout.totalBytes);
  if (bufferResult.hasError()) {
    return bufferResult;
  }

  uint32_t cubemapTexId = kInvalidTextureBindlessIndex;
  uint32_t hasCubemap = 0u;
  uint32_t irradianceTexId = kInvalidTextureBindlessIndex;
  uint32_t prefilteredGgxTexId = kInvalidTextureBindlessIndex;
  uint32_t prefilteredCharlieTexId = kInvalidTextureBindlessIndex;
  uint32_t brdfLutTexId = kInvalidTextureBindlessIndex;
  uint32_t flags = 0u;
  const uint32_t cubemapSamplerId = gpu_.getCubemapSamplerBindlessIndex();
  const uint32_t materialSamplerId =
      resolveMaterialSamplerId(gpu_, textureFilteringSettingsOrDefault(frame));
  const EnvironmentHandles environment = frame.scene->environment();

  if (const TextureRecord *cubemap =
          frame.resources->tryGet(environment.cubemap);
      cubemap != nullptr && nuri::isValid(cubemap->texture) &&
      cubemap->bindlessIndex != kInvalidTextureBindlessIndex) {
    cubemapTexId = cubemap->bindlessIndex;
    hasCubemap = 1u;
  }
  if (const TextureRecord *irradiance =
          frame.resources->tryGet(environment.irradiance);
      irradiance != nullptr && nuri::isValid(irradiance->texture) &&
      irradiance->bindlessIndex != kInvalidTextureBindlessIndex) {
    irradianceTexId = irradiance->bindlessIndex;
    flags |= kForwardSceneHasIblDiffuse;
  }
  if (const TextureRecord *prefilteredGgx =
          frame.resources->tryGet(environment.prefilteredGgx);
      prefilteredGgx != nullptr && nuri::isValid(prefilteredGgx->texture) &&
      prefilteredGgx->bindlessIndex != kInvalidTextureBindlessIndex) {
    prefilteredGgxTexId = prefilteredGgx->bindlessIndex;
    flags |= kForwardSceneHasIblSpecular;
  }
  if (const TextureRecord *prefilteredCharlie =
          frame.resources->tryGet(environment.prefilteredCharlie);
      prefilteredCharlie != nullptr &&
      nuri::isValid(prefilteredCharlie->texture) &&
      prefilteredCharlie->bindlessIndex != kInvalidTextureBindlessIndex) {
    prefilteredCharlieTexId = prefilteredCharlie->bindlessIndex;
    flags |= kForwardSceneHasIblSheen;
  } else if ((flags & kForwardSceneHasIblSpecular) != 0u) {
    // Reuse the GGX prefilter as a sheen approximation when Charlie is
    // missing; keep kForwardSceneHasIblSheen set so downstream shading knows a
    // fallback sheen source is available.
    prefilteredCharlieTexId = prefilteredGgxTexId;
    flags |= kForwardSceneHasIblSheen;
  }
  if (const TextureRecord *brdfLut =
          frame.resources->tryGet(environment.brdfLut);
      brdfLut != nullptr && nuri::isValid(brdfLut->texture) &&
      brdfLut->bindlessIndex != kInvalidTextureBindlessIndex) {
    brdfLutTexId = brdfLut->bindlessIndex;
    flags |= kForwardSceneHasBrdfLut;
  }

  uint32_t sceneColorTexId = kInvalidTextureBindlessIndex;
  uint32_t sceneColorHalfResTexId = kInvalidTextureBindlessIndex;
  uint32_t sceneColorQuarterResTexId = kInvalidTextureBindlessIndex;
  uint32_t sceneColorSamplerId = 0u;
  uint32_t sceneDepthTexId = kInvalidTextureBindlessIndex;
  uint32_t sceneDepthSamplerId = ctx.shared.sceneDepthSamplerId;
  uint32_t sceneDepthPyramidLevelCount = 0u;
  std::array<glm::uvec4, 4> sceneDepthPyramidTexIds{};
  if (nuri::isValid(ctx.shared.sceneColorTexture)) {
    sceneColorTexId =
        gpu_.getTextureBindlessIndex(ctx.shared.sceneColorTexture);
    sceneColorSamplerId = gpu_.getLinearRepeatSamplerBindlessIndex(true, 1u);
    if (sceneColorTexId != kInvalidTextureBindlessIndex) {
      flags |= kForwardSceneHasSceneColor;
    }
  }
  if ((flags & kForwardSceneHasSceneColor) != 0u &&
      nuri::isValid(ctx.shared.sceneColorHalfResTexture)) {
    sceneColorHalfResTexId =
        gpu_.getTextureBindlessIndex(ctx.shared.sceneColorHalfResTexture);
  }
  if ((flags & kForwardSceneHasSceneColor) != 0u &&
      nuri::isValid(ctx.shared.sceneColorQuarterResTexture)) {
    sceneColorQuarterResTexId =
        gpu_.getTextureBindlessIndex(ctx.shared.sceneColorQuarterResTexture);
  }
  if (nuri::isValid(ctx.shared.sceneDepthTexture)) {
    sceneDepthTexId =
        gpu_.getTextureBindlessIndex(ctx.shared.sceneDepthTexture);
    if (sceneDepthTexId != kInvalidTextureBindlessIndex) {
      flags |= kForwardSceneHasSceneDepth;
    }
  }
  if ((flags & kForwardSceneHasSceneDepth) != 0u &&
      ctx.shared.sceneDepthPyramidLevelCount > 0u) {
    const uint32_t candidateLevelCount = std::min<uint32_t>(
        ctx.shared.sceneDepthPyramidLevelCount, kMaxSceneDepthPyramidLevels);
    for (uint32_t level = 0u; level < candidateLevelCount; ++level) {
      const TextureHandle texture = ctx.shared.sceneDepthPyramidTextures[level];
      if (!nuri::isValid(texture)) {
        break;
      }
      const uint32_t texId = gpu_.getTextureBindlessIndex(texture);
      if (texId == kInvalidTextureBindlessIndex) {
        break;
      }
      sceneDepthPyramidTexIds[level >> 2u][level & 3u] = texId;
      sceneDepthPyramidLevelCount = level + 1u;
    }
    if (sceneDepthPyramidLevelCount > 0u) {
      flags |= kForwardSceneHasSceneDepthPyramid;
    }
  }
  if (gpu_.getSwapchainFormat() != Format::RGBA8_SRGB) {
    flags |= kForwardSceneOutputLinearToSrgb;
  }

  const uint64_t sceneDataBaseAddress =
      gpu_.getBufferDeviceAddress(sceneDataBuffer_->handle());
  if (sceneDataBaseAddress == 0u) {
    return Result<bool, std::string>::makeError(
        "SceneLightingProvider::prepare: invalid scene data buffer address");
  }
  const uint64_t directionalLightBufferAddress =
      directionalLights.empty()
          ? 0u
          : sceneDataBaseAddress + layout.directionalLightsOffset;
  const uint64_t localLightBufferAddress =
      localLights.empty() ? 0u
                          : sceneDataBaseAddress + layout.localLightsOffset;
  const uint32_t directionalLightCount = static_cast<uint32_t>(std::min<size_t>(
      directionalLights.size(), std::numeric_limits<uint32_t>::max()));
  const uint32_t localLightCount = static_cast<uint32_t>(std::min<size_t>(
      localLights.size(), std::numeric_limits<uint32_t>::max()));

  const ForwardSceneFrameData frameData{
      .view = frame.camera.view,
      .proj = frame.camera.proj,
      .cameraPos = frame.camera.cameraPos,
      .cubemapTexId = cubemapTexId,
      .hasCubemap = hasCubemap,
      .irradianceTexId = irradianceTexId,
      .prefilteredGgxTexId = prefilteredGgxTexId,
      .prefilteredCharlieTexId = prefilteredCharlieTexId,
      .brdfLutTexId = brdfLutTexId,
      .flags = flags,
      .cubemapSamplerId = cubemapSamplerId,
      .materialSamplerId = materialSamplerId,
      .sceneColorTexId = sceneColorTexId,
      .sceneColorSamplerId = sceneColorSamplerId,
      .sceneColorHalfResTexId = sceneColorHalfResTexId,
      .sceneColorQuarterResTexId = sceneColorQuarterResTexId,
      .sceneDepthTexId = sceneDepthTexId,
      .sceneDepthSamplerId = sceneDepthSamplerId,
      .sceneDepthPyramidLevelCount = sceneDepthPyramidLevelCount,
      .sceneDepthPyramidTexIds = sceneDepthPyramidTexIds,
      .directionalLightBufferAddress = directionalLightBufferAddress,
      .localLightBufferAddress = localLightBufferAddress,
      .materialHeaderBufferAddress =
          ctx.shared.materialTableGpuData->headerBufferAddress,
      .materialClearcoatBufferAddress =
          ctx.shared.materialTableGpuData->clearcoatBufferAddress,
      .materialSheenBufferAddress =
          ctx.shared.materialTableGpuData->sheenBufferAddress,
      .materialTransmissionBufferAddress =
          ctx.shared.materialTableGpuData->transmissionBufferAddress,
      .materialSpecularBufferAddress =
          ctx.shared.materialTableGpuData->specularBufferAddress,
      .directionalLightCount = directionalLightCount,
      .localLightCount = localLightCount,
  };

  const bool frameDataDirty =
      !frameDataUploadValid_ || uploadedFrameData_ != frameData;
  if (frameDataDirty) {
    const std::span<const std::byte> frameDataBytes{
        reinterpret_cast<const std::byte *>(&frameData), sizeof(frameData)};
    auto updateResult = gpu_.updateBuffer(
        sceneDataBuffer_->handle(), frameDataBytes, layout.frameDataOffset);
    if (updateResult.hasError()) {
      return updateResult;
    }
    uploadedFrameData_ = frameData;
    frameDataUploadValid_ = true;
  }

  const bool lightDirty =
      cachedLightTopologyVersion_ != frame.scene->lightTopologyVersion() ||
      cachedLightTransformVersion_ != frame.scene->lightTransformVersion();
  if (lightDirty) {
    if (!directionalLights.empty()) {
      const std::span<const std::byte> directionalLightBytes{
          reinterpret_cast<const std::byte *>(directionalLights.data()),
          directionalLights.size() * sizeof(DirectionalLightGpuData)};
      auto updateResult =
          gpu_.updateBuffer(sceneDataBuffer_->handle(), directionalLightBytes,
                            layout.directionalLightsOffset);
      if (updateResult.hasError()) {
        return updateResult;
      }
    }
    if (!localLights.empty()) {
      const std::span<const std::byte> localLightBytes{
          reinterpret_cast<const std::byte *>(localLights.data()),
          localLights.size() * sizeof(LocalLightGpuData)};
      auto updateResult =
          gpu_.updateBuffer(sceneDataBuffer_->handle(), localLightBytes,
                            layout.localLightsOffset);
      if (updateResult.hasError()) {
        return updateResult;
      }
    }
    cachedLightTopologyVersion_ = frame.scene->lightTopologyVersion();
    cachedLightTransformVersion_ = frame.scene->lightTransformVersion();
  }

  ctx.shared.forwardSceneGpuData = ForwardSceneGpuData{
      .buffer = sceneDataBuffer_->handle(),
      .frameDataAddress = sceneDataBaseAddress + layout.frameDataOffset,
      .directionalLightBufferAddress = directionalLightBufferAddress,
      .localLightBufferAddress = localLightBufferAddress,
      .directionalLightCount = directionalLightCount,
      .localLightCount = localLightCount,
  };

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
SceneLightingProvider::ensureBufferCapacity(size_t requiredBytes) {
  const size_t requested =
      std::max(requiredBytes, sizeof(ForwardSceneFrameData));
  if (sceneDataBuffer_ && sceneDataBuffer_->valid() &&
      sceneDataBufferCapacityBytes_ >= requested) {
    return Result<bool, std::string>::makeResult(true);
  }
  destroyBuffer();
  auto bufferResult = Buffer::create(gpu_,
                                     BufferDesc{.usage = BufferUsage::Storage,
                                                .storage = Storage::Device,
                                                .size = requested},
                                     "forward_scene_data");
  if (bufferResult.hasError()) {
    return Result<bool, std::string>::makeError(bufferResult.error());
  }
  sceneDataBuffer_ = std::move(bufferResult.value());
  sceneDataBufferCapacityBytes_ = requested;
  frameDataUploadValid_ = false;
  return Result<bool, std::string>::makeResult(true);
}

void SceneLightingProvider::destroyBuffer() {
  if (sceneDataBuffer_ && sceneDataBuffer_->valid()) {
    gpu_.destroyBuffer(sceneDataBuffer_->handle());
  }
  sceneDataBuffer_.reset();
  sceneDataBufferCapacityBytes_ = 0;
  frameDataUploadValid_ = false;
  cachedLightTopologyVersion_ = std::numeric_limits<uint64_t>::max();
  cachedLightTransformVersion_ = std::numeric_limits<uint64_t>::max();
}

} // namespace nuri
