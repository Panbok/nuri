#include "nuri/pch.h"

#include "nuri/gfx/pipeline/providers/scene_lighting_provider.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/scene/render_scene.h"
#include "nuri/utils/utils.h"

#include <algorithm>
#include <cmath>

namespace nuri {
namespace {

enum ForwardSceneFlags : uint32_t {
  kForwardSceneHasIblDiffuse = 1u << 0u,
  kForwardSceneHasIblSpecular = 1u << 1u,
  kForwardSceneHasIblSheen = 1u << 2u,
  kForwardSceneHasBrdfLut = 1u << 3u,
  kForwardSceneHasSceneColor = 1u << 5u,
  kForwardSceneHasSceneDepth = 1u << 6u,
  kForwardSceneHasSceneDepthPyramid = 1u << 7u,
  kForwardSceneTransmissionMipDebug = 1u << 8u,
  kForwardSceneHasAmbientOcclusion = 1u << 9u,
};

struct SceneDataBufferLayout {
  size_t frameDataOffset = 0u;
  size_t postTaaFrameDataOffset = 0u;
  size_t directionalLightsOffset = 0u;
  size_t localLightsOffset = 0u;
  size_t totalBytes = 0u;
};

[[nodiscard]] SceneDataBufferLayout
makeSceneDataBufferLayout(size_t frameDataBytes, size_t directionalLightBytes,
                          size_t localLightBytes) {
  const size_t postTaaFrameDataOffset =
      alignUp(frameDataBytes, alignof(ForwardSceneFrameData));
  const size_t directionalOffset =
      alignUp(postTaaFrameDataOffset + frameDataBytes,
              alignof(DirectionalLightGpuData));
  const size_t localOffset = alignUp(directionalOffset + directionalLightBytes,
                                     alignof(LocalLightGpuData));
  return SceneDataBufferLayout{
      .frameDataOffset = 0u,
      .postTaaFrameDataOffset = postTaaFrameDataOffset,
      .directionalLightsOffset = directionalOffset,
      .localLightsOffset = localOffset,
      .totalBytes = std::max(localOffset + localLightBytes,
                             postTaaFrameDataOffset + frameDataBytes),
  };
}

[[nodiscard]] const RenderSettings::TextureFilteringSettings &
textureFilteringSettingsOrDefault(const RenderFrameContext &frame) {
  static const RenderSettings kDefaultSettings{};
  return frame.settings ? frame.settings->textureFiltering
                        : kDefaultSettings.textureFiltering;
}

[[nodiscard]] uint32_t resolveDefaultMaterialSamplerId(
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

[[nodiscard]] float sanitizeTaaMaterialMipBias(float value) noexcept {
  return std::isfinite(value) ? std::clamp(value, -1.0f, 0.0f) : 0.0f;
}

[[nodiscard]] bool sameSamplerDesc(const SamplerDesc &lhs,
                                   const SamplerDesc &rhs) noexcept {
  return lhs.minFilter == rhs.minFilter && lhs.magFilter == rhs.magFilter &&
         lhs.mipMode == rhs.mipMode && lhs.wrapU == rhs.wrapU &&
         lhs.wrapV == rhs.wrapV && lhs.wrapW == rhs.wrapW &&
         lhs.mipLodMin == rhs.mipLodMin && lhs.mipLodMax == rhs.mipLodMax &&
         lhs.mipLodBias == rhs.mipLodBias &&
         lhs.maxAnisotropy == rhs.maxAnisotropy &&
         lhs.depthCompareEnabled == rhs.depthCompareEnabled &&
         lhs.depthCompareOp == rhs.depthCompareOp;
}

} // namespace

SceneLightingProvider::SceneLightingProvider(GPUDevice &gpu) : gpu_(gpu) {}

SceneLightingProvider::~SceneLightingProvider() {
  destroyBuffers();
  destroyCachedSamplers();
}

Result<uint32_t, std::string>
SceneLightingProvider::resolveMaterialSamplerId(RenderFrameContext &frame) {
  const RenderSettings &settings = renderSettingsOrDefault(frame);
  const RenderSettings::TextureFilteringSettings &filtering =
      textureFilteringSettingsOrDefault(frame);
  const TextureFilterMode filterMode =
      sanitizeTextureFilterMode(filtering.mode);
  const bool mipFilteringActive = filterMode != TextureFilterMode::Bilinear;
  const bool taaMode = sanitizeAntiAliasingMode(settings.antiAliasing.mode) ==
                       AntiAliasingMode::TAA;
  const float mipBias = sanitizeTaaMaterialMipBias(
      settings.antiAliasing.debug.taaMaterialMipBias);

  AntiAliasingFrameMetrics &aaMetrics = frame.metrics.antiAliasing;
  aaMetrics.taaMaterialMipBiasEnabled =
      settings.antiAliasing.debug.taaMaterialMipBiasEnabled;
  aaMetrics.taaMaterialMipBias = mipBias;
  aaMetrics.taaMaterialMipBiasApplied = false;

  if (!taaMode || !settings.antiAliasing.debug.taaMaterialMipBiasEnabled ||
      !mipFilteringActive || mipBias >= 0.0f) {
    return Result<uint32_t, std::string>::makeResult(
        resolveDefaultMaterialSamplerId(gpu_, filtering));
  }

  SamplerDesc samplerDesc{
      .minFilter = SamplerFilter::Linear,
      .magFilter = SamplerFilter::Linear,
      .mipMode = SamplerMipMode::Linear,
      .wrapU = SamplerWrapMode::Repeat,
      .wrapV = SamplerWrapMode::Repeat,
      .wrapW = SamplerWrapMode::Repeat,
      .mipLodBias = mipBias,
      .maxAnisotropy = 1u,
  };
  if (filterMode == TextureFilterMode::Anisotropic) {
    samplerDesc.maxAnisotropy =
        sanitizeTextureFilterAnisotropy(filtering.anisotropy);
  }

  auto samplerResult = ensureTaaMaterialMipBiasSampler(samplerDesc);
  if (samplerResult.hasError()) {
    return Result<uint32_t, std::string>::makeError(samplerResult.error());
  }
  const uint32_t samplerId =
      gpu_.getSamplerBindlessIndex(samplerResult.value());
  if (samplerId == kInvalidTextureBindlessIndex) {
    return Result<uint32_t, std::string>::makeError(
        "SceneLightingProvider::resolveMaterialSamplerId: invalid TAA "
        "material sampler bindless index");
  }

  aaMetrics.taaMaterialMipBiasApplied = true;
  return Result<uint32_t, std::string>::makeResult(samplerId);
}

Result<SamplerHandle, std::string>
SceneLightingProvider::ensureTaaMaterialMipBiasSampler(
    const SamplerDesc &desc) {
  if (nuri::isValid(taaMaterialMipBiasSampler_) &&
      hasTaaMaterialMipBiasSamplerDesc_ &&
      sameSamplerDesc(taaMaterialMipBiasSamplerDesc_, desc)) {
    return Result<SamplerHandle, std::string>::makeResult(
        taaMaterialMipBiasSampler_);
  }

  if (nuri::isValid(taaMaterialMipBiasSampler_)) {
    gpu_.destroySampler(taaMaterialMipBiasSampler_);
    taaMaterialMipBiasSampler_ = {};
  }

  auto samplerResult =
      gpu_.createSampler(desc, "taa_material_mip_bias_sampler");
  if (samplerResult.hasError()) {
    hasTaaMaterialMipBiasSamplerDesc_ = false;
    return samplerResult;
  }

  taaMaterialMipBiasSampler_ = samplerResult.value();
  taaMaterialMipBiasSamplerDesc_ = desc;
  hasTaaMaterialMipBiasSamplerDesc_ = true;
  return Result<SamplerHandle, std::string>::makeResult(
      taaMaterialMipBiasSampler_);
}

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

  auto bufferResult = ensureBufferRingCapacity(
      layout.totalBytes, std::max(1u, gpu_.getSwapchainImageCount()));
  if (bufferResult.hasError()) {
    return bufferResult;
  }
  Buffer *const sceneDataBuffer = currentBuffer(frame.frameIndex);
  if (sceneDataBuffer == nullptr || !sceneDataBuffer->valid()) {
    return Result<bool, std::string>::makeError(
        "SceneLightingProvider::prepare: scene data buffer is unavailable");
  }

  uint32_t cubemapTexId = kInvalidTextureBindlessIndex;
  uint32_t hasCubemap = 0u;
  uint32_t irradianceTexId = kInvalidTextureBindlessIndex;
  uint32_t prefilteredGgxTexId = kInvalidTextureBindlessIndex;
  uint32_t prefilteredCharlieTexId = kInvalidTextureBindlessIndex;
  uint32_t brdfLutTexId = kInvalidTextureBindlessIndex;
  uint32_t flags = 0u;
  const uint32_t cubemapSamplerId = gpu_.getCubemapSamplerBindlessIndex();
  const RenderSettings &renderSettings = renderSettingsOrDefault(frame);
  const uint32_t materialCoverageSamplerId = resolveDefaultMaterialSamplerId(
      gpu_, textureFilteringSettingsOrDefault(frame));
  auto materialSamplerIdResult = resolveMaterialSamplerId(frame);
  if (materialSamplerIdResult.hasError()) {
    return Result<bool, std::string>::makeError(
        materialSamplerIdResult.error());
  }
  const uint32_t materialSamplerId = materialSamplerIdResult.value();
  const EnvironmentHandles environment = frame.scene->environment();
  if (sanitizeAntiAliasingDebugView(renderSettings.antiAliasing.debug.view) ==
      AntiAliasingDebugView::TAATransmissionMipSource) {
    flags |= kForwardSceneTransmissionMipDebug;
  }

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
    // missing; keep kForwardSceneHasIblSheen set so downstream shading knows
    // a fallback sheen source is available.
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
  std::array<glm::uvec4, kSceneDepthPyramidArraySize> sceneDepthPyramidTexIds{};
  uint32_t ambientOcclusionTexId = kInvalidTextureBindlessIndex;
  uint32_t ambientOcclusionSamplerId = 0u;
  uint32_t ambientOcclusionFlags = 0u;
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
      const uint32_t packIndex = level / kSceneDepthPyramidTexIdPackWidth;
      const uint32_t componentIndex = level % kSceneDepthPyramidTexIdPackWidth;
      sceneDepthPyramidTexIds[packIndex][componentIndex] = texId;
      sceneDepthPyramidLevelCount = level + 1u;
    }
    if (sceneDepthPyramidLevelCount > 0u) {
      flags |= kForwardSceneHasSceneDepthPyramid;
    }
  }
  RenderSettings::AmbientOcclusionSettings ambientOcclusionSettings =
      renderSettings.ambientOcclusion;
  sanitizeAmbientOcclusionSettings(ambientOcclusionSettings,
                                   renderSettings.opaque,
                                   renderSettings.antiAliasing);
  if (ambientOcclusionSettings.active &&
      nuri::isValid(ctx.shared.ambientOcclusionTexture)) {
    ambientOcclusionTexId =
        gpu_.getTextureBindlessIndex(ctx.shared.ambientOcclusionTexture);
    ambientOcclusionSamplerId = gpu_.getDefaultSamplerBindlessIndex();
    if (ambientOcclusionTexId != kInvalidTextureBindlessIndex) {
      flags |= kForwardSceneHasAmbientOcclusion;
      ambientOcclusionFlags =
          kAmbientOcclusionFlagScalarAo |
          ((static_cast<uint32_t>(ambientOcclusionSettings.debugView) &
            kAmbientOcclusionDebugViewMask)
           << kAmbientOcclusionDebugViewShift);
    }
  }
  const uint64_t sceneDataBaseAddress =
      gpu_.getBufferDeviceAddress(sceneDataBuffer->handle());
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
  const size_t slotIndex =
      static_cast<size_t>(frame.frameIndex % sceneDataBuffers_.size());
  SlotUploadState &slotState = slotUploadStates_[slotIndex];
  const uint64_t sceneId = frame.scene->id();
  uint64_t shadowFrameBufferAddress =
      ctx.shared.shadowFrameGpuData.has_value()
          ? ctx.shared.shadowFrameGpuData->bufferAddress
          : 0u;
  if (shadowFrameBufferAddress == 0u) {
    auto disabledShadowResult = ensureDisabledShadowFrameBuffer();
    if (disabledShadowResult.hasError()) {
      return Result<bool, std::string>::makeError(disabledShadowResult.error());
    }
    shadowFrameBufferAddress = disabledShadowResult.value();
  }
  uint32_t shadowFlags = 0u;
  RenderSettings::ShadowSettings shadowSettings =
      renderSettingsOrDefault(frame).shadow;
  sanitizeShadowSettings(shadowSettings);
  if (shadowSettings.enabled && shadowFrameBufferAddress != 0u &&
      directionalLightCount > 0u) {
    shadowFlags =
        kShadowFrameFlagEnabled | shadowDebugFrameFlags(shadowSettings.debug);
  }

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
      .ambientOcclusionTexId = ambientOcclusionTexId,
      .ambientOcclusionSamplerId = ambientOcclusionSamplerId,
      .ambientOcclusionFlags = ambientOcclusionFlags,
      .ambientOcclusionReserved0 = 0u,
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
      .shadowFrameBufferAddress = shadowFrameBufferAddress,
      .shadowFlags = shadowFlags,
      .materialCoverageSamplerId = materialCoverageSamplerId,
  };
  ForwardSceneFrameData postTaaFrameData = frameData;
  postTaaFrameData.proj = cameraCurrentUnjitteredProjection(frame.camera);

  if (loggedAddressProbeTopologyVersion_ != frame.scene->topologyVersion() ||
      loggedLightStateSignature_ != frame.scene->lightTransformVersion()) {
    loggedAddressProbeTopologyVersion_ = frame.scene->topologyVersion();
    loggedLightStateSignature_ = frame.scene->lightTransformVersion();
    NURI_LOG_TRACE(
        "SceneLightingProvider::prepare probe: sceneData=0x%llx "
        "frameData=0x%llx postTaaFrameData=0x%llx "
        "dirLights=0x%llx localLights=0x%llx materialHeader=0x%llx "
        "materialClearcoat=0x%llx materialSheen=0x%llx "
        "materialTransmission=0x%llx materialSpecular=0x%llx shadow=0x%llx "
        "flags=0x%08x shadowFlags=0x%08x dirCount=%u localCount=%u",
        static_cast<unsigned long long>(sceneDataBaseAddress),
        static_cast<unsigned long long>(sceneDataBaseAddress +
                                        layout.frameDataOffset),
        static_cast<unsigned long long>(sceneDataBaseAddress +
                                        layout.postTaaFrameDataOffset),
        static_cast<unsigned long long>(directionalLightBufferAddress),
        static_cast<unsigned long long>(localLightBufferAddress),
        static_cast<unsigned long long>(frameData.materialHeaderBufferAddress),
        static_cast<unsigned long long>(
            frameData.materialClearcoatBufferAddress),
        static_cast<unsigned long long>(frameData.materialSheenBufferAddress),
        static_cast<unsigned long long>(
            frameData.materialTransmissionBufferAddress),
        static_cast<unsigned long long>(
            frameData.materialSpecularBufferAddress),
        static_cast<unsigned long long>(frameData.shadowFrameBufferAddress),
        frameData.flags, frameData.shadowFlags, frameData.directionalLightCount,
        frameData.localLightCount);
  }

  if (!slotState.hasFrameData || slotState.frameData != frameData) {
    const std::span<const std::byte> frameDataBytes{
        reinterpret_cast<const std::byte *>(&frameData), sizeof(frameData)};
    auto updateResult = gpu_.updateBuffer(
        sceneDataBuffer->handle(), frameDataBytes, layout.frameDataOffset);
    if (updateResult.hasError()) {
      return updateResult;
    }
  }
  if (!slotState.hasFrameData ||
      slotState.postTaaFrameData != postTaaFrameData) {
    const std::span<const std::byte> frameDataBytes{
        reinterpret_cast<const std::byte *>(&postTaaFrameData),
        sizeof(postTaaFrameData)};
    auto updateResult =
        gpu_.updateBuffer(sceneDataBuffer->handle(), frameDataBytes,
                          layout.postTaaFrameDataOffset);
    if (updateResult.hasError()) {
      return updateResult;
    }
  }

  const bool lightDataDirty =
      slotState.scene != frame.scene || slotState.sceneId != sceneId ||
      slotState.lightTopologyVersion != frame.scene->lightTopologyVersion() ||
      slotState.lightTransformVersion != frame.scene->lightTransformVersion() ||
      slotState.directionalLightCount != directionalLightCount ||
      slotState.localLightCount != localLightCount;

  if (lightDataDirty && !directionalLights.empty()) {
    const std::span<const std::byte> directionalLightBytes{
        reinterpret_cast<const std::byte *>(directionalLights.data()),
        directionalLights.size() * sizeof(DirectionalLightGpuData)};
    auto lightUpdateResult =
        gpu_.updateBuffer(sceneDataBuffer->handle(), directionalLightBytes,
                          layout.directionalLightsOffset);
    if (lightUpdateResult.hasError()) {
      return lightUpdateResult;
    }
  }
  if (lightDataDirty && !localLights.empty()) {
    const std::span<const std::byte> localLightBytes{
        reinterpret_cast<const std::byte *>(localLights.data()),
        localLights.size() * sizeof(LocalLightGpuData)};
    auto lightUpdateResult = gpu_.updateBuffer(
        sceneDataBuffer->handle(), localLightBytes, layout.localLightsOffset);
    if (lightUpdateResult.hasError()) {
      return lightUpdateResult;
    }
  }

  slotState.scene = frame.scene;
  slotState.sceneId = sceneId;
  slotState.lightTopologyVersion = frame.scene->lightTopologyVersion();
  slotState.lightTransformVersion = frame.scene->lightTransformVersion();
  slotState.directionalLightCount = directionalLightCount;
  slotState.localLightCount = localLightCount;
  slotState.hasFrameData = true;
  slotState.frameData = frameData;
  slotState.postTaaFrameData = postTaaFrameData;

  ctx.shared.forwardSceneGpuData = ForwardSceneGpuData{
      .buffer = sceneDataBuffer->handle(),
      .frameDataAddress = sceneDataBaseAddress + layout.frameDataOffset,
      .postTaaFrameDataAddress =
          sceneDataBaseAddress + layout.postTaaFrameDataOffset,
      .directionalLightBufferAddress = directionalLightBufferAddress,
      .localLightBufferAddress = localLightBufferAddress,
      .shadowFrameBufferAddress = shadowFrameBufferAddress,
      .directionalLightCount = directionalLightCount,
      .localLightCount = localLightCount,
      .shadowFlags = shadowFlags,
  };

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
SceneLightingProvider::ensureBufferRingCapacity(size_t requiredBytes,
                                                uint32_t requiredCount) {
  const size_t requested =
      std::max(requiredBytes, sizeof(ForwardSceneFrameData));
  const uint32_t safeCount = std::max(requiredCount, 1u);
  const bool countsMatch = sceneDataBuffers_.size() == safeCount;
  bool allValid = countsMatch;
  if (countsMatch) {
    for (const std::unique_ptr<Buffer> &buffer : sceneDataBuffers_) {
      if (buffer == nullptr || !buffer->valid()) {
        allValid = false;
        break;
      }
    }
  }
  if (allValid && sceneDataBufferCapacityBytes_ >= requested) {
    if (slotUploadStates_.size() != safeCount) {
      slotUploadStates_.assign(safeCount, SlotUploadState{});
    }
    return Result<bool, std::string>::makeResult(true);
  }
  if (!sceneDataBuffers_.empty()) {
    gpu_.waitIdle();
  }
  destroyBuffers();
  sceneDataBuffers_.reserve(safeCount);
  slotUploadStates_.assign(safeCount, SlotUploadState{});
  for (uint32_t i = 0u; i < safeCount; ++i) {
    auto bufferResult = Buffer::create(gpu_,
                                       BufferDesc{.usage = BufferUsage::Storage,
                                                  .storage = Storage::Device,
                                                  .size = requested},
                                       "forward_scene_data");
    if (bufferResult.hasError()) {
      destroyBuffers();
      return Result<bool, std::string>::makeError(bufferResult.error());
    }
    sceneDataBuffers_.push_back(std::move(bufferResult.value()));
  }
  sceneDataBufferCapacityBytes_ = requested;
  return Result<bool, std::string>::makeResult(true);
}

Result<uint64_t, std::string>
SceneLightingProvider::ensureDisabledShadowFrameBuffer() {
  if (disabledShadowFrameBuffer_ == nullptr ||
      !disabledShadowFrameBuffer_->valid()) {
    auto bufferResult =
        Buffer::create(gpu_,
                       BufferDesc{.usage = BufferUsage::Storage,
                                  .storage = Storage::Device,
                                  .size = sizeof(ShadowFrameGpuData)},
                       "forward_scene_disabled_shadow_frame");
    if (bufferResult.hasError()) {
      return Result<uint64_t, std::string>::makeError(bufferResult.error());
    }
    disabledShadowFrameBuffer_ = std::move(bufferResult.value());

    const ShadowFrameGpuData disabledShadow{};
    const std::span<const std::byte> bytes{
        reinterpret_cast<const std::byte *>(&disabledShadow),
        sizeof(disabledShadow)};
    auto updateResult =
        gpu_.updateBuffer(disabledShadowFrameBuffer_->handle(), bytes, 0u);
    if (updateResult.hasError()) {
      gpu_.destroyBuffer(disabledShadowFrameBuffer_->handle());
      disabledShadowFrameBuffer_.reset();
      return Result<uint64_t, std::string>::makeError(updateResult.error());
    }
  }

  const uint64_t address =
      gpu_.getBufferDeviceAddress(disabledShadowFrameBuffer_->handle());
  if (address == 0u) {
    return Result<uint64_t, std::string>::makeError(
        "SceneLightingProvider::prepare: invalid disabled shadow frame buffer "
        "address");
  }
  return Result<uint64_t, std::string>::makeResult(address);
}

void SceneLightingProvider::destroyBuffers() {
  for (std::unique_ptr<Buffer> &buffer : sceneDataBuffers_) {
    if (buffer && buffer->valid()) {
      gpu_.destroyBuffer(buffer->handle());
    }
    buffer.reset();
  }
  sceneDataBuffers_.clear();
  slotUploadStates_.clear();
  if (disabledShadowFrameBuffer_ && disabledShadowFrameBuffer_->valid()) {
    gpu_.destroyBuffer(disabledShadowFrameBuffer_->handle());
  }
  disabledShadowFrameBuffer_.reset();
  sceneDataBufferCapacityBytes_ = 0;
}

void SceneLightingProvider::destroyCachedSamplers() {
  if (nuri::isValid(taaMaterialMipBiasSampler_)) {
    gpu_.destroySampler(taaMaterialMipBiasSampler_);
  }
  taaMaterialMipBiasSampler_ = {};
  taaMaterialMipBiasSamplerDesc_ = {};
  hasTaaMaterialMipBiasSamplerDesc_ = false;
}

Buffer *
SceneLightingProvider::currentBuffer(uint64_t frameIndex) const noexcept {
  if (sceneDataBuffers_.empty()) {
    return nullptr;
  }
  const size_t slot =
      static_cast<size_t>(frameIndex % sceneDataBuffers_.size());
  return sceneDataBuffers_[slot].get();
}

} // namespace nuri
