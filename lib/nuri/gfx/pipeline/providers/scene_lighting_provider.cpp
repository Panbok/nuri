#include "nuri/gfx/pipeline/providers/scene_lighting_provider.h"
#include "nuri/core/profiling.h"
#include "nuri/math/utils.h"
#include "nuri/pch.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/scene/render_scene.h"
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
enum ForwardSceneDepthPyramidFlags : uint32_t {
  kForwardSceneDepthPyramidPreviousFrame = 1u << 0u,
};
struct SceneDataBufferLayout {
  size_t postTaaFrameDataOffset = 0u;
  size_t directionalLightsOffset = 0u;
  size_t localLightsOffset = 0u;
  size_t totalBytes = 0u;
};
[[nodiscard]] SceneDataBufferLayout
makeSceneDataBufferLayout(size_t directionalLightBytes,
                          size_t localLightBytes) {
  constexpr size_t frameDataBytes = sizeof(ForwardSceneFrameData);
  const size_t postTaaFrameDataOffset =
      alignUp(frameDataBytes, alignof(ForwardSceneFrameData));
  const size_t directionalOffset =
      alignUp(postTaaFrameDataOffset + frameDataBytes,
              alignof(DirectionalLightGpuData));
  const size_t localOffset = alignUp(directionalOffset + directionalLightBytes,
                                     alignof(LocalLightGpuData));
  return SceneDataBufferLayout{
      .postTaaFrameDataOffset = postTaaFrameDataOffset,
      .directionalLightsOffset = directionalOffset,
      .localLightsOffset = localOffset,
      .totalBytes = std::max(localOffset + localLightBytes,
                             postTaaFrameDataOffset + frameDataBytes),
  };
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
} // namespace

SceneLightingProvider::SceneLightingProvider(GPUDevice &gpu) : gpu_(gpu) {}

SceneLightingProvider::~SceneLightingProvider() {
  if (nuri::isValid(taaMaterialMipBiasSampler_)) {
    gpu_.destroySampler(taaMaterialMipBiasSampler_);
  }
}

Result<uint32_t, std::string>
SceneLightingProvider::resolveMaterialSamplerId(RenderFrameContext &frame) {
  const RenderSettings &settings = renderSettingsOrDefault(frame);
  const RenderSettings::TextureFilteringSettings &filtering =
      settings.textureFiltering;
  const TextureFilterMode filterMode =
      sanitizeTextureFilterMode(filtering.mode);
  const bool mipFilteringActive = filterMode != TextureFilterMode::Bilinear;
  const bool taaMode = sanitizeAntiAliasingMode(settings.antiAliasing.mode) ==
                       AntiAliasingMode::TAA;
  const RenderSettings::AntiAliasingDebugSettings aaDebug =
      effectiveTemporalAADebugSettings(settings.antiAliasing);
  const float mipBias = sanitizeTaaMaterialMipBias(aaDebug.taaMaterialMipBias);
  AntiAliasingFrameMetrics &aaMetrics = frame.metrics.antiAliasing;
  aaMetrics.taaMaterialMipBiasEnabled = aaDebug.taaMaterialMipBiasEnabled;
  aaMetrics.taaMaterialMipBias = mipBias;
  aaMetrics.taaMaterialMipBiasApplied = false;
  if (!taaMode || !aaDebug.taaMaterialMipBiasEnabled || !mipFilteringActive ||
      mipBias >= 0.0f) {
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
  if (!nuri::isValid(taaMaterialMipBiasSampler_) ||
      taaMaterialMipBiasSamplerDesc_ != samplerDesc) {
    if (nuri::isValid(taaMaterialMipBiasSampler_)) {
      gpu_.destroySampler(taaMaterialMipBiasSampler_);
      taaMaterialMipBiasSampler_ = {};
    }
    auto result = gpu_.createSampler(samplerDesc, "taa_material_mip_bias");
    if (result.hasError()) {
      return Result<uint32_t, std::string>::makeError(result.error());
    }
    taaMaterialMipBiasSampler_ = result.value();
    taaMaterialMipBiasSamplerDesc_ = samplerDesc;
  }
  aaMetrics.taaMaterialMipBiasApplied = true;
  return Result<uint32_t, std::string>::makeResult(
      gpu_.getSamplerBindlessIndex(taaMaterialMipBiasSampler_));
}

Result<bool, std::string>
SceneLightingProvider::prepare(FrameBuildContext &ctx) {
  NURI_PROFILER_FUNCTION();
  RenderFrameContext &frame = ctx.frame;
  if (frame.scene == nullptr) {
    return Result<bool, std::string>::makeError(
        "SceneLightingProvider::prepare: frame scene is null");
  }
  const MaterialTableGpuData &materialTable = *ctx.shared.materialTableGpuData;
  const std::span<const DirectionalLightGpuData> directionalLights =
      frame.scene->packedDirectionalLights();
  const std::span<const LocalLightGpuData> localLights =
      frame.scene->packedLocalLights();
  const SceneDataBufferLayout layout = makeSceneDataBufferLayout(
      directionalLights.size() * sizeof(DirectionalLightGpuData),
      localLights.size() * sizeof(LocalLightGpuData));
  auto bufferResult = ensureBufferRingCapacity(
      layout.totalBytes, std::max(1u, gpu_.getSwapchainImageCount()));
  if (bufferResult.hasError()) {
    return bufferResult;
  }
  Slot &slot = slots_[static_cast<size_t>(frame.frameIndex % slots_.size())];
  Buffer &sceneDataBuffer = *slot.buffer.buffer;
  uint32_t cubemapTexId = kInvalidTextureBindlessIndex;
  uint32_t hasCubemap = 0u;
  uint32_t irradianceTexId = kInvalidTextureBindlessIndex;
  uint32_t prefilteredGgxTexId = kInvalidTextureBindlessIndex;
  uint32_t prefilteredCharlieTexId = kInvalidTextureBindlessIndex;
  uint32_t brdfLutTexId = kInvalidTextureBindlessIndex;
  uint32_t flags = 0u;
  const uint32_t cubemapSamplerId = gpu_.getCubemapSamplerBindlessIndex();
  const RenderSettings &renderSettings = renderSettingsOrDefault(frame);
  const uint32_t materialCoverageSamplerId =
      resolveDefaultMaterialSamplerId(gpu_, renderSettings.textureFiltering);
  const uint32_t materialDataSamplerId = materialCoverageSamplerId;
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
  const std::array environmentRefs{
      environment.cubemap, environment.irradiance, environment.prefilteredGgx,
      environment.prefilteredCharlie, environment.brdfLut};
  const std::array environmentIds{&cubemapTexId, &irradianceTexId,
                                  &prefilteredGgxTexId,
                                  &prefilteredCharlieTexId, &brdfLutTexId};
  constexpr std::array<uint32_t, 5u> environmentFlags{
      0u, kForwardSceneHasIblDiffuse, kForwardSceneHasIblSpecular,
      kForwardSceneHasIblSheen, kForwardSceneHasBrdfLut};
  for (size_t i = 0; i < environmentRefs.size(); ++i) {
    const TextureRecord *record = ctx.resources.tryGet(environmentRefs[i]);
    if (record != nullptr &&
        record->bindlessIndex != kInvalidTextureBindlessIndex) {
      *environmentIds[i] = record->bindlessIndex;
      flags |= environmentFlags[i];
    }
  }
  hasCubemap = cubemapTexId != kInvalidTextureBindlessIndex;
  if (prefilteredCharlieTexId == kInvalidTextureBindlessIndex &&
      prefilteredGgxTexId != kInvalidTextureBindlessIndex) {
    prefilteredCharlieTexId = prefilteredGgxTexId;
    flags |= kForwardSceneHasIblSheen;
  }
  uint32_t sceneColorTexId = kInvalidTextureBindlessIndex;
  uint32_t sceneColorHalfResTexId = kInvalidTextureBindlessIndex;
  uint32_t sceneColorQuarterResTexId = kInvalidTextureBindlessIndex;
  uint32_t sceneColorSamplerId = 0u;
  uint32_t sceneDepthTexId = kInvalidTextureBindlessIndex;
  uint32_t sceneDepthSamplerId = ctx.shared.sceneDepthSamplerId;
  uint32_t sceneDepthPyramidLevelCount = 0u;
  std::array<glm::uvec4, kSceneDepthPyramidArraySize> sceneDepthPyramidTexIds{};
  glm::uvec4 sceneDepthPyramidInfo{0u};
  uint32_t ambientOcclusionTexId = kInvalidTextureBindlessIndex;
  uint32_t ambientOcclusionSamplerId = 0u;
  uint32_t ambientOcclusionFlags = 0u;
  const auto textureIndex = [this](TextureHandle texture) {
    return nuri::isValid(texture) ? gpu_.getTextureBindlessIndex(texture)
                                  : kInvalidTextureBindlessIndex;
  };
  sceneColorTexId = textureIndex(ctx.shared.sceneColorTexture);
  sceneColorHalfResTexId = textureIndex(ctx.shared.sceneColorHalfResTexture);
  sceneColorQuarterResTexId =
      textureIndex(ctx.shared.sceneColorQuarterResTexture);
  sceneDepthTexId = textureIndex(ctx.shared.sceneDepthTexture);
  if (sceneColorTexId != kInvalidTextureBindlessIndex) {
    sceneColorSamplerId = gpu_.getLinearRepeatSamplerBindlessIndex(true, 1u);
    flags |= kForwardSceneHasSceneColor;
  }
  if (sceneDepthTexId != kInvalidTextureBindlessIndex) {
    flags |= kForwardSceneHasSceneDepth;
  }
  if ((flags & kForwardSceneHasSceneDepth) != 0u) {
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
      const TextureDimensions dimensions =
          gpu_.getTextureDimensions(ctx.shared.sceneDepthPyramidTextures[0]);
      uint32_t depthPyramidFlags = 0u;
      if (ctx.shared.sceneDepthPyramidSourceFrameIndex.has_value() &&
          *ctx.shared.sceneDepthPyramidSourceFrameIndex + 1u ==
              frame.frameIndex) {
        depthPyramidFlags |= kForwardSceneDepthPyramidPreviousFrame;
      }
      sceneDepthPyramidInfo = glm::uvec4(
          std::max(dimensions.width, 1u), std::max(dimensions.height, 1u),
          sceneDepthPyramidLevelCount, depthPyramidFlags);
    }
  }
  RenderSettings::AmbientOcclusionSettings ambientOcclusionSettings =
      renderSettings.ambientOcclusion;
  sanitizeAmbientOcclusionSettings(ambientOcclusionSettings,
                                   renderSettings.opaque,
                                   renderSettings.antiAliasing);
  if (ambientOcclusionSettings.active) {
    ambientOcclusionTexId = textureIndex(ctx.shared.ambientOcclusionTexture);
    if (ambientOcclusionTexId != kInvalidTextureBindlessIndex) {
      ambientOcclusionSamplerId = gpu_.getDefaultSamplerBindlessIndex();
      flags |= kForwardSceneHasAmbientOcclusion;
      ambientOcclusionFlags =
          kAmbientOcclusionFlagScalarAo |
          ((static_cast<uint32_t>(ambientOcclusionSettings.debugView) &
            kAmbientOcclusionDebugViewMask)
           << kAmbientOcclusionDebugViewShift);
    }
  }
  const uint64_t sceneDataBaseAddress =
      gpu_.getBufferDeviceAddress(sceneDataBuffer.handle());
  const uint64_t directionalLightBufferAddress =
      directionalLights.empty()
          ? 0u
          : sceneDataBaseAddress + layout.directionalLightsOffset;
  const uint64_t localLightBufferAddress =
      localLights.empty() ? 0u
                          : sceneDataBaseAddress + layout.localLightsOffset;
  const uint32_t directionalLightCount =
      static_cast<uint32_t>(directionalLights.size());
  const uint32_t localLightCount = static_cast<uint32_t>(localLights.size());
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
  if (shadowSettings.enabled && directionalLightCount > 0u) {
    shadowFlags =
        kShadowFrameFlagEnabled | shadowDebugFrameFlags(shadowSettings.debug);
  }
  const DDGIFrameGpuDataHandle *ddgiFrame =
      ctx.shared.ddgiFrameGpuData.has_value() ? &*ctx.shared.ddgiFrameGpuData
                                              : nullptr;
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
      .materialHeaderBufferAddress = materialTable.headerBufferAddress,
      .materialClearcoatBufferAddress = materialTable.clearcoatBufferAddress,
      .materialSheenBufferAddress = materialTable.sheenBufferAddress,
      .materialTransmissionBufferAddress =
          materialTable.transmissionBufferAddress,
      .materialSpecularBufferAddress = materialTable.specularBufferAddress,
      .directionalLightCount = directionalLightCount,
      .localLightCount = localLightCount,
      .shadowFrameBufferAddress = shadowFrameBufferAddress,
      .shadowFlags = shadowFlags,
      .materialCoverageSamplerId = materialCoverageSamplerId,
      .materialDataSamplerId = materialDataSamplerId,
      .ddgiFrameBufferAddress =
          ddgiFrame != nullptr ? ddgiFrame->bufferAddress : 0u,
      .ddgiFlags = ddgiFrame != nullptr ? ddgiFrame->flags : 0u,
      .ddgiDebugView = ddgiFrame != nullptr
                           ? static_cast<uint32_t>(ddgiFrame->debugView)
                           : 0u,
      .previousViewProj = ctx.shared.sceneDepthPyramidSourceViewProj.value_or(
          frame.camera.currentUnjitteredViewProj),
      .sceneDepthPyramidInfo = sceneDepthPyramidInfo,
  };
  ForwardSceneFrameData postTaaFrameData = frameData;
  postTaaFrameData.proj = cameraCurrentUnjitteredProjection(frame.camera);
  const std::array<const ForwardSceneFrameData *, 2u> frameValues{
      &frameData, &postTaaFrameData};
  const std::array frameCache{&slot.frameData, &slot.postTaaFrameData};
  const std::array<size_t, 2u> frameOffsets{0u, layout.postTaaFrameDataOffset};
  for (size_t i = 0; i < frameValues.size(); ++i) {
    if (!slot.hasFrameData || *frameCache[i] != *frameValues[i]) {
      auto result = gpu_.updateBuffer(
          sceneDataBuffer.handle(),
          std::as_bytes(std::span{frameValues[i], 1u}), frameOffsets[i]);
      if (result.hasError()) {
        return result;
      }
    }
  }
  const bool lightDataDirty =
      !slot.hasFrameData || slot.scene != frame.scene ||
      slot.sceneId != sceneId ||
      slot.lightTopologyVersion != frame.scene->lightTopologyVersion() ||
      slot.lightTransformVersion != frame.scene->lightTransformVersion() ||
      slot.directionalLightCount != directionalLightCount ||
      slot.localLightCount != localLightCount;
  if (lightDataDirty) {
    const std::array lightBytes{std::as_bytes(directionalLights),
                                std::as_bytes(localLights)};
    const std::array lightOffsets{layout.directionalLightsOffset,
                                  layout.localLightsOffset};
    for (size_t i = 0; i < lightBytes.size(); ++i) {
      if (lightBytes[i].empty()) {
        continue;
      }
      auto result = gpu_.updateBuffer(sceneDataBuffer.handle(), lightBytes[i],
                                      lightOffsets[i]);
      if (result.hasError()) {
        return result;
      }
    }
  }
  slot.scene = frame.scene;
  slot.sceneId = sceneId;
  slot.lightTopologyVersion = frame.scene->lightTopologyVersion();
  slot.lightTransformVersion = frame.scene->lightTransformVersion();
  slot.directionalLightCount = directionalLightCount;
  slot.localLightCount = localLightCount;
  slot.hasFrameData = true;
  slot.frameData = frameData;
  slot.postTaaFrameData = postTaaFrameData;
  ctx.shared.forwardSceneGpuData = ForwardSceneGpuData{
      .buffer = sceneDataBuffer.handle(),
      .frameData = frameData,
      .postTaaFrameData = postTaaFrameData,
      .frameDataAddress = sceneDataBaseAddress,
      .postTaaFrameDataAddress =
          sceneDataBaseAddress + layout.postTaaFrameDataOffset,
      .directionalLightBufferAddress = directionalLightBufferAddress,
      .localLightBufferAddress = localLightBufferAddress,
      .shadowFrameBufferAddress = shadowFrameBufferAddress,
      .directionalLightCount = directionalLightCount,
      .localLightCount = localLightCount,
      .shadowFlags = shadowFlags,
      .indirectDependencyBuffers = ddgiFrame != nullptr
                                       ? ddgiFrame->dependencyBuffers
                                       : std::span<const BufferHandle>{},
      .indirectDependencyTextures = ddgiFrame != nullptr
                                        ? ddgiFrame->dependencyTextures
                                        : std::span<const TextureHandle>{},
  };
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
SceneLightingProvider::ensureBufferRingCapacity(size_t requiredBytes,
                                                uint32_t requiredCount) {
  slots_.resize(requiredCount);
  for (Slot &slot : slots_) {
    auto result =
        ensureDynamicBufferCapacity(gpu_, slot.buffer,
                                    BufferDesc{.usage = BufferUsage::Storage,
                                               .storage = Storage::HostVisible,
                                               .size = requiredBytes},
                                    "forward_scene_data");
    if (result.hasError()) {
      return Result<bool, std::string>::makeError(result.error());
    }
    if (result.value()) {
      slot.hasFrameData = false;
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<uint64_t, std::string>
SceneLightingProvider::ensureDisabledShadowFrameBuffer() {
  if (!disabledShadowFrameBuffer_) {
    auto result = Buffer::create(gpu_,
                                 BufferDesc{.usage = BufferUsage::Storage,
                                            .storage = Storage::HostVisible,
                                            .size = sizeof(ShadowFrameGpuData)},
                                 "forward_scene_disabled_shadow_frame");
    if (result.hasError()) {
      return Result<uint64_t, std::string>::makeError(result.error());
    }
    std::unique_ptr<Buffer> buffer = std::move(result.value());
    const ShadowFrameGpuData disabledShadow{};
    auto update = gpu_.updateBuffer(
        buffer->handle(), std::as_bytes(std::span{&disabledShadow, 1u}), 0u);
    if (update.hasError()) {
      return Result<uint64_t, std::string>::makeError(update.error());
    }
    disabledShadowFrameBuffer_ = std::move(buffer);
  }
  return Result<uint64_t, std::string>::makeResult(
      gpu_.getBufferDeviceAddress(disabledShadowFrameBuffer_->handle()));
}

} // namespace nuri
