#include "nuri/pch.h"

#include "nuri/gfx/layers/scene_lighting_layer.h"

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

} // namespace

SceneLightingLayer::SceneLightingLayer(GPUDevice &gpu,
                                       std::pmr::memory_resource *memory)
    : gpu_(gpu) {}

SceneLightingLayer::~SceneLightingLayer() { onDetach(); }

void SceneLightingLayer::onDetach() { destroyBuffer(); }

Result<bool, std::string>
SceneLightingLayer::buildRenderGraph(RenderFrameContext &frame,
                                     RenderGraphBuilder &) {
  NURI_PROFILER_FUNCTION();
  if (frame.scene == nullptr) {
    return Result<bool, std::string>::makeError(
        "SceneLightingLayer::buildRenderGraph: frame scene is null");
  }
  if (frame.resources == nullptr) {
    return Result<bool, std::string>::makeError(
        "SceneLightingLayer::buildRenderGraph: frame resources are null");
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
  const EnvironmentHandles environment = frame.scene->environment();

  if (const TextureRecord *cubemap =
          frame.resources->tryGet(environment.cubemap);
      cubemap != nullptr && nuri::isValid(cubemap->texture)) {
    cubemapTexId = cubemap->bindlessIndex;
    hasCubemap = 1u;
  }
  if (const TextureRecord *irradiance =
          frame.resources->tryGet(environment.irradiance);
      irradiance != nullptr && nuri::isValid(irradiance->texture)) {
    irradianceTexId = irradiance->bindlessIndex;
    flags |= kForwardSceneHasIblDiffuse;
  }
  if (const TextureRecord *prefilteredGgx =
          frame.resources->tryGet(environment.prefilteredGgx);
      prefilteredGgx != nullptr && nuri::isValid(prefilteredGgx->texture)) {
    prefilteredGgxTexId = prefilteredGgx->bindlessIndex;
    flags |= kForwardSceneHasIblSpecular;
  }
  if (const TextureRecord *prefilteredCharlie =
          frame.resources->tryGet(environment.prefilteredCharlie);
      prefilteredCharlie != nullptr &&
      nuri::isValid(prefilteredCharlie->texture)) {
    prefilteredCharlieTexId = prefilteredCharlie->bindlessIndex;
    flags |= kForwardSceneHasIblSheen;
  } else if ((flags & kForwardSceneHasIblSpecular) != 0u) {
    prefilteredCharlieTexId = prefilteredGgxTexId;
    flags |= kForwardSceneHasIblSheen;
  }
  if (const TextureRecord *brdfLut =
          frame.resources->tryGet(environment.brdfLut);
      brdfLut != nullptr && nuri::isValid(brdfLut->texture)) {
    brdfLutTexId = brdfLut->bindlessIndex;
    flags |= kForwardSceneHasBrdfLut;
  }

  uint32_t sceneColorTexId = 0u;
  uint32_t sceneColorHalfResTexId = 0u;
  uint32_t sceneColorQuarterResTexId = 0u;
  uint32_t sceneColorSamplerId = 0u;
  if (const TextureHandle *sceneColorTexture =
          frame.channels.tryGet<TextureHandle>(kFrameChannelSceneColorTexture);
      sceneColorTexture != nullptr && nuri::isValid(*sceneColorTexture)) {
    sceneColorTexId = gpu_.getTextureBindlessIndex(*sceneColorTexture);
    sceneColorSamplerId = gpu_.getDefaultSamplerBindlessIndex();
    if (sceneColorTexId != kInvalidTextureBindlessIndex) {
      flags |= kForwardSceneHasSceneColor;
    }
  }
  if (const TextureHandle *sceneColorHalfResTexture =
          frame.channels.tryGet<TextureHandle>(
              kFrameChannelSceneColorHalfResTexture);
      sceneColorHalfResTexture != nullptr &&
      nuri::isValid(*sceneColorHalfResTexture)) {
    sceneColorHalfResTexId =
        gpu_.getTextureBindlessIndex(*sceneColorHalfResTexture);
  }
  if (const TextureHandle *sceneColorQuarterResTexture =
          frame.channels.tryGet<TextureHandle>(
              kFrameChannelSceneColorQuarterResTexture);
      sceneColorQuarterResTexture != nullptr &&
      nuri::isValid(*sceneColorQuarterResTexture)) {
    sceneColorQuarterResTexId =
        gpu_.getTextureBindlessIndex(*sceneColorQuarterResTexture);
  }
  if (gpu_.getSwapchainFormat() != Format::RGBA8_SRGB) {
    flags |= kForwardSceneOutputLinearToSrgb;
  }

  const uint64_t sceneDataBaseAddress =
      gpu_.getBufferDeviceAddress(sceneDataBuffer_->handle());
  if (sceneDataBaseAddress == 0u) {
    return Result<bool, std::string>::makeError(
        "SceneLightingLayer::buildRenderGraph: invalid scene data buffer "
        "address");
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
      .sceneColorTexId = sceneColorTexId,
      .sceneColorSamplerId = sceneColorSamplerId,
      .sceneColorHalfResTexId = sceneColorHalfResTexId,
      .sceneColorQuarterResTexId = sceneColorQuarterResTexId,
      .directionalLightBufferAddress = directionalLightBufferAddress,
      .localLightBufferAddress = localLightBufferAddress,
      .directionalLightCount = directionalLightCount,
      .localLightCount = localLightCount,
  };

  const bool frameDataDirty =
      !frameDataUploadValid_ ||
      std::memcmp(&uploadedFrameData_, &frameData, sizeof(frameData)) != 0;
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

  frame.channels.publish<ForwardSceneGpuData>(
      kFrameChannelForwardSceneGpuData,
      ForwardSceneGpuData{
          .buffer = sceneDataBuffer_->handle(),
          .frameDataAddress = sceneDataBaseAddress + layout.frameDataOffset,
          .directionalLightBufferAddress = directionalLightBufferAddress,
          .localLightBufferAddress = localLightBufferAddress,
          .directionalLightCount = directionalLightCount,
          .localLightCount = localLightCount,
      });

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
SceneLightingLayer::ensureBufferCapacity(size_t requiredBytes) {
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

void SceneLightingLayer::destroyBuffer() {
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
