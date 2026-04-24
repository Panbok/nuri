#include "nuri/pch.h"

#include "nuri/gfx/renderers/shadow_renderer.h"

#include "nuri/core/log.h"
#include "nuri/gfx/renderers/detail/renderable_material_resolution.h"
#include "nuri/gfx/renderers/detail/shadow_math.h"
#include "nuri/resources/gpu/resource_manager.h"

#include <chrono>

namespace nuri {
namespace {

constexpr uint32_t kShadowPassDebugColor = 0xff5a7dffu;
constexpr uint32_t kShadowMeshDebugColor = 0xff5aff9du;
constexpr uint32_t kShadowPreviewPassDebugColor = 0xff7d5affu;
constexpr std::string_view kShadowPassLabel = "ShadowDepthPass";
constexpr std::string_view kShadowMeshLabel = "ShadowCasterMesh";
constexpr std::string_view kShadowPreviewPassLabel = "ShadowDepthPreviewPass";
constexpr std::string_view kShadowPreviewDrawLabel = "ShadowDepthPreview";
constexpr uint64_t kFnvOffsetBasis64 = 14695981039346656037ull;
constexpr uint64_t kFnvPrime64 = 1099511628211ull;
constexpr uint32_t kShadowPreviewFlagInvert = 1u << 0u;
constexpr uint32_t kShadowPreviewFlagLog = 1u << 1u;
constexpr uint32_t kShadowPreviewFlagTiled = 1u << 2u;
constexpr float kCullingPaddingTexelMultiplier = 2.0f;
constexpr float kStaticOnlyGuardBandShadowMapFraction = 5.0f / 8.0f;
constexpr float kStaticOnlyGuardBandMaxTexels = 5120.0f;
constexpr float kStaticOnlyGuardBandMinTexels = 16.0f;
constexpr float kStaticOnlyDepthGuardBandShadowMapFraction = 7.0f / 16.0f;
constexpr float kStaticOnlyDepthGuardBandMaxTexels = 3584.0f;
constexpr float kStaticOnlyDepthGuardBandMinTexels = 8.0f;
constexpr float kStaticOnlyAdaptiveGuardBandMotionMultiplier = 2.25f;
constexpr float kStaticOnlyAdaptiveGuardBandMaxShadowMapFraction = 2.0f;
constexpr float kStaticOnlyAdaptiveDepthGuardBandMaxShadowMapFraction = 2.0f;
constexpr float kStaticOnlyPredictiveCenterMotionMultiplier = 1.0f;
constexpr float kStaticOnlyPredictiveExtentGrowthFraction = 1.0f / 3.0f;
constexpr float kStaticOnlyPredictiveTrailingGuardFraction = 3.0f / 8.0f;
constexpr float kStaticOnlyPredictiveMotionEpsilonTexels = 0.5f;
constexpr uint32_t kSdsmHistogramSourceMaxTexelCount = 4096u;
constexpr uint64_t kSdsmDiagnosticRefreshFrames = 120u;
constexpr float kSdsmHistogramClearDepthEpsilon = 1.0e-4f;
constexpr uint32_t kMinSdsmReduceResultRingCount = 16u;
constexpr uint32_t kSdsmGpuWarmupGraceMissFrames = 2u;

[[nodiscard]] float
elapsedMilliseconds(std::chrono::steady_clock::time_point begin,
                    std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<float, std::milli>(end - begin).count();
}
constexpr std::string_view kShadowPreviewVS = R"(
#version 460

vec2 fullscreenTriangleUv(uint vertexIndex) {
  return vec2((vertexIndex << 1u) & 2u, vertexIndex & 2u);
}

void main() {
  vec2 clipUv = fullscreenTriangleUv(uint(gl_VertexIndex));
  gl_Position = vec4(clipUv * 2.0 - 1.0, 0.0, 1.0);
}
)";
constexpr std::string_view kShadowPreviewFS = R"(
#version 460

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_samplerless_texture_functions : require

layout(set = 0, binding = 0) uniform texture2D kTextures2D[];
layout(location = 0) out vec4 out_FragColor;

const uint FLAG_INVERT = 1u;
const uint FLAG_LOG_SCALE = 2u;
const uint FLAG_TILED = 4u;
const uint MAX_PREVIEW_SOURCES = 4u;

layout(push_constant) uniform PreviewPushConstants {
  uvec4 sourceTexIds;
  uvec4 previewParams;
  vec4 depthParams;
} pc;

float fetchDepth(uint sourceTexId, vec2 tileUv) {
  ivec2 sourceSize = textureSize(nonuniformEXT(kTextures2D[sourceTexId]), 0);
  vec2 clampedUv = clamp(tileUv, vec2(0.0), vec2(0.99999994));
  ivec2 texelCoord =
      clamp(ivec2(clampedUv * vec2(sourceSize)), ivec2(0), sourceSize - 1);
  return texelFetch(nonuniformEXT(kTextures2D[sourceTexId]), texelCoord, 0).r;
}

void main() {
  vec2 previewSize = max(vec2(pc.previewParams.xy), vec2(1.0));
  uint sourceCount = clamp(pc.previewParams.z, 1u, MAX_PREVIEW_SOURCES);
  uint flags = pc.previewParams.w;
  float depth = 0.0;

  if ((flags & FLAG_TILED) != 0u) {
    vec2 tileSize = previewSize * 0.5;
    vec2 fragPos = clamp(gl_FragCoord.xy, vec2(0.0), previewSize - vec2(1.0));
    uvec2 tileCoord =
        uvec2(clamp(floor(fragPos / tileSize), vec2(0.0), vec2(1.0)));
    uint tileIndex = tileCoord.x + tileCoord.y * 2u;
    if (tileIndex >= sourceCount) {
      out_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
      return;
    }
    vec2 tileOrigin = vec2(tileCoord) * tileSize;
    vec2 tileUv = (fragPos - tileOrigin) / max(tileSize, vec2(1.0));
    depth = fetchDepth(pc.sourceTexIds[tileIndex], tileUv);
  } else {
    depth = fetchDepth(pc.sourceTexIds.x, gl_FragCoord.xy / previewSize);
  }

  float preview = clamp(depth * pc.depthParams.x + pc.depthParams.y, 0.0, 1.0);
  if ((flags & FLAG_LOG_SCALE) != 0u) {
    preview = log2(1.0 + preview * 255.0) / 8.0;
  }
  if ((flags & FLAG_INVERT) != 0u) {
    preview = 1.0 - preview;
  }
  out_FragColor = vec4(vec3(preview), 1.0);
}
)";

[[nodiscard]] std::pmr::memory_resource *
resolveMemoryResource(std::pmr::memory_resource *memory) {
  return memory != nullptr ? memory : std::pmr::get_default_resource();
}

[[nodiscard]] uint64_t hashBytes(std::span<const std::byte> bytes) {
  uint64_t hash = kFnvOffsetBasis64;
  for (const std::byte value : bytes) {
    hash ^= static_cast<uint64_t>(std::to_integer<uint8_t>(value));
    hash *= kFnvPrime64;
  }
  return hash;
}

[[nodiscard]] uint64_t hashCombine64(uint64_t hash, uint64_t value) {
  hash ^= value;
  hash *= kFnvPrime64;
  return hash;
}

[[nodiscard]] uint64_t hashCombineBytes(uint64_t hash,
                                        std::span<const std::byte> bytes) {
  for (const std::byte value : bytes) {
    hash ^= static_cast<uint64_t>(std::to_integer<uint8_t>(value));
    hash *= kFnvPrime64;
  }
  return hash;
}

template <typename T>
[[nodiscard]] uint64_t hashCombineValue(uint64_t hash, const T &value) {
  return hashCombineBytes(hash, std::as_bytes(std::span<const T>(&value, 1u)));
}

[[nodiscard]] uint64_t foldHandle(uint32_t index, uint32_t generation) {
  return (static_cast<uint64_t>(generation) << 32u) | index;
}

[[nodiscard]] bool isSameBufferHandle(BufferHandle lhs, BufferHandle rhs) {
  return lhs.index == rhs.index && lhs.generation == rhs.generation;
}

[[nodiscard]] std::string_view shadowSdsmModeName(ShadowSdsmMode mode) {
  switch (mode) {
  case ShadowSdsmMode::Disabled:
    return "Disabled";
  case ShadowSdsmMode::PreviousFrameMinMax:
    return "PreviousFrameMinMax";
  case ShadowSdsmMode::Histogram:
    return "Histogram";
  }
  return "Unknown";
}

[[nodiscard]] std::string_view shadowSdsmStatusName(ShadowSdsmStatus status) {
  switch (status) {
  case ShadowSdsmStatus::Disabled:
    return "Disabled";
  case ShadowSdsmStatus::Active:
    return "Active";
  case ShadowSdsmStatus::Unavailable:
    return "Unavailable";
  case ShadowSdsmStatus::Stale:
    return "Stale";
  case ShadowSdsmStatus::Invalid:
    return "Invalid";
  case ShadowSdsmStatus::FallbackFixed:
    return "FallbackFixed";
  }
  return "Unknown";
}

[[nodiscard]] std::string_view
shadowSdsmReductionBackendName(ShadowSdsmReductionBackend backend) {
  switch (sanitizeShadowSdsmReductionBackend(backend)) {
  case ShadowSdsmReductionBackend::Auto:
    return "Auto";
  case ShadowSdsmReductionBackend::Cpu:
    return "Cpu";
  case ShadowSdsmReductionBackend::Gpu:
    return "Gpu";
  }
  return "Unknown";
}

[[nodiscard]] std::string_view shadowFilterModeName(ShadowFilterMode mode) {
  switch (sanitizeShadowFilterMode(mode)) {
  case ShadowFilterMode::Hard:
    return "Hard";
  case ShadowFilterMode::PCF3x3:
    return "PCF3x3";
  case ShadowFilterMode::PoissonPCF:
    return "PoissonPCF";
  case ShadowFilterMode::PCSS:
    return "PCSS";
  }
  return "Unknown";
}

[[nodiscard]] std::string_view
shadowCascadeSplitModeName(ShadowCascadeSplitMode mode) {
  switch (sanitizeShadowCascadeSplitMode(mode)) {
  case ShadowCascadeSplitMode::Uniform:
    return "Uniform";
  case ShadowCascadeSplitMode::Logarithmic:
    return "Logarithmic";
  case ShadowCascadeSplitMode::Practical:
    return "Practical";
  }
  return "Unknown";
}

[[nodiscard]] std::string_view
projectionTypeName(ProjectionType projectionType) {
  switch (projectionType) {
  case ProjectionType::Perspective:
    return "Perspective";
  case ProjectionType::Orthographic:
    return "Orthographic";
  }
  return "Unknown";
}

[[nodiscard]] std::string_view shadowStaticOnlyReuseStatusName(
    ShadowCascadeDebugFrameData::StaticOnlyReuseStatus status) {
  switch (status) {
  case ShadowCascadeDebugFrameData::StaticOnlyReuseStatus::None:
    return "None";
  case ShadowCascadeDebugFrameData::StaticOnlyReuseStatus::Reused:
    return "Reused";
  case ShadowCascadeDebugFrameData::StaticOnlyReuseStatus::StaticCacheRebuilt:
    return "StaticCacheRebuilt";
  case ShadowCascadeDebugFrameData::StaticOnlyReuseStatus::HasDynamicCasters:
    return "HasDynamicCasters";
  case ShadowCascadeDebugFrameData::StaticOnlyReuseStatus::
      NoPreviousStaticOnlyPass:
    return "NoPreviousStaticOnlyPass";
  case ShadowCascadeDebugFrameData::StaticOnlyReuseStatus::RasterStateChanged:
    return "RasterStateChanged";
  case ShadowCascadeDebugFrameData::StaticOnlyReuseStatus::AdaptiveRefresh:
    return "AdaptiveRefresh";
  }
  return "Unknown";
}

template <typename T>
[[nodiscard]] bool rawBytesEqual(const T &lhs, const T &rhs) {
  return std::memcmp(&lhs, &rhs, sizeof(T)) == 0;
}

[[nodiscard]] uint64_t buildShadowDiagnosticSignature(
    const RenderFrameContext &frame,
    const RenderSettings::ShadowSettings &settings,
    const ShadowDebugFrameData &debugFrameData,
    const ShadowFrameMetrics &metrics, bool hasActiveShadowLightForFrame,
    bool hasPreparedShadowDepthPasses, bool hasPreparedShadowPreviewPass) {
  uint64_t hash = kFnvOffsetBasis64;
  hash = hashCombineValue(hash, settings.enabled);
  hash = hashCombineValue(hash, settings.cascadeCount);
  hash = hashCombineValue(hash, settings.shadowMapSize);
  hash = hashCombineValue(hash, settings.maxDistance);
  hash = hashCombineValue(hash, settings.splitMode);
  hash = hashCombineValue(hash, settings.splitLambda);
  hash = hashCombineValue(hash, settings.stabilizeCascades);
  hash = hashCombineValue(hash, settings.cascadeBlendFraction);
  hash = hashCombineValue(hash, settings.constantBias);
  hash = hashCombineValue(hash, settings.slopeBias);
  hash = hashCombineValue(hash, settings.normalBias);
  hash = hashCombineValue(hash, settings.filterMode);
  hash = hashCombineValue(hash, settings.pcfSampleCount);
  hash = hashCombineValue(hash, settings.pcssBlockerSampleCount);
  hash = hashCombineValue(hash, settings.pcssFilterSampleCount);
  hash = hashCombineValue(hash, settings.pcssLightRadiusScale);
  hash = hashCombineValue(hash, settings.pcssSearchRadiusClampTexels);
  hash = hashCombineValue(hash, settings.pcssFilterRadiusClampTexels);
  hash = hashCombineValue(hash, settings.sdsmMode);
  hash = hashCombineValue(hash, settings.sdsmReductionBackend);
  hash = hashCombineValue(hash, settings.sdsmTemporalBlend);
  hash = hashCombineValue(hash, settings.sdsmHistogramBucketCount);
  hash = hashCombineValue(hash, settings.sdsmHistogramTrimLowPercent);
  hash = hashCombineValue(hash, settings.sdsmHistogramTrimHighPercent);
  hash = hashCombineValue(hash, settings.debug.freezeCascades);
  hash = hashCombineValue(hash, settings.debug.freezeLightView);
  hash = hashCombineValue(hash, settings.debug.enableCascadeCasterCulling);
  hash = hashCombineValue(hash, settings.debug.debugCascadeIndex);
  hash = hashCombineValue(hash, frame.camera.projectionType);
  hash = hashCombineValue(hash, frame.camera.nearPlane);
  hash = hashCombineValue(hash, frame.camera.farPlane);
  hash = hashCombineValue(hash, frame.camera.aspectRatio);
  hash = hashCombineBytes(hash, std::as_bytes(std::span<const glm::vec4>(
                                    &frame.camera.cameraPos, 1u)));
  hash = hashCombineBytes(
      hash, std::as_bytes(std::span<const glm::mat4>(&frame.camera.view, 1u)));
  hash = hashCombineBytes(
      hash, std::as_bytes(std::span<const glm::mat4>(&frame.camera.proj, 1u)));
  hash = hashCombineValue(hash, hasActiveShadowLightForFrame);
  hash = hashCombineValue(hash, hasPreparedShadowDepthPasses);
  hash = hashCombineValue(hash, hasPreparedShadowPreviewPass);
  hash = hashCombineValue(hash, isValid(debugFrameData.selectedShadowLightId));
  hash = hashCombineValue(hash, isValid(debugFrameData.selectedShadowLightId)
                                    ? debugFrameData.selectedShadowLightId.value
                                    : 0u);
  hash = hashCombineValue(hash, debugFrameData.cascadeCount);
  hash = hashCombineValue(hash, debugFrameData.rawSamplerId);
  hash = hashCombineValue(hash, debugFrameData.compareSamplerId);

  const ShadowSdsmDebugFrameData &sdsm = debugFrameData.sdsm;
  hash = hashCombineValue(hash, sdsm.mode);
  hash = hashCombineValue(hash, sdsm.requestedReductionBackend);
  hash = hashCombineValue(hash, sdsm.activeReductionBackend);
  hash = hashCombineValue(hash, sdsm.status);
  hash = hashCombineValue(hash, sdsm.reductionFallbackActive);
  hash = hashCombineValue(hash, sdsm.fixedFallbackActive);
  hash = hashCombineValue(hash, sdsm.sourceFrameIndex);
  hash = hashCombineValue(hash, sdsm.histogramSourceLevel);
  hash = hashCombineBytes(hash, std::as_bytes(std::span<const glm::uvec2>(
                                    &sdsm.histogramSourceDimensions, 1u)));
  hash = hashCombineValue(hash, sdsm.histogramBucketCount);
  hash = hashCombineValue(hash, sdsm.histogramValidTileCount);
  hash = hashCombineValue(hash, sdsm.gpuResultRingSlotCount);
  hash = hashCombineValue(hash, sdsm.gpuResultSelectedSlot);
  hash = hashCombineValue(hash, sdsm.gpuReductionResultAvailable);
  hash = hashCombineValue(hash, sdsm.gpuSplitPayloadValid);
  hash = hashCombineValue(hash, sdsm.gpuResultSourceFrameIndex);
  hash = hashCombineValue(hash, sdsm.splitCount);
  hash = hashCombineValue(hash, sdsm.fixedRangeNear);
  hash = hashCombineValue(hash, sdsm.fixedRangeFar);
  hash = hashCombineValue(hash, sdsm.rawDeviceMin);
  hash = hashCombineValue(hash, sdsm.rawDeviceMax);
  hash = hashCombineValue(hash, sdsm.rawLinearMin);
  hash = hashCombineValue(hash, sdsm.rawLinearMax);
  hash = hashCombineValue(hash, sdsm.smoothedLinearMin);
  hash = hashCombineValue(hash, sdsm.smoothedLinearMax);
  hash = hashCombineValue(hash, sdsm.histogramTotalWeight);
  hash = hashCombineValue(hash, sdsm.histogramTrimLowPercent);
  hash = hashCombineValue(hash, sdsm.histogramTrimHighPercent);
  hash = hashCombineValue(hash, sdsm.histogramTrimmedRangeNear);
  hash = hashCombineValue(hash, sdsm.histogramTrimmedRangeFar);
  hash = hashCombineValue(hash, sdsm.effectiveRangeNear);
  hash = hashCombineValue(hash, sdsm.effectiveRangeFar);
  hash = hashCombineBytes(
      hash, std::as_bytes(std::span<const float>(
                sdsm.fixedSplitDepths.data(), sdsm.fixedSplitDepths.size())));
  hash = hashCombineBytes(
      hash, std::as_bytes(std::span<const float>(
                sdsm.minMaxSplitDepths.data(), sdsm.minMaxSplitDepths.size())));
  hash = hashCombineBytes(hash, std::as_bytes(std::span<const float>(
                                    sdsm.histogramSplitDepths.data(),
                                    sdsm.histogramSplitDepths.size())));
  hash = hashCombineBytes(hash, std::as_bytes(std::span<const float>(
                                    sdsm.effectiveSplitDepths.data(),
                                    sdsm.effectiveSplitDepths.size())));
  if (sdsm.histogramBucketCount > 0u) {
    const size_t histogramWeightCount = std::min<size_t>(
        sdsm.histogramBucketCount, sdsm.histogramBucketWeights.size());
    hash = hashCombineBytes(
        hash, std::as_bytes(std::span<const float>(
                  sdsm.histogramBucketWeights.data(), histogramWeightCount)));
  }

  const uint32_t cascadeCount =
      std::min(debugFrameData.cascadeCount, kMaxShadowCascades);
  for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
       ++cascadeIndex) {
    const ShadowCascadeDebugFrameData &cascade =
        debugFrameData.cascades[cascadeIndex];
    hash = hashCombineValue(hash, cascade.splitNear);
    hash = hashCombineValue(hash, cascade.splitFar);
    hash = hashCombineValue(hash, cascade.texelWorldSize);
    hash = hashCombineBytes(hash, std::as_bytes(std::span<const glm::mat4>(
                                      &cascade.lightViewProj, 1u)));
    hash = hashCombineBytes(hash, std::as_bytes(std::span<const glm::vec4>(
                                      &cascade.lightSpaceBoundsMin, 1u)));
    hash = hashCombineBytes(hash, std::as_bytes(std::span<const glm::vec4>(
                                      &cascade.lightSpaceBoundsMax, 1u)));
    hash = hashCombineBytes(hash, std::as_bytes(std::span<const glm::vec4>(
                                      &cascade.unsnappedCenter, 1u)));
    hash = hashCombineBytes(hash, std::as_bytes(std::span<const glm::vec4>(
                                      &cascade.snappedCenter, 1u)));
    hash = hashCombineValue(hash, cascade.textureBindlessId);
    hash = hashCombineValue(hash, cascade.drawCount);
    hash = hashCombineValue(hash, cascade.culledCount);
    hash = hashCombineValue(hash, cascade.staticDrawCount);
    hash = hashCombineValue(hash, cascade.dynamicDrawCount);
    hash = hashCombineValue(hash, cascade.staticOnlyReuseStatus);
    hash = hashCombineValue(hash, cascade.staticOnlyReuseCandidate);
    hash = hashCombineValue(hash, cascade.staticOnlyReusePreviousValid);
    hash = hashCombineValue(hash, cascade.staticOnlyReuseLightViewProjChanged);
    hash = hashCombineValue(hash, cascade.staticOnlyReuseBiasChanged);
    hash =
        hashCombineValue(hash, cascade.staticOnlyReuseCasterSignatureChanged);
    hash = hashCombineValue(hash, cascade.staticOnlyReuseAdaptiveRefresh);
    hash = hashCombineValue(hash, cascade.currentStaticOnlyRasterSignature);
    hash = hashCombineValue(hash, cascade.previousStaticOnlyRasterSignature);
    hash =
        hashCombineValue(hash, cascade.currentStaticOnlyLightViewProjSignature);
    hash = hashCombineValue(hash,
                            cascade.previousStaticOnlyLightViewProjSignature);
    hash = hashCombineValue(hash, cascade.currentStaticOnlyBiasSignature);
    hash = hashCombineValue(hash, cascade.previousStaticOnlyBiasSignature);
    hash = hashCombineValue(hash, cascade.currentStaticOnlyCasterSignature);
    hash = hashCombineValue(hash, cascade.previousStaticOnlyCasterSignature);
  }

  hash = hashCombineValue(hash, metrics.cascadeCount);
  hash = hashCombineValue(hash, metrics.shadowMapSize);
  hash = hashCombineValue(hash, metrics.totalDraws);
  hash = hashCombineValue(hash, metrics.totalCulledDraws);
  hash = hashCombineValue(hash, metrics.totalIndexCountEstimate);
  hash = hashCombineValue(hash, metrics.staticCasterEntries);
  hash = hashCombineValue(hash, metrics.dynamicCasterEntries);
  hash = hashCombineValue(hash, metrics.staticCacheReused);
  hash = hashCombineValue(hash, metrics.staticOnlyCandidateCount);
  hash = hashCombineValue(hash, metrics.reusedStaticOnlyCascadeCount);
  hash = hashCombineValue(hash,
                          metrics.staticOnlyReuseMissStaticCacheRebuiltCount);
  hash = hashCombineValue(hash, metrics.staticOnlyReuseMissDynamicCasterCount);
  hash = hashCombineValue(hash, metrics.staticOnlyReuseMissNoPreviousCount);
  hash = hashCombineValue(hash,
                          metrics.staticOnlyReuseMissRasterStateChangedCount);
  hash =
      hashCombineValue(hash, metrics.staticOnlyReuseMissAdaptiveRefreshCount);
  hash = hashCombineValue(hash, metrics.cascadeTextureBytes);
  hash = hashCombineValue(hash, metrics.filterSampleBudget);
  hash = hashCombineValue(hash, metrics.pcssBlockerSampleBudget);
  hash = hashCombineValue(hash, metrics.pcssFilterSampleBudget);
  hash = hashCombineValue(hash, metrics.pcssMaxSamplesPerReceiver);
  hash = hashCombineValue(hash, metrics.pcssMaxSamplesPerBlendedReceiver);
  hash = hashCombineValue(hash, metrics.sdsmComputePassCount);
  hash = hashCombineValue(hash, metrics.sdsmReadbackBytes);
  hash = hashCombineValue(hash, metrics.sdsmReductionSourceSamples);
  hash = hashCombineValue(hash, metrics.sdsmHistogramSourceSamples);
  hash = hashCombineValue(hash, metrics.sdsmActiveReductionBackend);
  hash = hashCombineValue(hash, metrics.gpuTimeMs);
  hash = hashCombineValue(hash, metrics.depthGpuTimeMs);
  hash = hashCombineValue(hash, metrics.sdsmGpuTimeMs);
  hash = hashCombineValue(hash, metrics.sdsmCpuReductionTimeMs);
  hash = hashCombineValue(hash, metrics.sdsmCpuHistogramTimeMs);
  hash = hashCombineValue(hash, metrics.gpuTimingSourceFrameIndex);
  hash = hashCombineValue(hash, metrics.depthGpuTimingSourceFrameIndex);
  hash = hashCombineValue(hash, metrics.sdsmGpuTimingSourceFrameIndex);
  hash = hashCombineValue(hash, metrics.gpuTimingAvailable);
  hash = hashCombineValue(hash, metrics.depthGpuTimingAvailable);
  hash = hashCombineValue(hash, metrics.sdsmGpuTimingAvailable);
  return hash;
}

[[nodiscard]] IndexFormat
resolveGeometryIndexFormat(const GeometryAllocationView &geometry) {
  if (geometry.indexCount == 0u) {
    return IndexFormat::U32;
  }
  const uint64_t indexStride =
      geometry.indexByteSize / static_cast<uint64_t>(geometry.indexCount);
  return indexStride == sizeof(uint16_t) ? IndexFormat::U16 : IndexFormat::U32;
}

constexpr uint64_t kSdsmMaxCachedSourceFrameLag = 2u;

void appendUniqueBufferDependency(
    std::pmr::vector<BufferDependency> &dependencies, BufferHandle handle,
    RenderGraphAccessMode mode = RenderGraphAccessMode::Read) {
  if (!nuri::isValid(handle)) {
    return;
  }
  for (const BufferDependency &existing : dependencies) {
    if (isSameBufferHandle(existing.handle, handle)) {
      return;
    }
  }
  dependencies.push_back(BufferDependency{
      .handle = handle,
      .mode = mode,
  });
}

void appendUniqueTextureDependency(
    std::pmr::vector<TextureDependency> &dependencies, TextureHandle handle,
    RenderGraphAccessMode mode = RenderGraphAccessMode::Read) {
  if (!nuri::isValid(handle)) {
    return;
  }
  for (const TextureDependency &existing : dependencies) {
    if (existing.handle.index == handle.index &&
        existing.handle.generation == handle.generation) {
      return;
    }
  }
  dependencies.push_back(TextureDependency{
      .handle = handle,
      .mode = mode,
  });
}

struct ShadowSplitRange {
  float nearDepth = 0.0f;
  float farDepth = 0.0f;
};

struct SdsmLogDiagnostics {
  std::string_view reason = "not_evaluated";
  uint32_t sourceLevelCount = 0u;
  bool sourceFrameAvailable = false;
  bool hasValidSource = false;
  bool sourceTextureValid = false;
  bool reusedCachedRange = false;
  bool readbackError = false;
  bool rawDepthsValid = false;
  bool rawLinearValid = false;
  bool historyRangeValid = false;
  bool historyTexelValid = false;
};

struct SdsmHistogramAnalysis {
  bool valid = false;
  bool clearOnlySource = false;
  uint32_t sourceLevel = 0u;
  glm::uvec2 sourceDimensions{0u};
  uint32_t validTileCount = 0u;
  float totalWeight = 0.0f;
  float rawDeviceMin = 0.0f;
  float rawDeviceMax = 0.0f;
  float rawLinearMin = 0.0f;
  float rawLinearMax = 0.0f;
  float trimmedNear = 0.0f;
  float trimmedFar = 0.0f;
  std::array<float, kMaxShadowCascades + 1u> rawSplitDepths{};
  std::array<float, kMaxShadowSdsmHistogramBucketCount> bucketWeights{};
};

struct CachedSdsmHistogramTile {
  float rawDeviceMin = 0.0f;
  float rawDeviceMax = 0.0f;
  float rawLinearMin = 0.0f;
  float rawLinearMax = 0.0f;
  bool valid = false;
  bool clearOnly = false;
  bool clearMax = false;
};

[[nodiscard]] SdsmHistogramAnalysis buildSdsmHistogramAnalysis(
    const CameraFrameState &camera, ShadowSplitRange fixedSplitRange,
    uint32_t cascadeCount, uint32_t bucketCount, float trimLowPercent,
    float trimHighPercent, uint32_t sourceLevel, glm::uvec2 sourceDimensions,
    std::span<const float> minMaxPairs) {
  SdsmHistogramAnalysis analysis{};
  analysis.sourceLevel = sourceLevel;
  analysis.sourceDimensions = sourceDimensions;

  const uint32_t safeBucketCount =
      std::clamp(bucketCount, kMinShadowSdsmHistogramBucketCount,
                 kMaxShadowSdsmHistogramBucketCount);
  if (minMaxPairs.size() < 2u || (minMaxPairs.size() % 2u) != 0u) {
    return analysis;
  }

  analysis.rawDeviceMin = 1.0f;
  analysis.rawDeviceMax = 0.0f;
  analysis.rawLinearMin = fixedSplitRange.farDepth;
  analysis.rawLinearMax = fixedSplitRange.nearDepth;
  bool hasNonClearLinearMax = false;
  float maxNonClearLinearMax = fixedSplitRange.nearDepth;
  uint32_t rawValidTileCount = 0u;
  uint32_t clearOnlyTileCount = 0u;
  bool malformedTile = false;
  std::vector<CachedSdsmHistogramTile> cachedTiles(minMaxPairs.size() / 2u);
  const auto isClearDepthSample = [](float rawDeviceDepth) {
    return rawDeviceDepth >= (1.0f - kSdsmHistogramClearDepthEpsilon);
  };

  for (size_t tileIndex = 0u; tileIndex < cachedTiles.size(); ++tileIndex) {
    CachedSdsmHistogramTile &cachedTile = cachedTiles[tileIndex];
    cachedTile.rawDeviceMin = minMaxPairs[tileIndex * 2u];
    cachedTile.rawDeviceMax = minMaxPairs[tileIndex * 2u + 1u];
    const bool rawDepthsValid =
        std::isfinite(cachedTile.rawDeviceMin) &&
        std::isfinite(cachedTile.rawDeviceMax) &&
        cachedTile.rawDeviceMin >= -1.0e-4f &&
        cachedTile.rawDeviceMax <= (1.0f + 1.0e-4f) &&
        cachedTile.rawDeviceMax >= cachedTile.rawDeviceMin;
    if (!rawDepthsValid) {
      malformedTile = true;
      continue;
    }

    cachedTile.rawLinearMin = shadow_detail::linearizeDeviceDepthToViewDepth(
        cachedTile.rawDeviceMin, camera);
    cachedTile.rawLinearMax = shadow_detail::linearizeDeviceDepthToViewDepth(
        cachedTile.rawDeviceMax, camera);
    if (!std::isfinite(cachedTile.rawLinearMin) ||
        !std::isfinite(cachedTile.rawLinearMax) ||
        cachedTile.rawLinearMax < cachedTile.rawLinearMin) {
      malformedTile = true;
      continue;
    }

    const bool clearMin = isClearDepthSample(cachedTile.rawDeviceMin);
    cachedTile.clearMax = isClearDepthSample(cachedTile.rawDeviceMax);
    cachedTile.clearOnly = clearMin && cachedTile.clearMax;
    cachedTile.valid = true;
    ++rawValidTileCount;
    if (cachedTile.clearOnly) {
      ++clearOnlyTileCount;
      continue;
    }

    analysis.rawDeviceMin =
        std::min(analysis.rawDeviceMin, cachedTile.rawDeviceMin);
    analysis.rawDeviceMax =
        std::max(analysis.rawDeviceMax, cachedTile.rawDeviceMax);
    analysis.rawLinearMin =
        std::min(analysis.rawLinearMin, cachedTile.rawLinearMin);
    analysis.rawLinearMax =
        std::max(analysis.rawLinearMax, cachedTile.rawLinearMax);
    if (!cachedTile.clearMax) {
      maxNonClearLinearMax =
          std::max(maxNonClearLinearMax, cachedTile.rawLinearMax);
      hasNonClearLinearMax = true;
    }
  }

  for (const CachedSdsmHistogramTile &cachedTile : cachedTiles) {
    if (!cachedTile.valid || cachedTile.clearOnly) {
      continue;
    }

    float histogramLinearMax = cachedTile.rawLinearMax;
    if (cachedTile.clearMax) {
      // A tile that touches clear depth does not prove geometry extends all
      // the way to the camera far plane. Cap it to the deepest non-clear
      // sample seen in this mip so histogram trimming does not jump on sky
      // contamination.
      histogramLinearMax =
          hasNonClearLinearMax
              ? std::max(cachedTile.rawLinearMin, maxNonClearLinearMax)
              : cachedTile.rawLinearMin;
    }

    shadow_detail::accumulateSdsmHistogramInterval(
        std::span<float>(analysis.bucketWeights.data(), safeBucketCount),
        fixedSplitRange.nearDepth, fixedSplitRange.farDepth,
        cachedTile.rawLinearMin, histogramLinearMax, 1.0f);
    ++analysis.validTileCount;
    analysis.totalWeight += 1.0f;
  }

  if (analysis.validTileCount == 0u || analysis.totalWeight <= 1.0e-6f) {
    const bool clearOnlySource = !malformedTile && rawValidTileCount > 0u &&
                                 clearOnlyTileCount == rawValidTileCount;
    if (clearOnlySource) {
      const float clearLinearDepth =
          shadow_detail::linearizeDeviceDepthToViewDepth(1.0f, camera);
      if (std::isfinite(clearLinearDepth)) {
        analysis.valid = true;
        analysis.clearOnlySource = true;
        analysis.validTileCount = clearOnlyTileCount;
        analysis.rawDeviceMin = 1.0f;
        analysis.rawDeviceMax = 1.0f;
        analysis.rawLinearMin = clearLinearDepth;
        analysis.rawLinearMax = clearLinearDepth;
        analysis.trimmedNear = fixedSplitRange.farDepth;
        analysis.trimmedFar = fixedSplitRange.farDepth;
        analysis.rawSplitDepths.fill(fixedSplitRange.farDepth);
        analysis.rawSplitDepths[0] = fixedSplitRange.nearDepth;
      }
    }
    return analysis;
  }

  const float lowPercentile = std::clamp(trimLowPercent / 100.0f, 0.0f, 0.95f);
  const float highPercentile = std::clamp(1.0f - trimHighPercent / 100.0f,
                                          lowPercentile + 1.0e-3f, 1.0f);
  analysis.trimmedNear = shadow_detail::sdsmHistogramPercentileDepth(
      std::span<const float>(analysis.bucketWeights.data(), safeBucketCount),
      fixedSplitRange.nearDepth, fixedSplitRange.farDepth, lowPercentile);
  analysis.trimmedFar = shadow_detail::sdsmHistogramPercentileDepth(
      std::span<const float>(analysis.bucketWeights.data(), safeBucketCount),
      fixedSplitRange.nearDepth, fixedSplitRange.farDepth, highPercentile);

  analysis.rawSplitDepths.fill(fixedSplitRange.farDepth);
  analysis.rawSplitDepths[0] = fixedSplitRange.nearDepth;
  for (uint32_t i = 1u; i < std::clamp(cascadeCount, 1u, kMaxShadowCascades);
       ++i) {
    const float t = static_cast<float>(i) / static_cast<float>(cascadeCount);
    const float percentile = glm::mix(lowPercentile, highPercentile, t);
    analysis.rawSplitDepths[i] = shadow_detail::sdsmHistogramPercentileDepth(
        std::span<const float>(analysis.bucketWeights.data(), safeBucketCount),
        fixedSplitRange.nearDepth, fixedSplitRange.farDepth, percentile);
  }
  analysis.rawSplitDepths[cascadeCount] = analysis.trimmedFar;
  shadow_detail::enforceMonotonicShadowSplitDepths(
      analysis.rawSplitDepths, cascadeCount, fixedSplitRange.nearDepth,
      std::max(analysis.trimmedFar, fixedSplitRange.nearDepth + 1.0e-4f));
  analysis.valid = std::isfinite(analysis.trimmedNear) &&
                   std::isfinite(analysis.trimmedFar) &&
                   analysis.trimmedFar >= analysis.trimmedNear;
  return analysis;
}

[[nodiscard]] ShadowSplitRange
computeFixedShadowSplitRange(const CameraFrameState &camera,
                             float maxDistance) {
  const float nearDepth = std::max(camera.nearPlane, 0.01f);
  const float requestedFar =
      std::min(std::max(maxDistance, nearDepth + 0.01f),
               std::max(camera.farPlane, nearDepth + 0.01f));
  return ShadowSplitRange{
      .nearDepth = nearDepth,
      .farDepth = std::max(nearDepth + 0.01f, requestedFar),
  };
}

[[nodiscard]] float
computeSdsmFarUpdateThreshold(float farCascadeTexelWorldSize) {
  return std::max(std::max(farCascadeTexelWorldSize, 0.0f) * 16.0f, 0.5f);
}

[[nodiscard]] CameraFrameState
cameraWithShadowSplitRange(const CameraFrameState &camera,
                           ShadowSplitRange range) {
  CameraFrameState adjustedCamera = camera;
  adjustedCamera.nearPlane = std::max(range.nearDepth, 0.01f);
  adjustedCamera.farPlane =
      std::max(adjustedCamera.nearPlane + 0.01f, range.farDepth);
  return adjustedCamera;
}

template <typename DependencyT, typename HandleT>
void splitDependencies(std::span<const DependencyT> dependencies,
                       std::pmr::vector<HandleT> &handles,
                       std::pmr::vector<RenderGraphAccessMode> &accessModes) {
  handles.clear();
  accessModes.clear();
  handles.reserve(dependencies.size());
  accessModes.reserve(dependencies.size());
  for (const DependencyT &dependency : dependencies) {
    handles.push_back(dependency.handle);
    accessModes.push_back(dependency.mode);
  }
}

const AnimationSceneFrameData *
resolveAnimationSceneFrameData(const RenderFrameContext &frame) {
  if (!frame.sharedResources.animationSceneGpuData.has_value()) {
    return nullptr;
  }
  const AnimationSceneFrameData &data =
      *frame.sharedResources.animationSceneGpuData;
  if (!nuri::isValid(data.instanceMatricesBuffer) ||
      data.instanceMatricesAddress == 0u) {
    return nullptr;
  }
  if (frame.scene == nullptr || data.scene != frame.scene ||
      data.sceneTopologyVersion != frame.scene->topologyVersion() ||
      data.renderableCount != frame.scene->renderables().size() ||
      data.geometryOverridesByRenderable.size() != data.renderableCount) {
    return nullptr;
  }
  return &data;
}

bool animationOverrideCoversSubmesh(
    const AnimatedRenderableGeometryOverride &geometryOverride,
    const Submesh &submesh) noexcept {
  const uint64_t requiredVertexCount =
      static_cast<uint64_t>(submesh.vertexOffset) + submesh.vertexCount;
  return static_cast<uint64_t>(geometryOverride.vertexCount) >=
         requiredVertexCount;
}

void appendAnimatedGeometryDependencies(
    std::pmr::vector<BufferDependency> &dependencies,
    const AnimationSceneFrameData *animationSceneData) {
  if (animationSceneData == nullptr) {
    return;
  }
  for (const AnimatedRenderableGeometryOverride &geometryOverride :
       animationSceneData->geometryOverridesByRenderable) {
    if (!geometryOverride.enabled ||
        !nuri::isValid(geometryOverride.vertexBuffer)) {
      continue;
    }
    appendUniqueBufferDependency(dependencies, geometryOverride.vertexBuffer);
  }
}

[[nodiscard]] std::span<const ComputeDispatchItem>
shadowCascadePreDispatches(const AnimationSceneFrameData *animationSceneData,
                           uint32_t cascadeIndex) {
  if (cascadeIndex != 0u || animationSceneData == nullptr) {
    return {};
  }
  return animationSceneData->preDispatches;
}

struct StaticOnlyGuardBandTexels {
  float ortho = 0.0f;
  float depth = 0.0f;
};

struct StaticOnlyDirectedTexels {
  glm::vec2 x{0.0f};
  glm::vec2 y{0.0f};
  glm::vec2 z{0.0f};
};

[[nodiscard]] float staticOnlyGuardBandTexels(uint32_t shadowMapSize) {
  return std::clamp(
      static_cast<float>(shadowMapSize) * kStaticOnlyGuardBandShadowMapFraction,
      kStaticOnlyGuardBandMinTexels, kStaticOnlyGuardBandMaxTexels);
}

[[nodiscard]] float staticOnlyDepthGuardBandTexels(uint32_t shadowMapSize) {
  return std::clamp(static_cast<float>(shadowMapSize) *
                        kStaticOnlyDepthGuardBandShadowMapFraction,
                    kStaticOnlyDepthGuardBandMinTexels,
                    kStaticOnlyDepthGuardBandMaxTexels);
}

[[nodiscard]] float
staticOnlyAdaptiveGuardBandMaxTexels(uint32_t shadowMapSize) {
  return std::max(staticOnlyGuardBandTexels(shadowMapSize),
                  static_cast<float>(shadowMapSize) *
                      kStaticOnlyAdaptiveGuardBandMaxShadowMapFraction);
}

[[nodiscard]] float
staticOnlyAdaptiveDepthGuardBandMaxTexels(uint32_t shadowMapSize) {
  return std::max(staticOnlyDepthGuardBandTexels(shadowMapSize),
                  static_cast<float>(shadowMapSize) *
                      kStaticOnlyAdaptiveDepthGuardBandMaxShadowMapFraction);
}

[[nodiscard]] StaticOnlyGuardBandTexels staticOnlyRawMotionTexels(
    const shadow_detail::DirectionalShadowFit &currentRawFit,
    const shadow_detail::DirectionalShadowFit &previousRawFit,
    bool previousRawFitValid) {
  StaticOnlyGuardBandTexels motion{};
  if (!previousRawFitValid ||
      !shadow_detail::canReuseDirectionalShadowFitAnchor(currentRawFit,
                                                         previousRawFit)) {
    return motion;
  }

  const float texelWorldSize = std::max(currentRawFit.texelWorldSize, 1.0e-6f);
  const float orthoMotionWorld = std::max({
      std::abs(currentRawFit.lightSpaceBoundsMin.x -
               previousRawFit.lightSpaceBoundsMin.x),
      std::abs(currentRawFit.lightSpaceBoundsMax.x -
               previousRawFit.lightSpaceBoundsMax.x),
      std::abs(currentRawFit.lightSpaceBoundsMin.y -
               previousRawFit.lightSpaceBoundsMin.y),
      std::abs(currentRawFit.lightSpaceBoundsMax.y -
               previousRawFit.lightSpaceBoundsMax.y),
  });
  const float depthMotionWorld =
      std::max(std::abs(currentRawFit.lightSpaceBoundsMin.z -
                        previousRawFit.lightSpaceBoundsMin.z),
               std::abs(currentRawFit.lightSpaceBoundsMax.z -
                        previousRawFit.lightSpaceBoundsMax.z));

  motion.ortho = std::max(orthoMotionWorld / texelWorldSize, 0.0f);
  motion.depth = std::max(depthMotionWorld / texelWorldSize, 0.0f);
  return motion;
}

[[nodiscard]] StaticOnlyGuardBandTexels staticOnlyMotionGuardBandTexels(
    const shadow_detail::DirectionalShadowFit &currentRawFit,
    const shadow_detail::DirectionalShadowFit &previousRawFit,
    bool previousRawFitValid, uint32_t shadowMapSize) {
  const StaticOnlyGuardBandTexels motion = staticOnlyRawMotionTexels(
      currentRawFit, previousRawFit, previousRawFitValid);
  return StaticOnlyGuardBandTexels{
      .ortho = std::clamp(
          motion.ortho * kStaticOnlyAdaptiveGuardBandMotionMultiplier, 0.0f,
          staticOnlyAdaptiveGuardBandMaxTexels(shadowMapSize)),
      .depth = std::clamp(
          motion.depth * kStaticOnlyAdaptiveGuardBandMotionMultiplier, 0.0f,
          staticOnlyAdaptiveDepthGuardBandMaxTexels(shadowMapSize)),
  };
}

[[nodiscard]] StaticOnlyGuardBandTexels staticOnlyRenderGuardBandTexels(
    const shadow_detail::DirectionalShadowFit &currentRawFit,
    const shadow_detail::DirectionalShadowFit &previousRawFit,
    bool previousRawFitValid, uint32_t shadowMapSize) {
  const StaticOnlyGuardBandTexels motionGuard = staticOnlyMotionGuardBandTexels(
      currentRawFit, previousRawFit, previousRawFitValid, shadowMapSize);
  return StaticOnlyGuardBandTexels{
      .ortho =
          std::max(staticOnlyGuardBandTexels(shadowMapSize), motionGuard.ortho),
      .depth = std::max(staticOnlyDepthGuardBandTexels(shadowMapSize),
                        motionGuard.depth),
  };
}

[[nodiscard]] StaticOnlyDirectedTexels staticOnlyDirectedMotionTexels(
    const shadow_detail::DirectionalShadowFit &currentRawFit,
    const shadow_detail::DirectionalShadowFit &previousRawFit,
    bool previousRawFitValid) {
  StaticOnlyDirectedTexels motion{};
  if (!previousRawFitValid ||
      !shadow_detail::canReuseDirectionalShadowFitAnchor(currentRawFit,
                                                         previousRawFit)) {
    return motion;
  }

  const float texelWorldSize = std::max(currentRawFit.texelWorldSize, 1.0e-6f);
  const auto resolveAxisMotion = [texelWorldSize](
                                     float currentMin, float currentMax,
                                     float previousMin, float previousMax) {
    const float deltaMin = currentMin - previousMin;
    const float deltaMax = currentMax - previousMax;
    return glm::vec2(std::max({-deltaMin, -deltaMax, 0.0f}) / texelWorldSize,
                     std::max({deltaMin, deltaMax, 0.0f}) / texelWorldSize);
  };
  motion.x = resolveAxisMotion(currentRawFit.lightSpaceBoundsMin.x,
                               currentRawFit.lightSpaceBoundsMax.x,
                               previousRawFit.lightSpaceBoundsMin.x,
                               previousRawFit.lightSpaceBoundsMax.x);
  motion.y = resolveAxisMotion(currentRawFit.lightSpaceBoundsMin.y,
                               currentRawFit.lightSpaceBoundsMax.y,
                               previousRawFit.lightSpaceBoundsMin.y,
                               previousRawFit.lightSpaceBoundsMax.y);
  motion.z = resolveAxisMotion(currentRawFit.lightSpaceBoundsMin.z,
                               currentRawFit.lightSpaceBoundsMax.z,
                               previousRawFit.lightSpaceBoundsMin.z,
                               previousRawFit.lightSpaceBoundsMax.z);
  return motion;
}

[[nodiscard]] StaticOnlyDirectedTexels staticOnlyDirectedRemainingMarginTexels(
    const shadow_detail::DirectionalShadowFit &renderedFit,
    const shadow_detail::DirectionalShadowFit &currentRawFit) {
  StaticOnlyDirectedTexels margin{};
  if (!shadow_detail::canReuseDirectionalShadowFitAnchor(renderedFit,
                                                         currentRawFit)) {
    return margin;
  }

  const float texelWorldSize = std::max(currentRawFit.texelWorldSize, 1.0e-6f);
  const auto resolveAxisMargin = [texelWorldSize](
                                     float renderedMin, float renderedMax,
                                     float currentMin, float currentMax) {
    return glm::vec2(std::max(currentMin - renderedMin, 0.0f) / texelWorldSize,
                     std::max(renderedMax - currentMax, 0.0f) / texelWorldSize);
  };
  margin.x = resolveAxisMargin(
      renderedFit.lightSpaceBoundsMin.x, renderedFit.lightSpaceBoundsMax.x,
      currentRawFit.lightSpaceBoundsMin.x, currentRawFit.lightSpaceBoundsMax.x);
  margin.y = resolveAxisMargin(
      renderedFit.lightSpaceBoundsMin.y, renderedFit.lightSpaceBoundsMax.y,
      currentRawFit.lightSpaceBoundsMin.y, currentRawFit.lightSpaceBoundsMax.y);
  margin.z = resolveAxisMargin(
      renderedFit.lightSpaceBoundsMin.z, renderedFit.lightSpaceBoundsMax.z,
      currentRawFit.lightSpaceBoundsMin.z, currentRawFit.lightSpaceBoundsMax.z);
  return margin;
}

[[nodiscard]] bool staticOnlyNeedsAdaptiveRefresh(
    const shadow_detail::DirectionalShadowFit &renderedFit,
    const shadow_detail::DirectionalShadowFit &currentRawFit,
    const shadow_detail::DirectionalShadowFit &previousRawFit,
    bool previousRawFitValid, uint32_t shadowMapSize) {
  const StaticOnlyDirectedTexels motion = staticOnlyDirectedMotionTexels(
      currentRawFit, previousRawFit, previousRawFitValid);
  const StaticOnlyDirectedTexels margin =
      staticOnlyDirectedRemainingMarginTexels(renderedFit, currentRawFit);
  const StaticOnlyDirectedTexels previousMargin =
      staticOnlyDirectedRemainingMarginTexels(renderedFit, previousRawFit);
  const float baseOrthoGuard = staticOnlyGuardBandTexels(shadowMapSize);
  const float baseDepthGuard = staticOnlyDepthGuardBandTexels(shadowMapSize);
  const auto needsAxisRefresh = [](glm::vec2 currentMargin,
                                   glm::vec2 previousAxisMargin,
                                   glm::vec2 axisMotion, float baseGuard) {
    const bool negativeMarginWillExpire =
        currentMargin.x + kStaticOnlyPredictiveMotionEpsilonTexels <
        axisMotion.x;
    const bool positiveMarginWillExpire =
        currentMargin.y + kStaticOnlyPredictiveMotionEpsilonTexels <
        axisMotion.y;
    const bool negativeLargeStep =
        axisMotion.x > baseGuard + kStaticOnlyPredictiveMotionEpsilonTexels;
    const bool positiveLargeStep =
        axisMotion.y > baseGuard + kStaticOnlyPredictiveMotionEpsilonTexels;
    const bool negativeCacheAlreadyOffset =
        previousAxisMargin.x + kStaticOnlyPredictiveMotionEpsilonTexels <
        baseGuard;
    const bool positiveCacheAlreadyOffset =
        previousAxisMargin.y + kStaticOnlyPredictiveMotionEpsilonTexels <
        baseGuard;
    return (negativeMarginWillExpire &&
            (negativeLargeStep || negativeCacheAlreadyOffset)) ||
           (positiveMarginWillExpire &&
            (positiveLargeStep || positiveCacheAlreadyOffset));
  };
  return needsAxisRefresh(margin.x, previousMargin.x, motion.x,
                          baseOrthoGuard) ||
         needsAxisRefresh(margin.y, previousMargin.y, motion.y,
                          baseOrthoGuard) ||
         needsAxisRefresh(margin.z, previousMargin.z, motion.z, baseDepthGuard);
}

void applyStaticOnlyGuardBand(
    shadow_detail::DirectionalShadowFit &fit,
    const shadow_detail::DirectionalShadowFit &previousRawFit,
    bool previousRawFitValid, uint32_t shadowMapSize,
    StaticOnlyGuardBandTexels guard) {
  const float sourceTexelWorldSize = fit.texelWorldSize;
  const float baseOrthoPadding = shadow_detail::quantizeShadowPaddingUp(
      sourceTexelWorldSize * staticOnlyGuardBandTexels(shadowMapSize),
      sourceTexelWorldSize);
  const float baseDepthPadding = shadow_detail::quantizeShadowPaddingUp(
      sourceTexelWorldSize * staticOnlyDepthGuardBandTexels(shadowMapSize),
      sourceTexelWorldSize);
  const float adaptiveOrthoPadding = shadow_detail::quantizeShadowPaddingUp(
      sourceTexelWorldSize * guard.ortho, sourceTexelWorldSize);
  const float adaptiveDepthPadding = shadow_detail::quantizeShadowPaddingUp(
      sourceTexelWorldSize * guard.depth, sourceTexelWorldSize);
  const float extraOrthoPadding =
      std::max(adaptiveOrthoPadding - baseOrthoPadding, 0.0f);
  const float extraDepthPadding =
      std::max(adaptiveDepthPadding - baseDepthPadding, 0.0f);
  const bool hasMotionAnchor =
      previousRawFitValid &&
      shadow_detail::canReuseDirectionalShadowFitAnchor(fit, previousRawFit);
  if (!hasMotionAnchor) {
    shadow_detail::expandDirectionalShadowFitBounds(
        fit, shadowMapSize, adaptiveOrthoPadding, adaptiveDepthPadding);
    return;
  }

  const float predictiveOrthoPadding =
      baseOrthoPadding +
      extraOrthoPadding * kStaticOnlyPredictiveExtentGrowthFraction;
  const float predictiveDepthPadding =
      baseDepthPadding +
      extraDepthPadding * kStaticOnlyPredictiveExtentGrowthFraction;

  const glm::vec3 currentCenter = shadow_detail::lightSpaceCenter(
      fit.lightSpaceBoundsMin, fit.lightSpaceBoundsMax);
  const glm::vec3 currentExtent =
      fit.lightSpaceBoundsMax - fit.lightSpaceBoundsMin;

  const auto resolvePredictiveShift =
      [sourceTexelWorldSize](float deltaMin, float deltaMax,
                             float extentPadding) {
        const float negativeMotion = std::max({-deltaMin, -deltaMax, 0.0f});
        const float positiveMotion = std::max({deltaMin, deltaMax, 0.0f});
        if (std::max(negativeMotion, positiveMotion) <=
            sourceTexelWorldSize * kStaticOnlyPredictiveMotionEpsilonTexels) {
          return 0.0f;
        }
        const float trailingReserve = std::min(
            extentPadding,
            std::max(extentPadding * kStaticOnlyPredictiveTrailingGuardFraction,
                     sourceTexelWorldSize));
        const float maxShift = std::max(extentPadding - trailingReserve, 0.0f);
        const float desiredShift = (positiveMotion - negativeMotion) *
                                   kStaticOnlyPredictiveCenterMotionMultiplier;
        return std::clamp(desiredShift, -maxShift, maxShift);
      };

  const glm::vec3 predictiveShift(
      resolvePredictiveShift(
          fit.lightSpaceBoundsMin.x - previousRawFit.lightSpaceBoundsMin.x,
          fit.lightSpaceBoundsMax.x - previousRawFit.lightSpaceBoundsMax.x,
          predictiveOrthoPadding),
      resolvePredictiveShift(
          fit.lightSpaceBoundsMin.y - previousRawFit.lightSpaceBoundsMin.y,
          fit.lightSpaceBoundsMax.y - previousRawFit.lightSpaceBoundsMax.y,
          predictiveOrthoPadding),
      resolvePredictiveShift(
          fit.lightSpaceBoundsMin.z - previousRawFit.lightSpaceBoundsMin.z,
          fit.lightSpaceBoundsMax.z - previousRawFit.lightSpaceBoundsMax.z,
          predictiveDepthPadding));

  glm::vec3 lightMin = currentCenter + predictiveShift - currentExtent * 0.5f;
  glm::vec3 lightMax = currentCenter + predictiveShift + currentExtent * 0.5f;
  lightMin.x -= predictiveOrthoPadding;
  lightMax.x += predictiveOrthoPadding;
  lightMin.y -= predictiveOrthoPadding;
  lightMax.y += predictiveOrthoPadding;
  lightMin.z -= predictiveDepthPadding;
  lightMax.z += predictiveDepthPadding;

  const glm::mat4 inverseLightView = glm::inverse(fit.lightView);
  shadow_detail::stabilizeOrthoBounds(lightMin, lightMax, shadowMapSize, true,
                                      fit, inverseLightView);
  if (fit.texelWorldSize > 0.0f) {
    lightMin.z =
        shadow_detail::quantizeShadowBoundDown(lightMin.z, fit.texelWorldSize);
    lightMax.z =
        shadow_detail::quantizeShadowBoundUp(lightMax.z, fit.texelWorldSize);
  }

  const float nearPlane = -lightMax.z;
  float farPlane = -lightMin.z;
  if (farPlane <= nearPlane + 0.01f) {
    farPlane = nearPlane + 0.01f;
  }
  fit.lightProj = glm::orthoRH_ZO(lightMin.x, lightMax.x, lightMin.y,
                                  lightMax.y, nearPlane, farPlane);
  fit.lightViewProj = fit.lightProj * fit.lightView;
  fit.lightSpaceBoundsMin = glm::vec3(lightMin.x, lightMin.y, -farPlane);
  fit.lightSpaceBoundsMax = glm::vec3(lightMax.x, lightMax.y, -nearPlane);
}

[[nodiscard]] bool shadowCasterOverlapsDirectionalShadowFit(
    std::span<const glm::vec3, 8> casterWorldCorners,
    const shadow_detail::DirectionalShadowFit &fit) {
  const float cullingPadding =
      std::max(fit.texelWorldSize * kCullingPaddingTexelMultiplier, 0.01f);
  return shadow_detail::shadowCasterOverlapsLightSpaceBounds(
      casterWorldCorners, fit.lightView, fit.lightSpaceBoundsMin,
      fit.lightSpaceBoundsMax, cullingPadding);
}

[[nodiscard]] bool staticOnlyRenderedFitContainsCurrent(
    const shadow_detail::DirectionalShadowFit &renderedFit,
    const shadow_detail::DirectionalShadowFit &currentFit) {
  return shadow_detail::directionalShadowFitContains(renderedFit, currentFit) &&
         shadow_detail::nearlyEqualShadowValue(
             renderedFit.splitNear, currentFit.splitNear,
             shadow_detail::kShadowStabilizationCompatibilityEpsilon) &&
         shadow_detail::nearlyEqualShadowValue(
             renderedFit.splitFar, currentFit.splitFar,
             shadow_detail::kShadowStabilizationCompatibilityEpsilon);
}

void writeShadowCascadeFit(const shadow_detail::DirectionalShadowFit &fit,
                           uint32_t cascadeIndex,
                           const RenderSettings::ShadowSettings &settings,
                           uint32_t compareSamplerId, uint32_t rawSamplerId,
                           TextureHandle shadowDepthTexture, GPUDevice &gpu,
                           ShadowFrameGpuData &shadowFrameGpuData,
                           ShadowDebugFrameData &shadowDebugFrameData) {
  shadowFrameGpuData.cascades[cascadeIndex] = ShadowCascadeGpuData{
      .lightViewProj = fit.lightViewProj,
      .lightView = fit.lightView,
      .splitDepthTexelSize =
          glm::vec4(fit.splitNear, fit.splitFar, fit.texelWorldSize, 0.0f),
      .uvScaleBias = glm::vec4(1.0f, 1.0f, 0.0f, 0.0f),
      .biasParams = glm::vec4(settings.constantBias, settings.slopeBias,
                              settings.normalBias, 0.0f),
      .pcssParams = glm::vec4(
          std::max(fit.lightSpaceBoundsMax.z - fit.lightSpaceBoundsMin.z,
                   1.0e-6f),
          settings.pcssSearchRadiusClampTexels,
          settings.pcssFilterRadiusClampTexels, 0.0f),
      .textureSampler =
          glm::uvec4(gpu.getTextureBindlessIndex(shadowDepthTexture),
                     compareSamplerId, rawSamplerId, 0u),
  };

  ShadowCascadeDebugFrameData &cascadeDebug =
      shadowDebugFrameData.cascades[cascadeIndex];
  cascadeDebug.splitNear = fit.splitNear;
  cascadeDebug.splitFar = fit.splitFar;
  cascadeDebug.texelWorldSize = fit.texelWorldSize;
  cascadeDebug.lightView = fit.lightView;
  cascadeDebug.lightProj = fit.lightProj;
  cascadeDebug.lightViewProj = fit.lightViewProj;
  cascadeDebug.inverseLightView = glm::inverse(fit.lightView);
  cascadeDebug.inverseLightProj = glm::inverse(fit.lightProj);
  cascadeDebug.lightSpaceBoundsMin = glm::vec4(fit.lightSpaceBoundsMin, 1.0f);
  cascadeDebug.lightSpaceBoundsMax = glm::vec4(fit.lightSpaceBoundsMax, 1.0f);
  cascadeDebug.unsnappedCenter = glm::vec4(fit.unsnappedCenter, 1.0f);
  cascadeDebug.snappedCenter = glm::vec4(fit.snappedCenter, 1.0f);
  for (size_t i = 0u; i < fit.frustumCorners.size(); ++i) {
    cascadeDebug.worldFrustumCorners[i] =
        glm::vec4(fit.frustumCorners[i], 1.0f);
  }
}

[[nodiscard]] shadow_detail::DirectionalShadowFit
makeDirectionalShadowFitFromDebugCascade(
    const ShadowCascadeDebugFrameData &cascade) {
  shadow_detail::DirectionalShadowFit fit{};
  fit.splitNear = cascade.splitNear;
  fit.splitFar = cascade.splitFar;
  fit.texelWorldSize = cascade.texelWorldSize;
  fit.lightView = cascade.lightView;
  fit.lightProj = cascade.lightProj;
  fit.lightViewProj = cascade.lightViewProj;
  fit.lightSpaceBoundsMin = glm::vec3(cascade.lightSpaceBoundsMin);
  fit.lightSpaceBoundsMax = glm::vec3(cascade.lightSpaceBoundsMax);
  fit.unsnappedCenter = glm::vec3(cascade.unsnappedCenter);
  fit.snappedCenter = glm::vec3(cascade.snappedCenter);

  glm::vec3 frustumCenter(0.0f);
  for (size_t i = 0u; i < fit.frustumCorners.size(); ++i) {
    fit.frustumCorners[i] = glm::vec3(cascade.worldFrustumCorners[i]);
    frustumCenter += fit.frustumCorners[i];
  }
  fit.frustumCenter =
      frustumCenter / static_cast<float>(fit.frustumCorners.size());

  const glm::vec3 unsnappedLightSpaceCenter =
      glm::vec3(fit.lightView * glm::vec4(fit.unsnappedCenter, 1.0f));
  const glm::vec3 snappedLightSpaceCenter =
      glm::vec3(fit.lightView * glm::vec4(fit.snappedCenter, 1.0f));
  fit.unsnappedLightSpaceCenter =
      glm::vec2(unsnappedLightSpaceCenter.x, unsnappedLightSpaceCenter.y);
  fit.snappedLightSpaceCenter =
      glm::vec2(snappedLightSpaceCenter.x, snappedLightSpaceCenter.y);
  return fit;
}

[[nodiscard]] shadow_detail::DirectionalShadowFit fitSingleShadowMapForRange(
    const CameraFrameState &camera, ShadowSplitRange range,
    glm::vec3 lightDirection, uint32_t shadowMapSize, bool stabilize,
    bool hasCasterBounds, bool hasAnimatedGeometryOverrides,
    glm::vec3 casterBoundsMin, glm::vec3 casterBoundsMax) {
  const CameraFrameState adjustedCamera =
      cameraWithShadowSplitRange(camera, range);
  if (hasCasterBounds && !hasAnimatedGeometryOverrides) {
    return shadow_detail::fitDirectionalShadowMapToBounds(
        adjustedCamera, casterBoundsMin, casterBoundsMax, lightDirection,
        range.farDepth, shadowMapSize, stabilize);
  }
  return shadow_detail::fitSingleDirectionalShadowMap(
      adjustedCamera, lightDirection, range.farDepth, shadowMapSize, stabilize);
}

[[nodiscard]] std::optional<SubmeshLod>
resolveShadowLod(const Submesh &submesh, const RenderSettings &settings) {
  if (settings.opaque.forcedMeshLod < 0) {
    if (submesh.indexCount > 0u) {
      return SubmeshLod{.indexOffset = submesh.indexOffset,
                        .indexCount = submesh.indexCount,
                        .error = 0.0f};
    }
    for (uint32_t lod = 0u; lod < std::max(submesh.lodCount, 1u); ++lod) {
      if (submesh.lods[lod].indexCount > 0u) {
        return submesh.lods[lod];
      }
    }
    return std::nullopt;
  }

  const uint32_t lodCount =
      std::clamp(submesh.lodCount, 1u, Submesh::kMaxLodCount);
  uint32_t candidate = std::min(
      static_cast<uint32_t>(settings.opaque.forcedMeshLod), lodCount - 1u);
  while (candidate > 0u && submesh.lods[candidate].indexCount == 0u) {
    --candidate;
  }
  if (submesh.lods[candidate].indexCount > 0u) {
    return submesh.lods[candidate];
  }
  if (submesh.indexCount > 0u) {
    return SubmeshLod{.indexOffset = submesh.indexOffset,
                      .indexCount = submesh.indexCount,
                      .error = 0.0f};
  }
  return std::nullopt;
}

[[nodiscard]] RenderPipelineDesc
shadowDepthPipelineDesc(ShaderHandle vertexShader, ShaderHandle fragmentShader,
                        CullMode cullMode) {
  return RenderPipelineDesc{
      .vertexInput = {},
      .vertexShader = vertexShader,
      .fragmentShader = fragmentShader,
      .colorFormats = {Format::RGBA8_UNORM},
      .colorAttachmentCount = 0u,
      .depthFormat = kDefaultShadowMapDepthFormat,
      .cullMode = cullMode,
      .polygonMode = PolygonMode::Fill,
      .topology = Topology::Triangle,
      .blendEnabled = false,
  };
}

[[nodiscard]] RenderPipelineDesc
shadowPreviewPipelineDesc(ShaderHandle vertexShader,
                          ShaderHandle fragmentShader) {
  return RenderPipelineDesc{
      .vertexInput = {},
      .vertexShader = vertexShader,
      .fragmentShader = fragmentShader,
      .colorFormats = {Format::RGBA8_UNORM},
      .depthFormat = Format::Count,
      .cullMode = CullMode::None,
      .polygonMode = PolygonMode::Fill,
      .topology = Topology::Triangle,
      .blendEnabled = false,
  };
}

[[nodiscard]] uint32_t saturateToU32(size_t value) {
  return static_cast<uint32_t>(
      std::min(value, size_t(std::numeric_limits<uint32_t>::max())));
}

[[nodiscard]] bool isValidShadowDepthTexture(const GPUDevice &gpu,
                                             TextureHandle texture,
                                             uint32_t shadowMapSize) {
  if (!nuri::isValid(texture)) {
    return false;
  }
  const TextureDimensions dimensions = gpu.getTextureDimensions(texture);
  return dimensions.width == shadowMapSize &&
         dimensions.height == shadowMapSize &&
         gpu.getTextureFormat(texture) == kDefaultShadowMapDepthFormat;
}

[[nodiscard]] TextureDimensions
shadowPreviewDimensions(uint32_t shadowMapSize, ShadowPreviewMode mode) {
  if (sanitizeShadowPreviewMode(mode) == ShadowPreviewMode::TiledAllCascades) {
    return TextureDimensions{
        .width = shadowMapSize * 2u,
        .height = shadowMapSize * 2u,
        .depth = 1u,
    };
  }
  return TextureDimensions{
      .width = shadowMapSize,
      .height = shadowMapSize,
      .depth = 1u,
  };
}

[[nodiscard]] bool isValidShadowPreviewTexture(const GPUDevice &gpu,
                                               TextureHandle texture,
                                               uint32_t shadowMapSize,
                                               ShadowPreviewMode mode) {
  if (!nuri::isValid(texture) ||
      gpu.getTextureFormat(texture) != Format::RGBA8_UNORM) {
    return false;
  }
  const TextureDimensions dimensions = gpu.getTextureDimensions(texture);
  const TextureDimensions expected =
      shadowPreviewDimensions(shadowMapSize, mode);
  return dimensions.width == expected.width &&
         dimensions.height == expected.height;
}

[[nodiscard]] uint64_t
makeStaticOnlyCascadeRasterSignatureSeed(const ShadowCascadeGpuData &cascade,
                                         float constantBias, float slopeBias) {
  const uint64_t lightViewProjSignature = hashBytes(
      std::as_bytes(std::span<const glm::mat4>(&cascade.lightViewProj, 1u)));
  uint64_t biasSignature = kFnvOffsetBasis64;
  biasSignature = hashCombineValue(biasSignature, constantBias);
  biasSignature = hashCombineValue(biasSignature, slopeBias);
  uint64_t signature = kFnvOffsetBasis64;
  signature = hashCombine64(signature, lightViewProjSignature);
  signature = hashCombine64(signature, biasSignature);
  return signature;
}

[[nodiscard]] uint64_t makeStaticOnlyCascadeLightViewProjSignature(
    const ShadowCascadeGpuData &cascade) {
  return hashBytes(
      std::as_bytes(std::span<const glm::mat4>(&cascade.lightViewProj, 1u)));
}

[[nodiscard]] uint64_t makeStaticOnlyCascadeBiasSignature(float constantBias,
                                                          float slopeBias) {
  uint64_t signature = kFnvOffsetBasis64;
  signature = hashCombineValue(signature, constantBias);
  signature = hashCombineValue(signature, slopeBias);
  return signature;
}

template <typename Entry>
[[nodiscard]] uint64_t
hashStaticShadowCasterRasterSignature(uint64_t signature, const Entry &entry) {
  signature =
      hashCombine64(signature, static_cast<uint64_t>(entry.templateIndex));
  signature =
      hashCombine64(signature, static_cast<uint64_t>(entry.instanceIndex));
  signature =
      hashCombine64(signature, foldHandle(entry.indexBuffer.index,
                                          entry.indexBuffer.generation));
  signature = hashCombineValue(signature, entry.indexBufferOffset);
  signature =
      hashCombine64(signature, static_cast<uint64_t>(entry.indexFormat));
  signature =
      hashCombine64(signature, foldHandle(entry.vertexBuffer.index,
                                          entry.vertexBuffer.generation));
  signature = hashCombineValue(signature, entry.vertexBufferAddress);
  signature = hashCombineValue(signature, entry.vertexDecodeBufferAddress);
  signature =
      hashCombine64(signature, static_cast<uint64_t>(entry.vertexDecodeIndex));
  signature =
      hashCombine64(signature, static_cast<uint64_t>(entry.packedVertexFormat));
  signature =
      hashCombine64(signature, static_cast<uint64_t>(entry.materialIndex));
  signature = hashCombine64(signature, static_cast<uint64_t>(entry.indexCount));
  signature = hashCombine64(signature, static_cast<uint64_t>(entry.firstIndex));
  signature = hashCombine64(signature, entry.doubleSided ? 1ull : 0ull);
  signature = hashCombine64(signature, entry.alphaMasked ? 1ull : 0ull);
  return signature;
}

} // namespace

ShadowRenderer::ShadowRenderer(GPUDevice &gpu,
                               std::pmr::memory_resource *memory)
    : ShadowRenderer(gpu, ShadowRendererConfig{}, memory) {}

ShadowRenderer::ShadowRenderer(GPUDevice &gpu,
                               const ShadowRendererConfig &config,
                               std::pmr::memory_resource *memory)
    : gpu_(gpu), config_(config), memory_(resolveMemoryResource(memory)),
      instanceMatricesRing_(memory_), instanceRemapRing_(memory_),
      shadowFrameRing_(memory_), sdsmReduceResultRing_(memory_),
      instanceDataRingUploadVersions_(memory_),
      shadowFrameUploadSignatures_(memory_),
      sdsmReduceResultRingPublishedFrames_(memory_),
      meshDrawTemplates_(memory_), staticShadowTemplateIndices_(memory_),
      dynamicShadowTemplateIndices_(memory_), staticShadowCasterCache_(memory_),
      instanceMatrices_(memory_), instanceRemap_(memory_),
      passBufferDependencies_(memory_), passTextureDependencies_(memory_),
      previewTextureDependencies_(memory_), sdsmReadbackBuffer_(memory_) {
  sdsmReadbackBuffer_.resize(
      static_cast<size_t>(kSdsmHistogramSourceMaxTexelCount) * sizeof(float) *
      2u);
  for (uint32_t cascadeIndex = 0u; cascadeIndex < kMaxShadowCascades;
       ++cascadeIndex) {
    cascadePushConstants_[cascadeIndex] =
        std::pmr::vector<PushConstants>(memory_);
    cascadeDrawItems_[cascadeIndex] = std::pmr::vector<DrawItem>(memory_);
  }
}

ShadowRenderer::~ShadowRenderer() {
  destroyShadowResources();
  destroyBuffers();
  destroyPipelineState();
  destroyShaders();
}

Result<bool, std::string> ShadowRenderer::createShaders() {
  destroyShaders();
  if (config_.shaderBasePath.empty()) {
    return Result<bool, std::string>::makeError(
        "ShadowRenderer::createShaders: shader base path is not configured");
  }

  shadowShader_ = Shader::create("shadow_depth", gpu_);
  depthShader_ = Shader::create("shadow_depth_only", gpu_);
  depthAlphaShader_ = Shader::create("shadow_depth_alpha", gpu_);
  if (!shadowShader_ || !depthShader_ || !depthAlphaShader_) {
    return Result<bool, std::string>::makeError(
        "ShadowRenderer::createShaders: failed to create shader wrappers");
  }

  const std::filesystem::path shadowVertexPath =
      config_.shaderBasePath / "shadow_depth.vert";
  const std::filesystem::path depthFragmentPath =
      config_.shaderBasePath / "opaque_depth.frag";
  const std::filesystem::path depthAlphaFragmentPath =
      config_.shaderBasePath / "opaque_depth_alpha.frag";

  auto vertexResult = shadowShader_->compileFromFile(shadowVertexPath.string(),
                                                     ShaderStage::Vertex);
  if (vertexResult.hasError()) {
    return Result<bool, std::string>::makeError(vertexResult.error());
  }
  auto fragmentResult = depthShader_->compileFromFile(
      depthFragmentPath.string(), ShaderStage::Fragment);
  if (fragmentResult.hasError()) {
    return Result<bool, std::string>::makeError(fragmentResult.error());
  }
  auto alphaFragmentResult = depthAlphaShader_->compileFromFile(
      depthAlphaFragmentPath.string(), ShaderStage::Fragment);
  if (alphaFragmentResult.hasError()) {
    return Result<bool, std::string>::makeError(alphaFragmentResult.error());
  }

  shadowVertexShader_ = vertexResult.value();
  depthFragmentShader_ = fragmentResult.value();
  depthAlphaFragmentShader_ = alphaFragmentResult.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> ShadowRenderer::createPreviewShaders() {
  if (nuri::isValid(previewVertexShader_)) {
    gpu_.destroyShaderModule(previewVertexShader_);
    previewVertexShader_ = {};
  }
  if (nuri::isValid(previewFragmentShader_)) {
    gpu_.destroyShaderModule(previewFragmentShader_);
    previewFragmentShader_ = {};
  }

  auto vertexResult =
      gpu_.createShaderModule(ShaderDesc{.moduleName = "shadow_preview_vs",
                                         .source = kShadowPreviewVS,
                                         .stage = ShaderStage::Vertex});
  if (vertexResult.hasError()) {
    return Result<bool, std::string>::makeError(vertexResult.error());
  }

  auto fragmentResult =
      gpu_.createShaderModule(ShaderDesc{.moduleName = "shadow_preview_fs",
                                         .source = kShadowPreviewFS,
                                         .stage = ShaderStage::Fragment});
  if (fragmentResult.hasError()) {
    gpu_.destroyShaderModule(vertexResult.value());
    return Result<bool, std::string>::makeError(fragmentResult.error());
  }

  previewVertexShader_ = vertexResult.value();
  previewFragmentShader_ = fragmentResult.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> ShadowRenderer::createSdsmReduceShaders() {
  if (sdsmReduceShader_) {
    sdsmReduceShader_.reset();
  }
  sdsmReduceComputeShader_ = {};

  if (config_.shaderBasePath.empty()) {
    return Result<bool, std::string>::makeError(
        "ShadowRenderer::createSdsmReduceShaders: shader base path is not "
        "configured");
  }

  sdsmReduceShader_ = Shader::create("shadow_sdsm_prev_frame_minmax", gpu_);
  if (!sdsmReduceShader_) {
    return Result<bool, std::string>::makeError(
        "ShadowRenderer::createSdsmReduceShaders: failed to create shader "
        "wrapper");
  }

  const std::filesystem::path computePath =
      config_.shaderBasePath / "shadow_sdsm_prev_frame_minmax.comp";
  auto computeResult = sdsmReduceShader_->compileFromFile(computePath.string(),
                                                          ShaderStage::Compute);
  if (computeResult.hasError()) {
    return Result<bool, std::string>::makeError(computeResult.error());
  }
  sdsmReduceComputeShader_ = computeResult.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> ShadowRenderer::createSdsmHistogramReduceShaders() {
  if (sdsmHistogramReduceShader_) {
    sdsmHistogramReduceShader_.reset();
  }
  sdsmHistogramReduceComputeShader_ = {};

  if (config_.shaderBasePath.empty()) {
    return Result<bool, std::string>::makeError(
        "ShadowRenderer::createSdsmHistogramReduceShaders: shader base path "
        "is not configured");
  }

  sdsmHistogramReduceShader_ = Shader::create("shadow_sdsm_histogram", gpu_);
  if (!sdsmHistogramReduceShader_) {
    return Result<bool, std::string>::makeError(
        "ShadowRenderer::createSdsmHistogramReduceShaders: failed to create "
        "shader wrapper");
  }

  const std::filesystem::path computePath =
      config_.shaderBasePath / "shadow_sdsm_histogram.comp";
  auto computeResult = sdsmHistogramReduceShader_->compileFromFile(
      computePath.string(), ShaderStage::Compute);
  if (computeResult.hasError()) {
    return Result<bool, std::string>::makeError(computeResult.error());
  }
  sdsmHistogramReduceComputeShader_ = computeResult.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> ShadowRenderer::createPipelines() {
  destroyPipelineState();
  auto shadowResult = gpu_.createRenderPipeline(
      shadowDepthPipelineDesc(shadowVertexShader_, depthFragmentShader_,
                              CullMode::Back),
      "shadow_depth_opaque");
  if (shadowResult.hasError()) {
    return Result<bool, std::string>::makeError(shadowResult.error());
  }
  shadowPipelineHandle_ = shadowResult.value();

  auto doubleSidedResult = gpu_.createRenderPipeline(
      shadowDepthPipelineDesc(shadowVertexShader_, depthFragmentShader_,
                              CullMode::None),
      "shadow_depth_opaque_double_sided");
  if (doubleSidedResult.hasError()) {
    gpu_.destroyRenderPipeline(shadowPipelineHandle_);
    shadowPipelineHandle_ = {};
    return Result<bool, std::string>::makeError(doubleSidedResult.error());
  }
  shadowDoubleSidedPipelineHandle_ = doubleSidedResult.value();

  auto alphaResult = gpu_.createRenderPipeline(
      shadowDepthPipelineDesc(shadowVertexShader_, depthAlphaFragmentShader_,
                              CullMode::Back),
      "shadow_depth_alpha");
  if (alphaResult.hasError()) {
    gpu_.destroyRenderPipeline(shadowDoubleSidedPipelineHandle_);
    gpu_.destroyRenderPipeline(shadowPipelineHandle_);
    shadowDoubleSidedPipelineHandle_ = {};
    shadowPipelineHandle_ = {};
    return Result<bool, std::string>::makeError(alphaResult.error());
  }
  shadowAlphaPipelineHandle_ = alphaResult.value();

  auto alphaDoubleSidedResult = gpu_.createRenderPipeline(
      shadowDepthPipelineDesc(shadowVertexShader_, depthAlphaFragmentShader_,
                              CullMode::None),
      "shadow_depth_alpha_double_sided");
  if (alphaDoubleSidedResult.hasError()) {
    gpu_.destroyRenderPipeline(shadowAlphaPipelineHandle_);
    gpu_.destroyRenderPipeline(shadowDoubleSidedPipelineHandle_);
    gpu_.destroyRenderPipeline(shadowPipelineHandle_);
    shadowAlphaPipelineHandle_ = {};
    shadowDoubleSidedPipelineHandle_ = {};
    shadowPipelineHandle_ = {};
    return Result<bool, std::string>::makeError(alphaDoubleSidedResult.error());
  }
  shadowAlphaDoubleSidedPipelineHandle_ = alphaDoubleSidedResult.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> ShadowRenderer::createPreviewPipeline() {
  if (nuri::isValid(previewPipelineHandle_)) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (!nuri::isValid(previewVertexShader_) ||
      !nuri::isValid(previewFragmentShader_)) {
    return Result<bool, std::string>::makeError(
        "ShadowRenderer::createPreviewPipeline: invalid preview shader handle");
  }
  auto pipelineResult = gpu_.createRenderPipeline(
      shadowPreviewPipelineDesc(previewVertexShader_, previewFragmentShader_),
      "shadow_depth_preview");
  if (pipelineResult.hasError()) {
    return Result<bool, std::string>::makeError(pipelineResult.error());
  }
  previewPipelineHandle_ = pipelineResult.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> ShadowRenderer::createSdsmReducePipeline() {
  if (nuri::isValid(sdsmReducePipelineHandle_)) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (!nuri::isValid(sdsmReduceComputeShader_)) {
    return Result<bool, std::string>::makeError(
        "ShadowRenderer::createSdsmReducePipeline: invalid compute shader");
  }
  auto pipelineResult = gpu_.createComputePipeline(
      ComputePipelineDesc{.computeShader = sdsmReduceComputeShader_},
      "shadow_sdsm_prev_frame_minmax");
  if (pipelineResult.hasError()) {
    return Result<bool, std::string>::makeError(pipelineResult.error());
  }
  sdsmReducePipelineHandle_ = pipelineResult.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> ShadowRenderer::createSdsmHistogramReducePipeline() {
  if (nuri::isValid(sdsmHistogramReducePipelineHandle_)) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (!nuri::isValid(sdsmHistogramReduceComputeShader_)) {
    return Result<bool, std::string>::makeError(
        "ShadowRenderer::createSdsmHistogramReducePipeline: invalid compute "
        "shader");
  }
  auto pipelineResult = gpu_.createComputePipeline(
      ComputePipelineDesc{.computeShader = sdsmHistogramReduceComputeShader_},
      "shadow_sdsm_histogram");
  if (pipelineResult.hasError()) {
    return Result<bool, std::string>::makeError(pipelineResult.error());
  }
  sdsmHistogramReducePipelineHandle_ = pipelineResult.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> ShadowRenderer::ensureSdsmReduceResources() {
  if (!nuri::isValid(sdsmReduceComputeShader_)) {
    auto shaderResult = createSdsmReduceShaders();
    if (shaderResult.hasError()) {
      return shaderResult;
    }
  }
  if (!nuri::isValid(sdsmReducePipelineHandle_)) {
    auto pipelineResult = createSdsmReducePipeline();
    if (pipelineResult.hasError()) {
      return pipelineResult;
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> ShadowRenderer::ensureSdsmHistogramReduceResources() {
  if (!nuri::isValid(sdsmHistogramReduceComputeShader_)) {
    auto shaderResult = createSdsmHistogramReduceShaders();
    if (shaderResult.hasError()) {
      return shaderResult;
    }
  }
  if (!nuri::isValid(sdsmHistogramReducePipelineHandle_)) {
    auto pipelineResult = createSdsmHistogramReducePipeline();
    if (pipelineResult.hasError()) {
      return pipelineResult;
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> ShadowRenderer::ensureInitialized() {
  if (initialized_) {
    return Result<bool, std::string>::makeResult(true);
  }
  auto shaderResult = createShaders();
  if (shaderResult.hasError()) {
    return shaderResult;
  }
  auto pipelineResult = createPipelines();
  if (pipelineResult.hasError()) {
    return pipelineResult;
  }
  initialized_ = true;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> ShadowRenderer::ensureShadowResources(
    const RenderSettings::ShadowSettings &settings) {
  const bool previewEnabled = settings.debug.showShadowMapViewport;
  const ShadowPreviewMode previewMode =
      sanitizeShadowPreviewMode(settings.debug.previewMode);
  const uint32_t requestedCascadeCount =
      std::clamp(settings.cascadeCount, 1u, kMaxShadowCascades);
  bool depthValid = nuri::isValid(rawDepthSampler_) &&
                    nuri::isValid(compareDepthSampler_) &&
                    activeCascadeCount_ == requestedCascadeCount;
  for (uint32_t cascadeIndex = 0u;
       depthValid && cascadeIndex < requestedCascadeCount; ++cascadeIndex) {
    depthValid = isValidShadowDepthTexture(
        gpu_, shadowDepthTextures_[cascadeIndex], settings.shadowMapSize);
  }
  for (uint32_t cascadeIndex = requestedCascadeCount;
       depthValid && cascadeIndex < kMaxShadowCascades; ++cascadeIndex) {
    depthValid = !nuri::isValid(shadowDepthTextures_[cascadeIndex]);
  }
  const bool previewTextureValid = isValidShadowPreviewTexture(
      gpu_, shadowDebugPreviewTexture_, settings.shadowMapSize, previewMode);
  const bool previewValid = previewEnabled
                                ? previewTextureValid
                                : !nuri::isValid(shadowDebugPreviewTexture_);
  if (depthValid && previewValid) {
    return Result<bool, std::string>::makeResult(true);
  }

  if (depthValid) {
    if (previewEnabled && !previewTextureValid) {
      gpu_.waitIdle();
      const TextureHandle oldPreviewTexture = shadowDebugPreviewTexture_;
      const TextureDimensions previewDimensions =
          shadowPreviewDimensions(settings.shadowMapSize, previewMode);
      const TextureDesc previewDesc{
          .type = TextureType::Texture2D,
          .format = Format::RGBA8_UNORM,
          .dimensions = previewDimensions,
          .usage = TextureUsage::AttachmentSampled,
          .storage = Storage::Device,
          .numLayers = 1u,
          .numSamples = 1u,
          .numMipLevels = 1u,
          .data = {},
          .dataNumMipLevels = 1u,
          .generateMipmaps = false,
      };
      auto previewResult =
          gpu_.createTexture(previewDesc, "shadow_depth_preview_phase4");
      if (previewResult.hasError()) {
        return Result<bool, std::string>::makeError(previewResult.error());
      }
      shadowDebugPreviewTexture_ = previewResult.value();
      if (nuri::isValid(oldPreviewTexture)) {
        gpu_.destroyTexture(oldPreviewTexture);
      }
      return Result<bool, std::string>::makeResult(true);
    }
    if (!previewEnabled && nuri::isValid(shadowDebugPreviewTexture_)) {
      gpu_.waitIdle();
      gpu_.destroyTexture(shadowDebugPreviewTexture_);
      shadowDebugPreviewTexture_ = {};
      return Result<bool, std::string>::makeResult(true);
    }
  }

  bool hasLiveDepthTextures = false;
  for (const TextureHandle texture : shadowDepthTextures_) {
    hasLiveDepthTextures = hasLiveDepthTextures || nuri::isValid(texture);
  }
  const bool hasLiveResources =
      hasLiveDepthTextures || nuri::isValid(shadowDebugPreviewTexture_) ||
      nuri::isValid(rawDepthSampler_) || nuri::isValid(compareDepthSampler_);
  if (hasLiveResources) {
    gpu_.waitIdle();
  }

  std::array<TextureHandle, kMaxShadowCascades> newShadowDepthTextures{};
  for (uint32_t cascadeIndex = 0u; cascadeIndex < requestedCascadeCount;
       ++cascadeIndex) {
    const TextureDesc shadowDesc{
        .type = TextureType::Texture2D,
        .format = kDefaultShadowMapDepthFormat,
        .dimensions = {.width = settings.shadowMapSize,
                       .height = settings.shadowMapSize,
                       .depth = 1u},
        .usage = TextureUsage::AttachmentSampled,
        .storage = Storage::Device,
        .numLayers = 1u,
        .numSamples = 1u,
        .numMipLevels = 1u,
        .data = {},
        .dataNumMipLevels = 1u,
        .generateMipmaps = false,
    };
    const std::string debugName =
        "shadow_depth_map_phase4_cascade" + std::to_string(cascadeIndex);
    auto textureResult = gpu_.createTexture(shadowDesc, debugName);
    if (textureResult.hasError()) {
      for (TextureHandle texture : newShadowDepthTextures) {
        if (nuri::isValid(texture)) {
          gpu_.destroyTexture(texture);
        }
      }
      return Result<bool, std::string>::makeError(textureResult.error());
    }
    newShadowDepthTextures[cascadeIndex] = textureResult.value();
  }

  const SamplerDesc rawSamplerDesc{
      .minFilter = SamplerFilter::Nearest,
      .magFilter = SamplerFilter::Nearest,
      .mipMode = SamplerMipMode::Disabled,
      .wrapU = SamplerWrapMode::Clamp,
      .wrapV = SamplerWrapMode::Clamp,
      .wrapW = SamplerWrapMode::Clamp,
      .mipLodMin = 0.0f,
      .mipLodMax = 0.0f,
      .maxAnisotropy = 1u,
      .depthCompareEnabled = false,
      .depthCompareOp = CompareOp::LessEqual,
  };
  auto samplerResult =
      gpu_.createSampler(rawSamplerDesc, "shadow_raw_depth_sampler");
  if (samplerResult.hasError()) {
    for (TextureHandle texture : newShadowDepthTextures) {
      if (nuri::isValid(texture)) {
        gpu_.destroyTexture(texture);
      }
    }
    return Result<bool, std::string>::makeError(samplerResult.error());
  }
  const SamplerHandle newRawDepthSampler = samplerResult.value();

  SamplerDesc compareSamplerDesc = rawSamplerDesc;
  compareSamplerDesc.depthCompareEnabled = true;
  compareSamplerDesc.depthCompareOp = CompareOp::LessEqual;
  auto compareSamplerResult =
      gpu_.createSampler(compareSamplerDesc, "shadow_compare_depth_sampler");
  if (compareSamplerResult.hasError()) {
    gpu_.destroySampler(newRawDepthSampler);
    for (TextureHandle texture : newShadowDepthTextures) {
      if (nuri::isValid(texture)) {
        gpu_.destroyTexture(texture);
      }
    }
    return Result<bool, std::string>::makeError(compareSamplerResult.error());
  }
  const SamplerHandle newCompareDepthSampler = compareSamplerResult.value();

  TextureHandle newPreviewTexture{};
  if (previewEnabled) {
    const TextureDimensions previewDimensions =
        shadowPreviewDimensions(settings.shadowMapSize, previewMode);
    const TextureDesc previewDesc{
        .type = TextureType::Texture2D,
        .format = Format::RGBA8_UNORM,
        .dimensions = previewDimensions,
        .usage = TextureUsage::AttachmentSampled,
        .storage = Storage::Device,
        .numLayers = 1u,
        .numSamples = 1u,
        .numMipLevels = 1u,
        .data = {},
        .dataNumMipLevels = 1u,
        .generateMipmaps = false,
    };
    auto previewResult =
        gpu_.createTexture(previewDesc, "shadow_depth_preview_phase4");
    if (previewResult.hasError()) {
      gpu_.destroySampler(newCompareDepthSampler);
      gpu_.destroySampler(newRawDepthSampler);
      for (TextureHandle texture : newShadowDepthTextures) {
        if (nuri::isValid(texture)) {
          gpu_.destroyTexture(texture);
        }
      }
      return Result<bool, std::string>::makeError(previewResult.error());
    }
    newPreviewTexture = previewResult.value();
  }

  const std::array<TextureHandle, kMaxShadowCascades> oldShadowDepthTextures =
      shadowDepthTextures_;
  const TextureHandle oldPreviewTexture = shadowDebugPreviewTexture_;
  const SamplerHandle oldRawDepthSampler = rawDepthSampler_;
  const SamplerHandle oldCompareDepthSampler = compareDepthSampler_;

  shadowDepthTextures_ = newShadowDepthTextures;
  shadowDebugPreviewTexture_ = newPreviewTexture;
  rawDepthSampler_ = newRawDepthSampler;
  compareDepthSampler_ = newCompareDepthSampler;
  shadowMapSize_ = settings.shadowMapSize;
  activeCascadeCount_ = requestedCascadeCount;

  if (nuri::isValid(oldCompareDepthSampler)) {
    gpu_.destroySampler(oldCompareDepthSampler);
  }
  if (nuri::isValid(oldRawDepthSampler)) {
    gpu_.destroySampler(oldRawDepthSampler);
  }
  if (nuri::isValid(oldPreviewTexture)) {
    gpu_.destroyTexture(oldPreviewTexture);
  }
  for (const TextureHandle texture : oldShadowDepthTextures) {
    if (nuri::isValid(texture)) {
      gpu_.destroyTexture(texture);
    }
  }
  resetFrozenShadowFit();
  resetCascadeStabilizationHistory();

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
ShadowRenderer::ensureRingBufferCount(uint32_t requiredCount) {
  const uint32_t safeCount = std::max(requiredCount, 1u);
  while (instanceMatricesRing_.size() < safeCount) {
    instanceMatricesRing_.push_back(DynamicBufferSlot{});
  }
  while (instanceRemapRing_.size() < safeCount) {
    instanceRemapRing_.push_back(DynamicBufferSlot{});
  }
  while (shadowFrameRing_.size() < safeCount) {
    shadowFrameRing_.push_back(DynamicBufferSlot{});
  }
  instanceDataRingUploadVersions_.resize(safeCount,
                                         std::numeric_limits<uint64_t>::max());
  shadowFrameUploadSignatures_.resize(safeCount,
                                      std::numeric_limits<uint64_t>::max());
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> ShadowRenderer::ensureRingCapacity(
    std::pmr::vector<DynamicBufferSlot> &ring, size_t requiredBytes,
    std::string_view debugName, std::span<uint64_t> uploadVersions) {
  NURI_ASSERT(uploadVersions.size() >= ring.size(),
              "ShadowRenderer::ensureRingCapacity: upload version count "
              "must cover ring slot count");

  bool needsResize = false;
  bool hasLiveBuffers = false;
  for (const DynamicBufferSlot &slot : ring) {
    if (slot.buffer && slot.buffer->valid()) {
      hasLiveBuffers = true;
      if (slot.capacityBytes < requiredBytes) {
        needsResize = true;
        break;
      }
    }
  }
  if (needsResize && hasLiveBuffers) {
    gpu_.waitIdle();
  }
  for (size_t i = 0; i < ring.size(); ++i) {
    DynamicBufferSlot &slot = ring[i];
    if (slot.buffer && slot.buffer->valid() &&
        slot.capacityBytes >= requiredBytes) {
      continue;
    }
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
    slot.buffer.reset();
    auto bufferResult = Buffer::create(gpu_,
                                       BufferDesc{.usage = BufferUsage::Storage,
                                                  .storage = Storage::Device,
                                                  .size = requiredBytes},
                                       debugName);
    if (bufferResult.hasError()) {
      return Result<bool, std::string>::makeError(bufferResult.error());
    }
    slot.buffer = std::move(bufferResult.value());
    slot.capacityBytes = requiredBytes;
    uploadVersions[i] = std::numeric_limits<uint64_t>::max();
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
ShadowRenderer::ensureInstanceMatricesRingCapacity(size_t requiredBytes) {
  return ensureRingCapacity(instanceMatricesRing_, requiredBytes,
                            "shadow_instance_matrices",
                            instanceDataRingUploadVersions_);
}

Result<bool, std::string>
ShadowRenderer::ensureInstanceRemapRingCapacity(size_t requiredBytes) {
  return ensureRingCapacity(instanceRemapRing_, requiredBytes,
                            "shadow_instance_remap",
                            instanceDataRingUploadVersions_);
}

Result<bool, std::string>
ShadowRenderer::ensureShadowFrameRingCapacity(size_t requiredBytes) {
  return ensureRingCapacity(shadowFrameRing_, requiredBytes,
                            "shadow_frame_gpu_data",
                            shadowFrameUploadSignatures_);
}

Result<bool, std::string>
ShadowRenderer::ensureSdsmReduceResultRingCount(uint32_t requiredCount) {
  const uint32_t safeCount = std::max(requiredCount, 1u);
  const uint64_t invalidPublishedFrame = std::numeric_limits<uint64_t>::max();
  while (sdsmReduceResultRing_.size() < safeCount) {
    sdsmReduceResultRing_.push_back(DynamicBufferSlot{});
  }
  sdsmReduceResultRingPublishedFrames_.resize(safeCount, invalidPublishedFrame);
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
ShadowRenderer::ensureSdsmReduceResultRingCapacity(size_t requiredBytes) {
  const uint64_t invalidPublishedFrame = std::numeric_limits<uint64_t>::max();
  bool needsResize = false;
  bool hasLiveBuffers = false;
  for (const DynamicBufferSlot &slot : sdsmReduceResultRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      hasLiveBuffers = true;
      if (slot.capacityBytes < requiredBytes) {
        needsResize = true;
        break;
      }
    }
  }
  if (needsResize && hasLiveBuffers) {
    gpu_.waitIdle();
  }
  for (size_t slotIndex = 0u; slotIndex < sdsmReduceResultRing_.size();
       ++slotIndex) {
    DynamicBufferSlot &slot = sdsmReduceResultRing_[slotIndex];
    if (slot.buffer && slot.buffer->valid() &&
        slot.capacityBytes >= requiredBytes) {
      continue;
    }
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
    slot.buffer.reset();
    auto bufferResult =
        Buffer::create(gpu_,
                       BufferDesc{.usage = BufferUsage::Storage,
                                  .storage = Storage::HostVisible,
                                  .size = requiredBytes},
                       "shadow_sdsm_minmax_result");
    if (bufferResult.hasError()) {
      return Result<bool, std::string>::makeError(bufferResult.error());
    }
    slot.buffer = std::move(bufferResult.value());
    slot.capacityBytes = requiredBytes;

    SdsmGpuMinMaxResult clearedResult{};
    auto clearResult = gpu_.updateBuffer(
        slot.buffer->handle(),
        std::as_bytes(std::span<const SdsmGpuMinMaxResult>(&clearedResult, 1u)),
        0u);
    if (clearResult.hasError()) {
      return Result<bool, std::string>::makeError(clearResult.error());
    }
    if (slotIndex < sdsmReduceResultRingPublishedFrames_.size()) {
      sdsmReduceResultRingPublishedFrames_[slotIndex] = invalidPublishedFrame;
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
ShadowRenderer::rebuildSceneCache(const RenderScene &scene,
                                  const ResourceManager &resources,
                                  uint32_t materialCount) {
  (void)materialCount;
  meshDrawTemplates_.clear();
  staticShadowTemplateIndices_.clear();
  dynamicShadowTemplateIndices_.clear();
  passTextureDependencies_.clear();

  const std::span<const Renderable> renderables = scene.renderables();
  if (renderables.size() >
      static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    return Result<bool, std::string>::makeError(
        "ShadowRenderer::rebuildSceneCache: renderables count exceeds "
        "UINT32_MAX");
  }

  for (uint32_t renderableIndex = 0u;
       renderableIndex < static_cast<uint32_t>(renderables.size());
       ++renderableIndex) {
    const Renderable &renderable = renderables[renderableIndex];
    const bool dynamicCaster =
        !renderable.morphWeights.empty() || !renderable.skinPalette.empty();
    const ModelRecord *modelRecord = resources.tryGet(renderable.model);
    if (!modelRecord || !modelRecord->model) {
      return Result<bool, std::string>::makeError(
          "ShadowRenderer::rebuildSceneCache: failed to resolve model");
    }

    GeometryAllocationView geometry{};
    if (!gpu_.resolveGeometry(modelRecord->model->geometryHandle(), geometry)) {
      return Result<bool, std::string>::makeError(
          "ShadowRenderer::rebuildSceneCache: failed to resolve geometry");
    }
    const uint64_t vertexBufferAddress = gpu_.getBufferDeviceAddress(
        geometry.vertexBuffer, geometry.vertexByteOffset);
    if (vertexBufferAddress == 0u) {
      return Result<bool, std::string>::makeError(
          "ShadowRenderer::rebuildSceneCache: invalid vertex buffer address");
    }

    const std::span<const Submesh> submeshes = modelRecord->model->submeshes();
    for (uint32_t submeshIndex = 0u;
         submeshIndex < static_cast<uint32_t>(submeshes.size());
         ++submeshIndex) {
      const Submesh &submesh = submeshes[submeshIndex];
      const MaterialRef resolvedMaterial =
          resolveRenderableMaterial(renderable, *modelRecord, submeshIndex);
      const MaterialRecord *materialRecord = resources.tryGet(resolvedMaterial);
      if (materialRecord != nullptr &&
          materialRecord->desc.alphaMode == MaterialAlphaMode::Blend) {
        continue;
      }
      const bool alphaMasked =
          materialRecord != nullptr &&
          materialRecord->desc.alphaMode == MaterialAlphaMode::Mask;
      if (alphaMasked) {
        const TextureRecord *baseColor =
            resources.tryGet(materialRecord->textureRefs.baseColor);
        if (baseColor != nullptr) {
          appendUniqueTextureDependency(passTextureDependencies_,
                                        baseColor->texture);
        }
      }

      meshDrawTemplates_.push_back(MeshDrawTemplate{
          .renderable = &renderable,
          .submesh = &submesh,
          .submeshIndex = submeshIndex,
          .instanceIndex = renderableIndex,
          .indexBuffer = geometry.indexBuffer,
          .indexBufferOffset = geometry.indexByteOffset,
          .indexFormat = resolveGeometryIndexFormat(geometry),
          .baseVertexBuffer = geometry.vertexBuffer,
          .vertexBufferByteOffset = geometry.vertexByteOffset,
          .vertexBufferAddress = vertexBufferAddress,
          .vertexDecodeBufferAddress =
              modelRecord->model->vertexDecodeBufferAddress(),
          .vertexDecodeIndex = submesh.vertexOffset,
          .packedVertexFormat =
              static_cast<uint32_t>(modelRecord->model->drawVertexFormat()),
          .materialIndex = materialRecord != nullptr
                               ? resources.materialTableIndex(resolvedMaterial)
                               : 0u,
          .doubleSided = materialRecord != nullptr
                             ? materialRecord->desc.doubleSided
                             : false,
          .alphaMasked = alphaMasked,
          .dynamicCaster = dynamicCaster,
      });
      const uint32_t templateIndex =
          static_cast<uint32_t>(meshDrawTemplates_.size() - 1u);
      if (dynamicCaster) {
        dynamicShadowTemplateIndices_.push_back(templateIndex);
      } else {
        staticShadowTemplateIndices_.push_back(templateIndex);
      }
    }
  }

  invalidateStaticShadowCasterCache();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
ShadowRenderer::rebuildStaticShadowCasterCache(const RenderScene &scene,
                                               const RenderSettings &settings) {
  staticShadowCasterCache_.clear();
  if (staticShadowTemplateIndices_.empty()) {
    staticShadowCasterCacheValid_ = true;
    return Result<bool, std::string>::makeResult(true);
  }

  const std::span<const Renderable> renderables = scene.renderables();
  const bool enableCascadeCasterCulling =
      settings.shadow.debug.enableCascadeCasterCulling;
  staticShadowCasterCache_.reserve(staticShadowTemplateIndices_.size());
  for (const uint32_t templateIndex : staticShadowTemplateIndices_) {
    if (templateIndex >= meshDrawTemplates_.size()) {
      return Result<bool, std::string>::makeError(
          "ShadowRenderer::rebuildStaticShadowCasterCache: template index out "
          "of range");
    }

    const MeshDrawTemplate &entry = meshDrawTemplates_[templateIndex];
    if (entry.renderable == nullptr || entry.submesh == nullptr ||
        entry.instanceIndex >= renderables.size()) {
      continue;
    }

    const std::optional<SubmeshLod> lod =
        resolveShadowLod(*entry.submesh, settings);
    if (!lod.has_value()) {
      continue;
    }

    StaticShadowCasterCacheEntry cachedEntry{
        .templateIndex = templateIndex,
        .instanceIndex = entry.instanceIndex,
        .indexBuffer = entry.indexBuffer,
        .indexBufferOffset = entry.indexBufferOffset,
        .indexFormat = entry.indexFormat,
        .vertexBuffer = entry.baseVertexBuffer,
        .vertexBufferAddress = entry.vertexBufferAddress,
        .vertexDecodeBufferAddress = entry.vertexDecodeBufferAddress,
        .vertexDecodeIndex = entry.vertexDecodeIndex,
        .packedVertexFormat = entry.packedVertexFormat,
        .materialIndex = entry.materialIndex,
        .indexCount = lod->indexCount,
        .firstIndex = lod->indexOffset,
        .doubleSided = entry.doubleSided,
        .alphaMasked = entry.alphaMasked,
        .hasCasterCullingBounds = false,
        .casterWorldCorners = {},
    };
    if (enableCascadeCasterCulling) {
      const BoundingBox worldBounds = entry.submesh->bounds.getTransformed(
          renderables[entry.instanceIndex].modelMatrix);
      cachedEntry.casterWorldCorners = shadow_detail::computeBoundsCorners(
          worldBounds.min_, worldBounds.max_);
      cachedEntry.hasCasterCullingBounds = true;
    }
    staticShadowCasterCache_.push_back(cachedEntry);
  }

  staticShadowCasterCacheValid_ = true;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> ShadowRenderer::updateShadowFrameData(
    RenderFrameContext &frame, const RenderSettings::ShadowSettings &settings,
    uint32_t shadowMapSize) {
  shadowFrameCpuData_ = {};
  shadowDebugFrameData_ = {};
  currentRawShadowFits_ = {};
  hasActiveShadowLightForFrame_ = false;
  const ShadowSdsmMode sdsmMode = sanitizeShadowSdsmMode(settings.sdsmMode);
  const ShadowSdsmReductionBackend requestedReductionBackend =
      sanitizeShadowSdsmReductionBackend(settings.sdsmReductionBackend);
  const uint32_t cascadeCount =
      std::clamp(activeCascadeCount_, 1u, kMaxShadowCascades);
  const ShadowSplitRange fixedSplitRange =
      computeFixedShadowSplitRange(frame.camera, settings.maxDistance);
  const std::array<float, kMaxShadowCascades + 1u> fixedSplitDepths =
      shadow_detail::computeCascadeSplitDepthsForRange(
          fixedSplitRange.nearDepth, fixedSplitRange.farDepth, cascadeCount,
          settings.splitMode, settings.splitLambda);
  const float minimumFarDepth = cascadeCount > 1u
                                    ? fixedSplitDepths[cascadeCount - 1u]
                                    : fixedSplitRange.farDepth;
  std::array<float, kMaxShadowCascades + 1u> minMaxSplitDepths =
      fixedSplitDepths;
  std::array<float, kMaxShadowCascades + 1u> histogramSplitDepths =
      fixedSplitDepths;
  shadowDebugFrameData_.rawSamplerId = frame.sharedResources.shadowRawSamplerId;
  shadowDebugFrameData_.compareSamplerId =
      frame.sharedResources.shadowCompareSamplerId;
  shadowDebugFrameData_.sdsm.mode = sdsmMode;
  shadowDebugFrameData_.sdsm.requestedReductionBackend =
      requestedReductionBackend;
  shadowDebugFrameData_.sdsm.activeReductionBackend =
      ShadowSdsmReductionBackend::Cpu;
  shadowDebugFrameData_.sdsm.status = sdsmMode == ShadowSdsmMode::Disabled
                                          ? ShadowSdsmStatus::Disabled
                                          : ShadowSdsmStatus::FallbackFixed;
  shadowDebugFrameData_.sdsm.reductionFallbackActive = false;
  shadowDebugFrameData_.sdsm.sourceFrameIndex =
      frame.sharedResources.sceneDepthPyramidSourceFrameIndex.value_or(
          std::numeric_limits<uint64_t>::max());
  shadowDebugFrameData_.sdsm.histogramBucketCount =
      settings.sdsmHistogramBucketCount;
  shadowDebugFrameData_.sdsm.histogramTrimLowPercent =
      settings.sdsmHistogramTrimLowPercent;
  shadowDebugFrameData_.sdsm.histogramTrimHighPercent =
      settings.sdsmHistogramTrimHighPercent;
  shadowDebugFrameData_.sdsm.gpuResultSelectedSlot =
      std::numeric_limits<uint32_t>::max();
  shadowDebugFrameData_.sdsm.gpuReductionResultAvailable = false;
  shadowDebugFrameData_.sdsm.gpuSplitPayloadValid = false;
  shadowDebugFrameData_.sdsm.gpuResultSourceFrameIndex =
      std::numeric_limits<uint64_t>::max();
  shadowDebugFrameData_.sdsm.splitCount = cascadeCount;
  shadowDebugFrameData_.sdsm.fixedRangeNear = fixedSplitRange.nearDepth;
  shadowDebugFrameData_.sdsm.fixedRangeFar = fixedSplitRange.farDepth;
  shadowDebugFrameData_.sdsm.smoothedLinearMin = fixedSplitRange.nearDepth;
  shadowDebugFrameData_.sdsm.smoothedLinearMax = fixedSplitRange.farDepth;
  shadowDebugFrameData_.sdsm.histogramTrimmedRangeNear =
      fixedSplitRange.nearDepth;
  shadowDebugFrameData_.sdsm.histogramTrimmedRangeFar =
      fixedSplitRange.farDepth;
  shadowDebugFrameData_.sdsm.effectiveRangeNear = fixedSplitRange.nearDepth;
  shadowDebugFrameData_.sdsm.effectiveRangeFar = fixedSplitRange.farDepth;
  shadowDebugFrameData_.sdsm.fixedSplitDepths = fixedSplitDepths;
  shadowDebugFrameData_.sdsm.minMaxSplitDepths = fixedSplitDepths;
  shadowDebugFrameData_.sdsm.histogramSplitDepths = fixedSplitDepths;
  shadowDebugFrameData_.sdsm.effectiveSplitDepths = fixedSplitDepths;
  shadowDebugFrameData_.sdsm.fixedFallbackActive =
      sdsmMode == ShadowSdsmMode::Histogram;
  frame.metrics.shadow.sdsmActiveReductionBackend =
      ShadowSdsmReductionBackend::Cpu;
  frame.metrics.shadow.sdsmReadbackBytes = 0u;
  SdsmLogDiagnostics sdsmLog{};
  bool suppressGpuWarmupWarning = false;
  sdsmLog.reason = sdsmMode == ShadowSdsmMode::Disabled
                       ? "sdsm_disabled"
                       : "initial_fallback_state";
  sdsmLog.sourceLevelCount = frame.sharedResources.sceneDepthPyramidLevelCount;
  sdsmLog.sourceFrameAvailable =
      frame.sharedResources.sceneDepthPyramidSourceFrameIndex.has_value();
  sdsmLog.historyRangeValid = sdsmState_.hasValidSdsmRange_;
  sdsmLog.historyTexelValid = sdsmState_.hasValidSdsmFarCascadeTexelSize_;
  const auto publishGpuReductionResultDebug = [&](uint32_t selectedSlot,
                                                  uint64_t sourceFrameIndex,
                                                  bool splitPayloadValid) {
    shadowDebugFrameData_.sdsm.gpuReductionResultAvailable = true;
    shadowDebugFrameData_.sdsm.gpuSplitPayloadValid = splitPayloadValid;
    shadowDebugFrameData_.sdsm.gpuResultSelectedSlot = selectedSlot;
    shadowDebugFrameData_.sdsm.gpuResultSourceFrameIndex = sourceFrameIndex;
  };
  for (uint32_t cascadeIndex = 0u; cascadeIndex < activeCascadeCount_;
       ++cascadeIndex) {
    shadowDebugFrameData_.cascades[cascadeIndex].texture =
        shadowDepthTextures_[cascadeIndex];
    shadowDebugFrameData_.cascades[cascadeIndex].textureBindlessId =
        gpu_.getTextureBindlessIndex(shadowDepthTextures_[cascadeIndex]);
  }
  const auto publishInactiveShadowFrame = [&]() -> Result<bool, std::string> {
    invalidateReusableStaticOnlyCascadeCache();
    frame.sharedResources.selectedShadowLightId.reset();
    frame.sharedResources.shadowDebugFrameData = shadowDebugFrameData_;
    return Result<bool, std::string>::makeResult(true);
  };
  const auto invalidateSdsmRange = [&]() {
    sdsmState_.hasValidSdsmRange_ = false;
    sdsmState_.hasValidSdsmFarCascadeTexelSize_ = false;
    sdsmState_.hasValidSdsmHistogramSplits_ = false;
    sdsmState_.sdsmHistogramCascadeCount_ = 0u;
    sdsmState_.sdsmSmoothedHistogramSplitDepths_ = {};
    sdsmState_.lastValidSdsmSourceFrameIndex_ =
        std::numeric_limits<uint64_t>::max();
  };
  const auto canReuseCachedSdsmRange = [&]() {
    if (!sdsmState_.hasValidSdsmRange_ ||
        sdsmState_.lastValidSdsmSourceFrameIndex_ ==
            std::numeric_limits<uint64_t>::max() ||
        frame.frameIndex <= sdsmState_.lastValidSdsmSourceFrameIndex_) {
      return false;
    }
    return (frame.frameIndex - sdsmState_.lastValidSdsmSourceFrameIndex_) <=
           kSdsmMaxCachedSourceFrameLag;
  };
  const auto applyCachedSdsmRange = [&]() {
    NURI_ASSERT(sdsmState_.hasValidSdsmRange_,
                "Cached SDSM range requested without valid history");
    shadowDebugFrameData_.sdsm.smoothedLinearMin =
        sdsmState_.sdsmSmoothedMinDepth_;
    shadowDebugFrameData_.sdsm.smoothedLinearMax =
        sdsmState_.sdsmSmoothedMaxDepth_;
    return fixedSplitRange;
  };
  ShadowSplitRange effectiveSplitRange = fixedSplitRange;
  std::array<float, kMaxShadowCascades + 1u> effectiveSplitDepths =
      fixedSplitDepths;
  const auto updateEffectiveSplitRange = [&](ShadowSplitRange range) {
    effectiveSplitRange = range;
    effectiveSplitDepths = shadow_detail::computeCascadeSplitDepthsForRange(
        effectiveSplitRange.nearDepth, effectiveSplitRange.farDepth,
        cascadeCount, settings.splitMode, settings.splitLambda);
  };
  const auto updateHistogramEffectiveSplitDepths =
      [&](const std::array<float, kMaxShadowCascades + 1u> &rawSplitDepths,
          bool allowSmoothing) {
        if (!allowSmoothing || !sdsmState_.hasValidSdsmHistogramSplits_ ||
            sdsmState_.sdsmHistogramCascadeCount_ != cascadeCount) {
          sdsmState_.sdsmSmoothedHistogramSplitDepths_ = rawSplitDepths;
        } else {
          const float historyWeight =
              std::clamp(settings.sdsmTemporalBlend, 0.0f, 1.0f);
          const float sampleWeight = 1.0f - historyWeight;
          for (uint32_t i = 1u; i < cascadeCount; ++i) {
            sdsmState_.sdsmSmoothedHistogramSplitDepths_[i] =
                rawSplitDepths[i] * sampleWeight +
                sdsmState_.sdsmSmoothedHistogramSplitDepths_[i] * historyWeight;
          }
        }
        sdsmState_.sdsmSmoothedHistogramSplitDepths_[0] = rawSplitDepths[0];
        sdsmState_.sdsmSmoothedHistogramSplitDepths_[cascadeCount] =
            rawSplitDepths[cascadeCount];
        shadow_detail::enforceMonotonicShadowSplitDepths(
            sdsmState_.sdsmSmoothedHistogramSplitDepths_, cascadeCount,
            rawSplitDepths[0],
            std::max(rawSplitDepths[cascadeCount],
                     rawSplitDepths[0] + 1.0e-4f));
        sdsmState_.hasValidSdsmHistogramSplits_ = true;
        sdsmState_.sdsmHistogramCascadeCount_ = cascadeCount;
      };
  const bool histogramSdsm = sdsmMode == ShadowSdsmMode::Histogram;
  const bool supportsGpuReductionRequest =
      requestedReductionBackend != ShadowSdsmReductionBackend::Cpu &&
      (sdsmMode == ShadowSdsmMode::PreviousFrameMinMax ||
       sdsmMode == ShadowSdsmMode::Histogram);
  const bool gpuReductionPipelineAvailable =
      sdsmMode == ShadowSdsmMode::Histogram
          ? nuri::isValid(sdsmHistogramReducePipelineHandle_)
          : nuri::isValid(sdsmReducePipelineHandle_);
  const bool gpuReductionResourcesAvailable =
      supportsGpuReductionRequest && gpuReductionPipelineAvailable &&
      !sdsmReduceResultRing_.empty() &&
      std::all_of(sdsmReduceResultRing_.begin(), sdsmReduceResultRing_.end(),
                  [](const DynamicBufferSlot &slot) {
                    return slot.buffer && slot.buffer->valid();
                  });
  ShadowSdsmReductionBackend activeReductionBackend =
      supportsGpuReductionRequest && gpuReductionResourcesAvailable
          ? ShadowSdsmReductionBackend::Gpu
          : ShadowSdsmReductionBackend::Cpu;
  const auto activateReductionFallback = [&](std::string_view warningReason) {
    shadowDebugFrameData_.sdsm.reductionFallbackActive = true;
    if (requestedReductionBackend != ShadowSdsmReductionBackend::Gpu ||
        sdsmState_.loggedGpuReductionFallbackWarning_) {
      return;
    }
    NURI_LOG_WARNING(
        "ShadowRenderer::updateShadowFrameData: requested GPU SDSM "
        "reduction fell back to CPU path (%s)",
        std::string(warningReason).c_str());
    sdsmState_.loggedGpuReductionFallbackWarning_ = true;
  };
  if (requestedReductionBackend == ShadowSdsmReductionBackend::Gpu &&
      !gpuReductionResourcesAvailable) {
    activateReductionFallback("gpu_reduction_unavailable");
  }
  if (activeReductionBackend != ShadowSdsmReductionBackend::Gpu) {
    sdsmState_.gpuReductionConsecutiveMissingResultFrames_ = 0u;
  }
  shadowDebugFrameData_.sdsm.activeReductionBackend = activeReductionBackend;
  shadowDebugFrameData_.sdsm.gpuResultRingSlotCount =
      activeReductionBackend == ShadowSdsmReductionBackend::Gpu
          ? saturateToU32(sdsmReduceResultRing_.size())
          : 0u;
  frame.metrics.shadow.sdsmActiveReductionBackend = activeReductionBackend;
  if (sdsmMode == ShadowSdsmMode::PreviousFrameMinMax) {
    frame.metrics.shadow.sdsmReadbackBytes =
        activeReductionBackend == ShadowSdsmReductionBackend::Gpu
            ? static_cast<uint32_t>(sizeof(SdsmGpuMinMaxResult))
            : static_cast<uint32_t>(sizeof(float) * 2u);
  }
  const auto buildHistogramSplitDepthsForRange =
      [&](const std::array<float, kMaxShadowCascades + 1u> &histogramDepths,
          ShadowSplitRange range) {
        std::array<float, kMaxShadowCascades + 1u> mappedSplitDepths =
            histogramDepths;
        const float mappedNear = std::max(range.nearDepth, 0.01f);
        const float mappedFar = std::max(mappedNear + 1.0e-4f, range.farDepth);
        mappedSplitDepths[0] = mappedNear;
        if (cascadeCount == 1u) {
          mappedSplitDepths[1] = mappedFar;
          return mappedSplitDepths;
        }

        const float histogramFar = std::max(histogramDepths[cascadeCount],
                                            histogramDepths[0] + 1.0e-4f);
        const float histogramSpan =
            std::max(histogramFar - histogramDepths[0], 1.0e-4f);
        const float mappedSpan = mappedFar - mappedNear;
        for (uint32_t i = 1u; i < cascadeCount; ++i) {
          const float normalized = std::clamp(
              (histogramDepths[i] - histogramDepths[0]) / histogramSpan, 0.0f,
              1.0f);
          mappedSplitDepths[i] = mappedNear + normalized * mappedSpan;
        }
        shadow_detail::enforceMonotonicShadowSplitDepths(
            mappedSplitDepths, cascadeCount, mappedNear, mappedFar);
        return mappedSplitDepths;
      };
  const auto reuseCachedSdsmRangeOrFallback = [&](ShadowSdsmStatus status,
                                                  std::string_view reason) {
    shadowDebugFrameData_.sdsm.status = status;
    sdsmLog.reason = reason;
    if (canReuseCachedSdsmRange()) {
      sdsmLog.reusedCachedRange = true;
      shadowDebugFrameData_.sdsm.fixedFallbackActive = false;
      const ShadowSplitRange cachedRange = applyCachedSdsmRange();
      effectiveSplitRange = cachedRange;
      if (sdsmMode == ShadowSdsmMode::Histogram &&
          sdsmState_.hasValidSdsmHistogramSplits_ &&
          sdsmState_.sdsmHistogramCascadeCount_ == cascadeCount) {
        histogramSplitDepths = sdsmState_.sdsmSmoothedHistogramSplitDepths_;
      }
      updateEffectiveSplitRange(cachedRange);
      return;
    }
    invalidateSdsmRange();
    shadowDebugFrameData_.sdsm.fixedFallbackActive = true;
  };

  if (frame.scene == nullptr) {
    resetFrozenShadowFit();
    resetCascadeStabilizationHistory();
    resetSdsmState();
    return publishInactiveShadowFrame();
  }

  const std::span<const DirectionalLightGpuData> directionalLights =
      frame.scene->packedDirectionalLights();
  const std::span<const LightId> directionalLightIds =
      frame.scene->packedDirectionalLightIds();
  if (directionalLights.empty() || directionalLightIds.empty()) {
    resetFrozenShadowFit();
    resetCascadeStabilizationHistory();
    resetSdsmState();
    return publishInactiveShadowFrame();
  }

  uint32_t selectedLightIndex = 0u;
  LightId selectedLightId = directionalLightIds.front();
  if (frame.sharedResources.selectedShadowLightId.has_value() &&
      isValid(*frame.sharedResources.selectedShadowLightId)) {
    for (uint32_t i = 0u; i < directionalLightIds.size(); ++i) {
      if (directionalLightIds[i] ==
          *frame.sharedResources.selectedShadowLightId) {
        selectedLightIndex = i;
        selectedLightId = directionalLightIds[i];
        break;
      }
    }
  }
  frame.sharedResources.selectedShadowLightId = selectedLightId;
  const bool freezeShadowFits =
      settings.debug.freezeLightView || settings.debug.freezeCascades;
  if (!freezeShadowFits) {
    resetFrozenShadowFit();
  }
  if (!settings.stabilizeCascades) {
    resetCascadeStabilizationHistory();
  } else if (cascadeStabilizationHistory_.valid &&
             (cascadeStabilizationHistory_.lightId != selectedLightId ||
              cascadeStabilizationHistory_.shadowMapSize != shadowMapSize ||
              cascadeStabilizationHistory_.cascadeCount != cascadeCount)) {
    resetCascadeStabilizationHistory();
  }
  const bool canReuseCascadeStabilizationHistory =
      settings.stabilizeCascades && cascadeStabilizationHistory_.valid;
  hasActiveShadowLightForFrame_ = true;

  const DirectionalLightGpuData &light = directionalLights[selectedLightIndex];
  const glm::vec3 lightDirection = shadow_detail::normalizeSafe(
      glm::vec3(light.directionIlluminance), glm::vec3(0.0f, -1.0f, 0.0f));
  const AnimationSceneFrameData *animationSceneData =
      resolveAnimationSceneFrameData(frame);
  std::pmr::vector<glm::vec3> casterPoints(memory_);
  casterPoints.reserve(meshDrawTemplates_.size() * 8u);
  BoundingBox casterBounds{};
  bool hasCasterBounds = false;
  bool hasAnimatedGeometryOverrides = false;
  for (const MeshDrawTemplate &entry : meshDrawTemplates_) {
    if (entry.renderable == nullptr || entry.submesh == nullptr) {
      continue;
    }
    bool usesAnimatedOverride = false;
    if (animationSceneData != nullptr &&
        entry.instanceIndex <
            animationSceneData->geometryOverridesByRenderable.size()) {
      const AnimatedRenderableGeometryOverride &geometryOverride =
          animationSceneData
              ->geometryOverridesByRenderable[entry.instanceIndex];
      usesAnimatedOverride =
          geometryOverride.enabled &&
          nuri::isValid(geometryOverride.vertexBuffer) &&
          animationOverrideCoversSubmesh(geometryOverride, *entry.submesh);
    }
    if (usesAnimatedOverride) {
      hasAnimatedGeometryOverrides = true;
      continue;
    }
    const BoundingBox worldBounds =
        entry.submesh->bounds.getTransformed(entry.renderable->modelMatrix);
    const auto worldBoundsCorners =
        shadow_detail::computeBoundsCorners(worldBounds.min_, worldBounds.max_);
    casterPoints.insert(casterPoints.end(), worldBoundsCorners.begin(),
                        worldBoundsCorners.end());
    casterBounds.combinePoint(worldBounds.min_);
    casterBounds.combinePoint(worldBounds.max_);
    hasCasterBounds = true;
  }
  shadowDebugFrameData_.cascadeCount = cascadeCount;
  if (sdsmMode == ShadowSdsmMode::Disabled) {
    sdsmLog.reason = "sdsm_disabled";
    invalidateSdsmRange();
  } else {
    TextureHandle latestSdsmPyramidTexture{};
    bool hasValidSdsmSource =
        frame.sharedResources.sceneDepthPyramidSourceFrameIndex.has_value() &&
        frame.frameIndex > 0u &&
        *frame.sharedResources.sceneDepthPyramidSourceFrameIndex ==
            (frame.frameIndex - 1u) &&
        frame.sharedResources.sceneDepthPyramidLevelCount > 0u;
    if (frame.sharedResources.sceneDepthPyramidLevelCount > 0u) {
      latestSdsmPyramidTexture =
          frame.sharedResources.sceneDepthPyramidTextures
              [frame.sharedResources.sceneDepthPyramidLevelCount - 1u];
      sdsmLog.sourceTextureValid = nuri::isValid(latestSdsmPyramidTexture);
      hasValidSdsmSource = hasValidSdsmSource && sdsmLog.sourceTextureValid;
    }
    sdsmLog.hasValidSource = hasValidSdsmSource;
    if (!hasValidSdsmSource) {
      sdsmState_.gpuReductionConsecutiveMissingResultFrames_ = 0u;
      const ShadowSdsmStatus sourceStatus =
          frame.sharedResources.sceneDepthPyramidSourceFrameIndex.has_value() &&
                  frame.frameIndex > 0u
              ? ShadowSdsmStatus::Stale
              : ShadowSdsmStatus::Unavailable;
      reuseCachedSdsmRangeOrFallback(sourceStatus,
                                     sourceStatus == ShadowSdsmStatus::Stale
                                         ? "stale_source_frame"
                                         : "missing_previous_frame_data");
    } else {
      const float effectiveNear = fixedSplitRange.nearDepth;
      const auto activateStableSdsmCoverage = [&]() {
        sdsmLog.reason = "valid_previous_frame_data";
        shadowDebugFrameData_.sdsm.status = ShadowSdsmStatus::Active;
        shadowDebugFrameData_.sdsm.fixedFallbackActive = false;
        shadowDebugFrameData_.sdsm.smoothedLinearMin =
            sdsmState_.sdsmSmoothedMinDepth_;
        shadowDebugFrameData_.sdsm.smoothedLinearMax =
            sdsmState_.sdsmSmoothedMaxDepth_;
        // Keep full shadow coverage stable. SDSM still tracks visible depth
        // distribution for debug and, in histogram mode, can redistribute
        // internal split placement, but it no longer contracts the overall
        // shadow distance and then snaps back as camera zoom/dolly changes.
        effectiveSplitRange = fixedSplitRange;
      };
      const auto consumeRawDeviceMinMax =
          [&](float rawDeviceMin, float rawDeviceMax,
              std::string_view invalidDepthReason,
              std::string_view invalidLinearReason) {
            shadowDebugFrameData_.sdsm.rawDeviceMin = rawDeviceMin;
            shadowDebugFrameData_.sdsm.rawDeviceMax = rawDeviceMax;

            const bool rawDepthsValid =
                std::isfinite(rawDeviceMin) && std::isfinite(rawDeviceMax) &&
                rawDeviceMin >= -1.0e-4f && rawDeviceMax <= (1.0f + 1.0e-4f) &&
                rawDeviceMax >= rawDeviceMin;
            sdsmLog.rawDepthsValid = rawDepthsValid;
            if (!rawDepthsValid) {
              reuseCachedSdsmRangeOrFallback(ShadowSdsmStatus::Invalid,
                                             invalidDepthReason);
              return;
            }

            const float rawLinearMin =
                shadow_detail::linearizeDeviceDepthToViewDepth(rawDeviceMin,
                                                               frame.camera);
            const float rawLinearMax =
                shadow_detail::linearizeDeviceDepthToViewDepth(rawDeviceMax,
                                                               frame.camera);
            shadowDebugFrameData_.sdsm.rawLinearMin = rawLinearMin;
            shadowDebugFrameData_.sdsm.rawLinearMax = rawLinearMax;
            const bool rawLinearValid = std::isfinite(rawLinearMin) &&
                                        std::isfinite(rawLinearMax) &&
                                        rawLinearMax >= rawLinearMin;
            sdsmLog.rawLinearValid = rawLinearValid;
            if (!rawLinearValid) {
              reuseCachedSdsmRangeOrFallback(ShadowSdsmStatus::Invalid,
                                             invalidLinearReason);
              return;
            }

            if (!sdsmState_.hasValidSdsmRange_) {
              sdsmState_.sdsmSmoothedMinDepth_ = rawLinearMin;
              sdsmState_.sdsmSmoothedMaxDepth_ = rawLinearMax;
            } else {
              const float historyWeight =
                  std::clamp(settings.sdsmTemporalBlend, 0.0f, 1.0f);
              const float sampleWeight = 1.0f - historyWeight;
              sdsmState_.sdsmSmoothedMinDepth_ =
                  rawLinearMin * sampleWeight +
                  sdsmState_.sdsmSmoothedMinDepth_ * historyWeight;
              sdsmState_.sdsmSmoothedMaxDepth_ =
                  rawLinearMax * sampleWeight +
                  sdsmState_.sdsmSmoothedMaxDepth_ * historyWeight;
            }
            sdsmState_.hasValidSdsmRange_ = true;
            sdsmState_.lastValidSdsmSourceFrameIndex_ =
                *frame.sharedResources.sceneDepthPyramidSourceFrameIndex;
            activateStableSdsmCoverage();
            updateEffectiveSplitRange(effectiveSplitRange);
            minMaxSplitDepths = effectiveSplitDepths;
          };

      if (sdsmMode == ShadowSdsmMode::Histogram) {
        std::array<glm::uvec2, kMaxSceneDepthPyramidLevels> levelDimensions{};
        for (uint32_t level = 0u;
             level < frame.sharedResources.sceneDepthPyramidLevelCount;
             ++level) {
          const TextureHandle texture =
              frame.sharedResources.sceneDepthPyramidTextures[level];
          if (!nuri::isValid(texture)) {
            levelDimensions[level] = glm::uvec2(1u);
            continue;
          }
          const TextureDimensions dimensions =
              gpu_.getTextureDimensions(texture);
          levelDimensions[level] = glm::uvec2(std::max(dimensions.width, 1u),
                                              std::max(dimensions.height, 1u));
        }
        const shadow_detail::ShadowSdsmHistogramSourceSelection
            sourceSelection = shadow_detail::selectSdsmHistogramSourceLevel(
                std::span<const glm::uvec2>(
                    levelDimensions.data(),
                    frame.sharedResources.sceneDepthPyramidLevelCount),
                frame.sharedResources.sceneDepthPyramidLevelCount,
                kSdsmHistogramSourceMaxTexelCount);
        shadowDebugFrameData_.sdsm.histogramSourceLevel = sourceSelection.level;
        shadowDebugFrameData_.sdsm.histogramSourceDimensions =
            sourceSelection.dimensions;
        TextureHandle sdsmTexture =
            frame.sharedResources
                .sceneDepthPyramidTextures[sourceSelection.level];
        sdsmLog.sourceTextureValid = nuri::isValid(sdsmTexture);
        if (!nuri::isValid(sdsmTexture)) {
          reuseCachedSdsmRangeOrFallback(ShadowSdsmStatus::Unavailable,
                                         "invalid_source_texture");
        } else {
          const uint32_t width = std::max(sourceSelection.dimensions.x, 1u);
          const uint32_t height = std::max(sourceSelection.dimensions.y, 1u);
          frame.metrics.shadow.sdsmReductionSourceSamples = width * height;
          frame.metrics.shadow.sdsmHistogramSourceSamples = width * height;
          const auto histogramStart = std::chrono::steady_clock::now();
          const auto consumeCpuHistogramSource = [&]() -> bool {
            const size_t requiredBytes =
                static_cast<size_t>(width) * height * sizeof(float) * 2u;
            if (sdsmReadbackBuffer_.capacity() < requiredBytes) {
              sdsmReadbackBuffer_.reserve(requiredBytes);
            }
            if (sdsmReadbackBuffer_.size() < requiredBytes) {
              sdsmReadbackBuffer_.resize(requiredBytes);
            }
            std::span<std::byte> sdsmBytes(sdsmReadbackBuffer_.data(),
                                           requiredBytes);
            const TextureReadbackRegion readbackRegion{
                .x = 0u,
                .y = 0u,
                .width = width,
                .height = height,
                .mipLevel = 0u,
                .layer = 0u,
            };
            auto readResult =
                gpu_.readTexture(sdsmTexture, readbackRegion, sdsmBytes);
            if (readResult.hasError()) {
              sdsmLog.readbackError = true;
              reuseCachedSdsmRangeOrFallback(ShadowSdsmStatus::Invalid,
                                             "source_readback_error");
              return false;
            }
            sdsmLog.readbackError = false;

            const float *sdsmValues =
                reinterpret_cast<const float *>(sdsmBytes.data());
            const std::span<const float> minMaxPairs(
                sdsmValues, sdsmBytes.size() / sizeof(float));
            const SdsmHistogramAnalysis analysis = buildSdsmHistogramAnalysis(
                frame.camera, fixedSplitRange, cascadeCount,
                settings.sdsmHistogramBucketCount,
                settings.sdsmHistogramTrimLowPercent,
                settings.sdsmHistogramTrimHighPercent, sourceSelection.level,
                sourceSelection.dimensions, minMaxPairs);
            shadowDebugFrameData_.sdsm.histogramValidTileCount =
                analysis.validTileCount;
            shadowDebugFrameData_.sdsm.histogramTotalWeight =
                analysis.totalWeight;
            shadowDebugFrameData_.sdsm.histogramTrimmedRangeNear =
                analysis.trimmedNear;
            shadowDebugFrameData_.sdsm.histogramTrimmedRangeFar =
                analysis.trimmedFar;
            shadowDebugFrameData_.sdsm.histogramBucketWeights =
                analysis.bucketWeights;
            shadowDebugFrameData_.sdsm.rawDeviceMin = analysis.rawDeviceMin;
            shadowDebugFrameData_.sdsm.rawDeviceMax = analysis.rawDeviceMax;
            shadowDebugFrameData_.sdsm.rawLinearMin = analysis.rawLinearMin;
            shadowDebugFrameData_.sdsm.rawLinearMax = analysis.rawLinearMax;
            sdsmLog.rawDepthsValid = analysis.validTileCount > 0u;
            sdsmLog.rawLinearValid = analysis.valid;
            if (!analysis.valid) {
              reuseCachedSdsmRangeOrFallback(ShadowSdsmStatus::Invalid,
                                             "invalid_histogram_source_data");
              return false;
            }

            const float sdsmSampleNear = analysis.clearOnlySource
                                             ? analysis.rawLinearMin
                                             : analysis.trimmedNear;
            const float sdsmSampleFar = analysis.clearOnlySource
                                            ? analysis.rawLinearMax
                                            : analysis.trimmedFar;
            const float historyWeight =
                std::clamp(settings.sdsmTemporalBlend, 0.0f, 1.0f);
            const float sampleWeight = 1.0f - historyWeight;
            if (!sdsmState_.hasValidSdsmRange_) {
              sdsmState_.sdsmSmoothedMinDepth_ = sdsmSampleNear;
              sdsmState_.sdsmSmoothedMaxDepth_ = sdsmSampleFar;
            } else {
              sdsmState_.sdsmSmoothedMinDepth_ =
                  sdsmSampleNear * sampleWeight +
                  sdsmState_.sdsmSmoothedMinDepth_ * historyWeight;
              sdsmState_.sdsmSmoothedMaxDepth_ =
                  sdsmSampleFar * sampleWeight +
                  sdsmState_.sdsmSmoothedMaxDepth_ * historyWeight;
            }
            sdsmState_.hasValidSdsmRange_ = true;
            sdsmState_.lastValidSdsmSourceFrameIndex_ =
                *frame.sharedResources.sceneDepthPyramidSourceFrameIndex;

            const float comparisonFar =
                analysis.clearOnlySource
                    ? fixedSplitRange.farDepth
                    : std::clamp(std::max(analysis.trimmedFar, minimumFarDepth),
                                 effectiveNear + 1.0e-4f,
                                 fixedSplitRange.farDepth);
            minMaxSplitDepths =
                shadow_detail::computeCascadeSplitDepthsForRange(
                    effectiveNear, comparisonFar, cascadeCount,
                    settings.splitMode, settings.splitLambda);
            histogramSplitDepths = analysis.clearOnlySource
                                       ? fixedSplitDepths
                                       : analysis.rawSplitDepths;
            activateStableSdsmCoverage();
            updateHistogramEffectiveSplitDepths(
                histogramSplitDepths,
                !analysis.clearOnlySource &&
                    sdsmState_.hasValidSdsmHistogramSplits_);
            histogramSplitDepths = sdsmState_.sdsmSmoothedHistogramSplitDepths_;
            updateEffectiveSplitRange(effectiveSplitRange);
            return true;
          };
          if (activeReductionBackend == ShadowSdsmReductionBackend::Gpu) {
            bool anyReadbackError = false;
            std::optional<SdsmGpuHistogramResult> selectedGpuResult;
            uint32_t selectedGpuResultSlot =
                std::numeric_limits<uint32_t>::max();
            for (size_t slotIndex = 0u;
                 slotIndex < sdsmReduceResultRing_.size(); ++slotIndex) {
              const DynamicBufferSlot &slot = sdsmReduceResultRing_[slotIndex];
              if (!slot.buffer || !slot.buffer->valid()) {
                continue;
              }
              const uint64_t expectedPublishedFrame =
                  slotIndex < sdsmReduceResultRingPublishedFrames_.size()
                      ? sdsmReduceResultRingPublishedFrames_[slotIndex]
                      : std::numeric_limits<uint64_t>::max();
              if (expectedPublishedFrame ==
                  std::numeric_limits<uint64_t>::max()) {
                continue;
              }

              SdsmGpuHistogramResult gpuResult{};
              auto readResult = gpu_.readBuffer(
                  slot.buffer->handle(), 0u,
                  std::as_writable_bytes(
                      std::span<SdsmGpuHistogramResult>(&gpuResult, 1u)));
              if (readResult.hasError()) {
                anyReadbackError = true;
                continue;
              }
              const uint32_t valid = gpuResult.metadata.y;
              const uint32_t sourceFrameIndex = gpuResult.metadata.x;
              if (valid == 0u ||
                  static_cast<uint64_t>(sourceFrameIndex) >= frame.frameIndex) {
                continue;
              }
              if (static_cast<uint64_t>(sourceFrameIndex) !=
                  expectedPublishedFrame) {
                continue;
              }
              if (!selectedGpuResult.has_value() ||
                  sourceFrameIndex > selectedGpuResult->metadata.x) {
                selectedGpuResult = gpuResult;
                selectedGpuResultSlot = saturateToU32(slotIndex);
              }
            }

            if (!selectedGpuResult.has_value()) {
              const std::string_view fallbackReason =
                  anyReadbackError ? "gpu_histogram_readback_error"
                                   : "gpu_histogram_invalid_or_stale";
              sdsmLog.readbackError = anyReadbackError;
              if (!anyReadbackError) {
                ++sdsmState_.gpuReductionConsecutiveMissingResultFrames_;
                suppressGpuWarmupWarning =
                    sdsmState_.gpuReductionConsecutiveMissingResultFrames_ <=
                    kSdsmGpuWarmupGraceMissFrames;
              } else {
                sdsmState_.gpuReductionConsecutiveMissingResultFrames_ = 0u;
                suppressGpuWarmupWarning = false;
              }
              if (!suppressGpuWarmupWarning) {
                activateReductionFallback(fallbackReason);
              }
              (void)consumeCpuHistogramSource();
            } else {
              sdsmState_.gpuReductionConsecutiveMissingResultFrames_ = 0u;
              sdsmState_.loggedGpuReductionFallbackWarning_ = false;
              sdsmState_.loggedGpuResultRingDiagnosticWarning_ = false;
              const SdsmGpuHistogramResult &result = *selectedGpuResult;
              const uint32_t resultBucketCount = std::clamp(
                  result.metadata.w, kMinShadowSdsmHistogramBucketCount,
                  kMaxShadowSdsmHistogramBucketCount);
              shadowDebugFrameData_.sdsm.sourceFrameIndex = result.metadata.x;
              publishGpuReductionResultDebug(selectedGpuResultSlot,
                                             result.metadata.x, true);
              shadowDebugFrameData_.sdsm.histogramValidTileCount =
                  result.metadata.z;
              shadowDebugFrameData_.sdsm.histogramTotalWeight =
                  result.histogramRangeWeightClear.z;
              shadowDebugFrameData_.sdsm.histogramTrimmedRangeNear =
                  result.histogramRangeWeightClear.x;
              shadowDebugFrameData_.sdsm.histogramTrimmedRangeFar =
                  result.histogramRangeWeightClear.y;
              shadowDebugFrameData_.sdsm.histogramBucketWeights =
                  result.bucketWeights;
              shadowDebugFrameData_.sdsm.rawDeviceMin =
                  result.rawDeviceMinMaxLinearMinMax.x;
              shadowDebugFrameData_.sdsm.rawDeviceMax =
                  result.rawDeviceMinMaxLinearMinMax.y;
              shadowDebugFrameData_.sdsm.rawLinearMin =
                  result.rawDeviceMinMaxLinearMinMax.z;
              shadowDebugFrameData_.sdsm.rawLinearMax =
                  result.rawDeviceMinMaxLinearMinMax.w;
              shadowDebugFrameData_.sdsm.histogramBucketCount =
                  resultBucketCount;
              sdsmLog.rawDepthsValid = result.metadata.z > 0u;
              sdsmLog.rawLinearValid = true;

              const bool clearOnlySource =
                  result.histogramRangeWeightClear.w > 0.5f;
              const float sdsmSampleNear =
                  clearOnlySource ? result.rawDeviceMinMaxLinearMinMax.z
                                  : result.histogramRangeWeightClear.x;
              const float sdsmSampleFar =
                  clearOnlySource ? result.rawDeviceMinMaxLinearMinMax.w
                                  : result.histogramRangeWeightClear.y;
              const float historyWeight =
                  std::clamp(settings.sdsmTemporalBlend, 0.0f, 1.0f);
              const float sampleWeight = 1.0f - historyWeight;
              if (!sdsmState_.hasValidSdsmRange_) {
                sdsmState_.sdsmSmoothedMinDepth_ = sdsmSampleNear;
                sdsmState_.sdsmSmoothedMaxDepth_ = sdsmSampleFar;
              } else {
                sdsmState_.sdsmSmoothedMinDepth_ =
                    sdsmSampleNear * sampleWeight +
                    sdsmState_.sdsmSmoothedMinDepth_ * historyWeight;
                sdsmState_.sdsmSmoothedMaxDepth_ =
                    sdsmSampleFar * sampleWeight +
                    sdsmState_.sdsmSmoothedMaxDepth_ * historyWeight;
              }
              sdsmState_.hasValidSdsmRange_ = true;
              sdsmState_.lastValidSdsmSourceFrameIndex_ = result.metadata.x;

              const float comparisonFar =
                  clearOnlySource
                      ? fixedSplitRange.farDepth
                      : std::clamp(std::max(result.histogramRangeWeightClear.y,
                                            minimumFarDepth),
                                   effectiveNear + 1.0e-4f,
                                   fixedSplitRange.farDepth);
              minMaxSplitDepths =
                  shadow_detail::computeCascadeSplitDepthsForRange(
                      effectiveNear, comparisonFar, cascadeCount,
                      settings.splitMode, settings.splitLambda);
              histogramSplitDepths = fixedSplitDepths;
              if (!clearOnlySource) {
                histogramSplitDepths[0] = result.splitDepths0.x;
                if (cascadeCount >= 1u) {
                  histogramSplitDepths[1] = result.splitDepths0.y;
                }
                if (cascadeCount >= 2u) {
                  histogramSplitDepths[2] = result.splitDepths0.z;
                }
                if (cascadeCount >= 3u) {
                  histogramSplitDepths[3] = result.splitDepths0.w;
                }
                if (cascadeCount >= 4u) {
                  histogramSplitDepths[4] = result.splitDepths1.x;
                }
                shadow_detail::enforceMonotonicShadowSplitDepths(
                    histogramSplitDepths, cascadeCount, histogramSplitDepths[0],
                    std::max(histogramSplitDepths[cascadeCount],
                             histogramSplitDepths[0] + 1.0e-4f));
              }
              activateStableSdsmCoverage();
              updateHistogramEffectiveSplitDepths(
                  histogramSplitDepths,
                  !clearOnlySource && sdsmState_.hasValidSdsmHistogramSplits_);
              histogramSplitDepths =
                  sdsmState_.sdsmSmoothedHistogramSplitDepths_;
              updateEffectiveSplitRange(effectiveSplitRange);
            }
          } else {
            (void)consumeCpuHistogramSource();
          }
          const float histogramElapsedMs = elapsedMilliseconds(
              histogramStart, std::chrono::steady_clock::now());
          frame.metrics.shadow.sdsmCpuReductionTimeMs = histogramElapsedMs;
          frame.metrics.shadow.sdsmCpuHistogramTimeMs = histogramElapsedMs;
        }
      } else {
        if (activeReductionBackend == ShadowSdsmReductionBackend::Gpu) {
          const auto reductionStart = std::chrono::steady_clock::now();
          bool anyReadbackError = false;
          std::optional<SdsmGpuMinMaxResult> selectedGpuResult;
          uint32_t selectedGpuResultSlot = std::numeric_limits<uint32_t>::max();
          for (size_t slotIndex = 0u; slotIndex < sdsmReduceResultRing_.size();
               ++slotIndex) {
            const DynamicBufferSlot &slot = sdsmReduceResultRing_[slotIndex];
            if (!slot.buffer || !slot.buffer->valid()) {
              continue;
            }
            const uint64_t expectedPublishedFrame =
                slotIndex < sdsmReduceResultRingPublishedFrames_.size()
                    ? sdsmReduceResultRingPublishedFrames_[slotIndex]
                    : std::numeric_limits<uint64_t>::max();
            if (expectedPublishedFrame ==
                std::numeric_limits<uint64_t>::max()) {
              continue;
            }

            SdsmGpuMinMaxResult gpuResult{};
            auto readResult = gpu_.readBuffer(
                slot.buffer->handle(), 0u,
                std::as_writable_bytes(
                    std::span<SdsmGpuMinMaxResult>(&gpuResult, 1u)));
            if (readResult.hasError()) {
              anyReadbackError = true;
              continue;
            }
            if (gpuResult.valid == 0u ||
                static_cast<uint64_t>(gpuResult.sourceFrameIndex) >=
                    frame.frameIndex) {
              continue;
            }
            if (static_cast<uint64_t>(gpuResult.sourceFrameIndex) !=
                expectedPublishedFrame) {
              continue;
            }
            if (!selectedGpuResult.has_value() ||
                gpuResult.sourceFrameIndex >
                    selectedGpuResult->sourceFrameIndex) {
              selectedGpuResult = gpuResult;
              selectedGpuResultSlot = saturateToU32(slotIndex);
            }
          }

          if (!selectedGpuResult.has_value()) {
            const std::string_view fallbackReason =
                anyReadbackError ? "gpu_result_readback_error"
                                 : "gpu_result_invalid_or_stale";
            sdsmLog.readbackError = anyReadbackError;
            if (!anyReadbackError) {
              ++sdsmState_.gpuReductionConsecutiveMissingResultFrames_;
              suppressGpuWarmupWarning =
                  sdsmState_.gpuReductionConsecutiveMissingResultFrames_ <=
                  kSdsmGpuWarmupGraceMissFrames;
            } else {
              sdsmState_.gpuReductionConsecutiveMissingResultFrames_ = 0u;
              suppressGpuWarmupWarning = false;
            }
            if (!suppressGpuWarmupWarning &&
                !sdsmState_.loggedGpuResultRingDiagnosticWarning_) {
              sdsmState_.loggedGpuResultRingDiagnosticWarning_ = true;
              const uint64_t requestedSourceFrameIndex =
                  frame.sharedResources.sceneDepthPyramidSourceFrameIndex
                      .value_or(std::numeric_limits<uint64_t>::max());
              NURI_LOG_WARNING(
                  "ShadowRenderer::updateShadowFrameData: GPU SDSM ring "
                  "diagnostics frame=%llu requestedSourceFrame=%llu "
                  "slotCount=%zu reason=%s",
                  static_cast<unsigned long long>(frame.frameIndex),
                  static_cast<unsigned long long>(
                      requestedSourceFrameIndex ==
                              std::numeric_limits<uint64_t>::max()
                          ? 0u
                          : requestedSourceFrameIndex),
                  sdsmReduceResultRing_.size(), fallbackReason.data());
              for (size_t slotIndex = 0u;
                   slotIndex < sdsmReduceResultRing_.size(); ++slotIndex) {
                const DynamicBufferSlot &slot =
                    sdsmReduceResultRing_[slotIndex];
                const uint64_t expectedPublishedFrame =
                    slotIndex < sdsmReduceResultRingPublishedFrames_.size()
                        ? sdsmReduceResultRingPublishedFrames_[slotIndex]
                        : std::numeric_limits<uint64_t>::max();
                if (!slot.buffer || !slot.buffer->valid()) {
                  NURI_LOG_WARNING(
                      "ShadowRenderer::updateShadowFrameData: GPU SDSM ring "
                      "slot=%zu bufferValid=0 expectedFrame=%llu readOk=0 "
                      "valid=0 sourceFrame=0 rawDeviceMinMax=(0.000000, "
                      "0.000000)",
                      slotIndex,
                      static_cast<unsigned long long>(
                          expectedPublishedFrame ==
                                  std::numeric_limits<uint64_t>::max()
                              ? 0u
                              : expectedPublishedFrame));
                  continue;
                }

                SdsmGpuMinMaxResult gpuResult{};
                auto readResult = gpu_.readBuffer(
                    slot.buffer->handle(), 0u,
                    std::as_writable_bytes(
                        std::span<SdsmGpuMinMaxResult>(&gpuResult, 1u)));
                const bool readOk = !readResult.hasError();
                NURI_LOG_WARNING(
                    "ShadowRenderer::updateShadowFrameData: GPU SDSM ring "
                    "slot=%zu bufferValid=1 expectedFrame=%llu readOk=%u "
                    "valid=%u sourceFrame=%u rawDeviceMinMax=(%.6f, %.6f)",
                    slotIndex,
                    static_cast<unsigned long long>(
                        expectedPublishedFrame ==
                                std::numeric_limits<uint64_t>::max()
                            ? 0u
                            : expectedPublishedFrame),
                    readOk ? 1u : 0u, gpuResult.valid,
                    gpuResult.sourceFrameIndex, gpuResult.rawDeviceMinMax.x,
                    gpuResult.rawDeviceMinMax.y);
              }
            }
            if (!suppressGpuWarmupWarning) {
              activateReductionFallback(fallbackReason);
            }
            reuseCachedSdsmRangeOrFallback(anyReadbackError
                                               ? ShadowSdsmStatus::Invalid
                                               : ShadowSdsmStatus::Stale,
                                           fallbackReason);
          } else {
            sdsmState_.gpuReductionConsecutiveMissingResultFrames_ = 0u;
            sdsmState_.loggedGpuReductionFallbackWarning_ = false;
            sdsmState_.loggedGpuResultRingDiagnosticWarning_ = false;
            shadowDebugFrameData_.sdsm.sourceFrameIndex =
                selectedGpuResult->sourceFrameIndex;
            publishGpuReductionResultDebug(selectedGpuResultSlot,
                                           selectedGpuResult->sourceFrameIndex,
                                           false);
            consumeRawDeviceMinMax(selectedGpuResult->rawDeviceMinMax.x,
                                   selectedGpuResult->rawDeviceMinMax.y,
                                   "invalid_raw_device_depths",
                                   "invalid_raw_linear_depths");
          }
          frame.metrics.shadow.sdsmReductionSourceSamples = 1u;
          frame.metrics.shadow.sdsmCpuReductionTimeMs = elapsedMilliseconds(
              reductionStart, std::chrono::steady_clock::now());
        } else {
          const auto reductionStart = std::chrono::steady_clock::now();
          sdsmState_.gpuReductionConsecutiveMissingResultFrames_ = 0u;
          const TextureHandle sdsmTexture =
              frame.sharedResources.sceneDepthPyramidTextures
                  [frame.sharedResources.sceneDepthPyramidLevelCount - 1u];
          sdsmLog.sourceTextureValid = nuri::isValid(sdsmTexture);
          if (!nuri::isValid(sdsmTexture)) {
            reuseCachedSdsmRangeOrFallback(ShadowSdsmStatus::Unavailable,
                                           "invalid_source_texture");
          } else {
            std::array<std::byte, sizeof(float) * 2u> sdsmBytes{};
            const TextureReadbackRegion readbackRegion{
                .x = 0u,
                .y = 0u,
                .width = 1u,
                .height = 1u,
                .mipLevel = 0u,
                .layer = 0u,
            };
            auto readResult =
                gpu_.readTexture(sdsmTexture, readbackRegion, sdsmBytes);
            if (readResult.hasError()) {
              sdsmLog.readbackError = true;
              reuseCachedSdsmRangeOrFallback(ShadowSdsmStatus::Invalid,
                                             "source_readback_error");
            } else {
              std::array<float, 2u> rawDeviceDepths{};
              std::memcpy(rawDeviceDepths.data(), sdsmBytes.data(),
                          sizeof(rawDeviceDepths));
              consumeRawDeviceMinMax(rawDeviceDepths[0], rawDeviceDepths[1],
                                     "invalid_raw_device_depths",
                                     "invalid_raw_linear_depths");
            }
          }
          frame.metrics.shadow.sdsmReductionSourceSamples = 1u;
          frame.metrics.shadow.sdsmCpuReductionTimeMs = elapsedMilliseconds(
              reductionStart, std::chrono::steady_clock::now());
        }
      }
    }
  }
  shadowDebugFrameData_.sdsm.effectiveRangeNear = effectiveSplitRange.nearDepth;
  shadowDebugFrameData_.sdsm.effectiveRangeFar = effectiveSplitRange.farDepth;
  const bool hasHistogramEffectiveSplitSource =
      sdsmMode == ShadowSdsmMode::Histogram &&
      !shadowDebugFrameData_.sdsm.fixedFallbackActive &&
      sdsmState_.hasValidSdsmHistogramSplits_ &&
      sdsmState_.sdsmHistogramCascadeCount_ == cascadeCount;
  if (hasHistogramEffectiveSplitSource) {
    const float histogramDistributionActivationDepth =
        computeSdsmFarUpdateThreshold(
            sdsmState_.hasValidSdsmFarCascadeTexelSize_
                ? sdsmState_.sdsmFarCascadeTexelWorldSize_
                : 0.0f);
    const float histogramDistributionFar = std::max(
        histogramSplitDepths[cascadeCount], histogramSplitDepths[0] + 1.0e-4f);
    // When histogram coverage no longer extends meaningfully into the far
    // cascade, keep the fixed split layout stable. This avoids churn from
    // redistributing internal splits based on near-only histogram content.
    if (histogramDistributionFar >
        minimumFarDepth + histogramDistributionActivationDepth) {
      effectiveSplitDepths = buildHistogramSplitDepthsForRange(
          histogramSplitDepths, effectiveSplitRange);
    }
  }
  shadowDebugFrameData_.sdsm.minMaxSplitDepths = minMaxSplitDepths;
  shadowDebugFrameData_.sdsm.histogramSplitDepths = histogramSplitDepths;
  shadowDebugFrameData_.sdsm.effectiveSplitDepths = effectiveSplitDepths;
  const bool reuseFrozenFit = freezeShadowFits && hasFrozenShadowFit_ &&
                              frozenShadowLightId_ == selectedLightId &&
                              frozenShadowMapSize_ == shadowMapSize &&
                              frozenCascadeCount_ == cascadeCount;
  if (reuseFrozenFit) {
    for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
         ++cascadeIndex) {
      const shadow_detail::DirectionalShadowFit &fit =
          frozenShadowFits_[cascadeIndex];
      currentRawShadowFits_[cascadeIndex] = fit;
      writeShadowCascadeFit(fit, cascadeIndex, settings,
                            frame.sharedResources.shadowCompareSamplerId,
                            frame.sharedResources.shadowRawSamplerId,
                            shadowDepthTextures_[cascadeIndex], gpu_,
                            shadowFrameCpuData_, shadowDebugFrameData_);
      if (settings.stabilizeCascades) {
        cascadeStabilizationHistory_.fits[cascadeIndex] = fit;
      }
    }
  } else {
    for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
         ++cascadeIndex) {
      shadow_detail::DirectionalShadowFit fit{};
      if (cascadeCount == 1u) {
        fit = fitSingleShadowMapForRange(
            frame.camera, effectiveSplitRange, lightDirection, shadowMapSize,
            settings.stabilizeCascades, hasCasterBounds,
            hasAnimatedGeometryOverrides, casterBounds.min_, casterBounds.max_);
      } else {
        fit = shadow_detail::fitDirectionalShadowCascadeSlice(
            frame.camera, effectiveSplitDepths[cascadeIndex],
            effectiveSplitDepths[cascadeIndex + 1u], lightDirection,
            shadowMapSize,
            hasAnimatedGeometryOverrides
                ? std::span<const glm::vec3>()
                : std::span<const glm::vec3>(casterPoints.data(),
                                             casterPoints.size()),
            settings.stabilizeCascades);
      }
      if (canReuseCascadeStabilizationHistory) {
        shadow_detail::applyDirectionalShadowFitHysteresis(
            fit, cascadeStabilizationHistory_.fits[cascadeIndex],
            shadowMapSize);
      }
      frozenShadowFits_[cascadeIndex] = fit;
      currentRawShadowFits_[cascadeIndex] = fit;
      writeShadowCascadeFit(fit, cascadeIndex, settings,
                            frame.sharedResources.shadowCompareSamplerId,
                            frame.sharedResources.shadowRawSamplerId,
                            shadowDepthTextures_[cascadeIndex], gpu_,
                            shadowFrameCpuData_, shadowDebugFrameData_);
      if (settings.stabilizeCascades) {
        cascadeStabilizationHistory_.fits[cascadeIndex] = fit;
      }
    }
    if (freezeShadowFits) {
      hasFrozenShadowFit_ = true;
      frozenShadowLightId_ = selectedLightId;
      frozenShadowMapSize_ = shadowMapSize;
      frozenCascadeCount_ = cascadeCount;
    }
  }
  if (settings.stabilizeCascades) {
    cascadeStabilizationHistory_.valid = true;
    cascadeStabilizationHistory_.lightId = selectedLightId;
    cascadeStabilizationHistory_.shadowMapSize = shadowMapSize;
    cascadeStabilizationHistory_.cascadeCount = cascadeCount;
    for (uint32_t cascadeIndex = cascadeCount;
         cascadeIndex < kMaxShadowCascades; ++cascadeIndex) {
      cascadeStabilizationHistory_.fits[cascadeIndex] = {};
    }
  }
  if ((sdsmMode == ShadowSdsmMode::PreviousFrameMinMax ||
       sdsmMode == ShadowSdsmMode::Histogram) &&
      shadowDebugFrameData_.sdsm.status == ShadowSdsmStatus::Active &&
      cascadeCount > 0u) {
    const float farCascadeTexelWorldSize =
        shadowDebugFrameData_.cascades[cascadeCount - 1u].texelWorldSize;
    if (std::isfinite(farCascadeTexelWorldSize) &&
        farCascadeTexelWorldSize > 0.0f) {
      sdsmState_.hasValidSdsmFarCascadeTexelSize_ = true;
      sdsmState_.sdsmFarCascadeTexelWorldSize_ = farCascadeTexelWorldSize;
    } else {
      sdsmState_.hasValidSdsmFarCascadeTexelSize_ = false;
      sdsmState_.sdsmFarCascadeTexelWorldSize_ = 0.0f;
    }
  }
  const ShadowSdsmDebugFrameData &sdsm = shadowDebugFrameData_.sdsm;
  const std::string_view sdsmModeName = shadowSdsmModeName(sdsm.mode);
  const std::string_view sdsmStatusName = shadowSdsmStatusName(sdsm.status);
  const uint64_t sourceFrameIndex = sdsm.sourceFrameIndex;
  const uint64_t sourceLag =
      sourceFrameIndex != std::numeric_limits<uint64_t>::max() &&
              frame.frameIndex >= sourceFrameIndex
          ? (frame.frameIndex - sourceFrameIndex)
          : 0u;
  const bool opaqueDepthPyramidEnabled =
      frame.settings != nullptr && frame.settings->opaque.enableDepthPyramid;
  const auto logSdsmSnapshot = [&](LogLevel logLevel, const char *label) {
    logMessagef(
        logLevel,
        "ShadowRenderer::updateShadowFrameData: %s frame=%llu mode=%s "
        "status=%s reason=%s fixedFallback=%u reusedCached=%u "
        "opaqueDepthPyramid=%u sourceFrameAvailable=%u sourceFrame=%llu "
        "sourceLag=%llu sourceLevels=%u hasValidSource=%u "
        "sourceTextureValid=%u readbackError=%u rawDepthsValid=%u "
        "rawLinearValid=%u historyBefore=[range=%u texel=%u] "
        "historyNow=[range=%u texel=%u] settings=[maxDistance=%.3f "
        "blend=%.3f cascades=%u "
        "lambda=%.3f] camera=[near=%.3f far=%.3f] "
        "values=[rawDevice=(%.6f, %.6f) rawLinear=(%.6f, %.6f) "
        "smoothed=(%.6f, %.6f) fixed=(%.6f, %.6f) effective=(%.6f, %.6f) "
        "minFar=%.6f farTexel=%.6f "
        "fixedSplits=(%.6f, %.6f, %.6f, %.6f, %.6f) "
        "effectiveSplits=(%.6f, %.6f, %.6f, %.6f, %.6f)]",
        label, static_cast<unsigned long long>(frame.frameIndex),
        sdsmModeName.data(), sdsmStatusName.data(), sdsmLog.reason.data(),
        sdsm.fixedFallbackActive ? 1u : 0u, sdsmLog.reusedCachedRange ? 1u : 0u,
        opaqueDepthPyramidEnabled ? 1u : 0u,
        sdsmLog.sourceFrameAvailable ? 1u : 0u,
        static_cast<unsigned long long>(
            sourceFrameIndex == std::numeric_limits<uint64_t>::max()
                ? 0u
                : sourceFrameIndex),
        static_cast<unsigned long long>(sourceLag), sdsmLog.sourceLevelCount,
        sdsmLog.hasValidSource ? 1u : 0u, sdsmLog.sourceTextureValid ? 1u : 0u,
        sdsmLog.readbackError ? 1u : 0u, sdsmLog.rawDepthsValid ? 1u : 0u,
        sdsmLog.rawLinearValid ? 1u : 0u, sdsmLog.historyRangeValid ? 1u : 0u,
        sdsmLog.historyTexelValid ? 1u : 0u,
        sdsmState_.hasValidSdsmRange_ ? 1u : 0u,
        sdsmState_.hasValidSdsmFarCascadeTexelSize_ ? 1u : 0u,
        settings.maxDistance, settings.sdsmTemporalBlend, cascadeCount,
        settings.splitLambda, frame.camera.nearPlane, frame.camera.farPlane,
        sdsm.rawDeviceMin, sdsm.rawDeviceMax, sdsm.rawLinearMin,
        sdsm.rawLinearMax, sdsm.smoothedLinearMin, sdsm.smoothedLinearMax,
        sdsm.fixedRangeNear, sdsm.fixedRangeFar, sdsm.effectiveRangeNear,
        sdsm.effectiveRangeFar, minimumFarDepth,
        sdsmState_.sdsmFarCascadeTexelWorldSize_, sdsm.fixedSplitDepths[0],
        sdsm.fixedSplitDepths[1], sdsm.fixedSplitDepths[2],
        sdsm.fixedSplitDepths[3], sdsm.fixedSplitDepths[4],
        sdsm.effectiveSplitDepths[0], sdsm.effectiveSplitDepths[1],
        sdsm.effectiveSplitDepths[2], sdsm.effectiveSplitDepths[3],
        sdsm.effectiveSplitDepths[4]);
  };
  const bool benignMissingPreviousFrame =
      sdsm.status == ShadowSdsmStatus::Unavailable &&
      sdsmLog.reason == "missing_previous_frame_data" &&
      !sdsmLog.sourceFrameAvailable && !sdsmLog.historyRangeValid &&
      !sdsmState_.hasValidSdsmRange_;
  const bool warmupGpuResultMissing =
      sdsm.status == ShadowSdsmStatus::Stale &&
      (sdsmLog.reason == "gpu_result_invalid_or_stale" ||
       sdsmLog.reason == "gpu_histogram_invalid_or_stale") &&
      !sdsmLog.readbackError && suppressGpuWarmupWarning;
  const bool warningStatus = !benignMissingPreviousFrame &&
                             (sdsm.status == ShadowSdsmStatus::Unavailable ||
                              sdsm.status == ShadowSdsmStatus::Stale ||
                              sdsm.status == ShadowSdsmStatus::Invalid);
  if (warningStatus && !warmupGpuResultMissing) {
    const bool warningRefreshDue =
        sdsmState_.lastLoggedSdsmWarningFrameIndex_ ==
            std::numeric_limits<uint64_t>::max() ||
        frame.frameIndex >= sdsmState_.lastLoggedSdsmWarningFrameIndex_ +
                                kSdsmDiagnosticRefreshFrames;
    const bool warningChanged =
        sdsm.status != sdsmState_.lastLoggedSdsmWarningStatus_ ||
        sdsm.fixedFallbackActive !=
            sdsmState_.lastLoggedSdsmWarningFixedFallbackActive_ ||
        sdsmLog.reusedCachedRange !=
            sdsmState_.lastLoggedSdsmWarningReusedCachedRange_;
    if (warningChanged || warningRefreshDue) {
      logSdsmSnapshot(LogLevel::Warning, "SDSM source warning");
      sdsmState_.lastLoggedSdsmWarningStatus_ = sdsm.status;
      sdsmState_.lastLoggedSdsmWarningFixedFallbackActive_ =
          sdsm.fixedFallbackActive;
      sdsmState_.lastLoggedSdsmWarningReusedCachedRange_ =
          sdsmLog.reusedCachedRange;
      sdsmState_.lastLoggedSdsmWarningSourceFrameIndex_ = sourceFrameIndex;
      sdsmState_.lastLoggedSdsmWarningFrameIndex_ = frame.frameIndex;
    }
  } else {
    sdsmState_.lastLoggedSdsmWarningStatus_ = ShadowSdsmStatus::Disabled;
    sdsmState_.lastLoggedSdsmWarningFixedFallbackActive_ = false;
    sdsmState_.lastLoggedSdsmWarningReusedCachedRange_ = false;
    sdsmState_.lastLoggedSdsmWarningSourceFrameIndex_ =
        std::numeric_limits<uint64_t>::max();
    sdsmState_.lastLoggedSdsmWarningFrameIndex_ =
        std::numeric_limits<uint64_t>::max();
  }
  const uint32_t shadowFlags =
      kShadowFrameFlagEnabled | shadowDebugFrameFlags(settings.debug);

  shadowFrameCpuData_.flagsCascadeCountLightIndex = glm::uvec4(
      shadowFlags, cascadeCount, selectedLightIndex,
      static_cast<uint32_t>(sanitizeShadowFilterMode(settings.filterMode)));
  shadowFrameCpuData_.filterParams = glm::uvec4(
      settings.pcfSampleCount, settings.pcssBlockerSampleCount,
      settings.pcssFilterSampleCount, settings.debug.poissonRotationSeed);
  shadowFrameCpuData_.fadeParams =
      glm::vec4(shadowDebugFrameData_.cascades[0].splitNear,
                shadowDebugFrameData_.cascades[cascadeCount - 1u].splitFar,
                settings.cascadeBlendFraction, settings.pcssLightRadiusScale);

  shadowDebugFrameData_.selectedShadowLightId = selectedLightId;

  frame.sharedResources.shadowDebugFrameData = shadowDebugFrameData_;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
ShadowRenderer::buildShadowDraws(RenderFrameContext &frame, uint32_t frameSlot,
                                 const ForwardSceneGpuData &sceneGpu) {
  const RenderSettings &settings = renderSettingsOrDefault(frame);
  RenderSettings::ShadowSettings shadowSettings = settings.shadow;
  sanitizeShadowSettings(shadowSettings);
  const auto emitShadowDiagnostics = [&]() {
    if (!shadowSettings.debug.logDiagnostics) {
      diagnosticLogState_ = {};
      return;
    }

    const uint64_t signature = buildShadowDiagnosticSignature(
        frame, shadowSettings, shadowDebugFrameData_, frame.metrics.shadow,
        hasActiveShadowLightForFrame_, hasPreparedShadowDepthPasses_,
        hasPreparedShadowPreviewPass_);
    const bool firstLog = !diagnosticLogState_.hasLastSignature;
    const bool signatureChanged =
        firstLog || diagnosticLogState_.lastSignature != signature;
    const bool intervalDue =
        firstLog ||
        diagnosticLogState_.lastLoggedFrameIndex ==
            std::numeric_limits<uint64_t>::max() ||
        frame.frameIndex >=
            diagnosticLogState_.lastLoggedFrameIndex +
                shadowSettings.debug.diagnosticLogIntervalFrames;
    if ((shadowSettings.debug.diagnosticLogOnlyOnChange && !signatureChanged) ||
        (!shadowSettings.debug.diagnosticLogOnlyOnChange && !signatureChanged &&
         !intervalDue)) {
      return;
    }

    const LogLevel logLevel = shadowSettings.debug.diagnosticLogLevel;
    const ShadowSdsmDebugFrameData &sdsm = shadowDebugFrameData_.sdsm;
    const auto orthoExtentsFromBounds = [](const glm::vec4 &boundsMin,
                                           const glm::vec4 &boundsMax) {
      return glm::vec3(boundsMax - boundsMin);
    };
    const auto lightSpaceCenterFromBounds = [](const glm::vec4 &boundsMin,
                                               const glm::vec4 &boundsMax) {
      return glm::vec3(boundsMin + boundsMax) * 0.5f;
    };
    const auto lightDirectionFromView = [](const glm::mat4 &lightView) {
      const glm::mat4 inverseLightView = glm::inverse(lightView);
      return shadow_detail::normalizeSafe(-glm::vec3(inverseLightView[2]),
                                          glm::vec3(0.0f, -1.0f, 0.0f));
    };
    const uint64_t sourceFrameIndex = sdsm.sourceFrameIndex;
    const uint64_t sourceLag =
        sourceFrameIndex != std::numeric_limits<uint64_t>::max() &&
                frame.frameIndex >= sourceFrameIndex
            ? (frame.frameIndex - sourceFrameIndex)
            : 0u;
    const uint64_t selectedLightValue =
        isValid(shadowDebugFrameData_.selectedShadowLightId)
            ? shadowDebugFrameData_.selectedShadowLightId.value
            : 0u;
    const glm::vec3 cameraPos = glm::vec3(frame.camera.cameraPos);
    logMessagef(
        logLevel,
        "ShadowRenderer::buildShadowDraws: Shadow diagnostics summary "
        "frame=%llu prepared=[depth=%u preview=%u] activeShadowLight=%u "
        "selectedLight=%llu camera=[pos=(%.3f, %.3f, %.3f) near=%.3f "
        "far=%.3f aspect=%.3f projection=%s] settings=[split=%s filter=%s "
        "stabilize=%u freeze=(%u,%u) casterCull=%u map=%u cascades=%u "
        "maxDistance=%.3f lambda=%.3f blend=%.3f bias=(%.6f, %.3f, %.3f) "
        "pcf=%u pcss=(%u,%u,%.3f,%.3f,%.3f)] metrics=[draws=%u culled=%u "
        "indices=%u staticDynamic=(%u,%u) staticCacheReused=%u "
        "staticOnly=(candidates=%u reused=%u misses=%u/%u/%u/%u/%u) "
        "cascadeBytes=%llu filterBudget=%u pcssCost=(%u,%u) "
        "sdsmCompute=%u sdsmReadback=%u sdsmSamples=(%u,%u) "
        "sdsmCpuMs=(%.3f, %.3f) gpuMs=(%.3f, %.3f, %.3f) "
        "gpuFrames=(%llu, %llu, %llu)] "
        "sdsm=[mode=%s requested=%s active=%s status=%s fixedFallback=%u "
        "reductionFallback=%u sourceFrame=%llu sourceLag=%llu "
        "rawDevice=(%.6f, %.6f) rawLinear=(%.6f, %.6f) "
        "smoothed=(%.6f, %.6f) fixedRange=(%.6f, %.6f) "
        "histogramRange=(%.6f, %.6f) effectiveRange=(%.6f, %.6f)]",
        static_cast<unsigned long long>(frame.frameIndex),
        hasPreparedShadowDepthPasses_ ? 1u : 0u,
        hasPreparedShadowPreviewPass_ ? 1u : 0u,
        hasActiveShadowLightForFrame_ ? 1u : 0u,
        static_cast<unsigned long long>(selectedLightValue), cameraPos.x,
        cameraPos.y, cameraPos.z, frame.camera.nearPlane, frame.camera.farPlane,
        frame.camera.aspectRatio,
        projectionTypeName(frame.camera.projectionType).data(),
        shadowCascadeSplitModeName(shadowSettings.splitMode).data(),
        shadowFilterModeName(shadowSettings.filterMode).data(),
        shadowSettings.stabilizeCascades ? 1u : 0u,
        shadowSettings.debug.freezeCascades ? 1u : 0u,
        shadowSettings.debug.freezeLightView ? 1u : 0u,
        shadowSettings.debug.enableCascadeCasterCulling ? 1u : 0u,
        shadowSettings.shadowMapSize, shadowSettings.cascadeCount,
        shadowSettings.maxDistance, shadowSettings.splitLambda,
        shadowSettings.cascadeBlendFraction, shadowSettings.constantBias,
        shadowSettings.slopeBias, shadowSettings.normalBias,
        shadowSettings.pcfSampleCount, shadowSettings.pcssBlockerSampleCount,
        shadowSettings.pcssFilterSampleCount,
        shadowSettings.pcssLightRadiusScale,
        shadowSettings.pcssSearchRadiusClampTexels,
        shadowSettings.pcssFilterRadiusClampTexels,
        frame.metrics.shadow.totalDraws, frame.metrics.shadow.totalCulledDraws,
        frame.metrics.shadow.totalIndexCountEstimate,
        frame.metrics.shadow.staticCasterEntries,
        frame.metrics.shadow.dynamicCasterEntries,
        frame.metrics.shadow.staticCacheReused,
        frame.metrics.shadow.staticOnlyCandidateCount,
        frame.metrics.shadow.reusedStaticOnlyCascadeCount,
        frame.metrics.shadow.staticOnlyReuseMissStaticCacheRebuiltCount,
        frame.metrics.shadow.staticOnlyReuseMissDynamicCasterCount,
        frame.metrics.shadow.staticOnlyReuseMissNoPreviousCount,
        frame.metrics.shadow.staticOnlyReuseMissRasterStateChangedCount,
        frame.metrics.shadow.staticOnlyReuseMissAdaptiveRefreshCount,
        static_cast<unsigned long long>(
            frame.metrics.shadow.cascadeTextureBytes),
        frame.metrics.shadow.filterSampleBudget,
        frame.metrics.shadow.pcssMaxSamplesPerReceiver,
        frame.metrics.shadow.pcssMaxSamplesPerBlendedReceiver,
        frame.metrics.shadow.sdsmComputePassCount,
        frame.metrics.shadow.sdsmReadbackBytes,
        frame.metrics.shadow.sdsmReductionSourceSamples,
        frame.metrics.shadow.sdsmHistogramSourceSamples,
        frame.metrics.shadow.sdsmCpuReductionTimeMs,
        frame.metrics.shadow.sdsmCpuHistogramTimeMs,
        frame.metrics.shadow.gpuTimeMs, frame.metrics.shadow.depthGpuTimeMs,
        frame.metrics.shadow.sdsmGpuTimeMs,
        static_cast<unsigned long long>(
            frame.metrics.shadow.gpuTimingSourceFrameIndex ==
                    std::numeric_limits<uint64_t>::max()
                ? 0u
                : frame.metrics.shadow.gpuTimingSourceFrameIndex),
        static_cast<unsigned long long>(
            frame.metrics.shadow.depthGpuTimingSourceFrameIndex ==
                    std::numeric_limits<uint64_t>::max()
                ? 0u
                : frame.metrics.shadow.depthGpuTimingSourceFrameIndex),
        static_cast<unsigned long long>(
            frame.metrics.shadow.sdsmGpuTimingSourceFrameIndex ==
                    std::numeric_limits<uint64_t>::max()
                ? 0u
                : frame.metrics.shadow.sdsmGpuTimingSourceFrameIndex),
        shadowSdsmModeName(sdsm.mode).data(),
        shadowSdsmReductionBackendName(sdsm.requestedReductionBackend).data(),
        shadowSdsmReductionBackendName(sdsm.activeReductionBackend).data(),
        shadowSdsmStatusName(sdsm.status).data(),
        sdsm.fixedFallbackActive ? 1u : 0u,
        sdsm.reductionFallbackActive ? 1u : 0u,
        static_cast<unsigned long long>(
            sourceFrameIndex == std::numeric_limits<uint64_t>::max()
                ? 0u
                : sourceFrameIndex),
        static_cast<unsigned long long>(sourceLag), sdsm.rawDeviceMin,
        sdsm.rawDeviceMax, sdsm.rawLinearMin, sdsm.rawLinearMax,
        sdsm.smoothedLinearMin, sdsm.smoothedLinearMax, sdsm.fixedRangeNear,
        sdsm.fixedRangeFar, sdsm.histogramTrimmedRangeNear,
        sdsm.histogramTrimmedRangeFar, sdsm.effectiveRangeNear,
        sdsm.effectiveRangeFar);

    const uint32_t cascadeCount =
        std::min(shadowDebugFrameData_.cascadeCount, kMaxShadowCascades);
    for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
         ++cascadeIndex) {
      const ShadowCascadeDebugFrameData &cascade =
          shadowDebugFrameData_.cascades[cascadeIndex];
      const DiagnosticLogState::CascadeLightState &previousLightState =
          diagnosticLogState_.cascadeLightStates[cascadeIndex];
      const glm::vec3 unsnappedCenter = glm::vec3(cascade.unsnappedCenter);
      const glm::vec3 snappedCenter = glm::vec3(cascade.snappedCenter);
      const glm::vec3 snapDelta = snappedCenter - unsnappedCenter;
      logMessagef(
          logLevel,
          "ShadowRenderer::buildShadowDraws: Shadow diagnostics cascade "
          "frame=%llu idx=%u split=(%.6f, %.6f) texel=%.6f bindless=%u "
          "draws=(total=%u culled=%u static=%u dynamic=%u) reuse=%s "
          "candidate=%u prevValid=%u adaptiveRefresh=%u "
          "sigChanges=(light=%u bias=%u caster=%u) "
          "rasterSig=(0x%016llX, 0x%016llX) "
          "lightSig=(0x%016llX, 0x%016llX) "
          "biasSig=(0x%016llX, 0x%016llX) "
          "casterSig=(0x%016llX, 0x%016llX) "
          "boundsMin=(%.3f, %.3f, %.3f) boundsMax=(%.3f, %.3f, %.3f) "
          "centers=[unsnapped=(%.3f, %.3f, %.3f) snapped=(%.3f, %.3f, %.3f) "
          "delta=(%.5f, %.5f, %.5f)]",
          static_cast<unsigned long long>(frame.frameIndex), cascadeIndex,
          cascade.splitNear, cascade.splitFar, cascade.texelWorldSize,
          cascade.textureBindlessId, cascade.drawCount, cascade.culledCount,
          cascade.staticDrawCount, cascade.dynamicDrawCount,
          shadowStaticOnlyReuseStatusName(cascade.staticOnlyReuseStatus).data(),
          cascade.staticOnlyReuseCandidate ? 1u : 0u,
          cascade.staticOnlyReusePreviousValid ? 1u : 0u,
          cascade.staticOnlyReuseAdaptiveRefresh ? 1u : 0u,
          cascade.staticOnlyReuseLightViewProjChanged ? 1u : 0u,
          cascade.staticOnlyReuseBiasChanged ? 1u : 0u,
          cascade.staticOnlyReuseCasterSignatureChanged ? 1u : 0u,
          static_cast<unsigned long long>(
              cascade.currentStaticOnlyRasterSignature),
          static_cast<unsigned long long>(
              cascade.previousStaticOnlyRasterSignature),
          static_cast<unsigned long long>(
              cascade.currentStaticOnlyLightViewProjSignature),
          static_cast<unsigned long long>(
              cascade.previousStaticOnlyLightViewProjSignature),
          static_cast<unsigned long long>(
              cascade.currentStaticOnlyBiasSignature),
          static_cast<unsigned long long>(
              cascade.previousStaticOnlyBiasSignature),
          static_cast<unsigned long long>(
              cascade.currentStaticOnlyCasterSignature),
          static_cast<unsigned long long>(
              cascade.previousStaticOnlyCasterSignature),
          cascade.lightSpaceBoundsMin.x, cascade.lightSpaceBoundsMin.y,
          cascade.lightSpaceBoundsMin.z, cascade.lightSpaceBoundsMax.x,
          cascade.lightSpaceBoundsMax.y, cascade.lightSpaceBoundsMax.z,
          unsnappedCenter.x, unsnappedCenter.y, unsnappedCenter.z,
          snappedCenter.x, snappedCenter.y, snappedCenter.z, snapDelta.x,
          snapDelta.y, snapDelta.z);

      const bool hasPreviousLightState = previousLightState.valid;
      const bool lightViewProjChanged =
          hasPreviousLightState &&
          !rawBytesEqual(cascade.lightViewProj,
                         previousLightState.lightViewProj);
      if (lightViewProjChanged) {
        const bool basisChanged =
            !rawBytesEqual(cascade.lightView, previousLightState.lightView);
        const glm::vec3 previousOrthoExtents =
            orthoExtentsFromBounds(previousLightState.lightSpaceBoundsMin,
                                   previousLightState.lightSpaceBoundsMax);
        const glm::vec3 currentOrthoExtents = orthoExtentsFromBounds(
            cascade.lightSpaceBoundsMin, cascade.lightSpaceBoundsMax);
        const bool orthoExtentsChanged =
            !rawBytesEqual(previousOrthoExtents, currentOrthoExtents);
        const glm::vec2 previousSnappedLightCenter = glm::vec2(
            lightSpaceCenterFromBounds(previousLightState.lightSpaceBoundsMin,
                                       previousLightState.lightSpaceBoundsMax));
        const glm::vec2 currentSnappedLightCenter =
            glm::vec2(lightSpaceCenterFromBounds(cascade.lightSpaceBoundsMin,
                                                 cascade.lightSpaceBoundsMax));
        const bool snappedCenterChanged = !rawBytesEqual(
            previousSnappedLightCenter, currentSnappedLightCenter);
        const glm::vec2 previousDepthRange(
            previousLightState.lightSpaceBoundsMin.z,
            previousLightState.lightSpaceBoundsMax.z);
        const glm::vec2 currentDepthRange(cascade.lightSpaceBoundsMin.z,
                                          cascade.lightSpaceBoundsMax.z);
        const bool orthoDepthChanged =
            !rawBytesEqual(previousDepthRange, currentDepthRange);
        const glm::vec3 previousLightDirection =
            lightDirectionFromView(previousLightState.lightView);
        const glm::vec3 currentLightDirection =
            lightDirectionFromView(cascade.lightView);
        const glm::vec3 previousUnsnappedCenter =
            glm::vec3(previousLightState.unsnappedCenter);
        const glm::vec3 unsnappedCenterDelta =
            unsnappedCenter - previousUnsnappedCenter;
        const glm::vec3 previousSnappedCenter =
            glm::vec3(previousLightState.snappedCenter);
        const glm::vec3 snappedCenterDelta =
            snappedCenter - previousSnappedCenter;
        logMessagef(
            logLevel,
            "ShadowRenderer::buildShadowDraws: Shadow diagnostics light "
            "change frame=%llu idx=%u components=(basis=%u orthoExtents=%u "
            "orthoDepth=%u snappedCenter=%u) basisDir=[prev=(%.6f, %.6f, "
            "%.6f) curr=(%.6f, %.6f, %.6f)] ortho=[prevExtents=(%.6f, %.6f, "
            "%.6f) currExtents=(%.6f, %.6f, %.6f) prevDepth=(%.6f, %.6f) "
            "currDepth=(%.6f, %.6f)] snappedCenter=[prevLight=(%.6f, %.6f) "
            "currLight=(%.6f, %.6f) prevWorld=(%.6f, %.6f, %.6f) currWorld="
            "(%.6f, %.6f, %.6f) delta=(%.6f, %.6f, %.6f)] unsnappedWorld="
            "[prev=(%.6f, %.6f, %.6f) curr=(%.6f, %.6f, %.6f) delta=(%.6f, "
            "%.6f, %.6f)]",
            static_cast<unsigned long long>(frame.frameIndex), cascadeIndex,
            basisChanged ? 1u : 0u, orthoExtentsChanged ? 1u : 0u,
            orthoDepthChanged ? 1u : 0u, snappedCenterChanged ? 1u : 0u,
            previousLightDirection.x, previousLightDirection.y,
            previousLightDirection.z, currentLightDirection.x,
            currentLightDirection.y, currentLightDirection.z,
            previousOrthoExtents.x, previousOrthoExtents.y,
            previousOrthoExtents.z, currentOrthoExtents.x,
            currentOrthoExtents.y, currentOrthoExtents.z, previousDepthRange.x,
            previousDepthRange.y, currentDepthRange.x, currentDepthRange.y,
            previousSnappedLightCenter.x, previousSnappedLightCenter.y,
            currentSnappedLightCenter.x, currentSnappedLightCenter.y,
            previousSnappedCenter.x, previousSnappedCenter.y,
            previousSnappedCenter.z, snappedCenter.x, snappedCenter.y,
            snappedCenter.z, snappedCenterDelta.x, snappedCenterDelta.y,
            snappedCenterDelta.z, previousUnsnappedCenter.x,
            previousUnsnappedCenter.y, previousUnsnappedCenter.z,
            unsnappedCenter.x, unsnappedCenter.y, unsnappedCenter.z,
            unsnappedCenterDelta.x, unsnappedCenterDelta.y,
            unsnappedCenterDelta.z);
      }

      diagnosticLogState_.cascadeLightStates[cascadeIndex] = {
          .valid = true,
          .lightView = cascade.lightView,
          .lightViewProj = cascade.lightViewProj,
          .lightSpaceBoundsMin = cascade.lightSpaceBoundsMin,
          .lightSpaceBoundsMax = cascade.lightSpaceBoundsMax,
          .unsnappedCenter = cascade.unsnappedCenter,
          .snappedCenter = cascade.snappedCenter,
      };
    }
    for (uint32_t cascadeIndex = cascadeCount;
         cascadeIndex < kMaxShadowCascades; ++cascadeIndex) {
      diagnosticLogState_.cascadeLightStates[cascadeIndex].valid = false;
    }

    diagnosticLogState_.hasLastSignature = true;
    diagnosticLogState_.lastSignature = signature;
    diagnosticLogState_.lastLoggedFrameIndex = frame.frameIndex;
  };
  if (frame.scene == nullptr || meshDrawTemplates_.empty()) {
    emitShadowDiagnostics();
    return Result<bool, std::string>::makeResult(true);
  }
  if (activeCascadeCount_ == 0u ||
      shadowFrameCpuData_.flagsCascadeCountLightIndex.y == 0u) {
    emitShadowDiagnostics();
    return Result<bool, std::string>::makeResult(true);
  }

  auto initResult = ensureInitialized();
  if (initResult.hasError()) {
    return initResult;
  }

  const std::span<const Renderable> renderables = frame.scene->renderables();
  const AnimationSceneFrameData *animationSceneData =
      resolveAnimationSceneFrameData(frame);
  const uint64_t transformVersion = frame.scene->transformVersion();
  if (cachedTransformVersion_ != transformVersion ||
      instanceMatrices_.size() != renderables.size()) {
    instanceMatrices_.clear();
    instanceRemap_.clear();
    instanceMatrices_.reserve(renderables.size());
    instanceRemap_.reserve(renderables.size());
    for (uint32_t i = 0u; i < static_cast<uint32_t>(renderables.size()); ++i) {
      instanceMatrices_.push_back(makeInstanceData(renderables[i].modelMatrix));
      instanceRemap_.push_back(i);
    }
    cachedTransformVersion_ = transformVersion;
    std::fill(instanceDataRingUploadVersions_.begin(),
              instanceDataRingUploadVersions_.end(),
              std::numeric_limits<uint64_t>::max());
  }

  if (instanceMatrices_.empty() || instanceRemap_.empty()) {
    emitShadowDiagnostics();
    return Result<bool, std::string>::makeResult(true);
  }

  auto matricesResult = ensureInstanceMatricesRingCapacity(std::max(
      instanceMatrices_.size() * sizeof(InstanceData), sizeof(InstanceData)));
  if (matricesResult.hasError()) {
    return matricesResult;
  }
  auto remapResult = ensureInstanceRemapRingCapacity(
      std::max(instanceRemap_.size() * sizeof(uint32_t), sizeof(uint32_t)));
  if (remapResult.hasError()) {
    return remapResult;
  }

  const bool needsInstanceUpload =
      instanceDataRingUploadVersions_[frameSlot] != cachedTransformVersion_;
  if (needsInstanceUpload) {
    const std::span<const std::byte> matrixBytes{
        reinterpret_cast<const std::byte *>(instanceMatrices_.data()),
        instanceMatrices_.size() * sizeof(InstanceData)};
    auto updateMatricesResult = gpu_.updateBuffer(
        instanceMatricesRing_[frameSlot].buffer->handle(), matrixBytes, 0u);
    if (updateMatricesResult.hasError()) {
      return updateMatricesResult;
    }

    const std::span<const std::byte> remapBytes{
        reinterpret_cast<const std::byte *>(instanceRemap_.data()),
        instanceRemap_.size() * sizeof(uint32_t)};
    auto updateRemapResult = gpu_.updateBuffer(
        instanceRemapRing_[frameSlot].buffer->handle(), remapBytes, 0u);
    if (updateRemapResult.hasError()) {
      return updateRemapResult;
    }
    instanceDataRingUploadVersions_[frameSlot] = cachedTransformVersion_;
  }

  const BufferHandle instanceMatricesBuffer =
      animationSceneData != nullptr
          ? animationSceneData->instanceMatricesBuffer
          : instanceMatricesRing_[frameSlot].buffer->handle();
  const BufferHandle instanceRemapBuffer =
      instanceRemapRing_[frameSlot].buffer->handle();
  const uint64_t instanceMatricesAddress =
      animationSceneData != nullptr
          ? animationSceneData->instanceMatricesAddress
          : gpu_.getBufferDeviceAddress(instanceMatricesBuffer);
  const uint64_t instanceRemapAddress =
      gpu_.getBufferDeviceAddress(instanceRemapBuffer);
  if (sceneGpu.shadowFrameBufferAddress == 0u) {
    emitShadowDiagnostics();
    return Result<bool, std::string>::makeResult(true);
  }
  if (sceneGpu.frameDataAddress == 0u || instanceMatricesAddress == 0u ||
      instanceRemapAddress == 0u) {
    emitShadowDiagnostics();
    return Result<bool, std::string>::makeError(
        "ShadowRenderer::buildShadowDraws: invalid GPU buffer address");
  }

  const uint32_t cascadeCount =
      std::clamp(shadowFrameCpuData_.flagsCascadeCountLightIndex.y, 1u,
                 activeCascadeCount_);
  for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
       ++cascadeIndex) {
    cascadePushConstants_[cascadeIndex].clear();
    cascadeDrawItems_[cascadeIndex].clear();
    cascadePushConstants_[cascadeIndex].reserve(meshDrawTemplates_.size());
    cascadeDrawItems_[cascadeIndex].reserve(meshDrawTemplates_.size());
  }

  const bool enableCascadeCasterCulling =
      settings.shadow.debug.enableCascadeCasterCulling;
  const uint64_t cachePipelineSignature =
      shadowPipelineSignature() ^
      (enableCascadeCasterCulling ? 0x9e3779b97f4a7c15ull : 0ull);
  const bool needsStaticCacheRebuild =
      !staticShadowCasterCacheValid_ ||
      staticShadowCasterCacheTransformVersion_ != transformVersion ||
      staticShadowCasterCacheForcedMeshLod_ != settings.opaque.forcedMeshLod ||
      staticShadowCasterCachePipelineSignature_ != cachePipelineSignature;
  if (needsStaticCacheRebuild) {
    auto cacheResult = rebuildStaticShadowCasterCache(*frame.scene, settings);
    if (cacheResult.hasError()) {
      return cacheResult;
    }
    staticShadowCasterCacheTransformVersion_ = transformVersion;
    staticShadowCasterCacheForcedMeshLod_ = settings.opaque.forcedMeshLod;
    staticShadowCasterCachePipelineSignature_ = cachePipelineSignature;
  }
  const bool staticCacheReused = !needsStaticCacheRebuild;
  std::array<ShadowRenderer::StaticOnlyCascadeReuseState, kMaxShadowCascades>
      currentStaticOnlyCascadeStates{};

  std::pmr::vector<uint32_t> frameDynamicTemplateIndices(memory_);
  frameDynamicTemplateIndices.reserve(dynamicShadowTemplateIndices_.size() +
                                      staticShadowTemplateIndices_.size());
  frameDynamicTemplateIndices.insert(frameDynamicTemplateIndices.end(),
                                     dynamicShadowTemplateIndices_.begin(),
                                     dynamicShadowTemplateIndices_.end());
  std::pmr::vector<uint8_t> staticTemplatesUsingDynamicPath(
      meshDrawTemplates_.size(), uint8_t{0}, memory_);
  if (animationSceneData != nullptr) {
    for (const uint32_t templateIndex : staticShadowTemplateIndices_) {
      if (templateIndex >= meshDrawTemplates_.size()) {
        continue;
      }
      const MeshDrawTemplate &entry = meshDrawTemplates_[templateIndex];
      if (entry.submesh == nullptr ||
          entry.instanceIndex >=
              animationSceneData->geometryOverridesByRenderable.size()) {
        continue;
      }
      const AnimatedRenderableGeometryOverride &geometryOverride =
          animationSceneData
              ->geometryOverridesByRenderable[entry.instanceIndex];
      if (!geometryOverride.enabled ||
          !nuri::isValid(geometryOverride.vertexBuffer) ||
          !animationOverrideCoversSubmesh(geometryOverride, *entry.submesh)) {
        continue;
      }
      staticTemplatesUsingDynamicPath[templateIndex] = 1u;
      frameDynamicTemplateIndices.push_back(templateIndex);
    }
  }

  const uint32_t renderableCount = saturateToU32(renderables.size());
  const auto appendShadowDraw =
      [&](uint32_t cascadeIndex, BufferHandle vertexBuffer,
          BufferHandle indexBuffer, uint64_t indexBufferOffset,
          IndexFormat indexFormat, uint32_t indexCount, uint32_t firstIndex,
          uint32_t firstInstance, uint64_t vertexBufferAddress,
          uint64_t vertexDecodeBufferAddress, uint32_t vertexDecodeIndex,
          uint32_t packedVertexFormat, uint32_t materialIndex, bool doubleSided,
          bool alphaMasked) {
        cascadePushConstants_[cascadeIndex].push_back(PushConstants{
            .frameDataAddress = sceneGpu.frameDataAddress,
            .vertexBufferAddress = vertexBufferAddress,
            .vertexDecodeBufferAddress = vertexDecodeBufferAddress,
            .instanceMatricesAddress = instanceMatricesAddress,
            .instanceRemapAddress = instanceRemapAddress,
            .instanceCentersPhaseAddress = 0u,
            .instanceBaseMatricesAddress = 0u,
            .instanceCount = renderableCount,
            .materialIndex = materialIndex,
            .vertexDecodeIndex = vertexDecodeIndex,
            .packedVertexFormat = packedVertexFormat,
            .timeSeconds = static_cast<float>(frame.timeSeconds),
            .tessNearDistance = 1.0f,
            .tessFarDistance = 8.0f,
            .tessMinFactor = 1.0f,
            .tessMaxFactor = 1.0f,
            .debugVisualizationMode = 0u,
            .shadowCascadeIndex = cascadeIndex,
        });
        const PushConstants &pc = cascadePushConstants_[cascadeIndex].back();

        DrawItem draw{};
        if (alphaMasked) {
          draw.pipeline =
              doubleSided &&
                      nuri::isValid(shadowAlphaDoubleSidedPipelineHandle_)
                  ? shadowAlphaDoubleSidedPipelineHandle_
                  : shadowAlphaPipelineHandle_;
        } else {
          draw.pipeline =
              doubleSided && nuri::isValid(shadowDoubleSidedPipelineHandle_)
                  ? shadowDoubleSidedPipelineHandle_
                  : shadowPipelineHandle_;
        }
        draw.vertexBuffer = vertexBuffer;
        draw.indexBuffer = indexBuffer;
        draw.indexBufferOffset = indexBufferOffset;
        draw.indexFormat = indexFormat;
        draw.indexCount = indexCount;
        draw.instanceCount = 1u;
        draw.firstIndex = firstIndex;
        draw.firstInstance = firstInstance;
        draw.useDepthState = true;
        draw.depthState = {.compareOp = CompareOp::Less,
                           .isDepthWriteEnabled = true};
        draw.depthBiasEnable = true;
        draw.depthBiasConstant = settings.shadow.constantBias;
        draw.depthBiasSlope = settings.shadow.slopeBias;
        draw.depthBiasClamp = 0.0f;
        draw.pushConstants = std::span<const std::byte>(
            reinterpret_cast<const std::byte *>(&pc), sizeof(PushConstants));
        draw.debugLabel = kShadowMeshLabel;
        draw.debugColor = kShadowMeshDebugColor;
        cascadeDrawItems_[cascadeIndex].push_back(draw);
      };
  uint32_t overriddenStaticTemplateCount = 0u;
  for (const uint8_t usesDynamicPath : staticTemplatesUsingDynamicPath) {
    overriddenStaticTemplateCount += usesDynamicPath != 0u ? 1u : 0u;
  }

  std::array<shadow_detail::DirectionalShadowFit, kMaxShadowCascades>
      guardedStaticOnlyFits = currentRawShadowFits_;
  for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
       ++cascadeIndex) {
    const StaticOnlyGuardBandTexels guard = staticOnlyRenderGuardBandTexels(
        currentRawShadowFits_[cascadeIndex],
        reusableStaticOnlyCascadeStates_[cascadeIndex].rawFit,
        reusableStaticOnlyCascadeValid_[cascadeIndex],
        settings.shadow.shadowMapSize);
    applyStaticOnlyGuardBand(
        guardedStaticOnlyFits[cascadeIndex],
        reusableStaticOnlyCascadeStates_[cascadeIndex].rawFit,
        reusableStaticOnlyCascadeValid_[cascadeIndex],
        settings.shadow.shadowMapSize, guard);
  }

  std::array<uint8_t, kMaxShadowCascades> cascadeHasGuardedDynamicDraw{};
  for (const uint32_t templateIndex : frameDynamicTemplateIndices) {
    if (templateIndex >= meshDrawTemplates_.size()) {
      continue;
    }
    const MeshDrawTemplate &entry = meshDrawTemplates_[templateIndex];
    if (entry.renderable == nullptr || entry.submesh == nullptr) {
      continue;
    }
    const std::optional<SubmeshLod> lod =
        resolveShadowLod(*entry.submesh, settings);
    if (!lod.has_value()) {
      continue;
    }

    bool usesAnimatedOverride = false;
    if (animationSceneData != nullptr &&
        entry.instanceIndex <
            animationSceneData->geometryOverridesByRenderable.size()) {
      const AnimatedRenderableGeometryOverride &geometryOverride =
          animationSceneData
              ->geometryOverridesByRenderable[entry.instanceIndex];
      usesAnimatedOverride =
          geometryOverride.enabled &&
          nuri::isValid(geometryOverride.vertexBuffer) &&
          animationOverrideCoversSubmesh(geometryOverride, *entry.submesh);
    }

    std::array<glm::vec3, 8> casterWorldCorners{};
    bool hasCasterCullingBounds = false;
    if (enableCascadeCasterCulling && !usesAnimatedOverride) {
      const BoundingBox worldBounds =
          entry.submesh->bounds.getTransformed(entry.renderable->modelMatrix);
      casterWorldCorners = shadow_detail::computeBoundsCorners(
          worldBounds.min_, worldBounds.max_);
      hasCasterCullingBounds = true;
    }

    for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
         ++cascadeIndex) {
      if (hasCasterCullingBounds &&
          !shadowCasterOverlapsDirectionalShadowFit(
              std::span<const glm::vec3, 8>(casterWorldCorners),
              guardedStaticOnlyFits[cascadeIndex])) {
        continue;
      }
      cascadeHasGuardedDynamicDraw[cascadeIndex] = 1u;
    }
  }

  for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
       ++cascadeIndex) {
    const bool useGuardedStaticOnlyFit =
        cascadeHasGuardedDynamicDraw[cascadeIndex] == 0u;
    const shadow_detail::DirectionalShadowFit &renderedFit =
        useGuardedStaticOnlyFit ? guardedStaticOnlyFits[cascadeIndex]
                                : currentRawShadowFits_[cascadeIndex];
    writeShadowCascadeFit(renderedFit, cascadeIndex, settings.shadow,
                          frame.sharedResources.shadowCompareSamplerId,
                          frame.sharedResources.shadowRawSamplerId,
                          shadowDepthTextures_[cascadeIndex], gpu_,
                          shadowFrameCpuData_, shadowDebugFrameData_);

    StaticOnlyCascadeReuseState &state =
        currentStaticOnlyCascadeStates[cascadeIndex];
    state.renderedFit = renderedFit;
    state.rawFit = currentRawShadowFits_[cascadeIndex];
    state.lightViewProjSignature = makeStaticOnlyCascadeLightViewProjSignature(
        shadowFrameCpuData_.cascades[cascadeIndex]);
    state.biasSignature = makeStaticOnlyCascadeBiasSignature(
        settings.shadow.constantBias, settings.shadow.slopeBias);
    state.casterSignature = kFnvOffsetBasis64;
    state.rasterSignature = makeStaticOnlyCascadeRasterSignatureSeed(
        shadowFrameCpuData_.cascades[cascadeIndex],
        settings.shadow.constantBias, settings.shadow.slopeBias);
  }

  for (const StaticShadowCasterCacheEntry &entry : staticShadowCasterCache_) {
    if (entry.templateIndex < staticTemplatesUsingDynamicPath.size() &&
        staticTemplatesUsingDynamicPath[entry.templateIndex] != 0u) {
      continue;
    }
    for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
         ++cascadeIndex) {
      if (entry.hasCasterCullingBounds) {
        const ShadowCascadeDebugFrameData &cascadeDebug =
            shadowDebugFrameData_.cascades[cascadeIndex];
        const float cullingPadding = std::max(
            cascadeDebug.texelWorldSize * kCullingPaddingTexelMultiplier,
            0.01f);
        if (!shadow_detail::shadowCasterOverlapsLightSpaceBounds(
                std::span<const glm::vec3, 8>(entry.casterWorldCorners),
                cascadeDebug.lightView,
                glm::vec3(cascadeDebug.lightSpaceBoundsMin),
                glm::vec3(cascadeDebug.lightSpaceBoundsMax), cullingPadding)) {
          ++cascadeCulledCounts_[cascadeIndex];
          continue;
        }
      }

      appendShadowDraw(cascadeIndex, entry.vertexBuffer, entry.indexBuffer,
                       entry.indexBufferOffset, entry.indexFormat,
                       entry.indexCount, entry.firstIndex, entry.instanceIndex,
                       entry.vertexBufferAddress,
                       entry.vertexDecodeBufferAddress, entry.vertexDecodeIndex,
                       entry.packedVertexFormat, entry.materialIndex,
                       entry.doubleSided, entry.alphaMasked);
      currentStaticOnlyCascadeStates[cascadeIndex].casterSignature =
          hashStaticShadowCasterRasterSignature(
              currentStaticOnlyCascadeStates[cascadeIndex].casterSignature,
              entry);
      currentStaticOnlyCascadeStates[cascadeIndex].rasterSignature =
          hashStaticShadowCasterRasterSignature(
              currentStaticOnlyCascadeStates[cascadeIndex].rasterSignature,
              entry);
      cascadeIndexCountEstimates_[cascadeIndex] += entry.indexCount;
    }
  }

  for (const uint32_t templateIndex : frameDynamicTemplateIndices) {
    if (templateIndex >= meshDrawTemplates_.size()) {
      continue;
    }
    const MeshDrawTemplate &entry = meshDrawTemplates_[templateIndex];
    if (entry.renderable == nullptr || entry.submesh == nullptr) {
      continue;
    }
    const std::optional<SubmeshLod> lod =
        resolveShadowLod(*entry.submesh, settings);
    if (!lod.has_value()) {
      continue;
    }

    BufferHandle resolvedVertexBuffer = entry.baseVertexBuffer;
    uint64_t resolvedVertexBufferAddress = entry.vertexBufferAddress;
    uint64_t resolvedVertexDecodeBufferAddress =
        entry.vertexDecodeBufferAddress;
    uint32_t resolvedVertexDecodeIndex = entry.vertexDecodeIndex;
    uint32_t resolvedPackedVertexFormat = entry.packedVertexFormat;
    bool usesAnimatedOverride = false;
    if (animationSceneData != nullptr &&
        entry.instanceIndex <
            animationSceneData->geometryOverridesByRenderable.size()) {
      const AnimatedRenderableGeometryOverride &geometryOverride =
          animationSceneData
              ->geometryOverridesByRenderable[entry.instanceIndex];
      if (geometryOverride.enabled &&
          nuri::isValid(geometryOverride.vertexBuffer) &&
          animationOverrideCoversSubmesh(geometryOverride, *entry.submesh)) {
        usesAnimatedOverride = true;
        const uint64_t overrideVertexAddress = gpu_.getBufferDeviceAddress(
            geometryOverride.vertexBuffer, geometryOverride.vertexByteOffset);
        if (overrideVertexAddress != 0u) {
          resolvedVertexBuffer = geometryOverride.vertexBuffer;
          resolvedVertexBufferAddress = overrideVertexAddress;
          resolvedVertexDecodeBufferAddress = 0u;
          resolvedVertexDecodeIndex = 0u;
          resolvedPackedVertexFormat =
              static_cast<uint32_t>(PackedVertexFormat::AnimatedFloat32);
        }
      }
    }

    std::array<glm::vec3, 8> casterWorldCorners{};
    bool hasCasterCullingBounds = false;
    if (enableCascadeCasterCulling && !usesAnimatedOverride) {
      const BoundingBox worldBounds =
          entry.submesh->bounds.getTransformed(entry.renderable->modelMatrix);
      casterWorldCorners = shadow_detail::computeBoundsCorners(
          worldBounds.min_, worldBounds.max_);
      hasCasterCullingBounds = true;
    }

    for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
         ++cascadeIndex) {
      if (hasCasterCullingBounds) {
        const ShadowCascadeDebugFrameData &cascadeDebug =
            shadowDebugFrameData_.cascades[cascadeIndex];
        const float cullingPadding = std::max(
            cascadeDebug.texelWorldSize * kCullingPaddingTexelMultiplier,
            0.01f);
        if (!shadow_detail::shadowCasterOverlapsLightSpaceBounds(
                std::span<const glm::vec3, 8>(casterWorldCorners),
                cascadeDebug.lightView,
                glm::vec3(cascadeDebug.lightSpaceBoundsMin),
                glm::vec3(cascadeDebug.lightSpaceBoundsMax), cullingPadding)) {
          ++cascadeCulledCounts_[cascadeIndex];
          continue;
        }
      }

      appendShadowDraw(
          cascadeIndex, resolvedVertexBuffer, entry.indexBuffer,
          entry.indexBufferOffset, entry.indexFormat, lod->indexCount,
          lod->indexOffset, entry.instanceIndex, resolvedVertexBufferAddress,
          resolvedVertexDecodeBufferAddress, resolvedVertexDecodeIndex,
          resolvedPackedVertexFormat, entry.materialIndex, entry.doubleSided,
          entry.alphaMasked);
      ++cascadeDynamicDrawCounts_[cascadeIndex];
      cascadeIndexCountEstimates_[cascadeIndex] += lod->indexCount;
    }
  }

  passBufferDependencies_.clear();
  appendUniqueBufferDependency(passBufferDependencies_, sceneGpu.buffer);
  appendUniqueBufferDependency(passBufferDependencies_, instanceMatricesBuffer);
  appendUniqueBufferDependency(passBufferDependencies_, instanceRemapBuffer);
  appendAnimatedGeometryDependencies(passBufferDependencies_,
                                     animationSceneData);
  if (frame.sharedResources.shadowFrameGpuData.has_value()) {
    appendUniqueBufferDependency(
        passBufferDependencies_,
        frame.sharedResources.shadowFrameGpuData->buffer);
  }
  if (frame.sharedResources.materialTableGpuData.has_value()) {
    appendUniqueBufferDependency(
        passBufferDependencies_,
        frame.sharedResources.materialTableGpuData->headerBuffer);
  }
  for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
       ++cascadeIndex) {
    cascadeDrawCounts_[cascadeIndex] =
        saturateToU32(cascadeDrawItems_[cascadeIndex].size());
    const uint32_t dynamicDrawCount = cascadeDynamicDrawCounts_[cascadeIndex];
    const uint32_t staticDrawCount =
        cascadeDrawCounts_[cascadeIndex] >= dynamicDrawCount
            ? cascadeDrawCounts_[cascadeIndex] - dynamicDrawCount
            : 0u;
    currentStaticOnlyCascadeStates[cascadeIndex].staticDrawCount =
        staticDrawCount;
    currentStaticOnlyCascadeStates[cascadeIndex].dynamicDrawCount =
        dynamicDrawCount;
    ShadowCascadeDebugFrameData &cascadeDebug =
        shadowDebugFrameData_.cascades[cascadeIndex];
    cascadeDebug.drawCount = cascadeDrawCounts_[cascadeIndex];
    cascadeDebug.culledCount = cascadeCulledCounts_[cascadeIndex];
    cascadeDebug.staticDrawCount = staticDrawCount;
    cascadeDebug.dynamicDrawCount = dynamicDrawCount;
  }
  uint32_t totalDraws = 0u;
  uint32_t totalCulledDraws = 0u;
  uint32_t staticOnlyCandidateCount = 0u;
  uint32_t reusedStaticOnlyCascadeCount = 0u;
  uint32_t staticOnlyReuseMissStaticCacheRebuiltCount = 0u;
  uint32_t staticOnlyReuseMissDynamicCasterCount = 0u;
  uint32_t staticOnlyReuseMissNoPreviousCount = 0u;
  uint32_t staticOnlyReuseMissRasterStateChangedCount = 0u;
  uint32_t staticOnlyReuseMissAdaptiveRefreshCount = 0u;
  uint64_t actualTotalIndexCountEstimate = 0u;
  for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
       ++cascadeIndex) {
    ShadowCascadeDebugFrameData &cascadeDebug =
        shadowDebugFrameData_.cascades[cascadeIndex];
    StaticOnlyCascadeReuseState &currentState =
        currentStaticOnlyCascadeStates[cascadeIndex];
    const StaticOnlyCascadeReuseState &previousState =
        reusableStaticOnlyCascadeStates_[cascadeIndex];
    const bool staticOnlyCandidate =
        currentState.dynamicDrawCount == 0u &&
        cascadeHasGuardedDynamicDraw[cascadeIndex] == 0u;
    const bool previousValid = reusableStaticOnlyCascadeValid_[cascadeIndex];
    const bool lightViewProjChanged =
        previousValid && previousState.lightViewProjSignature !=
                             currentState.lightViewProjSignature;
    const bool biasChanged = previousValid && previousState.biasSignature !=
                                                  currentState.biasSignature;
    const bool casterSignatureChanged =
        previousValid &&
        previousState.casterSignature != currentState.casterSignature;
    const bool cachedRenderedFitContainsCurrent =
        previousValid &&
        staticOnlyRenderedFitContainsCurrent(
            previousState.renderedFit, currentRawShadowFits_[cascadeIndex]);
    const bool needsAdaptiveRefresh =
        cachedRenderedFitContainsCurrent &&
        staticOnlyNeedsAdaptiveRefresh(
            previousState.renderedFit, currentRawShadowFits_[cascadeIndex],
            previousState.rawFit, previousValid, settings.shadow.shadowMapSize);
    const StaticOnlyGuardBandTexels rawMotion =
        staticOnlyRawMotionTexels(currentRawShadowFits_[cascadeIndex],
                                  previousState.rawFit, previousValid);
    const bool motionFitsAdaptiveBudget =
        previousValid &&
        rawMotion.ortho <= staticOnlyAdaptiveGuardBandMaxTexels(
                               settings.shadow.shadowMapSize) +
                               kStaticOnlyPredictiveMotionEpsilonTexels &&
        rawMotion.depth <= staticOnlyAdaptiveDepthGuardBandMaxTexels(
                               settings.shadow.shadowMapSize) +
                               kStaticOnlyPredictiveMotionEpsilonTexels;
    const bool adaptiveRefresh =
        needsAdaptiveRefresh ||
        (!cachedRenderedFitContainsCurrent && staticOnlyCandidate &&
         previousValid && !biasChanged && !casterSignatureChanged &&
         motionFitsAdaptiveBudget);
    reuseStaticOnlyCascadePass_[cascadeIndex] =
        staticCacheReused && staticOnlyCandidate && previousValid &&
        !biasChanged && cachedRenderedFitContainsCurrent &&
        !needsAdaptiveRefresh;
    if (reuseStaticOnlyCascadePass_[cascadeIndex]) {
      const shadow_detail::DirectionalShadowFit rawFit =
          currentRawShadowFits_[cascadeIndex];
      writeShadowCascadeFit(previousState.renderedFit, cascadeIndex,
                            settings.shadow,
                            frame.sharedResources.shadowCompareSamplerId,
                            frame.sharedResources.shadowRawSamplerId,
                            shadowDepthTextures_[cascadeIndex], gpu_,
                            shadowFrameCpuData_, shadowDebugFrameData_);
      currentState = previousState;
      currentState.rawFit = rawFit;
      staticOnlyCascadeContentSignatures_[cascadeIndex] =
          previousState.rasterSignature;
    } else {
      staticOnlyCascadeContentSignatures_[cascadeIndex] =
          currentState.rasterSignature;
    }
    cascadeDebug.staticOnlyReuseCandidate = staticOnlyCandidate;
    cascadeDebug.staticOnlyReusePreviousValid = previousValid;
    cascadeDebug.staticOnlyReuseLightViewProjChanged = lightViewProjChanged;
    cascadeDebug.staticOnlyReuseBiasChanged = biasChanged;
    cascadeDebug.staticOnlyReuseCasterSignatureChanged = casterSignatureChanged;
    cascadeDebug.staticOnlyReuseAdaptiveRefresh = adaptiveRefresh;
    cascadeDebug.currentStaticOnlyRasterSignature =
        currentState.rasterSignature;
    cascadeDebug.previousStaticOnlyRasterSignature =
        previousState.rasterSignature;
    cascadeDebug.currentStaticOnlyLightViewProjSignature =
        currentState.lightViewProjSignature;
    cascadeDebug.previousStaticOnlyLightViewProjSignature =
        previousState.lightViewProjSignature;
    cascadeDebug.currentStaticOnlyBiasSignature = currentState.biasSignature;
    cascadeDebug.previousStaticOnlyBiasSignature = previousState.biasSignature;
    cascadeDebug.currentStaticOnlyCasterSignature =
        currentState.casterSignature;
    cascadeDebug.previousStaticOnlyCasterSignature =
        previousState.casterSignature;
    cascadeDebug.staticDrawCount = currentState.staticDrawCount;
    cascadeDebug.dynamicDrawCount = currentState.dynamicDrawCount;
    if (reuseStaticOnlyCascadePass_[cascadeIndex]) {
      cascadeDrawCounts_[cascadeIndex] = 0u;
      cascadeDebug.drawCount = 0u;
      cascadeDebug.staticOnlyReuseStatus =
          ShadowCascadeDebugFrameData::StaticOnlyReuseStatus::Reused;
    } else {
      actualTotalIndexCountEstimate +=
          cascadeIndexCountEstimates_[cascadeIndex];
      if (!staticCacheReused) {
        cascadeDebug.staticOnlyReuseStatus = ShadowCascadeDebugFrameData::
            StaticOnlyReuseStatus::StaticCacheRebuilt;
        ++staticOnlyReuseMissStaticCacheRebuiltCount;
      } else if (!staticOnlyCandidate) {
        cascadeDebug.staticOnlyReuseStatus = ShadowCascadeDebugFrameData::
            StaticOnlyReuseStatus::HasDynamicCasters;
        ++staticOnlyReuseMissDynamicCasterCount;
      } else if (!previousValid) {
        cascadeDebug.staticOnlyReuseStatus = ShadowCascadeDebugFrameData::
            StaticOnlyReuseStatus::NoPreviousStaticOnlyPass;
        ++staticOnlyReuseMissNoPreviousCount;
      } else if (adaptiveRefresh) {
        cascadeDebug.staticOnlyReuseStatus =
            ShadowCascadeDebugFrameData::StaticOnlyReuseStatus::AdaptiveRefresh;
        ++staticOnlyReuseMissAdaptiveRefreshCount;
      } else {
        cascadeDebug.staticOnlyReuseStatus = ShadowCascadeDebugFrameData::
            StaticOnlyReuseStatus::RasterStateChanged;
        ++staticOnlyReuseMissRasterStateChangedCount;
      }
    }
    staticOnlyCandidateCount += staticOnlyCandidate ? 1u : 0u;
    reusedStaticOnlyCascadeCount +=
        reuseStaticOnlyCascadePass_[cascadeIndex] ? 1u : 0u;
    totalDraws += cascadeDrawCounts_[cascadeIndex];
    totalCulledDraws += cascadeCulledCounts_[cascadeIndex];
  }
  for (uint32_t cascadeIndex = cascadeCount; cascadeIndex < kMaxShadowCascades;
       ++cascadeIndex) {
    reuseStaticOnlyCascadePass_[cascadeIndex] = false;
    staticOnlyCascadeContentSignatures_[cascadeIndex] = 0u;
    shadowDebugFrameData_.cascades[cascadeIndex].staticOnlyReuseStatus =
        ShadowCascadeDebugFrameData::StaticOnlyReuseStatus::None;
  }
  frame.metrics.shadow.cascadeCount = cascadeCount;
  frame.metrics.shadow.shadowMapSize = shadowMapSize_;
  frame.metrics.shadow.totalDraws = totalDraws;
  frame.metrics.shadow.totalCulledDraws = totalCulledDraws;
  frame.metrics.shadow.totalIndexCountEstimate =
      saturateToU32(actualTotalIndexCountEstimate);
  frame.metrics.shadow.staticCasterEntries =
      saturateToU32(staticShadowCasterCache_.size()) -
      std::min(saturateToU32(staticShadowCasterCache_.size()),
               overriddenStaticTemplateCount);
  frame.metrics.shadow.dynamicCasterEntries =
      saturateToU32(frameDynamicTemplateIndices.size());
  frame.metrics.shadow.staticCacheReused = staticCacheReused ? 1u : 0u;
  frame.metrics.shadow.staticOnlyCandidateCount = staticOnlyCandidateCount;
  frame.metrics.shadow.reusedStaticOnlyCascadeCount =
      reusedStaticOnlyCascadeCount;
  frame.metrics.shadow.staticOnlyReuseMissStaticCacheRebuiltCount =
      staticOnlyReuseMissStaticCacheRebuiltCount;
  frame.metrics.shadow.staticOnlyReuseMissDynamicCasterCount =
      staticOnlyReuseMissDynamicCasterCount;
  frame.metrics.shadow.staticOnlyReuseMissNoPreviousCount =
      staticOnlyReuseMissNoPreviousCount;
  frame.metrics.shadow.staticOnlyReuseMissRasterStateChangedCount =
      staticOnlyReuseMissRasterStateChangedCount;
  frame.metrics.shadow.staticOnlyReuseMissAdaptiveRefreshCount =
      staticOnlyReuseMissAdaptiveRefreshCount;
  frame.metrics.shadow.cascadeTextureBytes =
      static_cast<uint64_t>(cascadeCount) * shadowMapSize_ * shadowMapSize_ *
      2ull;
  frame.sharedResources.shadowDebugFrameData = shadowDebugFrameData_;
  emitShadowDiagnostics();
  return Result<bool, std::string>::makeResult(true);
}

uint64_t ShadowRenderer::shadowPipelineSignature() const noexcept {
  struct SignatureData {
    RenderPipelineHandle shadowPipeline{};
    RenderPipelineHandle shadowDoubleSidedPipeline{};
    RenderPipelineHandle shadowAlphaPipeline{};
    RenderPipelineHandle shadowAlphaDoubleSidedPipeline{};
  } signatureData{
      .shadowPipeline = shadowPipelineHandle_,
      .shadowDoubleSidedPipeline = shadowDoubleSidedPipelineHandle_,
      .shadowAlphaPipeline = shadowAlphaPipelineHandle_,
      .shadowAlphaDoubleSidedPipeline = shadowAlphaDoubleSidedPipelineHandle_,
  };
  return hashBytes(
      std::as_bytes(std::span<const SignatureData>(&signatureData, 1u)));
}

void ShadowRenderer::invalidateStaticShadowCasterCache() noexcept {
  staticShadowCasterCache_.clear();
  staticShadowCasterCacheTransformVersion_ =
      std::numeric_limits<uint64_t>::max();
  staticShadowCasterCachePipelineSignature_ = 0u;
  staticShadowCasterCacheForcedMeshLod_ = std::numeric_limits<int32_t>::min();
  staticShadowCasterCacheValid_ = false;
}

void ShadowRenderer::invalidateReusableStaticOnlyCascadeCache() noexcept {
  reusableStaticOnlyCascadeContentSignatures_.fill(0u);
  reusableStaticOnlyCascadeStates_.fill(StaticOnlyCascadeReuseState{});
  reusableStaticOnlyCascadeValid_.fill(false);
}

void ShadowRenderer::destroyShadowResources() {
  if (nuri::isValid(shadowDebugPreviewTexture_)) {
    gpu_.destroyTexture(shadowDebugPreviewTexture_);
    shadowDebugPreviewTexture_ = {};
  }
  for (TextureHandle &texture : shadowDepthTextures_) {
    if (nuri::isValid(texture)) {
      gpu_.destroyTexture(texture);
      texture = {};
    }
  }
  if (nuri::isValid(rawDepthSampler_)) {
    gpu_.destroySampler(rawDepthSampler_);
    rawDepthSampler_ = {};
  }
  if (nuri::isValid(compareDepthSampler_)) {
    gpu_.destroySampler(compareDepthSampler_);
    compareDepthSampler_ = {};
  }
  shadowMapSize_ = 0u;
  activeCascadeCount_ = 0u;
  invalidateReusableStaticOnlyCascadeCache();
  resetSdsmState();
  resetFrozenShadowFit();
  resetCascadeStabilizationHistory();
}

void ShadowRenderer::destroyBuffers() {
  for (DynamicBufferSlot &slot : instanceMatricesRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
    slot.buffer.reset();
    slot.capacityBytes = 0u;
  }
  for (DynamicBufferSlot &slot : instanceRemapRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
    slot.buffer.reset();
    slot.capacityBytes = 0u;
  }
  for (DynamicBufferSlot &slot : shadowFrameRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
    slot.buffer.reset();
    slot.capacityBytes = 0u;
  }
  for (DynamicBufferSlot &slot : sdsmReduceResultRing_) {
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
    slot.buffer.reset();
    slot.capacityBytes = 0u;
  }
  instanceMatricesRing_.clear();
  instanceRemapRing_.clear();
  shadowFrameRing_.clear();
  sdsmReduceResultRing_.clear();
  instanceDataRingUploadVersions_.clear();
  shadowFrameUploadSignatures_.clear();
  sdsmReduceResultRingPublishedFrames_.clear();
}

void ShadowRenderer::destroyShaders() {
  if (nuri::isValid(previewVertexShader_)) {
    gpu_.destroyShaderModule(previewVertexShader_);
    previewVertexShader_ = {};
  }
  if (nuri::isValid(previewFragmentShader_)) {
    gpu_.destroyShaderModule(previewFragmentShader_);
    previewFragmentShader_ = {};
  }
  if (nuri::isValid(shadowVertexShader_)) {
    gpu_.destroyShaderModule(shadowVertexShader_);
    shadowVertexShader_ = {};
  }
  if (nuri::isValid(depthFragmentShader_)) {
    gpu_.destroyShaderModule(depthFragmentShader_);
    depthFragmentShader_ = {};
  }
  if (nuri::isValid(depthAlphaFragmentShader_)) {
    gpu_.destroyShaderModule(depthAlphaFragmentShader_);
    depthAlphaFragmentShader_ = {};
  }
  if (nuri::isValid(sdsmReduceComputeShader_)) {
    gpu_.destroyShaderModule(sdsmReduceComputeShader_);
    sdsmReduceComputeShader_ = {};
  }
  if (nuri::isValid(sdsmHistogramReduceComputeShader_)) {
    gpu_.destroyShaderModule(sdsmHistogramReduceComputeShader_);
    sdsmHistogramReduceComputeShader_ = {};
  }
  shadowShader_.reset();
  depthShader_.reset();
  depthAlphaShader_.reset();
  sdsmReduceShader_.reset();
  sdsmHistogramReduceShader_.reset();
  initialized_ = false;
}

void ShadowRenderer::destroyPipelineState() {
  if (nuri::isValid(previewPipelineHandle_)) {
    gpu_.destroyRenderPipeline(previewPipelineHandle_);
    previewPipelineHandle_ = {};
  }
  if (nuri::isValid(shadowDoubleSidedPipelineHandle_)) {
    gpu_.destroyRenderPipeline(shadowDoubleSidedPipelineHandle_);
    shadowDoubleSidedPipelineHandle_ = {};
  }
  if (nuri::isValid(shadowAlphaDoubleSidedPipelineHandle_)) {
    gpu_.destroyRenderPipeline(shadowAlphaDoubleSidedPipelineHandle_);
    shadowAlphaDoubleSidedPipelineHandle_ = {};
  }
  if (nuri::isValid(shadowAlphaPipelineHandle_)) {
    gpu_.destroyRenderPipeline(shadowAlphaPipelineHandle_);
    shadowAlphaPipelineHandle_ = {};
  }
  if (nuri::isValid(shadowPipelineHandle_)) {
    gpu_.destroyRenderPipeline(shadowPipelineHandle_);
    shadowPipelineHandle_ = {};
  }
  if (nuri::isValid(sdsmReducePipelineHandle_)) {
    gpu_.destroyComputePipeline(sdsmReducePipelineHandle_);
    sdsmReducePipelineHandle_ = {};
  }
  if (nuri::isValid(sdsmHistogramReducePipelineHandle_)) {
    gpu_.destroyComputePipeline(sdsmHistogramReducePipelineHandle_);
    sdsmHistogramReducePipelineHandle_ = {};
  }
  initialized_ = false;
}

void ShadowRenderer::resetCachedState() {
  cachedScene_ = nullptr;
  cachedTopologyVersion_ = std::numeric_limits<uint64_t>::max();
  cachedMaterialVersion_ = std::numeric_limits<uint64_t>::max();
  cachedTransformVersion_ = std::numeric_limits<uint64_t>::max();
  cachedGeometryMutationVersion_ = std::numeric_limits<uint64_t>::max();
  meshDrawTemplates_.clear();
  staticShadowTemplateIndices_.clear();
  dynamicShadowTemplateIndices_.clear();
  invalidateStaticShadowCasterCache();
  invalidateReusableStaticOnlyCascadeCache();
  resetCascadeStabilizationHistory();
  instanceMatrices_.clear();
  instanceRemap_.clear();
  passTextureDependencies_.clear();
}

void ShadowRenderer::resetFrameBuildState() {
  hasPreparedShadowDepthPasses_ = false;
  hasPreparedShadowPreviewPass_ = false;
  hasActiveShadowLightForFrame_ = false;
  cascadeDrawCounts_.fill(0u);
  cascadeCulledCounts_.fill(0u);
  cascadeDynamicDrawCounts_.fill(0u);
  cascadeIndexCountEstimates_.fill(0u);
  staticOnlyCascadeContentSignatures_.fill(0u);
  reuseStaticOnlyCascadePass_.fill(false);
  currentRawShadowFits_ = {};
  for (uint32_t cascadeIndex = 0u; cascadeIndex < kMaxShadowCascades;
       ++cascadeIndex) {
    cascadePushConstants_[cascadeIndex].clear();
    cascadeDrawItems_[cascadeIndex].clear();
  }
  passBufferDependencies_.clear();
  previewTextureDependencies_.clear();
  previewPushConstants_ = {};
  previewDraw_ = {};
}

void ShadowRenderer::resetFrozenShadowFit() {
  hasFrozenShadowFit_ = false;
  frozenShadowLightId_ = kInvalidLightId;
  frozenShadowMapSize_ = 0u;
  frozenCascadeCount_ = 0u;
  frozenShadowFits_ = {};
}

void ShadowRenderer::resetCascadeStabilizationHistory() {
  cascadeStabilizationHistory_ = {};
}

void ShadowRenderer::resetSdsmState() { sdsmState_ = {}; }

Result<bool, std::string>
ShadowRenderer::publishFrameData(RenderFrameContext &frame) {
  frame.sharedResources.shadowCascadeTextures = {};
  frame.sharedResources.shadowCascadeGraphTextures = {};
  frame.sharedResources.shadowDebugPreviewTexture = {};
  frame.sharedResources.shadowDebugPreviewGraphTexture = {};
  frame.sharedResources.shadowFrameGpuData.reset();
  frame.sharedResources.shadowSdsmGpuReduceTarget.reset();
  frame.sharedResources.shadowSdsmGpuReducePipeline = {};
  frame.sharedResources.shadowRawSamplerId = kInvalidShadowBindlessIndex;
  frame.sharedResources.shadowCompareSamplerId = kInvalidShadowBindlessIndex;
  frame.sharedResources.shadowDebugFrameData.reset();
  frame.metrics.shadow = {};

  RenderSettings::ShadowSettings settings =
      renderSettingsOrDefault(frame).shadow;
  sanitizeShadowSettings(settings);
  if (settings.enabled) {
    frame.metrics.shadow.filterSampleBudget = shadowFilterSampleBudget(
        settings.filterMode, settings.pcfSampleCount,
        settings.pcssBlockerSampleCount, settings.pcssFilterSampleCount);
    if (settings.filterMode == ShadowFilterMode::PCSS) {
      frame.metrics.shadow.pcssBlockerSampleBudget =
          settings.pcssBlockerSampleCount;
      frame.metrics.shadow.pcssFilterSampleBudget =
          settings.pcssFilterSampleCount;
      frame.metrics.shadow.pcssMaxSamplesPerReceiver =
          settings.pcssBlockerSampleCount + settings.pcssFilterSampleCount;
      frame.metrics.shadow.pcssMaxSamplesPerBlendedReceiver =
          settings.cascadeCount > 1u && settings.cascadeBlendFraction > 0.0f
              ? frame.metrics.shadow.pcssMaxSamplesPerReceiver * 2u
              : frame.metrics.shadow.pcssMaxSamplesPerReceiver;
    }
    const GpuTimingReport timingReport =
        gpu_.getLatestCompletedGpuTimingReport();
    if (hasGpuTimingScope(timingReport, GpuTimingScope::Shadow)) {
      frame.metrics.shadow.gpuTimeMs = timingReport.shadowTimeMs;
      frame.metrics.shadow.gpuTimingSourceFrameIndex =
          timingReport.sourceFrameIndex;
      frame.metrics.shadow.gpuTimingAvailable = 1u;
    }
    if (hasGpuTimingScope(timingReport, GpuTimingScope::ShadowDepth)) {
      frame.metrics.shadow.depthGpuTimeMs = timingReport.shadowDepthTimeMs;
      frame.metrics.shadow.depthGpuTimingSourceFrameIndex =
          timingReport.shadowDepthSourceFrameIndex;
      frame.metrics.shadow.depthGpuTimingAvailable = 1u;
    }
    if (hasGpuTimingScope(timingReport, GpuTimingScope::ShadowSdsm)) {
      frame.metrics.shadow.sdsmGpuTimeMs = timingReport.shadowSdsmTimeMs;
      frame.metrics.shadow.sdsmGpuTimingSourceFrameIndex =
          timingReport.shadowSdsmSourceFrameIndex;
      frame.metrics.shadow.sdsmGpuTimingAvailable = 1u;
    }
  }

  auto ensureTextureResult = ensureShadowResources(settings);
  if (ensureTextureResult.hasError()) {
    return ensureTextureResult;
  }
  auto ringCountResult =
      ensureRingBufferCount(std::max(2u, gpu_.getSwapchainImageCount() + 1u));
  if (ringCountResult.hasError()) {
    return ringCountResult;
  }
  auto shadowFrameResult =
      ensureShadowFrameRingCapacity(sizeof(ShadowFrameGpuData));
  if (shadowFrameResult.hasError()) {
    return shadowFrameResult;
  }

  const uint32_t rawSamplerId = gpu_.getSamplerBindlessIndex(rawDepthSampler_);
  const uint32_t compareSamplerId =
      gpu_.getSamplerBindlessIndex(compareDepthSampler_);
  const uint32_t frameSlot =
      static_cast<uint32_t>(frame.frameIndex % shadowFrameRing_.size());
  const BufferHandle shadowFrameBuffer =
      shadowFrameRing_[frameSlot].buffer->handle();
  const uint64_t shadowFrameAddress =
      gpu_.getBufferDeviceAddress(shadowFrameBuffer);
  if (!nuri::isValid(shadowFrameBuffer) || shadowFrameAddress == 0u) {
    return Result<bool, std::string>::makeError(
        "ShadowRenderer::publishFrameData: invalid shadow frame buffer");
  }

  for (uint32_t cascadeIndex = 0u; cascadeIndex < activeCascadeCount_;
       ++cascadeIndex) {
    frame.sharedResources.shadowCascadeTextures[cascadeIndex] =
        shadowDepthTextures_[cascadeIndex];
  }
  frame.sharedResources.shadowDebugPreviewTexture = shadowDebugPreviewTexture_;
  frame.sharedResources.shadowRawSamplerId = rawSamplerId;
  frame.sharedResources.shadowCompareSamplerId = compareSamplerId;
  frame.sharedResources.shadowFrameGpuData = ShadowFrameGpuDataHandle{
      .buffer = shadowFrameBuffer,
      .bufferAddress = shadowFrameAddress,
  };

  const ShadowSdsmMode publishSdsmMode =
      sanitizeShadowSdsmMode(settings.sdsmMode);
  if (settings.enabled &&
      (publishSdsmMode == ShadowSdsmMode::PreviousFrameMinMax ||
       publishSdsmMode == ShadowSdsmMode::Histogram) &&
      sanitizeShadowSdsmReductionBackend(settings.sdsmReductionBackend) !=
          ShadowSdsmReductionBackend::Cpu) {
    auto sdsmResourceResult = publishSdsmMode == ShadowSdsmMode::Histogram
                                  ? ensureSdsmHistogramReduceResources()
                                  : ensureSdsmReduceResources();
    if (!sdsmResourceResult.hasError()) {
      auto sdsmRingCountResult = ensureSdsmReduceResultRingCount(std::max(
          kMinSdsmReduceResultRingCount, gpu_.getSwapchainImageCount() + 1u));
      if (!sdsmRingCountResult.hasError()) {
        const size_t resultBytes = publishSdsmMode == ShadowSdsmMode::Histogram
                                       ? sizeof(SdsmGpuHistogramResult)
                                       : sizeof(SdsmGpuMinMaxResult);
        auto sdsmRingResult = ensureSdsmReduceResultRingCapacity(resultBytes);
        if (!sdsmRingResult.hasError() && !sdsmReduceResultRing_.empty()) {
          const uint32_t sdsmFrameSlot = static_cast<uint32_t>(
              frame.frameIndex % sdsmReduceResultRing_.size());
          std::array<std::byte, sizeof(SdsmGpuHistogramResult)> clearedResult{};
          auto clearResult = gpu_.updateBuffer(
              sdsmReduceResultRing_[sdsmFrameSlot].buffer->handle(),
              std::span<const std::byte>(clearedResult.data(), resultBytes),
              0u);
          if (!clearResult.hasError()) {
            if (sdsmFrameSlot < sdsmReduceResultRingPublishedFrames_.size()) {
              sdsmReduceResultRingPublishedFrames_[sdsmFrameSlot] =
                  frame.frameIndex;
            }
            const BufferHandle sdsmBuffer =
                sdsmReduceResultRing_[sdsmFrameSlot].buffer->handle();
            const uint64_t sdsmBufferAddress =
                gpu_.getBufferDeviceAddress(sdsmBuffer);
            if (nuri::isValid(sdsmBuffer) && sdsmBufferAddress != 0u) {
              frame.sharedResources.shadowSdsmGpuReduceTarget =
                  ShadowSdsmGpuReduceTargetHandle{
                      .buffer = sdsmBuffer,
                      .bufferAddress = sdsmBufferAddress,
                  };
              frame.sharedResources.shadowSdsmGpuReducePipeline =
                  publishSdsmMode == ShadowSdsmMode::Histogram
                      ? sdsmHistogramReducePipelineHandle_
                      : sdsmReducePipelineHandle_;
            }
          }
        }
      }
    }
    if (!frame.sharedResources.shadowSdsmGpuReduceTarget.has_value() &&
        sanitizeShadowSdsmReductionBackend(settings.sdsmReductionBackend) ==
            ShadowSdsmReductionBackend::Gpu &&
        !sdsmState_.loggedGpuReductionUnavailableWarning_) {
      NURI_LOG_WARNING(
          "ShadowRenderer::publishFrameData: requested GPU SDSM reduction is "
          "unavailable; using CPU fallback");
      sdsmState_.loggedGpuReductionUnavailableWarning_ = true;
    }
  }

  ShadowDebugFrameData debug{};
  debug.cascadeCount = activeCascadeCount_;
  debug.rawSamplerId = rawSamplerId;
  debug.compareSamplerId = compareSamplerId;
  for (uint32_t cascadeIndex = 0u; cascadeIndex < activeCascadeCount_;
       ++cascadeIndex) {
    debug.cascades[cascadeIndex].texture = shadowDepthTextures_[cascadeIndex];
    debug.cascades[cascadeIndex].textureBindlessId =
        gpu_.getTextureBindlessIndex(shadowDepthTextures_[cascadeIndex]);
  }
  frame.sharedResources.shadowDebugFrameData = debug;

  if (!settings.enabled) {
    diagnosticLogState_ = {};
    invalidateReusableStaticOnlyCascadeCache();
    resetFrozenShadowFit();
    resetCascadeStabilizationHistory();
    resetSdsmState();
    shadowFrameCpuData_ = {};
    for (uint32_t cascadeIndex = 0u; cascadeIndex < activeCascadeCount_;
         ++cascadeIndex) {
      shadowFrameCpuData_.cascades[cascadeIndex].textureSampler = glm::uvec4(
          gpu_.getTextureBindlessIndex(shadowDepthTextures_[cascadeIndex]),
          compareSamplerId, rawSamplerId, 0u);
    }
    const std::span<const std::byte> shadowFrameBytes = std::as_bytes(
        std::span<const ShadowFrameGpuData>(&shadowFrameCpuData_, 1u));
    const uint64_t shadowFrameSignature = hashBytes(shadowFrameBytes);
    if (shadowFrameUploadSignatures_[frameSlot] != shadowFrameSignature) {
      auto updateResult =
          gpu_.updateBuffer(shadowFrameBuffer, shadowFrameBytes, 0u);
      if (updateResult.hasError()) {
        return updateResult;
      }
      shadowFrameUploadSignatures_[frameSlot] = shadowFrameSignature;
    }
    resetFrameBuildState();
  }

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
ShadowRenderer::prepareShadowGraphPasses(RenderFrameContext &frame) {
  resetFrameBuildState();

  RenderSettings::ShadowSettings settings =
      renderSettingsOrDefault(frame).shadow;
  sanitizeShadowSettings(settings);
  if (!settings.enabled) {
    diagnosticLogState_ = {};
    invalidateReusableStaticOnlyCascadeCache();
    resetCascadeStabilizationHistory();
    return Result<bool, std::string>::makeResult(true);
  }

  auto ensureTextureResult = ensureShadowResources(settings);
  if (ensureTextureResult.hasError()) {
    return ensureTextureResult;
  }
  auto ringCountResult =
      ensureRingBufferCount(std::max(2u, gpu_.getSwapchainImageCount() + 1u));
  if (ringCountResult.hasError()) {
    return ringCountResult;
  }
  auto shadowFrameResult =
      ensureShadowFrameRingCapacity(sizeof(ShadowFrameGpuData));
  if (shadowFrameResult.hasError()) {
    return shadowFrameResult;
  }

  if (frame.scene != nullptr && frame.resources != nullptr) {
    const MaterialTableSnapshot materialSnapshot =
        frame.resources->materialSnapshot();
    const bool topologyDirty =
        cachedScene_ != frame.scene ||
        cachedTopologyVersion_ != frame.scene->topologyVersion() ||
        cachedMaterialVersion_ != materialSnapshot.version ||
        cachedGeometryMutationVersion_ != frame.scene->deformationVersion();
    if (topologyDirty) {
      auto cacheResult = rebuildSceneCache(
          *frame.scene, *frame.resources,
          static_cast<uint32_t>(materialSnapshot.headers.size()));
      if (cacheResult.hasError()) {
        return cacheResult;
      }
      cachedScene_ = frame.scene;
      cachedTopologyVersion_ = frame.scene->topologyVersion();
      cachedMaterialVersion_ = materialSnapshot.version;
      cachedGeometryMutationVersion_ = frame.scene->deformationVersion();
    }
  } else {
    meshDrawTemplates_.clear();
    passTextureDependencies_.clear();
  }

  auto shadowFrameDataResult =
      updateShadowFrameData(frame, settings, settings.shadowMapSize);
  if (shadowFrameDataResult.hasError()) {
    return shadowFrameDataResult;
  }

  const uint32_t frameSlot =
      static_cast<uint32_t>(frame.frameIndex % shadowFrameRing_.size());
  if (frame.sharedResources.forwardSceneGpuData.has_value()) {
    auto drawResult = buildShadowDraws(
        frame, frameSlot, *frame.sharedResources.forwardSceneGpuData);
    if (drawResult.hasError()) {
      return drawResult;
    }
  }

  const std::span<const std::byte> shadowFrameBytes = std::as_bytes(
      std::span<const ShadowFrameGpuData>(&shadowFrameCpuData_, 1u));
  const uint64_t shadowFrameSignature = hashBytes(shadowFrameBytes);
  if (shadowFrameUploadSignatures_[frameSlot] != shadowFrameSignature) {
    auto updateResult = gpu_.updateBuffer(
        shadowFrameRing_[frameSlot].buffer->handle(), shadowFrameBytes, 0u);
    if (updateResult.hasError()) {
      return updateResult;
    }
    shadowFrameUploadSignatures_[frameSlot] = shadowFrameSignature;
  }

  hasPreparedShadowDepthPasses_ = hasActiveShadowLightForFrame_ &&
                                  activeCascadeCount_ > 0u &&
                                  nuri::isValid(shadowDepthTextures_[0]);
  if (hasActiveShadowLightForFrame_ && activeCascadeCount_ > 0u &&
      settings.debug.showShadowMapViewport &&
      nuri::isValid(shadowDebugPreviewTexture_)) {
    const ShadowPreviewMode previewMode =
        sanitizeShadowPreviewMode(settings.debug.previewMode);
    if (!nuri::isValid(previewVertexShader_) ||
        !nuri::isValid(previewFragmentShader_)) {
      auto previewShaderResult = createPreviewShaders();
      if (previewShaderResult.hasError()) {
        return previewShaderResult;
      }
    }
    if (!nuri::isValid(previewPipelineHandle_)) {
      auto previewPipelineResult = createPreviewPipeline();
      if (previewPipelineResult.hasError()) {
        return previewPipelineResult;
      }
    }
    const float previewDepthRange = std::max(settings.debug.previewDepthMax -
                                                 settings.debug.previewDepthMin,
                                             1.0e-6f);
    uint32_t previewFlags = 0u;
    if (settings.debug.previewDepthInvert) {
      previewFlags |= kShadowPreviewFlagInvert;
    }
    if (settings.debug.previewDepthLog) {
      previewFlags |= kShadowPreviewFlagLog;
    }
    if (previewMode == ShadowPreviewMode::TiledAllCascades) {
      previewFlags |= kShadowPreviewFlagTiled;
    }
    glm::uvec4 previewSourceTexIds{
        kInvalidShadowBindlessIndex, kInvalidShadowBindlessIndex,
        kInvalidShadowBindlessIndex, kInvalidShadowBindlessIndex};
    const uint32_t previewSourceCount =
        previewMode == ShadowPreviewMode::TiledAllCascades ? activeCascadeCount_
                                                           : 1u;
    if (previewMode == ShadowPreviewMode::TiledAllCascades) {
      for (uint32_t cascadeIndex = 0u; cascadeIndex < activeCascadeCount_;
           ++cascadeIndex) {
        const uint32_t sourceTexId =
            gpu_.getTextureBindlessIndex(shadowDepthTextures_[cascadeIndex]);
        if (sourceTexId == kInvalidShadowBindlessIndex) {
          return Result<bool, std::string>::makeError(
              "ShadowRenderer::prepareShadowGraphPasses: invalid shadow depth "
              "bindless index");
        }
        previewSourceTexIds[cascadeIndex] = sourceTexId;
      }
    } else {
      const uint32_t previewCascadeIndex =
          std::min(settings.debug.debugCascadeIndex, activeCascadeCount_ - 1u);
      const uint32_t sourceTexId = gpu_.getTextureBindlessIndex(
          shadowDepthTextures_[previewCascadeIndex]);
      if (sourceTexId == kInvalidShadowBindlessIndex) {
        return Result<bool, std::string>::makeError(
            "ShadowRenderer::prepareShadowGraphPasses: invalid shadow depth "
            "bindless index");
      }
      previewSourceTexIds.x = sourceTexId;
    }
    const TextureDimensions previewDimensions =
        gpu_.getTextureDimensions(shadowDebugPreviewTexture_);
    previewPushConstants_ = PreviewPushConstants{
        .sourceTexIds = previewSourceTexIds,
        .previewParams =
            glm::uvec4(previewDimensions.width, previewDimensions.height,
                       previewSourceCount, previewFlags),
        .depthParams = glm::vec4(
            1.0f / previewDepthRange,
            -settings.debug.previewDepthMin / previewDepthRange, 0.0f, 0.0f)};
    previewDraw_ = DrawItem{};
    previewDraw_.pipeline = previewPipelineHandle_;
    previewDraw_.vertexCount = 3u;
    previewDraw_.instanceCount = 1u;
    previewDraw_.pushConstants = std::span<const std::byte>(
        reinterpret_cast<const std::byte *>(&previewPushConstants_),
        sizeof(PreviewPushConstants));
    previewDraw_.debugLabel = kShadowPreviewDrawLabel;
    previewDraw_.debugColor = kShadowPreviewPassDebugColor;
    previewTextureDependencies_.clear();
    if (previewMode == ShadowPreviewMode::TiledAllCascades) {
      for (uint32_t cascadeIndex = 0u; cascadeIndex < activeCascadeCount_;
           ++cascadeIndex) {
        appendUniqueTextureDependency(previewTextureDependencies_,
                                      shadowDepthTextures_[cascadeIndex]);
      }
    } else {
      const uint32_t previewCascadeIndex =
          std::min(settings.debug.debugCascadeIndex, activeCascadeCount_ - 1u);
      appendUniqueTextureDependency(previewTextureDependencies_,
                                    shadowDepthTextures_[previewCascadeIndex]);
    }
    hasPreparedShadowPreviewPass_ = true;
  }
  return Result<bool, std::string>::makeResult(true);
}

bool ShadowRenderer::hasPreparedShadowDepthPasses() const noexcept {
  return hasPreparedShadowDepthPasses_;
}

Result<bool, std::string>
ShadowRenderer::appendShadowDepthPasses(RenderFrameContext &frame,
                                        RenderGraphBuilder &graph) {
  if (!hasPreparedShadowDepthPasses_ || activeCascadeCount_ == 0u ||
      !nuri::isValid(shadowDepthTextures_[0])) {
    return Result<bool, std::string>::makeResult(true);
  }
  const AnimationSceneFrameData *animationSceneData =
      resolveAnimationSceneFrameData(frame);

  std::pmr::vector<BufferHandle> dependencyBuffers(memory_);
  std::pmr::vector<RenderGraphAccessMode> dependencyBufferAccessModes(memory_);
  std::pmr::vector<TextureHandle> dependencyTextures(memory_);
  std::pmr::vector<RenderGraphAccessMode> dependencyTextureAccessModes(memory_);
  splitDependencies(
      std::span<const BufferDependency>(passBufferDependencies_.data(),
                                        passBufferDependencies_.size()),
      dependencyBuffers, dependencyBufferAccessModes);
  splitDependencies(
      std::span<const TextureDependency>(passTextureDependencies_.data(),
                                         passTextureDependencies_.size()),
      dependencyTextures, dependencyTextureAccessModes);

  for (uint32_t cascadeIndex = 0u; cascadeIndex < activeCascadeCount_;
       ++cascadeIndex) {
    const TextureHandle shadowDepthTexture = shadowDepthTextures_[cascadeIndex];
    if (!nuri::isValid(shadowDepthTexture)) {
      reusableStaticOnlyCascadeValid_[cascadeIndex] = false;
      reusableStaticOnlyCascadeContentSignatures_[cascadeIndex] = 0u;
      reusableStaticOnlyCascadeStates_[cascadeIndex] =
          StaticOnlyCascadeReuseState{};
      continue;
    }
    const bool reuseStaticOnlyCascade =
        reuseStaticOnlyCascadePass_[cascadeIndex];
    const TextureDimensions shadowDepthDimensions =
        gpu_.getTextureDimensions(shadowDepthTexture);
    const float shadowViewportWidth =
        static_cast<float>(std::max(shadowDepthDimensions.width, 1u));
    const float shadowViewportHeight =
        static_cast<float>(std::max(shadowDepthDimensions.height, 1u));

    const std::string importName =
        "shadow_depth_map_cascade" + std::to_string(cascadeIndex);
    auto depthImportResult =
        graph.importTexture(shadowDepthTexture, importName);
    if (depthImportResult.hasError()) {
      return Result<bool, std::string>::makeError(depthImportResult.error());
    }
    frame.sharedResources.shadowCascadeGraphTextures[cascadeIndex] =
        depthImportResult.value();

    const std::string passLabel =
        "ShadowDepthPass.Cascade" + std::to_string(cascadeIndex);
    const std::span<const ComputeDispatchItem> preDispatches =
        reuseStaticOnlyCascade
            ? std::span<const ComputeDispatchItem>()
            : shadowCascadePreDispatches(animationSceneData, cascadeIndex);
    const std::span<const BufferHandle> passDependencyBuffers =
        reuseStaticOnlyCascade
            ? std::span<const BufferHandle>()
            : std::span<const BufferHandle>(dependencyBuffers.data(),
                                            dependencyBuffers.size());
    const std::span<const RenderGraphAccessMode>
        passDependencyBufferAccessModes =
            reuseStaticOnlyCascade ? std::span<const RenderGraphAccessMode>()
                                   : std::span<const RenderGraphAccessMode>(
                                         dependencyBufferAccessModes.data(),
                                         dependencyBufferAccessModes.size());
    const std::span<const TextureHandle> passDependencyTextures =
        reuseStaticOnlyCascade
            ? std::span<const TextureHandle>()
            : std::span<const TextureHandle>(dependencyTextures.data(),
                                             dependencyTextures.size());
    const std::span<const RenderGraphAccessMode>
        passDependencyTextureAccessModes =
            reuseStaticOnlyCascade ? std::span<const RenderGraphAccessMode>()
                                   : std::span<const RenderGraphAccessMode>(
                                         dependencyTextureAccessModes.data(),
                                         dependencyTextureAccessModes.size());
    const std::span<const DrawItem> passDraws =
        reuseStaticOnlyCascade
            ? std::span<const DrawItem>()
            : std::span<const DrawItem>(cascadeDrawItems_[cascadeIndex].data(),
                                        cascadeDrawItems_[cascadeIndex].size());
    const RenderGraphGraphicsPassDesc desc{
        .color = {},
        .colorTexture = {},
        .hasColorAttachment = false,
        .depth = {.loadOp =
                      reuseStaticOnlyCascade ? LoadOp::Load : LoadOp::Clear,
                  .storeOp = StoreOp::Store,
                  .clearDepth = 1.0f,
                  .clearStencil = 0u},
        .depthTexture = depthImportResult.value(),
        .useViewport = true,
        .viewport = {.x = 0.0f,
                     .y = 0.0f,
                     .width = shadowViewportWidth,
                     .height = shadowViewportHeight,
                     .minDepth = 0.0f,
                     .maxDepth = 1.0f},
        .preDispatches = preDispatches,
        .dependencyBuffers = passDependencyBuffers,
        .dependencyBufferAccessModes = passDependencyBufferAccessModes,
        .dependencyTextures = passDependencyTextures,
        .dependencyTextureAccessModes = passDependencyTextureAccessModes,
        .draws = passDraws,
        .drawBuffersPreResolved = false,
        .preResolvedDrawBuffers = {},
        .gpuTimingScope = GpuTimingScope::ShadowDepth,
        .debugLabel = passLabel,
        .debugColor = kShadowPassDebugColor,
        .markColorAsFrameOutput = false,
        .markImplicitOutputSideEffect = false,
        .borrowPayload = false,
    };

    auto passResult = graph.addGraphicsPass(desc);
    if (passResult.hasError()) {
      return Result<bool, std::string>::makeError(passResult.error());
    }

    auto markResult = graph.markPassSideEffect(passResult.value());
    if (markResult.hasError()) {
      return markResult;
    }

    if (cascadeDynamicDrawCounts_[cascadeIndex] == 0u) {
      reusableStaticOnlyCascadeValid_[cascadeIndex] = true;
      reusableStaticOnlyCascadeContentSignatures_[cascadeIndex] =
          staticOnlyCascadeContentSignatures_[cascadeIndex];
      reusableStaticOnlyCascadeStates_[cascadeIndex] =
          StaticOnlyCascadeReuseState{
              .renderedFit = makeDirectionalShadowFitFromDebugCascade(
                  shadowDebugFrameData_.cascades[cascadeIndex]),
              .rawFit = currentRawShadowFits_[cascadeIndex],
              .rasterSignature =
                  staticOnlyCascadeContentSignatures_[cascadeIndex],
              .lightViewProjSignature =
                  shadowDebugFrameData_.cascades[cascadeIndex]
                      .currentStaticOnlyLightViewProjSignature,
              .biasSignature = shadowDebugFrameData_.cascades[cascadeIndex]
                                   .currentStaticOnlyBiasSignature,
              .casterSignature = shadowDebugFrameData_.cascades[cascadeIndex]
                                     .currentStaticOnlyCasterSignature,
              .staticDrawCount =
                  shadowDebugFrameData_.cascades[cascadeIndex].staticDrawCount,
              .dynamicDrawCount =
                  shadowDebugFrameData_.cascades[cascadeIndex].dynamicDrawCount,
          };
    } else {
      reusableStaticOnlyCascadeValid_[cascadeIndex] = false;
      reusableStaticOnlyCascadeContentSignatures_[cascadeIndex] = 0u;
      reusableStaticOnlyCascadeStates_[cascadeIndex] =
          StaticOnlyCascadeReuseState{};
    }
  }
  for (uint32_t cascadeIndex = activeCascadeCount_;
       cascadeIndex < kMaxShadowCascades; ++cascadeIndex) {
    reusableStaticOnlyCascadeValid_[cascadeIndex] = false;
    reusableStaticOnlyCascadeContentSignatures_[cascadeIndex] = 0u;
    reusableStaticOnlyCascadeStates_[cascadeIndex] =
        StaticOnlyCascadeReuseState{};
  }

  if (hasPreparedShadowPreviewPass_ &&
      nuri::isValid(shadowDebugPreviewTexture_)) {
    const TextureDimensions previewDimensions =
        gpu_.getTextureDimensions(shadowDebugPreviewTexture_);
    const float previewViewportWidth =
        static_cast<float>(std::max(previewDimensions.width, 1u));
    const float previewViewportHeight =
        static_cast<float>(std::max(previewDimensions.height, 1u));
    auto previewImportResult =
        graph.importTexture(shadowDebugPreviewTexture_, "shadow_depth_preview");
    if (previewImportResult.hasError()) {
      return Result<bool, std::string>::makeError(previewImportResult.error());
    }
    frame.sharedResources.shadowDebugPreviewGraphTexture =
        previewImportResult.value();

    std::pmr::vector<TextureHandle> previewDependencyTextures(memory_);
    std::pmr::vector<RenderGraphAccessMode> previewDependencyTextureAccessModes(
        memory_);
    splitDependencies(
        std::span<const TextureDependency>(previewTextureDependencies_.data(),
                                           previewTextureDependencies_.size()),
        previewDependencyTextures, previewDependencyTextureAccessModes);

    const RenderGraphGraphicsPassDesc previewDesc{
        .color = {.loadOp = LoadOp::Clear,
                  .storeOp = StoreOp::Store,
                  .clearColor = {1.0f, 1.0f, 1.0f, 1.0f}},
        .colorTexture = previewImportResult.value(),
        .hasColorAttachment = true,
        .depth = {},
        .depthTexture = {},
        .useViewport = true,
        .viewport = {.x = 0.0f,
                     .y = 0.0f,
                     .width = previewViewportWidth,
                     .height = previewViewportHeight,
                     .minDepth = 0.0f,
                     .maxDepth = 1.0f},
        .preDispatches = {},
        .dependencyBuffers = {},
        .dependencyBufferAccessModes = {},
        .dependencyTextures = std::span<const TextureHandle>(
            previewDependencyTextures.data(), previewDependencyTextures.size()),
        .dependencyTextureAccessModes = std::span<const RenderGraphAccessMode>(
            previewDependencyTextureAccessModes.data(),
            previewDependencyTextureAccessModes.size()),
        .draws = std::span<const DrawItem>(&previewDraw_, 1u),
        .drawBuffersPreResolved = false,
        .preResolvedDrawBuffers = {},
        .debugLabel = kShadowPreviewPassLabel,
        .debugColor = kShadowPreviewPassDebugColor,
        .markColorAsFrameOutput = false,
        .markImplicitOutputSideEffect = false,
        .borrowPayload = false,
    };
    auto previewPassResult = graph.addGraphicsPass(previewDesc);
    if (previewPassResult.hasError()) {
      return Result<bool, std::string>::makeError(previewPassResult.error());
    }
    auto previewMarkResult =
        graph.markPassSideEffect(previewPassResult.value());
    if (previewMarkResult.hasError()) {
      return previewMarkResult;
    }
  }

  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri
