#include "nuri/pch.h"

#include "nuri/gfx/renderers/shadow_renderer.h"

#include "nuri/core/containers/hash_map.h"
#include "nuri/core/log.h"
#include "nuri/core/pmr_scratch.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/renderers/detail/renderable_material_resolution.h"
#include "nuri/gfx/renderers/detail/shadow_math.h"
#include "nuri/resources/gpu/resource_manager.h"

#include <bit>

namespace nuri {
namespace {

constexpr uint32_t kShadowPassDebugColor = 0xff5a7dffu;
constexpr uint32_t kShadowMeshDebugColor = 0xff5aff9du;
constexpr uint32_t kShadowPreviewPassDebugColor = 0xff7d5affu;
constexpr std::string_view kShadowPassLabel = "ShadowDepthPass";
constexpr std::string_view kShadowMeshLabel = "ShadowCasterMesh";
constexpr std::string_view kShadowPreviewPassLabel = "ShadowDepthPreviewPass";
constexpr std::string_view kShadowPreviewDrawLabel = "ShadowDepthPreview";
constexpr std::string_view kShadowPassDependencyBufferLabel =
    "ShadowDepthPass.dependency_buffer";
constexpr std::string_view kShadowPassDependencyTextureLabel =
    "ShadowDepthPass.dependency_texture";
constexpr std::string_view kShadowPassDrawBufferLabel =
    "ShadowDepthPass.draw_buffer";
constexpr uint64_t kFnvOffsetBasis64 = 14695981039346656037ull;
constexpr uint64_t kFnvPrime64 = 1099511628211ull;
constexpr uint32_t kShadowPreviewFlagInvert = 1u << 0u;
constexpr uint32_t kShadowPreviewFlagLog = 1u << 1u;
constexpr uint32_t kShadowPreviewFlagTiled = 1u << 2u;
constexpr float kCullingPaddingTexelMultiplier = 2.0f;
constexpr uint32_t kStaticCasterLightGridResolution = 64u;
constexpr uint32_t kStaticCasterLightGridDepthResolution = 16u;
constexpr uint32_t kStaticCasterLightGridMaxCellsPerCaster = 128u;
constexpr size_t kStaticCasterLightGridSortedCandidateLimit = 1024u;
constexpr uint32_t kStaticBatchOverlapEmitMinInstances = 4u;
constexpr float kStaticOnlyGuardBandShadowMapFraction = 1.0f / 32.0f;
constexpr float kStaticOnlyGuardBandMaxTexels = 512.0f;
constexpr float kStaticOnlyGuardBandMinTexels = 8.0f;
constexpr float kStaticOnlyDepthGuardBandShadowMapFraction = 1.0f / 64.0f;
constexpr float kStaticOnlyDepthGuardBandMaxTexels = 256.0f;
constexpr float kStaticOnlyDepthGuardBandMinTexels = 4.0f;
constexpr float kStaticOnlyAdaptiveGuardBandMotionMultiplier = 2.25f;
constexpr float kStaticOnlyAdaptiveGuardBandMaxShadowMapFraction = 1.0f / 8.0f;
constexpr float kStaticOnlyAdaptiveDepthGuardBandMaxShadowMapFraction =
    1.0f / 16.0f;
constexpr float kStaticOnlyPredictiveCenterMotionMultiplier = 1.0f;
constexpr float kStaticOnlyPredictiveExtentGrowthFraction = 1.0f / 3.0f;
constexpr float kStaticOnlyPredictiveTrailingGuardFraction = 3.0f / 8.0f;
constexpr float kStaticOnlyPredictiveMotionEpsilonTexels = 0.5f;
constexpr uint32_t kSdsmHistogramSourceMaxTexelCount = 4096u;
constexpr uint64_t kSdsmDiagnosticRefreshFrames = 120u;
constexpr float kSdsmHistogramClearDepthEpsilon = 1.0e-4f;
constexpr uint32_t kMinSdsmReduceResultRingCount = 16u;
constexpr uint32_t kSdsmGpuWarmupGraceMissFrames = 2u;
constexpr uint32_t kMeshletFlagDoubleSided = 1u << 2u;
constexpr uint32_t kMeshletFlagShadowCascadeCulling = 1u << 6u;
constexpr uint32_t kMeshletFlagForcedLodShift = 8u;
constexpr uint32_t kMeshletFlagForcedLodMask = 0x3u;
constexpr uint32_t kShadowMeshletCounterFlagEnabled = 1u << 0u;

[[nodiscard]] constexpr Format
sanitizeShadowDepthFormat(Format format) noexcept {
  switch (format) {
  case Format::D16_UNORM:
  case Format::D32_FLOAT:
    return format;
  default:
    return kDefaultShadowMapDepthFormat;
  }
}

[[nodiscard]] constexpr uint64_t
shadowDepthTextureBytesPerPixel(Format format) noexcept {
  switch (sanitizeShadowDepthFormat(format)) {
  case Format::D32_FLOAT:
    return sizeof(float);
  case Format::D16_UNORM:
  default:
    return sizeof(uint16_t);
  }
}

[[nodiscard]] constexpr std::string_view
shadowCascadePassLabel(uint32_t cascadeIndex) {
  switch (cascadeIndex) {
  case 0u:
    return "ShadowDepthPass.Cascade0";
  case 1u:
    return "ShadowDepthPass.Cascade1";
  case 2u:
    return "ShadowDepthPass.Cascade2";
  case 3u:
    return "ShadowDepthPass.Cascade3";
  default:
    return kShadowPassLabel;
  }
}

[[nodiscard]] constexpr std::string_view
shadowCascadeTextureImportName(uint32_t cascadeIndex) {
  switch (cascadeIndex) {
  case 0u:
    return "shadow_depth_map_cascade0";
  case 1u:
    return "shadow_depth_map_cascade1";
  case 2u:
    return "shadow_depth_map_cascade2";
  case 3u:
    return "shadow_depth_map_cascade3";
  default:
    return "shadow_depth_map_cascade";
  }
}

[[nodiscard]] constexpr std::string_view
shadowCascadeCaptureName(uint32_t cascadeIndex) {
  switch (cascadeIndex) {
  case 0u:
    return "shadow_cascade_0";
  case 1u:
    return "shadow_cascade_1";
  case 2u:
    return "shadow_cascade_2";
  case 3u:
    return "shadow_cascade_3";
  default:
    return {};
  }
}

void logShadowVisibilityCounters(const RenderFrameContext &frame) {
  if (frame.settings == nullptr ||
      !frame.settings->visibility.debug.logCounters) {
    return;
  }

  const VisibilityFrameMetrics &visibility = frame.metrics.visibility;
  NURI_LOG_INFO(
      "ShadowRenderer::visibility counters frame=%llu "
      "meshlet(candidates=%u readback=%u source=%u stale=%u errors=%u "
      "rejectedBounds=%u) "
      "cpu(candidates=%u rejected=%u)",
      static_cast<unsigned long long>(frame.frameIndex),
      visibility.shadowMeshletCandidates,
      visibility.shadowMeshletReadbackAvailable,
      visibility.shadowMeshletReadbackSourceFrame,
      visibility.shadowMeshletReadbackStaleFrameCount,
      visibility.shadowMeshletReadbackErrorCount,
      visibility.shadowMeshletRejectedBounds, visibility.shadowCpuCandidates,
      visibility.shadowCpuRejected);
}

void publishRequestedCapture(RenderFrameContext &frame, GPUDevice &gpu,
                             std::string_view name, TextureHandle texture,
                             RenderCaptureValueKind kind,
                             RenderCaptureLifetimeClass lifetime,
                             std::string_view colorSpace,
                             std::string_view compareProfile,
                             std::string_view producerPassLabel) {
  if (!isRenderCaptureRequested(frame, name) || !nuri::isValid(texture)) {
    return;
  }
  frame.captureRegistry.publish(RenderCapturePoint{
      .name = name,
      .version = 1u,
      .texture = texture,
      .format = gpu.getTextureFormat(texture),
      .dimensions = gpu.getTextureDimensions(texture),
      .frameIndex = frame.frameIndex,
      .mip = 0u,
      .layer = 0u,
      .kind = kind,
      .lifetime = lifetime,
      .colorSpace = colorSpace,
      .defaultCompareProfile = compareProfile,
      .producerPassLabel = producerPassLabel,
      .debugLabel = name,
  });
}

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

[[nodiscard]] bool isSamePipelineHandle(RenderPipelineHandle lhs,
                                        RenderPipelineHandle rhs) {
  return lhs.index == rhs.index && lhs.generation == rhs.generation;
}

[[nodiscard]] bool isSameMeshletPipelineHandle(MeshletPipelineHandle lhs,
                                               MeshletPipelineHandle rhs) {
  return lhs.index == rhs.index && lhs.generation == rhs.generation;
}

[[nodiscard]] uint32_t maxMeshletCountForSubmesh(const Submesh &submesh) {
  uint32_t maxMeshletCount = 0u;
  const uint32_t lodCount =
      std::clamp(submesh.lodCount, 0u, Submesh::kMaxLodCount);
  for (uint32_t lodIndex = 0u; lodIndex < lodCount; ++lodIndex) {
    maxMeshletCount =
        std::max(maxMeshletCount, submesh.lods[lodIndex].meshletCount);
  }
  return maxMeshletCount;
}

[[nodiscard]] bool canUseShadowMeshlets(const Model::ModelMeshletGpuView *view,
                                        uint32_t submeshIndex,
                                        uint32_t meshletMaxCount,
                                        uint32_t selectedMeshletCount) {
  return view != nullptr && selectedMeshletCount != 0u &&
         meshletMaxCount != 0u && submeshIndex < view->lodRangeCount &&
         nuri::isValid(view->meshletBuffer) &&
         nuri::isValid(view->meshletVertexIndexBuffer) &&
         nuri::isValid(view->meshletPrimitiveIndexBuffer) &&
         nuri::isValid(view->lodRangeBuffer);
}

struct ShadowBatchKey {
  uint32_t cascadeIndex = 0u;
  bool dynamicCaster = false;
  bool useMeshlets = false;
  RenderPipelineHandle pipeline{};
  MeshletPipelineHandle meshletPipeline{};
  BufferHandle vertexBuffer{};
  BufferHandle vertexDecodeBuffer{};
  BufferHandle indexBuffer{};
  uint64_t indexBufferOffset = 0u;
  IndexFormat indexFormat = IndexFormat::U32;
  uint32_t indexCount = 0u;
  uint32_t firstIndex = 0u;
  uint64_t vertexBufferAddress = 0u;
  uint64_t vertexDecodeBufferAddress = 0u;
  uint32_t vertexDecodeIndex = 0u;
  uint32_t packedVertexFormat = 0u;
  uint32_t materialIndex = 0u;
  BufferHandle meshletBuffer{};
  BufferHandle meshletVertexIndexBuffer{};
  BufferHandle meshletPrimitiveIndexBuffer{};
  BufferHandle meshletLodRangeBuffer{};
  uint32_t submeshIndex = 0u;
  uint32_t meshletMaxCount = 0u;
  uint32_t meshletCount = 0u;
  uint32_t vertexOffset = 0u;
  bool enableMeshletCascadeCulling = false;

  bool operator==(const ShadowBatchKey &other) const noexcept {
    return cascadeIndex == other.cascadeIndex &&
           dynamicCaster == other.dynamicCaster &&
           useMeshlets == other.useMeshlets &&
           isSamePipelineHandle(pipeline, other.pipeline) &&
           isSameMeshletPipelineHandle(meshletPipeline,
                                       other.meshletPipeline) &&
           isSameBufferHandle(vertexBuffer, other.vertexBuffer) &&
           isSameBufferHandle(vertexDecodeBuffer, other.vertexDecodeBuffer) &&
           isSameBufferHandle(indexBuffer, other.indexBuffer) &&
           indexBufferOffset == other.indexBufferOffset &&
           indexFormat == other.indexFormat && indexCount == other.indexCount &&
           firstIndex == other.firstIndex &&
           vertexBufferAddress == other.vertexBufferAddress &&
           vertexDecodeBufferAddress == other.vertexDecodeBufferAddress &&
           vertexDecodeIndex == other.vertexDecodeIndex &&
           packedVertexFormat == other.packedVertexFormat &&
           materialIndex == other.materialIndex &&
           isSameBufferHandle(meshletBuffer, other.meshletBuffer) &&
           isSameBufferHandle(meshletVertexIndexBuffer,
                              other.meshletVertexIndexBuffer) &&
           isSameBufferHandle(meshletPrimitiveIndexBuffer,
                              other.meshletPrimitiveIndexBuffer) &&
           isSameBufferHandle(meshletLodRangeBuffer,
                              other.meshletLodRangeBuffer) &&
           submeshIndex == other.submeshIndex &&
           meshletMaxCount == other.meshletMaxCount &&
           meshletCount == other.meshletCount &&
           vertexOffset == other.vertexOffset &&
           enableMeshletCascadeCulling == other.enableMeshletCascadeCulling;
  }
};

struct ShadowBatchKeyHash {
  size_t operator()(const ShadowBatchKey &key) const noexcept {
    uint64_t hash = kFnvOffsetBasis64;
    hash = hashCombine64(hash, key.cascadeIndex);
    hash = hashCombine64(hash, key.dynamicCaster ? 1ull : 0ull);
    hash = hashCombine64(hash, key.useMeshlets ? 1ull : 0ull);
    hash = hashCombine64(
        hash, foldHandle(key.pipeline.index, key.pipeline.generation));
    hash = hashCombine64(hash, foldHandle(key.meshletPipeline.index,
                                          key.meshletPipeline.generation));
    hash = hashCombine64(
        hash, foldHandle(key.vertexBuffer.index, key.vertexBuffer.generation));
    hash = hashCombine64(hash, foldHandle(key.vertexDecodeBuffer.index,
                                          key.vertexDecodeBuffer.generation));
    hash = hashCombine64(
        hash, foldHandle(key.indexBuffer.index, key.indexBuffer.generation));
    hash = hashCombine64(hash, key.indexBufferOffset);
    hash = hashCombine64(hash, static_cast<uint64_t>(key.indexFormat));
    hash = hashCombine64(hash, (static_cast<uint64_t>(key.indexCount) << 32u) |
                                   key.firstIndex);
    hash = hashCombine64(hash, key.vertexBufferAddress);
    hash = hashCombine64(hash, key.vertexDecodeBufferAddress);
    hash = hashCombine64(hash,
                         (static_cast<uint64_t>(key.vertexDecodeIndex) << 32u) |
                             key.packedVertexFormat);
    hash = hashCombine64(hash, key.materialIndex);
    hash = hashCombine64(hash, foldHandle(key.meshletBuffer.index,
                                          key.meshletBuffer.generation));
    hash = hashCombine64(hash,
                         foldHandle(key.meshletVertexIndexBuffer.index,
                                    key.meshletVertexIndexBuffer.generation));
    hash = hashCombine64(
        hash, foldHandle(key.meshletPrimitiveIndexBuffer.index,
                         key.meshletPrimitiveIndexBuffer.generation));
    hash =
        hashCombine64(hash, foldHandle(key.meshletLodRangeBuffer.index,
                                       key.meshletLodRangeBuffer.generation));
    hash =
        hashCombine64(hash, (static_cast<uint64_t>(key.submeshIndex) << 32u) |
                                key.meshletMaxCount);
    hash =
        hashCombine64(hash, (static_cast<uint64_t>(key.meshletCount) << 32u) |
                                key.vertexOffset);
    hash = hashCombine64(hash, key.enableMeshletCascadeCulling ? 1ull : 0ull);
    return static_cast<size_t>(hash);
  }
};

struct ShadowBatchEntry {
  ShadowBatchKey key{};
  uint32_t firstInstance = 0u;
  uint32_t inlineInstanceIndex = 0u;
  uint32_t instanceCount = 0u;
  const uint32_t *externalInstanceIndices = nullptr;
  uint32_t externalInstanceCount = 0u;
  uint32_t instanceListOffset = 0u;
  bool hasInstanceList = false;

  void materializeInstances(std::pmr::vector<uint32_t> &instanceLists,
                            uint32_t reserveHint = 0u) {
    if (hasInstanceList) {
      return;
    }

    const uint32_t existingCount = instanceCount;
    const size_t requestedCapacity =
        instanceLists.size() +
        std::max<size_t>(reserveHint, std::max(existingCount, 2u));
    if (requestedCapacity > instanceLists.capacity()) {
      instanceLists.reserve(requestedCapacity);
    }

    NURI_ASSERT(instanceLists.size() <= std::numeric_limits<uint32_t>::max(),
                "Shadow batch instance list offset overflowed");
    instanceListOffset = static_cast<uint32_t>(instanceLists.size());
    hasInstanceList = true;
    if (externalInstanceCount != 0u) {
      instanceLists.insert(instanceLists.end(), externalInstanceIndices,
                           externalInstanceIndices + externalInstanceCount);
      externalInstanceIndices = nullptr;
      externalInstanceCount = 0u;
    } else if (existingCount != 0u) {
      instanceLists.push_back(inlineInstanceIndex);
    }
  }

  void appendInstance(uint32_t instanceIndex,
                      std::pmr::vector<uint32_t> &instanceLists,
                      uint32_t reserveHint = 0u) {
    if (instanceCount == 0u) {
      inlineInstanceIndex = instanceIndex;
      instanceCount = 1u;
      return;
    }

    materializeInstances(instanceLists,
                         std::max(reserveHint, instanceCount + 1u));
    instanceLists.push_back(instanceIndex);
    ++instanceCount;
  }

  void appendInstances(std::span<const uint32_t> indices,
                       std::pmr::vector<uint32_t> &instanceLists) {
    const uint32_t addedCount = static_cast<uint32_t>(
        std::min<size_t>(indices.size(), std::numeric_limits<uint32_t>::max()));
    if (addedCount == 0u) {
      return;
    }
    if (instanceCount == 0u) {
      externalInstanceIndices = indices.data();
      externalInstanceCount = addedCount;
      instanceCount = addedCount;
      return;
    }

    materializeInstances(instanceLists, instanceCount + addedCount);
    instanceLists.insert(instanceLists.end(), indices.begin(), indices.end());
    instanceCount += addedCount;
  }
};

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
  static_assert(std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>,
                "rawBytesEqual is only for plain layout values; use operator== "
                "or element-wise comparison for other types.");
  return std::memcmp(&lhs, &rhs, sizeof(T)) == 0;
}

[[nodiscard]] uint64_t
hashShadowSettings(uint64_t hash,
                   const RenderSettings::ShadowSettings &settings) {
  hash = hashCombineValue(hash, settings.enabled);
  hash = hashCombineValue(hash, settings.qualityPreset);
  hash = hashCombineValue(hash, settings.cascadeCount);
  hash = hashCombineValue(hash, settings.shadowMapSize);
  hash = hashCombineValue(hash, settings.depthFormat);
  hash = hashCombineValue(hash, settings.maxDistance);
  hash = hashCombineValue(hash, settings.maxDistanceFadeFraction);
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
  hash = hashCombineValue(hash, settings.enableMeshletCascadeCulling);
  hash = hashCombineValue(hash, settings.debug.freezeCascades);
  hash = hashCombineValue(hash, settings.debug.freezeLightView);
  hash = hashCombineValue(hash, settings.debug.enableCascadeCasterCulling);
  hash = hashCombineValue(hash, settings.debug.debugCascadeIndex);
  return hash;
}

[[nodiscard]] uint64_t hashCameraState(uint64_t hash,
                                       const RenderFrameContext &frame) {
  hash = hashCombineValue(hash, frame.camera.projectionType);
  hash = hashCombineValue(hash, frame.camera.nearPlane);
  hash = hashCombineValue(hash, frame.camera.farPlane);
  hash = hashCombineValue(hash, frame.camera.aspectRatio);
  hash = hashCombineBytes(hash, std::as_bytes(std::span<const glm::vec4>(
                                    &frame.camera.cameraPos, 1u)));
  hash = hashCombineBytes(
      hash, std::as_bytes(std::span<const glm::mat4>(&frame.camera.view, 1u)));
  const glm::mat4 &projection = cameraCurrentUnjitteredProjection(frame.camera);
  hash = hashCombineBytes(
      hash, std::as_bytes(std::span<const glm::mat4>(&projection, 1u)));
  return hash;
}

[[nodiscard]] uint64_t
hashShadowDebugSelection(uint64_t hash,
                         const ShadowDebugFrameData &debugFrameData) {
  hash = hashCombineValue(hash, isValid(debugFrameData.selectedShadowLightId));
  hash = hashCombineValue(hash, isValid(debugFrameData.selectedShadowLightId)
                                    ? debugFrameData.selectedShadowLightId.value
                                    : 0u);
  hash = hashCombineValue(hash, debugFrameData.cascadeCount);
  hash = hashCombineValue(hash, debugFrameData.rawSamplerId);
  hash = hashCombineValue(hash, debugFrameData.compareSamplerId);
  return hash;
}

[[nodiscard]] uint64_t hashSdsmState(uint64_t hash,
                                     const ShadowSdsmDebugFrameData &sdsm) {
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
  return hash;
}

[[nodiscard]] uint64_t
hashCascadeState(uint64_t hash, const ShadowCascadeDebugFrameData &cascade,
                 uint32_t cascadeIndex) {
  (void)cascadeIndex;
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
  hash = hashCombineValue(hash, cascade.staticBatchFullEmitCount);
  hash = hashCombineValue(hash, cascade.staticLightGridQueryCellCount);
  hash = hashCombineValue(hash, cascade.staticLightGridCandidateCount);
  hash = hashCombineValue(hash, cascade.staticLightGridFallbackScanCount);
  hash = hashCombineValue(hash, cascade.staticOnlyReuseStatus);
  hash = hashCombineValue(hash, cascade.staticOnlyReuseCandidate);
  hash = hashCombineValue(hash, cascade.staticOnlyReusePreviousValid);
  hash = hashCombineValue(hash, cascade.staticOnlyReuseLightViewProjChanged);
  hash = hashCombineValue(hash, cascade.staticOnlyReuseBiasChanged);
  hash = hashCombineValue(hash, cascade.staticOnlyReuseCasterSignatureChanged);
  hash = hashCombineValue(hash, cascade.staticOnlyReuseAdaptiveRefresh);
  hash = hashCombineValue(hash, cascade.currentStaticOnlyRasterSignature);
  hash = hashCombineValue(hash, cascade.previousStaticOnlyRasterSignature);
  hash =
      hashCombineValue(hash, cascade.currentStaticOnlyLightViewProjSignature);
  hash =
      hashCombineValue(hash, cascade.previousStaticOnlyLightViewProjSignature);
  hash = hashCombineValue(hash, cascade.currentStaticOnlyBiasSignature);
  hash = hashCombineValue(hash, cascade.previousStaticOnlyBiasSignature);
  hash = hashCombineValue(hash, cascade.currentStaticOnlyCasterSignature);
  hash = hashCombineValue(hash, cascade.previousStaticOnlyCasterSignature);
  return hash;
}

[[nodiscard]] uint64_t hashShadowMetrics(uint64_t hash,
                                         const ShadowFrameMetrics &metrics) {
  hash = hashCombineValue(hash, metrics.cascadeCount);
  hash = hashCombineValue(hash, metrics.shadowMapSize);
  hash = hashCombineValue(hash, metrics.totalDraws);
  hash = hashCombineValue(hash, metrics.totalCulledDraws);
  hash = hashCombineValue(hash, metrics.totalIndexCountEstimate);
  hash = hashCombineValue(hash, metrics.staticCasterEntries);
  hash = hashCombineValue(hash, metrics.dynamicCasterEntries);
  hash = hashCombineValue(hash, metrics.staticCacheReused);
  hash = hashCombineValue(hash, metrics.staticBatchTemplateCount);
  hash = hashCombineValue(hash, metrics.shadowBatchEntryCount);
  hash = hashCombineValue(hash, metrics.shadowMeshletDispatchCount);
  hash = hashCombineValue(hash, metrics.shadowMeshletTaskGroupCount);
  hash = hashCombineValue(hash, metrics.shadowInstanceRemapCount);
  hash = hashCombineValue(hash, metrics.staticBatchFullEmitCount);
  hash = hashCombineValue(hash, metrics.staticLightGridQueryCount);
  hash = hashCombineValue(hash, metrics.staticLightGridFallbackScanCount);
  hash = hashCombineValue(hash, metrics.staticLightGridQueryCellCount);
  hash = hashCombineValue(hash, metrics.staticLightGridCandidateCount);
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
  hash = hashCombineValue(hash, metrics.minCascadeTexelWorldSize);
  hash = hashCombineValue(hash, metrics.averageCascadeTexelWorldSize);
  hash = hashCombineValue(hash, metrics.maxCascadeTexelWorldSize);
  hash = hashCombineValue(hash, metrics.farCascadeTexelWorldSize);
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

[[nodiscard]] uint64_t buildShadowDiagnosticSignature(
    const RenderFrameContext &frame,
    const RenderSettings::ShadowSettings &settings,
    const ShadowDebugFrameData &debugFrameData,
    const ShadowFrameMetrics &metrics, bool hasActiveShadowLightForFrame,
    bool hasPreparedShadowDepthPasses, bool hasPreparedShadowPreviewPass) {
  uint64_t hash = kFnvOffsetBasis64;
  hash = hashShadowSettings(hash, settings);
  hash = hashCameraState(hash, frame);
  hash = hashCombineValue(hash, hasActiveShadowLightForFrame);
  hash = hashCombineValue(hash, hasPreparedShadowDepthPasses);
  hash = hashCombineValue(hash, hasPreparedShadowPreviewPass);
  hash = hashShadowDebugSelection(hash, debugFrameData);
  hash = hashSdsmState(hash, debugFrameData.sdsm);

  const uint32_t cascadeCount =
      std::min(debugFrameData.cascadeCount, kMaxShadowCascades);
  for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
       ++cascadeIndex) {
    hash = hashCascadeState(hash, debugFrameData.cascades[cascadeIndex],
                            cascadeIndex);
  }

  return hashShadowMetrics(hash, metrics);
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

void appendUniqueBufferHandle(std::pmr::vector<BufferHandle> &buffers,
                              BufferHandle handle) {
  if (!nuri::isValid(handle)) {
    return;
  }
  for (const BufferHandle existing : buffers) {
    if (isSameBufferHandle(existing, handle)) {
      return;
    }
  }
  buffers.push_back(handle);
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
    std::span<const float> minMaxPairs, std::pmr::memory_resource *memory) {
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
  std::pmr::vector<CachedSdsmHistogramTile> cachedTiles(
      minMaxPairs.size() / 2u, resolveMemoryResource(memory));
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
  shadow_detail::quantizeShadowZBounds(lightMin, lightMax, fit.texelWorldSize,
                                       true);

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

[[nodiscard]] bool lightSpaceBoundsOverlap(glm::vec3 casterMin,
                                           glm::vec3 casterMax,
                                           glm::vec3 cascadeMin,
                                           glm::vec3 cascadeMax,
                                           float padding) {
  const glm::vec3 normalizedCasterMin = glm::min(casterMin, casterMax);
  const glm::vec3 normalizedCasterMax = glm::max(casterMin, casterMax);
  casterMin = normalizedCasterMin;
  casterMax = normalizedCasterMax;
  const glm::vec3 normalizedCascadeMin = glm::min(cascadeMin, cascadeMax);
  const glm::vec3 normalizedCascadeMax = glm::max(cascadeMin, cascadeMax);
  cascadeMin = normalizedCascadeMin - glm::vec3(padding);
  cascadeMax = normalizedCascadeMax + glm::vec3(padding);
  return casterMax.x >= cascadeMin.x && casterMin.x <= cascadeMax.x &&
         casterMax.y >= cascadeMin.y && casterMin.y <= cascadeMax.y &&
         casterMax.z >= cascadeMin.z && casterMin.z <= cascadeMax.z;
}

[[nodiscard]] bool normalizedLightSpaceBoundsOverlap(
    const glm::vec3 &casterMin, const glm::vec3 &casterMax,
    const glm::vec3 &cascadeMin, const glm::vec3 &cascadeMax) {
  return casterMax.x >= cascadeMin.x && casterMin.x <= cascadeMax.x &&
         casterMax.y >= cascadeMin.y && casterMin.y <= cascadeMax.y &&
         casterMax.z >= cascadeMin.z && casterMin.z <= cascadeMax.z;
}

[[nodiscard]] bool staticOnlyRenderedFitContainsCurrent(
    const shadow_detail::DirectionalShadowFit &renderedFit,
    const shadow_detail::DirectionalShadowFit &currentFit) {
  return shadow_detail::directionalShadowFitContains(renderedFit, currentFit);
}

[[nodiscard]] shadow_detail::DirectionalShadowFit
makeStaticOnlyReuseFitForCurrentFrame(
    const shadow_detail::DirectionalShadowFit &cachedRenderedFit,
    const shadow_detail::DirectionalShadowFit &currentRawFit) {
  shadow_detail::DirectionalShadowFit fit = cachedRenderedFit;
  fit.splitNear = currentRawFit.splitNear;
  fit.splitFar = currentRawFit.splitFar;
  fit.frustumCenter = currentRawFit.frustumCenter;
  fit.frustumCorners = currentRawFit.frustumCorners;
  return fit;
}

[[nodiscard]] float
pcssReceiverDepthWorldScale(const shadow_detail::DirectionalShadowFit &fit) {
  const float splitDepthSpan = std::max(fit.splitFar - fit.splitNear, 1.0e-6f);
  const float lightDepthSpan =
      std::max(fit.lightSpaceBoundsMax.z - fit.lightSpaceBoundsMin.z, 1.0e-6f);
  return std::min(lightDepthSpan, splitDepthSpan * 2.0f);
}

[[nodiscard]] glm::vec3 paddedLightSpaceCullingMin(
    const shadow_detail::DirectionalShadowFit &fit) noexcept {
  const float padding =
      std::max(fit.texelWorldSize * kCullingPaddingTexelMultiplier, 0.01f);
  return glm::min(fit.lightSpaceBoundsMin, fit.lightSpaceBoundsMax) -
         glm::vec3(padding);
}

[[nodiscard]] glm::vec3 paddedLightSpaceCullingMax(
    const shadow_detail::DirectionalShadowFit &fit) noexcept {
  const float padding =
      std::max(fit.texelWorldSize * kCullingPaddingTexelMultiplier, 0.01f);
  return glm::max(fit.lightSpaceBoundsMin, fit.lightSpaceBoundsMax) +
         glm::vec3(padding);
}

void writeShadowCascadeFit(const shadow_detail::DirectionalShadowFit &fit,
                           uint32_t cascadeIndex,
                           const RenderSettings::ShadowSettings &settings,
                           uint32_t compareSamplerId, uint32_t rawSamplerId,
                           TextureHandle shadowDepthTexture, GPUDevice &gpu,
                           ShadowFrameGpuData &shadowFrameGpuData,
                           ShadowDebugFrameData &shadowDebugFrameData) {
  const glm::vec3 cullingBoundsMin = paddedLightSpaceCullingMin(fit);
  const glm::vec3 cullingBoundsMax = paddedLightSpaceCullingMax(fit);
  shadowFrameGpuData.cascades[cascadeIndex] = ShadowCascadeGpuData{
      .lightViewProj = fit.lightViewProj,
      .lightView = fit.lightView,
      .splitDepthTexelSize =
          glm::vec4(fit.splitNear, fit.splitFar, fit.texelWorldSize, 0.0f),
      .uvScaleBias = glm::vec4(1.0f, 1.0f, 0.0f, 0.0f),
      .biasParams = glm::vec4(settings.constantBias, settings.slopeBias,
                              settings.normalBias, 0.0f),
      .pcssParams = glm::vec4(pcssReceiverDepthWorldScale(fit),
                              settings.pcssSearchRadiusClampTexels,
                              settings.pcssFilterRadiusClampTexels, 0.0f),
      .textureSampler =
          glm::uvec4(gpu.getTextureBindlessIndex(shadowDepthTexture),
                     compareSamplerId, rawSamplerId, settings.shadowMapSize),
      .cullingBoundsMin = glm::vec4(cullingBoundsMin, 1.0f),
      .cullingBoundsMax = glm::vec4(cullingBoundsMax, 1.0f),
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
    const uint32_t lodCount =
        std::clamp(submesh.lodCount, 0u, Submesh::kMaxLodCount);
    if (lodCount > 0u && submesh.lods[0].indexCount > 0u) {
      return submesh.lods[0];
    }
    if (submesh.indexCount > 0u) {
      return SubmeshLod{.indexOffset = submesh.indexOffset,
                        .indexCount = submesh.indexCount,
                        .error = 0.0f};
    }
    for (uint32_t lod = 1u; lod < lodCount; ++lod) {
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
                        CullMode cullMode, Format depthFormat) {
  return RenderPipelineDesc{
      .vertexInput = {},
      .vertexShader = vertexShader,
      .fragmentShader = fragmentShader,
      .colorFormats = {Format::RGBA8_UNORM},
      .colorAttachmentCount = 0u,
      .depthFormat = sanitizeShadowDepthFormat(depthFormat),
      .cullMode = cullMode,
      .polygonMode = PolygonMode::Fill,
      .topology = Topology::Triangle,
      .blendEnabled = false,
  };
}

[[nodiscard]] MeshletPipelineDesc
shadowMeshletPipelineDesc(ShaderHandle taskShader, ShaderHandle meshShader,
                          ShaderHandle fragmentShader, CullMode cullMode,
                          Format depthFormat) {
  return MeshletPipelineDesc{
      .taskShader = taskShader,
      .meshShader = meshShader,
      .fragmentShader = fragmentShader,
      .colorFormats = {Format::RGBA8_UNORM},
      .colorAttachmentCount = 0u,
      .depthFormat = sanitizeShadowDepthFormat(depthFormat),
      .cullMode = cullMode,
      .polygonMode = PolygonMode::Fill,
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

[[nodiscard]] uint32_t visibilityReadbackAge(uint64_t currentFrame,
                                             uint32_t sourceFrame) {
  if (static_cast<uint64_t>(sourceFrame) >= currentFrame) {
    return 0u;
  }
  const uint64_t age = currentFrame - static_cast<uint64_t>(sourceFrame);
  return age > std::numeric_limits<uint32_t>::max()
             ? std::numeric_limits<uint32_t>::max()
             : static_cast<uint32_t>(age);
}

[[nodiscard]] bool isValidShadowDepthTexture(const GPUDevice &gpu,
                                             TextureHandle texture,
                                             uint32_t shadowMapSize,
                                             Format depthFormat) {
  if (!nuri::isValid(texture)) {
    return false;
  }
  const TextureDimensions dimensions = gpu.getTextureDimensions(texture);
  return dimensions.width == shadowMapSize &&
         dimensions.height == shadowMapSize &&
         gpu.getTextureFormat(texture) ==
             sanitizeShadowDepthFormat(depthFormat);
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
  signature = hashCombine64(signature, entry.useMeshlets ? 1ull : 0ull);
  signature =
      hashCombine64(signature, foldHandle(entry.indexBuffer.index,
                                          entry.indexBuffer.generation));
  signature = hashCombineValue(signature, entry.indexBufferOffset);
  signature =
      hashCombine64(signature, static_cast<uint64_t>(entry.indexFormat));
  signature =
      hashCombine64(signature, foldHandle(entry.vertexBuffer.index,
                                          entry.vertexBuffer.generation));
  signature =
      hashCombine64(signature, foldHandle(entry.vertexDecodeBuffer.index,
                                          entry.vertexDecodeBuffer.generation));
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
  if (entry.meshletView != nullptr) {
    signature = hashCombine64(
        signature, foldHandle(entry.meshletView->meshletBuffer.index,
                              entry.meshletView->meshletBuffer.generation));
    signature = hashCombine64(
        signature,
        foldHandle(entry.meshletView->meshletVertexIndexBuffer.index,
                   entry.meshletView->meshletVertexIndexBuffer.generation));
    signature = hashCombine64(
        signature,
        foldHandle(entry.meshletView->meshletPrimitiveIndexBuffer.index,
                   entry.meshletView->meshletPrimitiveIndexBuffer.generation));
    signature = hashCombine64(
        signature, foldHandle(entry.meshletView->lodRangeBuffer.index,
                              entry.meshletView->lodRangeBuffer.generation));
  }
  signature =
      hashCombine64(signature, static_cast<uint64_t>(entry.submeshIndex));
  signature =
      hashCombine64(signature, static_cast<uint64_t>(entry.meshletMaxCount));
  signature =
      hashCombine64(signature, static_cast<uint64_t>(entry.meshletCount));
  signature =
      hashCombine64(signature, static_cast<uint64_t>(entry.vertexOffset));
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
      shadowFrameRing_(memory_), shadowMeshletCounterRing_(memory_),
      sdsmReduceResultRing_(memory_), instanceDataRingUploadVersions_(memory_),
      instanceRemapUploadSignatures_(memory_),
      shadowFrameUploadSignatures_(memory_),
      shadowMeshletCounterRingPublishedFrames_(memory_),
      sdsmReduceResultRingPublishedFrames_(memory_),
      shadowMeshletCounterClear_(memory_), meshDrawTemplates_(memory_),
      batchBuildScratchArena_(memory_), staticShadowTemplateIndices_(memory_),
      dynamicShadowTemplateIndices_(memory_), staticShadowCasterCache_(memory_),
      staticShadowBatchTemplates_(memory_), staticShadowBatchIndexMap_(memory_),
      staticShadowBatchInstanceIndices_(memory_),
      staticShadowCasterDrawBuffers_(memory_),
      staticShadowCasterFitPoints_(memory_),
      staticShadowCasterLightSpaceBounds_(memory_),
      staticShadowBatchLightSpaceBounds_(memory_),
      staticShadowCasterLightGridCells_(memory_),
      staticShadowCasterLightGridEntries_(memory_),
      staticShadowCasterLargeLightGridEntries_(memory_),
      staticShadowBatchLightGridCells_(memory_),
      staticShadowBatchLightGridEntries_(memory_),
      staticShadowBatchLargeLightGridEntries_(memory_),
      staticShadowCasterLightGridQueryMarks_(memory_),
      staticShadowCasterLightGridQueryEntries_(memory_),
      instanceMatrices_(memory_), instanceRemap_(memory_),
      passBufferDependencies_(memory_), passDependencyBuffers_(memory_),
      passDependencyBufferAccessModes_(memory_),
      passDependencyBufferBindings_(memory_),
      passDependencyTextureBindings_(memory_), preResolvedDrawBuffers_(memory_),
      preResolvedDrawBufferIds_(memory_), passTextureDependencies_(memory_),
      previewTextureDependencies_(memory_), sdsmReadbackBuffer_(memory_) {
  sdsmReadbackBuffer_.resize(
      static_cast<size_t>(kSdsmHistogramSourceMaxTexelCount) * sizeof(float) *
      2u);
  for (uint32_t cascadeIndex = 0u; cascadeIndex < kMaxShadowCascades;
       ++cascadeIndex) {
    cascadePushConstants_[cascadeIndex] =
        std::pmr::vector<PushConstants>(memory_);
    cascadeDrawItems_[cascadeIndex] = std::pmr::vector<DrawItem>(memory_);
    cascadeMeshletPushConstants_[cascadeIndex] =
        std::pmr::vector<MeshletPushConstants>(memory_);
    cascadeMeshDispatchItems_[cascadeIndex] =
        std::pmr::vector<MeshDispatchItem>(memory_);
    cascadeMeshDispatchDependencyBuffers_[cascadeIndex] =
        std::pmr::vector<std::pmr::vector<BufferHandle>>(memory_);
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
  shadowOpaqueShader_ = Shader::create("shadow_depth_opaque", gpu_);
  depthShader_ = Shader::create("shadow_depth_only", gpu_);
  depthAlphaShader_ = Shader::create("shadow_depth_alpha", gpu_);
  if (!shadowShader_ || !shadowOpaqueShader_ || !depthShader_ ||
      !depthAlphaShader_) {
    return Result<bool, std::string>::makeError(
        "ShadowRenderer::createShaders: failed to create shader wrappers");
  }

  const std::filesystem::path shadowOpaqueVertexPath =
      config_.shaderBasePath / "shadow_depth_opaque.vert";
  const std::filesystem::path shadowVertexPath =
      config_.shaderBasePath / "shadow_depth.vert";
  const std::filesystem::path depthFragmentPath =
      config_.shaderBasePath / "opaque_depth.frag";
  const std::filesystem::path depthAlphaFragmentPath =
      config_.shaderBasePath / "opaque_depth_alpha.frag";

  auto opaqueVertexResult = shadowOpaqueShader_->compileFromFile(
      shadowOpaqueVertexPath.string(), ShaderStage::Vertex);
  if (opaqueVertexResult.hasError()) {
    return Result<bool, std::string>::makeError(opaqueVertexResult.error());
  }
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

  shadowOpaqueVertexShader_ = opaqueVertexResult.value();
  shadowVertexShader_ = vertexResult.value();
  depthFragmentShader_ = fragmentResult.value();
  depthAlphaFragmentShader_ = alphaFragmentResult.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> ShadowRenderer::createShadowMeshletShaders() {
  if (nuri::isValid(shadowMeshletTaskShader_) &&
      nuri::isValid(shadowMeshletMeshShader_) &&
      nuri::isValid(shadowMeshletDepthFragmentShader_) &&
      nuri::isValid(shadowMeshletDepthAlphaFragmentShader_)) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (!gpu_.supportsFeature(GPUFeature::Meshlets)) {
    return Result<bool, std::string>::makeError(
        "ShadowRenderer::createShadowMeshletShaders: GPU meshlets are "
        "unsupported");
  }
  if (config_.shaderBasePath.empty()) {
    return Result<bool, std::string>::makeError(
        "ShadowRenderer::createShadowMeshletShaders: shader base path is not "
        "configured");
  }

  shadowMeshletShader_ = Shader::create("shadow_meshlet", gpu_);
  if (!shadowMeshletShader_) {
    return Result<bool, std::string>::makeError(
        "ShadowRenderer::createShadowMeshletShaders: failed to create shader "
        "wrapper");
  }

  const std::filesystem::path taskPath =
      config_.shaderBasePath / "shadow_meshlet.task.glsl";
  const std::filesystem::path meshPath =
      config_.shaderBasePath / "shadow_meshlet.mesh.glsl";
  const std::filesystem::path depthFragmentPath =
      config_.shaderBasePath / "shadow_meshlet_depth.frag";
  const std::filesystem::path depthAlphaFragmentPath =
      config_.shaderBasePath / "shadow_meshlet_depth_alpha.frag";

  auto taskResult = shadowMeshletShader_->compileFromFile(taskPath.string(),
                                                          ShaderStage::Task);
  if (taskResult.hasError()) {
    shadowMeshletShader_.reset();
    return Result<bool, std::string>::makeError(taskResult.error());
  }
  auto meshResult = shadowMeshletShader_->compileFromFile(meshPath.string(),
                                                          ShaderStage::Mesh);
  if (meshResult.hasError()) {
    gpu_.destroyShaderModule(taskResult.value());
    shadowMeshletShader_.reset();
    return Result<bool, std::string>::makeError(meshResult.error());
  }
  auto fragmentResult = shadowMeshletShader_->compileFromFile(
      depthFragmentPath.string(), ShaderStage::Fragment);
  if (fragmentResult.hasError()) {
    gpu_.destroyShaderModule(meshResult.value());
    gpu_.destroyShaderModule(taskResult.value());
    shadowMeshletShader_.reset();
    return Result<bool, std::string>::makeError(fragmentResult.error());
  }
  auto alphaFragmentResult = shadowMeshletShader_->compileFromFile(
      depthAlphaFragmentPath.string(), ShaderStage::Fragment);
  if (alphaFragmentResult.hasError()) {
    gpu_.destroyShaderModule(fragmentResult.value());
    gpu_.destroyShaderModule(meshResult.value());
    gpu_.destroyShaderModule(taskResult.value());
    shadowMeshletShader_.reset();
    return Result<bool, std::string>::makeError(alphaFragmentResult.error());
  }

  shadowMeshletTaskShader_ = taskResult.value();
  shadowMeshletMeshShader_ = meshResult.value();
  shadowMeshletDepthFragmentShader_ = fragmentResult.value();
  shadowMeshletDepthAlphaFragmentShader_ = alphaFragmentResult.value();
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

Result<bool, std::string> ShadowRenderer::createPipelines(Format depthFormat) {
  const Format targetDepthFormat = sanitizeShadowDepthFormat(depthFormat);
  const auto destroyPipelineIfValid = [this](RenderPipelineHandle pipeline) {
    if (nuri::isValid(pipeline)) {
      gpu_.destroyRenderPipeline(pipeline);
    }
  };

  auto shadowResult = gpu_.createRenderPipeline(
      shadowDepthPipelineDesc(shadowOpaqueVertexShader_, depthFragmentShader_,
                              CullMode::Back, targetDepthFormat),
      "shadow_depth_opaque");
  if (shadowResult.hasError()) {
    return Result<bool, std::string>::makeError(shadowResult.error());
  }
  RenderPipelineHandle newShadowPipeline = shadowResult.value();

  auto doubleSidedResult = gpu_.createRenderPipeline(
      shadowDepthPipelineDesc(shadowOpaqueVertexShader_, depthFragmentShader_,
                              CullMode::None, targetDepthFormat),
      "shadow_depth_opaque_double_sided");
  if (doubleSidedResult.hasError()) {
    destroyPipelineIfValid(newShadowPipeline);
    return Result<bool, std::string>::makeError(doubleSidedResult.error());
  }
  RenderPipelineHandle newShadowDoubleSidedPipeline = doubleSidedResult.value();

  auto alphaResult = gpu_.createRenderPipeline(
      shadowDepthPipelineDesc(shadowVertexShader_, depthAlphaFragmentShader_,
                              CullMode::Back, targetDepthFormat),
      "shadow_depth_alpha");
  if (alphaResult.hasError()) {
    destroyPipelineIfValid(newShadowDoubleSidedPipeline);
    destroyPipelineIfValid(newShadowPipeline);
    return Result<bool, std::string>::makeError(alphaResult.error());
  }
  RenderPipelineHandle newShadowAlphaPipeline = alphaResult.value();

  auto alphaDoubleSidedResult = gpu_.createRenderPipeline(
      shadowDepthPipelineDesc(shadowVertexShader_, depthAlphaFragmentShader_,
                              CullMode::None, targetDepthFormat),
      "shadow_depth_alpha_double_sided");
  if (alphaDoubleSidedResult.hasError()) {
    destroyPipelineIfValid(newShadowAlphaPipeline);
    destroyPipelineIfValid(newShadowDoubleSidedPipeline);
    destroyPipelineIfValid(newShadowPipeline);
    return Result<bool, std::string>::makeError(alphaDoubleSidedResult.error());
  }
  RenderPipelineHandle newShadowAlphaDoubleSidedPipeline =
      alphaDoubleSidedResult.value();

  const RenderPipelineHandle oldShadowPipeline = shadowPipelineHandle_;
  const RenderPipelineHandle oldShadowDoubleSidedPipeline =
      shadowDoubleSidedPipelineHandle_;
  const RenderPipelineHandle oldShadowAlphaPipeline =
      shadowAlphaPipelineHandle_;
  const RenderPipelineHandle oldShadowAlphaDoubleSidedPipeline =
      shadowAlphaDoubleSidedPipelineHandle_;
  shadowPipelineHandle_ = newShadowPipeline;
  shadowDoubleSidedPipelineHandle_ = newShadowDoubleSidedPipeline;
  shadowAlphaPipelineHandle_ = newShadowAlphaPipeline;
  shadowAlphaDoubleSidedPipelineHandle_ = newShadowAlphaDoubleSidedPipeline;
  shadowDepthPipelineFormat_ = targetDepthFormat;

  destroyPipelineIfValid(oldShadowDoubleSidedPipeline);
  destroyPipelineIfValid(oldShadowAlphaDoubleSidedPipeline);
  destroyPipelineIfValid(oldShadowAlphaPipeline);
  destroyPipelineIfValid(oldShadowPipeline);
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
ShadowRenderer::ensureShadowMeshletPipelineState(Format depthFormat) {
  const Format targetDepthFormat = sanitizeShadowDepthFormat(depthFormat);
  if (nuri::isValid(shadowMeshletPipelineHandle_) &&
      nuri::isValid(shadowMeshletDoubleSidedPipelineHandle_) &&
      nuri::isValid(shadowMeshletAlphaPipelineHandle_) &&
      nuri::isValid(shadowMeshletAlphaDoubleSidedPipelineHandle_) &&
      shadowMeshletDepthPipelineFormat_ == targetDepthFormat) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (shadowMeshletPipelineUnsupported_) {
    return Result<bool, std::string>::makeError(
        "ShadowRenderer::ensureShadowMeshletPipelineState: meshlet shadow "
        "pipeline is unsupported");
  }
  if (!gpu_.supportsFeature(GPUFeature::Meshlets)) {
    shadowMeshletPipelineUnsupported_ = true;
    return Result<bool, std::string>::makeError(
        "ShadowRenderer::ensureShadowMeshletPipelineState: GPU meshlets are "
        "unsupported");
  }

  auto shaderResult = createShadowMeshletShaders();
  if (shaderResult.hasError()) {
    shadowMeshletPipelineUnsupported_ = true;
    return shaderResult;
  }
  if (!nuri::isValid(shadowMeshletDepthFragmentShader_) ||
      !nuri::isValid(shadowMeshletDepthAlphaFragmentShader_)) {
    return Result<bool, std::string>::makeError(
        "ShadowRenderer::ensureShadowMeshletPipelineState: invalid depth "
        "fragment shaders");
  }

  destroyShadowMeshletPipelineState();
  const auto destroyPipelineIfValid = [this](MeshletPipelineHandle pipeline) {
    if (nuri::isValid(pipeline)) {
      gpu_.destroyMeshletPipeline(pipeline);
    }
  };

  auto shadowResult = gpu_.createMeshletPipeline(
      shadowMeshletPipelineDesc(
          shadowMeshletTaskShader_, shadowMeshletMeshShader_,
          shadowMeshletDepthFragmentShader_, CullMode::Back, targetDepthFormat),
      "shadow_meshlet_depth");
  if (shadowResult.hasError()) {
    shadowMeshletPipelineUnsupported_ = true;
    return Result<bool, std::string>::makeError(shadowResult.error());
  }
  MeshletPipelineHandle newShadowPipeline = shadowResult.value();

  auto doubleSidedResult = gpu_.createMeshletPipeline(
      shadowMeshletPipelineDesc(
          shadowMeshletTaskShader_, shadowMeshletMeshShader_,
          shadowMeshletDepthFragmentShader_, CullMode::None, targetDepthFormat),
      "shadow_meshlet_depth_double_sided");
  if (doubleSidedResult.hasError()) {
    destroyPipelineIfValid(newShadowPipeline);
    shadowMeshletPipelineUnsupported_ = true;
    return Result<bool, std::string>::makeError(doubleSidedResult.error());
  }
  MeshletPipelineHandle newShadowDoubleSidedPipeline =
      doubleSidedResult.value();

  auto alphaResult = gpu_.createMeshletPipeline(
      shadowMeshletPipelineDesc(shadowMeshletTaskShader_,
                                shadowMeshletMeshShader_,
                                shadowMeshletDepthAlphaFragmentShader_,
                                CullMode::Back, targetDepthFormat),
      "shadow_meshlet_depth_alpha");
  if (alphaResult.hasError()) {
    destroyPipelineIfValid(newShadowDoubleSidedPipeline);
    destroyPipelineIfValid(newShadowPipeline);
    shadowMeshletPipelineUnsupported_ = true;
    return Result<bool, std::string>::makeError(alphaResult.error());
  }
  MeshletPipelineHandle newShadowAlphaPipeline = alphaResult.value();

  auto alphaDoubleSidedResult = gpu_.createMeshletPipeline(
      shadowMeshletPipelineDesc(shadowMeshletTaskShader_,
                                shadowMeshletMeshShader_,
                                shadowMeshletDepthAlphaFragmentShader_,
                                CullMode::None, targetDepthFormat),
      "shadow_meshlet_depth_alpha_double_sided");
  if (alphaDoubleSidedResult.hasError()) {
    destroyPipelineIfValid(newShadowAlphaPipeline);
    destroyPipelineIfValid(newShadowDoubleSidedPipeline);
    destroyPipelineIfValid(newShadowPipeline);
    shadowMeshletPipelineUnsupported_ = true;
    return Result<bool, std::string>::makeError(alphaDoubleSidedResult.error());
  }

  shadowMeshletPipelineHandle_ = newShadowPipeline;
  shadowMeshletDoubleSidedPipelineHandle_ = newShadowDoubleSidedPipeline;
  shadowMeshletAlphaPipelineHandle_ = newShadowAlphaPipeline;
  shadowMeshletAlphaDoubleSidedPipelineHandle_ = alphaDoubleSidedResult.value();
  shadowMeshletDepthPipelineFormat_ = targetDepthFormat;
  shadowMeshletPipelineUnsupported_ = false;
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
  auto pipelineResult = createPipelines(kDefaultShadowMapDepthFormat);
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
  const Format targetDepthFormat =
      sanitizeShadowDepthFormat(settings.depthFormat);
  bool depthValid = nuri::isValid(rawDepthSampler_) &&
                    nuri::isValid(compareDepthSampler_) &&
                    activeCascadeCount_ == requestedCascadeCount;
  for (uint32_t cascadeIndex = 0u;
       depthValid && cascadeIndex < requestedCascadeCount; ++cascadeIndex) {
    depthValid =
        isValidShadowDepthTexture(gpu_, shadowDepthTextures_[cascadeIndex],
                                  settings.shadowMapSize, targetDepthFormat);
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
        .format = targetDepthFormat,
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
  const SamplerFilter compareSamplerFilter =
      gpu_.supportsSampledImageLinearFiltering(targetDepthFormat)
          ? SamplerFilter::Linear
          : SamplerFilter::Nearest;
  compareSamplerDesc.minFilter = compareSamplerFilter;
  compareSamplerDesc.magFilter = compareSamplerFilter;
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
  while (shadowMeshletCounterRing_.size() < safeCount) {
    shadowMeshletCounterRing_.push_back(DynamicBufferSlot{});
  }
  instanceDataRingUploadVersions_.resize(safeCount,
                                         std::numeric_limits<uint64_t>::max());
  instanceRemapUploadSignatures_.resize(safeCount,
                                        std::numeric_limits<uint64_t>::max());
  shadowFrameUploadSignatures_.resize(safeCount,
                                      std::numeric_limits<uint64_t>::max());
  shadowMeshletCounterRingPublishedFrames_.resize(
      safeCount, std::numeric_limits<uint64_t>::max());
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
                            instanceRemapUploadSignatures_);
}

Result<bool, std::string>
ShadowRenderer::ensureShadowFrameRingCapacity(size_t requiredBytes) {
  return ensureRingCapacity(shadowFrameRing_, requiredBytes,
                            "shadow_frame_gpu_data",
                            shadowFrameUploadSignatures_);
}

Result<bool, std::string>
ShadowRenderer::ensureShadowMeshletCounterRingCapacity(size_t requiredBytes) {
  const size_t requested =
      std::max(requiredBytes, sizeof(VisibilityCounterGpuData));
  bool needsGrowth = false;
  for (const DynamicBufferSlot &slot : shadowMeshletCounterRing_) {
    if (slot.buffer && slot.buffer->valid() && slot.capacityBytes < requested) {
      needsGrowth = true;
      break;
    }
  }
  if (needsGrowth) {
    gpu_.waitIdle();
  }
  for (size_t i = 0u; i < shadowMeshletCounterRing_.size(); ++i) {
    DynamicBufferSlot &slot = shadowMeshletCounterRing_[i];
    if (slot.buffer && slot.buffer->valid() &&
        slot.capacityBytes >= requested) {
      continue;
    }
    if (slot.buffer && slot.buffer->valid()) {
      gpu_.destroyBuffer(slot.buffer->handle());
    }
    slot.buffer.reset();
    slot.capacityBytes = 0u;

    const BufferDesc desc{
        .usage = BufferUsage::Storage,
        .storage = Storage::HostVisible,
        .size = requested,
    };
    std::string debugName("shadow_meshlet_counter_buffer_");
    debugName += std::to_string(i);
    auto bufferResult = Buffer::create(gpu_, desc, debugName);
    if (bufferResult.hasError()) {
      return Result<bool, std::string>::makeError(bufferResult.error());
    }
    slot.buffer = std::move(bufferResult.value());
    slot.capacityBytes = requested;
    if (i < shadowMeshletCounterRingPublishedFrames_.size()) {
      shadowMeshletCounterRingPublishedFrames_[i] =
          std::numeric_limits<uint64_t>::max();
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

void ShadowRenderer::readLatestShadowMeshletCounterReadback(
    RenderFrameContext &frame) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_COPY);
  VisibilityFrameMetrics &metrics = frame.metrics.visibility;
  std::optional<VisibilityCounterGpuData> selectedCounter;
  uint32_t selectedSourceFrame = 0u;
  uint32_t readbackErrorCount = 0u;
  const auto readCounterSlot = [&](size_t slotIndex) {
    if (slotIndex >= shadowMeshletCounterRing_.size()) {
      return;
    }
    const DynamicBufferSlot &slot = shadowMeshletCounterRing_[slotIndex];
    if (!slot.buffer || !slot.buffer->valid()) {
      return;
    }
    const uint64_t expectedFrame =
        slotIndex < shadowMeshletCounterRingPublishedFrames_.size()
            ? shadowMeshletCounterRingPublishedFrames_[slotIndex]
            : std::numeric_limits<uint64_t>::max();
    if (expectedFrame == std::numeric_limits<uint64_t>::max()) {
      return;
    }

    VisibilityCounterGpuData counter{};
    auto readResult =
        gpu_.readBuffer(slot.buffer->handle(), 0u,
                        std::as_writable_bytes(
                            std::span<VisibilityCounterGpuData>(&counter, 1u)));
    if (readResult.hasError()) {
      ++readbackErrorCount;
      return;
    }
    const uint32_t valid = counter.status.w;
    const uint32_t sourceFrame = counter.status.z;
    if (valid == 0u || static_cast<uint64_t>(sourceFrame) >= frame.frameIndex ||
        sourceFrame != static_cast<uint32_t>(expectedFrame)) {
      return;
    }
    if (!selectedCounter.has_value() || sourceFrame > selectedSourceFrame) {
      selectedCounter = counter;
      selectedSourceFrame = sourceFrame;
    }
  };

  const size_t preferredSlotIndex =
      frame.frameIndex > 0u && !shadowMeshletCounterRing_.empty()
          ? static_cast<size_t>((frame.frameIndex - 1u) %
                                shadowMeshletCounterRing_.size())
          : std::numeric_limits<size_t>::max();
  readCounterSlot(preferredSlotIndex);
  if (!selectedCounter.has_value()) {
    for (size_t slotIndex = 0u; slotIndex < shadowMeshletCounterRing_.size();
         ++slotIndex) {
      if (slotIndex == preferredSlotIndex) {
        continue;
      }
      readCounterSlot(slotIndex);
    }
  }

  metrics.shadowMeshletReadbackErrorCount = readbackErrorCount;
  if (selectedCounter.has_value()) {
    metrics.shadowMeshletReadbackAvailable = 1u;
    metrics.shadowMeshletReadbackSourceFrame = selectedSourceFrame;
    metrics.shadowMeshletReadbackStaleFrameCount =
        visibilityReadbackAge(frame.frameIndex, selectedSourceFrame);
    metrics.shadowMeshletRejectedBounds = selectedCounter->meshlet.x;
  }
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
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_COPY);
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
    const Model::ModelMeshletGpuView *meshletView =
        modelRecord->model->hasMeshlets()
            ? &modelRecord->model->meshletGpuView()
            : nullptr;

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
          .vertexDecodeBuffer = modelRecord->model->vertexDecodeBuffer(),
          .vertexBufferByteOffset = geometry.vertexByteOffset,
          .vertexBufferAddress = vertexBufferAddress,
          .vertexDecodeBufferAddress =
              modelRecord->model->vertexDecodeBufferAddress(),
          .vertexDecodeIndex = submeshIndex,
          .packedVertexFormat =
              static_cast<uint32_t>(modelRecord->model->drawVertexFormat()),
          .materialIndex = materialRecord != nullptr
                               ? resources.materialTableIndex(resolvedMaterial)
                               : 0u,
          .meshletView = meshletView,
          .meshletMaxCount = maxMeshletCountForSubmesh(submesh),
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
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_COPY);
  staticShadowCasterCache_.clear();
  staticShadowBatchTemplates_.clear();
  staticShadowBatchIndexMap_.clear();
  staticShadowBatchInstanceIndices_.clear();
  staticShadowCasterDrawBuffers_.clear();
  staticShadowCasterFitPoints_.clear();
  staticShadowCasterLightSpaceBounds_.clear();
  staticShadowBatchLightSpaceBounds_.clear();
  staticShadowCasterLightGridCells_.clear();
  staticShadowCasterLightGridEntries_.clear();
  staticShadowCasterLargeLightGridEntries_.clear();
  staticShadowBatchLightGridCells_.clear();
  staticShadowBatchLightGridEntries_.clear();
  staticShadowBatchLargeLightGridEntries_.clear();
  staticShadowCasterLightGridQueryMarks_.clear();
  staticShadowCasterLightGridQueryEntries_.clear();
  staticShadowCasterLightGridQueryMarker_ = 1u;
  staticShadowCasterLightGrid_ = {};
  staticShadowBatchLightGrid_ = {};
  staticShadowCasterBoundsMin_ = glm::vec3(std::numeric_limits<float>::max());
  staticShadowCasterBoundsMax_ =
      glm::vec3(std::numeric_limits<float>::lowest());
  staticShadowCasterLightSpaceBoundsMin_ =
      glm::vec3(std::numeric_limits<float>::max());
  staticShadowCasterLightSpaceBoundsMax_ =
      glm::vec3(std::numeric_limits<float>::lowest());
  staticShadowCasterCacheContentSignature_ = kFnvOffsetBasis64;
  staticShadowCasterCacheIndexCountEstimate_ = 0u;
  hasStaticShadowCasterBounds_ = false;
  hasStaticShadowCasterLightDepthBounds_ = false;
  hasStaticShadowCasterLightSpaceBounds_ = false;
  if (staticShadowTemplateIndices_.empty()) {
    staticShadowCasterCacheValid_ = true;
    return Result<bool, std::string>::makeResult(true);
  }

  const std::span<const Renderable> renderables = scene.renderables();
  const bool enableCascadeCasterCulling =
      settings.shadow.debug.enableCascadeCasterCulling;
  staticShadowCasterCache_.reserve(staticShadowTemplateIndices_.size());
  staticShadowBatchTemplates_.reserve(staticShadowTemplateIndices_.size());
  staticShadowBatchIndexMap_.reserve(staticShadowTemplateIndices_.size());
  staticShadowCasterDrawBuffers_.reserve(staticShadowTemplateIndices_.size() *
                                         2u);
  staticShadowCasterFitPoints_.reserve(staticShadowTemplateIndices_.size() *
                                       8u);
  const auto selectShadowPipeline =
      [&](bool doubleSided, bool alphaMasked) -> RenderPipelineHandle {
    if (alphaMasked) {
      return doubleSided && nuri::isValid(shadowAlphaDoubleSidedPipelineHandle_)
                 ? shadowAlphaDoubleSidedPipelineHandle_
                 : shadowAlphaPipelineHandle_;
    }
    return doubleSided && nuri::isValid(shadowDoubleSidedPipelineHandle_)
               ? shadowDoubleSidedPipelineHandle_
               : shadowPipelineHandle_;
  };
  const bool shadowMeshletAvailable =
      nuri::isValid(shadowMeshletPipelineHandle_) &&
      nuri::isValid(shadowMeshletDoubleSidedPipelineHandle_) &&
      nuri::isValid(shadowMeshletAlphaPipelineHandle_) &&
      nuri::isValid(shadowMeshletAlphaDoubleSidedPipelineHandle_);
  const auto selectShadowMeshletPipeline =
      [&](bool doubleSided, bool alphaMasked) -> MeshletPipelineHandle {
    if (alphaMasked) {
      return doubleSided &&
                     nuri::isValid(shadowMeshletAlphaDoubleSidedPipelineHandle_)
                 ? shadowMeshletAlphaDoubleSidedPipelineHandle_
                 : shadowMeshletAlphaPipelineHandle_;
    }
    return doubleSided && nuri::isValid(shadowMeshletDoubleSidedPipelineHandle_)
               ? shadowMeshletDoubleSidedPipelineHandle_
               : shadowMeshletPipelineHandle_;
  };
  const auto makeStaticShadowBatchKey =
      [](const StaticShadowBatchTemplate &batchTemplate) {
        const Model::ModelMeshletGpuView *meshletView =
            batchTemplate.meshletView;
        return StaticShadowBatchKey{
            .useMeshlets = batchTemplate.useMeshlets,
            .pipeline = batchTemplate.pipeline,
            .meshletPipeline = batchTemplate.meshletPipeline,
            .vertexBuffer = batchTemplate.vertexBuffer,
            .vertexDecodeBuffer = batchTemplate.vertexDecodeBuffer,
            .indexBuffer = batchTemplate.indexBuffer,
            .indexBufferOffset = batchTemplate.indexBufferOffset,
            .indexFormat = batchTemplate.indexFormat,
            .indexCount = batchTemplate.indexCount,
            .firstIndex = batchTemplate.firstIndex,
            .vertexBufferAddress = batchTemplate.vertexBufferAddress,
            .vertexDecodeBufferAddress =
                batchTemplate.vertexDecodeBufferAddress,
            .vertexDecodeIndex = batchTemplate.vertexDecodeIndex,
            .packedVertexFormat = batchTemplate.packedVertexFormat,
            .materialIndex = batchTemplate.materialIndex,
            .meshletBuffer = meshletView != nullptr ? meshletView->meshletBuffer
                                                    : BufferHandle{},
            .meshletVertexIndexBuffer =
                meshletView != nullptr ? meshletView->meshletVertexIndexBuffer
                                       : BufferHandle{},
            .meshletPrimitiveIndexBuffer =
                meshletView != nullptr
                    ? meshletView->meshletPrimitiveIndexBuffer
                    : BufferHandle{},
            .meshletLodRangeBuffer = meshletView != nullptr
                                         ? meshletView->lodRangeBuffer
                                         : BufferHandle{},
            .submeshIndex = batchTemplate.submeshIndex,
            .meshletMaxCount = batchTemplate.meshletMaxCount,
            .meshletCount = batchTemplate.meshletCount,
            .vertexOffset = batchTemplate.vertexOffset,
            .enableMeshletCascadeCulling =
                batchTemplate.enableMeshletCascadeCulling,
        };
      };
  const auto resolveStaticBatchIndex =
      [&](const StaticShadowBatchTemplate &candidate) {
        const StaticShadowBatchKey key = makeStaticShadowBatchKey(candidate);
        const auto it = staticShadowBatchIndexMap_.find(key);
        if (it != staticShadowBatchIndexMap_.end()) {
          return it->second;
        }
        const uint32_t batchIndex =
            static_cast<uint32_t>(staticShadowBatchTemplates_.size());
        staticShadowBatchTemplates_.push_back(candidate);
        staticShadowBatchIndexMap_.emplace(key, batchIndex);
        return batchIndex;
      };
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
    const bool useMeshlets =
        shadowMeshletAvailable &&
        canUseShadowMeshlets(entry.meshletView, entry.submeshIndex,
                             entry.meshletMaxCount, lod->meshletCount);

    StaticShadowCasterCacheEntry cachedEntry{
        .templateIndex = templateIndex,
        .instanceIndex = entry.instanceIndex,
        .indexBuffer = entry.indexBuffer,
        .indexBufferOffset = entry.indexBufferOffset,
        .indexFormat = entry.indexFormat,
        .vertexBuffer = entry.baseVertexBuffer,
        .vertexDecodeBuffer = entry.vertexDecodeBuffer,
        .vertexBufferAddress = entry.vertexBufferAddress,
        .vertexDecodeBufferAddress = entry.vertexDecodeBufferAddress,
        .vertexDecodeIndex = entry.vertexDecodeIndex,
        .packedVertexFormat = entry.packedVertexFormat,
        .materialIndex = entry.materialIndex,
        .indexCount = lod->indexCount,
        .firstIndex = lod->indexOffset,
        .meshletView = entry.meshletView,
        .submeshIndex = entry.submeshIndex,
        .meshletMaxCount = entry.meshletMaxCount,
        .meshletCount = lod->meshletCount,
        .vertexOffset = entry.submesh->vertexOffset,
        .enableMeshletCascadeCulling =
            useMeshlets && settings.shadow.enableMeshletCascadeCulling,
        .doubleSided = entry.doubleSided,
        .alphaMasked = entry.alphaMasked,
        .useMeshlets = useMeshlets,
        .hasCasterCullingBounds = false,
        .casterWorldCorners = {},
    };
    cachedEntry.rasterSignature =
        hashStaticShadowCasterRasterSignature(kFnvOffsetBasis64, cachedEntry);
    const StaticShadowBatchTemplate batchTemplate{
        .useMeshlets = useMeshlets,
        .pipeline = selectShadowPipeline(entry.doubleSided, entry.alphaMasked),
        .meshletPipeline = useMeshlets
                               ? selectShadowMeshletPipeline(entry.doubleSided,
                                                             entry.alphaMasked)
                               : MeshletPipelineHandle{},
        .vertexBuffer = cachedEntry.vertexBuffer,
        .vertexDecodeBuffer = cachedEntry.vertexDecodeBuffer,
        .indexBuffer = cachedEntry.indexBuffer,
        .indexBufferOffset = cachedEntry.indexBufferOffset,
        .indexFormat = cachedEntry.indexFormat,
        .indexCount = cachedEntry.indexCount,
        .firstIndex = cachedEntry.firstIndex,
        .vertexBufferAddress = cachedEntry.vertexBufferAddress,
        .vertexDecodeBufferAddress = cachedEntry.vertexDecodeBufferAddress,
        .vertexDecodeIndex = cachedEntry.vertexDecodeIndex,
        .packedVertexFormat = cachedEntry.packedVertexFormat,
        .materialIndex = cachedEntry.materialIndex,
        .meshletView = cachedEntry.meshletView,
        .submeshIndex = cachedEntry.submeshIndex,
        .meshletMaxCount = cachedEntry.meshletMaxCount,
        .meshletCount = cachedEntry.meshletCount,
        .vertexOffset = cachedEntry.vertexOffset,
        .enableMeshletCascadeCulling = cachedEntry.enableMeshletCascadeCulling,
        .firstInstanceIndex = 0u,
        .instanceCount = 0u,
        .rasterSignature = kFnvOffsetBasis64,
        .indexCountEstimate = 0u,
    };
    cachedEntry.batchIndex = resolveStaticBatchIndex(batchTemplate);
    StaticShadowBatchTemplate &resolvedBatch =
        staticShadowBatchTemplates_[cachedEntry.batchIndex];
    ++resolvedBatch.instanceCount;
    resolvedBatch.rasterSignature = hashCombine64(resolvedBatch.rasterSignature,
                                                  cachedEntry.rasterSignature);
    resolvedBatch.indexCountEstimate += cachedEntry.indexCount;
    const BoundingBox worldBounds = entry.submesh->bounds.getTransformed(
        renderables[entry.instanceIndex].modelMatrix);
    cachedEntry.casterWorldCorners =
        shadow_detail::computeBoundsCorners(worldBounds.min_, worldBounds.max_);
    cachedEntry.hasCasterCullingBounds = enableCascadeCasterCulling;
    staticShadowCasterFitPoints_.insert(staticShadowCasterFitPoints_.end(),
                                        cachedEntry.casterWorldCorners.begin(),
                                        cachedEntry.casterWorldCorners.end());
    staticShadowCasterBoundsMin_ =
        glm::min(staticShadowCasterBoundsMin_, worldBounds.min_);
    staticShadowCasterBoundsMax_ =
        glm::max(staticShadowCasterBoundsMax_, worldBounds.max_);
    hasStaticShadowCasterBounds_ = true;
    staticShadowCasterCache_.push_back(cachedEntry);
    staticShadowCasterCacheContentSignature_ = hashCombine64(
        staticShadowCasterCacheContentSignature_, cachedEntry.rasterSignature);
    staticShadowCasterCacheIndexCountEstimate_ += cachedEntry.indexCount;
    if (!cachedEntry.useMeshlets) {
      appendUniqueBufferHandle(staticShadowCasterDrawBuffers_,
                               cachedEntry.vertexBuffer);
      appendUniqueBufferHandle(staticShadowCasterDrawBuffers_,
                               cachedEntry.indexBuffer);
    }
  }

  staticShadowBatchInstanceIndices_.resize(staticShadowCasterCache_.size());
  uint32_t nextBatchInstanceIndex = 0u;
  for (StaticShadowBatchTemplate &batch : staticShadowBatchTemplates_) {
    batch.firstInstanceIndex = nextBatchInstanceIndex;
    nextBatchInstanceIndex += batch.instanceCount;
  }
  std::pmr::vector<uint32_t> staticBatchWriteOffsets(memory_);
  staticBatchWriteOffsets.reserve(staticShadowBatchTemplates_.size());
  for (const StaticShadowBatchTemplate &batch : staticShadowBatchTemplates_) {
    staticBatchWriteOffsets.push_back(batch.firstInstanceIndex);
  }
  for (const StaticShadowCasterCacheEntry &entry : staticShadowCasterCache_) {
    NURI_ASSERT(entry.batchIndex < staticBatchWriteOffsets.size(),
                "Static shadow caster references an invalid batch");
    const uint32_t writeIndex = staticBatchWriteOffsets[entry.batchIndex]++;
    NURI_ASSERT(writeIndex < staticShadowBatchInstanceIndices_.size(),
                "Static shadow batch write offset is out of bounds");
    staticShadowBatchInstanceIndices_[writeIndex] = entry.instanceIndex;
  }

  staticShadowCasterCacheValid_ = true;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> ShadowRenderer::updateShadowFrameData(
    RenderFrameContext &frame, const RenderSettings::ShadowSettings &settings,
    uint32_t shadowMapSize, int32_t forcedMeshLod) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_COPY);
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
  const bool staticCasterCacheMatchesFrame =
      frame.scene != nullptr &&
      staticShadowCasterCacheTransformVersion_ ==
          frame.scene->transformVersion() &&
      staticShadowCasterCacheForcedMeshLod_ == forcedMeshLod;
  const bool canUseStaticCasterFitCache = animationSceneData == nullptr &&
                                          staticShadowCasterCacheValid_ &&
                                          staticCasterCacheMatchesFrame;
  const bool needCasterLightDepthBounds = cascadeCount > 1u;
  const glm::mat4 casterLightView =
      needCasterLightDepthBounds
          ? shadow_detail::makeDirectionalLightView(lightDirection)
          : glm::mat4(1.0f);
  glm::vec2 casterLightDepthBounds(std::numeric_limits<float>::max(),
                                   std::numeric_limits<float>::lowest());
  bool hasCasterLightDepthBounds = false;
  const auto combineCasterLightDepthBounds = [&](glm::vec2 bounds) {
    if (bounds.x > bounds.y) {
      return;
    }
    const float casterMin = std::min(bounds.x, bounds.y);
    const float casterMax = std::max(bounds.x, bounds.y);
    if (!std::isfinite(casterMin) || !std::isfinite(casterMax)) {
      return;
    }
    casterLightDepthBounds.x = std::min(casterLightDepthBounds.x, casterMin);
    casterLightDepthBounds.y = std::max(casterLightDepthBounds.y, casterMax);
    hasCasterLightDepthBounds = true;
  };
  const auto computeCasterLightDepthBounds =
      [&](std::span<const glm::vec3> points) {
        glm::vec2 bounds(std::numeric_limits<float>::max(),
                         std::numeric_limits<float>::lowest());
        bool hasBounds = false;
        for (const glm::vec3 point : points) {
          const float z = glm::vec3(casterLightView * glm::vec4(point, 1.0f)).z;
          if (!std::isfinite(z)) {
            continue;
          }
          bounds.x = std::min(bounds.x, z);
          bounds.y = std::max(bounds.y, z);
          hasBounds = true;
        }
        return hasBounds ? bounds
                         : glm::vec2(std::numeric_limits<float>::max(),
                                     std::numeric_limits<float>::lowest());
      };
  const auto ensureStaticCasterLightDepthBounds = [&]() {
    if (!needCasterLightDepthBounds || staticShadowCasterFitPoints_.empty()) {
      hasStaticShadowCasterLightDepthBounds_ = false;
      return;
    }
    if (hasStaticShadowCasterLightDepthBounds_ &&
        rawBytesEqual(staticShadowCasterLightDepthDirection_, lightDirection)) {
      return;
    }
    const glm::vec2 bounds = computeCasterLightDepthBounds(
        std::span<const glm::vec3>(staticShadowCasterFitPoints_.data(),
                                   staticShadowCasterFitPoints_.size()));
    staticShadowCasterLightDepthDirection_ = lightDirection;
    staticShadowCasterLightDepthBounds_ = bounds;
    hasStaticShadowCasterLightDepthBounds_ = bounds.x <= bounds.y;
  };
  BoundingBox casterBounds{};
  bool hasCasterBounds = false;
  bool hasAnimatedGeometryOverrides = false;
  const auto appendCasterFitBounds = [&](const MeshDrawTemplate &entry) {
    if (entry.renderable == nullptr || entry.submesh == nullptr) {
      return;
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
      return;
    }
    const BoundingBox worldBounds =
        entry.submesh->bounds.getTransformed(entry.renderable->modelMatrix);
    if (needCasterLightDepthBounds) {
      const auto worldBoundsCorners = shadow_detail::computeBoundsCorners(
          worldBounds.min_, worldBounds.max_);
      combineCasterLightDepthBounds(computeCasterLightDepthBounds(
          std::span<const glm::vec3, 8>(worldBoundsCorners)));
    }
    casterBounds.combinePoint(worldBounds.min_);
    casterBounds.combinePoint(worldBounds.max_);
    hasCasterBounds = true;
  };
  {
    NURI_PROFILER_ZONE("ShadowRenderer.update.static_caster_bounds",
                       NURI_PROFILER_COLOR_CMD_COPY);
    if (canUseStaticCasterFitCache && hasStaticShadowCasterBounds_) {
      casterBounds.combinePoint(staticShadowCasterBoundsMin_);
      casterBounds.combinePoint(staticShadowCasterBoundsMax_);
      hasCasterBounds = true;
      if (needCasterLightDepthBounds) {
        ensureStaticCasterLightDepthBounds();
        if (hasStaticShadowCasterLightDepthBounds_) {
          combineCasterLightDepthBounds(staticShadowCasterLightDepthBounds_);
        }
      }
    }
    NURI_PROFILER_ZONE_END();
  }
  {
    NURI_PROFILER_ZONE("ShadowRenderer.update.dynamic_caster_bounds",
                       NURI_PROFILER_COLOR_CMD_COPY);
    if (canUseStaticCasterFitCache) {
      for (const uint32_t templateIndex : dynamicShadowTemplateIndices_) {
        if (templateIndex < meshDrawTemplates_.size()) {
          appendCasterFitBounds(meshDrawTemplates_[templateIndex]);
        }
      }
    } else {
      for (const MeshDrawTemplate &entry : meshDrawTemplates_) {
        appendCasterFitBounds(entry);
      }
    }
    NURI_PROFILER_ZONE_END();
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
      const auto consumeCpuMinMaxSource = [&]() -> bool {
        const TextureHandle sdsmTexture =
            frame.sharedResources.sceneDepthPyramidTextures
                [frame.sharedResources.sceneDepthPyramidLevelCount - 1u];
        sdsmLog.sourceTextureValid = nuri::isValid(sdsmTexture);
        if (!nuri::isValid(sdsmTexture)) {
          reuseCachedSdsmRangeOrFallback(ShadowSdsmStatus::Unavailable,
                                         "invalid_source_texture");
          return false;
        }

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
          return false;
        }

        sdsmLog.readbackError = false;
        std::array<float, 2u> rawDeviceDepths{};
        std::memcpy(rawDeviceDepths.data(), sdsmBytes.data(),
                    sizeof(rawDeviceDepths));
        consumeRawDeviceMinMax(rawDeviceDepths[0], rawDeviceDepths[1],
                               "invalid_raw_device_depths",
                               "invalid_raw_linear_depths");
        return shadowDebugFrameData_.sdsm.status == ShadowSdsmStatus::Active;
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
                sourceSelection.dimensions, minMaxPairs, memory_);
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
              if (!suppressGpuWarmupWarning &&
                  !sdsmState_.loggedGpuResultRingDiagnosticWarning_) {
                sdsmState_.loggedGpuResultRingDiagnosticWarning_ = true;
                const uint64_t requestedSourceFrameIndex =
                    frame.sharedResources.sceneDepthPyramidSourceFrameIndex
                        .value_or(std::numeric_limits<uint64_t>::max());
                NURI_LOG_WARNING(
                    "ShadowRenderer::updateShadowFrameData: GPU SDSM "
                    "histogram ring diagnostics frame=%llu "
                    "requestedSourceFrame=%llu slotCount=%zu reason=%s",
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
                        "ShadowRenderer::updateShadowFrameData: GPU SDSM "
                        "histogram ring slot=%zu bufferValid=0 "
                        "expectedFrame=%llu readOk=0 valid=0 sourceFrame=0 "
                        "validTiles=0 bucketCount=0 rawDeviceMinMax=(0.000000, "
                        "0.000000) histogramRange=(0.000000, 0.000000) "
                        "totalWeight=0.000000 clear=0.000000",
                        slotIndex,
                        static_cast<unsigned long long>(
                            expectedPublishedFrame ==
                                    std::numeric_limits<uint64_t>::max()
                                ? 0u
                                : expectedPublishedFrame));
                    continue;
                  }

                  SdsmGpuHistogramResult gpuResult{};
                  auto readResult = gpu_.readBuffer(
                      slot.buffer->handle(), 0u,
                      std::as_writable_bytes(
                          std::span<SdsmGpuHistogramResult>(&gpuResult, 1u)));
                  const bool readOk = !readResult.hasError();
                  NURI_LOG_WARNING(
                      "ShadowRenderer::updateShadowFrameData: GPU SDSM "
                      "histogram ring slot=%zu bufferValid=1 "
                      "expectedFrame=%llu readOk=%u valid=%u sourceFrame=%u "
                      "validTiles=%u bucketCount=%u rawDeviceMinMax=(%.6f, "
                      "%.6f) histogramRange=(%.6f, %.6f) totalWeight=%.6f "
                      "clear=%.6f",
                      slotIndex,
                      static_cast<unsigned long long>(
                          expectedPublishedFrame ==
                                  std::numeric_limits<uint64_t>::max()
                              ? 0u
                              : expectedPublishedFrame),
                      readOk ? 1u : 0u, gpuResult.metadata.y,
                      gpuResult.metadata.x, gpuResult.metadata.z,
                      gpuResult.metadata.w,
                      gpuResult.rawDeviceMinMaxLinearMinMax.x,
                      gpuResult.rawDeviceMinMaxLinearMinMax.y,
                      gpuResult.histogramRangeWeightClear.x,
                      gpuResult.histogramRangeWeightClear.y,
                      gpuResult.histogramRangeWeightClear.z,
                      gpuResult.histogramRangeWeightClear.w);
                }
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
            (void)consumeCpuMinMaxSource();
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
          (void)consumeCpuMinMaxSource();
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
  {
    NURI_PROFILER_ZONE("ShadowRenderer.update.fit_cascades",
                       NURI_PROFILER_COLOR_CMD_COPY);
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
              hasAnimatedGeometryOverrides, casterBounds.min_,
              casterBounds.max_);
        } else {
          fit = shadow_detail::
              fitDirectionalShadowCascadeSliceWithCasterDepthBounds(
                  frame.camera, effectiveSplitDepths[cascadeIndex],
                  effectiveSplitDepths[cascadeIndex + 1u], casterLightView,
                  shadowMapSize,
                  !hasAnimatedGeometryOverrides && hasCasterLightDepthBounds,
                  casterLightDepthBounds, settings.stabilizeCascades);
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
    NURI_PROFILER_ZONE_END();
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
  const float fadeEnd =
      shadowDebugFrameData_.cascades[cascadeCount - 1u].splitFar;
  const float fadeNear = shadowDebugFrameData_.cascades[0].splitNear;
  const float fadeWidth =
      std::max(fadeEnd - fadeNear, 0.0f) * settings.maxDistanceFadeFraction;
  const float fadeStart = std::max(fadeNear, fadeEnd - fadeWidth);
  shadowFrameCpuData_.fadeParams =
      glm::vec4(fadeStart, fadeEnd, settings.cascadeBlendFraction,
                settings.pcssLightRadiusScale);
  shadowDebugFrameData_.maxDistanceFadeStart = fadeStart;
  shadowDebugFrameData_.maxDistanceFadeEnd = fadeEnd;

  shadowDebugFrameData_.selectedShadowLightId = selectedLightId;
  float minCascadeTexelWorldSize = std::numeric_limits<float>::max();
  float maxCascadeTexelWorldSize = 0.0f;
  float totalCascadeTexelWorldSize = 0.0f;
  uint32_t validCascadeTexelWorldSizeCount = 0u;
  for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
       ++cascadeIndex) {
    const float texelWorldSize =
        shadowDebugFrameData_.cascades[cascadeIndex].texelWorldSize;
    if (!std::isfinite(texelWorldSize) || texelWorldSize <= 0.0f) {
      continue;
    }
    minCascadeTexelWorldSize =
        std::min(minCascadeTexelWorldSize, texelWorldSize);
    maxCascadeTexelWorldSize =
        std::max(maxCascadeTexelWorldSize, texelWorldSize);
    totalCascadeTexelWorldSize += texelWorldSize;
    ++validCascadeTexelWorldSizeCount;
  }
  if (validCascadeTexelWorldSizeCount == 0u) {
    minCascadeTexelWorldSize = 0.0f;
  }
  frame.metrics.shadow.cascadeCount = cascadeCount;
  frame.metrics.shadow.shadowMapSize = shadowMapSize;
  frame.metrics.shadow.cascadeTextureBytes =
      static_cast<uint64_t>(cascadeCount) * shadowMapSize * shadowMapSize *
      shadowDepthTextureBytesPerPixel(settings.depthFormat);
  frame.metrics.shadow.minCascadeTexelWorldSize = minCascadeTexelWorldSize;
  frame.metrics.shadow.maxCascadeTexelWorldSize = maxCascadeTexelWorldSize;
  frame.metrics.shadow.averageCascadeTexelWorldSize =
      validCascadeTexelWorldSizeCount > 0u
          ? totalCascadeTexelWorldSize /
                static_cast<float>(validCascadeTexelWorldSizeCount)
          : 0.0f;
  frame.metrics.shadow.farCascadeTexelWorldSize =
      cascadeCount > 0u
          ? shadowDebugFrameData_.cascades[cascadeCount - 1u].texelWorldSize
          : 0.0f;

  frame.sharedResources.shadowDebugFrameData = shadowDebugFrameData_;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
ShadowRenderer::buildShadowDraws(RenderFrameContext &frame, uint32_t frameSlot,
                                 const ForwardSceneGpuData &sceneGpu) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_DRAW);
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
        "staticBatch=(templates=%u full=%u graph=%u remap=%u) "
        "grid=(queries=%u fallback=%u cells=%u candidates=%u) "
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
        frame.metrics.shadow.staticBatchTemplateCount,
        frame.metrics.shadow.staticBatchFullEmitCount,
        frame.metrics.shadow.shadowBatchEntryCount,
        frame.metrics.shadow.shadowInstanceRemapCount,
        frame.metrics.shadow.staticLightGridQueryCount,
        frame.metrics.shadow.staticLightGridFallbackScanCount,
        frame.metrics.shadow.staticLightGridQueryCellCount,
        frame.metrics.shadow.staticLightGridCandidateCount,
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
  const Format targetDepthFormat =
      sanitizeShadowDepthFormat(shadowSettings.depthFormat);
  if (shadowDepthPipelineFormat_ != targetDepthFormat) {
    auto pipelineResult = createPipelines(targetDepthFormat);
    if (pipelineResult.hasError()) {
      return pipelineResult;
    }
  }
  bool shadowMeshletActive = false;
  if (gpu_.supportsFeature(GPUFeature::Meshlets)) {
    auto meshletPipelineResult =
        ensureShadowMeshletPipelineState(targetDepthFormat);
    if (!meshletPipelineResult.hasError()) {
      shadowMeshletActive = true;
    } else if (!loggedShadowMeshletFallbackWarning_) {
      loggedShadowMeshletFallbackWarning_ = true;
      NURI_LOG_WARNING(
          "ShadowRenderer::buildShadowDraws: meshlet shadows disabled: %s",
          meshletPipelineResult.error().c_str());
    }
  }

  const std::span<const Renderable> renderables = frame.scene->renderables();
  const AnimationSceneFrameData *animationSceneData =
      resolveAnimationSceneFrameData(frame);
  const uint64_t transformVersion = frame.scene->transformVersion();
  if (cachedTransformVersion_ != transformVersion ||
      instanceMatrices_.size() != renderables.size()) {
    instanceMatrices_.clear();
    instanceMatrices_.reserve(renderables.size());
    for (uint32_t i = 0u; i < static_cast<uint32_t>(renderables.size()); ++i) {
      instanceMatrices_.push_back(makeInstanceData(renderables[i].modelMatrix));
    }
    cachedTransformVersion_ = transformVersion;
    std::fill(instanceDataRingUploadVersions_.begin(),
              instanceDataRingUploadVersions_.end(),
              std::numeric_limits<uint64_t>::max());
  }

  if (instanceMatrices_.empty()) {
    emitShadowDiagnostics();
    return Result<bool, std::string>::makeResult(true);
  }

  auto matricesResult = ensureInstanceMatricesRingCapacity(std::max(
      instanceMatrices_.size() * sizeof(InstanceData), sizeof(InstanceData)));
  if (matricesResult.hasError()) {
    return matricesResult;
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
    instanceDataRingUploadVersions_[frameSlot] = cachedTransformVersion_;
  }

  const BufferHandle instanceMatricesBuffer =
      animationSceneData != nullptr
          ? animationSceneData->instanceMatricesBuffer
          : instanceMatricesRing_[frameSlot].buffer->handle();
  const uint64_t instanceMatricesAddress =
      animationSceneData != nullptr
          ? animationSceneData->instanceMatricesAddress
          : gpu_.getBufferDeviceAddress(instanceMatricesBuffer);
  if (sceneGpu.shadowFrameBufferAddress == 0u) {
    emitShadowDiagnostics();
    return Result<bool, std::string>::makeResult(true);
  }
  if (sceneGpu.frameDataAddress == 0u || instanceMatricesAddress == 0u) {
    emitShadowDiagnostics();
    return Result<bool, std::string>::makeError(
        "ShadowRenderer::buildShadowDraws: invalid GPU buffer address");
  }

  BufferHandle shadowMeshletCounterBuffer{};
  uint64_t shadowMeshletCounterBufferAddress = 0u;
  uint32_t shadowMeshletCounterFlags = 0u;
  if (shadowMeshletActive) {
    auto counterResult = ensureShadowMeshletCounterRingCapacity(
        sizeof(VisibilityCounterGpuData));
    if (counterResult.hasError()) {
      return counterResult;
    }
    if (frameSlot < shadowMeshletCounterRing_.size() &&
        shadowMeshletCounterRing_[frameSlot].buffer &&
        shadowMeshletCounterRing_[frameSlot].buffer->valid()) {
      shadowMeshletCounterBuffer =
          shadowMeshletCounterRing_[frameSlot].buffer->handle();
      shadowMeshletCounterClear_.clear();
      shadowMeshletCounterClear_.push_back(VisibilityCounterGpuData{});
      auto clearCounterResult = gpu_.updateBuffer(
          shadowMeshletCounterBuffer,
          std::as_bytes(std::span<const VisibilityCounterGpuData>(
              shadowMeshletCounterClear_.data(),
              shadowMeshletCounterClear_.size())),
          0u);
      if (clearCounterResult.hasError()) {
        return clearCounterResult;
      }
      if (frameSlot < shadowMeshletCounterRingPublishedFrames_.size()) {
        shadowMeshletCounterRingPublishedFrames_[frameSlot] = frame.frameIndex;
      }
      shadowMeshletCounterBufferAddress =
          gpu_.getBufferDeviceAddress(shadowMeshletCounterBuffer);
      if (shadowMeshletCounterBufferAddress != 0u) {
        shadowMeshletCounterFlags |= kShadowMeshletCounterFlagEnabled;
      }
    }
  }

  const uint32_t cascadeCount =
      std::clamp(shadowFrameCpuData_.flagsCascadeCountLightIndex.y, 1u,
                 activeCascadeCount_);
  for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
       ++cascadeIndex) {
    cascadePushConstants_[cascadeIndex].clear();
    cascadeDrawItems_[cascadeIndex].clear();
    cascadeMeshletPushConstants_[cascadeIndex].clear();
    cascadeMeshDispatchItems_[cascadeIndex].clear();
    cascadeMeshDispatchDependencyBuffers_[cascadeIndex].clear();
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
                                      (animationSceneData != nullptr
                                           ? staticShadowTemplateIndices_.size()
                                           : 0u));
  preResolvedDrawBuffers_.reserve(staticShadowCasterDrawBuffers_.size() +
                                  frameDynamicTemplateIndices.capacity() * 2u);
  preResolvedDrawBuffers_.insert(preResolvedDrawBuffers_.end(),
                                 staticShadowCasterDrawBuffers_.begin(),
                                 staticShadowCasterDrawBuffers_.end());
  frameDynamicTemplateIndices.insert(frameDynamicTemplateIndices.end(),
                                     dynamicShadowTemplateIndices_.begin(),
                                     dynamicShadowTemplateIndices_.end());
  std::pmr::vector<uint8_t> staticTemplatesUsingDynamicPath(memory_);
  uint32_t overriddenStaticTemplateCount = 0u;
  if (animationSceneData != nullptr) {
    staticTemplatesUsingDynamicPath.resize(meshDrawTemplates_.size(),
                                           uint8_t{0});
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
      ++overriddenStaticTemplateCount;
      frameDynamicTemplateIndices.push_back(templateIndex);
    }
  }

  const size_t maxShadowInstanceRemapCount =
      std::max<size_t>(1u, static_cast<size_t>(cascadeCount) *
                               (staticShadowCasterCache_.size() +
                                frameDynamicTemplateIndices.size()));
  auto remapResult = ensureInstanceRemapRingCapacity(std::max(
      maxShadowInstanceRemapCount * sizeof(uint32_t), sizeof(uint32_t)));
  if (remapResult.hasError()) {
    return remapResult;
  }
  const BufferHandle instanceRemapBuffer =
      instanceRemapRing_[frameSlot].buffer->handle();
  const uint64_t instanceRemapAddress =
      gpu_.getBufferDeviceAddress(instanceRemapBuffer);
  if (instanceRemapAddress == 0u) {
    emitShadowDiagnostics();
    return Result<bool, std::string>::makeError(
        "ShadowRenderer::buildShadowDraws: invalid instance remap buffer "
        "address");
  }

  instanceRemap_.clear();
  instanceRemap_.reserve(maxShadowInstanceRemapCount);
  const size_t staticBatchEntryCount =
      static_cast<size_t>(cascadeCount) * staticShadowBatchTemplates_.size();
  const size_t dynamicBatchEstimate =
      static_cast<size_t>(cascadeCount) * frameDynamicTemplateIndices.size();
  ScopedScratch batchScratch(batchBuildScratchArena_);
  std::pmr::memory_resource *batchMemory = batchScratch.resource();
  PmrHashMap<ShadowBatchKey, uint32_t, ShadowBatchKeyHash> shadowBatchLookup(
      batchMemory);
  shadowBatchLookup.reserve(dynamicBatchEstimate + 1u);
  std::pmr::vector<ShadowBatchEntry> shadowBatchEntries(batchMemory);
  shadowBatchEntries.reserve(staticBatchEntryCount + dynamicBatchEstimate);
  std::pmr::vector<uint32_t> shadowBatchInstanceIndices(batchMemory);
  shadowBatchInstanceIndices.reserve(
      std::min<size_t>(maxShadowInstanceRemapCount, 4096u));
  std::array<uint32_t, kMaxShadowCascades> cascadeStaticCasterCounts{};
  std::array<uint32_t, kMaxShadowCascades> cascadeDynamicCasterCounts{};
  std::array<uint32_t, kMaxShadowCascades> cascadeStaticBatchFullEmitCounts{};
  std::array<uint32_t, kMaxShadowCascades>
      cascadeStaticLightGridQueryCellCounts{};
  std::array<uint32_t, kMaxShadowCascades>
      cascadeStaticLightGridCandidateCounts{};
  std::array<uint32_t, kMaxShadowCascades>
      cascadeStaticLightGridFallbackScanCounts{};
  uint32_t staticLightGridQueryCount = 0u;

  const uint32_t renderableCount = saturateToU32(renderables.size());
  const auto selectShadowPipeline =
      [&](bool doubleSided, bool alphaMasked) -> RenderPipelineHandle {
    if (alphaMasked) {
      return doubleSided && nuri::isValid(shadowAlphaDoubleSidedPipelineHandle_)
                 ? shadowAlphaDoubleSidedPipelineHandle_
                 : shadowAlphaPipelineHandle_;
    }
    return doubleSided && nuri::isValid(shadowDoubleSidedPipelineHandle_)
               ? shadowDoubleSidedPipelineHandle_
               : shadowPipelineHandle_;
  };
  const auto selectShadowMeshletPipeline =
      [&](bool doubleSided, bool alphaMasked) -> MeshletPipelineHandle {
    if (alphaMasked) {
      return doubleSided &&
                     nuri::isValid(shadowMeshletAlphaDoubleSidedPipelineHandle_)
                 ? shadowMeshletAlphaDoubleSidedPipelineHandle_
                 : shadowMeshletAlphaPipelineHandle_;
    }
    return doubleSided && nuri::isValid(shadowMeshletDoubleSidedPipelineHandle_)
               ? shadowMeshletDoubleSidedPipelineHandle_
               : shadowMeshletPipelineHandle_;
  };
  const auto appendShadowDraw =
      [&](uint32_t cascadeIndex, BufferHandle vertexBuffer,
          BufferHandle indexBuffer, uint64_t indexBufferOffset,
          IndexFormat indexFormat, uint32_t indexCount, uint32_t firstIndex,
          uint32_t firstInstance, uint64_t vertexBufferAddress,
          BufferHandle vertexDecodeBuffer, uint64_t vertexDecodeBufferAddress,
          uint32_t vertexDecodeIndex, uint32_t packedVertexFormat,
          uint32_t materialIndex, bool doubleSided, bool alphaMasked,
          const Model::ModelMeshletGpuView *meshletView, uint32_t submeshIndex,
          uint32_t meshletMaxCount, uint32_t meshletCount,
          uint32_t vertexOffset, bool dynamicCaster,
          bool enableMeshletCascadeCulling, bool buffersAlreadyPreResolved) {
        const bool useMeshlets =
            shadowMeshletActive &&
            canUseShadowMeshlets(meshletView, submeshIndex, meshletMaxCount,
                                 meshletCount);
        if (!useMeshlets && !buffersAlreadyPreResolved) {
          appendUniqueBufferHandle(preResolvedDrawBuffers_, vertexBuffer);
          appendUniqueBufferHandle(preResolvedDrawBuffers_, indexBuffer);
        }

        const ShadowBatchKey key{
            .cascadeIndex = cascadeIndex,
            .dynamicCaster = dynamicCaster,
            .useMeshlets = useMeshlets,
            .pipeline = selectShadowPipeline(doubleSided, alphaMasked),
            .meshletPipeline = useMeshlets ? selectShadowMeshletPipeline(
                                                 doubleSided, alphaMasked)
                                           : MeshletPipelineHandle{},
            .vertexBuffer = vertexBuffer,
            .vertexDecodeBuffer = vertexDecodeBuffer,
            .indexBuffer = indexBuffer,
            .indexBufferOffset = indexBufferOffset,
            .indexFormat = indexFormat,
            .indexCount = indexCount,
            .firstIndex = firstIndex,
            .vertexBufferAddress = vertexBufferAddress,
            .vertexDecodeBufferAddress = vertexDecodeBufferAddress,
            .vertexDecodeIndex = vertexDecodeIndex,
            .packedVertexFormat = packedVertexFormat,
            .materialIndex = materialIndex,
            .meshletBuffer = meshletView != nullptr ? meshletView->meshletBuffer
                                                    : BufferHandle{},
            .meshletVertexIndexBuffer =
                meshletView != nullptr ? meshletView->meshletVertexIndexBuffer
                                       : BufferHandle{},
            .meshletPrimitiveIndexBuffer =
                meshletView != nullptr
                    ? meshletView->meshletPrimitiveIndexBuffer
                    : BufferHandle{},
            .meshletLodRangeBuffer = meshletView != nullptr
                                         ? meshletView->lodRangeBuffer
                                         : BufferHandle{},
            .submeshIndex = submeshIndex,
            .meshletMaxCount = meshletMaxCount,
            .meshletCount = meshletCount,
            .vertexOffset = vertexOffset,
            .enableMeshletCascadeCulling =
                useMeshlets && enableMeshletCascadeCulling,
        };
        auto it = shadowBatchLookup.find(key);
        if (it == shadowBatchLookup.end()) {
          const uint32_t batchIndex =
              static_cast<uint32_t>(shadowBatchEntries.size());
          shadowBatchEntries.emplace_back();
          shadowBatchEntries.back().key = key;
          auto [insertedIt, _] = shadowBatchLookup.emplace(key, batchIndex);
          it = insertedIt;
        }
        ShadowBatchEntry &batch = shadowBatchEntries[it->second];
        batch.appendInstance(firstInstance, shadowBatchInstanceIndices);
        if (dynamicCaster) {
          ++cascadeDynamicCasterCounts[cascadeIndex];
        } else {
          ++cascadeStaticCasterCounts[cascadeIndex];
        }
      };

  const auto makeStaticShadowBatchKey =
      [](const StaticShadowBatchTemplate &batchTemplate,
         uint32_t cascadeIndex) {
        const Model::ModelMeshletGpuView *meshletView =
            batchTemplate.meshletView;
        return ShadowBatchKey{
            .cascadeIndex = cascadeIndex,
            .dynamicCaster = false,
            .useMeshlets = batchTemplate.useMeshlets,
            .pipeline = batchTemplate.pipeline,
            .meshletPipeline = batchTemplate.meshletPipeline,
            .vertexBuffer = batchTemplate.vertexBuffer,
            .vertexDecodeBuffer = batchTemplate.vertexDecodeBuffer,
            .indexBuffer = batchTemplate.indexBuffer,
            .indexBufferOffset = batchTemplate.indexBufferOffset,
            .indexFormat = batchTemplate.indexFormat,
            .indexCount = batchTemplate.indexCount,
            .firstIndex = batchTemplate.firstIndex,
            .vertexBufferAddress = batchTemplate.vertexBufferAddress,
            .vertexDecodeBufferAddress =
                batchTemplate.vertexDecodeBufferAddress,
            .vertexDecodeIndex = batchTemplate.vertexDecodeIndex,
            .packedVertexFormat = batchTemplate.packedVertexFormat,
            .materialIndex = batchTemplate.materialIndex,
            .meshletBuffer = meshletView != nullptr ? meshletView->meshletBuffer
                                                    : BufferHandle{},
            .meshletVertexIndexBuffer =
                meshletView != nullptr ? meshletView->meshletVertexIndexBuffer
                                       : BufferHandle{},
            .meshletPrimitiveIndexBuffer =
                meshletView != nullptr
                    ? meshletView->meshletPrimitiveIndexBuffer
                    : BufferHandle{},
            .meshletLodRangeBuffer = meshletView != nullptr
                                         ? meshletView->lodRangeBuffer
                                         : BufferHandle{},
            .submeshIndex = batchTemplate.submeshIndex,
            .meshletMaxCount = batchTemplate.meshletMaxCount,
            .meshletCount = batchTemplate.meshletCount,
            .vertexOffset = batchTemplate.vertexOffset,
            .enableMeshletCascadeCulling =
                batchTemplate.enableMeshletCascadeCulling,
        };
      };

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
  {
    NURI_PROFILER_ZONE("ShadowRenderer.dynamic_guard_scan",
                       NURI_PROFILER_COLOR_CMD_COPY);
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
    NURI_PROFILER_ZONE_END();
  }

  {
    NURI_PROFILER_ZONE("ShadowRenderer.write_cascade_fits",
                       NURI_PROFILER_COLOR_CMD_COPY);
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
      state.lightViewProjSignature =
          makeStaticOnlyCascadeLightViewProjSignature(
              shadowFrameCpuData_.cascades[cascadeIndex]);
      state.biasSignature = makeStaticOnlyCascadeBiasSignature(
          settings.shadow.constantBias, settings.shadow.slopeBias);
      state.casterSignature = kFnvOffsetBasis64;
      state.rasterSignature = makeStaticOnlyCascadeRasterSignatureSeed(
          shadowFrameCpuData_.cascades[cascadeIndex],
          settings.shadow.constantBias, settings.shadow.slopeBias);
    }
    NURI_PROFILER_ZONE_END();
  }

  std::array<uint8_t, kMaxShadowCascades> skipStaticCasterBuildForCascade{};
  const bool canUseFastStaticOnlyReusePath =
      staticCacheReused && overriddenStaticTemplateCount == 0u;
  if (canUseFastStaticOnlyReusePath) {
    for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
         ++cascadeIndex) {
      StaticOnlyCascadeReuseState &currentState =
          currentStaticOnlyCascadeStates[cascadeIndex];
      const StaticOnlyCascadeReuseState &previousState =
          reusableStaticOnlyCascadeStates_[cascadeIndex];
      const bool staticOnlyCandidate =
          cascadeHasGuardedDynamicDraw[cascadeIndex] == 0u;
      const bool previousValid = reusableStaticOnlyCascadeValid_[cascadeIndex];
      const bool biasChanged = previousValid && previousState.biasSignature !=
                                                    currentState.biasSignature;
      const bool cachedRenderedFitContainsCurrent =
          previousValid &&
          staticOnlyRenderedFitContainsCurrent(
              previousState.renderedFit, currentRawShadowFits_[cascadeIndex]);
      const bool needsAdaptiveRefresh =
          cachedRenderedFitContainsCurrent &&
          staticOnlyNeedsAdaptiveRefresh(previousState.renderedFit,
                                         currentRawShadowFits_[cascadeIndex],
                                         previousState.rawFit, previousValid,
                                         settings.shadow.shadowMapSize);
      if (!staticOnlyCandidate || !previousValid || biasChanged ||
          !cachedRenderedFitContainsCurrent || needsAdaptiveRefresh) {
        continue;
      }

      skipStaticCasterBuildForCascade[cascadeIndex] = 1u;
      currentState.casterSignature = previousState.casterSignature;
      currentState.rasterSignature = previousState.rasterSignature;
      currentState.staticDrawCount = previousState.staticDrawCount;
      currentState.dynamicDrawCount = 0u;
    }
  }

  if (enableCascadeCasterCulling && !staticShadowCasterCache_.empty()) {
    const glm::mat4 staticCasterLightView =
        shadowDebugFrameData_.cascades[0].lightView;
    if (!hasStaticShadowCasterLightSpaceBounds_ ||
        staticShadowCasterLightSpaceBounds_.size() !=
            staticShadowCasterCache_.size() ||
        staticShadowBatchLightSpaceBounds_.size() !=
            staticShadowBatchTemplates_.size() ||
        !rawBytesEqual(staticShadowCasterLightSpaceBoundsView_,
                       staticCasterLightView)) {
      NURI_PROFILER_ZONE("ShadowRenderer.static_caster_light_bounds",
                         NURI_PROFILER_COLOR_CMD_COPY);
      staticShadowCasterLightSpaceBounds_.resize(
          staticShadowCasterCache_.size());
      staticShadowBatchLightSpaceBounds_.resize(
          staticShadowBatchTemplates_.size());
      for (StaticShadowCasterLightSpaceBounds &bounds :
           staticShadowBatchLightSpaceBounds_) {
        bounds.min = glm::vec3(std::numeric_limits<float>::max());
        bounds.max = glm::vec3(std::numeric_limits<float>::lowest());
      }
      glm::vec3 staticLightBoundsMin(std::numeric_limits<float>::max());
      glm::vec3 staticLightBoundsMax(std::numeric_limits<float>::lowest());
      for (size_t entryIndex = 0u; entryIndex < staticShadowCasterCache_.size();
           ++entryIndex) {
        const StaticShadowCasterCacheEntry &entry =
            staticShadowCasterCache_[entryIndex];
        glm::vec3 lightMin(std::numeric_limits<float>::max());
        glm::vec3 lightMax(std::numeric_limits<float>::lowest());
        for (const glm::vec3 corner : entry.casterWorldCorners) {
          const glm::vec3 lightCorner =
              glm::vec3(staticCasterLightView * glm::vec4(corner, 1.0f));
          lightMin = glm::min(lightMin, lightCorner);
          lightMax = glm::max(lightMax, lightCorner);
        }
        staticShadowCasterLightSpaceBounds_[entryIndex] = {
            .min = lightMin,
            .max = lightMax,
        };
        if (entry.batchIndex < staticShadowBatchLightSpaceBounds_.size()) {
          StaticShadowCasterLightSpaceBounds &batchBounds =
              staticShadowBatchLightSpaceBounds_[entry.batchIndex];
          batchBounds.min = glm::min(batchBounds.min, lightMin);
          batchBounds.max = glm::max(batchBounds.max, lightMax);
        }
        staticLightBoundsMin =
            glm::min(staticLightBoundsMin, glm::min(lightMin, lightMax));
        staticLightBoundsMax =
            glm::max(staticLightBoundsMax, glm::max(lightMin, lightMax));
      }
      staticShadowCasterLightSpaceBoundsView_ = staticCasterLightView;
      staticShadowCasterLightSpaceBoundsMin_ = staticLightBoundsMin;
      staticShadowCasterLightSpaceBoundsMax_ = staticLightBoundsMax;
      hasStaticShadowCasterLightSpaceBounds_ = true;
      staticShadowCasterLightGridCells_.clear();
      staticShadowCasterLightGridEntries_.clear();
      staticShadowCasterLargeLightGridEntries_.clear();
      staticShadowBatchLightGridCells_.clear();
      staticShadowBatchLightGridEntries_.clear();
      staticShadowBatchLargeLightGridEntries_.clear();
      staticShadowCasterLightGrid_ = {};
      staticShadowBatchLightGrid_ = {};
      NURI_PROFILER_ZONE_END();
    }
    if ((!staticShadowCasterLightGrid_.valid ||
         !staticShadowBatchLightGrid_.valid) &&
        hasStaticShadowCasterLightSpaceBounds_) {
      NURI_PROFILER_ZONE("ShadowRenderer.static_caster_light_grid",
                         NURI_PROFILER_COLOR_CMD_COPY);
      glm::vec3 gridMin = staticShadowCasterLightSpaceBoundsMin_;
      glm::vec3 gridMax = staticShadowCasterLightSpaceBoundsMax_;

      const glm::vec3 gridExtent =
          glm::max(gridMax - gridMin, glm::vec3(0.01f));
      gridMax = gridMin + gridExtent;
      const glm::uvec3 gridDimensions(kStaticCasterLightGridResolution,
                                      kStaticCasterLightGridResolution,
                                      kStaticCasterLightGridDepthResolution);
      const uint32_t gridCellCount =
          gridDimensions.x * gridDimensions.y * gridDimensions.z;
      std::pmr::vector<uint32_t> cellCounts(batchMemory);
      cellCounts.resize(gridCellCount, 0u);
      staticShadowCasterLargeLightGridEntries_.clear();
      std::pmr::vector<uint32_t> batchCellCounts(batchMemory);
      batchCellCounts.resize(gridCellCount, 0u);
      staticShadowBatchLargeLightGridEntries_.clear();

      const glm::vec3 invCellSize = glm::vec3(gridDimensions) / gridExtent;
      struct StaticCasterLightGridRange {
        uint32_t minX = 0u;
        uint32_t minY = 0u;
        uint32_t minZ = 0u;
        uint32_t maxX = 0u;
        uint32_t maxY = 0u;
        uint32_t maxZ = 0u;
      };
      const auto gridCellIndex = [&](uint32_t x, uint32_t y, uint32_t z) {
        return (z * gridDimensions.y + y) * gridDimensions.x + x;
      };
      const auto gridCellRange =
          [&](const StaticShadowCasterLightSpaceBounds &bounds) {
            const glm::vec3 boundsMin = glm::min(bounds.min, bounds.max);
            const glm::vec3 boundsMax = glm::max(bounds.min, bounds.max);
            const glm::ivec3 minCell = glm::clamp(
                glm::ivec3(glm::floor((boundsMin - gridMin) * invCellSize)),
                glm::ivec3(0), glm::ivec3(gridDimensions) - glm::ivec3(1));
            const glm::ivec3 maxCell = glm::clamp(
                glm::ivec3(glm::floor((boundsMax - gridMin) * invCellSize)),
                glm::ivec3(0), glm::ivec3(gridDimensions) - glm::ivec3(1));
            return StaticCasterLightGridRange{
                .minX = static_cast<uint32_t>(std::min(minCell.x, maxCell.x)),
                .minY = static_cast<uint32_t>(std::min(minCell.y, maxCell.y)),
                .minZ = static_cast<uint32_t>(std::min(minCell.z, maxCell.z)),
                .maxX = static_cast<uint32_t>(std::max(minCell.x, maxCell.x)),
                .maxY = static_cast<uint32_t>(std::max(minCell.y, maxCell.y)),
                .maxZ = static_cast<uint32_t>(std::max(minCell.z, maxCell.z)),
            };
          };

      for (uint32_t entryIndex = 0u;
           entryIndex < staticShadowCasterLightSpaceBounds_.size();
           ++entryIndex) {
        const StaticCasterLightGridRange range =
            gridCellRange(staticShadowCasterLightSpaceBounds_[entryIndex]);
        const uint32_t coveredCells = (range.maxX - range.minX + 1u) *
                                      (range.maxY - range.minY + 1u) *
                                      (range.maxZ - range.minZ + 1u);
        if (coveredCells > kStaticCasterLightGridMaxCellsPerCaster) {
          staticShadowCasterLargeLightGridEntries_.push_back(entryIndex);
          continue;
        }
        for (uint32_t z = range.minZ; z <= range.maxZ; ++z) {
          for (uint32_t y = range.minY; y <= range.maxY; ++y) {
            for (uint32_t x = range.minX; x <= range.maxX; ++x) {
              ++cellCounts[gridCellIndex(x, y, z)];
            }
          }
        }
      }

      for (uint32_t batchIndex = 0u;
           batchIndex < staticShadowBatchLightSpaceBounds_.size();
           ++batchIndex) {
        if (batchIndex >= staticShadowBatchTemplates_.size() ||
            staticShadowBatchTemplates_[batchIndex].instanceCount <
                kStaticBatchOverlapEmitMinInstances) {
          continue;
        }
        const StaticCasterLightGridRange range =
            gridCellRange(staticShadowBatchLightSpaceBounds_[batchIndex]);
        const uint32_t coveredCells = (range.maxX - range.minX + 1u) *
                                      (range.maxY - range.minY + 1u) *
                                      (range.maxZ - range.minZ + 1u);
        if (coveredCells > kStaticCasterLightGridMaxCellsPerCaster) {
          staticShadowBatchLargeLightGridEntries_.push_back(batchIndex);
          continue;
        }
        for (uint32_t z = range.minZ; z <= range.maxZ; ++z) {
          for (uint32_t y = range.minY; y <= range.maxY; ++y) {
            for (uint32_t x = range.minX; x <= range.maxX; ++x) {
              ++batchCellCounts[gridCellIndex(x, y, z)];
            }
          }
        }
      }

      staticShadowCasterLightGridCells_.resize(gridCellCount);
      uint32_t runningEntryCount = 0u;
      for (uint32_t cellIndex = 0u; cellIndex < gridCellCount; ++cellIndex) {
        staticShadowCasterLightGridCells_[cellIndex] = {
            .firstEntry = runningEntryCount,
            .entryCount = cellCounts[cellIndex],
        };
        runningEntryCount += cellCounts[cellIndex];
      }

      staticShadowCasterLightGridEntries_.resize(runningEntryCount);
      std::pmr::vector<uint32_t> cellWriteOffsets(batchMemory);
      cellWriteOffsets.resize(gridCellCount, 0u);
      for (uint32_t cellIndex = 0u; cellIndex < gridCellCount; ++cellIndex) {
        cellWriteOffsets[cellIndex] =
            staticShadowCasterLightGridCells_[cellIndex].firstEntry;
      }
      for (uint32_t entryIndex = 0u;
           entryIndex < staticShadowCasterLightSpaceBounds_.size();
           ++entryIndex) {
        const StaticCasterLightGridRange range =
            gridCellRange(staticShadowCasterLightSpaceBounds_[entryIndex]);
        const uint32_t coveredCells = (range.maxX - range.minX + 1u) *
                                      (range.maxY - range.minY + 1u) *
                                      (range.maxZ - range.minZ + 1u);
        if (coveredCells > kStaticCasterLightGridMaxCellsPerCaster) {
          continue;
        }
        for (uint32_t z = range.minZ; z <= range.maxZ; ++z) {
          for (uint32_t y = range.minY; y <= range.maxY; ++y) {
            for (uint32_t x = range.minX; x <= range.maxX; ++x) {
              const uint32_t cellIndex = gridCellIndex(x, y, z);
              const uint32_t writeOffset = cellWriteOffsets[cellIndex]++;
              staticShadowCasterLightGridEntries_[writeOffset] = entryIndex;
            }
          }
        }
      }

      staticShadowBatchLightGridCells_.resize(gridCellCount);
      uint32_t runningBatchEntryCount = 0u;
      for (uint32_t cellIndex = 0u; cellIndex < gridCellCount; ++cellIndex) {
        staticShadowBatchLightGridCells_[cellIndex] = {
            .firstEntry = runningBatchEntryCount,
            .entryCount = batchCellCounts[cellIndex],
        };
        runningBatchEntryCount += batchCellCounts[cellIndex];
      }

      staticShadowBatchLightGridEntries_.resize(runningBatchEntryCount);
      std::pmr::vector<uint32_t> batchCellWriteOffsets(batchMemory);
      batchCellWriteOffsets.resize(gridCellCount, 0u);
      for (uint32_t cellIndex = 0u; cellIndex < gridCellCount; ++cellIndex) {
        batchCellWriteOffsets[cellIndex] =
            staticShadowBatchLightGridCells_[cellIndex].firstEntry;
      }
      for (uint32_t batchIndex = 0u;
           batchIndex < staticShadowBatchLightSpaceBounds_.size();
           ++batchIndex) {
        if (batchIndex >= staticShadowBatchTemplates_.size() ||
            staticShadowBatchTemplates_[batchIndex].instanceCount <
                kStaticBatchOverlapEmitMinInstances) {
          continue;
        }
        const StaticCasterLightGridRange range =
            gridCellRange(staticShadowBatchLightSpaceBounds_[batchIndex]);
        const uint32_t coveredCells = (range.maxX - range.minX + 1u) *
                                      (range.maxY - range.minY + 1u) *
                                      (range.maxZ - range.minZ + 1u);
        if (coveredCells > kStaticCasterLightGridMaxCellsPerCaster) {
          continue;
        }
        for (uint32_t z = range.minZ; z <= range.maxZ; ++z) {
          for (uint32_t y = range.minY; y <= range.maxY; ++y) {
            for (uint32_t x = range.minX; x <= range.maxX; ++x) {
              const uint32_t cellIndex = gridCellIndex(x, y, z);
              const uint32_t writeOffset = batchCellWriteOffsets[cellIndex]++;
              staticShadowBatchLightGridEntries_[writeOffset] = batchIndex;
            }
          }
        }
      }

      staticShadowCasterLightGrid_ = {
          .min = gridMin,
          .max = gridMax,
          .invCellSize = invCellSize,
          .dimensions = gridDimensions,
          .valid = true,
      };
      staticShadowBatchLightGrid_ = staticShadowCasterLightGrid_;
      NURI_PROFILER_ZONE_END();
    }
  }

  std::pmr::vector<uint32_t> staticBatchEntryIndices(batchMemory);
  staticBatchEntryIndices.resize(staticBatchEntryCount,
                                 std::numeric_limits<uint32_t>::max());
  std::pmr::vector<uint8_t> fullyEmittedStaticBatchEntries(batchMemory);
  fullyEmittedStaticBatchEntries.resize(staticBatchEntryCount, uint8_t{0});
  std::pmr::vector<uint32_t> staticBatchLightGridQueryEntries(batchMemory);
  std::pmr::vector<uint32_t> staticBatchLightGridQueryMarks(batchMemory);
  uint32_t staticBatchLightGridQueryMarker = 1u;
  const auto staticBatchEntrySlot = [&](uint32_t cascadeIndex,
                                        uint32_t staticBatchIndex) {
    return static_cast<size_t>(cascadeIndex) *
               staticShadowBatchTemplates_.size() +
           staticBatchIndex;
  };
  const auto resolveStaticBatchEntry =
      [&](uint32_t cascadeIndex,
          uint32_t staticBatchIndex) -> ShadowBatchEntry & {
    const size_t slotIndex =
        staticBatchEntrySlot(cascadeIndex, staticBatchIndex);
    uint32_t &entryIndex = staticBatchEntryIndices[slotIndex];
    if (entryIndex == std::numeric_limits<uint32_t>::max()) {
      entryIndex = static_cast<uint32_t>(shadowBatchEntries.size());
      shadowBatchEntries.emplace_back();
      ShadowBatchEntry &batch = shadowBatchEntries.back();
      const StaticShadowBatchTemplate &batchTemplate =
          staticShadowBatchTemplates_[staticBatchIndex];
      batch.key = makeStaticShadowBatchKey(batchTemplate, cascadeIndex);
    }
    return shadowBatchEntries[entryIndex];
  };

  std::array<glm::vec3, kMaxShadowCascades> cascadeLightBoundsMin{};
  std::array<glm::vec3, kMaxShadowCascades> cascadeLightBoundsMax{};
  std::array<uint8_t, kMaxShadowCascades> cascadeUsesCachedLightBounds{};
  for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
       ++cascadeIndex) {
    const ShadowCascadeDebugFrameData &cascadeDebug =
        shadowDebugFrameData_.cascades[cascadeIndex];
    const float cullingPadding = std::max(
        cascadeDebug.texelWorldSize * kCullingPaddingTexelMultiplier, 0.01f);
    const glm::vec3 rawBoundsMin(cascadeDebug.lightSpaceBoundsMin);
    const glm::vec3 rawBoundsMax(cascadeDebug.lightSpaceBoundsMax);
    cascadeLightBoundsMin[cascadeIndex] =
        glm::min(rawBoundsMin, rawBoundsMax) - glm::vec3(cullingPadding);
    cascadeLightBoundsMax[cascadeIndex] =
        glm::max(rawBoundsMin, rawBoundsMax) + glm::vec3(cullingPadding);
    cascadeUsesCachedLightBounds[cascadeIndex] =
        hasStaticShadowCasterLightSpaceBounds_ &&
                rawBytesEqual(staticShadowCasterLightSpaceBoundsView_,
                              cascadeDebug.lightView)
            ? 1u
            : 0u;
  }

  const uint32_t staticCandidateCasterCount = saturateToU32(
      staticShadowCasterCache_.size() -
      std::min(staticShadowCasterCache_.size(),
               static_cast<size_t>(overriddenStaticTemplateCount)));
  if (staticShadowCasterLightGrid_.valid) {
    if (staticShadowCasterLightGridQueryMarks_.size() !=
        staticShadowCasterCache_.size()) {
      staticShadowCasterLightGridQueryMarks_.assign(
          staticShadowCasterCache_.size(), 0u);
      staticShadowCasterLightGridQueryMarker_ = 1u;
    }
    staticShadowCasterLightGridQueryEntries_.reserve(
        std::min(staticShadowCasterCache_.size(),
                 kStaticCasterLightGridSortedCandidateLimit));
  }
  if (staticShadowBatchLightGrid_.valid) {
    staticBatchLightGridQueryMarks.resize(staticShadowBatchTemplates_.size(),
                                          0u);
    staticBatchLightGridQueryEntries.reserve(
        std::min(staticShadowBatchTemplates_.size(),
                 kStaticCasterLightGridSortedCandidateLimit));
  }

  const auto nextStaticCasterQueryMarker = [&]() {
    ++staticShadowCasterLightGridQueryMarker_;
    if (staticShadowCasterLightGridQueryMarker_ == 0u) {
      std::fill(staticShadowCasterLightGridQueryMarks_.begin(),
                staticShadowCasterLightGridQueryMarks_.end(), 0u);
      staticShadowCasterLightGridQueryMarker_ = 1u;
    }
    return staticShadowCasterLightGridQueryMarker_;
  };
  const auto nextStaticBatchQueryMarker = [&]() {
    ++staticBatchLightGridQueryMarker;
    if (staticBatchLightGridQueryMarker == 0u) {
      std::fill(staticBatchLightGridQueryMarks.begin(),
                staticBatchLightGridQueryMarks.end(), 0u);
      staticBatchLightGridQueryMarker = 1u;
    }
    return staticBatchLightGridQueryMarker;
  };
  const auto staticEntryUsesDynamicPath =
      [&](const StaticShadowCasterCacheEntry &entry) {
        return entry.templateIndex < staticTemplatesUsingDynamicPath.size() &&
               staticTemplatesUsingDynamicPath[entry.templateIndex] != 0u;
      };
  const auto staticBatchWasFullyEmitted = [&](uint32_t staticBatchIndex,
                                              uint32_t cascadeIndex) {
    if (staticBatchIndex >= staticShadowBatchTemplates_.size()) {
      return false;
    }
    const size_t slotIndex =
        staticBatchEntrySlot(cascadeIndex, staticBatchIndex);
    NURI_ASSERT(slotIndex < fullyEmittedStaticBatchEntries.size(),
                "Static shadow batch emitted slot is out of bounds");
    return fullyEmittedStaticBatchEntries[slotIndex] != 0u;
  };
  const auto staticCasterDepthOverlapsCascade = [&](uint32_t entryIndex,
                                                    uint32_t cascadeIndex) {
    const StaticShadowCasterLightSpaceBounds &bounds =
        staticShadowCasterLightSpaceBounds_[entryIndex];
    return bounds.max.z >= cascadeLightBoundsMin[cascadeIndex].z &&
           bounds.min.z <= cascadeLightBoundsMax[cascadeIndex].z;
  };
  const auto emitStaticCasterUnchecked = [&](uint32_t entryIndex,
                                             uint32_t cascadeIndex) {
    const StaticShadowCasterCacheEntry &entry =
        staticShadowCasterCache_[entryIndex];
    if (entry.batchIndex >= staticShadowBatchTemplates_.size()) {
      return false;
    }

    ShadowBatchEntry &batch =
        resolveStaticBatchEntry(cascadeIndex, entry.batchIndex);
    batch.appendInstance(
        entry.instanceIndex, shadowBatchInstanceIndices,
        staticShadowBatchTemplates_[entry.batchIndex].instanceCount);
    ++cascadeStaticCasterCounts[cascadeIndex];
    StaticOnlyCascadeReuseState &cascadeState =
        currentStaticOnlyCascadeStates[cascadeIndex];
    cascadeState.casterSignature =
        hashCombine64(cascadeState.casterSignature, entry.rasterSignature);
    cascadeState.rasterSignature =
        hashCombine64(cascadeState.rasterSignature, entry.rasterSignature);
    cascadeIndexCountEstimates_[cascadeIndex] += entry.indexCount;
    return true;
  };
  const auto emitStaticBatch = [&](uint32_t batchIndex, uint32_t cascadeIndex) {
    const StaticShadowBatchTemplate &batchTemplate =
        staticShadowBatchTemplates_[batchIndex];
    if (batchTemplate.instanceCount == 0u) {
      return;
    }
    const size_t firstInstanceIndex = batchTemplate.firstInstanceIndex;
    const size_t endInstanceIndex =
        firstInstanceIndex + batchTemplate.instanceCount;
    NURI_ASSERT(endInstanceIndex <= staticShadowBatchInstanceIndices_.size(),
                "Static shadow batch instance range is out of bounds");
    if (endInstanceIndex > staticShadowBatchInstanceIndices_.size()) {
      return;
    }

    ShadowBatchEntry &batch = resolveStaticBatchEntry(cascadeIndex, batchIndex);
    batch.appendInstances(
        std::span<const uint32_t>(staticShadowBatchInstanceIndices_.data() +
                                      firstInstanceIndex,
                                  batchTemplate.instanceCount),
        shadowBatchInstanceIndices);
  };
  const auto emitStaticBatchWithState = [&](uint32_t batchIndex,
                                            uint32_t cascadeIndex) {
    const StaticShadowBatchTemplate &batchTemplate =
        staticShadowBatchTemplates_[batchIndex];
    if (batchTemplate.instanceCount == 0u) {
      return;
    }
    emitStaticBatch(batchIndex, cascadeIndex);
    cascadeStaticCasterCounts[cascadeIndex] += batchTemplate.instanceCount;
    StaticOnlyCascadeReuseState &cascadeState =
        currentStaticOnlyCascadeStates[cascadeIndex];
    cascadeState.casterSignature = hashCombine64(cascadeState.casterSignature,
                                                 batchTemplate.rasterSignature);
    cascadeState.rasterSignature = hashCombine64(cascadeState.rasterSignature,
                                                 batchTemplate.rasterSignature);
    cascadeIndexCountEstimates_[cascadeIndex] +=
        batchTemplate.indexCountEstimate;
  };
  const auto emitAllStaticCastersForCascade = [&](uint32_t cascadeIndex) {
    for (uint32_t batchIndex = 0u;
         batchIndex < staticShadowBatchTemplates_.size(); ++batchIndex) {
      if (staticShadowBatchTemplates_[batchIndex].instanceCount != 0u) {
        ++cascadeStaticBatchFullEmitCounts[cascadeIndex];
      }
      emitStaticBatch(batchIndex, cascadeIndex);
    }

    cascadeStaticCasterCounts[cascadeIndex] += staticCandidateCasterCount;
    StaticOnlyCascadeReuseState &cascadeState =
        currentStaticOnlyCascadeStates[cascadeIndex];
    cascadeState.casterSignature = staticShadowCasterCacheContentSignature_;
    cascadeState.rasterSignature = hashCombine64(
        cascadeState.rasterSignature, staticShadowCasterCacheContentSignature_);
    cascadeIndexCountEstimates_[cascadeIndex] +=
        staticShadowCasterCacheIndexCountEstimate_;
  };
  const auto cascadeContainsAllStaticCasters = [&](uint32_t cascadeIndex) {
    if (staticShadowCasterCache_.empty()) {
      return true;
    }
    if (!enableCascadeCasterCulling) {
      return true;
    }
    if (cascadeUsesCachedLightBounds[cascadeIndex] == 0u ||
        !hasStaticShadowCasterLightSpaceBounds_) {
      return false;
    }

    const glm::vec3 &cascadeMin = cascadeLightBoundsMin[cascadeIndex];
    const glm::vec3 &cascadeMax = cascadeLightBoundsMax[cascadeIndex];
    return staticShadowCasterLightSpaceBoundsMin_.x >= cascadeMin.x &&
           staticShadowCasterLightSpaceBoundsMin_.y >= cascadeMin.y &&
           staticShadowCasterLightSpaceBoundsMin_.z >= cascadeMin.z &&
           staticShadowCasterLightSpaceBoundsMax_.x <= cascadeMax.x &&
           staticShadowCasterLightSpaceBoundsMax_.y <= cascadeMax.y &&
           staticShadowCasterLightSpaceBoundsMax_.z <= cascadeMax.z;
  };
  const auto emitOverlappingStaticBatch = [&](uint32_t cascadeIndex,
                                              uint32_t batchIndex) {
    if (batchIndex >= staticShadowBatchTemplates_.size()) {
      return;
    }
    const StaticShadowBatchTemplate &batchTemplate =
        staticShadowBatchTemplates_[batchIndex];
    if (batchTemplate.instanceCount < kStaticBatchOverlapEmitMinInstances) {
      return;
    }
    const StaticShadowCasterLightSpaceBounds &batchBounds =
        staticShadowBatchLightSpaceBounds_[batchIndex];
    if (!normalizedLightSpaceBoundsOverlap(
            batchBounds.min, batchBounds.max,
            cascadeLightBoundsMin[cascadeIndex],
            cascadeLightBoundsMax[cascadeIndex])) {
      return;
    }

    const size_t slotIndex = staticBatchEntrySlot(cascadeIndex, batchIndex);
    NURI_ASSERT(slotIndex < fullyEmittedStaticBatchEntries.size(),
                "Static shadow batch emitted slot is out of bounds");
    fullyEmittedStaticBatchEntries[slotIndex] = 1u;
    emitStaticBatchWithState(batchIndex, cascadeIndex);
    ++cascadeStaticBatchFullEmitCounts[cascadeIndex];
  };
  const auto scanOverlappingStaticBatchesForCascade =
      [&](uint32_t cascadeIndex) {
        for (uint32_t batchIndex = 0u;
             batchIndex < staticShadowBatchTemplates_.size(); ++batchIndex) {
          emitOverlappingStaticBatch(cascadeIndex, batchIndex);
        }
      };
  struct StaticLightGridQueryRange {
    uint32_t minX = 0u;
    uint32_t minY = 0u;
    uint32_t minZ = 0u;
    uint32_t maxX = 0u;
    uint32_t maxY = 0u;
    uint32_t maxZ = 0u;
    uint32_t cellCount = 0u;
    bool outside = false;
    bool tooBroad = false;
  };
  const auto makeStaticLightGridQueryRange =
      [&](const StaticShadowCasterLightGrid &grid,
          uint32_t cascadeIndex) -> StaticLightGridQueryRange {
    const glm::vec3 &queryMin = cascadeLightBoundsMin[cascadeIndex];
    const glm::vec3 &queryMax = cascadeLightBoundsMax[cascadeIndex];
    if (queryMax.x < grid.min.x || queryMin.x > grid.max.x ||
        queryMax.y < grid.min.y || queryMin.y > grid.max.y ||
        queryMax.z < grid.min.z || queryMin.z > grid.max.z) {
      return StaticLightGridQueryRange{.outside = true};
    }

    const glm::ivec3 minCell = glm::clamp(
        glm::ivec3(glm::floor((queryMin - grid.min) * grid.invCellSize)),
        glm::ivec3(0), glm::ivec3(grid.dimensions) - glm::ivec3(1));
    const glm::ivec3 maxCell = glm::clamp(
        glm::ivec3(glm::floor((queryMax - grid.min) * grid.invCellSize)),
        glm::ivec3(0), glm::ivec3(grid.dimensions) - glm::ivec3(1));
    StaticLightGridQueryRange range{
        .minX = static_cast<uint32_t>(std::min(minCell.x, maxCell.x)),
        .minY = static_cast<uint32_t>(std::min(minCell.y, maxCell.y)),
        .minZ = static_cast<uint32_t>(std::min(minCell.z, maxCell.z)),
        .maxX = static_cast<uint32_t>(std::max(minCell.x, maxCell.x)),
        .maxY = static_cast<uint32_t>(std::max(minCell.y, maxCell.y)),
        .maxZ = static_cast<uint32_t>(std::max(minCell.z, maxCell.z)),
    };
    range.cellCount = (range.maxX - range.minX + 1u) *
                      (range.maxY - range.minY + 1u) *
                      (range.maxZ - range.minZ + 1u);
    const uint32_t gridCellCount =
        grid.dimensions.x * grid.dimensions.y * grid.dimensions.z;
    range.tooBroad = range.cellCount > gridCellCount / 2u;
    return range;
  };
  const auto queryStaticBatchLightGrid = [&](uint32_t cascadeIndex,
                                             uint32_t queryMarker) {
    staticBatchLightGridQueryEntries.clear();
    if (!staticShadowBatchLightGrid_.valid ||
        staticShadowBatchLightGridCells_.empty() ||
        cascadeUsesCachedLightBounds[cascadeIndex] == 0u) {
      return false;
    }
    ++staticLightGridQueryCount;

    const StaticShadowCasterLightGrid &grid = staticShadowBatchLightGrid_;
    const StaticLightGridQueryRange query =
        makeStaticLightGridQueryRange(grid, cascadeIndex);
    if (query.outside) {
      return true;
    }
    cascadeStaticLightGridQueryCellCounts[cascadeIndex] += query.cellCount;
    if (query.tooBroad) {
      return false;
    }

    const auto appendCandidate = [&](uint32_t batchIndex) {
      if (batchIndex >= staticShadowBatchTemplates_.size() ||
          batchIndex >= staticBatchLightGridQueryMarks.size() ||
          staticShadowBatchTemplates_[batchIndex].instanceCount <
              kStaticBatchOverlapEmitMinInstances) {
        return;
      }
      uint32_t &mark = staticBatchLightGridQueryMarks[batchIndex];
      if (mark == queryMarker) {
        return;
      }
      mark = queryMarker;
      staticBatchLightGridQueryEntries.push_back(batchIndex);
    };
    for (const uint32_t batchIndex : staticShadowBatchLargeLightGridEntries_) {
      appendCandidate(batchIndex);
    }
    for (uint32_t z = query.minZ; z <= query.maxZ; ++z) {
      for (uint32_t y = query.minY; y <= query.maxY; ++y) {
        for (uint32_t x = query.minX; x <= query.maxX; ++x) {
          const uint32_t cellIndex =
              (z * grid.dimensions.y + y) * grid.dimensions.x + x;
          const StaticShadowCasterLightGridCell &cell =
              staticShadowBatchLightGridCells_[cellIndex];
          const uint32_t endEntry = cell.firstEntry + cell.entryCount;
          for (uint32_t i = cell.firstEntry; i < endEntry; ++i) {
            appendCandidate(staticShadowBatchLightGridEntries_[i]);
          }
        }
      }
    }
    std::sort(staticBatchLightGridQueryEntries.begin(),
              staticBatchLightGridQueryEntries.end());
    cascadeStaticLightGridCandidateCounts[cascadeIndex] +=
        saturateToU32(staticBatchLightGridQueryEntries.size());
    return true;
  };
  const auto emitOverlappingStaticBatchesForCascade =
      [&](uint32_t cascadeIndex) {
        if (overriddenStaticTemplateCount != 0u ||
            cascadeUsesCachedLightBounds[cascadeIndex] == 0u ||
            staticShadowBatchLightSpaceBounds_.size() !=
                staticShadowBatchTemplates_.size()) {
          return;
        }

        // Conservative CPU fast path: this may overdraw a batch, but it keeps
        // per-caster precision for small batches and never drops casters.
        const bool queriedBatchGrid = queryStaticBatchLightGrid(
            cascadeIndex, nextStaticBatchQueryMarker());
        if (!queriedBatchGrid) {
          ++cascadeStaticLightGridFallbackScanCounts[cascadeIndex];
          scanOverlappingStaticBatchesForCascade(cascadeIndex);
          return;
        }
        for (const uint32_t batchIndex : staticBatchLightGridQueryEntries) {
          emitOverlappingStaticBatch(cascadeIndex, batchIndex);
        }
      };
  const auto emitGridCandidateIfVisible = [&](uint32_t entryIndex,
                                              uint32_t cascadeIndex) {
    const StaticShadowCasterCacheEntry &entry =
        staticShadowCasterCache_[entryIndex];
    if (staticEntryUsesDynamicPath(entry) ||
        staticBatchWasFullyEmitted(entry.batchIndex, cascadeIndex)) {
      return false;
    }
    if (entry.hasCasterCullingBounds) {
      const bool overlapsCascade = normalizedLightSpaceBoundsOverlap(
          staticShadowCasterLightSpaceBounds_[entryIndex].min,
          staticShadowCasterLightSpaceBounds_[entryIndex].max,
          cascadeLightBoundsMin[cascadeIndex],
          cascadeLightBoundsMax[cascadeIndex]);
      if (!overlapsCascade) {
        return false;
      }
    }
    return emitStaticCasterUnchecked(entryIndex, cascadeIndex);
  };
  const auto queryStaticCasterLightGrid = [&](uint32_t cascadeIndex,
                                              uint32_t queryMarker,
                                              bool &emitMarkedByCacheOrder) {
    emitMarkedByCacheOrder = false;
    staticShadowCasterLightGridQueryEntries_.clear();
    if (!staticShadowCasterLightGrid_.valid ||
        cascadeUsesCachedLightBounds[cascadeIndex] == 0u) {
      return false;
    }
    ++staticLightGridQueryCount;

    const StaticShadowCasterLightGrid &grid = staticShadowCasterLightGrid_;
    const StaticLightGridQueryRange query =
        makeStaticLightGridQueryRange(grid, cascadeIndex);
    if (query.outside) {
      return true;
    }
    cascadeStaticLightGridQueryCellCounts[cascadeIndex] += query.cellCount;
    if (query.tooBroad) {
      return false;
    }

    const auto appendCandidate = [&](uint32_t entryIndex) {
      const StaticShadowCasterCacheEntry &entry =
          staticShadowCasterCache_[entryIndex];
      if (staticEntryUsesDynamicPath(entry) ||
          staticBatchWasFullyEmitted(entry.batchIndex, cascadeIndex) ||
          !staticCasterDepthOverlapsCascade(entryIndex, cascadeIndex)) {
        return;
      }
      uint32_t &mark = staticShadowCasterLightGridQueryMarks_[entryIndex];
      if (mark == queryMarker) {
        return;
      }
      mark = queryMarker;
      staticShadowCasterLightGridQueryEntries_.push_back(entryIndex);
    };
    for (const uint32_t entryIndex : staticShadowCasterLargeLightGridEntries_) {
      appendCandidate(entryIndex);
    }
    for (uint32_t z = query.minZ; z <= query.maxZ; ++z) {
      for (uint32_t y = query.minY; y <= query.maxY; ++y) {
        for (uint32_t x = query.minX; x <= query.maxX; ++x) {
          const uint32_t cellIndex =
              (z * grid.dimensions.y + y) * grid.dimensions.x + x;
          const StaticShadowCasterLightGridCell &cell =
              staticShadowCasterLightGridCells_[cellIndex];
          const uint32_t endEntry = cell.firstEntry + cell.entryCount;
          for (uint32_t i = cell.firstEntry; i < endEntry; ++i) {
            appendCandidate(staticShadowCasterLightGridEntries_[i]);
          }
        }
      }
    }
    if (staticShadowCasterLightGridQueryEntries_.size() >
        kStaticCasterLightGridSortedCandidateLimit) {
      emitMarkedByCacheOrder = true;
    } else {
      std::sort(staticShadowCasterLightGridQueryEntries_.begin(),
                staticShadowCasterLightGridQueryEntries_.end());
    }
    cascadeStaticLightGridCandidateCounts[cascadeIndex] +=
        saturateToU32(staticShadowCasterLightGridQueryEntries_.size());
    return true;
  };
  const auto scanStaticCastersForCascade = [&](uint32_t cascadeIndex) {
    for (uint32_t entryIndex = 0u; entryIndex < staticShadowCasterCache_.size();
         ++entryIndex) {
      const StaticShadowCasterCacheEntry &entry =
          staticShadowCasterCache_[entryIndex];
      if (staticEntryUsesDynamicPath(entry) ||
          staticBatchWasFullyEmitted(entry.batchIndex, cascadeIndex)) {
        continue;
      }
      if (entry.hasCasterCullingBounds) {
        const bool overlapsCascade =
            cascadeUsesCachedLightBounds[cascadeIndex] != 0u &&
                    entryIndex < staticShadowCasterLightSpaceBounds_.size()
                ? normalizedLightSpaceBoundsOverlap(
                      staticShadowCasterLightSpaceBounds_[entryIndex].min,
                      staticShadowCasterLightSpaceBounds_[entryIndex].max,
                      cascadeLightBoundsMin[cascadeIndex],
                      cascadeLightBoundsMax[cascadeIndex])
                : shadow_detail::shadowCasterOverlapsLightSpaceBounds(
                      std::span<const glm::vec3, 8>(entry.casterWorldCorners),
                      shadowDebugFrameData_.cascades[cascadeIndex].lightView,
                      cascadeLightBoundsMin[cascadeIndex],
                      cascadeLightBoundsMax[cascadeIndex], 0.0f);
        if (!overlapsCascade) {
          ++cascadeCulledCounts_[cascadeIndex];
          continue;
        }
      }
      (void)emitStaticCasterUnchecked(entryIndex, cascadeIndex);
    }
  };

  {
    NURI_PROFILER_ZONE("ShadowRenderer.static_caster_emit",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
         ++cascadeIndex) {
      if (skipStaticCasterBuildForCascade[cascadeIndex] != 0u) {
        continue;
      }

      if (overriddenStaticTemplateCount == 0u &&
          cascadeContainsAllStaticCasters(cascadeIndex)) {
        emitAllStaticCastersForCascade(cascadeIndex);
        continue;
      }

      const uint32_t staticDrawCountBefore =
          cascadeStaticCasterCounts[cascadeIndex];
      emitOverlappingStaticBatchesForCascade(cascadeIndex);

      bool emitMarkedByCacheOrder = false;
      const uint32_t queryMarker = nextStaticCasterQueryMarker();
      const bool queriedStaticLightGrid =
          enableCascadeCasterCulling &&
          queryStaticCasterLightGrid(cascadeIndex, queryMarker,
                                     emitMarkedByCacheOrder);
      if (!queriedStaticLightGrid) {
        if (enableCascadeCasterCulling) {
          ++cascadeStaticLightGridFallbackScanCounts[cascadeIndex];
        }
        scanStaticCastersForCascade(cascadeIndex);
        continue;
      }

      if (emitMarkedByCacheOrder) {
        for (uint32_t entryIndex = 0u;
             entryIndex < staticShadowCasterCache_.size(); ++entryIndex) {
          if (staticShadowCasterLightGridQueryMarks_[entryIndex] !=
              queryMarker) {
            continue;
          }
          (void)emitGridCandidateIfVisible(entryIndex, cascadeIndex);
        }
      } else {
        for (const uint32_t entryIndex :
             staticShadowCasterLightGridQueryEntries_) {
          (void)emitGridCandidateIfVisible(entryIndex, cascadeIndex);
        }
      }
      const uint32_t emittedStaticDrawCount =
          cascadeStaticCasterCounts[cascadeIndex] - staticDrawCountBefore;
      if (staticCandidateCasterCount > emittedStaticDrawCount) {
        cascadeCulledCounts_[cascadeIndex] +=
            staticCandidateCasterCount - emittedStaticDrawCount;
      }
    }
    NURI_PROFILER_ZONE_END();
  }

  {
    NURI_PROFILER_ZONE("ShadowRenderer.dynamic_caster_emit",
                       NURI_PROFILER_COLOR_CMD_DRAW);
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
        if (skipStaticCasterBuildForCascade[cascadeIndex] != 0u) {
          continue;
        }
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
                  glm::vec3(cascadeDebug.lightSpaceBoundsMax),
                  cullingPadding)) {
            ++cascadeCulledCounts_[cascadeIndex];
            continue;
          }
        }

        const bool deformedRenderable =
            !entry.renderable->morphWeights.empty() ||
            !entry.renderable->skinPalette.empty();
        const bool enableMeshletCascadeCulling =
            settings.shadow.enableMeshletCascadeCulling &&
            !usesAnimatedOverride && !deformedRenderable;
        appendShadowDraw(
            cascadeIndex, resolvedVertexBuffer, entry.indexBuffer,
            entry.indexBufferOffset, entry.indexFormat, lod->indexCount,
            lod->indexOffset, entry.instanceIndex, resolvedVertexBufferAddress,
            entry.vertexDecodeBuffer, resolvedVertexDecodeBufferAddress,
            resolvedVertexDecodeIndex, resolvedPackedVertexFormat,
            entry.materialIndex, entry.doubleSided, entry.alphaMasked,
            entry.meshletView, entry.submeshIndex, entry.meshletMaxCount,
            lod->meshletCount, entry.submesh->vertexOffset, true,
            enableMeshletCascadeCulling, false);
        cascadeIndexCountEstimates_[cascadeIndex] += lod->indexCount;
      }
    }
    NURI_PROFILER_ZONE_END();
  }

  uint32_t emittedShadowBatchEntryCount = 0u;
  uint32_t emittedShadowMeshletDispatchCount = 0u;
  uint32_t emittedShadowMeshletTaskGroupCount = 0u;
  uint64_t emittedShadowMeshletCandidateCount = 0u;
  uint32_t emittedShadowInstanceRemapCount = 0u;
  {
    NURI_PROFILER_ZONE("ShadowRenderer.batch_emit",
                       NURI_PROFILER_COLOR_CMD_DRAW);
    std::array<uint32_t, kMaxShadowCascades> cascadeBatchCounts{};
    const MeshletLimits meshletLimits = gpu_.getMeshletLimits();
    uint32_t maxTaskGroupsX = meshletLimits.maxTaskWorkGroupCountX != 0u
                                  ? meshletLimits.maxTaskWorkGroupCountX
                                  : std::numeric_limits<uint32_t>::max();
    if (meshletLimits.maxTaskWorkGroupTotalCount != 0u) {
      maxTaskGroupsX =
          std::min(maxTaskGroupsX, meshletLimits.maxTaskWorkGroupTotalCount);
    }
    maxTaskGroupsX = std::max(maxTaskGroupsX, 1u);

    size_t emittedInstanceRemapCount = 0u;
    for (const ShadowBatchEntry &batch : shadowBatchEntries) {
      if (batch.instanceCount == 0u) {
        continue;
      }
      ++emittedShadowBatchEntryCount;
      emittedInstanceRemapCount += batch.instanceCount;
      if (batch.key.cascadeIndex < cascadeBatchCounts.size()) {
        ++cascadeBatchCounts[batch.key.cascadeIndex];
      }
    }
    for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
         ++cascadeIndex) {
      cascadePushConstants_[cascadeIndex].reserve(
          cascadeBatchCounts[cascadeIndex]);
      cascadeDrawItems_[cascadeIndex].reserve(cascadeBatchCounts[cascadeIndex]);
      cascadeMeshletPushConstants_[cascadeIndex].reserve(
          cascadeBatchCounts[cascadeIndex]);
      cascadeMeshDispatchItems_[cascadeIndex].reserve(
          cascadeBatchCounts[cascadeIndex]);
      cascadeMeshDispatchDependencyBuffers_[cascadeIndex].reserve(
          cascadeBatchCounts[cascadeIndex]);
    }
    instanceRemap_.resize(emittedInstanceRemapCount);
    uint32_t *remapWrite = instanceRemap_.data();

    for (ShadowBatchEntry &batch : shadowBatchEntries) {
      if (batch.instanceCount == 0u) {
        continue;
      }
      const ShadowBatchKey &key = batch.key;
      batch.firstInstance = saturateToU32(
          static_cast<size_t>(remapWrite - instanceRemap_.data()));
      if (batch.externalInstanceCount != 0u) {
        NURI_ASSERT(batch.externalInstanceCount == batch.instanceCount,
                    "External shadow batch remap count is inconsistent");
        std::copy_n(batch.externalInstanceIndices, batch.externalInstanceCount,
                    remapWrite);
        remapWrite += batch.externalInstanceCount;
      } else if (batch.hasInstanceList) {
        const size_t remapEnd =
            static_cast<size_t>(batch.instanceListOffset) + batch.instanceCount;
        NURI_ASSERT(remapEnd <= shadowBatchInstanceIndices.size(),
                    "Materialized shadow batch remap count is inconsistent");
        std::copy_n(shadowBatchInstanceIndices.data() +
                        batch.instanceListOffset,
                    batch.instanceCount, remapWrite);
        remapWrite += batch.instanceCount;
      } else {
        NURI_ASSERT(batch.instanceCount == 1u,
                    "Inline shadow batch stores exactly one instance");
        *remapWrite++ = batch.inlineInstanceIndex;
      }

      if (key.useMeshlets) {
        const uint64_t meshletBufferAddress =
            gpu_.getBufferDeviceAddress(key.meshletBuffer);
        const uint64_t meshletVertexIndexAddress =
            gpu_.getBufferDeviceAddress(key.meshletVertexIndexBuffer);
        const uint64_t meshletPrimitiveIndexAddress =
            gpu_.getBufferDeviceAddress(key.meshletPrimitiveIndexBuffer);
        const uint64_t meshletLodRangeAddress =
            gpu_.getBufferDeviceAddress(key.meshletLodRangeBuffer);
        if (meshletBufferAddress == 0u || meshletVertexIndexAddress == 0u ||
            meshletPrimitiveIndexAddress == 0u ||
            meshletLodRangeAddress == 0u) {
          return Result<bool, std::string>::makeError(
              "ShadowRenderer::buildShadowDraws: invalid shadow meshlet "
              "buffer address");
        }

        const uint64_t candidateCount64 =
            static_cast<uint64_t>(key.meshletMaxCount) * batch.instanceCount;
        if (candidateCount64 > std::numeric_limits<uint32_t>::max()) {
          return Result<bool, std::string>::makeError(
              "ShadowRenderer::buildShadowDraws: shadow meshlet candidate "
              "count exceeds UINT32_MAX");
        }
        const uint32_t candidateCount = static_cast<uint32_t>(candidateCount64);
        emittedShadowMeshletCandidateCount += candidateCount;
        const uint32_t forcedMeshLod =
            settings.opaque.forcedMeshLod >= 0
                ? std::min(static_cast<uint32_t>(settings.opaque.forcedMeshLod),
                           Submesh::kMaxLodCount - 1u)
                : 0u;
        uint32_t meshletFlags = (forcedMeshLod & kMeshletFlagForcedLodMask)
                                << kMeshletFlagForcedLodShift;
        const bool doubleSidedMeshlet =
            isSameMeshletPipelineHandle(
                key.meshletPipeline, shadowMeshletDoubleSidedPipelineHandle_) ||
            isSameMeshletPipelineHandle(
                key.meshletPipeline,
                shadowMeshletAlphaDoubleSidedPipelineHandle_);
        meshletFlags |= doubleSidedMeshlet ? kMeshletFlagDoubleSided : 0u;
        meshletFlags |= key.enableMeshletCascadeCulling
                            ? kMeshletFlagShadowCascadeCulling
                            : 0u;
        uint32_t candidateOffset = 0u;
        while (candidateOffset < candidateCount) {
          const uint32_t groupCount =
              std::min(maxTaskGroupsX, candidateCount - candidateOffset);
          MeshletPushConstants &pc =
              cascadeMeshletPushConstants_[key.cascadeIndex].emplace_back();
          pc = MeshletPushConstants{
              .frameDataAddress = sceneGpu.frameDataAddress,
              .vertexBufferAddress = key.vertexBufferAddress,
              .vertexDecodeBufferAddress = key.vertexDecodeBufferAddress,
              .instanceMatricesAddress = instanceMatricesAddress,
              .instanceRemapAddress = instanceRemapAddress,
              .visibilityCounterAddress = shadowMeshletCounterBufferAddress,
              .meshletBufferAddress = meshletBufferAddress,
              .meshletVertexIndexBufferAddress = meshletVertexIndexAddress,
              .meshletPrimitiveIndexBufferAddress =
                  meshletPrimitiveIndexAddress,
              .meshletLodRangeBufferAddress = meshletLodRangeAddress,
              .instanceCount = batch.instanceCount,
              .materialIndex = key.materialIndex,
              .vertexDecodeIndex = key.vertexDecodeIndex,
              .packedVertexFormat = key.packedVertexFormat,
              .firstInstance = batch.firstInstance,
              .candidateOffset = candidateOffset,
              .meshletFlags = meshletFlags,
              .vertexOffset = key.vertexOffset,
              .lodThresholds = glm::vec4(
                  std::bit_cast<float>(static_cast<uint32_t>(frame.frameIndex)),
                  std::bit_cast<float>(shadowMeshletCounterFlags),
                  std::bit_cast<float>(key.cascadeIndex),
                  std::bit_cast<float>(key.submeshIndex)),
          };

          MeshDispatchItem &dispatch =
              cascadeMeshDispatchItems_[key.cascadeIndex].emplace_back();
          dispatch = MeshDispatchItem{};
          dispatch.command = MeshDispatchCommandType::Direct;
          dispatch.pipeline = key.meshletPipeline;
          dispatch.groupsX = groupCount;
          dispatch.groupsY = 1u;
          dispatch.groupsZ = 1u;
          dispatch.useDepthState = true;
          dispatch.depthState = {.compareOp = CompareOp::Less,
                                 .isDepthWriteEnabled = true};
          dispatch.depthBiasEnable = true;
          dispatch.depthBiasConstant = settings.shadow.constantBias;
          dispatch.depthBiasSlope = settings.shadow.slopeBias;
          dispatch.depthBiasClamp = 0.0f;
          dispatch.pushConstants = std::span<const std::byte>(
              reinterpret_cast<const std::byte *>(&pc),
              sizeof(MeshletPushConstants));

          cascadeMeshDispatchDependencyBuffers_[key.cascadeIndex].push_back(
              std::pmr::vector<BufferHandle>(memory_));
          std::pmr::vector<BufferHandle> &dependencies =
              cascadeMeshDispatchDependencyBuffers_[key.cascadeIndex].back();
          dependencies.reserve(6u);
          appendUniqueBufferHandle(dependencies, key.vertexBuffer);
          appendUniqueBufferHandle(dependencies, key.vertexDecodeBuffer);
          appendUniqueBufferHandle(dependencies, key.meshletBuffer);
          appendUniqueBufferHandle(dependencies, key.meshletVertexIndexBuffer);
          appendUniqueBufferHandle(dependencies,
                                   key.meshletPrimitiveIndexBuffer);
          appendUniqueBufferHandle(dependencies, key.meshletLodRangeBuffer);
          dispatch.dependencyBuffers = std::span<const BufferHandle>(
              dependencies.data(), dependencies.size());
          dispatch.debugLabel = kShadowMeshLabel;
          dispatch.debugColor = kShadowMeshDebugColor;
          ++emittedShadowMeshletDispatchCount;
          emittedShadowMeshletTaskGroupCount += groupCount;
          candidateOffset += groupCount;
        }
        continue;
      }

      PushConstants &pc =
          cascadePushConstants_[key.cascadeIndex].emplace_back();
      pc = PushConstants{
          .frameDataAddress = sceneGpu.frameDataAddress,
          .vertexBufferAddress = key.vertexBufferAddress,
          .vertexDecodeBufferAddress = key.vertexDecodeBufferAddress,
          .instanceMatricesAddress = instanceMatricesAddress,
          .instanceRemapAddress = instanceRemapAddress,
          .instanceCentersPhaseAddress = 0u,
          .instanceBaseMatricesAddress = 0u,
          .instanceCount = renderableCount,
          .materialIndex = key.materialIndex,
          .vertexDecodeIndex = key.vertexDecodeIndex,
          .packedVertexFormat = key.packedVertexFormat,
          .timeSeconds = static_cast<float>(frame.timeSeconds),
          .tessNearDistance = 1.0f,
          .tessFarDistance = 8.0f,
          .tessMinFactor = 1.0f,
          .tessMaxFactor = 1.0f,
          .debugVisualizationMode = 0u,
          .shadowCascadeIndex = key.cascadeIndex,
      };

      DrawItem &draw = cascadeDrawItems_[key.cascadeIndex].emplace_back();
      draw = DrawItem{};
      draw.pipeline = key.pipeline;
      draw.vertexBuffer = key.vertexBuffer;
      draw.indexBuffer = key.indexBuffer;
      draw.indexBufferOffset = key.indexBufferOffset;
      draw.indexFormat = key.indexFormat;
      draw.indexCount = key.indexCount;
      draw.instanceCount = batch.instanceCount;
      draw.firstIndex = key.firstIndex;
      draw.firstInstance = batch.firstInstance;
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
    }
    emittedShadowInstanceRemapCount = saturateToU32(emittedInstanceRemapCount);
    if (!instanceRemap_.empty()) {
      NURI_ASSERT(
          remapWrite == instanceRemap_.data() + instanceRemap_.size(),
          "Shadow instance remap write cursor did not consume the target span");
    }
    NURI_PROFILER_ZONE_END();
  }

  if (!instanceRemap_.empty()) {
    const std::span<const std::byte> remapBytes{
        reinterpret_cast<const std::byte *>(instanceRemap_.data()),
        instanceRemap_.size() * sizeof(uint32_t)};
    const uint64_t remapSignature = hashBytes(remapBytes);
    if (instanceRemapUploadSignatures_[frameSlot] != remapSignature) {
      auto updateRemapResult =
          gpu_.updateBuffer(instanceRemapBuffer, remapBytes, 0u);
      if (updateRemapResult.hasError()) {
        return updateRemapResult;
      }
      instanceRemapUploadSignatures_[frameSlot] = remapSignature;
    }
  } else if (frameSlot < instanceRemapUploadSignatures_.size()) {
    instanceRemapUploadSignatures_[frameSlot] =
        std::numeric_limits<uint64_t>::max();
  }

  passBufferDependencies_.clear();
  appendUniqueBufferDependency(passBufferDependencies_, sceneGpu.buffer);
  appendUniqueBufferDependency(passBufferDependencies_, instanceMatricesBuffer);
  appendUniqueBufferDependency(passBufferDependencies_, instanceRemapBuffer);
  appendAnimatedGeometryDependencies(passBufferDependencies_,
                                     animationSceneData);
  appendUniqueBufferDependency(
      passBufferDependencies_, shadowMeshletCounterBuffer,
      RenderGraphAccessMode::Read | RenderGraphAccessMode::Write);
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
    const uint32_t staticDrawCount = cascadeStaticCasterCounts[cascadeIndex];
    const uint32_t dynamicDrawCount = cascadeDynamicCasterCounts[cascadeIndex];
    cascadeDrawCounts_[cascadeIndex] = staticDrawCount + dynamicDrawCount;
    cascadeDynamicDrawCounts_[cascadeIndex] = dynamicDrawCount;
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
    cascadeDebug.staticBatchFullEmitCount =
        cascadeStaticBatchFullEmitCounts[cascadeIndex];
    cascadeDebug.staticLightGridQueryCellCount =
        cascadeStaticLightGridQueryCellCounts[cascadeIndex];
    cascadeDebug.staticLightGridCandidateCount =
        cascadeStaticLightGridCandidateCounts[cascadeIndex];
    cascadeDebug.staticLightGridFallbackScanCount =
        cascadeStaticLightGridFallbackScanCounts[cascadeIndex];
  }
  uint32_t totalDraws = 0u;
  uint32_t totalCulledDraws = 0u;
  uint32_t totalStaticBatchFullEmitCount = 0u;
  uint32_t totalStaticLightGridFallbackScanCount = 0u;
  uint32_t totalStaticLightGridQueryCellCount = 0u;
  uint32_t totalStaticLightGridCandidateCount = 0u;
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
    const TextureHandle currentShadowDepthTexture =
        shadowDepthTextures_[cascadeIndex];
    const bool previousTextureMatches =
        previousValid &&
        currentShadowDepthTexture.index ==
            previousState.shadowDepthTexture.index &&
        currentShadowDepthTexture.generation ==
            previousState.shadowDepthTexture.generation;
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
        previousTextureMatches && !biasChanged &&
        cachedRenderedFitContainsCurrent && !needsAdaptiveRefresh;
    if (reuseStaticOnlyCascadePass_[cascadeIndex]) {
      const shadow_detail::DirectionalShadowFit rawFit =
          currentRawShadowFits_[cascadeIndex];
      const shadow_detail::DirectionalShadowFit reuseFit =
          makeStaticOnlyReuseFitForCurrentFrame(previousState.renderedFit,
                                                rawFit);
      writeShadowCascadeFit(reuseFit, cascadeIndex, settings.shadow,
                            frame.sharedResources.shadowCompareSamplerId,
                            frame.sharedResources.shadowRawSamplerId,
                            shadowDepthTextures_[cascadeIndex], gpu_,
                            shadowFrameCpuData_, shadowDebugFrameData_);
      currentState = previousState;
      currentState.renderedFit = reuseFit;
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
    totalStaticBatchFullEmitCount +=
        cascadeStaticBatchFullEmitCounts[cascadeIndex];
    totalStaticLightGridFallbackScanCount +=
        cascadeStaticLightGridFallbackScanCounts[cascadeIndex];
    totalStaticLightGridQueryCellCount +=
        cascadeStaticLightGridQueryCellCounts[cascadeIndex];
    totalStaticLightGridCandidateCount +=
        cascadeStaticLightGridCandidateCounts[cascadeIndex];
  }
  for (uint32_t cascadeIndex = cascadeCount; cascadeIndex < kMaxShadowCascades;
       ++cascadeIndex) {
    reuseStaticOnlyCascadePass_[cascadeIndex] = false;
    staticOnlyCascadeContentSignatures_[cascadeIndex] = 0u;
    shadowDebugFrameData_.cascades[cascadeIndex].staticOnlyReuseStatus =
        ShadowCascadeDebugFrameData::StaticOnlyReuseStatus::None;
  }
  float minCascadeTexelWorldSize = std::numeric_limits<float>::max();
  float maxCascadeTexelWorldSize = 0.0f;
  float totalCascadeTexelWorldSize = 0.0f;
  uint32_t validCascadeTexelWorldSizeCount = 0u;
  for (uint32_t cascadeIndex = 0u; cascadeIndex < cascadeCount;
       ++cascadeIndex) {
    const float texelWorldSize =
        shadowDebugFrameData_.cascades[cascadeIndex].texelWorldSize;
    if (!std::isfinite(texelWorldSize) || texelWorldSize <= 0.0f) {
      continue;
    }
    minCascadeTexelWorldSize =
        std::min(minCascadeTexelWorldSize, texelWorldSize);
    maxCascadeTexelWorldSize =
        std::max(maxCascadeTexelWorldSize, texelWorldSize);
    totalCascadeTexelWorldSize += texelWorldSize;
    ++validCascadeTexelWorldSizeCount;
  }
  if (validCascadeTexelWorldSizeCount == 0u) {
    minCascadeTexelWorldSize = 0.0f;
  }
  frame.metrics.shadow.cascadeCount = cascadeCount;
  frame.metrics.shadow.shadowMapSize = shadowMapSize_;
  frame.metrics.shadow.totalDraws = totalDraws;
  frame.metrics.shadow.totalCulledDraws = totalCulledDraws;
  frame.metrics.visibility.shadowCpuRejected = totalCulledDraws;
  frame.metrics.visibility.shadowCpuCandidates = saturateToU32(
      static_cast<size_t>(totalDraws) + static_cast<size_t>(totalCulledDraws));
  frame.metrics.shadow.totalIndexCountEstimate =
      saturateToU32(actualTotalIndexCountEstimate);
  frame.metrics.shadow.staticCasterEntries =
      saturateToU32(staticShadowCasterCache_.size()) -
      std::min(saturateToU32(staticShadowCasterCache_.size()),
               overriddenStaticTemplateCount);
  frame.metrics.shadow.dynamicCasterEntries =
      saturateToU32(frameDynamicTemplateIndices.size());
  frame.metrics.shadow.staticCacheReused = staticCacheReused ? 1u : 0u;
  frame.metrics.shadow.staticBatchTemplateCount =
      saturateToU32(staticShadowBatchTemplates_.size());
  frame.metrics.shadow.shadowBatchEntryCount = emittedShadowBatchEntryCount;
  frame.metrics.shadow.shadowMeshletDispatchCount =
      emittedShadowMeshletDispatchCount;
  frame.metrics.shadow.shadowMeshletTaskGroupCount =
      emittedShadowMeshletTaskGroupCount;
  frame.metrics.visibility.shadowMeshletCandidates =
      saturateToU32(emittedShadowMeshletCandidateCount);
  frame.metrics.shadow.shadowInstanceRemapCount =
      emittedShadowInstanceRemapCount;
  frame.metrics.shadow.staticBatchFullEmitCount = totalStaticBatchFullEmitCount;
  frame.metrics.shadow.staticLightGridQueryCount = staticLightGridQueryCount;
  frame.metrics.shadow.staticLightGridFallbackScanCount =
      totalStaticLightGridFallbackScanCount;
  frame.metrics.shadow.staticLightGridQueryCellCount =
      totalStaticLightGridQueryCellCount;
  frame.metrics.shadow.staticLightGridCandidateCount =
      totalStaticLightGridCandidateCount;
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
      shadowDepthTextureBytesPerPixel(shadowDepthPipelineFormat_);
  frame.metrics.shadow.minCascadeTexelWorldSize = minCascadeTexelWorldSize;
  frame.metrics.shadow.maxCascadeTexelWorldSize = maxCascadeTexelWorldSize;
  frame.metrics.shadow.averageCascadeTexelWorldSize =
      validCascadeTexelWorldSizeCount > 0u
          ? totalCascadeTexelWorldSize /
                static_cast<float>(validCascadeTexelWorldSizeCount)
          : 0.0f;
  frame.metrics.shadow.farCascadeTexelWorldSize =
      cascadeCount > 0u
          ? shadowDebugFrameData_.cascades[cascadeCount - 1u].texelWorldSize
          : 0.0f;
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
    MeshletPipelineHandle shadowMeshletPipeline{};
    MeshletPipelineHandle shadowMeshletDoubleSidedPipeline{};
    MeshletPipelineHandle shadowMeshletAlphaPipeline{};
    MeshletPipelineHandle shadowMeshletAlphaDoubleSidedPipeline{};
  } signatureData{
      .shadowPipeline = shadowPipelineHandle_,
      .shadowDoubleSidedPipeline = shadowDoubleSidedPipelineHandle_,
      .shadowAlphaPipeline = shadowAlphaPipelineHandle_,
      .shadowAlphaDoubleSidedPipeline = shadowAlphaDoubleSidedPipelineHandle_,
      .shadowMeshletPipeline = shadowMeshletPipelineHandle_,
      .shadowMeshletDoubleSidedPipeline =
          shadowMeshletDoubleSidedPipelineHandle_,
      .shadowMeshletAlphaPipeline = shadowMeshletAlphaPipelineHandle_,
      .shadowMeshletAlphaDoubleSidedPipeline =
          shadowMeshletAlphaDoubleSidedPipelineHandle_,
  };
  return hashBytes(
      std::as_bytes(std::span<const SignatureData>(&signatureData, 1u)));
}

void ShadowRenderer::invalidateStaticShadowCasterCache() noexcept {
  staticShadowCasterCache_.clear();
  staticShadowBatchTemplates_.clear();
  staticShadowBatchIndexMap_.clear();
  staticShadowBatchInstanceIndices_.clear();
  staticShadowCasterDrawBuffers_.clear();
  staticShadowCasterFitPoints_.clear();
  staticShadowCasterLightSpaceBounds_.clear();
  staticShadowBatchLightSpaceBounds_.clear();
  staticShadowCasterLightGridCells_.clear();
  staticShadowCasterLightGridEntries_.clear();
  staticShadowCasterLargeLightGridEntries_.clear();
  staticShadowBatchLightGridCells_.clear();
  staticShadowBatchLightGridEntries_.clear();
  staticShadowBatchLargeLightGridEntries_.clear();
  staticShadowCasterLightGridQueryMarks_.clear();
  staticShadowCasterLightGridQueryEntries_.clear();
  staticShadowCasterLightGridQueryMarker_ = 1u;
  staticShadowCasterLightGrid_ = {};
  staticShadowBatchLightGrid_ = {};
  staticShadowCasterBoundsMin_ = glm::vec3(std::numeric_limits<float>::max());
  staticShadowCasterBoundsMax_ =
      glm::vec3(std::numeric_limits<float>::lowest());
  staticShadowCasterLightSpaceBoundsMin_ =
      glm::vec3(std::numeric_limits<float>::max());
  staticShadowCasterLightSpaceBoundsMax_ =
      glm::vec3(std::numeric_limits<float>::lowest());
  staticShadowCasterCacheContentSignature_ = 0u;
  staticShadowCasterCacheIndexCountEstimate_ = 0u;
  hasStaticShadowCasterBounds_ = false;
  hasStaticShadowCasterLightDepthBounds_ = false;
  hasStaticShadowCasterLightSpaceBounds_ = false;
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
  for (DynamicBufferSlot &slot : shadowMeshletCounterRing_) {
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
  shadowMeshletCounterRing_.clear();
  sdsmReduceResultRing_.clear();
  instanceDataRingUploadVersions_.clear();
  instanceRemapUploadSignatures_.clear();
  shadowFrameUploadSignatures_.clear();
  shadowMeshletCounterRingPublishedFrames_.clear();
  shadowMeshletCounterClear_.clear();
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
  if (nuri::isValid(shadowOpaqueVertexShader_)) {
    gpu_.destroyShaderModule(shadowOpaqueVertexShader_);
    shadowOpaqueVertexShader_ = {};
  }
  if (nuri::isValid(shadowMeshletTaskShader_)) {
    gpu_.destroyShaderModule(shadowMeshletTaskShader_);
    shadowMeshletTaskShader_ = {};
  }
  if (nuri::isValid(shadowMeshletMeshShader_)) {
    gpu_.destroyShaderModule(shadowMeshletMeshShader_);
    shadowMeshletMeshShader_ = {};
  }
  if (nuri::isValid(shadowMeshletDepthFragmentShader_)) {
    gpu_.destroyShaderModule(shadowMeshletDepthFragmentShader_);
    shadowMeshletDepthFragmentShader_ = {};
  }
  if (nuri::isValid(shadowMeshletDepthAlphaFragmentShader_)) {
    gpu_.destroyShaderModule(shadowMeshletDepthAlphaFragmentShader_);
    shadowMeshletDepthAlphaFragmentShader_ = {};
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
  shadowOpaqueShader_.reset();
  shadowMeshletShader_.reset();
  depthShader_.reset();
  depthAlphaShader_.reset();
  sdsmReduceShader_.reset();
  sdsmHistogramReduceShader_.reset();
  initialized_ = false;
  shadowMeshletPipelineUnsupported_ = false;
}

void ShadowRenderer::destroyShadowDepthPipelineState() {
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
  destroyShadowMeshletPipelineState();
  shadowDepthPipelineFormat_ = Format::Count;
}

void ShadowRenderer::destroyShadowMeshletPipelineState() {
  if (nuri::isValid(shadowMeshletAlphaDoubleSidedPipelineHandle_)) {
    gpu_.destroyMeshletPipeline(shadowMeshletAlphaDoubleSidedPipelineHandle_);
    shadowMeshletAlphaDoubleSidedPipelineHandle_ = {};
  }
  if (nuri::isValid(shadowMeshletAlphaPipelineHandle_)) {
    gpu_.destroyMeshletPipeline(shadowMeshletAlphaPipelineHandle_);
    shadowMeshletAlphaPipelineHandle_ = {};
  }
  if (nuri::isValid(shadowMeshletDoubleSidedPipelineHandle_)) {
    gpu_.destroyMeshletPipeline(shadowMeshletDoubleSidedPipelineHandle_);
    shadowMeshletDoubleSidedPipelineHandle_ = {};
  }
  if (nuri::isValid(shadowMeshletPipelineHandle_)) {
    gpu_.destroyMeshletPipeline(shadowMeshletPipelineHandle_);
    shadowMeshletPipelineHandle_ = {};
  }
  shadowMeshletDepthPipelineFormat_ = Format::Count;
}

void ShadowRenderer::destroyPipelineState() {
  if (nuri::isValid(previewPipelineHandle_)) {
    gpu_.destroyRenderPipeline(previewPipelineHandle_);
    previewPipelineHandle_ = {};
  }
  destroyShadowDepthPipelineState();
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
  staticShadowCasterDrawBuffers_.clear();
  invalidateStaticShadowCasterCache();
  invalidateReusableStaticOnlyCascadeCache();
  resetCascadeStabilizationHistory();
  instanceMatrices_.clear();
  instanceRemap_.clear();
  passTextureDependencies_.clear();
  passDependencyBuffers_.clear();
  passDependencyBufferAccessModes_.clear();
  passDependencyBufferBindings_.clear();
  passDependencyTextureBindings_.clear();
  preResolvedDrawBuffers_.clear();
  preResolvedDrawBufferIds_.clear();
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
    cascadeMeshletPushConstants_[cascadeIndex].clear();
    cascadeMeshDispatchItems_[cascadeIndex].clear();
    cascadeMeshDispatchDependencyBuffers_[cascadeIndex].clear();
  }
  passBufferDependencies_.clear();
  passDependencyBuffers_.clear();
  passDependencyBufferAccessModes_.clear();
  passDependencyBufferBindings_.clear();
  passDependencyTextureBindings_.clear();
  preResolvedDrawBuffers_.clear();
  preResolvedDrawBufferIds_.clear();
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
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_COPY);
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

  const RenderSettings &frameSettings = renderSettingsOrDefault(frame);
  RenderSettings::ShadowSettings settings = frameSettings.shadow;
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
          timingReport.shadowSourceFrameIndex;
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
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_COPY);
  resetFrameBuildState();
  frame.metrics.visibility.shadowMeshletCandidates = 0u;
  frame.metrics.visibility.shadowMeshletReadbackAvailable = 0u;
  frame.metrics.visibility.shadowMeshletReadbackSourceFrame = 0u;
  frame.metrics.visibility.shadowMeshletReadbackStaleFrameCount = 0u;
  frame.metrics.visibility.shadowMeshletReadbackErrorCount = 0u;
  frame.metrics.visibility.shadowMeshletRejectedBounds = 0u;
  frame.metrics.visibility.shadowCpuCandidates = 0u;
  frame.metrics.visibility.shadowCpuRejected = 0u;

  const RenderSettings &frameSettings = renderSettingsOrDefault(frame);
  RenderSettings::ShadowSettings settings = frameSettings.shadow;
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
  readLatestShadowMeshletCounterReadback(frame);
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
      updateShadowFrameData(frame, settings, settings.shadowMapSize,
                            frameSettings.opaque.forcedMeshLod);
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
  logShadowVisibilityCounters(frame);
  return Result<bool, std::string>::makeResult(true);
}

bool ShadowRenderer::hasPreparedShadowDepthPasses() const noexcept {
  return hasPreparedShadowDepthPasses_;
}

Result<bool, std::string>
ShadowRenderer::appendShadowDepthPasses(RenderFrameContext &frame,
                                        RenderGraphBuilder &graph) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_DRAW);
  if (!hasPreparedShadowDepthPasses_ || activeCascadeCount_ == 0u ||
      !nuri::isValid(shadowDepthTextures_[0])) {
    return Result<bool, std::string>::makeResult(true);
  }
  const AnimationSceneFrameData *animationSceneData =
      resolveAnimationSceneFrameData(frame);

  passDependencyBuffers_.clear();
  passDependencyBufferAccessModes_.clear();
  passDependencyBufferBindings_.clear();
  passDependencyTextureBindings_.clear();
  preResolvedDrawBufferIds_.clear();
  std::pmr::vector<TextureHandle> dependencyTextures(memory_);
  std::pmr::vector<RenderGraphAccessMode> dependencyTextureAccessModes(memory_);
  splitDependencies(
      std::span<const BufferDependency>(passBufferDependencies_.data(),
                                        passBufferDependencies_.size()),
      passDependencyBuffers_, passDependencyBufferAccessModes_);
  splitDependencies(
      std::span<const TextureDependency>(passTextureDependencies_.data(),
                                         passTextureDependencies_.size()),
      dependencyTextures, dependencyTextureAccessModes);

  bool hasNonReusedShadowCascade = false;
  for (uint32_t cascadeIndex = 0u; cascadeIndex < activeCascadeCount_;
       ++cascadeIndex) {
    hasNonReusedShadowCascade =
        hasNonReusedShadowCascade ||
        (nuri::isValid(shadowDepthTextures_[cascadeIndex]) &&
         !reuseStaticOnlyCascadePass_[cascadeIndex]);
  }
  if (hasNonReusedShadowCascade) {
    passDependencyBufferBindings_.reserve(passDependencyBuffers_.size());
    for (size_t i = 0; i < passDependencyBuffers_.size(); ++i) {
      const BufferHandle dependency = passDependencyBuffers_[i];
      if (!nuri::isValid(dependency)) {
        continue;
      }
      auto importResult =
          graph.importBuffer(dependency, kShadowPassDependencyBufferLabel);
      if (importResult.hasError()) {
        return Result<bool, std::string>::makeError(importResult.error());
      }
      passDependencyBufferBindings_.push_back(
          RenderGraphPreparedDependencyBufferBinding{
              .dependencyIndex = static_cast<uint32_t>(i),
              .buffer = importResult.value(),
              .mode = passDependencyBufferAccessModes_[i],
          });
    }

    passDependencyTextureBindings_.reserve(dependencyTextures.size());
    for (size_t i = 0; i < dependencyTextures.size(); ++i) {
      const TextureHandle dependency = dependencyTextures[i];
      if (!nuri::isValid(dependency)) {
        continue;
      }
      auto importResult =
          graph.importTexture(dependency, kShadowPassDependencyTextureLabel);
      if (importResult.hasError()) {
        return Result<bool, std::string>::makeError(importResult.error());
      }
      passDependencyTextureBindings_.push_back(
          RenderGraphPreparedDependencyTextureBinding{
              .texture = importResult.value(),
              .mode = dependencyTextureAccessModes[i],
          });
    }

    preResolvedDrawBufferIds_.reserve(preResolvedDrawBuffers_.size());
    for (const BufferHandle buffer : preResolvedDrawBuffers_) {
      if (!nuri::isValid(buffer)) {
        continue;
      }
      auto importResult =
          graph.importBuffer(buffer, kShadowPassDrawBufferLabel);
      if (importResult.hasError()) {
        return Result<bool, std::string>::makeError(importResult.error());
      }
      preResolvedDrawBufferIds_.push_back(importResult.value());
    }
  }

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

    auto depthImportResult = graph.importTexture(
        shadowDepthTexture, shadowCascadeTextureImportName(cascadeIndex));
    if (depthImportResult.hasError()) {
      return Result<bool, std::string>::makeError(depthImportResult.error());
    }
    frame.sharedResources.shadowCascadeGraphTextures[cascadeIndex] =
        depthImportResult.value();
    publishRequestedCapture(
        frame, gpu_, shadowCascadeCaptureName(cascadeIndex), shadowDepthTexture,
        RenderCaptureValueKind::ShadowDepth,
        RenderCaptureLifetimeClass::FeaturePersistentTexture, "linear_depth",
        "shadow_depth", shadowCascadePassLabel(cascadeIndex));

    const std::string_view passLabel = shadowCascadePassLabel(cascadeIndex);
    const std::span<const ComputeDispatchItem> preDispatches =
        reuseStaticOnlyCascade
            ? std::span<const ComputeDispatchItem>()
            : shadowCascadePreDispatches(animationSceneData, cascadeIndex);
    const std::span<const BufferHandle> passDependencyBuffers =
        reuseStaticOnlyCascade
            ? std::span<const BufferHandle>()
            : std::span<const BufferHandle>(passDependencyBuffers_.data(),
                                            passDependencyBuffers_.size());
    const std::span<const RenderGraphAccessMode>
        passDependencyBufferAccessModes =
            reuseStaticOnlyCascade
                ? std::span<const RenderGraphAccessMode>()
                : std::span<const RenderGraphAccessMode>(
                      passDependencyBufferAccessModes_.data(),
                      passDependencyBufferAccessModes_.size());
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
    const std::span<const MeshDispatchItem> passMeshDispatches =
        reuseStaticOnlyCascade
            ? std::span<const MeshDispatchItem>()
            : std::span<const MeshDispatchItem>(
                  cascadeMeshDispatchItems_[cascadeIndex].data(),
                  cascadeMeshDispatchItems_[cascadeIndex].size());
    const std::span<const BufferHandle> passPreResolvedDrawBuffers =
        reuseStaticOnlyCascade
            ? std::span<const BufferHandle>()
            : std::span<const BufferHandle>(preResolvedDrawBuffers_.data(),
                                            preResolvedDrawBuffers_.size());
    const std::span<const RenderGraphBufferId> passPreResolvedDrawBufferIds =
        reuseStaticOnlyCascade ? std::span<const RenderGraphBufferId>()
                               : std::span<const RenderGraphBufferId>(
                                     preResolvedDrawBufferIds_.data(),
                                     preResolvedDrawBufferIds_.size());

    Result<RenderGraphPassId, std::string> passResult =
        Result<RenderGraphPassId, std::string>::makeError(
            "ShadowRenderer::appendShadowDepthPasses: pass was not built");
    if (preDispatches.empty()) {
      const std::span<const RenderGraphPreparedDependencyBufferBinding>
          preparedDependencyBufferBindings =
              reuseStaticOnlyCascade
                  ? std::span<
                        const RenderGraphPreparedDependencyBufferBinding>()
                  : std::span<const RenderGraphPreparedDependencyBufferBinding>(
                        passDependencyBufferBindings_.data(),
                        passDependencyBufferBindings_.size());
      const std::span<const RenderGraphPreparedDependencyTextureBinding>
          preparedDependencyTextureBindings =
              reuseStaticOnlyCascade
                  ? std::span<
                        const RenderGraphPreparedDependencyTextureBinding>()
                  : std::span<
                        const RenderGraphPreparedDependencyTextureBinding>(
                        passDependencyTextureBindings_.data(),
                        passDependencyTextureBindings_.size());
      const RenderGraphPreparedGraphicsPassDesc desc{
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
          .preDispatches = {},
          .dependencyBuffers = passDependencyBuffers,
          .draws = passDraws,
          .meshDispatches = passMeshDispatches,
          .dependencyBufferBindings = preparedDependencyBufferBindings,
          .dependencyTextureBindings = preparedDependencyTextureBindings,
          .preDispatchDependencyBindings = {},
          .drawBufferBindings = {},
          .drawBuffersPreResolved = true,
          .preResolvedDrawBuffers = {},
          .preResolvedDrawBufferIds = passPreResolvedDrawBufferIds,
          .gpuTimingScope = GpuTimingScope::ShadowDepth,
          .debugLabel = passLabel,
          .debugColor = kShadowPassDebugColor,
          .markColorAsFrameOutput = false,
          .markImplicitOutputSideEffect = false,
          .borrowPayload = true,
      };
      passResult = graph.addPreparedGraphicsPass(desc);
    } else {
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
          .meshDispatches = passMeshDispatches,
          .drawBuffersPreResolved = true,
          .preResolvedDrawBuffers = passPreResolvedDrawBuffers,
          .gpuTimingScope = GpuTimingScope::ShadowDepth,
          .debugLabel = passLabel,
          .debugColor = kShadowPassDebugColor,
          .markColorAsFrameOutput = false,
          .markImplicitOutputSideEffect = false,
          .borrowPayload = false,
      };
      passResult = graph.addGraphicsPass(desc);
    }
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
              .shadowDepthTexture = shadowDepthTexture,
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
    publishRequestedCapture(
        frame, gpu_, "shadow_preview", shadowDebugPreviewTexture_,
        RenderCaptureValueKind::DebugPreview,
        RenderCaptureLifetimeClass::FeaturePersistentTexture, "display_sdr",
        "debug_preview", "Shadow Depth Preview Pass");

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
        .borrowPayload = true,
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
