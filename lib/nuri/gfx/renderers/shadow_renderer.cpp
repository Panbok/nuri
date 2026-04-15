#include "nuri/pch.h"

#include "nuri/gfx/renderers/shadow_renderer.h"

#include "nuri/core/log.h"
#include "nuri/gfx/renderers/detail/renderable_material_resolution.h"
#include "nuri/resources/gpu/resource_manager.h"

namespace nuri {
namespace {

constexpr uint32_t kShadowPassDebugColor = 0xff5a7dffu;
constexpr uint32_t kShadowMeshDebugColor = 0xff5a7dffu;
constexpr uint32_t kShadowPreviewPassDebugColor = 0xff7d5affu;
constexpr std::string_view kShadowPassLabel = "ShadowDepthPass";
constexpr std::string_view kShadowMeshLabel = "ShadowCasterMesh";
constexpr std::string_view kShadowPreviewPassLabel = "ShadowDepthPreviewPass";
constexpr std::string_view kShadowPreviewDrawLabel = "ShadowDepthPreview";
constexpr uint64_t kFnvOffsetBasis64 = 14695981039346656037ull;
constexpr uint64_t kFnvPrime64 = 1099511628211ull;
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

layout(push_constant) uniform PreviewPushConstants {
  uint sourceTexId;
  float depthScale;
  float depthBias;
} pc;

float fetchDepth() {
  ivec2 sourceSize = textureSize(nonuniformEXT(kTextures2D[pc.sourceTexId]), 0);
  ivec2 texelCoord = clamp(ivec2(gl_FragCoord.xy), ivec2(0), sourceSize - 1);
  return texelFetch(nonuniformEXT(kTextures2D[pc.sourceTexId]), texelCoord, 0).r;
}

void main() {
  float depth = fetchDepth();
  float preview = clamp(depth * pc.depthScale + pc.depthBias, 0.0, 1.0);
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

[[nodiscard]] bool isSameBufferHandle(BufferHandle lhs, BufferHandle rhs) {
  return lhs.index == rhs.index && lhs.generation == rhs.generation;
}

void appendUniqueBuffer(std::pmr::vector<BufferHandle> &handles,
                        BufferHandle handle) {
  if (!nuri::isValid(handle)) {
    return;
  }
  for (const BufferHandle existing : handles) {
    if (isSameBufferHandle(existing, handle)) {
      return;
    }
  }
  handles.push_back(handle);
}

[[nodiscard]] glm::vec3 normalizeSafe(glm::vec3 value, glm::vec3 fallback) {
  const float length = glm::length(value);
  if (!std::isfinite(length) || length <= 1.0e-6f) {
    return fallback;
  }
  return value / length;
}

[[nodiscard]] glm::vec3 chooseLightUp(glm::vec3 lightDirection) {
  const glm::vec3 worldUp = std::abs(lightDirection.y) < 0.99f
                                ? glm::vec3(0.0f, 1.0f, 0.0f)
                                : glm::vec3(0.0f, 0.0f, 1.0f);
  const glm::vec3 right = glm::cross(worldUp, lightDirection);
  if (glm::dot(right, right) <= 1.0e-8f) {
    return glm::vec3(1.0f, 0.0f, 0.0f);
  }
  return worldUp;
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

[[nodiscard]] glm::vec3 cameraRightFromView(const glm::mat4 &view) {
  const glm::mat4 invView = glm::inverse(view);
  return normalizeSafe(glm::vec3(invView[0]), glm::vec3(1.0f, 0.0f, 0.0f));
}

[[nodiscard]] glm::vec3 cameraUpFromView(const glm::mat4 &view) {
  const glm::mat4 invView = glm::inverse(view);
  return normalizeSafe(glm::vec3(invView[1]), glm::vec3(0.0f, 1.0f, 0.0f));
}

[[nodiscard]] glm::vec3 cameraForwardFromView(const glm::mat4 &view) {
  const glm::mat4 invView = glm::inverse(view);
  return normalizeSafe(-glm::vec3(invView[2]), glm::vec3(0.0f, 0.0f, -1.0f));
}

[[nodiscard]] std::array<glm::vec3, 8>
computeShadowCameraSliceCorners(const CameraFrameState &camera,
                                float requestedFar) {
  const glm::vec3 cameraPos = glm::vec3(camera.cameraPos);
  const glm::vec3 forward = cameraForwardFromView(camera.view);
  const glm::vec3 right = cameraRightFromView(camera.view);
  const glm::vec3 up = cameraUpFromView(camera.view);
  const float nearPlane = std::max(camera.nearPlane, 0.01f);
  const float farPlane = std::max(nearPlane + 0.01f, requestedFar);
  std::array<glm::vec3, 8> corners{};

  if (camera.projectionType == ProjectionType::Orthographic) {
    const float halfHeight = 0.5f * std::max(camera.orthoHeight, 0.01f);
    const float halfWidth = halfHeight * std::max(camera.aspectRatio, 0.01f);
    const glm::vec3 nearCenter = cameraPos + forward * nearPlane;
    const glm::vec3 farCenter = cameraPos + forward * farPlane;
    const glm::vec3 dx = right * halfWidth;
    const glm::vec3 dy = up * halfHeight;
    corners[0] = nearCenter - dx - dy;
    corners[1] = nearCenter + dx - dy;
    corners[2] = nearCenter + dx + dy;
    corners[3] = nearCenter - dx + dy;
    corners[4] = farCenter - dx - dy;
    corners[5] = farCenter + dx - dy;
    corners[6] = farCenter + dx + dy;
    corners[7] = farCenter - dx + dy;
    return corners;
  }

  const float tanHalfFov = std::tan(std::max(camera.fovYRadians, 0.01f) * 0.5f);
  const float nearHalfHeight = tanHalfFov * nearPlane;
  const float nearHalfWidth =
      nearHalfHeight * std::max(camera.aspectRatio, 0.01f);
  const float farHalfHeight = tanHalfFov * farPlane;
  const float farHalfWidth =
      farHalfHeight * std::max(camera.aspectRatio, 0.01f);
  const glm::vec3 nearCenter = cameraPos + forward * nearPlane;
  const glm::vec3 farCenter = cameraPos + forward * farPlane;
  const glm::vec3 nearDx = right * nearHalfWidth;
  const glm::vec3 nearDy = up * nearHalfHeight;
  const glm::vec3 farDx = right * farHalfWidth;
  const glm::vec3 farDy = up * farHalfHeight;

  corners[0] = nearCenter - nearDx - nearDy;
  corners[1] = nearCenter + nearDx - nearDy;
  corners[2] = nearCenter + nearDx + nearDy;
  corners[3] = nearCenter - nearDx + nearDy;
  corners[4] = farCenter - farDx - farDy;
  corners[5] = farCenter + farDx - farDy;
  corners[6] = farCenter + farDx + farDy;
  corners[7] = farCenter - farDx + farDy;
  return corners;
}

[[nodiscard]] std::array<glm::vec3, 8>
boundingBoxCorners(const BoundingBox &bounds) {
  return {
      glm::vec3(bounds.min_.x, bounds.min_.y, bounds.min_.z),
      glm::vec3(bounds.max_.x, bounds.min_.y, bounds.min_.z),
      glm::vec3(bounds.max_.x, bounds.max_.y, bounds.min_.z),
      glm::vec3(bounds.min_.x, bounds.max_.y, bounds.min_.z),
      glm::vec3(bounds.min_.x, bounds.min_.y, bounds.max_.z),
      glm::vec3(bounds.max_.x, bounds.min_.y, bounds.max_.z),
      glm::vec3(bounds.max_.x, bounds.max_.y, bounds.max_.z),
      glm::vec3(bounds.min_.x, bounds.max_.y, bounds.max_.z),
  };
}

[[nodiscard]] uint32_t saturateToU32(size_t value) {
  return static_cast<uint32_t>(
      std::min(value, size_t(std::numeric_limits<uint32_t>::max())));
}

} // namespace

ShadowRenderer::ShadowRenderer(GPUDevice &gpu,
                               std::pmr::memory_resource *memory)
    : ShadowRenderer(gpu, ShadowRendererConfig{}, memory) {}

ShadowRenderer::ShadowRenderer(GPUDevice &gpu, ShadowRendererConfig config,
                               std::pmr::memory_resource *memory)
    : gpu_(gpu), config_(std::move(config)),
      memory_(resolveMemoryResource(memory)), instanceMatricesRing_(memory_),
      instanceRemapRing_(memory_), shadowFrameRing_(memory_),
      instanceDataRingUploadVersions_(memory_),
      shadowFrameUploadSignatures_(memory_), meshDrawTemplates_(memory_),
      instanceMatrices_(memory_), instanceRemap_(memory_),
      drawPushConstants_(memory_), drawItems_(memory_),
      passDependencyBuffers_(memory_), previewDependencyTextures_(memory_),
      previewDependencyTextureAccessModes_(memory_) {}

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
  if (!shadowShader_ || !depthShader_) {
    return Result<bool, std::string>::makeError(
        "ShadowRenderer::createShaders: failed to create shader wrappers");
  }

  const std::filesystem::path shadowVertexPath =
      config_.shaderBasePath / "shadow_depth.vert";
  const std::filesystem::path depthFragmentPath =
      config_.shaderBasePath / "opaque_depth.frag";

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

  shadowVertexShader_ = vertexResult.value();
  depthFragmentShader_ = fragmentResult.value();
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
  const bool depthValid = nuri::isValid(shadowDepthTexture_) &&
                          shadowMapSize_ == settings.shadowMapSize &&
                          gpu_.getTextureFormat(shadowDepthTexture_) ==
                              kDefaultShadowMapDepthFormat &&
                          nuri::isValid(rawDepthSampler_);
  const bool previewValid =
      !previewEnabled || (nuri::isValid(shadowDebugPreviewTexture_) &&
                          gpu_.getTextureFormat(shadowDebugPreviewTexture_) ==
                              Format::RGBA8_UNORM);
  if (depthValid && previewValid) {
    return Result<bool, std::string>::makeResult(true);
  }

  destroyShadowResources();

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
  auto textureResult =
      gpu_.createTexture(shadowDesc, "shadow_depth_map_phase1");
  if (textureResult.hasError()) {
    return Result<bool, std::string>::makeError(textureResult.error());
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
    gpu_.destroyTexture(textureResult.value());
    return Result<bool, std::string>::makeError(samplerResult.error());
  }

  shadowDepthTexture_ = textureResult.value();
  rawDepthSampler_ = samplerResult.value();
  if (previewEnabled) {
    const TextureDesc previewDesc{
        .type = TextureType::Texture2D,
        .format = Format::RGBA8_UNORM,
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
    auto previewResult =
        gpu_.createTexture(previewDesc, "shadow_depth_preview_phase1");
    if (previewResult.hasError()) {
      destroyShadowResources();
      return Result<bool, std::string>::makeError(previewResult.error());
    }
    shadowDebugPreviewTexture_ = previewResult.value();
  }
  shadowMapSize_ = settings.shadowMapSize;
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

Result<bool, std::string>
ShadowRenderer::ensureInstanceMatricesRingCapacity(size_t requiredBytes) {
  bool needsResize = false;
  bool hasLiveBuffers = false;
  for (const DynamicBufferSlot &slot : instanceMatricesRing_) {
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
  for (size_t i = 0; i < instanceMatricesRing_.size(); ++i) {
    DynamicBufferSlot &slot = instanceMatricesRing_[i];
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
                                       "shadow_instance_matrices");
    if (bufferResult.hasError()) {
      return Result<bool, std::string>::makeError(bufferResult.error());
    }
    slot.buffer = std::move(bufferResult.value());
    slot.capacityBytes = requiredBytes;
    instanceDataRingUploadVersions_[i] = std::numeric_limits<uint64_t>::max();
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
ShadowRenderer::ensureInstanceRemapRingCapacity(size_t requiredBytes) {
  bool needsResize = false;
  bool hasLiveBuffers = false;
  for (const DynamicBufferSlot &slot : instanceRemapRing_) {
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
  for (size_t i = 0; i < instanceRemapRing_.size(); ++i) {
    DynamicBufferSlot &slot = instanceRemapRing_[i];
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
                                       "shadow_instance_remap");
    if (bufferResult.hasError()) {
      return Result<bool, std::string>::makeError(bufferResult.error());
    }
    slot.buffer = std::move(bufferResult.value());
    slot.capacityBytes = requiredBytes;
    instanceDataRingUploadVersions_[i] = std::numeric_limits<uint64_t>::max();
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
ShadowRenderer::ensureShadowFrameRingCapacity(size_t requiredBytes) {
  bool needsResize = false;
  bool hasLiveBuffers = false;
  for (const DynamicBufferSlot &slot : shadowFrameRing_) {
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
  for (size_t i = 0; i < shadowFrameRing_.size(); ++i) {
    DynamicBufferSlot &slot = shadowFrameRing_[i];
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
                                       "shadow_frame_gpu_data");
    if (bufferResult.hasError()) {
      return Result<bool, std::string>::makeError(bufferResult.error());
    }
    slot.buffer = std::move(bufferResult.value());
    slot.capacityBytes = requiredBytes;
    shadowFrameUploadSignatures_[i] = std::numeric_limits<uint64_t>::max();
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
ShadowRenderer::rebuildSceneCache(const RenderScene &scene,
                                  const ResourceManager &resources,
                                  uint32_t materialCount) {
  (void)materialCount;
  meshDrawTemplates_.clear();

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
          materialRecord->desc.alphaMode != MaterialAlphaMode::Opaque) {
        continue;
      }

      meshDrawTemplates_.push_back(MeshDrawTemplate{
          .renderable = &renderable,
          .submesh = &submesh,
          .submeshIndex = submeshIndex,
          .instanceIndex = renderableIndex,
          .indexBuffer = geometry.indexBuffer,
          .indexBufferOffset = geometry.indexByteOffset,
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
      });
    }
  }

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> ShadowRenderer::updateShadowFrameData(
    RenderFrameContext &frame, const RenderSettings::ShadowSettings &settings,
    uint32_t shadowMapSize) {
  shadowFrameCpuData_ = {};
  shadowDebugFrameData_ = {};
  shadowDebugFrameData_.cascadeCount = 1u;
  shadowDebugFrameData_.rawSamplerId = frame.sharedResources.shadowRawSamplerId;
  shadowDebugFrameData_.compareSamplerId = kInvalidShadowBindlessIndex;
  shadowDebugFrameData_.cascades[0].texture = shadowDepthTexture_;
  shadowDebugFrameData_.cascades[0].textureBindlessId =
      gpu_.getTextureBindlessIndex(shadowDepthTexture_);

  if (frame.scene == nullptr) {
    frame.sharedResources.shadowDebugFrameData = shadowDebugFrameData_;
    return Result<bool, std::string>::makeResult(true);
  }

  const std::span<const DirectionalLightGpuData> directionalLights =
      frame.scene->packedDirectionalLights();
  const std::span<const LightId> directionalLightIds =
      frame.scene->packedDirectionalLightIds();
  if (directionalLights.empty() || directionalLightIds.empty()) {
    frame.sharedResources.shadowDebugFrameData = shadowDebugFrameData_;
    return Result<bool, std::string>::makeResult(true);
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

  const DirectionalLightGpuData &light = directionalLights[selectedLightIndex];
  const glm::vec3 lightDirection = normalizeSafe(
      glm::vec3(light.directionIlluminance), glm::vec3(0.0f, -1.0f, 0.0f));
  float effectiveFar = std::min(settings.maxDistance, frame.camera.farPlane);
  float largestCasterRadius = 0.0f;
  float farthestCasterDepth = frame.camera.nearPlane;
  bool hasCasterDepthRange = false;
  for (const MeshDrawTemplate &entry : meshDrawTemplates_) {
    if (entry.renderable == nullptr || entry.submesh == nullptr) {
      continue;
    }
    const BoundingBox worldBounds =
        entry.submesh->bounds.getTransformed(entry.renderable->modelMatrix);
    largestCasterRadius = std::max(largestCasterRadius,
                                   0.5f * glm::length(worldBounds.getSize()));

    const std::array<glm::vec3, 8> worldCorners =
        boundingBoxCorners(worldBounds);
    float boundsMinDepth = std::numeric_limits<float>::max();
    float boundsMaxDepth = std::numeric_limits<float>::lowest();
    for (const glm::vec3 corner : worldCorners) {
      const glm::vec3 viewSpace =
          glm::vec3(frame.camera.view * glm::vec4(corner, 1.0f));
      const float depth = -viewSpace.z;
      boundsMinDepth = std::min(boundsMinDepth, depth);
      boundsMaxDepth = std::max(boundsMaxDepth, depth);
    }
    if (boundsMaxDepth < frame.camera.nearPlane ||
        boundsMinDepth > effectiveFar) {
      continue;
    }
    farthestCasterDepth = std::max(farthestCasterDepth, boundsMaxDepth);
    hasCasterDepthRange = true;
  }
  if (hasCasterDepthRange) {
    const float focusMargin = std::max(largestCasterRadius * 2.0f, 2.0f);
    effectiveFar = std::clamp(farthestCasterDepth + focusMargin,
                              frame.camera.nearPlane + 0.01f, effectiveFar);
  }
  const std::array<glm::vec3, 8> frustumCorners =
      computeShadowCameraSliceCorners(frame.camera, effectiveFar);

  glm::vec3 frustumCenter(0.0f);
  for (const glm::vec3 corner : frustumCorners) {
    frustumCenter += corner;
  }
  frustumCenter /= static_cast<float>(frustumCorners.size());

  float radius = 0.0f;
  for (const glm::vec3 corner : frustumCorners) {
    radius = std::max(radius, glm::length(corner - frustumCenter));
  }
  radius = std::max(radius, 1.0f);

  const glm::vec3 lightUp = chooseLightUp(lightDirection);
  const glm::vec3 eye = frustumCenter - lightDirection * (radius * 2.0f);
  const glm::mat4 lightView = glm::lookAt(eye, frustumCenter, lightUp);

  glm::vec3 lightMin(std::numeric_limits<float>::max());
  glm::vec3 lightMax(std::numeric_limits<float>::lowest());
  for (const glm::vec3 corner : frustumCorners) {
    const glm::vec3 lightSpace = glm::vec3(lightView * glm::vec4(corner, 1.0f));
    lightMin = glm::min(lightMin, lightSpace);
    lightMax = glm::max(lightMax, lightSpace);
  }

  const float xyPadding = std::max(radius * 0.1f, 1.0f);
  lightMin.x -= xyPadding;
  lightMin.y -= xyPadding;
  lightMax.x += xyPadding;
  lightMax.y += xyPadding;

  const float depthPadding = std::max(radius, 10.0f);
  const float nearPlane = std::max(0.01f, -lightMax.z - depthPadding);
  const float farPlane =
      std::max(nearPlane + 0.01f, -lightMin.z + depthPadding);
  const glm::mat4 lightProj = glm::ortho(lightMin.x, lightMax.x, lightMin.y,
                                         lightMax.y, nearPlane, farPlane);
  const glm::mat4 lightViewProj = lightProj * lightView;
  const glm::vec3 debugLightBoundsMin(lightMin.x, lightMin.y, -farPlane);
  const glm::vec3 debugLightBoundsMax(lightMax.x, lightMax.y, -nearPlane);
  const float texelWorldSize =
      std::max(lightMax.x - lightMin.x, lightMax.y - lightMin.y) /
      static_cast<float>(std::max(shadowMapSize, 1u));

  shadowFrameCpuData_.flagsCascadeCountLightIndex =
      glm::uvec4(0u, 1u, selectedLightIndex, 0u);
  shadowFrameCpuData_.fadeParams =
      glm::vec4(frame.camera.nearPlane, effectiveFar, 0.0f, 0.0f);
  shadowFrameCpuData_.cascades[0] = ShadowCascadeGpuData{
      .lightViewProj = lightViewProj,
      .lightView = lightView,
      .splitDepthTexelSize =
          glm::vec4(frame.camera.nearPlane, effectiveFar, texelWorldSize, 0.0f),
      .uvScaleBias = glm::vec4(1.0f, 1.0f, 0.0f, 0.0f),
      .biasParams = glm::vec4(settings.constantBias, settings.slopeBias,
                              settings.normalBias, 0.0f),
      .textureSampler =
          glm::uvec4(gpu_.getTextureBindlessIndex(shadowDepthTexture_),
                     frame.sharedResources.shadowRawSamplerId, 0u, 0u),
  };

  shadowDebugFrameData_.selectedShadowLightId = selectedLightId;
  shadowDebugFrameData_.cascades[0].splitNear = frame.camera.nearPlane;
  shadowDebugFrameData_.cascades[0].splitFar = effectiveFar;
  shadowDebugFrameData_.cascades[0].texelWorldSize = texelWorldSize;
  shadowDebugFrameData_.cascades[0].lightView = lightView;
  shadowDebugFrameData_.cascades[0].lightProj = lightProj;
  shadowDebugFrameData_.cascades[0].lightViewProj = lightViewProj;
  shadowDebugFrameData_.cascades[0].inverseLightView = glm::inverse(lightView);
  shadowDebugFrameData_.cascades[0].inverseLightProj = glm::inverse(lightProj);
  shadowDebugFrameData_.cascades[0].lightSpaceBoundsMin =
      glm::vec4(debugLightBoundsMin, 1.0f);
  shadowDebugFrameData_.cascades[0].lightSpaceBoundsMax =
      glm::vec4(debugLightBoundsMax, 1.0f);
  shadowDebugFrameData_.cascades[0].unsnappedCenter =
      glm::vec4(frustumCenter, 1.0f);
  shadowDebugFrameData_.cascades[0].snappedCenter =
      glm::vec4(frustumCenter, 1.0f);
  for (size_t i = 0; i < frustumCorners.size(); ++i) {
    shadowDebugFrameData_.cascades[0].worldFrustumCorners[i] =
        glm::vec4(frustumCorners[i], 1.0f);
  }

  frame.sharedResources.shadowDebugFrameData = shadowDebugFrameData_;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
ShadowRenderer::buildShadowDraws(RenderFrameContext &frame, uint32_t frameSlot,
                                 const ForwardSceneGpuData &sceneGpu) {
  if (frame.scene == nullptr || meshDrawTemplates_.empty()) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (!nuri::isValid(shadowDepthTexture_) ||
      shadowFrameCpuData_.flagsCascadeCountLightIndex.y == 0u) {
    return Result<bool, std::string>::makeResult(true);
  }

  auto initResult = ensureInitialized();
  if (initResult.hasError()) {
    return initResult;
  }

  const std::span<const Renderable> renderables = frame.scene->renderables();
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
      instanceMatricesRing_[frameSlot].buffer->handle();
  const BufferHandle instanceRemapBuffer =
      instanceRemapRing_[frameSlot].buffer->handle();
  const uint64_t instanceMatricesAddress =
      gpu_.getBufferDeviceAddress(instanceMatricesBuffer);
  const uint64_t instanceRemapAddress =
      gpu_.getBufferDeviceAddress(instanceRemapBuffer);
  if (sceneGpu.frameDataAddress == 0u || instanceMatricesAddress == 0u ||
      instanceRemapAddress == 0u) {
    return Result<bool, std::string>::makeError(
        "ShadowRenderer::buildShadowDraws: invalid GPU buffer address");
  }

  drawPushConstants_.clear();
  drawItems_.clear();
  drawPushConstants_.reserve(meshDrawTemplates_.size());
  drawItems_.reserve(meshDrawTemplates_.size());

  const RenderSettings &settings = renderSettingsOrDefault(frame);
  const uint32_t renderableCount = saturateToU32(renderables.size());
  for (const MeshDrawTemplate &entry : meshDrawTemplates_) {
    if (entry.renderable == nullptr || entry.submesh == nullptr) {
      continue;
    }
    const std::optional<SubmeshLod> lod =
        resolveShadowLod(*entry.submesh, settings);
    if (!lod.has_value()) {
      continue;
    }

    drawPushConstants_.push_back(PushConstants{
        .frameDataAddress = sceneGpu.frameDataAddress,
        .vertexBufferAddress = entry.vertexBufferAddress,
        .vertexDecodeBufferAddress = entry.vertexDecodeBufferAddress,
        .instanceMatricesAddress = instanceMatricesAddress,
        .instanceRemapAddress = instanceRemapAddress,
        .instanceCentersPhaseAddress = 0u,
        .instanceBaseMatricesAddress = 0u,
        .instanceCount = renderableCount,
        .materialIndex = entry.materialIndex,
        .vertexDecodeIndex = entry.vertexDecodeIndex,
        .packedVertexFormat = entry.packedVertexFormat,
        .timeSeconds = static_cast<float>(frame.timeSeconds),
        .tessNearDistance = 1.0f,
        .tessFarDistance = 8.0f,
        .tessMinFactor = 1.0f,
        .tessMaxFactor = 1.0f,
        .debugVisualizationMode = 0u,
    });
    const PushConstants &pc = drawPushConstants_.back();

    DrawItem draw{};
    draw.pipeline =
        entry.doubleSided && nuri::isValid(shadowDoubleSidedPipelineHandle_)
            ? shadowDoubleSidedPipelineHandle_
            : shadowPipelineHandle_;
    draw.vertexBuffer = entry.baseVertexBuffer;
    draw.indexBuffer = entry.indexBuffer;
    draw.indexBufferOffset = entry.indexBufferOffset;
    draw.indexFormat = IndexFormat::U32;
    draw.indexCount = lod->indexCount;
    draw.instanceCount = 1u;
    draw.firstIndex = lod->indexOffset;
    draw.firstInstance = entry.instanceIndex;
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
    drawItems_.push_back(draw);
  }

  passDependencyBuffers_.clear();
  appendUniqueBuffer(passDependencyBuffers_, sceneGpu.buffer);
  appendUniqueBuffer(passDependencyBuffers_, instanceMatricesBuffer);
  appendUniqueBuffer(passDependencyBuffers_, instanceRemapBuffer);
  if (frame.sharedResources.shadowFrameGpuData.has_value()) {
    appendUniqueBuffer(passDependencyBuffers_,
                       frame.sharedResources.shadowFrameGpuData->buffer);
  }
  preparedShadowDrawCount_ = saturateToU32(drawItems_.size());
  shadowDebugFrameData_.cascades[0].drawCount = preparedShadowDrawCount_;
  frame.sharedResources.shadowDebugFrameData = shadowDebugFrameData_;
  return Result<bool, std::string>::makeResult(true);
}

void ShadowRenderer::destroyShadowResources() {
  if (nuri::isValid(shadowDebugPreviewTexture_)) {
    gpu_.destroyTexture(shadowDebugPreviewTexture_);
    shadowDebugPreviewTexture_ = {};
  }
  if (nuri::isValid(shadowDepthTexture_)) {
    gpu_.destroyTexture(shadowDepthTexture_);
    shadowDepthTexture_ = {};
  }
  if (nuri::isValid(rawDepthSampler_)) {
    gpu_.destroySampler(rawDepthSampler_);
    rawDepthSampler_ = {};
  }
  shadowMapSize_ = 0u;
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
  instanceMatricesRing_.clear();
  instanceRemapRing_.clear();
  shadowFrameRing_.clear();
  instanceDataRingUploadVersions_.clear();
  shadowFrameUploadSignatures_.clear();
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
  shadowShader_.reset();
  depthShader_.reset();
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
  if (nuri::isValid(shadowPipelineHandle_)) {
    gpu_.destroyRenderPipeline(shadowPipelineHandle_);
    shadowPipelineHandle_ = {};
  }
  initialized_ = false;
}

void ShadowRenderer::resetCachedState() {
  cachedScene_ = nullptr;
  cachedTopologyVersion_ = std::numeric_limits<uint64_t>::max();
  cachedMaterialVersion_ = std::numeric_limits<uint64_t>::max();
  cachedTransformVersion_ = std::numeric_limits<uint64_t>::max();
  cachedGeometryMutationVersion_ = std::numeric_limits<uint64_t>::max();
  preparedShadowFrameSignature_ = std::numeric_limits<uint64_t>::max();
  meshDrawTemplates_.clear();
  instanceMatrices_.clear();
  instanceRemap_.clear();
}

void ShadowRenderer::resetFrameBuildState() {
  hasPreparedShadowDepthPasses_ = false;
  hasPreparedShadowPreviewPass_ = false;
  preparedShadowDrawCount_ = 0u;
  drawPushConstants_.clear();
  drawItems_.clear();
  passDependencyBuffers_.clear();
  previewDependencyTextures_.clear();
  previewDependencyTextureAccessModes_.clear();
  previewPushConstants_ = {};
  previewDraw_ = {};
}

Result<bool, std::string>
ShadowRenderer::publishFrameData(RenderFrameContext &frame) {
  frame.sharedResources.shadowCascadeTextures = {};
  frame.sharedResources.shadowCascadeGraphTextures = {};
  frame.sharedResources.shadowDebugPreviewTexture = {};
  frame.sharedResources.shadowDebugPreviewGraphTexture = {};
  frame.sharedResources.shadowFrameGpuData.reset();
  frame.sharedResources.shadowRawSamplerId = kInvalidShadowBindlessIndex;
  frame.sharedResources.shadowCompareSamplerId = kInvalidShadowBindlessIndex;
  frame.sharedResources.shadowDebugFrameData.reset();

  RenderSettings::ShadowSettings settings =
      renderSettingsOrDefault(frame).shadow;
  sanitizeShadowSettings(settings);
  if (!settings.enabled) {
    destroyShadowResources();
    destroyBuffers();
    resetCachedState();
    resetFrameBuildState();
    return Result<bool, std::string>::makeResult(true);
  }

  auto ensureTextureResult = ensureShadowResources(settings);
  if (ensureTextureResult.hasError()) {
    return ensureTextureResult;
  }
  auto ringCountResult =
      ensureRingBufferCount(std::max(1u, gpu_.getSwapchainImageCount()));
  if (ringCountResult.hasError()) {
    return ringCountResult;
  }
  auto shadowFrameResult =
      ensureShadowFrameRingCapacity(sizeof(ShadowFrameGpuData));
  if (shadowFrameResult.hasError()) {
    return shadowFrameResult;
  }

  const uint32_t rawSamplerId = gpu_.getSamplerBindlessIndex(rawDepthSampler_);
  const uint32_t textureId = gpu_.getTextureBindlessIndex(shadowDepthTexture_);
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

  frame.sharedResources.shadowCascadeTextures[0] = shadowDepthTexture_;
  frame.sharedResources.shadowDebugPreviewTexture = shadowDebugPreviewTexture_;
  frame.sharedResources.shadowRawSamplerId = rawSamplerId;
  frame.sharedResources.shadowFrameGpuData = ShadowFrameGpuDataHandle{
      .buffer = shadowFrameBuffer,
      .bufferAddress = shadowFrameAddress,
  };

  ShadowDebugFrameData debug{};
  debug.cascadeCount = 1u;
  debug.rawSamplerId = rawSamplerId;
  debug.compareSamplerId = kInvalidShadowBindlessIndex;
  debug.cascades[0].texture = shadowDepthTexture_;
  debug.cascades[0].textureBindlessId = textureId;
  frame.sharedResources.shadowDebugFrameData = debug;

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
ShadowRenderer::prepareShadowGraphPasses(RenderFrameContext &frame) {
  resetFrameBuildState();

  RenderSettings::ShadowSettings settings =
      renderSettingsOrDefault(frame).shadow;
  sanitizeShadowSettings(settings);
  if (!settings.enabled) {
    return Result<bool, std::string>::makeResult(true);
  }

  auto ensureTextureResult = ensureShadowResources(settings);
  if (ensureTextureResult.hasError()) {
    return ensureTextureResult;
  }
  auto ringCountResult =
      ensureRingBufferCount(std::max(1u, gpu_.getSwapchainImageCount()));
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
  }

  auto shadowFrameDataResult =
      updateShadowFrameData(frame, settings, settings.shadowMapSize);
  if (shadowFrameDataResult.hasError()) {
    return shadowFrameDataResult;
  }

  const uint32_t frameSlot =
      static_cast<uint32_t>(frame.frameIndex % shadowFrameRing_.size());
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
  preparedShadowFrameSignature_ = shadowFrameSignature;

  if (frame.sharedResources.forwardSceneGpuData.has_value()) {
    auto drawResult = buildShadowDraws(
        frame, frameSlot, *frame.sharedResources.forwardSceneGpuData);
    if (drawResult.hasError()) {
      return drawResult;
    }
  }

  hasPreparedShadowDepthPasses_ = nuri::isValid(shadowDepthTexture_);
  if (nuri::isValid(shadowDebugPreviewTexture_)) {
    const uint32_t sourceTexId =
        gpu_.getTextureBindlessIndex(shadowDepthTexture_);
    if (sourceTexId == kInvalidShadowBindlessIndex) {
      return Result<bool, std::string>::makeError(
          "ShadowRenderer::prepareShadowGraphPasses: invalid shadow depth "
          "bindless index");
    }
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
    previewPushConstants_ = PreviewPushConstants{
        .sourceTexId = sourceTexId,
        .depthScale = 1.0f,
        .depthBias = 0.0f,
    };
    previewDraw_ = DrawItem{};
    previewDraw_.pipeline = previewPipelineHandle_;
    previewDraw_.vertexCount = 3u;
    previewDraw_.instanceCount = 1u;
    previewDraw_.pushConstants = std::span<const std::byte>(
        reinterpret_cast<const std::byte *>(&previewPushConstants_),
        sizeof(PreviewPushConstants));
    previewDraw_.debugLabel = kShadowPreviewDrawLabel;
    previewDraw_.debugColor = kShadowPreviewPassDebugColor;
    previewDependencyTextures_.clear();
    previewDependencyTextures_.push_back(shadowDepthTexture_);
    previewDependencyTextureAccessModes_.clear();
    previewDependencyTextureAccessModes_.push_back(RenderGraphAccessMode::Read);
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
  if (!hasPreparedShadowDepthPasses_ || !nuri::isValid(shadowDepthTexture_)) {
    return Result<bool, std::string>::makeResult(true);
  }

  auto depthImportResult =
      graph.importTexture(shadowDepthTexture_, "shadow_depth_map");
  if (depthImportResult.hasError()) {
    return Result<bool, std::string>::makeError(depthImportResult.error());
  }
  frame.sharedResources.shadowCascadeGraphTextures[0] =
      depthImportResult.value();

  const RenderGraphGraphicsPassDesc desc{
      .color = {},
      .colorTexture = {},
      .hasColorAttachment = false,
      .depth = {.loadOp = LoadOp::Clear,
                .storeOp = StoreOp::Store,
                .clearDepth = 1.0f,
                .clearStencil = 0u},
      .depthTexture = depthImportResult.value(),
      .useViewport = true,
      .viewport = {.x = 0.0f,
                   .y = 0.0f,
                   .width = static_cast<float>(shadowMapSize_),
                   .height = static_cast<float>(shadowMapSize_),
                   .minDepth = 0.0f,
                   .maxDepth = 1.0f},
      .preDispatches = {},
      .dependencyBuffers = std::span<const BufferHandle>(
          passDependencyBuffers_.data(), passDependencyBuffers_.size()),
      .dependencyBufferAccessModes = {},
      .dependencyTextures = {},
      .dependencyTextureAccessModes = {},
      .draws = std::span<const DrawItem>(drawItems_.data(), drawItems_.size()),
      .drawBuffersPreResolved = false,
      .preResolvedDrawBuffers = {},
      .debugLabel = kShadowPassLabel,
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

  if (hasPreparedShadowPreviewPass_ &&
      nuri::isValid(shadowDebugPreviewTexture_)) {
    auto previewImportResult =
        graph.importTexture(shadowDebugPreviewTexture_, "shadow_depth_preview");
    if (previewImportResult.hasError()) {
      return Result<bool, std::string>::makeError(previewImportResult.error());
    }
    frame.sharedResources.shadowDebugPreviewGraphTexture =
        previewImportResult.value();

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
                     .width = static_cast<float>(shadowMapSize_),
                     .height = static_cast<float>(shadowMapSize_),
                     .minDepth = 0.0f,
                     .maxDepth = 1.0f},
        .preDispatches = {},
        .dependencyBuffers = {},
        .dependencyBufferAccessModes = {},
        .dependencyTextures =
            std::span<const TextureHandle>(previewDependencyTextures_.data(),
                                           previewDependencyTextures_.size()),
        .dependencyTextureAccessModes = std::span<const RenderGraphAccessMode>(
            previewDependencyTextureAccessModes_.data(),
            previewDependencyTextureAccessModes_.size()),
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
