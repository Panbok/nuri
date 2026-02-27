#include "nuri/pch.h"

#include "nuri/text/text_renderer.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/gpu_descriptors.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/shader.h"

namespace nuri {
namespace {

template <typename T, typename... Args>
[[nodiscard]] Result<T, std::string> makeError(Args &&...args) {
  std::ostringstream oss;
  (oss << ... << std::forward<Args>(args));
  return Result<T, std::string>::makeError(oss.str());
}

[[nodiscard]] uint32_t packColor(const TextColor &color) {
  const auto toU8 = [](float v) -> uint32_t {
    return static_cast<uint32_t>(
        std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f));
  };
  const uint32_t r = toU8(color.r);
  const uint32_t g = toU8(color.g);
  const uint32_t b = toU8(color.b);
  const uint32_t a = toU8(color.a);
  return (a << 24u) | (b << 16u) | (g << 8u) | r;
}

[[nodiscard]] uint32_t floatBits(float value) {
  uint32_t bits = 0u;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

[[nodiscard]] uint64_t hashMix(uint64_t hash, uint64_t value) {
  constexpr uint64_t kFnvPrime = 1099511628211ull;
  hash ^= value;
  hash *= kFnvPrime;
  return hash;
}

[[nodiscard]] constexpr size_t alignUp(size_t value, size_t alignment) {
  return (value + (alignment - 1u)) & ~(alignment - 1u);
}

constexpr float kBatchPxRangeEpsilon = 1.0e-4f;

void growBounds(TextBounds &bounds, bool &hasBounds, float minX, float minY,
                float maxX, float maxY) {
  if (!hasBounds) {
    bounds.minX = minX;
    bounds.minY = minY;
    bounds.maxX = maxX;
    bounds.maxY = maxY;
    hasBounds = true;
    return;
  }
  bounds.minX = std::min(bounds.minX, minX);
  bounds.minY = std::min(bounds.minY, minY);
  bounds.maxX = std::max(bounds.maxX, maxX);
  bounds.maxY = std::max(bounds.maxY, maxY);
}


[[nodiscard]] glm::mat4 decodeWorld(const std::array<float, 16> &raw) {
  bool any = false;
  for (float v : raw) {
    if (v != 0.0f) {
      any = true;
      break;
    }
  }
  if (!any) {
    return glm::mat4(1.0f);
  }
  glm::mat4 out(1.0f);
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < 4; ++r) {
      out[c][r] = raw[static_cast<size_t>(c * 4 + r)];
    }
  }
  return out;
}

[[nodiscard]] uint64_t estimateCompletedTimelineValue(const GPUDevice &gpu,
                                                      uint64_t frameIndex) {
  const uint64_t lag = std::max<uint64_t>(1u, gpu.getSwapchainImageCount());
  if (frameIndex <= lag) {
    return 0;
  }
  return frameIndex - lag;
}

[[nodiscard]] glm::vec2 computeAlignedOffset2D(const TextBounds &localBounds,
                                               const TextLayoutParams &params,
                                               float anchorX, float anchorY) {
  const float width = std::max(localBounds.maxX - localBounds.minX, 0.0f);
  const float height = std::max(localBounds.maxY - localBounds.minY, 0.0f);
  const float containerWidth =
      params.maxWidthPx > 0.0f ? params.maxWidthPx : width;
  const float containerHeight =
      params.maxHeightPx > 0.0f ? params.maxHeightPx : height;

  float targetMinX = anchorX;
  switch (params.alignH) {
  case TextAlignH::Left:
    targetMinX = anchorX;
    break;
  case TextAlignH::Center:
    targetMinX = anchorX + (containerWidth - width) * 0.5f;
    break;
  case TextAlignH::Right:
    targetMinX = anchorX + (containerWidth - width);
    break;
  }

  float dx = targetMinX - localBounds.minX;
  float dy = 0.0f;
  if (params.alignV == TextAlignV::Baseline) {
    dy = anchorY;
  } else {
    float targetMinY = anchorY;
    switch (params.alignV) {
    case TextAlignV::Top:
      targetMinY = anchorY;
      break;
    case TextAlignV::Middle:
      targetMinY = anchorY + (containerHeight - height) * 0.5f;
      break;
    case TextAlignV::Bottom:
      targetMinY = anchorY + (containerHeight - height);
      break;
    case TextAlignV::Baseline:
      break;
    }
    dy = targetMinY - localBounds.minY;
  }

  return glm::vec2(dx, dy);
}

[[nodiscard]] glm::vec2
computeAlignedOffsetLocal(const TextBounds &localBounds,
                          const TextLayoutParams &params) {
  return computeAlignedOffset2D(localBounds, params, 0.0f, 0.0f);
}

[[nodiscard]] glm::vec3 safeNormalize(const glm::vec3 &v,
                                      const glm::vec3 &fallback) {
  const float len2 = glm::dot(v, v);
  if (len2 <= 1.0e-8f) {
    return fallback;
  }
  return v * glm::inversesqrt(len2);
}

[[nodiscard]] glm::vec3 extractWorldScale(const glm::mat4 &world) {
  const float sx = glm::length(glm::vec3(world[0]));
  const float sy = glm::length(glm::vec3(world[1]));
  const float sz = glm::length(glm::vec3(world[2]));
  return glm::vec3(std::max(sx, 1.0e-4f), std::max(sy, 1.0e-4f),
                   std::max(sz, 1.0e-4f));
}

} // namespace

void TextRenderer::hashWorldTransform(uint64_t &hash,
                                      const WorldTransform &transform) {
  hash = hashMix(hash, static_cast<uint64_t>(transform.billboard));
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < 4; ++r) {
      hash = hashMix(hash, floatBits(transform.worldFromText[c][r]));
    }
  }
}

void TextRenderer::hashWorldQuad(uint64_t &hash, const WorldQuad &quad) {
  hash = hashMix(hash, floatBits(quad.minX));
  hash = hashMix(hash, floatBits(quad.minY));
  hash = hashMix(hash, floatBits(quad.maxX));
  hash = hashMix(hash, floatBits(quad.maxY));
  hash = hashMix(hash, floatBits(quad.uvMinX));
  hash = hashMix(hash, floatBits(quad.uvMinY));
  hash = hashMix(hash, floatBits(quad.uvMaxX));
  hash = hashMix(hash, floatBits(quad.uvMaxY));
  hash = hashMix(hash, floatBits(quad.pxRange));
  hash = hashMix(hash, quad.color);
  hash = hashMix(hash, quad.atlas);
  hash = hashMix(hash, quad.transformId);
}

uint64_t TextRenderer::hashCameraFrameState(const CameraFrameState &camera) {
  uint64_t hash = kHashSeed;
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < 4; ++r) {
      hash = hashMix(hash, floatBits(camera.view[c][r]));
    }
  }
  hash = hashMix(hash, floatBits(camera.cameraPos.x));
  hash = hashMix(hash, floatBits(camera.cameraPos.y));
  hash = hashMix(hash, floatBits(camera.cameraPos.z));
  return hash;
}

bool TextRenderer::uiBatchLess(const UiQuad &a, const UiQuad &b) {
  if (a.atlas != b.atlas) {
    return a.atlas < b.atlas;
  }
  return a.pxRange < b.pxRange;
}

bool TextRenderer::worldBatchLess(const WorldQuad &a, const WorldQuad &b) {
  if (a.atlas != b.atlas) {
    return a.atlas < b.atlas;
  }
  return a.pxRange < b.pxRange;
}

TextRenderer::BillboardFrameBasis
TextRenderer::buildBillboardFrameBasis(const CameraFrameState &camera) {
  BillboardFrameBasis basis{};
  basis.cameraPos = glm::vec3(camera.cameraPos);

  const glm::mat4 invView = glm::inverse(camera.view);
  glm::vec3 right =
      safeNormalize(glm::vec3(invView[0]), glm::vec3(1.0f, 0.0f, 0.0f));
  glm::vec3 up =
      safeNormalize(glm::vec3(invView[1]), glm::vec3(0.0f, 1.0f, 0.0f));
  glm::vec3 forward =
      safeNormalize(glm::cross(right, up), glm::vec3(0.0f, 0.0f, 1.0f));
  up = safeNormalize(glm::cross(forward, right), up);

  // Glyph quads are laid out in Y-down text space; convert billboard basis
  // from Y-up world convention while preserving handedness.
  basis.sphericalRight = right;
  basis.sphericalUp = -up;
  basis.sphericalForward = -forward;
  return basis;
}

glm::mat4
TextRenderer::resolveWorldFromBillboard(const WorldTransform &transform,
                                        const BillboardFrameBasis &basis) {
  if (transform.billboard == TextBillboardMode::None) {
    return transform.worldFromText;
  }

  const glm::vec3 translation = glm::vec3(transform.worldFromText[3]);
  const glm::vec3 scale = extractWorldScale(transform.worldFromText);

  glm::vec3 right{1.0f, 0.0f, 0.0f};
  glm::vec3 up{0.0f, 1.0f, 0.0f};
  glm::vec3 forward{0.0f, 0.0f, 1.0f};

  if (transform.billboard == TextBillboardMode::Spherical) {
    right = basis.sphericalRight;
    up = basis.sphericalUp;
    forward = basis.sphericalForward;
  } else if (transform.billboard == TextBillboardMode::CylindricalY) {
    up = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 toCamera = basis.cameraPos - translation;
    toCamera.y = 0.0f;
    forward = safeNormalize(toCamera, glm::vec3(0.0f, 0.0f, 1.0f));
    right =
        safeNormalize(glm::cross(up, forward), glm::vec3(1.0f, 0.0f, 0.0f));
    forward = safeNormalize(glm::cross(right, up), forward);
    // Glyph quads are laid out in Y-down text space; convert billboard basis
    // from Y-up world convention while preserving handedness.
    up = -up;
    forward = -forward;
  }

  glm::mat4 out(1.0f);
  out[0] = glm::vec4(right * scale.x, 0.0f);
  out[1] = glm::vec4(up * scale.y, 0.0f);
  out[2] = glm::vec4(forward * scale.z, 0.0f);
  out[3] = glm::vec4(translation, 1.0f);
  return out;
}

TextRenderer::TextRenderer(const CreateDesc &desc)
    : gpu_(desc.gpu), fonts_(desc.fonts), layouter_(desc.layouter),
      memory_(desc.memory), shaderPaths_(desc.shaderPaths),
      uiQueue_(&memory_), worldQueue_(&memory_), worldTransforms_(&memory_),
      resolvedWorldTransforms_(&memory_), worldInstances_(&memory_),
      uiVerts_(&memory_), uiBatches_(&memory_), worldBatches_(&memory_),
      uiDraws_(&memory_), worldDraws_(&memory_), uiPcs_(&memory_),
      worldPcs_(&memory_), uiFrames_(&memory_), worldFrames_(&memory_) {}

TextRenderer::~TextRenderer() { destroyGpu(); }

Result<bool, std::string> TextRenderer::beginFrame(uint64_t frameIndex) {
  if (frameIndex_ == frameIndex) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (frameIndex_ != std::numeric_limits<uint64_t>::max()) {
    emitPerfValidation(frameIndex_);
  }
  fonts_.collectGarbage(estimateCompletedTimelineValue(gpu_, frameIndex));
  frameIndex_ = frameIndex;
  clear();
  resetPerfCounters();
  return Result<bool, std::string>::makeResult(true);
}

Result<TextBounds, std::string>
TextRenderer::enqueue2D(const Text2DDesc &desc,
                              std::pmr::memory_resource &scratch) {
  NURI_PROFILER_FUNCTION();
  auto layoutResult = layouter_.layoutUtf8(desc.utf8, desc.style, desc.layout,
                                           scratch, scratch);
  if (layoutResult.hasError()) {
    return makeError<TextBounds>("TextRenderer::enqueue2D: ",
                                 layoutResult.error());
  }

  const TextLayout &layout = layoutResult.value();
  if (layout.glyphs.empty()) {
    return Result<TextBounds, std::string>::makeResult(TextBounds{});
  }

  const uint32_t color = packColor(desc.fillColor);
  const size_t queueStart = uiQueue_.size();
  uiQueue_.reserve(queueStart + layout.glyphs.size());

  TextBounds localBounds{};
  bool hasLocalBounds = false;
  FontHandle lastFont = kInvalidFontHandle;
  float cachedPxRange = 4.0f;

  for (const LayoutGlyph &glyph : layout.glyphs) {
    const uint32_t atlas = fonts_.atlasBindlessIndex(glyph.atlasPage);
    if (atlas == 0) {
      continue;
    }
    if (glyph.font != lastFont) {
      lastFont = glyph.font;
      cachedPxRange = fonts_.pxRange(glyph.font);
    }

    UiQuad quad{};
    quad.minX = glyph.x + glyph.metrics.planeMinX;
    quad.minY = glyph.y - glyph.metrics.planeMaxY;
    quad.maxX = glyph.x + glyph.metrics.planeMaxX;
    quad.maxY = glyph.y - glyph.metrics.planeMinY;
    quad.uvMinX = glyph.metrics.uvMinX;
    quad.uvMinY = glyph.metrics.uvMinY;
    quad.uvMaxX = glyph.metrics.uvMaxX;
    quad.uvMaxY = glyph.metrics.uvMaxY;
    quad.pxRange = cachedPxRange;
    quad.color = color;
    quad.atlas = atlas;
    if (queueStart > 0 && uiQueue_.size() == queueStart &&
        uiBatchLess(quad, uiQueue_[queueStart - 1u])) {
      uiQueueNeedsSort_ = true;
    }
    uiQueue_.push_back(quad);

    growBounds(localBounds, hasLocalBounds, quad.minX, quad.minY, quad.maxX,
               quad.maxY);
  }

  if (!hasLocalBounds || uiQueue_.size() == queueStart) {
    return Result<TextBounds, std::string>::makeResult(TextBounds{});
  }

  const glm::vec2 shift =
      computeAlignedOffset2D(localBounds, desc.layout, desc.x, desc.y);
  for (size_t i = queueStart; i < uiQueue_.size(); ++i) {
    uiQueue_[i].minX += shift.x;
    uiQueue_[i].maxX += shift.x;
    uiQueue_[i].minY += shift.y;
    uiQueue_[i].maxY += shift.y;
  }

  TextBounds finalBounds{};
  finalBounds.minX = localBounds.minX + shift.x;
  finalBounds.maxX = localBounds.maxX + shift.x;
  finalBounds.minY = localBounds.minY + shift.y;
  finalBounds.maxY = localBounds.maxY + shift.y;
  return Result<TextBounds, std::string>::makeResult(finalBounds);
}

Result<TextBounds, std::string>
TextRenderer::enqueue3D(const Text3DDesc &desc,
                              std::pmr::memory_resource &scratch) {
  NURI_PROFILER_FUNCTION();
  auto layoutResult = layouter_.layoutUtf8(desc.utf8, desc.style, desc.layout,
                                           scratch, scratch);
  if (layoutResult.hasError()) {
    return makeError<TextBounds>("TextRenderer::enqueue3D: ",
                                 layoutResult.error());
  }

  const TextLayout &layout = layoutResult.value();
  if (layout.glyphs.empty()) {
    return Result<TextBounds, std::string>::makeResult(TextBounds{});
  }

  const glm::mat4 world = decodeWorld(desc.worldFromText);
  if (worldTransforms_.size() >=
      static_cast<size_t>(std::numeric_limits<uint32_t>::max() - 1u)) {
    return makeError<TextBounds>(
        "TextRenderer::enqueue3D: world transform table overflow");
  }
  worldTransforms_.push_back(
      WorldTransform{.worldFromText = world, .billboard = desc.billboard});
  const uint32_t transformId = static_cast<uint32_t>(worldTransforms_.size());
  hashWorldTransform(worldQueueHash_, worldTransforms_.back());
  worldHasBillboards_ =
      worldHasBillboards_ || (desc.billboard != TextBillboardMode::None);
  const uint32_t color = packColor(desc.fillColor);

  const size_t queueStart = worldQueue_.size();
  worldQueue_.reserve(queueStart + layout.glyphs.size());

  TextBounds localBounds{};
  bool hasLocalBounds = false;
  FontHandle lastFont = kInvalidFontHandle;
  float cachedPxRange = 4.0f;

  for (const LayoutGlyph &glyph : layout.glyphs) {
    const uint32_t atlas = fonts_.atlasBindlessIndex(glyph.atlasPage);
    if (atlas == 0) {
      continue;
    }
    if (glyph.font != lastFont) {
      lastFont = glyph.font;
      cachedPxRange = fonts_.pxRange(glyph.font);
    }

    WorldQuad quad{};
    quad.minX = glyph.x + glyph.metrics.planeMinX;
    quad.minY = glyph.y - glyph.metrics.planeMaxY;
    quad.maxX = glyph.x + glyph.metrics.planeMaxX;
    quad.maxY = glyph.y - glyph.metrics.planeMinY;
    quad.uvMinX = glyph.metrics.uvMinX;
    quad.uvMinY = glyph.metrics.uvMinY;
    quad.uvMaxX = glyph.metrics.uvMaxX;
    quad.uvMaxY = glyph.metrics.uvMaxY;
    quad.pxRange = cachedPxRange;
    quad.color = color;
    quad.atlas = atlas;
    quad.transformId = transformId;
    hashWorldQuad(worldQueueHash_, quad);
    if (queueStart > 0 && worldQueue_.size() == queueStart &&
        worldBatchLess(quad, worldQueue_[queueStart - 1u])) {
      worldQueueNeedsSort_ = true;
    }
    worldQueue_.push_back(quad);

    growBounds(localBounds, hasLocalBounds, quad.minX, quad.minY, quad.maxX,
               quad.maxY);
  }

  if (!hasLocalBounds || worldQueue_.size() == queueStart) {
    worldTransforms_.pop_back();
    return Result<TextBounds, std::string>::makeResult(TextBounds{});
  }

  const glm::vec2 shift = computeAlignedOffsetLocal(localBounds, desc.layout);
  for (size_t i = queueStart; i < worldQueue_.size(); ++i) {
    worldQueue_[i].minX += shift.x;
    worldQueue_[i].minY += shift.y;
    worldQueue_[i].maxX += shift.x;
    worldQueue_[i].maxY += shift.y;
  }

  TextBounds finalBounds{};
  finalBounds.minX = localBounds.minX + shift.x;
  finalBounds.maxX = localBounds.maxX + shift.x;
  finalBounds.minY = localBounds.minY + shift.y;
  finalBounds.maxY = localBounds.maxY + shift.y;
  return Result<TextBounds, std::string>::makeResult(finalBounds);
}

void TextRenderer::clear() {
  uiQueue_.clear();
  worldQueue_.clear();
  worldTransforms_.clear();
  uiVerts_.clear();
  // Keep previously built world geometry/batches for hash-based reuse.
  uiBatches_.clear();
  uiDraws_.clear();
  worldDraws_.clear();
  uiPcs_.clear();
  worldPcs_.clear();
  uiAppended_ = false;
  worldAppended_ = false;
  uiQueueNeedsSort_ = false;
  worldQueueNeedsSort_ = false;
  worldQueueHash_ = kHashSeed;
  worldHasBillboards_ = false;
  worldGlyphBufferAddress_ = 0;
  worldTransformBufferAddress_ = 0;
  worldDependencyBuffers_[0] = {};
}

void TextRenderer::resetPerfCounters() { perf_ = PerfCounters{}; }

void TextRenderer::emitPerfValidation(uint64_t frameIndex) {
  constexpr uint64_t kValidationIntervalFrames = 120;
  if (frameIndex < lastPerfValidationFrame_ + kValidationIntervalFrames) {
    return;
  }
  lastPerfValidationFrame_ = frameIndex;
  NURI_LOG_DEBUG(
      "TextRenderer Perf frame=%llu ui(glyphs=%u batches=%u vb=%zu ib=%zu) "
      "world(glyphs=%u batches=%u vb=%zu ib=%zu)",
      static_cast<unsigned long long>(frameIndex),
      static_cast<unsigned int>(perf_.uiGlyphs),
      static_cast<unsigned int>(perf_.uiBatches), perf_.uiVertexUploadBytes,
      perf_.uiIndexUploadBytes, static_cast<unsigned int>(perf_.worldGlyphs),
      static_cast<unsigned int>(perf_.worldBatches),
      perf_.worldVertexUploadBytes, perf_.worldIndexUploadBytes);
}

Result<bool, std::string> TextRenderer::compileUiShaders() {
  if (::nuri::isValid(uiVs_) && ::nuri::isValid(uiFs_)) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (shaderPaths_.uiVertex.empty() || shaderPaths_.uiFragment.empty()) {
    return makeError<bool>("TextRenderer: ui shader paths are empty");
  }

  if (::nuri::isValid(uiVs_)) {
    gpu_.destroyShaderModule(uiVs_);
    uiVs_ = {};
  }
  if (::nuri::isValid(uiFs_)) {
    gpu_.destroyShaderModule(uiFs_);
    uiFs_ = {};
  }

  auto helper = Shader::create("text_2d_mtsdf", gpu_);
  auto vs = helper->compileFromFile(shaderPaths_.uiVertex.string(),
                                    ShaderStage::Vertex);
  if (vs.hasError()) {
    return Result<bool, std::string>::makeError(vs.error());
  }
  auto fs = helper->compileFromFile(shaderPaths_.uiFragment.string(),
                                    ShaderStage::Fragment);
  if (fs.hasError()) {
    gpu_.destroyShaderModule(vs.value());
    return Result<bool, std::string>::makeError(fs.error());
  }
  uiVs_ = vs.value();
  uiFs_ = fs.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> TextRenderer::compileWorldShaders() {
  if (::nuri::isValid(worldVs_) && ::nuri::isValid(worldFs_)) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (shaderPaths_.worldVertex.empty() || shaderPaths_.worldFragment.empty()) {
    return makeError<bool>("TextRenderer: world shader paths are empty");
  }

  if (::nuri::isValid(worldVs_)) {
    gpu_.destroyShaderModule(worldVs_);
    worldVs_ = {};
  }
  if (::nuri::isValid(worldFs_)) {
    gpu_.destroyShaderModule(worldFs_);
    worldFs_ = {};
  }

  auto helper = Shader::create("text_3d_mtsdf", gpu_);
  auto vs = helper->compileFromFile(shaderPaths_.worldVertex.string(),
                                    ShaderStage::Vertex);
  if (vs.hasError()) {
    return Result<bool, std::string>::makeError(vs.error());
  }
  auto fs = helper->compileFromFile(shaderPaths_.worldFragment.string(),
                                    ShaderStage::Fragment);
  if (fs.hasError()) {
    gpu_.destroyShaderModule(vs.value());
    return Result<bool, std::string>::makeError(fs.error());
  }
  worldVs_ = vs.value();
  worldFs_ = fs.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TextRenderer::ensureUiPipeline(Format colorFormat) {
  if (::nuri::isValid(uiPipeline_) && uiPipelineColor_ == colorFormat) {
    return Result<bool, std::string>::makeResult(true);
  }
  auto shaders = compileUiShaders();
  if (shaders.hasError()) {
    return shaders;
  }
  if (::nuri::isValid(uiPipeline_)) {
    gpu_.destroyRenderPipeline(uiPipeline_);
    uiPipeline_ = {};
  }

  static const VertexBinding bindings[] = {
      {.stride = static_cast<uint32_t>(sizeof(UiVertex))},
  };
  static const VertexAttribute attributes[] = {
      {.location = 0,
       .binding = 0,
       .offset = static_cast<uint32_t>(offsetof(UiVertex, pos)),
       .format = VertexFormat::Float2},
      {.location = 1,
       .binding = 0,
       .offset = static_cast<uint32_t>(offsetof(UiVertex, uv)),
       .format = VertexFormat::Float2},
      {.location = 2,
       .binding = 0,
       .offset = static_cast<uint32_t>(offsetof(UiVertex, color)),
       .format = VertexFormat::UByte4_Norm},
  };

  RenderPipelineDesc desc{};
  desc.vertexInput = VertexInput{
      .attributes = attributes,
      .bindings = bindings,
  };
  desc.vertexShader = uiVs_;
  desc.fragmentShader = uiFs_;
  desc.colorFormats = {colorFormat};
  desc.depthFormat = Format::Count;
  desc.cullMode = CullMode::None;
  desc.polygonMode = PolygonMode::Fill;
  desc.topology = Topology::Triangle;
  desc.blendEnabled = true;

  auto pipeline = gpu_.createRenderPipeline(desc, "Text2D Pipeline");
  if (pipeline.hasError()) {
    return Result<bool, std::string>::makeError(pipeline.error());
  }
  uiPipeline_ = pipeline.value();
  uiPipelineColor_ = colorFormat;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TextRenderer::ensureWorldPipeline(Format colorFormat,
                                        Format depthFormat) {
  if (::nuri::isValid(worldPipeline_) && worldPipelineColor_ == colorFormat &&
      worldPipelineDepth_ == depthFormat) {
    return Result<bool, std::string>::makeResult(true);
  }
  auto shaders = compileWorldShaders();
  if (shaders.hasError()) {
    return shaders;
  }
  if (::nuri::isValid(worldPipeline_)) {
    gpu_.destroyRenderPipeline(worldPipeline_);
    worldPipeline_ = {};
  }

  RenderPipelineDesc desc{};
  desc.vertexInput = {};
  desc.vertexShader = worldVs_;
  desc.fragmentShader = worldFs_;
  desc.colorFormats = {colorFormat};
  desc.depthFormat = depthFormat;
  desc.cullMode = CullMode::None;
  desc.polygonMode = PolygonMode::Fill;
  desc.topology = Topology::Triangle;
  desc.blendEnabled = true;

  auto pipeline = gpu_.createRenderPipeline(desc, "Text3D Pipeline");
  if (pipeline.hasError()) {
    return Result<bool, std::string>::makeError(pipeline.error());
  }
  worldPipeline_ = pipeline.value();
  worldPipelineColor_ = colorFormat;
  worldPipelineDepth_ = depthFormat;
  return Result<bool, std::string>::makeResult(true);
}

void TextRenderer::syncFrames(std::pmr::vector<FrameBuffers> &frames) {
  const uint32_t swapchainCount = std::max(1u, gpu_.getSwapchainImageCount());
  if (frames.size() == swapchainCount) {
    return;
  }

  for (FrameBuffers &f : frames) {
    if (::nuri::isValid(f.vb)) {
      gpu_.destroyBuffer(f.vb);
    }
    if (::nuri::isValid(f.ib)) {
      gpu_.destroyBuffer(f.ib);
    }
    f = FrameBuffers{};
  }
  frames.assign(swapchainCount, FrameBuffers{});
}

uint32_t TextRenderer::frameSlot(std::pmr::vector<FrameBuffers> &frames) {
  syncFrames(frames);
  const uint32_t count = static_cast<uint32_t>(frames.size());
  NURI_ASSERT(count > 0, "TextRenderer frame buffer count must be > 0");
  return gpu_.getSwapchainImageIndex() % count;
}

Result<bool, std::string>
TextRenderer::ensureFrameBuffers(FrameBuffers &frame, size_t vbBytes,
                                       size_t ibQuads,
                                       std::string_view debugPrefix) {
  constexpr size_t kIndicesPerQuad = 6u;
  constexpr size_t kVerticesPerQuad = 4u;
  if (!::nuri::isValid(frame.vb) || frame.vbBytes < vbBytes) {
    if (::nuri::isValid(frame.vb)) {
      gpu_.destroyBuffer(frame.vb);
    }
    const size_t newSize = std::max({vbBytes, frame.vbBytes * 2u, size_t{1}});
    auto vb = gpu_.createBuffer(
        BufferDesc{
            .usage = BufferUsage::Vertex,
            .storage = Storage::HostVisible,
            .size = newSize,
        },
        std::string(debugPrefix) + "_vb");
    if (vb.hasError()) {
      return Result<bool, std::string>::makeError(vb.error());
    }
    frame.vb = vb.value();
    frame.vbBytes = newSize;
  }

  if (!::nuri::isValid(frame.ib) || frame.ibQuadCapacity < ibQuads) {
    if (::nuri::isValid(frame.ib)) {
      gpu_.destroyBuffer(frame.ib);
    }
    const size_t newQuadCapacity =
        std::max({ibQuads, frame.ibQuadCapacity * 2u, size_t{1}});
    const size_t newSize = newQuadCapacity * kIndicesPerQuad * sizeof(uint32_t);
    auto ib = gpu_.createBuffer(
        BufferDesc{
            .usage = BufferUsage::Index,
            .storage = Storage::HostVisible,
            .size = newSize,
        },
        std::string(debugPrefix) + "_ib");
    if (ib.hasError()) {
      return Result<bool, std::string>::makeError(ib.error());
    }
    frame.ib = ib.value();
    frame.ibQuadCapacity = newQuadCapacity;
    frame.ibBytes = newSize;

    std::pmr::vector<uint32_t> indexPattern(&memory_);
    indexPattern.resize(newQuadCapacity * kIndicesPerQuad);
    for (size_t i = 0; i < newQuadCapacity; ++i) {
      const uint32_t baseVertex = static_cast<uint32_t>(i * kVerticesPerQuad);
      const size_t idx = i * kIndicesPerQuad;
      indexPattern[idx + 0u] = baseVertex + 0u;
      indexPattern[idx + 1u] = baseVertex + 1u;
      indexPattern[idx + 2u] = baseVertex + 2u;
      indexPattern[idx + 3u] = baseVertex + 2u;
      indexPattern[idx + 4u] = baseVertex + 3u;
      indexPattern[idx + 5u] = baseVertex + 0u;
    }

    auto up = gpu_.updateBuffer(
        frame.ib,
        std::span<const std::byte>(
            reinterpret_cast<const std::byte *>(indexPattern.data()), newSize),
        0);
    if (up.hasError()) {
      gpu_.destroyBuffer(frame.ib);
      frame.ib = {};
      frame.ibBytes = 0;
      frame.ibQuadCapacity = 0;
      return Result<bool, std::string>::makeError(up.error());
    }
  }

  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TextRenderer::ensureWorldInstanceBuffer(FrameBuffers &frame, size_t bytes,
                                              std::string_view debugPrefix) {
  if (!::nuri::isValid(frame.vb) || frame.vbBytes < bytes) {
    if (::nuri::isValid(frame.vb)) {
      gpu_.destroyBuffer(frame.vb);
    }
    const size_t newSize = std::max({bytes, frame.vbBytes * 2u, size_t{1}});
    auto vb = gpu_.createBuffer(
        BufferDesc{
            .usage = BufferUsage::Storage | BufferUsage::Vertex,
            .storage = Storage::HostVisible,
            .size = newSize,
        },
        std::string(debugPrefix) + "_instances");
    if (vb.hasError()) {
      return Result<bool, std::string>::makeError(vb.error());
    }
    frame.vb = vb.value();
    frame.vbBytes = newSize;
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> TextRenderer::uploadUi(uint32_t slot) {
  NURI_PROFILER_FUNCTION();
  constexpr size_t kVerticesPerQuad = 4u;
  constexpr size_t kIndicesPerQuad = 6u;
  FrameBuffers &frame = uiFrames_[slot];
  const size_t vbBytes = uiVerts_.size() * sizeof(UiVertex);
  const size_t quadCount = uiVerts_.size() / kVerticesPerQuad;
  const size_t ibBytes = quadCount * kIndicesPerQuad * sizeof(uint32_t);
  perf_.uiVertexUploadBytes = vbBytes;
  perf_.uiIndexUploadBytes = 0;
  auto ensure = ensureFrameBuffers(frame, vbBytes, quadCount, "text_ui");
  if (ensure.hasError()) {
    return ensure;
  }

  if (vbBytes > 0) {
    auto up = gpu_.updateBuffer(
        frame.vb,
        std::span<const std::byte>(
            reinterpret_cast<const std::byte *>(uiVerts_.data()), vbBytes),
        0);
    if (up.hasError()) {
      return Result<bool, std::string>::makeError(up.error());
    }
  }
  (void)ibBytes;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> TextRenderer::uploadWorld(uint32_t slot) {
  NURI_PROFILER_FUNCTION();
  FrameBuffers &frame = worldFrames_[slot];
  const size_t transformBytes =
      resolvedWorldTransforms_.size() * sizeof(ResolvedWorldTransform);
  const size_t instancesOffset = alignUp(transformBytes, 16u);
  const size_t instanceBytes =
      worldInstances_.size() * sizeof(WorldGlyphInstance);
  const size_t totalBytes = instancesOffset + instanceBytes;
  perf_.worldVertexUploadBytes = totalBytes;
  perf_.worldIndexUploadBytes = 0;
  auto ensure = ensureWorldInstanceBuffer(frame, totalBytes, "text_world");
  if (ensure.hasError()) {
    return ensure;
  }

  if (transformBytes > 0) {
    auto up = gpu_.updateBuffer(
        frame.vb,
        std::span<const std::byte>(reinterpret_cast<const std::byte *>(
                                       resolvedWorldTransforms_.data()),
                                   transformBytes),
        0);
    if (up.hasError()) {
      return Result<bool, std::string>::makeError(up.error());
    }
  }

  if (instanceBytes > 0) {
    auto up = gpu_.updateBuffer(
        frame.vb,
        std::span<const std::byte>(
            reinterpret_cast<const std::byte *>(worldInstances_.data()),
            instanceBytes),
        instancesOffset);
    if (up.hasError()) {
      return Result<bool, std::string>::makeError(up.error());
    }
  }

  worldTransformBufferAddress_ = gpu_.getBufferDeviceAddress(frame.vb, 0u);
  worldGlyphBufferAddress_ =
      gpu_.getBufferDeviceAddress(frame.vb, instancesOffset);
  return Result<bool, std::string>::makeResult(true);
}

void TextRenderer::buildUiGeometry() {
  NURI_PROFILER_ZONE("TextRenderer::buildUiGeometry",
                     NURI_PROFILER_COLOR_CMD_DRAW);
  if (uiQueueNeedsSort_) {
    std::sort(uiQueue_.begin(), uiQueue_.end(), uiBatchLess);
    uiQueueNeedsSort_ = false;
  }
  uiVerts_.clear();
  uiBatches_.clear();

  const size_t quadCount = uiQueue_.size();
  uiVerts_.reserve(quadCount * 4u);
  uiBatches_.reserve(16u);

  uint32_t currentAtlas = std::numeric_limits<uint32_t>::max();
  float currentPxRange = -1.0f;
  size_t indexWrite = 0;
  for (const UiQuad &q : uiQueue_) {
    if (currentAtlas != q.atlas ||
        std::abs(currentPxRange - q.pxRange) > kBatchPxRangeEpsilon) {
      currentAtlas = q.atlas;
      currentPxRange = q.pxRange;
      uiBatches_.push_back(UiBatch{
          .atlas = q.atlas,
          .pxRange = q.pxRange,
          .firstIndex = static_cast<uint32_t>(indexWrite),
          .indexCount = 0,
      });
    }

    uiVerts_.push_back(
        UiVertex{{q.minX, q.minY}, {q.uvMinX, q.uvMinY}, q.color});
    uiVerts_.push_back(
        UiVertex{{q.maxX, q.minY}, {q.uvMaxX, q.uvMinY}, q.color});
    uiVerts_.push_back(
        UiVertex{{q.maxX, q.maxY}, {q.uvMaxX, q.uvMaxY}, q.color});
    uiVerts_.push_back(
        UiVertex{{q.minX, q.maxY}, {q.uvMinX, q.uvMaxY}, q.color});
    indexWrite += 6u;
    uiBatches_.back().indexCount += 6u;
  }
  perf_.uiGlyphs = static_cast<uint32_t>(uiVerts_.size() / 4u);
  perf_.uiBatches = static_cast<uint32_t>(uiBatches_.size());
  NURI_PROFILER_ZONE_END();
}

void TextRenderer::buildWorldGeometry(const CameraFrameState &camera) {
  NURI_PROFILER_ZONE("TextRenderer::buildWorldGeometry",
                     NURI_PROFILER_COLOR_CMD_DRAW);
  if (worldQueueNeedsSort_) {
    std::sort(worldQueue_.begin(), worldQueue_.end(), worldBatchLess);
    worldQueueNeedsSort_ = false;
  }
  worldInstances_.clear();
  worldBatches_.clear();

  const size_t quadCount = worldQueue_.size();
  worldInstances_.reserve(quadCount);
  worldBatches_.reserve(16u);
  const BillboardFrameBasis basis = buildBillboardFrameBasis(camera);
  resolvedWorldTransforms_.resize(worldTransforms_.size());
  for (size_t i = 0; i < worldTransforms_.size(); ++i) {
    const glm::mat4 world =
        resolveWorldFromBillboard(worldTransforms_[i], basis);
    resolvedWorldTransforms_[i].basisX = glm::vec4(glm::vec3(world[0]), 0.0f);
    resolvedWorldTransforms_[i].basisY = glm::vec4(glm::vec3(world[1]), 0.0f);
    resolvedWorldTransforms_[i].translation =
        glm::vec4(glm::vec3(world[3]), 0.0f);
  }

  uint32_t currentAtlas = std::numeric_limits<uint32_t>::max();
  float currentPxRange = -1.0f;
  for (const WorldQuad &q : worldQueue_) {
    if (currentAtlas != q.atlas ||
        std::abs(currentPxRange - q.pxRange) > kBatchPxRangeEpsilon) {
      currentAtlas = q.atlas;
      currentPxRange = q.pxRange;
      worldBatches_.push_back(WorldBatch{
          .atlas = q.atlas,
          .pxRange = q.pxRange,
          .firstInstance = static_cast<uint32_t>(worldInstances_.size()),
          .instanceCount = 0,
      });
    }

    if (q.transformId == 0u) {
      continue;
    }
    const size_t transformIdx = static_cast<size_t>(q.transformId) - 1u;
    if (transformIdx >= resolvedWorldTransforms_.size()) {
      continue;
    }
    worldInstances_.push_back(WorldGlyphInstance{
        .rectMinMax = glm::vec4(q.minX, q.minY, q.maxX, q.maxY),
        .uvMinMax = glm::vec4(q.uvMinX, q.uvMinY, q.uvMaxX, q.uvMaxY),
        .color = q.color,
        .transformIndex = static_cast<uint32_t>(transformIdx),
    });
    worldBatches_.back().instanceCount += 1u;
  }
  perf_.worldGlyphs = static_cast<uint32_t>(worldInstances_.size());
  perf_.worldBatches = static_cast<uint32_t>(worldBatches_.size());
  NURI_PROFILER_ZONE_END();
}

Result<bool, std::string>
TextRenderer::append3DPasses(RenderFrameContext &frame,
                                   RenderPassList &out) {
  NURI_PROFILER_FUNCTION();
  if (worldQueue_.empty() || worldAppended_) {
    return Result<bool, std::string>::makeResult(true);
  }
  const uint64_t cameraHash =
      worldHasBillboards_ ? hashCameraFrameState(frame.camera) : 0ull;
  const bool canReuseWorldGeometry =
      worldGeometryValid_ && worldQueueHash_ == lastBuiltWorldQueueHash_ &&
      cameraHash == lastBuiltWorldCameraHash_;
  if (!canReuseWorldGeometry) {
    buildWorldGeometry(frame.camera);
    worldGeometryValid_ = true;
    lastBuiltWorldQueueHash_ = worldQueueHash_;
    lastBuiltWorldCameraHash_ = cameraHash;
  }
  if (worldBatches_.empty()) {
    worldAppended_ = true;
    return Result<bool, std::string>::makeResult(true);
  }

  const bool hasDepth = ::nuri::isValid(frame.sharedDepthTexture);
  const Format depthFormat =
      hasDepth ? gpu_.getTextureFormat(frame.sharedDepthTexture)
               : Format::Count;
  auto pipeline = ensureWorldPipeline(gpu_.getSwapchainFormat(), depthFormat);
  if (pipeline.hasError()) {
    return pipeline;
  }

  const uint32_t slot = frameSlot(worldFrames_);
  auto upload = uploadWorld(slot);
  if (upload.hasError()) {
    return upload;
  }
  if (worldGlyphBufferAddress_ == 0u || worldTransformBufferAddress_ == 0u) {
    return makeError<bool>(
        "TextRenderer::append3DPasses: invalid world buffer device address");
  }

  worldDraws_.clear();
  worldPcs_.clear();
  worldDraws_.reserve(worldBatches_.size());
  worldPcs_.reserve(worldBatches_.size());
  const glm::mat4 viewProj = frame.camera.proj * frame.camera.view;
  const FrameBuffers &buffers = worldFrames_[slot];
  for (const WorldBatch &b : worldBatches_) {
    worldPcs_.push_back(WorldPC{
        .viewProj = viewProj,
        .glyphBufferAddress = worldGlyphBufferAddress_,
        .transformBufferAddress = worldTransformBufferAddress_,
        .atlas = b.atlas,
        .pxRange = b.pxRange,
    });
    const WorldPC &pc = worldPcs_.back();
    DrawItem d{};
    d.pipeline = worldPipeline_;
    d.vertexCount = 6;
    d.instanceCount = b.instanceCount;
    d.firstVertex = 0;
    d.firstInstance = b.firstInstance;
    d.pushConstants = std::span<const std::byte>(
        reinterpret_cast<const std::byte *>(&pc), sizeof(pc));
    d.debugLabel = "Text3D Batch";
    d.debugColor = 0xff44cc88u;
    if (hasDepth) {
      d.useDepthState = true;
      d.depthState = {.compareOp = CompareOp::LessEqual,
                      .isDepthWriteEnabled = false};
    }
    worldDraws_.push_back(d);
  }
  worldDependencyBuffers_[0] = buffers.vb;

  int32_t w = 0;
  int32_t h = 0;
  gpu_.getFramebufferSize(w, h);
  RenderPass pass{};
  pass.color = {.loadOp = out.empty() ? LoadOp::Clear : LoadOp::Load,
                .storeOp = StoreOp::Store,
                .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
  pass.useViewport = true;
  pass.viewport = {.x = 0.0f,
                   .y = 0.0f,
                   .width = std::max(1.0f, static_cast<float>(w)),
                   .height = std::max(1.0f, static_cast<float>(h)),
                   .minDepth = 0.0f,
                   .maxDepth = 1.0f};
  if (hasDepth) {
    pass.depthTexture = frame.sharedDepthTexture;
    pass.depth = {.loadOp = LoadOp::Load,
                  .storeOp = StoreOp::Store,
                  .clearDepth = 1.0f,
                  .clearStencil = 0};
  }
  pass.draws =
      std::span<const DrawItem>(worldDraws_.data(), worldDraws_.size());
  pass.dependencyBuffers = std::span<const BufferHandle>(
      worldDependencyBuffers_.data(), worldDependencyBuffers_.size());
  pass.debugLabel = "Text3D Pass";
  pass.debugColor = 0xff44cc88u;
  out.push_back(pass);
  worldAppended_ = true;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TextRenderer::append2DPasses(RenderFrameContext &, RenderPassList &out) {
  NURI_PROFILER_FUNCTION();
  if (uiQueue_.empty() || uiAppended_) {
    return Result<bool, std::string>::makeResult(true);
  }
  buildUiGeometry();
  if (uiBatches_.empty()) {
    uiAppended_ = true;
    return Result<bool, std::string>::makeResult(true);
  }

  auto pipeline = ensureUiPipeline(gpu_.getSwapchainFormat());
  if (pipeline.hasError()) {
    return pipeline;
  }

  const uint32_t slot = frameSlot(uiFrames_);
  auto upload = uploadUi(slot);
  if (upload.hasError()) {
    return upload;
  }

  uiDraws_.clear();
  uiPcs_.clear();
  uiDraws_.reserve(uiBatches_.size());
  uiPcs_.reserve(uiBatches_.size());
  int32_t fbW = 0;
  int32_t fbH = 0;
  gpu_.getFramebufferSize(fbW, fbH);

  int32_t logicalW = 0;
  int32_t logicalH = 0;
  gpu_.getWindowSize(logicalW, logicalH);
  if (logicalW <= 0 || logicalH <= 0) {
    logicalW = fbW;
    logicalH = fbH;
  }

  const glm::mat4 proj = glm::ortho(
      0.0f, std::max(1.0f, static_cast<float>(logicalW)),
      std::max(1.0f, static_cast<float>(logicalH)), 0.0f, -1.0f, 1.0f);
  const FrameBuffers &buffers = uiFrames_[slot];
  for (const UiBatch &b : uiBatches_) {
    uiPcs_.push_back(
        UiPC{.proj = proj, .atlas = b.atlas, .pxRange = b.pxRange});
    const UiPC &pc = uiPcs_.back();
    DrawItem d{};
    d.pipeline = uiPipeline_;
    d.vertexBuffer = buffers.vb;
    d.indexBuffer = buffers.ib;
    d.indexFormat = IndexFormat::U32;
    d.indexCount = b.indexCount;
    d.instanceCount = 1;
    d.firstIndex = b.firstIndex;
    d.pushConstants = std::span<const std::byte>(
        reinterpret_cast<const std::byte *>(&pc), sizeof(pc));
    d.debugLabel = "Text2D Batch";
    d.debugColor = 0xffcc8844u;
    uiDraws_.push_back(d);
  }

  RenderPass pass{};
  pass.color = {.loadOp = out.empty() ? LoadOp::Clear : LoadOp::Load,
                .storeOp = StoreOp::Store,
                .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
  pass.useViewport = true;
  pass.viewport = {.x = 0.0f,
                   .y = 0.0f,
                   .width = std::max(1.0f, static_cast<float>(fbW)),
                   .height = std::max(1.0f, static_cast<float>(fbH)),
                   .minDepth = 0.0f,
                   .maxDepth = 1.0f};
  pass.draws = std::span<const DrawItem>(uiDraws_.data(), uiDraws_.size());
  pass.debugLabel = "Text2D Pass";
  pass.debugColor = 0xffcc8844u;
  out.push_back(pass);
  uiAppended_ = true;
  return Result<bool, std::string>::makeResult(true);
}

void TextRenderer::destroyGpu() {
  auto destroyFrames = [this](std::pmr::vector<FrameBuffers> &frames) {
    for (FrameBuffers &f : frames) {
      if (::nuri::isValid(f.vb)) {
        gpu_.destroyBuffer(f.vb);
      }
      if (::nuri::isValid(f.ib)) {
        gpu_.destroyBuffer(f.ib);
      }
      f = FrameBuffers{};
    }
    frames.clear();
  };
  destroyFrames(uiFrames_);
  destroyFrames(worldFrames_);

  if (::nuri::isValid(uiPipeline_)) {
    gpu_.destroyRenderPipeline(uiPipeline_);
    uiPipeline_ = {};
  }
  if (::nuri::isValid(worldPipeline_)) {
    gpu_.destroyRenderPipeline(worldPipeline_);
    worldPipeline_ = {};
  }
  if (::nuri::isValid(uiVs_)) {
    gpu_.destroyShaderModule(uiVs_);
    uiVs_ = {};
  }
  if (::nuri::isValid(uiFs_)) {
    gpu_.destroyShaderModule(uiFs_);
    uiFs_ = {};
  }
  if (::nuri::isValid(worldVs_)) {
    gpu_.destroyShaderModule(worldVs_);
    worldVs_ = {};
  }
  if (::nuri::isValid(worldFs_)) {
    gpu_.destroyShaderModule(worldFs_);
    worldFs_ = {};
  }
}

} // namespace nuri
