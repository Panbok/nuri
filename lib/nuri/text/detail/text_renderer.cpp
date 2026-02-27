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

constexpr uint64_t kHashSeed = 1469598103934665603ull;

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

struct FrameBuffers {
  BufferHandle vb{};
  BufferHandle ib{};
  size_t vbBytes = 0;
  size_t ibBytes = 0;
  size_t ibQuadCapacity = 0;
};

struct UiQuad {
  float minX = 0.0f;
  float minY = 0.0f;
  float maxX = 0.0f;
  float maxY = 0.0f;
  float uvMinX = 0.0f;
  float uvMinY = 0.0f;
  float uvMaxX = 0.0f;
  float uvMaxY = 0.0f;
  float pxRange = 4.0f;
  uint32_t color = 0xffffffffu;
  uint32_t atlas = 0;
};

struct WorldQuad {
  float minX = 0.0f;
  float minY = 0.0f;
  float maxX = 0.0f;
  float maxY = 0.0f;
  float uvMinX = 0.0f;
  float uvMinY = 0.0f;
  float uvMaxX = 0.0f;
  float uvMaxY = 0.0f;
  float pxRange = 4.0f;
  uint32_t color = 0xffffffffu;
  uint32_t atlas = 0;
  uint32_t transformId = 0;
};

struct WorldTransform {
  glm::mat4 worldFromText{1.0f};
  TextBillboardMode billboard = TextBillboardMode::None;
};

struct ResolvedWorldTransform {
  glm::vec4 basisX{1.0f, 0.0f, 0.0f, 0.0f};
  glm::vec4 basisY{0.0f, 1.0f, 0.0f, 0.0f};
  glm::vec4 translation{0.0f, 0.0f, 0.0f, 0.0f};
};

struct WorldGlyphInstance {
  glm::vec4 rectMinMax{0.0f}; // minX, minY, maxX, maxY
  glm::vec4 uvMinMax{0.0f};   // uvMinX, uvMinY, uvMaxX, uvMaxY
  uint32_t color = 0xffffffffu;
  uint32_t transformIndex = 0;
  uint32_t _pad0 = 0;
  uint32_t _pad1 = 0;
};

void hashWorldTransform(uint64_t &hash, const WorldTransform &transform) {
  hash = hashMix(hash, static_cast<uint64_t>(transform.billboard));
  for (int c = 0; c < 4; ++c) {
    for (int r = 0; r < 4; ++r) {
      hash = hashMix(hash, floatBits(transform.worldFromText[c][r]));
    }
  }
}

void hashWorldQuad(uint64_t &hash, const WorldQuad &quad) {
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

[[nodiscard]] uint64_t hashCameraFrameState(const CameraFrameState &camera) {
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

[[nodiscard]] bool uiBatchLess(const UiQuad &a, const UiQuad &b) {
  if (a.atlas != b.atlas) {
    return a.atlas < b.atlas;
  }
  return a.pxRange < b.pxRange;
}

[[nodiscard]] bool worldBatchLess(const WorldQuad &a, const WorldQuad &b) {
  if (a.atlas != b.atlas) {
    return a.atlas < b.atlas;
  }
  return a.pxRange < b.pxRange;
}

struct UiVertex {
  glm::vec2 pos{0.0f};
  glm::vec2 uv{0.0f};
  uint32_t color = 0xffffffffu;
};

struct WorldVertex {
  glm::vec3 pos{0.0f};
  glm::vec2 uv{0.0f};
  uint32_t color = 0xffffffffu;
};

struct UiBatch {
  uint32_t atlas = 0;
  float pxRange = 4.0f;
  uint32_t firstIndex = 0;
  uint32_t indexCount = 0;
};

struct WorldBatch {
  uint32_t atlas = 0;
  float pxRange = 4.0f;
  uint32_t firstInstance = 0;
  uint32_t instanceCount = 0;
};

struct UiPC {
  glm::mat4 proj{1.0f};
  uint32_t atlas = 0;
  float pxRange = 4.0f;
  float pad0 = 0.0f;
  float pad1 = 0.0f;
};

struct WorldPC {
  glm::mat4 viewProj{1.0f};
  uint64_t glyphBufferAddress = 0;
  uint64_t transformBufferAddress = 0;
  uint32_t atlas = 0;
  float pxRange = 4.0f;
  float pad0 = 0.0f;
  float pad1 = 0.0f;
};

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
  const float containerWidth = params.maxWidthPx > 0.0f ? params.maxWidthPx : width;
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

[[nodiscard]] glm::vec2 computeAlignedOffsetLocal(const TextBounds &localBounds,
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

struct BillboardFrameBasis {
  glm::vec3 sphericalRight{1.0f, 0.0f, 0.0f};
  glm::vec3 sphericalUp{0.0f, -1.0f, 0.0f};
  glm::vec3 sphericalForward{0.0f, 0.0f, -1.0f};
  glm::vec3 cameraPos{0.0f};
};

[[nodiscard]] BillboardFrameBasis
buildBillboardFrameBasis(const CameraFrameState &camera) {
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

[[nodiscard]] glm::mat4
resolveWorldFromBillboard(const WorldTransform &transform,
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
    right = safeNormalize(glm::cross(up, forward), glm::vec3(1.0f, 0.0f, 0.0f));
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

class TextRendererImpl final : public TextRenderer {
public:
  explicit TextRendererImpl(const CreateDesc &desc)
      : gpu_(desc.gpu), fonts_(desc.fonts), layouter_(desc.layouter),
        memory_(desc.memory), shaderPaths_(desc.shaderPaths), uiQueue_(&memory_),
        worldQueue_(&memory_), worldTransforms_(&memory_),
        resolvedWorldTransforms_(&memory_), worldInstances_(&memory_),
        uiVerts_(&memory_), uiBatches_(&memory_), worldBatches_(&memory_),
        uiDraws_(&memory_), worldDraws_(&memory_), uiPcs_(&memory_),
        worldPcs_(&memory_), uiFrames_(&memory_), worldFrames_(&memory_) {}

  ~TextRendererImpl() override { destroyGpu(); }

  Result<bool, std::string> beginFrame(uint64_t frameIndex) override;
  Result<TextBounds, std::string>
  enqueue2D(const Text2DDesc &desc, std::pmr::memory_resource &scratch) override;
  Result<TextBounds, std::string>
  enqueue3D(const Text3DDesc &desc, std::pmr::memory_resource &scratch) override;
  Result<bool, std::string>
  append3DPasses(RenderFrameContext &frame, RenderPassList &out) override;
  Result<bool, std::string>
  append2DPasses(RenderFrameContext &frame, RenderPassList &out) override;
  void clear() override;

private:
  Result<bool, std::string> compileUiShaders();
  Result<bool, std::string> compileWorldShaders();
  Result<bool, std::string> ensureUiPipeline(Format colorFormat);
  Result<bool, std::string> ensureWorldPipeline(Format colorFormat,
                                                Format depthFormat);
  void syncFrames(std::pmr::vector<FrameBuffers> &frames);
  uint32_t frameSlot(std::pmr::vector<FrameBuffers> &frames);
  Result<bool, std::string> ensureFrameBuffers(FrameBuffers &frame,
                                               size_t vbBytes, size_t ibQuads,
                                               std::string_view debugPrefix);
  Result<bool, std::string> ensureWorldInstanceBuffer(FrameBuffers &frame,
                                                      size_t bytes,
                                                      std::string_view debugPrefix);
  Result<bool, std::string> uploadUi(uint32_t slot);
  Result<bool, std::string> uploadWorld(uint32_t slot);
  void buildUiGeometry();
  void buildWorldGeometry(const CameraFrameState &camera);
  void resetPerfCounters();
  void emitPerfValidation(uint64_t frameIndex);
  void destroyGpu();

private:
  GPUDevice &gpu_;
  FontManager &fonts_;
  TextLayouter &layouter_;
  std::pmr::memory_resource &memory_;
  ShaderPaths shaderPaths_{};

  uint64_t frameIndex_ = std::numeric_limits<uint64_t>::max();
  bool uiAppended_ = false;
  bool worldAppended_ = false;
  bool uiQueueNeedsSort_ = false;
  bool worldQueueNeedsSort_ = false;
  uint64_t lastPerfValidationFrame_ = 0;

  struct PerfCounters {
    uint32_t uiGlyphs = 0;
    uint32_t uiBatches = 0;
    uint32_t worldGlyphs = 0;
    uint32_t worldBatches = 0;
    size_t uiVertexUploadBytes = 0;
    size_t uiIndexUploadBytes = 0;
    size_t worldVertexUploadBytes = 0;
    size_t worldIndexUploadBytes = 0;
  } perf_;

  ShaderHandle uiVs_{};
  ShaderHandle uiFs_{};
  ShaderHandle worldVs_{};
  ShaderHandle worldFs_{};
  RenderPipelineHandle uiPipeline_{};
  RenderPipelineHandle worldPipeline_{};
  Format uiPipelineColor_ = Format::Count;
  Format worldPipelineColor_ = Format::Count;
  Format worldPipelineDepth_ = Format::Count;

  std::pmr::vector<UiQuad> uiQueue_;
  std::pmr::vector<WorldQuad> worldQueue_;
  std::pmr::vector<WorldTransform> worldTransforms_;
  std::pmr::vector<ResolvedWorldTransform> resolvedWorldTransforms_;
  std::pmr::vector<WorldGlyphInstance> worldInstances_;
  std::pmr::vector<UiVertex> uiVerts_;
  std::pmr::vector<UiBatch> uiBatches_;
  std::pmr::vector<WorldBatch> worldBatches_;
  std::pmr::vector<DrawItem> uiDraws_;
  std::pmr::vector<DrawItem> worldDraws_;
  std::pmr::vector<UiPC> uiPcs_;
  std::pmr::vector<WorldPC> worldPcs_;
  std::pmr::vector<FrameBuffers> uiFrames_;
  std::pmr::vector<FrameBuffers> worldFrames_;

  uint64_t worldQueueHash_ = kHashSeed;
  uint64_t lastBuiltWorldQueueHash_ = 0;
  uint64_t lastBuiltWorldCameraHash_ = 0;
  bool worldGeometryValid_ = false;
  bool worldHasBillboards_ = false;
  uint64_t worldGlyphBufferAddress_ = 0;
  uint64_t worldTransformBufferAddress_ = 0;
  std::array<BufferHandle, 1> worldDependencyBuffers_{};
};

Result<bool, std::string> TextRendererImpl::beginFrame(uint64_t frameIndex) {
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
TextRendererImpl::enqueue2D(const Text2DDesc &desc,
                            std::pmr::memory_resource &scratch) {
  NURI_PROFILER_FUNCTION();
  auto layoutResult =
      layouter_.layoutUtf8(desc.utf8, desc.style, desc.layout, scratch, scratch);
  if (layoutResult.hasError()) {
    return makeError<TextBounds>("TextRenderer::enqueue2D: ",
                                 layoutResult.error());
  }

  const TextLayout &layout = layoutResult.value();
  if (layout.glyphs.empty()) {
    return Result<TextBounds, std::string>::makeResult(TextBounds{});
  }

  std::pmr::vector<UiQuad> localQuads(&scratch);
  localQuads.reserve(layout.glyphs.size());
  const uint32_t color = packColor(desc.fillColor);
  TextBounds localBounds{};
  bool hasLocalBounds = false;
  for (const LayoutGlyph &glyph : layout.glyphs) {
    const uint32_t atlas = fonts_.atlasBindlessIndex(glyph.atlasPage);
    if (atlas == 0) {
      continue;
    }

    UiQuad quad{};
    quad.minX = glyph.x + glyph.metrics.planeMinX;
    // MSDF plane coordinates are Y-up; screen/UI space is Y-down.
    quad.minY = glyph.y - glyph.metrics.planeMaxY;
    quad.maxX = glyph.x + glyph.metrics.planeMaxX;
    quad.maxY = glyph.y - glyph.metrics.planeMinY;
    quad.uvMinX = glyph.metrics.uvMinX;
    quad.uvMinY = glyph.metrics.uvMinY;
    quad.uvMaxX = glyph.metrics.uvMaxX;
    quad.uvMaxY = glyph.metrics.uvMaxY;
    quad.pxRange = fonts_.pxRange(glyph.font);
    quad.color = color;
    quad.atlas = atlas;
    localQuads.push_back(quad);

    growBounds(localBounds, hasLocalBounds, quad.minX, quad.minY, quad.maxX,
               quad.maxY);
  }

  if (!hasLocalBounds || localQuads.empty()) {
    return Result<TextBounds, std::string>::makeResult(TextBounds{});
  }

  const glm::vec2 shift = computeAlignedOffset2D(localBounds, desc.layout, desc.x, desc.y);
  const size_t queueStart = uiQueue_.size();
  TextBounds finalBounds{};
  bool hasFinalBounds = false;
  for (UiQuad quad : localQuads) {
    quad.minX += shift.x;
    quad.maxX += shift.x;
    quad.minY += shift.y;
    quad.maxY += shift.y;
    if (!uiQueue_.empty() && uiBatchLess(quad, uiQueue_.back())) {
      uiQueueNeedsSort_ = true;
    }
    uiQueue_.push_back(quad);

    growBounds(finalBounds, hasFinalBounds, quad.minX, quad.minY, quad.maxX,
               quad.maxY);
  }

  const float finalWidth = std::max(finalBounds.maxX - finalBounds.minX, 0.0f);
  const float finalHeight = std::max(finalBounds.maxY - finalBounds.minY, 0.0f);
  const float containerWidth =
      desc.layout.maxWidthPx > 0.0f ? desc.layout.maxWidthPx : finalWidth;
  const float containerHeight =
      desc.layout.maxHeightPx > 0.0f ? desc.layout.maxHeightPx : finalHeight;

  float desiredMinX = finalBounds.minX;
  switch (desc.layout.alignH) {
  case TextAlignH::Left:
    desiredMinX = desc.x;
    break;
  case TextAlignH::Center:
    desiredMinX = desc.x + (containerWidth - finalWidth) * 0.5f;
    break;
  case TextAlignH::Right:
    desiredMinX = desc.x + (containerWidth - finalWidth);
    break;
  }

  float desiredMinY = finalBounds.minY;
  if (desc.layout.alignV == TextAlignV::Top) {
    desiredMinY = desc.y;
  } else if (desc.layout.alignV == TextAlignV::Middle) {
    desiredMinY = desc.y + (containerHeight - finalHeight) * 0.5f;
  } else if (desc.layout.alignV == TextAlignV::Bottom) {
    desiredMinY = desc.y + (containerHeight - finalHeight);
  }

  const float corrX = desiredMinX - finalBounds.minX;
  const float corrY = desiredMinY - finalBounds.minY;
  if (std::abs(corrX) > kBatchPxRangeEpsilon ||
      std::abs(corrY) > kBatchPxRangeEpsilon) {
    for (size_t i = queueStart; i < uiQueue_.size(); ++i) {
      uiQueue_[i].minX += corrX;
      uiQueue_[i].maxX += corrX;
      uiQueue_[i].minY += corrY;
      uiQueue_[i].maxY += corrY;
    }
    finalBounds.minX += corrX;
    finalBounds.maxX += corrX;
    finalBounds.minY += corrY;
    finalBounds.maxY += corrY;
  }

  return Result<TextBounds, std::string>::makeResult(finalBounds);
}

Result<TextBounds, std::string>
TextRendererImpl::enqueue3D(const Text3DDesc &desc,
                            std::pmr::memory_resource &scratch) {
  NURI_PROFILER_FUNCTION();
  auto layoutResult =
      layouter_.layoutUtf8(desc.utf8, desc.style, desc.layout, scratch, scratch);
  if (layoutResult.hasError()) {
    return makeError<TextBounds>("TextRenderer::enqueue3D: ",
                                 layoutResult.error());
  }

  const TextLayout &layout = layoutResult.value();
  if (layout.glyphs.empty()) {
    return Result<TextBounds, std::string>::makeResult(TextBounds{});
  }

  TextBounds localBounds{};
  bool hasLocalBounds = false;
  size_t validGlyphCount = 0;
  for (const LayoutGlyph &glyph : layout.glyphs) {
    const uint32_t atlas = fonts_.atlasBindlessIndex(glyph.atlasPage);
    if (atlas == 0) {
      continue;
    }
    ++validGlyphCount;

    const float minX = glyph.x + glyph.metrics.planeMinX;
    const float minY = glyph.y - glyph.metrics.planeMaxY;
    const float maxX = glyph.x + glyph.metrics.planeMaxX;
    const float maxY = glyph.y - glyph.metrics.planeMinY;

    growBounds(localBounds, hasLocalBounds, minX, minY, maxX, maxY);
  }

  if (!hasLocalBounds || validGlyphCount == 0) {
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
  worldQueue_.reserve(worldQueue_.size() + validGlyphCount);

  const glm::vec2 shift = computeAlignedOffsetLocal(localBounds, desc.layout);
  TextBounds finalBounds{};
  finalBounds.minX = localBounds.minX + shift.x;
  finalBounds.maxX = localBounds.maxX + shift.x;
  finalBounds.minY = localBounds.minY + shift.y;
  finalBounds.maxY = localBounds.maxY + shift.y;
  for (const LayoutGlyph &glyph : layout.glyphs) {
    const uint32_t atlas = fonts_.atlasBindlessIndex(glyph.atlasPage);
    if (atlas == 0) {
      continue;
    }
    WorldQuad quad{};
    quad.minX = glyph.x + glyph.metrics.planeMinX + shift.x;
    quad.minY = glyph.y - glyph.metrics.planeMaxY + shift.y;
    quad.maxX = glyph.x + glyph.metrics.planeMaxX + shift.x;
    quad.maxY = glyph.y - glyph.metrics.planeMinY + shift.y;
    quad.uvMinX = glyph.metrics.uvMinX;
    quad.uvMinY = glyph.metrics.uvMinY;
    quad.uvMaxX = glyph.metrics.uvMaxX;
    quad.uvMaxY = glyph.metrics.uvMaxY;
    quad.pxRange = fonts_.pxRange(glyph.font);
    quad.color = color;
    quad.atlas = atlas;
    quad.transformId = transformId;
    hashWorldQuad(worldQueueHash_, quad);
    if (!worldQueue_.empty() && worldBatchLess(quad, worldQueue_.back())) {
      worldQueueNeedsSort_ = true;
    }
    worldQueue_.push_back(quad);
  }

  return Result<TextBounds, std::string>::makeResult(finalBounds);
}

void TextRendererImpl::clear() {
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

void TextRendererImpl::resetPerfCounters() { perf_ = PerfCounters{}; }

void TextRendererImpl::emitPerfValidation(uint64_t frameIndex) {
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
      static_cast<unsigned int>(perf_.worldBatches), perf_.worldVertexUploadBytes,
      perf_.worldIndexUploadBytes);
}

Result<bool, std::string> TextRendererImpl::compileUiShaders() {
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

Result<bool, std::string> TextRendererImpl::compileWorldShaders() {
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
TextRendererImpl::ensureUiPipeline(Format colorFormat) {
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
TextRendererImpl::ensureWorldPipeline(Format colorFormat, Format depthFormat) {
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

void TextRendererImpl::syncFrames(std::pmr::vector<FrameBuffers> &frames) {
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

uint32_t TextRendererImpl::frameSlot(std::pmr::vector<FrameBuffers> &frames) {
  syncFrames(frames);
  const uint32_t count = static_cast<uint32_t>(frames.size());
  NURI_ASSERT(count > 0, "TextRenderer frame buffer count must be > 0");
  return gpu_.getSwapchainImageIndex() % count;
}

Result<bool, std::string>
TextRendererImpl::ensureFrameBuffers(FrameBuffers &frame, size_t vbBytes,
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
      const uint32_t baseVertex =
          static_cast<uint32_t>(i * kVerticesPerQuad);
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
TextRendererImpl::ensureWorldInstanceBuffer(FrameBuffers &frame, size_t bytes,
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

Result<bool, std::string> TextRendererImpl::uploadUi(uint32_t slot) {
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

Result<bool, std::string> TextRendererImpl::uploadWorld(uint32_t slot) {
  NURI_PROFILER_FUNCTION();
  FrameBuffers &frame = worldFrames_[slot];
  const size_t transformBytes =
      resolvedWorldTransforms_.size() * sizeof(ResolvedWorldTransform);
  const size_t instancesOffset = alignUp(transformBytes, 16u);
  const size_t instanceBytes = worldInstances_.size() * sizeof(WorldGlyphInstance);
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
        std::span<const std::byte>(
            reinterpret_cast<const std::byte *>(resolvedWorldTransforms_.data()),
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

void TextRendererImpl::buildUiGeometry() {
  NURI_PROFILER_ZONE("TextRenderer::buildUiGeometry", NURI_PROFILER_COLOR_CMD_DRAW);
  if (uiQueueNeedsSort_) {
    std::stable_sort(uiQueue_.begin(), uiQueue_.end(), uiBatchLess);
    uiQueueNeedsSort_ = false;
  }
  uiVerts_.clear();
  uiBatches_.clear();

  const size_t quadCount = uiQueue_.size();
  uiVerts_.resize(quadCount * 4u);
  uiBatches_.reserve(uiQueue_.size());

  uint32_t currentAtlas = std::numeric_limits<uint32_t>::max();
  float currentPxRange = -1.0f;
  size_t vertexWrite = 0;
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

    uiVerts_[vertexWrite + 0u] =
        UiVertex{{q.minX, q.minY}, {q.uvMinX, q.uvMinY}, q.color};
    uiVerts_[vertexWrite + 1u] =
        UiVertex{{q.maxX, q.minY}, {q.uvMaxX, q.uvMinY}, q.color};
    uiVerts_[vertexWrite + 2u] =
        UiVertex{{q.maxX, q.maxY}, {q.uvMaxX, q.uvMaxY}, q.color};
    uiVerts_[vertexWrite + 3u] =
        UiVertex{{q.minX, q.maxY}, {q.uvMinX, q.uvMaxY}, q.color};
    vertexWrite += 4u;
    indexWrite += 6u;
    uiBatches_.back().indexCount += 6u;
  }
  uiVerts_.resize(vertexWrite);
  perf_.uiGlyphs = static_cast<uint32_t>(uiVerts_.size() / 4u);
  perf_.uiBatches = static_cast<uint32_t>(uiBatches_.size());
  NURI_PROFILER_ZONE_END();
}

void TextRendererImpl::buildWorldGeometry(const CameraFrameState &camera) {
  NURI_PROFILER_ZONE("TextRenderer::buildWorldGeometry",
                     NURI_PROFILER_COLOR_CMD_DRAW);
  if (worldQueueNeedsSort_) {
    std::stable_sort(worldQueue_.begin(), worldQueue_.end(), worldBatchLess);
    worldQueueNeedsSort_ = false;
  }
  worldInstances_.clear();
  worldBatches_.clear();

  const size_t quadCount = worldQueue_.size();
  worldInstances_.resize(quadCount);
  worldBatches_.reserve(worldQueue_.size());
  const BillboardFrameBasis basis = buildBillboardFrameBasis(camera);
  resolvedWorldTransforms_.resize(worldTransforms_.size());
  for (size_t i = 0; i < worldTransforms_.size(); ++i) {
    const glm::mat4 world = resolveWorldFromBillboard(worldTransforms_[i], basis);
    resolvedWorldTransforms_[i].basisX = glm::vec4(glm::vec3(world[0]), 0.0f);
    resolvedWorldTransforms_[i].basisY = glm::vec4(glm::vec3(world[1]), 0.0f);
    resolvedWorldTransforms_[i].translation =
        glm::vec4(glm::vec3(world[3]), 0.0f);
  }

  uint32_t currentAtlas = std::numeric_limits<uint32_t>::max();
  float currentPxRange = -1.0f;
  uint32_t instanceWrite = 0;
  for (const WorldQuad &q : worldQueue_) {
    if (currentAtlas != q.atlas ||
        std::abs(currentPxRange - q.pxRange) > kBatchPxRangeEpsilon) {
      currentAtlas = q.atlas;
      currentPxRange = q.pxRange;
      worldBatches_.push_back(WorldBatch{
          .atlas = q.atlas,
          .pxRange = q.pxRange,
          .firstInstance = instanceWrite,
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
    WorldGlyphInstance inst{};
    inst.rectMinMax = glm::vec4(q.minX, q.minY, q.maxX, q.maxY);
    inst.uvMinMax = glm::vec4(q.uvMinX, q.uvMinY, q.uvMaxX, q.uvMaxY);
    inst.color = q.color;
    inst.transformIndex = static_cast<uint32_t>(transformIdx);
    worldInstances_[instanceWrite] = inst;
    ++instanceWrite;
    worldBatches_.back().instanceCount += 1u;
  }
  worldInstances_.resize(instanceWrite);
  perf_.worldGlyphs = static_cast<uint32_t>(worldInstances_.size());
  perf_.worldBatches = static_cast<uint32_t>(worldBatches_.size());
  NURI_PROFILER_ZONE_END();
}

Result<bool, std::string>
TextRendererImpl::append3DPasses(RenderFrameContext &frame, RenderPassList &out) {
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
      hasDepth ? gpu_.getTextureFormat(frame.sharedDepthTexture) : Format::Count;
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
  pass.draws = std::span<const DrawItem>(worldDraws_.data(), worldDraws_.size());
  pass.dependencyBuffers = std::span<const BufferHandle>(
      worldDependencyBuffers_.data(), worldDependencyBuffers_.size());
  pass.debugLabel = "Text3D Pass";
  pass.debugColor = 0xff44cc88u;
  out.push_back(pass);
  worldAppended_ = true;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TextRendererImpl::append2DPasses(RenderFrameContext &, RenderPassList &out) {
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

  const glm::mat4 proj =
      glm::ortho(0.0f, std::max(1.0f, static_cast<float>(logicalW)),
                 std::max(1.0f, static_cast<float>(logicalH)), 0.0f, -1.0f,
                 1.0f);
  const FrameBuffers &buffers = uiFrames_[slot];
  for (const UiBatch &b : uiBatches_) {
    uiPcs_.push_back(UiPC{.proj = proj, .atlas = b.atlas, .pxRange = b.pxRange});
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

void TextRendererImpl::destroyGpu() {
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
} // namespace

std::unique_ptr<TextRenderer> TextRenderer::create(const CreateDesc &desc) {
  return std::make_unique<TextRendererImpl>(desc);
}

} // namespace nuri
