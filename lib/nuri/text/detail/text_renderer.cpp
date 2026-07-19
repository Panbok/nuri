#include "nuri/text/text_renderer.h"
#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/gpu_descriptors.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/render_graph/render_graph.h"
#include "nuri/gfx/shader.h"
#include "nuri/math/utils.h"
#include "nuri/pch.h"
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
[[nodiscard]] uint64_t hashMix(uint64_t hash, uint64_t value) {
  constexpr uint64_t kFnvPrime = 1099511628211ull;
  hash ^= value;
  hash *= kFnvPrime;
  return hash;
}
template <typename T> void hashValue(uint64_t &hash, const T &value) {
  const auto bytes = std::as_bytes(std::span(&value, 1u));
  size_t offset = 0u;
  while (offset + sizeof(uint64_t) <= bytes.size()) {
    uint64_t word;
    std::memcpy(&word, bytes.data() + offset, sizeof(word));
    hash = hashMix(hash, word);
    offset += sizeof(word);
  }
  if (offset != bytes.size()) {
    uint64_t tail = 0u;
    std::memcpy(&tail, bytes.data() + offset, bytes.size() - offset);
    hash = hashMix(hash, tail);
  }
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
template <typename Queue, typename Less, typename Decorate>
void appendGlyphQuads(FontManager &fonts, const TextLayout &layout,
                      uint32_t color, Queue &queue, bool &needsSort,
                      TextBounds &bounds, bool &hasBounds, Decorate decorate,
                      Less less) {
  FontHandle font = kInvalidFontHandle;
  float pxRange = 4.0f;
  const size_t start = queue.size();
  queue.reserve(start + layout.glyphs.size());
  for (const LayoutGlyph &glyph : layout.glyphs) {
    const uint32_t atlas = fonts.atlasBindlessIndex(glyph.atlasPage);
    if (atlas == 0u) {
      continue;
    }
    if (glyph.font != font) {
      font = glyph.font;
      pxRange = fonts.pxRange(font);
    }
    typename Queue::value_type quad{};
    quad.minX = glyph.x + glyph.metrics.planeMinX;
    quad.minY = glyph.y - glyph.metrics.planeMaxY;
    quad.maxX = glyph.x + glyph.metrics.planeMaxX;
    quad.maxY = glyph.y - glyph.metrics.planeMinY;
    quad.uvMinX = glyph.metrics.uvMinX;
    quad.uvMinY = glyph.metrics.uvMinY;
    quad.uvMaxX = glyph.metrics.uvMaxX;
    quad.uvMaxY = glyph.metrics.uvMaxY;
    quad.pxRange = pxRange;
    quad.color = color;
    quad.atlas = atlas;
    decorate(quad, glyph);
    if (start != 0u && queue.size() == start && less(quad, queue.back())) {
      needsSort = true;
    }
    queue.push_back(quad);
    growBounds(bounds, hasBounds, quad.minX, quad.minY, quad.maxX, quad.maxY);
  }
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
  hashValue(hash, transform);
}

void TextRenderer::hashUiQuad(uint64_t &hash, const UiQuad &quad) {
  hashValue(hash, quad);
}

void TextRenderer::hashWorldQuad(uint64_t &hash, const WorldQuad &quad) {
  hashValue(hash, quad);
}

uint64_t TextRenderer::hashCameraFrameState(const CameraFrameState &camera) {
  uint64_t hash = kHashSeed;
  hashValue(hash, camera.view);
  hashValue(hash, camera.cameraPos);
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
    right = safeNormalize(glm::cross(up, forward), glm::vec3(1.0f, 0.0f, 0.0f));
    forward = safeNormalize(glm::cross(right, up), forward);
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
      memory_(desc.memory), shaderPaths_(desc.shaderPaths), uiQueue_(&memory_),
      worldQueue_(&memory_), worldTransforms_(&memory_),
      resolvedWorldTransforms_(&memory_), worldInstances_(&memory_),
      uiVerts_(&memory_), uiBatches_(&memory_), worldBatches_(&memory_),
      uiDraws_(&memory_), worldDraws_(&memory_),
      worldTransparentDraws_(&memory_), uiPcs_(&memory_), worldPcs_(&memory_),
      uiFrames_(&memory_), worldFrames_(&memory_),
      worldTransparentTextureReadList_(&memory_) {}

TextRenderer::~TextRenderer() { destroyGpu(); }

void TextRenderer::beginFrame(uint64_t frameIndex) {
  if (frameIndex_ == frameIndex) {
    return;
  }
  frameIndex_ = frameIndex;
  clear();
}

Result<TextBounds, std::string>
TextRenderer::enqueue2D(const Text2DDesc &desc,
                        std::pmr::memory_resource &scratch) {
  NURI_PROFILER_FUNCTION();
  auto layoutResult =
      layouter_.layoutUtf8(desc.utf8, desc.style, desc.layout, scratch);
  if (layoutResult.hasError()) {
    return makeError<TextBounds>("TextRenderer::enqueue2D: ",
                                 layoutResult.error());
  }
  const TextLayout &layout = *layoutResult.value();
  const uint32_t color = packColor(desc.fillColor);
  const size_t queueStart = uiQueue_.size();
  TextBounds localBounds{};
  bool hasLocalBounds = false;
  appendGlyphQuads(
      fonts_, layout, color, uiQueue_, uiQueueNeedsSort_, localBounds,
      hasLocalBounds, [](UiQuad &, const auto &) {}, uiBatchLess);
  if (uiQueue_.size() == queueStart) {
    return Result<TextBounds, std::string>::makeResult(TextBounds{});
  }
  const glm::vec2 shift =
      computeAlignedOffset2D(localBounds, desc.layout, desc.x, desc.y);
  for (size_t i = queueStart; i < uiQueue_.size(); ++i) {
    uiQueue_[i].minX += shift.x;
    uiQueue_[i].maxX += shift.x;
    uiQueue_[i].minY += shift.y;
    uiQueue_[i].maxY += shift.y;
    hashUiQuad(uiQueueHash_, uiQueue_[i]);
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
  auto layoutResult =
      layouter_.layoutUtf8(desc.utf8, desc.style, desc.layout, scratch);
  if (layoutResult.hasError()) {
    return makeError<TextBounds>("TextRenderer::enqueue3D: ",
                                 layoutResult.error());
  }
  const TextLayout &layout = *layoutResult.value();
  if (layout.glyphs.empty()) {
    return Result<TextBounds, std::string>::makeResult(TextBounds{});
  }
  const glm::mat4 world = decodeWorld(desc.worldFromText);
  worldTransforms_.push_back(
      WorldTransform{.worldFromText = world, .billboard = desc.billboard});
  const uint32_t transformId = static_cast<uint32_t>(worldTransforms_.size());
  worldHasBillboards_ =
      worldHasBillboards_ || (desc.billboard != TextBillboardMode::None);
  const uint32_t color = packColor(desc.fillColor);
  const size_t queueStart = worldQueue_.size();
  TextBounds localBounds{};
  bool hasLocalBounds = false;
  appendGlyphQuads(
      fonts_, layout, color, worldQueue_, worldQueueNeedsSort_, localBounds,
      hasLocalBounds,
      [this, transformId](WorldQuad &quad, const LayoutGlyph &glyph) {
        quad.transformId = transformId;
        quad.atlasTexture = fonts_.atlasTexture(glyph.atlasPage);
      },
      worldBatchLess);
  if (worldQueue_.size() == queueStart) {
    worldTransforms_.pop_back();
    return Result<TextBounds, std::string>::makeResult(TextBounds{});
  }
  const glm::vec2 shift = computeAlignedOffsetLocal(localBounds, desc.layout);
  hashWorldTransform(worldQueueHash_, worldTransforms_.back());
  for (size_t i = queueStart; i < worldQueue_.size(); ++i) {
    worldQueue_[i].minX += shift.x;
    worldQueue_[i].minY += shift.y;
    worldQueue_[i].maxX += shift.x;
    worldQueue_[i].maxY += shift.y;
    hashWorldQuad(worldQueueHash_, worldQueue_[i]);
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
  uiDraws_.clear();
  worldDraws_.clear();
  worldTransparentDraws_.clear();
  uiPcs_.clear();
  worldPcs_.clear();
  uiAppended_ = false;
  worldAppended_ = false;
  uiQueueNeedsSort_ = false;
  worldQueueNeedsSort_ = false;
  uiQueueHash_ = kHashSeed;
  worldQueueHash_ = kHashSeed;
  worldHasBillboards_ = false;
  worldGlyphBufferAddress_ = 0;
  worldTransformBufferAddress_ = 0;
  worldDependencyBuffer_ = {};
  worldTransparentTextureReadList_.clear();
  worldPreparedSlot_ = 0;
}

Result<bool, std::string>
TextRenderer::compileShaders(std::string_view name,
                             const std::filesystem::path &vertexPath,
                             const std::filesystem::path &fragmentPath,
                             ShaderHandle &vertex, ShaderHandle &fragment) {
  if (::nuri::isValid(vertex) && ::nuri::isValid(fragment)) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (vertexPath.empty() || fragmentPath.empty()) {
    return makeError<bool>("TextRenderer: shader paths are empty");
  }
  auto helper = Shader::create(name, gpu_);
  auto vs = helper->compileFromFile(vertexPath.string(), ShaderStage::Vertex);
  if (vs.hasError()) {
    return Result<bool, std::string>::makeError(vs.error());
  }
  auto fs =
      helper->compileFromFile(fragmentPath.string(), ShaderStage::Fragment);
  if (fs.hasError()) {
    gpu_.destroyShaderModule(vs.value());
    return Result<bool, std::string>::makeError(fs.error());
  }
  vertex = vs.value();
  fragment = fs.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> TextRenderer::ensureUiPipeline(Format colorFormat) {
  if (::nuri::isValid(uiPipeline_) && uiPipelineColor_ == colorFormat) {
    return Result<bool, std::string>::makeResult(true);
  }
  auto shaders = compileShaders("text_2d_mtsdf", shaderPaths_.uiVertex,
                                shaderPaths_.uiFragment, uiVs_, uiFs_);
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
TextRenderer::ensureWorldPipeline(Format colorFormat, Format depthFormat) {
  if (::nuri::isValid(worldPipeline_) && worldPipelineColor_ == colorFormat &&
      worldPipelineDepth_ == depthFormat) {
    return Result<bool, std::string>::makeResult(true);
  }
  auto shaders = compileShaders("text_3d_mtsdf", shaderPaths_.worldVertex,
                                shaderPaths_.worldFragment, worldVs_, worldFs_);
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
  desc.rasterState = depthFormat != Format::Count
                         ? makeRasterPipelineState(
                               DepthState{.compareOp = CompareOp::LessEqual,
                                          .isDepthWriteEnabled = false})
                         : RasterPipelineState{};
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
  frames.clear();
  frames.resize(swapchainCount);
}

uint32_t TextRenderer::frameSlot(std::pmr::vector<FrameBuffers> &frames) {
  syncFrames(frames);
  const uint32_t count = static_cast<uint32_t>(frames.size());
  return static_cast<uint32_t>(frameIndex_ % count);
}

Result<bool, std::string>
TextRenderer::ensureUiFrameBuffers(FrameBuffers &frame, size_t vertexBytes,
                                   size_t quadCount) {
  constexpr size_t kIndicesPerQuad = 6u;
  constexpr size_t kVerticesPerQuad = 4u;
  auto vertex =
      ensureDynamicBufferCapacity(gpu_, frame.vertex,
                                  BufferDesc{.usage = BufferUsage::Vertex,
                                             .storage = Storage::HostVisible,
                                             .size = vertexBytes},
                                  "text_ui_vertices");
  if (vertex.hasError()) {
    return vertex;
  }
  if (vertex.value()) {
    frame.lastUploadHash = 0u;
  }
  auto index = ensureDynamicBufferCapacity(
      gpu_, frame.index,
      BufferDesc{.usage = BufferUsage::Index,
                 .storage = Storage::HostVisible,
                 .size = quadCount * kIndicesPerQuad * sizeof(uint32_t)},
      "text_ui_indices");
  if (index.hasError()) {
    return index;
  }
  if (index.value()) {
    const size_t indexCount = frame.index.capacityBytes / sizeof(uint32_t);
    std::pmr::vector<uint32_t> indexPattern(&memory_);
    indexPattern.resize(indexCount);
    for (size_t i = 0; i < indexCount / kIndicesPerQuad; ++i) {
      const uint32_t baseVertex = static_cast<uint32_t>(i * kVerticesPerQuad);
      const size_t idx = i * kIndicesPerQuad;
      indexPattern[idx + 0u] = baseVertex;
      indexPattern[idx + 1u] = baseVertex + 1u;
      indexPattern[idx + 2u] = baseVertex + 2u;
      indexPattern[idx + 3u] = baseVertex + 2u;
      indexPattern[idx + 4u] = baseVertex + 3u;
      indexPattern[idx + 5u] = baseVertex;
    }
    auto up = gpu_.updateBuffer(
        frame.index.buffer->handle(),
        std::span<const std::byte>(
            reinterpret_cast<const std::byte *>(indexPattern.data()),
            frame.index.capacityBytes),
        0);
    if (up.hasError()) {
      retireDynamicBuffer(frame.index);
      return Result<bool, std::string>::makeError(up.error());
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> TextRenderer::uploadUi(uint32_t slot) {
  NURI_PROFILER_FUNCTION();
  constexpr size_t kVerticesPerQuad = 4u;
  FrameBuffers &frame = uiFrames_[slot];
  const size_t vbBytes = uiVerts_.size() * sizeof(UiVertex);
  const size_t quadCount = uiVerts_.size() / kVerticesPerQuad;
  auto ensure = ensureUiFrameBuffers(frame, vbBytes, quadCount);
  if (ensure.hasError()) {
    return ensure;
  }
  if (frame.lastUploadHash == uiQueueHash_) {
    return Result<bool, std::string>::makeResult(true);
  }
  auto up = gpu_.updateBuffer(
      frame.vertex.buffer->handle(),
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(uiVerts_.data()), vbBytes),
      0);
  if (up.hasError()) {
    return Result<bool, std::string>::makeError(up.error());
  }
  frame.lastUploadHash = uiQueueHash_;
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
  auto ensure = ensureDynamicBufferCapacity(
      gpu_, frame.vertex,
      BufferDesc{.usage = BufferUsage::Storage | BufferUsage::Vertex,
                 .storage = Storage::HostVisible,
                 .size = totalBytes},
      "text_world_instances");
  if (ensure.hasError()) {
    return ensure;
  }
  auto up = gpu_.updateBuffer(
      frame.vertex.buffer->handle(),
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(resolvedWorldTransforms_.data()),
          transformBytes),
      0);
  if (up.hasError()) {
    return Result<bool, std::string>::makeError(up.error());
  }
  up = gpu_.updateBuffer(
      frame.vertex.buffer->handle(),
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(worldInstances_.data()),
          instanceBytes),
      instancesOffset);
  if (up.hasError()) {
    return Result<bool, std::string>::makeError(up.error());
  }
  worldTransformBufferAddress_ =
      gpu_.getBufferDeviceAddress(frame.vertex.buffer->handle(), 0u);
  worldGlyphBufferAddress_ = gpu_.getBufferDeviceAddress(
      frame.vertex.buffer->handle(), instancesOffset);
  return Result<bool, std::string>::makeResult(true);
}

void TextRenderer::buildUiGeometry() {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_DRAW);
  const bool canReuseUiGeometry =
      uiGeometryValid_ && uiQueueHash_ == lastBuiltUiQueueHash_;
  if (canReuseUiGeometry) {
    return;
  }
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
  uiGeometryValid_ = true;
  lastBuiltUiQueueHash_ = uiQueueHash_;
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
  uint32_t currentTransformId = std::numeric_limits<uint32_t>::max();
  float currentPxRange = -1.0f;
  const glm::mat4 view = camera.view;
  for (const WorldQuad &q : worldQueue_) {
    if (currentAtlas != q.atlas || currentTransformId != q.transformId ||
        std::abs(currentPxRange - q.pxRange) > kBatchPxRangeEpsilon) {
      currentAtlas = q.atlas;
      currentTransformId = q.transformId;
      currentPxRange = q.pxRange;
      const size_t transformIdx = static_cast<size_t>(q.transformId - 1u);
      const glm::vec3 translation =
          glm::vec3(resolvedWorldTransforms_[transformIdx].translation);
      worldBatches_.push_back(WorldBatch{
          .atlas = q.atlas,
          .pxRange = q.pxRange,
          .firstInstance = static_cast<uint32_t>(worldInstances_.size()),
          .instanceCount = 0,
          .atlasTexture = q.atlasTexture,
          .sortDepth = -(view * glm::vec4(translation, 1.0f)).z,
      });
    }
    const size_t transformIdx = static_cast<size_t>(q.transformId) - 1u;
    worldInstances_.push_back(WorldGlyphInstance{
        .rectMinMax = glm::vec4(q.minX, q.minY, q.maxX, q.maxY),
        .uvMinMax = glm::vec4(q.uvMinX, q.uvMinY, q.uvMaxX, q.uvMaxY),
        .color = q.color,
        .transformIndex = static_cast<uint32_t>(transformIdx),
    });
    worldBatches_.back().instanceCount += 1u;
  }
  NURI_PROFILER_ZONE_END();
}

Result<bool, std::string>
TextRenderer::prepareWorldRenderState(RenderFrameContext &frame,
                                      TextureHandle &outDepth,
                                      Format &outDepthFormat) {
  if (worldQueue_.empty() || worldAppended_) {
    return Result<bool, std::string>::makeResult(false);
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
  outDepth = resolveFrameDepthTexture(frame);
  outDepthFormat = ::nuri::isValid(outDepth) ? gpu_.getTextureFormat(outDepth)
                                             : Format::Count;
  const TextureHandle colorTexture = frame.sharedResources.frameColorTexture;
  const Format colorFormat = ::nuri::isValid(colorTexture)
                                 ? gpu_.getTextureFormat(colorTexture)
                                 : gpu_.getSwapchainFormat();
  auto pipeline = ensureWorldPipeline(colorFormat, outDepthFormat);
  if (pipeline.hasError()) {
    return pipeline;
  }
  worldPreparedSlot_ = frameSlot(worldFrames_);
  auto upload = uploadWorld(worldPreparedSlot_);
  if (upload.hasError()) {
    return upload;
  }
  worldDependencyBuffer_ =
      worldFrames_[worldPreparedSlot_].vertex.buffer->handle();
  worldDraws_.clear();
  worldPcs_.clear();
  worldTransparentTextureReadList_.clear();
  worldDraws_.reserve(worldBatches_.size());
  worldPcs_.reserve(worldBatches_.size());
  const glm::mat4 viewProj =
      cameraCurrentUnjitteredViewProjection(frame.camera);
  for (const WorldBatch &batch : worldBatches_) {
    worldPcs_.push_back(WorldPC{
        .viewProj = viewProj,
        .glyphBufferAddress = worldGlyphBufferAddress_,
        .transformBufferAddress = worldTransformBufferAddress_,
        .atlas = batch.atlas,
        .pxRange = batch.pxRange,
        .alphaDiscardThreshold = 1.0e-3f,
    });
    DrawItem &draw = worldDraws_.emplace_back();
    draw.pipeline = worldPipeline_;
    draw.vertexCount = 6;
    draw.instanceCount = batch.instanceCount;
    draw.firstInstance = batch.firstInstance;
    draw.pushConstants = std::span<const std::byte>(
        reinterpret_cast<const std::byte *>(&worldPcs_.back()),
        sizeof(WorldPC));
    draw.debugLabel = "Text3D Batch";
    draw.debugColor = 0xff44cc88u;
    if (::nuri::isValid(outDepth)) {
      draw.useDepthState = true;
      draw.depthState = {.compareOp = CompareOp::LessEqual,
                         .isDepthWriteEnabled = false};
    }
    appendWorldTransparentTextureRead(batch.atlasTexture);
  }
  return Result<bool, std::string>::makeResult(true);
}

void TextRenderer::appendWorldTransparentTextureRead(TextureHandle texture) {
  if (std::ranges::find(worldTransparentTextureReadList_, texture) ==
      worldTransparentTextureReadList_.end()) {
    worldTransparentTextureReadList_.push_back(texture);
  }
}

Result<bool, std::string> TextRenderer::append3DGraphPass(
    RenderFrameContext &frame, RenderGraphBuilder &graph,
    RenderGraphTextureId sceneDepthGraphTexture, bool hasPriorColorPass) {
  NURI_PROFILER_FUNCTION();
  TextureHandle depthTexture{};
  TextureHandle colorTexture{};
  [[maybe_unused]] Format depthFormat = Format::Count;
  auto prepare = prepareWorldRenderState(frame, depthTexture, depthFormat);
  if (prepare.hasError()) {
    return prepare;
  }
  if (!prepare.value()) {
    return Result<bool, std::string>::makeResult(true);
  }
  const bool hasDepth = ::nuri::isValid(depthTexture);
  colorTexture = frame.sharedResources.frameColorTexture;
  int32_t w = 0;
  int32_t h = 0;
  gpu_.getFramebufferSize(w, h);
  RenderGraphGraphicsPassDesc desc{};
  desc.color = {.loadOp = (hasPriorColorPass || ::nuri::isValid(colorTexture))
                              ? LoadOp::Load
                              : LoadOp::Clear,
                .storeOp = StoreOp::Store,
                .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
  if (::nuri::isValid(colorTexture)) {
    desc.colorTexture =
        ::nuri::isValid(frame.sharedResources.frameColorGraphTexture)
            ? frame.sharedResources.frameColorGraphTexture
            : graph.importTexture(colorTexture, "text3d_pass_color_texture")
                  .value();
  }
  desc.useViewport = true;
  desc.viewport = {.x = 0.0f,
                   .y = 0.0f,
                   .width = std::max(1.0f, static_cast<float>(w)),
                   .height = std::max(1.0f, static_cast<float>(h)),
                   .minDepth = 0.0f,
                   .maxDepth = 1.0f};
  if (hasDepth) {
    desc.depth = {.loadOp = LoadOp::Load,
                  .storeOp = StoreOp::Store,
                  .clearDepth = 1.0f,
                  .clearStencil = 0};
    if (::nuri::isValid(sceneDepthGraphTexture)) {
      desc.depthTexture = sceneDepthGraphTexture;
    } else {
      desc.depthTexture =
          graph.importTexture(depthTexture, "text3d_pass_depth_texture")
              .value();
    }
  }
  desc.draws =
      std::span<const DrawItem>(worldDraws_.data(), worldDraws_.size());
  desc.dependencyBuffers =
      std::span<const BufferHandle>(&worldDependencyBuffer_, 1u);
  desc.debugLabel = "Text3D Pass";
  desc.debugColor = 0xff44cc88u;
  const RenderGraphPassId pass = graph.addGraphicsPass(desc).value();
  for (const TextureHandle texture : worldTransparentTextureReadList_) {
    [[maybe_unused]] const bool read =
        graph
            .addTextureRead(
                pass,
                graph.importTexture(texture, "text3d_atlas_texture").value())
            .value();
  }
  worldAppended_ = true;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> TextRenderer::buildTransparentStageContribution(
    RenderFrameContext &frame, TransparentStageContribution &out) {
  NURI_PROFILER_FUNCTION();
  out = {};
  TextureHandle depthTexture{};
  [[maybe_unused]] Format depthFormat = Format::Count;
  auto prepare = prepareWorldRenderState(frame, depthTexture, depthFormat);
  if (prepare.hasError()) {
    return prepare;
  }
  if (!prepare.value()) {
    return Result<bool, std::string>::makeResult(true);
  }
  worldTransparentDraws_.clear();
  worldTransparentDraws_.reserve(worldBatches_.size());
  for (size_t i = 0; i < worldBatches_.size(); ++i) {
    worldTransparentDraws_.push_back(TransparentStageSortableDraw{
        .draw = worldDraws_[i],
        .sortDepth = worldBatches_[i].sortDepth,
        .stableOrder = static_cast<uint32_t>(i),
    });
  }
  out.sortableDraws = std::span<const TransparentStageSortableDraw>(
      worldTransparentDraws_.data(), worldTransparentDraws_.size());
  out.fixedDraws = {};
  out.dependencyBuffers =
      std::span<const BufferHandle>(&worldDependencyBuffer_, 1u);
  out.textureReads =
      std::span<const TextureHandle>(worldTransparentTextureReadList_.data(),
                                     worldTransparentTextureReadList_.size());
  worldAppended_ = true;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TextRenderer::append2DGraphPass(RenderFrameContext &frame,
                                RenderGraphBuilder &graph,
                                bool hasPriorColorPass) {
  NURI_PROFILER_FUNCTION();
  if (uiQueue_.empty() || uiAppended_) {
    return Result<bool, std::string>::makeResult(true);
  }
  buildUiGeometry();
  if (uiBatches_.empty()) {
    uiAppended_ = true;
    return Result<bool, std::string>::makeResult(true);
  }
  const TextureHandle colorTexture = frame.sharedResources.frameColorTexture;
  const Format colorFormat = ::nuri::isValid(colorTexture)
                                 ? gpu_.getTextureFormat(colorTexture)
                                 : gpu_.getSwapchainFormat();
  auto pipeline = ensureUiPipeline(colorFormat);
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
    d.vertexBuffer = buffers.vertex.buffer->handle();
    d.indexBuffer = buffers.index.buffer->handle();
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
  RenderGraphGraphicsPassDesc desc{};
  desc.color = {.loadOp = (hasPriorColorPass || ::nuri::isValid(colorTexture))
                              ? LoadOp::Load
                              : LoadOp::Clear,
                .storeOp = StoreOp::Store,
                .clearColor = {0.0f, 0.0f, 0.0f, 1.0f}};
  if (::nuri::isValid(colorTexture)) {
    desc.colorTexture =
        ::nuri::isValid(frame.sharedResources.frameColorGraphTexture)
            ? frame.sharedResources.frameColorGraphTexture
            : graph.importTexture(colorTexture, "text2d_pass_color_texture")
                  .value();
  }
  desc.useViewport = true;
  desc.viewport = {.x = 0.0f,
                   .y = 0.0f,
                   .width = std::max(1.0f, static_cast<float>(fbW)),
                   .height = std::max(1.0f, static_cast<float>(fbH)),
                   .minDepth = 0.0f,
                   .maxDepth = 1.0f};
  desc.draws = std::span<const DrawItem>(uiDraws_.data(), uiDraws_.size());
  desc.debugLabel = "Text2D Pass";
  desc.debugColor = 0xffcc8844u;
  [[maybe_unused]] const RenderGraphPassId pass =
      graph.addGraphicsPass(desc).value();
  uiAppended_ = true;
  return Result<bool, std::string>::makeResult(true);
}

void TextRenderer::destroyGpu() {
  uiFrames_.clear();
  worldFrames_.clear();
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
