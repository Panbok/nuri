#include "nuri/text/detail/text_renderer.h"
#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/gpu_descriptors.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/render_graph/render_graph.h"
#include "nuri/gfx/shader.h"
#include "nuri/math/utils.h"
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

void TextRenderer::hashGlyphPacket(uint64_t &hash, const GlyphPacket &packet) {
  hashValue(hash, packet.minX);
  hashValue(hash, packet.minY);
  hashValue(hash, packet.maxX);
  hashValue(hash, packet.maxY);
  hashValue(hash, packet.uvMinX);
  hashValue(hash, packet.uvMinY);
  hashValue(hash, packet.uvMaxX);
  hashValue(hash, packet.uvMaxY);
  hashValue(hash, packet.pxRange);
  hashValue(hash, packet.color);
  hashValue(hash, packet.atlas);
  hashValue(hash, packet.domain);
  hashValue(hash, packet.transformId);
  hashValue(hash, packet.atlasTexture);
}

uint64_t TextRenderer::hashCameraFrameState(const CameraFrameState &camera) {
  uint64_t hash = kHashSeed;
  hashValue(hash, camera.view);
  hashValue(hash, camera.cameraPos);
  return hash;
}

bool TextRenderer::glyphBatchLess(const GlyphPacket &a, const GlyphPacket &b) {
  if (a.domain != b.domain) {
    return a.domain < b.domain;
  }
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
      memory_(desc.memory), shaderPaths_(desc.shaderPaths),
      glyphQueue_(&memory_), worldTransforms_(&memory_),
      resolvedWorldTransforms_(&memory_), glyphInstances_(&memory_),
      glyphBatches_(&memory_), uiDraws_(&memory_), worldDraws_(&memory_),
      worldSortDepths_(&memory_), worldTransparentDraws_(&memory_),
      uiPcs_(&memory_), worldPcs_(&memory_),
      glyphBuffers_(
          gpu_,
          BufferDesc{.usage = BufferUsage::Storage | BufferUsage::Vertex,
                     .storage = Storage::HostVisible},
          "text_glyph_packets", &memory_),
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
  return enqueueCommon(TextCommonDesc{.utf8 = desc.utf8,
                                      .style = &desc.style,
                                      .layout = &desc.layout,
                                      .fillColor = desc.fillColor},
                       Text2DPayload{.anchor = {desc.x, desc.y}}, scratch);
}

Result<TextBounds, std::string>
TextRenderer::enqueue3D(const Text3DDesc &desc,
                        std::pmr::memory_resource &scratch) {
  NURI_PROFILER_FUNCTION();
  const glm::mat4 world = decodeWorld(desc.worldFromText);
  worldTransforms_.push_back(
      WorldTransform{.worldFromText = world, .billboard = desc.billboard});
  const uint32_t transformId = static_cast<uint32_t>(worldTransforms_.size());
  const size_t glyphCount = worldGlyphCount_;
  auto result =
      enqueueCommon(TextCommonDesc{.utf8 = desc.utf8,
                                   .style = &desc.style,
                                   .layout = &desc.layout,
                                   .fillColor = desc.fillColor},
                    Text3DPayload{.transformId = transformId}, scratch);
  if (result.hasError() || worldGlyphCount_ == glyphCount) {
    worldTransforms_.pop_back();
  }
  if (!result.hasError() && worldGlyphCount_ != glyphCount) {
    worldHasBillboards_ =
        worldHasBillboards_ || (desc.billboard != TextBillboardMode::None);
  }
  return result;
}

Result<TextBounds, std::string>
TextRenderer::enqueueCommon(const TextCommonDesc &desc,
                            const TextPayload &payload,
                            std::pmr::memory_resource &scratch) {
  auto layoutResult =
      layouter_.layoutUtf8(desc.utf8, *desc.style, *desc.layout, scratch);
  if (layoutResult.hasError()) {
    return makeError<TextBounds>("TextRenderer::enqueue: ",
                                 layoutResult.error());
  }
  const TextLayout &layout = *layoutResult.value();
  const Text2DPayload *ui = std::get_if<Text2DPayload>(&payload);
  const Text3DPayload *world = std::get_if<Text3DPayload>(&payload);
  const size_t queueStart = glyphQueue_.size();
  TextBounds localBounds{};
  bool hasLocalBounds = false;
  appendGlyphQuads(
      fonts_, layout, packColor(desc.fillColor), glyphQueue_,
      glyphQueueNeedsSort_, localBounds, hasLocalBounds,
      [this, ui, world](GlyphPacket &packet, const LayoutGlyph &glyph) {
        packet.domain = ui ? TextDomain::Ui : TextDomain::World;
        packet.transformId = world ? world->transformId : 0u;
        packet.atlasTexture = fonts_.atlasTexture(glyph.atlasPage);
      },
      glyphBatchLess);
  if (glyphQueue_.size() == queueStart) {
    return Result<TextBounds, std::string>::makeResult(TextBounds{});
  }
  const size_t appended = glyphQueue_.size() - queueStart;
  uiGlyphCount_ += ui ? appended : 0u;
  worldGlyphCount_ += world ? appended : 0u;
  const glm::vec2 shift =
      ui ? computeAlignedOffset2D(localBounds, *desc.layout, ui->anchor.x,
                                  ui->anchor.y)
         : computeAlignedOffsetLocal(localBounds, *desc.layout);
  if (world) {
    hashWorldTransform(glyphQueueHash_, worldTransforms_.back());
  }
  for (size_t i = queueStart; i < glyphQueue_.size(); ++i) {
    glyphQueue_[i].minX += shift.x;
    glyphQueue_[i].maxX += shift.x;
    glyphQueue_[i].minY += shift.y;
    glyphQueue_[i].maxY += shift.y;
    hashGlyphPacket(glyphQueueHash_, glyphQueue_[i]);
  }
  return Result<TextBounds, std::string>::makeResult(
      TextBounds{.minX = localBounds.minX + shift.x,
                 .minY = localBounds.minY + shift.y,
                 .maxX = localBounds.maxX + shift.x,
                 .maxY = localBounds.maxY + shift.y});
}

void TextRenderer::clear() {
  glyphQueue_.clear();
  uiGlyphCount_ = 0u;
  worldGlyphCount_ = 0u;
  worldTransforms_.clear();
  uiDraws_.clear();
  worldDraws_.clear();
  worldSortDepths_.clear();
  worldTransparentDraws_.clear();
  uiPcs_.clear();
  worldPcs_.clear();
  uiAppended_ = false;
  worldAppended_ = false;
  glyphQueueNeedsSort_ = false;
  glyphQueueHash_ = kHashSeed;
  glyphPrepared_ = false;
  worldHasBillboards_ = false;
  glyphBufferAddress_ = 0;
  worldTransformBufferAddress_ = 0;
  glyphDependencyBuffer_ = {};
  worldTransparentTextureReadList_.clear();
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
  auto vs =
      compileShaderFile(gpu_, name, vertexPath.string(), ShaderStage::Vertex);
  if (vs.hasError()) {
    return Result<bool, std::string>::makeError(vs.error());
  }
  auto fs = compileShaderFile(gpu_, name, fragmentPath.string(),
                              ShaderStage::Fragment);
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
  RenderPipelineDesc desc{};
  desc.vertexInput = {};
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

Result<bool, std::string> TextRenderer::uploadGlyphPackets() {
  NURI_PROFILER_FUNCTION();
  const size_t transformBytes =
      resolvedWorldTransforms_.size() * sizeof(ResolvedWorldTransform);
  const size_t instancesOffset = alignUp(transformBytes, 16u);
  const size_t instanceBytes = glyphInstances_.size() * sizeof(GlyphInstance);
  const size_t totalBytes = instancesOffset + instanceBytes;
  auto ensure = glyphBuffers_.acquire(
      frameIndex_, totalBytes, std::max(1u, gpu_.getSwapchainImageCount()));
  if (ensure.hasError()) {
    return Result<bool, std::string>::makeError(ensure.error());
  }
  glyphDependencyBuffer_ = ensure.value().buffer;
  if (transformBytes != 0u) {
    auto up = gpu_.updateBuffer(
        glyphDependencyBuffer_,
        std::span<const std::byte>(reinterpret_cast<const std::byte *>(
                                       resolvedWorldTransforms_.data()),
                                   transformBytes),
        0u);
    if (up.hasError()) {
      glyphBuffers_.abandonPrepared();
      return Result<bool, std::string>::makeError(up.error());
    }
  }
  auto up = gpu_.updateBuffer(
      glyphDependencyBuffer_,
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(glyphInstances_.data()),
          instanceBytes),
      instancesOffset);
  if (up.hasError()) {
    glyphBuffers_.abandonPrepared();
    return Result<bool, std::string>::makeError(up.error());
  }
  worldTransformBufferAddress_ =
      transformBytes == 0u
          ? 0u
          : gpu_.getBufferDeviceAddress(glyphDependencyBuffer_, 0u);
  glyphBufferAddress_ =
      gpu_.getBufferDeviceAddress(glyphDependencyBuffer_, instancesOffset);
  return Result<bool, std::string>::makeResult(true);
}

void TextRenderer::buildGlyphPackets(const CameraFrameState &camera) {
  NURI_PROFILER_ZONE("TextRenderer::buildGlyphPackets",
                     NURI_PROFILER_COLOR_CMD_DRAW);
  if (glyphQueueNeedsSort_) {
    std::sort(glyphQueue_.begin(), glyphQueue_.end(), glyphBatchLess);
    glyphQueueNeedsSort_ = false;
  }
  glyphInstances_.clear();
  glyphBatches_.clear();
  glyphInstances_.reserve(glyphQueue_.size());
  glyphBatches_.reserve(16u);
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
  TextDomain currentDomain = static_cast<TextDomain>(0xffu);
  for (const GlyphPacket &q : glyphQueue_) {
    if (currentDomain != q.domain || currentAtlas != q.atlas ||
        currentTransformId != q.transformId ||
        std::abs(currentPxRange - q.pxRange) > kBatchPxRangeEpsilon) {
      currentDomain = q.domain;
      currentAtlas = q.atlas;
      currentTransformId = q.transformId;
      currentPxRange = q.pxRange;
      float sortDepth = 0.0f;
      if (q.domain == TextDomain::World) {
        const size_t transformIdx = static_cast<size_t>(q.transformId - 1u);
        const glm::vec3 translation =
            glm::vec3(resolvedWorldTransforms_[transformIdx].translation);
        sortDepth = -(view * glm::vec4(translation, 1.0f)).z;
      }
      glyphBatches_.push_back(GlyphBatch{
          .domain = q.domain,
          .atlas = q.atlas,
          .pxRange = q.pxRange,
          .firstInstance = static_cast<uint32_t>(glyphInstances_.size()),
          .instanceCount = 0,
          .atlasTexture = q.atlasTexture,
          .sortDepth = sortDepth,
      });
    }
    const uint32_t transformIndex =
        q.domain == TextDomain::World ? q.transformId - 1u : 0u;
    glyphInstances_.push_back(GlyphInstance{
        .rectMinMax = glm::vec4(q.minX, q.minY, q.maxX, q.maxY),
        .uvMinMax = glm::vec4(q.uvMinX, q.uvMinY, q.uvMaxX, q.uvMaxY),
        .color = q.color,
        .transformIndex = transformIndex,
    });
    glyphBatches_.back().instanceCount += 1u;
  }
  NURI_PROFILER_ZONE_END();
}

Result<bool, std::string>
TextRenderer::prepareGlyphPackets(const CameraFrameState &camera) {
  if (glyphPrepared_) {
    return Result<bool, std::string>::makeResult(true);
  }
  const uint64_t cameraHash =
      worldHasBillboards_ ? hashCameraFrameState(camera) : 0ull;
  const bool canReuse = glyphGeometryValid_ &&
                        glyphQueueHash_ == lastBuiltGlyphQueueHash_ &&
                        cameraHash == lastBuiltWorldCameraHash_;
  if (!canReuse) {
    buildGlyphPackets(camera);
    glyphGeometryValid_ = true;
    lastBuiltGlyphQueueHash_ = glyphQueueHash_;
    lastBuiltWorldCameraHash_ = cameraHash;
  }
  auto upload = uploadGlyphPackets();
  if (upload.hasError()) {
    return upload;
  }
  glyphPrepared_ = true;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
TextRenderer::prepareWorldRenderState(RenderFrameContext &frame,
                                      TextureHandle &outDepth,
                                      Format &outDepthFormat) {
  if (worldGlyphCount_ == 0u || worldAppended_) {
    return Result<bool, std::string>::makeResult(false);
  }
  auto packets = prepareGlyphPackets(frame.camera);
  if (packets.hasError()) {
    return packets;
  }
  outDepth = resolveFrameDepthTexture(frame);
  outDepthFormat = ::nuri::isValid(outDepth) ? gpu_.getTextureFormat(outDepth)
                                             : Format::Count;
  const TextureHandle colorTexture =
      frame.sharedResources[FrameTextureSlot::FrameColor].texture;
  const Format colorFormat = ::nuri::isValid(colorTexture)
                                 ? gpu_.getTextureFormat(colorTexture)
                                 : gpu_.getSwapchainFormat();
  auto pipeline = ensureWorldPipeline(colorFormat, outDepthFormat);
  if (pipeline.hasError()) {
    return pipeline;
  }
  worldDraws_.clear();
  worldSortDepths_.clear();
  worldPcs_.clear();
  worldTransparentTextureReadList_.clear();
  worldDraws_.reserve(glyphBatches_.size());
  worldPcs_.reserve(glyphBatches_.size());
  const glm::mat4 viewProj =
      cameraCurrentUnjitteredViewProjection(frame.camera);
  for (const GlyphBatch &batch : glyphBatches_) {
    if (batch.domain != TextDomain::World) {
      continue;
    }
    worldPcs_.push_back(WorldPC{
        .viewProj = viewProj,
        .glyphBufferAddress = glyphBufferAddress_,
        .transformBufferAddress = worldTransformBufferAddress_,
        .atlas = batch.atlas,
        .pxRange = batch.pxRange,
        .alphaDiscardThreshold = 1.0e-3f,
    });
    DrawItem &draw = worldDraws_.emplace_back();
    worldSortDepths_.push_back(batch.sortDepth);
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
  colorTexture = frame.sharedResources[FrameTextureSlot::FrameColor].texture;
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
        ::nuri::isValid(
            frame.sharedResources[FrameTextureSlot::FrameColor].graph)
            ? frame.sharedResources[FrameTextureSlot::FrameColor].graph
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
      std::span<const BufferHandle>(&glyphDependencyBuffer_, 1u);
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
  worldTransparentDraws_.reserve(worldDraws_.size());
  for (size_t i = 0; i < worldDraws_.size(); ++i) {
    worldTransparentDraws_.push_back(TransparentStageSortableDraw{
        .draw = worldDraws_[i],
        .sortDepth = worldSortDepths_[i],
        .stableOrder = static_cast<uint32_t>(i),
    });
  }
  out.sortableDraws = std::span<const TransparentStageSortableDraw>(
      worldTransparentDraws_.data(), worldTransparentDraws_.size());
  out.fixedDraws = {};
  out.dependencyBuffers =
      std::span<const BufferHandle>(&glyphDependencyBuffer_, 1u);
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
  if (uiGlyphCount_ == 0u || uiAppended_) {
    return Result<bool, std::string>::makeResult(true);
  }
  auto packets = prepareGlyphPackets(frame.camera);
  if (packets.hasError()) {
    return packets;
  }
  const TextureHandle colorTexture =
      frame.sharedResources[FrameTextureSlot::FrameColor].texture;
  const Format colorFormat = ::nuri::isValid(colorTexture)
                                 ? gpu_.getTextureFormat(colorTexture)
                                 : gpu_.getSwapchainFormat();
  auto pipeline = ensureUiPipeline(colorFormat);
  if (pipeline.hasError()) {
    return pipeline;
  }
  uiDraws_.clear();
  uiPcs_.clear();
  uiDraws_.reserve(glyphBatches_.size());
  uiPcs_.reserve(glyphBatches_.size());
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
  for (const GlyphBatch &b : glyphBatches_) {
    if (b.domain != TextDomain::Ui) {
      continue;
    }
    uiPcs_.push_back(UiPC{.proj = proj,
                          .glyphBufferAddress = glyphBufferAddress_,
                          .atlas = b.atlas,
                          .pxRange = b.pxRange});
    const UiPC &pc = uiPcs_.back();
    DrawItem d{};
    d.pipeline = uiPipeline_;
    d.vertexCount = 6u;
    d.instanceCount = b.instanceCount;
    d.firstInstance = b.firstInstance;
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
        ::nuri::isValid(
            frame.sharedResources[FrameTextureSlot::FrameColor].graph)
            ? frame.sharedResources[FrameTextureSlot::FrameColor].graph
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
  desc.dependencyBuffers =
      std::span<const BufferHandle>(&glyphDependencyBuffer_, 1u);
  desc.debugLabel = "Text2D Pass";
  desc.debugColor = 0xffcc8844u;
  const RenderGraphPassId pass = graph.addGraphicsPass(desc).value();
  for (const GlyphBatch &batch : glyphBatches_) {
    if (batch.domain == TextDomain::Ui) {
      (void)graph
          .addTextureRead(
              pass,
              graph.importTexture(batch.atlasTexture, "text2d_atlas_texture")
                  .value())
          .value();
    }
  }
  uiAppended_ = true;
  return Result<bool, std::string>::makeResult(true);
}

void TextRenderer::onFrameSubmitted(SubmissionHandle submission) noexcept {
  glyphBuffers_.submitPrepared(submission);
}

void TextRenderer::onFrameAbandoned() noexcept {
  glyphBuffers_.abandonPrepared();
}

void TextRenderer::destroyGpu() {
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
