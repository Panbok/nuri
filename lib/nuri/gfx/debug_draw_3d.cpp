#include "nuri/gfx/debug_draw_3d.h"

#include "nuri/core/profiling.h"
#include "nuri/gfx/gpu_descriptors.h"

#include <algorithm>
#include <limits>

namespace nuri {
namespace {

constexpr std::string_view kDebugDraw3DVS = R"(
#version 460
#extension GL_EXT_buffer_reference : require

layout(location = 0) out vec4 outColor;

struct Vertex {
  vec4 pos;
  vec4 rgba;
};

layout(std430, buffer_reference) readonly buffer VertexBuffer {
  Vertex vertices[];
};

layout(push_constant) uniform PushConstants {
  mat4 mvp;
  VertexBuffer vb;
} pc;

void main() {
  outColor = pc.vb.vertices[gl_VertexIndex].rgba;
  gl_Position = pc.mvp * pc.vb.vertices[gl_VertexIndex].pos;
}
)";

constexpr std::string_view kDebugDraw3DFS = R"(
#version 460

layout(location = 0) in vec4 inColor;
layout(location = 0) out vec4 outColor;

void main() {
  outColor = inColor;
}
)";

} // namespace

DebugDraw3D::DebugDraw3D(GPUDevice &gpu,
                         std::pmr::memory_resource *memoryResource)
    : gpu_(gpu),
      memory_resource_(memoryResource != nullptr ? memoryResource
                                                 : std::pmr::get_default_resource()),
      lines_(memory_resource_), frameBuffers_(memory_resource_) {}

DebugDraw3D::~DebugDraw3D() {
  for (const FrameBufferState &frame : frameBuffers_) {
    if (nuri::isValid(frame.buffer)) {
      gpu_.destroyBuffer(frame.buffer);
    }
  }

  if (nuri::isValid(pipeline_)) {
    gpu_.destroyRenderPipeline(pipeline_);
  }
  if (nuri::isValid(vert_)) {
    gpu_.destroyShaderModule(vert_);
  }
  if (nuri::isValid(frag_)) {
    gpu_.destroyShaderModule(frag_);
  }
}

void DebugDraw3D::line(const glm::vec3 &p1, const glm::vec3 &p2,
                       const glm::vec4 &c) {
  lines_.push_back({.pos = glm::vec4(p1, 1.0f), .color = c});
  lines_.push_back({.pos = glm::vec4(p2, 1.0f), .color = c});
}

void DebugDraw3D::plane(const glm::vec3 &o, const glm::vec3 &v1,
                        const glm::vec3 &v2, int n1, int n2, float s1, float s2,
                        const glm::vec4 &color, const glm::vec4 &outlineColor) {
  line(o - s1 / 2.0f * v1 - s2 / 2.0f * v2, o - s1 / 2.0f * v1 + s2 / 2.0f * v2,
       outlineColor);
  line(o + s1 / 2.0f * v1 - s2 / 2.0f * v2, o + s1 / 2.0f * v1 + s2 / 2.0f * v2,
       outlineColor);

  line(o - s1 / 2.0f * v1 + s2 / 2.0f * v2, o + s1 / 2.0f * v1 + s2 / 2.0f * v2,
       outlineColor);
  line(o - s1 / 2.0f * v1 - s2 / 2.0f * v2, o + s1 / 2.0f * v1 - s2 / 2.0f * v2,
       outlineColor);

  for (int i = 1; i < n1; i++) {
    float t = ((float)i - (float)n1 / 2.0f) * s1 / (float)n1;
    const glm::vec3 o1 = o + t * v1;
    line(o1 - s2 / 2.0f * v2, o1 + s2 / 2.0f * v2, color);
  }

  for (int i = 1; i < n2; i++) {
    const float t = ((float)i - (float)n2 / 2.0f) * s2 / (float)n2;
    const glm::vec3 o2 = o + t * v2;
    line(o2 - s1 / 2.0f * v1, o2 + s1 / 2.0f * v1, color);
  }
}

void DebugDraw3D::box(const glm::mat4 &m, const glm::vec3 &size,
                      const glm::vec4 &c) {
  glm::vec3 pts[8] = {
      glm::vec3(+size.x, +size.y, +size.z),
      glm::vec3(+size.x, +size.y, -size.z),
      glm::vec3(+size.x, -size.y, +size.z),
      glm::vec3(+size.x, -size.y, -size.z),
      glm::vec3(-size.x, +size.y, +size.z),
      glm::vec3(-size.x, +size.y, -size.z),
      glm::vec3(-size.x, -size.y, +size.z),
      glm::vec3(-size.x, -size.y, -size.z),
  };

  for (auto &p : pts)
    p = glm::vec3(m * glm::vec4(p, 1.f));

  line(pts[0], pts[1], c);
  line(pts[2], pts[3], c);
  line(pts[4], pts[5], c);
  line(pts[6], pts[7], c);

  line(pts[0], pts[2], c);
  line(pts[1], pts[3], c);
  line(pts[4], pts[6], c);
  line(pts[5], pts[7], c);

  line(pts[0], pts[4], c);
  line(pts[1], pts[5], c);
  line(pts[2], pts[6], c);
  line(pts[3], pts[7], c);
}

void DebugDraw3D::box(const glm::mat4 &m, const BoundingBox &box,
                      const glm::vec4 &color) {
  this->box(m * glm::translate(glm::mat4(1.f), .5f * (box.min_ + box.max_)),
            0.5f * glm::vec3(box.max_ - box.min_), color);
}

void DebugDraw3D::frustum(const glm::mat4 &camView, const glm::mat4 &camProj,
                          const glm::vec4 &color) {
  const glm::vec3 corners[] = {glm::vec3(-1, -1, -1), glm::vec3(+1, -1, -1),
                               glm::vec3(+1, +1, -1), glm::vec3(-1, +1, -1),
                               glm::vec3(-1, -1, +1), glm::vec3(+1, -1, +1),
                               glm::vec3(+1, +1, +1), glm::vec3(-1, +1, +1)};

  glm::vec3 pp[8];

  for (int i = 0; i < 8; i++) {
    glm::vec4 q = glm::inverse(camView) * glm::inverse(camProj) *
                  glm::vec4(corners[i], 1.0f);
    pp[i] = glm::vec3(q.x / q.w, q.y / q.w, q.z / q.w);
  }
  line(pp[0], pp[4], color);
  line(pp[1], pp[5], color);
  line(pp[2], pp[6], color);
  line(pp[3], pp[7], color);
  // near
  line(pp[0], pp[1], color);
  line(pp[1], pp[2], color);
  line(pp[2], pp[3], color);
  line(pp[3], pp[0], color);
  // x
  line(pp[0], pp[2], color);
  line(pp[1], pp[3], color);
  // far
  line(pp[4], pp[5], color);
  line(pp[5], pp[6], color);
  line(pp[6], pp[7], color);
  line(pp[7], pp[4], color);
  // x
  line(pp[4], pp[6], color);
  line(pp[5], pp[7], color);

  const glm::vec4 gridColor = color * 0.7f;
  const int gridLines = 100;

  // bottom
  {
    glm::vec3 p1 = pp[0];
    glm::vec3 p2 = pp[1];
    const glm::vec3 s1 = (pp[4] - pp[0]) / float(gridLines);
    const glm::vec3 s2 = (pp[5] - pp[1]) / float(gridLines);
    for (int i = 0; i != gridLines; i++, p1 += s1, p2 += s2)
      line(p1, p2, gridColor);
  }
  // top
  {
    glm::vec3 p1 = pp[2];
    glm::vec3 p2 = pp[3];
    const glm::vec3 s1 = (pp[6] - pp[2]) / float(gridLines);
    const glm::vec3 s2 = (pp[7] - pp[3]) / float(gridLines);
    for (int i = 0; i != gridLines; i++, p1 += s1, p2 += s2)
      line(p1, p2, gridColor);
  }
  // left
  {
    glm::vec3 p1 = pp[0];
    glm::vec3 p2 = pp[3];
    const glm::vec3 s1 = (pp[4] - pp[0]) / float(gridLines);
    const glm::vec3 s2 = (pp[7] - pp[3]) / float(gridLines);
    for (int i = 0; i != gridLines; i++, p1 += s1, p2 += s2)
      line(p1, p2, gridColor);
  }
  // right
  {
    glm::vec3 p1 = pp[1];
    glm::vec3 p2 = pp[2];
    const glm::vec3 s1 = (pp[5] - pp[1]) / float(gridLines);
    const glm::vec3 s2 = (pp[6] - pp[2]) / float(gridLines);
    for (int i = 0; i != gridLines; i++, p1 += s1, p2 += s2)
      line(p1, p2, gridColor);
  }
}

Result<bool, std::string> DebugDraw3D::ensureShaderModules() {
  if (nuri::isValid(vert_) && nuri::isValid(frag_)) {
    return Result<bool, std::string>::makeResult(true);
  }

  if (nuri::isValid(vert_)) {
    gpu_.destroyShaderModule(vert_);
    vert_ = ShaderHandle{};
  }
  if (nuri::isValid(frag_)) {
    gpu_.destroyShaderModule(frag_);
    frag_ = ShaderHandle{};
  }

  auto vertResult = gpu_.createShaderModule(ShaderDesc{
      .moduleName = "debug_draw_3d_vs",
      .source = kDebugDraw3DVS,
      .stage = ShaderStage::Vertex,
  });
  if (vertResult.hasError()) {
    return Result<bool, std::string>::makeError(vertResult.error());
  }

  auto fragResult = gpu_.createShaderModule(ShaderDesc{
      .moduleName = "debug_draw_3d_fs",
      .source = kDebugDraw3DFS,
      .stage = ShaderStage::Fragment,
  });
  if (fragResult.hasError()) {
    gpu_.destroyShaderModule(vertResult.value());
    return Result<bool, std::string>::makeError(fragResult.error());
  }

  vert_ = vertResult.value();
  frag_ = fragResult.value();
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> DebugDraw3D::ensurePipeline(Format colorFormat,
                                                      Format depthFormat) {
  if (nuri::isValid(pipeline_) && pipelineColorFormat_ == colorFormat &&
      pipelineDepthFormat_ == depthFormat) {
    return Result<bool, std::string>::makeResult(true);
  }

  auto shaderResult = ensureShaderModules();
  if (shaderResult.hasError()) {
    return shaderResult;
  }

  if (nuri::isValid(pipeline_)) {
    gpu_.destroyRenderPipeline(pipeline_);
    pipeline_ = RenderPipelineHandle{};
  }

  RenderPipelineDesc pipelineDesc{
      .vertexInput = {},
      .vertexShader = vert_,
      .fragmentShader = frag_,
      .colorFormats = {colorFormat},
      .depthFormat = depthFormat,
      .cullMode = CullMode::None,
      .polygonMode = PolygonMode::Fill,
      .topology = Topology::Line,
      .blendEnabled = true,
  };

  auto pipelineResult =
      gpu_.createRenderPipeline(pipelineDesc, "DebugDraw3D Pipeline");
  if (pipelineResult.hasError()) {
    return Result<bool, std::string>::makeError(pipelineResult.error());
  }

  pipeline_ = pipelineResult.value();
  pipelineColorFormat_ = colorFormat;
  pipelineDepthFormat_ = depthFormat;
  return Result<bool, std::string>::makeResult(true);
}

void DebugDraw3D::syncFrameBufferCount(uint32_t swapchainImageCount) {
  const uint32_t imageCount = std::max(1u, swapchainImageCount);
  if (frameBuffers_.size() == imageCount) {
    return;
  }

  for (const FrameBufferState &frame : frameBuffers_) {
    if (nuri::isValid(frame.buffer)) {
      gpu_.destroyBuffer(frame.buffer);
    }
  }

  frameBuffers_.assign(imageCount, FrameBufferState{});
}

Result<bool, std::string>
DebugDraw3D::ensureLineBufferCapacity(uint32_t frameIndex,
                                      size_t requiredSize) {
  if (frameIndex >= frameBuffers_.size()) {
    return Result<bool, std::string>::makeError(
        "DebugDraw3D: frame index out of range");
  }

  FrameBufferState &frame = frameBuffers_[frameIndex];
  if (nuri::isValid(frame.buffer) && frame.capacityBytes >= requiredSize) {
    return Result<bool, std::string>::makeResult(true);
  }

  if (nuri::isValid(frame.buffer)) {
    gpu_.destroyBuffer(frame.buffer);
    frame.buffer = BufferHandle{};
  }

  const size_t newSize =
      std::max({requiredSize, frame.capacityBytes * 2, size_t{1}});
  auto bufferResult = gpu_.createBuffer(
      BufferDesc{
          .usage = BufferUsage::Storage,
          .storage = Storage::HostVisible,
          .size = newSize,
      },
      "DebugDraw3D Buffer");
  if (bufferResult.hasError()) {
    return Result<bool, std::string>::makeError(bufferResult.error());
  }

  frame.buffer = bufferResult.value();
  frame.capacityBytes = newSize;
  return Result<bool, std::string>::makeResult(true);
}

Result<RenderPass, std::string>
DebugDraw3D::buildRenderPass(TextureHandle depthTexture) {
  NURI_PROFILER_FUNCTION();

  RenderPass pass{};
  pass.color.loadOp = LoadOp::Load;
  pass.color.storeOp = StoreOp::Store;
  pass.debugLabel = "DebugDraw3D Pass";
  pass.debugColor = 0xffffcc00u;

  if (nuri::isValid(depthTexture)) {
    pass.depthTexture = depthTexture;
    pass.depth.loadOp = LoadOp::Load;
    pass.depth.storeOp = StoreOp::Store;
    pass.depth.clearDepth = 1.0f;
    pass.depth.clearStencil = 0;
  }

  if (lines_.empty()) {
    pass.draws = {};
    lines_.clear();
    return Result<RenderPass, std::string>::makeResult(pass);
  }

  if (lines_.size() > std::numeric_limits<uint32_t>::max()) {
    return Result<RenderPass, std::string>::makeError(
        "DebugDraw3D: vertex count exceeds uint32_t range");
  }

  syncFrameBufferCount(gpu_.getSwapchainImageCount());

  const uint32_t imageCount = static_cast<uint32_t>(frameBuffers_.size());
  const uint32_t frameIndex = gpu_.getSwapchainImageIndex() % imageCount;
  const size_t requiredBytes = lines_.size() * sizeof(LineData);

  auto lineBufferResult = ensureLineBufferCapacity(frameIndex, requiredBytes);
  if (lineBufferResult.hasError()) {
    return Result<RenderPass, std::string>::makeError(lineBufferResult.error());
  }

  const std::span<const std::byte> lineBytes{
      reinterpret_cast<const std::byte *>(lines_.data()), requiredBytes};
  auto updateResult =
      gpu_.updateBuffer(frameBuffers_[frameIndex].buffer, lineBytes, 0);
  if (updateResult.hasError()) {
    return Result<RenderPass, std::string>::makeError(updateResult.error());
  }

  const Format depthFormat = nuri::isValid(depthTexture)
                                 ? gpu_.getTextureFormat(depthTexture)
                                 : Format::Count;
  auto pipelineResult = ensurePipeline(gpu_.getSwapchainFormat(), depthFormat);
  if (pipelineResult.hasError()) {
    return Result<RenderPass, std::string>::makeError(pipelineResult.error());
  }

  const uint64_t address =
      gpu_.getBufferDeviceAddress(frameBuffers_[frameIndex].buffer);
  if (address == 0) {
    return Result<RenderPass, std::string>::makeError(
        "DebugDraw3D: invalid line buffer GPU address");
  }

  pushConstants_.mvp = mvp_;
  pushConstants_.vertexBufferAddress = address;

  drawItem_ = DrawItem{};
  drawItem_.pipeline = pipeline_;
  drawItem_.vertexCount = static_cast<uint32_t>(lines_.size());
  drawItem_.instanceCount = 1;
  drawItem_.pushConstants = std::span<const std::byte>(
      reinterpret_cast<const std::byte *>(&pushConstants_),
      sizeof(pushConstants_));
  drawItem_.debugLabel = "DebugDraw3D Draw";
  drawItem_.debugColor = 0xffffcc00u;
  if (nuri::isValid(depthTexture)) {
    drawItem_.useDepthState = true;
    drawItem_.depthState = {
        .compareOp = CompareOp::LessEqual,
        .isDepthWriteEnabled = false,
    };
  }

  pass.draws = std::span<const DrawItem>(&drawItem_, 1);
  return Result<RenderPass, std::string>::makeResult(pass);
}

} // namespace nuri
