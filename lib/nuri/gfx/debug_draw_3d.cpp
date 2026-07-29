#include "nuri/gfx/debug_draw_3d.h"
#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/gpu_descriptors.h"
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
std::pmr::memory_resource *
resolveMemoryResource(std::pmr::memory_resource *memoryResource) {
  return memoryResource != nullptr ? memoryResource
                                   : std::pmr::get_default_resource();
}
} // namespace

DebugDraw3D::DebugDraw3D(GPUDevice &gpu,
                         std::pmr::memory_resource *memoryResource)
    : gpu_(gpu), lines_(resolveMemoryResource(memoryResource)),
      lineBuffers_(
          gpu,
          BufferDesc{.usage = BufferUsage::Storage | BufferUsage::Vertex,
                     .storage = Storage::HostVisible},
          "DebugDraw3D Buffer", resolveMemoryResource(memoryResource)) {}

DebugDraw3D::~DebugDraw3D() = default;

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
  constexpr std::array edges{
      std::pair{0u, 1u}, std::pair{2u, 3u}, std::pair{4u, 5u},
      std::pair{6u, 7u}, std::pair{0u, 2u}, std::pair{1u, 3u},
      std::pair{4u, 6u}, std::pair{5u, 7u}, std::pair{0u, 4u},
      std::pair{1u, 5u}, std::pair{2u, 6u}, std::pair{3u, 7u}};
  for (auto [from, to] : edges)
    line(pts[from], pts[to], c);
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
  constexpr std::array edges{
      std::pair{0u, 4u}, std::pair{1u, 5u}, std::pair{2u, 6u},
      std::pair{3u, 7u}, std::pair{0u, 1u}, std::pair{1u, 2u},
      std::pair{2u, 3u}, std::pair{3u, 0u}, std::pair{0u, 2u},
      std::pair{1u, 3u}, std::pair{4u, 5u}, std::pair{5u, 6u},
      std::pair{6u, 7u}, std::pair{7u, 4u}, std::pair{4u, 6u},
      std::pair{5u, 7u}};
  for (auto [from, to] : edges)
    line(pp[from], pp[to], color);
  const glm::vec4 gridColor = color * 0.7f;
  const int gridLines = 100;
  constexpr std::array gridEdges{std::pair{0u, 1u}, std::pair{2u, 3u},
                                 std::pair{0u, 3u}, std::pair{1u, 2u}};
  for (auto [from, to] : gridEdges) {
    glm::vec3 p1 = pp[from];
    glm::vec3 p2 = pp[to];
    const glm::vec3 s1 = (pp[from + 4u] - p1) / float(gridLines);
    const glm::vec3 s2 = (pp[to + 4u] - p2) / float(gridLines);
    for (int i = 0; i != gridLines; i++, p1 += s1, p2 += s2)
      line(p1, p2, gridColor);
  }
}

Result<bool, std::string> DebugDraw3D::ensureShaderModules() {
  if (shaders_[0]) {
    return Result<bool, std::string>::makeResult(true);
  }
  constexpr std::array descs{ShaderDesc{.moduleName = "debug_draw_3d_vs",
                                        .source = kDebugDraw3DVS,
                                        .stage = ShaderStage::Vertex},
                             ShaderDesc{.moduleName = "debug_draw_3d_fs",
                                        .source = kDebugDraw3DFS,
                                        .stage = ShaderStage::Fragment}};
  std::array<OwnedShaderHandle, 2u> createdShaders{};
  for (size_t index = 0; index < createdShaders.size(); ++index) {
    auto created = gpu_.createShaderModule(descs[index]);
    if (created.hasError()) {
      return Result<bool, std::string>::makeError(created.error());
    }
    createdShaders[index].reset(gpu_, created.value());
  }
  shaders_ = std::move(createdShaders);
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> DebugDraw3D::ensurePipeline(Format colorFormat,
                                                      Format depthFormat) {
  if (pipeline_ && pipelineColorFormat_ == colorFormat &&
      pipelineDepthFormat_ == depthFormat) {
    return Result<bool, std::string>::makeResult(true);
  }
  auto shaderResult = ensureShaderModules();
  if (shaderResult.hasError()) {
    return shaderResult;
  }
  pipeline_.reset();
  RenderPipelineDesc pipelineDesc{
      .vertexInput = {},
      .vertexShader = shaders_[0].get(),
      .fragmentShader = shaders_[1].get(),
      .colorFormats = {colorFormat},
      .depthFormat = depthFormat,
      .cullMode = CullMode::None,
      .polygonMode = PolygonMode::Fill,
      .topology = Topology::Line,
      .blendEnabled = true,
      .rasterState = depthFormat != Format::Count
                         ? makeRasterPipelineState(
                               DepthState{.compareOp = CompareOp::LessEqual,
                                          .isDepthWriteEnabled = false})
                         : RasterPipelineState{},
  };
  auto pipelineResult =
      gpu_.createRenderPipeline(pipelineDesc, "DebugDraw3D Pipeline");
  if (pipelineResult.hasError()) {
    return Result<bool, std::string>::makeError(pipelineResult.error());
  }
  pipeline_.reset(gpu_, pipelineResult.value());
  pipelineColorFormat_ = colorFormat;
  pipelineDepthFormat_ = depthFormat;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string>
DebugDraw3D::prepareDraw(uint64_t frameIndexValue, TextureHandle depthTexture,
                         Format colorFormat, bool enableDepthTest,
                         DrawItem &outDraw, BufferHandle &outDependency) {
  NURI_PROFILER_FUNCTION();
  outDraw = {};
  outDependency = {};
  if (lines_.empty()) {
    return Result<bool, std::string>::makeResult(false);
  }
  if (lines_.size() > std::numeric_limits<uint32_t>::max()) {
    return Result<bool, std::string>::makeError(
        "DebugDraw3D: vertex count exceeds uint32_t range");
  }
  const size_t swapchainImageCount =
      std::max<size_t>(gpu_.getSwapchainImageCount(), 1u);
  const size_t requiredBytes = lines_.size() * sizeof(LineData);
  const std::span<const std::byte> lineBytes{
      reinterpret_cast<const std::byte *>(lines_.data()), requiredBytes};
  auto lineBufferResult =
      lineBuffers_.upload(frameIndexValue, swapchainImageCount, lineBytes);
  if (lineBufferResult.hasError()) {
    return Result<bool, std::string>::makeError(lineBufferResult.error());
  }
  const Format depthFormat = enableDepthTest && nuri::isValid(depthTexture)
                                 ? gpu_.getTextureFormat(depthTexture)
                                 : Format::Count;
  auto pipelineResult = ensurePipeline(colorFormat, depthFormat);
  if (pipelineResult.hasError()) {
    return Result<bool, std::string>::makeError(pipelineResult.error());
  }
  const BufferHandle lineBufferHandle = lineBufferResult.value().buffer;
  const uint64_t address = gpu_.getBufferDeviceAddress(lineBufferHandle);
  pushConstants_.mvp = mvp_;
  pushConstants_.vertexBufferAddress = address;
  outDraw.pipeline = pipeline_.get();
  outDraw.vertexCount = static_cast<uint32_t>(lines_.size());
  outDraw.instanceCount = 1;
  outDraw.pushConstants = std::span<const std::byte>(
      reinterpret_cast<const std::byte *>(&pushConstants_),
      sizeof(pushConstants_));
  outDraw.debugLabel = "DebugDraw3D Draw";
  outDraw.debugColor = 0xffffcc00u;
  outDependency = lineBufferHandle;
  if (enableDepthTest && nuri::isValid(depthTexture)) {
    outDraw.useDepthState = true;
    outDraw.depthState = {
        .compareOp = CompareOp::LessEqual,
        .isDepthWriteEnabled = false,
    };
  }
  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri
