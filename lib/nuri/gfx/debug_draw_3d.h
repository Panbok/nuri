#pragma once
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/dynamic_buffer.h"
#include "nuri/gfx/gpu_render_types.h"
#include "nuri/gfx/gpu_types.h"
#include "nuri/gfx/owned_gpu_resource.h"
#include "nuri/math/types.h"
#include <cstdint>
#include <glm/glm.hpp>
#include <memory_resource>
#include <string>
namespace nuri {

class NURI_API DebugDraw3D {
public:
  explicit DebugDraw3D(GPUDevice &gpu,
                       std::pmr::memory_resource *memoryResource =
                           std::pmr::get_default_resource());
  ~DebugDraw3D();
  DebugDraw3D(const DebugDraw3D &) = delete;
  DebugDraw3D &operator=(const DebugDraw3D &) = delete;
  DebugDraw3D(DebugDraw3D &&) = delete;
  DebugDraw3D &operator=(DebugDraw3D &&) = delete;
  void clear() { lines_.clear(); }
  void line(const glm::vec3 &p1, const glm::vec3 &p2, const glm::vec4 &c);
  void plane(const glm::vec3 &orig, const glm::vec3 &v1, const glm::vec3 &v2,
             int n1, int n2, float s1, float s2, const glm::vec4 &color,
             const glm::vec4 &outlineColor);
  void box(const glm::mat4 &m, const BoundingBox &box, const glm::vec4 &color);
  void box(const glm::mat4 &m, const glm::vec3 &size, const glm::vec4 &color);
  void frustum(const glm::mat4 &camView, const glm::mat4 &camProj,
               const glm::vec4 &color);
  void setMatrix(const glm::mat4 &mvp) { mvp_ = mvp; }
  [[nodiscard]] Result<bool, std::string>
  prepareDraw(uint64_t frameIndex, TextureHandle depthTexture,
              Format colorFormat, bool enableDepthTest, DrawItem &outDraw,
              BufferHandle &outDependency);
  void onFrameSubmitted(SubmissionHandle submission) noexcept {
    lineBuffers_.submitPrepared(submission);
  }
  void onFrameAbandoned() noexcept { lineBuffers_.abandonPrepared(); }

private:
  struct LineData {
    glm::vec4 pos;
    glm::vec4 color;
  };
  struct PushConstants {
    glm::mat4 mvp{1.0f};
    uint64_t vertexBufferAddress = 0;
  };
  [[nodiscard]] Result<bool, std::string> ensureShaderModules();
  [[nodiscard]] Result<bool, std::string> ensurePipeline(Format colorFormat,
                                                         Format depthFormat);
  GPUDevice &gpu_;
  glm::mat4 mvp_ = glm::mat4(1.0f);
  std::pmr::vector<LineData> lines_;
  DynamicBufferRing lineBuffers_;
  std::array<OwnedShaderHandle, 2u> shaders_{};
  OwnedRenderPipelineHandle pipeline_{};
  Format pipelineColorFormat_ = Format::Count;
  Format pipelineDepthFormat_ = Format::Count;
  PushConstants pushConstants_{};
};

} // namespace nuri
