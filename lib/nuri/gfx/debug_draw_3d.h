#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/gpu_render_types.h"
#include "nuri/gfx/gpu_types.h"
#include "nuri/math/types.h"

#include <glm/glm.hpp>

#include <cstdint>
#include <memory_resource>
#include <string>

namespace nuri {

class NURI_API DebugDraw3D {
public:
  explicit DebugDraw3D(
      GPUDevice &gpu,
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
  [[nodiscard]] Result<RenderPass, std::string>
  buildRenderPass(TextureHandle depthTexture);

private:
  struct LineData {
    glm::vec4 pos;
    glm::vec4 color;
  };

  struct FrameBufferState {
    BufferHandle buffer{};
    size_t capacityBytes = 0;
  };

  struct PushConstants {
    glm::mat4 mvp{1.0f};
    uint64_t vertexBufferAddress = 0;
  };

  [[nodiscard]] Result<bool, std::string> ensureShaderModules();
  [[nodiscard]] Result<bool, std::string> ensurePipeline(Format colorFormat,
                                                         Format depthFormat);
  void syncFrameBufferCount(uint32_t swapchainImageCount);
  [[nodiscard]] Result<bool, std::string>
  ensureLineBufferCapacity(uint32_t frameIndex, size_t requiredSize);

  GPUDevice &gpu_;
  glm::mat4 mvp_ = glm::mat4(1.0f);
  std::pmr::memory_resource *memory_resource_ = std::pmr::get_default_resource();
  std::pmr::vector<LineData> lines_;
  std::pmr::vector<FrameBufferState> frameBuffers_;
  ShaderHandle vert_{};
  ShaderHandle frag_{};
  RenderPipelineHandle pipeline_{};
  Format pipelineColorFormat_ = Format::Count;
  Format pipelineDepthFormat_ = Format::Count;

  PushConstants pushConstants_{};
  DrawItem drawItem_{};
};

} // namespace nuri
