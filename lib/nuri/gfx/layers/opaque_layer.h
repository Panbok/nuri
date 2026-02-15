#pragma once

#include "nuri/core/layer.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline.h"
#include "nuri/gfx/shader.h"
#include "nuri/resources/cpu/mesh_data.h"
#include "nuri/resources/gpu/buffer.h"
#include "nuri/scene/render_scene.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <memory_resource>

#include <glm/glm.hpp>

namespace nuri {

class NURI_API OpaqueLayer final : public Layer {
public:
  explicit OpaqueLayer(GPUDevice &gpu);
  ~OpaqueLayer() override;

  OpaqueLayer(const OpaqueLayer &) = delete;
  OpaqueLayer &operator=(const OpaqueLayer &) = delete;
  OpaqueLayer(OpaqueLayer &&) = delete;
  OpaqueLayer &operator=(OpaqueLayer &&) = delete;

  static std::unique_ptr<OpaqueLayer> create(GPUDevice &gpu) {
    return std::make_unique<OpaqueLayer>(gpu);
  }

  void onAttach() override;
  void onDetach() override;
  void onResize(int32_t width, int32_t height) override;
  Result<bool, std::string> buildRenderPasses(RenderFrameContext &frame,
                                              RenderPassList &out) override;

private:
  struct PerFrameData {
    glm::mat4 model{1.0f};
    glm::mat4 view{1.0f};
    glm::mat4 proj{1.0f};
    glm::vec4 cameraPos{0.0f, 0.0f, 0.0f, 1.0f};
    uint32_t albedoTexId = 0;
    uint32_t cubemapTexId = 0;
    uint32_t hasCubemap = 0;
    uint32_t _padding0 = 0;
  };

  struct PushConstants {
    uint64_t perFrameAddress = 0;
  };

  struct RenderableTemplate {
    const OpaqueRenderable *renderable = nullptr;
  };

  struct MeshDrawTemplate {
    const OpaqueRenderable *renderable = nullptr;
    const Submesh *submesh = nullptr;
    uint32_t perFrameIndex = 0;
    BufferHandle vertexBuffer{};
    BufferHandle indexBuffer{};
    uint32_t vertexCount = 0;
  };

  Result<bool, std::string> ensureInitialized();
  Result<bool, std::string> recreateDepthTexture();
  Result<bool, std::string> ensurePerFrameBufferCapacity(size_t requiredBytes);
  Result<bool, std::string> rebuildSceneCache(const RenderScene &scene);
  Result<bool, std::string> createShaders();
  Result<bool, std::string> createPipelines();
  Result<bool, std::string> ensureWireframePipeline();
  void resetWireframePipelineState();
  void destroyDepthTexture();
  void destroyPerFrameBuffer();

  GPUDevice &gpu_;
  std::unique_ptr<Shader> meshShader_;
  std::unique_ptr<Pipeline> meshPipeline_;
  std::unique_ptr<Buffer> perFrameBuffer_;
  TextureHandle depthTexture_{};

  ShaderHandle meshVertexShader_{};
  ShaderHandle meshFragmentShader_{};
  RenderPipelineHandle meshFillPipelineHandle_{};
  RenderPipelineHandle meshWireframePipelineHandle_{};

  size_t perFrameBufferCapacityBytes_ = 0;
  bool initialized_ = false;
  bool wireframePipelineInitialized_ = false;
  bool wireframePipelineUnsupported_ = false;

  const RenderScene *cachedScene_ = nullptr;
  uint64_t cachedTopologyVersion_ = std::numeric_limits<uint64_t>::max();
  std::pmr::vector<RenderableTemplate> renderableTemplates_;
  std::pmr::vector<MeshDrawTemplate> meshDrawTemplates_;
  std::pmr::vector<PerFrameData> perFrameEntries_;
  std::pmr::vector<PushConstants> drawPushConstants_;
  std::pmr::vector<DrawItem> drawItems_;
  DrawItem baseMeshFillDraw_{};
  DrawItem baseMeshWireframeDraw_{};
};

} // namespace nuri
