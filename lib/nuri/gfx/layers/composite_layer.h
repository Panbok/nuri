#pragma once

#include "nuri/core/layer.h"
#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/resources/gpu/buffer.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <utility>

#include <glm/glm.hpp>

namespace nuri {

using CompositeLayerConfig = RuntimeOpaqueShaderConfig;

class Shader;

class NURI_API CompositeLayer final : public Layer {
public:
  explicit CompositeLayer(GPUDevice &gpu, CompositeLayerConfig config);
  ~CompositeLayer() override;

  CompositeLayer(const CompositeLayer &) = delete;
  CompositeLayer &operator=(const CompositeLayer &) = delete;
  CompositeLayer(CompositeLayer &&) = delete;
  CompositeLayer &operator=(CompositeLayer &&) = delete;

  static std::unique_ptr<CompositeLayer> create(GPUDevice &gpu,
                                                CompositeLayerConfig config) {
    return std::make_unique<CompositeLayer>(gpu, std::move(config));
  }

  void onAttach() override;
  void onDetach() override;
  Result<bool, std::string>
  buildRenderGraph(RenderFrameContext &frame,
                   RenderGraphBuilder &graph) override;

private:
  struct FrameData {
    glm::mat4 view{1.0f};
    glm::mat4 proj{1.0f};
    glm::vec4 cameraPos{0.0f, 0.0f, 0.0f, 1.0f};
    uint32_t cubemapTexId = 0;
    uint32_t hasCubemap = 0;
    uint32_t irradianceTexId = 0;
    uint32_t prefilteredGgxTexId = 0;
    uint32_t prefilteredCharlieTexId = 0;
    uint32_t brdfLutTexId = 0;
    uint32_t flags = 0;
    uint32_t cubemapSamplerId = 0;
    uint32_t sceneColorTexId = 0;
    uint32_t sceneColorSamplerId = 0;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;

    [[nodiscard]] bool operator==(const FrameData &other) const noexcept {
      for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
          if (view[column][row] != other.view[column][row] ||
              proj[column][row] != other.proj[column][row]) {
            return false;
          }
        }
      }
      return cameraPos.x == other.cameraPos.x &&
             cameraPos.y == other.cameraPos.y &&
             cameraPos.z == other.cameraPos.z &&
             cameraPos.w == other.cameraPos.w &&
             cubemapTexId == other.cubemapTexId &&
             hasCubemap == other.hasCubemap &&
             irradianceTexId == other.irradianceTexId &&
             prefilteredGgxTexId == other.prefilteredGgxTexId &&
             prefilteredCharlieTexId == other.prefilteredCharlieTexId &&
             brdfLutTexId == other.brdfLutTexId && flags == other.flags &&
             cubemapSamplerId == other.cubemapSamplerId &&
             sceneColorTexId == other.sceneColorTexId &&
             sceneColorSamplerId == other.sceneColorSamplerId &&
             reserved0 == other.reserved0 && reserved1 == other.reserved1;
    }
  };
  static_assert(sizeof(FrameData) == 192,
                "CompositeLayer::FrameData must match shader layout");

  struct PushConstants {
    uint64_t frameDataAddress = 0;
  };
  static_assert(sizeof(PushConstants) <= 128,
                "CompositeLayer::PushConstants exceeds Vulkan guarantee");

  Result<bool, std::string> ensureInitialized();
  Result<bool, std::string> createShaders();
  Result<bool, std::string> ensurePipeline();
  Result<bool, std::string> ensureFrameBufferCapacity(size_t requiredBytes);
  void destroyPipelineState();
  void destroyShaders();
  void destroyBuffers();

  GPUDevice &gpu_;
  std::unique_ptr<Shader> shader_;
  std::unique_ptr<Buffer> frameBuffer_;

  ShaderHandle vertexShader_{};
  ShaderHandle fragmentShader_{};
  RenderPipelineHandle pipelineHandle_{};
  Format pipelineColorFormat_ = Format::Count;

  size_t frameBufferCapacityBytes_ = 0;
  bool initialized_ = false;
  bool frameDataUploadValid_ = false;

  FrameData frameData_{};
  FrameData uploadedFrameData_{};
  PushConstants pushConstants_{};
  DrawItem drawItem_{};
  std::filesystem::path vertexPath_{};
  std::filesystem::path fragmentPath_{};
};

} // namespace nuri
