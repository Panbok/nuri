#pragma once

#include "nuri/core/layer.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/resources/gpu/buffer.h"

#include <limits>
#include <memory>
#include <memory_resource>

namespace nuri {

class NURI_API SceneLightingLayer final : public Layer {
public:
  explicit SceneLightingLayer(
      GPUDevice &gpu,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~SceneLightingLayer() override;

  SceneLightingLayer(const SceneLightingLayer &) = delete;
  SceneLightingLayer &operator=(const SceneLightingLayer &) = delete;
  SceneLightingLayer(SceneLightingLayer &&) = delete;
  SceneLightingLayer &operator=(SceneLightingLayer &&) = delete;

  static std::unique_ptr<SceneLightingLayer>
  create(GPUDevice &gpu,
         std::pmr::memory_resource *memory = std::pmr::get_default_resource()) {
    return std::make_unique<SceneLightingLayer>(gpu, memory);
  }

  void onDetach() override;
  Result<bool, std::string>
  buildRenderGraph(RenderFrameContext &frame,
                   RenderGraphBuilder &graph) override;

private:
  Result<bool, std::string> ensureBufferCapacity(size_t requiredBytes);
  void destroyBuffer();

  GPUDevice &gpu_;
  std::unique_ptr<Buffer> sceneDataBuffer_;
  size_t sceneDataBufferCapacityBytes_ = 0;
  ForwardSceneFrameData uploadedFrameData_{};
  bool frameDataUploadValid_ = false;
  uint64_t cachedLightTopologyVersion_ = std::numeric_limits<uint64_t>::max();
  uint64_t cachedLightTransformVersion_ = std::numeric_limits<uint64_t>::max();
};

} // namespace nuri
