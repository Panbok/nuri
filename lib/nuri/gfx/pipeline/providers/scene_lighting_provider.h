#pragma once

#include "nuri/defines.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/frame_data_provider.h"
#include "nuri/resources/gpu/buffer.h"

#include <limits>
#include <memory>

namespace nuri {

class NURI_API SceneLightingProvider final : public FrameDataProvider {
public:
  explicit SceneLightingProvider(GPUDevice &gpu);
  ~SceneLightingProvider() override;

  SceneLightingProvider(const SceneLightingProvider &) = delete;
  SceneLightingProvider &operator=(const SceneLightingProvider &) = delete;
  SceneLightingProvider(SceneLightingProvider &&) = delete;
  SceneLightingProvider &operator=(SceneLightingProvider &&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "SceneLightingProvider";
  }
  Result<bool, std::string> prepare(FrameBuildContext &ctx) override;

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
