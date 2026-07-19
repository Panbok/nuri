#pragma once
#include "nuri/defines.h"
#include "nuri/gfx/dynamic_buffer.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/resources/gpu/buffer.h"
#include <memory>
#include <string>
#include <vector>
namespace nuri {

class RenderScene;

class NURI_API SceneLightingProvider final {
public:
  explicit SceneLightingProvider(GPUDevice &gpu);
  ~SceneLightingProvider();
  SceneLightingProvider(const SceneLightingProvider &) = delete;
  SceneLightingProvider &operator=(const SceneLightingProvider &) = delete;
  SceneLightingProvider(SceneLightingProvider &&) = delete;
  SceneLightingProvider &operator=(SceneLightingProvider &&) = delete;
  Result<bool, std::string> prepare(FrameBuildContext &ctx);

private:
  struct Slot {
    DynamicBufferSlot buffer;
    const RenderScene *scene = nullptr;
    uint64_t sceneId = 0u;
    uint64_t lightTopologyVersion = 0u;
    uint64_t lightTransformVersion = 0u;
    uint32_t directionalLightCount = 0u;
    uint32_t localLightCount = 0u;
    bool hasFrameData = false;
    ForwardSceneFrameData frameData{};
    ForwardSceneFrameData postTaaFrameData{};
  };
  Result<bool, std::string> ensureBufferRingCapacity(size_t requiredBytes,
                                                     uint32_t requiredCount);
  Result<uint64_t, std::string> ensureDisabledShadowFrameBuffer();
  Result<uint32_t, std::string>
  resolveMaterialSamplerId(RenderFrameContext &frame);
  GPUDevice &gpu_;
  std::vector<Slot> slots_;
  std::unique_ptr<Buffer> disabledShadowFrameBuffer_;
  SamplerHandle taaMaterialMipBiasSampler_{};
  SamplerDesc taaMaterialMipBiasSamplerDesc_{};
};

} // namespace nuri
