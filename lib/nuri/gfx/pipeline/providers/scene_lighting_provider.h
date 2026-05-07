#pragma once

#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "nuri/defines.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/frame_data_provider.h"
#include "nuri/resources/gpu/buffer.h"

namespace nuri {

class RenderScene;

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
  struct SlotUploadState {
    const RenderScene *scene = nullptr;
    uint64_t sceneId = 0u;
    uint64_t lightTopologyVersion = std::numeric_limits<uint64_t>::max();
    uint64_t lightTransformVersion = std::numeric_limits<uint64_t>::max();
    uint32_t directionalLightCount = std::numeric_limits<uint32_t>::max();
    uint32_t localLightCount = std::numeric_limits<uint32_t>::max();
    bool hasFrameData = false;
    ForwardSceneFrameData frameData{};
    ForwardSceneFrameData postTaaFrameData{};
  };

  Result<bool, std::string> ensureBufferRingCapacity(size_t requiredBytes,
                                                     uint32_t requiredCount);
  Result<uint64_t, std::string> ensureDisabledShadowFrameBuffer();
  Result<uint32_t, std::string>
  resolveMaterialSamplerId(RenderFrameContext &frame);
  Result<SamplerHandle, std::string>
  ensureTaaMaterialMipBiasSampler(const SamplerDesc &desc);
  void destroyBuffers();
  void destroyCachedSamplers();
  [[nodiscard]] Buffer *currentBuffer(uint64_t frameIndex) const noexcept;

  GPUDevice &gpu_;
  std::vector<std::unique_ptr<Buffer>> sceneDataBuffers_;
  std::vector<SlotUploadState> slotUploadStates_;
  std::unique_ptr<Buffer> disabledShadowFrameBuffer_;
  SamplerHandle taaMaterialMipBiasSampler_{};
  std::optional<SamplerDesc> taaMaterialMipBiasSamplerDesc_{};
  size_t sceneDataBufferCapacityBytes_ = 0;
  uint64_t loggedAddressProbeTopologyVersion_ =
      std::numeric_limits<uint64_t>::max();
  uint64_t loggedLightStateSignature_ = std::numeric_limits<uint64_t>::max();
};

} // namespace nuri
