#pragma once
#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/gfx/shader.h"
#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>
namespace nuri {

class RenderPipeline;

enum class SpatialAAPlacement : uint8_t {
  SceneColor = 0,
  PostTransparent = 1,
};

struct SpatialAALifecycleSnapshot {
  uint32_t scratchSlotCount = 0u;
  uint32_t recordingLeaseCount = 0u;
  uint32_t submittedScratchCount = 0u;
  uint32_t retiredScratchGroupCount = 0u;
  uint32_t submittedPostAALedgerCount = 0u;
};

class NURI_API SpatialAAPass final {
public:
  explicit SpatialAAPass(
      GPUDevice &gpu, RuntimeCompositeConfig config,
      SpatialAAPlacement placement = SpatialAAPlacement::SceneColor);
  ~SpatialAAPass();
  [[nodiscard]] bool isEnabled(const FrameBuildContext &ctx) const;
  Result<bool, std::string> prepare(FrameBuildContext &ctx);
  Result<bool, std::string> build(FrameBuildContext &ctx);
  void onFrameSubmitted(const RenderFrameContext &frame) noexcept;
  void onFrameAbandoned(const RenderFrameContext &frame) noexcept;
  [[nodiscard]] SpatialAALifecycleSnapshot lifecycleSnapshot() const noexcept;

private:
  struct ScratchSlot {
    SubmissionHandle submission{};
    bool leased = false;
  };
  struct RetiredScratch {
    std::array<TextureHandle, 3> textures{};
    SubmissionHandle submission{};
  };
  struct PendingLease {
    uint64_t frameIndex = 0u;
    uint32_t slot = 0u;
    uint32_t passCount = 0u;
    bool postAA = false;
  };
  struct SubmittedPostAA {
    uint64_t sourceFrameIndex = 0u;
    SubmissionHandle submission{};
    uint32_t passCount = 0u;
  };
  GPUDevice &gpu_;
  RuntimeCompositeConfig config_{};
  SpatialAAPlacement placement_;
  ShaderHandle vertexShader_{};
  std::array<ShaderHandle, 3> fragmentShaders_{};
  std::array<RenderPipelineHandle, 3> pipelines_{};
  std::array<SamplerHandle, 2> samplers_{};
  std::array<TextureHandle, 2> luts_{};
  std::array<std::vector<TextureHandle>, 3> scratchTextures_{};
  std::vector<ScratchSlot> scratchSlots_{};
  std::vector<RetiredScratch> retiredScratch_{};
  std::vector<SubmittedPostAA> submittedPostAA_{};
  std::optional<PendingLease> pendingLease_{};
  std::string initializationError_{};
  uint32_t scratchWidth_ = 0u;
  uint32_t scratchHeight_ = 0u;
  uint32_t scratchRingCount_ = 0u;
  Format outputScratchFormat_ = Format::Count;
  Result<bool, std::string> initialize();
  Result<bool, std::string> ensureLuts();
  Result<bool, std::string> ensureScratchTextures(FrameBuildContext &ctx);
  void collectCompletedScratch();
  void retireCurrentScratch();
  void destroyResources();
};

NURI_API void registerSpatialAAStage(
    RenderPipeline &pipeline, GPUDevice &gpu, RuntimeCompositeConfig config,
    SpatialAAPlacement placement = SpatialAAPlacement::SceneColor);

} // namespace nuri
