#pragma once
#include "nuri/core/runtime_config.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/owned_program_bundle.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
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
  uint32_t recordingLeaseCount = 0u;
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
  struct PendingLease {
    uint64_t frameIndex = 0u;
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
  OwnedProgramBundle program_{};
  std::array<SamplerHandle, 2> samplers_{};
  std::array<TextureHandle, 2> luts_{};
  std::vector<SubmittedPostAA> submittedPostAA_{};
  std::optional<PendingLease> pendingLease_{};
  std::string initializationError_{};
  Result<bool, std::string> initialize();
  Result<bool, std::string> ensureLuts();
  void destroyResources();
};

NURI_API void registerSpatialAAStage(
    RenderPipeline &pipeline, GPUDevice &gpu, RuntimeCompositeConfig config,
    SpatialAAPlacement placement = SpatialAAPlacement::SceneColor);

} // namespace nuri
