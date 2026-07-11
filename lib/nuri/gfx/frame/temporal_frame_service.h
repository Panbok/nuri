#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/frame/presentation_aa_plan.h"

#include <cstdint>
#include <memory>
#include <string>

namespace nuri {

struct TemporalEpochs {
  uint64_t viewContinuity = 0u;
  uint64_t renderGrid = 0u;
  uint64_t providerConfiguration = 0u;
  uint64_t resourceGeneration = 0u;
  bool operator==(const TemporalEpochs &) const = default;
};

enum class TemporalResetReasonFlags : uint32_t {
  None = 0u,
  FirstUse = 1u << 0u,
  ExplicitReset = 1u << 1u,
  CameraCut = 1u << 2u,
  ProjectionChange = 1u << 3u,
  Resize = 1u << 4u,
  RenderScaleChange = 1u << 5u,
  ProviderChange = 1u << 6u,
  SceneDiscontinuity = 1u << 7u,
  ResourceRecreation = 1u << 8u,
  SkippedHistoryWrite = 1u << 9u,
  BackendRecreation = 1u << 10u,
};

[[nodiscard]] constexpr TemporalResetReasonFlags
operator|(TemporalResetReasonFlags lhs, TemporalResetReasonFlags rhs) noexcept {
  return static_cast<TemporalResetReasonFlags>(static_cast<uint32_t>(lhs) |
                                               static_cast<uint32_t>(rhs));
}

constexpr TemporalResetReasonFlags &
operator|=(TemporalResetReasonFlags &lhs,
           TemporalResetReasonFlags rhs) noexcept {
  lhs = lhs | rhs;
  return lhs;
}

[[nodiscard]] constexpr bool
hasTemporalResetReason(TemporalResetReasonFlags flags,
                       TemporalResetReasonFlags reason) noexcept {
  return (static_cast<uint32_t>(flags) & static_cast<uint32_t>(reason)) != 0u;
}

struct TemporalFrameFacts {
  TemporalEpochs epochs{};
  TemporalResetReasonFlags resetReasons = TemporalResetReasonFlags::None;
  uint64_t renderedFrameSerial = 0u;
  uint64_t sourceFrameIndex = 0u;
  double timeSeconds = 0.0;
  double deltaFromCommittedSeconds = 0.0;
  bool cameraContinuityValid = false;
  bool pendingCommit = false;
};

class NURI_API TemporalFrameService final {
public:
  TemporalFrameService();
  ~TemporalFrameService();

  TemporalFrameService(const TemporalFrameService &) = delete;
  TemporalFrameService &operator=(const TemporalFrameService &) = delete;
  TemporalFrameService(TemporalFrameService &&) = delete;
  TemporalFrameService &operator=(TemporalFrameService &&) = delete;

  [[nodiscard]] Result<CameraFrameState, std::string>
  prepareFrame(const Camera &camera, float aspectRatio,
               const RenderSettings::AntiAliasingSettings &antiAliasing,
               const PresentationAAPlan &plan,
               const TemporalCameraFrameDesc &desc, uint64_t frameIndex,
               double timeSeconds, double deltaSeconds);

  [[nodiscard]] bool commitFrame(uint64_t frameIndex) noexcept;
  void abandonFrame(uint64_t frameIndex) noexcept;
  void invalidateResources() noexcept;
  void invalidateBackend() noexcept;
  void reset() noexcept;

  [[nodiscard]] const TemporalFrameFacts &facts() const noexcept;
  [[nodiscard]] const TemporalCameraHistoryState &cameraHistory() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace nuri
