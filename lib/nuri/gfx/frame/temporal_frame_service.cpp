#include "nuri/gfx/frame/temporal_frame_service.h"
#include "nuri/gfx/frame/presentation_aa_plan.h"
#include <cmath>
namespace nuri {
namespace {
constexpr TemporalResetReasonFlags kViewResetReasons =
    TemporalResetReasonFlags::FirstUse |
    TemporalResetReasonFlags::ExplicitReset |
    TemporalResetReasonFlags::CameraCut |
    TemporalResetReasonFlags::ProjectionChange |
    TemporalResetReasonFlags::BackendRecreation;
constexpr TemporalResetReasonFlags kGridResetReasons =
    TemporalResetReasonFlags::Resize |
    TemporalResetReasonFlags::RenderScaleChange;
struct EpochRule {
  TemporalResetReasonFlags reasons;
  uint64_t TemporalEpochs::*epoch;
};
constexpr std::array kEpochRules{
    EpochRule{kViewResetReasons, &TemporalEpochs::viewContinuity},
    EpochRule{kGridResetReasons, &TemporalEpochs::renderGrid},
    EpochRule{TemporalResetReasonFlags::ProviderChange,
              &TemporalEpochs::providerConfiguration},
    EpochRule{TemporalResetReasonFlags::ResourceRecreation |
                  TemporalResetReasonFlags::BackendRecreation,
              &TemporalEpochs::resourceGeneration},
};
[[nodiscard]] TemporalResetReasonFlags
resetFlags(TemporalHistoryResetReason reason) noexcept {
  switch (reason) {
  case TemporalHistoryResetReason::None:
    return TemporalResetReasonFlags::None;
  case TemporalHistoryResetReason::FirstFrame:
    return TemporalResetReasonFlags::FirstUse;
  case TemporalHistoryResetReason::HistoryResetRequested:
    return TemporalResetReasonFlags::ExplicitReset;
  case TemporalHistoryResetReason::AntiAliasingModeChanged:
    return TemporalResetReasonFlags::ProviderChange;
  case TemporalHistoryResetReason::Resize:
    return TemporalResetReasonFlags::Resize;
  case TemporalHistoryResetReason::ProjectionChanged:
    return TemporalResetReasonFlags::ProjectionChange;
  case TemporalHistoryResetReason::RenderScaleChanged:
    return TemporalResetReasonFlags::RenderScaleChange;
  case TemporalHistoryResetReason::CameraCut:
    return TemporalResetReasonFlags::CameraCut;
  case TemporalHistoryResetReason::InvalidHistoryTexture:
    return TemporalResetReasonFlags::ResourceRecreation;
  case TemporalHistoryResetReason::SceneContentChanged:
    return TemporalResetReasonFlags::SceneDiscontinuity;
  }
  return TemporalResetReasonFlags::SceneDiscontinuity;
}
[[nodiscard]] bool invalidatesView(TemporalResetReasonFlags flags) noexcept {
  return (static_cast<uint32_t>(flags) &
          static_cast<uint32_t>(kViewResetReasons)) != 0u;
}
[[nodiscard]] bool invalidatesGrid(TemporalResetReasonFlags flags) noexcept {
  return (static_cast<uint32_t>(flags) &
          static_cast<uint32_t>(kGridResetReasons)) != 0u;
}
} // namespace

Result<CameraFrameState, std::string> TemporalFrameService::prepareFrame(
    const Camera &camera, float aspectRatio,
    const RenderSettings::AntiAliasingSettings &antiAliasing,
    const PresentationAAPlan &plan, const TemporalCameraFrameDesc &desc,
    uint64_t frameIndex, double timeSeconds, double deltaSeconds) {
  if (!plan.valid) {
    return Result<CameraFrameState, std::string>::makeError(
        "TemporalFrameService::prepareFrame: presentation AA plan is invalid");
  }
  if (pending_.has_value()) {
    return Result<CameraFrameState, std::string>::makeError(
        "TemporalFrameService::prepareFrame: prior frame is still pending");
  }
  Pending pending{};
  pending.plan = plan;
  const TemporalCameraHistoryState &committedHistory = committedCameraHistory_;
  pending.cameraHistory = committedCameraHistory_;
  CameraFrameState cameraState = makeTemporalCameraFrameState(
      camera, aspectRatio, antiAliasing, desc, pending.cameraHistory);
  TemporalResetReasonFlags reasons =
      resetFlags(cameraState.historyResetReason) | deferredResetReasons_;
  if (committedPlan_.valid &&
      !temporalAAContinuityEquivalent(committedPlan_, plan)) {
    reasons |= TemporalResetReasonFlags::ProviderChange;
  }
  pending.facts = committedFacts_;
  pending.facts.resetReasons = reasons;
  pending.facts.sourceFrameIndex = frameIndex;
  pending.facts.timeSeconds = std::isfinite(timeSeconds) ? timeSeconds : 0.0;
  pending.facts.deltaFromCommittedSeconds =
      committedFacts_.renderedFrameSerial > 0u && std::isfinite(timeSeconds)
          ? std::max(0.0, timeSeconds - committedFacts_.timeSeconds)
          : (std::isfinite(deltaSeconds) ? std::max(0.0, deltaSeconds) : 0.0);
  pending.facts.cameraContinuityValid = committedHistory.initialized &&
                                        !invalidatesView(reasons) &&
                                        !invalidatesGrid(reasons);
  pending.facts.pendingCommit = true;
  for (const EpochRule &rule : kEpochRules) {
    if ((static_cast<uint32_t>(reasons) &
         static_cast<uint32_t>(rule.reasons)) != 0u) {
      ++(pending.facts.epochs.*rule.epoch);
    }
  }
  cameraState.jitterEnabled = plan.jitterScene && cameraState.jitterEnabled;
  cameraState.historyValid =
      cameraState.historyValid &&
      !hasTemporalResetReason(reasons,
                              TemporalResetReasonFlags::ProviderChange) &&
      !hasTemporalResetReason(reasons,
                              TemporalResetReasonFlags::ResourceRecreation) &&
      !hasTemporalResetReason(reasons,
                              TemporalResetReasonFlags::BackendRecreation);
  if (!cameraState.jitterEnabled) {
    cameraState.jitterFrozen = false;
  }
  cameraState.cameraContinuityValid = pending.facts.cameraContinuityValid;
  if (cameraState.cameraContinuityValid) {
    cameraState.previousUnjitteredViewProj =
        committedHistory.previousUnjitteredViewProj;
    cameraState.previousJitteredViewProj =
        committedHistory.previousJitteredViewProj;
    cameraState.previousJitterPixelOffset =
        committedHistory.previousJitterPixelOffset;
    cameraState.previousCameraPos = committedHistory.previousCameraPos;
  }
  pending_.emplace(std::move(pending));
  return Result<CameraFrameState, std::string>::makeResult(cameraState);
}

bool TemporalFrameService::commitFrame(uint64_t frameIndex) noexcept {
  if (!pending_.has_value() || pending_->facts.sourceFrameIndex != frameIndex) {
    return false;
  }
  committedCameraHistory_ = pending_->cameraHistory;
  committedFacts_ = pending_->facts;
  committedPlan_ = pending_->plan;
  ++committedFacts_.renderedFrameSerial;
  committedFacts_.pendingCommit = false;
  pending_.reset();
  deferredResetReasons_ = TemporalResetReasonFlags::None;
  return true;
}

void TemporalFrameService::abandonFrame(uint64_t frameIndex) noexcept {
  if (!pending_.has_value() || pending_->facts.sourceFrameIndex != frameIndex) {
    return;
  }
  pending_.reset();
  deferredResetReasons_ |= TemporalResetReasonFlags::SkippedHistoryWrite;
}

void TemporalFrameService::invalidateResources() noexcept {
  deferredResetReasons_ |= TemporalResetReasonFlags::ResourceRecreation;
}

void TemporalFrameService::invalidateBackend() noexcept {
  deferredResetReasons_ |= TemporalResetReasonFlags::BackendRecreation;
}

void TemporalFrameService::reset() noexcept {
  committedCameraHistory_ = {};
  committedFacts_ = {};
  committedPlan_ = {};
  pending_.reset();
  deferredResetReasons_ = TemporalResetReasonFlags::None;
}

const TemporalFrameFacts &TemporalFrameService::facts() const noexcept {
  return pending_.has_value() ? pending_->facts : committedFacts_;
}

const TemporalCameraHistoryState &
TemporalFrameService::cameraHistory() const noexcept {
  return committedCameraHistory_;
}

} // namespace nuri
