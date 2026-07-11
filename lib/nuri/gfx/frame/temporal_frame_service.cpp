#include "nuri/pch.h"

#include "nuri/gfx/frame/temporal_frame_service.h"

#include <cmath>
#include <optional>

namespace nuri {
namespace {

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
  constexpr TemporalResetReasonFlags kViewReasons =
      TemporalResetReasonFlags::FirstUse |
      TemporalResetReasonFlags::ExplicitReset |
      TemporalResetReasonFlags::CameraCut |
      TemporalResetReasonFlags::ProjectionChange |
      TemporalResetReasonFlags::BackendRecreation;
  return (static_cast<uint32_t>(flags) &
          static_cast<uint32_t>(kViewReasons)) != 0u;
}

[[nodiscard]] bool invalidatesGrid(TemporalResetReasonFlags flags) noexcept {
  constexpr TemporalResetReasonFlags kGridReasons =
      TemporalResetReasonFlags::Resize |
      TemporalResetReasonFlags::RenderScaleChange;
  return (static_cast<uint32_t>(flags) &
          static_cast<uint32_t>(kGridReasons)) != 0u;
}

} // namespace

struct TemporalFrameService::Impl {
  struct Pending {
    TemporalCameraHistoryState cameraHistory{};
    TemporalFrameFacts facts{};
    PresentationAAPlan plan{};
  };

  TemporalCameraHistoryState committedCameraHistory{};
  TemporalFrameFacts committedFacts{};
  PresentationAAPlan committedPlan{};
  std::optional<Pending> pending{};
  TemporalResetReasonFlags deferredResetReasons =
      TemporalResetReasonFlags::None;
};

TemporalFrameService::TemporalFrameService() : impl_(std::make_unique<Impl>()) {}

TemporalFrameService::~TemporalFrameService() = default;

Result<CameraFrameState, std::string> TemporalFrameService::prepareFrame(
    const Camera &camera, float aspectRatio,
    const RenderSettings::AntiAliasingSettings &antiAliasing,
    const PresentationAAPlan &plan, const TemporalCameraFrameDesc &desc,
    uint64_t frameIndex, double timeSeconds, double deltaSeconds) {
  if (!plan.valid) {
    return Result<CameraFrameState, std::string>::makeError(
        "TemporalFrameService::prepareFrame: presentation AA plan is invalid");
  }
  if (impl_->pending.has_value()) {
    return Result<CameraFrameState, std::string>::makeError(
        "TemporalFrameService::prepareFrame: prior frame is still pending");
  }

  Impl::Pending pending{};
  pending.plan = plan;
  const TemporalCameraHistoryState &committedHistory =
      impl_->committedCameraHistory;
  pending.cameraHistory = impl_->committedCameraHistory;
  CameraFrameState cameraState = makeTemporalCameraFrameState(
      camera, aspectRatio, antiAliasing, desc, pending.cameraHistory);

  TemporalResetReasonFlags reasons =
      resetFlags(cameraState.historyResetReason) |
      impl_->deferredResetReasons;
  if (impl_->committedPlan.valid && impl_->committedPlan != plan) {
    reasons |= TemporalResetReasonFlags::ProviderChange;
  }
  pending.facts = impl_->committedFacts;
  pending.facts.resetReasons = reasons;
  pending.facts.sourceFrameIndex = frameIndex;
  pending.facts.timeSeconds = std::isfinite(timeSeconds) ? timeSeconds : 0.0;
  pending.facts.deltaFromCommittedSeconds =
      impl_->committedFacts.renderedFrameSerial > 0u &&
              std::isfinite(timeSeconds)
          ? std::max(0.0, timeSeconds - impl_->committedFacts.timeSeconds)
          : (std::isfinite(deltaSeconds) ? std::max(0.0, deltaSeconds) : 0.0);
  pending.facts.cameraContinuityValid =
      committedHistory.initialized && !invalidatesView(reasons) &&
      !invalidatesGrid(reasons);
  pending.facts.pendingCommit = true;

  if (invalidatesView(reasons)) {
    ++pending.facts.epochs.viewContinuity;
  }
  if (invalidatesGrid(reasons)) {
    ++pending.facts.epochs.renderGrid;
  }
  if (hasTemporalResetReason(reasons,
                             TemporalResetReasonFlags::ProviderChange)) {
    ++pending.facts.epochs.providerConfiguration;
  }
  if (hasTemporalResetReason(reasons,
                             TemporalResetReasonFlags::ResourceRecreation) ||
      hasTemporalResetReason(reasons,
                             TemporalResetReasonFlags::BackendRecreation)) {
    ++pending.facts.epochs.resourceGeneration;
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
  cameraState.cameraContinuityValid =
      pending.facts.cameraContinuityValid;
  if (cameraState.cameraContinuityValid) {
    cameraState.previousUnjitteredViewProj =
        committedHistory.previousUnjitteredViewProj;
    cameraState.previousJitteredViewProj =
        committedHistory.previousJitteredViewProj;
    cameraState.previousJitterPixelOffset =
        committedHistory.previousJitterPixelOffset;
    cameraState.previousCameraPos = committedHistory.previousCameraPos;
  }
  impl_->pending.emplace(std::move(pending));
  return Result<CameraFrameState, std::string>::makeResult(cameraState);
}

bool TemporalFrameService::commitFrame(uint64_t frameIndex) noexcept {
  if (!impl_->pending.has_value() ||
      impl_->pending->facts.sourceFrameIndex != frameIndex) {
    return false;
  }
  impl_->committedCameraHistory = impl_->pending->cameraHistory;
  impl_->committedFacts = impl_->pending->facts;
  impl_->committedPlan = impl_->pending->plan;
  ++impl_->committedFacts.renderedFrameSerial;
  impl_->committedFacts.pendingCommit = false;
  impl_->pending.reset();
  impl_->deferredResetReasons = TemporalResetReasonFlags::None;
  return true;
}

void TemporalFrameService::abandonFrame(uint64_t frameIndex) noexcept {
  if (!impl_->pending.has_value() ||
      impl_->pending->facts.sourceFrameIndex != frameIndex) {
    return;
  }
  impl_->pending.reset();
  impl_->deferredResetReasons |=
      TemporalResetReasonFlags::SkippedHistoryWrite;
}

void TemporalFrameService::invalidateResources() noexcept {
  impl_->deferredResetReasons |=
      TemporalResetReasonFlags::ResourceRecreation;
}

void TemporalFrameService::invalidateBackend() noexcept {
  impl_->deferredResetReasons |= TemporalResetReasonFlags::BackendRecreation;
}

void TemporalFrameService::reset() noexcept { *impl_ = Impl{}; }

const TemporalFrameFacts &TemporalFrameService::facts() const noexcept {
  return impl_->pending.has_value() ? impl_->pending->facts
                                    : impl_->committedFacts;
}

const TemporalCameraHistoryState &
TemporalFrameService::cameraHistory() const noexcept {
  return impl_->committedCameraHistory;
}

} // namespace nuri
