#pragma once

#include "nuri/core/result.h"
#include "nuri/gfx/frame/render_frame_context.h"

#include <string>
#include <string_view>

namespace nuri {

[[nodiscard]] constexpr PresentationAAUnsupportedReason
msaa4xUnsupportedReason(
    const PresentationAAGpuCapabilities &capabilities) noexcept {
  if (!capabilities.sample4Color) {
    return PresentationAAUnsupportedReason::Sample4Color;
  }
  if (!capabilities.sample4Depth) {
    return PresentationAAUnsupportedReason::Sample4Depth;
  }
  if (!capabilities.depthResolveMin) {
    return PresentationAAUnsupportedReason::DepthResolveMin;
  }
  if (!capabilities.alphaToCoverage) {
    return PresentationAAUnsupportedReason::AlphaToCoverage;
  }
  return PresentationAAUnsupportedReason::None;
}

[[nodiscard]] constexpr std::string_view presentationAAUnsupportedReasonName(
    PresentationAAUnsupportedReason reason) noexcept {
  switch (reason) {
  case PresentationAAUnsupportedReason::None:
    return "none";
  case PresentationAAUnsupportedReason::Sample4Color:
    return "sample4_color_unsupported";
  case PresentationAAUnsupportedReason::Sample4Depth:
    return "sample4_depth_unsupported";
  case PresentationAAUnsupportedReason::DepthResolveMin:
    return "depth_resolve_min_unsupported";
  case PresentationAAUnsupportedReason::AlphaToCoverage:
    return "alpha_to_coverage_unsupported";
  }
  return "unknown";
}

[[nodiscard]] inline TemporalReconstructionProvider
sanitizeTemporalReconstructionProvider(
    TemporalReconstructionProvider provider) noexcept {
  switch (provider) {
  case TemporalReconstructionProvider::Legacy:
  case TemporalReconstructionProvider::Reference:
  case TemporalReconstructionProvider::External:
    return provider;
  default:
    return TemporalReconstructionProvider::Legacy;
  }
}

[[nodiscard]] inline Result<PresentationAAPlan, std::string>
buildPresentationAAPlan(
    const RenderSettings &sourceSettings,
    const PresentationAAProviderCapabilities &providerCapabilities = {},
    const PresentationAAGpuCapabilities &gpuCapabilities = {}) {
  RenderSettings settings = sourceSettings;
  sanitizeAntiAliasingSettings(settings.antiAliasing);
  sanitizeAmbientOcclusionSettings(settings.ambientOcclusion, settings.opaque,
                                   settings.antiAliasing);

  PresentationAAPlan plan{};
  const AntiAliasingMode mode = settings.antiAliasing.mode;
  const TemporalReconstructionProvider temporalProvider =
      sanitizeTemporalReconstructionProvider(
          settings.antiAliasing.temporalProvider);

  switch (mode) {
  case AntiAliasingMode::None:
    break;
  case AntiAliasingMode::SpatialFallback:
    plan.spatialCleanup = SpatialCleanupPoint::PreComposition;
    break;
  case AntiAliasingMode::MSAA4x:
    if (const PresentationAAUnsupportedReason unsupportedReason =
            msaa4xUnsupportedReason(gpuCapabilities);
        unsupportedReason != PresentationAAUnsupportedReason::None) {
      return Result<PresentationAAPlan, std::string>::makeError(
          "MSAA4x unsupported: " +
          std::string(presentationAAUnsupportedReasonName(unsupportedReason)));
    }
    plan.coverage = CoverageMode::Sample4;
    plan.alphaCoverage = AlphaCoveragePolicy::ThresholdedAlphaToCoverage;
    plan.transparency = TransparencyAAPolicy::SingleSamplePostResolve;
    plan.sampleShadingSupported = gpuCapabilities.sampleRateShading;
    plan.sampleShadingEnabled = false;
    plan.spatialCleanup =
        settings.antiAliasing.debug.spatialPostMsaaCleanup
            ? SpatialCleanupPoint::PostTransparency
            : SpatialCleanupPoint::Off;
    break;
  case AntiAliasingMode::TAA:
    switch (temporalProvider) {
    case TemporalReconstructionProvider::Legacy:
      plan.reconstruction = ColorReconstruction::LegacyTAA;
      break;
    case TemporalReconstructionProvider::Reference:
      if (!providerCapabilities.referenceTemporal) {
        return Result<PresentationAAPlan, std::string>::makeError(
            "Reference TAA provider is unavailable");
      }
      plan.reconstruction = ColorReconstruction::ReferenceTAA;
      break;
    case TemporalReconstructionProvider::External:
      if (!providerCapabilities.externalTemporal) {
        return Result<PresentationAAPlan, std::string>::makeError(
            "External temporal provider is unavailable");
      }
      if (!providerCapabilities.reactiveMask ||
          !providerCapabilities.compositionMask) {
        return Result<PresentationAAPlan, std::string>::makeError(
            "External temporal provider lacks required mask capabilities");
      }
      plan.reconstruction = ColorReconstruction::ExternalTemporal;
      plan.needsCompositionMask = true;
      break;
    }
    plan.jitterScene = settings.antiAliasing.debug.jitterEnabled;
    plan.needsMotion = true;
    plan.needsReactiveMask = providerCapabilities.reactiveMask;
    plan.needsMotionClass = true;
    // Legacy owns its historical invalid-history spatial fallback. Reference
    // TAA is deliberately a temporal-only quality oracle: an invalid history
    // sample must resolve to the unfiltered current input, not silently route
    // through the legacy cleanup policy inherited by the quality presets.
    if (temporalProvider == TemporalReconstructionProvider::Legacy &&
        settings.antiAliasing.debug.spatialPostTaaCleanup) {
      plan.spatialCleanup = SpatialCleanupPoint::PreComposition;
    }
    break;
  }

  plan.gtaoTemporal = settings.ambientOcclusion.active &&
                      settings.ambientOcclusion.temporalAccumulation;
  plan.needsMotion = plan.needsMotion || plan.gtaoTemporal;
  // GTAO is an independent temporal consumer. Until motion producers publish
  // MotionClass directly, the early classification pass uses the reactive mask
  // as the per-pixel invalid-correspondence input. Request both resources even
  // when color reconstruction is disabled.
  plan.needsMotionClass = plan.needsMotionClass || plan.gtaoTemporal;
  plan.needsReactiveMask = plan.needsReactiveMask || plan.gtaoTemporal;
  plan.valid = true;
  return Result<PresentationAAPlan, std::string>::makeResult(plan);
}

[[nodiscard]] constexpr bool
usesTemporalColorReconstruction(const PresentationAAPlan &plan) noexcept {
  return plan.reconstruction != ColorReconstruction::Off;
}

[[nodiscard]] inline PresentationAAPlan
presentationAAPlanForFrame(const RenderFrameContext &frame) {
  if (frame.presentationAA.valid) {
    return frame.presentationAA;
  }
  const auto plan = buildPresentationAAPlan(renderSettingsOrDefault(frame));
  return plan.hasError() ? PresentationAAPlan{} : plan.value();
}

[[nodiscard]] constexpr bool
hasTemporalCameraContinuity(const CameraFrameState &camera) noexcept {
  // historyValid is retained as a compatibility alias for callers which build
  // feature contexts directly. Runtime frame construction publishes the
  // provider-independent cameraContinuityValid bit.
  return camera.cameraContinuityValid || camera.historyValid;
}

} // namespace nuri
