#pragma once
#include "nuri/core/result.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include <string>
#include <string_view>
namespace nuri {

[[nodiscard]] constexpr bool
isPostAASpecularDebugView(AntiAliasingDebugView view) noexcept {
  return view == AntiAliasingDebugView::SpecularAAVariance ||
         view == AntiAliasingDebugView::SpecularAARoughnessDelta;
}

[[nodiscard]] inline PostAAPlan
resolvePostAAPlan(const RenderSettings::AntiAliasingSettings &settings,
                  CoverageMode coverage) noexcept {
  const PostAASettings &resolved = settings.postAA;
  PostAAPlan plan{};
  plan.requested = resolved.enabled;
  const bool msaaEligible =
      coverage == CoverageMode::Sample4 || coverage == CoverageMode::Sample8;
  if (!msaaEligible) {
    if (resolved.enabled) {
      plan.inactiveReason = PostAAInactiveReason::CoverageIsSingleSample;
    }
    return plan;
  }
  plan.specularAADebugOverride = settings.debug.specularAAOverride;
  if (!resolved.enabled) {
    plan.resolvedMaterialSpecularAA =
        plan.specularAADebugOverride == SpecularAADebugOverride::ForceOff
            ? ResolvedMaterialSpecularAA::Off
            : ResolvedMaterialSpecularAA::LegacyShadingNormalDerivative;
    return plan;
  }
  plan.specular = resolved.specular;
  plan.spatial = resolved.spatial;
  plan.materialVarianceScale = resolved.materialVarianceScale;
  plan.geometricVarianceScale = resolved.geometricVarianceScale;
  plan.maxSlopeVariance = resolved.maxSlopeVariance;
  plan.active = plan.specular == PostAASpecularAlgorithm::BakedClean ||
                plan.spatial == PostAASpatialAlgorithm::Smaa1x;
  if (!plan.active) {
    plan.specular = PostAASpecularAlgorithm::InheritCurrent;
    plan.spatial = PostAASpatialAlgorithm::Off;
    plan.inactiveReason = PostAAInactiveReason::NoComponentEnabled;
    plan.resolvedMaterialSpecularAA =
        plan.specularAADebugOverride == SpecularAADebugOverride::ForceOff
            ? ResolvedMaterialSpecularAA::Off
            : ResolvedMaterialSpecularAA::LegacyShadingNormalDerivative;
    return plan;
  }
  plan.inactiveReason = PostAAInactiveReason::None;
  plan.resolvedMaterialSpecularAA =
      plan.specularAADebugOverride == SpecularAADebugOverride::ForceOff
          ? ResolvedMaterialSpecularAA::Off
          : (plan.specular == PostAASpecularAlgorithm::BakedClean
                 ? ResolvedMaterialSpecularAA::BakedClean
                 : ResolvedMaterialSpecularAA::LegacyShadingNormalDerivative);
  const AntiAliasingDebugView debugView = settings.debug.view;
  plan.debugView = plan.specular == PostAASpecularAlgorithm::BakedClean &&
                           isPostAASpecularDebugView(debugView)
                       ? debugView
                       : AntiAliasingDebugView::None;
  return plan;
}

[[nodiscard]] constexpr bool
temporalAAContinuityEquivalent(const PresentationAAPlan &lhs,
                               const PresentationAAPlan &rhs) noexcept {
  return lhs.coverage == rhs.coverage &&
         lhs.reconstruction == rhs.reconstruction &&
         lhs.alphaCoverage == rhs.alphaCoverage &&
         lhs.transparency == rhs.transparency &&
         lhs.sampleShadingSupported == rhs.sampleShadingSupported &&
         lhs.sampleShadingEnabled == rhs.sampleShadingEnabled &&
         lhs.jitterScene == rhs.jitterScene &&
         lhs.needsMotion == rhs.needsMotion &&
         lhs.needsReactiveMask == rhs.needsReactiveMask &&
         lhs.needsCompositionMask == rhs.needsCompositionMask &&
         lhs.needsMotionClass == rhs.needsMotionClass &&
         lhs.gtaoTemporal == rhs.gtaoTemporal && lhs.valid == rhs.valid;
}

[[nodiscard]] constexpr PresentationAAUnsupportedReason msaaUnsupportedReason(
    AntiAliasingMode mode,
    const PresentationAAGpuCapabilities &capabilities) noexcept {
  const bool sample8 = mode == AntiAliasingMode::MSAA8x;
  if (!(sample8 ? capabilities.sample8Color : capabilities.sample4Color)) {
    return sample8 ? PresentationAAUnsupportedReason::Sample8Color
                   : PresentationAAUnsupportedReason::Sample4Color;
  }
  if (!(sample8 ? capabilities.sample8Depth : capabilities.sample4Depth)) {
    return sample8 ? PresentationAAUnsupportedReason::Sample8Depth
                   : PresentationAAUnsupportedReason::Sample4Depth;
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
  case PresentationAAUnsupportedReason::Sample8Color:
    return "sample8_color_unsupported";
  case PresentationAAUnsupportedReason::Sample8Depth:
    return "sample8_depth_unsupported";
  case PresentationAAUnsupportedReason::DepthResolveMin:
    return "depth_resolve_min_unsupported";
  case PresentationAAUnsupportedReason::AlphaToCoverage:
    return "alpha_to_coverage_unsupported";
  }
  return "unknown";
}

[[nodiscard]] inline Result<PresentationAAPlan, std::string>
buildPresentationAAPlan(
    const ResolvedRenderSettings &settings,
    const PresentationAAProviderCapabilities &providerCapabilities = {},
    const PresentationAAGpuCapabilities &gpuCapabilities = {}) {
  PresentationAAPlan plan{};
  const AntiAliasingMode mode = settings->antiAliasing.mode;
  const TemporalReconstructionProvider temporalProvider =
      settings->antiAliasing.temporalProvider;
  switch (mode) {
  case AntiAliasingMode::None:
    break;
  case AntiAliasingMode::SpatialFallback:
    plan.spatialCleanup = SpatialCleanupPoint::PreComposition;
    break;
  case AntiAliasingMode::MSAA4x:
  case AntiAliasingMode::MSAA8x:
    if (const PresentationAAUnsupportedReason unsupportedReason =
            msaaUnsupportedReason(mode, gpuCapabilities);
        unsupportedReason != PresentationAAUnsupportedReason::None) {
      return Result<PresentationAAPlan, std::string>::makeError(
          std::string(mode == AntiAliasingMode::MSAA8x ? "MSAA8x" : "MSAA4x") +
          " unsupported: " +
          std::string(presentationAAUnsupportedReasonName(unsupportedReason)));
    }
    plan.coverage = mode == AntiAliasingMode::MSAA8x ? CoverageMode::Sample8
                                                     : CoverageMode::Sample4;
    plan.alphaCoverage = AlphaCoveragePolicy::ThresholdedAlphaToCoverage;
    plan.transparency = TransparencyAAPolicy::SingleSamplePostResolve;
    plan.sampleShadingSupported = gpuCapabilities.sampleRateShading;
    plan.sampleShadingEnabled = false;
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
    plan.jitterScene = settings->antiAliasing.debug.jitterEnabled;
    plan.needsMotion = true;
    plan.needsReactiveMask = providerCapabilities.reactiveMask;
    plan.needsMotionClass = true;
    if (temporalProvider == TemporalReconstructionProvider::Legacy &&
        settings->antiAliasing.temporalTuning.spatialPostTaaCleanup) {
      plan.spatialCleanup = SpatialCleanupPoint::PreComposition;
    }
    break;
  }
  plan.gtaoTemporal = settings->ambientOcclusion.active &&
                      settings->ambientOcclusion.temporalAccumulation;
  plan.needsMotion = plan.needsMotion || plan.gtaoTemporal;
  plan.needsReactiveMask = plan.needsReactiveMask || plan.gtaoTemporal;
  plan.postAA = resolvePostAAPlan(settings->antiAliasing, plan.coverage);
  plan.valid = true;
  return Result<PresentationAAPlan, std::string>::makeResult(plan);
}

[[nodiscard]] constexpr bool
usesTemporalColorReconstruction(const PresentationAAPlan &plan) noexcept {
  return plan.reconstruction != ColorReconstruction::Off;
}

[[nodiscard]] constexpr bool
hasTemporalCameraContinuity(const CameraFrameState &camera) noexcept {
  return camera.cameraContinuityValid || camera.historyValid;
}

} // namespace nuri
