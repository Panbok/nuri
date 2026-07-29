#include "nuri/gfx/frame/external_temporal_provider.h"
namespace nuri {
namespace {
[[nodiscard]] bool finitePositive(float value) noexcept {
  return std::isfinite(value) && value > 0.0f;
}
} // namespace

ExternalTemporalProviderProbe
probeFidelityFxFsr31(const ExternalTemporalProviderProbeDesc &desc) noexcept {
  if (!desc.buildRequested) {
    return {.status = ExternalTemporalProviderStatus::BuildDisabled};
  }
  if (!desc.dependencyPresent) {
    return {.status = ExternalTemporalProviderStatus::DependencyMissing};
  }
  if (!desc.backendCompiled) {
    return {.status = ExternalTemporalProviderStatus::BackendUnavailable};
  }
  if (!desc.runtimeLoaded) {
    return {.status = ExternalTemporalProviderStatus::RuntimeUnavailable};
  }
  if (desc.reportedProviderVersion != kFidelityFxFsrUpscalerVersion) {
    return {.status = ExternalTemporalProviderStatus::VersionMismatch};
  }
  return {.status = ExternalTemporalProviderStatus::Ready, .available = true};
}

std::string_view externalTemporalProviderStatusMessage(
    ExternalTemporalProviderStatus status) noexcept {
  switch (status) {
  case ExternalTemporalProviderStatus::BuildDisabled:
    return "NURI_WITH_FSR31 is disabled";
  case ExternalTemporalProviderStatus::DependencyMissing:
    return "the pinned FidelityFX SDK dependency is missing";
  case ExternalTemporalProviderStatus::BackendUnavailable:
    return "no FidelityFX backend adapter is compiled";
  case ExternalTemporalProviderStatus::RuntimeUnavailable:
    return "the FidelityFX runtime is unavailable";
  case ExternalTemporalProviderStatus::VersionMismatch:
    return "the FidelityFX provider version does not match the pinned version";
  case ExternalTemporalProviderStatus::Ready:
    return "ready";
  }
  return "unknown external temporal provider status";
}

ExternalTemporalProviderCapabilities fidelityFxFsr31Capabilities(
    const ExternalTemporalProviderProbe &probe) noexcept {
  if (!probe.available ||
      probe.status != ExternalTemporalProviderStatus::Ready) {
    return {};
  }
  return {
      .available = true,
      .nativeResolutionAA = true,
      .explicitMotionVectors = true,
      .reactiveMask = true,
      .compositionMask = true,
      .exposure = true,
      .dynamicResolution = true,
      .explicitMotionValidity = false,
  };
}

Result<bool, std::string> validateExternalTemporalExecuteDesc(
    const ExternalTemporalProviderExecuteDesc &desc,
    const ExternalTemporalProviderCapabilities &capabilities) {
  if (!capabilities.available || !capabilities.nativeResolutionAA) {
    return Result<bool, std::string>::makeError(
        "external temporal provider is unavailable");
  }
  if (!nuri::isValid(desc.sceneColor) || !nuri::isValid(desc.sceneDepth) ||
      !nuri::isValid(desc.motionVectors) || !nuri::isValid(desc.output)) {
    return Result<bool, std::string>::makeError(
        "external temporal provider requires color, depth, motion, and output "
        "textures");
  }
  if (desc.sceneColor.index == desc.output.index &&
      desc.sceneColor.generation == desc.output.generation) {
    return Result<bool, std::string>::makeError(
        "external temporal provider input and output textures must differ");
  }
  if (capabilities.reactiveMask && !nuri::isValid(desc.reactiveMask)) {
    return Result<bool, std::string>::makeError(
        "external temporal provider requires a reactive mask");
  }
  if (capabilities.compositionMask && !nuri::isValid(desc.compositionMask)) {
    return Result<bool, std::string>::makeError(
        "external temporal provider requires a composition mask");
  }
  if (desc.renderExtent.x == 0u || desc.renderExtent.y == 0u ||
      desc.outputExtent.x == 0u || desc.outputExtent.y == 0u) {
    return Result<bool, std::string>::makeError(
        "external temporal provider extents must be nonzero");
  }
  if (!finitePositive(desc.frameTimeDeltaMilliseconds) ||
      !finitePositive(desc.preExposure) || !finitePositive(desc.cameraNear) ||
      !finitePositive(desc.cameraFar) || desc.cameraFar <= desc.cameraNear ||
      !finitePositive(desc.cameraVerticalFovRadians)) {
    return Result<bool, std::string>::makeError(
        "external temporal provider camera and frame values are invalid");
  }
  if (!std::isfinite(desc.invalidMotionCoveragePercent) ||
      desc.invalidMotionCoveragePercent != 0.0f) {
    return Result<bool, std::string>::makeError(
        "external temporal provider cannot safely consume invalid motion "
        "coverage");
  }
  if (!std::isfinite(desc.sharpening) || desc.sharpening < 0.0f ||
      desc.sharpening > 1.0f) {
    return Result<bool, std::string>::makeError(
        "external temporal provider sharpening must be within [0, 1]");
  }
  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri
