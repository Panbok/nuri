#include "nuri/pch.h"

#include "nuri/gfx/frame/external_temporal_provider.h"

namespace nuri {
namespace {

#ifndef NURI_WITH_FSR31
#define NURI_WITH_FSR31 0
#endif

#ifndef NURI_FSR31_DEPENDENCY_PRESENT
#define NURI_FSR31_DEPENDENCY_PRESENT 0
#endif

class FidelityFxExternalTemporalProvider final
    : public ExternalTemporalProvider {
public:
  explicit FidelityFxExternalTemporalProvider(
      const ExternalTemporalProviderCreateDesc &desc)
      : desc_(desc) {
    refreshProbe();
  }

  ~FidelityFxExternalTemporalProvider() override {
    if (desc_.backend != nullptr) {
      desc_.backend->reset();
    }
  }

  [[nodiscard]] ExternalTemporalProviderProbe probe() const noexcept override {
    return probe_;
  }

  [[nodiscard]] ExternalTemporalProviderCapabilities
  capabilities() const noexcept override {
    return fidelityFxFsr31Capabilities(probe_);
  }

  [[nodiscard]] Result<ExternalTemporalProviderFramePlan, std::string>
  prepareFrame(const ExternalTemporalProviderPrepareDesc &prepareDesc) override {
    if (prepareDesc.renderExtent.x == 0u || prepareDesc.renderExtent.y == 0u ||
        prepareDesc.outputExtent.x == 0u ||
        prepareDesc.outputExtent.y == 0u) {
      return Result<ExternalTemporalProviderFramePlan, std::string>::makeError(
          "external temporal provider prepare extents must be nonzero");
    }
    if (desc_.backend != nullptr &&
        probe_.status != ExternalTemporalProviderStatus::BuildDisabled &&
        probe_.status != ExternalTemporalProviderStatus::DependencyMissing &&
        probe_.status != ExternalTemporalProviderStatus::BackendUnavailable) {
      auto backendPlan = desc_.backend->prepareFrame(prepareDesc);
      if (backendPlan.hasError()) {
        return Result<ExternalTemporalProviderFramePlan, std::string>::makeError(
            backendPlan.error());
      }
      refreshProbe(backendPlan.value());
      if (probe_.status == ExternalTemporalProviderStatus::Ready) {
        return Result<ExternalTemporalProviderFramePlan, std::string>::makeResult({
            .status = probe_.status,
            .jitterPixels = backendPlan.value().jitterPixels,
            .outputExtent = prepareDesc.outputExtent,
            .jitterPhaseIndex = backendPlan.value().jitterPhaseIndex,
            .jitterPhaseCount = backendPlan.value().jitterPhaseCount,
            .configurationEpoch = prepareDesc.configurationEpoch,
            .reconstructionActive = true,
            .sceneJitterActive = true,
            .requiresDepth = true,
            .requiresMotion = true,
            .requiresReactiveMask = true,
            .requiresCompositionMask = true,
            .requiresExposure = false,
        });
      }
    }
    return Result<ExternalTemporalProviderFramePlan, std::string>::makeResult({
        .status = probe_.status,
        .outputExtent = prepareDesc.outputExtent,
        .configurationEpoch = prepareDesc.configurationEpoch,
    });
  }

  [[nodiscard]] Result<TextureHandle, std::string>
  execute(RecordingContextHandle recordingContext,
          const ExternalTemporalProviderExecuteDesc &executeDesc) override {
    if (probe_.status != ExternalTemporalProviderStatus::Ready ||
        desc_.backend == nullptr) {
      return Result<TextureHandle, std::string>::makeError(
          std::string("external temporal provider unavailable: ") +
          std::string(externalTemporalProviderStatusMessage(probe_.status)));
    }
    auto validation =
        validateExternalTemporalExecuteDesc(executeDesc, capabilities());
    if (validation.hasError()) {
      return Result<TextureHandle, std::string>::makeError(validation.error());
    }
    auto recorded =
        desc_.backend->recordDispatch(recordingContext, executeDesc);
    if (recorded.hasError()) {
      return Result<TextureHandle, std::string>::makeError(recorded.error());
    }
    return Result<TextureHandle, std::string>::makeResult(executeDesc.output);
  }

private:
  void refreshProbe() {
    ExternalTemporalProviderBackendProbe backend{};
    if (desc_.backend != nullptr) {
      backend = desc_.backend->probe();
    }
    probe_ = probeFidelityFxFsr31({
        .buildRequested = desc_.buildRequested,
        .dependencyPresent = desc_.dependencyPresent,
        .backendCompiled = backend.dispatchWired,
        .runtimeLoaded = backend.runtimeLoaded,
        .reportedProviderVersion = backend.reportedProviderVersion,
    });
  }

  void refreshProbe(const ExternalTemporalProviderBackendFramePlan &plan) {
    ExternalTemporalProviderBackendProbe backend{};
    if (desc_.backend != nullptr) {
      backend = desc_.backend->probe();
    }
    probe_ = probeFidelityFxFsr31({
        .buildRequested = desc_.buildRequested,
        .dependencyPresent = desc_.dependencyPresent,
        .backendCompiled = backend.dispatchWired,
        .runtimeLoaded = plan.runtimeLoaded,
        .reportedProviderVersion = plan.reportedProviderVersion,
    });
  }

  ExternalTemporalProviderCreateDesc desc_{};
  ExternalTemporalProviderProbe probe_{};
};

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

std::unique_ptr<ExternalTemporalProvider> createExternalTemporalProvider() {
  return createExternalTemporalProvider(nullptr);
}

std::unique_ptr<ExternalTemporalProvider> createExternalTemporalProvider(
    ExternalTemporalProviderBackend *backend) {
  return createExternalTemporalProvider({
      .backend = backend,
      .buildRequested = NURI_WITH_FSR31 != 0,
      .dependencyPresent = NURI_FSR31_DEPENDENCY_PRESENT != 0,
  });
}

std::unique_ptr<ExternalTemporalProvider> createExternalTemporalProvider(
    const ExternalTemporalProviderCreateDesc &desc) {
  return std::make_unique<FidelityFxExternalTemporalProvider>(desc);
}

} // namespace nuri
