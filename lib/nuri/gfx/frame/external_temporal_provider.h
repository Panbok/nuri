#pragma once
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/frame/history_registry.h"
#include "nuri/gfx/gpu_types.h"
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <string_view>
namespace nuri {

inline constexpr std::string_view kFidelityFxSdkTag = "v1.1.4";
inline constexpr std::string_view kFidelityFxSdkRevision =
    "c6efa6bf7f2027b3ec94f28578bb5965eabb9e55";
inline constexpr std::string_view kFidelityFxSdkVersion = "1.1.4";
inline constexpr std::string_view kFidelityFxFsrUpscalerVersion = "3.1.4";
inline constexpr std::string_view kFidelityFxLicenseSpdx = "MIT";

enum class ExternalTemporalProviderStatus : uint8_t {
  BuildDisabled = 0u,
  DependencyMissing,
  BackendUnavailable,
  RuntimeUnavailable,
  VersionMismatch,
  Ready,
};

struct ExternalTemporalProviderProbeDesc {
  bool buildRequested = false;
  bool dependencyPresent = false;
  bool backendCompiled = false;
  bool runtimeLoaded = false;
  std::string_view reportedProviderVersion{};
};

struct ExternalTemporalProviderProbe {
  ExternalTemporalProviderStatus status =
      ExternalTemporalProviderStatus::BuildDisabled;
  bool available = false;
};

[[nodiscard]] NURI_API ExternalTemporalProviderProbe
probeFidelityFxFsr31(const ExternalTemporalProviderProbeDesc &desc) noexcept;

[[nodiscard]] NURI_API std::string_view externalTemporalProviderStatusMessage(
    ExternalTemporalProviderStatus status) noexcept;

struct ExternalTemporalProviderCapabilities {
  bool available = false;
  bool nativeResolutionAA = false;
  bool explicitMotionVectors = false;
  bool reactiveMask = false;
  bool compositionMask = false;
  bool exposure = false;
  bool dynamicResolution = false;
  bool explicitMotionValidity = false;
};

[[nodiscard]] NURI_API ExternalTemporalProviderCapabilities
fidelityFxFsr31Capabilities(
    const ExternalTemporalProviderProbe &probe) noexcept;

enum class ExternalTemporalDepthConvention : uint8_t {
  FiniteStandard = 0u,
  FiniteInverted,
  InfiniteStandard,
  InfiniteInverted,
};

enum class ExternalTemporalMotionConvention : uint8_t {
  CurrentToPreviousNormalizedUv = 0u,
};

enum class ExternalTemporalColorConvention : uint8_t {
  LinearHdr = 0u,
  LinearHdrPreExposed,
};

struct ExternalTemporalProviderPrepareDesc {
  glm::uvec2 renderExtent{0u};
  glm::uvec2 outputExtent{0u};
  uint64_t renderedFrameSerial = 0u;
  uint64_t configurationEpoch = 0u;
  bool reset = false;
};

struct ExternalTemporalProviderFramePlan {
  ExternalTemporalProviderStatus status =
      ExternalTemporalProviderStatus::BuildDisabled;
  glm::vec2 jitterPixels{0.0f};
  glm::uvec2 outputExtent{0u};
  uint32_t jitterPhaseIndex = 0u;
  uint32_t jitterPhaseCount = 0u;
  uint64_t configurationEpoch = 0u;
  bool reconstructionActive = false;
  bool sceneJitterActive = false;
  bool requiresDepth = false;
  bool requiresMotion = false;
  bool requiresReactiveMask = false;
  bool requiresCompositionMask = false;
  bool requiresExposure = false;
};

struct ExternalTemporalProviderExecuteDesc {
  TextureHandle sceneColor{};
  TextureHandle sceneDepth{};
  TextureHandle motionVectors{};
  TextureHandle reactiveMask{};
  TextureHandle compositionMask{};
  TextureHandle exposure{};
  TextureHandle output{};
  glm::uvec2 renderExtent{0u};
  glm::uvec2 outputExtent{0u};
  glm::vec2 jitterPixels{0.0f};
  ExternalTemporalDepthConvention depthConvention =
      ExternalTemporalDepthConvention::FiniteStandard;
  ExternalTemporalMotionConvention motionConvention =
      ExternalTemporalMotionConvention::CurrentToPreviousNormalizedUv;
  ExternalTemporalColorConvention colorConvention =
      ExternalTemporalColorConvention::LinearHdr;
  HistoryLease history{};
  uint64_t renderedFrameSerial = 0u;
  float frameTimeDeltaMilliseconds = 0.0f;
  float preExposure = 1.0f;
  float cameraNear = 0.1f;
  float cameraFar = 1000.0f;
  float cameraVerticalFovRadians = 1.0f;
  float invalidMotionCoveragePercent = 0.0f;
  float sharpening = 0.0f;
  bool reset = false;
};

struct ExternalTemporalProviderBackendProbe {
  bool dispatchWired = false;
  bool runtimeLoaded = false;
  std::string reportedProviderVersion{};
};

struct ExternalTemporalProviderBackendFramePlan {
  bool runtimeLoaded = false;
  std::string reportedProviderVersion{};
  glm::vec2 jitterPixels{0.0f};
  uint32_t jitterPhaseIndex = 0u;
  uint32_t jitterPhaseCount = 0u;
};

class NURI_API ExternalTemporalProviderBackend {
public:
  virtual ~ExternalTemporalProviderBackend() = default;
  [[nodiscard]] virtual ExternalTemporalProviderBackendProbe probe() const = 0;
  [[nodiscard]] virtual Result<ExternalTemporalProviderBackendFramePlan,
                               std::string>
  prepareFrame(const ExternalTemporalProviderPrepareDesc &desc) = 0;
  [[nodiscard]] virtual Result<bool, std::string>
  recordDispatch(RecordingContextHandle recordingContext,
                 const ExternalTemporalProviderExecuteDesc &desc) = 0;
  virtual void reset() noexcept = 0;
};

[[nodiscard]] NURI_API Result<bool, std::string>
validateExternalTemporalExecuteDesc(
    const ExternalTemporalProviderExecuteDesc &desc,
    const ExternalTemporalProviderCapabilities &capabilities);

class NURI_API ExternalTemporalProvider {
public:
  virtual ~ExternalTemporalProvider() = default;
  ExternalTemporalProvider(const ExternalTemporalProvider &) = delete;
  ExternalTemporalProvider &
  operator=(const ExternalTemporalProvider &) = delete;
  [[nodiscard]] virtual ExternalTemporalProviderProbe
  probe() const noexcept = 0;
  [[nodiscard]] virtual ExternalTemporalProviderCapabilities
  capabilities() const noexcept = 0;
  [[nodiscard]] virtual Result<ExternalTemporalProviderFramePlan, std::string>
  prepareFrame(const ExternalTemporalProviderPrepareDesc &desc) = 0;
  [[nodiscard]] virtual Result<TextureHandle, std::string>
  execute(RecordingContextHandle recordingContext,
          const ExternalTemporalProviderExecuteDesc &desc) = 0;

protected:
  ExternalTemporalProvider() = default;
};

struct ExternalTemporalDispatchItem {
  ExternalTemporalProvider *provider = nullptr;
  ExternalTemporalProviderExecuteDesc execute{};
};

struct ExternalTemporalProviderCreateDesc {
  ExternalTemporalProviderBackend *backend = nullptr;
  bool buildRequested = false;
  bool dependencyPresent = false;
};

[[nodiscard]] NURI_API std::unique_ptr<ExternalTemporalProvider>
createExternalTemporalProvider();
[[nodiscard]] NURI_API std::unique_ptr<ExternalTemporalProvider>
createExternalTemporalProvider(ExternalTemporalProviderBackend *backend);
[[nodiscard]] NURI_API std::unique_ptr<ExternalTemporalProvider>
createExternalTemporalProvider(const ExternalTemporalProviderCreateDesc &desc);

} // namespace nuri
