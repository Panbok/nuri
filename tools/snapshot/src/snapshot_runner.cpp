#include "nuri/tools/snapshot/snapshot_runner.h"

#include "nuri/core/log.h"
#include "nuri/core/runtime_config.h"
#include "nuri/core/window.h"
#include "nuri/gfx/frame/temporal_frame_service.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/default_render_pipeline.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/gfx/renderer.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/resources/scene_importer.h"
#include "nuri/scene/camera.h"
#include "nuri/scene/render_scene.h"
#include "nuri/tools/core/baseline_profile.h"
#include "nuri/tools/core/case_catalog.h"
#include "nuri/tools/core/fingerprint.h"
#include "nuri/tools/core/result_envelope_v2.h"
#include "nuri/tools/core/run_workspace.h"
#include "nuri/tools/snapshot/snapshot_baseline.h"
#include "nuri/tools/snapshot/snapshot_capture_artifacts.h"
#include "nuri/tools/snapshot/snapshot_capture_point.h"
#include "nuri/tools/snapshot/snapshot_html_report.h"
#include "nuri/tools/snapshot/snapshot_image.h"
#include "nuri/tools/snapshot/snapshot_manifest.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <memory_resource>
#include <optional>
#include <sstream>
#include <utility>

#if defined(_WIN32)
#include <stdlib.h>
#endif

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace nuri::tools::snapshot {
namespace {

[[nodiscard]] std::optional<std::string>
snapshotEnvironmentFingerprint(const SnapshotEnvironment &environment) {
  using nuri::tools::core::FingerprintField;
  auto fingerprint = nuri::tools::core::makeSha256Fingerprint({
      FingerprintField{"os.name", environment.osName},
      FingerprintField{"os.version", environment.osVersion},
      FingerprintField{"cpu.name", environment.cpuName},
      FingerprintField{"cpu.threads",
                       std::to_string(environment.cpuLogicalThreadCount)},
      FingerprintField{"gpu.backend", environment.gpuBackend},
      FingerprintField{"gpu.backendSource", environment.gpuBackendSource},
      FingerprintField{"gpu.name", environment.gpuDeviceName},
      FingerprintField{"gpu.vendor", std::to_string(environment.gpuVendorId)},
      FingerprintField{"gpu.device", std::to_string(environment.gpuDeviceId)},
      FingerprintField{"gpu.driver", environment.gpuDriverVersion},
      FingerprintField{"present.mode", environment.resolvedPresentMode},
      FingerprintField{"window.mode", environment.resolvedWindowMode},
      FingerprintField{"window.visible",
                       environment.windowVisible ? "true" : "false"},
      FingerprintField{"build.type", environment.buildType},
      FingerprintField{"build.profile", environment.cmakeToolProfile},
      FingerprintField{"build.features", environment.vcpkgManifestFeatures},
      FingerprintField{"build.shared",
                       environment.buildShared ? "true" : "false"},
      FingerprintField{"build.asserts",
                       environment.assertsEnabled ? "true" : "false"},
      FingerprintField{"build.devChecks",
                       environment.devChecks ? "true" : "false"},
      FingerprintField{"profiling.cpu",
                       environment.tracyEnabled ? "true" : "false"},
      FingerprintField{"profiling.gpu",
                       environment.tracyGpuEnabled ? "true" : "false"},
  });
  return fingerprint.hasError()
             ? std::nullopt
             : std::optional<std::string>{std::move(fingerprint.value())};
}

[[nodiscard]] std::optional<std::string>
snapshotWorkloadFingerprint(const SnapshotCase &snapshotCase) {
  using nuri::tools::core::FingerprintField;
  std::string manifestDigest;
  if (!snapshotCase.manifestPath.empty() &&
      std::filesystem::is_regular_file(snapshotCase.manifestPath)) {
    auto digest =
        nuri::tools::core::makeSha256FileFingerprint(snapshotCase.manifestPath);
    if (!digest.hasError()) {
      manifestDigest = std::move(digest.value());
    }
  }
  std::vector<FingerprintField> fields{
      {"case.id", snapshotCase.id},
      {"case.suite", snapshotCase.suite},
      {"manifest", std::move(manifestDigest)},
      {"scene.kind", snapshotCase.scene.kind},
      {"scene.content", snapshotCase.scene.contentHash},
      {"resolution.width", std::to_string(snapshotCase.resolution[0])},
      {"resolution.height", std::to_string(snapshotCase.resolution[1])},
      {"fixedDelta", std::format("{:.17g}", snapshotCase.fixedDeltaSeconds)},
      {"frames.warmup", std::to_string(snapshotCase.warmupFrames)},
      {"frames.capture", std::to_string(snapshotCase.captureFrame)},
  };
  for (const SnapshotCaptureTarget &capture : snapshotCase.captures) {
    fields.push_back({"capture." + capture.name + ".profile", capture.profile});
    fields.push_back({"capture." + capture.name + ".required",
                      capture.required ? "true" : "false"});
  }
  auto fingerprint =
      nuri::tools::core::makeSha256Fingerprint(std::move(fields));
  return fingerprint.hasError()
             ? std::nullopt
             : std::optional<std::string>{std::move(fingerprint.value())};
}

[[nodiscard]] std::optional<std::string> aggregateSnapshotFingerprint(
    std::string_view scope,
    const std::vector<std::pair<std::string, std::string>> &children) {
  std::vector<nuri::tools::core::FingerprintField> fields;
  fields.reserve(children.size() + 1u);
  fields.push_back({"scope", std::string(scope)});
  for (const auto &[id, fingerprint] : children) {
    fields.push_back({"child." + id, fingerprint});
  }
  auto aggregate = nuri::tools::core::makeSha256Fingerprint(std::move(fields));
  return aggregate.hasError()
             ? std::nullopt
             : std::optional<std::string>{std::move(aggregate.value())};
}

void evaluateSnapshotBaselineProfile(SnapshotReport &report) {
  auto profile = nuri::tools::core::loadBaselineProfile(
      snapshotRepoRoot() / "tools" / "profiles", report.baselineProfile);
  if (profile.hasError()) {
    report.baselineProfileCompatible = false;
    report.baselineProfileIncompatibilityReasons = {profile.error()};
    return;
  }
  const std::string profiling =
      report.environment.tracyGpuEnabled
          ? "cpu-gpu"
          : (report.environment.tracyEnabled ? "cpu" : "off");
  const auto compatibility = nuri::tools::core::evaluateBaselineProfile(
      profile.value(),
      nuri::tools::core::BaselineProfileObservedEnvironment{
          .os = report.environment.osName,
          .backend = report.environment.gpuBackend,
          .backendSource = report.environment.gpuBackendSource,
          .windowMode = report.environment.resolvedWindowMode,
          .windowVisible = report.environment.windowVisible,
          .gpuVendorId = report.environment.gpuVendorId,
          .gpuDeviceId = report.environment.gpuDeviceId,
          .driver = report.environment.gpuDriverVersion,
          .presentMode = report.environment.resolvedPresentMode,
          .profiling = profiling,
          .devChecks = report.environment.devChecks,
          .dirtyTree = report.environment.dirty,
      });
  report.baselineProfileCompatible = compatibility.compatible;
  report.baselineProfileIncompatibilityReasons = compatibility.reasons;
}

[[nodiscard]] Result<bool, std::string>
validateRunInputs(const SnapshotCase &snapshotCase,
                  const SnapshotRunOptions &options) {
  if (options.windowMode != "visible" && options.windowMode != "hidden" &&
      options.windowMode != "headless") {
    return Result<bool, std::string>::makeError(
        "windowMode must be visible, hidden, or headless");
  }
  for (const auto &[value, field] :
       {std::pair{std::string_view(snapshotCase.id),
                  std::string_view("case id")},
        std::pair{std::string_view(snapshotCase.suite),
                  std::string_view("suite")},
        std::pair{std::string_view(options.baselineProfile),
                  std::string_view("baseline profile")}}) {
    auto valid = validateSnapshotIdentifier(value, field);
    if (valid.hasError()) {
      return Result<bool, std::string>::makeError(valid.error());
    }
  }
  for (const SnapshotCaptureTarget &capture : snapshotCase.captures) {
    auto valid = validateSnapshotIdentifier(capture.name, "capture target");
    if (valid.hasError()) {
      return Result<bool, std::string>::makeError(valid.error());
    }
    const SnapshotCaptureCatalogEntry *catalog =
        findSnapshotCaptureCatalogEntry(capture.name);
    if (catalog == nullptr ||
        !isBuiltinSnapshotCompareProfile(capture.profile) ||
        !snapshotCompareProfileSupportsKind(capture.profile, catalog->kind)) {
      return Result<bool, std::string>::makeError(
          "invalid capture/profile descriptor for '" + capture.name + "'");
    }
  }
  auto profile = nuri::tools::core::loadBaselineProfile(
      snapshotRepoRoot() / "tools" / "profiles", options.baselineProfile);
  if (profile.hasError()) {
    return Result<bool, std::string>::makeError(profile.error());
  }
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] int outcomePrecedence(SnapshotExitCode code) noexcept {
  switch (code) {
  case SnapshotExitCode::RuntimeError:
    return 5;
  case SnapshotExitCode::InvalidInput:
    return 4;
  case SnapshotExitCode::EnvironmentUnavailable:
    return 3;
  case SnapshotExitCode::MissingBaseline:
    return 2;
  case SnapshotExitCode::VisualMismatch:
    return 1;
  case SnapshotExitCode::Success:
    return 0;
  }
  return 5;
}

[[nodiscard]] std::string_view outcomeStatus(SnapshotExitCode code) noexcept {
  switch (code) {
  case SnapshotExitCode::Success:
    return "pass";
  case SnapshotExitCode::VisualMismatch:
    return "fail";
  case SnapshotExitCode::InvalidInput:
    return "invalid";
  case SnapshotExitCode::EnvironmentUnavailable:
    return "unavailable";
  case SnapshotExitCode::RuntimeError:
    return "error";
  case SnapshotExitCode::MissingBaseline:
    return "missing_baseline";
  }
  return "error";
}

[[nodiscard]] nuri::tools::core::ToolOutcome
toolOutcome(SnapshotExitCode code, bool investigative = false) noexcept {
  using nuri::tools::core::ToolOutcome;
  if (investigative && code == SnapshotExitCode::Success) {
    return ToolOutcome::Investigative;
  }
  switch (code) {
  case SnapshotExitCode::Success:
    return ToolOutcome::Pass;
  case SnapshotExitCode::VisualMismatch:
    return ToolOutcome::Failure;
  case SnapshotExitCode::InvalidInput:
    return ToolOutcome::Invalid;
  case SnapshotExitCode::EnvironmentUnavailable:
    return ToolOutcome::EnvironmentUnavailable;
  case SnapshotExitCode::RuntimeError:
    return ToolOutcome::RuntimeError;
  case SnapshotExitCode::MissingBaseline:
    return ToolOutcome::MissingBaseline;
  }
  return ToolOutcome::RuntimeError;
}

[[nodiscard]] std::string jsonEscape(std::string_view value) {
  std::ostringstream out;
  for (const unsigned char c : value) {
    switch (c) {
    case '"':
      out << "\\\"";
      break;
    case '\\':
      out << "\\\\";
      break;
    case '\n':
      out << "\\n";
      break;
    case '\r':
      out << "\\r";
      break;
    case '\t':
      out << "\\t";
      break;
    default:
      if (c < 0x20u) {
        out << '?';
      } else {
        out << static_cast<char>(c);
      }
      break;
    }
  }
  return out.str();
}

class ScopedEnvVar final {
public:
  ScopedEnvVar(std::string name, std::string value)
      : name_(std::move(name)), oldValue_(readProcessEnvironment(name_)),
        hadOldValue_(!oldValue_.empty()) {
    set(value);
  }
  ~ScopedEnvVar() {
    if (hadOldValue_) {
      set(oldValue_);
    } else {
      unset();
    }
  }
  ScopedEnvVar(const ScopedEnvVar &) = delete;
  ScopedEnvVar &operator=(const ScopedEnvVar &) = delete;

private:
  void set(const std::string &value) {
#if defined(_WIN32)
    _putenv_s(name_.c_str(), value.c_str());
#else
    setenv(name_.c_str(), value.c_str(), 1);
#endif
  }
  void unset() {
#if defined(_WIN32)
    _putenv_s(name_.c_str(), "");
#else
    unsetenv(name_.c_str());
#endif
  }
  std::string name_;
  std::string oldValue_;
  bool hadOldValue_ = false;
};

class SnapshotLogGuard final {
public:
  explicit SnapshotLogGuard(const std::filesystem::path &path) {
    std::filesystem::create_directories(path.parent_path());
    LogConfig config{};
    config.filePath = path.string();
    config.logLevel = LogLevel::Info;
    config.consoleLevel = LogLevel::Warning;
    config.threadNames = false;
    Log::initialize(config);
  }
  ~SnapshotLogGuard() { Log::shutdown(); }
  SnapshotLogGuard(const SnapshotLogGuard &) = delete;
  SnapshotLogGuard &operator=(const SnapshotLogGuard &) = delete;
};

[[nodiscard]] GPUBackendPreference backendPreference(std::string_view backend) {
  if (backend == "lvk") {
    return GPUBackendPreference::Lvk;
  }
  if (backend == "nvrhi") {
    return GPUBackendPreference::Nvrhi;
  }
  return GPUBackendPreference::Default;
}

[[nodiscard]] std::string resolveBackendName(const SnapshotCase &snapshotCase,
                                             std::string &source) {
  const std::string envBackend = readProcessEnvironment("NURI_GPU_BACKEND");
  if (snapshotCase.backend != "default") {
    source = "manifest";
    return snapshotCase.backend;
  }
  if (!envBackend.empty()) {
    source = "NURI_GPU_BACKEND";
    return envBackend;
  }
  source = "default";
  return "nvrhi";
}

[[nodiscard]] std::string resolvePresentMode(const SnapshotCase &snapshotCase,
                                             std::string &source) {
  const std::string envPresent = readProcessEnvironment("NURI_PRESENT_MODE");
  if (snapshotCase.presentMode != "default") {
    source = "manifest";
    return snapshotCase.presentMode;
  }
  if (!envPresent.empty()) {
    source = "NURI_PRESENT_MODE";
    return envPresent;
  }
  source = "default";
  return "default";
}

[[nodiscard]] Result<bool, SnapshotExitCode>
checkRequirements(const SnapshotCase &snapshotCase, std::string_view backend,
                  std::string_view windowMode,
                  std::vector<std::string> &warnings, std::string &message) {
  if (windowMode == "visible" &&
      !snapshotCase.requirements.allowVisibleWindow) {
    message = "case does not permit visible-window execution";
    return Result<bool, SnapshotExitCode>::makeError(
        SnapshotExitCode::EnvironmentUnavailable);
  }
  if (!snapshotCase.requirements.backends.empty()) {
    bool supported = false;
    for (const std::string &allowed : snapshotCase.requirements.backends) {
      supported = supported || allowed == backend || allowed == "default";
    }
    if (!supported) {
      message = "backend '" + std::string(backend) +
                "' is not allowed by case requirements";
      return Result<bool, SnapshotExitCode>::makeError(
          SnapshotExitCode::EnvironmentUnavailable);
    }
  }
  for (const std::string &asset : snapshotCase.requirements.assets) {
    const size_t colon = asset.find(':');
    if (colon == std::string::npos) {
      message = "invalid asset requirement '" + asset + "'";
      return Result<bool, SnapshotExitCode>::makeError(
          SnapshotExitCode::InvalidInput);
    }
    auto path =
        resolveSnapshotPath(asset.substr(0, colon), asset.substr(colon + 1u));
    if (path.hasError()) {
      message = path.error();
      return Result<bool, SnapshotExitCode>::makeError(
          SnapshotExitCode::EnvironmentUnavailable);
    }
    if (!std::filesystem::exists(path.value())) {
      message = "missing required asset: " + path.value().string();
      return Result<bool, SnapshotExitCode>::makeError(
          SnapshotExitCode::EnvironmentUnavailable);
    }
  }
  (void)warnings;
  return Result<bool, SnapshotExitCode>::makeResult(true);
}

[[nodiscard]] Result<bool, std::string>
populateTransmissionTransparencyScene(const SnapshotCase &snapshotCase,
                                      Renderer &renderer, RenderScene &scene) {
  if (snapshotCase.scene.generator !=
      "nuri.procedural.transmission_transparency.v1") {
    return Result<bool, std::string>::makeResult(true);
  }

  auto planePath = resolveSnapshotPath("modelsRoot", "common/flat_plane.obj");
  if (planePath.hasError()) {
    return Result<bool, std::string>::makeError(planePath.error());
  }
  auto modelResult = renderer.resources().acquireModel(ModelRequest{
      .path = planePath.value().string(),
      .debugName = "snapshot_transmission_flat_plane",
  });
  if (modelResult.hasError()) {
    return Result<bool, std::string>::makeError(modelResult.error());
  }

  auto acquireMaterial =
      [&](std::string_view name, const glm::vec4 &color,
          MaterialAlphaMode alphaMode,
          float transmissionFactor) -> Result<MaterialRef, std::string> {
    MaterialRequest request{};
    request.debugName = std::string(name);
    request.desc.baseColorFactor = color;
    request.desc.emissiveFactor = glm::vec3(color) * 0.65f;
    request.desc.emissiveStrength = 1.35f;
    request.desc.metallicFactor = 0.0f;
    request.desc.roughnessFactor = 0.35f;
    request.desc.alphaMode = alphaMode;
    request.desc.doubleSided = true;
    if (transmissionFactor > 0.0f) {
      request.desc.featureMask |= kMaterialFeatureTransmission;
      request.desc.transmissionFactor = transmissionFactor;
      request.desc.thicknessFactor = 0.08f;
      request.desc.attenuationDistance = 2.0f;
    }
    return renderer.resources().acquireMaterial(request);
  };

  auto backgroundMaterial = acquireMaterial(
      "snapshot_transmission_background", glm::vec4(0.03f, 0.12f, 0.24f, 1.0f),
      MaterialAlphaMode::Opaque, 0.0f);
  if (backgroundMaterial.hasError()) {
    return Result<bool, std::string>::makeError(backgroundMaterial.error());
  }
  auto warmMaterial = acquireMaterial("snapshot_transmission_warm_band",
                                      glm::vec4(1.0f, 0.34f, 0.10f, 1.0f),
                                      MaterialAlphaMode::Opaque, 0.0f);
  if (warmMaterial.hasError()) {
    return Result<bool, std::string>::makeError(warmMaterial.error());
  }
  auto coolMaterial = acquireMaterial("snapshot_transmission_cool_band",
                                      glm::vec4(0.08f, 0.72f, 0.95f, 1.0f),
                                      MaterialAlphaMode::Opaque, 0.0f);
  if (coolMaterial.hasError()) {
    return Result<bool, std::string>::makeError(coolMaterial.error());
  }
  auto transparentMaterial = acquireMaterial(
      "snapshot_plain_transparent", glm::vec4(1.0f, 0.82f, 0.20f, 0.46f),
      MaterialAlphaMode::Blend, 0.0f);
  if (transparentMaterial.hasError()) {
    return Result<bool, std::string>::makeError(transparentMaterial.error());
  }
  auto transmissionMaterial = acquireMaterial(
      "snapshot_blended_transmission", glm::vec4(0.52f, 0.96f, 1.0f, 0.42f),
      MaterialAlphaMode::Blend, 1.0f);
  if (transmissionMaterial.hasError()) {
    return Result<bool, std::string>::makeError(transmissionMaterial.error());
  }

  const glm::mat4 upright = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f),
                                        glm::vec3(1.0f, 0.0f, 0.0f));
  auto addPlane = [&](std::string_view name, const glm::vec3 &translation,
                      const glm::vec3 &scale,
                      MaterialRef material) -> Result<bool, std::string> {
    const glm::mat4 transform = glm::translate(glm::mat4(1.0f), translation) *
                                upright * glm::scale(glm::mat4(1.0f), scale);
    auto nodeResult =
        scene.graph().createNode(scene.graph().rootNode(), name, transform);
    if (nodeResult.hasError()) {
      return Result<bool, std::string>::makeError(nodeResult.error());
    }
    auto renderableResult = scene.graph().addRenderable(
        nodeResult.value(), modelResult.value(), material);
    if (renderableResult.hasError()) {
      return Result<bool, std::string>::makeError(renderableResult.error());
    }
    return Result<bool, std::string>::makeResult(true);
  };

  auto result =
      addPlane("OpaqueBackground", glm::vec3(0.0f, 0.0f, -0.80f),
               glm::vec3(5.0f, 3.0f, 1.0f), backgroundMaterial.value());
  if (result.hasError()) {
    return result;
  }
  result = addPlane("OpaqueWarmBand", glm::vec3(-0.82f, -0.10f, -0.68f),
                    glm::vec3(1.85f, 1.18f, 1.0f), warmMaterial.value());
  if (result.hasError()) {
    return result;
  }
  result = addPlane("OpaqueCoolBand", glm::vec3(0.82f, 0.12f, -0.62f),
                    glm::vec3(1.85f, 1.18f, 1.0f), coolMaterial.value());
  if (result.hasError()) {
    return result;
  }
  result = addPlane("PlainTransparentBehind", glm::vec3(-0.38f, 0.04f, 0.0f),
                    glm::vec3(2.15f, 1.72f, 1.0f), transparentMaterial.value());
  if (result.hasError()) {
    return result;
  }
  return addPlane("BlendedTransmissionInFront", glm::vec3(0.34f, -0.02f, 0.25f),
                  glm::vec3(2.0f, 1.9f, 1.0f), transmissionMaterial.value());
}

[[nodiscard]] Result<bool, std::string>
populateReactiveMaskScene(const SnapshotCase &snapshotCase, Renderer &renderer,
                          RenderScene &scene) {
  if (snapshotCase.scene.generator != "nuri.procedural.reactive_mask.v1") {
    return Result<bool, std::string>::makeResult(true);
  }

  auto planePath = resolveSnapshotPath("modelsRoot", "common/flat_plane.obj");
  if (planePath.hasError()) {
    return Result<bool, std::string>::makeError(planePath.error());
  }
  auto modelResult = renderer.resources().acquireModel(ModelRequest{
      .path = planePath.value().string(),
      .debugName = "snapshot_reactive_flat_plane",
  });
  if (modelResult.hasError()) {
    return Result<bool, std::string>::makeError(modelResult.error());
  }

  auto acquireMaterial =
      [&](std::string_view name, const glm::vec4 &color,
          MaterialAlphaMode alphaMode) -> Result<MaterialRef, std::string> {
    MaterialRequest request{};
    request.debugName = std::string(name);
    request.desc.baseColorFactor = color;
    request.desc.emissiveFactor = glm::vec3(color) * 0.75f;
    request.desc.emissiveStrength = 1.6f;
    request.desc.metallicFactor = 0.0f;
    request.desc.roughnessFactor = 0.55f;
    request.desc.alphaMode = alphaMode;
    request.desc.alphaCutoff = 0.5f;
    request.desc.doubleSided = true;
    return renderer.resources().acquireMaterial(request);
  };

  auto backgroundMaterial = acquireMaterial(
      "snapshot_reactive_background", glm::vec4(0.04f, 0.12f, 0.22f, 1.0f),
      MaterialAlphaMode::Opaque);
  if (backgroundMaterial.hasError()) {
    return Result<bool, std::string>::makeError(backgroundMaterial.error());
  }
  auto maskedMaterial = acquireMaterial("snapshot_alpha_mask_reactive",
                                        glm::vec4(0.32f, 1.0f, 0.48f, 0.85f),
                                        MaterialAlphaMode::Mask);
  if (maskedMaterial.hasError()) {
    return Result<bool, std::string>::makeError(maskedMaterial.error());
  }

  const glm::mat4 upright = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f),
                                        glm::vec3(1.0f, 0.0f, 0.0f));
  auto addPlane = [&](std::string_view name, const glm::vec3 &translation,
                      const glm::vec3 &scale,
                      MaterialRef material) -> Result<bool, std::string> {
    const glm::mat4 transform = glm::translate(glm::mat4(1.0f), translation) *
                                upright * glm::scale(glm::mat4(1.0f), scale);
    auto nodeResult =
        scene.graph().createNode(scene.graph().rootNode(), name, transform);
    if (nodeResult.hasError()) {
      return Result<bool, std::string>::makeError(nodeResult.error());
    }
    auto renderableResult = scene.graph().addRenderable(
        nodeResult.value(), modelResult.value(), material);
    if (renderableResult.hasError()) {
      return Result<bool, std::string>::makeError(renderableResult.error());
    }
    return Result<bool, std::string>::makeResult(true);
  };

  auto result =
      addPlane("ReactiveBackground", glm::vec3(0.0f, 0.0f, -0.65f),
               glm::vec3(4.6f, 2.8f, 1.0f), backgroundMaterial.value());
  if (result.hasError()) {
    return result;
  }
  result = addPlane("ReactiveMaskTall", glm::vec3(-0.32f, 0.0f, 0.0f),
                    glm::vec3(1.0f, 1.9f, 1.0f), maskedMaterial.value());
  if (result.hasError()) {
    return result;
  }
  return addPlane("ReactiveMaskOffset", glm::vec3(0.54f, 0.22f, 0.08f),
                  glm::vec3(0.95f, 0.95f, 1.0f), maskedMaterial.value());
}

[[nodiscard]] Result<bool, std::string>
populateShadowPlanesScene(const SnapshotCase &snapshotCase, Renderer &renderer,
                          RenderScene &scene) {
  if (snapshotCase.scene.generator != "nuri.procedural.shadow_planes.v1") {
    return Result<bool, std::string>::makeResult(true);
  }

  auto planePath = resolveSnapshotPath("modelsRoot", "common/flat_plane.obj");
  if (planePath.hasError()) {
    return Result<bool, std::string>::makeError(planePath.error());
  }
  auto modelResult = renderer.resources().acquireModel(ModelRequest{
      .path = planePath.value().string(),
      .debugName = "snapshot_shadow_flat_plane",
  });
  if (modelResult.hasError()) {
    return Result<bool, std::string>::makeError(modelResult.error());
  }

  auto acquireMaterial =
      [&](std::string_view name,
          const glm::vec4 &color) -> Result<MaterialRef, std::string> {
    MaterialRequest request{};
    request.debugName = std::string(name);
    request.desc.baseColorFactor = color;
    request.desc.emissiveFactor = glm::vec3(color) * 0.45f;
    request.desc.emissiveStrength = 1.1f;
    request.desc.metallicFactor = 0.0f;
    request.desc.roughnessFactor = 0.72f;
    request.desc.doubleSided = true;
    return renderer.resources().acquireMaterial(request);
  };

  auto floorMaterial = acquireMaterial("snapshot_shadow_floor",
                                       glm::vec4(0.72f, 0.70f, 0.64f, 1.0f));
  if (floorMaterial.hasError()) {
    return Result<bool, std::string>::makeError(floorMaterial.error());
  }
  auto wallMaterial = acquireMaterial("snapshot_shadow_wall",
                                      glm::vec4(0.58f, 0.68f, 0.78f, 1.0f));
  if (wallMaterial.hasError()) {
    return Result<bool, std::string>::makeError(wallMaterial.error());
  }
  auto redMaterial = acquireMaterial("snapshot_shadow_red",
                                     glm::vec4(0.86f, 0.18f, 0.12f, 1.0f));
  if (redMaterial.hasError()) {
    return Result<bool, std::string>::makeError(redMaterial.error());
  }
  auto blueMaterial = acquireMaterial("snapshot_shadow_blue",
                                      glm::vec4(0.12f, 0.36f, 0.88f, 1.0f));
  if (blueMaterial.hasError()) {
    return Result<bool, std::string>::makeError(blueMaterial.error());
  }
  auto greenMaterial = acquireMaterial("snapshot_shadow_green",
                                       glm::vec4(0.16f, 0.72f, 0.34f, 1.0f));
  if (greenMaterial.hasError()) {
    return Result<bool, std::string>::makeError(greenMaterial.error());
  }

  const glm::mat4 upright = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f),
                                        glm::vec3(1.0f, 0.0f, 0.0f));
  auto addPlane = [&](std::string_view name, const glm::mat4 &orientation,
                      const glm::vec3 &translation, const glm::vec3 &scale,
                      MaterialRef material) -> Result<bool, std::string> {
    const glm::mat4 transform = glm::translate(glm::mat4(1.0f), translation) *
                                orientation *
                                glm::scale(glm::mat4(1.0f), scale);
    auto nodeResult =
        scene.graph().createNode(scene.graph().rootNode(), name, transform);
    if (nodeResult.hasError()) {
      return Result<bool, std::string>::makeError(nodeResult.error());
    }
    auto renderableResult = scene.graph().addRenderable(
        nodeResult.value(), modelResult.value(), material);
    if (renderableResult.hasError()) {
      return Result<bool, std::string>::makeError(renderableResult.error());
    }
    return Result<bool, std::string>::makeResult(true);
  };

  auto result =
      addPlane("ShadowFloor", glm::mat4(1.0f), glm::vec3(0.0f, -0.72f, 0.0f),
               glm::vec3(5.2f, 1.0f, 4.4f), floorMaterial.value());
  if (result.hasError()) {
    return result;
  }
  result = addPlane("ShadowBackWall", upright, glm::vec3(0.0f, 0.64f, -1.38f),
                    glm::vec3(5.2f, 2.7f, 1.0f), wallMaterial.value());
  if (result.hasError()) {
    return result;
  }
  result = addPlane("ShadowRedCaster", upright, glm::vec3(-1.2f, 0.14f, 0.15f),
                    glm::vec3(0.72f, 1.5f, 1.0f), redMaterial.value());
  if (result.hasError()) {
    return result;
  }
  result = addPlane("ShadowBlueCaster", upright, glm::vec3(0.0f, 0.34f, 0.0f),
                    glm::vec3(0.58f, 1.95f, 1.0f), blueMaterial.value());
  if (result.hasError()) {
    return result;
  }
  return addPlane("ShadowGreenCaster", upright, glm::vec3(1.16f, 0.06f, 0.35f),
                  glm::vec3(0.82f, 1.34f, 1.0f), greenMaterial.value());
}

[[nodiscard]] Result<bool, std::string>
populateScene(const SnapshotCase &snapshotCase, Renderer &renderer,
              RenderScene &scene, std::pmr::memory_resource *memory,
              std::optional<ScenePrefab> &prefab,
              std::optional<ScenePrefabAssets> &prefabAssets) {
  scene.bindResources(&renderer.resources());
  LightDesc keyLight{.type = LightType::Directional,
                     .name = "snapshot_key",
                     .color = glm::vec3(1.0f),
                     .intensity = 4.0f,
                     .enabled = true};
  if (snapshotCase.scene.generator == "nuri.procedural.shadow_planes.v1") {
    keyLight.rotation =
        glm::quatLookAt(glm::normalize(glm::vec3(-0.45f, -0.78f, -0.44f)),
                        glm::vec3(0.0f, 1.0f, 0.0f));
    keyLight.intensity = 5.5f;
  }
  auto lightResult = scene.graph().addLight(scene.graph().rootNode(), keyLight);
  if (lightResult.hasError()) {
    return Result<bool, std::string>::makeError(lightResult.error());
  }

  if (snapshotCase.scene.kind == "prefab") {
    if (snapshotCase.scene.pathBase.empty() ||
        snapshotCase.scene.path.empty()) {
      return Result<bool, std::string>::makeError(
          "prefab scene requires pathBase and path");
    }
    auto path = resolveSnapshotPath(snapshotCase.scene.pathBase,
                                    snapshotCase.scene.path);
    if (path.hasError()) {
      return Result<bool, std::string>::makeError(path.error());
    }
    if (!std::filesystem::exists(path.value())) {
      return Result<bool, std::string>::makeError("missing scene asset: " +
                                                  path.value().string());
    }
    SceneImportOptions importOptions{};
    importOptions.assetBuildOptions.flipUVs = snapshotCase.scene.flipUVs;
    importOptions.assetBuildOptions.generateMeshlets =
        snapshotCase.scene.generateMeshlets;
    importOptions.assetBuildOptions.meshletMaxVertices =
        snapshotCase.scene.meshletMaxVertices;
    importOptions.assetBuildOptions.meshletMaxPrimitives =
        snapshotCase.scene.meshletMaxPrimitives;
    importOptions.assetBuildOptions.meshletConeWeight =
        snapshotCase.scene.meshletConeWeight;
    auto prefabResult = SceneImporter::loadScenePrefabFromFile(
        path.value().string(), importOptions, memory);
    if (prefabResult.hasError()) {
      return Result<bool, std::string>::makeError(prefabResult.error());
    }
    prefab.emplace(std::move(prefabResult.value()));
    auto assetsResult = renderer.resources().acquireScenePrefabAssets(*prefab);
    if (assetsResult.hasError()) {
      return Result<bool, std::string>::makeError(assetsResult.error());
    }
    prefabAssets.emplace(std::move(assetsResult.value()));
    auto instantiateResult = scene.graph().instantiatePrefab(
        *prefab, scene.graph().rootNode(), *prefabAssets);
    if (instantiateResult.hasError()) {
      return Result<bool, std::string>::makeError(instantiateResult.error());
    }
  } else if (snapshotCase.scene.kind == "procedural") {
    auto proceduralResult =
        populateTransmissionTransparencyScene(snapshotCase, renderer, scene);
    if (proceduralResult.hasError()) {
      return proceduralResult;
    }
    proceduralResult = populateReactiveMaskScene(snapshotCase, renderer, scene);
    if (proceduralResult.hasError()) {
      return proceduralResult;
    }
    proceduralResult = populateShadowPlanesScene(snapshotCase, renderer, scene);
    if (proceduralResult.hasError()) {
      return proceduralResult;
    }
  }

  auto syncResult = scene.graph().syncWorldTransforms();
  (void)syncResult;
  auto commitResult = scene.commit();
  if (commitResult.hasError()) {
    return Result<bool, std::string>::makeError(commitResult.error());
  }
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Camera makeSnapshotCamera(const SnapshotCase &snapshotCase,
                                        uint64_t frameIndex) {
  Camera camera;
  camera.setPerspective(PerspectiveParams{
      .fovYRadians = glm::radians(snapshotCase.camera.verticalFovDegrees),
      .nearPlane = snapshotCase.camera.nearPlane,
      .farPlane = snapshotCase.camera.farPlane,
  });
  const glm::vec3 direction =
      glm::length(snapshotCase.camera.direction) > 1.0e-6f
          ? glm::normalize(snapshotCase.camera.direction)
          : glm::vec3(0.0f, 0.0f, -1.0f);
  const glm::vec3 position =
      snapshotCase.camera.position + snapshotCase.camera.positionDeltaPerFrame *
                                         static_cast<float>(frameIndex);
  camera.setLookAt(position, position + direction, glm::vec3(0.0f, 1.0f, 0.0f));
  return camera;
}

void buildFrameContext(RenderFrameContext &frameContext, RenderScene &scene,
                       Renderer &renderer, RenderSettings &settings,
                       TemporalFrameService &temporalFrameService,
                       const Camera &camera, uint64_t frameIndex,
                       double timeSeconds, double deltaSeconds, uint32_t width,
                       uint32_t height) {
  sanitizeSnapshotRenderSettings(settings);
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.frameIndex = frameIndex;
  const MaterialTableSnapshot materialSnapshot =
      renderer.resources().materialSnapshot();
  const TemporalSceneContentState sceneContent{
      .lightTopologyVersion = scene.lightTopologyVersion(),
      .lightTransformVersion = scene.lightTransformVersion(),
      .materialTableVersion = materialSnapshot.version,
      .environmentVersion = scene.environmentVersion(),
  };
  auto planResult = buildPresentationAAPlan(
      settings, {}, renderer.resources().gpuMultisampleCapabilities());
  NURI_ASSERT(!planResult.hasError(), "Invalid presentation AA plan: %s",
              planResult.error().c_str());
  frameContext.presentationAA = planResult.value();
  auto cameraResult = temporalFrameService.prepareFrame(
      camera, static_cast<float>(width) / static_cast<float>(height),
      settings.antiAliasing, frameContext.presentationAA,
      TemporalCameraFrameDesc{.renderExtent = glm::uvec2(width, height),
                              .sceneContent = sceneContent},
      frameIndex, timeSeconds, deltaSeconds);
  NURI_ASSERT(!cameraResult.hasError(), "Temporal frame prepare failed: %s",
              cameraResult.error().c_str());
  frameContext.camera = cameraResult.value();
  frameContext.temporalFrameService = &temporalFrameService;
  settings.antiAliasing.debug.resetHistoryRequested = false;
  frameContext.settings = &settings;
  frameContext.metrics = {};
  frameContext.metrics.frameIndex = frameContext.frameIndex;
  frameContext.metrics.antiAliasing =
      makeAntiAliasingFrameMetrics(frameContext.camera);
  frameContext.sharedDepthTexture = {};
  frameContext.timeSeconds = timeSeconds;
  frameContext.deltaSeconds = deltaSeconds;
}

[[nodiscard]] std::filesystem::path
relativeToCaseDir(const std::filesystem::path &caseDir,
                  const std::filesystem::path &path) {
  std::error_code ec;
  std::filesystem::path relative = std::filesystem::relative(path, caseDir, ec);
  return ec ? path : relative;
}

void initializeCaptureReports(SnapshotReport &report) {
  report.captures.clear();
  for (const SnapshotCaptureTarget &target : report.snapshotCase.captures) {
    SnapshotCaptureReport capture{};
    capture.target = target.name;
    capture.artifactStem = target.name;
    capture.profile = target.profile;
    capture.required = target.required;
    capture.status = "missing_capture_point";
    capture.statusReason = "not_rendered";
    report.captures.push_back(std::move(capture));
  }
}

void writeReports(SnapshotRunResult &result, SnapshotReport &report,
                  const std::filesystem::path &jsonPath,
                  const std::filesystem::path &htmlPath) {
  result.reportPath = jsonPath;
  result.htmlPath = htmlPath;
  report.artifacts.caseHtml = htmlPath;
  evaluateSnapshotBaselineProfile(report);
  auto writeJson = writeSnapshotReportFile(report, jsonPath);
  if (writeJson.hasError()) {
    result.exitCode = SnapshotExitCode::RuntimeError;
    result.message = writeJson.error();
    report.errors.push_back(writeJson.error());
  }
  auto writeHtml = writeSnapshotHtmlReportFile(report, htmlPath);
  if (writeHtml.hasError()) {
    result.exitCode = SnapshotExitCode::RuntimeError;
    result.message = writeHtml.error();
    report.errors.push_back(writeHtml.error());
    if (!writeJson.hasError()) {
      (void)writeSnapshotReportFile(report, jsonPath);
    }
  }

  const bool investigative =
      result.exitCode == SnapshotExitCode::Success &&
      (result.message.find("dry run") != std::string::npos ||
       !report.baselineProfileCompatible ||
       std::any_of(report.captures.begin(), report.captures.end(),
                   [](const SnapshotCaptureReport &capture) {
                     return capture.status == "investigative";
                   }));
  const nuri::tools::core::ToolOutcome outcome =
      toolOutcome(result.exitCode, investigative);
  nuri::tools::core::ResultEnvelopeV2 envelope{};
  envelope.tool = nuri::tools::core::ResultToolV2::Snapshot;
  envelope.runId = nuri::tools::core::createRunId();
  envelope.status = outcome;
  envelope.exitCode = static_cast<int>(result.exitCode);
  envelope.authoritative = false;
  envelope.environmentFingerprint =
      snapshotEnvironmentFingerprint(report.environment);
  envelope.workloadFingerprint =
      snapshotWorkloadFingerprint(report.snapshotCase);
  envelope.profile = nuri::tools::core::ResultProfileV2{
      .id = report.baselineProfile,
      .compatible = report.baselineProfileCompatible,
      .incompatibilityReasons = report.baselineProfileIncompatibilityReasons};
  envelope.reproduceCommand = report.reproduceCommand;
  envelope.selection = {
      .requested = report.snapshotCase.id,
      .selected = 1u,
      .attempted = 1u,
      .completed = 1u,
      .passed = outcome == nuri::tools::core::ToolOutcome::Pass ? 1u : 0u,
      .warned =
          outcome == nuri::tools::core::ToolOutcome::Investigative ? 1u : 0u,
      .failed =
          outcome == nuri::tools::core::ToolOutcome::Failure ||
                  outcome == nuri::tools::core::ToolOutcome::RuntimeError ||
                  outcome == nuri::tools::core::ToolOutcome::Invalid ||
                  outcome == nuri::tools::core::ToolOutcome::MissingBaseline
              ? 1u
              : 0u,
      .unavailable =
          outcome == nuri::tools::core::ToolOutcome::EnvironmentUnavailable
              ? 1u
              : 0u,
  };
  for (const std::string &warning : report.warnings) {
    envelope.diagnostics.push_back(
        {.code = "snapshot.warning",
         .severity = nuri::tools::core::ResultDiagnosticSeverityV2::Warning,
         .message = warning});
  }
  for (const std::string &error : report.errors) {
    envelope.diagnostics.push_back(
        {.code = "snapshot.error",
         .severity = nuri::tools::core::ResultDiagnosticSeverityV2::Error,
         .message = error});
  }
  nuri::tools::core::ResultChildV2 child{
      .id = report.snapshotCase.id,
      .status = std::string(nuri::tools::core::toolOutcomeName(outcome)),
      .exitCode = static_cast<int>(result.exitCode)};
  std::error_code relativeError;
  const std::filesystem::path relativeReport = std::filesystem::relative(
      jsonPath, report.artifacts.artifactDir, relativeError);
  if (!relativeError && !relativeReport.empty() &&
      *relativeReport.begin() != "..") {
    child.result = relativeReport;
  }
  envelope.children.push_back(std::move(child));
  const std::filesystem::path rootReport =
      report.artifacts.artifactDir / "run.json";
  auto rootWrite =
      nuri::tools::core::writeResultEnvelopeV2(rootReport, envelope);
  if (rootWrite.hasError()) {
    result.exitCode = SnapshotExitCode::RuntimeError;
    result.message = rootWrite.error();
    report.errors.push_back(rootWrite.error());
    if (!writeJson.hasError()) {
      (void)writeSnapshotReportFile(report, jsonPath);
    }
  }
}

[[nodiscard]] SnapshotReport makeInitialReport(
    const SnapshotCase &snapshotCase, const SnapshotRunOptions &options,
    const std::filesystem::path &artifactDir,
    const std::filesystem::path &caseDir, const std::filesystem::path &htmlPath,
    std::string_view backend, std::string_view backendSource,
    std::string_view presentMode, std::string_view presentSource) {
  SnapshotReport report{};
  report.generatedAtUtc = utcTimestampIso8601();
  report.command = options.command;
  report.baselineProfile = options.baselineProfile;
  report.snapshotCase = snapshotCase;
  report.artifacts.artifactDir = artifactDir;
  report.artifacts.caseDir = caseDir;
  report.artifacts.caseHtml = htmlPath;
  report.environment = collectSnapshotEnvironment(
      backend, backendSource, presentMode, presentSource, options.windowMode,
      options.windowMode);
  report.environment.renderGraphWorkerCount =
      snapshotCase.renderGraph.workerCount;
  report.environment.renderGraphParallelCompile =
      snapshotCase.renderGraph.parallelCompile;
  report.environment.renderGraphParallelRecording =
      snapshotCase.renderGraph.parallelRecording;
  report.reproduceCommand = "nuri-snapshot run --case " + snapshotCase.id +
                            " --baseline-profile " + options.baselineProfile;
  initializeCaptureReports(report);
  return report;
}

} // namespace

Result<std::string, std::string>
formatSnapshotCaseListJson(const std::vector<SnapshotCase> &cases,
                           std::string_view suite) {
  std::ostringstream out;
  out << "{\n  \"cases\": [\n";
  bool first = true;
  for (const SnapshotCase *snapshotCase :
       filterSnapshotCasesBySuite(cases, suite)) {
    if (!first) {
      out << ",\n";
    }
    first = false;
    out << "    {\"id\": \"" << snapshotCase->id << "\", \"suite\": \""
        << snapshotCase->suite << "\", \"description\": \""
        << snapshotCase->description << "\", \"captures\": [";
    for (size_t i = 0u; i < snapshotCase->captures.size(); ++i) {
      if (i != 0u) {
        out << ", ";
      }
      out << "\"" << snapshotCase->captures[i].name << "\"";
    }
    out << "]}";
  }
  out << "\n  ]\n}\n";
  return Result<std::string, std::string>::makeResult(out.str());
}

std::string formatSnapshotCaseListText(const std::vector<SnapshotCase> &cases,
                                       std::string_view suite) {
  std::ostringstream out;
  for (const SnapshotCase *snapshotCase :
       filterSnapshotCasesBySuite(cases, suite)) {
    out << snapshotCase->id << " [" << snapshotCase->suite << "] "
        << snapshotCase->description << "\n";
  }
  return out.str();
}

Result<std::string, std::string>
formatSnapshotCaseExplanationJson(const SnapshotCase &snapshotCase) {
  std::ostringstream out;
  out << "{\n"
      << "  \"id\": \"" << snapshotCase.id << "\",\n"
      << "  \"suite\": \"" << snapshotCase.suite << "\",\n"
      << "  \"description\": \"" << snapshotCase.description << "\",\n"
      << "  \"sceneKind\": \"" << snapshotCase.scene.kind << "\",\n"
      << "  \"backend\": \"" << snapshotCase.backend << "\",\n"
      << "  \"captures\": [\n";
  for (size_t i = 0u; i < snapshotCase.captures.size(); ++i) {
    const SnapshotCaptureTarget &capture = snapshotCase.captures[i];
    const SnapshotCaptureCatalogEntry *catalog =
        findSnapshotCaptureCatalogEntry(capture.name);
    out << "    {\"target\": \"" << capture.name << "\", \"profile\": \""
        << capture.profile << "\", \"availability\": \""
        << (catalog != nullptr &&
                    catalog->availability ==
                        SnapshotCaptureAvailability::KnownNotCapturable
                ? "known_not_capturable"
                : "first_slice")
        << "\"}";
    if (i + 1u != snapshotCase.captures.size()) {
      out << ",";
    }
    out << "\n";
  }
  out << "  ]\n}\n";
  return Result<std::string, std::string>::makeResult(out.str());
}

std::string
formatSnapshotCaseExplanationText(const SnapshotCase &snapshotCase) {
  std::ostringstream out;
  out << snapshotCase.id << "\n"
      << "suite: " << snapshotCase.suite << "\n"
      << "description: " << snapshotCase.description << "\n"
      << "scene: " << snapshotCase.scene.kind << "\n"
      << "backend: " << snapshotCase.backend << "\n"
      << "resolution: " << snapshotCase.resolution[0] << "x"
      << snapshotCase.resolution[1] << "\n"
      << "captures:";
  for (const SnapshotCaptureTarget &capture : snapshotCase.captures) {
    out << " " << capture.name;
  }
  out << "\n";
  return out.str();
}

Result<std::string, std::string>
formatSnapshotEffectiveConfigJson(const SnapshotCase &snapshotCase,
                                  const SnapshotRunOptions &options) {
  std::string backendSource;
  const std::string backend = resolveBackendName(snapshotCase, backendSource);
  std::string presentSource;
  const std::string present = resolvePresentMode(snapshotCase, presentSource);
  std::ostringstream out;
  out << "{\n"
      << "  \"case\": \"" << snapshotCase.id << "\",\n"
      << "  \"backend\": \"" << backend << "\",\n"
      << "  \"backendSource\": \"" << backendSource << "\",\n"
      << "  \"presentMode\": \"" << present << "\",\n"
      << "  \"presentModeSource\": \"" << presentSource << "\",\n"
      << "  \"windowMode\": \"" << options.windowMode << "\",\n"
      << "  \"artifactDir\": \"" << options.artifactDir.generic_string()
      << "\"\n"
      << "}\n";
  return Result<std::string, std::string>::makeResult(out.str());
}

SnapshotRunResult captureSnapshotCase(SnapshotCase snapshotCase,
                                      const SnapshotRunOptions &options) {
  SnapshotRunResult result{};
  auto validInputs = validateRunInputs(snapshotCase, options);
  if (validInputs.hasError()) {
    result.exitCode = SnapshotExitCode::InvalidInput;
    result.message = validInputs.error();
    return result;
  }
  std::string backendSource;
  const std::string backend = resolveBackendName(snapshotCase, backendSource);
  std::string presentSource;
  const std::string presentMode =
      resolvePresentMode(snapshotCase, presentSource);
  const std::filesystem::path artifactDir = [&]() -> std::filesystem::path {
    if (!options.artifactDir.empty()) {
      return options.artifactDir;
    }
    auto workspace = nuri::tools::core::createRunWorkspace(
        snapshotRepoRoot() / "artifacts" / "snapshots");
    if (workspace.hasError()) {
      result.exitCode = SnapshotExitCode::RuntimeError;
      result.message = workspace.error();
      return {};
    }
    return workspace.value().root;
  }();
  if (artifactDir.empty()) {
    return result;
  }
  auto caseDirResult = resolveSnapshotPathUnder(
      artifactDir, std::filesystem::path("cases") / snapshotCase.id);
  if (caseDirResult.hasError()) {
    result.exitCode = SnapshotExitCode::InvalidInput;
    result.message = caseDirResult.error();
    return result;
  }
  const std::filesystem::path caseDir = caseDirResult.value();
  const std::filesystem::path reportPath =
      options.jsonOut.empty() ? caseDir / "report.json" : options.jsonOut;
  const std::filesystem::path htmlPath =
      options.htmlOut.empty() ? caseDir / "report.html" : options.htmlOut;
  SnapshotReport report =
      makeInitialReport(snapshotCase, options, artifactDir, caseDir, htmlPath,
                        backend, backendSource, presentMode, presentSource);

  std::string requirementMessage;
  auto requirements =
      checkRequirements(snapshotCase, backend, options.windowMode,
                        report.warnings, requirementMessage);
  if (requirements.hasError()) {
    result.exitCode = requirements.error();
    result.message = requirementMessage;
    report.warnings.push_back(requirementMessage);
    for (SnapshotCaptureReport &capture : report.captures) {
      capture.status = "environment_unavailable";
      capture.statusReason = "requirements_unavailable";
    }
    writeReports(result, report, reportPath, htmlPath);
    result.report = std::move(report);
    return result;
  }
  if (options.windowMode == "headless") {
    result.exitCode = SnapshotExitCode::EnvironmentUnavailable;
    result.message = "true offscreen/headless mode is unavailable";
    report.warnings.push_back(result.message);
    for (SnapshotCaptureReport &capture : report.captures) {
      capture.status = "environment_unavailable";
      capture.statusReason = "window_mode_unavailable";
    }
    writeReports(result, report, reportPath, htmlPath);
    result.report = std::move(report);
    return result;
  }
  if (options.dryRun) {
    result.exitCode = SnapshotExitCode::Success;
    result.message = "dry run succeeded";
    report.warnings.push_back("dry run: renderer was not initialized");
    for (SnapshotCaptureReport &capture : report.captures) {
      capture.status = "environment_unavailable";
      capture.statusReason = "dry_run";
    }
    writeReports(result, report, reportPath, htmlPath);
    result.report = std::move(report);
    return result;
  }

  for (const char *generatedDir : {"actual", "diff"}) {
    auto generatedPath = resolveSnapshotPathUnder(caseDir, generatedDir);
    if (generatedPath.hasError()) {
      result.exitCode = SnapshotExitCode::InvalidInput;
      result.message = generatedPath.error();
      report.errors.push_back(result.message);
      writeReports(result, report, reportPath, htmlPath);
      result.report = std::move(report);
      return result;
    }
    std::error_code cleanupError;
    std::filesystem::remove_all(generatedPath.value(), cleanupError);
    if (cleanupError) {
      result.exitCode = SnapshotExitCode::RuntimeError;
      result.message = "failed to clean generated snapshot artifacts: " +
                       cleanupError.message();
      report.errors.push_back(result.message);
      writeReports(result, report, reportPath, htmlPath);
      result.report = std::move(report);
      return result;
    }
  }

  try {
    SnapshotLogGuard logGuard(caseDir / "logs" / "snapshot.log");
    std::vector<std::unique_ptr<ScopedEnvVar>> env;
    env.push_back(std::make_unique<ScopedEnvVar>(
        "NURI_RENDER_GRAPH_WORKER_COUNT",
        std::to_string(snapshotCase.renderGraph.workerCount)));
    env.push_back(std::make_unique<ScopedEnvVar>(
        "NURI_RENDER_GRAPH_DISABLE_PARALLEL_COMPILE",
        snapshotCase.renderGraph.parallelCompile ? "0" : "1"));
    env.push_back(std::make_unique<ScopedEnvVar>(
        "NURI_RENDER_GRAPH_DISABLE_PARALLEL_RECORDING",
        snapshotCase.renderGraph.parallelRecording ? "0" : "1"));
    if (presentSource == "manifest" && presentMode != "default") {
      env.push_back(
          std::make_unique<ScopedEnvVar>("NURI_PRESENT_MODE", presentMode));
    }

    auto configResult = loadRuntimeConfigFromEnvOrDefault();
    if (configResult.hasError()) {
      result.exitCode = SnapshotExitCode::EnvironmentUnavailable;
      result.message = configResult.error();
      report.warnings.push_back(result.message);
      for (SnapshotCaptureReport &capture : report.captures) {
        capture.status = "environment_unavailable";
        capture.statusReason = "runtime_config_unavailable";
      }
      writeReports(result, report, reportPath, htmlPath);
      result.report = std::move(report);
      return result;
    }
    RuntimeConfig config = std::move(configResult.value());
    config.window.title = "nuri-snapshot " + snapshotCase.id;
    config.window.width = static_cast<int32_t>(snapshotCase.resolution[0]);
    config.window.height = static_cast<int32_t>(snapshotCase.resolution[1]);
    config.window.mode = options.windowMode == "hidden" ? WindowMode::Hidden
                                                        : WindowMode::Windowed;

    std::unique_ptr<Window> window =
        Window::create(config.window.title, config.window.width,
                       config.window.height, config.window.mode);
    if (!window) {
      result.exitCode = SnapshotExitCode::EnvironmentUnavailable;
      result.message = "failed to create snapshot window";
      report.warnings.push_back(result.message);
      writeReports(result, report, reportPath, htmlPath);
      result.report = std::move(report);
      return result;
    }
    GPUDeviceCreateDesc deviceDesc{};
    deviceDesc.backend = backendPreference(backend);
    std::unique_ptr<GPUDevice> gpu = GPUDevice::create(*window, deviceDesc);
    if (!gpu) {
      result.exitCode = SnapshotExitCode::EnvironmentUnavailable;
      result.message = "failed to create GPU device";
      report.warnings.push_back(result.message);
      writeReports(result, report, reportPath, htmlPath);
      result.report = std::move(report);
      return result;
    }
    report.environment.swapchainImageCount = gpu->getSwapchainImageCount();
    const GPUAdapterInfo adapter = gpu->getAdapterInfo();
    report.environment.gpuDeviceName = adapter.name;
    report.environment.gpuVendorId = adapter.vendorId;
    report.environment.gpuDeviceId = adapter.deviceId;
    report.environment.gpuDriverVersion = adapter.driverVersion;

    std::pmr::unsynchronized_pool_resource rendererMemory;
    std::pmr::unsynchronized_pool_resource pipelineMemory;
    std::pmr::unsynchronized_pool_resource sceneMemory;
    std::unique_ptr<Renderer> renderer = Renderer::create(*gpu, rendererMemory);
    RenderPipeline pipeline(&pipelineMemory);
    auto pipelineResult = registerDefaultRenderPipeline(
        pipeline, *gpu, config.shaders, &pipelineMemory);
    if (pipelineResult.hasError()) {
      result.exitCode = SnapshotExitCode::RuntimeError;
      result.message = pipelineResult.error();
      report.errors.push_back(result.message);
      writeReports(result, report, reportPath, htmlPath);
      result.report = std::move(report);
      return result;
    }

    RenderScene scene(&sceneMemory);
    std::optional<ScenePrefab> prefab;
    std::optional<ScenePrefabAssets> prefabAssets;
    auto sceneResult = populateScene(snapshotCase, *renderer, scene,
                                     &sceneMemory, prefab, prefabAssets);
    if (sceneResult.hasError()) {
      result.exitCode = SnapshotExitCode::EnvironmentUnavailable;
      result.message = sceneResult.error();
      report.warnings.push_back(result.message);
      writeReports(result, report, reportPath, htmlPath);
      result.report = std::move(report);
      return result;
    }

    RenderSettings settings = snapshotCase.settings;
    TemporalFrameService temporalFrameService{};
    RenderFrameContext frameContext{};
    uint64_t frameIndex = 0u;
    double timeSeconds = 0.0;

    const auto renderOneFrame =
        [&](bool captureFrame) -> Result<bool, std::string> {
      window->pollEvents();
      auto commitResult = scene.commit();
      if (commitResult.hasError()) {
        return Result<bool, std::string>::makeError(commitResult.error());
      }
      const Camera camera = makeSnapshotCamera(snapshotCase, frameIndex);
      buildFrameContext(frameContext, scene, *renderer, settings,
                        temporalFrameService, camera, frameIndex, timeSeconds,
                        snapshotCase.fixedDeltaSeconds,
                        snapshotCase.resolution[0], snapshotCase.resolution[1]);
      frameContext.captureRequests.clear();
      if (captureFrame) {
        for (const SnapshotCaptureTarget &capture : snapshotCase.captures) {
          frameContext.captureRequests.request(capture.name);
        }
      }
      auto renderResult = renderer->render(pipeline, frameContext);
      if (renderResult.hasError()) {
        return Result<bool, std::string>::makeError(renderResult.error());
      }
      ++frameIndex;
      timeSeconds += snapshotCase.fixedDeltaSeconds;
      return Result<bool, std::string>::makeResult(true);
    };

    for (uint32_t i = 0u; i < snapshotCase.captureFrame; ++i) {
      auto frameResult = renderOneFrame(false);
      if (frameResult.hasError()) {
        result.exitCode = SnapshotExitCode::RuntimeError;
        result.message = frameResult.error();
        report.errors.push_back(result.message);
        writeReports(result, report, reportPath, htmlPath);
        result.report = std::move(report);
        return result;
      }
    }
    auto captureFrameResult = renderOneFrame(true);
    if (captureFrameResult.hasError()) {
      result.exitCode = SnapshotExitCode::RuntimeError;
      result.message = captureFrameResult.error();
      report.errors.push_back(result.message);
      writeReports(result, report, reportPath, htmlPath);
      result.report = std::move(report);
      return result;
    }
    gpu->waitIdle();

    report.rendererMetrics = frameContext.metrics;
    auto captures = writeSnapshotCaptureArtifacts(
        *gpu, frameContext, snapshotCase.captures, caseDir, caseDir / "actual");
    if (captures.hasError()) {
      result.exitCode = SnapshotExitCode::RuntimeError;
      result.message = captures.error();
      report.errors.push_back(result.message);
      writeReports(result, report, reportPath, htmlPath);
      result.report = std::move(report);
      return result;
    }
    SnapshotCaptureArtifactResult captureArtifacts =
        std::move(captures.value());
    report.captures = std::move(captureArtifacts.captures);
    report.availableCapturePoints =
        std::move(captureArtifacts.availableCapturePoints);
    if (result.exitCode == SnapshotExitCode::Success) {
      if (captureArtifacts.missingRequiredCapture) {
        result.exitCode = SnapshotExitCode::EnvironmentUnavailable;
        result.message = "required capture point missing";
      } else if (captureArtifacts.unsupportedRequiredCapture) {
        result.exitCode = SnapshotExitCode::EnvironmentUnavailable;
        result.message = "required capture format unsupported";
      } else if (captureArtifacts.readbackFailedRequiredCapture) {
        result.exitCode = SnapshotExitCode::RuntimeError;
        result.message = "required capture readback failed";
      } else if (captureArtifacts.incompatibleRequiredCapture) {
        result.exitCode = SnapshotExitCode::InvalidInput;
        result.message = "required capture descriptor is incompatible";
      }
    }
  } catch (const std::exception &ex) {
    result.exitCode = SnapshotExitCode::RuntimeError;
    result.message = ex.what();
    report.errors.push_back(result.message);
  }

  if (result.exitCode == SnapshotExitCode::Success) {
    result.message = "snapshot capture complete";
  }
  writeReports(result, report, reportPath, htmlPath);
  result.report = std::move(report);
  return result;
}

SnapshotRunResult compareSnapshotCase(SnapshotCase snapshotCase,
                                      const SnapshotRunOptions &options) {
  SnapshotRunResult result{};
  auto validInputs = validateRunInputs(snapshotCase, options);
  if (validInputs.hasError()) {
    result.exitCode = SnapshotExitCode::InvalidInput;
    result.message = validInputs.error();
    return result;
  }
  const std::filesystem::path artifactDir =
      options.artifactDir.empty()
          ? snapshotRepoRoot() / "artifacts" / "snapshots"
          : options.artifactDir;
  auto caseDirResult = resolveSnapshotPathUnder(
      artifactDir, std::filesystem::path("cases") / snapshotCase.id);
  if (caseDirResult.hasError()) {
    result.exitCode = SnapshotExitCode::InvalidInput;
    result.message = caseDirResult.error();
    return result;
  }
  const std::filesystem::path caseDir = caseDirResult.value();
  const std::filesystem::path reportPath =
      options.jsonOut.empty() ? caseDir / "report.json" : options.jsonOut;
  auto reportResult = readSnapshotReportFile(reportPath);
  if (reportResult.hasError()) {
    result.exitCode = SnapshotExitCode::InvalidInput;
    result.message = reportResult.error();
    return result;
  }
  SnapshotReport report = std::move(reportResult.value());
  report.baselineProfile = options.baselineProfile;
  evaluateSnapshotBaselineProfile(report);
  bool forcedInvestigative = options.force;
  if (options.force) {
    report.warnings.push_back(
        "forced comparison is investigative and cannot be authoritative");
  }
  if (!report.baselineProfileCompatible) {
    if (!options.force) {
      result.exitCode = SnapshotExitCode::InvalidInput;
      result.message = "snapshot runtime does not match baseline profile";
      report.errors.insert(report.errors.end(),
                           report.baselineProfileIncompatibilityReasons.begin(),
                           report.baselineProfileIncompatibilityReasons.end());
      writeReports(result, report, reportPath,
                   options.htmlOut.empty() ? caseDir / "report.html"
                                           : options.htmlOut);
      result.report = std::move(report);
      return result;
    }
    forcedInvestigative = true;
    report.warnings.insert(report.warnings.end(),
                           report.baselineProfileIncompatibilityReasons.begin(),
                           report.baselineProfileIncompatibilityReasons.end());
  }
  if (report.snapshotCase.id != snapshotCase.id ||
      report.snapshotCase.suite != snapshotCase.suite) {
    if (!options.force) {
      result.exitCode = SnapshotExitCode::InvalidInput;
      result.message = "snapshot report identity does not match requested case";
      result.report = std::move(report);
      return result;
    }
    forcedInvestigative = true;
    report.warnings.push_back(
        "forced comparison: snapshot report identity mismatch");
  }
  report.snapshotCase = snapshotCase;
  report.artifacts.caseDir = caseDir;
  report.artifacts.artifactDir = artifactDir;
  const std::filesystem::path baselineRoot = options.baselineRoot.empty()
                                                 ? defaultSnapshotBaselineRoot()
                                                 : options.baselineRoot;
  auto baselineCaseDir = resolveSnapshotPathUnder(
      baselineRoot, std::filesystem::path(options.baselineProfile) /
                        snapshotCase.suite / snapshotCase.id);
  if (baselineCaseDir.hasError()) {
    result.exitCode = SnapshotExitCode::InvalidInput;
    result.message = baselineCaseDir.error();
    result.report = std::move(report);
    return result;
  }
  auto governedBaseline = verifySnapshotBaseline(
      snapshotCase, options.baselineProfile, baselineRoot);
  if (governedBaseline.hasError()) {
    if (!options.force) {
      const bool baselineExists =
          std::filesystem::is_directory(baselineCaseDir.value());
      result.exitCode = baselineExists ? SnapshotExitCode::InvalidInput
                                       : SnapshotExitCode::MissingBaseline;
      result.message = governedBaseline.error();
      report.errors.push_back(result.message);
      writeReports(result, report, reportPath,
                   options.htmlOut.empty() ? caseDir / "report.html"
                                           : options.htmlOut);
      result.report = std::move(report);
      return result;
    }
    forcedInvestigative = true;
    report.warnings.push_back("forced comparison with unverified baseline: " +
                              governedBaseline.error());
  }
  bool missingBaseline = false;
  bool mismatch = false;
  for (SnapshotCaptureReport &capture : report.captures) {
    if (capture.actual.empty() || capture.status == "missing_capture_point" ||
        capture.status == "unsupported_format" ||
        capture.status == "readback_error") {
      continue;
    }
    capture.expectedHash.clear();
    capture.diff.clear();
    capture.metrics = {};
    capture.failedThresholds.clear();
    const auto manifestCapture =
        std::find_if(snapshotCase.captures.begin(), snapshotCase.captures.end(),
                     [&](const SnapshotCaptureTarget &target) {
                       return target.name == capture.target;
                     });
    const SnapshotCaptureCatalogEntry *catalog =
        findSnapshotCaptureCatalogEntry(capture.target);
    const bool descriptorMatches =
        manifestCapture != snapshotCase.captures.end() && catalog != nullptr &&
        manifestCapture->profile == capture.profile &&
        capture.capturePointVersion == catalog->version &&
        capture.kind == renderCaptureValueKindName(catalog->kind) &&
        snapshotCompareProfileSupportsKind(capture.profile, catalog->kind);
    if (!descriptorMatches) {
      if (!options.force) {
        capture.status = "invalid";
        capture.statusReason = "capture_descriptor_mismatch";
        result.exitCode = SnapshotExitCode::InvalidInput;
        continue;
      }
      forcedInvestigative = true;
      report.warnings.push_back(
          "forced comparison: capture descriptor mismatch for " +
          capture.target);
    }
    auto actualResult = resolveSnapshotPathUnder(caseDir, capture.actual);
    if (actualResult.hasError()) {
      capture.status = "invalid";
      capture.statusReason = "unsafe_actual_path";
      result.exitCode = SnapshotExitCode::InvalidInput;
      continue;
    }
    const std::filesystem::path actual = actualResult.value();
    const std::filesystem::path actualExtension = actual.extension();
    const bool usePreview =
        actualExtension == ".nuri_tex" || actualExtension.empty();
    auto expectedResult = resolveSnapshotPathUnder(
        baselineCaseDir.value(),
        usePreview
            ? std::filesystem::path(capture.target + "_preview.png")
            : std::filesystem::path(capture.target + actualExtension.string()));
    if (expectedResult.hasError()) {
      capture.status = "invalid";
      capture.statusReason = "unsafe_baseline_path";
      result.exitCode = SnapshotExitCode::InvalidInput;
      continue;
    }
    const std::filesystem::path expected = expectedResult.value();
    capture.expected = expected;
    if (!std::filesystem::exists(expected)) {
      capture.status = "missing_baseline";
      capture.statusReason = "baseline_artifact_missing";
      missingBaseline = true;
      continue;
    }
    auto actualMetadataPath =
        resolveSnapshotPathUnder(caseDir, capture.actualMetadata);
    auto expectedMetadataPath = resolveSnapshotPathUnder(
        baselineCaseDir.value(), capture.target + ".json");
    if (actualMetadataPath.hasError() || expectedMetadataPath.hasError()) {
      capture.status = "invalid";
      capture.statusReason = "unsafe_descriptor_path";
      result.exitCode = SnapshotExitCode::InvalidInput;
      continue;
    }
    auto actualMetadata =
        readSnapshotArtifactMetadata(actualMetadataPath.value());
    auto expectedMetadata =
        readSnapshotArtifactMetadata(expectedMetadataPath.value());
    const bool metadataMatches =
        !actualMetadata.hasError() && !expectedMetadata.hasError() &&
        actualMetadata.value().target == capture.target &&
        expectedMetadata.value().target == capture.target &&
        actualMetadata.value().capturePointVersion ==
            capture.capturePointVersion &&
        expectedMetadata.value().capturePointVersion ==
            capture.capturePointVersion &&
        actualMetadata.value().kind == capture.kind &&
        expectedMetadata.value().kind == capture.kind &&
        actualMetadata.value().profile == capture.profile &&
        expectedMetadata.value().profile == capture.profile &&
        actualMetadata.value().format == expectedMetadata.value().format &&
        actualMetadata.value().width == capture.width &&
        expectedMetadata.value().width == capture.width &&
        actualMetadata.value().height == capture.height &&
        expectedMetadata.value().height == capture.height &&
        actualMetadata.value().mip == capture.mip &&
        expectedMetadata.value().mip == capture.mip &&
        actualMetadata.value().layer == capture.layer &&
        expectedMetadata.value().layer == capture.layer &&
        actualMetadata.value().colorSpace == capture.colorSpace &&
        expectedMetadata.value().colorSpace == capture.colorSpace &&
        actualMetadata.value().origin == capture.origin &&
        expectedMetadata.value().origin == capture.origin &&
        actualMetadata.value().hash == capture.actualHash;
    if (!metadataMatches) {
      if (!options.force) {
        capture.status = "invalid";
        capture.statusReason = "artifact_descriptor_mismatch";
        result.exitCode = SnapshotExitCode::InvalidInput;
        continue;
      }
      forcedInvestigative = true;
      report.warnings.push_back(
          "forced comparison: artifact descriptor mismatch for " +
          capture.target);
    } else {
      capture.expectedHash = expectedMetadata.value().hash;
    }
    auto actualImagePath =
        usePreview
            ? resolveSnapshotPathUnder(caseDir, capture.preview)
            : Result<std::filesystem::path, std::string>::makeResult(actual);
    if (actualImagePath.hasError()) {
      capture.status = "invalid";
      capture.statusReason = "unsafe_preview_path";
      result.exitCode = SnapshotExitCode::InvalidInput;
      continue;
    }
    auto actualImage = readSnapshotImageFile(actualImagePath.value());
    auto expectedImage = readSnapshotImageFile(expected);
    if (actualImage.hasError() || expectedImage.hasError()) {
      capture.status = "runtime_error";
      capture.statusReason = "failed_to_load_compare_images";
      result.exitCode = SnapshotExitCode::RuntimeError;
      continue;
    }
    auto expectedPreviewPath = resolveSnapshotPathUnder(
        caseDir,
        std::filesystem::path("expected") / (capture.target + "_expected.png"));
    auto actualPreviewPath = resolveSnapshotPathUnder(caseDir, capture.preview);
    if (expectedPreviewPath.hasError() || actualPreviewPath.hasError()) {
      capture.status = "invalid";
      capture.statusReason = "unsafe_comparison_preview_path";
      result.exitCode = SnapshotExitCode::InvalidInput;
      continue;
    }
    auto comparisonPreviews = writeSnapshotComparisonPreviews(
        actualImage.value(), expectedImage.value(), capture.profile,
        actualPreviewPath.value(), expectedPreviewPath.value());
    if (comparisonPreviews.hasError()) {
      capture.status = "runtime_error";
      capture.statusReason = "failed_to_write_comparison_previews";
      report.errors.push_back(comparisonPreviews.error());
      result.exitCode = SnapshotExitCode::RuntimeError;
      continue;
    }
    capture.expected = relativeToCaseDir(caseDir, expectedPreviewPath.value());
    const SnapshotCompareProfile profile =
        builtinSnapshotCompareProfile(capture.profile);
    SnapshotCompareResult comparison = compareSnapshotImages(
        actualImage.value(), expectedImage.value(), profile);
    capture.metrics = comparison.metrics;
    capture.semanticMetrics = comparison.semantic;
    capture.failedThresholds = comparison.failedThresholds;
    if (!comparison.compatible) {
      capture.status = "runtime_error";
      capture.statusReason = "comparison_incompatible";
      result.exitCode = SnapshotExitCode::InvalidInput;
      continue;
    }
    if (comparison.passed) {
      capture.status = forcedInvestigative ? "investigative" : "pass";
      capture.statusReason = forcedInvestigative
                                 ? "forced_incompatible_comparison"
                                 : "within_thresholds";
    } else {
      capture.status = "fail";
      capture.statusReason = "thresholds_failed";
      mismatch = true;
      const std::filesystem::path diffPath =
          caseDir / "diff" / (capture.target + "_diff.png");
      auto diff = writeSnapshotDiffPng(actualImage.value(),
                                       expectedImage.value(), diffPath);
      if (!diff.hasError()) {
        capture.diff = relativeToCaseDir(caseDir, diffPath);
      }
    }
  }
  if (forcedInvestigative) {
    report.snapshotCase.authoritative = false;
    for (SnapshotCaptureReport &capture : report.captures) {
      if (capture.status == "pass") {
        capture.status = "investigative";
        capture.statusReason = "forced_incompatible_comparison";
      }
    }
  }
  const std::filesystem::path htmlPath =
      options.htmlOut.empty() ? caseDir / "report.html" : options.htmlOut;
  if (result.exitCode == SnapshotExitCode::Success) {
    if (missingBaseline) {
      result.exitCode = SnapshotExitCode::MissingBaseline;
      result.message = "snapshot baseline missing";
    } else if (mismatch) {
      result.exitCode = SnapshotExitCode::VisualMismatch;
      result.message = "snapshot mismatch";
    } else {
      result.message = forcedInvestigative ? "investigative comparison complete"
                                           : "snapshots matched";
    }
  } else if (result.message.empty()) {
    result.message = result.exitCode == SnapshotExitCode::InvalidInput
                         ? "snapshot comparison invalid"
                         : "snapshot comparison failed";
  }
  writeReports(result, report, reportPath, htmlPath);
  result.report = std::move(report);
  return result;
}

SnapshotRunResult runSnapshotCase(SnapshotCase snapshotCase,
                                  const SnapshotRunOptions &options) {
  SnapshotRunResult capture = captureSnapshotCase(snapshotCase, options);
  if (capture.exitCode != SnapshotExitCode::Success) {
    return capture;
  }
  SnapshotRunOptions compareOptions = options;
  compareOptions.artifactDir = capture.report.artifacts.artifactDir;
  compareOptions.jsonOut = capture.reportPath;
  return compareSnapshotCase(std::move(snapshotCase), compareOptions);
}

SnapshotSuiteRunResult runSnapshotSuite(std::vector<SnapshotCase> snapshotCases,
                                        std::string_view suite,
                                        const SnapshotRunOptions &options) {
  SnapshotSuiteRunResult suiteResult{};
  auto validSuite = validateSnapshotIdentifier(suite, "suite");
  if (validSuite.hasError()) {
    suiteResult.exitCode = SnapshotExitCode::InvalidInput;
    suiteResult.message = validSuite.error();
  }
  std::string runId = nuri::tools::core::createRunId();
  std::filesystem::path artifactDir = options.artifactDir;
  if (artifactDir.empty()) {
    auto workspace = nuri::tools::core::createRunWorkspace(
        snapshotRepoRoot() / "artifacts" / "snapshots");
    if (workspace.hasError()) {
      suiteResult.exitCode = SnapshotExitCode::RuntimeError;
      suiteResult.message = workspace.error();
      return suiteResult;
    }
    runId = workspace.value().runId;
    artifactDir = workspace.value().root;
  }
  std::vector<size_t> selectedIndices;
  if (suiteResult.exitCode == SnapshotExitCode::Success) {
    std::vector<nuri::tools::core::CaseCatalogEntry> catalog;
    catalog.reserve(snapshotCases.size());
    for (const SnapshotCase &snapshotCase : snapshotCases) {
      catalog.push_back({.id = snapshotCase.id,
                         .suite = snapshotCase.suite,
                         .manifestPath = snapshotCase.manifestPath});
    }
    auto selected = nuri::tools::core::selectCaseCatalog(
        catalog,
        nuri::tools::core::CaseCatalogSelector{.suite = std::string(suite)},
        nuri::tools::core::CaseCatalogZeroMatchPolicy::Reject, "snapshot");
    if (selected.hasError()) {
      suiteResult.exitCode = SnapshotExitCode::InvalidInput;
      suiteResult.message = selected.error();
    } else {
      selectedIndices = std::move(selected.value());
    }
  }
  for (const size_t index : selectedIndices) {
    SnapshotCase &snapshotCase = snapshotCases[index];
    SnapshotRunOptions caseOptions = options;
    caseOptions.artifactDir = artifactDir;
    caseOptions.jsonOut.clear();
    caseOptions.htmlOut.clear();
    SnapshotRunResult result =
        runSnapshotCase(std::move(snapshotCase), caseOptions);
    if (outcomePrecedence(result.exitCode) >
        outcomePrecedence(suiteResult.exitCode)) {
      suiteResult.exitCode = result.exitCode;
    }
    suiteResult.caseResults.push_back(std::move(result));
  }
  std::vector<SnapshotReport> reports;
  reports.reserve(suiteResult.caseResults.size());
  for (const SnapshotRunResult &caseResult : suiteResult.caseResults) {
    reports.push_back(caseResult.report);
  }
  if (suiteResult.message.empty()) {
    suiteResult.message = suiteResult.exitCode == SnapshotExitCode::Success
                              ? "suite run complete"
                              : "suite run completed with failures";
  }
  suiteResult.reportPath =
      options.jsonOut.empty() ? artifactDir / "run.json" : options.jsonOut;
  suiteResult.htmlPath =
      options.htmlOut.empty() ? artifactDir / "index.html" : options.htmlOut;
  uint64_t passed = 0u;
  uint64_t failed = 0u;
  uint64_t unavailable = 0u;
  uint64_t missingBaseline = 0u;
  for (const SnapshotRunResult &caseResult : suiteResult.caseResults) {
    passed += caseResult.exitCode == SnapshotExitCode::Success ? 1u : 0u;
    failed += (caseResult.exitCode == SnapshotExitCode::VisualMismatch ||
               caseResult.exitCode == SnapshotExitCode::RuntimeError ||
               caseResult.exitCode == SnapshotExitCode::InvalidInput)
                  ? 1u
                  : 0u;
    unavailable +=
        caseResult.exitCode == SnapshotExitCode::EnvironmentUnavailable ? 1u
                                                                        : 0u;
    missingBaseline +=
        caseResult.exitCode == SnapshotExitCode::MissingBaseline ? 1u : 0u;
  }
  nuri::tools::core::ResultEnvelopeV2 envelope{};
  envelope.tool = nuri::tools::core::ResultToolV2::Snapshot;
  envelope.runId = runId;
  const bool suiteProfileCompatible = std::all_of(
      suiteResult.caseResults.begin(), suiteResult.caseResults.end(),
      [](const SnapshotRunResult &child) {
        return child.report.baselineProfileCompatible;
      });
  envelope.status =
      toolOutcome(suiteResult.exitCode,
                  options.force || options.dryRun || !suiteProfileCompatible);
  envelope.exitCode = static_cast<int>(suiteResult.exitCode);
  envelope.authoritative = false;
  std::vector<std::pair<std::string, std::string>> environmentFingerprints;
  std::vector<std::pair<std::string, std::string>> workloadFingerprints;
  for (const SnapshotRunResult &child : suiteResult.caseResults) {
    if (auto fingerprint =
            snapshotEnvironmentFingerprint(child.report.environment)) {
      environmentFingerprints.emplace_back(child.report.snapshotCase.id,
                                           std::move(*fingerprint));
    }
    if (auto fingerprint =
            snapshotWorkloadFingerprint(child.report.snapshotCase)) {
      workloadFingerprints.emplace_back(child.report.snapshotCase.id,
                                        std::move(*fingerprint));
    }
  }
  envelope.environmentFingerprint = aggregateSnapshotFingerprint(
      "snapshot.suite.environment", environmentFingerprints);
  envelope.workloadFingerprint = aggregateSnapshotFingerprint(
      "snapshot.suite.workload", workloadFingerprints);
  envelope.selection = {
      .requested = std::string(suite),
      .selected = suiteResult.caseResults.size(),
      .attempted = suiteResult.caseResults.size(),
      .completed = suiteResult.caseResults.size(),
      .passed = options.force || options.dryRun || !suiteProfileCompatible
                    ? 0u
                    : passed,
      .warned = options.force || options.dryRun || !suiteProfileCompatible
                    ? passed
                    : 0u,
      .failed = failed + missingBaseline,
      .unavailable = unavailable,
  };
  std::vector<std::string> suiteProfileReasons;
  for (const SnapshotRunResult &child : suiteResult.caseResults) {
    suiteProfileReasons.insert(
        suiteProfileReasons.end(),
        child.report.baselineProfileIncompatibilityReasons.begin(),
        child.report.baselineProfileIncompatibilityReasons.end());
  }
  std::sort(suiteProfileReasons.begin(), suiteProfileReasons.end());
  suiteProfileReasons.erase(
      std::unique(suiteProfileReasons.begin(), suiteProfileReasons.end()),
      suiteProfileReasons.end());
  envelope.profile = nuri::tools::core::ResultProfileV2{
      .id = options.baselineProfile,
      .compatible = suiteProfileCompatible,
      .incompatibilityReasons = std::move(suiteProfileReasons)};
  envelope.diagnostics.push_back(
      {.code = "snapshot.suite.summary",
       .severity = suiteResult.exitCode == SnapshotExitCode::Success
                       ? nuri::tools::core::ResultDiagnosticSeverityV2::Info
                       : nuri::tools::core::ResultDiagnosticSeverityV2::Error,
       .message = suiteResult.message});
  for (const SnapshotRunResult &child : suiteResult.caseResults) {
    std::error_code relativeError;
    std::filesystem::path relativeReport =
        std::filesystem::relative(child.reportPath, artifactDir, relativeError);
    if (relativeError) {
      relativeReport = std::filesystem::path("cases") /
                       child.report.snapshotCase.id / "report.json";
    }
    envelope.children.push_back(
        {.id = child.report.snapshotCase.id,
         .status = std::string(outcomeStatus(child.exitCode)),
         .exitCode = static_cast<int>(child.exitCode),
         .result = relativeReport});
  }
  auto envelopeWrite = nuri::tools::core::writeResultEnvelopeV2(
      suiteResult.reportPath, envelope);
  if (envelopeWrite.hasError()) {
    suiteResult.exitCode = SnapshotExitCode::RuntimeError;
    suiteResult.message = envelopeWrite.error();
    return suiteResult;
  }
  auto html = writeSnapshotSuiteHtmlFile(reports, suite, suiteResult.htmlPath);
  if (html.hasError()) {
    suiteResult.exitCode = SnapshotExitCode::RuntimeError;
    suiteResult.message = html.error();
    return suiteResult;
  }
  return suiteResult;
}

} // namespace nuri::tools::snapshot
