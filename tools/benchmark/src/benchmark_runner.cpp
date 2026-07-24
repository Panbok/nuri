#include "nuri/tools/benchmark/benchmark_runner.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/core/runtime_config.h"
#include "nuri/core/window.h"
#include "nuri/gfx/frame/temporal_frame_service.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/owned_gpu_resource.h"
#include "nuri/gfx/pipeline/default_render_pipeline.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/gfx/renderer.h"
#include "nuri/gfx/renderers/detail/instance_data.h"
#include "nuri/gfx/sim/animation_scene_frame_data.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/resources/scene_importer.h"
#include "nuri/scene/camera.h"
#include "nuri/scene/render_scene.h"
#include "nuri/tools/benchmark/benchmark_environment.h"
#include "nuri/tools/benchmark/benchmark_manifest.h"
#include "nuri/tools/benchmark/benchmark_metric_registry.h"
#include "nuri/tools/core/atomic_file.h"
#include "nuri/tools/core/case_catalog.h"
#include "nuri/tools/core/fingerprint.h"
#include "nuri/tools/core/process.h"
#include "nuri/tools/core/result_envelope_v2.h"
#include "nuri/tools/core/run_workspace.h"
#include "nuri/tools/core/safe_path.h"
#include "nuri/tools/core/sha256.h"
#include "nuri/tools/runtime/render_tool_runtime.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <map>
#include <memory_resource>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <yyjson.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef PSAPI_VERSION
#define PSAPI_VERSION 2
#endif
// clang-format off
#include <stdlib.h>
#include <windows.h>
#include <psapi.h>
// clang-format on
#else
#include <sys/resource.h>
#endif

namespace nuri::tools::benchmark {
namespace {

constexpr double kBytesPerMiB = 1024.0 * 1024.0;

struct BenchmarkAnimationFixture {
  OwnedBufferHandle instanceMatrices{};
  std::vector<InstanceData> instances{};
  std::vector<AnimatedRenderableGeometryOverride> geometryOverrides{};
  std::vector<uint32_t> animatedRenderableIndices{};
  std::optional<AnimationSceneFrameData> frameData{};

  [[nodiscard]] Result<bool, std::string>
  publish(RenderScene &scene, GPUDevice &gpu, uint64_t frameIndex) {
    const std::span<const Renderable> renderables = scene.renderables();
    instances.clear();
    geometryOverrides.assign(renderables.size(), {});
    animatedRenderableIndices.clear();
    instances.reserve(renderables.size());
    for (uint32_t index = 0u; index < static_cast<uint32_t>(renderables.size());
         ++index) {
      const Renderable &renderable = renderables[index];
      glm::mat4 modelMatrix = renderable.modelMatrix;
      std::string_view name;
      if (scene.graph().getNodeName(renderable.node, name) &&
          name == "DDGI Benchmark Blue Caster") {
        animatedRenderableIndices.push_back(index);
        modelMatrix =
            glm::translate(
                glm::mat4(1.0f),
                glm::vec3(0.35f *
                              std::sin(static_cast<float>(frameIndex) * 0.15f),
                          0.0f, 0.0f)) *
            modelMatrix;
      }
      instances.push_back(makeInstanceData(modelMatrix));
    }
    if (instances.empty() || animatedRenderableIndices.empty()) {
      return Result<bool, std::string>::makeError(
          "DDGI deformation benchmark fixture is incomplete");
    }
    const size_t bytes = instances.size() * sizeof(InstanceData);
    if (!instanceMatrices.valid()) {
      auto buffer = gpu.createBuffer(BufferDesc{.usage = BufferUsage::Storage,
                                                .storage = Storage::Device,
                                                .size = bytes},
                                     "benchmark_ddgi_animation_instances");
      if (buffer.hasError()) {
        return Result<bool, std::string>::makeError(buffer.error());
      }
      instanceMatrices.reset(gpu, buffer.value());
    }
    auto upload = gpu.updateBuffer(instanceMatrices.get(),
                                   std::as_bytes(std::span(instances)));
    if (upload.hasError()) {
      return upload;
    }
    const uint64_t address = gpu.getBufferDeviceAddress(instanceMatrices.get());
    if (address == 0u) {
      return Result<bool, std::string>::makeError(
          "DDGI deformation benchmark instance buffer has no device address");
    }
    frameData = AnimationSceneFrameData{
        .instanceMatricesBuffer = instanceMatrices.get(),
        .instanceMatricesAddress = address,
        .previousInstanceMatricesBuffer = instanceMatrices.get(),
        .previousInstanceMatricesAddress = address,
        .geometryOverridesByRenderable = geometryOverrides,
        .previousGeometryOverridesByRenderable = geometryOverrides,
        .animatedRenderableIndices = animatedRenderableIndices,
        .scene = &scene,
        .sceneTopologyVersion = scene.topologyVersion(),
        .renderableCount = renderables.size(),
        .version = frameIndex + 1u,
    };
    return Result<bool, std::string>::makeResult(true);
  }
};

class BenchmarkAnimationFrameProvider final {
public:
  explicit BenchmarkAnimationFrameProvider(
      const BenchmarkAnimationFixture &fixture) noexcept
      : fixture_(&fixture) {}

  [[nodiscard]] Result<bool, std::string> prepare(FrameBuildContext &ctx) {
    if (fixture_ != nullptr && fixture_->frameData.has_value()) {
      ctx.shared.animationSceneGpuData = *fixture_->frameData;
    }
    return Result<bool, std::string>::makeResult(true);
  }

private:
  const BenchmarkAnimationFixture *fixture_ = nullptr;
};

[[nodiscard]] std::optional<NodeId>
findBenchmarkNodeByName(const SceneGraph &graph, std::string_view name) {
  std::vector<NodeId> pending{graph.rootNode()};
  while (!pending.empty()) {
    const NodeId node = pending.back();
    pending.pop_back();
    std::string_view candidate;
    if (graph.getNodeName(node, candidate) && candidate == name) {
      return node;
    }
    NodeId child{};
    if (!graph.getNodeFirstChild(node, child)) {
      continue;
    }
    for (;;) {
      pending.push_back(child);
      NodeId sibling{};
      if (!graph.getNodeNextSibling(child, sibling)) {
        break;
      }
      child = sibling;
    }
  }
  return std::nullopt;
}

using JsonMutDocPtr =
    std::unique_ptr<yyjson_mut_doc, decltype(&yyjson_mut_doc_free)>;

[[nodiscard]] std::string pathToUtf8(const std::filesystem::path &path) {
  const std::u8string encoded = path.generic_u8string();
  return {reinterpret_cast<const char *>(encoded.data()), encoded.size()};
}

[[nodiscard]] std::optional<std::string>
benchmarkEnvironmentFingerprint(const BenchmarkEnvironment &environment) {
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
      FingerprintField{"window.mode", environment.windowMode},
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
  });
  return fingerprint.hasError()
             ? std::nullopt
             : std::optional<std::string>{std::move(fingerprint.value())};
}

[[nodiscard]] std::optional<std::string>
benchmarkWorkloadFingerprint(const BenchmarkCase &benchmarkCase) {
  using nuri::tools::core::FingerprintField;
  std::string manifestDigest;
  if (!benchmarkCase.manifestPath.empty() &&
      std::filesystem::is_regular_file(benchmarkCase.manifestPath)) {
    auto digest = nuri::tools::core::makeSha256FileFingerprint(
        benchmarkCase.manifestPath);
    if (!digest.hasError()) {
      manifestDigest = std::move(digest.value());
    }
  }
  auto fingerprint = nuri::tools::core::makeSha256Fingerprint({
      FingerprintField{"case.id", benchmarkCase.id},
      FingerprintField{"case.suite", benchmarkCase.suite},
      FingerprintField{"case.comparisonGroup", benchmarkCase.comparisonGroup},
      FingerprintField{"case.variant", benchmarkCase.variant},
      FingerprintField{"manifest", std::move(manifestDigest)},
      FingerprintField{"config", benchmarkCase.configSignature},
      FingerprintField{"settings", benchmarkCase.settingsSignature},
      FingerprintField{"scene.kind", benchmarkCase.scene.kind},
      FingerprintField{"scene.content", benchmarkCase.scene.contentHash},
      FingerprintField{"resolution.width",
                       std::to_string(benchmarkCase.resolution[0])},
      FingerprintField{"resolution.height",
                       std::to_string(benchmarkCase.resolution[1])},
      FingerprintField{"fixedDelta",
                       std::format("{:.17g}", benchmarkCase.fixedDeltaSeconds)},
      FingerprintField{"frames.warmup",
                       std::to_string(benchmarkCase.warmupFrames)},
      FingerprintField{"frames.measurement",
                       std::to_string(benchmarkCase.measurementFrames)},
      FingerprintField{"frames.cooldown",
                       std::to_string(benchmarkCase.cooldownFrames)},
      FingerprintField{"samples", std::to_string(benchmarkCase.samples)},
  });
  return fingerprint.hasError()
             ? std::nullopt
             : std::optional<std::string>{std::move(fingerprint.value())};
}

[[nodiscard]] std::optional<std::string> aggregateFingerprint(
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

void addJsonString(yyjson_mut_doc *document, yyjson_mut_val *object,
                   const char *key, std::string_view value) {
  yyjson_mut_obj_add_strncpy(document, object, key, value.data(), value.size());
}

[[nodiscard]] std::optional<std::filesystem::path>
portableRelativePath(const std::filesystem::path &path,
                     const std::filesystem::path &runRoot) {
  std::error_code error;
  std::filesystem::path relative =
      std::filesystem::relative(path, runRoot, error);
  if (error || relative.empty() || relative.is_absolute()) {
    return std::nullopt;
  }
  for (const std::filesystem::path &component : relative) {
    if (component == "..") {
      return std::nullopt;
    }
  }
  return relative;
}

[[nodiscard]] nuri::tools::core::ToolOutcome
benchmarkOutcome(BenchmarkExitCode exitCode, bool authoritative = false) {
  using nuri::tools::core::ToolOutcome;
  switch (exitCode) {
  case BenchmarkExitCode::Success:
    return authoritative ? ToolOutcome::Pass : ToolOutcome::Investigative;
  case BenchmarkExitCode::Regression:
    return ToolOutcome::Failure;
  case BenchmarkExitCode::InvalidInput:
    return ToolOutcome::Invalid;
  case BenchmarkExitCode::EnvironmentUnavailable:
    return ToolOutcome::EnvironmentUnavailable;
  case BenchmarkExitCode::RuntimeError:
    return ToolOutcome::RuntimeError;
  case BenchmarkExitCode::MissingBaseline:
    return ToolOutcome::MissingBaseline;
  }
  return ToolOutcome::RuntimeError;
}

[[nodiscard]] Result<std::string, std::string>
makeBenchmarkSuitePayload(const BenchmarkSuiteRunResult &result,
                          std::string_view suite,
                          const BenchmarkRunOptions &options,
                          const std::filesystem::path &artifactDir) {
  JsonMutDocPtr document(yyjson_mut_doc_new(nullptr), &yyjson_mut_doc_free);
  if (!document) {
    return Result<std::string, std::string>::makeError(
        "failed to allocate benchmark suite payload");
  }
  yyjson_mut_val *root = yyjson_mut_obj(document.get());
  const bool authoritative =
      !result.caseResults.empty() &&
      std::all_of(result.caseResults.begin(), result.caseResults.end(),
                  [](const BenchmarkRunResult &child) {
                    return child.report.profile.authoritative;
                  });
  uint32_t completedRepetitions = 0u;
  bool allWarmupStable = !result.caseResults.empty();
  bool anyWarmupUnstable = false;
  if (!result.caseResults.empty()) {
    completedRepetitions = std::numeric_limits<uint32_t>::max();
    for (const BenchmarkRunResult &child : result.caseResults) {
      completedRepetitions = std::min(
          completedRepetitions, child.report.profile.completedRepetitions);
      allWarmupStable = allWarmupStable &&
                        child.report.profile.warmupStabilityStatus == "stable";
      anyWarmupUnstable =
          anyWarmupUnstable ||
          child.report.profile.warmupStabilityStatus == "unstable";
    }
  }
  yyjson_mut_doc_set_root(document.get(), root);
  yyjson_mut_obj_add_uint(document.get(), root, "schemaVersion", 1u);
  addJsonString(document.get(), root, "kind", "nuri.benchmark.suite_report");
  addJsonString(document.get(), root, "suite", suite);
  addJsonString(document.get(), root, "outcome",
                nuri::tools::core::toolOutcomeName(
                    benchmarkOutcome(result.exitCode, authoritative)));
  yyjson_mut_obj_add_int(document.get(), root, "exitCode",
                         static_cast<int>(result.exitCode));
  addJsonString(document.get(), root, "baselineProfile",
                options.baselineProfileId);
  yyjson_mut_obj_add_bool(document.get(), root, "profileAuthoritative",
                          options.baselineProfileAuthoritative);
  yyjson_mut_obj_add_bool(document.get(), root, "authoritative", authoritative);
  yyjson_mut_obj_add_uint(document.get(), root, "minimumIndependentRepetitions",
                          options.baselineProfileMinimumRepetitions);
  yyjson_mut_obj_add_uint(document.get(), root,
                          "completedIndependentRepetitions",
                          completedRepetitions);
  addJsonString(document.get(), root, "repetitionUnit",
                options.isolatedRepetitions.has_value() ? "isolated-process"
                                                        : "not-collected");
  addJsonString(document.get(), root, "warmupStabilityStatus",
                allWarmupStable ? "stable"
                                : (anyWarmupUnstable ? "unstable" : "unknown"));
  yyjson_mut_obj_add_uint(document.get(), root, "selectedCount",
                          result.caseResults.size());
  yyjson_mut_obj_add_uint(document.get(), root, "executedCount",
                          result.caseResults.size());

  std::map<std::string, std::vector<const BenchmarkRunResult *>>
      experimentGroups;
  for (const BenchmarkRunResult &child : result.caseResults) {
    if (!child.report.benchmarkCase.comparisonGroup.empty()) {
      experimentGroups[child.report.benchmarkCase.comparisonGroup].push_back(
          &child);
    }
  }
  yyjson_mut_val *groups = yyjson_mut_arr(document.get());
  for (const auto &[groupName, variants] : experimentGroups) {
    yyjson_mut_val *group = yyjson_mut_obj(document.get());
    addJsonString(document.get(), group, "comparisonGroup", groupName);
    yyjson_mut_obj_add_uint(document.get(), group, "variantCount",
                            variants.size());
    const bool complete =
        std::all_of(variants.begin(), variants.end(), [](const auto *child) {
          return child->exitCode == BenchmarkExitCode::Success;
        });
    yyjson_mut_obj_add_bool(document.get(), group, "complete", complete);
    yyjson_mut_val *variantRows = yyjson_mut_arr(document.get());
    for (const BenchmarkRunResult *child : variants) {
      yyjson_mut_val *row = yyjson_mut_obj(document.get());
      addJsonString(document.get(), row, "variant",
                    child->report.benchmarkCase.variant);
      addJsonString(document.get(), row, "caseId",
                    child->report.benchmarkCase.id);
      addJsonString(document.get(), row, "outcome",
                    nuri::tools::core::toolOutcomeName(benchmarkOutcome(
                        child->exitCode, child->report.profile.authoritative)));
      yyjson_mut_obj_add_int(document.get(), row, "exitCode",
                             static_cast<int>(child->exitCode));
      const auto relative =
          portableRelativePath(child->reportPath, artifactDir);
      if (relative.has_value()) {
        addJsonString(document.get(), row, "report", pathToUtf8(*relative));
      } else {
        yyjson_mut_obj_add_null(document.get(), row, "report");
      }
      yyjson_mut_arr_add_val(variantRows, row);
    }
    yyjson_mut_obj_add_val(document.get(), group, "variants", variantRows);
    yyjson_mut_arr_add_val(groups, group);
  }
  yyjson_mut_obj_add_val(document.get(), root, "experimentGroups", groups);

  yyjson_mut_val *caseReports = yyjson_mut_arr(document.get());
  for (const BenchmarkRunResult &child : result.caseResults) {
    const auto relative = portableRelativePath(child.reportPath, artifactDir);
    if (!relative.has_value()) {
      continue;
    }
    const std::string path = pathToUtf8(*relative);
    yyjson_mut_arr_add_strncpy(document.get(), caseReports, path.data(),
                               path.size());
  }
  yyjson_mut_obj_add_val(document.get(), root, "caseReports", caseReports);

  size_t length = 0u;
  char *json = yyjson_mut_write_opts(document.get(), YYJSON_WRITE_PRETTY,
                                     nullptr, &length, nullptr);
  if (json == nullptr) {
    return Result<std::string, std::string>::makeError(
        "failed to serialize benchmark suite payload");
  }
  std::string payload(json, length);
  std::free(json);
  return Result<std::string, std::string>::makeResult(std::move(payload));
}

void addBenchmarkProfile(nuri::tools::core::ResultEnvelopeV2 &envelope,
                         const BenchmarkProfileInfo &profile) {
  if (profile.id.empty()) {
    return;
  }
  std::vector<std::string> incompatibilityReasons = profile.authorityBlockers;
  if (!profile.authoritative && incompatibilityReasons.empty()) {
    incompatibilityReasons.push_back("benchmark result is investigative");
  }
  envelope.profile = nuri::tools::core::ResultProfileV2{
      .id = profile.id,
      .compatible = profile.authoritative,
      .incompatibilityReasons = std::move(incompatibilityReasons)};
}

[[nodiscard]] Result<void, std::string> writeBenchmarkCaseEnvelope(
    const BenchmarkRunResult &result, const BenchmarkReport &report,
    const std::filesystem::path &artifactDir,
    const std::filesystem::path &envelopePath, std::string_view runId,
    bool verboseFrames, bool detailedReportComplete) {
  auto payload = writeBenchmarkReportJson(report, verboseFrames);
  if (payload.hasError()) {
    return Result<void, std::string>::makeError(payload.error());
  }

  const nuri::tools::core::ToolOutcome outcome =
      benchmarkOutcome(result.exitCode, report.profile.authoritative);
  nuri::tools::core::ResultEnvelopeV2 envelope{};
  envelope.tool = nuri::tools::core::ResultToolV2::Benchmark;
  envelope.runId = std::string(runId);
  envelope.status = outcome;
  envelope.exitCode = static_cast<int>(result.exitCode);
  envelope.authoritative = report.profile.authoritative;
  envelope.environmentFingerprint =
      benchmarkEnvironmentFingerprint(report.environment);
  envelope.workloadFingerprint =
      benchmarkWorkloadFingerprint(report.benchmarkCase);
  if (!report.generatedAtUtc.empty()) {
    envelope.startedAtUtc = report.generatedAtUtc;
  }
  if (!report.command.empty()) {
    envelope.reproduceCommand = report.command;
  }
  envelope.selection.requested = report.benchmarkCase.id;
  if (result.repetitions.empty()) {
    envelope.selection.selected = 1u;
    envelope.selection.attempted = 1u;
    envelope.selection.completed = 1u;
    envelope.selection.passed =
        outcome == nuri::tools::core::ToolOutcome::Pass ? 1u : 0u;
    envelope.selection.warned =
        outcome == nuri::tools::core::ToolOutcome::Investigative ? 1u : 0u;
    envelope.selection.failed =
        outcome == nuri::tools::core::ToolOutcome::Failure ||
                outcome == nuri::tools::core::ToolOutcome::Invalid ||
                outcome == nuri::tools::core::ToolOutcome::RuntimeError ||
                outcome == nuri::tools::core::ToolOutcome::MissingBaseline
            ? 1u
            : 0u;
    envelope.selection.unavailable =
        outcome == nuri::tools::core::ToolOutcome::EnvironmentUnavailable ? 1u
                                                                          : 0u;
  } else {
    envelope.selection.selected = result.repetitions.size();
    envelope.selection.attempted = result.repetitions.size();
    envelope.selection.completed = result.repetitions.size();
    for (const BenchmarkRepetitionResult &repetition : result.repetitions) {
      const nuri::tools::core::ToolOutcome childOutcome =
          benchmarkOutcome(repetition.exitCode, false);
      switch (childOutcome) {
      case nuri::tools::core::ToolOutcome::Pass:
        ++envelope.selection.passed;
        break;
      case nuri::tools::core::ToolOutcome::Investigative:
      case nuri::tools::core::ToolOutcome::Warn:
        ++envelope.selection.warned;
        break;
      case nuri::tools::core::ToolOutcome::EnvironmentUnavailable:
        ++envelope.selection.unavailable;
        break;
      default:
        ++envelope.selection.failed;
        break;
      }
    }
  }
  addBenchmarkProfile(envelope, report.profile);
  for (const std::string &warning : report.warnings) {
    envelope.diagnostics.push_back(
        {.code = "benchmark.warning",
         .severity = nuri::tools::core::ResultDiagnosticSeverityV2::Warning,
         .message = warning});
  }
  if (result.exitCode != BenchmarkExitCode::Success &&
      !result.message.empty()) {
    envelope.diagnostics.push_back(
        {.code = "benchmark.result",
         .severity = nuri::tools::core::ResultDiagnosticSeverityV2::Error,
         .message = result.message});
  }
  const auto addArtifact = [&](std::string role,
                               const std::filesystem::path &path,
                               std::string mediaType) {
    const auto relative = portableRelativePath(path, artifactDir);
    if (!relative.has_value() || !std::filesystem::is_regular_file(path)) {
      return;
    }
    auto digest = nuri::tools::core::sha256File(path);
    if (digest.hasError()) {
      return;
    }
    envelope.artifacts.push_back(
        {.role = std::move(role),
         .path = *relative,
         .mediaType = std::move(mediaType),
         .digest = "sha256:" + digest.value(),
         .status = nuri::tools::core::ResultArtifactStatusV2::Complete});
  };
  if (const auto relative =
          portableRelativePath(result.reportPath, artifactDir);
      result.repetitions.empty() && detailedReportComplete &&
      relative.has_value()) {
    nuri::tools::core::ResultChildV2 child{
        .id = report.benchmarkCase.id,
        .status = std::string(nuri::tools::core::toolOutcomeName(outcome)),
        .exitCode = static_cast<int>(result.exitCode),
        .result = *relative};
    envelope.children.push_back(std::move(child));
    addArtifact("benchmark.case.report", result.reportPath, "application/json");
  }
  if (!result.repetitions.empty()) {
    if (detailedReportComplete) {
      addArtifact("benchmark.aggregate.report", result.reportPath,
                  "application/json");
    }
    for (const BenchmarkRepetitionResult &repetition : result.repetitions) {
      const nuri::tools::core::ToolOutcome childOutcome =
          benchmarkOutcome(repetition.exitCode, false);
      nuri::tools::core::ResultChildV2 child{
          .id = "repetition-" + std::to_string(repetition.index + 1u),
          .status =
              std::string(nuri::tools::core::toolOutcomeName(childOutcome)),
          .exitCode = static_cast<int>(repetition.exitCode)};
      const std::filesystem::path childResult =
          std::filesystem::is_regular_file(repetition.envelopePath)
              ? repetition.envelopePath
              : repetition.reportPath;
      if (const auto relative = portableRelativePath(childResult, artifactDir);
          relative.has_value() &&
          std::filesystem::is_regular_file(childResult)) {
        child.result = *relative;
      }
      envelope.children.push_back(std::move(child));
      addArtifact("benchmark.repetition.report", repetition.reportPath,
                  "application/json");
      addArtifact("benchmark.repetition.result", repetition.envelopePath,
                  "application/json");
      addArtifact("benchmark.repetition.stdout", repetition.stdoutLogPath,
                  "text/plain; charset=utf-8");
      addArtifact("benchmark.repetition.stderr", repetition.stderrLogPath,
                  "text/plain; charset=utf-8");
    }
  }
  for (const std::filesystem::path &path : report.artifacts.rgpArtifacts) {
    const bool trace = path.extension() == ".rgp";
    addArtifact(
        trace ? "benchmark.rgp.shader-trace" : "benchmark.rgp.capture-log",
        path, trace ? "application/octet-stream" : "text/plain; charset=utf-8");
  }
  for (const std::filesystem::path &path :
       report.artifacts.renderDocArtifacts) {
    const std::string extension = path.extension().string();
    if (extension == ".rdc") {
      addArtifact("benchmark.renderdoc.capture", path,
                  "application/octet-stream");
    } else if (extension == ".json") {
      addArtifact("benchmark.renderdoc.chrome-trace", path, "application/json");
    } else if (extension == ".png") {
      addArtifact("benchmark.renderdoc.thumbnail", path, "image/png");
    } else {
      addArtifact("benchmark.renderdoc.log", path, "text/plain; charset=utf-8");
    }
  }
  envelope.payloadJson = std::move(payload.value());
  return nuri::tools::core::writeResultEnvelopeV2(envelopePath, envelope);
}

[[nodiscard]] BenchmarkRunResult
finalizeBenchmarkCaseResult(BenchmarkRunResult result, BenchmarkReport report,
                            const std::filesystem::path &artifactDir,
                            std::string_view runId, bool verboseFrames) {
  auto reportWrite =
      writeBenchmarkReportFile(report, result.reportPath, verboseFrames);
  if (reportWrite.hasError()) {
    result.exitCode = BenchmarkExitCode::RuntimeError;
    result.message = reportWrite.error();
    report.run.validForComparison = false;
    report.profile.authoritative = false;
    report.warnings.push_back(reportWrite.error());
  }
  auto envelopeWrite = writeBenchmarkCaseEnvelope(
      result, report, artifactDir, result.envelopePath, runId, verboseFrames,
      !reportWrite.hasError());
  if (envelopeWrite.hasError()) {
    result.exitCode = BenchmarkExitCode::RuntimeError;
    result.message = envelopeWrite.error();
    report.run.validForComparison = false;
    report.profile.authoritative = false;
    report.warnings.push_back(envelopeWrite.error());
    if (!reportWrite.hasError()) {
      (void)writeBenchmarkReportFile(report, result.reportPath, verboseFrames);
    }
  }
  result.report = std::move(report);
  return result;
}

[[nodiscard]] Result<void, std::string> writeBenchmarkSuiteEnvelope(
    const BenchmarkSuiteRunResult &result, std::string_view suite,
    const BenchmarkRunOptions &options,
    const std::filesystem::path &artifactDir,
    const std::filesystem::path &envelopePath, std::string_view runId) {
  auto payload = makeBenchmarkSuitePayload(result, suite, options, artifactDir);
  if (payload.hasError()) {
    return Result<void, std::string>::makeError(payload.error());
  }
  const bool authoritative =
      !result.caseResults.empty() &&
      std::all_of(result.caseResults.begin(), result.caseResults.end(),
                  [](const BenchmarkRunResult &child) {
                    return child.report.profile.authoritative;
                  });
  const nuri::tools::core::ToolOutcome outcome =
      benchmarkOutcome(result.exitCode, authoritative);
  nuri::tools::core::ResultEnvelopeV2 envelope{};
  envelope.tool = nuri::tools::core::ResultToolV2::Benchmark;
  envelope.runId = std::string(runId);
  envelope.status = outcome;
  envelope.exitCode = static_cast<int>(result.exitCode);
  envelope.authoritative = authoritative;
  std::vector<std::pair<std::string, std::string>> environmentFingerprints;
  std::vector<std::pair<std::string, std::string>> workloadFingerprints;
  for (const BenchmarkRunResult &child : result.caseResults) {
    if (auto fingerprint =
            benchmarkEnvironmentFingerprint(child.report.environment)) {
      environmentFingerprints.emplace_back(child.report.benchmarkCase.id,
                                           std::move(*fingerprint));
    }
    if (auto fingerprint =
            benchmarkWorkloadFingerprint(child.report.benchmarkCase)) {
      workloadFingerprints.emplace_back(child.report.benchmarkCase.id,
                                        std::move(*fingerprint));
    }
  }
  envelope.environmentFingerprint = aggregateFingerprint(
      "benchmark.suite.environment", environmentFingerprints);
  envelope.workloadFingerprint =
      aggregateFingerprint("benchmark.suite.workload", workloadFingerprints);
  envelope.reproduceCommand = options.command.empty()
                                  ? std::optional<std::string>{}
                                  : std::optional<std::string>{options.command};
  envelope.selection.requested = std::string(suite);
  envelope.selection.selected = result.caseResults.size();
  envelope.selection.attempted = result.caseResults.size();
  envelope.selection.completed = result.caseResults.size();
  if (!options.baselineProfileId.empty()) {
    std::vector<std::string> blockers;
    for (const BenchmarkRunResult &child : result.caseResults) {
      blockers.insert(blockers.end(),
                      child.report.profile.authorityBlockers.begin(),
                      child.report.profile.authorityBlockers.end());
    }
    std::sort(blockers.begin(), blockers.end());
    blockers.erase(std::unique(blockers.begin(), blockers.end()),
                   blockers.end());
    if (!authoritative && blockers.empty()) {
      blockers.push_back("suite contains investigative benchmark results");
    }
    envelope.profile = nuri::tools::core::ResultProfileV2{
        .id = options.baselineProfileId,
        .compatible = authoritative,
        .incompatibilityReasons = std::move(blockers)};
  }
  for (const BenchmarkRunResult &child : result.caseResults) {
    const nuri::tools::core::ToolOutcome childOutcome =
        benchmarkOutcome(child.exitCode, child.report.profile.authoritative);
    switch (childOutcome) {
    case nuri::tools::core::ToolOutcome::Pass:
      ++envelope.selection.passed;
      break;
    case nuri::tools::core::ToolOutcome::Investigative:
    case nuri::tools::core::ToolOutcome::Warn:
      ++envelope.selection.warned;
      break;
    case nuri::tools::core::ToolOutcome::EnvironmentUnavailable:
      ++envelope.selection.unavailable;
      break;
    default:
      ++envelope.selection.failed;
      break;
    }
    nuri::tools::core::ResultChildV2 summary{
        .id = child.report.benchmarkCase.id,
        .status = std::string(nuri::tools::core::toolOutcomeName(childOutcome)),
        .exitCode = static_cast<int>(child.exitCode)};
    if (const auto relative =
            portableRelativePath(child.envelopePath, artifactDir);
        relative.has_value()) {
      summary.result = *relative;
      auto digest = nuri::tools::core::sha256File(child.envelopePath);
      if (!digest.hasError()) {
        envelope.artifacts.push_back(
            {.role = "benchmark.case.result",
             .path = *relative,
             .mediaType = "application/json",
             .digest = "sha256:" + digest.value(),
             .status = nuri::tools::core::ResultArtifactStatusV2::Complete});
      }
    }
    envelope.children.push_back(std::move(summary));
  }
  envelope.diagnostics.push_back(
      {.code = "benchmark.suite.summary",
       .severity = result.exitCode == BenchmarkExitCode::Success
                       ? nuri::tools::core::ResultDiagnosticSeverityV2::Info
                       : nuri::tools::core::ResultDiagnosticSeverityV2::Error,
       .message = result.message});
  envelope.payloadJson = std::move(payload.value());
  return nuri::tools::core::writeResultEnvelopeV2(envelopePath, envelope);
}

#if defined(_WIN32)
constexpr char kPathListSeparator = ';';
constexpr std::string_view kExeSuffix = ".exe";
#else
constexpr char kPathListSeparator = ':';
constexpr std::string_view kExeSuffix = "";
#endif

[[nodiscard]] double bytesToMiB(uint64_t bytes) {
  return static_cast<double>(bytes) / kBytesPerMiB;
}

[[nodiscard]] std::string metricSlug(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  bool previousUnderscore = false;
  for (const char c : text) {
    const unsigned char uc = static_cast<unsigned char>(c);
    const bool alnum = (uc >= '0' && uc <= '9') || (uc >= 'a' && uc <= 'z') ||
                       (uc >= 'A' && uc <= 'Z');
    if (alnum) {
      out.push_back(static_cast<char>(std::tolower(uc)));
      previousUnderscore = false;
      continue;
    }
    if (!previousUnderscore && !out.empty()) {
      out.push_back('_');
      previousUnderscore = true;
    }
  }
  while (!out.empty() && out.back() == '_') {
    out.pop_back();
  }
  if (out.empty()) {
    return "unnamed_pass";
  }
  return out;
}

[[nodiscard]] std::string quoteCommandArg(const std::filesystem::path &path) {
  std::string text = path.string();
  std::string out = "\"";
  for (const char c : text) {
    if (c == '"') {
      out += "\\\"";
    } else {
      out.push_back(c);
    }
  }
  out += "\"";
  return out;
}

[[nodiscard]] std::filesystem::path
executablePathInDir(const std::filesystem::path &dir, std::string_view name) {
  std::filesystem::path candidate = dir / std::string(name);
  if (!kExeSuffix.empty() && candidate.extension() != kExeSuffix) {
    candidate += std::string(kExeSuffix);
  }
  return candidate;
}

[[nodiscard]] std::optional<std::filesystem::path>
findExecutableInPath(std::string_view name) {
  const std::string pathValue = readProcessEnvironment("PATH");
  size_t begin = 0u;
  while (begin <= pathValue.size()) {
    const size_t end = pathValue.find(kPathListSeparator, begin);
    const std::string_view entry =
        end == std::string::npos
            ? std::string_view(pathValue).substr(begin)
            : std::string_view(pathValue).substr(begin, end - begin);
    if (!entry.empty()) {
      std::filesystem::path candidate =
          executablePathInDir(std::filesystem::path(entry), name);
      if (std::filesystem::exists(candidate)) {
        return candidate;
      }
    }
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1u;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::filesystem::path>
findTracyToolInRoot(const std::filesystem::path &root, std::string_view name) {
  if (root.empty() || !std::filesystem::exists(root)) {
    return std::nullopt;
  }
  const std::string fileName = std::string(name) + std::string(kExeSuffix);
  std::error_code ec;
  for (const auto &entry : std::filesystem::recursive_directory_iterator(
           root, std::filesystem::directory_options::skip_permission_denied,
           ec)) {
    if (ec) {
      break;
    }
    if (!entry.is_regular_file(ec) || ec) {
      ec.clear();
      continue;
    }
    if (entry.path().filename() == fileName &&
        entry.path().generic_string().find("/tools/tracy/") !=
            std::string::npos) {
      return entry.path();
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::filesystem::path>
findTracyTool(std::string_view name) {
  if (auto fromPath = findExecutableInPath(name); fromPath.has_value()) {
    return fromPath;
  }
  if (const std::string vcpkgRoot = readProcessEnvironment("VCPKG_ROOT");
      !vcpkgRoot.empty()) {
    if (auto fromVcpkg =
            findTracyToolInRoot(std::filesystem::path(vcpkgRoot), name);
        fromVcpkg.has_value()) {
      return fromVcpkg;
    }
  }
  return findTracyToolInRoot(benchmarkRepoRoot() / "build", name);
}

[[nodiscard]] std::optional<std::filesystem::path>
findConfiguredTool(const std::filesystem::path &configuredPath,
                   std::string_view executable) {
  if (!configuredPath.empty()) {
    std::error_code error;
    const std::filesystem::path absolute =
        std::filesystem::absolute(configuredPath, error);
    const std::filesystem::path &candidate = error ? configuredPath : absolute;
    return std::filesystem::is_regular_file(candidate)
               ? std::optional<std::filesystem::path>(candidate)
               : std::nullopt;
  }
  return findExecutableInPath(executable);
}

[[nodiscard]] std::optional<std::filesystem::path>
findRgpTool(const std::filesystem::path &configuredPath) {
  return findConfiguredTool(configuredPath, "RadeonDeveloperPanelCLI");
}

[[nodiscard]] std::optional<std::filesystem::path>
findRenderDocTool(const std::filesystem::path &configuredPath) {
  if (auto tool = findConfiguredTool(configuredPath, "renderdoccmd");
      tool.has_value() || !configuredPath.empty()) {
    return tool;
  }
#if defined(_WIN32)
  const std::string programFiles = readProcessEnvironment("ProgramFiles");
  if (!programFiles.empty()) {
    const std::filesystem::path installed =
        std::filesystem::path(programFiles) / "RenderDoc" / "renderdoccmd.exe";
    if (std::filesystem::is_regular_file(installed)) {
      return installed;
    }
  }
#endif
  return std::nullopt;
}

[[nodiscard]] int
processExitCode(const nuri::tools::core::ProcessResult &result) {
  if (result.status != nuri::tools::core::ProcessStatus::Exited ||
      !result.exitCode.has_value()) {
    return -1;
  }
  return *result.exitCode;
}

void writeProcessLog(std::ostream &log, std::string_view displayCommand,
                     const nuri::tools::core::ProcessResult &result) {
  log << displayCommand << "\n\n";
  if (!result.standardOutput.empty()) {
    log << "stdout:\n" << result.standardOutput;
    if (result.standardOutput.back() != '\n') {
      log << '\n';
    }
  }
  if (!result.standardError.empty()) {
    log << "stderr:\n" << result.standardError;
    if (result.standardError.back() != '\n') {
      log << '\n';
    }
  }
  if (!result.errorMessage.empty()) {
    log << "process error: " << result.errorMessage << '\n';
  }
  if (result.exitCode.has_value()) {
    log << "exit code: " << *result.exitCode << "\n\n";
  } else {
    log << "process did not exit normally\n\n";
  }
}

[[nodiscard]] int
runCommandToLog(const nuri::tools::core::ProcessCommand &command,
                std::string_view displayCommand,
                const std::filesystem::path &logPath,
                const nuri::tools::core::ProcessOptions &options = {}) {
  if (!logPath.parent_path().empty()) {
    std::filesystem::create_directories(logPath.parent_path());
  }
  std::ofstream log(logPath, std::ios::binary);
  if (!log) {
    return -1;
  }
  const nuri::tools::core::ProcessResult result =
      nuri::tools::core::runProcess(command, options);
  writeProcessLog(log, displayCommand, result);
  return processExitCode(result);
}

[[nodiscard]] int
runCommandToFile(const nuri::tools::core::ProcessCommand &command,
                 std::string_view displayCommand,
                 const std::filesystem::path &outputPath,
                 const std::filesystem::path &logPath) {
  if (!outputPath.parent_path().empty()) {
    std::filesystem::create_directories(outputPath.parent_path());
  }
  if (!logPath.parent_path().empty()) {
    std::filesystem::create_directories(logPath.parent_path());
  }
  std::ofstream output(outputPath, std::ios::binary);
  std::ofstream log(logPath, std::ios::binary | std::ios::app);
  if (!output || !log) {
    return -1;
  }

  const nuri::tools::core::ProcessResult result =
      nuri::tools::core::runProcess(command);
  output << result.standardOutput;
  writeProcessLog(log, displayCommand, result);
  return processExitCode(result);
}

void addArtifactOnce(std::vector<std::filesystem::path> &artifacts,
                     const std::filesystem::path &path) {
  if (path.empty()) {
    return;
  }
  if (std::find(artifacts.begin(), artifacts.end(), path) == artifacts.end()) {
    artifacts.push_back(path);
  }
}

void trimLineEnd(std::string &line) {
  while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
    line.pop_back();
  }
}

[[nodiscard]] std::vector<std::string> splitCsvLine(std::string_view line) {
  std::vector<std::string> fields;
  std::string field;
  bool quoted = false;
  for (size_t i = 0u; i < line.size(); ++i) {
    const char c = line[i];
    if (c == '"') {
      if (quoted && i + 1u < line.size() && line[i + 1u] == '"') {
        field.push_back('"');
        ++i;
      } else {
        quoted = !quoted;
      }
      continue;
    }
    if (c == ',' && !quoted) {
      fields.push_back(std::move(field));
      field.clear();
      continue;
    }
    field.push_back(c);
  }
  fields.push_back(std::move(field));
  return fields;
}

[[nodiscard]] std::string cleanNumericField(std::string_view text) {
  std::string cleaned;
  cleaned.reserve(text.size());
  for (const char c : text) {
    const unsigned char uc = static_cast<unsigned char>(c);
    if (c != ',' && !std::isspace(uc)) {
      cleaned.push_back(c);
    }
  }
  return cleaned;
}

[[nodiscard]] uint64_t parseU64Field(std::string_view text) {
  std::string cleaned = cleanNumericField(text);
  if (cleaned.empty()) {
    return 0u;
  }
  char *end = nullptr;
  const unsigned long long value = std::strtoull(cleaned.c_str(), &end, 10);
  return end == cleaned.c_str() ? 0u : static_cast<uint64_t>(value);
}

[[nodiscard]] double parseDoubleField(std::string_view text) {
  std::string cleaned = cleanNumericField(text);
  if (cleaned.empty()) {
    return 0.0;
  }
  char *end = nullptr;
  const double value = std::strtod(cleaned.c_str(), &end);
  return end == cleaned.c_str() ? 0.0 : value;
}

void sortAndLimitTracyZones(std::vector<BenchmarkTracyZoneStats> &zones) {
  std::sort(zones.begin(), zones.end(),
            [](const BenchmarkTracyZoneStats &lhs,
               const BenchmarkTracyZoneStats &rhs) {
              return lhs.totalNs != rhs.totalNs ? lhs.totalNs > rhs.totalNs
                                                : lhs.name < rhs.name;
            });
  constexpr size_t kMaxTracyZoneRows = 80u;
  if (zones.size() > kMaxTracyZoneRows) {
    zones.resize(kMaxTracyZoneRows);
  }
}

[[nodiscard]] std::vector<BenchmarkTracyZoneStats>
readTracyZoneCsv(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return {};
  }

  std::string line;
  if (!std::getline(file, line)) {
    return {};
  }

  std::vector<BenchmarkTracyZoneStats> zones;
  while (std::getline(file, line)) {
    trimLineEnd(line);
    if (line.empty()) {
      continue;
    }
    std::vector<std::string> fields = splitCsvLine(line);
    if (fields.size() < 10u) {
      continue;
    }
    BenchmarkTracyZoneStats zone{};
    zone.name = std::move(fields[0]);
    zone.sourceFile = std::filesystem::path(fields[1]);
    zone.sourceLine = static_cast<uint32_t>(parseU64Field(fields[2]));
    zone.totalNs = parseU64Field(fields[3]);
    zone.totalPercent = parseDoubleField(fields[4]);
    zone.count = parseU64Field(fields[5]);
    zone.meanNs = parseDoubleField(fields[6]);
    zone.minNs = parseU64Field(fields[7]);
    zone.maxNs = parseU64Field(fields[8]);
    zone.stddevNs = parseDoubleField(fields[9]);
    zones.push_back(std::move(zone));
  }

  sortAndLimitTracyZones(zones);
  return zones;
}

[[nodiscard]] bool
tracyCsvExporterSupportsGpuEvents(const std::filesystem::path &exportTool) {
  const nuri::tools::core::ProcessResult result =
      nuri::tools::core::runProcess(nuri::tools::core::ProcessCommand{
          .executable = exportTool,
          .arguments = {"--help"},
      });
  const std::string help = result.standardOutput + result.standardError;
  return help.find("--gpu") != std::string::npos ||
         help.find("Report each gpu zone event") != std::string::npos;
}

[[nodiscard]] std::vector<BenchmarkTracyZoneStats>
readTracyGpuEventCsv(const std::filesystem::path &path,
                     uint64_t &outEventCount) {
  outEventCount = 0u;
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return {};
  }
  std::string line;
  if (!std::getline(file, line)) {
    return {};
  }
  trimLineEnd(line);
  const std::vector<std::string> header = splitCsvLine(line);
  const auto findColumn = [&header](std::string_view name) {
    const auto found = std::find(header.begin(), header.end(), name);
    return found == header.end()
               ? std::numeric_limits<size_t>::max()
               : static_cast<size_t>(std::distance(header.begin(), found));
  };
  const size_t nameColumn = findColumn("name");
  const size_t sourceColumn = findColumn("src_file");
  const size_t durationColumn = findColumn("GPU execution time");
  if (nameColumn == std::numeric_limits<size_t>::max() ||
      sourceColumn == std::numeric_limits<size_t>::max() ||
      durationColumn == std::numeric_limits<size_t>::max()) {
    return {};
  }
  const size_t requiredColumn =
      std::max({nameColumn, sourceColumn, durationColumn});
  std::map<std::pair<std::string, std::string>, std::vector<double>> samples;
  while (std::getline(file, line)) {
    trimLineEnd(line);
    if (line.empty()) {
      continue;
    }
    std::vector<std::string> fields = splitCsvLine(line);
    if (fields.size() <= requiredColumn) {
      continue;
    }
    const uint64_t durationNs = parseU64Field(fields[durationColumn]);
    if (fields[nameColumn].empty() || durationNs == 0u) {
      continue;
    }
    auto [sample, inserted] = samples.try_emplace(std::pair{
        std::move(fields[nameColumn]), std::move(fields[sourceColumn])});
    (void)inserted;
    sample->second.push_back(static_cast<double>(durationNs));
    ++outEventCount;
  }

  uint64_t allZonesTotalNs = 0u;
  std::vector<BenchmarkTracyZoneStats> zones;
  zones.reserve(samples.size());
  for (const auto &[key, durations] : samples) {
    auto statsResult = computeMetricStats(durations);
    if (statsResult.hasError()) {
      continue;
    }
    const MetricStats &stats = statsResult.value();
    BenchmarkTracyZoneStats zone{};
    zone.name = key.first;
    zone.sourceFile = std::filesystem::path(key.second);
    zone.count = stats.count;
    zone.totalNs = static_cast<uint64_t>(
        std::llround(stats.mean * static_cast<double>(stats.count)));
    zone.meanNs = stats.mean;
    zone.medianNs = static_cast<uint64_t>(std::llround(stats.median));
    zone.p95Ns = static_cast<uint64_t>(std::llround(stats.p95));
    zone.minNs = static_cast<uint64_t>(std::llround(stats.min));
    zone.maxNs = static_cast<uint64_t>(std::llround(stats.max));
    zone.stddevNs = stats.stddev;
    allZonesTotalNs += zone.totalNs;
    zones.push_back(std::move(zone));
  }
  for (BenchmarkTracyZoneStats &zone : zones) {
    zone.totalPercent = allZonesTotalNs == 0u
                            ? 0.0
                            : 100.0 * static_cast<double>(zone.totalNs) /
                                  static_cast<double>(allZonesTotalNs);
  }
  sortAndLimitTracyZones(zones);
  return zones;
}

struct TracyZoneEvent {
  std::string name{};
  std::filesystem::path sourceFile{};
  uint32_t sourceLine = 0u;
  uint64_t startNs = 0u;
  uint64_t durationNs = 0u;
  std::string thread{};
};

[[nodiscard]] std::vector<TracyZoneEvent>
readTracyEventCsv(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return {};
  }

  std::string line;
  if (!std::getline(file, line)) {
    return {};
  }

  std::vector<TracyZoneEvent> events;
  while (std::getline(file, line)) {
    trimLineEnd(line);
    if (line.empty()) {
      continue;
    }
    std::vector<std::string> fields = splitCsvLine(line);
    if (fields.size() < 6u) {
      continue;
    }
    TracyZoneEvent event{};
    event.name = std::move(fields[0]);
    event.sourceFile = std::filesystem::path(fields[1]);
    event.sourceLine = static_cast<uint32_t>(parseU64Field(fields[2]));
    event.startNs = parseU64Field(fields[3]);
    event.durationNs = parseU64Field(fields[4]);
    event.thread = std::move(fields[5]);
    if (!event.name.empty() && event.durationNs > 0u) {
      events.push_back(std::move(event));
    }
  }
  return events;
}

struct TracyFlameBuildNode {
  BenchmarkTracyFlameNode node{};
  std::map<std::string, uint32_t> childrenByKey{};
  std::vector<uint32_t> children{};
};

struct TracyFlameStackEntry {
  uint64_t endNs = 0u;
  uint32_t nodeIndex = UINT32_MAX;
  bool inFrame = false;
};

[[nodiscard]] uint64_t saturatingAdd(uint64_t lhs, uint64_t rhs) {
  if (std::numeric_limits<uint64_t>::max() - lhs < rhs) {
    return std::numeric_limits<uint64_t>::max();
  }
  return lhs + rhs;
}

[[nodiscard]] std::string tracyFlameKey(std::string_view name,
                                        const std::filesystem::path &sourceFile,
                                        uint32_t sourceLine) {
  return std::string(name) + '\x1f' + sourceFile.generic_string() + '\x1f' +
         std::to_string(sourceLine);
}

[[nodiscard]] uint32_t
findNearestFlameParent(const std::vector<TracyFlameStackEntry> &stack,
                       uint32_t fallback) {
  for (auto it = stack.rbegin(); it != stack.rend(); ++it) {
    if (it->nodeIndex != UINT32_MAX) {
      return it->nodeIndex;
    }
  }
  return fallback;
}

[[nodiscard]] uint32_t
flameChildIndex(std::vector<TracyFlameBuildNode> &nodes, uint32_t parentIndex,
                std::string_view name, std::string_view thread,
                const std::filesystem::path &sourceFile, uint32_t sourceLine) {
  TracyFlameBuildNode &parent = nodes[parentIndex];
  const std::string key = tracyFlameKey(name, sourceFile, sourceLine);
  if (auto it = parent.childrenByKey.find(key);
      it != parent.childrenByKey.end()) {
    return it->second;
  }

  const uint32_t childIndex = static_cast<uint32_t>(nodes.size());
  parent.childrenByKey.emplace(key, childIndex);
  parent.children.push_back(childIndex);
  TracyFlameBuildNode child{};
  child.node.name = std::string(name);
  child.node.thread = std::string(thread);
  child.node.sourceFile = sourceFile;
  child.node.sourceLine = sourceLine;
  nodes.push_back(std::move(child));
  return childIndex;
}

void sortFlameChildren(std::vector<TracyFlameBuildNode> &nodes,
                       uint32_t nodeIndex) {
  std::vector<uint32_t> &children = nodes[nodeIndex].children;
  std::sort(children.begin(), children.end(), [&](uint32_t lhs, uint32_t rhs) {
    const BenchmarkTracyFlameNode &lhsNode = nodes[lhs].node;
    const BenchmarkTracyFlameNode &rhsNode = nodes[rhs].node;
    if (lhsNode.totalNs != rhsNode.totalNs) {
      return lhsNode.totalNs > rhsNode.totalNs;
    }
    return lhsNode.name < rhsNode.name;
  });
  for (const uint32_t child : children) {
    sortFlameChildren(nodes, child);
  }
}

uint64_t finalizeFlameNode(std::vector<TracyFlameBuildNode> &nodes,
                           uint32_t nodeIndex) {
  uint64_t childTotalNs = 0u;
  for (const uint32_t child : nodes[nodeIndex].children) {
    childTotalNs = saturatingAdd(childTotalNs, finalizeFlameNode(nodes, child));
  }
  BenchmarkTracyFlameNode &node = nodes[nodeIndex].node;
  if (node.count == 0u) {
    node.totalNs = childTotalNs;
  }
  node.selfNs = node.totalNs > childTotalNs ? node.totalNs - childTotalNs : 0u;
  return node.totalNs;
}

BenchmarkTracyFlameNode
copyFlameNode(const std::vector<TracyFlameBuildNode> &nodes, uint32_t nodeIndex,
              uint32_t depth, uint64_t minTotalNs, uint64_t &retainedNodeCount,
              uint32_t &maxDepth) {
  BenchmarkTracyFlameNode out = nodes[nodeIndex].node;
  out.children.clear();
  ++retainedNodeCount;
  maxDepth = std::max(maxDepth, depth);

  constexpr uint64_t kMaxRetainedFlameNodes = 900u;
  constexpr uint64_t kAlwaysKeepFlameNodes = 96u;
  for (const uint32_t child : nodes[nodeIndex].children) {
    if (retainedNodeCount >= kMaxRetainedFlameNodes) {
      break;
    }
    if (depth > 0u && retainedNodeCount >= kAlwaysKeepFlameNodes &&
        nodes[child].node.totalNs < minTotalNs) {
      continue;
    }
    out.children.push_back(copyFlameNode(nodes, child, depth + 1u, minTotalNs,
                                         retainedNodeCount, maxDepth));
  }
  return out;
}

[[nodiscard]] BenchmarkTracyFlameGraph
buildTracyFlameGraph(std::vector<TracyZoneEvent> events) {
  BenchmarkTracyFlameGraph flameGraph{};
  flameGraph.eventCount = events.size();
  if (events.empty()) {
    return flameGraph;
  }

  std::sort(events.begin(), events.end(),
            [](const TracyZoneEvent &lhs, const TracyZoneEvent &rhs) {
              if (lhs.thread != rhs.thread) {
                return lhs.thread < rhs.thread;
              }
              if (lhs.startNs != rhs.startNs) {
                return lhs.startNs < rhs.startNs;
              }
              if (lhs.durationNs != rhs.durationNs) {
                return lhs.durationNs > rhs.durationNs;
              }
              return lhs.name < rhs.name;
            });

  const bool frameScoped = std::find_if(events.begin(), events.end(),
                                        [](const TracyZoneEvent &event) {
                                          return event.name == "BenchmarkFrame";
                                        }) != events.end();

  std::vector<TracyFlameBuildNode> nodes;
  nodes.reserve(std::min<size_t>(events.size(), 4096u));
  nodes.push_back(TracyFlameBuildNode{});
  nodes[0].node.name =
      frameScoped ? "BenchmarkFrame stacks" : "All Tracy CPU zones";

  std::string activeThread;
  uint32_t threadIndex = UINT32_MAX;
  std::vector<TracyFlameStackEntry> stack;
  for (const TracyZoneEvent &event : events) {
    if (event.thread != activeThread) {
      activeThread = event.thread;
      stack.clear();
      const std::string threadName =
          activeThread.empty() ? std::string("unnamed thread") : activeThread;
      threadIndex =
          flameChildIndex(nodes, 0u, threadName, activeThread, {}, 0u);
    }

    const uint64_t endNs = saturatingAdd(event.startNs, event.durationNs);
    while (!stack.empty() && event.startNs >= stack.back().endNs) {
      stack.pop_back();
    }
    while (!stack.empty() && endNs > stack.back().endNs) {
      stack.pop_back();
    }

    const bool parentInFrame = !stack.empty() && stack.back().inFrame;
    const bool isFrame = event.name == "BenchmarkFrame";
    const bool include = !frameScoped || isFrame || parentInFrame;
    uint32_t nodeIndex = UINT32_MAX;
    if (include) {
      const uint32_t parentIndex =
          isFrame ? threadIndex : findNearestFlameParent(stack, threadIndex);
      nodeIndex = flameChildIndex(nodes, parentIndex, event.name, event.thread,
                                  event.sourceFile, event.sourceLine);
      BenchmarkTracyFlameNode &node = nodes[nodeIndex].node;
      node.totalNs = saturatingAdd(node.totalNs, event.durationNs);
      ++node.count;
    }
    stack.push_back(TracyFlameStackEntry{
        .endNs = endNs,
        .nodeIndex = nodeIndex,
        .inFrame = parentInFrame || isFrame,
    });
  }

  finalizeFlameNode(nodes, 0u);
  sortFlameChildren(nodes, 0u);
  const uint64_t minTotalNs =
      std::max<uint64_t>(nodes[0].node.totalNs / 2000u, 1'000u);
  flameGraph.frameScoped = frameScoped;
  flameGraph.root =
      copyFlameNode(nodes, 0u, 0u, minTotalNs, flameGraph.retainedNodeCount,
                    flameGraph.maxDepth);
  return flameGraph;
}

void readTracyCaptureSummary(BenchmarkTracyReport &tracy) {
  std::ifstream file(tracy.captureLogPath, std::ios::binary);
  if (!file) {
    return;
  }
  std::string line;
  while (std::getline(file, line)) {
    trimLineEnd(line);
    if (line.starts_with("Frames:")) {
      tracy.captureFrameCount = parseU64Field(line.substr(7u));
    } else if (line.starts_with("Time span:")) {
      tracy.captureTimeSpanSeconds = parseDoubleField(line.substr(10u));
    } else if (line.starts_with("Zones:")) {
      tracy.captureZoneEventCount = parseU64Field(line.substr(6u));
    }
  }
}

struct TracyCaptureSession {
  std::filesystem::path tracePath{};
  std::filesystem::path logPath{};
  std::string command{};
  std::future<int> exitCode{};
  bool started = false;

  void finish(BenchmarkReport &report) {
    if (!started) {
      return;
    }
    started = false;
    const int code = exitCode.valid() ? exitCode.get() : -1;
    addArtifactOnce(report.artifacts.tracyArtifacts, tracePath);
    addArtifactOnce(report.artifacts.tracyArtifacts, logPath);
    report.tracy.tracePath = tracePath;
    report.tracy.captureLogPath = logPath;
    report.tracy.captureCommand = command;
    readTracyCaptureSummary(report.tracy);
    if (code != 0) {
      report.warnings.push_back("tracy-capture exited with code " +
                                std::to_string(code));
      return;
    }
    if (!std::filesystem::exists(tracePath)) {
      report.warnings.push_back("tracy-capture did not produce a trace file");
      return;
    }

    report.tracy.available = true;
    const std::optional<std::filesystem::path> exportTool =
        findTracyTool("tracy-csvexport");
    if (!exportTool.has_value()) {
      report.warnings.push_back(
          "Tracy trace captured but tracy-csvexport was not found");
      return;
    }

    const std::filesystem::path zonesCsvPath =
        tracePath.parent_path() / (tracePath.stem().string() + ".zones.csv");
    const std::filesystem::path selfZonesCsvPath =
        tracePath.parent_path() /
        (tracePath.stem().string() + ".zones_self.csv");
    const std::filesystem::path eventsCsvPath =
        tracePath.parent_path() / (tracePath.stem().string() + ".events.csv");
    const std::filesystem::path exportLogPath =
        tracePath.parent_path() / (tracePath.stem().string() + ".export.log");

    const std::filesystem::path absoluteTracePath =
        std::filesystem::absolute(tracePath);
    report.tracy.zonesCsvPath = zonesCsvPath;
    report.tracy.selfZonesCsvPath = selfZonesCsvPath;
    report.tracy.exportLogPath = exportLogPath;
    report.tracy.flameGraph.eventsCsvPath = eventsCsvPath;
    report.tracy.gpuEventsExportSupported =
        tracyCsvExporterSupportsGpuEvents(*exportTool);
    report.tracy.zonesExportCommand =
        quoteCommandArg(*exportTool) + " " + quoteCommandArg(absoluteTracePath);
    report.tracy.selfZonesExportCommand = quoteCommandArg(*exportTool) +
                                          " -e " +
                                          quoteCommandArg(absoluteTracePath);
    report.tracy.flameGraph.eventsExportCommand =
        quoteCommandArg(*exportTool) + " -u " +
        quoteCommandArg(absoluteTracePath);

    const nuri::tools::core::ProcessCommand zonesCommand{
        .executable = *exportTool,
        .arguments = {pathToUtf8(absoluteTracePath)}};
    const int zonesCode =
        runCommandToFile(zonesCommand, report.tracy.zonesExportCommand,
                         zonesCsvPath, exportLogPath);
    if (zonesCode == 0) {
      report.tracy.zones = readTracyZoneCsv(zonesCsvPath);
      addArtifactOnce(report.artifacts.tracyArtifacts, zonesCsvPath);
    } else {
      report.warnings.push_back("tracy-csvexport aggregate export exited with "
                                "code " +
                                std::to_string(zonesCode));
    }

    const nuri::tools::core::ProcessCommand selfZonesCommand{
        .executable = *exportTool,
        .arguments = {"-e", pathToUtf8(absoluteTracePath)}};
    const int selfZonesCode =
        runCommandToFile(selfZonesCommand, report.tracy.selfZonesExportCommand,
                         selfZonesCsvPath, exportLogPath);
    if (selfZonesCode == 0) {
      report.tracy.selfZones = readTracyZoneCsv(selfZonesCsvPath);
      addArtifactOnce(report.artifacts.tracyArtifacts, selfZonesCsvPath);
    } else {
      report.warnings.push_back(
          "tracy-csvexport self-time export exited with code " +
          std::to_string(selfZonesCode));
    }

    const nuri::tools::core::ProcessCommand eventsCommand{
        .executable = *exportTool,
        .arguments = {"-u", pathToUtf8(absoluteTracePath)}};
    const int eventsCode = runCommandToFile(
        eventsCommand, report.tracy.flameGraph.eventsExportCommand,
        eventsCsvPath, exportLogPath);
    if (eventsCode == 0) {
      report.tracy.flameGraph =
          buildTracyFlameGraph(readTracyEventCsv(eventsCsvPath));
      report.tracy.flameGraph.eventsCsvPath = eventsCsvPath;
      report.tracy.flameGraph.eventsExportCommand =
          quoteCommandArg(*exportTool) + " -u " +
          quoteCommandArg(absoluteTracePath);
      addArtifactOnce(report.artifacts.tracyArtifacts, eventsCsvPath);
    } else {
      report.warnings.push_back(
          "tracy-csvexport event export exited with code " +
          std::to_string(eventsCode));
    }

    if (report.tracy.gpuEventsExportSupported) {
      const std::filesystem::path gpuEventsCsvPath =
          tracePath.parent_path() /
          (tracePath.stem().string() + ".gpu_events.csv");
      report.tracy.gpuEventsCsvPath = gpuEventsCsvPath;
      report.tracy.gpuEventsExportCommand = quoteCommandArg(*exportTool) +
                                            " -g " +
                                            quoteCommandArg(absoluteTracePath);
      const nuri::tools::core::ProcessCommand gpuEventsCommand{
          .executable = *exportTool,
          .arguments = {"-g", pathToUtf8(absoluteTracePath)}};
      const int gpuEventsCode = runCommandToFile(
          gpuEventsCommand, report.tracy.gpuEventsExportCommand,
          gpuEventsCsvPath, exportLogPath);
      if (gpuEventsCode == 0) {
        report.tracy.gpuZones = readTracyGpuEventCsv(
            gpuEventsCsvPath, report.tracy.gpuZoneEventCount);
        addArtifactOnce(report.artifacts.tracyArtifacts, gpuEventsCsvPath);
      } else {
        report.warnings.push_back(
            "tracy-csvexport GPU event export exited with code " +
            std::to_string(gpuEventsCode));
      }
    } else {
      report.warnings.push_back(
          "tracy-csvexport does not support GPU event CSV export; the raw "
          "trace remains available for Tracy GPU-zone inspection");
    }

    addArtifactOnce(report.artifacts.tracyArtifacts, exportLogPath);
    if (report.tracy.zones.empty() && report.tracy.selfZones.empty()) {
      report.warnings.push_back(
          "tracy-csvexport produced no aggregate zone stats");
    }
  }
};

[[nodiscard]] uint32_t tracyCaptureSeconds(const BenchmarkCase &benchmarkCase) {
  const double frameSeconds =
      static_cast<double>(benchmarkCase.warmupFrames +
                          benchmarkCase.measurementFrames +
                          benchmarkCase.cooldownFrames) *
      benchmarkCase.fixedDeltaSeconds;
  return static_cast<uint32_t>(
      std::clamp(std::ceil(frameSeconds) + 5.0, 5.0, 60.0));
}

[[nodiscard]] TracyCaptureSession
startTracyCaptureIfRequested(const BenchmarkCase &benchmarkCase,
                             const BenchmarkRunOptions &options,
                             BenchmarkReport &report) {
  TracyCaptureSession session{};
  if (!options.tracyDiagnostic) {
    return session;
  }
  if (!report.environment.tracyEnabled) {
    report.warnings.push_back(
        "Tracy diagnostic requested but NURI_WITH_TRACY is disabled");
    return session;
  }
  const std::optional<std::filesystem::path> captureTool =
      findTracyTool("tracy-capture");
  if (!captureTool.has_value()) {
    report.warnings.push_back(
        "Tracy diagnostic requested but tracy-capture was not found");
    return session;
  }

  const std::filesystem::path tracyDir = report.artifacts.artifactDir / "tracy";
  const std::string baseName =
      benchmarkCase.id + "_" + utcTimestampForPath() + "_tracy";
  session.tracePath = tracyDir / (baseName + ".tracy");
  session.logPath = tracyDir / (baseName + ".log");
  std::filesystem::create_directories(tracyDir);
  const uint32_t seconds = tracyCaptureSeconds(benchmarkCase);
  const std::filesystem::path traceOutputPath =
      std::filesystem::absolute(session.tracePath);
  session.command = quoteCommandArg(*captureTool) + " -o " +
                    quoteCommandArg(traceOutputPath) +
                    " -a 127.0.0.1 -p 8086 -s " + std::to_string(seconds) +
                    " -f";
  nuri::tools::core::ProcessCommand captureCommand{
      .executable = *captureTool,
      .arguments = {"-o", pathToUtf8(traceOutputPath), "-a", "127.0.0.1", "-p",
                    "8086", "-s", std::to_string(seconds), "-f"}};
  session.exitCode =
      std::async(std::launch::async,
                 [command = std::move(captureCommand),
                  displayCommand = session.command, logPath = session.logPath] {
                   return runCommandToLog(command, displayCommand, logPath);
                 });
  session.started = true;
  report.warnings.push_back("Tracy diagnostic capture is active; run is not "
                            "valid for authoritative comparison");
  return session;
}

struct RgpCaptureSession {
  std::filesystem::path tracePath{};
  std::filesystem::path logPath{};
  std::string command{};
  std::future<int> exitCode{};
  bool started = false;

  void finish(BenchmarkReport &report) {
    if (!started) {
      return;
    }
    started = false;
    const int code = exitCode.valid() ? exitCode.get() : -1;
    report.rgp.captureExitCode = code;
    report.rgp.tracePath = tracePath;
    report.rgp.captureLogPath = logPath;
    report.rgp.captureCommand = command;
    addArtifactOnce(report.artifacts.rgpArtifacts, logPath);
    std::ifstream log(logPath, std::ios::binary);
    std::string line;
    constexpr std::string_view counterPrefix =
        "Successfully processed derived counters for ";
    while (std::getline(log, line)) {
      const size_t counter = line.find(counterPrefix);
      if (counter != std::string::npos) {
        report.rgp.derivedCounterCount = static_cast<uint32_t>(parseU64Field(
            std::string_view(line).substr(counter + counterPrefix.size())));
      }
    }
    if (code != 0) {
      report.warnings.push_back("RGP shader diagnostic capture exited with "
                                "code " +
                                std::to_string(code));
      return;
    }
    std::error_code error;
    const uint64_t size = std::filesystem::file_size(tracePath, error);
    if (error || size == 0u) {
      report.warnings.push_back(
          "RGP shader diagnostic did not produce a non-empty trace");
      return;
    }
    report.rgp.available = true;
    report.rgp.traceSizeBytes = size;
    addArtifactOnce(report.artifacts.rgpArtifacts, tracePath);
    if (report.rgp.derivedCounterCount == 0u) {
      report.warnings.push_back(
          "RGP trace captured but no processed derived counters were reported; "
          "recapture before counter-based shader diagnosis");
    }
  }
};

[[nodiscard]] RgpCaptureSession
startRgpCaptureIfRequested(const BenchmarkCase &benchmarkCase,
                           const BenchmarkRunOptions &options,
                           BenchmarkReport &report) {
  RgpCaptureSession session{};
  const BenchmarkGpuDiagnosticOptions &diagnostic = options.gpuDiagnostic;
  const bool requested =
      diagnostic.kind == BenchmarkGpuDiagnosticKind::RgpShader;
  report.rgp.requested = requested;
  report.rgp.captureFrame = diagnostic.captureFrame;
  report.rgp.counterCollectionRequested = requested;
  if (!requested) {
    return session;
  }
  const std::optional<std::filesystem::path> tool =
      findRgpTool(diagnostic.toolPath);
  if (!tool.has_value()) {
    report.warnings.push_back(
        "RGP shader diagnostic requested but RadeonDeveloperPanelCLI was not "
        "found; pass --rgp-tool or add it to PATH");
    return session;
  }
  if (diagnostic.timeout <= std::chrono::milliseconds::zero()) {
    report.warnings.push_back(
        "RGP shader diagnostic timeout must be greater than zero");
    return session;
  }

  report.rgp.toolPath = *tool;
  const std::filesystem::path rgpDir = report.artifacts.artifactDir / "rgp";
  const std::string baseName =
      benchmarkCase.id + "_" + utcTimestampForPath() + "_shader";
  session.tracePath = rgpDir / (baseName + ".rgp");
  session.logPath = rgpDir / (baseName + ".log");
  std::filesystem::create_directories(rgpDir);
  const std::filesystem::path absoluteTracePath =
      std::filesystem::absolute(session.tracePath);
  std::string processFilter = options.processExecutable.stem().string();
  if (processFilter.empty()) {
    processFilter = "nuri-bench";
  }
  const std::string captureTrigger =
      "--rgp-auto-capture=frame:" + std::to_string(diagnostic.captureFrame);
  session.command = quoteCommandArg(*tool) + " -m profiling -p " +
                    quoteCommandArg(processFilter) + " -o " +
                    quoteCommandArg(absoluteTracePath) + " " + captureTrigger +
                    " --rgp-capture-mode frame --rgp-counter-collection "
                    "--rgp-sqtt-buffer-size default --verbose";
  nuri::tools::core::ProcessCommand command{
      .executable = *tool,
      .arguments = {"-m", "profiling", "-p", processFilter, "-o",
                    pathToUtf8(absoluteTracePath), captureTrigger,
                    "--rgp-capture-mode", "frame", "--rgp-counter-collection",
                    "--rgp-sqtt-buffer-size", "default", "--verbose"}};
  session.exitCode = std::async(
      std::launch::async,
      [command = std::move(command), displayCommand = session.command,
       logPath = session.logPath, timeout = diagnostic.timeout] {
        return runCommandToLog(command, displayCommand, logPath,
                               {.timeout = timeout});
      });
  session.started = true;
  report.warnings.push_back(
      "RGP capture is a shader diagnostic only; its counters, clocks, and "
      "timings are not overall GPU performance evidence");
  return session;
}

#if defined(_WIN32)
using RenderDocGenericFn = void (*)();
using RenderDocGetApiFn = int (*)(int, void **);
using RenderDocGetApiVersionFn = void (*)(int *, int *, int *);
using RenderDocGetNumCapturesFn = uint32_t (*)();
using RenderDocGetCaptureFn = uint32_t (*)(uint32_t, char *, uint32_t *,
                                           uint64_t *);
using RenderDocTriggerCaptureFn = void (*)();

// Stable prefix of RenderDoc's public RENDERDOC_API_1_0_0 function table.
struct RenderDocApiPrefix {
  RenderDocGetApiVersionFn getApiVersion = nullptr;
  RenderDocGenericFn unused[10]{};
  RenderDocGenericFn setCaptureFilePathTemplate = nullptr;
  RenderDocGenericFn getCaptureFilePathTemplate = nullptr;
  RenderDocGetNumCapturesFn getNumCaptures = nullptr;
  RenderDocGetCaptureFn getCapture = nullptr;
  RenderDocTriggerCaptureFn triggerCapture = nullptr;
};
#endif

struct RenderDocCaptureSession {
#if defined(_WIN32)
  RenderDocApiPrefix *api = nullptr;
#endif
  uint32_t initialCaptureCount = 0u;
  bool requested = false;
  bool triggered = false;
  bool finished = false;

  [[nodiscard]] bool ready() const noexcept {
#if defined(_WIN32)
    return api != nullptr;
#else
    return false;
#endif
  }

  void trigger(uint64_t frameIndex, BenchmarkReport &report) {
#if defined(_WIN32)
    if (api != nullptr && !triggered &&
        frameIndex == report.renderDoc.captureFrame) {
      api->triggerCapture();
      triggered = true;
      report.renderDoc.captureTriggered = true;
    }
#else
    (void)frameIndex;
    (void)report;
#endif
  }

  void finish(BenchmarkReport &report) {
    if (finished || !requested) {
      return;
    }
    finished = true;
#if defined(_WIN32)
    if (api == nullptr) {
      return;
    }
    if (!triggered) {
      report.warnings.push_back(
          "RenderDoc capture frame was not reached by the diagnostic run");
      return;
    }
    for (uint32_t attempt = 0u;
         api->getNumCaptures() <= initialCaptureCount && attempt < 100u;
         ++attempt) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    const uint32_t captureCount = api->getNumCaptures();
    if (captureCount <= initialCaptureCount) {
      report.warnings.push_back(
          "RenderDoc did not publish a capture after the requested frame");
      return;
    }
    uint32_t pathLength = 0u;
    const uint32_t captureIndex = captureCount - 1u;
    if (api->getCapture(captureIndex, nullptr, &pathLength, nullptr) == 0u ||
        pathLength == 0u) {
      report.warnings.push_back("RenderDoc did not return the capture path");
      return;
    }
    std::string path(pathLength, '\0');
    if (api->getCapture(captureIndex, path.data(), &pathLength, nullptr) ==
        0u) {
      report.warnings.push_back("RenderDoc capture path query failed");
      return;
    }
    path.resize(std::char_traits<char>::length(path.c_str()));
    report.renderDoc.capturePath = std::filesystem::path(path);
    std::error_code error;
    report.renderDoc.captureSizeBytes =
        std::filesystem::file_size(report.renderDoc.capturePath, error);
    if (error || report.renderDoc.captureSizeBytes == 0u) {
      report.warnings.push_back(
          "RenderDoc did not produce a non-empty capture file");
      return;
    }
    report.renderDoc.available = true;
    addArtifactOnce(report.artifacts.renderDocArtifacts,
                    report.renderDoc.capturePath);
#else
    report.warnings.push_back(
        "RenderDoc in-app capture is currently supported on Windows only");
#endif
  }
};

[[nodiscard]] RenderDocCaptureSession
startRenderDocCaptureIfRequested(const BenchmarkRunOptions &options,
                                 BenchmarkReport &report) {
  RenderDocCaptureSession session{};
  const BenchmarkGpuDiagnosticOptions &diagnostic = options.gpuDiagnostic;
  session.requested =
      diagnostic.kind == BenchmarkGpuDiagnosticKind::RenderDocFrame;
  report.renderDoc.requested = session.requested;
  report.renderDoc.captureFrame = diagnostic.captureFrame;
  if (!session.requested) {
    return session;
  }
#if defined(_WIN32)
  HMODULE module = GetModuleHandleA("renderdoc.dll");
  if (module == nullptr) {
    report.warnings.push_back(
        "RenderDoc diagnostic child was not injected with renderdoc.dll");
    return session;
  }
  const auto getApi = reinterpret_cast<RenderDocGetApiFn>(
      GetProcAddress(module, "RENDERDOC_GetAPI"));
  void *api = nullptr;
  constexpr int kRenderDocApiVersion100 = 10000;
  if (getApi == nullptr || getApi(kRenderDocApiVersion100, &api) == 0 ||
      api == nullptr) {
    report.warnings.push_back("RenderDoc in-app API 1.0.0 is unavailable");
    return session;
  }
  session.api = static_cast<RenderDocApiPrefix *>(api);
  if (session.api->getApiVersion == nullptr ||
      session.api->getNumCaptures == nullptr ||
      session.api->getCapture == nullptr ||
      session.api->triggerCapture == nullptr) {
    session.api = nullptr;
    report.warnings.push_back("RenderDoc in-app API table is incomplete");
    return session;
  }
  int major = 0;
  int minor = 0;
  int patch = 0;
  session.api->getApiVersion(&major, &minor, &patch);
  report.renderDoc.apiVersion = std::to_string(major) + "." +
                                std::to_string(minor) + "." +
                                std::to_string(patch);
  session.initialCaptureCount = session.api->getNumCaptures();
#endif
  return session;
}

[[nodiscard]] std::string passTimingMetricId(uint32_t orderedPassIndex,
                                             std::string_view passName,
                                             std::string_view suffix) {
  return std::format("rendergraph.pass.{:03}.{}.{}", orderedPassIndex,
                     metricSlug(passName), suffix);
}

[[nodiscard]] std::string_view
resolveTelemetryPassName(const RenderGraphTelemetrySnapshot &snapshot,
                         uint32_t passIndex) {
  if (passIndex >= snapshot.compile.passDebugNames.size()) {
    return "unnamed_pass";
  }
  const std::pmr::string &name = snapshot.compile.passDebugNames[passIndex];
  return name.empty() ? std::string_view("unnamed_pass")
                      : std::string_view(name.data(), name.size());
}

[[nodiscard]] bool
shouldIncludeGpuScopeInSum(const GpuTimingReport &report, GpuTimingScope scope,
                           uint64_t sourceFrameIndex) noexcept {
  if (scope == GpuTimingScope::WholeFrame) {
    return false;
  }
  const GpuTimingScope parent = gpuTimingParentScope(scope);
  return parent == GpuTimingScope::None || !hasGpuTimingScope(report, parent) ||
         gpuTimingScopeSourceFrame(report, parent) != sourceFrameIndex;
}

class TrackingMemoryResource final : public std::pmr::memory_resource {
public:
  explicit TrackingMemoryResource(
      std::pmr::memory_resource *upstream = std::pmr::get_default_resource())
      : upstream_(upstream) {}

  [[nodiscard]] uint64_t currentBytes() const noexcept { return currentBytes_; }
  [[nodiscard]] uint64_t peakBytes() const noexcept { return peakBytes_; }

private:
  void *do_allocate(size_t bytes, size_t alignment) override {
    void *ptr = upstream_->allocate(bytes, alignment);
    currentBytes_ += bytes;
    peakBytes_ = std::max(peakBytes_, currentBytes_);
    return ptr;
  }

  void do_deallocate(void *ptr, size_t bytes, size_t alignment) override {
    upstream_->deallocate(ptr, bytes, alignment);
    currentBytes_ -= bytes;
  }

  bool
  do_is_equal(const std::pmr::memory_resource &other) const noexcept override {
    return this == &other;
  }

  std::pmr::memory_resource *upstream_ = std::pmr::get_default_resource();
  uint64_t currentBytes_ = 0u;
  uint64_t peakBytes_ = 0u;
};

struct ProcessMemorySnapshot {
  bool available = false;
  uint64_t workingSetBytes = 0u;
  uint64_t peakWorkingSetBytes = 0u;
  uint64_t privateUsageBytes = 0u;
  uint64_t pagefileUsageBytes = 0u;
  uint64_t peakPagefileUsageBytes = 0u;
};

[[nodiscard]] ProcessMemorySnapshot collectProcessMemorySnapshot() {
  ProcessMemorySnapshot snapshot{};
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS_EX counters{};
  counters.cb = sizeof(counters);
  if (K32GetProcessMemoryInfo(
          GetCurrentProcess(),
          reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&counters),
          sizeof(counters)) == FALSE) {
    return snapshot;
  }
  snapshot.available = true;
  snapshot.workingSetBytes = static_cast<uint64_t>(counters.WorkingSetSize);
  snapshot.peakWorkingSetBytes =
      static_cast<uint64_t>(counters.PeakWorkingSetSize);
  snapshot.privateUsageBytes = static_cast<uint64_t>(counters.PrivateUsage);
  snapshot.pagefileUsageBytes = static_cast<uint64_t>(counters.PagefileUsage);
  snapshot.peakPagefileUsageBytes =
      static_cast<uint64_t>(counters.PeakPagefileUsage);
#else
  struct rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) {
    return snapshot;
  }
  snapshot.available = true;
#if defined(__APPLE__)
  snapshot.peakWorkingSetBytes = static_cast<uint64_t>(usage.ru_maxrss);
#else
  snapshot.peakWorkingSetBytes = static_cast<uint64_t>(usage.ru_maxrss) * 1024u;
#endif
#endif
  return snapshot;
}

[[nodiscard]] BenchmarkMetricIndex
registeredMetricIndex(std::string_view id) noexcept {
  const std::optional<BenchmarkMetricIndex> index =
      findExactBenchmarkMetricIndex(id);
  NURI_ASSERT(index.has_value(), "Unknown exact benchmark metric: {}", id);
  return index.value_or(BenchmarkMetricIndex{});
}

#define NURI_BENCHMARK_METRIC(id)                                              \
  ([]() noexcept {                                                             \
    static const BenchmarkMetricIndex index = registeredMetricIndex(id);       \
    return index;                                                              \
  }())

void appendCounter(BenchmarkFrameMeasurements &measurements,
                   BenchmarkMetricIndex index, uint64_t value) {
  // Renderer counters are published descriptors: zero is a valid observation,
  // while absence means the metric was unavailable.
  measurements.appendRegistered(index, static_cast<double>(value));
}

void appendValue(BenchmarkFrameMeasurements &measurements,
                 BenchmarkMetricIndex index, double value) {
  measurements.appendRegistered(index, value);
}

void appendBytesAsMiB(BenchmarkFrameMeasurements &measurements,
                      BenchmarkMetricIndex index, uint64_t bytes) {
  measurements.appendRegistered(index, bytesToMiB(bytes));
}

void addTextureResourceMetrics(BenchmarkFrameMeasurements &measurements,
                               GPUDevice &gpu) {
  const TextureCacheTelemetry cache = Texture::cacheTelemetry();
  appendCounter(measurements,
                NURI_BENCHMARK_METRIC("texture.cache.native_hits"),
                cache.nativeHits);
  appendCounter(measurements,
                NURI_BENCHMARK_METRIC("texture.cache.native_misses"),
                cache.nativeMisses);
  appendCounter(measurements,
                NURI_BENCHMARK_METRIC("texture.cache.native_stale"),
                cache.nativeStale);
  appendCounter(measurements,
                NURI_BENCHMARK_METRIC("texture.cache.native_corrupt"),
                cache.nativeCorrupt);
  appendCounter(measurements,
                NURI_BENCHMARK_METRIC("texture.cache.native_writes"),
                cache.nativeWrites);
  appendCounter(measurements,
                NURI_BENCHMARK_METRIC("texture.cache.native_write_failures"),
                cache.nativeWriteFailures);
  appendCounter(measurements,
                NURI_BENCHMARK_METRIC("texture.cache.artifact_builds"),
                cache.artifactBuilds);
  appendCounter(
      measurements,
      NURI_BENCHMARK_METRIC("texture.cache.normal_variance_artifact_builds"),
      cache.normalVarianceArtifactBuilds);
  appendCounter(
      measurements,
      NURI_BENCHMARK_METRIC("texture.cache.normal_variance_clean_texels"),
      cache.normalVarianceCleanTexels);
  appendCounter(measurements,
                NURI_BENCHMARK_METRIC(
                    "texture.cache.normal_variance_toksvig_fallback_texels"),
                cache.normalVarianceToksvigFallbackTexels);
  appendCounter(measurements,
                NURI_BENCHMARK_METRIC(
                    "texture.cache.normal_variance_contract_rejections"),
                cache.normalVarianceContractRejections);
  appendBytesAsMiB(measurements,
                   NURI_BENCHMARK_METRIC("texture.io.authored_source_read_mb"),
                   cache.authoredSourceBytesRead);
  appendBytesAsMiB(measurements,
                   NURI_BENCHMARK_METRIC("texture.io.native_artifact_read_mb"),
                   cache.nativeArtifactBytesRead);
  appendBytesAsMiB(
      measurements,
      NURI_BENCHMARK_METRIC("texture.io.normal_variance_artifact_write_mb"),
      cache.normalVarianceArtifactBytesWritten);
  appendBytesAsMiB(measurements,
                   NURI_BENCHMARK_METRIC("texture.io.dds_source_read_mb"),
                   cache.ddsSourceBytesRead);
  measurements.appendRegistered(
      NURI_BENCHMARK_METRIC("texture.artifact_build_ms"),
      static_cast<double>(cache.artifactBuildTimeNs) / 1'000'000.0);
  measurements.appendRegistered(
      NURI_BENCHMARK_METRIC("texture.normal_variance_artifact_build_ms"),
      static_cast<double>(cache.normalVarianceArtifactBuildTimeNs) /
          1'000'000.0);
  measurements.appendRegistered(NURI_BENCHMARK_METRIC("texture.dds_read_ms"),
                                static_cast<double>(cache.ddsReadTimeNs) /
                                    1'000'000.0);

  const DdsTexturePackTelemetry pack = ddsTexturePackTelemetry();
  appendCounter(measurements, NURI_BENCHMARK_METRIC("texture.pack.hits"),
                pack.hits);
  appendCounter(measurements, NURI_BENCHMARK_METRIC("texture.pack.misses"),
                pack.misses);
  appendCounter(measurements, NURI_BENCHMARK_METRIC("texture.pack.stale"),
                pack.stale);
  appendCounter(measurements, NURI_BENCHMARK_METRIC("texture.pack.corrupt"),
                pack.corrupt);
  appendCounter(measurements, NURI_BENCHMARK_METRIC("texture.pack.builds"),
                pack.builds);
  appendCounter(measurements,
                NURI_BENCHMARK_METRIC("texture.pack.build_failures"),
                pack.buildFailures);
  appendCounter(measurements,
                NURI_BENCHMARK_METRIC("texture.pack.read_failures"),
                pack.readFailures);
  appendCounter(measurements,
                NURI_BENCHMARK_METRIC("texture.pack.entries_served"),
                pack.entriesServed);
  appendBytesAsMiB(measurements,
                   NURI_BENCHMARK_METRIC("texture.io.pack_bytes_served_mb"),
                   pack.bytesServed);
  appendBytesAsMiB(
      measurements,
      NURI_BENCHMARK_METRIC("texture.io.pack_build_source_read_mb"),
      pack.buildSourceBytesRead);
  measurements.appendRegistered(NURI_BENCHMARK_METRIC("texture.pack.build_ms"),
                                static_cast<double>(pack.buildTimeNs) /
                                    1'000'000.0);
  measurements.appendRegistered(NURI_BENCHMARK_METRIC("texture.pack.open_ms"),
                                static_cast<double>(pack.openTimeNs) /
                                    1'000'000.0);
  measurements.appendRegistered(NURI_BENCHMARK_METRIC("texture.pack.read_ms"),
                                static_cast<double>(pack.readTimeNs) /
                                    1'000'000.0);

  const TextureUploadTelemetry uploads = gpu.getTextureUploadTelemetry();
  appendCounter(measurements,
                NURI_BENCHMARK_METRIC("texture.upload.textures_recorded"),
                uploads.texturesRecorded);
  appendBytesAsMiB(measurements,
                   NURI_BENCHMARK_METRIC("texture.upload.recorded_mb"),
                   uploads.bytesRecorded);
  appendCounter(measurements,
                NURI_BENCHMARK_METRIC("texture.upload.batches_submitted"),
                uploads.batchesSubmitted);
  appendCounter(measurements,
                NURI_BENCHMARK_METRIC("texture.upload.bounded_batch_flushes"),
                uploads.boundedBatchFlushes);
  appendCounter(measurements,
                NURI_BENCHMARK_METRIC("texture.upload.completion_waits"),
                uploads.completionWaits);
}

#define addIfNonzero(measurements, id, value)                                  \
  appendCounter(measurements, NURI_BENCHMARK_METRIC(id), value)
#define addBytesAsMiB(measurements, id, value)                                 \
  appendBytesAsMiB(measurements, NURI_BENCHMARK_METRIC(id), value)

void addProcessMemoryMetrics(BenchmarkFrameMeasurements &measurements) {
  const ProcessMemorySnapshot snapshot = collectProcessMemorySnapshot();
  if (!snapshot.available) {
    return;
  }
  addBytesAsMiB(measurements, "memory.process.working_set_mb",
                snapshot.workingSetBytes);
  addBytesAsMiB(measurements, "memory.process.peak_working_set_mb",
                snapshot.peakWorkingSetBytes);
  addBytesAsMiB(measurements, "memory.process.private_usage_mb",
                snapshot.privateUsageBytes);
  addBytesAsMiB(measurements, "memory.process.pagefile_usage_mb",
                snapshot.pagefileUsageBytes);
  addBytesAsMiB(measurements, "memory.process.peak_pagefile_usage_mb",
                snapshot.peakPagefileUsageBytes);
}

void addPmrMemoryMetrics(BenchmarkFrameMeasurements &measurements,
                         const TrackingMemoryResource &rendererMemory,
                         const TrackingMemoryResource &pipelineMemory,
                         const TrackingMemoryResource &sceneMemory) {
  addBytesAsMiB(measurements, "memory.pmr.renderer_current_mb",
                rendererMemory.currentBytes());
  addBytesAsMiB(measurements, "memory.pmr.renderer_peak_mb",
                rendererMemory.peakBytes());
  addBytesAsMiB(measurements, "memory.pmr.pipeline_current_mb",
                pipelineMemory.currentBytes());
  addBytesAsMiB(measurements, "memory.pmr.pipeline_peak_mb",
                pipelineMemory.peakBytes());
  addBytesAsMiB(measurements, "memory.pmr.scene_current_mb",
                sceneMemory.currentBytes());
  addBytesAsMiB(measurements, "memory.pmr.scene_peak_mb",
                sceneMemory.peakBytes());
}

void addRendererFrameMetrics(BenchmarkFrameMeasurements &measurements,
                             const RenderFrameMetrics &metrics) {
  const RenderFrameMetrics::AssetStreamingFrameMetrics &assets = metrics.assets;
  addIfNonzero(measurements, "renderer.assets.cpu_completions",
               assets.cpuCompletions);
  addIfNonzero(measurements, "renderer.assets.cpu_workers", assets.cpuWorkers);
  addIfNonzero(measurements, "renderer.assets.cpu_active_worker_limit",
               assets.cpuActiveWorkerLimit);
  addIfNonzero(measurements, "renderer.assets.cpu_interactive_mode",
               assets.cpuInteractiveMode);
  addIfNonzero(measurements, "renderer.assets.cpu_queued_jobs",
               assets.cpuQueuedJobs);
  addIfNonzero(measurements, "renderer.assets.cpu_running_jobs",
               assets.cpuRunningJobs);
  addIfNonzero(measurements, "renderer.assets.cpu_running_io",
               assets.cpuRunningIo);
  addIfNonzero(measurements, "renderer.assets.cpu_running_decode",
               assets.cpuRunningDecode);
  addIfNonzero(measurements, "renderer.assets.cpu_running_cook",
               assets.cpuRunningCook);
  addIfNonzero(measurements, "renderer.assets.cpu_running_transcode",
               assets.cpuRunningTranscode);
  addIfNonzero(measurements, "renderer.assets.cpu_running_metadata",
               assets.cpuRunningMetadata);
  addIfNonzero(measurements, "renderer.assets.dedicated_copy_queue",
               assets.dedicatedCopyQueue);
  addIfNonzero(measurements, "renderer.assets.gpu_materialized",
               assets.gpuMaterialized);
  addIfNonzero(measurements, "renderer.assets.published", assets.published);
  addIfNonzero(measurements, "renderer.assets.cancelled", assets.cancelled);
  addIfNonzero(measurements, "renderer.assets.failed", assets.failed);
  addIfNonzero(measurements, "renderer.assets.scene_patches",
               assets.scenePatches);
  addIfNonzero(measurements, "renderer.assets.scene_commits",
               assets.sceneCommits);
  addIfNonzero(measurements, "renderer.assets.deferred_cpu_completions",
               assets.deferredCpuCompletions);
  addIfNonzero(measurements, "renderer.assets.publication_deadline_exceeded",
               assets.publicationDeadlineExceeded);
  appendValue(
      measurements,
      NURI_BENCHMARK_METRIC("renderer.assets.publication_main_thread_ms"),
      assets.publicationMainThreadMilliseconds);
  appendValue(
      measurements,
      NURI_BENCHMARK_METRIC("renderer.assets.publication_max_operation_ms"),
      assets.publicationMaxOperationMilliseconds);
  addIfNonzero(measurements, "renderer.assets.cpu_in_flight_bytes",
               assets.cpuInFlightBytes);
  addIfNonzero(measurements, "renderer.assets.upload_bytes",
               assets.uploadBytes);
  addIfNonzero(measurements, "renderer.assets.submitted_jobs",
               assets.submittedJobs);
  addIfNonzero(measurements, "renderer.assets.completed_jobs",
               assets.completedJobs);
  addIfNonzero(measurements, "renderer.assets.cancelled_jobs",
               assets.cancelledJobs);
  addIfNonzero(measurements, "renderer.assets.rejected_jobs",
               assets.rejectedJobs);

  const OpaqueFrameMetrics &opaque = metrics.opaque;
  addIfNonzero(measurements, "renderer.opaque.total_instances",
               opaque.totalInstances);
  addIfNonzero(measurements, "renderer.opaque.visible_instances",
               opaque.visibleInstances);
  addIfNonzero(measurements, "renderer.opaque.instanced_draws",
               opaque.instancedDraws);
  addIfNonzero(measurements, "renderer.opaque.indirect_draw_calls",
               opaque.indirectDrawCalls);
  addIfNonzero(measurements, "renderer.opaque.indirect_commands",
               opaque.indirectCommands);
  addIfNonzero(measurements, "renderer.opaque.compute_dispatches",
               opaque.computeDispatches);
  addIfNonzero(measurements, "renderer.opaque.depth_prepass_draws",
               opaque.depthPrepassDraws);
  addIfNonzero(measurements, "renderer.opaque.tessellated_draws",
               opaque.tessellatedDraws);
  addIfNonzero(measurements, "renderer.opaque.meshlet_dispatches",
               opaque.meshletDispatches);
  addIfNonzero(measurements, "renderer.opaque.meshlet_task_groups",
               opaque.meshletTaskGroups);
  addIfNonzero(measurements, "renderer.opaque.meshlet_candidates",
               opaque.meshletCandidateCount);
  addIfNonzero(measurements, "renderer.opaque.meshlet_mode_required",
               opaque.meshletModeRequired);
  addIfNonzero(measurements, "renderer.opaque.meshlet_mode_active",
               opaque.meshletModeActive);
  addIfNonzero(measurements, "renderer.opaque.meshlet_rejected_missing_feature",
               opaque.meshletRejectedMissingFeature);
  addIfNonzero(measurements,
               "renderer.opaque.meshlet_rejected_missing_asset_data",
               opaque.meshletRejectedMissingAssetData);
  addIfNonzero(measurements,
               "renderer.opaque.meshlet_rejected_incompatible_frame",
               opaque.meshletRejectedIncompatibleFrame);
  addIfNonzero(measurements, "renderer.opaque.meshlet_hybrid_active",
               opaque.meshletHybridActive);
  addIfNonzero(measurements, "renderer.opaque.meshlet_hybrid_classic_batches",
               opaque.meshletHybridClassicBatches);
  addIfNonzero(measurements, "renderer.opaque.meshlet_hybrid_classic_instances",
               opaque.meshletHybridClassicInstances);
  addIfNonzero(measurements,
               "renderer.opaque.meshlet_hybrid_coverage_classic_batches",
               opaque.meshletHybridCoverageClassicBatches);
  addIfNonzero(measurements,
               "renderer.opaque.meshlet_hybrid_coverage_classic_instances",
               opaque.meshletHybridCoverageClassicInstances);
  addIfNonzero(measurements, "renderer.opaque.meshlet_hybrid_meshlet_batches",
               opaque.meshletHybridMeshletBatches);
  addIfNonzero(measurements, "renderer.opaque.meshlet_hybrid_meshlet_instances",
               opaque.meshletHybridMeshletInstances);
  addIfNonzero(measurements, "renderer.opaque.auto_lod_active",
               opaque.autoLodActive);
  addIfNonzero(measurements, "renderer.opaque.auto_lod_history_reset",
               opaque.autoLodHistoryReset);
  addIfNonzero(measurements, "renderer.opaque.auto_lod_transitions",
               opaque.autoLodTransitions);
  addIfNonzero(measurements, "renderer.opaque.auto_lod_lod0_instances",
               opaque.autoLodLod0Instances);
  addIfNonzero(measurements, "renderer.opaque.auto_lod_lod1_instances",
               opaque.autoLodLod1Instances);
  addIfNonzero(measurements, "renderer.opaque.classic_main_draws",
               opaque.classicMainDraws);
  addIfNonzero(measurements, "renderer.opaque.classic_alpha_masked_main_draws",
               opaque.classicAlphaMaskedMainDraws);
  addIfNonzero(measurements, "renderer.opaque.meshlet_main_dispatches",
               opaque.meshletMainDispatches);
  addIfNonzero(measurements, "renderer.opaque.meshlet_main_represented_items",
               opaque.meshletMainRepresentedItems);
  addIfNonzero(measurements,
               "renderer.opaque.meshlet_alpha_masked_main_dispatches",
               opaque.meshletAlphaMaskedMainDispatches);
  addIfNonzero(measurements, "renderer.opaque.meshlet_alpha_masked_main_items",
               opaque.meshletAlphaMaskedMainItems);
  addIfNonzero(measurements, "renderer.opaque.msaa_depth_prepass_draws",
               opaque.msaaDepthPrepassDraws);
  addIfNonzero(measurements, "renderer.opaque.msaa_depth_prepass_dispatches",
               opaque.msaaDepthPrepassDispatches);
  addIfNonzero(measurements, "renderer.opaque.gtao_auxiliary_prepass_draws",
               opaque.gtaoAuxiliaryPrepassDraws);
  addIfNonzero(measurements,
               "renderer.opaque.gtao_auxiliary_prepass_dispatches",
               opaque.gtaoAuxiliaryPrepassDispatches);
  addIfNonzero(measurements,
               "renderer.opaque.gtao_auxiliary_writes_single_sample_depth",
               opaque.gtaoAuxiliaryWritesSingleSampleDepth);
  addIfNonzero(measurements, "renderer.opaque.main_equal_readonly_draws",
               opaque.mainEqualReadOnlyDraws);
  addIfNonzero(measurements, "renderer.opaque.main_equal_readonly_dispatches",
               opaque.mainEqualReadOnlyDispatches);
  addIfNonzero(measurements, "renderer.opaque.main_less_write_draws",
               opaque.mainLessWriteDraws);
  addIfNonzero(measurements, "renderer.opaque.main_less_write_dispatches",
               opaque.mainLessWriteDispatches);
  addIfNonzero(measurements, "renderer.opaque.depth_pyramid_requested",
               opaque.depthPyramidRequested);
  addIfNonzero(measurements, "renderer.opaque.depth_pyramid_active",
               opaque.depthPyramidActive);
  addIfNonzero(measurements, "renderer.visibility.hiz_requested",
               opaque.hiZRequested);
  addIfNonzero(measurements, "renderer.visibility.hiz_active",
               opaque.hiZActive);
  addIfNonzero(measurements, "renderer.visibility.hiz_source_frame_policy",
               static_cast<uint32_t>(opaque.hiZSourceFramePolicy));

  const VisibilityFrameMetrics &visibility = metrics.visibility;
  addIfNonzero(measurements, "renderer.visibility.cpu_main_candidates",
               visibility.cpuMainCandidates);
  addIfNonzero(measurements, "renderer.visibility.cpu_main_visible_candidates",
               visibility.cpuMainVisibleCandidates);
  addIfNonzero(measurements, "renderer.visibility.cpu_main_rejected",
               visibility.cpuMainRejected);
  addIfNonzero(measurements, "renderer.visibility.gpu_main_candidates",
               visibility.gpuMainCandidates);
  addIfNonzero(measurements, "renderer.visibility.gpu_main_visible_candidates",
               visibility.gpuMainVisibleCandidates);
  addIfNonzero(measurements, "renderer.visibility.gpu_main_rejected_frustum",
               visibility.gpuMainRejectedFrustum);
  addIfNonzero(measurements, "renderer.visibility.gpu_main_rejected_occlusion",
               visibility.gpuMainRejectedOcclusion);
  addIfNonzero(measurements, "renderer.visibility.gpu_output_overflow_count",
               visibility.gpuOutputOverflowCount);
  addIfNonzero(measurements, "renderer.visibility.gpu_main_readback_available",
               visibility.gpuMainReadbackAvailable);
  addIfNonzero(measurements,
               "renderer.visibility.gpu_main_readback_source_frame",
               visibility.gpuMainReadbackSourceFrame);
  addIfNonzero(measurements,
               "renderer.visibility.gpu_main_readback_stale_frame_count",
               visibility.gpuMainReadbackStaleFrameCount);
  addIfNonzero(measurements,
               "renderer.visibility.gpu_main_readback_error_count",
               visibility.gpuMainReadbackErrorCount);
  addIfNonzero(measurements,
               "renderer.visibility.gpu_main_readback_visible_candidates",
               visibility.gpuMainReadbackVisibleCandidates);
  addIfNonzero(measurements,
               "renderer.visibility.gpu_main_visible_list_mismatches",
               visibility.gpuMainVisibleListMismatches);
  addIfNonzero(measurements, "renderer.visibility.gpu_indirect_draw_used",
               visibility.gpuIndirectDrawUsed);
  addIfNonzero(measurements, "renderer.visibility.gpu_indirect_draw_fallback",
               visibility.gpuIndirectDrawFallback);
  addIfNonzero(measurements, "renderer.visibility.gpu_indirect_draw_commands",
               visibility.gpuIndirectDrawCommands);
  addIfNonzero(measurements,
               "renderer.visibility.gpu_indirect_draw_readback_commands",
               visibility.gpuIndirectDrawReadbackCommands);
  addIfNonzero(measurements,
               "renderer.visibility.gpu_indirect_draw_readback_tombstoned",
               visibility.gpuIndirectDrawReadbackTombstoned);
  addIfNonzero(measurements,
               "renderer.visibility.gpu_indirect_draw_readback_visible",
               visibility.gpuIndirectDrawReadbackVisible);
  addIfNonzero(measurements, "renderer.visibility.indirect_mesh_dispatch_count",
               visibility.indirectMeshDispatchCount);
  addIfNonzero(measurements, "renderer.visibility.meshlet_rejected_frustum",
               visibility.meshletRejectedFrustum);
  addIfNonzero(measurements, "renderer.visibility.meshlet_rejected_cone",
               visibility.meshletRejectedCone);
  addIfNonzero(measurements, "renderer.visibility.meshlet_rejected_occlusion",
               visibility.meshletRejectedOcclusion);
  addIfNonzero(measurements, "renderer.visibility.meshlet_occlusion_available",
               visibility.meshletOcclusionAvailable);
  addIfNonzero(measurements, "renderer.visibility.meshlet_occlusion_mode",
               visibility.meshletOcclusionMode);
  addIfNonzero(measurements,
               "renderer.visibility.meshlet_occlusion_source_frame",
               visibility.meshletOcclusionSourceFrame);
  addIfNonzero(measurements, "renderer.visibility.meshlet_occlusion_source_age",
               visibility.meshletOcclusionSourceAge);
  addIfNonzero(measurements, "renderer.visibility.current_frame_hiz_active",
               visibility.currentFrameHiZActive);
  addIfNonzero(measurements,
               "renderer.visibility.meshlet_pre_task_compaction_active",
               visibility.meshletPreTaskCompactionActive);
  addIfNonzero(measurements,
               "renderer.visibility.meshlet_pre_task_candidates_input",
               visibility.meshletPreTaskCandidatesInput);
  addIfNonzero(measurements,
               "renderer.visibility.meshlet_pre_task_candidates_output",
               visibility.meshletPreTaskCandidatesOutput);
  addIfNonzero(measurements,
               "renderer.visibility.meshlet_pre_task_task_groups_input",
               visibility.meshletPreTaskTaskGroupsInput);
  addIfNonzero(measurements,
               "renderer.visibility.meshlet_pre_task_task_groups_output",
               visibility.meshletPreTaskTaskGroupsOutput);
  addIfNonzero(measurements,
               "renderer.visibility.meshlet_pre_task_task_groups_saved",
               visibility.meshletPreTaskTaskGroupsSaved);
  addIfNonzero(measurements,
               "renderer.visibility.meshlet_pre_task_overflow_count",
               visibility.meshletPreTaskOverflowCount);
  addIfNonzero(measurements,
               "renderer.visibility.meshlet_pre_task_mismatch_count",
               visibility.meshletPreTaskMismatchCount);
  addIfNonzero(measurements,
               "renderer.visibility.meshlet_payload_overflow_count",
               visibility.meshletPayloadOverflowCount);
  addIfNonzero(measurements, "renderer.visibility.meshlet_readback_available",
               visibility.meshletReadbackAvailable);
  addIfNonzero(measurements,
               "renderer.visibility.meshlet_readback_source_frame",
               visibility.meshletReadbackSourceFrame);
  addIfNonzero(measurements,
               "renderer.visibility.meshlet_readback_stale_frame_count",
               visibility.meshletReadbackStaleFrameCount);
  addIfNonzero(measurements, "renderer.visibility.meshlet_readback_error_count",
               visibility.meshletReadbackErrorCount);
  addIfNonzero(measurements, "renderer.visibility.meshlet_emitted",
               visibility.meshletEmitted);
  addIfNonzero(measurements, "renderer.visibility.meshlet_task_groups_executed",
               visibility.meshletTaskGroupsExecuted);
  addIfNonzero(measurements, "renderer.visibility.uncertain_visible",
               visibility.uncertainVisible);
  addIfNonzero(measurements, "renderer.visibility.shadow_cpu_candidates",
               visibility.shadowCpuCandidates);
  addIfNonzero(measurements, "renderer.visibility.shadow_cpu_rejected",
               visibility.shadowCpuRejected);
  addIfNonzero(measurements, "renderer.visibility.occlusion_available",
               visibility.occlusionAvailable);

  const ShadowFrameMetrics &shadow = metrics.shadow;
  addIfNonzero(measurements, "renderer.shadow.cascades", shadow.cascadeCount);
  addIfNonzero(measurements, "renderer.shadow.total_draws", shadow.totalDraws);
  addIfNonzero(measurements, "renderer.shadow.total_culled_draws",
               shadow.totalCulledDraws);
  addIfNonzero(measurements, "renderer.shadow.static_caster_entries",
               shadow.staticCasterEntries);
  addIfNonzero(measurements, "renderer.shadow.dynamic_caster_entries",
               shadow.dynamicCasterEntries);
  addIfNonzero(measurements, "renderer.shadow.static_batch_templates",
               shadow.staticBatchTemplateCount);
  addIfNonzero(measurements, "renderer.shadow.batch_entries",
               shadow.shadowBatchEntryCount);
  addIfNonzero(measurements, "renderer.shadow.instance_remaps",
               shadow.shadowInstanceRemapCount);
  addIfNonzero(measurements, "renderer.shadow.total_index_count_estimate",
               shadow.totalIndexCountEstimate);
  addIfNonzero(measurements, "renderer.shadow.submitted_draw_items",
               shadow.submittedDrawItemCount);
  addIfNonzero(measurements, "renderer.shadow.indirect_commands",
               shadow.indirectCommandCount);
  addIfNonzero(measurements, "renderer.shadow.draw_packet_bytes",
               shadow.drawPacketBytes);
  addIfNonzero(measurements, "renderer.shadow.filter_sample_budget",
               shadow.filterSampleBudget);
  addIfNonzero(measurements, "renderer.shadow.frame_gpu_bytes",
               shadow.frameGpuBytes);
  addIfNonzero(measurements, "renderer.shadow.sdsm_compute_passes",
               shadow.sdsmComputePassCount);
  addBytesAsMiB(measurements, "gpu.memory.shadow.cascade_texture_mb",
                shadow.cascadeTextureBytes);

  const AntiAliasingFrameMetrics &aa = metrics.antiAliasing;
  const PostAAPlan &postAAPlan = aa.postAAPlan;
  const PostAAFrameFacts &postAA = aa.postAA;
  addIfNonzero(measurements, "renderer.aa.post_aa_requested",
               postAAPlan.requested);
  addIfNonzero(measurements, "renderer.aa.post_aa_resolved_active",
               postAAPlan.active);
  addIfNonzero(measurements, "renderer.aa.post_aa_inactive_reason",
               static_cast<uint32_t>(postAAPlan.inactiveReason));
  addIfNonzero(measurements, "renderer.aa.post_aa_specular_algorithm",
               static_cast<uint32_t>(postAAPlan.specular));
  addIfNonzero(measurements, "renderer.aa.post_aa_spatial_algorithm",
               static_cast<uint32_t>(postAAPlan.spatial));
  appendValue(
      measurements,
      NURI_BENCHMARK_METRIC("renderer.aa.post_aa_material_variance_scale"),
      postAAPlan.materialVarianceScale);
  appendValue(
      measurements,
      NURI_BENCHMARK_METRIC("renderer.aa.post_aa_geometric_variance_scale"),
      postAAPlan.geometricVarianceScale);
  appendValue(measurements,
              NURI_BENCHMARK_METRIC("renderer.aa.post_aa_max_slope_variance"),
              postAAPlan.maxSlopeVariance);
  addIfNonzero(measurements, "renderer.aa.post_aa_specular_selected",
               postAA.specularSelected);
  addIfNonzero(measurements, "renderer.aa.post_aa_smaa_planned",
               postAA.smaaPlanned);
  addIfNonzero(measurements, "renderer.aa.post_aa_smaa_submitted",
               postAA.smaaSubmitted);
  addIfNonzero(measurements, "renderer.aa.post_aa_smaa_submitted_passes",
               postAA.smaaSubmittedPassCount);
  addIfNonzero(measurements, "renderer.aa.post_aa_smaa_completed",
               postAA.smaaCompleted);
  if (postAA.smaaCompletedSourceFrameIndex !=
      std::numeric_limits<uint64_t>::max()) {
    addIfNonzero(measurements,
                 "renderer.aa.post_aa_smaa_completed_source_frame",
                 postAA.smaaCompletedSourceFrameIndex);
  }
  addIfNonzero(measurements, "renderer.aa.post_aa_degradation_mask",
               static_cast<uint32_t>(postAA.degradation));
  addIfNonzero(measurements, "renderer.aa.resolved_material_specular_aa",
               static_cast<uint32_t>(postAAPlan.resolvedMaterialSpecularAA));
  addIfNonzero(measurements, "renderer.aa.debug_view",
               static_cast<uint32_t>(postAAPlan.debugView));
  addIfNonzero(measurements, "renderer.aa.specular_aa_debug_override",
               static_cast<uint32_t>(postAAPlan.specularAADebugOverride));
  addIfNonzero(measurements,
               "renderer.aa.normal_variance_contract_materials_live",
               aa.normalVarianceContractMaterialsLive);
  addIfNonzero(measurements,
               "renderer.aa.normal_variance_contract_textures_live",
               aa.normalVarianceContractTexturesLive);
  addIfNonzero(measurements,
               "renderer.aa.normal_variance_unavailable_slots_live",
               aa.normalVarianceUnavailableSlotsLive);
  addBytesAsMiB(measurements,
                "gpu.memory.aa.normal_variance_contract_textures_mb",
                aa.normalVarianceContractTextureBytesLive);
  addIfNonzero(measurements, "renderer.aa.motion_vector_textures",
               aa.motionVectorTextureCount);
  addIfNonzero(measurements, "renderer.aa.motion_class_textures",
               aa.motionClassTextureCount);
  addIfNonzero(measurements, "renderer.aa.history_color_textures",
               aa.historyColorTextureCount);
  addIfNonzero(measurements, "renderer.aa.motion_vector_allocations",
               aa.motionVectorAllocationCount);
  addIfNonzero(measurements, "renderer.aa.motion_vector_reallocations",
               aa.motionVectorReallocationCount);
  addIfNonzero(measurements,
               "renderer.aa.motion_vector_depth_reprojection_passes",
               aa.motionVectorDepthReprojectionPassCount);
  addIfNonzero(measurements, "renderer.aa.velocity_passes",
               aa.velocityPassCount);
  addIfNonzero(measurements, "renderer.aa.velocity_draws",
               aa.velocityDrawCount);
  addIfNonzero(measurements, "renderer.aa.velocity_instances",
               aa.velocityInstanceCount);
  addIfNonzero(measurements, "renderer.aa.reactive_mask_passes",
               aa.reactiveMaskPassCount);
  addIfNonzero(measurements, "renderer.aa.reactive_mask_draws",
               aa.reactiveMaskDrawCount);
  addIfNonzero(measurements, "renderer.aa.taa_resolve_passes",
               aa.taaResolvePassCount);
  addIfNonzero(measurements, "renderer.aa.taa_copy_back_passes",
               aa.taaCopyBackPassCount);
  addIfNonzero(measurements, "renderer.aa.spatial_aa_passes",
               aa.spatialAAPassCount);
  addIfNonzero(measurements, "renderer.aa.msaa_resolve_passes",
               aa.msaaResolvePassCount);
  addIfNonzero(measurements, "renderer.aa.msaa_color_resolves",
               aa.msaaColorResolveCount);
  addIfNonzero(measurements, "renderer.aa.msaa_depth_resolves",
               aa.msaaDepthResolveCount);
  addIfNonzero(measurements, "renderer.aa.msaa_resolved_sample_count",
               aa.msaaResolvedSampleCount);
  addIfNonzero(measurements, "renderer.aa.msaa_sample_count",
               aa.msaaSampleCount);
  addIfNonzero(measurements, "renderer.aa.msaa_color_textures",
               aa.msaaColorTextureCount);
  addIfNonzero(measurements, "renderer.aa.msaa_depth_textures",
               aa.msaaDepthTextureCount);
  addIfNonzero(measurements, "renderer.aa.msaa_ring_slots", aa.msaaRingSlots);
  addIfNonzero(measurements, "renderer.aa.msaa_color_allocations",
               aa.msaaColorAllocationCount);
  addIfNonzero(measurements, "renderer.aa.msaa_color_reallocations",
               aa.msaaColorReallocationCount);
  addIfNonzero(measurements, "renderer.aa.msaa_depth_allocations",
               aa.msaaDepthAllocationCount);
  addIfNonzero(measurements, "renderer.aa.msaa_depth_reallocations",
               aa.msaaDepthReallocationCount);
  addIfNonzero(measurements, "renderer.aa.spatial_aa_allocations",
               aa.spatialAAAllocationCount);
  addIfNonzero(measurements, "renderer.aa.spatial_aa_reallocations",
               aa.spatialAAReallocationCount);
  addIfNonzero(measurements, "renderer.aa.msaa_sample4_color_supported",
               aa.msaaSample4ColorSupported);
  addIfNonzero(measurements, "renderer.aa.msaa_sample4_depth_supported",
               aa.msaaSample4DepthSupported);
  addIfNonzero(measurements, "renderer.aa.msaa_sample8_color_supported",
               aa.msaaSample8ColorSupported);
  addIfNonzero(measurements, "renderer.aa.msaa_sample8_depth_supported",
               aa.msaaSample8DepthSupported);
  addIfNonzero(measurements, "renderer.aa.msaa_depth_resolve_min_supported",
               aa.msaaDepthResolveMinSupported);
  addIfNonzero(measurements, "renderer.aa.msaa_alpha_to_coverage_supported",
               aa.msaaAlphaToCoverageSupported);
  addIfNonzero(measurements, "renderer.aa.msaa_sample_rate_shading_supported",
               aa.msaaSampleRateShadingSupported);
  addIfNonzero(measurements, "renderer.aa.msaa_alpha_to_coverage_active",
               aa.msaaAlphaToCoverageEnabled);
  addIfNonzero(measurements, "renderer.aa.msaa_sample_shading_active",
               aa.msaaSampleShadingEnabled);
  addIfNonzero(measurements, "renderer.aa.msaa_alpha_coverage_requested",
               aa.msaaAlphaCoverageRequested);
  addIfNonzero(measurements, "renderer.aa.msaa_spatial_cleanup_requested",
               aa.msaaSpatialCleanupRequested);
  addIfNonzero(measurements, "renderer.aa.msaa_spatial_cleanup_active",
               aa.msaaSpatialCleanupActive);
  addIfNonzero(measurements, "renderer.aa.msaa_unsupported_reason",
               static_cast<uint32_t>(aa.msaaUnsupportedReason));
  addIfNonzero(measurements, "renderer.aa.msaa_alpha_coverage_policy",
               static_cast<uint32_t>(aa.msaaAlphaCoveragePolicy));
  addIfNonzero(measurements, "renderer.aa.msaa_transparency_policy",
               static_cast<uint32_t>(aa.msaaTransparencyPolicy));
  addIfNonzero(measurements, "renderer.aa.msaa_resolve_placement",
               static_cast<uint32_t>(aa.msaaResolvePlacement));
  addIfNonzero(measurements, "renderer.aa.msaa_main_color_format",
               static_cast<uint32_t>(aa.msaaMainColorFormat));
  addIfNonzero(measurements, "renderer.aa.msaa_main_depth_format",
               static_cast<uint32_t>(aa.msaaMainDepthFormat));
  addIfNonzero(measurements, "renderer.aa.msaa_main_attachment_sample_count",
               aa.msaaMainAttachmentSampleCount);
  addIfNonzero(measurements, "renderer.aa.msaa_extent_width",
               aa.msaaExtentWidth);
  addIfNonzero(measurements, "renderer.aa.msaa_extent_height",
               aa.msaaExtentHeight);
  addIfNonzero(measurements, "renderer.aa.msaa_color_texel_bytes",
               aa.msaaColorTexelBytes);
  addIfNonzero(measurements, "renderer.aa.msaa_depth_texel_bytes",
               aa.msaaDepthTexelBytes);
  addIfNonzero(measurements, "renderer.aa.msaa_traffic_formula_version",
               aa.msaaTrafficFormulaVersion);
  addBytesAsMiB(measurements, "gpu.memory.aa.motion_vector_total_mb",
                aa.motionVectorTotalBytes);
  addBytesAsMiB(measurements, "gpu.memory.aa.reactive_mask_total_mb",
                aa.reactiveMaskTotalBytes);
  addBytesAsMiB(measurements, "gpu.memory.aa.motion_class_total_mb",
                aa.motionClassTotalBytes);
  addBytesAsMiB(measurements, "gpu.memory.aa.history_color_total_mb",
                aa.historyColorTotalBytes);
  addBytesAsMiB(measurements, "gpu.memory.aa.spatial_aa_total_mb",
                aa.spatialAATotalBytes);
  addBytesAsMiB(measurements, "gpu.memory.aa.msaa_total_mb", aa.msaaTotalBytes);
  addBytesAsMiB(measurements, "gpu.memory.aa.msaa_active_color_mb",
                aa.msaaColorTextureBytes);
  addBytesAsMiB(measurements, "gpu.memory.aa.msaa_active_depth_mb",
                aa.msaaDepthTextureBytes);
  addBytesAsMiB(measurements, "gpu.memory.aa.msaa_ring_color_mb",
                aa.msaaRingColorBytes);
  addBytesAsMiB(measurements, "gpu.memory.aa.msaa_ring_depth_mb",
                aa.msaaRingDepthBytes);
  addBytesAsMiB(measurements, "gpu.traffic.aa.msaa_resolve_read_estimated_mb",
                aa.msaaResolveReadEstimateBytes);
  addBytesAsMiB(measurements, "gpu.traffic.aa.msaa_resolve_write_estimated_mb",
                aa.msaaResolveWriteEstimateBytes);
  addIfNonzero(measurements, "renderer.aa.transparent_transmission_blend_draws",
               aa.transparentTransmissionBlendDrawCount);
  addIfNonzero(measurements,
               "renderer.aa.transparent_transmission_feedback_refreshes",
               aa.transparentTransmissionFeedbackRefreshCount);
  addIfNonzero(measurements,
               "renderer.aa.transparent_transmission_feedback_available",
               aa.transparentTransmissionFeedbackSourceAvailable);

  const AmbientOcclusionFrameMetrics &ao = metrics.ambientOcclusion;
  appendValue(measurements, NURI_BENCHMARK_METRIC("renderer.ao.input_mode"),
              static_cast<uint32_t>(ao.inputMode));
  appendValue(measurements,
              NURI_BENCHMARK_METRIC("renderer.ao.working_resolution"),
              static_cast<uint32_t>(ao.workingResolution));
  appendValue(measurements, NURI_BENCHMARK_METRIC("renderer.ao.output_width"),
              ao.width);
  appendValue(measurements, NURI_BENCHMARK_METRIC("renderer.ao.output_height"),
              ao.height);
  appendValue(measurements, NURI_BENCHMARK_METRIC("renderer.ao.working_width"),
              ao.workingWidth);
  appendValue(measurements, NURI_BENCHMARK_METRIC("renderer.ao.working_height"),
              ao.workingHeight);
  appendValue(measurements,
              NURI_BENCHMARK_METRIC("renderer.ao.working_pixel_count"),
              static_cast<uint64_t>(ao.workingWidth) *
                  static_cast<uint64_t>(ao.workingHeight));
  addIfNonzero(measurements, "renderer.ao.input_pass_draws", ao.inputPassDraws);
  addIfNonzero(measurements, "renderer.ao.normal_prepass_draws",
               ao.normalPrepassDraws);
  addIfNonzero(measurements, "renderer.ao.depth_prefilter_passes",
               ao.depthPrefilterPassCount);
  addIfNonzero(measurements, "renderer.ao.main_passes", ao.mainPassCount);
  addIfNonzero(measurements, "renderer.ao.temporal_passes",
               ao.temporalPassCount);
  addIfNonzero(measurements, "renderer.ao.temporal_reactive_mask_consumed",
               ao.temporalReactiveMaskConsumed);
  addIfNonzero(measurements, "renderer.ao.temporal_motion_class_consumed",
               ao.temporalMotionClassConsumed);
  addIfNonzero(measurements, "renderer.ao.temporal_previous_depth_consumed",
               ao.temporalPreviousDepthConsumed);
  addIfNonzero(measurements, "renderer.ao.texture_count", ao.textureCount);
  addIfNonzero(measurements, "renderer.ao.normal_texture_count",
               ao.normalTextureCount);
  addIfNonzero(measurements, "renderer.ao.history_texture_count",
               ao.ambientOcclusionTextureCount);
  addBytesAsMiB(measurements, "gpu.memory.ao.total_texture_mb",
                ao.totalTextureBytes);
  addBytesAsMiB(measurements, "gpu.memory.ao.scratch_texture_mb",
                ao.scratchTextureBytes);
  addBytesAsMiB(measurements, "renderer.ao.allocated_texture_mb",
                ao.totalTextureBytes);
  addBytesAsMiB(measurements, "renderer.ao.logical_active_texture_mb",
                ao.logicalActiveTextureBytes);
  addBytesAsMiB(measurements, "renderer.ao.provider_texture_mb",
                ao.providerTextureBytes);
  addBytesAsMiB(measurements, "renderer.ao.feature_texture_mb",
                ao.featureTextureBytes);

  const HDRPostProcessFrameMetrics &hdr = metrics.hdrPostProcess;
  addIfNonzero(measurements, "renderer.hdr.bloom_passes", hdr.bloomPassCount);
  addIfNonzero(measurements, "renderer.hdr.luminance_passes",
               hdr.luminancePassCount);
  addIfNonzero(measurements, "renderer.hdr.adaptation_passes",
               hdr.adaptationPassCount);
  addIfNonzero(measurements, "renderer.hdr.texture_count", hdr.textureCount);
  appendValue(
      measurements,
      NURI_BENCHMARK_METRIC("renderer.hdr.exposure_telemetry_available"),
      static_cast<uint32_t>(hdr.exposureTelemetryAvailable));
  addIfNonzero(measurements, "renderer.hdr.exposure_telemetry_source_frame",
               hdr.exposureTelemetrySourceFrameIndex ==
                       std::numeric_limits<uint64_t>::max()
                   ? 0u
                   : hdr.exposureTelemetrySourceFrameIndex);
  appendValue(
      measurements,
      NURI_BENCHMARK_METRIC("renderer.hdr.exposure_telemetry_stale_frames"),
      hdr.exposureTelemetryStaleFrames);
  appendValue(
      measurements,
      NURI_BENCHMARK_METRIC("renderer.hdr.exposure_telemetry_pending_slots"),
      hdr.exposureTelemetryPendingSlots);
  appendValue(
      measurements,
      NURI_BENCHMARK_METRIC("renderer.hdr.exposure_telemetry_dropped_samples"),
      hdr.exposureTelemetryDroppedSamples);
  appendValue(measurements,
              NURI_BENCHMARK_METRIC("renderer.hdr.automatic_exposure_ev"),
              hdr.automaticExposureEv);
  appendValue(measurements,
              NURI_BENCHMARK_METRIC("renderer.hdr.exposure_target_ev"),
              hdr.exposureTargetEv);
  appendValue(measurements,
              NURI_BENCHMARK_METRIC("renderer.hdr.exposure_metered_luminance"),
              hdr.exposureMeteredLuminance);
  appendValue(measurements,
              NURI_BENCHMARK_METRIC("renderer.hdr.effective_exposure_ev"),
              hdr.effectiveExposureEv);
  appendValue(
      measurements,
      NURI_BENCHMARK_METRIC("renderer.hdr.exposure_invalid_sample_fraction"),
      hdr.exposureInvalidSampleFraction);
  addBytesAsMiB(measurements, "gpu.memory.hdr.texture_mb", hdr.textureBytes);

  addIfNonzero(measurements, "renderer.transparent.mesh_draws",
               metrics.transparent.meshDraws);
  addIfNonzero(measurements, "renderer.transparent.contributor_sortable_draws",
               metrics.transparent.contributorSortableDraws);
  addIfNonzero(measurements, "renderer.transparent.contributor_fixed_draws",
               metrics.transparent.contributorFixedDraws);
  addIfNonzero(measurements, "renderer.transparent.pick_draws",
               metrics.transparent.pickDraws);

  const RayTracingSceneFrameMetrics &rt = metrics.rayTracingScene;
  addIfNonzero(measurements, "renderer.ray_tracing.static_instances",
               rt.staticInstances);
  addIfNonzero(measurements, "renderer.ray_tracing.dynamic_instances",
               rt.dynamicInstances);
  addIfNonzero(measurements, "renderer.ray_tracing.excluded_dynamic_instances",
               rt.excludedDynamicInstances);
  addIfNonzero(measurements, "renderer.ray_tracing.static_blas_count",
               rt.staticBlasCount);
  addIfNonzero(measurements, "renderer.ray_tracing.dynamic_blas_count",
               rt.dynamicBlasCount);
  addIfNonzero(measurements, "renderer.ray_tracing.tlas_count", rt.tlasCount);
  addIfNonzero(measurements, "renderer.ray_tracing.unique_static_geometry",
               rt.uniqueStaticGeometry);
  addIfNonzero(measurements, "renderer.ray_tracing.geometry_records",
               rt.geometryRecords);
  addIfNonzero(measurements, "renderer.ray_tracing.triangles", rt.triangles);
  addIfNonzero(measurements, "renderer.ray_tracing.queued_blas_builds",
               rt.queuedBlasBuilds);
  addIfNonzero(measurements, "renderer.ray_tracing.decoded_vertices",
               rt.decodedVertices);
  addIfNonzero(measurements, "renderer.ray_tracing.decode_dispatches",
               rt.decodeDispatches);
  addIfNonzero(measurements, "renderer.ray_tracing.blas_builds", rt.blasBuilds);
  addIfNonzero(measurements, "renderer.ray_tracing.tlas_builds", rt.tlasBuilds);
  addIfNonzero(measurements, "renderer.ray_tracing.tlas_updates",
               rt.tlasUpdates);
  addIfNonzero(measurements, "renderer.ray_tracing.dynamic_blas_updates",
               rt.dynamicBlasUpdates);
  addIfNonzero(measurements, "renderer.ray_tracing.dynamic_vertex_dispatches",
               rt.dynamicVertexDispatches);
  addIfNonzero(measurements, "renderer.ray_tracing.readiness",
               static_cast<uint32_t>(rt.readiness));
  addIfNonzero(measurements, "renderer.ray_tracing.consumed_rebuild_epoch",
               rt.consumedRebuildEpoch);
  addBytesAsMiB(measurements, "gpu.memory.ray_tracing.decoded_positions_mb",
                rt.decodedPositionBytes);
  addBytesAsMiB(measurements, "gpu.memory.ray_tracing.tables_mb",
                rt.tableBytes);
  addBytesAsMiB(measurements, "gpu.memory.ray_tracing.blas_mb",
                rt.blasAllocationBytes);
  addBytesAsMiB(measurements, "gpu.memory.ray_tracing.tlas_mb",
                rt.tlasAllocationBytes);
  addBytesAsMiB(measurements, "gpu.memory.ray_tracing.as_scratch_high_water_mb",
                rt.asScratchHighWaterBytes);
  addIfNonzero(measurements,
               "renderer.ray_tracing.direct_binding_pool_high_water",
               rt.directBindingPoolHighWater);
  const DDGIFrameMetrics &ddgi = metrics.ddgi;
  addIfNonzero(measurements, "renderer.ddgi.requested", ddgi.requested);
  addIfNonzero(measurements, "renderer.ddgi.active", ddgi.active);
  addIfNonzero(measurements, "renderer.ddgi.active_volumes",
               ddgi.activeVolumes);
  addIfNonzero(measurements, "renderer.ddgi.ready_volumes", ddgi.readyVolumes);
  addIfNonzero(measurements, "renderer.ddgi.total_probes", ddgi.totalProbes);
  addIfNonzero(measurements, "renderer.ddgi.vigilant_probes",
               ddgi.vigilantProbes);
  addIfNonzero(measurements, "renderer.ddgi.uninitialized_probes",
               ddgi.uninitializedProbes);
  addIfNonzero(measurements, "renderer.ddgi.off_probes", ddgi.offProbes);
  addIfNonzero(measurements, "renderer.ddgi.sleeping_probes",
               ddgi.sleepingProbes);
  addIfNonzero(measurements, "renderer.ddgi.newly_awake_probes",
               ddgi.newlyAwakeProbes);
  addIfNonzero(measurements, "renderer.ddgi.awake_probes", ddgi.awakeProbes);
  addIfNonzero(measurements, "renderer.ddgi.newly_vigilant_probes",
               ddgi.newlyVigilantProbes);
  addIfNonzero(measurements, "renderer.ddgi.relocated_probes",
               ddgi.relocatedProbes);
  addIfNonzero(measurements, "renderer.ddgi.probe_state_readback_available",
               ddgi.probeStateReadbackAvailable);
  addIfNonzero(measurements, "renderer.ddgi.probe_state_readback_source_frame",
               ddgi.probeStateReadbackSourceFrame);
  addIfNonzero(measurements, "renderer.ddgi.probe_state_readback_stale_frames",
               ddgi.probeStateReadbackStaleFrames);
  addIfNonzero(measurements, "renderer.ddgi.max_relocation",
               ddgi.maxRelocation);
  addIfNonzero(measurements, "renderer.ddgi.updated_probes",
               ddgi.updatedProbes);
  addIfNonzero(measurements, "renderer.ddgi.primary_queries",
               ddgi.primaryQueries);
  addIfNonzero(measurements, "renderer.ddgi.classification_probe_updates",
               ddgi.classificationProbeUpdates);
  addIfNonzero(measurements, "renderer.ddgi.classification_primary_queries",
               ddgi.classificationPrimaryQueries);
  addIfNonzero(measurements, "renderer.ddgi.irradiance_primary_queries",
               ddgi.irradiancePrimaryQueries);
  addIfNonzero(measurements, "renderer.ddgi.primary_queries_issued",
               ddgi.primaryQueriesIssued);
  addIfNonzero(measurements, "renderer.ddgi.trace_counter_source_frame",
               ddgi.traceCounterSourceFrame);
  addIfNonzero(measurements, "renderer.ddgi.trace_counter_stale_frames",
               ddgi.traceCounterStaleFrames);
  appendValue(measurements,
              NURI_BENCHMARK_METRIC("renderer.ddgi.readback_waits"),
              ddgi.readbackWaits);
  appendValue(measurements,
              NURI_BENCHMARK_METRIC("renderer.ddgi.readback_copy_bytes"),
              ddgi.readbackCopyBytes);
  appendValue(measurements,
              NURI_BENCHMARK_METRIC("renderer.ddgi.readback_pending_slots"),
              ddgi.readbackPendingSlots);
  appendValue(measurements,
              NURI_BENCHMARK_METRIC("renderer.ddgi.readback_dropped_samples"),
              ddgi.readbackDroppedSamples);
  appendValue(
      measurements,
      NURI_BENCHMARK_METRIC("renderer.ddgi.readback_oldest_pending_age"),
      ddgi.readbackOldestPendingAge);
  appendValue(
      measurements,
      NURI_BENCHMARK_METRIC("renderer.ddgi.readback_blocking_fallbacks"),
      ddgi.readbackBlockingFallbacks);
  appendValue(
      measurements,
      NURI_BENCHMARK_METRIC("renderer.ddgi.readback_generation_mismatches"),
      ddgi.readbackGenerationMismatches);
  appendValue(
      measurements,
      NURI_BENCHMARK_METRIC("renderer.ddgi.readback_early_reuse_attempts"),
      ddgi.readbackEarlyReuseAttempts);
  addIfNonzero(measurements, "renderer.ddgi.secondary_queries_reserved",
               ddgi.secondaryQueriesReserved);
  addIfNonzero(measurements, "renderer.ddgi.secondary_queries_unused",
               ddgi.secondaryQueriesUnused);
  addIfNonzero(measurements, "renderer.ddgi.secondary_query_budget_overflows",
               ddgi.secondaryQueryBudgetOverflows);
  addIfNonzero(measurements, "renderer.ddgi.secondary_queries",
               ddgi.secondaryQueries);
  addIfNonzero(measurements, "renderer.ddgi.directional_secondary_queries",
               ddgi.directionalSecondaryQueries);
  addIfNonzero(measurements, "renderer.ddgi.local_secondary_queries",
               ddgi.localSecondaryQueries);
  addIfNonzero(measurements, "renderer.ddgi.total_queries_issued",
               static_cast<uint64_t>(ddgi.primaryQueriesIssued) +
                   ddgi.secondaryQueries);
  addIfNonzero(measurements, "renderer.ddgi.primary_candidate_intersections",
               ddgi.primaryCandidateIntersections);
  addIfNonzero(measurements, "renderer.ddgi.secondary_candidate_intersections",
               ddgi.secondaryCandidateIntersections);
  addIfNonzero(measurements, "renderer.ddgi.alpha_candidate_rejections",
               ddgi.alphaCandidateRejections);
  addIfNonzero(measurements, "renderer.ddgi.backface_candidate_rejections",
               ddgi.backfaceCandidateRejections);
  addIfNonzero(measurements, "renderer.ddgi.candidate_overflows",
               ddgi.candidateOverflows);
  addIfNonzero(measurements, "renderer.ddgi.local_light_truncations",
               ddgi.localLightTruncations);
  addIfNonzero(measurements, "renderer.ddgi.non_finite_radiance_rejects",
               ddgi.nonFiniteRadianceRejects);
  addIfNonzero(measurements, "renderer.ddgi.emissive_radiance_clamps",
               ddgi.emissiveRadianceClamps);
  addIfNonzero(measurements, "renderer.ddgi.direct_radiance_clamps",
               ddgi.directRadianceClamps);
  addIfNonzero(measurements, "renderer.ddgi.sky_radiance_clamps",
               ddgi.skyRadianceClamps);
  addIfNonzero(measurements, "renderer.ddgi.multi_bounce_radiance_clamps",
               ddgi.multiBounceRadianceClamps);
  addIfNonzero(measurements, "renderer.ddgi.final_radiance_clamps",
               ddgi.finalRadianceClamps);
  appendValue(
      measurements,
      NURI_BENCHMARK_METRIC("renderer.ddgi.diagnostic_counters_enabled"),
      ddgi.diagnosticCountersEnabled);
  appendValue(
      measurements,
      NURI_BENCHMARK_METRIC("renderer.ddgi.surface_gather_architecture"),
      static_cast<uint32_t>(ddgi.surfaceGatherArchitecture));
  addIfNonzero(measurements, "renderer.ddgi.surface_gather_width",
               ddgi.surfaceGatherWidth);
  addIfNonzero(measurements, "renderer.ddgi.surface_gather_height",
               ddgi.surfaceGatherHeight);
  addIfNonzero(measurements,
               "renderer.ddgi.surface_gather_max_candidate_volumes",
               ddgi.surfaceGatherMaxCandidateVolumes);
  addIfNonzero(measurements, "renderer.ddgi.surface_gather_max_sampled_volumes",
               ddgi.surfaceGatherMaxSampledVolumes);
  addIfNonzero(measurements,
               "renderer.ddgi.surface_gather_max_state_loads_per_pixel",
               ddgi.surfaceGatherMaxStateLoadsPerPixel);
  addIfNonzero(measurements,
               "renderer.ddgi.surface_gather_max_atlas_samples_per_pixel",
               ddgi.surfaceGatherMaxAtlasSamplesPerPixel);
  addIfNonzero(measurements, "renderer.ddgi.ray_query_capacity",
               ddgi.rayQueryCapacity);
  addIfNonzero(measurements, "renderer.ddgi.probe_update_capacity",
               ddgi.probeUpdateCapacity);
  appendValue(
      measurements,
      NURI_BENCHMARK_METRIC("renderer.ddgi.requested_probe_update_capacity"),
      ddgi.requestedProbeUpdateCapacity);
  appendValue(
      measurements,
      NURI_BENCHMARK_METRIC("renderer.ddgi.effective_probe_update_capacity"),
      ddgi.effectiveProbeUpdateCapacity);
  appendValue(measurements,
              NURI_BENCHMARK_METRIC("renderer.ddgi.startup_phase"),
              static_cast<uint32_t>(ddgi.startupPhase));
  appendValue(measurements,
              NURI_BENCHMARK_METRIC(
                  "renderer.ddgi.sky_remainder_over_threshold_percentage"),
              ddgi.skyRemainderOverThresholdPercentage);
  addIfNonzero(measurements, "renderer.ddgi.reset_count", ddgi.resetCount);
  addIfNonzero(measurements, "renderer.ddgi.scroll_count", ddgi.scrollCount);
  addIfNonzero(measurements, "renderer.ddgi.invalidated_probes",
               ddgi.invalidatedProbes);
  addIfNonzero(measurements, "renderer.ddgi.failed_volumes",
               ddgi.failedVolumes);
  addIfNonzero(measurements, "renderer.ddgi.effective_volumes",
               ddgi.effectiveVolumes);
  addIfNonzero(measurements, "renderer.ddgi.authored_volumes",
               ddgi.authoredVolumes);
  addIfNonzero(measurements, "renderer.ddgi.generated_volumes",
               ddgi.generatedVolumes);
  addIfNonzero(measurements, "renderer.ddgi.redundant_authored_volumes",
               ddgi.redundantAuthoredVolumes);
  addIfNonzero(measurements, "renderer.ddgi.redundant_authored_probes",
               ddgi.redundantAuthoredProbes);
  addBytesAsMiB(measurements, "gpu.memory.ddgi.redundant_authored_mb",
                ddgi.redundantAuthoredBytes);
  addIfNonzero(measurements, "renderer.ddgi.coverage_mode", ddgi.coverageMode);
  addIfNonzero(measurements, "renderer.ddgi.coverage_status",
               static_cast<uint32_t>(ddgi.coverageStatus));
  addIfNonzero(measurements, "renderer.ddgi.coverage_error",
               static_cast<uint32_t>(ddgi.coverageError));
  addIfNonzero(measurements, "renderer.ddgi.limiting_constraint",
               static_cast<uint32_t>(ddgi.limitingConstraint));
  appendValue(measurements,
              NURI_BENCHMARK_METRIC("renderer.ddgi.requested_half_extent_x"),
              ddgi.requestedCoverageHalfExtents.x);
  appendValue(measurements,
              NURI_BENCHMARK_METRIC("renderer.ddgi.requested_half_extent_y"),
              ddgi.requestedCoverageHalfExtents.y);
  appendValue(measurements,
              NURI_BENCHMARK_METRIC("renderer.ddgi.requested_half_extent_z"),
              ddgi.requestedCoverageHalfExtents.z);
  appendValue(measurements,
              NURI_BENCHMARK_METRIC("renderer.ddgi.achieved_half_extent_x"),
              ddgi.achievedCoverageHalfExtents.x);
  appendValue(measurements,
              NURI_BENCHMARK_METRIC("renderer.ddgi.achieved_half_extent_y"),
              ddgi.achievedCoverageHalfExtents.y);
  appendValue(measurements,
              NURI_BENCHMARK_METRIC("renderer.ddgi.achieved_half_extent_z"),
              ddgi.achievedCoverageHalfExtents.z);
  appendValue(measurements,
              NURI_BENCHMARK_METRIC("renderer.ddgi.scene_coverage_ratio"),
              ddgi.sceneCoverageRatio);
  appendValue(measurements,
              NURI_BENCHMARK_METRIC("renderer.ddgi.coverage_resolve_cpu_ms"),
              ddgi.coverageResolveCpuTimeMs);
  addIfNonzero(measurements, "renderer.ddgi.diagnostic_sample_count",
               ddgi.diagnosticSampleCount);
  addIfNonzero(measurements, "renderer.ddgi.uncovered_diagnostic_samples",
               ddgi.uncoveredDiagnosticSamples);
  addIfNonzero(measurements, "renderer.ddgi.sky_remainder_samples",
               ddgi.skyRemainderSamples);
  addIfNonzero(measurements, "renderer.ddgi.diagnostic_samples_available",
               ddgi.diagnosticSamplesAvailable);
  addIfNonzero(measurements, "renderer.ddgi.dirty_regions_produced",
               ddgi.dirtyRegionsProduced);
  addIfNonzero(measurements, "renderer.ddgi.dirty_regions_merged",
               ddgi.dirtyRegionsMerged);
  addIfNonzero(measurements, "renderer.ddgi.dirty_regions_overflowed",
               ddgi.dirtyRegionsOverflowed);
  addIfNonzero(measurements, "renderer.ddgi.dirty_regions_pending",
               ddgi.dirtyRegionsPending);
  addIfNonzero(measurements, "renderer.ddgi.dirty_probes_affected",
               ddgi.dirtyProbesAffected);
  addIfNonzero(measurements, "renderer.ddgi.classification_fallbacks",
               ddgi.classificationFallbacks);
  addIfNonzero(measurements, "renderer.ddgi.classification_overflows",
               ddgi.classificationOverflows);
  addIfNonzero(measurements, "renderer.ddgi.volume_failure_reason",
               static_cast<uint32_t>(ddgi.volumeFailureReason));
  addIfNonzero(measurements, "renderer.ddgi.history_ready", ddgi.historyReady);
  addIfNonzero(measurements, "renderer.ddgi.irradiance_response_remaining",
               ddgi.irradianceResponseRemaining);
  addIfNonzero(measurements, "renderer.ddgi.distance_response_remaining",
               ddgi.distanceResponseRemaining);
  addIfNonzero(measurements, "renderer.ddgi.inspection_available",
               ddgi.inspectionAvailable);
  addIfNonzero(measurements, "renderer.ddgi.inspection_valid",
               ddgi.inspectionValid);
  addIfNonzero(measurements, "renderer.ddgi.inspection_ray_count",
               ddgi.inspectionRayCount);
  addIfNonzero(measurements, "renderer.ddgi.inspection_hit_count",
               ddgi.inspectionHitCount);
  addIfNonzero(measurements, "renderer.ddgi.inspection_miss_count",
               ddgi.inspectionMissCount);
  addIfNonzero(measurements, "renderer.ddgi.inspection_candidate_overflows",
               ddgi.inspectionCandidateOverflows);
  addIfNonzero(measurements, "renderer.ddgi.inspection_event_overflows",
               ddgi.inspectionEventOverflows);
  addIfNonzero(measurements, "renderer.ddgi.sky_fallback_active",
               ddgi.skyFallbackActive);
  addIfNonzero(measurements, "renderer.ddgi.fallback_reason",
               static_cast<uint32_t>(ddgi.fallbackReason));
  addIfNonzero(measurements, "renderer.ddgi.submitted_sequence",
               ddgi.submittedSequence);
  addIfNonzero(measurements, "renderer.ddgi.layout_generation",
               ddgi.layoutGeneration);
  addIfNonzero(measurements, "renderer.ddgi.resource_generation",
               ddgi.resourceGeneration);
  addIfNonzero(measurements, "renderer.ddgi.device_epoch", ddgi.deviceEpoch);
  addIfNonzero(measurements, "renderer.ddgi.consumed_reset_epoch",
               ddgi.consumedResetEpoch);
  addIfNonzero(measurements, "renderer.ddgi.consumed_force_update_epoch",
               ddgi.consumedForceUpdateEpoch);
  addBytesAsMiB(measurements, "gpu.memory.ddgi.persistent_mb",
                ddgi.persistentBytes);
  addBytesAsMiB(measurements, "gpu.memory.ddgi.frame_batch_mb",
                ddgi.frameBatchBytes);
  addBytesAsMiB(measurements, "gpu.memory.ddgi.committed_atlas_mb",
                ddgi.committedAtlasBytes);
  addBytesAsMiB(measurements, "gpu.memory.ddgi.pending_atlas_mb",
                ddgi.pendingAtlasBytes);
  addBytesAsMiB(measurements, "gpu.memory.ddgi.peak_atlas_mb",
                ddgi.peakAtlasBytes);
  for (size_t volumeIndex = 0u; volumeIndex < ddgi.volumes.size();
       ++volumeIndex) {
    const DDGIVolumeFrameMetrics &volume = ddgi.volumes[volumeIndex];
    const std::string prefix =
        "renderer.ddgi.volume" + std::to_string(volumeIndex) + ".";
    const auto addVolumeMetric = [&](std::string_view suffix, double value) {
      appendValue(measurements,
                  registeredMetricIndex(prefix + std::string(suffix)), value);
    };
    addVolumeMetric("active", volume.active);
    addVolumeMetric("effective_kind", volume.effectiveKind);
    addVolumeMetric("tier", volume.tier);
    addVolumeMetric("cascade_index", volume.cascadeIndex);
    addVolumeMetric("total_probes", volume.totalProbes);
    addVolumeMetric("initialized_probes", volume.initializedProbes);
    addVolumeMetric("shading_enabled_probes", volume.shadingEnabledProbes);
    addVolumeMetric("invalid_probes", volume.invalidProbes);
    addVolumeMetric("newly_exposed_probes", volume.newlyExposedProbes);
    addVolumeMetric("updates", volume.updates);
    addVolumeMetric("primary_queries", volume.primaryQueries);
    addVolumeMetric("primary_queries_issued", volume.primaryQueriesIssued);
    addVolumeMetric("secondary_queries", volume.secondaryQueries);
    addVolumeMetric("update_age_median", volume.updateAgeMedian);
    addVolumeMetric("update_age_p95", volume.updateAgeP95);
    addVolumeMetric("update_age_maximum", volume.updateAgeMaximum);
    addVolumeMetric("scheduled_quota", volume.scheduledQuota);
    addVolumeMetric("used_quota", volume.usedQuota);
    addVolumeMetric("deficit", static_cast<double>(volume.deficit));
    addVolumeMetric("starvation_frames", volume.starvationFrames);
    addVolumeMetric("estimated_full_refresh_frames",
                    volume.estimatedFullRefreshFrames);
    addVolumeMetric("persistent_mb",
                    static_cast<double>(volume.persistentBytes) /
                        (1024.0 * 1024.0));
    addVolumeMetric("unique_coverage_percentage",
                    volume.uniqueCoveragePercentage);
    addVolumeMetric("redundant_coverage", volume.redundantCoverage);
    addVolumeMetric("history_ready_percentage", volume.historyReadyPercentage);
    addVolumeMetric("coverage_ready_percentage",
                    volume.coverageReadyPercentage);
    addVolumeMetric("confidence", volume.confidence);
  }
  const uint64_t estimatedFrameTextureBytes =
      shadow.cascadeTextureBytes + aa.motionVectorTotalBytes +
      aa.reactiveMaskTotalBytes + aa.motionClassTotalBytes +
      aa.historyColorTotalBytes + aa.spatialAATotalBytes + aa.msaaTotalBytes +
      ao.totalTextureBytes + hdr.textureBytes;
  addBytesAsMiB(measurements, "gpu.memory.frame_textures_estimated_mb",
                estimatedFrameTextureBytes);
}

#undef addBytesAsMiB
#undef addIfNonzero

void addRenderGraphTelemetryMetrics(BenchmarkFrameMeasurements &measurements,
                                    const Renderer &renderer) {
  const RenderGraphTelemetrySnapshot *snapshot =
      renderer.renderGraphTelemetry().latestSnapshot();
  if (snapshot == nullptr) {
    return;
  }

  const RenderGraphTelemetrySnapshot::Summary &summary = snapshot->summary;
#define addCount(id, value)                                                    \
  measurements.appendRegistered(NURI_BENCHMARK_METRIC(id),                     \
                                static_cast<double>(value))
  addCount("rendergraph.summary.declared_pass_count",
           summary.declaredPassCount);
  addCount("rendergraph.summary.culled_pass_count", summary.culledPassCount);
  addCount("rendergraph.summary.root_pass_count", summary.rootPassCount);
  addCount("rendergraph.summary.pass_count", summary.passCount);
  addCount("rendergraph.summary.edge_count", summary.edgeCount);
  addCount("rendergraph.summary.recorded_graphics_pass_count",
           summary.recordedGraphicsPassCount);
  addCount("rendergraph.summary.pass_barrier_plan_count",
           summary.passBarrierPlanCount);
  addCount("rendergraph.summary.final_barrier_record_count",
           summary.finalBarrierRecordCount);
  addCount("rendergraph.summary.pass_barrier_record_count",
           summary.passBarrierRecordCount);
  addCount("rendergraph.summary.recorded_command_buffer_count",
           summary.recordedCommandBufferCount);
  addCount("rendergraph.summary.submit_batch_count", summary.submitBatchCount);
  addCount("rendergraph.summary.pass_range_count", summary.passRangeCount);
  addCount("rendergraph.summary.pass_timing_count", summary.passTimingCount);
  addCount("rendergraph.summary.imported_texture_count",
           summary.importedTextures);
  addCount("rendergraph.summary.transient_texture_count",
           summary.transientTextures);
  addCount("rendergraph.summary.imported_buffer_count",
           summary.importedBuffers);
  addCount("rendergraph.summary.transient_buffer_count",
           summary.transientBuffers);
  addCount("rendergraph.summary.transient_texture_lifetime_count",
           summary.transientTextureLifetimeCount);
  addCount("rendergraph.summary.transient_buffer_lifetime_count",
           summary.transientBufferLifetimeCount);
  addCount("rendergraph.summary.transient_texture_physical_count",
           summary.transientTexturePhysicalCount);
  addCount("rendergraph.summary.transient_buffer_physical_count",
           summary.transientBufferPhysicalCount);
  addCount("rendergraph.summary.transient_texture_allocation_map_size",
           summary.transientTextureAllocationMapSize);
  addCount("rendergraph.summary.transient_buffer_allocation_map_size",
           summary.transientBufferAllocationMapSize);
  addCount("rendergraph.summary.transient_texture_physical_allocation_count",
           summary.transientTexturePhysicalAllocationCount);
  addCount("rendergraph.summary.transient_buffer_physical_allocation_count",
           summary.transientBufferPhysicalAllocationCount);
  addCount("rendergraph.summary.unresolved_texture_binding_count",
           summary.unresolvedTextureBindingCount);
  addCount("rendergraph.summary.resolved_dependency_buffer_slot_count",
           summary.resolvedDependencyBufferSlotCount);
  addCount("rendergraph.summary.unresolved_dependency_buffer_binding_count",
           summary.unresolvedDependencyBufferBindingCount);
  addCount("rendergraph.summary.owned_pre_dispatch_count",
           summary.ownedPreDispatchCount);
  addCount("rendergraph.summary.owned_draw_item_count",
           summary.ownedDrawItemCount);
  addCount("rendergraph.summary.owned_mesh_dispatch_item_count",
           summary.ownedMeshDispatchItemCount);

  for (const RenderGraphPassExecutionTiming &timing :
       snapshot->execution.passTimings) {
    const uint32_t orderedPassIndex = timing.orderedPassIndex;
    if (orderedPassIndex >= snapshot->compile.orderedPassIndices.size()) {
      continue;
    }
    const uint32_t declaredPassIndex =
        snapshot->compile.orderedPassIndices[orderedPassIndex];
    const std::string_view passName =
        resolveTelemetryPassName(*snapshot, declaredPassIndex);
    measurements.appendOwned(
        passTimingMetricId(orderedPassIndex, passName, "cpu_ms"),
        static_cast<double>(timing.cpuTimeMs));
  }
#undef addCount
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

class BenchmarkLogGuard final {
public:
  explicit BenchmarkLogGuard(const std::filesystem::path &path) {
    std::filesystem::create_directories(path.parent_path());
    LogConfig config{};
    config.filePath = path.string();
    config.logLevel = LogLevel::Info;
    config.consoleLevel = LogLevel::Warning;
    Log::initialize(config);
  }
  ~BenchmarkLogGuard() { Log::shutdown(); }
  BenchmarkLogGuard(const BenchmarkLogGuard &) = delete;
  BenchmarkLogGuard &operator=(const BenchmarkLogGuard &) = delete;
};

[[nodiscard]] std::string resolveBackendName(const BenchmarkCase &benchmarkCase,
                                             std::string &source) {
  if (benchmarkCase.backend != "default") {
    source = "manifest";
    return benchmarkCase.backend;
  }
  source = "default";
  return "nvrhi";
}

[[nodiscard]] std::string resolvePresentMode(const BenchmarkCase &benchmarkCase,
                                             std::string &source) {
  const std::string envPresent = readProcessEnvironment("NURI_PRESENT_MODE");
  if (benchmarkCase.presentMode != "default") {
    source = "manifest";
    return benchmarkCase.presentMode;
  }
  if (!envPresent.empty()) {
    source = "NURI_PRESENT_MODE";
    return envPresent;
  }
  source = "default";
  return "default";
}

[[nodiscard]] Result<bool, BenchmarkExitCode>
checkRequirements(const BenchmarkCase &benchmarkCase, std::string_view backend,
                  std::vector<std::string> &warnings, std::string &message) {
  if (backend != "nvrhi") {
    message = "unsupported backend '" + std::string(backend) +
              "'; nvrhi is the only available backend";
    return Result<bool, BenchmarkExitCode>::makeError(
        BenchmarkExitCode::InvalidInput);
  }
  if (!benchmarkCase.requirements.allowVisibleWindow) {
    message = "case requires hidden/headless execution, which is unavailable";
    return Result<bool, BenchmarkExitCode>::makeError(
        BenchmarkExitCode::EnvironmentUnavailable);
  }
  if (!benchmarkCase.requirements.backends.empty()) {
    bool supported = false;
    for (const std::string &allowed : benchmarkCase.requirements.backends) {
      if (allowed != "default" && allowed != "nvrhi") {
        message = "unsupported backend requirement '" + allowed + "'";
        return Result<bool, BenchmarkExitCode>::makeError(
            BenchmarkExitCode::InvalidInput);
      }
      supported = supported || allowed == backend || allowed == "default";
    }
    if (!supported) {
      message = "backend '" + std::string(backend) +
                "' is not allowed by case requirements";
      return Result<bool, BenchmarkExitCode>::makeError(
          BenchmarkExitCode::EnvironmentUnavailable);
    }
  }
  for (const std::string &asset : benchmarkCase.requirements.assets) {
    const size_t colon = asset.find(':');
    if (colon == std::string::npos) {
      message = "invalid asset requirement '" + asset + "'";
      return Result<bool, BenchmarkExitCode>::makeError(
          BenchmarkExitCode::InvalidInput);
    }
    auto path =
        resolveBenchmarkPath(asset.substr(0, colon), asset.substr(colon + 1u));
    if (path.hasError()) {
      message = path.error();
      return Result<bool, BenchmarkExitCode>::makeError(
          BenchmarkExitCode::EnvironmentUnavailable);
    }
    if (!std::filesystem::exists(path.value())) {
      message = "missing required asset: " + path.value().string();
      return Result<bool, BenchmarkExitCode>::makeError(
          BenchmarkExitCode::EnvironmentUnavailable);
    }
  }
  if (benchmarkCase.scene.kind == "prefab" &&
      benchmarkCase.scene.baseModelKind == "fitRadius") {
    warnings.push_back(
        "prefab fitRadius transforms are parsed but not enabled for benchmark "
        "comparison in this slice");
  }
  return Result<bool, BenchmarkExitCode>::makeResult(true);
}

void addGpuTimingMetric(BenchmarkFrameMeasurements &measurements,
                        BenchmarkMetricIndex metricIndex,
                        const GpuTimingReport &report, GpuTimingScope scope,
                        float timeMs) {
  if (hasGpuTimingScope(report, scope)) {
    measurements.appendRegistered(metricIndex, static_cast<double>(timeMs));
  }
}

void applyGpuTimingReport(BenchmarkReport &report,
                          const GpuTimingReport &timingReport,
                          const std::map<uint64_t, size_t> &frameByIndex) {
  std::map<uint64_t, double> scopeSumsByFrame;
  const auto add = [&](BenchmarkMetricIndex index, GpuTimingScope scope,
                       uint64_t sourceFrameIndex, float ms) -> bool {
    if (!hasGpuTimingScope(timingReport, scope)) {
      return false;
    }
    const auto frameIt = frameByIndex.find(sourceFrameIndex);
    if (frameIt == frameByIndex.end()) {
      return false;
    }
    report.frames[frameIt->second].measurements.appendRegistered(
        index, static_cast<double>(ms));
    if (shouldIncludeGpuScopeInSum(timingReport, scope, sourceFrameIndex)) {
      scopeSumsByFrame[sourceFrameIndex] += static_cast<double>(ms);
    }
    return true;
  };
  const auto addAlias = [&](BenchmarkMetricIndex index, GpuTimingScope scope,
                            uint64_t sourceFrameIndex, float ms) {
    if (!hasGpuTimingScope(timingReport, scope)) {
      return;
    }
    const auto frameIt = frameByIndex.find(sourceFrameIndex);
    if (frameIt == frameByIndex.end()) {
      return;
    }
    report.frames[frameIt->second].measurements.appendRegistered(
        index, static_cast<double>(ms));
  };
  add(NURI_BENCHMARK_METRIC("gpu.frame_ms"), GpuTimingScope::WholeFrame,
      timingReport.wholeFrameSourceFrameIndex, timingReport.wholeFrameTimeMs);
  const bool shadowAdded =
      add(NURI_BENCHMARK_METRIC("gpu.scopes.shadow_ms"), GpuTimingScope::Shadow,
          timingReport.shadowSourceFrameIndex, timingReport.shadowTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.shadow_depth_ms"),
      GpuTimingScope::ShadowDepth, timingReport.shadowDepthSourceFrameIndex,
      timingReport.shadowDepthTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.shadow_sdsm_ms"),
      GpuTimingScope::ShadowSdsm, timingReport.shadowSdsmSourceFrameIndex,
      timingReport.shadowSdsmTimeMs);
  const bool opaqueAdded =
      add(NURI_BENCHMARK_METRIC("gpu.scopes.opaque_ms"), GpuTimingScope::Opaque,
          timingReport.opaqueSourceFrameIndex, timingReport.opaqueTimeMs);
  const bool opaqueDepthAdded =
      add(NURI_BENCHMARK_METRIC("gpu.scopes.opaque_depth_ms"),
          GpuTimingScope::OpaqueDepth, timingReport.opaqueDepthSourceFrameIndex,
          timingReport.opaqueDepthTimeMs);
  const bool opaqueNormalAdded = add(
      NURI_BENCHMARK_METRIC("gpu.scopes.opaque_normal_ms"),
      GpuTimingScope::OpaqueNormal, timingReport.opaqueNormalSourceFrameIndex,
      timingReport.opaqueNormalTimeMs);
  addAlias(NURI_BENCHMARK_METRIC("renderer.ao.input_ms"),
           GpuTimingScope::OpaqueNormal,
           timingReport.opaqueNormalSourceFrameIndex,
           timingReport.opaqueNormalTimeMs);
  const bool opaqueMainAdded =
      add(NURI_BENCHMARK_METRIC("gpu.scopes.opaque_main_ms"),
          GpuTimingScope::OpaqueMain, timingReport.opaqueMainSourceFrameIndex,
          timingReport.opaqueMainTimeMs);
  const bool gtaoAdded =
      add(NURI_BENCHMARK_METRIC("gpu.scopes.gtao_ms"), GpuTimingScope::GTAO,
          timingReport.gtaoSourceFrameIndex, timingReport.gtaoTimeMs);
  add(NURI_BENCHMARK_METRIC("renderer.ao.prefilter_edges_ms"),
      GpuTimingScope::GTAOPrefilterEdges,
      timingReport.gtaoPrefilterEdgesSourceFrameIndex,
      timingReport.gtaoPrefilterEdgesTimeMs);
  add(NURI_BENCHMARK_METRIC("renderer.ao.main_ms"), GpuTimingScope::GTAOMain,
      timingReport.gtaoMainSourceFrameIndex, timingReport.gtaoMainTimeMs);
  add(NURI_BENCHMARK_METRIC("renderer.ao.denoise_ms"),
      GpuTimingScope::GTAODenoise, timingReport.gtaoDenoiseSourceFrameIndex,
      timingReport.gtaoDenoiseTimeMs);
  add(NURI_BENCHMARK_METRIC("renderer.ao.upscale_ms"),
      GpuTimingScope::GTAOUpscale, timingReport.gtaoUpscaleSourceFrameIndex,
      timingReport.gtaoUpscaleTimeMs);
  const bool msaaResolveAdded =
      add(NURI_BENCHMARK_METRIC("gpu.scopes.msaa_resolve_ms"),
          GpuTimingScope::MsaaResolve, timingReport.msaaResolveSourceFrameIndex,
          timingReport.msaaResolveTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.scene_color_downsample_ms"),
      GpuTimingScope::SceneColorDownsample,
      timingReport.sceneColorDownsampleSourceFrameIndex,
      timingReport.sceneColorDownsampleTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.taa_resolve_ms"),
      GpuTimingScope::TemporalAAResolve,
      timingReport.temporalAAResolveSourceFrameIndex,
      timingReport.temporalAAResolveTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.taa_debug_ms"),
      GpuTimingScope::TemporalAADebug,
      timingReport.temporalAADebugSourceFrameIndex,
      timingReport.temporalAADebugTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.spatial_aa_ms"),
      GpuTimingScope::SpatialAA, timingReport.spatialAASourceFrameIndex,
      timingReport.spatialAATimeMs);
  const bool transmissionAdded = add(
      NURI_BENCHMARK_METRIC("gpu.scopes.transmission_ms"),
      GpuTimingScope::Transmission, timingReport.transmissionSourceFrameIndex,
      timingReport.transmissionTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.hdr_postprocess_ms"),
      GpuTimingScope::HDRPostProcess,
      timingReport.hdrPostProcessSourceFrameIndex,
      timingReport.hdrPostProcessTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.skybox_ms"), GpuTimingScope::Skybox,
      timingReport.skyboxSourceFrameIndex, timingReport.skyboxTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.velocity_ms"), GpuTimingScope::Velocity,
      timingReport.velocitySourceFrameIndex, timingReport.velocityTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.reactive_mask_ms"),
      GpuTimingScope::ReactiveMask, timingReport.reactiveMaskSourceFrameIndex,
      timingReport.reactiveMaskTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.taa_copy_back_ms"),
      GpuTimingScope::TemporalAACopyBack,
      timingReport.temporalAACopyBackSourceFrameIndex,
      timingReport.temporalAACopyBackTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.gtao_temporal_ms"),
      GpuTimingScope::GTAOTemporal, timingReport.gtaoTemporalSourceFrameIndex,
      timingReport.gtaoTemporalTimeMs);
  addAlias(NURI_BENCHMARK_METRIC("renderer.ao.temporal_ms"),
           GpuTimingScope::GTAOTemporal,
           timingReport.gtaoTemporalSourceFrameIndex,
           timingReport.gtaoTemporalTimeMs);
  addAlias(NURI_BENCHMARK_METRIC("renderer.ao.upscale_ms"),
           GpuTimingScope::GTAOTemporal,
           timingReport.gtaoTemporalSourceFrameIndex,
           timingReport.gtaoTemporalTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.ray_tracing_scene_ms"),
      GpuTimingScope::RayTracingScene,
      timingReport.rayTracingSceneSourceFrameIndex,
      timingReport.rayTracingSceneTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.ray_tracing_blas_ms"),
      GpuTimingScope::RayTracingBLAS,
      timingReport.rayTracingBlasSourceFrameIndex,
      timingReport.rayTracingBlasTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.ray_tracing_tlas_ms"),
      GpuTimingScope::RayTracingTLAS,
      timingReport.rayTracingTlasSourceFrameIndex,
      timingReport.rayTracingTlasTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.ddgi_ms"), GpuTimingScope::DDGI,
      timingReport.ddgiSourceFrameIndex, timingReport.ddgiTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.ddgi_trace_ms"),
      GpuTimingScope::DDGITrace, timingReport.ddgiTraceSourceFrameIndex,
      timingReport.ddgiTraceTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.ddgi_update_ms"),
      GpuTimingScope::DDGIUpdate, timingReport.ddgiUpdateSourceFrameIndex,
      timingReport.ddgiUpdateTimeMs);
  add(NURI_BENCHMARK_METRIC("gpu.scopes.ddgi_relocate_classify_ms"),
      GpuTimingScope::DDGIRelocateClassify,
      timingReport.ddgiRelocateClassifySourceFrameIndex,
      timingReport.ddgiRelocateClassifyTimeMs);
  if (const auto frameIt =
          frameByIndex.find(timingReport.wholeFrameSourceFrameIndex);
      frameIt != frameByIndex.end()) {
    auto &measurements = report.frames[frameIt->second].measurements;
    const auto addAbsentPhase = [&](BenchmarkMetricIndex index, bool added) {
      if (!added) {
        measurements.appendRegistered(index, 0.0);
      }
    };
    addAbsentPhase(NURI_BENCHMARK_METRIC("gpu.scopes.shadow_ms"), shadowAdded);
    addAbsentPhase(NURI_BENCHMARK_METRIC("gpu.scopes.opaque_ms"), opaqueAdded);
    addAbsentPhase(NURI_BENCHMARK_METRIC("gpu.scopes.opaque_depth_ms"),
                   opaqueDepthAdded);
    addAbsentPhase(NURI_BENCHMARK_METRIC("gpu.scopes.opaque_normal_ms"),
                   opaqueNormalAdded);
    addAbsentPhase(NURI_BENCHMARK_METRIC("gpu.scopes.opaque_main_ms"),
                   opaqueMainAdded);
    addAbsentPhase(NURI_BENCHMARK_METRIC("gpu.scopes.gtao_ms"), gtaoAdded);
    addAbsentPhase(NURI_BENCHMARK_METRIC("gpu.scopes.msaa_resolve_ms"),
                   msaaResolveAdded);
    addAbsentPhase(NURI_BENCHMARK_METRIC("gpu.scopes.transmission_ms"),
                   transmissionAdded);
  }
  if (const auto frameIt = frameByIndex.find(
          timingReport.opaquePipelineStatisticsSourceFrameIndex);
      frameIt != frameByIndex.end()) {
    auto &measurements = report.frames[frameIt->second].measurements;
    measurements.appendRegistered(
        NURI_BENCHMARK_METRIC("renderer.opaque.pipeline_statistics_requested"),
        timingReport.opaquePipelineStatisticsRequested ? 1.0 : 0.0);
    measurements.appendRegistered(
        NURI_BENCHMARK_METRIC("renderer.opaque.pipeline_statistics_available"),
        timingReport.opaquePipelineStatisticsAvailable ? 1.0 : 0.0);
    if (timingReport.opaquePipelineStatisticsAvailable) {
      measurements.appendRegistered(
          NURI_BENCHMARK_METRIC(
              "renderer.opaque.pipeline_statistics_input_assembly_vertices"),
          static_cast<double>(timingReport.opaqueInputAssemblyVertices));
      measurements.appendRegistered(
          NURI_BENCHMARK_METRIC(
              "renderer.opaque.pipeline_statistics_input_assembly_primitives"),
          static_cast<double>(timingReport.opaqueInputAssemblyPrimitives));
      measurements.appendRegistered(
          NURI_BENCHMARK_METRIC(
              "renderer.opaque.pipeline_statistics_clipping_invocations"),
          static_cast<double>(timingReport.opaqueClippingInvocations));
      measurements.appendRegistered(
          NURI_BENCHMARK_METRIC(
              "renderer.opaque.pipeline_statistics_clipping_primitives"),
          static_cast<double>(timingReport.opaqueClippingPrimitives));
      measurements.appendRegistered(
          NURI_BENCHMARK_METRIC("renderer.opaque.pipeline_statistics_fragment_"
                                "shader_invocations"),
          static_cast<double>(timingReport.opaqueFragmentShaderInvocations));
    }
  }
  for (const auto &[sourceFrameIndex, sum] : scopeSumsByFrame) {
    const auto frameIt = frameByIndex.find(sourceFrameIndex);
    if (frameIt != frameByIndex.end()) {
      report.frames[frameIt->second].measurements.appendRegistered(
          NURI_BENCHMARK_METRIC("gpu.scopes_sum_ms"), sum);
    }
  }
  for (uint32_t orderedPassIndex = 0u;
       orderedPassIndex < timingReport.passTimings.size(); ++orderedPassIndex) {
    const GpuTimingReport::PassTiming &timing =
        timingReport.passTimings[orderedPassIndex];
    const auto frameIt = frameByIndex.find(timing.sourceFrameIndex);
    if (frameIt == frameByIndex.end()) {
      continue;
    }
    report.frames[frameIt->second].measurements.appendOwned(
        passTimingMetricId(orderedPassIndex, timing.debugName, "gpu_ms"),
        static_cast<double>(timing.timeMs));
  }
}

void drainGpuTimings(GPUDevice &gpu, BenchmarkReport &report,
                     const std::map<uint64_t, size_t> &frameByIndex) {
  std::array<GpuTimingReport, 32> reports{};
  size_t drained = 0u;
  do {
    drained = gpu.drainCompletedGpuTimingReports(reports);
    for (size_t i = 0u; i < drained; ++i) {
      applyGpuTimingReport(report, reports[i], frameByIndex);
    }
  } while (drained == reports.size());
}

[[nodiscard]] Result<bool, std::string>
populateScene(const BenchmarkCase &benchmarkCase, Renderer &renderer,
              RenderScene &scene, std::pmr::memory_resource *memory,
              SceneLoadHandle &sceneLoad) {
  (void)memory;
  scene.bindResources(&renderer.resources());
  auto lightResult = scene.graph().addLight(scene.graph().rootNode(),
                                            LightDesc{
                                                .type = LightType::Directional,
                                                .name = "benchmark_key",
                                                .color = glm::vec3(1.0f),
                                                .intensity = 4.0f,
                                                .enabled = true,
                                            });
  if (lightResult.hasError()) {
    return Result<bool, std::string>::makeError(lightResult.error());
  }

  if (benchmarkCase.scene.kind == "prefab") {
    if (benchmarkCase.scene.pathBase.empty() ||
        benchmarkCase.scene.path.empty()) {
      return Result<bool, std::string>::makeError(
          "prefab scene requires pathBase and path");
    }
    auto path = resolveBenchmarkPath(benchmarkCase.scene.pathBase,
                                     benchmarkCase.scene.path);
    if (path.hasError()) {
      return Result<bool, std::string>::makeError(path.error());
    }
    if (!std::filesystem::exists(path.value())) {
      return Result<bool, std::string>::makeError("missing scene asset: " +
                                                  path.value().string());
    }
    SceneImportOptions importOptions{};
    importOptions.assetBuildOptions.flipUVs = benchmarkCase.scene.flipUVs;
    importOptions.assetBuildOptions.generateMeshlets =
        benchmarkCase.scene.generateMeshlets;
    importOptions.assetBuildOptions.meshletMaxVertices =
        benchmarkCase.scene.meshletMaxVertices;
    importOptions.assetBuildOptions.meshletMaxPrimitives =
        benchmarkCase.scene.meshletMaxPrimitives;
    importOptions.assetBuildOptions.meshletConeWeight =
        benchmarkCase.scene.meshletConeWeight;
    auto requested = renderer.assets().requestScene(SceneLoadRequest{
        .path = path.value().string(),
        .importOptions = importOptions,
        .priority = AssetPriority::Critical,
        .publication = ScenePublicationPolicy::CompleteOnly,
        .failurePolicy = SceneFailurePolicy::BestEffort,
        .debugName = benchmarkCase.scene.path.string(),
    });
    if (requested.hasError()) {
      return Result<bool, std::string>::makeError(requested.error());
    }
    sceneLoad = requested.value();
  }

  const bool ddgiMicro = benchmarkCase.scene.generator.starts_with(
                             "nuri.procedural.ddgi_benchmark_") &&
                         benchmarkCase.scene.generator.ends_with(".v1");
  if (ddgiMicro) {
    auto planePath =
        resolveBenchmarkPath("modelsRoot", "common/flat_plane.obj");
    if (planePath.hasError()) {
      return Result<bool, std::string>::makeError(planePath.error());
    }
    auto model = renderer.resources().acquireModel(ModelRequest{
        .path = planePath.value().string(),
        .debugName = "benchmark_ddgi_flat_plane",
    });
    if (model.hasError()) {
      return Result<bool, std::string>::makeError(model.error());
    }
    const bool alphaFoliage = benchmarkCase.scene.generator ==
                              "nuri.procedural.ddgi_benchmark_alpha.v1";
    const auto acquireMaterial =
        [&](std::string_view name, const glm::vec4 &color,
            MaterialAlphaMode alphaMode =
                MaterialAlphaMode::Opaque) -> Result<MaterialRef, std::string> {
      MaterialRequest request{};
      request.debugName = std::string(name);
      request.desc.baseColorFactor = color;
      request.desc.emissiveFactor = glm::vec3(color) * 0.2f;
      request.desc.emissiveStrength = 1.0f;
      request.desc.metallicFactor = 0.0f;
      request.desc.roughnessFactor = 0.75f;
      request.desc.alphaMode = alphaMode;
      request.desc.alphaCutoff = 0.5f;
      request.desc.doubleSided = true;
      return renderer.resources().acquireMaterial(request);
    };
    auto neutral = acquireMaterial("benchmark_ddgi_neutral",
                                   glm::vec4(0.65f, 0.67f, 0.70f, 1.0f));
    auto red = acquireMaterial(
        "benchmark_ddgi_red",
        alphaFoliage ? glm::vec4(0.82f, 0.16f, 0.10f, 0.2f)
                     : glm::vec4(0.82f, 0.16f, 0.10f, 1.0f),
        alphaFoliage ? MaterialAlphaMode::Mask : MaterialAlphaMode::Opaque);
    auto blue = acquireMaterial("benchmark_ddgi_blue",
                                glm::vec4(0.10f, 0.28f, 0.82f, 1.0f));
    if (neutral.hasError() || red.hasError() || blue.hasError()) {
      return Result<bool, std::string>::makeError(
          "failed to create DDGI benchmark materials");
    }
    const glm::mat4 upright = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f),
                                          glm::vec3(1.0f, 0.0f, 0.0f));
    const auto addPlane =
        [&](std::string_view name, const glm::mat4 &orientation,
            const glm::vec3 &translation, const glm::vec3 &scale,
            MaterialRef material) -> Result<bool, std::string> {
      const glm::mat4 transform = glm::translate(glm::mat4(1.0f), translation) *
                                  orientation *
                                  glm::scale(glm::mat4(1.0f), scale);
      auto node =
          scene.graph().createNode(scene.graph().rootNode(), name, transform);
      if (node.hasError()) {
        return Result<bool, std::string>::makeError(node.error());
      }
      auto renderable =
          scene.graph().addRenderable(node.value(), model.value(), material);
      return renderable.hasError()
                 ? Result<bool, std::string>::makeError(renderable.error())
                 : Result<bool, std::string>::makeResult(true);
    };
    for (const auto &[name, orientation, translation, scale, material] :
         {std::tuple<std::string_view, glm::mat4, glm::vec3, glm::vec3,
                     MaterialRef>{"DDGI Benchmark Floor", glm::mat4(1.0f),
                                  glm::vec3(0.0f, -0.75f, 0.0f),
                                  glm::vec3(5.0f, 1.0f, 4.0f), neutral.value()},
          {"DDGI Benchmark Wall", upright, glm::vec3(0.0f, 0.65f, -1.4f),
           glm::vec3(5.0f, 2.8f, 1.0f), neutral.value()},
          {"DDGI Benchmark Red Caster", upright, glm::vec3(-0.9f, 0.2f, 0.1f),
           glm::vec3(0.7f, 1.6f, 1.0f), red.value()},
          {"DDGI Benchmark Blue Caster", upright, glm::vec3(0.4f, 0.3f, 0.0f),
           glm::vec3(0.7f, 1.9f, 1.0f), blue.value()}}) {
      auto added = addPlane(name, orientation, translation, scale, material);
      if (added.hasError()) {
        return added;
      }
    }
    if (alphaFoliage) {
      for (uint32_t index = 0u; index < 16u; ++index) {
        const float x = -1.8f + 0.24f * static_cast<float>(index);
        const float z = -0.9f + 0.1f * static_cast<float>(index % 5u);
        auto added = addPlane("DDGI Benchmark Alpha Foliage", upright,
                              glm::vec3(x, 0.15f, z),
                              glm::vec3(0.35f, 1.5f, 1.0f), red.value());
        if (added.hasError()) {
          return added;
        }
      }
    }
    if (benchmarkCase.scene.generator ==
        "nuri.procedural.ddgi_benchmark_lights.v1") {
      for (uint32_t index = 0u; index < 16u; ++index) {
        const float x = -2.0f + 0.27f * static_cast<float>(index);
        auto light = scene.graph().addLight(
            scene.graph().rootNode(),
            LightDesc{.type = LightType::Point,
                      .name = "benchmark_ddgi_local",
                      .position = glm::vec3(x, 0.9f, -0.5f),
                      .color = glm::vec3(1.0f, 0.65f, 0.35f),
                      .intensity = 8.0f,
                      .range = 5.0f,
                      .enabled = true});
        if (light.hasError()) {
          return Result<bool, std::string>::makeError(light.error());
        }
      }
    }
    uint32_t volumeCount = 1u;
    if (benchmarkCase.scene.generator ==
        "nuri.procedural.ddgi_benchmark_four_volume.v1") {
      volumeCount = 4u;
    } else if (benchmarkCase.scene.generator ==
               "nuri.procedural.ddgi_benchmark_eight_volume.v1") {
      volumeCount = 8u;
    }
    for (uint32_t index = 0u; index < volumeCount; ++index) {
      const glm::vec3 offset =
          volumeCount == 1u ? glm::vec3(0.0f)
                            : glm::vec3((index & 1u) != 0u ? 1.4f : -1.4f,
                                        (index & 4u) != 0u ? 0.5f : -0.5f,
                                        (index & 2u) != 0u ? 1.0f : -1.0f);
      auto node = scene.graph().createNode(
          scene.graph().rootNode(), "DDGI Benchmark Volume",
          glm::translate(glm::mat4(1.0f), offset));
      if (node.hasError()) {
        return Result<bool, std::string>::makeError(node.error());
      }
      auto volume = scene.graph().addDDGIVolume(
          node.value(),
          DDGIVolumeDesc{.name = "DDGI Benchmark Volume",
                         .probeCounts = volumeCount == 1u
                                            ? glm::uvec3(12u, 6u, 12u)
                                            : glm::uvec3(8u, 4u, 8u),
                         .probeSpacing = {0.65f, 0.65f, 0.65f},
                         .blendDistance = 0.65f,
                         .maxRayDistance = 12.0f,
                         .priority = static_cast<int32_t>(index)});
      if (volume.hasError()) {
        return Result<bool, std::string>::makeError(volume.error());
      }
    }
  }

  if (benchmarkCase.scene.generator == "nuri.ddgi.volume.v1") {
    auto volumeNode = scene.graph().createNode(
        scene.graph().rootNode(), "Benchmark DDGI Volume", glm::mat4(1.0f));
    if (volumeNode.hasError()) {
      return Result<bool, std::string>::makeError(volumeNode.error());
    }
    auto volume = scene.graph().addDDGIVolume(
        volumeNode.value(), DDGIVolumeDesc{.name = "Benchmark DDGI Volume",
                                           .probeCounts = {24u, 12u, 24u},
                                           .probeSpacing = {4.0f, 4.0f, 4.0f},
                                           .blendDistance = 8.0f,
                                           .maxRayDistance = 120.0f});
    if (volume.hasError()) {
      return Result<bool, std::string>::makeError(volume.error());
    }
  }

  if (benchmarkCase.scene.generator ==
      "nuri.procedural.specular_minification.v1") {
    nuri::tools::runtime::ToolRuntimeDesc runtimeDesc{};
    runtimeDesc.scene.kind = benchmarkCase.scene.kind;
    runtimeDesc.scene.generator = benchmarkCase.scene.generator;
    runtimeDesc.resolvePath = resolveBenchmarkPath;
    auto populated =
        nuri::tools::runtime::populateSpecularMinificationToolScene(
            runtimeDesc, renderer, scene);
    if (populated.hasError()) {
      return populated;
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

[[nodiscard]] Result<bool, std::string>
waitForBenchmarkAssets(Renderer &renderer, RenderScene &scene,
                       SceneLoadHandle sceneLoad) {
  if (!isValid(sceneLoad)) {
    return Result<bool, std::string>::makeResult(true);
  }
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(120);
  for (;;) {
    const SceneLoadSnapshot status = renderer.assets().query(sceneLoad);
    if (status.terminal()) {
      if (status.state == SceneLoadState::Complete ||
          status.state == SceneLoadState::CompleteWithErrors) {
        return Result<bool, std::string>::makeResult(true);
      }
      std::ostringstream message;
      message << "benchmark async scene load failed: state="
              << static_cast<uint32_t>(status.state)
              << " progress=" << status.progress;
      if (!status.error.empty()) {
        message << " error=" << status.error;
      }
      return Result<bool, std::string>::makeError(message.str());
    }
    auto pumped = renderer.assets().prepareFrame(AssetPublicationContext{
        .scene = &scene,
    });
    if (pumped.hasError()) {
      return Result<bool, std::string>::makeError(pumped.error());
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      const SceneLoadSnapshot timedOut = renderer.assets().query(sceneLoad);
      std::ostringstream message;
      message << "benchmark async scene load timed out: state="
              << static_cast<uint32_t>(timedOut.state)
              << " progress=" << timedOut.progress
              << " models=" << timedOut.models.published << "/"
              << timedOut.models.total
              << " materials=" << timedOut.materials.published << "/"
              << timedOut.materials.total
              << " textures=" << timedOut.textures.published << "/"
              << timedOut.textures.total;
      return Result<bool, std::string>::makeError(message.str());
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

[[nodiscard]] glm::vec3 normalizedOrDefault(const glm::vec3 &value,
                                            const glm::vec3 &fallback) {
  return glm::length(value) > 1.0e-6f ? glm::normalize(value) : fallback;
}

[[nodiscard]] BenchmarkCameraConfig
cameraFromKeyframe(const BenchmarkCase &benchmarkCase,
                   const BenchmarkCameraKeyframe &keyframe) {
  BenchmarkCameraConfig camera = benchmarkCase.camera;
  camera.position = keyframe.position;
  if (keyframe.hasTarget) {
    camera.target = keyframe.target;
    camera.hasTarget = true;
    camera.direction = normalizedOrDefault(keyframe.target - keyframe.position,
                                           camera.direction);
  }
  return camera;
}

[[nodiscard]] bool frameInPath(const BenchmarkCameraPath &path,
                               uint32_t frame) {
  return frame >= path.startFrame && frame <= path.endFrame &&
         !path.keyframes.empty();
}

[[nodiscard]] float smoothstep(float t) { return t * t * (3.0f - 2.0f * t); }

[[nodiscard]] Result<BenchmarkCameraConfig, std::string>
evaluatePath(const BenchmarkCase &benchmarkCase,
             const BenchmarkCameraPath &path, uint32_t frame) {
  if (path.keyframes.empty()) {
    return Result<BenchmarkCameraConfig, std::string>::makeError(
        "camera path '" + path.id + "' has no keyframes");
  }

  const auto upper =
      std::lower_bound(path.keyframes.begin(), path.keyframes.end(), frame,
                       [](const BenchmarkCameraKeyframe &keyframe,
                          uint32_t value) { return keyframe.frame < value; });
  if (upper == path.keyframes.begin()) {
    return Result<BenchmarkCameraConfig, std::string>::makeResult(
        cameraFromKeyframe(benchmarkCase, *upper));
  }
  if (upper == path.keyframes.end()) {
    return Result<BenchmarkCameraConfig, std::string>::makeResult(
        cameraFromKeyframe(benchmarkCase, path.keyframes.back()));
  }

  const BenchmarkCameraKeyframe &rhs = *upper;
  const BenchmarkCameraKeyframe &lhs = *(upper - 1);
  const uint32_t span = rhs.frame - lhs.frame;
  float t = span == 0u ? 0.0f
                       : static_cast<float>(frame - lhs.frame) /
                             static_cast<float>(span);
  if (path.interpolation == "smoothstep") {
    t = smoothstep(t);
  } else if (path.interpolation != "linear") {
    return Result<BenchmarkCameraConfig, std::string>::makeError(
        "camera path '" + path.id + "' has unsupported interpolation '" +
        path.interpolation + "'");
  }

  BenchmarkCameraConfig camera = benchmarkCase.camera;
  camera.position = lhs.position + (rhs.position - lhs.position) * t;
  if (lhs.hasTarget && rhs.hasTarget) {
    camera.target = lhs.target + (rhs.target - lhs.target) * t;
    camera.hasTarget = true;
    camera.direction =
        normalizedOrDefault(camera.target - camera.position, camera.direction);
  }
  return Result<BenchmarkCameraConfig, std::string>::makeResult(camera);
}

[[nodiscard]] Result<BenchmarkCameraConfig, std::string>
evaluateBenchmarkCameraAtFrame(const BenchmarkCase &benchmarkCase,
                               uint32_t frame) {
  BenchmarkCameraConfig camera = benchmarkCase.camera;
  for (const BenchmarkCameraPath &path : benchmarkCase.timeline.cameraPaths) {
    if (!frameInPath(path, frame)) {
      continue;
    }
    auto evaluated = evaluatePath(benchmarkCase, path, frame);
    if (evaluated.hasError()) {
      return evaluated;
    }
    camera = evaluated.value();
  }
  return Result<BenchmarkCameraConfig, std::string>::makeResult(camera);
}

void applyBenchmarkCamera(Camera &camera,
                          const BenchmarkCameraConfig &cameraConfig) {
  camera.setPerspective(PerspectiveParams{
      .fovYRadians = glm::radians(cameraConfig.verticalFovDegrees),
      .nearPlane = cameraConfig.nearPlane,
      .farPlane = cameraConfig.farPlane,
  });
  const glm::vec3 direction =
      normalizedOrDefault(cameraConfig.direction, glm::vec3(0.0f, 0.0f, -1.0f));
  const glm::vec3 target = cameraConfig.hasTarget
                               ? cameraConfig.target
                               : cameraConfig.position + direction;
  camera.setLookAt(cameraConfig.position, target, glm::vec3(0.0f, 1.0f, 0.0f));
}

[[nodiscard]] Camera makeBenchmarkCamera(const BenchmarkCase &benchmarkCase) {
  Camera camera;
  applyBenchmarkCamera(camera, benchmarkCase.camera);
  return camera;
}

void buildFrameContext(RenderFrameContext &frameContext, RenderScene &scene,
                       Renderer &renderer, RenderSettings &settings,
                       TemporalFrameService &temporalFrameService,
                       const Camera &camera, uint64_t frameIndex,
                       double timeSeconds, double deltaSeconds, uint32_t width,
                       uint32_t height) {
  sanitizeBenchmarkRenderSettings(settings);
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
      TemporalCameraFrameDesc{
          .renderExtent = glm::uvec2(width, height),
          .sceneContent = sceneContent,
      },
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

[[nodiscard]] double elapsedMs(std::chrono::steady_clock::time_point begin,
                               std::chrono::steady_clock::time_point end =
                                   std::chrono::steady_clock::now()) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

} // namespace

Result<bool, BenchmarkExitCode>
checkBenchmarkGpuRequirements(const BenchmarkRequirements &requirements,
                              const GpuMultisampleCapabilities &capabilities,
                              std::string &message) {
  const uint32_t samples = requirements.msaaSamples.value_or(1u);
  if (samples == 1u) {
    return Result<bool, BenchmarkExitCode>::makeResult(true);
  }
  std::vector<std::string_view> missing;
  if (samples == 4u) {
    if (!capabilities.sample4Color) {
      missing.push_back("sample4_color");
    }
    if (!capabilities.sample4Depth) {
      missing.push_back("sample4_depth");
    }
  } else {
    if (!capabilities.sample8Color) {
      missing.push_back("sample8_color");
    }
    if (!capabilities.sample8Depth) {
      missing.push_back("sample8_depth");
    }
  }
  if (!capabilities.depthResolveMin) {
    missing.push_back("depth_resolve_min");
  }
  if (!capabilities.alphaToCoverage) {
    missing.push_back("alpha_to_coverage");
  }
  if (missing.empty()) {
    return Result<bool, BenchmarkExitCode>::makeResult(true);
  }
  message =
      "required MSAA" + std::to_string(samples) + "x capability unavailable:";
  for (const std::string_view capability : missing) {
    message += " ";
    message += capability;
  }
  return Result<bool, BenchmarkExitCode>::makeError(
      BenchmarkExitCode::EnvironmentUnavailable);
}

Result<std::string, std::string>
formatBenchmarkCaseListJson(const std::vector<BenchmarkCase> &cases,
                            std::string_view suite) {
  BenchmarkReport dummy{};
  (void)dummy;
  std::ostringstream out;
  out << "{\n  \"cases\": [\n";
  bool first = true;
  for (const BenchmarkCase *benchmarkCase :
       filterBenchmarkCasesBySuite(cases, suite)) {
    if (!first) {
      out << ",\n";
    }
    first = false;
    out << "    {\"id\": \"" << benchmarkCase->id << "\", \"suite\": \""
        << benchmarkCase->suite << "\", \"comparisonGroup\": \""
        << benchmarkCase->comparisonGroup << "\", \"variant\": \""
        << benchmarkCase->variant << "\", \"description\": \""
        << benchmarkCase->description << "\", \"msaaSamples\": ";
    if (benchmarkCase->requirements.msaaSamples.has_value()) {
      out << *benchmarkCase->requirements.msaaSamples;
    } else {
      out << "null";
    }
    out << "}";
  }
  out << "\n  ]\n}\n";
  return Result<std::string, std::string>::makeResult(out.str());
}

std::string formatBenchmarkCaseListText(const std::vector<BenchmarkCase> &cases,
                                        std::string_view suite) {
  std::ostringstream out;
  for (const BenchmarkCase *benchmarkCase :
       filterBenchmarkCasesBySuite(cases, suite)) {
    out << benchmarkCase->id << " [" << benchmarkCase->suite << "] "
        << benchmarkCase->description << " (MSAA requirement: ";
    if (benchmarkCase->requirements.msaaSamples.has_value()) {
      out << *benchmarkCase->requirements.msaaSamples << "x";
    } else {
      out << "none";
    }
    out << ")\n";
  }
  return out.str();
}

Result<std::string, std::string>
formatBenchmarkCaseExplanationJson(const BenchmarkCase &benchmarkCase) {
  std::ostringstream out;
  out << "{\n"
      << "  \"id\": \"" << benchmarkCase.id << "\",\n"
      << "  \"suite\": \"" << benchmarkCase.suite << "\",\n"
      << "  \"comparisonGroup\": \"" << benchmarkCase.comparisonGroup << "\",\n"
      << "  \"variant\": \"" << benchmarkCase.variant << "\",\n"
      << "  \"description\": \"" << benchmarkCase.description << "\",\n"
      << "  \"sceneKind\": \"" << benchmarkCase.scene.kind << "\",\n"
      << "  \"backend\": \"" << benchmarkCase.backend << "\",\n"
      << "  \"msaaSamples\": ";
  if (benchmarkCase.requirements.msaaSamples.has_value()) {
    out << *benchmarkCase.requirements.msaaSamples;
  } else {
    out << "null";
  }
  out << ",\n"
      << "  \"samples\": " << benchmarkCase.samples << ",\n"
      << "  \"measurementFrames\": " << benchmarkCase.measurementFrames << "\n"
      << "}\n";
  return Result<std::string, std::string>::makeResult(out.str());
}

std::string
formatBenchmarkCaseExplanationText(const BenchmarkCase &benchmarkCase) {
  std::ostringstream out;
  out << benchmarkCase.id << "\n"
      << "suite: " << benchmarkCase.suite << "\n"
      << "comparison group: " << benchmarkCase.comparisonGroup << "\n"
      << "variant: " << benchmarkCase.variant << "\n"
      << "description: " << benchmarkCase.description << "\n"
      << "scene: " << benchmarkCase.scene.kind << "\n"
      << "backend: " << benchmarkCase.backend << "\n"
      << "MSAA requirement: ";
  if (benchmarkCase.requirements.msaaSamples.has_value()) {
    out << *benchmarkCase.requirements.msaaSamples << "x\n";
  } else {
    out << "none\n";
  }
  out << "resolution: " << benchmarkCase.resolution[0] << "x"
      << benchmarkCase.resolution[1] << "\n"
      << "frames: warmup=" << benchmarkCase.warmupFrames
      << " measured=" << benchmarkCase.measurementFrames << "\n";
  return out.str();
}

Result<std::string, std::string>
formatEffectiveConfigJson(const BenchmarkCase &benchmarkCase,
                          const BenchmarkRunOptions &options) {
  std::string backendSource;
  const std::string backend = resolveBackendName(benchmarkCase, backendSource);
  std::string presentSource;
  const std::string present = resolvePresentMode(benchmarkCase, presentSource);
  std::ostringstream out;
  out << "{\n"
      << "  \"case\": \"" << benchmarkCase.id << "\",\n"
      << "  \"comparisonGroup\": \"" << benchmarkCase.comparisonGroup << "\",\n"
      << "  \"variant\": \"" << benchmarkCase.variant << "\",\n"
      << "  \"baselineProfile\": \"" << options.baselineProfileId << "\",\n"
      << "  \"profileAuthoritative\": "
      << (options.baselineProfileAuthoritative ? "true" : "false") << ",\n"
      << "  \"authoritative\": false,\n"
      << "  \"minimumIndependentRepetitions\": "
      << options.baselineProfileMinimumRepetitions << ",\n"
      << "  \"completedIndependentRepetitions\": 0,\n"
      << "  \"requestedIndependentRepetitions\": "
      << options.isolatedRepetitions.value_or(0u) << ",\n"
      << "  \"repetitionTimeoutMs\": " << options.repetitionTimeout.count()
      << ",\n"
      << "  \"warmupStabilityPolicy\": \""
      << options.baselineProfileWarmupStability << "\",\n"
      << "  \"warmupStabilityStatus\": \"unknown\",\n"
      << "  \"warmupWindowFrames\": "
      << options.baselineProfileWarmupWindowFrames << ",\n"
      << "  \"warmupMaxDriftPercent\": "
      << options.baselineProfileWarmupMaxDriftPercent << ",\n"
      << "  \"sampleWindows\": "
      << options.samplesOverride.value_or(benchmarkCase.samples) << ",\n"
      << "  \"backend\": \"" << backend << "\",\n"
      << "  \"backendSource\": \"" << backendSource << "\",\n"
      << "  \"presentMode\": \"" << present << "\",\n"
      << "  \"presentModeSource\": \"" << presentSource << "\",\n"
      << "  \"requiredMsaaSamples\": "
      << benchmarkCase.requirements.msaaSamples.value_or(0u) << ",\n"
      << "  \"opaquePipelineStatisticsDiagnostic\": "
      << (options.opaquePipelineStatisticsDiagnostic ? "true" : "false")
      << ",\n"
      << "  \"artifactDir\": \"" << options.artifactDir.generic_string()
      << "\"\n"
      << "}\n";
  return Result<std::string, std::string>::makeResult(out.str());
}

namespace {

[[nodiscard]] BenchmarkExitCode
benchmarkExitFromProcess(const nuri::tools::core::ProcessResult &process) {
  if (process.status != nuri::tools::core::ProcessStatus::Exited ||
      !process.exitCode.has_value() || *process.exitCode < 0 ||
      *process.exitCode >
          static_cast<int>(BenchmarkExitCode::MissingBaseline)) {
    return BenchmarkExitCode::RuntimeError;
  }
  return static_cast<BenchmarkExitCode>(*process.exitCode);
}

[[nodiscard]] BenchmarkExitCode
benchmarkExitFromOutcome(nuri::tools::core::ToolOutcome outcome) {
  using nuri::tools::core::ToolOutcome;
  switch (outcome) {
  case ToolOutcome::Pass:
  case ToolOutcome::Warn:
  case ToolOutcome::Investigative:
    return BenchmarkExitCode::Success;
  case ToolOutcome::Failure:
    return BenchmarkExitCode::Regression;
  case ToolOutcome::Invalid:
    return BenchmarkExitCode::InvalidInput;
  case ToolOutcome::EnvironmentUnavailable:
    return BenchmarkExitCode::EnvironmentUnavailable;
  case ToolOutcome::RuntimeError:
    return BenchmarkExitCode::RuntimeError;
  case ToolOutcome::MissingBaseline:
    return BenchmarkExitCode::MissingBaseline;
  case ToolOutcome::Cancelled:
  case ToolOutcome::Incomplete:
    return BenchmarkExitCode::RuntimeError;
  }
  return BenchmarkExitCode::RuntimeError;
}

[[nodiscard]] bool sameIsolatedEnvironment(const BenchmarkEnvironment &lhs,
                                           const BenchmarkEnvironment &rhs) {
  return lhs.commitHash == rhs.commitHash && lhs.dirty == rhs.dirty &&
         lhs.osName == rhs.osName && lhs.osVersion == rhs.osVersion &&
         lhs.cpuName == rhs.cpuName &&
         lhs.cpuLogicalThreadCount == rhs.cpuLogicalThreadCount &&
         lhs.gpuBackend == rhs.gpuBackend &&
         lhs.gpuDeviceName == rhs.gpuDeviceName &&
         lhs.gpuVendorId == rhs.gpuVendorId &&
         lhs.gpuDeviceId == rhs.gpuDeviceId &&
         lhs.gpuDriverVersion == rhs.gpuDriverVersion &&
         lhs.swapchainImageCount == rhs.swapchainImageCount &&
         lhs.resolvedPresentMode == rhs.resolvedPresentMode &&
         lhs.windowMode == rhs.windowMode &&
         lhs.windowVisible == rhs.windowVisible &&
         lhs.renderGraphWorkerCount == rhs.renderGraphWorkerCount &&
         lhs.renderGraphParallelCompile == rhs.renderGraphParallelCompile &&
         lhs.renderGraphParallelRecording == rhs.renderGraphParallelRecording &&
         lhs.buildType == rhs.buildType &&
         lhs.cmakeToolProfile == rhs.cmakeToolProfile &&
         lhs.vcpkgManifestFeatures == rhs.vcpkgManifestFeatures &&
         lhs.buildShared == rhs.buildShared &&
         lhs.loggingEnabled == rhs.loggingEnabled &&
         lhs.assertsEnabled == rhs.assertsEnabled &&
         lhs.tracyEnabled == rhs.tracyEnabled &&
         lhs.tracyDiagnostic == rhs.tracyDiagnostic &&
         lhs.devChecks == rhs.devChecks;
}

[[nodiscard]] std::optional<bool>
repetitionWarmupStatus(const BenchmarkReport &report) {
  if (report.sampleStats.empty()) {
    return std::nullopt;
  }
  bool allStable = true;
  for (const BenchmarkSampleStats &sample : report.sampleStats) {
    if (!sample.warmupStable.has_value()) {
      return std::nullopt;
    }
    if (!*sample.warmupStable) {
      allStable = false;
    }
  }
  return allStable;
}

[[nodiscard]] nuri::tools::core::BaselineProfileObservedEnvironment
observedProfileEnvironment(const BenchmarkEnvironment &environment) {
  std::string profiling = "off";
  if (environment.tracyEnabled) {
    profiling = "cpu";
  }
  return {
      .os = environment.osName,
      .backend = environment.gpuBackend,
      .backendSource = environment.gpuBackendSource,
      .windowMode = environment.windowMode,
      .windowVisible = environment.windowVisible,
      .gpuVendorId = environment.gpuVendorId,
      .gpuDeviceId = environment.gpuDeviceId,
      .driver = environment.gpuDriverVersion,
      .presentMode = environment.resolvedPresentMode,
      .profiling = std::move(profiling),
      .devChecks = environment.devChecks,
      .dirtyTree = environment.dirty,
  };
}

[[nodiscard]] std::string
processFailureMessage(const nuri::tools::core::ProcessResult &process) {
  using nuri::tools::core::ProcessStatus;
  switch (process.status) {
  case ProcessStatus::TimedOut:
    return "repetition timed out";
  case ProcessStatus::SpawnFailed:
    return "failed to spawn repetition: " + process.errorMessage;
  case ProcessStatus::InternalError:
    return "repetition process error: " + process.errorMessage;
  case ProcessStatus::Exited:
    if (process.exitCode.has_value()) {
      return "repetition exited with code " + std::to_string(*process.exitCode);
    }
    return "repetition exited without an exit code";
  }
  return "unknown repetition process error";
}

[[nodiscard]] bool
summarizeRenderDocChromeTrace(const std::filesystem::path &path,
                              BenchmarkRenderDocReport &renderDoc) {
  std::ifstream file(path, std::ios::binary);
  std::ostringstream text;
  text << file.rdbuf();
  std::string json = text.str();
  yyjson_read_err error{};
  std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)> document(
      yyjson_read_opts(json.data(), json.size(), 0u, nullptr, &error),
      &yyjson_doc_free);
  if (!document) {
    return false;
  }
  yyjson_val *events =
      yyjson_obj_get(yyjson_doc_get_root(document.get()), "traceEvents");
  if (!yyjson_is_arr(events)) {
    return false;
  }
  renderDoc.chromeEventCount = static_cast<uint32_t>(yyjson_arr_size(events));
  size_t index = 0u;
  size_t maximum = 0u;
  yyjson_val *event = nullptr;
  yyjson_arr_foreach(events, index, maximum, event) {
    const char *rawName = yyjson_get_str(yyjson_obj_get(event, "name"));
    if (rawName == nullptr) {
      continue;
    }
    const std::string_view name(rawName);
    renderDoc.drawCallCount += name.starts_with("vkCmdDraw");
    renderDoc.dispatchCallCount += name.starts_with("vkCmdDispatch");
    renderDoc.barrierCallCount +=
        name.find("Barrier") != std::string_view::npos;
    renderDoc.copyCallCount += name.starts_with("vkCmdCopy") ||
                               name.starts_with("vkCmdBlit") ||
                               name.starts_with("vkCmdResolve");
  }
  return true;
}

} // namespace

BenchmarkRunResult
runBenchmarkCaseIsolated(BenchmarkCase benchmarkCase,
                         const BenchmarkRunOptions &options) {
  BenchmarkRunResult result{};
  const uint32_t repetitions = options.isolatedRepetitions.value_or(0u);
  std::string runId = nuri::tools::core::createRunId();
  std::filesystem::path artifactDir = options.artifactDir;
  if (artifactDir.empty()) {
    auto workspace = nuri::tools::core::createRunWorkspace(
        benchmarkRepoRoot() / "artifacts" / "bench");
    if (workspace.hasError()) {
      result.exitCode = BenchmarkExitCode::RuntimeError;
      result.message = workspace.error();
      return result;
    }
    runId = workspace.value().runId;
    artifactDir = workspace.value().root;
  } else {
    std::error_code createError;
    std::filesystem::create_directories(artifactDir, createError);
    if (createError) {
      result.exitCode = BenchmarkExitCode::RuntimeError;
      result.message = "failed to create benchmark artifact directory: " +
                       createError.message();
      return result;
    }
  }

  result.reportPath = options.jsonOut.empty()
                          ? artifactDir / "cases" / (benchmarkCase.id + ".json")
                          : options.jsonOut;
  result.envelopePath = options.envelopeOut.empty() ? artifactDir / "run.json"
                                                    : options.envelopeOut;
  if (result.envelopePath.lexically_normal() ==
      result.reportPath.lexically_normal()) {
    result.envelopePath = artifactDir / "result.json";
  }

  BenchmarkReport report{};
  report.generatedAtUtc = utcTimestampIso8601();
  report.command = options.command;
  report.benchmarkCase = benchmarkCase;
  report.run.samples = options.samplesOverride.value_or(benchmarkCase.samples);
  report.run.warmupFrames = benchmarkCase.warmupFrames;
  report.run.measurementFrames = benchmarkCase.measurementFrames;
  report.run.cooldownFrames = benchmarkCase.cooldownFrames;
  report.run.maxDrainFrames = benchmarkCase.maxDrainFrames;
  report.run.drainTimeoutMs = benchmarkCase.drainTimeoutMs;
  report.run.fixedDeltaSeconds = benchmarkCase.fixedDeltaSeconds;
  report.run.validForComparison = false;
  report.artifacts.artifactDir = artifactDir;
  report.profile.id = options.baselineProfileId;
  report.profile.profileAuthoritative = options.baselineProfileAuthoritative;
  report.profile.minimumRepetitions = options.baselineProfileMinimumRepetitions;
  report.profile.completedRepetitions = 0u;
  report.profile.repetitionRequirementSatisfied = false;
  report.profile.repetitionUnit = "isolated-process";
  report.profile.warmupStabilityPolicy = options.baselineProfileWarmupStability;
  report.profile.warmupStabilityStatus = "unknown";
  report.profile.warmupWindowFrames = options.baselineProfileWarmupWindowFrames;
  report.profile.warmupMaxDriftPercent =
      options.baselineProfileWarmupMaxDriftPercent;
  report.profile.requiredMetrics = options.baselineProfileRequiredMetrics;
  report.repeatObservations.unit = "isolated-process";
  report.repeatObservations.independent = true;
  report.repeatObservations.count = 0u;

  const auto finalize = [&]() -> BenchmarkRunResult {
    return finalizeBenchmarkCaseResult(std::move(result), std::move(report),
                                       artifactDir, runId,
                                       options.verboseFrames);
  };

  if (repetitions == 0u) {
    result.exitCode = BenchmarkExitCode::InvalidInput;
    result.message = "repetitions must be greater than zero";
    report.warnings.push_back(result.message);
    return finalize();
  }
  if (options.processExecutable.empty()) {
    result.exitCode = BenchmarkExitCode::InvalidInput;
    result.message = "isolated repetitions require a process executable";
    report.warnings.push_back(result.message);
    return finalize();
  }
  if (options.repetitionTimeout <= std::chrono::milliseconds::zero()) {
    result.exitCode = BenchmarkExitCode::InvalidInput;
    result.message = "repetition timeout must be greater than zero";
    report.warnings.push_back(result.message);
    return finalize();
  }
  if (options.baselineProfileAuthoritative &&
      (options.baselineProfileMinimumRepetitions == 0u ||
       repetitions < options.baselineProfileMinimumRepetitions)) {
    result.exitCode = BenchmarkExitCode::InvalidInput;
    result.message = "authoritative profile requires at least " +
                     std::to_string(options.baselineProfileMinimumRepetitions) +
                     " isolated repetitions";
    report.profile.authorityBlockers.push_back(result.message);
    report.warnings.push_back(result.message);
    return finalize();
  }

  std::vector<std::pair<uint32_t, BenchmarkReport>> completedReports;
  completedReports.reserve(repetitions);
  nuri::tools::core::ToolOutcome aggregateOutcome =
      nuri::tools::core::ToolOutcome::Investigative;
  for (uint32_t index = 0u; index < repetitions; ++index) {
    BenchmarkRepetitionResult repetition{};
    repetition.index = index;
    const std::filesystem::path relativeWorkspace =
        std::filesystem::path("repetitions") / std::format("{:04}", index + 1u);
    auto confinedWorkspace =
        nuri::tools::core::resolvePathUnder(artifactDir, relativeWorkspace);
    if (confinedWorkspace.hasError()) {
      repetition.message = confinedWorkspace.error();
      result.repetitions.push_back(std::move(repetition));
      aggregateOutcome = nuri::tools::core::aggregateOutcome(
          aggregateOutcome, nuri::tools::core::ToolOutcome::RuntimeError);
      continue;
    }
    repetition.workspace = confinedWorkspace.value();
    std::error_code createError;
    std::filesystem::create_directories(repetition.workspace.parent_path(),
                                        createError);
    if (!createError &&
        !std::filesystem::create_directory(repetition.workspace, createError)) {
      if (!createError) {
        createError = std::make_error_code(std::errc::file_exists);
      }
    }
    if (createError) {
      repetition.message =
          "failed to create repetition workspace: " + createError.message();
      result.repetitions.push_back(std::move(repetition));
      aggregateOutcome = nuri::tools::core::aggregateOutcome(
          aggregateOutcome, nuri::tools::core::ToolOutcome::RuntimeError);
      continue;
    }

    repetition.reportPath = repetition.workspace / "report.json";
    repetition.envelopePath = repetition.workspace / "run.json";
    repetition.stdoutLogPath = repetition.workspace / "stdout.log";
    repetition.stderrLogPath = repetition.workspace / "stderr.log";
    nuri::tools::core::ProcessCommand command{
        .executable = options.processExecutable,
        .arguments = {"__run-child", "--case", benchmarkCase.id,
                      "--artifact-dir", pathToUtf8(repetition.workspace),
                      "--repetition-index", std::to_string(index)}};
    if (!options.baselineProfileId.empty()) {
      command.arguments.push_back("--profile");
      command.arguments.push_back(options.baselineProfileId);
    }
    if (options.samplesOverride.has_value()) {
      command.arguments.push_back("--samples");
      command.arguments.push_back(std::to_string(*options.samplesOverride));
    }
    if (options.dryRun) {
      command.arguments.push_back("--dry-run");
    }
    if (options.tracyDiagnostic) {
      command.arguments.push_back("--tracy-diagnostic");
    }
    if (options.verboseFrames) {
      command.arguments.push_back("--verbose-frames");
    }

    const nuri::tools::core::ProcessResult process =
        nuri::tools::core::runProcess(command,
                                      {.workingDirectory = benchmarkRepoRoot(),
                                       .timeout = options.repetitionTimeout});
    auto stdoutWrite = nuri::tools::core::atomicWriteTextFile(
        repetition.stdoutLogPath, process.standardOutput);
    auto stderrWrite = nuri::tools::core::atomicWriteTextFile(
        repetition.stderrLogPath, process.standardError);
    repetition.exitCode = benchmarkExitFromProcess(process);
    repetition.timedOut =
        process.status == nuri::tools::core::ProcessStatus::TimedOut;
    if (stdoutWrite.hasError() || stderrWrite.hasError()) {
      repetition.exitCode = BenchmarkExitCode::RuntimeError;
      repetition.message =
          stdoutWrite.hasError() ? stdoutWrite.error() : stderrWrite.error();
    } else if (process.status != nuri::tools::core::ProcessStatus::Exited ||
               repetition.exitCode != BenchmarkExitCode::Success) {
      repetition.message = processFailureMessage(process);
    } else {
      auto childReport = readBenchmarkReportFile(repetition.reportPath);
      if (childReport.hasError()) {
        repetition.exitCode = BenchmarkExitCode::RuntimeError;
        repetition.message = childReport.error();
      } else if (childReport.value().kind != "nuri.benchmark.report" ||
                 childReport.value().benchmarkCase.id != benchmarkCase.id) {
        repetition.exitCode = BenchmarkExitCode::RuntimeError;
        repetition.message = "child report identity does not match request";
      } else {
        repetition.completed = true;
        completedReports.emplace_back(index, std::move(childReport.value()));
      }
    }
    if (!repetition.completed) {
      report.warnings.push_back("repetition " + std::to_string(index + 1u) +
                                ": " + repetition.message);
    }
    aggregateOutcome = nuri::tools::core::aggregateOutcome(
        aggregateOutcome, benchmarkOutcome(repetition.exitCode, false));
    result.repetitions.push_back(std::move(repetition));
  }

  const std::string parentGeneratedAt = report.generatedAtUtc;
  std::vector<std::string> orchestrationWarnings = std::move(report.warnings);
  if (!completedReports.empty()) {
    const BenchmarkReport &first = completedReports.front().second;
    report = first;
    report.generatedAtUtc = parentGeneratedAt;
    report.command = options.command;
    report.benchmarkCase = benchmarkCase;
    report.artifacts.artifactDir = artifactDir;
    report.frames.clear();
    report.sampleStats.clear();
    report.stats.clear();
    report.warnings = std::move(orchestrationWarnings);
    report.unavailableMetrics.clear();
    report.unregisteredObservedMetrics.clear();
  } else {
    report.warnings = std::move(orchestrationWarnings);
  }

  std::map<std::string, std::vector<double>> repetitionValues;
  std::set<std::string> unavailableMetrics;
  std::set<std::string> unregisteredMetrics;
  bool environmentsMatch = true;
  bool childrenComparable = completedReports.size() == repetitions;
  bool profileCompatible =
      options.baselineProfileId.empty() || options.baselineProfile.has_value();
  std::set<std::string> profileCompatibilityReasons;
  bool anyWarmupUnstable = false;
  bool allWarmupStable = !completedReports.empty();
  report.timingDrain = {};
  for (const auto &[index, child] : completedReports) {
    BenchmarkSampleStats sample{};
    sample.sampleIndex = index;
    sample.measuredFrameCount = static_cast<uint32_t>(std::count_if(
        child.frames.begin(), child.frames.end(),
        [](const BenchmarkFrameRecord &frame) { return frame.measured; }));
    sample.stats = child.stats;
    sample.warmupStable = repetitionWarmupStatus(child);
    if (!sample.warmupStable.has_value()) {
      allWarmupStable = false;
    } else if (!*sample.warmupStable) {
      allWarmupStable = false;
      anyWarmupUnstable = true;
    }
    for (const auto &[metricId, stats] : child.stats) {
      repetitionValues[metricId].push_back(stats.median);
    }
    unavailableMetrics.insert(child.unavailableMetrics.begin(),
                              child.unavailableMetrics.end());
    unregisteredMetrics.insert(child.unregisteredObservedMetrics.begin(),
                               child.unregisteredObservedMetrics.end());
    if (&child != &completedReports.front().second &&
        !sameIsolatedEnvironment(completedReports.front().second.environment,
                                 child.environment)) {
      environmentsMatch = false;
    }
    childrenComparable = childrenComparable && child.run.validForComparison;
    if (options.baselineProfile.has_value()) {
      const auto compatibility = nuri::tools::core::evaluateBaselineProfile(
          *options.baselineProfile,
          observedProfileEnvironment(child.environment));
      profileCompatible = profileCompatible && compatibility.compatible;
      if (!compatibility.compatible) {
        profileCompatibilityReasons.insert(compatibility.reasons.begin(),
                                           compatibility.reasons.end());
      }
    } else if (!options.baselineProfileId.empty()) {
      profileCompatibilityReasons.insert(
          "baseline profile compatibility was not evaluated");
    }
    report.timingDrain.drainComplete =
        report.timingDrain.drainComplete && child.timingDrain.drainComplete;
    report.timingDrain.drainFrames += child.timingDrain.drainFrames;
    report.timingDrain.drainTimeoutMs = std::max(
        report.timingDrain.drainTimeoutMs, child.timingDrain.drainTimeoutMs);
    report.timingDrain.missingGpuTimingFrames +=
        child.timingDrain.missingGpuTimingFrames;
    report.timingDrain.droppedGpuTimingReports +=
        child.timingDrain.droppedGpuTimingReports;
    report.sampleStats.push_back(std::move(sample));
  }
  for (auto &[metricId, values] : repetitionValues) {
    auto stats = computeMetricStats(values);
    if (!stats.hasError()) {
      report.stats.emplace(metricId, std::move(stats.value()));
    }
  }
  report.unregisteredObservedMetrics.assign(unregisteredMetrics.begin(),
                                            unregisteredMetrics.end());
  report.repeatObservations = {
      .unit = "isolated-process",
      .independent = true,
      .count = static_cast<uint32_t>(completedReports.size())};
  report.profile.id = options.baselineProfileId;
  report.profile.profileAuthoritative = options.baselineProfileAuthoritative;
  report.profile.authoritative = false;
  report.profile.minimumRepetitions = options.baselineProfileMinimumRepetitions;
  report.profile.completedRepetitions =
      static_cast<uint32_t>(completedReports.size());
  report.profile.repetitionRequirementSatisfied =
      completedReports.size() >= options.baselineProfileMinimumRepetitions;
  report.profile.repetitionUnit = "isolated-process";
  report.profile.warmupStabilityPolicy = options.baselineProfileWarmupStability;
  report.profile.warmupWindowFrames = options.baselineProfileWarmupWindowFrames;
  report.profile.warmupMaxDriftPercent =
      options.baselineProfileWarmupMaxDriftPercent;
  report.profile.warmupStabilityStatus =
      allWarmupStable ? "stable" : (anyWarmupUnstable ? "unstable" : "unknown");
  report.profile.requiredMetrics = options.baselineProfileRequiredMetrics;
  report.profile.authorityBlockers.clear();

  std::set<std::string> requiredMetrics(benchmarkCase.requiredMetrics.begin(),
                                        benchmarkCase.requiredMetrics.end());
  requiredMetrics.insert(options.baselineProfileRequiredMetrics.begin(),
                         options.baselineProfileRequiredMetrics.end());
  bool requiredMetricsComplete = true;
  for (const std::string &metricId : requiredMetrics) {
    const auto values = repetitionValues.find(metricId);
    if (values == repetitionValues.end() ||
        values->second.size() != completedReports.size() ||
        completedReports.size() != repetitions) {
      requiredMetricsComplete = false;
      unavailableMetrics.insert(metricId);
    }
  }
  report.unavailableMetrics.assign(unavailableMetrics.begin(),
                                   unavailableMetrics.end());
  if (!options.baselineProfileAuthoritative &&
      !options.baselineProfileId.empty()) {
    report.profile.authorityBlockers.push_back(
        "baseline profile is investigative");
  }
  if (completedReports.size() != repetitions) {
    report.profile.authorityBlockers.push_back(
        "not every requested isolated repetition completed");
  }
  if (!report.profile.repetitionRequirementSatisfied) {
    report.profile.authorityBlockers.push_back(
        "minimum isolated repetition requirement was not satisfied");
  }
  if (report.profile.warmupStabilityStatus != "stable") {
    report.profile.authorityBlockers.push_back(
        "warmup stability was not measured stable for every repetition");
  }
  if (!requiredMetricsComplete) {
    report.profile.authorityBlockers.push_back(
        "required metrics were not complete in every repetition");
  }
  if (!environmentsMatch) {
    report.profile.authorityBlockers.push_back(
        "isolated repetition environments did not match");
  }
  if (!profileCompatible) {
    report.profile.authorityBlockers.insert(
        report.profile.authorityBlockers.end(),
        profileCompatibilityReasons.begin(), profileCompatibilityReasons.end());
  }
  if (!childrenComparable) {
    report.profile.authorityBlockers.push_back(
        "one or more repetitions were not valid for comparison");
  }
  if (options.tracyDiagnostic) {
    report.profile.authorityBlockers.push_back(
        "Tracy diagnostic execution is not authoritative");
  }
  if (options.dryRun) {
    report.profile.authorityBlockers.push_back(
        "dry-run execution is not authoritative");
  }
  report.run.validForComparison =
      completedReports.size() == repetitions && requiredMetricsComplete &&
      environmentsMatch && profileCompatible && childrenComparable &&
      !options.tracyDiagnostic && !options.dryRun;
  report.profile.authoritative = options.baselineProfileAuthoritative &&
                                 report.profile.authorityBlockers.empty() &&
                                 report.run.validForComparison;
  report.benchmarkCase.authoritative = report.profile.authoritative;

  result.exitCode = benchmarkExitFromOutcome(aggregateOutcome);
  if (result.exitCode == BenchmarkExitCode::Success &&
      !requiredMetricsComplete) {
    result.exitCode = BenchmarkExitCode::RuntimeError;
  }
  if (result.exitCode == BenchmarkExitCode::Success &&
      options.baselineProfileAuthoritative && !report.profile.authoritative) {
    result.exitCode = BenchmarkExitCode::EnvironmentUnavailable;
  }
  result.message =
      result.exitCode == BenchmarkExitCode::Success
          ? "isolated benchmark repetitions complete"
          : "isolated benchmark repetitions completed without authority";
  return finalize();
}

BenchmarkRunResult
runBenchmarkCaseRenderDoc(BenchmarkCase benchmarkCase,
                          const BenchmarkRunOptions &options) {
  BenchmarkRunResult result{};
  const BenchmarkGpuDiagnosticOptions &diagnostic = options.gpuDiagnostic;
  std::string runId = nuri::tools::core::createRunId();
  std::filesystem::path artifactDir = options.artifactDir;
  if (artifactDir.empty()) {
    auto workspace = nuri::tools::core::createRunWorkspace(
        benchmarkRepoRoot() / "artifacts" / "bench");
    if (workspace.hasError()) {
      result.exitCode = BenchmarkExitCode::RuntimeError;
      result.message = workspace.error();
      return result;
    }
    runId = workspace.value().runId;
    artifactDir = workspace.value().root;
  }
  result.reportPath = options.jsonOut.empty()
                          ? artifactDir / "cases" / (benchmarkCase.id + ".json")
                          : options.jsonOut;
  result.envelopePath = options.envelopeOut.empty() ? artifactDir / "run.json"
                                                    : options.envelopeOut;
  if (result.envelopePath.lexically_normal() ==
      result.reportPath.lexically_normal()) {
    result.envelopePath = artifactDir / "result.json";
  }

  BenchmarkReport report{};
  report.generatedAtUtc = utcTimestampIso8601();
  report.command = options.command;
  report.benchmarkCase = benchmarkCase;
  report.run.validForComparison = false;
  report.profile.id = options.baselineProfileId;
  report.profile.profileAuthoritative = options.baselineProfileAuthoritative;
  report.profile.authoritative = false;
  report.profile.authorityBlockers.push_back(
      "RenderDoc frame-forensics execution cannot be compared or accepted as "
      "a benchmark baseline");
  report.artifacts.artifactDir = artifactDir;
  report.renderDoc.requested = true;
  report.renderDoc.captureFrame = diagnostic.captureFrame;

  const auto finalize = [&]() -> BenchmarkRunResult {
    return finalizeBenchmarkCaseResult(std::move(result), std::move(report),
                                       artifactDir, runId,
                                       options.verboseFrames);
  };

  const std::optional<std::filesystem::path> tool =
      findRenderDocTool(diagnostic.toolPath);
  if (!tool.has_value()) {
    result.exitCode = BenchmarkExitCode::EnvironmentUnavailable;
    result.message =
        "RenderDoc diagnostic requested but renderdoccmd was not found; pass "
        "--renderdoc-tool or add it to PATH";
    report.warnings.push_back(result.message);
    return finalize();
  }
  report.renderDoc.toolPath = *tool;
  if (options.processExecutable.empty()) {
    result.exitCode = BenchmarkExitCode::InvalidInput;
    result.message = "RenderDoc diagnostic requires a process executable";
    report.warnings.push_back(result.message);
    return finalize();
  }
  if (diagnostic.timeout <= std::chrono::milliseconds::zero()) {
    result.exitCode = BenchmarkExitCode::InvalidInput;
    result.message = "RenderDoc diagnostic timeout must be greater than zero";
    report.warnings.push_back(result.message);
    return finalize();
  }

  const std::filesystem::path renderDocDir = artifactDir / "renderdoc";
  std::filesystem::create_directories(renderDocDir);
  const std::string baseName = benchmarkCase.id + "_" + utcTimestampForPath() +
                               "_frame" +
                               std::to_string(diagnostic.captureFrame);
  const std::filesystem::path captureTemplate = renderDocDir / baseName;
  const std::filesystem::path launchLog = renderDocDir / (baseName + ".log");
  const std::filesystem::path workerDir = renderDocDir / (baseName + "_worker");
  const std::filesystem::path workerReport = workerDir / "report.json";
  const std::filesystem::path workerEnvelope = workerDir / "run.json";
  nuri::tools::core::ProcessCommand command{
      .executable = *tool,
      .arguments = {
          "capture", "-w", "-d", pathToUtf8(benchmarkRepoRoot()), "-c",
          pathToUtf8(std::filesystem::absolute(captureTemplate)),
          pathToUtf8(options.processExecutable), "__run-child", "--case",
          benchmarkCase.id, "--repetition-index", "0", "--artifact-dir",
          pathToUtf8(workerDir), "--renderdoc-diagnostic", "--renderdoc-tool",
          pathToUtf8(*tool), "--renderdoc-capture-frame",
          std::to_string(diagnostic.captureFrame)}};
  if (!options.baselineProfileId.empty()) {
    command.arguments.push_back("--profile");
    command.arguments.push_back(options.baselineProfileId);
  }
  if (options.samplesOverride.has_value()) {
    command.arguments.push_back("--samples");
    command.arguments.push_back(std::to_string(*options.samplesOverride));
  }
  if (options.verboseFrames) {
    command.arguments.push_back("--verbose-frames");
  }
  std::string captureCommand = quoteCommandArg(*tool);
  for (const std::string &argument : command.arguments) {
    captureCommand += " " + quoteCommandArg(argument);
  }
  report.renderDoc.captureCommand = captureCommand;
  const ScopedEnvVar fifoPresentMode("NURI_PRESENT_MODE", "fifo");
  const nuri::tools::core::ProcessResult process =
      nuri::tools::core::runProcess(command,
                                    {.workingDirectory = benchmarkRepoRoot(),
                                     .timeout = diagnostic.timeout});
  {
    std::ofstream log(launchLog, std::ios::binary);
    writeProcessLog(log, report.renderDoc.captureCommand, process);
  }
  report.renderDoc.captureLogPath = launchLog;
  report.renderDoc.launcherExitCode = processExitCode(process);
  addArtifactOnce(report.artifacts.renderDocArtifacts, launchLog);

  std::ifstream envelopeFile(workerEnvelope, std::ios::binary);
  std::ostringstream envelopeText;
  envelopeText << envelopeFile.rdbuf();
  envelopeFile.close();
  auto childEnvelope =
      nuri::tools::core::readResultEnvelopeV2(envelopeText.str());
  auto childReport = readBenchmarkReportFile(workerReport);
  std::error_code workerCleanupError;
  std::filesystem::remove_all(workerDir, workerCleanupError);
  const bool childInvalid = childReport.hasError() || childEnvelope.hasError();
  if (!childInvalid) {
    report = std::move(childReport.value());
  }
  if (workerCleanupError) {
    report.warnings.push_back("failed to remove RenderDoc worker workspace: " +
                              workerCleanupError.message());
  }
  if (childInvalid) {
    result.exitCode = BenchmarkExitCode::RuntimeError;
    result.message =
        childReport.hasError() ? childReport.error() : childEnvelope.error();
    report.warnings.push_back(result.message);
    return finalize();
  }
  report.command = options.command;
  report.artifacts.artifactDir = artifactDir;
  report.renderDoc.toolPath = *tool;
  report.renderDoc.captureLogPath = launchLog;
  report.renderDoc.captureCommand = std::move(captureCommand);
  report.renderDoc.launcherExitCode = processExitCode(process);
  addArtifactOnce(report.artifacts.renderDocArtifacts, launchLog);

  if (childEnvelope.value().exitCode != 0) {
    result.exitCode = static_cast<BenchmarkExitCode>(
        std::clamp(childEnvelope.value().exitCode, 1, 5));
    result.message = "RenderDoc diagnostic child did not complete";
    report.warnings.push_back(result.message);
    return finalize();
  }
  if (!report.renderDoc.available || report.renderDoc.capturePath.empty()) {
    result.exitCode = BenchmarkExitCode::RuntimeError;
    result.message = "RenderDoc diagnostic capture did not complete";
    report.warnings.push_back(result.message);
    return finalize();
  }

  report.renderDoc.chromeTracePath = report.renderDoc.capturePath;
  report.renderDoc.chromeTracePath.replace_extension(".chrome.json");
  report.renderDoc.conversionLogPath =
      renderDocDir / (baseName + "_convert.log");
  nuri::tools::core::ProcessCommand convertCommand{
      .executable = *tool,
      .arguments = {"convert", "-f", pathToUtf8(report.renderDoc.capturePath),
                    "-o", pathToUtf8(report.renderDoc.chromeTracePath), "-c",
                    "chrome.json"}};
  const std::string convertDisplay =
      quoteCommandArg(*tool) + " convert -f " +
      quoteCommandArg(report.renderDoc.capturePath) + " -o " +
      quoteCommandArg(report.renderDoc.chromeTracePath) + " -c chrome.json";
  report.renderDoc.conversionExitCode = runCommandToLog(
      convertCommand, convertDisplay, report.renderDoc.conversionLogPath,
      {.workingDirectory = benchmarkRepoRoot(), .timeout = diagnostic.timeout});
  addArtifactOnce(report.artifacts.renderDocArtifacts,
                  report.renderDoc.conversionLogPath);
  std::error_code sizeError;
  report.renderDoc.chromeTraceSizeBytes =
      std::filesystem::file_size(report.renderDoc.chromeTracePath, sizeError);
  if (report.renderDoc.conversionExitCode == 0 && !sizeError &&
      report.renderDoc.chromeTraceSizeBytes > 0u) {
    addArtifactOnce(report.artifacts.renderDocArtifacts,
                    report.renderDoc.chromeTracePath);
    if (!summarizeRenderDocChromeTrace(report.renderDoc.chromeTracePath,
                                       report.renderDoc)) {
      report.warnings.push_back(
          "RenderDoc Chrome trace was created but could not be summarized");
    }
  } else {
    report.renderDoc.chromeTracePath.clear();
    report.renderDoc.chromeTraceSizeBytes = 0u;
    report.warnings.push_back(
        "RenderDoc capture succeeded but Chrome trace conversion failed");
  }

  report.renderDoc.thumbnailPath = report.renderDoc.capturePath;
  report.renderDoc.thumbnailPath.replace_extension(".png");
  const std::filesystem::path thumbnailLog =
      renderDocDir / (baseName + "_thumbnail.log");
  nuri::tools::core::ProcessCommand thumbnailCommand{
      .executable = *tool,
      .arguments = {"thumb", "--out",
                    pathToUtf8(report.renderDoc.thumbnailPath),
                    pathToUtf8(report.renderDoc.capturePath)}};
  const int thumbnailExit = runCommandToLog(
      thumbnailCommand,
      quoteCommandArg(*tool) + " thumb --out " +
          quoteCommandArg(report.renderDoc.thumbnailPath) + " " +
          quoteCommandArg(report.renderDoc.capturePath),
      thumbnailLog,
      {.workingDirectory = benchmarkRepoRoot(), .timeout = diagnostic.timeout});
  addArtifactOnce(report.artifacts.renderDocArtifacts, thumbnailLog);
  if (thumbnailExit == 0 &&
      std::filesystem::is_regular_file(report.renderDoc.thumbnailPath)) {
    addArtifactOnce(report.artifacts.renderDocArtifacts,
                    report.renderDoc.thumbnailPath);
  } else {
    report.renderDoc.thumbnailPath.clear();
    report.warnings.push_back(
        "RenderDoc capture succeeded but thumbnail extraction failed");
  }

  result.exitCode = BenchmarkExitCode::Success;
  result.message = "RenderDoc frame-forensics diagnostic capture complete";
  return finalize();
}

BenchmarkRunResult runBenchmarkCase(BenchmarkCase benchmarkCase,
                                    const BenchmarkRunOptions &options) {
  BenchmarkRunResult result{};
  const BenchmarkGpuDiagnosticOptions &diagnostic = options.gpuDiagnostic;
  const bool rgpDiagnostic =
      diagnostic.kind == BenchmarkGpuDiagnosticKind::RgpShader;
  const bool renderDocDiagnostic =
      diagnostic.kind == BenchmarkGpuDiagnosticKind::RenderDocFrame;
  if (!options.baselineProfileId.empty()) {
    benchmarkCase.authoritative = false;
    benchmarkCase.configSignature.clear();
  }
  if (renderDocDiagnostic) {
    benchmarkCase.presentMode = "fifo";
  }
  const uint32_t requestedSamples =
      options.samplesOverride.value_or(benchmarkCase.samples);
  uint32_t samples = requestedSamples;
  if (rgpDiagnostic) {
    constexpr uint32_t kRgpMinimumFrameBudget = 1'000u;
    const uint32_t framesPerSample =
        std::max(benchmarkCase.warmupFrames + benchmarkCase.measurementFrames +
                     benchmarkCase.cooldownFrames,
                 1u);
    const uint32_t minimumSamples =
        (kRgpMinimumFrameBudget + framesPerSample - 1u) / framesPerSample;
    samples = std::max(samples, minimumSamples);
  }
  if (renderDocDiagnostic) {
    const uint32_t framesPerSample =
        std::max(benchmarkCase.warmupFrames + benchmarkCase.measurementFrames +
                     benchmarkCase.cooldownFrames,
                 1u);
    const uint64_t requiredFrames =
        static_cast<uint64_t>(diagnostic.captureFrame) + 1u;
    const uint32_t minimumSamples = static_cast<uint32_t>(
        (requiredFrames + framesPerSample - 1u) / framesPerSample);
    samples = std::max(samples, minimumSamples);
  }
  benchmarkCase.samples = samples;
  std::string backendSource;
  const std::string backend = resolveBackendName(benchmarkCase, backendSource);
  std::string presentSource;
  const std::string presentMode =
      resolvePresentMode(benchmarkCase, presentSource);
  std::string runId = nuri::tools::core::createRunId();
  std::filesystem::path artifactDir = options.artifactDir;
  if (artifactDir.empty()) {
    auto workspace = nuri::tools::core::createRunWorkspace(
        benchmarkRepoRoot() / "artifacts" / "bench");
    if (workspace.hasError()) {
      result.exitCode = BenchmarkExitCode::RuntimeError;
      result.message = workspace.error();
      result.report.benchmarkCase = std::move(benchmarkCase);
      result.report.run.validForComparison = false;
      return result;
    }
    runId = workspace.value().runId;
    artifactDir = workspace.value().root;
  }
  const std::filesystem::path reportPath =
      options.jsonOut.empty()
          ? artifactDir / "cases" / (benchmarkCase.id + ".json")
          : options.jsonOut;
  std::filesystem::path envelopePath = options.envelopeOut.empty()
                                           ? artifactDir / "run.json"
                                           : options.envelopeOut;
  if (envelopePath.lexically_normal() == reportPath.lexically_normal()) {
    envelopePath = artifactDir / "result.json";
  }
  result.reportPath = reportPath;
  result.envelopePath = envelopePath;

  BenchmarkReport report{};
  report.generatedAtUtc = utcTimestampIso8601();
  report.command = options.command;
  report.benchmarkCase = benchmarkCase;
  if (!options.baselineProfileId.empty()) {
    report.profile.id = options.baselineProfileId;
    report.profile.profileAuthoritative = options.baselineProfileAuthoritative;
    report.profile.authoritative = false;
    report.profile.minimumRepetitions =
        options.baselineProfileMinimumRepetitions;
    report.profile.completedRepetitions = 0u;
    report.profile.repetitionRequirementSatisfied =
        options.baselineProfileMinimumRepetitions == 0u;
    report.profile.repetitionUnit = "not-collected";
    report.profile.warmupStabilityPolicy =
        options.baselineProfileWarmupStability;
    report.profile.warmupStabilityStatus = "unknown";
    report.profile.warmupWindowFrames =
        options.baselineProfileWarmupWindowFrames;
    report.profile.warmupMaxDriftPercent =
        options.baselineProfileWarmupMaxDriftPercent;
    report.profile.requiredMetrics = options.baselineProfileRequiredMetrics;
    if (!options.baselineProfileAuthoritative) {
      report.profile.authorityBlockers.push_back(
          "baseline profile is investigative");
    }
    if (options.baselineProfileMinimumRepetitions > 0u) {
      report.profile.authorityBlockers.push_back(
          "independent repetitions are not collected; in-process sample "
          "windows are not profile repetitions");
    }
  }
  report.run.samples = samples;
  report.run.warmupFrames = benchmarkCase.warmupFrames;
  report.run.measurementFrames = benchmarkCase.measurementFrames;
  report.run.cooldownFrames = benchmarkCase.cooldownFrames;
  report.run.maxDrainFrames = benchmarkCase.maxDrainFrames;
  report.run.drainTimeoutMs = benchmarkCase.drainTimeoutMs;
  report.run.fixedDeltaSeconds = benchmarkCase.fixedDeltaSeconds;
  report.artifacts.artifactDir = artifactDir;
  report.rgp.requested = rgpDiagnostic;
  report.rgp.captureFrame = diagnostic.captureFrame;
  report.rgp.counterCollectionRequested = rgpDiagnostic;
  report.renderDoc.requested = renderDocDiagnostic;
  report.renderDoc.toolPath =
      renderDocDiagnostic ? diagnostic.toolPath : std::filesystem::path{};
  report.renderDoc.captureFrame = diagnostic.captureFrame;
  report.timingDrain.drainTimeoutMs = benchmarkCase.drainTimeoutMs;
  report.environment =
      collectBenchmarkEnvironment(backend, backendSource, presentMode,
                                  presentSource, options.tracyDiagnostic);
  report.environment.renderGraphWorkerCount =
      benchmarkCase.renderGraph.workerCount;
  report.environment.renderGraphParallelCompile =
      benchmarkCase.renderGraph.parallelCompile;
  report.environment.renderGraphParallelRecording =
      benchmarkCase.renderGraph.parallelRecording;
  report.warnings.push_back(
      "first-slice benchmark uses the swapchain-present renderer path");
  if (!options.baselineProfileId.empty()) {
    report.warnings.push_back(options.baselineProfileWarning.empty()
                                  ? "baseline profile '" +
                                        options.baselineProfileId +
                                        "' is investigative; run is not "
                                        "authoritative"
                                  : options.baselineProfileWarning);
  }
  if (options.tracyDiagnostic) {
    report.run.validForComparison = false;
    report.warnings.push_back(
        "Tracy diagnostic mode is not valid for authoritative comparison");
  }
  if (options.opaquePipelineStatisticsDiagnostic) {
    report.run.validForComparison = false;
    report.profile.authoritative = false;
    report.profile.authorityBlockers.push_back(
        "opaque pipeline-statistics diagnostics cannot be compared or "
        "accepted as a benchmark baseline");
    report.warnings.push_back(
        "Opaque pipeline statistics are diagnostic-only; ignore timing data "
        "from this run");
  }
  if (rgpDiagnostic) {
    report.run.validForComparison = false;
    report.profile.authoritative = false;
    report.profile.authorityBlockers.push_back(
        "RGP shader diagnostic execution cannot be compared or accepted as a "
        "benchmark baseline");
    report.warnings.push_back(
        "RGP is restricted to shader diagnosis; ignore benchmark timing data "
        "from this capture run");
    if (samples != requestedSamples) {
      report.warnings.push_back(
          "RGP shader diagnostic increased sample windows from " +
          std::to_string(requestedSamples) + " to " + std::to_string(samples) +
          " to keep the capture target alive");
    }
  }
  if (renderDocDiagnostic) {
    report.run.validForComparison = false;
    report.profile.authoritative = false;
    report.profile.authorityBlockers.push_back(
        "RenderDoc frame-forensics execution cannot be compared or accepted "
        "as a benchmark baseline");
    report.warnings.push_back(
        "RenderDoc is restricted to frame forensics; capture/replay timings "
        "and counters are not GPU performance evidence");
    report.warnings.push_back(
        "RenderDoc diagnostic forced FIFO present mode for capture stability");
    if (samples != requestedSamples) {
      report.warnings.push_back(
          "RenderDoc diagnostic increased sample windows from " +
          std::to_string(requestedSamples) + " to " + std::to_string(samples) +
          " to reach capture frame " + std::to_string(diagnostic.captureFrame));
    }
  }

  const auto finalizeResult = [&]() -> BenchmarkRunResult {
    if (diagnostic.kind != BenchmarkGpuDiagnosticKind::None) {
      report.frames.clear();
      report.sampleStats.clear();
      report.stats.clear();
      report.unavailableMetrics.clear();
      report.unregisteredObservedMetrics.clear();
      report.repeatObservations = {};
      report.warnings.push_back(
          "Benchmark timing samples are intentionally omitted from GPU "
          "diagnostic reports");
    }
    return finalizeBenchmarkCaseResult(std::move(result), std::move(report),
                                       artifactDir, runId,
                                       options.verboseFrames);
  };

  const auto validateRequiredMetrics =
      [&result, &report](const std::vector<std::string> &metricIds,
                         std::string_view field) {
        for (const std::string &metricId : metricIds) {
          auto descriptor = requireBenchmarkMetricDescriptor(metricId, field);
          if (descriptor.hasError()) {
            result.exitCode = BenchmarkExitCode::InvalidInput;
            result.message = descriptor.error();
            report.run.validForComparison = false;
            report.warnings.push_back(descriptor.error());
            return false;
          }
        }
        return true;
      };
  if (options.baselineProfileAuthoritative && !options.internalIsolatedChild) {
    result.exitCode = BenchmarkExitCode::InvalidInput;
    result.message =
        "authoritative benchmark profiles require isolated repetitions";
    report.run.validForComparison = false;
    report.warnings.push_back(result.message);
    computeBenchmarkReportStats(report);
    return finalizeResult();
  }
  if (options.tracyDiagnostic &&
      diagnostic.kind != BenchmarkGpuDiagnosticKind::None) {
    result.exitCode = BenchmarkExitCode::InvalidInput;
    result.message = "Tracy, RGP, and RenderDoc diagnostics must be collected "
                     "in separate runs";
    report.run.validForComparison = false;
    report.warnings.push_back(result.message);
    computeBenchmarkReportStats(report);
    return finalizeResult();
  }
  if (!validateRequiredMetrics(benchmarkCase.requiredMetrics,
                               "case requiredMetrics") ||
      !validateRequiredMetrics(options.baselineProfileRequiredMetrics,
                               "profile requiredMetrics")) {
    computeBenchmarkReportStats(report);
    return finalizeResult();
  }
  if (samples == 0u) {
    result.exitCode = BenchmarkExitCode::InvalidInput;
    result.message = "samples must be greater than zero";
    report.run.validForComparison = false;
    report.warnings.push_back(result.message);
    computeBenchmarkReportStats(report);
    return finalizeResult();
  }

  std::string requirementMessage;
  auto requirements = checkRequirements(benchmarkCase, backend, report.warnings,
                                        requirementMessage);
  if (requirements.hasError()) {
    report.run.validForComparison = false;
    report.warnings.push_back(requirementMessage);
    result.exitCode = requirements.error();
    result.message = requirementMessage;
    computeBenchmarkReportStats(report);
    return finalizeResult();
  }

  if (options.dryRun) {
    result.exitCode = BenchmarkExitCode::Success;
    result.message = "dry run succeeded";
    report.warnings.push_back("dry run: renderer was not initialized");
    report.run.validForComparison = false;
    computeBenchmarkReportStats(report);
    return finalizeResult();
  }

  TracyCaptureSession tracySession{};
  RgpCaptureSession rgpSession{};
  RenderDocCaptureSession renderDocSession{};
  const auto finishDiagnosticsAndFinalize = [&]() -> BenchmarkRunResult {
    renderDocSession.finish(report);
    rgpSession.finish(report);
    tracySession.finish(report);
    computeBenchmarkReportStats(report);
    return finalizeResult();
  };
  try {
    BenchmarkLogGuard logGuard(artifactDir / "logs" /
                               (benchmarkCase.id + ".log"));
    std::vector<std::unique_ptr<ScopedEnvVar>> env;
    env.push_back(std::make_unique<ScopedEnvVar>(
        "NURI_RENDER_GRAPH_WORKER_COUNT",
        std::to_string(benchmarkCase.renderGraph.workerCount)));
    env.push_back(std::make_unique<ScopedEnvVar>(
        "NURI_RENDER_GRAPH_DISABLE_PARALLEL_COMPILE",
        benchmarkCase.renderGraph.parallelCompile ? "0" : "1"));
    env.push_back(std::make_unique<ScopedEnvVar>(
        "NURI_RENDER_GRAPH_DISABLE_PARALLEL_RECORDING",
        benchmarkCase.renderGraph.parallelRecording ? "0" : "1"));
    if (presentSource == "manifest" && presentMode != "default") {
      env.push_back(
          std::make_unique<ScopedEnvVar>("NURI_PRESENT_MODE", presentMode));
    }
    if (options.opaquePipelineStatisticsDiagnostic) {
      env.push_back(std::make_unique<ScopedEnvVar>(
          "NURI_GPU_OPAQUE_PIPELINE_STATISTICS", "1"));
    }

    auto configResult = loadRuntimeConfigFromEnvOrDefault();
    if (configResult.hasError()) {
      result.exitCode = BenchmarkExitCode::EnvironmentUnavailable;
      result.message = configResult.error();
      report.run.validForComparison = false;
      report.warnings.push_back(configResult.error());
      return finishDiagnosticsAndFinalize();
    }
    RuntimeConfig config = std::move(configResult.value());
    config.window.title = "nuri-bench " + benchmarkCase.id;
    config.window.width = static_cast<int32_t>(benchmarkCase.resolution[0]);
    config.window.height = static_cast<int32_t>(benchmarkCase.resolution[1]);
    config.window.mode = WindowMode::Windowed;

    rgpSession = startRgpCaptureIfRequested(benchmarkCase, options, report);
    if (rgpDiagnostic && !rgpSession.started) {
      result.exitCode = report.rgp.toolPath.empty()
                            ? BenchmarkExitCode::EnvironmentUnavailable
                            : BenchmarkExitCode::InvalidInput;
      result.message = report.warnings.back();
      report.run.validForComparison = false;
      return finishDiagnosticsAndFinalize();
    }
    if (rgpSession.started) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    std::unique_ptr<Window> window =
        Window::create(config.window.title, config.window.width,
                       config.window.height, config.window.mode);
    if (!window) {
      result.exitCode = BenchmarkExitCode::EnvironmentUnavailable;
      result.message = "failed to create benchmark window";
      report.run.validForComparison = false;
      report.warnings.push_back(result.message);
      return finishDiagnosticsAndFinalize();
    }
    std::unique_ptr<GPUDevice> gpu = GPUDevice::create(*window);
    if (!gpu) {
      result.exitCode = BenchmarkExitCode::EnvironmentUnavailable;
      result.message = "failed to create GPU device";
      report.run.validForComparison = false;
      report.warnings.push_back(result.message);
      return finishDiagnosticsAndFinalize();
    }
    report.environment.swapchainImageCount = gpu->getSwapchainImageCount();
    const GPUAdapterInfo adapter = gpu->getAdapterInfo();
    report.environment.gpuDeviceName = adapter.name;
    report.environment.gpuVendorId = adapter.vendorId;
    report.environment.gpuDeviceId = adapter.deviceId;
    report.environment.gpuDriverVersion = adapter.driverVersion;
    const RayTracingCapabilities &rayTracingCaps =
        gpu->getDeviceCaps().rayTracing;
    if ((benchmarkCase.requirements.accelerationStructure &&
         !rayTracingCaps.accelerationStructure) ||
        (benchmarkCase.requirements.rayQuery && !rayTracingCaps.rayQuery)) {
      result.exitCode = BenchmarkExitCode::EnvironmentUnavailable;
      result.message = "required ray-tracing capability is unavailable";
      report.run.validForComparison = false;
      report.warnings.push_back(result.message);
      return finishDiagnosticsAndFinalize();
    }
    std::string gpuRequirementMessage;
    auto gpuRequirements = checkBenchmarkGpuRequirements(
        benchmarkCase.requirements, gpu->getMultisampleCapabilities(),
        gpuRequirementMessage);
    if (gpuRequirements.hasError()) {
      result.exitCode = gpuRequirements.error();
      result.message = gpuRequirementMessage;
      report.run.validForComparison = false;
      report.warnings.push_back(result.message);
      return finishDiagnosticsAndFinalize();
    }

    TrackingMemoryResource rendererMemoryTracker;
    TrackingMemoryResource pipelineMemoryTracker;
    TrackingMemoryResource sceneMemoryTracker;
    std::pmr::unsynchronized_pool_resource rendererMemory(
        &rendererMemoryTracker);
    std::pmr::unsynchronized_pool_resource pipelineMemory(
        &pipelineMemoryTracker);
    std::pmr::unsynchronized_pool_resource sceneMemory(&sceneMemoryTracker);
    std::unique_ptr<Renderer> renderer = Renderer::create(*gpu, rendererMemory);
    BenchmarkAnimationFixture animationFixture{};
    RenderPipeline pipeline(&pipelineMemory);
    pipeline.addProvider(
        std::make_unique<BenchmarkAnimationFrameProvider>(animationFixture));
    auto pipelineResult = registerDefaultRenderPipeline(
        pipeline, *gpu, config.shaders, &pipelineMemory);
    if (pipelineResult.hasError()) {
      result.exitCode = BenchmarkExitCode::RuntimeError;
      result.message = pipelineResult.error();
      report.run.validForComparison = false;
      report.warnings.push_back(result.message);
      return finishDiagnosticsAndFinalize();
    }

    RenderScene scene(&sceneMemory);
    SceneLoadHandle sceneLoad{};
    const auto sceneResourcePrepareBegin = std::chrono::steady_clock::now();
    auto sceneResult =
        populateScene(benchmarkCase, *renderer, scene, &sceneMemory, sceneLoad);
    if (sceneResult.hasError()) {
      result.exitCode = BenchmarkExitCode::EnvironmentUnavailable;
      result.message = sceneResult.error();
      report.run.validForComparison = false;
      report.warnings.push_back(result.message);
      return finishDiagnosticsAndFinalize();
    }
    auto assetsReady = waitForBenchmarkAssets(*renderer, scene, sceneLoad);
    const double sceneResourcePrepareMs = elapsedMs(sceneResourcePrepareBegin);
    if (assetsReady.hasError()) {
      result.exitCode = BenchmarkExitCode::RuntimeError;
      result.message = assetsReady.error();
      report.run.validForComparison = false;
      report.warnings.push_back(result.message);
      return finishDiagnosticsAndFinalize();
    }

    renderDocSession = startRenderDocCaptureIfRequested(options, report);
    if (renderDocDiagnostic && !renderDocSession.ready()) {
      result.exitCode = BenchmarkExitCode::EnvironmentUnavailable;
      result.message = report.warnings.back();
      report.run.validForComparison = false;
      return finishDiagnosticsAndFinalize();
    }

    RenderSettings settings = benchmarkCase.settings;
    Camera camera = makeBenchmarkCamera(benchmarkCase);
    TemporalFrameService temporalFrameService{};
    RenderFrameContext frameContext{};
    uint64_t frameIndex = 0u;
    double timeSeconds = 0.0;
    std::map<uint64_t, size_t> measuredFrameByIndex;

    const auto renderOneFrame =
        [&](uint32_t sampleIndex, uint32_t sampleFrame,
            bool measured) -> Result<bool, std::string> {
      NURI_PROFILER_FRAME(nullptr);
      NURI_PROFILER_FRAME("BenchmarkFrame");
      NURI_PROFILER_ZONE_STATIC("BenchmarkFrame", NURI_PROFILER_COLOR_SUBMIT);
      window->pollEvents();
      if (renderDocDiagnostic) {
        renderDocSession.trigger(frameIndex, report);
      }
      auto evaluatedCamera =
          evaluateBenchmarkCameraAtFrame(benchmarkCase, sampleFrame);
      if (evaluatedCamera.hasError()) {
        return Result<bool, std::string>::makeError(evaluatedCamera.error());
      }
      double cameraPositionDelta = 0.0;
      double cameraDirectionDelta = 0.0;
      if (sampleFrame > 0u) {
        auto previousCamera =
            evaluateBenchmarkCameraAtFrame(benchmarkCase, sampleFrame - 1u);
        if (previousCamera.hasError()) {
          return Result<bool, std::string>::makeError(previousCamera.error());
        }
        cameraPositionDelta =
            static_cast<double>(glm::length(evaluatedCamera.value().position -
                                            previousCamera.value().position));
        cameraDirectionDelta = static_cast<double>(
            glm::length(normalizedOrDefault(evaluatedCamera.value().direction,
                                            glm::vec3(0.0f, 0.0f, -1.0f)) -
                        normalizedOrDefault(previousCamera.value().direction,
                                            glm::vec3(0.0f, 0.0f, -1.0f))));
      }
      applyBenchmarkCamera(camera, evaluatedCamera.value());

      BenchmarkFrameRecord frame{};
      frame.measurements.reserve(exactBenchmarkMetricCount() + 16u);
      frame.frameIndex = frameIndex;
      frame.sampleIndex = sampleIndex;
      frame.measured = measured;
      const auto totalBegin = std::chrono::steady_clock::now();
      if (benchmarkCase.scene.generator ==
          "nuri.procedural.ddgi_benchmark_rigid.v1") {
        const std::optional<NodeId> node = findBenchmarkNodeByName(
            scene.graph(), "DDGI Benchmark Blue Caster");
        glm::mat4 transform{1.0f};
        if (!node.has_value() ||
            !scene.graph().getNodeLocalTransform(*node, transform)) {
          return Result<bool, std::string>::makeError(
              "DDGI rigid benchmark target was not found");
        }
        transform[3].x =
            0.4f + 0.35f * std::sin(static_cast<float>(frameIndex) * 0.15f);
        if (!scene.graph().setNodeLocalTransform(*node, transform)) {
          return Result<bool, std::string>::makeError(
              "DDGI rigid benchmark target move failed");
        }
      }
      const auto commitBegin = std::chrono::steady_clock::now();
      auto commitResult = scene.commit();
      if (commitResult.hasError()) {
        return Result<bool, std::string>::makeError(commitResult.error());
      }
      const double sceneCommitMs = elapsedMs(commitBegin);
      if (benchmarkCase.scene.generator ==
          "nuri.procedural.ddgi_benchmark_deformation.v1") {
        auto animation = animationFixture.publish(scene, *gpu, frameIndex);
        if (animation.hasError()) {
          return animation;
        }
      }
      buildFrameContext(
          frameContext, scene, *renderer, settings, temporalFrameService,
          camera, frameIndex, timeSeconds, benchmarkCase.fixedDeltaSeconds,
          benchmarkCase.resolution[0], benchmarkCase.resolution[1]);
      if (measured) {
        renderer->renderGraphTelemetry().requestCapture(
            RenderGraphTelemetryLevel::PassTimings);
      }
      const BackendCreationTelemetry creationBefore =
          gpu->getBackendCreationTelemetry();
      const auto renderBegin = std::chrono::steady_clock::now();
      auto renderResult = renderer->render(pipeline, frameContext);
      const double renderSubmitMs = elapsedMs(renderBegin);
      if (renderResult.hasError()) {
        return Result<bool, std::string>::makeError(renderResult.error());
      }
      const BackendCreationTelemetry creationAfter =
          gpu->getBackendCreationTelemetry();
      const double totalMs = elapsedMs(totalBegin);
      frame.metrics = frameContext.metrics;
      if (!measured) {
        // Warmup stability uses the same submission timer as the measured
        // phase. Cooldown records are ignored by the report's phase bounds.
        frame.measurements.appendRegistered(
            NURI_BENCHMARK_METRIC("cpu.render_submit_ms"), renderSubmitMs);
      } else {
        frame.measurements.appendRegistered(
            NURI_BENCHMARK_METRIC("cpu.total_ms"), totalMs);
        frame.measurements.appendRegistered(
            NURI_BENCHMARK_METRIC("cpu.scene_commit_ms"), sceneCommitMs);
        frame.measurements.appendRegistered(
            NURI_BENCHMARK_METRIC("cpu.render_submit_ms"), renderSubmitMs);
        frame.measurements.appendRegistered(
            NURI_BENCHMARK_METRIC("benchmark.camera.position_delta"),
            cameraPositionDelta);
        frame.measurements.appendRegistered(
            NURI_BENCHMARK_METRIC("benchmark.camera.direction_delta"),
            cameraDirectionDelta);
        frame.measurements.appendRegistered(
            NURI_BENCHMARK_METRIC("cpu.scene_resource_prepare_ms"),
            sceneResourcePrepareMs);
        frame.measurements.appendRegistered(
            NURI_BENCHMARK_METRIC("renderer.backend.render_pipeline_creations"),
            static_cast<double>(creationAfter.renderPipelines -
                                creationBefore.renderPipelines));
        frame.measurements.appendRegistered(
            NURI_BENCHMARK_METRIC(
                "renderer.backend.compute_pipeline_creations"),
            static_cast<double>(creationAfter.computePipelines -
                                creationBefore.computePipelines));
        frame.measurements.appendRegistered(
            NURI_BENCHMARK_METRIC(
                "renderer.backend.meshlet_pipeline_creations"),
            static_cast<double>(creationAfter.meshletPipelines -
                                creationBefore.meshletPipelines));
        frame.measurements.appendRegistered(
            NURI_BENCHMARK_METRIC("renderer.backend.framebuffer_creations"),
            static_cast<double>(creationAfter.framebuffers -
                                creationBefore.framebuffers));
        addRendererFrameMetrics(frame.measurements, frame.metrics);
        addRenderGraphTelemetryMetrics(frame.measurements, *renderer);
        addTextureResourceMetrics(frame.measurements, *gpu);
        addProcessMemoryMetrics(frame.measurements);
        addPmrMemoryMetrics(frame.measurements, rendererMemoryTracker,
                            pipelineMemoryTracker, sceneMemoryTracker);
        measuredFrameByIndex.emplace(frame.frameIndex, report.frames.size());
      }
      report.frames.push_back(std::move(frame));
      ++frameIndex;
      timeSeconds += benchmarkCase.fixedDeltaSeconds;
      drainGpuTimings(*gpu, report, measuredFrameByIndex);
      NURI_PROFILER_ZONE_END();
      return Result<bool, std::string>::makeResult(true);
    };

    tracySession = startTracyCaptureIfRequested(benchmarkCase, options, report);
    if (tracySession.started) {
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }

    for (uint32_t sampleIndex = 0u; sampleIndex < samples; ++sampleIndex) {
      temporalFrameService.reset();
      settings.antiAliasing.debug.resetHistoryRequested = true;
      for (uint32_t i = 0u; i < benchmarkCase.warmupFrames; ++i) {
        auto frameResult = renderOneFrame(sampleIndex, i, false);
        if (frameResult.hasError()) {
          result.exitCode = BenchmarkExitCode::RuntimeError;
          result.message = frameResult.error();
          report.warnings.push_back(result.message);
          report.run.validForComparison = false;
          return finishDiagnosticsAndFinalize();
        }
      }
      for (uint32_t i = 0u; i < benchmarkCase.measurementFrames; ++i) {
        auto frameResult =
            renderOneFrame(sampleIndex, benchmarkCase.warmupFrames + i, true);
        if (frameResult.hasError()) {
          result.exitCode = BenchmarkExitCode::RuntimeError;
          result.message = frameResult.error();
          report.warnings.push_back(result.message);
          report.run.validForComparison = false;
          return finishDiagnosticsAndFinalize();
        }
      }
      for (uint32_t i = 0u; i < benchmarkCase.cooldownFrames; ++i) {
        auto frameResult = renderOneFrame(
            sampleIndex,
            benchmarkCase.warmupFrames + benchmarkCase.measurementFrames + i,
            false);
        if (frameResult.hasError()) {
          break;
        }
      }
    }

    const auto drainBegin = std::chrono::steady_clock::now();
    for (uint32_t drainFrame = 0u; drainFrame < benchmarkCase.maxDrainFrames;
         ++drainFrame) {
      drainGpuTimings(*gpu, report, measuredFrameByIndex);
      report.timingDrain.drainFrames = drainFrame;
      if (elapsedMs(drainBegin) >
          static_cast<double>(benchmarkCase.drainTimeoutMs)) {
        report.timingDrain.drainComplete = false;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    report.timingDrain.droppedGpuTimingReports =
        gpu->droppedGpuTimingReportCount();
    gpu->waitIdle();
    drainGpuTimings(*gpu, report, measuredFrameByIndex);
    for (const BenchmarkFrameRecord &frame : report.frames) {
      if (!frame.measured) {
        continue;
      }
      const auto scopes = frame.measurements.find("gpu.scopes_sum_ms");
      if (scopes == frame.measurements.end()) {
        ++report.timingDrain.missingGpuTimingFrames;
        continue;
      }
      const auto whole = frame.measurements.find("gpu.frame_ms");
      const auto commandBuffers = frame.measurements.find(
          "rendergraph.summary.recorded_command_buffer_count");
      const auto submitBatches =
          frame.measurements.find("rendergraph.summary.submit_batch_count");
      if (whole == frame.measurements.end() ||
          commandBuffers == frame.measurements.end() ||
          submitBatches == frame.measurements.end() ||
          commandBuffers->second != 1.0 || submitBatches->second != 1.0) {
        continue;
      }
      constexpr double kGpuTimingContainmentToleranceMs = 0.05;
      if (scopes->second > whole->second + kGpuTimingContainmentToleranceMs) {
        ++report.timingDrain.scopeContainmentViolations;
      }
    }
    if (report.timingDrain.droppedGpuTimingReports > 0u) {
      report.run.validForComparison = false;
      report.warnings.push_back(
          "GPU timing reports were dropped; run is not comparable");
    }
    if (report.timingDrain.missingGpuTimingFrames > 0u) {
      report.run.validForComparison = false;
      report.warnings.push_back(
          "one or more measured frames are missing GPU timing reports");
    }
    if (report.timingDrain.scopeContainmentViolations > 0u) {
      report.run.validForComparison = false;
      report.warnings.push_back(
          std::to_string(report.timingDrain.scopeContainmentViolations) +
          " single-command-buffer frames report scoped GPU work outside the "
          "same frame's whole-GPU interval");
    }
    renderDocSession.finish(report);
    rgpSession.finish(report);
  } catch (const std::exception &ex) {
    result.exitCode = BenchmarkExitCode::RuntimeError;
    result.message = ex.what();
    report.run.validForComparison = false;
    report.warnings.push_back(result.message);
  }
  renderDocSession.finish(report);
  rgpSession.finish(report);
  tracySession.finish(report);

  if (rgpDiagnostic && !report.rgp.available &&
      result.exitCode == BenchmarkExitCode::Success) {
    result.exitCode = BenchmarkExitCode::RuntimeError;
    result.message = "RGP shader diagnostic capture did not complete";
    report.warnings.push_back(result.message);
  }
  if (renderDocDiagnostic && !report.renderDoc.available &&
      result.exitCode == BenchmarkExitCode::Success) {
    result.exitCode = BenchmarkExitCode::RuntimeError;
    result.message = "RenderDoc diagnostic capture did not complete";
    report.warnings.push_back(result.message);
  }

  computeBenchmarkReportStats(report);
  const auto markUnavailable = [&](std::string_view metric) {
    if (report.stats.find(std::string(metric)) == report.stats.end() &&
        std::find(report.unavailableMetrics.begin(),
                  report.unavailableMetrics.end(),
                  metric) == report.unavailableMetrics.end()) {
      report.unavailableMetrics.emplace_back(metric);
    }
  };
  markUnavailable("gpu.frame_ms");
  markUnavailable("gpu.scopes_sum_ms");
  markUnavailable("renderer.aa.motion_class_invalid_pixels");
  markUnavailable("renderer.aa.motion_class_static_camera_only_pixels");
  markUnavailable("renderer.aa.motion_class_full_pixels");
  markUnavailable("renderer.aa.motion_class_background_rotation_pixels");
  markUnavailable("renderer.aa.motion_class_invalid_ratio");
  markUnavailable("renderer.aa.motion_class_static_camera_only_ratio");
  markUnavailable("renderer.aa.motion_class_full_ratio");
  markUnavailable("renderer.aa.motion_class_background_rotation_ratio");
  if (result.exitCode == BenchmarkExitCode::Success) {
    result.message = "benchmark run complete";
  }
  return finalizeResult();
}

BenchmarkSuiteRunResult
runBenchmarkSuite(std::vector<BenchmarkCase> benchmarkCases,
                  std::string_view suite, const BenchmarkRunOptions &options) {
  BenchmarkSuiteRunResult suiteResult{};
  std::string runId = nuri::tools::core::createRunId();
  std::filesystem::path artifactDir = options.artifactDir;
  if (artifactDir.empty()) {
    auto workspace = nuri::tools::core::createRunWorkspace(
        benchmarkRepoRoot() / "artifacts" / "bench");
    if (workspace.hasError()) {
      suiteResult.exitCode = BenchmarkExitCode::RuntimeError;
      suiteResult.message = workspace.error();
      return suiteResult;
    }
    runId = workspace.value().runId;
    artifactDir = workspace.value().root;
  }
  suiteResult.reportPath =
      options.jsonOut.empty() ? artifactDir / "run.json" : options.jsonOut;

  std::vector<nuri::tools::core::CaseCatalogEntry> catalog;
  catalog.reserve(benchmarkCases.size());
  for (const BenchmarkCase &benchmarkCase : benchmarkCases) {
    catalog.push_back({.id = benchmarkCase.id,
                       .suite = benchmarkCase.suite,
                       .manifestPath = benchmarkCase.manifestPath});
  }
  auto selected = nuri::tools::core::selectCaseCatalog(
      catalog,
      nuri::tools::core::CaseCatalogSelector{.suite = std::string(suite)},
      nuri::tools::core::CaseCatalogZeroMatchPolicy::Reject, "benchmark");
  if (selected.hasError()) {
    suiteResult.exitCode = BenchmarkExitCode::InvalidInput;
    suiteResult.message = selected.error();
    auto written =
        writeBenchmarkSuiteEnvelope(suiteResult, suite, options, artifactDir,
                                    suiteResult.reportPath, runId);
    if (written.hasError()) {
      suiteResult.exitCode = BenchmarkExitCode::RuntimeError;
      suiteResult.message = written.error();
    }
    return suiteResult;
  }
  for (const size_t index : selected.value()) {
    BenchmarkCase &benchmarkCase = benchmarkCases[index];
    BenchmarkRunOptions caseOptions = options;
    caseOptions.artifactDir = artifactDir;
    caseOptions.jsonOut.clear();
    caseOptions.envelopeOut =
        artifactDir / "cases" / (benchmarkCase.id + ".result.json");
    auto result =
        options.isolatedRepetitions.has_value()
            ? runBenchmarkCaseIsolated(std::move(benchmarkCase), caseOptions)
            : runBenchmarkCase(std::move(benchmarkCase), caseOptions);
    const auto aggregate = nuri::tools::core::aggregateOutcome(
        benchmarkOutcome(suiteResult.exitCode),
        benchmarkOutcome(result.exitCode));
    if (aggregate == benchmarkOutcome(result.exitCode)) {
      suiteResult.exitCode = result.exitCode;
    }
    suiteResult.caseResults.push_back(std::move(result));
  }
  suiteResult.message = suiteResult.exitCode == BenchmarkExitCode::Success
                            ? "suite run complete"
                            : "suite run completed with failures";
  auto written = writeBenchmarkSuiteEnvelope(
      suiteResult, suite, options, artifactDir, suiteResult.reportPath, runId);
  if (written.hasError()) {
    suiteResult.exitCode = BenchmarkExitCode::RuntimeError;
    suiteResult.message = written.error();
  }
  return suiteResult;
}

} // namespace nuri::tools::benchmark
