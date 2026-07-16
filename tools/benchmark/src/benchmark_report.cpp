#include "nuri/tools/benchmark/benchmark_report.h"

#include "nuri/tools/benchmark/benchmark_metric_registry.h"
#include "nuri/tools/core/atomic_file.h"
#include "nuri/tools/core/json_contract.h"
#include "nuri/tools/core/result_envelope_v2.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <string_view>
#include <utility>

#include <yyjson.h>

namespace nuri::tools::benchmark {
namespace {

using JsonMutDocPtr =
    std::unique_ptr<yyjson_mut_doc, decltype(&yyjson_mut_doc_free)>;
using JsonDocPtr = std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)>;
using JsonField = nuri::tools::core::JsonFieldContract;
using JsonType = nuri::tools::core::JsonFieldType;

template <size_t Size>
[[nodiscard]] Result<void, std::string>
validateObject(yyjson_val *object, const std::array<JsonField, Size> &fields,
               std::string_view path) {
  return nuri::tools::core::validateJsonObject(object, fields, path);
}

[[nodiscard]] Result<void, std::string>
validateStringArray(yyjson_val *value, std::string_view path) {
  if (!yyjson_is_arr(value)) {
    return Result<void, std::string>::makeError(std::string(path) +
                                                " must be an array");
  }
  yyjson_arr_iter iterator{};
  yyjson_arr_iter_init(value, &iterator);
  yyjson_val *entry = nullptr;
  size_t index = 0u;
  while ((entry = yyjson_arr_iter_next(&iterator)) != nullptr) {
    if (!yyjson_is_str(entry)) {
      return Result<void, std::string>::makeError(std::string(path) + "[" +
                                                  std::to_string(index) +
                                                  "] must be a string");
    }
    ++index;
  }
  return Result<void, std::string>::makeResult();
}

[[nodiscard]] Result<void, std::string>
validateNumberMap(yyjson_val *value, std::string_view path) {
  if (!yyjson_is_obj(value)) {
    return Result<void, std::string>::makeError(std::string(path) +
                                                " must be an object");
  }
  yyjson_obj_iter iterator{};
  yyjson_obj_iter_init(value, &iterator);
  yyjson_val *key = nullptr;
  while ((key = yyjson_obj_iter_next(&iterator)) != nullptr) {
    if (!yyjson_is_num(yyjson_obj_iter_get_val(key))) {
      return Result<void, std::string>::makeError(
          std::string(path) + "." + yyjson_get_str(key) + " must be a number");
    }
  }
  return Result<void, std::string>::makeResult();
}

[[nodiscard]] Result<void, std::string>
validateStatsMap(yyjson_val *value, std::string_view path) {
  if (!yyjson_is_obj(value)) {
    return Result<void, std::string>::makeError(std::string(path) +
                                                " must be an object");
  }
  static constexpr std::array statsFields{
      JsonField{"count", JsonType::Unsigned},
      JsonField{"min", JsonType::Number},
      JsonField{"median", JsonType::Number},
      JsonField{"p90", JsonType::Number},
      JsonField{"p95", JsonType::Number},
      JsonField{"p99", JsonType::Number, false},
      JsonField{"max", JsonType::Number},
      JsonField{"mean", JsonType::Number},
      JsonField{"stddev", JsonType::Number},
      JsonField{"mad", JsonType::Number},
      JsonField{"iqr", JsonType::Number},
      JsonField{"coefficientOfVariation", JsonType::Number},
  };
  yyjson_obj_iter iterator{};
  yyjson_obj_iter_init(value, &iterator);
  yyjson_val *key = nullptr;
  while ((key = yyjson_obj_iter_next(&iterator)) != nullptr) {
    auto valid = validateObject(yyjson_obj_iter_get_val(key), statsFields,
                                std::string(path) + "." + yyjson_get_str(key));
    if (valid.hasError()) {
      return valid;
    }
  }
  return Result<void, std::string>::makeResult();
}

[[nodiscard]] Result<void, std::string>
validateBenchmarkReportV1(yyjson_val *root) {
  auto valid = nuri::tools::core::rejectDuplicateJsonFieldsRecursively(root);
  if (valid.hasError()) {
    return valid;
  }
  static constexpr std::array rootFields{
      JsonField{"schemaVersion", JsonType::Unsigned},
      JsonField{"kind", JsonType::String},
      JsonField{"generatedAtUtc", JsonType::String},
      JsonField{"command", JsonType::String},
      JsonField{"environment", JsonType::Object},
      JsonField{"case", JsonType::Object},
      JsonField{"profile", JsonType::Object, false},
      JsonField{"repeatObservations", JsonType::Object, false},
      JsonField{"run", JsonType::Object},
      JsonField{"artifacts", JsonType::Object},
      JsonField{"tracy", JsonType::Object, false},
      JsonField{"frames", JsonType::Array},
      JsonField{"sampleStats", JsonType::Array},
      JsonField{"stats", JsonType::Object},
      JsonField{"renderGraph", JsonType::Object},
      JsonField{"resourceStats", JsonType::Object},
      JsonField{"timingDrain", JsonType::Object},
      JsonField{"unavailableMetrics", JsonType::Array},
      JsonField{"unregisteredObservedMetrics", JsonType::Array, false},
      JsonField{"warnings", JsonType::Array},
  };
  valid = validateObject(root, rootFields, "$");
  if (valid.hasError()) {
    return valid;
  }

  static constexpr std::array environmentFields{
      JsonField{"repoRoot", JsonType::String},
      JsonField{"commitHash", JsonType::String},
      JsonField{"branchName", JsonType::String},
      JsonField{"dirty", JsonType::Boolean},
      JsonField{"osName", JsonType::String},
      JsonField{"osVersion", JsonType::String},
      JsonField{"cpuName", JsonType::String},
      JsonField{"cpuLogicalThreadCount", JsonType::Unsigned},
      JsonField{"gpuBackend", JsonType::String},
      JsonField{"gpuBackendSource", JsonType::String},
      JsonField{"gpuDeviceName", JsonType::String},
      JsonField{"gpuVendorId", JsonType::Unsigned, false},
      JsonField{"gpuDeviceId", JsonType::Unsigned, false},
      JsonField{"gpuDriverVersion", JsonType::String, false},
      JsonField{"swapchainImageCount", JsonType::Unsigned},
      JsonField{"requestedPresentMode", JsonType::String},
      JsonField{"resolvedPresentMode", JsonType::String},
      JsonField{"presentModeSource", JsonType::String},
      JsonField{"windowMode", JsonType::String},
      JsonField{"windowVisible", JsonType::Boolean},
      JsonField{"renderGraphWorkerCount", JsonType::Unsigned},
      JsonField{"renderGraphParallelCompile", JsonType::Boolean},
      JsonField{"renderGraphParallelRecording", JsonType::Boolean},
      JsonField{"renderGraphWorkerCountSource", JsonType::String},
      JsonField{"renderGraphParallelCompileSource", JsonType::String},
      JsonField{"renderGraphParallelRecordingSource", JsonType::String},
      JsonField{"buildType", JsonType::String},
      JsonField{"cmakeToolProfile", JsonType::String},
      JsonField{"vcpkgManifestFeatures", JsonType::String},
      JsonField{"NURI_BUILD_SHARED", JsonType::Boolean},
      JsonField{"NURI_WITH_LOGGING", JsonType::Boolean},
      JsonField{"NURI_WITH_ASSERTS", JsonType::Boolean},
      JsonField{"NURI_WITH_TRACY", JsonType::Boolean},
      JsonField{"tracyDiagnostic", JsonType::Boolean},
      JsonField{"devChecks", JsonType::Boolean},
  };
  valid = validateObject(yyjson_obj_get(root, "environment"), environmentFields,
                         "$.environment");
  if (valid.hasError()) {
    return valid;
  }

  static constexpr std::array caseFields{
      JsonField{"schemaVersion", JsonType::Unsigned},
      JsonField{"id", JsonType::String},
      JsonField{"suite", JsonType::String},
      JsonField{"comparisonGroup", JsonType::String, false},
      JsonField{"variant", JsonType::String, false},
      JsonField{"description", JsonType::String},
      JsonField{"manifestPath", JsonType::String},
      JsonField{"backend", JsonType::String},
      JsonField{"resolution", JsonType::Array},
      JsonField{"presentMode", JsonType::String},
      JsonField{"authoritative", JsonType::Boolean},
      JsonField{"fixedDeltaSeconds", JsonType::Number},
      JsonField{"warmupFrames", JsonType::Unsigned},
      JsonField{"measurementFrames", JsonType::Unsigned},
      JsonField{"cooldownFrames", JsonType::Unsigned},
      JsonField{"maxDrainFrames", JsonType::Unsigned},
      JsonField{"drainTimeoutMs", JsonType::Unsigned},
      JsonField{"samples", JsonType::Unsigned},
      JsonField{"settingsSignature", JsonType::String},
      JsonField{"configSignature", JsonType::String},
      JsonField{"requiredMetrics", JsonType::Array},
      JsonField{"thresholds", JsonType::Object},
      JsonField{"scene", JsonType::Object},
      JsonField{"renderGraph", JsonType::Object},
      JsonField{"camera", JsonType::Object},
      JsonField{"timeline", JsonType::Object, false},
      JsonField{"requirements", JsonType::Object},
      JsonField{"settings", JsonType::Object},
  };
  yyjson_val *caseObject = yyjson_obj_get(root, "case");
  valid = validateObject(caseObject, caseFields, "$.case");
  if (valid.hasError()) {
    return valid;
  }
  valid = nuri::tools::core::validateJsonArtifactPath(caseObject,
                                                      "manifestPath", "$.case");
  if (valid.hasError()) {
    return valid;
  }
  valid = validateStringArray(yyjson_obj_get(caseObject, "requiredMetrics"),
                              "$.case.requiredMetrics");
  if (valid.hasError()) {
    return valid;
  }
  yyjson_val *resolution = yyjson_obj_get(caseObject, "resolution");
  if (yyjson_arr_size(resolution) != 2u ||
      !yyjson_is_uint(yyjson_arr_get(resolution, 0u)) ||
      !yyjson_is_uint(yyjson_arr_get(resolution, 1u))) {
    return Result<void, std::string>::makeError(
        "$.case.resolution must contain two unsigned integers");
  }
  static constexpr std::array thresholdFields{
      JsonField{"failPercent", JsonType::Number},
      JsonField{"failAbsoluteMs", JsonType::Number},
      JsonField{"warnPercent", JsonType::Number},
      JsonField{"warnAbsoluteMs", JsonType::Number},
  };
  valid = validateObject(yyjson_obj_get(caseObject, "thresholds"),
                         thresholdFields, "$.case.thresholds");
  if (valid.hasError()) {
    return valid;
  }
  static constexpr std::array settingsFields{
      JsonField{"opaque", JsonType::Object},
      JsonField{"antiAliasing", JsonType::Object},
      JsonField{"ambientOcclusion", JsonType::Object},
      JsonField{"shadow", JsonType::Object},
      JsonField{"visibility", JsonType::Object},
      JsonField{"hdrPostProcess", JsonType::Object},
      JsonField{"transmission", JsonType::Object},
      JsonField{"transparent", JsonType::Object},
      JsonField{"textureFiltering", JsonType::Object},
  };
  valid = validateObject(yyjson_obj_get(caseObject, "settings"), settingsFields,
                         "$.case.settings");
  if (valid.hasError()) {
    return valid;
  }

  static constexpr std::array profileFields{
      JsonField{"id", JsonType::String},
      JsonField{"profileAuthoritative", JsonType::Boolean},
      JsonField{"authoritative", JsonType::Boolean},
      JsonField{"minimumRepetitions", JsonType::Unsigned},
      JsonField{"completedRepetitions", JsonType::Unsigned},
      JsonField{"repetitionRequirementSatisfied", JsonType::Boolean},
      JsonField{"repetitionUnit", JsonType::String},
      JsonField{"warmupStabilityPolicy", JsonType::String},
      JsonField{"warmupStabilityStatus", JsonType::String},
      JsonField{"warmupWindowFrames", JsonType::Unsigned, false},
      JsonField{"warmupMaxDriftPercent", JsonType::Number, false},
      JsonField{"requiredMetrics", JsonType::Array},
      JsonField{"authorityBlockers", JsonType::Array},
  };
  if (yyjson_val *profile = yyjson_obj_get(root, "profile")) {
    valid = validateObject(profile, profileFields, "$.profile");
    if (valid.hasError()) {
      return valid;
    }
    valid = validateStringArray(yyjson_obj_get(profile, "requiredMetrics"),
                                "$.profile.requiredMetrics");
    if (valid.hasError()) {
      return valid;
    }
    valid = validateStringArray(yyjson_obj_get(profile, "authorityBlockers"),
                                "$.profile.authorityBlockers");
    if (valid.hasError()) {
      return valid;
    }
  }

  static constexpr std::array runFields{
      JsonField{"samples", JsonType::Unsigned},
      JsonField{"warmupFrames", JsonType::Unsigned},
      JsonField{"measurementFrames", JsonType::Unsigned},
      JsonField{"cooldownFrames", JsonType::Unsigned},
      JsonField{"maxDrainFrames", JsonType::Unsigned},
      JsonField{"drainTimeoutMs", JsonType::Unsigned},
      JsonField{"validForComparison", JsonType::Boolean},
      JsonField{"fixedDeltaSeconds", JsonType::Number},
  };
  valid = validateObject(yyjson_obj_get(root, "run"), runFields, "$.run");
  if (valid.hasError()) {
    return valid;
  }
  static constexpr std::array artifactFields{
      JsonField{"artifactDir", JsonType::String},
      JsonField{"caseReports", JsonType::Array},
      JsonField{"tracy", JsonType::Array},
  };
  yyjson_val *artifacts = yyjson_obj_get(root, "artifacts");
  valid = validateObject(artifacts, artifactFields, "$.artifacts");
  if (valid.hasError()) {
    return valid;
  }
  valid = nuri::tools::core::validateJsonArtifactPath(artifacts, "artifactDir",
                                                      "$.artifacts");
  if (valid.hasError()) {
    return valid;
  }
  for (std::string_view field : {"caseReports", "tracy"}) {
    yyjson_val *paths = yyjson_obj_getn(artifacts, field.data(), field.size());
    yyjson_arr_iter iterator{};
    yyjson_arr_iter_init(paths, &iterator);
    yyjson_val *entry = nullptr;
    size_t index = 0u;
    while ((entry = yyjson_arr_iter_next(&iterator)) != nullptr) {
      if (!yyjson_is_str(entry)) {
        return Result<void, std::string>::makeError(
            "$.artifacts." + std::string(field) + "[" + std::to_string(index) +
            "] must be a string");
      }
      valid = nuri::tools::core::validateJsonArtifactPath(
          std::string_view(yyjson_get_str(entry), yyjson_get_len(entry)),
          "$.artifacts." + std::string(field) + "[" + std::to_string(index) +
              "]");
      if (valid.hasError()) {
        return valid;
      }
      ++index;
    }
  }

  static constexpr std::array frameFields{
      JsonField{"frameIndex", JsonType::Unsigned},
      JsonField{"sampleIndex", JsonType::Unsigned},
      JsonField{"measured", JsonType::Boolean},
      JsonField{"measurements", JsonType::Object},
      JsonField{"metrics", JsonType::Object, false},
  };
  yyjson_arr_iter frameIterator{};
  yyjson_arr_iter_init(yyjson_obj_get(root, "frames"), &frameIterator);
  yyjson_val *frame = nullptr;
  size_t frameIndex = 0u;
  while ((frame = yyjson_arr_iter_next(&frameIterator)) != nullptr) {
    const std::string path = "$.frames[" + std::to_string(frameIndex++) + "]";
    valid = validateObject(frame, frameFields, path);
    if (valid.hasError()) {
      return valid;
    }
    valid = validateNumberMap(yyjson_obj_get(frame, "measurements"),
                              path + ".measurements");
    if (valid.hasError()) {
      return valid;
    }
  }
  static constexpr std::array sampleFields{
      JsonField{"sampleIndex", JsonType::Unsigned},
      JsonField{"warmupStable", JsonType::NullOrBoolean, false},
      JsonField{"measuredFrameStart", JsonType::Unsigned},
      JsonField{"measuredFrameCount", JsonType::Unsigned},
      JsonField{"stats", JsonType::Object},
      JsonField{"warnings", JsonType::Array},
  };
  yyjson_arr_iter sampleIterator{};
  yyjson_arr_iter_init(yyjson_obj_get(root, "sampleStats"), &sampleIterator);
  yyjson_val *sample = nullptr;
  size_t sampleIndex = 0u;
  while ((sample = yyjson_arr_iter_next(&sampleIterator)) != nullptr) {
    const std::string path =
        "$.sampleStats[" + std::to_string(sampleIndex++) + "]";
    valid = validateObject(sample, sampleFields, path);
    if (valid.hasError()) {
      return valid;
    }
    valid = validateStatsMap(yyjson_obj_get(sample, "stats"), path + ".stats");
    if (valid.hasError()) {
      return valid;
    }
    valid = validateStringArray(yyjson_obj_get(sample, "warnings"),
                                path + ".warnings");
    if (valid.hasError()) {
      return valid;
    }
  }
  valid = validateStatsMap(yyjson_obj_get(root, "stats"), "$.stats");
  if (valid.hasError()) {
    return valid;
  }
  static constexpr std::array drainFields{
      JsonField{"drainComplete", JsonType::Boolean},
      JsonField{"drainFrames", JsonType::Unsigned},
      JsonField{"drainTimeoutMs", JsonType::Unsigned},
      JsonField{"missingGpuTimingFrames", JsonType::Unsigned},
      JsonField{"scopeContainmentViolations", JsonType::Unsigned, false},
      JsonField{"droppedGpuTimingReports", JsonType::Unsigned},
  };
  valid = validateObject(yyjson_obj_get(root, "timingDrain"), drainFields,
                         "$.timingDrain");
  if (valid.hasError()) {
    return valid;
  }
  for (std::string_view field :
       {"unavailableMetrics", "unregisteredObservedMetrics", "warnings"}) {
    if (yyjson_val *array = yyjson_obj_getn(root, field.data(), field.size())) {
      valid = validateStringArray(array, "$." + std::string(field));
      if (valid.hasError()) {
        return valid;
      }
    }
  }
  return Result<void, std::string>::makeResult();
}

void addString(yyjson_mut_doc *doc, yyjson_mut_val *object, const char *key,
               const std::string &value) {
  yyjson_mut_obj_add_strcpy(doc, object, key, value.c_str());
}

[[nodiscard]] std::string pathToUtf8(const std::filesystem::path &value) {
  const std::u8string encoded = value.generic_u8string();
  return {reinterpret_cast<const char *>(encoded.data()), encoded.size()};
}

void addPath(yyjson_mut_doc *doc, yyjson_mut_val *object, const char *key,
             const std::filesystem::path &value) {
  addString(doc, object, key, pathToUtf8(value));
}

yyjson_mut_val *makeStringArray(yyjson_mut_doc *doc,
                                const std::vector<std::string> &values) {
  yyjson_mut_val *array = yyjson_mut_arr(doc);
  for (const std::string &value : values) {
    yyjson_mut_arr_add_strcpy(doc, array, value.c_str());
  }
  return array;
}

yyjson_mut_val *
makePathArray(yyjson_mut_doc *doc,
              const std::vector<std::filesystem::path> &values) {
  yyjson_mut_val *array = yyjson_mut_arr(doc);
  for (const std::filesystem::path &value : values) {
    const std::string encoded = pathToUtf8(value);
    yyjson_mut_arr_add_strcpy(doc, array, encoded.c_str());
  }
  return array;
}

yyjson_mut_val *makeStatsObject(yyjson_mut_doc *doc, const MetricStats &stats) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_uint(doc, object, "count", stats.count);
  yyjson_mut_obj_add_real(doc, object, "min", stats.min);
  yyjson_mut_obj_add_real(doc, object, "median", stats.median);
  yyjson_mut_obj_add_real(doc, object, "p90", stats.p90);
  yyjson_mut_obj_add_real(doc, object, "p95", stats.p95);
  yyjson_mut_obj_add_real(doc, object, "p99", stats.p99);
  yyjson_mut_obj_add_real(doc, object, "max", stats.max);
  yyjson_mut_obj_add_real(doc, object, "mean", stats.mean);
  yyjson_mut_obj_add_real(doc, object, "stddev", stats.stddev);
  yyjson_mut_obj_add_real(doc, object, "mad", stats.mad);
  yyjson_mut_obj_add_real(doc, object, "iqr", stats.iqr);
  yyjson_mut_obj_add_real(doc, object, "coefficientOfVariation",
                          stats.coefficientOfVariation);
  return object;
}

yyjson_mut_val *
makeStatsMapObject(yyjson_mut_doc *doc,
                   const std::map<std::string, MetricStats> &stats) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  for (const auto &[metricId, metricStats] : stats) {
    yyjson_mut_obj_add_val(doc, object, metricId.c_str(),
                           makeStatsObject(doc, metricStats));
  }
  return object;
}

template <typename Value>
void appendSignatureField(std::ostringstream &out, std::string_view key,
                          const Value &value) {
  out << '|' << key << '=' << value;
}

void appendSignatureField(std::ostringstream &out, std::string_view key,
                          bool value) {
  out << '|' << key << '=' << (value ? 1 : 0);
}

void appendSignatureField(std::ostringstream &out, std::string_view key,
                          const std::filesystem::path &value) {
  out << '|' << key << '=' << pathToUtf8(value);
}

void appendSignatureList(std::ostringstream &out, std::string_view key,
                         const std::vector<std::string> &values) {
  out << '|' << key << "=[";
  for (size_t i = 0u; i < values.size(); ++i) {
    if (i != 0u) {
      out << ',';
    }
    out << values[i];
  }
  out << ']';
}

void appendSignatureVec3(std::ostringstream &out, std::string_view key,
                         const glm::vec3 &value) {
  out << '|' << key << '=' << value.x << ',' << value.y << ',' << value.z;
}

template <typename Enum> [[nodiscard]] uint32_t enumValue(Enum value) {
  return static_cast<uint32_t>(value);
}

[[nodiscard]] const char *antiAliasingModeName(AntiAliasingMode mode) {
  switch (mode) {
  case AntiAliasingMode::None:
    return "None";
  case AntiAliasingMode::TAA:
    return "TAA";
  case AntiAliasingMode::SpatialFallback:
    return "SpatialFallback";
  case AntiAliasingMode::MSAA4x:
    return "MSAA4x";
  }
  return "Unknown";
}

[[nodiscard]] const char *
temporalReconstructionProviderName(TemporalReconstructionProvider provider) {
  switch (provider) {
  case TemporalReconstructionProvider::Legacy:
    return "Legacy";
  case TemporalReconstructionProvider::Reference:
    return "Reference";
  case TemporalReconstructionProvider::External:
    return "External";
  }
  return "Unknown";
}

[[nodiscard]] const char *
temporalAAQualityPresetName(TemporalAAQualityPreset preset) {
  switch (preset) {
  case TemporalAAQualityPreset::Performance:
    return "Performance";
  case TemporalAAQualityPreset::Balanced:
    return "Balanced";
  case TemporalAAQualityPreset::Quality:
    return "Quality";
  case TemporalAAQualityPreset::Ultra:
    return "Ultra";
  case TemporalAAQualityPreset::Custom:
    return "Custom";
  }
  return "Unknown";
}

[[nodiscard]] const char *ambientOcclusionModeName(AmbientOcclusionMode mode) {
  switch (mode) {
  case AmbientOcclusionMode::Disabled:
    return "Disabled";
  case AmbientOcclusionMode::GTAO:
    return "GTAO";
  }
  return "Unknown";
}

[[nodiscard]] const char *
ambientOcclusionPresetName(AmbientOcclusionPreset preset) {
  switch (preset) {
  case AmbientOcclusionPreset::Low:
    return "Low";
  case AmbientOcclusionPreset::Balanced:
    return "Balanced";
  case AmbientOcclusionPreset::High:
    return "High";
  case AmbientOcclusionPreset::Ultra:
    return "Ultra";
  case AmbientOcclusionPreset::Custom:
    return "Custom";
  }
  return "Unknown";
}

[[nodiscard]] const char *shadowQualityPresetName(ShadowQualityPreset preset) {
  switch (preset) {
  case ShadowQualityPreset::Custom:
    return "Custom";
  case ShadowQualityPreset::Low:
    return "Low";
  case ShadowQualityPreset::Medium:
    return "Medium";
  case ShadowQualityPreset::High:
    return "High";
  case ShadowQualityPreset::Ultra:
    return "Ultra";
  }
  return "Unknown";
}

[[nodiscard]] const char *textureFilterModeName(TextureFilterMode mode) {
  switch (mode) {
  case TextureFilterMode::Bilinear:
    return "Bilinear";
  case TextureFilterMode::Trilinear:
    return "Trilinear";
  case TextureFilterMode::Anisotropic:
    return "Anisotropic";
  }
  return "Unknown";
}

[[nodiscard]] const char *
visibilityCullingModeName(VisibilityCullingMode mode) {
  switch (mode) {
  case VisibilityCullingMode::Disabled:
    return "Disabled";
  case VisibilityCullingMode::CpuCoarse:
    return "CpuCoarse";
  case VisibilityCullingMode::Hybrid:
    return "Hybrid";
  case VisibilityCullingMode::GpuDriven:
    return "GpuDriven";
  }
  return "Unknown";
}

[[nodiscard]] const char *
visibilityOcclusionModeName(VisibilityOcclusionMode mode) {
  switch (mode) {
  case VisibilityOcclusionMode::Disabled:
    return "Disabled";
  case VisibilityOcclusionMode::PreviousFrameHiZ:
    return "PreviousFrameHiZ";
  case VisibilityOcclusionMode::CurrentFrameHiZExperimental:
    return "CurrentFrameHiZExperimental";
  }
  return "Unknown";
}

[[nodiscard]] const char *meshletRenderModeName(MeshletRenderMode mode) {
  switch (mode) {
  case MeshletRenderMode::Disabled:
    return "Disabled";
  case MeshletRenderMode::Opportunistic:
    return "Opportunistic";
  case MeshletRenderMode::Required:
    return "Required";
  }
  return "Unknown";
}

[[nodiscard]] std::string
makeSettingsSignature(const RenderSettings &sourceSettings) {
  RenderSettings settings = sourceSettings;
  sanitizeBenchmarkRenderSettings(settings);

  std::ostringstream out;
  out << std::setprecision(9) << "settings.v1";
  appendSignatureField(out, "opaque.enabled", settings.opaque.enabled);
  appendSignatureField(out, "opaque.depthPrepass",
                       settings.opaque.enableDepthPrepass);
  appendSignatureField(out, "opaque.depthPyramid",
                       settings.opaque.enableDepthPyramid);
  appendSignatureField(out, "opaque.instanceCompute",
                       settings.opaque.enableInstanceCompute);
  appendSignatureField(out, "opaque.indirectDraw",
                       settings.opaque.enableIndirectDraw);
  appendSignatureField(out, "opaque.instancedDraw",
                       settings.opaque.enableInstancedDraw);
  appendSignatureField(out, "opaque.meshLod", settings.opaque.enableMeshLod);
  appendSignatureField(out, "opaque.meshLodTargetPixelError",
                       settings.opaque.meshLodTargetPixelError);
  appendSignatureField(out, "opaque.meshLodHysteresisRatio",
                       settings.opaque.meshLodHysteresisRatio);
  appendSignatureField(out, "opaque.cpuFrustum",
                       settings.opaque.enableCpuFrustumCulling);
  appendSignatureField(out, "opaque.meshletMode",
                       enumValue(settings.opaque.meshletMode));
  appendSignatureField(out, "opaque.hybridClassicMaxMeshlets",
                       settings.opaque.hybridClassicMaxMeshlets);
  appendSignatureField(out, "opaque.meshletFrustum",
                       settings.opaque.enableMeshletFrustumCulling);
  appendSignatureField(out, "opaque.meshletCone",
                       settings.opaque.enableMeshletConeCulling);
  appendSignatureField(out, "opaque.instanceAnimation",
                       settings.opaque.enableInstanceAnimation);
  appendSignatureField(out, "opaque.tessellation",
                       settings.opaque.enableTessellation);
  appendSignatureField(out, "opaque.forcedMeshLod",
                       settings.opaque.forcedMeshLod);
  appendSignatureField(out, "aa.mode", enumValue(settings.antiAliasing.mode));
  appendSignatureField(out, "aa.temporalProvider",
                       enumValue(settings.antiAliasing.temporalProvider));
  appendSignatureField(out, "aa.quality",
                       enumValue(settings.antiAliasing.qualityPreset));
  appendSignatureField(out, "aa.spatialPostMsaaCleanup",
                       settings.antiAliasing.debug.spatialPostMsaaCleanup);
  appendSignatureField(out, "ao.mode",
                       enumValue(settings.ambientOcclusion.mode));
  appendSignatureField(out, "ao.preset",
                       enumValue(settings.ambientOcclusion.preset));
  appendSignatureField(out, "shadow.enabled", settings.shadow.enabled);
  appendSignatureField(out, "shadow.quality",
                       enumValue(settings.shadow.qualityPreset));
  appendSignatureField(out, "visibility.main",
                       enumValue(settings.visibility.mainViewMode));
  appendSignatureField(out, "visibility.shadow",
                       enumValue(settings.visibility.shadowMode));
  appendSignatureField(out, "visibility.occlusion",
                       enumValue(settings.visibility.occlusionMode));
  appendSignatureField(out, "visibility.meshletFrustum",
                       settings.visibility.enableMeshletFrustumCulling);
  appendSignatureField(out, "visibility.meshletCone",
                       settings.visibility.enableMeshletConeCulling);
  appendSignatureField(out, "visibility.gpuInstance",
                       settings.visibility.enableGpuInstanceCulling);
  appendSignatureField(out, "visibility.gpuIndirect",
                       settings.visibility.enableGpuIndirectDraw);
  appendSignatureField(out, "visibility.indirectMesh",
                       settings.visibility.enableIndirectMeshDispatch);
  appendSignatureField(out, "visibility.preTaskCompaction",
                       settings.visibility.enableMeshletPreTaskCompaction);
  appendSignatureField(out, "visibility.visibleOnUncertain",
                       settings.visibility.visibleOnUncertain);
  appendSignatureField(out, "hdr.bloom", settings.hdrPostProcess.bloomEnabled);
  appendSignatureField(out, "hdr.adaptation",
                       settings.hdrPostProcess.adaptationEnabled);
  appendSignatureField(out, "transmission.enabled",
                       settings.transmission.enabled);
  appendSignatureField(out, "transparent.enabled",
                       settings.transparent.enabled);
  appendSignatureField(out, "texture.mode",
                       enumValue(settings.textureFiltering.mode));
  appendSignatureField(
      out, "texture.anisotropy",
      static_cast<uint32_t>(settings.textureFiltering.anisotropy));
  return out.str();
}

[[nodiscard]] std::string
makeConfigSignature(const BenchmarkCase &benchmarkCase,
                    std::string_view settingsSignature) {
  std::ostringstream out;
  out << std::setprecision(9) << "config.v1";
  appendSignatureField(out, "suite", benchmarkCase.suite);
  appendSignatureField(out, "comparisonGroup", benchmarkCase.comparisonGroup);
  appendSignatureField(out, "variant", benchmarkCase.variant);
  appendSignatureField(out, "backend", benchmarkCase.backend);
  appendSignatureField(out, "width", benchmarkCase.resolution[0]);
  appendSignatureField(out, "height", benchmarkCase.resolution[1]);
  appendSignatureField(out, "presentMode", benchmarkCase.presentMode);
  appendSignatureField(out, "fixedDeltaSeconds",
                       benchmarkCase.fixedDeltaSeconds);
  appendSignatureField(out, "warmupFrames", benchmarkCase.warmupFrames);
  appendSignatureField(out, "measurementFrames",
                       benchmarkCase.measurementFrames);
  appendSignatureField(out, "cooldownFrames", benchmarkCase.cooldownFrames);
  appendSignatureField(out, "maxDrainFrames", benchmarkCase.maxDrainFrames);
  appendSignatureField(out, "drainTimeoutMs", benchmarkCase.drainTimeoutMs);
  appendSignatureField(out, "samples", benchmarkCase.samples);
  appendSignatureField(out, "authoritative", benchmarkCase.authoritative);
  appendSignatureField(out, "renderGraph.workerCount",
                       benchmarkCase.renderGraph.workerCount);
  appendSignatureField(out, "renderGraph.parallelCompile",
                       benchmarkCase.renderGraph.parallelCompile);
  appendSignatureField(out, "renderGraph.parallelRecording",
                       benchmarkCase.renderGraph.parallelRecording);
  appendSignatureVec3(out, "camera.position", benchmarkCase.camera.position);
  appendSignatureVec3(out, "camera.direction", benchmarkCase.camera.direction);
  appendSignatureField(out, "camera.verticalFovDegrees",
                       benchmarkCase.camera.verticalFovDegrees);
  appendSignatureField(out, "camera.nearPlane", benchmarkCase.camera.nearPlane);
  appendSignatureField(out, "camera.farPlane", benchmarkCase.camera.farPlane);
  appendSignatureField(out, "scene.kind", benchmarkCase.scene.kind);
  appendSignatureField(out, "scene.pathBase", benchmarkCase.scene.pathBase);
  appendSignatureField(out, "scene.path", benchmarkCase.scene.path);
  appendSignatureField(out, "scene.flipUVs", benchmarkCase.scene.flipUVs);
  appendSignatureField(out, "scene.generateMeshlets",
                       benchmarkCase.scene.generateMeshlets);
  appendSignatureField(out, "scene.meshletMaxVertices",
                       benchmarkCase.scene.meshletMaxVertices);
  appendSignatureField(out, "scene.meshletMaxPrimitives",
                       benchmarkCase.scene.meshletMaxPrimitives);
  appendSignatureField(out, "scene.meshletConeWeight",
                       benchmarkCase.scene.meshletConeWeight);
  appendSignatureField(out, "scene.baseModelKind",
                       benchmarkCase.scene.baseModelKind);
  appendSignatureField(out, "scene.baseModelTargetRadius",
                       benchmarkCase.scene.baseModelTargetRadius);
  appendSignatureField(out, "scene.baseModelMinScale",
                       benchmarkCase.scene.baseModelMinScale);
  appendSignatureField(out, "scene.baseModelMaxScale",
                       benchmarkCase.scene.baseModelMaxScale);
  appendSignatureField(out, "scene.generator", benchmarkCase.scene.generator);
  appendSignatureField(out, "scene.seed", benchmarkCase.scene.seed);
  appendSignatureField(out, "scene.contentHash",
                       benchmarkCase.scene.contentHash);
  appendSignatureField(out, "timeline.cameraPathCount",
                       benchmarkCase.timeline.cameraPaths.size());
  for (size_t pathIndex = 0u;
       pathIndex < benchmarkCase.timeline.cameraPaths.size(); ++pathIndex) {
    const BenchmarkCameraPath &path =
        benchmarkCase.timeline.cameraPaths[pathIndex];
    const std::string pathPrefix =
        "timeline.cameraPath." + std::to_string(pathIndex) + ".";
    appendSignatureField(out, pathPrefix + "id", path.id);
    appendSignatureField(out, pathPrefix + "startFrame", path.startFrame);
    appendSignatureField(out, pathPrefix + "endFrame", path.endFrame);
    appendSignatureField(out, pathPrefix + "interpolation", path.interpolation);
    appendSignatureField(out, pathPrefix + "keyframeCount",
                         path.keyframes.size());
    for (size_t keyframeIndex = 0u; keyframeIndex < path.keyframes.size();
         ++keyframeIndex) {
      const BenchmarkCameraKeyframe &keyframe = path.keyframes[keyframeIndex];
      const std::string keyframePrefix =
          pathPrefix + "keyframe." + std::to_string(keyframeIndex) + ".";
      appendSignatureField(out, keyframePrefix + "frame", keyframe.frame);
      appendSignatureVec3(out, keyframePrefix + "position", keyframe.position);
      appendSignatureField(out, keyframePrefix + "hasTarget",
                           keyframe.hasTarget);
      appendSignatureVec3(out, keyframePrefix + "target", keyframe.target);
    }
  }
  appendSignatureList(out, "requirements.assets",
                      benchmarkCase.requirements.assets);
  appendSignatureList(out, "requirements.backends",
                      benchmarkCase.requirements.backends);
  appendSignatureField(out, "requirements.allowVisibleWindow",
                       benchmarkCase.requirements.allowVisibleWindow);
  appendSignatureField(out, "requirements.msaa4x",
                       benchmarkCase.requirements.msaa4x);
  appendSignatureField(out, "settingsSignature",
                       std::string(settingsSignature));
  return out.str();
}

yyjson_mut_val *makeVec3Array(yyjson_mut_doc *doc, const glm::vec3 &value) {
  yyjson_mut_val *array = yyjson_mut_arr(doc);
  yyjson_mut_arr_add_real(doc, array, value.x);
  yyjson_mut_arr_add_real(doc, array, value.y);
  yyjson_mut_arr_add_real(doc, array, value.z);
  return array;
}

yyjson_mut_val *makeTimelineObject(yyjson_mut_doc *doc,
                                   const BenchmarkTimeline &timeline) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  yyjson_mut_val *paths = yyjson_mut_arr(doc);
  for (const BenchmarkCameraPath &path : timeline.cameraPaths) {
    yyjson_mut_val *pathObject = yyjson_mut_obj(doc);
    addString(doc, pathObject, "id", path.id);
    yyjson_mut_obj_add_uint(doc, pathObject, "startFrame", path.startFrame);
    yyjson_mut_obj_add_uint(doc, pathObject, "endFrame", path.endFrame);
    addString(doc, pathObject, "interpolation", path.interpolation);
    yyjson_mut_val *keyframes = yyjson_mut_arr(doc);
    for (const BenchmarkCameraKeyframe &keyframe : path.keyframes) {
      yyjson_mut_val *keyframeObject = yyjson_mut_obj(doc);
      yyjson_mut_obj_add_uint(doc, keyframeObject, "frame", keyframe.frame);
      yyjson_mut_obj_add_val(doc, keyframeObject, "position",
                             makeVec3Array(doc, keyframe.position));
      yyjson_mut_obj_add_bool(doc, keyframeObject, "hasTarget",
                              keyframe.hasTarget);
      yyjson_mut_obj_add_val(doc, keyframeObject, "target",
                             makeVec3Array(doc, keyframe.target));
      yyjson_mut_arr_add_val(keyframes, keyframeObject);
    }
    yyjson_mut_obj_add_val(doc, pathObject, "keyframes", keyframes);
    yyjson_mut_arr_add_val(paths, pathObject);
  }
  yyjson_mut_obj_add_val(doc, object, "cameraPaths", paths);
  return object;
}

yyjson_mut_val *makeSettingsObject(yyjson_mut_doc *doc,
                                   const RenderSettings &sourceSettings) {
  RenderSettings settings = sourceSettings;
  sanitizeBenchmarkRenderSettings(settings);

  yyjson_mut_val *object = yyjson_mut_obj(doc);
  yyjson_mut_val *opaque = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_bool(doc, opaque, "enabled", settings.opaque.enabled);
  yyjson_mut_obj_add_bool(doc, opaque, "enableDepthPrepass",
                          settings.opaque.enableDepthPrepass);
  yyjson_mut_obj_add_bool(doc, opaque, "enableDepthPyramid",
                          settings.opaque.enableDepthPyramid);
  yyjson_mut_obj_add_bool(doc, opaque, "enableInstanceCompute",
                          settings.opaque.enableInstanceCompute);
  yyjson_mut_obj_add_bool(doc, opaque, "enableIndirectDraw",
                          settings.opaque.enableIndirectDraw);
  yyjson_mut_obj_add_bool(doc, opaque, "enableInstancedDraw",
                          settings.opaque.enableInstancedDraw);
  yyjson_mut_obj_add_bool(doc, opaque, "enableMeshLod",
                          settings.opaque.enableMeshLod);
  yyjson_mut_obj_add_real(doc, opaque, "meshLodTargetPixelError",
                          settings.opaque.meshLodTargetPixelError);
  yyjson_mut_obj_add_real(doc, opaque, "meshLodHysteresisRatio",
                          settings.opaque.meshLodHysteresisRatio);
  yyjson_mut_obj_add_bool(doc, opaque, "enableCpuFrustumCulling",
                          settings.opaque.enableCpuFrustumCulling);
  addString(doc, opaque, "meshletMode",
            meshletRenderModeName(settings.opaque.meshletMode));
  yyjson_mut_obj_add_uint(doc, opaque, "hybridClassicMaxMeshlets",
                          settings.opaque.hybridClassicMaxMeshlets);
  yyjson_mut_obj_add_bool(doc, opaque, "enableMeshletFrustumCulling",
                          settings.opaque.enableMeshletFrustumCulling);
  yyjson_mut_obj_add_bool(doc, opaque, "enableMeshletConeCulling",
                          settings.opaque.enableMeshletConeCulling);
  yyjson_mut_obj_add_bool(doc, opaque, "enableInstanceAnimation",
                          settings.opaque.enableInstanceAnimation);
  yyjson_mut_obj_add_bool(doc, opaque, "enableTessellation",
                          settings.opaque.enableTessellation);
  yyjson_mut_obj_add_sint(doc, opaque, "forcedMeshLod",
                          settings.opaque.forcedMeshLod);
  yyjson_mut_obj_add_val(doc, object, "opaque", opaque);

  yyjson_mut_val *antiAliasing = yyjson_mut_obj(doc);
  addString(doc, antiAliasing, "mode",
            antiAliasingModeName(settings.antiAliasing.mode));
  addString(doc, antiAliasing, "temporalProvider",
            temporalReconstructionProviderName(
                settings.antiAliasing.temporalProvider));
  addString(doc, antiAliasing, "qualityPreset",
            temporalAAQualityPresetName(settings.antiAliasing.qualityPreset));
  yyjson_mut_obj_add_bool(doc, antiAliasing, "spatialPostMsaaCleanup",
                          settings.antiAliasing.debug.spatialPostMsaaCleanup);
  yyjson_mut_obj_add_val(doc, object, "antiAliasing", antiAliasing);

  yyjson_mut_val *ambientOcclusion = yyjson_mut_obj(doc);
  addString(doc, ambientOcclusion, "mode",
            ambientOcclusionModeName(settings.ambientOcclusion.mode));
  addString(doc, ambientOcclusion, "preset",
            ambientOcclusionPresetName(settings.ambientOcclusion.preset));
  yyjson_mut_obj_add_val(doc, object, "ambientOcclusion", ambientOcclusion);

  yyjson_mut_val *shadow = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_bool(doc, shadow, "enabled", settings.shadow.enabled);
  addString(doc, shadow, "qualityPreset",
            shadowQualityPresetName(settings.shadow.qualityPreset));
  yyjson_mut_obj_add_val(doc, object, "shadow", shadow);

  yyjson_mut_val *visibility = yyjson_mut_obj(doc);
  addString(doc, visibility, "mainViewMode",
            visibilityCullingModeName(settings.visibility.mainViewMode));
  addString(doc, visibility, "shadowMode",
            visibilityCullingModeName(settings.visibility.shadowMode));
  addString(doc, visibility, "occlusionMode",
            visibilityOcclusionModeName(settings.visibility.occlusionMode));
  yyjson_mut_obj_add_bool(doc, visibility, "enableMeshletFrustumCulling",
                          settings.visibility.enableMeshletFrustumCulling);
  yyjson_mut_obj_add_bool(doc, visibility, "enableMeshletConeCulling",
                          settings.visibility.enableMeshletConeCulling);
  yyjson_mut_obj_add_bool(doc, visibility, "enableGpuInstanceCulling",
                          settings.visibility.enableGpuInstanceCulling);
  yyjson_mut_obj_add_bool(doc, visibility, "enableGpuIndirectDraw",
                          settings.visibility.enableGpuIndirectDraw);
  yyjson_mut_obj_add_bool(doc, visibility, "enableIndirectMeshDispatch",
                          settings.visibility.enableIndirectMeshDispatch);
  yyjson_mut_obj_add_bool(doc, visibility, "enableMeshletPreTaskCompaction",
                          settings.visibility.enableMeshletPreTaskCompaction);
  yyjson_mut_obj_add_bool(doc, visibility, "visibleOnUncertain",
                          settings.visibility.visibleOnUncertain);
  yyjson_mut_obj_add_val(doc, object, "visibility", visibility);

  yyjson_mut_val *hdr = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_bool(doc, hdr, "bloomEnabled",
                          settings.hdrPostProcess.bloomEnabled);
  yyjson_mut_obj_add_bool(doc, hdr, "adaptationEnabled",
                          settings.hdrPostProcess.adaptationEnabled);
  yyjson_mut_obj_add_val(doc, object, "hdrPostProcess", hdr);

  yyjson_mut_val *transmission = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_bool(doc, transmission, "enabled",
                          settings.transmission.enabled);
  yyjson_mut_obj_add_val(doc, object, "transmission", transmission);

  yyjson_mut_val *transparent = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_bool(doc, transparent, "enabled",
                          settings.transparent.enabled);
  yyjson_mut_obj_add_val(doc, object, "transparent", transparent);

  yyjson_mut_val *textureFiltering = yyjson_mut_obj(doc);
  addString(doc, textureFiltering, "mode",
            textureFilterModeName(settings.textureFiltering.mode));
  yyjson_mut_obj_add_uint(doc, textureFiltering, "anisotropy",
                          settings.textureFiltering.anisotropy);
  yyjson_mut_obj_add_val(doc, object, "textureFiltering", textureFiltering);
  return object;
}

yyjson_mut_val *makeEnvironmentObject(yyjson_mut_doc *doc,
                                      const BenchmarkEnvironment &env) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  addPath(doc, object, "repoRoot", env.repoRoot);
  addString(doc, object, "commitHash", env.commitHash);
  addString(doc, object, "branchName", env.branchName);
  yyjson_mut_obj_add_bool(doc, object, "dirty", env.dirty);
  addString(doc, object, "osName", env.osName);
  addString(doc, object, "osVersion", env.osVersion);
  addString(doc, object, "cpuName", env.cpuName);
  yyjson_mut_obj_add_uint(doc, object, "cpuLogicalThreadCount",
                          env.cpuLogicalThreadCount);
  addString(doc, object, "gpuBackend", env.gpuBackend);
  addString(doc, object, "gpuBackendSource", env.gpuBackendSource);
  addString(doc, object, "gpuDeviceName", env.gpuDeviceName);
  yyjson_mut_obj_add_uint(doc, object, "gpuVendorId", env.gpuVendorId);
  yyjson_mut_obj_add_uint(doc, object, "gpuDeviceId", env.gpuDeviceId);
  addString(doc, object, "gpuDriverVersion", env.gpuDriverVersion);
  yyjson_mut_obj_add_uint(doc, object, "swapchainImageCount",
                          env.swapchainImageCount);
  addString(doc, object, "requestedPresentMode", env.requestedPresentMode);
  addString(doc, object, "resolvedPresentMode", env.resolvedPresentMode);
  addString(doc, object, "presentModeSource", env.presentModeSource);
  addString(doc, object, "windowMode", env.windowMode);
  yyjson_mut_obj_add_bool(doc, object, "windowVisible", env.windowVisible);
  yyjson_mut_obj_add_uint(doc, object, "renderGraphWorkerCount",
                          env.renderGraphWorkerCount);
  yyjson_mut_obj_add_bool(doc, object, "renderGraphParallelCompile",
                          env.renderGraphParallelCompile);
  yyjson_mut_obj_add_bool(doc, object, "renderGraphParallelRecording",
                          env.renderGraphParallelRecording);
  addString(doc, object, "renderGraphWorkerCountSource",
            env.renderGraphWorkerCountSource);
  addString(doc, object, "renderGraphParallelCompileSource",
            env.renderGraphParallelCompileSource);
  addString(doc, object, "renderGraphParallelRecordingSource",
            env.renderGraphParallelRecordingSource);
  addString(doc, object, "buildType", env.buildType);
  addString(doc, object, "cmakeToolProfile", env.cmakeToolProfile);
  addString(doc, object, "vcpkgManifestFeatures", env.vcpkgManifestFeatures);
  yyjson_mut_obj_add_bool(doc, object, "NURI_BUILD_SHARED", env.buildShared);
  yyjson_mut_obj_add_bool(doc, object, "NURI_WITH_LOGGING", env.loggingEnabled);
  yyjson_mut_obj_add_bool(doc, object, "NURI_WITH_ASSERTS", env.assertsEnabled);
  yyjson_mut_obj_add_bool(doc, object, "NURI_WITH_TRACY", env.tracyEnabled);
  yyjson_mut_obj_add_bool(doc, object, "tracyDiagnostic", env.tracyDiagnostic);
  yyjson_mut_obj_add_bool(doc, object, "devChecks", env.devChecks);
  return object;
}

yyjson_mut_val *makeCaseObject(yyjson_mut_doc *doc,
                               const BenchmarkCase &benchmarkCase) {
  const std::string settingsSignature =
      benchmarkCase.settingsSignature.empty()
          ? makeSettingsSignature(benchmarkCase.settings)
          : benchmarkCase.settingsSignature;
  const std::string configSignature =
      benchmarkCase.configSignature.empty()
          ? makeConfigSignature(benchmarkCase, settingsSignature)
          : benchmarkCase.configSignature;

  yyjson_mut_val *object = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_uint(doc, object, "schemaVersion",
                          benchmarkCase.schemaVersion);
  addString(doc, object, "id", benchmarkCase.id);
  addString(doc, object, "suite", benchmarkCase.suite);
  addString(doc, object, "comparisonGroup", benchmarkCase.comparisonGroup);
  addString(doc, object, "variant", benchmarkCase.variant);
  addString(doc, object, "description", benchmarkCase.description);
  addPath(doc, object, "manifestPath", benchmarkCase.manifestPath);
  addString(doc, object, "backend", benchmarkCase.backend);
  yyjson_mut_val *resolution = yyjson_mut_arr(doc);
  yyjson_mut_arr_add_uint(doc, resolution, benchmarkCase.resolution[0]);
  yyjson_mut_arr_add_uint(doc, resolution, benchmarkCase.resolution[1]);
  yyjson_mut_obj_add_val(doc, object, "resolution", resolution);
  addString(doc, object, "presentMode", benchmarkCase.presentMode);
  yyjson_mut_obj_add_bool(doc, object, "authoritative",
                          benchmarkCase.authoritative);
  yyjson_mut_obj_add_real(doc, object, "fixedDeltaSeconds",
                          benchmarkCase.fixedDeltaSeconds);
  yyjson_mut_obj_add_uint(doc, object, "warmupFrames",
                          benchmarkCase.warmupFrames);
  yyjson_mut_obj_add_uint(doc, object, "measurementFrames",
                          benchmarkCase.measurementFrames);
  yyjson_mut_obj_add_uint(doc, object, "cooldownFrames",
                          benchmarkCase.cooldownFrames);
  yyjson_mut_obj_add_uint(doc, object, "maxDrainFrames",
                          benchmarkCase.maxDrainFrames);
  yyjson_mut_obj_add_uint(doc, object, "drainTimeoutMs",
                          benchmarkCase.drainTimeoutMs);
  yyjson_mut_obj_add_uint(doc, object, "samples", benchmarkCase.samples);
  addString(doc, object, "settingsSignature", settingsSignature);
  addString(doc, object, "configSignature", configSignature);
  yyjson_mut_obj_add_val(doc, object, "requiredMetrics",
                         makeStringArray(doc, benchmarkCase.requiredMetrics));

  yyjson_mut_val *thresholds = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_real(doc, thresholds, "failPercent",
                          benchmarkCase.thresholds.failPercent);
  yyjson_mut_obj_add_real(doc, thresholds, "failAbsoluteMs",
                          benchmarkCase.thresholds.failAbsoluteMs);
  yyjson_mut_obj_add_real(doc, thresholds, "warnPercent",
                          benchmarkCase.thresholds.warnPercent);
  yyjson_mut_obj_add_real(doc, thresholds, "warnAbsoluteMs",
                          benchmarkCase.thresholds.warnAbsoluteMs);
  yyjson_mut_obj_add_val(doc, object, "thresholds", thresholds);

  yyjson_mut_val *scene = yyjson_mut_obj(doc);
  addString(doc, scene, "kind", benchmarkCase.scene.kind);
  addString(doc, scene, "pathBase", benchmarkCase.scene.pathBase);
  addPath(doc, scene, "path", benchmarkCase.scene.path);
  yyjson_mut_obj_add_bool(doc, scene, "flipUVs", benchmarkCase.scene.flipUVs);
  yyjson_mut_obj_add_bool(doc, scene, "generateMeshlets",
                          benchmarkCase.scene.generateMeshlets);
  yyjson_mut_obj_add_uint(doc, scene, "meshletMaxVertices",
                          benchmarkCase.scene.meshletMaxVertices);
  yyjson_mut_obj_add_uint(doc, scene, "meshletMaxPrimitives",
                          benchmarkCase.scene.meshletMaxPrimitives);
  yyjson_mut_obj_add_real(doc, scene, "meshletConeWeight",
                          benchmarkCase.scene.meshletConeWeight);
  addString(doc, scene, "baseModelKind", benchmarkCase.scene.baseModelKind);
  yyjson_mut_obj_add_real(doc, scene, "baseModelTargetRadius",
                          benchmarkCase.scene.baseModelTargetRadius);
  yyjson_mut_obj_add_real(doc, scene, "baseModelMinScale",
                          benchmarkCase.scene.baseModelMinScale);
  yyjson_mut_obj_add_real(doc, scene, "baseModelMaxScale",
                          benchmarkCase.scene.baseModelMaxScale);
  addString(doc, scene, "generator", benchmarkCase.scene.generator);
  yyjson_mut_obj_add_uint(doc, scene, "seed", benchmarkCase.scene.seed);
  addString(doc, scene, "contentHash", benchmarkCase.scene.contentHash);
  yyjson_mut_obj_add_val(doc, object, "scene", scene);

  yyjson_mut_val *renderGraph = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_uint(doc, renderGraph, "workerCount",
                          benchmarkCase.renderGraph.workerCount);
  yyjson_mut_obj_add_bool(doc, renderGraph, "parallelCompile",
                          benchmarkCase.renderGraph.parallelCompile);
  yyjson_mut_obj_add_bool(doc, renderGraph, "parallelRecording",
                          benchmarkCase.renderGraph.parallelRecording);
  yyjson_mut_obj_add_val(doc, object, "renderGraph", renderGraph);

  yyjson_mut_val *camera = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_val(doc, camera, "position",
                         makeVec3Array(doc, benchmarkCase.camera.position));
  yyjson_mut_obj_add_val(doc, camera, "direction",
                         makeVec3Array(doc, benchmarkCase.camera.direction));
  yyjson_mut_obj_add_val(doc, camera, "target",
                         makeVec3Array(doc, benchmarkCase.camera.target));
  yyjson_mut_obj_add_bool(doc, camera, "hasTarget",
                          benchmarkCase.camera.hasTarget);
  yyjson_mut_obj_add_real(doc, camera, "verticalFovDegrees",
                          benchmarkCase.camera.verticalFovDegrees);
  yyjson_mut_obj_add_real(doc, camera, "nearPlane",
                          benchmarkCase.camera.nearPlane);
  yyjson_mut_obj_add_real(doc, camera, "farPlane",
                          benchmarkCase.camera.farPlane);
  yyjson_mut_obj_add_val(doc, object, "camera", camera);
  yyjson_mut_obj_add_val(doc, object, "timeline",
                         makeTimelineObject(doc, benchmarkCase.timeline));

  yyjson_mut_val *requirements = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_val(
      doc, requirements, "assets",
      makeStringArray(doc, benchmarkCase.requirements.assets));
  yyjson_mut_obj_add_val(
      doc, requirements, "backends",
      makeStringArray(doc, benchmarkCase.requirements.backends));
  yyjson_mut_obj_add_bool(doc, requirements, "allowVisibleWindow",
                          benchmarkCase.requirements.allowVisibleWindow);
  yyjson_mut_obj_add_bool(doc, requirements, "msaa4x",
                          benchmarkCase.requirements.msaa4x);
  yyjson_mut_obj_add_val(doc, object, "requirements", requirements);
  yyjson_mut_obj_add_val(doc, object, "settings",
                         makeSettingsObject(doc, benchmarkCase.settings));
  return object;
}

yyjson_mut_val *makeFrameObject(yyjson_mut_doc *doc,
                                const BenchmarkFrameRecord &frame,
                                bool verboseMetrics) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_uint(doc, object, "frameIndex", frame.frameIndex);
  yyjson_mut_obj_add_uint(doc, object, "sampleIndex", frame.sampleIndex);
  yyjson_mut_obj_add_bool(doc, object, "measured", frame.measured);
  yyjson_mut_val *measurements = yyjson_mut_obj(doc);
  for (const BenchmarkMeasurement &measurement : frame.measurements) {
    const std::string_view metricId = measurement.id();
    yyjson_mut_obj_add(
        measurements, yyjson_mut_strncpy(doc, metricId.data(), metricId.size()),
        yyjson_mut_real(doc, measurement.second));
  }
  yyjson_mut_obj_add_val(doc, object, "measurements", measurements);
  if (verboseMetrics) {
    yyjson_mut_val *metrics = yyjson_mut_obj(doc);
    yyjson_mut_val *opaque = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_uint(doc, opaque, "totalInstances",
                            frame.metrics.opaque.totalInstances);
    yyjson_mut_obj_add_uint(doc, opaque, "visibleInstances",
                            frame.metrics.opaque.visibleInstances);
    yyjson_mut_obj_add_val(doc, metrics, "opaque", opaque);
    yyjson_mut_val *shadow = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_uint(doc, shadow, "totalDraws",
                            frame.metrics.shadow.totalDraws);
    yyjson_mut_obj_add_val(doc, metrics, "shadow", shadow);
    yyjson_mut_obj_add_val(doc, object, "metrics", metrics);
  }
  return object;
}

yyjson_mut_val *makeSampleStatsObject(yyjson_mut_doc *doc,
                                      const BenchmarkSampleStats &sample) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_uint(doc, object, "sampleIndex", sample.sampleIndex);
  if (sample.warmupStable.has_value()) {
    yyjson_mut_obj_add_bool(doc, object, "warmupStable", *sample.warmupStable);
  } else {
    yyjson_mut_obj_add_null(doc, object, "warmupStable");
  }
  yyjson_mut_obj_add_uint(doc, object, "measuredFrameStart",
                          sample.measuredFrameStart);
  yyjson_mut_obj_add_uint(doc, object, "measuredFrameCount",
                          sample.measuredFrameCount);
  yyjson_mut_obj_add_val(doc, object, "stats",
                         makeStatsMapObject(doc, sample.stats));
  yyjson_mut_obj_add_val(doc, object, "warnings",
                         makeStringArray(doc, sample.warnings));
  return object;
}

yyjson_mut_val *makeProfileObject(yyjson_mut_doc *doc,
                                  const BenchmarkProfileInfo &profile) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  addString(doc, object, "id", profile.id);
  yyjson_mut_obj_add_bool(doc, object, "profileAuthoritative",
                          profile.profileAuthoritative);
  yyjson_mut_obj_add_bool(doc, object, "authoritative", profile.authoritative);
  yyjson_mut_obj_add_uint(doc, object, "minimumRepetitions",
                          profile.minimumRepetitions);
  yyjson_mut_obj_add_uint(doc, object, "completedRepetitions",
                          profile.completedRepetitions);
  yyjson_mut_obj_add_bool(doc, object, "repetitionRequirementSatisfied",
                          profile.repetitionRequirementSatisfied);
  addString(doc, object, "repetitionUnit", profile.repetitionUnit);
  addString(doc, object, "warmupStabilityPolicy",
            profile.warmupStabilityPolicy);
  addString(doc, object, "warmupStabilityStatus",
            profile.warmupStabilityStatus);
  yyjson_mut_obj_add_uint(doc, object, "warmupWindowFrames",
                          profile.warmupWindowFrames);
  yyjson_mut_obj_add_real(doc, object, "warmupMaxDriftPercent",
                          profile.warmupMaxDriftPercent);
  yyjson_mut_obj_add_val(doc, object, "requiredMetrics",
                         makeStringArray(doc, profile.requiredMetrics));
  yyjson_mut_obj_add_val(doc, object, "authorityBlockers",
                         makeStringArray(doc, profile.authorityBlockers));
  return object;
}

yyjson_mut_val *
makeRepeatObservationsObject(yyjson_mut_doc *doc,
                             const BenchmarkRepeatObservationInfo &info) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  addString(doc, object, "unit", info.unit);
  yyjson_mut_obj_add_bool(doc, object, "independent", info.independent);
  yyjson_mut_obj_add_uint(doc, object, "count", info.count);
  return object;
}

yyjson_mut_val *makeTracyZoneObject(yyjson_mut_doc *doc,
                                    const BenchmarkTracyZoneStats &zone) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  addString(doc, object, "name", zone.name);
  addPath(doc, object, "sourceFile", zone.sourceFile);
  yyjson_mut_obj_add_uint(doc, object, "sourceLine", zone.sourceLine);
  yyjson_mut_obj_add_uint(doc, object, "totalNs", zone.totalNs);
  yyjson_mut_obj_add_real(doc, object, "totalPercent", zone.totalPercent);
  yyjson_mut_obj_add_uint(doc, object, "count", zone.count);
  yyjson_mut_obj_add_real(doc, object, "meanNs", zone.meanNs);
  yyjson_mut_obj_add_uint(doc, object, "minNs", zone.minNs);
  yyjson_mut_obj_add_uint(doc, object, "maxNs", zone.maxNs);
  yyjson_mut_obj_add_real(doc, object, "stddevNs", zone.stddevNs);
  return object;
}

yyjson_mut_val *
makeTracyZoneArray(yyjson_mut_doc *doc,
                   const std::vector<BenchmarkTracyZoneStats> &zones) {
  yyjson_mut_val *array = yyjson_mut_arr(doc);
  for (const BenchmarkTracyZoneStats &zone : zones) {
    yyjson_mut_arr_add_val(array, makeTracyZoneObject(doc, zone));
  }
  return array;
}

yyjson_mut_val *makeTracyFlameNodeObject(yyjson_mut_doc *doc,
                                         const BenchmarkTracyFlameNode &node) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  addString(doc, object, "name", node.name);
  addString(doc, object, "thread", node.thread);
  addPath(doc, object, "sourceFile", node.sourceFile);
  yyjson_mut_obj_add_uint(doc, object, "sourceLine", node.sourceLine);
  yyjson_mut_obj_add_uint(doc, object, "totalNs", node.totalNs);
  yyjson_mut_obj_add_uint(doc, object, "selfNs", node.selfNs);
  yyjson_mut_obj_add_uint(doc, object, "count", node.count);
  yyjson_mut_val *children = yyjson_mut_arr(doc);
  for (const BenchmarkTracyFlameNode &child : node.children) {
    yyjson_mut_arr_add_val(children, makeTracyFlameNodeObject(doc, child));
  }
  yyjson_mut_obj_add_val(doc, object, "children", children);
  return object;
}

yyjson_mut_val *
makeTracyFlameGraphObject(yyjson_mut_doc *doc,
                          const BenchmarkTracyFlameGraph &flameGraph) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  addPath(doc, object, "eventsCsvPath", flameGraph.eventsCsvPath);
  addString(doc, object, "eventsExportCommand", flameGraph.eventsExportCommand);
  yyjson_mut_obj_add_bool(doc, object, "frameScoped", flameGraph.frameScoped);
  yyjson_mut_obj_add_uint(doc, object, "eventCount", flameGraph.eventCount);
  yyjson_mut_obj_add_uint(doc, object, "retainedNodeCount",
                          flameGraph.retainedNodeCount);
  yyjson_mut_obj_add_uint(doc, object, "maxDepth", flameGraph.maxDepth);
  yyjson_mut_obj_add_val(doc, object, "root",
                         makeTracyFlameNodeObject(doc, flameGraph.root));
  return object;
}

yyjson_mut_val *makeTracyObject(yyjson_mut_doc *doc,
                                const BenchmarkTracyReport &tracy) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  yyjson_mut_obj_add_bool(doc, object, "available", tracy.available);
  addPath(doc, object, "tracePath", tracy.tracePath);
  addPath(doc, object, "captureLogPath", tracy.captureLogPath);
  addPath(doc, object, "zonesCsvPath", tracy.zonesCsvPath);
  addPath(doc, object, "selfZonesCsvPath", tracy.selfZonesCsvPath);
  addPath(doc, object, "exportLogPath", tracy.exportLogPath);
  addString(doc, object, "captureCommand", tracy.captureCommand);
  addString(doc, object, "zonesExportCommand", tracy.zonesExportCommand);
  addString(doc, object, "selfZonesExportCommand",
            tracy.selfZonesExportCommand);
  yyjson_mut_obj_add_uint(doc, object, "captureFrameCount",
                          tracy.captureFrameCount);
  yyjson_mut_obj_add_real(doc, object, "captureTimeSpanSeconds",
                          tracy.captureTimeSpanSeconds);
  yyjson_mut_obj_add_uint(doc, object, "captureZoneEventCount",
                          tracy.captureZoneEventCount);
  yyjson_mut_obj_add_val(doc, object, "zones",
                         makeTracyZoneArray(doc, tracy.zones));
  yyjson_mut_obj_add_val(doc, object, "selfZones",
                         makeTracyZoneArray(doc, tracy.selfZones));
  yyjson_mut_obj_add_val(doc, object, "flameGraph",
                         makeTracyFlameGraphObject(doc, tracy.flameGraph));
  return object;
}

[[nodiscard]] std::string readString(yyjson_val *object, const char *key,
                                     std::string defaultValue = {}) {
  yyjson_val *value = yyjson_obj_get(object, key);
  if (!yyjson_is_str(value)) {
    return defaultValue;
  }
  return std::string(yyjson_get_str(value), yyjson_get_len(value));
}

[[nodiscard]] bool readBool(yyjson_val *object, const char *key,
                            bool defaultValue = false) {
  yyjson_val *value = yyjson_obj_get(object, key);
  return yyjson_is_bool(value) ? yyjson_get_bool(value) : defaultValue;
}

[[nodiscard]] uint32_t readU32(yyjson_val *object, const char *key,
                               uint32_t defaultValue = 0u) {
  yyjson_val *value = yyjson_obj_get(object, key);
  return yyjson_is_uint(value) ? static_cast<uint32_t>(yyjson_get_uint(value))
                               : defaultValue;
}

[[nodiscard]] uint64_t readU64(yyjson_val *object, const char *key,
                               uint64_t defaultValue = 0u) {
  yyjson_val *value = yyjson_obj_get(object, key);
  return yyjson_is_uint(value) ? static_cast<uint64_t>(yyjson_get_uint(value))
                               : defaultValue;
}

[[nodiscard]] double readReal(yyjson_val *object, const char *key,
                              double defaultValue = 0.0) {
  yyjson_val *value = yyjson_obj_get(object, key);
  return yyjson_is_num(value) ? yyjson_get_num(value) : defaultValue;
}

[[nodiscard]] int32_t readS32(yyjson_val *object, const char *key,
                              int32_t defaultValue = 0) {
  yyjson_val *value = yyjson_obj_get(object, key);
  if (yyjson_is_sint(value)) {
    return static_cast<int32_t>(yyjson_get_sint(value));
  }
  if (yyjson_is_uint(value)) {
    return static_cast<int32_t>(yyjson_get_uint(value));
  }
  return defaultValue;
}

template <typename Enum>
[[nodiscard]] Enum
readEnumValue(yyjson_val *object, const char *key, Enum defaultValue,
              std::initializer_list<std::pair<std::string_view, Enum>> values) {
  const std::string text = readString(object, key);
  for (const auto &[name, value] : values) {
    if (text == name) {
      return value;
    }
  }
  return defaultValue;
}

[[nodiscard]] RenderSettings readSettingsObject(yyjson_val *object) {
  RenderSettings settings{};
  if (!yyjson_is_obj(object)) {
    return settings;
  }
  yyjson_val *opaque = yyjson_obj_get(object, "opaque");
  if (yyjson_is_obj(opaque)) {
    settings.opaque.enabled =
        readBool(opaque, "enabled", settings.opaque.enabled);
    settings.opaque.enableDepthPrepass = readBool(
        opaque, "enableDepthPrepass", settings.opaque.enableDepthPrepass);
    settings.opaque.enableDepthPyramid = readBool(
        opaque, "enableDepthPyramid", settings.opaque.enableDepthPyramid);
    settings.opaque.enableInstanceCompute = readBool(
        opaque, "enableInstanceCompute", settings.opaque.enableInstanceCompute);
    settings.opaque.enableIndirectDraw = readBool(
        opaque, "enableIndirectDraw", settings.opaque.enableIndirectDraw);
    settings.opaque.enableInstancedDraw = readBool(
        opaque, "enableInstancedDraw", settings.opaque.enableInstancedDraw);
    settings.opaque.enableMeshLod =
        readBool(opaque, "enableMeshLod", settings.opaque.enableMeshLod);
    settings.opaque.meshLodTargetPixelError =
        static_cast<float>(readReal(opaque, "meshLodTargetPixelError",
                                    settings.opaque.meshLodTargetPixelError));
    settings.opaque.meshLodHysteresisRatio =
        static_cast<float>(readReal(opaque, "meshLodHysteresisRatio",
                                    settings.opaque.meshLodHysteresisRatio));
    settings.opaque.enableCpuFrustumCulling =
        readBool(opaque, "enableCpuFrustumCulling",
                 settings.opaque.enableCpuFrustumCulling);
    settings.opaque.meshletMode =
        readEnumValue(opaque, "meshletMode", settings.opaque.meshletMode,
                      {{"Disabled", MeshletRenderMode::Disabled},
                       {"Opportunistic", MeshletRenderMode::Opportunistic},
                       {"Required", MeshletRenderMode::Required}});
    settings.opaque.hybridClassicMaxMeshlets =
        readU32(opaque, "hybridClassicMaxMeshlets",
                settings.opaque.hybridClassicMaxMeshlets);
    settings.opaque.enableMeshletFrustumCulling =
        readBool(opaque, "enableMeshletFrustumCulling",
                 settings.opaque.enableMeshletFrustumCulling);
    settings.opaque.enableMeshletConeCulling =
        readBool(opaque, "enableMeshletConeCulling",
                 settings.opaque.enableMeshletConeCulling);
    settings.opaque.enableInstanceAnimation =
        readBool(opaque, "enableInstanceAnimation",
                 settings.opaque.enableInstanceAnimation);
    settings.opaque.enableTessellation = readBool(
        opaque, "enableTessellation", settings.opaque.enableTessellation);
    settings.opaque.forcedMeshLod =
        readS32(opaque, "forcedMeshLod", settings.opaque.forcedMeshLod);
  }

  yyjson_val *antiAliasing = yyjson_obj_get(object, "antiAliasing");
  if (yyjson_is_obj(antiAliasing)) {
    settings.antiAliasing.mode =
        readEnumValue(antiAliasing, "mode", settings.antiAliasing.mode,
                      {{"None", AntiAliasingMode::None},
                       {"TAA", AntiAliasingMode::TAA},
                       {"SpatialFallback", AntiAliasingMode::SpatialFallback},
                       {"MSAA4x", AntiAliasingMode::MSAA4x}});
    settings.antiAliasing.temporalProvider =
        readEnumValue(antiAliasing, "temporalProvider",
                      settings.antiAliasing.temporalProvider,
                      {{"Legacy", TemporalReconstructionProvider::Legacy},
                       {"Reference", TemporalReconstructionProvider::Reference},
                       {"External", TemporalReconstructionProvider::External}});
    settings.antiAliasing.qualityPreset = readEnumValue(
        antiAliasing, "qualityPreset", settings.antiAliasing.qualityPreset,
        {{"Performance", TemporalAAQualityPreset::Performance},
         {"Balanced", TemporalAAQualityPreset::Balanced},
         {"Quality", TemporalAAQualityPreset::Quality},
         {"Ultra", TemporalAAQualityPreset::Ultra},
         {"Custom", TemporalAAQualityPreset::Custom}});
    settings.antiAliasing.debug.spatialPostMsaaCleanup =
        readBool(antiAliasing, "spatialPostMsaaCleanup",
                 settings.antiAliasing.debug.spatialPostMsaaCleanup);
  }

  yyjson_val *ambientOcclusion = yyjson_obj_get(object, "ambientOcclusion");
  if (yyjson_is_obj(ambientOcclusion)) {
    settings.ambientOcclusion.mode =
        readEnumValue(ambientOcclusion, "mode", settings.ambientOcclusion.mode,
                      {{"Disabled", AmbientOcclusionMode::Disabled},
                       {"GTAO", AmbientOcclusionMode::GTAO}});
    settings.ambientOcclusion.preset = readEnumValue(
        ambientOcclusion, "preset", settings.ambientOcclusion.preset,
        {{"Low", AmbientOcclusionPreset::Low},
         {"Balanced", AmbientOcclusionPreset::Balanced},
         {"High", AmbientOcclusionPreset::High},
         {"Ultra", AmbientOcclusionPreset::Ultra},
         {"Custom", AmbientOcclusionPreset::Custom}});
  }

  yyjson_val *shadow = yyjson_obj_get(object, "shadow");
  if (yyjson_is_obj(shadow)) {
    settings.shadow.enabled =
        readBool(shadow, "enabled", settings.shadow.enabled);
    settings.shadow.qualityPreset =
        readEnumValue(shadow, "qualityPreset", settings.shadow.qualityPreset,
                      {{"Custom", ShadowQualityPreset::Custom},
                       {"Low", ShadowQualityPreset::Low},
                       {"Medium", ShadowQualityPreset::Medium},
                       {"High", ShadowQualityPreset::High},
                       {"Ultra", ShadowQualityPreset::Ultra}});
  }

  yyjson_val *visibility = yyjson_obj_get(object, "visibility");
  if (yyjson_is_obj(visibility)) {
    static constexpr std::array cullingModes{
        std::pair{"Disabled", VisibilityCullingMode::Disabled},
        std::pair{"CpuCoarse", VisibilityCullingMode::CpuCoarse},
        std::pair{"Hybrid", VisibilityCullingMode::Hybrid},
        std::pair{"GpuDriven", VisibilityCullingMode::GpuDriven}};
    const auto readCullingMode = [&](const char *key,
                                     VisibilityCullingMode defaultValue) {
      const std::string text = readString(visibility, key);
      for (const auto &[name, value] : cullingModes) {
        if (text == name) {
          return value;
        }
      }
      return defaultValue;
    };
    settings.visibility.mainViewMode =
        readCullingMode("mainViewMode", settings.visibility.mainViewMode);
    settings.visibility.shadowMode =
        readCullingMode("shadowMode", settings.visibility.shadowMode);
    settings.visibility.occlusionMode = readEnumValue(
        visibility, "occlusionMode", settings.visibility.occlusionMode,
        {{"Disabled", VisibilityOcclusionMode::Disabled},
         {"PreviousFrameHiZ", VisibilityOcclusionMode::PreviousFrameHiZ},
         {"CurrentFrameHiZExperimental",
          VisibilityOcclusionMode::CurrentFrameHiZExperimental}});
    settings.visibility.enableMeshletFrustumCulling =
        readBool(visibility, "enableMeshletFrustumCulling",
                 settings.visibility.enableMeshletFrustumCulling);
    settings.visibility.enableMeshletConeCulling =
        readBool(visibility, "enableMeshletConeCulling",
                 settings.visibility.enableMeshletConeCulling);
    settings.visibility.enableGpuInstanceCulling =
        readBool(visibility, "enableGpuInstanceCulling",
                 settings.visibility.enableGpuInstanceCulling);
    settings.visibility.enableGpuIndirectDraw =
        readBool(visibility, "enableGpuIndirectDraw",
                 settings.visibility.enableGpuIndirectDraw);
    settings.visibility.enableIndirectMeshDispatch =
        readBool(visibility, "enableIndirectMeshDispatch",
                 settings.visibility.enableIndirectMeshDispatch);
    settings.visibility.enableMeshletPreTaskCompaction =
        readBool(visibility, "enableMeshletPreTaskCompaction",
                 settings.visibility.enableMeshletPreTaskCompaction);
    settings.visibility.visibleOnUncertain =
        readBool(visibility, "visibleOnUncertain",
                 settings.visibility.visibleOnUncertain);
  }

  yyjson_val *hdr = yyjson_obj_get(object, "hdrPostProcess");
  if (yyjson_is_obj(hdr)) {
    settings.hdrPostProcess.bloomEnabled =
        readBool(hdr, "bloomEnabled", settings.hdrPostProcess.bloomEnabled);
    settings.hdrPostProcess.adaptationEnabled = readBool(
        hdr, "adaptationEnabled", settings.hdrPostProcess.adaptationEnabled);
  }
  yyjson_val *transmission = yyjson_obj_get(object, "transmission");
  if (yyjson_is_obj(transmission)) {
    settings.transmission.enabled =
        readBool(transmission, "enabled", settings.transmission.enabled);
  }
  yyjson_val *transparent = yyjson_obj_get(object, "transparent");
  if (yyjson_is_obj(transparent)) {
    settings.transparent.enabled =
        readBool(transparent, "enabled", settings.transparent.enabled);
  }
  yyjson_val *textureFiltering = yyjson_obj_get(object, "textureFiltering");
  if (yyjson_is_obj(textureFiltering)) {
    settings.textureFiltering.mode =
        readEnumValue(textureFiltering, "mode", settings.textureFiltering.mode,
                      {{"Bilinear", TextureFilterMode::Bilinear},
                       {"Trilinear", TextureFilterMode::Trilinear},
                       {"Anisotropic", TextureFilterMode::Anisotropic}});
    settings.textureFiltering.anisotropy = static_cast<uint8_t>(readU32(
        textureFiltering, "anisotropy", settings.textureFiltering.anisotropy));
  }
  sanitizeBenchmarkRenderSettings(settings);
  return settings;
}

[[nodiscard]] glm::vec3 readVec3(yyjson_val *object, const char *key,
                                 glm::vec3 defaultValue = {}) {
  yyjson_val *value = yyjson_obj_get(object, key);
  if (!yyjson_is_arr(value) || yyjson_arr_size(value) != 3u) {
    return defaultValue;
  }
  glm::vec3 out{};
  for (uint32_t i = 0u; i < 3u; ++i) {
    yyjson_val *entry = yyjson_arr_get(value, i);
    if (!yyjson_is_num(entry)) {
      return defaultValue;
    }
    out[i] = static_cast<float>(yyjson_get_num(entry));
  }
  return out;
}

[[nodiscard]] std::vector<std::string> readStringArray(yyjson_val *array) {
  std::vector<std::string> values;
  if (!yyjson_is_arr(array)) {
    return values;
  }
  values.reserve(yyjson_arr_size(array));
  yyjson_arr_iter iter;
  yyjson_arr_iter_init(array, &iter);
  yyjson_val *entry = nullptr;
  while ((entry = yyjson_arr_iter_next(&iter)) != nullptr) {
    if (yyjson_is_str(entry)) {
      values.emplace_back(yyjson_get_str(entry), yyjson_get_len(entry));
    }
  }
  return values;
}

[[nodiscard]] std::vector<std::filesystem::path>
readPathArray(yyjson_val *array) {
  std::vector<std::filesystem::path> values;
  std::vector<std::string> strings = readStringArray(array);
  values.reserve(strings.size());
  for (std::string &value : strings) {
    values.emplace_back(std::move(value));
  }
  return values;
}

[[nodiscard]] BenchmarkTimeline readTimelineObject(yyjson_val *object) {
  BenchmarkTimeline timeline{};
  if (!yyjson_is_obj(object)) {
    return timeline;
  }
  yyjson_val *paths = yyjson_obj_get(object, "cameraPaths");
  if (!yyjson_is_arr(paths)) {
    return timeline;
  }
  timeline.cameraPaths.reserve(yyjson_arr_size(paths));
  yyjson_arr_iter pathIter;
  yyjson_arr_iter_init(paths, &pathIter);
  yyjson_val *pathValue = nullptr;
  while ((pathValue = yyjson_arr_iter_next(&pathIter)) != nullptr) {
    if (!yyjson_is_obj(pathValue)) {
      continue;
    }
    BenchmarkCameraPath path{};
    path.id = readString(pathValue, "id");
    path.startFrame = readU32(pathValue, "startFrame");
    path.endFrame = readU32(pathValue, "endFrame");
    path.interpolation =
        readString(pathValue, "interpolation", path.interpolation);
    yyjson_val *keyframes = yyjson_obj_get(pathValue, "keyframes");
    if (yyjson_is_arr(keyframes)) {
      path.keyframes.reserve(yyjson_arr_size(keyframes));
      yyjson_arr_iter keyframeIter;
      yyjson_arr_iter_init(keyframes, &keyframeIter);
      yyjson_val *keyframeValue = nullptr;
      while ((keyframeValue = yyjson_arr_iter_next(&keyframeIter)) != nullptr) {
        if (!yyjson_is_obj(keyframeValue)) {
          continue;
        }
        BenchmarkCameraKeyframe keyframe{};
        keyframe.frame = readU32(keyframeValue, "frame");
        keyframe.position = readVec3(keyframeValue, "position");
        keyframe.target = readVec3(keyframeValue, "target");
        keyframe.hasTarget = readBool(keyframeValue, "hasTarget");
        path.keyframes.push_back(keyframe);
      }
    }
    timeline.cameraPaths.push_back(std::move(path));
  }
  return timeline;
}

[[nodiscard]] MetricStats readStats(yyjson_val *object) {
  MetricStats stats{};
  if (!yyjson_is_obj(object)) {
    return stats;
  }
  stats.count = static_cast<size_t>(readU32(object, "count"));
  stats.min = readReal(object, "min");
  stats.median = readReal(object, "median");
  stats.p90 = readReal(object, "p90");
  stats.p95 = readReal(object, "p95");
  stats.p99 = yyjson_obj_get(object, "p99") != nullptr ? readReal(object, "p99")
                                                       : stats.p95;
  stats.max = readReal(object, "max");
  stats.mean = readReal(object, "mean");
  stats.stddev = readReal(object, "stddev");
  stats.mad = readReal(object, "mad");
  stats.iqr = readReal(object, "iqr");
  stats.coefficientOfVariation = readReal(object, "coefficientOfVariation");
  return stats;
}

[[nodiscard]] std::map<std::string, MetricStats>
readStatsMap(yyjson_val *object) {
  std::map<std::string, MetricStats> stats;
  if (!yyjson_is_obj(object)) {
    return stats;
  }
  yyjson_obj_iter iter;
  yyjson_obj_iter_init(object, &iter);
  yyjson_val *key = nullptr;
  while ((key = yyjson_obj_iter_next(&iter)) != nullptr) {
    stats.emplace(std::string(yyjson_get_str(key), yyjson_get_len(key)),
                  readStats(yyjson_obj_iter_get_val(key)));
  }
  return stats;
}

[[nodiscard]] BenchmarkFrameMeasurements readMeasurements(yyjson_val *object) {
  BenchmarkFrameMeasurements measurements;
  if (!yyjson_is_obj(object)) {
    return measurements;
  }
  yyjson_obj_iter iter;
  yyjson_obj_iter_init(object, &iter);
  yyjson_val *key = nullptr;
  while ((key = yyjson_obj_iter_next(&iter)) != nullptr) {
    yyjson_val *value = yyjson_obj_iter_get_val(key);
    if (yyjson_is_num(value)) {
      measurements.emplace(
          std::string(yyjson_get_str(key), yyjson_get_len(key)),
          yyjson_get_num(value));
    }
  }
  return measurements;
}

[[nodiscard]] BenchmarkFrameRecord readFrameRecord(yyjson_val *object) {
  BenchmarkFrameRecord frame{};
  if (!yyjson_is_obj(object)) {
    return frame;
  }
  frame.frameIndex = readU64(object, "frameIndex");
  frame.sampleIndex = readU32(object, "sampleIndex");
  frame.measured = readBool(object, "measured");
  frame.measurements = readMeasurements(yyjson_obj_get(object, "measurements"));
  yyjson_val *metrics = yyjson_obj_get(object, "metrics");
  if (yyjson_is_obj(metrics)) {
    yyjson_val *opaque = yyjson_obj_get(metrics, "opaque");
    if (yyjson_is_obj(opaque)) {
      frame.metrics.opaque.totalInstances = readU32(opaque, "totalInstances");
      frame.metrics.opaque.visibleInstances =
          readU32(opaque, "visibleInstances");
    }
    yyjson_val *shadow = yyjson_obj_get(metrics, "shadow");
    if (yyjson_is_obj(shadow)) {
      frame.metrics.shadow.totalDraws = readU32(shadow, "totalDraws");
    }
  }
  return frame;
}

[[nodiscard]] BenchmarkSampleStats readSampleStats(yyjson_val *object) {
  BenchmarkSampleStats sample{};
  if (!yyjson_is_obj(object)) {
    return sample;
  }
  sample.sampleIndex = readU32(object, "sampleIndex");
  yyjson_val *warmupStable = yyjson_obj_get(object, "warmupStable");
  if (yyjson_is_bool(warmupStable)) {
    sample.warmupStable = yyjson_get_bool(warmupStable);
  }
  sample.measuredFrameStart = readU64(object, "measuredFrameStart");
  sample.measuredFrameCount = readU32(object, "measuredFrameCount");
  sample.stats = readStatsMap(yyjson_obj_get(object, "stats"));
  sample.warnings = readStringArray(yyjson_obj_get(object, "warnings"));
  return sample;
}

[[nodiscard]] BenchmarkProfileInfo readProfileInfo(yyjson_val *object) {
  BenchmarkProfileInfo profile{};
  if (!yyjson_is_obj(object)) {
    return profile;
  }
  profile.id = readString(object, "id");
  profile.profileAuthoritative = readBool(object, "profileAuthoritative");
  profile.authoritative = readBool(object, "authoritative");
  profile.minimumRepetitions = readU32(object, "minimumRepetitions");
  profile.completedRepetitions = readU32(object, "completedRepetitions");
  profile.repetitionRequirementSatisfied =
      readBool(object, "repetitionRequirementSatisfied");
  profile.repetitionUnit =
      readString(object, "repetitionUnit", "not-collected");
  profile.warmupStabilityPolicy =
      readString(object, "warmupStabilityPolicy", "unknown");
  profile.warmupStabilityStatus =
      readString(object, "warmupStabilityStatus", "unknown");
  profile.warmupWindowFrames = readU32(object, "warmupWindowFrames");
  profile.warmupMaxDriftPercent = readReal(object, "warmupMaxDriftPercent");
  profile.requiredMetrics =
      readStringArray(yyjson_obj_get(object, "requiredMetrics"));
  profile.authorityBlockers =
      readStringArray(yyjson_obj_get(object, "authorityBlockers"));
  return profile;
}

[[nodiscard]] BenchmarkRepeatObservationInfo
readRepeatObservationsInfo(yyjson_val *object) {
  BenchmarkRepeatObservationInfo info{};
  if (!yyjson_is_obj(object)) {
    return info;
  }
  info.unit = readString(object, "unit", "in-process-sample-window");
  info.independent = readBool(object, "independent");
  info.count = readU32(object, "count");
  return info;
}

[[nodiscard]] BenchmarkTracyZoneStats readTracyZoneStats(yyjson_val *object) {
  BenchmarkTracyZoneStats zone{};
  if (!yyjson_is_obj(object)) {
    return zone;
  }
  zone.name = readString(object, "name");
  zone.sourceFile = std::filesystem::path(readString(object, "sourceFile"));
  zone.sourceLine = readU32(object, "sourceLine");
  zone.totalNs = readU64(object, "totalNs");
  zone.totalPercent = readReal(object, "totalPercent");
  zone.count = readU64(object, "count");
  zone.meanNs = readReal(object, "meanNs");
  zone.minNs = readU64(object, "minNs");
  zone.maxNs = readU64(object, "maxNs");
  zone.stddevNs = readReal(object, "stddevNs");
  return zone;
}

[[nodiscard]] std::vector<BenchmarkTracyZoneStats>
readTracyZoneArray(yyjson_val *array) {
  std::vector<BenchmarkTracyZoneStats> zones;
  if (!yyjson_is_arr(array)) {
    return zones;
  }
  zones.reserve(yyjson_arr_size(array));
  yyjson_arr_iter iter;
  yyjson_arr_iter_init(array, &iter);
  yyjson_val *entry = nullptr;
  while ((entry = yyjson_arr_iter_next(&iter)) != nullptr) {
    if (yyjson_is_obj(entry)) {
      zones.push_back(readTracyZoneStats(entry));
    }
  }
  return zones;
}

[[nodiscard]] BenchmarkTracyFlameNode readTracyFlameNode(yyjson_val *object) {
  BenchmarkTracyFlameNode node{};
  if (!yyjson_is_obj(object)) {
    return node;
  }
  node.name = readString(object, "name");
  node.thread = readString(object, "thread");
  node.sourceFile = std::filesystem::path(readString(object, "sourceFile"));
  node.sourceLine = readU32(object, "sourceLine");
  node.totalNs = readU64(object, "totalNs");
  node.selfNs = readU64(object, "selfNs");
  node.count = readU64(object, "count");
  yyjson_val *children = yyjson_obj_get(object, "children");
  if (yyjson_is_arr(children)) {
    node.children.reserve(yyjson_arr_size(children));
    yyjson_arr_iter iter;
    yyjson_arr_iter_init(children, &iter);
    yyjson_val *entry = nullptr;
    while ((entry = yyjson_arr_iter_next(&iter)) != nullptr) {
      if (yyjson_is_obj(entry)) {
        node.children.push_back(readTracyFlameNode(entry));
      }
    }
  }
  return node;
}

[[nodiscard]] BenchmarkTracyFlameGraph readTracyFlameGraph(yyjson_val *object) {
  BenchmarkTracyFlameGraph flameGraph{};
  if (!yyjson_is_obj(object)) {
    return flameGraph;
  }
  flameGraph.eventsCsvPath =
      std::filesystem::path(readString(object, "eventsCsvPath"));
  flameGraph.eventsExportCommand = readString(object, "eventsExportCommand");
  flameGraph.frameScoped = readBool(object, "frameScoped");
  flameGraph.eventCount = readU64(object, "eventCount");
  flameGraph.retainedNodeCount = readU64(object, "retainedNodeCount");
  flameGraph.maxDepth = readU32(object, "maxDepth");
  flameGraph.root = readTracyFlameNode(yyjson_obj_get(object, "root"));
  return flameGraph;
}

[[nodiscard]] BenchmarkTracyReport readTracyReport(yyjson_val *object) {
  BenchmarkTracyReport tracy{};
  if (!yyjson_is_obj(object)) {
    return tracy;
  }
  tracy.available = readBool(object, "available");
  tracy.tracePath = std::filesystem::path(readString(object, "tracePath"));
  tracy.captureLogPath =
      std::filesystem::path(readString(object, "captureLogPath"));
  tracy.zonesCsvPath =
      std::filesystem::path(readString(object, "zonesCsvPath"));
  tracy.selfZonesCsvPath =
      std::filesystem::path(readString(object, "selfZonesCsvPath"));
  tracy.exportLogPath =
      std::filesystem::path(readString(object, "exportLogPath"));
  tracy.captureCommand = readString(object, "captureCommand");
  tracy.zonesExportCommand = readString(object, "zonesExportCommand");
  tracy.selfZonesExportCommand = readString(object, "selfZonesExportCommand");
  tracy.captureFrameCount = readU64(object, "captureFrameCount");
  tracy.captureTimeSpanSeconds = readReal(object, "captureTimeSpanSeconds");
  tracy.captureZoneEventCount = readU64(object, "captureZoneEventCount");
  tracy.zones = readTracyZoneArray(yyjson_obj_get(object, "zones"));
  tracy.selfZones = readTracyZoneArray(yyjson_obj_get(object, "selfZones"));
  tracy.flameGraph = readTracyFlameGraph(yyjson_obj_get(object, "flameGraph"));
  return tracy;
}

} // namespace

void computeBenchmarkReportStats(BenchmarkReport &report) {
  report.stats.clear();
  report.unregisteredObservedMetrics.clear();
  std::map<std::string, std::vector<double>> valuesByMetric;
  std::set<std::string> unregistered;
  for (const BenchmarkFrameRecord &frame : report.frames) {
    if (!frame.measured) {
      continue;
    }
    for (const BenchmarkMeasurement &measurement : frame.measurements) {
      const std::string metricId(measurement.id());
      valuesByMetric[metricId].push_back(measurement.second);
      if (findBenchmarkMetricDescriptor(metricId) == nullptr) {
        unregistered.insert(metricId);
      }
    }
  }
  report.unregisteredObservedMetrics.assign(unregistered.begin(),
                                            unregistered.end());
  for (auto &[metricId, values] : valuesByMetric) {
    auto stats = computeMetricStats(std::move(values));
    if (!stats.hasError()) {
      report.stats.emplace(metricId, stats.value());
    }
  }

  report.sampleStats.clear();
  report.repeatObservations.unit = "in-process-sample-window";
  report.repeatObservations.independent = false;
  report.repeatObservations.count = 0u;
  const uint32_t samples = report.run.samples;
  bool allWarmupStable = samples > 0u;
  bool anyWarmupUnstable = false;
  for (uint32_t sampleIndex = 0u; sampleIndex < samples; ++sampleIndex) {
    BenchmarkSampleStats sample{};
    sample.sampleIndex = sampleIndex;
    sample.measuredFrameCount = 0u;
    sample.measuredFrameStart =
        static_cast<uint64_t>(sampleIndex) *
            (report.run.warmupFrames + report.run.measurementFrames +
             report.run.cooldownFrames) +
        report.run.warmupFrames;
    std::map<std::string, std::vector<double>> sampleValues;
    std::vector<double> warmupRenderSubmitMs;
    warmupRenderSubmitMs.reserve(report.run.warmupFrames);
    bool measurementPhaseReached = false;
    for (const BenchmarkFrameRecord &frame : report.frames) {
      if (frame.sampleIndex != sampleIndex) {
        continue;
      }
      if (!frame.measured && !measurementPhaseReached &&
          warmupRenderSubmitMs.size() < report.run.warmupFrames) {
        const auto timing = frame.measurements.find("cpu.render_submit_ms");
        if (timing != frame.measurements.end()) {
          warmupRenderSubmitMs.push_back(timing->second);
        }
        continue;
      }
      if (!frame.measured) {
        continue;
      }
      measurementPhaseReached = true;
      if (sample.measuredFrameCount == 0u) {
        sample.measuredFrameStart = frame.frameIndex;
      }
      ++sample.measuredFrameCount;
      for (const BenchmarkMeasurement &measurement : frame.measurements) {
        sampleValues[std::string(measurement.id())].push_back(
            measurement.second);
      }
    }
    for (auto &[metricId, values] : sampleValues) {
      auto stats = computeMetricStats(std::move(values));
      if (!stats.hasError()) {
        sample.stats.emplace(metricId, stats.value());
      }
    }
    if (!sample.stats.empty()) {
      ++report.repeatObservations.count;
    }

    const uint32_t window = report.profile.warmupWindowFrames;
    if (window > 0u && report.profile.warmupMaxDriftPercent > 0.0) {
      if (warmupRenderSubmitMs.size() < static_cast<size_t>(window) * 2u) {
        sample.warnings.push_back(
            "warmup stability requires two non-overlapping windows of " +
            std::to_string(window) + " frames");
      } else {
        std::vector<double> firstWindow(warmupRenderSubmitMs.begin(),
                                        warmupRenderSubmitMs.begin() + window);
        std::vector<double> lastWindow(warmupRenderSubmitMs.end() - window,
                                       warmupRenderSubmitMs.end());
        auto firstStats = computeMetricStats(std::move(firstWindow));
        auto lastStats = computeMetricStats(std::move(lastWindow));
        if (!firstStats.hasError() && !lastStats.hasError()) {
          const double firstMedian = firstStats.value().median;
          const double lastMedian = lastStats.value().median;
          const double denominator = std::abs(firstMedian);
          const double driftPercent =
              denominator > 1.0e-12
                  ? std::abs(lastMedian - firstMedian) / denominator * 100.0
                  : (std::abs(lastMedian) <= 1.0e-12
                         ? 0.0
                         : std::numeric_limits<double>::infinity());
          sample.warmupStable =
              std::isfinite(driftPercent) &&
              driftPercent <= report.profile.warmupMaxDriftPercent;
          if (!*sample.warmupStable) {
            sample.warnings.push_back(
                "warmup median drift exceeded profile limit: " +
                std::to_string(driftPercent) + "% > " +
                std::to_string(report.profile.warmupMaxDriftPercent) + "%");
          }
        }
      }
    }
    if (!sample.warmupStable.has_value()) {
      allWarmupStable = false;
    } else if (!*sample.warmupStable) {
      allWarmupStable = false;
      anyWarmupUnstable = true;
    }
    report.sampleStats.push_back(std::move(sample));
  }

  report.profile.warmupStabilityStatus =
      allWarmupStable ? "stable" : (anyWarmupUnstable ? "unstable" : "unknown");

  std::set<std::string> unavailable(report.unavailableMetrics.begin(),
                                    report.unavailableMetrics.end());
  std::set<std::string> required(report.benchmarkCase.requiredMetrics.begin(),
                                 report.benchmarkCase.requiredMetrics.end());
  required.insert(report.profile.requiredMetrics.begin(),
                  report.profile.requiredMetrics.end());
  for (const std::string &metricId : required) {
    if (report.stats.find(metricId) == report.stats.end()) {
      unavailable.insert(metricId);
      report.run.validForComparison = false;
    }
  }
  report.unavailableMetrics.assign(unavailable.begin(), unavailable.end());
}

Result<std::string, std::string>
writeBenchmarkReportJson(const BenchmarkReport &report, bool verboseFrames) {
  JsonMutDocPtr doc(yyjson_mut_doc_new(nullptr), &yyjson_mut_doc_free);
  if (!doc) {
    return Result<std::string, std::string>::makeError(
        "writeBenchmarkReportJson: failed to allocate JSON document");
  }
  yyjson_mut_val *root = yyjson_mut_obj(doc.get());
  yyjson_mut_doc_set_root(doc.get(), root);
  yyjson_mut_obj_add_uint(doc.get(), root, "schemaVersion",
                          report.schemaVersion);
  addString(doc.get(), root, "kind", report.kind);
  addString(doc.get(), root, "generatedAtUtc", report.generatedAtUtc);
  addString(doc.get(), root, "command", report.command);
  yyjson_mut_obj_add_val(doc.get(), root, "environment",
                         makeEnvironmentObject(doc.get(), report.environment));
  yyjson_mut_obj_add_val(doc.get(), root, "case",
                         makeCaseObject(doc.get(), report.benchmarkCase));
  yyjson_mut_obj_add_val(doc.get(), root, "profile",
                         makeProfileObject(doc.get(), report.profile));
  yyjson_mut_obj_add_val(
      doc.get(), root, "repeatObservations",
      makeRepeatObservationsObject(doc.get(), report.repeatObservations));

  yyjson_mut_val *run = yyjson_mut_obj(doc.get());
  yyjson_mut_obj_add_uint(doc.get(), run, "samples", report.run.samples);
  yyjson_mut_obj_add_uint(doc.get(), run, "warmupFrames",
                          report.run.warmupFrames);
  yyjson_mut_obj_add_uint(doc.get(), run, "measurementFrames",
                          report.run.measurementFrames);
  yyjson_mut_obj_add_uint(doc.get(), run, "cooldownFrames",
                          report.run.cooldownFrames);
  yyjson_mut_obj_add_uint(doc.get(), run, "maxDrainFrames",
                          report.run.maxDrainFrames);
  yyjson_mut_obj_add_uint(doc.get(), run, "drainTimeoutMs",
                          report.run.drainTimeoutMs);
  yyjson_mut_obj_add_bool(doc.get(), run, "validForComparison",
                          report.run.validForComparison);
  yyjson_mut_obj_add_real(doc.get(), run, "fixedDeltaSeconds",
                          report.run.fixedDeltaSeconds);
  yyjson_mut_obj_add_val(doc.get(), root, "run", run);

  yyjson_mut_val *artifacts = yyjson_mut_obj(doc.get());
  addPath(doc.get(), artifacts, "artifactDir", report.artifacts.artifactDir);
  yyjson_mut_obj_add_val(
      doc.get(), artifacts, "caseReports",
      makePathArray(doc.get(), report.artifacts.caseReports));
  yyjson_mut_obj_add_val(
      doc.get(), artifacts, "tracy",
      makePathArray(doc.get(), report.artifacts.tracyArtifacts));
  yyjson_mut_obj_add_val(doc.get(), root, "artifacts", artifacts);
  yyjson_mut_obj_add_val(doc.get(), root, "tracy",
                         makeTracyObject(doc.get(), report.tracy));

  yyjson_mut_val *frames = yyjson_mut_arr(doc.get());
  for (const BenchmarkFrameRecord &frame : report.frames) {
    yyjson_mut_arr_add_val(frames,
                           makeFrameObject(doc.get(), frame, verboseFrames));
  }
  yyjson_mut_obj_add_val(doc.get(), root, "frames", frames);

  yyjson_mut_val *samples = yyjson_mut_arr(doc.get());
  for (const BenchmarkSampleStats &sample : report.sampleStats) {
    yyjson_mut_arr_add_val(samples, makeSampleStatsObject(doc.get(), sample));
  }
  yyjson_mut_obj_add_val(doc.get(), root, "sampleStats", samples);
  yyjson_mut_obj_add_val(doc.get(), root, "stats",
                         makeStatsMapObject(doc.get(), report.stats));

  yyjson_mut_obj_add_val(doc.get(), root, "renderGraph",
                         yyjson_mut_obj(doc.get()));
  yyjson_mut_obj_add_val(doc.get(), root, "resourceStats",
                         yyjson_mut_obj(doc.get()));

  yyjson_mut_val *drain = yyjson_mut_obj(doc.get());
  yyjson_mut_obj_add_bool(doc.get(), drain, "drainComplete",
                          report.timingDrain.drainComplete);
  yyjson_mut_obj_add_uint(doc.get(), drain, "drainFrames",
                          report.timingDrain.drainFrames);
  yyjson_mut_obj_add_uint(doc.get(), drain, "drainTimeoutMs",
                          report.timingDrain.drainTimeoutMs);
  yyjson_mut_obj_add_uint(doc.get(), drain, "missingGpuTimingFrames",
                          report.timingDrain.missingGpuTimingFrames);
  yyjson_mut_obj_add_uint(doc.get(), drain, "scopeContainmentViolations",
                          report.timingDrain.scopeContainmentViolations);
  yyjson_mut_obj_add_uint(doc.get(), drain, "droppedGpuTimingReports",
                          report.timingDrain.droppedGpuTimingReports);
  yyjson_mut_obj_add_val(doc.get(), root, "timingDrain", drain);
  yyjson_mut_obj_add_val(doc.get(), root, "unavailableMetrics",
                         makeStringArray(doc.get(), report.unavailableMetrics));
  yyjson_mut_obj_add_val(
      doc.get(), root, "unregisteredObservedMetrics",
      makeStringArray(doc.get(), report.unregisteredObservedMetrics));
  yyjson_mut_obj_add_val(doc.get(), root, "warnings",
                         makeStringArray(doc.get(), report.warnings));

  size_t length = 0u;
  char *json = yyjson_mut_write_opts(doc.get(), YYJSON_WRITE_PRETTY, nullptr,
                                     &length, nullptr);
  if (json == nullptr) {
    return Result<std::string, std::string>::makeError(
        "writeBenchmarkReportJson: failed to serialize JSON");
  }
  std::string out(json, length);
  std::free(json);
  return Result<std::string, std::string>::makeResult(std::move(out));
}

Result<bool, std::string>
writeBenchmarkReportFile(const BenchmarkReport &report,
                         const std::filesystem::path &path,
                         bool verboseFrames) {
  auto json = writeBenchmarkReportJson(report, verboseFrames);
  if (json.hasError()) {
    return Result<bool, std::string>::makeError(json.error());
  }
  const auto written =
      nuri::tools::core::atomicWriteTextFile(path, json.value());
  if (written.hasError()) {
    return Result<bool, std::string>::makeError("writeBenchmarkReportFile: " +
                                                written.error());
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<BenchmarkReport, std::string>
readBenchmarkReportFile(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return Result<BenchmarkReport, std::string>::makeError(
        "readBenchmarkReportFile: failed to open " + path.string());
  }
  std::string json((std::istreambuf_iterator<char>(file)),
                   std::istreambuf_iterator<char>());
  yyjson_read_err error{};
  JsonDocPtr doc(yyjson_read_opts(json.data(), json.size(), 0, nullptr, &error),
                 &yyjson_doc_free);
  if (!doc) {
    return Result<BenchmarkReport, std::string>::makeError(
        "readBenchmarkReportFile: JSON parse failed at byte " +
        std::to_string(error.pos) + ": " + error.msg);
  }
  yyjson_val *root = yyjson_doc_get_root(doc.get());
  if (!yyjson_is_obj(root)) {
    return Result<BenchmarkReport, std::string>::makeError(
        "readBenchmarkReportFile: root must be an object");
  }
  yyjson_val *schemaVersion = yyjson_obj_get(root, "schemaVersion");
  if (!yyjson_is_uint(schemaVersion)) {
    return Result<BenchmarkReport, std::string>::makeError(
        "readBenchmarkReportFile: schemaVersion is required and must be an "
        "unsigned integer");
  }
  if (yyjson_get_uint(schemaVersion) == 2u) {
    auto envelope = nuri::tools::core::readResultEnvelopeV2(json);
    if (envelope.hasError()) {
      return Result<BenchmarkReport, std::string>::makeError(
          "readBenchmarkReportFile: invalid v2 result envelope: " +
          envelope.error());
    }
    if (envelope.value().tool != nuri::tools::core::ResultToolV2::Benchmark) {
      return Result<BenchmarkReport, std::string>::makeError(
          "readBenchmarkReportFile: v2 result envelope is not a benchmark "
          "result");
    }
    json = envelope.value().payloadJson;
    yyjson_read_err payloadError{};
    JsonDocPtr payloadDoc(
        yyjson_read_opts(json.data(), json.size(), 0, nullptr, &payloadError),
        &yyjson_doc_free);
    if (!payloadDoc) {
      return Result<BenchmarkReport, std::string>::makeError(
          "readBenchmarkReportFile: benchmark payload JSON parse failed at "
          "byte " +
          std::to_string(payloadError.pos) + ": " + payloadError.msg);
    }
    doc = std::move(payloadDoc);
    root = yyjson_doc_get_root(doc.get());
    schemaVersion = yyjson_obj_get(root, "schemaVersion");
    if (!yyjson_is_uint(schemaVersion)) {
      return Result<BenchmarkReport, std::string>::makeError(
          "readBenchmarkReportFile: benchmark payload schemaVersion is "
          "required and must be an unsigned integer");
    }
  }
  if (yyjson_get_uint(schemaVersion) != 1u) {
    return Result<BenchmarkReport, std::string>::makeError(
        "readBenchmarkReportFile: unsupported schemaVersion " +
        std::to_string(yyjson_get_uint(schemaVersion)));
  }
  yyjson_val *kind = yyjson_obj_get(root, "kind");
  if (!yyjson_is_str(kind)) {
    return Result<BenchmarkReport, std::string>::makeError(
        "readBenchmarkReportFile: kind is required and must be a string");
  }
  const std::string_view reportKind(yyjson_get_str(kind), yyjson_get_len(kind));
  if (reportKind != "nuri.benchmark.report") {
    return Result<BenchmarkReport, std::string>::makeError(
        "readBenchmarkReportFile: unsupported kind '" +
        std::string(reportKind) + "'");
  }
  auto contract = validateBenchmarkReportV1(root);
  if (contract.hasError()) {
    return Result<BenchmarkReport, std::string>::makeError(
        "readBenchmarkReportFile: invalid v1 report: " + contract.error());
  }
  BenchmarkReport report{};
  report.schemaVersion = 1u;
  report.kind = std::string(reportKind);
  report.generatedAtUtc = readString(root, "generatedAtUtc");
  report.command = readString(root, "command");
  report.profile = readProfileInfo(yyjson_obj_get(root, "profile"));
  report.repeatObservations =
      readRepeatObservationsInfo(yyjson_obj_get(root, "repeatObservations"));

  yyjson_val *caseObject = yyjson_obj_get(root, "case");
  if (yyjson_is_obj(caseObject)) {
    report.benchmarkCase.schemaVersion =
        readU32(caseObject, "schemaVersion", 1u);
    report.benchmarkCase.id = readString(caseObject, "id");
    report.benchmarkCase.suite = readString(caseObject, "suite");
    report.benchmarkCase.comparisonGroup =
        readString(caseObject, "comparisonGroup");
    report.benchmarkCase.variant = readString(caseObject, "variant");
    report.benchmarkCase.description = readString(caseObject, "description");
    report.benchmarkCase.backend = readString(caseObject, "backend", "default");
    report.benchmarkCase.manifestPath =
        std::filesystem::path(readString(caseObject, "manifestPath"));
    yyjson_val *resolution = yyjson_obj_get(caseObject, "resolution");
    if (yyjson_is_arr(resolution) && yyjson_arr_size(resolution) == 2u) {
      report.benchmarkCase.resolution[0] =
          static_cast<uint32_t>(yyjson_get_uint(yyjson_arr_get(resolution, 0)));
      report.benchmarkCase.resolution[1] =
          static_cast<uint32_t>(yyjson_get_uint(yyjson_arr_get(resolution, 1)));
    }
    report.benchmarkCase.presentMode =
        readString(caseObject, "presentMode", report.benchmarkCase.presentMode);
    report.benchmarkCase.authoritative = readBool(
        caseObject, "authoritative", report.benchmarkCase.authoritative);
    report.benchmarkCase.fixedDeltaSeconds =
        readReal(caseObject, "fixedDeltaSeconds",
                 report.benchmarkCase.fixedDeltaSeconds);
    report.benchmarkCase.warmupFrames =
        readU32(caseObject, "warmupFrames", report.benchmarkCase.warmupFrames);
    report.benchmarkCase.measurementFrames =
        readU32(caseObject, "measurementFrames",
                report.benchmarkCase.measurementFrames);
    report.benchmarkCase.cooldownFrames = readU32(
        caseObject, "cooldownFrames", report.benchmarkCase.cooldownFrames);
    report.benchmarkCase.maxDrainFrames = readU32(
        caseObject, "maxDrainFrames", report.benchmarkCase.maxDrainFrames);
    report.benchmarkCase.drainTimeoutMs = readU32(
        caseObject, "drainTimeoutMs", report.benchmarkCase.drainTimeoutMs);
    report.benchmarkCase.samples =
        readU32(caseObject, "samples", report.benchmarkCase.samples);
    report.benchmarkCase.settingsSignature =
        readString(caseObject, "settingsSignature");
    report.benchmarkCase.configSignature =
        readString(caseObject, "configSignature");

    yyjson_val *required = yyjson_obj_get(caseObject, "requiredMetrics");
    report.benchmarkCase.requiredMetrics = readStringArray(required);
    yyjson_val *thresholds = yyjson_obj_get(caseObject, "thresholds");
    if (yyjson_is_obj(thresholds)) {
      report.benchmarkCase.thresholds.failPercent =
          readReal(thresholds, "failPercent", 10.0);
      report.benchmarkCase.thresholds.failAbsoluteMs =
          readReal(thresholds, "failAbsoluteMs", 0.2);
      report.benchmarkCase.thresholds.warnPercent =
          readReal(thresholds, "warnPercent", 5.0);
      report.benchmarkCase.thresholds.warnAbsoluteMs =
          readReal(thresholds, "warnAbsoluteMs", 0.1);
    }
    yyjson_val *scene = yyjson_obj_get(caseObject, "scene");
    if (yyjson_is_obj(scene)) {
      report.benchmarkCase.scene.kind =
          readString(scene, "kind", report.benchmarkCase.scene.kind);
      report.benchmarkCase.scene.pathBase =
          readString(scene, "pathBase", report.benchmarkCase.scene.pathBase);
      report.benchmarkCase.scene.path =
          std::filesystem::path(readString(scene, "path"));
      report.benchmarkCase.scene.flipUVs =
          readBool(scene, "flipUVs", report.benchmarkCase.scene.flipUVs);
      report.benchmarkCase.scene.generateMeshlets =
          readBool(scene, "generateMeshlets",
                   report.benchmarkCase.scene.generateMeshlets);
      report.benchmarkCase.scene.meshletMaxVertices =
          readU32(scene, "meshletMaxVertices",
                  report.benchmarkCase.scene.meshletMaxVertices);
      report.benchmarkCase.scene.meshletMaxPrimitives =
          readU32(scene, "meshletMaxPrimitives",
                  report.benchmarkCase.scene.meshletMaxPrimitives);
      report.benchmarkCase.scene.meshletConeWeight = static_cast<float>(
          readReal(scene, "meshletConeWeight",
                   report.benchmarkCase.scene.meshletConeWeight));
      report.benchmarkCase.scene.baseModelKind = readString(
          scene, "baseModelKind", report.benchmarkCase.scene.baseModelKind);
      report.benchmarkCase.scene.baseModelTargetRadius =
          readReal(scene, "baseModelTargetRadius",
                   report.benchmarkCase.scene.baseModelTargetRadius);
      report.benchmarkCase.scene.baseModelMinScale =
          readReal(scene, "baseModelMinScale",
                   report.benchmarkCase.scene.baseModelMinScale);
      report.benchmarkCase.scene.baseModelMaxScale =
          readReal(scene, "baseModelMaxScale",
                   report.benchmarkCase.scene.baseModelMaxScale);
      report.benchmarkCase.scene.generator =
          readString(scene, "generator", report.benchmarkCase.scene.generator);
      report.benchmarkCase.scene.seed =
          readU32(scene, "seed", report.benchmarkCase.scene.seed);
      report.benchmarkCase.scene.contentHash = readString(
          scene, "contentHash", report.benchmarkCase.scene.contentHash);
    }
    yyjson_val *renderGraph = yyjson_obj_get(caseObject, "renderGraph");
    if (yyjson_is_obj(renderGraph)) {
      report.benchmarkCase.renderGraph.workerCount =
          readU32(renderGraph, "workerCount",
                  report.benchmarkCase.renderGraph.workerCount);
      report.benchmarkCase.renderGraph.parallelCompile =
          readBool(renderGraph, "parallelCompile",
                   report.benchmarkCase.renderGraph.parallelCompile);
      report.benchmarkCase.renderGraph.parallelRecording =
          readBool(renderGraph, "parallelRecording",
                   report.benchmarkCase.renderGraph.parallelRecording);
    }
    yyjson_val *camera = yyjson_obj_get(caseObject, "camera");
    if (yyjson_is_obj(camera)) {
      report.benchmarkCase.camera.position =
          readVec3(camera, "position", report.benchmarkCase.camera.position);
      report.benchmarkCase.camera.direction =
          readVec3(camera, "direction", report.benchmarkCase.camera.direction);
      report.benchmarkCase.camera.target =
          readVec3(camera, "target", report.benchmarkCase.camera.target);
      report.benchmarkCase.camera.hasTarget =
          readBool(camera, "hasTarget", report.benchmarkCase.camera.hasTarget);
      report.benchmarkCase.camera.verticalFovDegrees = static_cast<float>(
          readReal(camera, "verticalFovDegrees",
                   report.benchmarkCase.camera.verticalFovDegrees));
      report.benchmarkCase.camera.nearPlane = static_cast<float>(
          readReal(camera, "nearPlane", report.benchmarkCase.camera.nearPlane));
      report.benchmarkCase.camera.farPlane = static_cast<float>(
          readReal(camera, "farPlane", report.benchmarkCase.camera.farPlane));
    }
    report.benchmarkCase.timeline =
        readTimelineObject(yyjson_obj_get(caseObject, "timeline"));
    yyjson_val *requirements = yyjson_obj_get(caseObject, "requirements");
    if (yyjson_is_obj(requirements)) {
      report.benchmarkCase.requirements.assets =
          readStringArray(yyjson_obj_get(requirements, "assets"));
      report.benchmarkCase.requirements.backends =
          readStringArray(yyjson_obj_get(requirements, "backends"));
      report.benchmarkCase.requirements.allowVisibleWindow =
          readBool(requirements, "allowVisibleWindow",
                   report.benchmarkCase.requirements.allowVisibleWindow);
      report.benchmarkCase.requirements.msaa4x = readBool(
          requirements, "msaa4x", report.benchmarkCase.requirements.msaa4x);
    }
    report.benchmarkCase.settings =
        readSettingsObject(yyjson_obj_get(caseObject, "settings"));
  }

  yyjson_val *environment = yyjson_obj_get(root, "environment");
  if (yyjson_is_obj(environment)) {
    report.environment.repoRoot =
        std::filesystem::path(readString(environment, "repoRoot"));
    report.environment.commitHash = readString(environment, "commitHash");
    report.environment.branchName = readString(environment, "branchName");
    report.environment.dirty = readBool(environment, "dirty");
    report.environment.osName = readString(environment, "osName");
    report.environment.osVersion = readString(environment, "osVersion");
    report.environment.cpuName = readString(environment, "cpuName");
    report.environment.cpuLogicalThreadCount =
        readU32(environment, "cpuLogicalThreadCount");
    report.environment.gpuBackend = readString(environment, "gpuBackend");
    report.environment.gpuBackendSource =
        readString(environment, "gpuBackendSource");
    report.environment.gpuDeviceName = readString(environment, "gpuDeviceName");
    report.environment.gpuVendorId = readU32(environment, "gpuVendorId");
    report.environment.gpuDeviceId = readU32(environment, "gpuDeviceId");
    report.environment.gpuDriverVersion =
        readString(environment, "gpuDriverVersion", "unknown");
    report.environment.swapchainImageCount =
        readU32(environment, "swapchainImageCount");
    report.environment.requestedPresentMode =
        readString(environment, "requestedPresentMode");
    report.environment.resolvedPresentMode =
        readString(environment, "resolvedPresentMode");
    report.environment.presentModeSource =
        readString(environment, "presentModeSource");
    report.environment.windowMode = readString(environment, "windowMode");
    report.environment.windowVisible =
        readBool(environment, "windowVisible", true);
    report.environment.renderGraphWorkerCount =
        readU32(environment, "renderGraphWorkerCount", 1u);
    report.environment.renderGraphParallelCompile =
        readBool(environment, "renderGraphParallelCompile");
    report.environment.renderGraphParallelRecording =
        readBool(environment, "renderGraphParallelRecording");
    report.environment.renderGraphWorkerCountSource =
        readString(environment, "renderGraphWorkerCountSource");
    report.environment.renderGraphParallelCompileSource =
        readString(environment, "renderGraphParallelCompileSource");
    report.environment.renderGraphParallelRecordingSource =
        readString(environment, "renderGraphParallelRecordingSource");
    report.environment.buildType = readString(environment, "buildType");
    report.environment.cmakeToolProfile =
        readString(environment, "cmakeToolProfile");
    report.environment.vcpkgManifestFeatures =
        readString(environment, "vcpkgManifestFeatures");
    report.environment.buildShared = readBool(environment, "NURI_BUILD_SHARED");
    report.environment.loggingEnabled =
        readBool(environment, "NURI_WITH_LOGGING");
    report.environment.assertsEnabled =
        readBool(environment, "NURI_WITH_ASSERTS");
    report.environment.tracyEnabled = readBool(environment, "NURI_WITH_TRACY");
    report.environment.tracyDiagnostic =
        readBool(environment, "tracyDiagnostic");
    report.environment.devChecks = readBool(environment, "devChecks");
  }

  yyjson_val *run = yyjson_obj_get(root, "run");
  if (yyjson_is_obj(run)) {
    report.run.samples = readU32(run, "samples", 1u);
    report.run.warmupFrames = readU32(run, "warmupFrames");
    report.run.measurementFrames = readU32(run, "measurementFrames");
    report.run.cooldownFrames = readU32(run, "cooldownFrames");
    report.run.maxDrainFrames = readU32(run, "maxDrainFrames");
    report.run.drainTimeoutMs = readU32(run, "drainTimeoutMs");
    report.run.validForComparison = readBool(run, "validForComparison", true);
    report.run.fixedDeltaSeconds =
        readReal(run, "fixedDeltaSeconds", report.run.fixedDeltaSeconds);
  }

  yyjson_val *artifacts = yyjson_obj_get(root, "artifacts");
  if (yyjson_is_obj(artifacts)) {
    report.artifacts.artifactDir =
        std::filesystem::path(readString(artifacts, "artifactDir"));
    report.artifacts.caseReports =
        readPathArray(yyjson_obj_get(artifacts, "caseReports"));
    report.artifacts.tracyArtifacts =
        readPathArray(yyjson_obj_get(artifacts, "tracy"));
  }
  report.tracy = readTracyReport(yyjson_obj_get(root, "tracy"));

  yyjson_val *frames = yyjson_obj_get(root, "frames");
  if (yyjson_is_arr(frames)) {
    report.frames.reserve(yyjson_arr_size(frames));
    yyjson_arr_iter iter;
    yyjson_arr_iter_init(frames, &iter);
    yyjson_val *frame = nullptr;
    while ((frame = yyjson_arr_iter_next(&iter)) != nullptr) {
      if (yyjson_is_obj(frame)) {
        report.frames.push_back(readFrameRecord(frame));
      }
    }
  }
  yyjson_val *sampleStats = yyjson_obj_get(root, "sampleStats");
  if (yyjson_is_arr(sampleStats)) {
    report.sampleStats.reserve(yyjson_arr_size(sampleStats));
    yyjson_arr_iter iter;
    yyjson_arr_iter_init(sampleStats, &iter);
    yyjson_val *sample = nullptr;
    while ((sample = yyjson_arr_iter_next(&iter)) != nullptr) {
      if (yyjson_is_obj(sample)) {
        report.sampleStats.push_back(readSampleStats(sample));
      }
    }
  }

  yyjson_val *drain = yyjson_obj_get(root, "timingDrain");
  if (yyjson_is_obj(drain)) {
    report.timingDrain.drainComplete = readBool(drain, "drainComplete", true);
    report.timingDrain.drainFrames = readU32(drain, "drainFrames");
    report.timingDrain.drainTimeoutMs = readU32(drain, "drainTimeoutMs");
    report.timingDrain.missingGpuTimingFrames =
        readU32(drain, "missingGpuTimingFrames");
    report.timingDrain.scopeContainmentViolations =
        readU32(drain, "scopeContainmentViolations");
    report.timingDrain.droppedGpuTimingReports =
        readU64(drain, "droppedGpuTimingReports");
  }

  report.unavailableMetrics =
      readStringArray(yyjson_obj_get(root, "unavailableMetrics"));
  report.unregisteredObservedMetrics =
      readStringArray(yyjson_obj_get(root, "unregisteredObservedMetrics"));
  report.warnings = readStringArray(yyjson_obj_get(root, "warnings"));

  report.stats = readStatsMap(yyjson_obj_get(root, "stats"));
  return Result<BenchmarkReport, std::string>::makeResult(std::move(report));
}

} // namespace nuri::tools::benchmark
