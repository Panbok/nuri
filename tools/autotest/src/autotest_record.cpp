#include "nuri/tools/autotest/autotest_record.h"

#include "nuri/tools/snapshot/snapshot_image.h"

#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <memory>
#include <span>
#include <sstream>

#include <yyjson.h>

namespace nuri::tools::autotest {
namespace {

using JsonDocPtr = std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)>;

[[nodiscard]] std::string jsonEscape(std::string_view value) {
  std::string out;
  out.reserve(value.size());
  for (char ch : value) {
    switch (ch) {
    case '\\':
      out += "\\\\";
      break;
    case '"':
      out += "\\\"";
      break;
    case '\n':
      out += "\\n";
      break;
    default:
      out += ch;
      break;
    }
  }
  return out;
}

[[nodiscard]] std::string
manifestHashForMetadata(const AutotestCase &testCase) {
  if (!testCase.manifestPath.empty() &&
      std::filesystem::exists(testCase.manifestPath)) {
    std::ifstream file(testCase.manifestPath, std::ios::binary);
    if (file) {
      std::string bytes((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
      return nuri::tools::snapshot::snapshotHashBytes(
          std::span<const std::byte>(
              reinterpret_cast<const std::byte *>(bytes.data()), bytes.size()));
    }
  }
  return "content:" + testCase.scene.contentHash;
}

[[nodiscard]] std::string readString(yyjson_val *object, const char *key,
                                     std::string defaultValue = {}) {
  yyjson_val *value = yyjson_obj_get(object, key);
  if (!yyjson_is_str(value)) {
    return defaultValue;
  }
  return std::string(yyjson_get_str(value), yyjson_get_len(value));
}

[[nodiscard]] uint32_t readU32(yyjson_val *object, const char *key,
                               uint32_t defaultValue = 0u) {
  yyjson_val *value = yyjson_obj_get(object, key);
  return yyjson_is_uint(value) ? static_cast<uint32_t>(yyjson_get_uint(value))
                               : defaultValue;
}

[[nodiscard]] double readDouble(yyjson_val *object, const char *key,
                                double defaultValue = 0.0) {
  yyjson_val *value = yyjson_obj_get(object, key);
  return yyjson_is_num(value) ? yyjson_get_num(value) : defaultValue;
}

void addMismatch(AutotestBaselineMetadataCompatibility &out, bool mismatch,
                 std::string message) {
  if (!mismatch) {
    return;
  }
  out.compatible = false;
  out.errors.push_back(std::move(message));
}

[[nodiscard]] yyjson_val *findCheckpointMetadata(yyjson_val *checkpoints,
                                                 std::string_view id,
                                                 uint32_t frame) {
  if (!yyjson_is_arr(checkpoints)) {
    return nullptr;
  }
  yyjson_arr_iter iter;
  yyjson_arr_iter_init(checkpoints, &iter);
  yyjson_val *entry = nullptr;
  while ((entry = yyjson_arr_iter_next(&iter)) != nullptr) {
    if (yyjson_is_obj(entry) && readString(entry, "id") == id &&
        readU32(entry, "frame") == frame) {
      return entry;
    }
  }
  return nullptr;
}

[[nodiscard]] yyjson_val *findCaptureMetadata(yyjson_val *captures,
                                              std::string_view target) {
  if (!yyjson_is_arr(captures)) {
    return nullptr;
  }
  yyjson_arr_iter iter;
  yyjson_arr_iter_init(captures, &iter);
  yyjson_val *entry = nullptr;
  while ((entry = yyjson_arr_iter_next(&iter)) != nullptr) {
    if (yyjson_is_obj(entry) && readString(entry, "target") == target) {
      return entry;
    }
  }
  return nullptr;
}

} // namespace

Result<bool, std::string>
writeAutotestRecordMetadataFile(const AutotestReport &report,
                                const std::filesystem::path &caseDir,
                                std::string_view baselineProfile) {
  std::filesystem::create_directories(caseDir);
  std::ofstream metadata(caseDir / "metadata.json", std::ios::binary);
  if (!metadata) {
    return Result<bool, std::string>::makeError(
        "writeAutotestRecordMetadataFile: failed to open " +
        (caseDir / "metadata.json").string());
  }

  const AutotestCase &testCase = report.testCase;
  metadata << "{\n"
           << "  \"kind\": \"nuri.autotest.record_metadata\",\n"
           << "  \"schemaVersion\": 1,\n"
           << "  \"caseId\": \"" << jsonEscape(testCase.id) << "\",\n"
           << "  \"case\": \"" << jsonEscape(testCase.id) << "\",\n"
           << "  \"suite\": \"" << jsonEscape(testCase.suite) << "\",\n"
           << "  \"manifestHash\": \""
           << jsonEscape(manifestHashForMetadata(testCase)) << "\",\n"
           << "  \"baselineProfile\": \"" << jsonEscape(baselineProfile)
           << "\",\n"
           << "  \"backend\": \"" << jsonEscape(testCase.backend) << "\",\n"
           << "  \"resolvedBackend\": \""
           << jsonEscape(report.environment.gpuBackend) << "\",\n"
           << "  \"gpuBackendSource\": \""
           << jsonEscape(report.environment.gpuBackendSource) << "\",\n"
           << "  \"presentMode\": \"" << jsonEscape(testCase.presentMode)
           << "\",\n"
           << "  \"resolvedPresentMode\": \""
           << jsonEscape(report.environment.resolvedPresentMode) << "\",\n"
           << "  \"windowMode\": \"" << jsonEscape(testCase.windowMode)
           << "\",\n"
           << "  \"resolvedWindowMode\": \""
           << jsonEscape(report.environment.resolvedWindowMode) << "\",\n"
           << "  \"resolution\": [" << testCase.resolution[0] << ", "
           << testCase.resolution[1] << "],\n"
           << "  \"fixedDeltaSeconds\": " << std::setprecision(17)
           << testCase.fixedDeltaSeconds << ",\n"
           << "  \"build\": {\n"
           << "    \"buildType\": \""
           << jsonEscape(report.environment.buildType) << "\",\n"
           << "    \"cmakeToolProfile\": \""
           << jsonEscape(report.environment.cmakeToolProfile) << "\",\n"
           << "    \"vcpkgManifestFeatures\": \""
           << jsonEscape(report.environment.vcpkgManifestFeatures) << "\",\n"
           << "    \"NURI_WITH_TRACY\": "
           << (report.environment.tracyEnabled ? "true" : "false") << ",\n"
           << "    \"NURI_WITH_TRACY_GPU\": "
           << (report.environment.tracyGpuEnabled ? "true" : "false") << ",\n"
           << "    \"devChecks\": "
           << (report.environment.devChecks ? "true" : "false") << "\n"
           << "  },\n"
           << "  \"generatedAtUtc\": \"" << jsonEscape(report.generatedAtUtc)
           << "\",\n"
           << "  \"checkpoints\": [\n";
  for (size_t i = 0u; i < report.checkpoints.size(); ++i) {
    const AutotestCheckpointReport &checkpoint = report.checkpoints[i];
    if (i != 0u) {
      metadata << ",\n";
    }
    metadata << "    {\"id\": \"" << jsonEscape(checkpoint.id)
             << "\", \"frame\": " << checkpoint.frame << ", \"captures\": [";
    for (size_t captureIndex = 0u; captureIndex < checkpoint.captures.size();
         ++captureIndex) {
      const AutotestCaptureReport &capture = checkpoint.captures[captureIndex];
      if (captureIndex != 0u) {
        metadata << ", ";
      }
      metadata << "{\"target\": \"" << jsonEscape(capture.target)
               << "\", \"profile\": \"" << jsonEscape(capture.profile)
               << "\", \"required\": " << (capture.required ? "true" : "false")
               << ", \"compare\": " << (capture.compare ? "true" : "false")
               << ", \"width\": " << capture.snapshot.width
               << ", \"height\": " << capture.snapshot.height
               << ", \"format\": \"" << jsonEscape(capture.snapshot.format)
               << "\", \"colorSpace\": \""
               << jsonEscape(capture.snapshot.colorSpace) << "\", \"hash\": \""
               << jsonEscape(capture.snapshot.actualHash) << "\"}";
    }
    metadata << "]}";
  }
  metadata << "\n  ]\n}\n";
  return Result<bool, std::string>::makeResult(true);
}

Result<AutotestBaselineMetadataCompatibility, std::string>
validateAutotestBaselineMetadataFile(const AutotestCase &testCase,
                                     const AutotestEnvironment &environment,
                                     const std::filesystem::path &caseDir,
                                     std::string_view baselineProfile) {
  AutotestBaselineMetadataCompatibility out{};
  const std::filesystem::path metadataPath = caseDir / "metadata.json";
  if (!std::filesystem::exists(metadataPath)) {
    out.compatible = false;
    out.errors.push_back("baseline metadata missing: " + metadataPath.string());
    return Result<AutotestBaselineMetadataCompatibility,
                  std::string>::makeResult(std::move(out));
  }

  std::ifstream file(metadataPath, std::ios::binary);
  if (!file) {
    return Result<AutotestBaselineMetadataCompatibility, std::string>::
        makeError("validateAutotestBaselineMetadataFile: failed to open " +
                  metadataPath.string());
  }
  std::string json((std::istreambuf_iterator<char>(file)),
                   std::istreambuf_iterator<char>());
  yyjson_read_err error{};
  JsonDocPtr doc(yyjson_read_opts(json.data(), json.size(), 0, nullptr, &error),
                 &yyjson_doc_free);
  if (!doc) {
    return Result<AutotestBaselineMetadataCompatibility, std::string>::
        makeError("validateAutotestBaselineMetadataFile: JSON parse failed at "
                  "byte " +
                  std::to_string(error.pos) + ": " + error.msg);
  }
  yyjson_val *root = yyjson_doc_get_root(doc.get());
  if (!yyjson_is_obj(root)) {
    return Result<AutotestBaselineMetadataCompatibility, std::string>::
        makeError("validateAutotestBaselineMetadataFile: root must be object");
  }

  const std::string caseId =
      readString(root, "caseId", readString(root, "case"));
  addMismatch(out, readString(root, "kind") != "nuri.autotest.record_metadata",
              "baseline metadata kind mismatch");
  addMismatch(out, caseId != testCase.id,
              "baseline case mismatch: " + caseId + " != " + testCase.id);
  addMismatch(out, readString(root, "suite") != testCase.suite,
              "baseline suite mismatch");
  addMismatch(out,
              readString(root, "manifestHash") !=
                  manifestHashForMetadata(testCase),
              "baseline manifest hash mismatch");
  addMismatch(out, readString(root, "baselineProfile") != baselineProfile,
              "baseline profile mismatch");
  addMismatch(out,
              readString(root, "resolvedBackend") != environment.gpuBackend,
              "baseline backend mismatch");
  addMismatch(out,
              readString(root, "resolvedPresentMode") !=
                  environment.resolvedPresentMode,
              "baseline present mode mismatch");
  addMismatch(out,
              readString(root, "resolvedWindowMode") !=
                  environment.resolvedWindowMode,
              "baseline window mode mismatch");

  yyjson_val *resolution = yyjson_obj_get(root, "resolution");
  const bool resolutionMatches =
      yyjson_is_arr(resolution) && yyjson_arr_size(resolution) == 2u &&
      yyjson_is_uint(yyjson_arr_get(resolution, 0u)) &&
      yyjson_is_uint(yyjson_arr_get(resolution, 1u)) &&
      yyjson_get_uint(yyjson_arr_get(resolution, 0u)) ==
          testCase.resolution[0] &&
      yyjson_get_uint(yyjson_arr_get(resolution, 1u)) == testCase.resolution[1];
  addMismatch(out, !resolutionMatches, "baseline resolution mismatch");
  addMismatch(out,
              std::abs(readDouble(root, "fixedDeltaSeconds") -
                       testCase.fixedDeltaSeconds) > 1.0e-9,
              "baseline fixedDeltaSeconds mismatch");

  yyjson_val *checkpoints = yyjson_obj_get(root, "checkpoints");
  for (const AutotestCheckpoint &checkpoint : testCase.checkpoints) {
    yyjson_val *checkpointMetadata =
        findCheckpointMetadata(checkpoints, checkpoint.id, checkpoint.frame);
    addMismatch(out, checkpointMetadata == nullptr,
                "baseline checkpoint missing: " + checkpoint.id);
    if (checkpointMetadata == nullptr) {
      continue;
    }
    yyjson_val *captures = yyjson_obj_get(checkpointMetadata, "captures");
    for (const AutotestCaptureTarget &capture : checkpoint.captures) {
      yyjson_val *captureMetadata =
          findCaptureMetadata(captures, capture.target);
      addMismatch(out, captureMetadata == nullptr,
                  "baseline capture missing: " + checkpoint.id + "/" +
                      capture.target);
      if (captureMetadata == nullptr) {
        continue;
      }
      addMismatch(out,
                  readString(captureMetadata, "profile") != capture.profile,
                  "baseline capture profile mismatch: " + checkpoint.id + "/" +
                      capture.target);
    }
  }

  return Result<AutotestBaselineMetadataCompatibility, std::string>::makeResult(
      std::move(out));
}

AutotestRunResult recordAutotestCase(AutotestCase testCase,
                                     const AutotestRunOptions &options) {
  for (AutotestCheckpoint &checkpoint : testCase.checkpoints) {
    for (AutotestCaptureTarget &capture : checkpoint.captures) {
      capture.compare = false;
    }
  }
  return runAutotestCase(std::move(testCase), options);
}

} // namespace nuri::tools::autotest
