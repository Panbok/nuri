#include "nuri/tools/snapshot/snapshot_baseline.h"

#include "nuri/tools/snapshot/snapshot_environment.h"

#include <filesystem>
#include <fstream>

namespace nuri::tools::snapshot {

std::filesystem::path defaultSnapshotBaselineRoot() {
  return snapshotRepoRoot() / "tools" / "baselines" / "render";
}

SnapshotBaselineLookup
snapshotBaselineLookup(const SnapshotCase &snapshotCase,
                       std::string_view baselineProfile) {
  SnapshotBaselineLookup lookup{};
  lookup.profileDir = defaultSnapshotBaselineRoot() / baselineProfile;
  lookup.caseDir = lookup.profileDir / snapshotCase.suite / snapshotCase.id;
  return lookup;
}

Result<bool, std::string>
approveSnapshotBaselines(const SnapshotReport &report,
                         std::string_view baselineProfile,
                         std::string_view reason) {
  if (reason.empty()) {
    return Result<bool, std::string>::makeError(
        "approveSnapshotBaselines: --reason is required");
  }
  const SnapshotBaselineLookup lookup =
      snapshotBaselineLookup(report.snapshotCase, baselineProfile);
  std::filesystem::create_directories(lookup.caseDir);

  std::ofstream metadata(lookup.caseDir / "approval.json", std::ios::binary);
  if (!metadata) {
    return Result<bool, std::string>::makeError(
        "approveSnapshotBaselines: failed to open approval metadata");
  }
  metadata << "{\n"
           << "  \"case\": \"" << report.snapshotCase.id << "\",\n"
           << "  \"baselineProfile\": \"" << baselineProfile << "\",\n"
           << "  \"reason\": \"" << reason << "\",\n"
           << "  \"approvedAtUtc\": \"" << utcTimestampIso8601() << "\",\n"
           << "  \"sourceArtifactDir\": \""
           << report.artifacts.caseDir.generic_string() << "\",\n"
           << "  \"captures\": [\n";

  bool first = true;
  for (const SnapshotCaptureReport &capture : report.captures) {
    if (capture.actual.empty() || capture.status == "missing_capture_point" ||
        capture.status == "unsupported_format" ||
        capture.status == "readback_error") {
      continue;
    }
    const std::filesystem::path sourceRaw =
        report.artifacts.caseDir / capture.actual;
    const std::filesystem::path sourceMeta =
        report.artifacts.caseDir / capture.actualMetadata;
    const std::filesystem::path sourcePreview =
        report.artifacts.caseDir / capture.preview;
    const std::filesystem::path targetStem = lookup.caseDir / capture.target;
    const std::filesystem::path targetActual =
        targetStem.string() + sourceRaw.extension().string();
    std::error_code ec;
    std::filesystem::copy_file(sourceRaw, targetActual,
                               std::filesystem::copy_options::overwrite_existing,
                               ec);
    if (ec) {
      return Result<bool, std::string>::makeError(
          "approveSnapshotBaselines: failed to copy actual artifact: " +
          ec.message());
    }
    std::filesystem::copy_file(sourceMeta, targetStem.string() + ".json",
                               std::filesystem::copy_options::overwrite_existing,
                               ec);
    if (ec) {
      return Result<bool, std::string>::makeError(
          "approveSnapshotBaselines: failed to copy metadata artifact: " +
          ec.message());
    }
    if (std::filesystem::exists(sourcePreview)) {
      std::filesystem::copy_file(
          sourcePreview, targetStem.string() + "_preview.png",
          std::filesystem::copy_options::overwrite_existing, ec);
      if (ec) {
        return Result<bool, std::string>::makeError(
            "approveSnapshotBaselines: failed to copy preview artifact: " +
            ec.message());
      }
    }
    if (!first) {
      metadata << ",\n";
    }
    first = false;
    metadata << "    {\"target\": \"" << capture.target
             << "\", \"newHash\": \"" << capture.actualHash << "\"}";
  }
  metadata << "\n  ]\n}\n";
  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri::tools::snapshot
