#include "nuri/tools/snapshot/snapshot_baseline.h"

#include "nuri/tools/core/atomic_file.h"
#include "nuri/tools/core/sha256.h"
#include "nuri/tools/snapshot/snapshot_capture_point.h"
#include "nuri/tools/snapshot/snapshot_environment.h"
#include "nuri/tools/snapshot/snapshot_image.h"
#include "nuri/tools/snapshot/snapshot_manifest.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <sstream>
#include <utility>
#include <vector>

#include <yyjson.h>

namespace nuri::tools::snapshot {
namespace {

[[nodiscard]] bool hasNonWhitespace(std::string_view text) {
  for (const unsigned char c : text) {
    if (!std::isspace(c)) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] bool isSha256Digest(std::string_view digest) {
  return digest.size() == 71u && digest.starts_with("sha256:") &&
         std::all_of(digest.begin() + 7, digest.end(), [](unsigned char value) {
           return (value >= '0' && value <= '9') ||
                  (value >= 'a' && value <= 'f');
         });
}

[[nodiscard]] std::string
computeSnapshotBaselinePlanDigest(const SnapshotBaselinePlan &plan) {
  std::ostringstream canonical;
  canonical << "nuri.snapshot.baseline_plan.v1\n"
            << plan.caseId << '\n'
            << plan.suite << '\n'
            << plan.profileId << '\n'
            << plan.reason << '\n'
            << plan.actor << '\n'
            << plan.sourceCommit << '\n'
            << plan.sourceReportDigest << '\n';
  for (const SnapshotBaselinePlanEntry &entry : plan.entries) {
    canonical << entry.path << '|' << entry.operation << '|'
              << entry.previousDigest << '|' << entry.sourceDigest << '\n';
  }
  const std::string text = canonical.str();
  return "sha256:" + nuri::tools::core::sha256Hex(
                         std::as_bytes(std::span(text.data(), text.size())));
}

[[nodiscard]] Result<std::filesystem::path, std::string>
sourceArtifactPath(const SnapshotReport &report,
                   const std::filesystem::path &relative,
                   std::string_view role) {
  auto path = resolveSnapshotPathUnder(report.artifacts.caseDir, relative);
  if (path.hasError()) {
    return Result<std::filesystem::path, std::string>::makeError(
        "approveSnapshotBaselines: unsafe " + std::string(role) +
        " path: " + path.error());
  }
  if (!std::filesystem::is_regular_file(path.value())) {
    return Result<std::filesystem::path, std::string>::makeError(
        "approveSnapshotBaselines: missing " + std::string(role) +
        " artifact: " + path.value().string());
  }
  return path;
}

} // namespace

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
verifySnapshotBaseline(const SnapshotCase &snapshotCase,
                       std::string_view baselineProfile,
                       const std::filesystem::path &baselineRootOverride) {
  for (const auto &[value, field] :
       {std::pair{std::string_view(snapshotCase.id),
                  std::string_view("case id")},
        std::pair{std::string_view(snapshotCase.suite),
                  std::string_view("suite")},
        std::pair{baselineProfile, std::string_view("baseline profile")}}) {
    auto valid = validateSnapshotIdentifier(value, field);
    if (valid.hasError()) {
      return Result<bool, std::string>::makeError("verifySnapshotBaseline: " +
                                                  valid.error());
    }
  }
  const std::filesystem::path baselineRoot = baselineRootOverride.empty()
                                                 ? defaultSnapshotBaselineRoot()
                                                 : baselineRootOverride;
  const std::filesystem::path relativeCase =
      std::filesystem::path(baselineProfile) / snapshotCase.suite /
      snapshotCase.id;
  auto caseDir = resolveSnapshotPathUnder(baselineRoot, relativeCase);
  if (caseDir.hasError()) {
    return Result<bool, std::string>::makeError("verifySnapshotBaseline: " +
                                                caseDir.error());
  }
  if (!std::filesystem::is_directory(caseDir.value())) {
    return Result<bool, std::string>::makeError(
        "verifySnapshotBaseline: baseline is missing");
  }
  const std::filesystem::path approvalPath = caseDir.value() / "approval.json";
  std::ifstream approvalFile(approvalPath, std::ios::binary);
  if (!approvalFile) {
    return Result<bool, std::string>::makeError(
        "verifySnapshotBaseline: approval.json is missing");
  }
  std::string approvalText((std::istreambuf_iterator<char>(approvalFile)),
                           std::istreambuf_iterator<char>());
  using JsonDoc = std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)>;
  JsonDoc approvalDoc(yyjson_read(approvalText.data(), approvalText.size(), 0u),
                      &yyjson_doc_free);
  yyjson_val *approval =
      approvalDoc ? yyjson_doc_get_root(approvalDoc.get()) : nullptr;
  const auto stringField = [](yyjson_val *object,
                              const char *key) -> std::string_view {
    yyjson_val *value = yyjson_obj_get(object, key);
    return yyjson_is_str(value)
               ? std::string_view(yyjson_get_str(value), yyjson_get_len(value))
               : std::string_view{};
  };
  if (!yyjson_is_obj(approval) || yyjson_obj_size(approval) != 14u ||
      !yyjson_is_uint(yyjson_obj_get(approval, "schemaVersion")) ||
      yyjson_get_uint(yyjson_obj_get(approval, "schemaVersion")) != 1u ||
      stringField(approval, "kind") != "nuri.baseline.approval" ||
      stringField(approval, "caseId") != snapshotCase.id ||
      stringField(approval, "suite") != snapshotCase.suite ||
      stringField(approval, "profileId") != baselineProfile) {
    return Result<bool, std::string>::makeError(
        "verifySnapshotBaseline: approval identity is invalid");
  }
  if (stringField(approval, "backend") != snapshotCase.backend ||
      !hasNonWhitespace(stringField(approval, "reason")) ||
      !hasNonWhitespace(stringField(approval, "actor")) ||
      !hasNonWhitespace(stringField(approval, "approvedAtUtc")) ||
      !hasNonWhitespace(stringField(approval, "sourceCommit")) ||
      !isSha256Digest(stringField(approval, "sourceDigest")) ||
      !yyjson_is_arr(yyjson_obj_get(approval, "captures"))) {
    return Result<bool, std::string>::makeError(
        "verifySnapshotBaseline: approval provenance is invalid");
  }
  const std::string planDigest(stringField(approval, "planDigest"));
  if (!isSha256Digest(planDigest)) {
    return Result<bool, std::string>::makeError(
        "verifySnapshotBaseline: approval plan digest is invalid");
  }
  const std::filesystem::path historyDir = caseDir.value() / "history";
  if (!std::filesystem::is_directory(historyDir)) {
    return Result<bool, std::string>::makeError(
        "verifySnapshotBaseline: approval history is missing");
  }
  const std::string planSuffix = planDigest.substr(7u, 16u) + ".plan.json";
  std::filesystem::path planPath;
  std::error_code treeError;
  for (const auto &entry :
       std::filesystem::recursive_directory_iterator(historyDir, treeError)) {
    const auto status = entry.symlink_status(treeError);
    if (treeError || std::filesystem::is_symlink(status)) {
      return Result<bool, std::string>::makeError(
          "verifySnapshotBaseline: approval history contains a link");
    }
    if (entry.is_directory()) {
      return Result<bool, std::string>::makeError(
          "verifySnapshotBaseline: approval history contains a directory");
    }
    const std::string name = entry.path().filename().string();
    if (entry.is_regular_file() && name.ends_with(planSuffix)) {
      if (!planPath.empty()) {
        return Result<bool, std::string>::makeError(
            "verifySnapshotBaseline: approval plan history is ambiguous");
      }
      planPath = entry.path();
    }
  }
  if (planPath.empty()) {
    return Result<bool, std::string>::makeError(
        "verifySnapshotBaseline: reviewed plan history is missing");
  }
  std::ifstream planFile(planPath, std::ios::binary);
  const std::string planText((std::istreambuf_iterator<char>(planFile)),
                             std::istreambuf_iterator<char>());
  JsonDoc planDoc(yyjson_read(planText.data(), planText.size(), 0u),
                  &yyjson_doc_free);
  yyjson_val *plan = planDoc ? yyjson_doc_get_root(planDoc.get()) : nullptr;
  if (!yyjson_is_obj(plan) || yyjson_obj_size(plan) != 11u ||
      stringField(plan, "kind") != "nuri.snapshot.baseline_plan" ||
      stringField(plan, "caseId") != snapshotCase.id ||
      stringField(plan, "suite") != snapshotCase.suite ||
      stringField(plan, "profileId") != baselineProfile ||
      stringField(plan, "planDigest") != planDigest) {
    return Result<bool, std::string>::makeError(
        "verifySnapshotBaseline: reviewed plan identity is invalid");
  }
  SnapshotBaselinePlan parsedPlan{
      .caseId = std::string(stringField(plan, "caseId")),
      .suite = std::string(stringField(plan, "suite")),
      .profileId = std::string(stringField(plan, "profileId")),
      .reason = std::string(stringField(plan, "reason")),
      .actor = std::string(stringField(plan, "actor")),
      .sourceCommit = std::string(stringField(plan, "sourceCommit")),
      .sourceReportDigest =
          std::string(stringField(plan, "sourceReportDigest")),
      .digest = planDigest,
  };
  if (!hasNonWhitespace(parsedPlan.reason) ||
      !hasNonWhitespace(parsedPlan.actor) ||
      !hasNonWhitespace(parsedPlan.sourceCommit) ||
      !isSha256Digest(parsedPlan.sourceReportDigest) ||
      parsedPlan.reason != stringField(approval, "reason") ||
      parsedPlan.actor != stringField(approval, "actor") ||
      parsedPlan.sourceCommit != stringField(approval, "sourceCommit") ||
      parsedPlan.sourceReportDigest != stringField(approval, "sourceDigest")) {
    return Result<bool, std::string>::makeError(
        "verifySnapshotBaseline: approval does not match reviewed plan");
  }
  yyjson_val *entries = yyjson_obj_get(plan, "entries");
  if (!yyjson_is_arr(entries)) {
    return Result<bool, std::string>::makeError(
        "verifySnapshotBaseline: reviewed plan entries are invalid");
  }
  std::set<std::string> plannedFiles;
  std::string previousPath;
  bool plannedApproval = false;
  yyjson_arr_iter iterator{};
  yyjson_arr_iter_init(entries, &iterator);
  yyjson_val *entry = nullptr;
  while ((entry = yyjson_arr_iter_next(&iterator)) != nullptr) {
    if (!yyjson_is_obj(entry) || yyjson_obj_size(entry) != 4u) {
      return Result<bool, std::string>::makeError(
          "verifySnapshotBaseline: reviewed plan entry is invalid");
    }
    const std::string path(stringField(entry, "path"));
    const std::string operation(stringField(entry, "operation"));
    const std::string sourceDigest(stringField(entry, "sourceDigest"));
    const std::string previousDigest(stringField(entry, "previousDigest"));
    const bool add = operation == "add";
    const bool keep = operation == "keep";
    const bool replace = operation == "replace";
    const bool remove = operation == "remove";
    const bool digestsValid =
        (add && previousDigest.empty() && isSha256Digest(sourceDigest)) ||
        ((keep || replace) && isSha256Digest(previousDigest) &&
         isSha256Digest(sourceDigest)) ||
        (remove && isSha256Digest(previousDigest) && sourceDigest.empty());
    const std::filesystem::path relativePath(path);
    if (path.empty() || relativePath.has_parent_path() ||
        (operation != "add" && operation != "replace" && operation != "keep" &&
         operation != "remove") ||
        !digestsValid || (!previousPath.empty() && previousPath >= path) ||
        !plannedFiles.insert(path).second) {
      return Result<bool, std::string>::makeError(
          "verifySnapshotBaseline: reviewed plan entries are invalid");
    }
    previousPath = path;
    parsedPlan.entries.push_back(
        SnapshotBaselinePlanEntry{.path = path,
                                  .operation = operation,
                                  .previousDigest = previousDigest,
                                  .sourceDigest = sourceDigest});
    auto resolved = resolveSnapshotPathUnder(caseDir.value(), path);
    if (resolved.hasError()) {
      return Result<bool, std::string>::makeError(
          "verifySnapshotBaseline: unsafe reviewed path");
    }
    if (operation == "remove") {
      if (std::filesystem::exists(resolved.value())) {
        return Result<bool, std::string>::makeError(
            "verifySnapshotBaseline: removed file is still present: " + path);
      }
      continue;
    }
    if (!std::filesystem::is_regular_file(resolved.value())) {
      return Result<bool, std::string>::makeError(
          "verifySnapshotBaseline: approved file is missing: " + path);
    }
    if (path == "approval.json") {
      plannedApproval = true;
      if ((operation != "add" && operation != "replace") ||
          sourceDigest != parsedPlan.sourceReportDigest) {
        return Result<bool, std::string>::makeError(
            "verifySnapshotBaseline: reviewed approval entry is invalid");
      }
      continue;
    }
    auto digest = nuri::tools::core::sha256File(resolved.value());
    if (digest.hasError() || sourceDigest != "sha256:" + digest.value()) {
      return Result<bool, std::string>::makeError(
          "verifySnapshotBaseline: approved file digest mismatch: " + path);
    }
  }
  auto canonicalPlan = writeSnapshotBaselinePlanJson(parsedPlan);
  if (!plannedApproval ||
      computeSnapshotBaselinePlanDigest(parsedPlan) != planDigest ||
      canonicalPlan.hasError() || canonicalPlan.value() != planText) {
    return Result<bool, std::string>::makeError(
        "verifySnapshotBaseline: reviewed plan digest is invalid");
  }

  const std::string planPrefix = planPath.filename().string().substr(
      0u,
      planPath.filename().string().size() - std::string("plan.json").size());
  const std::filesystem::path approvalHistoryPath =
      historyDir / (planPrefix + "approval.json");
  std::ifstream approvalHistoryFile(approvalHistoryPath, std::ios::binary);
  const std::string approvalHistoryText(
      (std::istreambuf_iterator<char>(approvalHistoryFile)),
      std::istreambuf_iterator<char>());
  if (!approvalHistoryFile || approvalHistoryText != approvalText) {
    return Result<bool, std::string>::makeError(
        "verifySnapshotBaseline: current approval history mismatch");
  }

  std::map<std::string, std::string> approvedCaptureHashes;
  yyjson_arr_iter captureIterator{};
  yyjson_arr_iter_init(yyjson_obj_get(approval, "captures"), &captureIterator);
  while ((entry = yyjson_arr_iter_next(&captureIterator)) != nullptr) {
    if (!yyjson_is_obj(entry) || yyjson_obj_size(entry) != 2u) {
      return Result<bool, std::string>::makeError(
          "verifySnapshotBaseline: approval capture is invalid");
    }
    const std::string target(stringField(entry, "target"));
    const std::string hash(stringField(entry, "newHash"));
    if (target.empty() || hash.empty() ||
        !approvedCaptureHashes.emplace(target, hash).second) {
      return Result<bool, std::string>::makeError(
          "verifySnapshotBaseline: approval captures are invalid");
    }
  }

  std::set<std::string> descriptorTargets;
  for (const SnapshotBaselinePlanEntry &planned : parsedPlan.entries) {
    if (planned.operation == "remove" || planned.path == "approval.json" ||
        std::filesystem::path(planned.path).extension() != ".json") {
      continue;
    }
    const std::string target =
        std::filesystem::path(planned.path).stem().string();
    const auto manifestCapture =
        std::find_if(snapshotCase.captures.begin(), snapshotCase.captures.end(),
                     [&](const SnapshotCaptureTarget &capture) {
                       return capture.name == target;
                     });
    const SnapshotCaptureCatalogEntry *catalog =
        findSnapshotCaptureCatalogEntry(target);
    auto metadata =
        readSnapshotArtifactMetadata(caseDir.value() / planned.path);
    if (manifestCapture == snapshotCase.captures.end() || catalog == nullptr ||
        metadata.hasError() || metadata.value().target != target ||
        metadata.value().capturePointVersion != catalog->version ||
        metadata.value().kind != renderCaptureValueKindName(catalog->kind) ||
        metadata.value().profile != manifestCapture->profile ||
        std::filesystem::path(metadata.value().payload).filename() !=
            metadata.value().payload ||
        !plannedFiles.contains(metadata.value().payload) ||
        approvedCaptureHashes[target] != metadata.value().hash ||
        !descriptorTargets.insert(target).second) {
      return Result<bool, std::string>::makeError(
          "verifySnapshotBaseline: capture descriptor/payload mismatch: " +
          target);
    }
  }
  if (descriptorTargets.size() != approvedCaptureHashes.size()) {
    return Result<bool, std::string>::makeError(
        "verifySnapshotBaseline: approval capture set mismatch");
  }

  for (const auto &entry : std::filesystem::recursive_directory_iterator(
           caseDir.value(), treeError)) {
    const auto status = entry.symlink_status(treeError);
    if (treeError || std::filesystem::is_symlink(status)) {
      return Result<bool, std::string>::makeError(
          "verifySnapshotBaseline: baseline contains a link");
    }
    const std::string relative =
        entry.path().lexically_relative(caseDir.value()).generic_string();
    if (relative == "history" || relative.starts_with("history/")) {
      continue;
    }
    if (entry.is_directory() ||
        (entry.is_regular_file() && !plannedFiles.contains(relative))) {
      return Result<bool, std::string>::makeError(
          "verifySnapshotBaseline: unreviewed baseline entry: " + relative);
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<std::string, std::string>
inspectSnapshotBaseline(const SnapshotCase &snapshotCase,
                        std::string_view baselineProfile,
                        const std::filesystem::path &baselineRootOverride) {
  for (const auto &[value, field] :
       {std::pair{std::string_view(snapshotCase.id),
                  std::string_view("case id")},
        std::pair{std::string_view(snapshotCase.suite),
                  std::string_view("suite")},
        std::pair{baselineProfile, std::string_view("baseline profile")}}) {
    auto valid = validateSnapshotIdentifier(value, field);
    if (valid.hasError()) {
      return Result<std::string, std::string>::makeError(
          "inspectSnapshotBaseline: " + valid.error());
    }
  }
  const std::filesystem::path baselineRoot = baselineRootOverride.empty()
                                                 ? defaultSnapshotBaselineRoot()
                                                 : baselineRootOverride;
  auto caseDir = resolveSnapshotPathUnder(
      baselineRoot, std::filesystem::path(baselineProfile) /
                        snapshotCase.suite / snapshotCase.id);
  if (caseDir.hasError()) {
    return Result<std::string, std::string>::makeError(
        "inspectSnapshotBaseline: " + caseDir.error());
  }
  const SnapshotBaselineLookup lookup{
      .profileDir = caseDir.value().parent_path().parent_path(),
      .caseDir = caseDir.value()};
  auto verified =
      verifySnapshotBaseline(snapshotCase, baselineProfile, baselineRoot);
  using JsonDoc =
      std::unique_ptr<yyjson_mut_doc, decltype(&yyjson_mut_doc_free)>;
  JsonDoc doc(yyjson_mut_doc_new(nullptr), &yyjson_mut_doc_free);
  if (!doc) {
    return Result<std::string, std::string>::makeError(
        "inspectSnapshotBaseline: allocation failed");
  }
  yyjson_mut_val *root = yyjson_mut_obj(doc.get());
  yyjson_mut_doc_set_root(doc.get(), root);
  yyjson_mut_obj_add_uint(doc.get(), root, "schemaVersion", 1u);
  yyjson_mut_obj_add_strcpy(doc.get(), root, "kind",
                            "nuri.snapshot.baseline_inspection");
  yyjson_mut_obj_add_strcpy(doc.get(), root, "caseId", snapshotCase.id.c_str());
  yyjson_mut_obj_add_strcpy(doc.get(), root, "suite",
                            snapshotCase.suite.c_str());
  yyjson_mut_obj_add_strncpy(doc.get(), root, "profileId",
                             baselineProfile.data(), baselineProfile.size());
  yyjson_mut_obj_add_bool(doc.get(), root, "exists",
                          std::filesystem::is_directory(lookup.caseDir));
  yyjson_mut_obj_add_bool(doc.get(), root, "verified", !verified.hasError());
  yyjson_mut_obj_add_strcpy(doc.get(), root, "diagnostic",
                            verified.hasError() ? verified.error().c_str()
                                                : "");
  yyjson_mut_val *files = yyjson_mut_arr(doc.get());
  if (std::filesystem::is_directory(lookup.caseDir)) {
    std::vector<std::filesystem::path> filePaths;
    for (const auto &entry :
         std::filesystem::recursive_directory_iterator(lookup.caseDir)) {
      if (!entry.is_regular_file() || entry.is_symlink()) {
        continue;
      }
      filePaths.push_back(entry.path());
    }
    std::sort(filePaths.begin(), filePaths.end());
    for (const std::filesystem::path &filePath : filePaths) {
      auto digest = nuri::tools::core::sha256File(filePath);
      if (digest.hasError()) {
        return Result<std::string, std::string>::makeError(digest.error());
      }
      yyjson_mut_val *file = yyjson_mut_obj(doc.get());
      const std::string relative =
          filePath.lexically_relative(lookup.caseDir).generic_string();
      yyjson_mut_obj_add_strcpy(doc.get(), file, "path", relative.c_str());
      yyjson_mut_obj_add_uint(doc.get(), file, "size",
                              std::filesystem::file_size(filePath));
      const std::string digestText = "sha256:" + digest.value();
      yyjson_mut_obj_add_strcpy(doc.get(), file, "digest", digestText.c_str());
      yyjson_mut_arr_add_val(files, file);
    }
  }
  yyjson_mut_obj_add_val(doc.get(), root, "files", files);
  size_t length = 0u;
  char *json = yyjson_mut_write_opts(doc.get(), YYJSON_WRITE_PRETTY, nullptr,
                                     &length, nullptr);
  if (json == nullptr) {
    return Result<std::string, std::string>::makeError(
        "inspectSnapshotBaseline: serialization failed");
  }
  std::string text(json, length);
  std::free(json);
  return Result<std::string, std::string>::makeResult(std::move(text));
}

Result<SnapshotBaselinePlan, std::string>
planSnapshotBaselines(const SnapshotReport &report,
                      std::string_view baselineProfile, std::string_view reason,
                      std::string_view actor,
                      const std::filesystem::path &baselineRootOverride) {
  if (!hasNonWhitespace(reason) || !hasNonWhitespace(actor)) {
    return Result<SnapshotBaselinePlan, std::string>::makeError(
        "planSnapshotBaselines: reason and actor are required");
  }
  for (const auto &[value, field] :
       {std::pair{std::string_view(report.snapshotCase.id),
                  std::string_view("case id")},
        std::pair{std::string_view(report.snapshotCase.suite),
                  std::string_view("suite")},
        std::pair{baselineProfile, std::string_view("baseline profile")}}) {
    auto valid = validateSnapshotIdentifier(value, field);
    if (valid.hasError()) {
      return Result<SnapshotBaselinePlan, std::string>::makeError(
          "planSnapshotBaselines: " + valid.error());
    }
  }
  if (!report.errors.empty() || report.captures.empty()) {
    return Result<SnapshotBaselinePlan, std::string>::makeError(
        "planSnapshotBaselines: source run is incomplete");
  }
  if (!hasNonWhitespace(report.environment.commitHash)) {
    return Result<SnapshotBaselinePlan, std::string>::makeError(
        "planSnapshotBaselines: source commit is required");
  }

  auto reportJson = writeSnapshotReportJson(report);
  if (reportJson.hasError()) {
    return Result<SnapshotBaselinePlan, std::string>::makeError(
        reportJson.error());
  }
  SnapshotBaselinePlan plan{};
  plan.caseId = report.snapshotCase.id;
  plan.suite = report.snapshotCase.suite;
  plan.profileId = std::string(baselineProfile);
  plan.reason = std::string(reason);
  plan.actor = std::string(actor);
  plan.sourceCommit = report.environment.commitHash;
  plan.sourceReportDigest =
      "sha256:" + nuri::tools::core::sha256Hex(std::as_bytes(std::span(
                      reportJson.value().data(), reportJson.value().size())));

  const std::filesystem::path baselineRoot = baselineRootOverride.empty()
                                                 ? defaultSnapshotBaselineRoot()
                                                 : baselineRootOverride;
  const std::filesystem::path relativeCase =
      std::filesystem::path(baselineProfile) / report.snapshotCase.suite /
      report.snapshotCase.id;
  auto targetCase = resolveSnapshotPathUnder(baselineRoot, relativeCase);
  if (targetCase.hasError()) {
    return Result<SnapshotBaselinePlan, std::string>::makeError(
        "planSnapshotBaselines: " + targetCase.error());
  }

  std::set<std::string> plannedPaths;
  const auto appendFile = [&](const std::filesystem::path &source,
                              const std::filesystem::path &destination)
      -> Result<void, std::string> {
    auto sourceDigest = nuri::tools::core::sha256File(source);
    if (sourceDigest.hasError()) {
      return Result<void, std::string>::makeError(sourceDigest.error());
    }
    const std::string relativePath = destination.generic_string();
    plannedPaths.insert(relativePath);
    SnapshotBaselinePlanEntry entry{.path = relativePath,
                                    .operation = "add",
                                    .sourceDigest =
                                        "sha256:" + sourceDigest.value()};
    auto previousPath =
        resolveSnapshotPathUnder(baselineRoot, relativeCase / destination);
    if (previousPath.hasError()) {
      return Result<void, std::string>::makeError(previousPath.error());
    }
    if (std::filesystem::is_regular_file(previousPath.value())) {
      auto previousDigest = nuri::tools::core::sha256File(previousPath.value());
      if (previousDigest.hasError()) {
        return Result<void, std::string>::makeError(previousDigest.error());
      }
      entry.previousDigest = "sha256:" + previousDigest.value();
      entry.operation =
          entry.previousDigest == entry.sourceDigest ? "keep" : "replace";
    }
    plan.entries.push_back(std::move(entry));
    return Result<void, std::string>::makeResult();
  };

  std::set<std::string> captureNames;
  for (const SnapshotCaptureReport &capture : report.captures) {
    if (!captureNames.insert(capture.target).second ||
        (capture.required && (!capture.available || capture.actual.empty() ||
                              capture.actualMetadata.empty()))) {
      return Result<SnapshotBaselinePlan, std::string>::makeError(
          "planSnapshotBaselines: incomplete or duplicate capture '" +
          capture.target + "'");
    }
    if (capture.actual.empty()) {
      continue;
    }
    auto raw = sourceArtifactPath(report, capture.actual, "actual");
    auto metadata =
        sourceArtifactPath(report, capture.actualMetadata, "metadata");
    if (raw.hasError() || metadata.hasError()) {
      return Result<SnapshotBaselinePlan, std::string>::makeError(
          raw.hasError() ? raw.error() : metadata.error());
    }
    const std::filesystem::path stem = capture.target;
    auto added = appendFile(raw.value(),
                            stem.string() + raw.value().extension().string());
    if (!added.hasError()) {
      added = appendFile(metadata.value(), stem.string() + ".json");
    }
    if (!added.hasError() && !capture.preview.empty()) {
      auto preview = sourceArtifactPath(report, capture.preview, "preview");
      if (preview.hasError()) {
        return Result<SnapshotBaselinePlan, std::string>::makeError(
            preview.error());
      }
      added = appendFile(preview.value(), stem.string() + "_preview.png");
    }
    if (added.hasError()) {
      return Result<SnapshotBaselinePlan, std::string>::makeError(
          added.error());
    }
  }

  plannedPaths.insert("approval.json");
  SnapshotBaselinePlanEntry approvalEntry{
      .path = "approval.json",
      .operation = std::filesystem::exists(targetCase.value() / "approval.json")
                       ? "replace"
                       : "add",
      .sourceDigest = plan.sourceReportDigest};
  if (approvalEntry.operation == "replace") {
    auto previousApprovalDigest =
        nuri::tools::core::sha256File(targetCase.value() / "approval.json");
    if (previousApprovalDigest.hasError()) {
      return Result<SnapshotBaselinePlan, std::string>::makeError(
          previousApprovalDigest.error());
    }
    approvalEntry.previousDigest = "sha256:" + previousApprovalDigest.value();
  }
  plan.entries.push_back(std::move(approvalEntry));
  if (std::filesystem::is_directory(targetCase.value())) {
    for (const auto &entry :
         std::filesystem::directory_iterator(targetCase.value())) {
      if (!entry.is_regular_file()) {
        continue;
      }
      const std::string name = entry.path().filename().generic_string();
      if (plannedPaths.contains(name)) {
        continue;
      }
      auto digest = nuri::tools::core::sha256File(entry.path());
      if (digest.hasError()) {
        return Result<SnapshotBaselinePlan, std::string>::makeError(
            digest.error());
      }
      plan.entries.push_back(SnapshotBaselinePlanEntry{
          .path = name,
          .operation = "remove",
          .previousDigest = "sha256:" + digest.value()});
    }
  }
  std::sort(plan.entries.begin(), plan.entries.end(),
            [](const auto &left, const auto &right) {
              return left.path < right.path;
            });
  plan.digest = computeSnapshotBaselinePlanDigest(plan);
  return Result<SnapshotBaselinePlan, std::string>::makeResult(std::move(plan));
}

Result<std::string, std::string>
writeSnapshotBaselinePlanJson(const SnapshotBaselinePlan &plan) {
  using JsonDoc =
      std::unique_ptr<yyjson_mut_doc, decltype(&yyjson_mut_doc_free)>;
  JsonDoc doc(yyjson_mut_doc_new(nullptr), &yyjson_mut_doc_free);
  if (!doc) {
    return Result<std::string, std::string>::makeError(
        "writeSnapshotBaselinePlanJson: allocation failed");
  }
  yyjson_mut_val *root = yyjson_mut_obj(doc.get());
  yyjson_mut_doc_set_root(doc.get(), root);
  yyjson_mut_obj_add_uint(doc.get(), root, "schemaVersion", 1u);
  yyjson_mut_obj_add_strcpy(doc.get(), root, "kind",
                            "nuri.snapshot.baseline_plan");
  yyjson_mut_obj_add_strcpy(doc.get(), root, "caseId", plan.caseId.c_str());
  yyjson_mut_obj_add_strcpy(doc.get(), root, "suite", plan.suite.c_str());
  yyjson_mut_obj_add_strcpy(doc.get(), root, "profileId",
                            plan.profileId.c_str());
  yyjson_mut_obj_add_strcpy(doc.get(), root, "reason", plan.reason.c_str());
  yyjson_mut_obj_add_strcpy(doc.get(), root, "actor", plan.actor.c_str());
  yyjson_mut_obj_add_strcpy(doc.get(), root, "sourceCommit",
                            plan.sourceCommit.c_str());
  yyjson_mut_obj_add_strcpy(doc.get(), root, "sourceReportDigest",
                            plan.sourceReportDigest.c_str());
  yyjson_mut_obj_add_strcpy(doc.get(), root, "planDigest", plan.digest.c_str());
  yyjson_mut_val *entries = yyjson_mut_arr(doc.get());
  for (const SnapshotBaselinePlanEntry &entry : plan.entries) {
    yyjson_mut_val *object = yyjson_mut_obj(doc.get());
    yyjson_mut_obj_add_strcpy(doc.get(), object, "path", entry.path.c_str());
    yyjson_mut_obj_add_strcpy(doc.get(), object, "operation",
                              entry.operation.c_str());
    yyjson_mut_obj_add_strcpy(doc.get(), object, "previousDigest",
                              entry.previousDigest.c_str());
    yyjson_mut_obj_add_strcpy(doc.get(), object, "sourceDigest",
                              entry.sourceDigest.c_str());
    yyjson_mut_arr_add_val(entries, object);
  }
  yyjson_mut_obj_add_val(doc.get(), root, "entries", entries);
  size_t length = 0u;
  char *json = yyjson_mut_write_opts(doc.get(), YYJSON_WRITE_PRETTY, nullptr,
                                     &length, nullptr);
  if (json == nullptr) {
    return Result<std::string, std::string>::makeError(
        "writeSnapshotBaselinePlanJson: serialization failed");
  }
  std::string text(json, length);
  std::free(json);
  return Result<std::string, std::string>::makeResult(std::move(text));
}

Result<bool, std::string> approveSnapshotBaselines(
    const SnapshotReport &report, std::string_view baselineProfile,
    std::string_view reason, std::string_view confirmPlanDigest,
    std::string_view actor, const std::filesystem::path &baselineRootOverride) {
  if (!hasNonWhitespace(reason)) {
    return Result<bool, std::string>::makeError(
        "approveSnapshotBaselines: --reason is required");
  }
  for (const auto &[value, field] :
       {std::pair{std::string_view(report.snapshotCase.id),
                  std::string_view("case id")},
        std::pair{std::string_view(report.snapshotCase.suite),
                  std::string_view("suite")},
        std::pair{baselineProfile, std::string_view("baseline profile")}}) {
    auto valid = validateSnapshotIdentifier(value, field);
    if (valid.hasError()) {
      return Result<bool, std::string>::makeError("approveSnapshotBaselines: " +
                                                  valid.error());
    }
  }
  if (!report.errors.empty()) {
    return Result<bool, std::string>::makeError(
        "approveSnapshotBaselines: source report contains runtime errors");
  }
  if (report.captures.empty()) {
    return Result<bool, std::string>::makeError(
        "approveSnapshotBaselines: source report has no captures");
  }

  std::set<std::string> captureNames;
  for (const SnapshotCaptureReport &capture : report.captures) {
    if (!captureNames.insert(capture.target).second) {
      return Result<bool, std::string>::makeError(
          "approveSnapshotBaselines: duplicate capture '" + capture.target +
          "'");
    }
    auto valid = validateSnapshotIdentifier(capture.target, "capture target");
    const SnapshotCaptureCatalogEntry *catalog =
        findSnapshotCaptureCatalogEntry(capture.target);
    if (valid.hasError() || catalog == nullptr) {
      return Result<bool, std::string>::makeError(
          "approveSnapshotBaselines: invalid capture target '" +
          capture.target + "'");
    }
    if (capture.required &&
        (!capture.available || capture.actual.empty() ||
         capture.actualMetadata.empty() || capture.actualHash.empty() ||
         (capture.status != "captured" && capture.status != "pass" &&
          capture.status != "fail"))) {
      return Result<bool, std::string>::makeError(
          "approveSnapshotBaselines: required capture '" + capture.target +
          "' is incomplete");
    }
    if (!capture.actual.empty() &&
        (capture.capturePointVersion != catalog->version ||
         capture.kind != renderCaptureValueKindName(catalog->kind) ||
         !snapshotCompareProfileSupportsKind(capture.profile, catalog->kind))) {
      return Result<bool, std::string>::makeError(
          "approveSnapshotBaselines: incompatible descriptor for capture '" +
          capture.target + "'");
    }
  }
  for (const SnapshotCaptureTarget &required : report.snapshotCase.captures) {
    if (!required.required) {
      continue;
    }
    const auto found = std::find_if(
        report.captures.begin(), report.captures.end(),
        [&](const SnapshotCaptureReport &capture) {
          return capture.target == required.name && capture.required;
        });
    if (found == report.captures.end()) {
      return Result<bool, std::string>::makeError(
          "approveSnapshotBaselines: required capture '" + required.name +
          "' is absent from the source report");
    }
  }

  auto approvalPlan = planSnapshotBaselines(report, baselineProfile, reason,
                                            actor, baselineRootOverride);
  if (approvalPlan.hasError()) {
    return Result<bool, std::string>::makeError(approvalPlan.error());
  }
  if (confirmPlanDigest.empty() ||
      confirmPlanDigest != approvalPlan.value().digest) {
    return Result<bool, std::string>::makeError(
        "approveSnapshotBaselines: --confirm-plan must match the reviewed "
        "plan digest " +
        approvalPlan.value().digest);
  }
  auto approvalJson = writeSnapshotBaselinePlanJson(approvalPlan.value());
  if (approvalJson.hasError()) {
    return Result<bool, std::string>::makeError(approvalJson.error());
  }

  const std::filesystem::path baselineRoot = baselineRootOverride.empty()
                                                 ? defaultSnapshotBaselineRoot()
                                                 : baselineRootOverride;
  const std::filesystem::path relativeCase =
      std::filesystem::path(baselineProfile) / report.snapshotCase.suite /
      report.snapshotCase.id;
  auto targetResult = resolveSnapshotPathUnder(baselineRoot, relativeCase);
  if (targetResult.hasError()) {
    return Result<bool, std::string>::makeError("approveSnapshotBaselines: " +
                                                targetResult.error());
  }
  const std::string nonce = utcTimestampForPath();
  const std::filesystem::path stageRelative =
      relativeCase.parent_path() / (report.snapshotCase.id + ".stage-" + nonce);
  const std::filesystem::path backupRelative =
      relativeCase.parent_path() /
      (report.snapshotCase.id + ".backup-" + nonce);
  auto stageResult = resolveSnapshotPathUnder(baselineRoot, stageRelative);
  auto backupResult = resolveSnapshotPathUnder(baselineRoot, backupRelative);
  if (stageResult.hasError() || backupResult.hasError()) {
    return Result<bool, std::string>::makeError(
        "approveSnapshotBaselines: failed containment check for staging");
  }
  const std::filesystem::path stageDir = stageResult.value();

  std::error_code ec;
  std::filesystem::create_directories(stageDir, ec);
  if (ec) {
    return Result<bool, std::string>::makeError(
        "approveSnapshotBaselines: failed to create staging directory: " +
        ec.message());
  }
  const auto cleanupStage = [&]() {
    auto checkedStage = resolveSnapshotPathUnder(baselineRoot, stageRelative);
    if (!checkedStage.hasError()) {
      std::error_code cleanupError;
      std::filesystem::remove_all(checkedStage.value(), cleanupError);
    }
  };

  for (const SnapshotCaptureReport &capture : report.captures) {
    if (capture.actual.empty()) {
      continue;
    }
    auto sourceRaw = sourceArtifactPath(report, capture.actual, "actual");
    auto sourceMeta =
        sourceArtifactPath(report, capture.actualMetadata, "metadata");
    if (sourceRaw.hasError() || sourceMeta.hasError()) {
      cleanupStage();
      return Result<bool, std::string>::makeError(
          sourceRaw.hasError() ? sourceRaw.error() : sourceMeta.error());
    }
    auto sourceDescriptor = readSnapshotArtifactMetadata(sourceMeta.value());
    if (sourceDescriptor.hasError() ||
        sourceDescriptor.value().target != capture.target ||
        sourceDescriptor.value().capturePointVersion !=
            capture.capturePointVersion ||
        sourceDescriptor.value().kind != capture.kind ||
        sourceDescriptor.value().profile != capture.profile ||
        sourceDescriptor.value().width != capture.width ||
        sourceDescriptor.value().height != capture.height ||
        sourceDescriptor.value().mip != capture.mip ||
        sourceDescriptor.value().layer != capture.layer ||
        sourceDescriptor.value().colorSpace != capture.colorSpace ||
        sourceDescriptor.value().origin != capture.origin ||
        sourceDescriptor.value().hash != capture.actualHash) {
      cleanupStage();
      return Result<bool, std::string>::makeError(
          "approveSnapshotBaselines: source descriptor mismatch for capture '" +
          capture.target + "'");
    }
    const std::filesystem::path targetStem = stageDir / capture.target;
    std::filesystem::copy_file(sourceRaw.value(),
                               targetStem.string() +
                                   sourceRaw.value().extension().string(),
                               std::filesystem::copy_options::none, ec);
    if (!ec) {
      std::filesystem::copy_file(sourceMeta.value(),
                                 targetStem.string() + ".json",
                                 std::filesystem::copy_options::none, ec);
    }
    if (!ec && !capture.preview.empty()) {
      auto sourcePreview =
          sourceArtifactPath(report, capture.preview, "preview");
      if (sourcePreview.hasError()) {
        cleanupStage();
        return Result<bool, std::string>::makeError(sourcePreview.error());
      }
      std::filesystem::copy_file(sourcePreview.value(),
                                 targetStem.string() + "_preview.png",
                                 std::filesystem::copy_options::none, ec);
    }
    if (ec) {
      cleanupStage();
      return Result<bool, std::string>::makeError(
          "approveSnapshotBaselines: failed to stage capture: " + ec.message());
    }
  }

  const std::filesystem::path previousHistory =
      targetResult.value() / "history";
  const std::filesystem::path stagedHistory = stageDir / "history";
  if (std::filesystem::exists(previousHistory, ec)) {
    auto safePreviousHistory =
        resolveSnapshotPathUnder(baselineRoot, relativeCase / "history");
    if (safePreviousHistory.hasError()) {
      cleanupStage();
      return Result<bool, std::string>::makeError(
          "approveSnapshotBaselines: unsafe approval history path");
    }
    std::filesystem::copy(safePreviousHistory.value(), stagedHistory,
                          std::filesystem::copy_options::recursive |
                              std::filesystem::copy_options::skip_symlinks,
                          ec);
    if (ec) {
      cleanupStage();
      return Result<bool, std::string>::makeError(
          "approveSnapshotBaselines: failed to preserve approval history: " +
          ec.message());
    }
  } else {
    std::filesystem::create_directories(stagedHistory, ec);
  }
  if (ec) {
    cleanupStage();
    return Result<bool, std::string>::makeError(
        "approveSnapshotBaselines: failed to create approval history: " +
        ec.message());
  }
  const std::filesystem::path previousApproval =
      targetResult.value() / "approval.json";
  if (std::filesystem::is_regular_file(previousApproval)) {
    auto safePreviousApproval =
        resolveSnapshotPathUnder(baselineRoot, relativeCase / "approval.json");
    if (safePreviousApproval.hasError()) {
      cleanupStage();
      return Result<bool, std::string>::makeError(
          "approveSnapshotBaselines: unsafe previous approval path");
    }
    auto previousApprovalDigest =
        nuri::tools::core::sha256File(safePreviousApproval.value());
    if (previousApprovalDigest.hasError()) {
      cleanupStage();
      return Result<bool, std::string>::makeError(
          previousApprovalDigest.error());
    }
    const std::filesystem::path legacyPath =
        stagedHistory /
        ("legacy-" + previousApprovalDigest.value().substr(0u, 16u) +
         ".approval.json");
    if (!std::filesystem::exists(legacyPath)) {
      std::filesystem::copy_file(safePreviousApproval.value(), legacyPath,
                                 std::filesystem::copy_options::none, ec);
      if (ec) {
        cleanupStage();
        return Result<bool, std::string>::makeError(
            "approveSnapshotBaselines: failed to preserve previous approval: " +
            ec.message());
      }
    }
  }
  const std::string historyName =
      nonce + "-" + approvalPlan.value().digest.substr(7u, 16u) + ".plan.json";
  auto historyWrite = nuri::tools::core::atomicWriteTextFile(
      stagedHistory / historyName, approvalJson.value());
  if (historyWrite.hasError()) {
    cleanupStage();
    return Result<bool, std::string>::makeError(historyWrite.error());
  }

  using ApprovalDoc =
      std::unique_ptr<yyjson_mut_doc, decltype(&yyjson_mut_doc_free)>;
  ApprovalDoc metadataDoc(yyjson_mut_doc_new(nullptr), &yyjson_mut_doc_free);
  if (!metadataDoc) {
    cleanupStage();
    return Result<bool, std::string>::makeError(
        "approveSnapshotBaselines: approval allocation failed");
  }
  yyjson_mut_val *metadataRoot = yyjson_mut_obj(metadataDoc.get());
  yyjson_mut_doc_set_root(metadataDoc.get(), metadataRoot);
  yyjson_mut_obj_add_uint(metadataDoc.get(), metadataRoot, "schemaVersion", 1u);
  yyjson_mut_obj_add_strcpy(metadataDoc.get(), metadataRoot, "kind",
                            "nuri.baseline.approval");
  yyjson_mut_obj_add_strcpy(metadataDoc.get(), metadataRoot, "caseId",
                            report.snapshotCase.id.c_str());
  yyjson_mut_obj_add_strcpy(metadataDoc.get(), metadataRoot, "suite",
                            report.snapshotCase.suite.c_str());
  yyjson_mut_obj_add_strncpy(metadataDoc.get(), metadataRoot, "profileId",
                             baselineProfile.data(), baselineProfile.size());
  yyjson_mut_obj_add_strcpy(metadataDoc.get(), metadataRoot, "backend",
                            report.snapshotCase.backend.c_str());
  yyjson_mut_obj_add_strcpy(metadataDoc.get(), metadataRoot, "resolvedBackend",
                            report.environment.gpuBackend.c_str());
  yyjson_mut_obj_add_strncpy(metadataDoc.get(), metadataRoot, "reason",
                             reason.data(), reason.size());
  yyjson_mut_obj_add_strncpy(metadataDoc.get(), metadataRoot, "actor",
                             actor.data(), actor.size());
  const std::string approvedAt = utcTimestampIso8601();
  yyjson_mut_obj_add_strcpy(metadataDoc.get(), metadataRoot, "approvedAtUtc",
                            approvedAt.c_str());
  yyjson_mut_obj_add_strcpy(metadataDoc.get(), metadataRoot, "sourceCommit",
                            report.environment.commitHash.c_str());
  yyjson_mut_obj_add_strcpy(metadataDoc.get(), metadataRoot, "sourceDigest",
                            approvalPlan.value().sourceReportDigest.c_str());
  yyjson_mut_obj_add_strcpy(metadataDoc.get(), metadataRoot, "planDigest",
                            approvalPlan.value().digest.c_str());
  yyjson_mut_val *metadataCaptures = yyjson_mut_arr(metadataDoc.get());
  for (const SnapshotCaptureReport &capture : report.captures) {
    if (capture.actual.empty()) {
      continue;
    }
    yyjson_mut_val *captureObject = yyjson_mut_obj(metadataDoc.get());
    yyjson_mut_obj_add_strcpy(metadataDoc.get(), captureObject, "target",
                              capture.target.c_str());
    yyjson_mut_obj_add_strcpy(metadataDoc.get(), captureObject, "newHash",
                              capture.actualHash.c_str());
    yyjson_mut_arr_add_val(metadataCaptures, captureObject);
  }
  yyjson_mut_obj_add_val(metadataDoc.get(), metadataRoot, "captures",
                         metadataCaptures);
  size_t metadataLength = 0u;
  char *metadataJson =
      yyjson_mut_write_opts(metadataDoc.get(), YYJSON_WRITE_PRETTY, nullptr,
                            &metadataLength, nullptr);
  if (metadataJson == nullptr) {
    cleanupStage();
    return Result<bool, std::string>::makeError(
        "approveSnapshotBaselines: approval serialization failed");
  }
  const std::string metadataText(metadataJson, metadataLength);
  std::free(metadataJson);
  auto metadataWrite = nuri::tools::core::atomicWriteTextFile(
      stageDir / "approval.json", metadataText);
  if (metadataWrite.hasError()) {
    cleanupStage();
    return Result<bool, std::string>::makeError("approveSnapshotBaselines: " +
                                                metadataWrite.error());
  }
  auto approvalHistoryWrite = nuri::tools::core::atomicWriteTextFile(
      stagedHistory /
          (nonce + "-" + approvalPlan.value().digest.substr(7u, 16u) +
           ".approval.json"),
      metadataText);
  if (approvalHistoryWrite.hasError()) {
    cleanupStage();
    return Result<bool, std::string>::makeError("approveSnapshotBaselines: " +
                                                approvalHistoryWrite.error());
  }

  for (const SnapshotBaselinePlanEntry &entry : approvalPlan.value().entries) {
    if (entry.operation == "remove" || entry.path == "approval.json") {
      continue;
    }
    auto stagedPath = resolveSnapshotPathUnder(stageDir, entry.path);
    if (stagedPath.hasError() ||
        !std::filesystem::is_regular_file(stagedPath.value())) {
      cleanupStage();
      return Result<bool, std::string>::makeError(
          "approveSnapshotBaselines: staged reviewed file is missing: " +
          entry.path);
    }
    auto stagedDigest = nuri::tools::core::sha256File(stagedPath.value());
    if (stagedDigest.hasError() ||
        entry.sourceDigest != "sha256:" + stagedDigest.value()) {
      cleanupStage();
      return Result<bool, std::string>::makeError(
          "approveSnapshotBaselines: staged file changed after review: " +
          entry.path);
    }
  }

  targetResult = resolveSnapshotPathUnder(baselineRoot, relativeCase);
  stageResult = resolveSnapshotPathUnder(baselineRoot, stageRelative);
  backupResult = resolveSnapshotPathUnder(baselineRoot, backupRelative);
  if (targetResult.hasError() || stageResult.hasError() ||
      backupResult.hasError()) {
    cleanupStage();
    return Result<bool, std::string>::makeError(
        "approveSnapshotBaselines: containment changed before promotion");
  }
  const bool hadPrevious = std::filesystem::exists(targetResult.value(), ec);
  if (hadPrevious) {
    std::filesystem::rename(targetResult.value(), backupResult.value(), ec);
    if (ec) {
      cleanupStage();
      return Result<bool, std::string>::makeError(
          "approveSnapshotBaselines: failed to stage previous baseline: " +
          ec.message());
    }
  }
  std::filesystem::rename(stageResult.value(), targetResult.value(), ec);
  if (ec) {
    if (hadPrevious) {
      std::error_code rollbackError;
      std::filesystem::rename(backupResult.value(), targetResult.value(),
                              rollbackError);
    }
    cleanupStage();
    return Result<bool, std::string>::makeError(
        "approveSnapshotBaselines: failed to promote staged baseline: " +
        ec.message());
  }
  if (hadPrevious) {
    auto safeBackup = resolveSnapshotPathUnder(baselineRoot, backupRelative);
    if (!safeBackup.hasError()) {
      std::filesystem::remove_all(safeBackup.value(), ec);
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri::tools::snapshot
