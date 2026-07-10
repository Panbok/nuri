#include "nuri/tools/autotest/autotest_record.h"

#include "nuri/tools/autotest/autotest_manifest.h"
#include "nuri/tools/core/atomic_file.h"
#include "nuri/tools/core/safe_path.h"
#include "nuri/tools/core/sha256.h"
#include "nuri/tools/snapshot/snapshot_baseline.h"
#include "nuri/tools/snapshot/snapshot_capture_point.h"
#include "nuri/tools/snapshot/snapshot_image.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <sstream>
#include <system_error>
#include <utility>

#include <yyjson.h>

namespace nuri::tools::autotest {
namespace {

using JsonDocPtr = std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)>;

[[nodiscard]] bool hasNonWhitespace(std::string_view text) {
  return std::any_of(text.begin(), text.end(),
                     [](unsigned char value) { return !std::isspace(value); });
}

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
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    case '\b':
      out += "\\b";
      break;
    case '\f':
      out += "\\f";
      break;
    default:
      if (static_cast<unsigned char>(ch) < 0x20u) {
        constexpr char digits[] = "0123456789abcdef";
        const auto value = static_cast<unsigned char>(ch);
        out += "\\u00";
        out += digits[value >> 4u];
        out += digits[value & 0x0fu];
      } else {
        out += ch;
      }
      break;
    }
  }
  return out;
}

[[nodiscard]] bool descriptorFormatMatches(std::string_view encoded,
                                           std::string_view expected) {
  uint32_t value = 0u;
  const auto [end, error] =
      std::from_chars(encoded.data(), encoded.data() + encoded.size(), value);
  return error == std::errc{} && end == encoded.data() + encoded.size() &&
         value < static_cast<uint32_t>(Format::Count) &&
         nuri::tools::snapshot::snapshotFormatName(
             static_cast<Format>(value)) == expected;
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

[[nodiscard]] std::filesystem::path
baselineRootFor(const AutotestBaselineApprovalOptions &options) {
  return options.baselineRoot.empty() ? defaultAutotestBaselineRoot()
                                      : options.baselineRoot;
}

[[nodiscard]] std::filesystem::path
baselineRelativeCase(const AutotestCase &testCase,
                     std::string_view baselineProfile) {
  return std::filesystem::path(baselineProfile) / "autotests" / testCase.suite /
         testCase.id;
}

struct AutotestHistoryState {
  std::string digest{};
  uint32_t fileCount = 0u;
};

struct AutotestApprovalDocument {
  std::string reason{};
  std::string actor{};
  std::string approvedAtUtc{};
  std::string sourceCommit{};
  std::string sourceReportDigest{};
  std::string carriedHistoryDigest{};
  uint32_t carriedHistoryFileCount = 0u;
  std::string planDigest{};
};

struct AutotestInspectedFile {
  std::string path{};
  uintmax_t size = 0u;
  std::string digest{};
};

struct AutotestBaselineTree {
  std::vector<AutotestInspectedFile> files{};
  std::vector<std::string> directories{};
};

[[nodiscard]] bool isSha256Digest(std::string_view value) {
  if (value.size() != 71u || !value.starts_with("sha256:")) {
    return false;
  }
  return std::all_of(value.begin() + 7, value.end(), [](unsigned char ch) {
    return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
  });
}

[[nodiscard]] std::string
computeAutotestBaselinePlanDigest(const AutotestBaselinePlan &plan) {
  std::ostringstream canonical;
  canonical << "nuri.autotest.baseline_plan.v1\n"
            << plan.caseId << '\n'
            << plan.suite << '\n'
            << plan.profileId << '\n'
            << plan.reason << '\n'
            << plan.actor << '\n'
            << plan.sourceCommit << '\n'
            << plan.sourceReportDigest << '\n'
            << plan.historyDigest << '\n'
            << plan.historyFileCount << '\n';
  for (const AutotestBaselinePlanEntry &entry : plan.entries) {
    canonical << entry.path << '|' << entry.operation << '|'
              << entry.previousDigest << '|' << entry.sourceDigest << '\n';
  }
  const std::string text = canonical.str();
  return "sha256:" + nuri::tools::core::sha256Hex(
                         std::as_bytes(std::span(text.data(), text.size())));
}

[[nodiscard]] Result<std::string, std::string>
readBaselineTextFile(const std::filesystem::path &path, std::string_view role) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return Result<std::string, std::string>::makeError(
        "failed to open autotest baseline " + std::string(role) + ": " +
        path.string());
  }
  std::string text((std::istreambuf_iterator<char>(file)),
                   std::istreambuf_iterator<char>());
  if (file.bad()) {
    return Result<std::string, std::string>::makeError(
        "failed to read autotest baseline " + std::string(role) + ": " +
        path.string());
  }
  return Result<std::string, std::string>::makeResult(std::move(text));
}

[[nodiscard]] std::string
serializeAutotestApproval(const AutotestCase &expectedCase,
                          std::string_view baselineProfile,
                          const AutotestApprovalDocument &approval) {
  std::ostringstream out;
  out << "{\n  \"schemaVersion\": 1,\n"
      << "  \"kind\": \"nuri.autotest.baseline_approval\",\n"
      << "  \"caseId\": \"" << jsonEscape(expectedCase.id)
      << "\",\n  \"suite\": \"" << jsonEscape(expectedCase.suite)
      << "\",\n  \"profileId\": \"" << jsonEscape(baselineProfile)
      << "\",\n  \"reason\": \"" << jsonEscape(approval.reason)
      << "\",\n  \"actor\": \"" << jsonEscape(approval.actor)
      << "\",\n  \"approvedAtUtc\": \"" << jsonEscape(approval.approvedAtUtc)
      << "\",\n  \"sourceCommit\": \"" << jsonEscape(approval.sourceCommit)
      << "\",\n  \"sourceReportDigest\": \""
      << jsonEscape(approval.sourceReportDigest)
      << "\",\n  \"carriedHistoryDigest\": \""
      << jsonEscape(approval.carriedHistoryDigest)
      << "\",\n  \"carriedHistoryFileCount\": "
      << approval.carriedHistoryFileCount << ",\n  \"planDigest\": \""
      << jsonEscape(approval.planDigest) << "\"\n}\n";
  return out.str();
}

[[nodiscard]] Result<AutotestApprovalDocument, std::string>
parseAutotestApproval(std::string_view text, const AutotestCase &expectedCase,
                      std::string_view baselineProfile) {
  yyjson_read_err error{};
  JsonDocPtr doc(yyjson_read_opts(const_cast<char *>(text.data()), text.size(),
                                  0u, nullptr, &error),
                 &yyjson_doc_free);
  yyjson_val *root = doc ? yyjson_doc_get_root(doc.get()) : nullptr;
  if (!yyjson_is_obj(root) || yyjson_obj_size(root) != 13u ||
      !yyjson_is_uint(yyjson_obj_get(root, "schemaVersion")) ||
      yyjson_get_uint(yyjson_obj_get(root, "schemaVersion")) != 1u ||
      readString(root, "kind") != "nuri.autotest.baseline_approval" ||
      readString(root, "caseId") != expectedCase.id ||
      readString(root, "suite") != expectedCase.suite ||
      readString(root, "profileId") != baselineProfile) {
    return Result<AutotestApprovalDocument, std::string>::makeError(
        "autotest baseline approval identity is invalid");
  }
  AutotestApprovalDocument approval{
      .reason = readString(root, "reason"),
      .actor = readString(root, "actor"),
      .approvedAtUtc = readString(root, "approvedAtUtc"),
      .sourceCommit = readString(root, "sourceCommit"),
      .sourceReportDigest = readString(root, "sourceReportDigest"),
      .carriedHistoryDigest = readString(root, "carriedHistoryDigest"),
      .planDigest = readString(root, "planDigest")};
  yyjson_val *historyFileCount =
      yyjson_obj_get(root, "carriedHistoryFileCount");
  if (!yyjson_is_uint(historyFileCount) ||
      yyjson_get_uint(historyFileCount) >
          std::numeric_limits<uint32_t>::max() ||
      !hasNonWhitespace(approval.reason) || !hasNonWhitespace(approval.actor) ||
      approval.approvedAtUtc.empty() ||
      !isSha256Digest(approval.sourceReportDigest) ||
      !isSha256Digest(approval.carriedHistoryDigest) ||
      !isSha256Digest(approval.planDigest)) {
    return Result<AutotestApprovalDocument, std::string>::makeError(
        "autotest baseline approval fields are invalid");
  }
  approval.carriedHistoryFileCount =
      static_cast<uint32_t>(yyjson_get_uint(historyFileCount));
  if (serializeAutotestApproval(expectedCase, baselineProfile, approval) !=
      text) {
    return Result<AutotestApprovalDocument, std::string>::makeError(
        "autotest baseline approval is not canonical");
  }
  return Result<AutotestApprovalDocument, std::string>::makeResult(
      std::move(approval));
}

[[nodiscard]] Result<AutotestBaselinePlan, std::string>
parseAutotestBaselinePlan(std::string_view text,
                          const AutotestCase &expectedCase,
                          std::string_view baselineProfile) {
  yyjson_read_err error{};
  JsonDocPtr doc(yyjson_read_opts(const_cast<char *>(text.data()), text.size(),
                                  0u, nullptr, &error),
                 &yyjson_doc_free);
  yyjson_val *root = doc ? yyjson_doc_get_root(doc.get()) : nullptr;
  if (!yyjson_is_obj(root) || yyjson_obj_size(root) != 13u ||
      !yyjson_is_uint(yyjson_obj_get(root, "schemaVersion")) ||
      yyjson_get_uint(yyjson_obj_get(root, "schemaVersion")) != 1u ||
      readString(root, "kind") != "nuri.autotest.baseline_plan" ||
      readString(root, "caseId") != expectedCase.id ||
      readString(root, "suite") != expectedCase.suite ||
      readString(root, "profileId") != baselineProfile) {
    return Result<AutotestBaselinePlan, std::string>::makeError(
        "autotest reviewed plan identity is invalid");
  }
  AutotestBaselinePlan plan{};
  plan.caseId = readString(root, "caseId");
  plan.suite = readString(root, "suite");
  plan.profileId = readString(root, "profileId");
  plan.reason = readString(root, "reason");
  plan.actor = readString(root, "actor");
  plan.sourceCommit = readString(root, "sourceCommit");
  plan.sourceReportDigest = readString(root, "sourceReportDigest");
  plan.historyDigest = readString(root, "historyDigest");
  plan.digest = readString(root, "planDigest");
  yyjson_val *historyFileCount = yyjson_obj_get(root, "historyFileCount");
  yyjson_val *entries = yyjson_obj_get(root, "entries");
  if (!hasNonWhitespace(plan.reason) || !hasNonWhitespace(plan.actor) ||
      !isSha256Digest(plan.sourceReportDigest) ||
      !isSha256Digest(plan.historyDigest) || !isSha256Digest(plan.digest) ||
      !yyjson_is_uint(historyFileCount) ||
      yyjson_get_uint(historyFileCount) >
          std::numeric_limits<uint32_t>::max() ||
      !yyjson_is_arr(entries)) {
    return Result<AutotestBaselinePlan, std::string>::makeError(
        "autotest reviewed plan fields are invalid");
  }
  plan.historyFileCount =
      static_cast<uint32_t>(yyjson_get_uint(historyFileCount));
  std::set<std::string> paths;
  yyjson_arr_iter iterator{};
  yyjson_arr_iter_init(entries, &iterator);
  yyjson_val *entry = nullptr;
  std::string previousPath;
  bool hasApproval = false;
  while ((entry = yyjson_arr_iter_next(&iterator)) != nullptr) {
    if (!yyjson_is_obj(entry) || yyjson_obj_size(entry) != 4u ||
        !yyjson_is_str(yyjson_obj_get(entry, "path")) ||
        !yyjson_is_str(yyjson_obj_get(entry, "operation")) ||
        !yyjson_is_str(yyjson_obj_get(entry, "previousDigest")) ||
        !yyjson_is_str(yyjson_obj_get(entry, "sourceDigest"))) {
      return Result<AutotestBaselinePlan, std::string>::makeError(
          "autotest reviewed plan entry is invalid");
    }
    AutotestBaselinePlanEntry parsed{
        .path = readString(entry, "path"),
        .operation = readString(entry, "operation"),
        .previousDigest = readString(entry, "previousDigest"),
        .sourceDigest = readString(entry, "sourceDigest")};
    const bool add = parsed.operation == "add";
    const bool keep = parsed.operation == "keep";
    const bool replace = parsed.operation == "replace";
    const bool remove = parsed.operation == "remove";
    const bool digestsValid =
        (add && parsed.previousDigest.empty() &&
         isSha256Digest(parsed.sourceDigest)) ||
        ((keep || replace) && isSha256Digest(parsed.previousDigest) &&
         isSha256Digest(parsed.sourceDigest)) ||
        (remove && isSha256Digest(parsed.previousDigest) &&
         parsed.sourceDigest.empty());
    if (parsed.path.empty() || (!add && !keep && !replace && !remove) ||
        !digestsValid || !paths.insert(parsed.path).second ||
        (!previousPath.empty() && previousPath >= parsed.path)) {
      return Result<AutotestBaselinePlan, std::string>::makeError(
          "autotest reviewed plan entries are invalid or unsorted");
    }
    previousPath = parsed.path;
    if (parsed.path == "approval.json") {
      if ((parsed.operation != "add" && parsed.operation != "replace") ||
          parsed.sourceDigest != plan.sourceReportDigest) {
        return Result<AutotestBaselinePlan, std::string>::makeError(
            "autotest reviewed plan approval entry is invalid");
      }
      hasApproval = true;
    }
    plan.entries.push_back(std::move(parsed));
  }
  auto canonical = writeAutotestBaselinePlanJson(plan);
  if (!hasApproval || computeAutotestBaselinePlanDigest(plan) != plan.digest ||
      canonical.hasError() || canonical.value() != text) {
    return Result<AutotestBaselinePlan, std::string>::makeError(
        "autotest reviewed plan digest is invalid");
  }
  return Result<AutotestBaselinePlan, std::string>::makeResult(std::move(plan));
}

[[nodiscard]] Result<AutotestHistoryState, std::string>
makeAutotestHistoryState(
    std::vector<std::pair<std::string, std::string>> files) {
  if (files.size() > std::numeric_limits<uint32_t>::max()) {
    return Result<AutotestHistoryState, std::string>::makeError(
        "autotest baseline history contains too many files");
  }
  std::sort(files.begin(), files.end());
  std::ostringstream canonical;
  canonical << "nuri.autotest.baseline_history.v1\n";
  for (const auto &[path, digest] : files) {
    canonical << path << '|' << digest << '\n';
  }
  const std::string text = canonical.str();
  return Result<AutotestHistoryState, std::string>::makeResult(
      AutotestHistoryState{.digest = "sha256:" +
                                     nuri::tools::core::sha256Hex(std::as_bytes(
                                         std::span(text.data(), text.size()))),
                           .fileCount = static_cast<uint32_t>(files.size())});
}

[[nodiscard]] Result<AutotestHistoryState, std::string>
readAutotestHistoryState(const std::filesystem::path &baselineRoot,
                         const std::filesystem::path &relativeHistory) {
  auto history =
      nuri::tools::core::resolvePathUnder(baselineRoot, relativeHistory);
  if (history.hasError()) {
    return Result<AutotestHistoryState, std::string>::makeError(
        history.error());
  }
  std::vector<std::pair<std::string, std::string>> files;
  std::error_code error;
  const bool exists = std::filesystem::exists(history.value(), error);
  if (error) {
    return Result<AutotestHistoryState, std::string>::makeError(
        "failed to inspect autotest baseline history: " + error.message());
  }
  if (exists && !std::filesystem::is_directory(history.value(), error)) {
    return Result<AutotestHistoryState, std::string>::makeError(
        "autotest baseline history path is not a directory" +
        (error ? " (" + error.message() + ")" : std::string{}));
  }
  if (exists) {
    for (const auto &entry : std::filesystem::recursive_directory_iterator(
             history.value(), error)) {
      if (error) {
        return Result<AutotestHistoryState, std::string>::makeError(
            "failed to inspect autotest baseline history: " + error.message());
      }
      const std::filesystem::path relative =
          std::filesystem::relative(entry.path(), history.value(), error);
      if (error) {
        return Result<AutotestHistoryState, std::string>::makeError(
            "failed to relativize autotest baseline history: " +
            error.message());
      }
      auto safeEntry = nuri::tools::core::resolvePathUnder(
          baselineRoot, relativeHistory / relative);
      if (safeEntry.hasError()) {
        return Result<AutotestHistoryState, std::string>::makeError(
            safeEntry.error());
      }
      if (entry.is_directory(error)) {
        if (error) {
          return Result<AutotestHistoryState, std::string>::makeError(
              "failed to inspect autotest baseline history entry: " +
              error.message());
        }
        continue;
      }
      if (!entry.is_regular_file(error) || error) {
        return Result<AutotestHistoryState, std::string>::makeError(
            "autotest baseline history contains an unsupported entry: " +
            relative.generic_string() +
            (error ? " (" + error.message() + ")" : std::string{}));
      }
      auto digest = nuri::tools::core::sha256File(safeEntry.value());
      if (digest.hasError()) {
        return Result<AutotestHistoryState, std::string>::makeError(
            digest.error());
      }
      files.emplace_back(relative.generic_string(), digest.value());
    }
    if (error) {
      return Result<AutotestHistoryState, std::string>::makeError(
          "failed to inspect autotest baseline history: " + error.message());
    }
  }
  return makeAutotestHistoryState(std::move(files));
}

[[nodiscard]] Result<AutotestBaselineTree, std::string>
readAutotestBaselineTree(const std::filesystem::path &baselineRoot,
                         const std::filesystem::path &relativeCase) {
  auto caseDir =
      nuri::tools::core::resolvePathUnder(baselineRoot, relativeCase);
  if (caseDir.hasError()) {
    return Result<AutotestBaselineTree, std::string>::makeError(
        caseDir.error());
  }
  std::error_code error;
  if (!std::filesystem::is_directory(caseDir.value(), error) || error) {
    return Result<AutotestBaselineTree, std::string>::makeError(
        "autotest baseline case directory is missing" +
        (error ? " (" + error.message() + ")" : std::string{}));
  }
  AutotestBaselineTree tree{};
  for (const auto &entry :
       std::filesystem::recursive_directory_iterator(caseDir.value(), error)) {
    if (error) {
      return Result<AutotestBaselineTree, std::string>::makeError(
          "failed to inspect autotest baseline tree: " + error.message());
    }
    const std::filesystem::path relative =
        entry.path().lexically_relative(caseDir.value());
    auto safeEntry = nuri::tools::core::resolvePathUnder(
        baselineRoot, relativeCase / relative);
    if (safeEntry.hasError()) {
      return Result<AutotestBaselineTree, std::string>::makeError(
          "autotest baseline tree contains a link or escape: " +
          relative.generic_string());
    }
    if (entry.is_directory(error)) {
      if (error) {
        return Result<AutotestBaselineTree, std::string>::makeError(
            "failed to inspect autotest baseline directory: " +
            error.message());
      }
      tree.directories.push_back(relative.generic_string());
      continue;
    }
    if (!entry.is_regular_file(error) || error) {
      return Result<AutotestBaselineTree, std::string>::makeError(
          "autotest baseline tree contains an unsupported entry: " +
          relative.generic_string() +
          (error ? " (" + error.message() + ")" : std::string{}));
    }
    auto digest = nuri::tools::core::sha256File(safeEntry.value());
    const uintmax_t size = std::filesystem::file_size(safeEntry.value(), error);
    if (digest.hasError() || error) {
      return Result<AutotestBaselineTree, std::string>::makeError(
          digest.hasError()
              ? digest.error()
              : "failed to inspect autotest baseline file: " + error.message());
    }
    tree.files.push_back(
        AutotestInspectedFile{.path = relative.generic_string(),
                              .size = size,
                              .digest = "sha256:" + digest.value()});
  }
  if (error) {
    return Result<AutotestBaselineTree, std::string>::makeError(
        "failed to inspect autotest baseline tree: " + error.message());
  }
  std::sort(tree.files.begin(), tree.files.end(),
            [](const auto &left, const auto &right) {
              return left.path < right.path;
            });
  std::sort(tree.directories.begin(), tree.directories.end());
  return Result<AutotestBaselineTree, std::string>::makeResult(std::move(tree));
}

[[nodiscard]] Result<std::filesystem::path, std::string>
candidateFile(const AutotestReport &report,
              const std::filesystem::path &relative, std::string_view role) {
  auto resolved =
      nuri::tools::core::resolvePathUnder(report.artifacts.caseDir, relative);
  if (resolved.hasError()) {
    return Result<std::filesystem::path, std::string>::makeError(
        "autotest baseline candidate has unsafe " + std::string(role) +
        " path: " + resolved.error());
  }
  std::error_code error;
  if (!std::filesystem::is_regular_file(resolved.value(), error) || error) {
    return Result<std::filesystem::path, std::string>::makeError(
        "autotest baseline candidate is missing " + std::string(role) + ": " +
        resolved.value().string() +
        (error ? " (" + error.message() + ")" : std::string{}));
  }
  return resolved;
}

[[nodiscard]] const AutotestCheckpointReport *
findCheckpointReport(const AutotestReport &report,
                     const AutotestCheckpoint &expected) {
  const auto found =
      std::find_if(report.checkpoints.begin(), report.checkpoints.end(),
                   [&](const AutotestCheckpointReport &checkpoint) {
                     return checkpoint.id == expected.id &&
                            checkpoint.frame == expected.frame;
                   });
  return found == report.checkpoints.end() ? nullptr : &*found;
}

[[nodiscard]] const AutotestCaptureReport *
findCaptureReport(const AutotestCheckpointReport &checkpoint,
                  const AutotestCaptureTarget &expected) {
  const auto found =
      std::find_if(checkpoint.captures.begin(), checkpoint.captures.end(),
                   [&](const AutotestCaptureReport &capture) {
                     return capture.target == expected.target;
                   });
  return found == checkpoint.captures.end() ? nullptr : &*found;
}

[[nodiscard]] Result<bool, std::string>
validateApprovalCandidate(const AutotestCase &expectedCase,
                          const AutotestReport &report,
                          std::string_view baselineProfile) {
  auto valid = validateAutotestCase(expectedCase);
  if (valid.hasError()) {
    return Result<bool, std::string>::makeError(
        "autotest baseline candidate case is invalid: " + valid.error());
  }
  for (const auto &[value, field] :
       {std::pair{std::string_view(expectedCase.id), std::string_view("case")},
        std::pair{std::string_view(expectedCase.suite),
                  std::string_view("suite")},
        std::pair{baselineProfile, std::string_view("baseline profile")}}) {
    valid = validateAutotestIdentifier(value, field, false);
    if (valid.hasError()) {
      return Result<bool, std::string>::makeError(valid.error());
    }
  }
  if (report.kind != "nuri.autotest.report" || report.schemaVersion != 1u ||
      report.testCase.id != expectedCase.id ||
      report.testCase.suite != expectedCase.suite ||
      report.baselineProfile != baselineProfile) {
    return Result<bool, std::string>::makeError(
        "autotest baseline candidate report identity does not match the "
        "requested case");
  }
  if (report.status != "pass" || report.exitCode != AutotestExitCode::Success ||
      !report.errors.empty() || !report.run.validForComparison ||
      report.selection.selected != 1u || report.selection.attempted != 1u ||
      report.selection.completed != 1u || report.selection.passed != 1u) {
    return Result<bool, std::string>::makeError(
        "autotest baseline candidate run is incomplete or did not pass");
  }
  if (report.artifacts.caseDir.empty()) {
    return Result<bool, std::string>::makeError(
        "autotest baseline candidate case directory is missing");
  }
  for (const AutotestCheckpointReport &checkpoint : report.checkpoints) {
    if (!checkpoint.errors.empty()) {
      return Result<bool, std::string>::makeError(
          "autotest baseline candidate checkpoint contains errors: " +
          checkpoint.id);
    }
  }
  for (const AutotestMetricWindowReport &window : report.metricWindows) {
    if (!window.errors.empty()) {
      return Result<bool, std::string>::makeError(
          "autotest baseline candidate metric window contains errors: " +
          window.id);
    }
  }

  auto metadata = validateAutotestBaselineMetadataFile(
      expectedCase, report.environment, report.artifacts.caseDir,
      baselineProfile);
  if (metadata.hasError()) {
    return Result<bool, std::string>::makeError(metadata.error());
  }
  if (!metadata.value().compatible) {
    return Result<bool, std::string>::makeError(
        metadata.value().errors.empty()
            ? "autotest baseline candidate metadata is incompatible"
            : metadata.value().errors.front());
  }

  size_t expectedCaptureCount = 0u;
  std::set<std::string> reportCheckpointIds;
  for (const AutotestCheckpointReport &checkpoint : report.checkpoints) {
    if (!reportCheckpointIds.insert(checkpoint.id).second) {
      return Result<bool, std::string>::makeError(
          "autotest baseline candidate contains duplicate checkpoint '" +
          checkpoint.id + "'");
    }
  }
  for (const AutotestCheckpoint &checkpoint : expectedCase.checkpoints) {
    const AutotestCheckpointReport *checkpointReport =
        findCheckpointReport(report, checkpoint);
    if (checkpointReport == nullptr ||
        checkpointReport->captures.size() != checkpoint.captures.size()) {
      return Result<bool, std::string>::makeError(
          "autotest baseline candidate checkpoint does not match manifest: " +
          checkpoint.id);
    }
    std::set<std::string> reportCaptureIds;
    for (const AutotestCaptureReport &capture : checkpointReport->captures) {
      if (!reportCaptureIds.insert(capture.target).second) {
        return Result<bool, std::string>::makeError(
            "autotest baseline candidate contains duplicate capture '" +
            checkpoint.id + "/" + capture.target + "'");
      }
    }
    for (const AutotestCaptureTarget &expected : checkpoint.captures) {
      ++expectedCaptureCount;
      const AutotestCaptureReport *capture =
          findCaptureReport(*checkpointReport, expected);
      if (capture == nullptr || capture->profile != expected.profile ||
          capture->required != expected.required ||
          capture->checkpointId != checkpoint.id ||
          capture->checkpointFrame != checkpoint.frame ||
          capture->snapshot.target != expected.target ||
          capture->snapshot.profile != expected.profile ||
          capture->snapshot.required != expected.required ||
          capture->snapshot.artifactStem != expected.target ||
          capture->snapshot.captureFrameIndex != checkpoint.frame) {
        return Result<bool, std::string>::makeError(
            "autotest baseline candidate capture does not match manifest: " +
            checkpoint.id + "/" + expected.target);
      }
      const auto &snapshot = capture->snapshot;
      if (!snapshot.available || snapshot.actual.empty() ||
          snapshot.actualMetadata.empty() || snapshot.preview.empty() ||
          snapshot.actualHash.empty() || snapshot.capturePointVersion == 0u ||
          snapshot.kind.empty() || snapshot.format.empty() ||
          snapshot.colorSpace.empty() || snapshot.width == 0u ||
          snapshot.height == 0u || snapshot.origin.empty() ||
          (snapshot.status != "captured" && snapshot.status != "pass" &&
           snapshot.status != "fail")) {
        return Result<bool, std::string>::makeError(
            "autotest baseline candidate capture is incomplete: " +
            checkpoint.id + "/" + expected.target);
      }
      const auto *catalog =
          nuri::tools::snapshot::findSnapshotCaptureCatalogEntry(
              expected.target);
      if (catalog == nullptr ||
          snapshot.capturePointVersion != catalog->version ||
          snapshot.kind != nuri::tools::snapshot::renderCaptureValueKindName(
                               catalog->kind) ||
          !nuri::tools::snapshot::snapshotCompareProfileSupportsKind(
              snapshot.profile, catalog->kind)) {
        return Result<bool, std::string>::makeError(
            "autotest baseline candidate capture descriptor is unsupported: " +
            checkpoint.id + "/" + expected.target);
      }
      auto raw = candidateFile(report, snapshot.actual, "capture payload");
      auto sidecar =
          candidateFile(report, snapshot.actualMetadata, "capture metadata");
      if (raw.hasError() || sidecar.hasError()) {
        return Result<bool, std::string>::makeError(
            raw.hasError() ? raw.error() : sidecar.error());
      }
      const std::filesystem::path expectedDir =
          std::filesystem::path("checkpoints") /
          (checkpoint.id + "_frame_" + std::to_string(checkpoint.frame));
      const std::string extension = snapshot.actual.extension().string();
      if (snapshot.actual.parent_path() != expectedDir ||
          snapshot.actual.filename().stem() != expected.target ||
          (extension != ".png" && extension != ".exr" &&
           extension != ".nuri_tex") ||
          snapshot.actualMetadata !=
              expectedDir / (expected.target + ".json") ||
          (!snapshot.preview.empty() &&
           snapshot.preview !=
               expectedDir / (expected.target + "_preview.png"))) {
        return Result<bool, std::string>::makeError(
            "autotest baseline candidate artifact layout is incompatible: " +
            checkpoint.id + "/" + expected.target);
      }
      auto descriptor =
          nuri::tools::snapshot::readSnapshotArtifactMetadata(sidecar.value());
      if (descriptor.hasError() ||
          descriptor.value().target != expected.target ||
          descriptor.value().capturePointVersion !=
              snapshot.capturePointVersion ||
          descriptor.value().kind != snapshot.kind ||
          !descriptorFormatMatches(descriptor.value().format,
                                   snapshot.format) ||
          descriptor.value().profile != snapshot.profile ||
          descriptor.value().width != snapshot.width ||
          descriptor.value().height != snapshot.height ||
          descriptor.value().mip != snapshot.mip ||
          descriptor.value().layer != snapshot.layer ||
          descriptor.value().colorSpace != snapshot.colorSpace ||
          descriptor.value().origin != snapshot.origin ||
          descriptor.value().hash != snapshot.actualHash ||
          descriptor.value().payload !=
              raw.value().filename().generic_string()) {
        return Result<bool, std::string>::makeError(
            "autotest baseline candidate descriptor mismatch: " +
            checkpoint.id + "/" + expected.target);
      }
      if (!snapshot.preview.empty()) {
        auto preview =
            candidateFile(report, snapshot.preview, "capture preview");
        if (preview.hasError()) {
          return Result<bool, std::string>::makeError(preview.error());
        }
      }
    }
  }
  if (expectedCaptureCount == 0u ||
      report.checkpoints.size() != expectedCase.checkpoints.size()) {
    return Result<bool, std::string>::makeError(
        "autotest baseline candidate has no exact capture set to approve");
  }
  return Result<bool, std::string>::makeResult(true);
}

} // namespace

std::filesystem::path defaultAutotestBaselineRoot() {
  return nuri::tools::snapshot::defaultSnapshotBaselineRoot();
}

Result<bool, std::string>
verifyAutotestBaseline(const AutotestCase &expectedCase,
                       std::string_view baselineProfile,
                       const std::filesystem::path &baselineRootOverride) {
  auto valid = validateAutotestCase(expectedCase);
  if (valid.hasError()) {
    return Result<bool, std::string>::makeError("verifyAutotestBaseline: " +
                                                valid.error());
  }
  for (const auto &[value, field] :
       {std::pair{std::string_view(expectedCase.id), std::string_view("case")},
        std::pair{std::string_view(expectedCase.suite),
                  std::string_view("suite")},
        std::pair{baselineProfile, std::string_view("baseline profile")}}) {
    valid = validateAutotestIdentifier(value, field, false);
    if (valid.hasError()) {
      return Result<bool, std::string>::makeError("verifyAutotestBaseline: " +
                                                  valid.error());
    }
  }
  const std::filesystem::path baselineRoot = baselineRootOverride.empty()
                                                 ? defaultAutotestBaselineRoot()
                                                 : baselineRootOverride;
  const std::filesystem::path relativeCase =
      baselineRelativeCase(expectedCase, baselineProfile);
  auto caseDir =
      nuri::tools::core::resolvePathUnder(baselineRoot, relativeCase);
  if (caseDir.hasError()) {
    return Result<bool, std::string>::makeError("verifyAutotestBaseline: " +
                                                caseDir.error());
  }
  auto tree = readAutotestBaselineTree(baselineRoot, relativeCase);
  if (tree.hasError()) {
    return Result<bool, std::string>::makeError("verifyAutotestBaseline: " +
                                                tree.error());
  }
  std::map<std::string, const AutotestInspectedFile *> files;
  for (const AutotestInspectedFile &file : tree.value().files) {
    files.emplace(file.path, &file);
  }
  const auto approvalFile = files.find("approval.json");
  if (approvalFile == files.end()) {
    return Result<bool, std::string>::makeError(
        "verifyAutotestBaseline: approval.json is missing");
  }
  auto approvalPath = nuri::tools::core::resolvePathUnder(
      baselineRoot, relativeCase / "approval.json");
  if (approvalPath.hasError()) {
    return Result<bool, std::string>::makeError(
        "verifyAutotestBaseline: unsafe approval path");
  }
  auto approvalText =
      readBaselineTextFile(approvalPath.value(), "approval record");
  if (approvalText.hasError()) {
    return Result<bool, std::string>::makeError("verifyAutotestBaseline: " +
                                                approvalText.error());
  }
  auto approval = parseAutotestApproval(approvalText.value(), expectedCase,
                                        baselineProfile);
  if (approval.hasError()) {
    return Result<bool, std::string>::makeError("verifyAutotestBaseline: " +
                                                approval.error());
  }

  const auto metadataFile = files.find("metadata.json");
  if (metadataFile == files.end()) {
    return Result<bool, std::string>::makeError(
        "verifyAutotestBaseline: metadata.json is missing");
  }
  auto metadataPath = nuri::tools::core::resolvePathUnder(
      baselineRoot, relativeCase / "metadata.json");
  auto metadataText =
      metadataPath.hasError()
          ? Result<std::string, std::string>::makeError(metadataPath.error())
          : readBaselineTextFile(metadataPath.value(), "metadata");
  if (metadataText.hasError()) {
    return Result<bool, std::string>::makeError("verifyAutotestBaseline: " +
                                                metadataText.error());
  }
  JsonDocPtr metadataDoc(
      yyjson_read(metadataText.value().data(), metadataText.value().size(), 0u),
      &yyjson_doc_free);
  yyjson_val *metadata =
      metadataDoc ? yyjson_doc_get_root(metadataDoc.get()) : nullptr;
  if (!yyjson_is_obj(metadata)) {
    return Result<bool, std::string>::makeError(
        "verifyAutotestBaseline: metadata is invalid");
  }
  AutotestEnvironment metadataEnvironment{};
  metadataEnvironment.gpuBackend = readString(metadata, "resolvedBackend");
  metadataEnvironment.gpuDeviceName = readString(metadata, "gpuDeviceName");
  metadataEnvironment.gpuVendorId = readU32(metadata, "gpuVendorId");
  metadataEnvironment.gpuDeviceId = readU32(metadata, "gpuDeviceId");
  metadataEnvironment.gpuDriverVersion =
      readString(metadata, "gpuDriverVersion");
  metadataEnvironment.resolvedPresentMode =
      readString(metadata, "resolvedPresentMode");
  metadataEnvironment.resolvedWindowMode =
      readString(metadata, "resolvedWindowMode");
  auto metadataCompatibility = validateAutotestBaselineMetadataFile(
      expectedCase, metadataEnvironment, caseDir.value(), baselineProfile);
  if (metadataCompatibility.hasError() ||
      !metadataCompatibility.value().compatible) {
    return Result<bool, std::string>::makeError(
        "verifyAutotestBaseline: baseline metadata is incompatible" +
        (metadataCompatibility.hasError() ? ": " + metadataCompatibility.error()
         : metadataCompatibility.value().errors.empty()
             ? std::string{}
             : ": " + metadataCompatibility.value().errors.front()));
  }

  if (!std::binary_search(tree.value().directories.begin(),
                          tree.value().directories.end(), "history")) {
    return Result<bool, std::string>::makeError(
        "verifyAutotestBaseline: approval history is missing");
  }
  for (const std::string &directory : tree.value().directories) {
    if (directory.starts_with("history/")) {
      return Result<bool, std::string>::makeError(
          "verifyAutotestBaseline: approval history contains an unexpected "
          "directory: " +
          directory);
    }
  }

  std::map<std::string, const AutotestInspectedFile *> historyFiles;
  for (const AutotestInspectedFile &file : tree.value().files) {
    if (!file.path.starts_with("history/")) {
      continue;
    }
    const std::string name = file.path.substr(8u);
    if (name.empty() || name.find('/') != std::string::npos) {
      return Result<bool, std::string>::makeError(
          "verifyAutotestBaseline: approval history contains an unexpected "
          "entry: " +
          file.path);
    }
    historyFiles.emplace(name, &file);
  }

  struct ParsedHistoryPlan {
    AutotestBaselinePlan plan{};
    std::string planName{};
    std::string approvalName{};
  };
  std::map<std::string, ParsedHistoryPlan> plans;
  std::set<std::string> pairedApprovalNames;
  for (const auto &[name, inspected] : historyFiles) {
    (void)inspected;
    if (!name.ends_with("-plan.json")) {
      continue;
    }
    auto path = nuri::tools::core::resolvePathUnder(
        baselineRoot, relativeCase / "history" / name);
    auto text = path.hasError()
                    ? Result<std::string, std::string>::makeError(path.error())
                    : readBaselineTextFile(path.value(), "reviewed plan");
    if (text.hasError()) {
      return Result<bool, std::string>::makeError("verifyAutotestBaseline: " +
                                                  text.error());
    }
    auto plan =
        parseAutotestBaselinePlan(text.value(), expectedCase, baselineProfile);
    if (plan.hasError() ||
        !name.ends_with(plan.value().digest.substr(7u, 16u) + "-plan.json")) {
      return Result<bool, std::string>::makeError(
          "verifyAutotestBaseline: reviewed plan history is invalid: " + name);
    }
    const std::string approvalName =
        name.substr(0u, name.size() - std::string("plan.json").size()) +
        "approval.json";
    const auto paired = historyFiles.find(approvalName);
    if (paired == historyFiles.end()) {
      return Result<bool, std::string>::makeError(
          "verifyAutotestBaseline: approval history pair is missing for: " +
          name);
    }
    auto pairedPath = nuri::tools::core::resolvePathUnder(
        baselineRoot, relativeCase / "history" / approvalName);
    auto pairedText =
        pairedPath.hasError()
            ? Result<std::string, std::string>::makeError(pairedPath.error())
            : readBaselineTextFile(pairedPath.value(), "approval history");
    auto pairedApproval =
        pairedText.hasError()
            ? Result<AutotestApprovalDocument, std::string>::makeError(
                  pairedText.error())
            : parseAutotestApproval(pairedText.value(), expectedCase,
                                    baselineProfile);
    if (pairedApproval.hasError() ||
        pairedApproval.value().planDigest != plan.value().digest ||
        pairedApproval.value().reason != plan.value().reason ||
        pairedApproval.value().actor != plan.value().actor ||
        pairedApproval.value().sourceCommit != plan.value().sourceCommit ||
        pairedApproval.value().sourceReportDigest !=
            plan.value().sourceReportDigest ||
        pairedApproval.value().carriedHistoryDigest !=
            plan.value().historyDigest ||
        pairedApproval.value().carriedHistoryFileCount !=
            plan.value().historyFileCount) {
      return Result<bool, std::string>::makeError(
          "verifyAutotestBaseline: approval history does not match reviewed "
          "plan: " +
          name);
    }
    const std::string parsedPlanDigest = plan.value().digest;
    if (!plans
             .emplace(parsedPlanDigest,
                      ParsedHistoryPlan{.plan = std::move(plan.value()),
                                        .planName = name,
                                        .approvalName = approvalName})
             .second) {
      return Result<bool, std::string>::makeError(
          "verifyAutotestBaseline: reviewed plan history is ambiguous");
    }
    pairedApprovalNames.insert(approvalName);
  }
  if (plans.empty()) {
    return Result<bool, std::string>::makeError(
        "verifyAutotestBaseline: reviewed plan history is missing");
  }

  std::vector<std::string> legacyNames;
  for (const auto &[name, inspected] : historyFiles) {
    if (name.ends_with("-plan.json") || pairedApprovalNames.contains(name)) {
      continue;
    }
    if (!name.starts_with("legacy-") || !name.ends_with(".approval.json") ||
        name !=
            "legacy-" + inspected->digest.substr(7u, 16u) + ".approval.json") {
      return Result<bool, std::string>::makeError(
          "verifyAutotestBaseline: unexpected approval history entry: " + name);
    }
    auto legacyPath = nuri::tools::core::resolvePathUnder(
        baselineRoot, relativeCase / "history" / name);
    auto legacyText =
        legacyPath.hasError()
            ? Result<std::string, std::string>::makeError(legacyPath.error())
            : readBaselineTextFile(legacyPath.value(), "legacy approval");
    auto legacyApproval =
        legacyText.hasError()
            ? Result<AutotestApprovalDocument, std::string>::makeError(
                  legacyText.error())
            : parseAutotestApproval(legacyText.value(), expectedCase,
                                    baselineProfile);
    if (legacyApproval.hasError()) {
      return Result<bool, std::string>::makeError(
          "verifyAutotestBaseline: invalid legacy approval history: " + name);
    }
    const bool duplicatesPairedApproval =
        std::any_of(pairedApprovalNames.begin(), pairedApprovalNames.end(),
                    [&](const std::string &pairedName) {
                      const auto paired = historyFiles.find(pairedName);
                      return paired != historyFiles.end() &&
                             paired->second->digest == inspected->digest;
                    });
    if (!duplicatesPairedApproval) {
      return Result<bool, std::string>::makeError(
          "verifyAutotestBaseline: legacy approval has no reviewed history "
          "source: " +
          name);
    }
    legacyNames.push_back(name);
  }

  const auto currentPlan = plans.find(approval.value().planDigest);
  if (currentPlan == plans.end()) {
    return Result<bool, std::string>::makeError(
        "verifyAutotestBaseline: current reviewed plan history is missing");
  }
  const AutotestBaselinePlan &plan = currentPlan->second.plan;
  if (approval.value().reason != plan.reason ||
      approval.value().actor != plan.actor ||
      approval.value().sourceCommit != plan.sourceCommit ||
      approval.value().sourceReportDigest != plan.sourceReportDigest ||
      approval.value().carriedHistoryDigest != plan.historyDigest ||
      approval.value().carriedHistoryFileCount != plan.historyFileCount) {
    return Result<bool, std::string>::makeError(
        "verifyAutotestBaseline: current approval does not match reviewed "
        "plan");
  }
  const auto currentApprovalHistory =
      historyFiles.find(currentPlan->second.approvalName);
  if (currentApprovalHistory == historyFiles.end() ||
      currentApprovalHistory->second->digest != approvalFile->second->digest) {
    return Result<bool, std::string>::makeError(
        "verifyAutotestBaseline: current approval history digest mismatch");
  }

  const auto carriedMatches = [&](std::string_view excludedLegacy) {
    std::vector<std::pair<std::string, std::string>> carried;
    for (const auto &[name, inspected] : historyFiles) {
      if (name == currentPlan->second.planName ||
          name == currentPlan->second.approvalName || name == excludedLegacy) {
        continue;
      }
      carried.emplace_back(name, inspected->digest.substr(7u));
    }
    auto state = makeAutotestHistoryState(std::move(carried));
    return !state.hasError() && state.value().digest == plan.historyDigest &&
           state.value().fileCount == plan.historyFileCount;
  };
  size_t carriedMatchesCount = carriedMatches({}) ? 1u : 0u;
  for (const std::string &legacyName : legacyNames) {
    carriedMatchesCount += carriedMatches(legacyName) ? 1u : 0u;
  }
  if (carriedMatchesCount != 1u) {
    return Result<bool, std::string>::makeError(
        "verifyAutotestBaseline: carried approval history digest mismatch");
  }

  std::set<std::string> plannedFiles;
  std::set<std::string> plannedDirectories{"history"};
  for (const AutotestBaselinePlanEntry &entry : plan.entries) {
    if (entry.path.starts_with("history/")) {
      return Result<bool, std::string>::makeError(
          "verifyAutotestBaseline: reviewed plan aliases approval history");
    }
    auto resolved = nuri::tools::core::resolvePathUnder(
        baselineRoot, relativeCase / entry.path);
    if (resolved.hasError()) {
      return Result<bool, std::string>::makeError(
          "verifyAutotestBaseline: reviewed plan path is unsafe: " +
          entry.path);
    }
    if (entry.operation == "remove") {
      std::error_code existsError;
      if (std::filesystem::exists(resolved.value(), existsError) ||
          existsError) {
        return Result<bool, std::string>::makeError(
            "verifyAutotestBaseline: removed entry is still present: " +
            entry.path);
      }
      continue;
    }
    plannedFiles.insert(entry.path);
    for (std::filesystem::path parent =
             std::filesystem::path(entry.path).parent_path();
         !parent.empty(); parent = parent.parent_path()) {
      plannedDirectories.insert(parent.generic_string());
    }
    const auto actual = files.find(entry.path);
    if (actual == files.end()) {
      return Result<bool, std::string>::makeError(
          "verifyAutotestBaseline: approved entry is missing: " + entry.path);
    }
    if (entry.path == "approval.json") {
      if (entry.sourceDigest != plan.sourceReportDigest) {
        return Result<bool, std::string>::makeError(
            "verifyAutotestBaseline: approval provenance digest mismatch");
      }
      continue;
    }
    if (actual->second->digest != entry.sourceDigest) {
      return Result<bool, std::string>::makeError(
          "verifyAutotestBaseline: approved entry digest mismatch: " +
          entry.path);
    }
  }
  for (const AutotestInspectedFile &file : tree.value().files) {
    if (!file.path.starts_with("history/") &&
        !plannedFiles.contains(file.path)) {
      return Result<bool, std::string>::makeError(
          "verifyAutotestBaseline: unreviewed baseline file: " + file.path);
    }
  }
  for (const std::string &directory : tree.value().directories) {
    if (!directory.starts_with("history/") &&
        !plannedDirectories.contains(directory)) {
      return Result<bool, std::string>::makeError(
          "verifyAutotestBaseline: unreviewed baseline directory: " +
          directory);
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<std::string, std::string>
inspectAutotestBaseline(const AutotestCase &expectedCase,
                        std::string_view baselineProfile,
                        const std::filesystem::path &baselineRootOverride) {
  auto valid = validateAutotestCase(expectedCase);
  if (valid.hasError()) {
    return Result<std::string, std::string>::makeError(
        "inspectAutotestBaseline: " + valid.error());
  }
  for (const auto &[value, field] :
       {std::pair{std::string_view(expectedCase.id), std::string_view("case")},
        std::pair{std::string_view(expectedCase.suite),
                  std::string_view("suite")},
        std::pair{baselineProfile, std::string_view("baseline profile")}}) {
    valid = validateAutotestIdentifier(value, field, false);
    if (valid.hasError()) {
      return Result<std::string, std::string>::makeError(
          "inspectAutotestBaseline: " + valid.error());
    }
  }
  const std::filesystem::path baselineRoot = baselineRootOverride.empty()
                                                 ? defaultAutotestBaselineRoot()
                                                 : baselineRootOverride;
  const std::filesystem::path relativeCase =
      baselineRelativeCase(expectedCase, baselineProfile);
  auto caseDir =
      nuri::tools::core::resolvePathUnder(baselineRoot, relativeCase);
  if (caseDir.hasError()) {
    return Result<std::string, std::string>::makeError(
        "inspectAutotestBaseline: " + caseDir.error());
  }
  std::error_code error;
  const bool exists = std::filesystem::exists(caseDir.value(), error);
  const bool isDirectory =
      exists && std::filesystem::is_directory(caseDir.value(), error);
  if (error) {
    return Result<std::string, std::string>::makeError(
        "inspectAutotestBaseline: failed to inspect baseline: " +
        error.message());
  }
  auto verified =
      verifyAutotestBaseline(expectedCase, baselineProfile, baselineRoot);
  auto tree = isDirectory
                  ? readAutotestBaselineTree(baselineRoot, relativeCase)
                  : Result<AutotestBaselineTree, std::string>::makeResult(
                        AutotestBaselineTree{});
  std::string diagnostic = verified.hasError() ? verified.error() : "";
  if (tree.hasError() && diagnostic.empty()) {
    diagnostic += tree.error();
  }
  const bool inspectionVerified = !verified.hasError() && !tree.hasError();
  AutotestApprovalDocument approvalSummary{};
  std::string approvalDigest;
  bool approvalSummaryValid = false;
  if (!tree.hasError()) {
    const auto found =
        std::find_if(tree.value().files.begin(), tree.value().files.end(),
                     [](const AutotestInspectedFile &file) {
                       return file.path == "approval.json";
                     });
    if (found != tree.value().files.end()) {
      auto path = nuri::tools::core::resolvePathUnder(
          baselineRoot, relativeCase / "approval.json");
      auto text =
          path.hasError()
              ? Result<std::string, std::string>::makeError(path.error())
              : readBaselineTextFile(path.value(), "approval record");
      auto parsed =
          text.hasError()
              ? Result<AutotestApprovalDocument, std::string>::makeError(
                    text.error())
              : parseAutotestApproval(text.value(), expectedCase,
                                      baselineProfile);
      if (!parsed.hasError()) {
        approvalSummary = std::move(parsed.value());
        approvalDigest = found->digest;
        approvalSummaryValid = true;
      }
    }
  }

  using JsonMutDoc =
      std::unique_ptr<yyjson_mut_doc, decltype(&yyjson_mut_doc_free)>;
  JsonMutDoc doc(yyjson_mut_doc_new(nullptr), &yyjson_mut_doc_free);
  if (!doc) {
    return Result<std::string, std::string>::makeError(
        "inspectAutotestBaseline: allocation failed");
  }
  yyjson_mut_val *root = yyjson_mut_obj(doc.get());
  yyjson_mut_doc_set_root(doc.get(), root);
  yyjson_mut_obj_add_uint(doc.get(), root, "schemaVersion", 1u);
  yyjson_mut_obj_add_strcpy(doc.get(), root, "kind",
                            "nuri.autotest.baseline_inspection");
  yyjson_mut_obj_add_strcpy(doc.get(), root, "caseId", expectedCase.id.c_str());
  yyjson_mut_obj_add_strcpy(doc.get(), root, "suite",
                            expectedCase.suite.c_str());
  yyjson_mut_obj_add_strncpy(doc.get(), root, "profileId",
                             baselineProfile.data(), baselineProfile.size());
  yyjson_mut_obj_add_bool(doc.get(), root, "exists", exists);
  yyjson_mut_obj_add_bool(doc.get(), root, "verified", inspectionVerified);
  yyjson_mut_obj_add_strcpy(doc.get(), root, "diagnostic", diagnostic.c_str());
  yyjson_mut_val *approvalJson = yyjson_mut_obj(doc.get());
  yyjson_mut_obj_add_bool(doc.get(), approvalJson, "valid",
                          approvalSummaryValid);
  yyjson_mut_obj_add_strcpy(doc.get(), approvalJson, "digest",
                            approvalDigest.c_str());
  yyjson_mut_obj_add_strcpy(doc.get(), approvalJson, "planDigest",
                            approvalSummary.planDigest.c_str());
  yyjson_mut_obj_add_strcpy(doc.get(), approvalJson, "reason",
                            approvalSummary.reason.c_str());
  yyjson_mut_obj_add_strcpy(doc.get(), approvalJson, "actor",
                            approvalSummary.actor.c_str());
  yyjson_mut_obj_add_strcpy(doc.get(), approvalJson, "approvedAtUtc",
                            approvalSummary.approvedAtUtc.c_str());
  yyjson_mut_obj_add_strcpy(doc.get(), approvalJson, "sourceCommit",
                            approvalSummary.sourceCommit.c_str());
  yyjson_mut_obj_add_strcpy(doc.get(), approvalJson, "sourceReportDigest",
                            approvalSummary.sourceReportDigest.c_str());
  yyjson_mut_obj_add_strcpy(doc.get(), approvalJson, "carriedHistoryDigest",
                            approvalSummary.carriedHistoryDigest.c_str());
  yyjson_mut_obj_add_uint(doc.get(), approvalJson, "carriedHistoryFileCount",
                          approvalSummary.carriedHistoryFileCount);
  yyjson_mut_obj_add_val(doc.get(), root, "approval", approvalJson);
  yyjson_mut_val *filesJson = yyjson_mut_arr(doc.get());
  uintmax_t totalBytes = 0u;
  if (!tree.hasError()) {
    for (const AutotestInspectedFile &file : tree.value().files) {
      yyjson_mut_val *fileJson = yyjson_mut_obj(doc.get());
      yyjson_mut_obj_add_strcpy(doc.get(), fileJson, "path", file.path.c_str());
      yyjson_mut_obj_add_uint(doc.get(), fileJson, "size", file.size);
      yyjson_mut_obj_add_strcpy(doc.get(), fileJson, "digest",
                                file.digest.c_str());
      yyjson_mut_arr_add_val(filesJson, fileJson);
      totalBytes += file.size;
    }
  }
  yyjson_mut_obj_add_uint(doc.get(), root, "fileCount",
                          tree.hasError() ? 0u : tree.value().files.size());
  yyjson_mut_obj_add_uint(doc.get(), root, "totalBytes", totalBytes);
  yyjson_mut_obj_add_val(doc.get(), root, "files", filesJson);
  size_t length = 0u;
  char *json = yyjson_mut_write_opts(doc.get(), YYJSON_WRITE_PRETTY, nullptr,
                                     &length, nullptr);
  if (json == nullptr) {
    return Result<std::string, std::string>::makeError(
        "inspectAutotestBaseline: serialization failed");
  }
  std::string text(json, length);
  std::free(json);
  return Result<std::string, std::string>::makeResult(std::move(text));
}

Result<AutotestBaselinePlan, std::string> planAutotestBaselines(
    const AutotestCase &expectedCase, const AutotestReport &report,
    std::string_view baselineProfile, std::string_view reason,
    std::string_view actor, const AutotestBaselineApprovalOptions &options) {
  if (!hasNonWhitespace(reason) || !hasNonWhitespace(actor)) {
    return Result<AutotestBaselinePlan, std::string>::makeError(
        "planAutotestBaselines: reason and actor are required");
  }
  auto valid = validateApprovalCandidate(expectedCase, report, baselineProfile);
  if (valid.hasError()) {
    return Result<AutotestBaselinePlan, std::string>::makeError(
        "planAutotestBaselines: " + valid.error());
  }
  auto reportJson = writeAutotestReportJson(report);
  if (reportJson.hasError()) {
    return Result<AutotestBaselinePlan, std::string>::makeError(
        reportJson.error());
  }

  AutotestBaselinePlan plan{};
  plan.caseId = expectedCase.id;
  plan.suite = expectedCase.suite;
  plan.profileId = std::string(baselineProfile);
  plan.reason = std::string(reason);
  plan.actor = std::string(actor);
  plan.sourceCommit = report.environment.commitHash;
  plan.sourceReportDigest =
      "sha256:" + nuri::tools::core::sha256Hex(std::as_bytes(std::span(
                      reportJson.value().data(), reportJson.value().size())));

  const std::filesystem::path baselineRoot = baselineRootFor(options);
  const std::filesystem::path relativeCase =
      baselineRelativeCase(expectedCase, baselineProfile);
  auto targetCase =
      nuri::tools::core::resolvePathUnder(baselineRoot, relativeCase);
  if (targetCase.hasError()) {
    return Result<AutotestBaselinePlan, std::string>::makeError(
        "planAutotestBaselines: " + targetCase.error());
  }

  std::set<std::string> plannedPaths;
  const auto appendCandidate =
      [&](const std::filesystem::path &relative) -> Result<void, std::string> {
    auto source = candidateFile(report, relative, "planned artifact");
    if (source.hasError()) {
      return Result<void, std::string>::makeError(source.error());
    }
    const std::string destination = relative.generic_string();
    if (!plannedPaths.insert(destination).second) {
      return Result<void, std::string>::makeError(
          "duplicate planned autotest baseline path: " + destination);
    }
    auto target = nuri::tools::core::resolvePathUnder(baselineRoot,
                                                      relativeCase / relative);
    if (target.hasError()) {
      return Result<void, std::string>::makeError(target.error());
    }
    auto sourceDigest = nuri::tools::core::sha256File(source.value());
    if (sourceDigest.hasError()) {
      return Result<void, std::string>::makeError(sourceDigest.error());
    }
    AutotestBaselinePlanEntry entry{.path = destination,
                                    .operation = "add",
                                    .sourceDigest =
                                        "sha256:" + sourceDigest.value()};
    std::error_code targetError;
    const bool targetExists =
        std::filesystem::exists(target.value(), targetError);
    if (targetError) {
      return Result<void, std::string>::makeError(
          "failed to inspect current autotest baseline artifact: " +
          targetError.message());
    }
    if (targetExists &&
        !std::filesystem::is_regular_file(target.value(), targetError)) {
      return Result<void, std::string>::makeError(
          "current autotest baseline path is not a regular file: " +
          destination +
          (targetError ? " (" + targetError.message() + ")" : std::string{}));
    }
    if (targetExists) {
      auto previousDigest = nuri::tools::core::sha256File(target.value());
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

  auto appended = appendCandidate("metadata.json");
  if (appended.hasError()) {
    return Result<AutotestBaselinePlan, std::string>::makeError(
        appended.error());
  }
  for (const AutotestCheckpointReport &checkpoint : report.checkpoints) {
    for (const AutotestCaptureReport &capture : checkpoint.captures) {
      for (const std::filesystem::path &relative :
           {capture.snapshot.actual, capture.snapshot.actualMetadata,
            capture.snapshot.preview}) {
        if (relative.empty()) {
          continue;
        }
        appended = appendCandidate(relative);
        if (appended.hasError()) {
          return Result<AutotestBaselinePlan, std::string>::makeError(
              appended.error());
        }
      }
    }
  }

  AutotestBaselinePlanEntry approvalEntry{.path = "approval.json",
                                          .operation = "add",
                                          .sourceDigest =
                                              plan.sourceReportDigest};
  auto previousApproval = nuri::tools::core::resolvePathUnder(
      baselineRoot, relativeCase / "approval.json");
  if (previousApproval.hasError()) {
    return Result<AutotestBaselinePlan, std::string>::makeError(
        previousApproval.error());
  }
  std::error_code approvalError;
  const bool approvalExists =
      std::filesystem::exists(previousApproval.value(), approvalError);
  if (approvalError) {
    return Result<AutotestBaselinePlan, std::string>::makeError(
        "failed to inspect current autotest baseline approval: " +
        approvalError.message());
  }
  if (approvalExists && !std::filesystem::is_regular_file(
                            previousApproval.value(), approvalError)) {
    return Result<AutotestBaselinePlan, std::string>::makeError(
        "current autotest baseline approval path is not a regular file" +
        (approvalError ? " (" + approvalError.message() + ")" : std::string{}));
  }
  if (approvalExists) {
    auto previousDigest =
        nuri::tools::core::sha256File(previousApproval.value());
    if (previousDigest.hasError()) {
      return Result<AutotestBaselinePlan, std::string>::makeError(
          previousDigest.error());
    }
    approvalEntry.previousDigest = "sha256:" + previousDigest.value();
    approvalEntry.operation = "replace";
  }
  plannedPaths.insert(approvalEntry.path);
  plan.entries.push_back(std::move(approvalEntry));

  auto history =
      readAutotestHistoryState(baselineRoot, relativeCase / "history");
  if (history.hasError()) {
    return Result<AutotestBaselinePlan, std::string>::makeError(
        history.error());
  }
  plan.historyDigest = history.value().digest;
  plan.historyFileCount = history.value().fileCount;

  std::error_code error;
  const bool targetExists = std::filesystem::exists(targetCase.value(), error);
  if (error) {
    return Result<AutotestBaselinePlan, std::string>::makeError(
        "failed to inspect autotest baseline target: " + error.message());
  }
  if (targetExists &&
      !std::filesystem::is_directory(targetCase.value(), error)) {
    return Result<AutotestBaselinePlan, std::string>::makeError(
        "autotest baseline target is not a directory" +
        (error ? " (" + error.message() + ")" : std::string{}));
  }
  if (error) {
    return Result<AutotestBaselinePlan, std::string>::makeError(
        "failed to inspect autotest baseline target: " + error.message());
  }
  if (targetExists) {
    for (const auto &entry : std::filesystem::recursive_directory_iterator(
             targetCase.value(), error)) {
      if (error) {
        return Result<AutotestBaselinePlan, std::string>::makeError(
            "failed to inspect current autotest baseline: " + error.message());
      }
      const std::filesystem::path relative =
          std::filesystem::relative(entry.path(), targetCase.value(), error);
      if (error) {
        return Result<AutotestBaselinePlan, std::string>::makeError(
            "failed to relativize current autotest baseline: " +
            error.message());
      }
      auto safeEntry = nuri::tools::core::resolvePathUnder(
          baselineRoot, relativeCase / relative);
      if (safeEntry.hasError()) {
        return Result<AutotestBaselinePlan, std::string>::makeError(
            safeEntry.error());
      }
      if (*relative.begin() == "history") {
        continue;
      }
      if (plannedPaths.contains(relative.generic_string())) {
        continue;
      }
      if (entry.is_directory(error)) {
        if (error) {
          return Result<AutotestBaselinePlan, std::string>::makeError(
              "failed to inspect current autotest baseline entry: " +
              error.message());
        }
        continue;
      }
      if (!entry.is_regular_file(error) || error) {
        return Result<AutotestBaselinePlan, std::string>::makeError(
            "current autotest baseline contains an unsupported entry: " +
            relative.generic_string() +
            (error ? " (" + error.message() + ")" : std::string{}));
      }
      auto digest = nuri::tools::core::sha256File(safeEntry.value());
      if (digest.hasError()) {
        return Result<AutotestBaselinePlan, std::string>::makeError(
            digest.error());
      }
      plan.entries.push_back(AutotestBaselinePlanEntry{
          .path = relative.generic_string(),
          .operation = "remove",
          .previousDigest = "sha256:" + digest.value()});
    }
    if (error) {
      return Result<AutotestBaselinePlan, std::string>::makeError(
          "failed to inspect current autotest baseline: " + error.message());
    }
  }

  std::sort(plan.entries.begin(), plan.entries.end(),
            [](const auto &left, const auto &right) {
              return left.path < right.path;
            });
  plan.digest = computeAutotestBaselinePlanDigest(plan);
  return Result<AutotestBaselinePlan, std::string>::makeResult(std::move(plan));
}

Result<std::string, std::string>
writeAutotestBaselinePlanJson(const AutotestBaselinePlan &plan) {
  std::ostringstream out;
  out << "{\n  \"schemaVersion\": 1,\n"
      << "  \"kind\": \"nuri.autotest.baseline_plan\",\n"
      << "  \"caseId\": \"" << jsonEscape(plan.caseId) << "\",\n"
      << "  \"suite\": \"" << jsonEscape(plan.suite) << "\",\n"
      << "  \"profileId\": \"" << jsonEscape(plan.profileId) << "\",\n"
      << "  \"reason\": \"" << jsonEscape(plan.reason) << "\",\n"
      << "  \"actor\": \"" << jsonEscape(plan.actor) << "\",\n"
      << "  \"sourceCommit\": \"" << jsonEscape(plan.sourceCommit)
      << "\",\n  \"sourceReportDigest\": \""
      << jsonEscape(plan.sourceReportDigest) << "\",\n  \"historyDigest\": \""
      << jsonEscape(plan.historyDigest)
      << "\",\n  \"historyFileCount\": " << plan.historyFileCount
      << ",\n  \"planDigest\": \"" << jsonEscape(plan.digest)
      << "\",\n  \"entries\": [\n";
  for (size_t index = 0u; index < plan.entries.size(); ++index) {
    const AutotestBaselinePlanEntry &entry = plan.entries[index];
    if (index != 0u) {
      out << ",\n";
    }
    out << "    {\"path\": \"" << jsonEscape(entry.path)
        << "\", \"operation\": \"" << jsonEscape(entry.operation)
        << "\", \"previousDigest\": \"" << jsonEscape(entry.previousDigest)
        << "\", \"sourceDigest\": \"" << jsonEscape(entry.sourceDigest)
        << "\"}";
  }
  out << "\n  ]\n}\n";
  return Result<std::string, std::string>::makeResult(out.str());
}

Result<bool, std::string> approveAutotestBaselines(
    const AutotestCase &expectedCase, const AutotestReport &report,
    std::string_view baselineProfile, std::string_view reason,
    std::string_view confirmPlanDigest, std::string_view actor,
    const AutotestBaselineApprovalOptions &options) {
  auto plan = planAutotestBaselines(expectedCase, report, baselineProfile,
                                    reason, actor, options);
  if (plan.hasError()) {
    return Result<bool, std::string>::makeError("approveAutotestBaselines: " +
                                                plan.error());
  }
  if (confirmPlanDigest.empty() || confirmPlanDigest != plan.value().digest) {
    return Result<bool, std::string>::makeError(
        "approveAutotestBaselines: --confirm-plan must match the reviewed plan "
        "digest " +
        plan.value().digest);
  }
  auto planJson = writeAutotestBaselinePlanJson(plan.value());
  if (planJson.hasError()) {
    return Result<bool, std::string>::makeError(planJson.error());
  }

  const std::filesystem::path baselineRoot = baselineRootFor(options);
  const std::filesystem::path relativeCase =
      baselineRelativeCase(expectedCase, baselineProfile);
  const std::string nonce =
      utcTimestampForPath() + "-" + plan.value().digest.substr(7u, 16u);
  const std::filesystem::path stageRelative =
      relativeCase.parent_path() / (expectedCase.id + ".stage-" + nonce);
  const std::filesystem::path backupRelative =
      relativeCase.parent_path() / (expectedCase.id + ".backup-" + nonce);
  auto target = nuri::tools::core::resolvePathUnder(baselineRoot, relativeCase);
  auto stage = nuri::tools::core::resolvePathUnder(baselineRoot, stageRelative);
  auto backup =
      nuri::tools::core::resolvePathUnder(baselineRoot, backupRelative);
  if (target.hasError() || stage.hasError() || backup.hasError()) {
    return Result<bool, std::string>::makeError(
        "approveAutotestBaselines: staging containment validation failed");
  }

  std::error_code error;
  if (std::filesystem::exists(stage.value(), error) || error ||
      std::filesystem::exists(backup.value(), error) || error) {
    return Result<bool, std::string>::makeError(
        "approveAutotestBaselines: staging or backup path already exists");
  }
  std::filesystem::create_directories(stage.value(), error);
  if (error) {
    return Result<bool, std::string>::makeError(
        "approveAutotestBaselines: failed to create staging directory: " +
        error.message());
  }
  const auto cleanupStage = [&]() {
    (void)nuri::tools::core::removeTreeUnder(baselineRoot, stageRelative);
  };

  for (const AutotestBaselinePlanEntry &entry : plan.value().entries) {
    if (entry.sourceDigest.empty() || entry.path == "approval.json") {
      continue;
    }
    const std::filesystem::path relative(entry.path);
    auto source = candidateFile(report, relative, "promotion artifact");
    if (source.hasError()) {
      cleanupStage();
      return Result<bool, std::string>::makeError(source.error());
    }
    auto sourceDigest = nuri::tools::core::sha256File(source.value());
    if (sourceDigest.hasError() ||
        "sha256:" + sourceDigest.value() != entry.sourceDigest) {
      cleanupStage();
      return Result<bool, std::string>::makeError(
          "approveAutotestBaselines: candidate changed after plan review: " +
          entry.path);
    }
    auto destination = nuri::tools::core::resolvePathUnder(
        baselineRoot, stageRelative / relative);
    if (destination.hasError()) {
      cleanupStage();
      return Result<bool, std::string>::makeError(destination.error());
    }
    std::filesystem::create_directories(destination.value().parent_path(),
                                        error);
    if (!error) {
      std::filesystem::copy_file(source.value(), destination.value(),
                                 std::filesystem::copy_options::none, error);
    }
    if (error) {
      cleanupStage();
      return Result<bool, std::string>::makeError(
          "approveAutotestBaselines: failed to stage candidate artifact: " +
          error.message());
    }
    auto stagedDigest = nuri::tools::core::sha256File(destination.value());
    if (stagedDigest.hasError() ||
        "sha256:" + stagedDigest.value() != entry.sourceDigest) {
      cleanupStage();
      return Result<bool, std::string>::makeError(
          "approveAutotestBaselines: staged artifact digest mismatch: " +
          entry.path);
    }
  }

  auto stagedHistory = nuri::tools::core::resolvePathUnder(
      baselineRoot, stageRelative / "history");
  auto previousHistory = nuri::tools::core::resolvePathUnder(
      baselineRoot, relativeCase / "history");
  if (stagedHistory.hasError() || previousHistory.hasError()) {
    cleanupStage();
    return Result<bool, std::string>::makeError(
        "approveAutotestBaselines: history containment validation failed");
  }
  std::filesystem::create_directories(stagedHistory.value(), error);
  if (error) {
    cleanupStage();
    return Result<bool, std::string>::makeError(
        "approveAutotestBaselines: failed to create staged history: " +
        error.message());
  }
  const bool previousHistoryExists =
      std::filesystem::exists(previousHistory.value(), error);
  if (error) {
    cleanupStage();
    return Result<bool, std::string>::makeError(
        "approveAutotestBaselines: failed to inspect prior history: " +
        error.message());
  }
  if (previousHistoryExists &&
      !std::filesystem::is_directory(previousHistory.value(), error)) {
    cleanupStage();
    return Result<bool, std::string>::makeError(
        "approveAutotestBaselines: prior history is not a directory" +
        (error ? " (" + error.message() + ")" : std::string{}));
  }
  if (error) {
    cleanupStage();
    return Result<bool, std::string>::makeError(
        "approveAutotestBaselines: failed to inspect prior history: " +
        error.message());
  }
  if (previousHistoryExists) {
    for (const auto &entry : std::filesystem::recursive_directory_iterator(
             previousHistory.value(), error)) {
      if (error) {
        cleanupStage();
        return Result<bool, std::string>::makeError(
            "approveAutotestBaselines: failed to inspect approval history: " +
            error.message());
      }
      const std::filesystem::path relative = std::filesystem::relative(
          entry.path(), previousHistory.value(), error);
      if (error) {
        cleanupStage();
        return Result<bool, std::string>::makeError(
            "approveAutotestBaselines: failed to relativize approval "
            "history: " +
            error.message());
      }
      auto safeSource = nuri::tools::core::resolvePathUnder(
          baselineRoot, relativeCase / "history" / relative);
      auto safeDestination = nuri::tools::core::resolvePathUnder(
          baselineRoot, stageRelative / "history" / relative);
      if (safeSource.hasError() || safeDestination.hasError()) {
        cleanupStage();
        return Result<bool, std::string>::makeError(
            "approveAutotestBaselines: unsafe approval history entry");
      }
      if (entry.is_directory(error)) {
        std::filesystem::create_directories(safeDestination.value(), error);
      } else if (entry.is_regular_file(error)) {
        std::filesystem::create_directories(
            safeDestination.value().parent_path(), error);
        if (!error) {
          std::filesystem::copy_file(
              safeSource.value(), safeDestination.value(),
              std::filesystem::copy_options::none, error);
        }
        if (!error) {
          auto sourceDigest = nuri::tools::core::sha256File(safeSource.value());
          auto destinationDigest =
              nuri::tools::core::sha256File(safeDestination.value());
          if (sourceDigest.hasError() || destinationDigest.hasError() ||
              sourceDigest.value() != destinationDigest.value()) {
            error = std::make_error_code(std::errc::io_error);
          }
        }
      } else {
        error = std::make_error_code(std::errc::operation_not_supported);
      }
      if (error) {
        cleanupStage();
        return Result<bool, std::string>::makeError(
            "approveAutotestBaselines: failed to carry immutable history: " +
            error.message());
      }
    }
    if (error) {
      cleanupStage();
      return Result<bool, std::string>::makeError(
          "approveAutotestBaselines: failed to inspect approval history: " +
          error.message());
    }
  }
  auto stagedHistoryState =
      readAutotestHistoryState(baselineRoot, stageRelative / "history");
  if (stagedHistoryState.hasError() ||
      stagedHistoryState.value().digest != plan.value().historyDigest ||
      stagedHistoryState.value().fileCount != plan.value().historyFileCount) {
    cleanupStage();
    return Result<bool, std::string>::makeError(
        "approveAutotestBaselines: carried history changed after plan review");
  }

  auto previousApproval = nuri::tools::core::resolvePathUnder(
      baselineRoot, relativeCase / "approval.json");
  if (previousApproval.hasError()) {
    cleanupStage();
    return Result<bool, std::string>::makeError(previousApproval.error());
  }
  const bool hasPreviousApproval =
      std::filesystem::exists(previousApproval.value(), error);
  if (error) {
    cleanupStage();
    return Result<bool, std::string>::makeError(
        "approveAutotestBaselines: failed to inspect previous approval: " +
        error.message());
  }
  if (hasPreviousApproval &&
      !std::filesystem::is_regular_file(previousApproval.value(), error)) {
    cleanupStage();
    return Result<bool, std::string>::makeError(
        "approveAutotestBaselines: previous approval is not a regular file" +
        (error ? " (" + error.message() + ")" : std::string{}));
  }
  if (error) {
    cleanupStage();
    return Result<bool, std::string>::makeError(
        "approveAutotestBaselines: failed to inspect previous approval: " +
        error.message());
  }
  if (hasPreviousApproval) {
    auto digest = nuri::tools::core::sha256File(previousApproval.value());
    if (digest.hasError()) {
      cleanupStage();
      return Result<bool, std::string>::makeError(digest.error());
    }
    const std::filesystem::path legacyName =
        "legacy-" + digest.value().substr(0u, 16u) + ".approval.json";
    auto legacy = nuri::tools::core::resolvePathUnder(
        baselineRoot, stageRelative / "history" / legacyName);
    if (legacy.hasError()) {
      cleanupStage();
      return Result<bool, std::string>::makeError(legacy.error());
    }
    const bool hasLegacy = std::filesystem::exists(legacy.value(), error);
    if (error) {
      cleanupStage();
      return Result<bool, std::string>::makeError(
          "approveAutotestBaselines: failed to inspect prior approval "
          "history: " +
          error.message());
    }
    if (!hasLegacy) {
      std::filesystem::copy_file(previousApproval.value(), legacy.value(),
                                 std::filesystem::copy_options::none, error);
      if (error) {
        cleanupStage();
        return Result<bool, std::string>::makeError(
            "approveAutotestBaselines: failed to preserve previous approval: " +
            error.message());
      }
    }
    auto legacyDigest = nuri::tools::core::sha256File(legacy.value());
    if (legacyDigest.hasError() || legacyDigest.value() != digest.value()) {
      cleanupStage();
      return Result<bool, std::string>::makeError(
          "approveAutotestBaselines: previous approval history digest "
          "mismatch");
    }
  }

  const std::string historyPrefix = nonce + "-";
  auto planHistory = nuri::tools::core::resolvePathUnder(
      baselineRoot, stageRelative / "history" / (historyPrefix + "plan.json"));
  auto approvalHistory = nuri::tools::core::resolvePathUnder(
      baselineRoot,
      stageRelative / "history" / (historyPrefix + "approval.json"));
  if (planHistory.hasError() || approvalHistory.hasError() ||
      std::filesystem::exists(planHistory.value()) ||
      std::filesystem::exists(approvalHistory.value())) {
    cleanupStage();
    return Result<bool, std::string>::makeError(
        "approveAutotestBaselines: immutable history entry already exists");
  }
  auto planHistoryWrite = nuri::tools::core::atomicWriteTextFile(
      planHistory.value(), planJson.value());
  if (planHistoryWrite.hasError()) {
    cleanupStage();
    return Result<bool, std::string>::makeError(planHistoryWrite.error());
  }

  const std::string approvalText = serializeAutotestApproval(
      expectedCase, baselineProfile,
      AutotestApprovalDocument{
          .reason = std::string(reason),
          .actor = std::string(actor),
          .approvedAtUtc = utcTimestampIso8601(),
          .sourceCommit = report.environment.commitHash,
          .sourceReportDigest = plan.value().sourceReportDigest,
          .carriedHistoryDigest = plan.value().historyDigest,
          .carriedHistoryFileCount = plan.value().historyFileCount,
          .planDigest = plan.value().digest});
  auto approvalWrite = nuri::tools::core::atomicWriteTextFile(
      stage.value() / "approval.json", approvalText);
  if (approvalWrite.hasError()) {
    cleanupStage();
    return Result<bool, std::string>::makeError(approvalWrite.error());
  }
  auto approvalHistoryWrite = nuri::tools::core::atomicWriteTextFile(
      approvalHistory.value(), approvalText);
  if (approvalHistoryWrite.hasError()) {
    cleanupStage();
    return Result<bool, std::string>::makeError(approvalHistoryWrite.error());
  }

  auto finalPlan = planAutotestBaselines(expectedCase, report, baselineProfile,
                                         reason, actor, options);
  if (finalPlan.hasError() || finalPlan.value().digest != confirmPlanDigest) {
    cleanupStage();
    return Result<bool, std::string>::makeError(
        "approveAutotestBaselines: candidate or baseline changed during "
        "staging");
  }

  target = nuri::tools::core::resolvePathUnder(baselineRoot, relativeCase);
  stage = nuri::tools::core::resolvePathUnder(baselineRoot, stageRelative);
  backup = nuri::tools::core::resolvePathUnder(baselineRoot, backupRelative);
  if (target.hasError() || stage.hasError() || backup.hasError()) {
    cleanupStage();
    return Result<bool, std::string>::makeError(
        "approveAutotestBaselines: containment changed before promotion");
  }
  const bool hadPrevious = std::filesystem::exists(target.value(), error);
  if (error) {
    cleanupStage();
    return Result<bool, std::string>::makeError(
        "approveAutotestBaselines: failed to inspect target: " +
        error.message());
  }
  if (hadPrevious) {
    std::filesystem::rename(target.value(), backup.value(), error);
    if (error) {
      cleanupStage();
      return Result<bool, std::string>::makeError(
          "approveAutotestBaselines: failed to stage previous baseline: " +
          error.message());
    }
  }
  if (options.failAfterBackupForTesting) {
    error = std::make_error_code(std::errc::io_error);
  } else {
    std::filesystem::rename(stage.value(), target.value(), error);
  }
  if (error) {
    const std::string promotionError = error.message();
    if (hadPrevious) {
      std::error_code rollbackError;
      std::filesystem::rename(backup.value(), target.value(), rollbackError);
      if (rollbackError) {
        return Result<bool, std::string>::makeError(
            "approveAutotestBaselines: promotion failed (" + promotionError +
            ") and rollback failed: " + rollbackError.message());
      }
    }
    cleanupStage();
    return Result<bool, std::string>::makeError(
        "approveAutotestBaselines: failed to promote staged baseline: " +
        promotionError);
  }
  if (hadPrevious) {
    auto removed =
        nuri::tools::core::removeTreeUnder(baselineRoot, backupRelative);
    if (removed.hasError()) {
      return Result<bool, std::string>::makeError(
          "approveAutotestBaselines: baseline promoted but backup cleanup "
          "failed: " +
          removed.error());
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

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
           << "  \"gpuDeviceName\": \""
           << jsonEscape(report.environment.gpuDeviceName) << "\",\n"
           << "  \"gpuVendorId\": " << report.environment.gpuVendorId << ",\n"
           << "  \"gpuDeviceId\": " << report.environment.gpuDeviceId << ",\n"
           << "  \"gpuDriverVersion\": \""
           << jsonEscape(report.environment.gpuDriverVersion) << "\",\n"
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
              readString(root, "gpuDeviceName") != environment.gpuDeviceName ||
                  readU32(root, "gpuVendorId") != environment.gpuVendorId ||
                  readU32(root, "gpuDeviceId") != environment.gpuDeviceId ||
                  readString(root, "gpuDriverVersion") !=
                      environment.gpuDriverVersion,
              "baseline GPU adapter/driver mismatch");
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
