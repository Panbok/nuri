#include "nuri/tools/benchmark/benchmark_baseline.h"

#include "nuri/tools/benchmark/benchmark_environment.h"
#include "nuri/tools/core/atomic_file.h"
#include "nuri/tools/core/identifier.h"
#include "nuri/tools/core/safe_path.h"
#include "nuri/tools/core/sha256.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <memory>
#include <set>
#include <span>
#include <sstream>
#include <tuple>
#include <utility>

#include <yyjson.h>

namespace nuri::tools::benchmark {
namespace {

using nuri::tools::core::BaselineProfile;
using nuri::tools::core::IdentifierShape;
using JsonDoc = std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)>;
using JsonMutDoc =
    std::unique_ptr<yyjson_mut_doc, decltype(&yyjson_mut_doc_free)>;

struct BaselinePaths {
  std::filesystem::path root{};
  std::filesystem::path relativeSuite{};
  std::filesystem::path suiteDir{};
  std::filesystem::path reportPath{};
  std::filesystem::path approvalPath{};
  std::filesystem::path historyDir{};
};

struct PreparedBaseline {
  BenchmarkBaselinePlan plan{};
  BaselinePaths paths{};
  std::string acceptedReportJson{};
};

struct ApprovalRecord {
  std::string caseId{};
  std::string suite{};
  std::string profileId{};
  bool profileAuthoritative = false;
  bool authoritative = false;
  std::string sourceReportDigest{};
  std::string acceptedReportDigest{};
  std::string planDigest{};
  std::string planDocumentDigest{};
  std::string previousApprovalDigest{};
  std::string historyKey{};
  BenchmarkBaselineGatePolicy gatePolicy{};
};

[[nodiscard]] bool hasNonWhitespace(std::string_view text) {
  return std::any_of(text.begin(), text.end(),
                     [](unsigned char value) { return !std::isspace(value); });
}

[[nodiscard]] bool isSha256(std::string_view digest) {
  if (digest.size() != 71u || !digest.starts_with("sha256:")) {
    return false;
  }
  return std::all_of(digest.begin() + 7, digest.end(),
                     [](unsigned char c) { return std::isxdigit(c) != 0; });
}

[[nodiscard]] std::string sha256Text(std::string_view text) {
  return "sha256:" + nuri::tools::core::sha256Hex(
                         std::as_bytes(std::span(text.data(), text.size())));
}

[[nodiscard]] Result<std::string, std::string>
sha256FileLabel(const std::filesystem::path &path) {
  auto digest = nuri::tools::core::sha256File(path);
  if (digest.hasError()) {
    return Result<std::string, std::string>::makeError(digest.error());
  }
  return Result<std::string, std::string>::makeResult("sha256:" +
                                                      digest.value());
}

[[nodiscard]] bool finiteStats(const MetricStats &stats) {
  return stats.count > 0u && std::isfinite(stats.min) &&
         std::isfinite(stats.median) && std::isfinite(stats.p90) &&
         std::isfinite(stats.p95) && std::isfinite(stats.max) &&
         std::isfinite(stats.mean) && std::isfinite(stats.stddev) &&
         std::isfinite(stats.mad) && std::isfinite(stats.iqr) &&
         std::isfinite(stats.coefficientOfVariation);
}

[[nodiscard]] bool validThresholds(const BenchmarkThresholds &thresholds) {
  return std::isfinite(thresholds.failPercent) &&
         std::isfinite(thresholds.failAbsoluteMs) &&
         std::isfinite(thresholds.warnPercent) &&
         std::isfinite(thresholds.warnAbsoluteMs) &&
         thresholds.failPercent >= 0.0 && thresholds.failAbsoluteMs >= 0.0 &&
         thresholds.warnPercent >= 0.0 && thresholds.warnAbsoluteMs >= 0.0 &&
         thresholds.warnPercent <= thresholds.failPercent &&
         thresholds.warnAbsoluteMs <= thresholds.failAbsoluteMs;
}

[[nodiscard]] bool equalThresholds(const BenchmarkThresholds &left,
                                   const BenchmarkThresholds &right) {
  return left.failPercent == right.failPercent &&
         left.failAbsoluteMs == right.failAbsoluteMs &&
         left.warnPercent == right.warnPercent &&
         left.warnAbsoluteMs == right.warnAbsoluteMs;
}

[[nodiscard]] Result<void, std::string>
validateIdentity(std::string_view caseId, std::string_view suite,
                 std::string_view profileId, std::string_view operation) {
  for (const auto &[value, field, shape] :
       {std::tuple{caseId, std::string_view("case id"),
                   IdentifierShape::Dotted},
        std::tuple{suite, std::string_view("suite"), IdentifierShape::Segment},
        std::tuple{profileId, std::string_view("profile"),
                   IdentifierShape::Segment}}) {
    auto valid = nuri::tools::core::validateIdentifier(value, field, shape);
    if (valid.hasError()) {
      return Result<void, std::string>::makeError(std::string(operation) +
                                                  ": " + valid.error());
    }
  }
  return Result<void, std::string>::makeResult();
}

[[nodiscard]] std::filesystem::path
effectiveRoot(const std::filesystem::path &root) {
  return root.empty() ? defaultBenchmarkBaselineRoot() : root;
}

[[nodiscard]] Result<BaselinePaths, std::string>
makePaths(const std::filesystem::path &root, std::string_view profileId,
          std::string_view suite, std::string_view caseId) {
  BaselinePaths paths{};
  paths.root = effectiveRoot(root);
  paths.relativeSuite = std::filesystem::path(profileId) / suite;
  auto suiteDir =
      nuri::tools::core::resolvePathUnder(paths.root, paths.relativeSuite);
  auto report = nuri::tools::core::resolvePathUnder(
      paths.root, paths.relativeSuite / (std::string(caseId) + ".json"));
  auto approval = nuri::tools::core::resolvePathUnder(
      paths.root,
      paths.relativeSuite / (std::string(caseId) + ".approval.json"));
  auto history = nuri::tools::core::resolvePathUnder(
      paths.root, paths.relativeSuite / "history" / caseId);
  if (suiteDir.hasError() || report.hasError() || approval.hasError() ||
      history.hasError()) {
    return Result<BaselinePaths, std::string>::makeError(
        "benchmark baseline path failed containment validation");
  }
  paths.suiteDir = suiteDir.value();
  paths.reportPath = report.value();
  paths.approvalPath = approval.value();
  paths.historyDir = history.value();
  return Result<BaselinePaths, std::string>::makeResult(std::move(paths));
}

[[nodiscard]] Result<void, std::string>
validateSourceReport(const BenchmarkBaselineSource &source,
                     const BaselineProfile &profile) {
  const BenchmarkReport &report = source.report;
  auto identity =
      validateIdentity(report.benchmarkCase.id, report.benchmarkCase.suite,
                       profile.id, "planBenchmarkBaseline");
  if (identity.hasError()) {
    return identity;
  }
  if (!std::filesystem::is_regular_file(source.reportPath)) {
    return Result<void, std::string>::makeError(
        "planBenchmarkBaseline: source report is not a regular file");
  }
  if (report.kind != "nuri.benchmark.report" || report.schemaVersion != 1u) {
    return Result<void, std::string>::makeError(
        "planBenchmarkBaseline: unsupported source report identity");
  }
  if (report.profile.id != profile.id ||
      report.profile.profileAuthoritative != profile.authority.authoritative ||
      report.profile.minimumRepetitions !=
          profile.benchmarkPolicy.minimumRepetitions ||
      report.profile.warmupStabilityPolicy !=
          profile.benchmarkPolicy.warmupStability ||
      report.profile.warmupWindowFrames !=
          profile.benchmarkPolicy.warmupWindowFrames ||
      report.profile.warmupMaxDriftPercent !=
          profile.benchmarkPolicy.warmupMaxDriftPercent ||
      report.profile.requiredMetrics !=
          profile.benchmarkPolicy.requiredMetrics) {
    return Result<void, std::string>::makeError(
        "planBenchmarkBaseline: source report profile policy mismatch");
  }
  if ((!profile.authority.authoritative && report.profile.authoritative) ||
      (profile.authority.authoritative && !report.profile.authoritative)) {
    return Result<void, std::string>::makeError(
        "planBenchmarkBaseline: source authority does not satisfy profile");
  }
  if (report.environment.tracyDiagnostic) {
    return Result<void, std::string>::makeError(
        "planBenchmarkBaseline: Tracy diagnostic reports cannot be accepted");
  }
  if (!validThresholds(report.benchmarkCase.thresholds)) {
    return Result<void, std::string>::makeError(
        "planBenchmarkBaseline: source gate policy is invalid");
  }
  if (report.run.samples == 0u || report.run.measurementFrames == 0u) {
    return Result<void, std::string>::makeError(
        "planBenchmarkBaseline: source report has no measured work");
  }
  const uint64_t expectedFrames =
      static_cast<uint64_t>(report.run.samples) * report.run.measurementFrames;
  const uint64_t measuredFrames = static_cast<uint64_t>(std::count_if(
      report.frames.begin(), report.frames.end(),
      [](const BenchmarkFrameRecord &frame) { return frame.measured; }));
  const bool isolatedRepetitions =
      report.repeatObservations.independent &&
      report.repeatObservations.unit == "isolated-process" &&
      report.profile.repetitionUnit == "isolated-process";
  const uint64_t expectedObservations =
      isolatedRepetitions ? report.profile.completedRepetitions
                          : report.run.samples;
  if ((isolatedRepetitions
           ? measuredFrames != 0u ||
                 report.repeatObservations.count != expectedObservations
           : measuredFrames != expectedFrames) ||
      report.sampleStats.size() != expectedObservations) {
    return Result<void, std::string>::makeError(
        "planBenchmarkBaseline: source measurements are incomplete");
  }

  std::set<std::string> required(report.benchmarkCase.requiredMetrics.begin(),
                                 report.benchmarkCase.requiredMetrics.end());
  required.insert(profile.benchmarkPolicy.requiredMetrics.begin(),
                  profile.benchmarkPolicy.requiredMetrics.end());
  for (const std::string &metric : required) {
    const auto overall = report.stats.find(metric);
    if (overall == report.stats.end() || !finiteStats(overall->second) ||
        overall->second.count !=
            (isolatedRepetitions ? expectedObservations : expectedFrames) ||
        std::find(report.unavailableMetrics.begin(),
                  report.unavailableMetrics.end(),
                  metric) != report.unavailableMetrics.end()) {
      return Result<void, std::string>::makeError(
          "planBenchmarkBaseline: required metric '" + metric +
          "' is incomplete");
    }
    for (const BenchmarkSampleStats &sample : report.sampleStats) {
      const auto perSample = sample.stats.find(metric);
      if (perSample == sample.stats.end() || !finiteStats(perSample->second) ||
          perSample->second.count != (isolatedRepetitions
                                          ? sample.measuredFrameCount
                                          : report.run.measurementFrames) ||
          (isolatedRepetitions && sample.measuredFrameCount == 0u)) {
        return Result<void, std::string>::makeError(
            "planBenchmarkBaseline: required per-sample metric '" + metric +
            "' is incomplete");
      }
    }
  }
  if (profile.authority.authoritative && !report.run.validForComparison) {
    return Result<void, std::string>::makeError(
        "planBenchmarkBaseline: authoritative source is not comparable");
  }
  return Result<void, std::string>::makeResult();
}

[[nodiscard]] Result<PreparedBaseline, std::string>
prepareBaseline(const BenchmarkBaselineSource &source,
                const BaselineProfile &profile, std::string_view reason,
                std::string_view actor,
                const std::filesystem::path &baselineRoot) {
  if (!hasNonWhitespace(reason) || !hasNonWhitespace(actor)) {
    return Result<PreparedBaseline, std::string>::makeError(
        "planBenchmarkBaseline: reason and actor are required");
  }
  auto sourceValid = validateSourceReport(source, profile);
  if (sourceValid.hasError()) {
    return Result<PreparedBaseline, std::string>::makeError(
        sourceValid.error());
  }
  auto paths =
      makePaths(baselineRoot, profile.id, source.report.benchmarkCase.suite,
                source.report.benchmarkCase.id);
  if (paths.hasError()) {
    return Result<PreparedBaseline, std::string>::makeError(paths.error());
  }
  auto sourceDigest = sha256FileLabel(source.reportPath);
  if (sourceDigest.hasError()) {
    return Result<PreparedBaseline, std::string>::makeError(
        sourceDigest.error());
  }

  BenchmarkReport accepted = source.report;
  accepted.artifacts = {};
  accepted.tracy.tracePath.clear();
  accepted.tracy.captureLogPath.clear();
  accepted.tracy.zonesCsvPath.clear();
  accepted.tracy.selfZonesCsvPath.clear();
  accepted.tracy.exportLogPath.clear();
  accepted.tracy.flameGraph.eventsCsvPath.clear();
  if (!profile.authority.authoritative) {
    accepted.benchmarkCase.authoritative = false;
    accepted.profile.authoritative = false;
    accepted.run.validForComparison = false;
  }

  BenchmarkBaselineGatePolicy gatePolicy{.thresholds =
                                             accepted.benchmarkCase.thresholds,
                                         .source = "source-report-initial"};
  std::string previousReportDigest;
  if (std::filesystem::is_regular_file(paths.value().reportPath)) {
    auto previous = readBenchmarkReportFile(paths.value().reportPath);
    if (previous.hasError() ||
        previous.value().benchmarkCase.id != accepted.benchmarkCase.id ||
        previous.value().benchmarkCase.suite != accepted.benchmarkCase.suite ||
        !validThresholds(previous.value().benchmarkCase.thresholds)) {
      return Result<PreparedBaseline, std::string>::makeError(
          "planBenchmarkBaseline: existing baseline is invalid");
    }
    gatePolicy.thresholds = previous.value().benchmarkCase.thresholds;
    gatePolicy.source = "previous-baseline";
    accepted.benchmarkCase.thresholds = gatePolicy.thresholds;
    auto previousDigest = sha256FileLabel(paths.value().reportPath);
    if (previousDigest.hasError()) {
      return Result<PreparedBaseline, std::string>::makeError(
          previousDigest.error());
    }
    previousReportDigest = previousDigest.value();
  }

  auto acceptedJson = writeBenchmarkReportJson(accepted, false);
  if (acceptedJson.hasError()) {
    return Result<PreparedBaseline, std::string>::makeError(
        acceptedJson.error());
  }
  const std::string acceptedDigest = sha256Text(acceptedJson.value());
  std::string previousApprovalDigest;
  if (std::filesystem::is_regular_file(paths.value().approvalPath)) {
    auto digest = sha256FileLabel(paths.value().approvalPath);
    if (digest.hasError()) {
      return Result<PreparedBaseline, std::string>::makeError(digest.error());
    }
    previousApprovalDigest = digest.value();
  }

  BenchmarkBaselinePlan plan{};
  plan.caseId = accepted.benchmarkCase.id;
  plan.suite = accepted.benchmarkCase.suite;
  plan.profileId = profile.id;
  plan.reason = std::string(reason);
  plan.actor = std::string(actor);
  plan.profileAuthoritative = profile.authority.authoritative;
  plan.authoritative =
      profile.authority.authoritative && accepted.profile.authoritative;
  plan.sourceCommit = accepted.environment.commitHash;
  plan.sourceReportDigest = sourceDigest.value();
  plan.acceptedReportDigest = acceptedDigest;
  plan.previousApprovalDigest = previousApprovalDigest;
  plan.gatePolicy = gatePolicy;
  plan.entries.push_back(BenchmarkBaselinePlanEntry{
      .path = (paths.value().relativeSuite / (plan.caseId + ".json"))
                  .generic_string(),
      .operation =
          previousReportDigest.empty()
              ? "add"
              : (previousReportDigest == acceptedDigest ? "keep" : "replace"),
      .previousDigest = previousReportDigest,
      .sourceDigest = acceptedDigest});
  plan.entries.push_back(BenchmarkBaselinePlanEntry{
      .path = (paths.value().relativeSuite / (plan.caseId + ".approval.json"))
                  .generic_string(),
      .operation = previousApprovalDigest.empty() ? "add" : "replace",
      .previousDigest = previousApprovalDigest,
      .sourceDigest = plan.sourceReportDigest});

  std::ostringstream canonical;
  canonical << std::setprecision(17) << "nuri.benchmark.baseline_plan.v1\n"
            << plan.caseId << '\n'
            << plan.suite << '\n'
            << plan.profileId << '\n'
            << plan.reason << '\n'
            << plan.actor << '\n'
            << (plan.profileAuthoritative ? 1 : 0) << '\n'
            << (plan.authoritative ? 1 : 0) << '\n'
            << plan.sourceCommit << '\n'
            << plan.sourceReportDigest << '\n'
            << plan.acceptedReportDigest << '\n'
            << plan.previousApprovalDigest << '\n'
            << gatePolicy.source << '\n'
            << gatePolicy.thresholds.failPercent << '\n'
            << gatePolicy.thresholds.failAbsoluteMs << '\n'
            << gatePolicy.thresholds.warnPercent << '\n'
            << gatePolicy.thresholds.warnAbsoluteMs << '\n';
  for (const BenchmarkBaselinePlanEntry &entry : plan.entries) {
    canonical << entry.path << '|' << entry.operation << '|'
              << entry.previousDigest << '|' << entry.sourceDigest << '\n';
  }
  plan.digest = sha256Text(canonical.str());
  return Result<PreparedBaseline, std::string>::makeResult(
      PreparedBaseline{.plan = std::move(plan),
                       .paths = std::move(paths.value()),
                       .acceptedReportJson = std::move(acceptedJson.value())});
}

void addString(yyjson_mut_doc *doc, yyjson_mut_val *object, const char *key,
               const std::string &value) {
  yyjson_mut_obj_add_strcpy(doc, object, key, value.c_str());
}

[[nodiscard]] yyjson_mut_val *
makeGatePolicyObject(yyjson_mut_doc *doc,
                     const BenchmarkBaselineGatePolicy &policy) {
  yyjson_mut_val *object = yyjson_mut_obj(doc);
  addString(doc, object, "source", policy.source);
  yyjson_mut_obj_add_real(doc, object, "failPercent",
                          policy.thresholds.failPercent);
  yyjson_mut_obj_add_real(doc, object, "failAbsoluteMs",
                          policy.thresholds.failAbsoluteMs);
  yyjson_mut_obj_add_real(doc, object, "warnPercent",
                          policy.thresholds.warnPercent);
  yyjson_mut_obj_add_real(doc, object, "warnAbsoluteMs",
                          policy.thresholds.warnAbsoluteMs);
  return object;
}

[[nodiscard]] Result<std::string, std::string>
serializeMutableDocument(JsonMutDoc &doc, std::string_view operation) {
  size_t length = 0u;
  char *json = yyjson_mut_write_opts(doc.get(), YYJSON_WRITE_PRETTY, nullptr,
                                     &length, nullptr);
  if (json == nullptr) {
    return Result<std::string, std::string>::makeError(
        std::string(operation) + ": serialization failed");
  }
  std::string text(json, length);
  std::free(json);
  return Result<std::string, std::string>::makeResult(std::move(text));
}

[[nodiscard]] std::string uniqueNonce() {
  const uint64_t ticks = static_cast<uint64_t>(
      std::chrono::high_resolution_clock::now().time_since_epoch().count());
  std::ostringstream out;
  out << std::hex << std::setfill('0') << std::setw(12)
      << (ticks & 0xffffffffffffull);
  return out.str();
}

[[nodiscard]] Result<void, std::string>
rejectLinksInTree(const std::filesystem::path &root,
                  const std::filesystem::path &relativeTree) {
  auto tree = nuri::tools::core::resolvePathUnder(root, relativeTree);
  if (tree.hasError()) {
    return Result<void, std::string>::makeError(tree.error());
  }
  if (!std::filesystem::exists(tree.value())) {
    return Result<void, std::string>::makeResult();
  }
  std::error_code error;
  for (const auto &entry :
       std::filesystem::recursive_directory_iterator(tree.value(), error)) {
    if (error) {
      return Result<void, std::string>::makeError(
          "failed to inspect baseline tree: " + error.message());
    }
    const auto status = entry.symlink_status(error);
    if (error || std::filesystem::is_symlink(status)) {
      return Result<void, std::string>::makeError(
          "baseline tree contains an unsupported link");
    }
    if (std::filesystem::is_regular_file(status) &&
        std::filesystem::hard_link_count(entry.path(), error) != 1u) {
      return Result<void, std::string>::makeError(
          error ? "failed to inspect baseline file links: " + error.message()
                : "baseline tree contains an unsupported hard link");
    }
    const std::filesystem::path relative =
        std::filesystem::relative(entry.path(), root, error);
    if (error ||
        nuri::tools::core::resolvePathUnder(root, relative).hasError()) {
      return Result<void, std::string>::makeError(
          "baseline tree failed containment validation");
    }
  }
  return Result<void, std::string>::makeResult();
}

[[nodiscard]] Result<std::string, std::string> writeApprovalJson(
    const PreparedBaseline &prepared, const std::string &historyKey,
    const std::string &acceptedAtUtc, const std::string &planDocumentDigest) {
  JsonMutDoc doc(yyjson_mut_doc_new(nullptr), &yyjson_mut_doc_free);
  if (!doc) {
    return Result<std::string, std::string>::makeError(
        "writeBenchmarkBaselineApproval: allocation failed");
  }
  yyjson_mut_val *root = yyjson_mut_obj(doc.get());
  yyjson_mut_doc_set_root(doc.get(), root);
  yyjson_mut_obj_add_uint(doc.get(), root, "schemaVersion", 1u);
  yyjson_mut_obj_add_strcpy(doc.get(), root, "kind",
                            "nuri.benchmark.baseline_approval");
  addString(doc.get(), root, "caseId", prepared.plan.caseId);
  addString(doc.get(), root, "suite", prepared.plan.suite);
  addString(doc.get(), root, "profileId", prepared.plan.profileId);
  addString(doc.get(), root, "reason", prepared.plan.reason);
  addString(doc.get(), root, "actor", prepared.plan.actor);
  yyjson_mut_obj_add_bool(doc.get(), root, "profileAuthoritative",
                          prepared.plan.profileAuthoritative);
  yyjson_mut_obj_add_bool(doc.get(), root, "authoritative",
                          prepared.plan.authoritative);
  addString(doc.get(), root, "acceptedAtUtc", acceptedAtUtc);
  addString(doc.get(), root, "sourceCommit", prepared.plan.sourceCommit);
  addString(doc.get(), root, "sourceReportDigest",
            prepared.plan.sourceReportDigest);
  addString(doc.get(), root, "acceptedReportDigest",
            prepared.plan.acceptedReportDigest);
  addString(doc.get(), root, "planDigest", prepared.plan.digest);
  addString(doc.get(), root, "planDocumentDigest", planDocumentDigest);
  addString(doc.get(), root, "previousApprovalDigest",
            prepared.plan.previousApprovalDigest);
  addString(doc.get(), root, "historyKey", historyKey);
  yyjson_mut_obj_add_val(
      doc.get(), root, "gatePolicy",
      makeGatePolicyObject(doc.get(), prepared.plan.gatePolicy));
  return serializeMutableDocument(doc, "writeBenchmarkBaselineApproval");
}

[[nodiscard]] std::string readString(yyjson_val *object, const char *key) {
  yyjson_val *value = yyjson_obj_get(object, key);
  return yyjson_is_str(value)
             ? std::string(yyjson_get_str(value), yyjson_get_len(value))
             : std::string{};
}

[[nodiscard]] double readReal(yyjson_val *object, const char *key) {
  yyjson_val *value = yyjson_obj_get(object, key);
  return yyjson_is_num(value) ? yyjson_get_num(value) : 0.0;
}

[[nodiscard]] Result<ApprovalRecord, std::string>
readApproval(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return Result<ApprovalRecord, std::string>::makeError(
        "baseline approval is missing");
  }
  std::string json((std::istreambuf_iterator<char>(file)),
                   std::istreambuf_iterator<char>());
  yyjson_read_err error{};
  JsonDoc doc(yyjson_read_opts(json.data(), json.size(), 0, nullptr, &error),
              &yyjson_doc_free);
  if (!doc) {
    return Result<ApprovalRecord, std::string>::makeError(
        "baseline approval JSON is invalid");
  }
  yyjson_val *root = yyjson_doc_get_root(doc.get());
  yyjson_val *schema = yyjson_obj_get(root, "schemaVersion");
  yyjson_val *kind = yyjson_obj_get(root, "kind");
  if (!yyjson_is_obj(root) || !yyjson_is_uint(schema) ||
      yyjson_get_uint(schema) != 1u || !yyjson_is_str(kind) ||
      std::string_view(yyjson_get_str(kind), yyjson_get_len(kind)) !=
          "nuri.benchmark.baseline_approval") {
    return Result<ApprovalRecord, std::string>::makeError(
        "baseline approval identity is invalid");
  }
  ApprovalRecord record{};
  record.caseId = readString(root, "caseId");
  record.suite = readString(root, "suite");
  record.profileId = readString(root, "profileId");
  yyjson_val *profileAuthoritative =
      yyjson_obj_get(root, "profileAuthoritative");
  yyjson_val *authoritative = yyjson_obj_get(root, "authoritative");
  if (!yyjson_is_bool(profileAuthoritative) || !yyjson_is_bool(authoritative)) {
    return Result<ApprovalRecord, std::string>::makeError(
        "baseline approval authority metadata is invalid");
  }
  record.profileAuthoritative = yyjson_get_bool(profileAuthoritative);
  record.authoritative = yyjson_get_bool(authoritative);
  record.sourceReportDigest = readString(root, "sourceReportDigest");
  record.acceptedReportDigest = readString(root, "acceptedReportDigest");
  record.planDigest = readString(root, "planDigest");
  record.planDocumentDigest = readString(root, "planDocumentDigest");
  record.previousApprovalDigest = readString(root, "previousApprovalDigest");
  record.historyKey = readString(root, "historyKey");
  yyjson_val *policy = yyjson_obj_get(root, "gatePolicy");
  if (!yyjson_is_obj(policy)) {
    return Result<ApprovalRecord, std::string>::makeError(
        "baseline approval gate policy is missing");
  }
  record.gatePolicy.source = readString(policy, "source");
  record.gatePolicy.thresholds.failPercent = readReal(policy, "failPercent");
  record.gatePolicy.thresholds.failAbsoluteMs =
      readReal(policy, "failAbsoluteMs");
  record.gatePolicy.thresholds.warnPercent = readReal(policy, "warnPercent");
  record.gatePolicy.thresholds.warnAbsoluteMs =
      readReal(policy, "warnAbsoluteMs");
  if (record.caseId.empty() || record.suite.empty() ||
      record.profileId.empty() || !isSha256(record.sourceReportDigest) ||
      !isSha256(record.acceptedReportDigest) || !isSha256(record.planDigest) ||
      !isSha256(record.planDocumentDigest) ||
      (!record.previousApprovalDigest.empty() &&
       !isSha256(record.previousApprovalDigest)) ||
      !validThresholds(record.gatePolicy.thresholds)) {
    return Result<ApprovalRecord, std::string>::makeError(
        "baseline approval content is invalid");
  }
  return Result<ApprovalRecord, std::string>::makeResult(std::move(record));
}

[[nodiscard]] bool reportHasVerboseFrames(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return true;
  }
  std::string json((std::istreambuf_iterator<char>(file)),
                   std::istreambuf_iterator<char>());
  JsonDoc doc(yyjson_read(json.data(), json.size(), 0), &yyjson_doc_free);
  if (!doc) {
    return true;
  }
  yyjson_val *frames = yyjson_obj_get(yyjson_doc_get_root(doc.get()), "frames");
  if (!yyjson_is_arr(frames)) {
    return true;
  }
  yyjson_arr_iter iterator;
  yyjson_arr_iter_init(frames, &iterator);
  yyjson_val *frame = nullptr;
  while ((frame = yyjson_arr_iter_next(&iterator)) != nullptr) {
    if (yyjson_obj_get(frame, "metrics") != nullptr) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] std::string readPlanDigest(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return {};
  }
  std::string json((std::istreambuf_iterator<char>(file)),
                   std::istreambuf_iterator<char>());
  JsonDoc doc(yyjson_read(json.data(), json.size(), 0), &yyjson_doc_free);
  if (!doc) {
    return {};
  }
  yyjson_val *root = yyjson_doc_get_root(doc.get());
  yyjson_val *kind = yyjson_obj_get(root, "kind");
  if (!yyjson_is_str(kind) ||
      std::string_view(yyjson_get_str(kind), yyjson_get_len(kind)) !=
          "nuri.benchmark.baseline_plan") {
    return {};
  }
  return readString(root, "planDigest");
}

} // namespace

std::filesystem::path defaultBenchmarkBaselineRoot() {
  return benchmarkRepoRoot() / "tools" / "baselines" / "benchmark";
}

Result<BenchmarkBaselineSource, std::string>
loadBenchmarkBaselineSource(const std::filesystem::path &runRoot,
                            std::string_view caseId) {
  auto valid = nuri::tools::core::validateIdentifier(caseId, "case id",
                                                     IdentifierShape::Dotted);
  if (valid.hasError()) {
    return Result<BenchmarkBaselineSource, std::string>::makeError(
        "loadBenchmarkBaselineSource: " + valid.error());
  }
  auto reportPath = nuri::tools::core::resolvePathUnder(
      runRoot,
      std::filesystem::path("cases") / (std::string(caseId) + ".json"));
  if (reportPath.hasError() ||
      !std::filesystem::is_regular_file(reportPath.value())) {
    return Result<BenchmarkBaselineSource, std::string>::makeError(
        "loadBenchmarkBaselineSource: source case report is missing");
  }
  auto report = readBenchmarkReportFile(reportPath.value());
  if (report.hasError()) {
    return Result<BenchmarkBaselineSource, std::string>::makeError(
        report.error());
  }
  if (report.value().benchmarkCase.id != caseId) {
    return Result<BenchmarkBaselineSource, std::string>::makeError(
        "loadBenchmarkBaselineSource: source report case mismatch");
  }
  return Result<BenchmarkBaselineSource, std::string>::makeResult(
      BenchmarkBaselineSource{.report = std::move(report.value()),
                              .reportPath = reportPath.value()});
}

Result<BenchmarkBaselinePlan, std::string>
planBenchmarkBaseline(const BenchmarkBaselineSource &source,
                      const BaselineProfile &profile, std::string_view reason,
                      std::string_view actor,
                      const std::filesystem::path &baselineRoot) {
  auto prepared = prepareBaseline(source, profile, reason, actor, baselineRoot);
  if (prepared.hasError()) {
    return Result<BenchmarkBaselinePlan, std::string>::makeError(
        prepared.error());
  }
  return Result<BenchmarkBaselinePlan, std::string>::makeResult(
      std::move(prepared.value().plan));
}

Result<std::string, std::string>
writeBenchmarkBaselinePlanJson(const BenchmarkBaselinePlan &plan) {
  JsonMutDoc doc(yyjson_mut_doc_new(nullptr), &yyjson_mut_doc_free);
  if (!doc) {
    return Result<std::string, std::string>::makeError(
        "writeBenchmarkBaselinePlanJson: allocation failed");
  }
  yyjson_mut_val *root = yyjson_mut_obj(doc.get());
  yyjson_mut_doc_set_root(doc.get(), root);
  yyjson_mut_obj_add_uint(doc.get(), root, "schemaVersion", 1u);
  yyjson_mut_obj_add_strcpy(doc.get(), root, "kind",
                            "nuri.benchmark.baseline_plan");
  addString(doc.get(), root, "caseId", plan.caseId);
  addString(doc.get(), root, "suite", plan.suite);
  addString(doc.get(), root, "profileId", plan.profileId);
  addString(doc.get(), root, "reason", plan.reason);
  addString(doc.get(), root, "actor", plan.actor);
  yyjson_mut_obj_add_bool(doc.get(), root, "profileAuthoritative",
                          plan.profileAuthoritative);
  yyjson_mut_obj_add_bool(doc.get(), root, "authoritative", plan.authoritative);
  addString(doc.get(), root, "sourceCommit", plan.sourceCommit);
  addString(doc.get(), root, "sourceReportDigest", plan.sourceReportDigest);
  addString(doc.get(), root, "acceptedReportDigest", plan.acceptedReportDigest);
  addString(doc.get(), root, "previousApprovalDigest",
            plan.previousApprovalDigest);
  addString(doc.get(), root, "planDigest", plan.digest);
  yyjson_mut_obj_add_val(doc.get(), root, "gatePolicy",
                         makeGatePolicyObject(doc.get(), plan.gatePolicy));
  yyjson_mut_val *entries = yyjson_mut_arr(doc.get());
  for (const BenchmarkBaselinePlanEntry &entry : plan.entries) {
    yyjson_mut_val *object = yyjson_mut_obj(doc.get());
    addString(doc.get(), object, "path", entry.path);
    addString(doc.get(), object, "operation", entry.operation);
    addString(doc.get(), object, "previousDigest", entry.previousDigest);
    addString(doc.get(), object, "sourceDigest", entry.sourceDigest);
    yyjson_mut_arr_add_val(entries, object);
  }
  yyjson_mut_obj_add_val(doc.get(), root, "entries", entries);
  return serializeMutableDocument(doc, "writeBenchmarkBaselinePlanJson");
}

Result<bool, std::string>
acceptBenchmarkBaseline(const BenchmarkBaselineSource &source,
                        const BaselineProfile &profile, std::string_view reason,
                        std::string_view actor,
                        std::string_view confirmPlanDigest,
                        const BenchmarkBaselineAcceptOptions &options) {
  auto prepared =
      prepareBaseline(source, profile, reason, actor, options.baselineRoot);
  if (prepared.hasError()) {
    return Result<bool, std::string>::makeError(prepared.error());
  }
  if (confirmPlanDigest.empty() ||
      confirmPlanDigest != prepared.value().plan.digest) {
    return Result<bool, std::string>::makeError(
        "acceptBenchmarkBaseline: --confirm-plan must match the reviewed "
        "plan digest " +
        prepared.value().plan.digest);
  }
  auto planJson = writeBenchmarkBaselinePlanJson(prepared.value().plan);
  if (planJson.hasError()) {
    return Result<bool, std::string>::makeError(planJson.error());
  }
  const std::string planDocumentDigest = sha256Text(planJson.value());
  const std::string nonce = uniqueNonce();
  const std::string historyKey =
      "h-" + prepared.value().plan.digest.substr(7u, 16u) + "-" + nonce;
  const std::string acceptedAtUtc = utcTimestampIso8601();
  auto approvalJson = writeApprovalJson(prepared.value(), historyKey,
                                        acceptedAtUtc, planDocumentDigest);
  if (approvalJson.hasError()) {
    return Result<bool, std::string>::makeError(approvalJson.error());
  }

  const std::filesystem::path &root = prepared.value().paths.root;
  std::error_code error;
  std::filesystem::create_directories(root, error);
  if (error) {
    return Result<bool, std::string>::makeError(
        "acceptBenchmarkBaseline: failed to create baseline root: " +
        error.message());
  }
  auto profileDir = nuri::tools::core::resolvePathUnder(root, profile.id);
  if (profileDir.hasError()) {
    return Result<bool, std::string>::makeError(profileDir.error());
  }
  std::filesystem::create_directories(profileDir.value(), error);
  if (error) {
    return Result<bool, std::string>::makeError(
        "acceptBenchmarkBaseline: failed to create profile directory: " +
        error.message());
  }

  const std::filesystem::path stageRelative =
      std::filesystem::path(profile.id) /
      (prepared.value().plan.suite + ".stage-" + nonce);
  const std::filesystem::path backupRelative =
      std::filesystem::path(profile.id) /
      (prepared.value().plan.suite + ".backup-" + nonce);
  auto stage = nuri::tools::core::resolvePathUnder(root, stageRelative);
  auto backup = nuri::tools::core::resolvePathUnder(root, backupRelative);
  if (stage.hasError() || backup.hasError()) {
    return Result<bool, std::string>::makeError(
        "acceptBenchmarkBaseline: staging path failed containment");
  }
  auto links = rejectLinksInTree(root, prepared.value().paths.relativeSuite);
  if (links.hasError()) {
    return Result<bool, std::string>::makeError(links.error());
  }
  const auto cleanupStage = [&]() {
    (void)nuri::tools::core::removeTreeUnder(root, stageRelative);
  };

  const bool hadPrevious =
      std::filesystem::is_directory(prepared.value().paths.suiteDir);
  if (hadPrevious) {
    std::filesystem::copy(prepared.value().paths.suiteDir, stage.value(),
                          std::filesystem::copy_options::recursive, error);
  } else {
    std::filesystem::create_directories(stage.value(), error);
  }
  if (error) {
    cleanupStage();
    return Result<bool, std::string>::makeError(
        "acceptBenchmarkBaseline: failed to stage suite: " + error.message());
  }
  const std::filesystem::path stagedReport =
      stage.value() / (prepared.value().plan.caseId + ".json");
  const std::filesystem::path stagedApproval =
      stage.value() / (prepared.value().plan.caseId + ".approval.json");
  const std::filesystem::path stagedHistory =
      stage.value() / "history" / prepared.value().plan.caseId;
  std::filesystem::create_directories(stagedHistory, error);
  if (error) {
    cleanupStage();
    return Result<bool, std::string>::makeError(
        "acceptBenchmarkBaseline: failed to create history: " +
        error.message());
  }

  if (std::filesystem::is_regular_file(stagedReport)) {
    auto digest = sha256FileLabel(stagedReport);
    if (digest.hasError()) {
      cleanupStage();
      return Result<bool, std::string>::makeError(digest.error());
    }
    const std::filesystem::path prior =
        stagedHistory /
        ("prior-" + digest.value().substr(7u, 16u) + ".report.json");
    if (!std::filesystem::exists(prior)) {
      std::filesystem::copy_file(stagedReport, prior,
                                 std::filesystem::copy_options::none, error);
    }
  }
  if (!error && std::filesystem::is_regular_file(stagedApproval)) {
    auto digest = sha256FileLabel(stagedApproval);
    if (digest.hasError()) {
      cleanupStage();
      return Result<bool, std::string>::makeError(digest.error());
    }
    const std::filesystem::path prior =
        stagedHistory /
        ("prior-" + digest.value().substr(7u, 16u) + ".approval.json");
    if (!std::filesystem::exists(prior)) {
      std::filesystem::copy_file(stagedApproval, prior,
                                 std::filesystem::copy_options::none, error);
    }
  }
  if (error) {
    cleanupStage();
    return Result<bool, std::string>::makeError(
        "acceptBenchmarkBaseline: failed to preserve previous evidence: " +
        error.message());
  }

  const std::array<std::filesystem::path, 3u> newHistoryPaths{
      stagedHistory / (historyKey + ".report.json"),
      stagedHistory / (historyKey + ".approval.json"),
      stagedHistory / (historyKey + ".plan.json")};
  if (std::any_of(newHistoryPaths.begin(), newHistoryPaths.end(),
                  [](const std::filesystem::path &path) {
                    return std::filesystem::exists(path);
                  })) {
    cleanupStage();
    return Result<bool, std::string>::makeError(
        "acceptBenchmarkBaseline: immutable history key already exists");
  }
  for (const auto &[path, text] :
       {std::pair{stagedReport,
                  std::string_view(prepared.value().acceptedReportJson)},
        std::pair{stagedApproval, std::string_view(approvalJson.value())},
        std::pair{newHistoryPaths[0],
                  std::string_view(prepared.value().acceptedReportJson)},
        std::pair{newHistoryPaths[1], std::string_view(approvalJson.value())},
        std::pair{newHistoryPaths[2], std::string_view(planJson.value())}}) {
    auto written = nuri::tools::core::atomicWriteTextFile(path, text);
    if (written.hasError()) {
      cleanupStage();
      return Result<bool, std::string>::makeError("acceptBenchmarkBaseline: " +
                                                  written.error());
    }
  }

  if (hadPrevious) {
    std::filesystem::rename(prepared.value().paths.suiteDir, backup.value(),
                            error);
    if (error) {
      cleanupStage();
      return Result<bool, std::string>::makeError(
          "acceptBenchmarkBaseline: failed to stage previous suite: " +
          error.message());
    }
  }
  if (options.promotionFault ==
      BenchmarkBaselinePromotionFault::AfterBackupRenameForTesting) {
    if (hadPrevious) {
      std::error_code rollback;
      std::filesystem::rename(backup.value(), prepared.value().paths.suiteDir,
                              rollback);
      if (rollback) {
        return Result<bool, std::string>::makeError(
            "acceptBenchmarkBaseline: injected failure rollback failed: " +
            rollback.message());
      }
    }
    cleanupStage();
    return Result<bool, std::string>::makeError(
        "acceptBenchmarkBaseline: injected promotion failure");
  }

  std::filesystem::rename(stage.value(), prepared.value().paths.suiteDir,
                          error);
  if (error) {
    if (hadPrevious) {
      std::error_code rollback;
      std::filesystem::rename(backup.value(), prepared.value().paths.suiteDir,
                              rollback);
    }
    cleanupStage();
    return Result<bool, std::string>::makeError(
        "acceptBenchmarkBaseline: promotion failed: " + error.message());
  }
  if (hadPrevious) {
    auto removed = nuri::tools::core::removeTreeUnder(root, backupRelative);
    if (removed.hasError()) {
      return Result<bool, std::string>::makeError(
          "acceptBenchmarkBaseline: baseline promoted but backup cleanup "
          "failed: " +
          removed.error());
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<BenchmarkBaselineVerification, std::string>
verifyBenchmarkBaseline(std::string_view caseId, std::string_view suite,
                        const BaselineProfile &profile,
                        const std::filesystem::path &baselineRoot) {
  auto identity =
      validateIdentity(caseId, suite, profile.id, "verifyBenchmarkBaseline");
  if (identity.hasError()) {
    return Result<BenchmarkBaselineVerification, std::string>::makeError(
        identity.error());
  }
  auto paths = makePaths(baselineRoot, profile.id, suite, caseId);
  if (paths.hasError()) {
    return Result<BenchmarkBaselineVerification, std::string>::makeError(
        paths.error());
  }
  BenchmarkBaselineVerification verification{.caseId = std::string(caseId),
                                             .suite = std::string(suite),
                                             .profileId = profile.id};
  verification.exists =
      std::filesystem::is_regular_file(paths.value().reportPath);
  if (!verification.exists) {
    return Result<BenchmarkBaselineVerification, std::string>::makeResult(
        std::move(verification));
  }
  auto treeSafety =
      rejectLinksInTree(paths.value().root, paths.value().relativeSuite);
  if (treeSafety.hasError()) {
    verification.errors.push_back(treeSafety.error());
    return Result<BenchmarkBaselineVerification, std::string>::makeResult(
        std::move(verification));
  }
  auto reportDigest = sha256FileLabel(paths.value().reportPath);
  auto approvalDigest = sha256FileLabel(paths.value().approvalPath);
  auto report = readBenchmarkReportFile(paths.value().reportPath);
  auto approval = readApproval(paths.value().approvalPath);
  if (reportDigest.hasError() || approvalDigest.hasError() ||
      report.hasError() || approval.hasError()) {
    verification.errors.push_back(reportDigest.hasError() ? reportDigest.error()
                                  : approvalDigest.hasError()
                                      ? approvalDigest.error()
                                  : report.hasError() ? report.error()
                                                      : approval.error());
    return Result<BenchmarkBaselineVerification, std::string>::makeResult(
        std::move(verification));
  }
  verification.reportDigest = reportDigest.value();
  verification.approvalDigest = approvalDigest.value();
  verification.planDigest = approval.value().planDigest;
  verification.sourceReportDigest = approval.value().sourceReportDigest;
  verification.historyKey = approval.value().historyKey;
  verification.gatePolicy = approval.value().gatePolicy;
  verification.authoritative = approval.value().authoritative;

  if (approval.value().caseId != caseId || approval.value().suite != suite ||
      approval.value().profileId != profile.id ||
      report.value().benchmarkCase.id != caseId ||
      report.value().benchmarkCase.suite != suite ||
      report.value().profile.id != profile.id) {
    verification.errors.push_back("baseline identity mismatch");
  }
  if (report.value().profile.profileAuthoritative !=
          profile.authority.authoritative ||
      report.value().profile.minimumRepetitions !=
          profile.benchmarkPolicy.minimumRepetitions ||
      report.value().profile.warmupStabilityPolicy !=
          profile.benchmarkPolicy.warmupStability ||
      report.value().profile.requiredMetrics !=
          profile.benchmarkPolicy.requiredMetrics) {
    verification.errors.push_back("baseline profile policy mismatch");
  }
  if (approval.value().acceptedReportDigest != verification.reportDigest) {
    verification.errors.push_back("accepted report digest mismatch");
  }
  if (!equalThresholds(report.value().benchmarkCase.thresholds,
                       approval.value().gatePolicy.thresholds)) {
    verification.errors.push_back("baseline-owned gate policy mismatch");
  }
  if (approval.value().profileAuthoritative !=
          profile.authority.authoritative ||
      approval.value().authoritative != report.value().profile.authoritative ||
      (profile.authority.authoritative && !approval.value().authoritative) ||
      (!profile.authority.authoritative && approval.value().authoritative)) {
    verification.errors.push_back("baseline authority metadata mismatch");
  }
  if (reportHasVerboseFrames(paths.value().reportPath)) {
    verification.errors.push_back("accepted report contains verbose frames");
  }
  auto historyValid = nuri::tools::core::validateIdentifier(
      approval.value().historyKey, "history key", IdentifierShape::Segment);
  if (historyValid.hasError()) {
    verification.errors.push_back("baseline history key is invalid");
  } else {
    const std::filesystem::path relativeHistory =
        paths.value().relativeSuite / "history" / caseId;
    auto historyReport = nuri::tools::core::resolvePathUnder(
        paths.value().root,
        relativeHistory / (approval.value().historyKey + ".report.json"));
    auto historyApproval = nuri::tools::core::resolvePathUnder(
        paths.value().root,
        relativeHistory / (approval.value().historyKey + ".approval.json"));
    auto historyPlan = nuri::tools::core::resolvePathUnder(
        paths.value().root,
        relativeHistory / (approval.value().historyKey + ".plan.json"));
    if (historyReport.hasError() || historyApproval.hasError() ||
        historyPlan.hasError()) {
      verification.errors.push_back("baseline history path is unsafe");
    } else {
      auto historyReportDigest = sha256FileLabel(historyReport.value());
      auto historyApprovalDigest = sha256FileLabel(historyApproval.value());
      auto historyPlanDigest = sha256FileLabel(historyPlan.value());
      if (historyReportDigest.hasError() ||
          historyReportDigest.value() != verification.reportDigest) {
        verification.errors.push_back("baseline report history mismatch");
      }
      if (historyApprovalDigest.hasError() ||
          historyApprovalDigest.value() != verification.approvalDigest) {
        verification.errors.push_back("baseline approval history mismatch");
      }
      if (historyPlanDigest.hasError() ||
          historyPlanDigest.value() != approval.value().planDocumentDigest) {
        verification.errors.push_back("baseline plan history mismatch");
      }
      if (readPlanDigest(historyPlan.value()) != approval.value().planDigest) {
        verification.errors.push_back("baseline confirmed plan mismatch");
      }
    }
  }
  if (!approval.value().previousApprovalDigest.empty()) {
    bool foundPreviousApproval = false;
    std::error_code error;
    for (const auto &entry :
         std::filesystem::directory_iterator(paths.value().historyDir, error)) {
      if (error || !entry.is_regular_file()) {
        continue;
      }
      auto digest = sha256FileLabel(entry.path());
      foundPreviousApproval =
          !digest.hasError() &&
          digest.value() == approval.value().previousApprovalDigest;
      if (foundPreviousApproval) {
        break;
      }
    }
    if (!foundPreviousApproval) {
      verification.errors.push_back("previous approval history is missing");
    }
  }
  verification.valid = verification.errors.empty();
  if (!verification.authoritative) {
    verification.warnings.push_back(
        "baseline is investigative and cannot produce an authoritative gate");
  }
  return Result<BenchmarkBaselineVerification, std::string>::makeResult(
      std::move(verification));
}

Result<VerifiedBenchmarkBaseline, std::string>
loadVerifiedBenchmarkBaseline(std::string_view caseId, std::string_view suite,
                              const BaselineProfile &profile,
                              const std::filesystem::path &baselineRoot) {
  auto verification =
      verifyBenchmarkBaseline(caseId, suite, profile, baselineRoot);
  if (verification.hasError()) {
    return Result<VerifiedBenchmarkBaseline, std::string>::makeError(
        "loadVerifiedBenchmarkBaseline: " + verification.error());
  }
  if (!verification.value().exists) {
    return Result<VerifiedBenchmarkBaseline, std::string>::makeError(
        "loadVerifiedBenchmarkBaseline: governed baseline is missing");
  }
  if (!verification.value().valid) {
    std::string message =
        "loadVerifiedBenchmarkBaseline: governed baseline verification failed";
    for (const std::string &error : verification.value().errors) {
      message += ": " + error;
    }
    return Result<VerifiedBenchmarkBaseline, std::string>::makeError(
        std::move(message));
  }
  auto paths = makePaths(baselineRoot, profile.id, suite, caseId);
  if (paths.hasError()) {
    return Result<VerifiedBenchmarkBaseline, std::string>::makeError(
        "loadVerifiedBenchmarkBaseline: " + paths.error());
  }
  auto report = readBenchmarkReportFile(paths.value().reportPath);
  auto digest = sha256FileLabel(paths.value().reportPath);
  if (report.hasError() || digest.hasError() ||
      digest.value() != verification.value().reportDigest) {
    return Result<VerifiedBenchmarkBaseline, std::string>::makeError(
        report.hasError() ? "loadVerifiedBenchmarkBaseline: " + report.error()
        : digest.hasError()
            ? "loadVerifiedBenchmarkBaseline: " + digest.error()
            : "loadVerifiedBenchmarkBaseline: baseline changed during "
              "verification");
  }
  return Result<VerifiedBenchmarkBaseline, std::string>::makeResult(
      VerifiedBenchmarkBaseline{.report = std::move(report.value()),
                                .verification = std::move(verification.value()),
                                .reportPath = paths.value().reportPath});
}

Result<std::string, std::string> writeBenchmarkBaselineVerificationJson(
    const BenchmarkBaselineVerification &verification, std::string_view kind) {
  JsonMutDoc doc(yyjson_mut_doc_new(nullptr), &yyjson_mut_doc_free);
  if (!doc) {
    return Result<std::string, std::string>::makeError(
        "writeBenchmarkBaselineVerificationJson: allocation failed");
  }
  yyjson_mut_val *root = yyjson_mut_obj(doc.get());
  yyjson_mut_doc_set_root(doc.get(), root);
  yyjson_mut_obj_add_uint(doc.get(), root, "schemaVersion", 1u);
  yyjson_mut_obj_add_strncpy(doc.get(), root, "kind", kind.data(), kind.size());
  yyjson_mut_obj_add_bool(doc.get(), root, "exists", verification.exists);
  yyjson_mut_obj_add_bool(doc.get(), root, "valid", verification.valid);
  yyjson_mut_obj_add_bool(doc.get(), root, "authoritative",
                          verification.authoritative);
  addString(doc.get(), root, "caseId", verification.caseId);
  addString(doc.get(), root, "suite", verification.suite);
  addString(doc.get(), root, "profileId", verification.profileId);
  addString(doc.get(), root, "reportDigest", verification.reportDigest);
  addString(doc.get(), root, "approvalDigest", verification.approvalDigest);
  addString(doc.get(), root, "planDigest", verification.planDigest);
  addString(doc.get(), root, "sourceReportDigest",
            verification.sourceReportDigest);
  addString(doc.get(), root, "historyKey", verification.historyKey);
  if (verification.exists) {
    yyjson_mut_obj_add_val(
        doc.get(), root, "gatePolicy",
        makeGatePolicyObject(doc.get(), verification.gatePolicy));
  } else {
    yyjson_mut_obj_add_null(doc.get(), root, "gatePolicy");
  }
  yyjson_mut_val *errors = yyjson_mut_arr(doc.get());
  for (const std::string &error : verification.errors) {
    yyjson_mut_arr_add_strcpy(doc.get(), errors, error.c_str());
  }
  yyjson_mut_obj_add_val(doc.get(), root, "errors", errors);
  yyjson_mut_val *warnings = yyjson_mut_arr(doc.get());
  for (const std::string &warning : verification.warnings) {
    yyjson_mut_arr_add_strcpy(doc.get(), warnings, warning.c_str());
  }
  yyjson_mut_obj_add_val(doc.get(), root, "warnings", warnings);
  return serializeMutableDocument(doc,
                                  "writeBenchmarkBaselineVerificationJson");
}

} // namespace nuri::tools::benchmark
