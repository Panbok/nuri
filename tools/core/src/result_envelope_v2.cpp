#include "nuri/tools/core/result_envelope_v2.h"

#include "nuri/tools/core/atomic_file.h"
#include "nuri/tools/core/identifier.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <regex>
#include <unordered_set>
#include <utility>

#include <yyjson.h>

namespace nuri::tools::core {
namespace {

using JsonDocument = std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)>;
using JsonMutDocument =
    std::unique_ptr<yyjson_mut_doc, decltype(&yyjson_mut_doc_free)>;

template <typename T>
[[nodiscard]] Result<T, std::string> envelopeError(std::string message) {
  return Result<T, std::string>::makeError(std::move(message));
}

[[nodiscard]] Result<void, std::string>
rejectUnknownFields(yyjson_val *object,
                    std::initializer_list<std::string_view> allowed,
                    std::string_view path) {
  if (!yyjson_is_obj(object)) {
    return envelopeError<void>(std::string(path) + " must be an object");
  }
  std::unordered_set<std::string> seen;
  yyjson_obj_iter iterator{};
  yyjson_obj_iter_init(object, &iterator);
  yyjson_val *key = nullptr;
  while ((key = yyjson_obj_iter_next(&iterator)) != nullptr) {
    const std::string_view name(yyjson_get_str(key), yyjson_get_len(key));
    if (!seen.emplace(name).second) {
      return envelopeError<void>(std::string(path) +
                                 " contains duplicate field '" +
                                 std::string(name) + "'");
    }
    if (std::find(allowed.begin(), allowed.end(), name) == allowed.end()) {
      return envelopeError<void>(std::string(path) +
                                 " contains unknown field '" +
                                 std::string(name) + "'");
    }
  }
  return Result<void, std::string>::makeResult();
}

[[nodiscard]] Result<void, std::string>
rejectDuplicateKeysRecursively(yyjson_val *value, std::string_view path) {
  if (yyjson_is_obj(value)) {
    std::unordered_set<std::string> seen;
    yyjson_obj_iter iterator{};
    yyjson_obj_iter_init(value, &iterator);
    yyjson_val *key = nullptr;
    while ((key = yyjson_obj_iter_next(&iterator)) != nullptr) {
      const std::string_view name(yyjson_get_str(key), yyjson_get_len(key));
      if (!seen.emplace(name).second) {
        return envelopeError<void>(std::string(path) +
                                   " contains duplicate field '" +
                                   std::string(name) + "'");
      }
      auto child = rejectDuplicateKeysRecursively(yyjson_obj_iter_get_val(key),
                                                  std::string(path) + "." +
                                                      std::string(name));
      if (child.hasError()) {
        return child;
      }
    }
  } else if (yyjson_is_arr(value)) {
    size_t index = 0u;
    yyjson_arr_iter iterator{};
    yyjson_arr_iter_init(value, &iterator);
    yyjson_val *entry = nullptr;
    while ((entry = yyjson_arr_iter_next(&iterator)) != nullptr) {
      auto child = rejectDuplicateKeysRecursively(
          entry, std::string(path) + "[" + std::to_string(index++) + "]");
      if (child.hasError()) {
        return child;
      }
    }
  }
  return Result<void, std::string>::makeResult();
}

[[nodiscard]] Result<JsonDocument, std::string>
parseJsonObject(std::string_view json, std::string_view path) {
  std::string mutableJson(json);
  yyjson_read_err readError{};
  JsonDocument document(yyjson_read_opts(mutableJson.data(), mutableJson.size(),
                                         0, nullptr, &readError),
                        &yyjson_doc_free);
  if (!document) {
    return envelopeError<JsonDocument>(
        std::string(path) + " must contain valid JSON: " +
        (readError.msg != nullptr ? readError.msg : "parse error"));
  }
  yyjson_val *root = yyjson_doc_get_root(document.get());
  if (!yyjson_is_obj(root)) {
    return envelopeError<JsonDocument>(std::string(path) +
                                       " must be a JSON object");
  }
  auto duplicates = rejectDuplicateKeysRecursively(root, path);
  if (duplicates.hasError()) {
    return envelopeError<JsonDocument>(duplicates.error());
  }
  return Result<JsonDocument, std::string>::makeResult(std::move(document));
}

[[nodiscard]] Result<std::string, std::string>
jsonObjectText(yyjson_val *value, std::string_view path) {
  if (!yyjson_is_obj(value)) {
    return envelopeError<std::string>(std::string(path) + " must be an object");
  }
  size_t size = 0u;
  yyjson_write_err writeError{};
  char *text = yyjson_val_write_opts(value, 0, nullptr, &size, &writeError);
  if (text == nullptr) {
    return envelopeError<std::string>(
        std::string(path) + " could not be serialized: " +
        (writeError.msg != nullptr ? writeError.msg : "write error"));
  }
  std::string result(text, size);
  std::free(text);
  return Result<std::string, std::string>::makeResult(std::move(result));
}

[[nodiscard]] Result<std::string, std::string>
requireString(yyjson_val *object, std::string_view key, std::string_view path,
              bool allowEmpty = false) {
  yyjson_val *value = yyjson_obj_getn(object, key.data(), key.size());
  if (!yyjson_is_str(value)) {
    return envelopeError<std::string>(std::string(path) + "." +
                                      std::string(key) + " must be a string");
  }
  std::string text(yyjson_get_str(value), yyjson_get_len(value));
  if (!allowEmpty && text.empty()) {
    return envelopeError<std::string>(std::string(path) + "." +
                                      std::string(key) + " must not be empty");
  }
  return Result<std::string, std::string>::makeResult(std::move(text));
}

[[nodiscard]] Result<std::optional<std::string>, std::string>
optionalString(yyjson_val *object, std::string_view key,
               std::string_view path) {
  yyjson_val *value = yyjson_obj_getn(object, key.data(), key.size());
  if (value == nullptr) {
    return Result<std::optional<std::string>, std::string>::makeResult(
        std::optional<std::string>{});
  }
  if (!yyjson_is_str(value)) {
    return envelopeError<std::optional<std::string>>(
        std::string(path) + "." + std::string(key) + " must be a string");
  }
  return Result<std::optional<std::string>, std::string>::makeResult(
      std::optional<std::string>{
          std::string(yyjson_get_str(value), yyjson_get_len(value))});
}

[[nodiscard]] Result<uint64_t, std::string>
requireUint(yyjson_val *object, std::string_view key, std::string_view path) {
  yyjson_val *value = yyjson_obj_getn(object, key.data(), key.size());
  if (!yyjson_is_uint(value)) {
    return envelopeError<uint64_t>(std::string(path) + "." + std::string(key) +
                                   " must be a non-negative integer");
  }
  return Result<uint64_t, std::string>::makeResult(yyjson_get_uint(value));
}

[[nodiscard]] Result<int, std::string> requireExitCode(yyjson_val *object,
                                                       std::string_view key,
                                                       std::string_view path) {
  auto value = requireUint(object, key, path);
  if (value.hasError() || value.value() > 5u) {
    return envelopeError<int>(value.hasError()
                                  ? value.error()
                                  : std::string(path) + "." + std::string(key) +
                                        " must be between 0 and 5");
  }
  return Result<int, std::string>::makeResult(static_cast<int>(value.value()));
}

[[nodiscard]] Result<bool, std::string>
requireBool(yyjson_val *object, std::string_view key, std::string_view path) {
  yyjson_val *value = yyjson_obj_getn(object, key.data(), key.size());
  if (!yyjson_is_bool(value)) {
    return envelopeError<bool>(std::string(path) + "." + std::string(key) +
                               " must be a boolean");
  }
  return Result<bool, std::string>::makeResult(yyjson_get_bool(value));
}

[[nodiscard]] Result<yyjson_val *, std::string>
requireObject(yyjson_val *object, std::string_view key, std::string_view path) {
  yyjson_val *value = yyjson_obj_getn(object, key.data(), key.size());
  if (!yyjson_is_obj(value)) {
    return envelopeError<yyjson_val *>(std::string(path) + "." +
                                       std::string(key) + " must be an object");
  }
  return Result<yyjson_val *, std::string>::makeResult(value);
}

[[nodiscard]] Result<yyjson_val *, std::string>
requireArray(yyjson_val *object, std::string_view key, std::string_view path) {
  yyjson_val *value = yyjson_obj_getn(object, key.data(), key.size());
  if (!yyjson_is_arr(value)) {
    return envelopeError<yyjson_val *>(std::string(path) + "." +
                                       std::string(key) + " must be an array");
  }
  return Result<yyjson_val *, std::string>::makeResult(value);
}

[[nodiscard]] Result<std::vector<std::string>, std::string>
readStringArray(yyjson_val *array, std::string_view path) {
  if (!yyjson_is_arr(array)) {
    return envelopeError<std::vector<std::string>>(std::string(path) +
                                                   " must be an array");
  }
  std::vector<std::string> result;
  result.reserve(yyjson_arr_size(array));
  size_t index = 0u;
  yyjson_arr_iter iterator{};
  yyjson_arr_iter_init(array, &iterator);
  yyjson_val *entry = nullptr;
  while ((entry = yyjson_arr_iter_next(&iterator)) != nullptr) {
    if (!yyjson_is_str(entry)) {
      return envelopeError<std::vector<std::string>>(std::string(path) + "[" +
                                                     std::to_string(index) +
                                                     "] must be a string");
    }
    result.emplace_back(yyjson_get_str(entry), yyjson_get_len(entry));
    ++index;
  }
  return Result<std::vector<std::string>, std::string>::makeResult(
      std::move(result));
}

[[nodiscard]] Result<ResultToolV2, std::string>
parseTool(std::string_view value) {
  if (value == "benchmark") {
    return Result<ResultToolV2, std::string>::makeResult(
        ResultToolV2::Benchmark);
  }
  if (value == "snapshot") {
    return Result<ResultToolV2, std::string>::makeResult(
        ResultToolV2::Snapshot);
  }
  if (value == "autotest") {
    return Result<ResultToolV2, std::string>::makeResult(
        ResultToolV2::Autotest);
  }
  if (value == "validate") {
    return Result<ResultToolV2, std::string>::makeResult(
        ResultToolV2::Validate);
  }
  return envelopeError<ResultToolV2>("$.tool has an unsupported value");
}

[[nodiscard]] Result<ToolOutcome, std::string>
parseOutcome(std::string_view value) {
  for (ToolOutcome outcome :
       {ToolOutcome::Pass, ToolOutcome::Warn, ToolOutcome::Failure,
        ToolOutcome::Invalid, ToolOutcome::EnvironmentUnavailable,
        ToolOutcome::MissingBaseline, ToolOutcome::RuntimeError,
        ToolOutcome::Cancelled, ToolOutcome::Incomplete,
        ToolOutcome::Investigative}) {
    if (value == toolOutcomeName(outcome)) {
      return Result<ToolOutcome, std::string>::makeResult(outcome);
    }
  }
  return envelopeError<ToolOutcome>("$.status has an unsupported value");
}

[[nodiscard]] Result<ResultDiagnosticSeverityV2, std::string>
parseSeverity(std::string_view value, std::string_view path) {
  if (value == "info") {
    return Result<ResultDiagnosticSeverityV2, std::string>::makeResult(
        ResultDiagnosticSeverityV2::Info);
  }
  if (value == "warning") {
    return Result<ResultDiagnosticSeverityV2, std::string>::makeResult(
        ResultDiagnosticSeverityV2::Warning);
  }
  if (value == "error") {
    return Result<ResultDiagnosticSeverityV2, std::string>::makeResult(
        ResultDiagnosticSeverityV2::Error);
  }
  return envelopeError<ResultDiagnosticSeverityV2>(std::string(path) +
                                                   " has an unsupported value");
}

[[nodiscard]] Result<ResultArtifactStatusV2, std::string>
parseArtifactStatus(std::string_view value, std::string_view path) {
  if (value == "complete") {
    return Result<ResultArtifactStatusV2, std::string>::makeResult(
        ResultArtifactStatusV2::Complete);
  }
  if (value == "missing") {
    return Result<ResultArtifactStatusV2, std::string>::makeResult(
        ResultArtifactStatusV2::Missing);
  }
  if (value == "invalid") {
    return Result<ResultArtifactStatusV2, std::string>::makeResult(
        ResultArtifactStatusV2::Invalid);
  }
  return envelopeError<ResultArtifactStatusV2>(std::string(path) +
                                               " has an unsupported value");
}

[[nodiscard]] std::string pathToUtf8(const std::filesystem::path &path) {
  const std::u8string value = path.generic_u8string();
  return {reinterpret_cast<const char *>(value.data()), value.size()};
}

[[nodiscard]] std::filesystem::path pathFromUtf8(std::string_view value) {
  const std::u8string encoded(reinterpret_cast<const char8_t *>(value.data()),
                              value.size());
  return std::filesystem::path(encoded);
}

[[nodiscard]] Result<void, std::string>
validateRelativePath(std::string_view value, std::string_view path) {
  if (value.empty() || value.front() == '/' || value.front() == '\\' ||
      (value.size() >= 2u && value[1] == ':')) {
    return envelopeError<void>(std::string(path) +
                               " must be a non-empty relative path");
  }
  for (const unsigned char character : value) {
    if (character <= 0x1fu || character == ':') {
      return envelopeError<void>(std::string(path) +
                                 " contains a forbidden character");
    }
  }
  size_t start = 0u;
  while (start <= value.size()) {
    const size_t end = value.find_first_of("/\\", start);
    const std::string_view segment =
        value.substr(start, end == std::string_view::npos ? value.size() - start
                                                          : end - start);
    if (segment == "..") {
      return envelopeError<void>(std::string(path) +
                                 " must not contain parent traversal");
    }
    if (end == std::string_view::npos) {
      break;
    }
    start = end + 1u;
  }
  return Result<void, std::string>::makeResult();
}

[[nodiscard]] bool matches(std::string_view value, const char *pattern) {
  return std::regex_match(value.begin(), value.end(), std::regex(pattern));
}

[[nodiscard]] bool validDateTime(std::string_view value) {
  if (!matches(value, "^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}("
                      "\\.[0-9]+)?(Z|[+-][0-9]{2}:[0-9]{2})$")) {
    return false;
  }
  const auto digits = [value](size_t offset) {
    return static_cast<unsigned>((value[offset] - '0') * 10 +
                                 (value[offset + 1u] - '0'));
  };
  const unsigned year =
      static_cast<unsigned>((value[0] - '0') * 1000 + (value[1] - '0') * 100 +
                            (value[2] - '0') * 10 + (value[3] - '0'));
  const unsigned month = digits(5u);
  const unsigned day = digits(8u);
  const unsigned hour = digits(11u);
  const unsigned minute = digits(14u);
  const unsigned second = digits(17u);
  if (month == 0u || month > 12u || hour > 23u || minute > 59u ||
      second > 59u) {
    return false;
  }
  constexpr unsigned daysByMonth[] = {0u,  31u, 28u, 31u, 30u, 31u, 30u,
                                      31u, 31u, 30u, 31u, 30u, 31u};
  unsigned days = daysByMonth[month];
  if (month == 2u &&
      (year % 400u == 0u || (year % 4u == 0u && year % 100u != 0u))) {
    ++days;
  }
  if (day == 0u || day > days) {
    return false;
  }
  const size_t zone = value.find_last_of("Z+-");
  return zone != std::string_view::npos &&
         (value[zone] == 'Z' ||
          (digits(zone + 1u) <= 23u && digits(zone + 4u) <= 59u));
}

[[nodiscard]] Result<void, std::string>
validateIdentifierField(std::string_view value, std::string_view path) {
  return validateIdentifier(value, path, IdentifierShape::Dotted);
}

void addString(yyjson_mut_doc *document, yyjson_mut_val *object,
               const char *key, std::string_view value) {
  yyjson_mut_obj_add_strncpy(document, object, key, value.data(), value.size());
}

[[nodiscard]] yyjson_mut_val *
makeStringArray(yyjson_mut_doc *document,
                const std::vector<std::string> &values) {
  yyjson_mut_val *array = yyjson_mut_arr(document);
  for (const std::string &value : values) {
    yyjson_mut_arr_add_strncpy(document, array, value.data(), value.size());
  }
  return array;
}

[[nodiscard]] Result<yyjson_mut_val *, std::string>
copyJsonObject(yyjson_mut_doc *target, std::string_view json,
               std::string_view path) {
  auto source = parseJsonObject(json, path);
  if (source.hasError()) {
    return envelopeError<yyjson_mut_val *>(source.error());
  }
  yyjson_mut_val *copy =
      yyjson_val_mut_copy(target, yyjson_doc_get_root(source.value().get()));
  if (copy == nullptr) {
    return envelopeError<yyjson_mut_val *>(std::string(path) +
                                           " could not be copied");
  }
  return Result<yyjson_mut_val *, std::string>::makeResult(copy);
}

[[nodiscard]] Result<void, std::string>
validateSelection(const ResultSelectionV2 &selection) {
  const auto sumExceeds = [](uint64_t first, uint64_t second, uint64_t third,
                             uint64_t limit) {
    return first > limit || second > limit - first ||
           third > limit - first - second;
  };
  if (selection.attempted > selection.selected ||
      selection.completed > selection.attempted ||
      sumExceeds(selection.passed, selection.warned, selection.failed,
                 selection.completed) ||
      sumExceeds(selection.skipped, selection.unavailable, selection.notRun,
                 selection.selected)) {
    return envelopeError<void>("$.selection counts are inconsistent");
  }
  return Result<void, std::string>::makeResult();
}

} // namespace

std::string_view resultToolV2Name(ResultToolV2 tool) noexcept {
  switch (tool) {
  case ResultToolV2::Benchmark:
    return "benchmark";
  case ResultToolV2::Snapshot:
    return "snapshot";
  case ResultToolV2::Autotest:
    return "autotest";
  case ResultToolV2::Validate:
    return "validate";
  }
  return "validate";
}

std::string_view
resultDiagnosticSeverityV2Name(ResultDiagnosticSeverityV2 severity) noexcept {
  switch (severity) {
  case ResultDiagnosticSeverityV2::Info:
    return "info";
  case ResultDiagnosticSeverityV2::Warning:
    return "warning";
  case ResultDiagnosticSeverityV2::Error:
    return "error";
  }
  return "error";
}

std::string_view
resultArtifactStatusV2Name(ResultArtifactStatusV2 status) noexcept {
  switch (status) {
  case ResultArtifactStatusV2::Complete:
    return "complete";
  case ResultArtifactStatusV2::Missing:
    return "missing";
  case ResultArtifactStatusV2::Invalid:
    return "invalid";
  }
  return "invalid";
}

Result<void, std::string>
validateResultEnvelopeV2(const ResultEnvelopeV2 &envelope) {
  if (!matches(envelope.runId,
               "^[0-9]{8}T[0-9]{6}\\.[0-9]{3,6}Z-[A-Za-z0-9_-]+$")) {
    return envelopeError<void>("$.runId has an invalid format");
  }
  if (envelope.exitCode < 0 || envelope.exitCode > 5 ||
      envelope.exitCode != toolOutcomeExitCode(envelope.status)) {
    return envelopeError<void>("$.exitCode does not match $.status");
  }
  if (envelope.startedAtUtc.has_value() &&
      !validDateTime(*envelope.startedAtUtc)) {
    return envelopeError<void>("$.startedAtUtc must be an RFC 3339 date-time");
  }
  if (envelope.durationMs.has_value() &&
      (!std::isfinite(*envelope.durationMs) || *envelope.durationMs < 0.0)) {
    return envelopeError<void>("$.durationMs must be finite and non-negative");
  }
  auto selection = validateSelection(envelope.selection);
  if (selection.hasError()) {
    return selection;
  }
  if (envelope.tool != ResultToolV2::Validate &&
      (envelope.status == ToolOutcome::Pass ||
       envelope.status == ToolOutcome::Investigative ||
       envelope.status == ToolOutcome::Warn) &&
      envelope.selection.selected == 0u) {
    return envelopeError<void>(
        "successful tool outcomes require a non-empty selection");
  }
  if (envelope.status == ToolOutcome::Pass &&
      (envelope.selection.failed > 0u || envelope.selection.unavailable > 0u)) {
    return envelopeError<void>(
        "$.status cannot be pass when selection has failures or unavailable "
        "work");
  }
  if (envelope.authoritative && envelope.status != ToolOutcome::Pass) {
    return envelopeError<void>("$.authoritative requires $.status pass");
  }
  if (envelope.profile.has_value()) {
    auto validId =
        validateIdentifierField(envelope.profile->id, "$.profile.id");
    if (validId.hasError()) {
      return validId;
    }
    if (envelope.status == ToolOutcome::Pass && !envelope.profile->compatible) {
      return envelopeError<void>(
          "$.status cannot be pass when $.profile.compatible is false");
    }
    if (envelope.authoritative && !envelope.profile->compatible) {
      return envelopeError<void>(
          "$.authoritative requires a compatible profile");
    }
  }
  if (envelope.authoritative && !envelope.profile.has_value()) {
    return envelopeError<void>(
        "$.authoritative requires an explicit compatible profile");
  }
  const auto validateFingerprint = [](const std::optional<std::string> &value,
                                      std::string_view path) {
    if (value.has_value() && !matches(*value, "^sha256:[0-9a-f]{64}$")) {
      return envelopeError<void>(std::string(path) +
                                 " must be a SHA-256 fingerprint");
    }
    return Result<void, std::string>::makeResult();
  };
  auto environmentFingerprint = validateFingerprint(
      envelope.environmentFingerprint, "$.environmentFingerprint");
  if (environmentFingerprint.hasError()) {
    return environmentFingerprint;
  }
  auto workloadFingerprint = validateFingerprint(envelope.workloadFingerprint,
                                                 "$.workloadFingerprint");
  if (workloadFingerprint.hasError()) {
    return workloadFingerprint;
  }
  for (size_t index = 0u; index < envelope.diagnostics.size(); ++index) {
    const ResultDiagnosticV2 &diagnostic = envelope.diagnostics[index];
    const std::string path = "$.diagnostics[" + std::to_string(index) + "]";
    auto validCode = validateIdentifierField(diagnostic.code, path + ".code");
    if (validCode.hasError()) {
      return validCode;
    }
    if (diagnostic.contextJson.has_value()) {
      auto context =
          parseJsonObject(*diagnostic.contextJson, path + ".context");
      if (context.hasError()) {
        return envelopeError<void>(context.error());
      }
    }
  }
  for (size_t index = 0u; index < envelope.artifacts.size(); ++index) {
    const ResultArtifactV2 &artifact = envelope.artifacts[index];
    const std::string path = "$.artifacts[" + std::to_string(index) + "]";
    auto validRole = validateIdentifierField(artifact.role, path + ".role");
    if (validRole.hasError()) {
      return validRole;
    }
    auto validPath =
        validateRelativePath(pathToUtf8(artifact.path), path + ".path");
    if (validPath.hasError()) {
      return validPath;
    }
    if (!matches(artifact.digest, "^(sha256|blake3):[0-9a-f]{64}$")) {
      return envelopeError<void>(path + ".digest has an invalid format");
    }
  }
  for (size_t index = 0u; index < envelope.children.size(); ++index) {
    const ResultChildV2 &child = envelope.children[index];
    const std::string path = "$.children[" + std::to_string(index) + "]";
    auto validId = validateIdentifierField(child.id, path + ".id");
    if (validId.hasError()) {
      return validId;
    }
    if (child.status.empty() || child.exitCode < 0 || child.exitCode > 5) {
      return envelopeError<void>(path +
                                 " must have a status and exit code 0..5");
    }
    if (child.result.has_value()) {
      auto validPath =
          validateRelativePath(pathToUtf8(*child.result), path + ".result");
      if (validPath.hasError()) {
        return validPath;
      }
    }
  }
  auto payload = parseJsonObject(envelope.payloadJson, "$.payload");
  if (payload.hasError()) {
    return envelopeError<void>(payload.error());
  }
  if (envelope.extensionsJson.has_value()) {
    auto extensions = parseJsonObject(*envelope.extensionsJson, "$.extensions");
    if (extensions.hasError()) {
      return envelopeError<void>(extensions.error());
    }
  }
  return Result<void, std::string>::makeResult();
}

Result<std::string, std::string>
serializeResultEnvelopeV2(const ResultEnvelopeV2 &envelope) {
  auto valid = validateResultEnvelopeV2(envelope);
  if (valid.hasError()) {
    return envelopeError<std::string>(valid.error());
  }

  JsonMutDocument document(yyjson_mut_doc_new(nullptr), &yyjson_mut_doc_free);
  if (!document) {
    return envelopeError<std::string>("failed to allocate result document");
  }
  yyjson_mut_val *root = yyjson_mut_obj(document.get());
  yyjson_mut_doc_set_root(document.get(), root);
  yyjson_mut_obj_add_uint(document.get(), root, "schemaVersion", 2u);
  addString(document.get(), root, "kind", "nuri.tool.result");
  addString(document.get(), root, "tool", resultToolV2Name(envelope.tool));
  if (envelope.toolVersion.has_value()) {
    addString(document.get(), root, "toolVersion", *envelope.toolVersion);
  }
  addString(document.get(), root, "runId", envelope.runId);
  addString(document.get(), root, "status", toolOutcomeName(envelope.status));
  yyjson_mut_obj_add_int(document.get(), root, "exitCode", envelope.exitCode);
  yyjson_mut_obj_add_bool(document.get(), root, "authoritative",
                          envelope.authoritative);
  if (envelope.startedAtUtc.has_value()) {
    addString(document.get(), root, "startedAtUtc", *envelope.startedAtUtc);
  }
  if (envelope.durationMs.has_value()) {
    yyjson_mut_obj_add_real(document.get(), root, "durationMs",
                            *envelope.durationMs);
  }
  if (envelope.command.has_value()) {
    yyjson_mut_obj_add_val(document.get(), root, "command",
                           makeStringArray(document.get(), *envelope.command));
  }
  if (envelope.reproduceCommand.has_value()) {
    addString(document.get(), root, "reproduceCommand",
              *envelope.reproduceCommand);
  }

  yyjson_mut_val *selection = yyjson_mut_obj(document.get());
  if (envelope.selection.requested.has_value()) {
    addString(document.get(), selection, "requested",
              *envelope.selection.requested);
  }
  yyjson_mut_obj_add_uint(document.get(), selection, "selected",
                          envelope.selection.selected);
  yyjson_mut_obj_add_uint(document.get(), selection, "attempted",
                          envelope.selection.attempted);
  yyjson_mut_obj_add_uint(document.get(), selection, "completed",
                          envelope.selection.completed);
  yyjson_mut_obj_add_uint(document.get(), selection, "passed",
                          envelope.selection.passed);
  yyjson_mut_obj_add_uint(document.get(), selection, "warned",
                          envelope.selection.warned);
  yyjson_mut_obj_add_uint(document.get(), selection, "failed",
                          envelope.selection.failed);
  yyjson_mut_obj_add_uint(document.get(), selection, "skipped",
                          envelope.selection.skipped);
  yyjson_mut_obj_add_uint(document.get(), selection, "unavailable",
                          envelope.selection.unavailable);
  yyjson_mut_obj_add_uint(document.get(), selection, "notRun",
                          envelope.selection.notRun);
  yyjson_mut_obj_add_val(document.get(), root, "selection", selection);

  if (envelope.profile.has_value()) {
    yyjson_mut_val *profile = yyjson_mut_obj(document.get());
    addString(document.get(), profile, "id", envelope.profile->id);
    yyjson_mut_obj_add_bool(document.get(), profile, "compatible",
                            envelope.profile->compatible);
    yyjson_mut_obj_add_val(
        document.get(), profile, "incompatibilityReasons",
        makeStringArray(document.get(),
                        envelope.profile->incompatibilityReasons));
    yyjson_mut_obj_add_val(document.get(), root, "profile", profile);
  }
  if (envelope.environmentFingerprint.has_value()) {
    addString(document.get(), root, "environmentFingerprint",
              *envelope.environmentFingerprint);
  }
  if (envelope.workloadFingerprint.has_value()) {
    addString(document.get(), root, "workloadFingerprint",
              *envelope.workloadFingerprint);
  }

  yyjson_mut_val *diagnostics = yyjson_mut_arr(document.get());
  for (const ResultDiagnosticV2 &diagnostic : envelope.diagnostics) {
    yyjson_mut_val *object = yyjson_mut_obj(document.get());
    addString(document.get(), object, "code", diagnostic.code);
    addString(document.get(), object, "severity",
              resultDiagnosticSeverityV2Name(diagnostic.severity));
    addString(document.get(), object, "message", diagnostic.message);
    if (diagnostic.contextJson.has_value()) {
      auto context = copyJsonObject(document.get(), *diagnostic.contextJson,
                                    "$.diagnostics[].context");
      if (context.hasError()) {
        return envelopeError<std::string>(context.error());
      }
      yyjson_mut_obj_add_val(document.get(), object, "context",
                             context.value());
    }
    yyjson_mut_arr_add_val(diagnostics, object);
  }
  yyjson_mut_obj_add_val(document.get(), root, "diagnostics", diagnostics);

  yyjson_mut_val *artifacts = yyjson_mut_arr(document.get());
  for (const ResultArtifactV2 &artifact : envelope.artifacts) {
    yyjson_mut_val *object = yyjson_mut_obj(document.get());
    addString(document.get(), object, "role", artifact.role);
    addString(document.get(), object, "path", pathToUtf8(artifact.path));
    addString(document.get(), object, "mediaType", artifact.mediaType);
    addString(document.get(), object, "digest", artifact.digest);
    addString(document.get(), object, "status",
              resultArtifactStatusV2Name(artifact.status));
    yyjson_mut_arr_add_val(artifacts, object);
  }
  yyjson_mut_obj_add_val(document.get(), root, "artifacts", artifacts);

  yyjson_mut_val *children = yyjson_mut_arr(document.get());
  for (const ResultChildV2 &child : envelope.children) {
    yyjson_mut_val *object = yyjson_mut_obj(document.get());
    addString(document.get(), object, "id", child.id);
    addString(document.get(), object, "status", child.status);
    yyjson_mut_obj_add_int(document.get(), object, "exitCode", child.exitCode);
    if (child.result.has_value()) {
      addString(document.get(), object, "result", pathToUtf8(*child.result));
    }
    yyjson_mut_arr_add_val(children, object);
  }
  yyjson_mut_obj_add_val(document.get(), root, "children", children);

  auto payload =
      copyJsonObject(document.get(), envelope.payloadJson, "$.payload");
  if (payload.hasError()) {
    return envelopeError<std::string>(payload.error());
  }
  yyjson_mut_obj_add_val(document.get(), root, "payload", payload.value());
  if (envelope.extensionsJson.has_value()) {
    auto extensions = copyJsonObject(document.get(), *envelope.extensionsJson,
                                     "$.extensions");
    if (extensions.hasError()) {
      return envelopeError<std::string>(extensions.error());
    }
    yyjson_mut_obj_add_val(document.get(), root, "extensions",
                           extensions.value());
  }

  size_t size = 0u;
  yyjson_write_err writeError{};
  char *text = yyjson_mut_write_opts(document.get(), YYJSON_WRITE_PRETTY,
                                     nullptr, &size, &writeError);
  if (text == nullptr) {
    return envelopeError<std::string>(
        std::string("failed to serialize result envelope: ") +
        (writeError.msg != nullptr ? writeError.msg : "write error"));
  }
  std::string result(text, size);
  std::free(text);
  return Result<std::string, std::string>::makeResult(std::move(result));
}

Result<ResultEnvelopeV2, std::string>
readResultEnvelopeV2(std::string_view json) {
  std::string mutableJson(json);
  yyjson_read_err readError{};
  JsonDocument document(yyjson_read_opts(mutableJson.data(), mutableJson.size(),
                                         0, nullptr, &readError),
                        &yyjson_doc_free);
  if (!document) {
    return envelopeError<ResultEnvelopeV2>(
        std::string("failed to parse result envelope: ") +
        (readError.msg != nullptr ? readError.msg : "parse error"));
  }
  yyjson_val *root = yyjson_doc_get_root(document.get());
  auto fields = rejectUnknownFields(root,
                                    {"schemaVersion",
                                     "kind",
                                     "tool",
                                     "toolVersion",
                                     "runId",
                                     "status",
                                     "exitCode",
                                     "authoritative",
                                     "startedAtUtc",
                                     "durationMs",
                                     "command",
                                     "reproduceCommand",
                                     "selection",
                                     "profile",
                                     "environmentFingerprint",
                                     "workloadFingerprint",
                                     "diagnostics",
                                     "artifacts",
                                     "children",
                                     "payload",
                                     "extensions"},
                                    "$");
  if (fields.hasError()) {
    return envelopeError<ResultEnvelopeV2>(fields.error());
  }
  auto duplicates = rejectDuplicateKeysRecursively(root, "$");
  if (duplicates.hasError()) {
    return envelopeError<ResultEnvelopeV2>(duplicates.error());
  }
  auto schema = requireUint(root, "schemaVersion", "$");
  auto kind = requireString(root, "kind", "$", true);
  if (schema.hasError() || schema.value() != 2u || kind.hasError() ||
      kind.value() != "nuri.tool.result") {
    return envelopeError<ResultEnvelopeV2>(
        schema.hasError()      ? schema.error()
        : schema.value() != 2u ? "$.schemaVersion must equal 2"
        : kind.hasError()      ? kind.error()
                               : "$.kind must equal nuri.tool.result");
  }

  ResultEnvelopeV2 envelope{};
  auto toolText = requireString(root, "tool", "$", true);
  auto runId = requireString(root, "runId", "$", true);
  auto statusText = requireString(root, "status", "$", true);
  auto exitCode = requireExitCode(root, "exitCode", "$");
  auto authoritative = requireBool(root, "authoritative", "$");
  if (toolText.hasError() || runId.hasError() || statusText.hasError() ||
      exitCode.hasError() || authoritative.hasError()) {
    return envelopeError<ResultEnvelopeV2>(
        toolText.hasError()     ? toolText.error()
        : runId.hasError()      ? runId.error()
        : statusText.hasError() ? statusText.error()
        : exitCode.hasError()   ? exitCode.error()
                                : authoritative.error());
  }
  auto tool = parseTool(toolText.value());
  auto status = parseOutcome(statusText.value());
  if (tool.hasError() || status.hasError()) {
    return envelopeError<ResultEnvelopeV2>(tool.hasError() ? tool.error()
                                                           : status.error());
  }
  envelope.tool = tool.value();
  envelope.runId = std::move(runId.value());
  envelope.status = status.value();
  envelope.exitCode = exitCode.value();
  envelope.authoritative = authoritative.value();

  auto toolVersion = optionalString(root, "toolVersion", "$");
  auto startedAtUtc = optionalString(root, "startedAtUtc", "$");
  auto reproduce = optionalString(root, "reproduceCommand", "$");
  auto environmentFingerprint =
      optionalString(root, "environmentFingerprint", "$");
  auto workloadFingerprint = optionalString(root, "workloadFingerprint", "$");
  if (toolVersion.hasError() || startedAtUtc.hasError() ||
      reproduce.hasError() || environmentFingerprint.hasError() ||
      workloadFingerprint.hasError()) {
    return envelopeError<ResultEnvelopeV2>(
        toolVersion.hasError()              ? toolVersion.error()
        : startedAtUtc.hasError()           ? startedAtUtc.error()
        : reproduce.hasError()              ? reproduce.error()
        : environmentFingerprint.hasError() ? environmentFingerprint.error()
                                            : workloadFingerprint.error());
  }
  envelope.toolVersion = std::move(toolVersion.value());
  envelope.startedAtUtc = std::move(startedAtUtc.value());
  envelope.reproduceCommand = std::move(reproduce.value());
  envelope.environmentFingerprint = std::move(environmentFingerprint.value());
  envelope.workloadFingerprint = std::move(workloadFingerprint.value());

  yyjson_val *duration = yyjson_obj_get(root, "durationMs");
  if (duration != nullptr) {
    if (!yyjson_is_num(duration)) {
      return envelopeError<ResultEnvelopeV2>("$.durationMs must be a number");
    }
    envelope.durationMs = yyjson_get_num(duration);
  }
  yyjson_val *command = yyjson_obj_get(root, "command");
  if (command != nullptr) {
    auto values = readStringArray(command, "$.command");
    if (values.hasError()) {
      return envelopeError<ResultEnvelopeV2>(values.error());
    }
    envelope.command = std::move(values.value());
  }

  auto selectionObject = requireObject(root, "selection", "$");
  if (selectionObject.hasError()) {
    return envelopeError<ResultEnvelopeV2>(selectionObject.error());
  }
  auto selectionFields = rejectUnknownFields(
      selectionObject.value(),
      {"requested", "selected", "attempted", "completed", "passed", "warned",
       "failed", "skipped", "unavailable", "notRun"},
      "$.selection");
  if (selectionFields.hasError()) {
    return envelopeError<ResultEnvelopeV2>(selectionFields.error());
  }
  auto requested =
      optionalString(selectionObject.value(), "requested", "$.selection");
  auto selected =
      requireUint(selectionObject.value(), "selected", "$.selection");
  auto attempted =
      requireUint(selectionObject.value(), "attempted", "$.selection");
  auto completed =
      requireUint(selectionObject.value(), "completed", "$.selection");
  auto passed = requireUint(selectionObject.value(), "passed", "$.selection");
  auto warned = requireUint(selectionObject.value(), "warned", "$.selection");
  auto failed = requireUint(selectionObject.value(), "failed", "$.selection");
  auto skipped = requireUint(selectionObject.value(), "skipped", "$.selection");
  auto unavailable =
      requireUint(selectionObject.value(), "unavailable", "$.selection");
  auto notRun = requireUint(selectionObject.value(), "notRun", "$.selection");
  if (requested.hasError() || selected.hasError() || attempted.hasError() ||
      completed.hasError() || passed.hasError() || warned.hasError() ||
      failed.hasError() || skipped.hasError() || unavailable.hasError() ||
      notRun.hasError()) {
    return envelopeError<ResultEnvelopeV2>(
        "$.selection is incomplete or invalid");
  }
  envelope.selection = {.requested = std::move(requested.value()),
                        .selected = selected.value(),
                        .attempted = attempted.value(),
                        .completed = completed.value(),
                        .passed = passed.value(),
                        .warned = warned.value(),
                        .failed = failed.value(),
                        .skipped = skipped.value(),
                        .unavailable = unavailable.value(),
                        .notRun = notRun.value()};

  yyjson_val *profileObject = yyjson_obj_get(root, "profile");
  if (profileObject != nullptr) {
    auto profileFields = rejectUnknownFields(
        profileObject, {"id", "compatible", "incompatibilityReasons"},
        "$.profile");
    if (profileFields.hasError()) {
      return envelopeError<ResultEnvelopeV2>(profileFields.error());
    }
    auto id = requireString(profileObject, "id", "$.profile", true);
    auto compatible = requireBool(profileObject, "compatible", "$.profile");
    auto reasonsArray =
        requireArray(profileObject, "incompatibilityReasons", "$.profile");
    if (id.hasError() || compatible.hasError() || reasonsArray.hasError()) {
      return envelopeError<ResultEnvelopeV2>(
          "$.profile is incomplete or invalid");
    }
    auto reasons = readStringArray(reasonsArray.value(),
                                   "$.profile.incompatibilityReasons");
    if (reasons.hasError()) {
      return envelopeError<ResultEnvelopeV2>(reasons.error());
    }
    envelope.profile =
        ResultProfileV2{.id = std::move(id.value()),
                        .compatible = compatible.value(),
                        .incompatibilityReasons = std::move(reasons.value())};
  }

  auto diagnosticsArray = requireArray(root, "diagnostics", "$");
  auto artifactsArray = requireArray(root, "artifacts", "$");
  auto childrenArray = requireArray(root, "children", "$");
  if (diagnosticsArray.hasError() || artifactsArray.hasError() ||
      childrenArray.hasError()) {
    return envelopeError<ResultEnvelopeV2>(
        diagnosticsArray.hasError() ? diagnosticsArray.error()
        : artifactsArray.hasError() ? artifactsArray.error()
                                    : childrenArray.error());
  }

  size_t index = 0u;
  yyjson_arr_iter arrayIterator{};
  yyjson_arr_iter_init(diagnosticsArray.value(), &arrayIterator);
  yyjson_val *entry = nullptr;
  while ((entry = yyjson_arr_iter_next(&arrayIterator)) != nullptr) {
    const std::string path = "$.diagnostics[" + std::to_string(index++) + "]";
    auto entryFields = rejectUnknownFields(
        entry, {"code", "severity", "message", "context"}, path);
    if (entryFields.hasError()) {
      return envelopeError<ResultEnvelopeV2>(entryFields.error());
    }
    auto code = requireString(entry, "code", path, true);
    auto severityText = requireString(entry, "severity", path, true);
    auto message = requireString(entry, "message", path, true);
    if (code.hasError() || severityText.hasError() || message.hasError()) {
      return envelopeError<ResultEnvelopeV2>(path +
                                             " is incomplete or invalid");
    }
    auto severity = parseSeverity(severityText.value(), path + ".severity");
    if (severity.hasError()) {
      return envelopeError<ResultEnvelopeV2>(severity.error());
    }
    ResultDiagnosticV2 diagnostic{.code = std::move(code.value()),
                                  .severity = severity.value(),
                                  .message = std::move(message.value())};
    yyjson_val *context = yyjson_obj_get(entry, "context");
    if (context != nullptr) {
      auto contextText = jsonObjectText(context, path + ".context");
      if (contextText.hasError()) {
        return envelopeError<ResultEnvelopeV2>(contextText.error());
      }
      diagnostic.contextJson = std::move(contextText.value());
    }
    envelope.diagnostics.push_back(std::move(diagnostic));
  }

  index = 0u;
  yyjson_arr_iter_init(artifactsArray.value(), &arrayIterator);
  while ((entry = yyjson_arr_iter_next(&arrayIterator)) != nullptr) {
    const std::string path = "$.artifacts[" + std::to_string(index++) + "]";
    auto entryFields = rejectUnknownFields(
        entry, {"role", "path", "mediaType", "digest", "status"}, path);
    if (entryFields.hasError()) {
      return envelopeError<ResultEnvelopeV2>(entryFields.error());
    }
    auto role = requireString(entry, "role", path, true);
    auto artifactPath = requireString(entry, "path", path, true);
    auto mediaType = requireString(entry, "mediaType", path, true);
    auto digest = requireString(entry, "digest", path, true);
    auto statusText = requireString(entry, "status", path, true);
    if (role.hasError() || artifactPath.hasError() || mediaType.hasError() ||
        digest.hasError() || statusText.hasError()) {
      return envelopeError<ResultEnvelopeV2>(path +
                                             " is incomplete or invalid");
    }
    auto artifactStatus =
        parseArtifactStatus(statusText.value(), path + ".status");
    if (artifactStatus.hasError()) {
      return envelopeError<ResultEnvelopeV2>(artifactStatus.error());
    }
    envelope.artifacts.push_back({.role = std::move(role.value()),
                                  .path = pathFromUtf8(artifactPath.value()),
                                  .mediaType = std::move(mediaType.value()),
                                  .digest = std::move(digest.value()),
                                  .status = artifactStatus.value()});
  }

  index = 0u;
  yyjson_arr_iter_init(childrenArray.value(), &arrayIterator);
  while ((entry = yyjson_arr_iter_next(&arrayIterator)) != nullptr) {
    const std::string path = "$.children[" + std::to_string(index++) + "]";
    auto entryFields = rejectUnknownFields(
        entry, {"id", "status", "exitCode", "result"}, path);
    if (entryFields.hasError()) {
      return envelopeError<ResultEnvelopeV2>(entryFields.error());
    }
    auto id = requireString(entry, "id", path, true);
    auto childStatus = requireString(entry, "status", path, true);
    auto childExit = requireExitCode(entry, "exitCode", path);
    auto result = optionalString(entry, "result", path);
    if (id.hasError() || childStatus.hasError() || childExit.hasError() ||
        result.hasError()) {
      return envelopeError<ResultEnvelopeV2>(path +
                                             " is incomplete or invalid");
    }
    ResultChildV2 child{.id = std::move(id.value()),
                        .status = std::move(childStatus.value()),
                        .exitCode = childExit.value()};
    if (result.value().has_value()) {
      child.result = pathFromUtf8(*result.value());
    }
    envelope.children.push_back(std::move(child));
  }

  auto payloadObject = requireObject(root, "payload", "$");
  if (payloadObject.hasError()) {
    return envelopeError<ResultEnvelopeV2>(payloadObject.error());
  }
  auto payload = jsonObjectText(payloadObject.value(), "$.payload");
  if (payload.hasError()) {
    return envelopeError<ResultEnvelopeV2>(payload.error());
  }
  envelope.payloadJson = std::move(payload.value());
  yyjson_val *extensionsObject = yyjson_obj_get(root, "extensions");
  if (extensionsObject != nullptr) {
    auto extensions = jsonObjectText(extensionsObject, "$.extensions");
    if (extensions.hasError()) {
      return envelopeError<ResultEnvelopeV2>(extensions.error());
    }
    envelope.extensionsJson = std::move(extensions.value());
  }

  auto valid = validateResultEnvelopeV2(envelope);
  if (valid.hasError()) {
    return envelopeError<ResultEnvelopeV2>(valid.error());
  }
  return Result<ResultEnvelopeV2, std::string>::makeResult(std::move(envelope));
}

Result<void, std::string>
writeResultEnvelopeV2(const std::filesystem::path &path,
                      const ResultEnvelopeV2 &envelope) {
  auto json = serializeResultEnvelopeV2(envelope);
  if (json.hasError()) {
    return envelopeError<void>(json.error());
  }
  return atomicWriteTextFile(path, json.value());
}

} // namespace nuri::tools::core
