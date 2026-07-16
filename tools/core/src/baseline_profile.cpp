#include "nuri/tools/core/baseline_profile.h"

#include "nuri/tools/core/identifier.h"
#include "nuri/tools/core/safe_path.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <string_view>
#include <unordered_set>
#include <utility>

#include <yyjson.h>

namespace nuri::tools::core {
namespace {

using JsonDocument = std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)>;

template <typename T>
[[nodiscard]] Result<T, std::string> profileError(std::string message) {
  return Result<T, std::string>::makeError(std::move(message));
}

[[nodiscard]] Result<void, std::string>
rejectUnknownFields(yyjson_val *object,
                    std::initializer_list<std::string_view> allowed,
                    std::string_view path) {
  if (!yyjson_is_obj(object)) {
    return Result<void, std::string>::makeError(std::string(path) +
                                                " must be an object");
  }
  yyjson_obj_iter iterator{};
  yyjson_obj_iter_init(object, &iterator);
  std::unordered_set<std::string> seen;
  yyjson_val *key = nullptr;
  while ((key = yyjson_obj_iter_next(&iterator)) != nullptr) {
    const std::string_view name(yyjson_get_str(key), yyjson_get_len(key));
    if (!seen.emplace(name).second) {
      return Result<void, std::string>::makeError(
          std::string(path) + " contains duplicate field '" +
          std::string(name) + "'");
    }
    if (std::find(allowed.begin(), allowed.end(), name) == allowed.end()) {
      return Result<void, std::string>::makeError(std::string(path) +
                                                  " contains unknown field '" +
                                                  std::string(name) + "'");
    }
  }
  return Result<void, std::string>::makeResult();
}

[[nodiscard]] Result<yyjson_val *, std::string>
requireObject(yyjson_val *object, std::string_view key, std::string_view path) {
  yyjson_val *value = yyjson_obj_getn(object, key.data(), key.size());
  if (!yyjson_is_obj(value)) {
    return profileError<yyjson_val *>(std::string(path) + "." +
                                      std::string(key) + " must be an object");
  }
  return Result<yyjson_val *, std::string>::makeResult(value);
}

[[nodiscard]] Result<std::string, std::string>
requireString(yyjson_val *object, std::string_view key, std::string_view path,
              bool allowEmpty = false) {
  yyjson_val *value = yyjson_obj_getn(object, key.data(), key.size());
  if (!yyjson_is_str(value)) {
    return profileError<std::string>(std::string(path) + "." +
                                     std::string(key) + " must be a string");
  }
  std::string text(yyjson_get_str(value), yyjson_get_len(value));
  if (!allowEmpty && text.empty()) {
    return profileError<std::string>(std::string(path) + "." +
                                     std::string(key) + " must not be empty");
  }
  return Result<std::string, std::string>::makeResult(std::move(text));
}

[[nodiscard]] Result<std::string, std::string>
optionalString(yyjson_val *object, std::string_view key,
               std::string_view path) {
  yyjson_val *value = yyjson_obj_getn(object, key.data(), key.size());
  if (value == nullptr) {
    return Result<std::string, std::string>::makeResult(std::string{});
  }
  if (!yyjson_is_str(value)) {
    return profileError<std::string>(std::string(path) + "." +
                                     std::string(key) + " must be a string");
  }
  return Result<std::string, std::string>::makeResult(
      std::string(yyjson_get_str(value), yyjson_get_len(value)));
}

[[nodiscard]] Result<bool, std::string>
requireBool(yyjson_val *object, std::string_view key, std::string_view path) {
  yyjson_val *value = yyjson_obj_getn(object, key.data(), key.size());
  if (!yyjson_is_bool(value)) {
    return profileError<bool>(std::string(path) + "." + std::string(key) +
                              " must be a boolean");
  }
  return Result<bool, std::string>::makeResult(yyjson_get_bool(value));
}

[[nodiscard]] Result<uint32_t, std::string> requireU32(yyjson_val *object,
                                                       std::string_view key,
                                                       std::string_view path,
                                                       bool positive = false) {
  yyjson_val *value = yyjson_obj_getn(object, key.data(), key.size());
  if (!yyjson_is_uint(value) ||
      yyjson_get_uint(value) > std::numeric_limits<uint32_t>::max() ||
      (positive && yyjson_get_uint(value) == 0u)) {
    return profileError<uint32_t>(
        std::string(path) + "." + std::string(key) +
        (positive ? " must be a positive uint32" : " must be a uint32"));
  }
  return Result<uint32_t, std::string>::makeResult(
      static_cast<uint32_t>(yyjson_get_uint(value)));
}

[[nodiscard]] Result<double, std::string>
requirePositiveReal(yyjson_val *object, std::string_view key,
                    std::string_view path) {
  yyjson_val *value = yyjson_obj_getn(object, key.data(), key.size());
  if (!yyjson_is_num(value)) {
    return profileError<double>(std::string(path) + "." + std::string(key) +
                                " must be a positive finite number");
  }
  const double number = yyjson_get_real(value);
  if (!std::isfinite(number) || number <= 0.0) {
    return profileError<double>(std::string(path) + "." + std::string(key) +
                                " must be a positive finite number");
  }
  return Result<double, std::string>::makeResult(number);
}

[[nodiscard]] Result<std::optional<uint32_t>, std::string>
optionalNullableU32(yyjson_val *object, std::string_view key,
                    std::string_view path) {
  yyjson_val *value = yyjson_obj_getn(object, key.data(), key.size());
  if (value == nullptr || yyjson_is_null(value)) {
    return Result<std::optional<uint32_t>, std::string>::makeResult(
        std::optional<uint32_t>{});
  }
  if (!yyjson_is_uint(value) ||
      yyjson_get_uint(value) > std::numeric_limits<uint32_t>::max()) {
    return profileError<std::optional<uint32_t>>(std::string(path) + "." +
                                                 std::string(key) +
                                                 " must be a uint32 or null");
  }
  return Result<std::optional<uint32_t>, std::string>::makeResult(
      std::optional<uint32_t>{static_cast<uint32_t>(yyjson_get_uint(value))});
}

[[nodiscard]] Result<std::optional<std::string>, std::string>
optionalNullableString(yyjson_val *object, std::string_view key,
                       std::string_view path) {
  yyjson_val *value = yyjson_obj_getn(object, key.data(), key.size());
  if (value == nullptr || yyjson_is_null(value)) {
    return Result<std::optional<std::string>, std::string>::makeResult(
        std::optional<std::string>{});
  }
  if (!yyjson_is_str(value)) {
    return profileError<std::optional<std::string>>(
        std::string(path) + "." + std::string(key) +
        " must be a string or null");
  }
  return Result<std::optional<std::string>, std::string>::makeResult(
      std::optional<std::string>{
          std::string(yyjson_get_str(value), yyjson_get_len(value))});
}

[[nodiscard]] bool isOneOf(std::string_view value,
                           std::initializer_list<std::string_view> choices) {
  return std::find(choices.begin(), choices.end(), value) != choices.end();
}

[[nodiscard]] Result<std::vector<std::string>, std::string>
requireUniqueStringArray(yyjson_val *object, std::string_view key,
                         std::string_view path) {
  yyjson_val *value = yyjson_obj_getn(object, key.data(), key.size());
  if (!yyjson_is_arr(value)) {
    return profileError<std::vector<std::string>>(
        std::string(path) + "." + std::string(key) + " must be an array");
  }
  std::vector<std::string> strings;
  strings.reserve(yyjson_arr_size(value));
  std::unordered_set<std::string> seen;
  yyjson_arr_iter iterator{};
  yyjson_arr_iter_init(value, &iterator);
  yyjson_val *entry = nullptr;
  while ((entry = yyjson_arr_iter_next(&iterator)) != nullptr) {
    if (!yyjson_is_str(entry) || yyjson_get_len(entry) == 0u) {
      return profileError<std::vector<std::string>>(
          std::string(path) + "." + std::string(key) +
          " entries must be non-empty strings");
    }
    std::string text(yyjson_get_str(entry), yyjson_get_len(entry));
    if (!seen.insert(text).second) {
      return profileError<std::vector<std::string>>(std::string(path) + "." +
                                                    std::string(key) +
                                                    " entries must be unique");
    }
    strings.push_back(std::move(text));
  }
  return Result<std::vector<std::string>, std::string>::makeResult(
      std::move(strings));
}

[[nodiscard]] Result<BaselineProfile, std::string>
parseBaselineProfile(std::string_view json, std::string_view expectedId,
                     const std::filesystem::path &sourcePath) {
  std::string mutableJson(json);
  yyjson_read_err parseError{};
  JsonDocument document(yyjson_read_opts(mutableJson.data(), mutableJson.size(),
                                         0, nullptr, &parseError),
                        &yyjson_doc_free);
  if (!document) {
    return profileError<BaselineProfile>("failed to parse baseline profile: " +
                                         std::string(parseError.msg != nullptr
                                                         ? parseError.msg
                                                         : "invalid JSON"));
  }
  yyjson_val *root = yyjson_doc_get_root(document.get());
  auto rootFields = rejectUnknownFields(
      root,
      {"schemaVersion", "kind", "id", "description", "authority", "environment",
       "execution", "benchmarkPolicy"},
      "$");
  if (rootFields.hasError()) {
    return profileError<BaselineProfile>(rootFields.error());
  }

  BaselineProfile profile{};
  auto schemaVersion = requireU32(root, "schemaVersion", "$");
  if (schemaVersion.hasError() || schemaVersion.value() != 1u) {
    return profileError<BaselineProfile>(schemaVersion.hasError()
                                             ? schemaVersion.error()
                                             : "$.schemaVersion must equal 1");
  }
  profile.schemaVersion = schemaVersion.value();
  auto kind = requireString(root, "kind", "$");
  if (kind.hasError() || kind.value() != "nuri.baseline.profile") {
    return profileError<BaselineProfile>(
        kind.hasError() ? kind.error()
                        : "$.kind must equal nuri.baseline.profile");
  }
  profile.kind = std::move(kind.value());
  auto id = requireString(root, "id", "$");
  if (id.hasError()) {
    return profileError<BaselineProfile>(id.error());
  }
  auto validId = validateIdentifier(id.value(), "$.id");
  if (validId.hasError()) {
    return profileError<BaselineProfile>(validId.error());
  }
  if (id.value() != expectedId) {
    return profileError<BaselineProfile>(
        "$.id does not match requested profile '" + std::string(expectedId) +
        "'");
  }
  profile.id = std::move(id.value());
  auto description = optionalString(root, "description", "$");
  if (description.hasError()) {
    return profileError<BaselineProfile>(description.error());
  }
  profile.description = std::move(description.value());

  auto authorityObject = requireObject(root, "authority", "$");
  if (authorityObject.hasError()) {
    return profileError<BaselineProfile>(authorityObject.error());
  }
  auto authorityFields = rejectUnknownFields(
      authorityObject.value(), {"authoritative", "allowDirtyTree", "reason"},
      "$.authority");
  if (authorityFields.hasError()) {
    return profileError<BaselineProfile>(authorityFields.error());
  }
  auto authoritative =
      requireBool(authorityObject.value(), "authoritative", "$.authority");
  auto allowDirtyTree =
      requireBool(authorityObject.value(), "allowDirtyTree", "$.authority");
  auto reason =
      optionalString(authorityObject.value(), "reason", "$.authority");
  if (authoritative.hasError() || allowDirtyTree.hasError() ||
      reason.hasError()) {
    return profileError<BaselineProfile>(authoritative.hasError()
                                             ? authoritative.error()
                                             : (allowDirtyTree.hasError()
                                                    ? allowDirtyTree.error()
                                                    : reason.error()));
  }
  profile.authority.authoritative = authoritative.value();
  profile.authority.allowDirtyTree = allowDirtyTree.value();
  profile.authority.reason = std::move(reason.value());

  auto environmentObject = requireObject(root, "environment", "$");
  if (environmentObject.hasError()) {
    return profileError<BaselineProfile>(environmentObject.error());
  }
  auto environmentFields = rejectUnknownFields(
      environmentObject.value(),
      {"os", "backend", "windowMode", "gpuVendorId", "gpuDeviceId", "driver"},
      "$.environment");
  if (environmentFields.hasError()) {
    return profileError<BaselineProfile>(environmentFields.error());
  }
  auto os = requireString(environmentObject.value(), "os", "$.environment");
  auto backend =
      requireString(environmentObject.value(), "backend", "$.environment");
  auto windowMode =
      requireString(environmentObject.value(), "windowMode", "$.environment");
  auto gpuVendorId = optionalNullableU32(environmentObject.value(),
                                         "gpuVendorId", "$.environment");
  auto gpuDeviceId = optionalNullableU32(environmentObject.value(),
                                         "gpuDeviceId", "$.environment");
  auto driver = optionalNullableString(environmentObject.value(), "driver",
                                       "$.environment");
  if (os.hasError() || backend.hasError() || windowMode.hasError() ||
      gpuVendorId.hasError() || gpuDeviceId.hasError() || driver.hasError()) {
    return profileError<BaselineProfile>(
        os.hasError()            ? os.error()
        : backend.hasError()     ? backend.error()
        : windowMode.hasError()  ? windowMode.error()
        : gpuVendorId.hasError() ? gpuVendorId.error()
        : gpuDeviceId.hasError() ? gpuDeviceId.error()
                                 : driver.error());
  }
  if (!isOneOf(windowMode.value(),
               {"visible", "hidden", "headless", "offscreen"})) {
    return profileError<BaselineProfile>(
        "$.environment.windowMode has an unsupported value");
  }
  profile.environment.os = std::move(os.value());
  profile.environment.backend = std::move(backend.value());
  profile.environment.windowMode = std::move(windowMode.value());
  profile.environment.gpuVendorId = gpuVendorId.value();
  profile.environment.gpuDeviceId = gpuDeviceId.value();
  profile.environment.driver = std::move(driver.value());

  auto executionObject = requireObject(root, "execution", "$");
  if (executionObject.hasError()) {
    return profileError<BaselineProfile>(executionObject.error());
  }
  auto executionFields = rejectUnknownFields(
      executionObject.value(), {"presentMode", "profiling", "devChecks"},
      "$.execution");
  if (executionFields.hasError()) {
    return profileError<BaselineProfile>(executionFields.error());
  }
  auto presentMode =
      requireString(executionObject.value(), "presentMode", "$.execution");
  auto profiling =
      requireString(executionObject.value(), "profiling", "$.execution");
  auto devChecks =
      requireBool(executionObject.value(), "devChecks", "$.execution");
  if (presentMode.hasError() || profiling.hasError() || devChecks.hasError()) {
    return profileError<BaselineProfile>(
        presentMode.hasError()
            ? presentMode.error()
            : (profiling.hasError() ? profiling.error() : devChecks.error()));
  }
  if (!isOneOf(presentMode.value(),
               {"default", "immediate", "mailbox", "fifo"})) {
    return profileError<BaselineProfile>(
        "$.execution.presentMode has an unsupported value");
  }
  if (!isOneOf(profiling.value(), {"off", "cpu"})) {
    return profileError<BaselineProfile>(
        "$.execution.profiling has an unsupported value");
  }
  profile.execution.presentMode = std::move(presentMode.value());
  profile.execution.profiling = std::move(profiling.value());
  profile.execution.devChecks = devChecks.value();

  auto policyObject = requireObject(root, "benchmarkPolicy", "$");
  if (policyObject.hasError()) {
    return profileError<BaselineProfile>(policyObject.error());
  }
  auto policyFields = rejectUnknownFields(
      policyObject.value(),
      {"minimumRepetitions", "warmupStability", "warmupWindowFrames",
       "warmupMaxDriftPercent", "requiredMetrics", "thresholdOwnership"},
      "$.benchmarkPolicy");
  if (policyFields.hasError()) {
    return profileError<BaselineProfile>(policyFields.error());
  }
  auto minimumRepetitions = requireU32(
      policyObject.value(), "minimumRepetitions", "$.benchmarkPolicy", true);
  auto warmupStability = requireString(policyObject.value(), "warmupStability",
                                       "$.benchmarkPolicy");
  auto warmupWindowFrames = requireU32(
      policyObject.value(), "warmupWindowFrames", "$.benchmarkPolicy", true);
  auto warmupMaxDriftPercent = requirePositiveReal(
      policyObject.value(), "warmupMaxDriftPercent", "$.benchmarkPolicy");
  auto requiredMetrics = requireUniqueStringArray(
      policyObject.value(), "requiredMetrics", "$.benchmarkPolicy");
  auto thresholdOwnership = requireString(
      policyObject.value(), "thresholdOwnership", "$.benchmarkPolicy");
  if (minimumRepetitions.hasError() || warmupStability.hasError() ||
      warmupWindowFrames.hasError() || warmupMaxDriftPercent.hasError() ||
      requiredMetrics.hasError() || thresholdOwnership.hasError()) {
    return profileError<BaselineProfile>(
        minimumRepetitions.hasError()      ? minimumRepetitions.error()
        : warmupStability.hasError()       ? warmupStability.error()
        : warmupWindowFrames.hasError()    ? warmupWindowFrames.error()
        : warmupMaxDriftPercent.hasError() ? warmupMaxDriftPercent.error()
        : requiredMetrics.hasError()       ? requiredMetrics.error()
                                           : thresholdOwnership.error());
  }
  if (!isOneOf(warmupStability.value(),
               {"required", "investigative", "unknown"})) {
    return profileError<BaselineProfile>(
        "$.benchmarkPolicy.warmupStability has an unsupported value");
  }
  if (thresholdOwnership.value() != "baseline") {
    return profileError<BaselineProfile>(
        "$.benchmarkPolicy.thresholdOwnership must equal baseline");
  }
  for (const std::string &metric : requiredMetrics.value()) {
    auto validMetric = validateIdentifier(
        metric, "$.benchmarkPolicy.requiredMetrics[]", IdentifierShape::Dotted);
    if (validMetric.hasError()) {
      return profileError<BaselineProfile>(validMetric.error());
    }
  }
  profile.benchmarkPolicy.minimumRepetitions = minimumRepetitions.value();
  profile.benchmarkPolicy.warmupStability = std::move(warmupStability.value());
  profile.benchmarkPolicy.warmupWindowFrames = warmupWindowFrames.value();
  profile.benchmarkPolicy.warmupMaxDriftPercent = warmupMaxDriftPercent.value();
  profile.benchmarkPolicy.requiredMetrics = std::move(requiredMetrics.value());
  profile.benchmarkPolicy.thresholdOwnership =
      std::move(thresholdOwnership.value());
  if (profile.authority.authoritative) {
    if (profile.authority.allowDirtyTree) {
      return profileError<BaselineProfile>(
          "authoritative baseline profiles cannot allow a dirty tree");
    }
    if (profile.environment.os == "any" ||
        !profile.environment.gpuVendorId.has_value() ||
        !profile.environment.gpuDeviceId.has_value() ||
        !profile.environment.driver.has_value()) {
      return profileError<BaselineProfile>(
          "authoritative baseline profiles require pinned OS, GPU vendor, "
          "GPU device, and driver selectors");
    }
    if (profile.benchmarkPolicy.minimumRepetitions < 3u ||
        profile.benchmarkPolicy.warmupStability != "required") {
      return profileError<BaselineProfile>(
          "authoritative baseline profiles require at least three "
          "independent repetitions and required warmup stability");
    }
  }
  profile.sourcePath = sourcePath;
  return Result<BaselineProfile, std::string>::makeResult(std::move(profile));
}

} // namespace

Result<BaselineProfile, std::string>
loadBaselineProfile(const std::filesystem::path &profilesDirectory,
                    std::string_view profileId) {
  auto validId = validateIdentifier(profileId, "profile id");
  if (validId.hasError()) {
    return profileError<BaselineProfile>(validId.error());
  }
  auto profilePath =
      resolvePathUnder(profilesDirectory,
                       std::filesystem::path(std::string(profileId) + ".json"));
  if (profilePath.hasError()) {
    return profileError<BaselineProfile>(profilePath.error());
  }

  std::ifstream file(profilePath.value(), std::ios::binary);
  if (!file) {
    return profileError<BaselineProfile>(
        "baseline profile does not exist or cannot be read: " +
        profilePath.value().string());
  }
  const std::string json((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
  if (!file.good() && !file.eof()) {
    return profileError<BaselineProfile>("failed to read baseline profile: " +
                                         profilePath.value().string());
  }
  return parseBaselineProfile(json, profileId, profilePath.value());
}

BaselineProfileCompatibility
evaluateBaselineProfile(const BaselineProfile &profile,
                        const BaselineProfileObservedEnvironment &observed) {
  BaselineProfileCompatibility out{};
  const auto asciiEqual = [](std::string_view left, std::string_view right) {
    if (left.size() != right.size()) {
      return false;
    }
    for (size_t index = 0u; index < left.size(); ++index) {
      const auto lower = [](char value) {
        return value >= 'A' && value <= 'Z'
                   ? static_cast<char>(value - 'A' + 'a')
                   : value;
      };
      if (lower(left[index]) != lower(right[index])) {
        return false;
      }
    }
    return true;
  };
  const auto mismatch = [&out](bool condition, std::string reason) {
    if (condition) {
      out.reasons.push_back(std::move(reason));
    }
  };

  mismatch(profile.environment.os != "any" &&
               !asciiEqual(profile.environment.os, observed.os),
           "operating system does not match profile");
  const bool defaultBackend =
      asciiEqual(observed.backendSource, "default") &&
      asciiEqual(profile.environment.backend, observed.backend + "-default");
  mismatch(!asciiEqual(profile.environment.backend, observed.backend) &&
               !defaultBackend,
           "GPU backend does not match profile");
  const bool visibleMode =
      profile.environment.windowMode == "visible" && observed.windowVisible &&
      (observed.windowMode == "visible" || observed.windowMode == "windowed");
  mismatch(!visibleMode &&
               !asciiEqual(profile.environment.windowMode, observed.windowMode),
           "window mode does not match profile");
  mismatch(profile.environment.gpuVendorId.has_value() &&
               *profile.environment.gpuVendorId != observed.gpuVendorId,
           "GPU vendor does not match profile");
  mismatch(profile.environment.gpuDeviceId.has_value() &&
               *profile.environment.gpuDeviceId != observed.gpuDeviceId,
           "GPU device does not match profile");
  mismatch(profile.environment.driver.has_value() &&
               *profile.environment.driver != observed.driver,
           "GPU driver does not match profile");
  mismatch(profile.execution.presentMode != "default" &&
               !asciiEqual(profile.execution.presentMode, observed.presentMode),
           "present mode does not match profile");
  mismatch(!asciiEqual(profile.execution.profiling, observed.profiling),
           "profiling mode does not match profile");
  mismatch(profile.execution.devChecks != observed.devChecks,
           "dev-check mode does not match profile");
  mismatch(observed.dirtyTree && !profile.authority.allowDirtyTree,
           "dirty source tree is not allowed by profile");
  if (profile.authority.authoritative) {
    mismatch(observed.gpuVendorId == 0u || observed.gpuDeviceId == 0u ||
                 observed.driver.empty() || observed.driver == "unknown",
             "authoritative profile requires known GPU and driver identity");
  }
  out.compatible = out.reasons.empty();
  return out;
}

} // namespace nuri::tools::core
