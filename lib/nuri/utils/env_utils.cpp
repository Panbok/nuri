#include "nuri/pch.h"

#include "nuri/utils/env_utils.h"

#include <cctype>

namespace nuri {
namespace {

[[nodiscard]] bool envValueEqualsIgnoreCase(std::string_view value,
                                            std::string_view expected) {
  if (value.size() != expected.size()) {
    return false;
  }
  for (size_t i = 0; i < value.size(); ++i) {
    const char lhs =
        static_cast<char>(std::tolower(static_cast<unsigned char>(value[i])));
    const char rhs = static_cast<char>(
        std::tolower(static_cast<unsigned char>(expected[i])));
    if (lhs != rhs) {
      return false;
    }
  }
  return true;
}

} // namespace

std::optional<std::string> readEnvVar(std::string_view variableName) {
#if defined(_WIN32)
  struct CFreeDeleter {
    void operator()(char *value) const noexcept { std::free(value); }
  };

  std::string key(variableName);
  char *rawValue = nullptr;
  size_t valueLength = 0;
  if (_dupenv_s(&rawValue, &valueLength, key.c_str()) != 0 ||
      rawValue == nullptr) {
    return std::nullopt;
  }

  std::unique_ptr<char, CFreeDeleter> value(rawValue);
  if (value.get()[0] == '\0') {
    return std::nullopt;
  }
  return std::string(value.get());
#else
  std::string key(variableName);
  const char *value = std::getenv(key.c_str());
  if (value == nullptr || value[0] == '\0') {
    return std::nullopt;
  }
  return std::string(value);
#endif
}

bool readEnvFlag(std::string_view variableName) {
  const std::optional<std::string> value = readEnvVar(variableName);
  if (!value.has_value()) {
    return false;
  }

  const std::string_view view = *value;
  if (envValueEqualsIgnoreCase(view, "1") ||
      envValueEqualsIgnoreCase(view, "true") ||
      envValueEqualsIgnoreCase(view, "on") ||
      envValueEqualsIgnoreCase(view, "yes")) {
    return true;
  }
  return false;
}

[[nodiscard]] const char *boolToString(bool value) {
  return value ? "true" : "false";
}

[[nodiscard]] std::optional<bool>
readEnvBoolOverride(std::string_view variableName) {
  const std::optional<std::string> value = readEnvVar(variableName);
  if (!value.has_value()) {
    return std::nullopt;
  }

  const std::string_view view = *value;
  if (view == "1" || view == "true" || view == "TRUE" || view == "on" ||
      view == "ON" || view == "yes" || view == "YES") {
    return true;
  }
  if (view == "0" || view == "false" || view == "FALSE" || view == "off" ||
      view == "OFF" || view == "no" || view == "NO") {
    return false;
  }

  NURI_LOG_WARNING(
      "LvkGPUDevice: ignoring unrecognized boolean environment override "
      "'%.*s=%s'",
      static_cast<int>(variableName.size()), variableName.data(),
      value->c_str());
  return std::nullopt;
}

} // namespace nuri
