#include "nuri/pch.h"

#include "nuri/utils/env_utils.h"

namespace nuri {

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

} // namespace nuri
