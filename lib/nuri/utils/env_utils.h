#pragma once

#include "nuri/defines.h"

#include <optional>
#include <string>
#include <string_view>

namespace nuri {

[[nodiscard]] NURI_API std::optional<std::string>
readEnvVar(std::string_view variableName);

// readEnvFlag returns true only for "1", "true", "on", or "yes"
// (case-insensitive). It returns false for unset/empty variables, explicit
// false values "0", "false", "off", or "no", and unrecognized values.
[[nodiscard]] NURI_API bool readEnvFlag(std::string_view variableName);

[[nodiscard]] NURI_API std::optional<bool>
readEnvBoolOverride(std::string_view variableName);

[[nodiscard]] NURI_API const char *boolToString(bool value);

} // namespace nuri
