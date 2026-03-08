#pragma once

#include "nuri/defines.h"

#include <optional>
#include <string>
#include <string_view>

namespace nuri {

[[nodiscard]] NURI_API std::optional<std::string>
readEnvVar(std::string_view variableName);

} // namespace nuri
