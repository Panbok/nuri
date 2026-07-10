#pragma once

#include "nuri/core/result.h"

#include <filesystem>
#include <string>
#include <string_view>

namespace nuri::tools::core {

[[nodiscard]] Result<void, std::string>
atomicWriteTextFile(const std::filesystem::path &path, std::string_view text);

} // namespace nuri::tools::core
