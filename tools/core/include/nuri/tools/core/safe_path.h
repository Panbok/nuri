#pragma once

#include "nuri/core/result.h"

#include <filesystem>
#include <string>

namespace nuri::tools::core {

[[nodiscard]] Result<std::filesystem::path, std::string>
resolvePathUnder(const std::filesystem::path &root,
                 const std::filesystem::path &relative);

[[nodiscard]] Result<uintmax_t, std::string>
removeTreeUnder(const std::filesystem::path &root,
                const std::filesystem::path &relative);

} // namespace nuri::tools::core
