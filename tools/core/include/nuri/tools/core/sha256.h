#pragma once

#include "nuri/core/result.h"

#include <cstddef>
#include <filesystem>
#include <span>
#include <string>

namespace nuri::tools::core {

[[nodiscard]] std::string sha256Hex(std::span<const std::byte> data);
[[nodiscard]] Result<std::string, std::string>
sha256File(const std::filesystem::path &path);

} // namespace nuri::tools::core
