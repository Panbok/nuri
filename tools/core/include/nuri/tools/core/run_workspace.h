#pragma once

#include "nuri/core/result.h"

#include <filesystem>
#include <string>
#include <string_view>

namespace nuri::tools::core {

struct RunWorkspace {
  std::string runId{};
  std::filesystem::path root{};

  [[nodiscard]] Result<std::filesystem::path, std::string>
  caseDirectory(std::string_view caseId) const;
};

[[nodiscard]] std::string createRunId();
[[nodiscard]] Result<RunWorkspace, std::string>
createRunWorkspace(const std::filesystem::path &toolArtifactRoot);

} // namespace nuri::tools::core
