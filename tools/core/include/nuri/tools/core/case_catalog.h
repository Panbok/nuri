#pragma once

#include "nuri/core/result.h"

#include <cstddef>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace nuri::tools::core {

struct CaseCatalogEntry {
  std::string id{};
  std::string suite{};
  std::vector<std::string> tags{};
  std::filesystem::path manifestPath{};
};

struct CaseCatalogSelector {
  std::string id{};
  std::string suite{};
  std::vector<std::string> tags{};
};

enum class CaseCatalogZeroMatchPolicy { Allow, Reject };

// Recursively discovers lowercase .json manifests in UTF-8 lexical path order.
// A missing root is an empty catalog; enumeration failures are errors.
[[nodiscard]] Result<std::vector<std::filesystem::path>, std::string>
discoverCaseManifestPaths(const std::filesystem::path &root);

[[nodiscard]] Result<void, std::string>
validateCaseCatalog(std::span<const CaseCatalogEntry> entries,
                    std::string_view catalogName);

// Selectors are exact and conjunctive. Every requested tag must be present.
// Results are ordered by case id, independent of discovery/input order.
[[nodiscard]] Result<std::vector<size_t>, std::string>
selectCaseCatalog(std::span<const CaseCatalogEntry> entries,
                  const CaseCatalogSelector &selector,
                  CaseCatalogZeroMatchPolicy zeroMatchPolicy,
                  std::string_view catalogName);

} // namespace nuri::tools::core
