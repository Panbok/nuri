#include "nuri/tools/core/case_catalog.h"

#include "nuri/tools/core/identifier.h"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <utility>

namespace nuri::tools::core {
namespace {

[[nodiscard]] std::string pathForDiagnostic(const std::filesystem::path &path) {
  if (path.empty()) {
    return "<unknown>";
  }
  const std::u8string utf8 = path.generic_u8string();
  return std::string(utf8.begin(), utf8.end());
}

[[nodiscard]] std::vector<size_t>
sortedCatalogIndices(std::span<const CaseCatalogEntry> entries) {
  std::vector<size_t> indices(entries.size());
  std::iota(indices.begin(), indices.end(), 0u);
  std::sort(indices.begin(), indices.end(), [&](size_t lhs, size_t rhs) {
    if (entries[lhs].id != entries[rhs].id) {
      return entries[lhs].id < entries[rhs].id;
    }
    return entries[lhs].manifestPath.generic_u8string() <
           entries[rhs].manifestPath.generic_u8string();
  });
  return indices;
}

[[nodiscard]] bool hasAllTags(const CaseCatalogEntry &entry,
                              const CaseCatalogSelector &selector) {
  return std::all_of(selector.tags.begin(), selector.tags.end(),
                     [&](const std::string &tag) {
                       return std::find(entry.tags.begin(), entry.tags.end(),
                                        tag) != entry.tags.end();
                     });
}

[[nodiscard]] std::string zeroMatchMessage(const CaseCatalogSelector &selector,
                                           std::string_view catalogName) {
  std::ostringstream message;
  message << "selector matched zero " << catalogName << " cases";
  if (!selector.id.empty()) {
    message << " for id '" << selector.id << "'";
  } else if (!selector.suite.empty()) {
    message << " for suite '" << selector.suite << "'";
  } else if (!selector.tags.empty()) {
    message << " for requested tags";
  }
  return message.str();
}

} // namespace

Result<std::vector<std::filesystem::path>, std::string>
discoverCaseManifestPaths(const std::filesystem::path &root) {
  std::vector<std::filesystem::path> paths;
  std::error_code error;
  const bool exists = std::filesystem::exists(root, error);
  if (error) {
    return Result<std::vector<std::filesystem::path>, std::string>::makeError(
        "failed to inspect case catalog root '" + pathForDiagnostic(root) +
        "': " + error.message());
  }
  if (!exists) {
    return Result<std::vector<std::filesystem::path>, std::string>::makeResult(
        std::move(paths));
  }
  if (!std::filesystem::is_directory(root, error) || error) {
    const std::string detail = error ? ": " + error.message() : std::string{};
    return Result<std::vector<std::filesystem::path>, std::string>::makeError(
        "case catalog root is not a directory '" + pathForDiagnostic(root) +
        "'" + detail);
  }

  std::filesystem::recursive_directory_iterator iterator(root, error);
  const std::filesystem::recursive_directory_iterator end;
  if (error) {
    return Result<std::vector<std::filesystem::path>, std::string>::makeError(
        "failed to enumerate case catalog root '" + pathForDiagnostic(root) +
        "': " + error.message());
  }
  while (iterator != end) {
    const std::filesystem::directory_entry &entry = *iterator;
    const bool regular = entry.is_regular_file(error);
    if (error) {
      return Result<std::vector<std::filesystem::path>, std::string>::makeError(
          "failed to inspect case catalog entry '" +
          pathForDiagnostic(entry.path()) + "': " + error.message());
    }
    if (regular && entry.path().extension() == ".json") {
      paths.push_back(entry.path());
    }
    iterator.increment(error);
    if (error) {
      return Result<std::vector<std::filesystem::path>, std::string>::makeError(
          "failed to enumerate case catalog root '" + pathForDiagnostic(root) +
          "': " + error.message());
    }
  }

  std::sort(
      paths.begin(), paths.end(),
      [](const std::filesystem::path &lhs, const std::filesystem::path &rhs) {
        return lhs.generic_u8string() < rhs.generic_u8string();
      });
  return Result<std::vector<std::filesystem::path>, std::string>::makeResult(
      std::move(paths));
}

Result<void, std::string>
validateCaseCatalog(std::span<const CaseCatalogEntry> entries,
                    std::string_view catalogName) {
  for (const CaseCatalogEntry &entry : entries) {
    auto id = validateIdentifier(entry.id, "case id", IdentifierShape::Dotted);
    if (id.hasError()) {
      return Result<void, std::string>::makeError(std::string(catalogName) +
                                                  " catalog " + id.error());
    }
    auto suite =
        validateIdentifier(entry.suite, "case suite", IdentifierShape::Dotted);
    if (suite.hasError()) {
      return Result<void, std::string>::makeError(std::string(catalogName) +
                                                  " catalog " + suite.error());
    }
    for (const std::string &tag : entry.tags) {
      auto validTag = validateIdentifier(tag, "case tag");
      if (validTag.hasError()) {
        return Result<void, std::string>::makeError(
            std::string(catalogName) + " catalog " + validTag.error());
      }
    }
  }

  const std::vector<size_t> indices = sortedCatalogIndices(entries);
  for (size_t index = 1u; index < indices.size(); ++index) {
    const CaseCatalogEntry &previous = entries[indices[index - 1u]];
    const CaseCatalogEntry &current = entries[indices[index]];
    if (previous.id == current.id) {
      return Result<void, std::string>::makeError(
          "duplicate " + std::string(catalogName) + " case id '" + current.id +
          "' in '" + pathForDiagnostic(previous.manifestPath) + "' and '" +
          pathForDiagnostic(current.manifestPath) + "'");
    }
  }
  return Result<void, std::string>::makeResult();
}

Result<std::vector<size_t>, std::string>
selectCaseCatalog(std::span<const CaseCatalogEntry> entries,
                  const CaseCatalogSelector &selector,
                  CaseCatalogZeroMatchPolicy zeroMatchPolicy,
                  std::string_view catalogName) {
  auto validCatalog = validateCaseCatalog(entries, catalogName);
  if (validCatalog.hasError()) {
    return Result<std::vector<size_t>, std::string>::makeError(
        validCatalog.error());
  }
  if (!selector.id.empty()) {
    auto validId = validateIdentifier(selector.id, "case selector",
                                      IdentifierShape::Dotted);
    if (validId.hasError()) {
      return Result<std::vector<size_t>, std::string>::makeError(
          validId.error());
    }
  }
  if (!selector.suite.empty()) {
    auto validSuite = validateIdentifier(selector.suite, "suite selector",
                                         IdentifierShape::Dotted);
    if (validSuite.hasError()) {
      return Result<std::vector<size_t>, std::string>::makeError(
          validSuite.error());
    }
  }
  for (const std::string &tag : selector.tags) {
    auto validTag = validateIdentifier(tag, "tag selector");
    if (validTag.hasError()) {
      return Result<std::vector<size_t>, std::string>::makeError(
          validTag.error());
    }
  }

  std::vector<size_t> selected;
  for (const size_t index : sortedCatalogIndices(entries)) {
    const CaseCatalogEntry &entry = entries[index];
    if ((!selector.id.empty() && entry.id != selector.id) ||
        (!selector.suite.empty() && entry.suite != selector.suite) ||
        !hasAllTags(entry, selector)) {
      continue;
    }
    selected.push_back(index);
  }
  if (selected.empty() &&
      zeroMatchPolicy == CaseCatalogZeroMatchPolicy::Reject) {
    return Result<std::vector<size_t>, std::string>::makeError(
        zeroMatchMessage(selector, catalogName));
  }
  return Result<std::vector<size_t>, std::string>::makeResult(
      std::move(selected));
}

} // namespace nuri::tools::core
