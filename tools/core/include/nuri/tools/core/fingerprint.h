#pragma once

#include "nuri/core/result.h"

#include <filesystem>
#include <string>
#include <vector>

namespace nuri::tools::core {

struct FingerprintField {
  std::string name{};
  std::string value{};
};

// Field order does not affect the digest. Names must be unique and non-empty;
// length-prefixed encoding prevents delimiter ambiguity in arbitrary values.
[[nodiscard]] Result<std::string, std::string>
makeSha256Fingerprint(std::vector<FingerprintField> fields);

[[nodiscard]] Result<std::string, std::string>
makeSha256FileFingerprint(const std::filesystem::path &path);

} // namespace nuri::tools::core
