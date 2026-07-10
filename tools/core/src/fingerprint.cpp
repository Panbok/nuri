#include "nuri/tools/core/fingerprint.h"

#include "nuri/tools/core/sha256.h"

#include <algorithm>
#include <span>
#include <string_view>

namespace nuri::tools::core {

Result<std::string, std::string>
makeSha256Fingerprint(std::vector<FingerprintField> fields) {
  std::sort(fields.begin(), fields.end(),
            [](const FingerprintField &left, const FingerprintField &right) {
              return left.name < right.name;
            });
  std::string canonical;
  for (size_t index = 0u; index < fields.size(); ++index) {
    const FingerprintField &field = fields[index];
    if (field.name.empty()) {
      return Result<std::string, std::string>::makeError(
          "fingerprint field name must not be empty");
    }
    if (index > 0u && fields[index - 1u].name == field.name) {
      return Result<std::string, std::string>::makeError(
          "fingerprint field names must be unique: " + field.name);
    }
    canonical += std::to_string(field.name.size());
    canonical.push_back(':');
    canonical += field.name;
    canonical += std::to_string(field.value.size());
    canonical.push_back(':');
    canonical += field.value;
  }
  const std::span<const char> characters(canonical.data(), canonical.size());
  return Result<std::string, std::string>::makeResult(
      "sha256:" + sha256Hex(std::as_bytes(characters)));
}

Result<std::string, std::string>
makeSha256FileFingerprint(const std::filesystem::path &path) {
  auto digest = sha256File(path);
  if (digest.hasError()) {
    return Result<std::string, std::string>::makeError(digest.error());
  }
  return Result<std::string, std::string>::makeResult("sha256:" +
                                                      digest.value());
}

} // namespace nuri::tools::core
