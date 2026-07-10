#include "nuri/tools/core/identifier.h"

#include <cctype>

namespace nuri::tools::core {
namespace {

[[nodiscard]] bool isLowerAlphaNumeric(char value) noexcept {
  return (value >= 'a' && value <= 'z') || (value >= '0' && value <= '9');
}

[[nodiscard]] bool isSegmentCharacter(char value) noexcept {
  return isLowerAlphaNumeric(value) || value == '_' || value == '-';
}

[[nodiscard]] bool isReservedWindowsName(std::string_view segment) noexcept {
  if (segment == "con" || segment == "prn" || segment == "aux" ||
      segment == "nul") {
    return true;
  }
  if (segment.size() == 4u &&
      (segment.starts_with("com") || segment.starts_with("lpt"))) {
    return segment[3] >= '1' && segment[3] <= '9';
  }
  return false;
}

} // namespace

Result<void, std::string> validateIdentifier(std::string_view value,
                                             std::string_view field,
                                             IdentifierShape shape) {
  if (value.empty()) {
    return Result<void, std::string>::makeError(std::string(field) +
                                                " must not be empty");
  }
  if (value.size() > 255u) {
    return Result<void, std::string>::makeError(std::string(field) +
                                                " is too long");
  }

  size_t segmentLength = 0u;
  bool atSegmentStart = true;
  size_t segmentStart = 0u;
  size_t index = 0u;
  for (const char valueCharacter : value) {
    const unsigned char byte = static_cast<unsigned char>(valueCharacter);
    if (byte < 0x20u || byte == 0x7fu) {
      return Result<void, std::string>::makeError(std::string(field) +
                                                  " contains a control character");
    }
    if (valueCharacter == '.') {
      if (shape != IdentifierShape::Dotted || atSegmentStart) {
        return Result<void, std::string>::makeError(std::string(field) +
                                                    " contains an invalid dot");
      }
      if (isReservedWindowsName(value.substr(segmentStart, segmentLength))) {
        return Result<void, std::string>::makeError(std::string(field) +
                                                    " contains a reserved device name");
      }
      atSegmentStart = true;
      segmentLength = 0u;
      segmentStart = index + 1u;
      ++index;
      continue;
    }
    if (atSegmentStart && !isLowerAlphaNumeric(valueCharacter)) {
      return Result<void, std::string>::makeError(
          std::string(field) + " segments must start with a lowercase letter or digit");
    }
    if (!isSegmentCharacter(valueCharacter)) {
      return Result<void, std::string>::makeError(std::string(field) +
                                                  " contains an invalid character");
    }
    atSegmentStart = false;
    ++segmentLength;
    if (segmentLength > 64u) {
      return Result<void, std::string>::makeError(std::string(field) +
                                                  " contains a segment longer than 64 characters");
    }
    ++index;
  }
  if (atSegmentStart) {
    return Result<void, std::string>::makeError(std::string(field) +
                                                " must not end with a dot");
  }
  if (isReservedWindowsName(value.substr(segmentStart, segmentLength))) {
    return Result<void, std::string>::makeError(std::string(field) +
                                                " contains a reserved device name");
  }
  return Result<void, std::string>::makeResult();
}

} // namespace nuri::tools::core
