#pragma once

#include "nuri/core/result.h"

#include <string>
#include <string_view>

namespace nuri::tools::core {

enum class IdentifierShape { Segment, Dotted };

[[nodiscard]] Result<void, std::string>
validateIdentifier(std::string_view value, std::string_view field,
                   IdentifierShape shape = IdentifierShape::Segment);

} // namespace nuri::tools::core
