#include "nuri/pch.h"

#include "nuri/math/utils.h"

namespace nuri {

// Keep this TU non-empty after moving sanitization helpers to the header.
[[maybe_unused]] static auto kSanitizeFiniteVec3 = &sanitizeFiniteVec3;
[[maybe_unused]] static auto kSanitizeRotation = &sanitizeRotation;
[[maybe_unused]] static auto kSanitizeNonNegative = &sanitizeNonNegative;
} // namespace nuri