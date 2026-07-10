#pragma once

#include <cstdint>
#include <string_view>

namespace nuri::tools::core {

enum class ToolOutcome : uint8_t {
  Pass,
  Warn,
  Failure,
  Invalid,
  EnvironmentUnavailable,
  MissingBaseline,
  RuntimeError,
  Cancelled,
  Incomplete,
  Investigative,
};

struct SelectionCounts {
  uint32_t selected = 0u;
  uint32_t attempted = 0u;
  uint32_t completed = 0u;
  uint32_t passed = 0u;
  uint32_t warned = 0u;
  uint32_t failed = 0u;
  uint32_t skipped = 0u;
  uint32_t unavailable = 0u;
  uint32_t notRun = 0u;
};

[[nodiscard]] std::string_view toolOutcomeName(ToolOutcome outcome) noexcept;
[[nodiscard]] int toolOutcomeExitCode(ToolOutcome outcome) noexcept;
[[nodiscard]] ToolOutcome aggregateOutcome(ToolOutcome current,
                                           ToolOutcome candidate) noexcept;
[[nodiscard]] bool selectionIsComplete(const SelectionCounts &counts) noexcept;

} // namespace nuri::tools::core
