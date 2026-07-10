#include "nuri/tools/core/result_protocol.h"

namespace nuri::tools::core {
namespace {

[[nodiscard]] int outcomePrecedence(ToolOutcome outcome) noexcept {
  switch (outcome) {
  case ToolOutcome::RuntimeError:
  case ToolOutcome::Cancelled:
    return 9;
  case ToolOutcome::Invalid:
    return 8;
  case ToolOutcome::Incomplete:
    return 7;
  case ToolOutcome::EnvironmentUnavailable:
    return 6;
  case ToolOutcome::MissingBaseline:
    return 5;
  case ToolOutcome::Failure:
    return 4;
  case ToolOutcome::Warn:
    return 3;
  case ToolOutcome::Investigative:
    return 2;
  case ToolOutcome::Pass:
    return 1;
  }
  return 9;
}

} // namespace

std::string_view toolOutcomeName(ToolOutcome outcome) noexcept {
  switch (outcome) {
  case ToolOutcome::Pass:
    return "pass";
  case ToolOutcome::Warn:
    return "warn";
  case ToolOutcome::Failure:
    return "fail";
  case ToolOutcome::Invalid:
    return "invalid";
  case ToolOutcome::EnvironmentUnavailable:
    return "unavailable";
  case ToolOutcome::MissingBaseline:
    return "missing_baseline";
  case ToolOutcome::RuntimeError:
    return "error";
  case ToolOutcome::Cancelled:
    return "cancelled";
  case ToolOutcome::Incomplete:
    return "incomplete";
  case ToolOutcome::Investigative:
    return "investigative";
  }
  return "error";
}

int toolOutcomeExitCode(ToolOutcome outcome) noexcept {
  switch (outcome) {
  case ToolOutcome::Pass:
  case ToolOutcome::Warn:
  case ToolOutcome::Investigative:
    return 0;
  case ToolOutcome::Failure:
    return 1;
  case ToolOutcome::Invalid:
    return 2;
  case ToolOutcome::EnvironmentUnavailable:
    return 3;
  case ToolOutcome::RuntimeError:
  case ToolOutcome::Cancelled:
  case ToolOutcome::Incomplete:
    return 4;
  case ToolOutcome::MissingBaseline:
    return 5;
  }
  return 4;
}

ToolOutcome aggregateOutcome(ToolOutcome current,
                             ToolOutcome candidate) noexcept {
  return outcomePrecedence(candidate) > outcomePrecedence(current) ? candidate
                                                                  : current;
}

bool selectionIsComplete(const SelectionCounts &counts) noexcept {
  return counts.selected > 0u && counts.attempted == counts.selected &&
         counts.completed == counts.attempted && counts.notRun == 0u &&
         counts.skipped == 0u && counts.unavailable == 0u;
}

} // namespace nuri::tools::core
