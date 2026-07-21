#include "nuri/tools/autotest/autotest_case.h"

#include "nuri/tools/snapshot/snapshot_case.h"

namespace nuri::tools::autotest {

std::string autotestExitCodeName(AutotestExitCode code) {
  switch (code) {
  case AutotestExitCode::Success:
    return "success";
  case AutotestExitCode::ScenarioFailure:
    return "scenario_failure";
  case AutotestExitCode::InvalidInput:
    return "invalid_input";
  case AutotestExitCode::EnvironmentUnavailable:
    return "environment_unavailable";
  case AutotestExitCode::RuntimeError:
    return "runtime_error";
  case AutotestExitCode::MissingBaseline:
    return "missing_baseline";
  }
  return "unknown";
}

void sanitizeAutotestRenderSettings(RenderSettings &settings) {
  nuri::tools::snapshot::sanitizeSnapshotRenderSettings(settings);
  sanitizeDDGISettings(settings.ddgi);
}

} // namespace nuri::tools::autotest
