#pragma once

#include "nuri/core/result.h"
#include "nuri/tools/autotest/autotest_case.h"

#include <cstdint>
#include <string>
#include <vector>

namespace nuri::tools::autotest {

inline constexpr uint32_t kAutotestReadoutDrainFrameLimit = 4u;

struct AutotestFramePlan {
  uint32_t frame = 0u;
  AutotestCameraConfig camera{};
  RenderSettings settings{};
  std::array<uint32_t, 2> resolution{0u, 0u};
  bool resetTemporalHistory = false;
  bool resizeRequested = false;
  bool cameraCut = false;
  bool drainOnly = false;
  std::string resetReason{};
  std::vector<const AutotestTimelineEvent *> sceneEvents{};
  std::vector<const AutotestCheckpoint *> checkpoints{};
};

[[nodiscard]] Result<AutotestCameraConfig, std::string>
evaluateAutotestCameraAtFrame(const AutotestCase &testCase, uint32_t frame);
[[nodiscard]] Result<std::vector<AutotestFramePlan>, std::string>
compileAutotestTimeline(const AutotestCase &testCase);

} // namespace nuri::tools::autotest
