#pragma once

#include "nuri/core/result.h"
#include "nuri/tools/autotest/autotest_case.h"
#include "nuri/tools/snapshot/snapshot_image.h"

#include <cstdint>
#include <string>
#include <vector>

namespace nuri::tools::autotest {

struct AutotestQualityOracleReport {
  std::string status = "not_run";
  std::string statusReason = "not_run";
  std::string outputTarget{};
  std::string referencePath{};
  uint32_t schemaVersion = 1u;
  uint32_t referenceVersion = 0u;
  uint32_t maskVersion = 0u;
  double lscale = 0.0;
  uint32_t selectedPixelCount = 0u;
  uint32_t finitePixelCount = 0u;
  uint64_t nonFiniteValueCount = 0u;
  double normalizedHdrMae = 0.0;
  double normalizedHdrRmse = 0.0;
  double lumaSsim = 0.0;
  uint32_t darkCollapsePixelCount = 0u;
  double darkCollapsePercent = 0.0;
  uint32_t darkCollapseMaxComponentPixels = 0u;
  double relativeLumaEnergyDrift = 0.0;
  bool edgeAvailable = false;
  std::string edgeAxis{};
  uint32_t edgeProfileCount = 0u;
  uint32_t edgeUnresolvedProfileCount = 0u;
  double referenceEdgeWidth10To90 = 0.0;
  double outputEdgeWidth10To90 = 0.0;
  double edgeWidthRatio = 0.0;
  double edgeOvershoot = 0.0;
  double edgeUndershoot = 0.0;
  bool temporalAvailable = false;
  uint32_t temporalSampleCount = 0u;
  double temporalError = 0.0;
  bool revealAvailable = false;
  uint32_t revealPixelCount = 0u;
  double ghostEnergy = 0.0;
  double recoveryRmse = 0.0;
  AutotestQualityOracleBudgets budgets{};
  std::vector<std::string> failedThresholds{};
};

struct AutotestQualityOracleInputs {
  const nuri::tools::snapshot::SnapshotImage *output = nullptr;
  const nuri::tools::snapshot::SnapshotImage *reference = nullptr;
  const nuri::tools::snapshot::SnapshotImage *mask = nullptr;
  const nuri::tools::snapshot::SnapshotImage *previousOutput = nullptr;
  const nuri::tools::snapshot::SnapshotImage *previousReference = nullptr;
  const nuri::tools::snapshot::SnapshotImage *analyticMotion = nullptr;
  const nuri::tools::snapshot::SnapshotImage *revealMask = nullptr;
};

[[nodiscard]] Result<AutotestQualityOracleReport, std::string>
evaluateAutotestQualityOracle(const AutotestQualityOracle &oracle,
                              const AutotestQualityOracleInputs &inputs);

} // namespace nuri::tools::autotest
