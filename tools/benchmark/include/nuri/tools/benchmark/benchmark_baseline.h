#pragma once

#include "nuri/core/result.h"
#include "nuri/tools/benchmark/benchmark_report.h"
#include "nuri/tools/core/baseline_profile.h"

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace nuri::tools::benchmark {

struct BenchmarkBaselineSource {
  BenchmarkReport report{};
  std::filesystem::path reportPath{};
};

struct BenchmarkBaselinePlanEntry {
  std::string path{};
  std::string operation{};
  std::string previousDigest{};
  std::string sourceDigest{};
};

struct BenchmarkBaselineGatePolicy {
  BenchmarkThresholds thresholds{};
  std::string source{};
};

struct BenchmarkBaselinePlan {
  std::string caseId{};
  std::string suite{};
  std::string profileId{};
  std::string reason{};
  std::string actor{};
  bool profileAuthoritative = false;
  bool authoritative = false;
  std::string sourceCommit{};
  std::string sourceReportDigest{};
  std::string acceptedReportDigest{};
  std::string previousApprovalDigest{};
  BenchmarkBaselineGatePolicy gatePolicy{};
  std::string digest{};
  std::vector<BenchmarkBaselinePlanEntry> entries{};
};

struct BenchmarkBaselineVerification {
  bool exists = false;
  bool valid = false;
  bool authoritative = false;
  std::string caseId{};
  std::string suite{};
  std::string profileId{};
  std::string reportDigest{};
  std::string approvalDigest{};
  std::string planDigest{};
  std::string sourceReportDigest{};
  std::string historyKey{};
  BenchmarkBaselineGatePolicy gatePolicy{};
  std::vector<std::string> errors{};
  std::vector<std::string> warnings{};
};

struct VerifiedBenchmarkBaseline {
  BenchmarkReport report{};
  BenchmarkBaselineVerification verification{};
  std::filesystem::path reportPath{};
};

enum class BenchmarkBaselinePromotionFault {
  None,
  AfterBackupRenameForTesting,
};

struct BenchmarkBaselineAcceptOptions {
  std::filesystem::path baselineRoot{};
  BenchmarkBaselinePromotionFault promotionFault =
      BenchmarkBaselinePromotionFault::None;
};

[[nodiscard]] std::filesystem::path defaultBenchmarkBaselineRoot();

[[nodiscard]] Result<BenchmarkBaselineSource, std::string>
loadBenchmarkBaselineSource(const std::filesystem::path &runRoot,
                            std::string_view caseId);

[[nodiscard]] Result<BenchmarkBaselinePlan, std::string>
planBenchmarkBaseline(const BenchmarkBaselineSource &source,
                      const nuri::tools::core::BaselineProfile &profile,
                      std::string_view reason, std::string_view actor,
                      const std::filesystem::path &baselineRoot = {});

[[nodiscard]] Result<std::string, std::string>
writeBenchmarkBaselinePlanJson(const BenchmarkBaselinePlan &plan);

[[nodiscard]] Result<bool, std::string>
acceptBenchmarkBaseline(const BenchmarkBaselineSource &source,
                        const nuri::tools::core::BaselineProfile &profile,
                        std::string_view reason, std::string_view actor,
                        std::string_view confirmPlanDigest,
                        const BenchmarkBaselineAcceptOptions &options = {});

[[nodiscard]] Result<BenchmarkBaselineVerification, std::string>
verifyBenchmarkBaseline(std::string_view caseId, std::string_view suite,
                        const nuri::tools::core::BaselineProfile &profile,
                        const std::filesystem::path &baselineRoot = {});

[[nodiscard]] Result<VerifiedBenchmarkBaseline, std::string>
loadVerifiedBenchmarkBaseline(std::string_view caseId, std::string_view suite,
                              const nuri::tools::core::BaselineProfile &profile,
                              const std::filesystem::path &baselineRoot = {});

[[nodiscard]] Result<std::string, std::string>
writeBenchmarkBaselineVerificationJson(
    const BenchmarkBaselineVerification &verification,
    std::string_view kind = "nuri.benchmark.baseline_verification");

} // namespace nuri::tools::benchmark
