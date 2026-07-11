#pragma once

#include "nuri/core/result.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace nuri::tools::benchmark {

enum class BenchmarkMetricIdRule : uint8_t {
  Exact,
  RenderGraphPassCpuTiming,
  RenderGraphPassGpuTiming,
};

enum class BenchmarkMetricUnit : uint8_t {
  Milliseconds,
  Mebibytes,
  Count,
  Ratio,
  WorldUnits,
  NormalizedVectorDelta,
};

enum class BenchmarkMetricNumericType : uint8_t {
  Float64,
  Uint64EncodedAsFloat64,
};

enum class BenchmarkMetricDirection : uint8_t {
  LowerIsBetter,
  Informational,
};

enum class BenchmarkMetricAggregation : uint8_t {
  MedianAndP95AcrossMeasuredFrames,
};

enum class BenchmarkMetricAvailability : uint8_t {
  EveryMeasuredFrame,
  WhenGpuTimingAvailable,
  WhenWholeFrameGpuTimingAvailable,
  WhenRenderGraphTelemetryAvailable,
  WhenPlatformMemoryAvailable,
  WhenMotionClassCoverageAvailable,
};

enum class BenchmarkMetricSamplingPhase : uint8_t {
  CpuMeasuredRegion,
  PostRenderMeasuredFrame,
  DelayedGpuReadback,
};

enum class BenchmarkMetricGateRole : uint8_t {
  Primary,
  Eligible,
  WorkloadCharacterization,
  Diagnostic,
};

struct BenchmarkMetricDescriptor {
  // Exact metric ID, or the documented rule text when idRule is not Exact.
  std::string_view idOrRule{};
  BenchmarkMetricIdRule idRule = BenchmarkMetricIdRule::Exact;
  BenchmarkMetricUnit unit = BenchmarkMetricUnit::Count;
  BenchmarkMetricNumericType numericType = BenchmarkMetricNumericType::Float64;
  BenchmarkMetricDirection direction = BenchmarkMetricDirection::Informational;
  BenchmarkMetricAggregation aggregation =
      BenchmarkMetricAggregation::MedianAndP95AcrossMeasuredFrames;
  BenchmarkMetricAvailability availability =
      BenchmarkMetricAvailability::EveryMeasuredFrame;
  BenchmarkMetricSamplingPhase samplingPhase =
      BenchmarkMetricSamplingPhase::PostRenderMeasuredFrame;
  BenchmarkMetricGateRole gateRole = BenchmarkMetricGateRole::Diagnostic;
};

struct BenchmarkMetricIndex {
  static constexpr uint16_t Invalid = UINT16_MAX;

  uint16_t value = Invalid;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return value != Invalid;
  }
  [[nodiscard]] friend constexpr bool
  operator==(BenchmarkMetricIndex, BenchmarkMetricIndex) noexcept = default;
};

[[nodiscard]] std::span<const BenchmarkMetricDescriptor>
benchmarkMetricDescriptors() noexcept;
[[nodiscard]] const BenchmarkMetricDescriptor *
findBenchmarkMetricDescriptor(std::string_view metricId) noexcept;
[[nodiscard]] std::optional<BenchmarkMetricIndex>
findExactBenchmarkMetricIndex(std::string_view metricId) noexcept;
[[nodiscard]] const BenchmarkMetricDescriptor *
benchmarkMetricDescriptor(BenchmarkMetricIndex index) noexcept;
[[nodiscard]] size_t exactBenchmarkMetricCount() noexcept;
[[nodiscard]] Result<const BenchmarkMetricDescriptor *, std::string>
requireBenchmarkMetricDescriptor(std::string_view metricId,
                                 std::string_view field);

[[nodiscard]] std::string_view
benchmarkMetricIdRuleName(BenchmarkMetricIdRule rule) noexcept;
[[nodiscard]] std::string_view
benchmarkMetricUnitName(BenchmarkMetricUnit unit) noexcept;
[[nodiscard]] std::string_view
benchmarkMetricNumericTypeName(BenchmarkMetricNumericType type) noexcept;
[[nodiscard]] std::string_view
benchmarkMetricDirectionName(BenchmarkMetricDirection direction) noexcept;
[[nodiscard]] std::string_view
benchmarkMetricAggregationName(BenchmarkMetricAggregation aggregation) noexcept;
[[nodiscard]] std::string_view benchmarkMetricAvailabilityName(
    BenchmarkMetricAvailability availability) noexcept;
[[nodiscard]] std::string_view
benchmarkMetricSamplingPhaseName(BenchmarkMetricSamplingPhase phase) noexcept;
[[nodiscard]] std::string_view
benchmarkMetricGateRoleName(BenchmarkMetricGateRole role) noexcept;

} // namespace nuri::tools::benchmark
