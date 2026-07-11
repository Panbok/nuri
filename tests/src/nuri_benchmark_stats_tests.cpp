#include "nuri/tools/benchmark/benchmark_stats.h"

#include <gtest/gtest.h>

TEST(NuriBenchmarkStatsTest, FrameStatisticsIncludeR7P99) {
  std::vector<double> values;
  for (int value = 0; value < 100; ++value) {
    values.push_back(static_cast<double>(value));
  }
  const auto result = nuri::tools::benchmark::computeMetricStats(values);
  ASSERT_FALSE(result.hasError());
  EXPECT_DOUBLE_EQ(result.value().p95, 94.05);
  EXPECT_DOUBLE_EQ(result.value().p99, 98.01);
  EXPECT_DOUBLE_EQ(result.value().max, 99.0);
}

TEST(NuriBenchmarkStatsTest, RepeatComparisonReportsConfidenceAndEffect) {
  const auto result = nuri::tools::benchmark::computeRepeatComparison(
      {10.0, 10.1, 9.9, 10.0}, {11.0, 11.1, 10.9, 11.0});
  ASSERT_FALSE(result.hasError());
  EXPECT_EQ(result.value().baselineRepetitions, 4u);
  EXPECT_EQ(result.value().currentRepetitions, 4u);
  EXPECT_NEAR(result.value().absoluteDelta, 1.0, 1.0e-9);
  EXPECT_TRUE(result.value().percentDeltaDefined);
  EXPECT_GT(result.value().robustEffect, 1.0);
  EXPECT_FALSE(result.value().lowConfidence);
}

TEST(NuriBenchmarkStatsTest, NearZeroBaselineUsesAbsoluteDelta) {
  const auto result = nuri::tools::benchmark::computeRepeatComparison(
      {0.0, 0.0, 0.0}, {0.01, 0.01, 0.01});
  ASSERT_FALSE(result.hasError());
  EXPECT_FALSE(result.value().percentDeltaDefined);
  EXPECT_DOUBLE_EQ(result.value().percentDelta, 0.0);
  EXPECT_DOUBLE_EQ(result.value().absoluteDelta, 0.01);
}

TEST(NuriBenchmarkStatsTest, FewerThanThreeRepetitionsIsLowConfidence) {
  const auto result = nuri::tools::benchmark::computeRepeatComparison(
      {1.0}, {2.0});
  ASSERT_FALSE(result.hasError());
  EXPECT_TRUE(result.value().lowConfidence);
}

TEST(NuriBenchmarkStatsTest, InvalidRepeatInputFails) {
  EXPECT_TRUE(nuri::tools::benchmark::computeRepeatComparison({}, {1.0})
                  .hasError());
  EXPECT_TRUE(nuri::tools::benchmark::computeRepeatComparison({1.0}, {1.0}, 0.0)
                  .hasError());
}
