#include "nuri/tools/snapshot/snapshot_compare.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace nuri::tools::snapshot {

SnapshotCompareProfile builtinSnapshotCompareProfile(std::string_view id) {
  SnapshotCompareProfile profile{};
  profile.id = std::string(id);
  if (id == "ldr_color" || id == "debug_preview") {
    profile.maxAbsError = 1.0 / 255.0;
    profile.meanAbsError = 0.0005;
    profile.rmse = 0.001;
    profile.p99AbsError = 1.0 / 255.0;
    profile.maxFailingValues = 0u;
  } else if (id == "hdr_color") {
    profile.maxAbsError = 0.01;
    profile.meanAbsError = 0.001;
    profile.rmse = 0.002;
    profile.p99AbsError = 0.005;
    profile.maxFailingValues = 0u;
  } else if (id == "depth" || id == "shadow_depth" || id == "scalar" ||
             id == "mask" || id == "normal" || id == "velocity") {
    profile.maxAbsError = 0.0001;
    profile.meanAbsError = 0.00001;
    profile.rmse = 0.00002;
    profile.p99AbsError = 0.0001;
    profile.maxFailingValues = 0u;
  } else {
    profile.id = "exact";
  }
  return profile;
}

SnapshotCompareResult
compareSnapshotImages(const SnapshotImage &actual,
                      const SnapshotImage &expected,
                      const SnapshotCompareProfile &profile) {
  SnapshotCompareResult out{};
  if (actual.width != expected.width || actual.height != expected.height ||
      actual.channelCount != expected.channelCount) {
    out.compatible = false;
    out.passed = false;
    out.errors.push_back("image dimensions or channel count mismatch");
    return out;
  }
  if (actual.values.size() != expected.values.size()) {
    out.compatible = false;
    out.passed = false;
    out.errors.push_back("image value count mismatch");
    return out;
  }

  std::vector<double> absErrors;
  absErrors.reserve(actual.values.size());
  double absSum = 0.0;
  double squareSum = 0.0;
  for (size_t i = 0u; i < actual.values.size(); ++i) {
    const float a = actual.values[i];
    const float b = expected.values[i];
    if (!std::isfinite(a) || !std::isfinite(b)) {
      out.compatible = false;
      out.passed = false;
      out.errors.push_back("NaN or Inf value encountered");
      return out;
    }
    const double err = std::abs(static_cast<double>(a) - b);
    absErrors.push_back(err);
    absSum += err;
    squareSum += err * err;
    if (err > profile.maxAbsError) {
      ++out.metrics.failingValues;
    }
  }
  out.metrics.comparedValues = static_cast<uint64_t>(absErrors.size());
  if (!absErrors.empty()) {
    std::sort(absErrors.begin(), absErrors.end());
    const size_t p99Index = std::min(
        absErrors.size() - 1u,
        static_cast<size_t>(std::ceil(absErrors.size() * 0.99)) - 1u);
    out.metrics.meanAbsError = absSum / static_cast<double>(absErrors.size());
    out.metrics.rmse =
        std::sqrt(squareSum / static_cast<double>(absErrors.size()));
    out.metrics.maxAbsError = absErrors.back();
    out.metrics.p99AbsError = absErrors[p99Index];
  }

  const auto fail = [&](bool condition, std::string name) {
    if (condition) {
      out.failedThresholds.push_back(std::move(name));
      out.passed = false;
    }
  };
  fail(out.metrics.maxAbsError > profile.maxAbsError, "max_abs_error");
  fail(out.metrics.meanAbsError > profile.meanAbsError, "mean_abs_error");
  fail(out.metrics.rmse > profile.rmse, "rmse");
  fail(out.metrics.p99AbsError > profile.p99AbsError, "p99_abs_error");
  fail(out.metrics.failingValues > profile.maxFailingValues,
       "failing_values");
  return out;
}

Result<bool, std::string>
writeSnapshotDiffPng(const SnapshotImage &actual, const SnapshotImage &expected,
                     const std::filesystem::path &path) {
  if (actual.width != expected.width || actual.height != expected.height ||
      actual.channelCount != expected.channelCount ||
      actual.values.size() != expected.values.size()) {
    return Result<bool, std::string>::makeError(
        "writeSnapshotDiffPng: incompatible images");
  }
  SnapshotImage diff{};
  diff.width = actual.width;
  diff.height = actual.height;
  diff.channelCount = 4u;
  diff.values.resize(static_cast<size_t>(diff.width) * diff.height * 4u);
  const size_t pixelCount = static_cast<size_t>(diff.width) * diff.height;
  for (size_t i = 0u; i < pixelCount; ++i) {
    const size_t src = i * actual.channelCount;
    const size_t dst = i * 4u;
    double maxErr = 0.0;
    for (uint32_t c = 0u; c < actual.channelCount; ++c) {
      maxErr = std::max(maxErr, std::abs(static_cast<double>(
                                 actual.values[src + c] -
                                 expected.values[src + c])));
    }
    const float heat = static_cast<float>(std::min(maxErr * 16.0, 1.0));
    diff.values[dst + 0u] = heat;
    diff.values[dst + 1u] = 0.0f;
    diff.values[dst + 2u] = 1.0f - heat;
    diff.values[dst + 3u] = 1.0f;
  }
  return writeSnapshotPreviewPng(diff, path);
}

} // namespace nuri::tools::snapshot

