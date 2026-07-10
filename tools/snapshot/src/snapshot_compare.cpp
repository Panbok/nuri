#include "nuri/tools/snapshot/snapshot_compare.h"

#include "nuri/tools/core/atomic_file.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

#include <yyjson.h>

namespace nuri::tools::snapshot {
namespace {

void recordSemanticError(SnapshotSemanticMetrics &metrics, double error,
                         uint32_t x, uint32_t y) {
  metrics.meanError += error;
  ++metrics.validPixels;
  if (error > metrics.maxError) {
    metrics.maxError = error;
    metrics.maxErrorX = x;
    metrics.maxErrorY = y;
  }
  if (error <= 0.0) {
    return;
  }
  ++metrics.changedPixels;
  if (!metrics.changedBoundsValid) {
    metrics.changedBoundsValid = true;
    metrics.minChangedX = x;
    metrics.minChangedY = y;
    metrics.maxChangedX = x;
    metrics.maxChangedY = y;
    return;
  }
  metrics.minChangedX = std::min(metrics.minChangedX, x);
  metrics.minChangedY = std::min(metrics.minChangedY, y);
  metrics.maxChangedX = std::max(metrics.maxChangedX, x);
  metrics.maxChangedY = std::max(metrics.maxChangedY, y);
}

[[nodiscard]] double percentile99(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const size_t index =
      std::min(values.size() - 1u,
               static_cast<size_t>(std::ceil(values.size() * 0.99)) - 1u);
  return values[index];
}

bool computeSemanticMetrics(const SnapshotImage &actual,
                            const SnapshotImage &expected,
                            const SnapshotCompareProfile &profile,
                            const SnapshotCompareOptions &options,
                            SnapshotSemanticMetrics &metrics,
                            std::vector<bool> &validPixels,
                            std::string &error) {
  const uint32_t channels = actual.channelCount;
  if (profile.id == "normal" && channels < 3u) {
    error = "normal comparison requires at least three channels";
    return false;
  }
  if (profile.id == "velocity" && channels < 2u) {
    error = "velocity comparison requires at least two channels";
    return false;
  }
  if (profile.id == "hdr_color" && channels < 3u) {
    error = "HDR comparison requires at least three channels";
    return false;
  }
  metrics.unit = profile.id == "normal"      ? "degrees"
                 : profile.id == "velocity"  ? "vector_magnitude"
                 : profile.id == "mask"      ? "class_change"
                 : profile.id == "hdr_color" ? "log_luminance"
                 : profile.id == "depth" || profile.id == "shadow_depth" ||
                         profile.id == "scalar"
                     ? "relative"
                     : "absolute";
  if (profile.id == "hdr_color") {
    metrics.secondaryUnit = "relative_luminance";
  } else if (profile.id == "depth" || profile.id == "shadow_depth" ||
             profile.id == "scalar") {
    metrics.secondaryUnit = "absolute";
  }

  const size_t pixelCount = static_cast<size_t>(actual.width) * actual.height;
  validPixels.assign(pixelCount, false);
  if (!options.validityMask.empty() &&
      options.validityMask.size() != pixelCount) {
    error = "validity mask size does not match image dimensions";
    return false;
  }
  const bool hasRoi = options.roi.width != 0u || options.roi.height != 0u ||
                      options.roi.x != 0u || options.roi.y != 0u;
  if (hasRoi &&
      (options.roi.width == 0u || options.roi.height == 0u ||
       options.roi.x >= actual.width || options.roi.y >= actual.height ||
       options.roi.width > actual.width - options.roi.x ||
       options.roi.height > actual.height - options.roi.y)) {
    error = "ROI is outside image dimensions";
    return false;
  }
  std::vector<double> primaryErrors;
  std::vector<double> secondaryErrors;
  primaryErrors.reserve(pixelCount);
  secondaryErrors.reserve(pixelCount);
  double primarySquareSum = 0.0;
  double secondarySquareSum = 0.0;
  for (size_t pixel = 0u; pixel < pixelCount; ++pixel) {
    const size_t offset = pixel * channels;
    const uint32_t x = static_cast<uint32_t>(pixel % actual.width);
    const uint32_t y = static_cast<uint32_t>(pixel / actual.width);
    const bool inRoi = !hasRoi || (x >= options.roi.x && y >= options.roi.y &&
                                   x < options.roi.x + options.roi.width &&
                                   y < options.roi.y + options.roi.height);
    if (!inRoi ||
        (!options.validityMask.empty() && options.validityMask[pixel] == 0u)) {
      ++metrics.ignoredPixels;
      continue;
    }
    for (uint32_t channel = 0u; channel < channels; ++channel) {
      if (!std::isfinite(actual.values[offset + channel]) ||
          !std::isfinite(expected.values[offset + channel])) {
        error = "NaN or Inf value encountered in valid comparison region";
        return false;
      }
    }
    double semanticError = 0.0;
    double secondaryError = 0.0;
    bool hasSecondaryError = false;
    if (profile.id == "normal") {
      const double ax = actual.values[offset + 0u];
      const double ay = actual.values[offset + 1u];
      const double az = actual.values[offset + 2u];
      const double bx = expected.values[offset + 0u];
      const double by = expected.values[offset + 1u];
      const double bz = expected.values[offset + 2u];
      const double aLength = std::sqrt(ax * ax + ay * ay + az * az);
      const double bLength = std::sqrt(bx * bx + by * by + bz * bz);
      if (aLength <= 1.0e-12 && bLength <= 1.0e-12) {
        ++metrics.ignoredPixels;
        continue;
      }
      if (aLength <= 1.0e-12 || bLength <= 1.0e-12) {
        semanticError = 180.0;
      } else {
        const double cosine = std::clamp(
            (ax * bx + ay * by + az * bz) / (aLength * bLength), -1.0, 1.0);
        semanticError = std::acos(cosine) * 180.0 / std::numbers::pi;
      }
    } else if (profile.id == "velocity") {
      const double x = static_cast<double>(actual.values[offset + 0u]) -
                       expected.values[offset + 0u];
      const double y = static_cast<double>(actual.values[offset + 1u]) -
                       expected.values[offset + 1u];
      semanticError = std::sqrt(x * x + y * y);
    } else if (profile.id == "mask") {
      bool actualSet = false;
      bool expectedSet = false;
      for (uint32_t channel = 0u; channel < channels; ++channel) {
        actualSet = actualSet || actual.values[offset + channel] != 0.0f;
        expectedSet = expectedSet || expected.values[offset + channel] != 0.0f;
        if (actual.values[offset + channel] !=
            expected.values[offset + channel]) {
          semanticError = 1.0;
        }
      }
      if (actualSet && expectedSet) {
        ++metrics.truePositivePixels;
      } else if (actualSet) {
        ++metrics.falsePositivePixels;
      } else if (expectedSet) {
        ++metrics.falseNegativePixels;
      } else {
        ++metrics.trueNegativePixels;
      }
    } else if (profile.id == "hdr_color" && channels >= 3u) {
      const auto luminance = [&](const std::vector<float> &values) {
        return std::max(0.0, 0.2126 * values[offset + 0u] +
                                 0.7152 * values[offset + 1u] +
                                 0.0722 * values[offset + 2u]);
      };
      const double actualLuminance = luminance(actual.values);
      const double expectedLuminance = luminance(expected.values);
      const double absolute = std::abs(actualLuminance - expectedLuminance);
      semanticError =
          std::abs(std::log1p(actualLuminance) - std::log1p(expectedLuminance));
      secondaryError = absolute / std::max(1.0e-6, std::abs(expectedLuminance));
      hasSecondaryError = true;
    } else if (profile.id == "depth" || profile.id == "shadow_depth" ||
               profile.id == "scalar") {
      const double absolute = std::abs(
          static_cast<double>(actual.values[offset]) - expected.values[offset]);
      semanticError =
          absolute /
          std::max(1.0e-6,
                   std::abs(static_cast<double>(expected.values[offset])));
      secondaryError = absolute;
      hasSecondaryError = true;
    } else {
      for (uint32_t channel = 0u; channel < channels; ++channel) {
        semanticError = std::max(
            semanticError,
            std::abs(static_cast<double>(actual.values[offset + channel]) -
                     expected.values[offset + channel]));
      }
    }
    recordSemanticError(metrics, semanticError, x, y);
    validPixels[pixel] = true;
    primaryErrors.push_back(semanticError);
    primarySquareSum += semanticError * semanticError;
    if (semanticError > profile.maxAbsError) {
      ++metrics.failingPixels;
    }
    if (hasSecondaryError) {
      metrics.meanSecondaryError += secondaryError;
      metrics.maxSecondaryError =
          std::max(metrics.maxSecondaryError, secondaryError);
      secondaryErrors.push_back(secondaryError);
      secondarySquareSum += secondaryError * secondaryError;
      if (secondaryError > profile.maxAbsError) {
        ++metrics.secondaryFailingPixels;
      }
    }
  }
  if (metrics.validPixels == 0u) {
    error = "comparison region contains no valid pixels";
    return false;
  }
  metrics.meanError /= static_cast<double>(metrics.validPixels);
  metrics.rmse =
      std::sqrt(primarySquareSum / static_cast<double>(metrics.validPixels));
  metrics.p99Error = percentile99(std::move(primaryErrors));
  if (!secondaryErrors.empty()) {
    metrics.meanSecondaryError /= static_cast<double>(secondaryErrors.size());
    metrics.secondaryRmse = std::sqrt(
        secondarySquareSum / static_cast<double>(secondaryErrors.size()));
    metrics.p99SecondaryError = percentile99(std::move(secondaryErrors));
  }
  if (profile.id == "mask") {
    const uint64_t unionPixels = metrics.truePositivePixels +
                                 metrics.falsePositivePixels +
                                 metrics.falseNegativePixels;
    metrics.intersectionOverUnion =
        unionPixels == 0u ? 1.0
                          : static_cast<double>(metrics.truePositivePixels) /
                                static_cast<double>(unionPixels);
  }
  return true;
}

} // namespace

bool isBuiltinSnapshotCompareProfile(std::string_view id) noexcept {
  return id == "exact" || id == "ldr_color" || id == "debug_preview" ||
         id == "hdr_color" || id == "depth" || id == "shadow_depth" ||
         id == "scalar" || id == "mask" || id == "normal" || id == "velocity";
}

bool snapshotCompareProfileSupportsKind(std::string_view profile,
                                        RenderCaptureValueKind kind) noexcept {
  if (profile == "exact") {
    return true;
  }
  switch (kind) {
  case RenderCaptureValueKind::Color:
    return profile == "ldr_color";
  case RenderCaptureValueKind::LinearHdrColor:
    return profile == "hdr_color";
  case RenderCaptureValueKind::Depth:
    return profile == "depth";
  case RenderCaptureValueKind::ShadowDepth:
    return profile == "shadow_depth";
  case RenderCaptureValueKind::Normal:
    return profile == "normal";
  case RenderCaptureValueKind::Velocity:
    return profile == "velocity";
  case RenderCaptureValueKind::Mask:
    return profile == "mask";
  case RenderCaptureValueKind::Scalar:
    return profile == "scalar";
  case RenderCaptureValueKind::DebugPreview:
    return profile == "debug_preview";
  }
  return false;
}

SnapshotCompareProfile builtinSnapshotCompareProfile(std::string_view id) {
  SnapshotCompareProfile profile{};
  profile.id = std::string(id);
  if (id == "exact") {
    // Zero-initialized thresholds intentionally require exact equality.
  } else if (id == "ldr_color" || id == "debug_preview") {
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
    profile.valid = false;
  }
  return profile;
}

SnapshotCompareResult
compareSnapshotImages(const SnapshotImage &actual,
                      const SnapshotImage &expected,
                      const SnapshotCompareProfile &profile,
                      const SnapshotCompareOptions &options) {
  SnapshotCompareResult out{};
  if (!profile.valid || !isBuiltinSnapshotCompareProfile(profile.id)) {
    out.compatible = false;
    out.passed = false;
    out.errors.push_back("unknown compare profile '" + profile.id + "'");
    return out;
  }
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
  std::string semanticError;
  std::vector<bool> validPixels;
  if (!computeSemanticMetrics(actual, expected, profile, options, out.semantic,
                              validPixels, semanticError)) {
    out.compatible = false;
    out.passed = false;
    out.errors.push_back(std::move(semanticError));
    return out;
  }

  std::vector<double> absErrors;
  absErrors.reserve(actual.values.size());
  double absSum = 0.0;
  double squareSum = 0.0;
  for (size_t pixel = 0u; pixel < validPixels.size(); ++pixel) {
    if (!validPixels[pixel]) {
      continue;
    }
    const size_t offset = pixel * actual.channelCount;
    for (uint32_t channel = 0u; channel < actual.channelCount; ++channel) {
      const size_t i = offset + channel;
      const double err =
          std::abs(static_cast<double>(actual.values[i]) - expected.values[i]);
      absErrors.push_back(err);
      absSum += err;
      squareSum += err * err;
      if (err > profile.maxAbsError) {
        ++out.metrics.failingValues;
      }
    }
  }
  out.metrics.comparedValues = static_cast<uint64_t>(absErrors.size());
  if (!absErrors.empty()) {
    std::sort(absErrors.begin(), absErrors.end());
    const size_t p99Index =
        std::min(absErrors.size() - 1u,
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
  if (profile.id == "exact") {
    fail(out.metrics.maxAbsError > profile.maxAbsError, "max_abs_error");
    fail(out.metrics.meanAbsError > profile.meanAbsError, "mean_abs_error");
    fail(out.metrics.rmse > profile.rmse, "rmse");
    fail(out.metrics.p99AbsError > profile.p99AbsError, "p99_abs_error");
    fail(out.metrics.failingValues > profile.maxFailingValues,
         "failing_values");
    return out;
  }
  fail(out.semantic.maxError > profile.maxAbsError, "semantic_max_error");
  fail(out.semantic.meanError > profile.meanAbsError, "semantic_mean_error");
  fail(out.semantic.rmse > profile.rmse, "semantic_rmse");
  fail(out.semantic.p99Error > profile.p99AbsError, "semantic_p99_error");
  fail(out.semantic.failingPixels > profile.maxFailingValues,
       profile.id == "mask" ? "mask_changed_pixels"
                            : "semantic_failing_pixels");
  if (!out.semantic.secondaryUnit.empty()) {
    fail(out.semantic.maxSecondaryError > profile.maxAbsError,
         "semantic_secondary_max_error");
    fail(out.semantic.meanSecondaryError > profile.meanAbsError,
         "semantic_secondary_mean_error");
    fail(out.semantic.secondaryRmse > profile.rmse, "semantic_secondary_rmse");
    fail(out.semantic.p99SecondaryError > profile.p99AbsError,
         "semantic_secondary_p99_error");
    fail(out.semantic.secondaryFailingPixels > profile.maxFailingValues,
         "semantic_secondary_failing_pixels");
  }
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
      maxErr = std::max(
          maxErr, std::abs(static_cast<double>(actual.values[src + c] -
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

Result<bool, std::string>
writeSnapshotComparisonFile(const SnapshotCompareResult &comparison,
                            std::string_view profile,
                            const std::filesystem::path &path) {
  using JsonDoc =
      std::unique_ptr<yyjson_mut_doc, decltype(&yyjson_mut_doc_free)>;
  JsonDoc doc(yyjson_mut_doc_new(nullptr), &yyjson_mut_doc_free);
  if (!doc) {
    return Result<bool, std::string>::makeError(
        "writeSnapshotComparisonFile: failed to allocate JSON document");
  }
  yyjson_mut_val *root = yyjson_mut_obj(doc.get());
  yyjson_mut_doc_set_root(doc.get(), root);
  yyjson_mut_obj_add_uint(doc.get(), root, "schemaVersion", 1u);
  yyjson_mut_obj_add_strcpy(doc.get(), root, "kind",
                            "nuri.snapshot.comparison");
  const std::string profileText(profile);
  yyjson_mut_obj_add_strcpy(doc.get(), root, "profile", profileText.c_str());
  yyjson_mut_obj_add_bool(doc.get(), root, "compatible", comparison.compatible);
  yyjson_mut_obj_add_bool(doc.get(), root, "passed", comparison.passed);
  yyjson_mut_val *metrics = yyjson_mut_obj(doc.get());
  yyjson_mut_obj_add_real(doc.get(), metrics, "meanAbsError",
                          comparison.metrics.meanAbsError);
  yyjson_mut_obj_add_real(doc.get(), metrics, "rmse", comparison.metrics.rmse);
  yyjson_mut_obj_add_real(doc.get(), metrics, "maxAbsError",
                          comparison.metrics.maxAbsError);
  yyjson_mut_obj_add_real(doc.get(), metrics, "p99AbsError",
                          comparison.metrics.p99AbsError);
  yyjson_mut_obj_add_uint(doc.get(), metrics, "failingValues",
                          comparison.metrics.failingValues);
  yyjson_mut_obj_add_uint(doc.get(), metrics, "comparedValues",
                          comparison.metrics.comparedValues);
  yyjson_mut_obj_add_val(doc.get(), root, "metrics", metrics);
  yyjson_mut_val *semantic = yyjson_mut_obj(doc.get());
  yyjson_mut_obj_add_strcpy(doc.get(), semantic, "unit",
                            comparison.semantic.unit.c_str());
  yyjson_mut_obj_add_real(doc.get(), semantic, "meanError",
                          comparison.semantic.meanError);
  yyjson_mut_obj_add_real(doc.get(), semantic, "maxError",
                          comparison.semantic.maxError);
  yyjson_mut_obj_add_real(doc.get(), semantic, "rmse",
                          comparison.semantic.rmse);
  yyjson_mut_obj_add_real(doc.get(), semantic, "p99Error",
                          comparison.semantic.p99Error);
  yyjson_mut_obj_add_uint(doc.get(), semantic, "failingPixels",
                          comparison.semantic.failingPixels);
  yyjson_mut_obj_add_uint(doc.get(), semantic, "validPixels",
                          comparison.semantic.validPixels);
  yyjson_mut_obj_add_uint(doc.get(), semantic, "ignoredPixels",
                          comparison.semantic.ignoredPixels);
  yyjson_mut_obj_add_uint(doc.get(), semantic, "changedPixels",
                          comparison.semantic.changedPixels);
  if (!comparison.semantic.secondaryUnit.empty()) {
    yyjson_mut_obj_add_strcpy(doc.get(), semantic, "secondaryUnit",
                              comparison.semantic.secondaryUnit.c_str());
    yyjson_mut_obj_add_real(doc.get(), semantic, "meanSecondaryError",
                            comparison.semantic.meanSecondaryError);
    yyjson_mut_obj_add_real(doc.get(), semantic, "maxSecondaryError",
                            comparison.semantic.maxSecondaryError);
    yyjson_mut_obj_add_real(doc.get(), semantic, "secondaryRmse",
                            comparison.semantic.secondaryRmse);
    yyjson_mut_obj_add_real(doc.get(), semantic, "p99SecondaryError",
                            comparison.semantic.p99SecondaryError);
    yyjson_mut_obj_add_uint(doc.get(), semantic, "secondaryFailingPixels",
                            comparison.semantic.secondaryFailingPixels);
  }
  yyjson_mut_obj_add_uint(doc.get(), semantic, "truePositivePixels",
                          comparison.semantic.truePositivePixels);
  yyjson_mut_obj_add_uint(doc.get(), semantic, "trueNegativePixels",
                          comparison.semantic.trueNegativePixels);
  yyjson_mut_obj_add_uint(doc.get(), semantic, "falsePositivePixels",
                          comparison.semantic.falsePositivePixels);
  yyjson_mut_obj_add_uint(doc.get(), semantic, "falseNegativePixels",
                          comparison.semantic.falseNegativePixels);
  yyjson_mut_obj_add_real(doc.get(), semantic, "intersectionOverUnion",
                          comparison.semantic.intersectionOverUnion);
  yyjson_mut_obj_add_bool(doc.get(), semantic, "changedBoundsValid",
                          comparison.semantic.changedBoundsValid);
  if (comparison.semantic.changedBoundsValid) {
    yyjson_mut_val *bounds = yyjson_mut_arr(doc.get());
    yyjson_mut_arr_add_uint(doc.get(), bounds, comparison.semantic.minChangedX);
    yyjson_mut_arr_add_uint(doc.get(), bounds, comparison.semantic.minChangedY);
    yyjson_mut_arr_add_uint(doc.get(), bounds, comparison.semantic.maxChangedX);
    yyjson_mut_arr_add_uint(doc.get(), bounds, comparison.semantic.maxChangedY);
    yyjson_mut_obj_add_val(doc.get(), semantic, "changedBounds", bounds);
  }
  yyjson_mut_val *maxCoordinate = yyjson_mut_arr(doc.get());
  yyjson_mut_arr_add_uint(doc.get(), maxCoordinate,
                          comparison.semantic.maxErrorX);
  yyjson_mut_arr_add_uint(doc.get(), maxCoordinate,
                          comparison.semantic.maxErrorY);
  yyjson_mut_obj_add_val(doc.get(), semantic, "maxErrorCoordinate",
                         maxCoordinate);
  yyjson_mut_obj_add_val(doc.get(), root, "semanticMetrics", semantic);
  const auto addStrings = [&](const char *name,
                              const std::vector<std::string> &values) {
    yyjson_mut_val *array = yyjson_mut_arr(doc.get());
    for (const std::string &value : values) {
      yyjson_mut_arr_add_strcpy(doc.get(), array, value.c_str());
    }
    yyjson_mut_obj_add_val(doc.get(), root, name, array);
  };
  addStrings("failedThresholds", comparison.failedThresholds);
  addStrings("errors", comparison.errors);

  size_t length = 0u;
  char *json = yyjson_mut_write_opts(doc.get(), YYJSON_WRITE_PRETTY, nullptr,
                                     &length, nullptr);
  if (json == nullptr) {
    return Result<bool, std::string>::makeError(
        "writeSnapshotComparisonFile: failed to serialize JSON");
  }
  const auto written = nuri::tools::core::atomicWriteTextFile(
      path, std::string_view(json, length));
  std::free(json);
  if (written.hasError()) {
    return Result<bool, std::string>::makeError(
        "writeSnapshotComparisonFile: " + written.error());
  }
  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri::tools::snapshot
