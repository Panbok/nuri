#include "nuri/tools/autotest/autotest_quality_oracle.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace nuri::tools::autotest {
namespace {

using nuri::tools::snapshot::SnapshotImage;

[[nodiscard]] bool imagePayloadValid(const SnapshotImage &image,
                                     uint32_t requiredChannels) noexcept {
  return image.width > 0u && image.height > 0u &&
         image.channelCount >= requiredChannels &&
         image.values.size() >= static_cast<size_t>(image.width) *
                                    image.height * image.channelCount;
}

[[nodiscard]] bool sameExtent(const SnapshotImage &lhs,
                              const SnapshotImage &rhs) noexcept {
  return lhs.width == rhs.width && lhs.height == rhs.height;
}

[[nodiscard]] size_t pixelBase(const SnapshotImage &image,
                               size_t pixel) noexcept {
  return pixel * image.channelCount;
}

[[nodiscard]] double luma(const SnapshotImage &image, size_t pixel) noexcept {
  const size_t base = pixelBase(image, pixel);
  return 0.2126 * static_cast<double>(image.values[base]) +
         0.7152 * static_cast<double>(image.values[base + 1u]) +
         0.0722 * static_cast<double>(image.values[base + 2u]);
}

[[nodiscard]] bool finiteRgb(const SnapshotImage &image,
                             size_t pixel) noexcept {
  const size_t base = pixelBase(image, pixel);
  return std::isfinite(image.values[base]) &&
         std::isfinite(image.values[base + 1u]) &&
         std::isfinite(image.values[base + 2u]);
}

[[nodiscard]] Result<std::vector<uint8_t>, std::string>
makeSelection(const SnapshotImage &base, const SnapshotImage *mask,
              std::string_view label) {
  const size_t pixelCount = static_cast<size_t>(base.width) * base.height;
  std::vector<uint8_t> selected(pixelCount, 1u);
  if (mask == nullptr) {
    return Result<std::vector<uint8_t>, std::string>::makeResult(
        std::move(selected));
  }
  if (!imagePayloadValid(*mask, 1u) || !sameExtent(base, *mask)) {
    return Result<std::vector<uint8_t>, std::string>::makeError(
        std::string(label) + " mask is missing or incompatible");
  }
  for (size_t pixel = 0u; pixel < pixelCount; ++pixel) {
    const float value = mask->values[pixelBase(*mask, pixel)];
    if (!std::isfinite(value)) {
      return Result<std::vector<uint8_t>, std::string>::makeError(
          std::string(label) + " mask contains a non-finite value");
    }
    selected[pixel] = value > 0.5f ? 1u : 0u;
  }
  if (std::none_of(selected.begin(), selected.end(),
                   [](uint8_t value) { return value != 0u; })) {
    return Result<std::vector<uint8_t>, std::string>::makeError(
        std::string(label) + " mask selects no pixels");
  }
  return Result<std::vector<uint8_t>, std::string>::makeResult(
      std::move(selected));
}

[[nodiscard]] uint32_t
largestEightConnectedComponent(std::span<const uint8_t> pixels, uint32_t width,
                               uint32_t height) {
  std::vector<uint8_t> visited(pixels.size(), 0u);
  std::vector<size_t> queue;
  uint32_t largest = 0u;
  for (size_t seed = 0u; seed < pixels.size(); ++seed) {
    if (pixels[seed] == 0u || visited[seed] != 0u) {
      continue;
    }
    queue.clear();
    queue.push_back(seed);
    visited[seed] = 1u;
    size_t read = 0u;
    uint32_t count = 0u;
    while (read < queue.size()) {
      const size_t pixel = queue[read++];
      ++count;
      const int x = static_cast<int>(pixel % width);
      const int y = static_cast<int>(pixel / width);
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          if (dx == 0 && dy == 0) {
            continue;
          }
          const int nx = x + dx;
          const int ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= static_cast<int>(width) ||
              ny >= static_cast<int>(height)) {
            continue;
          }
          const size_t neighbor =
              static_cast<size_t>(ny) * width + static_cast<size_t>(nx);
          if (pixels[neighbor] != 0u && visited[neighbor] == 0u) {
            visited[neighbor] = 1u;
            queue.push_back(neighbor);
          }
        }
      }
    }
    largest = std::max(largest, count);
  }
  return largest;
}

[[nodiscard]] std::optional<double>
bilinearLuma(const SnapshotImage &image, double uvX, double uvY,
             std::span<const uint8_t> stableMask) {
  const double sampleX = uvX * image.width - 0.5;
  const double sampleY = uvY * image.height - 0.5;
  const int x0 = static_cast<int>(std::floor(sampleX));
  const int y0 = static_cast<int>(std::floor(sampleY));
  const double tx = sampleX - x0;
  const double ty = sampleY - y0;
  const int x1 = tx <= 1.0e-12 ? x0 : x0 + 1;
  const int y1 = ty <= 1.0e-12 ? y0 : y0 + 1;
  if (x0 < 0 || y0 < 0 || x1 >= static_cast<int>(image.width) ||
      y1 >= static_cast<int>(image.height)) {
    return std::nullopt;
  }
  const std::array<size_t, 4> pixels{
      static_cast<size_t>(y0) * image.width + static_cast<size_t>(x0),
      static_cast<size_t>(y0) * image.width + static_cast<size_t>(x1),
      static_cast<size_t>(y1) * image.width + static_cast<size_t>(x0),
      static_cast<size_t>(y1) * image.width + static_cast<size_t>(x1),
  };
  for (const size_t pixel : pixels) {
    if (!finiteRgb(image, pixel) ||
        (!stableMask.empty() && stableMask[pixel] == 0u)) {
      return std::nullopt;
    }
  }
  const double top =
      std::lerp(luma(image, pixels[0]), luma(image, pixels[1]), tx);
  const double bottom =
      std::lerp(luma(image, pixels[2]), luma(image, pixels[3]), tx);
  return std::lerp(top, bottom, ty);
}

[[nodiscard]] bool finiteNonNegative(double value) noexcept {
  return std::isfinite(value) && value >= 0.0;
}

struct EdgeMetrics {
  std::string_view axis{};
  uint32_t profileCount = 0u;
  uint32_t unresolvedProfileCount = 0u;
  uint32_t resolvedProfileCount = 0u;
  double referenceWidthSum = 0.0;
  double outputWidthSum = 0.0;
  double overshoot = 0.0;
  double undershoot = 0.0;
};

[[nodiscard]] std::optional<double>
crossingPosition(std::span<const double> profile, double target, bool forward) {
  for (size_t step = 1u; step < profile.size(); ++step) {
    const size_t previousIndex = forward ? step - 1u : profile.size() - step;
    const size_t currentIndex = forward ? step : profile.size() - step - 1u;
    const double previous = profile[previousIndex];
    const double current = profile[currentIndex];
    if (previous <= target && current >= target && current != previous) {
      const double t = (target - previous) / (current - previous);
      return std::lerp(static_cast<double>(previousIndex),
                       static_cast<double>(currentIndex), t);
    }
  }
  return std::nullopt;
}

[[nodiscard]] EdgeMetrics evaluateEdgeAxis(const SnapshotImage &output,
                                           const SnapshotImage &reference,
                                           std::span<const uint8_t> selected,
                                           double lscale, bool horizontal) {
  EdgeMetrics metrics{.axis = horizontal ? "horizontal" : "vertical"};
  const uint32_t lineCount = horizontal ? output.height : output.width;
  const uint32_t profileLength = horizontal ? output.width : output.height;
  if (profileLength < 3u) {
    return metrics;
  }
  std::vector<double> outputProfile(profileLength);
  std::vector<double> referenceProfile(profileLength);
  for (uint32_t line = 0u; line < lineCount; ++line) {
    bool validProfile = true;
    for (uint32_t position = 0u; position < profileLength; ++position) {
      const uint32_t x = horizontal ? position : line;
      const uint32_t y = horizontal ? line : position;
      const size_t pixel = static_cast<size_t>(y) * output.width + x;
      if (selected[pixel] == 0u || !finiteRgb(output, pixel) ||
          !finiteRgb(reference, pixel)) {
        validProfile = false;
        break;
      }
      outputProfile[position] = luma(output, pixel);
      referenceProfile[position] = luma(reference, pixel);
    }
    if (!validProfile) {
      continue;
    }
    const auto [referenceMinIt, referenceMaxIt] =
        std::minmax_element(referenceProfile.begin(), referenceProfile.end());
    const double referenceLow = *referenceMinIt;
    const double referenceHigh = *referenceMaxIt;
    const double referenceRange = referenceHigh - referenceLow;
    if (referenceRange < 0.05 * lscale) {
      continue;
    }
    ++metrics.profileCount;
    const size_t endpointCount =
        std::max<size_t>(1u, referenceProfile.size() / 4u);
    double firstMean = 0.0;
    double lastMean = 0.0;
    for (size_t i = 0u; i < endpointCount; ++i) {
      firstMean += referenceProfile[i];
      lastMean += referenceProfile[referenceProfile.size() - i - 1u];
    }
    const bool forward = firstMean <= lastMean;
    const double threshold10 = referenceLow + 0.10 * referenceRange;
    const double threshold90 = referenceLow + 0.90 * referenceRange;
    const std::optional<double> reference10 =
        crossingPosition(referenceProfile, threshold10, forward);
    const std::optional<double> reference90 =
        crossingPosition(referenceProfile, threshold90, forward);
    const std::optional<double> output10 =
        crossingPosition(outputProfile, threshold10, forward);
    const std::optional<double> output90 =
        crossingPosition(outputProfile, threshold90, forward);
    const auto [outputMinIt, outputMaxIt] =
        std::minmax_element(outputProfile.begin(), outputProfile.end());
    metrics.overshoot =
        std::max(metrics.overshoot,
                 std::max(0.0, *outputMaxIt - referenceHigh) / lscale);
    metrics.undershoot =
        std::max(metrics.undershoot,
                 std::max(0.0, referenceLow - *outputMinIt) / lscale);
    if (!reference10.has_value() || !reference90.has_value() ||
        !output10.has_value() || !output90.has_value()) {
      ++metrics.unresolvedProfileCount;
      continue;
    }
    ++metrics.resolvedProfileCount;
    metrics.referenceWidthSum += std::abs(*reference90 - *reference10);
    metrics.outputWidthSum += std::abs(*output90 - *output10);
  }
  return metrics;
}

[[nodiscard]] bool
validConfiguration(const AutotestQualityOracle &oracle) noexcept {
  const AutotestQualityOracleBudgets &budgets = oracle.budgets;
  return oracle.schemaVersion == 1u && std::isfinite(oracle.lscale) &&
         oracle.lscale > 0.0 && finiteNonNegative(budgets.normalizedMaeMax) &&
         finiteNonNegative(budgets.normalizedRmseMax) &&
         std::isfinite(budgets.lumaSsimMin) && budgets.lumaSsimMin >= 0.0 &&
         budgets.lumaSsimMin <= 1.0 &&
         finiteNonNegative(budgets.darkCollapsePercentMax) &&
         budgets.darkCollapsePercentMax <= 100.0 &&
         budgets.darkCollapseComponentMaxPixels > 0u &&
         finiteNonNegative(budgets.relativeLumaEnergyDriftMax) &&
         finiteNonNegative(budgets.edgeWidthRatioMin) &&
         finiteNonNegative(budgets.edgeWidthRatioMax) &&
         budgets.edgeWidthRatioMin <= budgets.edgeWidthRatioMax &&
         finiteNonNegative(budgets.edgeOvershootMax) &&
         finiteNonNegative(budgets.edgeUndershootMax) &&
         (!oracle.temporal.has_value() ||
          (finiteNonNegative(budgets.temporalErrorMax) &&
           finiteNonNegative(budgets.ghostEnergyMax) &&
           finiteNonNegative(budgets.recoveryRmseMax)));
}

} // namespace

Result<AutotestQualityOracleReport, std::string>
evaluateAutotestQualityOracle(const AutotestQualityOracle &oracle,
                              const AutotestQualityOracleInputs &inputs) {
  if (!validConfiguration(oracle)) {
    return Result<AutotestQualityOracleReport, std::string>::makeError(
        "quality oracle configuration is invalid");
  }
  if (inputs.output == nullptr || inputs.reference == nullptr ||
      !imagePayloadValid(*inputs.output, 3u) ||
      !imagePayloadValid(*inputs.reference, 3u) ||
      !sameExtent(*inputs.output, *inputs.reference)) {
    return Result<AutotestQualityOracleReport, std::string>::makeError(
        "quality oracle requires compatible RGB output and reference images");
  }
  const SnapshotImage &output = *inputs.output;
  const SnapshotImage &reference = *inputs.reference;
  auto selectionResult = makeSelection(output, inputs.mask, "quality oracle");
  if (selectionResult.hasError()) {
    return Result<AutotestQualityOracleReport, std::string>::makeError(
        selectionResult.error());
  }
  const std::vector<uint8_t> &selected = selectionResult.value();

  AutotestQualityOracleReport report{};
  report.outputTarget = oracle.outputTarget;
  report.referencePath = oracle.reference.path.generic_string();
  report.schemaVersion = oracle.schemaVersion;
  report.referenceVersion = oracle.reference.version;
  report.maskVersion = oracle.mask.has_value() ? oracle.mask->version : 0u;
  report.lscale = oracle.lscale;
  report.budgets = oracle.budgets;

  const size_t pixelCount = static_cast<size_t>(output.width) * output.height;
  std::vector<uint8_t> darkCollapse(pixelCount, 0u);
  double absoluteErrorSum = 0.0;
  double squaredErrorSum = 0.0;
  double outputLumaSum = 0.0;
  double referenceLumaSum = 0.0;
  double outputLumaSquaredSum = 0.0;
  double referenceLumaSquaredSum = 0.0;
  double lumaCrossSum = 0.0;
  uint64_t finiteComponentCount = 0u;
  for (size_t pixel = 0u; pixel < pixelCount; ++pixel) {
    if (selected[pixel] == 0u) {
      continue;
    }
    ++report.selectedPixelCount;
    const size_t outputBase = pixelBase(output, pixel);
    const size_t referenceBase = pixelBase(reference, pixel);
    bool finite = true;
    for (uint32_t channel = 0u; channel < 3u; ++channel) {
      const float outputValue = output.values[outputBase + channel];
      const float referenceValue = reference.values[referenceBase + channel];
      if (!std::isfinite(outputValue)) {
        ++report.nonFiniteValueCount;
        finite = false;
      }
      if (!std::isfinite(referenceValue)) {
        ++report.nonFiniteValueCount;
        finite = false;
      }
    }
    if (!finite) {
      continue;
    }
    ++report.finitePixelCount;
    for (uint32_t channel = 0u; channel < 3u; ++channel) {
      const double error =
          static_cast<double>(output.values[outputBase + channel]) -
          reference.values[referenceBase + channel];
      absoluteErrorSum += std::abs(error);
      squaredErrorSum += error * error;
      ++finiteComponentCount;
    }
    const double outputLuma = luma(output, pixel);
    const double referenceLuma = luma(reference, pixel);
    outputLumaSum += outputLuma;
    referenceLumaSum += referenceLuma;
    outputLumaSquaredSum += outputLuma * outputLuma;
    referenceLumaSquaredSum += referenceLuma * referenceLuma;
    lumaCrossSum += outputLuma * referenceLuma;
    if (referenceLuma >= 0.05 * oracle.lscale &&
        outputLuma < 0.10 * referenceLuma) {
      darkCollapse[pixel] = 1u;
      ++report.darkCollapsePixelCount;
    }
  }
  if (report.selectedPixelCount == 0u) {
    return Result<AutotestQualityOracleReport, std::string>::makeError(
        "quality oracle selected no pixels");
  }
  if (finiteComponentCount > 0u) {
    report.normalizedHdrMae = absoluteErrorSum /
                              static_cast<double>(finiteComponentCount) /
                              oracle.lscale;
    report.normalizedHdrRmse =
        std::sqrt(squaredErrorSum / static_cast<double>(finiteComponentCount)) /
        oracle.lscale;
  }
  if (report.finitePixelCount > 0u) {
    const double count = static_cast<double>(report.finitePixelCount);
    const double outputMean = outputLumaSum / count;
    const double referenceMean = referenceLumaSum / count;
    const double outputVariance =
        std::max(0.0, outputLumaSquaredSum / count - outputMean * outputMean);
    const double referenceVariance = std::max(
        0.0, referenceLumaSquaredSum / count - referenceMean * referenceMean);
    const double covariance = lumaCrossSum / count - outputMean * referenceMean;
    const double c1 = std::pow(0.01 * oracle.lscale, 2.0);
    const double c2 = std::pow(0.03 * oracle.lscale, 2.0);
    report.lumaSsim =
        ((2.0 * outputMean * referenceMean + c1) * (2.0 * covariance + c2)) /
        ((outputMean * outputMean + referenceMean * referenceMean + c1) *
         (outputVariance + referenceVariance + c2));
    report.relativeLumaEnergyDrift =
        std::abs(outputLumaSum - referenceLumaSum) /
        std::max(std::abs(referenceLumaSum), 1.0e-12);
  }
  report.darkCollapsePercent =
      100.0 * static_cast<double>(report.darkCollapsePixelCount) /
      static_cast<double>(report.selectedPixelCount);
  report.darkCollapseMaxComponentPixels =
      largestEightConnectedComponent(darkCollapse, output.width, output.height);

  const EdgeMetrics horizontal =
      evaluateEdgeAxis(output, reference, selected, oracle.lscale, true);
  const EdgeMetrics vertical =
      evaluateEdgeAxis(output, reference, selected, oracle.lscale, false);
  const EdgeMetrics &edge =
      vertical.profileCount > horizontal.profileCount ? vertical : horizontal;
  if (edge.profileCount > 0u) {
    report.edgeAvailable = true;
    report.edgeAxis = std::string(edge.axis);
    report.edgeProfileCount = edge.profileCount;
    report.edgeUnresolvedProfileCount = edge.unresolvedProfileCount;
    report.edgeOvershoot = edge.overshoot;
    report.edgeUndershoot = edge.undershoot;
    if (edge.resolvedProfileCount > 0u) {
      report.referenceEdgeWidth10To90 =
          edge.referenceWidthSum / edge.resolvedProfileCount;
      report.outputEdgeWidth10To90 =
          edge.outputWidthSum / edge.resolvedProfileCount;
      report.edgeWidthRatio =
          report.outputEdgeWidth10To90 /
          std::max(report.referenceEdgeWidth10To90, 1.0e-12);
    }
  }

  if (oracle.temporal.has_value()) {
    if (inputs.previousOutput == nullptr ||
        inputs.previousReference == nullptr ||
        inputs.analyticMotion == nullptr || inputs.revealMask == nullptr ||
        !imagePayloadValid(*inputs.previousOutput, 3u) ||
        !imagePayloadValid(*inputs.previousReference, 3u) ||
        !imagePayloadValid(*inputs.analyticMotion, 2u) ||
        !sameExtent(output, *inputs.previousOutput) ||
        !sameExtent(output, *inputs.previousReference) ||
        !sameExtent(output, *inputs.analyticMotion)) {
      return Result<AutotestQualityOracleReport, std::string>::makeError(
          "quality oracle temporal inputs are missing or incompatible");
    }
    double temporalErrorSum = 0.0;
    for (size_t pixel = 0u; pixel < pixelCount; ++pixel) {
      if (selected[pixel] == 0u || !finiteRgb(output, pixel) ||
          !finiteRgb(reference, pixel)) {
        continue;
      }
      const size_t motionBase = pixelBase(*inputs.analyticMotion, pixel);
      const float motionX = inputs.analyticMotion->values[motionBase];
      const float motionY = inputs.analyticMotion->values[motionBase + 1u];
      if (!std::isfinite(motionX) || !std::isfinite(motionY)) {
        continue;
      }
      const uint32_t x = static_cast<uint32_t>(pixel % output.width);
      const uint32_t y = static_cast<uint32_t>(pixel / output.width);
      const double historyUvX =
          (static_cast<double>(x) + 0.5) / output.width + motionX;
      const double historyUvY =
          (static_cast<double>(y) + 0.5) / output.height + motionY;
      const std::optional<double> previousOutputLuma = bilinearLuma(
          *inputs.previousOutput, historyUvX, historyUvY, selected);
      const std::optional<double> previousReferenceLuma = bilinearLuma(
          *inputs.previousReference, historyUvX, historyUvY, selected);
      if (!previousOutputLuma.has_value() ||
          !previousReferenceLuma.has_value()) {
        continue;
      }
      const double outputResidual = luma(output, pixel) - *previousOutputLuma;
      const double referenceResidual =
          luma(reference, pixel) - *previousReferenceLuma;
      temporalErrorSum += std::abs(outputResidual - referenceResidual);
      ++report.temporalSampleCount;
    }
    if (report.temporalSampleCount == 0u) {
      return Result<AutotestQualityOracleReport, std::string>::makeError(
          "quality oracle temporal mask has no valid warped samples");
    }
    report.temporalAvailable = true;
    report.temporalError = temporalErrorSum /
                           static_cast<double>(report.temporalSampleCount) /
                           oracle.lscale;

    auto revealSelection =
        makeSelection(output, inputs.revealMask, "quality oracle reveal");
    if (revealSelection.hasError()) {
      return Result<AutotestQualityOracleReport, std::string>::makeError(
          revealSelection.error());
    }
    double ghostSum = 0.0;
    double recoverySquaredSum = 0.0;
    uint64_t recoveryComponentCount = 0u;
    for (size_t pixel = 0u; pixel < pixelCount; ++pixel) {
      if (revealSelection.value()[pixel] == 0u || !finiteRgb(output, pixel) ||
          !finiteRgb(reference, pixel)) {
        continue;
      }
      ++report.revealPixelCount;
      ghostSum += std::abs(luma(output, pixel) - luma(reference, pixel));
      const size_t outputBase = pixelBase(output, pixel);
      const size_t referenceBase = pixelBase(reference, pixel);
      for (uint32_t channel = 0u; channel < 3u; ++channel) {
        const double error =
            static_cast<double>(output.values[outputBase + channel]) -
            reference.values[referenceBase + channel];
        recoverySquaredSum += error * error;
        ++recoveryComponentCount;
      }
    }
    if (report.revealPixelCount == 0u) {
      return Result<AutotestQualityOracleReport, std::string>::makeError(
          "quality oracle reveal mask has no finite samples");
    }
    report.revealAvailable = true;
    report.ghostEnergy =
        ghostSum / static_cast<double>(report.revealPixelCount) / oracle.lscale;
    report.recoveryRmse =
        std::sqrt(recoverySquaredSum /
                  static_cast<double>(recoveryComponentCount)) /
        oracle.lscale;
  }

  const AutotestQualityOracleBudgets &budgets = oracle.budgets;
  if (report.nonFiniteValueCount > 0u) {
    report.failedThresholds.emplace_back("non_finite_values");
  }
  if (report.normalizedHdrMae > budgets.normalizedMaeMax) {
    report.failedThresholds.emplace_back("normalized_hdr_mae");
  }
  if (report.normalizedHdrRmse > budgets.normalizedRmseMax) {
    report.failedThresholds.emplace_back("normalized_hdr_rmse");
  }
  if (report.lumaSsim < budgets.lumaSsimMin) {
    report.failedThresholds.emplace_back("luma_ssim");
  }
  if (report.darkCollapsePercent > budgets.darkCollapsePercentMax) {
    report.failedThresholds.emplace_back("dark_collapse_percent");
  }
  if (report.darkCollapseMaxComponentPixels >
      budgets.darkCollapseComponentMaxPixels) {
    report.failedThresholds.emplace_back("dark_collapse_component_pixels");
  }
  if (report.relativeLumaEnergyDrift > budgets.relativeLumaEnergyDriftMax) {
    report.failedThresholds.emplace_back("relative_luma_energy_drift");
  }
  if (report.edgeAvailable) {
    if (report.edgeUnresolvedProfileCount > 0u) {
      report.failedThresholds.emplace_back("edge_profile_unresolved");
    }
    if (report.edgeWidthRatio < budgets.edgeWidthRatioMin ||
        report.edgeWidthRatio > budgets.edgeWidthRatioMax) {
      report.failedThresholds.emplace_back("edge_width_ratio");
    }
    if (report.edgeOvershoot > budgets.edgeOvershootMax) {
      report.failedThresholds.emplace_back("edge_overshoot");
    }
    if (report.edgeUndershoot > budgets.edgeUndershootMax) {
      report.failedThresholds.emplace_back("edge_undershoot");
    }
  }
  if (oracle.temporal.has_value()) {
    if (report.temporalError > budgets.temporalErrorMax) {
      report.failedThresholds.emplace_back("temporal_error");
    }
    if (report.ghostEnergy > budgets.ghostEnergyMax) {
      report.failedThresholds.emplace_back("ghost_energy");
    }
    if (report.recoveryRmse > budgets.recoveryRmseMax) {
      report.failedThresholds.emplace_back("recovery_rmse");
    }
  }
  report.status = report.failedThresholds.empty() ? "pass" : "fail";
  report.statusReason = report.failedThresholds.empty()
                            ? "within_quality_budgets"
                            : "quality_oracle_thresholds_failed";
  return Result<AutotestQualityOracleReport, std::string>::makeResult(
      std::move(report));
}

} // namespace nuri::tools::autotest
