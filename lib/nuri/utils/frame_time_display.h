#pragma once

#include "nuri/core/log.h"

#include <cmath>
#include <cstdint>
#include <optional>

namespace nuri {

struct GpuFrameTimeSample {
  uint64_t sourceFrame = 0u;
  float milliseconds = 0.0f;
};

struct FrameTimeDisplayValues {
  float cpuMilliseconds = 0.0f;
  float gpuMilliseconds = 0.0f;
  bool cpuAvailable = false;
  bool gpuAvailable = false;
};

class FrameTimeDisplaySampler {
public:
  explicit FrameTimeDisplaySampler(double updateIntervalSeconds) noexcept
      : updateIntervalSeconds_(updateIntervalSeconds) {
    NURI_ASSERT(std::isfinite(updateIntervalSeconds) &&
                    updateIntervalSeconds > 0.0,
                "Frame-time display interval must be finite and positive");
  }

  bool tick(double deltaSeconds,
            std::optional<GpuFrameTimeSample> gpuSample) noexcept {
    if (std::isfinite(deltaSeconds) && deltaSeconds > 0.0) {
      intervalSeconds_ += deltaSeconds;
      cpuMillisecondsSum_ += deltaSeconds * 1000.0;
      ++cpuSampleCount_;
    }

    if (gpuSample.has_value() && std::isfinite(gpuSample->milliseconds) &&
        gpuSample->milliseconds >= 0.0f &&
        (!lastGpuSourceFrameAvailable_ ||
         gpuSample->sourceFrame != lastGpuSourceFrame_)) {
      gpuMillisecondsSum_ += gpuSample->milliseconds;
      ++gpuSampleCount_;
      lastGpuSourceFrame_ = gpuSample->sourceFrame;
      lastGpuSourceFrameAvailable_ = true;
    }

    if (intervalSeconds_ < updateIntervalSeconds_) {
      return false;
    }

    if (cpuSampleCount_ != 0u) {
      values_.cpuMilliseconds = static_cast<float>(
          cpuMillisecondsSum_ / static_cast<double>(cpuSampleCount_));
      values_.cpuAvailable = true;
    }
    if (gpuSampleCount_ != 0u) {
      values_.gpuMilliseconds = static_cast<float>(
          gpuMillisecondsSum_ / static_cast<double>(gpuSampleCount_));
      values_.gpuAvailable = true;
    }

    intervalSeconds_ = 0.0;
    cpuMillisecondsSum_ = 0.0;
    gpuMillisecondsSum_ = 0.0;
    cpuSampleCount_ = 0u;
    gpuSampleCount_ = 0u;
    return true;
  }

  [[nodiscard]] const FrameTimeDisplayValues &values() const noexcept {
    return values_;
  }

private:
  double updateIntervalSeconds_ = 0.25;
  double intervalSeconds_ = 0.0;
  double cpuMillisecondsSum_ = 0.0;
  double gpuMillisecondsSum_ = 0.0;
  uint64_t lastGpuSourceFrame_ = 0u;
  uint32_t cpuSampleCount_ = 0u;
  uint32_t gpuSampleCount_ = 0u;
  bool lastGpuSourceFrameAvailable_ = false;
  FrameTimeDisplayValues values_{};
};

} // namespace nuri
