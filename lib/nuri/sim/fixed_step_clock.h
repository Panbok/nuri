#pragma once

#include "nuri/defines.h"

#include <cstdint>

namespace nuri {

struct NURI_API FixedStepAdvanceResult {
  uint32_t stepCount = 0u;
  double consumedSeconds = 0.0;
  double remainingAccumulatorSeconds = 0.0;
  bool clamped = false;
};

class NURI_API FixedStepClock {
public:
  void reset() noexcept { accumulatorSeconds_ = 0.0; }

  // advance(frameDeltaSeconds, fixedDeltaSeconds, maxStepsPerFrame,
  // maxAccumulatedSeconds, allowFrameDropping) requires frameDeltaSeconds >= 0,
  // maxAccumulatedSeconds >= 0, and fixedDeltaSeconds > 0. Invalid or
  // non-finite frame/fixed deltas return a zero-step result without consuming
  // accumulated time. maxStepsPerFrame == 0 is treated the same as 1, and
  // allowFrameDropping only discards extra accumulated time after that cap is
  // reached for the current frame.
  [[nodiscard]] FixedStepAdvanceResult
  advance(double frameDeltaSeconds, double fixedDeltaSeconds,
          uint32_t maxStepsPerFrame, double maxAccumulatedSeconds,
          bool allowFrameDropping) noexcept;

  [[nodiscard]] double accumulatorSeconds() const noexcept {
    return accumulatorSeconds_;
  }

private:
  double accumulatorSeconds_ = 0.0;
};

} // namespace nuri
