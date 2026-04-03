#include "nuri/pch.h"

#include "nuri/sim/fixed_step_clock.h"

#include <algorithm>
#include <cmath>

namespace nuri {

FixedStepAdvanceResult
FixedStepClock::advance(double frameDeltaSeconds, double fixedDeltaSeconds,
                        uint32_t maxStepsPerFrame, double maxAccumulatedSeconds,
                        bool allowFrameDropping) noexcept {
  FixedStepAdvanceResult result{};
  if (!std::isfinite(frameDeltaSeconds) || frameDeltaSeconds <= 0.0 ||
      !std::isfinite(fixedDeltaSeconds) || fixedDeltaSeconds <= 0.0) {
    result.remainingAccumulatorSeconds = accumulatorSeconds_;
    return result;
  }

  accumulatorSeconds_ += frameDeltaSeconds;
  if (maxAccumulatedSeconds > 0.0 &&
      accumulatorSeconds_ > maxAccumulatedSeconds) {
    accumulatorSeconds_ = maxAccumulatedSeconds;
    result.clamped = true;
  }

  const uint32_t maxSteps = std::max(1u, maxStepsPerFrame);
  while (accumulatorSeconds_ + 1.0e-12 >= fixedDeltaSeconds &&
         result.stepCount < maxSteps) {
    accumulatorSeconds_ -= fixedDeltaSeconds;
    ++result.stepCount;
    result.consumedSeconds += fixedDeltaSeconds;
  }

  if (allowFrameDropping && accumulatorSeconds_ >= fixedDeltaSeconds &&
      result.stepCount == maxSteps) {
    const double remainder = std::fmod(accumulatorSeconds_, fixedDeltaSeconds);
    accumulatorSeconds_ = remainder;
    result.clamped = true;
  }

  result.remainingAccumulatorSeconds = accumulatorSeconds_;
  return result;
}

} // namespace nuri
