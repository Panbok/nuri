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
