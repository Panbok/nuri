#include "fsp_counter.h"
#include "nuri/core/log.h"

namespace nuri {

FPSCounter::FPSCounter(float avgInterval) : avgInterval_(avgInterval) {
  NURI_ASSERT(avgInterval > 0.0f, "Average interval must be greater than 0.0f");
}

bool FPSCounter::tick(double deltaTime, bool frameRendered) {
  if (frameRendered)
    frameCount_++;

  timeAccumulator_ += deltaTime;

  if (timeAccumulator_ > avgInterval_) {
    currentFPS_ = static_cast<float>(frameCount_) / timeAccumulator_;
    timeAccumulator_ = 0.0;
    frameCount_ = 0;
    return true;
  }

  return false;
}

} // namespace nuri