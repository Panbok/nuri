#pragma once

namespace nuri {

class FPSCounter {
public:
  explicit FPSCounter(float avgInterval);
  virtual ~FPSCounter() = default;

  bool tick(double deltaTime, bool frameRendered = true);
  inline float getFPS() const { return currentFPS_; }

private:
  float avgInterval_ = 0.5f;
  std::uint32_t frameCount_ = 0;
  double timeAccumulator_ = 0.0;
  float currentFPS_ = 0.0f;
};

} // namespace nuri