#pragma once

#include "nuri/defines.h"
#include "nuri/gfx/sim/simulation_gpu_types.h"

#include <cstdint>

namespace nuri {

// Per-frame GPU-side simulation state published by SceneRuntimeHost.
// Lifecycle: reset() -> beginFrame(frameIndex) -> publishFrameResources(...).
// Not thread-safe; call from the render/runtime thread that owns the host.
// resourceVersion() increments only when the published buffer slices change.
class NURI_API SimulationGpuContext {
public:
  SimulationGpuContext() = default;
  ~SimulationGpuContext() = default;
  SimulationGpuContext(const SimulationGpuContext &) = delete;
  SimulationGpuContext &operator=(const SimulationGpuContext &) = delete;
  SimulationGpuContext(SimulationGpuContext &&) noexcept = default;
  SimulationGpuContext &operator=(SimulationGpuContext &&) noexcept = default;

  // Clears the current frame index, published resources, and version state.
  // Safe to call at any time on the owning thread. Never throws.
  void reset() noexcept;

  // Starts a new frame and invalidates the previous frame's published view.
  // Call before publishFrameResources() for the target frame. Never throws.
  void beginFrame(uint64_t frameIndex) noexcept;

  // Publishes the current frame resources. Increments resourceVersion() only
  // when the buffer slices differ from the last published state. Never throws.
  void
  publishFrameResources(const SimulationFrameGpuResources &resources) noexcept;

  [[nodiscard]] uint64_t frameIndex() const noexcept { return frameIndex_; }
  [[nodiscard]] uint64_t resourceVersion() const noexcept {
    return resourceVersion_;
  }
  [[nodiscard]] const SimulationFrameGpuResources &
  frameResources() const noexcept {
    return frameResources_;
  }

private:
  uint64_t frameIndex_ = 0u;
  uint64_t resourceVersion_ = 0u;
  SimulationFrameGpuResources frameResources_{};
};

} // namespace nuri
