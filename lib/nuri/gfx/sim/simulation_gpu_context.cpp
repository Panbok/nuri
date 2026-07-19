#include "nuri/gfx/sim/simulation_gpu_context.h"
#include "nuri/pch.h"
namespace nuri {
namespace {
[[nodiscard]] bool bufferSliceEqual(const SimulationBufferSlice &lhs,
                                    const SimulationBufferSlice &rhs) noexcept {
  return lhs.buffer.index == rhs.buffer.index &&
         lhs.buffer.generation == rhs.buffer.generation &&
         lhs.offsetBytes == rhs.offsetBytes && lhs.sizeBytes == rhs.sizeBytes &&
         lhs.version == rhs.version;
}
[[nodiscard]] bool
frameResourcesEqual(const SimulationFrameGpuResources &lhs,
                    const SimulationFrameGpuResources &rhs) noexcept {
  return bufferSliceEqual(lhs.paramUpload, rhs.paramUpload) &&
         bufferSliceEqual(lhs.transientScratch, rhs.transientScratch) &&
         bufferSliceEqual(lhs.writeback, rhs.writeback) &&
         bufferSliceEqual(lhs.dispatchMetadata, rhs.dispatchMetadata);
}
} // namespace

void SimulationGpuContext::reset() noexcept {
  frameIndex_ = 0u;
  resourceVersion_ = 0u;
  frameResources_ = {};
}

void SimulationGpuContext::beginFrame(uint64_t frameIndex) noexcept {
  frameIndex_ = frameIndex;
  frameResources_ = {};
}

void SimulationGpuContext::publishFrameResources(
    const SimulationFrameGpuResources &resources) noexcept {
  if (frameResourcesEqual(frameResources_, resources)) {
    return;
  }
  ++resourceVersion_;
  frameResources_ = resources;
  frameResources_.resourceVersion = resourceVersion_;
  frameResources_.frameGeneration = frameIndex_;
}

} // namespace nuri
