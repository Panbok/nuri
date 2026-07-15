#pragma once

#include "nuri/core/result.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/resources/gpu/buffer.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace nuri {

struct DynamicBufferSlot {
  std::unique_ptr<Buffer> buffer;
  size_t capacityBytes = 0u;
};

[[nodiscard]] constexpr size_t
nextDynamicBufferCapacity(size_t currentCapacity, size_t requiredCapacity) {
  constexpr size_t kMinimumCapacity = 256u;
  constexpr size_t kCapacityAlignment = 256u;
  if (currentCapacity >= requiredCapacity) {
    return currentCapacity;
  }

  const size_t growthBase = std::max(currentCapacity, kMinimumCapacity);
  const size_t geometricCapacity = growthBase + growthBase / 2u;
  const size_t targetCapacity =
      std::max(requiredCapacity, geometricCapacity);
  return ((targetCapacity + kCapacityAlignment - 1u) / kCapacityAlignment) *
         kCapacityAlignment;
}

// Allocates first, publishes the new handle second, and retires the old native
// resource last. GPUDevice owns completion tracking, so callers never need to
// stall the device merely to grow a whole-buffer allocation.
[[nodiscard]] inline Result<bool, std::string>
ensureDynamicBufferCapacity(GPUDevice &gpu, std::unique_ptr<Buffer> &buffer,
                            size_t &capacityBytes, BufferDesc desc,
                            std::string_view debugName) {
  if (buffer && buffer->valid() && capacityBytes >= desc.size) {
    return Result<bool, std::string>::makeResult(false);
  }

  desc.size = nextDynamicBufferCapacity(capacityBytes, desc.size);
  auto replacementResult = Buffer::create(gpu, desc, debugName);
  if (replacementResult.hasError()) {
    return Result<bool, std::string>::makeError(replacementResult.error());
  }

  std::unique_ptr<Buffer> previous = std::move(buffer);
  buffer = std::move(replacementResult.value());
  capacityBytes = desc.size;
  previous.reset();
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] inline Result<bool, std::string>
ensureDynamicBufferCapacity(GPUDevice &gpu, DynamicBufferSlot &slot,
                            BufferDesc desc, std::string_view debugName) {
  return ensureDynamicBufferCapacity(gpu, slot.buffer, slot.capacityBytes, desc,
                                     debugName);
}

inline void retireDynamicBuffer(GPUDevice &, std::unique_ptr<Buffer> &buffer,
                                size_t &capacityBytes) {
  buffer.reset();
  capacityBytes = 0u;
}

inline void retireDynamicBuffer(GPUDevice &gpu, DynamicBufferSlot &slot) {
  retireDynamicBuffer(gpu, slot.buffer, slot.capacityBytes);
}

inline void retireDynamicBufferRing(GPUDevice &gpu,
                                    std::span<DynamicBufferSlot> ring) {
  for (DynamicBufferSlot &slot : ring) {
    retireDynamicBuffer(gpu, slot);
  }
}

} // namespace nuri
