#pragma once
#include "nuri/core/result.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/resources/gpu/buffer.h"
#include <algorithm>
#include <cstddef>
#include <memory>
#include <memory_resource>
#include <span>
#include <string>
#include <string_view>
#include <vector>
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
  const size_t targetCapacity = std::max(requiredCapacity, geometricCapacity);
  return ((targetCapacity + kCapacityAlignment - 1u) / kCapacityAlignment) *
         kCapacityAlignment;
}

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

inline void retireDynamicBuffer(std::unique_ptr<Buffer> &buffer,
                                size_t &capacityBytes) {
  buffer.reset();
  capacityBytes = 0u;
}

inline void retireDynamicBuffer(DynamicBufferSlot &slot) {
  retireDynamicBuffer(slot.buffer, slot.capacityBytes);
}

inline void retireDynamicBufferRing(std::span<DynamicBufferSlot> ring) {
  for (DynamicBufferSlot &slot : ring) {
    retireDynamicBuffer(slot);
  }
}

inline uint32_t growDynamicBufferRings(
    uint32_t count,
    std::span<std::pmr::vector<DynamicBufferSlot> *const> rings) {
  count = std::max(count, 1u);
  for (auto *ring : rings) {
    ring->resize(std::max(ring->size(), static_cast<size_t>(count)));
  }
  return count;
}

inline bool
resizeDynamicBufferRings(uint32_t count,
                         std::span<std::pmr::vector<DynamicBufferSlot>> rings) {
  count = std::max(count, 1u);
  if (std::ranges::all_of(
          rings, [count](const auto &ring) { return ring.size() == count; })) {
    return false;
  }
  for (auto &ring : rings) {
    retireDynamicBufferRing(ring);
    ring.clear();
    ring.resize(count);
  }
  return true;
}

template <typename OnReplacement>
[[nodiscard]] Result<bool, std::string> ensureDynamicBufferRingCapacity(
    GPUDevice &gpu, std::span<DynamicBufferSlot> ring, BufferDesc desc,
    std::string_view debugName, OnReplacement &&onReplacement) {
  bool replaced = false;
  for (size_t i = 0; i < ring.size(); ++i) {
    auto result = ensureDynamicBufferCapacity(gpu, ring[i], desc, debugName);
    if (result.hasError()) {
      return result;
    }
    if (result.value()) {
      replaced = true;
      onReplacement(i);
    }
  }
  return Result<bool, std::string>::makeResult(replaced);
}

[[nodiscard]] inline Result<bool, std::string>
ensureDynamicBufferRingCapacity(GPUDevice &gpu,
                                std::span<DynamicBufferSlot> ring,
                                BufferDesc desc, std::string_view debugName) {
  return ensureDynamicBufferRingCapacity(gpu, ring, desc, debugName,
                                         [](size_t) {});
}

} // namespace nuri
