#pragma once

#include "nuri/defines.h"
#include "nuri/gfx/gpu_types.h"

#include <cstdint>
#include <memory_resource>
#include <string>

namespace nuri {

enum class SimulationBufferUsage : uint8_t {
  None = 0,
  Params = 1,
  State = 2,
  Constraints = 3,
  Scratch = 4,
  Writeback = 5,
  Debug = 6,
  DispatchMetadata = 7,
};

struct NURI_API SimulationBufferSlice {
  BufferHandle buffer{};
  uint64_t offsetBytes = 0u;
  uint64_t sizeBytes = 0u;
  uint64_t version = 0u;
};

struct NURI_API SimulationPersistentBufferDesc {
  explicit SimulationPersistentBufferDesc(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : debugName(memory) {}

  std::pmr::string debugName;
  SimulationBufferUsage usage = SimulationBufferUsage::None;
  size_t sizeBytes = 0u;
  bool hostVisible = false;
};

struct NURI_API SimulationTransientBufferDesc {
  explicit SimulationTransientBufferDesc(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : debugName(memory) {}

  std::pmr::string debugName;
  SimulationBufferUsage usage = SimulationBufferUsage::None;
  size_t sizeBytes = 0u;
  uint32_t alignmentBytes = 1u;
};

struct NURI_API SimulationFrameGpuResources {
  SimulationBufferSlice paramUpload{};
  SimulationBufferSlice transientScratch{};
  SimulationBufferSlice writeback{};
  SimulationBufferSlice dispatchMetadata{};
  uint64_t frameGeneration = 0u;
  uint64_t resourceVersion = 0u;
};

} // namespace nuri
