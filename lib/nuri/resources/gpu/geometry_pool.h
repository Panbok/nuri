#pragma once

#include "nuri/core/result.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/gpu_types.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory_resource>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace nuri {

class GeometryPool final {
public:
  explicit GeometryPool(
      GPUDevice &gpu, GeometryPoolConfig config = {},
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~GeometryPool();

  GeometryPool(const GeometryPool &) = delete;
  GeometryPool &operator=(const GeometryPool &) = delete;
  GeometryPool(GeometryPool &&) = delete;
  GeometryPool &operator=(GeometryPool &&) = delete;

  [[nodiscard]] Result<bool, std::string> beginFrame(uint64_t frameIndex);
  [[nodiscard]] Result<GeometryAllocationHandle, std::string>
  allocate(std::span<const std::byte> vertexBytes, uint32_t vertexCount,
           std::span<const std::byte> indexBytes, uint32_t indexCount,
           std::string_view debugName);
  void release(GeometryAllocationHandle handle);
  [[nodiscard]] bool resolve(GeometryAllocationHandle handle,
                             GeometryAllocationView &out) const;

private:
  static constexpr uint32_t kInvalidChunkIndex = UINT32_MAX;
  static constexpr size_t kVertexAlignment = 16;
  static constexpr size_t kIndexAlignment = 4;

  static std::pmr::memory_resource *
  ensureMemory(std::pmr::memory_resource *memory) {
    return memory != nullptr ? memory : std::pmr::get_default_resource();
  }

  struct Block {
    size_t offset = 0;
    size_t size = 0;
  };

  struct Chunk {
    BufferHandle buffer{};
    size_t sizeBytes = 0;
    size_t freeBytes = 0;
    std::pmr::vector<Block> freeBlocks;

    explicit Chunk(std::pmr::memory_resource *memory)
        : freeBlocks(ensureMemory(memory)) {}
  };

  struct SubAllocation {
    uint32_t chunkIndex = kInvalidChunkIndex;
    size_t offset = 0;
    size_t size = 0;
  };

  struct AllocationEntry {
    enum class State : uint8_t { Dead, Live, PendingFree };

    uint32_t generation = 0;
    State state = State::Dead;
    SubAllocation vertex{};
    SubAllocation index{};
    uint32_t vertexCount = 0;
    uint32_t indexCount = 0;
    uint64_t retireFrame = 0;
    std::pmr::string debugName;

    explicit AllocationEntry(std::pmr::memory_resource *memory)
        : debugName(ensureMemory(memory)) {}
  };

  struct RetiredChunk {
    BufferHandle buffer{};
    uint64_t retireFrame = 0;
  };

  struct PoolCompactionMove {
    uint32_t allocationIndex = 0;
    bool isVertex = true;
    SubAllocation dst{};
  };

  struct AllocationTarget {
    uint32_t chunkIndex = kInvalidChunkIndex;
    size_t offset = 0;
  };

  struct PoolCompactionPlan {
    std::pmr::vector<Chunk> newChunks;
    std::pmr::vector<BufferCopyRegion> copyRegions;
    std::pmr::vector<PoolCompactionMove> moves;

    explicit PoolCompactionPlan(std::pmr::memory_resource *memory)
        : newChunks(ensureMemory(memory)), copyRegions(ensureMemory(memory)),
          moves(ensureMemory(memory)) {}
  };

  [[nodiscard]] static size_t alignUp(size_t value, size_t alignment);

  [[nodiscard]] Result<bool, std::string>
  createChunk(std::pmr::vector<Chunk> &chunks, size_t minimumSize,
              BufferUsage usage, std::string_view debugPrefix);
  [[nodiscard]] Result<SubAllocation, std::string>
  allocateFromPool(std::pmr::vector<Chunk> &chunks, size_t sizeBytes,
                   size_t alignment, size_t defaultChunkSize, BufferUsage usage,
                   std::string_view debugPrefix);
  static void freeInPool(std::pmr::vector<Chunk> &chunks,
                         const SubAllocation &allocation);

  [[nodiscard]] uint64_t reclaimLagFrames() const;
  void reclaimRetiredAllocations();
  void reclaimRetiredChunks(std::pmr::deque<RetiredChunk> &retiredChunks);

  [[nodiscard]] bool
  shouldCompactPool(const std::pmr::vector<Chunk> &chunks) const;
  [[nodiscard]] Result<PoolCompactionPlan, std::string>
  buildCompactionPlan(std::pmr::vector<Chunk> &chunks, bool forVertexPool);
  [[nodiscard]] Result<bool, std::string>
  applyCompactionPlan(std::pmr::vector<Chunk> &chunks,
                      std::pmr::deque<RetiredChunk> &retiredChunks,
                      PoolCompactionPlan &plan);
  [[nodiscard]] Result<bool, std::string> compactIfNeeded();

  [[nodiscard]] bool isHandleLive(GeometryAllocationHandle handle) const;

  GPUDevice &gpu_;
  GeometryPoolConfig config_{};
  uint64_t currentFrameIndex_ = 0;

  std::pmr::memory_resource *memory_ = nullptr;

  std::pmr::vector<Chunk> vertexChunks_;
  std::pmr::vector<Chunk> indexChunks_;
  std::pmr::vector<AllocationEntry> allocations_;
  std::pmr::vector<uint32_t> freeAllocationIndices_;
  std::pmr::deque<RetiredChunk> retiredVertexChunks_;
  std::pmr::deque<RetiredChunk> retiredIndexChunks_;
};

} // namespace nuri
