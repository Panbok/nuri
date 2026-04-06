#pragma once

#include "nuri/core/containers/slot_pool.h"
#include "nuri/core/result.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/gpu_types.h"

#include <cstddef>
#include <cstdint>
#include <future>
#include <memory_resource>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace nuri {

class NURI_API GeometryPool final {
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
  [[nodiscard]] uint64_t mutationVersion() const noexcept {
    return mutationVersion_;
  }

private:
  static constexpr size_t kVertexAlignment = 16;
  static constexpr size_t kIndexAlignment = 4;

  struct ChunkHandle {
    uint32_t index = 0;
    uint32_t generation = 0;
  };

  struct Block {
    size_t offset = 0;
    size_t size = 0;
  };

  enum class ChunkRole : uint8_t {
    Dead,
    ActiveAllocatable,
    FrozenSource,
    Shadow,
    Retired,
  };

  struct Chunk {
    BufferHandle buffer{};
    size_t sizeBytes = 0;
    size_t freeBytes = 0;
    uint64_t retireFrame = 0;
    ChunkRole role = ChunkRole::Dead;
    std::pmr::vector<Block> freeBlocks;

    explicit Chunk(std::pmr::memory_resource *memory)
        : freeBlocks(ensureMemory(memory)) {}

    void reset() {
      buffer = {};
      sizeBytes = 0;
      freeBytes = 0;
      retireFrame = 0;
      role = ChunkRole::Dead;
      freeBlocks.clear();
    }
  };

  struct SubAllocation {
    ChunkHandle chunk{};
    size_t offset = 0;
    size_t size = 0;
  };

  struct AllocationEntry {
    enum class State : uint8_t { Dead, Live, PendingFree };

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

  struct SnapshotAllocation {
    uint32_t allocationIndex = 0;
    uint32_t allocationGeneration = 0;
    SubAllocation vertex{};
    SubAllocation index{};
  };

  struct CompactionMove {
    uint32_t allocationIndex = 0;
    uint32_t allocationGeneration = 0;
    uint32_t dstVertexChunkIndex = 0;
    size_t dstVertexOffset = 0;
    size_t dstVertexSize = 0;
    uint32_t dstIndexChunkIndex = 0;
    size_t dstIndexOffset = 0;
    size_t dstIndexSize = 0;
  };

  struct PreparedShadowChunk {
    size_t sizeBytes = 0;
    size_t usedBytes = 0;
  };

  struct PreparedCopy {
    size_t snapshotIndex = 0;
    ChunkHandle srcChunk{};
    size_t srcOffset = 0;
    size_t size = 0;
    uint32_t dstChunkIndex = 0;
    size_t dstOffset = 0;
    bool isVertex = true;
  };

  struct PreparedCompactionPlan {
    bool worthwhile = false;
    std::vector<PreparedShadowChunk> vertexChunks;
    std::vector<PreparedShadowChunk> indexChunks;
    std::vector<PreparedCopy> copies;
    std::vector<CompactionMove> moves;
  };

  enum class CompactionJobState : uint8_t {
    Idle,
    Planning,
    MaterializingShadow,
    ReadyToSubmit,
    WaitingForCopy,
    ReadyToCommit,
  };

  struct CompactionJob {
    CompactionJobState state = CompactionJobState::Idle;
    std::pmr::vector<ChunkHandle> frozenVertexChunks;
    std::pmr::vector<ChunkHandle> frozenIndexChunks;
    std::pmr::vector<ChunkHandle> shadowVertexChunks;
    std::pmr::vector<ChunkHandle> shadowIndexChunks;
    std::shared_future<Result<PreparedCompactionPlan, std::string>>
        planningFuture;
    std::optional<PreparedCompactionPlan> preparedPlan;
    size_t nextShadowVertexChunkIndex = 0;
    size_t nextShadowIndexChunkIndex = 0;
    size_t nextCopyIndex = 0;
    SubmissionHandle inFlightSubmission{};

    explicit CompactionJob(std::pmr::memory_resource *memory)
        : frozenVertexChunks(ensureMemory(memory)),
          frozenIndexChunks(ensureMemory(memory)),
          shadowVertexChunks(ensureMemory(memory)),
          shadowIndexChunks(ensureMemory(memory)) {}

    [[nodiscard]] bool active() const noexcept {
      return state != CompactionJobState::Idle;
    }

    void reset() {
      state = CompactionJobState::Idle;
      frozenVertexChunks.clear();
      frozenIndexChunks.clear();
      shadowVertexChunks.clear();
      shadowIndexChunks.clear();
      planningFuture = {};
      preparedPlan.reset();
      nextShadowVertexChunkIndex = 0;
      nextShadowIndexChunkIndex = 0;
      nextCopyIndex = 0;
      inFlightSubmission = {};
    }
  };

  static std::pmr::memory_resource *
  ensureMemory(std::pmr::memory_resource *memory) {
    return memory != nullptr ? memory : std::pmr::get_default_resource();
  }

  [[nodiscard]] static bool isValid(ChunkHandle handle) noexcept {
    return handle.generation != 0u;
  }

  [[nodiscard]] Result<ChunkHandle, std::string>
  createChunk(std::pmr::vector<Chunk> &chunks,
              SlotPool<UnmaskedNonZeroGenerationPolicy> &chunkSlots,
              size_t minimumSize, BufferUsage usage,
              std::string_view debugPrefix, ChunkRole role);
  [[nodiscard]] Result<SubAllocation, std::string>
  allocateFromPool(std::pmr::vector<Chunk> &chunks,
                   SlotPool<UnmaskedNonZeroGenerationPolicy> &chunkSlots,
                   size_t sizeBytes, size_t alignment, size_t defaultChunkSize,
                   BufferUsage usage, std::string_view debugPrefix);
  void freeInPool(std::pmr::vector<Chunk> &chunks,
                  SlotPool<UnmaskedNonZeroGenerationPolicy> &chunkSlots,
                  const SubAllocation &allocation);

  [[nodiscard]] uint64_t reclaimLagFrames() const;
  void reclaimRetiredAllocations();
  void
  reclaimRetiredChunks(std::pmr::vector<Chunk> &chunks,
                       SlotPool<UnmaskedNonZeroGenerationPolicy> &chunkSlots);
  void
  freezeAllocatableChunks(std::pmr::vector<Chunk> &chunks,
                          SlotPool<UnmaskedNonZeroGenerationPolicy> &chunkSlots,
                          std::pmr::vector<ChunkHandle> &frozenHandles);
  void promoteChunks(std::pmr::vector<Chunk> &chunks,
                     SlotPool<UnmaskedNonZeroGenerationPolicy> &chunkSlots,
                     std::span<const ChunkHandle> handles);
  void retireChunks(std::pmr::vector<Chunk> &chunks,
                    SlotPool<UnmaskedNonZeroGenerationPolicy> &chunkSlots,
                    std::span<const ChunkHandle> handles, uint64_t retireFrame);
  void
  restoreFrozenChunks(std::pmr::vector<Chunk> &chunks,
                      SlotPool<UnmaskedNonZeroGenerationPolicy> &chunkSlots,
                      std::span<const ChunkHandle> handles);
  void destroyChunks(std::pmr::vector<Chunk> &chunks,
                     SlotPool<UnmaskedNonZeroGenerationPolicy> &chunkSlots,
                     std::span<const ChunkHandle> handles);

  [[nodiscard]] bool shouldStartCompaction() const;
  [[nodiscard]] static size_t poolBytes(const std::pmr::vector<Chunk> &chunks,
                                        ChunkRole role) noexcept;
  [[nodiscard]] static Result<PreparedCompactionPlan, std::string>
  buildPreparedCompactionPlan(std::span<const SnapshotAllocation> snapshot,
                              size_t currentBytes,
                              const GeometryPoolConfig &config);
  [[nodiscard]] Result<bool, std::string> startCompactionPlanning();
  [[nodiscard]] Result<bool, std::string> pollCompactionPlanning();
  [[nodiscard]] Result<bool, std::string> materializeShadowChunks();
  [[nodiscard]] Result<bool, std::string> submitNextCopyBatch();
  [[nodiscard]] Result<bool, std::string> pollCompactionJob();
  [[nodiscard]] Result<bool, std::string> commitCompactionJob();
  void abortCompactionJob();

  [[nodiscard]] bool isHandleLive(GeometryAllocationHandle handle) const;
  void bumpMutationVersion() noexcept;

  [[nodiscard]] Chunk *
  findChunk(std::pmr::vector<Chunk> &chunks,
            const SlotPool<UnmaskedNonZeroGenerationPolicy> &chunkSlots,
            ChunkHandle handle);
  [[nodiscard]] const Chunk *
  findChunk(const std::pmr::vector<Chunk> &chunks,
            const SlotPool<UnmaskedNonZeroGenerationPolicy> &chunkSlots,
            ChunkHandle handle) const;

  GPUDevice &gpu_;
  GeometryPoolConfig config_{};
  uint64_t currentFrameIndex_ = 0;
  uint64_t lastCompactionStartFrame_ = 0;

  std::pmr::memory_resource *memory_ = nullptr;

  std::pmr::vector<Chunk> vertexChunks_;
  std::pmr::vector<Chunk> indexChunks_;
  SlotPool<UnmaskedNonZeroGenerationPolicy> vertexChunkSlots_;
  SlotPool<UnmaskedNonZeroGenerationPolicy> indexChunkSlots_;

  std::pmr::vector<AllocationEntry> allocations_;
  SlotPool<UnmaskedNonZeroGenerationPolicy> allocationSlots_;
  CompactionJob compactionJob_;
  uint64_t mutationVersion_ = 1;
};

} // namespace nuri
