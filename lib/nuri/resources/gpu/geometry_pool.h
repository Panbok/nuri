#pragma once
#include "nuri/core/containers/slot_pool.h"
#include "nuri/core/result.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/gpu_types.h"
#include <array>
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
  [[nodiscard]] Result<GeometryAllocationHandle, std::string>
  adoptPrepared(BufferHandle vertexBuffer, size_t vertexBytes,
                uint32_t vertexCount, BufferHandle indexBuffer,
                size_t indexBytes, uint32_t indexCount,
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
  enum class PoolKind : uint8_t { Vertex, Index, Count };
  static constexpr size_t kPoolCount = static_cast<size_t>(PoolKind::Count);
  [[nodiscard]] static constexpr size_t poolIndex(PoolKind kind) noexcept {
    return static_cast<size_t>(kind);
  }
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
    Replacement,
    Retired,
  };
  struct Chunk {
    BufferHandle buffer{};
    size_t sizeBytes = 0;
    size_t freeBytes = 0;
    SubmissionHandle retirementSubmission{};
    ChunkRole role = ChunkRole::Dead;
    bool mutableSuballocations = true;
    std::pmr::vector<Block> freeBlocks;
    explicit Chunk(std::pmr::memory_resource *memory)
        : freeBlocks(ensureMemory(memory)) {}
    void reset() {
      buffer = {};
      sizeBytes = 0;
      freeBytes = 0;
      retirementSubmission = {};
      role = ChunkRole::Dead;
      mutableSuballocations = true;
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
    std::array<SubAllocation, kPoolCount> allocations{};
    std::array<uint32_t, kPoolCount> counts{};
    SubmissionHandle retirementSubmission{};
    bool retirementCaptureFailed = false;
    std::pmr::string debugName;
    explicit AllocationEntry(std::pmr::memory_resource *memory)
        : debugName(ensureMemory(memory)) {}
  };
  struct SnapshotAllocation {
    uint32_t allocationIndex = 0;
    uint32_t allocationGeneration = 0;
    std::array<SubAllocation, kPoolCount> allocations{};
  };
  struct CompactionMove {
    uint32_t allocationIndex = 0;
    uint32_t allocationGeneration = 0;
    std::array<uint32_t, kPoolCount> dstChunkIndices{};
    std::array<size_t, kPoolCount> dstOffsets{};
    std::array<size_t, kPoolCount> dstSizes{};
  };
  struct PreparedReplacementChunk {
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
    PoolKind pool = PoolKind::Vertex;
  };
  struct PreparedCompactionPlan {
    bool worthwhile = false;
    std::array<std::vector<PreparedReplacementChunk>, kPoolCount> chunks;
    std::vector<PreparedCopy> copies;
    std::vector<CompactionMove> moves;
  };
  enum class CompactionJobState : uint8_t {
    Idle,
    Planning,
    MaterializingReplacement,
    ReadyToSubmit,
    WaitingForCopy,
    ReadyToCommit,
  };
  struct CompactionJob {
    CompactionJobState state = CompactionJobState::Idle;
    std::array<std::pmr::vector<ChunkHandle>, kPoolCount> frozenChunks;
    std::array<std::pmr::vector<ChunkHandle>, kPoolCount> replacementChunks;
    std::shared_future<Result<PreparedCompactionPlan, std::string>>
        planningFuture;
    std::optional<PreparedCompactionPlan> preparedPlan;
    std::array<size_t, kPoolCount> nextReplacementChunkIndices{};
    size_t nextCopyIndex = 0;
    SubmissionHandle inFlightSubmission{};
    explicit CompactionJob(std::pmr::memory_resource *memory)
        : frozenChunks{std::pmr::vector<ChunkHandle>(ensureMemory(memory)),
                       std::pmr::vector<ChunkHandle>(ensureMemory(memory))},
          replacementChunks{
              std::pmr::vector<ChunkHandle>(ensureMemory(memory)),
              std::pmr::vector<ChunkHandle>(ensureMemory(memory))} {}
    [[nodiscard]] bool active() const noexcept {
      return state != CompactionJobState::Idle;
    }
    void reset() {
      state = CompactionJobState::Idle;
      for (auto &handles : frozenChunks)
        handles.clear();
      for (auto &handles : replacementChunks)
        handles.clear();
      planningFuture = {};
      preparedPlan.reset();
      nextReplacementChunkIndices = {};
      nextCopyIndex = 0;
      inFlightSubmission = {};
    }
  };
  struct ChunkPool {
    std::pmr::vector<Chunk> chunks;
    SlotPool<UnmaskedNonZeroGenerationPolicy> slots;
    explicit ChunkPool(std::pmr::memory_resource *memory)
        : chunks(ensureMemory(memory)) {}
  };
  static std::pmr::memory_resource *
  ensureMemory(std::pmr::memory_resource *memory) {
    return memory != nullptr ? memory : std::pmr::get_default_resource();
  }
  [[nodiscard]] static bool isValid(ChunkHandle handle) noexcept {
    return handle.generation != 0u;
  }
  [[nodiscard]] Result<ChunkHandle, std::string>
  createChunk(ChunkPool &pool, size_t minimumSize, BufferUsage usage,
              std::string_view debugPrefix, ChunkRole role);
  [[nodiscard]] Result<SubAllocation, std::string>
  allocateFromPool(ChunkPool &pool, size_t sizeBytes, size_t alignment,
                   size_t defaultChunkSize, BufferUsage usage,
                   std::string_view debugPrefix);
  void freeInPool(ChunkPool &pool, const SubAllocation &allocation);
  void reclaimRetiredAllocations();
  void reclaimRetiredChunks(ChunkPool &pool);
  void freezeAllocatableChunks(ChunkPool &pool,
                               std::pmr::vector<ChunkHandle> &frozenHandles);
  void promoteChunks(ChunkPool &pool, std::span<const ChunkHandle> handles);
  void retireChunks(ChunkPool &pool, std::span<const ChunkHandle> handles,
                    SubmissionHandle retirementSubmission);
  void restoreFrozenChunks(ChunkPool &pool,
                           std::span<const ChunkHandle> handles);
  void destroyChunks(ChunkPool &pool, std::span<const ChunkHandle> handles);
  [[nodiscard]] bool shouldStartCompaction() const;
  [[nodiscard]] static size_t poolBytes(const ChunkPool &pool,
                                        ChunkRole role) noexcept;
  [[nodiscard]] static Result<PreparedCompactionPlan, std::string>
  buildPreparedCompactionPlan(std::span<const SnapshotAllocation> snapshot,
                              size_t currentBytes,
                              const GeometryPoolConfig &config);
  [[nodiscard]] Result<bool, std::string> startCompactionPlanning();
  [[nodiscard]] Result<bool, std::string> pollCompactionPlanning();
  [[nodiscard]] Result<bool, std::string> materializeReplacementChunks();
  [[nodiscard]] Result<bool, std::string> submitNextCopyBatch();
  [[nodiscard]] Result<bool, std::string> pollCompactionJob();
  [[nodiscard]] Result<bool, std::string> commitCompactionJob();
  void abortCompactionJob();
  [[nodiscard]] bool isHandleLive(GeometryAllocationHandle handle) const;
  void bumpMutationVersion() noexcept;
  [[nodiscard]] Chunk *findChunk(ChunkPool &pool, ChunkHandle handle);
  [[nodiscard]] const Chunk *findChunk(const ChunkPool &pool,
                                       ChunkHandle handle) const;
  GPUDevice &gpu_;
  GeometryPoolConfig config_{};
  uint64_t currentFrameIndex_ = 0;
  uint64_t lastCompactionStartFrame_ = 0;
  std::pmr::memory_resource *memory_ = nullptr;
  std::array<ChunkPool, kPoolCount> pools_;
  std::pmr::vector<AllocationEntry> allocations_;
  SlotPool<UnmaskedNonZeroGenerationPolicy> allocationSlots_;
  CompactionJob compactionJob_;
  uint64_t mutationVersion_ = 1;
};

} // namespace nuri
