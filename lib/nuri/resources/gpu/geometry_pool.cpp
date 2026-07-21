#include "nuri/resources/gpu/geometry_pool.h"
#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/gfx/gpu_descriptors.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/math/utils.h"
#include "nuri/pch.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <numeric>
#include <optional>
namespace nuri {

GeometryPool::GeometryPool(GPUDevice &gpu, GeometryPoolConfig config,
                           std::pmr::memory_resource *memory)
    : gpu_(gpu), accelerationStructureBuildInputEnabled_(
                     gpu.getDeviceCaps().rayTracing.accelerationStructure),
      config_(config), memory_(ensureMemory(memory)),
      pools_{ChunkPool(memory_), ChunkPool(memory_)}, allocations_(memory_),
      compactionJob_(memory_) {}

GeometryPool::~GeometryPool() {
  for (const ChunkPool &pool : pools_) {
    for (const Chunk &chunk : pool.chunks) {
      if (nuri::isValid(chunk.buffer)) {
        gpu_.destroyBuffer(chunk.buffer);
      }
    }
  }
}

Result<GeometryPool::ChunkHandle, std::string>
GeometryPool::createChunk(ChunkPool &pool, size_t minimumSize,
                          BufferUsage usage, std::string_view debugPrefix,
                          ChunkRole role) {
  const size_t requestedSize = std::max<size_t>(minimumSize, 1u);
  if (accelerationStructureBuildInputEnabled_) {
    usage = usage | BufferUsage::AccelerationStructureBuildInput;
  }
  const BufferDesc desc{
      .usage = usage,
      .storage = Storage::Device,
      .size = requestedSize,
  };
  const SlotReservation slot = pool.slots.acquire();
  if (slot.appended) {
    pool.chunks.emplace_back(memory_);
  } else {
    pool.chunks[slot.index].reset();
  }
  const std::string debugName =
      std::string(debugPrefix) + "_" + std::to_string(slot.index);
  auto bufferResult = gpu_.createBuffer(desc, debugName);
  if (bufferResult.hasError()) {
    pool.slots.release(slot.index);
    pool.chunks[slot.index].reset();
    return Result<ChunkHandle, std::string>::makeError(bufferResult.error());
  }
  Chunk &chunk = pool.chunks[slot.index];
  chunk.buffer = bufferResult.value();
  chunk.sizeBytes = requestedSize;
  chunk.freeBytes = requestedSize;
  chunk.retirementSubmission = {};
  chunk.role = role;
  chunk.freeBlocks.clear();
  chunk.freeBlocks.emplace_back(Block{.offset = 0, .size = requestedSize});
  return Result<ChunkHandle, std::string>::makeResult(
      ChunkHandle{.index = slot.index, .generation = slot.generation});
}

Result<GeometryPool::SubAllocation, std::string> GeometryPool::allocateFromPool(
    ChunkPool &pool, size_t sizeBytes, size_t alignment,
    size_t defaultChunkSize, BufferUsage usage, std::string_view debugPrefix) {
  if (sizeBytes == 0u) {
    return Result<SubAllocation, std::string>::makeError(
        "GeometryPool::allocateFromPool: allocation size is zero");
  }
  const auto tryAllocateInChunk =
      [sizeBytes, alignment](
          Chunk &chunk, ChunkHandle handle) -> std::optional<SubAllocation> {
    if (chunk.role != ChunkRole::ActiveAllocatable ||
        !chunk.mutableSuballocations) {
      return std::nullopt;
    }
    for (size_t blockIndex = 0; blockIndex < chunk.freeBlocks.size();
         ++blockIndex) {
      Block &block = chunk.freeBlocks[blockIndex];
      const size_t alignedOffset = alignUp(block.offset, alignment);
      const size_t padding = alignedOffset - block.offset;
      if (padding > block.size || sizeBytes > block.size - padding) {
        continue;
      }
      const size_t totalConsumed = padding + sizeBytes;
      const size_t remaining = block.size - totalConsumed;
      std::array<Block, 2> replacement{};
      size_t replacementCount = 0;
      if (padding > 0u) {
        replacement[replacementCount++] =
            Block{.offset = block.offset, .size = padding};
      }
      if (remaining > 0u) {
        replacement[replacementCount++] = Block{
            .offset = alignedOffset + sizeBytes,
            .size = remaining,
        };
      }
      chunk.freeBlocks.erase(chunk.freeBlocks.begin() +
                             static_cast<std::ptrdiff_t>(blockIndex));
      for (size_t replaceIndex = 0; replaceIndex < replacementCount;
           ++replaceIndex) {
        chunk.freeBlocks.insert(
            chunk.freeBlocks.begin() +
                static_cast<std::ptrdiff_t>(blockIndex + replaceIndex),
            replacement[replaceIndex]);
      }
      chunk.freeBytes -= sizeBytes;
      return SubAllocation{
          .chunk = handle,
          .offset = alignedOffset,
          .size = sizeBytes,
      };
    }
    return std::nullopt;
  };
  for (uint32_t chunkIndex = 0; chunkIndex < pool.chunks.size(); ++chunkIndex) {
    Chunk &chunk = pool.chunks[chunkIndex];
    if (chunk.role != ChunkRole::ActiveAllocatable ||
        !chunk.mutableSuballocations) {
      continue;
    }
    const ChunkHandle handle{
        .index = chunkIndex,
        .generation = pool.slots.generation(chunkIndex),
    };
    if (auto allocation = tryAllocateInChunk(chunk, handle)) {
      return Result<SubAllocation, std::string>::makeResult(*allocation);
    }
  }
  auto createResult =
      createChunk(pool, std::max(defaultChunkSize, sizeBytes), usage,
                  debugPrefix, ChunkRole::ActiveAllocatable);
  if (createResult.hasError()) {
    return Result<SubAllocation, std::string>::makeError(createResult.error());
  }
  const ChunkHandle newHandle = createResult.value();
  Chunk *newChunk = findChunk(pool, newHandle);
  if (auto allocation = tryAllocateInChunk(*newChunk, newHandle)) {
    return Result<SubAllocation, std::string>::makeResult(*allocation);
  }
  return Result<SubAllocation, std::string>::makeError(
      "GeometryPool::allocateFromPool: failed to allocate in a freshly "
      "created chunk");
}

void GeometryPool::freeInPool(ChunkPool &pool,
                              const SubAllocation &allocation) {
  Chunk *chunk = findChunk(pool, allocation.chunk);
  if (chunk == nullptr || allocation.size == 0u) {
    return;
  }
  if (!chunk->mutableSuballocations) {
    chunk->freeBlocks.clear();
    chunk->freeBytes = chunk->sizeBytes;
    if (chunk->role == ChunkRole::ActiveAllocatable) {
      chunk->role = ChunkRole::Retired;
      chunk->retirementSubmission = {};
    }
    return;
  }
  if (chunk->role != ChunkRole::ActiveAllocatable &&
      chunk->role != ChunkRole::FrozenSource) {
    return;
  }
  chunk->freeBytes += allocation.size;
  const Block incoming{.offset = allocation.offset, .size = allocation.size};
  auto insertIt = std::lower_bound(
      chunk->freeBlocks.begin(), chunk->freeBlocks.end(), incoming.offset,
      [](const Block &block, size_t offset) { return block.offset < offset; });
  insertIt = chunk->freeBlocks.insert(insertIt, incoming);
  if (insertIt != chunk->freeBlocks.begin()) {
    auto prevIt = insertIt - 1;
    if (prevIt->offset + prevIt->size == insertIt->offset) {
      prevIt->size += insertIt->size;
      insertIt = chunk->freeBlocks.erase(insertIt);
      insertIt = prevIt;
    }
  }
  if (insertIt + 1 != chunk->freeBlocks.end()) {
    auto nextIt = insertIt + 1;
    if (insertIt->offset + insertIt->size == nextIt->offset) {
      insertIt->size += nextIt->size;
      chunk->freeBlocks.erase(nextIt);
    }
  }
}

void GeometryPool::reclaimRetiredAllocations() {
  for (uint32_t index = 0; index < allocations_.size(); ++index) {
    AllocationEntry &entry = allocations_[index];
    if (entry.state != AllocationEntry::State::PendingFree) {
      continue;
    }
    if (entry.retirementCaptureFailed ||
        !gpu_.isSubmissionComplete(entry.retirementSubmission)) {
      continue;
    }
    for (size_t pool = 0; pool < kPoolCount; ++pool)
      freeInPool(pools_[pool], entry.allocations[pool]);
    entry.state = AllocationEntry::State::Dead;
    entry.allocations = {};
    entry.counts = {};
    entry.retirementSubmission = {};
    entry.retirementCaptureFailed = false;
    entry.debugName.clear();
    allocationSlots_.release(index);
  }
}

void GeometryPool::reclaimRetiredChunks(ChunkPool &pool) {
  for (uint32_t index = 0; index < pool.chunks.size(); ++index) {
    Chunk &chunk = pool.chunks[index];
    if (chunk.role != ChunkRole::Retired) {
      continue;
    }
    if (nuri::isValid(chunk.retirementSubmission) &&
        !gpu_.isSubmissionComplete(chunk.retirementSubmission)) {
      continue;
    }
    if (nuri::isValid(chunk.buffer)) {
      gpu_.destroyBuffer(chunk.buffer);
    }
    chunk.reset();
    if (pool.slots.isLive(index)) {
      pool.slots.release(index);
    }
  }
}

void GeometryPool::freezeAllocatableChunks(
    ChunkPool &pool, std::pmr::vector<ChunkHandle> &frozenHandles) {
  for (uint32_t chunkIndex = 0; chunkIndex < pool.chunks.size(); ++chunkIndex) {
    Chunk &chunk = pool.chunks[chunkIndex];
    if (chunk.role != ChunkRole::ActiveAllocatable) {
      continue;
    }
    chunk.role = ChunkRole::FrozenSource;
    frozenHandles.emplace_back(ChunkHandle{
        .index = chunkIndex,
        .generation = pool.slots.generation(chunkIndex),
    });
  }
}

void GeometryPool::promoteChunks(ChunkPool &pool,
                                 std::span<const ChunkHandle> handles) {
  for (ChunkHandle handle : handles) {
    Chunk *chunk = findChunk(pool, handle);
    if (chunk == nullptr) {
      continue;
    }
    chunk->role = ChunkRole::ActiveAllocatable;
    chunk->retirementSubmission = {};
  }
}

void GeometryPool::retireChunks(ChunkPool &pool,
                                std::span<const ChunkHandle> handles,
                                SubmissionHandle retirementSubmission) {
  for (ChunkHandle handle : handles) {
    Chunk *chunk = findChunk(pool, handle);
    if (chunk == nullptr) {
      continue;
    }
    chunk->role = ChunkRole::Retired;
    chunk->retirementSubmission = retirementSubmission;
    chunk->freeBlocks.clear();
    chunk->freeBytes = 0u;
  }
}

void GeometryPool::restoreFrozenChunks(ChunkPool &pool,
                                       std::span<const ChunkHandle> handles) {
  for (ChunkHandle handle : handles) {
    Chunk *chunk = findChunk(pool, handle);
    if (chunk != nullptr && chunk->role == ChunkRole::FrozenSource) {
      if (!chunk->mutableSuballocations &&
          chunk->freeBytes == chunk->sizeBytes) {
        chunk->role = ChunkRole::Retired;
        chunk->retirementSubmission = {};
      } else {
        chunk->role = ChunkRole::ActiveAllocatable;
      }
    }
  }
}

void GeometryPool::destroyChunks(ChunkPool &pool,
                                 std::span<const ChunkHandle> handles) {
  for (ChunkHandle handle : handles) {
    Chunk *chunk = findChunk(pool, handle);
    if (chunk == nullptr) {
      continue;
    }
    if (nuri::isValid(chunk->buffer)) {
      gpu_.destroyBuffer(chunk->buffer);
    }
    chunk->reset();
    if (pool.slots.isLive(handle.index)) {
      pool.slots.release(handle.index);
    }
  }
}

size_t GeometryPool::poolBytes(const ChunkPool &pool, ChunkRole role) noexcept {
  size_t total = 0u;
  for (const Chunk &chunk : pool.chunks) {
    if (chunk.role == role) {
      total += chunk.sizeBytes;
    }
  }
  return total;
}

Result<GeometryPool::PreparedCompactionPlan, std::string>
GeometryPool::buildPreparedCompactionPlan(
    std::span<const SnapshotAllocation> snapshot, size_t currentBytes,
    const GeometryPoolConfig &config) {
  struct PoolSourceRef {
    size_t snapshotIndex = 0u;
    ChunkHandle chunk{};
    size_t offset = 0u;
    size_t size = 0u;
  };
  struct PreparedPoolPlan {
    std::vector<PreparedReplacementChunk> chunks;
    std::vector<PreparedCopy> copies;
    [[nodiscard]] size_t totalBytes() const noexcept {
      size_t total = 0u;
      for (const PreparedReplacementChunk &chunk : chunks) {
        total += chunk.sizeBytes;
      }
      return total;
    }
  };
  const std::array alignments{kVertexAlignment, kIndexAlignment};
  const std::array defaultChunkBytes{config.vertexChunkSizeBytes,
                                     config.indexChunkSizeBytes};
  const auto buildPoolPlan = [&snapshot, &alignments,
                              &defaultChunkBytes](PoolKind kind) {
    const size_t pool = poolIndex(kind);
    PreparedPoolPlan plan{};
    std::vector<PoolSourceRef> sources{};
    sources.reserve(snapshot.size());
    for (size_t snapshotIndex = 0; snapshotIndex < snapshot.size();
         ++snapshotIndex) {
      const SubAllocation &src = snapshot[snapshotIndex].allocations[pool];
      if (!isValid(src.chunk) || src.size == 0u) {
        continue;
      }
      sources.emplace_back(PoolSourceRef{
          .snapshotIndex = snapshotIndex,
          .chunk = src.chunk,
          .offset = src.offset,
          .size = src.size,
      });
    }
    std::sort(sources.begin(), sources.end(),
              [](const PoolSourceRef &lhs, const PoolSourceRef &rhs) {
                if (lhs.chunk.index != rhs.chunk.index) {
                  return lhs.chunk.index < rhs.chunk.index;
                }
                if (lhs.chunk.generation != rhs.chunk.generation) {
                  return lhs.chunk.generation < rhs.chunk.generation;
                }
                return lhs.offset < rhs.offset;
              });
    for (const PoolSourceRef &src : sources) {
      bool placed = false;
      for (uint32_t chunkIndex = 0; chunkIndex < plan.chunks.size();
           ++chunkIndex) {
        const size_t alignedOffset =
            alignUp(plan.chunks[chunkIndex].usedBytes, alignments[pool]);
        if (alignedOffset + src.size > plan.chunks[chunkIndex].sizeBytes) {
          continue;
        }
        plan.chunks[chunkIndex].usedBytes = alignedOffset + src.size;
        plan.copies.emplace_back(PreparedCopy{
            .snapshotIndex = src.snapshotIndex,
            .srcChunk = src.chunk,
            .srcOffset = src.offset,
            .size = src.size,
            .dstChunkIndex = chunkIndex,
            .dstOffset = alignedOffset,
            .pool = kind,
        });
        placed = true;
        break;
      }
      if (placed) {
        continue;
      }
      const size_t newChunkBytes = std::max(defaultChunkBytes[pool], src.size);
      plan.chunks.emplace_back(PreparedReplacementChunk{
          .sizeBytes = newChunkBytes,
          .usedBytes = src.size,
      });
      plan.copies.emplace_back(PreparedCopy{
          .snapshotIndex = src.snapshotIndex,
          .srcChunk = src.chunk,
          .srcOffset = src.offset,
          .size = src.size,
          .dstChunkIndex = static_cast<uint32_t>(plan.chunks.size() - 1u),
          .dstOffset = 0u,
          .pool = kind,
      });
    }
    return plan;
  };
  std::array poolPlans{buildPoolPlan(PoolKind::Vertex),
                       buildPoolPlan(PoolKind::Index)};
  const size_t replacementBytes =
      poolPlans[0].totalBytes() + poolPlans[1].totalBytes();
  PreparedCompactionPlan plan{};
  if (currentBytes == 0u || replacementBytes >= currentBytes) {
    return Result<PreparedCompactionPlan, std::string>::makeResult(
        std::move(plan));
  }
  const size_t savingsBytes = currentBytes - replacementBytes;
  const float savingsRatio =
      static_cast<float>(savingsBytes) / static_cast<float>(currentBytes);
  if (savingsBytes < config.compactionMinSavingsBytes ||
      savingsRatio <
          std::clamp(config.compactionFragmentationThreshold, 0.0f, 1.0f)) {
    return Result<PreparedCompactionPlan, std::string>::makeResult(
        std::move(plan));
  }
  plan.worthwhile = true;
  plan.moves.resize(snapshot.size());
  plan.copies.reserve(poolPlans[0].copies.size() + poolPlans[1].copies.size());
  for (size_t snapshotIndex = 0; snapshotIndex < snapshot.size();
       ++snapshotIndex) {
    plan.moves[snapshotIndex].allocationIndex =
        snapshot[snapshotIndex].allocationIndex;
    plan.moves[snapshotIndex].allocationGeneration =
        snapshot[snapshotIndex].allocationGeneration;
    plan.moves[snapshotIndex].dstChunkIndices.fill(UINT32_MAX);
  }
  for (size_t pool = 0; pool < kPoolCount; ++pool) {
    plan.chunks[pool] = std::move(poolPlans[pool].chunks);
    for (const PreparedCopy &copy : poolPlans[pool].copies) {
      plan.copies.emplace_back(copy);
      CompactionMove &move = plan.moves[copy.snapshotIndex];
      move.dstChunkIndices[pool] = copy.dstChunkIndex;
      move.dstOffsets[pool] = copy.dstOffset;
      move.dstSizes[pool] = copy.size;
    }
  }
  return Result<PreparedCompactionPlan, std::string>::makeResult(
      std::move(plan));
}

bool GeometryPool::shouldStartCompaction() const {
  if (compactionJob_.active() || currentFrameIndex_ == 0u) {
    return false;
  }
  if (config_.compactionCooldownFrames == 0u) {
    return true;
  }
  return currentFrameIndex_ >=
         lastCompactionStartFrame_ + config_.compactionCooldownFrames;
}

Result<bool, std::string> GeometryPool::startCompactionPlanning() {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!shouldStartCompaction()) {
    return Result<bool, std::string>::makeResult(true);
  }
  const size_t currentBytes =
      poolBytes(pools_[0], ChunkRole::ActiveAllocatable) +
      poolBytes(pools_[1], ChunkRole::ActiveAllocatable);
  if (currentBytes == 0u) {
    return Result<bool, std::string>::makeResult(true);
  }
  compactionJob_.reset();
  for (size_t pool = 0; pool < kPoolCount; ++pool)
    freezeAllocatableChunks(pools_[pool], compactionJob_.frozenChunks[pool]);
  std::vector<SnapshotAllocation> snapshot;
  snapshot.reserve(allocationSlots_.liveCount());
  for (uint32_t allocationIndex = 0; allocationIndex < allocations_.size();
       ++allocationIndex) {
    const AllocationEntry &entry = allocations_[allocationIndex];
    if (entry.state != AllocationEntry::State::Live) {
      continue;
    }
    snapshot.emplace_back(SnapshotAllocation{
        .allocationIndex = allocationIndex,
        .allocationGeneration = allocationSlots_.generation(allocationIndex),
        .allocations = entry.allocations,
    });
  }
  if (snapshot.empty()) {
    abortCompactionJob();
    return Result<bool, std::string>::makeResult(true);
  }
  try {
    compactionJob_.planningFuture =
        std::async(
            std::launch::async,
            [snapshot = std::move(snapshot), currentBytes, config = config_]()
                -> Result<PreparedCompactionPlan, std::string> {
              NURI_PROFILER_THREAD("geometry_pool_plan");
              NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
              return buildPreparedCompactionPlan(snapshot, currentBytes,
                                                 config);
            })
            .share();
  } catch (const std::exception &e) {
    abortCompactionJob();
    return Result<bool, std::string>::makeError(
        std::string("GeometryPool::startCompactionPlanning: failed to launch "
                    "planning task: ") +
        e.what());
  } catch (...) {
    abortCompactionJob();
    return Result<bool, std::string>::makeError(
        "GeometryPool::startCompactionPlanning: failed to launch planning "
        "task");
  }
  lastCompactionStartFrame_ = currentFrameIndex_;
  compactionJob_.state = CompactionJobState::Planning;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> GeometryPool::pollCompactionPlanning() {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_WAIT);
  if (compactionJob_.state != CompactionJobState::Planning) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (!compactionJob_.planningFuture.valid() ||
      compactionJob_.planningFuture.wait_for(std::chrono::seconds(0)) !=
          std::future_status::ready) {
    return Result<bool, std::string>::makeResult(true);
  }
  Result<PreparedCompactionPlan, std::string> planResult =
      compactionJob_.planningFuture.get();
  compactionJob_.planningFuture = {};
  if (planResult.hasError()) {
    abortCompactionJob();
    return Result<bool, std::string>::makeError(planResult.error());
  }
  if (!planResult.value().worthwhile) {
    abortCompactionJob();
    return Result<bool, std::string>::makeResult(true);
  }
  compactionJob_.preparedPlan.emplace(std::move(planResult.value()));
  compactionJob_.nextReplacementChunkIndices = {};
  compactionJob_.nextCopyIndex = 0u;
  compactionJob_.state = CompactionJobState::MaterializingReplacement;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> GeometryPool::materializeReplacementChunks() {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (compactionJob_.state != CompactionJobState::MaterializingReplacement ||
      !compactionJob_.preparedPlan.has_value()) {
    return Result<bool, std::string>::makeResult(true);
  }
  const auto materializeOne =
      [this](const PreparedReplacementChunk &planChunk, ChunkPool &pool,
             std::pmr::vector<ChunkHandle> &handles, BufferUsage usage,
             std::string_view debugPrefix) -> Result<bool, std::string> {
    auto createResult = createChunk(pool, planChunk.sizeBytes, usage,
                                    debugPrefix, ChunkRole::Replacement);
    if (createResult.hasError()) {
      return Result<bool, std::string>::makeError(createResult.error());
    }
    handles.emplace_back(createResult.value());
    Chunk *chunk = findChunk(pool, handles.back());
    chunk->freeBlocks.clear();
    if (planChunk.usedBytes < chunk->sizeBytes) {
      chunk->freeBlocks.emplace_back(Block{
          .offset = planChunk.usedBytes,
          .size = chunk->sizeBytes - planChunk.usedBytes,
      });
    }
    chunk->freeBytes = chunk->sizeBytes - planChunk.usedBytes;
    return Result<bool, std::string>::makeResult(true);
  };
  const PreparedCompactionPlan &plan = *compactionJob_.preparedPlan;
  static constexpr std::array usages{BufferUsage::Storage | BufferUsage::Vertex,
                                     BufferUsage::Index};
  static constexpr std::array names{"geometry_pool_compact_vb",
                                    "geometry_pool_compact_ib"};
  bool complete = true;
  for (size_t pool = 0; pool < kPoolCount; ++pool) {
    size_t &next = compactionJob_.nextReplacementChunkIndices[pool];
    if (next < plan.chunks[pool].size()) {
      auto created = materializeOne(plan.chunks[pool][next], pools_[pool],
                                    compactionJob_.replacementChunks[pool],
                                    usages[pool], names[pool]);
      if (created.hasError()) {
        abortCompactionJob();
        return created;
      }
      ++next;
    }
    complete &= next == plan.chunks[pool].size();
  }
  if (complete) {
    compactionJob_.state = plan.copies.empty()
                               ? CompactionJobState::ReadyToCommit
                               : CompactionJobState::ReadyToSubmit;
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> GeometryPool::submitNextCopyBatch() {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CMD_COPY);
  if (compactionJob_.state != CompactionJobState::ReadyToSubmit ||
      !compactionJob_.preparedPlan.has_value()) {
    return Result<bool, std::string>::makeResult(true);
  }
  const PreparedCompactionPlan &plan = *compactionJob_.preparedPlan;
  if (compactionJob_.nextCopyIndex >= plan.copies.size()) {
    compactionJob_.state = CompactionJobState::ReadyToCommit;
    return Result<bool, std::string>::makeResult(true);
  }
  const size_t copyBudgetBytes =
      std::max<size_t>(1u, config_.compactionCopyBudgetBytesPerFrame);
  std::pmr::vector<BufferCopyRegion> regions(memory_);
  regions.reserve(plan.copies.size() - compactionJob_.nextCopyIndex);
  size_t batchBytes = 0u;
  size_t nextCopyIndex = compactionJob_.nextCopyIndex;
  while (nextCopyIndex < plan.copies.size()) {
    const PreparedCopy &copy = plan.copies[nextCopyIndex];
    if (copy.size == 0u) {
      ++nextCopyIndex;
      continue;
    }
    const bool wouldExceedBudget =
        !regions.empty() && batchBytes + copy.size > copyBudgetBytes;
    if (wouldExceedBudget) {
      break;
    }
    const size_t pool = poolIndex(copy.pool);
    const Chunk &srcChunk = *findChunk(pools_[pool], copy.srcChunk);
    const Chunk &dstChunk =
        *findChunk(pools_[pool],
                   compactionJob_.replacementChunks[pool][copy.dstChunkIndex]);
    regions.emplace_back(BufferCopyRegion{
        .srcBuffer = srcChunk.buffer,
        .dstBuffer = dstChunk.buffer,
        .srcOffset = copy.srcOffset,
        .dstOffset = copy.dstOffset,
        .size = copy.size,
    });
    batchBytes += copy.size;
    ++nextCopyIndex;
  }
  if (regions.empty()) {
    compactionJob_.nextCopyIndex = nextCopyIndex;
    compactionJob_.state = nextCopyIndex < plan.copies.size()
                               ? CompactionJobState::ReadyToSubmit
                               : CompactionJobState::ReadyToCommit;
    return Result<bool, std::string>::makeResult(true);
  }
  auto submitResult =
      gpu_.submitBackgroundBufferCopies(regions, "geometry_pool_compaction");
  if (submitResult.hasError()) {
    abortCompactionJob();
    return Result<bool, std::string>::makeError(submitResult.error());
  }
  compactionJob_.nextCopyIndex = nextCopyIndex;
  compactionJob_.inFlightSubmission = submitResult.value();
  compactionJob_.state = CompactionJobState::WaitingForCopy;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> GeometryPool::pollCompactionJob() {
  switch (compactionJob_.state) {
  case CompactionJobState::Idle:
    return Result<bool, std::string>::makeResult(true);
  case CompactionJobState::Planning:
    return pollCompactionPlanning();
  case CompactionJobState::MaterializingReplacement:
    return materializeReplacementChunks();
  case CompactionJobState::ReadyToSubmit:
    return submitNextCopyBatch();
  case CompactionJobState::WaitingForCopy: {
    if (!gpu_.isSubmissionComplete(compactionJob_.inFlightSubmission)) {
      return Result<bool, std::string>::makeResult(true);
    }
    auto visibility =
        gpu_.makeSubmissionVisibleToGraphics(compactionJob_.inFlightSubmission);
    if (visibility.hasError()) {
      abortCompactionJob();
      return Result<bool, std::string>::makeError(visibility.error());
    }
    if (!visibility.value()) {
      return Result<bool, std::string>::makeResult(true);
    }
    compactionJob_.inFlightSubmission = {};
    compactionJob_.state = (compactionJob_.preparedPlan.has_value() &&
                            compactionJob_.nextCopyIndex <
                                compactionJob_.preparedPlan->copies.size())
                               ? CompactionJobState::ReadyToSubmit
                               : CompactionJobState::ReadyToCommit;
    return Result<bool, std::string>::makeResult(true);
  }
  case CompactionJobState::ReadyToCommit:
    return commitCompactionJob();
  }
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> GeometryPool::commitCompactionJob() {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (compactionJob_.state != CompactionJobState::ReadyToCommit ||
      !compactionJob_.preparedPlan.has_value()) {
    return Result<bool, std::string>::makeResult(true);
  }
  auto retirement = gpu_.captureWorkCompletion();
  if (retirement.hasError())
    return Result<bool, std::string>::makeError(retirement.error());
  for (const CompactionMove &move : compactionJob_.preparedPlan->moves) {
    if (!allocationSlots_.isValid(move.allocationIndex,
                                  move.allocationGeneration)) {
      continue;
    }
    AllocationEntry &entry = allocations_[move.allocationIndex];
    if (entry.state != AllocationEntry::State::Live) {
      continue;
    }
    for (size_t pool = 0; pool < kPoolCount; ++pool) {
      if (move.dstChunkIndices[pool] != UINT32_MAX) {
        entry.allocations[pool] = SubAllocation{
            .chunk = compactionJob_
                         .replacementChunks[pool][move.dstChunkIndices[pool]],
            .offset = move.dstOffsets[pool],
            .size = move.dstSizes[pool],
        };
      }
    }
  }
  for (size_t pool = 0; pool < kPoolCount; ++pool) {
    promoteChunks(pools_[pool], compactionJob_.replacementChunks[pool]);
    retireChunks(pools_[pool], compactionJob_.frozenChunks[pool],
                 retirement.value());
  }
  compactionJob_.reset();
  bumpMutationVersion();
  return Result<bool, std::string>::makeResult(true);
}

void GeometryPool::abortCompactionJob() {
  for (size_t pool = 0; pool < kPoolCount; ++pool) {
    restoreFrozenChunks(pools_[pool], compactionJob_.frozenChunks[pool]);
    destroyChunks(pools_[pool], compactionJob_.replacementChunks[pool]);
  }
  compactionJob_.reset();
}

bool GeometryPool::isHandleLive(GeometryAllocationHandle handle) const {
  return allocationSlots_.isValid(handle.index, handle.generation) &&
         allocations_[handle.index].state == AllocationEntry::State::Live;
}

void GeometryPool::bumpMutationVersion() noexcept {
  ++mutationVersion_;
  if (mutationVersion_ == 0u) {
    mutationVersion_ = 1u;
  }
}

GeometryPool::Chunk *GeometryPool::findChunk(ChunkPool &pool,
                                             ChunkHandle handle) {
  if (!pool.slots.isValid(handle.index, handle.generation)) {
    return nullptr;
  }
  return &pool.chunks[handle.index];
}

const GeometryPool::Chunk *GeometryPool::findChunk(const ChunkPool &pool,
                                                   ChunkHandle handle) const {
  if (!pool.slots.isValid(handle.index, handle.generation)) {
    return nullptr;
  }
  return &pool.chunks[handle.index];
}

Result<bool, std::string> GeometryPool::beginFrame(uint64_t frameIndex) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  currentFrameIndex_ = frameIndex;
  NURI_PROFILER_ZONE("GeometryPool.reclaim_allocations",
                     NURI_PROFILER_COLOR_DESTROY);
  reclaimRetiredAllocations();
  NURI_PROFILER_ZONE_END();
  NURI_PROFILER_ZONE("GeometryPool.reclaim_chunks",
                     NURI_PROFILER_COLOR_DESTROY);
  for (ChunkPool &pool : pools_)
    reclaimRetiredChunks(pool);
  NURI_PROFILER_ZONE_END();
  if (shouldStartCompaction()) {
    auto startResult = startCompactionPlanning();
    if (startResult.hasError()) {
      return startResult;
    }
  }
  auto pollResult = Result<bool, std::string>::makeResult(true);
  NURI_PROFILER_ZONE("GeometryPool.advance_compaction",
                     NURI_PROFILER_COLOR_CREATE);
  pollResult = pollCompactionJob();
  NURI_PROFILER_ZONE_END();
  return pollResult;
}

Result<GeometryAllocationHandle, std::string>
GeometryPool::allocate(std::span<const std::byte> vertexBytes,
                       uint32_t vertexCount,
                       std::span<const std::byte> indexBytes,
                       uint32_t indexCount, std::string_view debugName) {
  if (vertexBytes.empty() || indexBytes.empty()) {
    return Result<GeometryAllocationHandle, std::string>::makeError(
        "GeometryPool::allocate: geometry data is empty");
  }
  auto vertexAllocResult = allocateFromPool(
      pools_[0], vertexBytes.size(), kVertexAlignment,
      config_.vertexChunkSizeBytes, BufferUsage::Storage | BufferUsage::Vertex,
      "geometry_pool_vb");
  if (vertexAllocResult.hasError()) {
    return Result<GeometryAllocationHandle, std::string>::makeError(
        vertexAllocResult.error());
  }
  const SubAllocation vertexAllocation = vertexAllocResult.value();
  auto indexAllocResult = allocateFromPool(
      pools_[1], indexBytes.size(), kIndexAlignment,
      config_.indexChunkSizeBytes, BufferUsage::Index, "geometry_pool_ib");
  if (indexAllocResult.hasError()) {
    freeInPool(pools_[0], vertexAllocation);
    return Result<GeometryAllocationHandle, std::string>::makeError(
        indexAllocResult.error());
  }
  const SubAllocation indexAllocation = indexAllocResult.value();
  const Chunk *vertexChunk = findChunk(pools_[0], vertexAllocation.chunk);
  const Chunk *indexChunk = findChunk(pools_[1], indexAllocation.chunk);
  const std::array uploads{
      BufferUpdate{.buffer = vertexChunk->buffer,
                   .data = vertexBytes,
                   .offset = vertexAllocation.offset},
      BufferUpdate{.buffer = indexChunk->buffer,
                   .data = indexBytes,
                   .offset = indexAllocation.offset},
  };
  auto uploadResult = gpu_.updateBuffers(uploads);
  if (uploadResult.hasError()) {
    freeInPool(pools_[0], vertexAllocation);
    freeInPool(pools_[1], indexAllocation);
    return Result<GeometryAllocationHandle, std::string>::makeError(
        uploadResult.error());
  }
  const SlotReservation slot = allocationSlots_.acquire();
  const uint32_t allocationIndex = slot.index;
  if (slot.appended) {
    allocations_.emplace_back(memory_);
  }
  AllocationEntry &entry = allocations_[allocationIndex];
  entry.state = AllocationEntry::State::Live;
  entry.allocations = {vertexAllocation, indexAllocation};
  entry.counts = {vertexCount, indexCount};
  entry.retirementSubmission = {};
  entry.retirementCaptureFailed = false;
  entry.debugName.assign(debugName.data(), debugName.size());
  bumpMutationVersion();
  return Result<GeometryAllocationHandle, std::string>::makeResult(
      GeometryAllocationHandle{
          .index = allocationIndex,
          .generation = slot.generation,
      });
}

Result<GeometryAllocationHandle, std::string>
GeometryPool::adoptPrepared(BufferHandle vertexBuffer, size_t vertexBytes,
                            uint32_t vertexCount, BufferHandle indexBuffer,
                            size_t indexBytes, uint32_t indexCount,
                            std::string_view debugName) {
  if (!nuri::isValid(vertexBuffer) || !gpu_.isValid(vertexBuffer) ||
      vertexBytes == 0u || vertexCount == 0u) {
    return Result<GeometryAllocationHandle, std::string>::makeError(
        "GeometryPool::adoptPrepared: invalid vertex buffer");
  }
  if (!nuri::isValid(indexBuffer) || !gpu_.isValid(indexBuffer) ||
      indexBytes == 0u || indexCount == 0u) {
    return Result<GeometryAllocationHandle, std::string>::makeError(
        "GeometryPool::adoptPrepared: invalid index buffer");
  }
  const auto adoptChunk = [this](ChunkPool &pool, BufferHandle buffer,
                                 size_t sizeBytes) -> ChunkHandle {
    const SlotReservation slot = pool.slots.acquire();
    if (slot.appended) {
      pool.chunks.emplace_back(memory_);
    } else {
      pool.chunks[slot.index].reset();
    }
    Chunk &chunk = pool.chunks[slot.index];
    chunk.buffer = buffer;
    chunk.sizeBytes = sizeBytes;
    chunk.freeBytes = 0u;
    chunk.retirementSubmission = {};
    chunk.role = ChunkRole::ActiveAllocatable;
    chunk.mutableSuballocations = false;
    chunk.freeBlocks.clear();
    return ChunkHandle{.index = slot.index, .generation = slot.generation};
  };
  const ChunkHandle vertexChunk =
      adoptChunk(pools_[0], vertexBuffer, vertexBytes);
  const ChunkHandle indexChunk = adoptChunk(pools_[1], indexBuffer, indexBytes);
  const SlotReservation slot = allocationSlots_.acquire();
  if (slot.appended) {
    allocations_.emplace_back(memory_);
  }
  AllocationEntry &entry = allocations_[slot.index];
  entry.state = AllocationEntry::State::Live;
  entry.allocations = {
      SubAllocation{.chunk = vertexChunk, .offset = 0u, .size = vertexBytes},
      SubAllocation{.chunk = indexChunk, .offset = 0u, .size = indexBytes}};
  entry.counts = {vertexCount, indexCount};
  entry.retirementSubmission = {};
  entry.retirementCaptureFailed = false;
  entry.debugName.assign(debugName.data(), debugName.size());
  bumpMutationVersion();
  return Result<GeometryAllocationHandle, std::string>::makeResult(
      GeometryAllocationHandle{
          .index = slot.index,
          .generation = slot.generation,
      });
}

void GeometryPool::release(GeometryAllocationHandle handle) {
  if (!isHandleLive(handle)) {
    return;
  }
  AllocationEntry &entry = allocations_[handle.index];
  entry.state = AllocationEntry::State::PendingFree;
  auto captureResult = gpu_.captureWorkCompletion();
  if (captureResult.hasError()) {
    entry.retirementCaptureFailed = true;
    NURI_LOG_ERROR(
        "GeometryPool::release: failed to capture GPU completion for '%s': %s",
        entry.debugName.c_str(), captureResult.error().c_str());
  } else {
    entry.retirementSubmission = captureResult.value();
    entry.retirementCaptureFailed = false;
  }
  bumpMutationVersion();
}

bool GeometryPool::resolve(GeometryAllocationHandle handle,
                           GeometryAllocationView &out) const {
  if (!isHandleLive(handle)) {
    return false;
  }
  const AllocationEntry &entry = allocations_[handle.index];
  const SubAllocation &vertex = entry.allocations[0];
  const SubAllocation &index = entry.allocations[1];
  const Chunk *vertexChunk = findChunk(pools_[0], vertex.chunk);
  const Chunk *indexChunk = findChunk(pools_[1], index.chunk);
  if (vertexChunk == nullptr || indexChunk == nullptr) {
    return false;
  }
  out = GeometryAllocationView{
      .vertexBuffer = vertexChunk->buffer,
      .vertexByteOffset = vertex.offset,
      .vertexByteSize = vertex.size,
      .indexBuffer = indexChunk->buffer,
      .indexByteOffset = index.offset,
      .indexByteSize = index.size,
      .vertexCount = entry.counts[0],
      .indexCount = entry.counts[1],
  };
  return true;
}

} // namespace nuri
