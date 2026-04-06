#include "nuri/pch.h"

#include "nuri/resources/gpu/geometry_pool.h"

#include "nuri/core/profiling.h"
#include "nuri/gfx/gpu_descriptors.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/utils/utils.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <numeric>
#include <optional>

namespace nuri {

GeometryPool::GeometryPool(GPUDevice &gpu, GeometryPoolConfig config,
                           std::pmr::memory_resource *memory)
    : gpu_(gpu), config_(config), memory_(ensureMemory(memory)),
      vertexChunks_(memory_), indexChunks_(memory_), allocations_(memory_),
      compactionJob_(memory_) {}

GeometryPool::~GeometryPool() {
  const auto destroyChunks = [this](const std::pmr::vector<Chunk> &chunks) {
    for (const Chunk &chunk : chunks) {
      if (nuri::isValid(chunk.buffer)) {
        gpu_.destroyBuffer(chunk.buffer);
      }
    }
  };
  destroyChunks(vertexChunks_);
  destroyChunks(indexChunks_);
}

Result<GeometryPool::ChunkHandle, std::string>
GeometryPool::createChunk(std::pmr::vector<Chunk> &chunks,
                          SlotPool<UnmaskedNonZeroGenerationPolicy> &chunkSlots,
                          size_t minimumSize, BufferUsage usage,
                          std::string_view debugPrefix, ChunkRole role) {
  const size_t requestedSize = std::max<size_t>(minimumSize, 1u);
  const BufferDesc desc{
      .usage = usage,
      .storage = Storage::Device,
      .size = requestedSize,
  };

  const SlotReservation slot = chunkSlots.acquire();
  if (slot.appended) {
    NURI_ASSERT(
        slot.index == chunks.size(),
        "GeometryPool::createChunk: appended slot index=%u does not match "
        "chunks.size()=%zu",
        slot.index, chunks.size());
    chunks.emplace_back(memory_);
  } else {
    NURI_ASSERT(slot.index < chunks.size(),
                "GeometryPool::createChunk: reused slot index=%u is out of "
                "range for chunks.size()=%zu",
                slot.index, chunks.size());
    chunks[slot.index].reset();
  }

  const std::string debugName =
      std::string(debugPrefix) + "_" + std::to_string(slot.index);
  auto bufferResult = gpu_.createBuffer(desc, debugName);
  if (bufferResult.hasError()) {
    chunkSlots.release(slot.index);
    chunks[slot.index].reset();
    return Result<ChunkHandle, std::string>::makeError(bufferResult.error());
  }

  Chunk &chunk = chunks[slot.index];
  chunk.buffer = bufferResult.value();
  chunk.sizeBytes = requestedSize;
  chunk.freeBytes = requestedSize;
  chunk.retireFrame = 0;
  chunk.role = role;
  chunk.freeBlocks.clear();
  chunk.freeBlocks.emplace_back(Block{.offset = 0, .size = requestedSize});

  return Result<ChunkHandle, std::string>::makeResult(
      ChunkHandle{.index = slot.index, .generation = slot.generation});
}

Result<GeometryPool::SubAllocation, std::string> GeometryPool::allocateFromPool(
    std::pmr::vector<Chunk> &chunks,
    SlotPool<UnmaskedNonZeroGenerationPolicy> &chunkSlots, size_t sizeBytes,
    size_t alignment, size_t defaultChunkSize, BufferUsage usage,
    std::string_view debugPrefix) {
  if (sizeBytes == 0u) {
    return Result<SubAllocation, std::string>::makeError(
        "GeometryPool::allocateFromPool: allocation size is zero");
  }

  const auto tryAllocateInChunk =
      [sizeBytes, alignment](
          Chunk &chunk, ChunkHandle handle) -> std::optional<SubAllocation> {
    if (chunk.role != ChunkRole::ActiveAllocatable) {
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

  for (uint32_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex) {
    Chunk &chunk = chunks[chunkIndex];
    if (chunk.role != ChunkRole::ActiveAllocatable) {
      continue;
    }
    const ChunkHandle handle{
        .index = chunkIndex,
        .generation = chunkSlots.generation(chunkIndex),
    };
    if (auto allocation = tryAllocateInChunk(chunk, handle)) {
      return Result<SubAllocation, std::string>::makeResult(*allocation);
    }
  }

  auto createResult =
      createChunk(chunks, chunkSlots, std::max(defaultChunkSize, sizeBytes),
                  usage, debugPrefix, ChunkRole::ActiveAllocatable);
  if (createResult.hasError()) {
    return Result<SubAllocation, std::string>::makeError(createResult.error());
  }

  const ChunkHandle newHandle = createResult.value();
  Chunk *newChunk = findChunk(chunks, chunkSlots, newHandle);
  NURI_ASSERT(newChunk != nullptr,
              "GeometryPool::allocateFromPool: freshly created chunk is not "
              "accessible");
  if (auto allocation = tryAllocateInChunk(*newChunk, newHandle)) {
    return Result<SubAllocation, std::string>::makeResult(*allocation);
  }

  return Result<SubAllocation, std::string>::makeError(
      "GeometryPool::allocateFromPool: failed to allocate in a freshly "
      "created chunk");
}

void GeometryPool::freeInPool(
    std::pmr::vector<Chunk> &chunks,
    SlotPool<UnmaskedNonZeroGenerationPolicy> &chunkSlots,
    const SubAllocation &allocation) {
  Chunk *chunk = findChunk(chunks, chunkSlots, allocation.chunk);
  if (chunk == nullptr || allocation.size == 0u) {
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

uint64_t GeometryPool::reclaimLagFrames() const {
  return static_cast<uint64_t>(std::max(1u, gpu_.getSwapchainImageCount())) +
         1u;
}

void GeometryPool::reclaimRetiredAllocations() {
  const uint64_t lag = reclaimLagFrames();
  for (uint32_t index = 0; index < allocations_.size(); ++index) {
    AllocationEntry &entry = allocations_[index];
    if (entry.state != AllocationEntry::State::PendingFree) {
      continue;
    }
    if (currentFrameIndex_ < entry.retireFrame + lag) {
      continue;
    }

    freeInPool(vertexChunks_, vertexChunkSlots_, entry.vertex);
    freeInPool(indexChunks_, indexChunkSlots_, entry.index);

    entry.state = AllocationEntry::State::Dead;
    entry.vertex = {};
    entry.index = {};
    entry.vertexCount = 0;
    entry.indexCount = 0;
    entry.retireFrame = 0;
    entry.debugName.clear();
    allocationSlots_.release(index);
  }
}

void GeometryPool::reclaimRetiredChunks(
    std::pmr::vector<Chunk> &chunks,
    SlotPool<UnmaskedNonZeroGenerationPolicy> &chunkSlots) {
  const uint64_t lag = reclaimLagFrames();
  for (uint32_t index = 0; index < chunks.size(); ++index) {
    Chunk &chunk = chunks[index];
    if (chunk.role != ChunkRole::Retired) {
      continue;
    }
    if (currentFrameIndex_ < chunk.retireFrame + lag) {
      continue;
    }

    if (nuri::isValid(chunk.buffer)) {
      gpu_.destroyBuffer(chunk.buffer);
    }
    chunk.reset();
    if (chunkSlots.isLive(index)) {
      chunkSlots.release(index);
    }
  }
}

void GeometryPool::freezeAllocatableChunks(
    std::pmr::vector<Chunk> &chunks,
    SlotPool<UnmaskedNonZeroGenerationPolicy> &chunkSlots,
    std::pmr::vector<ChunkHandle> &frozenHandles) {
  for (uint32_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex) {
    Chunk &chunk = chunks[chunkIndex];
    if (chunk.role != ChunkRole::ActiveAllocatable) {
      continue;
    }
    chunk.role = ChunkRole::FrozenSource;
    frozenHandles.emplace_back(ChunkHandle{
        .index = chunkIndex,
        .generation = chunkSlots.generation(chunkIndex),
    });
  }
}

void GeometryPool::promoteChunks(
    std::pmr::vector<Chunk> &chunks,
    SlotPool<UnmaskedNonZeroGenerationPolicy> &chunkSlots,
    std::span<const ChunkHandle> handles) {
  for (ChunkHandle handle : handles) {
    Chunk *chunk = findChunk(chunks, chunkSlots, handle);
    if (chunk == nullptr) {
      continue;
    }
    chunk->role = ChunkRole::ActiveAllocatable;
    chunk->retireFrame = 0u;
  }
}

void GeometryPool::retireChunks(
    std::pmr::vector<Chunk> &chunks,
    SlotPool<UnmaskedNonZeroGenerationPolicy> &chunkSlots,
    std::span<const ChunkHandle> handles, uint64_t retireFrame) {
  for (ChunkHandle handle : handles) {
    Chunk *chunk = findChunk(chunks, chunkSlots, handle);
    if (chunk == nullptr) {
      continue;
    }
    chunk->role = ChunkRole::Retired;
    chunk->retireFrame = retireFrame;
    chunk->freeBlocks.clear();
    chunk->freeBytes = 0u;
  }
}

void GeometryPool::restoreFrozenChunks(
    std::pmr::vector<Chunk> &chunks,
    SlotPool<UnmaskedNonZeroGenerationPolicy> &chunkSlots,
    std::span<const ChunkHandle> handles) {
  for (ChunkHandle handle : handles) {
    Chunk *chunk = findChunk(chunks, chunkSlots, handle);
    if (chunk != nullptr && chunk->role == ChunkRole::FrozenSource) {
      chunk->role = ChunkRole::ActiveAllocatable;
    }
  }
}

void GeometryPool::destroyChunks(
    std::pmr::vector<Chunk> &chunks,
    SlotPool<UnmaskedNonZeroGenerationPolicy> &chunkSlots,
    std::span<const ChunkHandle> handles) {
  for (ChunkHandle handle : handles) {
    Chunk *chunk = findChunk(chunks, chunkSlots, handle);
    if (chunk == nullptr) {
      continue;
    }
    if (nuri::isValid(chunk->buffer)) {
      gpu_.destroyBuffer(chunk->buffer);
    }
    chunk->reset();
    if (chunkSlots.isLive(handle.index)) {
      chunkSlots.release(handle.index);
    }
  }
}

size_t GeometryPool::poolBytes(const std::pmr::vector<Chunk> &chunks,
                               ChunkRole role) noexcept {
  size_t total = 0u;
  for (const Chunk &chunk : chunks) {
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
    std::vector<PreparedShadowChunk> chunks;
    std::vector<PreparedCopy> copies;

    [[nodiscard]] size_t totalBytes() const noexcept {
      size_t total = 0u;
      for (const PreparedShadowChunk &chunk : chunks) {
        total += chunk.sizeBytes;
      }
      return total;
    }
  };

  const auto buildPoolPlan = [&snapshot, &config](bool forVertexPool) {
    PreparedPoolPlan plan{};
    std::vector<PoolSourceRef> sources{};
    sources.reserve(snapshot.size());
    for (size_t snapshotIndex = 0; snapshotIndex < snapshot.size();
         ++snapshotIndex) {
      const SubAllocation &src = forVertexPool ? snapshot[snapshotIndex].vertex
                                               : snapshot[snapshotIndex].index;
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

    const size_t alignment = forVertexPool ? kVertexAlignment : kIndexAlignment;
    const size_t defaultChunkBytes = forVertexPool ? config.vertexChunkSizeBytes
                                                   : config.indexChunkSizeBytes;

    for (const PoolSourceRef &src : sources) {
      bool placed = false;
      for (uint32_t chunkIndex = 0; chunkIndex < plan.chunks.size();
           ++chunkIndex) {
        const size_t alignedOffset =
            alignUp(plan.chunks[chunkIndex].usedBytes, alignment);
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
            .isVertex = forVertexPool,
        });
        placed = true;
        break;
      }

      if (placed) {
        continue;
      }

      const size_t newChunkBytes = std::max(defaultChunkBytes, src.size);
      plan.chunks.emplace_back(PreparedShadowChunk{
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
          .isVertex = forVertexPool,
      });
    }

    return plan;
  };

  PreparedPoolPlan vertexPlan = buildPoolPlan(true);
  PreparedPoolPlan indexPlan = buildPoolPlan(false);
  const size_t shadowBytes = vertexPlan.totalBytes() + indexPlan.totalBytes();

  PreparedCompactionPlan plan{};
  if (currentBytes == 0u || shadowBytes >= currentBytes) {
    return Result<PreparedCompactionPlan, std::string>::makeResult(
        std::move(plan));
  }

  const size_t savingsBytes = currentBytes - shadowBytes;
  const float savingsRatio =
      static_cast<float>(savingsBytes) / static_cast<float>(currentBytes);
  if (savingsBytes < config.compactionMinSavingsBytes ||
      savingsRatio <
          std::clamp(config.compactionFragmentationThreshold, 0.0f, 1.0f)) {
    return Result<PreparedCompactionPlan, std::string>::makeResult(
        std::move(plan));
  }

  plan.worthwhile = true;
  plan.vertexChunks = std::move(vertexPlan.chunks);
  plan.indexChunks = std::move(indexPlan.chunks);
  plan.moves.resize(snapshot.size());
  plan.copies.reserve(vertexPlan.copies.size() + indexPlan.copies.size());

  for (size_t snapshotIndex = 0; snapshotIndex < snapshot.size();
       ++snapshotIndex) {
    plan.moves[snapshotIndex].allocationIndex =
        snapshot[snapshotIndex].allocationIndex;
    plan.moves[snapshotIndex].allocationGeneration =
        snapshot[snapshotIndex].allocationGeneration;
    plan.moves[snapshotIndex].dstVertexChunkIndex = UINT32_MAX;
    plan.moves[snapshotIndex].dstIndexChunkIndex = UINT32_MAX;
  }

  for (const PreparedCopy &copy : vertexPlan.copies) {
    plan.copies.emplace_back(copy);
    CompactionMove &move = plan.moves[copy.snapshotIndex];
    move.dstVertexChunkIndex = copy.dstChunkIndex;
    move.dstVertexOffset = copy.dstOffset;
    move.dstVertexSize = copy.size;
  }
  for (const PreparedCopy &copy : indexPlan.copies) {
    plan.copies.emplace_back(copy);
    CompactionMove &move = plan.moves[copy.snapshotIndex];
    move.dstIndexChunkIndex = copy.dstChunkIndex;
    move.dstIndexOffset = copy.dstOffset;
    move.dstIndexSize = copy.size;
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
      poolBytes(vertexChunks_, ChunkRole::ActiveAllocatable) +
      poolBytes(indexChunks_, ChunkRole::ActiveAllocatable);
  if (currentBytes == 0u) {
    return Result<bool, std::string>::makeResult(true);
  }

  compactionJob_.reset();
  freezeAllocatableChunks(vertexChunks_, vertexChunkSlots_,
                          compactionJob_.frozenVertexChunks);
  freezeAllocatableChunks(indexChunks_, indexChunkSlots_,
                          compactionJob_.frozenIndexChunks);

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
        .vertex = entry.vertex,
        .index = entry.index,
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
  compactionJob_.nextShadowVertexChunkIndex = 0u;
  compactionJob_.nextShadowIndexChunkIndex = 0u;
  compactionJob_.nextCopyIndex = 0u;
  compactionJob_.state = CompactionJobState::MaterializingShadow;
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> GeometryPool::materializeShadowChunks() {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (compactionJob_.state != CompactionJobState::MaterializingShadow ||
      !compactionJob_.preparedPlan.has_value()) {
    return Result<bool, std::string>::makeResult(true);
  }

  const auto materializeOne =
      [this](const PreparedShadowChunk &planChunk,
             std::pmr::vector<Chunk> &chunks,
             SlotPool<UnmaskedNonZeroGenerationPolicy> &chunkSlots,
             std::pmr::vector<ChunkHandle> &handles, BufferUsage usage,
             std::string_view debugPrefix) -> Result<bool, std::string> {
    auto createResult = createChunk(chunks, chunkSlots, planChunk.sizeBytes,
                                    usage, debugPrefix, ChunkRole::Shadow);
    if (createResult.hasError()) {
      return Result<bool, std::string>::makeError(createResult.error());
    }
    handles.emplace_back(createResult.value());

    Chunk *chunk = findChunk(chunks, chunkSlots, handles.back());
    NURI_ASSERT(chunk != nullptr,
                "GeometryPool::materializeShadowChunks: missing shadow "
                "chunk");
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
  if (compactionJob_.nextShadowVertexChunkIndex < plan.vertexChunks.size()) {
    auto createResult = materializeOne(
        plan.vertexChunks[compactionJob_.nextShadowVertexChunkIndex],
        vertexChunks_, vertexChunkSlots_, compactionJob_.shadowVertexChunks,
        BufferUsage::Storage, "geometry_pool_compact_vb");
    if (createResult.hasError()) {
      abortCompactionJob();
      return createResult;
    }
    ++compactionJob_.nextShadowVertexChunkIndex;
  }

  if (compactionJob_.nextShadowIndexChunkIndex < plan.indexChunks.size()) {
    auto createResult = materializeOne(
        plan.indexChunks[compactionJob_.nextShadowIndexChunkIndex],
        indexChunks_, indexChunkSlots_, compactionJob_.shadowIndexChunks,
        BufferUsage::Index, "geometry_pool_compact_ib");
    if (createResult.hasError()) {
      abortCompactionJob();
      return createResult;
    }
    ++compactionJob_.nextShadowIndexChunkIndex;
  }

  if (compactionJob_.nextShadowVertexChunkIndex >= plan.vertexChunks.size() &&
      compactionJob_.nextShadowIndexChunkIndex >= plan.indexChunks.size()) {
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

    const Chunk *srcChunk =
        copy.isVertex
            ? findChunk(vertexChunks_, vertexChunkSlots_, copy.srcChunk)
            : findChunk(indexChunks_, indexChunkSlots_, copy.srcChunk);
    if (srcChunk == nullptr) {
      abortCompactionJob();
      return Result<bool, std::string>::makeError(
          "GeometryPool::submitNextCopyBatch: missing source chunk");
    }

    const std::pmr::vector<ChunkHandle> &shadowHandles =
        copy.isVertex ? compactionJob_.shadowVertexChunks
                      : compactionJob_.shadowIndexChunks;
    if (copy.dstChunkIndex >= shadowHandles.size()) {
      abortCompactionJob();
      return Result<bool, std::string>::makeError(
          "GeometryPool::submitNextCopyBatch: destination chunk index is out "
          "of range");
    }

    const Chunk *dstChunk = copy.isVertex
                                ? findChunk(vertexChunks_, vertexChunkSlots_,
                                            shadowHandles[copy.dstChunkIndex])
                                : findChunk(indexChunks_, indexChunkSlots_,
                                            shadowHandles[copy.dstChunkIndex]);
    if (dstChunk == nullptr) {
      abortCompactionJob();
      return Result<bool, std::string>::makeError(
          "GeometryPool::submitNextCopyBatch: missing destination chunk");
    }

    regions.emplace_back(BufferCopyRegion{
        .srcBuffer = srcChunk->buffer,
        .dstBuffer = dstChunk->buffer,
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
  case CompactionJobState::MaterializingShadow:
    return materializeShadowChunks();
  case CompactionJobState::ReadyToSubmit:
    return submitNextCopyBatch();
  case CompactionJobState::WaitingForCopy: {
    if (!gpu_.isSubmissionComplete(compactionJob_.inFlightSubmission)) {
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

  for (const CompactionMove &move : compactionJob_.preparedPlan->moves) {
    if (move.allocationIndex >= allocations_.size()) {
      continue;
    }
    if (!allocationSlots_.isValid(move.allocationIndex,
                                  move.allocationGeneration)) {
      continue;
    }

    AllocationEntry &entry = allocations_[move.allocationIndex];
    if (entry.state != AllocationEntry::State::Live) {
      continue;
    }

    if (move.dstVertexChunkIndex != UINT32_MAX) {
      NURI_ASSERT(
          move.dstVertexChunkIndex < compactionJob_.shadowVertexChunks.size(),
          "GeometryPool::commitCompactionJob: vertex chunk index is out of "
          "range");
      entry.vertex = SubAllocation{
          .chunk = compactionJob_.shadowVertexChunks[move.dstVertexChunkIndex],
          .offset = move.dstVertexOffset,
          .size = move.dstVertexSize,
      };
    }
    if (move.dstIndexChunkIndex != UINT32_MAX) {
      NURI_ASSERT(
          move.dstIndexChunkIndex < compactionJob_.shadowIndexChunks.size(),
          "GeometryPool::commitCompactionJob: index chunk index is out of "
          "range");
      entry.index = SubAllocation{
          .chunk = compactionJob_.shadowIndexChunks[move.dstIndexChunkIndex],
          .offset = move.dstIndexOffset,
          .size = move.dstIndexSize,
      };
    }
  }

  promoteChunks(vertexChunks_, vertexChunkSlots_,
                compactionJob_.shadowVertexChunks);
  promoteChunks(indexChunks_, indexChunkSlots_,
                compactionJob_.shadowIndexChunks);
  retireChunks(vertexChunks_, vertexChunkSlots_,
               compactionJob_.frozenVertexChunks, currentFrameIndex_);
  retireChunks(indexChunks_, indexChunkSlots_, compactionJob_.frozenIndexChunks,
               currentFrameIndex_);

  compactionJob_.reset();
  bumpMutationVersion();
  return Result<bool, std::string>::makeResult(true);
}

void GeometryPool::abortCompactionJob() {
  restoreFrozenChunks(vertexChunks_, vertexChunkSlots_,
                      compactionJob_.frozenVertexChunks);
  restoreFrozenChunks(indexChunks_, indexChunkSlots_,
                      compactionJob_.frozenIndexChunks);
  destroyChunks(vertexChunks_, vertexChunkSlots_,
                compactionJob_.shadowVertexChunks);
  destroyChunks(indexChunks_, indexChunkSlots_,
                compactionJob_.shadowIndexChunks);
  compactionJob_.reset();
}

bool GeometryPool::isHandleLive(GeometryAllocationHandle handle) const {
  if (!nuri::isValid(handle) || handle.index >= allocations_.size()) {
    return false;
  }
  const AllocationEntry &entry = allocations_[handle.index];
  return allocationSlots_.isValid(handle.index, handle.generation) &&
         entry.state == AllocationEntry::State::Live;
}

void GeometryPool::bumpMutationVersion() noexcept {
  ++mutationVersion_;
  if (mutationVersion_ == 0u) {
    mutationVersion_ = 1u;
  }
}

GeometryPool::Chunk *GeometryPool::findChunk(
    std::pmr::vector<Chunk> &chunks,
    const SlotPool<UnmaskedNonZeroGenerationPolicy> &chunkSlots,
    ChunkHandle handle) {
  if (!isValid(handle) || handle.index >= chunks.size() ||
      !chunkSlots.isValid(handle.index, handle.generation)) {
    return nullptr;
  }
  return &chunks[handle.index];
}

const GeometryPool::Chunk *GeometryPool::findChunk(
    const std::pmr::vector<Chunk> &chunks,
    const SlotPool<UnmaskedNonZeroGenerationPolicy> &chunkSlots,
    ChunkHandle handle) const {
  if (!isValid(handle) || handle.index >= chunks.size() ||
      !chunkSlots.isValid(handle.index, handle.generation)) {
    return nullptr;
  }
  return &chunks[handle.index];
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
  reclaimRetiredChunks(vertexChunks_, vertexChunkSlots_);
  reclaimRetiredChunks(indexChunks_, indexChunkSlots_);
  NURI_PROFILER_ZONE_END();

  if (!compactionJob_.active() && shouldStartCompaction()) {
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
  if (vertexBytes.empty()) {
    return Result<GeometryAllocationHandle, std::string>::makeError(
        "GeometryPool::allocate: vertex data is empty");
  }
  if (indexBytes.empty()) {
    return Result<GeometryAllocationHandle, std::string>::makeError(
        "GeometryPool::allocate: index data is empty");
  }

  auto vertexAllocResult = allocateFromPool(
      vertexChunks_, vertexChunkSlots_, vertexBytes.size(), kVertexAlignment,
      config_.vertexChunkSizeBytes, BufferUsage::Storage, "geometry_pool_vb");
  if (vertexAllocResult.hasError()) {
    return Result<GeometryAllocationHandle, std::string>::makeError(
        vertexAllocResult.error());
  }
  const SubAllocation vertexAllocation = vertexAllocResult.value();

  auto indexAllocResult = allocateFromPool(
      indexChunks_, indexChunkSlots_, indexBytes.size(), kIndexAlignment,
      config_.indexChunkSizeBytes, BufferUsage::Index, "geometry_pool_ib");
  if (indexAllocResult.hasError()) {
    freeInPool(vertexChunks_, vertexChunkSlots_, vertexAllocation);
    return Result<GeometryAllocationHandle, std::string>::makeError(
        indexAllocResult.error());
  }
  const SubAllocation indexAllocation = indexAllocResult.value();

  const Chunk *vertexChunk =
      findChunk(vertexChunks_, vertexChunkSlots_, vertexAllocation.chunk);
  const Chunk *indexChunk =
      findChunk(indexChunks_, indexChunkSlots_, indexAllocation.chunk);
  NURI_ASSERT(vertexChunk != nullptr && indexChunk != nullptr,
              "GeometryPool::allocate: missing chunk after allocation");

  auto uploadVertices = gpu_.updateBuffer(vertexChunk->buffer, vertexBytes,
                                          vertexAllocation.offset);
  if (uploadVertices.hasError()) {
    freeInPool(vertexChunks_, vertexChunkSlots_, vertexAllocation);
    freeInPool(indexChunks_, indexChunkSlots_, indexAllocation);
    return Result<GeometryAllocationHandle, std::string>::makeError(
        uploadVertices.error());
  }

  auto uploadIndices =
      gpu_.updateBuffer(indexChunk->buffer, indexBytes, indexAllocation.offset);
  if (uploadIndices.hasError()) {
    freeInPool(vertexChunks_, vertexChunkSlots_, vertexAllocation);
    freeInPool(indexChunks_, indexChunkSlots_, indexAllocation);
    return Result<GeometryAllocationHandle, std::string>::makeError(
        uploadIndices.error());
  }

  const SlotReservation slot = allocationSlots_.acquire();
  const uint32_t allocationIndex = slot.index;
  if (slot.appended) {
    NURI_ASSERT(allocationIndex == allocations_.size(),
                "GeometryPool::allocate: appended slot index=%u but "
                "allocations_.size()="
                "%zu",
                allocationIndex, allocations_.size());
    allocations_.emplace_back(memory_);
  } else {
    NURI_ASSERT(allocationIndex < allocations_.size(),
                "GeometryPool::allocate: reused slot index=%u is out of range "
                "for allocations_.size()=%zu",
                allocationIndex, allocations_.size());
  }

  AllocationEntry &entry = allocations_[allocationIndex];
  entry.state = AllocationEntry::State::Live;
  entry.vertex = vertexAllocation;
  entry.index = indexAllocation;
  entry.vertexCount = vertexCount;
  entry.indexCount = indexCount;
  entry.retireFrame = 0u;
  entry.debugName.assign(debugName.data(), debugName.size());
  bumpMutationVersion();

  return Result<GeometryAllocationHandle, std::string>::makeResult(
      GeometryAllocationHandle{
          .index = allocationIndex,
          .generation = slot.generation,
      });
}

void GeometryPool::release(GeometryAllocationHandle handle) {
  if (!isHandleLive(handle)) {
    return;
  }

  AllocationEntry &entry = allocations_[handle.index];
  entry.state = AllocationEntry::State::PendingFree;
  entry.retireFrame = currentFrameIndex_;
  bumpMutationVersion();
}

bool GeometryPool::resolve(GeometryAllocationHandle handle,
                           GeometryAllocationView &out) const {
  if (!isHandleLive(handle)) {
    return false;
  }

  const AllocationEntry &entry = allocations_[handle.index];
  const Chunk *vertexChunk =
      findChunk(vertexChunks_, vertexChunkSlots_, entry.vertex.chunk);
  const Chunk *indexChunk =
      findChunk(indexChunks_, indexChunkSlots_, entry.index.chunk);
  if (vertexChunk == nullptr || indexChunk == nullptr) {
    return false;
  }

  out = GeometryAllocationView{
      .vertexBuffer = vertexChunk->buffer,
      .vertexByteOffset = entry.vertex.offset,
      .vertexByteSize = entry.vertex.size,
      .indexBuffer = indexChunk->buffer,
      .indexByteOffset = entry.index.offset,
      .indexByteSize = entry.index.size,
      .vertexCount = entry.vertexCount,
      .indexCount = entry.indexCount,
  };
  return true;
}

} // namespace nuri
