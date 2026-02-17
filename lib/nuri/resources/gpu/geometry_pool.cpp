#include "nuri/resources/gpu/geometry_pool.h"

#include "nuri/gfx/gpu_descriptors.h"
#include "nuri/gfx/gpu_device.h"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>

namespace nuri {
namespace {

struct PoolSourceRef {
  uint32_t allocationIndex = 0;
  uint32_t chunkIndex = 0;
  size_t offset = 0;
  size_t size = 0;
};

} // namespace

GeometryPool::GeometryPool(GPUDevice &gpu, GeometryPoolConfig config,
                           std::pmr::memory_resource *memory)
    : gpu_(gpu), config_(config),
      memory_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      vertexChunks_(memory_), indexChunks_(memory_), allocations_(memory_),
      freeAllocationIndices_(memory_), retiredVertexChunks_(memory_),
      retiredIndexChunks_(memory_) {}

GeometryPool::~GeometryPool() {
  for (const Chunk &chunk : vertexChunks_) {
    if (nuri::isValid(chunk.buffer)) {
      gpu_.destroyBuffer(chunk.buffer);
    }
  }
  for (const Chunk &chunk : indexChunks_) {
    if (nuri::isValid(chunk.buffer)) {
      gpu_.destroyBuffer(chunk.buffer);
    }
  }
  for (const RetiredChunk &chunk : retiredVertexChunks_) {
    if (nuri::isValid(chunk.buffer)) {
      gpu_.destroyBuffer(chunk.buffer);
    }
  }
  for (const RetiredChunk &chunk : retiredIndexChunks_) {
    if (nuri::isValid(chunk.buffer)) {
      gpu_.destroyBuffer(chunk.buffer);
    }
  }
}

size_t GeometryPool::alignUp(size_t value, size_t alignment) {
  if (alignment == 0) {
    return value;
  }
  const size_t mask = alignment - 1;
  return (value + mask) & ~mask;
}

Result<bool, std::string>
GeometryPool::createChunk(std::pmr::vector<Chunk> &chunks, size_t minimumSize,
                          BufferUsage usage, std::string_view debugPrefix) {
  const size_t requestedSize = std::max<size_t>(minimumSize, 1);
  const BufferDesc desc{
      .usage = usage,
      .storage = Storage::Device,
      .size = requestedSize,
  };

  const std::string debugName =
      std::string(debugPrefix) + "_" + std::to_string(chunks.size());
  auto bufferResult = gpu_.createBuffer(desc, debugName);
  if (bufferResult.hasError()) {
    return Result<bool, std::string>::makeError(bufferResult.error());
  }

  chunks.emplace_back(memory_);
  Chunk &chunk = chunks.back();
  chunk.buffer = bufferResult.value();
  chunk.sizeBytes = requestedSize;
  chunk.freeBytes = requestedSize;
  chunk.freeBlocks.emplace_back(Block{.offset = 0, .size = requestedSize});
  return Result<bool, std::string>::makeResult(true);
}

Result<GeometryPool::SubAllocation, std::string> GeometryPool::allocateFromPool(
    std::pmr::vector<Chunk> &chunks, size_t sizeBytes, size_t alignment,
    size_t defaultChunkSize, BufferUsage usage, std::string_view debugPrefix) {
  if (sizeBytes == 0) {
    return Result<SubAllocation, std::string>::makeError(
        "GeometryPool::allocateFromPool: allocation size is zero");
  }

  const auto tryAllocateInChunk =
      [sizeBytes, alignment](
          Chunk &chunk, uint32_t chunkIndex) -> std::optional<SubAllocation> {
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
      if (padding > 0) {
        replacement[replacementCount++] =
            Block{.offset = block.offset, .size = padding};
      }
      if (remaining > 0) {
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
          .chunkIndex = chunkIndex,
          .offset = alignedOffset,
          .size = sizeBytes,
      };
    }
    return std::nullopt;
  };

  for (uint32_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex) {
    if (auto allocation = tryAllocateInChunk(chunks[chunkIndex], chunkIndex)) {
      return Result<SubAllocation, std::string>::makeResult(*allocation);
    }
  }

  const size_t grownChunkSize = std::max(defaultChunkSize, sizeBytes);
  auto createResult = createChunk(chunks, grownChunkSize, usage, debugPrefix);
  if (createResult.hasError()) {
    return Result<SubAllocation, std::string>::makeError(createResult.error());
  }

  Chunk &newChunk = chunks.back();
  if (auto allocation = tryAllocateInChunk(
          newChunk, static_cast<uint32_t>(chunks.size() - 1));
      allocation) {
    return Result<SubAllocation, std::string>::makeResult(*allocation);
  }

  return Result<SubAllocation, std::string>::makeError(
      "GeometryPool::allocateFromPool: failed to allocate in a freshly created "
      "chunk");
}

void GeometryPool::freeInPool(std::pmr::vector<Chunk> &chunks,
                              const SubAllocation &allocation) {
  if (allocation.chunkIndex == kInvalidChunkIndex || allocation.size == 0 ||
      allocation.chunkIndex >= chunks.size()) {
    return;
  }

  Chunk &chunk = chunks[allocation.chunkIndex];
  chunk.freeBytes += allocation.size;

  const Block incoming{.offset = allocation.offset, .size = allocation.size};
  auto insertIt = std::lower_bound(
      chunk.freeBlocks.begin(), chunk.freeBlocks.end(), incoming.offset,
      [](const Block &block, size_t offset) { return block.offset < offset; });

  insertIt = chunk.freeBlocks.insert(insertIt, incoming);

  if (insertIt != chunk.freeBlocks.begin()) {
    auto prevIt = insertIt - 1;
    if (prevIt->offset + prevIt->size == insertIt->offset) {
      prevIt->size += insertIt->size;
      insertIt = chunk.freeBlocks.erase(insertIt);
      insertIt = prevIt;
    }
  }

  if (insertIt + 1 != chunk.freeBlocks.end()) {
    auto nextIt = insertIt + 1;
    if (insertIt->offset + insertIt->size == nextIt->offset) {
      insertIt->size += nextIt->size;
      chunk.freeBlocks.erase(nextIt);
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

    freeInPool(vertexChunks_, entry.vertex);
    freeInPool(indexChunks_, entry.index);

    entry.state = AllocationEntry::State::Dead;
    entry.vertex = {};
    entry.index = {};
    entry.vertexCount = 0;
    entry.indexCount = 0;
    entry.retireFrame = 0;
    entry.debugName.clear();
    freeAllocationIndices_.emplace_back(index);
  }
}

void GeometryPool::reclaimRetiredChunks(
    std::pmr::deque<RetiredChunk> &retiredChunks) {
  const uint64_t lag = reclaimLagFrames();
  while (!retiredChunks.empty()) {
    const RetiredChunk &chunk = retiredChunks.front();
    if (currentFrameIndex_ < chunk.retireFrame + lag) {
      break;
    }

    if (nuri::isValid(chunk.buffer)) {
      gpu_.destroyBuffer(chunk.buffer);
    }
    retiredChunks.pop_front();
  }
}

bool GeometryPool::shouldCompactPool(
    const std::pmr::vector<Chunk> &chunks) const {
  const float threshold =
      std::clamp(config_.compactionFragmentationThreshold, 0.0f, 1.0f);
  for (const Chunk &chunk : chunks) {
    if (chunk.sizeBytes == 0) {
      continue;
    }
    const float freeRatio = static_cast<float>(chunk.freeBytes) /
                            static_cast<float>(chunk.sizeBytes);
    if (freeRatio >= threshold) {
      return true;
    }
  }
  return false;
}

Result<GeometryPool::PoolCompactionPlan, std::string>
GeometryPool::buildCompactionPlan(std::pmr::vector<Chunk> &chunks,
                                  bool forVertexPool) {
  PoolCompactionPlan plan(memory_);
  if (chunks.empty()) {
    return Result<PoolCompactionPlan, std::string>::makeResult(std::move(plan));
  }

  std::pmr::vector<PoolSourceRef> sources(memory_);
  for (uint32_t allocationIndex = 0; allocationIndex < allocations_.size();
       ++allocationIndex) {
    const AllocationEntry &entry = allocations_[allocationIndex];
    if (entry.state == AllocationEntry::State::Dead) {
      continue;
    }

    const SubAllocation &src = forVertexPool ? entry.vertex : entry.index;
    if (src.chunkIndex == kInvalidChunkIndex || src.size == 0) {
      continue;
    }

    sources.emplace_back(PoolSourceRef{
        .allocationIndex = allocationIndex,
        .chunkIndex = src.chunkIndex,
        .offset = src.offset,
        .size = src.size,
    });
  }

  if (sources.empty()) {
    return Result<PoolCompactionPlan, std::string>::makeResult(std::move(plan));
  }

  std::sort(sources.begin(), sources.end(),
            [](const PoolSourceRef &a, const PoolSourceRef &b) {
              if (a.chunkIndex != b.chunkIndex) {
                return a.chunkIndex < b.chunkIndex;
              }
              return a.offset < b.offset;
            });

  const BufferUsage usage =
      forVertexPool ? BufferUsage::Storage : BufferUsage::Index;
  const size_t alignment = forVertexPool ? kVertexAlignment : kIndexAlignment;
  const size_t defaultChunkBytes = forVertexPool ? config_.vertexChunkSizeBytes
                                                 : config_.indexChunkSizeBytes;
  const std::string_view debugPrefix =
      forVertexPool ? "geometry_pool_compact_vb" : "geometry_pool_compact_ib";

  std::pmr::vector<size_t> usedBytes(memory_);
  usedBytes.reserve(sources.size());

  const auto allocateDestination =
      [&](size_t sizeBytes) -> Result<AllocationTarget, std::string> {
    for (uint32_t chunkIndex = 0; chunkIndex < plan.newChunks.size();
         ++chunkIndex) {
      const size_t alignedOffset = alignUp(usedBytes[chunkIndex], alignment);
      if (alignedOffset + sizeBytes <= plan.newChunks[chunkIndex].sizeBytes) {
        usedBytes[chunkIndex] = alignedOffset + sizeBytes;
        return Result<AllocationTarget, std::string>::makeResult(
            AllocationTarget{
                .chunkIndex = chunkIndex,
                .offset = alignedOffset,
            });
      }
    }

    const size_t newChunkSize = std::max(defaultChunkBytes, sizeBytes);
    auto createResult =
        createChunk(plan.newChunks, newChunkSize, usage, debugPrefix);
    if (createResult.hasError()) {
      return Result<AllocationTarget, std::string>::makeError(
          createResult.error());
    }

    usedBytes.emplace_back(0);
    const uint32_t chunkIndex =
        static_cast<uint32_t>(plan.newChunks.size() - 1);
    const size_t alignedOffset = alignUp(usedBytes[chunkIndex], alignment);
    if (alignedOffset + sizeBytes > plan.newChunks[chunkIndex].sizeBytes) {
      return Result<AllocationTarget, std::string>::makeError(
          "GeometryPool::buildCompactionPlan: failed to place allocation in "
          "new chunk");
    }
    usedBytes[chunkIndex] = alignedOffset + sizeBytes;
    return Result<AllocationTarget, std::string>::makeResult(AllocationTarget{
        .chunkIndex = chunkIndex,
        .offset = alignedOffset,
    });
  };

  for (const PoolSourceRef &srcRef : sources) {
    if (srcRef.chunkIndex >= chunks.size()) {
      return Result<PoolCompactionPlan, std::string>::makeError(
          "GeometryPool::buildCompactionPlan: source chunk index out of range");
    }

    auto dstResult = allocateDestination(srcRef.size);
    if (dstResult.hasError()) {
      for (const Chunk &chunk : plan.newChunks) {
        if (nuri::isValid(chunk.buffer)) {
          gpu_.destroyBuffer(chunk.buffer);
        }
      }
      return Result<PoolCompactionPlan, std::string>::makeError(
          dstResult.error());
    }
    const AllocationTarget dst = dstResult.value();

    plan.copyRegions.emplace_back(BufferCopyRegion{
        .srcBuffer = chunks[srcRef.chunkIndex].buffer,
        .dstBuffer = plan.newChunks[dst.chunkIndex].buffer,
        .srcOffset = srcRef.offset,
        .dstOffset = dst.offset,
        .size = srcRef.size,
    });
    plan.moves.emplace_back(PoolCompactionMove{
        .allocationIndex = srcRef.allocationIndex,
        .isVertex = forVertexPool,
        .dst =
            SubAllocation{
                .chunkIndex = dst.chunkIndex,
                .offset = dst.offset,
                .size = srcRef.size,
            },
    });
  }

  for (uint32_t chunkIndex = 0; chunkIndex < plan.newChunks.size();
       ++chunkIndex) {
    Chunk &chunk = plan.newChunks[chunkIndex];
    const size_t used = usedBytes[chunkIndex];
    chunk.freeBlocks.clear();
    if (used < chunk.sizeBytes) {
      chunk.freeBlocks.emplace_back(Block{
          .offset = used,
          .size = chunk.sizeBytes - used,
      });
    }
    chunk.freeBytes = chunk.sizeBytes - used;
  }

  return Result<PoolCompactionPlan, std::string>::makeResult(std::move(plan));
}

Result<bool, std::string>
GeometryPool::applyCompactionPlan(std::pmr::vector<Chunk> &chunks,
                                  std::pmr::deque<RetiredChunk> &retiredChunks,
                                  PoolCompactionPlan &plan) {
  if (plan.newChunks.empty()) {
    for (const Chunk &chunk : chunks) {
      retiredChunks.emplace_back(RetiredChunk{
          .buffer = chunk.buffer, .retireFrame = currentFrameIndex_});
    }
    chunks.clear();
    return Result<bool, std::string>::makeResult(true);
  }

  if (!plan.copyRegions.empty()) {
    auto copyResult = gpu_.copyBufferRegions(std::span<const BufferCopyRegion>(
        plan.copyRegions.data(), plan.copyRegions.size()));
    if (copyResult.hasError()) {
      for (const Chunk &chunk : plan.newChunks) {
        if (nuri::isValid(chunk.buffer)) {
          gpu_.destroyBuffer(chunk.buffer);
        }
      }
      return copyResult;
    }
  }

  for (const PoolCompactionMove &move : plan.moves) {
    if (move.allocationIndex >= allocations_.size()) {
      continue;
    }
    AllocationEntry &entry = allocations_[move.allocationIndex];
    if (entry.state == AllocationEntry::State::Dead) {
      continue;
    }
    if (move.isVertex) {
      entry.vertex = move.dst;
    } else {
      entry.index = move.dst;
    }
  }

  for (const Chunk &chunk : chunks) {
    retiredChunks.emplace_back(RetiredChunk{.buffer = chunk.buffer,
                                            .retireFrame = currentFrameIndex_});
  }
  chunks = std::move(plan.newChunks);
  return Result<bool, std::string>::makeResult(true);
}

Result<bool, std::string> GeometryPool::compactIfNeeded() {
  if (!config_.enableCompaction || config_.compactionIntervalFrames == 0) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (currentFrameIndex_ == 0 ||
      (currentFrameIndex_ % config_.compactionIntervalFrames) != 0) {
    return Result<bool, std::string>::makeResult(true);
  }

  if (shouldCompactPool(vertexChunks_)) {
    auto planResult = buildCompactionPlan(vertexChunks_, true);
    if (planResult.hasError()) {
      return Result<bool, std::string>::makeError(planResult.error());
    }
    PoolCompactionPlan plan = std::move(planResult.value());
    auto applyResult =
        applyCompactionPlan(vertexChunks_, retiredVertexChunks_, plan);
    if (applyResult.hasError()) {
      return applyResult;
    }
  }

  if (shouldCompactPool(indexChunks_)) {
    auto planResult = buildCompactionPlan(indexChunks_, false);
    if (planResult.hasError()) {
      return Result<bool, std::string>::makeError(planResult.error());
    }
    PoolCompactionPlan plan = std::move(planResult.value());
    auto applyResult =
        applyCompactionPlan(indexChunks_, retiredIndexChunks_, plan);
    if (applyResult.hasError()) {
      return applyResult;
    }
  }

  return Result<bool, std::string>::makeResult(true);
}

bool GeometryPool::isHandleLive(GeometryAllocationHandle handle) const {
  if (!nuri::isValid(handle) || handle.index >= allocations_.size()) {
    return false;
  }
  const AllocationEntry &entry = allocations_[handle.index];
  return entry.generation == handle.generation &&
         entry.state == AllocationEntry::State::Live;
}

Result<bool, std::string> GeometryPool::beginFrame(uint64_t frameIndex) {
  currentFrameIndex_ = frameIndex;
  reclaimRetiredAllocations();
  reclaimRetiredChunks(retiredVertexChunks_);
  reclaimRetiredChunks(retiredIndexChunks_);
  return compactIfNeeded();
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
      vertexChunks_, vertexBytes.size(), kVertexAlignment,
      config_.vertexChunkSizeBytes, BufferUsage::Storage, "geometry_pool_vb");
  if (vertexAllocResult.hasError()) {
    return Result<GeometryAllocationHandle, std::string>::makeError(
        vertexAllocResult.error());
  }
  const SubAllocation vertexAllocation = vertexAllocResult.value();

  auto indexAllocResult = allocateFromPool(
      indexChunks_, indexBytes.size(), kIndexAlignment,
      config_.indexChunkSizeBytes, BufferUsage::Index, "geometry_pool_ib");
  if (indexAllocResult.hasError()) {
    freeInPool(vertexChunks_, vertexAllocation);
    return Result<GeometryAllocationHandle, std::string>::makeError(
        indexAllocResult.error());
  }
  const SubAllocation indexAllocation = indexAllocResult.value();

  auto uploadVertices =
      gpu_.updateBuffer(vertexChunks_[vertexAllocation.chunkIndex].buffer,
                        vertexBytes, vertexAllocation.offset);
  if (uploadVertices.hasError()) {
    freeInPool(vertexChunks_, vertexAllocation);
    freeInPool(indexChunks_, indexAllocation);
    return Result<GeometryAllocationHandle, std::string>::makeError(
        uploadVertices.error());
  }

  auto uploadIndices =
      gpu_.updateBuffer(indexChunks_[indexAllocation.chunkIndex].buffer,
                        indexBytes, indexAllocation.offset);
  if (uploadIndices.hasError()) {
    freeInPool(vertexChunks_, vertexAllocation);
    freeInPool(indexChunks_, indexAllocation);
    return Result<GeometryAllocationHandle, std::string>::makeError(
        uploadIndices.error());
  }

  uint32_t allocationIndex = 0;
  if (!freeAllocationIndices_.empty()) {
    allocationIndex = freeAllocationIndices_.back();
    freeAllocationIndices_.pop_back();
  } else {
    allocations_.emplace_back(memory_);
    allocationIndex = static_cast<uint32_t>(allocations_.size() - 1);
  }

  AllocationEntry &entry = allocations_[allocationIndex];
  entry.generation += 1;
  if (entry.generation == 0) {
    entry.generation = 1;
  }
  entry.state = AllocationEntry::State::Live;
  entry.vertex = vertexAllocation;
  entry.index = indexAllocation;
  entry.vertexCount = vertexCount;
  entry.indexCount = indexCount;
  entry.retireFrame = 0;
  entry.debugName.assign(debugName.data(), debugName.size());

  return Result<GeometryAllocationHandle, std::string>::makeResult(
      GeometryAllocationHandle{
          .index = allocationIndex,
          .generation = entry.generation,
      });
}

void GeometryPool::release(GeometryAllocationHandle handle) {
  if (!isHandleLive(handle)) {
    return;
  }

  AllocationEntry &entry = allocations_[handle.index];
  entry.state = AllocationEntry::State::PendingFree;
  entry.retireFrame = currentFrameIndex_;
}

bool GeometryPool::resolve(GeometryAllocationHandle handle,
                           GeometryAllocationView &out) const {
  if (!isHandleLive(handle)) {
    return false;
  }

  const AllocationEntry &entry = allocations_[handle.index];
  if (entry.vertex.chunkIndex >= vertexChunks_.size() ||
      entry.index.chunkIndex >= indexChunks_.size()) {
    return false;
  }

  out = GeometryAllocationView{
      .vertexBuffer = vertexChunks_[entry.vertex.chunkIndex].buffer,
      .vertexByteOffset = entry.vertex.offset,
      .vertexByteSize = entry.vertex.size,
      .indexBuffer = indexChunks_[entry.index.chunkIndex].buffer,
      .indexByteOffset = entry.index.offset,
      .indexByteSize = entry.index.size,
      .vertexCount = entry.vertexCount,
      .indexCount = entry.indexCount,
  };
  return true;
}

} // namespace nuri
