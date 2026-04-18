#include "tests_pch.h"

#include "nuri/resources/gpu/geometry_pool.h"
#include "render_graph_test_support.h"

#include <chrono>
#include <thread>

namespace {

using nuri::GeometryAllocationHandle;
using nuri::GeometryAllocationView;
using nuri::GeometryPool;
using nuri::GeometryPoolConfig;
using nuri::test_support::FakeExecutorGPUDevice;
using nuri::test_support::sameHandle;

std::vector<std::byte> makeBytes(uint8_t seed, size_t size) {
  std::vector<std::byte> bytes(size);
  for (size_t i = 0; i < size; ++i) {
    bytes[i] = static_cast<std::byte>(seed + static_cast<uint8_t>(i));
  }
  return bytes;
}

std::vector<std::byte> readBytes(FakeExecutorGPUDevice &gpu,
                                 nuri::BufferHandle buffer, size_t offset,
                                 size_t size) {
  std::vector<std::byte> bytes(size);
  auto result = gpu.readBuffer(buffer, offset, bytes);
  EXPECT_FALSE(result.hasError()) << result.error();
  return bytes;
}

void advanceFrame(FakeExecutorGPUDevice &gpu, GeometryPool &pool,
                  uint64_t frameIndex) {
  auto gpuResult = gpu.beginFrame(frameIndex);
  ASSERT_FALSE(gpuResult.hasError()) << gpuResult.error();
  auto poolResult = pool.beginFrame(frameIndex);
  ASSERT_FALSE(poolResult.hasError()) << poolResult.error();
}

template <typename Predicate>
void advanceFramesUntil(FakeExecutorGPUDevice &gpu, GeometryPool &pool,
                        uint64_t &frameIndex, uint32_t maxFrames,
                        Predicate &&predicate) {
  for (uint32_t step = 0; step < maxFrames && !predicate(); ++step) {
    advanceFrame(gpu, pool, ++frameIndex);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

GeometryPoolConfig makeConfig() {
  GeometryPoolConfig config{};
  config.vertexChunkSizeBytes = 32u;
  config.indexChunkSizeBytes = 16u;
  config.compactionCooldownFrames = 1u;
  config.compactionFragmentationThreshold = 0.1f;
  config.compactionMinSavingsBytes = 1u;
  config.compactionCopyBudgetBytesPerFrame = 64u;
  return config;
}

TEST(GeometryPoolTests, CompactionSnapshotsOnlyLiveAllocations) {
  FakeExecutorGPUDevice gpu;
  GeometryPool pool(gpu, makeConfig());
  uint64_t frameIndex = 1u;

  const std::vector<std::byte> vertexA = makeBytes(0x10u, 24u);
  const std::vector<std::byte> indexA = makeBytes(0x20u, 8u);
  const std::vector<std::byte> vertexB = makeBytes(0x30u, 24u);
  const std::vector<std::byte> indexB = makeBytes(0x40u, 8u);

  advanceFrame(gpu, pool, frameIndex);
  auto allocA = pool.allocate(vertexA, 3u, indexA, 6u, "a");
  ASSERT_FALSE(allocA.hasError()) << allocA.error();
  auto allocB = pool.allocate(vertexB, 3u, indexB, 6u, "b");
  ASSERT_FALSE(allocB.hasError()) << allocB.error();
  pool.release(allocB.value());

  advanceFramesUntil(gpu, pool, frameIndex, 8u,
                     [&gpu] { return gpu.backgroundCopySubmitCount == 1u; });

  ASSERT_EQ(gpu.backgroundCopySubmitCount, 1u);
  ASSERT_EQ(gpu.backgroundCopyBatchSizes.size(), 1u);
  ASSERT_EQ(gpu.backgroundCopyBatchBytes.size(), 1u);
  EXPECT_EQ(gpu.backgroundCopyBatchSizes[0], 2u);
  EXPECT_EQ(gpu.backgroundCopyBatchBytes[0], vertexA.size() + indexA.size());
}

TEST(GeometryPoolTests, AllocationsCreatedDuringCompactionRemainValid) {
  FakeExecutorGPUDevice gpu;
  GeometryPool pool(gpu, makeConfig());
  uint64_t frameIndex = 1u;

  const std::vector<std::byte> vertexA = makeBytes(0x10u, 24u);
  const std::vector<std::byte> indexA = makeBytes(0x20u, 8u);
  const std::vector<std::byte> vertexB = makeBytes(0x30u, 24u);
  const std::vector<std::byte> indexB = makeBytes(0x40u, 8u);
  const std::vector<std::byte> vertexC = makeBytes(0x50u, 24u);
  const std::vector<std::byte> indexC = makeBytes(0x60u, 8u);

  advanceFrame(gpu, pool, frameIndex);
  auto allocA = pool.allocate(vertexA, 3u, indexA, 6u, "a");
  ASSERT_FALSE(allocA.hasError()) << allocA.error();
  auto allocB = pool.allocate(vertexB, 3u, indexB, 6u, "b");
  ASSERT_FALSE(allocB.hasError()) << allocB.error();
  pool.release(allocB.value());

  GeometryAllocationView viewABefore{};
  ASSERT_TRUE(pool.resolve(allocA.value(), viewABefore));
  const uint64_t versionBeforeCompaction = pool.mutationVersion();

  advanceFrame(gpu, pool, ++frameIndex);
  EXPECT_EQ(pool.mutationVersion(), versionBeforeCompaction);

  auto allocC = pool.allocate(vertexC, 3u, indexC, 6u, "c");
  ASSERT_FALSE(allocC.hasError()) << allocC.error();

  GeometryAllocationView viewCBefore{};
  ASSERT_TRUE(pool.resolve(allocC.value(), viewCBefore));

  advanceFramesUntil(
      gpu, pool, frameIndex, 10u, [&pool, versionBeforeCompaction] {
        return pool.mutationVersion() == versionBeforeCompaction + 2u;
      });

  GeometryAllocationView viewAAfter{};
  GeometryAllocationView viewCAfter{};
  ASSERT_TRUE(pool.resolve(allocA.value(), viewAAfter));
  ASSERT_TRUE(pool.resolve(allocC.value(), viewCAfter));

  EXPECT_FALSE(sameHandle(viewABefore.vertexBuffer, viewAAfter.vertexBuffer));
  EXPECT_FALSE(sameHandle(viewABefore.indexBuffer, viewAAfter.indexBuffer));
  EXPECT_TRUE(sameHandle(viewCBefore.vertexBuffer, viewCAfter.vertexBuffer));
  EXPECT_TRUE(sameHandle(viewCBefore.indexBuffer, viewCAfter.indexBuffer));
  EXPECT_EQ(pool.mutationVersion(), versionBeforeCompaction + 2u);

  EXPECT_EQ(readBytes(gpu, viewAAfter.vertexBuffer, viewAAfter.vertexByteOffset,
                      viewAAfter.vertexByteSize),
            vertexA);
  EXPECT_EQ(readBytes(gpu, viewAAfter.indexBuffer, viewAAfter.indexByteOffset,
                      viewAAfter.indexByteSize),
            indexA);
  EXPECT_EQ(readBytes(gpu, viewCAfter.vertexBuffer, viewCAfter.vertexByteOffset,
                      viewCAfter.vertexByteSize),
            vertexC);
  EXPECT_EQ(readBytes(gpu, viewCAfter.indexBuffer, viewCAfter.indexByteOffset,
                      viewCAfter.indexByteSize),
            indexC);
}

TEST(GeometryPoolTests, ReleasedAllocationsAreNotPatchedAndRetiredChunksLag) {
  FakeExecutorGPUDevice gpu;
  GeometryPool pool(gpu, makeConfig());
  uint64_t frameIndex = 1u;

  const std::vector<std::byte> vertexA = makeBytes(0x10u, 24u);
  const std::vector<std::byte> indexA = makeBytes(0x20u, 8u);
  const std::vector<std::byte> vertexB = makeBytes(0x30u, 24u);
  const std::vector<std::byte> indexB = makeBytes(0x40u, 8u);
  const std::vector<std::byte> vertexC = makeBytes(0x50u, 24u);
  const std::vector<std::byte> indexC = makeBytes(0x60u, 8u);

  advanceFrame(gpu, pool, frameIndex);
  auto allocA = pool.allocate(vertexA, 3u, indexA, 6u, "a");
  ASSERT_FALSE(allocA.hasError()) << allocA.error();
  auto allocB = pool.allocate(vertexB, 3u, indexB, 6u, "b");
  ASSERT_FALSE(allocB.hasError()) << allocB.error();
  auto allocC = pool.allocate(vertexC, 3u, indexC, 6u, "c");
  ASSERT_FALSE(allocC.hasError()) << allocC.error();
  pool.release(allocC.value());

  advanceFrame(gpu, pool, ++frameIndex);
  pool.release(allocA.value());
  const uint64_t versionBeforeCommit = pool.mutationVersion();

  advanceFramesUntil(gpu, pool, frameIndex, 10u, [&pool, versionBeforeCommit] {
    return pool.mutationVersion() == versionBeforeCommit + 1u;
  });

  GeometryAllocationView viewB{};
  EXPECT_FALSE(pool.resolve(allocA.value(), viewB));
  ASSERT_TRUE(pool.resolve(allocB.value(), viewB));
  EXPECT_EQ(gpu.destroyedBufferCount, 0u);

  advanceFramesUntil(gpu, pool, frameIndex, 12u,
                     [&gpu] { return gpu.destroyedBufferCount > 0u; });
  EXPECT_GT(gpu.destroyedBufferCount, 0u);
}

TEST(GeometryPoolTests, CopySubmissionFailureAbortsWithoutCorruptingLiveState) {
  FakeExecutorGPUDevice gpu;
  GeometryPool pool(gpu, makeConfig());
  uint64_t frameIndex = 1u;

  const std::vector<std::byte> vertexA = makeBytes(0x10u, 24u);
  const std::vector<std::byte> indexA = makeBytes(0x20u, 8u);
  const std::vector<std::byte> vertexB = makeBytes(0x30u, 24u);
  const std::vector<std::byte> indexB = makeBytes(0x40u, 8u);

  advanceFrame(gpu, pool, frameIndex);
  auto allocA = pool.allocate(vertexA, 3u, indexA, 6u, "a");
  ASSERT_FALSE(allocA.hasError()) << allocA.error();
  auto allocB = pool.allocate(vertexB, 3u, indexB, 6u, "b");
  ASSERT_FALSE(allocB.hasError()) << allocB.error();
  pool.release(allocB.value());

  gpu.failBackgroundCopySubmitAtCall = 1u;
  bool sawError = false;
  for (uint32_t step = 0; step < 8u && !sawError; ++step) {
    auto gpuResult = gpu.beginFrame(++frameIndex);
    ASSERT_FALSE(gpuResult.hasError()) << gpuResult.error();
    auto poolResult = pool.beginFrame(frameIndex);
    sawError = poolResult.hasError();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_TRUE(sawError);

  GeometryAllocationView viewA{};
  ASSERT_TRUE(pool.resolve(allocA.value(), viewA));
  EXPECT_EQ(readBytes(gpu, viewA.vertexBuffer, viewA.vertexByteOffset,
                      viewA.vertexByteSize),
            vertexA);

  auto allocC = pool.allocate(vertexA, 3u, indexA, 6u, "c");
  EXPECT_FALSE(allocC.hasError()) << allocC.error();
}

TEST(GeometryPoolTests,
     ReplacementAllocationFailureAbortsWithoutCorruptingLiveState) {
  FakeExecutorGPUDevice gpu;
  GeometryPool pool(gpu, makeConfig());
  uint64_t frameIndex = 1u;

  const std::vector<std::byte> vertexA = makeBytes(0x10u, 24u);
  const std::vector<std::byte> indexA = makeBytes(0x20u, 8u);
  const std::vector<std::byte> vertexB = makeBytes(0x30u, 24u);
  const std::vector<std::byte> indexB = makeBytes(0x40u, 8u);

  advanceFrame(gpu, pool, frameIndex);
  auto allocA = pool.allocate(vertexA, 3u, indexA, 6u, "a");
  ASSERT_FALSE(allocA.hasError()) << allocA.error();
  auto allocB = pool.allocate(vertexB, 3u, indexB, 6u, "b");
  ASSERT_FALSE(allocB.hasError()) << allocB.error();
  pool.release(allocB.value());

  gpu.failCreateBufferAtCall = gpu.createdBufferCount + 1u;
  bool sawError = false;
  for (uint32_t step = 0; step < 8u && !sawError; ++step) {
    auto gpuResult = gpu.beginFrame(++frameIndex);
    ASSERT_FALSE(gpuResult.hasError()) << gpuResult.error();
    auto poolResult = pool.beginFrame(frameIndex);
    sawError = poolResult.hasError();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  ASSERT_TRUE(sawError);

  GeometryAllocationView viewA{};
  ASSERT_TRUE(pool.resolve(allocA.value(), viewA));
  EXPECT_EQ(readBytes(gpu, viewA.vertexBuffer, viewA.vertexByteOffset,
                      viewA.vertexByteSize),
            vertexA);
}

} // namespace
