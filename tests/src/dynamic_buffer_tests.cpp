#include "nuri/pch.h"

#include "nuri/gfx/dynamic_buffer.h"
#include "nuri/resources/gpu/texture.h"
#include "render_graph_test_support.h"

#include <gtest/gtest.h>

namespace nuri::test {
namespace {

class FailingBufferGPUDevice final : public test_support::FakeGPUDeviceBase {
public:
  Result<BufferHandle, std::string>
  createBuffer(const BufferDesc &desc, std::string_view debugName) override {
    ++createBufferCallCount;
    if (failCreateBufferAtCall != 0u &&
        createBufferCallCount == failCreateBufferAtCall) {
      return Result<BufferHandle, std::string>::makeError(
          "injected buffer allocation failure");
    }
    return FakeGPUDeviceBase::createBuffer(desc, debugName);
  }

  uint32_t createBufferCallCount = 0u;
  uint32_t failCreateBufferAtCall = 0u;
};

class OwnedAliasGPUDevice final : public test_support::FakeGPUDeviceBase {
public:
  void destroyShaderModule(ShaderHandle) override { ++destroyedShaders; }
  void destroyRenderPipeline(RenderPipelineHandle) override {
    ++destroyedRenderPipelines;
  }
  void destroyComputePipeline(ComputePipelineHandle) override {
    ++destroyedComputePipelines;
  }
  void destroyMeshletPipeline(MeshletPipelineHandle) override {
    ++destroyedMeshletPipelines;
  }

  uint32_t destroyedShaders = 0u;
  uint32_t destroyedRenderPipelines = 0u;
  uint32_t destroyedComputePipelines = 0u;
  uint32_t destroyedMeshletPipelines = 0u;
};

BufferDesc storageBufferDesc(size_t size) {
  return BufferDesc{
      .usage = BufferUsage::Storage,
      .storage = Storage::Device,
      .size = size,
  };
}

TEST(DynamicBufferTest, GrowthIsAlignedGeometricAndNeverWaitsForIdle) {
  FailingBufferGPUDevice gpu;
  std::unique_ptr<Buffer> buffer;
  size_t capacityBytes = 0u;

  auto initial = ensureDynamicBufferCapacity(gpu, buffer, capacityBytes,
                                             storageBufferDesc(300u), "test");
  ASSERT_TRUE(initial.hasValue());
  EXPECT_TRUE(initial.value());
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(capacityBytes, 512u);
  const BufferHandle initialHandle = buffer->handle();

  auto adequate = ensureDynamicBufferCapacity(gpu, buffer, capacityBytes,
                                              storageBufferDesc(400u), "test");
  ASSERT_TRUE(adequate.hasValue());
  EXPECT_FALSE(adequate.value());
  EXPECT_EQ(gpu.createdBufferCount, 1u);
  EXPECT_EQ(gpu.destroyedBufferCount, 0u);

  auto grown = ensureDynamicBufferCapacity(gpu, buffer, capacityBytes,
                                           storageBufferDesc(513u), "test");
  ASSERT_TRUE(grown.hasValue());
  EXPECT_TRUE(grown.value());
  EXPECT_EQ(capacityBytes, 768u);
  EXPECT_NE(buffer->handle().index, initialHandle.index);
  EXPECT_FALSE(gpu.isValid(initialHandle));
  EXPECT_EQ(gpu.destroyedBufferCount, 1u);
  EXPECT_EQ(gpu.waitIdleCallCount, 0u);

  retireDynamicBuffer(buffer, capacityBytes);
  EXPECT_EQ(buffer, nullptr);
  EXPECT_EQ(capacityBytes, 0u);
  EXPECT_EQ(gpu.destroyedBufferCount, 2u);
  EXPECT_EQ(gpu.waitIdleCallCount, 0u);
}

TEST(DynamicBufferTest, FailedGrowthPreservesPublishedBuffer) {
  FailingBufferGPUDevice gpu;
  std::unique_ptr<Buffer> buffer;
  size_t capacityBytes = 0u;

  auto initial = ensureDynamicBufferCapacity(gpu, buffer, capacityBytes,
                                             storageBufferDesc(256u), "test");
  ASSERT_TRUE(initial.hasValue());
  const BufferHandle initialHandle = buffer->handle();
  const size_t initialCapacity = capacityBytes;

  gpu.failCreateBufferAtCall = gpu.createBufferCallCount + 1u;
  auto failed = ensureDynamicBufferCapacity(
      gpu, buffer, capacityBytes, storageBufferDesc(initialCapacity + 1u),
      "test");
  ASSERT_TRUE(failed.hasError());
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(buffer->handle().index, initialHandle.index);
  EXPECT_EQ(buffer->handle().generation, initialHandle.generation);
  EXPECT_EQ(capacityBytes, initialCapacity);
  EXPECT_TRUE(gpu.isValid(initialHandle));
  EXPECT_EQ(gpu.destroyedBufferCount, 0u);
  EXPECT_EQ(gpu.waitIdleCallCount, 0u);

  retireDynamicBuffer(buffer, capacityBytes);
}

TEST(DynamicBufferRingTest, BusyPreferredLaneGrowsUntilSubmissionCompletes) {
  FailingBufferGPUDevice gpu;
  DynamicBufferRing ring(gpu, storageBufferDesc(0u), "ring");

  ASSERT_TRUE(gpu.beginFrame(0u).hasValue());
  auto first = ring.acquire(0u, 256u, 1u);
  ASSERT_TRUE(first.hasValue());
  auto firstCompletion = gpu.captureWorkCompletion();
  ASSERT_TRUE(firstCompletion.hasValue());
  ring.submitPrepared(firstCompletion.value());

  auto second = ring.acquire(1u, 256u, 1u);
  ASSERT_TRUE(second.hasValue());
  EXPECT_EQ(second.value().lane, 1u);
  EXPECT_NE(second.value().buffer.index, first.value().buffer.index);
  auto secondCompletion = gpu.captureWorkCompletion();
  ASSERT_TRUE(secondCompletion.hasValue());
  ring.submitPrepared(secondCompletion.value());

  ASSERT_TRUE(gpu.beginFrame(100u).hasValue());
  auto reused = ring.acquire(2u, 128u, 1u);
  ASSERT_TRUE(reused.hasValue());
  EXPECT_EQ(reused.value().lane, 0u);
  EXPECT_EQ(reused.value().buffer.index, first.value().buffer.index);
  EXPECT_FALSE(reused.value().replaced);
  ring.abandonPrepared();
}

TEST(DynamicBufferRoleRingTest,
     KeepsRolesAlignedAndReusesOnlyCompletedSubmissionLanes) {
  FailingBufferGPUDevice gpu;
  DynamicBufferRoleRing ring(gpu, 2u);
  ASSERT_TRUE(ring.ensureLaneCount(1u).hasValue());
  ASSERT_TRUE(
      ring.ensureRole(0u, storageBufferDesc(256u), "role_zero").hasValue());
  ASSERT_TRUE(
      ring.ensureRole(1u, storageBufferDesc(512u), "role_one").hasValue());

  ASSERT_TRUE(gpu.beginFrame(0u).hasValue());
  auto first = ring.acquire(0u, 1u);
  ASSERT_TRUE(first.hasValue());
  EXPECT_EQ(first.value().lane, 0u);
  const BufferHandle firstRoleZero = ring.handle(0u, first.value().lane);
  const BufferHandle firstRoleOne = ring.handle(1u, first.value().lane);
  ASSERT_TRUE(nuri::isValid(firstRoleZero));
  ASSERT_TRUE(nuri::isValid(firstRoleOne));
  auto firstCompletion = gpu.captureWorkCompletion();
  ASSERT_TRUE(firstCompletion.hasValue());
  ring.submitPrepared(firstCompletion.value());

  auto second = ring.acquire(1u, 1u);
  ASSERT_TRUE(second.hasValue());
  EXPECT_EQ(second.value().lane, 1u);
  EXPECT_NE(ring.handle(0u, second.value().lane).index, firstRoleZero.index);
  EXPECT_NE(ring.handle(1u, second.value().lane).index, firstRoleOne.index);
  ring.abandonPrepared();

  ASSERT_TRUE(gpu.beginFrame(100u).hasValue());
  EXPECT_TRUE(ring.completed(0u));
  auto reused = ring.acquire(2u, 1u);
  ASSERT_TRUE(reused.hasValue());
  EXPECT_EQ(reused.value().lane, 0u);
  EXPECT_EQ(ring.handle(0u, reused.value().lane).index, firstRoleZero.index);
  EXPECT_EQ(ring.handle(1u, reused.value().lane).index, firstRoleOne.index);
  ring.abandonPrepared();
}

TEST(OwnedBufferTest, ScopeExitDestroysExactlyOnce) {
  FailingBufferGPUDevice gpu;

  {
    auto buffer = Buffer::create(gpu, storageBufferDesc(256u), "owned");
    ASSERT_TRUE(buffer.hasValue());
    ASSERT_TRUE(buffer.value()->valid());
    EXPECT_EQ(gpu.destroyedBufferCount, 0u);
  }

  EXPECT_EQ(gpu.createdBufferCount, 1u);
  EXPECT_EQ(gpu.destroyedBufferCount, 1u);
}

TEST(OwnedBufferTest, MoveTransfersSingleDestruction) {
  FailingBufferGPUDevice gpu;
  auto raw = gpu.createBuffer(storageBufferDesc(256u), "owned");
  ASSERT_TRUE(raw.hasValue());

  {
    OwnedBufferHandle source(gpu, raw.value());
    OwnedBufferHandle destination(std::move(source));
    EXPECT_FALSE(source.valid());
    EXPECT_TRUE(destination.valid());

    OwnedBufferHandle finalOwner;
    finalOwner = std::move(destination);
    EXPECT_FALSE(destination.valid());
    EXPECT_TRUE(finalOwner.valid());
    EXPECT_EQ(gpu.destroyedBufferCount, 0u);
  }

  EXPECT_EQ(gpu.destroyedBufferCount, 1u);
}

TEST(OwnedBufferTest, ReleaseTransfersWithoutDestroying) {
  FailingBufferGPUDevice gpu;
  auto raw = gpu.createBuffer(storageBufferDesc(256u), "owned");
  ASSERT_TRUE(raw.hasValue());

  BufferHandle released{};
  {
    OwnedBufferHandle owner(gpu, raw.value());
    released = owner.release();
    EXPECT_FALSE(owner.valid());
  }

  EXPECT_TRUE(gpu.isValid(released));
  EXPECT_EQ(gpu.destroyedBufferCount, 0u);
  gpu.destroyBuffer(released);
  EXPECT_EQ(gpu.destroyedBufferCount, 1u);
}

TEST(OwnedTextureTest, ScopeExitDestroysExactlyOnce) {
  FailingBufferGPUDevice gpu;

  {
    auto texture = Texture::create(gpu,
                                   TextureDesc{.format = Format::RGBA8_UNORM,
                                               .dimensions = {4u, 4u, 1u},
                                               .usage = TextureUsage::Sampled,
                                               .storage = Storage::Device},
                                   "owned_texture");
    ASSERT_TRUE(texture.hasValue());
    ASSERT_TRUE(texture.value()->valid());
    EXPECT_EQ(gpu.destroyedTextureCount, 0u);
  }

  EXPECT_EQ(gpu.createdTextureCount, 1u);
  EXPECT_EQ(gpu.destroyedTextureCount, 1u);
}

TEST(OwnedGpuResourceTest, ShaderAndPipelineAliasesDestroyExactlyOnce) {
  OwnedAliasGPUDevice gpu;

  {
    OwnedShaderHandle shader(gpu, ShaderHandle{.index = 1u, .generation = 1u});
    OwnedRenderPipelineHandle renderPipeline(
        gpu, RenderPipelineHandle{.index = 2u, .generation = 1u});
    OwnedComputePipelineHandle computePipeline(
        gpu, ComputePipelineHandle{.index = 3u, .generation = 1u});
    OwnedMeshletPipelineHandle meshletPipeline(
        gpu, MeshletPipelineHandle{.index = 4u, .generation = 1u});

    renderPipeline.reset();
    renderPipeline.reset();
    EXPECT_EQ(gpu.destroyedRenderPipelines, 1u);
    EXPECT_EQ(gpu.destroyedShaders, 0u);
    EXPECT_EQ(gpu.destroyedComputePipelines, 0u);
    EXPECT_EQ(gpu.destroyedMeshletPipelines, 0u);
  }

  EXPECT_EQ(gpu.destroyedShaders, 1u);
  EXPECT_EQ(gpu.destroyedRenderPipelines, 1u);
  EXPECT_EQ(gpu.destroyedComputePipelines, 1u);
  EXPECT_EQ(gpu.destroyedMeshletPipelines, 1u);
}

} // namespace
} // namespace nuri::test
