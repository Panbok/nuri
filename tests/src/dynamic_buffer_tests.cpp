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
  DynamicBufferSlot slot;

  auto initial =
      ensureDynamicBufferCapacity(gpu, slot, storageBufferDesc(300u), "test");
  ASSERT_TRUE(initial.hasValue());
  EXPECT_TRUE(initial.value());
  ASSERT_NE(slot.buffer, nullptr);
  EXPECT_EQ(slot.capacityBytes, 512u);
  const BufferHandle initialHandle = slot.buffer->handle();

  auto adequate =
      ensureDynamicBufferCapacity(gpu, slot, storageBufferDesc(400u), "test");
  ASSERT_TRUE(adequate.hasValue());
  EXPECT_FALSE(adequate.value());
  EXPECT_EQ(gpu.createdBufferCount, 1u);
  EXPECT_EQ(gpu.destroyedBufferCount, 0u);

  auto grown =
      ensureDynamicBufferCapacity(gpu, slot, storageBufferDesc(513u), "test");
  ASSERT_TRUE(grown.hasValue());
  EXPECT_TRUE(grown.value());
  EXPECT_EQ(slot.capacityBytes, 768u);
  EXPECT_NE(slot.buffer->handle().index, initialHandle.index);
  EXPECT_FALSE(gpu.isValid(initialHandle));
  EXPECT_EQ(gpu.destroyedBufferCount, 1u);
  EXPECT_EQ(gpu.waitIdleCallCount, 0u);

  retireDynamicBuffer(gpu, slot);
  EXPECT_EQ(slot.buffer, nullptr);
  EXPECT_EQ(slot.capacityBytes, 0u);
  EXPECT_EQ(gpu.destroyedBufferCount, 2u);
  EXPECT_EQ(gpu.waitIdleCallCount, 0u);
}

TEST(DynamicBufferTest, FailedGrowthPreservesPublishedBuffer) {
  FailingBufferGPUDevice gpu;
  DynamicBufferSlot slot;

  auto initial =
      ensureDynamicBufferCapacity(gpu, slot, storageBufferDesc(256u), "test");
  ASSERT_TRUE(initial.hasValue());
  const BufferHandle initialHandle = slot.buffer->handle();
  const size_t initialCapacity = slot.capacityBytes;

  gpu.failCreateBufferAtCall = gpu.createBufferCallCount + 1u;
  auto failed = ensureDynamicBufferCapacity(
      gpu, slot, storageBufferDesc(initialCapacity + 1u), "test");
  ASSERT_TRUE(failed.hasError());
  ASSERT_NE(slot.buffer, nullptr);
  EXPECT_EQ(slot.buffer->handle().index, initialHandle.index);
  EXPECT_EQ(slot.buffer->handle().generation, initialHandle.generation);
  EXPECT_EQ(slot.capacityBytes, initialCapacity);
  EXPECT_TRUE(gpu.isValid(initialHandle));
  EXPECT_EQ(gpu.destroyedBufferCount, 0u);
  EXPECT_EQ(gpu.waitIdleCallCount, 0u);

  retireDynamicBuffer(gpu, slot);
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
