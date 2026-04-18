#include "tests_pch.h"

#include "render_graph_test_support.h"

#include <gtest/gtest.h>

#define private public
#include "nuri/gfx/renderers/transmission_renderer.h"
#undef private

#include "nuri/resources/gpu/texture.h"

#include <array>

namespace {

using namespace nuri;
using namespace nuri::test_support;

std::unique_ptr<Texture> createTestTexture(FakeRendererGPUDevice &gpu,
                                           Format format, TextureUsage usage,
                                           std::string_view debugName) {
  const TextureDesc desc{
      .type = TextureType::Texture2D,
      .format = format,
      .dimensions = {.width = 32u, .height = 32u, .depth = 1u},
      .usage = usage,
      .storage = Storage::Device,
      .numLayers = 1u,
      .numSamples = 1u,
      .numMipLevels = 1u,
      .data = {},
      .dataNumMipLevels = 1u,
      .generateMipmaps = false,
  };
  auto textureResult = Texture::create(gpu, desc, debugName);
  EXPECT_FALSE(textureResult.hasError()) << textureResult.error();
  if (textureResult.hasError()) {
    return {};
  }
  return std::move(textureResult.value());
}

Result<bool, std::string> compileAndExecute(RenderGraphBuilder &builder,
                                            FakeRendererGPUDevice &gpu,
                                            uint64_t frameIndex,
                                            std::pmr::memory_resource *memory) {
  RenderGraphRuntime runtime;
  auto compileResult = builder.compile(runtime);
  if (compileResult.hasError()) {
    return Result<bool, std::string>::makeError(compileResult.error());
  }

  auto beginResult = gpu.beginFrame(frameIndex);
  if (beginResult.hasError()) {
    return Result<bool, std::string>::makeError(beginResult.error());
  }

  RenderGraphExecutor executor(memory);
  auto executeResult = executor.execute(runtime, gpu, compileResult.value());
  if (executeResult.hasError()) {
    return Result<bool, std::string>::makeError(executeResult.error());
  }
  return Result<bool, std::string>::makeResult(true);
}

void seedPreparedTransmissionPass(TransmissionRenderer &renderer,
                                  TextureHandle frameColorTexture) {
  renderer.preparedFrameColorTexture_ = frameColorTexture;

  DrawItem passDraw{};
  passDraw.debugLabel = "Transmission Test Draw";
  renderer.passDrawItems_.push_back(passDraw);
}

TEST(TransmissionRendererTest,
     AppendTransmissionMainPassAddsMainPassWhenPrepared) {
  std::array<std::byte, 32 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeRendererGPUDevice gpu;
  TransmissionRenderer renderer(gpu, {}, &memory);

  auto frameColorTexture = createTestTexture(gpu, Format::RGBA8_UNORM,
                                             TextureUsage::AttachmentSampled,
                                             "transmission_test_frame_color");
  ASSERT_NE(frameColorTexture, nullptr);

  seedPreparedTransmissionPass(renderer, frameColorTexture->handle());

  RenderFrameContext frame{};
  frame.frameIndex = 0u;

  RenderGraphBuilder builder(&memory);
  builder.beginFrame(frame.frameIndex);
  auto appendResult = renderer.appendTransmissionMainPass(frame, builder);
  ASSERT_FALSE(appendResult.hasError()) << appendResult.error();
  ASSERT_TRUE(appendResult.value());

  auto executeResult =
      compileAndExecute(builder, gpu, frame.frameIndex, &memory);
  ASSERT_FALSE(executeResult.hasError()) << executeResult.error();
  ASSERT_TRUE(executeResult.value());

  ASSERT_EQ(gpu.submittedPassCount, 1u);
  ASSERT_EQ(gpu.submittedPassLabels.size(), 1u);
  EXPECT_EQ(gpu.submittedPassLabels[0], "Transmission Pass");
}

} // namespace
