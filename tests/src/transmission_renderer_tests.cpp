#include "tests_pch.h"

#include "render_graph_test_support.h"

#include <gtest/gtest.h>

#define private public
#include "nuri/gfx/renderers/transmission_renderer.h"
#undef private

#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/resources/gpu/texture.h"
#include "nuri/scene/render_scene.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>

namespace {

using namespace nuri;
using namespace nuri::test_support;

class FakeTransmissionPrepareGpuDevice final : public FakeGPUDeviceBase {
public:
  Result<ShaderHandle, std::string>
  createShaderModule(const ShaderDesc &) override {
    return Result<ShaderHandle, std::string>::makeResult(
        ShaderHandle{.index = nextShaderIndex_++, .generation = 1u});
  }

  Result<RenderPipelineHandle, std::string>
  createRenderPipeline(const RenderPipelineDesc &, std::string_view) override {
    return Result<RenderPipelineHandle, std::string>::makeResult(
        RenderPipelineHandle{.index = nextPipelineIndex_++, .generation = 1u});
  }

  Result<ComputePipelineHandle, std::string>
  createComputePipeline(const ComputePipelineDesc &,
                        std::string_view) override {
    return Result<ComputePipelineHandle, std::string>::makeResult(
        ComputePipelineHandle{.index = nextComputePipelineIndex_++,
                              .generation = 1u});
  }

  Result<GeometryAllocationHandle, std::string>
  allocateGeometry(std::span<const std::byte> vertexBytes, uint32_t vertexCount,
                   std::span<const std::byte> indexBytes, uint32_t indexCount,
                   std::string_view) override {
    auto vertexBufferResult = createBufferImpl(BufferDesc{
        .usage = BufferUsage::Vertex | BufferUsage::Storage,
        .storage = Storage::Device,
        .size = vertexBytes.size(),
        .data = vertexBytes,
    });
    if (vertexBufferResult.hasError()) {
      return Result<GeometryAllocationHandle, std::string>::makeError(
          vertexBufferResult.error());
    }
    auto indexBufferResult = createBufferImpl(BufferDesc{
        .usage = BufferUsage::Index,
        .storage = Storage::Device,
        .size = indexBytes.size(),
        .data = indexBytes,
    });
    if (indexBufferResult.hasError()) {
      destroyBuffer(vertexBufferResult.value());
      return Result<GeometryAllocationHandle, std::string>::makeError(
          indexBufferResult.error());
    }

    const GeometryAllocationHandle handle{
        .index = static_cast<uint32_t>(allocations_.size() + 1u),
        .generation = 1u,
    };
    allocations_.push_back(Allocation{
        .handle = handle,
        .view =
            GeometryAllocationView{
                .vertexBuffer = vertexBufferResult.value(),
                .vertexByteOffset = 0u,
                .vertexByteSize = vertexBytes.size(),
                .indexBuffer = indexBufferResult.value(),
                .indexByteOffset = 0u,
                .indexByteSize = indexBytes.size(),
                .vertexCount = vertexCount,
                .indexCount = indexCount,
            },
    });
    ++geometryMutationVersion_;
    return Result<GeometryAllocationHandle, std::string>::makeResult(handle);
  }

  void releaseGeometry(GeometryAllocationHandle handle) override {
    for (Allocation &allocation : allocations_) {
      if (allocation.handle.index != handle.index ||
          allocation.handle.generation != handle.generation) {
        continue;
      }
      if (nuri::isValid(allocation.view.vertexBuffer)) {
        destroyBuffer(allocation.view.vertexBuffer);
        allocation.view.vertexBuffer = {};
      }
      if (nuri::isValid(allocation.view.indexBuffer)) {
        destroyBuffer(allocation.view.indexBuffer);
        allocation.view.indexBuffer = {};
      }
      allocation.handle = {};
      ++geometryMutationVersion_;
      return;
    }
  }

  bool resolveGeometry(GeometryAllocationHandle handle,
                       GeometryAllocationView &out) const override {
    for (const Allocation &allocation : allocations_) {
      if (allocation.handle.index == handle.index &&
          allocation.handle.generation == handle.generation) {
        out = allocation.view;
        return true;
      }
    }
    return false;
  }

  uint64_t geometryMutationVersion() const override {
    return geometryMutationVersion_;
  }

  void bumpGeometryMutationVersion() { ++geometryMutationVersion_; }

private:
  struct Allocation {
    GeometryAllocationHandle handle{};
    GeometryAllocationView view{};
  };

  std::vector<Allocation> allocations_{};
  uint64_t geometryMutationVersion_ = 0u;
  uint32_t nextShaderIndex_ = 1u;
  uint32_t nextPipelineIndex_ = 1u;
  uint32_t nextComputePipelineIndex_ = 1u;
};

std::unique_ptr<Texture> createTestTexture(GPUDevice &gpu, Format format,
                                           TextureUsage usage,
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

TransmissionRendererConfig makeTransmissionConfig() {
  const std::filesystem::path shaders =
      std::filesystem::path(PROJECT_SOURCE_DIR) / "assets" / "shaders";
  return TransmissionRendererConfig{
      .meshFragment = shaders / "main.frag",
  };
}

MeshData makeTransmissionTriangleMesh(std::pmr::memory_resource *memory) {
  MeshData mesh(memory);
  mesh.vertices.resize(3u);
  mesh.vertices[0].position = glm::vec3(-0.5f, -0.5f, 0.0f);
  mesh.vertices[0].normal = glm::vec3(0.0f, 0.0f, 1.0f);
  mesh.vertices[1].position = glm::vec3(0.5f, -0.5f, 0.0f);
  mesh.vertices[1].normal = glm::vec3(0.0f, 0.0f, 1.0f);
  mesh.vertices[2].position = glm::vec3(0.0f, 0.5f, 0.0f);
  mesh.vertices[2].normal = glm::vec3(0.0f, 0.0f, 1.0f);
  mesh.indices = {0u, 1u, 2u};

  Submesh submesh{};
  submesh.vertexOffset = 0u;
  submesh.vertexCount = 3u;
  submesh.indexOffset = 0u;
  submesh.indexCount = 3u;
  submesh.bounds =
      BoundingBox(glm::vec3(-0.5f, -0.5f, 0.0f), glm::vec3(0.5f, 0.5f, 0.0f));
  submesh.authoredScale = glm::vec3(1.0f);
  submesh.lodCount = 1u;
  submesh.lods[0] = SubmeshLod{.indexOffset = 0u, .indexCount = 3u};
  mesh.submeshes.push_back(submesh);
  return mesh;
}

MaterialRequest makeTransmissionMaterialRequest(float transmissionFactor,
                                                bool doubleSided = false) {
  MaterialRequest request{};
  request.desc.featureMask =
      kMaterialFeatureMetallicRoughness | kMaterialFeatureTransmission;
  request.desc.transmissionFactor = transmissionFactor;
  request.desc.thicknessFactor = 0.1f;
  request.desc.alphaMode = MaterialAlphaMode::Mask;
  request.desc.doubleSided = doubleSided;
  return request;
}

BufferHandle createStorageBuffer(GPUDevice &gpu,
                                 std::span<const std::byte> data,
                                 std::string_view debugName) {
  const BufferDesc desc{
      .usage = BufferUsage::Storage,
      .storage = Storage::Device,
      .size = std::max<size_t>(data.size(), 1u),
      .data = data,
  };
  auto result = gpu.createBuffer(desc, debugName);
  EXPECT_FALSE(result.hasError()) << result.error();
  return result.hasError() ? BufferHandle{} : result.value();
}

template <typename T>
BufferHandle createStorageTableBuffer(GPUDevice &gpu, std::span<const T> rows,
                                      std::string_view debugName) {
  std::span<const std::byte> bytes{};
  if (!rows.empty()) {
    bytes = std::span<const std::byte>{
        reinterpret_cast<const std::byte *>(rows.data()),
        rows.size() * sizeof(T)};
  }
  return createStorageBuffer(gpu, bytes, debugName);
}

MaterialTableGpuData
createMaterialTableGpuData(GPUDevice &gpu,
                           const MaterialTableSnapshot &snapshot) {
  MaterialTableGpuData data{};
  data.headerBuffer =
      createStorageTableBuffer(gpu, snapshot.headers, "material_headers");
  data.clearcoatBuffer =
      createStorageTableBuffer(gpu, snapshot.clearcoat, "material_clearcoat");
  data.sheenBuffer =
      createStorageTableBuffer(gpu, snapshot.sheen, "material_sheen");
  data.transmissionBuffer = createStorageTableBuffer(gpu, snapshot.transmission,
                                                     "material_transmission");
  data.specularBuffer =
      createStorageTableBuffer(gpu, snapshot.specular, "material_specular");
  data.headerBufferAddress = gpu.getBufferDeviceAddress(data.headerBuffer);
  data.clearcoatBufferAddress =
      gpu.getBufferDeviceAddress(data.clearcoatBuffer);
  data.sheenBufferAddress = gpu.getBufferDeviceAddress(data.sheenBuffer);
  data.transmissionBufferAddress =
      gpu.getBufferDeviceAddress(data.transmissionBuffer);
  data.specularBufferAddress = gpu.getBufferDeviceAddress(data.specularBuffer);
  data.version = snapshot.version;
  return data;
}

ForwardSceneGpuData
createForwardSceneGpuData(GPUDevice &gpu,
                          const MaterialTableGpuData &materialGpu,
                          TextureHandle sceneColorTexture) {
  ForwardSceneFrameData frameData{};
  frameData.flags = 1u << 5u;
  frameData.sceneColorTexId = gpu.getTextureBindlessIndex(sceneColorTexture);
  frameData.sceneColorSamplerId = gpu.getLinearRepeatSamplerBindlessIndex(true);
  frameData.materialSamplerId = gpu.getDefaultSamplerBindlessIndex();
  frameData.materialHeaderBufferAddress = materialGpu.headerBufferAddress;
  frameData.materialClearcoatBufferAddress = materialGpu.clearcoatBufferAddress;
  frameData.materialSheenBufferAddress = materialGpu.sheenBufferAddress;
  frameData.materialTransmissionBufferAddress =
      materialGpu.transmissionBufferAddress;
  frameData.materialSpecularBufferAddress = materialGpu.specularBufferAddress;
  const std::span<const std::byte> frameDataBytes{
      reinterpret_cast<const std::byte *>(&frameData), sizeof(frameData)};
  const BufferHandle buffer =
      createStorageBuffer(gpu, frameDataBytes, "forward_scene_frame");
  return ForwardSceneGpuData{
      .buffer = buffer,
      .frameData = frameData,
      .postTaaFrameData = {},
      .frameDataAddress = gpu.getBufferDeviceAddress(buffer),
      .postTaaFrameDataAddress = 0u,
  };
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

TEST(TransmissionRendererTest,
     RepeatedPrepareReusesStaticDrawTemplateAddresses) {
  std::array<std::byte, 256 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeTransmissionPrepareGpuDevice gpu;
  ResourceManager resources(gpu, &memory);
  RenderScene scene(&memory);

  MeshData mesh = makeTransmissionTriangleMesh(&memory);
  auto modelResult =
      resources.acquireGeneratedModel(mesh, "transmission_prepare_triangle");
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();
  auto materialResult =
      resources.acquireMaterial(makeTransmissionMaterialRequest(0.65f));
  ASSERT_FALSE(materialResult.hasError()) << materialResult.error();

  NodeId firstNode{};
  RenderableId firstRenderable{};
  constexpr uint32_t kRenderableCount = 4u;
  for (uint32_t i = 0u; i < kRenderableCount; ++i) {
    const glm::mat4 transform = glm::translate(
        glm::mat4(1.0f), glm::vec3(static_cast<float>(i) * 2.0f, 0.0f, 0.0f));
    auto nodeResult = scene.graph().createNode(scene.graph().rootNode(),
                                               "transmission_node", transform);
    ASSERT_FALSE(nodeResult.hasError()) << nodeResult.error();
    auto renderableResult = scene.graph().addRenderable(
        nodeResult.value(), modelResult.value(), materialResult.value());
    ASSERT_FALSE(renderableResult.hasError()) << renderableResult.error();
    if (i == 0u) {
      firstNode = nodeResult.value();
      firstRenderable = renderableResult.value();
    }
  }
  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();

  auto sceneColorTexture = createTestTexture(
      gpu, Format::RGBA16_FLOAT, TextureUsage::AttachmentSampled,
      "transmission_prepare_scene_color");
  auto frameColorTexture = createTestTexture(
      gpu, Format::RGBA16_FLOAT, TextureUsage::AttachmentSampled,
      "transmission_prepare_frame_color");
  ASSERT_NE(sceneColorTexture, nullptr);
  ASSERT_NE(frameColorTexture, nullptr);

  RenderSettings settings{};
  MaterialTableGpuData materialGpu =
      createMaterialTableGpuData(gpu, resources.materialSnapshot());
  ForwardSceneGpuData sceneGpu =
      createForwardSceneGpuData(gpu, materialGpu, sceneColorTexture->handle());

  RenderFrameContext frame{};
  frame.scene = &scene;
  frame.resources = &resources;
  frame.settings = &settings;
  frame.sharedResources.sceneColorTexture = sceneColorTexture->handle();
  frame.sharedResources.frameColorTexture = frameColorTexture->handle();
  frame.sharedResources.materialTableGpuData = materialGpu;
  frame.sharedResources.forwardSceneGpuData = sceneGpu;

  TransmissionRenderer renderer(gpu, makeTransmissionConfig(), &memory);
  auto firstPrepare = renderer.prepareTransmissionPasses(frame);
  ASSERT_FALSE(firstPrepare.hasError()) << firstPrepare.error();
  ASSERT_TRUE(firstPrepare.value());
  ASSERT_EQ(renderer.meshDrawTemplates_.size(), kRenderableCount);
  ASSERT_FALSE(renderer.passDrawItems_.empty());

  gpu.bufferDeviceAddressCallCount = 0u;
  frame.frameIndex = 1u;
  auto secondPrepare = renderer.prepareTransmissionPasses(frame);
  ASSERT_FALSE(secondPrepare.hasError()) << secondPrepare.error();
  EXPECT_TRUE(secondPrepare.value());
  EXPECT_LE(gpu.bufferDeviceAddressCallCount, 2u);

  ASSERT_TRUE(scene.graph().setNodeLocalTransform(
      firstNode, glm::scale(glm::mat4(1.0f), glm::vec3(2.0f, 3.0f, 4.0f))));
  commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();

  gpu.bufferDeviceAddressCallCount = 0u;
  frame.frameIndex = 2u;
  auto transformPrepare = renderer.prepareTransmissionPasses(frame);
  ASSERT_FALSE(transformPrepare.hasError()) << transformPrepare.error();
  ASSERT_TRUE(transformPrepare.value());
  EXPECT_NEAR(renderer.meshDrawTemplates_[0].transmissionScale.x, 2.0f,
              1.0e-5f);
  EXPECT_NEAR(renderer.meshDrawTemplates_[0].transmissionScale.y, 3.0f,
              1.0e-5f);
  EXPECT_NEAR(renderer.meshDrawTemplates_[0].transmissionScale.z, 4.0f,
              1.0e-5f);
  EXPECT_LE(gpu.bufferDeviceAddressCallCount, 2u);

  auto updatedMaterialResult =
      resources.acquireMaterial(makeTransmissionMaterialRequest(0.35f, true));
  ASSERT_FALSE(updatedMaterialResult.hasError())
      << updatedMaterialResult.error();
  ASSERT_TRUE(scene.graph().setRenderableMaterial(
      firstRenderable, updatedMaterialResult.value()));
  commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();
  materialGpu = createMaterialTableGpuData(gpu, resources.materialSnapshot());
  sceneGpu =
      createForwardSceneGpuData(gpu, materialGpu, sceneColorTexture->handle());
  frame.sharedResources.materialTableGpuData = materialGpu;
  frame.sharedResources.forwardSceneGpuData = sceneGpu;

  gpu.bufferDeviceAddressCallCount = 0u;
  frame.frameIndex = 3u;
  auto materialPrepare = renderer.prepareTransmissionPasses(frame);
  ASSERT_FALSE(materialPrepare.hasError()) << materialPrepare.error();
  ASSERT_TRUE(materialPrepare.value());
  EXPECT_GT(gpu.bufferDeviceAddressCallCount, 2u);
  EXPECT_EQ(renderer.cachedMaterialVersion_,
            resources.materialSnapshot().version);

  gpu.bumpGeometryMutationVersion();
  gpu.bufferDeviceAddressCallCount = 0u;
  frame.frameIndex = 4u;
  auto geometryPrepare = renderer.prepareTransmissionPasses(frame);
  ASSERT_FALSE(geometryPrepare.hasError()) << geometryPrepare.error();
  ASSERT_TRUE(geometryPrepare.value());
  EXPECT_GT(gpu.bufferDeviceAddressCallCount, 2u);
  EXPECT_EQ(renderer.cachedGeometryMutationVersion_,
            gpu.geometryMutationVersion());
}

TEST(TransmissionRendererTest, StableSortedTransmissionReusesSortDepthCache) {
  std::array<std::byte, 256 * 1024> scratchBytes{};
  std::pmr::monotonic_buffer_resource memory(scratchBytes.data(),
                                             scratchBytes.size());
  FakeTransmissionPrepareGpuDevice gpu;
  ResourceManager resources(gpu, &memory);
  RenderScene scene(&memory);

  MeshData mesh = makeTransmissionTriangleMesh(&memory);
  auto modelResult =
      resources.acquireGeneratedModel(mesh, "transmission_sorted_triangle");
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();
  MaterialRequest materialRequest = makeTransmissionMaterialRequest(0.65f);
  materialRequest.desc.alphaMode = MaterialAlphaMode::Blend;
  auto materialResult = resources.acquireMaterial(materialRequest);
  ASSERT_FALSE(materialResult.hasError()) << materialResult.error();

  NodeId firstNode{};
  constexpr uint32_t kRenderableCount = 3u;
  for (uint32_t i = 0u; i < kRenderableCount; ++i) {
    const glm::mat4 transform = glm::translate(
        glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -static_cast<float>(i)));
    auto nodeResult =
        scene.graph().createNode(scene.graph().rootNode(),
                                 "transmission_sorted_node", transform);
    ASSERT_FALSE(nodeResult.hasError()) << nodeResult.error();
    auto renderableResult = scene.graph().addRenderable(
        nodeResult.value(), modelResult.value(), materialResult.value());
    ASSERT_FALSE(renderableResult.hasError()) << renderableResult.error();
    if (i == 0u) {
      firstNode = nodeResult.value();
    }
  }
  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();

  auto sceneColorTexture = createTestTexture(
      gpu, Format::RGBA16_FLOAT, TextureUsage::AttachmentSampled,
      "transmission_sorted_scene_color");
  auto frameColorTexture = createTestTexture(
      gpu, Format::RGBA16_FLOAT, TextureUsage::AttachmentSampled,
      "transmission_sorted_frame_color");
  ASSERT_NE(sceneColorTexture, nullptr);
  ASSERT_NE(frameColorTexture, nullptr);

  RenderSettings settings{};
  MaterialTableGpuData materialGpu =
      createMaterialTableGpuData(gpu, resources.materialSnapshot());
  ForwardSceneGpuData sceneGpu =
      createForwardSceneGpuData(gpu, materialGpu, sceneColorTexture->handle());

  RenderFrameContext frame{};
  frame.scene = &scene;
  frame.resources = &resources;
  frame.settings = &settings;
  frame.camera.view = glm::mat4(1.0f);
  frame.sharedResources.sceneColorTexture = sceneColorTexture->handle();
  frame.sharedResources.frameColorTexture = frameColorTexture->handle();
  frame.sharedResources.materialTableGpuData = materialGpu;
  frame.sharedResources.forwardSceneGpuData = sceneGpu;

  TransmissionRenderer renderer(gpu, makeTransmissionConfig(), &memory);
  auto firstPrepare = renderer.prepareTransmissionPasses(frame);
  ASSERT_FALSE(firstPrepare.hasError()) << firstPrepare.error();
  ASSERT_TRUE(firstPrepare.value());
  ASSERT_EQ(renderer.blendedSortableDraws_.size(), kRenderableCount);
  ASSERT_EQ(renderer.cachedSortedDepths_.size(), kRenderableCount);

  renderer.cachedSortedDepths_[0] = 12345.0f;
  frame.frameIndex = 1u;
  auto stablePrepare = renderer.prepareTransmissionPasses(frame);
  ASSERT_FALSE(stablePrepare.hasError()) << stablePrepare.error();
  ASSERT_TRUE(stablePrepare.value());
  ASSERT_EQ(renderer.blendedSortableDraws_.size(), kRenderableCount);
  EXPECT_FLOAT_EQ(renderer.blendedSortableDraws_[0].sortDepth, 12345.0f);

  renderer.cachedSortedDepths_[0] = 54321.0f;
  frame.camera.view =
      glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -2.0f));
  frame.frameIndex = 2u;
  auto cameraPrepare = renderer.prepareTransmissionPasses(frame);
  ASSERT_FALSE(cameraPrepare.hasError()) << cameraPrepare.error();
  ASSERT_TRUE(cameraPrepare.value());
  EXPECT_NE(renderer.blendedSortableDraws_[0].sortDepth, 54321.0f);

  renderer.cachedSortedDepths_[0] = 77777.0f;
  ASSERT_TRUE(scene.graph().setNodeLocalTransform(
      firstNode, glm::translate(glm::mat4(1.0f),
                                glm::vec3(5.0f, 0.0f, -4.0f))));
  commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();
  frame.frameIndex = 3u;
  auto transformPrepare = renderer.prepareTransmissionPasses(frame);
  ASSERT_FALSE(transformPrepare.hasError()) << transformPrepare.error();
  ASSERT_TRUE(transformPrepare.value());
  EXPECT_NE(renderer.blendedSortableDraws_[0].sortDepth, 77777.0f);
}

} // namespace
