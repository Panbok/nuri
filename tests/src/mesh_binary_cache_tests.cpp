#include "tests_pch.h"

#include "nuri/resources/gpu/model.h"
#include "nuri/resources/storage/mesh/mesh_binary_format.h"
#include "nuri/resources/storage/mesh/mesh_binary_serializer.h"
#include "nuri/resources/storage/mesh/mesh_cache_utils.h"
#include "render_graph_test_support.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <span>
#include <unordered_map>
#include <vector>

namespace {

struct MeshBinaryTestData {
  std::array<std::byte, nuri::kMeshBinaryStaticVertexStrideBytes * 3u>
      vertices{};
  std::array<nuri::StaticVertexDecodeGpuData, 1u> staticVertexDecode{};
  std::array<uint32_t, 3u> indices{0u, 1u, 2u};
  std::array<nuri::Submesh, 1u> submeshes{};
  std::array<nuri::MeshletDescriptor, 1u> meshlets{};
  std::array<uint32_t, 3u> meshletVertexIndices{0u, 1u, 2u};
  std::array<uint8_t, 3u> meshletPrimitiveIndices{0u, 1u, 2u};

  MeshBinaryTestData() {
    auto &submesh = submeshes[0];
    submesh.vertexOffset = 0u;
    submesh.vertexCount = 3u;
    submesh.indexOffset = 0u;
    submesh.indexCount = 3u;
    submesh.materialIndex = 4u;
    submesh.lodCount = 1u;
    submesh.lods[0] = nuri::SubmeshLod{
        .indexOffset = 0u,
        .indexCount = 3u,
        .meshletOffset = 0u,
        .meshletCount = 1u,
        .error = 0.125f,
    };

    meshlets[0] = nuri::MeshletDescriptor{
        .vertexOffset = 0u,
        .vertexCount = 3u,
        .primitiveOffset = 0u,
        .primitiveCount = 1u,
        .boundsSphere = {1.0f, 2.0f, 3.0f, 4.0f},
        .coneApex = {5.0f, 6.0f, 7.0f, 1.0f},
        .coneAxisCutoff = {0.0f, 1.0f, 0.0f, 0.5f},
    };
  }

  [[nodiscard]] nuri::MeshBinarySerializeInput
  makeInput(bool includeMeshlets) const {
    nuri::MeshBinarySerializeInput input{};
    input.sourcePathHash = 0x1020304050607080ull;
    input.importOptionsHash = 0x8877665544332211ull;
    input.sourceSizeBytes = 123u;
    input.sourceMtimeNs = 456;
    input.bounds = nuri::BoundingBox{glm::vec3(-1.0f), glm::vec3(1.0f)};
    input.vertexLayoutId = nuri::kMeshBinaryLayoutIdStaticQuantized20;
    input.vertices = nuri::BufferLayout<std::span<const std::byte>>{
        .data = std::span<const std::byte>(vertices.data(), vertices.size()),
        .count = 3u,
        .strideBytes = nuri::kMeshBinaryStaticVertexStrideBytes,
    };
    input.staticVertexDecode = nuri::BufferLayout<std::span<const std::byte>>{
        .data = std::as_bytes(std::span<const nuri::StaticVertexDecodeGpuData>(
            staticVertexDecode.data(), staticVertexDecode.size())),
        .count = static_cast<uint32_t>(staticVertexDecode.size()),
        .strideBytes = sizeof(nuri::StaticVertexDecodeGpuData),
    };
    input.indices = std::span<const uint32_t>(indices.data(), indices.size());
    input.submeshes =
        std::span<const nuri::Submesh>(submeshes.data(), submeshes.size());
    if (includeMeshlets) {
      input.meshlets = std::span<const nuri::MeshletDescriptor>(
          meshlets.data(), meshlets.size());
      input.meshletVertexIndices = std::span<const uint32_t>(
          meshletVertexIndices.data(), meshletVertexIndices.size());
      input.meshletPrimitiveIndices = std::span<const uint8_t>(
          meshletPrimitiveIndices.data(), meshletPrimitiveIndices.size());
    }
    return input;
  }
};

struct TestMeshletDescriptorGpu {
  glm::uvec4 offsetsCounts{0u};
  glm::vec4 boundsSphere{0.0f};
  glm::vec4 coneApex{0.0f};
  glm::vec4 coneAxisCutoff{0.0f};
};

static_assert(sizeof(TestMeshletDescriptorGpu) == 64);

class FakeMeshletUploadGpuDevice final
    : public nuri::test_support::FakeGPUDeviceBase {
public:
  nuri::Result<nuri::GeometryAllocationHandle, std::string>
  allocateGeometry(std::span<const std::byte> vertexBytes, uint32_t vertexCount,
                   std::span<const std::byte> indexBytes, uint32_t indexCount,
                   std::string_view) override {
    auto vertexBuffer = createBufferImpl(nuri::BufferDesc{
        .usage = nuri::BufferUsage::Vertex,
        .storage = nuri::Storage::Device,
        .size = vertexBytes.size(),
        .data = vertexBytes,
    });
    if (vertexBuffer.hasError()) {
      return nuri::Result<nuri::GeometryAllocationHandle,
                          std::string>::makeError(vertexBuffer.error());
    }

    auto indexBuffer = createBufferImpl(nuri::BufferDesc{
        .usage = nuri::BufferUsage::Index,
        .storage = nuri::Storage::Device,
        .size = indexBytes.size(),
        .data = indexBytes,
    });
    if (indexBuffer.hasError()) {
      destroyBufferImpl(vertexBuffer.value());
      return nuri::Result<nuri::GeometryAllocationHandle,
                          std::string>::makeError(indexBuffer.error());
    }

    const nuri::GeometryAllocationHandle handle{.index = nextGeometryIndex_++,
                                                .generation = 1u};
    geometries_.emplace(handle.index,
                        GeometryEntry{.generation = handle.generation,
                                      .view = nuri::GeometryAllocationView{
                                          .vertexBuffer = vertexBuffer.value(),
                                          .vertexByteOffset = 0u,
                                          .vertexByteSize = vertexBytes.size(),
                                          .indexBuffer = indexBuffer.value(),
                                          .indexByteOffset = 0u,
                                          .indexByteSize = indexBytes.size(),
                                          .vertexCount = vertexCount,
                                          .indexCount = indexCount,
                                      }});
    ++geometryMutationVersion_;
    return nuri::Result<nuri::GeometryAllocationHandle,
                        std::string>::makeResult(handle);
  }

  void releaseGeometry(nuri::GeometryAllocationHandle h) override {
    const auto it = geometries_.find(h.index);
    if (it == geometries_.end() || it->second.generation != h.generation) {
      return;
    }
    destroyBufferImpl(it->second.view.vertexBuffer);
    destroyBufferImpl(it->second.view.indexBuffer);
    geometries_.erase(it);
    ++geometryMutationVersion_;
  }

  bool resolveGeometry(nuri::GeometryAllocationHandle h,
                       nuri::GeometryAllocationView &out) const override {
    const auto it = geometries_.find(h.index);
    if (it == geometries_.end() || it->second.generation != h.generation) {
      return false;
    }
    out = it->second.view;
    return true;
  }

  uint64_t geometryMutationVersion() const override {
    return geometryMutationVersion_;
  }

private:
  struct GeometryEntry {
    uint32_t generation = 0u;
    nuri::GeometryAllocationView view{};
  };

  uint32_t nextGeometryIndex_ = 1u;
  uint64_t geometryMutationVersion_ = 1u;
  std::unordered_map<uint32_t, GeometryEntry> geometries_{};
};

nuri::MeshData makeMeshletUploadMesh() {
  nuri::MeshData mesh;
  mesh.vertices.resize(6u);
  for (uint32_t i = 0u; i < 6u; ++i) {
    mesh.vertices[i].position = glm::vec3(static_cast<float>(i), 0.0f, 0.0f);
    mesh.vertices[i].normal = glm::vec3(0.0f, 1.0f, 0.0f);
    mesh.vertices[i].tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
  }
  mesh.indices = {0u, 1u, 2u, 3u, 4u, 5u};

  mesh.submeshes.resize(1u);
  nuri::Submesh &submesh = mesh.submeshes[0];
  submesh.vertexOffset = 0u;
  submesh.vertexCount = 6u;
  submesh.indexOffset = 0u;
  submesh.indexCount = 6u;
  submesh.lodCount = 1u;
  submesh.lods[0] = nuri::SubmeshLod{
      .indexOffset = 0u,
      .indexCount = 6u,
      .meshletOffset = 0u,
      .meshletCount = 2u,
      .error = 0.5f,
  };

  mesh.meshlets.resize(2u);
  mesh.meshlets[0] = nuri::MeshletDescriptor{
      .vertexOffset = 0u,
      .vertexCount = 3u,
      .primitiveOffset = 0u,
      .primitiveCount = 1u,
      .boundsSphere = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f),
      .coneApex = glm::vec4(0.0f),
      .coneAxisCutoff = glm::vec4(0.0f, 1.0f, 0.0f, 0.25f),
  };
  mesh.meshlets[1] = nuri::MeshletDescriptor{
      .vertexOffset = 3u,
      .vertexCount = 3u,
      .primitiveOffset = 3u,
      .primitiveCount = 1u,
      .boundsSphere = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f),
      .coneApex = glm::vec4(1.0f),
      .coneAxisCutoff = glm::vec4(0.0f, 1.0f, 0.0f, 0.5f),
  };
  mesh.meshletVertexIndices = {0u, 1u, 2u, 3u, 4u, 5u};
  mesh.meshletPrimitiveIndices = {0u, 1u, 2u, 2u, 1u, 0u};
  return mesh;
}

nuri::MeshBinaryDeserializeContext
makeContext(const nuri::MeshBinarySerializeInput &input) {
  return nuri::MeshBinaryDeserializeContext{
      .expectedSourcePathHash = input.sourcePathHash,
      .expectedImportOptionsHash = input.importOptionsHash,
      .validateSourceFingerprint = true,
      .sourceExists = true,
      .sourceSizeBytes = input.sourceSizeBytes,
      .sourceMtimeNs = input.sourceMtimeNs,
  };
}

} // namespace

TEST(MeshBinaryCacheTests, MeshletSectionsRoundTrip) {
  const MeshBinaryTestData data;
  const nuri::MeshBinarySerializeInput input = data.makeInput(true);

  auto serializeResult = nuri::meshBinarySerialize(input);
  ASSERT_FALSE(serializeResult.hasError()) << serializeResult.error();

  auto deserializeResult =
      nuri::meshBinaryDeserialize(serializeResult.value(), makeContext(input));
  ASSERT_FALSE(deserializeResult.hasError())
      << deserializeResult.error().message;

  const nuri::MeshBinaryDecodedMesh &decoded = deserializeResult.value();
  ASSERT_EQ(decoded.meshlets.size(), 1u);
  EXPECT_EQ(decoded.meshletVertexIndices, std::vector<uint32_t>({0u, 1u, 2u}));
  EXPECT_EQ(decoded.meshletPrimitiveIndices,
            std::vector<uint8_t>({0u, 1u, 2u}));
  EXPECT_EQ(decoded.meshlets[0].vertexOffset, 0u);
  EXPECT_EQ(decoded.meshlets[0].vertexCount, 3u);
  EXPECT_EQ(decoded.meshlets[0].primitiveOffset, 0u);
  EXPECT_EQ(decoded.meshlets[0].primitiveCount, 1u);
  EXPECT_FLOAT_EQ(decoded.meshlets[0].boundsSphere.w, 4.0f);
  EXPECT_FLOAT_EQ(decoded.meshlets[0].coneAxisCutoff.w, 0.5f);

  ASSERT_EQ(decoded.submeshes.size(), 1u);
  EXPECT_EQ(decoded.submeshes[0].lods[0].meshletOffset, 0u);
  EXPECT_EQ(decoded.submeshes[0].lods[0].meshletCount, 1u);
  EXPECT_FLOAT_EQ(decoded.submeshes[0].lods[0].error, 0.125f);
}

TEST(MeshBinaryCacheTests, CacheWithoutMeshletSectionsStillLoads) {
  MeshBinaryTestData data;
  data.submeshes[0].lods[0].meshletCount = 0u;
  const nuri::MeshBinarySerializeInput input = data.makeInput(false);

  auto serializeResult = nuri::meshBinarySerialize(input);
  ASSERT_FALSE(serializeResult.hasError()) << serializeResult.error();

  auto deserializeResult =
      nuri::meshBinaryDeserialize(serializeResult.value(), makeContext(input));
  ASSERT_FALSE(deserializeResult.hasError())
      << deserializeResult.error().message;

  const nuri::MeshBinaryDecodedMesh &decoded = deserializeResult.value();
  EXPECT_TRUE(decoded.meshlets.empty());
  EXPECT_TRUE(decoded.meshletVertexIndices.empty());
  EXPECT_TRUE(decoded.meshletPrimitiveIndices.empty());
  ASSERT_EQ(decoded.submeshes.size(), 1u);
  EXPECT_EQ(decoded.submeshes[0].lods[0].meshletOffset, 0u);
  EXPECT_EQ(decoded.submeshes[0].lods[0].meshletCount, 0u);
}

TEST(MeshBinaryCacheTests, MeshletImportOptionsAffectCacheKey) {
  nuri::MeshImportOptions baseOptions{};
  nuri::MeshImportOptions meshletOptions = baseOptions;
  meshletOptions.generateMeshlets = true;

  nuri::MeshImportOptions differentSizeOptions = meshletOptions;
  differentSizeOptions.meshletMaxVertices = 32u;

  nuri::MeshImportOptions differentConeOptions = meshletOptions;
  differentConeOptions.meshletConeWeight = 0.25f;

  EXPECT_NE(nuri::hashMeshImportOptions(baseOptions),
            nuri::hashMeshImportOptions(meshletOptions));
  EXPECT_NE(nuri::hashMeshImportOptions(meshletOptions),
            nuri::hashMeshImportOptions(differentSizeOptions));
  EXPECT_NE(nuri::hashMeshImportOptions(meshletOptions),
            nuri::hashMeshImportOptions(differentConeOptions));
}

TEST(MeshBinaryCacheTests, ModelUploadPacksMeshletGpuBuffersForShaders) {
  FakeMeshletUploadGpuDevice gpu;
  nuri::MeshData mesh = makeMeshletUploadMesh();

  auto modelResult = nuri::Model::create(gpu, mesh, "meshlet_gpu_upload");
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();

  const nuri::Model &model = *modelResult.value();
  ASSERT_TRUE(model.hasMeshlets());
  const nuri::Model::ModelMeshletGpuView &view = model.meshletGpuView();
  EXPECT_EQ(view.meshletCount, 2u);
  EXPECT_EQ(view.meshletVertexIndexCount, 6u);
  EXPECT_EQ(view.meshletPrimitiveIndexCount, 2u);

  std::array<TestMeshletDescriptorGpu, 2u> descriptors{};
  auto descriptorReadResult =
      gpu.readBuffer(view.meshletBuffer, 0u,
                     std::as_writable_bytes(std::span<TestMeshletDescriptorGpu>(
                         descriptors.data(), descriptors.size())));
  ASSERT_FALSE(descriptorReadResult.hasError()) << descriptorReadResult.error();

  EXPECT_EQ(descriptors[0].offsetsCounts.x, 0u);
  EXPECT_EQ(descriptors[0].offsetsCounts.y, 0u);
  EXPECT_EQ(descriptors[0].offsetsCounts.z, 3u);
  EXPECT_EQ(descriptors[0].offsetsCounts.w, 1u);
  EXPECT_EQ(descriptors[1].offsetsCounts.x, 3u);
  EXPECT_EQ(descriptors[1].offsetsCounts.y, 1u);
  EXPECT_EQ(descriptors[1].offsetsCounts.z, 3u);
  EXPECT_EQ(descriptors[1].offsetsCounts.w, 1u);

  std::array<uint32_t, 2u> primitiveWords{};
  auto primitiveReadResult =
      gpu.readBuffer(view.meshletPrimitiveIndexBuffer, 0u,
                     std::as_writable_bytes(std::span<uint32_t>(
                         primitiveWords.data(), primitiveWords.size())));
  ASSERT_FALSE(primitiveReadResult.hasError()) << primitiveReadResult.error();

  EXPECT_EQ(primitiveWords[0], 0x00020100u);
  EXPECT_EQ(primitiveWords[1], 0x00000102u);
}

TEST(MeshBinaryCacheTests, PreparedModelOwnsCpuPayloadUntilGpuCreation) {
  FakeMeshletUploadGpuDevice gpu;
  nuri::MeshData mesh = makeMeshletUploadMesh();
  const uint32_t expectedVertexCount =
      static_cast<uint32_t>(mesh.vertices.size());
  const uint32_t expectedIndexCount =
      static_cast<uint32_t>(mesh.indices.size());
  const uint32_t expectedMeshletCount =
      static_cast<uint32_t>(mesh.meshlets.size());

  auto preparedResult = nuri::Model::prepare(std::move(mesh));
  ASSERT_FALSE(preparedResult.hasError()) << preparedResult.error();
  EXPECT_GT(preparedResult.value().uploadBytes(), 0u);

  auto modelResult = nuri::Model::createPrepared(
      gpu, std::move(preparedResult.value()), "prepared_model");
  ASSERT_FALSE(modelResult.hasError()) << modelResult.error();
  ASSERT_NE(modelResult.value(), nullptr);
  EXPECT_EQ(modelResult.value()->vertexCount(), expectedVertexCount);
  EXPECT_EQ(modelResult.value()->indexCount(), expectedIndexCount);
  EXPECT_EQ(modelResult.value()->meshletGpuView().meshletCount,
            expectedMeshletCount);
  EXPECT_EQ(gpu.waitIdleCallCount, 0u);
}

TEST(MeshBinaryCacheTests, ModelUploadRejectsMeshletsAboveShaderLimits) {
  {
    FakeMeshletUploadGpuDevice gpu;
    nuri::MeshData mesh = makeMeshletUploadMesh();
    mesh.submeshes[0].lods[0].meshletCount = 1u;
    mesh.meshlets.resize(1u);
    mesh.meshlets[0].vertexCount = 65u;
    mesh.meshletVertexIndices.assign(65u, 0u);

    auto modelResult = nuri::Model::create(gpu, mesh, "meshlet_vertex_limit");
    ASSERT_TRUE(modelResult.hasError());
    EXPECT_NE(modelResult.error().find("vertex count exceeds"),
              std::string::npos);
  }

  {
    FakeMeshletUploadGpuDevice gpu;
    nuri::MeshData mesh = makeMeshletUploadMesh();
    mesh.submeshes[0].lods[0].meshletCount = 1u;
    mesh.meshlets.resize(1u);
    mesh.meshlets[0].primitiveCount = 125u;
    mesh.meshletPrimitiveIndices.assign(125u * 3u, 0u);

    auto modelResult =
        nuri::Model::create(gpu, mesh, "meshlet_primitive_limit");
    ASSERT_TRUE(modelResult.hasError());
    EXPECT_NE(modelResult.error().find("primitive count exceeds"),
              std::string::npos);
  }
}

TEST(MeshBinaryCacheTests, ModelUploadRejectsInvalidMeshletLocalIndices) {
  {
    FakeMeshletUploadGpuDevice gpu;
    nuri::MeshData mesh = makeMeshletUploadMesh();
    mesh.meshletPrimitiveIndices[2u] = 3u;

    auto modelResult =
        nuri::Model::create(gpu, mesh, "meshlet_primitive_local_index");
    ASSERT_TRUE(modelResult.hasError());
    EXPECT_NE(modelResult.error().find("primitive local index out of bounds"),
              std::string::npos);
  }

  {
    FakeMeshletUploadGpuDevice gpu;
    nuri::MeshData mesh = makeMeshletUploadMesh();
    mesh.meshletVertexIndices[1u] = mesh.submeshes[0].vertexCount;

    auto modelResult =
        nuri::Model::create(gpu, mesh, "meshlet_vertex_local_index");
    ASSERT_TRUE(modelResult.hasError());
    EXPECT_NE(modelResult.error().find("vertex local index out of submesh"),
              std::string::npos);
  }
}
