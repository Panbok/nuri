#pragma once
#include "nuri/core/result.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/math/types.h"
#include "nuri/resources/cpu/mesh_data.h"
#include "nuri/resources/gpu/buffer.h"
#include "nuri/resources/mesh_importer.h"
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <memory_resource>
#include <span>
#include <string>
#include <string_view>
#include <vector>
namespace nuri {

class Model;
struct ModelAnimationPackedData;

class NURI_API PreparedGpuModelData final {
public:
  ~PreparedGpuModelData();
  PreparedGpuModelData(const PreparedGpuModelData &) = delete;
  PreparedGpuModelData &operator=(const PreparedGpuModelData &) = delete;
  PreparedGpuModelData(PreparedGpuModelData &&) = delete;
  PreparedGpuModelData &operator=(PreparedGpuModelData &&) = delete;

private:
  struct Impl;
  explicit PreparedGpuModelData(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_{};
  friend class Model;
};

enum class PackedVertexFormat : uint8_t {
  StaticQuantized20 = 0,
  AnimatedFloat24 = 1,
  AnimatedFloat32 = 2,
};

struct PreparedModelBufferData {
  std::vector<std::byte> bytes{};
  uint32_t count = 0u;
  uint32_t stride = 0u;
};

struct NURI_API PreparedModelData {
  explicit PreparedModelData(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : mesh(memory) {}
  MeshData mesh;
  std::vector<std::byte> packedVertexBytes{};
  PackedVertexFormat packedVertexFormat = PackedVertexFormat::StaticQuantized20;
  PreparedModelBufferData staticDecode{};
  PreparedModelBufferData skinInfluences{};
  PreparedModelBufferData morphMeta{};
  PreparedModelBufferData morphDeltas{};
  [[nodiscard]] uint64_t uploadBytes() const noexcept {
    return static_cast<uint64_t>(packedVertexBytes.size()) +
           static_cast<uint64_t>(mesh.indices.size()) * sizeof(uint32_t) +
           static_cast<uint64_t>(staticDecode.bytes.size()) +
           static_cast<uint64_t>(skinInfluences.bytes.size()) +
           static_cast<uint64_t>(morphMeta.bytes.size()) +
           static_cast<uint64_t>(morphDeltas.bytes.size()) +
           static_cast<uint64_t>(mesh.meshlets.size()) *
               sizeof(MeshletDescriptor) +
           static_cast<uint64_t>(mesh.meshletVertexIndices.size()) *
               sizeof(uint32_t) +
           static_cast<uint64_t>(mesh.meshletPrimitiveIndices.size());
  }
};

struct StaticVertexDecodeGpuData {
  glm::vec4 offset{0.0f, 0.0f, 0.0f, 0.0f};
  glm::vec4 scale{1.0f, 1.0f, 1.0f, 0.0f};
};

class NURI_API Model final {
public:
  static constexpr uint32_t kInvalidMaterialIndex =
      std::numeric_limits<uint32_t>::max();
  struct ModelAnimationGpuView {
    BufferHandle skinInfluenceBuffer{};
    BufferHandle morphMetaBuffer{};
    BufferHandle morphDeltaBuffer{};
    uint32_t skinInfluenceCount = 0;
    uint32_t morphTargetCount = 0;
    uint32_t morphVertexCount = 0;
  };
  struct SubmeshMeshletLodRangeGpu {
    std::array<uint32_t, Submesh::kMaxLodCount> meshletOffset{};
    std::array<uint32_t, Submesh::kMaxLodCount> meshletCount{};
    std::array<float, Submesh::kMaxLodCount> error{};
    uint32_t lodCount = 0;
    uint32_t _pad0 = 0;
    uint32_t _pad1 = 0;
    uint32_t _pad2 = 0;
  };
  struct ModelMeshletGpuView {
    BufferHandle meshletBuffer{};
    BufferHandle meshletVertexIndexBuffer{};
    BufferHandle meshletPrimitiveIndexBuffer{};
    BufferHandle lodRangeBuffer{};
    uint32_t meshletCount = 0;
    uint32_t meshletVertexIndexCount = 0;
    uint32_t meshletPrimitiveIndexCount = 0;
    uint32_t lodRangeCount = 0;
  };
  ~Model();
  Model(const Model &) = delete;
  Model &operator=(const Model &) = delete;
  Model(Model &&) = delete;
  Model &operator=(Model &&) = delete;
  [[nodiscard]] static Result<std::unique_ptr<Model>, std::string>
  create(GPUDevice &gpu, const MeshData &data, std::string_view debugName = {});
  [[nodiscard]] static Result<PreparedModelData, std::string>
  prepare(MeshData data);
  [[nodiscard]] static Result<PreparedModelData, std::string> prepareFromFile(
      std::string_view path, const MeshImportOptions &options = {},
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  [[nodiscard]] static Result<PreparedModelData, std::string>
  prepareSceneMeshFromFile(
      std::string_view path, uint32_t sceneMeshIndex,
      const MeshImportOptions &options = {},
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  [[nodiscard]] static Result<std::unique_ptr<Model>, std::string>
  createPrepared(GPUDevice &gpu, PreparedModelData data,
                 std::string_view debugName = {});
  [[nodiscard]] static Result<std::unique_ptr<PreparedGpuModelData>,
                              std::string>
  prepareGpu(GPUDevice &gpu, PreparedModelData data,
             std::string_view debugName = {});
  [[nodiscard]] static Result<std::unique_ptr<Model>, std::string>
  publishPreparedGpu(GPUDevice &gpu,
                     std::unique_ptr<PreparedGpuModelData> prepared);
  [[nodiscard]] static Result<std::unique_ptr<Model>, std::string>
  createFromFile(
      GPUDevice &gpu, std::string_view path,
      const MeshImportOptions &options = {},
      std::pmr::memory_resource *mem = std::pmr::get_default_resource(),
      std::string_view debugName = {});
  [[nodiscard]] GeometryAllocationHandle geometryHandle() const noexcept {
    return geometry_;
  }
  [[nodiscard]] std::span<const Submesh> submeshes() const noexcept {
    return submeshes_;
  }
  [[nodiscard]] uint32_t vertexCount() const noexcept { return vertexCount_; }
  [[nodiscard]] uint32_t indexCount() const noexcept { return indexCount_; }
  [[nodiscard]] const BoundingBox &bounds() const noexcept { return bounds_; }
  [[nodiscard]] const ModelAnimationGpuView &animationGpuView() const noexcept {
    return animationGpuView_;
  }
  [[nodiscard]] const ModelMeshletGpuView &meshletGpuView() const noexcept {
    return meshletGpuView_;
  }
  [[nodiscard]] std::span<const glm::vec4>
  meshletBoundsSpheres() const noexcept {
    return meshletBoundsSpheres_;
  }
  [[nodiscard]] bool hasMeshlets() const noexcept {
    return meshletGpuView_.meshletCount != 0u &&
           nuri::isValid(meshletGpuView_.meshletBuffer) &&
           nuri::isValid(meshletGpuView_.meshletVertexIndexBuffer) &&
           nuri::isValid(meshletGpuView_.meshletPrimitiveIndexBuffer) &&
           nuri::isValid(meshletGpuView_.lodRangeBuffer);
  }
  [[nodiscard]] PackedVertexFormat drawVertexFormat() const noexcept {
    return drawVertexFormat_;
  }
  [[nodiscard]] BufferHandle vertexDecodeBuffer() const noexcept {
    return vertexDecodeBuffer_ != nullptr ? vertexDecodeBuffer_->handle()
                                          : BufferHandle{};
  }
  [[nodiscard]] uint64_t vertexDecodeBufferAddress() const noexcept {
    return vertexDecodeBufferAddress_;
  }
  [[nodiscard]] bool hasAnimationData() const noexcept {
    return nuri::isValid(animationGpuView_.skinInfluenceBuffer) ||
           nuri::isValid(animationGpuView_.morphDeltaBuffer);
  }
  [[nodiscard]] uint32_t sourceMaterialCount() const noexcept {
    return static_cast<uint32_t>(sourceMaterialToRuntime_.size());
  }
  [[nodiscard]] uint32_t
  materialIndexForSource(uint32_t sourceMaterialIndex) const noexcept;
  [[nodiscard]] uint32_t
  materialIndexForSubmesh(uint32_t submeshIndex) const noexcept;
  [[nodiscard]] bool setMaterialIndexForSource(uint32_t sourceMaterialIndex,
                                               uint32_t materialIndex) noexcept;
  void setMaterialIndexForAllSources(uint32_t materialIndex) noexcept;

private:
  struct PackedSource;
  [[nodiscard]] static Result<std::unique_ptr<Model>, std::string>
  createPacked(GPUDevice &gpu, const PackedSource &source,
               std::string_view debugName);
  Model(GPUDevice &gpu, GeometryAllocationHandle geometry,
        std::pmr::vector<Submesh> submeshes, uint32_t vertexCount,
        uint32_t indexCount, BoundingBox bounds,
        PackedVertexFormat drawVertexFormat,
        ModelAnimationGpuView animationGpuView,
        ModelMeshletGpuView meshletGpuView, uint64_t vertexDecodeBufferAddress,
        std::unique_ptr<Buffer> vertexDecodeBuffer,
        std::unique_ptr<Buffer> skinInfluenceBuffer,
        std::unique_ptr<Buffer> morphMetaBuffer,
        std::unique_ptr<Buffer> morphDeltaBuffer,
        std::unique_ptr<Buffer> meshletBuffer,
        std::unique_ptr<Buffer> meshletVertexIndexBuffer,
        std::unique_ptr<Buffer> meshletPrimitiveIndexBuffer,
        std::unique_ptr<Buffer> meshletLodRangeBuffer,
        std::pmr::vector<glm::vec4> meshletBoundsSpheres,
        std::pmr::vector<uint32_t> sourceMaterialToRuntime)
      : gpu_(&gpu), geometry_(geometry), submeshes_(std::move(submeshes)),
        vertexCount_(vertexCount), indexCount_(indexCount), bounds_(bounds),
        drawVertexFormat_(drawVertexFormat),
        animationGpuView_(animationGpuView), meshletGpuView_(meshletGpuView),
        vertexDecodeBufferAddress_(vertexDecodeBufferAddress),
        vertexDecodeBuffer_(std::move(vertexDecodeBuffer)),
        skinInfluenceBuffer_(std::move(skinInfluenceBuffer)),
        morphMetaBuffer_(std::move(morphMetaBuffer)),
        morphDeltaBuffer_(std::move(morphDeltaBuffer)),
        meshletBuffer_(std::move(meshletBuffer)),
        meshletVertexIndexBuffer_(std::move(meshletVertexIndexBuffer)),
        meshletPrimitiveIndexBuffer_(std::move(meshletPrimitiveIndexBuffer)),
        meshletLodRangeBuffer_(std::move(meshletLodRangeBuffer)),
        meshletBoundsSpheres_(std::move(meshletBoundsSpheres)),
        sourceMaterialToRuntime_(std::move(sourceMaterialToRuntime)) {}
  GPUDevice *gpu_ = nullptr;
  GeometryAllocationHandle geometry_{};
  std::pmr::vector<Submesh> submeshes_;
  uint32_t vertexCount_ = 0;
  uint32_t indexCount_ = 0;
  BoundingBox bounds_{};
  PackedVertexFormat drawVertexFormat_ = PackedVertexFormat::StaticQuantized20;
  ModelAnimationGpuView animationGpuView_{};
  ModelMeshletGpuView meshletGpuView_{};
  uint64_t vertexDecodeBufferAddress_ = 0u;
  std::unique_ptr<Buffer> vertexDecodeBuffer_;
  std::unique_ptr<Buffer> skinInfluenceBuffer_;
  std::unique_ptr<Buffer> morphMetaBuffer_;
  std::unique_ptr<Buffer> morphDeltaBuffer_;
  std::unique_ptr<Buffer> meshletBuffer_;
  std::unique_ptr<Buffer> meshletVertexIndexBuffer_;
  std::unique_ptr<Buffer> meshletPrimitiveIndexBuffer_;
  std::unique_ptr<Buffer> meshletLodRangeBuffer_;
  std::pmr::vector<glm::vec4> meshletBoundsSpheres_;
  std::pmr::vector<uint32_t> sourceMaterialToRuntime_;
};

} // namespace nuri
