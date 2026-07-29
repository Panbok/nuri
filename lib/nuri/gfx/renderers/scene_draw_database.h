#pragma once
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_types.h"
#include "nuri/resources/gpu/model.h"
#include "nuri/resources/gpu/resource_handles.h"
#include <array>
#include <cstdint>
#include <memory_resource>
#include <span>
#include <string>
#include <vector>
namespace nuri {

class GPUDevice;
class RenderScene;
class ResourceManager;
struct FrameBuildContext;
struct RenderScenePreparationContext;
struct ModelRecord;
struct Renderable;
struct Submesh;

struct SceneInstanceRecord {
  const Renderable *renderable = nullptr;
  const Model *model = nullptr;
  uint32_t firstDraw = 0u;
  uint32_t drawCount = 0u;
  bool dynamicCaster = false;
};

struct SceneDrawRecord {
  const Renderable *renderable = nullptr;
  const Model *model = nullptr;
  const Submesh *submesh = nullptr;
  const Model::ModelMeshletGpuView *meshletView = nullptr;
  uint32_t submeshIndex = 0;
  uint32_t instanceIndex = 0;
  GeometryAllocationHandle geometryHandle{};
  BufferHandle indexBuffer{};
  uint64_t indexBufferOffset = 0;
  IndexFormat indexFormat = IndexFormat::U32;
  BufferHandle baseVertexBuffer{};
  BufferHandle vertexBuffer{};
  BufferHandle baseVertexDecodeBuffer{};
  BufferHandle vertexDecodeBuffer{};
  uint64_t vertexBufferByteOffset = 0;
  uint64_t baseVertexBufferAddress = 0;
  uint64_t baseVertexDecodeBufferAddress = 0;
  uint64_t vertexBufferAddress = 0;
  uint64_t vertexDecodeBufferAddress = 0;
  uint32_t basePackedVertexFormat = 0;
  uint32_t vertexDecodeIndex = 0;
  uint32_t packedVertexFormat = 0;
  MaterialRef material = kInvalidMaterialRef;
  uint32_t materialIndex = 0;
  TextureHandle baseColorTexture{};
  bool doubleSided = false;
  bool alphaMasked = false;
  bool alphaBlended = false;
  bool transmission = false;
  bool sortedTransmissionFeedback = false;
  bool materialNormalRequired = false;
};

enum class SceneDrawCategory : uint8_t {
  Opaque,
  AlphaMasked,
  AlphaBlended,
  Transmission,
  RayTracing,
  Count,
};

struct SceneDrawSourceVersion {
  const RenderScene *scene = nullptr;
  uint64_t topology = UINT64_MAX;
  uint64_t material = UINT64_MAX;
  uint64_t materialBinding = UINT64_MAX;
  uint64_t geometry = UINT64_MAX;
  bool operator==(const SceneDrawSourceVersion &) const noexcept = default;
};

class NURI_API SceneDrawDatabase {
public:
  explicit SceneDrawDatabase(
      GPUDevice &gpu,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  [[nodiscard]] Result<bool, std::string>
  update(const RenderScene &scene, const ResourceManager &resources);
  [[nodiscard]] Result<bool, std::string> prepare(FrameBuildContext &ctx);
  [[nodiscard]] Result<bool, std::string>
  prepareScene(RenderScenePreparationContext &ctx);
  [[nodiscard]] std::span<const SceneInstanceRecord>
  instances() const noexcept {
    return instances_;
  }
  [[nodiscard]] std::span<const SceneDrawRecord> draws() const noexcept {
    return draws_;
  }
  [[nodiscard]] std::span<const TextureHandle>
  rayTracingMaterialTextures() const noexcept {
    return rayTracingMaterialTextures_;
  }
  [[nodiscard]] std::span<const uint32_t>
  category(SceneDrawCategory category) const noexcept {
    return categories_[static_cast<size_t>(category)];
  }
  [[nodiscard]] uint64_t generation() const noexcept { return generation_; }
  [[nodiscard]] const SceneDrawSourceVersion &sourceVersion() const noexcept {
    return sourceVersion_;
  }

private:
  GPUDevice &gpu_;
  std::pmr::vector<SceneInstanceRecord> instances_;
  std::pmr::vector<SceneDrawRecord> draws_;
  std::pmr::vector<TextureHandle> rayTracingMaterialTextures_;
  std::array<std::pmr::vector<uint32_t>,
             static_cast<size_t>(SceneDrawCategory::Count)>
      categories_;
  SceneDrawSourceVersion sourceVersion_{};
  uint64_t generation_ = 0u;
};

} // namespace nuri
