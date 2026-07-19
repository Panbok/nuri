#pragma once
#include "nuri/core/containers/hash_map.h"
#include "nuri/core/containers/slot_pool.h"
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/resources/gpu/material.h"
#include "nuri/resources/gpu/model.h"
#include "nuri/resources/gpu/resource_handles.h"
#include "nuri/resources/gpu/resource_keys.h"
#include "nuri/resources/gpu/texture.h"
#include "nuri/resources/storage/texture/dds_texture_pack.h"
#include "nuri/scene/scene_prefab.h"
#include <cstdint>
#include <limits>
#include <memory>
#include <memory_resource>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>
namespace nuri {

struct NURI_API TextureRequest {
  std::string path{};
  TextureLoadOptions loadOptions{};
  TextureRequestKind kind = TextureRequestKind::Texture2D;
  std::string debugName{};
  DdsTexturePack *ddsPack = nullptr;
};

struct NURI_API ModelRequest {
  std::string path{};
  MeshImportOptions importOptions{};
  std::string debugName{};
  uint32_t sceneMeshIndex = std::numeric_limits<uint32_t>::max();
};

struct NURI_API MaterialRequest {
  MaterialDesc desc{};
  using TextureRefs = MaterialTextureSlots<TextureRef>;
  TextureRefs textureRefs{};
  std::string debugName{};
  std::string sourceIdentity{};
};

template <typename Fn>
constexpr void
forEachMaterialTextureRef(const MaterialRequest::TextureRefs &refs,
                          Fn &&fn) noexcept(noexcept(fn(refs.front()))) {
  for (TextureRef ref : refs) {
    fn(ref);
  }
}

struct NURI_API TextureRecord {
  TextureRef ref = kInvalidTextureRef;
  TextureHandle texture{};
  uint32_t bindlessIndex = kInvalidTextureBindlessIndex;
  TextureType type = TextureType::Texture2D;
  Format format = Format::RGBA8_UNORM;
  TextureUsage usage = TextureUsage::Sampled;
  TextureDimensions dimensions{};
  Storage storage = Storage::Device;
  uint32_t numLayers = 1;
  uint32_t numSamples = 1;
  uint32_t numMipLevels = 1;
  TextureRequestKind sourceKind = TextureRequestKind::Texture2D;
  TextureLoadOptions loadOptions{};
  std::pmr::string canonicalPath;
  std::pmr::string debugName;
  explicit TextureRecord(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : canonicalPath(memory), debugName(memory) {}
};

struct NURI_API MaterialRecord {
  MaterialRef ref = kInvalidMaterialRef;
  MaterialDesc desc{};
  MaterialRequest::TextureRefs textureRefs{};
  MaterialPackedGpuData packedGpuData{};
  uint64_t descHash = 0;
  std::pmr::string debugName;
  std::pmr::string sourceIdentity;
  explicit MaterialRecord(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : debugName(memory), sourceIdentity(memory) {}
};

struct NURI_API ModelRecord {
  ModelRef ref = kInvalidModelRef;
  std::unique_ptr<Model> model{};
  std::pmr::string canonicalPath;
  uint64_t importOptionsHash = 0;
  uint32_t sceneMeshIndex = std::numeric_limits<uint32_t>::max();
  std::pmr::vector<MaterialRef> sourceMaterialToRuntime;
  explicit ModelRecord(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : canonicalPath(memory), sourceMaterialToRuntime(memory) {}
  [[nodiscard]] MaterialRef materialForSource(uint32_t sourceMaterial) const {
    if (sourceMaterial >= sourceMaterialToRuntime.size()) {
      return kInvalidMaterialRef;
    }
    return sourceMaterialToRuntime[sourceMaterial];
  }
  [[nodiscard]] MaterialRef materialForSubmesh(uint32_t submeshIndex) const;
};

struct NURI_API MaterialTableDirtyRange {
  uint32_t first = 0u;
  uint32_t count = 0u;
  [[nodiscard]] bool empty() const noexcept { return count == 0u; }
};

struct NURI_API MaterialTableSnapshot {
  std::span<const MaterialHeaderGpuData> headers{};
  std::span<const MaterialClearcoatGpuData> clearcoat{};
  std::span<const MaterialSheenGpuData> sheen{};
  std::span<const MaterialTransmissionGpuData> transmission{};
  std::span<const MaterialSpecularGpuData> specular{};
  MaterialTableDirtyRange dirtyHeaders{};
  MaterialTableDirtyRange dirtyClearcoat{};
  MaterialTableDirtyRange dirtySheen{};
  MaterialTableDirtyRange dirtyTransmission{};
  MaterialTableDirtyRange dirtySpecular{};
  uint64_t dirtyBaseVersion = 0u;
  uint64_t version = 0;
};

struct NURI_API PoolStats {
  uint32_t liveTextures = 0;
  uint32_t liveMaterials = 0;
  uint32_t liveModels = 0;
  uint32_t retiredTextures = 0;
  uint32_t retiredMaterials = 0;
  uint32_t retiredModels = 0;
};

class NURI_API ResourceManager final {
public:
  explicit ResourceManager(
      GPUDevice &gpu,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~ResourceManager();
  ResourceManager(const ResourceManager &) = delete;
  ResourceManager &operator=(const ResourceManager &) = delete;
  ResourceManager(ResourceManager &&) = delete;
  ResourceManager &operator=(ResourceManager &&) = delete;
  [[nodiscard]] Result<TextureRef, std::string>
  acquireTexture(const TextureRequest &request);
  [[nodiscard]] std::optional<TextureRef>
  tryAcquireTexture(const TextureRequest &request);
  [[nodiscard]] Result<TextureRef, std::string>
  adoptPreparedTexture(const TextureRequest &request,
                       std::unique_ptr<Texture> texture);
  [[nodiscard]] Result<ModelRef, std::string>
  acquireModel(const ModelRequest &request);
  [[nodiscard]] std::optional<ModelRef>
  tryAcquireModel(const ModelRequest &request);
  [[nodiscard]] Result<ModelRef, std::string>
  adoptPreparedModel(const ModelRequest &request, std::unique_ptr<Model> model);
  [[nodiscard]] Result<ModelRef, std::string>
  acquireGeneratedModel(const MeshData &meshData,
                        std::string_view debugName = {});
  [[nodiscard]] Result<MaterialRef, std::string>
  acquireMaterial(const MaterialRequest &request);
  [[nodiscard]] Result<ScenePrefabAssets, std::string>
  acquireScenePrefabAssets(const ScenePrefab &prefab);
  void retain(TextureRef ref);
  void release(TextureRef ref);
  void retain(ModelRef ref);
  void release(ModelRef ref);
  void retain(MaterialRef ref);
  void release(MaterialRef ref);
  [[nodiscard]] bool owns(TextureRef ref) const noexcept;
  [[nodiscard]] bool owns(ModelRef ref) const noexcept;
  [[nodiscard]] bool owns(MaterialRef ref) const noexcept;
  [[nodiscard]] const TextureRecord *tryGet(TextureRef ref) const;
  [[nodiscard]] const ModelRecord *tryGet(ModelRef ref) const;
  [[nodiscard]] const MaterialRecord *tryGet(MaterialRef ref) const;
  [[nodiscard]] MaterialTableSnapshot materialSnapshot() const noexcept {
    return MaterialTableSnapshot{
        .headers = std::span<const MaterialHeaderGpuData>(
            materialHeaderTable_.data(), materialHeaderTable_.size()),
        .clearcoat = std::span<const MaterialClearcoatGpuData>(
            materialClearcoatTable_.data(), materialClearcoatTable_.size()),
        .sheen = std::span<const MaterialSheenGpuData>(
            materialSheenTable_.data(), materialSheenTable_.size()),
        .transmission = std::span<const MaterialTransmissionGpuData>(
            materialTransmissionTable_.data(),
            materialTransmissionTable_.size()),
        .specular = std::span<const MaterialSpecularGpuData>(
            materialSpecularTable_.data(), materialSpecularTable_.size()),
        .dirtyHeaders = materialHeaderDirtyRange_,
        .dirtyClearcoat = materialClearcoatDirtyRange_,
        .dirtySheen = materialSheenDirtyRange_,
        .dirtyTransmission = materialTransmissionDirtyRange_,
        .dirtySpecular = materialSpecularDirtyRange_,
        .dirtyBaseVersion = materialTableDirtyBaseVersion_,
        .version = materialTableVersion_,
    };
  }
  [[nodiscard]] uint64_t materialVersion() const noexcept {
    return materialTableVersion_;
  }
  [[nodiscard]] uint64_t modelMaterialBindingVersion() const noexcept {
    return modelMaterialBindingVersion_;
  }
  [[nodiscard]] GpuMultisampleCapabilities gpuMultisampleCapabilities() const {
    return gpu_.getMultisampleCapabilities();
  }
  [[nodiscard]] TextureCompressionCaps textureCompressionCaps() const {
    return gpu_.getTextureCompressionCaps();
  }
  void beginPublicationBatch();
  void endPublicationBatch();
  void collectGarbage();
  [[nodiscard]] PoolStats stats() const;
  [[nodiscard]] uint32_t materialTableIndex(MaterialRef ref) const;
  [[nodiscard]] MaterialRef
  modelMaterialForSubmesh(ModelRef model, uint32_t submeshIndex) const;
  bool setModelMaterialForSource(ModelRef model, uint32_t sourceMaterialIndex,
                                 MaterialRef material);
  void setModelMaterialForAllSources(ModelRef model, MaterialRef material);

private:
  template <typename Record, typename Ref, typename Owner = std::monostate>
  struct ResourcePool {
    struct Slot {
      uint32_t refCount = 0;
      Owner owner{};
      Record record;
      explicit Slot(std::pmr::memory_resource *memory) : record(memory) {}
    };
    explicit ResourcePool(std::pmr::memory_resource *memory)
        : slots(memory), meta(memory), pending(memory) {}
    std::pmr::vector<Slot> slots;
    SlotPool<MaskedNonZeroGenerationPolicy<kResourceHandleGenerationMask>> meta;
    std::pmr::vector<Ref> pending;
  };
  using TexturePool =
      ResourcePool<TextureRecord, TextureRef, OwnedTextureHandle>;
  using MaterialPool = ResourcePool<MaterialRecord, MaterialRef>;
  using ModelPool = ResourcePool<ModelRecord, ModelRef>;
  using TextureSlot = TexturePool::Slot;
  using MaterialSlot = MaterialPool::Slot;
  using ModelSlot = ModelPool::Slot;
  [[nodiscard]] Result<SlotReservation, std::string> allocateMaterialSlot();
  void destroyTextureSlot(uint32_t index);
  void destroyMaterialSlot(uint32_t index);
  void destroyModelSlot(uint32_t index);
  void rebuildPackedMaterialTables();
  void markMaterialTablesDirty();
  void markModelMaterialBindingsDirty();
  void flushPublicationVersions();
  [[nodiscard]] Result<TextureRef, std::string>
  storeAcquiredTexture(const TextureKey &key, std::string_view canonicalPath,
                       const TextureRequest &request,
                       std::unique_ptr<Texture> texture);
  [[nodiscard]] ModelRef tryAcquireCachedModel(const ModelKey &key);
  [[nodiscard]] Result<ModelRef, std::string>
  storeAcquiredModel(const ModelKey &key, std::string_view canonicalPath,
                     uint64_t optionsHash, const ModelRequest &request,
                     std::unique_ptr<Model> model);
  GPUDevice &gpu_;
  std::pmr::memory_resource *memory_ = std::pmr::get_default_resource();
  TexturePool textures_;
  MaterialPool materials_;
  ModelPool models_;
  std::pmr::vector<MaterialHeaderGpuData> materialHeaderTable_;
  std::pmr::vector<MaterialClearcoatGpuData> materialClearcoatTable_;
  std::pmr::vector<MaterialSheenGpuData> materialSheenTable_;
  std::pmr::vector<MaterialTransmissionGpuData> materialTransmissionTable_;
  std::pmr::vector<MaterialSpecularGpuData> materialSpecularTable_;
  MaterialTableDirtyRange materialHeaderDirtyRange_{};
  MaterialTableDirtyRange materialClearcoatDirtyRange_{};
  MaterialTableDirtyRange materialSheenDirtyRange_{};
  MaterialTableDirtyRange materialTransmissionDirtyRange_{};
  MaterialTableDirtyRange materialSpecularDirtyRange_{};
  uint64_t materialTableDirtyBaseVersion_ = 0u;
  uint64_t materialTableVersion_ = 0;
  uint64_t modelMaterialBindingVersion_ = 0;
  uint32_t publicationBatchDepth_ = 0u;
  bool materialTablesDirty_ = false;
  bool modelMaterialBindingsDirty_ = false;
  HashMap<TextureKey, TextureRef, TextureKeyHash> textureCache_;
  HashMap<MaterialKey, MaterialRef, MaterialKeyHash> materialCache_;
  HashMap<ModelKey, ModelRef, ModelKeyHash> modelCache_;
};

} // namespace nuri
