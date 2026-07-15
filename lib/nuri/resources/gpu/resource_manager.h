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
#include "nuri/scene/scene_prefab.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <memory_resource>
#include <span>
#include <string>
#include <vector>

namespace nuri {

struct NURI_API TextureRequest {
  std::string path{};
  TextureLoadOptions loadOptions{};
  TextureRequestKind kind = TextureRequestKind::Texture2D;
  std::string debugName{};
};

struct NURI_API ModelRequest {
  std::string path{};
  MeshImportOptions importOptions{};
  std::string debugName{};
  uint32_t sceneMeshIndex = std::numeric_limits<uint32_t>::max();
};

struct NURI_API MaterialRequest {
  MaterialDesc desc{};
  struct TextureRefs {
    TextureRef baseColor = kInvalidTextureRef;
    TextureRef metallicRoughness = kInvalidTextureRef;
    TextureRef normal = kInvalidTextureRef;
    TextureRef occlusion = kInvalidTextureRef;
    TextureRef emissive = kInvalidTextureRef;
    TextureRef clearcoat = kInvalidTextureRef;
    TextureRef clearcoatRoughness = kInvalidTextureRef;
    TextureRef clearcoatNormal = kInvalidTextureRef;
    TextureRef specular = kInvalidTextureRef;
    TextureRef specularColor = kInvalidTextureRef;
    TextureRef sheenColor = kInvalidTextureRef;
    TextureRef sheenRoughness = kInvalidTextureRef;
    TextureRef transmission = kInvalidTextureRef;
    TextureRef thickness = kInvalidTextureRef;
  } textureRefs{};
  std::string debugName{};
  std::string sourceIdentity{};
};

template <typename Fn>
constexpr void forEachMaterialTextureRef(
    const MaterialRequest::TextureRefs &refs,
    Fn &&fn) noexcept(noexcept(fn(refs.baseColor)) &&
                      noexcept(fn(refs.metallicRoughness)) &&
                      noexcept(fn(refs.normal)) &&
                      noexcept(fn(refs.occlusion)) &&
                      noexcept(fn(refs.emissive)) &&
                      noexcept(fn(refs.clearcoat)) &&
                      noexcept(fn(refs.clearcoatRoughness)) &&
                      noexcept(fn(refs.clearcoatNormal)) &&
                      noexcept(fn(refs.specular)) &&
                      noexcept(fn(refs.specularColor)) &&
                      noexcept(fn(refs.sheenColor)) &&
                      noexcept(fn(refs.sheenRoughness)) &&
                      noexcept(fn(refs.transmission)) &&
                      noexcept(fn(refs.thickness))) {
  // Keep this enumeration in sync with MaterialRequest::TextureRefs.
  fn(refs.baseColor);
  fn(refs.metallicRoughness);
  fn(refs.normal);
  fn(refs.occlusion);
  fn(refs.emissive);
  fn(refs.clearcoat);
  fn(refs.clearcoatRoughness);
  fn(refs.clearcoatNormal);
  fn(refs.specular);
  fn(refs.specularColor);
  fn(refs.sheenColor);
  fn(refs.sheenRoughness);
  fn(refs.transmission);
  fn(refs.thickness);
}

struct NURI_API ImportedMaterialRequest {
  std::string modelPath{};
  ModelRef model = kInvalidModelRef;
  std::string debugNamePrefix{};
};

struct NURI_API ImportedMaterialBatch {
  MaterialRef firstMaterial = kInvalidMaterialRef;
  uint32_t createdMaterialCount = 0;
};

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

// headers is indexed by material table slot; clearcoat, sheen, transmission,
// and specular are compact extension tables keyed by indices stored in headers,
// so those spans are not required to have equal lengths. Empty/missing
// extension components use kInvalidMaterialExtensionIndex in headers. version
// identifies the snapshot contents and changes when any table changes.
struct NURI_API MaterialTableSnapshot {
  std::span<const MaterialHeaderGpuData> headers{};
  std::span<const MaterialClearcoatGpuData> clearcoat{};
  std::span<const MaterialSheenGpuData> sheen{};
  std::span<const MaterialTransmissionGpuData> transmission{};
  std::span<const MaterialSpecularGpuData> specular{};
  uint64_t version = 0;
};

struct NURI_API PoolStats {
  uint32_t liveTextures = 0;
  uint32_t liveMaterials = 0;
  uint32_t liveModels = 0;
  uint32_t retiredTextures = 0;
  uint32_t retiredMaterials = 0;
  uint32_t retiredModels = 0;
  uint64_t textureCacheEntries = 0;
  uint64_t materialCacheEntries = 0;
  uint64_t modelCacheEntries = 0;
  uint64_t textureAcquireHits = 0;
  uint64_t textureAcquireMisses = 0;
  uint64_t modelAcquireHits = 0;
  uint64_t modelAcquireMisses = 0;
  uint64_t materialAcquireHits = 0;
  uint64_t materialAcquireMisses = 0;
  uint64_t invalidTextureLookups = 0;
  uint64_t staleTextureLookups = 0;
  uint64_t invalidMaterialLookups = 0;
  uint64_t staleMaterialLookups = 0;
  uint64_t invalidModelLookups = 0;
  uint64_t staleModelLookups = 0;
  uint64_t staleTextureReleases = 0;
  uint64_t staleMaterialReleases = 0;
  uint64_t staleModelReleases = 0;
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
  [[nodiscard]] Result<ModelRef, std::string>
  acquireModel(const ModelRequest &request);
  [[nodiscard]] Result<ModelRef, std::string>
  acquireGeneratedModel(const MeshData &meshData,
                        std::string_view debugName = {});
  [[nodiscard]] Result<MaterialRef, std::string>
  acquireMaterial(const MaterialRequest &request);
  [[nodiscard]] Result<ImportedMaterialBatch, std::string>
  acquireMaterialsFromModel(const ImportedMaterialRequest &request);
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
        .version = materialTableVersion_,
    };
  }
  [[nodiscard]] uint64_t materialVersion() const noexcept {
    return materialTableVersion_;
  }
  [[nodiscard]] GpuMultisampleCapabilities gpuMultisampleCapabilities() const {
    return gpu_.getMultisampleCapabilities();
  }

  void beginFrame(uint64_t frameIndex);
  void collectGarbage(uint64_t completedFrameIndex);

  [[nodiscard]] PoolStats stats() const;

  [[nodiscard]] uint32_t materialTableIndex(MaterialRef ref) const;
  [[nodiscard]] MaterialRef
  modelMaterialForSubmesh(ModelRef model, uint32_t submeshIndex) const;
  bool setModelMaterialForSource(ModelRef model, uint32_t sourceMaterialIndex,
                                 MaterialRef material);
  void setModelMaterialForAllSources(ModelRef model, MaterialRef material);

private:
  static constexpr uint64_t kRetireFrameUnset =
      std::numeric_limits<uint64_t>::max();

  struct TextureSlot {
    uint32_t refCount = 0;
    uint64_t retireAfterFrame = kRetireFrameUnset;
    OwnedTextureHandle owner;
    TextureRecord record;

    explicit TextureSlot(
        std::pmr::memory_resource *memory = std::pmr::get_default_resource())
        : record(memory) {}
  };

  struct MaterialSlot {
    uint32_t refCount = 0;
    uint64_t retireAfterFrame = kRetireFrameUnset;
    MaterialRecord record;

    explicit MaterialSlot(
        std::pmr::memory_resource *memory = std::pmr::get_default_resource())
        : record(memory) {}
  };

  struct ModelSlot {
    uint32_t refCount = 0;
    uint64_t retireAfterFrame = kRetireFrameUnset;
    ModelRecord record;

    explicit ModelSlot(
        std::pmr::memory_resource *memory = std::pmr::get_default_resource())
        : record(memory) {}
  };

  [[nodiscard]] uint64_t retireLagFrames() const;
  [[nodiscard]] TextureRef makeTextureRefForSlot(uint32_t index) const;
  [[nodiscard]] MaterialRef makeMaterialRefForSlot(uint32_t index) const;
  [[nodiscard]] ModelRef makeModelRefForSlot(uint32_t index) const;

  [[nodiscard]] TextureSlot *tryGetSlot(TextureRef ref);
  [[nodiscard]] MaterialSlot *tryGetSlot(MaterialRef ref);
  [[nodiscard]] ModelSlot *tryGetSlot(ModelRef ref);
  [[nodiscard]] const TextureSlot *tryGetSlot(TextureRef ref) const;
  [[nodiscard]] const MaterialSlot *tryGetSlot(MaterialRef ref) const;
  [[nodiscard]] const ModelSlot *tryGetSlot(ModelRef ref) const;

  [[nodiscard]] Result<SlotReservation, std::string> allocateTextureSlot();
  [[nodiscard]] Result<SlotReservation, std::string> allocateMaterialSlot();
  [[nodiscard]] Result<SlotReservation, std::string> allocateModelSlot();

  void destroyTextureSlot(uint32_t index);
  void destroyMaterialSlot(uint32_t index, bool skipRebuild = false);
  void destroyModelSlot(uint32_t index);
  void rebuildPackedMaterialTables();

  [[nodiscard]] ModelRef tryAcquireCachedModel(const ModelKey &key);
  [[nodiscard]] Result<ModelRef, std::string>
  storeAcquiredModel(const ModelKey &key, std::string_view canonicalPath,
                     uint64_t optionsHash, const ModelRequest &request,
                     std::unique_ptr<Model> model);

  [[nodiscard]] static MaterialDesc
  materialDescFromImported(const ImportedMaterialInfo &imported,
                           const MaterialTextureHandles &textures);

  GPUDevice &gpu_;
  std::pmr::memory_resource *memory_ = std::pmr::get_default_resource();
  uint64_t currentFrameIndex_ = 0;

  std::pmr::vector<TextureSlot> textureSlots_;
  std::pmr::vector<MaterialSlot> materialSlots_;
  std::pmr::vector<ModelSlot> modelSlots_;

  SlotPool<MaskedNonZeroGenerationPolicy<kResourceHandleGenerationMask>>
      textureSlotsMeta_;
  SlotPool<MaskedNonZeroGenerationPolicy<kResourceHandleGenerationMask>>
      materialSlotsMeta_;
  SlotPool<MaskedNonZeroGenerationPolicy<kResourceHandleGenerationMask>>
      modelSlotsMeta_;

  std::pmr::vector<MaterialHeaderGpuData> materialHeaderTable_;
  std::pmr::vector<MaterialClearcoatGpuData> materialClearcoatTable_;
  std::pmr::vector<MaterialSheenGpuData> materialSheenTable_;
  std::pmr::vector<MaterialTransmissionGpuData> materialTransmissionTable_;
  std::pmr::vector<MaterialSpecularGpuData> materialSpecularTable_;
  uint64_t materialTableVersion_ = 0;

  HashMap<TextureKey, TextureRef, TextureKeyHash> textureCache_;
  HashMap<MaterialKey, MaterialRef, MaterialKeyHash> materialCache_;
  HashMap<ModelKey, ModelRef, ModelKeyHash> modelCache_;

  std::pmr::vector<TextureRef> pendingRetireTextures_;
  std::pmr::vector<MaterialRef> pendingRetireMaterials_;
  std::pmr::vector<ModelRef> pendingRetireModels_;
  struct Telemetry {
    uint64_t textureAcquireHits = 0;
    uint64_t textureAcquireMisses = 0;
    uint64_t modelAcquireHits = 0;
    uint64_t modelAcquireMisses = 0;
    uint64_t materialAcquireHits = 0;
    uint64_t materialAcquireMisses = 0;
    uint64_t invalidTextureLookups = 0;
    uint64_t staleTextureLookups = 0;
    uint64_t invalidMaterialLookups = 0;
    uint64_t staleMaterialLookups = 0;
    uint64_t invalidModelLookups = 0;
    uint64_t staleModelLookups = 0;
    uint64_t staleTextureReleases = 0;
    uint64_t staleMaterialReleases = 0;
    uint64_t staleModelReleases = 0;
  };
  mutable Telemetry telemetry_{};
};

} // namespace nuri
