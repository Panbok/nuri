#include "nuri/pch.h"

#include "nuri/resources/gpu/resource_manager.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/resources/mesh_importer.h"

namespace nuri {

namespace {

template <typename SlotT, typename RefT>
[[nodiscard]] bool isSlotLiveForRef(const std::pmr::vector<SlotT> &slots,
                                    RefT ref) {
  if (!isValid(ref)) {
    return false;
  }
  const ResourceHandleParts parts = unpackResourceHandle(ref.value);
  if (parts.index >= slots.size()) {
    return false;
  }
  const SlotT &slot = slots[parts.index];
  return slot.live && slot.generation == parts.generation;
}

template <typename SlotT, typename RefT>
[[nodiscard]] SlotT *tryGetSlotImpl(std::pmr::vector<SlotT> &slots, RefT ref) {
  if (!isValid(ref)) {
    return nullptr;
  }
  const ResourceHandleParts parts = unpackResourceHandle(ref.value);
  if (parts.index >= slots.size()) {
    return nullptr;
  }
  SlotT &slot = slots[parts.index];
  if (!slot.live || slot.generation != parts.generation) {
    return nullptr;
  }
  return &slot;
}

template <typename SlotT, typename RefT>
[[nodiscard]] const SlotT *tryGetSlotImpl(const std::pmr::vector<SlotT> &slots,
                                          RefT ref) {
  if (!isValid(ref)) {
    return nullptr;
  }
  const ResourceHandleParts parts = unpackResourceHandle(ref.value);
  if (parts.index >= slots.size()) {
    return nullptr;
  }
  const SlotT &slot = slots[parts.index];
  if (!slot.live || slot.generation != parts.generation) {
    return nullptr;
  }
  return &slot;
}

template <typename Fn>
void forEachTextureRef(const MaterialRequest::TextureRefs &refs, Fn &&fn) {
  fn(refs.baseColor);
  fn(refs.metallicRoughness);
  fn(refs.normal);
  fn(refs.occlusion);
  fn(refs.emissive);
  fn(refs.clearcoat);
  fn(refs.clearcoatRoughness);
  fn(refs.clearcoatNormal);
}

} // namespace

MaterialRef ModelRecord::materialForSubmesh(uint32_t submeshIndex) const {
  if (!model || submeshIndex >= model->submeshes().size()) {
    return kInvalidMaterialRef;
  }
  return materialForSource(model->submeshes()[submeshIndex].materialIndex);
}

ResourceManager::ResourceManager(GPUDevice &gpu,
                                 std::pmr::memory_resource *memory)
    : gpu_(gpu),
      memory_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      textureSlots_(memory_), materialSlots_(memory_), modelSlots_(memory_),
      freeTextureSlots_(memory_), freeMaterialSlots_(memory_),
      freeModelSlots_(memory_), materialGpuTable_(memory_), textureCache_(),
      materialCache_(), modelCache_() {}

ResourceManager::~ResourceManager() {
  // Dependency order:
  // model mappings reference materials, materials reference textures.
  for (uint32_t i = 0; i < modelSlots_.size(); ++i) {
    if (modelSlots_[i].live) {
      destroyModelSlot(i);
    }
  }
  for (uint32_t i = 0; i < materialSlots_.size(); ++i) {
    if (materialSlots_[i].live) {
      destroyMaterialSlot(i);
    }
  }
  for (uint32_t i = 0; i < textureSlots_.size(); ++i) {
    if (textureSlots_[i].live) {
      destroyTextureSlot(i);
    }
  }
}

uint64_t ResourceManager::retireLagFrames() const {
  return static_cast<uint64_t>(std::max(1u, gpu_.getSwapchainImageCount())) +
         1ull;
}

TextureRef ResourceManager::makeTextureRefForSlot(uint32_t index) const {
  return makeTextureRef(index, textureSlots_[index].generation);
}

MaterialRef ResourceManager::makeMaterialRefForSlot(uint32_t index) const {
  return makeMaterialRef(index, materialSlots_[index].generation);
}

ModelRef ResourceManager::makeModelRefForSlot(uint32_t index) const {
  return makeModelRef(index, modelSlots_[index].generation);
}

ResourceManager::TextureSlot *ResourceManager::tryGetSlot(TextureRef ref) {
  return tryGetSlotImpl(textureSlots_, ref);
}

ResourceManager::MaterialSlot *ResourceManager::tryGetSlot(MaterialRef ref) {
  return tryGetSlotImpl(materialSlots_, ref);
}

ResourceManager::ModelSlot *ResourceManager::tryGetSlot(ModelRef ref) {
  return tryGetSlotImpl(modelSlots_, ref);
}

const ResourceManager::TextureSlot *
ResourceManager::tryGetSlot(TextureRef ref) const {
  return tryGetSlotImpl(textureSlots_, ref);
}

const ResourceManager::MaterialSlot *
ResourceManager::tryGetSlot(MaterialRef ref) const {
  return tryGetSlotImpl(materialSlots_, ref);
}

const ResourceManager::ModelSlot *
ResourceManager::tryGetSlot(ModelRef ref) const {
  return tryGetSlotImpl(modelSlots_, ref);
}

uint32_t ResourceManager::allocateTextureSlot() {
  if (!freeTextureSlots_.empty()) {
    const uint32_t index = freeTextureSlots_.back();
    freeTextureSlots_.pop_back();
    return index;
  }
  textureSlots_.emplace_back(memory_);
  return static_cast<uint32_t>(textureSlots_.size() - 1u);
}

uint32_t ResourceManager::allocateMaterialSlot() {
  if (!freeMaterialSlots_.empty()) {
    const uint32_t index = freeMaterialSlots_.back();
    freeMaterialSlots_.pop_back();
    return index;
  }
  materialSlots_.emplace_back(memory_);
  materialGpuTable_.push_back(MaterialGpuData{});
  return static_cast<uint32_t>(materialSlots_.size() - 1u);
}

uint32_t ResourceManager::allocateModelSlot() {
  if (!freeModelSlots_.empty()) {
    const uint32_t index = freeModelSlots_.back();
    freeModelSlots_.pop_back();
    return index;
  }
  modelSlots_.emplace_back(memory_);
  return static_cast<uint32_t>(modelSlots_.size() - 1u);
}

void ResourceManager::destroyTextureSlot(uint32_t index) {
  TextureSlot &slot = textureSlots_[index];
  if (!slot.live) {
    return;
  }

  if (nuri::isValid(slot.record.texture)) {
    gpu_.destroyTexture(slot.record.texture);
  }

  const TextureKey key{
      .canonicalPath = std::string(slot.record.canonicalPath),
      .optionsHash = hashTextureLoadOptions(slot.record.loadOptions),
      .kind = slot.record.sourceKind,
  };
  textureCache_.erase(key);

  slot.live = false;
  slot.refCount = 0;
  slot.retireAfterFrame = kRetireFrameUnset;
  slot.generation = nextResourceGeneration(slot.generation);
  slot.record = TextureRecord(memory_);
  freeTextureSlots_.push_back(index);
}

void ResourceManager::destroyMaterialSlot(uint32_t index) {
  MaterialSlot &slot = materialSlots_[index];
  if (!slot.live) {
    return;
  }

  forEachTextureRef(slot.record.textureRefs, [this](TextureRef textureRef) {
    if (isValid(textureRef)) {
      release(textureRef);
    }
  });

  const MaterialKey key{
      .descHash = slot.record.descHash,
      .sourceIdentity = std::string(slot.record.sourceIdentity),
  };
  materialCache_.erase(key);

  slot.live = false;
  slot.refCount = 0;
  slot.retireAfterFrame = kRetireFrameUnset;
  slot.generation = nextResourceGeneration(slot.generation);
  slot.record = MaterialRecord(memory_);
  if (index < materialGpuTable_.size()) {
    materialGpuTable_[index] = MaterialGpuData{};
  }
  ++materialTableVersion_;
  freeMaterialSlots_.push_back(index);
}

void ResourceManager::destroyModelSlot(uint32_t index) {
  ModelSlot &slot = modelSlots_[index];
  if (!slot.live) {
    return;
  }

  for (const MaterialRef mappedMaterial : slot.record.sourceMaterialToRuntime) {
    if (isValid(mappedMaterial)) {
      release(mappedMaterial);
    }
  }

  const ModelKey key{
      .canonicalPath = std::string(slot.record.canonicalPath),
      .importOptionsHash = slot.record.importOptionsHash,
  };
  modelCache_.erase(key);

  slot.live = false;
  slot.refCount = 0;
  slot.retireAfterFrame = kRetireFrameUnset;
  slot.generation = nextResourceGeneration(slot.generation);
  slot.record = ModelRecord(memory_);
  freeModelSlots_.push_back(index);
}

Result<TextureRef, std::string>
ResourceManager::acquireTexture(const TextureRequest &request) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (request.path.empty()) {
    return Result<TextureRef, std::string>::makeError(
        "ResourceManager::acquireTexture: path is empty");
  }

  const std::string canonicalPath = canonicalizeResourcePath(request.path);
  TextureKey key{
      .canonicalPath = canonicalPath,
      .optionsHash = hashTextureLoadOptions(request.loadOptions),
      .kind = request.kind,
  };

  if (auto it = textureCache_.find(key); it != textureCache_.end()) {
    if (TextureSlot *cached = tryGetSlot(it->second)) {
      ++cached->refCount;
      cached->retireAfterFrame = kRetireFrameUnset;
      ++telemetry_.textureAcquireHits;
      return Result<TextureRef, std::string>::makeResult(it->second);
    }
    textureCache_.erase(it);
  }
  ++telemetry_.textureAcquireMisses;

  Result<std::unique_ptr<Texture>, std::string> textureResult =
      Result<std::unique_ptr<Texture>, std::string>::makeError(
          "ResourceManager::acquireTexture: uninitialized result");

  switch (request.kind) {
  case TextureRequestKind::Texture2D:
    textureResult = Texture::loadTexture(
        gpu_, canonicalPath, request.loadOptions, request.debugName);
    break;
  case TextureRequestKind::Ktx2Texture2D:
    textureResult =
        Texture::loadTextureKtx2(gpu_, canonicalPath, request.debugName);
    break;
  case TextureRequestKind::Ktx2Cubemap:
    textureResult =
        Texture::loadCubemapKtx2(gpu_, canonicalPath, request.debugName);
    break;
  case TextureRequestKind::EquirectHdrCubemap:
    textureResult = Texture::loadCubemapFromEquirectangularHDR(
        gpu_, canonicalPath, request.debugName);
    break;
  }

  if (textureResult.hasError()) {
    return Result<TextureRef, std::string>::makeError(textureResult.error());
  }

  std::unique_ptr<Texture> texture = std::move(textureResult.value());
  if (!texture || !texture->valid()) {
    return Result<TextureRef, std::string>::makeError(
        "ResourceManager::acquireTexture: loaded texture is invalid");
  }

  const uint32_t slotIndex = allocateTextureSlot();
  TextureSlot &slot = textureSlots_[slotIndex];
  const TextureRef ref = makeTextureRefForSlot(slotIndex);
  slot.live = true;
  slot.refCount = 1;
  slot.retireAfterFrame = kRetireFrameUnset;

  slot.record = TextureRecord(memory_);
  slot.record.ref = ref;
  slot.record.texture = texture->handle();
  slot.record.bindlessIndex = gpu_.getTextureBindlessIndex(texture->handle());
  slot.record.type = texture->type();
  slot.record.format = texture->format();
  slot.record.usage = texture->usage();
  slot.record.dimensions = texture->dimensions();
  slot.record.storage = texture->storage();
  slot.record.numLayers = texture->numLayers();
  slot.record.numSamples = texture->numSamples();
  slot.record.numMipLevels = texture->numMipLevels();
  slot.record.sourceKind = request.kind;
  slot.record.loadOptions = request.loadOptions;
  slot.record.canonicalPath = canonicalPath;
  slot.record.debugName = request.debugName;

  texture.reset();

  textureCache_.emplace(std::move(key), ref);
  return Result<TextureRef, std::string>::makeResult(ref);
}

Result<ModelRef, std::string>
ResourceManager::acquireModel(const ModelRequest &request) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (request.path.empty()) {
    return Result<ModelRef, std::string>::makeError(
        "ResourceManager::acquireModel: path is empty");
  }

  const std::string canonicalPath = canonicalizeResourcePath(request.path);
  const uint64_t optionsHash = hashModelImportOptions(request.importOptions);
  ModelKey key{.canonicalPath = canonicalPath,
               .importOptionsHash = optionsHash};

  if (auto it = modelCache_.find(key); it != modelCache_.end()) {
    if (ModelSlot *cached = tryGetSlot(it->second)) {
      ++cached->refCount;
      cached->retireAfterFrame = kRetireFrameUnset;
      ++telemetry_.modelAcquireHits;
      return Result<ModelRef, std::string>::makeResult(it->second);
    }
    modelCache_.erase(it);
  }
  ++telemetry_.modelAcquireMisses;

  auto modelResult = Model::createFromFile(
      gpu_, canonicalPath, request.importOptions, memory_, request.debugName);
  if (modelResult.hasError()) {
    return Result<ModelRef, std::string>::makeError(modelResult.error());
  }

  std::unique_ptr<Model> model = std::move(modelResult.value());
  if (!model) {
    return Result<ModelRef, std::string>::makeError(
        "ResourceManager::acquireModel: model creation returned null");
  }

  const uint32_t slotIndex = allocateModelSlot();
  ModelSlot &slot = modelSlots_[slotIndex];
  const ModelRef ref = makeModelRefForSlot(slotIndex);
  slot.live = true;
  slot.refCount = 1;
  slot.retireAfterFrame = kRetireFrameUnset;

  slot.record = ModelRecord(memory_);
  slot.record.ref = ref;
  slot.record.model = std::move(model);
  slot.record.canonicalPath = canonicalPath;
  slot.record.importOptionsHash = optionsHash;
  slot.record.sourceMaterialToRuntime.assign(
      slot.record.model->sourceMaterialCount(), kInvalidMaterialRef);

  modelCache_.emplace(std::move(key), ref);
  return Result<ModelRef, std::string>::makeResult(ref);
}

Result<MaterialRef, std::string>
ResourceManager::acquireMaterial(const MaterialRequest &request) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  MaterialDesc resolvedDesc = request.desc;
  const auto resolveTextureSlot =
      [this](TextureRef textureRef, TextureHandle &outHandle,
             std::string_view slotName) -> Result<bool, std::string> {
    if (!isValid(textureRef)) {
      return Result<bool, std::string>::makeResult(true);
    }
    const TextureRecord *record = tryGet(textureRef);
    if (record == nullptr) {
      return Result<bool, std::string>::makeError(
          "ResourceManager::acquireMaterial: stale texture ref for slot '" +
          std::string(slotName) + "'");
    }
    outHandle = record->texture;
    return Result<bool, std::string>::makeResult(true);
  };

  auto baseColorResult =
      resolveTextureSlot(request.textureRefs.baseColor,
                         resolvedDesc.textures.baseColor, "baseColor");
  if (baseColorResult.hasError()) {
    return Result<MaterialRef, std::string>::makeError(baseColorResult.error());
  }
  auto metallicRoughnessResult = resolveTextureSlot(
      request.textureRefs.metallicRoughness,
      resolvedDesc.textures.metallicRoughness, "metallicRoughness");
  if (metallicRoughnessResult.hasError()) {
    return Result<MaterialRef, std::string>::makeError(
        metallicRoughnessResult.error());
  }
  auto normalResult = resolveTextureSlot(
      request.textureRefs.normal, resolvedDesc.textures.normal, "normal");
  if (normalResult.hasError()) {
    return Result<MaterialRef, std::string>::makeError(normalResult.error());
  }
  auto occlusionResult =
      resolveTextureSlot(request.textureRefs.occlusion,
                         resolvedDesc.textures.occlusion, "occlusion");
  if (occlusionResult.hasError()) {
    return Result<MaterialRef, std::string>::makeError(occlusionResult.error());
  }
  auto emissiveResult = resolveTextureSlot(
      request.textureRefs.emissive, resolvedDesc.textures.emissive, "emissive");
  if (emissiveResult.hasError()) {
    return Result<MaterialRef, std::string>::makeError(emissiveResult.error());
  }
  auto clearcoatResult =
      resolveTextureSlot(request.textureRefs.clearcoat,
                         resolvedDesc.textures.clearcoat, "clearcoat");
  if (clearcoatResult.hasError()) {
    return Result<MaterialRef, std::string>::makeError(clearcoatResult.error());
  }
  auto clearcoatRoughnessResult = resolveTextureSlot(
      request.textureRefs.clearcoatRoughness,
      resolvedDesc.textures.clearcoatRoughness, "clearcoatRoughness");
  if (clearcoatRoughnessResult.hasError()) {
    return Result<MaterialRef, std::string>::makeError(
        clearcoatRoughnessResult.error());
  }
  auto clearcoatNormalResult = resolveTextureSlot(
      request.textureRefs.clearcoatNormal,
      resolvedDesc.textures.clearcoatNormal, "clearcoatNormal");
  if (clearcoatNormalResult.hasError()) {
    return Result<MaterialRef, std::string>::makeError(
        clearcoatNormalResult.error());
  }

  resolvedDesc.featureMask = kMaterialFeatureMetallicRoughness;
  if (resolvedDesc.sheenWeight > 0.0f) {
    resolvedDesc.featureMask |= kMaterialFeatureSheen;
  }
  const bool hasAnyClearcoatTexture =
      isValid(request.textureRefs.clearcoat) ||
      isValid(request.textureRefs.clearcoatRoughness) ||
      isValid(request.textureRefs.clearcoatNormal);
  if (resolvedDesc.clearcoatFactor > 0.0f && hasAnyClearcoatTexture) {
    resolvedDesc.featureMask |= kMaterialFeatureClearcoat;
  }

  const uint64_t descHash = hashMaterialDesc(resolvedDesc);
  MaterialKey key{.descHash = descHash,
                  .sourceIdentity = request.sourceIdentity};

  if (auto it = materialCache_.find(key); it != materialCache_.end()) {
    if (MaterialSlot *cached = tryGetSlot(it->second)) {
      ++cached->refCount;
      cached->retireAfterFrame = kRetireFrameUnset;
      ++telemetry_.materialAcquireHits;
      return Result<MaterialRef, std::string>::makeResult(it->second);
    }
    materialCache_.erase(it);
  }
  ++telemetry_.materialAcquireMisses;

  auto materialResult = Material::create(gpu_, resolvedDesc, request.debugName);
  if (materialResult.hasError()) {
    return Result<MaterialRef, std::string>::makeError(materialResult.error());
  }

  const Material &material = *materialResult.value();
  const uint32_t slotIndex = allocateMaterialSlot();
  MaterialSlot &slot = materialSlots_[slotIndex];
  const MaterialRef ref = makeMaterialRefForSlot(slotIndex);
  slot.live = true;
  slot.refCount = 1;
  slot.retireAfterFrame = kRetireFrameUnset;

  slot.record = MaterialRecord(memory_);
  slot.record.ref = ref;
  slot.record.desc = material.desc();
  slot.record.textureRefs = request.textureRefs;
  slot.record.gpuData = material.gpuData();
  slot.record.descHash = descHash;
  slot.record.debugName = request.debugName;
  slot.record.sourceIdentity = request.sourceIdentity;

  forEachTextureRef(slot.record.textureRefs, [this](TextureRef textureRef) {
    if (isValid(textureRef)) {
      retain(textureRef);
    }
  });

  if (slotIndex >= materialGpuTable_.size()) {
    materialGpuTable_.resize(slotIndex + 1u);
  }
  materialGpuTable_[slotIndex] = slot.record.gpuData;
  ++materialTableVersion_;

  materialCache_.emplace(std::move(key), ref);
  return Result<MaterialRef, std::string>::makeResult(ref);
}

MaterialDesc ResourceManager::materialDescFromImported(
    const ImportedMaterialInfo &imported,
    const MaterialTextureHandles &textures) {
  MaterialDesc desc{};
  desc.baseColorFactor = imported.baseColorFactor;
  desc.emissiveFactor = imported.emissiveFactor;
  desc.metallicFactor = imported.metallicFactor;
  desc.roughnessFactor = imported.roughnessFactor;
  desc.sheenColorFactor = imported.sheenColorFactor;
  desc.sheenWeight = imported.sheenWeight;
  desc.sheenRoughnessFactor = imported.sheenRoughnessFactor;
  desc.clearcoatFactor = imported.clearcoatFactor;
  desc.clearcoatRoughnessFactor = imported.clearcoatRoughnessFactor;
  desc.clearcoatNormalScale = imported.clearcoatNormalScale;
  desc.normalScale = imported.normalScale;
  desc.occlusionStrength = imported.occlusionStrength;
  desc.alphaCutoff = imported.alphaCutoff;
  desc.doubleSided = imported.doubleSided;
  desc.alphaMode = imported.alphaMode;
  desc.featureMask = kMaterialFeatureMetallicRoughness;
  if (desc.sheenWeight > 0.0f) {
    desc.featureMask |= kMaterialFeatureSheen;
  }
  const bool hasAnyClearcoatTexture =
      nuri::isValid(textures.clearcoat) ||
      nuri::isValid(textures.clearcoatRoughness) ||
      nuri::isValid(textures.clearcoatNormal);
  if (desc.clearcoatFactor > 0.0f && hasAnyClearcoatTexture) {
    desc.featureMask |= kMaterialFeatureClearcoat;
  }
  desc.textures = textures;
  desc.uvSets = MaterialTextureUvSets{
      .baseColor = imported.baseColor.uvSet,
      .metallicRoughness = imported.metallicRoughness.uvSet,
      .normal = imported.normal.uvSet,
      .occlusion = imported.occlusion.uvSet,
      .emissive = imported.emissive.uvSet,
      .clearcoat = imported.clearcoat.uvSet,
      .clearcoatRoughness = imported.clearcoatRoughness.uvSet,
      .clearcoatNormal = imported.clearcoatNormal.uvSet,
  };
  desc.samplers = MaterialTextureSamplers{
      .baseColor = imported.baseColor.samplerIndex,
      .metallicRoughness = imported.metallicRoughness.samplerIndex,
      .normal = imported.normal.samplerIndex,
      .occlusion = imported.occlusion.samplerIndex,
      .emissive = imported.emissive.samplerIndex,
      .clearcoat = imported.clearcoat.samplerIndex,
      .clearcoatRoughness = imported.clearcoatRoughness.samplerIndex,
      .clearcoatNormal = imported.clearcoatNormal.samplerIndex,
  };
  return desc;
}

Result<ImportedMaterialBatch, std::string>
ResourceManager::acquireMaterialsFromModel(
    const ImportedMaterialRequest &request) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (request.modelPath.empty()) {
    return Result<ImportedMaterialBatch, std::string>::makeError(
        "ResourceManager::acquireMaterialsFromModel: model path is empty");
  }
  ModelSlot *modelSlot = tryGetSlot(request.model);
  if (modelSlot == nullptr || !modelSlot->record.model) {
    return Result<ImportedMaterialBatch, std::string>::makeError(
        "ResourceManager::acquireMaterialsFromModel: invalid model handle");
  }

  auto materialInfoResult =
      MeshImporter::loadMaterialInfoFromFile(request.modelPath);
  if (materialInfoResult.hasError()) {
    return Result<ImportedMaterialBatch, std::string>::makeError(
        "ResourceManager::acquireMaterialsFromModel: failed to parse material "
        "metadata: " +
        materialInfoResult.error());
  }

  const ImportedMaterialSet &materialSet = materialInfoResult.value();
  ImportedMaterialBatch batch{};

  const auto acquireTextureRef =
      [this](const ImportedMaterialTexture &slotData, bool srgb,
             std::string_view debugName) -> Result<TextureRef, std::string> {
    if (slotData.path.empty() || slotData.isEmbedded) {
      return Result<TextureRef, std::string>::makeResult(kInvalidTextureRef);
    }

    TextureRequest textureRequest{};
    textureRequest.path = slotData.path;
    textureRequest.loadOptions =
        TextureLoadOptions{.srgb = srgb, .generateMipmaps = true};
    textureRequest.kind = TextureRequestKind::Texture2D;
    textureRequest.debugName = std::string(debugName);

    auto textureRefResult = acquireTexture(textureRequest);
    if (textureRefResult.hasError()) {
      return Result<TextureRef, std::string>::makeError(
          textureRefResult.error());
    }
    return Result<TextureRef, std::string>::makeResult(
        textureRefResult.value());
  };

  for (uint32_t sourceMaterialIndex = 0;
       sourceMaterialIndex < materialSet.materials.size();
       ++sourceMaterialIndex) {
    const ImportedMaterialInfo &imported =
        materialSet.materials[sourceMaterialIndex];

    MaterialRequest::TextureRefs textureRefs{};

    auto baseColorResult =
        acquireTextureRef(imported.baseColor, true,
                          request.debugNamePrefix + "_base_color_" +
                              std::to_string(sourceMaterialIndex));
    if (baseColorResult.hasError()) {
      NURI_LOG_WARNING(
          "ResourceManager::acquireMaterialsFromModel: baseColor load failed "
          "for material %u: %s",
          sourceMaterialIndex, baseColorResult.error().c_str());
    } else {
      textureRefs.baseColor = baseColorResult.value();
    }

    auto metallicRoughnessResult =
        acquireTextureRef(imported.metallicRoughness, false,
                          request.debugNamePrefix + "_metal_rough_" +
                              std::to_string(sourceMaterialIndex));
    if (metallicRoughnessResult.hasError()) {
      NURI_LOG_WARNING(
          "ResourceManager::acquireMaterialsFromModel: metal/rough load "
          "failed for material %u: %s",
          sourceMaterialIndex, metallicRoughnessResult.error().c_str());
    } else {
      textureRefs.metallicRoughness = metallicRoughnessResult.value();
    }

    auto normalResult =
        acquireTextureRef(imported.normal, false,
                          request.debugNamePrefix + "_normal_" +
                              std::to_string(sourceMaterialIndex));
    if (normalResult.hasError()) {
      NURI_LOG_WARNING(
          "ResourceManager::acquireMaterialsFromModel: normal load failed "
          "for material %u: %s",
          sourceMaterialIndex, normalResult.error().c_str());
    } else {
      textureRefs.normal = normalResult.value();
    }

    auto occlusionResult =
        acquireTextureRef(imported.occlusion, false,
                          request.debugNamePrefix + "_occlusion_" +
                              std::to_string(sourceMaterialIndex));
    if (occlusionResult.hasError()) {
      NURI_LOG_WARNING(
          "ResourceManager::acquireMaterialsFromModel: occlusion load failed "
          "for material %u: %s",
          sourceMaterialIndex, occlusionResult.error().c_str());
    } else {
      textureRefs.occlusion = occlusionResult.value();
    }

    auto emissiveResult =
        acquireTextureRef(imported.emissive, true,
                          request.debugNamePrefix + "_emissive_" +
                              std::to_string(sourceMaterialIndex));
    if (emissiveResult.hasError()) {
      NURI_LOG_WARNING(
          "ResourceManager::acquireMaterialsFromModel: emissive load failed "
          "for material %u: %s",
          sourceMaterialIndex, emissiveResult.error().c_str());
    } else {
      textureRefs.emissive = emissiveResult.value();
    }

    auto clearcoatResult =
        acquireTextureRef(imported.clearcoat, false,
                          request.debugNamePrefix + "_clearcoat_" +
                              std::to_string(sourceMaterialIndex));
    if (clearcoatResult.hasError()) {
      NURI_LOG_WARNING(
          "ResourceManager::acquireMaterialsFromModel: clearcoat load failed "
          "for material %u: %s",
          sourceMaterialIndex, clearcoatResult.error().c_str());
    } else {
      textureRefs.clearcoat = clearcoatResult.value();
    }

    auto clearcoatRoughnessResult =
        acquireTextureRef(imported.clearcoatRoughness, false,
                          request.debugNamePrefix + "_clearcoat_roughness_" +
                              std::to_string(sourceMaterialIndex));
    if (clearcoatRoughnessResult.hasError()) {
      NURI_LOG_WARNING(
          "ResourceManager::acquireMaterialsFromModel: clearcoat roughness "
          "load failed for material %u: %s",
          sourceMaterialIndex, clearcoatRoughnessResult.error().c_str());
    } else {
      textureRefs.clearcoatRoughness = clearcoatRoughnessResult.value();
    }

    auto clearcoatNormalResult =
        acquireTextureRef(imported.clearcoatNormal, false,
                          request.debugNamePrefix + "_clearcoat_normal_" +
                              std::to_string(sourceMaterialIndex));
    if (clearcoatNormalResult.hasError()) {
      NURI_LOG_WARNING(
          "ResourceManager::acquireMaterialsFromModel: clearcoat normal load "
          "failed for material %u: %s",
          sourceMaterialIndex, clearcoatNormalResult.error().c_str());
    } else {
      textureRefs.clearcoatNormal = clearcoatNormalResult.value();
    }

    const MaterialTextureHandles emptyHandles{};
    const MaterialDesc desc = materialDescFromImported(imported, emptyHandles);
    const std::string sourceIdentity =
        canonicalizeResourcePath(request.modelPath) + "#" +
        std::to_string(sourceMaterialIndex);
    const std::string debugName =
        imported.name.empty() ? request.debugNamePrefix + "_material_" +
                                    std::to_string(sourceMaterialIndex)
                              : request.debugNamePrefix + "_" + imported.name;

    auto acquireMaterialResult = acquireMaterial(MaterialRequest{
        .desc = desc,
        .textureRefs = textureRefs,
        .debugName = debugName,
        .sourceIdentity = sourceIdentity,
    });
    forEachTextureRef(textureRefs, [this](TextureRef textureRef) {
      if (isValid(textureRef)) {
        release(textureRef);
      }
    });
    if (acquireMaterialResult.hasError()) {
      NURI_LOG_WARNING(
          "ResourceManager::acquireMaterialsFromModel: material acquire failed "
          "for source material %u: %s",
          sourceMaterialIndex, acquireMaterialResult.error().c_str());
      continue;
    }

    const MaterialRef runtimeMaterial = acquireMaterialResult.value();
    const bool mappedToModel = setModelMaterialForSource(
        request.model, sourceMaterialIndex, runtimeMaterial);
    if (!mappedToModel) {
      NURI_LOG_DEBUG(
          "ResourceManager::acquireMaterialsFromModel: source material %u not "
          "mapped to model",
          sourceMaterialIndex);
    }

    if (!isValid(batch.firstMaterial)) {
      batch.firstMaterial = runtimeMaterial;
      retain(runtimeMaterial);
    }
    ++batch.createdMaterialCount;
    release(runtimeMaterial);
  }

  return Result<ImportedMaterialBatch, std::string>::makeResult(batch);
}

void ResourceManager::retain(TextureRef ref) {
  if (!isValid(ref)) {
    return;
  }
  TextureSlot *slot = tryGetSlot(ref);
  if (slot == nullptr) {
    NURI_ASSERT(false, "ResourceManager::retain(TextureRef): stale handle");
    return;
  }
  ++slot->refCount;
  slot->retireAfterFrame = kRetireFrameUnset;
}

void ResourceManager::release(TextureRef ref) {
  if (!isValid(ref)) {
    ++telemetry_.staleTextureReleases;
    return;
  }
  TextureSlot *slot = tryGetSlot(ref);
  if (slot == nullptr) {
    ++telemetry_.staleTextureReleases;
    NURI_ASSERT(false, "ResourceManager::release(TextureRef): stale handle");
    return;
  }
  if (slot->refCount == 0u) {
    ++telemetry_.staleTextureReleases;
    NURI_ASSERT(false,
                "ResourceManager::release(TextureRef): refcount underflow");
    return;
  }
  --slot->refCount;
  if (slot->refCount == 0u) {
    slot->retireAfterFrame = currentFrameIndex_ + retireLagFrames();
  }
}

void ResourceManager::retain(ModelRef ref) {
  if (!isValid(ref)) {
    return;
  }
  ModelSlot *slot = tryGetSlot(ref);
  if (slot == nullptr) {
    NURI_ASSERT(false, "ResourceManager::retain(ModelRef): stale handle");
    return;
  }
  ++slot->refCount;
  slot->retireAfterFrame = kRetireFrameUnset;
}

void ResourceManager::release(ModelRef ref) {
  if (!isValid(ref)) {
    ++telemetry_.staleModelReleases;
    return;
  }
  ModelSlot *slot = tryGetSlot(ref);
  if (slot == nullptr) {
    ++telemetry_.staleModelReleases;
    NURI_ASSERT(false, "ResourceManager::release(ModelRef): stale handle");
    return;
  }
  if (slot->refCount == 0u) {
    ++telemetry_.staleModelReleases;
    NURI_ASSERT(false,
                "ResourceManager::release(ModelRef): refcount underflow");
    return;
  }
  --slot->refCount;
  if (slot->refCount == 0u) {
    slot->retireAfterFrame = currentFrameIndex_ + retireLagFrames();
  }
}

void ResourceManager::retain(MaterialRef ref) {
  if (!isValid(ref)) {
    return;
  }
  MaterialSlot *slot = tryGetSlot(ref);
  if (slot == nullptr) {
    NURI_ASSERT(false, "ResourceManager::retain(MaterialRef): stale handle");
    return;
  }
  ++slot->refCount;
  slot->retireAfterFrame = kRetireFrameUnset;
}

void ResourceManager::release(MaterialRef ref) {
  if (!isValid(ref)) {
    ++telemetry_.staleMaterialReleases;
    return;
  }
  MaterialSlot *slot = tryGetSlot(ref);
  if (slot == nullptr) {
    ++telemetry_.staleMaterialReleases;
    NURI_ASSERT(false, "ResourceManager::release(MaterialRef): stale handle");
    return;
  }
  if (slot->refCount == 0u) {
    ++telemetry_.staleMaterialReleases;
    NURI_ASSERT(false,
                "ResourceManager::release(MaterialRef): refcount underflow");
    return;
  }
  --slot->refCount;
  if (slot->refCount == 0u) {
    slot->retireAfterFrame = currentFrameIndex_ + retireLagFrames();
  }
}

bool ResourceManager::owns(TextureRef ref) const noexcept {
  return isSlotLiveForRef(textureSlots_, ref);
}

bool ResourceManager::owns(ModelRef ref) const noexcept {
  return isSlotLiveForRef(modelSlots_, ref);
}

bool ResourceManager::owns(MaterialRef ref) const noexcept {
  return isSlotLiveForRef(materialSlots_, ref);
}

const TextureRecord *ResourceManager::tryGet(TextureRef ref) const {
  if (!isValid(ref)) {
    ++telemetry_.invalidTextureLookups;
    return nullptr;
  }
  const TextureSlot *slot = tryGetSlot(ref);
  if (slot == nullptr) {
    ++telemetry_.staleTextureLookups;
  }
  return slot != nullptr ? &slot->record : nullptr;
}

const ModelRecord *ResourceManager::tryGet(ModelRef ref) const {
  if (!isValid(ref)) {
    ++telemetry_.invalidModelLookups;
    return nullptr;
  }
  const ModelSlot *slot = tryGetSlot(ref);
  if (slot == nullptr) {
    ++telemetry_.staleModelLookups;
  }
  return slot != nullptr ? &slot->record : nullptr;
}

const MaterialRecord *ResourceManager::tryGet(MaterialRef ref) const {
  if (!isValid(ref)) {
    ++telemetry_.invalidMaterialLookups;
    return nullptr;
  }
  const MaterialSlot *slot = tryGetSlot(ref);
  if (slot == nullptr) {
    ++telemetry_.staleMaterialLookups;
  }
  return slot != nullptr ? &slot->record : nullptr;
}

void ResourceManager::beginFrame(uint64_t frameIndex) {
  currentFrameIndex_ = frameIndex;
}

void ResourceManager::collectGarbage(uint64_t completedFrameIndex) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);

  // Keep destruction order consistent with dependencies:
  // models -> materials -> textures.
  for (uint32_t i = 0; i < modelSlots_.size(); ++i) {
    const ModelSlot &slot = modelSlots_[i];
    if (!slot.live || slot.refCount != 0u ||
        slot.retireAfterFrame == kRetireFrameUnset ||
        completedFrameIndex < slot.retireAfterFrame) {
      continue;
    }
    destroyModelSlot(i);
  }

  for (uint32_t i = 0; i < materialSlots_.size(); ++i) {
    const MaterialSlot &slot = materialSlots_[i];
    if (!slot.live || slot.refCount != 0u ||
        slot.retireAfterFrame == kRetireFrameUnset ||
        completedFrameIndex < slot.retireAfterFrame) {
      continue;
    }
    destroyMaterialSlot(i);
  }

  for (uint32_t i = 0; i < textureSlots_.size(); ++i) {
    const TextureSlot &slot = textureSlots_[i];
    if (!slot.live || slot.refCount != 0u ||
        slot.retireAfterFrame == kRetireFrameUnset ||
        completedFrameIndex < slot.retireAfterFrame) {
      continue;
    }
    destroyTextureSlot(i);
  }
}

PoolStats ResourceManager::stats() const {
  PoolStats s{};
  for (const TextureSlot &slot : textureSlots_) {
    if (slot.live) {
      if (slot.refCount > 0u) {
        ++s.liveTextures;
      } else {
        ++s.retiredTextures;
      }
    }
  }
  for (const MaterialSlot &slot : materialSlots_) {
    if (slot.live) {
      if (slot.refCount > 0u) {
        ++s.liveMaterials;
      } else {
        ++s.retiredMaterials;
      }
    }
  }
  for (const ModelSlot &slot : modelSlots_) {
    if (slot.live) {
      if (slot.refCount > 0u) {
        ++s.liveModels;
      } else {
        ++s.retiredModels;
      }
    }
  }
  s.textureCacheEntries = textureCache_.size();
  s.materialCacheEntries = materialCache_.size();
  s.modelCacheEntries = modelCache_.size();
  s.textureAcquireHits = telemetry_.textureAcquireHits;
  s.textureAcquireMisses = telemetry_.textureAcquireMisses;
  s.modelAcquireHits = telemetry_.modelAcquireHits;
  s.modelAcquireMisses = telemetry_.modelAcquireMisses;
  s.materialAcquireHits = telemetry_.materialAcquireHits;
  s.materialAcquireMisses = telemetry_.materialAcquireMisses;
  s.invalidTextureLookups = telemetry_.invalidTextureLookups;
  s.staleTextureLookups = telemetry_.staleTextureLookups;
  s.invalidMaterialLookups = telemetry_.invalidMaterialLookups;
  s.staleMaterialLookups = telemetry_.staleMaterialLookups;
  s.invalidModelLookups = telemetry_.invalidModelLookups;
  s.staleModelLookups = telemetry_.staleModelLookups;
  s.staleTextureReleases = telemetry_.staleTextureReleases;
  s.staleMaterialReleases = telemetry_.staleMaterialReleases;
  s.staleModelReleases = telemetry_.staleModelReleases;
  return s;
}

uint32_t ResourceManager::materialTableIndex(MaterialRef ref) const {
  if (!isSlotLiveForRef(materialSlots_, ref)) {
    return 0u;
  }
  return indexOf(ref);
}

MaterialRef
ResourceManager::modelMaterialForSubmesh(ModelRef model,
                                         uint32_t submeshIndex) const {
  const ModelRecord *record = tryGet(model);
  if (record == nullptr) {
    return kInvalidMaterialRef;
  }
  return record->materialForSubmesh(submeshIndex);
}

bool ResourceManager::setModelMaterialForSource(ModelRef model,
                                                uint32_t sourceMaterialIndex,
                                                MaterialRef material) {
  ModelSlot *slot = tryGetSlot(model);
  if (slot == nullptr) {
    return false;
  }
  if (sourceMaterialIndex >= slot->record.sourceMaterialToRuntime.size()) {
    return false;
  }
  if (isValid(material) && tryGetSlot(material) == nullptr) {
    return false;
  }

  MaterialRef &mappedMaterial =
      slot->record.sourceMaterialToRuntime[sourceMaterialIndex];
  if (mappedMaterial.value == material.value) {
    return true;
  }

  if (isValid(material)) {
    retain(material);
  }
  if (isValid(mappedMaterial)) {
    release(mappedMaterial);
  }
  mappedMaterial = material;
  return true;
}

void ResourceManager::setModelMaterialForAllSources(ModelRef model,
                                                    MaterialRef material) {
  ModelSlot *slot = tryGetSlot(model);
  if (slot == nullptr) {
    return;
  }
  if (isValid(material) && tryGetSlot(material) == nullptr) {
    return;
  }

  for (uint32_t sourceMaterialIndex = 0;
       sourceMaterialIndex < slot->record.sourceMaterialToRuntime.size();
       ++sourceMaterialIndex) {
    setModelMaterialForSource(model, sourceMaterialIndex, material);
  }
}

} // namespace nuri
