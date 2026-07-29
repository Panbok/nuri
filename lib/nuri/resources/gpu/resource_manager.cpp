#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/resources/mesh_importer.h"
#include "nuri/resources/storage/cache_utils.h"
#include "nuri/resources/storage/material/material_binary_serializer.h"
#include "nuri/resources/storage/texture/dds_texture_pack.h"
#include "nuri/resources/storage/texture/texture_artifact_builder.h"
#include "nuri/resources/storage/texture/texture_cache_utils.h"
#include "nuri/utils/env_utils.h"
#include <unordered_set>
namespace nuri {

namespace {
[[nodiscard]] uint64_t
textureStorageBytes(const TextureRecord &record) noexcept {
  const uint64_t texelBytes = formatTexelBytes(record.format);
  const bool bc7 = record.format == Format::BC7_RGBA_UNORM ||
                   record.format == Format::BC7_RGBA_SRGB;
  if (texelBytes == 0u && !bc7) {
    return 0u;
  }
  uint32_t width = std::max(1u, record.dimensions.width);
  uint32_t height = std::max(1u, record.dimensions.height);
  uint32_t depth = std::max(1u, record.dimensions.depth);
  uint64_t bytes = 0u;
  for (uint32_t mip = 0u; mip < std::max(1u, record.numMipLevels); ++mip) {
    const uint64_t mipBytes =
        bc7 ? static_cast<uint64_t>((width + 3u) / 4u) * ((height + 3u) / 4u) *
                  depth * 16u
            : static_cast<uint64_t>(width) * height * depth * texelBytes;
    bytes += mipBytes * std::max(1u, record.numLayers) *
             std::max(1u, record.numSamples);
    width = std::max(1u, width >> 1u);
    height = std::max(1u, height >> 1u);
    depth = std::max(1u, depth >> 1u);
  }
  return bytes;
}

template <typename T>
[[nodiscard]] MaterialTableDirtyRange
calculateDirtyRange(std::span<const T> previous, std::span<const T> current) {
  static_assert(std::is_trivially_copyable_v<T>);
  const size_t commonSize = std::min(previous.size(), current.size());
  size_t first = commonSize;
  for (size_t index = 0u; index < commonSize; ++index) {
    if (std::memcmp(&previous[index], &current[index], sizeof(T)) != 0) {
      first = index;
      break;
    }
  }
  if (first == commonSize) {
    if (current.size() <= previous.size()) {
      return {};
    }
    first = previous.size();
  }
  return MaterialTableDirtyRange{
      .first = static_cast<uint32_t>(first),
      .count = static_cast<uint32_t>(current.size() - first),
  };
}
[[nodiscard]] TextureRequestKind
resolveTextureRequestKindForPath(std::string_view path,
                                 TextureRequestKind fallback) {
  if (fallback == TextureRequestKind::Texture2D &&
      (hasExtensionCaseInsensitive(path, ".ktx2") ||
       hasExtensionCaseInsensitive(path, ".ktx"))) {
    return TextureRequestKind::Ktx2Texture2D;
  }
  return fallback;
}
[[nodiscard]] Result<bool, std::string>
validateTextureRequestContentContract(const TextureRequest &request) {
  if (request.contentContract !=
      TextureContentContract::NormalRgbCleanVarianceA) {
    return Result<bool, std::string>::makeResult(true);
  }
  if (request.loadOptions.mipSemantic != TextureMipSemantic::NormalMap ||
      request.loadOptions.srgb ||
      request.kind != TextureRequestKind::Ktx2Texture2D) {
    return Result<bool, std::string>::makeError(
        "NormalRgbCleanVarianceA requires a linear, normal-semantic Nuri KTX2 "
        "artifact");
  }
  auto metadata =
      readNativeTextureCacheMetadata(std::filesystem::path(request.path));
  if (metadata.hasError() ||
      metadata.value().contentContract != request.contentContract ||
      metadata.value().contentEncodingVersion !=
          kNormalVarianceEncodingVersion) {
    return Result<bool, std::string>::makeError(
        "NormalRgbCleanVarianceA is not proven by trusted native metadata");
  }
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] bool formatPreservesNormalVarianceAlpha(Format format) noexcept {
  return format == Format::RGBA8_UNORM || format == Format::BC7_RGBA_UNORM;
}
[[nodiscard]] ModelKey makeSceneMeshModelKey(std::string_view canonicalPath,
                                             uint64_t optionsHash,
                                             uint32_t sceneMeshIndex) {
  return ModelKey{.canonicalPath = std::string(canonicalPath),
                  .importOptionsHash = optionsHash,
                  .sceneMeshIndex = sceneMeshIndex};
}
[[nodiscard]] std::string
makeScenePrefabMeshDebugName(uint32_t sceneMeshIndex) {
  return "scene_prefab_mesh_" + std::to_string(sceneMeshIndex);
}
template <typename Pool, typename Ref>
[[nodiscard]] auto tryGetSlotImpl(Pool &pool, Ref ref)
    -> decltype(&pool.slots[0]) {
  if (!isValid(ref)) {
    return nullptr;
  }
  const uint32_t index = indexOf(ref);
  if (index >= pool.slots.size() ||
      !pool.meta.isValid(index, generationOf(ref))) {
    return nullptr;
  }
  return &pool.slots[index];
}
template <typename Ref, typename Pool>
[[nodiscard]] Ref makeRefForSlot(const Pool &pool, uint32_t index) {
  return {packResourceHandle(index, pool.meta.generation(index))};
}
template <typename Pool>
[[nodiscard]] Result<SlotReservation, std::string>
allocateSlot(Pool &pool, std::pmr::memory_resource *memory,
             std::string_view context) {
  const uint32_t count = pool.meta.slotCount();
  if (count > kResourceHandleIndexMask + 1u ||
      (count == kResourceHandleIndexMask + 1u &&
       pool.meta.liveCount() == count)) {
    return Result<SlotReservation, std::string>::makeError(
        std::string(context) + ": slot pool exhausted");
  }
  const SlotReservation slot = pool.meta.acquire();
  if (slot.appended) {
    pool.slots.emplace_back(memory);
  }
  return Result<SlotReservation, std::string>::makeResult(slot);
}
template <typename Pool, typename Ref> void retainRef(Pool &pool, Ref ref) {
  if (auto *slot = tryGetSlotImpl(pool, ref))
    ++slot->refCount;
}
template <typename Pool, typename Ref> void releaseRef(Pool &pool, Ref ref) {
  auto *slot = tryGetSlotImpl(pool, ref);
  if (!slot || slot->refCount == 0u) {
    return;
  }
  if (--slot->refCount == 0u)
    pool.pending.push_back(ref);
}
template <typename Pool, typename Destroy>
uint32_t collectRetired(Pool &pool, Destroy destroy) {
  uint32_t destroyed = 0u;
  for (size_t index = 0; index < pool.pending.size();) {
    const auto ref = pool.pending[index];
    const auto *slot = tryGetSlotImpl(pool, ref);
    if (!slot || slot->refCount != 0u) {
      pool.pending[index] = pool.pending.back();
      pool.pending.pop_back();
    } else {
      destroy(indexOf(ref));
      pool.pending[index] = pool.pending.back();
      pool.pending.pop_back();
      ++destroyed;
    }
  }
  return destroyed;
}
template <typename Pool>
[[nodiscard]] std::pair<uint32_t, uint32_t> resourceCounts(const Pool &pool) {
  uint32_t live = 0u;
  uint32_t retired = 0u;
  for (uint32_t index = 0; index < pool.slots.size(); ++index) {
    if (pool.meta.isLive(index)) {
      pool.slots[index].refCount ? ++live : ++retired;
    }
  }
  return {live, retired};
}
[[nodiscard]] std::optional<SceneMaterialCacheData>
tryLoadSceneMaterialCache(std::string_view sourcePath) {
  auto cacheKeyResult = buildSceneMaterialCacheKey(
      std::filesystem::path(std::string(sourcePath)));
  if (cacheKeyResult.hasError()) {
    NURI_LOG_WARNING("tryLoadSceneMaterialCache: failed to build cache key "
                     "for '%.*s': %s",
                     static_cast<int>(sourcePath.size()), sourcePath.data(),
                     cacheKeyResult.error().c_str());
    return std::nullopt;
  }
  const SceneMaterialCacheKey &cacheKey = cacheKeyResult.value();
  std::error_code ec;
  if (!std::filesystem::exists(cacheKey.cachePath, ec) || ec ||
      !std::filesystem::is_regular_file(cacheKey.cachePath, ec) || ec) {
    return std::nullopt;
  }
  auto readResult = readBinaryFile(cacheKey.cachePath);
  if (readResult.hasError()) {
    NURI_LOG_WARNING("tryLoadSceneMaterialCache: failed to read cache '%s': %s",
                     cacheKey.cachePath.string().c_str(),
                     readResult.error().c_str());
    return std::nullopt;
  }
  const SourceFingerprint sourceFingerprint =
      querySourceFingerprint(cacheKey.normalizedSourcePath);
  MaterialBinaryDeserializeContext context{};
  context.expectedSourcePathHash = cacheKey.sourcePathHash;
  context.validateSourceFingerprint = true;
  context.sourceExists = sourceFingerprint.exists;
  context.sourceSizeBytes = sourceFingerprint.sizeBytes;
  context.sourceMtimeNs = sourceFingerprint.mtimeNs;
  auto deserializeResult =
      materialBinaryDeserialize(readResult.value(), context);
  if (deserializeResult.hasError()) {
    const MaterialBinaryDeserializeError &error = deserializeResult.error();
    if (error.isStale()) {
      NURI_LOG_DEBUG("tryLoadSceneMaterialCache: stale cache '%s': %s",
                     cacheKey.cachePath.string().c_str(),
                     error.message.c_str());
    } else {
      NURI_LOG_WARNING(
          "tryLoadSceneMaterialCache: failed to deserialize cache '%s': %s",
          cacheKey.cachePath.string().c_str(), error.message.c_str());
    }
    return std::nullopt;
  }
  return deserializeResult.value();
}
[[nodiscard]] Result<TextureRef, std::string> acquireExternalImportedTexture(
    ResourceManager &resources, const ImportedMaterialTexture &slotData,
    const TextureLoadOptions &options, TextureRequestKind kind,
    std::string_view debugName, DdsTexturePack *ddsPack = nullptr) {
  if (slotData.sourceKind != MaterialTextureSourceKind::ExternalFile ||
      slotData.path.empty()) {
    return Result<TextureRef, std::string>::makeResult(kInvalidTextureRef);
  }
  TextureRequest textureRequest{};
  textureRequest.path = slotData.path;
  if (ddsPack != nullptr &&
      hasExtensionCaseInsensitive(slotData.path, ".dds")) {
    const std::string canonicalPath = canonicalizeResourcePath(slotData.path);
    if (ddsPack->sourceFingerprint(canonicalPath).has_value()) {
      textureRequest.path = canonicalPath;
      textureRequest.ddsPack = ddsPack;
    }
  }
  textureRequest.loadOptions = options;
  textureRequest.kind =
      resolveTextureRequestKindForPath(textureRequest.path, kind);
  textureRequest.debugName = std::string(debugName);
  return resources.acquireTexture(textureRequest);
}
[[nodiscard]] std::vector<DdsTexturePackSource>
collectDdsTexturePackSources(const ImportedMaterialSet &materialSet) {
  std::vector<DdsTexturePackSource> sources{};
  std::unordered_set<std::string> seen{};
  for (const ImportedMaterialInfo &material : materialSet.materials) {
    for (const MaterialTextureSlotDesc &spec : kMaterialTextureSlotDescs) {
      const ImportedMaterialTexture &slot = material.textures[spec.slot];
      if (slot.sourceKind != MaterialTextureSourceKind::ExternalFile ||
          !hasExtensionCaseInsensitive(slot.path, ".dds")) {
        continue;
      }
      const std::string canonicalPath = canonicalizeResourcePath(slot.path);
      if (!canonicalPath.empty() && seen.emplace(canonicalPath).second) {
        sources.push_back(DdsTexturePackSource{.path = canonicalPath});
      }
    }
  }
  return sources;
}
[[nodiscard]] std::unique_ptr<DdsTexturePack>
tryEnsureDdsTexturePack(std::string_view scenePath,
                        const ImportedMaterialSet &materialSet,
                        std::string_view logContext) {
  std::vector<DdsTexturePackSource> sources =
      collectDdsTexturePackSources(materialSet);
  if (sources.size() < kMinAutomaticDdsTexturePackEntries) {
    return {};
  }
  auto packResult = ensureDdsTexturePack(
      std::filesystem::path(std::string(scenePath)), sources);
  if (packResult.hasError()) {
    NURI_LOG_WARNING("%.*s: DDS texture pack unavailable for '%.*s': %s",
                     static_cast<int>(logContext.size()), logContext.data(),
                     static_cast<int>(scenePath.size()), scenePath.data(),
                     packResult.error().c_str());
    return {};
  }
  return std::move(packResult.value().pack);
}
void releaseMaterialTextureRefs(ResourceManager &resources,
                                const MaterialRequest::TextureRefs &refs) {
  forEachMaterialTextureRef(refs, [&resources](TextureRef textureRef) {
    if (isValid(textureRef)) {
      resources.release(textureRef);
    }
  });
}
[[nodiscard]] std::string
makeImportedTextureDebugName(std::string_view debugNamePrefix,
                             std::string_view debugToken,
                             uint32_t sourceMaterialIndex) {
  return std::string(debugNamePrefix) + "_" + std::string(debugToken) + "_" +
         std::to_string(sourceMaterialIndex);
}
[[nodiscard]] std::string
makeImportedMaterialDebugName(std::string_view debugNamePrefix,
                              const ImportedMaterialInfo &imported,
                              uint32_t sourceMaterialIndex) {
  if (imported.name.empty()) {
    return std::string(debugNamePrefix) + "_material_" +
           std::to_string(sourceMaterialIndex);
  }
  return std::string(debugNamePrefix) + "_" + imported.name;
}
[[nodiscard]] std::string
makeImportedMaterialSourceIdentity(std::string_view canonicalModelPath,
                                   uint32_t sourceMaterialIndex) {
  return std::string(canonicalModelPath) + "#" +
         std::to_string(sourceMaterialIndex);
}
struct ImportedTextureRefsResult {
  MaterialRequest::TextureRefs refs{};
  bool complete = true;
};
[[nodiscard]] ImportedTextureRefsResult acquireImportedTextureRefs(
    ResourceManager &resources, const ImportedMaterialInfo &imported,
    std::string_view scenePath, std::string_view logContext,
    std::string_view debugNamePrefix, uint32_t sourceMaterialIndex,
    const SceneMaterialRecord *cached = nullptr,
    DdsTexturePack *ddsPack = nullptr) {
  ImportedTextureRefsResult result{};
  auto builderResult =
      SceneTextureArtifactBuilder::create(std::filesystem::path(scenePath));
  if (builderResult.hasError()) {
    result.complete = false;
    NURI_LOG_WARNING("%.*s: texture artifact builder unavailable: %s",
                     static_cast<int>(logContext.size()), logContext.data(),
                     builderResult.error().c_str());
    return result;
  }
  SceneTextureArtifactBuilder builder = std::move(builderResult.value());
  const std::string canonicalScenePath = canonicalizeResourcePath(scenePath);
  const TextureCompressionCaps caps = resources.textureCompressionCaps();
  for (const MaterialTextureSlotDesc &spec : kMaterialTextureSlotDescs) {
    const ImportedMaterialTexture &source = imported.textures[spec.slot];
    if (source.sourceKind == MaterialTextureSourceKind::None) {
      continue;
    }
    const TextureArtifactBuildOptions options =
        materialTextureArtifactBuildOptions(imported, spec.slot);
    const std::string debugName = makeImportedTextureDebugName(
        debugNamePrefix, spec.debugToken, sourceMaterialIndex);
    Result<TextureRef, std::string> acquired =
        Result<TextureRef, std::string>::makeResult(kInvalidTextureRef);
    if (source.sourceKind == MaterialTextureSourceKind::ExternalFile &&
        hasExtensionCaseInsensitive(source.path, ".dds")) {
      acquired = acquireExternalImportedTexture(
          resources, source, options.loadOptions, TextureRequestKind::Texture2D,
          debugName, ddsPack);
    } else {
      uint64_t identity =
          cached ? cached->textureCache[static_cast<size_t>(spec.slot)]
                       .artifactIdentityHash
                 : 0u;
      if (identity == 0u) {
        identity = hashSceneTextureSourceIdentity(
            canonicalScenePath, source, options.loadOptions.srgb,
            textureArtifactProcessingTag(options));
      }
      const Format targetFormat = selectTextureArtifactTargetFormat(
          caps.bc7, caps.etc2, options.loadOptions.srgb, 4u);
      auto artifact = builder.ensure(source, identity, targetFormat, options);
      acquired =
          artifact.hasError()
              ? Result<TextureRef, std::string>::makeError(artifact.error())
              : resources.acquireTexture(TextureRequest{
                    .path = artifact.value().artifactPath.string(),
                    .loadOptions = options.loadOptions,
                    .contentContract = options.contentContract,
                    .kind = TextureRequestKind::Ktx2Texture2D,
                    .debugName = debugName,
                });
    }
    if (acquired.hasError() || !isValid(acquired.value())) {
      result.complete = false;
      NURI_LOG_WARNING("%.*s: %s load failed for material %u: %s",
                       static_cast<int>(logContext.size()), logContext.data(),
                       spec.name, sourceMaterialIndex,
                       acquired.hasError() ? acquired.error().c_str()
                                           : "invalid texture reference");
      if (cached) {
        break;
      }
      continue;
    }
    result.refs[spec.slot] = acquired.value();
  }
  return result;
}
[[nodiscard]] Result<MaterialRef, std::string> acquireImportedMaterial(
    ResourceManager &resources, const ImportedMaterialInfo &imported,
    std::string_view scenePath, std::string_view logContext,
    std::string_view canonicalModelPath, std::string_view debugNamePrefix,
    uint32_t sourceMaterialIndex, const SceneMaterialRecord *cached,
    DdsTexturePack *ddsPack) {
  const ImportedTextureRefsResult textures = acquireImportedTextureRefs(
      resources, imported, scenePath, logContext, debugNamePrefix,
      sourceMaterialIndex, cached, ddsPack);
  if (cached && !textures.complete) {
    releaseMaterialTextureRefs(resources, textures.refs);
    return Result<MaterialRef, std::string>::makeError(
        "cached material texture unavailable");
  }
  const MaterialTextureHandles emptyHandles{};
  auto result = resources.acquireMaterial(MaterialRequest{
      .desc = Material::descFromImported(imported, emptyHandles),
      .textureRefs = textures.refs,
      .debugName = makeImportedMaterialDebugName(debugNamePrefix, imported,
                                                 sourceMaterialIndex),
      .sourceIdentity = makeImportedMaterialSourceIdentity(canonicalModelPath,
                                                           sourceMaterialIndex),
  });
  releaseMaterialTextureRefs(resources, textures.refs);
  return result;
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
      textures_(memory_), materials_(memory_), models_(memory_),
      materialHeaderTable_(memory_), materialClearcoatTable_(memory_),
      materialSheenTable_(memory_), materialTransmissionTable_(memory_),
      materialSpecularTable_(memory_), textureCache_(), materialCache_(),
      modelCache_() {}

ResourceManager::~ResourceManager() {
  for (uint32_t i = 0; i < models_.slots.size(); ++i) {
    if (models_.meta.isLive(i)) {
      destroyModelSlot(i);
    }
  }
  bool materialTablesDirty = false;
  for (uint32_t i = 0; i < materials_.slots.size(); ++i) {
    if (materials_.meta.isLive(i)) {
      destroyMaterialSlot(i);
      materialTablesDirty = true;
    }
  }
  if (materialTablesDirty) {
    rebuildPackedMaterialTables();
  }
  for (uint32_t i = 0; i < textures_.slots.size(); ++i) {
    if (textures_.meta.isLive(i)) {
      destroyTextureSlot(i);
    }
  }
}

Result<SlotReservation, std::string> ResourceManager::allocateMaterialSlot() {
  auto result = allocateSlot(materials_, memory_,
                             "ResourceManager::allocateMaterialSlot");
  if (!result.hasError() && result.value().appended) {
    materialHeaderTable_.push_back(MaterialHeaderGpuData{});
  }
  return result;
}

void ResourceManager::destroyTextureSlot(uint32_t index) {
  TextureSlot &slot = textures_.slots[index];
  const TextureKey key{
      .canonicalPath = std::string(slot.record.canonicalPath),
      .optionsHash = hashTextureLoadOptions(slot.record.loadOptions),
      .kind = slot.record.sourceKind,
      .contentContract = slot.record.contentContract,
  };
  textureCache_.erase(key);
  if (slot.record.contentContract ==
      TextureContentContract::NormalRgbCleanVarianceA) {
    NURI_ASSERT(normalVarianceContractTexturesLive_ > 0u,
                "Normal-variance texture live count underflow");
    const uint64_t bytes = textureStorageBytes(slot.record);
    NURI_ASSERT(normalVarianceContractTextureBytesLive_ >= bytes,
                "Normal-variance texture byte count underflow");
    --normalVarianceContractTexturesLive_;
    normalVarianceContractTextureBytesLive_ -= bytes;
  }
  slot.owner.reset();
  slot.refCount = 0;
  slot.record = TextureRecord(memory_);
  textures_.meta.release(index);
}

void ResourceManager::destroyMaterialSlot(uint32_t index) {
  MaterialSlot &slot = materials_.slots[index];
  forEachMaterialTextureRef(slot.record.textureRefs,
                            [this](TextureRef textureRef) {
                              if (isValid(textureRef)) {
                                release(textureRef);
                              }
                            });
  const MaterialKey key{
      .descHash = slot.record.descHash,
      .sourceIdentity = std::string(slot.record.sourceIdentity),
  };
  materialCache_.erase(key);
  slot.refCount = 0;
  slot.record = MaterialRecord(memory_);
  materialHeaderTable_[index] = {};
  materials_.meta.release(index);
  materialTablesDirty_ = true;
}

void ResourceManager::destroyModelSlot(uint32_t index) {
  ModelSlot &slot = models_.slots[index];
  for (const MaterialRef mappedMaterial : slot.record.sourceMaterialToRuntime) {
    if (isValid(mappedMaterial)) {
      release(mappedMaterial);
    }
  }
  const ModelKey key{
      .canonicalPath = std::string(slot.record.canonicalPath),
      .importOptionsHash = slot.record.importOptionsHash,
      .sceneMeshIndex = slot.record.sceneMeshIndex,
  };
  modelCache_.erase(key);
  slot.refCount = 0;
  slot.record = ModelRecord(memory_);
  models_.meta.release(index);
  markModelMaterialBindingsDirty();
}

void ResourceManager::rebuildPackedMaterialTables() {
  std::pmr::vector<MaterialHeaderGpuData> nextHeaders(memory_);
  std::pmr::vector<MaterialClearcoatGpuData> nextClearcoat(memory_);
  std::pmr::vector<MaterialSheenGpuData> nextSheen(memory_);
  std::pmr::vector<MaterialTransmissionGpuData> nextTransmission(memory_);
  std::pmr::vector<MaterialSpecularGpuData> nextSpecular(memory_);
  nextHeaders.resize(materials_.slots.size());
  const auto textureIndex = [this](TextureRef ref) -> uint32_t {
    const TextureRecord *record = tryGet(ref);
    return record != nullptr ? record->bindlessIndex
                             : kInvalidTextureBindlessIndex;
  };
  normalVarianceContractMaterialsLive_ = 0u;
  normalVarianceUnavailableSlotsLive_ = 0u;
  for (uint32_t index = 0; index < materials_.slots.size(); ++index) {
    if (!materials_.meta.isLive(index)) {
      nextHeaders[index] = MaterialHeaderGpuData{};
      continue;
    }
    MaterialPackedGpuData &packed =
        materials_.slots[index].record.packedGpuData;
    const MaterialRequest::TextureRefs &textureRefs =
        materials_.slots[index].record.textureRefs;
    packed.header.commonTextureIndices = glm::uvec4(
        textureIndex(textureRefs[kMaterialTextureSlotBaseColor]),
        textureIndex(textureRefs[kMaterialTextureSlotMetallicRoughness]),
        textureIndex(textureRefs[kMaterialTextureSlotNormal]),
        textureIndex(textureRefs[kMaterialTextureSlotOcclusion]));
    packed.header.emissiveTextureIndex =
        textureIndex(textureRefs[kMaterialTextureSlotEmissive]);
    packed.clearcoat.textureIndices = glm::uvec4(
        textureIndex(textureRefs[kMaterialTextureSlotClearcoat]),
        textureIndex(textureRefs[kMaterialTextureSlotClearcoatRoughness]),
        textureIndex(textureRefs[kMaterialTextureSlotClearcoatNormal]), 0u);
    packed.sheen.textureIndices = glm::uvec4(
        textureIndex(textureRefs[kMaterialTextureSlotSheenColor]),
        textureIndex(textureRefs[kMaterialTextureSlotSheenRoughness]), 0u, 0u);
    packed.transmission.textureIndices = glm::uvec4(
        textureIndex(textureRefs[kMaterialTextureSlotTransmission]),
        textureIndex(textureRefs[kMaterialTextureSlotThickness]), 0u, 0u);
    packed.specular.textureIndices = glm::uvec4(
        textureIndex(textureRefs[kMaterialTextureSlotSpecular]),
        textureIndex(textureRefs[kMaterialTextureSlotSpecularColor]), 0u, 0u);
    MaterialHeaderGpuData header = packed.header;
    header.materialFlags &= ~kMaterialFlagsNormalVarianceMask;
    const auto projectNormalVariance = [this, &header](TextureRef ref,
                                                       uint32_t flag) {
      if (!isValid(ref)) {
        return;
      }
      const TextureRecord *record = tryGet(ref);
      if (record != nullptr &&
          record->contentContract ==
              TextureContentContract::NormalRgbCleanVarianceA) {
        header.materialFlags |= flag;
      } else {
        ++normalVarianceUnavailableSlotsLive_;
      }
    };
    projectNormalVariance(textureRefs[kMaterialTextureSlotNormal],
                          kMaterialFlagsBaseNormalVarianceBit);
    projectNormalVariance(textureRefs[kMaterialTextureSlotClearcoatNormal],
                          kMaterialFlagsClearcoatNormalVarianceBit);
    if ((header.materialFlags & kMaterialFlagsNormalVarianceMask) != 0u) {
      ++normalVarianceContractMaterialsLive_;
    }
    header.clearcoatExtensionIndex = kInvalidMaterialExtensionIndex;
    header.sheenExtensionIndex = kInvalidMaterialExtensionIndex;
    header.transmissionExtensionIndex = kInvalidMaterialExtensionIndex;
    header.specularExtensionIndex = kInvalidMaterialExtensionIndex;
    if (packed.hasClearcoat) {
      header.clearcoatExtensionIndex =
          static_cast<uint32_t>(nextClearcoat.size());
      nextClearcoat.push_back(packed.clearcoat);
    }
    if (packed.hasSheen) {
      header.sheenExtensionIndex = static_cast<uint32_t>(nextSheen.size());
      nextSheen.push_back(packed.sheen);
    }
    if (packed.hasTransmissionOrVolume) {
      header.transmissionExtensionIndex =
          static_cast<uint32_t>(nextTransmission.size());
      nextTransmission.push_back(packed.transmission);
    }
    if (packed.hasSpecular) {
      header.specularExtensionIndex =
          static_cast<uint32_t>(nextSpecular.size());
      nextSpecular.push_back(packed.specular);
    }
    packed.header = header;
    nextHeaders[index] = header;
  }
  materialHeaderDirtyRange_ = calculateDirtyRange<MaterialHeaderGpuData>(
      materialHeaderTable_, nextHeaders);
  materialClearcoatDirtyRange_ = calculateDirtyRange<MaterialClearcoatGpuData>(
      materialClearcoatTable_, nextClearcoat);
  materialSheenDirtyRange_ =
      calculateDirtyRange<MaterialSheenGpuData>(materialSheenTable_, nextSheen);
  materialTransmissionDirtyRange_ =
      calculateDirtyRange<MaterialTransmissionGpuData>(
          materialTransmissionTable_, nextTransmission);
  materialSpecularDirtyRange_ = calculateDirtyRange<MaterialSpecularGpuData>(
      materialSpecularTable_, nextSpecular);
  materialHeaderTable_.swap(nextHeaders);
  materialClearcoatTable_.swap(nextClearcoat);
  materialSheenTable_.swap(nextSheen);
  materialTransmissionTable_.swap(nextTransmission);
  materialSpecularTable_.swap(nextSpecular);
}

void ResourceManager::markMaterialTablesDirty() {
  materialTablesDirty_ = true;
  flushPublicationVersions();
}

void ResourceManager::markModelMaterialBindingsDirty() {
  modelMaterialBindingsDirty_ = true;
  flushPublicationVersions();
}

void ResourceManager::flushPublicationVersions() {
  if (publicationBatchDepth_ != 0u) {
    return;
  }
  if (materialTablesDirty_) {
    materialTableDirtyBaseVersion_ = materialTableVersion_;
    rebuildPackedMaterialTables();
    ++materialTableVersion_;
    materialTablesDirty_ = false;
  }
  if (modelMaterialBindingsDirty_) {
    ++modelMaterialBindingVersion_;
    modelMaterialBindingsDirty_ = false;
  }
}

std::optional<TextureRef>
ResourceManager::tryAcquireTexture(const TextureRequest &request) {
  return tryAcquireTexture(resolveTextureRequest(request));
}

ResolvedTextureRequest
ResourceManager::resolveTextureRequest(const TextureRequest &request) const {
  ResolvedTextureRequest resolved{.request = request};
  resolved.key = TextureKey{
      .canonicalPath = canonicalizeResourcePath(request.path),
      .optionsHash = hashTextureLoadOptions(request.loadOptions),
      .kind = request.kind,
      .contentContract = request.contentContract,
  };
  return resolved;
}

std::optional<TextureRef>
ResourceManager::tryAcquireTexture(const ResolvedTextureRequest &resolved) {
  if (resolved.request.path.empty()) {
    return std::nullopt;
  }
  if (validateTextureRequestContentContract(resolved.request).hasError()) {
    return std::nullopt;
  }
  if (auto it = textureCache_.find(resolved.key); it != textureCache_.end()) {
    if (TextureSlot *cached = tryGetSlotImpl(textures_, it->second)) {
      ++cached->refCount;
      return it->second;
    }
    textureCache_.erase(it);
  }
  return std::nullopt;
}

Result<TextureRef, std::string>
ResourceManager::acquireTexture(const TextureRequest &request) {
  return acquireTexture(resolveTextureRequest(request));
}

Result<TextureRef, std::string>
ResourceManager::acquireTexture(const ResolvedTextureRequest &resolved) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  const TextureRequest &request = resolved.request;
  if (request.path.empty()) {
    return Result<TextureRef, std::string>::makeError(
        "ResourceManager::acquireTexture: path is empty");
  }
  if (auto validation = validateTextureRequestContentContract(request);
      validation.hasError()) {
    return Result<TextureRef, std::string>::makeError(validation.error());
  }
  if (const std::optional<TextureRef> cached = tryAcquireTexture(resolved);
      cached.has_value()) {
    return Result<TextureRef, std::string>::makeResult(*cached);
  }
  const std::string &canonicalPath = resolved.key.canonicalPath;
  Result<std::unique_ptr<Texture>, std::string> textureResult =
      Result<std::unique_ptr<Texture>, std::string>::makeError(
          "ResourceManager::acquireTexture: uninitialized result");
  switch (request.kind) {
  case TextureRequestKind::Texture2D:
    if (request.ddsPack != nullptr) {
      if (!hasExtensionCaseInsensitive(canonicalPath, ".dds")) {
        return Result<TextureRef, std::string>::makeError(
            "ResourceManager::acquireTexture: texture packs are only "
            "supported for DDS textures");
      }
      auto packedData = request.ddsPack->readOwned(canonicalPath);
      if (packedData.hasError()) {
        return Result<TextureRef, std::string>::makeError(packedData.error());
      }
      textureResult = Texture::loadDdsTexture(gpu_, packedData.value(),
                                              canonicalPath, request.debugName);
    } else {
      textureResult = Texture::loadTexture(
          gpu_, canonicalPath, request.loadOptions, request.debugName);
    }
    break;
  case TextureRequestKind::Ktx2Texture2D:
    textureResult =
        Texture::loadTextureKtx2(gpu_, canonicalPath, request.debugName);
    if (textureResult.hasError()) {
      TextureArtifactBuildOptions buildOptions{
          .loadOptions = request.loadOptions,
          .encoding = TextureArtifactEncoding::Uastc,
          .contentContract = request.contentContract,
      };
      const uint64_t identity =
          hashTextureSourceIdentity(canonicalPath, request.loadOptions.srgb,
                                    textureArtifactProcessingTag(buildOptions));
      const TextureCompressionCaps caps = gpu_.getTextureCompressionCaps();
      const Format targetFormat = selectTextureArtifactTargetFormat(
          caps.bc7, caps.etc2, request.loadOptions.srgb, 4u);
      auto artifact =
          ensureTextureArtifactFromFile(std::filesystem::path(canonicalPath),
                                        identity, targetFormat, buildOptions);
      if (!artifact.hasError()) {
        textureResult = Texture::loadTextureKtx2(
            gpu_, artifact.value().artifactPath.string(), request.debugName);
      }
    }
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
  return storeAcquiredTexture(resolved.key, canonicalPath, request,
                              std::move(textureResult.value()));
}

Result<TextureRef, std::string> ResourceManager::storeAcquiredTexture(
    const TextureKey &key, std::string_view canonicalPath,
    const TextureRequest &request, std::unique_ptr<Texture> texture) {
  if (!texture || !texture->valid()) {
    return Result<TextureRef, std::string>::makeError(
        "ResourceManager::storeAcquiredTexture: loaded texture is invalid");
  }
  if (request.contentContract ==
          TextureContentContract::NormalRgbCleanVarianceA &&
      !formatPreservesNormalVarianceAlpha(texture->format())) {
    return Result<TextureRef, std::string>::makeError(
        "NormalRgbCleanVarianceA texture lost its alpha channel");
  }
  auto slotResult =
      allocateSlot(textures_, memory_, "ResourceManager::acquireTexture");
  if (slotResult.hasError()) {
    return Result<TextureRef, std::string>::makeError(slotResult.error());
  }
  const uint32_t slotIndex = slotResult.value().index;
  TextureSlot &slot = textures_.slots[slotIndex];
  const TextureRef ref = makeRefForSlot<TextureRef>(textures_, slotIndex);
  slot.refCount = 1;
  slot.record = TextureRecord(memory_);
  slot.record.ref = ref;
  const TextureHandle textureHandle = texture->handle();
  slot.record.texture = textureHandle;
  slot.record.bindlessIndex = gpu_.getTextureBindlessIndex(textureHandle);
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
  slot.record.contentContract = request.contentContract;
  if (request.contentContract ==
      TextureContentContract::NormalRgbCleanVarianceA) {
    ++normalVarianceContractTexturesLive_;
    normalVarianceContractTextureBytesLive_ += textureStorageBytes(slot.record);
  }
  slot.record.canonicalPath = canonicalPath;
  slot.record.debugName = request.debugName;
  slot.owner.reset(gpu_, texture->release());
  texture.reset();
  textureCache_.emplace(std::move(key), ref);
  return Result<TextureRef, std::string>::makeResult(ref);
}

Result<TextureRef, std::string>
ResourceManager::adoptPreparedTexture(const TextureRequest &request,
                                      std::unique_ptr<Texture> texture) {
  return adoptPreparedTexture(resolveTextureRequest(request),
                              std::move(texture));
}

Result<TextureRef, std::string>
ResourceManager::adoptPreparedTexture(const ResolvedTextureRequest &resolved,
                                      std::unique_ptr<Texture> texture) {
  const TextureRequest &request = resolved.request;
  if (request.path.empty()) {
    return Result<TextureRef, std::string>::makeError(
        "ResourceManager::adoptPreparedTexture: path is empty");
  }
  if (auto validation = validateTextureRequestContentContract(request);
      validation.hasError()) {
    return Result<TextureRef, std::string>::makeError(validation.error());
  }
  if (const std::optional<TextureRef> cached = tryAcquireTexture(resolved);
      cached.has_value()) {
    return Result<TextureRef, std::string>::makeResult(*cached);
  }
  return storeAcquiredTexture(resolved.key, resolved.key.canonicalPath, request,
                              std::move(texture));
}

ModelRef ResourceManager::tryAcquireCachedModel(const ModelKey &key) {
  if (auto it = modelCache_.find(key); it != modelCache_.end()) {
    if (ModelSlot *cached = tryGetSlotImpl(models_, it->second)) {
      ++cached->refCount;
      return it->second;
    }
    modelCache_.erase(it);
  }
  return kInvalidModelRef;
}

std::optional<ModelRef>
ResourceManager::tryAcquireModel(const ModelRequest &request) {
  return tryAcquireModel(resolveModelRequest(request));
}

ResolvedModelRequest
ResourceManager::resolveModelRequest(const ModelRequest &request) const {
  ResolvedModelRequest resolved{.request = request};
  resolved.key = ModelKey{
      .canonicalPath = canonicalizeResourcePath(request.path),
      .importOptionsHash = hashModelImportOptions(request.importOptions),
      .sceneMeshIndex = request.sceneMeshIndex,
  };
  return resolved;
}

std::optional<ModelRef>
ResourceManager::tryAcquireModel(const ResolvedModelRequest &resolved) {
  if (resolved.request.path.empty()) {
    return std::nullopt;
  }
  const ModelRef ref = tryAcquireCachedModel(resolved.key);
  return isValid(ref) ? std::optional<ModelRef>(ref) : std::nullopt;
}

Result<ModelRef, std::string> ResourceManager::storeAcquiredModel(
    const ModelKey &key, std::string_view canonicalPath, uint64_t optionsHash,
    const ModelRequest &request, std::unique_ptr<Model> model) {
  if (!model) {
    return Result<ModelRef, std::string>::makeError(
        "ResourceManager::storeAcquiredModel: model creation returned null");
  }
  auto slotResult =
      allocateSlot(models_, memory_, "ResourceManager::acquireModel");
  if (slotResult.hasError()) {
    return Result<ModelRef, std::string>::makeError(slotResult.error());
  }
  const uint32_t slotIndex = slotResult.value().index;
  ModelSlot &slot = models_.slots[slotIndex];
  const ModelRef ref = makeRefForSlot<ModelRef>(models_, slotIndex);
  slot.refCount = 1;
  slot.record = ModelRecord(memory_);
  slot.record.ref = ref;
  slot.record.model = std::move(model);
  slot.record.canonicalPath.assign(canonicalPath.data(), canonicalPath.size());
  slot.record.importOptionsHash = optionsHash;
  slot.record.sceneMeshIndex = request.sceneMeshIndex;
  slot.record.sourceMaterialToRuntime.assign(
      slot.record.model->sourceMaterialCount(), kInvalidMaterialRef);
  modelCache_.emplace(key, ref);
  markModelMaterialBindingsDirty();
  return Result<ModelRef, std::string>::makeResult(ref);
}

Result<ModelRef, std::string>
ResourceManager::acquireModel(const ModelRequest &request) {
  return acquireModel(resolveModelRequest(request));
}

Result<ModelRef, std::string>
ResourceManager::acquireModel(const ResolvedModelRequest &resolved) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  const ModelRequest &request = resolved.request;
  if (request.path.empty()) {
    return Result<ModelRef, std::string>::makeError(
        "ResourceManager::acquireModel: path is empty");
  }
  if (const std::optional<ModelRef> cached = tryAcquireModel(resolved);
      cached.has_value()) {
    return Result<ModelRef, std::string>::makeResult(*cached);
  }
  const std::string &canonicalPath = resolved.key.canonicalPath;
  Result<std::unique_ptr<Model>, std::string> modelResult =
      Result<std::unique_ptr<Model>, std::string>::makeError(
          "ResourceManager::acquireModel: uninitialized result");
  if (request.sceneMeshIndex == std::numeric_limits<uint32_t>::max()) {
    modelResult = Model::createFromFile(
        gpu_, canonicalPath, request.importOptions, memory_, request.debugName);
  } else {
    auto meshDataResult = MeshImporter::loadSceneMeshFromFile(
        canonicalPath, request.sceneMeshIndex, request.importOptions, memory_);
    if (meshDataResult.hasError()) {
      return Result<ModelRef, std::string>::makeError(meshDataResult.error());
    }
    modelResult =
        Model::create(gpu_, meshDataResult.value(), request.debugName);
  }
  if (modelResult.hasError()) {
    return Result<ModelRef, std::string>::makeError(modelResult.error());
  }
  return storeAcquiredModel(resolved.key, canonicalPath,
                            resolved.key.importOptionsHash, request,
                            std::move(modelResult.value()));
}

Result<ModelRef, std::string>
ResourceManager::adoptPreparedModel(const ModelRequest &request,
                                    std::unique_ptr<Model> model) {
  return adoptPreparedModel(resolveModelRequest(request), std::move(model));
}

Result<ModelRef, std::string>
ResourceManager::adoptPreparedModel(const ResolvedModelRequest &resolved,
                                    std::unique_ptr<Model> model) {
  const ModelRequest &request = resolved.request;
  if (request.path.empty()) {
    return Result<ModelRef, std::string>::makeError(
        "ResourceManager::adoptPreparedModel: path is empty");
  }
  if (const std::optional<ModelRef> cached = tryAcquireModel(resolved);
      cached.has_value()) {
    return Result<ModelRef, std::string>::makeResult(*cached);
  }
  return storeAcquiredModel(resolved.key, resolved.key.canonicalPath,
                            resolved.key.importOptionsHash, request,
                            std::move(model));
}

Result<ModelRef, std::string>
ResourceManager::acquireGeneratedModel(const MeshData &meshData,
                                       std::string_view debugName) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  auto modelResult = Model::create(gpu_, meshData, debugName);
  if (modelResult.hasError()) {
    return Result<ModelRef, std::string>::makeError(modelResult.error());
  }
  auto model = std::move(modelResult.value());
  if (!model) {
    return Result<ModelRef, std::string>::makeError(
        "ResourceManager::acquireGeneratedModel: model creation returned "
        "null");
  }
  auto slotResult =
      allocateSlot(models_, memory_, "ResourceManager::adoptPreparedModel");
  if (slotResult.hasError()) {
    return Result<ModelRef, std::string>::makeError(slotResult.error());
  }
  const uint32_t slotIndex = slotResult.value().index;
  ModelSlot &slot = models_.slots[slotIndex];
  const ModelRef ref = makeRefForSlot<ModelRef>(models_, slotIndex);
  slot.refCount = 1;
  slot.record = ModelRecord(memory_);
  slot.record.ref = ref;
  slot.record.model = std::move(model);
  slot.record.importOptionsHash = 0u;
  slot.record.sceneMeshIndex = std::numeric_limits<uint32_t>::max();
  slot.record.sourceMaterialToRuntime.assign(
      slot.record.model->sourceMaterialCount(), kInvalidMaterialRef);
  return Result<ModelRef, std::string>::makeResult(ref);
}

Result<MaterialRef, std::string>
ResourceManager::acquireMaterial(const MaterialRequest &request) {
  auto resolved = resolveMaterialRequest(request);
  if (resolved.hasError())
    return Result<MaterialRef, std::string>::makeError(resolved.error());
  return acquireMaterial(resolved.value());
}

Result<ResolvedMaterialRequest, std::string>
ResourceManager::resolveMaterialRequest(const MaterialRequest &request) const {
  ResolvedMaterialRequest resolved{.request = request};
  MaterialDesc &resolvedDesc = resolved.request.desc;
  for (const MaterialTextureSlotDesc &spec : kMaterialTextureSlotDescs) {
    const TextureRef ref = request.textureRefs[spec.slot];
    if (!isValid(ref))
      continue;
    const TextureRecord *record = tryGet(ref);
    if (!record) {
      return Result<ResolvedMaterialRequest, std::string>::makeError(
          "ResourceManager::resolveMaterialRequest: stale texture ref for slot "
          "'" +
          std::string(spec.name) + "'");
    }
    resolvedDesc.textures[spec.slot] = record->texture;
  }
  Material::finalizeDesc(resolvedDesc);
  resolved.key = MaterialKey{.descHash = hashMaterialDesc(resolvedDesc),
                             .sourceIdentity = request.sourceIdentity};
  return Result<ResolvedMaterialRequest, std::string>::makeResult(
      std::move(resolved));
}

Result<MaterialRef, std::string>
ResourceManager::acquireMaterial(const ResolvedMaterialRequest &resolved) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  const MaterialRequest &request = resolved.request;
  const MaterialKey &key = resolved.key;
  if (auto it = materialCache_.find(key); it != materialCache_.end()) {
    if (MaterialSlot *cached = tryGetSlotImpl(materials_, it->second)) {
      ++cached->refCount;
      return Result<MaterialRef, std::string>::makeResult(it->second);
    }
    materialCache_.erase(it);
  }
  auto materialResult = Material::create(gpu_, request.desc, request.debugName);
  if (materialResult.hasError()) {
    return Result<MaterialRef, std::string>::makeError(materialResult.error());
  }
  const Material &material = *materialResult.value();
  auto slotResult = allocateMaterialSlot();
  if (slotResult.hasError()) {
    return Result<MaterialRef, std::string>::makeError(slotResult.error());
  }
  const uint32_t slotIndex = slotResult.value().index;
  MaterialSlot &slot = materials_.slots[slotIndex];
  const MaterialRef ref = makeRefForSlot<MaterialRef>(materials_, slotIndex);
  slot.refCount = 1;
  slot.record = MaterialRecord(memory_);
  slot.record.ref = ref;
  slot.record.desc = material.desc();
  slot.record.textureRefs = request.textureRefs;
  slot.record.packedGpuData = material.packedGpuData();
  slot.record.descHash = key.descHash;
  slot.record.debugName = request.debugName;
  slot.record.sourceIdentity = request.sourceIdentity;
  forEachMaterialTextureRef(slot.record.textureRefs,
                            [this](TextureRef textureRef) {
                              if (isValid(textureRef)) {
                                retain(textureRef);
                              }
                            });
  if (slotIndex >= materialHeaderTable_.size()) {
    materialHeaderTable_.resize(slotIndex + 1u);
  }
  markMaterialTablesDirty();
  materialCache_.emplace(key, ref);
  return Result<MaterialRef, std::string>::makeResult(ref);
}

Result<ScenePrefabAssets, std::string>
ResourceManager::acquireScenePrefabAssets(const ScenePrefab &prefab) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (prefab.sourcePath.empty()) {
    return Result<ScenePrefabAssets, std::string>::makeError(
        "ResourceManager::acquireScenePrefabAssets: prefab source path is "
        "empty");
  }
  ScenePrefabAssets assets(memory_);
  assets.models.resize(prefab.meshAssets.size(), kInvalidModelRef);
  assets.materials.resize(prefab.materialAssets.size(), kInvalidMaterialRef);
  const auto cleanupAndError =
      [&](std::string err) -> Result<ScenePrefabAssets, std::string> {
    for (const MaterialRef materialRef : assets.materials) {
      if (isValid(materialRef)) {
        release(materialRef);
      }
    }
    for (const ModelRef modelRef : assets.models) {
      if (isValid(modelRef)) {
        release(modelRef);
      }
    }
    return Result<ScenePrefabAssets, std::string>::makeError(std::move(err));
  };
  const std::string canonicalModelPath =
      canonicalizeResourcePath(prefab.sourcePath);
  HashMap<uint32_t, uint32_t> sourceMaterialToAssetIndex{};
  for (uint32_t assetIndex = 0u; assetIndex < prefab.materialAssets.size();
       ++assetIndex) {
    sourceMaterialToAssetIndex.emplace(
        prefab.materialAssets[assetIndex].sourceIndex, assetIndex);
  }
  std::optional<ImportedMaterialSet> importedMaterialSet{};
  const auto ensureImportedMaterialSet =
      [&]() -> Result<const ImportedMaterialSet *, std::string> {
    if (!importedMaterialSet.has_value()) {
      auto materialInfoResult =
          MeshImporter::loadMaterialInfoFromFile(prefab.sourcePath);
      if (materialInfoResult.hasError()) {
        return Result<const ImportedMaterialSet *, std::string>::makeError(
            materialInfoResult.error());
      }
      importedMaterialSet = std::move(materialInfoResult.value());
    }
    return Result<const ImportedMaterialSet *, std::string>::makeResult(
        &importedMaterialSet.value());
  };
  const ImportedMaterialSet *importedMaterialSetPtr = nullptr;
  if (auto importedResult = ensureImportedMaterialSet();
      !importedResult.hasError()) {
    importedMaterialSetPtr = importedResult.value();
  }
  std::unique_ptr<DdsTexturePack> ddsPack =
      importedMaterialSetPtr != nullptr
          ? tryEnsureDdsTexturePack(prefab.sourcePath, *importedMaterialSetPtr,
                                    "ResourceManager::acquireScenePrefabAssets")
          : nullptr;
  bool loadedFromCache = false;
  if (auto cachedMaterials = tryLoadSceneMaterialCache(prefab.sourcePath);
      cachedMaterials.has_value()) {
    std::vector<MaterialRef> pendingMaterials(assets.materials.size(),
                                              kInvalidMaterialRef);
    bool cacheUsable = true;
    for (const SceneMaterialRecord &cached : cachedMaterials->materials) {
      auto assetIt =
          sourceMaterialToAssetIndex.find(cached.sourceMaterialIndex);
      if (assetIt == sourceMaterialToAssetIndex.end()) {
        continue;
      }
      const ImportedMaterialInfo *imported = &cached.sourceMaterial;
      if (importedMaterialSetPtr != nullptr &&
          cached.sourceMaterialIndex <
              importedMaterialSetPtr->materials.size()) {
        imported =
            &importedMaterialSetPtr->materials[cached.sourceMaterialIndex];
      }
      auto acquireMaterialResult = acquireImportedMaterial(
          *this, *imported, prefab.sourcePath,
          "ResourceManager::acquireScenePrefabAssets", canonicalModelPath,
          "scene_prefab", cached.sourceMaterialIndex, &cached, ddsPack.get());
      if (acquireMaterialResult.hasError()) {
        cacheUsable = false;
        NURI_LOG_WARNING(
            "ResourceManager::acquireScenePrefabAssets: failed to create "
            "cached material %u: %s",
            cached.sourceMaterialIndex, acquireMaterialResult.error().c_str());
        break;
      }
      pendingMaterials[assetIt->second] = acquireMaterialResult.value();
    }
    if (!cacheUsable) {
      for (const MaterialRef materialRef : pendingMaterials) {
        if (isValid(materialRef)) {
          release(materialRef);
        }
      }
    } else {
      for (size_t materialIndex = 0; materialIndex < pendingMaterials.size();
           ++materialIndex) {
        if (!isValid(pendingMaterials[materialIndex])) {
          continue;
        }
        assets.materials[materialIndex] = pendingMaterials[materialIndex];
      }
      loadedFromCache = std::ranges::all_of(
          assets.materials, [](MaterialRef ref) { return isValid(ref); });
    }
  }
  if (!loadedFromCache) {
    auto materialInfoResult = ensureImportedMaterialSet();
    if (materialInfoResult.hasError()) {
      return cleanupAndError(
          "ResourceManager::acquireScenePrefabAssets: failed to parse material "
          "metadata: " +
          materialInfoResult.error());
    }
    const ImportedMaterialSet &materialSet = *materialInfoResult.value();
    for (uint32_t assetIndex = 0u; assetIndex < prefab.materialAssets.size();
         ++assetIndex) {
      if (isValid(assets.materials[assetIndex])) {
        continue;
      }
      const uint32_t sourceMaterialIndex =
          prefab.materialAssets[assetIndex].sourceIndex;
      if (sourceMaterialIndex >= materialSet.materials.size()) {
        continue;
      }
      const ImportedMaterialInfo &imported =
          materialSet.materials[sourceMaterialIndex];
      auto acquireMaterialResult = acquireImportedMaterial(
          *this, imported, prefab.sourcePath,
          "ResourceManager::acquireScenePrefabAssets", canonicalModelPath,
          "scene_prefab", sourceMaterialIndex, nullptr, ddsPack.get());
      if (acquireMaterialResult.hasError()) {
        return cleanupAndError(
            "ResourceManager::acquireScenePrefabAssets: failed to create "
            "material " +
            std::to_string(sourceMaterialIndex) + ": " +
            acquireMaterialResult.error());
      }
      assets.materials[assetIndex] = acquireMaterialResult.value();
    }
  }
  MaterialRef fallbackMaterial = kInvalidMaterialRef;
  for (const MaterialRef material : assets.materials) {
    if (isValid(material)) {
      fallbackMaterial = material;
      break;
    }
  }
  if (!isValid(fallbackMaterial) && !prefab.renderables.empty()) {
    auto fallbackMaterialResult = acquireMaterial(MaterialRequest{
        .desc = MaterialDesc{},
        .textureRefs = {},
        .debugName = "scene_prefab_default_material",
        .sourceIdentity = canonicalModelPath + "#default",
    });
    if (fallbackMaterialResult.hasError()) {
      return cleanupAndError(fallbackMaterialResult.error());
    }
    fallbackMaterial = fallbackMaterialResult.value();
    if (assets.materials.empty()) {
      assets.materials.resize(1u, kInvalidMaterialRef);
    }
    assets.materials[0] = fallbackMaterial;
  }
  const uint64_t optionsHash = hashModelImportOptions(prefab.importOptions);
  std::pmr::vector<uint32_t> pendingMeshIndices(memory_);
  pendingMeshIndices.reserve(prefab.meshAssets.size());
  const auto configurePrefabModelMaterials =
      [this, &assets, &prefab, fallbackMaterial](ModelRef modelRef) {
        if (isValid(fallbackMaterial)) {
          setModelMaterialForAllSources(modelRef, fallbackMaterial);
        }
        for (uint32_t assetIndex = 0u; assetIndex < assets.materials.size();
             ++assetIndex) {
          if (isValid(assets.materials[assetIndex]) &&
              assetIndex < prefab.materialAssets.size()) {
            (void)setModelMaterialForSource(
                modelRef, prefab.materialAssets[assetIndex].sourceIndex,
                assets.materials[assetIndex]);
          }
        }
      };
  for (uint32_t meshAssetIndex = 0u; meshAssetIndex < prefab.meshAssets.size();
       ++meshAssetIndex) {
    const uint32_t sourceSceneMeshIndex =
        prefab.meshAssets[meshAssetIndex].sourceIndex;
    const ModelKey key = makeSceneMeshModelKey(canonicalModelPath, optionsHash,
                                               sourceSceneMeshIndex);
    if (const ModelRef cachedRef = tryAcquireCachedModel(key);
        isValid(cachedRef)) {
      assets.models[meshAssetIndex] = cachedRef;
      continue;
    }
    pendingMeshIndices.push_back(meshAssetIndex);
  }
  if (!pendingMeshIndices.empty()) {
    std::pmr::vector<uint32_t> sourceSceneMeshIndices(memory_);
    sourceSceneMeshIndices.reserve(pendingMeshIndices.size());
    for (const uint32_t meshAssetIndex : pendingMeshIndices) {
      sourceSceneMeshIndices.push_back(
          prefab.meshAssets[meshAssetIndex].sourceIndex);
    }
    auto sceneMeshesResult = MeshImporter::loadSceneMeshesFromFile(
        prefab.sourcePath,
        std::span<const uint32_t>(sourceSceneMeshIndices.data(),
                                  sourceSceneMeshIndices.size()),
        prefab.importOptions, memory_);
    if (sceneMeshesResult.hasError()) {
      return cleanupAndError(
          "ResourceManager::acquireScenePrefabAssets: failed to batch load "
          "scene meshes: " +
          sceneMeshesResult.error());
    }
    std::pmr::vector<MeshData> sceneMeshes =
        std::move(sceneMeshesResult.value());
    for (size_t meshOrdinal = 0; meshOrdinal < pendingMeshIndices.size();
         ++meshOrdinal) {
      const uint32_t meshIndex = pendingMeshIndices[meshOrdinal];
      const uint32_t sourceSceneMeshIndex =
          prefab.meshAssets[meshIndex].sourceIndex;
      ModelRequest request{
          .path = std::string(prefab.sourcePath),
          .importOptions = prefab.importOptions,
          .debugName = makeScenePrefabMeshDebugName(sourceSceneMeshIndex),
          .sceneMeshIndex = sourceSceneMeshIndex,
      };
      auto modelResult =
          Model::create(gpu_, sceneMeshes[meshOrdinal], request.debugName);
      if (modelResult.hasError()) {
        return cleanupAndError(
            "ResourceManager::acquireScenePrefabAssets: failed to create "
            "scene mesh " +
            std::to_string(meshIndex) + ": " + modelResult.error());
      }
      const ModelKey key = makeSceneMeshModelKey(
          canonicalModelPath, optionsHash, sourceSceneMeshIndex);
      auto storeModelResult =
          storeAcquiredModel(key, canonicalModelPath, optionsHash, request,
                             std::move(modelResult.value()));
      if (storeModelResult.hasError()) {
        return cleanupAndError(
            "ResourceManager::acquireScenePrefabAssets: failed to register "
            "scene mesh " +
            std::to_string(meshIndex) + ": " + storeModelResult.error());
      }
      assets.models[meshIndex] = storeModelResult.value();
    }
  }
  for (const ModelRef modelRef : assets.models) {
    configurePrefabModelMaterials(modelRef);
  }
  return Result<ScenePrefabAssets, std::string>::makeResult(std::move(assets));
}

void ResourceManager::retain(TextureRef ref) { retainRef(textures_, ref); }

void ResourceManager::release(TextureRef ref) { releaseRef(textures_, ref); }

void ResourceManager::retain(ModelRef ref) { retainRef(models_, ref); }

void ResourceManager::release(ModelRef ref) { releaseRef(models_, ref); }

void ResourceManager::retain(MaterialRef ref) { retainRef(materials_, ref); }

void ResourceManager::release(MaterialRef ref) { releaseRef(materials_, ref); }

bool ResourceManager::owns(TextureRef ref) const noexcept {
  return tryGetSlotImpl(textures_, ref) != nullptr;
}

bool ResourceManager::owns(ModelRef ref) const noexcept {
  return tryGetSlotImpl(models_, ref) != nullptr;
}

bool ResourceManager::owns(MaterialRef ref) const noexcept {
  return tryGetSlotImpl(materials_, ref) != nullptr;
}

const TextureRecord *ResourceManager::tryGet(TextureRef ref) const {
  const auto *slot = tryGetSlotImpl(textures_, ref);
  return slot ? &slot->record : nullptr;
}

const ModelRecord *ResourceManager::tryGet(ModelRef ref) const {
  const auto *slot = tryGetSlotImpl(models_, ref);
  return slot ? &slot->record : nullptr;
}

const MaterialRecord *ResourceManager::tryGet(MaterialRef ref) const {
  const auto *slot = tryGetSlotImpl(materials_, ref);
  return slot ? &slot->record : nullptr;
}

void ResourceManager::beginPublicationBatch() { ++publicationBatchDepth_; }

void ResourceManager::endPublicationBatch() {
  --publicationBatchDepth_;
  flushPublicationVersions();
}

void ResourceManager::collectGarbage() {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  collectRetired(models_, [this](uint32_t index) { destroyModelSlot(index); });
  const uint32_t materialsDestroyed = collectRetired(
      materials_, [this](uint32_t index) { destroyMaterialSlot(index); });
  if (materialsDestroyed != 0u) {
    flushPublicationVersions();
  }
  collectRetired(textures_,
                 [this](uint32_t index) { destroyTextureSlot(index); });
}
PoolStats ResourceManager::stats() const {
  PoolStats s{};
  std::tie(s.liveTextures, s.retiredTextures) = resourceCounts(textures_);
  std::tie(s.liveMaterials, s.retiredMaterials) = resourceCounts(materials_);
  std::tie(s.liveModels, s.retiredModels) = resourceCounts(models_);
  s.normalVarianceContractTexturesLive = normalVarianceContractTexturesLive_;
  s.normalVarianceContractMaterialsLive = normalVarianceContractMaterialsLive_;
  s.normalVarianceUnavailableSlotsLive = normalVarianceUnavailableSlotsLive_;
  s.normalVarianceContractTextureBytesLive =
      normalVarianceContractTextureBytesLive_;
  return s;
}

uint32_t ResourceManager::materialTableIndex(MaterialRef ref) const {
  return tryGetSlotImpl(materials_, ref) ? indexOf(ref) : 0u;
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

MaterialRef ResourceManager::resolveRenderableMaterial(
    ModelRef model, uint32_t submeshIndex, MaterialRef fallback,
    MaterialRef overrideMaterial) const noexcept {
  if (isValid(overrideMaterial)) {
    return overrideMaterial;
  }
  const ModelRecord *record = tryGet(model);
  if (record == nullptr) {
    return fallback;
  }
  MaterialRef material = record->materialForSubmesh(submeshIndex);
  if (!isValid(material)) {
    material = record->materialForSource(submeshIndex);
  }
  return isValid(material) ? material : fallback;
}

bool ResourceManager::setModelMaterialForSource(ModelRef model,
                                                uint32_t sourceMaterialIndex,
                                                MaterialRef material) {
  ModelSlot *slot = tryGetSlotImpl(models_, model);
  if (slot == nullptr) {
    return false;
  }
  if (sourceMaterialIndex >= slot->record.sourceMaterialToRuntime.size()) {
    return false;
  }
  if (isValid(material) && !tryGetSlotImpl(materials_, material)) {
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
  markModelMaterialBindingsDirty();
  return true;
}

void ResourceManager::setModelMaterialForAllSources(ModelRef model,
                                                    MaterialRef material) {
  ModelSlot *slot = tryGetSlotImpl(models_, model);
  if (slot == nullptr ||
      (isValid(material) && !tryGetSlotImpl(materials_, material))) {
    return;
  }
  bool changed = false;
  for (MaterialRef &mapped : slot->record.sourceMaterialToRuntime) {
    if (mapped == material)
      continue;
    if (isValid(material))
      retain(material);
    if (isValid(mapped))
      release(mapped);
    mapped = material;
    changed = true;
  }
  if (changed)
    markModelMaterialBindingsDirty();
}

} // namespace nuri
