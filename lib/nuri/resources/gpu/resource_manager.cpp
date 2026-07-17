#include "nuri/pch.h"

#include "nuri/resources/gpu/resource_manager.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/resources/mesh_importer.h"
#include "nuri/resources/storage/material/material_binary_serializer.h"
#include "nuri/resources/storage/material/material_cache_utils.h"
#include "nuri/resources/storage/mesh/mesh_cache_utils.h"
#include "nuri/resources/storage/texture/dds_texture_pack.h"
#include "nuri/resources/storage/texture/material_texture_artifact.h"
#include "nuri/resources/storage/texture/texture_artifact_builder.h"
#include "nuri/resources/storage/texture/texture_cache_utils.h"
#include "nuri/utils/env_utils.h"

#include <unordered_set>

namespace nuri {

namespace {

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

[[nodiscard]] bool hasTextureExtension(std::string_view path,
                                       std::string_view extension) {
  if (path.size() < extension.size()) {
    return false;
  }
  const size_t start = path.size() - extension.size();
  for (size_t i = 0; i < extension.size(); ++i) {
    const char lhs = static_cast<char>(
        std::tolower(static_cast<unsigned char>(path[start + i])));
    const char rhs = static_cast<char>(
        std::tolower(static_cast<unsigned char>(extension[i])));
    if (lhs != rhs) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] TextureRequestKind
resolveTextureRequestKindForPath(std::string_view path,
                                 TextureRequestKind fallback) {
  if (fallback == TextureRequestKind::Texture2D &&
      (hasTextureExtension(path, ".ktx2") ||
       hasTextureExtension(path, ".ktx"))) {
    return TextureRequestKind::Ktx2Texture2D;
  }
  return fallback;
}

struct MaterialTextureResolveSpec {
  const char *slotName = nullptr;
  TextureRef MaterialRequest::TextureRefs::*ref = nullptr;
  TextureHandle MaterialTextureHandles::*handle = nullptr;
};

struct ImportedTextureAcquireSpec {
  const char *logName = nullptr;
  const char *debugSuffix = nullptr;
  size_t artifactSpecIndex = 0u;
  MaterialTextureSlotData MaterialData::*slot = nullptr;
  TextureRef MaterialRequest::TextureRefs::*outRef = nullptr;
};

constexpr std::array<MaterialTextureResolveSpec, kMaterialTextureSlotCount>
    kMaterialTextureResolveSpecs{{
        {"baseColor", &MaterialRequest::TextureRefs::baseColor,
         &MaterialTextureHandles::baseColor},
        {"metallicRoughness", &MaterialRequest::TextureRefs::metallicRoughness,
         &MaterialTextureHandles::metallicRoughness},
        {"normal", &MaterialRequest::TextureRefs::normal,
         &MaterialTextureHandles::normal},
        {"occlusion", &MaterialRequest::TextureRefs::occlusion,
         &MaterialTextureHandles::occlusion},
        {"emissive", &MaterialRequest::TextureRefs::emissive,
         &MaterialTextureHandles::emissive},
        {"clearcoat", &MaterialRequest::TextureRefs::clearcoat,
         &MaterialTextureHandles::clearcoat},
        {"clearcoatRoughness",
         &MaterialRequest::TextureRefs::clearcoatRoughness,
         &MaterialTextureHandles::clearcoatRoughness},
        {"clearcoatNormal", &MaterialRequest::TextureRefs::clearcoatNormal,
         &MaterialTextureHandles::clearcoatNormal},
        {"specular", &MaterialRequest::TextureRefs::specular,
         &MaterialTextureHandles::specular},
        {"specularColor", &MaterialRequest::TextureRefs::specularColor,
         &MaterialTextureHandles::specularColor},
        {"sheenColor", &MaterialRequest::TextureRefs::sheenColor,
         &MaterialTextureHandles::sheenColor},
        {"sheenRoughness", &MaterialRequest::TextureRefs::sheenRoughness,
         &MaterialTextureHandles::sheenRoughness},
        {"transmission", &MaterialRequest::TextureRefs::transmission,
         &MaterialTextureHandles::transmission},
        {"thickness", &MaterialRequest::TextureRefs::thickness,
         &MaterialTextureHandles::thickness},
    }};

constexpr std::array<ImportedTextureAcquireSpec, kMaterialTextureSlotCount>
    kImportedTextureAcquireSpecs{{
        {"baseColor", "_base_color_", 0u, &MaterialData::baseColor,
         &MaterialRequest::TextureRefs::baseColor},
        {"metal/rough", "_metal_rough_", 1u, &MaterialData::metallicRoughness,
         &MaterialRequest::TextureRefs::metallicRoughness},
        {"normal", "_normal_", 2u, &MaterialData::normal,
         &MaterialRequest::TextureRefs::normal},
        {"occlusion", "_occlusion_", 3u, &MaterialData::occlusion,
         &MaterialRequest::TextureRefs::occlusion},
        {"emissive", "_emissive_", 4u, &MaterialData::emissive,
         &MaterialRequest::TextureRefs::emissive},
        {"clearcoat", "_clearcoat_", 5u, &MaterialData::clearcoat,
         &MaterialRequest::TextureRefs::clearcoat},
        {"clearcoat roughness", "_clearcoat_roughness_", 6u,
         &MaterialData::clearcoatRoughness,
         &MaterialRequest::TextureRefs::clearcoatRoughness},
        {"clearcoat normal", "_clearcoat_normal_", 7u,
         &MaterialData::clearcoatNormal,
         &MaterialRequest::TextureRefs::clearcoatNormal},
        {"specular", "_specular_", 8u, &MaterialData::specular,
         &MaterialRequest::TextureRefs::specular},
        {"specular color", "_specular_color_", 9u, &MaterialData::specularColor,
         &MaterialRequest::TextureRefs::specularColor},
        {"sheen color", "_sheen_color_", 10u, &MaterialData::sheenColor,
         &MaterialRequest::TextureRefs::sheenColor},
        {"sheen roughness", "_sheen_roughness_", 11u,
         &MaterialData::sheenRoughness,
         &MaterialRequest::TextureRefs::sheenRoughness},
        {"transmission", "_transmission_", 12u, &MaterialData::transmission,
         &MaterialRequest::TextureRefs::transmission},
        {"thickness", "_thickness_", 13u, &MaterialData::thickness,
         &MaterialRequest::TextureRefs::thickness},
    }};

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

template <typename SlotT, typename SlotMetaT, typename RefT>
[[nodiscard]] bool isSlotLiveForRef(const std::pmr::vector<SlotT> &slots,
                                    const SlotMetaT &slotMeta, RefT ref) {
  if (!isValid(ref)) {
    return false;
  }
  const ResourceHandleParts parts = unpackResourceHandle(ref.value);
  return parts.index < slots.size() &&
         slotMeta.isValid(parts.index, parts.generation);
}

template <typename SlotT, typename SlotMetaT, typename RefT>
[[nodiscard]] SlotT *tryGetSlotImpl(std::pmr::vector<SlotT> &slots,
                                    const SlotMetaT &slotMeta, RefT ref) {
  if (!isValid(ref)) {
    return nullptr;
  }
  const ResourceHandleParts parts = unpackResourceHandle(ref.value);
  if (parts.index >= slots.size() ||
      !slotMeta.isValid(parts.index, parts.generation)) {
    return nullptr;
  }
  return &slots[parts.index];
}

template <typename SlotT, typename SlotMetaT, typename RefT>
[[nodiscard]] const SlotT *tryGetSlotImpl(const std::pmr::vector<SlotT> &slots,
                                          const SlotMetaT &slotMeta, RefT ref) {
  if (!isValid(ref)) {
    return nullptr;
  }
  const ResourceHandleParts parts = unpackResourceHandle(ref.value);
  if (parts.index >= slots.size() ||
      !slotMeta.isValid(parts.index, parts.generation)) {
    return nullptr;
  }
  return &slots[parts.index];
}

[[nodiscard]] Result<SlotReservation, std::string>
makeResourceSlotOverflowError(std::string_view context) {
  return Result<SlotReservation, std::string>::makeError(
      std::string(context) + ": slot pool exhausted");
}

template <typename Pool>
[[nodiscard]] bool slotPoolExhausted(const Pool &pool,
                                     uint32_t maxIndex) noexcept {
  const uint32_t slotCount = pool.slotCount();
  return slotCount > maxIndex + 1u ||
         (slotCount == maxIndex + 1u && pool.liveCount() == slotCount);
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

  const SceneSourceFingerprint sourceFingerprint =
      querySceneSourceFingerprint(cacheKey.normalizedSourcePath);
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
  if (ddsPack != nullptr && hasTextureExtension(slotData.path, ".dds")) {
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
    for (const ImportedTextureAcquireSpec &spec :
         kImportedTextureAcquireSpecs) {
      const ImportedMaterialTexture &slot = material.*(spec.slot);
      if (slot.sourceKind != MaterialTextureSourceKind::ExternalFile ||
          !hasTextureExtension(slot.path, ".dds")) {
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
                             std::string_view debugSuffix,
                             uint32_t sourceMaterialIndex) {
  return std::string(debugNamePrefix) + std::string(debugSuffix) +
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

[[nodiscard]] MaterialRequest::TextureRefs acquireRawImportedTextureRefs(
    ResourceManager &resources, const ImportedMaterialInfo &imported,
    std::string_view scenePath, std::string_view logContext,
    std::string_view debugNamePrefix, uint32_t sourceMaterialIndex,
    DdsTexturePack *ddsPack = nullptr) {
  MaterialRequest::TextureRefs textureRefs{};
  auto builderResult =
      SceneTextureArtifactBuilder::create(std::filesystem::path(scenePath));
  if (builderResult.hasError()) {
    NURI_LOG_WARNING("%.*s: texture artifact builder unavailable: %s",
                     static_cast<int>(logContext.size()), logContext.data(),
                     builderResult.error().c_str());
    return textureRefs;
  }
  SceneTextureArtifactBuilder builder = std::move(builderResult.value());
  const std::string canonicalScenePath = canonicalizeResourcePath(scenePath);
  const TextureCompressionCaps caps = resources.textureCompressionCaps();

  for (const ImportedTextureAcquireSpec &spec : kImportedTextureAcquireSpecs) {
    const ImportedMaterialTexture &source = imported.*(spec.slot);
    if (source.sourceKind == MaterialTextureSourceKind::None) {
      continue;
    }
    const TextureArtifactBuildOptions buildOptions =
        makeMaterialTextureArtifactBuildOptions(imported,
                                                spec.artifactSpecIndex);
    const std::string debugName = makeImportedTextureDebugName(
        debugNamePrefix, spec.debugSuffix, sourceMaterialIndex);
    Result<TextureRef, std::string> textureResult =
        Result<TextureRef, std::string>::makeResult(kInvalidTextureRef);
    if (source.sourceKind == MaterialTextureSourceKind::ExternalFile &&
        hasTextureExtension(source.path, ".dds")) {
      textureResult = acquireExternalImportedTexture(
          resources, source, buildOptions.loadOptions,
          TextureRequestKind::Texture2D, debugName, ddsPack);
    } else {
      const uint64_t identity = hashSceneTextureSourceIdentity(
          canonicalScenePath, source, buildOptions.loadOptions.srgb,
          textureArtifactProcessingTag(buildOptions));
      const Format targetFormat = selectTextureArtifactTargetFormat(
          caps.bc7, caps.etc2, buildOptions.loadOptions.srgb, 4u);
      auto artifact =
          builder.ensure(source, identity, targetFormat, buildOptions);
      if (artifact.hasError()) {
        textureResult =
            Result<TextureRef, std::string>::makeError(artifact.error());
      } else {
        textureResult = resources.acquireTexture(TextureRequest{
            .path = artifact.value().artifactPath.string(),
            .kind = TextureRequestKind::Ktx2Texture2D,
            .debugName = debugName,
        });
      }
    }
    if (textureResult.hasError()) {
      NURI_LOG_WARNING("%.*s: %s load failed for material %u: %s",
                       static_cast<int>(logContext.size()), logContext.data(),
                       spec.logName, sourceMaterialIndex,
                       textureResult.error().c_str());
      continue;
    }
    textureRefs.*(spec.outRef) = textureResult.value();
  }
  return textureRefs;
}

struct CachedTextureRefsAcquireResult {
  MaterialRequest::TextureRefs refs{};
  bool cacheUsable = true;
};

[[nodiscard]] CachedTextureRefsAcquireResult acquireCachedImportedTextureRefs(
    ResourceManager &resources, const ImportedMaterialInfo &imported,
    const SceneMaterialRecord &cached, std::string_view scenePath,
    std::string_view logContext, std::string_view debugNamePrefix,
    DdsTexturePack *ddsPack = nullptr) {
  CachedTextureRefsAcquireResult result{};
  auto builderResult =
      SceneTextureArtifactBuilder::create(std::filesystem::path(scenePath));
  if (builderResult.hasError()) {
    result.cacheUsable = false;
    NURI_LOG_WARNING("%.*s: texture artifact builder unavailable: %s",
                     static_cast<int>(logContext.size()), logContext.data(),
                     builderResult.error().c_str());
    return result;
  }
  SceneTextureArtifactBuilder builder = std::move(builderResult.value());
  const std::string canonicalScenePath = canonicalizeResourcePath(scenePath);
  const TextureCompressionCaps caps = resources.textureCompressionCaps();

  for (size_t slotIndex = 0; slotIndex < kImportedTextureAcquireSpecs.size();
       ++slotIndex) {
    const ImportedTextureAcquireSpec &spec =
        kImportedTextureAcquireSpecs[slotIndex];
    const SceneMaterialTextureCacheRecord &cacheRecord =
        cached.textureCache[slotIndex];
    const ImportedMaterialTexture &sourceSlot = imported.*(spec.slot);
    if (sourceSlot.sourceKind == MaterialTextureSourceKind::None) {
      continue;
    }
    const TextureArtifactBuildOptions buildOptions =
        makeMaterialTextureArtifactBuildOptions(imported,
                                                spec.artifactSpecIndex);
    const std::string debugName = makeImportedTextureDebugName(
        debugNamePrefix, spec.debugSuffix, cached.sourceMaterialIndex);
    Result<TextureRef, std::string> textureResult =
        Result<TextureRef, std::string>::makeResult(kInvalidTextureRef);
    if (sourceSlot.sourceKind == MaterialTextureSourceKind::ExternalFile &&
        hasTextureExtension(sourceSlot.path, ".dds")) {
      textureResult = acquireExternalImportedTexture(
          resources, sourceSlot, buildOptions.loadOptions,
          TextureRequestKind::Texture2D, debugName, ddsPack);
    } else {
      uint64_t identity = cacheRecord.artifactIdentityHash;
      if (identity == 0u) {
        identity = hashSceneTextureSourceIdentity(
            canonicalScenePath, sourceSlot, buildOptions.loadOptions.srgb,
            textureArtifactProcessingTag(buildOptions));
      }
      const Format targetFormat = selectTextureArtifactTargetFormat(
          caps.bc7, caps.etc2, buildOptions.loadOptions.srgb, 4u);
      auto artifact =
          builder.ensure(sourceSlot, identity, targetFormat, buildOptions);
      if (artifact.hasError()) {
        textureResult =
            Result<TextureRef, std::string>::makeError(artifact.error());
      } else {
        textureResult = resources.acquireTexture(TextureRequest{
            .path = artifact.value().artifactPath.string(),
            .kind = TextureRequestKind::Ktx2Texture2D,
            .debugName = debugName,
        });
      }
    }

    if (textureResult.hasError() || !isValid(textureResult.value())) {
      result.cacheUsable = false;
      NURI_LOG_WARNING(
          "%.*s: texture artifact %s load failed for material %u: %s",
          static_cast<int>(logContext.size()), logContext.data(), spec.logName,
          cached.sourceMaterialIndex,
          textureResult.hasError() ? textureResult.error().c_str()
                                   : "invalid texture reference");
      break;
    }
    result.refs.*(spec.outRef) = textureResult.value();
  }
  return result;
}

[[nodiscard]] Result<MaterialRef, std::string> acquireImportedMaterialInstance(
    ResourceManager &resources, const ImportedMaterialInfo &imported,
    const MaterialRequest::TextureRefs &textureRefs,
    std::string_view canonicalModelPath, std::string_view debugNamePrefix,
    uint32_t sourceMaterialIndex) {
  const MaterialTextureHandles emptyHandles{};
  return resources.acquireMaterial(MaterialRequest{
      .desc = Material::descFromImported(imported, emptyHandles),
      .textureRefs = textureRefs,
      .debugName = makeImportedMaterialDebugName(debugNamePrefix, imported,
                                                 sourceMaterialIndex),
      .sourceIdentity = makeImportedMaterialSourceIdentity(canonicalModelPath,
                                                           sourceMaterialIndex),
  });
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
      textureSlotsMeta_(memory_), materialSlotsMeta_(memory_),
      modelSlotsMeta_(memory_), materialHeaderTable_(memory_),
      materialClearcoatTable_(memory_), materialSheenTable_(memory_),
      materialTransmissionTable_(memory_), materialSpecularTable_(memory_),
      textureCache_(), materialCache_(), modelCache_(),
      pendingRetireTextures_(memory_), pendingRetireMaterials_(memory_),
      pendingRetireModels_(memory_) {}

ResourceManager::~ResourceManager() {
  // Dependency order:
  // model mappings reference materials, materials reference textures.
  for (uint32_t i = 0; i < modelSlots_.size(); ++i) {
    if (modelSlotsMeta_.isLive(i)) {
      destroyModelSlot(i);
    }
  }
  bool materialTablesDirty = false;
  for (uint32_t i = 0; i < materialSlots_.size(); ++i) {
    if (materialSlotsMeta_.isLive(i)) {
      destroyMaterialSlot(i, true);
      materialTablesDirty = true;
    }
  }
  if (materialTablesDirty) {
    rebuildPackedMaterialTables();
  }
  for (uint32_t i = 0; i < textureSlots_.size(); ++i) {
    if (textureSlotsMeta_.isLive(i)) {
      destroyTextureSlot(i);
    }
  }
}

uint64_t ResourceManager::retireLagFrames() const {
  return static_cast<uint64_t>(std::max(1u, gpu_.getSwapchainImageCount())) +
         1ull;
}

TextureRef ResourceManager::makeTextureRefForSlot(uint32_t index) const {
  return makeTextureRef(index, textureSlotsMeta_.generation(index));
}

MaterialRef ResourceManager::makeMaterialRefForSlot(uint32_t index) const {
  return makeMaterialRef(index, materialSlotsMeta_.generation(index));
}

ModelRef ResourceManager::makeModelRefForSlot(uint32_t index) const {
  return makeModelRef(index, modelSlotsMeta_.generation(index));
}

ResourceManager::TextureSlot *ResourceManager::tryGetSlot(TextureRef ref) {
  return tryGetSlotImpl(textureSlots_, textureSlotsMeta_, ref);
}

ResourceManager::MaterialSlot *ResourceManager::tryGetSlot(MaterialRef ref) {
  return tryGetSlotImpl(materialSlots_, materialSlotsMeta_, ref);
}

ResourceManager::ModelSlot *ResourceManager::tryGetSlot(ModelRef ref) {
  return tryGetSlotImpl(modelSlots_, modelSlotsMeta_, ref);
}

const ResourceManager::TextureSlot *
ResourceManager::tryGetSlot(TextureRef ref) const {
  return tryGetSlotImpl(textureSlots_, textureSlotsMeta_, ref);
}

const ResourceManager::MaterialSlot *
ResourceManager::tryGetSlot(MaterialRef ref) const {
  return tryGetSlotImpl(materialSlots_, materialSlotsMeta_, ref);
}

const ResourceManager::ModelSlot *
ResourceManager::tryGetSlot(ModelRef ref) const {
  return tryGetSlotImpl(modelSlots_, modelSlotsMeta_, ref);
}

Result<SlotReservation, std::string> ResourceManager::allocateTextureSlot() {
  if (slotPoolExhausted(textureSlotsMeta_, kResourceHandleIndexMask)) {
    return makeResourceSlotOverflowError(
        "ResourceManager::allocateTextureSlot");
  }
  const SlotReservation slot = textureSlotsMeta_.acquire();
  if (slot.appended) {
    textureSlots_.emplace_back(memory_);
  }
  return Result<SlotReservation, std::string>::makeResult(slot);
}

Result<SlotReservation, std::string> ResourceManager::allocateMaterialSlot() {
  if (slotPoolExhausted(materialSlotsMeta_, kResourceHandleIndexMask)) {
    return makeResourceSlotOverflowError(
        "ResourceManager::allocateMaterialSlot");
  }
  const SlotReservation slot = materialSlotsMeta_.acquire();
  if (slot.appended) {
    materialSlots_.emplace_back(memory_);
    materialHeaderTable_.push_back(MaterialHeaderGpuData{});
  }
  return Result<SlotReservation, std::string>::makeResult(slot);
}

Result<SlotReservation, std::string> ResourceManager::allocateModelSlot() {
  if (slotPoolExhausted(modelSlotsMeta_, kResourceHandleIndexMask)) {
    return makeResourceSlotOverflowError("ResourceManager::allocateModelSlot");
  }
  const SlotReservation slot = modelSlotsMeta_.acquire();
  if (slot.appended) {
    modelSlots_.emplace_back(memory_);
  }
  return Result<SlotReservation, std::string>::makeResult(slot);
}

void ResourceManager::destroyTextureSlot(uint32_t index) {
  TextureSlot &slot = textureSlots_[index];
  if (!textureSlotsMeta_.isLive(index)) {
    return;
  }

  const TextureKey key{
      .canonicalPath = std::string(slot.record.canonicalPath),
      .optionsHash = hashTextureLoadOptions(slot.record.loadOptions),
      .kind = slot.record.sourceKind,
  };
  textureCache_.erase(key);
  slot.owner.reset();

  slot.refCount = 0;
  slot.retireAfterFrame = kRetireFrameUnset;
  slot.record = TextureRecord(memory_);
  textureSlotsMeta_.release(index);
}

void ResourceManager::destroyMaterialSlot(uint32_t index, bool skipRebuild) {
  MaterialSlot &slot = materialSlots_[index];
  if (!materialSlotsMeta_.isLive(index)) {
    return;
  }

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
  slot.retireAfterFrame = kRetireFrameUnset;
  slot.record = MaterialRecord(memory_);
  if (index < materialHeaderTable_.size()) {
    materialHeaderTable_[index] = MaterialHeaderGpuData{};
  }
  materialSlotsMeta_.release(index);
  if (skipRebuild) {
    materialTablesDirty_ = true;
  } else {
    markMaterialTablesDirty();
  }
}

void ResourceManager::destroyModelSlot(uint32_t index) {
  ModelSlot &slot = modelSlots_[index];
  if (!modelSlotsMeta_.isLive(index)) {
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
      .sceneMeshIndex = slot.record.sceneMeshIndex,
  };
  modelCache_.erase(key);

  slot.refCount = 0;
  slot.retireAfterFrame = kRetireFrameUnset;
  slot.record = ModelRecord(memory_);
  modelSlotsMeta_.release(index);
  markModelMaterialBindingsDirty();
}

void ResourceManager::rebuildPackedMaterialTables() {
  std::pmr::vector<MaterialHeaderGpuData> nextHeaders(memory_);
  std::pmr::vector<MaterialClearcoatGpuData> nextClearcoat(memory_);
  std::pmr::vector<MaterialSheenGpuData> nextSheen(memory_);
  std::pmr::vector<MaterialTransmissionGpuData> nextTransmission(memory_);
  std::pmr::vector<MaterialSpecularGpuData> nextSpecular(memory_);
  nextHeaders.resize(materialSlots_.size());

  const auto textureIndex = [this](TextureRef ref) -> uint32_t {
    const TextureRecord *record = tryGet(ref);
    return record != nullptr ? record->bindlessIndex
                             : kInvalidTextureBindlessIndex;
  };

  for (uint32_t index = 0; index < materialSlots_.size(); ++index) {
    if (!materialSlotsMeta_.isLive(index)) {
      nextHeaders[index] = MaterialHeaderGpuData{};
      continue;
    }

    MaterialPackedGpuData &packed = materialSlots_[index].record.packedGpuData;
    const MaterialRequest::TextureRefs &textureRefs =
        materialSlots_[index].record.textureRefs;

    packed.header.commonTextureIndices = glm::uvec4(
        textureIndex(textureRefs.baseColor),
        textureIndex(textureRefs.metallicRoughness),
        textureIndex(textureRefs.normal), textureIndex(textureRefs.occlusion));
    packed.header.emissiveTextureIndex = textureIndex(textureRefs.emissive);
    packed.clearcoat.textureIndices =
        glm::uvec4(textureIndex(textureRefs.clearcoat),
                   textureIndex(textureRefs.clearcoatRoughness),
                   textureIndex(textureRefs.clearcoatNormal), 0u);
    packed.sheen.textureIndices =
        glm::uvec4(textureIndex(textureRefs.sheenColor),
                   textureIndex(textureRefs.sheenRoughness), 0u, 0u);
    packed.transmission.textureIndices =
        glm::uvec4(textureIndex(textureRefs.transmission),
                   textureIndex(textureRefs.thickness), 0u, 0u);
    packed.specular.textureIndices =
        glm::uvec4(textureIndex(textureRefs.specular),
                   textureIndex(textureRefs.specularColor), 0u, 0u);

    MaterialHeaderGpuData header = packed.header;
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
  if (request.path.empty()) {
    return std::nullopt;
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
      return it->second;
    }
    textureCache_.erase(it);
  }
  return std::nullopt;
}

Result<TextureRef, std::string>
ResourceManager::acquireTexture(const TextureRequest &request) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (request.path.empty()) {
    return Result<TextureRef, std::string>::makeError(
        "ResourceManager::acquireTexture: path is empty");
  }
  if (const std::optional<TextureRef> cached = tryAcquireTexture(request);
      cached.has_value()) {
    ++telemetry_.textureAcquireHits;
    return Result<TextureRef, std::string>::makeResult(*cached);
  }
  ++telemetry_.textureAcquireMisses;

  const std::string canonicalPath = canonicalizeResourcePath(request.path);
  TextureKey key{
      .canonicalPath = canonicalPath,
      .optionsHash = hashTextureLoadOptions(request.loadOptions),
      .kind = request.kind,
  };

  Result<std::unique_ptr<Texture>, std::string> textureResult =
      Result<std::unique_ptr<Texture>, std::string>::makeError(
          "ResourceManager::acquireTexture: uninitialized result");

  switch (request.kind) {
  case TextureRequestKind::Texture2D:
    if (request.ddsPack != nullptr) {
      if (!hasTextureExtension(canonicalPath, ".dds")) {
        return Result<TextureRef, std::string>::makeError(
            "ResourceManager::acquireTexture: texture packs are only "
            "supported for DDS textures");
      }
      auto packedData = request.ddsPack->read(canonicalPath);
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
  return storeAcquiredTexture(key, canonicalPath, request,
                              std::move(textureResult.value()));
}

Result<TextureRef, std::string> ResourceManager::storeAcquiredTexture(
    const TextureKey &key, std::string_view canonicalPath,
    const TextureRequest &request, std::unique_ptr<Texture> texture) {
  if (!texture || !texture->valid()) {
    return Result<TextureRef, std::string>::makeError(
        "ResourceManager::storeAcquiredTexture: loaded texture is invalid");
  }

  auto slotResult = allocateTextureSlot();
  if (slotResult.hasError()) {
    return Result<TextureRef, std::string>::makeError(slotResult.error());
  }
  const uint32_t slotIndex = slotResult.value().index;
  TextureSlot &slot = textureSlots_[slotIndex];
  const TextureRef ref = makeTextureRefForSlot(slotIndex);
  slot.refCount = 1;
  slot.retireAfterFrame = kRetireFrameUnset;

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
  if (request.path.empty()) {
    return Result<TextureRef, std::string>::makeError(
        "ResourceManager::adoptPreparedTexture: path is empty");
  }
  if (const std::optional<TextureRef> cached = tryAcquireTexture(request);
      cached.has_value()) {
    return Result<TextureRef, std::string>::makeResult(*cached);
  }
  const std::string canonicalPath = canonicalizeResourcePath(request.path);
  const TextureKey key{
      .canonicalPath = canonicalPath,
      .optionsHash = hashTextureLoadOptions(request.loadOptions),
      .kind = request.kind,
  };
  return storeAcquiredTexture(key, canonicalPath, request, std::move(texture));
}

ModelRef ResourceManager::tryAcquireCachedModel(const ModelKey &key) {
  if (auto it = modelCache_.find(key); it != modelCache_.end()) {
    if (ModelSlot *cached = tryGetSlot(it->second)) {
      ++cached->refCount;
      cached->retireAfterFrame = kRetireFrameUnset;
      return it->second;
    }
    modelCache_.erase(it);
  }
  return kInvalidModelRef;
}

std::optional<ModelRef>
ResourceManager::tryAcquireModel(const ModelRequest &request) {
  if (request.path.empty()) {
    return std::nullopt;
  }
  const std::string canonicalPath = canonicalizeResourcePath(request.path);
  const ModelKey key{
      .canonicalPath = canonicalPath,
      .importOptionsHash = hashModelImportOptions(request.importOptions),
      .sceneMeshIndex = request.sceneMeshIndex,
  };
  const ModelRef ref = tryAcquireCachedModel(key);
  return isValid(ref) ? std::optional<ModelRef>(ref) : std::nullopt;
}

Result<ModelRef, std::string> ResourceManager::storeAcquiredModel(
    const ModelKey &key, std::string_view canonicalPath, uint64_t optionsHash,
    const ModelRequest &request, std::unique_ptr<Model> model) {
  if (!model) {
    return Result<ModelRef, std::string>::makeError(
        "ResourceManager::storeAcquiredModel: model creation returned null");
  }

  auto slotResult = allocateModelSlot();
  if (slotResult.hasError()) {
    return Result<ModelRef, std::string>::makeError(slotResult.error());
  }
  const uint32_t slotIndex = slotResult.value().index;
  ModelSlot &slot = modelSlots_[slotIndex];
  const ModelRef ref = makeModelRefForSlot(slotIndex);
  slot.refCount = 1;
  slot.retireAfterFrame = kRetireFrameUnset;

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
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (request.path.empty()) {
    return Result<ModelRef, std::string>::makeError(
        "ResourceManager::acquireModel: path is empty");
  }

  if (const std::optional<ModelRef> cached = tryAcquireModel(request);
      cached.has_value()) {
    ++telemetry_.modelAcquireHits;
    return Result<ModelRef, std::string>::makeResult(*cached);
  }
  ++telemetry_.modelAcquireMisses;

  const std::string canonicalPath = canonicalizeResourcePath(request.path);
  const uint64_t optionsHash = hashModelImportOptions(request.importOptions);
  ModelKey key{.canonicalPath = canonicalPath,
               .importOptionsHash = optionsHash,
               .sceneMeshIndex = request.sceneMeshIndex};

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

  return storeAcquiredModel(key, canonicalPath, optionsHash, request,
                            std::move(modelResult.value()));
}

Result<ModelRef, std::string>
ResourceManager::adoptPreparedModel(const ModelRequest &request,
                                    std::unique_ptr<Model> model) {
  if (request.path.empty()) {
    return Result<ModelRef, std::string>::makeError(
        "ResourceManager::adoptPreparedModel: path is empty");
  }
  if (const std::optional<ModelRef> cached = tryAcquireModel(request);
      cached.has_value()) {
    return Result<ModelRef, std::string>::makeResult(*cached);
  }
  const std::string canonicalPath = canonicalizeResourcePath(request.path);
  const uint64_t optionsHash = hashModelImportOptions(request.importOptions);
  const ModelKey key{
      .canonicalPath = canonicalPath,
      .importOptionsHash = optionsHash,
      .sceneMeshIndex = request.sceneMeshIndex,
  };
  return storeAcquiredModel(key, canonicalPath, optionsHash, request,
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

  auto slotResult = allocateModelSlot();
  if (slotResult.hasError()) {
    return Result<ModelRef, std::string>::makeError(slotResult.error());
  }

  const uint32_t slotIndex = slotResult.value().index;
  ModelSlot &slot = modelSlots_[slotIndex];
  const ModelRef ref = makeModelRefForSlot(slotIndex);
  slot.refCount = 1;
  slot.retireAfterFrame = kRetireFrameUnset;

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

  for (const MaterialTextureResolveSpec &spec : kMaterialTextureResolveSpecs) {
    auto resolveResult =
        resolveTextureSlot(request.textureRefs.*(spec.ref),
                           resolvedDesc.textures.*(spec.handle), spec.slotName);
    if (resolveResult.hasError()) {
      return Result<MaterialRef, std::string>::makeError(resolveResult.error());
    }
  }

  Material::finalizeDesc(resolvedDesc);

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
  auto slotResult = allocateMaterialSlot();
  if (slotResult.hasError()) {
    return Result<MaterialRef, std::string>::makeError(slotResult.error());
  }
  const uint32_t slotIndex = slotResult.value().index;
  MaterialSlot &slot = materialSlots_[slotIndex];
  const MaterialRef ref = makeMaterialRefForSlot(slotIndex);
  slot.refCount = 1;
  slot.retireAfterFrame = kRetireFrameUnset;

  slot.record = MaterialRecord(memory_);
  slot.record.ref = ref;
  slot.record.desc = material.desc();
  slot.record.textureRefs = request.textureRefs;
  slot.record.packedGpuData = material.packedGpuData();
  slot.record.descHash = descHash;
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

  materialCache_.emplace(std::move(key), ref);
  return Result<MaterialRef, std::string>::makeResult(ref);
}

MaterialDesc ResourceManager::materialDescFromImported(
    const ImportedMaterialInfo &imported,
    const MaterialTextureHandles &textures) {
  return Material::descFromImported(imported, textures);
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

  ImportedMaterialBatch batch{};
  const std::string canonicalModelPath =
      canonicalizeResourcePath(request.modelPath);
  std::optional<ImportedMaterialSet> importedMaterialSet{};
  std::string materialInfoError{};
  if (auto materialInfoResult =
          MeshImporter::loadMaterialInfoFromFile(request.modelPath);
      !materialInfoResult.hasError()) {
    importedMaterialSet = std::move(materialInfoResult.value());
  } else {
    materialInfoError = materialInfoResult.error();
  }
  std::unique_ptr<DdsTexturePack> ddsPack =
      importedMaterialSet.has_value()
          ? tryEnsureDdsTexturePack(
                request.modelPath, importedMaterialSet.value(),
                "ResourceManager::acquireMaterialsFromModel")
          : nullptr;

  if (auto cachedMaterials = tryLoadSceneMaterialCache(request.modelPath);
      cachedMaterials.has_value()) {
    struct PendingCachedMaterial {
      uint32_t sourceMaterialIndex = 0u;
      MaterialRef material = kInvalidMaterialRef;
    };

    std::vector<PendingCachedMaterial> pendingMaterials{};
    pendingMaterials.reserve(cachedMaterials->materials.size());
    bool cacheUsable = true;

    for (const SceneMaterialRecord &cached : cachedMaterials->materials) {
      const ImportedMaterialInfo *imported = &cached.sourceMaterial;
      if (importedMaterialSet.has_value() &&
          cached.sourceMaterialIndex < importedMaterialSet->materials.size()) {
        imported = &importedMaterialSet->materials[cached.sourceMaterialIndex];
      }

      const CachedTextureRefsAcquireResult textureRefsResult =
          acquireCachedImportedTextureRefs(
              *this, *imported, cached, request.modelPath,
              "ResourceManager::acquireMaterialsFromModel",
              request.debugNamePrefix, ddsPack.get());

      if (!textureRefsResult.cacheUsable) {
        cacheUsable = false;
        releaseMaterialTextureRefs(*this, textureRefsResult.refs);
        break;
      }

      const uint32_t sourceMaterialIndex = cached.sourceMaterialIndex;
      auto acquireMaterialResult = acquireImportedMaterialInstance(
          *this, *imported, textureRefsResult.refs, canonicalModelPath,
          request.debugNamePrefix, sourceMaterialIndex);
      releaseMaterialTextureRefs(*this, textureRefsResult.refs);
      if (acquireMaterialResult.hasError()) {
        NURI_LOG_WARNING("ResourceManager::acquireMaterialsFromModel: cached "
                         "material acquire failed for source material %u: %s",
                         sourceMaterialIndex,
                         acquireMaterialResult.error().c_str());
        cacheUsable = false;
        break;
      }

      pendingMaterials.push_back(PendingCachedMaterial{
          .sourceMaterialIndex = sourceMaterialIndex,
          .material = acquireMaterialResult.value(),
      });
    }

    if (!cacheUsable) {
      for (const PendingCachedMaterial &pending : pendingMaterials) {
        if (isValid(pending.material)) {
          release(pending.material);
        }
      }
    } else if (!pendingMaterials.empty()) {
      for (const PendingCachedMaterial &pending : pendingMaterials) {
        const bool mappedToModel = setModelMaterialForSource(
            request.model, pending.sourceMaterialIndex, pending.material);
        if (!mappedToModel) {
          NURI_LOG_DEBUG(
              "ResourceManager::acquireMaterialsFromModel: source material %u "
              "not mapped to model",
              pending.sourceMaterialIndex);
        }
        if (!isValid(batch.firstMaterial)) {
          batch.firstMaterial = pending.material;
          retain(pending.material);
        }
        ++batch.createdMaterialCount;
        release(pending.material);
      }
      return Result<ImportedMaterialBatch, std::string>::makeResult(batch);
    }
  }

  if (!importedMaterialSet.has_value()) {
    return Result<ImportedMaterialBatch, std::string>::makeError(
        "ResourceManager::acquireMaterialsFromModel: failed to parse material "
        "metadata: " +
        materialInfoError);
  }

  const ImportedMaterialSet &materialSet = importedMaterialSet.value();
  for (uint32_t sourceMaterialIndex = 0;
       sourceMaterialIndex < materialSet.materials.size();
       ++sourceMaterialIndex) {
    const ImportedMaterialInfo &imported =
        materialSet.materials[sourceMaterialIndex];

    const MaterialRequest::TextureRefs textureRefs =
        acquireRawImportedTextureRefs(
            *this, imported, request.modelPath,
            "ResourceManager::acquireMaterialsFromModel",
            request.debugNamePrefix, sourceMaterialIndex, ddsPack.get());
    auto acquireMaterialResult = acquireImportedMaterialInstance(
        *this, imported, textureRefs, canonicalModelPath,
        request.debugNamePrefix, sourceMaterialIndex);
    releaseMaterialTextureRefs(*this, textureRefs);
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
        prefab.materialAssets[assetIndex].sourceMaterialIndex, assetIndex);
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
      const CachedTextureRefsAcquireResult textureRefsResult =
          acquireCachedImportedTextureRefs(
              *this, *imported, cached, prefab.sourcePath,
              "ResourceManager::acquireScenePrefabAssets", "scene_prefab",
              ddsPack.get());

      if (!textureRefsResult.cacheUsable) {
        cacheUsable = false;
        releaseMaterialTextureRefs(*this, textureRefsResult.refs);
        break;
      }

      auto acquireMaterialResult = acquireImportedMaterialInstance(
          *this, *imported, textureRefsResult.refs, canonicalModelPath,
          "scene_prefab", cached.sourceMaterialIndex);
      releaseMaterialTextureRefs(*this, textureRefsResult.refs);
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
          prefab.materialAssets[assetIndex].sourceMaterialIndex;
      if (sourceMaterialIndex >= materialSet.materials.size()) {
        continue;
      }
      const ImportedMaterialInfo &imported =
          materialSet.materials[sourceMaterialIndex];

      const MaterialRequest::TextureRefs textureRefs =
          acquireRawImportedTextureRefs(
              *this, imported, prefab.sourcePath,
              "ResourceManager::acquireScenePrefabAssets", "scene_prefab",
              sourceMaterialIndex, ddsPack.get());
      auto acquireMaterialResult = acquireImportedMaterialInstance(
          *this, imported, textureRefs, canonicalModelPath, "scene_prefab",
          sourceMaterialIndex);
      releaseMaterialTextureRefs(*this, textureRefs);
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
  pendingMeshIndices.reserve(prefab.renderables.size());
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
                modelRef, prefab.materialAssets[assetIndex].sourceMaterialIndex,
                assets.materials[assetIndex]);
          }
        }
      };

  for (const ScenePrefabRenderable &prefabRenderable : prefab.renderables) {
    if (prefabRenderable.meshIndex >= assets.models.size() ||
        isValid(assets.models[prefabRenderable.meshIndex])) {
      continue;
    }

    const uint32_t sourceSceneMeshIndex =
        prefab.meshAssets[prefabRenderable.meshIndex].sourceSceneMeshIndex;
    const ModelKey key = makeSceneMeshModelKey(canonicalModelPath, optionsHash,
                                               sourceSceneMeshIndex);
    if (const ModelRef cachedRef = tryAcquireCachedModel(key);
        isValid(cachedRef)) {
      ++telemetry_.modelAcquireHits;
      assets.models[prefabRenderable.meshIndex] = cachedRef;
      continue;
    }

    ++telemetry_.modelAcquireMisses;
    pendingMeshIndices.push_back(prefabRenderable.meshIndex);
  }

  if (!pendingMeshIndices.empty()) {
    std::pmr::vector<uint8_t> pendingMeshSeen(memory_);
    pendingMeshSeen.resize(assets.models.size(), 0u);
    std::pmr::vector<uint32_t> uniquePendingMeshIndices(memory_);
    uniquePendingMeshIndices.reserve(pendingMeshIndices.size());
    for (const uint32_t meshAssetIndex : pendingMeshIndices) {
      if (meshAssetIndex >= pendingMeshSeen.size() ||
          pendingMeshSeen[meshAssetIndex] != 0u) {
        continue;
      }
      pendingMeshSeen[meshAssetIndex] = 1u;
      uniquePendingMeshIndices.push_back(meshAssetIndex);
    }

    std::pmr::vector<uint32_t> sourceSceneMeshIndices(memory_);
    sourceSceneMeshIndices.reserve(uniquePendingMeshIndices.size());
    for (const uint32_t meshAssetIndex : uniquePendingMeshIndices) {
      sourceSceneMeshIndices.push_back(
          prefab.meshAssets[meshAssetIndex].sourceSceneMeshIndex);
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
    if (sceneMeshes.size() != uniquePendingMeshIndices.size()) {
      return cleanupAndError(
          "ResourceManager::acquireScenePrefabAssets: batched scene mesh "
          "count mismatch");
    }

    for (size_t meshOrdinal = 0; meshOrdinal < uniquePendingMeshIndices.size();
         ++meshOrdinal) {
      const uint32_t meshIndex = uniquePendingMeshIndices[meshOrdinal];
      const uint32_t sourceSceneMeshIndex =
          prefab.meshAssets[meshIndex].sourceSceneMeshIndex;
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

  std::pmr::vector<uint8_t> configuredMeshes(memory_);
  configuredMeshes.resize(assets.models.size(), 0u);
  for (const ScenePrefabRenderable &prefabRenderable : prefab.renderables) {
    if (prefabRenderable.meshIndex >= assets.models.size() ||
        configuredMeshes[prefabRenderable.meshIndex] != 0u) {
      continue;
    }

    const ModelRef modelRef = assets.models[prefabRenderable.meshIndex];
    if (!isValid(modelRef)) {
      return cleanupAndError(
          "ResourceManager::acquireScenePrefabAssets: scene mesh " +
          std::to_string(prefabRenderable.meshIndex) + " was not resolved");
    }
    configuredMeshes[prefabRenderable.meshIndex] = 1u;
    configurePrefabModelMaterials(modelRef);
  }

  return Result<ScenePrefabAssets, std::string>::makeResult(std::move(assets));
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
    pendingRetireTextures_.push_back(ref);
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
    pendingRetireModels_.push_back(ref);
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
    pendingRetireMaterials_.push_back(ref);
  }
}

bool ResourceManager::owns(TextureRef ref) const noexcept {
  return isSlotLiveForRef(textureSlots_, textureSlotsMeta_, ref);
}

bool ResourceManager::owns(ModelRef ref) const noexcept {
  return isSlotLiveForRef(modelSlots_, modelSlotsMeta_, ref);
}

bool ResourceManager::owns(MaterialRef ref) const noexcept {
  return isSlotLiveForRef(materialSlots_, materialSlotsMeta_, ref);
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

void ResourceManager::beginPublicationBatch() { ++publicationBatchDepth_; }

void ResourceManager::endPublicationBatch() {
  NURI_ASSERT(publicationBatchDepth_ > 0u,
              "ResourceManager::endPublicationBatch: no batch is active");
  if (publicationBatchDepth_ == 0u) {
    return;
  }
  --publicationBatchDepth_;
  flushPublicationVersions();
}

void ResourceManager::collectGarbage(uint64_t completedFrameIndex) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);

  // Keep destruction order consistent with dependencies:
  // models -> materials -> textures.
  // Only entries in the pending-retire lists are checked; stale refs (slot
  // destroyed or reused) and revived refs (retain called after release) are
  // detected via tryGetSlot and removed from the list without destroying.
  // destroyModelSlot / destroyMaterialSlot may push additional entries to
  // the material / texture lists, which will be processed in the same call.

  {
    size_t i = 0;
    while (i < pendingRetireModels_.size()) {
      const ModelRef ref = pendingRetireModels_[i];
      const ModelSlot *slot = tryGetSlot(ref);
      if (slot == nullptr || slot->refCount != 0u ||
          slot->retireAfterFrame == kRetireFrameUnset) {
        pendingRetireModels_[i] = pendingRetireModels_.back();
        pendingRetireModels_.pop_back();
        continue;
      }
      if (completedFrameIndex < slot->retireAfterFrame) {
        ++i;
        continue;
      }
      destroyModelSlot(indexOf(ref));
      pendingRetireModels_[i] = pendingRetireModels_.back();
      pendingRetireModels_.pop_back();
    }
  }

  {
    bool materialTablesDirty = false;
    size_t i = 0;
    while (i < pendingRetireMaterials_.size()) {
      const MaterialRef ref = pendingRetireMaterials_[i];
      const MaterialSlot *slot = tryGetSlot(ref);
      if (slot == nullptr || slot->refCount != 0u ||
          slot->retireAfterFrame == kRetireFrameUnset) {
        pendingRetireMaterials_[i] = pendingRetireMaterials_.back();
        pendingRetireMaterials_.pop_back();
        continue;
      }
      if (completedFrameIndex < slot->retireAfterFrame) {
        ++i;
        continue;
      }
      destroyMaterialSlot(indexOf(ref), true);
      materialTablesDirty = true;
      pendingRetireMaterials_[i] = pendingRetireMaterials_.back();
      pendingRetireMaterials_.pop_back();
    }
    if (materialTablesDirty) {
      flushPublicationVersions();
    }
  }

  {
    size_t i = 0;
    while (i < pendingRetireTextures_.size()) {
      const TextureRef ref = pendingRetireTextures_[i];
      const TextureSlot *slot = tryGetSlot(ref);
      if (slot == nullptr || slot->refCount != 0u ||
          slot->retireAfterFrame == kRetireFrameUnset) {
        pendingRetireTextures_[i] = pendingRetireTextures_.back();
        pendingRetireTextures_.pop_back();
        continue;
      }
      if (completedFrameIndex < slot->retireAfterFrame) {
        ++i;
        continue;
      }
      destroyTextureSlot(indexOf(ref));
      pendingRetireTextures_[i] = pendingRetireTextures_.back();
      pendingRetireTextures_.pop_back();
    }
  }
}

PoolStats ResourceManager::stats() const {
  PoolStats s{};
  for (uint32_t i = 0; i < textureSlots_.size(); ++i) {
    const TextureSlot &slot = textureSlots_[i];
    if (textureSlotsMeta_.isLive(i)) {
      if (slot.refCount > 0u) {
        ++s.liveTextures;
      } else {
        ++s.retiredTextures;
      }
    }
  }
  for (uint32_t i = 0; i < materialSlots_.size(); ++i) {
    const MaterialSlot &slot = materialSlots_[i];
    if (materialSlotsMeta_.isLive(i)) {
      if (slot.refCount > 0u) {
        ++s.liveMaterials;
      } else {
        ++s.retiredMaterials;
      }
    }
  }
  for (uint32_t i = 0; i < modelSlots_.size(); ++i) {
    const ModelSlot &slot = modelSlots_[i];
    if (modelSlotsMeta_.isLive(i)) {
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
  s.textureCache = Texture::cacheTelemetry();
  s.ddsTexturePacks = ddsTexturePackTelemetry();
  s.textureUploads = gpu_.getTextureUploadTelemetry();
  return s;
}

uint32_t ResourceManager::materialTableIndex(MaterialRef ref) const {
  if (!isSlotLiveForRef(materialSlots_, materialSlotsMeta_, ref)) {
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
  markModelMaterialBindingsDirty();
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
