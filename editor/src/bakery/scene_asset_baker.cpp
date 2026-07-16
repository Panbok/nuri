#include "nuri/editor_pch.h"

#include "nuri/bakery/scene_asset_baker.h"

#include "nuri/core/profiling.h"
#include "nuri/resources/gpu/resource_keys.h"
#include "nuri/resources/mesh_importer.h"
#include "nuri/resources/storage/material/material_binary_serializer.h"
#include "nuri/resources/storage/material/material_cache_utils.h"
#include "nuri/resources/storage/mesh/mesh_cache_utils.h"
#include "nuri/resources/storage/texture/dds_texture_pack.h"
#include "nuri/resources/storage/texture/material_texture_artifact.h"
#include "nuri/resources/storage/texture/texture_artifact_builder.h"

#include <unordered_set>

namespace nuri::bakery::detail {
namespace {

[[nodiscard]] TextureArtifactTarget
toArtifactTarget(SceneTextureArtifactTarget target) noexcept {
  switch (target) {
  case SceneTextureArtifactTarget::BC7:
    return TextureArtifactTarget::BC7;
  case SceneTextureArtifactTarget::ETC2:
    return TextureArtifactTarget::ETC2;
  case SceneTextureArtifactTarget::RGBA8:
    return TextureArtifactTarget::RGBA8;
  }
  return TextureArtifactTarget::RGBA8;
}

[[nodiscard]] Result<bool, std::string>
writeSceneMaterialCacheToDisk(const SceneTextureArtifactBakePlan &plan,
                              const SceneMaterialCacheData &cacheData) {
  auto cacheKey = buildSceneMaterialCacheKey(plan.scenePath);
  if (cacheKey.hasError()) {
    return Result<bool, std::string>::makeError(cacheKey.error());
  }
  const SceneSourceFingerprint sourceFingerprint =
      querySceneSourceFingerprint(cacheKey.value().normalizedSourcePath);
  auto bytes = materialBinarySerialize(MaterialBinarySerializeInput{
      .sourcePathHash = cacheKey.value().sourcePathHash,
      .sourceSizeBytes = sourceFingerprint.sizeBytes,
      .sourceMtimeNs = sourceFingerprint.mtimeNs,
      .materials = std::span<const SceneMaterialRecord>(
          cacheData.materials.data(), cacheData.materials.size()),
  });
  if (bytes.hasError()) {
    return Result<bool, std::string>::makeError(bytes.error());
  }

  std::error_code ec;
  if (!plan.forceRebuild &&
      std::filesystem::exists(plan.materialCachePath, ec) && !ec) {
    auto existing = readBinaryFile(plan.materialCachePath);
    if (!existing.hasError() && existing.value() == bytes.value()) {
      return Result<bool, std::string>::makeResult(false);
    }
  }
  auto write = writeBinaryFileAtomic(plan.materialCachePath, bytes.value());
  if (write.hasError()) {
    return Result<bool, std::string>::makeError(write.error());
  }
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] std::vector<DdsTexturePackSource>
collectDdsTexturePackSources(const SceneMaterialCacheData &cacheData) {
  std::vector<DdsTexturePackSource> sources{};
  std::unordered_set<std::string> seen{};
  for (const SceneMaterialRecord &record : cacheData.materials) {
    for (const MaterialTextureArtifactSpec &spec :
         kMaterialTextureArtifactSpecs) {
      const MaterialTextureSlotData &slot = record.sourceMaterial.*(spec.slot);
      if (slot.sourceKind != MaterialTextureSourceKind::ExternalFile) {
        continue;
      }
      std::string extension =
          std::filesystem::path(slot.path).extension().string();
      std::ranges::transform(
          extension, extension.begin(),
          [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
      if (extension != ".dds") {
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

} // namespace

Result<SceneTextureArtifactBakePlan, std::string>
planSceneTextureArtifactsBake(const SceneTextureArtifactsBakeRequest &request) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (request.scenePath.empty()) {
    return Result<SceneTextureArtifactBakePlan, std::string>::makeError(
        "Scene texture artifact bake: scene path is empty");
  }
  auto cacheKey = buildSceneMaterialCacheKey(request.scenePath);
  if (cacheKey.hasError()) {
    return Result<SceneTextureArtifactBakePlan, std::string>::makeError(
        cacheKey.error());
  }
  return Result<SceneTextureArtifactBakePlan, std::string>::makeResult(
      SceneTextureArtifactBakePlan{
          .shouldBake = true,
          .scenePath = request.scenePath,
          .materialCachePath = cacheKey.value().cachePath,
          .prebuildNativeTargets = request.prebuildNativeTargets,
          .forceRebuild = request.forceRebuild,
      });
}

Result<SceneTextureArtifactBakeStats, std::string>
bakeSceneTextureArtifactsToDisk(const SceneTextureArtifactBakePlan &plan) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  auto materialInfo =
      MeshImporter::loadMaterialInfoFromFile(plan.scenePath.string());
  if (materialInfo.hasError()) {
    return Result<SceneTextureArtifactBakeStats, std::string>::makeError(
        "Scene texture artifact bake: failed to read material info: " +
        materialInfo.error());
  }
  auto builder = SceneTextureArtifactBuilder::create(plan.scenePath);
  if (builder.hasError()) {
    return Result<SceneTextureArtifactBakeStats, std::string>::makeError(
        builder.error());
  }

  const std::string canonicalScenePath =
      canonicalizeResourcePath(plan.scenePath.string());
  SceneMaterialCacheData cacheData{};
  cacheData.materials.reserve(materialInfo.value().materials.size());
  SceneTextureArtifactBakeStats stats{};

  for (uint32_t materialIndex = 0u;
       materialIndex < materialInfo.value().materials.size(); ++materialIndex) {
    const MaterialData &material =
        materialInfo.value().materials[materialIndex];
    SceneMaterialRecord record{};
    record.sourceMaterialIndex = materialIndex;
    record.sourceMaterial = material;

    for (size_t slotIndex = 0u;
         slotIndex < kMaterialTextureArtifactSpecs.size(); ++slotIndex) {
      const MaterialTextureArtifactSpec &spec =
          kMaterialTextureArtifactSpecs[slotIndex];
      const MaterialTextureSlotData &slot = material.*(spec.slot);
      if (slot.sourceKind == MaterialTextureSourceKind::None) {
        continue;
      }
      const TextureArtifactBuildOptions options =
          makeMaterialTextureArtifactBuildOptions(material, slotIndex);
      const uint64_t identity = hashSceneTextureSourceIdentity(
          canonicalScenePath, slot, options.loadOptions.srgb,
          textureArtifactProcessingTag(options));
      record.textureCache[slotIndex].artifactIdentityHash = identity;

      if (slot.sourceKind == MaterialTextureSourceKind::ExternalFile) {
        std::string extension =
            std::filesystem::path(slot.path).extension().string();
        std::ranges::transform(extension, extension.begin(),
                               [](unsigned char ch) {
                                 return static_cast<char>(std::tolower(ch));
                               });
        if (extension == ".dds") {
          continue;
        }
      }

      std::array<Format, 3> builtFormats{};
      size_t builtFormatCount = 0u;
      for (SceneTextureArtifactTarget requestedTarget :
           plan.prebuildNativeTargets) {
        const Format targetFormat = resolveTextureArtifactTargetFormat(
            toArtifactTarget(requestedTarget), options.loadOptions.srgb, 4u);
        if (std::find(builtFormats.begin(),
                      builtFormats.begin() +
                          static_cast<ptrdiff_t>(builtFormatCount),
                      targetFormat) !=
            builtFormats.begin() + static_cast<ptrdiff_t>(builtFormatCount)) {
          continue;
        }
        builtFormats[builtFormatCount++] = targetFormat;
        auto artifact = builder.value().ensure(slot, identity, targetFormat,
                                               options, plan.forceRebuild);
        if (artifact.hasError()) {
          return Result<SceneTextureArtifactBakeStats, std::string>::makeError(
              artifact.error());
        }
        if (artifact.value().built) {
          ++stats.artifactsWritten;
          stats.artifactBytesWritten += artifact.value().artifactSizeBytes;
          stats.wroteAnyFiles = true;
        }
      }
    }
    cacheData.materials.push_back(std::move(record));
  }

  std::vector<DdsTexturePackSource> ddsSources =
      collectDdsTexturePackSources(cacheData);
  if (ddsSources.size() >= kMinAutomaticDdsTexturePackEntries) {
    auto ddsPack = ensureDdsTexturePack(plan.scenePath, ddsSources);
    if (ddsPack.hasError()) {
      return Result<SceneTextureArtifactBakeStats, std::string>::makeError(
          ddsPack.error());
    }
    stats.ddsPackEntries = ddsPack.value().pack->entryCount();
    stats.ddsPackBytes = ddsPack.value().pack->artifactSizeBytes();
    stats.wroteDdsPack = ddsPack.value().built;
    stats.wroteAnyFiles = stats.wroteAnyFiles || stats.wroteDdsPack;
  }

  auto materialCache = writeSceneMaterialCacheToDisk(plan, cacheData);
  if (materialCache.hasError()) {
    return Result<SceneTextureArtifactBakeStats, std::string>::makeError(
        materialCache.error());
  }
  stats.wroteMaterialCache = materialCache.value();
  stats.wroteAnyFiles = stats.wroteAnyFiles || stats.wroteMaterialCache;
  return Result<SceneTextureArtifactBakeStats, std::string>::makeResult(
      std::move(stats));
}

} // namespace nuri::bakery::detail
