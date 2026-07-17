#include "nuri/pch.h"

#include "nuri/resources/async/scene_asset_preparation.h"

#include "nuri/core/profiling.h"
#include "nuri/resources/mesh_importer.h"
#include "nuri/resources/storage/texture/material_texture_artifact.h"
#include "nuri/resources/storage/texture/texture_artifact_builder.h"
#include "nuri/resources/storage/texture/texture_cache_utils.h"

#include <array>
#include <filesystem>
#include <mutex>

namespace nuri {
namespace {

constexpr std::array<std::string_view, kMaterialTextureSlotCount>
    kTextureDebugSuffixes{
        "base_color",          "metallic_roughness", "normal",
        "occlusion",           "emissive",           "clearcoat",
        "clearcoat_roughness", "clearcoat_normal",   "specular",
        "specular_color",      "sheen_color",        "sheen_roughness",
        "transmission",        "thickness",
    };

[[nodiscard]] bool hasExtensionCaseInsensitive(std::string_view path,
                                               std::string_view extension) {
  if (path.size() < extension.size()) {
    return false;
  }
  const size_t offset = path.size() - extension.size();
  for (size_t index = 0u; index < extension.size(); ++index) {
    const char lhs = static_cast<char>(
        std::tolower(static_cast<unsigned char>(path[offset + index])));
    const char rhs = static_cast<char>(
        std::tolower(static_cast<unsigned char>(extension[index])));
    if (lhs != rhs) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] TextureRequestKind
requestKindForPath(std::string_view path) noexcept {
  return hasExtensionCaseInsensitive(path, ".ktx2") ||
                 hasExtensionCaseInsensitive(path, ".ktx")
             ? TextureRequestKind::Ktx2Texture2D
             : TextureRequestKind::Texture2D;
}

[[nodiscard]] std::string makeTextureDebugName(std::string_view prefix,
                                               std::string_view suffix,
                                               uint32_t materialIndex) {
  return std::string(prefix) + "_" + std::string(suffix) + "_" +
         std::to_string(materialIndex);
}

std::array<std::mutex, 64u> gArtifactBuildMutexes{};

[[nodiscard]] std::mutex &artifactBuildMutex(uint64_t identity) noexcept {
  return gArtifactBuildMutexes[identity % gArtifactBuildMutexes.size()];
}

} // namespace

Result<PreparedSceneManifest, std::string>
prepareSceneManifest(std::string_view path, const SceneImportOptions &options) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  SceneImportOptions manifestOptions = options;
  manifestOptions.adaptAssetSources = true;
  auto imported = SceneImporter::loadSceneFromFile(path, manifestOptions);
  if (imported.hasError()) {
    return Result<PreparedSceneManifest, std::string>::makeError(
        imported.error());
  }
  auto prefab = SceneImporter::buildScenePrefab(imported.value());
  if (prefab.hasError()) {
    return Result<PreparedSceneManifest, std::string>::makeError(
        prefab.error());
  }
  PreparedSceneManifest manifest{
      .prefab = std::move(prefab.value()),
  };
  manifest.meshes.reserve(imported.value().adaptedMeshes.size());
  for (AdaptedSceneMesh &mesh : imported.value().adaptedMeshes) {
    manifest.meshes.push_back(std::move(mesh));
  }
  manifest.embeddedTextures =
      std::make_shared<const std::vector<EmbeddedSceneTextureData>>(
          std::move(imported.value().embeddedTextures));
  manifest.materials.reserve(manifest.prefab.materialAssets.size());
  for (const ScenePrefabMaterialAssetRef &asset :
       manifest.prefab.materialAssets) {
    if (asset.sourceMaterialIndex <
        imported.value().adaptedMaterials.materials.size()) {
      manifest.materials.push_back(
          imported.value()
              .adaptedMaterials.materials[asset.sourceMaterialIndex]);
      continue;
    }
    MaterialData fallback{};
    fallback.name = asset.sourceName;
    manifest.materials.push_back(std::move(fallback));
  }
  return Result<PreparedSceneManifest, std::string>::makeResult(
      std::move(manifest));
}

Result<PreparedImportedMaterial, std::string> prepareImportedMaterial(
    const MaterialData &material, std::string_view scenePath,
    uint32_t sourceMaterialIndex, const TextureCompressionCaps &compressionCaps,
    std::span<const EmbeddedSceneTextureData> embeddedTextures,
    std::string_view debugNamePrefix) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (scenePath.empty()) {
    return Result<PreparedImportedMaterial, std::string>::makeError(
        "prepareImportedMaterial: scene path is empty");
  }

  const std::string canonicalScenePath = canonicalizeResourcePath(scenePath);
  PreparedImportedMaterial prepared{
      .desc = Material::descFromImported(material),
      .debugName = material.name.empty()
                       ? std::string(debugNamePrefix) + "_material_" +
                             std::to_string(sourceMaterialIndex)
                       : std::string(debugNamePrefix) + "_" + material.name,
      .sourceIdentity =
          canonicalScenePath + "#" + std::to_string(sourceMaterialIndex),
  };

  auto builderResult = SceneTextureArtifactBuilder::create(
      std::filesystem::path(scenePath), embeddedTextures);
  if (builderResult.hasError()) {
    return Result<PreparedImportedMaterial, std::string>::makeError(
        builderResult.error());
  }
  SceneTextureArtifactBuilder builder = std::move(builderResult.value());

  for (size_t slotIndex = 0u; slotIndex < kMaterialTextureArtifactSpecs.size();
       ++slotIndex) {
    const MaterialTextureArtifactSpec &spec =
        kMaterialTextureArtifactSpecs[slotIndex];
    const MaterialTextureSlotData &source = material.*(spec.slot);
    if (source.sourceKind == MaterialTextureSourceKind::None) {
      continue;
    }

    const TextureArtifactBuildOptions buildOptions =
        makeMaterialTextureArtifactBuildOptions(material, slotIndex);
    const std::string debugName = makeTextureDebugName(
        debugNamePrefix, kTextureDebugSuffixes[slotIndex], sourceMaterialIndex);
    if (source.sourceKind == MaterialTextureSourceKind::ExternalFile &&
        hasExtensionCaseInsensitive(source.path, ".dds")) {
      prepared.textures[slotIndex] = TextureRequest{
          .path = source.path,
          .loadOptions = buildOptions.loadOptions,
          .kind = TextureRequestKind::Texture2D,
          .debugName = debugName,
      };
      continue;
    }

    const uint64_t identity = hashSceneTextureSourceIdentity(
        canonicalScenePath, source, buildOptions.loadOptions.srgb,
        textureArtifactProcessingTag(buildOptions));
    const Format targetFormat = selectTextureArtifactTargetFormat(
        compressionCaps.bc7, compressionCaps.etc2,
        buildOptions.loadOptions.srgb, 4u);
    Result<TextureArtifactBuildResult, std::string> artifact =
        Result<TextureArtifactBuildResult, std::string>::makeError(
            "uninitialized");
    {
      std::lock_guard lock(artifactBuildMutex(identity));
      artifact =
          builder.ensure(source, identity, targetFormat, buildOptions, false);
    }
    if (artifact.hasError()) {
      prepared.optionalTextureErrors.push_back(
          std::string(kTextureDebugSuffixes[slotIndex]) + ": " +
          artifact.error());
      continue;
    }
    prepared.textures[slotIndex] = TextureRequest{
        .path = artifact.value().artifactPath.string(),
        .kind = TextureRequestKind::Ktx2Texture2D,
        .debugName = debugName,
    };
  }

  return Result<PreparedImportedMaterial, std::string>::makeResult(
      std::move(prepared));
}

} // namespace nuri
