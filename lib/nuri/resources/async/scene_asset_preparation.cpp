#include "nuri/resources/async/scene_asset_preparation.h"
#include "nuri/core/profiling.h"
#include "nuri/resources/mesh_importer.h"
#include "nuri/resources/storage/cache_utils.h"
#include "nuri/resources/storage/texture/texture_artifact_builder.h"
#include "nuri/resources/storage/texture/texture_cache_utils.h"
#include <array>
#include <filesystem>
#include <mutex>
namespace nuri {
namespace {
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
  ScenePrefab &scene = imported.value();
  PreparedSceneManifest manifest{};
  manifest.meshes.assign(std::make_move_iterator(scene.adaptedMeshes.begin()),
                         std::make_move_iterator(scene.adaptedMeshes.end()));
  manifest.embeddedTextures =
      std::make_shared<const std::vector<EmbeddedSceneTextureData>>(
          std::move(scene.embeddedTextures));
  manifest.materials.reserve(scene.materialAssets.size());
  for (const ScenePrefabAssetRef &asset : scene.materialAssets)
    manifest.materials.push_back(
        scene.adaptedMaterials.materials[asset.sourceIndex]);
  manifest.prefab = std::move(scene);
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
  for (const MaterialTextureSlotDesc &slotDesc : kMaterialTextureSlotDescs) {
    const size_t slotIndex = static_cast<size_t>(slotDesc.slot);
    const MaterialTextureSlotData &source = material.textures[slotIndex];
    if (source.sourceKind == MaterialTextureSourceKind::None) {
      continue;
    }
    const TextureArtifactBuildOptions buildOptions =
        materialTextureArtifactBuildOptions(material, slotDesc.slot);
    const std::string debugName = makeTextureDebugName(
        debugNamePrefix, kMaterialTextureSlotDescs[slotIndex].debugToken,
        sourceMaterialIndex);
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
          std::string(kMaterialTextureSlotDescs[slotIndex].debugToken) + ": " +
          artifact.error());
      continue;
    }
    prepared.textures[slotIndex] = TextureRequest{
        .path = artifact.value().artifactPath.string(),
        .loadOptions = buildOptions.loadOptions,
        .contentContract = buildOptions.contentContract,
        .kind = TextureRequestKind::Ktx2Texture2D,
        .debugName = debugName,
    };
  }
  return Result<PreparedImportedMaterial, std::string>::makeResult(
      std::move(prepared));
}

} // namespace nuri
