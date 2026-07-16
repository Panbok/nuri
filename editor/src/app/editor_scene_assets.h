#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "nuri/app/editor_runtime.h"

namespace nuri {

[[nodiscard]] Result<void, std::string> prepareImportedPrefabSceneResources(
    EditorRuntime &runtime, std::string_view sceneName,
    const std::filesystem::path &modelPath,
    const MeshImportOptions &importOptions, ImportedPrefabSceneResources &out);

[[nodiscard]] Result<void, std::string> prepareSimpleImportedModelSceneAssets(
    EditorRuntime &runtime, std::string_view sceneName,
    const std::filesystem::path &modelPath,
    const MeshImportOptions &importOptions, std::string_view modelDebugName,
    std::string_view importedMaterialPrefix,
    std::string_view fallbackMaterialDebugName, bool loadImportedLights,
    SimpleModelSceneAssets &out);

[[nodiscard]] MaterialRef acquireImportedMaterialOrFallback(
    ResourceManager &resources, std::string_view sceneName,
    std::string_view modelPath, ModelRef modelRef,
    std::string_view debugNamePrefix, std::string_view fallbackDebugName);

void loadImportedLightsForScene(std::string_view sceneName,
                                std::string_view modelPath,
                                std::vector<ImportedSceneLight> &outLights);

void queueSceneTextureArtifactBakeIfNeeded(
    EditorRuntime &runtime, ImportedPrefabSceneResources &assets);
void queueSceneTextureArtifactBakeIfNeeded(EditorRuntime &runtime,
                                           SimpleModelSceneAssets &assets);
void queueSceneTextureArtifactBakeIfNeeded(EditorRuntime &runtime,
                                           StreamingSceneState &assets);

} // namespace nuri
