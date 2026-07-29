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
[[nodiscard]] Result<bool, std::string>
refreshImportedPrefabSceneResources(EditorRuntime &runtime,
                                    std::string_view sceneName,
                                    ImportedPrefabSceneResources &out);

void loadImportedLightsForScene(std::string_view sceneName,
                                std::string_view modelPath,
                                std::vector<ScenePrefabLight> &outLights);

void queueSceneTextureArtifactBakeIfNeeded(
    EditorRuntime &runtime, ImportedPrefabSceneResources &assets);
void queueSceneTextureArtifactBakeIfNeeded(EditorRuntime &runtime,
                                           SimpleModelSceneAssets &assets);
void queueSceneTextureArtifactBakeIfNeeded(EditorRuntime &runtime,
                                           StreamingSceneState &assets);

} // namespace nuri
