#include "nuri/pch.h"

#include "nuri/resources/gltf_scene_importer.h"

#include "nuri/core/profiling.h"
#include "nuri/math/light.h"
#include "nuri/resources/detail/gltf_json_utils.h"
#include "nuri/resources/scene_importer.h"

namespace nuri {
namespace {

glm::mat4 computeWorldMatrix(uint32_t nodeIndex, const ImportedScene &scene,
                             std::span<glm::mat4> cache,
                             std::span<uint8_t> state) {
  if (nodeIndex >= scene.nodes.size()) {
    return glm::mat4(1.0f);
  }
  if (state[nodeIndex] == 2u) {
    return cache[nodeIndex];
  }
  if (state[nodeIndex] == 1u) {
    return glm::mat4(1.0f);
  }

  state[nodeIndex] = 1u;
  const ImportedSceneNode &node = scene.nodes[nodeIndex];
  glm::mat4 world = node.localFromParent;
  if (node.parentIndex != kInvalidScenePrefabIndex) {
    world = computeWorldMatrix(node.parentIndex, scene, cache, state) * world;
  }
  cache[nodeIndex] = world;
  state[nodeIndex] = 2u;
  return world;
}

} // namespace

Result<ImportedLightSet, std::string>
GltfSceneImporter::loadLightsFromFile(std::string_view path) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!detail::isGltfJsonAssetPath(path)) {
    return Result<ImportedLightSet, std::string>::makeResult({});
  }

  auto sceneResult = SceneImporter::loadSceneFromFile(path);
  if (sceneResult.hasError()) {
    return Result<ImportedLightSet, std::string>::makeError(
        sceneResult.error());
  }
  const ImportedScene &scene = sceneResult.value();
  if (scene.lights.empty()) {
    return Result<ImportedLightSet, std::string>::makeResult({});
  }

  ImportedLightSet lights;
  lights.reserve(scene.lights.size());
  std::vector<glm::mat4> worldCache(scene.nodes.size(), glm::mat4(1.0f));
  std::vector<uint8_t> state(scene.nodes.size(), 0u);
  for (const ImportedSceneLight &sceneLight : scene.lights) {
    ImportedLightInfo imported{};
    imported.desc = sceneLight.light;
    const glm::mat4 worldMatrix =
        computeWorldMatrix(sceneLight.nodeIndex, scene, worldCache, state);
    imported.desc.position = glm::vec3(worldMatrix[3]);
    imported.desc.rotation = rotationFromMatrixOrIdentity(worldMatrix);
    if (!sceneLight.sourceName.empty()) {
      imported.desc.name = sceneLight.sourceName;
      imported.sourceName = sceneLight.sourceName;
    }
    imported.sourceNodeIndex = sceneLight.sourceNodeIndex;
    lights.push_back(std::move(imported));
  }

  return Result<ImportedLightSet, std::string>::makeResult(std::move(lights));
}

Result<ScenePrefab, std::string>
GltfSceneImporter::loadScenePrefabFromFile(std::string_view path,
                                           std::pmr::memory_resource *memory) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!detail::isGltfJsonAssetPath(path)) {
    return Result<ScenePrefab, std::string>::makeError(
        "GltfSceneImporter::loadScenePrefabFromFile: path is not .gltf/.glb");
  }
  return SceneImporter::loadScenePrefabFromFile(path, {}, memory);
}

} // namespace nuri
