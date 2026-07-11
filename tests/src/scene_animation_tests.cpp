#include "tests_pch.h"

#include "nuri/resources/scene_importer.h"
#include "nuri/scene/render_scene.h"
#include "nuri/scene/scene_animation_player.h"

#include <array>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

namespace {

[[nodiscard]] std::filesystem::path medievalFantasyBookPath() {
  return std::filesystem::path(PROJECT_SOURCE_DIR) / "assets" / "models" /
         "MedievalFantasyBook" / "scene.gltf";
}

[[nodiscard]] std::filesystem::path foxPath() {
  return std::filesystem::path(PROJECT_SOURCE_DIR) / "assets" / "models" /
         "Fox" / "Fox.gltf";
}

[[nodiscard]] std::filesystem::path animatedMorphCubePath() {
  return std::filesystem::path(PROJECT_SOURCE_DIR) / "assets" / "models" /
         "AnimatedMorphCube" / "AnimatedMorphCube.gltf";
}

[[nodiscard]] int findNodeByName(const nuri::ScenePrefab &prefab,
                                 std::string_view name) {
  for (uint32_t index = 0; index < prefab.nodes.size(); ++index) {
    if (std::string_view(prefab.nodes[index].name) == name) {
      return static_cast<int>(index);
    }
  }
  return -1;
}

[[nodiscard]] bool mat4Near(const glm::mat4 &lhs, const glm::mat4 &rhs,
                            float epsilon = 1.0e-5f) {
  for (uint32_t column = 0; column < 4u; ++column) {
    for (uint32_t row = 0; row < 4u; ++row) {
      if (std::abs(lhs[column][row] - rhs[column][row]) > epsilon) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] glm::vec3 translationOf(const glm::mat4 &matrix) {
  return glm::vec3(matrix[3]);
}

[[nodiscard]] nuri::AnimationClipData makeTestClip() {
  nuri::AnimationClipData clip;
  clip.name = "TestClip";
  clip.durationSeconds = 1.0f;

  clip.samplers.emplace_back();
  clip.samplers.back().interpolation = nuri::AnimationInterpolation::Linear;
  clip.samplers.back().valueArity = 3u;
  clip.samplers.back().keyTimes = {0.0f, 1.0f};
  clip.samplers.back().values = {0.0f, 0.0f, 0.0f, 4.0f, 0.0f, 0.0f};

  clip.samplers.emplace_back();
  clip.samplers.back().interpolation = nuri::AnimationInterpolation::Linear;
  clip.samplers.back().valueArity = 2u;
  clip.samplers.back().keyTimes = {0.0f, 1.0f};
  clip.samplers.back().values = {0.0f, 1.0f, 1.0f, 0.0f};

  clip.channels.push_back(nuri::AnimationChannelData{
      .samplerIndex = 0u,
      .targetNodeIndex = 0u,
      .path = nuri::AnimationTargetPath::Translation,
  });
  clip.channels.push_back(nuri::AnimationChannelData{
      .samplerIndex = 1u,
      .targetNodeIndex = 1u,
      .path = nuri::AnimationTargetPath::Weights,
  });

  return clip;
}

} // namespace

TEST(SceneAnimationTests, AnimatedMorphCubeImportsWeightAnimationAndTangents) {
  const std::filesystem::path path = animatedMorphCubePath();
  ASSERT_TRUE(std::filesystem::exists(path)) << path.string();

  auto prefabResult =
      nuri::SceneImporter::loadScenePrefabFromFile(path.string());
  ASSERT_FALSE(prefabResult.hasError()) << prefabResult.error();
  const nuri::ScenePrefab &prefab = prefabResult.value();

  ASSERT_EQ(prefab.animations.size(), 1u);
  EXPECT_EQ(prefab.animations[0].name, "Square");
  ASSERT_EQ(prefab.animations[0].channels.size(), 1u);
  const nuri::AnimationChannelData &channel = prefab.animations[0].channels[0];
  EXPECT_EQ(channel.path, nuri::AnimationTargetPath::Weights);
  ASSERT_LT(channel.targetNodeIndex, prefab.nodes.size());
  ASSERT_LT(channel.samplerIndex, prefab.animations[0].samplers.size());
  const nuri::AnimationSamplerData &sampler =
      prefab.animations[0].samplers[channel.samplerIndex];
  EXPECT_EQ(sampler.valueArity, 2u);
  EXPECT_EQ(prefab.nodes[channel.targetNodeIndex].morphWeights.size(), 2u);
  EXPECT_FLOAT_EQ(prefab.nodes[channel.targetNodeIndex].morphWeights[0], 0.0f);
  EXPECT_FLOAT_EQ(prefab.nodes[channel.targetNodeIndex].morphWeights[1], 0.0f);

  auto sceneAssetsResult =
      nuri::SceneImporter::loadSceneAssetsFromFile(path.string());
  ASSERT_FALSE(sceneAssetsResult.hasError()) << sceneAssetsResult.error();
  ASSERT_EQ(sceneAssetsResult.value().meshes.size(), 1u);
  const nuri::MeshData &mesh = sceneAssetsResult.value().meshes[0];
  ASSERT_EQ(mesh.morphTargets.size(), 2u);
  ASSERT_FALSE(mesh.submeshes.empty());
  const uint32_t morphVertexCount = mesh.submeshes[0].vertexCount;
  ASSERT_GT(morphVertexCount, 0u);
  for (const nuri::MorphTarget &target : mesh.morphTargets) {
    EXPECT_EQ(target.positionDeltas.size(), morphVertexCount);
    EXPECT_EQ(target.normalDeltas.size(), morphVertexCount);
    EXPECT_EQ(target.tangentDeltas.size(), morphVertexCount);
  }
}

TEST(SceneAnimationTests, FoxImportsSkinPayloadAndSkinIndices) {
  const std::filesystem::path path = foxPath();
  ASSERT_TRUE(std::filesystem::exists(path)) << path.string();

  auto prefabResult =
      nuri::SceneImporter::loadScenePrefabFromFile(path.string());
  ASSERT_FALSE(prefabResult.hasError()) << prefabResult.error();
  const nuri::ScenePrefab &prefab = prefabResult.value();

  ASSERT_EQ(prefab.skins.size(), 1u);
  ASSERT_EQ(prefab.skins[0].jointNodeIndices.size(), 24u);
  ASSERT_EQ(prefab.skins[0].inverseBindMatrices.size(), 24u);
  ASSERT_FALSE(prefab.renderables.empty());
  EXPECT_EQ(prefab.renderables[0].skinIndex, 0u);
}

TEST(SceneAnimationTests, PlayerSamplesNodeTransformsAndMorphWeights) {
  nuri::ScenePrefab prefab;
  prefab.nodes.resize(2u);
  prefab.nodes[0].name = "AnimatedRoot";
  prefab.nodes[1].name = "MorphNode";
  prefab.nodes[1].parentIndex = 0u;
  prefab.nodes[1].morphWeights = {0.0f, 1.0f};
  prefab.meshAssets.push_back(nuri::ScenePrefabMeshAssetRef{
      .sourceSceneMeshIndex = 0u, .sourceName = "mesh"});
  prefab.materialAssets.push_back(nuri::ScenePrefabMaterialAssetRef{
      .sourceMaterialIndex = 0u,
      .sourceName = "material",
  });
  prefab.renderables.push_back(nuri::ScenePrefabRenderable{
      .nodeIndex = 1u,
      .meshIndex = 0u,
      .materialIndex = 0u,
  });
  prefab.animations.push_back(makeTestClip());

  nuri::ScenePrefabAssets assets;
  assets.models.push_back(nuri::makeModelRef(1u, 1u));
  assets.materials.push_back(nuri::makeMaterialRef(2u, 1u));

  nuri::RenderScene scene;
  nuri::SceneInstantiationMap instantiation;
  auto instantiateResult = scene.graph().instantiatePrefab(
      prefab, scene.graph().rootNode(), assets, &instantiation);
  ASSERT_FALSE(instantiateResult.hasError()) << instantiateResult.error();

  nuri::SceneAnimationPlayer player(prefab, instantiation);
  auto playResult = player.play(0u, nuri::AnimationPlaybackMode::Loop);
  ASSERT_FALSE(playResult.hasError()) << playResult.error();

  player.seek(0.5f);
  player.update(scene.graph(), 0.0f);

  glm::mat4 nodeLocal(1.0f);
  ASSERT_TRUE(
      scene.graph().getNodeLocalTransform(instantiation.nodes[0], nodeLocal));
  EXPECT_NEAR(translationOf(nodeLocal).x, 2.0f, 1.0e-5f);

  const std::span<const float> morphWeights =
      scene.graph().getRenderableMorphWeights(instantiation.renderables[0]);
  ASSERT_EQ(morphWeights.size(), 2u);
  EXPECT_NEAR(morphWeights[0], 0.5f, 1.0e-5f);
  EXPECT_NEAR(morphWeights[1], 0.5f, 1.0e-5f);

  auto commitResult = scene.commit();
  ASSERT_FALSE(commitResult.hasError()) << commitResult.error();
  ASSERT_EQ(scene.renderables().size(), 1u);
  ASSERT_EQ(scene.renderables()[0].morphWeights.size(), 2u);
  EXPECT_NEAR(scene.renderables()[0].morphWeights[0], 0.5f, 1.0e-5f);
  EXPECT_NEAR(scene.renderables()[0].morphWeights[1], 0.5f, 1.0e-5f);

  player.stop();
  player.update(scene.graph(), 1.0f);
  const std::span<const float> stoppedWeights =
      scene.graph().getRenderableMorphWeights(instantiation.renderables[0]);
  ASSERT_EQ(stoppedWeights.size(), 2u);
  EXPECT_NEAR(stoppedWeights[0], 0.5f, 1.0e-5f);
  EXPECT_NEAR(stoppedWeights[1], 0.5f, 1.0e-5f);
}

TEST(SceneAnimationTests, PlayerUsesShortestPathForLinearQuaternionRotation) {
  nuri::ScenePrefab prefab;
  prefab.nodes.resize(1u);
  prefab.nodes[0].name = "RotatingNode";
  prefab.animations.emplace_back();
  nuri::AnimationClipData &clip = prefab.animations.back();
  clip.name = "Rotate";
  clip.durationSeconds = 1.0f;
  clip.samplers.emplace_back();
  clip.samplers.back().interpolation = nuri::AnimationInterpolation::Linear;
  clip.samplers.back().valueArity = 4u;
  clip.samplers.back().keyTimes = {0.0f, 1.0f};
  clip.samplers.back().values = {
      0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, -0.70710678f, -0.70710678f,
  };
  clip.channels.push_back(nuri::AnimationChannelData{
      .samplerIndex = 0u,
      .targetNodeIndex = 0u,
      .path = nuri::AnimationTargetPath::Rotation,
  });

  nuri::RenderScene scene;
  nuri::ScenePrefabAssets assets;
  nuri::SceneInstantiationMap instantiation;
  auto instantiateResult = scene.graph().instantiatePrefab(
      prefab, scene.graph().rootNode(), assets, &instantiation);
  ASSERT_FALSE(instantiateResult.hasError()) << instantiateResult.error();

  nuri::SceneAnimationPlayer player(prefab, instantiation);
  auto playResult = player.play(0u, nuri::AnimationPlaybackMode::Loop);
  ASSERT_FALSE(playResult.hasError()) << playResult.error();

  player.seek(0.5f);
  player.update(scene.graph(), 0.0f);

  glm::mat4 nodeLocal(1.0f);
  ASSERT_TRUE(
      scene.graph().getNodeLocalTransform(instantiation.nodes[0], nodeLocal));
  const glm::vec3 rotatedXAxis =
      glm::normalize(glm::vec3(nodeLocal * glm::vec4(1.0f, 0.0f, 0.0f, 0.0f)));
  EXPECT_NEAR(rotatedXAxis.x, 0.70710678f, 1.0e-4f);
  EXPECT_NEAR(rotatedXAxis.y, 0.70710678f, 1.0e-4f);
  EXPECT_NEAR(rotatedXAxis.z, 0.0f, 1.0e-4f);
}

TEST(SceneAnimationTests, PlayerBuildsSkinPaletteFromJointWorldTransforms) {
  nuri::ScenePrefab prefab;
  prefab.nodes.resize(2u);
  prefab.nodes[0].name = "SkinnedNode";
  prefab.nodes[0].localFromParent =
      glm::translate(glm::mat4(1.0f), glm::vec3(3.0f, 0.0f, 0.0f));
  prefab.nodes[1].name = "JointNode";
  prefab.nodes[1].localFromParent =
      glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, 0.0f));

  prefab.meshAssets.push_back(nuri::ScenePrefabMeshAssetRef{
      .sourceSceneMeshIndex = 0u, .sourceName = "mesh"});
  prefab.materialAssets.push_back(nuri::ScenePrefabMaterialAssetRef{
      .sourceMaterialIndex = 0u,
      .sourceName = "material",
  });
  prefab.renderables.push_back(nuri::ScenePrefabRenderable{
      .nodeIndex = 0u,
      .meshIndex = 0u,
      .materialIndex = 0u,
      .skinIndex = 0u,
  });
  prefab.skins.emplace_back();
  prefab.skins[0].jointNodeIndices = {1u};
  prefab.skins[0].inverseBindMatrices = {glm::mat4(1.0f)};
  prefab.animations.emplace_back();
  prefab.animations[0].name = "Idle";

  nuri::ScenePrefabAssets assets;
  assets.models.push_back(nuri::makeModelRef(1u, 1u));
  assets.materials.push_back(nuri::makeMaterialRef(2u, 1u));

  nuri::RenderScene scene;
  nuri::SceneInstantiationMap instantiation;
  auto instantiateResult = scene.graph().instantiatePrefab(
      prefab, scene.graph().rootNode(), assets, &instantiation);
  ASSERT_FALSE(instantiateResult.hasError()) << instantiateResult.error();

  nuri::SceneAnimationPlayer player(prefab, instantiation);
  auto playResult = player.play(0u, nuri::AnimationPlaybackMode::Once);
  ASSERT_FALSE(playResult.hasError()) << playResult.error();
  player.update(scene.graph(), 0.0f);

  const std::span<const glm::mat4> palette =
      scene.graph().getRenderableSkinPalette(instantiation.renderables[0]);
  ASSERT_EQ(palette.size(), 1u);

  glm::mat4 expectedJointWorld(1.0f);
  (void)scene.graph().syncWorldTransforms();
  ASSERT_TRUE(scene.graph().getCachedNodeWorldTransform(instantiation.nodes[1],
                                                        expectedJointWorld));
  EXPECT_TRUE(mat4Near(palette[0], expectedJointWorld));
}
