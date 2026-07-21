#include "tests_pch.h"

#include "nuri/scene/render_scene.h"

#include <array>

namespace {

[[nodiscard]] glm::mat4 translate(const glm::vec3 &value) {
  return glm::translate(glm::mat4(1.0f), value);
}

TEST(DDGIVolumeStoreTests, RejectsInvalidDescriptorsWithoutAllocatingAHandle) {
  nuri::RenderScene scene;
  nuri::DDGIVolumeDesc desc{};
  desc.probeCounts.x = 1u;

  const auto result = scene.graph().addDDGIVolume(scene.graph().rootNode(), desc);

  EXPECT_TRUE(result.hasError());
  uint32_t volumeCount = 0u;
  scene.graph().forEachDDGIVolumeId(
      [&volumeCount](nuri::DDGIVolumeId) { ++volumeCount; });
  EXPECT_EQ(volumeCount, 0u);
}

TEST(DDGIVolumeStoreTests, RemovedHandleIsStaleAndSlotGenerationAdvances) {
  nuri::RenderScene scene;
  nuri::SceneGraph &graph = scene.graph();
  nuri::DDGIVolumeDesc desc{};
  desc.name = "First";

  const auto firstResult = graph.addDDGIVolume(graph.rootNode(), desc);
  ASSERT_FALSE(firstResult.hasError()) << firstResult.error();
  const nuri::DDGIVolumeId first = firstResult.value();
  ASSERT_TRUE(graph.removeDDGIVolume(first));
  EXPECT_FALSE(graph.getDDGIVolume(first, desc));

  desc.name = "Second";
  const auto secondResult = graph.addDDGIVolume(graph.rootNode(), desc);
  ASSERT_FALSE(secondResult.hasError()) << secondResult.error();
  const nuri::DDGIVolumeId second = secondResult.value();

  EXPECT_EQ(nuri::indexOf(first), nuri::indexOf(second));
  EXPECT_NE(nuri::generationOf(first), nuri::generationOf(second));
}

TEST(DDGIVolumeStoreTests, DestroyingNodeSubtreeRemovesAttachedVolumes) {
  nuri::RenderScene scene;
  nuri::SceneGraph &graph = scene.graph();
  const auto parent = graph.createNode(graph.rootNode(), "Parent");
  ASSERT_FALSE(parent.hasError()) << parent.error();
  const auto child = graph.createNode(parent.value(), "Child");
  ASSERT_FALSE(child.hasError()) << child.error();
  const auto volume = graph.addDDGIVolume(child.value(), {});
  ASSERT_FALSE(volume.hasError()) << volume.error();

  ASSERT_TRUE(graph.destroyNodeSubtree(parent.value()));

  nuri::DDGIVolumeDesc stored{};
  EXPECT_FALSE(graph.getDDGIVolume(volume.value(), stored));
  uint32_t volumeCount = 0u;
  graph.forEachDDGIVolumeId(
      [&volumeCount](nuri::DDGIVolumeId) { ++volumeCount; });
  EXPECT_EQ(volumeCount, 0u);
}

TEST(DDGIVolumeStoreTests, CommitPublishesEnabledVolumesInStablePriorityOrder) {
  nuri::RenderScene scene;
  nuri::SceneGraph &graph = scene.graph();
  constexpr std::array<int32_t, 6u> kPriorities = {1, 5, 5, -2, 7, 100};
  std::array<nuri::DDGIVolumeId, kPriorities.size()> ids{};
  for (size_t index = 0u; index < kPriorities.size(); ++index) {
    nuri::DDGIVolumeDesc desc{};
    desc.name = "Volume " + std::to_string(index);
    desc.priority = kPriorities[index];
    desc.enabled = index != kPriorities.size() - 1u;
    const auto added = graph.addDDGIVolume(graph.rootNode(), desc);
    ASSERT_FALSE(added.hasError()) << added.error();
    ids[index] = added.value();
  }

  const auto commit = scene.commit();
  ASSERT_FALSE(commit.hasError()) << commit.error();
  ASSERT_EQ(scene.ddgiVolumes().size(), 5u);
  EXPECT_EQ(scene.ddgiVolumes()[0].id, ids[4]);
  EXPECT_EQ(scene.ddgiVolumes()[1].id, ids[1]);
  EXPECT_EQ(scene.ddgiVolumes()[2].id, ids[2]);
  EXPECT_EQ(scene.ddgiVolumes()[3].id, ids[0]);
  EXPECT_EQ(scene.ddgiVolumes()[4].id, ids[3]);
  EXPECT_EQ(scene.ddgiVolumeTopologyVersion(), 1u);
  EXPECT_EQ(scene.ddgiVolumeTransformVersion(), 1u);
  EXPECT_EQ(scene.ddgiVolumeSettingsVersion(), 1u);
}

TEST(DDGIVolumeStoreTests, TransformAndSettingsVersionsAdvanceIndependently) {
  nuri::RenderScene scene;
  nuri::SceneGraph &graph = scene.graph();
  const auto node = graph.createNode(graph.rootNode(), "Volume");
  ASSERT_FALSE(node.hasError()) << node.error();
  const auto unrelatedNode = graph.createNode(graph.rootNode(), "Unrelated");
  ASSERT_FALSE(unrelatedNode.hasError()) << unrelatedNode.error();
  nuri::DDGIVolumeDesc desc{};
  const auto volume = graph.addDDGIVolume(node.value(), desc);
  ASSERT_FALSE(volume.hasError()) << volume.error();
  auto commit = scene.commit();
  ASSERT_FALSE(commit.hasError()) << commit.error();
  ASSERT_EQ(scene.ddgiVolumeTopologyVersion(), 1u);
  ASSERT_EQ(scene.ddgiVolumeTransformVersion(), 1u);
  ASSERT_EQ(scene.ddgiVolumeSettingsVersion(), 1u);

  // This ownership rule is complete and deterministic at the SceneGraph seam;
  // a renderer scenario observes only the downstream history-reset symptom.
  ASSERT_TRUE(graph.setNodeLocalTransform(
      unrelatedNode.value(), translate(glm::vec3(-1.0f, 0.0f, 0.0f))));
  commit = scene.commit();
  ASSERT_FALSE(commit.hasError()) << commit.error();
  EXPECT_EQ(scene.ddgiVolumeTopologyVersion(), 1u);
  EXPECT_EQ(scene.ddgiVolumeTransformVersion(), 1u);
  EXPECT_EQ(scene.ddgiVolumeSettingsVersion(), 1u);

  ASSERT_TRUE(graph.setNodeLocalTransform(
      node.value(), translate(glm::vec3(2.0f, 3.0f, 4.0f))));
  commit = scene.commit();
  ASSERT_FALSE(commit.hasError()) << commit.error();
  EXPECT_EQ(scene.ddgiVolumeTopologyVersion(), 1u);
  EXPECT_EQ(scene.ddgiVolumeTransformVersion(), 2u);
  EXPECT_EQ(scene.ddgiVolumeSettingsVersion(), 1u);

  desc.priority = 42;
  ASSERT_TRUE(graph.updateDDGIVolume(volume.value(), desc));
  commit = scene.commit();
  ASSERT_FALSE(commit.hasError()) << commit.error();
  EXPECT_EQ(scene.ddgiVolumeTopologyVersion(), 1u);
  EXPECT_EQ(scene.ddgiVolumeTransformVersion(), 2u);
  EXPECT_EQ(scene.ddgiVolumeSettingsVersion(), 2u);
  ASSERT_EQ(scene.ddgiVolumes().size(), 1u);
  EXPECT_EQ(scene.ddgiVolumes()[0].priority, 42);
}

TEST(DDGIVolumeStoreTests, ClearingVolumesInvalidatesHandlesAndAllSnapshots) {
  nuri::RenderScene scene;
  nuri::SceneGraph &graph = scene.graph();
  const auto volume = graph.addDDGIVolume(graph.rootNode(), {});
  ASSERT_FALSE(volume.hasError()) << volume.error();
  auto commit = scene.commit();
  ASSERT_FALSE(commit.hasError()) << commit.error();

  graph.clearDDGIVolumes();
  commit = scene.commit();
  ASSERT_FALSE(commit.hasError()) << commit.error();

  nuri::DDGIVolumeDesc stored{};
  EXPECT_FALSE(graph.getDDGIVolume(volume.value(), stored));
  EXPECT_TRUE(scene.ddgiVolumes().empty());
  EXPECT_EQ(scene.ddgiVolumeTopologyVersion(), 2u);
  EXPECT_EQ(scene.ddgiVolumeTransformVersion(), 2u);
  EXPECT_EQ(scene.ddgiVolumeSettingsVersion(), 2u);
}

} // namespace
