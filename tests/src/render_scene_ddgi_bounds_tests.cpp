#include "tests_pch.h"

#include "render_graph_test_support.h"

#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/scene/render_scene.h"

#include <array>

namespace {

using namespace nuri;
using namespace nuri::test_support;

[[nodiscard]] MeshData makeCoverageTriangle(std::pmr::memory_resource *memory) {
  MeshData mesh(memory);
  mesh.vertices.resize(3u);
  mesh.vertices[0].position = glm::vec3(-0.5f, -0.5f, 0.0f);
  mesh.vertices[1].position = glm::vec3(0.5f, -0.5f, 0.0f);
  mesh.vertices[2].position = glm::vec3(0.0f, 0.5f, 0.0f);
  for (Vertex &vertex : mesh.vertices) {
    vertex.normal = glm::vec3(0.0f, 0.0f, 1.0f);
  }
  mesh.indices = {0u, 1u, 2u};
  Submesh submesh{};
  submesh.vertexCount = 3u;
  submesh.indexCount = 3u;
  submesh.bounds =
      BoundingBox(glm::vec3(-0.5f, -0.5f, 0.0f), glm::vec3(0.5f, 0.5f, 0.0f));
  submesh.lodCount = 1u;
  submesh.lods[0] = SubmeshLod{.indexCount = 3u};
  mesh.submeshes.push_back(submesh);
  return mesh;
}

[[nodiscard]] MaterialRequest materialRequest(MaterialAlphaMode alphaMode,
                                              bool transmission = false) {
  MaterialRequest request{};
  request.desc.alphaMode = alphaMode;
  if (transmission) {
    request.desc.featureMask |= kMaterialFeatureTransmission;
    request.desc.transmissionFactor = 1.0f;
  }
  return request;
}

[[nodiscard]] NodeId addRenderable(RenderScene &scene, ModelRef model,
                                   MaterialRef material, float x,
                                   MaterialRef overrideMaterial = {}) {
  auto node = scene.graph().createNode(
      scene.graph().rootNode(), "coverage_renderable",
      glm::translate(glm::mat4(1.0f), glm::vec3(x, 0.0f, 0.0f)));
  EXPECT_FALSE(node.hasError()) << node.error();
  if (node.hasError()) {
    return kInvalidNodeId;
  }
  auto renderable = scene.graph().addRenderable(*node, model, material);
  EXPECT_FALSE(renderable.hasError()) << renderable.error();
  if (!renderable.hasError() && isValid(overrideMaterial)) {
    EXPECT_TRUE(scene.graph().setRenderableMaterialOverride(*renderable,
                                                            overrideMaterial));
  }
  return *node;
}

void expectBounds(const DDGISceneCoverageBounds &bounds, glm::vec3 minimum,
                  glm::vec3 maximum) {
  ASSERT_TRUE(bounds.valid);
  EXPECT_TRUE(bounds.complete);
  EXPECT_EQ(bounds.bounds.min_, minimum);
  EXPECT_EQ(bounds.bounds.max_, maximum);
}

TEST(RenderSceneDDGIBoundsTests,
     PublishesOnlyOpaqueAndAlphaMaskedCanonicalBounds) {
  std::array<std::byte, 256u * 1024u> storage{};
  std::pmr::monotonic_buffer_resource memory(storage.data(), storage.size());
  FakeGPUDeviceBase gpu;
  ResourceManager resources(gpu, &memory);
  RenderScene scene(&memory);
  scene.bindResources(&resources);

  auto model = resources.acquireGeneratedModel(makeCoverageTriangle(&memory),
                                               "coverage_triangle");
  ASSERT_FALSE(model.hasError()) << model.error();
  auto opaque =
      resources.acquireMaterial(materialRequest(MaterialAlphaMode::Opaque));
  auto masked =
      resources.acquireMaterial(materialRequest(MaterialAlphaMode::Mask));
  auto blended =
      resources.acquireMaterial(materialRequest(MaterialAlphaMode::Blend));
  auto transmission =
      resources.acquireMaterial(materialRequest(MaterialAlphaMode::Mask, true));
  ASSERT_FALSE(opaque.hasError()) << opaque.error();
  ASSERT_FALSE(masked.hasError()) << masked.error();
  ASSERT_FALSE(blended.hasError()) << blended.error();
  ASSERT_FALSE(transmission.hasError()) << transmission.error();

  (void)addRenderable(scene, *model, *opaque, 0.0f);
  (void)addRenderable(scene, *model, *masked, 10.0f);
  (void)addRenderable(scene, *model, *blended, 100.0f);
  (void)addRenderable(scene, *model, *transmission, 200.0f);
  (void)addRenderable(scene, *model, *opaque, 300.0f, *blended);

  auto commit = scene.commit();
  ASSERT_FALSE(commit.hasError()) << commit.error();
  expectBounds(scene.ddgiCurrentCoverageBounds(), glm::vec3(-0.5f, -0.5f, 0.0f),
               glm::vec3(10.5f, 0.5f, 0.0f));
  EXPECT_FALSE(scene.ddgiStaticCoverageBounds().valid);
}

TEST(RenderSceneDDGIBoundsTests,
     ActivationSnapshotIsExplicitImmutableAndRefittable) {
  std::array<std::byte, 256u * 1024u> storage{};
  std::pmr::monotonic_buffer_resource memory(storage.data(), storage.size());
  FakeGPUDeviceBase gpu;
  ResourceManager resources(gpu, &memory);
  RenderScene scene(&memory);
  scene.bindResources(&resources);

  auto model = resources.acquireGeneratedModel(makeCoverageTriangle(&memory),
                                               "coverage_triangle");
  auto material =
      resources.acquireMaterial(materialRequest(MaterialAlphaMode::Opaque));
  ASSERT_FALSE(model.hasError()) << model.error();
  ASSERT_FALSE(material.hasError()) << material.error();
  const NodeId node = addRenderable(scene, *model, *material, 0.0f);
  ASSERT_TRUE(isValid(node));
  auto commit = scene.commit();
  ASSERT_FALSE(commit.hasError()) << commit.error();
  EXPECT_TRUE(
      scene.stageDDGIStaticCoverageBounds(scene.ddgiCurrentCoverageBounds()));

  EXPECT_FALSE(scene.ddgiActivationCoverageBoundsSealed());
  EXPECT_TRUE(scene.sealDDGIActivationCoverageBounds());
  const DDGISceneCoverageBounds activation =
      scene.ddgiActivationCoverageBounds();
  expectBounds(activation, glm::vec3(-0.5f, -0.5f, 0.0f),
               glm::vec3(0.5f, 0.5f, 0.0f));
  EXPECT_FALSE(scene.sealDDGIActivationCoverageBounds());

  ASSERT_TRUE(scene.graph().setNodeLocalTransform(
      node, glm::translate(glm::mat4(1.0f), glm::vec3(20.0f, 0.0f, 0.0f))));
  commit = scene.commit();
  ASSERT_FALSE(commit.hasError()) << commit.error();
  EXPECT_TRUE(
      scene.stageDDGIStaticCoverageBounds(scene.ddgiCurrentCoverageBounds()));
  expectBounds(scene.ddgiCurrentCoverageBounds(), glm::vec3(19.5f, -0.5f, 0.0f),
               glm::vec3(20.5f, 0.5f, 0.0f));
  expectBounds(scene.ddgiActivationCoverageBounds(),
               glm::vec3(-0.5f, -0.5f, 0.0f), glm::vec3(0.5f, 0.5f, 0.0f));
  expectBounds(scene.ddgiStaticCoverageBounds(), glm::vec3(-0.5f, -0.5f, 0.0f),
               glm::vec3(0.5f, 0.5f, 0.0f));
  expectBounds(scene.ddgiPendingStaticCoverageBounds(),
               glm::vec3(-0.5f, -0.5f, 0.0f), glm::vec3(20.5f, 0.5f, 0.0f));
  EXPECT_TRUE(scene.refitDDGIStaticCoverageBounds());
  expectBounds(scene.ddgiStaticCoverageBounds(), glm::vec3(-0.5f, -0.5f, 0.0f),
               glm::vec3(20.5f, 0.5f, 0.0f));
  EXPECT_FALSE(scene.refitDDGIStaticCoverageBounds());

  EXPECT_TRUE(scene.refitDDGIActivationCoverageBounds());
  expectBounds(scene.ddgiActivationCoverageBounds(),
               glm::vec3(19.5f, -0.5f, 0.0f), glm::vec3(20.5f, 0.5f, 0.0f));
  EXPECT_GT(scene.ddgiActivationCoverageBounds().generation,
            activation.generation);

  EXPECT_TRUE(scene.resetDDGIActivationCoverageBounds());
  EXPECT_FALSE(scene.ddgiActivationCoverageBoundsSealed());
  EXPECT_FALSE(scene.ddgiActivationCoverageBounds().valid);
  EXPECT_FALSE(scene.ddgiActivationCoverageBounds().complete);
  EXPECT_FALSE(scene.resetDDGIActivationCoverageBounds());
}

TEST(RenderSceneDDGIBoundsTests,
     MarksCoverageIncompleteWhenCommittedCandidatesCannotResolve) {
  RenderScene scene;
  constexpr ModelRef model{packResourceHandle(1u, 1u)};
  constexpr MaterialRef material{packResourceHandle(2u, 1u)};
  (void)addRenderable(scene, model, material, 0.0f);

  auto commit = scene.commit();
  ASSERT_FALSE(commit.hasError()) << commit.error();
  EXPECT_FALSE(scene.ddgiCurrentCoverageBounds().valid);
  EXPECT_FALSE(scene.ddgiCurrentCoverageBounds().complete);
  EXPECT_FALSE(scene.ddgiStaticCoverageBounds().valid);
  EXPECT_FALSE(scene.ddgiStaticCoverageBounds().complete);
  EXPECT_FALSE(scene.ddgiPendingStaticCoverageBounds().valid);
  EXPECT_FALSE(scene.ddgiPendingStaticCoverageBounds().complete);

  EXPECT_TRUE(scene.sealDDGIActivationCoverageBounds());
  EXPECT_TRUE(scene.ddgiActivationCoverageBoundsSealed());
  EXPECT_FALSE(scene.ddgiActivationCoverageBounds().valid);
  EXPECT_FALSE(scene.ddgiActivationCoverageBounds().complete);
}

} // namespace
