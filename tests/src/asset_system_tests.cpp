#include "tests_pch.h"

#include "nuri/resources/async/asset_system.h"
#include "nuri/scene/render_scene.h"
#include "render_graph_test_support.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory_resource>
#include <thread>

namespace {

using namespace std::chrono_literals;

class ScopedAssetTempDir {
public:
  ScopedAssetTempDir() {
    path = std::filesystem::current_path() / ".scratch" /
           ("nuri_asset_system_" +
            std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(path);
  }

  ~ScopedAssetTempDir() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }

  std::filesystem::path path{};
};

void writePpm(const std::filesystem::path &path, uint8_t red) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(output.is_open());
  output << "P6\n1 1\n255\n";
  const std::array<char, 3> pixel{
      static_cast<char>(red),
      static_cast<char>(0),
      static_cast<char>(0),
  };
  output.write(pixel.data(), static_cast<std::streamsize>(pixel.size()));
  ASSERT_TRUE(output.good());
}

void writeObjScene(const std::filesystem::path &directory) {
  writePpm(directory / "red.ppm", 255u);
  {
    std::ofstream material(directory / "scene.mtl",
                           std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(material.is_open());
    material << "newmtl red\n"
                "Kd 1 1 1\n"
                "map_Kd red.ppm\n";
    ASSERT_TRUE(material.good());
  }
  {
    std::ofstream model(directory / "scene.obj",
                        std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(model.is_open());
    model << "mtllib scene.mtl\n"
             "o triangle\n"
             "v 0 0 0\n"
             "v 1 0 0\n"
             "v 0 1 0\n"
             "vt 0 0\n"
             "vt 1 0\n"
             "vt 0 1\n"
             "vn 0 0 1\n"
             "usemtl red\n"
             "f 1/1/1 2/2/1 3/3/1\n";
    ASSERT_TRUE(model.good());
  }
}

template <typename Handle>
void pumpUntilTerminal(nuri::test_support::FakeExecutorGPUDevice &gpu,
                       nuri::ResourceManager &resources,
                       nuri::AssetSystem &assets, Handle handle,
                       uint64_t &frameIndex) {
  for (uint32_t attempt = 0u; attempt < 200u; ++attempt) {
    ASSERT_FALSE(gpu.beginFrame(frameIndex).hasError());
    resources.beginFrame(frameIndex);
    auto prepared = assets.prepareFrame();
    ASSERT_FALSE(prepared.hasError()) << prepared.error();
    if (assets.query(handle).terminal()) {
      return;
    }
    ++frameIndex;
    std::this_thread::sleep_for(1ms);
  }
  FAIL() << "asset did not reach a terminal state";
}

TEST(AssetSystemTests, DeduplicatesAndPublishesOnlyAfterUploadCompletion) {
  ScopedAssetTempDir dir;
  const std::filesystem::path texturePath = dir.path / "red.ppm";
  writePpm(texturePath, 255u);

  nuri::test_support::FakeExecutorGPUDevice gpu;
  std::pmr::monotonic_buffer_resource memory;
  nuri::ResourceManager resources(gpu, &memory);
  nuri::AssetSystem assets(
      gpu, resources,
      nuri::AssetSystemConfig{
          .cpu = nuri::AssetCpuSchedulerConfig{
              .workerCount = 1u,
              .maxInFlightJobs = 8u,
              .maxInFlightBytes = 16ull * 1024ull * 1024ull,
          },
          .maxGpuMaterializationsPerFrame = 4u,
          .maxGpuUploadBytesPerFrame = 16ull * 1024ull * 1024ull,
      });

  const nuri::TextureRequest request{
      .path = texturePath.string(),
      .debugName = "async_red",
  };
  auto first = assets.requestTexture(request, nuri::AssetPriority::Visible);
  auto duplicate =
      assets.requestTexture(request, nuri::AssetPriority::Critical);
  ASSERT_FALSE(first.hasError()) << first.error();
  ASSERT_FALSE(duplicate.hasError()) << duplicate.error();
  EXPECT_EQ(first.value(), duplicate.value());

  uint64_t frameIndex = 1u;
  bool sawSubmitted = false;
  for (uint32_t attempt = 0u; attempt < 200u; ++attempt) {
    ASSERT_FALSE(gpu.beginFrame(frameIndex).hasError());
    resources.beginFrame(frameIndex);
    auto prepared = assets.prepareFrame();
    ASSERT_FALSE(prepared.hasError()) << prepared.error();
    const nuri::AssetLoadSnapshot snapshot = assets.query(first.value());
    if (snapshot.state == nuri::AssetState::GpuSubmitted) {
      sawSubmitted = true;
      EXPECT_FALSE(assets.tryResolve(first.value()).has_value());
    }
    if (snapshot.terminal()) {
      break;
    }
    ++frameIndex;
    std::this_thread::sleep_for(1ms);
  }

  EXPECT_TRUE(sawSubmitted);
  ASSERT_EQ(assets.query(first.value()).state, nuri::AssetState::Published);
  EXPECT_TRUE(assets.tryResolve(first.value()).has_value());
  EXPECT_EQ(gpu.createdTextureCount, 1u);
  EXPECT_EQ(gpu.pendingUploadSubmitCount, 1u);
}

TEST(AssetSystemTests, CancellationAfterSubmissionRetiresWithoutPublishing) {
  ScopedAssetTempDir dir;
  const std::filesystem::path texturePath = dir.path / "cancel.ppm";
  writePpm(texturePath, 128u);

  nuri::test_support::FakeExecutorGPUDevice gpu;
  std::pmr::monotonic_buffer_resource memory;
  nuri::ResourceManager resources(gpu, &memory);
  nuri::AssetSystem assets(
      gpu, resources,
      nuri::AssetSystemConfig{
          .cpu = nuri::AssetCpuSchedulerConfig{
              .workerCount = 1u,
              .maxInFlightJobs = 8u,
              .maxInFlightBytes = 16ull * 1024ull * 1024ull,
          },
      });
  auto request = assets.requestTexture(nuri::TextureRequest{
      .path = texturePath.string(),
      .debugName = "cancel_after_submit",
  });
  ASSERT_FALSE(request.hasError()) << request.error();

  uint64_t frameIndex = 1u;
  bool cancelled = false;
  for (uint32_t attempt = 0u; attempt < 200u; ++attempt) {
    ASSERT_FALSE(gpu.beginFrame(frameIndex).hasError());
    resources.beginFrame(frameIndex);
    auto prepared = assets.prepareFrame();
    ASSERT_FALSE(prepared.hasError()) << prepared.error();
    if (!cancelled &&
        assets.query(request.value()).state == nuri::AssetState::GpuSubmitted) {
      assets.cancel(request.value());
      cancelled = true;
    }
    if (assets.query(request.value()).terminal()) {
      break;
    }
    ++frameIndex;
    std::this_thread::sleep_for(1ms);
  }

  EXPECT_TRUE(cancelled);
  EXPECT_EQ(assets.query(request.value()).state, nuri::AssetState::Cancelled);
  EXPECT_FALSE(assets.tryResolve(request.value()).has_value());
  EXPECT_EQ(resources.stats().liveTextures, 0u);
}

TEST(AssetSystemTests, SharedRequestSurvivesOneSubscriberCancellation) {
  ScopedAssetTempDir dir;
  const std::filesystem::path texturePath = dir.path / "shared.ppm";
  writePpm(texturePath, 64u);

  nuri::test_support::FakeExecutorGPUDevice gpu;
  std::pmr::monotonic_buffer_resource memory;
  nuri::ResourceManager resources(gpu, &memory);
  nuri::AssetSystem assets(
      gpu, resources,
      nuri::AssetSystemConfig{
          .cpu = nuri::AssetCpuSchedulerConfig{
              .workerCount = 1u,
              .maxInFlightJobs = 8u,
              .maxInFlightBytes = 16ull * 1024ull * 1024ull,
          },
      });
  const nuri::TextureRequest request{
      .path = texturePath.string(),
      .debugName = "shared_texture",
  };
  auto first = assets.requestTexture(request);
  auto second = assets.requestTexture(request);
  ASSERT_FALSE(first.hasError()) << first.error();
  ASSERT_FALSE(second.hasError()) << second.error();
  ASSERT_EQ(first.value(), second.value());

  assets.cancel(first.value());
  uint64_t frameIndex = 1u;
  pumpUntilTerminal(gpu, resources, assets, second.value(), frameIndex);

  EXPECT_EQ(assets.query(second.value()).state, nuri::AssetState::Published);
  EXPECT_TRUE(assets.tryResolve(second.value()).has_value());
  EXPECT_EQ(gpu.createdTextureCount, 1u);

  assets.cancel(second.value());
  EXPECT_EQ(assets.query(second.value()).state, nuri::AssetState::Cancelled);
  EXPECT_FALSE(assets.tryResolve(second.value()).has_value());
  EXPECT_EQ(resources.stats().liveTextures, 0u);
}

TEST(AssetSystemTests, PublishesMaterialBatchWithSingleVersionAdvance) {
  nuri::test_support::FakeExecutorGPUDevice gpu;
  std::pmr::monotonic_buffer_resource memory;
  nuri::ResourceManager resources(gpu, &memory);
  nuri::AssetSystem assets(
      gpu, resources,
      nuri::AssetSystemConfig{
          .cpu = nuri::AssetCpuSchedulerConfig{
              .workerCount = 1u,
              .maxInFlightJobs = 8u,
              .maxInFlightBytes = 16ull * 1024ull * 1024ull,
          },
      });

  nuri::MaterialAssetRequest firstRequest{
      .debugName = "async_material_first",
      .sourceIdentity = "async_material_first",
  };
  nuri::MaterialAssetRequest secondRequest{
      .debugName = "async_material_second",
      .sourceIdentity = "async_material_second",
  };
  secondRequest.desc.baseColorFactor = glm::vec4(0.25f, 0.5f, 0.75f, 1.0f);

  auto first = assets.requestMaterial(firstRequest);
  auto second = assets.requestMaterial(secondRequest);
  ASSERT_FALSE(first.hasError()) << first.error();
  ASSERT_FALSE(second.hasError()) << second.error();
  const uint64_t previousVersion = resources.materialVersion();

  auto prepared = assets.prepareFrame();
  ASSERT_FALSE(prepared.hasError()) << prepared.error();
  EXPECT_EQ(prepared.value().published, 2u);
  EXPECT_EQ(resources.materialVersion(), previousVersion + 1u);
  const nuri::MaterialTableSnapshot firstSnapshot =
      resources.materialSnapshot();
  EXPECT_EQ(firstSnapshot.dirtyBaseVersion, previousVersion);
  EXPECT_EQ(firstSnapshot.dirtyHeaders.first, 0u);
  EXPECT_EQ(firstSnapshot.dirtyHeaders.count, 2u);
  EXPECT_EQ(assets.query(first.value()).state, nuri::AssetState::Published);
  EXPECT_EQ(assets.query(second.value()).state, nuri::AssetState::Published);
  EXPECT_TRUE(assets.tryResolve(first.value()).has_value());
  EXPECT_TRUE(assets.tryResolve(second.value()).has_value());

  nuri::MaterialAssetRequest thirdRequest{
      .debugName = "async_material_third",
      .sourceIdentity = "async_material_third",
  };
  thirdRequest.desc.baseColorFactor = glm::vec4(0.75f, 0.5f, 0.25f, 1.0f);
  auto third = assets.requestMaterial(thirdRequest);
  ASSERT_FALSE(third.hasError()) << third.error();
  prepared = assets.prepareFrame();
  ASSERT_FALSE(prepared.hasError()) << prepared.error();
  const nuri::MaterialTableSnapshot secondSnapshot =
      resources.materialSnapshot();
  EXPECT_EQ(secondSnapshot.dirtyBaseVersion, firstSnapshot.version);
  EXPECT_EQ(secondSnapshot.dirtyHeaders.first, 2u);
  EXPECT_EQ(secondSnapshot.dirtyHeaders.count, 1u);
}

TEST(AssetSystemTests,
     ProgressiveScenePublishesHierarchyBeforeResidentRenderable) {
  ScopedAssetTempDir dir;
  writeObjScene(dir.path);

  nuri::test_support::FakeExecutorGPUDevice gpu;
  std::pmr::monotonic_buffer_resource memory;
  nuri::ResourceManager resources(gpu, &memory);
  nuri::AssetSystem assets(
      gpu, resources,
      nuri::AssetSystemConfig{
          .cpu = nuri::AssetCpuSchedulerConfig{
              .workerCount = 3u,
              .maxInFlightJobs = 32u,
              .maxInFlightBytes = 64ull * 1024ull * 1024ull,
          },
          .maxGpuMaterializationsPerFrame = 4u,
          .maxGpuUploadBytesPerFrame = 32ull * 1024ull * 1024ull,
      });
  nuri::RenderScene scene(&memory);
  scene.bindResources(&resources);
  auto requested = assets.requestScene(nuri::SceneLoadRequest{
      .path = (dir.path / "scene.obj").string(),
      .priority = nuri::AssetPriority::Visible,
  });
  ASSERT_FALSE(requested.hasError()) << requested.error();

  bool sawHierarchyWithoutRenderable = false;
  float previousProgress = 0.0f;
  uint64_t frameIndex = 1u;
  for (uint32_t attempt = 0u; attempt < 500u; ++attempt) {
    ASSERT_FALSE(gpu.beginFrame(frameIndex).hasError());
    resources.beginFrame(frameIndex);
    auto prepared = assets.prepareFrame(nuri::AssetPublicationContext{
        .scene = &scene,
    });
    ASSERT_FALSE(prepared.hasError()) << prepared.error();
    const nuri::SceneLoadSnapshot snapshot = assets.query(requested.value());
    EXPECT_GE(snapshot.progress, previousProgress);
    previousProgress = snapshot.progress;
    if (snapshot.hierarchyPublished && scene.renderables().empty()) {
      sawHierarchyWithoutRenderable = true;
    }
    if (snapshot.terminal()) {
      break;
    }
    ++frameIndex;
    std::this_thread::sleep_for(2ms);
  }

  const nuri::SceneLoadSnapshot finalSnapshot =
      assets.query(requested.value());
  EXPECT_TRUE(sawHierarchyWithoutRenderable);
  EXPECT_EQ(finalSnapshot.state, nuri::SceneLoadState::Complete)
      << "models failed=" << finalSnapshot.models.failed
      << " materials failed=" << finalSnapshot.materials.failed
      << " textures failed=" << finalSnapshot.textures.failed
      << " optional failures=" << finalSnapshot.optionalFailures
      << " error=" << finalSnapshot.error;
  EXPECT_EQ(finalSnapshot.publishedRenderables, 1u);
  ASSERT_EQ(scene.renderables().size(), 1u);
  EXPECT_TRUE(nuri::isValid(scene.renderables()[0].model));
  EXPECT_TRUE(nuri::isValid(scene.renderables()[0].material));
}

TEST(AssetSystemTests, CompleteOnlyScenePublishesAsOneGraphTransaction) {
  ScopedAssetTempDir dir;
  writeObjScene(dir.path);

  nuri::test_support::FakeExecutorGPUDevice gpu;
  std::pmr::monotonic_buffer_resource memory;
  nuri::ResourceManager resources(gpu, &memory);
  nuri::AssetSystem assets(
      gpu, resources,
      nuri::AssetSystemConfig{
          .cpu = nuri::AssetCpuSchedulerConfig{
              .workerCount = 3u,
              .maxInFlightJobs = 32u,
              .maxInFlightBytes = 64ull * 1024ull * 1024ull,
          },
          .maxScenePatchesPerFrame = 1u,
      });
  nuri::RenderScene scene(&memory);
  scene.bindResources(&resources);
  auto requested = assets.requestScene(nuri::SceneLoadRequest{
      .path = (dir.path / "scene.obj").string(),
      .publication = nuri::ScenePublicationPolicy::CompleteOnly,
  });
  ASSERT_FALSE(requested.hasError()) << requested.error();

  uint64_t frameIndex = 1u;
  for (uint32_t attempt = 0u; attempt < 500u; ++attempt) {
    ASSERT_FALSE(gpu.beginFrame(frameIndex).hasError());
    resources.beginFrame(frameIndex);
    auto prepared = assets.prepareFrame(nuri::AssetPublicationContext{
        .scene = &scene,
    });
    ASSERT_FALSE(prepared.hasError()) << prepared.error();
    const nuri::SceneLoadSnapshot snapshot = assets.query(requested.value());
    if (!snapshot.terminal()) {
      EXPECT_FALSE(snapshot.hierarchyPublished);
      EXPECT_TRUE(scene.renderables().empty());
    }
    if (snapshot.terminal()) {
      break;
    }
    ++frameIndex;
    std::this_thread::sleep_for(2ms);
  }

  const nuri::SceneLoadSnapshot snapshot = assets.query(requested.value());
  EXPECT_EQ(snapshot.state, nuri::SceneLoadState::Complete)
      << snapshot.error;
  EXPECT_TRUE(snapshot.hierarchyPublished);
  ASSERT_EQ(scene.renderables().size(), 1u);
  EXPECT_TRUE(nuri::isValid(scene.renderables()[0].model));
  EXPECT_TRUE(nuri::isValid(scene.renderables()[0].material));
}

TEST(AssetSystemTests, SceneCancellationRemovesPublishedHierarchy) {
  ScopedAssetTempDir dir;
  writeObjScene(dir.path);

  nuri::test_support::FakeExecutorGPUDevice gpu;
  std::pmr::monotonic_buffer_resource memory;
  nuri::ResourceManager resources(gpu, &memory);
  nuri::AssetSystem assets(
      gpu, resources,
      nuri::AssetSystemConfig{
          .cpu = nuri::AssetCpuSchedulerConfig{
              .workerCount = 3u,
              .maxInFlightJobs = 32u,
              .maxInFlightBytes = 64ull * 1024ull * 1024ull,
          },
      });
  nuri::RenderScene scene(&memory);
  scene.bindResources(&resources);
  auto requested = assets.requestScene(nuri::SceneLoadRequest{
      .path = (dir.path / "scene.obj").string(),
  });
  ASSERT_FALSE(requested.hasError()) << requested.error();

  bool cancellationRequested = false;
  uint64_t frameIndex = 1u;
  for (uint32_t attempt = 0u; attempt < 500u; ++attempt) {
    ASSERT_FALSE(gpu.beginFrame(frameIndex).hasError());
    resources.beginFrame(frameIndex);
    auto prepared = assets.prepareFrame(nuri::AssetPublicationContext{
        .scene = &scene,
    });
    ASSERT_FALSE(prepared.hasError()) << prepared.error();
    const nuri::SceneLoadSnapshot snapshot = assets.query(requested.value());
    if (!cancellationRequested && snapshot.hierarchyPublished) {
      assets.cancel(requested.value());
      cancellationRequested = true;
    }
    if (snapshot.terminal()) {
      break;
    }
    ++frameIndex;
    std::this_thread::sleep_for(2ms);
  }

  EXPECT_TRUE(cancellationRequested);
  EXPECT_EQ(assets.query(requested.value()).state,
            nuri::SceneLoadState::Cancelled);
  EXPECT_TRUE(scene.renderables().empty());

  // Asset-system generations are an internal lifecycle contract that renderer
  // autotests cannot observe directly: once the last subscriber cancels a
  // terminal scene, the next request may reuse its slot but the stale handle
  // must never resolve to the replacement generation.
  auto replacement = assets.requestScene(nuri::SceneLoadRequest{
      .path = (dir.path / "scene.obj").string(),
  });
  ASSERT_FALSE(replacement.hasError()) << replacement.error();
  EXPECT_EQ(replacement.value().index, requested.value().index);
  EXPECT_NE(replacement.value().generation, requested.value().generation);
  EXPECT_EQ(assets.query(requested.value()).state,
            nuri::SceneLoadState::Failed);
  EXPECT_EQ(assets.query(requested.value()).error,
            "invalid or stale scene load handle");
  assets.cancel(replacement.value());
}

TEST(AssetSystemTests, EnvironmentPublishesRequiredSetTransactionally) {
  ScopedAssetTempDir dir;
  const std::filesystem::path cubemapPath = dir.path / "cubemap.ppm";
  const std::filesystem::path irradiancePath = dir.path / "irradiance.ppm";
  writePpm(cubemapPath, 32u);
  writePpm(irradiancePath, 96u);

  nuri::test_support::FakeExecutorGPUDevice gpu;
  std::pmr::monotonic_buffer_resource memory;
  nuri::ResourceManager resources(gpu, &memory);
  nuri::AssetSystem assets(
      gpu, resources,
      nuri::AssetSystemConfig{
          .cpu = nuri::AssetCpuSchedulerConfig{
              .workerCount = 2u,
              .maxInFlightJobs = 8u,
              .maxInFlightBytes = 16ull * 1024ull * 1024ull,
          },
          .maxGpuMaterializationsPerFrame = 1u,
      });
  nuri::RenderScene scene(&memory);
  scene.bindResources(&resources);
  auto requested =
      assets.requestEnvironment(nuri::EnvironmentAssetRequest{
          .cubemap = nuri::TextureRequest{
              .path = cubemapPath.string(),
              .debugName = "environment_cubemap",
          },
          .irradiance = nuri::TextureRequest{
              .path = irradiancePath.string(),
              .debugName = "environment_irradiance",
          },
      });
  ASSERT_FALSE(requested.hasError()) << requested.error();

  uint64_t frameIndex = 1u;
  for (uint32_t attempt = 0u; attempt < 300u; ++attempt) {
    ASSERT_FALSE(gpu.beginFrame(frameIndex).hasError());
    resources.beginFrame(frameIndex);
    auto prepared = assets.prepareFrame(nuri::AssetPublicationContext{
        .scene = &scene,
    });
    ASSERT_FALSE(prepared.hasError()) << prepared.error();
    const nuri::AssetLoadSnapshot snapshot = assets.query(requested.value());
    if (snapshot.state != nuri::AssetState::Published) {
      EXPECT_FALSE(nuri::isValid(scene.environment().cubemap));
      EXPECT_FALSE(nuri::isValid(scene.environment().irradiance));
    }
    if (snapshot.terminal()) {
      break;
    }
    ++frameIndex;
    std::this_thread::sleep_for(1ms);
  }

  EXPECT_EQ(assets.query(requested.value()).state,
            nuri::AssetState::Published);
  EXPECT_TRUE(nuri::isValid(scene.environment().cubemap));
  EXPECT_TRUE(nuri::isValid(scene.environment().irradiance));
  EXPECT_TRUE(assets.tryResolve(requested.value()).has_value());

  assets.cancel(requested.value());
  ASSERT_FALSE(gpu.beginFrame(++frameIndex).hasError());
  resources.beginFrame(frameIndex);
  auto prepared = assets.prepareFrame(nuri::AssetPublicationContext{
      .scene = &scene,
  });
  ASSERT_FALSE(prepared.hasError()) << prepared.error();
  EXPECT_EQ(assets.query(requested.value()).state,
            nuri::AssetState::Cancelled);
  EXPECT_FALSE(nuri::isValid(scene.environment().cubemap));
  EXPECT_FALSE(nuri::isValid(scene.environment().irradiance));
}

} // namespace
