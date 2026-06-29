#include "nuri/tools/snapshot/snapshot_runner.h"

#include "nuri/core/log.h"
#include "nuri/core/runtime_config.h"
#include "nuri/core/window.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/default_render_pipeline.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/gfx/renderer.h"
#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/resources/scene_importer.h"
#include "nuri/scene/camera.h"
#include "nuri/scene/render_scene.h"
#include "nuri/tools/snapshot/snapshot_baseline.h"
#include "nuri/tools/snapshot/snapshot_capture_artifacts.h"
#include "nuri/tools/snapshot/snapshot_capture_point.h"
#include "nuri/tools/snapshot/snapshot_html_report.h"
#include "nuri/tools/snapshot/snapshot_image.h"
#include "nuri/tools/snapshot/snapshot_manifest.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <memory_resource>
#include <optional>
#include <sstream>

#if defined(_WIN32)
#include <stdlib.h>
#endif

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace nuri::tools::snapshot {
namespace {

class ScopedEnvVar final {
public:
  ScopedEnvVar(std::string name, std::string value)
      : name_(std::move(name)), oldValue_(readProcessEnvironment(name_)),
        hadOldValue_(!oldValue_.empty()) {
    set(value);
  }
  ~ScopedEnvVar() {
    if (hadOldValue_) {
      set(oldValue_);
    } else {
      unset();
    }
  }
  ScopedEnvVar(const ScopedEnvVar &) = delete;
  ScopedEnvVar &operator=(const ScopedEnvVar &) = delete;

private:
  void set(const std::string &value) {
#if defined(_WIN32)
    _putenv_s(name_.c_str(), value.c_str());
#else
    setenv(name_.c_str(), value.c_str(), 1);
#endif
  }
  void unset() {
#if defined(_WIN32)
    _putenv_s(name_.c_str(), "");
#else
    unsetenv(name_.c_str());
#endif
  }
  std::string name_;
  std::string oldValue_;
  bool hadOldValue_ = false;
};

class SnapshotLogGuard final {
public:
  SnapshotLogGuard() {
    std::filesystem::create_directories("logs");
    LogConfig config{};
    config.filePath = (std::filesystem::path("logs") /
                       (utcTimestampForPath() + "_nuri_snapshot.log"))
                          .string();
    config.logLevel = LogLevel::Info;
    config.consoleLevel = LogLevel::Warning;
    config.threadNames = false;
    Log::initialize(config);
  }
  ~SnapshotLogGuard() { Log::shutdown(); }
  SnapshotLogGuard(const SnapshotLogGuard &) = delete;
  SnapshotLogGuard &operator=(const SnapshotLogGuard &) = delete;
};

[[nodiscard]] GPUBackendPreference backendPreference(std::string_view backend) {
  if (backend == "lvk") {
    return GPUBackendPreference::Lvk;
  }
  if (backend == "nvrhi") {
    return GPUBackendPreference::Nvrhi;
  }
  return GPUBackendPreference::Default;
}

[[nodiscard]] std::string resolveBackendName(const SnapshotCase &snapshotCase,
                                             std::string &source) {
  const std::string envBackend = readProcessEnvironment("NURI_GPU_BACKEND");
  if (snapshotCase.backend != "default") {
    source = "manifest";
    return snapshotCase.backend;
  }
  if (!envBackend.empty()) {
    source = "NURI_GPU_BACKEND";
    return envBackend;
  }
  source = "default";
  return "nvrhi";
}

[[nodiscard]] std::string
resolvePresentMode(const SnapshotCase &snapshotCase, std::string &source) {
  const std::string envPresent = readProcessEnvironment("NURI_PRESENT_MODE");
  if (snapshotCase.presentMode != "default") {
    source = "manifest";
    return snapshotCase.presentMode;
  }
  if (!envPresent.empty()) {
    source = "NURI_PRESENT_MODE";
    return envPresent;
  }
  source = "default";
  return "default";
}

[[nodiscard]] Result<bool, SnapshotExitCode>
checkRequirements(const SnapshotCase &snapshotCase, std::string_view backend,
                  std::vector<std::string> &warnings, std::string &message) {
  if (!snapshotCase.requirements.allowVisibleWindow) {
    message = "case requires hidden/headless execution, which is unavailable";
    return Result<bool, SnapshotExitCode>::makeError(
        SnapshotExitCode::EnvironmentUnavailable);
  }
  if (!snapshotCase.requirements.backends.empty()) {
    bool supported = false;
    for (const std::string &allowed : snapshotCase.requirements.backends) {
      supported = supported || allowed == backend || allowed == "default";
    }
    if (!supported) {
      message = "backend '" + std::string(backend) +
                "' is not allowed by case requirements";
      return Result<bool, SnapshotExitCode>::makeError(
          SnapshotExitCode::EnvironmentUnavailable);
    }
  }
  for (const std::string &asset : snapshotCase.requirements.assets) {
    const size_t colon = asset.find(':');
    if (colon == std::string::npos) {
      message = "invalid asset requirement '" + asset + "'";
      return Result<bool, SnapshotExitCode>::makeError(
          SnapshotExitCode::InvalidInput);
    }
    auto path =
        resolveSnapshotPath(asset.substr(0, colon), asset.substr(colon + 1u));
    if (path.hasError()) {
      message = path.error();
      return Result<bool, SnapshotExitCode>::makeError(
          SnapshotExitCode::EnvironmentUnavailable);
    }
    if (!std::filesystem::exists(path.value())) {
      message = "missing required asset: " + path.value().string();
      return Result<bool, SnapshotExitCode>::makeError(
          SnapshotExitCode::EnvironmentUnavailable);
    }
  }
  (void)warnings;
  return Result<bool, SnapshotExitCode>::makeResult(true);
}

[[nodiscard]] Result<bool, std::string>
populateTransmissionTransparencyScene(const SnapshotCase &snapshotCase,
                                      Renderer &renderer, RenderScene &scene) {
  if (snapshotCase.scene.generator !=
      "nuri.procedural.transmission_transparency.v1") {
    return Result<bool, std::string>::makeResult(true);
  }

  auto planePath = resolveSnapshotPath("modelsRoot", "common/flat_plane.obj");
  if (planePath.hasError()) {
    return Result<bool, std::string>::makeError(planePath.error());
  }
  auto modelResult = renderer.resources().acquireModel(ModelRequest{
      .path = planePath.value().string(),
      .debugName = "snapshot_transmission_flat_plane",
  });
  if (modelResult.hasError()) {
    return Result<bool, std::string>::makeError(modelResult.error());
  }

  auto acquireMaterial =
      [&](std::string_view name, const glm::vec4 &color,
          MaterialAlphaMode alphaMode,
          float transmissionFactor) -> Result<MaterialRef, std::string> {
    MaterialRequest request{};
    request.debugName = std::string(name);
    request.desc.baseColorFactor = color;
    request.desc.emissiveFactor = glm::vec3(color) * 0.65f;
    request.desc.emissiveStrength = 1.35f;
    request.desc.metallicFactor = 0.0f;
    request.desc.roughnessFactor = 0.35f;
    request.desc.alphaMode = alphaMode;
    request.desc.doubleSided = true;
    if (transmissionFactor > 0.0f) {
      request.desc.featureMask |= kMaterialFeatureTransmission;
      request.desc.transmissionFactor = transmissionFactor;
      request.desc.thicknessFactor = 0.08f;
      request.desc.attenuationDistance = 2.0f;
    }
    return renderer.resources().acquireMaterial(request);
  };

  auto backgroundMaterial =
      acquireMaterial("snapshot_transmission_background",
                      glm::vec4(0.03f, 0.12f, 0.24f, 1.0f),
                      MaterialAlphaMode::Opaque, 0.0f);
  if (backgroundMaterial.hasError()) {
    return Result<bool, std::string>::makeError(backgroundMaterial.error());
  }
  auto warmMaterial =
      acquireMaterial("snapshot_transmission_warm_band",
                      glm::vec4(1.0f, 0.34f, 0.10f, 1.0f),
                      MaterialAlphaMode::Opaque, 0.0f);
  if (warmMaterial.hasError()) {
    return Result<bool, std::string>::makeError(warmMaterial.error());
  }
  auto coolMaterial =
      acquireMaterial("snapshot_transmission_cool_band",
                      glm::vec4(0.08f, 0.72f, 0.95f, 1.0f),
                      MaterialAlphaMode::Opaque, 0.0f);
  if (coolMaterial.hasError()) {
    return Result<bool, std::string>::makeError(coolMaterial.error());
  }
  auto transparentMaterial =
      acquireMaterial("snapshot_plain_transparent",
                      glm::vec4(1.0f, 0.82f, 0.20f, 0.46f),
                      MaterialAlphaMode::Blend, 0.0f);
  if (transparentMaterial.hasError()) {
    return Result<bool, std::string>::makeError(transparentMaterial.error());
  }
  auto transmissionMaterial =
      acquireMaterial("snapshot_blended_transmission",
                      glm::vec4(0.52f, 0.96f, 1.0f, 0.42f),
                      MaterialAlphaMode::Blend, 1.0f);
  if (transmissionMaterial.hasError()) {
    return Result<bool, std::string>::makeError(transmissionMaterial.error());
  }

  const glm::mat4 upright =
      glm::rotate(glm::mat4(1.0f), glm::radians(90.0f),
                  glm::vec3(1.0f, 0.0f, 0.0f));
  auto addPlane = [&](std::string_view name, const glm::vec3 &translation,
                      const glm::vec3 &scale,
                      MaterialRef material) -> Result<bool, std::string> {
    const glm::mat4 transform =
        glm::translate(glm::mat4(1.0f), translation) * upright *
        glm::scale(glm::mat4(1.0f), scale);
    auto nodeResult =
        scene.graph().createNode(scene.graph().rootNode(), name, transform);
    if (nodeResult.hasError()) {
      return Result<bool, std::string>::makeError(nodeResult.error());
    }
    auto renderableResult =
        scene.graph().addRenderable(nodeResult.value(), modelResult.value(),
                                    material);
    if (renderableResult.hasError()) {
      return Result<bool, std::string>::makeError(renderableResult.error());
    }
    return Result<bool, std::string>::makeResult(true);
  };

  auto result = addPlane("OpaqueBackground", glm::vec3(0.0f, 0.0f, -0.80f),
                         glm::vec3(5.0f, 3.0f, 1.0f),
                         backgroundMaterial.value());
  if (result.hasError()) {
    return result;
  }
  result = addPlane("OpaqueWarmBand", glm::vec3(-0.82f, -0.10f, -0.68f),
                    glm::vec3(1.85f, 1.18f, 1.0f), warmMaterial.value());
  if (result.hasError()) {
    return result;
  }
  result = addPlane("OpaqueCoolBand", glm::vec3(0.82f, 0.12f, -0.62f),
                    glm::vec3(1.85f, 1.18f, 1.0f), coolMaterial.value());
  if (result.hasError()) {
    return result;
  }
  result = addPlane("PlainTransparentBehind", glm::vec3(-0.38f, 0.04f, 0.0f),
                    glm::vec3(2.15f, 1.72f, 1.0f),
                    transparentMaterial.value());
  if (result.hasError()) {
    return result;
  }
  return addPlane("BlendedTransmissionInFront", glm::vec3(0.34f, -0.02f, 0.25f),
                  glm::vec3(2.0f, 1.9f, 1.0f),
                  transmissionMaterial.value());
}

[[nodiscard]] Result<bool, std::string>
populateReactiveMaskScene(const SnapshotCase &snapshotCase, Renderer &renderer,
                          RenderScene &scene) {
  if (snapshotCase.scene.generator != "nuri.procedural.reactive_mask.v1") {
    return Result<bool, std::string>::makeResult(true);
  }

  auto planePath = resolveSnapshotPath("modelsRoot", "common/flat_plane.obj");
  if (planePath.hasError()) {
    return Result<bool, std::string>::makeError(planePath.error());
  }
  auto modelResult = renderer.resources().acquireModel(ModelRequest{
      .path = planePath.value().string(),
      .debugName = "snapshot_reactive_flat_plane",
  });
  if (modelResult.hasError()) {
    return Result<bool, std::string>::makeError(modelResult.error());
  }

  auto acquireMaterial =
      [&](std::string_view name, const glm::vec4 &color,
          MaterialAlphaMode alphaMode) -> Result<MaterialRef, std::string> {
    MaterialRequest request{};
    request.debugName = std::string(name);
    request.desc.baseColorFactor = color;
    request.desc.emissiveFactor = glm::vec3(color) * 0.75f;
    request.desc.emissiveStrength = 1.6f;
    request.desc.metallicFactor = 0.0f;
    request.desc.roughnessFactor = 0.55f;
    request.desc.alphaMode = alphaMode;
    request.desc.alphaCutoff = 0.5f;
    request.desc.doubleSided = true;
    return renderer.resources().acquireMaterial(request);
  };

  auto backgroundMaterial =
      acquireMaterial("snapshot_reactive_background",
                      glm::vec4(0.04f, 0.12f, 0.22f, 1.0f),
                      MaterialAlphaMode::Opaque);
  if (backgroundMaterial.hasError()) {
    return Result<bool, std::string>::makeError(backgroundMaterial.error());
  }
  auto maskedMaterial =
      acquireMaterial("snapshot_alpha_mask_reactive",
                      glm::vec4(0.32f, 1.0f, 0.48f, 0.85f),
                      MaterialAlphaMode::Mask);
  if (maskedMaterial.hasError()) {
    return Result<bool, std::string>::makeError(maskedMaterial.error());
  }

  const glm::mat4 upright =
      glm::rotate(glm::mat4(1.0f), glm::radians(90.0f),
                  glm::vec3(1.0f, 0.0f, 0.0f));
  auto addPlane = [&](std::string_view name, const glm::vec3 &translation,
                      const glm::vec3 &scale,
                      MaterialRef material) -> Result<bool, std::string> {
    const glm::mat4 transform =
        glm::translate(glm::mat4(1.0f), translation) * upright *
        glm::scale(glm::mat4(1.0f), scale);
    auto nodeResult =
        scene.graph().createNode(scene.graph().rootNode(), name, transform);
    if (nodeResult.hasError()) {
      return Result<bool, std::string>::makeError(nodeResult.error());
    }
    auto renderableResult =
        scene.graph().addRenderable(nodeResult.value(), modelResult.value(),
                                    material);
    if (renderableResult.hasError()) {
      return Result<bool, std::string>::makeError(renderableResult.error());
    }
    return Result<bool, std::string>::makeResult(true);
  };

  auto result = addPlane("ReactiveBackground", glm::vec3(0.0f, 0.0f, -0.65f),
                         glm::vec3(4.6f, 2.8f, 1.0f),
                         backgroundMaterial.value());
  if (result.hasError()) {
    return result;
  }
  result = addPlane("ReactiveMaskTall", glm::vec3(-0.32f, 0.0f, 0.0f),
                    glm::vec3(1.0f, 1.9f, 1.0f), maskedMaterial.value());
  if (result.hasError()) {
    return result;
  }
  return addPlane("ReactiveMaskOffset", glm::vec3(0.54f, 0.22f, 0.08f),
                  glm::vec3(0.95f, 0.95f, 1.0f), maskedMaterial.value());
}

[[nodiscard]] Result<bool, std::string>
populateShadowPlanesScene(const SnapshotCase &snapshotCase, Renderer &renderer,
                          RenderScene &scene) {
  if (snapshotCase.scene.generator != "nuri.procedural.shadow_planes.v1") {
    return Result<bool, std::string>::makeResult(true);
  }

  auto planePath = resolveSnapshotPath("modelsRoot", "common/flat_plane.obj");
  if (planePath.hasError()) {
    return Result<bool, std::string>::makeError(planePath.error());
  }
  auto modelResult = renderer.resources().acquireModel(ModelRequest{
      .path = planePath.value().string(),
      .debugName = "snapshot_shadow_flat_plane",
  });
  if (modelResult.hasError()) {
    return Result<bool, std::string>::makeError(modelResult.error());
  }

  auto acquireMaterial =
      [&](std::string_view name, const glm::vec4 &color)
      -> Result<MaterialRef, std::string> {
    MaterialRequest request{};
    request.debugName = std::string(name);
    request.desc.baseColorFactor = color;
    request.desc.emissiveFactor = glm::vec3(color) * 0.45f;
    request.desc.emissiveStrength = 1.1f;
    request.desc.metallicFactor = 0.0f;
    request.desc.roughnessFactor = 0.72f;
    request.desc.doubleSided = true;
    return renderer.resources().acquireMaterial(request);
  };

  auto floorMaterial =
      acquireMaterial("snapshot_shadow_floor",
                      glm::vec4(0.72f, 0.70f, 0.64f, 1.0f));
  if (floorMaterial.hasError()) {
    return Result<bool, std::string>::makeError(floorMaterial.error());
  }
  auto wallMaterial =
      acquireMaterial("snapshot_shadow_wall",
                      glm::vec4(0.58f, 0.68f, 0.78f, 1.0f));
  if (wallMaterial.hasError()) {
    return Result<bool, std::string>::makeError(wallMaterial.error());
  }
  auto redMaterial =
      acquireMaterial("snapshot_shadow_red",
                      glm::vec4(0.86f, 0.18f, 0.12f, 1.0f));
  if (redMaterial.hasError()) {
    return Result<bool, std::string>::makeError(redMaterial.error());
  }
  auto blueMaterial =
      acquireMaterial("snapshot_shadow_blue",
                      glm::vec4(0.12f, 0.36f, 0.88f, 1.0f));
  if (blueMaterial.hasError()) {
    return Result<bool, std::string>::makeError(blueMaterial.error());
  }
  auto greenMaterial =
      acquireMaterial("snapshot_shadow_green",
                      glm::vec4(0.16f, 0.72f, 0.34f, 1.0f));
  if (greenMaterial.hasError()) {
    return Result<bool, std::string>::makeError(greenMaterial.error());
  }

  const glm::mat4 upright =
      glm::rotate(glm::mat4(1.0f), glm::radians(90.0f),
                  glm::vec3(1.0f, 0.0f, 0.0f));
  auto addPlane = [&](std::string_view name, const glm::mat4 &orientation,
                      const glm::vec3 &translation, const glm::vec3 &scale,
                      MaterialRef material) -> Result<bool, std::string> {
    const glm::mat4 transform =
        glm::translate(glm::mat4(1.0f), translation) * orientation *
        glm::scale(glm::mat4(1.0f), scale);
    auto nodeResult =
        scene.graph().createNode(scene.graph().rootNode(), name, transform);
    if (nodeResult.hasError()) {
      return Result<bool, std::string>::makeError(nodeResult.error());
    }
    auto renderableResult =
        scene.graph().addRenderable(nodeResult.value(), modelResult.value(),
                                    material);
    if (renderableResult.hasError()) {
      return Result<bool, std::string>::makeError(renderableResult.error());
    }
    return Result<bool, std::string>::makeResult(true);
  };

  auto result = addPlane("ShadowFloor", glm::mat4(1.0f),
                         glm::vec3(0.0f, -0.72f, 0.0f),
                         glm::vec3(5.2f, 1.0f, 4.4f),
                         floorMaterial.value());
  if (result.hasError()) {
    return result;
  }
  result = addPlane("ShadowBackWall", upright, glm::vec3(0.0f, 0.64f, -1.38f),
                    glm::vec3(5.2f, 2.7f, 1.0f), wallMaterial.value());
  if (result.hasError()) {
    return result;
  }
  result = addPlane("ShadowRedCaster", upright, glm::vec3(-1.2f, 0.14f, 0.15f),
                    glm::vec3(0.72f, 1.5f, 1.0f), redMaterial.value());
  if (result.hasError()) {
    return result;
  }
  result = addPlane("ShadowBlueCaster", upright, glm::vec3(0.0f, 0.34f, 0.0f),
                    glm::vec3(0.58f, 1.95f, 1.0f), blueMaterial.value());
  if (result.hasError()) {
    return result;
  }
  return addPlane("ShadowGreenCaster", upright, glm::vec3(1.16f, 0.06f, 0.35f),
                  glm::vec3(0.82f, 1.34f, 1.0f), greenMaterial.value());
}

[[nodiscard]] Result<bool, std::string>
populateScene(const SnapshotCase &snapshotCase, Renderer &renderer,
              RenderScene &scene, std::pmr::memory_resource *memory,
              std::optional<ScenePrefab> &prefab,
              std::optional<ScenePrefabAssets> &prefabAssets) {
  scene.bindResources(&renderer.resources());
  LightDesc keyLight{.type = LightType::Directional,
                     .name = "snapshot_key",
                     .color = glm::vec3(1.0f),
                     .intensity = 4.0f,
                     .enabled = true};
  if (snapshotCase.scene.generator == "nuri.procedural.shadow_planes.v1") {
    keyLight.rotation = glm::quatLookAt(
        glm::normalize(glm::vec3(-0.45f, -0.78f, -0.44f)),
        glm::vec3(0.0f, 1.0f, 0.0f));
    keyLight.intensity = 5.5f;
  }
  auto lightResult = scene.graph().addLight(
      scene.graph().rootNode(), keyLight);
  if (lightResult.hasError()) {
    return Result<bool, std::string>::makeError(lightResult.error());
  }

  if (snapshotCase.scene.kind == "prefab") {
    if (snapshotCase.scene.pathBase.empty() || snapshotCase.scene.path.empty()) {
      return Result<bool, std::string>::makeError(
          "prefab scene requires pathBase and path");
    }
    auto path = resolveSnapshotPath(snapshotCase.scene.pathBase,
                                    snapshotCase.scene.path);
    if (path.hasError()) {
      return Result<bool, std::string>::makeError(path.error());
    }
    if (!std::filesystem::exists(path.value())) {
      return Result<bool, std::string>::makeError(
          "missing scene asset: " + path.value().string());
    }
    SceneImportOptions importOptions{};
    importOptions.assetBuildOptions.flipUVs = snapshotCase.scene.flipUVs;
    importOptions.assetBuildOptions.generateMeshlets =
        snapshotCase.scene.generateMeshlets;
    importOptions.assetBuildOptions.meshletMaxVertices =
        snapshotCase.scene.meshletMaxVertices;
    importOptions.assetBuildOptions.meshletMaxPrimitives =
        snapshotCase.scene.meshletMaxPrimitives;
    importOptions.assetBuildOptions.meshletConeWeight =
        snapshotCase.scene.meshletConeWeight;
    auto prefabResult = SceneImporter::loadScenePrefabFromFile(
        path.value().string(), importOptions, memory);
    if (prefabResult.hasError()) {
      return Result<bool, std::string>::makeError(prefabResult.error());
    }
    prefab.emplace(std::move(prefabResult.value()));
    auto assetsResult = renderer.resources().acquireScenePrefabAssets(*prefab);
    if (assetsResult.hasError()) {
      return Result<bool, std::string>::makeError(assetsResult.error());
    }
    prefabAssets.emplace(std::move(assetsResult.value()));
    auto instantiateResult = scene.graph().instantiatePrefab(
        *prefab, scene.graph().rootNode(), *prefabAssets);
    if (instantiateResult.hasError()) {
      return Result<bool, std::string>::makeError(instantiateResult.error());
    }
  } else if (snapshotCase.scene.kind == "procedural") {
    auto proceduralResult =
        populateTransmissionTransparencyScene(snapshotCase, renderer, scene);
    if (proceduralResult.hasError()) {
      return proceduralResult;
    }
    proceduralResult = populateReactiveMaskScene(snapshotCase, renderer, scene);
    if (proceduralResult.hasError()) {
      return proceduralResult;
    }
    proceduralResult = populateShadowPlanesScene(snapshotCase, renderer, scene);
    if (proceduralResult.hasError()) {
      return proceduralResult;
    }
  }

  auto syncResult = scene.graph().syncWorldTransforms();
  (void)syncResult;
  auto commitResult = scene.commit();
  if (commitResult.hasError()) {
    return Result<bool, std::string>::makeError(commitResult.error());
  }
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Camera makeSnapshotCamera(const SnapshotCase &snapshotCase,
                                        uint64_t frameIndex) {
  Camera camera;
  camera.setPerspective(PerspectiveParams{
      .fovYRadians = glm::radians(snapshotCase.camera.verticalFovDegrees),
      .nearPlane = snapshotCase.camera.nearPlane,
      .farPlane = snapshotCase.camera.farPlane,
  });
  const glm::vec3 direction =
      glm::length(snapshotCase.camera.direction) > 1.0e-6f
          ? glm::normalize(snapshotCase.camera.direction)
          : glm::vec3(0.0f, 0.0f, -1.0f);
  const glm::vec3 position =
      snapshotCase.camera.position +
      snapshotCase.camera.positionDeltaPerFrame * static_cast<float>(frameIndex);
  camera.setLookAt(position,
                   position + direction,
                   glm::vec3(0.0f, 1.0f, 0.0f));
  return camera;
}

void buildFrameContext(RenderFrameContext &frameContext, RenderScene &scene,
                       Renderer &renderer, RenderSettings &settings,
                       TemporalCameraHistoryState &cameraHistory,
                       const Camera &camera, uint64_t frameIndex,
                       double timeSeconds, double deltaSeconds,
                       uint32_t width, uint32_t height) {
  sanitizeSnapshotRenderSettings(settings);
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.frameIndex = frameIndex;
  const MaterialTableSnapshot materialSnapshot =
      renderer.resources().materialSnapshot();
  const TemporalSceneContentState sceneContent{
      .lightTopologyVersion = scene.lightTopologyVersion(),
      .lightTransformVersion = scene.lightTransformVersion(),
      .materialTableVersion = materialSnapshot.version,
      .environmentVersion = scene.environmentVersion(),
  };
  frameContext.camera = makeTemporalCameraFrameState(
      camera, static_cast<float>(width) / static_cast<float>(height),
      settings.antiAliasing,
      TemporalCameraFrameDesc{.renderExtent = glm::uvec2(width, height),
                              .sceneContent = sceneContent},
      cameraHistory);
  settings.antiAliasing.debug.resetHistoryRequested = false;
  frameContext.settings = &settings;
  frameContext.metrics = {};
  frameContext.metrics.frameIndex = frameContext.frameIndex;
  frameContext.metrics.antiAliasing =
      makeAntiAliasingFrameMetrics(frameContext.camera);
  frameContext.sharedDepthTexture = {};
  frameContext.timeSeconds = timeSeconds;
  frameContext.deltaSeconds = deltaSeconds;
}

[[nodiscard]] std::filesystem::path
relativeToCaseDir(const std::filesystem::path &caseDir,
                  const std::filesystem::path &path) {
  std::error_code ec;
  std::filesystem::path relative = std::filesystem::relative(path, caseDir, ec);
  return ec ? path : relative;
}

void initializeCaptureReports(SnapshotReport &report) {
  report.captures.clear();
  for (const SnapshotCaptureTarget &target : report.snapshotCase.captures) {
    SnapshotCaptureReport capture{};
    capture.target = target.name;
    capture.artifactStem = target.name;
    capture.profile = target.profile;
    capture.required = target.required;
    capture.status = "missing_capture_point";
    capture.statusReason = "not_rendered";
    report.captures.push_back(std::move(capture));
  }
}

void writeReports(SnapshotRunResult &result, SnapshotReport &report,
                  const std::filesystem::path &jsonPath,
                  const std::filesystem::path &htmlPath) {
  result.reportPath = jsonPath;
  result.htmlPath = htmlPath;
  report.artifacts.caseHtml = htmlPath;
  auto writeJson = writeSnapshotReportFile(report, jsonPath);
  if (writeJson.hasError() &&
      result.exitCode == SnapshotExitCode::Success) {
    result.exitCode = SnapshotExitCode::RuntimeError;
    result.message = writeJson.error();
  }
  auto writeHtml = writeSnapshotHtmlReportFile(report, htmlPath);
  if (writeHtml.hasError() &&
      result.exitCode == SnapshotExitCode::Success) {
    result.exitCode = SnapshotExitCode::RuntimeError;
    result.message = writeHtml.error();
  }
}

[[nodiscard]] SnapshotReport
makeInitialReport(const SnapshotCase &snapshotCase,
                  const SnapshotRunOptions &options,
                  const std::filesystem::path &artifactDir,
                  const std::filesystem::path &caseDir,
                  const std::filesystem::path &htmlPath,
                  std::string_view backend, std::string_view backendSource,
                  std::string_view presentMode,
                  std::string_view presentSource) {
  SnapshotReport report{};
  report.generatedAtUtc = utcTimestampIso8601();
  report.command = options.command;
  report.snapshotCase = snapshotCase;
  report.artifacts.artifactDir = artifactDir;
  report.artifacts.caseDir = caseDir;
  report.artifacts.caseHtml = htmlPath;
  report.environment = collectSnapshotEnvironment(
      backend, backendSource, presentMode, presentSource, options.windowMode,
      options.windowMode);
  report.environment.renderGraphWorkerCount = snapshotCase.renderGraph.workerCount;
  report.environment.renderGraphParallelCompile =
      snapshotCase.renderGraph.parallelCompile;
  report.environment.renderGraphParallelRecording =
      snapshotCase.renderGraph.parallelRecording;
  report.reproduceCommand = "nuri-snapshot run --case " + snapshotCase.id +
                            " --baseline-profile " + options.baselineProfile;
  initializeCaptureReports(report);
  return report;
}

} // namespace

Result<std::string, std::string>
formatSnapshotCaseListJson(const std::vector<SnapshotCase> &cases,
                           std::string_view suite) {
  std::ostringstream out;
  out << "{\n  \"cases\": [\n";
  bool first = true;
  for (const SnapshotCase &snapshotCase : cases) {
    if (!suite.empty() && snapshotCase.suite != suite) {
      continue;
    }
    if (!first) {
      out << ",\n";
    }
    first = false;
    out << "    {\"id\": \"" << snapshotCase.id << "\", \"suite\": \""
        << snapshotCase.suite << "\", \"description\": \""
        << snapshotCase.description << "\", \"captures\": [";
    for (size_t i = 0u; i < snapshotCase.captures.size(); ++i) {
      if (i != 0u) {
        out << ", ";
      }
      out << "\"" << snapshotCase.captures[i].name << "\"";
    }
    out << "]}";
  }
  out << "\n  ]\n}\n";
  return Result<std::string, std::string>::makeResult(out.str());
}

std::string formatSnapshotCaseListText(const std::vector<SnapshotCase> &cases,
                                       std::string_view suite) {
  std::ostringstream out;
  for (const SnapshotCase &snapshotCase : cases) {
    if (!suite.empty() && snapshotCase.suite != suite) {
      continue;
    }
    out << snapshotCase.id << " [" << snapshotCase.suite << "] "
        << snapshotCase.description << "\n";
  }
  return out.str();
}

Result<std::string, std::string>
formatSnapshotCaseExplanationJson(const SnapshotCase &snapshotCase) {
  std::ostringstream out;
  out << "{\n"
      << "  \"id\": \"" << snapshotCase.id << "\",\n"
      << "  \"suite\": \"" << snapshotCase.suite << "\",\n"
      << "  \"description\": \"" << snapshotCase.description << "\",\n"
      << "  \"sceneKind\": \"" << snapshotCase.scene.kind << "\",\n"
      << "  \"backend\": \"" << snapshotCase.backend << "\",\n"
      << "  \"captures\": [\n";
  for (size_t i = 0u; i < snapshotCase.captures.size(); ++i) {
    const SnapshotCaptureTarget &capture = snapshotCase.captures[i];
    const SnapshotCaptureCatalogEntry *catalog =
        findSnapshotCaptureCatalogEntry(capture.name);
    out << "    {\"target\": \"" << capture.name << "\", \"profile\": \""
        << capture.profile << "\", \"availability\": \""
        << (catalog != nullptr &&
                    catalog->availability ==
                        SnapshotCaptureAvailability::KnownNotCapturable
                ? "known_not_capturable"
                : "first_slice")
        << "\"}";
    if (i + 1u != snapshotCase.captures.size()) {
      out << ",";
    }
    out << "\n";
  }
  out << "  ]\n}\n";
  return Result<std::string, std::string>::makeResult(out.str());
}

std::string formatSnapshotCaseExplanationText(
    const SnapshotCase &snapshotCase) {
  std::ostringstream out;
  out << snapshotCase.id << "\n"
      << "suite: " << snapshotCase.suite << "\n"
      << "description: " << snapshotCase.description << "\n"
      << "scene: " << snapshotCase.scene.kind << "\n"
      << "backend: " << snapshotCase.backend << "\n"
      << "resolution: " << snapshotCase.resolution[0] << "x"
      << snapshotCase.resolution[1] << "\n"
      << "captures:";
  for (const SnapshotCaptureTarget &capture : snapshotCase.captures) {
    out << " " << capture.name;
  }
  out << "\n";
  return out.str();
}

Result<std::string, std::string>
formatSnapshotEffectiveConfigJson(const SnapshotCase &snapshotCase,
                                  const SnapshotRunOptions &options) {
  std::string backendSource;
  const std::string backend = resolveBackendName(snapshotCase, backendSource);
  std::string presentSource;
  const std::string present = resolvePresentMode(snapshotCase, presentSource);
  std::ostringstream out;
  out << "{\n"
      << "  \"case\": \"" << snapshotCase.id << "\",\n"
      << "  \"backend\": \"" << backend << "\",\n"
      << "  \"backendSource\": \"" << backendSource << "\",\n"
      << "  \"presentMode\": \"" << present << "\",\n"
      << "  \"presentModeSource\": \"" << presentSource << "\",\n"
      << "  \"windowMode\": \"" << options.windowMode << "\",\n"
      << "  \"artifactDir\": \"" << options.artifactDir.generic_string()
      << "\"\n"
      << "}\n";
  return Result<std::string, std::string>::makeResult(out.str());
}

SnapshotRunResult captureSnapshotCase(SnapshotCase snapshotCase,
                                      const SnapshotRunOptions &options) {
  SnapshotRunResult result{};
  std::string backendSource;
  const std::string backend = resolveBackendName(snapshotCase, backendSource);
  std::string presentSource;
  const std::string presentMode =
      resolvePresentMode(snapshotCase, presentSource);
  const std::filesystem::path artifactDir =
      options.artifactDir.empty()
          ? snapshotRepoRoot() / "artifacts" / "snapshots" /
                utcTimestampForPath()
          : options.artifactDir;
  const std::filesystem::path caseDir =
      artifactDir / "cases" / snapshotCase.id;
  const std::filesystem::path reportPath =
      options.jsonOut.empty() ? caseDir / "report.json" : options.jsonOut;
  const std::filesystem::path htmlPath =
      options.htmlOut.empty() ? caseDir / "report.html" : options.htmlOut;
  SnapshotReport report = makeInitialReport(
      snapshotCase, options, artifactDir, caseDir, htmlPath, backend,
      backendSource, presentMode, presentSource);

  std::string requirementMessage;
  auto requirements = checkRequirements(snapshotCase, backend, report.warnings,
                                        requirementMessage);
  if (requirements.hasError()) {
    result.exitCode = requirements.error();
    result.message = requirementMessage;
    report.warnings.push_back(requirementMessage);
    for (SnapshotCaptureReport &capture : report.captures) {
      capture.status = "environment_unavailable";
      capture.statusReason = "requirements_unavailable";
    }
    writeReports(result, report, reportPath, htmlPath);
    result.report = std::move(report);
    return result;
  }
  if (options.windowMode != "visible") {
    result.exitCode = SnapshotExitCode::EnvironmentUnavailable;
    result.message = "only visible window mode is available";
    report.warnings.push_back(result.message);
    for (SnapshotCaptureReport &capture : report.captures) {
      capture.status = "environment_unavailable";
      capture.statusReason = "window_mode_unavailable";
    }
    writeReports(result, report, reportPath, htmlPath);
    result.report = std::move(report);
    return result;
  }
  if (options.dryRun) {
    result.exitCode = SnapshotExitCode::Success;
    result.message = "dry run succeeded";
    report.warnings.push_back("dry run: renderer was not initialized");
    for (SnapshotCaptureReport &capture : report.captures) {
      capture.status = "environment_unavailable";
      capture.statusReason = "dry_run";
    }
    writeReports(result, report, reportPath, htmlPath);
    result.report = std::move(report);
    return result;
  }

  for (const char *generatedDir : {"actual", "diff"}) {
    std::error_code cleanupError;
    std::filesystem::remove_all(caseDir / generatedDir, cleanupError);
    if (cleanupError) {
      result.exitCode = SnapshotExitCode::RuntimeError;
      result.message = "failed to clean generated snapshot artifacts: " +
                       cleanupError.message();
      report.errors.push_back(result.message);
      writeReports(result, report, reportPath, htmlPath);
      result.report = std::move(report);
      return result;
    }
  }

  try {
    SnapshotLogGuard logGuard;
    std::vector<std::unique_ptr<ScopedEnvVar>> env;
    env.push_back(std::make_unique<ScopedEnvVar>(
        "NURI_RENDER_GRAPH_WORKER_COUNT",
        std::to_string(snapshotCase.renderGraph.workerCount)));
    env.push_back(std::make_unique<ScopedEnvVar>(
        "NURI_RENDER_GRAPH_DISABLE_PARALLEL_COMPILE",
        snapshotCase.renderGraph.parallelCompile ? "0" : "1"));
    env.push_back(std::make_unique<ScopedEnvVar>(
        "NURI_RENDER_GRAPH_DISABLE_PARALLEL_RECORDING",
        snapshotCase.renderGraph.parallelRecording ? "0" : "1"));
    if (presentSource == "manifest" && presentMode != "default") {
      env.push_back(
          std::make_unique<ScopedEnvVar>("NURI_PRESENT_MODE", presentMode));
    }

    auto configResult = loadRuntimeConfigFromEnvOrDefault();
    if (configResult.hasError()) {
      result.exitCode = SnapshotExitCode::EnvironmentUnavailable;
      result.message = configResult.error();
      report.warnings.push_back(result.message);
      for (SnapshotCaptureReport &capture : report.captures) {
        capture.status = "environment_unavailable";
        capture.statusReason = "runtime_config_unavailable";
      }
      writeReports(result, report, reportPath, htmlPath);
      result.report = std::move(report);
      return result;
    }
    RuntimeConfig config = std::move(configResult.value());
    config.window.title = "nuri-snapshot " + snapshotCase.id;
    config.window.width = static_cast<int32_t>(snapshotCase.resolution[0]);
    config.window.height = static_cast<int32_t>(snapshotCase.resolution[1]);
    config.window.mode = WindowMode::Windowed;

    std::unique_ptr<Window> window = Window::create(
        config.window.title, config.window.width, config.window.height,
        config.window.mode);
    if (!window) {
      result.exitCode = SnapshotExitCode::EnvironmentUnavailable;
      result.message = "failed to create snapshot window";
      report.warnings.push_back(result.message);
      writeReports(result, report, reportPath, htmlPath);
      result.report = std::move(report);
      return result;
    }
    GPUDeviceCreateDesc deviceDesc{};
    deviceDesc.backend = backendPreference(backend);
    std::unique_ptr<GPUDevice> gpu = GPUDevice::create(*window, deviceDesc);
    if (!gpu) {
      result.exitCode = SnapshotExitCode::EnvironmentUnavailable;
      result.message = "failed to create GPU device";
      report.warnings.push_back(result.message);
      writeReports(result, report, reportPath, htmlPath);
      result.report = std::move(report);
      return result;
    }
    report.environment.swapchainImageCount = gpu->getSwapchainImageCount();

    std::pmr::unsynchronized_pool_resource rendererMemory;
    std::pmr::unsynchronized_pool_resource pipelineMemory;
    std::pmr::unsynchronized_pool_resource sceneMemory;
    std::unique_ptr<Renderer> renderer =
        Renderer::create(*gpu, rendererMemory);
    RenderPipeline pipeline(&pipelineMemory);
    auto pipelineResult = registerDefaultRenderPipeline(
        pipeline, *gpu, config.shaders, &pipelineMemory);
    if (pipelineResult.hasError()) {
      result.exitCode = SnapshotExitCode::RuntimeError;
      result.message = pipelineResult.error();
      report.errors.push_back(result.message);
      writeReports(result, report, reportPath, htmlPath);
      result.report = std::move(report);
      return result;
    }

    RenderScene scene(&sceneMemory);
    std::optional<ScenePrefab> prefab;
    std::optional<ScenePrefabAssets> prefabAssets;
    auto sceneResult =
        populateScene(snapshotCase, *renderer, scene, &sceneMemory, prefab,
                      prefabAssets);
    if (sceneResult.hasError()) {
      result.exitCode = SnapshotExitCode::EnvironmentUnavailable;
      result.message = sceneResult.error();
      report.warnings.push_back(result.message);
      writeReports(result, report, reportPath, htmlPath);
      result.report = std::move(report);
      return result;
    }

    RenderSettings settings = snapshotCase.settings;
    TemporalCameraHistoryState cameraHistory{};
    RenderFrameContext frameContext{};
    uint64_t frameIndex = 0u;
    double timeSeconds = 0.0;

    const auto renderOneFrame =
        [&](bool captureFrame) -> Result<bool, std::string> {
      window->pollEvents();
      auto commitResult = scene.commit();
      if (commitResult.hasError()) {
        return Result<bool, std::string>::makeError(commitResult.error());
      }
      const Camera camera = makeSnapshotCamera(snapshotCase, frameIndex);
      buildFrameContext(frameContext, scene, *renderer, settings, cameraHistory,
                        camera, frameIndex, timeSeconds,
                        snapshotCase.fixedDeltaSeconds,
                        snapshotCase.resolution[0], snapshotCase.resolution[1]);
      frameContext.captureRequests.clear();
      if (captureFrame) {
        for (const SnapshotCaptureTarget &capture : snapshotCase.captures) {
          frameContext.captureRequests.request(capture.name);
        }
      }
      auto renderResult = renderer->render(pipeline, frameContext);
      if (renderResult.hasError()) {
        return Result<bool, std::string>::makeError(renderResult.error());
      }
      ++frameIndex;
      timeSeconds += snapshotCase.fixedDeltaSeconds;
      return Result<bool, std::string>::makeResult(true);
    };

    for (uint32_t i = 0u; i < snapshotCase.captureFrame; ++i) {
      auto frameResult = renderOneFrame(false);
      if (frameResult.hasError()) {
        result.exitCode = SnapshotExitCode::RuntimeError;
        result.message = frameResult.error();
        report.errors.push_back(result.message);
        writeReports(result, report, reportPath, htmlPath);
        result.report = std::move(report);
        return result;
      }
    }
    auto captureFrameResult = renderOneFrame(true);
    if (captureFrameResult.hasError()) {
      result.exitCode = SnapshotExitCode::RuntimeError;
      result.message = captureFrameResult.error();
      report.errors.push_back(result.message);
      writeReports(result, report, reportPath, htmlPath);
      result.report = std::move(report);
      return result;
    }
    gpu->waitIdle();

    report.rendererMetrics = frameContext.metrics;
    auto captures = writeSnapshotCaptureArtifacts(
        *gpu, frameContext, snapshotCase.captures, caseDir, caseDir / "actual");
    if (captures.hasError()) {
      result.exitCode = SnapshotExitCode::RuntimeError;
      result.message = captures.error();
      report.errors.push_back(result.message);
      writeReports(result, report, reportPath, htmlPath);
      result.report = std::move(report);
      return result;
    }
    SnapshotCaptureArtifactResult captureArtifacts =
        std::move(captures.value());
    report.captures = std::move(captureArtifacts.captures);
    report.availableCapturePoints =
        std::move(captureArtifacts.availableCapturePoints);
    if (result.exitCode == SnapshotExitCode::Success) {
      if (captureArtifacts.missingRequiredCapture) {
        result.exitCode = SnapshotExitCode::EnvironmentUnavailable;
        result.message = "required capture point missing";
      } else if (captureArtifacts.unsupportedRequiredCapture) {
        result.exitCode = SnapshotExitCode::EnvironmentUnavailable;
        result.message = "required capture format unsupported";
      } else if (captureArtifacts.readbackFailedRequiredCapture) {
        result.exitCode = SnapshotExitCode::RuntimeError;
        result.message = "required capture readback failed";
      }
    }
  } catch (const std::exception &ex) {
    result.exitCode = SnapshotExitCode::RuntimeError;
    result.message = ex.what();
    report.errors.push_back(result.message);
  }

  if (result.exitCode == SnapshotExitCode::Success) {
    result.message = "snapshot capture complete";
  }
  writeReports(result, report, reportPath, htmlPath);
  result.report = std::move(report);
  return result;
}

SnapshotRunResult compareSnapshotCase(SnapshotCase snapshotCase,
                                      const SnapshotRunOptions &options) {
  SnapshotRunResult result{};
  const std::filesystem::path artifactDir =
      options.artifactDir.empty()
          ? snapshotRepoRoot() / "artifacts" / "snapshots"
          : options.artifactDir;
  const std::filesystem::path caseDir = artifactDir / "cases" / snapshotCase.id;
  const std::filesystem::path reportPath =
      options.jsonOut.empty() ? caseDir / "report.json" : options.jsonOut;
  auto reportResult = readSnapshotReportFile(reportPath);
  if (reportResult.hasError()) {
    result.exitCode = SnapshotExitCode::InvalidInput;
    result.message = reportResult.error();
    return result;
  }
  SnapshotReport report = std::move(reportResult.value());
  report.snapshotCase = snapshotCase;
  report.artifacts.caseDir = caseDir;
  report.artifacts.artifactDir = artifactDir;
  const SnapshotBaselineLookup baseline =
      snapshotBaselineLookup(snapshotCase, options.baselineProfile);
  bool missingBaseline = false;
  bool mismatch = false;
  for (SnapshotCaptureReport &capture : report.captures) {
    if (capture.actual.empty() || capture.status == "missing_capture_point" ||
        capture.status == "unsupported_format" ||
        capture.status == "readback_error") {
      continue;
    }
    const std::filesystem::path actual = caseDir / capture.actual;
    const std::filesystem::path actualExtension = actual.extension();
    const bool usePreview =
        actualExtension == ".nuri_tex" || actualExtension.empty();
    const std::filesystem::path expected =
        usePreview ? baseline.caseDir / (capture.target + "_preview.png")
                   : baseline.caseDir /
                         (capture.target + actualExtension.string());
    capture.expected = expected;
    if (!std::filesystem::exists(expected)) {
      capture.status = "missing_baseline";
      capture.statusReason = "baseline_artifact_missing";
      missingBaseline = true;
      continue;
    }
    auto actualImage =
        readSnapshotImageFile(usePreview ? caseDir / capture.preview : actual);
    auto expectedImage = readSnapshotImageFile(expected);
    if (actualImage.hasError() || expectedImage.hasError()) {
      capture.status = "runtime_error";
      capture.statusReason = "failed_to_load_compare_images";
      result.exitCode = SnapshotExitCode::RuntimeError;
      continue;
    }
    const SnapshotCompareProfile profile =
        builtinSnapshotCompareProfile(capture.profile);
    SnapshotCompareResult comparison =
        compareSnapshotImages(actualImage.value(), expectedImage.value(),
                              profile);
    capture.metrics = comparison.metrics;
    capture.failedThresholds = comparison.failedThresholds;
    if (!comparison.compatible) {
      capture.status = "runtime_error";
      capture.statusReason = "comparison_incompatible";
      result.exitCode = SnapshotExitCode::InvalidInput;
      continue;
    }
    if (comparison.passed) {
      capture.status = "pass";
      capture.statusReason = "within_thresholds";
    } else {
      capture.status = "fail";
      capture.statusReason = "thresholds_failed";
      mismatch = true;
      const std::filesystem::path diffPath =
          caseDir / "diff" / (capture.target + "_diff.png");
      auto diff = writeSnapshotDiffPng(actualImage.value(),
                                       expectedImage.value(), diffPath);
      if (!diff.hasError()) {
        capture.diff = relativeToCaseDir(caseDir, diffPath);
      }
    }
  }
  const std::filesystem::path htmlPath =
      options.htmlOut.empty() ? caseDir / "report.html" : options.htmlOut;
  if (result.exitCode == SnapshotExitCode::Success) {
    if (mismatch) {
      result.exitCode = SnapshotExitCode::VisualMismatch;
      result.message = "snapshot mismatch";
    } else if (missingBaseline) {
      result.exitCode = SnapshotExitCode::MissingBaseline;
      result.message = "snapshot baseline missing";
    } else {
      result.message = "snapshots matched";
    }
  }
  writeReports(result, report, reportPath, htmlPath);
  result.report = std::move(report);
  return result;
}

SnapshotRunResult runSnapshotCase(SnapshotCase snapshotCase,
                                  const SnapshotRunOptions &options) {
  SnapshotRunResult capture = captureSnapshotCase(snapshotCase, options);
  if (capture.exitCode != SnapshotExitCode::Success) {
    return capture;
  }
  SnapshotRunOptions compareOptions = options;
  compareOptions.artifactDir = capture.report.artifacts.artifactDir;
  compareOptions.jsonOut = capture.reportPath;
  return compareSnapshotCase(std::move(snapshotCase), compareOptions);
}

SnapshotSuiteRunResult
runSnapshotSuite(std::vector<SnapshotCase> snapshotCases,
                 std::string_view suite, const SnapshotRunOptions &options) {
  SnapshotSuiteRunResult suiteResult{};
  for (SnapshotCase &snapshotCase : snapshotCases) {
    if (snapshotCase.suite != suite) {
      continue;
    }
    SnapshotRunOptions caseOptions = options;
    caseOptions.jsonOut.clear();
    caseOptions.htmlOut.clear();
    SnapshotRunResult result = runSnapshotCase(std::move(snapshotCase),
                                               caseOptions);
    if (static_cast<int>(result.exitCode) >
        static_cast<int>(suiteResult.exitCode)) {
      suiteResult.exitCode = result.exitCode;
    }
    suiteResult.caseResults.push_back(std::move(result));
  }
  std::vector<SnapshotReport> reports;
  reports.reserve(suiteResult.caseResults.size());
  for (const SnapshotRunResult &caseResult : suiteResult.caseResults) {
    reports.push_back(caseResult.report);
  }
  const std::filesystem::path artifactDir =
      options.artifactDir.empty()
          ? snapshotRepoRoot() / "artifacts" / "snapshots" /
                utcTimestampForPath()
          : options.artifactDir;
  suiteResult.reportPath =
      options.jsonOut.empty() ? artifactDir / "run.json" : options.jsonOut;
  suiteResult.htmlPath =
      options.htmlOut.empty() ? artifactDir / "index.html" : options.htmlOut;
  std::filesystem::create_directories(suiteResult.reportPath.parent_path());
  std::ofstream json(suiteResult.reportPath, std::ios::binary);
  json << "{\n  \"kind\": \"nuri.snapshot.suite_report\",\n"
          "  \"caseReports\": [\n";
  for (size_t i = 0u; i < suiteResult.caseResults.size(); ++i) {
    if (i != 0u) {
      json << ",\n";
    }
    json << "    \"" << suiteResult.caseResults[i].reportPath.generic_string()
         << "\"";
  }
  json << "\n  ]\n}\n";
  (void)writeSnapshotSuiteHtmlFile(reports, suite, suiteResult.htmlPath);
  suiteResult.message = "suite run complete";
  return suiteResult;
}

} // namespace nuri::tools::snapshot
