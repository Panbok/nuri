#include "nuri/tools/runtime/render_tool_runtime.h"

#include "nuri/core/runtime_config.h"
#include "nuri/core/window.h"
#include "nuri/gfx/pipeline/default_render_pipeline.h"
#include "nuri/resources/scene_importer.h"

#include <cstdlib>
#include <memory>
#include <optional>
#include <vector>

#if defined(_WIN32)
#include <stdlib.h>
#endif

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace nuri::tools::runtime {
namespace {

class ScopedEnvVar final {
public:
  ScopedEnvVar(std::string name, std::string value)
      : name_(std::move(name)), oldValue_(get(name_)),
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
  [[nodiscard]] static std::string get(const std::string &name) {
#if defined(_WIN32)
    char *buffer = nullptr;
    size_t length = 0u;
    if (_dupenv_s(&buffer, &length, name.c_str()) != 0 || buffer == nullptr) {
      return {};
    }
    std::string out(buffer);
    std::free(buffer);
    return out;
#else
    const char *value = std::getenv(name.c_str());
    return value == nullptr ? std::string{} : std::string(value);
#endif
  }

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

  std::string name_{};
  std::string oldValue_{};
  bool hadOldValue_ = false;
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

[[nodiscard]] Result<std::filesystem::path, std::string>
resolveToolPath(const ToolRuntimeDesc &runtime, std::string_view base,
                const std::filesystem::path &path) {
  if (runtime.resolvePath == nullptr) {
    return Result<std::filesystem::path, std::string>::makeError(
        "tool runtime path resolver is not configured");
  }
  return runtime.resolvePath(base, path);
}

[[nodiscard]] Result<bool, std::string>
populateTransmissionTransparencyScene(const ToolRuntimeDesc &runtime,
                                      Renderer &renderer, RenderScene &scene) {
  if (runtime.scene.generator !=
      "nuri.procedural.transmission_transparency.v1") {
    return Result<bool, std::string>::makeResult(true);
  }

  auto planePath = resolveToolPath(runtime, "modelsRoot",
                                   "common/flat_plane.obj");
  if (planePath.hasError()) {
    return Result<bool, std::string>::makeError(planePath.error());
  }
  auto modelResult = renderer.resources().acquireModel(ModelRequest{
      .path = planePath.value().string(),
      .debugName = "tool_transmission_flat_plane",
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
      acquireMaterial("tool_transmission_background",
                      glm::vec4(0.03f, 0.12f, 0.24f, 1.0f),
                      MaterialAlphaMode::Opaque, 0.0f);
  auto warmMaterial =
      acquireMaterial("tool_transmission_warm_band",
                      glm::vec4(1.0f, 0.34f, 0.10f, 1.0f),
                      MaterialAlphaMode::Opaque, 0.0f);
  auto coolMaterial =
      acquireMaterial("tool_transmission_cool_band",
                      glm::vec4(0.08f, 0.72f, 0.95f, 1.0f),
                      MaterialAlphaMode::Opaque, 0.0f);
  auto transparentMaterial =
      acquireMaterial("tool_plain_transparent",
                      glm::vec4(1.0f, 0.82f, 0.20f, 0.46f),
                      MaterialAlphaMode::Blend, 0.0f);
  auto transmissionMaterial =
      acquireMaterial("tool_blended_transmission",
                      glm::vec4(0.52f, 0.96f, 1.0f, 0.42f),
                      MaterialAlphaMode::Blend, 1.0f);
  for (const auto *material :
       {&backgroundMaterial, &warmMaterial, &coolMaterial,
        &transparentMaterial, &transmissionMaterial}) {
    if (material->hasError()) {
      return Result<bool, std::string>::makeError(material->error());
    }
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
  return addPlane("BlendedTransmissionInFront",
                  glm::vec3(0.34f, -0.02f, 0.25f),
                  glm::vec3(2.0f, 1.9f, 1.0f),
                  transmissionMaterial.value());
}

[[nodiscard]] Result<bool, std::string>
populateReactiveMaskScene(const ToolRuntimeDesc &runtime, Renderer &renderer,
                          RenderScene &scene) {
  if (runtime.scene.generator != "nuri.procedural.reactive_mask.v1") {
    return Result<bool, std::string>::makeResult(true);
  }

  auto planePath = resolveToolPath(runtime, "modelsRoot",
                                   "common/flat_plane.obj");
  if (planePath.hasError()) {
    return Result<bool, std::string>::makeError(planePath.error());
  }
  auto modelResult = renderer.resources().acquireModel(ModelRequest{
      .path = planePath.value().string(),
      .debugName = "tool_reactive_flat_plane",
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
      acquireMaterial("tool_reactive_background",
                      glm::vec4(0.04f, 0.12f, 0.22f, 1.0f),
                      MaterialAlphaMode::Opaque);
  auto maskedMaterial =
      acquireMaterial("tool_alpha_mask_reactive",
                      glm::vec4(0.32f, 1.0f, 0.48f, 0.85f),
                      MaterialAlphaMode::Mask);
  if (backgroundMaterial.hasError()) {
    return Result<bool, std::string>::makeError(backgroundMaterial.error());
  }
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
                  glm::vec3(0.95f, 0.95f, 1.0f),
                  maskedMaterial.value());
}

[[nodiscard]] Result<bool, std::string>
populateShadowPlanesScene(const ToolRuntimeDesc &runtime, Renderer &renderer,
                          RenderScene &scene) {
  if (runtime.scene.generator != "nuri.procedural.shadow_planes.v1") {
    return Result<bool, std::string>::makeResult(true);
  }

  auto planePath = resolveToolPath(runtime, "modelsRoot",
                                   "common/flat_plane.obj");
  if (planePath.hasError()) {
    return Result<bool, std::string>::makeError(planePath.error());
  }
  auto modelResult = renderer.resources().acquireModel(ModelRequest{
      .path = planePath.value().string(),
      .debugName = "tool_shadow_flat_plane",
  });
  if (modelResult.hasError()) {
    return Result<bool, std::string>::makeError(modelResult.error());
  }

  auto acquireMaterial = [&](std::string_view name,
                             const glm::vec4 &color)
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

  auto floorMaterial = acquireMaterial(
      "tool_shadow_floor", glm::vec4(0.72f, 0.70f, 0.64f, 1.0f));
  auto wallMaterial = acquireMaterial(
      "tool_shadow_wall", glm::vec4(0.58f, 0.68f, 0.78f, 1.0f));
  auto redMaterial = acquireMaterial(
      "tool_shadow_red", glm::vec4(0.86f, 0.18f, 0.12f, 1.0f));
  auto blueMaterial = acquireMaterial(
      "tool_shadow_blue", glm::vec4(0.12f, 0.36f, 0.88f, 1.0f));
  auto greenMaterial = acquireMaterial(
      "tool_shadow_green", glm::vec4(0.16f, 0.72f, 0.34f, 1.0f));
  for (const auto *material :
       {&floorMaterial, &wallMaterial, &redMaterial, &blueMaterial,
        &greenMaterial}) {
    if (material->hasError()) {
      return Result<bool, std::string>::makeError(material->error());
    }
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
                  glm::vec3(0.82f, 1.34f, 1.0f),
                  greenMaterial.value());
}

[[nodiscard]] Result<bool, std::string>
populateToolScene(const ToolRuntimeDesc &runtime, Renderer &renderer,
                  RenderScene &scene, std::pmr::memory_resource *memory,
                  std::optional<ScenePrefab> &prefab,
                  std::optional<ScenePrefabAssets> &prefabAssets) {
  scene.bindResources(&renderer.resources());
  LightDesc keyLight{.type = LightType::Directional,
                     .name = "tool_key",
                     .color = glm::vec3(1.0f),
                     .intensity = 4.0f,
                     .enabled = true};
  if (runtime.scene.generator == "nuri.procedural.shadow_planes.v1") {
    keyLight.rotation = glm::quatLookAt(
        glm::normalize(glm::vec3(-0.45f, -0.78f, -0.44f)),
        glm::vec3(0.0f, 1.0f, 0.0f));
    keyLight.intensity = 5.5f;
  }
  auto lightResult =
      scene.graph().addLight(scene.graph().rootNode(), keyLight);
  if (lightResult.hasError()) {
    return Result<bool, std::string>::makeError(lightResult.error());
  }

  if (runtime.scene.kind == "prefab") {
    if (runtime.scene.pathBase.empty() || runtime.scene.path.empty()) {
      return Result<bool, std::string>::makeError(
          "prefab scene requires pathBase and path");
    }
    auto path = resolveToolPath(runtime, runtime.scene.pathBase,
                                runtime.scene.path);
    if (path.hasError()) {
      return Result<bool, std::string>::makeError(path.error());
    }
    if (!std::filesystem::exists(path.value())) {
      return Result<bool, std::string>::makeError(
          "missing scene asset: " + path.value().string());
    }
    SceneImportOptions importOptions{};
    importOptions.assetBuildOptions.flipUVs = runtime.scene.flipUVs;
    importOptions.assetBuildOptions.generateMeshlets =
        runtime.scene.generateMeshlets;
    importOptions.assetBuildOptions.meshletMaxVertices =
        runtime.scene.meshletMaxVertices;
    importOptions.assetBuildOptions.meshletMaxPrimitives =
        runtime.scene.meshletMaxPrimitives;
    importOptions.assetBuildOptions.meshletConeWeight =
        runtime.scene.meshletConeWeight;
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
  } else if (runtime.scene.kind == "procedural") {
    auto proceduralResult =
        populateTransmissionTransparencyScene(runtime, renderer, scene);
    if (proceduralResult.hasError()) {
      return proceduralResult;
    }
    proceduralResult = populateReactiveMaskScene(runtime, renderer, scene);
    if (proceduralResult.hasError()) {
      return proceduralResult;
    }
    proceduralResult = populateShadowPlanesScene(runtime, renderer, scene);
    if (proceduralResult.hasError()) {
      return proceduralResult;
    }
  }

  auto syncResult = scene.graph().syncWorldTransforms();
  (void)syncResult;
  return scene.commit();
}

} // namespace

struct ToolRendererRuntime::Impl {
  explicit Impl(const ToolRuntimeDesc &desc)
      : pipeline(&pipelineMemory), scene(&sceneMemory) {
    env.push_back(std::make_unique<ScopedEnvVar>(
        "NURI_RENDER_GRAPH_WORKER_COUNT",
        std::to_string(desc.renderGraph.workerCount)));
    env.push_back(std::make_unique<ScopedEnvVar>(
        "NURI_RENDER_GRAPH_DISABLE_PARALLEL_COMPILE",
        desc.renderGraph.parallelCompile ? "0" : "1"));
    env.push_back(std::make_unique<ScopedEnvVar>(
        "NURI_RENDER_GRAPH_DISABLE_PARALLEL_RECORDING",
        desc.renderGraph.parallelRecording ? "0" : "1"));
    if (desc.presentMode != "default") {
      env.push_back(
          std::make_unique<ScopedEnvVar>("NURI_PRESENT_MODE", desc.presentMode));
    }
  }

  std::vector<std::unique_ptr<ScopedEnvVar>> env{};
  std::pmr::unsynchronized_pool_resource rendererMemory{};
  std::pmr::unsynchronized_pool_resource pipelineMemory{};
  std::pmr::unsynchronized_pool_resource sceneMemory{};
  std::unique_ptr<Window> window{};
  std::unique_ptr<GPUDevice> gpu{};
  std::unique_ptr<Renderer> renderer{};
  RenderPipeline pipeline;
  RenderScene scene;
  std::optional<ScenePrefab> prefab{};
  std::optional<ScenePrefabAssets> prefabAssets{};
  TemporalCameraHistoryState cameraHistory{};
  RenderFrameContext frameContext{};
};

ToolRendererRuntime::ToolRendererRuntime(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

ToolRendererRuntime::~ToolRendererRuntime() = default;

Window &ToolRendererRuntime::window() noexcept { return *impl_->window; }
GPUDevice &ToolRendererRuntime::gpu() noexcept { return *impl_->gpu; }
Renderer &ToolRendererRuntime::renderer() noexcept { return *impl_->renderer; }
RenderPipeline &ToolRendererRuntime::pipeline() noexcept {
  return impl_->pipeline;
}
RenderScene &ToolRendererRuntime::scene() noexcept { return impl_->scene; }
RenderFrameContext &ToolRendererRuntime::frameContext() noexcept {
  return impl_->frameContext;
}
TemporalCameraHistoryState &ToolRendererRuntime::cameraHistory() noexcept {
  return impl_->cameraHistory;
}
uint32_t ToolRendererRuntime::swapchainImageCount() const noexcept {
  return impl_->gpu != nullptr ? impl_->gpu->getSwapchainImageCount() : 0u;
}
Result<bool, std::string> ToolRendererRuntime::commitScene() {
  return impl_->scene.commit();
}

Result<std::unique_ptr<ToolRendererRuntime>, std::string>
createToolRendererRuntime(const ToolRuntimeDesc &desc) {
  auto configResult = loadRuntimeConfigFromEnvOrDefault();
  if (configResult.hasError()) {
    return Result<std::unique_ptr<ToolRendererRuntime>, std::string>::makeError(
        configResult.error());
  }
  RuntimeConfig config = std::move(configResult.value());
  config.window.title = desc.title;
  config.window.width = static_cast<int32_t>(desc.resolution[0]);
  config.window.height = static_cast<int32_t>(desc.resolution[1]);
  config.window.mode = WindowMode::Windowed;

  auto impl = std::unique_ptr<ToolRendererRuntime::Impl>(
      new ToolRendererRuntime::Impl(desc));
  impl->window = Window::create(config.window.title, config.window.width,
                                config.window.height, config.window.mode);
  if (!impl->window) {
    return Result<std::unique_ptr<ToolRendererRuntime>, std::string>::makeError(
        "failed to create tool window");
  }
  GPUDeviceCreateDesc deviceDesc{};
  deviceDesc.backend = backendPreference(desc.backend);
  impl->gpu = GPUDevice::create(*impl->window, deviceDesc);
  if (!impl->gpu) {
    return Result<std::unique_ptr<ToolRendererRuntime>, std::string>::makeError(
        "failed to create GPU device");
  }
  impl->renderer = Renderer::create(*impl->gpu, impl->rendererMemory);
  auto pipelineResult = registerDefaultRenderPipeline(
      impl->pipeline, *impl->gpu, config.shaders, &impl->pipelineMemory);
  if (pipelineResult.hasError()) {
    return Result<std::unique_ptr<ToolRendererRuntime>, std::string>::makeError(
        pipelineResult.error());
  }
  auto sceneResult = populateToolScene(desc, *impl->renderer, impl->scene,
                                       &impl->sceneMemory, impl->prefab,
                                       impl->prefabAssets);
  if (sceneResult.hasError()) {
    return Result<std::unique_ptr<ToolRendererRuntime>, std::string>::makeError(
        sceneResult.error());
  }
  return Result<std::unique_ptr<ToolRendererRuntime>, std::string>::makeResult(
      std::unique_ptr<ToolRendererRuntime>(
          new ToolRendererRuntime(std::move(impl))));
}

Camera makeToolCamera(const ToolCameraDesc &desc) {
  Camera camera;
  camera.setPerspective(PerspectiveParams{
      .fovYRadians = glm::radians(desc.verticalFovDegrees),
      .nearPlane = desc.nearPlane,
      .farPlane = desc.farPlane,
  });
  const glm::vec3 direction =
      desc.hasTarget && glm::length(desc.target - desc.position) > 1.0e-6f
          ? glm::normalize(desc.target - desc.position)
          : (glm::length(desc.direction) > 1.0e-6f
                 ? glm::normalize(desc.direction)
                 : glm::vec3(0.0f, 0.0f, -1.0f));
  camera.setLookAt(desc.position, desc.position + direction,
                   glm::vec3(0.0f, 1.0f, 0.0f));
  return camera;
}

void buildToolFrameContext(RenderFrameContext &frameContext, RenderScene &scene,
                           Renderer &renderer, RenderSettings &settings,
                           TemporalCameraHistoryState &cameraHistory,
                           const Camera &camera, const ToolFrameDesc &desc) {
  frameContext.scene = &scene;
  frameContext.resources = &renderer.resources();
  frameContext.frameIndex = desc.frameIndex;
  const MaterialTableSnapshot materialSnapshot =
      renderer.resources().materialSnapshot();
  const TemporalSceneContentState sceneContent{
      .lightTopologyVersion = scene.lightTopologyVersion(),
      .lightTransformVersion = scene.lightTransformVersion(),
      .materialTableVersion = materialSnapshot.version,
      .environmentVersion = scene.environmentVersion(),
  };
  frameContext.camera = makeTemporalCameraFrameState(
      camera, static_cast<float>(desc.width) / static_cast<float>(desc.height),
      settings.antiAliasing,
      TemporalCameraFrameDesc{.renderExtent = glm::uvec2(desc.width, desc.height),
                              .sceneContent = sceneContent,
                              .cameraCutRequested =
                                  desc.cameraCutRequested},
      cameraHistory);
  settings.antiAliasing.debug.resetHistoryRequested = false;
  frameContext.settings = &settings;
  frameContext.metrics = {};
  frameContext.metrics.frameIndex = frameContext.frameIndex;
  frameContext.metrics.antiAliasing =
      makeAntiAliasingFrameMetrics(frameContext.camera);
  frameContext.sharedDepthTexture = {};
  frameContext.timeSeconds = desc.timeSeconds;
  frameContext.deltaSeconds = desc.deltaSeconds;
}

} // namespace nuri::tools::runtime
