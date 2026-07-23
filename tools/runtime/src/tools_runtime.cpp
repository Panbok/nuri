#include "nuri/tools/runtime/render_tool_runtime.h"

#include "nuri/core/runtime_config.h"
#include "nuri/core/window.h"
#include "nuri/gfx/frame/presentation_aa_plan.h"
#include "nuri/gfx/pipeline/default_render_pipeline.h"
#include "nuri/gfx/sim/animation_scene_frame_data.h"
#include "nuri/resources/gpu/resource_manager.h"
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

class ToolAnimationFrameProvider final {
public:
  explicit ToolAnimationFrameProvider(
      const std::optional<AnimationSceneFrameData> &frameData) noexcept
      : frameData_(&frameData) {}

  [[nodiscard]] Result<bool, std::string> prepare(FrameBuildContext &ctx) {
    if (frameData_ != nullptr && frameData_->has_value()) {
      ctx.shared.animationSceneGpuData = **frameData_;
    }
    return Result<bool, std::string>::makeResult(true);
  }

private:
  const std::optional<AnimationSceneFrameData> *frameData_ = nullptr;
};

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

[[nodiscard]] Result<std::filesystem::path, std::string>
resolveToolPath(const ToolRuntimeDesc &runtime, std::string_view base,
                const std::filesystem::path &path) {
  if (runtime.resolvePath == nullptr) {
    return Result<std::filesystem::path, std::string>::makeError(
        "tool runtime path resolver is not configured");
  }
  return runtime.resolvePath(base, path);
}

[[nodiscard]] Result<TextureRequestKind, std::string>
parseToolTextureKind(std::string_view kind) {
  if (kind == "Texture2D") {
    return Result<TextureRequestKind, std::string>::makeResult(
        TextureRequestKind::Texture2D);
  }
  if (kind == "Ktx2Texture2D") {
    return Result<TextureRequestKind, std::string>::makeResult(
        TextureRequestKind::Ktx2Texture2D);
  }
  if (kind == "Ktx2Cubemap") {
    return Result<TextureRequestKind, std::string>::makeResult(
        TextureRequestKind::Ktx2Cubemap);
  }
  if (kind == "EquirectHdrCubemap") {
    return Result<TextureRequestKind, std::string>::makeResult(
        TextureRequestKind::EquirectHdrCubemap);
  }
  return Result<TextureRequestKind, std::string>::makeError(
      "unsupported tool environment texture kind '" + std::string(kind) + "'");
}

[[nodiscard]] Result<std::optional<TextureRequest>, std::string>
makeToolEnvironmentTextureRequest(const ToolRuntimeDesc &runtime,
                                  const ToolEnvironmentTextureDesc &desc,
                                  std::string_view fallbackDebugName) {
  if (!desc.enabled) {
    return Result<std::optional<TextureRequest>, std::string>::makeResult(
        std::nullopt);
  }
  auto path = resolveToolPath(runtime, desc.pathBase, desc.path);
  if (path.hasError()) {
    return Result<std::optional<TextureRequest>, std::string>::makeError(
        path.error());
  }
  if (!std::filesystem::exists(path.value())) {
    if (desc.required) {
      return Result<std::optional<TextureRequest>, std::string>::makeError(
          "missing required environment texture: " + path.value().string());
    }
    return Result<std::optional<TextureRequest>, std::string>::makeResult(
        std::nullopt);
  }
  auto kind = parseToolTextureKind(desc.kind);
  if (kind.hasError()) {
    return Result<std::optional<TextureRequest>, std::string>::makeError(
        kind.error());
  }
  return Result<std::optional<TextureRequest>, std::string>::makeResult(
      TextureRequest{
          .path = path.value().string(),
          .kind = kind.value(),
          .debugName = desc.debugName.empty() ? std::string(fallbackDebugName)
                                              : desc.debugName,
      });
}

[[nodiscard]] Result<bool, std::string>
requestToolEnvironment(const ToolRuntimeDesc &runtime, Renderer &renderer,
                       RenderScene &scene,
                       EnvironmentAssetHandle &outEnvironment) {
  auto cubemap = makeToolEnvironmentTextureRequest(
      runtime, runtime.environment.cubemap, "tool_environment_cubemap");
  if (cubemap.hasError()) {
    return Result<bool, std::string>::makeError(cubemap.error());
  }
  auto irradiance = makeToolEnvironmentTextureRequest(
      runtime, runtime.environment.irradiance, "tool_environment_irradiance");
  if (irradiance.hasError()) {
    return Result<bool, std::string>::makeError(irradiance.error());
  }
  auto prefilteredGgx = makeToolEnvironmentTextureRequest(
      runtime, runtime.environment.prefilteredGgx,
      "tool_environment_prefiltered_ggx");
  if (prefilteredGgx.hasError()) {
    return Result<bool, std::string>::makeError(prefilteredGgx.error());
  }
  auto prefilteredCharlie = makeToolEnvironmentTextureRequest(
      runtime, runtime.environment.prefilteredCharlie,
      "tool_environment_prefiltered_charlie");
  if (prefilteredCharlie.hasError()) {
    return Result<bool, std::string>::makeError(prefilteredCharlie.error());
  }
  auto brdfLut = makeToolEnvironmentTextureRequest(
      runtime, runtime.environment.brdfLut, "tool_environment_brdf_lut");
  if (brdfLut.hasError()) {
    return Result<bool, std::string>::makeError(brdfLut.error());
  }

  if (!cubemap.value().has_value() && !irradiance.value().has_value() &&
      !prefilteredGgx.value().has_value() &&
      !prefilteredCharlie.value().has_value() && !brdfLut.value().has_value()) {
    scene.setEnvironment(EnvironmentHandles{});
    outEnvironment = {};
    return Result<bool, std::string>::makeResult(true);
  }
  auto requested = renderer.assets().requestEnvironment(EnvironmentAssetRequest{
      .textures = {std::move(cubemap.value()), std::move(irradiance.value()),
                   std::move(prefilteredGgx.value()),
                   std::move(prefilteredCharlie.value()),
                   std::move(brdfLut.value())},
      .priority = AssetPriority::Critical,
      .optionalTextures = {!runtime.environment.cubemap.required,
                           !runtime.environment.irradiance.required,
                           !runtime.environment.prefilteredGgx.required,
                           !runtime.environment.prefilteredCharlie.required,
                           !runtime.environment.brdfLut.required},
      .debugName = "tool_environment",
  });
  if (requested.hasError()) {
    return Result<bool, std::string>::makeError(requested.error());
  }
  outEnvironment = requested.value();
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<bool, std::string>
populateTransmissionTransparencyScene(const ToolRuntimeDesc &runtime,
                                      Renderer &renderer, RenderScene &scene) {
  if (runtime.scene.generator !=
      "nuri.procedural.transmission_transparency.v1") {
    return Result<bool, std::string>::makeResult(true);
  }

  auto planePath =
      resolveToolPath(runtime, "modelsRoot", "common/flat_plane.obj");
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

  auto backgroundMaterial = acquireMaterial(
      "tool_transmission_background", glm::vec4(0.03f, 0.12f, 0.24f, 1.0f),
      MaterialAlphaMode::Opaque, 0.0f);
  auto warmMaterial = acquireMaterial("tool_transmission_warm_band",
                                      glm::vec4(1.0f, 0.34f, 0.10f, 1.0f),
                                      MaterialAlphaMode::Opaque, 0.0f);
  auto coolMaterial = acquireMaterial("tool_transmission_cool_band",
                                      glm::vec4(0.08f, 0.72f, 0.95f, 1.0f),
                                      MaterialAlphaMode::Opaque, 0.0f);
  auto transparentMaterial = acquireMaterial(
      "tool_plain_transparent", glm::vec4(1.0f, 0.82f, 0.20f, 0.46f),
      MaterialAlphaMode::Blend, 0.0f);
  auto transmissionMaterial = acquireMaterial(
      "tool_blended_transmission", glm::vec4(0.52f, 0.96f, 1.0f, 0.42f),
      MaterialAlphaMode::Blend, 1.0f);
  for (const auto *material :
       {&backgroundMaterial, &warmMaterial, &coolMaterial, &transparentMaterial,
        &transmissionMaterial}) {
    if (material->hasError()) {
      return Result<bool, std::string>::makeError(material->error());
    }
  }

  const glm::mat4 upright = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f),
                                        glm::vec3(1.0f, 0.0f, 0.0f));
  auto addPlane = [&](std::string_view name, const glm::vec3 &translation,
                      const glm::vec3 &scale,
                      MaterialRef material) -> Result<bool, std::string> {
    const glm::mat4 transform = glm::translate(glm::mat4(1.0f), translation) *
                                upright * glm::scale(glm::mat4(1.0f), scale);
    auto nodeResult =
        scene.graph().createNode(scene.graph().rootNode(), name, transform);
    if (nodeResult.hasError()) {
      return Result<bool, std::string>::makeError(nodeResult.error());
    }
    auto renderableResult = scene.graph().addRenderable(
        nodeResult.value(), modelResult.value(), material);
    if (renderableResult.hasError()) {
      return Result<bool, std::string>::makeError(renderableResult.error());
    }
    return Result<bool, std::string>::makeResult(true);
  };

  auto result =
      addPlane("OpaqueBackground", glm::vec3(0.0f, 0.0f, -0.80f),
               glm::vec3(5.0f, 3.0f, 1.0f), backgroundMaterial.value());
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
                    glm::vec3(2.15f, 1.72f, 1.0f), transparentMaterial.value());
  if (result.hasError()) {
    return result;
  }
  return addPlane("BlendedTransmissionInFront", glm::vec3(0.34f, -0.02f, 0.25f),
                  glm::vec3(2.0f, 1.9f, 1.0f), transmissionMaterial.value());
}

[[nodiscard]] Result<bool, std::string>
populateReactiveMaskScene(const ToolRuntimeDesc &runtime, Renderer &renderer,
                          RenderScene &scene) {
  if (runtime.scene.generator != "nuri.procedural.reactive_mask.v1") {
    return Result<bool, std::string>::makeResult(true);
  }

  auto planePath =
      resolveToolPath(runtime, "modelsRoot", "common/flat_plane.obj");
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

  auto backgroundMaterial = acquireMaterial(
      "tool_reactive_background", glm::vec4(0.04f, 0.12f, 0.22f, 1.0f),
      MaterialAlphaMode::Opaque);
  auto maskedMaterial = acquireMaterial("tool_alpha_mask_reactive",
                                        glm::vec4(0.32f, 1.0f, 0.48f, 0.85f),
                                        MaterialAlphaMode::Mask);
  if (backgroundMaterial.hasError()) {
    return Result<bool, std::string>::makeError(backgroundMaterial.error());
  }
  if (maskedMaterial.hasError()) {
    return Result<bool, std::string>::makeError(maskedMaterial.error());
  }

  const glm::mat4 upright = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f),
                                        glm::vec3(1.0f, 0.0f, 0.0f));
  auto addPlane = [&](std::string_view name, const glm::vec3 &translation,
                      const glm::vec3 &scale,
                      MaterialRef material) -> Result<bool, std::string> {
    const glm::mat4 transform = glm::translate(glm::mat4(1.0f), translation) *
                                upright * glm::scale(glm::mat4(1.0f), scale);
    auto nodeResult =
        scene.graph().createNode(scene.graph().rootNode(), name, transform);
    if (nodeResult.hasError()) {
      return Result<bool, std::string>::makeError(nodeResult.error());
    }
    auto renderableResult = scene.graph().addRenderable(
        nodeResult.value(), modelResult.value(), material);
    if (renderableResult.hasError()) {
      return Result<bool, std::string>::makeError(renderableResult.error());
    }
    return Result<bool, std::string>::makeResult(true);
  };

  auto result =
      addPlane("ReactiveBackground", glm::vec3(0.0f, 0.0f, -0.65f),
               glm::vec3(4.6f, 2.8f, 1.0f), backgroundMaterial.value());
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
populateOcclusionWallScene(const ToolRuntimeDesc &runtime, Renderer &renderer,
                           RenderScene &scene) {
  if (runtime.scene.generator != "nuri.procedural.occlusion_wall.v1") {
    return Result<bool, std::string>::makeResult(true);
  }

  auto planePath =
      resolveToolPath(runtime, "modelsRoot", "common/flat_plane.obj");
  if (planePath.hasError()) {
    return Result<bool, std::string>::makeError(planePath.error());
  }
  auto modelResult = renderer.resources().acquireModel(ModelRequest{
      .path = planePath.value().string(),
      .debugName = "tool_occlusion_flat_plane",
  });
  if (modelResult.hasError()) {
    return Result<bool, std::string>::makeError(modelResult.error());
  }

  auto acquireMaterial =
      [&](std::string_view name,
          const glm::vec4 &color) -> Result<MaterialRef, std::string> {
    MaterialRequest request{};
    request.debugName = std::string(name);
    request.desc.baseColorFactor = color;
    request.desc.emissiveFactor = glm::vec3(color) * 0.55f;
    request.desc.emissiveStrength = 1.2f;
    request.desc.metallicFactor = 0.0f;
    request.desc.roughnessFactor = 0.6f;
    request.desc.doubleSided = true;
    return renderer.resources().acquireMaterial(request);
  };

  auto wallMaterial = acquireMaterial("tool_occlusion_wall",
                                      glm::vec4(0.16f, 0.38f, 0.72f, 1.0f));
  auto hiddenMaterial = acquireMaterial("tool_occlusion_hidden",
                                        glm::vec4(1.0f, 0.24f, 0.10f, 1.0f));
  if (wallMaterial.hasError()) {
    return Result<bool, std::string>::makeError(wallMaterial.error());
  }
  if (hiddenMaterial.hasError()) {
    return Result<bool, std::string>::makeError(hiddenMaterial.error());
  }

  const glm::mat4 upright = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f),
                                        glm::vec3(1.0f, 0.0f, 0.0f));
  auto addPlane = [&](std::string_view name, const glm::vec3 &translation,
                      const glm::vec3 &scale,
                      MaterialRef material) -> Result<bool, std::string> {
    const glm::mat4 transform = glm::translate(glm::mat4(1.0f), translation) *
                                upright * glm::scale(glm::mat4(1.0f), scale);
    auto nodeResult =
        scene.graph().createNode(scene.graph().rootNode(), name, transform);
    if (nodeResult.hasError()) {
      return Result<bool, std::string>::makeError(nodeResult.error());
    }
    auto renderableResult = scene.graph().addRenderable(
        nodeResult.value(), modelResult.value(), material);
    if (renderableResult.hasError()) {
      return Result<bool, std::string>::makeError(renderableResult.error());
    }
    return Result<bool, std::string>::makeResult(true);
  };

  auto result = addPlane("OcclusionHidden", glm::vec3(0.0f, 0.55f, -1.20f),
                         glm::vec3(0.36f, 1.0f, 0.36f), hiddenMaterial.value());
  if (result.hasError()) {
    return result;
  }
  return addPlane("OcclusionWall", glm::vec3(0.0f, 0.55f, 0.0f),
                  glm::vec3(20.0f, 1.0f, 20.0f), wallMaterial.value());
}

[[nodiscard]] Result<bool, std::string>
populateShadowPlanesScene(const ToolRuntimeDesc &runtime, Renderer &renderer,
                          RenderScene &scene) {
  const bool ddgiScene =
      runtime.scene.generator.starts_with("nuri.procedural.ddgi_") &&
      runtime.scene.generator.ends_with(".v1");
  if (runtime.scene.generator != "nuri.procedural.shadow_planes.v1" &&
      !ddgiScene) {
    return Result<bool, std::string>::makeResult(true);
  }

  auto planePath =
      resolveToolPath(runtime, "modelsRoot", "common/flat_plane.obj");
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

  auto acquireMaterial = [&](std::string_view name, const glm::vec4 &color,
                             bool doubleSided = true,
                             bool alphaMasked =
                                 false) -> Result<MaterialRef, std::string> {
    MaterialRequest request{};
    request.debugName = std::string(name);
    request.desc.baseColorFactor = color;
    request.desc.emissiveFactor = glm::vec3(color) * 0.45f;
    request.desc.emissiveStrength = 1.1f;
    request.desc.metallicFactor = 0.0f;
    request.desc.roughnessFactor = 0.72f;
    request.desc.doubleSided = doubleSided;
    request.desc.alphaMode =
        alphaMasked ? MaterialAlphaMode::Mask : MaterialAlphaMode::Opaque;
    request.desc.alphaCutoff = 0.5f;
    return renderer.resources().acquireMaterial(request);
  };

  const bool alphaParityScene =
      runtime.scene.generator ==
      "nuri.procedural.ddgi_alpha_mask_double_sided.v1";

  auto floorMaterial = acquireMaterial("tool_shadow_floor",
                                       glm::vec4(0.72f, 0.70f, 0.64f, 1.0f));
  auto wallMaterial =
      acquireMaterial("tool_shadow_wall", glm::vec4(0.58f, 0.68f, 0.78f, 1.0f),
                      !alphaParityScene);
  auto redMaterial = acquireMaterial(
      "tool_shadow_red",
      glm::vec4(0.86f, 0.18f, 0.12f, alphaParityScene ? 0.2f : 1.0f), true,
      alphaParityScene);
  auto blueMaterial =
      acquireMaterial("tool_shadow_blue", glm::vec4(0.12f, 0.36f, 0.88f, 1.0f));
  auto greenMaterial = acquireMaterial("tool_shadow_green",
                                       glm::vec4(0.16f, 0.72f, 0.34f, 1.0f));
  for (const auto *material : {&floorMaterial, &wallMaterial, &redMaterial,
                               &blueMaterial, &greenMaterial}) {
    if (material->hasError()) {
      return Result<bool, std::string>::makeError(material->error());
    }
  }

  const glm::mat4 upright = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f),
                                        glm::vec3(1.0f, 0.0f, 0.0f));
  auto addPlane = [&](std::string_view name, const glm::mat4 &orientation,
                      const glm::vec3 &translation, const glm::vec3 &scale,
                      MaterialRef material) -> Result<bool, std::string> {
    const glm::mat4 transform = glm::translate(glm::mat4(1.0f), translation) *
                                orientation *
                                glm::scale(glm::mat4(1.0f), scale);
    auto nodeResult =
        scene.graph().createNode(scene.graph().rootNode(), name, transform);
    if (nodeResult.hasError()) {
      return Result<bool, std::string>::makeError(nodeResult.error());
    }
    auto renderableResult = scene.graph().addRenderable(
        nodeResult.value(), modelResult.value(), material);
    if (renderableResult.hasError()) {
      return Result<bool, std::string>::makeError(renderableResult.error());
    }
    return Result<bool, std::string>::makeResult(true);
  };

  auto result =
      addPlane("ShadowFloor", glm::mat4(1.0f), glm::vec3(0.0f, -0.72f, 0.0f),
               glm::vec3(5.2f, 1.0f, 4.4f), floorMaterial.value());
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
  result =
      addPlane("ShadowGreenCaster", upright, glm::vec3(1.16f, 0.06f, 0.35f),
               glm::vec3(0.82f, 1.34f, 1.0f), greenMaterial.value());
  if (result.hasError() || !ddgiScene) {
    return result;
  }
  auto volumeNode = scene.graph().createNode(
      scene.graph().rootNode(), "DDGI Color Bleed Volume", glm::mat4(1.0f));
  if (volumeNode.hasError()) {
    return Result<bool, std::string>::makeError(volumeNode.error());
  }
  auto volume = scene.graph().addDDGIVolume(
      volumeNode.value(),
      DDGIVolumeDesc{
          .name = "DDGI Color Bleed Volume",
          .probeCounts = {6u, 4u, 6u},
          .probeSpacing = {0.85f, 0.85f, 0.85f},
          .blendDistance = 0.85f,
          .maxRayDistance = 12.0f,
          .mode = runtime.scene.generator ==
                          "nuri.procedural.ddgi_camera_tracking_scroll.v1"
                      ? DDGIVolumeMode::CameraTracked
                      : DDGIVolumeMode::Authored});
  if (volume.hasError()) {
    return Result<bool, std::string>::makeError(volume.error());
  }
  if (runtime.scene.generator ==
      "nuri.procedural.ddgi_eight_volume_selection.v1") {
    for (uint32_t index = 1u; index < 8u; ++index) {
      const glm::vec3 offset{
          (index & 1u) != 0u ? 0.18f : -0.18f,
          (index & 2u) != 0u ? 0.12f : -0.12f,
          (index & 4u) != 0u ? 0.16f : -0.16f,
      };
      auto overlapNode = scene.graph().createNode(
          scene.graph().rootNode(), "DDGI Eight-Way Overlap Volume",
          glm::translate(glm::mat4(1.0f), offset));
      if (overlapNode.hasError()) {
        return Result<bool, std::string>::makeError(overlapNode.error());
      }
      auto overlap = scene.graph().addDDGIVolume(
          overlapNode.value(), DDGIVolumeDesc{
                                   .name = "DDGI Eight-Way Overlap Volume",
                                   .probeCounts = {3u, 2u, 3u},
                                   .probeSpacing = {1.1f, 1.1f, 1.1f},
                                   .blendDistance = 0.5f,
                                   .maxRayDistance = 12.0f,
                                   .priority = static_cast<int32_t>(index),
                               });
      if (overlap.hasError()) {
        return Result<bool, std::string>::makeError(overlap.error());
      }
    }
  }
  if (runtime.scene.generator == "nuri.procedural.ddgi_dirty_light_region.v1") {
    auto localLight = scene.graph().addLight(
        scene.graph().rootNode(), LightDesc{
                                      .type = LightType::Point,
                                      .name = "DDGI Local Dirty Light",
                                      .position = glm::vec3(-1.1f, 0.7f, 0.15f),
                                      .color = glm::vec3(1.0f, 0.32f, 0.12f),
                                      .intensity = 2.0f,
                                      .range = 0.35f,
                                      .enabled = true,
                                  });
    if (localLight.hasError()) {
      return Result<bool, std::string>::makeError(localLight.error());
    }
  }
  if (runtime.scene.generator == "nuri.procedural.ddgi_failure_isolation.v1") {
    for (uint32_t index = 0u; index < 2u; ++index) {
      auto largeNode = scene.graph().createNode(
          scene.graph().rootNode(), "DDGI Large Volume", glm::mat4(1.0f));
      if (largeNode.hasError()) {
        return Result<bool, std::string>::makeError(largeNode.error());
      }
      auto large = scene.graph().addDDGIVolume(
          largeNode.value(),
          DDGIVolumeDesc{.name = "DDGI Large Volume",
                         .probeCounts = {64u, 16u, 64u},
                         .probeSpacing = {0.1f, 0.1f, 0.1f},
                         .blendDistance = 0.1f,
                         .maxRayDistance = 12.0f,
                         .priority = -static_cast<int32_t>(index + 1u)});
      if (large.hasError()) {
        return Result<bool, std::string>::makeError(large.error());
      }
    }
  }
  if (runtime.scene.generator ==
      "nuri.procedural.ddgi_resource_replacement.v1") {
    auto guardNode = scene.graph().createNode(
        scene.graph().rootNode(), "DDGI Replacement Budget Guard",
        glm::translate(glm::mat4(1.0f), glm::vec3(-8.0f, 0.0f, 0.0f)));
    if (guardNode.hasError()) {
      return Result<bool, std::string>::makeError(guardNode.error());
    }
    auto guard = scene.graph().addDDGIVolume(
        guardNode.value(),
        DDGIVolumeDesc{.name = "DDGI Replacement Budget Guard",
                       .probeCounts = {32u, 8u, 32u},
                       .probeSpacing = {0.1f, 0.1f, 0.1f},
                       .blendDistance = 0.1f,
                       .maxRayDistance = 12.0f,
                       .priority = -10});
    if (guard.hasError()) {
      return Result<bool, std::string>::makeError(guard.error());
    }
    auto replacementNode = scene.graph().createNode(
        scene.graph().rootNode(), "DDGI Replacement Target",
        glm::translate(glm::mat4(1.0f), glm::vec3(8.0f, 0.0f, 0.0f)));
    if (replacementNode.hasError()) {
      return Result<bool, std::string>::makeError(replacementNode.error());
    }
    auto replacement = scene.graph().addDDGIVolume(
        replacementNode.value(),
        DDGIVolumeDesc{.name = "DDGI Replacement Target",
                       .probeCounts = {5u, 4u, 5u},
                       .probeSpacing = {0.85f, 0.85f, 0.85f},
                       .blendDistance = 0.85f,
                       .maxRayDistance = 12.0f,
                       .priority = -5});
    if (replacement.hasError()) {
      return Result<bool, std::string>::makeError(replacement.error());
    }
  }
  if (runtime.scene.generator == "nuri.procedural.ddgi_overlap_fallback.v1") {
    auto overlapNode = scene.graph().createNode(
        scene.graph().rootNode(), "DDGI Priority Overlap Volume",
        glm::translate(glm::mat4(1.0f), glm::vec3(0.35f, 0.0f, 0.0f)));
    if (overlapNode.hasError()) {
      return Result<bool, std::string>::makeError(overlapNode.error());
    }
    auto overlap = scene.graph().addDDGIVolume(
        overlapNode.value(),
        DDGIVolumeDesc{.name = "DDGI Priority Overlap Volume",
                       .probeCounts = {5u, 4u, 5u},
                       .probeSpacing = {0.85f, 0.85f, 0.85f},
                       .blendDistance = 1.2f,
                       .maxRayDistance = 12.0f,
                       .priority = 10});
    if (overlap.hasError()) {
      return Result<bool, std::string>::makeError(overlap.error());
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

[[nodiscard]] Result<bool, std::string>
populateToolScene(const ToolRuntimeDesc &runtime, Renderer &renderer,
                  RenderScene &scene, std::pmr::memory_resource *memory,
                  SceneLoadHandle &sceneLoad,
                  EnvironmentAssetHandle &environmentLoad) {
  (void)memory;
  scene.bindResources(&renderer.resources());
  LightDesc keyLight{.type = LightType::Directional,
                     .name = "tool_key",
                     .color = glm::vec3(1.0f),
                     .intensity = 4.0f,
                     .enabled = true};
  if (runtime.scene.generator == "nuri.procedural.shadow_planes.v1" ||
      (runtime.scene.generator.starts_with("nuri.procedural.ddgi_") &&
       runtime.scene.generator.ends_with(".v1"))) {
    keyLight.rotation =
        glm::quatLookAt(glm::normalize(glm::vec3(-0.45f, -0.78f, -0.44f)),
                        glm::vec3(0.0f, 1.0f, 0.0f));
    keyLight.intensity = 5.5f;
  }
  auto lightResult = scene.graph().addLight(scene.graph().rootNode(), keyLight);
  if (lightResult.hasError()) {
    return Result<bool, std::string>::makeError(lightResult.error());
  }

  if (runtime.scene.kind == "prefab") {
    if (runtime.scene.pathBase.empty() || runtime.scene.path.empty()) {
      return Result<bool, std::string>::makeError(
          "prefab scene requires pathBase and path");
    }
    auto path =
        resolveToolPath(runtime, runtime.scene.pathBase, runtime.scene.path);
    if (path.hasError()) {
      return Result<bool, std::string>::makeError(path.error());
    }
    if (!std::filesystem::exists(path.value())) {
      return Result<bool, std::string>::makeError("missing scene asset: " +
                                                  path.value().string());
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
    auto requested = renderer.assets().requestScene(SceneLoadRequest{
        .path = path.value().string(),
        .importOptions = importOptions,
        .priority = AssetPriority::Critical,
        .publication = ScenePublicationPolicy::Progressive,
        .failurePolicy = SceneFailurePolicy::BestEffort,
        .debugName = runtime.scene.path.string(),
    });
    if (requested.hasError()) {
      return Result<bool, std::string>::makeError(requested.error());
    }
    sceneLoad = requested.value();
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
    proceduralResult = populateOcclusionWallScene(runtime, renderer, scene);
    if (proceduralResult.hasError()) {
      return proceduralResult;
    }
    proceduralResult = populateShadowPlanesScene(runtime, renderer, scene);
    if (proceduralResult.hasError()) {
      return proceduralResult;
    }
  }

  auto environmentResult =
      requestToolEnvironment(runtime, renderer, scene, environmentLoad);
  if (environmentResult.hasError()) {
    return environmentResult;
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
      env.push_back(std::make_unique<ScopedEnvVar>("NURI_PRESENT_MODE",
                                                   desc.presentMode));
    }
  }

  std::vector<std::unique_ptr<ScopedEnvVar>> env{};
  std::pmr::unsynchronized_pool_resource rendererMemory{};
  std::pmr::unsynchronized_pool_resource pipelineMemory{};
  std::pmr::unsynchronized_pool_resource sceneMemory{};
  std::unique_ptr<Window> window{};
  std::unique_ptr<GPUDevice> gpu{};
  std::unique_ptr<Renderer> renderer{};
  std::optional<AnimationSceneFrameData> externalAnimationSceneFrameData{};
  RenderPipeline pipeline;
  RenderScene scene;
  SceneLoadHandle sceneLoad{};
  EnvironmentAssetHandle environmentLoad{};
  TemporalFrameService temporalFrameService{};
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
TemporalFrameService &ToolRendererRuntime::temporalFrameService() noexcept {
  return impl_->temporalFrameService;
}
uint32_t ToolRendererRuntime::swapchainImageCount() const noexcept {
  return impl_->gpu != nullptr ? impl_->gpu->getSwapchainImageCount() : 0u;
}
ToolAssetLoadStatus ToolRendererRuntime::assetLoadStatus() const {
  ToolAssetLoadStatus status{};
  status.sceneRequested = isValid(impl_->sceneLoad);
  if (status.sceneRequested) {
    status.scene = impl_->renderer->assets().query(impl_->sceneLoad);
  }
  status.environmentRequested = isValid(impl_->environmentLoad);
  if (status.environmentRequested) {
    status.environment =
        impl_->renderer->assets().query(impl_->environmentLoad);
  }
  return status;
}
Result<AssetPublicationStats, std::string>
ToolRendererRuntime::pumpAssetLoads() {
  return impl_->renderer->assets().prepareFrame(AssetPublicationContext{
      .scene = &impl_->scene,
  });
}
Result<bool, std::string> ToolRendererRuntime::commitScene() {
  return impl_->scene.commit();
}
Result<std::array<uint32_t, 2>, std::string>
ToolRendererRuntime::resize(uint32_t width, uint32_t height) {
  if (width == 0u || height == 0u || width > static_cast<uint32_t>(INT32_MAX) ||
      height > static_cast<uint32_t>(INT32_MAX)) {
    return Result<std::array<uint32_t, 2>, std::string>::makeError(
        "tool runtime resize requires a non-zero int32 extent");
  }
  impl_->window->setWindowSize(static_cast<int32_t>(width),
                               static_cast<int32_t>(height));
  impl_->window->pollEvents();
  int32_t framebufferWidth = 0;
  int32_t framebufferHeight = 0;
  impl_->window->getFramebufferSize(framebufferWidth, framebufferHeight);
  if (framebufferWidth <= 0 || framebufferHeight <= 0) {
    return Result<std::array<uint32_t, 2>, std::string>::makeError(
        "tool runtime resize produced an empty framebuffer");
  }
  impl_->renderer->onResize(static_cast<uint32_t>(framebufferWidth),
                            static_cast<uint32_t>(framebufferHeight));
  return Result<std::array<uint32_t, 2>, std::string>::makeResult(
      {static_cast<uint32_t>(framebufferWidth),
       static_cast<uint32_t>(framebufferHeight)});
}
void ToolRendererRuntime::setExternalAnimationSceneFrameData(
    const AnimationSceneFrameData &frameData) noexcept {
  impl_->externalAnimationSceneFrameData = frameData;
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
  config.window.mode =
      desc.windowVisible ? WindowMode::Windowed : WindowMode::Hidden;

  auto impl = std::unique_ptr<ToolRendererRuntime::Impl>(
      new ToolRendererRuntime::Impl(desc));
  impl->window = Window::create(config.window.title, config.window.width,
                                config.window.height, config.window.mode);
  if (!impl->window) {
    return Result<std::unique_ptr<ToolRendererRuntime>, std::string>::makeError(
        "failed to create tool window");
  }
  impl->gpu = GPUDevice::create(*impl->window);
  if (!impl->gpu) {
    return Result<std::unique_ptr<ToolRendererRuntime>, std::string>::makeError(
        "failed to create GPU device");
  }
  if (const uint32_t samples = desc.requiredMsaaSamples.value_or(1u);
      samples != 1u) {
    const AntiAliasingMode mode =
        samples == 8u ? AntiAliasingMode::MSAA8x : AntiAliasingMode::MSAA4x;
    const PresentationAAUnsupportedReason reason =
        msaaUnsupportedReason(mode, impl->gpu->getMultisampleCapabilities());
    if (reason != PresentationAAUnsupportedReason::None) {
      return Result<std::unique_ptr<ToolRendererRuntime>, std::string>::
          makeError("required MSAA" + std::to_string(samples) +
                    "x capability unavailable: " +
                    std::string(presentationAAUnsupportedReasonName(reason)));
    }
  }
  impl->renderer = Renderer::create(*impl->gpu, impl->rendererMemory);
  impl->pipeline.addProvider(std::make_unique<ToolAnimationFrameProvider>(
      impl->externalAnimationSceneFrameData));
  auto pipelineResult = registerDefaultRenderPipeline(
      impl->pipeline, *impl->gpu, config.shaders, &impl->pipelineMemory);
  if (pipelineResult.hasError()) {
    return Result<std::unique_ptr<ToolRendererRuntime>, std::string>::makeError(
        pipelineResult.error());
  }
  auto sceneResult =
      populateToolScene(desc, *impl->renderer, impl->scene, &impl->sceneMemory,
                        impl->sceneLoad, impl->environmentLoad);
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
                           TemporalFrameService &temporalFrameService,
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
  auto planResult = buildPresentationAAPlan(
      settings, {}, renderer.resources().gpuMultisampleCapabilities());
  NURI_ASSERT(!planResult.hasError(), "Invalid presentation AA plan: %s",
              planResult.error().c_str());
  frameContext.presentationAA = planResult.value();
  auto cameraResult = temporalFrameService.prepareFrame(
      camera, static_cast<float>(desc.width) / static_cast<float>(desc.height),
      settings.antiAliasing, frameContext.presentationAA,
      TemporalCameraFrameDesc{.renderExtent =
                                  glm::uvec2(desc.width, desc.height),
                              .sceneContent = sceneContent,
                              .cameraCutRequested = desc.cameraCutRequested},
      desc.frameIndex, desc.timeSeconds, desc.deltaSeconds);
  NURI_ASSERT(!cameraResult.hasError(), "Temporal frame prepare failed: %s",
              cameraResult.error().c_str());
  frameContext.camera = cameraResult.value();
  frameContext.temporalFrameService = &temporalFrameService;
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
