#include "nuri/pch.h"

#include "nuri/scene/render_scene.h"

#include "nuri/core/profiling.h"
#include "nuri/math/light.h"
#include "nuri/math/utils.h"
#include "nuri/resources/gpu/resource_manager.h"

namespace nuri {
namespace {

constexpr uint32_t kMaxDirectionalLightCount = 4u;
constexpr uint32_t kMaxLocalLightCount = 64u;
constexpr glm::quat kIdentityRotation(1.0f, 0.0f, 0.0f, 0.0f);

template <typename Fn>
void forEachEnvironmentTextureRef(const EnvironmentHandles &handles, Fn &&fn) {
  fn(handles.cubemap);
  fn(handles.irradiance);
  fn(handles.prefilteredGgx);
  fn(handles.prefilteredCharlie);
  fn(handles.brdfLut);
}

[[nodiscard]] LightDesc sanitizeLightDesc(const LightDesc &desc) {
  LightDesc sanitized = desc;
  sanitized.position = nuri::sanitizeFiniteVec3(desc.position, glm::vec3(0.0f));
  sanitized.rotation = nuri::sanitizeRotation(desc.rotation);
  sanitized.color = glm::max(
      nuri::sanitizeFiniteVec3(desc.color, glm::vec3(1.0f)), glm::vec3(0.0f));
  sanitized.intensity = nuri::sanitizeNonNegative(desc.intensity, 1.0f);
  sanitized.enabled = desc.enabled;

  switch (sanitized.type) {
  case LightType::Directional:
    sanitized.range = 0.0f;
    sanitized.innerConeAngleRadians = 0.0f;
    sanitized.outerConeAngleRadians = 0.0f;
    break;
  case LightType::Point:
    sanitized.range = nuri::sanitizeNonNegative(desc.range, 0.0f);
    sanitized.innerConeAngleRadians = 0.0f;
    sanitized.outerConeAngleRadians = 0.0f;
    break;
  case LightType::Spot:
    sanitized.range = nuri::sanitizeNonNegative(desc.range, 0.0f);
    sanitized.outerConeAngleRadians =
        std::clamp(nuri::sanitizeNonNegative(desc.outerConeAngleRadians,
                                             glm::quarter_pi<float>()),
                   0.0f, glm::half_pi<float>() - 1.0e-4f);
    sanitized.innerConeAngleRadians =
        std::clamp(nuri::sanitizeNonNegative(desc.innerConeAngleRadians, 0.0f),
                   0.0f, sanitized.outerConeAngleRadians);
    break;
  }

  return sanitized;
}

[[nodiscard]] DirectionalLightGpuData
packDirectionalLight(const glm::quat &rotation, const glm::vec3 &color,
                     float intensity) {
  const glm::vec3 direction = nuri::lightDirectionFromRotation(rotation);
  return DirectionalLightGpuData{
      .directionIlluminance =
          glm::vec4(direction.x, direction.y, direction.z, intensity),
      .colorReserved = glm::vec4(color, 0.0f),
  };
}

[[nodiscard]] LocalLightGpuData packPointLight(const glm::vec3 &position,
                                               const glm::quat &rotation,
                                               const glm::vec3 &color,
                                               float intensity, float range) {
  const glm::vec3 direction = nuri::lightDirectionFromRotation(rotation);
  return LocalLightGpuData{
      .positionRange = glm::vec4(position, range),
      .directionOuterCos = glm::vec4(direction, -1.0f),
      .colorIntensity = glm::vec4(color, intensity),
      .innerCosTypeEnabledReserved =
          glm::uvec4(nuri::floatBitsToUint(-1.0f),
                     static_cast<uint32_t>(LocalLightGpuType::Point), 1u, 0u),
  };
}

[[nodiscard]] LocalLightGpuData
packSpotLight(const glm::vec3 &position, const glm::quat &rotation,
              const glm::vec3 &color, float intensity, float range,
              float innerConeAngleRadians, float outerConeAngleRadians) {
  const glm::vec3 direction = nuri::lightDirectionFromRotation(rotation);
  return LocalLightGpuData{
      .positionRange = glm::vec4(position, range),
      .directionOuterCos =
          glm::vec4(direction, std::cos(outerConeAngleRadians)),
      .colorIntensity = glm::vec4(color, intensity),
      .innerCosTypeEnabledReserved =
          glm::uvec4(nuri::floatBitsToUint(std::cos(innerConeAngleRadians)),
                     static_cast<uint32_t>(LocalLightGpuType::Spot), 1u, 0u),
  };
}

[[nodiscard]] uint32_t nextLightSlotIndex(std::pmr::vector<uint32_t> &freeSlots,
                                          size_t currentSize) {
  if (!freeSlots.empty()) {
    const uint32_t index = freeSlots.back();
    freeSlots.pop_back();
    return index;
  }
  return static_cast<uint32_t>(currentSize);
}

[[nodiscard]] uint32_t
liveSlotCount(size_t slotCount, const std::pmr::vector<uint32_t> &freeSlots) {
  return static_cast<uint32_t>(slotCount - freeSlots.size());
}

} // namespace

RenderScene::RenderScene(std::pmr::memory_resource *memory)
    : memory_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      renderables_(memory_), directionalLights_(memory_), pointLights_(memory_),
      spotLights_(memory_), packedLocalLights_(memory_),
      packedLocalLightIds_(memory_) {}

RenderScene::~RenderScene() {
  clearRenderables();
  clearLights();
  setEnvironment(EnvironmentHandles{});
}

Result<uint32_t, std::string>
RenderScene::addRenderable(ModelRef model, MaterialRef material,
                           const glm::mat4 &modelMatrix) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!isValid(model)) {
    return Result<uint32_t, std::string>::makeError(
        "RenderScene::addRenderable: model handle is invalid");
  }
  if (!isValid(material)) {
    return Result<uint32_t, std::string>::makeError(
        "RenderScene::addRenderable: material handle is invalid");
  }
  if (resources_ != nullptr) {
    if (resources_->tryGet(model) == nullptr) {
      return Result<uint32_t, std::string>::makeError(
          "RenderScene::addRenderable: model handle is stale");
    }
    if (resources_->tryGet(material) == nullptr) {
      return Result<uint32_t, std::string>::makeError(
          "RenderScene::addRenderable: material handle is stale");
    }
  }
  if (renderables_.size() >=
      static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    return Result<uint32_t, std::string>::makeError(
        "RenderScene::addRenderable: renderable count exceeds UINT32_MAX");
  }

  Renderable renderable{};
  renderable.model = model;
  renderable.material = material;
  renderable.modelMatrix = modelMatrix;

  renderables_.emplace_back(renderable);
  retainRenderable(renderable);
  ++topologyVersion_;
  ++transformVersion_;
  return Result<uint32_t, std::string>::makeResult(
      static_cast<uint32_t>(renderables_.size() - 1));
}

Result<uint32_t, std::string>
RenderScene::addRenderablesInstanced(ModelRef model, MaterialRef material,
                                     std::span<const glm::mat4> modelMatrices) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!isValid(model)) {
    return Result<uint32_t, std::string>::makeError(
        "RenderScene::addRenderablesInstanced: model handle is invalid");
  }
  if (!isValid(material)) {
    return Result<uint32_t, std::string>::makeError(
        "RenderScene::addRenderablesInstanced: material handle is invalid");
  }
  if (resources_ != nullptr) {
    if (resources_->tryGet(model) == nullptr) {
      return Result<uint32_t, std::string>::makeError(
          "RenderScene::addRenderablesInstanced: model handle is stale");
    }
    if (resources_->tryGet(material) == nullptr) {
      return Result<uint32_t, std::string>::makeError(
          "RenderScene::addRenderablesInstanced: material handle is stale");
    }
  }
  if (modelMatrices.empty()) {
    return Result<uint32_t, std::string>::makeError(
        "RenderScene::addRenderablesInstanced: modelMatrices is empty");
  }
  if (modelMatrices.size() >
      static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    return Result<uint32_t, std::string>::makeError(
        "RenderScene::addRenderablesInstanced: instance count exceeds "
        "UINT32_MAX");
  }

  const size_t startIndex = renderables_.size();
  const size_t requiredSize = startIndex + modelMatrices.size();
  if (requiredSize >
      static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    return Result<uint32_t, std::string>::makeError(
        "RenderScene::addRenderablesInstanced: total renderable count exceeds "
        "UINT32_MAX");
  }

  renderables_.reserve(requiredSize);
  for (const glm::mat4 &modelMatrix : modelMatrices) {
    Renderable renderable{};
    renderable.model = model;
    renderable.material = material;
    renderable.modelMatrix = modelMatrix;
    retainRenderable(renderable);
    renderables_.push_back(renderable);
  }
  ++topologyVersion_;
  ++transformVersion_;
  return Result<uint32_t, std::string>::makeResult(
      static_cast<uint32_t>(startIndex));
}

bool RenderScene::setRenderableTransform(uint32_t index,
                                         const glm::mat4 &modelMatrix) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (index >= renderables_.size()) {
    return false;
  }
  renderables_[index].modelMatrix = modelMatrix;
  ++transformVersion_;
  return true;
}

Result<LightId, std::string> RenderScene::addLight(const LightDesc &desc) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  const LightDesc sanitized = sanitizeLightDesc(desc);

  switch (sanitized.type) {
  case LightType::Directional: {
    if (liveSlotCount(directionalLights_.generations.size(),
                      directionalLights_.freeSlots) >=
        kMaxDirectionalLightCount) {
      return Result<LightId, std::string>::makeError(
          "RenderScene::addLight: directional light cap reached");
    }

    const uint32_t index = nextLightSlotIndex(
        directionalLights_.freeSlots, directionalLights_.generations.size());
    if (index == directionalLights_.generations.size()) {
      directionalLights_.generations.push_back(1u);
      directionalLights_.live.push_back(0u);
      directionalLights_.packedIndices.push_back(kInvalidPackedLightIndex);
      directionalLights_.names.emplace_back();
      directionalLights_.positions.push_back(glm::vec3(0.0f));
      directionalLights_.rotations.push_back(kIdentityRotation);
      directionalLights_.colors.push_back(glm::vec3(1.0f));
      directionalLights_.intensities.push_back(1.0f);
      directionalLights_.enabled.push_back(0u);
    }

    directionalLights_.live[index] = 1u;
    directionalLights_.packedIndices[index] = kInvalidPackedLightIndex;
    directionalLights_.names[index] = sanitized.name;
    directionalLights_.positions[index] = sanitized.position;
    directionalLights_.rotations[index] = sanitized.rotation;
    directionalLights_.colors[index] = sanitized.color;
    directionalLights_.intensities[index] = sanitized.intensity;
    directionalLights_.enabled[index] = sanitized.enabled ? 1u : 0u;

    rebuildPackedDirectionalLights();
    noteLightTopologyChanged();
    return Result<LightId, std::string>::makeResult(makeLightId(
        LightType::Directional, index, directionalLights_.generations[index]));
  }
  case LightType::Point: {
    if (liveSlotCount(pointLights_.generations.size(), pointLights_.freeSlots) +
            liveSlotCount(spotLights_.generations.size(),
                          spotLights_.freeSlots) >=
        kMaxLocalLightCount) {
      return Result<LightId, std::string>::makeError(
          "RenderScene::addLight: local light cap reached");
    }

    const uint32_t index = nextLightSlotIndex(pointLights_.freeSlots,
                                              pointLights_.generations.size());
    if (index == pointLights_.generations.size()) {
      pointLights_.generations.push_back(1u);
      pointLights_.live.push_back(0u);
      pointLights_.packedIndices.push_back(kInvalidPackedLightIndex);
      pointLights_.names.emplace_back();
      pointLights_.positions.push_back(glm::vec3(0.0f));
      pointLights_.rotations.push_back(kIdentityRotation);
      pointLights_.colors.push_back(glm::vec3(1.0f));
      pointLights_.intensities.push_back(1.0f);
      pointLights_.ranges.push_back(0.0f);
      pointLights_.enabled.push_back(0u);
    }

    pointLights_.live[index] = 1u;
    pointLights_.packedIndices[index] = kInvalidPackedLightIndex;
    pointLights_.names[index] = sanitized.name;
    pointLights_.positions[index] = sanitized.position;
    pointLights_.rotations[index] = sanitized.rotation;
    pointLights_.colors[index] = sanitized.color;
    pointLights_.intensities[index] = sanitized.intensity;
    pointLights_.ranges[index] = sanitized.range;
    pointLights_.enabled[index] = sanitized.enabled ? 1u : 0u;

    rebuildPackedLocalLights();
    noteLightTopologyChanged();
    return Result<LightId, std::string>::makeResult(
        makeLightId(LightType::Point, index, pointLights_.generations[index]));
  }
  case LightType::Spot: {
    if (liveSlotCount(pointLights_.generations.size(), pointLights_.freeSlots) +
            liveSlotCount(spotLights_.generations.size(),
                          spotLights_.freeSlots) >=
        kMaxLocalLightCount) {
      return Result<LightId, std::string>::makeError(
          "RenderScene::addLight: local light cap reached");
    }

    const uint32_t index = nextLightSlotIndex(spotLights_.freeSlots,
                                              spotLights_.generations.size());
    if (index == spotLights_.generations.size()) {
      spotLights_.generations.push_back(1u);
      spotLights_.live.push_back(0u);
      spotLights_.packedIndices.push_back(kInvalidPackedLightIndex);
      spotLights_.names.emplace_back();
      spotLights_.positions.push_back(glm::vec3(0.0f));
      spotLights_.rotations.push_back(kIdentityRotation);
      spotLights_.colors.push_back(glm::vec3(1.0f));
      spotLights_.intensities.push_back(1.0f);
      spotLights_.ranges.push_back(0.0f);
      spotLights_.innerConeAngles.push_back(0.0f);
      spotLights_.outerConeAngles.push_back(glm::quarter_pi<float>());
      spotLights_.enabled.push_back(0u);
    }

    spotLights_.live[index] = 1u;
    spotLights_.packedIndices[index] = kInvalidPackedLightIndex;
    spotLights_.names[index] = sanitized.name;
    spotLights_.positions[index] = sanitized.position;
    spotLights_.rotations[index] = sanitized.rotation;
    spotLights_.colors[index] = sanitized.color;
    spotLights_.intensities[index] = sanitized.intensity;
    spotLights_.ranges[index] = sanitized.range;
    spotLights_.innerConeAngles[index] = sanitized.innerConeAngleRadians;
    spotLights_.outerConeAngles[index] = sanitized.outerConeAngleRadians;
    spotLights_.enabled[index] = sanitized.enabled ? 1u : 0u;

    rebuildPackedLocalLights();
    noteLightTopologyChanged();
    return Result<LightId, std::string>::makeResult(
        makeLightId(LightType::Spot, index, spotLights_.generations[index]));
  }
  }

  return Result<LightId, std::string>::makeError(
      "RenderScene::addLight: unknown light type");
}

bool RenderScene::removeLight(LightId id) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!isValid(id)) {
    return false;
  }

  const uint32_t index = indexOf(id);
  switch (id.type) {
  case LightType::Directional:
    if (!directionalSlotValid(id)) {
      return false;
    }
    directionalLights_.live[index] = 0u;
    directionalLights_.enabled[index] = 0u;
    directionalLights_.packedIndices[index] = kInvalidPackedLightIndex;
    directionalLights_.names[index].clear();
    directionalLights_.generations[index] =
        nextResourceGeneration(directionalLights_.generations[index]);
    directionalLights_.freeSlots.push_back(index);
    rebuildPackedDirectionalLights();
    noteLightTopologyChanged();
    return true;
  case LightType::Point:
    if (!pointSlotValid(id)) {
      return false;
    }
    pointLights_.live[index] = 0u;
    pointLights_.enabled[index] = 0u;
    pointLights_.packedIndices[index] = kInvalidPackedLightIndex;
    pointLights_.names[index].clear();
    pointLights_.generations[index] =
        nextResourceGeneration(pointLights_.generations[index]);
    pointLights_.freeSlots.push_back(index);
    rebuildPackedLocalLights();
    noteLightTopologyChanged();
    return true;
  case LightType::Spot:
    if (!spotSlotValid(id)) {
      return false;
    }
    spotLights_.live[index] = 0u;
    spotLights_.enabled[index] = 0u;
    spotLights_.packedIndices[index] = kInvalidPackedLightIndex;
    spotLights_.names[index].clear();
    spotLights_.generations[index] =
        nextResourceGeneration(spotLights_.generations[index]);
    spotLights_.freeSlots.push_back(index);
    rebuildPackedLocalLights();
    noteLightTopologyChanged();
    return true;
  }

  return false;
}

bool RenderScene::getLightDesc(LightId id, LightDesc &out) const {
  if (!isValid(id)) {
    return false;
  }

  const uint32_t index = indexOf(id);
  switch (id.type) {
  case LightType::Directional:
    if (!directionalSlotValid(id)) {
      return false;
    }
    out.type = LightType::Directional;
    out.name = directionalLights_.names[index];
    out.position = directionalLights_.positions[index];
    out.rotation = directionalLights_.rotations[index];
    out.color = directionalLights_.colors[index];
    out.intensity = directionalLights_.intensities[index];
    out.range = 0.0f;
    out.innerConeAngleRadians = 0.0f;
    out.outerConeAngleRadians = 0.0f;
    out.enabled = directionalLights_.enabled[index] != 0u;
    return true;
  case LightType::Point:
    if (!pointSlotValid(id)) {
      return false;
    }
    out.type = LightType::Point;
    out.name = pointLights_.names[index];
    out.position = pointLights_.positions[index];
    out.rotation = pointLights_.rotations[index];
    out.color = pointLights_.colors[index];
    out.intensity = pointLights_.intensities[index];
    out.range = pointLights_.ranges[index];
    out.innerConeAngleRadians = 0.0f;
    out.outerConeAngleRadians = 0.0f;
    out.enabled = pointLights_.enabled[index] != 0u;
    return true;
  case LightType::Spot:
    if (!spotSlotValid(id)) {
      return false;
    }
    out.type = LightType::Spot;
    out.name = spotLights_.names[index];
    out.position = spotLights_.positions[index];
    out.rotation = spotLights_.rotations[index];
    out.color = spotLights_.colors[index];
    out.intensity = spotLights_.intensities[index];
    out.range = spotLights_.ranges[index];
    out.innerConeAngleRadians = spotLights_.innerConeAngles[index];
    out.outerConeAngleRadians = spotLights_.outerConeAngles[index];
    out.enabled = spotLights_.enabled[index] != 0u;
    return true;
  }

  return false;
}

bool RenderScene::updateLight(LightId id, const LightDesc &desc) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!isValid(id) || desc.type != id.type) {
    return false;
  }

  const LightDesc sanitized = sanitizeLightDesc(desc);
  const uint32_t index = indexOf(id);

  switch (id.type) {
  case LightType::Directional: {
    if (!directionalSlotValid(id)) {
      return false;
    }

    const bool oldEnabled = directionalLights_.enabled[index] != 0u;
    const bool newEnabled = sanitized.enabled;
    const bool gpuFieldsChanged =
        !nuri::vec3Equal(directionalLights_.positions[index],
                         sanitized.position) ||
        !nuri::quatEqual(directionalLights_.rotations[index],
                         sanitized.rotation) ||
        !nuri::vec3Equal(directionalLights_.colors[index], sanitized.color) ||
        directionalLights_.intensities[index] != sanitized.intensity;
    const bool nameChanged =
        directionalLights_.names[index].compare(sanitized.name) != 0;

    directionalLights_.names[index] = sanitized.name;
    directionalLights_.positions[index] = sanitized.position;
    directionalLights_.rotations[index] = sanitized.rotation;
    directionalLights_.colors[index] = sanitized.color;
    directionalLights_.intensities[index] = sanitized.intensity;
    directionalLights_.enabled[index] = newEnabled ? 1u : 0u;

    if (oldEnabled != newEnabled) {
      rebuildPackedDirectionalLights();
      noteLightTopologyChanged();
      return true;
    }

    if (newEnabled && gpuFieldsChanged) {
      const uint32_t packedIndex = directionalLights_.packedIndices[index];
      NURI_ASSERT(packedIndex != kInvalidPackedLightIndex,
                  "RenderScene::updateLight: enabled directional light must "
                  "have packed index");
      directionalLights_.packedGpu[packedIndex] = packDirectionalLight(
          sanitized.rotation, sanitized.color, sanitized.intensity);
      noteLightTransformChanged();
      return true;
    }

    return nameChanged;
  }
  case LightType::Point: {
    if (!pointSlotValid(id)) {
      return false;
    }

    const bool oldEnabled = pointLights_.enabled[index] != 0u;
    const bool newEnabled = sanitized.enabled;
    const bool gpuFieldsChanged =
        !nuri::vec3Equal(pointLights_.positions[index], sanitized.position) ||
        !nuri::quatEqual(pointLights_.rotations[index], sanitized.rotation) ||
        !nuri::vec3Equal(pointLights_.colors[index], sanitized.color) ||
        pointLights_.intensities[index] != sanitized.intensity ||
        pointLights_.ranges[index] != sanitized.range;
    const bool nameChanged =
        pointLights_.names[index].compare(sanitized.name) != 0;

    pointLights_.names[index] = sanitized.name;
    pointLights_.positions[index] = sanitized.position;
    pointLights_.rotations[index] = sanitized.rotation;
    pointLights_.colors[index] = sanitized.color;
    pointLights_.intensities[index] = sanitized.intensity;
    pointLights_.ranges[index] = sanitized.range;
    pointLights_.enabled[index] = newEnabled ? 1u : 0u;

    if (oldEnabled != newEnabled) {
      rebuildPackedLocalLights();
      noteLightTopologyChanged();
      return true;
    }

    if (newEnabled && gpuFieldsChanged) {
      const uint32_t packedIndex = pointLights_.packedIndices[index];
      NURI_ASSERT(packedIndex != kInvalidPackedLightIndex,
                  "RenderScene::updateLight: enabled point light must have "
                  "packed index");
      packedLocalLights_[packedIndex] =
          packPointLight(sanitized.position, sanitized.rotation,
                         sanitized.color, sanitized.intensity, sanitized.range);
      noteLightTransformChanged();
      return true;
    }

    return nameChanged;
  }
  case LightType::Spot: {
    if (!spotSlotValid(id)) {
      return false;
    }

    const bool oldEnabled = spotLights_.enabled[index] != 0u;
    const bool newEnabled = sanitized.enabled;
    const bool gpuFieldsChanged =
        !nuri::vec3Equal(spotLights_.positions[index], sanitized.position) ||
        !nuri::quatEqual(spotLights_.rotations[index], sanitized.rotation) ||
        !nuri::vec3Equal(spotLights_.colors[index], sanitized.color) ||
        spotLights_.intensities[index] != sanitized.intensity ||
        spotLights_.ranges[index] != sanitized.range ||
        spotLights_.innerConeAngles[index] != sanitized.innerConeAngleRadians ||
        spotLights_.outerConeAngles[index] != sanitized.outerConeAngleRadians;
    const bool nameChanged =
        spotLights_.names[index].compare(sanitized.name) != 0;

    spotLights_.names[index] = sanitized.name;
    spotLights_.positions[index] = sanitized.position;
    spotLights_.rotations[index] = sanitized.rotation;
    spotLights_.colors[index] = sanitized.color;
    spotLights_.intensities[index] = sanitized.intensity;
    spotLights_.ranges[index] = sanitized.range;
    spotLights_.innerConeAngles[index] = sanitized.innerConeAngleRadians;
    spotLights_.outerConeAngles[index] = sanitized.outerConeAngleRadians;
    spotLights_.enabled[index] = newEnabled ? 1u : 0u;

    if (oldEnabled != newEnabled) {
      rebuildPackedLocalLights();
      noteLightTopologyChanged();
      return true;
    }

    if (newEnabled && gpuFieldsChanged) {
      const uint32_t packedIndex = spotLights_.packedIndices[index];
      NURI_ASSERT(packedIndex != kInvalidPackedLightIndex,
                  "RenderScene::updateLight: enabled spot light must have "
                  "packed index");
      packedLocalLights_[packedIndex] = packSpotLight(
          sanitized.position, sanitized.rotation, sanitized.color,
          sanitized.intensity, sanitized.range, sanitized.innerConeAngleRadians,
          sanitized.outerConeAngleRadians);
      noteLightTransformChanged();
      return true;
    }

    return nameChanged;
  }
  }

  return false;
}

bool RenderScene::setLightTransform(LightId id, const glm::vec3 &position,
                                    const glm::quat &rotation) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (!isValid(id)) {
    return false;
  }

  const glm::vec3 sanitizedPosition =
      nuri::sanitizeFiniteVec3(position, glm::vec3(0.0f));
  const glm::quat sanitizedRotation = nuri::sanitizeRotation(rotation);
  const uint32_t index = indexOf(id);

  switch (id.type) {
  case LightType::Directional:
    if (!directionalSlotValid(id)) {
      return false;
    }
    if (nuri::vec3Equal(directionalLights_.positions[index],
                        sanitizedPosition) &&
        nuri::quatEqual(directionalLights_.rotations[index],
                        sanitizedRotation)) {
      return true;
    }
    directionalLights_.positions[index] = sanitizedPosition;
    directionalLights_.rotations[index] = sanitizedRotation;
    if (directionalLights_.enabled[index] != 0u) {
      const uint32_t packedIndex = directionalLights_.packedIndices[index];
      NURI_ASSERT(packedIndex != kInvalidPackedLightIndex,
                  "RenderScene::setLightTransform: enabled directional light "
                  "must have packed index");
      directionalLights_.packedGpu[packedIndex] = packDirectionalLight(
          sanitizedRotation, directionalLights_.colors[index],
          directionalLights_.intensities[index]);
      noteLightTransformChanged();
    }
    return true;
  case LightType::Point:
    if (!pointSlotValid(id)) {
      return false;
    }
    if (nuri::vec3Equal(pointLights_.positions[index], sanitizedPosition) &&
        nuri::quatEqual(pointLights_.rotations[index], sanitizedRotation)) {
      return true;
    }
    pointLights_.positions[index] = sanitizedPosition;
    pointLights_.rotations[index] = sanitizedRotation;
    if (pointLights_.enabled[index] != 0u) {
      const uint32_t packedIndex = pointLights_.packedIndices[index];
      NURI_ASSERT(packedIndex != kInvalidPackedLightIndex,
                  "RenderScene::setLightTransform: enabled point light must "
                  "have packed index");
      packedLocalLights_[packedIndex] = packPointLight(
          sanitizedPosition, sanitizedRotation, pointLights_.colors[index],
          pointLights_.intensities[index], pointLights_.ranges[index]);
      noteLightTransformChanged();
    }
    return true;
  case LightType::Spot:
    if (!spotSlotValid(id)) {
      return false;
    }
    if (nuri::vec3Equal(spotLights_.positions[index], sanitizedPosition) &&
        nuri::quatEqual(spotLights_.rotations[index], sanitizedRotation)) {
      return true;
    }
    spotLights_.positions[index] = sanitizedPosition;
    spotLights_.rotations[index] = sanitizedRotation;
    if (spotLights_.enabled[index] != 0u) {
      const uint32_t packedIndex = spotLights_.packedIndices[index];
      NURI_ASSERT(packedIndex != kInvalidPackedLightIndex,
                  "RenderScene::setLightTransform: enabled spot light must "
                  "have packed index");
      packedLocalLights_[packedIndex] = packSpotLight(
          sanitizedPosition, sanitizedRotation, spotLights_.colors[index],
          spotLights_.intensities[index], spotLights_.ranges[index],
          spotLights_.innerConeAngles[index],
          spotLights_.outerConeAngles[index]);
      noteLightTransformChanged();
    }
    return true;
  }

  return false;
}

const Renderable *RenderScene::renderable(uint32_t index) const {
  if (index >= renderables_.size()) {
    return nullptr;
  }
  return &renderables_[index];
}

void RenderScene::clearRenderables() {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (renderables_.empty()) {
    return;
  }
  for (const Renderable &renderable : renderables_) {
    releaseRenderable(renderable);
  }
  renderables_.clear();
  ++topologyVersion_;
  ++transformVersion_;
}

void RenderScene::clearLights() {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  const bool hadLights = !directionalLights_.generations.empty() ||
                         !pointLights_.generations.empty() ||
                         !spotLights_.generations.empty();
  if (!hadLights) {
    return;
  }

  directionalLights_ = DirectionalLightStore(memory_);
  pointLights_ = PointLightStore(memory_);
  spotLights_ = SpotLightStore(memory_);
  packedLocalLights_.clear();
  packedLocalLightIds_.clear();
  noteLightTopologyChanged();
}

void RenderScene::bindResources(ResourceManager *resources) {
  if (resources_ == resources) {
    return;
  }

  if (resources_ != nullptr) {
    for (const Renderable &renderable : renderables_) {
      releaseRenderable(renderable);
    }
    releaseEnvironment(environment_);
  }

  resources_ = resources;

  if (resources_ == nullptr) {
    return;
  }

  size_t writeIndex = 0;
  for (size_t readIndex = 0; readIndex < renderables_.size(); ++readIndex) {
    const Renderable renderable = renderables_[readIndex];
    if (!resources_->owns(renderable.model) ||
        !resources_->owns(renderable.material)) {
      continue;
    }
    renderables_[writeIndex] = renderable;
    retainRenderable(renderables_[writeIndex]);
    ++writeIndex;
  }

  if (writeIndex != renderables_.size()) {
    renderables_.resize(writeIndex);
    ++topologyVersion_;
    ++transformVersion_;
  }

  const auto sanitizeTextureRef = [this](TextureRef &ref) {
    if (isValid(ref) && !resources_->owns(ref)) {
      ref = kInvalidTextureRef;
    }
  };
  sanitizeTextureRef(environment_.cubemap);
  sanitizeTextureRef(environment_.irradiance);
  sanitizeTextureRef(environment_.prefilteredGgx);
  sanitizeTextureRef(environment_.prefilteredCharlie);
  sanitizeTextureRef(environment_.brdfLut);
  retainEnvironment(environment_);
}

void RenderScene::setEnvironment(EnvironmentHandles handles) {
  if (resources_ == nullptr) {
    environment_ = handles;
    return;
  }

  const auto updateTextureRef = [this](TextureRef &currentRef,
                                       TextureRef nextRef) {
    if (currentRef.value == nextRef.value) {
      return;
    }
    if (isValid(currentRef)) {
      resources_->release(currentRef);
    }
    if (isValid(nextRef)) {
      if (resources_->tryGet(nextRef) == nullptr) {
        NURI_ASSERT(false, "RenderScene::setEnvironment: stale texture handle");
        nextRef = kInvalidTextureRef;
      } else {
        resources_->retain(nextRef);
      }
    }
    currentRef = nextRef;
  };

  updateTextureRef(environment_.cubemap, handles.cubemap);
  updateTextureRef(environment_.irradiance, handles.irradiance);
  updateTextureRef(environment_.prefilteredGgx, handles.prefilteredGgx);
  updateTextureRef(environment_.prefilteredCharlie, handles.prefilteredCharlie);
  updateTextureRef(environment_.brdfLut, handles.brdfLut);
}

bool RenderScene::directionalSlotValid(LightId id) const noexcept {
  if (id.type != LightType::Directional || !isValid(id)) {
    return false;
  }
  const uint32_t index = indexOf(id);
  return index < directionalLights_.generations.size() &&
         directionalLights_.live[index] != 0u &&
         directionalLights_.generations[index] == generationOf(id);
}

bool RenderScene::pointSlotValid(LightId id) const noexcept {
  if (id.type != LightType::Point || !isValid(id)) {
    return false;
  }
  const uint32_t index = indexOf(id);
  return index < pointLights_.generations.size() &&
         pointLights_.live[index] != 0u &&
         pointLights_.generations[index] == generationOf(id);
}

bool RenderScene::spotSlotValid(LightId id) const noexcept {
  if (id.type != LightType::Spot || !isValid(id)) {
    return false;
  }
  const uint32_t index = indexOf(id);
  return index < spotLights_.generations.size() &&
         spotLights_.live[index] != 0u &&
         spotLights_.generations[index] == generationOf(id);
}

void RenderScene::rebuildPackedDirectionalLights() {
  directionalLights_.packedGpu.clear();
  directionalLights_.packedIds.clear();

  for (uint32_t index = 0; index < directionalLights_.generations.size();
       ++index) {
    directionalLights_.packedIndices[index] = kInvalidPackedLightIndex;
    if (directionalLights_.live[index] == 0u ||
        directionalLights_.enabled[index] == 0u) {
      continue;
    }

    directionalLights_.packedIndices[index] =
        static_cast<uint32_t>(directionalLights_.packedGpu.size());
    directionalLights_.packedGpu.push_back(packDirectionalLight(
        directionalLights_.rotations[index], directionalLights_.colors[index],
        directionalLights_.intensities[index]));
    directionalLights_.packedIds.push_back(makeLightId(
        LightType::Directional, index, directionalLights_.generations[index]));
  }
}

void RenderScene::rebuildPackedLocalLights() {
  packedLocalLights_.clear();
  packedLocalLightIds_.clear();

  for (uint32_t index = 0; index < pointLights_.generations.size(); ++index) {
    pointLights_.packedIndices[index] = kInvalidPackedLightIndex;
    if (pointLights_.live[index] == 0u || pointLights_.enabled[index] == 0u) {
      continue;
    }

    pointLights_.packedIndices[index] =
        static_cast<uint32_t>(packedLocalLights_.size());
    packedLocalLights_.push_back(packPointLight(
        pointLights_.positions[index], pointLights_.rotations[index],
        pointLights_.colors[index], pointLights_.intensities[index],
        pointLights_.ranges[index]));
    packedLocalLightIds_.push_back(
        makeLightId(LightType::Point, index, pointLights_.generations[index]));
  }

  for (uint32_t index = 0; index < spotLights_.generations.size(); ++index) {
    spotLights_.packedIndices[index] = kInvalidPackedLightIndex;
    if (spotLights_.live[index] == 0u || spotLights_.enabled[index] == 0u) {
      continue;
    }

    spotLights_.packedIndices[index] =
        static_cast<uint32_t>(packedLocalLights_.size());
    packedLocalLights_.push_back(packSpotLight(
        spotLights_.positions[index], spotLights_.rotations[index],
        spotLights_.colors[index], spotLights_.intensities[index],
        spotLights_.ranges[index], spotLights_.innerConeAngles[index],
        spotLights_.outerConeAngles[index]));
    packedLocalLightIds_.push_back(
        makeLightId(LightType::Spot, index, spotLights_.generations[index]));
  }
}

void RenderScene::noteLightTopologyChanged() noexcept {
  ++lightTopologyVersion_;
  ++lightTransformVersion_;
}

void RenderScene::noteLightTransformChanged() noexcept {
  ++lightTransformVersion_;
}

void RenderScene::retainRenderable(const Renderable &renderable) {
  if (resources_ == nullptr) {
    return;
  }
  resources_->retain(renderable.model);
  resources_->retain(renderable.material);
}

void RenderScene::releaseRenderable(const Renderable &renderable) {
  if (resources_ == nullptr) {
    return;
  }
  resources_->release(renderable.model);
  resources_->release(renderable.material);
}

void RenderScene::retainEnvironment(const EnvironmentHandles &handles) {
  if (resources_ == nullptr) {
    return;
  }
  forEachEnvironmentTextureRef(handles, [this](TextureRef textureRef) {
    if (isValid(textureRef)) {
      resources_->retain(textureRef);
    }
  });
}

void RenderScene::releaseEnvironment(const EnvironmentHandles &handles) {
  if (resources_ == nullptr) {
    return;
  }
  forEachEnvironmentTextureRef(handles, [this](TextureRef textureRef) {
    if (isValid(textureRef)) {
      resources_->release(textureRef);
    }
  });
}

} // namespace nuri
