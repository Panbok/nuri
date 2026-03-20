#include "nuri/pch.h"

#include "nuri/scene/render_scene.h"

#include "nuri/core/profiling.h"
#include "nuri/math/light.h"
#include "nuri/resources/gpu/resource_manager.h"

namespace nuri {
namespace {

template <typename Ref>
[[nodiscard]] bool resourceAlive(ResourceManager *resources, Ref ref) {
  return resources != nullptr && isValid(ref) && resources->owns(ref) &&
         resources->tryGet(ref) != nullptr;
}

template <typename Ref>
void retainResourceIfAlive(ResourceManager *resources, Ref ref) {
  if (resourceAlive(resources, ref)) {
    resources->retain(ref);
  }
}

template <typename Ref>
void releaseResourceIfOwned(ResourceManager *resources, Ref ref) {
  if (resources != nullptr && isValid(ref) && resources->owns(ref)) {
    resources->release(ref);
  }
}

[[nodiscard]] glm::quat sanitizeRotation(const glm::quat &rotation) {
  const float length = glm::length(rotation);
  if (!std::isfinite(length) || length <= 1.0e-6f) {
    return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
  }
  return glm::normalize(rotation);
}

template <typename Fn>
void forEachEnvironmentTextureRef(const EnvironmentHandles &handles, Fn &&fn) {
  fn(handles.cubemap);
  fn(handles.irradiance);
  fn(handles.prefilteredGgx);
  fn(handles.prefilteredCharlie);
  fn(handles.brdfLut);
}

[[nodiscard]] glm::vec3 lightDirectionFromRotation(const glm::quat &rotation) {
  const glm::vec3 direction =
      sanitizeRotation(rotation) * glm::vec3(0.0f, 0.0f, -1.0f);
  const float length = glm::length(direction);
  if (!std::isfinite(length) || length <= 1.0e-6f) {
    return glm::vec3(0.0f, 0.0f, -1.0f);
  }
  return direction / length;
}

[[nodiscard]] uint32_t floatBitsToUint(float value) {
  return std::bit_cast<uint32_t>(value);
}

[[nodiscard]] DirectionalLightGpuData
packDirectionalLight(const glm::quat &rotation, const glm::vec3 &color,
                     float intensity) {
  const glm::vec3 direction = lightDirectionFromRotation(rotation);
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
  const glm::vec3 direction = lightDirectionFromRotation(rotation);
  return LocalLightGpuData{
      .positionRange = glm::vec4(position, range),
      .directionOuterCos = glm::vec4(direction, -1.0f),
      .colorIntensity = glm::vec4(color, intensity),
      .innerCosTypeEnabledReserved =
          glm::uvec4(floatBitsToUint(-1.0f),
                     static_cast<uint32_t>(LocalLightGpuType::Point), 1u, 0u),
  };
}

[[nodiscard]] LocalLightGpuData
packSpotLight(const glm::vec3 &position, const glm::quat &rotation,
              const glm::vec3 &color, float intensity, float range,
              float innerConeAngleRadians, float outerConeAngleRadians) {
  const glm::vec3 direction = lightDirectionFromRotation(rotation);
  return LocalLightGpuData{
      .positionRange = glm::vec4(position, range),
      .directionOuterCos =
          glm::vec4(direction, std::cos(outerConeAngleRadians)),
      .colorIntensity = glm::vec4(color, intensity),
      .innerCosTypeEnabledReserved =
          glm::uvec4(floatBitsToUint(std::cos(innerConeAngleRadians)),
                     static_cast<uint32_t>(LocalLightGpuType::Spot), 1u, 0u),
  };
}

template <typename Store>
[[nodiscard]] LightDesc makeLocalLightDesc(const Store &store, uint32_t index,
                                           LightType type) {
  LightDesc out{};
  out.type = type;
  out.name = store.names[index];
  out.position = store.localPositions[index];
  out.rotation = store.localRotations[index];
  out.color = store.colors[index];
  out.intensity = store.intensities[index];
  out.enabled = store.enabled[index] != 0u;
  return out;
}

} // namespace

RenderScene::RenderScene(std::pmr::memory_resource *memory)
    : memory_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      sceneGraph_(memory_), renderables_(memory_),
      packedDirectionalLights_(memory_), packedLocalLights_(memory_),
      packedDirectionalLightIds_(memory_), packedLocalLightIds_(memory_) {}

RenderScene::~RenderScene() {
  for (const Renderable &renderable : renderables_) {
    releaseRenderableRefs(renderable.model, renderable.material,
                          renderable.materialOverride);
  }
  setEnvironment(EnvironmentHandles{});
}

const Renderable *RenderScene::renderable(uint32_t index) const {
  if (index >= renderables_.size()) {
    return nullptr;
  }
  return &renderables_[index];
}

void RenderScene::retainRenderableRefs(ModelRef model, MaterialRef material,
                                       MaterialRef materialOverride) {
  retainResourceIfAlive(resources_, model);
  retainResourceIfAlive(resources_, material);
  retainResourceIfAlive(resources_, materialOverride);
}

void RenderScene::releaseRenderableRefs(ModelRef model, MaterialRef material,
                                        MaterialRef materialOverride) {
  releaseResourceIfOwned(resources_, model);
  releaseResourceIfOwned(resources_, material);
  releaseResourceIfOwned(resources_, materialOverride);
}

void RenderScene::retainEnvironment(const EnvironmentHandles &handles) {
  if (resources_ == nullptr) {
    return;
  }
  forEachEnvironmentTextureRef(handles, [this](TextureRef textureRef) {
    retainResourceIfAlive(resources_, textureRef);
  });
}

void RenderScene::releaseEnvironment(const EnvironmentHandles &handles) {
  if (resources_ == nullptr) {
    return;
  }
  forEachEnvironmentTextureRef(handles, [this](TextureRef textureRef) {
    releaseResourceIfOwned(resources_, textureRef);
  });
}

void RenderScene::sanitizeGraphRenderableRefs() {
  if (resources_ == nullptr) {
    return;
  }

  auto &components = sceneGraph_.renderableComponents_;
  for (uint32_t index = 0; index < components.generations.size(); ++index) {
    if (components.live[index] == 0u) {
      continue;
    }
    const ModelRef model = components.models[index];
    const MaterialRef material = components.materials[index];
    const MaterialRef materialOverride = components.materialOverrides[index];
    if (resourceAlive(resources_, model) &&
        resourceAlive(resources_, material)) {
      if (!isValid(materialOverride) ||
          resourceAlive(resources_, materialOverride)) {
        continue;
      }
      components.materialOverrides[index] = kInvalidMaterialRef;
      sceneGraph_.renderableTopologyDirty_ = true;
      continue;
    }
    components.live[index] = 0u;
    components.node[index] = kInvalidIndex;
    components.models[index] = kInvalidModelRef;
    components.materials[index] = kInvalidMaterialRef;
    components.materialOverrides[index] = kInvalidMaterialRef;
    components.flatRenderableIndex[index] = kInvalidIndex;
    components.generations[index] =
        nextResourceGeneration(components.generations[index]);
    components.freeSlots.push_back(index);
    sceneGraph_.renderableTopologyDirty_ = true;
  }
}

void RenderScene::rebuildFlatRenderables() {
  renderables_.clear();
  auto &components = sceneGraph_.renderableComponents_;
  const auto &nodes = sceneGraph_.nodes_;

  size_t liveCount = 0u;
  for (uint32_t index = 0; index < components.generations.size(); ++index) {
    if (components.live[index] != 0u) {
      ++liveCount;
    }
    if (index < components.flatRenderableIndex.size()) {
      components.flatRenderableIndex[index] = kInvalidIndex;
    }
  }

  renderables_.reserve(liveCount);
  for (uint32_t index = 0; index < components.generations.size(); ++index) {
    if (components.live[index] == 0u) {
      continue;
    }
    const uint32_t nodeIndex = components.node[index];
    const glm::mat4 world =
        nodeIndex < nodes.worldFromRoot.size() && nodes.live[nodeIndex] != 0u
            ? nodes.worldFromRoot[nodeIndex]
            : glm::mat4(1.0f);
    components.flatRenderableIndex[index] =
        static_cast<uint32_t>(renderables_.size());
    renderables_.push_back(Renderable{
        .id = makeRenderableId(index, components.generations[index]),
        .node = nodeIndex < nodes.generations.size()
                    ? makeNodeId(nodeIndex, nodes.generations[nodeIndex])
                    : kInvalidNodeId,
        .model = components.models[index],
        .material = components.materials[index],
        .materialOverride = components.materialOverrides[index],
        .modelMatrix = world,
    });
  }
}

void RenderScene::rebuildPackedDirectionalLights() {
  packedDirectionalLights_.clear();
  packedDirectionalLightIds_.clear();

  auto &store = sceneGraph_.directionalLights_;
  const auto &nodes = sceneGraph_.nodes_;
  for (uint32_t index = 0; index < store.generations.size(); ++index) {
    store.packedIndices[index] = SceneGraph::kInvalidPackedLightIndex;
    if (store.live[index] == 0u || store.enabled[index] == 0u) {
      continue;
    }
    const uint32_t nodeIndex = store.node[index];
    if (nodeIndex >= nodes.worldFromRoot.size() ||
        nodes.live[nodeIndex] == 0u) {
      continue;
    }
    LightDesc local = makeLocalLightDesc(store, index, LightType::Directional);
    local.range = 0.0f;
    local.innerConeAngleRadians = 0.0f;
    local.outerConeAngleRadians = 0.0f;
    const LightDesc world =
        transformLightDesc(local, nodes.worldFromRoot[nodeIndex]);
    store.packedIndices[index] =
        static_cast<uint32_t>(packedDirectionalLights_.size());
    packedDirectionalLights_.push_back(
        packDirectionalLight(world.rotation, world.color, world.intensity));
    packedDirectionalLightIds_.push_back(
        makeLightId(LightType::Directional, index, store.generations[index]));
  }
}

void RenderScene::rebuildPackedLocalLights() {
  packedLocalLights_.clear();
  packedLocalLightIds_.clear();
  const auto &nodes = sceneGraph_.nodes_;

  auto &pointStore = sceneGraph_.pointLights_;
  for (uint32_t index = 0; index < pointStore.generations.size(); ++index) {
    pointStore.packedIndices[index] = SceneGraph::kInvalidPackedLightIndex;
    if (pointStore.live[index] == 0u || pointStore.enabled[index] == 0u) {
      continue;
    }
    const uint32_t nodeIndex = pointStore.node[index];
    if (nodeIndex >= nodes.worldFromRoot.size() ||
        nodes.live[nodeIndex] == 0u) {
      continue;
    }
    LightDesc local = makeLocalLightDesc(pointStore, index, LightType::Point);
    local.range = pointStore.ranges[index];
    const LightDesc world =
        transformLightDesc(local, nodes.worldFromRoot[nodeIndex]);
    pointStore.packedIndices[index] =
        static_cast<uint32_t>(packedLocalLights_.size());
    packedLocalLights_.push_back(packPointLight(world.position, world.rotation,
                                                world.color, world.intensity,
                                                world.range));
    packedLocalLightIds_.push_back(
        makeLightId(LightType::Point, index, pointStore.generations[index]));
  }

  auto &spotStore = sceneGraph_.spotLights_;
  for (uint32_t index = 0; index < spotStore.generations.size(); ++index) {
    spotStore.packedIndices[index] = SceneGraph::kInvalidPackedLightIndex;
    if (spotStore.live[index] == 0u || spotStore.enabled[index] == 0u) {
      continue;
    }
    const uint32_t nodeIndex = spotStore.node[index];
    if (nodeIndex >= nodes.worldFromRoot.size() ||
        nodes.live[nodeIndex] == 0u) {
      continue;
    }
    LightDesc local = makeLocalLightDesc(spotStore, index, LightType::Spot);
    local.range = spotStore.ranges[index];
    local.innerConeAngleRadians = spotStore.innerConeAngles[index];
    local.outerConeAngleRadians = spotStore.outerConeAngles[index];
    const LightDesc world =
        transformLightDesc(local, nodes.worldFromRoot[nodeIndex]);
    spotStore.packedIndices[index] =
        static_cast<uint32_t>(packedLocalLights_.size());
    packedLocalLights_.push_back(packSpotLight(
        world.position, world.rotation, world.color, world.intensity,
        world.range, world.innerConeAngleRadians, world.outerConeAngleRadians));
    packedLocalLightIds_.push_back(
        makeLightId(LightType::Spot, index, spotStore.generations[index]));
  }
}

Result<bool, std::string> RenderScene::commit() {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  // Commit is the authored-state to derived-cache boundary: hierarchy,
  // components, and local light data stay in SceneGraph; RenderScene rebuilds
  // the flat renderer-facing views here.
  bool changed = false;
  (void)sceneGraph_.syncWorldTransforms();

  if (sceneGraph_.renderableTopologyDirty_) {
    sanitizeGraphRenderableRefs();
    if (resources_ != nullptr) {
      for (const Renderable &renderable : renderables_) {
        releaseRenderableRefs(renderable.model, renderable.material,
                              renderable.materialOverride);
      }
    }
    rebuildFlatRenderables();
    if (resources_ != nullptr) {
      for (const Renderable &renderable : renderables_) {
        retainRenderableRefs(renderable.model, renderable.material,
                             renderable.materialOverride);
      }
    }
    ++topologyVersion_;
    ++transformVersion_;
    sceneGraph_.renderableTopologyDirty_ = false;
    sceneGraph_.renderableTransformsDirty_ = false;
    changed = true;
  } else if (sceneGraph_.renderableTransformsDirty_) {
    bool updatedAny = false;
    const auto &components = sceneGraph_.renderableComponents_;
    const auto &nodes = sceneGraph_.nodes_;
    for (uint32_t index = 0; index < components.generations.size(); ++index) {
      if (components.live[index] == 0u) {
        continue;
      }
      const uint32_t flatIndex = components.flatRenderableIndex[index];
      const uint32_t nodeIndex = components.node[index];
      if (flatIndex == kInvalidIndex || flatIndex >= renderables_.size() ||
          nodeIndex >= nodes.worldFromRoot.size() ||
          nodes.live[nodeIndex] == 0u) {
        continue;
      }
      renderables_[flatIndex].modelMatrix = nodes.worldFromRoot[nodeIndex];
      renderables_[flatIndex].node =
          makeNodeId(nodeIndex, nodes.generations[nodeIndex]);
      updatedAny = true;
    }
    if (updatedAny) {
      ++transformVersion_;
      changed = true;
    }
    sceneGraph_.renderableTransformsDirty_ = false;
  }

  if (sceneGraph_.lightTopologyDirty_) {
    rebuildPackedDirectionalLights();
    rebuildPackedLocalLights();
    noteLightTopologyChanged();
    sceneGraph_.lightTopologyDirty_ = false;
    sceneGraph_.lightDataDirty_ = false;
    changed = true;
  } else if (sceneGraph_.lightDataDirty_) {
    rebuildPackedDirectionalLights();
    rebuildPackedLocalLights();
    noteLightTransformChanged();
    sceneGraph_.lightDataDirty_ = false;
    changed = true;
  }

  return Result<bool, std::string>::makeResult(changed);
}

void RenderScene::bindResources(ResourceManager *resources) {
  if (resources_ == resources) {
    return;
  }

  if (resources_ != nullptr) {
    for (const Renderable &renderable : renderables_) {
      releaseRenderableRefs(renderable.model, renderable.material,
                            renderable.materialOverride);
    }
    releaseEnvironment(environment_);
  }

  resources_ = resources;
  if (resources_ == nullptr) {
    return;
  }

  sanitizeGraphRenderableRefs();

  for (const Renderable &renderable : renderables_) {
    retainRenderableRefs(renderable.model, renderable.material,
                         renderable.materialOverride);
  }

  const auto sanitizeTextureRef = [this](TextureRef &ref) {
    if (isValid(ref) && !resourceAlive(resources_, ref)) {
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
    releaseResourceIfOwned(resources_, currentRef);
    if (isValid(nextRef)) {
      if (!resourceAlive(resources_, nextRef)) {
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

void RenderScene::noteLightTopologyChanged() noexcept {
  ++lightTopologyVersion_;
  ++lightTransformVersion_;
}

void RenderScene::noteLightTransformChanged() noexcept {
  ++lightTransformVersion_;
}

} // namespace nuri
