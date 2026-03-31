#include "nuri/pch.h"

#include "nuri/scene/render_scene.h"

#include "nuri/core/profiling.h"
#include "nuri/math/light.h"
#include "nuri/math/utils.h"
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

template <typename Fn>
void forEachEnvironmentTextureRef(const EnvironmentHandles &handles, Fn &&fn) {
  fn(handles.cubemap);
  fn(handles.irradiance);
  fn(handles.prefilteredGgx);
  fn(handles.prefilteredCharlie);
  fn(handles.brdfLut);
}

} // namespace

RenderScene::RenderScene(std::pmr::memory_resource *memory)
    : memory_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      sceneGraph_(memory_), renderables_(memory_),
      renderableMorphWeights_(memory_), renderableSkinPalettes_(memory_),
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
  for (uint32_t index = 0; index < components.slots.slotCount(); ++index) {
    if (!components.slots.isLive(index)) {
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
    sceneGraph_.recycleRenderableSlot(index);
    sceneGraph_.renderableTopologyDirty_ = true;
  }
}

void RenderScene::rebuildFlatRenderables() {
  renderables_.clear();
  renderableMorphWeights_.clear();
  renderableSkinPalettes_.clear();
  auto &components = sceneGraph_.renderableComponents_;
  const auto &nodes = sceneGraph_.nodes_;

  const size_t liveCount = components.slots.liveCount();
  for (uint32_t index = 0; index < components.slots.slotCount(); ++index) {
    if (index < components.flatRenderableIndex.size()) {
      components.flatRenderableIndex[index] = kInvalidIndex;
    }
  }

  renderables_.reserve(liveCount);
  renderableMorphWeights_.reserve(liveCount);
  renderableSkinPalettes_.reserve(liveCount);
  for (uint32_t index = 0; index < components.slots.slotCount(); ++index) {
    if (!components.slots.isLive(index)) {
      continue;
    }
    const uint32_t nodeIndex = components.node[index];
    const glm::mat4 world =
        nodeIndex < nodes.worldFromRoot.size() && nodes.slots.isLive(nodeIndex)
            ? nodes.worldFromRoot[nodeIndex]
            : glm::mat4(1.0f);
    renderableMorphWeights_.emplace_back();
    renderableMorphWeights_.back().assign(
        components.morphWeights[index].begin(),
        components.morphWeights[index].end());
    renderableSkinPalettes_.emplace_back();
    renderableSkinPalettes_.back().assign(components.skinPalette[index].begin(),
                                          components.skinPalette[index].end());
    components.flatRenderableIndex[index] =
        static_cast<uint32_t>(renderables_.size());
    renderables_.push_back(Renderable{
        .id = makeRenderableId(index, components.slots.generation(index)),
        .node = nodeIndex < nodes.slots.slotCount()
                    ? makeNodeId(nodeIndex, nodes.slots.generation(nodeIndex))
                    : kInvalidNodeId,
        .model = components.models[index],
        .material = components.materials[index],
        .materialOverride = components.materialOverrides[index],
        .morphWeights =
            std::span<const float>(renderableMorphWeights_.back().data(),
                                   renderableMorphWeights_.back().size()),
        .skinPalette =
            std::span<const glm::mat4>(renderableSkinPalettes_.back().data(),
                                       renderableSkinPalettes_.back().size()),
        .modelMatrix = world,
    });
  }
}

void RenderScene::rebuildPackedDirectionalLights() {
  packedDirectionalLights_.clear();
  packedDirectionalLightIds_.clear();

  auto &store = sceneGraph_.directionalLights_;
  const auto &nodes = sceneGraph_.nodes_;
  for (uint32_t index = 0; index < store.slots.slotCount(); ++index) {
    store.packedIndices[index] = SceneGraph::kInvalidPackedLightIndex;
    if (!store.slots.isLive(index) || store.enabled[index] == 0u) {
      continue;
    }
    const uint32_t nodeIndex = store.node[index];
    if (nodeIndex >= nodes.worldFromRoot.size() ||
        !nodes.slots.isLive(nodeIndex)) {
      continue;
    }
    LightDesc local =
        nuri::makeLocalLightDesc(store, index, LightType::Directional);
    local.range = 0.0f;
    local.innerConeAngleRadians = 0.0f;
    local.outerConeAngleRadians = 0.0f;
    const LightDesc world =
        transformLightDesc(local, nodes.worldFromRoot[nodeIndex]);
    store.packedIndices[index] =
        static_cast<uint32_t>(packedDirectionalLights_.size());
    packedDirectionalLights_.push_back(nuri::packDirectionalLight(
        world.rotation, world.color, world.intensity));
    packedDirectionalLightIds_.push_back(makeLightId(
        LightType::Directional, index, store.slots.generation(index)));
  }
}

void RenderScene::rebuildPackedLocalLights() {
  packedLocalLights_.clear();
  packedLocalLightIds_.clear();
  const auto &nodes = sceneGraph_.nodes_;

  auto &pointStore = sceneGraph_.pointLights_;
  for (uint32_t index = 0; index < pointStore.slots.slotCount(); ++index) {
    pointStore.packedIndices[index] = SceneGraph::kInvalidPackedLightIndex;
    if (!pointStore.slots.isLive(index) || pointStore.enabled[index] == 0u) {
      continue;
    }
    const uint32_t nodeIndex = pointStore.node[index];
    if (nodeIndex >= nodes.worldFromRoot.size() ||
        !nodes.slots.isLive(nodeIndex)) {
      continue;
    }
    LightDesc local =
        nuri::makeLocalLightDesc(pointStore, index, LightType::Point);
    const LightDesc world =
        transformLightDesc(local, nodes.worldFromRoot[nodeIndex]);
    pointStore.packedIndices[index] =
        static_cast<uint32_t>(packedLocalLights_.size());
    packedLocalLights_.push_back(
        nuri::packPointLight(world.position, world.rotation, world.color,
                             world.intensity, world.range, world.enabled));
    packedLocalLightIds_.push_back(makeLightId(
        LightType::Point, index, pointStore.slots.generation(index)));
  }

  auto &spotStore = sceneGraph_.spotLights_;
  for (uint32_t index = 0; index < spotStore.slots.slotCount(); ++index) {
    spotStore.packedIndices[index] = SceneGraph::kInvalidPackedLightIndex;
    if (!spotStore.slots.isLive(index) || spotStore.enabled[index] == 0u) {
      continue;
    }
    const uint32_t nodeIndex = spotStore.node[index];
    if (nodeIndex >= nodes.worldFromRoot.size() ||
        !nodes.slots.isLive(nodeIndex)) {
      continue;
    }
    LightDesc local =
        nuri::makeLocalLightDesc(spotStore, index, LightType::Spot);
    const LightDesc world =
        transformLightDesc(local, nodes.worldFromRoot[nodeIndex]);
    spotStore.packedIndices[index] =
        static_cast<uint32_t>(packedLocalLights_.size());
    packedLocalLights_.push_back(nuri::packSpotLight(
        world.position, world.rotation, world.color, world.intensity,
        world.range, world.innerConeAngleRadians, world.outerConeAngleRadians));
    packedLocalLightIds_.push_back(
        makeLightId(LightType::Spot, index, spotStore.slots.generation(index)));
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
    for (uint32_t index = 0; index < components.slots.slotCount(); ++index) {
      if (!components.slots.isLive(index)) {
        continue;
      }
      const uint32_t flatIndex = components.flatRenderableIndex[index];
      const uint32_t nodeIndex = components.node[index];
      if (flatIndex == kInvalidIndex || flatIndex >= renderables_.size() ||
          nodeIndex >= nodes.worldFromRoot.size() ||
          !nodes.slots.isLive(nodeIndex)) {
        continue;
      }
      renderables_[flatIndex].modelMatrix = nodes.worldFromRoot[nodeIndex];
      renderables_[flatIndex].node =
          makeNodeId(nodeIndex, nodes.slots.generation(nodeIndex));
      renderableMorphWeights_[flatIndex].assign(
          components.morphWeights[index].begin(),
          components.morphWeights[index].end());
      renderables_[flatIndex].morphWeights =
          std::span<const float>(renderableMorphWeights_[flatIndex].data(),
                                 renderableMorphWeights_[flatIndex].size());
      renderableSkinPalettes_[flatIndex].assign(
          components.skinPalette[index].begin(),
          components.skinPalette[index].end());
      renderables_[flatIndex].skinPalette =
          std::span<const glm::mat4>(renderableSkinPalettes_[flatIndex].data(),
                                     renderableSkinPalettes_[flatIndex].size());
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
  rebuildFlatRenderables();

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
