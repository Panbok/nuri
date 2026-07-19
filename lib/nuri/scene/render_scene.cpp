#include "nuri/scene/render_scene.h"
#include "nuri/core/profiling.h"
#include "nuri/core/thread_priority.h"
#include "nuri/math/light.h"
#include "nuri/math/utils.h"
#include "nuri/pch.h"
#include "nuri/resources/gpu/resource_manager.h"
namespace nuri {
namespace {
std::atomic<uint64_t> gNextRenderSceneId{1u};
template <typename Ref>
[[nodiscard]] bool resourceAlive(ResourceManager *resources, Ref ref) {
  return resources != nullptr && resources->tryGet(ref) != nullptr;
}
template <typename Ref>
void retainResource(ResourceManager *resources, Ref ref) {
  if (resources != nullptr) {
    resources->retain(ref);
  }
}
template <typename Ref>
void releaseResourceIfOwned(ResourceManager *resources, Ref ref) {
  if (resources != nullptr) {
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

struct RenderScene::IncrementalCommitState {
  explicit IncrementalCommitState(std::pmr::memory_resource *memory)
      : renderables(memory), renderableIndexById(memory), morphWeights(memory),
        skinPalettes(memory) {}
  std::pmr::vector<Renderable> renderables;
  std::pmr::unordered_map<RenderableId, uint32_t> renderableIndexById;
  std::pmr::vector<std::pmr::vector<float>> morphWeights;
  std::pmr::vector<std::pmr::vector<glm::mat4>> skinPalettes;
  uint64_t graphTopologyVersion = 0u;
  size_t liveRenderableCount = 0u;
  uint32_t nextRenderableSlot = 0u;
  std::atomic_bool allocationReady = false;
  std::atomic_bool allocationFailed = false;
};

RenderScene::RenderScene(std::pmr::memory_resource *memory)
    : memory_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      sceneGraph_(memory_), renderables_(std::pmr::new_delete_resource()),
      renderableIndexById_(std::pmr::new_delete_resource()),
      renderableMorphWeights_(std::pmr::new_delete_resource()),
      renderableSkinPalettes_(std::pmr::new_delete_resource()),
      packedDirectionalLights_(memory_), packedLocalLights_(memory_),
      packedDirectionalLightIds_(memory_), packedLocalLightIds_(memory_),
      id_(gNextRenderSceneId.fetch_add(1u, std::memory_order_relaxed)),
      topologyVersion_(0u), transformVersion_(0u), deformationVersion_(0u),
      lightTopologyVersion_(0u), lightTransformVersion_(0u) {}

RenderScene::~RenderScene() {
  discardIncrementalCommit();
  for (const Renderable &renderable : renderables_) {
    releaseRenderableRefs(renderable.model, renderable.material,
                          renderable.materialOverride);
  }
  setEnvironment(EnvironmentHandles{});
}

void RenderScene::discardIncrementalCommit() noexcept {
  std::shared_ptr<IncrementalCommitState> pending =
      std::move(incrementalCommit_);
  if (pending == nullptr) {
    return;
  }
  if (!pending->allocationReady.load(std::memory_order_acquire)) {
    return;
  }
  for (const Renderable &renderable : pending->renderables) {
    releaseRenderableRefs(renderable.model, renderable.material,
                          renderable.materialOverride);
  }
}

const Renderable *RenderScene::renderable(uint32_t index) const {
  if (index >= renderables_.size()) {
    return nullptr;
  }
  return &renderables_[index];
}

std::optional<uint32_t>
RenderScene::findRenderableIndex(RenderableId id) const {
  const auto it = renderableIndexById_.find(id);
  if (it == renderableIndexById_.end())
    return std::nullopt;
  return it->second;
}

void RenderScene::retainRenderableRefs(ModelRef model, MaterialRef material,
                                       MaterialRef materialOverride) {
  retainResource(resources_, model);
  retainResource(resources_, material);
  retainResource(resources_, materialOverride);
}

void RenderScene::releaseRenderableRefs(ModelRef model, MaterialRef material,
                                        MaterialRef materialOverride) {
  releaseResourceIfOwned(resources_, model);
  releaseResourceIfOwned(resources_, material);
  releaseResourceIfOwned(resources_, materialOverride);
}

void RenderScene::retainEnvironment(const EnvironmentHandles &handles) {
  forEachEnvironmentTextureRef(handles, [this](TextureRef textureRef) {
    retainResource(resources_, textureRef);
  });
}

void RenderScene::releaseEnvironment(const EnvironmentHandles &handles) {
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
  renderableIndexById_.clear();
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
  renderableIndexById_.reserve(liveCount);
  renderableMorphWeights_.reserve(liveCount);
  renderableSkinPalettes_.reserve(liveCount);
  for (uint32_t index = 0; index < components.slots.slotCount(); ++index) {
    if (!components.slots.isLive(index)) {
      continue;
    }
    const uint32_t nodeIndex = components.node[index];
    const glm::mat4 world = nodes.worldFromRoot[nodeIndex];
    renderableMorphWeights_.emplace_back();
    renderableMorphWeights_.back().assign(
        components.morphWeights[index].begin(),
        components.morphWeights[index].end());
    renderableSkinPalettes_.emplace_back();
    renderableSkinPalettes_.back().assign(components.skinPalette[index].begin(),
                                          components.skinPalette[index].end());
    components.flatRenderableIndex[index] =
        static_cast<uint32_t>(renderables_.size());
    const Renderable renderable{
        .id = makeRenderableId(index, components.slots.generation(index)),
        .node = makeNodeId(nodeIndex, nodes.slots.generation(nodeIndex)),
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
    };
    renderableIndexById_.emplace(renderable.id,
                                 static_cast<uint32_t>(renderables_.size()));
    renderables_.push_back(renderable);
  }
}

void RenderScene::rebuildPackedDirectionalLights() {
  packedDirectionalLights_.clear();
  packedDirectionalLightIds_.clear();
  constexpr size_t kTypeIndex = static_cast<size_t>(LightType::Directional);
  auto &store = sceneGraph_.lights_[kTypeIndex];
  const auto &nodes = sceneGraph_.nodes_;
  for (uint32_t index = 0; index < store.slots.slotCount(); ++index) {
    auto &record = store.records[index];
    record.packedIndex = SceneGraph::kInvalidIndex;
    if (!store.slots.isLive(index) || !record.enabled) {
      continue;
    }
    const uint32_t nodeIndex = record.node;
    const LightDesc local =
        nuri::makeLocalLightDesc(record, LightType::Directional);
    const LightDesc world =
        transformLightDesc(local, nodes.worldFromRoot[nodeIndex]);
    record.packedIndex = static_cast<uint32_t>(packedDirectionalLights_.size());
    packedDirectionalLights_.push_back(
        nuri::packDirectionalLight(world.rotation, world.color, world.intensity,
                                   world.angularRadiusDegrees));
    packedDirectionalLightIds_.push_back(makeLightId(
        LightType::Directional, index, store.slots.generation(index)));
  }
}

void RenderScene::rebuildPackedLocalLights() {
  packedLocalLights_.clear();
  packedLocalLightIds_.clear();
  const auto &nodes = sceneGraph_.nodes_;
  for (size_t typeIndex = static_cast<size_t>(LightType::Point);
       typeIndex <= static_cast<size_t>(LightType::Spot); ++typeIndex) {
    const LightType type = static_cast<LightType>(typeIndex);
    auto &store = sceneGraph_.lights_[typeIndex];
    for (uint32_t index = 0; index < store.slots.slotCount(); ++index) {
      auto &record = store.records[index];
      record.packedIndex = SceneGraph::kInvalidIndex;
      if (!store.slots.isLive(index) || !record.enabled) {
        continue;
      }
      const uint32_t nodeIndex = record.node;
      const LightDesc world =
          transformLightDesc(nuri::makeLocalLightDesc(record, type),
                             nodes.worldFromRoot[nodeIndex]);
      record.packedIndex = static_cast<uint32_t>(packedLocalLights_.size());
      packedLocalLights_.push_back(
          type == LightType::Point
              ? nuri::packPointLight(world.position, world.rotation,
                                     world.color, world.intensity, world.range,
                                     world.enabled)
              : nuri::packSpotLight(world.position, world.rotation, world.color,
                                    world.intensity, world.range,
                                    world.innerConeAngleRadians,
                                    world.outerConeAngleRadians));
      packedLocalLightIds_.push_back(
          makeLightId(type, index, store.slots.generation(index)));
    }
  }
}

bool RenderScene::commitPackedLights() {
  if (!sceneGraph_.lightTopologyDirty_ && !sceneGraph_.lightDataDirty_)
    return false;
  rebuildPackedDirectionalLights();
  rebuildPackedLocalLights();
  sceneGraph_.lightTopologyDirty_ ? noteLightTopologyChanged()
                                  : noteLightTransformChanged();
  sceneGraph_.lightTopologyDirty_ = false;
  sceneGraph_.lightDataDirty_ = false;
  return true;
}

Result<bool, std::string> RenderScene::commit() {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  if (incrementalCommit_ != nullptr) {
    return Result<bool, std::string>::makeError(
        "RenderScene::commit: an inactive incremental commit is in progress");
  }
  bool changed = false;
  (void)sceneGraph_.syncWorldTransforms();
  if (sceneGraph_.renderableTopologyDirty_) {
    sanitizeGraphRenderableRefs();
    for (const Renderable &renderable : renderables_) {
      releaseRenderableRefs(renderable.model, renderable.material,
                            renderable.materialOverride);
    }
    rebuildFlatRenderables();
    for (const Renderable &renderable : renderables_) {
      retainRenderableRefs(renderable.model, renderable.material,
                           renderable.materialOverride);
    }
    ++topologyVersion_;
    ++transformVersion_;
    ++deformationVersion_;
    sceneGraph_.renderableTopologyDirty_ = false;
    sceneGraph_.renderableTransformsDirty_ = false;
    sceneGraph_.renderableDeformationsDirty_ = false;
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
      renderables_[flatIndex].modelMatrix = nodes.worldFromRoot[nodeIndex];
      renderables_[flatIndex].node =
          makeNodeId(nodeIndex, nodes.slots.generation(nodeIndex));
      updatedAny = true;
    }
    if (updatedAny) {
      ++transformVersion_;
      changed = true;
    }
    sceneGraph_.renderableTransformsDirty_ = false;
  }
  if (!sceneGraph_.renderableTopologyDirty_ &&
      sceneGraph_.renderableDeformationsDirty_) {
    bool updatedAny = false;
    for (uint32_t index = 0;
         index < sceneGraph_.renderableComponents_.slots.slotCount(); ++index) {
      if (!sceneGraph_.renderableComponents_.slots.isLive(index)) {
        continue;
      }
      const uint32_t flatIndex =
          sceneGraph_.renderableComponents_.flatRenderableIndex[index];
      renderableMorphWeights_[flatIndex].assign(
          sceneGraph_.renderableComponents_.morphWeights[index].begin(),
          sceneGraph_.renderableComponents_.morphWeights[index].end());
      renderables_[flatIndex].morphWeights =
          std::span<const float>(renderableMorphWeights_[flatIndex].data(),
                                 renderableMorphWeights_[flatIndex].size());
      renderableSkinPalettes_[flatIndex].assign(
          sceneGraph_.renderableComponents_.skinPalette[index].begin(),
          sceneGraph_.renderableComponents_.skinPalette[index].end());
      renderables_[flatIndex].skinPalette =
          std::span<const glm::mat4>(renderableSkinPalettes_[flatIndex].data(),
                                     renderableSkinPalettes_[flatIndex].size());
      updatedAny = true;
    }
    if (updatedAny) {
      ++deformationVersion_;
      sceneGraph_.renderableDeformationsDirty_ = false;
      changed = true;
    }
  }
  changed |= commitPackedLights();
  return Result<bool, std::string>::makeResult(changed);
}

Result<bool, std::string>
RenderScene::commitInactiveStep(uint32_t maxOperations) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  const uint32_t operationBudget = std::max(maxOperations, 1u);
  if (!sceneGraph_.syncWorldTransformsStep(operationBudget)) {
    return Result<bool, std::string>::makeResult(false);
  }
  if (!sceneGraph_.renderableTopologyDirty_) {
    auto committed = commit();
    if (committed.hasError()) {
      return Result<bool, std::string>::makeError(committed.error());
    }
    return Result<bool, std::string>::makeResult(true);
  }
  if (incrementalCommit_ == nullptr) {
    if (!renderables_.empty()) {
      return Result<bool, std::string>::makeError(
          "RenderScene::commitInactiveStep requires a freshly built scene");
    }
    sanitizeGraphRenderableRefs();
    incrementalCommit_ = std::make_shared<IncrementalCommitState>(
        std::pmr::new_delete_resource());
    incrementalCommit_->graphTopologyVersion = sceneGraph_.topologyVersion();
    incrementalCommit_->liveRenderableCount =
        sceneGraph_.renderableComponents_.slots.liveCount();
    std::shared_ptr<IncrementalCommitState> pendingAllocation =
        incrementalCommit_;
    std::thread([pendingAllocation] {
      setCurrentThreadBackgroundPriority();
      try {
        pendingAllocation->renderables.reserve(
            pendingAllocation->liveRenderableCount);
        pendingAllocation->renderableIndexById.reserve(
            pendingAllocation->liveRenderableCount);
        pendingAllocation->morphWeights.reserve(
            pendingAllocation->liveRenderableCount);
        pendingAllocation->skinPalettes.reserve(
            pendingAllocation->liveRenderableCount);
      } catch (...) {
        pendingAllocation->allocationFailed.store(true,
                                                  std::memory_order_release);
      }
      pendingAllocation->allocationReady.store(true, std::memory_order_release);
    }).detach();
    return Result<bool, std::string>::makeResult(false);
  }
  IncrementalCommitState &pending = *incrementalCommit_;
  if (!pending.allocationReady.load(std::memory_order_acquire)) {
    return Result<bool, std::string>::makeResult(false);
  }
  if (pending.allocationFailed.load(std::memory_order_acquire)) {
    discardIncrementalCommit();
    return Result<bool, std::string>::makeError(
        "RenderScene::commitInactiveStep: background cache allocation "
        "failed");
  }
  if (pending.graphTopologyVersion != sceneGraph_.topologyVersion()) {
    discardIncrementalCommit();
    return Result<bool, std::string>::makeError(
        "RenderScene::commitInactiveStep: scene mutated during finalization");
  }
  auto &components = sceneGraph_.renderableComponents_;
  const auto &nodes = sceneGraph_.nodes_;
  uint32_t processed = 0u;
  while (pending.nextRenderableSlot < components.slots.slotCount() &&
         processed < operationBudget) {
    const uint32_t index = pending.nextRenderableSlot++;
    ++processed;
    if (index < components.flatRenderableIndex.size()) {
      components.flatRenderableIndex[index] = kInvalidIndex;
    }
    if (!components.slots.isLive(index)) {
      continue;
    }
    const ModelRef model = components.models[index];
    const MaterialRef material = components.materials[index];
    const MaterialRef materialOverride = components.materialOverrides[index];
    const uint32_t nodeIndex = components.node[index];
    const glm::mat4 world = nodes.worldFromRoot[nodeIndex];
    pending.morphWeights.emplace_back();
    pending.morphWeights.back().assign(components.morphWeights[index].begin(),
                                       components.morphWeights[index].end());
    pending.skinPalettes.emplace_back();
    pending.skinPalettes.back().assign(components.skinPalette[index].begin(),
                                       components.skinPalette[index].end());
    const uint32_t flatIndex =
        static_cast<uint32_t>(pending.renderables.size());
    components.flatRenderableIndex[index] = flatIndex;
    const Renderable renderable{
        .id = makeRenderableId(index, components.slots.generation(index)),
        .node = makeNodeId(nodeIndex, nodes.slots.generation(nodeIndex)),
        .model = model,
        .material = material,
        .materialOverride = materialOverride,
        .morphWeights =
            std::span<const float>(pending.morphWeights.back().data(),
                                   pending.morphWeights.back().size()),
        .skinPalette =
            std::span<const glm::mat4>(pending.skinPalettes.back().data(),
                                       pending.skinPalettes.back().size()),
        .modelMatrix = world,
    };
    retainRenderableRefs(model, material, materialOverride);
    pending.renderableIndexById.emplace(renderable.id, flatIndex);
    pending.renderables.push_back(renderable);
  }
  if (pending.nextRenderableSlot < components.slots.slotCount()) {
    return Result<bool, std::string>::makeResult(false);
  }
  renderables_.swap(pending.renderables);
  renderableIndexById_.swap(pending.renderableIndexById);
  renderableMorphWeights_.swap(pending.morphWeights);
  renderableSkinPalettes_.swap(pending.skinPalettes);
  incrementalCommit_.reset();
  ++topologyVersion_;
  ++transformVersion_;
  ++deformationVersion_;
  sceneGraph_.renderableTopologyDirty_ = false;
  sceneGraph_.renderableTransformsDirty_ = false;
  sceneGraph_.renderableDeformationsDirty_ = false;
  commitPackedLights();
  return Result<bool, std::string>::makeResult(true);
}

bool RenderScene::retireInactiveStep(uint32_t maxOperations) noexcept {
  uint32_t processed = 0u;
  const uint32_t operationBudget = std::max(maxOperations, 1u);
  if (incrementalCommit_ != nullptr) {
    IncrementalCommitState &pending = *incrementalCommit_;
    if (!pending.allocationReady.load(std::memory_order_acquire)) {
      return false;
    }
    while (retirementCursor_ < pending.renderables.size() &&
           processed < operationBudget) {
      Renderable &renderable = pending.renderables[retirementCursor_++];
      releaseRenderableRefs(renderable.model, renderable.material,
                            renderable.materialOverride);
      ++processed;
    }
    if (retirementCursor_ < pending.renderables.size()) {
      return false;
    }
    incrementalCommit_.reset();
    retirementCursor_ = 0u;
  }
  while (retirementCursor_ < renderables_.size() &&
         processed < operationBudget) {
    Renderable &renderable = renderables_[retirementCursor_++];
    releaseRenderableRefs(renderable.model, renderable.material,
                          renderable.materialOverride);
    ++processed;
  }
  if (retirementCursor_ < renderables_.size()) {
    return false;
  }
  if (!retirementEnvironmentReleased_) {
    setEnvironment(EnvironmentHandles{});
    retirementEnvironmentReleased_ = true;
  }
  return true;
}

void RenderScene::bindResources(ResourceManager *resources) {
  if (resources_ == resources) {
    return;
  }
  for (const Renderable &renderable : renderables_) {
    releaseRenderableRefs(renderable.model, renderable.material,
                          renderable.materialOverride);
  }
  releaseEnvironment(environment_);
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
  const EnvironmentHandles previousEnvironment = environment_;
  sanitizeTextureRef(environment_.cubemap);
  sanitizeTextureRef(environment_.irradiance);
  sanitizeTextureRef(environment_.prefilteredGgx);
  sanitizeTextureRef(environment_.prefilteredCharlie);
  sanitizeTextureRef(environment_.brdfLut);
  if (previousEnvironment != environment_) {
    ++environmentVersion_;
  }
  retainEnvironment(environment_);
}

void RenderScene::setEnvironment(EnvironmentHandles handles) {
  bool environmentChanged = false;
  const auto updateTextureRef =
      [this, &environmentChanged](TextureRef &currentRef, TextureRef nextRef) {
        if (currentRef.value == nextRef.value) {
          return;
        }
        releaseResourceIfOwned(resources_, currentRef);
        if (resources_ != nullptr && isValid(nextRef)) {
          if (!resourceAlive(resources_, nextRef)) {
            nextRef = kInvalidTextureRef;
          } else {
            resources_->retain(nextRef);
          }
        }
        currentRef = nextRef;
        environmentChanged = true;
      };
  updateTextureRef(environment_.cubemap, handles.cubemap);
  updateTextureRef(environment_.irradiance, handles.irradiance);
  updateTextureRef(environment_.prefilteredGgx, handles.prefilteredGgx);
  updateTextureRef(environment_.prefilteredCharlie, handles.prefilteredCharlie);
  updateTextureRef(environment_.brdfLut, handles.brdfLut);
  if (environmentChanged) {
    ++environmentVersion_;
  }
}

void RenderScene::noteLightTopologyChanged() noexcept {
  ++lightTopologyVersion_;
  ++lightTransformVersion_;
}

void RenderScene::noteLightTransformChanged() noexcept {
  ++lightTransformVersion_;
}

} // namespace nuri
