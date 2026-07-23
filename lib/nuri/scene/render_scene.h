#pragma once
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/resources/gpu/resource_handles.h"
#include "nuri/scene/ddgi_coverage_bounds.h"
#include "nuri/scene/light.h"
#include "nuri/scene/scene_graph.h"
#include <cstdint>
#include <glm/glm.hpp>
#include <limits>
#include <memory>
#include <memory_resource>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
namespace nuri {
class ResourceManager;

struct NURI_API Renderable {
  RenderableId id = kInvalidRenderableId;
  NodeId node = kInvalidNodeId;
  ModelRef model = kInvalidModelRef;
  MaterialRef material = kInvalidMaterialRef;
  MaterialRef materialOverride = kInvalidMaterialRef;
  std::span<const float> morphWeights{};
  std::span<const glm::mat4> skinPalette{};
  glm::mat4 modelMatrix{1.0f};
};

struct NURI_API EnvironmentHandles {
  TextureRef cubemap = kInvalidTextureRef;
  TextureRef irradiance = kInvalidTextureRef;
  TextureRef prefilteredGgx = kInvalidTextureRef;
  TextureRef prefilteredCharlie = kInvalidTextureRef;
  TextureRef brdfLut = kInvalidTextureRef;
  constexpr bool operator==(const EnvironmentHandles &) const = default;
};

struct NURI_API RenderDDGIVolume {
  DDGIVolumeId id = kInvalidDDGIVolumeId;
  NodeId node = kInvalidNodeId;
  std::string_view name{};
  glm::uvec3 probeCounts{16u, 8u, 16u};
  glm::vec3 probeSpacing{2.0f};
  float blendDistance = 2.0f;
  float maxRayDistance = 20.0f;
  int32_t priority = 0;
  DDGIVolumeMode mode = DDGIVolumeMode::Authored;
  glm::mat4 worldFromLocal{1.0f};
};

class NURI_API RenderScene {
public:
  explicit RenderScene(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~RenderScene();
  RenderScene(const RenderScene &) = delete;
  RenderScene &operator=(const RenderScene &) = delete;
  RenderScene(RenderScene &&) = delete;
  RenderScene &operator=(RenderScene &&) = delete;
  [[nodiscard]] SceneGraph &graph() noexcept { return sceneGraph_; }
  [[nodiscard]] const SceneGraph &graph() const noexcept { return sceneGraph_; }
  [[nodiscard]] Result<bool, std::string> commit();
  [[nodiscard]] Result<bool, std::string>
  commitInactiveStep(uint32_t maxOperations);
  [[nodiscard]] bool retireInactiveStep(uint32_t maxOperations) noexcept;
  [[nodiscard]] const Renderable *renderable(uint32_t index) const;
  [[nodiscard]] std::optional<uint32_t>
  findRenderableIndex(RenderableId id) const;
  [[nodiscard]] std::span<const Renderable> renderables() const {
    return renderables_;
  }
  template <typename Fn> void forEachLightId(Fn &&fn) const {
    sceneGraph_.forEachLightId(std::forward<Fn>(fn));
  }
  [[nodiscard]] std::span<const RenderDDGIVolume> ddgiVolumes() const noexcept {
    return ddgiVolumes_;
  }
  [[nodiscard]] const DDGISceneCoverageBounds &
  ddgiCurrentCoverageBounds() const noexcept {
    return ddgiCurrentCoverageBounds_;
  }
  [[nodiscard]] const DDGISceneCoverageBounds &
  ddgiActivationCoverageBounds() const noexcept {
    return ddgiActivationCoverageBounds_;
  }
  [[nodiscard]] const DDGISceneCoverageBounds &
  ddgiStaticCoverageBounds() const noexcept {
    return ddgiStaticCoverageBounds_;
  }
  [[nodiscard]] const DDGISceneCoverageBounds &
  ddgiPendingStaticCoverageBounds() const noexcept {
    return ddgiPendingStaticCoverageBounds_;
  }
  [[nodiscard]] bool ddgiActivationCoverageBoundsSealed() const noexcept {
    return ddgiActivationCoverageBoundsSealed_;
  }
  [[nodiscard]] bool sealDDGIActivationCoverageBounds() noexcept;
  [[nodiscard]] bool refitDDGIActivationCoverageBounds() noexcept;
  [[nodiscard]] bool resetDDGIActivationCoverageBounds() noexcept;
  [[nodiscard]] bool
  stageDDGIStaticCoverageBounds(const DDGISceneCoverageBounds &bounds) noexcept;
  [[nodiscard]] bool refitDDGIStaticCoverageBounds() noexcept;
  [[nodiscard]] uint64_t topologyVersion() const noexcept {
    return topologyVersion_;
  }
  [[nodiscard]] uint64_t id() const noexcept { return id_; }
  [[nodiscard]] uint64_t transformVersion() const noexcept {
    return transformVersion_;
  }
  [[nodiscard]] uint64_t deformationVersion() const noexcept {
    return deformationVersion_;
  }
  [[nodiscard]] std::span<const DirectionalLightGpuData>
  packedDirectionalLights() const noexcept {
    return packedDirectionalLights_;
  }
  [[nodiscard]] std::span<const LocalLightGpuData>
  packedLocalLights() const noexcept {
    return packedLocalLights_;
  }
  [[nodiscard]] std::span<const LightId>
  packedDirectionalLightIds() const noexcept {
    return packedDirectionalLightIds_;
  }
  [[nodiscard]] std::span<const LightId> packedLocalLightIds() const noexcept {
    return packedLocalLightIds_;
  }
  [[nodiscard]] uint64_t lightTopologyVersion() const noexcept {
    return lightTopologyVersion_;
  }
  [[nodiscard]] uint64_t lightTransformVersion() const noexcept {
    return lightTransformVersion_;
  }
  [[nodiscard]] uint64_t ddgiVolumeTopologyVersion() const noexcept {
    return ddgiVolumeTopologyVersion_;
  }
  [[nodiscard]] uint64_t ddgiVolumeTransformVersion() const noexcept {
    return ddgiVolumeTransformVersion_;
  }
  [[nodiscard]] uint64_t ddgiVolumeSettingsVersion() const noexcept {
    return ddgiVolumeSettingsVersion_;
  }
  void bindResources(ResourceManager *resources);
  [[nodiscard]] const ResourceManager *resources() const noexcept {
    return resources_;
  }
  void setEnvironment(EnvironmentHandles handles);
  [[nodiscard]] const EnvironmentHandles &environment() const noexcept {
    return environment_;
  }
  [[nodiscard]] uint64_t environmentVersion() const noexcept {
    return environmentVersion_;
  }

private:
  struct IncrementalCommitState;
  static constexpr uint32_t kInvalidIndex =
      std::numeric_limits<uint32_t>::max();
  void rebuildFlatRenderables();
  void rebuildPackedDirectionalLights();
  void rebuildPackedLocalLights();
  bool commitPackedLights();
  bool commitDDGIVolumes();
  bool commitDDGICoverageBounds(bool renderableFactsChanged) noexcept;
  void rebuildDDGIVolumes();
  void sanitizeGraphRenderableRefs();
  void noteLightTopologyChanged() noexcept;
  void noteLightTransformChanged() noexcept;
  void retainRenderableRefs(ModelRef model, MaterialRef material,
                            MaterialRef materialOverride);
  void releaseRenderableRefs(ModelRef model, MaterialRef material,
                             MaterialRef materialOverride);
  void retainEnvironment(const EnvironmentHandles &handles);
  void releaseEnvironment(const EnvironmentHandles &handles);
  void discardIncrementalCommit() noexcept;
  std::pmr::memory_resource *memory_ = std::pmr::get_default_resource();
  SceneGraph sceneGraph_;
  std::pmr::vector<Renderable> renderables_;
  std::pmr::unordered_map<RenderableId, uint32_t> renderableIndexById_;
  std::pmr::vector<std::pmr::vector<float>> renderableMorphWeights_;
  std::pmr::vector<std::pmr::vector<glm::mat4>> renderableSkinPalettes_;
  std::pmr::vector<DirectionalLightGpuData> packedDirectionalLights_;
  std::pmr::vector<LocalLightGpuData> packedLocalLights_;
  std::pmr::vector<LightId> packedDirectionalLightIds_;
  std::pmr::vector<LightId> packedLocalLightIds_;
  std::pmr::vector<RenderDDGIVolume> ddgiVolumes_;
  ResourceManager *resources_ = nullptr;
  EnvironmentHandles environment_{};
  uint64_t id_ = 0u;
  uint64_t topologyVersion_ = 0u;
  uint64_t transformVersion_ = 0u;
  uint64_t deformationVersion_ = 0u;
  uint64_t lightTopologyVersion_ = 0u;
  uint64_t lightTransformVersion_ = 0u;
  uint64_t ddgiVolumeTopologyVersion_ = 0u;
  uint64_t ddgiVolumeTransformVersion_ = 0u;
  uint64_t ddgiVolumeSettingsVersion_ = 0u;
  uint64_t environmentVersion_ = 0u;
  DDGISceneCoverageBounds ddgiCurrentCoverageBounds_{};
  DDGISceneCoverageBounds ddgiActivationCoverageBounds_{};
  DDGISceneCoverageBounds ddgiStaticCoverageBounds_{};
  DDGISceneCoverageBounds ddgiPendingStaticCoverageBounds_{};
  uint64_t ddgiCoverageBoundsGeneration_ = 0u;
  uint64_t ddgiCoverageMaterialVersion_ = UINT64_MAX;
  uint64_t ddgiCoverageModelMaterialBindingVersion_ = UINT64_MAX;
  std::shared_ptr<IncrementalCommitState> incrementalCommit_{};
  uint32_t retirementCursor_ = 0u;
  bool retirementEnvironmentReleased_ = false;
  bool ddgiCoverageFactsDirty_ = true;
  bool ddgiActivationCoverageBoundsSealed_ = false;
};

} // namespace nuri
