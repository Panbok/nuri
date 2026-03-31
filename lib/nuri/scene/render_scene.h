#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/resources/gpu/resource_handles.h"
#include "nuri/scene/light.h"
#include "nuri/scene/scene_graph.h"

#include <cstdint>
#include <limits>
#include <memory_resource>
#include <span>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

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
};

class NURI_API RenderScene {
public:
  // RenderScene owns the committed renderer-facing caches derived from the
  // authored SceneGraph plus scene-global environment bindings.
  explicit RenderScene(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~RenderScene();

  RenderScene(const RenderScene &) = delete;
  RenderScene &operator=(const RenderScene &) = delete;
  RenderScene(RenderScene &&) = delete;
  RenderScene &operator=(RenderScene &&) = delete;

  [[nodiscard]] SceneGraph &graph() noexcept { return sceneGraph_; }
  [[nodiscard]] const SceneGraph &graph() const noexcept { return sceneGraph_; }

  // commit() returns true when derived renderer-facing caches changed and false
  // when the authored scene state was already up to date.
  [[nodiscard]] Result<bool, std::string> commit();

  [[nodiscard]] const Renderable *renderable(uint32_t index) const;
  [[nodiscard]] std::span<const Renderable> renderables() const {
    return renderables_;
  }

  template <typename Fn> void forEachLightId(Fn &&fn) const {
    sceneGraph_.forEachLightId(std::forward<Fn>(fn));
  }

  [[nodiscard]] uint64_t topologyVersion() const noexcept {
    return topologyVersion_;
  }
  [[nodiscard]] uint64_t transformVersion() const noexcept {
    return transformVersion_;
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

  void bindResources(ResourceManager *resources);

  void setEnvironment(EnvironmentHandles handles);
  [[nodiscard]] const EnvironmentHandles &environment() const noexcept {
    return environment_;
  }

private:
  static constexpr uint32_t kInvalidIndex =
      std::numeric_limits<uint32_t>::max();

  void rebuildFlatRenderables();
  void rebuildPackedDirectionalLights();
  void rebuildPackedLocalLights();
  void sanitizeGraphRenderableRefs();
  void noteLightTopologyChanged() noexcept;
  void noteLightTransformChanged() noexcept;
  void retainRenderableRefs(ModelRef model, MaterialRef material,
                            MaterialRef materialOverride);
  void releaseRenderableRefs(ModelRef model, MaterialRef material,
                             MaterialRef materialOverride);
  void retainEnvironment(const EnvironmentHandles &handles);
  void releaseEnvironment(const EnvironmentHandles &handles);

  std::pmr::memory_resource *memory_ = std::pmr::get_default_resource();
  SceneGraph sceneGraph_;
  std::pmr::vector<Renderable> renderables_;
  std::pmr::vector<std::pmr::vector<float>> renderableMorphWeights_;
  std::pmr::vector<std::pmr::vector<glm::mat4>> renderableSkinPalettes_;
  std::pmr::vector<DirectionalLightGpuData> packedDirectionalLights_;
  std::pmr::vector<LocalLightGpuData> packedLocalLights_;
  std::pmr::vector<LightId> packedDirectionalLightIds_;
  std::pmr::vector<LightId> packedLocalLightIds_;
  ResourceManager *resources_ = nullptr;
  EnvironmentHandles environment_{};
  uint64_t topologyVersion_ = 0u;
  uint64_t transformVersion_ = 0u;
  uint64_t lightTopologyVersion_ = 0u;
  uint64_t lightTransformVersion_ = 0u;
};

} // namespace nuri
