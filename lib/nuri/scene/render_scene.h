#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/resources/gpu/resource_handles.h"
#include "nuri/scene/light.h"

#include <cstdint>
#include <limits>
#include <memory_resource>
#include <span>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace nuri {
class ResourceManager;

struct NURI_API Renderable {
  ModelRef model = kInvalidModelRef;
  MaterialRef material = kInvalidMaterialRef;
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
  explicit RenderScene(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~RenderScene();

  RenderScene(const RenderScene &) = delete;
  RenderScene &operator=(const RenderScene &) = delete;
  RenderScene(RenderScene &&) = delete;
  RenderScene &operator=(RenderScene &&) = delete;

  [[nodiscard]] Result<uint32_t, std::string>
  addRenderable(ModelRef model, MaterialRef material,
                const glm::mat4 &modelMatrix = glm::mat4(1.0f));
  [[nodiscard]] Result<uint32_t, std::string>
  addRenderablesInstanced(ModelRef model, MaterialRef material,
                          std::span<const glm::mat4> modelMatrices);
  [[nodiscard]] bool setRenderableTransform(uint32_t index,
                                            const glm::mat4 &modelMatrix);
  [[nodiscard]] Result<LightId, std::string> addLight(const LightDesc &desc);
  [[nodiscard]] bool removeLight(LightId id);
  [[nodiscard]] bool getLightDesc(LightId id, LightDesc &out) const;
  [[nodiscard]] bool updateLight(LightId id, const LightDesc &desc);
  [[nodiscard]] bool setLightTransform(LightId id, const glm::vec3 &position,
                                       const glm::quat &rotation);

  [[nodiscard]] const Renderable *renderable(uint32_t index) const;
  [[nodiscard]] std::span<const Renderable> renderables() const {
    return renderables_;
  }
  template <typename Fn> void forEachLightId(Fn &&fn) const {
    for (uint32_t index = 0; index < directionalLights_.generations.size();
         ++index) {
      if (directionalLights_.live[index] == 0u) {
        continue;
      }
      fn(makeLightId(LightType::Directional, index,
                     directionalLights_.generations[index]));
    }
    for (uint32_t index = 0; index < pointLights_.generations.size(); ++index) {
      if (pointLights_.live[index] == 0u) {
        continue;
      }
      fn(makeLightId(LightType::Point, index, pointLights_.generations[index]));
    }
    for (uint32_t index = 0; index < spotLights_.generations.size(); ++index) {
      if (spotLights_.live[index] == 0u) {
        continue;
      }
      fn(makeLightId(LightType::Spot, index, spotLights_.generations[index]));
    }
  }
  void clearRenderables();
  void clearLights();
  [[nodiscard]] uint64_t topologyVersion() const noexcept {
    return topologyVersion_;
  }
  [[nodiscard]] uint64_t transformVersion() const noexcept {
    return transformVersion_;
  }
  [[nodiscard]] std::span<const DirectionalLightGpuData>
  packedDirectionalLights() const noexcept {
    return std::span<const DirectionalLightGpuData>(
        directionalLights_.packedGpu.data(),
        directionalLights_.packedGpu.size());
  }
  [[nodiscard]] std::span<const LocalLightGpuData>
  packedLocalLights() const noexcept {
    return std::span<const LocalLightGpuData>(packedLocalLights_.data(),
                                              packedLocalLights_.size());
  }
  [[nodiscard]] std::span<const LightId>
  packedDirectionalLightIds() const noexcept {
    return std::span<const LightId>(directionalLights_.packedIds.data(),
                                    directionalLights_.packedIds.size());
  }
  [[nodiscard]] std::span<const LightId> packedLocalLightIds() const noexcept {
    return std::span<const LightId>(packedLocalLightIds_.data(),
                                    packedLocalLightIds_.size());
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
  static constexpr uint32_t kInvalidPackedLightIndex =
      std::numeric_limits<uint32_t>::max();

  struct DirectionalLightStore {
    explicit DirectionalLightStore(
        std::pmr::memory_resource *memory = std::pmr::get_default_resource())
        : generations(memory), live(memory), freeSlots(memory),
          packedIndices(memory), names(memory), positions(memory),
          rotations(memory), colors(memory), intensities(memory),
          enabled(memory), packedGpu(memory), packedIds(memory) {}

    std::pmr::vector<uint32_t> generations;
    std::pmr::vector<uint8_t> live;
    std::pmr::vector<uint32_t> freeSlots;
    std::pmr::vector<uint32_t> packedIndices;
    std::pmr::vector<std::pmr::string> names;
    std::pmr::vector<glm::vec3> positions;
    std::pmr::vector<glm::quat> rotations;
    std::pmr::vector<glm::vec3> colors;
    std::pmr::vector<float> intensities;
    std::pmr::vector<uint8_t> enabled;
    std::pmr::vector<DirectionalLightGpuData> packedGpu;
    std::pmr::vector<LightId> packedIds;
  };

  struct PointLightStore {
    explicit PointLightStore(
        std::pmr::memory_resource *memory = std::pmr::get_default_resource())
        : generations(memory), live(memory), freeSlots(memory),
          packedIndices(memory), names(memory), positions(memory),
          rotations(memory), colors(memory), intensities(memory),
          ranges(memory), enabled(memory) {}

    std::pmr::vector<uint32_t> generations;
    std::pmr::vector<uint8_t> live;
    std::pmr::vector<uint32_t> freeSlots;
    std::pmr::vector<uint32_t> packedIndices;
    std::pmr::vector<std::pmr::string> names;
    std::pmr::vector<glm::vec3> positions;
    std::pmr::vector<glm::quat> rotations;
    std::pmr::vector<glm::vec3> colors;
    std::pmr::vector<float> intensities;
    std::pmr::vector<float> ranges;
    std::pmr::vector<uint8_t> enabled;
  };

  struct SpotLightStore {
    explicit SpotLightStore(
        std::pmr::memory_resource *memory = std::pmr::get_default_resource())
        : generations(memory), live(memory), freeSlots(memory),
          packedIndices(memory), names(memory), positions(memory),
          rotations(memory), colors(memory), intensities(memory),
          ranges(memory), innerConeAngles(memory), outerConeAngles(memory),
          enabled(memory) {}

    std::pmr::vector<uint32_t> generations;
    std::pmr::vector<uint8_t> live;
    std::pmr::vector<uint32_t> freeSlots;
    std::pmr::vector<uint32_t> packedIndices;
    std::pmr::vector<std::pmr::string> names;
    std::pmr::vector<glm::vec3> positions;
    std::pmr::vector<glm::quat> rotations;
    std::pmr::vector<glm::vec3> colors;
    std::pmr::vector<float> intensities;
    std::pmr::vector<float> ranges;
    std::pmr::vector<float> innerConeAngles;
    std::pmr::vector<float> outerConeAngles;
    std::pmr::vector<uint8_t> enabled;
  };

  [[nodiscard]] bool directionalSlotValid(LightId id) const noexcept;
  [[nodiscard]] bool pointSlotValid(LightId id) const noexcept;
  [[nodiscard]] bool spotSlotValid(LightId id) const noexcept;
  void rebuildPackedDirectionalLights();
  void rebuildPackedLocalLights();
  void noteLightTopologyChanged() noexcept;
  void noteLightTransformChanged() noexcept;
  void retainRenderable(const Renderable &renderable);
  void releaseRenderable(const Renderable &renderable);
  void retainEnvironment(const EnvironmentHandles &handles);
  void releaseEnvironment(const EnvironmentHandles &handles);

  std::pmr::memory_resource *memory_ = std::pmr::get_default_resource();
  std::pmr::vector<Renderable> renderables_;
  DirectionalLightStore directionalLights_;
  PointLightStore pointLights_;
  SpotLightStore spotLights_;
  std::pmr::vector<LocalLightGpuData> packedLocalLights_;
  std::pmr::vector<LightId> packedLocalLightIds_;
  ResourceManager *resources_ = nullptr;
  EnvironmentHandles environment_{};
  uint64_t topologyVersion_ = 0;
  uint64_t transformVersion_ = 0;
  uint64_t lightTopologyVersion_ = 0;
  uint64_t lightTransformVersion_ = 0;
};

} // namespace nuri
