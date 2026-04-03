#pragma once

#include "nuri/defines.h"
#include "nuri/scene/scene_handles.h"

#include <array>
#include <cstdint>
#include <memory_resource>
#include <vector>

#include <glm/glm.hpp>

namespace nuri {

struct NURI_API NodeLocalTransformWrite {
  NodeId node = kInvalidNodeId;
  glm::mat4 localFromParent{1.0f};
  uint64_t versionStamp = 0u;
};

struct NURI_API RenderableDeformationWrite {
  RenderableId renderable = kInvalidRenderableId;
  uint32_t skinPaletteOffset = 0u;
  uint32_t skinPaletteCount = 0u;
  uint32_t morphWeightOffset = 0u;
  uint32_t morphWeightCount = 0u;
  uint64_t versionStamp = 0u;
};

struct NURI_API ScenePropertyWrite {
  uint64_t propertyId = 0u;
  uint64_t versionStamp = 0u;
};

class NURI_API SimulationWritebackState {
public:
  explicit SimulationWritebackState(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource())
      : nodeLocalTransforms_(memory), renderableDeformations_(memory),
        sceneProperties_(memory) {}

  void clear() {
    nodeLocalTransforms_.clear();
    renderableDeformations_.clear();
    sceneProperties_.clear();
  }

  [[nodiscard]] bool empty() const noexcept {
    return nodeLocalTransforms_.empty() && renderableDeformations_.empty() &&
           sceneProperties_.empty();
  }

  [[nodiscard]] std::pmr::vector<NodeLocalTransformWrite> &
  nodeLocalTransforms() noexcept {
    return nodeLocalTransforms_;
  }
  [[nodiscard]] const std::pmr::vector<NodeLocalTransformWrite> &
  nodeLocalTransforms() const noexcept {
    return nodeLocalTransforms_;
  }

  [[nodiscard]] std::pmr::vector<RenderableDeformationWrite> &
  renderableDeformations() noexcept {
    return renderableDeformations_;
  }
  [[nodiscard]] const std::pmr::vector<RenderableDeformationWrite> &
  renderableDeformations() const noexcept {
    return renderableDeformations_;
  }

  [[nodiscard]] std::pmr::vector<ScenePropertyWrite> &sceneProperties() noexcept {
    return sceneProperties_;
  }
  [[nodiscard]] const std::pmr::vector<ScenePropertyWrite> &
  sceneProperties() const noexcept {
    return sceneProperties_;
  }

private:
  std::pmr::vector<NodeLocalTransformWrite> nodeLocalTransforms_;
  std::pmr::vector<RenderableDeformationWrite> renderableDeformations_;
  std::pmr::vector<ScenePropertyWrite> sceneProperties_;
};

} // namespace nuri
