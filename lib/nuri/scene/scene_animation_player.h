#pragma once

#include "nuri/core/result.h"
#include "nuri/scene/scene_graph.h"
#include "nuri/scene/scene_prefab.h"

#include <cstdint>
#include <limits>
#include <memory_resource>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace nuri {

enum class AnimationPlaybackMode : uint8_t {
  Once = 0,
  Loop = 1,
};

class NURI_API SceneAnimationPlayer final {
public:
  // prefab and instantiationMap must outlive the player; this type keeps
  // non-owning pointers to both inputs.
  SceneAnimationPlayer(
      const ScenePrefab &prefab, const SceneInstantiationMap &instantiationMap,
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  SceneAnimationPlayer(const SceneAnimationPlayer &) = delete;
  SceneAnimationPlayer &operator=(const SceneAnimationPlayer &) = delete;
  SceneAnimationPlayer(SceneAnimationPlayer &&) = delete;
  SceneAnimationPlayer &operator=(SceneAnimationPlayer &&) = delete;

  [[nodiscard]] Result<bool, std::string>
  play(uint32_t clipIndex,
       AnimationPlaybackMode mode = AnimationPlaybackMode::Loop);
  void stop();
  void seek(float timeSeconds);
  void update(SceneGraph &graph, float deltaSeconds);

  [[nodiscard]] uint32_t clipIndex() const noexcept { return clipIndex_; }
  [[nodiscard]] float timeSeconds() const noexcept { return timeSeconds_; }
  [[nodiscard]] bool playing() const noexcept { return playing_; }

private:
  struct NodeState {
    glm::vec3 translation{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 scale{1.0f};
    std::pmr::vector<float> morphWeights;

    explicit NodeState(
        std::pmr::memory_resource *memory = std::pmr::get_default_resource())
        : morphWeights(memory) {}
  };

  [[nodiscard]] const AnimationClipData *activeClip() const noexcept;
  void applyClip(SceneGraph &graph) const;

  // Non-owning; constructor inputs must outlive this instance.
  const ScenePrefab *prefab_ = nullptr;
  const SceneInstantiationMap *instantiationMap_ = nullptr;
  std::pmr::memory_resource *memory_ = std::pmr::get_default_resource();
  std::pmr::vector<NodeState> baseNodeStates_;
  mutable std::pmr::vector<NodeState> sampledNodeStates_;
  uint32_t clipIndex_ = std::numeric_limits<uint32_t>::max();
  float timeSeconds_ = 0.0f;
  bool playing_ = false;
  AnimationPlaybackMode mode_ = AnimationPlaybackMode::Loop;
};

} // namespace nuri
