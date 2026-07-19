#pragma once
#include "nuri/core/containers/hash_map.h"
#include "nuri/defines.h"
#include "nuri/scene/render_scene.h"
#include "nuri/sim/simulation_bindings.h"
#include <cstdint>
#include <memory_resource>
#include <span>
#include <vector>
namespace nuri {

using RuntimeNodeBindingIndex = uint32_t;
using RuntimeRenderableBindingIndex = uint32_t;

class NURI_API SceneRuntimeBindings {
public:
  explicit SceneRuntimeBindings(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  void clear();
  [[nodiscard]] bool rebuild(const RenderScene *scene);
  [[nodiscard]] RuntimeNodeBindingIndex
  runtimeNodeIndex(NodeId node) const noexcept;
  [[nodiscard]] RuntimeRenderableBindingIndex
  runtimeRenderableIndex(RenderableId renderable) const noexcept;
  [[nodiscard]] bool contains(NodeId node) const noexcept;
  [[nodiscard]] bool contains(RenderableId renderable) const noexcept;
  [[nodiscard]] uint64_t version() const noexcept { return version_; }
  [[nodiscard]] std::span<const NodeId> nodes() const noexcept {
    return std::span<const NodeId>(nodes_);
  }
  [[nodiscard]] std::span<const RenderableId> renderables() const noexcept {
    return std::span<const RenderableId>(renderables_);
  }

private:
  std::pmr::memory_resource *memory_ = std::pmr::get_default_resource();
  std::pmr::vector<NodeId> nodes_;
  std::pmr::vector<RenderableId> renderables_;
  PmrHashMap<NodeId, RuntimeNodeBindingIndex> nodeIndexById_;
  PmrHashMap<RenderableId, RuntimeRenderableBindingIndex> renderableIndexById_;
  uint64_t version_ = 0u;
};

} // namespace nuri
