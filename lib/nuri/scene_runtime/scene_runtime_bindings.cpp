#include "nuri/pch.h"

#include "nuri/scene_runtime/scene_runtime_bindings.h"

#include <algorithm>

namespace nuri {
namespace {

void collectNodeSubtree(const SceneGraph &graph, NodeId node,
                        std::pmr::vector<NodeId> &outNodes,
                        std::pmr::vector<RenderableId> &outRenderables) {
  outNodes.push_back(node);
  graph.forEachRenderableOnNode(node, [&](RenderableId renderable) {
    outRenderables.push_back(renderable);
  });

  NodeId child = kInvalidNodeId;
  if (!graph.getNodeFirstChild(node, child)) {
    return;
  }
  for (NodeId current = child; isValid(current);) {
    collectNodeSubtree(graph, current, outNodes, outRenderables);

    NodeId sibling = kInvalidNodeId;
    if (!graph.getNodeNextSibling(current, sibling)) {
      break;
    }
    current = sibling;
  }
}

template <typename T>
[[nodiscard]] bool spansEqual(std::span<const T> lhs,
                              std::span<const T> rhs) noexcept {
  return lhs.size() == rhs.size() &&
         std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
}

} // namespace

SceneRuntimeBindings::SceneRuntimeBindings(std::pmr::memory_resource *memory)
    : memory_(memory != nullptr ? memory : std::pmr::get_default_resource()),
      nodes_(memory_), renderables_(memory_), nodeIndexById_(memory_),
      renderableIndexById_(memory_) {}

void SceneRuntimeBindings::clear() {
  nodes_.clear();
  renderables_.clear();
  nodeIndexById_.clear();
  renderableIndexById_.clear();
}

bool SceneRuntimeBindings::rebuild(const RenderScene *scene) {
  if (scene == nullptr) {
    const bool changed = !nodes_.empty() || !renderables_.empty();
    clear();
    if (changed) {
      ++version_;
    }
    return changed;
  }

  std::pmr::vector<NodeId> discoveredNodes(memory_);
  std::pmr::vector<RenderableId> discoveredRenderables(memory_);
  const SceneGraph &graph = scene->graph();
  const NodeId root = graph.rootNode();
  if (isValid(root)) {
    collectNodeSubtree(graph, root, discoveredNodes, discoveredRenderables);
  }

  const bool changed =
      !spansEqual<NodeId>(std::span<const NodeId>(nodes_.data(), nodes_.size()),
                          std::span<const NodeId>(discoveredNodes.data(),
                                                  discoveredNodes.size())) ||
      !spansEqual<RenderableId>(
          std::span<const RenderableId>(renderables_.data(),
                                        renderables_.size()),
          std::span<const RenderableId>(discoveredRenderables.data(),
                                        discoveredRenderables.size()));
  if (!changed) {
    return false;
  }

  nodes_ = std::move(discoveredNodes);
  renderables_ = std::move(discoveredRenderables);
  nodeIndexById_.clear();
  renderableIndexById_.clear();
  nodeIndexById_.reserve(nodes_.size());
  renderableIndexById_.reserve(renderables_.size());
  for (uint32_t index = 0; index < nodes_.size(); ++index) {
    nodeIndexById_.insert_or_assign(nodes_[index], index);
  }
  for (uint32_t index = 0; index < renderables_.size(); ++index) {
    renderableIndexById_.insert_or_assign(renderables_[index], index);
  }
  ++version_;
  return true;
}

RuntimeNodeBindingIndex
SceneRuntimeBindings::runtimeNodeIndex(NodeId node) const noexcept {
  const auto it = nodeIndexById_.find(node);
  return it == nodeIndexById_.end() ? kInvalidSimulationBindingIndex
                                    : it->second;
}

RuntimeRenderableBindingIndex SceneRuntimeBindings::runtimeRenderableIndex(
    RenderableId renderable) const noexcept {
  const auto it = renderableIndexById_.find(renderable);
  return it == renderableIndexById_.end() ? kInvalidSimulationBindingIndex
                                          : it->second;
}

bool SceneRuntimeBindings::contains(NodeId node) const noexcept {
  return runtimeNodeIndex(node) != kInvalidSimulationBindingIndex;
}

bool SceneRuntimeBindings::contains(RenderableId renderable) const noexcept {
  return runtimeRenderableIndex(renderable) != kInvalidSimulationBindingIndex;
}

} // namespace nuri
