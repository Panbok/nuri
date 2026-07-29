#include "nuri/scene_runtime/scene_runtime_bindings.h"
#include <algorithm>
#include <limits>
namespace nuri {
namespace {
void collectNodeSubtree(const SceneGraph &graph, NodeId node,
                        std::pmr::vector<NodeId> &outNodes,
                        std::pmr::vector<RenderableId> &outRenderables) {
  std::pmr::vector<NodeId> stack(outNodes.get_allocator().resource());
  std::pmr::vector<NodeId> children(outNodes.get_allocator().resource());
  stack.push_back(node);
  while (!stack.empty()) {
    const NodeId currentNode = stack.back();
    stack.pop_back();
    outNodes.push_back(currentNode);
    graph.forEachRenderableOnNode(currentNode, [&](RenderableId renderable) {
      outRenderables.push_back(renderable);
    });
    NodeId child = kInvalidNodeId;
    if (!graph.getNodeFirstChild(currentNode, child)) {
      continue;
    }
    children.clear();
    for (NodeId currentChild = child; isValid(currentChild);) {
      children.push_back(currentChild);
      NodeId sibling = kInvalidNodeId;
      if (!graph.getNodeNextSibling(currentChild, sibling)) {
        break;
      }
      currentChild = sibling;
    }
    for (size_t index = children.size(); index > 0; --index) {
      stack.push_back(children[index - 1]);
    }
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
      !spansEqual<NodeId>(nodes(), std::span<const NodeId>(discoveredNodes)) ||
      !spansEqual<RenderableId>(
          renderables(), std::span<const RenderableId>(discoveredRenderables));
  if (!changed) {
    return false;
  }
  nodes_ = std::move(discoveredNodes);
  renderables_ = std::move(discoveredRenderables);
  nodeIndexById_.clear();
  renderableIndexById_.clear();
  nodeIndexById_.reserve(nodes_.size());
  renderableIndexById_.reserve(renderables_.size());
  for (size_t index = 0; index < nodes_.size(); ++index) {
    nodeIndexById_.insert_or_assign(
        nodes_[index], static_cast<RuntimeNodeBindingIndex>(index));
  }
  for (size_t index = 0; index < renderables_.size(); ++index) {
    renderableIndexById_.insert_or_assign(
        renderables_[index], static_cast<RuntimeRenderableBindingIndex>(index));
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
