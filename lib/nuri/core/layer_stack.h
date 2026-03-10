#pragma once

#include "nuri/core/layer.h"
#include "nuri/defines.h"
#include "nuri/gfx/render_graph/render_graph.h"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <memory_resource>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace nuri {

namespace detail {

template <typename ResultT>
concept LayerReverseResult = requires(
    std::remove_reference_t<ResultT> &result) {
  { result.hasError() } -> std::convertible_to<bool>;
  { result.error() } -> std::convertible_to<std::string>;
};

} // namespace detail

class NURI_API LayerStack {
public:
  explicit LayerStack(
      std::pmr::memory_resource *mem = std::pmr::get_default_resource());
  ~LayerStack();

  LayerStack(const LayerStack &) = delete;
  LayerStack &operator=(const LayerStack &) = delete;
  LayerStack(LayerStack &&) = delete;
  LayerStack &operator=(LayerStack &&) = delete;

  Layer *pushLayer(std::unique_ptr<Layer> layer);
  Layer *pushOverlay(std::unique_ptr<Layer> layer);
  bool popLayer(Layer *layer);
  bool popOverlay(Layer *layer);
  void clear();

  bool empty() const { return layers_.empty(); }
  size_t size() const { return layers_.size(); }

  template <typename Fn> void forEachLayerReverse(Fn &&fn) const {
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
      if (Layer *layer = it->get(); layer != nullptr) {
        fn(*layer);
      }
    }
  }

  template <typename Fn>
    requires std::invocable<Fn &, Layer &> &&
             detail::LayerReverseResult<std::invoke_result_t<Fn &, Layer &>>
  Result<bool, std::string> forEachLayerReverseResult(Fn &&fn) const {
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
      if (Layer *layer = it->get(); layer != nullptr) {
        auto result = fn(*layer);
        if (result.hasError()) {
          return Result<bool, std::string>::makeError(result.error());
        }
      }
    }
    return Result<bool, std::string>::makeResult(true);
  }

  void onUpdate(double deltaTime);
  void onResize(int32_t width, int32_t height);
  bool onInput(const InputEvent &event);

  Result<bool, std::string> buildRenderGraph(RenderFrameContext &frame,
                                             RenderGraphBuilder &graph);

private:
  bool removeLayer(Layer *layer);

  std::pmr::vector<std::unique_ptr<Layer>> layers_;
  size_t overlayStart_ = 0;
};

} // namespace nuri
