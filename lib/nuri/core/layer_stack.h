#pragma once

#include "nuri/core/layer.h"
#include "nuri/defines.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <memory_resource>
#include <string>
#include <vector>

namespace nuri {

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

  void onUpdate(double deltaTime);
  void onResize(int32_t width, int32_t height);
  bool onInput(const InputEvent &event);

  Result<bool, std::string> appendRenderPasses(RenderFrameContext &frame,
                                               RenderPassList &out);

private:
  bool removeLayer(Layer *layer);

  std::pmr::vector<std::unique_ptr<Layer>> layers_;
  size_t overlayStart_ = 0;
};

} // namespace nuri
