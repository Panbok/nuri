#include "nuri/pch.h"

#include "nuri/core/layer_stack.h"
#include "nuri/core/log.h"

namespace nuri {

LayerStack::LayerStack(std::pmr::memory_resource *mem)
    : layers_(std::pmr::polymorphic_allocator<std::unique_ptr<Layer>>(mem)) {
  NURI_LOG_DEBUG("LayerStack::LayerStack: Layer stack created");
}

LayerStack::~LayerStack() {
  clear();
  NURI_LOG_DEBUG("LayerStack::~LayerStack: Layer stack destroyed");
}

Layer *LayerStack::pushLayer(std::unique_ptr<Layer> layer) {
  if (!layer) {
    return nullptr;
  }

  const auto insertPos =
      layers_.begin() + static_cast<std::ptrdiff_t>(overlayStart_);
  auto it = layers_.insert(insertPos, std::move(layer));
  ++overlayStart_;

  Layer *ptr = it->get();
  if (ptr) {
    ptr->onAttach();
  }
  return ptr;
}

Layer *LayerStack::pushOverlay(std::unique_ptr<Layer> layer) {
  if (!layer) {
    return nullptr;
  }

  layers_.push_back(std::move(layer));
  Layer *ptr = layers_.back().get();
  if (ptr) {
    ptr->onAttach();
  }
  return ptr;
}

bool LayerStack::popLayer(Layer *layer) { return removeLayer(layer); }

bool LayerStack::popOverlay(Layer *layer) { return removeLayer(layer); }

bool LayerStack::removeLayer(Layer *layer) {
  if (!layer) {
    return false;
  }

  auto it = std::find_if(layers_.begin(), layers_.end(),
                         [layer](const std::unique_ptr<Layer> &entry) {
                           return entry.get() == layer;
                         });
  if (it == layers_.end()) {
    return false;
  }

  const size_t index = static_cast<size_t>(it - layers_.begin());
  (*it)->onDetach();
  layers_.erase(it);

  if (index < overlayStart_ && overlayStart_ > 0) {
    --overlayStart_;
  }
  return true;
}

void LayerStack::onUpdate(double deltaTime) {
  for (auto &layer : layers_) {
    if (layer) {
      layer->onUpdate(deltaTime);
    }
  }
}

void LayerStack::onResize(int32_t width, int32_t height) {
  for (auto &layer : layers_) {
    if (layer) {
      layer->onResize(width, height);
    }
  }
}

bool LayerStack::onInput(const InputEvent &event) {
  for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
    Layer *layer = it->get();
    if (!layer) {
      continue;
    }
    if (layer->onInput(event)) {
      return true;
    }
  }
  return false;
}

void LayerStack::clear() {
  for (auto &layer : layers_) {
    if (layer) {
      layer->onDetach();
    }
  }
  layers_.clear();
  overlayStart_ = 0;
}

Result<bool, std::string>
LayerStack::appendRenderPasses(RenderFrameContext &frame, RenderPassList &out) {
  // Prepare in reverse input order so overlays can inject frame-scoped data
  // before render layers consume it.
  for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
    Layer *layer = it->get();
    if (!layer) {
      continue;
    }
    layer->prepareFrameContext(frame);
  }

  for (auto &layer : layers_) {
    if (!layer) {
      continue;
    }
    auto result = layer->buildRenderPasses(frame, out);
    if (result.hasError()) {
      return Result<bool, std::string>::makeError(result.error());
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri
