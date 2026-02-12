#pragma once

#include "nuri/core/input_events.h"
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_render_types.h"

#include <cstdint>
#include <memory_resource>
#include <string>
#include <vector>

namespace nuri {

using RenderPassList = std::pmr::vector<RenderPass>;

class NURI_API Layer {
public:
  virtual ~Layer() = default;

  Layer(const Layer &) = delete;
  Layer &operator=(const Layer &) = delete;
  Layer(Layer &&) = delete;
  Layer &operator=(Layer &&) = delete;

  virtual void onAttach() {}
  virtual void onDetach() {}
  virtual void onUpdate(double deltaTime) {}
  virtual void onResize(int32_t width, int32_t height) {}
  virtual bool onInput(const InputEvent &event) { return false; }
  virtual Result<bool, std::string> buildRenderPasses(RenderPassList &out);

protected:
  Layer() = default;
};

inline Result<bool, std::string> Layer::buildRenderPasses(RenderPassList &) {
  return Result<bool, std::string>::makeResult(true);
}

} // namespace nuri
