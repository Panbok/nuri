#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_render_types.h"

#include <cstdint>
#include <memory>
#include <string>

namespace nuri {

class GPUDevice;

class NURI_API Renderer {
public:
  explicit Renderer(GPUDevice &gpu);
  ~Renderer() = default;

  Renderer(const Renderer &) = delete;
  Renderer &operator=(const Renderer &) = delete;
  Renderer(Renderer &&) = delete;
  Renderer &operator=(Renderer &&) = delete;

  static std::unique_ptr<Renderer> create(GPUDevice &gpu) {
    return std::make_unique<Renderer>(gpu);
  }

  Result<bool, std::string> render(const RenderFrame &frame);

  void onResize(uint32_t width, uint32_t height);

private:
  GPUDevice &gpu_;
};

} // namespace nuri
