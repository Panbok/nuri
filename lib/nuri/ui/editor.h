#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/gpu_render_types.h"

#include <string>

namespace nuri {

class GPUDevice;
class Window;

class NURI_API Editor {
public:
  virtual ~Editor() = default;

  Editor(const Editor &) = delete;
  Editor &operator=(const Editor &) = delete;
  Editor(Editor &&) = delete;
  Editor &operator=(Editor &&) = delete;

  virtual void beginFrame() = 0;
  virtual Result<RenderPass, std::string> endFrame() = 0;

protected:
  Editor() = default;
};

} // namespace nuri
