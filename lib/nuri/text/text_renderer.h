#pragma once

#include "nuri/core/layer.h"
#include "nuri/core/result.h"
#include "nuri/gfx/gpu_render_types.h"
#include "nuri/text/text_layouter.h"

#include <filesystem>
#include <memory>
#include <memory_resource>
#include <string>

namespace nuri {

class GPUDevice;

class NURI_API TextRenderer {
public:
  struct ShaderPaths {
    std::filesystem::path uiVertex;
    std::filesystem::path uiFragment;
    std::filesystem::path worldVertex;
    std::filesystem::path worldFragment;
  };

  struct CreateDesc {
    GPUDevice &gpu;
    FontManager &fonts;
    TextLayouter &layouter;
    std::pmr::memory_resource &memory;
    ShaderPaths shaderPaths;
  };

  static std::unique_ptr<TextRenderer> create(const CreateDesc &desc);
  virtual ~TextRenderer() = default;

  virtual Result<bool, std::string> beginFrame(uint64_t frameIndex) = 0;
  virtual Result<TextBounds, std::string>
  enqueue2D(const Text2DDesc &desc, std::pmr::memory_resource &scratch) = 0;
  virtual Result<TextBounds, std::string>
  enqueue3D(const Text3DDesc &desc, std::pmr::memory_resource &scratch) = 0;
  virtual Result<bool, std::string> append3DPasses(RenderFrameContext &frame,
                                                   RenderPassList &out) = 0;
  virtual Result<bool, std::string> append2DPasses(RenderFrameContext &frame,
                                                   RenderPassList &out) = 0;
  virtual void clear() = 0;
};

} // namespace nuri
