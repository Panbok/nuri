#pragma once

#include "nuri/core/result.h"
#include "nuri/text/text_renderer.h"

#include <filesystem>
#include <memory>
#include <memory_resource>
#include <string>
#include <string_view>

namespace nuri {

class NURI_API TextSystem {
public:
  struct CreateDesc {
    GPUDevice &gpu;
    std::pmr::memory_resource &memory;
    std::filesystem::path defaultFontPath;
    bool requireDefaultFont = false;
    TextRenderer::ShaderPaths shaderPaths;
  };

  static Result<std::unique_ptr<TextSystem>, std::string>
  create(const CreateDesc &desc);
  TextSystem() = default;
  virtual ~TextSystem() = default;

  TextSystem(const TextSystem &) = delete;
  TextSystem &operator=(const TextSystem &) = delete;
  TextSystem(TextSystem &&) = delete;
  TextSystem &operator=(TextSystem &&) = delete;

  virtual FontManager &fonts() = 0;
  virtual TextRenderer &renderer() = 0;
  virtual FontHandle defaultFont() const = 0;
  virtual Result<FontHandle, std::string>
  loadAndSetDefaultFont(std::string_view nfontPath,
                        std::string_view debugName = {}) = 0;
  virtual float defaultFontSizePx() const = 0;
  virtual void setDefaultFontSizePx(float sizePx) = 0;
};

} // namespace nuri
