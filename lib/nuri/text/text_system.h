#pragma once

#include "nuri/core/result.h"
#include "nuri/text/text_layouter.h"
#include "nuri/text/text_renderer.h"
#include "nuri/text/text_shaper.h"

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
  ~TextSystem();

  TextSystem(const TextSystem &) = delete;
  TextSystem &operator=(const TextSystem &) = delete;
  TextSystem(TextSystem &&) = delete;
  TextSystem &operator=(TextSystem &&) = delete;

  FontManager &fonts();
  TextRenderer &renderer();
  FontHandle defaultFont() const;
  Result<FontHandle, std::string>
  loadAndSetDefaultFont(std::string_view fontPath,
                        std::string_view debugName = {});
  float defaultFontSizePx() const;
  void setDefaultFontSizePx(float sizePx);

private:
  explicit TextSystem(CreateDesc desc);
  Result<bool, std::string> initialize();

  GPUDevice &gpu_;
  std::pmr::memory_resource &memory_;
  std::filesystem::path defaultFontPath_;
  bool requireDefaultFont_ = false;
  TextRenderer::ShaderPaths shaderPaths_{};
  FontHandle defaultFont_ = kInvalidFontHandle;
  float defaultFontSizePx_ = 42.0f;
  std::unique_ptr<FontManager> fonts_;
  std::unique_ptr<TextShaper> shaper_;
  std::unique_ptr<TextLayouter> layouter_;
  std::unique_ptr<TextRenderer> renderer_;
};

} // namespace nuri
