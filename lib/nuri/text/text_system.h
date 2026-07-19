#pragma once
#include "nuri/core/result.h"
#include "nuri/text/font_types.h"
#include <filesystem>
#include <memory>
#include <memory_resource>
#include <string>
#include <string_view>
namespace nuri {

class FontManager;
class GPUDevice;
class RenderPipeline;
class TextLayouter;
class TextRenderer;
class TextShaper;

struct TextShaderPaths {
  std::filesystem::path uiVertex;
  std::filesystem::path uiFragment;
  std::filesystem::path worldVertex;
  std::filesystem::path worldFragment;
};

class NURI_API TextSystem {
public:
  struct CreateDesc {
    GPUDevice &gpu;
    std::pmr::memory_resource &memory;
    std::filesystem::path defaultFontPath;
    bool requireDefaultFont = false;
    TextShaderPaths shaderPaths;
  };
  static Result<std::unique_ptr<TextSystem>, std::string>
  create(const CreateDesc &desc);
  ~TextSystem();
  TextSystem(const TextSystem &) = delete;
  TextSystem &operator=(const TextSystem &) = delete;
  TextSystem(TextSystem &&) = delete;
  TextSystem &operator=(TextSystem &&) = delete;
  void beginFrame(uint64_t frameIndex);
  Result<TextBounds, std::string> enqueue2D(const Text2DDesc &desc,
                                            std::pmr::memory_resource &scratch);
  Result<TextBounds, std::string> enqueue3D(const Text3DDesc &desc,
                                            std::pmr::memory_resource &scratch);
  FontHandle defaultFont() const;
  Result<FontHandle, std::string>
  loadAndSetDefaultFont(std::string_view fontPath,
                        std::string_view debugName = {});
  float defaultFontSizePx() const;
  void setDefaultFontSizePx(float sizePx);

private:
  TextSystem(GPUDevice &gpu, std::pmr::memory_resource &memory);
  Result<bool, std::string> initialize(const CreateDesc &desc);
  friend void registerText3DStage(RenderPipeline &pipeline, TextSystem &text);
  friend void registerText2DStage(RenderPipeline &pipeline, TextSystem &text);
  GPUDevice &gpu_;
  std::pmr::memory_resource &memory_;
  FontHandle defaultFont_ = kInvalidFontHandle;
  float defaultFontSizePx_ = 42.0f;
  std::unique_ptr<FontManager> fonts_;
  std::unique_ptr<TextShaper> shaper_;
  std::unique_ptr<TextLayouter> layouter_;
  std::unique_ptr<TextRenderer> renderer_;
};

NURI_API void registerText3DStage(RenderPipeline &pipeline, TextSystem &text);
NURI_API void registerText2DStage(RenderPipeline &pipeline, TextSystem &text);

} // namespace nuri
