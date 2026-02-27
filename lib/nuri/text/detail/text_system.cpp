#include "nuri/pch.h"

#include "nuri/core/log.h"
#include "nuri/text/text_system.h"

namespace nuri {
namespace {

constexpr float MIN_FONT_SIZE_PX = 8.0f;
constexpr float MAX_FONT_SIZE_PX = 256.0f;
constexpr float DEFAULT_FONT_SIZE_PX = 42.0f;

template <typename T, typename... Args>
[[nodiscard]] Result<T, std::string> makeError(Args &&...args) {
  std::ostringstream oss;
  (oss << ... << std::forward<Args>(args));
  return Result<T, std::string>::makeError(oss.str());
}

class TextSystemImpl final : public TextSystem {
public:
  explicit TextSystemImpl(CreateDesc desc)
      : gpu_(desc.gpu), memory_(desc.memory),
        defaultFontPath_(std::move(desc.defaultFontPath)),
        requireDefaultFont_(desc.requireDefaultFont),
        shaderPaths_(std::move(desc.shaderPaths)) {}

  Result<bool, std::string> initialize() {
    auto fonts = FontManager::create(FontManager::CreateDesc{
        .gpu = gpu_,
        .memory = memory_,
    });
    if (!fonts) {
      return Result<bool, std::string>::makeError(
          "TextSystem: failed to create FontManager");
    }

    auto shaper = TextShaper::create(TextShaper::CreateDesc{
        .fonts = *fonts,
        .memory = memory_,
    });
    if (!shaper) {
      return Result<bool, std::string>::makeError(
          "TextSystem: failed to create TextShaper");
    }

    auto layouter = TextLayouter::create(TextLayouter::CreateDesc{
        .fonts = *fonts,
        .shaper = *shaper,
        .memory = memory_,
    });
    if (!layouter) {
      return Result<bool, std::string>::makeError(
          "TextSystem: failed to create TextLayouter");
    }

    auto renderer = std::make_unique<TextRenderer>(TextRenderer::CreateDesc{
        .gpu = gpu_,
        .fonts = *fonts,
        .layouter = *layouter,
        .memory = memory_,
        .shaderPaths = shaderPaths_,
    });

    if (!defaultFontPath_.empty()) {
      const std::string defaultFontPathString = defaultFontPath_.string();
      auto loadResult = fonts->loadFont(FontLoadDesc{
          .path = defaultFontPathString,
          .debugName = "default_ui",
          .memory = &memory_,
      });
      if (loadResult.hasError()) {
        if (requireDefaultFont_) {
          return makeError<bool>("TextSystem: failed to load default font '",
                                 defaultFontPathString, "' (",
                                 loadResult.error(), ")");
        }
        NURI_LOG_WARNING(
            "TextSystem: failed to load optional default font '%s': %s",
            defaultFontPathString.c_str(), loadResult.error().c_str());
      } else {
        defaultFont_ = loadResult.value();
      }
    }

    fonts_ = std::move(fonts);
    shaper_ = std::move(shaper);
    layouter_ = std::move(layouter);
    renderer_ = std::move(renderer);
    return Result<bool, std::string>::makeResult(true);
  }

  FontManager &fonts() override {
    NURI_ASSERT(fonts_ != nullptr, "TextSystem::fonts is not initialized");
    return *fonts_;
  }

  TextRenderer &renderer() override {
    NURI_ASSERT(renderer_ != nullptr,
                "TextSystem::renderer is not initialized");
    return *renderer_;
  }

  FontHandle defaultFont() const override { return defaultFont_; }

  Result<FontHandle, std::string>
  loadAndSetDefaultFont(std::string_view nfontPath,
                        std::string_view debugName) override {
    if (nfontPath.empty()) {
      return makeError<FontHandle>(
          "TextSystem::loadAndSetDefaultFont: nfontPath is empty");
    }
    if (fonts_ == nullptr) {
      return makeError<FontHandle>(
          "TextSystem::loadAndSetDefaultFont: FontManager is not initialized");
    }

    const std::filesystem::path path{std::string(nfontPath)};
    const std::string pathString = path.string();
    const std::string resolvedDebugName =
        !debugName.empty() ? std::string(debugName) : path.stem().string();
    auto loadResult = fonts_->loadFont(FontLoadDesc{
        .path = pathString,
        .debugName = resolvedDebugName,
        .memory = &memory_,
    });
    if (loadResult.hasError()) {
      return Result<FontHandle, std::string>::makeError(loadResult.error());
    }

    const FontHandle newDefault = loadResult.value();
    const FontHandle oldDefault = defaultFont_;
    defaultFont_ = newDefault;
    defaultFontPath_ = path;

    if (::nuri::isValid(oldDefault) && oldDefault.value != newDefault.value) {
      auto unloadResult = fonts_->unloadFont(oldDefault);
      if (unloadResult.hasError()) {
        NURI_LOG_WARNING(
            "TextSystem::loadAndSetDefaultFont: failed to unload previous "
            "default font: %s",
            unloadResult.error().c_str());
      }
    }

    NURI_LOG_INFO("TextSystem: default font switched to '%s'",
                  defaultFontPath_.string().c_str());
    return Result<FontHandle, std::string>::makeResult(newDefault);
  }

  float defaultFontSizePx() const override { return defaultFontSizePx_; }

  void setDefaultFontSizePx(float sizePx) override {
    defaultFontSizePx_ = std::clamp(sizePx, MIN_FONT_SIZE_PX, MAX_FONT_SIZE_PX);
  }

private:
  GPUDevice &gpu_;
  std::pmr::memory_resource &memory_;
  std::filesystem::path defaultFontPath_;
  bool requireDefaultFont_ = false;
  TextRenderer::ShaderPaths shaderPaths_{};
  FontHandle defaultFont_ = kInvalidFontHandle;
  float defaultFontSizePx_ = DEFAULT_FONT_SIZE_PX;
  std::unique_ptr<FontManager> fonts_;
  std::unique_ptr<TextShaper> shaper_;
  std::unique_ptr<TextLayouter> layouter_;
  std::unique_ptr<TextRenderer> renderer_;
};

} // namespace

Result<std::unique_ptr<TextSystem>, std::string>
TextSystem::create(const CreateDesc &desc) {
  auto system = std::make_unique<TextSystemImpl>(desc);
  auto initResult = system->initialize();
  if (initResult.hasError()) {
    return Result<std::unique_ptr<TextSystem>, std::string>::makeError(
        initResult.error());
  }
  return Result<std::unique_ptr<TextSystem>, std::string>::makeResult(
      std::move(system));
}

} // namespace nuri
