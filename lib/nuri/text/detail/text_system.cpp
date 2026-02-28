#include "nuri/pch.h"

#include "nuri/core/log.h"
#include "nuri/text/text_system.h"

namespace nuri {
namespace {

constexpr float MIN_FONT_SIZE_PX = 8.0f;
constexpr float MAX_FONT_SIZE_PX = 256.0f;

template <typename T, typename... Args>
[[nodiscard]] Result<T, std::string> makeError(Args &&...args) {
  std::ostringstream oss;
  (oss << ... << std::forward<Args>(args));
  return Result<T, std::string>::makeError(oss.str());
}

} // namespace

TextSystem::TextSystem(CreateDesc desc)
    : gpu_(desc.gpu), memory_(desc.memory),
      defaultFontPath_(std::move(desc.defaultFontPath)),
      requireDefaultFont_(desc.requireDefaultFont),
      shaderPaths_(std::move(desc.shaderPaths)) {}

TextSystem::~TextSystem() = default;

Result<bool, std::string> TextSystem::initialize() {
  auto fonts = FontManager::create(FontManager::CreateDesc{
      .gpu = gpu_,
      .memory = memory_,
  });
  if (!fonts) {
    return Result<bool, std::string>::makeError(
        "TextSystem: failed to create FontManager");
  }

  std::unique_ptr<TextShaper> shaper;
  try {
    shaper = std::make_unique<TextShaper>(TextShaper::CreateDesc{
        .fonts = *fonts,
        .memory = memory_,
    });
  } catch (const std::exception &e) {
    return makeError<bool>("TextSystem: failed to create TextShaper (",
                          e.what(), ")");
  } catch (...) {
    return Result<bool, std::string>::makeError(
        "TextSystem: failed to create TextShaper (unknown exception)");
  }

  auto layouter = std::make_unique<TextLayouter>(TextLayouter::CreateDesc{
      .fonts = *fonts,
      .shaper = *shaper,
      .memory = memory_,
  });

  std::unique_ptr<TextRenderer> rendererPtr;
  try {
    rendererPtr = std::make_unique<TextRenderer>(TextRenderer::CreateDesc{
        .gpu = gpu_,
        .fonts = *fonts,
        .layouter = *layouter,
        .memory = memory_,
        .shaderPaths = shaderPaths_,
    });
  } catch (const std::exception &e) {
    return makeError<bool>("TextSystem: failed to create TextRenderer (",
                           e.what(), ")");
  } catch (...) {
    return Result<bool, std::string>::makeError(
        "TextSystem: failed to create TextRenderer (unknown exception)");
  }

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
  renderer_ = std::move(rendererPtr);
  return Result<bool, std::string>::makeResult(true);
}

FontManager &TextSystem::fonts() {
  NURI_ASSERT(fonts_ != nullptr, "TextSystem::fonts is not initialized");
  return *fonts_;
}

TextRenderer &TextSystem::renderer() {
  NURI_ASSERT(renderer_ != nullptr,
              "TextSystem::renderer is not initialized");
  return *renderer_;
}

FontHandle TextSystem::defaultFont() const { return defaultFont_; }

Result<FontHandle, std::string>
TextSystem::loadAndSetDefaultFont(std::string_view fontPath,
                                  std::string_view debugName) {
  if (fontPath.empty()) {
    return makeError<FontHandle>(
        "TextSystem::loadAndSetDefaultFont: font path is empty");
  }
  if (fonts_ == nullptr) {
    return makeError<FontHandle>(
        "TextSystem::loadAndSetDefaultFont: FontManager is not initialized");
  }

  const std::filesystem::path path{std::string(fontPath)};
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

float TextSystem::defaultFontSizePx() const { return defaultFontSizePx_; }

void TextSystem::setDefaultFontSizePx(float sizePx) {
  defaultFontSizePx_ = std::clamp(sizePx, MIN_FONT_SIZE_PX, MAX_FONT_SIZE_PX);
}

Result<std::unique_ptr<TextSystem>, std::string>
TextSystem::create(const CreateDesc &desc) {
  auto system = std::unique_ptr<TextSystem>(new TextSystem(desc));
  auto initResult = system->initialize();
  if (initResult.hasError()) {
    return Result<std::unique_ptr<TextSystem>, std::string>::makeError(
        initResult.error());
  }
  return Result<std::unique_ptr<TextSystem>, std::string>::makeResult(
      std::move(system));
}

} // namespace nuri
