#include "nuri/text/text_system.h"
#include "nuri/core/log.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/pch.h"
#include "nuri/text/font_manager.h"
#include "nuri/text/text_layouter.h"
#include "nuri/text/text_renderer.h"
#include "nuri/text/text_shaper.h"
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

TextSystem::TextSystem(GPUDevice &gpu, std::pmr::memory_resource &memory)
    : gpu_(gpu), memory_(memory) {}

TextSystem::~TextSystem() = default;

Result<bool, std::string> TextSystem::initialize(const CreateDesc &desc) {
  fonts_ = std::make_unique<FontManager>(FontManager::CreateDesc{
      .gpu = gpu_,
      .memory = memory_,
  });
  shaper_ = std::make_unique<TextShaper>(TextShaper::CreateDesc{
      .fonts = *fonts_,
  });
  layouter_ = std::make_unique<TextLayouter>(TextLayouter::CreateDesc{
      .fonts = *fonts_,
      .shaper = *shaper_,
      .memory = memory_,
  });
  renderer_.reset(new TextRenderer(TextRenderer::CreateDesc{
      .gpu = gpu_,
      .fonts = *fonts_,
      .layouter = *layouter_,
      .memory = memory_,
      .shaderPaths = desc.shaderPaths,
  }));
  if (!desc.defaultFontPath.empty()) {
    const std::string defaultFontPathString = desc.defaultFontPath.string();
    auto loadResult = fonts_->loadFont(FontLoadDesc{
        .path = defaultFontPathString,
        .debugName = "default_ui",
        .memory = &memory_,
    });
    if (loadResult.hasError()) {
      if (desc.requireDefaultFont) {
        return makeError<bool>("TextSystem: failed to load default font '",
                               defaultFontPathString, "' (", loadResult.error(),
                               ")");
      }
      NURI_LOG_WARNING(
          "TextSystem: failed to load optional default font '%s': %s",
          defaultFontPathString.c_str(), loadResult.error().c_str());
    } else {
      defaultFont_ = loadResult.value();
    }
  }
  return Result<bool, std::string>::makeResult(true);
}

void TextSystem::beginFrame(uint64_t frameIndex) {
  renderer_->beginFrame(frameIndex);
}

Result<TextBounds, std::string>
TextSystem::enqueue2D(const Text2DDesc &desc,
                      std::pmr::memory_resource &scratch) {
  return renderer_->enqueue2D(desc, scratch);
}

Result<TextBounds, std::string>
TextSystem::enqueue3D(const Text3DDesc &desc,
                      std::pmr::memory_resource &scratch) {
  return renderer_->enqueue3D(desc, scratch);
}

FontHandle TextSystem::defaultFont() const { return defaultFont_; }

Result<FontHandle, std::string>
TextSystem::loadAndSetDefaultFont(std::string_view fontPath,
                                  std::string_view debugName) {
  if (fontPath.empty()) {
    return makeError<FontHandle>(
        "TextSystem::loadAndSetDefaultFont: font path is empty");
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
  if (::nuri::isValid(oldDefault) && oldDefault.value != newDefault.value) {
    fonts_->unloadFont(oldDefault);
  }
  return Result<FontHandle, std::string>::makeResult(newDefault);
}

float TextSystem::defaultFontSizePx() const { return defaultFontSizePx_; }

void TextSystem::setDefaultFontSizePx(float sizePx) {
  defaultFontSizePx_ = std::clamp(sizePx, MIN_FONT_SIZE_PX, MAX_FONT_SIZE_PX);
}

Result<std::unique_ptr<TextSystem>, std::string>
TextSystem::create(const CreateDesc &desc) {
  auto system =
      std::unique_ptr<TextSystem>(new TextSystem(desc.gpu, desc.memory));
  auto initResult = system->initialize(desc);
  if (initResult.hasError()) {
    return Result<std::unique_ptr<TextSystem>, std::string>::makeError(
        initResult.error());
  }
  return Result<std::unique_ptr<TextSystem>, std::string>::makeResult(
      std::move(system));
}

namespace {
Result<bool, std::string> beginTextFrame(void *state, FrameBuildContext &ctx) {
  static_cast<TextSystem *>(state)->beginFrame(ctx.frame.frameIndex);
  return Result<bool, std::string>::makeResult(true);
}
} // namespace

void registerText3DStage(RenderPipeline &pipeline, TextSystem &text) {
  pipeline.addBorrowedComponent(
      &text,
      PipelineComponentDesc{
          .publish =
              [](void *state, FrameBuildContext &ctx) {
                ctx.frame.transparentContributors.publish(
                    TransparentContributionCollector{
                        .user =
                            static_cast<TextSystem *>(state)->renderer_.get(),
                        .collect =
                            [](void *user, RenderFrameContext &frame,
                               TransparentStageContribution &out) {
                              return static_cast<TextRenderer *>(user)
                                  ->buildTransparentStageContribution(frame,
                                                                      out);
                            },
                    });
                return Result<bool, std::string>::makeResult(true);
              },
          .prepare = beginTextFrame,
      });
  pipeline.addStage(PipelineStageDesc{
      .componentName = "Text3DFeature",
      .name = "Text3DPass",
      .state = &text,
      .enabled =
          [](const void *, const FrameBuildContext &ctx) {
            return !ctx.frame.sharedResources.transparentStageEnabled;
          },
      .build =
          [](void *state, FrameBuildContext &ctx) {
            return static_cast<TextSystem *>(state)
                ->renderer_->append3DGraphPass(
                    ctx.frame, ctx.graph,
                    ctx.frame.sharedResources.sceneDepthGraphTexture,
                    ctx.graph.passCount() > 0u);
          },
  });
}

void registerText2DStage(RenderPipeline &pipeline, TextSystem &text) {
  pipeline.addBorrowedComponent(&text, PipelineComponentDesc{
                                           .prepare = beginTextFrame,
                                       });
  pipeline.addStage(PipelineStageDesc{
      .componentName = "Text2DFeature",
      .name = "Text2DPass",
      .state = &text,
      .build =
          [](void *state, FrameBuildContext &ctx) {
            return static_cast<TextSystem *>(state)
                ->renderer_->append2DGraphPass(ctx.frame, ctx.graph,
                                               ctx.graph.passCount() > 0u);
          },
  });
}

} // namespace nuri
