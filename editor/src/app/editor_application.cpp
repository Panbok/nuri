#include "nuri/editor_pch.h"

#include "nuri/app/editor_application.h"

#include "nuri/core/log.h"
#include "nuri/core/profiling.h"
#include "nuri/core/runtime_config.h"

namespace nuri {
Result<void, std::string> registerBuiltInScenes(EditorSceneCatalog &catalog,
                                                const RuntimeConfig &config);

ApplicationConfig
EditorApplication::makeEditorApplicationConfig(const RuntimeConfig &config) {
  ApplicationConfig appConfig = makeApplicationConfig(config);
  appConfig.renderComposition = RenderCompositionMode::PipelineOnly;
  return appConfig;
}

EditorApplication::EditorApplication(RuntimeConfig config)
    : Application(makeEditorApplicationConfig(config)), config_(config),
      runtime_(*this, config), scenes_() {}

void EditorApplication::onInit() {
  auto registerResult = registerBuiltInScenes(scenes_, config_);
  NURI_ASSERT(!registerResult.hasError(),
              "Failed to register built-in editor scenes: %s",
              registerResult.error().c_str());
  runtime_.initialize();
  const std::string_view initialSceneId = scenes_.initialSceneId();
  NURI_ASSERT(!initialSceneId.empty(),
              "Failed to determine initial editor scene");
  NURI_ASSERT(scenes_.requestActivate(initialSceneId),
              "Failed to request initial scene activation");
  NURI_LOG_INFO("Editor application initialized");
}

void EditorApplication::onUpdate(double deltaTime) {
  scenes_.updateActive(runtime_, deltaTime);
  runtime_.update(deltaTime);
}

void EditorApplication::onDraw() {
  auto activationResult = scenes_.applyPendingActivation(runtime_);
  NURI_ASSERT(!activationResult.hasError(), "Scene activation failed: %s",
              activationResult.error().c_str());
  runtime_.syncSceneSelectionUi(scenes_);
  runtime_.draw();
  if (const auto requestedScene = runtime_.takeSceneSelectionRequest();
      requestedScene.has_value()) {
    (void)scenes_.requestActivate(*requestedScene);
  }
}

void EditorApplication::onResize(std::int32_t width, std::int32_t height) {
  runtime_.resize(width, height);
}

bool EditorApplication::onInput(const InputEvent &event) {
  return runtime_.onInput(event) ? true : Application::onInput(event);
}

void EditorApplication::onShutdown() {
  scenes_.shutdown(runtime_);
  runtime_.shutdown();
  NURI_LOG_INFO("Editor application shutdown");
}

} // namespace nuri
