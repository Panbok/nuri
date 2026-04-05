#pragma once

#include "nuri/app/editor_runtime.h"
#include "nuri/app/editor_scene_catalog.h"
#include "nuri/core/application.h"

namespace nuri {

class EditorApplication final : public Application {
public:
  explicit EditorApplication(RuntimeConfig config);
  ~EditorApplication() override = default;
  EditorApplication(const EditorApplication &) = delete;
  EditorApplication &operator=(const EditorApplication &) = delete;
  EditorApplication(EditorApplication &&) = delete;
  EditorApplication &operator=(EditorApplication &&) = delete;

  void onInit() override;
  void onUpdate(double deltaTime) override;
  void onDraw() override;
  void onResize(std::int32_t width, std::int32_t height) override;
  bool onInput(const InputEvent &event) override;
  void onShutdown() override;

private:
  static ApplicationConfig
  makeEditorApplicationConfig(const RuntimeConfig &config);

  const RuntimeConfig config_;
  EditorRuntime runtime_;
  EditorSceneCatalog scenes_;
};

} // namespace nuri
