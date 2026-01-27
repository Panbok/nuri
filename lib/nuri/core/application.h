#pragma once

#include "nuri/defines.h"
#include "nuri/gfx/renderer.h"
#include "nuri/pch.h"

namespace nuri {
class NURI_API Application {
public:
  Application(const std::string &title, std::int32_t width,
              std::int32_t height);
  virtual ~Application();

  Application(const Application &) = delete;
  Application &operator=(const Application &) = delete;
  Application(Application &&) = delete;
  Application &operator=(Application &&) = delete;

  void run();
  double getTime() const;

  virtual void onInit() = 0;
  virtual void onUpdate(float deltaTime) = 0;
  virtual void onDraw() = 0;
  virtual void onShutdown() = 0;

  inline std::unique_ptr<lvk::IContext> &getContext() { return context_; }
  inline float getAspectRatio() const {
    return width_ / static_cast<float>(height_);
  }
  inline std::int32_t getWidth() const { return width_; }
  inline std::int32_t getHeight() const { return height_; }
  inline std::unique_ptr<nuri::Renderer> &getRenderer() { return renderer_; }

private:
  std::string title_;
  std::int32_t width_;
  std::int32_t height_;
  lvk::LVKwindow *window_;
  std::unique_ptr<lvk::IContext> context_;
  std::unique_ptr<nuri::Renderer> renderer_;
};

} // namespace nuri
