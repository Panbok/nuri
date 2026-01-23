#pragma once

#include "defines.h"
#include "pch.h"

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

  virtual void onUpdate(float deltaTime) = 0;
  virtual void onRender() = 0;
  virtual void onShutdown() = 0;

private:
  std::string title_;
  std::int32_t width_;
  std::int32_t height_;
  lvk::LVKwindow *window_;
  std::unique_ptr<lvk::IContext> context_;
};

} // namespace nuri
