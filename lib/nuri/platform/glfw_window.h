#pragma once

#include "nuri/core/window.h"

namespace nuri {

class GlfwWindow final : public Window {
public:
  static std::unique_ptr<GlfwWindow> create(std::string_view title,
                                            int32_t width, int32_t height,
                                            WindowMode mode = WindowMode::Windowed);
  ~GlfwWindow() override;

  GlfwWindow(const GlfwWindow &) = delete;
  GlfwWindow &operator=(const GlfwWindow &) = delete;
  GlfwWindow(GlfwWindow &&) = delete;
  GlfwWindow &operator=(GlfwWindow &&) = delete;

  void pollEvents() override;
  bool shouldClose() const override;
  void getWindowSize(int32_t &outWidth, int32_t &outHeight) const override;
  void getFramebufferSize(int32_t &outWidth,
                          int32_t &outHeight) const override;
  double getTime() const override;
  void *nativeHandle() const override;
  void requestClose() override;
  void setCursorMode(CursorMode mode) override;
  CursorMode getCursorMode() const override;
  void bindEventManager(EventManager *events) override;

private:
  GlfwWindow();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace nuri
