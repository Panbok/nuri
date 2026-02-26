#pragma once

#include "nuri/core/event_manager.h"
#include "nuri/defines.h"
#include "nuri/pch.h"

namespace nuri {

enum class WindowMode : uint8_t {
  Windowed,
  Fullscreen,
  BorderlessFullscreen,
};

enum class CursorMode : uint8_t {
  Normal,
  Hidden,
  Disabled,
};

class NURI_API Window {
public:
  static std::unique_ptr<Window> create(std::string_view title, int32_t width,
                                        int32_t height,
                                        WindowMode mode = WindowMode::Windowed);
  virtual ~Window() = default;

  Window(const Window &) = delete;
  Window &operator=(const Window &) = delete;
  Window(Window &&) = delete;
  Window &operator=(Window &&) = delete;

  virtual void pollEvents() = 0;
  virtual bool shouldClose() const = 0;
  virtual void getWindowSize(int32_t &outWidth, int32_t &outHeight) const = 0;
  virtual void getFramebufferSize(int32_t &outWidth,
                                  int32_t &outHeight) const = 0;
  virtual double getTime() const = 0;
  virtual void *nativeHandle() const = 0;
  virtual void requestClose() = 0;
  virtual void setCursorMode(CursorMode mode) = 0;
  [[nodiscard]] virtual CursorMode getCursorMode() const = 0;

  /** Binds an event manager to receive window events.
   * @param events Non-owning pointer to the event manager; must outlive this
   * window.
   * Pass nullptr to unbind.
   */
  virtual void bindEventManager(EventManager *events) = 0;

protected:
  Window() = default;
};

} // namespace nuri
