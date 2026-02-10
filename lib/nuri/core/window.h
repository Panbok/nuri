#pragma once

#include "nuri/defines.h"
#include "nuri/pch.h"

namespace nuri {

enum class WindowMode {
  Windowed,
  Fullscreen,
  BorderlessFullscreen,
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
  virtual void getFramebufferSize(int32_t &outWidth,
                                  int32_t &outHeight) const = 0;
  virtual double getTime() const = 0;
  virtual void *nativeHandle() const = 0;

protected:
  Window() = default;
};

} // namespace nuri
