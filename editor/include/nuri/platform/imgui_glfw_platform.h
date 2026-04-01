#pragma once

#include "nuri/core/event_manager.h"
#include "nuri/core/input_events.h"
#include "nuri/core/window.h"
#include "nuri/defines.h"

#include <GLFW/glfw3.h>
#include <array>
#include <memory>

namespace nuri {

class ImGuiGlfwPlatform final {
public:
  static std::unique_ptr<ImGuiGlfwPlatform> create(Window &window,
                                                   EventManager &events);
  ~ImGuiGlfwPlatform();

  ImGuiGlfwPlatform(const ImGuiGlfwPlatform &) = delete;
  ImGuiGlfwPlatform &operator=(const ImGuiGlfwPlatform &) = delete;
  ImGuiGlfwPlatform(ImGuiGlfwPlatform &&) = delete;
  ImGuiGlfwPlatform &operator=(ImGuiGlfwPlatform &&) = delete;

  void newFrame();

private:
  ImGuiGlfwPlatform(Window &window, EventManager &events);
  static void onGlfwKey(GLFWwindow *window, int key, int scancode, int action,
                        int mods);
  static void onGlfwChar(GLFWwindow *window, unsigned int codepoint);
  static void onGlfwMouseButton(GLFWwindow *window, int button, int action,
                                int mods);
  static void onGlfwCursorPos(GLFWwindow *window, double x, double y);
  static void onGlfwScroll(GLFWwindow *window, double xOffset, double yOffset);
  static void onGlfwFocus(GLFWwindow *window, int focused);
  static void onGlfwCursorEnter(GLFWwindow *window, int entered);
  void handleGlfwKey(int key, int scancode, int action, int mods);
  void handleGlfwChar(unsigned int codepoint);
  void handleGlfwMouseButton(int button, int action, int mods);
  void handleGlfwMouseMove(double x, double y);
  void handleGlfwMouseScroll(double xOffset, double yOffset);
  void handleGlfwFocus(int focused);
  void handleGlfwCursorEnter(int entered);
  void invokePrevKeyCallback(int key, int scancode, int action, int mods) const;
  void invokePrevCharCallback(unsigned int codepoint) const;
  void invokePrevMouseButtonCallback(int button, int action, int mods) const;
  void invokePrevCursorPosCallback(double x, double y) const;
  void invokePrevScrollCallback(double xOffset, double yOffset) const;
  void invokePrevFocusCallback(int focused) const;
  void invokePrevCursorEnterCallback(int entered) const;

  Window &window_;
  GLFWwindow *glfwWindow_ = nullptr;
  std::array<bool, 5> mouseButtons_{};
  GLFWkeyfun prevKeyCallback_ = nullptr;
  GLFWcharfun prevCharCallback_ = nullptr;
  GLFWmousebuttonfun prevMouseButtonCallback_ = nullptr;
  GLFWcursorposfun prevCursorPosCallback_ = nullptr;
  GLFWscrollfun prevScrollCallback_ = nullptr;
  GLFWwindowfocusfun prevFocusCallback_ = nullptr;
  GLFWcursorenterfun prevCursorEnterCallback_ = nullptr;
};

} // namespace nuri
