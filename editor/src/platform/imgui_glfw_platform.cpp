#include "nuri/platform/imgui_glfw_platform.h"

#include <GLFW/glfw3.h>
#include <imgui_impl_glfw.h>
#include <mutex>
#include <unordered_map>

namespace nuri {

namespace {

std::mutex g_platformMapMutex;
std::unordered_map<GLFWwindow *, ImGuiGlfwPlatform *> g_platformMap;

ImGuiGlfwPlatform *findPlatform(GLFWwindow *window) {
  std::scoped_lock lock(g_platformMapMutex);
  const auto it = g_platformMap.find(window);
  return it != g_platformMap.end() ? it->second : nullptr;
}

void registerPlatform(GLFWwindow *window, ImGuiGlfwPlatform *platform) {
  std::scoped_lock lock(g_platformMapMutex);
  if (platform == nullptr) {
    g_platformMap.erase(window);
    return;
  }
  g_platformMap[window] = platform;
}

} // namespace

std::unique_ptr<ImGuiGlfwPlatform> ImGuiGlfwPlatform::create(Window &window) {
  return std::unique_ptr<ImGuiGlfwPlatform>(new ImGuiGlfwPlatform(window));
}

ImGuiGlfwPlatform::ImGuiGlfwPlatform(Window &window) : window_(window) {
  glfwWindow_ = static_cast<GLFWwindow *>(window_.nativeHandle());
  ImGui_ImplGlfw_InitForVulkan(glfwWindow_, /*install_callbacks=*/false);
  registerPlatform(glfwWindow_, this);
  prevKeyCallback_ =
      glfwSetKeyCallback(glfwWindow_, &ImGuiGlfwPlatform::onGlfwKey);
  prevCharCallback_ =
      glfwSetCharCallback(glfwWindow_, &ImGuiGlfwPlatform::onGlfwChar);
  prevMouseButtonCallback_ = glfwSetMouseButtonCallback(
      glfwWindow_, &ImGuiGlfwPlatform::onGlfwMouseButton);
  prevCursorPosCallback_ = glfwSetCursorPosCallback(
      glfwWindow_, &ImGuiGlfwPlatform::onGlfwCursorPos);
  prevScrollCallback_ =
      glfwSetScrollCallback(glfwWindow_, &ImGuiGlfwPlatform::onGlfwScroll);
  prevFocusCallback_ =
      glfwSetWindowFocusCallback(glfwWindow_, &ImGuiGlfwPlatform::onGlfwFocus);
  prevCursorEnterCallback_ = glfwSetCursorEnterCallback(
      glfwWindow_, &ImGuiGlfwPlatform::onGlfwCursorEnter);
}

ImGuiGlfwPlatform::~ImGuiGlfwPlatform() {
  if (glfwWindow_ != nullptr) {
    glfwSetKeyCallback(glfwWindow_, prevKeyCallback_);
    glfwSetCharCallback(glfwWindow_, prevCharCallback_);
    glfwSetMouseButtonCallback(glfwWindow_, prevMouseButtonCallback_);
    glfwSetCursorPosCallback(glfwWindow_, prevCursorPosCallback_);
    glfwSetScrollCallback(glfwWindow_, prevScrollCallback_);
    glfwSetWindowFocusCallback(glfwWindow_, prevFocusCallback_);
    glfwSetCursorEnterCallback(glfwWindow_, prevCursorEnterCallback_);
    registerPlatform(glfwWindow_, nullptr);
  }
  ImGui_ImplGlfw_Shutdown();
  glfwWindow_ = nullptr;
}

void ImGuiGlfwPlatform::newFrame() {
  if (glfwWindow_ && ImGui::GetCurrentContext() != nullptr) {
    ImGuiIO &io = ImGui::GetIO();
    // This polling is deliberate: callbacks remain the primary source of
    // mouse input, but sampling the GLFW state here keeps `mouseButtons_`
    // synchronized when events are missed before ImGui_ImplGlfw_NewFrame().
    for (int button = 0; button < static_cast<int>(mouseButtons_.size());
         ++button) {
      const bool down = glfwGetMouseButton(glfwWindow_, button) == GLFW_PRESS;
      if (down != mouseButtons_[static_cast<size_t>(button)]) {
        io.AddMouseButtonEvent(button, down);
        mouseButtons_[static_cast<size_t>(button)] = down;
      }
    }
  }

  ImGui_ImplGlfw_NewFrame();
}

void ImGuiGlfwPlatform::onGlfwKey(GLFWwindow *window, int key, int scancode,
                                  int action, int mods) {
  if (ImGuiGlfwPlatform *platform = findPlatform(window); platform != nullptr) {
    platform->invokePrevKeyCallback(key, scancode, action, mods);
    platform->handleGlfwKey(key, scancode, action, mods);
  }
}

void ImGuiGlfwPlatform::onGlfwChar(GLFWwindow *window, unsigned int codepoint) {
  if (ImGuiGlfwPlatform *platform = findPlatform(window); platform != nullptr) {
    platform->invokePrevCharCallback(codepoint);
    platform->handleGlfwChar(codepoint);
  }
}

void ImGuiGlfwPlatform::onGlfwMouseButton(GLFWwindow *window, int button,
                                          int action, int mods) {
  if (ImGuiGlfwPlatform *platform = findPlatform(window); platform != nullptr) {
    platform->invokePrevMouseButtonCallback(button, action, mods);
    platform->handleGlfwMouseButton(button, action, mods);
  }
}

void ImGuiGlfwPlatform::onGlfwCursorPos(GLFWwindow *window, double x,
                                        double y) {
  if (ImGuiGlfwPlatform *platform = findPlatform(window); platform != nullptr) {
    platform->invokePrevCursorPosCallback(x, y);
    platform->handleGlfwMouseMove(x, y);
  }
}

void ImGuiGlfwPlatform::onGlfwScroll(GLFWwindow *window, double xOffset,
                                     double yOffset) {
  if (ImGuiGlfwPlatform *platform = findPlatform(window); platform != nullptr) {
    platform->invokePrevScrollCallback(xOffset, yOffset);
    platform->handleGlfwMouseScroll(xOffset, yOffset);
  }
}

void ImGuiGlfwPlatform::onGlfwFocus(GLFWwindow *window, int focused) {
  if (ImGuiGlfwPlatform *platform = findPlatform(window); platform != nullptr) {
    platform->invokePrevFocusCallback(focused);
    platform->handleGlfwFocus(focused);
  }
}

void ImGuiGlfwPlatform::onGlfwCursorEnter(GLFWwindow *window, int entered) {
  if (ImGuiGlfwPlatform *platform = findPlatform(window); platform != nullptr) {
    platform->invokePrevCursorEnterCallback(entered);
    platform->handleGlfwCursorEnter(entered);
  }
}

void ImGuiGlfwPlatform::handleGlfwKey(int key, int scancode, int action,
                                      int mods) {
  ImGui_ImplGlfw_KeyCallback(glfwWindow_, key, scancode, action, mods);
}

void ImGuiGlfwPlatform::handleGlfwChar(unsigned int codepoint) {
  ImGui_ImplGlfw_CharCallback(glfwWindow_, codepoint);
}

void ImGuiGlfwPlatform::handleGlfwMouseButton(int button, int action,
                                              int mods) {
  if (button >= 0 && button < static_cast<int>(mouseButtons_.size())) {
    mouseButtons_[static_cast<size_t>(button)] = action == GLFW_PRESS;
  }
  ImGui_ImplGlfw_MouseButtonCallback(glfwWindow_, button, action, mods);
}

void ImGuiGlfwPlatform::handleGlfwMouseMove(double x, double y) {
  ImGui_ImplGlfw_CursorPosCallback(glfwWindow_, x, y);
}

void ImGuiGlfwPlatform::handleGlfwMouseScroll(double xOffset, double yOffset) {
  ImGui_ImplGlfw_ScrollCallback(glfwWindow_, xOffset, yOffset);
}

void ImGuiGlfwPlatform::handleGlfwFocus(int focused) {
  ImGui_ImplGlfw_WindowFocusCallback(glfwWindow_, focused);
}

void ImGuiGlfwPlatform::handleGlfwCursorEnter(int entered) {
  ImGui_ImplGlfw_CursorEnterCallback(glfwWindow_, entered);
}

void ImGuiGlfwPlatform::invokePrevKeyCallback(int key, int scancode, int action,
                                              int mods) const {
  if (prevKeyCallback_ != nullptr) {
    prevKeyCallback_(glfwWindow_, key, scancode, action, mods);
  }
}

void ImGuiGlfwPlatform::invokePrevCharCallback(unsigned int codepoint) const {
  if (prevCharCallback_ != nullptr) {
    prevCharCallback_(glfwWindow_, codepoint);
  }
}

void ImGuiGlfwPlatform::invokePrevMouseButtonCallback(int button, int action,
                                                      int mods) const {
  if (prevMouseButtonCallback_ != nullptr) {
    prevMouseButtonCallback_(glfwWindow_, button, action, mods);
  }
}

void ImGuiGlfwPlatform::invokePrevCursorPosCallback(double x, double y) const {
  if (prevCursorPosCallback_ != nullptr) {
    prevCursorPosCallback_(glfwWindow_, x, y);
  }
}

void ImGuiGlfwPlatform::invokePrevScrollCallback(double xOffset,
                                                 double yOffset) const {
  if (prevScrollCallback_ != nullptr) {
    prevScrollCallback_(glfwWindow_, xOffset, yOffset);
  }
}

void ImGuiGlfwPlatform::invokePrevFocusCallback(int focused) const {
  if (prevFocusCallback_ != nullptr) {
    prevFocusCallback_(glfwWindow_, focused);
  }
}

void ImGuiGlfwPlatform::invokePrevCursorEnterCallback(int entered) const {
  if (prevCursorEnterCallback_ != nullptr) {
    prevCursorEnterCallback_(glfwWindow_, entered);
  }
}

} // namespace nuri
