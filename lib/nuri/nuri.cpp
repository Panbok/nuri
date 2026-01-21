#include "nuri/nuri.h"

namespace nuri {
const char *hello() {
  // Touch a few types so the headers are not entirely unused.
  glm::vec3 dummy(0.0f);
  (void)dummy;
  (void)sizeof(ImGuiIO);
  GLFWwindow *glfwWindow = nullptr;
  (void)glfwWindow;
  lvk::LVKwindow *lvkWindow = nullptr;
  (void)lvkWindow;

  return "Hello from Nuri";
}
} // namespace nuri
