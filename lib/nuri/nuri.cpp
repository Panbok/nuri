#include "nuri/nuri.h"

namespace nuri {
void init() {
  minilog::initialize(nullptr, {.threadNames = false});
  std::int32_t width = 960;
  std::int32_t height = 540;
  lvk::LVKwindow *window = lvk::initWindow("Nuri", width, height);
  std::unique_ptr<lvk::IContext> context =
      lvk::createVulkanContextWithSwapchain(window, width, height,
                                            lvk::ContextConfig{});

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    glfwGetFramebufferSize(window, &width, &height);
    if (!width || !height)
      continue;
    lvk::ICommandBuffer &commandBuffer = context->acquireCommandBuffer();
    context->submit(commandBuffer, context->getCurrentSwapchainTexture());
  }

  context.reset();
  glfwDestroyWindow(window);
  glfwTerminate();
  minilog::deinitialize();
}
} // namespace nuri
