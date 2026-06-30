#pragma once

#include "nuri/core/result.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/gpu_device.h"
#include "nuri/gfx/pipeline/render_pipeline.h"
#include "nuri/gfx/renderer.h"
#include "nuri/scene/camera.h"
#include "nuri/scene/render_scene.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <memory_resource>
#include <string>
#include <string_view>

#include <glm/glm.hpp>

namespace nuri {
class Window;
}

namespace nuri::tools::runtime {

using ToolResolvePathFn = Result<std::filesystem::path, std::string> (*)(
    std::string_view base, const std::filesystem::path &path);

struct ToolRenderGraphDesc {
  uint32_t workerCount = 1u;
  bool parallelCompile = false;
  bool parallelRecording = false;
};

struct ToolSceneDesc {
  std::string kind = "procedural";
  std::string pathBase{};
  std::filesystem::path path{};
  bool flipUVs = false;
  bool generateMeshlets = false;
  uint32_t meshletMaxVertices = 64u;
  uint32_t meshletMaxPrimitives = 124u;
  float meshletConeWeight = 0.0f;
  std::string generator = "nuri.procedural.v1";
  uint32_t seed = 1u;
  std::string contentHash = "procedural-empty-v1";
};

struct ToolCameraDesc {
  glm::vec3 position{0.0f, 1.0f, 4.0f};
  glm::vec3 direction{0.0f, 0.0f, -1.0f};
  glm::vec3 target{0.0f, 1.0f, 3.0f};
  bool hasTarget = false;
  float verticalFovDegrees = 60.0f;
  float nearPlane = 0.05f;
  float farPlane = 500.0f;
};

struct ToolFrameDesc {
  uint64_t frameIndex = 0u;
  double timeSeconds = 0.0;
  double deltaSeconds = 1.0 / 60.0;
  uint32_t width = 1280u;
  uint32_t height = 720u;
  bool cameraCutRequested = false;
};

struct ToolRuntimeDesc {
  std::string title = "nuri-tool";
  std::string backend = "default";
  std::string presentMode = "default";
  std::array<uint32_t, 2> resolution{1280u, 720u};
  ToolRenderGraphDesc renderGraph{};
  ToolSceneDesc scene{};
  ToolResolvePathFn resolvePath = nullptr;
};

class ToolRendererRuntime {
public:
  ~ToolRendererRuntime();

  ToolRendererRuntime(const ToolRendererRuntime &) = delete;
  ToolRendererRuntime &operator=(const ToolRendererRuntime &) = delete;
  ToolRendererRuntime(ToolRendererRuntime &&) = delete;
  ToolRendererRuntime &operator=(ToolRendererRuntime &&) = delete;

  [[nodiscard]] Window &window() noexcept;
  [[nodiscard]] GPUDevice &gpu() noexcept;
  [[nodiscard]] Renderer &renderer() noexcept;
  [[nodiscard]] RenderPipeline &pipeline() noexcept;
  [[nodiscard]] RenderScene &scene() noexcept;
  [[nodiscard]] RenderFrameContext &frameContext() noexcept;
  [[nodiscard]] TemporalCameraHistoryState &cameraHistory() noexcept;
  [[nodiscard]] uint32_t swapchainImageCount() const noexcept;
  [[nodiscard]] Result<bool, std::string> commitScene();

private:
  struct Impl;
  explicit ToolRendererRuntime(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;

  friend Result<std::unique_ptr<ToolRendererRuntime>, std::string>
  createToolRendererRuntime(const ToolRuntimeDesc &desc);
};

[[nodiscard]] Result<std::unique_ptr<ToolRendererRuntime>, std::string>
createToolRendererRuntime(const ToolRuntimeDesc &desc);
[[nodiscard]] Camera makeToolCamera(const ToolCameraDesc &desc);
void buildToolFrameContext(RenderFrameContext &frameContext, RenderScene &scene,
                           Renderer &renderer, RenderSettings &settings,
                           TemporalCameraHistoryState &cameraHistory,
                           const Camera &camera, const ToolFrameDesc &desc);

} // namespace nuri::tools::runtime
