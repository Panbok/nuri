#pragma once
#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/frame/render_frame_context.h"
#include "nuri/gfx/render_graph/render_graph.h"
#include <memory>
#include <memory_resource>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
namespace nuri {

class Camera;
class RenderScene;
class ResourceManager;
class SceneDrawDatabase;
struct RenderSettings;

struct FrameBuildContext {
  RenderFrameContext &frame;
  RenderGraphBuilder &graph;
  const ResourceManager &resources;
  FrameSharedResources &shared;
};

struct RenderScenePreparationContext {
  RenderScene &scene;
  ResourceManager &resources;
  SceneDrawDatabase *sceneDrawDatabase = nullptr;
  uint32_t maxOperations = 128u;
  const RenderSettings *settings = nullptr;
  const Camera *camera = nullptr;
  float aspectRatio = 1.0f;
  uint32_t renderWidth = 1u;
  uint32_t renderHeight = 1u;
};

struct NURI_API RenderPipelinePassInfo {
  size_t index = 0u;
  std::string_view featureName{};
  std::string_view passName{};
  bool enabled = true;
};

using PipelineFrameCallback =
    Result<bool, std::string> (*)(void *, FrameBuildContext &);
using PipelineSceneCallback =
    Result<bool, std::string> (*)(void *, RenderScenePreparationContext &);
using PipelineCompletionCallback =
    void (*)(void *, const RenderFrameContext &) noexcept;

struct PipelineComponentDesc {
  PipelineFrameCallback publish = nullptr;
  PipelineFrameCallback provide = nullptr;
  PipelineFrameCallback prepare = nullptr;
  PipelineSceneCallback prepareScene = nullptr;
  PipelineCompletionCallback submitted = nullptr;
  PipelineCompletionCallback abandoned = nullptr;
};

struct PipelineStageDesc {
  std::string_view componentName{};
  std::string_view name{};
  void *state = nullptr;
  bool terminal = false;
  bool (*enabled)(const void *, const FrameBuildContext &) = nullptr;
  PipelineFrameCallback prepare = nullptr;
  PipelineFrameCallback build = nullptr;
};

class NURI_API RenderPipeline {
  using Owner = std::unique_ptr<void, void (*)(void *)>;
  struct Component {
    Owner owner;
    PipelineComponentDesc desc{};
  };
  struct Stage {
    PipelineStageDesc desc{};
    bool enabled = true;
  };

public:
  explicit RenderPipeline(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~RenderPipeline() = default;
  RenderPipeline(const RenderPipeline &) = delete;
  RenderPipeline &operator=(const RenderPipeline &) = delete;
  RenderPipeline(RenderPipeline &&) = default;
  RenderPipeline &operator=(RenderPipeline &&) = default;
  template <typename T>
  T *addComponent(std::unique_ptr<T> component, PipelineComponentDesc desc) {
    T *raw = component.release();
    components_.push_back(Component{
        .owner =
            Owner(raw, [](void *value) { delete static_cast<T *>(value); }),
        .desc = desc,
    });
    return raw;
  }
  template <typename T> T *addProvider(std::unique_ptr<T> provider) {
    PipelineComponentDesc desc{
        .provide =
            [](void *state, FrameBuildContext &ctx) {
              return static_cast<T *>(state)->prepare(ctx);
            },
    };
    if constexpr (requires(T &value, const RenderFrameContext &frame) {
                    value.onFrameSubmitted(frame);
                  }) {
      desc.submitted = [](void *state,
                          const RenderFrameContext &frame) noexcept {
        static_cast<T *>(state)->onFrameSubmitted(frame);
      };
    }
    if constexpr (requires(T &value, const RenderFrameContext &frame) {
                    value.onFrameAbandoned(frame);
                  }) {
      desc.abandoned = [](void *state,
                          const RenderFrameContext &frame) noexcept {
        static_cast<T *>(state)->onFrameAbandoned(frame);
      };
    }
    if constexpr (requires(T &value, RenderScenePreparationContext &ctx) {
                    value.prepareScene(ctx);
                  }) {
      desc.prepareScene = [](void *state, RenderScenePreparationContext &ctx) {
        return static_cast<T *>(state)->prepareScene(ctx);
      };
    }
    return addComponent(std::move(provider), desc);
  }
  template <typename T>
  T *addStage(std::unique_ptr<T> stage, std::string_view componentName,
              std::string_view name, bool terminal = false,
              PipelineComponentDesc component = {}) {
    T *raw = addComponent(std::move(stage), component);
    addStage(PipelineStageDesc{
        .componentName = componentName,
        .name = name,
        .state = raw,
        .terminal = terminal,
        .enabled =
            [](const void *state, const FrameBuildContext &ctx) {
              return static_cast<const T *>(state)->isEnabled(ctx);
            },
        .prepare =
            [](void *state, FrameBuildContext &ctx) {
              return static_cast<T *>(state)->prepare(ctx);
            },
        .build =
            [](void *state, FrameBuildContext &ctx) {
              return static_cast<T *>(state)->build(ctx);
            },
    });
    return raw;
  }
  void addBorrowedComponent(void *state, PipelineComponentDesc desc) {
    components_.push_back(Component{
        .owner = Owner(state, [](void *) {}),
        .desc = desc,
    });
  }
  void addStage(PipelineStageDesc stage);
  [[nodiscard]] Result<bool, std::string>
  buildRenderGraph(RenderFrameContext &frame, ResourceManager &resources,
                   RenderGraphBuilder &graph);
  [[nodiscard]] Result<bool, std::string>
  prepareSceneStep(RenderScene &scene, ResourceManager &resources,
                   uint32_t maxOperations = 128u,
                   const RenderSettings *settings = nullptr,
                   const Camera *camera = nullptr, float aspectRatio = 1.0f,
                   uint32_t renderWidth = 1u, uint32_t renderHeight = 1u);
  void onFrameSubmitted(const RenderFrameContext &frame) noexcept;
  void onFrameAbandoned(const RenderFrameContext &frame) noexcept;
  [[nodiscard]] bool empty() const noexcept {
    return components_.empty() && stages_.empty();
  }
  [[nodiscard]] size_t passCount() const noexcept { return stages_.size(); }
  [[nodiscard]] std::optional<RenderPipelinePassInfo>
  passInfo(size_t index) const noexcept;
  [[nodiscard]] std::optional<bool> isPassEnabled(size_t index) const noexcept;
  bool setPassEnabled(size_t index, bool enabled) noexcept;

private:
  std::pmr::vector<Component> components_;
  std::pmr::vector<Stage> stages_;
};

} // namespace nuri
