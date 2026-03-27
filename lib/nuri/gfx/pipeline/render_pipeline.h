#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/pipeline/frame_build_context.h"
#include "nuri/gfx/pipeline/frame_data_provider.h"
#include "nuri/gfx/pipeline/render_feature.h"

#include <memory>
#include <memory_resource>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace nuri {

struct NURI_API RenderPipelinePassInfo {
  size_t index = 0u;
  std::string_view featureName{};
  std::string_view passName{};
  bool enabled = true;
};

class NURI_API RenderPipeline {
public:
  explicit RenderPipeline(
      std::pmr::memory_resource *memory = std::pmr::get_default_resource());
  ~RenderPipeline() = default;

  RenderPipeline(const RenderPipeline &) = delete;
  RenderPipeline &operator=(const RenderPipeline &) = delete;
  RenderPipeline(RenderPipeline &&) = default;
  RenderPipeline &operator=(RenderPipeline &&) = default;

  FrameDataProvider *addProvider(std::unique_ptr<FrameDataProvider> provider);
  RenderFeature *addFeature(std::unique_ptr<RenderFeature> feature);

  [[nodiscard]] Result<bool, std::string>
  buildRenderGraph(RenderFrameContext &frame, ResourceManager &resources,
                   RenderGraphBuilder &graph);

  [[nodiscard]] bool empty() const noexcept {
    return providers_.empty() && features_.empty() && passes_.empty();
  }
  [[nodiscard]] size_t passCount() const noexcept { return passes_.size(); }
  [[nodiscard]] std::optional<RenderPipelinePassInfo>
  passInfo(size_t index) const noexcept;
  [[nodiscard]] std::optional<bool> isPassEnabled(size_t index) const noexcept;
  bool setPassEnabled(size_t index, bool enabled) noexcept;

private:
  struct RegisteredPass {
    RenderFeature *feature = nullptr;
    RenderFeaturePass *pass = nullptr;
    bool enabled = true;
  };

  void registerFeaturePasses(RenderFeature &feature);

  std::pmr::vector<std::unique_ptr<FrameDataProvider>> providers_;
  std::pmr::vector<std::unique_ptr<RenderFeature>> features_;
  std::pmr::vector<RegisteredPass> passes_;
};

} // namespace nuri
