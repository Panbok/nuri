#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/pipeline/frame_build_context.h"
#include "nuri/gfx/pipeline/render_feature_pass.h"

#include <span>
#include <string>
#include <string_view>

namespace nuri {

class NURI_API RenderFeature {
public:
  virtual ~RenderFeature() = default;

  virtual std::string_view name() const noexcept = 0;
  virtual Result<bool, std::string> publishFrameData(FrameBuildContext &ctx) {
    (void)ctx;
    return Result<bool, std::string>::makeResult(true);
  }
  virtual Result<bool, std::string> prepare(FrameBuildContext &ctx) {
    (void)ctx;
    return Result<bool, std::string>::makeResult(true);
  }
  [[nodiscard]] virtual std::span<RenderFeaturePass *const>
  passes() noexcept = 0;
};

} // namespace nuri
