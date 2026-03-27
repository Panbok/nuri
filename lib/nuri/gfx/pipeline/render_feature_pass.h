#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/pipeline/frame_build_context.h"

#include <string>
#include <string_view>

namespace nuri {

class NURI_API RenderFeaturePass {
public:
  virtual ~RenderFeaturePass() = default;

  virtual std::string_view name() const noexcept = 0;
  virtual bool isEnabled(const FrameBuildContext &ctx) const = 0;
  virtual Result<bool, std::string> prepare(FrameBuildContext &ctx) = 0;
  virtual Result<bool, std::string> build(FrameBuildContext &ctx) = 0;
};

} // namespace nuri
