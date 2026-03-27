#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/pipeline/frame_build_context.h"

#include <string>
#include <string_view>

namespace nuri {

class NURI_API FrameDataProvider {
public:
  virtual ~FrameDataProvider() = default;

  virtual std::string_view name() const noexcept = 0;
  virtual Result<bool, std::string> prepare(FrameBuildContext &ctx) = 0;
};

} // namespace nuri
