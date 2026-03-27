#pragma once

#include <string>
#include <string_view>

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/pipeline/frame_build_context.h"

namespace nuri {

class NURI_API FrameDataProvider {
public:
  virtual ~FrameDataProvider() = default;

  virtual std::string_view name() const noexcept = 0;
  // On success, `true` means the provider published or updated frame data;
  // `false` means there was nothing to do. The error string is populated only
  // on failure.
  [[nodiscard]] virtual Result<bool, std::string>
  prepare(FrameBuildContext &ctx) = 0;
};

} // namespace nuri
