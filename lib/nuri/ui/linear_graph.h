#pragma once

#include "nuri/defines.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

namespace nuri {

struct LinearGraphStyle {
  float heightPixels = 64.0f;
  std::uint32_t lineColorRgba = 0u;
  bool fillUnderLine = true;
};

class NURI_API LinearGraph {
public:
  virtual ~LinearGraph() = default;

  virtual void pushSample(float value) = 0;
  virtual void clear() = 0;
  virtual void draw(std::string_view plotId, std::string_view seriesLabel,
                    const LinearGraphStyle &style) = 0;
};

NURI_API std::unique_ptr<LinearGraph>
createImPlotLinearGraph(std::size_t capacity);

} // namespace nuri
