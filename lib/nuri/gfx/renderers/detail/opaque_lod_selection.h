#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>
namespace nuri::detail {

struct OpaqueLodProjection {
  float pixelScaleY = 0.0f;
  float nearestDepth = 1.0f;
  bool orthographic = false;
};

[[nodiscard]] inline float
projectedLodErrorPixels(float worldError,
                        const OpaqueLodProjection &projection) noexcept {
  const float error = std::max(worldError, 0.0f);
  const float scale = std::max(projection.pixelScaleY, 0.0f);
  if (projection.orthographic) {
    return error * scale;
  }
  return error * scale / std::max(projection.nearestDepth, 1.0e-5f);
}

[[nodiscard]] inline uint32_t
selectOpaqueLod(std::span<const float> worldErrors, float targetPixelError,
                float hysteresisRatio, const OpaqueLodProjection &projection,
                std::optional<uint32_t> previousLod = std::nullopt) noexcept {
  if (worldErrors.empty()) {
    return 0u;
  }
  const uint32_t lastLod = static_cast<uint32_t>(worldErrors.size() - 1u);
  const float target = std::max(targetPixelError, 1.0e-3f);
  const float hysteresis = std::clamp(hysteresisRatio, 0.0f, 0.95f);
  const auto errorPixels = [&](uint32_t lod) {
    return projectedLodErrorPixels(worldErrors[lod], projection);
  };
  if (!previousLod) {
    uint32_t selected = 0u;
    for (uint32_t lod = 1u; lod <= lastLod; ++lod) {
      if (errorPixels(lod) > target) {
        break;
      }
      selected = lod;
    }
    return selected;
  }
  uint32_t selected = std::min(*previousLod, lastLod);
  const float refineThreshold = target * (1.0f + hysteresis);
  while (selected > 0u && errorPixels(selected) > refineThreshold) {
    --selected;
  }
  const float coarsenThreshold = target * (1.0f - hysteresis);
  while (selected < lastLod && errorPixels(selected + 1u) <= coarsenThreshold) {
    ++selected;
  }
  return selected;
}

} // namespace nuri::detail
