#pragma once

#include "nuri/core/result.h"
#include "nuri/defines.h"
#include "nuri/gfx/ddgi/ddgi_types.h"
#include <cstdint>
#include <glm/glm.hpp>

namespace nuri {

inline constexpr glm::uvec2 kDDGIIrradianceTileExtent{10u, 10u};
inline constexpr glm::uvec2 kDDGIDistanceTileExtent{18u, 18u};
inline constexpr uint32_t kDDGIIrradianceBytesPerTexel = 8u;
inline constexpr uint32_t kDDGIDistanceBytesPerTexel = 4u;
inline constexpr uint32_t kDDGIProbeStateBytes = 32u;

enum class DDGIAtlasError : uint8_t {
  InvalidArgument = 0,
  TextureLimitExceeded,
  ArithmeticOverflow,
};

struct DDGIMemoryEstimate {
  uint64_t irradianceBytes = 0u;
  uint64_t distanceBytes = 0u;
  uint64_t probeStateBytes = 0u;
  uint64_t persistentBytes = 0u;
};

[[nodiscard]] NURI_API Result<DDGIAtlasLayout, DDGIAtlasError>
packDDGIAtlas(uint32_t probeCount, glm::uvec2 tileExtent,
              glm::uvec2 maxTextureExtent) noexcept;
[[nodiscard]] NURI_API Result<DDGIMemoryEstimate, DDGIAtlasError>
estimateDDGIMemory(uint32_t probeCount,
                   const DDGIAtlasLayout &irradianceAtlas,
                   const DDGIAtlasLayout &distanceAtlas) noexcept;
[[nodiscard]] NURI_API glm::uvec2
ddgiAtlasTileCoordinate(uint32_t probeIndex,
                        const DDGIAtlasLayout &atlas) noexcept;
[[nodiscard]] NURI_API glm::uvec2
ddgiAtlasBorderCopyCoordinate(glm::uvec2 tilePixel,
                              uint32_t interiorExtent) noexcept;

} // namespace nuri
