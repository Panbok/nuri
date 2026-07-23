#include "nuri/gfx/ddgi/ddgi_atlas.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace nuri {
namespace {

[[nodiscard]] constexpr uint64_t ceilDiv(uint64_t value,
                                         uint64_t divisor) noexcept {
  return value / divisor + (value % divisor != 0u ? 1u : 0u);
}

[[nodiscard]] uint32_t ceilSqrt(uint32_t value) noexcept {
  uint32_t root = static_cast<uint32_t>(std::sqrt(static_cast<double>(value)));
  while (static_cast<uint64_t>(root) * root < value) {
    ++root;
  }
  while (root > 0u && static_cast<uint64_t>(root - 1u) * (root - 1u) >= value) {
    --root;
  }
  return root;
}

[[nodiscard]] bool checkedMultiply(uint64_t left, uint64_t right,
                                   uint64_t &out) noexcept {
  if (right != 0u && left > std::numeric_limits<uint64_t>::max() / right) {
    return false;
  }
  out = left * right;
  return true;
}

[[nodiscard]] bool checkedAdd(uint64_t left, uint64_t right,
                              uint64_t &out) noexcept {
  if (left > std::numeric_limits<uint64_t>::max() - right) {
    return false;
  }
  out = left + right;
  return true;
}

[[nodiscard]] bool atlasBytes(const DDGIAtlasLayout &atlas,
                              uint32_t bytesPerTexel, uint64_t &out) noexcept {
  uint64_t texels = 0u;
  return atlas.columns != 0u && atlas.rows != 0u &&
         atlas.textureExtent ==
             atlas.tileExtent * glm::uvec2(atlas.columns, atlas.rows) &&
         checkedMultiply(atlas.textureExtent.x, atlas.textureExtent.y,
                         texels) &&
         checkedMultiply(texels, bytesPerTexel, out);
}

} // namespace

Result<DDGIAtlasLayout, DDGIAtlasError>
packDDGIAtlas(uint32_t probeCount, glm::uvec2 tileExtent,
              glm::uvec2 maxTextureExtent) noexcept {
  using PackingResult = Result<DDGIAtlasLayout, DDGIAtlasError>;
  if (probeCount == 0u || tileExtent.x == 0u || tileExtent.y == 0u ||
      maxTextureExtent.x == 0u || maxTextureExtent.y == 0u) {
    return PackingResult::makeError(DDGIAtlasError::InvalidArgument);
  }

  const uint32_t maxColumns = maxTextureExtent.x / tileExtent.x;
  const uint32_t maxRows = maxTextureExtent.y / tileExtent.y;
  if (maxColumns == 0u || maxRows == 0u) {
    return PackingResult::makeError(DDGIAtlasError::TextureLimitExceeded);
  }

  uint64_t columns = std::min<uint64_t>(maxColumns, ceilSqrt(probeCount));
  uint64_t rows = ceilDiv(probeCount, columns);
  if (rows > maxRows) {
    columns = ceilDiv(probeCount, maxRows);
    rows = ceilDiv(probeCount, columns);
  }
  if (columns == 0u || columns > maxColumns || rows == 0u || rows > maxRows) {
    return PackingResult::makeError(DDGIAtlasError::TextureLimitExceeded);
  }

  const uint64_t width = columns * tileExtent.x;
  const uint64_t height = rows * tileExtent.y;
  if (width > std::numeric_limits<uint32_t>::max() ||
      height > std::numeric_limits<uint32_t>::max()) {
    return PackingResult::makeError(DDGIAtlasError::ArithmeticOverflow);
  }
  return PackingResult::makeResult(DDGIAtlasLayout{
      .tileExtent = tileExtent,
      .columns = static_cast<uint32_t>(columns),
      .rows = static_cast<uint32_t>(rows),
      .textureExtent = glm::uvec2(static_cast<uint32_t>(width),
                                  static_cast<uint32_t>(height)),
  });
}

Result<DDGIMemoryEstimate, DDGIAtlasError>
estimateDDGIMemory(uint32_t probeCount, const DDGIAtlasLayout &irradianceAtlas,
                   const DDGIAtlasLayout &distanceAtlas) noexcept {
  using MemoryResult = Result<DDGIMemoryEstimate, DDGIAtlasError>;
  if (probeCount == 0u) {
    return MemoryResult::makeError(DDGIAtlasError::InvalidArgument);
  }

  DDGIMemoryEstimate estimate{};
  if (!atlasBytes(irradianceAtlas, kDDGIIrradianceBytesPerTexel,
                  estimate.irradianceBytes) ||
      !atlasBytes(distanceAtlas, kDDGIDistanceBytesPerTexel,
                  estimate.distanceBytes) ||
      !checkedMultiply(probeCount, kDDGIProbeStateBytes,
                       estimate.probeStateBytes)) {
    return MemoryResult::makeError(DDGIAtlasError::ArithmeticOverflow);
  }
  uint64_t atlasTotal = 0u;
  if (!checkedAdd(estimate.irradianceBytes, estimate.distanceBytes,
                  atlasTotal) ||
      !checkedAdd(atlasTotal, estimate.probeStateBytes,
                  estimate.persistentBytes)) {
    return MemoryResult::makeError(DDGIAtlasError::ArithmeticOverflow);
  }
  return MemoryResult::makeResult(estimate);
}

glm::uvec2 ddgiAtlasTileCoordinate(uint32_t probeIndex,
                                   const DDGIAtlasLayout &atlas) noexcept {
  if (atlas.columns == 0u) {
    return glm::uvec2(0u);
  }
  return glm::uvec2(probeIndex % atlas.columns, probeIndex / atlas.columns);
}

glm::uvec2 ddgiAtlasBorderCopyCoordinate(glm::uvec2 tilePixel,
                                         uint32_t interiorExtent) noexcept {
  const uint32_t tileExtent = interiorExtent + 2u;
  const bool corner = (tilePixel.x == 0u || tilePixel.x == tileExtent - 1u) &&
                      (tilePixel.y == 0u || tilePixel.y == tileExtent - 1u);
  const bool row = tilePixel.x > 0u && tilePixel.x < tileExtent - 1u;
  if (corner) {
    return glm::uvec2(tilePixel.x > 0u ? 1u : interiorExtent,
                      tilePixel.y > 0u ? 1u : interiorExtent);
  }
  if (row) {
    return glm::uvec2(tileExtent - 1u - tilePixel.x,
                      tilePixel.y > 0u ? tilePixel.y - 1u : tilePixel.y + 1u);
  }
  return glm::uvec2(tilePixel.x > 0u ? tilePixel.x - 1u : tilePixel.x + 1u,
                    tileExtent - 1u - tilePixel.y);
}

} // namespace nuri
