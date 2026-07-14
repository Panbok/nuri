#pragma once

#include "nuri/gfx/frame/render_frame_context.h"

#include <cstdint>

namespace nuri::detail {

// Index-only generated LODs share the LOD0 vertex attributes. LOD1 remains
// stable with attribute-aware simplification, while deeper levels require
// per-LOD vertex attributes to avoid visible shading and UV deformation.
// Keep deeper levels available for explicit inspection, but exclude them from
// automatic opaque rendering until the asset format can store those vertices.
inline constexpr uint32_t kMaxStableGeneratedOpaqueLod = 1u;
[[nodiscard]] constexpr uint32_t
resolveOpaqueAutomaticLod(uint32_t requestedLod, bool alphaMasked,
                          bool automaticLod) noexcept {
  if (!automaticLod) {
    return requestedLod;
  }
  if (alphaMasked) {
    return 0u;
  }
  return requestedLod > kMaxStableGeneratedOpaqueLod
             ? kMaxStableGeneratedOpaqueLod
             : requestedLod;
}

[[nodiscard]] constexpr bool shouldUseMeshletsForOpaqueBatch(
    MeshletRenderMode mode, bool hybridEligible, bool alphaMasked,
    bool doubleSided, bool meshletDataValid, uint32_t resolvedLodMeshletCount,
    uint32_t hybridClassicMaxMeshlets) noexcept {
  if (mode == MeshletRenderMode::Disabled || !meshletDataValid ||
      resolvedLodMeshletCount == 0u) {
    return false;
  }
  if (mode == MeshletRenderMode::Required || !hybridEligible ||
      hybridClassicMaxMeshlets == 0u) {
    return true;
  }
  if (alphaMasked || doubleSided) {
    return false;
  }
  return resolvedLodMeshletCount > hybridClassicMaxMeshlets;
}

} // namespace nuri::detail
