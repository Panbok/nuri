#pragma once

#include "nuri/resources/gpu/resource_manager.h"

namespace nuri {

[[nodiscard]] inline MaterialRef
resolveSceneSubmeshMaterial(const ModelRecord &model, uint32_t submeshIndex,
                            MaterialRef fallback,
                            MaterialRef overrideMaterial) noexcept {
  if (isValid(overrideMaterial)) {
    return overrideMaterial;
  }
  MaterialRef material = model.materialForSubmesh(submeshIndex);
  if (!isValid(material)) {
    material = model.materialForSource(submeshIndex);
  }
  return isValid(material) ? material : fallback;
}

[[nodiscard]] inline bool
isDDGIRayVisibleMaterial(const MaterialRecord &material) noexcept {
  return material.desc.alphaMode != MaterialAlphaMode::Blend &&
         (material.desc.featureMask & kMaterialFeatureTransmission) == 0u;
}

} // namespace nuri
