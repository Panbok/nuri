#pragma once

#include "nuri/resources/gpu/resource_manager.h"
#include "nuri/scene/render_scene.h"

namespace nuri {

[[nodiscard]] inline MaterialRef
resolveRenderableMaterial(const Renderable &renderable,
                          const ModelRecord &modelRecord,
                          uint32_t submeshIndex) {
  if (isValid(renderable.materialOverride)) {
    return renderable.materialOverride;
  }

  MaterialRef modelMaterial = modelRecord.materialForSubmesh(submeshIndex);
  if (!isValid(modelMaterial)) {
    modelMaterial = modelRecord.materialForSource(submeshIndex);
  }
  if (isValid(modelMaterial)) {
    return modelMaterial;
  }

  return renderable.material;
}

} // namespace nuri
