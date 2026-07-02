#include "meshlet_common.sp"

layout(location = 0) in vec2 inUv0;
layout(location = 1) in vec2 inUv1;

vec2 shadowMeshletAlphaBaseColorUv(MaterialData material) {
  return applyTextureTransform(
      selectUv(inUv0, inUv1,
               getMaterialUvSet(material, kMaterialTextureSlotBaseColor)),
      getMaterialTransform(material, kMaterialTextureSlotBaseColor));
}

void main() {
  const MaterialData material = loadMaterialData(pc.materialIndex);
  const uint baseColorTexId =
      getMaterialTextureIndex(material, kMaterialTextureSlotBaseColor);
  const vec2 baseColorUv = shadowMeshletAlphaBaseColorUv(material);

  vec4 baseColor = material.header.baseColorFactor;
  if (baseColorTexId != kInvalidTextureBindlessIndex) {
    baseColor *= textureBindless2D(
        baseColorTexId, pc.frameData.materialCoverageSamplerId, baseColorUv);
  }

  if (baseColor.a < material.header.metallicRoughnessOcclusionAlphaCutoff.w) {
    discard;
  }
}
