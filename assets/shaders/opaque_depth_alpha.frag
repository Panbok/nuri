#include "common.sp"

layout(location = 0) in PerVertex vtx;

void main() {
  const MaterialData material = loadMaterialData(pc.materialIndex);
  const uint baseColorTexId =
      getMaterialTextureIndex(material, kMaterialTextureSlotBaseColor);
  const vec2 baseColorUv =
      transformedUv(material, vtx, kMaterialTextureSlotBaseColor);

  vec4 baseColor = material.header.baseColorFactor;
  if (baseColorTexId != kInvalidTextureBindlessIndex) {
    baseColor *= textureBindless2D(
        baseColorTexId, pc.frameData.materialCoverageSamplerId, baseColorUv);
  }

  if (baseColor.a < material.header.metallicRoughnessOcclusionAlphaCutoff.w) {
    discard;
  }
}
