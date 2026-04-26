#include "common.sp"

layout(location = 0) in PerVertex vtx;

layout(location = 0) out float out_ReactiveMask;

void main() {
  const MaterialData material = loadMaterialData(pc.materialIndex);
  const uint baseColorTexId =
      getMaterialTextureIndex(material, kMaterialTextureSlotBaseColor);
  const vec2 baseColorUv =
      transformedUv(material, vtx, kMaterialTextureSlotBaseColor);

  vec4 baseColor = material.header.baseColorFactor;
  if (baseColorTexId != kInvalidTextureBindlessIndex) {
    baseColor *= textureBindless2D(baseColorTexId,
                                   pc.frameData.materialSamplerId, baseColorUv);
  }

  if (baseColor.a < material.header.metallicRoughnessOcclusionAlphaCutoff.w) {
    discard;
  }

  out_ReactiveMask = 1.0;
}
