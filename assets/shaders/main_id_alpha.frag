#include "common.sp"

layout(location = 0) in PerVertex vtx;
layout(location = 10) flat in uint inInstanceId;

layout(location = 0) out uint outObjectId;

void main() {
  const MaterialData material = loadMaterialData(pc.materialIndex);
  const uint baseColorTexId =
      getMaterialTextureIndex(material, kMaterialTextureSlotBaseColor);
  const uint baseColorSampler = pc.frameData.materialCoverageSamplerId;

  const vec2 baseColorUv =
      transformedUv(material, vtx, kMaterialTextureSlotBaseColor);

  vec4 baseColor = material.header.baseColorFactor;
  if (baseColorTexId != kInvalidTextureBindlessIndex) {
    baseColor *=
        textureBindless2D(baseColorTexId, baseColorSampler, baseColorUv);
  }

  if (baseColor.a <= 1.0e-3) {
    discard;
  }

  outObjectId =
      (inInstanceId == 0xFFFFFFFFu) ? 0xFFFFFFFFu : (inInstanceId + 1u);
}
