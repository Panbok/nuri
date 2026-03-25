#include "common.sp"

layout(location = 0) in PerVertex vtx;
layout(location = 10) flat in uint inInstanceId;

layout(location = 0) out uint outObjectId;

void main() {
  const MaterialGpuData material = pc.materialBuffer.materials[pc.materialIndex];
  const uint baseColorTexId =
      GET_TEXTURE_INDEX(material, kMaterialTextureSlotBaseColor);
  const uint baseColorSampler = pc.frameData.materialSamplerId;

  const vec2 baseColorUv =
      transformedUv(material, vtx, kMaterialTextureSlotBaseColor);

  vec4 baseColor = material.baseColorFactor;
  if (baseColorTexId != kInvalidTextureBindlessIndex) {
    baseColor *=
        textureBindless2D(baseColorTexId, baseColorSampler, baseColorUv);
  }

  if (baseColor.a <= 1.0e-3) {
    discard;
  }

  outObjectId = (inInstanceId >= 0xFFFFFFFFu) ? 0xFFFFFFFFu : (inInstanceId + 1u);
}
