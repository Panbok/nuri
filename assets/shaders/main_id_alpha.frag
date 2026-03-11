#include "common.sp"

layout(location = 0) in PerVertex vtx;
layout(location = 10) flat in uint inInstanceId;

layout(location = 0) out uint outObjectId;

vec2 selectUv(vec2 uv0, vec2 uv1, uint uvSet) {
  return (uvSet == 1u) ? uv1 : uv0;
}

void main() {
  const MaterialGpuData material = pc.materialBuffer.materials[pc.materialIndex];
  const uint baseColorTexId = material.textureIndices0.x;
  const uint baseColorUvSet = material.textureUvSets0.x;
  const uint baseColorSampler = material.textureSamplerIndices0.x;

  vec2 baseColorUv = applyTextureTransform(
      selectUv(vtx.uv0, vtx.uv1, baseColorUvSet),
      material.textureTransformOffsetScale[kMaterialTextureSlotBaseColor],
      material.textureTransformRotation[kMaterialTextureSlotBaseColor]);

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
