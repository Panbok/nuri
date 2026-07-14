#include "common.sp"

layout(location = 0) in vec2 inUv0;
layout(location = 1) in vec2 inUv1;

vec2 depthAlphaBaseColorUv(MaterialHeaderGpuData material) {
  return applyTextureTransform(
      selectUv(inUv0, inUv1, getPackedUvBit(material.uvSetBits, 0u)),
      material.commonTransforms[0]);
}

void main() {
  const MaterialHeaderGpuData material =
      pc.frameData.materialHeaderBuffer.materials[pc.materialIndex];
  const uint baseColorTexId = material.commonTextureIndices.x;
  const vec2 baseColorUv = depthAlphaBaseColorUv(material);

  vec4 baseColor = material.baseColorFactor;
  if (baseColorTexId != kInvalidTextureBindlessIndex) {
    baseColor *= textureBindless2D(
        baseColorTexId, pc.frameData.materialCoverageSamplerId, baseColorUv);
  }

  if (baseColor.a < material.metallicRoughnessOcclusionAlphaCutoff.w) {
    discard;
  }
}
