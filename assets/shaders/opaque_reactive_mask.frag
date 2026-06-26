#include "common.sp"

layout(location = 0) in PerVertex vtx;
layout(location = 10) flat in uint inMotionReactive;

layout(location = 0) out float out_ReactiveMask;

const float kMotionReactiveWeight = 1.0;
const float kAlphaMaskReactiveWeight = 0.12;

void main() {
  const MaterialHeaderGpuData material =
      pc.frameData.materialHeaderBuffer.materials[pc.materialIndex];
  const bool alphaMasked =
      (material.materialFlags & kMaterialFlagsAlphaModeMask) == kAlphaModeMask;

  if (alphaMasked) {
    const uint baseColorTexId = material.commonTextureIndices.x;
    const vec2 baseColorUv = applyTextureTransform(
        selectUv(vtx.uv0, vtx.uv1, getPackedUvBit(material.uvSetBits, 0u)),
        material.commonTransforms[0]);

    vec4 baseColor = material.baseColorFactor;
    if (baseColorTexId != kInvalidTextureBindlessIndex) {
      baseColor *= textureBindless2D(
          baseColorTexId, pc.frameData.materialCoverageSamplerId, baseColorUv);
    }

    if (baseColor.a < material.metallicRoughnessOcclusionAlphaCutoff.w) {
      discard;
    }
  } else if (inMotionReactive == 0u) {
    discard;
  }

  out_ReactiveMask =
      inMotionReactive != 0u ? kMotionReactiveWeight : kAlphaMaskReactiveWeight;
}
