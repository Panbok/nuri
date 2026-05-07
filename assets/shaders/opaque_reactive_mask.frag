#include "common.sp"

layout(location = 0) in PerVertex vtx;
layout(location = 10) flat in uint inMotionReactive;

layout(location = 0) out float out_ReactiveMask;

const float kFullReactiveWeight = 1.0;

void main() {
  if (inMotionReactive == 0u) {
    discard;
  }

  const MaterialData material = loadMaterialData(pc.materialIndex);
  const bool alphaMasked = materialAlphaMode(material) == kAlphaModeMask;

  if (alphaMasked) {
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

  out_ReactiveMask = kFullReactiveWeight;
}
