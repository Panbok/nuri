#include "common.sp"

layout(location = 0) in PerVertex vtx;
layout(location = 10) in vec4 inCurrentClipNoJitter;
layout(location = 11) in vec4 inPreviousClipNoJitter;
layout(location = 12) flat in uint inVelocityFlags;

layout(location = 0) out vec2 out_FragVelocity;

void main() {
  const MaterialData material = loadMaterialData(pc.materialIndex);
  if (materialAlphaMode(material) == 1u) {
    const uint baseColorTexId =
        getMaterialTextureIndex(material, kMaterialTextureSlotBaseColor);
    const vec2 baseColorUv =
        transformedUv(material, vtx, kMaterialTextureSlotBaseColor);

    vec4 baseColor = material.header.baseColorFactor;
    if (baseColorTexId != kInvalidTextureBindlessIndex) {
      baseColor *= textureBindless2D(
          baseColorTexId, pc.frameData.materialSamplerId, baseColorUv);
    }

    if (baseColor.a < material.header.metallicRoughnessOcclusionAlphaCutoff.w) {
      discard;
    }
  }

  if (inVelocityFlags == 0u || inCurrentClipNoJitter.w == 0.0 ||
      inPreviousClipNoJitter.w == 0.0) {
    out_FragVelocity = vec2(0.0);
    return;
  }

  const vec2 currentNdc = inCurrentClipNoJitter.xy / inCurrentClipNoJitter.w;
  const vec2 previousNdc = inPreviousClipNoJitter.xy / inPreviousClipNoJitter.w;
  const vec2 currentUv = currentNdc * 0.5 + vec2(0.5);
  const vec2 previousUv = previousNdc * 0.5 + vec2(0.5);
  out_FragVelocity = previousUv - currentUv;
}
