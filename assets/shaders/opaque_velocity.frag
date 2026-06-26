#include "common.sp"

layout(location = 0) in PerVertex vtx;
layout(location = 10) in vec4 inCurrentClipNoJitter;
layout(location = 11) in vec4 inPreviousClipNoJitter;
layout(location = 12) flat in uint inVelocityFlags;

layout(location = 0) out vec2 out_FragVelocity;

vec2 clipNdcToTaaScreenUv(vec2 ndc) {
  // Must match taa_resolve.frag::taaScreenUv(). fullscreen_copy.vert feeds TAA
  // with the render-target UV space where Y is flipped relative to clip NDC.
  return vec2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

void main() {
  const MaterialHeaderGpuData material =
      pc.frameData.materialHeaderBuffer.materials[pc.materialIndex];
  if ((material.materialFlags & kMaterialFlagsAlphaModeMask) ==
      kAlphaModeMask) {
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
  }

  if (inVelocityFlags == 0u || inCurrentClipNoJitter.w == 0.0 ||
      inPreviousClipNoJitter.w == 0.0) {
    out_FragVelocity = vec2(0.0);
    return;
  }

  const vec2 currentNdc = inCurrentClipNoJitter.xy / inCurrentClipNoJitter.w;
  const vec2 previousNdc = inPreviousClipNoJitter.xy / inPreviousClipNoJitter.w;
  // Velocity is stored in TAA screen-UV space. The resolve samples:
  // historyUv = currentUv + velocity. Jitter is excluded by the no-jitter
  // matrices above; only the fullscreen-copy Y convention is applied here.
  const vec2 currentUv = clipNdcToTaaScreenUv(currentNdc);
  const vec2 previousUv = clipNdcToTaaScreenUv(previousNdc);
  out_FragVelocity = previousUv - currentUv;
}
