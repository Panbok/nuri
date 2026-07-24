#define NURI_OPAQUE_MESHLET_BATCHED 1
// clang-format off
#include "meshlet_common.sp"
#include "opaque_meshlet_vertex.sp"
#include "BRDF.sp"
#include "material_inputs.sp"
#include "material_lighting.sp"
// clang-format on

layout(location = 0) in OpaqueMeshletVertex vtx;
layout(location = 5) flat in uint meshletDebugId;
layout(location = 6) flat in uint meshletDebugLod;
layout(location = 7) flat in uint meshletMaterialIndex;
layout(location = 8) flat in uint meshletFlags;

layout(location = 0) out vec4 out_FragColor;

vec3 meshletDebugPalette(uint value) {
  value ^= value >> 16u;
  value *= 0x7feb352du;
  value ^= value >> 15u;
  value *= 0x846ca68bu;
  value ^= value >> 16u;
  return vec3(float(value & 0xffu), float((value >> 8u) & 0xffu),
              float((value >> 16u) & 0xffu)) *
         (1.0 / 255.0);
}

vec3 meshletLodDebugColor(uint lod) {
  const vec3 colors[4] =
      vec3[4](vec3(0.08, 0.80, 0.92), vec3(0.26, 0.86, 0.25),
              vec3(0.95, 0.72, 0.12), vec3(0.95, 0.22, 0.14));
  return colors[int(min(lod, 3u))];
}

vec3 shadowCascadeDebugColor(uint cascadeIndex) {
  const vec3 colors[4] = vec3[4](vec3(0.0, 1.0, 0.0), vec3(0.0, 1.0, 1.0),
                                 vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 1.0));
  return colors[int(min(cascadeIndex, 3u))];
}

bool shadowDebugFlag(uint flag) {
  return (pc.frameData.shadowFlags & flag) != 0u;
}

vec4 shadowDebugScalar(float value) { return vec4(vec3(saturate(value)), 1.0); }

void main() {
  if ((meshletFlags & kMeshletFlagDebugMeshletId) != 0u) {
    out_FragColor = vec4(meshletDebugPalette(meshletDebugId), 1.0);
    return;
  }
  if ((meshletFlags & kMeshletFlagDebugSelectedLod) != 0u) {
    out_FragColor = vec4(meshletLodDebugColor(meshletDebugLod), 1.0);
    return;
  }

  const MaterialData material = loadMaterialData(meshletMaterialIndex);
  const uint alphaMode = materialAlphaMode(material);
  const bool visualizeCascadeIndex =
      shadowDebugFlag(kShadowFrameFlagVisualizeCascadeIndex);

  const PerVertex materialVertex = opaqueMeshletMaterialVertex(vtx);
  ShadedMaterial sm = evaluateMaterial(material, materialVertex);

  const float alphaCutoff = materialAlphaCutoff(material);
  if (alphaMode == kAlphaModeMask && sm.baseColor.a < alphaCutoff) {
    discard;
  }

  vec4 aoDebugColor;
  if (tryAmbientOcclusionDebugColor(pc.frameData, sm, aoDebugColor)) {
    out_FragColor = aoDebugColor;
    return;
  }
  vec4 specularAADebugColor;
  if (trySpecularAADebugColor(sm, specularAADebugColor)) {
    out_FragColor = specularAADebugColor;
    return;
  }
  vec4 ddgiDebugColor;
  if (tryDDGIDebugColor(sm, vtx.worldPos, ddgiDebugColor)) {
    out_FragColor = ddgiDebugColor;
    return;
  }

  DirectLightingResult direct = evaluateDirectLighting(sm, vtx.worldPos);
  if (visualizeCascadeIndex) {
    const uint cascadeIndex = uint(clamp(direct.shadowCascadeIndexDebug, 0.0,
                                         float(kMaxShadowCascades - 1u)));
    vec3 cascadeColor = shadowCascadeDebugColor(cascadeIndex);
    const float cascadeBlend = saturate(direct.shadowCascadeBlendDebug);
    if (cascadeBlend > 0.0 && cascadeIndex + 1u < kMaxShadowCascades) {
      cascadeColor =
          mix(cascadeColor, shadowCascadeDebugColor(cascadeIndex + 1u),
              cascadeBlend);
    }
    out_FragColor = vec4(cascadeColor, 1.0);
    return;
  }
  if (shadowDebugFlag(kShadowFrameFlagVisualizePCFResult)) {
    out_FragColor = shadowDebugScalar(direct.shadowPcfFactorDebug);
    return;
  }
  if (shadowDebugFlag(kShadowFrameFlagVisualizeReceiverDepth)) {
    out_FragColor = shadowDebugScalar(direct.shadowReceiverDepthDebug);
    return;
  }
  if (shadowDebugFlag(kShadowFrameFlagVisualizeShadowMapDepth)) {
    out_FragColor = shadowDebugScalar(direct.shadowMapDepthDebug);
    return;
  }
  if (shadowDebugFlag(kShadowFrameFlagVisualizeShadowFactor)) {
    out_FragColor = shadowDebugScalar(direct.shadowFactorDebug);
    return;
  }

  IblResult ibl = evaluateIbl(sm, vtx.worldPos);

  vec3 indirectLighting =
      sm.clearcoatAttenuation *
          (ibl.iblSheen +
           ibl.indirectScale * (ibl.iblDiffuse + ibl.iblSpecular)) +
      ibl.clearcoatIblSpecular;
  if (ibl.hasIndirectLighting) {
    indirectLighting *= sm.ao * sm.screenAo;
  }

  vec3 directLighting =
      sm.clearcoatAttenuation *
          (direct.directSheen + direct.directDiffuse + direct.directSpecular) +
      direct.clearcoatDirectLighting;
  vec3 color =
      directLighting + indirectLighting + sm.clearcoatAttenuation * sm.emissive;
  color = max(color, vec3(0.0));

  float outAlpha = (alphaMode == kAlphaModeOpaque) ? 1.0 : sm.baseColor.a;
  out_FragColor = vec4(color, outAlpha);
}
