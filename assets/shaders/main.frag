#include "BRDF.sp"
#include "common.sp"
#include "material_inputs.sp"
#include "material_lighting.sp"

layout(location = 0) in PerVertex vtx;

layout(location = 0) out vec4 out_FragColor;

vec3 shadowCascadeDebugColor(uint cascadeIndex) {
  vec3 cascadeColor = vec3(0.0, 1.0, 0.0);
  if (cascadeIndex == 1u) {
    cascadeColor = vec3(0.0, 1.0, 1.0);
  } else if (cascadeIndex == 2u) {
    cascadeColor = vec3(1.0, 1.0, 0.0);
  } else if (cascadeIndex == 3u) {
    cascadeColor = vec3(1.0, 0.0, 1.0);
  }
  return cascadeColor;
}

void main() {
  const MaterialData material = loadMaterialData(pc.materialIndex);
  const uint alphaMode = materialAlphaMode(material);
  const bool visualizeCascadeIndex =
      (pc.frameData.shadowFlags & kShadowFrameFlagVisualizeCascadeIndex) != 0u;

  ShadedMaterial sm = evaluateMaterial(material, vtx);

  const float alphaCutoff = materialAlphaCutoff(material);
  if (alphaMode == kAlphaModeMask && sm.baseColor.a < alphaCutoff) {
    discard;
  }

  // Direct lighting ---------------------------------------------------
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
  if ((pc.frameData.shadowFlags & kShadowFrameFlagVisualizeShadowFactor) !=
      0u) {
    out_FragColor = vec4(vec3(direct.shadowFactorDebug), 1.0);
    return;
  }

  // IBL ---------------------------------------------------------------
  IblResult ibl = evaluateIbl(sm);

  vec3 indirectLighting =
      sm.clearcoatAttenuation *
          (ibl.iblSheen +
           ibl.indirectScale * (ibl.iblDiffuse + ibl.iblSpecular)) +
      ibl.clearcoatIblSpecular;
  if (ibl.hasIndirectLighting) {
    indirectLighting *= sm.ao;
  }

  // Composition -------------------------------------------------------
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
