#include "BRDF.sp"
#include "common.sp"
#include "material_inputs.sp"
#include "material_lighting.sp"

layout(location = 0) in PerVertex vtx;

layout(location = 0) out vec4 out_FragColor;

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
  const MaterialData material = loadMaterialData(pc.materialIndex);
  const uint alphaMode = materialAlphaMode(material);
  const bool visualizeCascadeIndex =
      shadowDebugFlag(kShadowFrameFlagVisualizeCascadeIndex);

  ShadedMaterial sm = evaluateMaterial(material, vtx);

  const float alphaCutoff = materialAlphaCutoff(material);
  if (alphaMode == kAlphaModeMask && sm.baseColor.a < alphaCutoff) {
    discard;
  }

  const uint aoDebugView = getAmbientOcclusionDebugView(pc.frameData);
  if ((pc.frameData.flags & kFrameDataFlagHasAmbientOcclusion) != 0u) {
    if (aoDebugView == kAmbientOcclusionDebugViewVisibility) {
      out_FragColor = vec4(vec3(sm.screenAo), 1.0);
      return;
    }
    if (aoDebugView == kAmbientOcclusionDebugViewBentNormal) {
      out_FragColor = vec4(normalize(sm.ambientBentNormal) * 0.5 + 0.5, 1.0);
      return;
    }
    if (aoDebugView == kAmbientOcclusionDebugViewNormals) {
      out_FragColor = vec4(normalize(sm.nBase) * 0.5 + 0.5, 1.0);
      return;
    }
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
  if (shadowDebugFlag(kShadowFrameFlagVisualizePCSSBlockers)) {
    out_FragColor = shadowDebugScalar(direct.shadowPcssBlockerRatioDebug);
    return;
  }
  if (shadowDebugFlag(kShadowFrameFlagVisualizePCSSAverageBlockerDepth)) {
    out_FragColor =
        shadowDebugScalar(direct.shadowPcssAverageBlockerDepthDebug);
    return;
  }
  if (shadowDebugFlag(kShadowFrameFlagVisualizePCSSFilterRadius)) {
    out_FragColor = shadowDebugScalar(direct.shadowPcssFilterRadiusDebug);
    return;
  }
  if (shadowDebugFlag(kShadowFrameFlagVisualizeShadowFactor)) {
    out_FragColor = shadowDebugScalar(direct.shadowFactorDebug);
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
    indirectLighting *= sm.ao * sm.screenAo;
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
