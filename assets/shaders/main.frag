#include "common.sp"
#include "BRDF.sp"
#include "material_inputs.sp"
#include "material_lighting.sp"

layout(location = 0) in PerVertex vtx;

layout(location = 0) out vec4 out_FragColor;

void main() {
  const MaterialGpuData material = pc.materialBuffer.materials[pc.materialIndex];
  const uint alphaMode = material.materialFlags.x;

  ShadedMaterial sm = evaluateMaterial(material, vtx);

  const float alphaCutoff = material.metallicRoughnessOcclusionAlphaCutoff.w;
  if (alphaMode == kAlphaModeMask && sm.baseColor.a < alphaCutoff) {
    discard;
  }

  // Direct lighting ---------------------------------------------------
  vec3 directDiffuse           = vec3(0.0);
  vec3 directSpecular          = vec3(0.0);
  vec3 directSheen             = vec3(0.0);
  vec3 clearcoatDirectLighting = vec3(0.0);

  for (uint i = 0u; i < pc.frameData.directionalLightCount; ++i) {
    DirectionalLightGpuData light =
        pc.frameData.directionalLightBuffer.lights[i];
    vec3 l = normalize(-directionalLightDirection(light));
    vec3 lr =
        directionalLightColor(light) * directionalLightIlluminance(light);
    accumulateSurfaceLightContribution(lr, l, sm,
        directDiffuse, directSpecular, directSheen, clearcoatDirectLighting);
  }

  for (uint i = 0u; i < pc.frameData.localLightCount; ++i) {
    LocalLightGpuData light = pc.frameData.localLightBuffer.lights[i];
    vec3 ptl = localLightPosition(light) - vtx.worldPos;
    float dsq = dot(ptl, ptl);
    if (dsq <= kEpsilon) {
      continue;
    }
    vec3  l   = ptl * inversesqrt(dsq);
    float att = punctualRangeAttenuation(dsq, localLightRange(light));
    if (localLightType(light) == kLocalLightTypeSpot) {
      att *= spotAngularAttenuation(localLightDirection(light), ptl,
                                    localLightInnerCos(light),
                                    localLightOuterCos(light));
    }
    if (att <= 0.0) {
      continue;
    }
    vec3 lr =
        localLightColor(light) * localLightIntensity(light) * att;
    accumulateSurfaceLightContribution(lr, l, sm,
        directDiffuse, directSpecular, directSheen, clearcoatDirectLighting);
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
          (directSheen + directDiffuse + directSpecular) +
      clearcoatDirectLighting;
  vec3 color =
      directLighting + indirectLighting + sm.clearcoatAttenuation * sm.emissive;
  color = max(color, vec3(0.0));
  if ((pc.frameData.flags & kFrameDataFlagOutputLinearToSrgb) != 0u) {
    color = linearToSrgb(color);
  }

  float outAlpha = (alphaMode == kAlphaModeOpaque) ? 1.0 : sm.baseColor.a;
  out_FragColor = vec4(color, outAlpha);
}
