#include "BRDF.sp"
#include "common.sp"
#include "material_inputs.sp"
#include "material_lighting.sp"


layout(location = 0) in PerVertex vtx;
layout(location = 10) flat in uint inInstanceId;

layout(location = 0) out vec4 out_FragColor;

// ---------------------------------------------------------------------------
// Transmission-specific helpers
// ---------------------------------------------------------------------------

float applyIorToRoughness(float roughness, float ior) {
  if (isIorCompatMode(ior)) {
    return roughness;
  }
  return roughness * clamp(ior * 2.0 - 2.0, 0.0, 1.0);
}

vec3 getVolumeTransmissionRay(vec3 n, vec3 v, float thickness, float ior,
                              mat4 modelMatrix) {
  vec3 refractionVector = refract(-v, n, 1.0 / max(ior, 1.0));
  if (dot(refractionVector, refractionVector) <= kEpsilon) {
    refractionVector = -v;
  }
  vec3 modelScale = vec3(length(modelMatrix[0].xyz), length(modelMatrix[1].xyz),
                         length(modelMatrix[2].xyz));
  return refractionVector * thickness * modelScale;
}

vec3 applyVolumeAttenuation(vec3 radiance, float transmissionDistance,
                            vec3 attenuationColor, float attenuationDistance) {
  if (attenuationDistance == 0.0) {
    return radiance;
  }
  vec3 safeColor = max(attenuationColor, vec3(1.0e-4));
  vec3 attenuationCoefficient = -log(safeColor) / attenuationDistance;
  vec3 transmittance = exp(-attenuationCoefficient * transmissionDistance);
  return transmittance * radiance;
}

float transmissionFramebufferLod(float roughness, float ior) {
  return applyIorToRoughness(roughness, ior) * 2.0;
}

vec3 sampleTransmissionColor(vec2 uv, float roughness, float ior) {
  vec2 clampedUv = clamp(uv, vec2(0.0), vec2(1.0));
  vec3 level0 = textureBindless2D(getSceneColorPyramidTexId(pc.frameData, 0u),
                                  pc.frameData.sceneColorSamplerId, clampedUv)
                    .rgb;
  vec3 level1 = level0;
  vec3 level2 = level0;

  uint level1TexId = getSceneColorPyramidTexId(pc.frameData, 1u);
  if (level1TexId != kInvalidTextureBindlessIndex) {
    level1 = textureBindless2D(level1TexId, pc.frameData.sceneColorSamplerId,
                               clampedUv)
                 .rgb;
  }
  uint level2TexId = getSceneColorPyramidTexId(pc.frameData, 2u);
  if (level2TexId != kInvalidTextureBindlessIndex) {
    level2 = textureBindless2D(level2TexId, pc.frameData.sceneColorSamplerId,
                               clampedUv)
                 .rgb;
  } else {
    level2 = level1;
  }

  float lod = clamp(transmissionFramebufferLod(roughness, ior), 0.0, 2.0);
  if (lod <= 1.0) {
    return mix(level0, level1, lod);
  }
  return mix(level1, level2, lod - 1.0);
}

vec2 currentScreenUv() {
  ivec2 sceneSize = textureBindlessSize2D(pc.frameData.sceneColorTexId);
  vec2 safeSize = max(vec2(sceneSize), vec2(1.0));
  return clamp(gl_FragCoord.xy / safeSize, vec2(0.0), vec2(1.0));
}

vec2 resolveTransmissionUv(vec3 refractedRayExit) {
  vec4 clipPos =
      pc.frameData.proj * pc.frameData.view * vec4(refractedRayExit, 1.0);
  if (clipPos.w <= kEpsilon) {
    return currentScreenUv();
  }
  vec2 ndc = clipPos.xy / clipPos.w;
  vec2 uv = ndc * 0.5 + 0.5;
  uv.y = 1.0 - uv.y;
  return clamp(uv, vec2(0.0), vec2(1.0));
}

vec3 getIndirectTransmission(vec3 n, vec3 v, float roughness, vec3 baseColor,
                             vec3 f0, vec3 f90, vec3 worldPos, mat4 modelMatrix,
                             float ior, float thickness, vec3 attenuationColor,
                             float attenuationDistance) {
  vec3 transmissionRay =
      getVolumeTransmissionRay(n, v, thickness, ior, modelMatrix);
  vec3 refractedRayExit = worldPos + transmissionRay;

  vec2 refractionCoords = resolveTransmissionUv(refractedRayExit);
  vec3 transmittedLight =
      sampleTransmissionColor(refractionCoords, roughness, ior);
  vec3 attenuatedColor =
      applyVolumeAttenuation(transmittedLight, length(transmissionRay),
                             attenuationColor, attenuationDistance);

  float ndotv = max(dot(n, v), 0.001);
  vec3 specularColor = vec3(0.0);
  if ((pc.frameData.flags & kFrameDataFlagHasBrdfLut) != 0u &&
      pc.frameData.brdfLutTexId != kInvalidTextureBindlessIndex) {
    // The transmission BRDF term follows the glTF reference path and samples
    // the LUT in perceptual-roughness space. Using the engine's generic
    // environment-space remap here can collapse smooth transmissive surfaces
    // toward zero contribution.
    vec2 brdfUv = clamp(vec2(ndotv, roughness), vec2(0.0), vec2(1.0));
    vec2 brdf = textureBindless2D(pc.frameData.brdfLutTexId, 0u, brdfUv).rg;
    specularColor = f0 * brdf.x + f90 * brdf.y;
  } else {
    specularColor = fresnelSchlick(ndotv, f0);
  }
  return (vec3(1.0) - specularColor) * attenuatedColor * baseColor;
}

vec3 getDirectTransmission(vec3 n, vec3 v, vec3 pointToLight,
                           float alphaRoughness, vec3 f0, vec3 f90,
                           vec3 baseColor, float ior) {
  // alphaRoughness = roughness² is passed through, scaled by the IOR factor.
  // distributionGGX and geometryOcclusion both consume alpha-space roughness²,
  // matching the convention used in accumulateSurfaceLightContribution.
  float transmissionRoughness = applyIorToRoughness(alphaRoughness, ior);

  vec3 l = normalize(pointToLight);
  vec3 lMirror = normalize(l + 2.0 * n * dot(-l, n));
  vec3 h = normalize(lMirror + v);

  float ndoth = clamp(dot(n, h), 0.0, 1.0);
  float ndotlMirror = clamp(dot(n, lMirror), 0.0, 1.0);
  float ndotv = clamp(dot(n, v), 0.0, 1.0);
  float vdoth = clamp(dot(v, h), 0.0, 1.0);

  float d = distributionGGX(ndoth, transmissionRoughness);
  vec3 f = specularReflection(vdoth, f0, f90);
  float g = geometryOcclusion(ndotlMirror, ndotv, transmissionRoughness);
  return (vec3(1.0) - f) * baseColor * d * g;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

void main() {
  const MaterialGpuData material =
      pc.materialBuffer.materials[pc.materialIndex];
  const uint alphaMode = material.materialFlags.x;
  const uint featureMask = material.materialFlags.z;

  ShadedMaterial sm = evaluateMaterial(material, vtx);

  const float alphaCutoff = material.metallicRoughnessOcclusionAlphaCutoff.w;
  if (alphaMode == kAlphaModeMask && sm.baseColor.a < alphaCutoff) {
    discard;
  }

  // Transmission-specific material fields ----------------------------
  const uint matSampler = pc.frameData.materialSamplerId;
  const uint transmissionTexId =
      GET_TEXTURE_INDEX(material, kMaterialTextureSlotTransmission);
  const uint thicknessTexId =
      GET_TEXTURE_INDEX(material, kMaterialTextureSlotThickness);
  const vec2 uvTransmission =
      transformedUv(material, vtx, kMaterialTextureSlotTransmission);
  const vec2 uvThickness =
      transformedUv(material, vtx, kMaterialTextureSlotThickness);

  float transmissionFactor = material.transmissionThicknessIorPadding.x;
  if (transmissionTexId != kInvalidTextureBindlessIndex) {
    transmissionFactor *=
        textureBindless2D(transmissionTexId, matSampler, uvTransmission).r;
  }
  transmissionFactor = saturate(transmissionFactor);

  float thickness = max(material.transmissionThicknessIorPadding.y, 0.0);
  if (thicknessTexId != kInvalidTextureBindlessIndex) {
    thickness *= textureBindless2D(thicknessTexId, matSampler, uvThickness).g;
  }
  thickness = max(thickness, 0.0);

  vec3 attenuationColor =
      clamp(material.attenuationColorDistance.rgb, vec3(0.0), vec3(1.0));
  float attenuationDistance = max(material.attenuationColorDistance.w, 0.0);

  // glTF: packed ior==0 means KHR_materials_ior was omitted (compat mode for
  // base-layer F0).  For transmission we still follow the extension default
  // IOR of 1.5.  If we gated transmission on !iorCompatMode, materials with
  // omitted IOR would keep transmissionFactor mixing (below) but never
  // accumulate indirectTransmission/directTransmission → diffuse terms go to
  // zero and glass reads black.
  const float transmissionIor = sm.iorCompatMode ? 1.5 : sm.ior;
  const bool hasTransmission = transmissionFactor > 0.0;
  mat4 modelMatrix = mat4(1.0);
  if (hasTransmission) {
    modelMatrix = pc.instanceMatrices.instances[inInstanceId].modelMatrix;
  }

  // Transmission ray for direct lights --------------------------------
  vec3 transmissionRay = vec3(0.0);
  float transmissionRayLength = 0.0;
  if (hasTransmission) {
    transmissionRay = getVolumeTransmissionRay(sm.nBase, sm.v, thickness,
                                               transmissionIor, modelMatrix);
    transmissionRayLength = length(transmissionRay);
  }

  // Direct lighting ---------------------------------------------------
  vec3 directDiffuse = vec3(0.0);
  vec3 directSpecular = vec3(0.0);
  vec3 directSheen = vec3(0.0);
  vec3 clearcoatDirectLighting = vec3(0.0);
  vec3 directTransmission = vec3(0.0);

  for (uint i = 0u; i < pc.frameData.directionalLightCount; ++i) {
    DirectionalLightGpuData light =
        pc.frameData.directionalLightBuffer.lights[i];
    vec3 l = normalize(-directionalLightDirection(light));
    vec3 lr = directionalLightColor(light) * directionalLightIlluminance(light);
    accumulateSurfaceLightContribution(lr, l, sm, directDiffuse, directSpecular,
                                       directSheen, clearcoatDirectLighting);

    if (hasTransmission) {
      float sheenScale =
          sm.sheenWeight > 0.0
              ? computeSheenAlbedoScalingDirect(sm.sheenColor, sm.ndotv,
                                                max(dot(sm.nBase, l), 0.0),
                                                sm.sheenRoughness)
              : 1.0;
      vec3 tc = lr * getDirectTransmission(sm.nBase, sm.v, l, sm.alphaRoughness,
                                           sm.f0, sm.f90, sm.diffuseColor,
                                           transmissionIor);
      if ((featureMask & kMaterialFeatureVolume) != 0u) {
        tc = applyVolumeAttenuation(tc, transmissionRayLength, attenuationColor,
                                    attenuationDistance);
      }
      directTransmission += sheenScale * tc;
    }
  }

  for (uint i = 0u; i < pc.frameData.localLightCount; ++i) {
    LocalLightGpuData light = pc.frameData.localLightBuffer.lights[i];
    vec3 ptl = localLightPosition(light) - vtx.worldPos;
    float dsq = dot(ptl, ptl);
    if (dsq <= kEpsilon) {
      continue;
    }
    vec3 l = ptl * inversesqrt(dsq);
    float att = punctualRangeAttenuation(dsq, localLightRange(light));
    if (localLightType(light) == kLocalLightTypeSpot) {
      att *= spotAngularAttenuation(localLightDirection(light), ptl,
                                    localLightInnerCos(light),
                                    localLightOuterCos(light));
    }
    if (att <= 0.0) {
      continue;
    }
    vec3 lr = localLightColor(light) * localLightIntensity(light) * att;
    accumulateSurfaceLightContribution(lr, l, sm, directDiffuse, directSpecular,
                                       directSheen, clearcoatDirectLighting);

    if (hasTransmission) {
      float sheenScale =
          sm.sheenWeight > 0.0
              ? computeSheenAlbedoScalingDirect(sm.sheenColor, sm.ndotv,
                                                max(dot(sm.nBase, l), 0.0),
                                                sm.sheenRoughness)
              : 1.0;
      vec3 transmissionPtl = ptl - transmissionRay;
      vec3 tc = lr * getDirectTransmission(sm.nBase, sm.v, transmissionPtl,
                                           sm.alphaRoughness, sm.f0, sm.f90,
                                           sm.diffuseColor, transmissionIor);
      if ((featureMask & kMaterialFeatureVolume) != 0u) {
        tc = applyVolumeAttenuation(tc, transmissionRayLength, attenuationColor,
                                    attenuationDistance);
      }
      directTransmission += sheenScale * tc;
    }
  }

  // IBL ---------------------------------------------------------------
  IblResult ibl = evaluateIbl(sm);

  vec3 indirectTransmission = vec3(0.0);
  if (hasTransmission &&
      (pc.frameData.flags & kFrameDataFlagHasSceneColor) != 0u &&
      pc.frameData.sceneColorTexId != kInvalidTextureBindlessIndex) {
    indirectTransmission = getIndirectTransmission(
        sm.nBase, sm.v, sm.roughness, sm.diffuseColor, sm.f0, sm.f90,
        vtx.worldPos, modelMatrix, transmissionIor, thickness, attenuationColor,
        attenuationDistance);
  }

  // Mix transmission into the diffuse terms ---------------------------
  vec3 indirectDiffuseTerm =
      mix(ibl.iblDiffuse, indirectTransmission, transmissionFactor);
  vec3 directDiffuseTerm =
      mix(directDiffuse, directTransmission, transmissionFactor);

  // Composition -------------------------------------------------------
  vec3 indirectLighting =
      sm.clearcoatAttenuation *
          (ibl.iblSheen +
           ibl.indirectScale * (indirectDiffuseTerm + ibl.iblSpecular)) +
      ibl.clearcoatIblSpecular;
  if (ibl.hasIndirectLighting) {
    indirectLighting *= sm.ao;
  }

  vec3 directLighting = sm.clearcoatAttenuation *
                            (directSheen + directDiffuseTerm + directSpecular) +
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
