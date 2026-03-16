#include "common.sp"
#include "BRDF.sp"
#include "material_inputs.sp"

layout(location = 0) in PerVertex vtx;
layout(location = 10) flat in uint inInstanceId;

layout(location = 0) out vec4 out_FragColor;

const uint kAlphaModeOpaque = 0u;
const uint kAlphaModeMask = 1u;

float applyIorToRoughness(float roughness, float ior) {
  if (isIorCompatMode(ior)) {
    return roughness;
  }
  return roughness * clamp(ior * 2.0 - 2.0, 0.0, 1.0);
}

vec3 getAuthoredScale() {
  // Transmission uses the shared mesh push-constant layout from common.sp, so
  // the tessellation distance/factor slots carry imported local-space authored
  // scale for this pass instead of tessellation settings.
  return max(vec3(pc.tessNearDistance, pc.tessFarDistance, pc.tessMinFactor),
             vec3(1.0e-6));
}

vec3 srgbFromLinear(vec3 linearColor) {
  const bvec3 useLinearSegment =
      lessThanEqual(linearColor, vec3(0.0031308));
  const vec3 linearSegment = linearColor * 12.92;
  const vec3 nonlinearSegment =
      1.055 * pow(max(linearColor, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
  return mix(nonlinearSegment, linearSegment, useLinearSegment);
}

vec3 getVolumeTransmissionRay(vec3 n, vec3 v, float thickness, float ior,
                              mat4 modelMatrix) {
  vec3 refractionVector = refract(-v, n, 1.0 / max(ior, 1.0));
  if (dot(refractionVector, refractionVector) <= kBrdfEpsilon) {
    refractionVector = -v;
  }
  vec3 modelScale = vec3(length(modelMatrix[0].xyz), length(modelMatrix[1].xyz),
                         length(modelMatrix[2].xyz));
  return normalize(refractionVector) * thickness * modelScale *
         getAuthoredScale();
}

vec3 applyVolumeAttenuation(vec3 radiance, float transmissionDistance,
                            vec3 attenuationColor,
                            float attenuationDistance) {
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
  vec3 level0 =
      textureBindless2D(getSceneColorPyramidTexId(pc.frameData, 0u),
                        pc.frameData.sceneColorSamplerId, clampedUv)
          .rgb;
  vec3 level1 = level0;
  vec3 level2 = level0;

  uint level1TexId = getSceneColorPyramidTexId(pc.frameData, 1u);
  if (level1TexId != kInvalidTextureBindlessIndex) {
    level1 = textureBindless2D(level1TexId,
                               pc.frameData.sceneColorSamplerId, clampedUv)
                 .rgb;
  }
  uint level2TexId = getSceneColorPyramidTexId(pc.frameData, 2u);
  if (level2TexId != kInvalidTextureBindlessIndex) {
    level2 = textureBindless2D(level2TexId,
                               pc.frameData.sceneColorSamplerId, clampedUv)
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
  vec4 clipPos = pc.frameData.proj * pc.frameData.view * vec4(refractedRayExit, 1.0);
  if (clipPos.w <= kBrdfEpsilon) {
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
    vec2 brdfUv = clamp(vec2(ndotv, 1.0 - roughness * roughness), vec2(0.0),
                        vec2(1.0));
    vec2 brdf =
        textureBindless2D(pc.frameData.brdfLutTexId, 0u, brdfUv).rg;
    specularColor = f0 * brdf.x + f90 * brdf.y;
  } else {
    specularColor = fresnelSchlick(ndotv, f0);
  }

  return (vec3(1.0) - specularColor) * attenuatedColor * baseColor;
}

vec3 getDirectTransmission(vec3 n, vec3 v, vec3 pointToLight,
                           float alphaRoughness, vec3 f0, vec3 f90,
                           vec3 baseColor, float ior) {
  float transmissionRoughness = applyIorToRoughness(alphaRoughness, ior);

  vec3 l = normalize(pointToLight);
  vec3 lMirror = normalize(l + 2.0 * n * dot(-l, n));
  vec3 h = normalize(lMirror + v);

  float d = distributionGGX(clamp(dot(n, h), 0.0, 1.0), transmissionRoughness);
  vec3 f = specularReflection(clamp(dot(v, h), 0.0, 1.0), f0, f90);
  float vis = geometrySmith(clamp(dot(n, lMirror), 0.0, 1.0),
                            clamp(dot(n, v), 0.0, 1.0),
                            transmissionRoughness);
  return (vec3(1.0) - f) * baseColor * d * vis;
}

void main() {
  const MaterialGpuData material = pc.materialBuffer.materials[pc.materialIndex];

  const uint baseColorTexId =
      GET_TEXTURE_INDEX(material, kMaterialTextureSlotBaseColor);
  const uint metallicRoughnessTexId =
      GET_TEXTURE_INDEX(material, kMaterialTextureSlotMetallicRoughness);
  const uint normalTexId =
      GET_TEXTURE_INDEX(material, kMaterialTextureSlotNormal);
  const uint occlusionTexId =
      GET_TEXTURE_INDEX(material, kMaterialTextureSlotOcclusion);
  const uint emissiveTexId =
      GET_TEXTURE_INDEX(material, kMaterialTextureSlotEmissive);
  const uint clearcoatTexId =
      GET_TEXTURE_INDEX(material, kMaterialTextureSlotClearcoat);
  const uint clearcoatRoughnessTexId =
      GET_TEXTURE_INDEX(material, kMaterialTextureSlotClearcoatRoughness);
  const uint clearcoatNormalTexId =
      GET_TEXTURE_INDEX(material, kMaterialTextureSlotClearcoatNormal);
  const uint specularTexId =
      GET_TEXTURE_INDEX(material, kMaterialTextureSlotSpecular);
  const uint specularColorTexId =
      GET_TEXTURE_INDEX(material, kMaterialTextureSlotSpecularColor);
  const uint sheenColorTexId =
      GET_TEXTURE_INDEX(material, kMaterialTextureSlotSheenColor);
  const uint sheenRoughnessTexId =
      GET_TEXTURE_INDEX(material, kMaterialTextureSlotSheenRoughness);
  const uint transmissionTexId =
      GET_TEXTURE_INDEX(material, kMaterialTextureSlotTransmission);
  const uint thicknessTexId =
      GET_TEXTURE_INDEX(material, kMaterialTextureSlotThickness);
  const uint alphaMode = material.materialFlags.x;
  const uint featureMask = material.materialFlags.z;

  const uint baseColorSampler =
      GET_SAMPLER_INDEX(material, kMaterialTextureSlotBaseColor);
  const uint metallicRoughnessSampler =
      GET_SAMPLER_INDEX(material, kMaterialTextureSlotMetallicRoughness);
  const uint normalSampler =
      GET_SAMPLER_INDEX(material, kMaterialTextureSlotNormal);
  const uint occlusionSampler =
      GET_SAMPLER_INDEX(material, kMaterialTextureSlotOcclusion);
  const uint emissiveSampler =
      GET_SAMPLER_INDEX(material, kMaterialTextureSlotEmissive);
  const uint clearcoatSampler =
      GET_SAMPLER_INDEX(material, kMaterialTextureSlotClearcoat);
  const uint clearcoatRoughnessSampler =
      GET_SAMPLER_INDEX(material, kMaterialTextureSlotClearcoatRoughness);
  const uint clearcoatNormalSampler =
      GET_SAMPLER_INDEX(material, kMaterialTextureSlotClearcoatNormal);
  const uint specularSampler =
      GET_SAMPLER_INDEX(material, kMaterialTextureSlotSpecular);
  const uint specularColorSampler =
      GET_SAMPLER_INDEX(material, kMaterialTextureSlotSpecularColor);
  const uint sheenColorSampler =
      GET_SAMPLER_INDEX(material, kMaterialTextureSlotSheenColor);
  const uint sheenRoughnessSampler =
      GET_SAMPLER_INDEX(material, kMaterialTextureSlotSheenRoughness);
  const uint transmissionSampler =
      GET_SAMPLER_INDEX(material, kMaterialTextureSlotTransmission);
  const uint thicknessSampler =
      GET_SAMPLER_INDEX(material, kMaterialTextureSlotThickness);

  const vec2 uvBaseColor =
      transformedUv(material, vtx, kMaterialTextureSlotBaseColor);
  const vec2 uvMetallicRoughness =
      transformedUv(material, vtx, kMaterialTextureSlotMetallicRoughness);
  const vec2 uvNormal =
      transformedUv(material, vtx, kMaterialTextureSlotNormal);
  const vec2 uvOcclusion =
      transformedUv(material, vtx, kMaterialTextureSlotOcclusion);
  const vec2 uvEmissive =
      transformedUv(material, vtx, kMaterialTextureSlotEmissive);
  const vec2 uvClearcoat =
      transformedUv(material, vtx, kMaterialTextureSlotClearcoat);
  const vec2 uvClearcoatRoughness =
      transformedUv(material, vtx, kMaterialTextureSlotClearcoatRoughness);
  const vec2 uvClearcoatNormal =
      transformedUv(material, vtx, kMaterialTextureSlotClearcoatNormal);
  const vec2 uvSpecular =
      transformedUv(material, vtx, kMaterialTextureSlotSpecular);
  const vec2 uvSpecularColor =
      transformedUv(material, vtx, kMaterialTextureSlotSpecularColor);
  const vec2 uvSheenColor =
      transformedUv(material, vtx, kMaterialTextureSlotSheenColor);
  const vec2 uvSheenRoughness =
      transformedUv(material, vtx, kMaterialTextureSlotSheenRoughness);
  const vec2 uvTransmission =
      transformedUv(material, vtx, kMaterialTextureSlotTransmission);
  const vec2 uvThickness =
      transformedUv(material, vtx, kMaterialTextureSlotThickness);

  vec4 baseColor = material.baseColorFactor;
  if (baseColorTexId != kInvalidTextureBindlessIndex) {
    baseColor *=
        textureBindless2D(baseColorTexId, baseColorSampler, uvBaseColor);
  }

  const float alphaCutoff = material.metallicRoughnessOcclusionAlphaCutoff.w;
  if (alphaMode == kAlphaModeMask && baseColor.a < alphaCutoff) {
    discard;
  }

  vec4 mrSample = vec4(1.0);
  if (metallicRoughnessTexId != kInvalidTextureBindlessIndex) {
    mrSample = textureBindless2D(metallicRoughnessTexId,
                                 metallicRoughnessSampler, uvMetallicRoughness);
  }

  float metallic = saturate(material.metallicRoughnessOcclusionAlphaCutoff.x *
                            mrSample.b);
  float roughness = clamp(material.metallicRoughnessOcclusionAlphaCutoff.y *
                              mrSample.g,
                          kBrdfMinRoughness, 1.0);

  float occlusion = 1.0;
  if (occlusionTexId != kInvalidTextureBindlessIndex) {
    occlusion =
        textureBindless2D(occlusionTexId, occlusionSampler, uvOcclusion).r;
  } else if (metallicRoughnessTexId != kInvalidTextureBindlessIndex) {
    occlusion = mrSample.r;
  }
  float ao = mix(1.0, occlusion,
                 saturate(material.metallicRoughnessOcclusionAlphaCutoff.z));

  vec3 nGeom = normalize(vtx.worldNormal);
  if (!gl_FrontFacing) {
    nGeom *= -1.0;
  }

  bool hasClearcoat = (featureMask & kMaterialFeatureClearcoat) != 0u;
  float clearcoat = 0.0;
  float clearcoatRoughness = kBrdfMinRoughness;
  vec3 clearcoatF0 = vec3(0.04);
  vec3 clearcoatReflectance90 = vec3(1.0);
  vec3 clearcoatAttenuation = vec3(1.0);
  if (hasClearcoat) {
    clearcoat = saturate(material.sheenRoughnessClearcoatFactors.y);
    if (clearcoatTexId != kInvalidTextureBindlessIndex) {
      clearcoat *=
          textureBindless2D(clearcoatTexId, clearcoatSampler, uvClearcoat).r;
    }
    clearcoatRoughness =
        clamp(material.sheenRoughnessClearcoatFactors.z, kBrdfMinRoughness, 1.0);
    if (clearcoatRoughnessTexId != kInvalidTextureBindlessIndex) {
      clearcoatRoughness =
          clamp(clearcoatRoughness *
                    textureBindless2D(clearcoatRoughnessTexId,
                                      clearcoatRoughnessSampler,
                                      uvClearcoatRoughness)
                        .g,
                kBrdfMinRoughness, 1.0);
    }
  }

  vec3 nBase = nGeom;
  if (normalTexId != kInvalidTextureBindlessIndex) {
    vec3 normalTexel =
        textureBindless2D(normalTexId, normalSampler, uvNormal).xyz * 2.0 - 1.0;
    normalTexel.xy *= material.emissiveFactorNormalScale.w;
    float normalTexelLen = length(normalTexel);
    if (normalTexelLen > kBrdfEpsilon) {
      normalTexel /= normalTexelLen;
    } else {
      normalTexel = vec3(0.0, 0.0, 1.0);
    }
    nBase = applyNormalMap(nBase, vtx.worldPos, uvNormal, normalTexel);
  }

  vec3 nClearcoat = nGeom;
  if (hasClearcoat && clearcoatNormalTexId != kInvalidTextureBindlessIndex) {
    vec3 clearcoatNormalTexel =
        textureBindless2D(clearcoatNormalTexId, clearcoatNormalSampler,
                          uvClearcoatNormal)
            .xyz *
        2.0 - 1.0;
    clearcoatNormalTexel.xy *= material.sheenRoughnessClearcoatFactors.w;
    float clearcoatNormalTexelLen = length(clearcoatNormalTexel);
    if (clearcoatNormalTexelLen > kBrdfEpsilon) {
      clearcoatNormalTexel /= clearcoatNormalTexelLen;
    } else {
      clearcoatNormalTexel = vec3(0.0, 0.0, 1.0);
    }
    vec3 perturbedClearcoatNormal =
        applyNormalMap(nClearcoat, vtx.worldPos, uvClearcoatNormal,
                       clearcoatNormalTexel);
    float clearcoatNormalBlend =
        clamp(sqrt(clearcoatRoughness), kBrdfMinRoughness, 1.0);
    nClearcoat =
        normalize(mix(nClearcoat, perturbedClearcoatNormal, clearcoatNormalBlend));
  }

  vec3 emissive = material.emissiveFactorNormalScale.xyz;
  if (emissiveTexId != kInvalidTextureBindlessIndex) {
    emissive *=
        textureBindless2D(emissiveTexId, emissiveSampler, uvEmissive).rgb;
  }

  float transmissionFactor = material.transmissionThicknessIorPadding.x;
  if (transmissionTexId != kInvalidTextureBindlessIndex) {
    transmissionFactor *=
        textureBindless2D(transmissionTexId, transmissionSampler, uvTransmission)
            .r;
  }
  transmissionFactor = saturate(transmissionFactor);

  float thickness = max(material.transmissionThicknessIorPadding.y, 0.0);
  if (thicknessTexId != kInvalidTextureBindlessIndex) {
    thickness *=
        textureBindless2D(thicknessTexId, thicknessSampler, uvThickness).g;
  }
  thickness = max(thickness, 0.0);

  float ior = material.transmissionThicknessIorPadding.z;
  bool iorCompatMode = isIorCompatMode(ior);
  vec3 attenuationColor = clamp(material.attenuationColorDistance.rgb, vec3(0.0),
                                vec3(1.0));
  float attenuationDistance = max(material.attenuationColorDistance.w, 0.0);

  vec3 v = normalize(pc.frameData.cameraPos.xyz - vtx.worldPos);
  const bool hasTransmission = transmissionFactor > 0.0 && !iorCompatMode;
  mat4 modelMatrix = mat4(1.0);
  if (hasTransmission) {
    modelMatrix = pc.instanceMatrices.matrices[inInstanceId];
  }
  float ndotv = max(dot(nBase, v), 0.001);
  float clearcoatNdotV = max(dot(nClearcoat, v), 0.001);

  float specularWeight = sampleMaterialSpecularWeight(
      material, uvSpecular, specularTexId, specularSampler);
  vec3 specularColor = sampleMaterialSpecularColor(
      material, uvSpecularColor, specularColorTexId, specularColorSampler);
  vec3 dielectricF0 = vec3(0.0);
  vec3 dielectricF90 = vec3(0.0);
  computeDielectricSpecularTerms(ior, specularColor, specularWeight,
                                 dielectricF0, dielectricF90);
  vec3 f0 = mix(dielectricF0, baseColor.rgb, metallic);
  vec3 f90 = mix(dielectricF90, vec3(1.0), metallic);
  vec3 diffuseColor = mix(baseColor.rgb, vec3(0.0), metallic);
  float alphaRoughness = roughness * roughness;
  float sheenWeight =
      ((featureMask & kMaterialFeatureSheen) != 0u)
          ? saturate(material.sheenColorFactorWeight.w)
          : 0.0;
  float sheenRoughness = clamp(material.sheenRoughnessClearcoatFactors.x,
                               kBrdfMinRoughness, 1.0);
  vec3 sheenColor = material.sheenColorFactorWeight.xyz;
  if (sheenColorTexId != kInvalidTextureBindlessIndex) {
    sheenColor *=
        textureBindless2D(sheenColorTexId, sheenColorSampler, uvSheenColor).rgb;
  }
  if (sheenRoughnessTexId != kInvalidTextureBindlessIndex) {
    sheenRoughness =
        clamp(sheenRoughness *
                  textureBindless2D(sheenRoughnessTexId, sheenRoughnessSampler,
                                    uvSheenRoughness)
                      .a,
              kBrdfMinRoughness, 1.0);
  }
  if (hasClearcoat) {
    vec3 clearcoatLayerF = fresnelSchlick(clearcoatNdotV, clearcoatF0);
    clearcoatAttenuation = vec3(1.0) - clearcoat * clearcoatLayerF;
  }

  const vec3 lightPos = vec3(0.0, 0.0, -5.0);
  const vec3 lightColor = vec3(1.0);
  vec3 l = normalize(lightPos - vtx.worldPos);
  float ndotl = max(dot(nBase, l), 0.0);
  float clearcoatNdotL = max(dot(nClearcoat, l), 0.0);
  vec3 directDiffuse = vec3(0.0);
  vec3 directSpecular = vec3(0.0);
  vec3 directSheen = vec3(0.0);
  vec3 clearcoatDirectLighting = vec3(0.0);
  vec3 directTransmission = vec3(0.0);
  vec3 halfVector = v + l;
  float halfLenSq = dot(halfVector, halfVector);
  if (ndotl > 0.0 && halfLenSq > kBrdfEpsilon) {
    vec3 h = halfVector * inversesqrt(halfLenSq);
    float ndoth = max(dot(nBase, h), 0.0);
    float ldoth = max(dot(l, h), 0.0);
    float vdoth = max(dot(v, h), 0.0);

    vec3 f = specularReflection(vdoth, f0, f90);
    float g = geometryOcclusion(ndotl, ndotv, alphaRoughness);
    float d = microfacetDistribution(ndoth, alphaRoughness);
    directDiffuse = ndotl * lightColor *
                    ((1.0 - max3(f)) *
                     diffuseBurley(diffuseColor, ndotl, ndotv, ldoth,
                                   alphaRoughness));
    directSpecular =
        ndotl * lightColor * (f * g * d / max(4.0 * ndotl * ndotv, kBrdfEpsilon));
    directSheen =
        computeDirectSheen(sheenColor, sheenWeight, sheenRoughness, ndotl,
                           ndotv, ndoth, lightColor);
  }
  if (hasClearcoat && clearcoat > 0.0 && clearcoatNdotL > 0.0 &&
      halfLenSq > kBrdfEpsilon) {
    vec3 h = halfVector * inversesqrt(halfLenSq);
    float clearcoatNdotH = max(dot(nClearcoat, h), 0.0);
    float clearcoatVdotH = max(dot(v, h), 0.0);
    float clearcoatAlphaRoughness = clearcoatRoughness * clearcoatRoughness;
    vec3 clearcoatF = specularReflection(clearcoatVdotH, clearcoatF0,
                                         clearcoatReflectance90);
    float clearcoatG =
        geometryOcclusion(clearcoatNdotL, clearcoatNdotV, clearcoatAlphaRoughness);
    float clearcoatD =
        microfacetDistribution(clearcoatNdotH, clearcoatAlphaRoughness);
    vec3 clearcoatSpecular =
        clearcoatF * clearcoatG * clearcoatD /
        max(4.0 * clearcoatNdotL * clearcoatNdotV, kBrdfEpsilon);
    clearcoatDirectLighting =
        clearcoat * clearcoatNdotL * lightColor * clearcoatSpecular;
  }

  if (hasTransmission) {
    vec3 transmissionRay =
        getVolumeTransmissionRay(nBase, v, thickness, ior, modelMatrix);
    vec3 pointToLight = lightPos - vtx.worldPos - transmissionRay;
    directTransmission =
        lightColor *
        getDirectTransmission(nBase, v, pointToLight, alphaRoughness, f0, f90,
                              diffuseColor, ior);
    if ((featureMask & kMaterialFeatureVolume) != 0u) {
      directTransmission = applyVolumeAttenuation(
          directTransmission, length(transmissionRay), attenuationColor,
          attenuationDistance);
    }
  }

  vec3 baseBrdfLutSample = vec3(0.0);
  vec3 sheenBrdfLutSample = vec3(0.0);
  bool hasBrdfLut = (pc.frameData.flags & kFrameDataFlagHasBrdfLut) != 0u &&
                    pc.frameData.brdfLutTexId != kInvalidTextureBindlessIndex;
  if (hasBrdfLut) {
    vec2 baseBrdfUv = clamp(vec2(ndotv, 1.0 - roughness * roughness),
                            vec2(0.0), vec2(1.0));
    baseBrdfLutSample =
        textureBindless2D(pc.frameData.brdfLutTexId, 0u, baseBrdfUv).rgb;
    if (sheenWeight > 0.0) {
      vec2 sheenBrdfUv = clamp(vec2(ndotv, 1.0 - sheenRoughness * sheenRoughness),
                               vec2(0.0), vec2(1.0));
      sheenBrdfLutSample =
          textureBindless2D(pc.frameData.brdfLutTexId, 0u, sheenBrdfUv).rgb;
    }
  }

  float directScale = 1.0;
  float indirectScale = 1.0;
  if (sheenWeight > 0.0) {
    directScale = computeSheenAlbedoScalingDirect(
        sheenColor, ndotv, ndotl, sheenRoughness);
    if (hasBrdfLut) {
      indirectScale = computeSheenAlbedoScalingIndirect(
          sheenColor, sheenWeight, sheenBrdfLutSample);
    }
  }

  vec3 iblDiffuse = vec3(0.0);
  vec3 iblSpecular = vec3(0.0);
  vec3 iblSheen = vec3(0.0);
  vec3 clearcoatIblSpecular = vec3(0.0);
  vec3 indirectTransmission = vec3(0.0);
  bool hasIndirectLighting = false;

  if ((pc.frameData.flags & kFrameDataFlagHasIblDiffuse) != 0u &&
      pc.frameData.irradianceTexId != kInvalidTextureBindlessIndex) {
    vec3 irradiance =
        textureBindlessCube(pc.frameData.irradianceTexId,
                            pc.frameData.cubemapSamplerId, nBase)
            .rgb;
    if (iorCompatMode) {
      iblDiffuse = vec3(0.0);
    } else if (hasBrdfLut) {
      iblDiffuse =
          computeIblDiffuse(diffuseColor, f0, f90, irradiance, baseBrdfLutSample);
    } else {
      iblDiffuse =
          diffuseColor * (1.0 - max3(fresnelSchlick(ndotv, f0))) * irradiance;
    }
    hasIndirectLighting = true;
  }

  if ((pc.frameData.flags & kFrameDataFlagHasIblSpecular) != 0u &&
      pc.frameData.prefilteredGgxTexId != kInvalidTextureBindlessIndex) {
    vec3 r = reflect(-v, nBase);
    if (hasBrdfLut) {
      float mipCount =
          float(textureBindlessQueryLevelsCube(pc.frameData.prefilteredGgxTexId));
      float lod = roughness * max(mipCount - 1.0, 0.0);
      vec3 prefiltered =
          textureBindlessCubeLod(pc.frameData.prefilteredGgxTexId,
                                 pc.frameData.cubemapSamplerId, r, lod)
              .rgb;
      iblSpecular =
          computeIblSpecular(f0, f90, roughness, ndotv, prefiltered,
                             baseBrdfLutSample);
    } else {
      iblSpecular =
          textureBindlessCube(pc.frameData.prefilteredGgxTexId,
                              pc.frameData.cubemapSamplerId, r)
              .rgb *
          fresnelSchlick(ndotv, f0);
    }
    hasIndirectLighting = true;
  }

  if ((pc.frameData.flags & kFrameDataFlagHasIblSheen) != 0u &&
      pc.frameData.prefilteredCharlieTexId != kInvalidTextureBindlessIndex &&
      hasBrdfLut && sheenWeight > 0.0) {
    vec3 r = reflect(-v, nBase);
    float mipCount = float(
        textureBindlessQueryLevelsCube(pc.frameData.prefilteredCharlieTexId));
    float lod = sheenRoughness * max(mipCount - 1.0, 0.0);
    vec3 sheenEnv =
        textureBindlessCubeLod(pc.frameData.prefilteredCharlieTexId,
                               pc.frameData.cubemapSamplerId, r, lod)
            .rgb;
    iblSheen = computeIblSheen(sheenColor, sheenWeight, sheenEnv,
                               sheenBrdfLutSample);
    hasIndirectLighting = true;
  }

  if (hasClearcoat && clearcoat > 0.0 &&
      (pc.frameData.flags & kFrameDataFlagHasIblSpecular) != 0u &&
      pc.frameData.prefilteredGgxTexId != kInvalidTextureBindlessIndex) {
    vec3 clearcoatR = reflect(-v, nClearcoat);
    if (hasBrdfLut) {
      vec2 clearcoatBrdfUv =
          clamp(vec2(clearcoatNdotV, 1.0 - clearcoatRoughness * clearcoatRoughness),
                vec2(0.0), vec2(1.0));
      vec3 clearcoatBrdfLutSample =
          textureBindless2D(pc.frameData.brdfLutTexId, 0u, clearcoatBrdfUv).rgb;
      float mipCount =
          float(textureBindlessQueryLevelsCube(pc.frameData.prefilteredGgxTexId));
      float lod = clearcoatRoughness * max(mipCount - 1.0, 0.0);
      vec3 prefiltered =
          textureBindlessCubeLod(pc.frameData.prefilteredGgxTexId,
                                 pc.frameData.cubemapSamplerId, clearcoatR, lod)
              .rgb;
      clearcoatIblSpecular =
          clearcoat *
          computeIblSpecular(clearcoatF0, clearcoatReflectance90,
                             clearcoatRoughness, clearcoatNdotV,
                             prefiltered, clearcoatBrdfLutSample);
    } else {
      clearcoatIblSpecular =
          clearcoat *
          textureBindlessCube(pc.frameData.prefilteredGgxTexId,
                              pc.frameData.cubemapSamplerId, clearcoatR)
              .rgb *
          fresnelSchlick(clearcoatNdotV, clearcoatF0);
    }
    hasIndirectLighting = true;
  }

  if (hasTransmission &&
      (pc.frameData.flags & kFrameDataFlagHasSceneColor) != 0u &&
      pc.frameData.sceneColorTexId != kInvalidTextureBindlessIndex) {
    indirectTransmission = getIndirectTransmission(
        nBase, v, roughness, diffuseColor, f0, f90, vtx.worldPos, modelMatrix,
        ior, thickness, attenuationColor, attenuationDistance);
  }

  vec3 indirectDiffuseTerm = mix(iblDiffuse, indirectTransmission,
                                 transmissionFactor);
  vec3 directDiffuseTerm =
      mix(directDiffuse, directTransmission, transmissionFactor);

  vec3 indirectLighting =
      clearcoatAttenuation *
          (iblSheen +
           indirectScale * (indirectDiffuseTerm + iblSpecular)) +
      clearcoatIblSpecular;
  if (hasIndirectLighting) {
    indirectLighting *= ao;
  }
  vec3 directLighting =
      clearcoatAttenuation *
          (directSheen + directScale * (directDiffuseTerm + directSpecular)) +
      clearcoatDirectLighting;
  vec3 color =
      directLighting + indirectLighting + clearcoatAttenuation * emissive;
  color = max(color, vec3(0.0));
  if ((pc.frameData.flags & kFrameDataFlagOutputLinearToSrgb) != 0u) {
    color = srgbFromLinear(color);
  }

  float outAlpha = (alphaMode == kAlphaModeOpaque) ? 1.0 : baseColor.a;
  out_FragColor = vec4(color, outAlpha);
}
