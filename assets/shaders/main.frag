#include "common.sp"
#include "BRDF.sp"

layout(location = 0) in PerVertex vtx;

layout(location = 0) out vec4 out_FragColor;

const uint kAlphaModeOpaque = 0u;
const uint kAlphaModeMask = 1u;

vec2 selectUv(vec2 uv0, vec2 uv1, uint uvSet) {
  return (uvSet == 1u) ? uv1 : uv0;
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
  const uint sheenColorTexId =
      GET_TEXTURE_INDEX(material, kMaterialTextureSlotSheenColor);
  const uint sheenRoughnessTexId =
      GET_TEXTURE_INDEX(material, kMaterialTextureSlotSheenRoughness);
  const uint alphaMode = material.materialFlags.x;
  const uint featureMask = material.materialFlags.z;

  const uint baseColorUvSet =
      GET_UV_SET(material, kMaterialTextureSlotBaseColor);
  const uint metallicRoughnessUvSet =
      GET_UV_SET(material, kMaterialTextureSlotMetallicRoughness);
  const uint normalUvSet = GET_UV_SET(material, kMaterialTextureSlotNormal);
  const uint occlusionUvSet =
      GET_UV_SET(material, kMaterialTextureSlotOcclusion);
  const uint emissiveUvSet =
      GET_UV_SET(material, kMaterialTextureSlotEmissive);
  const uint clearcoatUvSet =
      GET_UV_SET(material, kMaterialTextureSlotClearcoat);
  const uint clearcoatRoughnessUvSet =
      GET_UV_SET(material, kMaterialTextureSlotClearcoatRoughness);
  const uint clearcoatNormalUvSet =
      GET_UV_SET(material, kMaterialTextureSlotClearcoatNormal);
  const uint sheenColorUvSet =
      GET_UV_SET(material, kMaterialTextureSlotSheenColor);
  const uint sheenRoughnessUvSet =
      GET_UV_SET(material, kMaterialTextureSlotSheenRoughness);

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
  const uint sheenColorSampler =
      GET_SAMPLER_INDEX(material, kMaterialTextureSlotSheenColor);
  const uint sheenRoughnessSampler =
      GET_SAMPLER_INDEX(material, kMaterialTextureSlotSheenRoughness);

  const vec2 uvBaseColor = applyTextureTransform(
      selectUv(vtx.uv0, vtx.uv1, baseColorUvSet),
      material.textureTransformOffsetScale[kMaterialTextureSlotBaseColor],
      material.textureTransformRotation[kMaterialTextureSlotBaseColor]);
  const vec2 uvMetallicRoughness = applyTextureTransform(
      selectUv(vtx.uv0, vtx.uv1, metallicRoughnessUvSet),
      material.textureTransformOffsetScale[kMaterialTextureSlotMetallicRoughness],
      material.textureTransformRotation[kMaterialTextureSlotMetallicRoughness]);
  const vec2 uvNormal = applyTextureTransform(
      selectUv(vtx.uv0, vtx.uv1, normalUvSet),
      material.textureTransformOffsetScale[kMaterialTextureSlotNormal],
      material.textureTransformRotation[kMaterialTextureSlotNormal]);
  const vec2 uvOcclusion = applyTextureTransform(
      selectUv(vtx.uv0, vtx.uv1, occlusionUvSet),
      material.textureTransformOffsetScale[kMaterialTextureSlotOcclusion],
      material.textureTransformRotation[kMaterialTextureSlotOcclusion]);
  const vec2 uvEmissive = applyTextureTransform(
      selectUv(vtx.uv0, vtx.uv1, emissiveUvSet),
      material.textureTransformOffsetScale[kMaterialTextureSlotEmissive],
      material.textureTransformRotation[kMaterialTextureSlotEmissive]);
  const vec2 uvClearcoat = applyTextureTransform(
      selectUv(vtx.uv0, vtx.uv1, clearcoatUvSet),
      material.textureTransformOffsetScale[kMaterialTextureSlotClearcoat],
      material.textureTransformRotation[kMaterialTextureSlotClearcoat]);
  const vec2 uvClearcoatRoughness = applyTextureTransform(
      selectUv(vtx.uv0, vtx.uv1, clearcoatRoughnessUvSet),
      material.textureTransformOffsetScale[kMaterialTextureSlotClearcoatRoughness],
      material.textureTransformRotation[kMaterialTextureSlotClearcoatRoughness]);
  const vec2 uvClearcoatNormal = applyTextureTransform(
      selectUv(vtx.uv0, vtx.uv1, clearcoatNormalUvSet),
      material.textureTransformOffsetScale[kMaterialTextureSlotClearcoatNormal],
      material.textureTransformRotation[kMaterialTextureSlotClearcoatNormal]);
  const vec2 uvSheenColor = applyTextureTransform(
      selectUv(vtx.uv0, vtx.uv1, sheenColorUvSet),
      material.textureTransformOffsetScale[kMaterialTextureSlotSheenColor],
      material.textureTransformRotation[kMaterialTextureSlotSheenColor]);
  const vec2 uvSheenRoughness = applyTextureTransform(
      selectUv(vtx.uv0, vtx.uv1, sheenRoughnessUvSet),
      material.textureTransformOffsetScale[kMaterialTextureSlotSheenRoughness],
      material.textureTransformRotation[kMaterialTextureSlotSheenRoughness]);

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
      clearcoat *= textureBindless2D(clearcoatTexId, clearcoatSampler, uvClearcoat).r;
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
    // Derivative-based TBN avoids reliance on imported tangent sign
    // conventions.
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
    // Very glossy clearcoat turns harsh normal-map distortion into a plastic
    // shell without specular AA. Bias back toward the geometric normal so the
    // reflection remains coherent.
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

  vec3 v = normalize(pc.frameData.cameraPos.xyz - vtx.worldPos);
  float ndotv = max(dot(nBase, v), 0.001);
  float clearcoatNdotV = max(dot(nClearcoat, v), 0.001);

  vec3 f0 = mix(vec3(0.04), baseColor.rgb, metallic);
  vec3 diffuseColor = mix(baseColor.rgb, vec3(0.0), metallic);
  float alphaRoughness = roughness * roughness;
  float reflectance = max(max(f0.r, f0.g), f0.b);
  vec3 reflectance90 = vec3(clamp(reflectance * 25.0, 0.0, 1.0));
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
  vec3 baseDirectLighting = vec3(0.0);
  vec3 directSheen = vec3(0.0);
  vec3 clearcoatDirectLighting = vec3(0.0);
  vec3 halfVector = v + l;
  float halfLenSq = dot(halfVector, halfVector);
  if (ndotl > 0.0 && halfLenSq > kBrdfEpsilon) {
    vec3 h = halfVector * inversesqrt(halfLenSq);
    float ndoth = max(dot(nBase, h), 0.0);
    float ldoth = max(dot(l, h), 0.0);
    float vdoth = max(dot(v, h), 0.0);

    vec3 f = specularReflection(vdoth, f0, reflectance90);
    float g = geometryOcclusion(ndotl, ndotv, alphaRoughness);
    float d = microfacetDistribution(ndoth, alphaRoughness);
    vec3 diffuse = (1.0 - f) *
                   diffuseBurley(diffuseColor, ndotl, ndotv, ldoth,
                                 alphaRoughness);
    vec3 specular = f * g * d / max(4.0 * ndotl * ndotv, kBrdfEpsilon);
    baseDirectLighting = ndotl * lightColor * (diffuse + specular);
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

  vec3 baseBrdfLutSample = vec3(0.0);
  vec3 sheenBrdfLutSample = vec3(0.0);
  bool hasBrdfLut = (pc.frameData.flags & kFrameDataFlagHasBrdfLut) != 0u &&
                    pc.frameData.brdfLutTexId != kInvalidTextureBindlessIndex;
  if (hasBrdfLut) {
    vec2 baseBrdfUv = clamp(vec2(ndotv, 1.0 - roughness * roughness),
                            vec2(0.0), vec2(1.0));
    baseBrdfLutSample =
        textureBindless2D(pc.frameData.brdfLutTexId, 0, baseBrdfUv).rgb;
    if (sheenWeight > 0.0) {
      vec2 sheenBrdfUv = clamp(vec2(ndotv, 1.0 - sheenRoughness * sheenRoughness),
                               vec2(0.0), vec2(1.0));
      sheenBrdfLutSample =
          textureBindless2D(pc.frameData.brdfLutTexId, 0, sheenBrdfUv).rgb;
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
  bool hasIndirectLighting = false;
  if ((pc.frameData.flags & kFrameDataFlagHasIblDiffuse) != 0u &&
      pc.frameData.irradianceTexId != kInvalidTextureBindlessIndex) {
    vec3 irradiance =
        textureBindlessCube(pc.frameData.irradianceTexId,
                            pc.frameData.cubemapSamplerId, nBase)
            .rgb;
    if (hasBrdfLut) {
      iblDiffuse = computeIblDiffuse(diffuseColor, f0, roughness, ndotv,
                                     irradiance, baseBrdfLutSample);
    } else {
      iblDiffuse = diffuseColor * irradiance;
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
          computeIblSpecular(f0, roughness, ndotv, prefiltered,
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
          textureBindless2D(pc.frameData.brdfLutTexId, 0, clearcoatBrdfUv).rgb;
      float mipCount =
          float(textureBindlessQueryLevelsCube(pc.frameData.prefilteredGgxTexId));
      float lod = clearcoatRoughness * max(mipCount - 1.0, 0.0);
      vec3 prefiltered =
          textureBindlessCubeLod(pc.frameData.prefilteredGgxTexId,
                                 pc.frameData.cubemapSamplerId, clearcoatR, lod)
              .rgb;
      clearcoatIblSpecular =
          clearcoat *
          computeIblSpecular(clearcoatF0, clearcoatRoughness, clearcoatNdotV,
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

  vec3 indirectLighting =
      clearcoatAttenuation *
          (iblSheen + indirectScale * (iblDiffuse + iblSpecular)) +
      clearcoatIblSpecular;
  if (hasIndirectLighting) {
    indirectLighting *= ao;
  }
  vec3 directLighting =
      clearcoatAttenuation * (directSheen + directScale * baseDirectLighting) +
      clearcoatDirectLighting;
  vec3 color =
      directLighting + indirectLighting + clearcoatAttenuation * emissive;
  color = max(color, vec3(0.0));
  if ((pc.frameData.flags & kFrameDataFlagOutputLinearToSrgb) != 0u) {
    color = pow(color, vec3(1.0 / 2.2));
  }

  float outAlpha = (alphaMode == kAlphaModeOpaque) ? 1.0 : baseColor.a;
  out_FragColor = vec4(color, outAlpha);
}
