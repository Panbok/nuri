// Shared material evaluation and lighting for the opaque and transmissive
// fragment shader passes.  Include after common.sp, BRDF.sp, and
// material_inputs.sp.

const uint kAlphaModeOpaque = 0u;
const uint kAlphaModeMask   = 1u;

float applySpecularAARoughnessBias(float roughness, vec3 shadingNormal) {
  vec3 dndx = dFdx(shadingNormal);
  vec3 dndy = dFdy(shadingNormal);
  float normalVariance =
      clamp(0.5 * (dot(dndx, dndx) + dot(dndy, dndy)), 0.0, 1.0);
  float alpha = max(roughness * roughness, kBrdfMinRoughness);
  alpha = clamp(alpha + 0.35 * normalVariance, kBrdfMinRoughness, 1.0);
  return clamp(sqrt(alpha), kBrdfMinRoughness, 1.0);
}

// Fully evaluated material state at a fragment.
struct ShadedMaterial {
  vec4  baseColor;
  vec3  diffuseColor;
  vec3  f0;
  vec3  f90;
  float metallic;
  float roughness;
  float alphaRoughness;
  float ao;
  float ior;
  bool  iorCompatMode;
  // Normals
  vec3  nGeom;
  vec3  nBase;
  vec3  nClearcoat;
  // View direction
  vec3  v;
  float ndotv;
  float clearcoatNdotV;
  // Clearcoat
  bool  hasClearcoat;
  float clearcoat;
  float clearcoatRoughness;
  vec3  clearcoatF0;
  vec3  clearcoatReflectance90;
  vec3  clearcoatAttenuation;
  // Sheen
  float sheenWeight;
  float sheenRoughness;
  vec3  sheenColor;
  // Emissive
  vec3  emissive;
};

// Evaluate all material textures and BRDF parameters for a fragment.
// Does not include the alpha-mask discard check or transmission-specific fields.
ShadedMaterial evaluateMaterial(MaterialGpuData material, PerVertex vtx) {
  ShadedMaterial sm;

  const uint featureMask = material.materialFlags.z;
  const uint workflow    = material.materialFlags.w;
  const uint matSampler  = pc.frameData.materialSamplerId;

  // Texture IDs -------------------------------------------------------
  const uint baseColorTexId          = GET_TEXTURE_INDEX(material, kMaterialTextureSlotBaseColor);
  const uint metallicRoughnessTexId  = GET_TEXTURE_INDEX(material, kMaterialTextureSlotMetallicRoughness);
  const uint normalTexId             = GET_TEXTURE_INDEX(material, kMaterialTextureSlotNormal);
  const uint occlusionTexId          = GET_TEXTURE_INDEX(material, kMaterialTextureSlotOcclusion);
  const uint emissiveTexId           = GET_TEXTURE_INDEX(material, kMaterialTextureSlotEmissive);
  const uint clearcoatTexId          = GET_TEXTURE_INDEX(material, kMaterialTextureSlotClearcoat);
  const uint clearcoatRoughnessTexId = GET_TEXTURE_INDEX(material, kMaterialTextureSlotClearcoatRoughness);
  const uint clearcoatNormalTexId    = GET_TEXTURE_INDEX(material, kMaterialTextureSlotClearcoatNormal);
  const uint specularTexId           = GET_TEXTURE_INDEX(material, kMaterialTextureSlotSpecular);
  const uint specularColorTexId      = GET_TEXTURE_INDEX(material, kMaterialTextureSlotSpecularColor);
  const uint sheenColorTexId         = GET_TEXTURE_INDEX(material, kMaterialTextureSlotSheenColor);
  const uint sheenRoughnessTexId     = GET_TEXTURE_INDEX(material, kMaterialTextureSlotSheenRoughness);

  // UV transforms -----------------------------------------------------
  const vec2 uvBaseColor          = transformedUv(material, vtx, kMaterialTextureSlotBaseColor);
  const vec2 uvMetallicRoughness  = transformedUv(material, vtx, kMaterialTextureSlotMetallicRoughness);
  const vec2 uvNormal             = transformedUv(material, vtx, kMaterialTextureSlotNormal);
  const vec2 uvOcclusion          = transformedUv(material, vtx, kMaterialTextureSlotOcclusion);
  const vec2 uvEmissive           = transformedUv(material, vtx, kMaterialTextureSlotEmissive);
  const vec2 uvClearcoat          = transformedUv(material, vtx, kMaterialTextureSlotClearcoat);
  const vec2 uvClearcoatRoughness = transformedUv(material, vtx, kMaterialTextureSlotClearcoatRoughness);
  const vec2 uvClearcoatNormal    = transformedUv(material, vtx, kMaterialTextureSlotClearcoatNormal);
  const vec2 uvSpecular           = transformedUv(material, vtx, kMaterialTextureSlotSpecular);
  const vec2 uvSpecularColor      = transformedUv(material, vtx, kMaterialTextureSlotSpecularColor);
  const vec2 uvSheenColor         = transformedUv(material, vtx, kMaterialTextureSlotSheenColor);
  const vec2 uvSheenRoughness     = transformedUv(material, vtx, kMaterialTextureSlotSheenRoughness);

  // Base color --------------------------------------------------------
  sm.baseColor = material.baseColorFactor;
  if (baseColorTexId != kInvalidTextureBindlessIndex) {
    sm.baseColor *= textureBindless2D(baseColorTexId, matSampler, uvBaseColor);
  }

  // Metallic/roughness ------------------------------------------------
  vec4 mrSample = vec4(1.0);
  if (metallicRoughnessTexId != kInvalidTextureBindlessIndex) {
    mrSample = textureBindless2D(metallicRoughnessTexId, matSampler, uvMetallicRoughness);
  }
  sm.metallic     = 0.0;
  sm.roughness    = kBrdfMinRoughness;
  sm.f0           = vec3(0.0);
  sm.f90          = vec3(0.0);
  sm.diffuseColor = vec3(0.0);
  sm.ior = material.transmissionThicknessIorPadding.z;
  decodeMaterialBaseWorkflow(
      material, workflow, sm.baseColor, mrSample,
      uvSpecular, specularTexId, matSampler,
      uvSpecularColor, specularColorTexId, matSampler,
      sm.ior, sm.metallic, sm.roughness, sm.f0, sm.f90, sm.diffuseColor);

  // Occlusion ---------------------------------------------------------
  float occlusion = sampleMaterialOcclusion(
      material, workflow, mrSample, metallicRoughnessTexId,
      uvOcclusion, occlusionTexId, matSampler);
  sm.ao = mix(1.0, occlusion,
              saturate(material.metallicRoughnessOcclusionAlphaCutoff.z));

  // Geometric normal (front-face aware) --------------------------------
  sm.nGeom = normalize(vtx.worldNormal);
  if (!gl_FrontFacing) { sm.nGeom *= -1.0; }

  // Clearcoat factors -------------------------------------------------
  sm.hasClearcoat          = (featureMask & kMaterialFeatureClearcoat) != 0u;
  sm.clearcoat             = 0.0;
  sm.clearcoatRoughness    = kBrdfMinRoughness;
  sm.clearcoatF0           = vec3(0.04);
  sm.clearcoatReflectance90 = vec3(1.0);
  sm.clearcoatAttenuation  = vec3(1.0);
  if (sm.hasClearcoat) {
    sm.clearcoat = saturate(material.sheenRoughnessClearcoatFactors.y);
    if (clearcoatTexId != kInvalidTextureBindlessIndex) {
      sm.clearcoat *=
          textureBindless2D(clearcoatTexId, matSampler, uvClearcoat).r;
    }
    sm.clearcoatRoughness =
        clamp(material.sheenRoughnessClearcoatFactors.z, kBrdfMinRoughness, 1.0);
    if (clearcoatRoughnessTexId != kInvalidTextureBindlessIndex) {
      sm.clearcoatRoughness = clamp(
          sm.clearcoatRoughness *
              textureBindless2D(clearcoatRoughnessTexId, matSampler,
                                uvClearcoatRoughness).g,
          kBrdfMinRoughness, 1.0);
    }
  }

  // Surface normal with normal map ------------------------------------
  sm.nBase = sm.nGeom;
  if (normalTexId != kInvalidTextureBindlessIndex) {
    vec3 n = textureBindless2D(normalTexId, matSampler, uvNormal).xyz * 2.0 - 1.0;
    n.xy *= materialNormalScale(material);
    float nLen = length(n);
    if (nLen > kEpsilon) { n /= nLen; } else { n = vec3(0.0, 0.0, 1.0); }
    // Derivative-based TBN avoids reliance on imported tangent sign conventions.
    sm.nBase = applyNormalMap(sm.nBase, vtx.worldPos, uvNormal, n);
  }
  sm.roughness = applySpecularAARoughnessBias(sm.roughness, sm.nBase);

  // Clearcoat normal with normal map + roughness bias -----------------
  sm.nClearcoat = sm.nGeom;
  if (sm.hasClearcoat && clearcoatNormalTexId != kInvalidTextureBindlessIndex) {
    vec3 n =
        textureBindless2D(clearcoatNormalTexId, matSampler, uvClearcoatNormal)
            .xyz * 2.0 - 1.0;
    n.xy *= material.sheenRoughnessClearcoatFactors.w;
    float nLen = length(n);
    if (nLen > kEpsilon) { n /= nLen; } else { n = vec3(0.0, 0.0, 1.0); }
    vec3 perturbed =
        applyNormalMap(sm.nClearcoat, vtx.worldPos, uvClearcoatNormal, n);
    // Very glossy clearcoat turns harsh normal-map distortion into a plastic
    // shell without specular AA.  Bias back toward the geometric normal so the
    // reflection remains coherent.
    float blend = clamp(sqrt(sm.clearcoatRoughness), kBrdfMinRoughness, 1.0);
    sm.nClearcoat = normalize(mix(sm.nClearcoat, perturbed, blend));
  }
  sm.clearcoatRoughness =
      applySpecularAARoughnessBias(sm.clearcoatRoughness, sm.nClearcoat);

  // Emissive ----------------------------------------------------------
  sm.emissive = material.emissiveFactorStrength.xyz;
  if (emissiveTexId != kInvalidTextureBindlessIndex) {
    sm.emissive *=
        textureBindless2D(emissiveTexId, matSampler, uvEmissive).rgb;
  }
  sm.emissive *= materialEmissiveStrength(material);

  // View direction and dot products ------------------------------------
  sm.iorCompatMode  = isIorCompatMode(sm.ior);
  sm.v              = normalize(pc.frameData.cameraPos.xyz - vtx.worldPos);
  sm.ndotv          = max(dot(sm.nBase, sm.v), 0.001);
  sm.clearcoatNdotV = max(dot(sm.nClearcoat, sm.v), 0.001);
  sm.alphaRoughness = sm.roughness * sm.roughness;

  // Sheen -------------------------------------------------------------
  sm.sheenWeight = ((featureMask & kMaterialFeatureSheen) != 0u)
      ? saturate(material.sheenColorFactorWeight.w) : 0.0;
  sm.sheenRoughness = clamp(material.sheenRoughnessClearcoatFactors.x,
                            kBrdfMinRoughness, 1.0);
  sm.sheenColor = material.sheenColorFactorWeight.xyz;
  if (sheenColorTexId != kInvalidTextureBindlessIndex) {
    sm.sheenColor *=
        textureBindless2D(sheenColorTexId, matSampler, uvSheenColor).rgb;
  }
  if (sheenRoughnessTexId != kInvalidTextureBindlessIndex) {
    sm.sheenRoughness = clamp(
        sm.sheenRoughness *
            textureBindless2D(sheenRoughnessTexId, matSampler,
                              uvSheenRoughness).a,
        kBrdfMinRoughness, 1.0);
  }

  // Clearcoat attenuation factor --------------------------------------
  if (sm.hasClearcoat) {
    vec3 ccF = fresnelSchlick(sm.clearcoatNdotV, sm.clearcoatF0);
    sm.clearcoatAttenuation = vec3(1.0) - sm.clearcoat * ccF;
  }

  return sm;
}

// Accumulate one light's contribution, splitting diffuse and specular so
// the transmission pass can independently blend them with transmission.
void accumulateSurfaceLightContribution(
    vec3 lightRadiance, vec3 l, ShadedMaterial sm,
    inout vec3 directDiffuse, inout vec3 directSpecular,
    inout vec3 directSheen, inout vec3 clearcoatDirectLighting) {

  float ndotl          = max(dot(sm.nBase, l), 0.0);
  float clearcoatNdotL = max(dot(sm.nClearcoat, l), 0.0);
  vec3  halfVector     = sm.v + l;
  float halfLenSq      = dot(halfVector, halfVector);

  if (ndotl > 0.0 && halfLenSq > kEpsilon) {
    vec3  h     = halfVector * inversesqrt(halfLenSq);
    float ndoth = max(dot(sm.nBase, h), 0.0);
    float ldoth = max(dot(l, h), 0.0);
    float vdoth = max(dot(sm.v, h), 0.0);

    vec3  f = specularReflection(vdoth, sm.f0, sm.f90);
    float g = geometryOcclusion(ndotl, sm.ndotv, sm.alphaRoughness);
    float d = distributionGGX(ndoth, sm.alphaRoughness);

    float sheenScale = sm.sheenWeight > 0.0
        ? computeSheenAlbedoScalingDirect(sm.sheenColor, sm.ndotv, ndotl,
                                          sm.sheenRoughness)
        : 1.0;
    directDiffuse +=
        sheenScale * ndotl * lightRadiance *
        ((1.0 - max3(f)) *
         diffuseBurley(sm.diffuseColor, ndotl, sm.ndotv, ldoth,
                       sm.alphaRoughness));
    directSpecular +=
        sheenScale * ndotl * lightRadiance *
        (f * g * d / max(4.0 * ndotl * sm.ndotv, kEpsilon));
    directSheen += computeDirectSheen(sm.sheenColor, sm.sheenWeight,
                                      sm.sheenRoughness, ndotl, sm.ndotv,
                                      ndoth, lightRadiance);
  }

  if (sm.hasClearcoat && sm.clearcoat > 0.0 &&
      clearcoatNdotL > 0.0 && halfLenSq > kEpsilon) {
    vec3  h              = halfVector * inversesqrt(halfLenSq);
    float ccNdotH        = max(dot(sm.nClearcoat, h), 0.0);
    float ccVdotH        = max(dot(sm.v, h), 0.0);
    float ccAlpha        = sm.clearcoatRoughness * sm.clearcoatRoughness;
    vec3  ccF            = specularReflection(ccVdotH, sm.clearcoatF0,
                                              sm.clearcoatReflectance90);
    float ccG            = geometryOcclusion(clearcoatNdotL, sm.clearcoatNdotV,
                                             ccAlpha);
    float ccD            = distributionGGX(ccNdotH, ccAlpha);
    clearcoatDirectLighting +=
        sm.clearcoat * clearcoatNdotL * lightRadiance *
        (ccF * ccG * ccD /
         max(4.0 * clearcoatNdotL * sm.clearcoatNdotV, kEpsilon));
  }
}

// Output of the image-based lighting evaluation.
struct IblResult {
  vec3  iblDiffuse;
  vec3  iblSpecular;
  vec3  iblSheen;
  vec3  clearcoatIblSpecular;
  float indirectScale;
  bool  hasIndirectLighting;
};

// Evaluate all IBL terms.  AO is NOT applied here; the caller controls it
// so that the transmission pass can substitute the diffuse term before scaling.
IblResult evaluateIbl(ShadedMaterial sm) {
  IblResult r;
  r.iblDiffuse           = vec3(0.0);
  r.iblSpecular          = vec3(0.0);
  r.iblSheen             = vec3(0.0);
  r.clearcoatIblSpecular = vec3(0.0);
  r.indirectScale        = 1.0;
  r.hasIndirectLighting  = false;

  vec3 baseBrdfLutSample  = vec3(0.0);
  vec3 sheenBrdfLutSample = vec3(0.0);
  bool hasBrdfLut =
      (pc.frameData.flags & kFrameDataFlagHasBrdfLut) != 0u &&
      pc.frameData.brdfLutTexId != kInvalidTextureBindlessIndex;
  if (hasBrdfLut) {
    vec2 baseBrdfUv =
        clamp(vec2(sm.ndotv, 1.0 - sm.roughness * sm.roughness),
              vec2(0.0), vec2(1.0));
    baseBrdfLutSample =
        textureBindless2D(pc.frameData.brdfLutTexId, 0u, baseBrdfUv).rgb;
    if (sm.sheenWeight > 0.0) {
      vec2 sheenBrdfUv =
          clamp(vec2(sm.ndotv,
                     1.0 - sm.sheenRoughness * sm.sheenRoughness),
                vec2(0.0), vec2(1.0));
      sheenBrdfLutSample =
          textureBindless2D(pc.frameData.brdfLutTexId, 0u,
                            sheenBrdfUv).rgb;
    }
  }
  if (sm.sheenWeight > 0.0 && hasBrdfLut) {
    r.indirectScale = computeSheenAlbedoScalingIndirect(
        sm.sheenColor, sm.sheenWeight, sheenBrdfLutSample);
  }

  if ((pc.frameData.flags & kFrameDataFlagHasIblDiffuse) != 0u &&
      pc.frameData.irradianceTexId != kInvalidTextureBindlessIndex) {
    vec3 irradiance =
        textureBindlessCube(pc.frameData.irradianceTexId,
                            pc.frameData.cubemapSamplerId, sm.nBase).rgb;
    if (!sm.iorCompatMode) {
      r.iblDiffuse = hasBrdfLut
          ? computeIblDiffuse(sm.diffuseColor, sm.f0, sm.f90, irradiance,
                              baseBrdfLutSample)
          : sm.diffuseColor *
                (1.0 - max3(fresnelSchlick(sm.ndotv, sm.f0, sm.f90))) *
                irradiance;
    }
    r.hasIndirectLighting = true;
  }

  if ((pc.frameData.flags & kFrameDataFlagHasIblSpecular) != 0u &&
      pc.frameData.prefilteredGgxTexId != kInvalidTextureBindlessIndex) {
    vec3 ref = reflect(-sm.v, sm.nBase);
    if (hasBrdfLut) {
      float mipCount = float(
          textureBindlessQueryLevelsCube(pc.frameData.prefilteredGgxTexId));
      float lod = sm.roughness * max(mipCount - 1.0, 0.0);
      vec3 prefiltered =
          textureBindlessCubeLod(pc.frameData.prefilteredGgxTexId,
                                 pc.frameData.cubemapSamplerId, ref, lod).rgb;
      r.iblSpecular =
          computeIblSpecular(sm.f0, sm.f90, prefiltered, baseBrdfLutSample);
    } else {
      r.iblSpecular =
          textureBindlessCube(pc.frameData.prefilteredGgxTexId,
                              pc.frameData.cubemapSamplerId, ref).rgb *
          fresnelSchlick(sm.ndotv, sm.f0);
    }
    r.hasIndirectLighting = true;
  }

  if ((pc.frameData.flags & kFrameDataFlagHasIblSheen) != 0u &&
      pc.frameData.prefilteredCharlieTexId != kInvalidTextureBindlessIndex &&
      hasBrdfLut && sm.sheenWeight > 0.0) {
    vec3 ref = reflect(-sm.v, sm.nBase);
    float mipCount = float(
        textureBindlessQueryLevelsCube(pc.frameData.prefilteredCharlieTexId));
    float lod = sm.sheenRoughness * max(mipCount - 1.0, 0.0);
    vec3 sheenEnv =
        textureBindlessCubeLod(pc.frameData.prefilteredCharlieTexId,
                               pc.frameData.cubemapSamplerId, ref, lod).rgb;
    r.iblSheen = computeIblSheen(sm.sheenColor, sm.sheenWeight, sheenEnv,
                                  sheenBrdfLutSample);
    r.hasIndirectLighting = true;
  }

  if (sm.hasClearcoat && sm.clearcoat > 0.0 &&
      (pc.frameData.flags & kFrameDataFlagHasIblSpecular) != 0u &&
      pc.frameData.prefilteredGgxTexId != kInvalidTextureBindlessIndex) {
    vec3 ccRef = reflect(-sm.v, sm.nClearcoat);
    if (hasBrdfLut) {
      vec2 ccBrdfUv = clamp(
          vec2(sm.clearcoatNdotV,
               1.0 - sm.clearcoatRoughness * sm.clearcoatRoughness),
          vec2(0.0), vec2(1.0));
      vec3 ccBrdfSample =
          textureBindless2D(pc.frameData.brdfLutTexId, 0u, ccBrdfUv).rgb;
      float mipCount = float(
          textureBindlessQueryLevelsCube(pc.frameData.prefilteredGgxTexId));
      float lod = sm.clearcoatRoughness * max(mipCount - 1.0, 0.0);
      vec3 prefiltered =
          textureBindlessCubeLod(pc.frameData.prefilteredGgxTexId,
                                 pc.frameData.cubemapSamplerId, ccRef,
                                 lod).rgb;
      r.clearcoatIblSpecular =
          sm.clearcoat *
          computeIblSpecular(sm.clearcoatF0, sm.clearcoatReflectance90,
                             prefiltered, ccBrdfSample);
    } else {
      r.clearcoatIblSpecular =
          sm.clearcoat *
          textureBindlessCube(pc.frameData.prefilteredGgxTexId,
                              pc.frameData.cubemapSamplerId, ccRef).rgb *
          fresnelSchlick(sm.clearcoatNdotV, sm.clearcoatF0);
    }
    r.hasIndirectLighting = true;
  }

  return r;
}
