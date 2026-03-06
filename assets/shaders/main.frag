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

  const uint baseColorTexId = material.textureIndices0.x;
  const uint metallicRoughnessTexId = material.textureIndices0.y;
  const uint normalTexId = material.textureIndices0.z;
  const uint occlusionTexId = material.textureIndices0.w;
  const uint emissiveTexId = material.textureIndices1.x;
  const uint alphaMode = material.textureIndices1.y;

  const uint baseColorUvSet = material.textureUvSets0.x;
  const uint metallicRoughnessUvSet = material.textureUvSets0.y;
  const uint normalUvSet = material.textureUvSets0.z;
  const uint occlusionUvSet = material.textureUvSets0.w;
  const uint emissiveUvSet = material.textureUvSets1.x;

  const uint baseColorSampler = material.textureSamplerIndices0.x;
  const uint metallicRoughnessSampler = material.textureSamplerIndices0.y;
  const uint normalSampler = material.textureSamplerIndices0.z;
  const uint occlusionSampler = material.textureSamplerIndices0.w;
  const uint emissiveSampler = material.textureSamplerIndices1.x;

  const vec2 uvBaseColor = selectUv(vtx.uv0, vtx.uv1, baseColorUvSet);
  const vec2 uvMetallicRoughness =
      selectUv(vtx.uv0, vtx.uv1, metallicRoughnessUvSet);
  const vec2 uvNormal = selectUv(vtx.uv0, vtx.uv1, normalUvSet);
  const vec2 uvOcclusion = selectUv(vtx.uv0, vtx.uv1, occlusionUvSet);
  const vec2 uvEmissive = selectUv(vtx.uv0, vtx.uv1, emissiveUvSet);

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
    occlusion = textureBindless2D(occlusionTexId, occlusionSampler, uvOcclusion).r;
  } else if (metallicRoughnessTexId != kInvalidTextureBindlessIndex) {
    occlusion = mrSample.r;
  }
  float ao = mix(1.0, occlusion,
                 saturate(material.metallicRoughnessOcclusionAlphaCutoff.z));

  vec3 n = normalize(vtx.worldNormal);
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
    // Derivative-based TBN avoids
    // reliance on imported tangent sign conventions.
    n = applyNormalMap(n, vtx.worldPos, uvNormal, normalTexel);
  }
  if (!gl_FrontFacing) {
    n *= -1.0;
  }

  vec3 emissive = material.emissiveFactorNormalScale.xyz;
  if (emissiveTexId != kInvalidTextureBindlessIndex) {
    emissive *= textureBindless2D(emissiveTexId, emissiveSampler, uvEmissive).rgb;
  }

  vec3 v = normalize(pc.frameData.cameraPos.xyz - vtx.worldPos);
  float ndotv = max(dot(n, v), 0.001);

  vec3 f0 = mix(vec3(0.04), baseColor.rgb, metallic);
  vec3 diffuseColor = mix(baseColor.rgb, vec3(0.0), metallic);
  float alphaRoughness = roughness * roughness;
  float reflectance = max(max(f0.r, f0.g), f0.b);
  vec3 reflectance90 = vec3(clamp(reflectance * 25.0, 0.0, 1.0));

  const vec3 lightPos = vec3(0.0, 0.0, -5.0);
  const vec3 lightColor = vec3(1.0);
  vec3 l = normalize(lightPos - vtx.worldPos);
  float ndotl = max(dot(n, l), 0.0);
  vec3 directLighting = vec3(0.0);
  vec3 halfVector = v + l;
  float halfLenSq = dot(halfVector, halfVector);
  if (ndotl > 0.0 && halfLenSq > kBrdfEpsilon) {
    vec3 h = halfVector * inversesqrt(halfLenSq);
    float ndoth = max(dot(n, h), 0.0);
    float ldoth = max(dot(l, h), 0.0);
    float vdoth = max(dot(v, h), 0.0);

    vec3 f = specularReflection(vdoth, f0, reflectance90);
    float g = geometryOcclusion(ndotl, ndotv, alphaRoughness);
    float d = microfacetDistribution(ndoth, alphaRoughness);
    vec3 diffuse = (1.0 - f) *
                   diffuseBurley(diffuseColor, ndotl, ndotv, ldoth,
                                 alphaRoughness);
    vec3 specular = f * g * d / max(4.0 * ndotl * ndotv, kBrdfEpsilon);
    directLighting = ndotl * lightColor * (diffuse + specular);
  }

  vec3 brdfLutSample = vec3(0.0);
  bool hasBrdfLut = (pc.frameData.flags & kFrameDataFlagHasBrdfLut) != 0u &&
                    pc.frameData.brdfLutTexId != kInvalidTextureBindlessIndex;
  if (hasBrdfLut) {
    vec2 brdfUv = clamp(vec2(ndotv, 1.0 - roughness * roughness), vec2(0.0),
                        vec2(1.0));
    brdfLutSample = textureBindless2D(pc.frameData.brdfLutTexId, 0, brdfUv).rgb;
  }

  vec3 iblDiffuse = vec3(0.0);
  vec3 iblSpecular = vec3(0.0);
  vec3 iblSheen = vec3(0.0);
  bool hasIndirectLighting = false;
  if ((pc.frameData.flags & kFrameDataFlagHasIblDiffuse) != 0u &&
      pc.frameData.irradianceTexId != kInvalidTextureBindlessIndex) {
    vec3 irradiance =
        textureBindlessCube(pc.frameData.irradianceTexId,
                            pc.frameData.cubemapSamplerId, n)
            .rgb;
    if (hasBrdfLut) {
      iblDiffuse = computeIblDiffuse(diffuseColor, f0, roughness, ndotv,
                                     irradiance, brdfLutSample);
    } else {
      iblDiffuse = diffuseColor * irradiance;
    }
    hasIndirectLighting = true;
  }

  if ((pc.frameData.flags & kFrameDataFlagHasIblSpecular) != 0u &&
      pc.frameData.prefilteredGgxTexId != kInvalidTextureBindlessIndex) {
    vec3 r = reflect(-v, n);
    if (hasBrdfLut) {
      float mipCount =
          float(textureBindlessQueryLevelsCube(pc.frameData.prefilteredGgxTexId));
      float lod = roughness * max(mipCount - 1.0, 0.0);
      vec3 prefiltered =
          textureBindlessCubeLod(pc.frameData.prefilteredGgxTexId,
                                 pc.frameData.cubemapSamplerId, r, lod)
              .rgb;
      iblSpecular =
          computeIblSpecular(f0, roughness, ndotv, prefiltered, brdfLutSample);
    } else {
      iblSpecular =
          textureBindlessCube(pc.frameData.prefilteredGgxTexId,
                              pc.frameData.cubemapSamplerId, r)
              .rgb *
          fresnelSchlick(ndotv, f0);
    }
    hasIndirectLighting = true;
  }

  float sheenWeight = saturate(material.sheenColorFactorWeight.w);
  float sheenRoughness =
      clamp(material.sheenRoughnessReserved.x, kBrdfMinRoughness, 1.0);
  vec3 sheenColor = material.sheenColorFactorWeight.xyz;
  if ((pc.frameData.flags & kFrameDataFlagHasIblSheen) != 0u &&
      pc.frameData.prefilteredCharlieTexId != kInvalidTextureBindlessIndex &&
      hasBrdfLut && sheenWeight > 0.0) {
    vec3 r = reflect(-v, n);
    float mipCount = float(
        textureBindlessQueryLevelsCube(pc.frameData.prefilteredCharlieTexId));
    float lod = sheenRoughness * max(mipCount - 1.0, 0.0);
    vec3 sheenEnv =
        textureBindlessCubeLod(pc.frameData.prefilteredCharlieTexId,
                               pc.frameData.cubemapSamplerId, r, lod)
            .rgb;
    iblSheen = computeIblSheen(sheenColor, sheenWeight, sheenEnv, brdfLutSample);
    hasIndirectLighting = true;
  }

  vec3 indirectLighting = iblDiffuse + iblSpecular + iblSheen;
  if (hasIndirectLighting) {
    indirectLighting *= ao;
  }
  vec3 color = directLighting + indirectLighting + emissive;
  color = max(color, vec3(0.0));
  if ((pc.frameData.flags & kFrameDataFlagOutputLinearToSrgb) != 0u) {
    color = pow(color, vec3(1.0 / 2.2));
  }

  float outAlpha = (alphaMode == kAlphaModeOpaque) ? 1.0 : baseColor.a;
  out_FragColor = vec4(color, outAlpha);
}
