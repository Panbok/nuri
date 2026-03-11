const float kBrdfPi = 3.14159265359;
const float kBrdfMinRoughness = 0.04;
const float kBrdfEpsilon = 1.0e-5;

#include "sheen.sp"

float saturate(float x) { return clamp(x, 0.0, 1.0); }
float max3(vec3 v) { return max(max(v.x, v.y), v.z); }
float pow5(float x) {
  float x2 = x * x;
  return x2 * x2 * x;
}

vec3 fresnelSchlick(float cosTheta, vec3 f0) {
  return f0 + (1.0 - f0) * pow5(1.0 - saturate(cosTheta));
}

float distributionGGX(float ndoth, float roughness) {
  float alpha = roughness * roughness;
  float alpha2 = alpha * alpha;
  float denom = ndoth * ndoth * (alpha2 - 1.0) + 1.0;
  return alpha2 / max(kBrdfPi * denom * denom, kBrdfEpsilon);
}

float geometrySchlickGGX(float ndotx, float roughness) {
  float r = roughness + 1.0;
  float k = (r * r) * 0.125;
  return ndotx / max(ndotx * (1.0 - k) + k, kBrdfEpsilon);
}

float geometrySmith(float ndotv, float ndotl, float roughness) {
  return geometrySchlickGGX(ndotv, roughness) *
         geometrySchlickGGX(ndotl, roughness);
}

// Khronos-compatible direct-light terms for metallic-roughness.
vec3 diffuseBurley(vec3 diffuseColor, float ndotl, float ndotv, float ldoth,
                   float alphaRoughness) {
  float f90 = 2.0 * ldoth * ldoth * alphaRoughness - 0.5;
  return (diffuseColor / kBrdfPi) *
         (1.0 + f90 * pow5(1.0 - ndotl)) *
         (1.0 + f90 * pow5(1.0 - ndotv));
}

vec3 specularReflection(float vdoth, vec3 reflectance0, vec3 reflectance90) {
  return reflectance0 +
         (reflectance90 - reflectance0) *
             pow5(clamp(1.0 - vdoth, 0.0, 1.0));
}

float geometryOcclusion(float ndotl, float ndotv, float alphaRoughness) {
  float rSqr = alphaRoughness * alphaRoughness;
  float attenuationL =
      2.0 * ndotl / (ndotl + sqrt(rSqr + (1.0 - rSqr) * (ndotl * ndotl)));
  float attenuationV =
      2.0 * ndotv / (ndotv + sqrt(rSqr + (1.0 - rSqr) * (ndotv * ndotv)));
  return attenuationL * attenuationV;
}

float microfacetDistribution(float ndoth, float alphaRoughness) {
  float roughnessSq = alphaRoughness * alphaRoughness;
  float f = (ndoth * roughnessSq - ndoth) * ndoth + 1.0;
  return roughnessSq / max(kBrdfPi * f * f, kBrdfEpsilon);
}

mat3 cotangentFrame(vec3 normal, vec3 worldPos, vec2 uv) {
  vec3 dp1 = dFdx(worldPos);
  vec3 dp2 = dFdy(worldPos);
  vec2 duv1 = dFdx(uv);
  vec2 duv2 = dFdy(uv);

  vec3 dp2perp = cross(dp2, normal);
  vec3 dp1perp = cross(normal, dp1);
  vec3 tangent = dp2perp * duv1.x + dp1perp * duv2.x;
  vec3 bitangent = dp2perp * duv1.y + dp1perp * duv2.y;

  float tangentLen2 = dot(tangent, tangent);
  float bitangentLen2 = dot(bitangent, bitangent);
  float maxLen2 = max(tangentLen2, bitangentLen2);
  if (maxLen2 < kBrdfEpsilon) {
    return mat3(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), normal);
  }

  float handedness = dot(cross(normal, tangent), bitangent) < 0.0 ? -1.0 : 1.0;
  tangent = normalize(tangent) * handedness;
  bitangent = normalize(bitangent);
  return mat3(tangent, bitangent, normal);
}

vec3 applyNormalMap(vec3 baseNormal, vec3 worldPos, vec2 uv,
                    vec3 tangentNormal) {
  return normalize(cotangentFrame(baseNormal, worldPos, uv) * tangentNormal);
}

vec3 applyNormalMap(vec3 baseNormal, vec4 worldTangent, vec3 worldPos, vec2 uv,
                    vec3 tangentNormal) {
  vec3 tangent = worldTangent.xyz;
  const float tangentLen2 = dot(tangent, tangent);
  if (tangentLen2 < kBrdfEpsilon) {
    return applyNormalMap(baseNormal, worldPos, uv, tangentNormal);
  }
  tangent = normalize(tangent);
  vec3 bitangent = normalize(cross(baseNormal, tangent)) *
                   (worldTangent.w >= 0.0 ? 1.0 : -1.0);
  mat3 tbn = mat3(tangent, bitangent, baseNormal);
  return normalize(tbn * tangentNormal);
}

vec3 computeIblDiffuse(vec3 diffuseColor, vec3 f0, float roughness, float ndotv,
                       vec3 irradiance, vec3 brdfLutSample) {
  vec3 fr = max(vec3(1.0 - roughness), f0) - f0;
  vec3 kS = f0 + fr * pow5(1.0 - ndotv);
  vec3 fssEss = kS * brdfLutSample.x + brdfLutSample.y;

  float ems = 1.0 - (brdfLutSample.x + brdfLutSample.y);
  vec3 fAvg = f0 + (1.0 - f0) / 21.0;
  vec3 fmsEms = (ems * fssEss * fAvg) /
                max(vec3(kBrdfEpsilon), 1.0 - fAvg * ems);
  vec3 kdIbl = diffuseColor * (1.0 - fssEss + fmsEms);
  return (fmsEms + kdIbl) * irradiance;
}

vec3 computeIblSpecular(vec3 f0, float roughness, float ndotv, vec3 prefiltered,
                        vec3 brdfLutSample) {
  vec3 fr = max(vec3(1.0 - roughness), f0) - f0;
  vec3 kS = f0 + fr * pow5(1.0 - ndotv);
  vec3 fssEss = kS * brdfLutSample.x + brdfLutSample.y;
  return prefiltered * fssEss;
}

vec3 computeIblSheen(vec3 sheenColor, float sheenWeight, vec3 sheenEnv,
                     vec3 brdfLutSample) {
  return sheenEnv * sheenColor * (brdfLutSample.z * sheenWeight);
}

vec3 computeDirectSheen(vec3 sheenColor, float sheenWeight,
                        float sheenRoughness, float ndotl, float ndotv,
                        float ndoth, vec3 lightColor) {
  float sheenDistribution = DCharlie(sheenRoughness, ndoth);
  float sheenVisibility = VSheen(ndotl, ndotv, sheenRoughness);
  return sheenWeight * ndotl * lightColor * sheenColor *
         (sheenDistribution * sheenVisibility);
}

float computeSheenAlbedoScalingIndirect(vec3 sheenColor,
                                        vec3 brdfLutSample) {
  return saturate(1.0 - max3(sheenColor) * brdfLutSample.z);
}

float computeSheenAlbedoScalingDirect(vec3 sheenColor, float ndotv,
                                      float ndotl, float sheenRoughness) {
  float energyV =
      1.0 - max3(sheenColor) *
                albedoSheenScalingFactor(ndotv, sheenRoughness);
  float energyL =
      1.0 - max3(sheenColor) *
                albedoSheenScalingFactor(ndotl, sheenRoughness);
  return saturate(min(energyV, energyL));
}
