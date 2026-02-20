#include "common.sp"

layout(location = 0) in PerVertex vtx;
layout(location = 9) flat in uint inInstanceId;

layout(location = 0) out vec4 out_FragColor;

float hash1(uint v) {
  v ^= 2747636419u;
  v *= 2654435769u;
  v ^= v >> 16u;
  v *= 2654435769u;
  v ^= v >> 16u;
  v *= 2654435769u;
  return float(v) / 4294967295.0;
}

vec3 instanceTint(uint instanceId) {
  const float r = hash1(instanceId * 3u + 1u);
  const float g = hash1(instanceId * 3u + 2u);
  const float b = hash1(instanceId * 3u + 3u);
  return normalize(vec3(0.35 + r, 0.35 + g, 0.35 + b));
}

void main() {
  const vec3 n = normalize(vtx.worldNormal);
  const vec3 v = normalize(pc.frameData.cameraPos.xyz - vtx.worldPos);
  const vec3 reflection = -normalize(reflect(v, n));

  vec4 reflectedColor = vec4(0.0);
  if (pc.frameData.hasCubemap != 0u) {
    reflectedColor =
        textureBindlessCube(pc.frameData.cubemapTexId, 0, reflection);
  }

  const float ndotl = clamp(dot(n, normalize(vec3(0.0, 0.0, -1.0))), 0.1, 1.0);
  const uint albedoTexId = pc.instanceMeta.albedoTexIds[inInstanceId];
  vec4 albedo = textureBindless2D(albedoTexId, 0, vtx.uv) * ndotl;

  const vec3 tint = instanceTint(inInstanceId);
  albedo.rgb = mix(albedo.rgb, albedo.rgb * tint, 0.42);
  out_FragColor = reflectedColor * 0.3 + albedo;
}
