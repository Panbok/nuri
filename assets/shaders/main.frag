#include "common.sp"

layout(location = 0) in PerVertex vtx;
layout(location = 9) flat in uint inInstanceId;

layout(location = 0) out vec4 out_FragColor;

void main() {
  const vec3 n = normalize(vtx.worldNormal);
  const vec3 lightDir = normalize(vec3(-1.0, 1.0, -1.0));
  const float ndotl = clamp(dot(n, lightDir), 0.5, 1.0);
  const uint albedoTexId = pc.instanceMeta.albedoTexIds[inInstanceId];
  const vec4 albedo = textureBindless2D(albedoTexId, 0, vtx.uv);
  out_FragColor = vec4(albedo.rgb * ndotl, albedo.a);
}
