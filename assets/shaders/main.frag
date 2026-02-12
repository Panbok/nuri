#include "common.sp"

layout(location = 0) in PerVertex vtx;

layout(location = 0) out vec4 out_FragColor;

void main() {
  const vec3 n = normalize(vtx.worldNormal);
  const vec3 v = normalize(pc.perFrame.cameraPos.xyz - vtx.worldPos);
  const vec3 reflection = -normalize(reflect(v, n));

  const vec4 reflectedColor =
      textureBindlessCube(pc.perFrame.cubemapTexId, 0, reflection);
  const float ndotl = clamp(dot(n, normalize(vec3(0.0, 0.0, -1.0))), 0.1, 1.0);
  const vec4 albedo =
      textureBindless2D(pc.perFrame.albedoTexId, 0, vtx.uv) * ndotl;

  out_FragColor = reflectedColor * 0.3 + albedo;
}
