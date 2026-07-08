#define NURI_OPAQUE_MESHLET_BATCHED 1
#include "meshlet_common.sp"

layout(location = 0) in vec3 vtxWorldNormal;

layout(location = 0) out vec4 out_Normal;

void main() {
  vec3 worldNormal = normalize(vtxWorldNormal);
  if (!gl_FrontFacing) {
    worldNormal *= -1.0;
  }

  const vec3 viewNormal = normalize(mat3(pc.frameData.view) * worldNormal);
  out_Normal = vec4(viewNormal * 0.5 + 0.5, 1.0);
}
