#include "BRDF.sp"
#include "common.sp"
#include "material_inputs.sp"
#include "material_lighting.sp"

layout(location = 0) in PerVertex vtx;

layout(location = 0) out vec4 out_ShadowInspect;

void main() {
  if ((pc.frameData.shadowFlags & kShadowFrameFlagEnabled) == 0u) {
    out_ShadowInspect = vec4(0.0);
    return;
  }

  const MaterialData material = loadMaterialData(pc.materialIndex);
  const uint alphaMode = materialAlphaMode(material);

  ShadedMaterial sm = evaluateMaterial(material, vtx);
  const float alphaCutoff = materialAlphaCutoff(material);
  if (alphaMode == kAlphaModeMask && sm.baseColor.a < alphaCutoff) {
    discard;
  }

  ShadowFrameBuffer shadow = pc.frameData.shadowFrameBuffer;
  uvec4 shadowState = shadow.flagsCascadeCountLightIndex;
  if (shadowState.y == 0u ||
      shadowState.z >= pc.frameData.directionalLightCount) {
    out_ShadowInspect = vec4(0.0);
    return;
  }
  DirectionalLightGpuData light =
      pc.frameData.directionalLightBuffer.lights[shadowState.z];
  vec3 l = normalize(-directionalLightDirection(light));
  const HardShadowInspectResult inspect =
      inspectHardDirectionalShadow(shadow, vtx.worldPos, sm.nBase, l);
  out_ShadowInspect = vec4(inspect.receiverDepth, inspect.receiverCompareDepth,
                           inspect.sampledDepth, inspect.valid);
}
