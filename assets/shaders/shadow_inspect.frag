#include "BRDF.sp"
#include "common.sp"
#include "material_inputs.sp"
#include "material_lighting.sp"

layout(location = 0) in PerVertex vtx;

layout(location = 0) out vec4 out_ShadowInspect;

void main() {
  const MaterialData material = loadMaterialData(pc.materialIndex);
  const uint alphaMode = materialAlphaMode(material);

  ShadedMaterial sm = evaluateMaterial(material, vtx);
  const float alphaCutoff = materialAlphaCutoff(material);
  if (alphaMode == kAlphaModeMask && sm.baseColor.a < alphaCutoff) {
    discard;
  }

  if ((pc.frameData.shadowFlags & kShadowFrameFlagEnabled) == 0u) {
    out_ShadowInspect = vec4(0.0);
    return;
  }

  const HardShadowInspectResult inspect = inspectHardDirectionalShadow(
      pc.frameData.shadowFrameBuffer, vtx.worldPos);
  out_ShadowInspect = vec4(inspect.receiverDepth, inspect.receiverCompareDepth,
                           inspect.sampledDepth, inspect.valid);
}
