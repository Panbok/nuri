#include "common.sp"

layout(location = 0) out PerVertex vtx;
layout(location = 10) out vec3 outCurrentWorldPos;
layout(location = 11) out vec3 outPreviousWorldPos;
layout(location = 12) out uint outVelocityFlags;

void main() {
  const uint globalInstanceId = pc.instanceRemap.ids[gl_InstanceIndex];
  const vec3 pos = decodePackedPosition(gl_VertexIndex);
  const vec3 normal = decodePackedNormal(gl_VertexIndex);
  const vec4 tangent = decodePackedTangent(gl_VertexIndex);
  const vec2 uv0 = decodePackedUv(gl_VertexIndex);
  const vec2 uv1 = decodePackedUv1(gl_VertexIndex);

  const InstanceData inst = pc.instanceMatrices.instances[globalInstanceId];
  const mat4 model = inst.modelMatrix;

  const vec4 localPos = vec4(pos, 1.0);
  const vec4 worldPos4 = model * localPos;

  uint velocityFlags = 0u;
  const uint instanceFlagsMode = pc.velocityFrameData.data.instanceFlagsMode.x;
  if (instanceFlagsMode == kVelocityInstanceFlagsModeAllValid) {
    velocityFlags = 1u;
  } else if (instanceFlagsMode == kVelocityInstanceFlagsModeBuffer) {
    velocityFlags = pc.velocityInstanceFlags.flags[globalInstanceId];
  }

  vec3 previousPos = pos;
  if (velocityFlags != 0u) {
    const bool currentDrawAnimated =
        pc.packedVertexFormat == kPackedVertexFormatAnimatedFloat24 ||
        pc.packedVertexFormat == kPackedVertexFormatAnimatedFloat32;
    if (currentDrawAnimated &&
        globalInstanceId < pc.velocityFrameData.data.previousGeometryInfo.x) {
      VelocityRenderableGeometryData previousGeometry =
          pc.velocityFrameData.data.previousGeometry.values[globalInstanceId];
      const bool hasPreviousVertexBuffer =
          (previousGeometry.metadata.x &
           kVelocityGeometryFlagPreviousVertexBuffer) != 0u;
      const uint previousVertexCount = previousGeometry.metadata.z;
      if (hasPreviousVertexBuffer && uint(gl_VertexIndex) < previousVertexCount) {
        previousPos = decodeAnimatedPositionFrom(
            previousGeometry.previousVertexBuffer, uint(gl_VertexIndex),
            previousGeometry.metadata.y);
      }
    }
  }

  const InstanceData previousInst =
      pc.previousInstanceMatrices.instances[globalInstanceId];
  const vec4 previousWorldPos4 =
      velocityFlags != 0u ? previousInst.modelMatrix * vec4(previousPos, 1.0)
                          : worldPos4;

  gl_Position = pc.frameData.proj * pc.frameData.view * worldPos4;

  const mat3 normalMatrix = mat3(inst.normalMatCol0.xyz, inst.normalMatCol1.xyz,
                                 inst.normalMatCol2.xyz);
  const vec3 worldNormal = normalize(normalMatrix * normal);
  vec3 transformedTangent = mat3(model) * tangent.xyz;
  transformedTangent -= worldNormal * dot(transformedTangent, worldNormal);
  vtx.uv0 = uv0;
  vtx.uv1 = uv1;
  vtx.worldNormal = worldNormal;
  vtx.worldTangent = dot(transformedTangent, transformedTangent) > 1.0e-10
                         ? vec4(normalize(transformedTangent), tangent.w)
                         : vec4(0.0, 0.0, 0.0, 1.0);
  vtx.worldPos = worldPos4.xyz;
  vtx.patchBarycentric = vec3(0.0);
  vtx.triBarycentric = vec3(0.0);
  vtx.patchOuterFactors = vec3(1.0);
  vtx.patchInnerFactor = 1.0;
  vtx.tessellatedFlag = 0.0;
  outCurrentWorldPos = worldPos4.xyz;
  outPreviousWorldPos = previousWorldPos4.xyz;
  outVelocityFlags = velocityFlags;
}
