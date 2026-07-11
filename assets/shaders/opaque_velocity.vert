#include "common.sp"

layout(location = 0) out PerVertex vtx;
layout(location = 10) out vec4 outCurrentClipNoJitter;
layout(location = 11) out vec4 outPreviousClipNoJitter;
layout(location = 12) flat out uint outVelocityFlags;

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
  const vec4 currentClipNoJitter =
      pc.velocityFrameData.data.currentViewProjNoJitter * worldPos4;

  uint velocityFlags = 0u;
  const uint instanceFlagsMode = pc.velocityFrameData.data.instanceFlagsMode.x;
  if (instanceFlagsMode == kVelocityInstanceFlagsModeAllValid) {
    velocityFlags = 1u;
  } else if (instanceFlagsMode == kVelocityInstanceFlagsModeBuffer) {
    velocityFlags = pc.velocityInstanceFlags.flags[globalInstanceId];
  }

  vec4 previousClipNoJitter = currentClipNoJitter;
  if (velocityFlags != 0u) {
    vec3 previousPos = pos;
    const bool currentDrawAnimated =
        pc.packedVertexFormat == kPackedVertexFormatAnimatedFloat24 ||
        pc.packedVertexFormat == kPackedVertexFormatAnimatedFloat32;
    bool hasTrustworthyPreviousPosition = !currentDrawAnimated;
    if (currentDrawAnimated &&
        globalInstanceId < pc.velocityFrameData.data.previousGeometryInfo.x) {
      VelocityRenderableGeometryData previousGeometry =
          pc.velocityFrameData.data.previousGeometry.values[globalInstanceId];
      const bool hasPreviousVertexBuffer =
          (previousGeometry.metadata.x &
           kVelocityGeometryFlagPreviousVertexBuffer) != 0u;
      const uint previousVertexCount = previousGeometry.metadata.z;
      hasTrustworthyPreviousPosition =
          hasPreviousVertexBuffer && uint(gl_VertexIndex) < previousVertexCount;
      if (hasTrustworthyPreviousPosition) {
        previousPos = decodeAnimatedPositionFrom(
            previousGeometry.previousVertexBuffer, uint(gl_VertexIndex),
            previousGeometry.metadata.y);
      }
    }
    if (!hasTrustworthyPreviousPosition) {
      velocityFlags = 0u;
    }

    if (velocityFlags != 0u) {
      const InstanceData previousInst =
          pc.previousInstanceMatrices.instances[globalInstanceId];
      previousClipNoJitter =
          pc.velocityFrameData.data.previousViewProjNoJitter *
          previousInst.modelMatrix * vec4(previousPos, 1.0);
    }
  }

  gl_Position = pc.frameData.proj * pc.frameData.view * worldPos4;

  const mat3 normalMatrix = mat3(inst.normalMatCol0.xyz, inst.normalMatCol1.xyz,
                                 inst.normalMatCol2.xyz);
  vtx.uv0 = uv0;
  vtx.uv1 = uv1;
  const vec3 worldNormal = normalize(normalMatrix * normal);
  vtx.worldNormal = worldNormal;
  vec3 transformedTangent = mat3(model) * tangent.xyz;
  transformedTangent -= worldNormal * dot(transformedTangent, worldNormal);
  vtx.worldTangent = dot(transformedTangent, transformedTangent) > 1.0e-10
                         ? vec4(normalize(transformedTangent), tangent.w)
                         : vec4(0.0, 0.0, 0.0, 1.0);
  vtx.worldPos = worldPos4.xyz;
  vtx.patchBarycentric = vec3(0.0);
  vtx.triBarycentric = vec3(0.0);
  vtx.patchOuterFactors = vec3(1.0);
  vtx.patchInnerFactor = 1.0;
  vtx.tessellatedFlag = 0.0;
  outCurrentClipNoJitter = currentClipNoJitter;
  outPreviousClipNoJitter = previousClipNoJitter;
  outVelocityFlags = velocityFlags;
}
