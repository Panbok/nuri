#extension GL_EXT_mesh_shader : require

#define NURI_OPAQUE_MESHLET_BATCHED 1
#include "meshlet_common.sp"

layout(local_size_x = 32) in;
layout(triangles, max_vertices = 64, max_primitives = 124) out;

out gl_MeshPerVertexEXT { vec4 gl_Position; }
gl_MeshVerticesEXT[];

layout(location = 0) out PerVertex vtx[];
layout(location = 10) out vec4 outCurrentClipNoJitter[];
layout(location = 11) out vec4 outPreviousClipNoJitter[];
layout(location = 12) flat out uint outVelocityFlags[];
layout(location = 13) flat out uint meshletMaterialIndex[];

taskPayloadSharedEXT MeshletTaskPayload meshletPayload;

uint meshletVelocityFlags(uint globalInstanceId) {
  const uint instanceFlagsMode = pc.velocityFrameData.data.instanceFlagsMode.x;
  if (instanceFlagsMode == kVelocityInstanceFlagsModeAllValid) {
    return 1u;
  }
  if (instanceFlagsMode == kVelocityInstanceFlagsModeBuffer) {
    return pc.velocityFrameData.data.velocityInstanceFlags
        .flags[globalInstanceId];
  }
  return 0u;
}

bool meshletPreviousPosition(vec3 currentPosition, uint vertexIndex,
                             uint globalInstanceId, uint packedVertexFormat,
                             out vec3 previousPosition) {
  previousPosition = currentPosition;
  const bool currentDrawAnimated =
      packedVertexFormat == kPackedVertexFormatAnimatedFloat24 ||
      packedVertexFormat == kPackedVertexFormatAnimatedFloat32;
  if (!currentDrawAnimated) {
    return true;
  }
  if (globalInstanceId >= pc.velocityFrameData.data.previousGeometryInfo.x) {
    return false;
  }

  VelocityRenderableGeometryData previousGeometry =
      pc.velocityFrameData.data.previousGeometry.values[globalInstanceId];
  const bool hasPreviousVertexBuffer =
      (previousGeometry.metadata.x &
       kVelocityGeometryFlagPreviousVertexBuffer) != 0u;
  const uint previousVertexCount = previousGeometry.metadata.z;
  if (!hasPreviousVertexBuffer || vertexIndex >= previousVertexCount) {
    return false;
  }

  previousPosition =
      decodeAnimatedPositionFrom(previousGeometry.previousVertexBuffer,
                                 vertexIndex, previousGeometry.metadata.y);
  return true;
}

void writeMeshletVelocityVertex(uint outputIndex, uint vertexIndex,
                                InstanceData inst, MeshletBatchGpuData batch,
                                uint globalInstanceId) {
  const uint packedVertexFormat = batch.mesh.x;
  const vec3 pos =
      decodePackedPositionFrom(batch.vertexBuffer, batch.vertexDecodeBuffer,
                               batch.draw.w, packedVertexFormat, vertexIndex);
  const vec2 uv0 =
      decodePackedUvFrom(batch.vertexBuffer, packedVertexFormat, vertexIndex);
  const vec2 uv1 =
      decodePackedUv1From(batch.vertexBuffer, packedVertexFormat, vertexIndex);

  const vec4 localPos = vec4(pos, 1.0);
  const vec4 worldPos4 = inst.modelMatrix * localPos;
  const vec4 currentClipNoJitter =
      pc.velocityFrameData.data.currentViewProjNoJitter * worldPos4;

  uint velocityFlags = meshletVelocityFlags(globalInstanceId);
  vec4 previousClipNoJitter = currentClipNoJitter;
  if (velocityFlags != 0u) {
    vec3 previousPos;
    if (!meshletPreviousPosition(pos, vertexIndex, globalInstanceId,
                                 packedVertexFormat, previousPos)) {
      velocityFlags = 0u;
    } else {
      const InstanceData previousInst =
          pc.velocityFrameData.data.previousInstanceMatrices
              .instances[globalInstanceId];
      previousClipNoJitter =
          pc.velocityFrameData.data.previousViewProjNoJitter *
          previousInst.modelMatrix * vec4(previousPos, 1.0);
    }
  }

  gl_MeshVerticesEXT[outputIndex].gl_Position =
      pc.frameData.proj * pc.frameData.view * worldPos4;

  vtx[outputIndex].uv0 = uv0;
  vtx[outputIndex].uv1 = uv1;
  vtx[outputIndex].worldNormal = vec3(0.0, 1.0, 0.0);
  vtx[outputIndex].worldTangent = vec4(1.0, 0.0, 0.0, 1.0);
  vtx[outputIndex].worldPos = worldPos4.xyz;
  vtx[outputIndex].patchBarycentric = vec3(0.0);
  vtx[outputIndex].triBarycentric = vec3(0.0);
  vtx[outputIndex].patchOuterFactors = vec3(1.0);
  vtx[outputIndex].patchInnerFactor = 1.0;
  vtx[outputIndex].tessellatedFlag = 0.0;
  outCurrentClipNoJitter[outputIndex] = currentClipNoJitter;
  outPreviousClipNoJitter[outputIndex] = previousClipNoJitter;
  outVelocityFlags[outputIndex] = velocityFlags;
  meshletMaterialIndex[outputIndex] = batch.draw.z;
}

void main() {
  const uint lane = gl_LocalInvocationIndex;
  const uint payloadIndex = gl_WorkGroupID.x;
  const uint batchIndex = meshletPayload.batchIndex[payloadIndex];
  MeshletBatchGpuData batch = pc.meshletBatches.values[batchIndex];
  const uint meshletIndex = meshletPayload.meshletIndex[payloadIndex];
  const uint globalInstanceId = meshletPayload.globalInstanceId[payloadIndex];
  const MeshletDescriptorGpuData meshlet = batch.meshlets.values[meshletIndex];
  const uint vertexOffset = meshlet.offsetsCounts.x;
  const uint primitiveOffset = meshlet.offsetsCounts.y;
  const uint vertexCount = meshlet.offsetsCounts.z;
  const uint primitiveCount = meshlet.offsetsCounts.w;
  const InstanceData inst = pc.instanceMatrices.instances[globalInstanceId];

  SetMeshOutputsEXT(vertexCount, primitiveCount);

  for (uint vertex = lane; vertex < vertexCount; vertex += gl_WorkGroupSize.x) {
    const uint localVertexIndex =
        batch.meshletVertices.indices[vertexOffset + vertex];
    writeMeshletVelocityVertex(vertex, batch.mesh.y + localVertexIndex, inst,
                               batch, globalInstanceId);
  }

  for (uint primitive = lane; primitive < primitiveCount;
       primitive += gl_WorkGroupSize.x) {
    const uint packed =
        batch.meshletPrimitives.indices[primitiveOffset + primitive];
    const uint i0 = packed & 0xffu;
    const uint i1 = (packed >> 8u) & 0xffu;
    const uint i2 = (packed >> 16u) & 0xffu;
    gl_PrimitiveTriangleIndicesEXT[primitive] = uvec3(i0, i1, i2);
  }
}
