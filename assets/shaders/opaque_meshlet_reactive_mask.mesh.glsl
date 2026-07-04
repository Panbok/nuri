#extension GL_EXT_mesh_shader : require

#define NURI_OPAQUE_MESHLET_BATCHED 1
#include "meshlet_common.sp"

layout(local_size_x = 32) in;
layout(triangles, max_vertices = 64, max_primitives = 124) out;

out gl_MeshPerVertexEXT {
  vec4 gl_Position;
} gl_MeshVerticesEXT[];

layout(location = 0) out PerVertex vtx[];
layout(location = 10) flat out uint outMotionReactive[];
layout(location = 11) flat out uint meshletMaterialIndex[];

taskPayloadSharedEXT MeshletTaskPayload meshletPayload;

uint meshletReactiveVelocityFlags(uint globalInstanceId) {
  const uint instanceFlagsMode = pc.velocityFrameData.data.instanceFlagsMode.x;
  if (instanceFlagsMode == kVelocityInstanceFlagsModeAllInvalid) {
    return 0u;
  }
  if (instanceFlagsMode == kVelocityInstanceFlagsModeBuffer) {
    return pc.velocityInstanceFlags.flags[globalInstanceId];
  }
  return 1u;
}

void writeMeshletReactiveVertex(uint outputIndex, uint vertexIndex,
                                InstanceData inst,
                                MeshletBatchGpuData batch,
                                uint globalInstanceId) {
  const uint packedVertexFormat = batch.mesh.x;
  const vec3 pos = decodePackedPositionFrom(
      batch.vertexBuffer, batch.vertexDecodeBuffer, batch.draw.w,
      packedVertexFormat, vertexIndex);
  const vec2 uv0 =
      decodePackedUvFrom(batch.vertexBuffer, packedVertexFormat, vertexIndex);
  const vec2 uv1 =
      decodePackedUv1From(batch.vertexBuffer, packedVertexFormat, vertexIndex);
  const vec4 worldPos4 = inst.modelMatrix * vec4(pos, 1.0);

  gl_MeshVerticesEXT[outputIndex].gl_Position =
      pc.frameData.proj * pc.frameData.view * worldPos4;

  const uint velocityFlags = meshletReactiveVelocityFlags(globalInstanceId);
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
  outMotionReactive[outputIndex] = velocityFlags == 0u ? 1u : 0u;
  meshletMaterialIndex[outputIndex] = batch.draw.z;
}

void main() {
  const uint lane = gl_LocalInvocationIndex;
  const uint payloadIndex = gl_WorkGroupID.x;
  const uint batchIndex = meshletPayload.batchIndex[payloadIndex];
  MeshletBatchGpuData batch = pc.meshletBatches.values[batchIndex];
  const uint meshletIndex = meshletPayload.meshletIndex[payloadIndex];
  const uint globalInstanceId =
      meshletPayload.globalInstanceId[payloadIndex];
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
    writeMeshletReactiveVertex(vertex, batch.mesh.y + localVertexIndex, inst,
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
