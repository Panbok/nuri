#extension GL_EXT_mesh_shader : require

#define NURI_MESHLET_TASK_PAYLOAD_BATCHED 1
#include "meshlet_common.sp"

layout(local_size_x = 32) in;
layout(triangles, max_vertices = 64, max_primitives = 124) out;

out gl_MeshPerVertexEXT { vec4 gl_Position; }
gl_MeshVerticesEXT[];

taskPayloadSharedEXT MeshletTaskPayload meshletPayload;

uint shadowCascadeIndex() { return floatBitsToUint(pc.lodThresholds.z); }

void writeMeshletVertex(uint outputIndex, uint vertexIndex, InstanceData inst) {
  const vec3 pos = decodePackedPosition(vertexIndex);
  const mat4 lightViewProj =
      pc.frameData.shadowFrameBuffer.cascades[shadowCascadeIndex()]
          .lightViewProj;

  gl_MeshVerticesEXT[outputIndex].gl_Position =
      lightViewProj * inst.modelMatrix * vec4(pos, 1.0);
}

void main() {
  const uint lane = gl_LocalInvocationIndex;
  const uint payloadIndex = gl_WorkGroupID.x;
  const uint meshletIndex = meshletPayload.meshletIndex[payloadIndex];
  const uint globalInstanceId = meshletPayload.globalInstanceId[payloadIndex];
  const MeshletDescriptorGpuData meshlet = pc.meshlets.values[meshletIndex];
  const uint vertexOffset = meshlet.offsetsCounts.x;
  const uint primitiveOffset = meshlet.offsetsCounts.y;
  const uint vertexCount = meshlet.offsetsCounts.z;
  const uint primitiveCount = meshlet.offsetsCounts.w;
  const InstanceData inst = pc.instanceMatrices.instances[globalInstanceId];

  SetMeshOutputsEXT(vertexCount, primitiveCount);

  for (uint vertex = lane; vertex < vertexCount; vertex += gl_WorkGroupSize.x) {
    const uint localVertexIndex =
        pc.meshletVertices.indices[vertexOffset + vertex];
    writeMeshletVertex(vertex, pc.vertexOffset + localVertexIndex, inst);
  }

  for (uint primitive = lane; primitive < primitiveCount;
       primitive += gl_WorkGroupSize.x) {
    const uint packed =
        pc.meshletPrimitives.indices[primitiveOffset + primitive];
    const uint i0 = packed & 0xffu;
    const uint i1 = (packed >> 8u) & 0xffu;
    const uint i2 = (packed >> 16u) & 0xffu;
    gl_PrimitiveTriangleIndicesEXT[primitive] = uvec3(i0, i1, i2);
  }
}
