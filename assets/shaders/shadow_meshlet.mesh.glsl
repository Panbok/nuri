#extension GL_EXT_mesh_shader : require

#include "meshlet_common.sp"

layout(local_size_x = 32) in;
layout(triangles, max_vertices = 64, max_primitives = 124) out;

out gl_MeshPerVertexEXT { vec4 gl_Position; }
gl_MeshVerticesEXT[];

layout(location = 0) out vec2 outUv0[];
layout(location = 1) out vec2 outUv1[];

taskPayloadSharedEXT MeshletTaskPayload meshletPayload;

uint shadowCascadeIndex() { return floatBitsToUint(pc.lodThresholds.z); }

void writeMeshletVertex(uint outputIndex, uint vertexIndex, InstanceData inst) {
  const vec3 pos = decodePackedPosition(vertexIndex);
  const vec2 uv0 = decodePackedUv(vertexIndex);
  const vec2 uv1 = decodePackedUv1(vertexIndex);
  const mat4 lightViewProj =
      pc.frameData.shadowFrameBuffer.cascades[shadowCascadeIndex()]
          .lightViewProj;

  gl_MeshVerticesEXT[outputIndex].gl_Position =
      lightViewProj * inst.modelMatrix * vec4(pos, 1.0);
  outUv0[outputIndex] = uv0;
  outUv1[outputIndex] = uv1;
}

void main() {
  const uint lane = gl_LocalInvocationIndex;
  const uint meshletIndex = meshletPayload.meshletIndex;
  const uint globalInstanceId = meshletPayload.globalInstanceId;
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
