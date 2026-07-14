#extension GL_EXT_mesh_shader : require

#define NURI_OPAQUE_MESHLET_BATCHED 1
#include "meshlet_common.sp"
#include "opaque_meshlet_vertex.sp"

layout(local_size_x = 128) in;
layout(triangles, max_vertices = 64, max_primitives = 124) out;

out gl_MeshPerVertexEXT { vec4 gl_Position; }
gl_MeshVerticesEXT[];

layout(location = 0) out OpaqueMeshletVertex vtx[];
layout(location = 5) flat out uint meshletDebugId[];
layout(location = 6) flat out uint meshletDebugLod[];
layout(location = 7) flat out uint meshletMaterialIndex[];
layout(location = 8) flat out uint meshletFlags[];

taskPayloadSharedEXT MeshletTaskPayload meshletPayload;

void writeMeshletVertex(uint outputIndex, uint vertexIndex, InstanceData inst,
                        MeshletBatchGpuData batch, uint meshletIndex,
                        uint selectedLod) {
  const uint packedVertexFormat = batch.mesh.x;
  const vec3 pos =
      decodePackedPositionFrom(batch.vertexBuffer, batch.vertexDecodeBuffer,
                               batch.draw.w, packedVertexFormat, vertexIndex);
  const vec3 normal = decodePackedNormalFrom(batch.vertexBuffer,
                                             packedVertexFormat, vertexIndex);
  const vec4 tangent = decodePackedTangentFrom(batch.vertexBuffer,
                                               packedVertexFormat, vertexIndex);
  const vec2 uv0 =
      decodePackedUvFrom(batch.vertexBuffer, packedVertexFormat, vertexIndex);
  const vec2 uv1 =
      decodePackedUv1From(batch.vertexBuffer, packedVertexFormat, vertexIndex);

  const mat4 model = inst.modelMatrix;
  const vec4 worldPos4 = model * vec4(pos, 1.0);
  gl_MeshVerticesEXT[outputIndex].gl_Position =
      pc.frameData.proj * pc.frameData.view * worldPos4;

  const mat3 normalMatrix = mat3(inst.normalMatCol0.xyz, inst.normalMatCol1.xyz,
                                 inst.normalMatCol2.xyz);
  const vec3 worldNormal = normalize(normalMatrix * normal);
  vec3 transformedTangent = mat3(model) * tangent.xyz;
  transformedTangent -= worldNormal * dot(transformedTangent, worldNormal);

  vtx[outputIndex].uv0 = uv0;
  vtx[outputIndex].uv1 = uv1;
  vtx[outputIndex].worldNormal = worldNormal;
  vtx[outputIndex].worldTangent =
      dot(transformedTangent, transformedTangent) > 1.0e-10
          ? vec4(normalize(transformedTangent), tangent.w)
          : vec4(0.0, 0.0, 0.0, 1.0);
  vtx[outputIndex].worldPos = worldPos4.xyz;
  meshletDebugId[outputIndex] = meshletIndex;
  meshletDebugLod[outputIndex] = selectedLod;
  meshletMaterialIndex[outputIndex] = batch.draw.z;
  meshletFlags[outputIndex] = batch.flags.x;
}

void main() {
  const uint lane = gl_LocalInvocationIndex;
  const uint payloadIndex = gl_WorkGroupID.x;
  const uint batchIndex = meshletPayload.batchIndex[payloadIndex];
  MeshletBatchGpuData batch = pc.meshletBatches.values[batchIndex];
  const uint meshletIndex = meshletPayload.meshletIndex[payloadIndex];
  const uint globalInstanceId = meshletPayload.globalInstanceId[payloadIndex];
  const uint selectedLod = meshletPayload.selectedLod[payloadIndex];
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
    writeMeshletVertex(vertex, batch.mesh.y + localVertexIndex, inst, batch,
                       meshletIndex, selectedLod);
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
