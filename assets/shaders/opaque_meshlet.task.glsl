#extension GL_EXT_mesh_shader : require

#define NURI_OPAQUE_MESHLET_BATCHED 1
#include "meshlet_common.sp"

layout(local_size_x = 1) in;

taskPayloadSharedEXT MeshletTaskPayload meshletPayload;

float meshletInstanceMaxScale(mat4 model) {
  return max(max(length(model[0].xyz), length(model[1].xyz)),
             length(model[2].xyz));
}

vec4 matrixRow(mat4 value, uint row) {
  return vec4(value[0][row], value[1][row], value[2][row], value[3][row]);
}

bool sphereOutsideViewPlane(vec4 plane, vec4 viewCenter, float radius) {
  const float signedDistance = dot(plane, viewCenter);
  const float planeRadius = radius * length(plane.xyz);
  return signedDistance < -planeRadius;
}

bool meshletFrustumCull(MeshletDescriptorGpuData meshlet, InstanceData inst,
                        uint meshletFlags) {
  if ((meshletFlags & kMeshletFlagFrustumCulling) == 0u) {
    return false;
  }

  const vec3 localCenter = meshlet.boundsSphere.xyz;
  const float localRadius = max(meshlet.boundsSphere.w, 0.0);
  const vec4 worldCenter4 = inst.modelMatrix * vec4(localCenter, 1.0);
  const vec4 viewCenter = pc.frameData.view * worldCenter4;
  const float radius = localRadius * meshletInstanceMaxScale(inst.modelMatrix);

  const vec4 clipRowX = matrixRow(pc.frameData.proj, 0u);
  const vec4 clipRowY = matrixRow(pc.frameData.proj, 1u);
  const vec4 clipRowZ = matrixRow(pc.frameData.proj, 2u);
  const vec4 clipRowW = matrixRow(pc.frameData.proj, 3u);

  return sphereOutsideViewPlane(clipRowW + clipRowX, viewCenter, radius) ||
         sphereOutsideViewPlane(clipRowW - clipRowX, viewCenter, radius) ||
         sphereOutsideViewPlane(clipRowW + clipRowY, viewCenter, radius) ||
         sphereOutsideViewPlane(clipRowW - clipRowY, viewCenter, radius) ||
         sphereOutsideViewPlane(clipRowW + clipRowZ, viewCenter, radius) ||
         sphereOutsideViewPlane(clipRowW - clipRowZ, viewCenter, radius);
}

bool meshletConeCull(MeshletDescriptorGpuData meshlet, InstanceData inst,
                     uint meshletFlags) {
  if ((meshletFlags & kMeshletFlagConeCulling) == 0u ||
      (meshletFlags & kMeshletFlagDoubleSided) != 0u) {
    return false;
  }

  const vec3 localAxis = meshlet.coneAxisCutoff.xyz;
  if (dot(localAxis, localAxis) <= 1.0e-10) {
    return false;
  }
  const mat3 modelBasis = mat3(inst.modelMatrix);
  const float handedness =
      dot(cross(modelBasis[0], modelBasis[1]), modelBasis[2]);
  if (handedness < 0.0) {
    return false;
  }
  const mat3 normalMatrix =
      mat3(inst.normalMatCol0.xyz, inst.normalMatCol1.xyz,
           inst.normalMatCol2.xyz);
  const vec3 transformedAxis = normalMatrix * localAxis;
  if (dot(transformedAxis, transformedAxis) <= 1.0e-10) {
    return false;
  }
  const vec3 worldAxis = normalize(transformedAxis);
  const vec3 worldApex =
      (inst.modelMatrix * vec4(meshlet.coneApex.xyz, 1.0)).xyz;
  const vec3 cameraToApex = worldApex - pc.frameData.cameraPos.xyz;
  const float distanceSq = dot(cameraToApex, cameraToApex);
  if (distanceSq <= 1.0e-10) {
    return false;
  }
  const vec3 viewDir = cameraToApex * inversesqrt(distanceSq);
  return dot(viewDir, worldAxis) >= meshlet.coneAxisCutoff.w + 0.02;
}

uint resolveAvailableMeshletLod(SubmeshMeshletLodRangeGpuData lodRange,
                                uint requestedLod) {
  const uint lodCount = clamp(lodRange.metadata.x, 1u, 4u);
  uint candidate = min(requestedLod, lodCount - 1u);
  while (candidate > 0u && lodRange.meshletCount[candidate] == 0u) {
    --candidate;
  }
  return candidate;
}

uint meshletCandidateSpan(SubmeshMeshletLodRangeGpuData lodRange) {
  return max(max(lodRange.meshletCount.x, lodRange.meshletCount.y),
             max(lodRange.meshletCount.z, lodRange.meshletCount.w));
}

uint meshletForcedLod(uint meshletFlags) {
  return (meshletFlags >> kMeshletFlagForcedLodShift) &
         kMeshletFlagForcedLodMask;
}

uint selectMeshletLod(SubmeshMeshletLodRangeGpuData lodRange,
                      uint globalInstanceId, uint meshletFlags) {
  if ((meshletFlags & kMeshletFlagGpuLod) == 0u) {
    return resolveAvailableMeshletLod(lodRange,
                                      meshletForcedLod(meshletFlags));
  }

  const vec4 lodBounds = pc.instanceLodBounds.values[globalInstanceId];
  const vec3 delta = pc.frameData.cameraPos.xyz - lodBounds.xyz;
  const float normalizedDistanceSq = dot(delta, delta) * max(lodBounds.w, 0.0);
  const vec3 thresholdsSq = pc.lodThresholds.xyz * pc.lodThresholds.xyz;

  uint requestedLod = 0u;
  if (normalizedDistanceSq >= thresholdsSq.z) {
    requestedLod = 3u;
  } else if (normalizedDistanceSq >= thresholdsSq.y) {
    requestedLod = 2u;
  } else if (normalizedDistanceSq >= thresholdsSq.x) {
    requestedLod = 1u;
  }
  return resolveAvailableMeshletLod(lodRange, requestedLod);
}

void main() {
  const uint batchIndex = pc.batchBase + gl_WorkGroupID.y;
  MeshletBatchGpuData batch = pc.meshletBatches.values[batchIndex];
  const uint meshletFlags = batch.flags.x;
  const SubmeshMeshletLodRangeGpuData lodRange =
      batch.meshletLodRanges.values[batch.mesh.z];
  const uint candidateSpan = batch.mesh.w;
  const uint instanceCount = batch.draw.x;
  if (candidateSpan == 0u || instanceCount == 0u) {
    EmitMeshTasksEXT(0u, 1u, 1u);
    return;
  }

  const uint candidate = pc.candidateOffset + gl_WorkGroupID.x;
  const uint instanceLocal = candidate / candidateSpan;
  if (instanceLocal >= instanceCount) {
    EmitMeshTasksEXT(0u, 1u, 1u);
    return;
  }

  const uint meshletLocal = candidate - instanceLocal * candidateSpan;
  const uint remapIndex = batch.draw.y + instanceLocal;
  const uint globalInstanceId = pc.instanceRemap.ids[remapIndex];
  const uint selectedLod =
      selectMeshletLod(lodRange, globalInstanceId, meshletFlags);
  const uint selectedMeshletCount = lodRange.meshletCount[selectedLod];
  if (meshletLocal >= selectedMeshletCount) {
    EmitMeshTasksEXT(0u, 1u, 1u);
    return;
  }

  const uint meshletIndex = lodRange.meshletOffset[selectedLod] + meshletLocal;
  const InstanceData inst = pc.instanceMatrices.instances[globalInstanceId];
  const MeshletDescriptorGpuData meshlet = batch.meshlets.values[meshletIndex];
  if (meshletFrustumCull(meshlet, inst, meshletFlags) ||
      meshletConeCull(meshlet, inst, meshletFlags)) {
    EmitMeshTasksEXT(0u, 1u, 1u);
    return;
  }

  meshletPayload.meshletIndex = meshletIndex;
  meshletPayload.globalInstanceId = globalInstanceId;
  meshletPayload.selectedLod = selectedLod;
  meshletPayload.batchIndex = batchIndex;
  EmitMeshTasksEXT(1u, 1u, 1u);
}
