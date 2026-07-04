#define NURI_MESHLET_COMMON 1
#include "common.sp"

const uint kMeshletFlagFrustumCulling = 1u << 0u;
const uint kMeshletFlagConeCulling = 1u << 1u;
const uint kMeshletFlagDoubleSided = 1u << 2u;
const uint kMeshletFlagDebugMeshletId = 1u << 3u;
const uint kMeshletFlagGpuLod = 1u << 4u;
const uint kMeshletFlagDebugSelectedLod = 1u << 5u;
const uint kMeshletFlagShadowCascadeCulling = 1u << 6u;
const uint kMeshletFlagOcclusionCulling = 1u << 7u;
const uint kMeshletFlagForcedLodShift = 8u;
const uint kMeshletFlagForcedLodMask = 0x3u;
const uint kOpaqueMeshletTaskPayloadCapacity = 32u;

float meshletInstanceMaxScale(mat4 model) {
  return max(max(length(model[0].xyz), length(model[1].xyz)),
             length(model[2].xyz));
}

struct MeshletTaskPayload {
#ifdef NURI_OPAQUE_MESHLET_BATCHED
  uint visibleCount;
  uint meshletIndex[kOpaqueMeshletTaskPayloadCapacity];
  uint globalInstanceId[kOpaqueMeshletTaskPayloadCapacity];
  uint selectedLod[kOpaqueMeshletTaskPayloadCapacity];
  uint batchIndex[kOpaqueMeshletTaskPayloadCapacity];
#else
  uint meshletIndex;
  uint globalInstanceId;
  uint selectedLod;
#endif
};
