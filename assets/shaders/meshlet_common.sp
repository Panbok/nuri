#define NURI_MESHLET_COMMON 1
#include "common.sp"

const uint kMeshletFlagFrustumCulling = 1u << 0u;
const uint kMeshletFlagConeCulling = 1u << 1u;
const uint kMeshletFlagDoubleSided = 1u << 2u;
const uint kMeshletFlagDebugMeshletId = 1u << 3u;
const uint kMeshletFlagGpuLod = 1u << 4u;
const uint kMeshletFlagDebugSelectedLod = 1u << 5u;
const uint kMeshletFlagForcedLodShift = 8u;
const uint kMeshletFlagForcedLodMask = 0x3u;

struct MeshletTaskPayload {
  uint meshletIndex;
  uint globalInstanceId;
  uint selectedLod;
#ifdef NURI_OPAQUE_MESHLET_BATCHED
  uint batchIndex;
#endif
};
