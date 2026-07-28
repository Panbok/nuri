const uint kDDGIFrameGpuDataVersion = 5u;
const uint kDDGIGatherVariantProduct = 0u;
const uint kDDGIGatherVariantBypass = 1u;
const uint kDDGIGatherVariantCandidates = 2u;
const uint kDDGIGatherVariantProbeVisibility = 3u;
const uint kDDGIGatherVariantAtlas = 4u;
const uint kDDGIMaxVolumes = 8u;
const uint kDDGIMaxSampledVolumes = 2u;
const uint kDDGIEffectiveKindAuthored = 0u;
const uint kDDGIEffectiveKindSceneFit = 1u;
const uint kDDGIEffectiveKindClipmapCascade = 2u;
const uint kDDGIProbeStateUninitialized = 0u;
const uint kDDGIProbeStateOff = 1u;
const uint kDDGIProbeStateSleeping = 2u;
const uint kDDGIProbeStateNewlyAwake = 3u;
const uint kDDGIProbeStateAwake = 4u;
const uint kDDGIProbeStateNewlyVigilant = 5u;
const uint kDDGIProbeStateVigilant = 6u;
const uint kDDGIProbeUpdateReasonBootstrap = 1u << 8u;

struct DDGIProbeStateGpuData {
  vec4 relocation;
  // x: state, y: last submitted sequence, z: classification iteration,
  // w: reserved.
  uvec4 stateAgeFlags;
};

struct DDGIProbeUpdateEntryGpu {
  uint volumeSlot;
  uint probeId;
  uint rayBase;
  uint rayCount;
  uint flags;
  uint resultState;
  uint resultSubmittedSequence;
  uint resultIteration;
  vec4 resultRelocation;
};

struct DDGIRayResultGpuData {
  vec4 radianceAndDistance;
  uvec4 metadata;
};

layout(std430, buffer_reference) readonly buffer DDGIProbeStateBuffer {
  DDGIProbeStateGpuData values[];
};

struct DDGIVolumeGpuData {
  mat4 worldFromLocal;
  mat4 localFromWorld;
  DDGIProbeStateBuffer probeStates;
  // x: resource flags; y high/low 16: trace-light subset offset/count.
  uvec2 resourceFlagsLocalLightSubset;
  vec4 probeSpacingAndBias;
  vec4 rayBiases;
  vec4 centerHalfExtentsAndMaxDistance;
  uvec4 probeCountsAndCount;
  uvec4 irradianceAtlas;
  uvec4 distanceAtlas;
  uvec4 ringOriginAndFlags;
  uvec4 generations;
  uvec4 effectiveIdentity;
  uvec4 tierTransitionCoverageFlags;
  vec4 continuousCameraLocal;
  vec4 fadeStartHalfExtents;
  vec4 fadeEndHalfExtents;
};

layout(std430, buffer_reference) readonly buffer DDGIFrameBuffer {
  uvec4 activeCountDebugFlagsSampler;
  DDGIVolumeGpuData volumes[kDDGIMaxVolumes];
};
