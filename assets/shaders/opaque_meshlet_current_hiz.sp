// Same-frame depth contains the candidates themselves. Reject only across a
// locally coherent surface; layered depth is visibility-uncertain.
const float kMeshletCurrentFrameOcclusionMaxDepthRange = 0.002;
// Level zero reduces 2x2 source pixels. A 1.5-texel guard therefore covers
// three source-depth pixels around the projected bound.
const float kMeshletCurrentFrameOcclusionPaddingTexels = 1.5;
const uint kMeshletCurrentFrameVerificationMaxTexelSpan = 8u;
const float kMeshletCurrentFrameVerificationPaddingTexels = 1.0;

bool currentFrameDepthConfirmsOcclusion(vec2 projectedUvMin,
                                        vec2 projectedUvMax,
                                        float nearestDepth,
                                        float depthBias) {
  const uint texId = pc.currentDepthVerificationTexId;
  if (texId == kInvalidTextureBindlessIndex) {
    return false;
  }
  const uint extentPacked = pc.currentDepthVerificationExtentPacked;
  const ivec2 extent = ivec2(extentPacked & 0xffffu, extentPacked >> 16u);
  if (extent.x <= 0 || extent.y <= 0) {
    return false;
  }

  const vec2 textureSize = vec2(extent);
  const vec2 padding =
      vec2(kMeshletCurrentFrameVerificationPaddingTexels) / textureSize;
  const vec2 uvMin = max(projectedUvMin - padding, vec2(0.0));
  const vec2 uvMax = min(projectedUvMax + padding, vec2(1.0));
  const ivec2 texelMin =
      clamp(ivec2(floor(uvMin * textureSize)), ivec2(0), extent - ivec2(1));
  const ivec2 texelMax =
      clamp(ivec2(floor(uvMax * textureSize)), ivec2(0), extent - ivec2(1));
  const ivec2 texelSpan = texelMax - texelMin + ivec2(1);
  if (texelSpan.x <= 0 || texelSpan.y <= 0 ||
      texelSpan.x > int(kMeshletCurrentFrameVerificationMaxTexelSpan) ||
      texelSpan.y > int(kMeshletCurrentFrameVerificationMaxTexelSpan)) {
    return false;
  }

  float occluderMinDepth = 1.0;
  float occluderMaxDepth = 0.0;
  for (uint y = 0u; y < kMeshletCurrentFrameVerificationMaxTexelSpan; ++y) {
    if (y >= uint(texelSpan.y)) {
      break;
    }
    for (uint x = 0u; x < kMeshletCurrentFrameVerificationMaxTexelSpan; ++x) {
      if (x >= uint(texelSpan.x)) {
        break;
      }
      const ivec2 texel = texelMin + ivec2(x, y);
      const vec2 uv = (vec2(texel) + vec2(0.5)) / textureSize;
      const float depth =
          texture(nonuniformEXT(sampler2D(kTextures2D[nonuniformEXT(texId)],
                                          kSamplers[nonuniformEXT(
                                              pc.frameData.sceneDepthSamplerId)])),
                  uv)
              .r;
      occluderMinDepth = min(occluderMinDepth, depth);
      occluderMaxDepth = max(occluderMaxDepth, depth);
    }
  }
  return occluderMaxDepth - occluderMinDepth <=
             kMeshletCurrentFrameOcclusionMaxDepthRange &&
         nearestDepth > occluderMaxDepth + depthBias;
}
