layout(location = 0) in vec2 uv;
layout(location = 0) out vec2 out_FragColor;

layout(push_constant) uniform DepthPyramidPushConstants {
  uint sourceTexId;
  uint sourceSamplerId;
  uint sourceIsRawDepth;
  uint reserved0;
}
pc;

vec2 sampleMinMax(ivec2 coord, ivec2 sourceSize) {
  ivec2 clampedCoord = clamp(coord, ivec2(0), sourceSize - ivec2(1));
  vec2 sourceUv =
      (vec2(clampedCoord) + vec2(0.5)) / max(vec2(sourceSize), vec2(1.0));
  if (pc.sourceIsRawDepth != 0u) {
    float depth =
        textureBindless2D(pc.sourceTexId, pc.sourceSamplerId, sourceUv).r;
    return vec2(depth, depth);
  }
  return textureBindless2D(pc.sourceTexId, pc.sourceSamplerId, sourceUv).rg;
}

void main() {
  ivec2 sourceSize = max(textureBindlessSize2D(pc.sourceTexId), ivec2(1));
  // Reduction is addressed in texels; scene-copy UVs have a different Y
  // contract and are not valid for constructing this hierarchy.
  const ivec2 sourceBase = ivec2(gl_FragCoord.xy) * 2;

  vec2 s00 = sampleMinMax(sourceBase + ivec2(0, 0), sourceSize);
  vec2 s10 = sampleMinMax(sourceBase + ivec2(1, 0), sourceSize);
  vec2 s01 = sampleMinMax(sourceBase + ivec2(0, 1), sourceSize);
  vec2 s11 = sampleMinMax(sourceBase + ivec2(1, 1), sourceSize);

  out_FragColor = vec2(min(min(s00.x, s10.x), min(s01.x, s11.x)),
                       max(max(s00.y, s10.y), max(s01.y, s11.y)));
}
