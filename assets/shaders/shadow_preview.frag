#version 460

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_samplerless_texture_functions : require

layout(set = 0, binding = 0) uniform texture2D kTextures2D[];
layout(location = 0) out vec4 out_FragColor;

const uint FLAG_INVERT = 1u;
const uint FLAG_LOG_SCALE = 2u;
const uint FLAG_TILED = 4u;
const uint MAX_PREVIEW_SOURCES = 4u;

layout(push_constant) uniform PreviewPushConstants {
  uvec4 sourceTexIds;
  uvec4 previewParams;
  vec4 depthParams;
} pc;

float fetchDepth(uint sourceTexId, vec2 tileUv) {
  ivec2 sourceSize = textureSize(nonuniformEXT(kTextures2D[sourceTexId]), 0);
  vec2 clampedUv = clamp(tileUv, vec2(0.0), vec2(0.99999994));
  ivec2 texelCoord =
      clamp(ivec2(clampedUv * vec2(sourceSize)), ivec2(0), sourceSize - 1);
  return texelFetch(nonuniformEXT(kTextures2D[sourceTexId]), texelCoord, 0).r;
}

void main() {
  vec2 previewSize = max(vec2(pc.previewParams.xy), vec2(1.0));
  uint sourceCount = clamp(pc.previewParams.z, 1u, MAX_PREVIEW_SOURCES);
  uint flags = pc.previewParams.w;
  float depth = 0.0;

  if ((flags & FLAG_TILED) != 0u) {
    vec2 tileSize = previewSize * 0.5;
    vec2 fragPos = clamp(gl_FragCoord.xy, vec2(0.0), previewSize - vec2(1.0));
    uvec2 tileCoord =
        uvec2(clamp(floor(fragPos / tileSize), vec2(0.0), vec2(1.0)));
    uint tileIndex = tileCoord.x + tileCoord.y * 2u;
    if (tileIndex >= sourceCount) {
      out_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
      return;
    }
    vec2 tileOrigin = vec2(tileCoord) * tileSize;
    vec2 tileUv = (fragPos - tileOrigin) / max(tileSize, vec2(1.0));
    depth = fetchDepth(pc.sourceTexIds[tileIndex], tileUv);
  } else {
    depth = fetchDepth(pc.sourceTexIds.x, gl_FragCoord.xy / previewSize);
  }

  float preview = clamp(depth * pc.depthParams.x + pc.depthParams.y, 0.0, 1.0);
  if ((flags & FLAG_LOG_SCALE) != 0u) {
    preview = log2(1.0 + preview * 255.0) / 8.0;
  }
  if ((flags & FLAG_INVERT) != 0u) {
    preview = 1.0 - preview;
  }
  out_FragColor = vec4(vec3(preview), 1.0);
}
