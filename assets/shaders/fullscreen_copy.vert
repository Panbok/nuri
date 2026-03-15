layout(location = 0) out vec2 outUv;

vec2 fullscreenTriangleUv(uint vertexIndex) {
  return vec2((vertexIndex << 1u) & 2u, vertexIndex & 2u);
}

void main() {
  vec2 clipUv = fullscreenTriangleUv(uint(gl_VertexIndex));
  // This copy path samples the offscreen scene-color texture using the
  // framebuffer-space convention expected by the transmission copy pass.
  outUv = vec2(clipUv.x, 2.0 - clipUv.y);
  gl_Position = vec4(clipUv * 2.0 - 1.0, 0.0, 1.0);
}
