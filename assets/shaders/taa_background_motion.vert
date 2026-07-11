layout(location = 0) out vec2 outUv;

vec2 fullscreenTriangleUv(uint vertexIndex) {
  return vec2((vertexIndex << 1u) & 2u, vertexIndex & 2u);
}

void main() {
  const vec2 clipUv = fullscreenTriangleUv(uint(gl_VertexIndex));
  outUv = vec2(clipUv.x, 2.0 - clipUv.y);
  // Equal-depth testing against the cleared depth value limits the fragment
  // shader to sky/background pixels without sampling depth in the shader.
  gl_Position = vec4(clipUv * 2.0 - 1.0, 1.0, 1.0);
}
