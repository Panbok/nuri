layout(location = 0) out vec2 outUv;

vec2 fullscreenTriangleUv(uint vertexIndex) {
  return vec2((vertexIndex << 1u) & 2u, vertexIndex & 2u);
}

void main() {
  outUv = fullscreenTriangleUv(uint(gl_VertexIndex));
  gl_Position = vec4(outUv * 2.0 - 1.0, 0.0, 1.0);
}
