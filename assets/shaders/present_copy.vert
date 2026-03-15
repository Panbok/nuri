#version 460

layout(location = 0) out vec2 uv;

vec2 fullscreenTriangleUv(uint vertexIndex) {
  return vec2((vertexIndex << 1u) & 2u, vertexIndex & 2u);
}

void main() {
  uv = fullscreenTriangleUv(uint(gl_VertexIndex));
  gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
