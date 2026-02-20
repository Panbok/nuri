#version 460
#include "common.sp"

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

layout(location = 0) in PerVertex inVtx[];

layout(location = 0) out PerVertex outVtx;

void main() {
  const vec3 triBarycentrics[3] = {
      vec3(1.0, 0.0, 0.0),
      vec3(0.0, 1.0, 0.0),
      vec3(0.0, 0.0, 1.0),
  };

  for (int i = 0; i < 3; ++i) {
    outVtx = inVtx[i];
    outVtx.triBarycentric = triBarycentrics[i];
    gl_Position = gl_in[i].gl_Position;
    EmitVertex();
  }
  EndPrimitive();
}
