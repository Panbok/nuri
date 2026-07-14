struct OpaqueMeshletVertex {
  vec2 uv0;
  vec2 uv1;
  vec3 worldNormal;
  vec4 worldTangent;
  vec3 worldPos;
};

PerVertex opaqueMeshletMaterialVertex(OpaqueMeshletVertex vertex) {
  return PerVertex(vertex.uv0, vertex.uv1, vertex.worldNormal,
                   vertex.worldTangent, vertex.worldPos, vec3(0.0), vec3(0.0),
                   vec3(1.0), 1.0, 0.0);
}
