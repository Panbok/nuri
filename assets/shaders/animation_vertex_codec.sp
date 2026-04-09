struct PackedVertex {
  uint word0;
  uint word1;
  uint word2;
  uint word3;
  uint word4;
  uint word5;
  uint word6;
  uint word7;
};

const float kPackedVertexDegenerateEpsilon = 1.0e-6;

vec2 unpackSnorm2x16Custom(uint packed) {
  const int x = int(packed << 16u) >> 16;
  const int y = int(packed) >> 16;
  return clamp(vec2(float(x), float(y)) / 32767.0, vec2(-1.0), vec2(1.0));
}

vec2 signNotZero(vec2 value) {
  return vec2(value.x >= 0.0 ? 1.0 : -1.0, value.y >= 0.0 ? 1.0 : -1.0);
}

vec3 decodeOctNormal(vec2 encoded) {
  vec3 normal = vec3(encoded.xy, 1.0 - abs(encoded.x) - abs(encoded.y));
  if (normal.z < 0.0) {
    normal.xy = (vec2(1.0) - abs(normal.yx)) * signNotZero(normal.xy);
  }
  const float lenSq = dot(normal, normal);
  if (lenSq <= 1.0e-10) {
    return vec3(0.0, 1.0, 0.0);
  }
  return normalize(normal);
}

vec3 normalizePackedDirectionOrZero(vec3 value) {
  const float lenSq = dot(value, value);
  return lenSq > kPackedVertexDegenerateEpsilon ? value * inversesqrt(lenSq)
                                                : vec3(0.0);
}

vec3 orthogonalizePackedTangent(vec3 tangent, vec3 normal) {
  tangent -= normal * dot(tangent, normal);
  return normalizePackedDirectionOrZero(tangent);
}

vec2 encodeOctNormal(vec3 normal) {
  const float lenSq = dot(normal, normal);
  if (lenSq <= 1.0e-10) {
    normal = vec3(0.0, 1.0, 0.0);
  } else {
    normal *= inversesqrt(lenSq);
  }
  normal /= abs(normal.x) + abs(normal.y) + abs(normal.z);
  vec2 encoded = normal.xy;
  if (normal.z < 0.0) {
    encoded = (vec2(1.0) - abs(encoded.yx)) * signNotZero(encoded);
  }
  return clamp(encoded, vec2(-1.0), vec2(1.0));
}

vec3 decodePackedPosition(PackedVertex vertex) {
  return vec3(uintBitsToFloat(vertex.word0), uintBitsToFloat(vertex.word1),
              uintBitsToFloat(vertex.word2));
}

vec2 decodePackedUv(PackedVertex vertex) { return unpackHalf2x16(vertex.word6); }
vec2 decodePackedUv1(PackedVertex vertex) { return unpackHalf2x16(vertex.word7); }

vec3 decodePackedNormal(PackedVertex vertex) {
  return decodeOctNormal(unpackSnorm2x16Custom(vertex.word3));
}

vec4 decodePackedTangent(PackedVertex vertex) {
  if (vertex.word5 == 0u) {
    return vec4(0.0, 0.0, 0.0, 1.0);
  }
  const vec3 normal = decodePackedNormal(vertex);
  vec3 tangent = decodeOctNormal(unpackSnorm2x16Custom(vertex.word4));
  tangent = orthogonalizePackedTangent(tangent, normal);
  return vec4(tangent, uintBitsToFloat(vertex.word5));
}

PackedVertex encodePackedVertex(vec3 position, vec3 normal, vec4 tangent,
                                vec2 uv0, vec2 uv1) {
  PackedVertex packed;
  vec2 oct = encodeOctNormal(normal);
  packed.word0 = floatBitsToUint(position.x);
  packed.word1 = floatBitsToUint(position.y);
  packed.word2 = floatBitsToUint(position.z);
  packed.word3 = packSnorm2x16(oct);
  if (dot(tangent.xyz, tangent.xyz) > kPackedVertexDegenerateEpsilon) {
    packed.word4 = packSnorm2x16(encodeOctNormal(tangent.xyz));
    packed.word5 = floatBitsToUint(tangent.w >= 0.0 ? 1.0 : -1.0);
  } else {
    packed.word4 = 0u;
    packed.word5 = 0u;
  }
  packed.word6 = packHalf2x16(uv0);
  packed.word7 = packHalf2x16(uv1);
  return packed;
}
