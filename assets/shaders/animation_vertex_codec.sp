struct PackedVertex {
  uint word0;
  uint word1;
  uint word2;
  uint word3;
  uint word4;
  uint word5;
};

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

vec2 encodeOctNormal(vec3 normal) {
  const float lenSq = dot(normal, normal);
  if (lenSq <= 1.0e-10) {
    return vec2(0.0);
  }
  normal = normalize(normal);
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

vec2 decodePackedUv(PackedVertex vertex) { return unpackHalf2x16(vertex.word4); }
vec2 decodePackedUv1(PackedVertex vertex) { return unpackHalf2x16(vertex.word5); }

vec3 decodePackedNormal(PackedVertex vertex) {
  return decodeOctNormal(unpackSnorm2x16Custom(vertex.word3));
}

PackedVertex encodePackedVertex(vec3 position, vec3 normal, vec2 uv0, vec2 uv1) {
  PackedVertex packed;
  vec3 safeNormal = dot(normal, normal) > 1.0e-10 ? normalize(normal)
                                                  : vec3(0.0, 1.0, 0.0);
  vec2 oct = encodeOctNormal(safeNormal);
  packed.word0 = floatBitsToUint(position.x);
  packed.word1 = floatBitsToUint(position.y);
  packed.word2 = floatBitsToUint(position.z);
  packed.word3 = packSnorm2x16(oct);
  packed.word4 = packHalf2x16(uv0);
  packed.word5 = packHalf2x16(uv1);
  return packed;
}
