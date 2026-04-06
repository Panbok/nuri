struct PackedVertex {
  uint word0;
  uint word1;
  uint word2;
  uint word3;
  uint word4;
  uint word5;
  uint word6;
  uint word7;
  uint word8;
};

vec2 unpackSnorm2x16Custom(uint packed) {
  const int x = int(packed << 16u) >> 16;
  const int y = int(packed) >> 16;
  return clamp(vec2(float(x), float(y)) / 32767.0, vec2(-1.0), vec2(1.0));
}

vec3 decodePackedPosition(PackedVertex vertex) {
  return vec3(uintBitsToFloat(vertex.word0), uintBitsToFloat(vertex.word1),
              uintBitsToFloat(vertex.word2));
}

vec2 decodePackedUv(PackedVertex vertex) { return unpackHalf2x16(vertex.word3); }
vec2 decodePackedUv1(PackedVertex vertex) { return unpackHalf2x16(vertex.word8); }

vec3 decodePackedNormal(PackedVertex vertex) {
  const vec2 normalXY = unpackSnorm2x16Custom(vertex.word4);
  const vec2 normalZHandedness = unpackSnorm2x16Custom(vertex.word5);
  return normalize(vec3(normalXY, normalZHandedness.x));
}

vec4 decodePackedTangent(PackedVertex vertex) {
  const vec2 tangentXY = unpackSnorm2x16Custom(vertex.word6);
  const vec2 tangentZ = unpackSnorm2x16Custom(vertex.word7);
  const vec2 normalZHandedness = unpackSnorm2x16Custom(vertex.word5);
  vec3 tangent = vec3(tangentXY, tangentZ.x);
  const float tangentLen = length(tangent);
  if (tangentLen > 1.0e-6) {
    tangent /= tangentLen;
  } else {
    tangent = vec3(1.0, 0.0, 0.0);
  }
  const float handedness = normalZHandedness.y >= 0.0 ? 1.0 : -1.0;
  return vec4(tangent, handedness);
}

// normal and tangent.xyz are expected to be unit length and within [-1, 1];
// tangent.w is expected to carry only handedness sign.
PackedVertex encodePackedVertex(vec3 position, vec3 normal, vec4 tangent,
                                vec2 uv0, vec2 uv1) {
  PackedVertex packed;
  const float tangentHandedness = tangent.w >= 0.0 ? 1.0 : -1.0;
  packed.word0 = floatBitsToUint(position.x);
  packed.word1 = floatBitsToUint(position.y);
  packed.word2 = floatBitsToUint(position.z);
  packed.word3 = packHalf2x16(uv0);
  packed.word4 = packSnorm2x16(vec2(normal.x, normal.y));
  packed.word5 = packSnorm2x16(vec2(normal.z, tangentHandedness));
  packed.word6 = packSnorm2x16(vec2(tangent.x, tangent.y));
  packed.word7 = packSnorm2x16(vec2(tangent.z, 0.0));
  packed.word8 = packHalf2x16(uv1);
  return packed;
}
