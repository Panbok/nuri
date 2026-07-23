const uint kRtGeometryDoubleSided = 1u << 0u;
const uint kRtGeometryAlphaMasked = 1u << 1u;
const uint kRtGeometryAlphaBlended = 1u << 2u;
const uint kRtGeometryTransmission = 1u << 3u;

struct RtInstanceGpuData {
  uvec4 geometryRenderableFlags;
  mat4 worldFromObject;
  mat4 objectFromWorld;
};

struct RtGeometryGpuData {
  PackedVertexWordBuffer indexBuffer;
  PackedVertexWordBuffer vertexBuffer;
  StaticVertexDecodeBuffer vertexDecodeBuffer;
  uint materialIndex;
  uint indexOffset;
  uint vertexOffset;
  uint indexFormatAndVertexFormat;
  uint flags;
};

layout(std430, buffer_reference) readonly buffer RtInstanceBuffer {
  RtInstanceGpuData values[];
};
layout(std430, buffer_reference) readonly buffer RtGeometryBuffer {
  RtGeometryGpuData values[];
};

uint rtIndex(RtGeometryGpuData geometry, uint index) {
  const uint format = geometry.indexFormatAndVertexFormat & 0xffu;
  const uint element = geometry.indexOffset + index;
  if (format == 0u) {
    const uint packed = geometry.indexBuffer.words[element >> 1u];
    return (element & 1u) == 0u ? packed & 0xffffu : packed >> 16u;
  }
  return geometry.indexBuffer.words[element];
}

uint rtPackedVertexFormat(RtGeometryGpuData geometry) {
  return geometry.indexFormatAndVertexFormat >> 8u;
}

uvec3 rtTriangleIndices(RtGeometryGpuData geometry, uint primitive) {
  return uvec3(rtIndex(geometry, primitive * 3u + 0u),
               rtIndex(geometry, primitive * 3u + 1u),
               rtIndex(geometry, primitive * 3u + 2u)) + geometry.vertexOffset;
}

vec3 rtBarycentrics(vec2 barycentrics) {
  return vec3(1.0 - barycentrics.x - barycentrics.y,
              barycentrics.x, barycentrics.y);
}

vec2 rtHitUvSet(RtGeometryGpuData geometry, uint primitive,
                vec2 barycentrics, uint uvSet) {
  const uvec3 indices = rtTriangleIndices(geometry, primitive);
  const uint format = rtPackedVertexFormat(geometry);
  const vec3 bary = rtBarycentrics(barycentrics);
  if (uvSet == 1u) {
    return decodePackedUv1From(geometry.vertexBuffer, format, indices.x) * bary.x +
           decodePackedUv1From(geometry.vertexBuffer, format, indices.y) * bary.y +
           decodePackedUv1From(geometry.vertexBuffer, format, indices.z) * bary.z;
  }
  return decodePackedUvFrom(geometry.vertexBuffer, format, indices.x) * bary.x +
         decodePackedUvFrom(geometry.vertexBuffer, format, indices.y) * bary.y +
         decodePackedUvFrom(geometry.vertexBuffer, format, indices.z) * bary.z;
}

vec2 rtMaterialUv(RtGeometryGpuData geometry, uint primitive,
                  vec2 barycentrics, MaterialHeaderGpuData material,
                  uint textureSlot) {
  const uint uvSet = getPackedUvBit(material.uvSetBits, textureSlot);
  return applyTextureTransform(
      rtHitUvSet(geometry, primitive, barycentrics, uvSet),
      material.commonTransforms[textureSlot]);
}

float rtTextureLod(RtGeometryGpuData geometry, RtInstanceGpuData instance,
                   uint primitive, MaterialHeaderGpuData material,
                   uint textureSlot, uint textureId, float hitDistance,
                   uint raysPerProbe) {
  const uvec3 indices = rtTriangleIndices(geometry, primitive);
  const uint format = rtPackedVertexFormat(geometry);
  const vec3 p0 = (instance.worldFromObject * vec4(decodePackedPositionFrom(
      geometry.vertexBuffer, geometry.vertexDecodeBuffer, 0u, format,
      indices.x), 1.0)).xyz;
  const vec3 p1 = (instance.worldFromObject * vec4(decodePackedPositionFrom(
      geometry.vertexBuffer, geometry.vertexDecodeBuffer, 0u, format,
      indices.y), 1.0)).xyz;
  const vec3 p2 = (instance.worldFromObject * vec4(decodePackedPositionFrom(
      geometry.vertexBuffer, geometry.vertexDecodeBuffer, 0u, format,
      indices.z), 1.0)).xyz;
  const uint uvSet = getPackedUvBit(material.uvSetBits, textureSlot);
  vec2 uv0 = uvSet == 1u
      ? decodePackedUv1From(geometry.vertexBuffer, format, indices.x)
      : decodePackedUvFrom(geometry.vertexBuffer, format, indices.x);
  vec2 uv1 = uvSet == 1u
      ? decodePackedUv1From(geometry.vertexBuffer, format, indices.y)
      : decodePackedUvFrom(geometry.vertexBuffer, format, indices.y);
  vec2 uv2 = uvSet == 1u
      ? decodePackedUv1From(geometry.vertexBuffer, format, indices.z)
      : decodePackedUvFrom(geometry.vertexBuffer, format, indices.z);
  const PackedMaterialTransformGpuData transform =
      material.commonTransforms[textureSlot];
  uv0 = applyTextureTransform(uv0, transform);
  uv1 = applyTextureTransform(uv1, transform);
  uv2 = applyTextureTransform(uv2, transform);
  const vec2 textureExtent = vec2(textureBindlessSize2D(textureId));
  float texelsPerWorldUnit = 0.0;
  texelsPerWorldUnit = max(texelsPerWorldUnit,
      length((uv1 - uv0) * textureExtent) / max(length(p1 - p0), 1.0e-5));
  texelsPerWorldUnit = max(texelsPerWorldUnit,
      length((uv2 - uv0) * textureExtent) / max(length(p2 - p0), 1.0e-5));
  texelsPerWorldUnit = max(texelsPerWorldUnit,
      length((uv2 - uv1) * textureExtent) / max(length(p2 - p1), 1.0e-5));
  const float angularFootprint =
      sqrt(12.566370614359172 / float(max(raysPerProbe, 1u)));
  const float footprint = max(hitDistance, 1.0e-4) * angularFootprint;
  const int levelCount = textureQueryLevels(nonuniformEXT(kTextures2D[textureId]));
  return clamp(log2(max(footprint * texelsPerWorldUnit, 1.0)), 0.0,
               float(max(levelCount - 1, 0)));
}

vec3 rtHitNormal(RtGeometryGpuData geometry, RtInstanceGpuData instance,
                 uint primitive, vec2 barycentrics) {
  const uvec3 indices = rtTriangleIndices(geometry, primitive);
  const uint format = rtPackedVertexFormat(geometry);
  const vec3 bary = rtBarycentrics(barycentrics);
  const vec3 objectNormal = normalize(
      decodePackedNormalFrom(geometry.vertexBuffer, format, indices.x) * bary.x +
      decodePackedNormalFrom(geometry.vertexBuffer, format, indices.y) * bary.y +
      decodePackedNormalFrom(geometry.vertexBuffer, format, indices.z) * bary.z);
  return normalize(transpose(mat3(instance.objectFromWorld)) * objectNormal);
}

vec3 rtHitGeometricNormal(RtGeometryGpuData geometry,
                          RtInstanceGpuData instance, uint primitive) {
  const uvec3 indices = rtTriangleIndices(geometry, primitive);
  const uint format = rtPackedVertexFormat(geometry);
  const vec3 p0 = (instance.worldFromObject * vec4(decodePackedPositionFrom(
      geometry.vertexBuffer, geometry.vertexDecodeBuffer, 0u, format,
      indices.x), 1.0)).xyz;
  const vec3 p1 = (instance.worldFromObject * vec4(decodePackedPositionFrom(
      geometry.vertexBuffer, geometry.vertexDecodeBuffer, 0u, format,
      indices.y), 1.0)).xyz;
  const vec3 p2 = (instance.worldFromObject * vec4(decodePackedPositionFrom(
      geometry.vertexBuffer, geometry.vertexDecodeBuffer, 0u, format,
      indices.z), 1.0)).xyz;
  const vec3 faceNormal = cross(p1 - p0, p2 - p0);
  return dot(faceNormal, faceNormal) > 1.0e-12
             ? normalize(faceNormal)
             : rtHitNormal(geometry, instance, primitive, vec2(0.0));
}

vec3 rtHitShadingNormal(RtGeometryGpuData geometry, RtInstanceGpuData instance,
                        uint primitive, vec2 barycentrics,
                        MaterialHeaderGpuData material, uint samplerId,
                        float hitDistance, uint raysPerProbe) {
  const vec3 geometricNormal =
      rtHitNormal(geometry, instance, primitive, barycentrics);
  const uint textureId = material.commonTextureIndices.z;
  if (textureId == kInvalidTextureBindlessIndex) {
    return geometricNormal;
  }
  const uvec3 indices = rtTriangleIndices(geometry, primitive);
  const uint format = rtPackedVertexFormat(geometry);
  const vec3 bary = rtBarycentrics(barycentrics);
  const vec4 objectTangent =
      decodePackedTangentFrom(geometry.vertexBuffer, format, indices.x) * bary.x +
      decodePackedTangentFrom(geometry.vertexBuffer, format, indices.y) * bary.y +
      decodePackedTangentFrom(geometry.vertexBuffer, format, indices.z) * bary.z;
  vec3 tangent = normalize(mat3(instance.worldFromObject) * objectTangent.xyz);
  tangent = normalize(tangent - geometricNormal * dot(tangent, geometricNormal));
  const vec3 bitangent =
      normalize(cross(geometricNormal, tangent)) *
      (objectTangent.w < 0.0 ? -1.0 : 1.0);
  const vec2 uv = rtMaterialUv(geometry, primitive, barycentrics, material,
                               kMaterialTextureSlotNormal);
  const float lod = rtTextureLod(
      geometry, instance, primitive, material, kMaterialTextureSlotNormal,
      textureId, hitDistance, raysPerProbe);
  vec3 mapped = textureBindless2DLod(textureId, samplerId, uv, lod).xyz * 2.0 - 1.0;
  mapped.xy *= material.normalScaleIorReserved.x;
  return normalize(mat3(tangent, bitangent, geometricNormal) * normalize(mapped));
}

bool rtAlphaPasses(RtGeometryGpuData geometry, RtInstanceGpuData instance,
                   uint primitive,
                   vec2 barycentrics, MaterialHeaderBuffer materials,
                   uint samplerId, float hitDistance, uint raysPerProbe) {
  if ((geometry.flags & kRtGeometryAlphaMasked) == 0u) {
    return true;
  }
  const MaterialHeaderGpuData material = materials.materials[geometry.materialIndex];
  const uint textureId = material.commonTextureIndices.x;
  float alpha = material.baseColorFactor.a;
  if (textureId != kInvalidTextureBindlessIndex) {
    const vec2 uv = rtMaterialUv(geometry, primitive, barycentrics, material,
                                 kMaterialTextureSlotBaseColor);
    const float lod = rtTextureLod(
        geometry, instance, primitive, material, kMaterialTextureSlotBaseColor,
        textureId, hitDistance, raysPerProbe);
    alpha *= textureBindless2DLod(textureId, samplerId,
                                  uv, lod).a;
  }
  return alpha >= material.metallicRoughnessOcclusionAlphaCutoff.w;
}

vec4 rtBaseColor(RtGeometryGpuData geometry, RtInstanceGpuData instance,
                 uint primitive,
                 vec2 barycentrics, MaterialHeaderGpuData material,
                 uint samplerId, float hitDistance, uint raysPerProbe) {
  vec4 color = material.baseColorFactor;
  const uint textureId = material.commonTextureIndices.x;
  if (textureId != kInvalidTextureBindlessIndex) {
    const vec2 uv = rtMaterialUv(geometry, primitive, barycentrics, material,
                                 kMaterialTextureSlotBaseColor);
    const float lod = rtTextureLod(
        geometry, instance, primitive, material, kMaterialTextureSlotBaseColor,
        textureId, hitDistance, raysPerProbe);
    color *= textureBindless2DLod(textureId, samplerId,
                                  uv, lod);
  }
  return color;
}

float rtMetallic(RtGeometryGpuData geometry, RtInstanceGpuData instance,
                 uint primitive, vec2 barycentrics,
                 MaterialHeaderGpuData material, uint samplerId,
                 float hitDistance, uint raysPerProbe) {
  float metallic = material.metallicRoughnessOcclusionAlphaCutoff.x;
  const uint textureId = material.commonTextureIndices.y;
  if (textureId != kInvalidTextureBindlessIndex) {
    const vec2 uv = rtMaterialUv(geometry, primitive, barycentrics, material,
                                 kMaterialTextureSlotMetallicRoughness);
    const float lod = rtTextureLod(
        geometry, instance, primitive, material,
        kMaterialTextureSlotMetallicRoughness, textureId, hitDistance,
        raysPerProbe);
    metallic *= textureBindless2DLod(textureId, samplerId, uv, lod).b;
  }
  return clamp(metallic, 0.0, 1.0);
}

vec3 rtEmissive(RtGeometryGpuData geometry, RtInstanceGpuData instance,
                uint primitive, vec2 barycentrics,
                MaterialHeaderGpuData material, uint samplerId,
                float hitDistance, uint raysPerProbe) {
  vec3 emissive = material.emissiveFactorStrength.xyz *
                  material.emissiveFactorStrength.w;
  const uint textureId = material.emissiveTextureIndex;
  if (textureId != kInvalidTextureBindlessIndex) {
    const vec2 uv = rtMaterialUv(geometry, primitive, barycentrics, material,
                                 kMaterialTextureSlotEmissive);
    const float lod = rtTextureLod(
        geometry, instance, primitive, material, kMaterialTextureSlotEmissive,
        textureId, hitDistance, raysPerProbe);
    emissive *= textureBindless2DLod(textureId, samplerId, uv, lod).rgb;
  }
  return emissive;
}
