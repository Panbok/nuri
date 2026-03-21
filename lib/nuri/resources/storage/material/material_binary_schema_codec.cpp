#include "nuri/pch.h"

#include "nuri/resources/storage/material/material_binary_schema_codec.h"

namespace nuri::material_binary_schema_codec {
namespace {

using SlotPtr = MaterialTextureSlotData MaterialData::*;

constexpr std::array<SlotPtr, kMaterialTextureSlotCount> kMaterialSlotPtrs{{
    &MaterialData::baseColor,
    &MaterialData::metallicRoughness,
    &MaterialData::normal,
    &MaterialData::occlusion,
    &MaterialData::emissive,
    &MaterialData::clearcoat,
    &MaterialData::clearcoatRoughness,
    &MaterialData::clearcoatNormal,
    &MaterialData::specular,
    &MaterialData::specularColor,
    &MaterialData::sheenColor,
    &MaterialData::sheenRoughness,
    &MaterialData::transmission,
    &MaterialData::thickness,
}};
static_assert(kMaterialSlotPtrs.size() == kMaterialTextureSlotCount,
              "kMaterialSlotPtrs must match kMaterialTextureSlotCount");

constexpr uint8_t kSceneMaterialRecordFormatVersion = 1u;

[[nodiscard]] constexpr bool
isValidMaterialTextureSourceKind(uint32_t value) noexcept {
  return value <=
         static_cast<uint32_t>(MaterialTextureSourceKind::EmbeddedSceneTexture);
}

[[nodiscard]] constexpr bool isValidMaterialAlphaMode(uint32_t value) noexcept {
  return value <= static_cast<uint32_t>(MaterialAlphaMode::Blend);
}

template <typename T>
[[nodiscard]] Result<T, MaterialBinaryDeserializeError>
makeDeserializeError(std::string message) {
  return Result<T, MaterialBinaryDeserializeError>::makeError(
      MaterialBinaryDeserializeError{.message = std::move(message)});
}

void writeVec3(material_binary_codec::Writer &writer, const glm::vec3 &value) {
  writer.writeF32(value.x);
  writer.writeF32(value.y);
  writer.writeF32(value.z);
}

void writeVec4(material_binary_codec::Writer &writer, const glm::vec4 &value) {
  writer.writeF32(value.x);
  writer.writeF32(value.y);
  writer.writeF32(value.z);
  writer.writeF32(value.w);
}

[[nodiscard]] Result<glm::vec3, MaterialBinaryDeserializeError>
readVec3(material_binary_codec::Reader &reader) {
  auto x = reader.readF32();
  auto y = reader.readF32();
  auto z = reader.readF32();
  if (x.hasError() || y.hasError() || z.hasError()) {
    return makeDeserializeError<glm::vec3>(
        "materialBinaryDeserialize: failed to read vec3");
  }
  return Result<glm::vec3, MaterialBinaryDeserializeError>::makeResult(
      glm::vec3(x.value(), y.value(), z.value()));
}

[[nodiscard]] Result<glm::vec4, MaterialBinaryDeserializeError>
readVec4(material_binary_codec::Reader &reader) {
  auto x = reader.readF32();
  auto y = reader.readF32();
  auto z = reader.readF32();
  auto w = reader.readF32();
  if (x.hasError() || y.hasError() || z.hasError() || w.hasError()) {
    return makeDeserializeError<glm::vec4>(
        "materialBinaryDeserialize: failed to read vec4");
  }
  return Result<glm::vec4, MaterialBinaryDeserializeError>::makeResult(
      glm::vec4(x.value(), y.value(), z.value(), w.value()));
}

void writeTextureSlot(material_binary_codec::Writer &writer,
                      const MaterialTextureSlotData &slot) {
  writer.writeString(slot.path);
  writer.writeU8(static_cast<uint8_t>(slot.sourceKind));
  writer.writeU32(slot.embeddedIndex);
  writer.writeU32(slot.uvSet);
  writer.writeU32(slot.samplerIndex);
  writer.writeF32(slot.scale);
  writer.writeF32(slot.transform.offset.x);
  writer.writeF32(slot.transform.offset.y);
  writer.writeF32(slot.transform.scale.x);
  writer.writeF32(slot.transform.scale.y);
  writer.writeF32(slot.transform.rotationRadians);
}

[[nodiscard]] Result<MaterialTextureSlotData, MaterialBinaryDeserializeError>
readTextureSlot(material_binary_codec::Reader &reader) {
  MaterialTextureSlotData slot{};
  auto path = reader.readString();
  auto sourceKind = reader.readU8();
  auto embeddedIndex = reader.readU32();
  auto uvSet = reader.readU32();
  auto samplerIndex = reader.readU32();
  auto scale = reader.readF32();
  auto offsetX = reader.readF32();
  auto offsetY = reader.readF32();
  auto scaleX = reader.readF32();
  auto scaleY = reader.readF32();
  auto rotation = reader.readF32();
  if (path.hasError() || sourceKind.hasError() || embeddedIndex.hasError() ||
      uvSet.hasError() || samplerIndex.hasError() || scale.hasError() ||
      offsetX.hasError() || offsetY.hasError() || scaleX.hasError() ||
      scaleY.hasError() || rotation.hasError()) {
    return makeDeserializeError<MaterialTextureSlotData>(
        "materialBinaryDeserialize: failed to read texture slot");
  }
  slot.path = std::move(path.value());
  if (!isValidMaterialTextureSourceKind(sourceKind.value())) {
    return makeDeserializeError<MaterialTextureSlotData>(
        std::format("materialBinaryDeserialize: invalid texture source kind {}",
                    sourceKind.value()));
  }
  slot.sourceKind = static_cast<MaterialTextureSourceKind>(sourceKind.value());
  slot.embeddedIndex = embeddedIndex.value();
  slot.uvSet = uvSet.value();
  slot.samplerIndex = samplerIndex.value();
  slot.scale = scale.value();
  slot.transform.offset = glm::vec2(offsetX.value(), offsetY.value());
  slot.transform.scale = glm::vec2(scaleX.value(), scaleY.value());
  slot.transform.rotationRadians = rotation.value();
  return Result<MaterialTextureSlotData,
                MaterialBinaryDeserializeError>::makeResult(std::move(slot));
}

} // namespace

void writeSceneMaterialRecord(material_binary_codec::Writer &writer,
                              const SceneMaterialRecord &record) {
  const MaterialData &material = record.sourceMaterial;
  writer.writeU8(kSceneMaterialRecordFormatVersion);
  writer.writeU32(record.sourceMaterialIndex);
  writer.writeString(material.name);
  writeVec4(writer, material.baseColorFactor);
  writeVec3(writer, material.emissiveFactor);
  writer.writeF32(material.emissiveStrength);
  writer.writeF32(material.metallicFactor);
  writer.writeF32(material.roughnessFactor);
  writeVec3(writer, material.specularColorFactor);
  writer.writeF32(material.specularFactor);
  writeVec3(writer, material.sheenColorFactor);
  writer.writeF32(material.sheenWeight);
  writer.writeF32(material.sheenRoughnessFactor);
  writer.writeF32(material.clearcoatFactor);
  writer.writeF32(material.clearcoatRoughnessFactor);
  writer.writeF32(material.clearcoatNormalScale);
  writer.writeF32(material.transmissionFactor);
  writer.writeF32(material.thicknessFactor);
  writeVec3(writer, material.attenuationColor);
  writer.writeF32(material.attenuationDistance);
  writer.writeF32(material.ior);
  writer.writeF32(material.normalScale);
  writer.writeF32(material.occlusionStrength);
  writer.writeF32(material.alphaCutoff);
  writer.writeU8(material.doubleSided ? 1u : 0u);
  writer.writeU8(static_cast<uint8_t>(material.alphaMode));

  for (size_t i = 0; i < kMaterialSlotPtrs.size(); ++i) {
    writeTextureSlot(writer, material.*(kMaterialSlotPtrs[i]));
    writer.writeString(record.textureCache[i].portablePath);
    writer.writeU8(record.textureCache[i].srgb ? 1u : 0u);
    writer.writeU64(record.textureCache[i].sourceIdentityHash);
  }
}

Result<SceneMaterialRecord, MaterialBinaryDeserializeError>
readSceneMaterialRecord(material_binary_codec::Reader &reader) {
  SceneMaterialRecord record{};
  auto formatVersion = reader.readU8();
  auto sourceMaterialIndex = reader.readU32();
  auto name = reader.readString();
  auto baseColorFactor = readVec4(reader);
  auto emissiveFactor = readVec3(reader);
  auto emissiveStrength = reader.readF32();
  auto metallicFactor = reader.readF32();
  auto roughnessFactor = reader.readF32();
  auto specularColorFactor = readVec3(reader);
  auto specularFactor = reader.readF32();
  auto sheenColorFactor = readVec3(reader);
  auto sheenWeight = reader.readF32();
  auto sheenRoughnessFactor = reader.readF32();
  auto clearcoatFactor = reader.readF32();
  auto clearcoatRoughnessFactor = reader.readF32();
  auto clearcoatNormalScale = reader.readF32();
  auto transmissionFactor = reader.readF32();
  auto thicknessFactor = reader.readF32();
  auto attenuationColor = readVec3(reader);
  auto attenuationDistance = reader.readF32();
  auto ior = reader.readF32();
  auto normalScale = reader.readF32();
  auto occlusionStrength = reader.readF32();
  auto alphaCutoff = reader.readF32();
  auto doubleSided = reader.readU8();
  auto alphaMode = reader.readU8();
  if (formatVersion.hasError() || sourceMaterialIndex.hasError() || name.hasError() ||
      baseColorFactor.hasError() || emissiveFactor.hasError() ||
      emissiveStrength.hasError() || metallicFactor.hasError() ||
      roughnessFactor.hasError() || specularColorFactor.hasError() ||
      specularFactor.hasError() || sheenColorFactor.hasError() ||
      sheenWeight.hasError() || sheenRoughnessFactor.hasError() ||
      clearcoatFactor.hasError() || clearcoatRoughnessFactor.hasError() ||
      clearcoatNormalScale.hasError() || transmissionFactor.hasError() ||
      thicknessFactor.hasError() || attenuationColor.hasError() ||
      attenuationDistance.hasError() || ior.hasError() ||
      normalScale.hasError() || occlusionStrength.hasError() ||
      alphaCutoff.hasError() || doubleSided.hasError() ||
      alphaMode.hasError()) {
    return makeDeserializeError<SceneMaterialRecord>(
        "materialBinaryDeserialize: failed to read material record");
  }
  if (formatVersion.value() != kSceneMaterialRecordFormatVersion) {
    return makeDeserializeError<SceneMaterialRecord>(
        "materialBinaryDeserialize: unsupported format version");
  }

  MaterialData material{};
  record.sourceMaterialIndex = sourceMaterialIndex.value();
  material.name = std::move(name.value());
  material.baseColorFactor = baseColorFactor.value();
  material.emissiveFactor = emissiveFactor.value();
  material.emissiveStrength = emissiveStrength.value();
  material.metallicFactor = metallicFactor.value();
  material.roughnessFactor = roughnessFactor.value();
  material.specularColorFactor = specularColorFactor.value();
  material.specularFactor = specularFactor.value();
  material.sheenColorFactor = sheenColorFactor.value();
  material.sheenWeight = sheenWeight.value();
  material.sheenRoughnessFactor = sheenRoughnessFactor.value();
  material.clearcoatFactor = clearcoatFactor.value();
  material.clearcoatRoughnessFactor = clearcoatRoughnessFactor.value();
  material.clearcoatNormalScale = clearcoatNormalScale.value();
  material.transmissionFactor = transmissionFactor.value();
  material.thicknessFactor = thicknessFactor.value();
  material.attenuationColor = attenuationColor.value();
  material.attenuationDistance = attenuationDistance.value();
  material.ior = ior.value();
  material.normalScale = normalScale.value();
  material.occlusionStrength = occlusionStrength.value();
  material.alphaCutoff = alphaCutoff.value();
  material.doubleSided = doubleSided.value() != 0u;
  if (isValidMaterialAlphaMode(alphaMode.value())) {
    material.alphaMode = static_cast<MaterialAlphaMode>(alphaMode.value());
  } else {
    material.alphaMode = MaterialAlphaMode::Opaque;
  }

  for (size_t i = 0; i < kMaterialSlotPtrs.size(); ++i) {
    auto slot = readTextureSlot(reader);
    auto portablePath = reader.readString();
    auto srgb = reader.readU8();
    auto sourceIdentityHash = reader.readU64();
    if (slot.hasError() || portablePath.hasError() || srgb.hasError() ||
        sourceIdentityHash.hasError()) {
      return makeDeserializeError<SceneMaterialRecord>(
          "materialBinaryDeserialize: failed to read material texture entry");
    }
    material.*(kMaterialSlotPtrs[i]) = std::move(slot.value());
    record.textureCache[i].portablePath = std::move(portablePath.value());
    record.textureCache[i].srgb = srgb.value() != 0u;
    record.textureCache[i].sourceIdentityHash = sourceIdentityHash.value();
  }

  record.sourceMaterial = std::move(material);
  return Result<SceneMaterialRecord,
                MaterialBinaryDeserializeError>::makeResult(std::move(record));
}

} // namespace nuri::material_binary_schema_codec
