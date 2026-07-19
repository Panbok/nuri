#include "nuri/resources/storage/material/material_binary_serializer.h"
#include "nuri/pch.h"
#include "nuri/resources/storage/binary_io.h"
#include "nuri/resources/storage/material/material_binary_format.h"
namespace nuri {
namespace {
using FloatField = float MaterialData::*;
constexpr std::array kEarlyFields{
    &MaterialData::emissiveStrength,
    &MaterialData::metallicFactor,
    &MaterialData::roughnessFactor,
};
constexpr std::array kSpecularFields{
    &MaterialData::specularFactor,
    &MaterialData::glossinessFactor,
};
constexpr std::array kExtensionFields{
    &MaterialData::sheenWeight,
    &MaterialData::sheenRoughnessFactor,
    &MaterialData::clearcoatFactor,
    &MaterialData::clearcoatRoughnessFactor,
    &MaterialData::clearcoatNormalScale,
    &MaterialData::transmissionFactor,
    &MaterialData::thicknessFactor,
};
constexpr std::array kLateFields{
    &MaterialData::attenuationDistance, &MaterialData::ior,
    &MaterialData::normalScale,         &MaterialData::occlusionStrength,
    &MaterialData::alphaCutoff,
};
constexpr uint8_t kRecordVersion = 1u;
constexpr size_t kMinimumRecordBytes = 830u;
template <typename T>
[[nodiscard]] Result<T, MaterialBinaryDeserializeError>
deserializeError(std::string message, bool stale = false) {
  return Result<T, MaterialBinaryDeserializeError>::makeError(
      {.message = std::move(message), .stale = stale});
}
void writeVec3(BinaryWriter &writer, const glm::vec3 &value) {
  writer.write(value.x);
  writer.write(value.y);
  writer.write(value.z);
}
void writeVec4(BinaryWriter &writer, const glm::vec4 &value) {
  writer.write(value.x);
  writer.write(value.y);
  writer.write(value.z);
  writer.write(value.w);
}
[[nodiscard]] glm::vec3 readVec3(BinaryReader &reader) {
  const float x = reader.read<float>();
  const float y = reader.read<float>();
  const float z = reader.read<float>();
  return {x, y, z};
}
[[nodiscard]] glm::vec4 readVec4(BinaryReader &reader) {
  const float x = reader.read<float>();
  const float y = reader.read<float>();
  const float z = reader.read<float>();
  const float w = reader.read<float>();
  return {x, y, z, w};
}
template <size_t N>
void writeFields(BinaryWriter &writer, const MaterialData &material,
                 const std::array<FloatField, N> &fields) {
  for (FloatField field : fields) {
    writer.write(material.*field);
  }
}
template <size_t N>
void readFields(BinaryReader &reader, MaterialData &material,
                const std::array<FloatField, N> &fields) {
  for (FloatField field : fields) {
    material.*field = reader.read<float>();
  }
}
void writeSlot(BinaryWriter &writer, const MaterialTextureSlotData &slot) {
  writer.writeString(slot.path);
  writer.write(static_cast<uint8_t>(slot.sourceKind));
  writer.write(slot.embeddedIndex);
  writer.write(slot.uvSet);
  writer.write(slot.samplerIndex);
  writer.write(slot.scale);
  writer.write(slot.transform.offset.x);
  writer.write(slot.transform.offset.y);
  writer.write(slot.transform.scale.x);
  writer.write(slot.transform.scale.y);
  writer.write(slot.transform.rotationRadians);
}
[[nodiscard]] bool readSlot(BinaryReader &reader,
                            MaterialTextureSlotData &slot) {
  slot.path = reader.readString();
  const uint8_t sourceKind = reader.read<uint8_t>();
  slot.embeddedIndex = reader.read<uint32_t>();
  slot.uvSet = reader.read<uint32_t>();
  slot.samplerIndex = reader.read<uint32_t>();
  slot.scale = reader.read<float>();
  slot.transform.offset.x = reader.read<float>();
  slot.transform.offset.y = reader.read<float>();
  slot.transform.scale.x = reader.read<float>();
  slot.transform.scale.y = reader.read<float>();
  slot.transform.rotationRadians = reader.read<float>();
  if (!reader.valid() ||
      sourceKind > static_cast<uint8_t>(
                       MaterialTextureSourceKind::EmbeddedSceneTexture)) {
    return false;
  }
  slot.sourceKind = static_cast<MaterialTextureSourceKind>(sourceKind);
  return true;
}
void writeRecord(BinaryWriter &writer, const SceneMaterialRecord &record) {
  const MaterialData &material = record.sourceMaterial;
  writer.write(kRecordVersion);
  writer.write(record.sourceMaterialIndex);
  writer.writeString(material.name);
  writer.write(static_cast<uint8_t>(material.workflow));
  writeVec4(writer, material.baseColorFactor);
  writeVec3(writer, material.emissiveFactor);
  writeFields(writer, material, kEarlyFields);
  writeVec3(writer, material.specularColorFactor);
  writeFields(writer, material, kSpecularFields);
  writeVec3(writer, material.sheenColorFactor);
  writeFields(writer, material, kExtensionFields);
  writeVec3(writer, material.attenuationColor);
  writeFields(writer, material, kLateFields);
  writer.write(static_cast<uint8_t>(material.doubleSided));
  writer.write(static_cast<uint8_t>(material.alphaMode));
  for (size_t i = 0; i < material.textures.size(); ++i) {
    writeSlot(writer, material.textures[i]);
    writer.write(record.textureCache[i].artifactIdentityHash);
  }
}
[[nodiscard]] bool readRecord(BinaryReader &reader,
                              SceneMaterialRecord &record) {
  const uint8_t version = reader.read<uint8_t>();
  record.sourceMaterialIndex = reader.read<uint32_t>();
  MaterialData &material = record.sourceMaterial;
  material.name = reader.readString();
  const uint8_t workflow = reader.read<uint8_t>();
  material.baseColorFactor = readVec4(reader);
  material.emissiveFactor = readVec3(reader);
  readFields(reader, material, kEarlyFields);
  material.specularColorFactor = readVec3(reader);
  readFields(reader, material, kSpecularFields);
  material.sheenColorFactor = readVec3(reader);
  readFields(reader, material, kExtensionFields);
  material.attenuationColor = readVec3(reader);
  readFields(reader, material, kLateFields);
  material.doubleSided = reader.read<uint8_t>() != 0u;
  const uint8_t alphaMode = reader.read<uint8_t>();
  if (!reader.valid() || version != kRecordVersion ||
      workflow > static_cast<uint8_t>(MaterialWorkflow::SpecularGlossiness) ||
      alphaMode > static_cast<uint8_t>(MaterialAlphaMode::Blend)) {
    return false;
  }
  material.workflow = static_cast<MaterialWorkflow>(workflow);
  material.alphaMode = static_cast<MaterialAlphaMode>(alphaMode);
  for (size_t i = 0; i < material.textures.size(); ++i) {
    if (!readSlot(reader, material.textures[i])) {
      return false;
    }
    record.textureCache[i].artifactIdentityHash = reader.read<uint64_t>();
  }
  return reader.valid();
}
} // namespace

Result<std::vector<std::byte>, std::string>
materialBinarySerialize(const MaterialBinarySerializeInput &input) {
  if (input.materials.size() > std::numeric_limits<uint32_t>::max()) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "materialBinarySerialize: too many materials");
  }
  MaterialBinaryHeader header{
      .magic = kMaterialBinaryMagic,
      .majorVersion = kMaterialBinaryFormatMajorVersion,
      .minorVersion = kMaterialBinaryFormatMinorVersion,
      .sourcePathHash = input.sourcePathHash,
      .sourceSizeBytes = input.sourceSizeBytes,
      .sourceMtimeNs = input.sourceMtimeNs,
      .materialCount = static_cast<uint32_t>(input.materials.size()),
  };
  BinaryWriter writer;
  writer.write(header);
  for (const SceneMaterialRecord &record : input.materials) {
    writeRecord(writer, record);
  }
  if (!writer.valid() ||
      writer.bytes().size() > std::numeric_limits<uint32_t>::max()) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "materialBinarySerialize: output exceeds format limits");
  }
  header.fileSize = static_cast<uint32_t>(writer.bytes().size());
  auto bytes = std::move(writer).take();
  std::memcpy(bytes.data(), &header, sizeof(header));
  return Result<std::vector<std::byte>, std::string>::makeResult(
      std::move(bytes));
}

Result<SceneMaterialCacheData, MaterialBinaryDeserializeError>
materialBinaryDeserialize(std::span<const std::byte> fileBytes,
                          const MaterialBinaryDeserializeContext &context) {
  if (fileBytes.size() < sizeof(MaterialBinaryHeader)) {
    return deserializeError<SceneMaterialCacheData>(
        "materialBinaryDeserialize: file too small");
  }
  MaterialBinaryHeader header{};
  std::memcpy(&header, fileBytes.data(), sizeof(header));
  if (header.magic != kMaterialBinaryMagic ||
      header.majorVersion != kMaterialBinaryFormatMajorVersion ||
      header.minorVersion > kMaterialBinaryFormatMinorVersion ||
      header.fileSize != fileBytes.size() ||
      header.sourcePathHash != context.expectedSourcePathHash) {
    return deserializeError<SceneMaterialCacheData>(
        "materialBinaryDeserialize: invalid cache header");
  }
  if (context.validateSourceFingerprint &&
      (!context.sourceExists ||
       header.sourceSizeBytes != context.sourceSizeBytes ||
       header.sourceMtimeNs != context.sourceMtimeNs)) {
    return deserializeError<SceneMaterialCacheData>(
        "materialBinaryDeserialize: stale source fingerprint", true);
  }
  const auto payload = fileBytes.subspan(sizeof(header));
  if (header.materialCount > payload.size() / kMinimumRecordBytes) {
    return deserializeError<SceneMaterialCacheData>(
        "materialBinaryDeserialize: invalid material count");
  }
  BinaryReader reader(payload);
  SceneMaterialCacheData data;
  data.materials.resize(header.materialCount);
  for (SceneMaterialRecord &record : data.materials) {
    if (!readRecord(reader, record)) {
      return deserializeError<SceneMaterialCacheData>(
          "materialBinaryDeserialize: invalid material record");
    }
  }
  if (!reader.empty()) {
    return deserializeError<SceneMaterialCacheData>(
        "materialBinaryDeserialize: trailing bytes");
  }
  return Result<SceneMaterialCacheData,
                MaterialBinaryDeserializeError>::makeResult(std::move(data));
}

} // namespace nuri
