#include "nuri/pch.h"

#include "nuri/resources/storage/material/material_binary_serializer.h"

#include "nuri/resources/storage/material/material_binary_codec.h"
#include "nuri/resources/storage/material/material_binary_format.h"
#include "nuri/resources/storage/material/material_binary_schema_codec.h"

namespace nuri {
namespace {

template <typename T>
[[nodiscard]] Result<T, MaterialBinaryDeserializeError>
makeDeserializeError(std::string message, bool stale = false) {
  return Result<T, MaterialBinaryDeserializeError>::makeError(
      MaterialBinaryDeserializeError{.message = std::move(message),
                                     .stale = stale});
}

constexpr size_t kSerializedU8Bytes = sizeof(uint8_t);
constexpr size_t kSerializedU32Bytes = sizeof(uint32_t);
constexpr size_t kSerializedU64Bytes = sizeof(uint64_t);
constexpr size_t kSerializedF32Bytes = sizeof(float);
constexpr size_t kSerializedEmptyStringBytes = kSerializedU32Bytes;
constexpr size_t kSerializedVec3Bytes = 3u * kSerializedF32Bytes;
constexpr size_t kSerializedVec4Bytes = 4u * kSerializedF32Bytes;

[[nodiscard]] constexpr size_t minSerializedTextureSlotBytes() noexcept {
  constexpr size_t kSerializedTextureTransformBytes =
      (2u * kSerializedF32Bytes) + (2u * kSerializedF32Bytes) +
      kSerializedF32Bytes;
  return kSerializedEmptyStringBytes + // `slot.path`
         kSerializedU8Bytes +          // `slot.sourceKind`
         (3u *
          kSerializedU32Bytes) + // `embeddedIndex`, `uvSet`, `samplerIndex`
         kSerializedF32Bytes +   // `slot.scale`
         kSerializedTextureTransformBytes; // `offset.xy`, `scale.xy`,
                                           // `rotation`
}

[[nodiscard]] constexpr size_t minSerializedTextureCacheEntryBytes() noexcept {
  return kSerializedEmptyStringBytes + // `portablePath`
         kSerializedU8Bytes +          // `srgb`
         kSerializedU64Bytes;          // `sourceIdentityHash`
}

[[nodiscard]] constexpr size_t
minSerializedSceneMaterialRecordBytes() noexcept {
  constexpr size_t kMaterialVec3FieldCount = 4u;
  constexpr size_t kMaterialScalarFloatFieldCount = 17u;
  return kSerializedU8Bytes +          // format version
         kSerializedU32Bytes +         // source material index
         kSerializedEmptyStringBytes + // material name
         kSerializedU8Bytes +          // workflow
         kSerializedVec4Bytes +        // baseColorFactor
         (kMaterialVec3FieldCount * kSerializedVec3Bytes) + // vec3 factors
         (kMaterialScalarFloatFieldCount *
          kSerializedF32Bytes) + // scalar factors
         kSerializedU8Bytes +    // doubleSided
         kSerializedU8Bytes +    // alphaMode
         (kMaterialTextureSlotCount * (minSerializedTextureSlotBytes() +
                                       minSerializedTextureCacheEntryBytes()));
}

static_assert(minSerializedSceneMaterialRecordBytes() == 900u);

} // namespace

Result<std::vector<std::byte>, std::string>
materialBinarySerialize(const MaterialBinarySerializeInput &input) {
  material_binary_codec::Writer writer{};
  MaterialBinaryHeader header{};
  header.magic = kMaterialBinaryMagic;
  header.majorVersion = kMaterialBinaryFormatMajorVersion;
  header.minorVersion = kMaterialBinaryFormatMinorVersion;
  header.sourcePathHash = input.sourcePathHash;
  header.sourceSizeBytes = input.sourceSizeBytes;
  header.sourceMtimeNs = input.sourceMtimeNs;
  if (input.materials.size() > std::numeric_limits<uint32_t>::max()) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "materialBinarySerialize: material count exceeds uint32_t range");
  }
  header.materialCount = static_cast<uint32_t>(input.materials.size());

  writer.writeBytes(
      {reinterpret_cast<const std::byte *>(&header), sizeof(header)});
  for (const SceneMaterialRecord &record : input.materials) {
    material_binary_schema_codec::writeSceneMaterialRecord(writer, record);
  }

  std::vector<std::byte> bytes = writer.bytes();
  if (bytes.size() > std::numeric_limits<uint32_t>::max()) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "materialBinarySerialize: output exceeds 4GB");
  }
  header.fileSize = static_cast<uint32_t>(bytes.size());
  std::memcpy(bytes.data(), &header, sizeof(header));
  return Result<std::vector<std::byte>, std::string>::makeResult(
      std::move(bytes));
}

Result<SceneMaterialCacheData, MaterialBinaryDeserializeError>
materialBinaryDeserialize(std::span<const std::byte> fileBytes,
                          const MaterialBinaryDeserializeContext &context) {
  if (fileBytes.size() < sizeof(MaterialBinaryHeader)) {
    return makeDeserializeError<SceneMaterialCacheData>(
        "materialBinaryDeserialize: file too small");
  }

  MaterialBinaryHeader header{};
  std::memcpy(&header, fileBytes.data(), sizeof(header));
  if (header.magic != kMaterialBinaryMagic) {
    return makeDeserializeError<SceneMaterialCacheData>(
        "materialBinaryDeserialize: magic mismatch");
  }
  if (header.majorVersion != kMaterialBinaryFormatMajorVersion) {
    return makeDeserializeError<SceneMaterialCacheData>(
        "materialBinaryDeserialize: unsupported format major version");
  }
  if (header.minorVersion > kMaterialBinaryFormatMinorVersion) {
    return makeDeserializeError<SceneMaterialCacheData>(
        "materialBinaryDeserialize: unsupported format minor version");
  }
  if (header.fileSize != fileBytes.size()) {
    return makeDeserializeError<SceneMaterialCacheData>(
        "materialBinaryDeserialize: file size mismatch");
  }
  if (header.sourcePathHash != context.expectedSourcePathHash) {
    return makeDeserializeError<SceneMaterialCacheData>(
        "materialBinaryDeserialize: source path hash mismatch");
  }
  if (context.validateSourceFingerprint) {
    if (!context.sourceExists) {
      return makeDeserializeError<SceneMaterialCacheData>(
          "materialBinaryDeserialize: source file is missing", true);
    }
    if (header.sourceSizeBytes != context.sourceSizeBytes ||
        header.sourceMtimeNs != context.sourceMtimeNs) {
      return makeDeserializeError<SceneMaterialCacheData>(
          "materialBinaryDeserialize: cache is stale for current source file",
          true);
    }
  }

  material_binary_codec::Reader reader(fileBytes.subspan(sizeof(header)));
  SceneMaterialCacheData data{};
  const size_t remainingBytes = fileBytes.size() - sizeof(header);
  const size_t safeMaxMaterialCount =
      remainingBytes / minSerializedSceneMaterialRecordBytes();
  if (header.materialCount > safeMaxMaterialCount) {
    return makeDeserializeError<SceneMaterialCacheData>(
        "materialBinaryDeserialize: material count exceeds remaining file "
        "size");
  }
  data.materials.reserve(static_cast<size_t>(header.materialCount));
  for (uint32_t i = 0; i < header.materialCount; ++i) {
    auto material =
        material_binary_schema_codec::readSceneMaterialRecord(reader);
    if (material.hasError()) {
      return Result<SceneMaterialCacheData,
                    MaterialBinaryDeserializeError>::makeError(material
                                                                   .error());
    }
    data.materials.push_back(std::move(material.value()));
  }
  if (!reader.empty()) {
    return makeDeserializeError<SceneMaterialCacheData>(
        "materialBinaryDeserialize: trailing bytes remain");
  }
  return Result<SceneMaterialCacheData,
                MaterialBinaryDeserializeError>::makeResult(std::move(data));
}

} // namespace nuri
