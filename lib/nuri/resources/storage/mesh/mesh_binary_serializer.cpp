#include "nuri/resources/storage/mesh/mesh_binary_serializer.h"
#include "nuri/core/pmr_scratch.h"
#include "nuri/pch.h"
#include "nuri/resources/storage/binary_io.h"
#include "nuri/resources/storage/mesh/mesh_binary_format.h"
namespace nuri {
namespace {
constexpr uint64_t kMeshBinarySectionAlignment = 16u;
[[nodiscard]] constexpr uint64_t alignSection(uint64_t value) {
  return (value + kMeshBinarySectionAlignment - 1u) &
         ~(kMeshBinarySectionAlignment - 1u);
}
template <typename T, typename... Args>
  requires(!std::is_same_v<T, MeshBinaryDecodedMesh>)
[[nodiscard]] Result<T, std::string> makeSerializerError(Args &&...args) {
  std::ostringstream oss;
  (oss << ... << std::forward<Args>(args));
  return Result<T, std::string>::makeError(oss.str());
}
template <typename T, typename... Args>
[[nodiscard]] Result<T, MeshBinaryDeserializeError>
makeDeserializeError(Args &&...args) {
  std::ostringstream oss;
  (oss << ... << std::forward<Args>(args));
  return Result<T, MeshBinaryDeserializeError>::makeError(
      MeshBinaryDeserializeError{
          .code = MeshBinaryDeserializeErrorCode::InvalidData,
          .message = oss.str(),
      });
}
template <typename T, typename... Args>
[[nodiscard]] Result<T, MeshBinaryDeserializeError>
makeDeserializeError(MeshBinaryDeserializeErrorCode code, Args &&...args) {
  std::ostringstream oss;
  (oss << ... << std::forward<Args>(args));
  return Result<T, MeshBinaryDeserializeError>::makeError(
      MeshBinaryDeserializeError{
          .code = code,
          .message = oss.str(),
      });
}
[[nodiscard]] Result<std::vector<std::byte>, std::string>
encodeVertexBuffer(std::span<const std::byte> source, uint32_t stride) {
  const size_t count = source.size() / stride;
  std::vector<std::byte> encoded(
      meshopt_encodeVertexBufferBound(count, stride));
  const size_t size = meshopt_encodeVertexBuffer(
      reinterpret_cast<unsigned char *>(encoded.data()), encoded.size(),
      source.data(), count, stride);
  if (size == 0u) {
    return makeSerializerError<std::vector<std::byte>>(
        "meshBinarySerialize: vertex compression failed");
  }
  encoded.resize(size);
  return Result<std::vector<std::byte>, std::string>::makeResult(
      std::move(encoded));
}
[[nodiscard]] Result<std::vector<std::byte>, std::string>
encodeIndexBuffer(std::span<const uint32_t> source, uint32_t vertexCount) {
  std::vector<std::byte> encoded(
      meshopt_encodeIndexBufferBound(source.size(), vertexCount));
  const size_t size = meshopt_encodeIndexBuffer(
      reinterpret_cast<unsigned char *>(encoded.data()), encoded.size(),
      source.data(), source.size());
  if (size == 0u) {
    return makeSerializerError<std::vector<std::byte>>(
        "meshBinarySerialize: index compression failed");
  }
  encoded.resize(size);
  return Result<std::vector<std::byte>, std::string>::makeResult(
      std::move(encoded));
}
[[nodiscard]] Result<std::vector<std::byte>, std::string>
decodeVertexBuffer(std::span<const std::byte> encoded, uint32_t count,
                   uint32_t stride) {
  if (count == 0u) {
    return Result<std::vector<std::byte>, std::string>::makeResult({});
  }
  if (stride == 0u || encoded.empty()) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "meshBinaryDeserialize: invalid compressed vertex buffer");
  }
  std::vector<std::byte> decoded(static_cast<size_t>(count) * stride);
  if (meshopt_decodeVertexBuffer(
          decoded.data(), count, stride,
          reinterpret_cast<const unsigned char *>(encoded.data()),
          encoded.size()) != 0) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "meshBinaryDeserialize: vertex decompression failed");
  }
  return Result<std::vector<std::byte>, std::string>::makeResult(
      std::move(decoded));
}
[[nodiscard]] Result<std::vector<std::byte>, std::string>
decodeIndexBuffer(std::span<const std::byte> encoded, uint32_t count,
                  uint32_t stride) {
  if (count == 0u) {
    return Result<std::vector<std::byte>, std::string>::makeResult({});
  }
  if (stride != sizeof(uint32_t) || encoded.empty()) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "meshBinaryDeserialize: invalid compressed index buffer");
  }
  std::vector<std::byte> decoded(static_cast<size_t>(count) * stride);
  if (meshopt_decodeIndexBuffer(
          decoded.data(), count, stride,
          reinterpret_cast<const unsigned char *>(encoded.data()),
          encoded.size()) != 0) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "meshBinaryDeserialize: index decompression failed");
  }
  return Result<std::vector<std::byte>, std::string>::makeResult(
      std::move(decoded));
}
struct SerializedSection {
  uint32_t fourcc = 0;
  uint32_t flags = 0;
  uint32_t count = 0;
  uint32_t stride = 0;
  std::vector<std::byte> payload;
};
enum class SectionId : size_t {
  Vlay,
  Smes,
  Lods,
  Vbuf,
  Ibuf,
  Vinf,
  Vdec,
  Mmta,
  Mdel,
  Mlds,
  Mlvi,
  Mlpi,
  Mlrg,
  Count,
};
constexpr std::array<uint32_t, static_cast<size_t>(SectionId::Count)>
    kSectionFourcc{
        kMeshBinarySectionVlay, kMeshBinarySectionSmes, kMeshBinarySectionLods,
        kMeshBinarySectionVbuf, kMeshBinarySectionIbuf, kMeshBinarySectionVinf,
        kMeshBinarySectionVdec, kMeshBinarySectionMmta, kMeshBinarySectionMdel,
        kMeshBinarySectionMlds, kMeshBinarySectionMlvi, kMeshBinarySectionMlpi,
        kMeshBinarySectionMlrg,
    };
using SectionTable = std::array<const MeshBinarySectionTocEntry *,
                                static_cast<size_t>(SectionId::Count)>;
[[nodiscard]] constexpr size_t sectionIndex(SectionId id) {
  return static_cast<size_t>(id);
}
[[nodiscard]] uint32_t vertexStrideForLayoutId(uint32_t layoutId) {
  switch (layoutId) {
  case kMeshBinaryLayoutIdStaticQuantized20:
    return kMeshBinaryStaticVertexStrideBytes;
  case kMeshBinaryLayoutIdAnimatedFloat24:
    return kMeshBinaryAnimatedVertexStrideBytes;
  case kMeshBinaryLayoutIdAnimatedFloat32:
    return kMeshBinaryAnimatedFloat32VertexStrideBytes;
  default:
    return 0u;
  }
}
[[nodiscard]] Result<SerializedSection, std::string>
buildVertexLayoutSection(const MeshBinarySerializeInput &input) {
  SerializedSection section{};
  section.fourcc = kMeshBinarySectionVlay;
  section.flags = 0;
  section.count = 1;
  section.stride = sizeof(MeshBinaryVertexLayoutRecord);
  MeshBinaryVertexLayoutRecord record{};
  record.layoutId = input.vertexLayoutId;
  record.strideBytes = vertexStrideForLayoutId(input.vertexLayoutId);
  if (input.vertexLayoutId == kMeshBinaryLayoutIdAnimatedFloat32) {
    record.attributeMask |= kMeshBinaryPackedAttributeTangent;
  }
  if (record.strideBytes == 0u) {
    return makeSerializerError<SerializedSection>(
        "meshBinarySerialize: unsupported vertex layout id");
  }
  appendPod(section.payload, record);
  return Result<SerializedSection, std::string>::makeResult(std::move(section));
}
[[nodiscard]] Result<std::pair<SerializedSection, SerializedSection>,
                     std::string>
buildSubmeshAndLodSections(std::span<const Submesh> submeshes,
                           uint32_t vertexLayoutId) {
  SerializedSection submeshSection{};
  submeshSection.fourcc = kMeshBinarySectionSmes;
  submeshSection.flags = 0;
  if (submeshes.size() > std::numeric_limits<uint32_t>::max()) {
    return makeSerializerError<std::pair<SerializedSection, SerializedSection>>(
        "meshBinarySerialize: submesh count exceeds uint32");
  }
  submeshSection.count = static_cast<uint32_t>(submeshes.size());
  submeshSection.stride = sizeof(MeshBinarySubmeshRecord);
  SerializedSection lodSection{};
  lodSection.fourcc = kMeshBinarySectionLods;
  lodSection.flags = 0;
  lodSection.stride = sizeof(MeshBinaryLodRecord);
  uint32_t lodCount = 0;
  for (const Submesh &submesh : submeshes) {
    if (submesh.lodCount == 0 || submesh.lodCount > Submesh::kMaxLodCount) {
      return makeSerializerError<
          std::pair<SerializedSection, SerializedSection>>(
          "meshBinarySerialize: invalid submesh LOD count");
    }
    MeshBinarySubmeshRecord submeshRecord{};
    submeshRecord.vertexOffset = submesh.vertexOffset;
    submeshRecord.vertexCount = submesh.vertexCount;
    submeshRecord.materialIndex = submesh.materialIndex;
    submeshRecord.lodFirst = lodCount;
    submeshRecord.lodCount = submesh.lodCount;
    submeshRecord.layoutId = vertexLayoutId;
    submeshRecord.morphTargetFirst = submesh.morphTargetFirst;
    submeshRecord.morphTargetCount = submesh.morphTargetCount;
    submeshRecord.boundsMin[0] = submesh.bounds.min_.x;
    submeshRecord.boundsMin[1] = submesh.bounds.min_.y;
    submeshRecord.boundsMin[2] = submesh.bounds.min_.z;
    submeshRecord.boundsMax[0] = submesh.bounds.max_.x;
    submeshRecord.boundsMax[1] = submesh.bounds.max_.y;
    submeshRecord.boundsMax[2] = submesh.bounds.max_.z;
    submeshRecord.authoredScale[0] = submesh.authoredScale.x;
    submeshRecord.authoredScale[1] = submesh.authoredScale.y;
    submeshRecord.authoredScale[2] = submesh.authoredScale.z;
    appendPod(submeshSection.payload, submeshRecord);
    for (uint32_t lodIndex = 0; lodIndex < submesh.lodCount; ++lodIndex) {
      const SubmeshLod &lod = submesh.lods[lodIndex];
      MeshBinaryLodRecord lodRecord{};
      lodRecord.indexOffset = lod.indexOffset;
      lodRecord.indexCount = lod.indexCount;
      lodRecord.error = lod.error;
      appendPod(lodSection.payload, lodRecord);
    }
    if (lodCount > (std::numeric_limits<uint32_t>::max() - submesh.lodCount)) {
      return makeSerializerError<
          std::pair<SerializedSection, SerializedSection>>(
          "meshBinarySerialize: total LOD count overflow");
    }
    lodCount += submesh.lodCount;
  }
  lodSection.count = lodCount;
  return Result<std::pair<SerializedSection, SerializedSection>, std::string>::
      makeResult(
          std::make_pair(std::move(submeshSection), std::move(lodSection)));
}
[[nodiscard]] Result<SerializedSection, std::string>
buildMeshletRangeSection(std::span<const Submesh> submeshes) {
  SerializedSection section{};
  section.fourcc = kMeshBinarySectionMlrg;
  section.flags = 0;
  section.stride = sizeof(MeshBinaryLodMeshletRangeRecord);
  uint32_t rangeCount = 0;
  for (const Submesh &submesh : submeshes) {
    if (rangeCount > std::numeric_limits<uint32_t>::max() - submesh.lodCount) {
      return makeSerializerError<SerializedSection>(
          "meshBinarySerialize: total meshlet LOD range count overflow");
    }
    rangeCount += submesh.lodCount;
    for (uint32_t lodIndex = 0; lodIndex < submesh.lodCount; ++lodIndex) {
      const SubmeshLod &lod = submesh.lods[lodIndex];
      appendPod(section.payload, MeshBinaryLodMeshletRangeRecord{
                                     .meshletOffset = lod.meshletOffset,
                                     .meshletCount = lod.meshletCount,
                                 });
    }
  }
  section.count = rangeCount;
  return Result<SerializedSection, std::string>::makeResult(std::move(section));
}
[[nodiscard]] MeshBinaryMeshletRecord
toMeshletRecord(const MeshletDescriptor &meshlet) {
  static_assert(sizeof(MeshBinaryMeshletRecord) == sizeof(MeshletDescriptor));
  return std::bit_cast<MeshBinaryMeshletRecord>(meshlet);
}
[[nodiscard]] MeshletDescriptor
fromMeshletRecord(const MeshBinaryMeshletRecord &record) {
  return std::bit_cast<MeshletDescriptor>(record);
}
template <typename T>
[[nodiscard]] Result<SerializedSection, std::string>
buildRawPodSection(uint32_t fourcc, std::span<const T> values,
                   std::string_view sectionName) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (values.size() > std::numeric_limits<uint32_t>::max()) {
    return makeSerializerError<SerializedSection>(
        "meshBinarySerialize: ", std::string(sectionName),
        " count exceeds uint32_t");
  }
  SerializedSection section{};
  section.fourcc = fourcc;
  section.flags = 0;
  section.count = static_cast<uint32_t>(values.size());
  section.stride = sizeof(T);
  appendPodArray(section.payload, values);
  return Result<SerializedSection, std::string>::makeResult(std::move(section));
}
[[nodiscard]] Result<SerializedSection, std::string>
buildCompressedBufferSection(uint32_t fourcc,
                             std::span<const std::byte> encoded,
                             uint32_t elementCount, uint32_t elementStrideBytes,
                             std::string_view sectionName) {
  if (encoded.size() >
      static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    return makeSerializerError<SerializedSection>(
        "meshBinarySerialize: encoded ", std::string(sectionName),
        " payload is too large");
  }
  SerializedSection section{};
  section.fourcc = fourcc;
  section.flags = kMeshBinarySectionFlagCompressed;
  section.count = 1;
  section.stride = sizeof(MeshBinaryBufferSectionHeader);
  MeshBinaryBufferSectionHeader meta{};
  meta.elementCount = elementCount;
  meta.elementStrideBytes = elementStrideBytes;
  meta.encodedSizeBytes = static_cast<uint32_t>(encoded.size());
  appendPod(section.payload, meta);
  appendPodArray(section.payload, encoded);
  return Result<SerializedSection, std::string>::makeResult(std::move(section));
}
[[nodiscard]] Result<SerializedSection, std::string>
buildOptionalVertexBufferSection(
    uint32_t fourcc, const BufferLayout<std::span<const std::byte>> &layout,
    std::string_view sectionName) {
  auto encodedResult = encodeVertexBuffer(layout.data, layout.strideBytes);
  if (encodedResult.hasError()) {
    return makeSerializerError<SerializedSection>(encodedResult.error());
  }
  return buildCompressedBufferSection(fourcc, encodedResult.value(),
                                      layout.count, layout.strideBytes,
                                      sectionName);
}
[[nodiscard]] Result<SerializedSection, std::string>
buildIndexBufferSection(std::span<const uint32_t> indices,
                        uint32_t vertexCount) {
  auto encodedResult = encodeIndexBuffer(indices, vertexCount);
  if (encodedResult.hasError()) {
    return makeSerializerError<SerializedSection>(encodedResult.error());
  }
  if (indices.size() > std::numeric_limits<uint32_t>::max()) {
    return makeSerializerError<SerializedSection>(
        "meshBinarySerialize: index count exceeds uint32_t");
  }
  return buildCompressedBufferSection(
      kMeshBinarySectionIbuf, encodedResult.value(),
      static_cast<uint32_t>(indices.size()), sizeof(uint32_t), "index");
}
[[nodiscard]] bool
sectionSizeMatchesCountStride(const MeshBinarySectionTocEntry &entry) {
  return static_cast<uint64_t>(entry.count) * entry.stride == entry.sizeBytes;
}
[[nodiscard]] Result<SectionTable, MeshBinaryDeserializeError>
indexSections(std::span<const MeshBinarySectionTocEntry> toc, size_t fileSize) {
  SectionTable sections{};
  for (const MeshBinarySectionTocEntry &entry : toc) {
    if (!binaryRangeValid(fileSize, entry.offset, entry.sizeBytes)) {
      return makeDeserializeError<SectionTable>(
          "meshBinaryDeserialize: invalid section range");
    }
    const auto found = std::ranges::find(kSectionFourcc, entry.fourcc);
    if (found == kSectionFourcc.end()) {
      continue;
    }
    const size_t index = static_cast<size_t>(found - kSectionFourcc.begin());
    if (sections[index] != nullptr) {
      return makeDeserializeError<SectionTable>(
          "meshBinaryDeserialize: duplicate section");
    }
    sections[index] = &entry;
  }
  if (std::ranges::any_of(sections.begin(),
                          sections.begin() + sectionIndex(SectionId::Vinf),
                          [](const auto *entry) { return entry == nullptr; })) {
    return makeDeserializeError<SectionTable>(
        "meshBinaryDeserialize: missing required section");
  }
  return Result<SectionTable, MeshBinaryDeserializeError>::makeResult(sections);
}
[[nodiscard]] Result<std::pmr::vector<MeshBinarySubmeshRecord>,
                     MeshBinaryDeserializeError>
readSubmeshRecords(std::span<const std::byte> fileBytes,
                   const MeshBinarySectionTocEntry &entry,
                   std::pmr::memory_resource *memory) {
  std::pmr::vector<MeshBinarySubmeshRecord> records(memory);
  if (entry.stride == sizeof(MeshBinarySubmeshRecord)) {
    readPodArrayAt(fileBytes, entry.offset, entry.count, records);
    return Result<std::pmr::vector<MeshBinarySubmeshRecord>,
                  MeshBinaryDeserializeError>::makeResult(std::move(records));
  }
  if (entry.stride == sizeof(MeshBinarySubmeshRecordV1)) {
    std::pmr::vector<MeshBinarySubmeshRecordV1> legacyRecords(memory);
    readPodArrayAt(fileBytes, entry.offset, entry.count, legacyRecords);
    records.reserve(legacyRecords.size());
    for (const MeshBinarySubmeshRecordV1 &legacy : legacyRecords) {
      MeshBinarySubmeshRecord record{};
      record.materialIndex = legacy.materialIndex;
      record.lodFirst = legacy.lodFirst;
      record.lodCount = legacy.lodCount;
      record.layoutId = legacy.layoutId;
      std::memcpy(record.boundsMin, legacy.boundsMin, sizeof(record.boundsMin));
      std::memcpy(record.boundsMax, legacy.boundsMax, sizeof(record.boundsMax));
      std::memcpy(record.authoredScale, legacy.authoredScale,
                  sizeof(record.authoredScale));
      records.push_back(record);
    }
    return Result<std::pmr::vector<MeshBinarySubmeshRecord>,
                  MeshBinaryDeserializeError>::makeResult(std::move(records));
  }
  if (entry.stride != sizeof(MeshBinarySubmeshRecordV0)) {
    return makeDeserializeError<std::pmr::vector<MeshBinarySubmeshRecord>>(
        "meshBinaryDeserialize: unsupported submesh record stride");
  }
  std::pmr::vector<MeshBinarySubmeshRecordV0> legacyRecords(memory);
  readPodArrayAt(fileBytes, entry.offset, entry.count, legacyRecords);
  records.reserve(legacyRecords.size());
  for (const MeshBinarySubmeshRecordV0 &legacy : legacyRecords) {
    MeshBinarySubmeshRecord record{};
    record.materialIndex = legacy.materialIndex;
    record.lodFirst = legacy.lodFirst;
    record.lodCount = legacy.lodCount;
    record.layoutId = legacy.layoutId;
    std::memcpy(record.boundsMin, legacy.boundsMin, sizeof(record.boundsMin));
    std::memcpy(record.boundsMax, legacy.boundsMax, sizeof(record.boundsMax));
    records.push_back(record);
  }
  return Result<std::pmr::vector<MeshBinarySubmeshRecord>,
                MeshBinaryDeserializeError>::makeResult(std::move(records));
}
struct DecodedBuffer {
  MeshBinaryBufferSectionHeader meta;
  std::vector<std::byte> bytes;
};
[[nodiscard]] Result<DecodedBuffer, MeshBinaryDeserializeError>
decodeBufferSection(std::span<const std::byte> fileBytes,
                    const MeshBinarySectionTocEntry &entry, bool index) {
  if (entry.count != 1 ||
      entry.stride != sizeof(MeshBinaryBufferSectionHeader) ||
      entry.sizeBytes < sizeof(MeshBinaryBufferSectionHeader)) {
    return makeDeserializeError<DecodedBuffer>(
        "meshBinaryDeserialize: invalid buffer section");
  }
  const auto meta =
      readPodAt<MeshBinaryBufferSectionHeader>(fileBytes, entry.offset);
  if (sizeof(MeshBinaryBufferSectionHeader) + meta.encodedSizeBytes !=
      entry.sizeBytes) {
    return makeDeserializeError<DecodedBuffer>(
        "meshBinaryDeserialize: invalid buffer payload size");
  }
  const std::span encoded(fileBytes.data() + static_cast<size_t>(entry.offset) +
                              sizeof(meta),
                          meta.encodedSizeBytes);
  auto decoded = index ? decodeIndexBuffer(encoded, meta.elementCount,
                                           meta.elementStrideBytes)
                       : decodeVertexBuffer(encoded, meta.elementCount,
                                            meta.elementStrideBytes);
  if (decoded.hasError()) {
    return makeDeserializeError<DecodedBuffer>(decoded.error());
  }
  return Result<DecodedBuffer, MeshBinaryDeserializeError>::makeResult(
      DecodedBuffer{meta, std::move(decoded.value())});
}
} // namespace

Result<std::vector<std::byte>, std::string>
meshBinarySerialize(const MeshBinarySerializeInput &input) {
  ScratchArena scratch;
  ScopedScratch scopedScratch(scratch);
  const uint32_t expectedVertexStride =
      vertexStrideForLayoutId(input.vertexLayoutId);
  if (input.vertices.empty() || !input.vertices.validate() ||
      input.indices.empty() || expectedVertexStride == 0u ||
      input.vertices.strideBytes != expectedVertexStride) {
    return makeSerializerError<std::vector<std::byte>>(
        "meshBinarySerialize: invalid vertex/index layout");
  }
  for (const Submesh &submesh : input.submeshes) {
    if (submesh.lodCount == 0 || submesh.lodCount > Submesh::kMaxLodCount) {
      return makeSerializerError<std::vector<std::byte>>(
          "meshBinarySerialize: invalid submesh LOD count");
    }
    for (uint32_t lodIndex = 0; lodIndex < submesh.lodCount; ++lodIndex) {
      const SubmeshLod &lod = submesh.lods[lodIndex];
      if (!binaryRangeValid(input.indices.size(), lod.indexOffset,
                            lod.indexCount)) {
        return makeSerializerError<std::vector<std::byte>>(
            "meshBinarySerialize: submesh index range out of bounds");
      }
    }
  }
  const bool hasMeshlets = !input.meshlets.empty();
  if (hasMeshlets != !input.meshletVertexIndices.empty() ||
      hasMeshlets != !input.meshletPrimitiveIndices.empty()) {
    return makeSerializerError<std::vector<std::byte>>(
        "meshBinarySerialize: incomplete meshlet data");
  }
  if (hasMeshlets) {
    for (const MeshletDescriptor &meshlet : input.meshlets) {
      if (!binaryRangeValid(input.meshletVertexIndices.size(),
                            meshlet.vertexOffset, meshlet.vertexCount)) {
        return makeSerializerError<std::vector<std::byte>>(
            "meshBinarySerialize: meshlet vertex range out of bounds");
      }
      if (!binaryRangeValid(
              input.meshletPrimitiveIndices.size(), meshlet.primitiveOffset,
              static_cast<uint64_t>(meshlet.primitiveCount) * 3u)) {
        return makeSerializerError<std::vector<std::byte>>(
            "meshBinarySerialize: meshlet primitive range out of bounds");
      }
    }
    for (const Submesh &submesh : input.submeshes) {
      for (uint32_t lodIndex = 0; lodIndex < submesh.lodCount; ++lodIndex) {
        const SubmeshLod &lod = submesh.lods[lodIndex];
        if (!binaryRangeValid(input.meshlets.size(), lod.meshletOffset,
                              lod.meshletCount)) {
          return makeSerializerError<std::vector<std::byte>>(
              "meshBinarySerialize: submesh meshlet range out of bounds");
        }
      }
    }
  }
  const bool staticLayout =
      input.vertexLayoutId == kMeshBinaryLayoutIdStaticQuantized20;
  if (!input.skinInfluences.validate() ||
      !input.staticVertexDecode.validate() || !input.morphMeta.validate() ||
      !input.morphDeltas.validate() ||
      staticLayout != !input.staticVertexDecode.empty() ||
      (staticLayout &&
       input.staticVertexDecode.count != input.submeshes.size()) ||
      input.morphMeta.empty() != input.morphDeltas.empty() ||
      (!input.morphMeta.empty() &&
       (input.morphMeta.count != 1u ||
        input.morphMeta.strideBytes != sizeof(MeshBinaryMorphMetaRecord)))) {
    return makeSerializerError<std::vector<std::byte>>(
        "meshBinarySerialize: invalid optional vertex layout");
  }
  std::pmr::vector<SerializedSection> sections(scopedScratch.resource());
  sections.reserve(12);
  std::string sectionError;
  const auto addSection = [&](Result<SerializedSection, std::string> result) {
    if (result.hasError()) {
      sectionError = result.error();
      return false;
    }
    sections.push_back(std::move(result.value()));
    return true;
  };
  if (!addSection(buildVertexLayoutSection(input))) {
    return makeSerializerError<std::vector<std::byte>>(sectionError);
  }
  auto smesLodsResult =
      buildSubmeshAndLodSections(input.submeshes, input.vertexLayoutId);
  if (smesLodsResult.hasError()) {
    return makeSerializerError<std::vector<std::byte>>(smesLodsResult.error());
  }
  auto smesLodsSections = std::move(smesLodsResult.value());
  sections.push_back(std::move(smesLodsSections.first));
  sections.push_back(std::move(smesLodsSections.second));
  if (!addSection(buildOptionalVertexBufferSection(kMeshBinarySectionVbuf,
                                                   input.vertices, "vertex")) ||
      !addSection(
          buildIndexBufferSection(input.indices, input.vertices.count))) {
    return makeSerializerError<std::vector<std::byte>>(sectionError);
  }
  if (!input.meshlets.empty()) {
    std::vector<MeshBinaryMeshletRecord> meshletRecords;
    meshletRecords.reserve(input.meshlets.size());
    for (const MeshletDescriptor &meshlet : input.meshlets) {
      meshletRecords.push_back(toMeshletRecord(meshlet));
    }
    if (!addSection(buildRawPodSection<MeshBinaryMeshletRecord>(
            kMeshBinarySectionMlds, meshletRecords, "meshlet descriptor")) ||
        !addSection(buildRawPodSection<uint32_t>(kMeshBinarySectionMlvi,
                                                 input.meshletVertexIndices,
                                                 "meshlet vertex index")) ||
        !addSection(buildRawPodSection<uint8_t>(kMeshBinarySectionMlpi,
                                                input.meshletPrimitiveIndices,
                                                "meshlet primitive index")) ||
        !addSection(buildMeshletRangeSection(input.submeshes))) {
      return makeSerializerError<std::vector<std::byte>>(sectionError);
    }
  }
  const std::array optionalBuffers{
      std::tuple{kMeshBinarySectionVinf, &input.skinInfluences,
                 std::string_view("skin influence")},
      std::tuple{kMeshBinarySectionVdec, &input.staticVertexDecode,
                 std::string_view("static vertex decode")},
      std::tuple{kMeshBinarySectionMmta, &input.morphMeta,
                 std::string_view("morph meta")},
      std::tuple{kMeshBinarySectionMdel, &input.morphDeltas,
                 std::string_view("morph delta")},
  };
  for (const auto &[fourcc, layout, name] : optionalBuffers) {
    if (!layout->empty() &&
        !addSection(buildOptionalVertexBufferSection(fourcc, *layout, name))) {
      return makeSerializerError<std::vector<std::byte>>(sectionError);
    }
  }
  const uint64_t tocCount = static_cast<uint64_t>(sections.size());
  const uint64_t tocBytes = tocCount * sizeof(MeshBinarySectionTocEntry);
  const uint64_t sectionStart =
      alignSection(sizeof(MeshBinaryHeader) + tocBytes);
  std::pmr::vector<MeshBinarySectionTocEntry> tocEntries(
      scopedScratch.resource());
  tocEntries.resize(sections.size());
  uint64_t cursor = sectionStart;
  for (size_t i = 0; i < sections.size(); ++i) {
    cursor = alignSection(cursor);
    const SerializedSection &section = sections[i];
    MeshBinarySectionTocEntry &entry = tocEntries[i];
    entry.fourcc = section.fourcc;
    entry.flags = section.flags;
    entry.offset = cursor;
    entry.sizeBytes = static_cast<uint64_t>(section.payload.size());
    entry.count = section.count;
    entry.stride = section.stride;
    cursor += entry.sizeBytes;
  }
  const uint64_t fileSize = alignSection(cursor);
  if (fileSize > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return makeSerializerError<std::vector<std::byte>>(
        "meshBinarySerialize: file size exceeds platform limits");
  }
  MeshBinaryHeader header{};
  header.magic = kMeshBinaryMagic;
  header.majorVersion = kMeshBinaryFormatMajorVersion;
  header.minorVersion = kMeshBinaryFormatMinorVersion;
  header.headerSize = static_cast<uint16_t>(sizeof(MeshBinaryHeader));
  header.tocEntrySize =
      static_cast<uint16_t>(sizeof(MeshBinarySectionTocEntry));
  header.flags =
      kBinaryHeaderFlagLittleEndian | kMeshBinaryHeaderFlagCompressed;
  header.fileSize = fileSize;
  header.tocOffset = sizeof(MeshBinaryHeader);
  header.tocCount = static_cast<uint32_t>(tocEntries.size());
  header.sourcePathHash = input.sourcePathHash;
  header.importOptionsHash = input.importOptionsHash;
  header.sourceSizeBytes = input.sourceSizeBytes;
  header.sourceMtimeNs = input.sourceMtimeNs;
  header.modelBoundsMin[0] = input.bounds.min_.x;
  header.modelBoundsMin[1] = input.bounds.min_.y;
  header.modelBoundsMin[2] = input.bounds.min_.z;
  header.modelBoundsMax[0] = input.bounds.max_.x;
  header.modelBoundsMax[1] = input.bounds.max_.y;
  header.modelBoundsMax[2] = input.bounds.max_.z;
  std::vector<std::byte> fileBytes;
  fileBytes.resize(static_cast<size_t>(fileSize), std::byte{0});
  std::memcpy(fileBytes.data(), &header, sizeof(header));
  std::memcpy(fileBytes.data() + static_cast<size_t>(header.tocOffset),
              tocEntries.data(), static_cast<size_t>(tocBytes));
  for (size_t i = 0; i < sections.size(); ++i) {
    const MeshBinarySectionTocEntry &entry = tocEntries[i];
    const SerializedSection &section = sections[i];
    if (!section.payload.empty()) {
      std::memcpy(fileBytes.data() + static_cast<size_t>(entry.offset),
                  section.payload.data(), section.payload.size());
    }
  }
  return Result<std::vector<std::byte>, std::string>::makeResult(
      std::move(fileBytes));
}

Result<MeshBinaryDecodedMesh, MeshBinaryDeserializeError>
meshBinaryDeserialize(std::span<const std::byte> fileBytes,
                      const MeshBinaryDeserializeContext &context) {
  ScratchArena scratch;
  ScopedScratch scopedScratch(scratch);
  if (fileBytes.size() < sizeof(MeshBinaryHeader)) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: file too small");
  }
  MeshBinaryHeader header{};
  std::memcpy(&header, fileBytes.data(), sizeof(header));
  if (header.magic != kMeshBinaryMagic ||
      header.majorVersion != kMeshBinaryFormatMajorVersion ||
      header.minorVersion > kMeshBinaryFormatMinorVersion ||
      (header.flags & kBinaryHeaderFlagLittleEndian) == 0u ||
      header.headerSize != sizeof(MeshBinaryHeader) ||
      header.tocEntrySize != sizeof(MeshBinarySectionTocEntry) ||
      header.fileSize != fileBytes.size() ||
      header.sourcePathHash != context.expectedSourcePathHash ||
      header.importOptionsHash != context.expectedImportOptionsHash) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: invalid cache header");
  }
  if (context.validateSourceFingerprint && context.sourceExists &&
      (header.sourceSizeBytes != context.sourceSizeBytes ||
       header.sourceMtimeNs != context.sourceMtimeNs)) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        MeshBinaryDeserializeErrorCode::StaleCache,
        "meshBinaryDeserialize: cache is stale for current source file");
  }
  const uint64_t tocBytes = static_cast<uint64_t>(header.tocCount) *
                            sizeof(MeshBinarySectionTocEntry);
  if (!binaryRangeValid(fileBytes.size(), header.tocOffset, tocBytes)) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: invalid TOC bounds");
  }
  std::pmr::vector<MeshBinarySectionTocEntry> toc(scopedScratch.resource());
  readPodArrayAt(fileBytes, header.tocOffset, header.tocCount, toc);
  auto sectionResult = indexSections(toc, fileBytes.size());
  if (sectionResult.hasError()) {
    return Result<MeshBinaryDecodedMesh, MeshBinaryDeserializeError>::makeError(
        sectionResult.error());
  }
  const SectionTable &sections = sectionResult.value();
  const auto &vlayEntry = *sections[sectionIndex(SectionId::Vlay)];
  const auto &smesEntry = *sections[sectionIndex(SectionId::Smes)];
  const auto &lodsEntry = *sections[sectionIndex(SectionId::Lods)];
  const auto &vbufEntry = *sections[sectionIndex(SectionId::Vbuf)];
  const auto &ibufEntry = *sections[sectionIndex(SectionId::Ibuf)];
  if (vlayEntry.count != 1u ||
      vlayEntry.stride != sizeof(MeshBinaryVertexLayoutRecord) ||
      !sectionSizeMatchesCountStride(vlayEntry) ||
      !sectionSizeMatchesCountStride(smesEntry) ||
      (smesEntry.stride != sizeof(MeshBinarySubmeshRecord) &&
       smesEntry.stride != sizeof(MeshBinarySubmeshRecordV1) &&
       smesEntry.stride != sizeof(MeshBinarySubmeshRecordV0)) ||
      lodsEntry.stride != sizeof(MeshBinaryLodRecord) ||
      !sectionSizeMatchesCountStride(lodsEntry)) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: invalid structural section layout");
  }
  const auto layoutRecord =
      readPodAt<MeshBinaryVertexLayoutRecord>(fileBytes, vlayEntry.offset);
  const uint32_t expectedVertexStride =
      vertexStrideForLayoutId(layoutRecord.layoutId);
  if (expectedVertexStride == 0u ||
      layoutRecord.strideBytes != expectedVertexStride) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: unsupported vertex layout");
  }
  auto vbufResult = decodeBufferSection(fileBytes, vbufEntry, false);
  auto ibufResult = decodeBufferSection(fileBytes, ibufEntry, true);
  if (vbufResult.hasError() || ibufResult.hasError()) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: invalid vertex/index buffer");
  }
  DecodedBuffer vbuf = std::move(vbufResult.value());
  DecodedBuffer ibuf = std::move(ibufResult.value());
  const auto &vbufMeta = vbuf.meta;
  const auto *vinfEntry = sections[sectionIndex(SectionId::Vinf)];
  const auto *vdecEntry = sections[sectionIndex(SectionId::Vdec)];
  const auto *mmtaEntry = sections[sectionIndex(SectionId::Mmta)];
  const auto *mdelEntry = sections[sectionIndex(SectionId::Mdel)];
  const auto *mldsEntry = sections[sectionIndex(SectionId::Mlds)];
  const auto *mlviEntry = sections[sectionIndex(SectionId::Mlvi)];
  const auto *mlpiEntry = sections[sectionIndex(SectionId::Mlpi)];
  const auto *mlrgEntry = sections[sectionIndex(SectionId::Mlrg)];
  if ((mmtaEntry == nullptr) != (mdelEntry == nullptr)) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: morph meta and morph delta sections must be "
        "present together");
  }
  const uint32_t meshletSectionPresence =
      (mldsEntry != nullptr ? 1u : 0u) + (mlviEntry != nullptr ? 1u : 0u) +
      (mlpiEntry != nullptr ? 1u : 0u) + (mlrgEntry != nullptr ? 1u : 0u);
  if (meshletSectionPresence != 0u && meshletSectionPresence != 4u) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: meshlet sections must be present together");
  }
  std::array<std::optional<DecodedBuffer>, 4> optionalBuffers;
  const std::array optionalEntries{vinfEntry, vdecEntry, mmtaEntry, mdelEntry};
  for (size_t i = 0; i < optionalEntries.size(); ++i) {
    if (optionalEntries[i] == nullptr) {
      continue;
    }
    auto result = decodeBufferSection(fileBytes, *optionalEntries[i], false);
    if (result.hasError()) {
      return Result<MeshBinaryDecodedMesh,
                    MeshBinaryDeserializeError>::makeError(result.error());
    }
    optionalBuffers[i] = std::move(result.value());
  }
  const auto &vinf = optionalBuffers[0];
  const auto &vdec = optionalBuffers[1];
  const auto &mmta = optionalBuffers[2];
  const auto &mdel = optionalBuffers[3];
  std::pmr::vector<MeshBinaryMeshletRecord> meshletRecords(
      scopedScratch.resource());
  std::pmr::vector<MeshBinaryLodMeshletRangeRecord> meshletRangeRecords(
      scopedScratch.resource());
  std::vector<uint32_t> decodedMeshletVertexIndices;
  std::vector<uint8_t> decodedMeshletPrimitiveIndices;
  if (mldsEntry != nullptr) {
    if (mldsEntry->stride != sizeof(MeshBinaryMeshletRecord) ||
        mlviEntry->stride != sizeof(uint32_t) ||
        mlpiEntry->stride != sizeof(uint8_t) ||
        mlrgEntry->stride != sizeof(MeshBinaryLodMeshletRangeRecord)) {
      return makeDeserializeError<MeshBinaryDecodedMesh>(
          "meshBinaryDeserialize: invalid meshlet section");
    }
    readPodArrayAt(fileBytes, mldsEntry->offset, mldsEntry->count,
                   meshletRecords);
    readPodArrayAt(fileBytes, mlviEntry->offset, mlviEntry->count,
                   decodedMeshletVertexIndices);
    readPodArrayAt(fileBytes, mlpiEntry->offset, mlpiEntry->count,
                   decodedMeshletPrimitiveIndices);
    readPodArrayAt(fileBytes, mlrgEntry->offset, mlrgEntry->count,
                   meshletRangeRecords);
  }
  if (vinf && vinf->meta.elementCount != vbufMeta.elementCount) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: skin influence count must match vertex count");
  }
  const bool staticLayout =
      layoutRecord.layoutId == kMeshBinaryLayoutIdStaticQuantized20;
  if (staticLayout != vdec.has_value()) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: invalid static decode section");
  }
  if (mmta) {
    if (mmta->meta.elementCount != 1u ||
        mmta->meta.elementStrideBytes != sizeof(MeshBinaryMorphMetaRecord) ||
        mmta->bytes.size() != sizeof(MeshBinaryMorphMetaRecord)) {
      return makeDeserializeError<MeshBinaryDecodedMesh>(
          "meshBinaryDeserialize: invalid morph meta payload");
    }
    const auto morphMetaRecord =
        readPodAt<MeshBinaryMorphMetaRecord>(mmta->bytes, 0);
    const uint64_t expectedMorphDeltaCount =
        static_cast<uint64_t>(morphMetaRecord.morphTargetCount) *
        morphMetaRecord.vertexCount;
    if (morphMetaRecord.vertexCount != vbufMeta.elementCount ||
        expectedMorphDeltaCount != mdel->meta.elementCount) {
      return makeDeserializeError<MeshBinaryDecodedMesh>(
          "meshBinaryDeserialize: invalid morph counts");
    }
  }
  auto submeshRecordsResult =
      readSubmeshRecords(fileBytes, smesEntry, scopedScratch.resource());
  if (submeshRecordsResult.hasError()) {
    return Result<MeshBinaryDecodedMesh, MeshBinaryDeserializeError>::makeError(
        submeshRecordsResult.error());
  }
  std::pmr::vector<MeshBinarySubmeshRecord> submeshRecords =
      std::move(submeshRecordsResult.value());
  if (staticLayout && vdec->meta.elementCount != submeshRecords.size()) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: static decode count mismatch");
  }
  std::pmr::vector<MeshBinaryLodRecord> lodRecords(scopedScratch.resource());
  readPodArrayAt(fileBytes, lodsEntry.offset, lodsEntry.count, lodRecords);
  MeshBinaryDecodedMesh decoded{};
  decoded.vertexLayoutId = layoutRecord.layoutId;
  decoded.vertices.data = std::move(vbuf.bytes);
  decoded.vertices.count = vbufMeta.elementCount;
  decoded.vertices.strideBytes = vbufMeta.elementStrideBytes;
  decoded.bounds =
      BoundingBox(glm::vec3(header.modelBoundsMin[0], header.modelBoundsMin[1],
                            header.modelBoundsMin[2]),
                  glm::vec3(header.modelBoundsMax[0], header.modelBoundsMax[1],
                            header.modelBoundsMax[2]));
  const auto assignOptional = [](auto &layout, auto &buffer) {
    if (buffer) {
      layout.data = std::move(buffer->bytes);
      layout.count = buffer->meta.elementCount;
      layout.strideBytes = buffer->meta.elementStrideBytes;
    }
  };
  assignOptional(decoded.skinInfluences, optionalBuffers[0]);
  assignOptional(decoded.staticVertexDecode, optionalBuffers[1]);
  assignOptional(decoded.morphMeta, optionalBuffers[2]);
  assignOptional(decoded.morphDeltas, optionalBuffers[3]);
  const std::vector<std::byte> &decodedIndexBytes = ibuf.bytes;
  if ((decodedIndexBytes.size() % sizeof(uint32_t)) != 0u) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: decoded index bytes are not uint32 aligned");
  }
  decoded.indices.resize(decodedIndexBytes.size() / sizeof(uint32_t));
  if (!decoded.indices.empty()) {
    std::memcpy(decoded.indices.data(), decodedIndexBytes.data(),
                decodedIndexBytes.size());
  }
  decoded.meshlets.reserve(meshletRecords.size());
  for (const MeshBinaryMeshletRecord &record : meshletRecords) {
    if (!binaryRangeValid(decodedMeshletVertexIndices.size(),
                          record.vertexOffset, record.vertexCount)) {
      return makeDeserializeError<MeshBinaryDecodedMesh>(
          "meshBinaryDeserialize: meshlet vertex range out of bounds");
    }
    if (!binaryRangeValid(decodedMeshletPrimitiveIndices.size(),
                          record.primitiveOffset,
                          static_cast<uint64_t>(record.primitiveCount) * 3u)) {
      return makeDeserializeError<MeshBinaryDecodedMesh>(
          "meshBinaryDeserialize: meshlet primitive range out of bounds");
    }
    decoded.meshlets.push_back(fromMeshletRecord(record));
  }
  decoded.meshletVertexIndices = std::move(decodedMeshletVertexIndices);
  decoded.meshletPrimitiveIndices = std::move(decodedMeshletPrimitiveIndices);
  decoded.submeshes.reserve(submeshRecords.size());
  uint32_t flatLodIndex = 0u;
  for (const MeshBinarySubmeshRecord &record : submeshRecords) {
    if (record.lodCount == 0 || record.lodCount > Submesh::kMaxLodCount) {
      return makeDeserializeError<MeshBinaryDecodedMesh>(
          "meshBinaryDeserialize: invalid submesh LOD count");
    }
    if (!binaryRangeValid(lodRecords.size(), record.lodFirst,
                          record.lodCount)) {
      return makeDeserializeError<MeshBinaryDecodedMesh>(
          "meshBinaryDeserialize: submesh LOD range out of bounds");
    }
    Submesh submesh{};
    submesh.vertexOffset = record.vertexOffset;
    submesh.vertexCount = record.vertexCount;
    submesh.materialIndex = record.materialIndex;
    submesh.morphTargetFirst = record.morphTargetFirst;
    submesh.morphTargetCount = record.morphTargetCount;
    submesh.bounds =
        BoundingBox(glm::vec3(record.boundsMin[0], record.boundsMin[1],
                              record.boundsMin[2]),
                    glm::vec3(record.boundsMax[0], record.boundsMax[1],
                              record.boundsMax[2]));
    submesh.authoredScale =
        glm::vec3(record.authoredScale[0], record.authoredScale[1],
                  record.authoredScale[2]);
    submesh.lodCount = record.lodCount;
    for (uint32_t lodIndex = 0; lodIndex < record.lodCount; ++lodIndex) {
      const MeshBinaryLodRecord &lodRecord =
          lodRecords[record.lodFirst + lodIndex];
      if (!binaryRangeValid(decoded.indices.size(), lodRecord.indexOffset,
                            lodRecord.indexCount)) {
        return makeDeserializeError<MeshBinaryDecodedMesh>(
            "meshBinaryDeserialize: submesh index range out of bounds");
      }
      submesh.lods[lodIndex] = SubmeshLod{
          .indexOffset = lodRecord.indexOffset,
          .indexCount = lodRecord.indexCount,
          .error = lodRecord.error,
      };
      if (!meshletRangeRecords.empty()) {
        if (flatLodIndex >= meshletRangeRecords.size()) {
          return makeDeserializeError<MeshBinaryDecodedMesh>(
              "meshBinaryDeserialize: MLRG record count mismatch");
        }
        const MeshBinaryLodMeshletRangeRecord &meshletRange =
            meshletRangeRecords[flatLodIndex];
        if (!binaryRangeValid(decoded.meshlets.size(),
                              meshletRange.meshletOffset,
                              meshletRange.meshletCount)) {
          return makeDeserializeError<MeshBinaryDecodedMesh>(
              "meshBinaryDeserialize: submesh meshlet range out of bounds");
        }
        submesh.lods[lodIndex].meshletOffset = meshletRange.meshletOffset;
        submesh.lods[lodIndex].meshletCount = meshletRange.meshletCount;
      }
      ++flatLodIndex;
      if (lodIndex == 0u) {
        submesh.indexOffset = lodRecord.indexOffset;
        submesh.indexCount = lodRecord.indexCount;
      }
    }
    decoded.submeshes.push_back(submesh);
  }
  if (!meshletRangeRecords.empty() &&
      flatLodIndex != meshletRangeRecords.size()) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: unused MLRG records");
  }
  return Result<MeshBinaryDecodedMesh, MeshBinaryDeserializeError>::makeResult(
      std::move(decoded));
}

} // namespace nuri
