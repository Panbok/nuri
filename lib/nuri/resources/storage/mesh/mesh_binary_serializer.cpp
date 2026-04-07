#include "nuri/pch.h"

#include "nuri/resources/storage/mesh/mesh_binary_serializer.h"

#include "nuri/core/pmr_scratch.h"
#include "nuri/resources/storage/mesh/mesh_binary_codec.h"
#include "nuri/resources/storage/mesh/mesh_binary_format.h"
#include "nuri/utils/utils.h"

namespace nuri {
namespace {
constexpr uint64_t kMeshBinarySectionAlignment = 16u;

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

[[nodiscard]] bool checkedAddToU64(uint64_t a, uint64_t b, uint64_t &out) {
  if (a > (std::numeric_limits<uint64_t>::max() - b)) {
    return false;
  }
  out = a + b;
  return true;
}

[[nodiscard]] bool checkedMulToU64(uint64_t a, uint64_t b, uint64_t &out) {
  if (a == 0 || b == 0) {
    out = 0;
    return true;
  }
  if (a > (std::numeric_limits<uint64_t>::max() / b)) {
    return false;
  }
  out = a * b;
  return true;
}

template <typename T>
void appendPod(std::vector<std::byte> &out, const T &value) {
  static_assert(std::is_trivially_copyable_v<T>);
  const size_t offset = out.size();
  out.resize(offset + sizeof(T));
  std::memcpy(out.data() + offset, &value, sizeof(T));
}

template <typename T>
[[nodiscard]] bool appendPodArray(std::vector<std::byte> &out,
                                  std::span<const T> values) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (values.empty()) {
    return true;
  }

  uint64_t byteCount = 0;
  if (!checkedMulToU64(static_cast<uint64_t>(values.size()), sizeof(T),
                       byteCount)) {
    return false;
  }

  const uint64_t offset = static_cast<uint64_t>(out.size());
  uint64_t resizedCount = 0;
  if (!checkedAddToU64(offset, byteCount, resizedCount) ||
      resizedCount >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return false;
  }

  const size_t byteCountSizeT = static_cast<size_t>(byteCount);
  const size_t offsetSizeT = out.size();
  out.resize(static_cast<size_t>(resizedCount));
  std::memcpy(out.data() + offsetSizeT, values.data(), byteCountSizeT);
  return true;
}

template <typename T>
[[nodiscard]] bool readPod(std::span<const std::byte> bytes, uint64_t offset,
                           T &out) {
  static_assert(std::is_trivially_copyable_v<T>);
  uint64_t end = 0;
  if (!checkedAddToU64(offset, sizeof(T), end) ||
      end > static_cast<uint64_t>(bytes.size())) {
    return false;
  }
  std::memcpy(&out, bytes.data() + offset, sizeof(T));
  return true;
}

template <typename T, typename Allocator>
[[nodiscard]] bool readPodArray(std::span<const std::byte> bytes,
                                uint64_t offset, uint32_t count,
                                std::vector<T, Allocator> &out) {
  static_assert(std::is_trivially_copyable_v<T>);
  uint64_t bytesSize = 0;
  if (!checkedMulToU64(static_cast<uint64_t>(count), sizeof(T), bytesSize)) {
    return false;
  }
  uint64_t end = 0;
  if (!checkedAddToU64(offset, bytesSize, end) ||
      end > static_cast<uint64_t>(bytes.size())) {
    return false;
  }
  out.resize(count);
  if (bytesSize > 0) {
    std::memcpy(out.data(), bytes.data() + offset,
                static_cast<size_t>(bytesSize));
  }
  return true;
}

[[nodiscard]] bool isLittleEndianHost() {
#if defined(__cpp_lib_endian) && (__cpp_lib_endian >= 201907L)
  return std::endian::native == std::endian::little;
#else
  constexpr uint16_t value = 0x1;
  return *reinterpret_cast<const uint8_t *>(&value) == 0x1;
#endif
}

struct SerializedSection {
  uint32_t fourcc = 0;
  uint32_t flags = 0;
  uint32_t count = 0;
  uint32_t stride = 0;
  std::vector<std::byte> payload;
};

struct MeshBinaryRequiredSections {
  const MeshBinarySectionTocEntry *vlay = nullptr;
  const MeshBinarySectionTocEntry *smes = nullptr;
  const MeshBinarySectionTocEntry *lods = nullptr;
  const MeshBinarySectionTocEntry *vbuf = nullptr;
  const MeshBinarySectionTocEntry *ibuf = nullptr;
};

[[nodiscard]] uint32_t vertexStrideForLayoutId(uint32_t layoutId) {
  switch (layoutId) {
  case kMeshBinaryLayoutIdStaticQuantized20:
    return kMeshBinaryStaticVertexStrideBytes;
  case kMeshBinaryLayoutIdAnimatedFloat24:
    return kMeshBinaryAnimatedVertexStrideBytes;
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
  if (!appendPodArray(section.payload, encoded)) {
    return makeSerializerError<SerializedSection>(
        "meshBinarySerialize: ", std::string(sectionName),
        " section payload size overflow");
  }

  return Result<SerializedSection, std::string>::makeResult(std::move(section));
}

[[nodiscard]] Result<SerializedSection, std::string> buildVertexBufferSection(
    const BufferLayout<std::span<const std::byte>> &layout) {
  auto encodedResult =
      meshBinaryEncodeVertexBuffer(layout.data, layout.strideBytes);
  if (encodedResult.hasError()) {
    return makeSerializerError<SerializedSection>(encodedResult.error());
  }

  return buildCompressedBufferSection(kMeshBinarySectionVbuf,
                                      encodedResult.value(), layout.count,
                                      layout.strideBytes, "vertex");
}

[[nodiscard]] Result<SerializedSection, std::string>
buildOptionalVertexBufferSection(
    uint32_t fourcc, const BufferLayout<std::span<const std::byte>> &layout,
    std::string_view sectionName) {
  auto encodedResult =
      meshBinaryEncodeVertexBuffer(layout.data, layout.strideBytes);
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
  auto encodedResult = meshBinaryEncodeIndexBuffer(indices, vertexCount);
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

[[nodiscard]] Result<const MeshBinarySectionTocEntry *, std::string>
findRequiredSection(std::span<const MeshBinarySectionTocEntry> toc,
                    uint32_t fourcc, std::string_view name) {
  const MeshBinarySectionTocEntry *result = nullptr;
  for (const MeshBinarySectionTocEntry &entry : toc) {
    if (entry.fourcc != fourcc) {
      continue;
    }
    if (result != nullptr) {
      return makeSerializerError<const MeshBinarySectionTocEntry *>(
          "meshBinaryDeserialize: duplicate required section '",
          std::string(name), "'");
    }
    result = &entry;
  }
  if (result == nullptr) {
    return makeSerializerError<const MeshBinarySectionTocEntry *>(
        "meshBinaryDeserialize: missing required section '", std::string(name),
        "'");
  }
  return Result<const MeshBinarySectionTocEntry *, std::string>::makeResult(
      result);
}

[[nodiscard]] bool validateSectionBounds(const MeshBinarySectionTocEntry &entry,
                                         size_t fileSize) {
  uint64_t end = 0;
  if (!checkedAddToU64(entry.offset, entry.sizeBytes, end)) {
    return false;
  }
  return end <= static_cast<uint64_t>(fileSize);
}

[[nodiscard]] bool
sectionSizeMatchesCountStride(const MeshBinarySectionTocEntry &entry) {
  uint64_t expectedBytes = 0;
  if (!checkedMulToU64(entry.count, entry.stride, expectedBytes)) {
    return false;
  }
  return expectedBytes == entry.sizeBytes;
}

[[nodiscard]] Result<MeshBinaryRequiredSections, MeshBinaryDeserializeError>
findRequiredSections(std::span<const MeshBinarySectionTocEntry> toc) {
  auto requireSection = [toc](uint32_t fourcc, std::string_view name)
      -> Result<const MeshBinarySectionTocEntry *, MeshBinaryDeserializeError> {
    auto sectionResult = findRequiredSection(toc, fourcc, name);
    if (sectionResult.hasError()) {
      return makeDeserializeError<const MeshBinarySectionTocEntry *>(
          sectionResult.error());
    }
    return Result<const MeshBinarySectionTocEntry *,
                  MeshBinaryDeserializeError>::makeResult(sectionResult
                                                              .value());
  };

  MeshBinaryRequiredSections sections{};
  auto requireAndSet =
      [&](const MeshBinarySectionTocEntry *MeshBinaryRequiredSections::*member,
          uint32_t fourcc, std::string_view name)
      -> Result<std::monostate, MeshBinaryDeserializeError> {
    auto sectionResult = requireSection(fourcc, name);
    if (sectionResult.hasError()) {
      return Result<std::monostate, MeshBinaryDeserializeError>::makeError(
          sectionResult.error());
    }
    sections.*member = sectionResult.value();
    return Result<std::monostate, MeshBinaryDeserializeError>::makeResult({});
  };

  auto vlay = requireAndSet(&MeshBinaryRequiredSections::vlay,
                            kMeshBinarySectionVlay, "VLAY");
  if (vlay.hasError()) {
    return Result<MeshBinaryRequiredSections,
                  MeshBinaryDeserializeError>::makeError(vlay.error());
  }
  auto smes = requireAndSet(&MeshBinaryRequiredSections::smes,
                            kMeshBinarySectionSmes, "SMES");
  if (smes.hasError()) {
    return Result<MeshBinaryRequiredSections,
                  MeshBinaryDeserializeError>::makeError(smes.error());
  }
  auto lods = requireAndSet(&MeshBinaryRequiredSections::lods,
                            kMeshBinarySectionLods, "LODS");
  if (lods.hasError()) {
    return Result<MeshBinaryRequiredSections,
                  MeshBinaryDeserializeError>::makeError(lods.error());
  }
  auto vbuf = requireAndSet(&MeshBinaryRequiredSections::vbuf,
                            kMeshBinarySectionVbuf, "VBUF");
  if (vbuf.hasError()) {
    return Result<MeshBinaryRequiredSections,
                  MeshBinaryDeserializeError>::makeError(vbuf.error());
  }
  auto ibuf = requireAndSet(&MeshBinaryRequiredSections::ibuf,
                            kMeshBinarySectionIbuf, "IBUF");
  if (ibuf.hasError()) {
    return Result<MeshBinaryRequiredSections,
                  MeshBinaryDeserializeError>::makeError(ibuf.error());
  }

  return Result<MeshBinaryRequiredSections,
                MeshBinaryDeserializeError>::makeResult(sections);
}

[[nodiscard]] Result<std::monostate, MeshBinaryDeserializeError>
validateFixedSectionLayout(const MeshBinarySectionTocEntry &entry,
                           uint32_t expectedCount, uint32_t expectedStride,
                           bool requireCountStrideMatch,
                           std::string_view errorMessage) {
  if (entry.count != expectedCount || entry.stride != expectedStride ||
      (requireCountStrideMatch && !sectionSizeMatchesCountStride(entry))) {
    return makeDeserializeError<std::monostate>(std::string(errorMessage));
  }
  return Result<std::monostate, MeshBinaryDeserializeError>::makeResult({});
}

[[nodiscard]] Result<std::monostate, MeshBinaryDeserializeError>
validateVariableSectionLayout(const MeshBinarySectionTocEntry &entry,
                              uint32_t expectedStride,
                              std::string_view errorMessage) {
  if (entry.stride != expectedStride || !sectionSizeMatchesCountStride(entry)) {
    return makeDeserializeError<std::monostate>(std::string(errorMessage));
  }
  return Result<std::monostate, MeshBinaryDeserializeError>::makeResult({});
}

[[nodiscard]] Result<std::pmr::vector<MeshBinarySubmeshRecord>,
                     MeshBinaryDeserializeError>
readSubmeshRecords(std::span<const std::byte> fileBytes,
                   const MeshBinarySectionTocEntry &entry,
                   std::pmr::memory_resource *memory) {
  std::pmr::vector<MeshBinarySubmeshRecord> records(memory);
  if (entry.stride == sizeof(MeshBinarySubmeshRecord)) {
    if (!readPodArray(fileBytes, entry.offset, entry.count, records)) {
      return makeDeserializeError<std::pmr::vector<MeshBinarySubmeshRecord>>(
          "meshBinaryDeserialize: failed to read submesh records");
    }
    return Result<std::pmr::vector<MeshBinarySubmeshRecord>,
                  MeshBinaryDeserializeError>::makeResult(std::move(records));
  }

  if (entry.stride == sizeof(MeshBinarySubmeshRecordV1)) {
    std::pmr::vector<MeshBinarySubmeshRecordV1> legacyRecords(memory);
    if (!readPodArray(fileBytes, entry.offset, entry.count, legacyRecords)) {
      return makeDeserializeError<std::pmr::vector<MeshBinarySubmeshRecord>>(
          "meshBinaryDeserialize: failed to read submesh records");
    }

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
  if (!readPodArray(fileBytes, entry.offset, entry.count, legacyRecords)) {
    return makeDeserializeError<std::pmr::vector<MeshBinarySubmeshRecord>>(
        "meshBinaryDeserialize: failed to read submesh records");
  }

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

[[nodiscard]] Result<MeshBinaryBufferSectionHeader, MeshBinaryDeserializeError>
readBufferSectionHeader(std::span<const std::byte> fileBytes,
                        const MeshBinarySectionTocEntry &entry,
                        std::string_view sectionName) {
  if (entry.count != 1 ||
      entry.stride != sizeof(MeshBinaryBufferSectionHeader)) {
    return makeDeserializeError<MeshBinaryBufferSectionHeader>(
        "meshBinaryDeserialize: invalid ", std::string(sectionName),
        " metadata layout");
  }
  if (entry.sizeBytes < sizeof(MeshBinaryBufferSectionHeader)) {
    return makeDeserializeError<MeshBinaryBufferSectionHeader>(
        "meshBinaryDeserialize: ", std::string(sectionName),
        " section too small");
  }

  MeshBinaryBufferSectionHeader meta{};
  if (!readPod(fileBytes, entry.offset, meta)) {
    return makeDeserializeError<MeshBinaryBufferSectionHeader>(
        "meshBinaryDeserialize: failed to read ", std::string(sectionName),
        " metadata");
  }

  const uint64_t expectedSize =
      sizeof(MeshBinaryBufferSectionHeader) + meta.encodedSizeBytes;
  if (expectedSize != entry.sizeBytes) {
    return makeDeserializeError<MeshBinaryBufferSectionHeader>(
        "meshBinaryDeserialize: ", std::string(sectionName), " size mismatch");
  }

  return Result<MeshBinaryBufferSectionHeader,
                MeshBinaryDeserializeError>::makeResult(meta);
}

} // namespace

Result<std::vector<std::byte>, std::string>
meshBinarySerialize(const MeshBinarySerializeInput &input) {
  ScratchArena scratch;
  ScopedScratch scopedScratch(scratch);

  if (!isLittleEndianHost()) {
    return makeSerializerError<std::vector<std::byte>>(
        "meshBinarySerialize: unsupported host endianness");
  }
  if (input.vertices.strideBytes == 0) {
    return makeSerializerError<std::vector<std::byte>>(
        "meshBinarySerialize: vertex stride is 0");
  }
  if (input.vertices.count == 0) {
    return makeSerializerError<std::vector<std::byte>>(
        "meshBinarySerialize: vertex count is 0");
  }
  if (input.vertices.empty()) {
    return makeSerializerError<std::vector<std::byte>>(
        "meshBinarySerialize: packed vertex bytes are empty");
  }
  if ((input.vertices.size() % input.vertices.strideBytes) != 0u) {
    return makeSerializerError<std::vector<std::byte>>(
        "meshBinarySerialize: packed vertex byte size is not aligned to "
        "vertex stride");
  }
  if (input.indices.empty()) {
    return makeSerializerError<std::vector<std::byte>>(
        "meshBinarySerialize: index buffer is empty");
  }
  for (const Submesh &submesh : input.submeshes) {
    if (submesh.lodCount == 0 || submesh.lodCount > Submesh::kMaxLodCount) {
      return makeSerializerError<std::vector<std::byte>>(
          "meshBinarySerialize: invalid submesh LOD count");
    }
    for (uint32_t lodIndex = 0; lodIndex < submesh.lodCount; ++lodIndex) {
      const SubmeshLod &lod = submesh.lods[lodIndex];
      uint64_t rangeEnd = 0;
      if (!checkedAddToU64(lod.indexOffset, lod.indexCount, rangeEnd) ||
          rangeEnd > input.indices.size()) {
        return makeSerializerError<std::vector<std::byte>>(
            "meshBinarySerialize: submesh index range out of bounds");
      }
    }
  }

  const size_t vertexCountFromBytes =
      input.vertices.size() / input.vertices.strideBytes;
  if (vertexCountFromBytes != input.vertices.count) {
    return makeSerializerError<std::vector<std::byte>>(
        "meshBinarySerialize: vertex count mismatch between metadata and "
        "bytes");
  }
  const uint32_t expectedVertexStride =
      vertexStrideForLayoutId(input.vertexLayoutId);
  if (expectedVertexStride == 0u ||
      input.vertices.strideBytes != expectedVertexStride) {
    return makeSerializerError<std::vector<std::byte>>(
        "meshBinarySerialize: vertex stride does not match layout id");
  }

  const auto validateOptionalLayout =
      [](const BufferLayout<std::span<const std::byte>> &layout,
         std::string_view name) -> Result<bool, std::string> {
    if (!layout.validate()) {
      return Result<bool, std::string>::makeError(
          "meshBinarySerialize: invalid " + std::string(name) + " byte layout");
    }
    return Result<bool, std::string>::makeResult(true);
  };

  auto skinLayoutResult =
      validateOptionalLayout(input.skinInfluences, "skin influence");
  if (skinLayoutResult.hasError()) {
    return makeSerializerError<std::vector<std::byte>>(
        skinLayoutResult.error());
  }
  auto morphMetaLayoutResult =
      validateOptionalLayout(input.morphMeta, "morph meta");
  if (morphMetaLayoutResult.hasError()) {
    return makeSerializerError<std::vector<std::byte>>(
        morphMetaLayoutResult.error());
  }
  auto morphDeltaLayoutResult =
      validateOptionalLayout(input.morphDeltas, "morph delta");
  if (morphDeltaLayoutResult.hasError()) {
    return makeSerializerError<std::vector<std::byte>>(
        morphDeltaLayoutResult.error());
  }
  auto staticDecodeLayoutResult =
      validateOptionalLayout(input.staticVertexDecode, "static vertex decode");
  if (staticDecodeLayoutResult.hasError()) {
    return makeSerializerError<std::vector<std::byte>>(
        staticDecodeLayoutResult.error());
  }
  if (input.vertexLayoutId == kMeshBinaryLayoutIdStaticQuantized20) {
    if (!input.staticVertexDecode.empty() &&
        input.staticVertexDecode.count != input.submeshes.size()) {
      return makeSerializerError<std::vector<std::byte>>(
          "meshBinarySerialize: static vertex decode count must match "
          "submeshes");
    }
  } else if (!input.staticVertexDecode.empty()) {
    return makeSerializerError<std::vector<std::byte>>(
        "meshBinarySerialize: static vertex decode is only valid for static "
        "layout");
  }

  if (input.morphMeta.empty() != input.morphDeltas.empty()) {
    return makeSerializerError<std::vector<std::byte>>(
        "meshBinarySerialize: morph meta and morph delta payload must be "
        "present together");
  }
  if (!input.morphMeta.empty() &&
      (input.morphMeta.count != 1u ||
       input.morphMeta.strideBytes != sizeof(MeshBinaryMorphMetaRecord))) {
    return makeSerializerError<std::vector<std::byte>>(
        "meshBinarySerialize: invalid morph meta payload");
  }

  std::pmr::vector<SerializedSection> sections(scopedScratch.resource());
  sections.reserve(8);

  auto vlayResult = buildVertexLayoutSection(input);
  if (vlayResult.hasError()) {
    return makeSerializerError<std::vector<std::byte>>(vlayResult.error());
  }
  sections.push_back(std::move(vlayResult.value()));

  auto smesLodsResult =
      buildSubmeshAndLodSections(input.submeshes, input.vertexLayoutId);
  if (smesLodsResult.hasError()) {
    return makeSerializerError<std::vector<std::byte>>(smesLodsResult.error());
  }
  auto smesLodsSections = std::move(smesLodsResult.value());
  sections.push_back(std::move(smesLodsSections.first));
  sections.push_back(std::move(smesLodsSections.second));

  auto vbufResult = buildVertexBufferSection(input.vertices);
  if (vbufResult.hasError()) {
    return makeSerializerError<std::vector<std::byte>>(vbufResult.error());
  }
  sections.push_back(std::move(vbufResult.value()));

  auto ibufResult =
      buildIndexBufferSection(input.indices, input.vertices.count);
  if (ibufResult.hasError()) {
    return makeSerializerError<std::vector<std::byte>>(ibufResult.error());
  }
  sections.push_back(std::move(ibufResult.value()));

  if (!input.skinInfluences.empty()) {
    auto vinfResult = buildOptionalVertexBufferSection(
        kMeshBinarySectionVinf, input.skinInfluences, "skin influence");
    if (vinfResult.hasError()) {
      return makeSerializerError<std::vector<std::byte>>(vinfResult.error());
    }
    sections.push_back(std::move(vinfResult.value()));
  }
  if (!input.staticVertexDecode.empty()) {
    auto vdecResult = buildOptionalVertexBufferSection(kMeshBinarySectionVdec,
                                                       input.staticVertexDecode,
                                                       "static vertex decode");
    if (vdecResult.hasError()) {
      return Result<std::vector<std::byte>, std::string>::makeError(
          vdecResult.error());
    }
    sections.push_back(std::move(vdecResult.value()));
  }
  if (!input.morphMeta.empty()) {
    auto mmtaResult = buildOptionalVertexBufferSection(
        kMeshBinarySectionMmta, input.morphMeta, "morph meta");
    if (mmtaResult.hasError()) {
      return makeSerializerError<std::vector<std::byte>>(mmtaResult.error());
    }
    sections.push_back(std::move(mmtaResult.value()));
  }
  if (!input.morphDeltas.empty()) {
    auto mdelResult = buildOptionalVertexBufferSection(
        kMeshBinarySectionMdel, input.morphDeltas, "morph delta");
    if (mdelResult.hasError()) {
      return makeSerializerError<std::vector<std::byte>>(mdelResult.error());
    }
    sections.push_back(std::move(mdelResult.value()));
  }

  const uint64_t tocCount = static_cast<uint64_t>(sections.size());
  uint64_t tocBytes = 0;
  if (!checkedMulToU64(tocCount, sizeof(MeshBinarySectionTocEntry), tocBytes)) {
    return makeSerializerError<std::vector<std::byte>>(
        "meshBinarySerialize: TOC byte size overflow");
  }

  uint64_t sectionStart = sizeof(MeshBinaryHeader);
  if (!checkedAddToU64(sectionStart, tocBytes, sectionStart)) {
    return makeSerializerError<std::vector<std::byte>>(
        "meshBinarySerialize: section start overflow");
  }
  if (!alignUpU64(sectionStart, kMeshBinarySectionAlignment, sectionStart)) {
    return makeSerializerError<std::vector<std::byte>>(
        "meshBinarySerialize: section alignment overflow");
  }

  std::pmr::vector<MeshBinarySectionTocEntry> tocEntries(
      scopedScratch.resource());
  tocEntries.resize(sections.size());

  uint64_t cursor = sectionStart;
  for (size_t i = 0; i < sections.size(); ++i) {
    if (!alignUpU64(cursor, kMeshBinarySectionAlignment, cursor)) {
      return makeSerializerError<std::vector<std::byte>>(
          "meshBinarySerialize: section cursor alignment overflow");
    }
    const SerializedSection &section = sections[i];
    MeshBinarySectionTocEntry &entry = tocEntries[i];
    entry.fourcc = section.fourcc;
    entry.flags = section.flags;
    entry.offset = cursor;
    entry.sizeBytes = static_cast<uint64_t>(section.payload.size());
    entry.count = section.count;
    entry.stride = section.stride;
    if (!checkedAddToU64(cursor, entry.sizeBytes, cursor)) {
      return makeSerializerError<std::vector<std::byte>>(
          "meshBinarySerialize: section cursor overflow");
    }
  }

  uint64_t fileSize = 0;
  if (!alignUpU64(cursor, kMeshBinarySectionAlignment, fileSize)) {
    return makeSerializerError<std::vector<std::byte>>(
        "meshBinarySerialize: final file alignment overflow");
  }
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
      kMeshBinaryHeaderFlagLittleEndian | kMeshBinaryHeaderFlagCompressed;
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

  if (!isLittleEndianHost()) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: unsupported host endianness");
  }
  if (fileBytes.size() < sizeof(MeshBinaryHeader)) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: file too small");
  }

  MeshBinaryHeader header{};
  std::memcpy(&header, fileBytes.data(), sizeof(header));

  if (header.magic != kMeshBinaryMagic) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: magic mismatch");
  }
  if (header.majorVersion != kMeshBinaryFormatMajorVersion) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: unsupported format major version");
  }
  if (header.minorVersion > kMeshBinaryFormatMinorVersion) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: unsupported format minor version");
  }
  if ((header.flags & kMeshBinaryHeaderFlagLittleEndian) == 0u) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: unsupported endian flag");
  }
  if (header.headerSize != sizeof(MeshBinaryHeader) ||
      header.tocEntrySize != sizeof(MeshBinarySectionTocEntry)) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: header or TOC entry size mismatch");
  }
  if (header.fileSize != fileBytes.size()) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: file size mismatch");
  }
  if (header.sourcePathHash != context.expectedSourcePathHash ||
      header.importOptionsHash != context.expectedImportOptionsHash) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: source or options hash mismatch");
  }
  if (context.validateSourceFingerprint && context.sourceExists) {
    if (header.sourceSizeBytes != context.sourceSizeBytes ||
        header.sourceMtimeNs != context.sourceMtimeNs) {
      return makeDeserializeError<MeshBinaryDecodedMesh>(
          MeshBinaryDeserializeErrorCode::StaleCache,
          "meshBinaryDeserialize: cache is stale for current source file");
    }
  }

  uint64_t tocBytes = 0;
  if (!checkedMulToU64(header.tocCount, sizeof(MeshBinarySectionTocEntry),
                       tocBytes)) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: TOC byte size overflow");
  }
  uint64_t tocEnd = 0;
  if (!checkedAddToU64(header.tocOffset, tocBytes, tocEnd) ||
      tocEnd > fileBytes.size()) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: invalid TOC bounds");
  }

  std::pmr::vector<MeshBinarySectionTocEntry> toc(scopedScratch.resource());
  if (!readPodArray(fileBytes, header.tocOffset, header.tocCount, toc)) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: failed to read TOC");
  }
  for (const MeshBinarySectionTocEntry &entry : toc) {
    if (!validateSectionBounds(entry, fileBytes.size())) {
      return makeDeserializeError<MeshBinaryDecodedMesh>(
          "meshBinaryDeserialize: section exceeds file bounds");
    }
  }

  auto requiredSectionsResult = findRequiredSections(toc);
  if (requiredSectionsResult.hasError()) {
    return Result<MeshBinaryDecodedMesh, MeshBinaryDeserializeError>::makeError(
        requiredSectionsResult.error());
  }
  const MeshBinaryRequiredSections &requiredSections =
      requiredSectionsResult.value();
  const MeshBinarySectionTocEntry &vlayEntry = *requiredSections.vlay;
  const MeshBinarySectionTocEntry &smesEntry = *requiredSections.smes;
  const MeshBinarySectionTocEntry &lodsEntry = *requiredSections.lods;
  const MeshBinarySectionTocEntry &vbufEntry = *requiredSections.vbuf;
  const MeshBinarySectionTocEntry &ibufEntry = *requiredSections.ibuf;

  auto vlayLayoutResult = validateFixedSectionLayout(
      vlayEntry, 1u, sizeof(MeshBinaryVertexLayoutRecord), true,
      "meshBinaryDeserialize: invalid VLAY layout");
  if (vlayLayoutResult.hasError()) {
    return Result<MeshBinaryDecodedMesh, MeshBinaryDeserializeError>::makeError(
        vlayLayoutResult.error());
  }
  if (!sectionSizeMatchesCountStride(smesEntry)) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: invalid SMES layout");
  }
  if (smesEntry.stride != sizeof(MeshBinarySubmeshRecord) &&
      smesEntry.stride != sizeof(MeshBinarySubmeshRecordV1) &&
      smesEntry.stride != sizeof(MeshBinarySubmeshRecordV0)) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: unsupported SMES stride");
  }
  auto lodsLayoutResult = validateVariableSectionLayout(
      lodsEntry, sizeof(MeshBinaryLodRecord),
      "meshBinaryDeserialize: invalid LODS stride");
  if (lodsLayoutResult.hasError()) {
    return Result<MeshBinaryDecodedMesh, MeshBinaryDeserializeError>::makeError(
        lodsLayoutResult.error());
  }

  MeshBinaryVertexLayoutRecord layoutRecord{};
  if (!readPod(fileBytes, vlayEntry.offset, layoutRecord)) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: failed to read VLAY record");
  }
  const uint32_t expectedVertexStride =
      vertexStrideForLayoutId(layoutRecord.layoutId);
  if (expectedVertexStride == 0u ||
      layoutRecord.strideBytes != expectedVertexStride) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: unsupported vertex layout");
  }

  auto vbufMetaResult = readBufferSectionHeader(fileBytes, vbufEntry, "VBUF");
  if (vbufMetaResult.hasError()) {
    return Result<MeshBinaryDecodedMesh, MeshBinaryDeserializeError>::makeError(
        vbufMetaResult.error());
  }
  auto ibufMetaResult = readBufferSectionHeader(fileBytes, ibufEntry, "IBUF");
  if (ibufMetaResult.hasError()) {
    return Result<MeshBinaryDecodedMesh, MeshBinaryDeserializeError>::makeError(
        ibufMetaResult.error());
  }
  const MeshBinaryBufferSectionHeader vbufMeta = vbufMetaResult.value();
  const MeshBinaryBufferSectionHeader ibufMeta = ibufMetaResult.value();
  const auto findOptionalSection = [&toc](uint32_t fourcc,
                                          std::string_view name)
      -> Result<const MeshBinarySectionTocEntry *, MeshBinaryDeserializeError> {
    const MeshBinarySectionTocEntry *result = nullptr;
    for (const MeshBinarySectionTocEntry &entry : toc) {
      if (entry.fourcc != fourcc) {
        continue;
      }
      if (result != nullptr) {
        return makeDeserializeError<const MeshBinarySectionTocEntry *>(
            "meshBinaryDeserialize: duplicate optional section '",
            std::string(name), "'");
      }
      result = &entry;
    }
    return Result<const MeshBinarySectionTocEntry *,
                  MeshBinaryDeserializeError>::makeResult(result);
  };
  auto vinfEntryResult = findOptionalSection(kMeshBinarySectionVinf, "VINF");
  if (vinfEntryResult.hasError()) {
    return Result<MeshBinaryDecodedMesh, MeshBinaryDeserializeError>::makeError(
        vinfEntryResult.error());
  }
  auto vdecEntryResult = findOptionalSection(kMeshBinarySectionVdec, "VDEC");
  if (vdecEntryResult.hasError()) {
    return Result<MeshBinaryDecodedMesh, MeshBinaryDeserializeError>::makeError(
        vdecEntryResult.error());
  }
  auto mmtaEntryResult = findOptionalSection(kMeshBinarySectionMmta, "MMTA");
  if (mmtaEntryResult.hasError()) {
    return Result<MeshBinaryDecodedMesh, MeshBinaryDeserializeError>::makeError(
        mmtaEntryResult.error());
  }
  auto mdelEntryResult = findOptionalSection(kMeshBinarySectionMdel, "MDEL");
  if (mdelEntryResult.hasError()) {
    return Result<MeshBinaryDecodedMesh, MeshBinaryDeserializeError>::makeError(
        mdelEntryResult.error());
  }
  const MeshBinarySectionTocEntry *vinfEntry = vinfEntryResult.value();
  const MeshBinarySectionTocEntry *vdecEntry = vdecEntryResult.value();
  const MeshBinarySectionTocEntry *mmtaEntry = mmtaEntryResult.value();
  const MeshBinarySectionTocEntry *mdelEntry = mdelEntryResult.value();
  if ((mmtaEntry == nullptr) != (mdelEntry == nullptr)) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: morph meta and morph delta sections must be "
        "present together");
  }

  const std::span<const std::byte> encodedVertices(
      fileBytes.data() + static_cast<size_t>(vbufEntry.offset) +
          sizeof(MeshBinaryBufferSectionHeader),
      vbufMeta.encodedSizeBytes);
  const std::span<const std::byte> encodedIndices(
      fileBytes.data() + static_cast<size_t>(ibufEntry.offset) +
          sizeof(MeshBinaryBufferSectionHeader),
      ibufMeta.encodedSizeBytes);

  auto decodedVerticesResult = meshBinaryDecodeVertexBuffer(
      encodedVertices, vbufMeta.elementCount, vbufMeta.elementStrideBytes);
  if (decodedVerticesResult.hasError()) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        decodedVerticesResult.error());
  }
  auto decodedIndicesResult = meshBinaryDecodeIndexBuffer(
      encodedIndices, ibufMeta.elementCount, ibufMeta.elementStrideBytes);
  if (decodedIndicesResult.hasError()) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        decodedIndicesResult.error());
  }
  std::optional<MeshBinaryBufferSectionHeader> vinfMeta;
  std::optional<MeshBinaryBufferSectionHeader> vdecMeta;
  std::optional<MeshBinaryBufferSectionHeader> mmtaMeta;
  std::optional<MeshBinaryBufferSectionHeader> mdelMeta;
  std::vector<std::byte> decodedSkinInfluences;
  std::vector<std::byte> decodedStaticVertexDecode;
  std::vector<std::byte> decodedMorphMeta;
  std::vector<std::byte> decodedMorphDeltas;
  auto decodeOptionalBufferSection =
      [&fileBytes](const MeshBinarySectionTocEntry *entry,
                   std::string_view name)
      -> Result<
          std::pair<MeshBinaryBufferSectionHeader, std::vector<std::byte>>,
          MeshBinaryDeserializeError> {
    NURI_ASSERT(entry != nullptr,
                "meshBinaryDeserialize: optional section entry must be valid");
    auto metaResult = readBufferSectionHeader(fileBytes, *entry, name);
    if (metaResult.hasError()) {
      return Result<
          std::pair<MeshBinaryBufferSectionHeader, std::vector<std::byte>>,
          MeshBinaryDeserializeError>::makeError(metaResult.error());
    }
    const MeshBinaryBufferSectionHeader meta = metaResult.value();
    const std::span<const std::byte> encoded(
        fileBytes.data() + static_cast<size_t>(entry->offset) +
            sizeof(MeshBinaryBufferSectionHeader),
        meta.encodedSizeBytes);
    auto decodedResult = meshBinaryDecodeVertexBuffer(
        encoded, meta.elementCount, meta.elementStrideBytes);
    if (decodedResult.hasError()) {
      return makeDeserializeError<
          std::pair<MeshBinaryBufferSectionHeader, std::vector<std::byte>>>(
          decodedResult.error());
    }
    return Result<
        std::pair<MeshBinaryBufferSectionHeader, std::vector<std::byte>>,
        MeshBinaryDeserializeError>::
        makeResult(std::make_pair(meta, std::move(decodedResult.value())));
  };
  if (vinfEntry != nullptr) {
    auto result = decodeOptionalBufferSection(vinfEntry, "VINF");
    if (result.hasError()) {
      return Result<MeshBinaryDecodedMesh,
                    MeshBinaryDeserializeError>::makeError(result.error());
    }
    vinfMeta = result.value().first;
    decodedSkinInfluences = std::move(result.value().second);
  }
  if (vdecEntry != nullptr) {
    auto result = decodeOptionalBufferSection(vdecEntry, "VDEC");
    if (result.hasError()) {
      return Result<MeshBinaryDecodedMesh,
                    MeshBinaryDeserializeError>::makeError(result.error());
    }
    vdecMeta = result.value().first;
    decodedStaticVertexDecode = std::move(result.value().second);
  }
  if (mmtaEntry != nullptr) {
    auto result = decodeOptionalBufferSection(mmtaEntry, "MMTA");
    if (result.hasError()) {
      return Result<MeshBinaryDecodedMesh,
                    MeshBinaryDeserializeError>::makeError(result.error());
    }
    mmtaMeta = result.value().first;
    decodedMorphMeta = std::move(result.value().second);
  }
  if (mdelEntry != nullptr) {
    auto result = decodeOptionalBufferSection(mdelEntry, "MDEL");
    if (result.hasError()) {
      return Result<MeshBinaryDecodedMesh,
                    MeshBinaryDeserializeError>::makeError(result.error());
    }
    mdelMeta = result.value().first;
    decodedMorphDeltas = std::move(result.value().second);
  }
  if (vinfMeta.has_value() && vinfMeta->elementCount != vbufMeta.elementCount) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: skin influence count must match vertex count");
  }
  if (layoutRecord.layoutId == kMeshBinaryLayoutIdStaticQuantized20) {
    if (!vdecMeta.has_value()) {
      return makeDeserializeError<MeshBinaryDecodedMesh>(
          "meshBinaryDeserialize: static layout requires VDEC section");
    }
  } else if (vdecMeta.has_value()) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: VDEC section is only valid for static layout");
  }
  if (mmtaMeta.has_value()) {
    if (mmtaMeta->elementCount != 1u ||
        mmtaMeta->elementStrideBytes != sizeof(MeshBinaryMorphMetaRecord) ||
        decodedMorphMeta.size() != sizeof(MeshBinaryMorphMetaRecord)) {
      return makeDeserializeError<MeshBinaryDecodedMesh>(
          "meshBinaryDeserialize: invalid morph meta payload");
    }
    MeshBinaryMorphMetaRecord morphMetaRecord{};
    std::memcpy(&morphMetaRecord, decodedMorphMeta.data(),
                sizeof(morphMetaRecord));
    if (morphMetaRecord.vertexCount != vbufMeta.elementCount) {
      return makeDeserializeError<MeshBinaryDecodedMesh>(
          "meshBinaryDeserialize: morph meta vertex count mismatch");
    }
    uint64_t expectedMorphDeltaCount = 0u;
    if (!checkedMulToU64(morphMetaRecord.morphTargetCount,
                         morphMetaRecord.vertexCount,
                         expectedMorphDeltaCount)) {
      return makeDeserializeError<MeshBinaryDecodedMesh>(
          "meshBinaryDeserialize: morph delta count overflow");
    }
    if (expectedMorphDeltaCount != mdelMeta->elementCount) {
      return makeDeserializeError<MeshBinaryDecodedMesh>(
          "meshBinaryDeserialize: morph delta count mismatch");
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
  std::pmr::vector<MeshBinaryLodRecord> lodRecords(scopedScratch.resource());
  if (!readPodArray(fileBytes, lodsEntry.offset, lodsEntry.count, lodRecords)) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: failed to read LOD records");
  }

  MeshBinaryDecodedMesh decoded{};
  decoded.vertexLayoutId = layoutRecord.layoutId;
  decoded.vertices.data = std::move(decodedVerticesResult.value());
  decoded.vertices.count = vbufMeta.elementCount;
  decoded.vertices.strideBytes = vbufMeta.elementStrideBytes;
  decoded.bounds =
      BoundingBox(glm::vec3(header.modelBoundsMin[0], header.modelBoundsMin[1],
                            header.modelBoundsMin[2]),
                  glm::vec3(header.modelBoundsMax[0], header.modelBoundsMax[1],
                            header.modelBoundsMax[2]));
  decoded.skinInfluences.data = std::move(decodedSkinInfluences);
  if (vinfMeta.has_value()) {
    decoded.skinInfluences.count = vinfMeta->elementCount;
    decoded.skinInfluences.strideBytes = vinfMeta->elementStrideBytes;
  }
  decoded.staticVertexDecode.data = std::move(decodedStaticVertexDecode);
  if (vdecMeta.has_value()) {
    decoded.staticVertexDecode.count = vdecMeta->elementCount;
    decoded.staticVertexDecode.strideBytes = vdecMeta->elementStrideBytes;
  }
  decoded.morphMeta.data = std::move(decodedMorphMeta);
  if (mmtaMeta.has_value()) {
    decoded.morphMeta.count = mmtaMeta->elementCount;
    decoded.morphMeta.strideBytes = mmtaMeta->elementStrideBytes;
  }
  decoded.morphDeltas.data = std::move(decodedMorphDeltas);
  if (mdelMeta.has_value()) {
    decoded.morphDeltas.count = mdelMeta->elementCount;
    decoded.morphDeltas.strideBytes = mdelMeta->elementStrideBytes;
  }

  const std::vector<std::byte> &decodedIndexBytes =
      decodedIndicesResult.value();
  if ((decodedIndexBytes.size() % sizeof(uint32_t)) != 0u) {
    return makeDeserializeError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: decoded index bytes are not uint32 aligned");
  }
  decoded.indices.resize(decodedIndexBytes.size() / sizeof(uint32_t));
  if (!decoded.indices.empty()) {
    std::memcpy(decoded.indices.data(), decodedIndexBytes.data(),
                decodedIndexBytes.size());
  }

  decoded.submeshes.reserve(submeshRecords.size());
  for (const MeshBinarySubmeshRecord &record : submeshRecords) {
    if (record.lodCount == 0 || record.lodCount > Submesh::kMaxLodCount) {
      return makeDeserializeError<MeshBinaryDecodedMesh>(
          "meshBinaryDeserialize: invalid submesh LOD count");
    }
    uint64_t lodEnd = 0;
    if (!checkedAddToU64(record.lodFirst, record.lodCount, lodEnd) ||
        lodEnd > lodRecords.size()) {
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
      uint64_t indexRangeEnd = 0;
      if (!checkedAddToU64(lodRecord.indexOffset, lodRecord.indexCount,
                           indexRangeEnd) ||
          indexRangeEnd > decoded.indices.size()) {
        return makeDeserializeError<MeshBinaryDecodedMesh>(
            "meshBinaryDeserialize: submesh index range out of bounds");
      }
      submesh.lods[lodIndex] = SubmeshLod{
          .indexOffset = lodRecord.indexOffset,
          .indexCount = lodRecord.indexCount,
          .error = lodRecord.error,
      };
      if (lodIndex == 0u) {
        submesh.indexOffset = lodRecord.indexOffset;
        submesh.indexCount = lodRecord.indexCount;
      }
    }

    decoded.submeshes.push_back(submesh);
  }

  return Result<MeshBinaryDecodedMesh, MeshBinaryDeserializeError>::makeResult(
      std::move(decoded));
}

} // namespace nuri
