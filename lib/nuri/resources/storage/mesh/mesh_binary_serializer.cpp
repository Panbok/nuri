#include "nuri/pch.h"

#include "nuri/resources/storage/mesh/mesh_binary_serializer.h"

#include "nuri/core/pmr_scratch.h"
#include "nuri/resources/storage/mesh/mesh_binary_codec.h"
#include "nuri/resources/storage/mesh/mesh_binary_format.h"

namespace nuri {
namespace {

template <typename T, typename... Args>
[[nodiscard]] Result<T, std::string> makeSerializerError(Args &&...args) {
  std::ostringstream oss;
  (oss << ... << std::forward<Args>(args));
  return Result<T, std::string>::makeError(oss.str());
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

[[nodiscard]] uint64_t alignUpU64(uint64_t value, uint64_t alignment) {
  if (alignment <= 1u) {
    return value;
  }
  const uint64_t mask = alignment - 1u;
  return (value + mask) & ~mask;
}

template <typename T>
void appendPod(std::vector<std::byte> &out, const T &value) {
  static_assert(std::is_trivially_copyable_v<T>);
  const size_t offset = out.size();
  out.resize(offset + sizeof(T));
  std::memcpy(out.data() + offset, &value, sizeof(T));
}

template <typename T>
void appendPodArray(std::vector<std::byte> &out, std::span<const T> values) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (values.empty()) {
    return;
  }
  const size_t bytes = values.size() * sizeof(T);
  const size_t offset = out.size();
  out.resize(offset + bytes);
  std::memcpy(out.data() + offset, values.data(), bytes);
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
[[nodiscard]] bool
readPodArray(std::span<const std::byte> bytes, uint64_t offset, uint32_t count,
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
  constexpr uint16_t value = 0x1;
  return *reinterpret_cast<const uint8_t *>(&value) == 0x1;
}

struct SerializedSection {
  uint32_t fourcc = 0;
  uint32_t flags = 0;
  uint32_t count = 0;
  uint32_t stride = 0;
  std::vector<std::byte> payload;
};

[[nodiscard]] Result<SerializedSection, std::string>
buildVertexLayoutSection() {
  SerializedSection section{};
  section.fourcc = kMeshBinarySectionVlay;
  section.flags = 0;
  section.count = 1;
  section.stride = sizeof(MeshBinaryVertexLayoutRecord);
  MeshBinaryVertexLayoutRecord record{};
  appendPod(section.payload, record);
  return Result<SerializedSection, std::string>::makeResult(std::move(section));
}

[[nodiscard]] Result<std::pair<SerializedSection, SerializedSection>,
                     std::string>
buildSubmeshAndLodSections(std::span<const Submesh> submeshes) {
  SerializedSection submeshSection{};
  submeshSection.fourcc = kMeshBinarySectionSmes;
  submeshSection.flags = 0;
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
    submeshRecord.materialIndex = submesh.materialIndex;
    submeshRecord.lodFirst = lodCount;
    submeshRecord.lodCount = submesh.lodCount;
    submeshRecord.layoutId = kMeshBinaryLayoutIdPacked32;
    submeshRecord.boundsMin[0] = submesh.bounds.min_.x;
    submeshRecord.boundsMin[1] = submesh.bounds.min_.y;
    submeshRecord.boundsMin[2] = submesh.bounds.min_.z;
    submeshRecord.boundsMax[0] = submesh.bounds.max_.x;
    submeshRecord.boundsMax[1] = submesh.bounds.max_.y;
    submeshRecord.boundsMax[2] = submesh.bounds.max_.z;
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
buildVertexBufferSection(std::span<const std::byte> packedVertexBytes,
                         uint32_t vertexCount, uint32_t vertexStrideBytes) {
  auto encodedResult =
      meshBinaryEncodeVertexBuffer(packedVertexBytes, vertexStrideBytes);
  if (encodedResult.hasError()) {
    return makeSerializerError<SerializedSection>(encodedResult.error());
  }

  const std::vector<std::byte> &encoded = encodedResult.value();
  if (encoded.size() > std::numeric_limits<uint32_t>::max()) {
    return makeSerializerError<SerializedSection>(
        "meshBinarySerialize: encoded vertex payload is too large");
  }

  SerializedSection section{};
  section.fourcc = kMeshBinarySectionVbuf;
  section.flags = kMeshBinarySectionFlagCompressed;
  section.count = 1;
  section.stride = sizeof(MeshBinaryBufferSectionHeader);

  MeshBinaryBufferSectionHeader meta{};
  meta.elementCount = vertexCount;
  meta.elementStrideBytes = vertexStrideBytes;
  meta.encodedSizeBytes = static_cast<uint32_t>(encoded.size());
  appendPod(section.payload, meta);
  appendPodArray(section.payload, std::span<const std::byte>(encoded));

  return Result<SerializedSection, std::string>::makeResult(std::move(section));
}

[[nodiscard]] Result<SerializedSection, std::string>
buildIndexBufferSection(std::span<const uint32_t> indices,
                        uint32_t vertexCount) {
  auto encodedResult = meshBinaryEncodeIndexBuffer(indices, vertexCount);
  if (encodedResult.hasError()) {
    return makeSerializerError<SerializedSection>(encodedResult.error());
  }

  const std::vector<std::byte> &encoded = encodedResult.value();
  if (encoded.size() > std::numeric_limits<uint32_t>::max()) {
    return makeSerializerError<SerializedSection>(
        "meshBinarySerialize: encoded index payload is too large");
  }

  SerializedSection section{};
  section.fourcc = kMeshBinarySectionIbuf;
  section.flags = kMeshBinarySectionFlagCompressed;
  section.count = 1;
  section.stride = sizeof(MeshBinaryBufferSectionHeader);

  MeshBinaryBufferSectionHeader meta{};
  meta.elementCount = static_cast<uint32_t>(indices.size());
  meta.elementStrideBytes = sizeof(uint32_t);
  meta.encodedSizeBytes = static_cast<uint32_t>(encoded.size());
  appendPod(section.payload, meta);
  appendPodArray(section.payload, std::span<const std::byte>(encoded));

  return Result<SerializedSection, std::string>::makeResult(std::move(section));
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

} // namespace

Result<std::vector<std::byte>, std::string>
meshBinarySerialize(const MeshBinarySerializeInput &input) {
  ScratchArena scratch;
  ScopedScratch scopedScratch(scratch);

  if (!isLittleEndianHost()) {
    return makeSerializerError<std::vector<std::byte>>(
        "meshBinarySerialize: unsupported host endianness");
  }
  if (input.vertexStrideBytes == 0) {
    return makeSerializerError<std::vector<std::byte>>(
        "meshBinarySerialize: vertex stride is 0");
  }
  if (input.vertexCount == 0) {
    return makeSerializerError<std::vector<std::byte>>(
        "meshBinarySerialize: vertex count is 0");
  }
  if (input.packedVertexBytes.empty()) {
    return makeSerializerError<std::vector<std::byte>>(
        "meshBinarySerialize: packed vertex bytes are empty");
  }
  if ((input.packedVertexBytes.size() % input.vertexStrideBytes) != 0u) {
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
      input.packedVertexBytes.size() / input.vertexStrideBytes;
  if (vertexCountFromBytes != input.vertexCount) {
    return makeSerializerError<std::vector<std::byte>>(
        "meshBinarySerialize: vertex count mismatch between metadata and "
        "bytes");
  }

  std::pmr::vector<SerializedSection> sections(scopedScratch.resource());
  sections.reserve(5);

  auto vlayResult = buildVertexLayoutSection();
  if (vlayResult.hasError()) {
    return makeSerializerError<std::vector<std::byte>>(vlayResult.error());
  }
  sections.push_back(std::move(vlayResult.value()));

  auto smesLodsResult = buildSubmeshAndLodSections(input.submeshes);
  if (smesLodsResult.hasError()) {
    return makeSerializerError<std::vector<std::byte>>(smesLodsResult.error());
  }
  auto smesLodsSections = std::move(smesLodsResult.value());
  sections.push_back(std::move(smesLodsSections.first));
  sections.push_back(std::move(smesLodsSections.second));

  auto vbufResult = buildVertexBufferSection(
      input.packedVertexBytes, input.vertexCount, input.vertexStrideBytes);
  if (vbufResult.hasError()) {
    return makeSerializerError<std::vector<std::byte>>(vbufResult.error());
  }
  sections.push_back(std::move(vbufResult.value()));

  auto ibufResult = buildIndexBufferSection(input.indices, input.vertexCount);
  if (ibufResult.hasError()) {
    return makeSerializerError<std::vector<std::byte>>(ibufResult.error());
  }
  sections.push_back(std::move(ibufResult.value()));

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
  sectionStart = alignUpU64(sectionStart, 16u);

  std::pmr::vector<MeshBinarySectionTocEntry> tocEntries(
      scopedScratch.resource());
  tocEntries.resize(sections.size());

  uint64_t cursor = sectionStart;
  for (size_t i = 0; i < sections.size(); ++i) {
    cursor = alignUpU64(cursor, 16u);
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

  const uint64_t fileSize = alignUpU64(cursor, 16u);
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

Result<MeshBinaryDecodedMesh, std::string>
meshBinaryDeserialize(std::span<const std::byte> fileBytes,
                      const MeshBinaryDeserializeContext &context) {
  ScratchArena scratch;
  ScopedScratch scopedScratch(scratch);

  if (!isLittleEndianHost()) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: unsupported host endianness");
  }
  if (fileBytes.size() < sizeof(MeshBinaryHeader)) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: file too small");
  }

  MeshBinaryHeader header{};
  std::memcpy(&header, fileBytes.data(), sizeof(header));

  if (header.magic != kMeshBinaryMagic) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: magic mismatch");
  }
  if (header.majorVersion != kMeshBinaryFormatMajorVersion) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: unsupported format major version");
  }
  if ((header.flags & kMeshBinaryHeaderFlagLittleEndian) == 0u) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: unsupported endian flag");
  }
  if (header.headerSize != sizeof(MeshBinaryHeader) ||
      header.tocEntrySize != sizeof(MeshBinarySectionTocEntry)) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: header or TOC entry size mismatch");
  }
  if (header.fileSize != fileBytes.size()) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: file size mismatch");
  }
  if (header.sourcePathHash != context.expectedSourcePathHash ||
      header.importOptionsHash != context.expectedImportOptionsHash) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: source or options hash mismatch");
  }
  if (context.validateSourceFingerprint && context.sourceExists) {
    if (header.sourceSizeBytes != context.sourceSizeBytes ||
        header.sourceMtimeNs != context.sourceMtimeNs) {
      return makeSerializerError<MeshBinaryDecodedMesh>(
          "meshBinaryDeserialize: cache is stale for current source file");
    }
  }

  uint64_t tocBytes = 0;
  if (!checkedMulToU64(header.tocCount, sizeof(MeshBinarySectionTocEntry),
                       tocBytes)) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: TOC byte size overflow");
  }
  uint64_t tocEnd = 0;
  if (!checkedAddToU64(header.tocOffset, tocBytes, tocEnd) ||
      tocEnd > fileBytes.size()) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: invalid TOC bounds");
  }

  std::pmr::vector<MeshBinarySectionTocEntry> toc(scopedScratch.resource());
  if (!readPodArray(fileBytes, header.tocOffset, header.tocCount, toc)) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: failed to read TOC");
  }
  for (const MeshBinarySectionTocEntry &entry : toc) {
    if (!validateSectionBounds(entry, fileBytes.size())) {
      return makeSerializerError<MeshBinaryDecodedMesh>(
          "meshBinaryDeserialize: section exceeds file bounds");
    }
  }

  auto vlaySectionResult =
      findRequiredSection(toc, kMeshBinarySectionVlay, "VLAY");
  if (vlaySectionResult.hasError()) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
        vlaySectionResult.error());
  }
  auto smesSectionResult =
      findRequiredSection(toc, kMeshBinarySectionSmes, "SMES");
  if (smesSectionResult.hasError()) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
        smesSectionResult.error());
  }
  auto lodsSectionResult =
      findRequiredSection(toc, kMeshBinarySectionLods, "LODS");
  if (lodsSectionResult.hasError()) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
        lodsSectionResult.error());
  }
  auto vbufSectionResult =
      findRequiredSection(toc, kMeshBinarySectionVbuf, "VBUF");
  if (vbufSectionResult.hasError()) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
        vbufSectionResult.error());
  }
  auto ibufSectionResult =
      findRequiredSection(toc, kMeshBinarySectionIbuf, "IBUF");
  if (ibufSectionResult.hasError()) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
        ibufSectionResult.error());
  }

  const MeshBinarySectionTocEntry &vlayEntry = *vlaySectionResult.value();
  const MeshBinarySectionTocEntry &smesEntry = *smesSectionResult.value();
  const MeshBinarySectionTocEntry &lodsEntry = *lodsSectionResult.value();
  const MeshBinarySectionTocEntry &vbufEntry = *vbufSectionResult.value();
  const MeshBinarySectionTocEntry &ibufEntry = *ibufSectionResult.value();

  if (vlayEntry.stride != sizeof(MeshBinaryVertexLayoutRecord) ||
      vlayEntry.count != 1 || !sectionSizeMatchesCountStride(vlayEntry)) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: invalid VLAY layout");
  }
  if (smesEntry.stride != sizeof(MeshBinarySubmeshRecord) ||
      !sectionSizeMatchesCountStride(smesEntry)) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: invalid SMES stride");
  }
  if (lodsEntry.stride != sizeof(MeshBinaryLodRecord) ||
      !sectionSizeMatchesCountStride(lodsEntry)) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: invalid LODS stride");
  }
  if (vbufEntry.count != 1 ||
      vbufEntry.stride != sizeof(MeshBinaryBufferSectionHeader)) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: invalid VBUF metadata layout");
  }
  if (ibufEntry.count != 1 ||
      ibufEntry.stride != sizeof(MeshBinaryBufferSectionHeader)) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: invalid IBUF metadata layout");
  }

  MeshBinaryVertexLayoutRecord layoutRecord{};
  if (!readPod(fileBytes, vlayEntry.offset, layoutRecord)) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: failed to read VLAY record");
  }
  if (layoutRecord.layoutId != kMeshBinaryLayoutIdPacked32 ||
      layoutRecord.strideBytes != kMeshBinaryPackedVertexStrideBytes) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: unsupported vertex layout");
  }

  if (vbufEntry.sizeBytes < sizeof(MeshBinaryBufferSectionHeader)) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: VBUF section too small");
  }
  if (ibufEntry.sizeBytes < sizeof(MeshBinaryBufferSectionHeader)) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: IBUF section too small");
  }

  MeshBinaryBufferSectionHeader vbufMeta{};
  if (!readPod(fileBytes, vbufEntry.offset, vbufMeta)) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: failed to read VBUF metadata");
  }
  MeshBinaryBufferSectionHeader ibufMeta{};
  if (!readPod(fileBytes, ibufEntry.offset, ibufMeta)) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: failed to read IBUF metadata");
  }

  const uint64_t vbufExpectedSize =
      sizeof(MeshBinaryBufferSectionHeader) + vbufMeta.encodedSizeBytes;
  if (vbufExpectedSize != vbufEntry.sizeBytes) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: VBUF size mismatch");
  }
  const uint64_t ibufExpectedSize =
      sizeof(MeshBinaryBufferSectionHeader) + ibufMeta.encodedSizeBytes;
  if (ibufExpectedSize != ibufEntry.sizeBytes) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: IBUF size mismatch");
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
    return makeSerializerError<MeshBinaryDecodedMesh>(
        decodedVerticesResult.error());
  }
  auto decodedIndicesResult = meshBinaryDecodeIndexBuffer(
      encodedIndices, ibufMeta.elementCount, ibufMeta.elementStrideBytes);
  if (decodedIndicesResult.hasError()) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
        decodedIndicesResult.error());
  }

  std::pmr::vector<MeshBinarySubmeshRecord> submeshRecords(
      scopedScratch.resource());
  if (!readPodArray(fileBytes, smesEntry.offset, smesEntry.count,
                    submeshRecords)) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: failed to read submesh records");
  }
  std::pmr::vector<MeshBinaryLodRecord> lodRecords(scopedScratch.resource());
  if (!readPodArray(fileBytes, lodsEntry.offset, lodsEntry.count, lodRecords)) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
        "meshBinaryDeserialize: failed to read LOD records");
  }

  MeshBinaryDecodedMesh decoded{};
  decoded.packedVertexBytes = std::move(decodedVerticesResult.value());
  decoded.vertexCount = vbufMeta.elementCount;
  decoded.vertexStrideBytes = vbufMeta.elementStrideBytes;
  decoded.bounds =
      BoundingBox(glm::vec3(header.modelBoundsMin[0], header.modelBoundsMin[1],
                            header.modelBoundsMin[2]),
                  glm::vec3(header.modelBoundsMax[0], header.modelBoundsMax[1],
                            header.modelBoundsMax[2]));

  const std::vector<std::byte> &decodedIndexBytes =
      decodedIndicesResult.value();
  if ((decodedIndexBytes.size() % sizeof(uint32_t)) != 0u) {
    return makeSerializerError<MeshBinaryDecodedMesh>(
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
      return makeSerializerError<MeshBinaryDecodedMesh>(
          "meshBinaryDeserialize: invalid submesh LOD count");
    }
    uint64_t lodEnd = 0;
    if (!checkedAddToU64(record.lodFirst, record.lodCount, lodEnd) ||
        lodEnd > lodRecords.size()) {
      return makeSerializerError<MeshBinaryDecodedMesh>(
          "meshBinaryDeserialize: submesh LOD range out of bounds");
    }

    Submesh submesh{};
    submesh.materialIndex = record.materialIndex;
    submesh.bounds =
        BoundingBox(glm::vec3(record.boundsMin[0], record.boundsMin[1],
                              record.boundsMin[2]),
                    glm::vec3(record.boundsMax[0], record.boundsMax[1],
                              record.boundsMax[2]));
    submesh.lodCount = record.lodCount;

    for (uint32_t lodIndex = 0; lodIndex < record.lodCount; ++lodIndex) {
      const MeshBinaryLodRecord &lodRecord =
          lodRecords[record.lodFirst + lodIndex];
      uint64_t indexRangeEnd = 0;
      if (!checkedAddToU64(lodRecord.indexOffset, lodRecord.indexCount,
                           indexRangeEnd) ||
          indexRangeEnd > decoded.indices.size()) {
        return makeSerializerError<MeshBinaryDecodedMesh>(
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

  return Result<MeshBinaryDecodedMesh, std::string>::makeResult(
      std::move(decoded));
}

} // namespace nuri
