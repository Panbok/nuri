#include "nuri/pch.h"

#include "nuri/resources/detail/gltf_buffer_utils.h"

#include "nuri/resources/detail/gltf_json_utils.h"
#include "nuri/resources/storage/mesh/mesh_cache_utils.h"

namespace nuri::detail {
namespace {

constexpr uint32_t kGlbMagic = 0x46546C67u;
constexpr uint32_t kGlbChunkTypeBin = 0x004E4942u;

[[nodiscard]] bool readU32(std::span<const std::byte> bytes, size_t offset,
                           uint32_t &out) {
  if (offset + sizeof(uint32_t) > bytes.size()) {
    return false;
  }
  const auto *data = reinterpret_cast<const uint8_t *>(
      bytes.data() + static_cast<ptrdiff_t>(offset));
  out = static_cast<uint32_t>(data[0]) |
        (static_cast<uint32_t>(data[1]) << 8u) |
        (static_cast<uint32_t>(data[2]) << 16u) |
        (static_cast<uint32_t>(data[3]) << 24u);
  return true;
}

[[nodiscard]] uint32_t gltfComponentTypeSize(uint32_t componentType) {
  switch (componentType) {
  case 5120:
  case 5121:
    return 1u;
  case 5122:
  case 5123:
    return 2u;
  case 5125:
  case 5126:
    return 4u;
  default:
    return 0u;
  }
}

[[nodiscard]] uint32_t gltfAccessorComponentCount(std::string_view type) {
  if (type == "SCALAR") {
    return 1u;
  }
  if (type == "VEC2") {
    return 2u;
  }
  if (type == "VEC3") {
    return 3u;
  }
  if (type == "VEC4") {
    return 4u;
  }
  if (type == "MAT4") {
    return 16u;
  }
  return 0u;
}

struct AccessorResolvedView {
  std::span<const std::byte> bytes{};
  uint32_t count = 0;
  uint32_t componentType = 0;
  uint32_t componentCount = 0;
  bool normalized = false;
  uint32_t elementStride = 0;
};

[[nodiscard]] Result<std::vector<std::byte>, std::string>
loadGlbBinaryChunk(const std::filesystem::path &path) {
  auto fileBytesResult = readBinaryFile(path);
  if (fileBytesResult.hasError()) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        fileBytesResult.error());
  }
  const std::vector<std::byte> &fileBytes = fileBytesResult.value();
  if (fileBytes.size() < 20u) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "glTF GLB file is too small");
  }
  uint32_t magic = 0u;
  if (!readU32(fileBytes, 0u, magic) || magic != kGlbMagic) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "glTF GLB magic mismatch");
  }

  size_t offset = 12u;
  while (offset + 8u <= fileBytes.size()) {
    uint32_t chunkLength = 0u;
    uint32_t chunkType = 0u;
    if (!readU32(fileBytes, offset, chunkLength) ||
        !readU32(fileBytes, offset + 4u, chunkType)) {
      return Result<std::vector<std::byte>, std::string>::makeError(
          "Failed to read GLB chunk header");
    }
    offset += 8u;
    if (offset + chunkLength > fileBytes.size()) {
      return Result<std::vector<std::byte>, std::string>::makeError(
          "GLB chunk exceeds file bounds");
    }
    if (chunkType == kGlbChunkTypeBin) {
      std::vector<std::byte> chunk(chunkLength);
      if (chunkLength > 0u) {
        std::memcpy(chunk.data(),
                    fileBytes.data() + static_cast<ptrdiff_t>(offset),
                    chunkLength);
      }
      return Result<std::vector<std::byte>, std::string>::makeResult(
          std::move(chunk));
    }
    offset += chunkLength;
  }

  return Result<std::vector<std::byte>, std::string>::makeError(
      "GLB binary chunk is missing");
}

[[nodiscard]] Result<AccessorResolvedView, std::string>
resolveAccessor(yyjson_val *root,
                std::span<const std::pmr::vector<std::byte>> buffers,
                uint32_t accessorIndex) {
  yyjson_val *accessorsValue = yyjson_obj_get(root, "accessors");
  if (!yyjson_is_arr(accessorsValue) ||
      accessorIndex >= yyjson_arr_size(accessorsValue)) {
    return Result<AccessorResolvedView, std::string>::makeError(
        "glTF accessor index is out of range");
  }
  yyjson_val *accessorValue = yyjson_arr_get(accessorsValue, accessorIndex);
  if (!yyjson_is_obj(accessorValue)) {
    return Result<AccessorResolvedView, std::string>::makeError(
        "glTF accessor entry is invalid");
  }
  if (yyjson_obj_get(accessorValue, "sparse") != nullptr) {
    return Result<AccessorResolvedView, std::string>::makeError(
        "glTF sparse accessors are not supported");
  }

  uint32_t bufferViewIndex = 0u;
  if (!tryReadJsonUint32(yyjson_obj_get(accessorValue, "bufferView"),
                         bufferViewIndex)) {
    return Result<AccessorResolvedView, std::string>::makeError(
        "glTF accessor bufferView is invalid");
  }
  uint32_t count = 0u;
  if (!tryReadJsonUint32(yyjson_obj_get(accessorValue, "count"), count)) {
    return Result<AccessorResolvedView, std::string>::makeError(
        "glTF accessor count is invalid");
  }
  uint32_t componentType = 0u;
  if (!tryReadJsonUint32(yyjson_obj_get(accessorValue, "componentType"),
                         componentType)) {
    return Result<AccessorResolvedView, std::string>::makeError(
        "glTF accessor componentType is invalid");
  }
  const std::string_view type = readJsonStringView(accessorValue, "type");
  const uint32_t componentCount = gltfAccessorComponentCount(type);
  const uint32_t componentSize = gltfComponentTypeSize(componentType);
  if (componentCount == 0u || componentSize == 0u) {
    return Result<AccessorResolvedView, std::string>::makeError(
        "glTF accessor layout is unsupported");
  }

  yyjson_val *bufferViewsValue = yyjson_obj_get(root, "bufferViews");
  if (!yyjson_is_arr(bufferViewsValue) ||
      bufferViewIndex >= yyjson_arr_size(bufferViewsValue)) {
    return Result<AccessorResolvedView, std::string>::makeError(
        "glTF bufferView index is out of range");
  }
  yyjson_val *bufferViewValue =
      yyjson_arr_get(bufferViewsValue, bufferViewIndex);
  if (!yyjson_is_obj(bufferViewValue)) {
    return Result<AccessorResolvedView, std::string>::makeError(
        "glTF bufferView entry is invalid");
  }

  uint32_t bufferIndex = 0u;
  if (!tryReadJsonUint32(yyjson_obj_get(bufferViewValue, "buffer"),
                         bufferIndex) ||
      bufferIndex >= buffers.size()) {
    return Result<AccessorResolvedView, std::string>::makeError(
        "glTF bufferView buffer index is invalid");
  }
  uint32_t byteOffset = 0u;
  (void)tryReadJsonUint32(yyjson_obj_get(bufferViewValue, "byteOffset"),
                          byteOffset);
  uint32_t accessorByteOffset = 0u;
  (void)tryReadJsonUint32(yyjson_obj_get(accessorValue, "byteOffset"),
                          accessorByteOffset);
  uint32_t byteLength = 0u;
  if (!tryReadJsonUint32(yyjson_obj_get(bufferViewValue, "byteLength"),
                         byteLength)) {
    return Result<AccessorResolvedView, std::string>::makeError(
        "glTF bufferView byteLength is invalid");
  }
  uint32_t byteStride = 0u;
  (void)tryReadJsonUint32(yyjson_obj_get(bufferViewValue, "byteStride"),
                          byteStride);
  const uint32_t packedStride = componentCount * componentSize;
  const uint32_t elementStride = byteStride != 0u ? byteStride : packedStride;
  if (elementStride < packedStride) {
    return Result<AccessorResolvedView, std::string>::makeError(
        "glTF accessor byteStride is smaller than packed element size");
  }

  const uint64_t start = static_cast<uint64_t>(byteOffset) + accessorByteOffset;
  const uint64_t extent =
      count == 0u
          ? 0u
          : static_cast<uint64_t>(elementStride) * (count - 1u) + packedStride;
  if (start + extent > buffers[bufferIndex].size() ||
      start > buffers[bufferIndex].size() ||
      static_cast<uint64_t>(byteOffset) + byteLength >
          buffers[bufferIndex].size()) {
    return Result<AccessorResolvedView, std::string>::makeError(
        "glTF accessor range exceeds buffer size");
  }

  bool normalized = false;
  (void)tryReadJsonBool(yyjson_obj_get(accessorValue, "normalized"),
                        normalized);
  return Result<AccessorResolvedView, std::string>::makeResult(
      AccessorResolvedView{
          .bytes = std::span<const std::byte>(buffers[bufferIndex].data() +
                                                  static_cast<ptrdiff_t>(start),
                                              static_cast<size_t>(extent)),
          .count = count,
          .componentType = componentType,
          .componentCount = componentCount,
          .normalized = normalized,
          .elementStride = elementStride,
      });
}

[[nodiscard]] float normalizedComponentValue(uint32_t componentType,
                                             bool normalized,
                                             const std::byte *ptr) {
  switch (componentType) {
  case 5126:
    return *reinterpret_cast<const float *>(ptr);
  case 5120: {
    const int8_t value = *reinterpret_cast<const int8_t *>(ptr);
    if (!normalized) {
      return static_cast<float>(value);
    }
    return std::max(static_cast<float>(value) / 127.0f, -1.0f);
  }
  case 5121: {
    const uint8_t value = *reinterpret_cast<const uint8_t *>(ptr);
    if (!normalized) {
      return static_cast<float>(value);
    }
    return static_cast<float>(value) / 255.0f;
  }
  case 5122: {
    const int16_t value = *reinterpret_cast<const int16_t *>(ptr);
    if (!normalized) {
      return static_cast<float>(value);
    }
    return std::max(static_cast<float>(value) / 32767.0f, -1.0f);
  }
  case 5123: {
    const uint16_t value = *reinterpret_cast<const uint16_t *>(ptr);
    if (!normalized) {
      return static_cast<float>(value);
    }
    return static_cast<float>(value) / 65535.0f;
  }
  case 5125:
    return static_cast<float>(*reinterpret_cast<const uint32_t *>(ptr));
  default:
    return 0.0f;
  }
}

[[nodiscard]] uint16_t u16ComponentValue(uint32_t componentType,
                                         const std::byte *ptr) {
  switch (componentType) {
  case 5121:
    return *reinterpret_cast<const uint8_t *>(ptr);
  case 5123:
    return *reinterpret_cast<const uint16_t *>(ptr);
  default:
    return 0u;
  }
}

} // namespace

Result<std::pmr::vector<std::pmr::vector<std::byte>>, std::string>
loadGltfBuffers(const std::filesystem::path &path, yyjson_val *root,
                std::pmr::memory_resource *memory) {
  if (memory == nullptr) {
    memory = std::pmr::get_default_resource();
  }
  std::pmr::vector<std::pmr::vector<std::byte>> buffers(memory);
  yyjson_val *buffersValue = yyjson_obj_get(root, "buffers");
  if (!yyjson_is_arr(buffersValue)) {
    return Result<std::pmr::vector<std::pmr::vector<std::byte>>,
                  std::string>::makeResult(std::move(buffers));
  }

  std::vector<std::byte> glbBinaryChunk;
  const bool isGlb = hasExtensionCaseInsensitive(path, ".glb");
  if (isGlb) {
    auto glbChunkResult = loadGlbBinaryChunk(path);
    if (glbChunkResult.hasError()) {
      return Result<std::pmr::vector<std::pmr::vector<std::byte>>,
                    std::string>::makeError(glbChunkResult.error());
    }
    glbBinaryChunk = std::move(glbChunkResult.value());
  }

  buffers.reserve(yyjson_arr_size(buffersValue));
  const std::filesystem::path directory = path.parent_path();
  yyjson_arr_iter iter = yyjson_arr_iter_with(buffersValue);
  yyjson_val *bufferValue = nullptr;
  while ((bufferValue = yyjson_arr_iter_next(&iter)) != nullptr) {
    if (!yyjson_is_obj(bufferValue)) {
      return Result<std::pmr::vector<std::pmr::vector<std::byte>>,
                    std::string>::makeError("glTF buffer entry is invalid");
    }
    uint32_t declaredLength = 0u;
    if (!tryReadJsonUint32(yyjson_obj_get(bufferValue, "byteLength"),
                           declaredLength)) {
      return Result<
          std::pmr::vector<std::pmr::vector<std::byte>>,
          std::string>::makeError("glTF buffer byteLength is invalid");
    }

    std::pmr::vector<std::byte> bytes(memory);
    const std::string_view uri = readJsonStringView(bufferValue, "uri");
    if (uri.empty()) {
      if (!isGlb) {
        return Result<std::pmr::vector<std::pmr::vector<std::byte>>,
                      std::string>::makeError("glTF buffer URI is missing");
      }
      bytes.assign(glbBinaryChunk.begin(), glbBinaryChunk.end());
    } else {
      if (uri.starts_with("data:")) {
        return Result<
            std::pmr::vector<std::pmr::vector<std::byte>>,
            std::string>::makeError("glTF data URI buffers are not supported");
      }
      const auto fileResult = readBinaryFile(directory / std::string(uri));
      if (fileResult.hasError()) {
        return Result<std::pmr::vector<std::pmr::vector<std::byte>>,
                      std::string>::makeError(fileResult.error());
      }
      const std::vector<std::byte> &fileBytes = fileResult.value();
      bytes.assign(fileBytes.begin(), fileBytes.end());
    }
    if (bytes.size() < declaredLength) {
      return Result<std::pmr::vector<std::pmr::vector<std::byte>>,
                    std::string>::
          makeError("glTF buffer byteLength exceeds loaded buffer size");
    }
    if (bytes.size() > declaredLength) {
      bytes.resize(declaredLength);
    }
    buffers.push_back(std::move(bytes));
  }

  return Result<std::pmr::vector<std::pmr::vector<std::byte>>,
                std::string>::makeResult(std::move(buffers));
}

Result<GltfAccessorInfo, std::string>
describeGltfAccessor(yyjson_val *root, uint32_t accessorIndex) {
  yyjson_val *accessorsValue = yyjson_obj_get(root, "accessors");
  if (!yyjson_is_arr(accessorsValue) ||
      accessorIndex >= yyjson_arr_size(accessorsValue)) {
    return Result<GltfAccessorInfo, std::string>::makeError(
        "glTF accessor index is out of range");
  }
  yyjson_val *accessorValue = yyjson_arr_get(accessorsValue, accessorIndex);
  uint32_t count = 0u;
  uint32_t componentType = 0u;
  if (!tryReadJsonUint32(yyjson_obj_get(accessorValue, "count"), count) ||
      !tryReadJsonUint32(yyjson_obj_get(accessorValue, "componentType"),
                         componentType)) {
    return Result<GltfAccessorInfo, std::string>::makeError(
        "glTF accessor metadata is invalid");
  }
  bool normalized = false;
  (void)tryReadJsonBool(yyjson_obj_get(accessorValue, "normalized"),
                        normalized);
  const uint32_t componentCount =
      gltfAccessorComponentCount(readJsonStringView(accessorValue, "type"));
  if (componentCount == 0u) {
    return Result<GltfAccessorInfo, std::string>::makeError(
        "glTF accessor type is unsupported");
  }
  return Result<GltfAccessorInfo, std::string>::makeResult(GltfAccessorInfo{
      .count = count,
      .componentType = componentType,
      .componentCount = componentCount,
      .normalized = normalized,
  });
}

Result<std::pmr::vector<float>, std::string> readGltfAccessorAsFloatArray(
    yyjson_val *root, std::span<const std::pmr::vector<std::byte>> buffers,
    uint32_t accessorIndex, std::pmr::memory_resource *memory) {
  if (memory == nullptr) {
    memory = std::pmr::get_default_resource();
  }
  auto resolvedResult = resolveAccessor(root, buffers, accessorIndex);
  if (resolvedResult.hasError()) {
    return Result<std::pmr::vector<float>, std::string>::makeError(
        resolvedResult.error());
  }
  const AccessorResolvedView resolved = resolvedResult.value();
  const uint32_t componentSize = gltfComponentTypeSize(resolved.componentType);
  std::pmr::vector<float> values(memory);
  values.resize(static_cast<size_t>(resolved.count) * resolved.componentCount);
  for (uint32_t elementIndex = 0; elementIndex < resolved.count;
       ++elementIndex) {
    const std::byte *elementPtr =
        resolved.bytes.data() +
        static_cast<ptrdiff_t>(elementIndex * resolved.elementStride);
    for (uint32_t componentIndex = 0; componentIndex < resolved.componentCount;
         ++componentIndex) {
      values[static_cast<size_t>(elementIndex) * resolved.componentCount +
             componentIndex] =
          normalizedComponentValue(
              resolved.componentType, resolved.normalized,
              elementPtr +
                  static_cast<ptrdiff_t>(componentIndex * componentSize));
    }
  }
  return Result<std::pmr::vector<float>, std::string>::makeResult(
      std::move(values));
}

Result<std::pmr::vector<uint16_t>, std::string> readGltfAccessorAsU16Array(
    yyjson_val *root, std::span<const std::pmr::vector<std::byte>> buffers,
    uint32_t accessorIndex, std::pmr::memory_resource *memory) {
  if (memory == nullptr) {
    memory = std::pmr::get_default_resource();
  }
  auto resolvedResult = resolveAccessor(root, buffers, accessorIndex);
  if (resolvedResult.hasError()) {
    return Result<std::pmr::vector<uint16_t>, std::string>::makeError(
        resolvedResult.error());
  }
  const AccessorResolvedView resolved = resolvedResult.value();
  if (resolved.componentType != 5121u && resolved.componentType != 5123u) {
    return Result<std::pmr::vector<uint16_t>, std::string>::makeError(
        "glTF accessor component type is not compatible with uint16 output");
  }
  const uint32_t componentSize = gltfComponentTypeSize(resolved.componentType);
  std::pmr::vector<uint16_t> values(memory);
  values.resize(static_cast<size_t>(resolved.count) * resolved.componentCount);
  for (uint32_t elementIndex = 0; elementIndex < resolved.count;
       ++elementIndex) {
    const std::byte *elementPtr =
        resolved.bytes.data() +
        static_cast<ptrdiff_t>(elementIndex * resolved.elementStride);
    for (uint32_t componentIndex = 0; componentIndex < resolved.componentCount;
         ++componentIndex) {
      values[static_cast<size_t>(elementIndex) * resolved.componentCount +
             componentIndex] =
          u16ComponentValue(resolved.componentType,
                            elementPtr + static_cast<ptrdiff_t>(componentIndex *
                                                                componentSize));
    }
  }
  return Result<std::pmr::vector<uint16_t>, std::string>::makeResult(
      std::move(values));
}

Result<std::pmr::vector<glm::mat4>, std::string> readGltfAccessorAsMat4Array(
    yyjson_val *root, std::span<const std::pmr::vector<std::byte>> buffers,
    uint32_t accessorIndex, std::pmr::memory_resource *memory) {
  auto infoResult = describeGltfAccessor(root, accessorIndex);
  if (infoResult.hasError()) {
    return Result<std::pmr::vector<glm::mat4>, std::string>::makeError(
        infoResult.error());
  }
  const GltfAccessorInfo info = infoResult.value();
  if (info.componentType != 5126u || info.componentCount != 16u) {
    return Result<std::pmr::vector<glm::mat4>, std::string>::makeError(
        "glTF accessor is not a float MAT4");
  }
  auto valuesResult =
      readGltfAccessorAsFloatArray(root, buffers, accessorIndex, memory);
  if (valuesResult.hasError()) {
    return Result<std::pmr::vector<glm::mat4>, std::string>::makeError(
        valuesResult.error());
  }
  std::pmr::vector<glm::mat4> matrices(memory);
  matrices.resize(info.count);
  const std::pmr::vector<float> &values = valuesResult.value();
  for (uint32_t matrixIndex = 0; matrixIndex < info.count; ++matrixIndex) {
    glm::mat4 matrix(1.0f);
    const size_t base = static_cast<size_t>(matrixIndex) * 16u;
    for (uint32_t column = 0; column < 4u; ++column) {
      for (uint32_t row = 0; row < 4u; ++row) {
        matrix[column][row] = values[base + column * 4u + row];
      }
    }
    matrices[matrixIndex] = matrix;
  }
  return Result<std::pmr::vector<glm::mat4>, std::string>::makeResult(
      std::move(matrices));
}

} // namespace nuri::detail
