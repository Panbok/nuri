#include "nuri/resources/detail/gltf_buffer_utils.h"
#include "nuri/resources/detail/gltf_json_utils.h"
#include "nuri/resources/storage/cache_utils.h"
namespace nuri::detail {
namespace {
constexpr uint32_t kGlbMagic = 0x46546C67u;
constexpr uint32_t kGlbChunkTypeBin = 0x004E4942u;
template <typename T> [[nodiscard]] T loadUnaligned(const std::byte *ptr) {
  static_assert(std::is_trivially_copyable_v<T>);
  T value{};
  std::memcpy(&value, ptr, sizeof(T));
  return value;
}
[[nodiscard]] bool readU32(std::span<const std::byte> bytes, size_t offset,
                           uint32_t &out) {
  if (offset + sizeof(uint32_t) > bytes.size()) {
    return false;
  }
  out = loadUnaligned<uint32_t>(bytes.data() + offset);
  return true;
}
[[nodiscard]] uint32_t gltfComponentTypeSize(uint32_t componentType) {
  static constexpr std::array sizes{1u, 1u, 2u, 2u, 4u, 4u};
  static constexpr std::array types{5120u, 5121u, 5122u, 5123u, 5125u, 5126u};
  for (size_t i = 0; i < types.size(); ++i) {
    if (componentType == types[i]) {
      return sizes[i];
    }
  }
  return 0u;
}
[[nodiscard]] uint32_t gltfAccessorComponentCount(std::string_view type) {
  static constexpr std::array names{"SCALAR", "VEC2", "VEC3", "VEC4",
                                    "MAT2",   "MAT3", "MAT4"};
  static constexpr std::array counts{1u, 2u, 3u, 4u, 4u, 9u, 16u};
  for (size_t i = 0; i < names.size(); ++i) {
    if (type == names[i]) {
      return counts[i];
    }
  }
  return 0u;
}
[[nodiscard]] bool
isPathWithinDirectory(const std::filesystem::path &path,
                      const std::filesystem::path &directory) {
  std::error_code ec;
  const auto canonicalPath = std::filesystem::weakly_canonical(path, ec);
  if (ec) {
    return false;
  }
  const auto canonicalDirectory =
      std::filesystem::weakly_canonical(directory, ec);
  if (ec) {
    return false;
  }
  const auto relative = canonicalPath.lexically_relative(canonicalDirectory);
  return relative.empty() || *relative.begin() != "..";
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
loadGlbBinaryChunk(std::span<const std::byte> fileBytes) {
  uint32_t magic = 0u;
  uint32_t jsonLength = 0u;
  if (fileBytes.size() < 28u || !readU32(fileBytes, 0u, magic) ||
      magic != kGlbMagic || !readU32(fileBytes, 12u, jsonLength)) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "glTF GLB header is invalid");
  }
  const size_t header = 20u + jsonLength;
  uint32_t chunkLength = 0u;
  uint32_t chunkType = 0u;
  if (!readU32(fileBytes, header, chunkLength) ||
      !readU32(fileBytes, header + 4u, chunkType) ||
      chunkType != kGlbChunkTypeBin ||
      header + 8u + chunkLength > fileBytes.size()) {
    return Result<std::vector<std::byte>, std::string>::makeError(
        "GLB binary chunk is missing or invalid");
  }
  std::vector<std::byte> chunk(chunkLength);
  std::memcpy(chunk.data(), fileBytes.data() + header + 8u, chunkLength);
  return Result<std::vector<std::byte>, std::string>::makeResult(
      std::move(chunk));
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
  auto infoResult = describeGltfAccessor(root, accessorIndex);
  if (infoResult.hasError()) {
    return Result<AccessorResolvedView, std::string>::makeError(
        infoResult.error());
  }
  const GltfAccessorInfo info = infoResult.value();
  const uint32_t componentSize = gltfComponentTypeSize(info.componentType);
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
  const uint32_t packedStride = info.componentCount * componentSize;
  const uint32_t elementStride = byteStride != 0u ? byteStride : packedStride;
  if (elementStride < packedStride) {
    return Result<AccessorResolvedView, std::string>::makeError(
        "glTF accessor byteStride is smaller than packed element size");
  }
  const uint64_t start = static_cast<uint64_t>(byteOffset) + accessorByteOffset;
  const uint64_t extent =
      info.count == 0u
          ? 0u
          : static_cast<uint64_t>(elementStride) * (info.count - 1u) +
                packedStride;
  if (start + extent > buffers[bufferIndex].size() ||
      start > buffers[bufferIndex].size() ||
      static_cast<uint64_t>(byteOffset) + byteLength >
          buffers[bufferIndex].size()) {
    return Result<AccessorResolvedView, std::string>::makeError(
        "glTF accessor range exceeds buffer size");
  }
  return Result<AccessorResolvedView, std::string>::makeResult(
      AccessorResolvedView{
          .bytes = std::span<const std::byte>(buffers[bufferIndex].data() +
                                                  static_cast<ptrdiff_t>(start),
                                              static_cast<size_t>(extent)),
          .count = info.count,
          .componentType = info.componentType,
          .componentCount = info.componentCount,
          .normalized = info.normalized,
          .elementStride = elementStride,
      });
}
[[nodiscard]] float normalizedComponentValue(uint32_t componentType,
                                             bool normalized,
                                             const std::byte *ptr) {
  switch (componentType) {
  case 5126:
    return loadUnaligned<float>(ptr);
  case 5120: {
    const int8_t value = std::bit_cast<int8_t>(*ptr);
    if (!normalized) {
      return static_cast<float>(value);
    }
    return std::max(static_cast<float>(value) / 127.0f, -1.0f);
  }
  case 5121: {
    const uint8_t value = std::to_integer<uint8_t>(*ptr);
    if (!normalized) {
      return static_cast<float>(value);
    }
    return static_cast<float>(value) / 255.0f;
  }
  case 5122: {
    const int16_t value = loadUnaligned<int16_t>(ptr);
    if (!normalized) {
      return static_cast<float>(value);
    }
    return std::max(static_cast<float>(value) / 32767.0f, -1.0f);
  }
  case 5123: {
    const uint16_t value = loadUnaligned<uint16_t>(ptr);
    if (!normalized) {
      return static_cast<float>(value);
    }
    return static_cast<float>(value) / 65535.0f;
  }
  case 5125:
    return static_cast<float>(loadUnaligned<uint32_t>(ptr));
  default:
    return 0.0f;
  }
}
[[nodiscard]] uint16_t u16ComponentValue(uint32_t componentType,
                                         const std::byte *ptr) {
  switch (componentType) {
  case 5121:
    return std::to_integer<uint8_t>(*ptr);
  case 5123:
    return loadUnaligned<uint16_t>(ptr);
  default:
    return 0u;
  }
}
template <typename T, typename Read>
[[nodiscard]] Result<std::pmr::vector<T>, std::string>
readAccessorValues(yyjson_val *root,
                   std::span<const std::pmr::vector<std::byte>> buffers,
                   uint32_t accessorIndex, std::pmr::memory_resource *memory,
                   std::string_view conversionError, Read &&read) {
  memory = memory ? memory : std::pmr::get_default_resource();
  auto resolvedResult = resolveAccessor(root, buffers, accessorIndex);
  if (resolvedResult.hasError()) {
    return Result<std::pmr::vector<T>, std::string>::makeError(
        resolvedResult.error());
  }
  const AccessorResolvedView resolved = resolvedResult.value();
  const uint32_t componentSize = gltfComponentTypeSize(resolved.componentType);
  std::pmr::vector<T> values(memory);
  values.resize(static_cast<size_t>(resolved.count) * resolved.componentCount);
  for (uint32_t element = 0; element < resolved.count; ++element) {
    const std::byte *source =
        resolved.bytes.data() + element * resolved.elementStride;
    for (uint32_t component = 0; component < resolved.componentCount;
         ++component) {
      T &value = values[static_cast<size_t>(element) * resolved.componentCount +
                        component];
      if (!read(resolved.componentType, resolved.normalized,
                source + component * componentSize, value)) {
        return Result<std::pmr::vector<T>, std::string>::makeError(
            std::string(conversionError));
      }
    }
  }
  return Result<std::pmr::vector<T>, std::string>::makeResult(
      std::move(values));
}
} // namespace

Result<std::pmr::vector<std::pmr::vector<std::byte>>, std::string>
loadGltfBuffers(const std::filesystem::path &path, yyjson_val *root,
                std::span<const std::byte> sourceBytes,
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
  const bool isGlb = pathHasExtensionCaseInsensitive(path, ".glb");
  if (isGlb) {
    auto glbChunkResult = loadGlbBinaryChunk(sourceBytes);
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
      std::filesystem::path resolvedUriPath;
      try {
        const std::filesystem::path baseDirectory =
            directory.empty() ? std::filesystem::path(".") : directory;
        const std::filesystem::path candidatePath =
            baseDirectory / std::filesystem::path(std::string(uri));
        resolvedUriPath = std::filesystem::weakly_canonical(candidatePath);
        if (!isPathWithinDirectory(resolvedUriPath, baseDirectory)) {
          return Result<std::pmr::vector<std::pmr::vector<std::byte>>,
                        std::string>::
              makeError(
                  "glTF buffer URI resolves outside the source directory");
        }
      } catch (const std::filesystem::filesystem_error &e) {
        return Result<std::pmr::vector<std::pmr::vector<std::byte>>,
                      std::string>::
            makeError(std::string("glTF buffer URI canonicalization failed: ") +
                      e.what());
      }
      const auto fileResult = readBinaryFile(resolvedUriPath);
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
  if (!yyjson_is_obj(accessorValue)) {
    return Result<GltfAccessorInfo, std::string>::makeError(
        "glTF accessor entry is not an object");
  }
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
  return readAccessorValues<float>(
      root, buffers, accessorIndex, memory, {},
      [](uint32_t type, bool normalized, const std::byte *source, float &out) {
        out = normalizedComponentValue(type, normalized, source);
        return true;
      });
}

Result<std::pmr::vector<uint16_t>, std::string> readGltfAccessorAsU16Array(
    yyjson_val *root, std::span<const std::pmr::vector<std::byte>> buffers,
    uint32_t accessorIndex, std::pmr::memory_resource *memory) {
  return readAccessorValues<uint16_t>(
      root, buffers, accessorIndex, memory,
      "glTF accessor value is not compatible with uint16 output",
      [](uint32_t type, bool, const std::byte *source, uint16_t &out) {
        if (type == 5125u) {
          const uint32_t value = loadUnaligned<uint32_t>(source);
          out = static_cast<uint16_t>(value);
          return value <= std::numeric_limits<uint16_t>::max();
        }
        out = u16ComponentValue(type, source);
        return type == 5121u || type == 5123u;
      });
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
  static_assert(sizeof(glm::mat4) == 16u * sizeof(float));
  std::pmr::vector<glm::mat4> matrices(memory);
  matrices.resize(info.count);
  const std::pmr::vector<float> &values = valuesResult.value();
  std::memcpy(matrices.data(), values.data(), values.size() * sizeof(float));
  return Result<std::pmr::vector<glm::mat4>, std::string>::makeResult(
      std::move(matrices));
}

} // namespace nuri::detail
