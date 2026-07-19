#include "nuri/resources/detail/gltf_json_utils.h"
#include "nuri/pch.h"
namespace nuri::detail {
namespace {
constexpr uint32_t kGlbMagic = 0x46546C67u;
constexpr uint32_t kGlbVersion2 = 2u;
constexpr uint32_t kGlbChunkTypeJson = 0x4E4F534Au;
[[nodiscard]] YyJsonDocResult parseJsonDocument(std::string_view jsonText) {
  std::string mutableJson(jsonText);
  yyjson_read_err parseError{};
  YyJsonDocPtr doc(yyjson_read_opts(mutableJson.data(), mutableJson.size(), 0,
                                    nullptr, &parseError),
                   &yyjson_doc_free);
  if (doc == nullptr) {
    std::string error = "Failed to parse glTF JSON near byte " +
                        std::to_string(static_cast<size_t>(parseError.pos));
    if (parseError.msg != nullptr && parseError.msg[0] != '\0') {
      error += ": ";
      error += parseError.msg;
    }
    return YyJsonDocResult::makeError(std::move(error));
  }
  return YyJsonDocResult::makeResult(std::move(doc));
}
[[nodiscard]] bool readU32(std::span<const uint8_t> bytes, size_t offset,
                           uint32_t &out) {
  if (offset + sizeof(uint32_t) > bytes.size()) {
    return false;
  }
  out = static_cast<uint32_t>(bytes[offset]) |
        (static_cast<uint32_t>(bytes[offset + 1u]) << 8u) |
        (static_cast<uint32_t>(bytes[offset + 2u]) << 16u) |
        (static_cast<uint32_t>(bytes[offset + 3u]) << 24u);
  return true;
}
template <size_t Size, typename Vector>
[[nodiscard]] bool readJsonVector(yyjson_val *value, Vector &out) {
  if (!yyjson_is_arr(value) || yyjson_arr_size(value) < Size) {
    return false;
  }
  Vector result{};
  for (size_t index = 0; index < Size; ++index) {
    if (!tryReadJsonFloat(yyjson_arr_get(value, index), result[index])) {
      return false;
    }
  }
  out = result;
  return true;
}
[[nodiscard]] YyJsonDocResult
loadGltfJsonDocumentFromBytes(const std::filesystem::path &path,
                              std::span<const std::byte> fileBytes) {
  if (hasExtensionCaseInsensitive(path, ".gltf")) {
    return parseJsonDocument(std::string_view(
        reinterpret_cast<const char *>(fileBytes.data()), fileBytes.size()));
  }
  if (!hasExtensionCaseInsensitive(path, ".glb")) {
    return YyJsonDocResult::makeError("Unsupported glTF file extension");
  }
  if (fileBytes.size() < 20u) {
    return YyJsonDocResult::makeError(".glb file is too small");
  }
  const std::span<const uint8_t> bytes(
      reinterpret_cast<const uint8_t *>(fileBytes.data()), fileBytes.size());
  uint32_t magic = 0u;
  uint32_t version = 0u;
  uint32_t declaredLength = 0u;
  if (!readU32(bytes, 0u, magic) || !readU32(bytes, 4u, version) ||
      !readU32(bytes, 8u, declaredLength) || magic != kGlbMagic ||
      version != kGlbVersion2 || declaredLength != bytes.size()) {
    return YyJsonDocResult::makeError("Invalid .glb header");
  }
  uint32_t chunkLength = 0u;
  uint32_t chunkType = 0u;
  constexpr size_t kChunkDataOffset = 20u;
  if (!readU32(bytes, 12u, chunkLength) || !readU32(bytes, 16u, chunkType) ||
      chunkType != kGlbChunkTypeJson ||
      chunkLength > bytes.size() - kChunkDataOffset) {
    return YyJsonDocResult::makeError("Invalid .glb JSON chunk");
  }
  return parseJsonDocument(std::string_view(
      reinterpret_cast<const char *>(bytes.data() + kChunkDataOffset),
      chunkLength));
}
} // namespace

bool hasExtensionCaseInsensitive(const std::filesystem::path &path,
                                 std::string_view extension) {
  const std::string pathExtension = path.extension().string();
  const auto trimLeadingDot = [](std::string_view value) {
    return value.starts_with('.') ? value.substr(1) : value;
  };
  const std::string_view lhs = trimLeadingDot(pathExtension);
  const std::string_view rhs = trimLeadingDot(extension);
  return std::ranges::equal(lhs, rhs, [](char left, char right) {
    return std::tolower(static_cast<unsigned char>(left)) ==
           std::tolower(static_cast<unsigned char>(right));
  });
}

bool isGltfJsonAssetPath(const std::filesystem::path &path) {
  return hasExtensionCaseInsensitive(path, ".gltf") ||
         hasExtensionCaseInsensitive(path, ".glb");
}

bool isGltfJsonAssetPath(std::string_view path) {
  return isGltfJsonAssetPath(std::filesystem::path(std::string(path)));
}

std::string_view readJsonStringView(yyjson_val *value) {
  return yyjson_is_str(value)
             ? std::string_view(yyjson_get_str(value), yyjson_get_len(value))
             : std::string_view{};
}

std::string_view readJsonStringView(yyjson_val *object, const char *key) {
  return readJsonStringView(yyjson_obj_get(object, key));
}

bool tryReadJsonFloat(yyjson_val *value, float &out) {
  if (!yyjson_is_num(value)) {
    return false;
  }
  out = static_cast<float>(yyjson_get_num(value));
  return true;
}

bool tryReadJsonUint32(yyjson_val *value, uint32_t &out) {
  if (!yyjson_is_uint(value) && !yyjson_is_sint(value)) {
    return false;
  }
  const int64_t raw = yyjson_is_uint(value)
                          ? static_cast<int64_t>(yyjson_get_uint(value))
                          : yyjson_get_sint(value);
  if (raw < 0 || raw > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  out = static_cast<uint32_t>(raw);
  return true;
}

bool tryReadJsonVec2(yyjson_val *value, glm::vec2 &out) {
  return readJsonVector<2>(value, out);
}

bool tryReadJsonVec3(yyjson_val *value, glm::vec3 &out) {
  return readJsonVector<3>(value, out);
}

bool tryReadJsonVec4(yyjson_val *value, glm::vec4 &out) {
  return readJsonVector<4>(value, out);
}

bool tryReadJsonMat4(yyjson_val *value, glm::mat4 &out) {
  if (!yyjson_is_arr(value) || yyjson_arr_size(value) < 16u) {
    return false;
  }
  glm::mat4 matrix(1.0f);
  for (uint32_t column = 0; column < 4u; ++column) {
    for (uint32_t row = 0; row < 4u; ++row) {
      float component = 0.0f;
      if (!tryReadJsonFloat(yyjson_arr_get(value, column * 4u + row),
                            component)) {
        return false;
      }
      matrix[column][row] = component;
    }
  }
  out = matrix;
  return true;
}

bool tryReadJsonBool(yyjson_val *value, bool &out) {
  if (!yyjson_is_bool(value)) {
    return false;
  }
  out = yyjson_get_bool(value);
  return true;
}

YyJsonDocResult loadGltfJsonDocument(const std::filesystem::path &path,
                                     std::string_view sourceLabel) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    return YyJsonDocResult::makeError("Failed to open " +
                                      std::string(sourceLabel));
  }
  file.seekg(0, std::ios::end);
  const std::streamoff fileSize = file.tellg();
  if (fileSize < 0) {
    return YyJsonDocResult::makeError("Failed to determine glTF file size");
  }
  file.seekg(0, std::ios::beg);
  std::vector<std::byte> bytes(static_cast<size_t>(fileSize));
  if (!bytes.empty() &&
      !file.read(reinterpret_cast<char *>(bytes.data()), fileSize)) {
    return YyJsonDocResult::makeError("Failed to read glTF file");
  }
  return loadGltfJsonDocumentFromBytes(path, bytes);
}

YyJsonDocResult loadGltfJsonDocument(const std::filesystem::path &path,
                                     std::span<const std::byte> fileBytes,
                                     std::string_view sourceLabel) {
  if (fileBytes.empty()) {
    return YyJsonDocResult::makeError("Failed to read " +
                                      std::string(sourceLabel));
  }
  return loadGltfJsonDocumentFromBytes(path, fileBytes);
}

Result<GltfPrimitiveMaterialMapping, std::string>
readGltfPrimitiveMaterialMapping(yyjson_val *root) {
  if (!yyjson_is_obj(root)) {
    return Result<GltfPrimitiveMaterialMapping, std::string>::makeError(
        "glTF root is not a JSON object");
  }
  GltfPrimitiveMaterialMapping mapping{};
  if (yyjson_val *materials = yyjson_obj_get(root, "materials");
      yyjson_is_arr(materials)) {
    mapping.materialCount = static_cast<uint32_t>(std::min<size_t>(
        yyjson_arr_size(materials), std::numeric_limits<uint32_t>::max()));
  }
  yyjson_val *meshes = yyjson_obj_get(root, "meshes");
  if (!yyjson_is_arr(meshes)) {
    return Result<GltfPrimitiveMaterialMapping, std::string>::makeResult(
        std::move(mapping));
  }
  const size_t meshArraySize = yyjson_arr_size(meshes);
  mapping.meshCount = static_cast<uint32_t>(
      std::min<size_t>(meshArraySize, std::numeric_limits<uint32_t>::max()));
  mapping.singlePrimitiveMeshMaterialIndices.reserve(mapping.meshCount);
  mapping.primitiveMaterialIndices.reserve(mapping.meshCount);
  yyjson_arr_iter meshIter = yyjson_arr_iter_with(meshes);
  yyjson_val *meshValue = nullptr;
  for (uint32_t meshIndex = 0;
       meshIndex < mapping.meshCount &&
       (meshValue = yyjson_arr_iter_next(&meshIter)) != nullptr;
       ++meshIndex) {
    yyjson_val *primitives = yyjson_obj_get(meshValue, "primitives");
    if (!yyjson_is_arr(primitives)) {
      mapping.singlePrimitiveMeshMaterialIndices.push_back(
          kInvalidMaterialIndex);
      continue;
    }
    const size_t primitiveCount = yyjson_arr_size(primitives);
    uint32_t singlePrimitiveMaterialIndex = kInvalidMaterialIndex;
    yyjson_arr_iter primitiveIter = yyjson_arr_iter_with(primitives);
    yyjson_val *primitiveValue = nullptr;
    while ((primitiveValue = yyjson_arr_iter_next(&primitiveIter)) != nullptr) {
      uint32_t materialIndex = kInvalidMaterialIndex;
      (void)tryReadJsonUint32(yyjson_obj_get(primitiveValue, "material"),
                              materialIndex);
      if (primitiveCount == 1u) {
        singlePrimitiveMaterialIndex = materialIndex;
      }
      mapping.primitiveMaterialIndices.push_back(materialIndex);
    }
    mapping.singlePrimitiveMeshMaterialIndices.push_back(
        singlePrimitiveMaterialIndex);
  }
  mapping.sceneMeshIndicesAreFlatPrimitiveOrder =
      mapping.meshCount == 1u && mapping.primitiveMaterialIndices.size() > 1u;
  return Result<GltfPrimitiveMaterialMapping, std::string>::makeResult(
      std::move(mapping));
}

} // namespace nuri::detail
