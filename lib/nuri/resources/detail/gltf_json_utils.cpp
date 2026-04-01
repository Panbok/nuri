#include "nuri/pch.h"

#include "nuri/resources/detail/gltf_json_utils.h"

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

[[nodiscard]] YyJsonDocResult
loadGltfJsonDocumentFromBytes(const std::filesystem::path &path,
                              std::span<const std::byte> fileBytes,
                              std::string_view sourceLabel) {
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
      !readU32(bytes, 8u, declaredLength)) {
    return YyJsonDocResult::makeError("Failed to read .glb header");
  }
  if (magic != kGlbMagic) {
    return YyJsonDocResult::makeError(".glb magic mismatch");
  }
  if (version != kGlbVersion2) {
    return YyJsonDocResult::makeError(".glb version is not 2");
  }
  if (declaredLength != bytes.size()) {
    return YyJsonDocResult::makeError(".glb declared length mismatch");
  }

  size_t chunkOffset = 12u;
  while (chunkOffset + 8u <= bytes.size()) {
    uint32_t chunkLength = 0u;
    uint32_t chunkType = 0u;
    if (!readU32(bytes, chunkOffset, chunkLength) ||
        !readU32(bytes, chunkOffset + 4u, chunkType)) {
      return YyJsonDocResult::makeError("Failed to read .glb chunk header");
    }
    chunkOffset += 8u;
    if (chunkLength > bytes.size() - chunkOffset) {
      return YyJsonDocResult::makeError(".glb chunk exceeds file bounds");
    }
    if (chunkType == kGlbChunkTypeJson) {
      return parseJsonDocument(std::string_view(
          reinterpret_cast<const char *>(bytes.data() + chunkOffset),
          chunkLength));
    }
    chunkOffset += chunkLength;
  }

  (void)sourceLabel;
  return YyJsonDocResult::makeError(".glb JSON chunk is missing");
}

} // namespace

bool hasExtensionCaseInsensitive(const std::filesystem::path &path,
                                 std::string_view extension) {
  if (!path.has_extension()) {
    return false;
  }

  const std::string pathExtension = path.extension().string();
  const auto trimLeadingDot = [](std::string_view value) {
    return value.starts_with('.') ? value.substr(1) : value;
  };

  const std::string_view lhs = trimLeadingDot(pathExtension);
  const std::string_view rhs = trimLeadingDot(extension);
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (size_t i = 0; i < lhs.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(lhs[i])) !=
        std::tolower(static_cast<unsigned char>(rhs[i]))) {
      return false;
    }
  }
  return true;
}

bool isGltfJsonAssetPath(const std::filesystem::path &path) {
  return hasExtensionCaseInsensitive(path, ".gltf") ||
         hasExtensionCaseInsensitive(path, ".glb");
}

bool isGltfJsonAssetPath(std::string_view path) {
  return isGltfJsonAssetPath(std::filesystem::path(std::string(path)));
}

std::string_view readJsonStringView(yyjson_val *value) {
  if (!yyjson_is_str(value)) {
    return {};
  }
  const char *raw = yyjson_get_str(value);
  return raw != nullptr ? std::string_view(raw) : std::string_view{};
}

std::string_view readJsonStringView(yyjson_val *object, const char *key) {
  if (!yyjson_is_obj(object)) {
    return {};
  }
  return readJsonStringView(yyjson_obj_get(object, key));
}

bool tryReadJsonFloat(yyjson_val *value, float &out) {
  if (yyjson_is_uint(value)) {
    out = static_cast<float>(yyjson_get_uint(value));
    return true;
  }
  if (yyjson_is_sint(value)) {
    out = static_cast<float>(yyjson_get_sint(value));
    return true;
  }
  if (yyjson_is_real(value) || yyjson_is_num(value)) {
    out = static_cast<float>(yyjson_get_real(value));
    return true;
  }
  return false;
}

bool tryReadJsonUint32(yyjson_val *value, uint32_t &out) {
  if (yyjson_is_uint(value)) {
    const uint64_t raw = yyjson_get_uint(value);
    if (raw > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
    out = static_cast<uint32_t>(raw);
    return true;
  }
  if (yyjson_is_sint(value)) {
    const int64_t raw = yyjson_get_sint(value);
    if (raw < 0 ||
        raw > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
      return false;
    }
    out = static_cast<uint32_t>(raw);
    return true;
  }
  return false;
}

bool tryReadJsonVec2(yyjson_val *value, glm::vec2 &out) {
  if (!yyjson_is_arr(value) || yyjson_arr_size(value) < 2u) {
    return false;
  }

  float x = 0.0f;
  float y = 0.0f;
  if (!tryReadJsonFloat(yyjson_arr_get(value, 0u), x) ||
      !tryReadJsonFloat(yyjson_arr_get(value, 1u), y)) {
    return false;
  }
  out = glm::vec2(x, y);
  return true;
}

bool tryReadJsonVec3(yyjson_val *value, glm::vec3 &out) {
  if (!yyjson_is_arr(value) || yyjson_arr_size(value) < 3u) {
    return false;
  }

  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  if (!tryReadJsonFloat(yyjson_arr_get(value, 0u), x) ||
      !tryReadJsonFloat(yyjson_arr_get(value, 1u), y) ||
      !tryReadJsonFloat(yyjson_arr_get(value, 2u), z)) {
    return false;
  }
  out = glm::vec3(x, y, z);
  return true;
}

bool tryReadJsonVec4(yyjson_val *value, glm::vec4 &out) {
  if (!yyjson_is_arr(value) || yyjson_arr_size(value) < 4u) {
    return false;
  }

  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float w = 0.0f;
  if (!tryReadJsonFloat(yyjson_arr_get(value, 0u), x) ||
      !tryReadJsonFloat(yyjson_arr_get(value, 1u), y) ||
      !tryReadJsonFloat(yyjson_arr_get(value, 2u), z) ||
      !tryReadJsonFloat(yyjson_arr_get(value, 3u), w)) {
    return false;
  }
  out = glm::vec4(x, y, z, w);
  return true;
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

  if (hasExtensionCaseInsensitive(path, ".gltf")) {
    file.seekg(0, std::ios::end);
    const std::streamoff fileSize = file.tellg();
    if (fileSize < 0) {
      return YyJsonDocResult::makeError("Failed to determine .gltf file size");
    }
    file.seekg(0, std::ios::beg);

    std::string jsonText(static_cast<size_t>(fileSize), '\0');
    if (!jsonText.empty() && !file.read(jsonText.data(), fileSize)) {
      return YyJsonDocResult::makeError("Failed to read .gltf file");
    }
    return parseJsonDocument(jsonText);
  }

  if (!hasExtensionCaseInsensitive(path, ".glb")) {
    return YyJsonDocResult::makeError("Unsupported glTF file extension");
  }

  file.seekg(0, std::ios::end);
  const std::streamoff fileSize = file.tellg();
  if (fileSize < 0) {
    return YyJsonDocResult::makeError("Failed to determine .glb file size");
  }
  file.seekg(0, std::ios::beg);

  std::vector<uint8_t> bytes(static_cast<size_t>(fileSize));
  if (!bytes.empty() &&
      !file.read(reinterpret_cast<char *>(bytes.data()), fileSize)) {
    return YyJsonDocResult::makeError("Failed to read .glb file");
  }
  return loadGltfJsonDocumentFromBytes(
      path,
      std::span<const std::byte>(
          reinterpret_cast<const std::byte *>(bytes.data()), bytes.size()),
      sourceLabel);
}

YyJsonDocResult loadGltfJsonDocument(const std::filesystem::path &path,
                                     std::span<const std::byte> fileBytes,
                                     std::string_view sourceLabel) {
  if (fileBytes.empty()) {
    return YyJsonDocResult::makeError("Failed to read " +
                                      std::string(sourceLabel));
  }
  return loadGltfJsonDocumentFromBytes(path, fileBytes, sourceLabel);
}

} // namespace nuri::detail
