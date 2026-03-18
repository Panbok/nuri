#include "nuri/pch.h"

#include "nuri/resources/gltf_scene_importer.h"

#include "nuri/core/profiling.h"
#include "nuri/math/light.h"

namespace nuri {
namespace {

constexpr uint32_t kGlbMagic = 0x46546C67u;
constexpr uint32_t kGlbVersion2 = 2u;
constexpr uint32_t kGlbChunkTypeJson = 0x4E4F534Au;
using YyJsonDocPtr = std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)>;
using YyJsonDocResult = Result<YyJsonDocPtr, std::string>;

struct ParsedLightDef {
  LightDesc desc{};
  std::string name{};
};

struct ParsedNode {
  std::string name{};
  glm::mat4 localMatrix{1.0f};
  std::vector<uint32_t> children{};
  std::optional<uint32_t> lightIndex{};
};

[[nodiscard]] bool hasExtension(const std::filesystem::path &path,
                                std::string_view extension) {
  return path.has_extension() && path.extension().string() == extension;
}

[[nodiscard]] bool isGltfJsonAssetPath(const std::filesystem::path &path) {
  return hasExtension(path, ".gltf") || hasExtension(path, ".glb");
}

[[nodiscard]] std::string_view readJsonStringView(yyjson_val *value) {
  if (!yyjson_is_str(value)) {
    return {};
  }
  const char *raw = yyjson_get_str(value);
  return raw != nullptr ? std::string_view(raw) : std::string_view{};
}

[[nodiscard]] std::string_view readJsonStringView(yyjson_val *object,
                                                  const char *key) {
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

[[nodiscard]] glm::quat sanitizeRotation(const glm::quat &rotation) {
  const float length = glm::length(rotation);
  if (!std::isfinite(length) || length <= kLightBasisMinLength) {
    return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
  }
  return glm::normalize(rotation);
}

[[nodiscard]] glm::mat4 parseNodeLocalMatrix(yyjson_val *nodeValue) {
  yyjson_val *matrixValue = yyjson_obj_get(nodeValue, "matrix");
  glm::mat4 matrix(1.0f);
  if (tryReadJsonMat4(matrixValue, matrix)) {
    return matrix;
  }

  glm::vec3 translation(0.0f);
  glm::vec4 rotationValues(0.0f, 0.0f, 0.0f, 1.0f);
  glm::vec3 scale(1.0f);
  (void)tryReadJsonVec3(yyjson_obj_get(nodeValue, "translation"), translation);
  (void)tryReadJsonVec4(yyjson_obj_get(nodeValue, "rotation"), rotationValues);
  (void)tryReadJsonVec3(yyjson_obj_get(nodeValue, "scale"), scale);

  const glm::quat rotation = sanitizeRotation(glm::quat(
      rotationValues.w, rotationValues.x, rotationValues.y, rotationValues.z));
  return glm::translate(glm::mat4(1.0f), translation) *
         glm::mat4_cast(rotation) * glm::scale(glm::mat4(1.0f), scale);
}

[[nodiscard]] std::string makeFallbackLightName(uint32_t lightIndex) {
  return "light_" + std::to_string(lightIndex);
}

[[nodiscard]] Result<LightType, std::string>
parseLightType(std::string_view typeName) {
  if (typeName == "directional") {
    return Result<LightType, std::string>::makeResult(LightType::Directional);
  }
  if (typeName == "point") {
    return Result<LightType, std::string>::makeResult(LightType::Point);
  }
  if (typeName == "spot") {
    return Result<LightType, std::string>::makeResult(LightType::Spot);
  }
  return Result<LightType, std::string>::makeError(
      "glTF punctual light type is invalid");
}

[[nodiscard]] float sanitizeNonNegative(float value, float fallback = 0.0f) {
  if (!std::isfinite(value)) {
    return fallback;
  }
  return std::max(value, 0.0f);
}

[[nodiscard]] LightDesc sanitizeImportedLightDesc(const LightDesc &desc) {
  LightDesc sanitized = desc;
  sanitized.rotation = sanitizeRotation(desc.rotation);
  sanitized.color = glm::max(desc.color, glm::vec3(0.0f));
  sanitized.intensity = sanitizeNonNegative(desc.intensity, 1.0f);
  sanitized.range = sanitizeNonNegative(desc.range, 0.0f);
  sanitized.innerConeAngleRadians =
      sanitizeNonNegative(desc.innerConeAngleRadians, 0.0f);
  sanitized.outerConeAngleRadians =
      sanitizeNonNegative(desc.outerConeAngleRadians, glm::quarter_pi<float>());

  if (sanitized.type == LightType::Directional) {
    sanitized.range = 0.0f;
    sanitized.innerConeAngleRadians = 0.0f;
    sanitized.outerConeAngleRadians = 0.0f;
  } else if (sanitized.type == LightType::Point) {
    sanitized.innerConeAngleRadians = 0.0f;
    sanitized.outerConeAngleRadians = 0.0f;
  } else {
    sanitized.outerConeAngleRadians = std::clamp(
        sanitized.outerConeAngleRadians, 0.0f, glm::half_pi<float>() - 1.0e-4f);
    sanitized.innerConeAngleRadians = std::clamp(
        sanitized.innerConeAngleRadians, 0.0f, sanitized.outerConeAngleRadians);
  }

  return sanitized;
}

[[nodiscard]] YyJsonDocResult parseJsonDocument(std::string_view jsonText) {
  std::string mutableJson(jsonText);
  yyjson_read_err parseError{};
  YyJsonDocPtr doc(yyjson_read_opts(mutableJson.data(), mutableJson.size(), 0,
                                    nullptr, &parseError),
                   &yyjson_doc_free);
  if (doc == nullptr) {
    return YyJsonDocResult::makeError(
        "Failed to parse glTF JSON near byte " +
        std::to_string(static_cast<size_t>(parseError.pos)));
  }
  return YyJsonDocResult::makeResult(std::move(doc));
}

[[nodiscard]] YyJsonDocResult
loadGltfJsonDocument(const std::filesystem::path &path) {
  const auto readU32 = [](std::span<const uint8_t> bytes, size_t offset,
                          uint32_t &out) -> bool {
    if (offset + sizeof(uint32_t) > bytes.size()) {
      return false;
    }
    out = static_cast<uint32_t>(bytes[offset]) |
          (static_cast<uint32_t>(bytes[offset + 1u]) << 8u) |
          (static_cast<uint32_t>(bytes[offset + 2u]) << 16u) |
          (static_cast<uint32_t>(bytes[offset + 3u]) << 24u);
    return true;
  };

  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    return YyJsonDocResult::makeError("Failed to open glTF scene source");
  }

  if (hasExtension(path, ".gltf")) {
    std::ostringstream jsonStream;
    jsonStream << file.rdbuf();
    return parseJsonDocument(jsonStream.str());
  }

  if (!hasExtension(path, ".glb")) {
    return YyJsonDocResult::makeError(
        "Unsupported glTF scene-light file extension");
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
  if (bytes.size() < 20u) {
    return YyJsonDocResult::makeError(".glb file is too small");
  }

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

  return YyJsonDocResult::makeError(".glb JSON chunk is missing");
}

Result<std::vector<ParsedLightDef>, std::string>
parseLightDefinitions(yyjson_val *root) {
  yyjson_val *extensionsValue = yyjson_obj_get(root, "extensions");
  if (!yyjson_is_obj(extensionsValue)) {
    return Result<std::vector<ParsedLightDef>, std::string>::makeResult({});
  }

  yyjson_val *lightExtension =
      yyjson_obj_get(extensionsValue, "KHR_lights_punctual");
  if (!yyjson_is_obj(lightExtension)) {
    return Result<std::vector<ParsedLightDef>, std::string>::makeResult({});
  }

  yyjson_val *lightsValue = yyjson_obj_get(lightExtension, "lights");
  if (!yyjson_is_arr(lightsValue)) {
    return Result<std::vector<ParsedLightDef>, std::string>::makeResult({});
  }

  std::vector<ParsedLightDef> lights;
  lights.reserve(yyjson_arr_size(lightsValue));
  yyjson_arr_iter iter;
  yyjson_arr_iter_init(lightsValue, &iter);
  yyjson_val *lightValue = nullptr;
  uint32_t lightIndex = 0u;
  while ((lightValue = yyjson_arr_iter_next(&iter)) != nullptr) {
    if (!yyjson_is_obj(lightValue)) {
      return Result<std::vector<ParsedLightDef>, std::string>::makeError(
          "glTF light entry is not an object");
    }

    ParsedLightDef parsed{};
    auto lightTypeResult =
        parseLightType(readJsonStringView(lightValue, "type"));
    if (lightTypeResult.hasError()) {
      return Result<std::vector<ParsedLightDef>, std::string>::makeError(
          lightTypeResult.error());
    }
    parsed.desc.type = lightTypeResult.value();
    parsed.desc.color = glm::vec3(1.0f);
    parsed.desc.intensity = 1.0f;
    parsed.desc.range = 0.0f;
    parsed.desc.innerConeAngleRadians = 0.0f;
    parsed.desc.outerConeAngleRadians = glm::quarter_pi<float>();
    parsed.desc.enabled = true;

    const std::string_view lightName = readJsonStringView(lightValue, "name");
    parsed.name = lightName.empty() ? makeFallbackLightName(lightIndex)
                                    : std::string(lightName);

    glm::vec3 color(1.0f);
    if (tryReadJsonVec3(yyjson_obj_get(lightValue, "color"), color)) {
      parsed.desc.color = color;
    }

    float intensity = 1.0f;
    if (tryReadJsonFloat(yyjson_obj_get(lightValue, "intensity"), intensity)) {
      parsed.desc.intensity = intensity;
    }

    float range = 0.0f;
    if (tryReadJsonFloat(yyjson_obj_get(lightValue, "range"), range)) {
      parsed.desc.range = range;
    }

    if (parsed.desc.type == LightType::Spot) {
      yyjson_val *spotValue = yyjson_obj_get(lightValue, "spot");
      if (yyjson_is_obj(spotValue)) {
        float innerCone = 0.0f;
        if (tryReadJsonFloat(yyjson_obj_get(spotValue, "innerConeAngle"),
                             innerCone)) {
          parsed.desc.innerConeAngleRadians = innerCone;
        }

        float outerCone = glm::quarter_pi<float>();
        if (tryReadJsonFloat(yyjson_obj_get(spotValue, "outerConeAngle"),
                             outerCone)) {
          parsed.desc.outerConeAngleRadians = outerCone;
        }
      }
    }

    parsed.desc = sanitizeImportedLightDesc(parsed.desc);
    lights.push_back(std::move(parsed));
    ++lightIndex;
  }

  return Result<std::vector<ParsedLightDef>, std::string>::makeResult(
      std::move(lights));
}

Result<std::vector<ParsedNode>, std::string> parseNodes(yyjson_val *root) {
  yyjson_val *nodesValue = yyjson_obj_get(root, "nodes");
  if (!yyjson_is_arr(nodesValue)) {
    return Result<std::vector<ParsedNode>, std::string>::makeResult({});
  }

  std::vector<ParsedNode> nodes;
  nodes.reserve(yyjson_arr_size(nodesValue));
  yyjson_arr_iter iter;
  yyjson_arr_iter_init(nodesValue, &iter);
  yyjson_val *nodeValue = nullptr;
  while ((nodeValue = yyjson_arr_iter_next(&iter)) != nullptr) {
    if (!yyjson_is_obj(nodeValue)) {
      return Result<std::vector<ParsedNode>, std::string>::makeError(
          "glTF node entry is not an object");
    }

    ParsedNode parsed{};
    const std::string_view nodeName = readJsonStringView(nodeValue, "name");
    if (!nodeName.empty()) {
      parsed.name.assign(nodeName);
    }
    parsed.localMatrix = parseNodeLocalMatrix(nodeValue);

    yyjson_val *childrenValue = yyjson_obj_get(nodeValue, "children");
    if (yyjson_is_arr(childrenValue)) {
      parsed.children.reserve(yyjson_arr_size(childrenValue));
      yyjson_arr_iter childrenIter;
      yyjson_arr_iter_init(childrenValue, &childrenIter);
      yyjson_val *childValue = nullptr;
      while ((childValue = yyjson_arr_iter_next(&childrenIter)) != nullptr) {
        uint32_t childIndex = 0u;
        if (!tryReadJsonUint32(childValue, childIndex)) {
          return Result<std::vector<ParsedNode>, std::string>::makeError(
              "glTF node child index is invalid");
        }
        parsed.children.push_back(childIndex);
      }
    }

    yyjson_val *extensionsValue = yyjson_obj_get(nodeValue, "extensions");
    if (yyjson_is_obj(extensionsValue)) {
      yyjson_val *lightExtension =
          yyjson_obj_get(extensionsValue, "KHR_lights_punctual");
      if (yyjson_is_obj(lightExtension)) {
        uint32_t lightIndex = 0u;
        if (tryReadJsonUint32(yyjson_obj_get(lightExtension, "light"),
                              lightIndex)) {
          parsed.lightIndex = lightIndex;
        }
      }
    }

    nodes.push_back(std::move(parsed));
  }

  return Result<std::vector<ParsedNode>, std::string>::makeResult(
      std::move(nodes));
}

Result<std::vector<uint32_t>, std::string>
resolveSceneRootNodes(yyjson_val *root, size_t nodeCount) {
  if (nodeCount == 0u) {
    return Result<std::vector<uint32_t>, std::string>::makeResult({});
  }

  yyjson_val *scenesValue = yyjson_obj_get(root, "scenes");
  if (yyjson_is_arr(scenesValue) && yyjson_arr_size(scenesValue) > 0u) {
    uint32_t sceneIndex = 0u;
    if (!tryReadJsonUint32(yyjson_obj_get(root, "scene"), sceneIndex)) {
      sceneIndex = 0u;
    }
    if (sceneIndex >= yyjson_arr_size(scenesValue)) {
      return Result<std::vector<uint32_t>, std::string>::makeError(
          "glTF selected scene index is out of range");
    }

    yyjson_val *sceneValue = yyjson_arr_get(scenesValue, sceneIndex);
    if (!yyjson_is_obj(sceneValue)) {
      return Result<std::vector<uint32_t>, std::string>::makeError(
          "glTF scene entry is not an object");
    }

    yyjson_val *sceneNodesValue = yyjson_obj_get(sceneValue, "nodes");
    if (!yyjson_is_arr(sceneNodesValue)) {
      return Result<std::vector<uint32_t>, std::string>::makeResult({});
    }

    std::vector<uint32_t> roots;
    roots.reserve(yyjson_arr_size(sceneNodesValue));
    yyjson_arr_iter iter;
    yyjson_arr_iter_init(sceneNodesValue, &iter);
    yyjson_val *rootNodeValue = nullptr;
    while ((rootNodeValue = yyjson_arr_iter_next(&iter)) != nullptr) {
      uint32_t rootNodeIndex = 0u;
      if (!tryReadJsonUint32(rootNodeValue, rootNodeIndex)) {
        return Result<std::vector<uint32_t>, std::string>::makeError(
            "glTF scene root node index is invalid");
      }
      roots.push_back(rootNodeIndex);
    }
    return Result<std::vector<uint32_t>, std::string>::makeResult(
        std::move(roots));
  }

  std::vector<uint32_t> roots;
  std::vector<uint8_t> referenced(nodeCount, 0u);
  yyjson_val *nodesValue = yyjson_obj_get(root, "nodes");
  if (yyjson_is_arr(nodesValue)) {
    for (size_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex) {
      yyjson_val *nodeValue = yyjson_arr_get(nodesValue, nodeIndex);
      if (!yyjson_is_obj(nodeValue)) {
        continue;
      }
      yyjson_val *childrenValue = yyjson_obj_get(nodeValue, "children");
      if (!yyjson_is_arr(childrenValue)) {
        continue;
      }
      yyjson_arr_iter iter;
      yyjson_arr_iter_init(childrenValue, &iter);
      yyjson_val *childValue = nullptr;
      while ((childValue = yyjson_arr_iter_next(&iter)) != nullptr) {
        uint32_t childIndex = 0u;
        if (!tryReadJsonUint32(childValue, childIndex) ||
            childIndex >= nodeCount) {
          return Result<std::vector<uint32_t>, std::string>::makeError(
              "glTF node child index is invalid");
        }
        referenced[childIndex] = 1u;
      }
    }
  }

  roots.reserve(nodeCount);
  for (uint32_t nodeIndex = 0u; nodeIndex < nodeCount; ++nodeIndex) {
    if (referenced[nodeIndex] == 0u) {
      roots.push_back(nodeIndex);
    }
  }
  if (roots.empty()) {
    for (uint32_t nodeIndex = 0u; nodeIndex < nodeCount; ++nodeIndex) {
      roots.push_back(nodeIndex);
    }
  }
  return Result<std::vector<uint32_t>, std::string>::makeResult(
      std::move(roots));
}

} // namespace

Result<ImportedLightSet, std::string>
GltfSceneImporter::loadLightsFromFile(std::string_view path) {
  NURI_PROFILER_FUNCTION_COLOR(NURI_PROFILER_COLOR_CREATE);
  const std::filesystem::path assetPath(path);
  if (!isGltfJsonAssetPath(assetPath)) {
    return Result<ImportedLightSet, std::string>::makeResult({});
  }

  auto docResult = loadGltfJsonDocument(assetPath);
  if (docResult.hasError()) {
    return Result<ImportedLightSet, std::string>::makeError(docResult.error());
  }
  YyJsonDocPtr doc = std::move(docResult.value());
  yyjson_val *root = yyjson_doc_get_root(doc.get());
  if (!yyjson_is_obj(root)) {
    return Result<ImportedLightSet, std::string>::makeError(
        "glTF root is not a JSON object");
  }

  auto lightsResult = parseLightDefinitions(root);
  if (lightsResult.hasError()) {
    return Result<ImportedLightSet, std::string>::makeError(
        lightsResult.error());
  }
  std::vector<ParsedLightDef> lights = std::move(lightsResult.value());
  if (lights.empty()) {
    return Result<ImportedLightSet, std::string>::makeResult({});
  }

  auto nodesResult = parseNodes(root);
  if (nodesResult.hasError()) {
    return Result<ImportedLightSet, std::string>::makeError(
        nodesResult.error());
  }
  std::vector<ParsedNode> nodes = std::move(nodesResult.value());
  if (nodes.empty()) {
    return Result<ImportedLightSet, std::string>::makeResult({});
  }

  for (const ParsedNode &node : nodes) {
    for (uint32_t childIndex : node.children) {
      if (childIndex >= nodes.size()) {
        return Result<ImportedLightSet, std::string>::makeError(
            "glTF node child index is out of range");
      }
    }
    if (node.lightIndex.has_value() && *node.lightIndex >= lights.size()) {
      return Result<ImportedLightSet, std::string>::makeError(
          "glTF punctual light index is out of range");
    }
  }

  auto rootsResult = resolveSceneRootNodes(root, nodes.size());
  if (rootsResult.hasError()) {
    return Result<ImportedLightSet, std::string>::makeError(
        rootsResult.error());
  }
  std::vector<uint32_t> roots = std::move(rootsResult.value());

  ImportedLightSet importedLights;
  std::vector<uint8_t> active(nodes.size(), 0u);

  std::function<std::optional<std::string>(uint32_t, const glm::mat4 &)>
      traverseNode;
  traverseNode =
      [&](uint32_t nodeIndex,
          const glm::mat4 &parentWorld) -> std::optional<std::string> {
    if (nodeIndex >= nodes.size()) {
      return std::string("glTF scene root node index is out of range");
    }
    if (active[nodeIndex] != 0u) {
      return std::string("glTF node hierarchy contains a cycle");
    }

    active[nodeIndex] = 1u;
    const ParsedNode &node = nodes[nodeIndex];
    const glm::mat4 worldMatrix = parentWorld * node.localMatrix;

    if (node.lightIndex.has_value()) {
      const ParsedLightDef &parsedLight = lights[*node.lightIndex];
      ImportedLightInfo imported{};
      imported.desc = parsedLight.desc;
      imported.desc.position = glm::vec3(worldMatrix[3]);
      imported.desc.rotation = rotationFromMatrixOrIdentity(worldMatrix);

      if (!node.name.empty()) {
        imported.desc.name = node.name;
        imported.sourceName = node.name;
      } else if (!parsedLight.name.empty()) {
        imported.desc.name = parsedLight.name;
        imported.sourceName = parsedLight.name;
      } else {
        imported.desc.name = makeFallbackLightName(*node.lightIndex);
        imported.sourceName = imported.desc.name;
      }
      imported.sourceNodeIndex = static_cast<int32_t>(nodeIndex);
      imported.desc = sanitizeImportedLightDesc(imported.desc);
      importedLights.push_back(std::move(imported));
    }

    for (uint32_t childIndex : node.children) {
      std::optional<std::string> childError =
          traverseNode(childIndex, worldMatrix);
      if (childError.has_value()) {
        active[nodeIndex] = 0u;
        return childError;
      }
    }

    active[nodeIndex] = 0u;
    return std::nullopt;
  };

  for (uint32_t rootNodeIndex : roots) {
    std::optional<std::string> traverseError =
        traverseNode(rootNodeIndex, glm::mat4(1.0f));
    if (traverseError.has_value()) {
      return Result<ImportedLightSet, std::string>::makeError(
          std::move(*traverseError));
    }
  }

  return Result<ImportedLightSet, std::string>::makeResult(
      std::move(importedLights));
}

} // namespace nuri
