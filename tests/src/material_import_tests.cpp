#include "tests_pch.h"

#include "nuri/resources/gpu/material.h"
#include "nuri/resources/gpu/resource_keys.h"
#include "nuri/resources/mesh_importer.h"

#include <chrono>
#include <cstring>
#include <optional>
#include <thread>

namespace {
constexpr std::string_view kIorTestMaterialName = "Ior Test";
constexpr std::string_view kSpecularTestMaterialName = "Specular Test";
constexpr std::string_view kSpecGlossTestMaterialName = "SpecGloss Test";
constexpr std::string_view kEmissiveStrengthTestMaterialName =
    "Emissive Strength Test";
constexpr std::string_view kIorTestBufferFileName = "scene.bin";
constexpr uint32_t kGlbMagic = 0x46546C67u;
constexpr uint32_t kGlbVersion2 = 2u;
constexpr uint32_t kGlbChunkTypeJson = 0x4E4F534Au;
constexpr uint32_t kGlbChunkTypeBin = 0x004E4942u;

struct ScopedTempDir {
  explicit ScopedTempDir(std::string_view prefix) {
    const auto uniqueId =
        static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count()) ^
        static_cast<uint64_t>(
            std::hash<std::thread::id>{}(std::this_thread::get_id()));
    path = std::filesystem::current_path() / ".tmp_tests" /
           (std::string(prefix) + "_" + std::to_string(uniqueId));
    std::filesystem::create_directories(path);
  }

  ~ScopedTempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }

  std::filesystem::path path;
};

std::vector<uint8_t> minimalTriangleBuffer() {
  constexpr std::array<float, 9> kPositions = {
      0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
  };
  constexpr std::array<uint16_t, 3> kIndices = {0u, 1u, 2u};

  std::vector<uint8_t> bytes(sizeof(kPositions) + sizeof(kIndices));
  std::memcpy(bytes.data(), kPositions.data(), sizeof(kPositions));
  std::memcpy(bytes.data() + sizeof(kPositions), kIndices.data(),
              sizeof(kIndices));
  return bytes;
}

std::string makeMinimalIorTestJson(float ior, bool externalBuffer) {
  std::ostringstream json;
  json << "{\n"
       << "  \"asset\": {\"version\": \"2.0\"},\n"
       << "  \"extensionsUsed\": [\"KHR_materials_ior\"],\n"
       << "  \"scene\": 0,\n"
       << "  \"scenes\": [{\"nodes\": [0]}],\n"
       << "  \"nodes\": [{\"mesh\": 0}],\n"
       << "  \"materials\": [{\n"
       << "    \"name\": \"" << kIorTestMaterialName << "\",\n"
       << "    \"pbrMetallicRoughness\": {\n"
       << "      \"baseColorFactor\": [1, 1, 1, 1],\n"
       << "      \"metallicFactor\": 0,\n"
       << "      \"roughnessFactor\": 0.5\n"
       << "    },\n"
       << "    \"extensions\": {\n"
       << "      \"KHR_materials_ior\": {\"ior\": " << ior << "}\n"
       << "    }\n"
       << "  }],\n"
       << "  \"meshes\": [{\"primitives\": [{\n"
       << "    \"attributes\": {\"POSITION\": 0},\n"
       << "    \"indices\": 1,\n"
       << "    \"material\": 0\n"
       << "  }]}],\n"
       << "  \"accessors\": [\n"
       << "    {\n"
       << "      \"bufferView\": 0,\n"
       << "      \"componentType\": 5126,\n"
       << "      \"count\": 3,\n"
       << "      \"type\": \"VEC3\",\n"
       << "      \"max\": [1, 1, 0],\n"
       << "      \"min\": [0, 0, 0]\n"
       << "    },\n"
       << "    {\n"
       << "      \"bufferView\": 1,\n"
       << "      \"componentType\": 5123,\n"
       << "      \"count\": 3,\n"
       << "      \"type\": \"SCALAR\",\n"
       << "      \"max\": [2],\n"
       << "      \"min\": [0]\n"
       << "    }\n"
       << "  ],\n"
       << "  \"bufferViews\": [\n"
       << "    {\"buffer\": 0, \"byteOffset\": 0, \"byteLength\": 36, "
          "\"target\": 34962},\n"
       << "    {\"buffer\": 0, \"byteOffset\": 36, \"byteLength\": 6, "
          "\"target\": 34963}\n"
       << "  ],\n"
       << "  \"buffers\": [{\"byteLength\": 42";
  if (externalBuffer) {
    json << ", \"uri\": \"" << kIorTestBufferFileName << "\"";
  }
  json << "}]\n"
       << "}\n";
  return json.str();
}

std::string
makeMinimalSpecularTestJson(float specularFactor,
                            const std::array<float, 3> &specularColor,
                            bool includeTextures, bool externalBuffer) {
  std::ostringstream json;
  json << "{\n"
       << "  \"asset\": {\"version\": \"2.0\"},\n"
       << "  \"extensionsUsed\": [\"KHR_materials_specular\", "
          "\"KHR_texture_transform\"],\n"
       << "  \"scene\": 0,\n"
       << "  \"scenes\": [{\"nodes\": [0]}],\n"
       << "  \"nodes\": [{\"mesh\": 0}],\n"
       << "  \"materials\": [{\n"
       << "    \"name\": \"" << kSpecularTestMaterialName << "\",\n"
       << "    \"pbrMetallicRoughness\": {\n"
       << "      \"baseColorFactor\": [1, 1, 1, 1],\n"
       << "      \"metallicFactor\": 0,\n"
       << "      \"roughnessFactor\": 0.5\n"
       << "    },\n"
       << "    \"extensions\": {\n"
       << "      \"KHR_materials_specular\": {\n"
       << "        \"specularFactor\": " << specularFactor << ",\n"
       << "        \"specularColorFactor\": [" << specularColor[0] << ", "
       << specularColor[1] << ", " << specularColor[2] << "]";
  if (includeTextures) {
    json << ",\n"
         << "        \"specularTexture\": {\n"
         << "          \"index\": 0,\n"
         << "          \"texCoord\": 1,\n"
         << "          \"extensions\": {\n"
         << "            \"KHR_texture_transform\": {\n"
         << "              \"offset\": [0.25, 0.5],\n"
         << "              \"scale\": [2.0, 3.0],\n"
         << "              \"rotation\": 0.785398163,\n"
         << "              \"texCoord\": 0\n"
         << "            }\n"
         << "          }\n"
         << "        },\n"
         << "        \"specularColorTexture\": {\n"
         << "          \"index\": 1,\n"
         << "          \"texCoord\": 0,\n"
         << "          \"extensions\": {\n"
         << "            \"KHR_texture_transform\": {\n"
         << "              \"offset\": [-1.0, 1.5],\n"
         << "              \"scale\": [4.0, 5.0],\n"
         << "              \"rotation\": 0.25,\n"
         << "              \"texCoord\": 1\n"
         << "            }\n"
         << "          }\n"
         << "        }\n";
  } else {
    json << "\n";
  }
  json << "      }\n"
       << "    }\n"
       << "  }],\n";

  if (includeTextures) {
    json << "  \"samplers\": [\n"
         << "    {\"magFilter\": 9729, \"minFilter\": 9987, \"wrapS\": "
            "10497, \"wrapT\": 10497},\n"
         << "    {\"magFilter\": 9728, \"minFilter\": 9984, \"wrapS\": "
            "33648, \"wrapT\": 33071}\n"
         << "  ],\n"
         << "  \"images\": [\n"
         << "    {\"uri\": \"textures/specular_strength.png\"},\n"
         << "    {\"uri\": \"textures/specular_color.png\"}\n"
         << "  ],\n"
         << "  \"textures\": [\n"
         << "    {\"sampler\": 1, \"source\": 0},\n"
         << "    {\"sampler\": 0, \"source\": 1}\n"
         << "  ],\n";
  }

  json << "  \"meshes\": [{\"primitives\": [{\n"
       << "    \"attributes\": {\"POSITION\": 0},\n"
       << "    \"indices\": 1,\n"
       << "    \"material\": 0\n"
       << "  }]}],\n"
       << "  \"accessors\": [\n"
       << "    {\n"
       << "      \"bufferView\": 0,\n"
       << "      \"componentType\": 5126,\n"
       << "      \"count\": 3,\n"
       << "      \"type\": \"VEC3\",\n"
       << "      \"max\": [1, 1, 0],\n"
       << "      \"min\": [0, 0, 0]\n"
       << "    },\n"
       << "    {\n"
       << "      \"bufferView\": 1,\n"
       << "      \"componentType\": 5123,\n"
       << "      \"count\": 3,\n"
       << "      \"type\": \"SCALAR\",\n"
       << "      \"max\": [2],\n"
       << "      \"min\": [0]\n"
       << "    }\n"
       << "  ],\n"
       << "  \"bufferViews\": [\n"
       << "    {\"buffer\": 0, \"byteOffset\": 0, \"byteLength\": 36, "
          "\"target\": 34962},\n"
       << "    {\"buffer\": 0, \"byteOffset\": 36, \"byteLength\": 6, "
          "\"target\": 34963}\n"
       << "  ],\n"
       << "  \"buffers\": [{\"byteLength\": 42";
  if (externalBuffer) {
    json << ", \"uri\": \"" << kIorTestBufferFileName << "\"";
  }
  json << "}]\n"
       << "}\n";
  return json.str();
}

std::string
makeMinimalEmissiveStrengthTestJson(const std::array<float, 3> &emissiveFactor,
                                    std::optional<float> emissiveStrength,
                                    bool externalBuffer) {
  std::ostringstream json;
  json << "{\n"
       << "  \"asset\": {\"version\": \"2.0\"},\n";
  if (emissiveStrength.has_value()) {
    json << "  \"extensionsUsed\": [\"KHR_materials_emissive_strength\"],\n";
  }
  json << "  \"scene\": 0,\n"
       << "  \"scenes\": [{\"nodes\": [0]}],\n"
       << "  \"nodes\": [{\"mesh\": 0}],\n"
       << "  \"materials\": [{\n"
       << "    \"name\": \"" << kEmissiveStrengthTestMaterialName << "\",\n"
       << "    \"emissiveFactor\": [" << emissiveFactor[0] << ", "
       << emissiveFactor[1] << ", " << emissiveFactor[2] << "],\n"
       << "    \"pbrMetallicRoughness\": {\n"
       << "      \"baseColorFactor\": [1, 1, 1, 1],\n"
       << "      \"metallicFactor\": 0,\n"
       << "      \"roughnessFactor\": 0.5\n"
       << "    }";
  if (emissiveStrength.has_value()) {
    json << ",\n"
         << "    \"extensions\": {\n"
         << "      \"KHR_materials_emissive_strength\": {\n"
         << "        \"emissiveStrength\": " << *emissiveStrength << "\n"
         << "      }\n"
         << "    }\n";
  } else {
    json << "\n";
  }
  json << "  }],\n"
       << "  \"meshes\": [{\"primitives\": [{\n"
       << "    \"attributes\": {\"POSITION\": 0},\n"
       << "    \"indices\": 1,\n"
       << "    \"material\": 0\n"
       << "  }]}],\n"
       << "  \"accessors\": [\n"
       << "    {\n"
       << "      \"bufferView\": 0,\n"
       << "      \"componentType\": 5126,\n"
       << "      \"count\": 3,\n"
       << "      \"type\": \"VEC3\",\n"
       << "      \"max\": [1, 1, 0],\n"
       << "      \"min\": [0, 0, 0]\n"
       << "    },\n"
       << "    {\n"
       << "      \"bufferView\": 1,\n"
       << "      \"componentType\": 5123,\n"
       << "      \"count\": 3,\n"
       << "      \"type\": \"SCALAR\",\n"
       << "      \"max\": [2],\n"
       << "      \"min\": [0]\n"
       << "    }\n"
       << "  ],\n"
       << "  \"bufferViews\": [\n"
       << "    {\"buffer\": 0, \"byteOffset\": 0, \"byteLength\": 36, "
          "\"target\": 34962},\n"
       << "    {\"buffer\": 0, \"byteOffset\": 36, \"byteLength\": 6, "
          "\"target\": 34963}\n"
       << "  ],\n"
       << "  \"buffers\": [{\"byteLength\": 42";
  if (externalBuffer) {
    json << ", \"uri\": \"" << kIorTestBufferFileName << "\"";
  }
  json << "}]\n"
       << "}\n";
  return json.str();
}

std::string makeMinimalSpecGlossTestJson(bool includeTextures,
                                         bool includeSpecularExtension,
                                         bool externalBuffer) {
  std::ostringstream json;
  json << "{\n"
       << "  \"asset\": {\"version\": \"2.0\"},\n"
       << "  \"extensionsUsed\": [\"KHR_materials_pbrSpecularGlossiness\"";
  if (includeTextures) {
    json << ", \"KHR_texture_transform\"";
  }
  if (includeSpecularExtension) {
    json << ", \"KHR_materials_specular\"";
  }
  json << "],\n"
       << "  \"scene\": 0,\n"
       << "  \"scenes\": [{\"nodes\": [0]}],\n"
       << "  \"nodes\": [{\"mesh\": 0}],\n"
       << "  \"materials\": [{\n"
       << "    \"name\": \"" << kSpecGlossTestMaterialName << "\",\n"
       << "    \"pbrMetallicRoughness\": {\n"
       << "      \"baseColorFactor\": [0.1, 0.2, 0.3, 0.4],\n"
       << "      \"metallicFactor\": 0.9,\n"
       << "      \"roughnessFactor\": 0.25";
  if (includeTextures) {
    json << ",\n"
         << "      \"metallicRoughnessTexture\": {\"index\": 2}\n";
  } else {
    json << "\n";
  }
  json << "    },\n"
       << "    \"extensions\": {\n"
       << "      \"KHR_materials_pbrSpecularGlossiness\": {\n"
       << "        \"diffuseFactor\": [0.9, 0.8, 0.7, 0.6],\n"
       << "        \"specularFactor\": [1.5, 0.5, 0.25],\n"
       << "        \"glossinessFactor\": 0.35";
  if (includeTextures) {
    json << ",\n"
         << "        \"diffuseTexture\": {\n"
         << "          \"index\": 0,\n"
         << "          \"texCoord\": 1,\n"
         << "          \"extensions\": {\n"
         << "            \"KHR_texture_transform\": {\n"
         << "              \"offset\": [0.125, 0.25],\n"
         << "              \"scale\": [1.5, 2.5],\n"
         << "              \"rotation\": 0.5,\n"
         << "              \"texCoord\": 0\n"
         << "            }\n"
         << "          }\n"
         << "        },\n"
         << "        \"specularGlossinessTexture\": {\n"
         << "          \"index\": 1,\n"
         << "          \"texCoord\": 0,\n"
         << "          \"extensions\": {\n"
         << "            \"KHR_texture_transform\": {\n"
         << "              \"offset\": [-0.5, 0.75],\n"
         << "              \"scale\": [4.0, 5.0],\n"
         << "              \"rotation\": 0.125,\n"
         << "              \"texCoord\": 1\n"
         << "            }\n"
         << "          }\n"
         << "        }\n";
  } else {
    json << "\n";
  }
  json << "      }";
  if (includeSpecularExtension) {
    json << ",\n"
         << "      \"KHR_materials_specular\": {\n"
         << "        \"specularFactor\": 0.1,\n"
         << "        \"specularColorFactor\": [0.2, 0.3, 0.4],\n"
         << "        \"specularTexture\": {\"index\": 0},\n"
         << "        \"specularColorTexture\": {\"index\": 1}\n"
         << "      }\n";
  } else {
    json << "\n";
  }
  json << "    }\n"
       << "  }],\n";

  if (includeTextures) {
    json << "  \"samplers\": [\n"
         << "    {\"magFilter\": 9729, \"minFilter\": 9987, \"wrapS\": 10497, "
            "\"wrapT\": 10497},\n"
         << "    {\"magFilter\": 9728, \"minFilter\": 9984, \"wrapS\": 33648, "
            "\"wrapT\": 33071},\n"
         << "    {\"magFilter\": 9729, \"minFilter\": 9729, \"wrapS\": 10497, "
            "\"wrapT\": 10497}\n"
         << "  ],\n"
         << "  \"images\": [\n"
         << "    {\"uri\": \"textures/specgloss_diffuse.png\"},\n"
         << "    {\"uri\": \"textures/specgloss_rgba.png\"},\n"
         << "    {\"uri\": \"textures/mr.png\"}\n"
         << "  ],\n"
         << "  \"textures\": [\n"
         << "    {\"sampler\": 1, \"source\": 0},\n"
         << "    {\"sampler\": 0, \"source\": 1},\n"
         << "    {\"sampler\": 2, \"source\": 2}\n"
         << "  ],\n";
  }

  json << "  \"meshes\": [{\"primitives\": [{\n"
       << "    \"attributes\": {\"POSITION\": 0},\n"
       << "    \"indices\": 1,\n"
       << "    \"material\": 0\n"
       << "  }]}],\n"
       << "  \"accessors\": [\n"
       << "    {\n"
       << "      \"bufferView\": 0,\n"
       << "      \"componentType\": 5126,\n"
       << "      \"count\": 3,\n"
       << "      \"type\": \"VEC3\",\n"
       << "      \"max\": [1, 1, 0],\n"
       << "      \"min\": [0, 0, 0]\n"
       << "    },\n"
       << "    {\n"
       << "      \"bufferView\": 1,\n"
       << "      \"componentType\": 5123,\n"
       << "      \"count\": 3,\n"
       << "      \"type\": \"SCALAR\",\n"
       << "      \"max\": [2],\n"
       << "      \"min\": [0]\n"
       << "    }\n"
       << "  ],\n"
       << "  \"bufferViews\": [\n"
       << "    {\"buffer\": 0, \"byteOffset\": 0, \"byteLength\": 36, "
          "\"target\": 34962},\n"
       << "    {\"buffer\": 0, \"byteOffset\": 36, \"byteLength\": 6, "
          "\"target\": 34963}\n"
       << "  ],\n"
       << "  \"buffers\": [{\"byteLength\": 42";
  if (externalBuffer) {
    json << ", \"uri\": \"" << kIorTestBufferFileName << "\"";
  }
  json << "}]\n"
       << "}\n";
  return json.str();
}

void writeTextFile(const std::filesystem::path &path, std::string_view text) {
  std::ofstream file(path, std::ios::binary);
  ASSERT_TRUE(file.is_open()) << path.string();
  file.write(text.data(), static_cast<std::streamsize>(text.size()));
  ASSERT_TRUE(file.good()) << path.string();
}

void writeBinaryFile(const std::filesystem::path &path,
                     std::span<const uint8_t> bytes) {
  std::ofstream file(path, std::ios::binary);
  ASSERT_TRUE(file.is_open()) << path.string();
  if (!bytes.empty()) {
    file.write(reinterpret_cast<const char *>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  }
  ASSERT_TRUE(file.good()) << path.string();
}

void appendU32Le(std::vector<uint8_t> &out, uint32_t value) {
  out.push_back(static_cast<uint8_t>(value & 0xffu));
  out.push_back(static_cast<uint8_t>((value >> 8u) & 0xffu));
  out.push_back(static_cast<uint8_t>((value >> 16u) & 0xffu));
  out.push_back(static_cast<uint8_t>((value >> 24u) & 0xffu));
}

std::vector<uint8_t> padTo4(std::vector<uint8_t> bytes, uint8_t padByte) {
  while ((bytes.size() % 4u) != 0u) {
    bytes.push_back(padByte);
  }
  return bytes;
}

std::filesystem::path writeMinimalGlb(const ScopedTempDir &dir,
                                      std::string_view fileName,
                                      std::string_view jsonText) {
  const std::filesystem::path glbPath = dir.path / std::string(fileName);
  std::vector<uint8_t> jsonBytes(jsonText.begin(), jsonText.end());
  jsonBytes = padTo4(std::move(jsonBytes), 0x20u);
  std::vector<uint8_t> binBytes = padTo4(minimalTriangleBuffer(), 0u);

  std::vector<uint8_t> glb;
  const uint32_t totalLength =
      static_cast<uint32_t>(12u + 8u + jsonBytes.size() + 8u + binBytes.size());
  appendU32Le(glb, kGlbMagic);
  appendU32Le(glb, kGlbVersion2);
  appendU32Le(glb, totalLength);
  appendU32Le(glb, static_cast<uint32_t>(jsonBytes.size()));
  appendU32Le(glb, kGlbChunkTypeJson);
  glb.insert(glb.end(), jsonBytes.begin(), jsonBytes.end());
  appendU32Le(glb, static_cast<uint32_t>(binBytes.size()));
  appendU32Le(glb, kGlbChunkTypeBin);
  glb.insert(glb.end(), binBytes.begin(), binBytes.end());

  writeBinaryFile(glbPath, glb);
  return glbPath;
}

std::filesystem::path writeMinimalIorTestGltf(const ScopedTempDir &dir,
                                              float ior) {
  const std::filesystem::path gltfPath = dir.path / "ior_test.gltf";
  writeTextFile(gltfPath, makeMinimalIorTestJson(ior, true));
  const std::vector<uint8_t> buffer = minimalTriangleBuffer();
  writeBinaryFile(dir.path / kIorTestBufferFileName, buffer);
  return gltfPath;
}

std::filesystem::path writeMinimalIorTestGlb(const ScopedTempDir &dir,
                                             float ior) {
  const std::string jsonText = makeMinimalIorTestJson(ior, false);
  return writeMinimalGlb(dir, "ior_test.glb", jsonText);
}

std::filesystem::path
writeMinimalSpecularTestGltf(const ScopedTempDir &dir, float specularFactor,
                             const std::array<float, 3> &specularColor,
                             bool includeTextures) {
  const std::filesystem::path gltfPath = dir.path / "specular_test.gltf";
  writeTextFile(gltfPath,
                makeMinimalSpecularTestJson(specularFactor, specularColor,
                                            includeTextures, true));
  writeBinaryFile(dir.path / kIorTestBufferFileName, minimalTriangleBuffer());
  return gltfPath;
}

std::filesystem::path
writeMinimalEmissiveStrengthTestGltf(const ScopedTempDir &dir,
                                     const std::array<float, 3> &emissiveFactor,
                                     std::optional<float> emissiveStrength) {
  const std::filesystem::path gltfPath =
      dir.path / "emissive_strength_test.gltf";
  writeTextFile(gltfPath, makeMinimalEmissiveStrengthTestJson(
                              emissiveFactor, emissiveStrength, true));
  writeBinaryFile(dir.path / kIorTestBufferFileName, minimalTriangleBuffer());
  return gltfPath;
}

std::filesystem::path
writeMinimalEmissiveStrengthTestGlb(const ScopedTempDir &dir,
                                    const std::array<float, 3> &emissiveFactor,
                                    std::optional<float> emissiveStrength) {
  const std::string jsonText = makeMinimalEmissiveStrengthTestJson(
      emissiveFactor, emissiveStrength, false);
  return writeMinimalGlb(dir, "emissive_strength_test.glb", jsonText);
}

std::filesystem::path
writeMinimalSpecGlossTestGltf(const ScopedTempDir &dir, bool includeTextures,
                              bool includeSpecularExtension) {
  const std::filesystem::path gltfPath = dir.path / "specgloss_test.gltf";
  writeTextFile(gltfPath, makeMinimalSpecGlossTestJson(
                              includeTextures, includeSpecularExtension, true));
  writeBinaryFile(dir.path / kIorTestBufferFileName, minimalTriangleBuffer());
  return gltfPath;
}

std::string modelPath(std::string_view relativePath) {
  const std::filesystem::path root(PROJECT_SOURCE_DIR);
  return (root / "assets" / "models" / std::filesystem::path(relativePath))
      .string();
}

const nuri::ImportedMaterialInfo *
findMaterialByName(const nuri::ImportedMaterialSet &set,
                   std::string_view name) {
  const auto it =
      std::find_if(set.materials.begin(), set.materials.end(),
                   [name](const nuri::ImportedMaterialInfo &material) {
                     return material.name == name;
                   });
  return (it == set.materials.end()) ? nullptr : &(*it);
}

const nuri::ImportedMaterialInfo *
loadSyntheticSpecGlossMaterial(const ScopedTempDir &dir, bool includeTextures,
                               bool includeSpecularExtension,
                               nuri::ImportedMaterialSet &outSet) {
  auto result = nuri::MeshImporter::loadMaterialInfoFromFile(
      writeMinimalSpecGlossTestGltf(dir, includeTextures,
                                    includeSpecularExtension)
          .string());
  EXPECT_FALSE(result.hasError()) << result.error();
  if (result.hasError()) {
    return nullptr;
  }
  outSet = std::move(result.value());
  return findMaterialByName(outSet, kSpecGlossTestMaterialName);
}

TEST(MaterialImportTests, ClearcoatWickerOverlayImportsClearcoatData) {
  auto result = nuri::MeshImporter::loadMaterialInfoFromFile(
      modelPath("ClearcoatWicker/ClearcoatWicker.gltf"));
  ASSERT_FALSE(result.hasError()) << result.error();

  const nuri::ImportedMaterialSet &set = result.value();
  ASSERT_FALSE(set.materials.empty());

  const auto it = std::find_if(set.materials.begin(), set.materials.end(),
                               [](const nuri::ImportedMaterialInfo &material) {
                                 return material.clearcoatFactor > 0.0f ||
                                        !material.clearcoatNormal.path.empty();
                               });
  ASSERT_NE(it, set.materials.end());

  const nuri::ImportedMaterialInfo &material = *it;
  EXPECT_FLOAT_EQ(material.clearcoatFactor, 1.0f);
  EXPECT_FLOAT_EQ(material.clearcoatRoughnessFactor, 0.1f);
  EXPECT_FLOAT_EQ(material.clearcoatNormalScale, 1.0f);
  EXPECT_TRUE(material.clearcoat.path.empty());
  EXPECT_TRUE(material.clearcoatRoughness.path.empty());
  EXPECT_FALSE(material.clearcoatNormal.path.empty());
  EXPECT_TRUE(
      std::filesystem::path(material.clearcoatNormal.path).is_absolute());
  EXPECT_EQ(std::filesystem::path(material.clearcoatNormal.path).filename(),
            std::filesystem::path("clearcoat_normal.png"));

  EXPECT_FALSE(material.baseColor.path.empty());
  EXPECT_FALSE(material.normal.path.empty());
  EXPECT_FALSE(material.metallicRoughness.path.empty());
}

TEST(MaterialImportTests, DamagedHelmetLeavesClearcoatDisabled) {
  auto result = nuri::MeshImporter::loadMaterialInfoFromFile(
      modelPath("DamagedHelmet/DamagedHelmet.gltf"));
  ASSERT_FALSE(result.hasError()) << result.error();

  const nuri::ImportedMaterialSet &set = result.value();
  ASSERT_FALSE(set.materials.empty());

  for (size_t i = 0; i < set.materials.size(); ++i) {
    const nuri::ImportedMaterialInfo &material = set.materials[i];
    SCOPED_TRACE(::testing::Message()
                 << "material[" << i << "] name=\"" << material.name << "\"");
    EXPECT_FLOAT_EQ(material.clearcoatFactor, 0.0f);
    EXPECT_FLOAT_EQ(material.clearcoatRoughnessFactor, 0.0f);
    EXPECT_FLOAT_EQ(material.clearcoatNormalScale, 1.0f);
    EXPECT_TRUE(material.clearcoat.path.empty());
    EXPECT_TRUE(material.clearcoatRoughness.path.empty());
    EXPECT_TRUE(material.clearcoatNormal.path.empty());
  }

  const auto texturedIt =
      std::find_if(set.materials.begin(), set.materials.end(),
                   [](const nuri::ImportedMaterialInfo &material) {
                     return !material.baseColor.path.empty();
                   });
  ASSERT_NE(texturedIt, set.materials.end());
  EXPECT_FALSE(texturedIt->normal.path.empty());
  EXPECT_FALSE(texturedIt->metallicRoughness.path.empty());
}

TEST(MaterialImportTests, SheenChairImportsSheenFactorsAndTextures) {
  auto result = nuri::MeshImporter::loadMaterialInfoFromFile(
      modelPath("SheenChair/SheenChair.gltf"));
  ASSERT_FALSE(result.hasError()) << result.error();

  const nuri::ImportedMaterialSet &set = result.value();
  ASSERT_FALSE(set.materials.empty());

  const nuri::ImportedMaterialInfo *mangoVelvet =
      findMaterialByName(set, "fabric Mystere Mango Velvet");
  ASSERT_NE(mangoVelvet, nullptr);
  EXPECT_FLOAT_EQ(mangoVelvet->sheenColorFactor.x, 1.0f);
  EXPECT_FLOAT_EQ(mangoVelvet->sheenColorFactor.y, 0.329f);
  EXPECT_FLOAT_EQ(mangoVelvet->sheenColorFactor.z, 0.1f);
  EXPECT_FLOAT_EQ(mangoVelvet->sheenRoughnessFactor, 0.8f);
  EXPECT_FLOAT_EQ(mangoVelvet->sheenWeight, 1.0f);
  EXPECT_TRUE(mangoVelvet->sheenColor.path.empty());
  EXPECT_TRUE(mangoVelvet->sheenRoughness.path.empty());

  const nuri::ImportedMaterialInfo *peacockVelvet =
      findMaterialByName(set, "fabric Mystere Peacock Velvet");
  ASSERT_NE(peacockVelvet, nullptr);
  EXPECT_FLOAT_EQ(peacockVelvet->sheenColorFactor.x, 0.013f);
  EXPECT_FLOAT_EQ(peacockVelvet->sheenColorFactor.y, 0.284f);
  EXPECT_FLOAT_EQ(peacockVelvet->sheenColorFactor.z, 0.298f);
  EXPECT_FLOAT_EQ(peacockVelvet->sheenRoughnessFactor, 0.8f);
  EXPECT_FLOAT_EQ(peacockVelvet->sheenWeight, 1.0f);
}

TEST(MaterialImportTests, SheenChairImportsTextureTransforms) {
  auto result = nuri::MeshImporter::loadMaterialInfoFromFile(
      modelPath("SheenChair/SheenChair.gltf"));
  ASSERT_FALSE(result.hasError()) << result.error();

  const nuri::ImportedMaterialSet &set = result.value();
  const nuri::ImportedMaterialInfo *mangoVelvet =
      findMaterialByName(set, "fabric Mystere Mango Velvet");
  ASSERT_NE(mangoVelvet, nullptr);

  EXPECT_EQ(mangoVelvet->baseColor.uvSet, 0u);
  EXPECT_FLOAT_EQ(mangoVelvet->baseColor.transform.offset.x, -3.0f);
  EXPECT_FLOAT_EQ(mangoVelvet->baseColor.transform.offset.y, 3.0f);
  EXPECT_FLOAT_EQ(mangoVelvet->baseColor.transform.scale.x, 7.0f);
  EXPECT_FLOAT_EQ(mangoVelvet->baseColor.transform.scale.y, 7.0f);
  EXPECT_FLOAT_EQ(mangoVelvet->baseColor.transform.rotationRadians, 0.0f);

  EXPECT_EQ(mangoVelvet->normal.uvSet, 0u);
  EXPECT_FLOAT_EQ(mangoVelvet->normalScale, 0.6f);
  EXPECT_FLOAT_EQ(mangoVelvet->normal.transform.offset.x, -0.5f);
  EXPECT_FLOAT_EQ(mangoVelvet->normal.transform.offset.y, 0.5f);
  EXPECT_FLOAT_EQ(mangoVelvet->normal.transform.scale.x, 2.0f);
  EXPECT_FLOAT_EQ(mangoVelvet->normal.transform.scale.y, 2.0f);
  EXPECT_FLOAT_EQ(mangoVelvet->normal.transform.rotationRadians, 0.0f);

  EXPECT_EQ(mangoVelvet->occlusion.uvSet, 1u);
  EXPECT_FLOAT_EQ(mangoVelvet->occlusion.transform.offset.x, 0.0f);
  EXPECT_FLOAT_EQ(mangoVelvet->occlusion.transform.offset.y, 0.0f);
  EXPECT_FLOAT_EQ(mangoVelvet->occlusion.transform.scale.x, 1.0f);
  EXPECT_FLOAT_EQ(mangoVelvet->occlusion.transform.scale.y, 1.0f);
  EXPECT_FLOAT_EQ(mangoVelvet->occlusion.transform.rotationRadians, 0.0f);

  const nuri::ImportedMaterialInfo *woodBrown =
      findMaterialByName(set, "wood Brown");
  ASSERT_NE(woodBrown, nullptr);
  EXPECT_EQ(woodBrown->baseColor.uvSet, 0u);
  EXPECT_NEAR(woodBrown->baseColor.transform.offset.x, -0.8635584f, 1.0e-6f);
  EXPECT_NEAR(woodBrown->baseColor.transform.offset.y, 1.12502563f, 1.0e-6f);
  EXPECT_FLOAT_EQ(woodBrown->baseColor.transform.scale.x, 3.0f);
  EXPECT_FLOAT_EQ(woodBrown->baseColor.transform.scale.y, 3.0f);
  EXPECT_NEAR(woodBrown->baseColor.transform.rotationRadians, 0.08726647f,
              1.0e-6f);
}

TEST(MaterialImportTests, SheenChairLeavesVariantsUnapplied) {
  auto result = nuri::MeshImporter::loadMaterialInfoFromFile(
      modelPath("SheenChair/SheenChair.gltf"));
  ASSERT_FALSE(result.hasError()) << result.error();

  const nuri::ImportedMaterialSet &set = result.value();
  ASSERT_FALSE(set.materials.empty());
  EXPECT_NE(findMaterialByName(set, "fabric Mystere Mango Velvet"), nullptr);
  EXPECT_NE(findMaterialByName(set, "fabric Mystere Peacock Velvet"), nullptr);
}

TEST(MaterialImportTests, SyntheticGltfPreservesExplicitIorValue) {
  const ScopedTempDir dir("nuri_ior_gltf");
  auto result = nuri::MeshImporter::loadMaterialInfoFromFile(
      writeMinimalIorTestGltf(dir, 1.33f).string());
  ASSERT_FALSE(result.hasError()) << result.error();

  const nuri::ImportedMaterialInfo *material =
      findMaterialByName(result.value(), kIorTestMaterialName);
  ASSERT_NE(material, nullptr);
  EXPECT_FLOAT_EQ(material->ior, 1.33f);
}

TEST(MaterialImportTests, SyntheticGltfPreservesCompatIorZero) {
  const ScopedTempDir dir("nuri_ior_zero");
  auto result = nuri::MeshImporter::loadMaterialInfoFromFile(
      writeMinimalIorTestGltf(dir, 0.0f).string());
  ASSERT_FALSE(result.hasError()) << result.error();

  const nuri::ImportedMaterialInfo *material =
      findMaterialByName(result.value(), kIorTestMaterialName);
  ASSERT_NE(material, nullptr);
  EXPECT_FLOAT_EQ(material->ior, 0.0f);
}

TEST(MaterialImportTests, SyntheticGltfSanitizesInvalidSubOneIor) {
  const ScopedTempDir dir("nuri_ior_invalid");
  auto result = nuri::MeshImporter::loadMaterialInfoFromFile(
      writeMinimalIorTestGltf(dir, 0.5f).string());
  ASSERT_FALSE(result.hasError()) << result.error();

  const nuri::ImportedMaterialInfo *material =
      findMaterialByName(result.value(), kIorTestMaterialName);
  ASSERT_NE(material, nullptr);
  EXPECT_FLOAT_EQ(material->ior, 1.0f);
}

TEST(MaterialImportTests, SyntheticGlbMatchesGltfIorImport) {
  const ScopedTempDir dir("nuri_ior_glb");
  auto gltfResult = nuri::MeshImporter::loadMaterialInfoFromFile(
      writeMinimalIorTestGltf(dir, 1.33f).string());
  ASSERT_FALSE(gltfResult.hasError()) << gltfResult.error();
  auto glbResult = nuri::MeshImporter::loadMaterialInfoFromFile(
      writeMinimalIorTestGlb(dir, 1.33f).string());
  ASSERT_FALSE(glbResult.hasError()) << glbResult.error();

  const nuri::ImportedMaterialInfo *gltfMaterial =
      findMaterialByName(gltfResult.value(), kIorTestMaterialName);
  const nuri::ImportedMaterialInfo *glbMaterial =
      findMaterialByName(glbResult.value(), kIorTestMaterialName);
  ASSERT_NE(gltfMaterial, nullptr);
  ASSERT_NE(glbMaterial, nullptr);
  EXPECT_FLOAT_EQ(glbMaterial->ior, gltfMaterial->ior);
  EXPECT_EQ(glbMaterial->name, gltfMaterial->name);
}

TEST(MaterialImportTests, SyntheticGltfPreservesExplicitEmissiveStrength) {
  const ScopedTempDir dir("nuri_emissive_strength");
  auto result = nuri::MeshImporter::loadMaterialInfoFromFile(
      writeMinimalEmissiveStrengthTestGltf(dir, {0.1f, 0.2f, 0.3f}, 3.5f)
          .string());
  ASSERT_FALSE(result.hasError()) << result.error();

  const nuri::ImportedMaterialInfo *material =
      findMaterialByName(result.value(), kEmissiveStrengthTestMaterialName);
  ASSERT_NE(material, nullptr);
  EXPECT_FLOAT_EQ(material->emissiveFactor.x, 0.1f);
  EXPECT_FLOAT_EQ(material->emissiveFactor.y, 0.2f);
  EXPECT_FLOAT_EQ(material->emissiveFactor.z, 0.3f);
  EXPECT_FLOAT_EQ(material->emissiveStrength, 3.5f);
}

TEST(MaterialImportTests, SyntheticGltfDefaultsEmissiveStrengthToOne) {
  const ScopedTempDir dir("nuri_emissive_default");
  auto result = nuri::MeshImporter::loadMaterialInfoFromFile(
      writeMinimalEmissiveStrengthTestGltf(dir, {0.4f, 0.5f, 0.6f},
                                           std::nullopt)
          .string());
  ASSERT_FALSE(result.hasError()) << result.error();

  const nuri::ImportedMaterialInfo *material =
      findMaterialByName(result.value(), kEmissiveStrengthTestMaterialName);
  ASSERT_NE(material, nullptr);
  EXPECT_FLOAT_EQ(material->emissiveStrength, 1.0f);
}

TEST(MaterialImportTests, SyntheticGltfPreservesHdrEmissiveStrength) {
  const ScopedTempDir dir("nuri_emissive_hdr");
  auto result = nuri::MeshImporter::loadMaterialInfoFromFile(
      writeMinimalEmissiveStrengthTestGltf(dir, {0.3f, 0.2f, 0.1f}, 10.0f)
          .string());
  ASSERT_FALSE(result.hasError()) << result.error();

  const nuri::ImportedMaterialInfo *material =
      findMaterialByName(result.value(), kEmissiveStrengthTestMaterialName);
  ASSERT_NE(material, nullptr);
  EXPECT_FLOAT_EQ(material->emissiveStrength, 10.0f);
}

TEST(MaterialImportTests, SyntheticGltfSanitizesNegativeEmissiveStrength) {
  const ScopedTempDir dir("nuri_emissive_negative");
  auto result = nuri::MeshImporter::loadMaterialInfoFromFile(
      writeMinimalEmissiveStrengthTestGltf(dir, {0.3f, 0.2f, 0.1f}, -2.0f)
          .string());
  ASSERT_FALSE(result.hasError()) << result.error();

  const nuri::ImportedMaterialInfo *material =
      findMaterialByName(result.value(), kEmissiveStrengthTestMaterialName);
  ASSERT_NE(material, nullptr);
  EXPECT_FLOAT_EQ(material->emissiveStrength, 0.0f);
}

TEST(MaterialImportTests, SyntheticGlbMatchesGltfEmissiveStrengthImport) {
  const ScopedTempDir dir("nuri_emissive_glb");
  auto gltfResult = nuri::MeshImporter::loadMaterialInfoFromFile(
      writeMinimalEmissiveStrengthTestGltf(dir, {0.7f, 0.8f, 0.9f}, 4.0f)
          .string());
  ASSERT_FALSE(gltfResult.hasError()) << gltfResult.error();
  auto glbResult = nuri::MeshImporter::loadMaterialInfoFromFile(
      writeMinimalEmissiveStrengthTestGlb(dir, {0.7f, 0.8f, 0.9f}, 4.0f)
          .string());
  ASSERT_FALSE(glbResult.hasError()) << glbResult.error();

  const nuri::ImportedMaterialInfo *gltfMaterial =
      findMaterialByName(gltfResult.value(), kEmissiveStrengthTestMaterialName);
  const nuri::ImportedMaterialInfo *glbMaterial =
      findMaterialByName(glbResult.value(), kEmissiveStrengthTestMaterialName);
  ASSERT_NE(gltfMaterial, nullptr);
  ASSERT_NE(glbMaterial, nullptr);
  EXPECT_FLOAT_EQ(glbMaterial->emissiveFactor.x,
                  gltfMaterial->emissiveFactor.x);
  EXPECT_FLOAT_EQ(glbMaterial->emissiveFactor.y,
                  gltfMaterial->emissiveFactor.y);
  EXPECT_FLOAT_EQ(glbMaterial->emissiveFactor.z,
                  gltfMaterial->emissiveFactor.z);
  EXPECT_FLOAT_EQ(glbMaterial->emissiveStrength,
                  gltfMaterial->emissiveStrength);
  EXPECT_EQ(glbMaterial->name, gltfMaterial->name);
}

TEST(MaterialImportTests, SyntheticGltfPreservesSpecularFactors) {
  const ScopedTempDir dir("nuri_specular_factor");
  auto result = nuri::MeshImporter::loadMaterialInfoFromFile(
      writeMinimalSpecularTestGltf(dir, 0.35f, {0.2f, 0.4f, 0.8f}, false)
          .string());
  ASSERT_FALSE(result.hasError()) << result.error();

  const nuri::ImportedMaterialInfo *material =
      findMaterialByName(result.value(), kSpecularTestMaterialName);
  ASSERT_NE(material, nullptr);
  EXPECT_FLOAT_EQ(material->specularFactor, 0.35f);
  EXPECT_FLOAT_EQ(material->specularColorFactor.x, 0.2f);
  EXPECT_FLOAT_EQ(material->specularColorFactor.y, 0.4f);
  EXPECT_FLOAT_EQ(material->specularColorFactor.z, 0.8f);
  EXPECT_TRUE(material->specular.path.empty());
  EXPECT_TRUE(material->specularColor.path.empty());
}

TEST(MaterialImportTests, SyntheticGltfPreservesSpecularColorAboveOne) {
  const ScopedTempDir dir("nuri_specular_hdr");
  auto result = nuri::MeshImporter::loadMaterialInfoFromFile(
      writeMinimalSpecularTestGltf(dir, 0.5f, {10.0f, 0.6f, 0.0f}, false)
          .string());
  ASSERT_FALSE(result.hasError()) << result.error();

  const nuri::ImportedMaterialInfo *material =
      findMaterialByName(result.value(), kSpecularTestMaterialName);
  ASSERT_NE(material, nullptr);
  EXPECT_FLOAT_EQ(material->specularColorFactor.x, 10.0f);
  EXPECT_FLOAT_EQ(material->specularColorFactor.y, 0.6f);
  EXPECT_FLOAT_EQ(material->specularColorFactor.z, 0.0f);
}

TEST(MaterialImportTests, SyntheticGltfImportsSpecularTexturesAndTransforms) {
  const ScopedTempDir dir("nuri_specular_textures");
  auto result = nuri::MeshImporter::loadMaterialInfoFromFile(
      writeMinimalSpecularTestGltf(dir, 0.75f, {1.0f, 0.5f, 0.25f}, true)
          .string());
  ASSERT_FALSE(result.hasError()) << result.error();

  const nuri::ImportedMaterialInfo *material =
      findMaterialByName(result.value(), kSpecularTestMaterialName);
  ASSERT_NE(material, nullptr);

  const std::filesystem::path expectedSpecularPath =
      (dir.path / "textures" / "specular_strength.png").lexically_normal();
  const std::filesystem::path expectedSpecularColorPath =
      (dir.path / "textures" / "specular_color.png").lexically_normal();

  EXPECT_EQ(std::filesystem::path(material->specular.path),
            expectedSpecularPath);
  EXPECT_EQ(std::filesystem::path(material->specularColor.path),
            expectedSpecularColorPath);
  EXPECT_EQ(material->specular.sourceKind,
            nuri::MaterialTextureSourceKind::ExternalFile);
  EXPECT_EQ(material->specularColor.sourceKind,
            nuri::MaterialTextureSourceKind::ExternalFile);
  EXPECT_EQ(material->specular.uvSet, 0u);
  EXPECT_EQ(material->specular.samplerIndex, 1u);
  EXPECT_FLOAT_EQ(material->specular.transform.offset.x, 0.25f);
  EXPECT_FLOAT_EQ(material->specular.transform.offset.y, 0.5f);
  EXPECT_FLOAT_EQ(material->specular.transform.scale.x, 2.0f);
  EXPECT_FLOAT_EQ(material->specular.transform.scale.y, 3.0f);
  EXPECT_NEAR(material->specular.transform.rotationRadians, 0.785398163f,
              1.0e-6f);

  EXPECT_EQ(material->specularColor.uvSet, 1u);
  EXPECT_EQ(material->specularColor.samplerIndex, 0u);
  EXPECT_FLOAT_EQ(material->specularColor.transform.offset.x, -1.0f);
  EXPECT_FLOAT_EQ(material->specularColor.transform.offset.y, 1.5f);
  EXPECT_FLOAT_EQ(material->specularColor.transform.scale.x, 4.0f);
  EXPECT_FLOAT_EQ(material->specularColor.transform.scale.y, 5.0f);
  EXPECT_NEAR(material->specularColor.transform.rotationRadians, 0.25f,
              1.0e-6f);
}

TEST(MaterialImportTests, SyntheticGltfPreservesSpecGlossFactors) {
  const ScopedTempDir dir("nuri_specgloss_factor");
  nuri::ImportedMaterialSet set{};
  const nuri::ImportedMaterialInfo *material =
      loadSyntheticSpecGlossMaterial(dir, false, false, set);
  ASSERT_NE(material, nullptr);
  EXPECT_EQ(material->workflow, nuri::MaterialWorkflow::SpecularGlossiness);
  EXPECT_FLOAT_EQ(material->baseColorFactor.x, 0.9f);
  EXPECT_FLOAT_EQ(material->baseColorFactor.w, 0.6f);
  EXPECT_FLOAT_EQ(material->specularColorFactor.x, 1.5f);
  EXPECT_FLOAT_EQ(material->specularColorFactor.y, 0.5f);
  EXPECT_FLOAT_EQ(material->specularColorFactor.z, 0.25f);
  EXPECT_FLOAT_EQ(material->glossinessFactor, 0.35f);
  EXPECT_FLOAT_EQ(material->metallicFactor, 0.0f);
  EXPECT_FLOAT_EQ(material->roughnessFactor, 1.0f);
  EXPECT_TRUE(material->metallicRoughness.path.empty());
  EXPECT_TRUE(material->specular.path.empty());
}

TEST(MaterialImportTests, SyntheticGltfImportsSpecGlossTexturesAndTransforms) {
  const ScopedTempDir dir("nuri_specgloss_textures");
  nuri::ImportedMaterialSet set{};
  const nuri::ImportedMaterialInfo *material =
      loadSyntheticSpecGlossMaterial(dir, true, false, set);
  ASSERT_NE(material, nullptr);
  EXPECT_EQ(material->workflow, nuri::MaterialWorkflow::SpecularGlossiness);
  EXPECT_EQ(std::filesystem::path(material->baseColor.path).filename(),
            std::filesystem::path("specgloss_diffuse.png"));
  EXPECT_EQ(std::filesystem::path(material->specularColor.path).filename(),
            std::filesystem::path("specgloss_rgba.png"));
  EXPECT_EQ(material->baseColor.uvSet, 0u);
  EXPECT_EQ(material->baseColor.samplerIndex, 1u);
  EXPECT_FLOAT_EQ(material->baseColor.transform.offset.x, 0.125f);
  EXPECT_FLOAT_EQ(material->baseColor.transform.scale.y, 2.5f);
  EXPECT_EQ(material->specularColor.uvSet, 1u);
  EXPECT_EQ(material->specularColor.samplerIndex, 0u);
  EXPECT_FLOAT_EQ(material->specularColor.transform.offset.x, -0.5f);
  EXPECT_FLOAT_EQ(material->specularColor.transform.scale.y, 5.0f);
  EXPECT_NEAR(material->specularColor.transform.rotationRadians, 0.125f,
              1.0e-6f);
}

TEST(MaterialImportTests, MaterialDescFromImportedIgnoresImportedSamplerState) {
  nuri::MaterialData material{};
  material.baseColor.samplerIndex = 5u;
  material.normal.samplerIndex = 3u;
  material.specularColor.samplerIndex = 7u;

  nuri::MaterialData differentSamplers = material;
  differentSamplers.baseColor.samplerIndex = 0u;
  differentSamplers.normal.samplerIndex = 11u;
  differentSamplers.specularColor.samplerIndex = 1u;

  const nuri::MaterialDesc desc = nuri::Material::descFromImported(material);
  const nuri::MaterialDesc descWithoutSamplers =
      nuri::Material::descFromImported(differentSamplers);
  EXPECT_EQ(nuri::hashMaterialDesc(desc),
            nuri::hashMaterialDesc(descWithoutSamplers));
  EXPECT_EQ(desc.uvSets.baseColor, descWithoutSamplers.uvSets.baseColor);
  EXPECT_EQ(desc.uvSets.normal, descWithoutSamplers.uvSets.normal);
  EXPECT_EQ(desc.uvSets.specularColor,
            descWithoutSamplers.uvSets.specularColor);
}

TEST(MaterialImportTests, SyntheticGltfSpecGlossWinsOverSpecularExtension) {
  const ScopedTempDir dir("nuri_specgloss_precedence");
  nuri::ImportedMaterialSet set{};
  const nuri::ImportedMaterialInfo *material =
      loadSyntheticSpecGlossMaterial(dir, true, true, set);
  ASSERT_NE(material, nullptr);
  EXPECT_EQ(material->workflow, nuri::MaterialWorkflow::SpecularGlossiness);
  EXPECT_FLOAT_EQ(material->specularColorFactor.x, 1.5f);
  EXPECT_FLOAT_EQ(material->glossinessFactor, 0.35f);
  EXPECT_TRUE(material->specular.path.empty());
  EXPECT_EQ(std::filesystem::path(material->specularColor.path).filename(),
            std::filesystem::path("specgloss_rgba.png"));
}

TEST(MaterialImportTests,
     MaterialDescPreservesSpecGlossWorkflowAndAffectsHash) {
  nuri::MaterialData imported{};
  imported.workflow = nuri::MaterialWorkflow::SpecularGlossiness;
  imported.baseColorFactor = glm::vec4(0.9f, 0.8f, 0.7f, 0.6f);
  imported.specularColorFactor = glm::vec3(1.4f, 0.5f, 0.25f);
  imported.glossinessFactor = 0.35f;

  const nuri::MaterialDesc desc = nuri::Material::descFromImported(imported);
  EXPECT_EQ(desc.workflow, nuri::MaterialWorkflow::SpecularGlossiness);
  EXPECT_FLOAT_EQ(desc.glossinessFactor, 0.35f);

  nuri::MaterialDesc differentWorkflow = desc;
  differentWorkflow.workflow = nuri::MaterialWorkflow::MetallicRoughness;
  differentWorkflow.specularFactor = 1.0f;
  const uint64_t specGlossHash = nuri::hashMaterialDesc(desc);
  const uint64_t metallicHash = nuri::hashMaterialDesc(differentWorkflow);
  EXPECT_NE(specGlossHash, metallicHash);

  nuri::MaterialDesc differentGlossiness = desc;
  differentGlossiness.glossinessFactor = 0.65f;
  EXPECT_NE(specGlossHash, nuri::hashMaterialDesc(differentGlossiness));
}

TEST(MaterialImportTests, SpecularSilkPoufImportsSpecularAndSheenData) {
  auto result = nuri::MeshImporter::loadMaterialInfoFromFile(
      modelPath("SpecularSilkPouf/SpecularSilkPouf.gltf"));
  ASSERT_FALSE(result.hasError()) << result.error();

  const nuri::ImportedMaterialInfo *material =
      findMaterialByName(result.value(), "shot silk");
  ASSERT_NE(material, nullptr);
  EXPECT_FLOAT_EQ(material->specularFactor, 0.5f);
  EXPECT_FLOAT_EQ(material->specularColorFactor.x, 10.0f);
  EXPECT_FLOAT_EQ(material->specularColorFactor.y, 0.6f);
  EXPECT_FLOAT_EQ(material->specularColorFactor.z, 0.0f);
  EXPECT_FLOAT_EQ(material->sheenColorFactor.x, 0.025f);
  EXPECT_FLOAT_EQ(material->sheenColorFactor.y, 0.03f);
  EXPECT_FLOAT_EQ(material->sheenColorFactor.z, 0.075f);
  EXPECT_FLOAT_EQ(material->sheenRoughnessFactor, 0.6f);
  EXPECT_TRUE(material->specular.path.empty());
  EXPECT_TRUE(material->specularColor.path.empty());
}

TEST(MaterialImportTests, DragonAttenuationKeepsDefaultIor) {
  auto result = nuri::MeshImporter::loadMaterialInfoFromFile(
      modelPath("DragonAttenuation/DragonAttenuation.gltf"));
  ASSERT_FALSE(result.hasError()) << result.error();

  const nuri::ImportedMaterialInfo *attenuation =
      findMaterialByName(result.value(), "Dragon with Attenuation");
  const nuri::ImportedMaterialInfo *surfaceColor =
      findMaterialByName(result.value(), "Dragon with Surface Coloring Only");
  ASSERT_NE(attenuation, nullptr);
  ASSERT_NE(surfaceColor, nullptr);
  EXPECT_FLOAT_EQ(attenuation->transmissionFactor, 1.0f);
  EXPECT_FLOAT_EQ(surfaceColor->transmissionFactor, 1.0f);
  EXPECT_FLOAT_EQ(attenuation->ior, 1.5f);
  EXPECT_FLOAT_EQ(surfaceColor->ior, 1.5f);
}

} // namespace
