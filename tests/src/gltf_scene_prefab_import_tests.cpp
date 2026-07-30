#include "tests_pch.h"

#include "nuri/core/log.h"
#include "nuri/resources/mesh_importer.h"
#include "nuri/resources/scene_importer.h"
#include "nuri/scene/scene_prefab.h"

#include <chrono>
#include <functional>
#include <sstream>
#include <thread>
#include <unordered_map>

#include <meshoptimizer.h>
#include <yyjson.h>

namespace {

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

  ScopedTempDir(const ScopedTempDir &) = delete;
  ScopedTempDir &operator=(const ScopedTempDir &) = delete;
  ScopedTempDir(ScopedTempDir &&) = delete;
  ScopedTempDir &operator=(ScopedTempDir &&) = delete;

  std::filesystem::path path;
};

struct TestGltfNode {
  std::string name{};
  std::vector<uint32_t> children{};
  std::optional<uint32_t> meshIndex{};
};

struct NodeMaterialExpectation {
  std::string path{};
  uint32_t materialIndex = std::numeric_limits<uint32_t>::max();
};

void writeTextFile(const std::filesystem::path &path, std::string_view text) {
  std::ofstream file(path, std::ios::binary);
  ASSERT_TRUE(file.is_open());
  file.write(text.data(), static_cast<std::streamsize>(text.size()));
  ASSERT_TRUE(file.good());
}

void writeBinaryFile(const std::filesystem::path &path,
                     std::span<const uint8_t> bytes) {
  std::ofstream file(path, std::ios::binary);
  ASSERT_TRUE(file.is_open());
  if (!bytes.empty()) {
    file.write(reinterpret_cast<const char *>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  }
  ASSERT_TRUE(file.good());
}

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

std::vector<uint8_t> twoMeshBuffer() {
  constexpr std::array<float, 18> kPositions = {
      0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
      2.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f, 2.0f, 1.0f, 0.0f,
  };
  constexpr std::array<uint16_t, 6> kIndices = {
      0u, 1u, 2u, 0u, 1u, 2u,
  };

  std::vector<uint8_t> bytes(sizeof(kPositions) + sizeof(kIndices));
  std::memcpy(bytes.data(), kPositions.data(), sizeof(kPositions));
  std::memcpy(bytes.data() + sizeof(kPositions), kIndices.data(),
              sizeof(kIndices));
  return bytes;
}

[[nodiscard]] int findNodeByName(const nuri::ScenePrefab &prefab,
                                 std::string_view name) {
  for (uint32_t index = 0; index < prefab.nodes.size(); ++index) {
    if (std::string_view(prefab.nodes[index].name) == name) {
      return static_cast<int>(index);
    }
  }
  return -1;
}

[[nodiscard]] glm::vec3 translationOf(const glm::mat4 &matrix) {
  return glm::vec3(matrix[3]);
}

void expectVec3Near(const glm::vec3 &actual, const glm::vec3 &expected,
                    float epsilon = 1.0e-5f) {
  EXPECT_NEAR(actual.x, expected.x, epsilon);
  EXPECT_NEAR(actual.y, expected.y, epsilon);
  EXPECT_NEAR(actual.z, expected.z, epsilon);
}

std::string modelPath(std::string_view relativePath) {
  const std::filesystem::path root(PROJECT_SOURCE_DIR);
  return (root / "assets" / "models" / std::filesystem::path(relativePath))
      .string();
}

[[nodiscard]] std::string testNodePath(std::string_view parentPath,
                                       std::string_view nodeName,
                                       uint32_t siblingOrdinal) {
  const std::string component = nodeName.empty()
                                    ? ("#" + std::to_string(siblingOrdinal))
                                    : std::string(nodeName);
  if (parentPath.empty()) {
    return component;
  }
  return std::string(parentPath) + "/" + component;
}

[[nodiscard]] bool testReadJsonUint32(yyjson_val *value, uint32_t &out) {
  if (!yyjson_is_uint(value)) {
    return false;
  }
  const uint64_t rawValue = yyjson_get_uint(value);
  if (rawValue > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  out = static_cast<uint32_t>(rawValue);
  return true;
}

[[nodiscard]] std::string_view testReadJsonStringView(yyjson_val *object,
                                                      const char *key) {
  yyjson_val *value = yyjson_obj_get(object, key);
  if (!yyjson_is_str(value)) {
    return {};
  }
  return std::string_view(yyjson_get_str(value), yyjson_get_len(value));
}

[[nodiscard]] std::optional<std::string>
readTextFileForTest(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    ADD_FAILURE() << "failed to open " << path.string();
    return std::nullopt;
  }
  std::ostringstream stream;
  stream << file.rdbuf();
  return stream.str();
}

[[nodiscard]] std::optional<std::vector<NodeMaterialExpectation>>
readSinglePrimitiveNodeMaterialExpectations(const std::filesystem::path &path) {
  std::optional<std::string> json = readTextFileForTest(path);
  if (!json.has_value()) {
    return std::nullopt;
  }

  yyjson_read_err parseError{};
  std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)> doc(
      yyjson_read_opts(json->data(), json->size(), 0, nullptr, &parseError),
      &yyjson_doc_free);
  if (doc == nullptr) {
    ADD_FAILURE() << "failed to parse " << path.string() << " near byte "
                  << parseError.pos;
    return std::nullopt;
  }

  yyjson_val *root = yyjson_doc_get_root(doc.get());
  yyjson_val *meshes = yyjson_obj_get(root, "meshes");
  if (!yyjson_is_arr(meshes)) {
    ADD_FAILURE() << "glTF meshes array is missing";
    return std::nullopt;
  }

  std::vector<uint32_t> meshMaterials;
  meshMaterials.reserve(yyjson_arr_size(meshes));
  yyjson_arr_iter meshIter = yyjson_arr_iter_with(meshes);
  yyjson_val *meshValue = nullptr;
  while ((meshValue = yyjson_arr_iter_next(&meshIter)) != nullptr) {
    uint32_t materialIndex = std::numeric_limits<uint32_t>::max();
    yyjson_val *primitives = yyjson_obj_get(meshValue, "primitives");
    if (yyjson_is_arr(primitives) && yyjson_arr_size(primitives) == 1u) {
      (void)testReadJsonUint32(
          yyjson_obj_get(yyjson_arr_get(primitives, 0), "material"),
          materialIndex);
    }
    meshMaterials.push_back(materialIndex);
  }

  yyjson_val *nodesValue = yyjson_obj_get(root, "nodes");
  if (!yyjson_is_arr(nodesValue)) {
    ADD_FAILURE() << "glTF nodes array is missing";
    return std::nullopt;
  }

  std::vector<TestGltfNode> nodes(yyjson_arr_size(nodesValue));
  for (uint32_t nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex) {
    yyjson_val *nodeValue = yyjson_arr_get(nodesValue, nodeIndex);
    const std::string_view name = testReadJsonStringView(nodeValue, "name");
    nodes[nodeIndex].name.assign(name.begin(), name.end());

    uint32_t meshIndex = 0u;
    if (testReadJsonUint32(yyjson_obj_get(nodeValue, "mesh"), meshIndex)) {
      nodes[nodeIndex].meshIndex = meshIndex;
    }

    yyjson_val *children = yyjson_obj_get(nodeValue, "children");
    if (yyjson_is_arr(children)) {
      nodes[nodeIndex].children.reserve(yyjson_arr_size(children));
      yyjson_arr_iter childIter = yyjson_arr_iter_with(children);
      yyjson_val *childValue = nullptr;
      while ((childValue = yyjson_arr_iter_next(&childIter)) != nullptr) {
        uint32_t childIndex = 0u;
        if (testReadJsonUint32(childValue, childIndex)) {
          nodes[nodeIndex].children.push_back(childIndex);
        }
      }
    }
  }

  yyjson_val *scenes = yyjson_obj_get(root, "scenes");
  if (!yyjson_is_arr(scenes)) {
    ADD_FAILURE() << "glTF scenes array is missing";
    return std::nullopt;
  }

  uint32_t sceneIndex = 0u;
  (void)testReadJsonUint32(yyjson_obj_get(root, "scene"), sceneIndex);
  if (sceneIndex >= yyjson_arr_size(scenes)) {
    ADD_FAILURE() << "glTF default scene index is out of range";
    return std::nullopt;
  }

  yyjson_val *rootNodes =
      yyjson_obj_get(yyjson_arr_get(scenes, sceneIndex), "nodes");
  if (!yyjson_is_arr(rootNodes)) {
    ADD_FAILURE() << "glTF scene nodes array is missing";
    return std::nullopt;
  }

  std::vector<NodeMaterialExpectation> expectations;
  std::vector<uint8_t> active(nodes.size(), 0u);
  std::function<void(uint32_t, const std::string &, uint32_t)> visit =
      [&](uint32_t nodeIndex, const std::string &parentPath,
          uint32_t siblingOrdinal) {
        if (nodeIndex >= nodes.size() || active[nodeIndex] != 0u) {
          return;
        }
        active[nodeIndex] = 1u;
        const TestGltfNode &node = nodes[nodeIndex];
        const std::string pathKey =
            testNodePath(parentPath, node.name, siblingOrdinal);
        if (node.meshIndex.has_value() &&
            *node.meshIndex < meshMaterials.size() &&
            meshMaterials[*node.meshIndex] !=
                std::numeric_limits<uint32_t>::max()) {
          expectations.push_back(NodeMaterialExpectation{
              .path = pathKey,
              .materialIndex = meshMaterials[*node.meshIndex],
          });
        }
        for (uint32_t childOrdinal = 0u; childOrdinal < node.children.size();
             ++childOrdinal) {
          visit(node.children[childOrdinal], pathKey, childOrdinal);
        }
        active[nodeIndex] = 0u;
      };

  yyjson_arr_iter rootIter = yyjson_arr_iter_with(rootNodes);
  yyjson_val *rootValue = nullptr;
  uint32_t rootOrdinal = 0u;
  while ((rootValue = yyjson_arr_iter_next(&rootIter)) != nullptr) {
    uint32_t rootNodeIndex = 0u;
    if (testReadJsonUint32(rootValue, rootNodeIndex)) {
      visit(rootNodeIndex, std::string(), rootOrdinal);
    }
    ++rootOrdinal;
  }

  return expectations;
}

[[nodiscard]] std::vector<std::string>
buildImportedNodePaths(const nuri::ScenePrefab &scene) {
  std::vector<std::vector<uint32_t>> children(scene.nodes.size());
  for (uint32_t nodeIndex = 0u; nodeIndex < scene.nodes.size(); ++nodeIndex) {
    const uint32_t parentIndex = scene.nodes[nodeIndex].parentIndex;
    if (parentIndex != nuri::kInvalidScenePrefabIndex &&
        parentIndex < children.size()) {
      children[parentIndex].push_back(nodeIndex);
    }
  }

  std::vector<std::string> paths(scene.nodes.size());
  std::vector<uint8_t> active(scene.nodes.size(), 0u);
  std::function<void(uint32_t, const std::string &, uint32_t)> visit =
      [&](uint32_t nodeIndex, const std::string &parentPath,
          uint32_t siblingOrdinal) {
        if (nodeIndex >= scene.nodes.size() || active[nodeIndex] != 0u) {
          return;
        }
        active[nodeIndex] = 1u;
        paths[nodeIndex] = testNodePath(parentPath, scene.nodes[nodeIndex].name,
                                        siblingOrdinal);
        for (uint32_t childOrdinal = 0u;
             childOrdinal < children[nodeIndex].size(); ++childOrdinal) {
          visit(children[nodeIndex][childOrdinal], paths[nodeIndex],
                childOrdinal);
        }
        active[nodeIndex] = 0u;
      };

  for (uint32_t rootOrdinal = 0u; rootOrdinal < scene.rootNodes.size();
       ++rootOrdinal) {
    visit(scene.rootNodes[rootOrdinal], std::string(), rootOrdinal);
  }
  return paths;
}

[[nodiscard]] std::string makeWavyGridObj(uint32_t sideLength) {
  std::ostringstream stream;
  stream << "o WavyGrid\n";
  for (uint32_t y = 0; y < sideLength; ++y) {
    for (uint32_t x = 0; x < sideLength; ++x) {
      const float z = 0.2f * std::sin(static_cast<float>(x) * 0.45f) *
                      std::cos(static_cast<float>(y) * 0.35f);
      stream << "v " << x << ' ' << y << ' ' << z << '\n';
    }
  }

  for (uint32_t y = 0; y + 1 < sideLength; ++y) {
    for (uint32_t x = 0; x + 1 < sideLength; ++x) {
      const uint32_t topLeft = y * sideLength + x + 1u;
      const uint32_t topRight = topLeft + 1u;
      const uint32_t bottomLeft = topLeft + sideLength;
      const uint32_t bottomRight = bottomLeft + 1u;
      stream << "f " << topLeft << ' ' << bottomLeft << ' ' << topRight << '\n';
      stream << "f " << topRight << ' ' << bottomLeft << ' ' << bottomRight
             << '\n';
    }
  }
  return stream.str();
}

} // namespace

TEST(GltfScenePrefabImport, LoadsPrefabHierarchyRenderablesAndLightsFromFile) {
  ScopedTempDir dir("nuri_gltf_scene_prefab");

  const std::vector<uint8_t> buffer = minimalTriangleBuffer();
  const std::filesystem::path bufferPath = dir.path / "scene.bin";
  writeBinaryFile(bufferPath, buffer);

  const std::string json = R"json(
{
  "asset": {"version": "2.0"},
  "extensionsUsed": ["KHR_lights_punctual"],
  "extensions": {
    "KHR_lights_punctual": {
      "lights": [
        {
          "name": "LampDef",
          "type": "spot",
          "color": [0.5, 0.75, 1.0],
          "intensity": 9.0,
          "range": 6.0,
          "spot": {
            "innerConeAngle": 0.2,
            "outerConeAngle": 0.5
          }
        }
      ]
    }
  },
  "scene": 0,
  "scenes": [
    {
      "name": "PrefabScene",
      "nodes": [0]
    }
  ],
  "nodes": [
    {
      "name": "ParentNode",
      "translation": [2.0, 0.0, 0.0],
      "children": [1, 2]
    },
    {
      "name": "MeshNode",
      "translation": [0.0, 3.0, 0.0],
      "mesh": 0
    },
    {
      "name": "LightNode",
      "translation": [0.0, 1.5, 0.0],
      "rotation": [0.0, 0.0, 0.38268343, 0.92387953],
      "extensions": {"KHR_lights_punctual": {"light": 0}}
    }
  ],
  "materials": [
    {
      "name": "Mat0",
      "pbrMetallicRoughness": {
        "baseColorFactor": [1, 1, 1, 1],
        "metallicFactor": 0.0,
        "roughnessFactor": 1.0
      }
    }
  ],
  "meshes": [
    {
      "primitives": [
        {
          "attributes": {"POSITION": 0},
          "indices": 1,
          "material": 0
        }
      ]
    }
  ],
  "accessors": [
    {
      "bufferView": 0,
      "componentType": 5126,
      "count": 3,
      "type": "VEC3",
      "max": [1, 1, 0],
      "min": [0, 0, 0]
    },
    {
      "bufferView": 1,
      "componentType": 5123,
      "count": 3,
      "type": "SCALAR",
      "max": [2],
      "min": [0]
    }
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962},
    {"buffer": 0, "byteOffset": 36, "byteLength": 6, "target": 34963}
  ],
  "buffers": [
    {"byteLength": 42, "uri": "scene.bin"}
  ]
}
)json";

  const std::filesystem::path gltfPath = dir.path / "prefab_scene.gltf";
  writeTextFile(gltfPath, json);

  auto result = nuri::SceneImporter::loadSceneFromFile(gltfPath.string());
  ASSERT_FALSE(result.hasError()) << result.error();

  const nuri::ScenePrefab &prefab = result.value().prefab;
  EXPECT_EQ(prefab.sourceSceneName, "PrefabScene");
  EXPECT_EQ(prefab.meshAssets.size(), 1u);
  EXPECT_GE(prefab.materialAssets.size(), 1u);
  EXPECT_EQ(prefab.renderables.size(), 1u);
  EXPECT_EQ(prefab.lights.size(), 1u);

  const int parentIndex = findNodeByName(prefab, "ParentNode");
  const int meshIndex = findNodeByName(prefab, "MeshNode");
  const int lightIndex = findNodeByName(prefab, "LightNode");
  ASSERT_GE(parentIndex, 0);
  ASSERT_GE(meshIndex, 0);
  ASSERT_GE(lightIndex, 0);

  expectVec3Near(translationOf(prefab.nodes[parentIndex].localFromParent),
                 glm::vec3(2.0f, 0.0f, 0.0f));
  EXPECT_EQ(prefab.nodes[meshIndex].parentIndex,
            static_cast<uint32_t>(parentIndex));
  EXPECT_EQ(prefab.nodes[lightIndex].parentIndex,
            static_cast<uint32_t>(parentIndex));
  expectVec3Near(translationOf(prefab.nodes[meshIndex].localFromParent),
                 glm::vec3(0.0f, 3.0f, 0.0f));
  expectVec3Near(translationOf(prefab.nodes[lightIndex].localFromParent),
                 glm::vec3(0.0f, 1.5f, 0.0f));

  EXPECT_EQ(prefab.renderables[0].nodeIndex, static_cast<uint32_t>(meshIndex));
  EXPECT_EQ(prefab.renderables[0].meshAssetIndex, 0u);
  EXPECT_LT(prefab.renderables[0].materialAssetIndex,
            prefab.materialAssets.size());

  EXPECT_EQ(prefab.lights[0].nodeIndex, static_cast<uint32_t>(lightIndex));
  EXPECT_EQ(prefab.lights[0].light.name, "LightNode");
  EXPECT_EQ(prefab.lights[0].light.type, nuri::LightType::Spot);
  EXPECT_FLOAT_EQ(prefab.lights[0].light.intensity, 9.0f);
  EXPECT_FLOAT_EQ(prefab.lights[0].light.range, 6.0f);
  EXPECT_FLOAT_EQ(prefab.lights[0].light.innerConeAngleRadians, 0.2f);
  EXPECT_FLOAT_EQ(prefab.lights[0].light.outerConeAngleRadians, 0.5f);
  expectVec3Near(prefab.lights[0].light.position, glm::vec3(0.0f));
}

TEST(GltfScenePrefabImport, BatchLoadsSceneMeshesFromSingleImportPass) {
  ScopedTempDir dir("nuri_gltf_scene_mesh_batch");

  const std::vector<uint8_t> buffer = twoMeshBuffer();
  const std::filesystem::path bufferPath = dir.path / "scene.bin";
  writeBinaryFile(bufferPath, buffer);

  const std::string json = R"json(
{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [
    {
      "nodes": [0, 1]
    }
  ],
  "nodes": [
    {
      "name": "MeshNode0",
      "mesh": 0
    },
    {
      "name": "MeshNode1",
      "mesh": 1
    }
  ],
  "materials": [
    {
      "name": "Mat0",
      "pbrMetallicRoughness": {
        "baseColorFactor": [1, 0, 0, 1],
        "metallicFactor": 0.0,
        "roughnessFactor": 1.0
      }
    },
    {
      "name": "Mat1",
      "pbrMetallicRoughness": {
        "baseColorFactor": [0, 1, 0, 1],
        "metallicFactor": 0.0,
        "roughnessFactor": 1.0
      }
    }
  ],
  "meshes": [
    {
      "primitives": [
        {
          "attributes": {"POSITION": 0},
          "indices": 2,
          "material": 0
        }
      ]
    },
    {
      "primitives": [
        {
          "attributes": {"POSITION": 1},
          "indices": 3,
          "material": 1
        }
      ]
    }
  ],
  "accessors": [
    {
      "bufferView": 0,
      "componentType": 5126,
      "count": 3,
      "type": "VEC3",
      "max": [1, 1, 0],
      "min": [0, 0, 0]
    },
    {
      "bufferView": 1,
      "componentType": 5126,
      "count": 3,
      "type": "VEC3",
      "max": [3, 1, 0],
      "min": [2, 0, 0]
    },
    {
      "bufferView": 2,
      "componentType": 5123,
      "count": 3,
      "type": "SCALAR",
      "max": [2],
      "min": [0]
    },
    {
      "bufferView": 3,
      "componentType": 5123,
      "count": 3,
      "type": "SCALAR",
      "max": [2],
      "min": [0]
    }
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962},
    {"buffer": 0, "byteOffset": 36, "byteLength": 36, "target": 34962},
    {"buffer": 0, "byteOffset": 72, "byteLength": 6, "target": 34963},
    {"buffer": 0, "byteOffset": 78, "byteLength": 6, "target": 34963}
  ],
  "buffers": [
    {"byteLength": 84, "uri": "scene.bin"}
  ]
}

)json";

  const std::filesystem::path gltfPath = dir.path / "scene_meshes.gltf";
  writeTextFile(gltfPath, json);

  nuri::MeshImportOptions options{};
  options.optimize = false;
  options.generateLods = false;
  options.lodCount = 1u;

  auto singleMesh0 =
      nuri::MeshImporter::loadSceneMeshFromFile(gltfPath.string(), 0u, options);
  auto singleMesh1 =
      nuri::MeshImporter::loadSceneMeshFromFile(gltfPath.string(), 1u, options);
  ASSERT_FALSE(singleMesh0.hasError()) << singleMesh0.error();
  ASSERT_FALSE(singleMesh1.hasError()) << singleMesh1.error();

  constexpr std::array<uint32_t, 2> kSceneMeshIndices = {0u, 1u};
  auto batchMeshes = nuri::MeshImporter::loadSceneMeshesFromFile(
      gltfPath.string(), kSceneMeshIndices, options);
  ASSERT_FALSE(batchMeshes.hasError()) << batchMeshes.error();
  ASSERT_EQ(batchMeshes.value().size(), kSceneMeshIndices.size());

  const nuri::MeshData &batchMesh0 = batchMeshes.value()[0];
  const nuri::MeshData &batchMesh1 = batchMeshes.value()[1];

  EXPECT_EQ(batchMesh0.vertices.size(), singleMesh0.value().vertices.size());
  EXPECT_EQ(batchMesh0.indices.size(), singleMesh0.value().indices.size());
  ASSERT_EQ(batchMesh0.submeshes.size(), singleMesh0.value().submeshes.size());
  ASSERT_GE(batchMesh0.vertices.size(), 3u);
  ASSERT_GE(singleMesh0.value().vertices.size(), 3u);
  EXPECT_EQ(batchMesh0.submeshes[0].materialIndex,
            singleMesh0.value().submeshes[0].materialIndex);
  expectVec3Near(batchMesh0.vertices[0].position,
                 singleMesh0.value().vertices[0].position);
  expectVec3Near(batchMesh0.vertices[2].position,
                 singleMesh0.value().vertices[2].position);

  EXPECT_EQ(batchMesh1.vertices.size(), singleMesh1.value().vertices.size());
  EXPECT_EQ(batchMesh1.indices.size(), singleMesh1.value().indices.size());
  ASSERT_EQ(batchMesh1.submeshes.size(), singleMesh1.value().submeshes.size());
  ASSERT_GE(batchMesh1.vertices.size(), 3u);
  ASSERT_GE(singleMesh1.value().vertices.size(), 3u);
  EXPECT_EQ(batchMesh1.submeshes[0].materialIndex,
            singleMesh1.value().submeshes[0].materialIndex);
  expectVec3Near(batchMesh1.vertices[0].position,
                 singleMesh1.value().vertices[0].position);
  expectVec3Near(batchMesh1.vertices[2].position,
                 singleMesh1.value().vertices[2].position);
}

TEST(GltfScenePrefabImport, PreservesSingleMeshPrimitiveMaterialBindings) {
  ScopedTempDir dir("nuri_gltf_single_mesh_primitives");

  const std::vector<uint8_t> buffer = twoMeshBuffer();
  const std::filesystem::path bufferPath = dir.path / "scene.bin";
  writeBinaryFile(bufferPath, buffer);

  const std::string json = R"json(
{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [{"name": "SingleMeshNode", "mesh": 0}],
  "materials": [
    {
      "name": "Mat0",
      "pbrMetallicRoughness": {
        "baseColorFactor": [1, 0, 0, 1],
        "metallicFactor": 0.0,
        "roughnessFactor": 1.0
      }
    },
    {
      "name": "Mat1",
      "pbrMetallicRoughness": {
        "baseColorFactor": [0, 1, 0, 1],
        "metallicFactor": 0.0,
        "roughnessFactor": 1.0
      }
    }
  ],
  "meshes": [
    {
      "primitives": [
        {"attributes": {"POSITION": 0}, "indices": 2, "material": 0},
        {"attributes": {"POSITION": 1}, "indices": 3, "material": 1}
      ]
    }
  ],
  "accessors": [
    {
      "bufferView": 0,
      "componentType": 5126,
      "count": 3,
      "type": "VEC3",
      "max": [1, 1, 0],
      "min": [0, 0, 0]
    },
    {
      "bufferView": 1,
      "componentType": 5126,
      "count": 3,
      "type": "VEC3",
      "max": [3, 1, 0],
      "min": [2, 0, 0]
    },
    {
      "bufferView": 2,
      "componentType": 5123,
      "count": 3,
      "type": "SCALAR",
      "max": [2],
      "min": [0]
    },
    {
      "bufferView": 3,
      "componentType": 5123,
      "count": 3,
      "type": "SCALAR",
      "max": [2],
      "min": [0]
    }
  ],
  "bufferViews": [
    {"buffer": 0, "byteOffset": 0, "byteLength": 36, "target": 34962},
    {"buffer": 0, "byteOffset": 36, "byteLength": 36, "target": 34962},
    {"buffer": 0, "byteOffset": 72, "byteLength": 6, "target": 34963},
    {"buffer": 0, "byteOffset": 78, "byteLength": 6, "target": 34963}
  ],
  "buffers": [{"byteLength": 84, "uri": "scene.bin"}]
}
)json";

  const std::filesystem::path gltfPath = dir.path / "single_mesh.gltf";
  writeTextFile(gltfPath, json);

  nuri::MeshImportOptions options{};
  options.optimize = false;
  options.generateLods = false;
  options.lodCount = 1u;

  auto sceneResult = nuri::SceneImporter::loadSceneFromFile(
      gltfPath.string(),
      nuri::SceneImportOptions{.assetBuildOptions = options});
  ASSERT_FALSE(sceneResult.hasError()) << sceneResult.error();

  const nuri::ScenePrefab &scene = sceneResult.value().prefab;
  ASSERT_EQ(scene.renderables.size(), 2u);
  ASSERT_EQ(scene.meshAssets.size(), 2u);
  ASSERT_GE(scene.materialAssets.size(), 2u);

  EXPECT_EQ(
      scene.materialAssets[scene.renderables[0].materialAssetIndex].sourceIndex,
      0u);
  EXPECT_EQ(
      scene.materialAssets[scene.renderables[1].materialAssetIndex].sourceIndex,
      1u);

  constexpr std::array<uint32_t, 2> kSceneMeshIndices = {0u, 1u};
  auto meshResult = nuri::MeshImporter::loadSceneMeshesFromFile(
      gltfPath.string(), kSceneMeshIndices, options);
  ASSERT_FALSE(meshResult.hasError()) << meshResult.error();
  ASSERT_EQ(meshResult.value().size(), 2u);
  ASSERT_EQ(meshResult.value()[0].submeshes.size(), 1u);
  ASSERT_EQ(meshResult.value()[1].submeshes.size(), 1u);
  EXPECT_EQ(meshResult.value()[0].submeshes[0].materialIndex, 0u);
  EXPECT_EQ(meshResult.value()[1].submeshes[0].materialIndex, 1u);
}

TEST(GltfScenePrefabImport, GeneratesProgressiveAttributeAwareLodChain) {
  ScopedTempDir dir("nuri_progressive_mesh_lods");
  const std::filesystem::path meshPath = dir.path / "wavy_grid.obj";
  writeTextFile(meshPath, makeWavyGridObj(24u));

  nuri::MeshImportOptions options{};
  options.calcTangents = false;
  options.optimize = false;
  options.generateLods = true;
  options.generateMeshlets = false;
  options.lodCount = 3u;
  options.lodTriangleRatios = {0.70f, 0.45f, 0.25f};
  options.lodTargetError = 1.0f;

  auto importResult =
      nuri::MeshImporter::loadFromFile(meshPath.string(), options);
  ASSERT_FALSE(importResult.hasError()) << importResult.error();

  const nuri::MeshData &mesh = importResult.value();
  ASSERT_EQ(mesh.submeshes.size(), 1u);
  const nuri::Submesh &submesh = mesh.submeshes.front();
  ASSERT_GE(submesh.lodCount, 3u);
  ASSERT_EQ(submesh.vertexOffset, 0u);

  const nuri::SubmeshLod &lod1 = submesh.lods[1];
  const nuri::SubmeshLod &lod2 = submesh.lods[2];
  ASSERT_GT(lod1.indexCount, lod2.indexCount);
  ASSERT_LE(static_cast<size_t>(lod2.indexOffset) + lod2.indexCount,
            mesh.indices.size());

  const std::span<const uint32_t> lod1Indices(
      mesh.indices.data() + lod1.indexOffset, lod1.indexCount);
  const std::span<const uint32_t> lod2Indices(
      mesh.indices.data() + lod2.indexOffset, lod2.indexCount);
  std::vector<uint32_t> expectedLod2(lod1Indices.size());
  std::vector<std::array<float, 5>> lodAttributes;
  lodAttributes.reserve(mesh.vertices.size());
  for (const nuri::Vertex &vertex : mesh.vertices) {
    lodAttributes.push_back({vertex.normal.x, vertex.normal.y, vertex.normal.z,
                             vertex.uv.x, vertex.uv.y});
  }
  constexpr std::array<float, 5> kAttributeWeights{0.5f, 0.5f, 0.5f, 1.0f,
                                                   1.0f};
  float stepError = 0.0f;
  const size_t expectedCount = meshopt_simplifyWithAttributes(
      expectedLod2.data(), lod1Indices.data(), lod1Indices.size(),
      &mesh.vertices.front().position.x, mesh.vertices.size(),
      sizeof(nuri::Vertex), lodAttributes.front().data(),
      sizeof(lodAttributes.front()), kAttributeWeights.data(),
      kAttributeWeights.size(), nullptr, lod2.indexCount,
      options.lodTargetError, 0, &stepError);
  const float simplificationScale =
      meshopt_simplifyScale(&mesh.vertices.front().position.x,
                            mesh.vertices.size(), sizeof(nuri::Vertex));

  ASSERT_EQ(expectedCount, lod2Indices.size());
  EXPECT_TRUE(std::equal(expectedLod2.begin(),
                         expectedLod2.begin() + expectedCount,
                         lod2Indices.begin(), lod2Indices.end()));
  EXPECT_NEAR(lod2.error, lod1.error + stepError * simplificationScale,
              1.0e-6f);
}

TEST(GltfScenePrefabImport, StopsQuietlyAtTheLastReducibleLod) {
  ScopedTempDir dir("nuri_terminal_mesh_lod");
  const std::filesystem::path meshPath = dir.path / "wavy_grid.obj";
  writeTextFile(meshPath, makeWavyGridObj(12u));

  std::vector<nuri::LogEntry> previousEntries;
  const nuri::LogReadResult previousLogs =
      nuri::readLogEntriesSince(0u, previousEntries);

  nuri::MeshImportOptions options{};
  options.calcTangents = false;
  options.optimize = false;
  options.generateLods = true;
  options.generateMeshlets = false;
  options.lodCount = 4u;
  options.lodTargetError = 0.0f;

  auto importResult =
      nuri::MeshImporter::loadFromFile(meshPath.string(), options);
  ASSERT_FALSE(importResult.hasError()) << importResult.error();

  std::vector<nuri::LogEntry> importEntries;
  nuri::readLogEntriesSince(previousLogs.lastSequence, importEntries);
  const bool emittedFalseFailure =
      std::ranges::any_of(importEntries, [](const nuri::LogEntry &entry) {
        return entry.level == nuri::LogLevel::Warning &&
               entry.message.find("simplification failed") != std::string::npos;
      });
  EXPECT_FALSE(emittedFalseFailure);
}
