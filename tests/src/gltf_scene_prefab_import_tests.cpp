#include "tests_pch.h"

#include "nuri/resources/gltf_scene_importer.h"
#include "nuri/resources/mesh_importer.h"
#include "nuri/scene/scene_prefab.h"

#include <chrono>
#include <thread>

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

  std::filesystem::path path;
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

  auto result =
      nuri::GltfSceneImporter::loadScenePrefabFromFile(gltfPath.string());
  ASSERT_FALSE(result.hasError()) << result.error();

  const nuri::ScenePrefab &prefab = result.value();
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
  EXPECT_EQ(prefab.renderables[0].meshIndex, 0u);
  EXPECT_LT(prefab.renderables[0].materialIndex, prefab.materialAssets.size());

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
  EXPECT_EQ(batchMesh0.submeshes[0].materialIndex,
            singleMesh0.value().submeshes[0].materialIndex);
  expectVec3Near(batchMesh0.vertices[0].position,
                 singleMesh0.value().vertices[0].position);
  expectVec3Near(batchMesh0.vertices[2].position,
                 singleMesh0.value().vertices[2].position);

  EXPECT_EQ(batchMesh1.vertices.size(), singleMesh1.value().vertices.size());
  EXPECT_EQ(batchMesh1.indices.size(), singleMesh1.value().indices.size());
  ASSERT_EQ(batchMesh1.submeshes.size(), singleMesh1.value().submeshes.size());
  EXPECT_EQ(batchMesh1.submeshes[0].materialIndex,
            singleMesh1.value().submeshes[0].materialIndex);
  expectVec3Near(batchMesh1.vertices[0].position,
                 singleMesh1.value().vertices[0].position);
  expectVec3Near(batchMesh1.vertices[2].position,
                 singleMesh1.value().vertices[2].position);
}
