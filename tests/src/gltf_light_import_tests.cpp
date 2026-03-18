#include "tests_pch.h"

#include "nuri/resources/gltf_scene_importer.h"

#include <chrono>
#include <thread>

#include <glm/gtc/matrix_transform.hpp>

namespace {
constexpr uint32_t kGlbMagic = 0x46546C67u;
constexpr uint32_t kGlbVersion2 = 2u;
constexpr uint32_t kGlbChunkTypeJson = 0x4E4F534Au;

struct ScopedTempDir {
  explicit ScopedTempDir(std::string_view prefix) {
    const auto uniqueId =
        static_cast<uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count()) ^
        static_cast<uint64_t>(
            std::hash<std::thread::id>{}(std::this_thread::get_id()));
    path = std::filesystem::temp_directory_path() /
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

void appendU32Le(std::vector<uint8_t> &out, uint32_t value) {
  out.push_back(static_cast<uint8_t>(value & 0xffu));
  out.push_back(static_cast<uint8_t>((value >> 8u) & 0xffu));
  out.push_back(static_cast<uint8_t>((value >> 16u) & 0xffu));
  out.push_back(static_cast<uint8_t>((value >> 24u) & 0xffu));
}

std::vector<uint8_t> padTo4(std::vector<uint8_t> bytes, uint8_t padByte) {
  while ((bytes.size() & 3u) != 0u) {
    bytes.push_back(padByte);
  }
  return bytes;
}

std::filesystem::path writeMinimalGlb(const ScopedTempDir &dir,
                                      std::string_view filename,
                                      std::string_view jsonText) {
  std::vector<uint8_t> jsonBytes(jsonText.begin(), jsonText.end());
  jsonBytes = padTo4(std::move(jsonBytes), 0x20u);

  std::vector<uint8_t> glb;
  glb.reserve(12u + 8u + jsonBytes.size());
  const uint32_t totalLength =
      static_cast<uint32_t>(12u + 8u + jsonBytes.size());
  appendU32Le(glb, kGlbMagic);
  appendU32Le(glb, kGlbVersion2);
  appendU32Le(glb, totalLength);
  appendU32Le(glb, static_cast<uint32_t>(jsonBytes.size()));
  appendU32Le(glb, kGlbChunkTypeJson);
  glb.insert(glb.end(), jsonBytes.begin(), jsonBytes.end());

  const std::filesystem::path glbPath = dir.path / std::string(filename);
  writeBinaryFile(glbPath, glb);
  return glbPath;
}

[[nodiscard]] glm::quat rotationFromMatrix(const glm::mat4 &matrix) {
  glm::vec3 basisX(matrix[0].x, matrix[0].y, matrix[0].z);
  glm::vec3 basisY(matrix[1].x, matrix[1].y, matrix[1].z);
  glm::vec3 basisZ(matrix[2].x, matrix[2].y, matrix[2].z);

  const float xLength = glm::length(basisX);
  const float yLength = glm::length(basisY);
  const float zLength = glm::length(basisZ);
  if (!std::isfinite(xLength) || !std::isfinite(yLength) ||
      !std::isfinite(zLength) || xLength <= 1.0e-6f || yLength <= 1.0e-6f ||
      zLength <= 1.0e-6f) {
    return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
  }

  basisX /= xLength;
  basisY /= yLength;
  basisZ /= zLength;

  glm::mat3 basis(1.0f);
  basis[0] = basisX;
  basis[1] = basisY;
  basis[2] = basisZ;
  if (glm::determinant(basis) < 0.0f) {
    basis[0] = -basis[0];
  }
  return glm::normalize(glm::quat_cast(basis));
}

[[nodiscard]] glm::vec3 lightDirection(const nuri::LightDesc &light) {
  return glm::normalize(light.rotation * glm::vec3(0.0f, 0.0f, -1.0f));
}

void expectVec3Near(const glm::vec3 &actual, const glm::vec3 &expected,
                    float epsilon = 1.0e-4f) {
  EXPECT_NEAR(actual.x, expected.x, epsilon);
  EXPECT_NEAR(actual.y, expected.y, epsilon);
  EXPECT_NEAR(actual.z, expected.z, epsilon);
}

} // namespace

TEST(GltfLightImport, LoadsDirectionalDefaultsFromGltf) {
  ScopedTempDir dir("nuri_gltf_lights_directional");
  const std::string json = R"json(
{
  "asset": {"version": "2.0"},
  "extensionsUsed": ["KHR_lights_punctual"],
  "extensions": {
    "KHR_lights_punctual": {
      "lights": [
        {"name": "Sun", "type": "directional"}
      ]
    }
  },
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [
    {
      "name": "Sun Node",
      "translation": [1.0, 2.0, 3.0],
      "extensions": {"KHR_lights_punctual": {"light": 0}}
    }
  ]
}
)json";
  const std::filesystem::path gltfPath = dir.path / "directional.gltf";
  writeTextFile(gltfPath, json);

  auto result = nuri::GltfSceneImporter::loadLightsFromFile(gltfPath.string());
  ASSERT_FALSE(result.hasError()) << result.error();
  ASSERT_EQ(result.value().size(), 1u);

  const nuri::ImportedLightInfo &light = result.value().front();
  EXPECT_EQ(light.desc.type, nuri::LightType::Directional);
  EXPECT_EQ(light.desc.name, "Sun Node");
  EXPECT_EQ(light.sourceName, "Sun Node");
  EXPECT_EQ(light.sourceNodeIndex, 0);
  expectVec3Near(light.desc.position, glm::vec3(1.0f, 2.0f, 3.0f));
  expectVec3Near(light.desc.color, glm::vec3(1.0f));
  EXPECT_FLOAT_EQ(light.desc.intensity, 1.0f);
  EXPECT_FLOAT_EQ(light.desc.range, 0.0f);
  EXPECT_TRUE(light.desc.enabled);
  expectVec3Near(lightDirection(light.desc), glm::vec3(0.0f, 0.0f, -1.0f));
}

TEST(GltfLightImport, LoadsPointLightValuesFromGltf) {
  ScopedTempDir dir("nuri_gltf_lights_point");
  const std::string json = R"json(
{
  "asset": {"version": "2.0"},
  "extensionsUsed": ["KHR_lights_punctual"],
  "extensions": {
    "KHR_lights_punctual": {
      "lights": [
        {
          "name": "Bulb",
          "type": "point",
          "color": [0.2, 0.4, 0.6],
          "intensity": 12.5,
          "range": 7.0
        }
      ]
    }
  },
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [
    {
      "translation": [4.0, 5.0, 6.0],
      "extensions": {"KHR_lights_punctual": {"light": 0}}
    }
  ]
}
)json";
  const std::filesystem::path gltfPath = dir.path / "point.gltf";
  writeTextFile(gltfPath, json);

  auto result = nuri::GltfSceneImporter::loadLightsFromFile(gltfPath.string());
  ASSERT_FALSE(result.hasError()) << result.error();
  ASSERT_EQ(result.value().size(), 1u);

  const nuri::ImportedLightInfo &light = result.value().front();
  EXPECT_EQ(light.desc.type, nuri::LightType::Point);
  EXPECT_EQ(light.desc.name, "Bulb");
  expectVec3Near(light.desc.position, glm::vec3(4.0f, 5.0f, 6.0f));
  expectVec3Near(light.desc.color, glm::vec3(0.2f, 0.4f, 0.6f));
  EXPECT_FLOAT_EQ(light.desc.intensity, 12.5f);
  EXPECT_FLOAT_EQ(light.desc.range, 7.0f);
  EXPECT_FLOAT_EQ(light.desc.innerConeAngleRadians, 0.0f);
  EXPECT_FLOAT_EQ(light.desc.outerConeAngleRadians, 0.0f);
}

TEST(GltfLightImport, AppliesHierarchyAndSanitizesSpotValues) {
  ScopedTempDir dir("nuri_gltf_lights_spot");
  const std::string json = R"json(
{
  "asset": {"version": "2.0"},
  "extensionsUsed": ["KHR_lights_punctual"],
  "extensions": {
    "KHR_lights_punctual": {
      "lights": [
        {
          "name": "Lamp",
          "type": "spot",
          "color": [0.9, 0.7, 0.5],
          "intensity": 3.0,
          "range": -2.0,
          "spot": {
            "innerConeAngle": 2.2,
            "outerConeAngle": 2.0
          }
        }
      ]
    }
  },
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [
    {
      "translation": [1.0, 0.0, 0.0],
      "rotation": [0.0, 0.70710678, 0.0, 0.70710678],
      "children": [1]
    },
    {
      "name": "Lamp Node",
      "translation": [0.0, 0.0, -2.0],
      "rotation": [0.38268343, 0.0, 0.0, 0.92387953],
      "extensions": {"KHR_lights_punctual": {"light": 0}}
    }
  ]
}
)json";
  const std::filesystem::path gltfPath = dir.path / "spot.gltf";
  writeTextFile(gltfPath, json);

  auto result = nuri::GltfSceneImporter::loadLightsFromFile(gltfPath.string());
  ASSERT_FALSE(result.hasError()) << result.error();
  ASSERT_EQ(result.value().size(), 1u);

  const nuri::ImportedLightInfo &light = result.value().front();
  EXPECT_EQ(light.desc.type, nuri::LightType::Spot);
  EXPECT_EQ(light.desc.name, "Lamp Node");
  expectVec3Near(light.desc.color, glm::vec3(0.9f, 0.7f, 0.5f));
  EXPECT_FLOAT_EQ(light.desc.intensity, 3.0f);
  EXPECT_FLOAT_EQ(light.desc.range, 0.0f);
  EXPECT_NEAR(light.desc.outerConeAngleRadians, glm::half_pi<float>() - 1.0e-4f,
              1.0e-5f);
  EXPECT_NEAR(light.desc.innerConeAngleRadians, glm::half_pi<float>() - 1.0e-4f,
              1.0e-5f);

  const glm::mat4 parentMatrix =
      glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 0.0f, 0.0f)) *
      glm::mat4_cast(glm::quat(0.70710678f, 0.0f, 0.70710678f, 0.0f));
  const glm::mat4 childMatrix =
      glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -2.0f)) *
      glm::mat4_cast(glm::quat(0.92387953f, 0.38268343f, 0.0f, 0.0f));
  const glm::mat4 expectedWorld = parentMatrix * childMatrix;
  expectVec3Near(light.desc.position, glm::vec3(expectedWorld[3]));
  expectVec3Near(lightDirection(light.desc),
                 glm::normalize(rotationFromMatrix(expectedWorld) *
                                glm::vec3(0.0f, 0.0f, -1.0f)));
}

TEST(GltfLightImport, LoadsLightsFromGlb) {
  ScopedTempDir dir("nuri_gltf_lights_glb");
  const std::string json = R"json(
{
  "asset": {"version": "2.0"},
  "extensionsUsed": ["KHR_lights_punctual"],
  "extensions": {
    "KHR_lights_punctual": {
      "lights": [
        {"name": "GlbSun", "type": "directional", "intensity": 4.0}
      ]
    }
  },
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [
    {
      "extensions": {"KHR_lights_punctual": {"light": 0}}
    }
  ]
}
)json";
  const std::filesystem::path glbPath =
      writeMinimalGlb(dir, "lights.glb", json);

  auto result = nuri::GltfSceneImporter::loadLightsFromFile(glbPath.string());
  ASSERT_FALSE(result.hasError()) << result.error();
  ASSERT_EQ(result.value().size(), 1u);
  EXPECT_EQ(result.value().front().desc.type, nuri::LightType::Directional);
  EXPECT_FLOAT_EQ(result.value().front().desc.intensity, 4.0f);
  EXPECT_EQ(result.value().front().desc.name, "GlbSun");
}
