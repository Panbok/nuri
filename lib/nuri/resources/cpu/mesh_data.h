#pragma once
#include "nuri/math/types.h"
#include "nuri/resources/cpu/animation_data.h"
#include <array>
#include <cstdint>
#include <memory_resource>
#include <string>
#include <vector>
namespace nuri {

struct Vertex {
  glm::vec3 position{};
  glm::vec3 normal{};
  glm::vec4 tangent{0.0f, 0.0f, 0.0f, 1.0f};
  glm::vec2 uv{};
  glm::vec2 uv1{};
};

struct VertexSkinInfluence {
  glm::u16vec4 joints{};
  glm::vec4 weights{};
};

struct MorphTarget {
  std::pmr::string name;
  std::pmr::vector<glm::vec3> positionDeltas;
  std::pmr::vector<glm::vec3> normalDeltas;
  std::pmr::vector<glm::vec3> tangentDeltas;
  explicit MorphTarget(
      std::pmr::memory_resource *mem = std::pmr::get_default_resource())
      : name(mem), positionDeltas(mem), normalDeltas(mem), tangentDeltas(mem) {}
};

struct SubmeshLod {
  uint32_t indexOffset = 0;
  uint32_t indexCount = 0;
  uint32_t meshletOffset = 0;
  uint32_t meshletCount = 0;
  float error = 0.0f;
};

struct MeshletDescriptor {
  uint32_t vertexOffset = 0;
  uint32_t vertexCount = 0;
  uint32_t primitiveOffset = 0;
  uint32_t primitiveCount = 0;
  glm::vec4 boundsSphere{0.0f};
  glm::vec4 coneApex{0.0f};
  glm::vec4 coneAxisCutoff{0.0f};
};

struct Submesh {
  static constexpr uint32_t kMaxLodCount = 4;
  uint32_t vertexOffset = 0;
  uint32_t vertexCount = 0;
  uint32_t indexOffset = 0;
  uint32_t indexCount = 0;
  uint32_t materialIndex = 0;
  uint32_t morphTargetFirst = 0;
  uint32_t morphTargetCount = 0;
  BoundingBox bounds{glm::vec3(0.0f), glm::vec3(0.0f)};
  glm::vec3 authoredScale{1.0f};
  uint32_t lodCount = 1;
  std::array<SubmeshLod, kMaxLodCount> lods{};
};

struct MeshData {
  std::pmr::vector<Vertex> vertices;
  std::pmr::vector<VertexSkinInfluence> skinInfluences;
  std::pmr::vector<uint32_t> indices;
  std::pmr::vector<Submesh> submeshes;
  std::pmr::vector<MeshletDescriptor> meshlets;
  std::pmr::vector<uint32_t> meshletVertexIndices;
  std::pmr::vector<uint8_t> meshletPrimitiveIndices;
  std::pmr::vector<MorphTarget> morphTargets;
  std::pmr::string name;
  explicit MeshData(
      std::pmr::memory_resource *mem = std::pmr::get_default_resource())
      : vertices(mem), skinInfluences(mem), indices(mem), submeshes(mem),
        meshlets(mem), meshletVertexIndices(mem), meshletPrimitiveIndices(mem),
        morphTargets(mem), name(mem) {}
};

} // namespace nuri
