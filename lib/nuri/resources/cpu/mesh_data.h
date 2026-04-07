#pragma once

#include "nuri/math/types.h"
#include "nuri/pch.h"
#include "nuri/resources/cpu/animation_data.h"

namespace nuri {

struct Vertex {
  glm::vec3 position{};
  glm::vec3 normal{};
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

  explicit MorphTarget(
      std::pmr::memory_resource *mem = std::pmr::get_default_resource())
      : name(mem), positionDeltas(mem), normalDeltas(mem) {}
};

struct SubmeshLod {
  uint32_t indexOffset = 0;
  uint32_t indexCount = 0;
  float error = 0.0f;
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
  std::pmr::vector<MorphTarget> morphTargets;
  std::pmr::string name;

  explicit MeshData(
      std::pmr::memory_resource *mem = std::pmr::get_default_resource())
      : vertices(mem), skinInfluences(mem), indices(mem), submeshes(mem),
        morphTargets(mem), name(mem) {}
};

} // namespace nuri
