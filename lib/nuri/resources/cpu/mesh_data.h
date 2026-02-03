#pragma once

#include "nuri/pch.h"

namespace nuri {

struct Vertex {
  glm::vec3 position{};
  glm::vec3 normal{};
  glm::vec2 uv{};
  glm::vec4 tangent{};
};

struct Submesh {
  uint32_t indexOffset = 0;
  uint32_t indexCount = 0;
  uint32_t materialIndex = 0;
};

struct MeshData {
  std::pmr::vector<Vertex> vertices;
  std::pmr::vector<uint32_t> indices;
  std::pmr::vector<Submesh> submeshes;
  std::pmr::string name;

  explicit MeshData(std::pmr::memory_resource *mem =
                        std::pmr::get_default_resource())
      : vertices(mem), indices(mem), submeshes(mem), name(mem) {}
};

} // namespace nuri
